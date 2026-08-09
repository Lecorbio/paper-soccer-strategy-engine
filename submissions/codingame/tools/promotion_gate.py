#!/usr/bin/env python3

"""Run and adjudicate the frozen CodinGame promotion ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMOTION = ROOT / "submissions/codingame/promotion"
DEFAULT_MANIFEST = PROMOTION / "manifest.json"
DEFAULT_BUILD = ROOT / "build/native"
DEFAULT_RESULTS = ROOT / "results/codingame/promotion"
HOLDOUT_LEDGER = ROOT / ".git/papersoccer-promotion"
REQUIRED_STAGES = ("initial", "development", "validation", "test")
STAGE_PREDECESSORS = {
    "initial": (),
    "development": ("initial",),
    "validation": ("initial", "development"),
    "test": ("initial", "development", "validation"),
}
OPERATIONAL_FIELDS = (
    "illegal_actions",
    "empty_actions",
    "incomplete_actions",
    "overlong_actions",
    "unfinished_games",
)
MAX_NODE_BUDGET = (1 << 64) - 1
MAX_TIME_BUDGET_MS = (1 << 32) - 1
PROFILE_BOOLEAN_THRESHOLDS = {
    "require_more_wins_than_incumbent",
    "require_at_least_as_many_wins_as_incumbent",
}
PROFILE_NUMERIC_THRESHOLDS = {
    "minimum_mean",
    "minimum_ci_lower",
    "minimum_color_score",
    "minimum_control_adjusted_uplift",
    "minimum_physical_color_uplift",
    "minimum_historical_role_score",
    "minimum_control_winner_retention",
    "minimum_stratum_score",
    "minimum_winner_tier_score",
    "minimum_elite_tier_score",
    "minimum_throughput_ratio",
}
BANK_FIELDS = (
    "opening_id", "split", "stratum", "source_agent_id", "source_game_id",
    "opponent_agent_id", "winner_player_id", "turn_index", "physical_edges",
    "state_key", "canonical_key", "ball_x", "ball_y", "mover",
    "winner_tier", "goal_distance_band", "used_edge_band", "shell_edge_band",
    "opening_family", "observed_winner_action", "transcript",
)
DIRECTIONS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


class GateError(RuntimeError):
    exit_code = 70


class UsageError(GateError):
    exit_code = 64


class IncompleteError(GateError):
    exit_code = 20


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized_segment(left, right):
    return tuple(sorted((left, right)))


def is_regular_point(point):
    return 0 <= point[0] <= 8 and 1 <= point[1] <= 11


def is_goal_point(point):
    return 3 <= point[0] <= 5 and point[1] in (0, 12)


def is_goal_mouth_point(point):
    return 3 <= point[0] <= 5 and point[1] in (1, 11)


def is_boundary_point(point):
    if not is_regular_point(point):
        return False
    if point[0] in (0, 8):
        return True
    return point[1] in (1, 11) and not 3 < point[0] < 5


def is_forbidden_boundary_segment(edge):
    left, right = edge
    if is_goal_point(left) or is_goal_point(right):
        vertical_post = left[0] == right[0] and left[0] in (3, 5)
        north_post = {left[1], right[1]} == {0, 1}
        south_post = {left[1], right[1]} == {11, 12}
        if vertical_post and (north_post or south_post):
            return True
    if not is_regular_point(left) or not is_regular_point(right):
        return False
    if not (is_boundary_point(left) and is_boundary_point(right)):
        return False
    dx = abs(left[0] - right[0])
    dy = abs(left[1] - right[1])
    if left[1] == right[1] and left[1] in (1, 11) and dx == 1:
        return True
    return left[0] == right[0] and left[0] in (0, 8) and dy == 1


def legal_destinations(ball, edges):
    destinations = []
    if not is_regular_point(ball):
        return destinations
    for dx, dy in DIRECTIONS:
        destination = ball[0] + dx, ball[1] + dy
        if not is_regular_point(destination):
            continue
        edge = normalized_segment(ball, destination)
        if edge not in edges and not is_forbidden_boundary_segment(edge):
            destinations.append(destination)
    if is_goal_mouth_point(ball):
        goal_y = 0 if ball[1] == 1 else 12
        for goal_x in range(3, 6):
            destination = goal_x, goal_y
            if max(abs(ball[0] - goal_x), abs(ball[1] - goal_y)) != 1:
                continue
            edge = normalized_segment(ball, destination)
            if edge not in edges and not is_forbidden_boundary_segment(edge):
                destinations.append(destination)
    return destinations


def state_text(ball, mover, edges):
    edge_text = ";".join(
        f"{left[0]},{left[1]}-{right[0]},{right[1]}"
        for left, right in sorted(edges)
    )
    return f"ball={ball[0]},{ball[1]}|mover={mover}|edges={edge_text}"


def independently_reconstruct_bank_state(row):
    ball = (4, 6)
    edges = set()
    visits = {ball: 1}
    mover = 0
    raw_transcript = row["transcript"]
    if raw_transcript == "-" and row.get("opening_id") != "initial":
        raise UsageError("only the initial sentinel may use an empty transcript marker")
    turns = [] if raw_transcript in ("", "-") else raw_transcript.split("/")
    for action in turns:
        if not action:
            raise UsageError("bank transcript contains an empty complete turn")
        action_mover = mover
        for index, character in enumerate(action):
            if character < "0" or character > "7":
                raise UsageError("bank transcript contains an invalid direction")
            dx, dy = DIRECTIONS[ord(character) - ord("0")]
            destination = ball[0] + dx, ball[1] + dy
            if destination not in legal_destinations(ball, edges):
                raise UsageError("bank transcript contains an illegal move")
            edge = normalized_segment(ball, destination)
            extra_turn = is_boundary_point(destination) or visits.get(destination, 0) > 0
            edges.add(edge)
            ball = destination
            visits[ball] = visits.get(ball, 0) + 1
            terminal = is_goal_point(ball) or not legal_destinations(ball, edges)
            if terminal:
                raise UsageError("bank transcript reaches a terminal position")
            mover = action_mover if extra_turn else 1 - action_mover
            final_character = index + 1 == len(action)
            if final_character == (mover == action_mover):
                detail = "ends during a rebound" if final_character else "continues after handoff"
                raise UsageError(f"bank complete turn {detail}")
    winner = int(row["winner_player_id"])
    normalized_ball = ball
    normalized_mover = mover
    normalized_edges = edges
    if winner == 1:
        rotate = lambda point: (8 - point[0], 12 - point[1])
        normalized_ball = rotate(ball)
        normalized_mover = 1 - mover
        normalized_edges = {
            normalized_segment(rotate(left), rotate(right))
            for left, right in edges
        }
    raw = state_text(normalized_ball, normalized_mover, normalized_edges)
    reflect = lambda point: (8 - point[0], point[1])
    reflected = state_text(
        reflect(normalized_ball),
        normalized_mover,
        {
            normalized_segment(reflect(left), reflect(right))
            for left, right in normalized_edges
        },
    )
    return {
        "ball": ball,
        "mover": mover,
        "edges": len(edges),
        "turn_index": len(turns),
        "state_key": sha256_text(raw),
        "canonical_key": sha256_text(min(raw, reflected)),
    }


def stable_json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temporary = pathlib.Path(output.name)
        output.write(stable_json(value))
    os.replace(temporary, path)


def atomic_json_create_exclusive(path: pathlib.Path, value) -> None:
    """Publish complete JSON only if no process has already claimed the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temporary = pathlib.Path(output.name)
        output.write(stable_json(value))
        output.flush()
        os.fsync(output.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_verified_bank(source: pathlib.Path, destination: pathlib.Path,
                           expected_sha256: str) -> pathlib.Path:
    """Freeze a validated bank before workers can observe later path changes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise UsageError("existing bank snapshot has an unexpected hash")
        return destination
    with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, delete=False
    ) as output_file:
        temporary = pathlib.Path(output_file.name)
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            output_file.write(block)
    try:
        if sha256(temporary) != expected_sha256:
            raise UsageError("bank changed while its snapshot was being created")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_manifest(path: pathlib.Path):
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UsageError(f"could not read manifest {path}: {error}") from error
    if manifest.get("schema") != "papersoccer.codingame-promotion-manifest.v1":
        raise UsageError("unsupported promotion manifest")
    return manifest


def resolve_repository_path(relative: str) -> pathlib.Path:
    candidate = (ROOT / relative).resolve()
    if ROOT.resolve() not in candidate.parents:
        raise UsageError(f"path escapes repository: {relative}")
    return candidate


def read_bank(path: pathlib.Path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != BANK_FIELDS:
            raise UsageError(f"promotion bank has an unexpected header: {path}")
        rows = list(reader)
    if not rows:
        raise UsageError(f"invalid or empty promotion bank: {path}")
    return rows


def verify_frozen_builder(manifest_path: pathlib.Path):
    if manifest_path.resolve() != DEFAULT_MANIFEST.resolve():
        raise UsageError(
            "custom promotion manifests are not accepted by the frozen gate; "
            "update the deterministic bank builder first"
        )
    builder = PROMOTION / "build_goal_shell_banks.py"
    completed = subprocess.run(
        [sys.executable, str(builder), "--check"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise UsageError(f"promotion artifacts differ from the frozen builder: {detail}")


def validate(manifest_path: pathlib.Path, bot: str | None = None):
    verify_frozen_builder(manifest_path)
    manifest = load_manifest(manifest_path)
    problems = []
    if bot is not None and manifest.get("candidate") != bot:
        problems.append(
            f"manifest candidate {manifest.get('candidate')!r} does not match {bot!r}"
        )
    for relative, expected in manifest["sources"].items():
        path = resolve_repository_path(relative)
        if not path.exists():
            problems.append(f"missing source {relative}")
        elif sha256(path) != expected:
            problems.append(f"source hash mismatch {relative}")

    all_keys = {}
    all_games = {}
    bank_summaries = {}
    for relative, specification in manifest["banks"].items():
        path = PROMOTION / relative
        if not path.exists():
            problems.append(f"missing bank {relative}")
            continue
        actual_hash = sha256(path)
        if actual_hash != specification["sha256"]:
            problems.append(f"bank hash mismatch {relative}")
        rows = read_bank(path)
        if len(rows) != specification["records"]:
            problems.append(f"bank record count mismatch {relative}")
        ids = set()
        keys = set()
        games = set()
        split = rows[0]["split"]
        for row in rows:
            if row["opening_id"] in ids:
                problems.append(f"duplicate opening id in {relative}")
            ids.add(row["opening_id"])
            if row["canonical_key"] in keys:
                problems.append(f"duplicate canonical state in {relative}")
            keys.add(row["canonical_key"])
            if row["split"] != split:
                problems.append(f"mixed split labels in {relative}")
            try:
                reconstructed = independently_reconstruct_bank_state(row)
                declared = {
                    "ball": (int(row["ball_x"]), int(row["ball_y"])),
                    "mover": int(row["mover"]),
                    "edges": int(row["physical_edges"]),
                    "turn_index": int(row["turn_index"]),
                    "state_key": row["state_key"],
                    "canonical_key": row["canonical_key"],
                }
                if reconstructed != declared:
                    problems.append(
                        f"reconstructed state metadata mismatch in {relative}: "
                        f"{row['opening_id']}"
                    )
            except (KeyError, TypeError, ValueError, UsageError) as error:
                problems.append(
                    f"could not reconstruct {relative}:{row.get('opening_id')}: {error}"
                )
            game = int(row["source_game_id"])
            if game:
                games.add(game)
        if split != "initial":
            for key in keys:
                previous = all_keys.get(key)
                if previous is not None:
                    problems.append(
                        f"canonical state shared by {previous} and {relative}"
                    )
                all_keys[key] = relative
            for game in games:
                previous = all_games.get(game)
                if previous is not None and previous != relative:
                    problems.append(f"source game shared by {previous} and {relative}")
                all_games[game] = relative
        bank_summaries[relative] = {
            "sha256": actual_hash,
            "records": len(rows),
            "split": split,
            "strata": sorted({row["stratum"] for row in rows}),
        }

    referenced_banks = set()
    for stage in REQUIRED_STAGES:
        config = manifest["stages"].get(stage)
        if config is None:
            problems.append(f"missing stage {stage}")
            continue
        if config["bank"] not in manifest["banks"]:
            problems.append(f"stage {stage} references an unknown bank")
            continue
        if config["bank"] in referenced_banks:
            problems.append(f"multiple stages reference bank {config['bank']}")
        referenced_banks.add(config["bank"])
        try:
            configured_strength_profiles(config)
            configured_required_jobs(config)
        except UsageError as error:
            problems.append(f"stage {stage}: {error}")
        summary = bank_summaries.get(config["bank"])
        if summary is not None and summary["split"] != stage:
            problems.append(
                f"stage {stage} references split {summary['split']}"
            )

    expected_incumbent = manifest["incumbent"]["submission_sha256"]
    incumbent = ROOT / "submissions/codingame/bots/rank_5/submission.cpp"
    if not incumbent.exists() or sha256(incumbent) != expected_incumbent:
        problems.append("immutable rank_5 submission hash mismatch")
    if problems:
        raise UsageError("; ".join(problems))
    return {
        "schema": "papersoccer.codingame-promotion-validation.v1",
        "valid": True,
        "manifest_sha256": sha256(manifest_path),
        "banks": bank_summaries,
        "incumbent_submission_sha256": expected_incumbent,
    }


def candidate_paths(bot: str, build: pathlib.Path):
    directory = ROOT / "submissions/codingame/bots" / bot
    if not directory.is_dir():
        raise UsageError(f"unknown candidate bot: {bot}")
    prefix = "papersoccer_codingame" if bot == "alpha_beta" else f"papersoccer_codingame_{bot}"
    return {
        "directory": directory,
        "submission": directory / "submission.cpp",
        "runner_source": directory / "comparison_gate.cpp",
        "runner": build / f"{prefix}_comparison_gate",
        "test": build / f"{prefix}_submission_test",
        "timing": build / f"{prefix}_timing_probe",
        "runner_target": f"{prefix}_comparison_gate",
        "test_target": f"{prefix}_submission_test",
        "timing_target": f"{prefix}_timing_probe",
    }


def run_checked(command, *, cwd=ROOT, capture=True):
    completed = subprocess.run(
        [str(part) for part in command], cwd=cwd, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise GateError(f"command failed ({completed.returncode}): {' '.join(map(str, command))}: {detail}")
    return completed


def build_targets(build: pathlib.Path, targets: list[str]):
    run_checked(["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release"])
    run_checked(["cmake", "--build", build, "--parallel", "--target", *targets])


def check_generated_submission(bot: str):
    run_checked([
        "node", ROOT / "submissions/codingame/tools/generate_submission.mjs",
        bot, "--check",
    ])


def result_directory(bot: str, candidate_hash: str, manifest_hash: str,
                     root: pathlib.Path):
    return root / bot / f"{candidate_hash[:16]}-{manifest_hash[:12]}"


def expected_stage_identity(manifest: dict, manifest_hash: str, stage: str,
                            candidate_hash: str, runner_hash: str):
    bank_relative = manifest["stages"][stage]["bank"]
    return {
        "manifest_sha256": manifest_hash,
        "bank_sha256": manifest["banks"][bank_relative]["sha256"],
        "candidate_submission_sha256": candidate_hash,
        "incumbent_submission_sha256": manifest["incumbent"]["submission_sha256"],
        "runner_sha256": runner_hash,
    }


def locked_test_consumption_path(bank_hash: str):
    return HOLDOUT_LEDGER / f"locked-test-consumption-{bank_hash[:16]}.json"


def locked_test_consumption_marker(manifest: dict, manifest_hash: str,
                                   candidate_hash: str, shard_count: int):
    config = manifest["stages"]["test"]
    bank_hash = manifest["banks"][config["bank"]]["sha256"]
    marker = {
        "candidate_submission_sha256": candidate_hash,
        "manifest_sha256": manifest_hash,
        "bank_sha256": bank_hash,
        "shard_count": shard_count,
    }
    profiles = configured_strength_profiles(config)
    if uses_explicit_strength_profiles(config):
        marker.update({
            "schema": "papersoccer.codingame-locked-test-consumption.v2",
            "strength_profiles": [
                strength_profile_identity(profile) for profile in profiles
            ],
        })
    else:
        marker.update({
            "schema": "papersoccer.codingame-locked-test-consumption.v1",
            "node_budgets": [profile["value"] for profile in profiles],
        })
    return marker


def stage_profile_shard_count(directory: pathlib.Path, manifest: dict,
                              stage: str) -> int:
    counts = []
    for profile in configured_strength_profiles(manifest["stages"][stage]):
        shard_directory = (
            directory / "shards" / stage / profile["directory"]
        )
        count = len(list(shard_directory.glob("shard-*-of-*.json")))
        if count <= 0:
            raise IncompleteError(
                f"stage {stage} has no {profile['id']} shards"
            )
        counts.append(count)
    if len(set(counts)) != 1:
        raise IncompleteError(
            f"stage {stage} profiles use different shard counts"
        )
    return counts[0]


def verify_locked_test_consumption(directory: pathlib.Path, manifest: dict,
                                   manifest_hash: str, candidate_hash: str):
    config = manifest["stages"]["test"]
    bank_hash = manifest["banks"][config["bank"]]["sha256"]
    path = locked_test_consumption_path(bank_hash)
    if not path.exists():
        raise IncompleteError("locked-test consumption record is missing")
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise IncompleteError(f"invalid locked-test consumption record: {error}") from error
    shard_count = marker.get("shard_count")
    if (isinstance(shard_count, bool) or not isinstance(shard_count, int) or
            shard_count <= 0 or marker != locked_test_consumption_marker(
                manifest, manifest_hash, candidate_hash, shard_count
            )):
        raise IncompleteError("locked-test consumption identity mismatch")
    if shard_count != stage_profile_shard_count(directory, manifest, "test"):
        raise IncompleteError(
            "locked-test consumption shard count does not match raw evidence"
        )
    return path


def write_incomplete_decision(directory: pathlib.Path, bot: str,
                              candidate_hash: str, manifest_hash: str,
                              manifest: dict, operation: str,
                              runner_hash: str | None = None):
    decision = {
        "schema": "papersoccer.codingame-promotion-decision.v1",
        "bot": bot,
        "candidate_submission_sha256": candidate_hash,
        "manifest_sha256": manifest_hash,
        "incumbent_submission_sha256": manifest["incumbent"]["submission_sha256"],
        "runner_sha256": runner_hash,
        "verdict": "INCOMPLETE",
        "submission_worthy": False,
        "current_operation": operation,
        "reason_codes": ["evaluation_in_progress"],
    }
    atomic_json(directory / "decision.json", decision)
    return decision


def write_nonstage_rejection(directory: pathlib.Path, bot: str,
                             failed_stage: str, report: dict, manifest: dict,
                             runner_hash: str, artifacts: list[pathlib.Path]):
    if failed_stage == "preflight":
        stage_status = {stage: "not_run_due_to_rejection" for stage in REQUIRED_STAGES}
    elif failed_stage == "timing":
        stage_status = {stage: "pass" for stage in REQUIRED_STAGES}
    else:
        raise ValueError(f"unsupported non-strength stage {failed_stage}")
    decision = {
        "schema": "papersoccer.codingame-promotion-decision.v1",
        "bot": bot,
        "candidate_submission_sha256": report["candidate_submission_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "incumbent_submission_sha256": manifest["incumbent"]["submission_sha256"],
        "runner_sha256": runner_hash,
        "verdict": "REJECT",
        "submission_worthy": False,
        "failed_stage": failed_stage,
        "reason_codes": report["reason_codes"],
        "stage_status": stage_status,
        "artifacts": [str(path.relative_to(ROOT)) for path in artifacts],
    }
    atomic_json(directory / "decision.json", decision)
    return decision


def candidate_hypothesis_requirement(manifest: dict, candidate_hash: str):
    expected_hash = manifest.get("candidate_submission_sha256", candidate_hash)
    return {
        "id": "candidate_matches_frozen_hypothesis",
        "passed": candidate_hash == expected_hash,
        "observed": candidate_hash,
        "operator": "==",
        "threshold": expected_hash,
    }


def preflight(bot: str, manifest_path: pathlib.Path, build: pathlib.Path,
              results_root: pathlib.Path):
    validation = validate(manifest_path, bot)
    manifest = load_manifest(manifest_path)
    paths = candidate_paths(bot, build)
    check_generated_submission(bot)
    check_generated_submission("rank_5")
    submission = paths["submission"]
    data = submission.read_bytes()
    candidate_hash = sha256(submission)
    directory = result_directory(
        bot, candidate_hash, validation["manifest_sha256"], results_root
    )
    write_incomplete_decision(
        directory, bot, candidate_hash, validation["manifest_sha256"],
        manifest, "preflight"
    )
    build_targets(build, [
        paths["test_target"], paths["runner_target"], paths["timing_target"]
    ])
    artifact_test_error = None
    try:
        run_checked([paths["test"]])
    except GateError as error:
        artifact_test_error = str(error)
    source_limit = manifest["source_limit"]
    requirements = [
        {"id": "generated_submission_exists", "passed": submission.exists()},
        {"id": "generated_submission_ascii", "passed": all(byte < 128 for byte in data)},
        candidate_hypothesis_requirement(manifest, candidate_hash),
        {"id": "generated_submission_size", "passed": len(data) <= source_limit,
         "observed": len(data), "operator": "<=", "threshold": source_limit},
        {"id": "artifact_tests", "passed": artifact_test_error is None,
         "detail": artifact_test_error},
        {"id": "banks_and_incumbent_frozen", "passed": validation["valid"]},
    ]
    passed = all(item["passed"] for item in requirements)
    report = {
        "schema": "papersoccer.codingame-promotion-preflight.v1",
        "bot": bot,
        "candidate_submission_sha256": candidate_hash,
        "candidate_submission_characters": len(data),
        "manifest_sha256": validation["manifest_sha256"],
        "artifact_test_binary_sha256": sha256(paths["test"]),
        "runner_binary_sha256": sha256(paths["runner"]),
        "timing_binary_sha256": sha256(paths["timing"]),
        "requirements": requirements,
        "reason_codes": [item["id"] for item in requirements if not item["passed"]],
        "passed": passed,
    }
    report_path = directory / "preflight.json"
    atomic_json(report_path, report)
    if not passed:
        write_nonstage_rejection(
            directory, bot, "preflight", report, manifest,
            sha256(paths["runner"]), [report_path]
        )
    print(stable_json(report), end="")
    return report


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires observations")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def source_game_cluster_bootstrap(pairs: list[dict], samples: int, seed: int):
    clusters = {}
    for pair in pairs:
        score = pair["candidate_pair_score"]
        if score is None:
            raise GateError("cannot bootstrap an unfinished pair")
        source_game = int(pair.get("source_game_id", 0))
        cluster = f"game-{source_game}" if source_game else pair["opening_id"]
        clusters.setdefault(cluster, []).append(float(score))
    cluster_scores = [
        sum(values) / len(values) for _, values in sorted(clusters.items())
    ]
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(generator.choice(cluster_scores) for _ in cluster_scores)
            / len(cluster_scores)
        )
    estimates.sort()
    estimate = sum(cluster_scores) / len(cluster_scores)
    return {
        "method": "source_game_cluster_percentile_bootstrap",
        "confidence": 0.95,
        "resamples": samples,
        "seed": seed,
        "opening_pairs": len(pairs),
        "source_game_clusters": len(cluster_scores),
        "estimate": estimate,
        "lower": quantile(estimates, 0.025),
        "upper": quantile(estimates, 0.975),
    }


def requirement(identifier: str, observed, operator: str, threshold):
    if operator == ">=":
        passed = observed >= threshold
    elif operator == ">":
        passed = observed > threshold
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unsupported operator {operator}")
    return {
        "id": identifier, "passed": passed, "observed": observed,
        "operator": operator, "threshold": threshold,
    }


def aggregate_stage(manifest: dict, stage: str, shards: list[dict], identity: dict,
                    node_budget: int | None = None,
                    strength_profile: dict | None = None):
    if node_budget is not None and strength_profile is not None:
        raise ValueError("stage aggregation received two profile selectors")
    pairs = []
    operational = {field: 0 for field in OPERATIONAL_FIELDS}
    for shard in shards:
        if shard.get("schema") != "papersoccer.codingame-promotion-shard.v2":
            raise GateError("runner produced an unsupported shard")
        if shard["identity"] != identity:
            raise GateError("shard identity mismatch")
        pairs.extend(shard["pairs"])
        for field in operational:
            operational[field] += int(shard["operational"].get(field, 0))
    expected = manifest["banks"][manifest["stages"][stage]["bank"]]["records"]
    ids = [pair["opening_id"] for pair in pairs]
    if len(pairs) != expected or len(set(ids)) != expected:
        raise GateError(f"stage {stage} is incomplete or contains duplicate pairs")
    pairs.sort(key=lambda pair: pair["opening_id"])

    scores = []
    candidate_wins = 0
    incumbent_wins = 0
    color_scores = {"candidate_player_0": [], "candidate_player_1": []}
    control_role_scores = {"candidate_player_0": [], "candidate_player_1": []}
    historical_candidate_scores = {"winner": [], "opponent": []}
    historical_control_scores = {"winner": [], "opponent": []}
    retained = converted = 0
    stratum_scores = {}
    winner_tier_scores = {}
    won_2_0 = split_1_1 = lost_0_2 = 0
    diagnostics = {
        "candidate_nodes": 0,
        "incumbent_nodes": 0,
        "candidate_searches": 0,
        "incumbent_searches": 0,
        "candidate_depth_sum": 0,
        "incumbent_depth_sum": 0,
        "candidate_ms": 0.0,
        "incumbent_ms": 0.0,
        "rebound_goal_probes": 0,
        "rebound_goal_hits": 0,
        "rebound_loss_hits": 0,
        "exchange_ply1_probes": 0,
        "exchange_ply1_win_hits": 0,
        "exchange_ply1_loss_hits": 0,
        "exchange_ply1_cutoffs": 0,
        "exchange_ply2_probes": 0,
        "exchange_ply2_win_hits": 0,
        "exchange_ply2_loss_hits": 0,
        "exchange_ply2_cutoffs": 0,
    }
    for pair in pairs:
        games = pair.get("games")
        if not isinstance(games, list) or len(games) != 2:
            raise GateError("runner pair does not contain exactly two candidate games")
        games_by_role = {}
        for game in games:
            candidate_player = int(game.get("candidate_player", -1))
            winner = game.get("winner")
            if candidate_player not in (0, 1) or winner not in (0, 1):
                raise GateError("runner candidate game has an invalid role or winner")
            if candidate_player in games_by_role:
                raise GateError("runner repeated a candidate role")
            games_by_role[candidate_player] = game
        if set(games_by_role) != {0, 1}:
            raise GateError("runner candidate roles are incomplete")

        control_winner = pair.get("incumbent_control", {}).get("winner")
        if control_winner not in (0, 1):
            raise GateError("runner did not finish the incumbent control game")
        control_turns = pair.get("incumbent_control", {}).get("turns")
        if not isinstance(control_turns, int) or control_turns <= 0:
            raise GateError("runner control game has an invalid turn count")

        candidate_outcomes = {
            role: float(games_by_role[role]["winner"] == role)
            for role in (0, 1)
        }
        recomputed_score = sum(candidate_outcomes.values()) / 2.0
        raw_score = pair.get("candidate_pair_score")
        if raw_score is None or not math.isclose(
            float(raw_score), recomputed_score, rel_tol=0.0, abs_tol=1e-12
        ):
            raise GateError("runner candidate pair score is inconsistent")
        score = recomputed_score
        scores.append(score)

        historical_winner = int(pair.get("historical_winner_player", -1))
        if historical_winner not in (0, 1):
            raise GateError("runner pair has an invalid historical winner")
        for role in (0, 1):
            control_role_scores[f"candidate_player_{role}"].append(
                float(control_winner == role)
            )
        for label, role in (
            ("winner", historical_winner),
            ("opponent", 1 - historical_winner),
        ):
            historical_candidate_scores[label].append(candidate_outcomes[role])
            historical_control_scores[label].append(float(control_winner == role))
        retained += int(candidate_outcomes[control_winner])
        converted += int(candidate_outcomes[1 - control_winner])
        if score == 1.0:
            won_2_0 += 1
        elif score == 0.5:
            split_1_1 += 1
        elif score == 0.0:
            lost_0_2 += 1
        stratum_scores.setdefault(pair["stratum"], []).append(score)
        winner_tier_scores.setdefault(pair["winner_tier"], []).append(score)
        for game in games:
            for key in diagnostics:
                if key not in ("candidate_ms", "incumbent_ms"):
                    diagnostics[key] += int(game.get(key, 0))
            diagnostics["candidate_ms"] += float(game.get("candidate_ms", 0.0))
            diagnostics["incumbent_ms"] += float(game.get("incumbent_ms", 0.0))
            winner = int(game["winner"])
            candidate_player = int(game["candidate_player"])
            won = winner == candidate_player
            candidate_wins += int(won)
            incumbent_wins += int(not won)
            color_scores[f"candidate_player_{candidate_player}"].append(float(won))

    mean = sum(scores) / len(scores) if scores else 0.0

    statistics = manifest["statistics"]
    confidence = None
    if stage != "initial":
        stage_seed = statistics["seed"] ^ int.from_bytes(
            hashlib.sha256(stage.encode()).digest()[:8], "big"
        )
        confidence = source_game_cluster_bootstrap(
            pairs, statistics["resamples"], stage_seed
        )
    promotion_mean = mean if confidence is None else confidence["estimate"]
    color_means = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in color_scores.items()
    }
    control_role_means = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in control_role_scores.items()
    }
    role_uplifts = {
        key: color_means[key] - control_role_means[key]
        for key in color_means
    }
    historical_candidate_means = {
        key: sum(values) / len(values)
        for key, values in historical_candidate_scores.items()
    }
    historical_control_means = {
        key: sum(values) / len(values)
        for key, values in historical_control_scores.items()
    }
    historical_uplifts = {
        key: historical_candidate_means[key] - historical_control_means[key]
        for key in historical_candidate_means
    }
    regressions = len(pairs) - retained
    improvements = converted
    retention_score = retained / len(pairs)
    conversion_score = converted / len(pairs)
    minimum_physical_color_uplift = min(role_uplifts.values())
    minimum_historical_uplift = min(historical_uplifts.values())
    minimum_historical_role_score = min(historical_candidate_means.values())
    minimum_adjusted_uplift = min(
        minimum_physical_color_uplift, minimum_historical_uplift
    )
    control_normalization = {
        "method": (
            "rank5_vs_rank5_per_opening_same_time_budget_with_node_cap"
            if strength_profile is not None and
            strength_profile["mode"] == "time_ms"
            else "rank5_vs_rank5_per_opening_same_node_budget"
        ),
        "openings": len(pairs),
        "winner_counts": {
            "player_0": sum(control_role_scores["candidate_player_0"]),
            "player_1": sum(control_role_scores["candidate_player_1"]),
        },
        "physical_baseline_scores": control_role_means,
        "candidate_physical_scores": color_means,
        "physical_uplifts": role_uplifts,
        "minimum_physical_color_uplift": minimum_physical_color_uplift,
        "historical_baseline_scores": historical_control_means,
        "candidate_historical_scores": historical_candidate_means,
        "historical_uplifts": historical_uplifts,
        "minimum_historical_uplift": minimum_historical_uplift,
        "minimum_historical_role_score": minimum_historical_role_score,
        "winner_role": {
            "retained": retained,
            "regressed": regressions,
            "score": retention_score,
        },
        "loser_role": {
            "converted": converted,
            "not_converted": len(pairs) - converted,
            "score": conversion_score,
        },
        "improvements": improvements,
        "regressions": regressions,
        "net_uplift": (improvements - regressions) / (2.0 * len(pairs)),
        "minimum_adjusted_uplift": minimum_adjusted_uplift,
    }
    stratum_means = {
        key: sum(values) / len(values) for key, values in sorted(stratum_scores.items())
    }
    winner_tier_means = {
        key: sum(values) / len(values)
        for key, values in sorted(winner_tier_scores.items())
    }
    elite_tier_values = [
        value for key in ("rank1", "elite")
        for value in winner_tier_scores.get(key, [])
    ]
    elite_tier_mean = (
        sum(elite_tier_values) / len(elite_tier_values)
        if elite_tier_values else 0.0
    )
    diagnostics["candidate_mean_nodes"] = (
        diagnostics["candidate_nodes"] / diagnostics["candidate_searches"]
        if diagnostics["candidate_searches"] else 0.0
    )
    diagnostics["incumbent_mean_nodes"] = (
        diagnostics["incumbent_nodes"] / diagnostics["incumbent_searches"]
        if diagnostics["incumbent_searches"] else 0.0
    )
    diagnostics["candidate_mean_completed_depth"] = (
        diagnostics["candidate_depth_sum"] / diagnostics["candidate_searches"]
        if diagnostics["candidate_searches"] else 0.0
    )
    diagnostics["incumbent_mean_completed_depth"] = (
        diagnostics["incumbent_depth_sum"] / diagnostics["incumbent_searches"]
        if diagnostics["incumbent_searches"] else 0.0
    )
    diagnostics["candidate_nodes_per_ms"] = (
        diagnostics["candidate_nodes"] / diagnostics["candidate_ms"]
        if diagnostics["candidate_ms"] else 0.0
    )
    diagnostics["incumbent_nodes_per_ms"] = (
        diagnostics["incumbent_nodes"] / diagnostics["incumbent_ms"]
        if diagnostics["incumbent_ms"] else 0.0
    )
    diagnostics["candidate_to_incumbent_throughput"] = (
        diagnostics["candidate_nodes_per_ms"] /
        diagnostics["incumbent_nodes_per_ms"]
        if diagnostics["incumbent_nodes_per_ms"] else 0.0
    )
    base_config = manifest["stages"][stage]
    config = dict(base_config)
    if node_budget is not None:
        config.update(base_config.get("node_budget_overrides", {}).get(
            str(node_budget), {}
        ))
    if strength_profile is not None:
        if strength_profile["mode"] == "nodes":
            config.update(base_config.get("node_budget_overrides", {}).get(
                str(strength_profile["value"]), {}
            ))
        config.update(strength_profile.get("thresholds", {}))
    requirements = [
        requirement("operational_counts_zero", sum(operational.values()), "==", 0),
        requirement("cluster_mean_score", promotion_mean, ">=", config["minimum_mean"]),
    ]
    if config.get("require_more_wins_than_incumbent"):
        requirements.append(
            requirement("candidate_game_wins", candidate_wins, ">", incumbent_wins)
        )
    if config.get("require_at_least_as_many_wins_as_incumbent"):
        requirements.append(requirement(
            "candidate_game_wins_noninferior", candidate_wins, ">=", incumbent_wins
        ))
    if config.get("minimum_throughput_ratio") is not None:
        requirements.append(requirement(
            "candidate_to_incumbent_throughput",
            diagnostics["candidate_to_incumbent_throughput"], ">=",
            config["minimum_throughput_ratio"],
        ))
    if "minimum_ci_lower" in config:
        requirements.append(requirement(
            "paired_bootstrap_lower", confidence["lower"], ">",
            config["minimum_ci_lower"],
        ))
    if "minimum_color_score" in config:
        requirements.append(requirement(
            "minimum_color_score", min(color_means.values()), ">=",
            config["minimum_color_score"],
        ))
    if "minimum_control_adjusted_uplift" in config:
        requirements.append(requirement(
            "minimum_control_adjusted_uplift", minimum_adjusted_uplift, ">=",
            config["minimum_control_adjusted_uplift"],
        ))
    if "minimum_physical_color_uplift" in config:
        requirements.append(requirement(
            "minimum_physical_color_uplift", minimum_physical_color_uplift, ">=",
            config["minimum_physical_color_uplift"],
        ))
    if "minimum_historical_role_score" in config:
        requirements.append(requirement(
            "minimum_historical_role_score", minimum_historical_role_score, ">=",
            config["minimum_historical_role_score"],
        ))
    if "minimum_control_winner_retention" in config:
        requirements.append(requirement(
            "control_winner_retention", retention_score, ">=",
            config["minimum_control_winner_retention"],
        ))
    if "minimum_stratum_score" in config:
        requirements.append(requirement(
            "minimum_stratum_score", min(stratum_means.values()), ">=",
            config["minimum_stratum_score"],
        ))
    if "minimum_winner_tier_score" in config:
        requirements.append(requirement(
            "minimum_winner_tier_score", min(winner_tier_means.values()), ">=",
            config["minimum_winner_tier_score"],
        ))
    if "minimum_elite_tier_score" in config:
        requirements.append(requirement(
            "elite_winner_tier_score", elite_tier_mean, ">=",
            config["minimum_elite_tier_score"],
        ))
    passed = all(item["passed"] for item in requirements)
    return {
        "schema": "papersoccer.codingame-promotion-stage.v2",
        "stage": stage,
        "identity": identity,
        "verdict": "pass" if passed else "reject",
        "passed": passed,
        "reason_codes": [item["id"] for item in requirements if not item["passed"]],
        "pairs": {
            "total": len(pairs), "won_2_0": won_2_0,
            "split_1_1": split_1_1, "lost_0_2": lost_0_2,
            "mean_score": promotion_mean, "opening_pair_mean_score": mean,
            "candidate_game_wins": candidate_wins,
            "incumbent_game_wins": incumbent_wins,
        },
        "confidence_interval": confidence,
        "color_scores": color_means,
        "control_normalization": control_normalization,
        "stratum_scores": stratum_means,
        "winner_tier_scores": winner_tier_means,
        "elite_winner_tier_score": elite_tier_mean,
        "diagnostics": diagnostics,
        "operational": operational,
        "requirements": requirements,
        "records": pairs,
    }


def configured_node_budgets(config: dict):
    values = config.get("node_budgets", [config.get("node_budget")])
    try:
        budgets = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise UsageError("stage has invalid node budgets") from error
    if not budgets or any(value <= 0 for value in budgets):
        raise UsageError("stage has invalid node budgets")
    if len(set(budgets)) != len(budgets):
        raise UsageError("stage repeats a node budget")
    return budgets


def uses_explicit_strength_profiles(config: dict) -> bool:
    return "strength_profiles" in config


def configured_required_jobs(config: dict) -> int | None:
    value = config.get("required_jobs")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UsageError("stage required_jobs must be a positive integer")
    return value


def configured_strength_profiles(config: dict):
    if not uses_explicit_strength_profiles(config):
        return [
            {
                "id": f"{budget}-nodes",
                "mode": "nodes",
                "value": budget,
                "max_nodes": budget,
                "node_budget": budget,
                "time_budget_ms": 0,
                "directory": f"{budget}-nodes",
                "thresholds": {},
                "legacy": True,
            }
            for budget in configured_node_budgets(config)
        ]
    if "node_budget" in config or "node_budgets" in config:
        raise UsageError(
            "stage cannot mix strength_profiles with legacy node budgets"
        )
    raw_profiles = config.get("strength_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise UsageError("stage strength_profiles must be a non-empty list")
    profiles = []
    ids = set()
    executions = set()
    allowed = {"id", "mode", "value", "max_nodes", "thresholds"}
    for raw in raw_profiles:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise UsageError("stage has an invalid strength profile object")
        profile_id = raw.get("id")
        mode = raw.get("mode")
        value = raw.get("value")
        if (not isinstance(profile_id, str) or
                re.fullmatch(r"[a-z0-9][a-z0-9-]*", profile_id) is None):
            raise UsageError("strength profile id must be a filesystem-safe slug")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise UsageError("strength profile value must be a positive integer")
        thresholds = raw.get("thresholds", {})
        if not isinstance(thresholds, dict) or any(
            not isinstance(key, str) for key in thresholds
        ):
            raise UsageError("strength profile thresholds must be an object")
        unknown_thresholds = set(thresholds) - (
            PROFILE_BOOLEAN_THRESHOLDS | PROFILE_NUMERIC_THRESHOLDS
        )
        if unknown_thresholds:
            raise UsageError(
                "strength profile has unknown threshold keys: " +
                ", ".join(sorted(unknown_thresholds))
            )
        for key, threshold in thresholds.items():
            if key in PROFILE_BOOLEAN_THRESHOLDS:
                if not isinstance(threshold, bool):
                    raise UsageError(
                        f"strength profile threshold {key} must be boolean"
                    )
            elif key == "minimum_throughput_ratio" and threshold is None:
                pass
            elif (isinstance(threshold, bool) or
                  not isinstance(threshold, (int, float)) or
                  not math.isfinite(float(threshold))):
                raise UsageError(
                    f"strength profile threshold {key} must be finite numeric"
                )
        if mode == "nodes":
            if value > MAX_NODE_BUDGET:
                raise UsageError("node profile value exceeds uint64")
            max_nodes = raw.get("max_nodes", value)
            if (isinstance(max_nodes, bool) or
                    not isinstance(max_nodes, int) or max_nodes <= 0 or
                    max_nodes != value):
                raise UsageError(
                    "node profile max_nodes must equal its profile value"
                )
            node_budget = value
            time_budget_ms = 0
        elif mode == "time_ms":
            if value > MAX_TIME_BUDGET_MS:
                raise UsageError("time profile value exceeds uint32")
            max_nodes = raw.get("max_nodes")
            if (isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or
                    max_nodes <= 0 or max_nodes > MAX_NODE_BUDGET):
                raise UsageError(
                    "time profile requires a uint64 max_nodes safety cap"
                )
            node_budget = max_nodes
            time_budget_ms = value
        else:
            raise UsageError("strength profile mode must be nodes or time_ms")
        execution = (node_budget, time_budget_ms)
        if profile_id in ids:
            raise UsageError(f"duplicate strength profile id {profile_id}")
        if execution in executions:
            raise UsageError("strength profiles repeat an execution budget")
        ids.add(profile_id)
        executions.add(execution)
        profiles.append({
            "id": profile_id,
            "mode": mode,
            "value": value,
            "max_nodes": max_nodes,
            "node_budget": node_budget,
            "time_budget_ms": time_budget_ms,
            "directory": profile_id,
            "thresholds": dict(thresholds),
            "legacy": False,
        })
    return profiles


def strength_profile_identity(profile: dict):
    return {
        "id": profile["id"],
        "mode": profile["mode"],
        "value": profile["value"],
        "max_nodes": profile["max_nodes"],
    }


def combine_budget_profiles(stage: str, identity: dict, profiles: list[dict]):
    if len(profiles) == 1:
        return profiles[0]
    combined_requirements = []
    for profile in profiles:
        for item in profile["requirements"]:
            combined = dict(item)
            combined["id"] = f"nodes_{profile['node_budget']}:{item['id']}"
            combined_requirements.append(combined)
    passed = all(profile["passed"] for profile in profiles)
    return {
        "schema": "papersoccer.codingame-promotion-stage.v2",
        "stage": stage,
        "identity": identity,
        "verdict": "pass" if passed else "reject",
        "passed": passed,
        "reason_codes": [
            item["id"] for item in combined_requirements if not item["passed"]
        ],
        "node_budget_profiles": profiles,
        "requirements": combined_requirements,
    }


def combine_strength_profiles(stage: str, identity: dict,
                              profiles: list[dict]):
    combined_requirements = []
    for profile in profiles:
        profile_id = profile["strength_profile"]["id"]
        for item in profile["requirements"]:
            combined = dict(item)
            combined["id"] = f"{profile_id}:{item['id']}"
            combined_requirements.append(combined)
    passed = all(profile["passed"] for profile in profiles)
    return {
        "schema": "papersoccer.codingame-promotion-stage.v2",
        "stage": stage,
        "identity": identity,
        "verdict": "pass" if passed else "reject",
        "passed": passed,
        "reason_codes": [
            item["id"] for item in combined_requirements if not item["passed"]
        ],
        "strength_profiles": profiles,
        "requirements": combined_requirements,
    }


def shard_configuration_matches_profile(configuration: dict, stage: str,
                                        profile: dict, maximum_turns: int,
                                        shard_count: int,
                                        shard_index: int | None = None):
    if not isinstance(configuration, dict):
        return False
    try:
        matches = (
            configuration.get("stage") == stage
            and int(configuration.get("node_budget", -1)) ==
                profile["node_budget"]
            and int(configuration.get("maximum_turns", -1)) == maximum_turns
            and int(configuration.get("shard_count", -1)) == shard_count
        )
        if profile["legacy"]:
            matches = matches and int(
                configuration.get("time_budget_ms", 0)
            ) == 0
        else:
            matches = matches and int(
                configuration.get("time_budget_ms", -1)
            ) == profile["time_budget_ms"]
        if shard_index is not None:
            matches = matches and int(
                configuration.get("shard_index", -1)
            ) == shard_index
        return matches
    except (TypeError, ValueError):
        return False


def aggregate_strength_profile(manifest: dict, stage: str, shards: list[dict],
                               identity: dict, profile: dict):
    if profile["legacy"]:
        report = aggregate_stage(
            manifest, stage, shards, identity, node_budget=profile["value"]
        )
        report["node_budget"] = profile["value"]
        return report
    report = aggregate_stage(
        manifest, stage, shards, identity, strength_profile=profile
    )
    report["strength_profile"] = strength_profile_identity(profile)
    return report


def combine_configured_profiles(stage: str, identity: dict, profiles: list[dict],
                                reports: list[dict]):
    if any(profile["legacy"] for profile in profiles):
        if not all(profile["legacy"] for profile in profiles):
            raise ValueError("internal mixed legacy/explicit profile list")
        return combine_budget_profiles(stage, identity, reports)
    return combine_strength_profiles(stage, identity, reports)


def reaggregate_stage_from_shards(directory: pathlib.Path, manifest: dict,
                                  stage: str, identity: dict):
    config = manifest["stages"][stage]
    profiles = configured_strength_profiles(config)
    reports = []
    expected_shard_count = None
    for profile in profiles:
        shard_directory = (
            directory / "shards" / stage / profile["directory"]
        )
        paths = sorted(shard_directory.glob("shard-*-of-*.json"))
        if not paths:
            raise IncompleteError(
                f"stage {stage} has no {profile['id']} shards"
            )
        shards = [json.loads(path.read_text()) for path in paths]
        shard_count = len(shards)
        if expected_shard_count is None:
            expected_shard_count = shard_count
        elif shard_count != expected_shard_count:
            raise IncompleteError(
                f"stage {stage} profiles use different shard counts"
            )
        indices = set()
        for shard in shards:
            if not isinstance(shard, dict):
                raise IncompleteError(
                    f"stage {stage} contains a malformed shard"
                )
            configuration = shard.get("configuration", {})
            if not shard_configuration_matches_profile(
                configuration, stage, profile, config["maximum_turns"],
                shard_count,
            ):
                raise IncompleteError(f"stage {stage} shard configuration mismatch")
            indices.add(int(configuration.get("shard_index", -1)))
        if indices != set(range(shard_count)):
            raise IncompleteError(f"stage {stage} shard indices are incomplete")
        reports.append(
            aggregate_strength_profile(manifest, stage, shards, identity, profile)
        )
    return combine_configured_profiles(stage, identity, profiles, reports)


def require_stage_prerequisites(directory: pathlib.Path, stage: str, bot: str,
                                manifest: dict, candidate_hash: str,
                                manifest_hash: str, paths: dict):
    preflight_path = directory / "preflight.json"
    if not preflight_path.exists():
        raise IncompleteError("run preflight before strength stages")
    preflight_report = json.loads(preflight_path.read_text())
    if (
        preflight_report.get("schema") != "papersoccer.codingame-promotion-preflight.v1"
        or preflight_report.get("bot") != bot
        or preflight_report.get("passed") is not True
        or preflight_report.get("candidate_submission_sha256") != candidate_hash
        or preflight_report.get("manifest_sha256") != manifest_hash
        or not all(paths[name].exists() for name in ("test", "runner", "timing"))
        or preflight_report.get("artifact_test_binary_sha256") != sha256(paths["test"])
        or preflight_report.get("runner_binary_sha256") != sha256(paths["runner"])
        or preflight_report.get("timing_binary_sha256") != sha256(paths["timing"])
    ):
        raise IncompleteError("current preflight is missing or did not pass")
    for predecessor in STAGE_PREDECESSORS[stage]:
        path = directory / f"{predecessor}.json"
        if not path.exists():
            raise IncompleteError(f"stage {stage} requires {predecessor}")
        report = json.loads(path.read_text())
        expected_identity = expected_stage_identity(
            manifest, manifest_hash, predecessor, candidate_hash,
            sha256(paths["runner"])
        )
        if (
            report.get("schema") != "papersoccer.codingame-promotion-stage.v2"
            or report.get("stage") != predecessor
            or report.get("passed") is not True
            or report.get("identity") != expected_identity
        ):
            raise IncompleteError(
                f"stage {stage} requires a passing current {predecessor} report"
            )
        try:
            recomputed = reaggregate_stage_from_shards(
                directory, manifest, predecessor, expected_identity
            )
        except (GateError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            raise IncompleteError(
                f"stage {stage} requires complete raw {predecessor} evidence"
            ) from error
        if report != recomputed:
            raise IncompleteError(
                f"stage {stage} predecessor {predecessor} report does not "
                "match its raw evidence"
            )


def write_stage_rejection(directory: pathlib.Path, bot: str, stage: str,
                          report: dict, manifest: dict):
    statuses = {}
    failed_index = REQUIRED_STAGES.index(stage)
    for index, name in enumerate(REQUIRED_STAGES):
        if index < failed_index:
            statuses[name] = "pass"
        elif name == stage:
            statuses[name] = "reject"
        else:
            statuses[name] = "not_run_due_to_rejection"
    identity = report["identity"]
    completed_artifacts = [directory / "preflight.json"] + [
        directory / f"{name}.json"
        for name in REQUIRED_STAGES[:failed_index + 1]
    ]
    decision = {
        "schema": "papersoccer.codingame-promotion-decision.v1",
        "bot": bot,
        "candidate_submission_sha256": identity["candidate_submission_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "incumbent_submission_sha256": manifest["incumbent"]["submission_sha256"],
        "runner_sha256": identity["runner_sha256"],
        "verdict": "REJECT",
        "submission_worthy": False,
        "failed_stage": stage,
        "reason_codes": report["reason_codes"],
        "stage_status": statuses,
        "artifacts": [str(path.relative_to(ROOT)) for path in completed_artifacts],
    }
    atomic_json(directory / "decision.json", decision)
    return decision


def run_stage(bot: str, stage: str, jobs: int, manifest_path: pathlib.Path,
              build: pathlib.Path, results_root: pathlib.Path):
    validation = validate(manifest_path, bot)
    manifest = load_manifest(manifest_path)
    if stage not in manifest["stages"]:
        raise UsageError(f"unknown stage {stage}")
    paths = candidate_paths(bot, build)
    check_generated_submission(bot)
    check_generated_submission("rank_5")
    build_targets(build, [
        paths["test_target"], paths["runner_target"], paths["timing_target"]
    ])
    if not paths["submission"].exists():
        raise IncompleteError("candidate submission has not been generated")
    candidate_hash = sha256(paths["submission"])
    bank_relative = manifest["stages"][stage]["bank"]
    bank = PROMOTION / bank_relative
    bank_hash = sha256(bank)
    identity = expected_stage_identity(
        manifest, validation["manifest_sha256"], stage,
        candidate_hash, sha256(paths["runner"])
    )
    if identity["bank_sha256"] != bank_hash:
        raise UsageError("current bank does not match the validated manifest")
    config = manifest["stages"][stage]
    required_jobs = configured_required_jobs(config)
    if required_jobs is not None and jobs != required_jobs:
        raise UsageError(
            f"stage {stage} requires exactly {required_jobs} jobs"
        )
    record_count = manifest["banks"][bank_relative]["records"]
    jobs = max(1, min(jobs, record_count))
    destination = result_directory(
        bot, candidate_hash, validation["manifest_sha256"], results_root
    )
    bank = snapshot_verified_bank(
        bank,
        destination / "banks" / f"{stage}-{bank_hash}.tsv",
        bank_hash,
    )
    require_stage_prerequisites(
        destination, stage, bot, manifest, candidate_hash,
        validation["manifest_sha256"], paths
    )
    strength_profiles = configured_strength_profiles(config)
    if stage == "test":
        if not (ROOT / ".git").is_dir():
            raise UsageError("locked-test consumption requires the repository Git directory")
        consumption = locked_test_consumption_path(bank_hash)
        marker = locked_test_consumption_marker(
            manifest, validation["manifest_sha256"], candidate_hash, jobs
        )
        try:
            atomic_json_create_exclusive(consumption, marker)
        except FileExistsError:
            try:
                existing_marker = json.loads(consumption.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise UsageError(
                    f"locked test bank has an invalid consumption marker: {error}"
                ) from error
            if existing_marker != marker:
                raise UsageError(
                    "locked test bank was already exposed to another run identity; "
                    "freeze a new game-disjoint test bank"
                )

    write_incomplete_decision(
        destination, bot, candidate_hash, validation["manifest_sha256"],
        manifest, f"stage:{stage}", identity["runner_sha256"]
    )

    def run_one(specification):
        profile, index = specification
        shard_directory = (
            destination / "shards" / stage / profile["directory"]
        )
        shard_directory.mkdir(parents=True, exist_ok=True)
        final_path = shard_directory / f"shard-{index:03d}-of-{jobs:03d}.json"
        temporary = final_path.with_suffix(".json.tmp")
        if final_path.exists():
            try:
                cached = json.loads(final_path.read_text())
                cached_config = cached["configuration"]
                if (
                    cached.get("schema") == "papersoccer.codingame-promotion-shard.v2"
                    and cached.get("identity") == identity
                    and shard_configuration_matches_profile(
                        cached_config, stage, profile, config["maximum_turns"],
                        jobs, index,
                    )
                    and not any(int(cached["operational"].get(field, 0))
                                for field in OPERATIONAL_FIELDS)
                ):
                    return profile["id"], cached
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        command = [
            paths["runner"], "--bank", bank, "--stage", stage,
            "--node-budget", profile["node_budget"],
            "--max-turns", config["maximum_turns"],
            "--shard-count", jobs, "--shard-index", index,
            "--output", temporary,
            "--manifest-sha256", identity["manifest_sha256"],
            "--bank-sha256", identity["bank_sha256"],
            "--candidate-sha256", identity["candidate_submission_sha256"],
            "--incumbent-sha256", identity["incumbent_submission_sha256"],
            "--runner-sha256", identity["runner_sha256"],
        ]
        if profile["time_budget_ms"]:
            command.extend(["--time-budget-ms", profile["time_budget_ms"]])
        run_checked(command)
        try:
            produced = json.loads(temporary.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GateError("runner produced an unreadable shard") from error
        produced_config = (
            produced.get("configuration", {})
            if isinstance(produced, dict) else {}
        )
        if (
            not isinstance(produced, dict)
            or produced.get("schema") !=
                "papersoccer.codingame-promotion-shard.v2"
            or produced.get("identity") != identity
            or not shard_configuration_matches_profile(
                produced_config, stage, profile, config["maximum_turns"],
                jobs, index,
            )
        ):
            raise GateError("runner produced a mismatched shard identity")
        os.replace(temporary, final_path)
        return profile["id"], produced

    specifications = [
        (profile, index)
        for profile in strength_profiles
        for index in range(jobs)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        completed = list(executor.map(run_one, specifications))
    shards_by_profile = {profile["id"]: [] for profile in strength_profiles}
    for profile_id, shard in completed:
        shards_by_profile[profile_id].append(shard)
    profile_reports = []
    for profile in strength_profiles:
        profile_reports.append(aggregate_strength_profile(
            manifest, stage, shards_by_profile[profile["id"]], identity,
            profile,
        ))
    report = combine_configured_profiles(
        stage, identity, strength_profiles, profile_reports
    )
    atomic_json(destination / f"{stage}.json", report)
    if not report["passed"]:
        write_stage_rejection(destination, bot, stage, report, manifest)
    printable = {key: value for key, value in report.items() if key != "records"}
    if "node_budget_profiles" in printable:
        printable["node_budget_profiles"] = [
            {key: value for key, value in profile.items() if key != "records"}
            for profile in printable["node_budget_profiles"]
        ]
    if "strength_profiles" in printable:
        printable["strength_profiles"] = [
            {key: value for key, value in profile.items() if key != "records"}
            for profile in printable["strength_profiles"]
        ]
    print(stable_json(printable), end="")
    return report


TIMING_PATTERN = re.compile(
    r"player=(?P<player>[01]) first_ms=(?P<first>[0-9.]+) later_ms=(?P<later>[0-9.]+)"
)
SHELL_TIMING_PATTERN = re.compile(
    r"shell=(?P<case>[a-z0-9-]+) later_ms=(?P<later>[0-9.]+)"
)


def make_timing_report(bot: str, candidate_hash: str, manifest_hash: str,
                       timing_hash: str, samples: list[dict], config: dict,
                       operational_error: str | None = None):
    requirements = []
    if operational_error is None:
        expected_cases = {
            "initial-player-0", "initial-player-1", *config["shell_cases"]
        }
        observed = {(item["sample"], item["case"]) for item in samples}
        expected = {
            (sample, case)
            for sample in range(config["fresh_process_samples"])
            for case in expected_cases
        }
        measurements_valid = (
            observed == expected and len(samples) == len(expected)
            and all(
                math.isfinite(float(item["later_ms"]))
                and float(item["later_ms"]) >= 0.0
                and (
                    item["case"] in config["shell_cases"]
                    or (
                        "first_ms" in item
                        and math.isfinite(float(item["first_ms"]))
                        and float(item["first_ms"]) >= 0.0
                    )
                )
                for item in samples
            )
        )
        requirements.append({
            "id": "timing_samples_complete",
            "passed": measurements_valid,
            "observed": len(samples),
            "operator": "==",
            "threshold": len(expected),
        })
        if measurements_valid:
            first = sorted(
                float(item["first_ms"]) for item in samples
                if "first_ms" in item
            )
            later = sorted(float(item["later_ms"]) for item in samples)
            requirements.extend([
                {"id": "first_p95", "passed": quantile(first, 0.95) < config["first_p95_ms"],
                 "observed": quantile(first, 0.95), "operator": "<", "threshold": config["first_p95_ms"]},
                {"id": "first_max", "passed": max(first) < config["first_max_ms"],
                 "observed": max(first), "operator": "<", "threshold": config["first_max_ms"]},
                {"id": "later_p95", "passed": quantile(later, 0.95) < config["later_p95_ms"],
                 "observed": quantile(later, 0.95), "operator": "<", "threshold": config["later_p95_ms"]},
                {"id": "later_max", "passed": max(later) < config["later_max_ms"],
                 "observed": max(later), "operator": "<", "threshold": config["later_max_ms"]},
            ])
    else:
        requirements.append({
            "id": "timing_probe_operational",
            "passed": False,
            "detail": operational_error,
        })
    passed = all(item["passed"] for item in requirements)
    return {
        "schema": "papersoccer.codingame-promotion-timing.v1",
        "bot": bot,
        "candidate_submission_sha256": candidate_hash,
        "manifest_sha256": manifest_hash,
        "timing_binary_sha256": timing_hash,
        "passed": passed,
        "reason_codes": [item["id"] for item in requirements if not item["passed"]],
        "requirements": requirements,
        "samples": samples,
    }


def timing(bot: str, manifest_path: pathlib.Path, build: pathlib.Path,
           results_root: pathlib.Path):
    validation = validate(manifest_path, bot)
    manifest = load_manifest(manifest_path)
    paths = candidate_paths(bot, build)
    check_generated_submission(bot)
    check_generated_submission("rank_5")
    build_targets(build, [
        paths["test_target"], paths["runner_target"], paths["timing_target"]
    ])
    candidate_hash = sha256(paths["submission"])
    destination = result_directory(
        bot, candidate_hash, validation["manifest_sha256"], results_root
    )
    require_stage_prerequisites(
        destination, "test", bot, manifest, candidate_hash,
        validation["manifest_sha256"], paths
    )
    test_report = json.loads((destination / "test.json").read_text())
    expected_test_identity = expected_stage_identity(
        manifest, validation["manifest_sha256"], "test", candidate_hash,
        sha256(paths["runner"])
    )
    if (
        test_report.get("schema") != "papersoccer.codingame-promotion-stage.v2"
        or test_report.get("stage") != "test"
        or test_report.get("identity") != expected_test_identity
        or test_report.get("passed") is not True
    ):
        raise IncompleteError("timing requires a passing locked test")
    try:
        recomputed_test_report = reaggregate_stage_from_shards(
            destination, manifest, "test", expected_test_identity
        )
    except (GateError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        raise IncompleteError(
            "timing requires complete raw locked-test evidence"
        ) from error
    if test_report != recomputed_test_report:
        raise IncompleteError(
            "timing locked-test report does not match its raw evidence"
        )
    verify_locked_test_consumption(
        destination, manifest, validation["manifest_sha256"], candidate_hash
    )
    write_incomplete_decision(
        destination, bot, candidate_hash, validation["manifest_sha256"],
        manifest, "timing", sha256(paths["runner"])
    )
    samples = []
    config = manifest["timing"]
    operational_error = None
    try:
        for sample in range(config["fresh_process_samples"]):
            completed = run_checked([paths["timing"]])
            initial_matches = list(TIMING_PATTERN.finditer(completed.stdout))
            shell_matches = list(SHELL_TIMING_PATTERN.finditer(completed.stdout))
            if (
                len(initial_matches) != 2
                or [match.group("case") for match in shell_matches]
                != config["shell_cases"]
            ):
                raise GateError("timing probe produced unexpected output")
            for match in initial_matches:
                player = int(match.group("player"))
                samples.append({
                    "sample": sample,
                    "case": f"initial-player-{player}",
                    "player": player,
                    "first_ms": float(match.group("first")),
                    "later_ms": float(match.group("later")),
                })
            for match in shell_matches:
                samples.append({
                    "sample": sample,
                    "case": match.group("case"),
                    "later_ms": float(match.group("later")),
                })
    except (GateError, KeyError, TypeError, ValueError) as error:
        operational_error = str(error)
    report = make_timing_report(
        bot, candidate_hash, validation["manifest_sha256"],
        sha256(paths["timing"]), samples, config, operational_error
    )
    report_path = destination / "timing.json"
    atomic_json(report_path, report)
    if not report["passed"]:
        artifacts = [destination / "preflight.json"] + [
            destination / f"{stage}.json" for stage in REQUIRED_STAGES
        ] + [report_path]
        write_nonstage_rejection(
            destination, bot, "timing", report, manifest,
            sha256(paths["runner"]), artifacts
        )
    print(stable_json(report), end="")
    return report


def evaluate(bot: str, manifest_path: pathlib.Path, build: pathlib.Path,
             results_root: pathlib.Path):
    validation = validate(manifest_path, bot)
    manifest = load_manifest(manifest_path)
    paths = candidate_paths(bot, build)
    check_generated_submission(bot)
    check_generated_submission("rank_5")
    build_targets(build, [
        paths["test_target"], paths["runner_target"], paths["timing_target"]
    ])
    run_checked([paths["test"]])
    submission = paths["submission"]
    if not submission.exists():
        raise IncompleteError("candidate submission is missing")
    candidate_hash = sha256(submission)
    submission_data = submission.read_bytes()
    manifest_hash = validation["manifest_sha256"]
    directory = result_directory(bot, candidate_hash, manifest_hash, results_root)
    runner_hash = sha256(paths["runner"])
    preflight_path = directory / "preflight.json"
    current_preflight = None
    if preflight_path.exists():
        current_preflight = json.loads(preflight_path.read_text())
    preflight_identity_matches = current_preflight is not None and (
        current_preflight.get("schema") ==
        "papersoccer.codingame-promotion-preflight.v1"
        and current_preflight.get("bot") == bot
        and current_preflight.get("candidate_submission_sha256") == candidate_hash
        and current_preflight.get("manifest_sha256") == manifest_hash
        and current_preflight.get("artifact_test_binary_sha256") == sha256(paths["test"])
        and current_preflight.get("runner_binary_sha256") == runner_hash
        and current_preflight.get("timing_binary_sha256") == sha256(paths["timing"])
    )
    preflight_failed_requirements = (
        [
            item.get("id") for item in current_preflight.get("requirements", [])
            if item.get("passed") is not True
        ]
        if current_preflight is not None else []
    )
    expected_preflight_requirements = [
        {"id": "generated_submission_exists", "passed": True},
        {"id": "generated_submission_ascii",
         "passed": all(byte < 128 for byte in submission_data)},
        candidate_hypothesis_requirement(manifest, candidate_hash),
        {"id": "generated_submission_size",
         "passed": len(submission_data) <= manifest["source_limit"],
         "observed": len(submission_data), "operator": "<=",
         "threshold": manifest["source_limit"]},
        {"id": "artifact_tests", "passed": True, "detail": None},
        {"id": "banks_and_incumbent_frozen", "passed": validation["valid"]},
    ]
    preflight_evidence_matches = preflight_identity_matches and (
        current_preflight.get("reason_codes") == preflight_failed_requirements
        and current_preflight.get("candidate_submission_characters") ==
            len(submission_data)
        and current_preflight.get("requirements") == expected_preflight_requirements
    )
    preflight_matches = preflight_evidence_matches and (
        current_preflight.get("passed") is True
        and not preflight_failed_requirements
    )
    decision_path = directory / "decision.json"
    if decision_path.exists():
        existing = json.loads(decision_path.read_text())
        failed_stage = existing.get("failed_stage")
        failed_report = None
        if failed_stage in REQUIRED_STAGES:
            failed_path = directory / f"{failed_stage}.json"
            if failed_path.exists():
                failed_report = json.loads(failed_path.read_text())
        failed_identity = (
            expected_stage_identity(
                manifest, manifest_hash, failed_stage, candidate_hash, runner_hash
            )
            if failed_stage in REQUIRED_STAGES else None
        )
        common_rejection_identity = (
            existing.get("schema") == "papersoccer.codingame-promotion-decision.v1"
            and existing.get("verdict") == "REJECT"
            and existing.get("submission_worthy") is False
            and existing.get("candidate_submission_sha256") == candidate_hash
            and existing.get("manifest_sha256") == manifest_hash
            and existing.get("incumbent_submission_sha256") ==
                manifest["incumbent"]["submission_sha256"]
            and existing.get("runner_sha256") == runner_hash
        )
        strength_rejection_valid = (
            common_rejection_identity
            and preflight_matches
            and failed_report is not None
            and failed_report.get("schema") ==
                "papersoccer.codingame-promotion-stage.v2"
            and failed_report.get("stage") == failed_stage
            and failed_report.get("identity") == failed_identity
            and failed_report.get("passed") is False
            and existing.get("reason_codes") == failed_report.get("reason_codes")
        )
        if strength_rejection_valid:
            try:
                strength_rejection_valid = failed_report == reaggregate_stage_from_shards(
                    directory, manifest, failed_stage, failed_identity
                )
            except (GateError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                strength_rejection_valid = False
        preflight_rejection_valid = (
            common_rejection_identity
            and failed_stage == "preflight"
            and preflight_evidence_matches
            and current_preflight.get("passed") is False
            and bool(preflight_failed_requirements)
            and existing.get("reason_codes") == preflight_failed_requirements
        )
        timing_rejection_valid = False
        if common_rejection_identity and failed_stage == "timing" and preflight_matches:
            timing_path = directory / "timing.json"
            if timing_path.exists():
                failed_timing = json.loads(timing_path.read_text())
                timing_rejection_valid = (
                    failed_timing.get("schema") ==
                        "papersoccer.codingame-promotion-timing.v1"
                    and failed_timing.get("bot") == bot
                    and failed_timing.get("candidate_submission_sha256") == candidate_hash
                    and failed_timing.get("manifest_sha256") == manifest_hash
                    and failed_timing.get("timing_binary_sha256") == sha256(paths["timing"])
                    and failed_timing.get("passed") is False
                    and existing.get("reason_codes") == failed_timing.get("reason_codes")
                )
        if strength_rejection_valid or preflight_rejection_valid or timing_rejection_valid:
            print(stable_json(existing), end="")
            return existing
    write_incomplete_decision(
        directory, bot, candidate_hash, manifest_hash, manifest,
        "evaluate", runner_hash
    )
    required_files = [directory / "preflight.json", directory / "timing.json"] + [
        directory / f"{stage}.json" for stage in REQUIRED_STAGES
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise IncompleteError("missing promotion artifacts: " + ", ".join(missing))
    reports = [json.loads(path.read_text()) for path in required_files]
    preflight_report, timing_report, *stage_reports = reports
    problems = []
    if not preflight_matches or preflight_report != current_preflight:
        problems.append("preflight_identity")
    if (
        timing_report.get("schema") != "papersoccer.codingame-promotion-timing.v1"
        or timing_report.get("bot") != bot
        or timing_report.get("candidate_submission_sha256") != candidate_hash
        or timing_report.get("manifest_sha256") != manifest_hash
        or timing_report.get("timing_binary_sha256") != sha256(paths["timing"])
    ):
        problems.append("timing_identity")
    else:
        try:
            recomputed_timing = make_timing_report(
                bot, candidate_hash, manifest_hash, sha256(paths["timing"]),
                timing_report["samples"], manifest["timing"]
            )
        except (KeyError, TypeError, ValueError):
            problems.append("timing_evidence")
        else:
            if timing_report != recomputed_timing:
                problems.append("timing_evidence")
    for stage, report in zip(REQUIRED_STAGES, stage_reports):
        expected_identity = expected_stage_identity(
            manifest, manifest_hash, stage, candidate_hash, runner_hash
        )
        if (
            report.get("schema") != "papersoccer.codingame-promotion-stage.v2"
            or report.get("stage") != stage
            or report.get("identity") != expected_identity
        ):
            problems.append(f"{stage}_identity")
            continue
        try:
            recomputed = reaggregate_stage_from_shards(
                directory, manifest, stage, expected_identity
            )
        except (GateError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            problems.append(f"{stage}_evidence")
            continue
        if report != recomputed:
            problems.append(f"{stage}_evidence")
    try:
        verify_locked_test_consumption(
            directory, manifest, manifest_hash, candidate_hash
        )
    except IncompleteError:
        problems.append("locked_test_consumption_identity")
    if problems:
        raise IncompleteError("promotion artifact identity mismatch: " + ", ".join(problems))
    passed = all(report.get("passed") is True for report in reports)
    reason_codes = []
    for path, report in zip(required_files, reports):
        if report.get("passed") is not True:
            reason_codes.append(path.stem)
            reason_codes.extend(report.get("reason_codes", []))
    final = {
        "schema": "papersoccer.codingame-promotion-decision.v1",
        "bot": bot,
        "candidate_submission_sha256": candidate_hash,
        "manifest_sha256": manifest_hash,
        "incumbent_submission_sha256": manifest["incumbent"]["submission_sha256"],
        "runner_sha256": runner_hash,
        "verdict": "PROMOTE" if passed else "REJECT",
        "submission_worthy": passed,
        "reason_codes": reason_codes,
        "artifacts": [str(path.relative_to(ROOT)) for path in required_files],
    }
    atomic_json(decision_path, final)
    print(stable_json(final), end="")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "preflight", "run", "timing", "evaluate", "all"))
    parser.add_argument("--bot", default="conservative_frontier_proof")
    parser.add_argument("--stage", choices=REQUIRED_STAGES)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build", type=pathlib.Path, default=DEFAULT_BUILD)
    parser.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULTS)
    arguments = parser.parse_args()
    if arguments.jobs <= 0:
        raise UsageError("jobs must be positive")
    if arguments.command == "validate":
        print(stable_json(validate(arguments.manifest, arguments.bot)), end="")
        return 0
    if arguments.command == "preflight":
        return 0 if preflight(arguments.bot, arguments.manifest, arguments.build, arguments.results)["passed"] else 10
    if arguments.command == "run":
        if arguments.stage is None:
            raise UsageError("run requires --stage")
        return 0 if run_stage(arguments.bot, arguments.stage, arguments.jobs,
                              arguments.manifest, arguments.build, arguments.results)["passed"] else 10
    if arguments.command == "timing":
        return 0 if timing(arguments.bot, arguments.manifest, arguments.build,
                           arguments.results)["passed"] else 10
    if arguments.command == "evaluate":
        return 0 if evaluate(arguments.bot, arguments.manifest, arguments.build,
                             arguments.results)["submission_worthy"] else 10
    preflight_report = preflight(
        arguments.bot, arguments.manifest, arguments.build, arguments.results
    )
    if not preflight_report["passed"]:
        return 10
    for stage in REQUIRED_STAGES:
        report = run_stage(arguments.bot, stage, arguments.jobs,
                           arguments.manifest, arguments.build, arguments.results)
        if not report["passed"]:
            return 10
    timing_report = timing(arguments.bot, arguments.manifest, arguments.build,
                           arguments.results)
    if not timing_report["passed"]:
        return 10
    return 0 if evaluate(arguments.bot, arguments.manifest, arguments.build,
                         arguments.results)["submission_worthy"] else 10


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"promotion gate: {error}", file=sys.stderr)
        raise SystemExit(error.exit_code)
