#!/usr/bin/env python3
"""Run and content-address one same-runtime Jacek-native search A/B."""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
from typing import Any, Iterator, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "papersoccer.jacek-native-search-ab-report/v2"
SOURCE_PATHS = (
    "tools/jacek_native_search_ab_gate.cpp",
    "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
    "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "src/bots/mcts_internal.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
)
FINAL_FORMULAS = (
    "value-log-visits",
    "value-only",
    "value-log-visits-plus3",
    "value-log-selection-visits-plus3",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
RUNTIME_SCHEMA = "papersoccer.jacek-native-runtime-model/v1"
MODEL_SCHEMA = "jacek_native_model/v1"
FEATURE_SCHEMA = "canonical-edges316-onehot-true-turn-distance105x8-v1"
RUNTIME_WEIGHT_COUNT = 1156 * 32 + 32 * 32 + 32
INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
FLOAT = re.compile(
    r"-?(?:(?:0|[1-9][0-9]*)\.[0-9]+|(?:0|[1-9][0-9]*))(?:[eE][+-]?[0-9]+)?"
)
PAIR = re.compile(
    r"pair=(0|[1-9][0-9]*) opening_turns=(0|[1-9][0-9]*) "
    r"seed=(0|[1-9][0-9]*) c([01])=(-1|0|1) c([01])=(-1|0|1)"
)
SUMMARY_FIELDS = {
    "candidate", "baseline", "unfinished", "candidate_player_one",
    "candidate_player_two", "games", "candidate_decisions",
    "candidate_expansions", "candidate_child_evaluations",
    "candidate_completed_actions", "candidate_partial_paths",
    "candidate_max_tree", "candidate_tree_cap_searches",
    "candidate_final_overrides", "candidate_ms", "candidate_max_first_ms",
    "candidate_max_later_ms", "candidate_deadline_searches",
    "candidate_headroom_failures", "candidate_operational_timeouts",
    "baseline_decisions", "baseline_expansions",
    "baseline_child_evaluations", "baseline_completed_actions",
    "baseline_partial_paths", "baseline_max_tree",
    "baseline_tree_cap_searches", "baseline_final_overrides", "baseline_ms",
    "baseline_max_first_ms", "baseline_max_later_ms",
    "baseline_deadline_searches", "baseline_headroom_failures",
    "baseline_operational_timeouts", "profile", "pairs", "maximum_turns",
    "opening_turns", "opening_seed", "shuffle_seed_policy",
    "runtime_policy", "game_order_policy", "timing_scope",
    "candidate_tree_nodes", "candidate_c", "candidate_fpu",
    "candidate_final", "baseline_tree_nodes", "baseline_c", "baseline_fpu",
    "baseline_final", "required_total", "required_per_color", "passed",
}
COUNT_SUFFIXES = (
    "decisions", "expansions", "child_evaluations", "completed_actions",
    "partial_paths", "max_tree", "tree_cap_searches", "final_overrides",
    "deadline_searches", "headroom_failures", "operational_timeouts",
)


class RecordError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode()


def key_values(line: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in line.split():
        key, separator, value = field.partition("=")
        if not separator or not key or not value or key in result:
            raise RecordError(f"{label} contains a malformed field")
        result[key] = value
    return result


def scalar(value: str) -> str | int | float | bool:
    if value == "true":
        return True
    if value == "false":
        return False
    if INTEGER.fullmatch(value):
        return int(value)
    if FLOAT.fullmatch(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise RecordError("gate output contains a non-finite number")
        return parsed
    return value


def _opening_depths(value: str) -> list[int]:
    fields = value.split(",")
    if not fields or any(INTEGER.fullmatch(field) is None for field in fields):
        raise RecordError("opening turns are not canonical integers")
    values = [int(field) for field in fields]
    if any(value < 0 for value in values):
        raise RecordError("opening turns must be nonnegative")
    return values


def _profile(arguments: argparse.Namespace, side: str) -> dict[str, Any]:
    return {
        "tree_nodes": getattr(arguments, f"{side}_tree_nodes"),
        "c": getattr(arguments, f"{side}_c"),
        "fpu": getattr(arguments, f"{side}_fpu"),
        "final": getattr(arguments, f"{side}_final"),
    }


def _require_unsigned(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordError(f"gate {label} is not an unsigned integer")
    return value


def _require_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"gate {label} is not numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RecordError(f"gate {label} is not finite/nonnegative")
    return parsed


def runtime_identity(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise RecordError("runtime checkpoint is not canonical line text")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RecordError("runtime checkpoint is not UTF-8") from error
    if (
        len(lines) != 7
        or lines[:3] != [RUNTIME_SCHEMA, MODEL_SCHEMA, FEATURE_SCHEMA]
        or SHA256.fullmatch(lines[3]) is None
        or SHA256.fullmatch(lines[4]) is None
    ):
        raise RecordError("runtime checkpoint metadata is malformed")
    try:
        scales = [float(value) for value in lines[5].split()]
        payload = base64.b64decode(lines[6], validate=True)
    except (ValueError, binascii.Error) as error:
        raise RecordError("runtime checkpoint payload is malformed") from error
    if (
        len(scales) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in scales)
        or len(payload) != (RUNTIME_WEIGHT_COUNT * 3 + 7) // 8
        or sha256(payload) != lines[4]
        or base64.b64encode(payload).decode("ascii") != lines[6]
    ):
        raise RecordError("runtime checkpoint payload identity is stale")
    return {
        "runtime_sha256": sha256(raw),
        "model_sha256": lines[3],
        "packed_sha256": lines[4],
    }


def _opening_seed_matches(base_seed: int, pair: int, observed: int) -> bool:
    mask = (1 << 64) - 1
    initial = (
        base_seed + (pair + 1) * 0x9E3779B97F4A7C15
    ) & mask
    retry_step = 0xD1B54A32D192ED03
    return any(
        ((initial + retry * retry_step) & mask) == observed
        for retry in range(4_096)
    )


def parse_gate_stdout(
    raw: bytes, arguments: argparse.Namespace
) -> dict[str, Any]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise RecordError("gate stdout is not canonical line text")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RecordError("gate stdout is not UTF-8") from error
    if not lines or any(not line or any(ord(char) < 32 for char in line)
                        for line in lines):
        raise RecordError("gate stdout contains an invalid line")
    if len(lines) != arguments.pairs + 4:
        raise RecordError("gate stdout has incomplete or extra transcript lines")
    if (
        not lines[0].startswith("runtime_sha256=")
        or not lines[1].startswith("candidate_tree_nodes=")
        or not lines[2].startswith("baseline_tree_nodes=")
        or not lines[-1].startswith("summary ")
    ):
        raise RecordError("gate stdout line ordering is not canonical")
    identity = key_values(lines[0], "runtime identity")
    if set(identity) != {
        "runtime_sha256", "model_sha256", "packed_sha256"
    } or not all(SHA256.fullmatch(value) for value in identity.values()):
        raise RecordError("gate runtime identity is malformed")
    candidate = {
        key.removeprefix("candidate_"): scalar(value)
        for key, value in key_values(lines[1], "candidate profile").items()
    }
    baseline = {
        key.removeprefix("baseline_"): scalar(value)
        for key, value in key_values(lines[2], "baseline profile").items()
    }
    expected_profile_fields = {"tree_nodes", "c", "fpu", "final"}
    if set(candidate) != expected_profile_fields or set(baseline) != expected_profile_fields:
        raise RecordError("gate profile fields are incomplete")
    expected_candidate = _profile(arguments, "candidate")
    expected_baseline = _profile(arguments, "baseline")
    if candidate != expected_candidate or baseline != expected_baseline:
        raise RecordError("gate profile disagrees with the invoked command")

    opening_depths = _opening_depths(arguments.opening_turns)
    pair_results = []
    totals = {
        "candidate": 0, "baseline": 0, "unfinished": 0,
        "candidate_player_one": 0, "candidate_player_two": 0,
    }
    observed_seeds: set[int] = set()
    for expected_pair, line in enumerate(lines[3:-1]):
        match = PAIR.fullmatch(line)
        if match is None:
            raise RecordError(f"gate pair {expected_pair} line is malformed")
        pair, depth, seed, first_player, first_winner, second_player, second_winner = (
            int(value) for value in match.groups()
        )
        expected_order = (0, 1) if pair % 2 == 0 else (1, 0)
        if (
            pair != expected_pair
            or depth != opening_depths[pair % len(opening_depths)]
            or (first_player, second_player) != expected_order
            or seed in observed_seeds
            or not _opening_seed_matches(arguments.seed, pair, seed)
        ):
            raise RecordError(f"gate pair {expected_pair} schedule is stale")
        observed_seeds.add(seed)
        games = []
        for candidate_player, winner in (
            (first_player, first_winner), (second_player, second_winner)
        ):
            if winner == -1:
                totals["unfinished"] += 1
            elif winner == candidate_player:
                totals["candidate"] += 1
                color = (
                    "candidate_player_one" if candidate_player == 0
                    else "candidate_player_two"
                )
                totals[color] += 1
            else:
                totals["baseline"] += 1
            games.append({
                "candidate_player": candidate_player,
                "winner": winner,
            })
        pair_results.append({
            "pair": pair, "opening_turns": depth, "seed": seed,
            "games": games,
        })

    summary = {
        key: scalar(value)
        for key, value in key_values(
            lines[-1][len("summary "):], "summary"
        ).items()
    }
    if set(summary) != SUMMARY_FIELDS:
        raise RecordError("gate summary fields are incomplete or unexpected")
    if (summary.get("runtime_policy") != "same" or
            summary.get("shuffle_seed_policy") != "deployment-constant" or
            summary.get("game_order_policy") != "pair-parity-color-swap" or
            summary.get("timing_scope") != "search-through-apply"):
        raise RecordError("gate summary violates the A/B isolation contract")
    expected_opening_turns = ",".join(map(str, opening_depths))
    expected_summary_config = {
        "profile": f"{arguments.first_ms}/{arguments.later_ms}",
        "pairs": arguments.pairs,
        "maximum_turns": arguments.maximum_turns,
        "opening_turns": expected_opening_turns,
        "opening_seed": arguments.seed,
        "candidate_tree_nodes": arguments.candidate_tree_nodes,
        "candidate_c": arguments.candidate_c,
        "candidate_fpu": arguments.candidate_fpu,
        "candidate_final": arguments.candidate_final,
        "baseline_tree_nodes": arguments.baseline_tree_nodes,
        "baseline_c": arguments.baseline_c,
        "baseline_fpu": arguments.baseline_fpu,
        "baseline_final": arguments.baseline_final,
        "required_total": arguments.minimum_candidate_wins,
        "required_per_color": arguments.minimum_wins_per_color,
    }
    if any(summary.get(key) != value for key, value in expected_summary_config.items()):
        raise RecordError("gate summary configuration disagrees with the command")
    if any(summary.get(key) != value for key, value in totals.items()):
        raise RecordError("gate summary result disagrees with the pair transcript")
    if summary["games"] != arguments.pairs * 2:
        raise RecordError("gate summary game count is stale")

    for side, profile in (("candidate", candidate), ("baseline", baseline)):
        counts = {
            suffix: _require_unsigned(
                summary[f"{side}_{suffix}"], f"{side}_{suffix}"
            )
            for suffix in COUNT_SUFFIXES
        }
        elapsed = _require_nonnegative_float(summary[f"{side}_ms"], f"{side}_ms")
        maxima = [
            _require_nonnegative_float(
                summary[f"{side}_{suffix}"], f"{side}_{suffix}"
            )
            for suffix in ("max_first_ms", "max_later_ms")
        ]
        if (
            counts["max_tree"] > profile["tree_nodes"]
            or any(counts[field] > counts["decisions"] for field in (
                "tree_cap_searches", "final_overrides", "deadline_searches",
                "headroom_failures", "operational_timeouts",
            ))
            or any(maximum > elapsed + 1e-9 for maximum in maxima)
        ):
            raise RecordError(f"gate {side} diagnostics contradict their limits")

    passed = (
        totals["unfinished"] == 0
        and summary["candidate_headroom_failures"] == 0
        and summary["baseline_headroom_failures"] == 0
        and summary["candidate_operational_timeouts"] == 0
        and summary["baseline_operational_timeouts"] == 0
        and totals["candidate"] >= arguments.minimum_candidate_wins
        and totals["candidate_player_one"] >= arguments.minimum_wins_per_color
        and totals["candidate_player_two"] >= arguments.minimum_wins_per_color
    )
    if summary["passed"] is not passed:
        raise RecordError("gate passed flag disagrees with the verified transcript")
    return {
        "identity": identity,
        "candidate_profile": candidate,
        "baseline_profile": baseline,
        "pairs": pair_results,
        "summary": summary,
    }


@contextlib.contextmanager
def serial_clock_lock() -> Iterator[None]:
    path = ROOT / "build/.jacek-native-search-ab-actual-clock.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecordError("another Jacek-native search A/B is running") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def command(arguments: argparse.Namespace) -> list[str]:
    return [
        str(arguments.gate_binary.resolve()),
        "--checkpoint", str(arguments.checkpoint.resolve()),
        "--pairs", str(arguments.pairs),
        "--first-ms", str(arguments.first_ms),
        "--later-ms", str(arguments.later_ms),
        "--maximum-turns", str(arguments.maximum_turns),
        "--opening-turns", arguments.opening_turns,
        "--seed", str(arguments.seed),
        "--candidate-tree-nodes", str(arguments.candidate_tree_nodes),
        "--baseline-tree-nodes", str(arguments.baseline_tree_nodes),
        "--candidate-c", str(arguments.candidate_c),
        "--baseline-c", str(arguments.baseline_c),
        "--candidate-fpu", str(arguments.candidate_fpu),
        "--baseline-fpu", str(arguments.baseline_fpu),
        "--candidate-final", arguments.candidate_final,
        "--baseline-final", arguments.baseline_final,
        "--minimum-candidate-wins", str(arguments.minimum_candidate_wins),
        "--minimum-wins-per-color", str(arguments.minimum_wins_per_color),
    ]


def exclusive_write(path: pathlib.Path, raw: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError as error:
        raise RecordError(f"refusing to overwrite immutable evidence: {path}") from error


def record(arguments: argparse.Namespace) -> pathlib.Path:
    if not arguments.gate_binary.is_file() or not os.access(arguments.gate_binary, os.X_OK):
        raise RecordError("gate binary does not exist or is not executable")
    if not arguments.checkpoint.is_file():
        raise RecordError("runtime checkpoint does not exist")
    invoked = command(arguments)
    with serial_clock_lock():
        completed = subprocess.run(
            invoked, cwd=ROOT, capture_output=True, check=False
        )
    if completed.returncode not in (0, 1) or completed.stderr:
        raise RecordError("gate did not produce a clean complete transcript")
    parsed = parse_gate_stdout(completed.stdout, arguments)
    checkpoint_identity = runtime_identity(arguments.checkpoint.read_bytes())
    if checkpoint_identity != parsed["identity"]:
        raise RecordError("gate stdout runtime identity disagrees with the checkpoint")
    if parsed["summary"].get("passed") != (completed.returncode == 0):
        raise RecordError("gate exit status disagrees with its summary")

    sources = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        sources.append({"path": relative, "sha256": sha256(path.read_bytes())})
    stdout_sha = sha256(completed.stdout)
    payload = {
        "schema": REPORT_SCHEMA,
        "runtime": parsed["identity"],
        "profiles": {
            "candidate": parsed["candidate_profile"],
            "baseline": parsed["baseline_profile"],
        },
        "result": parsed["summary"],
        "pairs": parsed["pairs"],
        "execution": {
            "command": [
                "$GATE_BINARY" if index == 0 else
                "$CHECKPOINT" if index > 0 and invoked[index - 1] == "--checkpoint" else
                value
                for index, value in enumerate(invoked)
            ],
            "exit_code": completed.returncode,
            "gate_binary_sha256": sha256(arguments.gate_binary.read_bytes()),
            "gate_sources": sources,
            "serial_actual_clock_lock": True,
        },
        "stdout": {
            "bytes": len(completed.stdout),
            "path": f"{stdout_sha}.stdout.txt",
            "sha256": stdout_sha,
        },
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = arguments.output_dir / payload["stdout"]["path"]
    report_raw = canonical_json(payload)
    report_path = arguments.output_dir / f"{sha256(report_raw)}.json"
    exclusive_write(stdout_path, completed.stdout)
    try:
        exclusive_write(report_path, report_raw)
    except Exception:
        stdout_path.unlink(missing_ok=True)
        raise
    print(report_path)
    return report_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Record one content-addressed same-runtime search A/B"
    )
    result.add_argument("--gate-binary", type=pathlib.Path, required=True)
    result.add_argument("--checkpoint", type=pathlib.Path, required=True)
    result.add_argument("--output-dir", type=pathlib.Path, required=True)
    result.add_argument("--pairs", type=int, default=64)
    result.add_argument("--first-ms", type=int, default=800)
    result.add_argument("--later-ms", type=int, default=155)
    result.add_argument("--maximum-turns", type=int, default=384)
    result.add_argument("--opening-turns", default="0,4,8,12")
    result.add_argument("--seed", type=int, default=6510615555426900575)
    result.add_argument("--candidate-tree-nodes", type=int, default=80000)
    result.add_argument("--baseline-tree-nodes", type=int, default=80000)
    result.add_argument("--candidate-c", type=float, default=0.95)
    result.add_argument("--baseline-c", type=float, default=0.95)
    result.add_argument("--candidate-fpu", type=float, default=0.5)
    result.add_argument("--baseline-fpu", type=float, default=0.5)
    result.add_argument("--candidate-final", choices=FINAL_FORMULAS,
                        default="value-log-visits")
    result.add_argument("--baseline-final", choices=FINAL_FORMULAS,
                        default="value-log-visits")
    result.add_argument("--minimum-candidate-wins", type=int, default=0)
    result.add_argument("--minimum-wins-per-color", type=int, default=0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        record(arguments)
    except (OSError, RecordError) as error:
        raise SystemExit(f"search A/B recording failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
