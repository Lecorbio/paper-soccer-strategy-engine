#!/usr/bin/env python3
"""Run the tiny manifest-driven development tournament and resume it safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.flagship_study import studylib


EXPECTED_CONFIGURATIONS = {
    "smoke-mcts-8": ("mcts", "mcts"),
    "smoke-alpha-beta-256": ("alpha_beta", "alpha-beta"),
}
EXPECTED_MATCHUP = "smoke-mcts-vs-alpha-beta"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise studylib.StudyError(f"CI smoke verification failed: {message}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _assert_tiny_scope(manifest: Mapping[str, Any],
                       repository: pathlib.Path) -> None:
    _require(manifest["study"]["study_class"] == "ci_smoke",
             "manifest is not marked ci_smoke")
    configurations = {
        config["id"]: config for config in manifest["configurations"]
    }
    _require(set(configurations) == set(EXPECTED_CONFIGURATIONS),
             "manifest must contain exactly the two smoke configurations")
    for identifier, (family, kind) in EXPECTED_CONFIGURATIONS.items():
        config = configurations[identifier]
        _require(config["family"] == family and config["kind"] == kind,
                 f"unexpected family or kind for {identifier}")

    mcts = configurations["smoke-mcts-8"]["settings"]
    alpha_beta = configurations["smoke-alpha-beta-256"]["settings"]
    _require(mcts["iterations"] == 8 and mcts["leaf_policy"] == "rollout_only",
             "MCTS smoke profile must remain at eight rollout-only iterations")
    _require(alpha_beta["max_turn_depth"] == 2 and
             alpha_beta["max_nodes"] == 256,
             "alpha-beta smoke profile must remain at depth two and 256 nodes")
    _require(manifest["openings"]["depths"] == [4],
             "smoke manifest must use only four-ply openings")
    _require(len(manifest["openings"]["banks"]) == 3 and
             all(bank["pairs"] == 1 for bank in manifest["openings"]["banks"]),
             "smoke manifest must contain one pair in each phase bank")
    _require(manifest["samples"] == {
        phase: {
            "color_swapped_pairs_per_depth_matchup": 1,
            "games_per_pair": 2,
        }
        for phase in studylib.FULL_PHASES
    }, "smoke sample counts changed")
    _require(manifest["schedule"]["test"] == [],
             "smoke manifest must not schedule test matchups")
    tuning = manifest["schedule"]["tuning"]
    _require(len(tuning) == 1 and tuning[0]["id"] == EXPECTED_MATCHUP,
             "smoke manifest must contain exactly one tuning matchup")

    allowed_root = (repository / "results" / "flagship_study_smoke").resolve()
    outputs = manifest["outputs"]
    output_values = [
        outputs["raw_results_root"], outputs["curated_root"],
        outputs["selection_lock"], outputs["report"],
        outputs["runtime_projection"],
        *outputs["curated_data"].values(), *outputs["charts"].values(),
    ]
    for value in output_values:
        resolved = (repository / value).resolve()
        _require(resolved == allowed_root or allowed_root in resolved.parents,
                 f"output escapes ignored smoke results: {value}")


def _snapshot(path: pathlib.Path) -> tuple[Any, ...]:
    if not path.exists():
        return ("missing",)
    _require(not path.is_symlink(), f"output snapshot target is a symlink: {path}")
    if path.is_file():
        return ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    entries: list[tuple[str, str, str | None]] = []
    for entry in sorted(path.rglob("*")):
        _require(not entry.is_symlink(), f"output snapshot entry is a symlink: {entry}")
        relative = str(entry.relative_to(path))
        if entry.is_dir():
            entries.append((relative, "directory", None))
        else:
            entries.append((relative, "file",
                            hashlib.sha256(entry.read_bytes()).hexdigest()))
    return ("directory", *entries)


def _protected_phase_outputs(manifest: Mapping[str, Any],
                             repository: pathlib.Path,
                             manifest_hash: str) -> list[pathlib.Path]:
    outputs = manifest["outputs"]
    raw_run_root = (repository / outputs["raw_results_root"] / manifest_hash)
    return [
        raw_run_root / "validation",
        raw_run_root / "test",
        repository / outputs["curated_data"]["validation"],
        repository / outputs["curated_data"]["test"],
        repository / outputs["selection_lock"],
        repository / outputs["report"],
        repository / outputs["runtime_projection"],
        *(repository / path for path in outputs["charts"].values()),
    ]


def _reset_development_outputs(manifest: Mapping[str, Any],
                               repository: pathlib.Path,
                               manifest_hash: str) -> None:
    """Reset only the CI smoke development namespace for repeatable CTest runs."""

    allowed_root = (repository / "results" / "flagship_study_smoke").resolve()
    targets = [
        repository / manifest["outputs"]["raw_results_root"] / manifest_hash /
        "development",
        repository / manifest["outputs"]["curated_data"]["development"],
    ]
    for target in targets:
        resolved = target.resolve()
        _require(allowed_root in resolved.parents,
                 f"development reset escapes smoke output root: {target}")
        _require(not target.is_symlink(),
                 f"development reset target is a symlink: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            _require(target.is_file(),
                     f"development reset target has unsupported type: {target}")
            target.unlink()


def _verify_raw_report(report: Mapping[str, Any], unit: studylib.StudyUnit,
                       opening: studylib.OpeningRecord) -> tuple[str, set[str]]:
    _require(report.get("schema") == "papersoccer.arena.v1",
             "raw report has the wrong arena schema")
    summary = _mapping(report.get("summary"), "raw summary")
    _require(summary.get("truncations") == 0 and
             summary.get("illegal_moves") == 0,
             "raw report contains a truncation or illegal move")
    openings = _array(report.get("openings"), "raw openings")
    _require(len(openings) == 1, "raw report must contain one opening")
    actual_opening = _mapping(openings[0], "raw opening")
    _require(actual_opening.get("opening_id") == opening.opening_id and
             actual_opening.get("state_hash") == opening.state_hash and
             actual_opening.get("actual_plies") == 4,
             "raw report did not preserve the frozen opening identity")

    pair_id = f"development:{unit.matchup_id}:{opening.opening_id}"
    games = _array(report.get("games"), "raw games")
    _require(len(games) == 2, "raw report must contain exactly two games")
    expected_game_ids = {f"{pair_id}:g0", f"{pair_id}:g1"}
    actual_game_ids: set[str] = set()
    saw_mcts = False
    saw_alpha_beta = False
    for game in games:
        game = _mapping(game, "raw game")
        outcome = _mapping(game.get("outcome"), "game outcome")
        _require(outcome.get("truncated") is False and
                 outcome.get("reason") == "terminal" and
                 outcome.get("winner") in ("candidate", "reference"),
                 "each smoke game must end with a decisive terminal winner")
        identifiers = _mapping(game.get("study_ids"), "game study IDs")
        _require(identifiers.get("opening") == opening.opening_id and
                 identifiers.get("pair") == pair_id,
                 "game IDs are not derived from the frozen opening")
        actual_game_ids.add(str(identifiers.get("game")))
        for decision in _array(game.get("decisions"), "game decisions"):
            decision = _mapping(decision, "game decision")
            _require(decision.get("legal") is True,
                     "arena emitted an unvalidated edge")
            if decision.get("bot") == "candidate":
                saw_mcts = True
                _require(isinstance(decision.get("mcts"), dict) and
                         decision.get("alpha_beta") is None,
                         "candidate decision lacks MCTS diagnostics")
            elif decision.get("bot") == "reference":
                saw_alpha_beta = True
                _require(isinstance(decision.get("alpha_beta"), dict) and
                         decision.get("mcts") is None,
                         "reference decision lacks alpha-beta diagnostics")
            else:
                _require(False, "decision identifies an unknown entrant")
            _require(decision.get("rank5_derived") is None,
                     "non-Rank5 smoke decision contains Rank5 diagnostics")
    _require(actual_game_ids == expected_game_ids,
             "game IDs are not stable and complete")
    _require(saw_mcts and saw_alpha_beta,
             "both smoke engines must expose decision diagnostics")
    return pair_id, actual_game_ids


def _verify_curated(curated: Mapping[str, Any], pair_id: str,
                    game_ids: set[str]) -> None:
    completeness = _mapping(curated.get("completeness"), "curated completeness")
    _require(completeness.get("expected_units") == 1 and
             completeness.get("completed_units") == 1,
             "curated output must contain exactly one unit")
    _require(completeness.get("expected_games") == 2 and
             completeness.get("completed_games") == 2 and
             completeness.get("unique_game_ids") == 2,
             "curated output must contain exactly two unique games")
    _require(completeness.get("truncations") == 0 and
             completeness.get("operationally_valid") is True,
             "curated output is not operationally valid")

    binary_games = _array(curated.get("binary_games"), "curated binary games")
    _require(len(binary_games) == 2 and
             {game["pair_id"] for game in binary_games} == {pair_id} and
             {game["game_id"] for game in binary_games} == game_ids,
             "curated pair/game IDs are not stable")
    _require(all(game["winner_config_id"] in EXPECTED_CONFIGURATIONS and
                 game["loser_config_id"] in EXPECTED_CONFIGURATIONS
                 for game in binary_games),
             "curated smoke games are not decisive between the two engines")

    matchups = _mapping(curated.get("matchups"), "curated matchups")
    _require(set(matchups) == {EXPECTED_MATCHUP},
             "curated output contains an unexpected matchup")
    matchup = _mapping(matchups[EXPECTED_MATCHUP], "curated matchup")
    _require(matchup.get("pairs") == 1 and matchup.get("games") == 2 and
             matchup.get("truncations") == 0,
             "curated matchup must be one complete pair without truncation")

    configurations = _mapping(curated.get("configurations"),
                              "curated configurations")
    _require(set(configurations) == set(EXPECTED_CONFIGURATIONS),
             "curated diagnostics omit a smoke configuration")
    mcts = _mapping(configurations["smoke-mcts-8"]["diagnostics"],
                    "MCTS diagnostics")
    alpha_beta = _mapping(
        configurations["smoke-alpha-beta-256"]["diagnostics"],
        "alpha-beta diagnostics",
    )
    _require(mcts.get("searches", 0) > 0 and mcts.get("iterations", 0) > 0 and
             mcts.get("nodes", 0) > 0,
             "curated MCTS diagnostics are empty")
    _require(alpha_beta.get("searches", 0) > 0 and
             alpha_beta.get("nodes", 0) > 0 and
             alpha_beta.get("max_attempted_turn_depth", 0) > 0,
             "curated alpha-beta diagnostics are empty")
    observations = _mapping(
        curated.get("calibration_observations"),
        "curated calibration observations",
    )
    _require(set(observations) == set(EXPECTED_CONFIGURATIONS) and
             all(_array(_mapping(observations[identifier], identifier).get("scores"),
                        f"{identifier} scores")
                 for identifier in EXPECTED_CONFIGURATIONS),
             "curated data omits reproducible prediction observations")
    _require(all(
        observations[identifier].get("phase") == "development" and
        observations[identifier].get("bot_id") == identifier and
        len(observations[identifier]["scores"]) ==
        len(observations[identifier]["outcomes"]) ==
        len(observations[identifier]["pair_cluster_ids"]) ==
        len(observations[identifier]["stratum_ids"]) and
        observations[identifier]["decision_count"] ==
        len(observations[identifier]["scores"]) +
        sum(observations[identifier]["excluded"].values())
        for identifier in EXPECTED_CONFIGURATIONS
    ), "curated prediction observations have unstable phase/configuration IDs")


def run_smoke(manifest_path: pathlib.Path, arena_path: pathlib.Path) -> dict[str, Any]:
    repository = studylib.repository_root_from_manifest(manifest_path)
    manifest = studylib.validate_manifest(
        studylib.load_json(manifest_path), repository, verify_files=True
    )
    studylib.verify_opening_phase_disjointness(manifest, repository)
    _assert_tiny_scope(manifest, repository)
    _require(arena_path.is_file(), f"arena executable does not exist: {arena_path}")

    manifest_hash = studylib.manifest_sha256(manifest_path)
    _reset_development_outputs(manifest, repository, manifest_hash)
    protected_paths = _protected_phase_outputs(
        manifest, repository, manifest_hash
    )
    untouched_before = {path: _snapshot(path) for path in protected_paths}

    first = studylib.run_phase(manifest_path, arena_path, "development")
    _require(first["phase"] == "development" and
             first["units_assigned"] == 1 and
             first["units_completed"] + first["units_resumed"] == 1,
             "first development run did not account for its only unit")
    aggregate = studylib.aggregate_phase(manifest_path, "development")
    _require(aggregate["units"] == 1 and aggregate["games"] == 2 and
             aggregate["truncations"] == 0,
             "development aggregation did not produce one decisive pair")

    units = studylib.units_for_phase(manifest, "development")
    _require(len(units) == 1, "development schedule expanded beyond one unit")
    unit = units[0]
    opening = studylib.parse_opening_bank(repository / unit.bank_path)[0]
    raw_path = (repository / manifest["outputs"]["raw_results_root"] /
                manifest_hash / "development" / "shards" /
                f"{unit.unit_id}.json")
    raw_before_resume = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    second = studylib.run_phase(manifest_path, arena_path, "development")
    _require(second["run_id"] == first["run_id"] and
             second["units_assigned"] == 1 and
             second["units_completed"] == 0 and
             second["units_resumed"] == 1,
             "second development run did not safely resume the completed unit")
    _require(hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before_resume,
             "resumption changed the completed raw shard")

    report = _mapping(studylib.load_json(raw_path), "raw report")
    pair_id, game_ids = _verify_raw_report(report, unit, opening)
    curated_path = repository / manifest["outputs"]["curated_data"]["development"]
    curated = _mapping(studylib.load_json(curated_path), "curated output")
    _verify_curated(curated, pair_id, game_ids)

    _require(studylib.manifest_sha256(manifest_path) == manifest_hash,
             "smoke execution changed the frozen manifest")
    untouched_after = {path: _snapshot(path) for path in protected_paths}
    _require(untouched_after == untouched_before,
             "smoke execution touched validation, test, or publication outputs")
    return {
        "valid": True,
        "phase": "development",
        "manifest_sha256": manifest_hash,
        "raw_sha256": raw_before_resume,
        "curated_sha256": studylib.sha256_file(curated_path),
        "games": 2,
        "pairs": 1,
        "truncations": 0,
        "resumed_units": second["units_resumed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    repository = studylib.repository_root_from_manifest(manifest_path)
    arena_path = args.arena
    if not arena_path.is_absolute():
        arena_path = (repository / arena_path).resolve()
    result = run_smoke(manifest_path, arena_path)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, studylib.StudyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
