#!/usr/bin/env python3
"""Generate the immutable opening banks and preregistered flagship manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.flagship_study import studylib


OPENING_SEEDS = {
    "development": {4: "1004101", 8: "1008101", 12: "1012101", 20: "1020101"},
    "validation": {4: "2004101", 8: "2008101", 12: "2012101", 20: "2020101"},
    "test": {4: "3004101", 8: "3008101", 12: "3012101", 20: "3020101"},
}
PHASE_SEEDS = {
    "bot": {"development": "4100001", "validation": "4200001", "test": "4300001"},
    "bootstrap": {"development": "5100001", "validation": "5200001", "test": "5300001"},
    "analysis": {"development": "7100001", "validation": "7200001", "test": "7300001"},
}
SUPERSEDED_MANIFEST_SHA256 = (
    "eab2728f4f5915926639ab67f20ab94137afe0275543f8eafd34b8047ab4ecf3"
)


def run_text(command: list[str], repository: pathlib.Path) -> str:
    process = subprocess.run(command, cwd=repository, capture_output=True, text=True,
                             check=False)
    if process.returncode != 0:
        raise studylib.StudyError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"{process.stderr.strip()}"
        )
    return process.stdout.strip()


def write_bytes_atomic(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise studylib.StudyError(f"refusing to replace frozen artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def config_mcts(iterations: int) -> dict[str, Any]:
    return {
        "id": f"mcts-{iterations}",
        "family": "mcts",
        "kind": "mcts",
        "public_label": "Tactical MctsBot",
        "role": "candidate",
        "settings": {
            "iterations": iterations,
            "exploration": 1.4142135623730951,
            "rollout_policy": "tactical",
            "leaf_policy": "rollout_only",
            "reuse_tree": True,
            "node_capacity": 65536,
            "quiescence_enabled": False,
            "quiescence_max_depth": 8,
            "quiescence_max_nodes": 256,
            "wall_clock_limit_ms": 0,
            "seed_derivation": "sha256-domain-separated-uint64/v1",
        },
    }


def config_alpha_beta(nodes: int, *, neural: bool) -> dict[str, Any]:
    family = "jacek_inspired" if neural else "alpha_beta"
    kind = "jacek-inspired" if neural else "alpha-beta"
    prefix = "jacek" if neural else "alpha-beta"
    settings: dict[str, Any] = {
        "max_turn_depth": 6,
        "max_nodes": nodes,
        "transposition_table_entries": 65536,
        "max_search_plies": 12,
        "wall_clock_limit_ms": 0,
        "seed_ignored": True,
    }
    if neural:
        settings.update({
            "model_path": "models/jacek_article_value_model.json",
            "model_sha256": "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084",
        })
    return {
        "id": f"{prefix}-{nodes // 1000}k",
        "family": family,
        "kind": kind,
        "public_label": (
            "Neural alpha-beta (JacekInspiredBot)" if neural
            else "Hand-evaluated AlphaBetaBot"
        ),
        "role": "candidate",
        "settings": settings,
    }


def config_rank5() -> dict[str, Any]:
    return {
        "id": "rank5-fixed-50k",
        "family": "rank5_derived",
        "kind": "rank5-derived",
        "public_label": studylib.PUBLIC_RANK5_LABEL,
        "role": "fixed_comparator",
        "settings": {
            "max_turn_depth": 32,
            "max_nodes": 50000,
            "transposition_table_entries": 65536,
            "evaluation_cache_entries": 32768,
            "wall_clock_limit_ms": 0,
            "replay_corrections": False,
            "learned_value_blend_percent": 0,
            "seed_ignored": True,
            "rules_profile": "standard-8x10-demo",
            "original_artifact_sha256": studylib.RANK5_SOURCE_SHA256,
        },
    }


def machine_metadata(repository: pathlib.Path,
                     build_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    def optional(command: list[str], default: str) -> str:
        try:
            value = run_text(command, repository)
        except (OSError, studylib.StudyError):
            return default
        return value or default

    compiler = optional(["c++", "--version"], "unknown").splitlines()
    cmake = optional(["cmake", "--version"], "unknown").splitlines()[0]
    cpu = optional(["sysctl", "-n", "machdep.cpu.brand_string"], platform.processor() or "unknown")
    physical = optional(["sysctl", "-n", "hw.physicalcpu"], str(os.cpu_count() or 1))
    logical = optional(["sysctl", "-n", "hw.logicalcpu"], str(os.cpu_count() or 1))
    memory = optional(["sysctl", "-n", "hw.memsize"], "1")
    power = optional(["pmset", "-g", "batt"], "power state unavailable").replace("\n", "; ")
    os_name = optional(["sw_vers", "-productName"], platform.system())
    os_version = optional(["sw_vers", "-productVersion"], platform.release())
    metadata = {
        "os": f"{os_name} {os_version}",
        "kernel": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "physical_cores": int(physical),
        "logical_cores": int(logical),
        "memory_bytes": int(memory),
        "compiler": "AppleClang" if compiler and "Apple clang" in compiler[0] else "C++ compiler",
        "compiler_version": compiler[0] if compiler else "unknown",
        "cmake_version": cmake,
        "python_version": platform.python_version(),
        "build_flags": "Release: -O3 -DNDEBUG -std=c++20 -arch arm64 -Wall -Wextra -Wpedantic",
        "machine_id": "apple-m4-pro-local-gate",
        "power_measurement": power,
    }
    if build_provenance is not None:
        metadata["compiler"] = str(build_provenance["compiler_id"])
        metadata["compiler_version"] = str(build_provenance["compiler_version"])
        metadata["build_flags"] = str(build_provenance["configured_flags"])
    return metadata


def validate_release_build_provenance(
        value: Any, source_commit: str, *, returncode: int = 0) -> dict[str, Any]:
    provenance = studylib.validate_arena_build_provenance(value)
    if returncode != 0 or \
       provenance["schema"] != "papersoccer.arena-build.v1" or \
       provenance["runtime"] != "native" or \
       provenance["build_type"] != "Release" or \
       provenance["ndebug"] is not True or \
       provenance["sanitizers_enabled"] is not False or \
       provenance["source_commit"] != source_commit or \
       provenance["source_dirty"] is not False:
        raise studylib.StudyError(
            "manifest generation requires the optimized native Release arena"
        )
    return provenance


def build_manifest(repository: pathlib.Path, source_commit: str,
                   banks: list[dict[str, Any]], preregistered_at: str,
                   build_provenance: dict[str, Any] | None = None,
                   arena_sha256: str = "0" * 64,
                   opening_tool_sha256: str = "0" * 64) -> dict[str, Any]:
    configurations = (
        [config_mcts(iterations) for iterations in (1000, 2000, 4000)] +
        [config_alpha_beta(nodes, neural=False) for nodes in (20000, 50000, 100000)] +
        [config_alpha_beta(nodes, neural=True) for nodes in (20000, 50000, 100000)] +
        [config_rank5()]
    )
    tuning_ids = [config["id"] for config in configurations if config["role"] == "candidate"]
    return {
        "schema_version": studylib.MANIFEST_SCHEMA_VERSION,
        "study": {
            "id": "competitive-demo-bots-flagship-2026-v2",
            "version": "1.1.0",
            "title": "Competitive demo-rule Paper Soccer bot study",
            "study_class": "flagship",
            "preregistered_at_utc": preregistered_at,
            "frozen": True,
            "public_labels": {
                "mcts": "Tactical MctsBot",
                "alpha_beta": "Hand-evaluated AlphaBetaBot",
                "jacek_inspired": "Neural alpha-beta (JacekInspiredBot)",
                "rank5_derived": studylib.PUBLIC_RANK5_LABEL,
            },
            "rank5_disclaimer": studylib.RANK5_DISCLAIMER,
        },
        "source": {
            "git_commit": source_commit,
            "dirty_worktree": False,
            "analysis_contract_path": "benchmarks/flagship_study/analysis_contract.md",
            "analysis_contract_sha256": studylib.sha256_file(
                repository / "benchmarks/flagship_study/analysis_contract.md"
            ),
            "arena_sha256": arena_sha256,
            "opening_tool_sha256": opening_tool_sha256,
            "protected_artifacts": {
                "rank5_submission_path": "submissions/codingame/bots/rank_5/submission.cpp",
                "rank5_submission_sha256": studylib.RANK5_SOURCE_SHA256,
                "jacek_model_path": "models/jacek_article_value_model.json",
                "jacek_model_sha256": "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084",
            },
        },
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "opponent_goal_only",
            "blocked_rule": "player_to_move_loses",
            "playable_edges": 316,
            "max_game_plies": 512,
            "maximum_game_length_policy": (
                "Stop at 512 total physical plies; any nonterminal stop is a truncation "
                "and invalidates strength/calibration publication."
            ),
            "opening_ply_definition": (
                "One physical legal edge application, including same-player rebound edges; "
                "opening plies count toward the 512-ply safety limit."
            ),
            "natural_draws": False,
        },
        "configurations": configurations,
        "candidate_grids": {
            "mcts": ["mcts-1000", "mcts-2000", "mcts-4000"],
            "alpha_beta": ["alpha-beta-20k", "alpha-beta-50k", "alpha-beta-100k"],
            "jacek_inspired": ["jacek-20k", "jacek-50k", "jacek-100k"],
            "rank5_derived": ["rank5-fixed-50k"],
        },
        "openings": {
            "generator": {
                "id": "uniform-legal-move-generator/v1",
                "description": "Separate deterministic uniform legal-edge data-generation mechanism",
                "selection": "SplitMix64 with unbiased rejection-sampled legal-edge index",
                "terminal_rejection": "Reject and advance deterministic candidate seed stream",
                "duplicate_policy": "Reject duplicate canonical game states across all phases",
                "canonical_equivalence": "Horizontal reflection about x=width/2",
                "state_hash_algorithm": "sha256-canonical-game-state/v1",
            },
            "depths": [4, 8, 12, 20],
            "banks": banks,
        },
        "seeds": {
            "opening": {phase: {str(depth): seed for depth, seed in depth_map.items()}
                        for phase, depth_map in OPENING_SEEDS.items()},
            **PHASE_SEEDS,
            "calibration": {"validation": "6100001"},
        },
        "samples": {
            phase: {"color_swapped_pairs_per_depth_matchup": pairs, "games_per_pair": 2}
            for phase, pairs in studylib.EXPECTED_PAIR_COUNTS.items()
        },
        "schedule": {
            "tuning": [
                {"id": f"tune-{identifier}-vs-rank5", "candidate": identifier,
                 "opponent": "rank5-fixed-50k", "phases": ["development", "validation"]}
                for identifier in tuning_ids
            ],
            "test": [
                {"id": "test-mcts-vs-alpha-beta", "left_slot": "selected:mcts",
                 "right_slot": "selected:alpha_beta"},
                {"id": "test-mcts-vs-jacek", "left_slot": "selected:mcts",
                 "right_slot": "selected:jacek_inspired"},
                {"id": "test-mcts-vs-rank5", "left_slot": "selected:mcts",
                 "right_slot": "fixed:rank5_derived"},
                {"id": "test-alpha-beta-vs-jacek", "left_slot": "selected:alpha_beta",
                 "right_slot": "selected:jacek_inspired"},
                {"id": "test-alpha-beta-vs-rank5", "left_slot": "selected:alpha_beta",
                 "right_slot": "fixed:rank5_derived"},
                {"id": "test-jacek-vs-rank5", "left_slot": "selected:jacek_inspired",
                 "right_slot": "fixed:rank5_derived"},
            ],
        },
        "latency_protocol": {
            "runtime": "native",
            "build_type": "Release",
            "single_threaded": True,
            "warmup": "Eight untimed deterministic decisions per entrant in each arena process",
            "timer_boundary": "steady_clock immediately around Bot::choose_move(state)",
            "state_copying": "Caller passes const state by reference; bot-internal setup/copying is timed",
            "quantiles": ["median", "p90", "p95", "p99", "maximum"],
            "gate_ms": 50,
            "rank5_gate_distribution": "fresh_root_only",
            "measurement_phase": "validation",
            "power_conditions": (
                "Gate machine on AC power, Low Power Mode disabled, nominal thermal state, "
                "single foreground arena process; observed state recorded with results"
            ),
        },
        "selection_rule": {
            "phase": "validation",
            "strength_metric": (
                "mean depth-stratified color-swapped pair score versus "
                f"{studylib.PUBLIC_RANK5_LABEL}"
            ),
            "practical_tie_percentage_points": 1.0,
            "tie_break_order": ["lower_p95_latency", "smaller_budget", "stable_config_id"],
            "no_eligible_family_policy": "stop_before_test",
            "rank5_policy": (
                "Fixed in test regardless of gate; exclude from constrained/Pareto-best claims "
                "when fresh-root p95 exceeds 50 ms"
            ),
        },
        "statistics": {
            "bootstrap": {"resamples": 10000, "confidence": 0.95,
                          "unit": "color_swapped_pair", "stratify_by": "opening_depth"},
            "pair_score": {"two_wins": 1.0, "split": 0.5, "two_losses": 0.0},
            "truncations": "reject_strength_and_calibration",
            "bradley_terry": {
                "identifiability": "sum_to_zero",
                "outcomes": "binary_games_only",
                "bootstrap_unit": "color_swapped_pair_stratified_by_matchup_and_opening_depth",
                "separation_policy": "fail_without_finite_estimate",
                "convergence_policy": "fail_without_finite_estimate",
                "minimum_bootstrap_success_fraction": 1.0,
            },
            "calibration": {
                "fit_phase": "validation",
                "link": "logistic_with_validation_score_standardization",
                "bins": 10,
                "outcome_perspective": "eventual binary winner from prediction player-to-move perspective",
                "rank5_predictions": "fresh_root_only",
                "test_metrics": [
                    "brier_score", "log_loss", "ten_bin_reliability",
                    "pair_clustered_95_intervals",
                ],
                "uncertainty_method": "pair_cluster_percentile_bootstrap",
                "bootstrap_resamples": 10_000,
                "bootstrap_unit": "color_swapped_pair",
                "bootstrap_stratify_by": ["matchup", "opening_depth"],
                "bootstrap_seed_source": "derived_from_seeds.analysis.test_per_bot",
                "minimum_bin_successful_resamples": 1_000,
            },
            "pareto": {"strength_source": ["development", "validation"],
                       "latency_source": "validation", "maximize": "paired_strength",
                       "minimize": "p95_latency"},
            "ablations": {
                "phases": ["development", "validation"],
                "practical_gain_threshold": 0.01,
                "comparison_unit": "aligned_color_swapped_opening_pair",
                "bootstrap_method": "paired_difference_percentile",
                "bootstrap_resamples": 10_000,
                "stratify_by": "opening_depth",
                "comparisons": {
                    "mcts": [
                        ["mcts-1000", "mcts-2000"],
                        ["mcts-2000", "mcts-4000"],
                        ["mcts-1000", "mcts-4000"],
                    ],
                    "alpha_beta": [
                        ["alpha-beta-20k", "alpha-beta-50k"],
                        ["alpha-beta-50k", "alpha-beta-100k"],
                        ["alpha-beta-20k", "alpha-beta-100k"],
                    ],
                    "jacek_inspired": [
                        ["jacek-20k", "jacek-50k"],
                        ["jacek-50k", "jacek-100k"],
                        ["jacek-20k", "jacek-100k"],
                    ],
                    "equal_budget_evaluator": [
                        ["alpha-beta-20k", "jacek-20k"],
                        ["alpha-beta-50k", "jacek-50k"],
                        ["alpha-beta-100k", "jacek-100k"],
                    ],
                },
                "scaling_classification": {
                    "supported_practical_gain": "interval_lower_gt_plus_0.01",
                    "supported_regression": "interval_upper_lt_0",
                    "supported_no_practical_gain": "interval_upper_lt_plus_0.01",
                    "unresolved_at_1pp": "otherwise",
                },
                "evaluator_classification": {
                    "neural_materially_stronger": "interval_lower_gt_plus_0.01",
                    "hand_materially_stronger": "interval_upper_lt_minus_0.01",
                    "practical_equivalence_supported":
                        "interval_within_minus_0.01_plus_0.01",
                    "unresolved_at_1pp": "otherwise",
                },
            },
            "claim_rule": (
                "A is stronger than B only when A's paired test 95% interval lower bound "
                "is strictly greater than 0.50; otherwise statistically unresolved."
            ),
        },
        "outputs": {
            "raw_results_root": "results/flagship_study",
            "curated_root": "benchmarks/flagship_study",
            "selection_lock": "benchmarks/flagship_study/selection_lock.json",
            "curated_data": {
                "development": "benchmarks/flagship_study/data/development.json",
                "validation": "benchmarks/flagship_study/data/validation.json",
                "test": "benchmarks/flagship_study/data/test.json",
            },
            "charts": {
                "bradley_terry": "benchmarks/flagship_study/charts/test_bradley_terry.svg",
                "pareto": "benchmarks/flagship_study/charts/validation_pareto.svg",
                "calibration": "benchmarks/flagship_study/charts/test_calibration.svg",
            },
            "report": "benchmarks/flagship_study/REPORT.md",
            "runtime_projection": "benchmarks/flagship_study/runtime_projection.json",
        },
        "environment": machine_metadata(repository, build_provenance),
    }


def generate_banks(repository: pathlib.Path, opening_tool: pathlib.Path) -> list[dict[str, Any]]:
    opening_directory = repository / "benchmarks/flagship_study/openings"
    generated_paths: list[pathlib.Path] = []
    manifests: list[dict[str, Any]] = []
    for phase in studylib.FULL_PHASES:
        for depth in studylib.EXPECTED_OPENING_DEPTHS:
            pairs = studylib.EXPECTED_PAIR_COUNTS[phase]
            path = opening_directory / f"{phase}_d{depth:02d}.tsv"
            command = [
                str(opening_tool), "generate", "--phase", phase,
                "--depth", str(depth), "--pairs", str(pairs),
                "--seed", OPENING_SEEDS[phase][depth],
            ]
            for excluded in generated_paths:
                command += ["--exclude-bank", str(excluded)]
            process = subprocess.run(command, cwd=repository, capture_output=True, check=False)
            if process.returncode != 0:
                raise studylib.StudyError(
                    f"opening generator failed for {phase}/{depth}:\n"
                    f"{process.stderr.decode('utf-8', errors='replace').strip()}"
                )
            write_bytes_atomic(path, process.stdout)
            records = studylib.parse_opening_bank(path)
            if len(records) != pairs:
                raise studylib.StudyError(f"opening generator produced wrong pair count: {path}")
            generated_paths.append(path)
            manifests.append({
                "id": f"openings-{phase}-d{depth:02d}",
                "phase": phase,
                "depth": depth,
                "pairs": pairs,
                "path": str(path.relative_to(repository)),
                "sha256": studylib.sha256_file(path),
                "seed": OPENING_SEEDS[phase][depth],
            })
    return manifests


def reuse_frozen_banks(repository: pathlib.Path) -> list[dict[str, Any]]:
    """Load the committed banks without regenerating a single opening."""

    superseded_path = (
        repository / "benchmarks/flagship_study/superseded" /
        "manifest-eab2728f.json"
    )
    if not superseded_path.is_file() or \
       studylib.sha256_file(superseded_path) != SUPERSEDED_MANIFEST_SHA256:
        raise studylib.StudyError(
            "audited superseded-manifest identity is missing or changed"
        )
    superseded = studylib.load_json(superseded_path)
    previous_banks = {
        (bank["phase"], bank["depth"]): bank
        for bank in superseded["openings"]["banks"]
    }
    if set(previous_banks) != {
        (phase, depth)
        for phase in studylib.FULL_PHASES
        for depth in studylib.EXPECTED_OPENING_DEPTHS
    }:
        raise studylib.StudyError("superseded manifest has the wrong opening design")

    manifests: list[dict[str, Any]] = []
    for phase in studylib.FULL_PHASES:
        for depth in studylib.EXPECTED_OPENING_DEPTHS:
            pairs = studylib.EXPECTED_PAIR_COUNTS[phase]
            path = (
                repository / "benchmarks/flagship_study/openings" /
                f"{phase}_d{depth:02d}.tsv"
            )
            if not path.is_file():
                raise studylib.StudyError(f"frozen opening bank is missing: {path}")
            records = studylib.parse_opening_bank(path)
            metadata = studylib.opening_bank_metadata(path)
            if len(records) != pairs or metadata.get("phase") != phase or \
               metadata.get("depth") != str(depth) or \
               metadata.get("pairs") != str(pairs) or \
               metadata.get("generator_seed") != OPENING_SEEDS[phase][depth]:
                raise studylib.StudyError(
                    f"frozen opening bank differs from the preregistered design: {path}"
                )
            previous = previous_banks[(phase, depth)]
            if previous.get("path") != str(path.relative_to(repository)) or \
               previous.get("seed") != OPENING_SEEDS[phase][depth] or \
               previous.get("pairs") != pairs or \
               previous.get("sha256") != studylib.sha256_file(path):
                raise studylib.StudyError(
                    f"frozen opening bank changed after the pre-test audit: {path}"
                )
            manifests.append({
                "id": f"openings-{phase}-d{depth:02d}",
                "phase": phase,
                "depth": depth,
                "pairs": pairs,
                "path": str(path.relative_to(repository)),
                "sha256": previous["sha256"],
                "seed": OPENING_SEEDS[phase][depth],
            })
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-tool", type=pathlib.Path, required=True)
    parser.add_argument("--source-commit", required=True,
                        help="full framework commit used for execution")
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/flagship_study/manifest.json"))
    parser.add_argument("--preregistered-at-utc",
                        default=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat())
    parser.add_argument(
        "--reuse-frozen-openings", action="store_true",
        help=(
            "reuse and verify the existing committed banks byte-for-byte; "
            "intended only for an audited pre-test manifest supersession"
        ),
    )
    args = parser.parse_args()
    repository = pathlib.Path(__file__).resolve().parents[2]
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise studylib.StudyError("--source-commit must be a full lowercase commit hash")
    dirty = run_text(["git", "status", "--porcelain", "--untracked-files=all"], repository)
    if dirty:
        raise studylib.StudyError(
            "manifest generation requires a clean framework commit; commit the tooling first"
        )
    current_commit = run_text(["git", "rev-parse", "HEAD"], repository)
    if current_commit != args.source_commit:
        raise studylib.StudyError(
            "--source-commit must equal the clean framework commit currently checked out"
        )
    opening_tool = args.opening_tool
    if not opening_tool.is_absolute():
        opening_tool = repository / opening_tool
    if not opening_tool.is_file() or not os.access(opening_tool, os.X_OK):
        raise studylib.StudyError(f"opening tool is not executable: {opening_tool}")
    arena_path = opening_tool.with_name("papersoccer_arena")
    if not arena_path.is_file() or not os.access(arena_path, os.X_OK):
        raise studylib.StudyError(
            f"native arena must be built beside the opening tool: {arena_path}"
        )
    provenance_process = subprocess.run(
        [str(arena_path), "provenance"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    try:
        build_provenance = validate_release_build_provenance(
            json.loads(provenance_process.stdout), args.source_commit,
            returncode=provenance_process.returncode,
        )
    except (json.JSONDecodeError, studylib.StudyError) as error:
        raise studylib.StudyError("arena returned invalid build provenance") from error
    banks = (
        reuse_frozen_banks(repository)
        if args.reuse_frozen_openings else generate_banks(repository, opening_tool)
    )
    manifest = build_manifest(
        repository, args.source_commit, banks, args.preregistered_at_utc,
        build_provenance, studylib.sha256_file(arena_path),
        studylib.sha256_file(opening_tool),
    )
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repository / manifest_path
    studylib.validate_manifest(manifest, repository, verify_files=True)
    studylib.verify_opening_phase_disjointness(manifest, repository)
    write_bytes_atomic(manifest_path, studylib.canonical_json_bytes(manifest))
    print(json.dumps({
        "manifest": str(manifest_path.relative_to(repository)),
        "manifest_sha256": studylib.sha256_file(manifest_path),
        "opening_banks": len(banks),
        "opening_records": sum(bank["pairs"] for bank in banks),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except studylib.StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
