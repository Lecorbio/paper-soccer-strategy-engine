from __future__ import annotations

import copy
import re
import unittest

from benchmarks.flagship_study import studylib
from benchmarks.flagship_study.prepare_manifest import (
    config_alpha_beta,
    config_mcts,
    config_rank5,
)
from benchmarks.flagship_study.report import ReportError, render_report


HASH = "a" * 64


def _completeness(*, units: int, games: int) -> dict[str, object]:
    return {
        "expected_units": units,
        "completed_units": units,
        "expected_games": games,
        "completed_games": games,
        "unique_game_ids": games,
        "decisions": games * 10,
        "truncations": 0,
        "operationally_valid": True,
    }


def _pair_summary(left: str, right: str) -> dict[str, object]:
    return {
        "left_config_id": left,
        "right_config_id": right,
        "games": 2,
        "left_wins": 1,
        "left_losses": 1,
        "right_wins": 1,
        "right_losses": 1,
        "truncations": 0,
        "pairs": 1,
        "pairs_won_2_0": 0,
        "pairs_split_1_1": 1,
        "pairs_lost_0_2": 0,
        "mean_pair_score": 0.5,
        "pair_bootstrap_95": {
            "method": "depth_stratified_pair_percentile",
            "seed": "1",
            "resamples": 10_000,
            "confidence": 0.95,
            "lower": 0.3,
            "upper": 0.7,
        },
    }


def _calibration_metrics() -> dict[str, object]:
    bins = []
    for index in range(10):
        prediction = (index + 0.5) / 10.0
        bins.append({
            "bin": index,
            "lower": index / 10.0,
            "upper": (index + 1) / 10.0,
            "upper_inclusive": index == 9,
            "count": 10,
            "pair_clusters": 5,
            "mean_prediction": prediction,
            "observed_frequency": prediction,
            "bootstrap_successful_resamples": 10_000,
            "observed_frequency_pair_bootstrap_95": {
                "method": "pair_cluster_percentile_stratified",
                "confidence": 0.95,
                "resamples": 10_000,
                "successful_resamples": 10_000,
                "lower": max(0.0, prediction - 0.1),
                "upper": min(1.0, prediction + 0.1),
            },
        })
    return {
        "samples": 100,
        "pair_clusters": 50,
        "brier_score": 0.2,
        "log_loss": 0.6,
        "pair_cluster_bootstrap_95": {
            "method": "pair_cluster_percentile_stratified",
            "seed": "17",
            "resamples": 10_000,
            "successful_resamples": 10_000,
            "confidence": 0.95,
            "stratify_by": "matchup_and_opening_depth",
            "brier_score": {"lower": 0.18, "upper": 0.22},
            "log_loss": {"lower": 0.56, "upper": 0.64},
        },
        "reliability_bins": bins,
    }


def _ablation_comparison(lower_id: str, higher_id: str,
                         *, evaluator: bool = False) -> dict[str, object]:
    phase = {
        "pairs": 100,
        "lower_score": 0.5,
        "higher_score": 0.5,
        "delta": 0.0,
        "pair_difference_bootstrap_95": {
            "method": "paired_difference_percentile_stratified",
            "seed": "19",
            "resamples": 10_000,
            "confidence": 0.95,
            "lower": -0.01,
            "upper": 0.01,
        },
    }
    return {
        "id": f"fixture:{lower_id}-to-{higher_id}",
        "contrast": "neural_minus_hand" if evaluator
        else "higher_budget_minus_lower_budget",
        "lower_config_id": lower_id,
        "higher_config_id": higher_id,
        "lower_budget": 20_000,
        "higher_budget": 50_000,
        "phases": {
            "development": copy.deepcopy(phase),
            "validation": copy.deepcopy(phase),
        },
        "validation_classification": (
            "practical_equivalence_supported" if evaluator
            else "unresolved_at_1pp"
        ),
    }


def _development_validation_ablations() -> dict[str, object]:
    scaling_pairs = {
        "mcts": (
            ("mcts-1000", "mcts-2000"),
            ("mcts-2000", "mcts-4000"),
            ("mcts-1000", "mcts-4000"),
        ),
        "alpha_beta": (
            ("alpha-beta-20k", "alpha-beta-50k"),
            ("alpha-beta-50k", "alpha-beta-100k"),
            ("alpha-beta-20k", "alpha-beta-100k"),
        ),
        "jacek_inspired": (
            ("jacek-20k", "jacek-50k"),
            ("jacek-50k", "jacek-100k"),
            ("jacek-20k", "jacek-100k"),
        ),
    }
    return {
        "schema": "papersoccer.flagship-study-ablations.v1",
        "source_phases": ["development", "validation"],
        "practical_gain_threshold": 0.01,
        "bootstrap": {
            "method": "paired_difference_percentile",
            "resamples": 10_000,
            "confidence": 0.95,
            "unit": "aligned_color_swapped_opening_pair",
            "stratify_by": "opening_depth",
        },
        "scaling": {
            family: [
                _ablation_comparison(lower_id, higher_id)
                for lower_id, higher_id in pairs
            ]
            for family, pairs in scaling_pairs.items()
        },
        "equal_budget_evaluator": [
            _ablation_comparison(
                f"alpha-beta-{budget}", f"jacek-{budget}", evaluator=True
            )
            for budget in ("20k", "50k", "100k")
        ],
    }
def _fixture() -> tuple[dict, dict, dict, dict, dict, dict]:
    configurations = (
        [config_mcts(iterations) for iterations in (1000, 2000, 4000)]
        + [config_alpha_beta(nodes, neural=False)
           for nodes in (20_000, 50_000, 100_000)]
        + [config_alpha_beta(nodes, neural=True)
           for nodes in (20_000, 50_000, 100_000)]
        + [config_rank5()]
    )
    selected = {
        "mcts": "mcts-2000",
        "alpha_beta": "alpha-beta-50k",
        "jacek_inspired": "jacek-50k",
    }
    schedule = [
        ("test-mcts-vs-alpha-beta", "selected:mcts", "selected:alpha_beta"),
        ("test-mcts-vs-jacek", "selected:mcts", "selected:jacek_inspired"),
        ("test-mcts-vs-rank5", "selected:mcts", "fixed:rank5_derived"),
        ("test-alpha-beta-vs-jacek", "selected:alpha_beta", "selected:jacek_inspired"),
        ("test-alpha-beta-vs-rank5", "selected:alpha_beta", "fixed:rank5_derived"),
        ("test-jacek-vs-rank5", "selected:jacek_inspired", "fixed:rank5_derived"),
    ]
    manifest = {
        "study": {
            "title": "Synthetic flagship report",
            "preregistered_at_utc": "2026-08-03T00:00:00+00:00",
            "public_labels": {
                "mcts": "Tactical MctsBot",
                "alpha_beta": "Hand-evaluated AlphaBetaBot",
                "jacek_inspired": "Neural alpha-beta (JacekInspiredBot)",
                "rank5_derived": studylib.PUBLIC_RANK5_LABEL,
            },
            "rank5_disclaimer": studylib.RANK5_DISCLAIMER,
        },
        "source": {
            "git_commit": "1" * 40,
            "arena_sha256": "a" * 64,
            "opening_tool_sha256": "b" * 64,
            "analysis_contract_path": "benchmarks/flagship_study/analysis_contract.md",
            "analysis_contract_sha256": "2" * 64,
            "protected_artifacts": {
                "rank5_submission_path": "submissions/codingame/bots/rank_5/submission.cpp",
                "rank5_submission_sha256": studylib.RANK5_SOURCE_SHA256,
                "jacek_model_path": "models/jacek_article_value_model.json",
                "jacek_model_sha256": "3" * 64,
            },
        },
        "rules": {
            "width": 8,
            "height": 10,
            "playable_edges": 316,
            "max_game_plies": 512,
            "opening_ply_definition": (
                "One physical edge, including a same-player rebound edge."
            ),
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
                "description": (
                    "Separate deterministic uniform legal-edge data-generation mechanism"
                ),
            },
            "depths": [4, 8, 12, 20],
            "banks": [
                {
                    "path": f"benchmarks/flagship_study/openings/{phase}_d{depth:02d}.tsv",
                    "sha256": f"{index:064x}",
                }
                for index, (phase, depth) in enumerate(
                    [
                        (phase, depth)
                        for phase in ("development", "validation", "test")
                        for depth in (4, 8, 12, 20)
                    ],
                    start=1,
                )
            ],
        },
        "schedule": {
            "test": [
                {"id": identifier, "left_slot": left, "right_slot": right}
                for identifier, left, right in schedule
            ],
        },
        "latency_protocol": {
            "build_type": "Release",
            "gate_ms": 50,
            "timer_boundary": "steady_clock around choose_move",
            "warmup": "Eight untimed decisions per entrant",
            "state_copying": "The caller passes state by const reference",
            "power_conditions": "AC power and nominal thermal state",
        },
        "selection_rule": {
            "strength_metric": "mean paired score against the fixed comparator",
            "practical_tie_percentage_points": 1.0,
        },
        "statistics": {
            "bootstrap": {
                "resamples": 10_000,
            },
        },
        "environment": {
            "machine_id": "synthetic-gate",
            "os": "Synthetic OS",
            "cpu": "Synthetic CPU",
            "compiler_version": "Synthetic compiler 1.0",
            "build_flags": "-O3 -DNDEBUG",
        },
        "outputs": {
            "selection_lock": "benchmarks/flagship_study/selection_lock.json",
            "curated_data": {
                "development": "benchmarks/flagship_study/data/development.json",
                "validation": "benchmarks/flagship_study/data/validation.json",
                "test": "benchmarks/flagship_study/data/test.json",
            },
            "charts": {
                "bradley_terry": "benchmarks/flagship_study/charts/test_bt.svg",
                "pareto": "benchmarks/flagship_study/charts/validation_pareto.svg",
                "calibration": "benchmarks/flagship_study/charts/test_calibration.svg",
            },
            "report": "benchmarks/flagship_study/REPORT.md",
        },
    }

    candidate_ids = [config["id"] for config in configurations
                     if config["role"] == "candidate"]
    validation_metrics = {}
    development_configs = {}
    pareto = []
    for index, identifier in enumerate(candidate_ids):
        family = next(config["family"] for config in configurations
                      if config["id"] == identifier)
        budget = next(
            config["settings"].get("iterations", config["settings"].get("max_nodes"))
            for config in configurations if config["id"] == identifier
        )
        strength = 0.45 + 0.01 * index
        p95 = 20.0 + 4.0 * index
        is_selected = identifier in selected.values()
        eligible = p95 <= 50.0
        validation_metrics[identifier] = {
            "id": identifier,
            "family": family,
            "validation_strength": strength,
            "validation_p95_ms": p95,
            "eligible": eligible,
            "budget": budget,
            "within_practical_tie": is_selected,
            "selected": is_selected,
        }
        development_configs[identifier] = {
            "strength": {"mean_pair_score": strength - 0.01},
        }
        pareto.append({
            "id": identifier,
            "family": family,
            "validation_strength": strength,
            "validation_strength_pairs": 200,
            "validation_strength_pair_bootstrap_95": {
                "lower": max(0.0, strength - 0.05),
                "upper": min(1.0, strength + 0.05),
            },
            "development_strength": strength - 0.01,
            "development_strength_pairs": 100,
            "development_strength_pair_bootstrap_95": {
                "lower": max(0.0, strength - 0.06),
                "upper": min(1.0, strength + 0.04),
            },
            "validation_p95_ms": p95,
            "validation_latency_decisions": 500 + index,
            "strength_phases": ["development", "validation"],
            "latency_phase": "validation",
            "gate_eligible": eligible,
            "selected": is_selected,
            "fixed": False,
            "pareto_optimal": index in (0, len(candidate_ids) - 1),
            "constrained_pareto_optimal": index in (0, len(candidate_ids) - 1),
            "unconstrained_pareto_optimal": index in (0, len(candidate_ids) - 1),
            "dominated_by": [] if index in (0, len(candidate_ids) - 1) else [candidate_ids[0]],
        })
    pareto.append({
        "id": "rank5-fixed-50k",
        "family": "rank5_derived",
        "validation_strength": 0.5,
        "validation_strength_pairs": None,
        "validation_strength_pair_bootstrap_95": None,
        "development_strength": 0.5,
        "development_strength_pairs": None,
        "development_strength_pair_bootstrap_95": None,
        "validation_p95_ms": 42.0,
        "validation_latency_decisions": 450,
        "validation_comparator_pairs_total": 1_800,
        "strength_phases": ["development", "validation"],
        "latency_phase": "validation",
        "gate_eligible": True,
        "selected": True,
        "fixed": True,
        "pareto_optimal": True,
        "constrained_pareto_optimal": True,
        "unconstrained_pareto_optimal": True,
        "dominated_by": [],
    })
    selection = {
        "manifest_sha256": HASH,
        "source_phase": "validation",
        "selected_configurations": selected,
        "fixed_rank5_configuration": "rank5-fixed-50k",
        "validation_metrics": validation_metrics,
        "rank5_latency": {
            "fresh_root_p95_ms": 42.0,
            "all_edge_p95_ms": 30.0,
            "eligible_under_50_ms": True,
        },
        "validation_execution_environments": [{
            "observed_at_utc": "2026-08-03T01:00:00+00:00",
            "processor": "Synthetic CPU",
            "platform": "Synthetic OS",
            "arena_sha256": "a" * 64,
            "power_source": "ac",
            "build_provenance": {
                "sanitizers_enabled": False,
                "compiler_id": "Clang",
                "compiler_version": "1.0",
                "configured_flags": "-O3 -DNDEBUG",
            },
        }],
        "development_validation_ablations": (
            _development_validation_ablations()
        ),
        "validation_pareto": pareto,
        "test_authorized": True,
    }
    development = {
        "phase": "development",
        "manifest_sha256": HASH,
        "completeness": _completeness(units=9, games=18),
        "configurations": development_configs,
    }
    validation = {
        "phase": "validation",
        "manifest_sha256": HASH,
        "completeness": _completeness(units=9, games=18),
        "configurations": {},
    }

    resolved = {
        "selected:mcts": selected["mcts"],
        "selected:alpha_beta": selected["alpha_beta"],
        "selected:jacek_inspired": selected["jacek_inspired"],
        "fixed:rank5_derived": "rank5-fixed-50k",
    }
    matchups = {
        identifier: _pair_summary(resolved[left], resolved[right])
        for identifier, left, right in schedule
    }
    selected_ids = [*selected.values(), "rank5-fixed-50k"]
    analysis = {
        "bradley_terry": {
            "identifiability": "sum_to_zero",
            "intervals": {
                identifier: {"estimate": 0.0, "lower": -0.2, "upper": 0.2}
                for identifier in selected_ids
            },
        },
        "calibration": {
            identifier: _calibration_metrics()
            for identifier in selected_ids
        },
        "negative_findings": [
            "Additional computation did not monotonically improve validation strength."
        ],
    }
    test = {
        "phase": "test",
        "manifest_sha256": HASH,
        "completeness": _completeness(units=6, games=12),
        "matchups": matchups,
        "analysis": analysis,
    }
    artifact_hashes = {
        "benchmarks/flagship_study/manifest.json": "4" * 64,
        "benchmarks/flagship_study/selection_lock.json": "5" * 64,
        "benchmarks/flagship_study/data/development.json": "6" * 64,
        "benchmarks/flagship_study/data/validation.json": "7" * 64,
        "benchmarks/flagship_study/data/test.json": "8" * 64,
        "benchmarks/flagship_study/charts/test_bt.svg": "9" * 64,
        "benchmarks/flagship_study/runtime_projection.json": "c" * 64,
    }
    return manifest, selection, development, validation, test, artifact_hashes


class ReportTests(unittest.TestCase):
    def test_required_sections_disclaimer_links_and_zero_truncations(self) -> None:
        report = render_report(*_fixture())

        headings = (
            "## Research question and hypotheses",
            "## Entrants",
            "## Controls and frozen openings",
            "## Candidate grids",
            "## Latency protocol",
            "## Selection rule",
            "## Development and validation findings",
            "## Locked configurations",
            "## Frozen test results",
            "## Bradley–Terry relative strength",
            "## Calibration",
            "## Validation Pareto frontier",
            "## Negative and statistically unresolved findings",
            "## Limitations and threats to validity",
            "## Exact reproduction commands",
            "## Artifact hashes",
            "## Integrity",
        )
        for heading in headings:
            self.assertIn(heading, report)
        self.assertIn(studylib.RANK5_DISCLAIMER, report)
        self.assertIn("zero truncations", report)
        self.assertIn("(charts/test_bt.svg)", report)
        self.assertIn("(data/test.json)", report)
        self.assertNotRegex(report, re.compile(r"random[\s_-]*bot", re.IGNORECASE))

    def test_material_tables_and_reproduction_contract_are_rendered(self) -> None:
        manifest, selection, *rest = _fixture()
        report = render_report(manifest, selection, *rest)

        ablation_section = report.split(
            "### Preregistered development/validation ablations", 1
        )[1].split("## Locked configurations", 1)[0]
        expected_scaling = (
            ("mcts-1000", "mcts-2000"),
            ("mcts-2000", "mcts-4000"),
            ("mcts-1000", "mcts-4000"),
            ("alpha-beta-20k", "alpha-beta-50k"),
            ("alpha-beta-50k", "alpha-beta-100k"),
            ("alpha-beta-20k", "alpha-beta-100k"),
            ("jacek-20k", "jacek-50k"),
            ("jacek-50k", "jacek-100k"),
            ("jacek-20k", "jacek-100k"),
        )
        for lower_id, higher_id in expected_scaling:
            self.assertIn(f"{lower_id} → {higher_id}", ablation_section)
        for budget in ("20k", "50k", "100k"):
            self.assertIn(
                f"alpha-beta-{budget} → jacek-{budget} (neural minus hand)",
                ablation_section,
            )

        reliability_section = report.split(
            "### Ten-bin reliability summaries", 1
        )[1].split("## Validation Pareto frontier", 1)[0]
        configs = {
            config["id"]: config for config in manifest["configurations"]
        }
        selected_ids = [
            *selection["selected_configurations"].values(),
            selection["fixed_rank5_configuration"],
        ]
        for identifier in selected_ids:
            label = configs[identifier]["public_label"]
            self.assertEqual(reliability_section.count(f"| {label} |"), 10)
        self.assertIn("Prediction n", reliability_section)
        self.assertIn("Pair n", reliability_section)
        self.assertIn("[0.000, 0.150] (10,000 populated replicates)",
                      reliability_section)

        pareto_section = report.split("## Validation Pareto frontier", 1)[1].split(
            "## Negative and statistically unresolved findings", 1
        )[0]
        self.assertIn("[40.0%, 50.0%] (n=200)", pareto_section)
        self.assertIn("20.000 (n=500)", pareto_section)
        self.assertGreaterEqual(pareto_section.count("defined; n=N/A"), 2)
        self.assertIn(studylib.PUBLIC_RANK5_LABEL, pareto_section)
        self.assertNotIn("| rank5-fixed-50k |", pareto_section)

        reproduction = report.split("## Exact reproduction commands", 1)[1].split(
            "## Artifact hashes", 1
        )[0]
        self.assertIn("prepare_manifest.py", reproduction)
        self.assertIn("Freeze flagship manifest and opening banks", reproduction)
        self.assertIn("Lock flagship validation selection", reproduction)
        self.assertIn(
            "for index in 0 3 4 7 8 11 12 15 16 19 20 23 24 27 28 31 32 35; do",
            reproduction,
        )
        self.assertEqual(reproduction.count("for index in $(seq 0 35); do"), 2)
        self.assertIn("for index in $(seq 0 23); do", reproduction)
        self.assertNotIn("--allow-test-override", reproduction)
        self.assertIn("Native arena", report)
        self.assertIn("Opening-bank generator", report)
        self.assertIn("[benchmarks/flagship_study/runtime_projection.json]", report)
        self.assertIn("`" + "a" * 64 + "`", report)
        self.assertIn("`" + "b" * 64 + "`", report)

    def test_percentile_intervals_need_not_contain_point_estimates(self) -> None:
        fixture = list(_fixture())
        selection = fixture[1]
        candidate = next(
            point for point in selection["validation_pareto"]
            if not point["fixed"]
        )
        candidate["validation_strength_pair_bootstrap_95"] = {
            "lower": 0.0,
            "upper": 0.1,
        }
        selected_id = next(iter(selection["selected_configurations"].values()))
        fixture[4]["analysis"]["calibration"][selected_id][
            "pair_cluster_bootstrap_95"
        ]["brier_score"] = {"lower": 0.0, "upper": 0.1}

        report = render_report(*fixture)

        self.assertIn("[0.0%, 10.0%]", report)
        self.assertIn("0.2000 [0.0000, 0.1000]", report)

    def test_fixed_pareto_finding_uses_the_mandated_public_label(self) -> None:
        fixture = list(_fixture())
        fixed = next(
            point for point in fixture[1]["validation_pareto"]
            if point["fixed"]
        )
        fixed["gate_eligible"] = False
        fixed["validation_p95_ms"] = 55.0

        report = render_report(*fixture)
        findings = report.split(
            "## Negative and statistically unresolved findings", 1
        )[1].split("## Limitations and threats to validity", 1)[0]

        self.assertIn(
            f"{studylib.PUBLIC_RANK5_LABEL} missed the 50 ms gate", findings
        )
        self.assertNotIn("rank5-fixed-50k missed", findings)

    def test_unresolved_and_both_directions_of_stronger_claim(self) -> None:
        fixture = list(_fixture())
        test = fixture[4]
        summaries = list(test["matchups"].values())
        # Left wins both; its lower bound clears 50%.
        summaries[0].update({
            "left_wins": 2,
            "left_losses": 0,
            "right_wins": 0,
            "right_losses": 2,
            "pairs_won_2_0": 1,
            "pairs_split_1_1": 0,
            "pairs_lost_0_2": 0,
            "mean_pair_score": 1.0,
            "pair_bootstrap_95": {"lower": 0.6, "upper": 1.0},
        })
        # Left loses both; transformed right lower = 1 - left upper clears 50%.
        summaries[1].update({
            "left_wins": 0,
            "left_losses": 2,
            "right_wins": 2,
            "right_losses": 0,
            "pairs_won_2_0": 0,
            "pairs_split_1_1": 0,
            "pairs_lost_0_2": 1,
            "mean_pair_score": 0.0,
            "pair_bootstrap_95": {"lower": 0.0, "upper": 0.4},
        })

        report = render_report(*fixture)

        self.assertIn("Tactical MctsBot is stronger than Hand-evaluated AlphaBetaBot", report)
        self.assertIn("Neural alpha-beta (JacekInspiredBot) is stronger than Tactical MctsBot", report)
        self.assertIn("Statistically unresolved.", report)

    def test_truncation_and_incomplete_phase_are_publication_errors(self) -> None:
        truncated = list(_fixture())
        truncated[4]["completeness"]["truncations"] = 1
        with self.assertRaisesRegex(ReportError, "truncations"):
            render_report(*truncated)

        incomplete = list(_fixture())
        incomplete[2]["completeness"]["completed_games"] -= 1
        with self.assertRaisesRegex(ReportError, "incomplete"):
            render_report(*incomplete)

    def test_prohibited_benchmark_label_is_rejected(self) -> None:
        fixture = list(_fixture())
        manifest = fixture[0]
        manifest["configurations"][0]["public_label"] = "RandomBot"

        with self.assertRaisesRegex(ReportError, "prohibited"):
            render_report(*fixture)

    def test_changed_fixed_rank5_profile_is_rejected(self) -> None:
        fixture = list(_fixture())
        rank5 = next(
            config
            for config in fixture[0]["configurations"]
            if config["id"] == "rank5-fixed-50k"
        )
        rank5["settings"]["max_nodes"] = 49_999

        with self.assertRaisesRegex(ReportError, "fixed Rank5Derived setting changed"):
            render_report(*fixture)

    def test_output_is_byte_identical_for_identical_inputs(self) -> None:
        inputs = _fixture()
        first = render_report(*copy.deepcopy(inputs)).encode("utf-8")
        second = render_report(*copy.deepcopy(inputs)).encode("utf-8")

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
