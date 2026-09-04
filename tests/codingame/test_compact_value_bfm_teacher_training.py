#!/usr/bin/env python3
"""Focused, non-heavy tests for the Rank-4 teacher training adapter."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import pathlib
import tempfile
import unittest
from unittest import mock

for _thread_variable in (
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ[
    "PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"
] = "1"

import numpy as np

from tools import compact_value_bfm_teacher_training as adapter


def active_row(category: int) -> np.ndarray:
    return np.asarray(
        [
            adapter.trainer.EDGE_COUNT
            + vertex * adapter.trainer.VERTEX_CATEGORIES
            + category
            for vertex in range(adapter.trainer.VERTEX_COUNT)
        ],
        dtype="<u2",
    )


def dataset(
    category: int, root: str, *, split: str,
) -> adapter.trainer.Dataset:
    active = active_row(category)
    return adapter.trainer.Dataset(
        indptr=np.asarray([0, len(active)], dtype="<i8"),
        indices=active,
        targets=np.asarray([0.0], dtype="<f4"),
        weights=np.asarray([1.0], dtype="<f4"),
        group_ids=np.asarray(
            [hashlib.sha256(root.encode()).digest()], dtype="V32"
        ),
        split=split,
        source_manifest_sha256=f"{category + 1:064x}",
        source_npz_sha256=f"{category + 11:064x}",
        source_route=f"fixture/{split}-{category}.json",
    )


def group(
    category: int, root: str, *, split: str,
) -> adapter.trainer.CompleteTurnGroup:
    return adapter.trainer.CompleteTurnGroup(
        group_id=f"{category + 101:064x}",
        parent_mover=0,
        successors=(
            adapter.trainer.CompleteTurnSuccessor(
                successor_id=f"{category + 201:064x}",
                active=active_row(category),
                teacher_value=0.5,
                value_mover=1,
                evidence={"transcript": "0"},
            ),
        ),
        evidence={
            "source_binding": {
                "split": split,
                "root_group_id": root,
                "prefix": "",
            }
        },
    )


def rankings(
    train_category: int = 5, validation_category: int = 6,
) -> adapter.trainer.SuccessorRankingLabels:
    return adapter.trainer.SuccessorRankingLabels(
        train=(group(train_category, "new-train", split="train"),),
        validation=(
            group(validation_category, "new-validation", split="validation"),
        ),
        teacher={"kind": "fixture"},
        source_bundle_body_sha256="a" * 64,
        artifact_sha256="b" * 64,
        body_sha256="c" * 64,
    )


def arm(weight: float, regret: float, flip: float, *, passed: bool = True):
    return {
        "ranking_weight": weight,
        "seed": 20260907,
        "offline_gate_passed": passed,
        "metrics": {
            "mean_teacher_regret": regret,
            "top1_agreement": 0.75,
            "float_vs_quantized_action_flip_rate": flip,
            "ranking_validation_groups": 125,
            "comparable_exhaustive_validation_groups": 110,
            "comparable_exhaustive_validation_fraction": 0.88,
        },
        "source": {"reserve_target_met": True},
    }


def summary(wins: int, failures: int = 0):
    return {
        "candidate_wins": wins,
        "operational_failures": failures,
        "unfinished": 0,
    }


def clean_summary(wins: int):
    return {
        **summary(wins),
        "failures": {
            name: 0
            for name in adapter.challenger.qualification.FAILURE_CATEGORIES
        },
    }


def standard_variant_cleanliness(**dirty):
    result = {}
    for variant in adapter.SEARCH_VARIANT_ORDER:
        clean = not dirty.get(variant, False)
        result[variant] = {
            "variant": variant,
            "role": "control" if variant == "baseline" else "search-change",
            "source_clean": clean,
            "timing_clean": clean,
            "retention_clean": variant == "baseline" or clean,
            "source_evidence": {"fixture": True},
            "timing_evidence": {"fixture": True},
        }
    return result


def pilot_variant_cleanliness(**dirty):
    return {
        "baseline": standard_variant_cleanliness(**dirty)["baseline"],
    }


def hard_state_density(*, hard: bool = False):
    unique = 10
    hard_groups = 2 if hard else 0
    multiplier = adapter.trainer.HARD_STATE_DENSITY_MULTIPLIER
    return {
        "policy": "deterministic-expanded-ranking-schedule-v1",
        "hard_teacher_ranking_profile": (
            adapter.pipeline.HARD_5PCT_2M_TEACHER_RANKING_PROFILE
        ),
        "hard_group_multiplier": multiplier,
        "unique_comparable_groups": unique,
        "hard_unique_groups": hard_groups,
        "scheduled_group_entries": unique + hard_groups * (multiplier - 1),
        "hard_scheduled_entries": hard_groups * multiplier,
        "density_increased": hard,
    }


def native_thread_execution():
    environment = dict(adapter.trainer.NATIVE_THREAD_ENVIRONMENT)
    return {
        "schema": adapter.trainer.NATIVE_THREAD_EXECUTION_SCHEMA,
        "native_threads_per_seed_maximum": 1,
        "environment_required": environment,
        "environment_at_numpy_import": environment,
        "environment_at_worker_launch": environment,
        "environment_precedes_numpy_import": True,
        "preimport_bootstrap_marker": "1",
        "limiter_scope": "outer-roster-established-before-seed-workers",
        "threadpoolctl_available": False,
        "threadpoolctl_version": None,
        "threadpool_controllers": [],
    }


def treatment_selection_fixture(profile: str):
    base_wins = {
        "baseline": 106,
        "no-feature-sort-only": 108,
        "single-pass-selection-only": 107,
        "combined": 109,
    }
    summaries = {
        name: clean_summary(wins) for name, wins in base_wins.items()
    }
    evidence = {}
    required_checks = {
        "identical_model": True,
        "identical_bank_and_configuration": True,
        "identical_pair_roster": True,
        "base_zero_failures": True,
        "treatment_zero_failures": True,
        "source_clean": True,
        "parity_clean": True,
        "timing_clean": True,
        "base_control_profile_clean": True,
        "intervention_activated": True,
        "strictly_better_paired_wins": True,
        "strictly_more_candidate_wins": True,
    }
    for base, wins in base_wins.items():
        treatment = adapter._treatment_variant(base, profile)
        summaries[treatment] = clean_summary(wins + 1)
        evidence[treatment] = {
            "search_throughput_profile": profile,
            "profile_compile_macro": (
                adapter.SEARCH_THROUGHPUT_PROFILE_MACROS[profile]
            ),
            "standard_base_variant": base,
            "treatment_variant": treatment,
            "pairs": 100,
            "candidate_win_delta": 1,
            "paired_candidate_win_delta": 1,
            "better_pairs": 2,
            "equal_pairs": 97,
            "worse_pairs": 1,
            "checks": dict(required_checks),
            "pairwise_supported": True,
        }
    return summaries, evidence


def pilot_treatment_selection_fixture(profile: str):
    summaries, evidence = treatment_selection_fixture(profile)
    treatment = adapter._treatment_variant("baseline", profile)
    return (
        {name: summaries[name] for name in ("baseline", treatment)},
        {treatment: evidence[treatment]},
    )


def gate_document(pair_scores, *, soft_overruns: int = 0):
    games = []
    color_wins = [0, 0]
    for pair, score in enumerate(pair_scores):
        for color in (0, 1):
            won = color < score
            if won:
                color_wins[color] += 1
            games.append({
                "pair_index": pair,
                "candidate_player": color,
                "winner": color if won else 1 - color,
                "failure": None,
            })
    decisions = len(games)
    return {
        "config": {
            "mode": "actual-clock",
            "pair_count": len(pair_scores),
        },
        "games": games,
        "result": {
            "games": len(games),
            "candidate_wins": sum(color_wins),
            "candidate_wins_player0": color_wins[0],
            "candidate_wins_player1": color_wins[1],
            "failures": 0,
            "unfinished": 0,
            "failure_categories": {},
            "candidate": {
                "decisions": decisions,
                "soft_overruns": soft_overruns,
                "headroom_failures": 0,
                "hard_timeouts": 0,
                "times_ms": [1.0] * decisions,
            },
        },
    }


def passing_full_gate_document(wins: int = 550):
    if not 500 <= wins <= 1_000:
        raise ValueError("fixture wins outside paired-game domain")
    swept_pairs = wins - 500
    split_pairs = 1_000 - wins
    games = []
    color_wins = [0, 0]
    for pair in range(adapter.FULL_PAIRS):
        winning_colors = (
            {0, 1}
            if pair < swept_pairs
            else {(pair - swept_pairs) % 2}
            if pair < swept_pairs + split_pairs
            else set()
        )
        for color in (0, 1):
            won = color in winning_colors
            if won:
                color_wins[color] += 1
            games.append({
                "pair_index": pair,
                "candidate_player": color,
                "winner": color if won else 1 - color,
                "failure": None,
            })
    decisions = len(games)
    return {
        "config": {
            "mode": "actual-clock",
            "pair_count": adapter.FULL_PAIRS,
        },
        "games": games,
        "result": {
            "games": len(games),
            "candidate_wins": sum(color_wins),
            "candidate_wins_player0": color_wins[0],
            "candidate_wins_player1": color_wins[1],
            "failures": 0,
            "unfinished": 0,
            "failure_categories": {},
            "candidate": {
                "decisions": decisions,
                "soft_overruns": 0,
                "headroom_failures": 0,
                "hard_timeouts": 0,
                "times_ms": [1.0] * decisions,
            },
        },
    }


def profile_request(profile: str, variant: str):
    metadata = adapter._search_variant_metadata(profile, variant)
    return {
        "body_sha256": "a" * 64,
        "training_selection": {"sha256": "1" * 64},
        "training_selection_body_sha256": "2" * 64,
        "ranking_weight": 0.10,
        "seed": 20260907,
        "runtime": {"sha256": "3" * 64},
        "runtime_body_sha256": "4" * 64,
        "runtime_payload_sha256": "5" * 64,
        "base_model_source": {"sha256": "6" * 64},
        "bank": {"gate_tsv": {"sha256": "7" * 64}},
        "configuration": {
            "mode": "actual-clock",
            "candidate_search_profile": metadata[
                "candidate_search_profile"
            ],
        },
        "search_throughput_profile": profile,
        "search_variant": variant,
        "search_variant_metadata": metadata,
        "compile_time_macros": metadata["compile_time_macros"],
        "macros_embedded_at_source_start": True,
        "source_is_default_for_variant": True,
        "source_reserve": 2_500,
        "candidate_source": {"bytes": 90_000, "sha256": "8" * 64},
    }


class IsolationTests(unittest.TestCase):
    def test_standard_external_csr_reference_uses_maintained_loader(self):
        from tools import jacek_replay_train as replay_train

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            npz, manifest, _document = replay_train.write_csr_shard(
                root / "shard",
                "train",
                [
                    adapter.corpus.LabeledSample(
                        tuple(map(int, active_row(0))),
                        0.25,
                        1.0,
                        "root",
                    )
                ],
            )
            reference = root / "train-reference.json"
            adapter._write_sealed(reference, {
                "schema": adapter.pipeline.SHARD_REFERENCE_SCHEMA,
                "pipeline_body_sha256": "a" * 64,
                "split": "train",
                "manifest": adapter._record(manifest),
                "npz": adapter._record(npz),
                "shard_schema": adapter.trainer.SHARD_SCHEMA,
                "protected_tests_opened": False,
            })
            loaded, binding = adapter._load_shard_reference(
                reference,
                pipeline_body_sha256="a" * 64,
                split="train",
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(binding["manifest"]["sha256"], adapter.sha256_file(manifest))

    def test_extended_isolation_accepts_disjoint_external_roots_and_successors(self):
        report = adapter._extended_isolation(
            external_train=dataset(0, "new-train", split="train"),
            external_validation=dataset(1, "new-validation", split="validation"),
            anchor=dataset(2, "anchor", split="train"),
            common=dataset(3, "common", split="validation"),
            canonical_validation=dataset(4, "canonical", split="validation"),
            rankings=rankings(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["ranking_train_successors"], 1)

    def test_extended_isolation_rejects_successor_symmetry_leak(self):
        with self.assertRaisesRegex(
            adapter.TeacherTrainingError, "successor exact-rotate-reflect"
        ):
            adapter._extended_isolation(
                external_train=dataset(0, "new-train", split="train"),
                external_validation=dataset(
                    1, "new-validation", split="validation"
                ),
                anchor=dataset(2, "anchor", split="train"),
                common=dataset(3, "common", split="validation"),
                canonical_validation=dataset(
                    4, "canonical", split="validation"
                ),
                rankings=rankings(5, 5),
            )

    def test_root_binding_must_cover_external_scalar_roster(self):
        bad = adapter.trainer.SuccessorRankingLabels(
            train=(group(5, "other-train", split="train"),),
            validation=(group(6, "new-validation", split="validation"),),
            teacher={},
            source_bundle_body_sha256="a" * 64,
            artifact_sha256="b" * 64,
            body_sha256="c" * 64,
        )
        with self.assertRaisesRegex(adapter.TeacherTrainingError, "every scalar root"):
            adapter._extended_isolation(
                external_train=dataset(0, "new-train", split="train"),
                external_validation=dataset(
                    1, "new-validation", split="validation"
                ),
                anchor=dataset(2, "anchor", split="train"),
                common=dataset(3, "common", split="validation"),
                canonical_validation=dataset(
                    4, "canonical", split="validation"
                ),
                rankings=bad,
            )


class QATProfileBindingTests(unittest.TestCase):
    def test_hard_state_density_is_bound_to_teacher_ranking_adaptation(self):
        self.assertTrue(adapter._validated_hard_state_density(
            hard_state_density(), teacher_ranking_profile="standard-v1"
        ))
        self.assertTrue(adapter._validated_hard_state_density(
            hard_state_density(hard=True),
            teacher_ranking_profile="hardest-5pct-2m-v1",
        ))
        with self.assertRaisesRegex(
            adapter.TeacherTrainingError, "does not match"
        ):
            adapter._validated_hard_state_density(
                hard_state_density(hard=True),
                teacher_ranking_profile="standard-v1",
            )

    def test_profile_is_derived_only_from_phase_adaptation_contract(self):
        self.assertNotIn(
            "qat_profile", inspect.signature(adapter.prepare_training).parameters
        )
        for name in ("standard-v1", "refined-adaptive-scales-v1"):
            adaptation = {
                **adapter.challenger.STANDARD_ADAPTATION_CONTRACT,
                "qat_profile": name,
            }
            observed_name, contract = adapter._phase_qat_profile({
                "phase": {"adaptation_contract": adaptation}
            })
            self.assertEqual(observed_name, name)
            self.assertEqual(
                contract, adapter.trainer.qat_profile_contract(name)
            )
        for phase_context in (
            {"phase": {"adaptation_contract": {}}},
            {"phase": {"adaptation_contract": {
                **adapter.challenger.STANDARD_ADAPTATION_CONTRACT,
                "qat_profile": "other",
            }}},
        ):
            with self.subTest(phase_context=phase_context), self.assertRaisesRegex(
                adapter.TeacherTrainingError, "adaptation contract"
            ):
                adapter._phase_qat_profile(phase_context)

    def test_seed_roster_requires_profile_and_execution_evidence(self):
        contract = adapter.trainer.qat_profile_contract(
            "refined-adaptive-scales-v1"
        )
        receipt = {
            "architecture": "capacity-12x8",
            "arm": "search-target",
            "qat_profile": "refined-adaptive-scales-v1",
            "qat_profile_contract": contract,
            "quantized_training": {"bound": True},
            "native_thread_execution": native_thread_execution(),
            "successor_ranking": {
                "labels_present": True,
                "loss_weight": 0.10,
                "hard_state_density": hard_state_density(),
                "schedule_execution": {"fixture": True},
            },
            "offline_gate": {},
            "quantized_validation": {},
            "float_checkpoint": {},
            "quantized_runtime": {},
        }
        receipts = [
            {**receipt, "seed": seed} for seed in adapter.trainer.FIXED_SEEDS
        ]
        with mock.patch.object(
            adapter.trainer,
            "validate_qat_execution_evidence",
            side_effect=lambda value, **_kwargs: dict(value),
        ), mock.patch.object(
            adapter.trainer,
            "validate_successor_schedule_execution",
            return_value={"fixture": True},
        ):
            self.assertEqual(
                len(adapter._validate_seed_roster(
                    receipts,
                    ranking_weight=0.10,
                    qat_profile="refined-adaptive-scales-v1",
                    teacher_ranking_profile="standard-v1",
                )),
                3,
            )
            tampered = copy.deepcopy(receipts)
            tampered[1]["qat_profile"] = "standard-v1"
            with self.assertRaisesRegex(
                adapter.TeacherTrainingError, "ranking/runtime contract"
            ):
                adapter._validate_seed_roster(
                    tampered,
                    ranking_weight=0.10,
                    qat_profile="refined-adaptive-scales-v1",
                    teacher_ranking_profile="standard-v1",
                )


class SelectionTests(unittest.TestCase):
    def test_model_is_selected_by_regret_flip_and_retention_before_games(self):
        result = adapter._model_selection(
            [
                arm(0.0, 0.10, 0.020),
                arm(0.10, 0.08, 0.024),
                arm(0.25, 0.07, 0.030),
            ]
        )
        self.assertEqual(result["status"], "model-selected-before-rank4-screen")
        self.assertEqual(result["selected_ranking_weight"], 0.10)
        self.assertAlmostEqual(
            result["selected_mean_teacher_regret_reduction_fraction"], 0.20
        )

    def test_model_selection_rejects_sub_ten_percent_regret_gain(self):
        result = adapter._model_selection(
            [
                arm(0.0, 0.10, 0.020),
                arm(0.10, 0.091, 0.020),
                arm(0.25, 0.12, 0.020),
            ]
        )
        self.assertEqual(result["status"], "offline-rejected-before-rank4-screen")
        self.assertIsNone(result["selected_ranking_weight"])
        self.assertEqual(result["diagnostic_ranking_weight"], 0.10)
        self.assertEqual(result["diagnostic_seed"], 20260907)

    def test_model_selection_rejects_sparse_comparable_validation(self):
        sparse = arm(0.10, 0.07, 0.020)
        sparse["metrics"]["comparable_exhaustive_validation_groups"] = 99
        sparse["metrics"]["comparable_exhaustive_validation_fraction"] = 0.792
        result = adapter._model_selection([
            arm(0.0, 0.10, 0.020),
            sparse,
            arm(0.25, 0.12, 0.020),
        ])
        self.assertEqual(result["status"], "offline-rejected-before-rank4-screen")

    def test_offline_rejected_diagnostic_still_reaches_gate_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan = {
                "outputs": {"gate_reference": str(root / "gate-reference.json")},
            }
            selection = {
                "model_selection": {
                    "status": "offline-rejected-before-rank4-screen",
                },
                "selected_model": {
                    "offline_eligible": False,
                    "diagnostic_only": True,
                },
            }
            with (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "load_training_selection", return_value=selection
                ),
                mock.patch.object(
                    adapter,
                    "sha256_file",
                    return_value=adapter.gate_support.RANK4_SHA256,
                ),
                mock.patch.object(
                    adapter,
                    "training_context",
                    side_effect=adapter.TeacherTrainingError(
                        "diagnostic reached gate inputs"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "diagnostic reached gate inputs"
                ):
                    adapter.prepare_gate(
                        root / "plan.json",
                        selection_path=root / "selection.json",
                        bank_path=root / "bank.json",
                    )

    def test_combined_search_requires_both_independent_ab_improvements(self):
        selected = adapter._select_complete_search_variant(
            {
                "baseline": summary(106),
                "no-feature-sort-only": summary(108),
                "single-pass-selection-only": summary(107),
                "combined": summary(109),
            },
            variant_cleanliness=standard_variant_cleanliness(),
        )
        self.assertEqual(selected["selected_variant"], "combined")
        self.assertTrue(
            selected["independent_changes"][
                "combined_supported_by_both_individual_arms"
            ]
        )

        unsupported = adapter._select_complete_search_variant(
            {
                "baseline": summary(106),
                "no-feature-sort-only": summary(106),
                "single-pass-selection-only": summary(108),
                "combined": summary(112),
            },
            variant_cleanliness=standard_variant_cleanliness(),
        )
        self.assertEqual(
            unsupported["selected_variant"], "single-pass-selection-only"
        )
        self.assertFalse(
            unsupported["independent_changes"][
                "combined_supported_by_both_individual_arms"
            ]
        )

    def test_variant_sources_encode_the_compile_time_reference_macros(self):
        base = b"int main(){return 0;}\n"
        baseline = adapter._variant_source(
            base, adapter.SEARCH_VARIANTS["baseline"]
        )
        self.assertTrue(
            baseline.startswith(
                b"#define COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT 1\n"
            )
        )
        self.assertIn(
            b"#define COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT 1\n",
            baseline,
        )
        self.assertEqual(adapter._variant_source(base, ()), base)


class SearchThroughputProfileTests(unittest.TestCase):
    def test_phase_gate_game_volumes_are_exact(self):
        standard_pilot = adapter.phase_gate_variants("pilot", "standard-v1")
        standard_full = adapter.phase_gate_variants("full", "standard-v1")
        self.assertEqual(
            adapter._expected_gate_game_volume(
                tuple(standard_pilot), pairs_per_variant=adapter.PILOT_PAIRS
            )["total_games"],
            200,
        )
        self.assertEqual(
            adapter._expected_gate_game_volume(
                tuple(standard_full), pairs_per_variant=adapter.FULL_PAIRS
            )["total_games"],
            4_000,
        )
        for profile in adapter.SEARCH_THROUGHPUT_PROFILE_ORDER[1:]:
            with self.subTest(profile=profile):
                pilot = adapter.phase_gate_variants("pilot", profile)
                full = adapter.phase_gate_variants("full", profile)
                self.assertEqual(len(pilot), 2)
                self.assertEqual(len(full), 8)
                self.assertEqual(
                    adapter._expected_gate_game_volume(
                        tuple(pilot), pairs_per_variant=adapter.PILOT_PAIRS
                    )["total_games"],
                    400,
                )
                self.assertEqual(
                    adapter._expected_gate_game_volume(
                        tuple(full), pairs_per_variant=adapter.FULL_PAIRS
                    )["total_games"],
                    8_000,
                )
        self.assertEqual(
            adapter._expected_gate_game_volume(
                ("selected",), pairs_per_variant=adapter.FULL_PAIRS
            )["total_games"],
            1_000,
        )

    def test_full_roster_stays_four_while_standard_pilot_is_baseline_only(self):
        self.assertEqual(
            adapter.active_search_variants("standard-v1"),
            adapter.SEARCH_VARIANTS,
        )
        self.assertEqual(
            adapter.phase_gate_variants("pilot", "standard-v1"),
            {"baseline": adapter.SEARCH_VARIANTS["baseline"]},
        )
        selected = adapter.select_pilot_search_variant(
            {"baseline": clean_summary(105)},
            variant_cleanliness=pilot_variant_cleanliness(),
        )
        self.assertEqual(selected["selected_variant"], "baseline")
        self.assertTrue(selected["selected_variant_passed_screen"])
        self.assertEqual(selected["expected_game_volume"]["total_games"], 200)
        policy = adapter._search_ab_policy(
            phase="pilot",
            profile="standard-v1",
            variants=("baseline",),
            pilot_prior_variant=None,
        )
        self.assertEqual(policy["roster_policy"], "single-baseline-model-screen")
        self.assertFalse(policy["full_reruns_complete_active_roster"])
        self.assertEqual(policy["expected_game_volume"]["total_games"], 200)
        dirty = pilot_variant_cleanliness(baseline=True)
        rejected = adapter.select_pilot_search_variant(
            {"baseline": clean_summary(120)},
            variant_cleanliness=dirty,
        )
        self.assertFalse(rejected["selected_variant_passed_screen"])
        self.assertEqual(rejected["retained_variants"], [])

    def test_each_intervention_adds_one_independent_arm_per_standard_base(self):
        for profile, macro in adapter.SEARCH_THROUGHPUT_PROFILE_MACROS.items():
            if profile == "standard-v1":
                continue
            with self.subTest(profile=profile):
                variants = adapter.active_search_variants(profile)
                self.assertEqual(
                    list(variants)[:4], list(adapter.SEARCH_VARIANT_ORDER)
                )
                self.assertEqual(len(variants), 8)
                for base in adapter.SEARCH_VARIANT_ORDER:
                    treatment = adapter._treatment_variant(base, profile)
                    self.assertEqual(variants[base], adapter.SEARCH_VARIANTS[base])
                    self.assertEqual(
                        adapter._search_variant_metadata(profile, base)[
                            "candidate_search_profile"
                        ],
                        "standard-v1",
                    )
                    self.assertEqual(
                        variants[treatment],
                        (*adapter.SEARCH_VARIANTS[base], macro),
                    )
                    self.assertEqual(
                        adapter._search_variant_metadata(profile, treatment)[
                            "candidate_search_profile"
                        ],
                        profile,
                    )

    def test_intervention_pilot_is_one_baseline_treatment_pair_with_fallback(self):
        profile = "state-evaluation-cache-v1"
        treatment = adapter._treatment_variant("baseline", profile)
        self.assertEqual(
            list(adapter.phase_gate_variants("pilot", profile)),
            ["baseline", treatment],
        )
        summaries, evidence = pilot_treatment_selection_fixture(profile)
        selected = adapter.select_pilot_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=evidence,
            variant_cleanliness=pilot_variant_cleanliness(),
        )
        self.assertEqual(selected["selected_variant"], treatment)
        self.assertEqual(selected["expected_game_volume"]["total_games"], 400)
        self.assertFalse(selected["control_fallback"])

        evidence[treatment]["checks"]["intervention_activated"] = False
        evidence[treatment]["pairwise_supported"] = False
        fallback = adapter.select_pilot_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=evidence,
            variant_cleanliness=pilot_variant_cleanliness(),
        )
        self.assertEqual(fallback["selected_variant"], "baseline")
        self.assertTrue(fallback["control_fallback"])

    def test_intervention_pilot_derives_only_its_baseline_pair_evidence(self):
        profile = "progressive-widening-v1"
        variants = adapter.phase_gate_variants("pilot", profile)
        documents = {
            variant: gate_document(
                [1, 1, 1]
                if adapter._search_variant_metadata(profile, variant)[
                    "is_treatment"
                ]
                else [0, 1, 1]
            )
            for variant in variants
        }
        requests = {
            variant: profile_request(profile, variant) for variant in variants
        }
        with mock.patch.object(
            adapter,
            "_profile_activation_evidence",
            side_effect=lambda _document, *, expected_profile: {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": expected_profile,
                "exercised": True,
            },
        ):
            paired = adapter._paired_ab_evidence(
                documents,
                search_throughput_profile=profile,
                requests=requests,
                phase="pilot",
            )
        treatment = adapter._treatment_variant("baseline", profile)
        self.assertEqual(set(paired["treatment_comparisons"]), {treatment})
        self.assertEqual(paired["expected_game_volume"]["total_games"], 400)
        selected = adapter.select_pilot_search_variant(
            {
                variant: adapter._result_summary(document)
                for variant, document in documents.items()
            },
            search_throughput_profile=profile,
            treatment_evidence=paired["treatment_comparisons"],
            variant_cleanliness=pilot_variant_cleanliness(),
        )
        self.assertEqual(selected["selected_variant"], treatment)

    def test_clean_strictly_better_treatment_can_be_selected(self):
        profile = "state-evaluation-cache-v1"
        summaries, evidence = treatment_selection_fixture(profile)
        selection = adapter._select_complete_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=evidence,
            variant_cleanliness=standard_variant_cleanliness(),
        )
        selected = adapter._treatment_variant("combined", profile)
        self.assertEqual(selection["selected_variant"], selected)
        self.assertIn(selected, selection["retained_variants"])
        self.assertTrue(
            selection["treatment_comparisons"][selected][
                "base_independently_retained"
            ]
        )
        self.assertEqual(
            selection["active_variant_roster"],
            list(adapter.active_search_variants(profile)),
        )

    def test_treatment_retention_fails_closed_on_every_cleanliness_check(self):
        profile = "progressive-widening-v1"
        for failed_check in (
            "identical_model",
            "identical_bank_and_configuration",
            "identical_pair_roster",
            "base_zero_failures",
            "treatment_zero_failures",
            "source_clean",
            "parity_clean",
            "timing_clean",
            "base_control_profile_clean",
            "intervention_activated",
            "strictly_better_paired_wins",
            "strictly_more_candidate_wins",
        ):
            with self.subTest(failed_check=failed_check):
                summaries, evidence = treatment_selection_fixture(profile)
                for treatment in evidence:
                    evidence[treatment]["checks"][failed_check] = False
                    evidence[treatment]["pairwise_supported"] = False
                selection = adapter._select_complete_search_variant(
                    summaries,
                    search_throughput_profile=profile,
                    treatment_evidence=evidence,
                    variant_cleanliness=standard_variant_cleanliness(),
                )
                self.assertTrue(
                    set(selection["retained_variants"])
                    <= set(adapter.SEARCH_VARIANTS)
                )

    def test_treatment_cannot_bypass_an_unretained_standard_base(self):
        profile = "state-evaluation-cache-v1"
        summaries, evidence = treatment_selection_fixture(profile)
        summaries["no-feature-sort-only"] = clean_summary(106)
        summaries["combined"] = clean_summary(112)
        for base in ("no-feature-sort-only", "combined"):
            treatment = adapter._treatment_variant(base, profile)
            summaries[treatment] = clean_summary(
                summaries[base]["candidate_wins"] + 1
            )
            evidence[treatment]["candidate_win_delta"] = 1
            evidence[treatment]["paired_candidate_win_delta"] = 1
        selection = adapter._select_complete_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=evidence,
            variant_cleanliness=standard_variant_cleanliness(),
        )
        for base in ("no-feature-sort-only", "combined"):
            treatment = adapter._treatment_variant(base, profile)
            self.assertTrue(
                selection["treatment_comparisons"][treatment][
                    "pairwise_supported"
                ]
            )
            self.assertFalse(
                selection["treatment_comparisons"][treatment][
                    "base_independently_retained"
                ]
            )
            self.assertNotIn(treatment, selection["retained_variants"])

    def test_equal_candidate_wins_cannot_retain_treatment(self):
        profile = "subtree-reuse-v1"
        summaries, evidence = treatment_selection_fixture(profile)
        for base in adapter.SEARCH_VARIANT_ORDER:
            treatment = adapter._treatment_variant(base, profile)
            summaries[treatment] = clean_summary(
                summaries[base]["candidate_wins"]
            )
            evidence[treatment]["candidate_win_delta"] = 0
            evidence[treatment]["paired_candidate_win_delta"] = 0
            evidence[treatment]["better_pairs"] = 1
            evidence[treatment]["worse_pairs"] = 1
            evidence[treatment]["checks"][
                "strictly_better_paired_wins"
            ] = False
            evidence[treatment]["checks"][
                "strictly_more_candidate_wins"
            ] = False
            evidence[treatment]["pairwise_supported"] = False
        selection = adapter._select_complete_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=evidence,
            variant_cleanliness=standard_variant_cleanliness(),
        )
        self.assertTrue(
            set(selection["retained_variants"]) <= set(adapter.SEARCH_VARIANTS)
        )

    def test_paired_evidence_derives_model_bank_source_parity_and_timing_checks(self):
        profile = "state-evaluation-cache-v1"
        variants = adapter.active_search_variants(profile)
        documents = {}
        requests = {}
        for variant in variants:
            metadata = adapter._search_variant_metadata(profile, variant)
            documents[variant] = gate_document(
                [1, 1, 1] if metadata["is_treatment"] else [0, 1, 1]
            )
            requests[variant] = profile_request(profile, variant)
        with mock.patch.object(
            adapter,
            "_profile_activation_evidence",
            side_effect=lambda _document, *, expected_profile: {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": expected_profile,
                "exercised": True,
            },
        ):
            evidence = adapter._paired_ab_evidence(
                documents,
                search_throughput_profile=profile,
                requests=requests,
            )
        for treatment in evidence["treatment_comparisons"].values():
            self.assertTrue(treatment["pairwise_supported"])
            self.assertTrue(all(treatment["checks"].values()))
            self.assertEqual(
                treatment["input_bindings"]["runtime_payload_sha256"],
                "5" * 64,
            )
            self.assertEqual(
                treatment["parity_evidence"]["treatment"][
                    "lockstep_mismatch"
                ],
                0,
            )
            self.assertEqual(
                treatment["timing_evidence"]["treatment"][
                    "hard_timeouts"
                ],
                0,
            )
            self.assertTrue(treatment["intervention_activation"]["exercised"])

        target = adapter._treatment_variant("combined", profile)
        requests[target] = {
            **requests[target],
            "runtime_payload_sha256": "9" * 64,
        }
        documents[target] = gate_document([1, 1, 1], soft_overruns=1)
        with mock.patch.object(
            adapter,
            "_profile_activation_evidence",
            side_effect=lambda _document, *, expected_profile: {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": expected_profile,
                "exercised": True,
            },
        ):
            dirty = adapter._treatment_ab_evidence(
                documents, requests, profile=profile
            )[target]
        self.assertFalse(dirty["checks"]["identical_model"])
        self.assertFalse(dirty["checks"]["timing_clean"])
        self.assertFalse(dirty["pairwise_supported"])

    def test_full_standard_ab_can_replace_the_pilot_prior(self):
        profile = "standard-v1"
        wins = {
            "baseline": 550,
            "no-feature-sort-only": 552,
            "single-pass-selection-only": 551,
            "combined": 553,
        }
        documents = {
            variant: passing_full_gate_document(count)
            for variant, count in wins.items()
        }
        summaries = {
            variant: adapter._result_summary(document)
            for variant, document in documents.items()
        }
        requests = {
            variant: profile_request(profile, variant)
            for variant in adapter.SEARCH_VARIANT_ORDER
        }
        selected = adapter.select_full_search_variant(
            summaries,
            documents=documents,
            requests=requests,
            search_throughput_profile=profile,
            treatment_evidence=None,
            variant_cleanliness=standard_variant_cleanliness(),
            pilot_prior_variant="baseline",
        )
        self.assertEqual(selected["selected_variant"], "combined")
        self.assertTrue(selected["selected_variant_frozen_before_qualification"])
        self.assertFalse(selected["qualification_bank_read"])
        self.assertEqual(
            selected["pilot_prior"]["role"], "equal-win-tie-context-only"
        )
        self.assertEqual(
            selected["active_variant_roster"], list(adapter.SEARCH_VARIANT_ORDER)
        )
        self.assertEqual(selected["expected_game_volume"]["total_games"], 4_000)

    def test_full_ab_freezes_variant_without_claiming_qualification(self):
        profile = "standard-v1"
        wins = {
            "baseline": 500,
            "no-feature-sort-only": 501,
            "single-pass-selection-only": 502,
            "combined": 503,
        }
        documents = {
            variant: passing_full_gate_document(count)
            for variant, count in wins.items()
        }
        summaries = {
            variant: adapter._result_summary(document)
            for variant, document in documents.items()
        }
        requests = {
            variant: profile_request(profile, variant)
            for variant in adapter.SEARCH_VARIANT_ORDER
        }
        selected = adapter.select_full_search_variant(
            summaries,
            documents=documents,
            requests=requests,
            search_throughput_profile=profile,
            treatment_evidence=None,
            variant_cleanliness=standard_variant_cleanliness(),
            pilot_prior_variant="baseline",
        )
        self.assertEqual(selected["selected_variant"], "combined")
        self.assertNotIn("full_gate_evidence", selected)
        self.assertFalse(selected["qualification_bank_read"])

    def test_full_policy_rejects_the_old_single_inherited_variant_roster(self):
        with self.assertRaisesRegex(
            adapter.TeacherTrainingError, "variant roster is invalid"
        ):
            adapter._search_ab_policy(
                phase="full",
                profile="standard-v1",
                variants=("baseline",),
                pilot_prior_variant="baseline",
            )
        policy = adapter._search_ab_policy(
            phase="full",
            profile="standard-v1",
            variants=adapter.SEARCH_VARIANT_ORDER,
            pilot_prior_variant="baseline",
        )
        self.assertTrue(policy["full_reruns_complete_active_roster"])
        self.assertEqual(policy["pilot_prior_variant"], "baseline")
        self.assertTrue(policy["pilot_prior_used_only_for_ties"])

    def test_legacy_search_change_with_dirty_timing_is_not_retained(self):
        summaries = {
            "baseline": summary(106),
            "no-feature-sort-only": summary(112),
            "single-pass-selection-only": summary(108),
            "combined": summary(114),
        }
        cleanliness = standard_variant_cleanliness()
        cleanliness["no-feature-sort-only"]["timing_clean"] = False
        cleanliness["no-feature-sort-only"]["retention_clean"] = False
        selection = adapter._select_complete_search_variant(
            summaries,
            variant_cleanliness=cleanliness,
        )
        self.assertNotIn(
            "no-feature-sort-only", selection["retained_variants"]
        )
        self.assertNotIn("combined", selection["retained_variants"])
        self.assertEqual(
            selection["selected_variant"], "single-pass-selection-only"
        )

    def test_unexercised_intervention_is_not_retained(self):
        profile = "subtree-reuse-v1"
        variants = adapter.active_search_variants(profile)
        documents = {}
        requests = {}
        for variant in variants:
            metadata = adapter._search_variant_metadata(profile, variant)
            documents[variant] = gate_document(
                [1, 1, 1] if metadata["is_treatment"] else [0, 1, 1]
            )
            requests[variant] = profile_request(profile, variant)
        with mock.patch.object(
            adapter,
            "_profile_activation_evidence",
            side_effect=lambda _document, *, expected_profile: {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": expected_profile,
                "exercised": expected_profile == "standard-v1",
            },
        ):
            evidence = adapter._paired_ab_evidence(
                documents,
                search_throughput_profile=profile,
                requests=requests,
            )
        for treatment in evidence["treatment_comparisons"].values():
            self.assertFalse(treatment["intervention_activation"]["exercised"])
            self.assertFalse(treatment["checks"]["intervention_activated"])
            self.assertFalse(treatment["pairwise_supported"])

    def test_gate_result_validation_binds_effective_compiled_profile(self):
        profile = "progressive-widening-v1"
        variant = adapter._treatment_variant("baseline", profile)
        metadata = adapter._search_variant_metadata(profile, variant)
        request = {
            "phase": "pilot",
            "search_variant": variant,
            "search_variant_metadata": metadata,
            "bank": {"gate_tsv": {"sha256": "1" * 64}},
            "candidate_source": {"sha256": "2" * 64},
            "runtime_body_sha256": "3" * 64,
            "runtime_payload_sha256": "4" * 64,
        }
        document = {
            "config": adapter._expected_gate_configuration(
                "pilot", candidate_search_profile=profile
            ),
            "bindings": {
                "candidate_source_sha256": "2" * 64,
                "bank_sha256": "1" * 64,
                "rank4_source_sha256": adapter.gate_support.RANK4_SHA256,
                "candidate_runtime_body_sha256": "3" * 64,
                "candidate_payload_sha256": "4" * 64,
            },
        }
        with mock.patch.object(
            adapter.gate_support, "validate_result", return_value=document,
        ) as validate:
            self.assertIs(
                adapter._validate_gate_result(request, pathlib.Path("result.json")),
                document,
            )
        self.assertEqual(
            validate.call_args.kwargs["expected_candidate_search_profile"],
            profile,
        )

    def test_full_intervention_repeats_all_controls_and_treatments(self):
        profile = "state-evaluation-cache-v1"
        variants = adapter.active_search_variants(profile)
        base_wins = {
            "baseline": 550,
            "no-feature-sort-only": 552,
            "single-pass-selection-only": 551,
            "combined": 553,
        }
        documents = {}
        requests = {}
        for variant in variants:
            metadata = adapter._search_variant_metadata(profile, variant)
            wins = base_wins[metadata["standard_base_variant"]] + (
                1 if metadata["is_treatment"] else 0
            )
            documents[variant] = passing_full_gate_document(wins)
            requests[variant] = profile_request(profile, variant)
        summaries = {
            variant: adapter._result_summary(document)
            for variant, document in documents.items()
        }
        with mock.patch.object(
            adapter,
            "_profile_activation_evidence",
            side_effect=lambda _document, *, expected_profile: {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": expected_profile,
                "exercised": True,
            },
        ):
            paired = adapter._paired_ab_evidence(
                documents,
                search_throughput_profile=profile,
                requests=requests,
            )
        selected = adapter.select_full_search_variant(
            summaries,
            documents=documents,
            requests=requests,
            search_throughput_profile=profile,
            treatment_evidence=paired["treatment_comparisons"],
            variant_cleanliness=standard_variant_cleanliness(),
            pilot_prior_variant="baseline",
        )
        self.assertEqual(len(selected["active_variant_roster"]), 8)
        self.assertEqual(selected["expected_game_volume"]["total_games"], 8_000)
        self.assertEqual(
            selected["selected_variant"],
            adapter._treatment_variant("combined", profile),
        )
        self.assertTrue(selected["selected_variant_frozen_before_qualification"])
        self.assertFalse(selected["qualification_bank_read"])


class GateExecutionTests(unittest.TestCase):
    def test_production_hooks_fail_closed_and_bank_generation_is_single(self):
        production = {
            "execution_authority": {
                "production_allowlist_enforced": True,
                "heavy_stage_lock": "/tmp/.rank4-teacher-heavy-stage.lock",
            }
        }
        with self.assertRaisesRegex(
            adapter.TeacherTrainingError, "nonproduction test evidence"
        ):
            adapter._guard_test_hooks(
                production, hooks_used=True,
                allow_injected_test_evidence=True,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            selection = root / "selection.json"
            selection.write_bytes(b"selection")
            plan = {
                "campaign_id": "fixture-campaign",
                "attempt": 1,
                "phase": "pilot",
                "body_sha256": "1" * 64,
                "execution_authority": production["execution_authority"],
            }
            exclusion_evidence = {
                "state_fingerprint_count": 0,
                "state_fingerprints_sha256": adapter.sha256_bytes(
                    adapter.canonical_json_bytes([])
                ),
                "feature_fingerprint_count": 0,
                "feature_fingerprints_sha256": adapter.sha256_bytes(
                    adapter.canonical_json_bytes([])
                ),
                "sources": {},
                "protected_or_live_data_read_as_fingerprints_only": True,
            }
            with mock.patch.object(
                adapter, "_development_generation_exclusions",
                return_value=(set(), set(), exclusion_evidence),
            ):
                first, claim = adapter._generate_production_development_bank(
                    plan,
                    selection_path=selection,
                    selection_body_sha256="2" * 64,
                    purpose="pilot-screen", count=1,
                    output_directory=root / "generated",
                    inputs=mock.Mock(), external_validation=mock.Mock(),
                )
                second, second_claim = adapter._generate_production_development_bank(
                    plan,
                    selection_path=selection,
                    selection_body_sha256="2" * 64,
                    purpose="pilot-screen", count=1,
                    output_directory=root / "generated",
                    inputs=mock.Mock(), external_validation=mock.Mock(),
                )
            self.assertEqual(first, second)
            self.assertEqual(claim, second_claim)
            self.assertEqual(
                adapter.challenger.openings.validate_bank(first)["opening_count"], 1
            )

    def fixture(
        self, root,
        variants=("baseline", "single-pass-selection-only", "combined"),
    ):
        training_plan = root / "training-plan.json"
        training_plan.write_bytes(b"training-plan")
        binary = root / "gate.rank4-gate"
        binary.write_bytes(b"binary")
        binary.chmod(0o755)
        runtime = root / "candidate.runtime.json"
        runtime.write_bytes(b"runtime")
        source = root / "candidate.cpp"
        source.write_bytes(b"source")
        bank = root / "development-bank.json"
        bank.write_bytes(b"bank")
        development_path = root / "development-fingerprints.json"
        development = adapter._write_sealed(development_path, {
            "schema": adapter.challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "campaign_id": "fixture-campaign",
            "attempt": 1,
            "phase": "pilot",
            "classification": "unprotected-development-fingerprints",
            "fingerprints": ["1" * 64],
            "protected_or_live_data_included": False,
        })
        gate_plan_path = root / "gate-plan.json"
        gate_plan_path.write_bytes(b"gate-plan")
        requests = []
        for variant in variants:
            request_path = root / f"{variant}.request.json"
            request = adapter._write_sealed(request_path, {
                "schema": adapter.SCREEN_REQUEST_SCHEMA,
                "search_variant": variant,
                "search_variant_metadata": {
                    "candidate_search_profile": "standard-v1",
                },
                "runtime": adapter._record(runtime),
                "candidate_source": adapter._record(source),
                "binary": adapter._record(binary),
                "argv": [str(binary), "--output", "RESULT_PATH"],
            })
            requests.append({
                "variant": variant,
                "request": adapter._record(request_path),
                "request_body_sha256": request["body_sha256"],
            })
        plan = {
            "campaign_id": "fixture-campaign",
            "attempt": 1,
            "phase": "pilot",
            "body_sha256": "a" * 64,
            "execution_authority": {
                "production_allowlist_enforced": False,
                "heavy_stage_lock": None,
            },
            "outputs": {
                "plan": str(training_plan),
                "gate_executions": str(root / "executions"),
                "gate_execution_reference": str(root / "execution-reference.json"),
            },
            "build_source_closure": {"fixture": True},
        }
        gate_plan = {
            "body_sha256": "b" * 64,
            "active_search_variant_roster": list(variants),
            "execution_policy": adapter._gate_execution_policy(variants),
            "execution_outputs": {
                "root": plan["outputs"]["gate_executions"],
                "reference": plan["outputs"]["gate_execution_reference"],
            },
            "requests": requests,
            "bank": {"manifest": adapter._record(bank)},
            "development_exclusion": {
                **adapter._record(development_path),
                "schema": adapter.challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
                "body_sha256": development["body_sha256"],
            },
        }
        return plan, gate_plan_path, gate_plan

    def test_serial_whole_bank_execution_seals_claims_and_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, gate_plan = self.fixture(root)
            times = iter(
                f"2026-09-04T00:00:{second:02d}Z" for second in range(1, 10)
            )
            calls = []

            def runner(command, repository, environment, output):
                calls.append((list(command), dict(environment), output))
                output.write_bytes(b"raw")
                return {
                    "pid": 100 + len(calls),
                    "process_nice": 0,
                    "returncode": 0,
                    "stderr_bytes": 0,
                    "stderr_sha256": adapter.sha256_bytes(b""),
                    "stdout_policy": "discarded-duplicate-of-raw-result",
                }

            audit = {
                "process_nice": 0,
                "competing_rank4_gate_processes": [],
                "ps_stdout_sha256": "1" * 64,
                "ps_stderr_sha256": "2" * 64,
            }
            document = {"result": {"passed": True}}
            activation = {
                "schema": adapter.gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
                "candidate_search_profile": "standard-v1",
                "exercised": True,
            }
            with (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "_load_gate_request_plan", return_value=gate_plan
                ),
                mock.patch.object(
                    adapter, "_validate_gate_result", return_value=document
                ),
                mock.patch.object(
                    adapter, "_profile_activation_evidence",
                    return_value=activation,
                ),
            ):
                execution_path = adapter.run_gate_execution(
                    root / "plan.json",
                    gate_plan_path=gate_plan_path,
                    runner=runner,
                    process_auditor=lambda _binary: audit,
                    clock=lambda: next(times),
                    allow_injected_test_evidence=True,
                )
                execution = adapter.load_gate_execution(
                    plan, gate_plan_path, execution_path
                )
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call[0][-1].endswith("raw-result.json") for call in calls))
            self.assertEqual(
                execution["status"], "complete-serial-one-worker-no-retry"
            )
            first = adapter._load_sealed(
                pathlib.Path(execution["variant_receipts"]["baseline"]["path"]),
                adapter.GATE_VARIANT_EXECUTION_SCHEMA,
                "first execution",
            )
            second = adapter._load_sealed(
                pathlib.Path(execution["variant_receipts"]["combined"]["path"]),
                adapter.GATE_VARIANT_EXECUTION_SCHEMA,
                "second execution",
            )
            self.assertLessEqual(
                first["execution"]["finished_at_utc"],
                adapter._load_sealed(
                    pathlib.Path(second["claim"]["path"]),
                    adapter.GATE_EXECUTION_CLAIM_SCHEMA,
                    "second claim",
                )["claimed_at_utc"],
            )

    def test_prelaunch_claim_makes_interrupted_variant_nonretryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, gate_plan = self.fixture(
                root, variants=("baseline",)
            )
            audit = {
                "process_nice": 0,
                "competing_rank4_gate_processes": [],
                "ps_stdout_sha256": "1" * 64,
                "ps_stderr_sha256": "2" * 64,
            }
            patches = (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "_load_gate_request_plan", return_value=gate_plan
                ),
            )
            with patches[0], patches[1]:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    adapter.run_gate_execution(
                        root / "plan.json",
                        gate_plan_path=gate_plan_path,
                        runner=lambda *_args: (_ for _ in ()).throw(
                            RuntimeError("interrupted")
                        ),
                        process_auditor=lambda _binary: audit,
                        clock=lambda: "2026-09-04T00:00:01Z",
                        allow_injected_test_evidence=True,
                    )
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "terminal and cannot retry"
                ):
                    adapter.run_gate_execution(
                        root / "plan.json",
                        gate_plan_path=gate_plan_path,
                        resume=True,
                        runner=lambda *_args: self.fail("retried claimed work"),
                        process_auditor=lambda _binary: audit,
                        clock=lambda: "2026-09-04T00:00:02Z",
                        allow_injected_test_evidence=True,
                    )

    def test_interrupted_variant_seals_idempotent_metric_free_abandonment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, gate_plan = self.fixture(
                root, variants=("baseline",)
            )
            audit = {
                "process_nice": 0,
                "competing_rank4_gate_processes": [],
                "ps_stdout_sha256": "1" * 64,
                "ps_stderr_sha256": "2" * 64,
            }
            with (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "_load_gate_request_plan", return_value=gate_plan
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    adapter.run_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        runner=lambda *_args: (_ for _ in ()).throw(
                            RuntimeError("interrupted")
                        ),
                        process_auditor=lambda _binary: audit,
                        clock=lambda: "2026-09-04T00:00:01Z",
                        allow_injected_test_evidence=True,
                    )
                invalid_receipt = (
                    root / "executions/baseline/receipt.json"
                )
                invalid_receipt.write_bytes(b"interrupted receipt")
                path = adapter.abandon_gate_execution(
                    root / "plan.json", gate_plan_path=gate_plan_path,
                    variant="baseline",
                    abandoned_at_utc="2026-09-04T00:00:02Z",
                    process_auditor=lambda _binary: audit,
                    allow_injected_test_evidence=True,
                )
                self.assertEqual(
                    adapter.abandon_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        variant="baseline",
                        abandoned_at_utc="2026-09-04T00:00:03Z",
                        process_auditor=lambda _binary: audit,
                        allow_injected_test_evidence=True,
                    ),
                    path,
                )
                value = adapter.validate_gate_abandonment(
                    plan, gate_plan_path, path
                )
                self.assertFalse(value["partial_metrics_read"])
                self.assertFalse(value["improvement_counted"])
                self.assertFalse(value["retry_authorized"])
                self.assertIsNotNone(value["claim"])
                self.assertEqual(
                    value["invalid_receipt"], adapter._record(invalid_receipt)
                )
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "abandon.*cannot retry"
                ):
                    adapter.run_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        resume=True,
                        runner=lambda *_args: self.fail("retried spent gate"),
                        process_auditor=lambda _binary: audit,
                        clock=lambda: "2026-09-04T00:00:04Z",
                        allow_injected_test_evidence=True,
                    )

    def test_complete_variant_cannot_be_abandoned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, gate_plan = self.fixture(
                root, variants=("baseline",)
            )
            audit = {
                "process_nice": 0,
                "competing_rank4_gate_processes": [],
                "ps_stdout_sha256": "1" * 64,
                "ps_stderr_sha256": "2" * 64,
            }
            def runner(_command, _repository, _environment, output):
                output.write_bytes(b"raw")
                return {
                    "pid": 123, "process_nice": 0, "returncode": 0,
                    "stderr_bytes": 0,
                    "stderr_sha256": adapter.sha256_bytes(b""),
                    "stdout_policy": "discarded-duplicate-of-raw-result",
                }
            with (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "_load_gate_request_plan", return_value=gate_plan
                ),
                mock.patch.object(
                    adapter, "_validate_gate_result",
                    return_value={"result": {"passed": True}},
                ),
                mock.patch.object(
                    adapter, "_profile_activation_evidence", return_value={}
                ),
            ):
                adapter.run_gate_execution(
                    root / "plan.json", gate_plan_path=gate_plan_path,
                    runner=runner, process_auditor=lambda _binary: audit,
                    clock=iter([
                        "2026-09-04T00:00:01Z",
                        "2026-09-04T00:00:02Z",
                        "2026-09-04T00:00:03Z",
                    ]).__next__,
                    allow_injected_test_evidence=True,
                )
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "cannot be abandoned"
                ):
                    adapter.abandon_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        variant="baseline",
                        abandoned_at_utc="2026-09-04T00:00:04Z",
                        process_auditor=lambda _binary: audit,
                        allow_injected_test_evidence=True,
                    )

    def test_abandonment_rejects_an_active_campaign_gate_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, _gate_plan = self.fixture(
                root, variants=("baseline",)
            )
            lock_path = root / ".rank4-teacher-heavy-stage.lock"
            plan["execution_authority"]["heavy_stage_lock"] = str(lock_path)
            descriptor = adapter.os.open(
                lock_path, adapter.os.O_CREAT | adapter.os.O_RDWR, 0o600
            )
            adapter.fcntl.flock(
                descriptor, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB
            )
            try:
                with (
                    mock.patch.object(
                        adapter, "load_training_plan", return_value=plan
                    ),
                    self.assertRaisesRegex(
                        adapter.TeacherTrainingError,
                        "another Rank-4 campaign heavy stage is active",
                    ),
                ):
                    adapter.abandon_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        variant="baseline",
                        abandoned_at_utc="2026-09-04T00:00:04Z",
                    )
            finally:
                adapter.fcntl.flock(descriptor, adapter.fcntl.LOCK_UN)
                adapter.os.close(descriptor)
            self.assertFalse(
                (root / "executions/baseline/aborted.json").exists()
            )

    def test_abandonment_rejects_a_still_running_gate_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan, gate_plan_path, gate_plan = self.fixture(
                root, variants=("baseline",)
            )
            clean = {
                "process_nice": 0,
                "competing_rank4_gate_processes": [],
                "ps_stdout_sha256": "1" * 64,
                "ps_stderr_sha256": "2" * 64,
            }
            active = {
                **clean,
                "competing_rank4_gate_processes": [{
                    "pid": 123, "nice": 0, "executable": "/tmp/gate",
                }],
            }
            with (
                mock.patch.object(adapter, "load_training_plan", return_value=plan),
                mock.patch.object(
                    adapter, "_load_gate_request_plan", return_value=gate_plan
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    adapter.run_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        runner=lambda *_args: (_ for _ in ()).throw(
                            RuntimeError("interrupted")
                        ),
                        process_auditor=lambda _binary: clean,
                        clock=lambda: "2026-09-04T00:00:01Z",
                        allow_injected_test_evidence=True,
                    )
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "not clean"
                ):
                    adapter.abandon_gate_execution(
                        root / "plan.json", gate_plan_path=gate_plan_path,
                        variant="baseline",
                        abandoned_at_utc="2026-09-04T00:00:02Z",
                        process_auditor=lambda _binary: active,
                        allow_injected_test_evidence=True,
                    )
            self.assertFalse(
                (root / "executions/baseline/aborted.json").exists()
            )

    def test_full_qualification_opens_disjoint_bank_after_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "candidate.cpp"
            source.write_bytes(b"int main(){return 0;}\n")
            binary = root / "candidate.rank4-gate"
            binary.write_bytes(b"binary")
            binary.chmod(0o755)
            runtime = root / "runtime.json"
            runtime.write_bytes(b"runtime")
            base_source = root / "base.cpp"
            base_source.write_bytes(source.read_bytes())
            ab_bank_path = root / "ab-bank.json"
            ab_bank_path.write_bytes(b"ab-bank")
            qualification_bank_path = root / "qualification-bank.json"
            qualification_bank_path.write_bytes(b"qualification-bank")
            gate_tsv = root / "qualification.tsv"
            gate_tsv.write_bytes(b"qualification-tsv")
            ab_request_path = root / "ab-request.json"
            ab_request = adapter._write_sealed(ab_request_path, {
                "schema": adapter.SCREEN_REQUEST_SCHEMA,
                "runtime_body_sha256": "1" * 64,
                "runtime_payload_sha256": "2" * 64,
                "base_model_source": adapter._record(base_source),
                "compiler": {"fixture": True},
                "binary_reference": {"fixture": True},
            })
            candidate = {
                "ranking_weight": 0.10,
                "seed": 20260907,
                "adaptation_contract": dict(
                    adapter.challenger.STANDARD_ADAPTATION_CONTRACT
                ),
                "qat_profile": "standard-v1",
                "qat_profile_contract": {},
                "qat_execution_evidence_sha256": "3" * 64,
                "hard_state_density": hard_state_density(),
                "runtime": adapter._record(runtime),
                "search_throughput_profile": "standard-v1",
                "candidate_search_profile": "standard-v1",
                "search_variant": "baseline",
                "standard_base_variant": "baseline",
                "search_treatment": False,
                "compile_time_macros": list(
                    adapter.SEARCH_VARIANTS["baseline"]
                ),
                "source": adapter._record(source),
                "binary": adapter._record(binary),
                "source_is_default_for_variant": True,
            }
            (root / "ab-plan.json").write_bytes(b"ab-plan")
            selection = {
                "body_sha256": "4" * 64,
                "selected_at_utc": "2026-09-04T00:00:01Z",
                "selected_before_qualification_bank_read": True,
                "qualification_bank_read": False,
                "search_ab_gate_plan": adapter._record(root / "ab-plan.json"),
                "selected_candidate": candidate,
            }
            (root / "selection.json").write_bytes(b"selection")
            ab_gate = {
                "bank": {"manifest": adapter._record(ab_bank_path)},
                "requests": [{
                    "variant": "baseline",
                    "request": adapter._record(ab_request_path),
                }],
            }
            qualification_bank = {"openings": [{"fixture": True}]}
            qualification_bank_record = {
                "manifest": adapter._record(qualification_bank_path),
                "manifest_body_sha256": "5" * 64,
                "gate_tsv": adapter._record(gate_tsv),
                "pairs": adapter.FULL_PAIRS,
                "classification": "fresh-unprotected",
            }
            plan = {
                "campaign_id": "fixture-campaign",
                "attempt": 1,
                "phase": "full",
                "body_sha256": "6" * 64,
                "search_throughput_profile": "standard-v1",
                "execution_authority": {
                    "production_allowlist_enforced": False,
                    "heavy_stage_lock": None,
                },
                "build_source_closure": {
                    "manifest": {"sha256": "7" * 64},
                    "repository_commit": "8" * 40,
                    "closure_sha256": "9" * 64,
                },
                "outputs": {
                    "full_qualification_banks": str(root / "banks"),
                    "development_fingerprints": str(root / "fingerprints"),
                    "gate_requests": str(root / "requests"),
                    "full_qualification_executions": str(root / "executions"),
                    "full_qualification_execution_reference": str(
                        root / "execution-reference.json"
                    ),
                },
            }
            order = []
            inputs = mock.Mock(successor_rankings=object())

            def load_selection(*_args, **_kwargs):
                order.append("selection")
                return selection

            def open_bank(*_args, **_kwargs):
                order.append("bank")
                return qualification_bank, gate_tsv, qualification_bank_record

            with (
                mock.patch.object(
                    adapter, "load_full_search_selection",
                    side_effect=load_selection,
                ),
                mock.patch.object(adapter, "load_gate_plan", return_value=ab_gate),
                mock.patch.object(
                    adapter.challenger.openings, "validate_bank",
                    return_value={"openings": [{"fixture": "ab"}]},
                ),
                mock.patch.object(adapter, "training_context", return_value=(
                    object(), inputs, object(),
                )),
                mock.patch.object(adapter, "_bank_input", side_effect=open_bank),
                mock.patch.object(
                    adapter, "_bank_fingerprints",
                    side_effect=[({"a"}, {"fa"}), ({"b"}, {"fb"})],
                ),
                mock.patch.object(
                    adapter, "_phase_game_state_fingerprints", return_value=set(),
                ),
                mock.patch.object(
                    adapter, "_gate_freshness_audit", return_value={"passed": True},
                ),
                mock.patch.object(
                    adapter, "_development_state_fingerprints", return_value={"b"},
                ),
            ):
                body = adapter._full_qualification_body(
                    plan,
                    selection_path=root / "selection.json",
                    bank_path=qualification_bank_path,
                    prepared_at_utc="2026-09-04T00:00:02Z",
                )
            self.assertEqual(order[:2], ["selection", "bank"])
            self.assertEqual(body["prepared_at_utc"], "2026-09-04T00:00:02Z")
            self.assertTrue(body["bank_disjointness"]["passed"])
            self.assertEqual(body["expected_game_volume"]["total_games"], 1_000)
            request = adapter._load_sealed(
                pathlib.Path(body["requests"][0]["request"]["path"]),
                adapter.SCREEN_REQUEST_SCHEMA,
                "qualification request",
            )
            self.assertEqual(request["gate_purpose"], "full-qualification")
            self.assertEqual(
                request["configuration"]["minimum_candidate_wins"], 550
            )
            self.assertEqual(
                request["configuration"]["minimum_wins_per_color"], 265
            )


class BuildSourceClosureTests(unittest.TestCase):
    def test_training_roster_extends_pipeline_and_every_submission_member(self):
        required = set(adapter.required_build_sources())
        self.assertTrue(
            required <= set(adapter.challenger._campaign_source_paths())
        )
        self.assertTrue(
            set(adapter.pipeline.PIPELINE_REQUIRED_BUILD_SOURCES) <= required
        )
        members = {
            line.strip()
            for line in adapter.SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        self.assertTrue(members <= required)
        self.assertTrue({
            "tools/compact_value_bfm_teacher_training.py",
            "submissions/codingame/bots/compact_value_bfm/export_submission.py",
            "submissions/codingame/bots/compact_value_bfm/rank4_gate_support.py",
            "submissions/codingame/bots/compact_value_bfm/rank4_gate.cpp",
            "submissions/codingame/bots/compact_value_bfm/inference_probe.cpp",
            "submissions/codingame/bots/compact_value_bfm/submission.json",
            "submissions/codingame/bots/compact_value_bfm/sources.txt",
            "submissions/codingame/bots/rank_4/submission.cpp",
        } <= required)

    def test_training_closure_is_a_verified_superset_of_pipeline_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            binary = root / "producer"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            producer_records = {
                role: adapter.challenger._regular(binary)
                for role in adapter.challenger.BUILD_BINARY_ROLES
            }
            body = {
                "schema": adapter.challenger.BUILD_MANIFEST_SCHEMA,
                "campaign_id": adapter.challenger.CAMPAIGN_ID,
                "status": "clean-source-compiler-binaries-frozen",
                "created_at_utc": "2026-09-04T00:00:00Z",
                "repository": adapter.challenger._repository_identity(),
                "source_closure": {
                    relative: adapter.challenger._regular(
                        adapter.REPOSITORY / relative
                    )
                    for relative in adapter.required_build_sources()
                },
                "compiler": adapter.challenger._compiler_identity(),
                "binaries": {
                    role: {**record, "executable": True}
                    for role, record in producer_records.items()
                },
                "build_contract": {
                    "system": "cmake",
                    "configuration": "Release",
                    "language_standard": "c++20",
                    "sources_clean": True,
                    "binaries_built_after_source_freeze": True,
                },
            }
            manifest = root / "build-manifest.json"
            adapter.challenger.qualification.write_sealed(manifest, body)
            campaign = {
                "plan": {"outputs": {"input_directory": str(root)}},
                "inputs": {},
            }
            phase = {
                "phase": {
                    "attempt_inputs": {
                        "build_manifest": adapter.challenger._regular(manifest)
                    },
                    "producer_binaries": producer_records,
                }
            }
            pipeline_closure = (
                adapter.challenger.verify_phase_build_source_closure(
                    required_sources=(
                        adapter.pipeline.PIPELINE_REQUIRED_BUILD_SOURCES
                    ),
                    campaign_context=campaign,
                    phase_context=phase,
                )
            )
            training_closure = (
                adapter.challenger.verify_phase_build_source_closure(
                    required_sources=adapter.required_build_sources(),
                    campaign_context=campaign,
                    phase_context=phase,
                )
            )
            adapter._validate_pipeline_build_subset(
                training_closure, pipeline_closure
            )
            self.assertEqual(
                adapter._revalidate_stored_build_source_closure(
                    training_closure, pipeline_closure=pipeline_closure
                ),
                training_closure,
            )
            self.assertEqual(
                adapter.challenger.validate_build_source_closure_evidence(
                    training_closure,
                    required_sources=adapter.required_build_sources(),
                ),
                training_closure,
            )
            tampered = copy.deepcopy(training_closure)
            tampered["closure_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                adapter.challenger.ChallengerError, "digest changed"
            ):
                adapter.challenger.validate_build_source_closure_evidence(
                    tampered,
                    required_sources=adapter.required_build_sources(),
                )

    def test_load_rechecks_build_closure_even_without_input_revalidation(self):
        root = pathlib.Path(tempfile.gettempdir()).resolve() / (
            "teacher-training-closure-fixture"
        )
        pipeline_authority = {
            "production_allowlist_enforced": False,
            "campaign_derived_output_base": None,
            "context_hooks_injected": False,
            "injected_test_evidence_authorized": False,
            "build_source_closure_sha256": None,
            "heavy_stage_lock": None,
        }
        plan = {
            "outputs": adapter._phase_paths(root),
            "phase": "pilot",
            "adaptation_contract": dict(
                adapter.challenger.STANDARD_ADAPTATION_CONTRACT
            ),
            "search_throughput_profile": "standard-v1",
            "architecture": {
                "name": "capacity-12x8",
                "dimensions": [6301, 12, 8, 1],
                "biases": False,
                "activations": list(adapter.trainer.ACTIVATIONS),
                "quantization_bits": 3,
                "policy_head": False,
            },
            "source_policy": {
                "limit_exclusive": adapter.SOURCE_LIMIT_EXCLUSIVE,
                "reserve_target": adapter.SOURCE_RESERVE_TARGET,
                "maximum_for_reserve_target": adapter.SOURCE_MAXIMUM_FOR_TARGET,
                "deterministic_compactor_required": True,
            },
            "search_variants": {
                name: list(macros)
                for name, macros in adapter.SEARCH_VARIANTS.items()
            },
            "pilot_admission": None,
            "training": {
                "ranking_weights": list(adapter.PILOT_WEIGHTS),
                "fixed_seeds": list(adapter.trainer.FIXED_SEEDS),
                "seed_workers": 2,
                "new_rows_per_batch": 64,
                "anchor_rows_per_batch": 192,
                "float_warmup_epochs": 1,
                "float_learning_rate": 0.00006,
                "qat_epochs": 4,
                "qat_profile": "standard-v1",
                "qat_profile_contract": adapter.trainer.qat_profile_contract(
                    "standard-v1"
                ),
                "native_thread_environment": dict(
                    adapter.trainer.NATIVE_THREAD_ENVIRONMENT
                ),
                "ranking_epoch_schedule": (
                    "balanced-full-weighted-pool-permutation-per-epoch-v1"
                ),
                "ranking_group_microbatch_objective": (
                    "mean-of-gap-normalized-group-losses"
                ),
                "ranking_lambda_application": "once-after-group-mean",
                "protected_tests_opened": False,
            },
            "campaign_plan": {},
            "phase_reference": {},
            "pipeline_plan": {},
            "final_pipeline_receipt": {},
            "initial_checkpoint": {},
            "build_source_closure": {},
            "execution_authority": {
                "production_allowlist_enforced": False,
                "campaign_derived_output_base": None,
                "pipeline_execution_authority": pipeline_authority,
                "build_source_closure_sha256": None,
                "injected_test_evidence_authorized": False,
                "heavy_stage_lock": None,
            },
            "pipeline_body_sha256": "a" * 64,
        }
        with (
            mock.patch.object(adapter, "_load_sealed", return_value=plan),
            mock.patch.object(adapter, "_validate_record", return_value=root / "x"),
            mock.patch.object(
                adapter.challenger,
                "validate_campaign",
                return_value={
                    "plan": {},
                    "inputs": {"production_allowlist_enforced": False},
                },
            ),
            mock.patch.object(
                adapter.challenger,
                "validate_phase_reference",
                return_value={
                    "phase": {
                        "adaptation_contract": dict(
                            adapter.challenger.STANDARD_ADAPTATION_CONTRACT
                        )
                    }
                },
            ),
            mock.patch.object(
                adapter.pipeline,
                "load_pipeline",
                return_value={
                    "body_sha256": "a" * 64,
                    "build_source_closure": {},
                    "execution_authority": pipeline_authority,
                    "adaptation_contract": dict(
                        adapter.challenger.STANDARD_ADAPTATION_CONTRACT
                    ),
                },
            ),
            mock.patch.object(
                adapter,
                "_revalidate_stored_build_source_closure",
                side_effect=adapter.TeacherTrainingError("closure drift"),
            ) as revalidate,
        ):
            with self.assertRaisesRegex(
                adapter.TeacherTrainingError, "closure drift"
            ):
                adapter.load_training_plan(
                    root / "training-plan.json", revalidate_inputs=False
                )
        revalidate.assert_called_once()


class ConfidenceAndResumeTests(unittest.TestCase):
    def test_recomputed_full_admission_requires_canonical_retention(self):
        control = arm(0.0, 0.10, 0.020)
        candidate = arm(0.10, 0.08, 0.020)
        document = passing_full_gate_document()
        summary_value = adapter._result_summary(document)
        selection = {
            "selected_model": {
                "ranking_weight": 0.10,
                "offline_eligible": True,
                "diagnostic_only": False,
            },
            "arms": [control, candidate],
            "model_selection": {
                "comparisons": [{
                    "ranking_weight": 0.10,
                    "mean_teacher_regret_reduction_fraction": 0.20,
                }],
            },
        }
        evidence = {
            "selected_variant": "baseline",
            "summaries": {"baseline": {"candidate_wins": 0}},
            "requests": {"baseline": {"body_sha256": "a" * 64}},
            "documents": {"baseline": {"games": []}},
        }
        self.assertTrue(adapter._recomputed_phase_admission(
            phase="full", selection=selection, search_evidence=evidence,
            qualification_evidence={
                "request": {"body_sha256": "b" * 64},
                "document": document,
                "summary": summary_value,
            },
        ))
        selection["selected_model"].update({
            "offline_eligible": False,
            "diagnostic_only": True,
        })
        self.assertFalse(adapter._recomputed_phase_admission(
            phase="full", selection=selection, search_evidence=evidence,
            qualification_evidence={
                "request": {"body_sha256": "b" * 64},
                "document": document,
                "summary": summary_value,
            },
        ))
        selection["selected_model"].update({
            "offline_eligible": True,
            "diagnostic_only": False,
        })
        candidate["offline_gate_passed"] = False
        self.assertFalse(adapter._recomputed_phase_admission(
            phase="full", selection=selection, search_evidence=evidence,
            qualification_evidence={
                "request": {"body_sha256": "b" * 64},
                "document": document,
                "summary": summary_value,
            },
        ))

    def test_admission_loader_deep_binds_candidate_results_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            qat_profile = "standard-v1"
            qat_contract = adapter.trainer.qat_profile_contract(qat_profile)
            qat_execution = {"fixture": "sealed-QAT-evidence"}
            adaptation_contract = dict(
                adapter.challenger.STANDARD_ADAPTATION_CONTRACT
            )

            def raw(name, payload=b"fixture"):
                path = root / name
                path.write_bytes(payload)
                return path

            build_source_closure_sha256 = "3" * 64
            training_plan = root / "training-plan.json"
            training_plan_document = adapter._write_sealed(training_plan, {
                "schema": adapter.PLAN_SCHEMA,
                "campaign_id": "fixture-campaign",
                "adaptation_contract": adaptation_contract,
                "search_throughput_profile": "standard-v1",
                "search_variants": {
                    name: list(macros)
                    for name, macros in adapter.SEARCH_VARIANTS.items()
                },
                "training": {
                    "qat_profile": qat_profile,
                    "qat_profile_contract": qat_contract,
                },
                "build_source_closure": {
                    "closure_sha256": build_source_closure_sha256
                },
            })
            pipeline_plan = raw("pipeline-plan.json")
            final_receipt = raw("final-receipt.json")
            training_selection = root / "training-selection.json"
            selection_document = adapter._write_sealed(training_selection, {
                "schema": adapter.SELECTION_SCHEMA,
                "plan_body_sha256": "a" * 64,
                "adaptation_contract": adaptation_contract,
                "search_throughput_profile": "standard-v1",
                "active_search_variant_roster": list(
                    adapter.SEARCH_VARIANT_ORDER
                ),
                "arms": [
                    {
                        "ranking_weight": 0.0,
                        "offline_gate_passed": True,
                        "metrics": {
                            "float_vs_quantized_action_flip_rate": 0.020,
                        },
                    },
                    {
                        "ranking_weight": 0.10,
                        "offline_gate_passed": True,
                        "metrics": {
                            "float_vs_quantized_action_flip_rate": 0.024,
                        },
                    },
                ],
                "model_selection": {
                    "comparisons": [{
                        "ranking_weight": 0.10,
                        "mean_teacher_regret_reduction_fraction": 0.20,
                    }],
                },
                "selected_model": {
                    "ranking_weight": 0.10,
                    "seed": 20260907,
                    "adaptation_contract": adaptation_contract,
                    "search_throughput_profile": "standard-v1",
                    "active_search_variant_roster": list(
                        adapter.SEARCH_VARIANT_ORDER
                    ),
                    "qat_profile": qat_profile,
                    "qat_profile_contract": qat_contract,
                    "qat_execution_evidence": qat_execution,
                    "hard_state_density": hard_state_density(),
                    "offline_eligible": True,
                    "diagnostic_only": False,
                },
            })
            gate_plan_path = root / "gate-plan.json"
            gate_document = adapter._write_sealed(gate_plan_path, {
                "schema": adapter.GATE_PLAN_SCHEMA,
                "plan_body_sha256": "a" * 64,
                "search_throughput_profile": "standard-v1",
                "active_search_variant_roster": ["baseline"],
                "expected_game_volume": adapter._expected_gate_game_volume(
                    ("baseline",), pairs_per_variant=adapter.PILOT_PAIRS
                ),
            })
            result_path = raw("gate-result.json")
            results = {"baseline": adapter._record(result_path)}
            runtime = raw("runtime.json")
            source = raw("source.cpp", b"int main(){return 0;}\n")
            binary = raw("binary", b"binary")
            selected = {
                "ranking_weight": 0.10,
                "seed": 20260907,
                "adaptation_contract": adaptation_contract,
                "qat_profile": qat_profile,
                "qat_profile_contract": qat_contract,
                "qat_execution_evidence_sha256": adapter.sha256_bytes(
                    adapter.canonical_json_bytes(qat_execution)
                ),
                "hard_state_density": hard_state_density(),
                "runtime": adapter._record(runtime),
                "offline_eligible": True,
                "diagnostic_only": False,
                "search_throughput_profile": "standard-v1",
                "candidate_search_profile": "standard-v1",
                "search_variant": "baseline",
                "standard_base_variant": "baseline",
                "search_treatment": False,
                "compile_time_macros": list(
                    adapter.SEARCH_VARIANTS["baseline"]
                ),
                "source": adapter._record(source),
                "binary": adapter._record(binary),
                "source_is_default_for_variant": True,
            }
            development = root / "development.json"
            development_document = adapter._write_sealed(development, {
                "schema": adapter.challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
                "fingerprints": ["1" * 64],
            })
            metrics = {
                "canonical_retention_passed": True,
                "offline_model_eligible": True,
                "diagnostic_only": False,
                "quantized_action_flip_rate": 0.024,
                "scalar_control_action_flip_rate": 0.020,
                "rank4_win_rate": 0.525,
                "rank4_absolute_margin_pp": 2.500000000000002,
                "strength_delta_pp": 2.500000000000002,
                "ranking_validation_groups": 125,
                "qat_profile": qat_profile,
                "candidate_search_profile": "standard-v1",
                "search_ab": {
                    "retained_variants": ["baseline"],
                },
                "development_gate_game_volume": (
                    adapter._expected_gate_game_volume(
                        ("baseline",), pairs_per_variant=adapter.PILOT_PAIRS
                    )
                ),
            }
            phase_evidence_path = root / "phase-evidence.json"
            phase_evidence_document = adapter._write_sealed(
                phase_evidence_path,
                {
                    "schema": adapter.challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
                    "campaign_id": "fixture-campaign",
                    "attempt": 1,
                    "phase": "pilot",
                    "candidate": {
                        "runtime_sha256": selected["runtime"]["sha256"],
                        "source_sha256": selected["source"]["sha256"],
                    },
                    "metrics_sha256": adapter.sha256_bytes(
                        adapter.canonical_json_bytes(metrics)
                    ),
                    "protected_or_live_metrics_read": False,
                    "qat_profile": qat_profile,
                    "qat_profile_contract": qat_contract,
                    "adaptation_contract": adaptation_contract,
                    "search_throughput_profile": "standard-v1",
                    "candidate_search_profile": "standard-v1",
                    "gate_execution": None,
                    "full_search_selection": None,
                    "full_qualification_plan": None,
                    "full_qualification_execution": None,
                    "qualification_result": None,
                    "injected_test_results": True,
                    "active_search_variant_roster": ["baseline"],
                    "evidence_closure": {
                        "training_plan": {
                            **adapter._record(training_plan),
                            "schema": adapter.PLAN_SCHEMA,
                            "body_sha256": adapter._load_sealed(
                                training_plan, adapter.PLAN_SCHEMA, "fixture"
                            )["body_sha256"],
                        },
                        "pipeline_plan": adapter._record(pipeline_plan),
                        "finalized_pipeline_receipt": adapter._record(
                            final_receipt
                        ),
                        "training_selection": {
                            **adapter._record(training_selection),
                            "schema": adapter.SELECTION_SCHEMA,
                            "body_sha256": selection_document["body_sha256"],
                        },
                        "gate_plan": {
                            **adapter._record(gate_plan_path),
                            "schema": adapter.GATE_PLAN_SCHEMA,
                            "body_sha256": gate_document["body_sha256"],
                        },
                        "gate_results": results,
                        "gate_execution": None,
                        "full_search_selection": None,
                        "full_qualification_plan": None,
                        "full_qualification_execution": None,
                        "qualification_result": None,
                        "selected_candidate": selected,
                        "input_audit_sha256": "2" * 64,
                        "build_source_closure_sha256": (
                            build_source_closure_sha256
                        ),
                        "protected_tests_opened": False,
                    },
                },
            )
            admission_body = {
                "schema": adapter.ADMISSION_SCHEMA,
                "campaign_id": "fixture-campaign",
                "attempt": 1,
                "phase": "pilot",
                "plan_body_sha256": "a" * 64,
                "gate_plan": adapter._record(gate_plan_path),
                "gate_plan_body_sha256": gate_document["body_sha256"],
                "gate_execution": None,
                "gate_execution_body_sha256": None,
                "injected_test_results": True,
                "full_search_selection": None,
                "full_qualification_plan": None,
                "full_qualification_execution": None,
                "qualification_result": None,
                "training_selection": adapter._record(training_selection),
                "finalized_pipeline_receipt": adapter._record(final_receipt),
                "results": results,
                "summaries": {"baseline": {"candidate_wins": 105}},
                "selected_candidate": selected,
                "metrics": metrics,
                "strength_delta_pp": 2.500000000000002,
                "teacher_regret_reduction_fraction": 0.2,
                "qat_profile": qat_profile,
                "qat_profile_contract": qat_contract,
                "adaptation_contract": adaptation_contract,
                "search_throughput_profile": "standard-v1",
                "candidate_search_profile": "standard-v1",
                "active_search_variant_roster": ["baseline"],
                "development_exclusion": {
                    **adapter._record(development),
                    "schema": adapter.challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
                    "body_sha256": development_document["body_sha256"],
                },
                "phase_outcome_evidence": {
                    **adapter._record(phase_evidence_path),
                    "schema": adapter.challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
                    "body_sha256": phase_evidence_document["body_sha256"],
                },
                "admitted": True,
                "next_route": "materialize-full",
                "protected_or_live_metrics_read": False,
            }
            admission = root / "admission.json"
            adapter._write_sealed(admission, admission_body)
            with mock.patch.object(
                adapter.challenger,
                "validate_build_source_closure_evidence",
                return_value={},
            ), mock.patch.object(
                adapter,
                "load_training_selection",
                return_value=selection_document,
            ), mock.patch.object(
                adapter,
                "_validate_admission_search_evidence",
                return_value={
                    "gate_plan": adapter._load_sealed(
                        gate_plan_path, adapter.GATE_PLAN_SCHEMA, "fixture gate"
                    ),
                    "selected_variant": "baseline",
                    "summaries": admission_body["summaries"],
                },
            ), mock.patch.object(
                adapter, "_validate_admission_gate_execution", return_value=None,
            ), mock.patch.object(
                adapter, "_validate_admission_full_qualification", return_value=None,
            ), mock.patch.object(
                adapter, "_recomputed_phase_admission", return_value=True,
            ), mock.patch.object(
                adapter.trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ):
                self.assertEqual(
                    adapter.load_phase_admission(admission)["selected_candidate"],
                    selected,
                )
                self.assertEqual(
                    adapter.load_phase_admission(admission)["adaptation_contract"],
                    adaptation_contract,
                )
                flipped = root / "flipped-admission.json"
                adapter._write_sealed(flipped, {
                    **admission_body,
                    "admitted": False,
                    "next_route": "open-next-leakage-isolated-attempt",
                })
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError,
                    "decision differs from sealed evidence",
                ):
                    adapter.load_phase_admission(flipped)

            bad_body = dict(admission_body)
            bad_selected = dict(selected)
            bad_selected["source"] = {**selected["source"], "sha256": "f" * 64}
            bad_body["selected_candidate"] = bad_selected
            bad = root / "bad-admission.json"
            adapter._write_sealed(bad, bad_body)
            with mock.patch.object(
                adapter.challenger,
                "validate_build_source_closure_evidence",
                return_value={},
            ), mock.patch.object(
                adapter,
                "load_training_selection",
                return_value=selection_document,
            ), mock.patch.object(
                adapter,
                "_validate_admission_search_evidence",
                return_value={
                    "gate_plan": adapter._load_sealed(
                        gate_plan_path, adapter.GATE_PLAN_SCHEMA, "fixture gate"
                    ),
                    "selected_variant": "baseline",
                    "summaries": admission_body["summaries"],
                },
            ), mock.patch.object(
                adapter, "_validate_admission_gate_execution", return_value=None,
            ), mock.patch.object(
                adapter, "_validate_admission_full_qualification", return_value=None,
            ), mock.patch.object(
                adapter, "_recomputed_phase_admission", return_value=True,
            ), mock.patch.object(
                adapter.trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ):
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "evidence closure"
                ):
                    adapter.load_phase_admission(bad)

            bad_evidence_body = dict(adapter._load_sealed(
                phase_evidence_path,
                adapter.challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
                "fixture phase evidence",
            ))
            bad_evidence_body.pop("body_sha256")
            bad_evidence_body["evidence_closure"] = {
                **bad_evidence_body["evidence_closure"],
                "build_source_closure_sha256": "f" * 64,
            }
            bad_evidence = root / "bad-closure-phase-evidence.json"
            bad_evidence_document = adapter._write_sealed(
                bad_evidence, bad_evidence_body
            )
            bad_admission_body = {
                **admission_body,
                "phase_outcome_evidence": {
                    **adapter._record(bad_evidence),
                    "schema": adapter.challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
                    "body_sha256": bad_evidence_document["body_sha256"],
                },
            }
            bad_admission = root / "bad-closure-admission.json"
            adapter._write_sealed(bad_admission, bad_admission_body)
            with mock.patch.object(
                adapter.challenger,
                "validate_build_source_closure_evidence",
                return_value={},
            ), mock.patch.object(
                adapter,
                "load_training_selection",
                return_value=selection_document,
            ), mock.patch.object(
                adapter,
                "_validate_admission_search_evidence",
                return_value={
                    "gate_plan": adapter._load_sealed(
                        gate_plan_path, adapter.GATE_PLAN_SCHEMA, "fixture gate"
                    ),
                    "selected_variant": "baseline",
                    "summaries": admission_body["summaries"],
                },
            ), mock.patch.object(
                adapter, "_validate_admission_gate_execution", return_value=None,
            ), mock.patch.object(
                adapter, "_validate_admission_full_qualification", return_value=None,
            ), mock.patch.object(
                adapter, "_recomputed_phase_admission", return_value=True,
            ), mock.patch.object(
                adapter.trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ):
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError, "evidence closure"
                ):
                    adapter.load_phase_admission(bad_admission)

    def test_pilot_and_full_admission_thresholds_are_exact(self):
        zero = {
            name: 0 for name in adapter.challenger.qualification.FAILURE_CATEGORIES
        }
        pilot = {
            "pairs": 100,
            "games": 200,
            "candidate_wins": 105,
            "candidate_color_wins": {"0": 52, "1": 53},
            "failures": zero,
            "operational_failures": 0,
            "unfinished": 0,
        }
        self.assertTrue(adapter.pilot_admission_passes(
            summary=pilot,
            canonical_retention_passed=True,
            regret_reduction=0.10,
            candidate_flip_rate=0.025,
            scalar_control_flip_rate=0.020,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=100,
            comparable_exhaustive_validation_fraction=0.80,
        ))
        self.assertFalse(adapter.pilot_admission_passes(
            summary={**pilot, "candidate_wins": 104},
            canonical_retention_passed=True,
            regret_reduction=0.10,
            candidate_flip_rate=0.025,
            scalar_control_flip_rate=0.020,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=100,
            comparable_exhaustive_validation_fraction=0.80,
        ))
        self.assertFalse(adapter.pilot_admission_passes(
            summary=pilot,
            canonical_retention_passed=True,
            regret_reduction=0.10,
            candidate_flip_rate=0.025,
            scalar_control_flip_rate=0.020,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=99,
            comparable_exhaustive_validation_fraction=0.792,
        ))
        full = {
            "pairs": 500,
            "games": 1_000,
            "candidate_wins": 550,
            "candidate_color_wins": {"0": 275, "1": 275},
            "failures": zero,
            "operational_failures": 0,
            "unfinished": 0,
        }
        self.assertTrue(adapter.full_admission_passes(
            summary=full,
            paired_lower_95=0.5001,
            canonical_retention_passed=True,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=100,
            comparable_exhaustive_validation_fraction=0.80,
        ))
        self.assertFalse(adapter.full_admission_passes(
            summary=full,
            paired_lower_95=0.5,
            canonical_retention_passed=True,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=100,
            comparable_exhaustive_validation_fraction=0.80,
        ))
        self.assertFalse(adapter.full_admission_passes(
            summary={
                **full,
                "candidate_color_wins": {"0": 264, "1": 286},
            },
            paired_lower_95=0.51,
            canonical_retention_passed=True,
            ranking_validation_groups=125,
            comparable_exhaustive_validation_groups=100,
            comparable_exhaustive_validation_fraction=0.80,
        ))

    def test_full_cluster_bootstrap_is_deterministic(self):
        # 600 candidate wins: 100 swept pairs plus 400 split pairs.
        games = []
        for pair in range(adapter.FULL_PAIRS):
            score = 2 if pair < 100 else 1
            for color in (0, 1):
                games.append({
                    "pair_index": pair,
                    "candidate_player": color,
                    "winner": color if color < score else 1 - color,
                    "failure": None,
                })
        document = {"games": games}
        first = adapter.paired_bootstrap_lower_95(
            document, seed_material="a" * 64, samples=2_000
        )
        second = adapter.paired_bootstrap_lower_95(
            document, seed_material="a" * 64, samples=2_000
        )
        self.assertEqual(first, second)
        self.assertGreater(first, 0.5)

    def test_training_receipt_resume_skips_all_seed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            checkpoint = root / "initial.bin"
            checkpoint.write_bytes(b"checkpoint")
            outputs = {
                "runs": str(root / "runs"),
                "sources": str(root / "sources"),
                "source_verification": str(root / "source-verification"),
                "selections": str(root / "selections"),
                "selection_reference": str(root / "selection-reference.json"),
            }
            plan = {
                "campaign_id": "fixture-campaign",
                "attempt": 1,
                "phase": "pilot",
                "adaptation_contract": dict(
                    adapter.challenger.STANDARD_ADAPTATION_CONTRACT
                ),
                "search_throughput_profile": "standard-v1",
                "search_variants": {
                    name: list(macros)
                    for name, macros in adapter.SEARCH_VARIANTS.items()
                },
                "body_sha256": "d" * 64,
                "execution_authority": {
                    "production_allowlist_enforced": False,
                    "heavy_stage_lock": None,
                },
                "source_bundle": {"body_sha256": "e" * 64},
                "initial_checkpoint": adapter._record(checkpoint),
                "training": {
                    "ranking_weights": [0.0, 0.10, 0.25],
                    "qat_profile": "standard-v1",
                    "qat_profile_contract": (
                        adapter.trainer.qat_profile_contract("standard-v1")
                    ),
                },
                "input_audit": {"passed": True},
                "outputs": outputs,
            }
            calls = []

            def run_roster(
                _bundle, _inputs, _architecture, _arm, arm_root,
                weight, _initial, qat_profile, resume,
            ):
                self.assertEqual(qat_profile, "standard-v1")
                calls.append((weight, resume))
                runtime = arm_root / "artifacts/runtime.json"
                float_checkpoint = arm_root / "artifacts/checkpoint.npz"
                runtime.parent.mkdir(parents=True, exist_ok=True)
                runtime.write_bytes(f"runtime-{weight}".encode())
                float_checkpoint.write_bytes(f"float-{weight}".encode())
                regret = {0.0: 0.10, 0.10: 0.08, 0.25: 0.09}[weight]
                return [
                    adapter._sealed({
                        "schema": adapter.trainer.SEED_RECEIPT_SCHEMA,
                        "binding": {
                            "fixture_seed": seed,
                            "fixture_weight": weight,
                        },
                        "architecture": "capacity-12x8",
                        "arm": "search-target",
                        "seed": seed,
                        "qat_profile": "standard-v1",
                        "qat_profile_contract": (
                            adapter.trainer.qat_profile_contract("standard-v1")
                        ),
                        "quantized_training": {"fixture": True},
                        "native_thread_execution": native_thread_execution(),
                        "float_checkpoint": {
                            "path": "artifacts/checkpoint.npz",
                            "sha256": adapter.sha256_file(float_checkpoint),
                            "bytes": float_checkpoint.stat().st_size,
                        },
                        "quantized_runtime": {
                            "path": "artifacts/runtime.json",
                            "sha256": adapter.sha256_file(runtime),
                            "bytes": runtime.stat().st_size,
                        },
                        "float_validation": {
                            "common_adjudicator": {
                                "objective_weighted_huber": 0.04,
                                "sign_accuracy": 0.90,
                            },
                            "canonical_validation": {
                                "objective_weighted_huber": 0.04,
                                "sign_accuracy": 0.90,
                            },
                        },
                        "quantized_validation": {
                            "common_adjudicator": {
                                "objective_weighted_huber": 0.04,
                                "sign_accuracy": 0.90,
                            },
                            "canonical_validation": {
                                "objective_weighted_huber": 0.04,
                                "sign_accuracy": 0.90,
                            },
                            "successor_ranking": {
                                "loss_weight": weight,
                                "groups": 125,
                                "comparable_groups": 110,
                                "mean_teacher_regret": regret,
                                "top1_agreement": 0.75,
                                "float_vs_quantized_action_flip_rate": 0.02,
                                "pairwise_loss": 0.1,
                            },
                        },
                        "offline_gate": {"passed": True},
                        "inference_parity": {"passed": True},
                        "successor_ranking": {
                            "labels_present": True,
                            "loss_weight": weight,
                            "hard_state_density": hard_state_density(),
                            "schedule_execution": {"fixture": True},
                            "float_per_layer_update_evidence": {"passed": True},
                            "qat_per_layer_update_evidence": {"passed": True},
                        },
                    })
                    for seed in adapter.trainer.FIXED_SEEDS
                ]

            def verify_source(_runtime, _source, _dataset, output):
                output.mkdir(parents=True, exist_ok=True)
                standalone = output / "standalone.bin"
                probe = output / "probe.bin"
                standalone.write_bytes(b"standalone")
                probe.write_bytes(b"probe")
                return {
                    "compiled": True,
                    "standalone_binary": adapter._record(standalone),
                    "inference_probe_binary": adapter._record(probe),
                    "states": 4_096,
                    "comparison": (
                        "scalar-float32-hex-vs-maintained-python-scalar"
                    ),
                    "maximum_absolute_error": 0.0,
                    "tolerance": 2e-6,
                    "mismatches": 0,
                    "compiler": {"fixture": True},
                    "passed": True,
                }

            with mock.patch.object(
                adapter, "load_training_plan", return_value=plan
            ), mock.patch.object(
                adapter,
                "training_context",
                return_value=(
                    object(), mock.Mock(common_adjudicator=object()), object()
                ),
            ), mock.patch.object(
                adapter.trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ), mock.patch.object(
                adapter.trainer,
                "validate_successor_schedule_execution",
                return_value={"fixture": True},
            ), mock.patch.object(
                adapter.trainer,
                "training_binding",
                side_effect=lambda _bundle, _inputs, _architecture, _arm, seed,
                _sidecar, weight, _checkpoint, _profile: {
                    "fixture_seed": seed,
                    "fixture_weight": weight,
                },
            ), mock.patch.object(
                adapter,
                "_runtime_source",
                return_value=b"int main(){return 0;}\n",
            ):
                first = adapter.run_training(
                    root / "plan.json",
                    roster_runner=run_roster,
                    renderer=lambda _runtime: b"int main(){return 0;}\n",
                    source_verifier=verify_source,
                    allow_injected_test_evidence=True,
                )
                before = list(calls)
                second = adapter.run_training(
                    root / "plan.json",
                    resume=True,
                    roster_runner=lambda *_args, **_kwargs: self.fail(
                        "resume reran a seed"
                    ),
                    renderer=lambda _runtime: self.fail("resume re-exported source"),
                    source_verifier=lambda *_args: self.fail(
                        "resume reverified source"
                    ),
                    allow_injected_test_evidence=True,
                )
            self.assertEqual(first, second)
            selection = adapter._load_sealed(
                first, adapter.SELECTION_SCHEMA, "fixture selection"
            )
            self.assertEqual(
                selection["adaptation_contract"], plan["adaptation_contract"]
            )
            self.assertEqual(
                selection["selected_model"]["adaptation_contract"],
                plan["adaptation_contract"],
            )
            self.assertTrue(selection["selected_model"]["offline_eligible"])
            self.assertFalse(selection["selected_model"]["diagnostic_only"])
            self.assertEqual(before, [(0.0, False), (0.10, False), (0.25, False)])

            forged_arm = copy.deepcopy(selection)
            forged_arm.pop("body_sha256")
            forged_arm["arms"][1]["float_validation"] = {"forged": True}
            forged_arm_path = root / "forged-arm-selection.json"
            adapter._write_sealed(forged_arm_path, forged_arm)

            forged_seed = copy.deepcopy(selection)
            forged_seed.pop("body_sha256")
            forged_seed["arms"][1]["seed_receipts"][0][
                "offline_gate_passed"
            ] = False
            forged_seed_path = root / "forged-seed-selection.json"
            adapter._write_sealed(forged_seed_path, forged_seed)

            with mock.patch.object(
                adapter,
                "training_context",
                return_value=(
                    object(), mock.Mock(common_adjudicator=object()), object()
                ),
            ), mock.patch.object(
                adapter.trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ), mock.patch.object(
                adapter.trainer,
                "validate_successor_schedule_execution",
                return_value={"fixture": True},
            ), mock.patch.object(
                adapter.trainer,
                "training_binding",
                side_effect=lambda _bundle, _inputs, _architecture, _arm, seed,
                _sidecar, weight, _checkpoint, _profile: {
                    "fixture_seed": seed,
                    "fixture_weight": weight,
                },
            ), mock.patch.object(
                adapter,
                "_runtime_source",
                return_value=b"int main(){return 0;}\n",
            ):
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError,
                    "differs from its copied seed receipts",
                ):
                    adapter.load_training_selection(plan, forged_arm_path)
                with self.assertRaisesRegex(
                    adapter.TeacherTrainingError,
                    "copied seed receipt identity changed",
                ):
                    adapter.load_training_selection(plan, forged_seed_path)


if __name__ == "__main__":
    unittest.main()
