#!/usr/bin/env python3
"""Focused, non-heavy tests for the Rank-4 teacher training adapter."""

from __future__ import annotations

import copy
import hashlib
import pathlib
import tempfile
import unittest
from unittest import mock

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

    def test_combined_search_requires_both_independent_ab_improvements(self):
        selected = adapter.select_pilot_search_variant({
            "baseline": summary(106),
            "no-feature-sort-only": summary(108),
            "single-pass-selection-only": summary(107),
            "combined": summary(109),
        })
        self.assertEqual(selected["selected_variant"], "combined")
        self.assertTrue(
            selected["independent_changes"][
                "combined_supported_by_both_individual_arms"
            ]
        )

        unsupported = adapter.select_pilot_search_variant({
            "baseline": summary(106),
            "no-feature-sort-only": summary(106),
            "single-pass-selection-only": summary(108),
            "combined": summary(112),
        })
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
        plan = {
            "outputs": adapter._phase_paths(root),
            "phase": "pilot",
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
                "protected_tests_opened": False,
            },
            "campaign_plan": {},
            "phase_reference": {},
            "pipeline_plan": {},
            "final_pipeline_receipt": {},
            "initial_checkpoint": {},
            "build_source_closure": {},
            "pipeline_body_sha256": "a" * 64,
        }
        with (
            mock.patch.object(adapter, "_load_sealed", return_value=plan),
            mock.patch.object(adapter, "_validate_record", return_value=root / "x"),
            mock.patch.object(
                adapter.challenger,
                "validate_campaign",
                return_value={"plan": {}},
            ),
            mock.patch.object(
                adapter.challenger,
                "validate_phase_reference",
                return_value={"phase": {}},
            ),
            mock.patch.object(
                adapter.pipeline,
                "load_pipeline",
                return_value={
                    "body_sha256": "a" * 64,
                    "build_source_closure": {},
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
    def test_admission_loader_deep_binds_candidate_results_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def raw(name, payload=b"fixture"):
                path = root / name
                path.write_bytes(payload)
                return path

            build_source_closure_sha256 = "3" * 64
            training_plan = root / "training-plan.json"
            training_plan_document = adapter._write_sealed(training_plan, {
                "schema": adapter.PLAN_SCHEMA,
                "campaign_id": "fixture-campaign",
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
            })
            gate_plan_path = root / "gate-plan.json"
            gate_document = adapter._write_sealed(gate_plan_path, {
                "schema": adapter.GATE_PLAN_SCHEMA,
                "plan_body_sha256": "a" * 64,
            })
            result_path = raw("gate-result.json")
            results = {"baseline": adapter._record(result_path)}
            runtime = raw("runtime.json")
            source = raw("source.cpp", b"int main(){return 0;}\n")
            binary = raw("binary", b"binary")
            selected = {
                "ranking_weight": 0.10,
                "seed": 20260907,
                "runtime": adapter._record(runtime),
                "search_variant": "baseline",
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
            metrics = {"ranking_validation_groups": 125}
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
                "training_selection": adapter._record(training_selection),
                "finalized_pipeline_receipt": adapter._record(final_receipt),
                "results": results,
                "summaries": {"baseline": {"candidate_wins": 105}},
                "selected_candidate": selected,
                "metrics": metrics,
                "strength_delta_pp": 2.5,
                "teacher_regret_reduction_fraction": 0.2,
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
            ):
                self.assertEqual(
                    adapter.load_phase_admission(admission)["selected_candidate"],
                    selected,
                )

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
                "body_sha256": "d" * 64,
                "source_bundle": {"body_sha256": "e" * 64},
                "initial_checkpoint": adapter._record(checkpoint),
                "training": {"ranking_weights": [0.0, 0.10, 0.25]},
                "input_audit": {"passed": True},
                "outputs": outputs,
            }
            calls = []

            def run_roster(
                _bundle, _inputs, _architecture, _arm, arm_root,
                weight, _initial, resume,
            ):
                calls.append((weight, resume))
                runtime = arm_root / "artifacts/runtime.json"
                float_checkpoint = arm_root / "artifacts/checkpoint.npz"
                runtime.parent.mkdir(parents=True, exist_ok=True)
                runtime.write_bytes(f"runtime-{weight}".encode())
                float_checkpoint.write_bytes(f"float-{weight}".encode())
                regret = {0.0: 0.10, 0.10: 0.08, 0.25: 0.09}[weight]
                return [
                    {
                        "architecture": "capacity-12x8",
                        "arm": "search-target",
                        "seed": seed,
                        "body_sha256": f"{seed:064x}"[-64:],
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
                            "float_per_layer_update_evidence": {"passed": True},
                            "qat_per_layer_update_evidence": {"passed": True},
                        },
                    }
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
            ):
                first = adapter.run_training(
                    root / "plan.json",
                    roster_runner=run_roster,
                    renderer=lambda _runtime: b"int main(){return 0;}\n",
                    source_verifier=verify_source,
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
                )
            self.assertEqual(first, second)
            self.assertEqual(before, [(0.0, False), (0.10, False), (0.25, False)])


if __name__ == "__main__":
    unittest.main()
