#!/usr/bin/env python3
"""Focused contracts for the compact value-BFM trainer."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import sys
import tempfile
import threading
import time
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


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compact_value_bfm_train", ROOT / "tools/compact_value_bfm_train.py"
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def active_row(category: int = 0, edge: int | None = None) -> np.ndarray:
    values = [] if edge is None else [edge]
    values.extend(
        trainer.EDGE_COUNT + vertex * trainer.VERTEX_CATEGORIES + category
        for vertex in range(trainer.VERTEX_COUNT)
    )
    return np.asarray(sorted(values), dtype="<u2")


def dataset(
    rows: list[np.ndarray],
    targets: list[float] | None = None,
    *,
    split: str = "train",
    teacher: list[float] | None = None,
    groups: list[str] | None = None,
) -> trainer.Dataset:
    indptr = np.zeros(len(rows) + 1, dtype="<i8")
    for index, row in enumerate(rows):
        indptr[index + 1] = indptr[index] + len(row)
    targets = targets or [(-1.0 if index % 2 else 1.0) for index in range(len(rows))]
    groups = groups or [f"group-{index}" for index in range(len(rows))]
    return trainer.Dataset(
        indptr=indptr,
        indices=np.concatenate(rows).astype("<u2"),
        targets=np.asarray(targets, dtype="<f4"),
        weights=np.ones(len(rows), dtype="<f4"),
        group_ids=np.asarray(
            [hashlib.sha256(value.encode()).digest() for value in groups],
            dtype="V32",
        ),
        split=split,
        source_manifest_sha256="1" * 64,
        source_npz_sha256="2" * 64,
        source_route=f"fixture/{split}.json",
        teacher_predictions=(
            None if teacher is None else np.asarray(teacher, dtype="<f4")
        ),
    )


def bundle_fixture(root: pathlib.Path) -> trainer.FrozenBundle:
    routes = {
        "pilot_search_manifests": [
            "new/search/pilot/train.json",
            "new/search/pilot/validation.json",
            "new/search/pilot/test.json",
        ],
        "full_search_manifests": [
            "new/search/full/train.json",
            "new/search/full/validation.json",
            "new/search/full/test.json",
        ],
        "pilot_rank4_manifests": [
            "new/rank4/pilot/train.json",
            "new/rank4/pilot/validation.json",
            "new/rank4/pilot/test.json",
        ],
        "full_rank4_manifests": [
            "new/rank4/full/train.json",
            "new/rank4/full/validation.json",
            "new/rank4/full/test.json",
        ],
        "canonical_splits": {
            split: [f"canonical/r{index}/{split}.json" for index in range(3)]
            for split in ("train", "validation", "test")
        },
        "common_adjudicator_manifest": "adjudicator/common.json",
    }
    records = []
    for value in (
        *routes["pilot_search_manifests"],
        *routes["full_search_manifests"],
        *routes["pilot_rank4_manifests"],
        *routes["full_rank4_manifests"],
        *routes["canonical_splits"]["train"],
        *routes["canonical_splits"]["validation"],
        *routes["canonical_splits"]["test"],
        routes["common_adjudicator_manifest"],
    ):
        records.append({
            "role": value,
            "relative_path": value,
            "sha256": "0" * 64,
            "bytes": 0,
        })
    manifest = {
        "routes": routes,
        "artifacts": records,
        "body_sha256": "a" * 64,
        "protected_splits": [
            "search:test", "rank4:test", "canonical:test"
        ],
    }
    return trainer.FrozenBundle(root / "bundle-manifest.json", b"", manifest)


class ArchitectureAndScheduleTests(unittest.TestCase):
    def test_exact_architecture_counts_and_capacity_size_rule(self) -> None:
        self.assertEqual(
            trainer.ARCHITECTURES["compact-8x8"].weight_counts["total"], 50_480
        )
        self.assertEqual(
            trainer.ARCHITECTURES["source-neutral-8x16"].weight_counts["total"],
            50_552,
        )
        self.assertEqual(
            trainer.ARCHITECTURES["capacity-12x8"].weight_counts["total"], 75_716
        )
        self.assertTrue(trainer.architecture_deployment_eligible("capacity-12x8", 95_000))
        self.assertFalse(trainer.architecture_deployment_eligible("capacity-12x8", 95_001))
        self.assertFalse(trainer.architecture_deployment_eligible("capacity-12x8"))

    def test_exact_paired_64_192_schedule_and_anchor_coverage(self) -> None:
        new, anchor = trainer.mixed_epoch_schedule(
            241_365, 997_914, seed=20260907, epoch=1
        )
        self.assertEqual(len(new), 3_772 * 64)
        self.assertEqual(len(anchor), 3_772 * 192)
        self.assertEqual(len(set(map(int, new[:241_365]))), 241_365)
        self.assertEqual(
            trainer.anchor_coverage_complete_epoch(241_365, 997_914), 2
        )
        again = trainer.mixed_epoch_schedule(
            241_365, 997_914, seed=20260907, epoch=1
        )
        self.assertTrue(np.array_equal(new, again[0]))
        self.assertTrue(np.array_equal(anchor, again[1]))

    def test_sources_are_normalized_independently_to_quarter_three_quarters(self) -> None:
        weights = trainer.independently_normalized_mixed_weights(
            np.arange(1, 65, dtype=np.float32),
            np.arange(1, 193, dtype=np.float32),
        )
        self.assertAlmostEqual(float(np.sum(weights[:64])), 0.25, places=6)
        self.assertAlmostEqual(float(np.sum(weights[64:])), 0.75, places=6)


class LossAndLeakageTests(unittest.TestCase):
    def test_teacher_assisted_is_equal_huber_not_averaged_target(self) -> None:
        prediction = np.asarray([0.8], dtype=np.float32)
        stored = np.asarray([-0.8], dtype=np.float32)
        teacher = np.asarray([0.7], dtype=np.float32)
        weights = np.asarray([1.0], dtype=np.float32)
        loss, gradient, report = trainer.arm_loss_gradient(
            "teacher-assisted", prediction, stored, weights, teacher
        )
        first = trainer._weighted_huber_loss_gradient(
            prediction, stored, weights
        )
        second = trainer._weighted_huber_loss_gradient(
            prediction, teacher, weights
        )
        self.assertAlmostEqual(loss, 0.5 * first[0] + 0.5 * second[0])
        np.testing.assert_array_equal(
            gradient, np.float32(0.5) * first[1] + np.float32(0.5) * second[1]
        )
        self.assertEqual(report["stored_target_loss_share"], 0.5)

    def test_group_and_reflection_leakage_fail_closed(self) -> None:
        base = active_row(0, 3)
        clean = active_row(1, 4)
        train_new = dataset([base])
        anchor = dataset([active_row(2, 5)], groups=["anchor"])
        common = dataset([clean], split="validation", groups=["common"])
        canonical = dataset(
            [active_row(3, 6)], split="validation", groups=["canonical"]
        )
        report = trainer.validate_unprotected_split_isolation(
            train_new, anchor, common, canonical
        )
        self.assertTrue(report["passed"])

        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import jacek_replay_features as features

        reflected = features.reflect_active(base.tolist())
        leaked = dataset(
            [np.asarray(reflected, dtype="<u2")],
            split="validation",
            groups=["different-group"],
        )
        with self.assertRaisesRegex(trainer.TrainingError, "rotate/reflect"):
            trainer.validate_unprotected_split_isolation(
                train_new, anchor, leaked, canonical
            )
        same_group = dataset(
            [clean], split="validation", groups=["group-0"]
        )
        with self.assertRaisesRegex(trainer.TrainingError, "root group"):
            trainer.validate_unprotected_split_isolation(
                train_new, anchor, same_group, canonical
            )

    def test_search_rank4_control_rows_and_weights_must_be_byte_equal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = bundle_fixture(pathlib.Path(temporary))
            fixtures = {}
            for route in (
                *bundle.arm_train_routes("search-target"),
                *bundle.arm_train_routes("rank4-control"),
            ):
                fixture = dataset([active_row(0, 1)], [0.25])
                fixture = trainer.dataclasses.replace(
                    fixture,
                    source_route=route,
                    source_manifest_sha256=hashlib.sha256(route.encode()).hexdigest(),
                )
                fixtures[route] = fixture
            with mock.patch.object(
                trainer, "load_shard", side_effect=lambda _bundle, route: fixtures[route]
            ):
                report = trainer.validate_matched_train_rows(bundle)
            self.assertTrue(report["passed"])

            changed_route = bundle.arm_train_routes("rank4-control")[0]
            fixtures[changed_route] = trainer.dataclasses.replace(
                fixtures[changed_route], weights=np.asarray([2.0], dtype="<f4")
            )
            with mock.patch.object(
                trainer, "load_shard", side_effect=lambda _bundle, route: fixtures[route]
            ):
                with self.assertRaisesRegex(trainer.TrainingError, "row schedule"):
                    trainer.validate_matched_train_rows(bundle)

    def test_bundle_level_input_audit_is_content_addressed_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            bundle = bundle_fixture(root)
            routes = trainer._input_audit_routes(bundle)
            fixtures = {}
            paired = dataset([active_row(0, 1)], [0.25])
            for route in (*routes["search_train"], *routes["rank4_train"]):
                fixtures[route] = trainer.dataclasses.replace(
                    paired,
                    source_route=route,
                    source_manifest_sha256=hashlib.sha256(route.encode()).hexdigest(),
                )
            for index, route in enumerate(routes["canonical_train"]):
                fixtures[route] = trainer.dataclasses.replace(
                    dataset(
                        [active_row(index + 1, index + 10)],
                        groups=[f"anchor-audit-{index}"],
                    ),
                    source_route=route,
                    source_manifest_sha256=hashlib.sha256(route.encode()).hexdigest(),
                )
            common_route = routes["common_adjudicator"][0]
            fixtures[common_route] = trainer.dataclasses.replace(
                dataset(
                    [active_row(10, 20)],
                    split="validation",
                    groups=["common-audit"],
                ),
                source_route=common_route,
                source_manifest_sha256=hashlib.sha256(common_route.encode()).hexdigest(),
            )
            for index, route in enumerate(routes["canonical_validation"]):
                fixtures[route] = trainer.dataclasses.replace(
                    dataset(
                        [active_row(index + 20, index + 30)],
                        split="validation",
                        groups=[f"validation-audit-{index}"],
                    ),
                    source_route=route,
                    source_manifest_sha256=hashlib.sha256(route.encode()).hexdigest(),
                )
            with mock.patch.object(
                trainer, "load_shard", side_effect=lambda _bundle, route: fixtures[route]
            ):
                audit_path = trainer.generate_input_audit(bundle, root / "audit")
            audit = trainer.validate_input_audit(bundle, audit_path)
            self.assertTrue(audit["paired_row_validation"]["passed"])
            self.assertTrue(audit["split_isolation"]["passed"])
            self.assertFalse(audit["protected_tests_opened"])


class SuccessorRankingTests(unittest.TestCase):
    @staticmethod
    def group(
        teacher_values,
        *,
        parent_mover=0,
        value_movers=None,
        exhaustive=True,
        teacher_ranking_profile="standard-v1",
    ):
        value_movers = value_movers or [parent_mover] * len(teacher_values)
        successors = tuple(
            trainer.CompleteTurnSuccessor(
                successor_id=f"{index + 1:064x}",
                active=active_row(index % 8, index % 16),
                teacher_value=float(value),
                value_mover=int(value_movers[index]),
                evidence={"bound": True},
            )
            for index, value in enumerate(teacher_values)
        )
        return trainer.CompleteTurnGroup(
            group_id="f" * 64,
            parent_mover=parent_mover,
            successors=successors,
            successors_exhaustive=exhaustive,
            evidence={
                "rich": True,
                "work_budget": {
                    "teacher_ranking_profile": teacher_ranking_profile,
                },
            },
        )

    def test_hard_teacher_groups_have_increased_schedule_density(self):
        standard = self.group([1.0, -1.0])
        hard = self.group(
            [0.8, -0.8],
            teacher_ranking_profile=trainer.HARD_TEACHER_RANKING_PROFILE,
        )
        hard = dataclasses.replace(hard, group_id="e" * 64)
        scheduled, evidence = trainer._density_weighted_ranking_groups(
            (standard, hard)
        )
        self.assertEqual(scheduled.count(standard), 1)
        self.assertEqual(
            scheduled.count(hard), trainer.HARD_STATE_DENSITY_MULTIPLIER
        )
        self.assertTrue(evidence["density_increased"])
        self.assertEqual(evidence["hard_unique_groups"], 1)

    @classmethod
    def ranking_inputs(cls):
        group = cls.group([1.0, -1.0])
        labels = trainer.SuccessorRankingLabels(
            train=(group,),
            validation=(group,),
            teacher={"artifact_sha256": "1" * 64},
            source_bundle_body_sha256="a" * 64,
            artifact_sha256="b" * 64,
            body_sha256="c" * 64,
        )
        return trainer.TrainingInputs(
            new=dataset([active_row(0, 1)]),
            anchor=dataset(
                [active_row(1, index % 16) for index in range(1_000)],
                groups=[f"warmup-anchor-{index}" for index in range(1_000)],
            ),
            common_adjudicator=dataset(
                [active_row(2, 20)], split="validation", groups=["warmup-common"]
            ),
            canonical_validation=dataset(
                [active_row(3, 21)],
                split="validation",
                groups=["warmup-canonical"],
            ),
            source_routes={},
            successor_rankings=labels,
        )

    def test_pairwise_gradient_matches_finite_difference_and_improves_margin(self):
        group = self.group([1.0, -1.0])
        predictions = np.asarray([0.0, 0.0], dtype=np.float32)
        loss, gradient, report = trainer.pairwise_successor_ranking_loss_gradient(
            group, predictions
        )
        self.assertAlmostEqual(loss, np.log(2.0), places=6)
        np.testing.assert_allclose(gradient, [-0.5, 0.5], atol=1e-7)
        self.assertEqual(report["pair_count"], 1)
        epsilon = 1e-3
        numerical = []
        for index in range(2):
            plus = predictions.copy()
            minus = predictions.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = trainer.pairwise_successor_ranking_loss_gradient(
                group, plus
            )[0]
            minus_loss = trainer.pairwise_successor_ranking_loss_gradient(
                group, minus
            )[0]
            numerical.append((plus_loss - minus_loss) / (2 * epsilon))
        np.testing.assert_allclose(gradient, numerical, atol=2e-4)

    def test_parent_to_successor_mover_perspective_flips_targets_and_gradients(self):
        group = self.group(
            [-0.8, 0.4], parent_mover=0, value_movers=[1, 1]
        )
        teacher_parent = trainer._teacher_parent_values(group)
        np.testing.assert_allclose(teacher_parent, [0.8, -0.4])
        loss, gradient, report = trainer.pairwise_successor_ranking_loss_gradient(
            group, np.asarray([0.2, -0.1], dtype=np.float32)
        )
        self.assertGreater(loss, 0.0)
        self.assertEqual(
            report["teacher_best_successor_id"], group.successors[0].successor_id
        )
        # Parent-frame best gradient is negative. Both successor values are in
        # the opponent frame, so the raw-network gradient must reverse sign.
        self.assertGreater(float(gradient[0]), 0.0)
        self.assertLess(float(gradient[1]), 0.0)

    def test_pair_cap_gap_weighting_and_schedule_are_deterministic(self):
        group = self.group([1.0] + [-1.0 + index * 0.1 for index in range(10)])
        first = trainer._ranking_pairs(group)
        second = trainer._ranking_pairs(group)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1], tuple(range(1, 9)))
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        schedule = trainer.successor_ranking_epoch_schedule(
            17, 5, seed=20260907, epoch=3
        )
        repeated = trainer.successor_ranking_epoch_schedule(
            17, 5, seed=20260907, epoch=3
        )
        self.assertTrue(all(
            np.array_equal(first, second)
            for first, second in zip(schedule, repeated, strict=True)
        ))
        next_epoch = trainer.successor_ranking_epoch_schedule(
            17, 5, seed=20260907, epoch=4
        )
        self.assertNotEqual(
            [[int(value) for value in batch] for batch in schedule],
            [[int(value) for value in batch] for batch in next_epoch],
        )
        self.assertEqual(
            sorted(int(value) for batch in schedule for value in batch),
            list(range(17)),
        )
        self.assertLessEqual(
            max(map(len, schedule)) - min(map(len, schedule)), 1
        )
        with self.assertRaises(trainer.TrainingError):
            trainer.successor_ranking_epoch_schedule(
                5, 17, seed=20260907, epoch=3
            )

    def test_weighted_pool_schedule_covers_every_hard_duplicate_once(self):
        standard = self.group([1.0, -1.0])
        hard = dataclasses.replace(
            self.group(
                [0.8, -0.8],
                teacher_ranking_profile=trainer.HARD_TEACHER_RANKING_PROFILE,
            ),
            group_id="e" * 64,
        )
        pool, density = trainer._density_weighted_ranking_groups(
            (standard, hard)
        )
        schedule = trainer.successor_ranking_epoch_schedule(
            len(pool), 3, seed=20260907, epoch=2
        )
        coverage = trainer.ranking_schedule_coverage(
            pool, schedule, epoch=2
        )
        self.assertEqual(
            coverage["executed_weighted_entries"], len(pool)
        )
        self.assertEqual(
            coverage["executed_hard_weighted_entries"],
            trainer.HARD_STATE_DENSITY_MULTIPLIER,
        )
        self.assertEqual(
            trainer.validate_ranking_schedule_coverage(
                coverage, density=density, epoch=2, scalar_batches=3,
                seed=20260907,
            ),
            coverage,
        )
        tampered = copy.deepcopy(coverage)
        tampered["executed_hard_weighted_entries"] -= 1
        with self.assertRaisesRegex(trainer.TrainingError, "weighted pool"):
            trainer.validate_ranking_schedule_coverage(
                tampered, density=density, epoch=2, scalar_batches=3,
                seed=20260907,
            )
        forged_hash = copy.deepcopy(coverage)
        forged_hash["schedule_sha256"] = "f" * 64
        with self.assertRaisesRegex(trainer.TrainingError, "weighted pool"):
            trainer.validate_ranking_schedule_coverage(
                forged_hash, density=density, epoch=2, scalar_batches=3,
                seed=20260907,
            )

    def test_microbatch_gradient_is_numerical_and_lambda_count_invariant(self):
        group = self.group([1.0, -1.0])
        predictions = np.asarray([0.2, -0.3], dtype=np.float32)
        single_loss, single_gradient, _ = (
            trainer.ranking_microbatch_loss_gradient((group,), predictions)
        )
        epsilon = 1e-3
        numerical = []
        for index in range(len(predictions)):
            plus = predictions.copy()
            minus = predictions.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical.append((
                trainer.ranking_microbatch_loss_gradient((group,), plus)[0]
                - trainer.ranking_microbatch_loss_gradient((group,), minus)[0]
            ) / (2 * epsilon))
        np.testing.assert_allclose(single_gradient, numerical, atol=2e-4)

        repeated_predictions = np.tile(predictions, 4)
        repeated_loss, repeated_gradient, report = (
            trainer.ranking_microbatch_loss_gradient(
                (group, group, group, group), repeated_predictions
            )
        )
        self.assertAlmostEqual(single_loss, repeated_loss, places=7)
        np.testing.assert_allclose(
            repeated_gradient.reshape(4, -1).sum(axis=0),
            single_gradient,
            atol=1e-7,
        )
        self.assertEqual(report["lambda_application"], "once-after-group-mean")

    def test_metrics_exclude_singleton_and_nonexhaustive_groups_and_report_flips(self):
        comparable = self.group([1.0, -1.0])
        singleton = self.group([0.5])
        nonexhaustive = self.group([1.0, -1.0], exhaustive=False)
        skipped_loss, skipped_gradient, skipped = (
            trainer.pairwise_successor_ranking_loss_gradient(
                nonexhaustive, np.asarray([0.0, 0.0], dtype=np.float32)
            )
        )
        self.assertEqual(skipped_loss, 0.0)
        np.testing.assert_array_equal(skipped_gradient, [0.0, 0.0])
        self.assertTrue(skipped["skipped_nonexhaustive"])
        architecture = trainer.ARCHITECTURES["compact-8x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        quantized = trainer.quantize_fixed(
            parameters,
            architecture,
            {"w1": 0.1, "w2": 0.1, "w3": 0.1},
        )
        with mock.patch.object(
            trainer,
            "forward",
            side_effect=(
                (np.asarray([0.0, 0.5], dtype=np.float32), None),
                (np.asarray([0.5, 0.0], dtype=np.float32), None),
            ),
        ):
            report = trainer.successor_ranking_metrics(
                parameters,
                architecture,
                (comparable, singleton, nonexhaustive),
                quantized=quantized,
            )
        self.assertEqual(report["groups"], 3)
        self.assertEqual(report["comparable_groups"], 1)
        self.assertEqual(report["singleton_groups"], 1)
        self.assertEqual(report["skipped_nonexhaustive_groups"], 1)
        self.assertEqual(report["float_vs_quantized_action_flips"], 1)
        self.assertEqual(report["top1_agreement"], 1.0)
        self.assertEqual(report["mean_teacher_regret"], 0.0)

    def test_weight_config_and_per_layer_evidence_fail_closed(self):
        self.assertEqual([trainer._ranking_weight(value) for value in (
            0, 0.10, 0.25
        )], [0.0, 0.10, 0.25])
        for value in (-0.1, 0.2, 1.0, True, "0.1"):
            with self.subTest(value=value), self.assertRaises(
                trainer.TrainingError
            ):
                trainer._ranking_weight(value)
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        before = trainer.initialize_parameters(architecture, 20260907)
        after = {name: value.copy() for name, value in before.items()}
        after["w1"][0, 0] += np.float32(0.25)
        evidence = trainer._parameter_update_evidence(before, after)
        self.assertTrue(evidence["w1"]["changed"])
        self.assertEqual(evidence["w1"]["changed_parameters"], 1)
        self.assertFalse(evidence["w2"]["changed"])
        self.assertFalse(evidence["w3"]["changed"])

    def test_qat_batch_adds_lambda_ranking_without_scaling_scalar_gradient(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        inputs = trainer.TrainingInputs(
            new=dataset([active_row(0, index % 16) for index in range(64)]),
            anchor=dataset(
                [active_row(1, index % 16) for index in range(192)],
                groups=[f"ranking-anchor-{index}" for index in range(192)],
            ),
            common_adjudicator=dataset(
                [active_row(2, 20)], split="validation", groups=["common"]
            ),
            canonical_validation=dataset(
                [active_row(3, 21)], split="validation", groups=["canonical"]
            ),
            source_routes={},
        )
        group = self.group([1.0, -1.0])
        scalar_gradients = {
            name: np.full_like(value, 1e-4) for name, value in parameters.items()
        }
        ranking_gradients = {
            name: np.full_like(value, 0.5e-4) for name, value in parameters.items()
        }
        quantized = mock.Mock()
        quantized.effective.return_value = parameters
        optimizer = mock.Mock()
        with (
            mock.patch.object(trainer, "quantize_fixed", return_value=quantized),
            mock.patch.object(
                trainer,
                "forward",
                side_effect=(
                    (np.zeros(256, dtype=np.float32), mock.Mock()),
                    (np.zeros(2, dtype=np.float32), mock.Mock()),
                ),
            ),
            mock.patch.object(
                trainer,
                "arm_loss_gradient",
                return_value=(
                    2.0, np.zeros(256, dtype=np.float32), {}
                ),
            ),
            mock.patch.object(
                trainer,
                "pairwise_successor_ranking_loss_gradient",
                return_value=(
                    3.0,
                    np.zeros(2, dtype=np.float32),
                    {
                        "successors_exhaustive": True,
                        "skipped_nonexhaustive": False,
                        "pair_count": 1,
                    },
                ),
            ),
            mock.patch.object(
                trainer,
                "_network_gradients",
                side_effect=(scalar_gradients, ranking_gradients),
            ),
        ):
            objective = trainer._train_mixed_batch(
                parameters,
                architecture,
                trainer.ARMS["search-target"],
                optimizer,
                inputs,
                np.arange(64, dtype=np.int64),
                np.arange(192, dtype=np.int64),
                fixed_scales={"w1": 0.1, "w2": 0.1, "w3": 0.1},
                ranking_group=group,
                ranking_weight=0.25,
            )
        self.assertEqual(objective, 2.75)
        optimizer.update.assert_called_once()
        combined = optimizer.update.call_args.args[1]
        for name in ("w1", "w2", "w3"):
            np.testing.assert_allclose(
                combined[name], np.full_like(parameters[name], 1.5e-4),
                rtol=1e-6, atol=0.0,
            )

    def test_successor_mode_uses_same_bound_initialization_and_one_epoch(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        initial = trainer.initialize_parameters(architecture, 20260907)
        inputs = self.ranking_inputs()
        validation = {
            "common_adjudicator": {
                "objective_weighted_huber": 0.1,
                "weighted_huber": 0.1,
                "sign_accuracy": 0.5,
            },
            "canonical_validation": {
                "objective_weighted_huber": 0.1,
                "weighted_huber": 0.1,
                "sign_accuracy": 0.5,
            },
            "successor_ranking": {
                "loss_weight": 0.0,
                "mean_teacher_regret": 0.0,
                "top1_agreement": 1.0,
                "float_vs_quantized_action_flip_rate": 0.0,
                "pairwise_loss": 0.1,
            },
        }
        results = []
        with (
            mock.patch.object(trainer, "_train_mixed_batch", return_value=0.0),
            mock.patch.object(
                trainer, "evaluate_validation_pair", return_value=validation
            ),
        ):
            for seed in trainer.FIXED_SEEDS[:2]:
                results.append(trainer.train_float_seed(
                    inputs,
                    architecture,
                    trainer.ARMS["search-target"],
                    seed,
                    maximum_epochs=1,
                    patience=1,
                    learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
                    ranking_weight=0.0,
                    initial_parameters=initial,
                ))
        self.assertEqual([result.epoch for result in results], [1, 1])
        self.assertEqual(
            results[0].report["initialization"]["parameters"],
            results[1].report["initialization"]["parameters"],
        )
        self.assertEqual(
            results[0].report["initialization"]["seed_affects"],
            "row-order-only",
        )
        self.assertEqual(
            results[0].report["selected_epoch_anchor_coverage"]["anchor"][
                "complete_permutations"
            ],
            0,
        )
        for kwargs in (
            {"initial_parameters": None},
            {"maximum_epochs": 2, "initial_parameters": initial},
            {"learning_rate": 6.1e-5, "initial_parameters": initial},
        ):
            arguments = {
                "maximum_epochs": 1,
                "patience": 1,
                "learning_rate": trainer.RANKING_FLOAT_LEARNING_RATE,
                "ranking_weight": 0.0,
                "initial_parameters": initial,
                **kwargs,
            }
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                trainer.TrainingError, "bound 12x8 initialization"
            ):
                trainer.train_float_seed(
                    inputs,
                    architecture,
                    trainer.ARMS["search-target"],
                    trainer.FIXED_SEEDS[0],
                    **arguments,
                )

    def test_training_binding_requires_content_addressed_initial_checkpoint(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        inputs = self.ranking_inputs()
        bundle = mock.Mock(body_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = trainer.write_float_checkpoint(
                pathlib.Path(temporary), parameters, architecture
            )
            first = trainer.training_binding(
                bundle,
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                trainer.FIXED_SEEDS[0],
                None,
                0.0,
                checkpoint,
            )
            second = trainer.training_binding(
                bundle,
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                trainer.FIXED_SEEDS[1],
                None,
                0.25,
                checkpoint,
            )
            first_checkpoint = first["successor_ranking"]["initial_checkpoint"]
            second_checkpoint = second["successor_ranking"]["initial_checkpoint"]
            self.assertEqual(first_checkpoint, second_checkpoint)
            self.assertEqual(
                first["successor_ranking"]["float_warmup"],
                {
                    "epochs": 1,
                    "learning_rate": 0.00006,
                    "seeds_affect_row_order_only": True,
                    "legacy_full_anchor_pass_required": False,
                },
            )
            with self.assertRaisesRegex(
                trainer.TrainingError, "initial checkpoint"
            ):
                trainer.training_binding(
                    bundle,
                    inputs,
                    architecture,
                    trainer.ARMS["search-target"],
                    trainer.FIXED_SEEDS[0],
                    None,
                    0.0,
                    None,
                )


class QATProfileTests(unittest.TestCase):
    @staticmethod
    def validation(value: float = 0.1):
        return {
            "common_adjudicator": {
                "objective_weighted_huber": value,
                "weighted_huber": value,
                "sign_accuracy": 1.0,
            },
            "canonical_validation": {
                "objective_weighted_huber": value,
                "weighted_huber": value,
                "sign_accuracy": 1.0,
            },
            "successor_ranking": {
                "loss_weight": 0.0,
                "float_vs_quantized_action_flip_rate": value,
                "mean_teacher_regret": value,
                "top1_agreement": 1.0 - min(value, 1.0),
            },
        }

    def test_profile_registry_is_closed_body_hashed_and_tamper_evident(self):
        self.assertEqual(
            set(trainer.QAT_PROFILES),
            {"standard-v1", "refined-adaptive-scales-v1"},
        )
        for name in trainer.QAT_PROFILES:
            contract = trainer.qat_profile_contract(name)
            self.assertEqual(
                trainer.validate_qat_profile_contract(
                    contract, expected_name=name
                ),
                contract,
            )
        tampered = copy.deepcopy(trainer.qat_profile_contract("standard-v1"))
        tampered["scale_selection"]["coordinate_search_passes"] = 3
        with self.assertRaisesRegex(trainer.TrainingError, "registry"):
            trainer.validate_qat_profile_contract(tampered)
        with self.assertRaisesRegex(trainer.TrainingError, "must be"):
            trainer.resolve_qat_profile("unregistered")

    def test_refined_profile_materially_changes_scale_candidate_selection(self):
        architecture = trainer.ARCHITECTURES["compact-8x8"]
        parameters = {
            name: np.linspace(-1.0, 1.0, num=np.prod(shape), dtype=np.float32)
            .reshape(shape)
            for name, shape in architecture.shapes.items()
        }

        def evaluate(_parameters, _architecture, _inputs, _arm, *, quantized, **_kwargs):
            return self.validation(sum(
                float(quantized.scales[name]) for name in ("w1", "w2", "w3")
            ))

        with mock.patch.object(
            trainer, "evaluate_validation_pair", side_effect=evaluate
        ):
            standard, standard_report = trainer.select_fixed_scales(
                parameters,
                architecture,
                object(),
                trainer.ARMS["search-target"],
                qat_profile="standard-v1",
            )
            refined, refined_report = trainer.select_fixed_scales(
                parameters,
                architecture,
                object(),
                trainer.ARMS["search-target"],
                qat_profile="refined-adaptive-scales-v1",
            )
        self.assertEqual(standard_report["passes"], 2)
        self.assertEqual(standard_report["maximum_candidate_quantile"], "p995-lower-rank")
        self.assertEqual(standard_report["local_refinement_trials"], 0)
        self.assertEqual(
            len(standard_report["trials"]),
            standard_report["passes"] * sum(
                len(values) for values in standard_report["candidates"].values()
            ),
        )
        self.assertEqual(refined_report["passes"], 3)
        self.assertEqual(refined_report["maximum_candidate_quantile"], "p998-lower-rank")
        self.assertGreater(refined_report["local_refinement_trials"], 0)
        self.assertTrue(any(
            standard.scales[name] != refined.scales[name]
            for name in ("w1", "w2", "w3")
        ))

    def test_refined_profile_prioritizes_the_quantized_action_flip_gap(self):
        scalar_better = self.validation(0.01)
        scalar_better["successor_ranking"][
            "float_vs_quantized_action_flip_rate"
        ] = 0.20
        flip_better = self.validation(0.10)
        flip_better["successor_ranking"][
            "float_vs_quantized_action_flip_rate"
        ] = 0.01
        standard = trainer.QAT_PROFILES["standard-v1"]
        refined = trainer.QAT_PROFILES["refined-adaptive-scales-v1"]
        self.assertLess(
            trainer._qat_validation_key(scalar_better, standard),
            trainer._qat_validation_key(flip_better, standard),
        )
        self.assertLess(
            trainer._qat_validation_key(flip_better, refined),
            trainer._qat_validation_key(scalar_better, refined),
        )

    def test_refined_profile_records_four_adaptive_all_layer_qat_epochs(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        inputs = SuccessorRankingTests.ranking_inputs()
        validation = self.validation()
        with (
            mock.patch.object(trainer, "_train_mixed_batch", return_value=0.0),
            mock.patch.object(
                trainer, "evaluate_validation_pair", return_value=validation
            ),
        ):
            floating = trainer.train_float_seed(
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                20260907,
                maximum_epochs=1,
                patience=1,
                learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
                ranking_weight=0.0,
                initial_parameters=parameters,
            )
            result = trainer.run_fixed_scale_qat(
                floating,
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                20260907,
                qat_profile="refined-adaptive-scales-v1",
            )
        evidence = trainer.validate_qat_execution_evidence(
            result.report, expected_profile="refined-adaptive-scales-v1"
        )
        self.assertEqual(evidence["executed_qat_epochs"], [1, 2, 3, 4])
        self.assertEqual(evidence["selected_qat_epoch"], 1)
        self.assertFalse(evidence["pre_qat_retained"])
        self.assertEqual(
            set(evidence["selected_per_layer_qat_evidence"]),
            {"w1", "w2", "w3"},
        )
        self.assertTrue(evidence["adaptive_scale_qat"])
        self.assertTrue(all(
            item["adapted_after_epoch"]
            for item in evidence["applied_scale_trajectory"]
        ))
        self.assertGreater(
            len(evidence["scale_search"]["trials"]),
            2 * 3 * len(trainer.ROBUST_SCALE_QUANTILES),
        )
        self.assertTrue(all(
            item["adaptive_scale_search"]["trials"]
            for item in evidence["history"]
        ))
        tampered = copy.deepcopy(evidence)
        tampered["history"][0]["fixed_scales"]["w1"] = float(np.float32(0.2))
        with self.assertRaisesRegex(trainer.TrainingError, "scale"):
            trainer.validate_qat_execution_evidence(
                tampered, expected_profile="refined-adaptive-scales-v1"
            )
        tampered = copy.deepcopy(evidence)
        tampered["selected_qat_epoch"] = 0
        tampered["pre_qat_retained"] = True
        with self.assertRaisesRegex(trainer.TrainingError, "schedule/profile"):
            trainer.validate_qat_execution_evidence(
                tampered, expected_profile="refined-adaptive-scales-v1"
            )

    def test_active_ranking_receipt_covers_pool_in_warmup_and_all_qat_epochs(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        inputs = SuccessorRankingTests.ranking_inputs()
        validation = self.validation()
        validation["successor_ranking"]["loss_weight"] = 0.25
        validation["successor_ranking"]["pairwise_loss"] = 0.1
        with (
            mock.patch.object(trainer, "_train_mixed_batch", return_value=0.0),
            mock.patch.object(
                trainer, "evaluate_validation_pair", return_value=validation
            ),
        ):
            floating = trainer.train_float_seed(
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                20260907,
                maximum_epochs=1,
                patience=1,
                learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
                ranking_weight=0.25,
                initial_parameters=parameters,
            )
            quantized = trainer.run_fixed_scale_qat(
                floating,
                inputs,
                architecture,
                trainer.ARMS["search-target"],
                20260907,
                ranking_weight=0.25,
            )
        evidence = trainer.validate_successor_schedule_execution(
            floating.report, quantized.report, seed=20260907
        )
        self.assertEqual(evidence["validated_full_pool_reports"], 5)
        self.assertEqual(
            [item["schedule_epoch"] for item in quantized.report["history"]],
            [2, 3, 4, 5],
        )
        forged_float = copy.deepcopy(floating.report)
        forged_float["history"][0]["ranking_schedule_coverage"][
            "schedule_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(trainer.TrainingError, "weighted pool"):
            trainer.validate_successor_schedule_execution(
                forged_float, quantized.report, seed=20260907
            )

    def test_fresh_qat_runs_are_byte_reproducible_across_output_roots(self):
        architecture = trainer.ARCHITECTURES["capacity-12x8"]
        arm = trainer.ARMS["search-target"]
        seed = 20260907
        ranking_weight = 0.10
        qat_profile = trainer.STANDARD_QAT_PROFILE
        group = SuccessorRankingTests.group([1.0, -1.0])
        labels = trainer.SuccessorRankingLabels(
            train=(group,),
            validation=(group,),
            teacher={"artifact_sha256": "1" * 64},
            source_bundle_body_sha256="a" * 64,
            artifact_sha256="b" * 64,
            body_sha256="c" * 64,
        )
        common_rows = [active_row(2, index % 32) for index in range(4_096)]
        inputs = trainer.TrainingInputs(
            new=dataset([
                active_row(index % 8, index % 32) for index in range(64)
            ]),
            anchor=dataset(
                [active_row((index + 1) % 8, index % 32) for index in range(192)],
                groups=[f"repro-anchor-{index}" for index in range(192)],
            ),
            common_adjudicator=dataset(
                common_rows,
                split="validation",
                groups=[f"repro-common-{index}" for index in range(4_096)],
            ),
            canonical_validation=dataset(
                [active_row((index + 3) % 8, index % 32) for index in range(32)],
                split="validation",
                groups=[f"repro-canonical-{index}" for index in range(32)],
            ),
            source_routes={},
            successor_rankings=labels,
        )
        bundle = mock.Mock(body_sha256="a" * 64)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            initial = trainer.write_float_checkpoint(
                root / "initial",
                trainer.initialize_parameters(architecture, seed),
                architecture,
            )
            outputs = [root / "first", root / "second"]
            receipts = [
                trainer.train_seed_candidate(
                    bundle,
                    inputs,
                    architecture,
                    arm,
                    seed,
                    output,
                    ranking_weight=ranking_weight,
                    initial_checkpoint=initial,
                    qat_profile=qat_profile,
                )
                for output in outputs
            ]

            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(
                receipts[0]["binding"]["successor_ranking"]["loss_weight"],
                ranking_weight,
            )
            self.assertEqual(receipts[0]["qat_profile"], qat_profile)
            self.assertEqual(
                receipts[0]["quantized_validation"],
                receipts[1]["quantized_validation"],
            )
            self.assertEqual(
                receipts[0]["quantized_training"],
                receipts[1]["quantized_training"],
            )

            runtimes = []
            receipt_payloads = []
            for output, receipt in zip(outputs, receipts, strict=True):
                runtime_path = output / receipt["quantized_runtime"]["path"]
                runtime_payload = runtime_path.read_bytes()
                loaded_architecture, quantized, selection, _document = (
                    trainer.load_runtime(runtime_path)
                )
                self.assertEqual(loaded_architecture, architecture)
                self.assertEqual(selection["seed"], seed)
                runtimes.append((runtime_payload, quantized))

                reference_path = trainer._seed_reference_path(
                    output, architecture, arm, seed
                )
                _reference_payload, reference = trainer._load_canonical_json(
                    reference_path, "QAT reproducibility seed reference"
                )
                receipt_path = output / reference["receipt"]
                receipt_payloads.append(receipt_path.read_bytes())

            self.assertEqual(runtimes[0][0], runtimes[1][0])
            self.assertEqual(receipt_payloads[0], receipt_payloads[1])
            for name in ("w1", "w2", "w3"):
                np.testing.assert_array_equal(
                    runtimes[0][1].integer[name], runtimes[1][1].integer[name]
                )
                self.assertEqual(
                    runtimes[0][1].scales[name], runtimes[1][1].scales[name]
                )


class FloatRankingDecisionCacheTests(unittest.TestCase):
    class UnreadableFeatures:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("skipped group features must not be read")

    def setUp(self):
        self.architecture = trainer.ARCHITECTURES["capacity-12x8"]
        self.arm = trainer.ARMS["search-target"]
        self.seed = trainer.FIXED_SEEDS[0]
        self.parameters = trainer.initialize_parameters(self.architecture, self.seed)
        self.scales = {name: float(np.max(np.abs(value))) / 3 for name, value in self.parameters.items()}

    def groups(self):
        tied_features = active_row(1, 2)
        tied_float = trainer.CompleteTurnGroup("float-tie", 0, tuple(
            trainer.CompleteTurnSuccessor(f"{identity:064x}", tied_features, target, 0,
                {"opaque": object(), "auxiliary_array": np.zeros(1)})
            for identity, target in ((9, .8), (1, -.2), (5, -.5))))
        perspectives = trainer.CompleteTurnGroup("terminal-perspectives", 1, tuple(
            trainer.CompleteTurnSuccessor(f"{20 + index:064x}", active_row(index + 2, index + 3), value, mover,
                {"proof": {"solved": abs(value) == 1, "proven_winner": mover if value == 1 else 1 - mover if value == -1 else None}})
            for index, (value, mover) in enumerate(((1., 0), (-1., 1), (.3, 1), (-.4, 0)))))
        unreadable = self.UnreadableFeatures()
        def skipped(name, values, exhaustive=True):
            return trainer.CompleteTurnGroup(name, 0, tuple(
                trainer.CompleteTurnSuccessor(f"{100 + index:064x}", unreadable, value, 0, {})
                for index, value in enumerate(values)), successors_exhaustive=exhaustive)
        return (tied_float, perspectives, skipped("singleton", [1.]),
                skipped("teacher-tied", [0., 0.]), skipped("nonexhaustive", [1., -1.], False))

    def inputs(self, groups=None):
        groups = self.groups() if groups is None else groups
        labels = trainer.SuccessorRankingLabels(train=groups[:2], validation=groups,
            teacher={"artifact_sha256": "1" * 64}, source_bundle_body_sha256="a" * 64,
            artifact_sha256="b" * 64, body_sha256="c" * 64)
        return trainer.TrainingInputs(
            new=dataset([active_row(0, 1), active_row(1, 2), active_row(2, 3)], [.2, -.3, .8]),
            anchor=dataset([active_row(3, 4), active_row(4, 5)], [-.7, .4]),
            common_adjudicator=dataset([active_row(5, 6), active_row(6, 7)], [.7, -.8], split="validation"),
            canonical_validation=dataset([active_row(7, 8), active_row(8, 9)], [-.4, .9], split="validation"),
            source_routes={}, successor_rankings=labels)

    def assert_json_equal(self, left, right):
        self.assertEqual(trainer.canonical_json_bytes(left), trainer.canonical_json_bytes(right))

    def assert_quantized_equal(self, left, right):
        for name in ("w1", "w2", "w3"):
            self.assertEqual(left.integer[name].tobytes(), right.integer[name].tobytes())
            self.assertEqual(left.scales[name].tobytes(), right.scales[name].tobytes())
        self.assertEqual(trainer.pack_signed_three_bit(trainer._flatten_quantized(left, self.architecture)),
                         trainer.pack_signed_three_bit(trainer._flatten_quantized(right, self.architecture)))

    def test_complete_metrics_are_identical_and_only_repeated_float_forwards_disappear(self):
        inputs = self.inputs(); groups = inputs.successor_rankings.validation
        quantized = [trainer.quantize_fixed(self.parameters, self.architecture,
            {name: value * factor for name, value in self.scales.items()}) for factor in (1., .85, 1.1)]
        for weight in (0., .1, .25):
            with self.subTest(weight=weight), trainer.native_thread_execution_scope():
                baseline = [trainer.evaluate_validation_pair(self.parameters, self.architecture, inputs, self.arm,
                    quantized=value, ranking_weight=weight) for value in quantized]
                cache = trainer._new_float_ranking_decision_cache(self.parameters, self.architecture, inputs)
                original = trainer.forward
                counts = {"float": 0, "quantized": 0}
                def observed(*args, **kwargs):
                    counts["float" if kwargs.get("quantized") is None else "quantized"] += 1
                    return original(*args, **kwargs)
                with mock.patch.object(trainer, "forward", side_effect=observed):
                    actual = [trainer.evaluate_validation_pair(self.parameters, self.architecture, inputs, self.arm,
                        quantized=value, ranking_weight=weight, _float_best_cache=cache) for value in quantized]
                self.assert_json_equal(actual, baseline)
                self.assertEqual(counts["float"], 2)
                self.assertEqual(counts["quantized"], 12)  # Two scalar sets plus two comparable groups per trial.
                self.assertEqual(cache.best[0], 1)  # ID-ascending tie, not first-index argmax.
                raw, _ = original(self.parameters, self.architecture,
                    tuple(s.active for s in groups[1].successors))
                parent, _ = trainer._parent_frame_values(groups[1], raw)
                self.assertEqual(cache.best[1], trainer._deterministic_best(groups[1], parent))
                self.assertEqual(cache.best[2:], [None, None, None])
                self.assertEqual(cache.comparable, [True, True, False, False, False])
                with self.assertRaisesRegex(trainer.TrainingError, "requires quantized validation"):
                    trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, _float_best_cache=cache)

    def test_stale_parameters_architecture_order_and_eager_content_are_rejected(self):
        quantized = trainer.quantize_fixed(self.parameters, self.architecture, self.scales)
        groups = self.groups()
        cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
        trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)
        self.parameters["w3"][0] += np.float32(.125)
        with self.assertRaisesRegex(trainer.TrainingError, "parameters, architecture or group sequence changed"):
            cache.validate(self.parameters, self.architecture, groups)
        self.parameters = trainer.initialize_parameters(self.architecture, self.seed)
        with self.assertRaisesRegex(trainer.TrainingError, "parameters, architecture or group sequence changed"):
            cache.validate(self.parameters, dataclasses.replace(self.architecture, name="other"), groups)
        with self.assertRaisesRegex(trainer.TrainingError, "group sequence changed"):
            cache.validate(self.parameters, self.architecture, tuple(reversed(groups)))
        groups[0].successors[0].active[0] = 4
        with self.assertRaisesRegex(trainer.TrainingError, "eager group content changed"):
            trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)
        for field, value in (("successor_id", "0" * 64), ("value_mover", 1), ("teacher_value", .6)):
            groups = self.groups()
            mutable = list(groups[0].successors)
            groups = (dataclasses.replace(groups[0], successors=mutable), *groups[1:])
            cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
            trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)
            mutable[0] = dataclasses.replace(mutable[0], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(trainer.TrainingError, "eager group content changed"):
                trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)

    def test_comparability_changes_never_reuse_skipped_or_previously_live_entries(self):
        quantized = trainer.quantize_fixed(self.parameters, self.architecture, self.scales)
        for initially_tied in (True, False):
            groups = self.groups(); mutable = list(groups[0].successors)
            if initially_tied:
                mutable[:] = [dataclasses.replace(s, teacher_value=0.) for s in mutable]
            groups = (dataclasses.replace(groups[0], successors=mutable),)
            cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
            trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)
            mutable[:] = [dataclasses.replace(s, teacher_value=(1. if index == 0 else -1.) if initially_tied else 0.)
                          for index, s in enumerate(mutable)]
            with self.subTest(initially_tied=initially_tied), self.assertRaisesRegex(trainer.TrainingError, "comparability changed"):
                trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized, _float_best_cache=cache)

    def test_standard_and_refined_search_and_adaptation_reports_are_byte_identical(self):
        inputs = self.inputs()
        for profile_name in (trainer.STANDARD_QAT_PROFILE, trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE):
            for weight in (0., .1, .25):
                arguments = (self.parameters, self.architecture, inputs, self.arm)
                with self.subTest(profile=profile_name, weight=weight), trainer.native_thread_execution_scope():
                    with mock.patch.object(trainer, "_new_float_ranking_decision_cache", return_value=None):
                        before, before_report = trainer.select_fixed_scales(*arguments, ranking_weight=weight, qat_profile=profile_name)
                    after, after_report = trainer.select_fixed_scales(*arguments, ranking_weight=weight, qat_profile=profile_name)
                    self.assert_json_equal(before_report, after_report)
                    self.assert_quantized_equal(before, after)
                    if profile_name == trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE:
                        profile = trainer.resolve_qat_profile(profile_name)
                        with mock.patch.object(trainer, "_new_float_ranking_decision_cache", return_value=None):
                            old, old_report = trainer._adapt_fixed_scales(*arguments, before.scales, profile, qat_epoch=1, ranking_weight=weight)
                        new, new_report = trainer._adapt_fixed_scales(*arguments, before.scales, profile, qat_epoch=1, ranking_weight=weight)
                        self.assert_json_equal(old_report, new_report)
                        self.assert_quantized_equal(old, new)

    def test_each_search_and_adaptive_epoch_gets_a_fresh_cache(self):
        inputs = self.inputs(); observed = []
        factory = trainer._new_float_ranking_decision_cache
        def capture(*args):
            result = factory(*args); observed.append(result); return result
        profile = trainer.resolve_qat_profile(trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE)
        with trainer.native_thread_execution_scope(), mock.patch.object(trainer, "_new_float_ranking_decision_cache", side_effect=capture):
            trainer.select_fixed_scales(self.parameters, self.architecture, inputs, self.arm)
            for epoch in (1, 2):
                self.parameters["w3"] *= np.float32(-1)
                trainer._adapt_fixed_scales(self.parameters, self.architecture, inputs, self.arm,
                    self.scales, profile, qat_epoch=epoch, ranking_weight=.1)
        self.assertEqual(len({id(value) for value in observed}), 3)
        self.assertNotEqual(observed[0].parameters, observed[1].parameters)
        self.assertNotEqual(observed[1].parameters, observed[2].parameters)

    def test_bounded_real_warmup_and_all_refined_qat_epochs_are_unchanged(self):
        inputs = self.inputs(); weight = .1
        with trainer.native_thread_execution_scope():
            floating = trainer.train_float_seed(inputs, self.architecture, self.arm, self.seed,
                maximum_epochs=1, patience=1, learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
                ranking_weight=weight, initial_parameters=self.parameters)
            with mock.patch.object(trainer, "_new_float_ranking_decision_cache", return_value=None):
                old = trainer.run_fixed_scale_qat(floating, inputs, self.architecture, self.arm, self.seed,
                    ranking_weight=weight, qat_profile=trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE)
            new = trainer.run_fixed_scale_qat(floating, inputs, self.architecture, self.arm, self.seed,
                ranking_weight=weight, qat_profile=trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE)
        self.assert_json_equal(old.report, new.report)
        self.assert_json_equal(old.metrics, new.metrics)
        self.assert_quantized_equal(old.quantized, new.quantized)
        self.assertEqual(new.report["executed_qat_epochs"], [1, 2, 3, 4])
        self.assertEqual(new.report["optimizer_steps"], 4)

    def test_cache_retains_no_feature_or_prediction_arrays(self):
        import gc
        import weakref
        active = active_row(2, 4)
        group = trainer.CompleteTurnGroup("temporary", 0, (
            trainer.CompleteTurnSuccessor("a" * 64, active, 1., 0, {}),
            trainer.CompleteTurnSuccessor("b" * 64, active, -1., 0, {})))
        groups = (group,); group_ref = weakref.ref(group); active_ref = weakref.ref(active)
        cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
        trainer.successor_ranking_metrics(self.parameters, self.architecture, groups,
            quantized=trainer.quantize_fixed(self.parameters, self.architecture, self.scales), _float_best_cache=cache)
        del active, group, groups
        gc.collect()
        self.assertIsNone(group_ref()); self.assertIsNone(active_ref())
        self.assertTrue(all(value is None or type(value) is int for value in cache.best))

    def test_mapped_backing_identity_ranges_and_readonly_guards(self):
        from tools import compact_value_bfm_ranking_store as store
        from tests.codingame import test_compact_value_bfm_ranking_store as fixtures
        with tempfile.TemporaryDirectory() as temporary:
            _rows, bundle, index = fixtures.RankingStoreTests().fixture(pathlib.Path(temporary))
            labels = store.RankingStore(index, bundle).labels(); groups = labels.validation
            cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
            self.assertTrue(cache.immutable_mapped_groups)
            mapped = groups[0].successors; original_begin, original_end = mapped.begin, mapped.end
            mapped.begin += 1; mapped.end += 1
            with self.assertRaisesRegex(trainer.TrainingError, "group sequence changed"):
                cache.validate(self.parameters, self.architecture, groups)
            mapped.begin, mapped.end = original_begin, original_end
            original_store = mapped.store
            mapped.store = store.RankingStore(index, bundle)
            with self.assertRaisesRegex(trainer.TrainingError, "group sequence changed"):
                cache.validate(self.parameters, self.architecture, groups)
            mapped.store = original_store
            for name in ("indices", "metadata", "transcripts"):
                original = getattr(mapped.store, name)
                replacement = np.asarray(original).copy(); replacement.flags.writeable = False
                setattr(mapped.store, name, replacement)
                with self.subTest(array=name), self.assertRaisesRegex(trainer.TrainingError, "group sequence changed"):
                    cache.validate(self.parameters, self.architecture, groups)
                setattr(mapped.store, name, original)
            cache.validate(self.parameters, self.architecture, groups)

    def test_actual_mapped_decisions_use_fastpath_and_custom_sequences_cannot_spoof_it(self):
        import types
        from tools import compact_value_bfm_ranking_store as store
        from tests.codingame import test_jacek_replay_corpus as fixtures
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            row = fixtures.JacekReplayCorpusTests.complete_turn_action_group_row(split="validation")
            row["source_bundle_body_sha256"] = "a" * 64
            state = fixtures.features.ReplayState()
            fixtures.features.apply_complete_turn(state, 0, "0")
            fixtures.features.apply_complete_turn(state, 1, "2")
            extra = copy.deepcopy(row["group"]["successors"][0])
            extra.update({"successor_id": fixtures.corpus._mover_canonical_position_identity(state),
                "active": list(fixtures.features.encode_active(state)), "transcript": "6",
                "teacher_value": .7, "value_mover": state.to_move})
            row["group"]["successors"].append(extra)
            row["group"]["successors"].sort(key=lambda successor: successor["successor_id"])
            source = root / "labels.jsonl"; source.write_bytes(fixtures.corpus.canonical_json_bytes(row))
            bundle = bundle_fixture(root)
            index = store.build_store([source], root / "store", bundle)
            labels = store.RankingStore(index, bundle).labels(); groups = labels.validation
            cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, groups)
            self.assertTrue(cache.immutable_mapped_groups)
            quantized = trainer.quantize_fixed(self.parameters, self.architecture, self.scales)
            expected = trainer.successor_ranking_metrics(self.parameters, self.architecture, groups, quantized=quantized)
            original = trainer.forward
            with (mock.patch.object(trainer, "_eager_float_ranking_group_content", side_effect=AssertionError("mapped content must not be rehashed")),
                  mock.patch.object(trainer, "forward", wraps=original) as forward):
                for _ in range(2):
                    actual = trainer.successor_ranking_metrics(self.parameters, self.architecture, groups,
                        quantized=quantized, _float_best_cache=cache)
                    self.assert_json_equal(actual, expected)
            self.assertEqual(sum(call.kwargs.get("quantized") is None for call in forward.call_args_list), 1)
            self.assertEqual(sum(call.kwargs.get("quantized") is not None for call in forward.call_args_list), 2)

            class SpoofedMappedSuccessors:
                def __init__(self, wrapped):
                    self.wrapped = wrapped; self.store = wrapped.store
                    self.begin = wrapped.begin; self.end = wrapped.end; self.root_termination = wrapped.root_termination
                def __len__(self): return len(self.wrapped)
                def __getitem__(self, item): return self.wrapped[item]

            spoof = types.ModuleType("mutable_ranking_cache_fixture")
            SpoofedMappedSuccessors.__module__ = spoof.__name__
            spoof.MappedSuccessors = SpoofedMappedSuccessors
            spoof.RankingStore = type(groups[0].successors.store)
            changed = (dataclasses.replace(groups[0], successors=SpoofedMappedSuccessors(groups[0].successors)),)
            with mock.patch.dict(sys.modules, {spoof.__name__: spoof}):
                cache = trainer._FloatRankingDecisionCache(self.parameters, self.architecture, changed)
                self.assertFalse(cache.immutable_mapped_groups)
                with mock.patch.object(trainer, "_eager_float_ranking_group_content", wraps=trainer._eager_float_ranking_group_content) as content:
                    for _ in range(2):
                        actual = trainer.successor_ranking_metrics(self.parameters, self.architecture, changed,
                            quantized=quantized, _float_best_cache=cache)
                        self.assert_json_equal(actual, expected)
                    self.assertEqual(content.call_count, 2)


class SeedWorkerOrchestrationTests(unittest.TestCase):
    @staticmethod
    def inputs(*, successor_mode: bool):
        inputs = SuccessorRankingTests.ranking_inputs()
        return inputs if successor_mode else trainer.dataclasses.replace(
            inputs, successor_rankings=None
        )

    @staticmethod
    def run_roster(inputs, workers):
        return trainer._train_seed_roster(
            mock.Mock(),
            inputs,
            trainer.ARCHITECTURES["capacity-12x8"],
            trainer.ARMS["search-target"],
            pathlib.Path("/unused"),
            seed_workers=workers,
            sidecar_index=None,
            ranking_weight=0.0,
            initial_checkpoint=None,
            resume=True,
        )

    def test_successor_seed_pool_caps_concurrency_and_preserves_seed_order(self):
        lock = threading.Lock()
        active = 0
        maximum = 0
        input_ids = set()
        native_executions = []

        def fake(_bundle, _inputs, _architecture, _arm, seed, _output, **_kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                input_ids.add(id(_inputs))
                native_executions.append(_kwargs["_native_thread_execution"])
            try:
                # Force completion order to differ from the frozen seed order.
                time.sleep({
                    trainer.FIXED_SEEDS[0]: 0.03,
                    trainer.FIXED_SEEDS[1]: 0.01,
                    trainer.FIXED_SEEDS[2]: 0.0,
                }[seed])
                return {"seed": seed}
            finally:
                with lock:
                    active -= 1

        with mock.patch.object(
            trainer, "train_seed_candidate", side_effect=fake
        ):
            receipts = self.run_roster(
                self.inputs(successor_mode=True), 2
            )
        self.assertEqual(maximum, 2)
        self.assertEqual(len(input_ids), 1)
        self.assertEqual(len(native_executions), len(trainer.FIXED_SEEDS))
        self.assertTrue(all(
            execution == native_executions[0]
            for execution in native_executions
        ))
        self.assertEqual(
            trainer.validate_native_thread_execution(native_executions[0])[
                "native_threads_per_seed_maximum"
            ],
            1,
        )
        self.assertEqual(
            [receipt["seed"] for receipt in receipts],
            list(trainer.FIXED_SEEDS),
        )

    def test_native_thread_scope_fails_closed_after_environment_drift(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with self.assertRaisesRegex(
                trainer.TrainingError, "before NumPy import"
            ):
                with trainer.native_thread_execution_scope():
                    self.fail("drifted native-thread environment was accepted")

    def test_seed_pool_propagates_failure_and_cancels_pending_work(self):
        failure = trainer.TrainingError("synthetic seed failure")

        def fake(_bundle, _inputs, _architecture, _arm, seed, _output, **_kwargs):
            if seed == trainer.FIXED_SEEDS[1]:
                raise failure
            time.sleep(0.01)
            return {"seed": seed}

        with mock.patch.object(
            trainer, "train_seed_candidate", side_effect=fake
        ), self.assertRaisesRegex(trainer.TrainingError, "synthetic seed failure"):
            self.run_roster(self.inputs(successor_mode=True), 2)

    def test_legacy_defaults_to_serial_and_worker_policy_rejects_bad_values(self):
        active = 0
        maximum = 0
        order = []

        def fake(_bundle, _inputs, _architecture, _arm, seed, _output, **_kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            order.append(seed)
            active -= 1
            return {"seed": seed}

        with mock.patch.object(
            trainer, "train_seed_candidate", side_effect=fake
        ):
            receipts = self.run_roster(
                self.inputs(successor_mode=False), 1
            )
        self.assertEqual(maximum, 1)
        self.assertEqual(order, list(trainer.FIXED_SEEDS))
        self.assertEqual(receipts, [{"seed": seed} for seed in trainer.FIXED_SEEDS])
        self.assertEqual(
            trainer._seed_worker_count(2, successor_mode=False), 2
        )
        for value in (0, 3, True):
            with self.subTest(value=value), self.assertRaises(
                trainer.TrainingError
            ):
                trainer._seed_worker_count(value, successor_mode=False)
        with self.assertRaisesRegex(
            trainer.TrainingError, "exactly two"
        ):
            trainer._seed_worker_count(1, successor_mode=True)

    def test_rich_corpus_aggregate_loads_and_unknown_evidence_rejects(self):
        from tools import jacek_replay_corpus as corpus
        from tests.codingame.test_jacek_replay_corpus import (
            JacekReplayCorpusTests,
        )

        fixture = JacekReplayCorpusTests()
        train = fixture.complete_turn_action_group_row(root_action="0")
        validation = fixture.complete_turn_action_group_row(
            position_id="position:" + "b" * 64,
            split="validation",
            root_action="2",
        )
        document = corpus.build_complete_turn_successor_labels(
            [train, validation]
        )
        document["source_bundle_body_sha256"] = "a" * 64
        body = dict(document)
        body.pop("body_sha256")
        document["body_sha256"] = corpus.sha256_bytes(
            corpus.canonical_json_bytes(body)
        )
        labels = trainer.validate_successor_label_document(
            document,
            source_bundle_body_sha256="a" * 64,
            artifact_sha256="b" * 64,
        )
        self.assertEqual(len(labels.train), 1)
        self.assertEqual(len(labels.validation), 1)
        successor = labels.train[0].successors[0]
        self.assertNotEqual(labels.train[0].parent_mover, successor.value_mover)
        self.assertIn("proof", successor.evidence)
        self.assertIn("work_budget", labels.train[0].evidence)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            payload = trainer.canonical_json_bytes(document)
            digest = trainer.sha256_bytes(payload)
            path = root / f"{digest}.successor-labels.json"
            path.write_bytes(payload)
            loaded = trainer.load_successor_ranking_labels(
                path, bundle_fixture(root)
            )
            self.assertEqual(loaded.artifact_sha256, digest)

        changed = copy.deepcopy(document)
        changed["splits"]["train"][0]["unknown_evidence"] = True
        changed_body = dict(changed)
        changed_body.pop("body_sha256")
        changed["body_sha256"] = corpus.sha256_bytes(
            corpus.canonical_json_bytes(changed_body)
        )
        with self.assertRaisesRegex(
            trainer.TrainingError, "rich successor-label|rich complete-turn"
        ):
            trainer.validate_successor_label_document(
                changed,
                source_bundle_body_sha256="a" * 64,
                artifact_sha256="b" * 64,
            )


class PackingAndRuntimeTests(unittest.TestCase):
    def quantized(self, name: str = "compact-8x8"):
        architecture = trainer.ARCHITECTURES[name]
        rng = np.random.default_rng(19)
        return architecture, trainer.QuantizedWeights(
            {
                tensor: rng.integers(-3, 4, size=shape, dtype=np.int8)
                for tensor, shape in architecture.shapes.items()
            },
            {
                "w1": np.float32(0.01),
                "w2": np.float32(0.02),
                "w3": np.float32(0.03),
            },
        )

    def document(self):
        architecture, quantized = self.quantized()
        return trainer.runtime_document(
            architecture,
            quantized,
            arm="search-target",
            seed=20260907,
            float_epoch=2,
            qat_epoch=0,
            source_bundle_body_sha256="a" * 64,
        )

    def test_signed_three_bit_roundtrip_rejects_minus_four_and_corruption(self) -> None:
        values = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
        payload = trainer.pack_signed_three_bit(values)
        np.testing.assert_array_equal(
            trainer.unpack_signed_three_bit(payload, len(values)), values
        )
        with self.assertRaisesRegex(trainer.TrainingError, "forbidden"):
            trainer.pack_signed_three_bit([-4])
        with self.assertRaisesRegex(trainer.TrainingError, "100"):
            trainer.unpack_signed_three_bit(bytes([0b00000100]), 1)
        with self.assertRaisesRegex(trainer.TrainingError, "padding"):
            trainer.unpack_signed_three_bit(bytes([0b10000000]), 1)
        with self.assertRaisesRegex(trainer.TrainingError, "expected"):
            trainer.unpack_signed_three_bit(payload + b"\0", len(values))

    def test_runtime_roundtrip_and_content_address(self) -> None:
        architecture, quantized = self.quantized()
        with tempfile.TemporaryDirectory() as temporary:
            path = trainer.write_runtime(
                pathlib.Path(temporary),
                architecture,
                quantized,
                arm="search-target",
                seed=20260907,
                float_epoch=2,
                qat_epoch=0,
                source_bundle_body_sha256="a" * 64,
            )
            loaded_architecture, loaded, selection, _ = trainer.load_runtime(path)
            self.assertEqual(loaded_architecture, architecture)
            self.assertEqual(selection["seed"], 20260907)
            for name in ("w1", "w2", "w3"):
                np.testing.assert_array_equal(
                    loaded.integer[name], quantized.integer[name]
                )

    def test_runtime_rejects_bad_scale_hash_length_and_forbidden_code(self) -> None:
        document = self.document()
        bad_scale = copy.deepcopy(document)
        bad_scale["quantization"]["scales"]["w1"] = 0.0
        body = dict(bad_scale)
        body.pop("body_sha256")
        bad_scale = trainer.body_hashed(body)
        with self.assertRaisesRegex(trainer.TrainingError, "scale"):
            trainer.validate_runtime_document(bad_scale)

        bad_hash = copy.deepcopy(document)
        bad_hash["quantization"]["payload_sha256"] = "0" * 64
        body = dict(bad_hash)
        body.pop("body_sha256")
        bad_hash = trainer.body_hashed(body)
        with self.assertRaisesRegex(trainer.TrainingError, "length or hash"):
            trainer.validate_runtime_document(bad_hash)

        forbidden = copy.deepcopy(document)
        raw = bytearray(base64.b64decode(forbidden["quantization"]["payload_base64"]))
        raw[0] = (raw[0] & ~0b111) | 0b100
        forbidden["quantization"]["payload_base64"] = base64.b64encode(raw).decode()
        forbidden["quantization"]["payload_sha256"] = trainer.sha256_bytes(raw)
        body = dict(forbidden)
        body.pop("body_sha256")
        forbidden = trainer.body_hashed(body)
        with self.assertRaisesRegex(trainer.TrainingError, "100"):
            trainer.validate_runtime_document(forbidden)

    def test_scalar_uses_activation_times_scale_then_integer(self) -> None:
        architecture, quantized = self.quantized()
        row = active_row(0, 1)
        observed = trainer.scalar_quantized_forward(
            quantized, architecture, row
        )
        q1, q2, q3 = (
            quantized.integer[name] for name in ("w1", "w2", "w3")
        )
        first = np.empty(architecture.hidden_one, dtype=np.float32)
        for hidden in range(architecture.hidden_one):
            total = sum(int(q1[int(index), hidden]) for index in row)
            value = np.float32(total * quantized.scales["w1"])
            first[hidden] = value * value if value >= 0 else np.float32(.01) * value
        second = np.empty(architecture.hidden_two, dtype=np.float32)
        for output in range(architecture.hidden_two):
            total = np.float32(0)
            for hidden in range(architecture.hidden_one):
                term = np.float32(
                    np.float32(first[hidden] * quantized.scales["w2"])
                    * np.float32(q2[hidden, output])
                )
                total = np.float32(total + term)
            second[output] = total if total >= 0 else np.float32(.01) * total
        total = np.float32(0)
        for hidden in range(architecture.hidden_two):
            term = np.float32(
                np.float32(second[hidden] * quantized.scales["w3"])
                * np.float32(q3[hidden])
            )
            total = np.float32(total + term)
        self.assertEqual(observed.tobytes(), trainer._fast_tanh_scalar(total).tobytes())


class GateAndProtectionTests(unittest.TestCase):
    @staticmethod
    def reports(common_sign, common_huber, canonical_sign, canonical_huber):
        return {
            "common_adjudicator": {
                "sign_accuracy": common_sign,
                "weighted_huber": common_huber,
            },
            "canonical_validation": {
                "sign_accuracy": canonical_sign,
                "weighted_huber": canonical_huber,
            },
        }

    def test_offline_thresholds_and_strict_relative_sign_loss(self) -> None:
        floating = self.reports(.8524, .054, .8662, .053)
        quantized = self.reports(.8475, .055, .8613, .054)
        self.assertTrue(
            trainer.offline_advancement_gate(floating, quantized)["passed"]
        )
        quantized = self.reports(.8474, .055, .8613, .054)
        self.assertFalse(
            trainer.offline_advancement_gate(floating, quantized)["passed"]
        )
        exact_loss = self.reports(.8474, .055, .8612, .054)
        floating_exact = self.reports(.8524, .054, .8662, .053)
        report = trainer.offline_advancement_gate(floating_exact, exact_loss)
        self.assertIn("sign loss", " ".join(report["errors"]))

    def test_protected_route_rejects_before_any_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = bundle_fixture(pathlib.Path(temporary))
            protected = bundle.routes["pilot_search_manifests"][2]
            with mock.patch.object(
                pathlib.Path,
                "resolve",
                side_effect=AssertionError("protected path was resolved"),
            ) as resolve:
                with self.assertRaisesRegex(trainer.TrainingError, "locked"):
                    bundle.artifact_path(protected)
                resolve.assert_not_called()

    def test_teacher_sidecars_are_limited_to_declared_unprotected_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = bundle_fixture(pathlib.Path(temporary))
            self.assertEqual(
                bundle.sidecar_role(bundle.routes["pilot_search_manifests"][0]),
                "train",
            )
            self.assertEqual(
                bundle.sidecar_role(bundle.routes["canonical_splits"]["validation"][0]),
                "canonical-validation",
            )
            with self.assertRaisesRegex(trainer.TrainingError, "limited"):
                bundle.sidecar_role(bundle.routes["pilot_search_manifests"][1])
            with self.assertRaisesRegex(trainer.TrainingError, "limited"):
                bundle.sidecar_role(bundle.routes["pilot_search_manifests"][2])


class SmokeTrainingTests(unittest.TestCase):
    def test_tiny_float_quantized_qat_smoke_and_earlier_qat_tie_policy(self) -> None:
        rows = [active_row(index % 4, index % 8) for index in range(4)]
        new = dataset(rows, [1.0, -1.0, .5, -.5])
        anchor = dataset(
            [active_row((index + 1) % 4, (index + 2) % 8) for index in range(4)],
            [.8, -.8, .4, -.4],
            groups=[f"anchor-{index}" for index in range(4)],
        )
        common = dataset(
            [active_row(5, 10), active_row(6, 11)],
            [1.0, -1.0],
            split="validation",
            groups=["common-a", "common-b"],
        )
        canonical = dataset(
            [active_row(7, 12), active_row(8, 13)],
            [.7, -.7],
            split="validation",
            groups=["canonical-a", "canonical-b"],
        )
        inputs = trainer.TrainingInputs(
            new, anchor, common, canonical, source_routes={}
        )
        architecture = trainer.ARCHITECTURES["compact-8x8"]
        arm = trainer.ARMS["search-target"]
        floating = trainer.train_float_seed(
            inputs,
            architecture,
            arm,
            20260907,
            maximum_epochs=1,
            patience=1,
        )
        self.assertEqual(floating.epoch, 1)
        with mock.patch.object(
            trainer,
            "_train_mixed_batch",
            return_value=0.0,
        ):
            quantized = trainer.run_fixed_scale_qat(
                floating, inputs, architecture, arm, 20260907
            )
        self.assertEqual(quantized.qat_epoch, 1)
        self.assertFalse(quantized.report["pre_qat_retained"])
        self.assertEqual(
            quantized.report["tie_break"],
            "prefer-earlier-qat-epoch-on-exact-tie",
        )

    def test_float_checkpoint_is_byte_stable_and_strict(self) -> None:
        architecture = trainer.ARCHITECTURES["compact-8x8"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            first = trainer.write_float_checkpoint(
                directory, parameters, architecture
            )
            second = trainer.write_float_checkpoint(
                directory, parameters, architecture
            )
            self.assertEqual(first, second)
            loaded = trainer.load_float_checkpoint(first, architecture)
            for name in parameters:
                np.testing.assert_array_equal(loaded[name], parameters[name])

    def test_completed_seed_resumes_but_interrupted_seed_restarts_epoch_zero(self) -> None:
        architecture = trainer.ARCHITECTURES["compact-8x8"]
        arm = trainer.ARMS["search-target"]
        parameters = trainer.initialize_parameters(architecture, 20260907)
        validation = {
            "common_adjudicator": {
                "samples": 4096,
                "weighted_huber": .05,
                "objective_weighted_huber": .05,
                "sign_accuracy": .86,
            },
            "canonical_validation": {
                "samples": 1,
                "weighted_huber": .05,
                "objective_weighted_huber": .05,
                "sign_accuracy": .87,
            },
        }
        floating = trainer.FloatTrainingResult(
            parameters,
            1,
            validation,
            {"best_float_epoch": 1},
        )
        quantized = trainer.quantize_fixed(
            parameters,
            architecture,
            {"w1": .01, "w2": .02, "w3": .03},
        )
        qat = trainer.QuantizedTrainingResult(
            quantized,
            0,
            validation,
            {
                "qat_profile": trainer.STANDARD_QAT_PROFILE,
                "qat_profile_contract": trainer.qat_profile_contract(
                    trainer.STANDARD_QAT_PROFILE
                ),
                "selected_qat_epoch": 0,
                "selected_scales": {
                    name: float(quantized.scales[name])
                    for name in ("w1", "w2", "w3")
                },
                "scale_search": {
                    "selected_scales": {
                        name: float(quantized.scales[name])
                        for name in ("w1", "w2", "w3")
                    }
                },
            },
        )
        repeated = [active_row(0, 1)] * 4096
        common = dataset(
            repeated,
            [1.0] * 4096,
            split="validation",
            groups=[f"resume-common-{index}" for index in range(4096)],
        )
        inputs = trainer.TrainingInputs(
            new=dataset([active_row(1, 2)]),
            anchor=dataset(
                [active_row(2, 3)], groups=["resume-anchor"]
            ),
            common_adjudicator=common,
            canonical_validation=dataset(
                [active_row(3, 4)],
                split="validation",
                groups=["resume-validation"],
            ),
            source_routes={
                "new": ("pilot.json", "full.json"),
                "anchor": ("r0.json", "r1.json", "r2.json"),
            },
        )
        bundle = mock.Mock(body_sha256="b" * 64)
        gate = {
            "passed": True,
            "status": "offline-evaluator-qualified-not-game-gated",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            patches = (
                mock.patch.object(trainer, "train_float_seed", return_value=floating),
                mock.patch.object(trainer, "run_fixed_scale_qat", return_value=qat),
                mock.patch.object(
                    trainer,
                    "validate_qat_execution_evidence",
                    side_effect=lambda value, **_kwargs: dict(value),
                ),
                mock.patch.object(trainer, "offline_advancement_gate", return_value=gate),
                mock.patch.object(
                    trainer,
                    "assert_quantized_inference_parity",
                    return_value={"states": 4096, "maximum_absolute_error": 0.0},
                ),
            )
            with (
                patches[0] as train_call,
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                first = trainer.train_seed_candidate(
                    bundle, inputs, architecture, arm, 20260907, output
                )
            self.assertEqual(train_call.call_count, 1)
            with self.assertRaisesRegex(trainer.TrainingError, "--resume"):
                trainer.train_seed_candidate(
                    bundle, inputs, architecture, arm, 20260907, output
                )
            with mock.patch.object(
                trainer,
                "train_float_seed",
                side_effect=AssertionError("completed seed retrained"),
            ), mock.patch.object(
                trainer,
                "validate_qat_execution_evidence",
                side_effect=lambda value, **_kwargs: dict(value),
            ):
                resumed = trainer.train_seed_candidate(
                    bundle,
                    inputs,
                    architecture,
                    arm,
                    20260907,
                    output,
                    resume=True,
                )
            self.assertEqual(resumed["body_sha256"], first["body_sha256"])

            second_output = output / "interrupted"
            (second_output / "float-checkpoints").mkdir(parents=True)
            (second_output / "float-checkpoints" / "orphan").write_bytes(b"orphan")
            with (
                patches[0] as restarted,
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                trainer.train_seed_candidate(
                    bundle,
                    inputs,
                    architecture,
                    arm,
                    20260907,
                    second_output,
                    resume=True,
                )
            self.assertEqual(restarted.call_count, 1)


if __name__ == "__main__":
    unittest.main()
