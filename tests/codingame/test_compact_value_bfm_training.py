#!/usr/bin/env python3
"""Focused contracts for the compact value-BFM trainer."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

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
    def test_tiny_float_quantized_qat_smoke_and_pre_qat_tie_policy(self) -> None:
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
        self.assertEqual(quantized.qat_epoch, 0)
        self.assertTrue(quantized.report["pre_qat_retained"])

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
                "selected_qat_epoch": 0,
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
                mock.patch.object(trainer, "offline_advancement_gate", return_value=gate),
                mock.patch.object(
                    trainer,
                    "assert_quantized_inference_parity",
                    return_value={"states": 4096, "maximum_absolute_error": 0.0},
                ),
            )
            with patches[0] as train_call, patches[1], patches[2], patches[3]:
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
            with patches[0] as restarted, patches[1], patches[2], patches[3]:
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
