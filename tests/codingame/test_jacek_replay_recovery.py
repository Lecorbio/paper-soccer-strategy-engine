import copy
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import numpy as np
    import jacek_replay_corpus as corpus
    import jacek_replay_recovery as recovery
    import jacek_replay_train as training
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    corpus = None
    recovery = None
    training = None


@unittest.skipIf(np is None, "recovery tests require requirements-research.txt")
class JacekReplayRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temporary.name)
        cls.runtime = cls.root / "initial.runtime"
        training.export_runtime(cls.runtime, training.initialize(20260826))
        cls.manifests = {}
        counts = {"new": 64, "anchor": 100, "selection": 5, "retention": 5}
        for role, count in counts.items():
            split = "train" if role in {"new", "anchor"} else "validation"
            _, manifest, _ = training.write_csr_shard(
                cls.root / role,
                split,
                cls._samples(role, count),
                provenance={"fixture": role},
            )
            cls.manifests[role] = manifest
        recovery.clear_recovery_input_cache()
        cls.inputs = cls._prepare()

    @classmethod
    def tearDownClass(cls):
        recovery.clear_recovery_input_cache()
        cls.temporary.cleanup()

    @staticmethod
    def _active(role_index, row):
        active = [
            training.features.EDGE_COUNT
            + vertex * training.features.VERTEX_CATEGORIES
            + (
                role_index * 13
                + row * 7
                + vertex * vertex * 3
                + vertex * 11
                + row * vertex
            )
            % training.features.VERTEX_CATEGORIES
            for vertex in range(training.features.VERTEX_COUNT)
        ]
        active.append(
            (role_index * 71 + row * 17) % training.features.EDGE_COUNT
        )
        return tuple(sorted(active))

    @classmethod
    def _samples(cls, role, count):
        role_index = {"new": 0, "anchor": 1, "selection": 2, "retention": 3}[role]
        return [
            corpus.LabeledSample(
                cls._active(role_index, row),
                -0.75 if row % 2 else 0.75,
                1.0 + (row % 3) * 0.25,
                f"recovery:{role}:{row}",
            )
            for row in range(count)
        ]

    @classmethod
    def _prepare(cls):
        return recovery.prepare_recovery_inputs(
            initial_runtime=cls.runtime,
            retention_reference_runtime=cls.runtime,
            new_reference_runtime=cls.runtime,
            new_manifests=[cls.manifests["new"]],
            anchor_manifests=[cls.manifests["anchor"]],
            selection_manifests=[cls.manifests["selection"]],
            retention_manifests=[cls.manifests["retention"]],
        )

    def test_fixed_contract_and_aliases_are_receipt_bound(self):
        configuration = recovery.RecoveryConfiguration(
            "w2+w3", 3e-6, 17, recovery.V5_NONINFERIORITY
        )
        report = recovery._configuration_report(self.inputs, configuration)
        self.assertEqual(report["optimizer"]["trainable_layers"], "w2-w3")
        self.assertEqual(report["batching"]["batch_size"], 256)
        self.assertEqual(report["batching"]["new_rows_per_batch"], 64)
        self.assertEqual(report["batching"]["anchor_rows_per_batch"], 192)
        self.assertEqual(report["batching"]["checkpoint_interval_updates"], 782)
        self.assertEqual(report["batching"]["maximum_anchor_passes"], 2)
        self.assertEqual(
            report["loss"],
            {
                "name": "separately-normalized-weighted-huber",
                "delta": 0.25,
                "new_coefficient": 0.25,
                "anchor_coefficient": 0.75,
            },
        )
        with self.assertRaisesRegex(ValueError, "trainable_layers"):
            recovery.RecoveryConfiguration("w1", 3e-6, 17).normalized()

    def test_v6_joint_inverse_update_learning_rate_is_exact(self):
        reference = recovery.v6_joint_learning_rate_policy(50_000)
        self.assertEqual(reference["reference_optimizer_steps"], 782)
        self.assertEqual(reference["actual_optimizer_steps"], 782)
        self.assertEqual(reference["scale"], 1.0)
        self.assertEqual(reference["effective_learning_rate"], 6e-5)
        small = recovery.v6_joint_learning_rate_policy(64)
        self.assertEqual(small["effective_learning_rate"], 6e-5)
        full = recovery.v6_joint_learning_rate_policy(250_000)
        self.assertEqual(full["actual_optimizer_steps"], 3_907)
        self.assertAlmostEqual(
            full["effective_learning_rate"], 6e-5 * 782 / 3_907
        )

    def test_v6_joint_configuration_binds_epoch_recipe(self):
        report = recovery._v6_joint_configuration_report(
            self.inputs, recovery.V6JointConfiguration(47)
        )
        self.assertEqual(report["optimizer"]["maximum_epochs"], 50)
        self.assertEqual(report["optimizer"]["patience"], 8)
        self.assertTrue(
            report["optimizer"][
                "patience_starts_after_complete_anchor_coverage"
            ]
        )
        self.assertEqual(
            report["batching"]["epoch_length"],
            "ceil-new-rows/new-quota-batches",
        )
        self.assertEqual(
            report["batching"]["new_stream"],
            "fresh-complete-permutation-each-epoch-with-padding",
        )
        self.assertEqual(
            report["batching"]["anchor_cross_epoch"],
            "continuous-no-repeat-until-permutation-complete",
        )
        self.assertEqual(report["selection"]["policy"], "epoch-zero-improvement")

    def test_prepared_inputs_are_read_only_cached_and_revalidated(self):
        self.assertFalse(self.inputs.anchor.targets.flags.writeable)
        self.assertFalse(self.inputs.initial_parameters["w1"].flags.writeable)
        with self.assertRaises(ValueError):
            self.inputs.anchor.targets[0] = 1.0
        with self.assertRaises(ValueError):
            self.inputs.anchor.targets.setflags(write=True)
        with self.assertRaises(ValueError):
            self.inputs.initial_parameters["w1"].setflags(write=True)
        with mock.patch.object(
            recovery,
            "_validate_role_isolation",
            side_effect=AssertionError("content cache was not reused"),
        ):
            second = self._prepare()
        self.assertIs(second.anchor, self.inputs.anchor)
        self.assertIs(second.initial_parameters, self.inputs.initial_parameters)

        prepared_datasets = recovery.prepare_recovery_datasets(
            new_manifests=[self.manifests["new"]],
            anchor_manifests=[self.manifests["anchor"]],
            selection_manifests=[self.manifests["selection"]],
            retention_manifests=[self.manifests["retention"]],
        )
        with mock.patch.object(
            recovery,
            "_manifest_probe",
            side_effect=AssertionError("runtime binding rehashed a dataset"),
        ):
            rebound = recovery.bind_recovery_runtimes(
                prepared_datasets,
                initial_runtime=self.runtime,
                retention_reference_runtime=self.runtime,
                new_reference_runtime=self.runtime,
            )
        self.assertIs(rebound.anchor, prepared_datasets.anchor)
        self.assertEqual(
            rebound.receipt_identity()["datasets"],
            prepared_datasets.receipt_identity()["datasets"],
        )

        # Cache reuse must still hash the backing NPZ and reject changed bytes.
        manifest = json.loads(self.manifests["selection"].read_bytes())
        npz = self.manifests["selection"].parent / manifest["npz"]
        original = npz.read_bytes()
        try:
            npz.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self._prepare()
        finally:
            npz.write_bytes(original)

    def test_cross_role_group_or_fingerprint_leakage_is_rejected(self):
        recovery.clear_recovery_input_cache()
        with self.assertRaisesRegex(ValueError, "crosses new and anchor"):
            recovery.prepare_recovery_inputs(
                initial_runtime=self.runtime,
                retention_reference_runtime=self.runtime,
                new_reference_runtime=self.runtime,
                new_manifests=[self.manifests["new"]],
                anchor_manifests=[self.manifests["new"]],
                selection_manifests=[self.manifests["selection"]],
                retention_manifests=[self.manifests["retention"]],
            )
        # Restore the ordinary bundle to the process cache for later tests.
        type(self).inputs = self._prepare()

    def test_content_identical_manifest_copy_cannot_double_weight_rows(self):
        source_manifest = self.manifests["new"]
        manifest = json.loads(source_manifest.read_bytes())
        copied_directory = self.root / "copied-new-shard"
        copied_directory.mkdir()
        copied_manifest = copied_directory / source_manifest.name
        copied_npz = copied_directory / manifest["npz"]
        shutil.copyfile(source_manifest, copied_manifest)
        shutil.copyfile(source_manifest.parent / manifest["npz"], copied_npz)
        with self.assertRaisesRegex(ValueError, "content-identical"):
            recovery.prepare_recovery_datasets(
                new_manifests=[source_manifest, copied_manifest],
                anchor_manifests=[self.manifests["anchor"]],
                selection_manifests=[self.manifests["selection"]],
                retention_manifests=[self.manifests["retention"]],
            )

    def test_freeze_mask_limits_optimizer_state_and_parameter_updates(self):
        parameters = {
            name: value.copy()
            for name, value in self.inputs.initial_parameters.items()
        }
        before = {name: value.copy() for name, value in parameters.items()}
        optimizer = training.AdamW({"w3": parameters["w3"]}, 1e-3, 1e-5)
        mixed = training.MixedTraining(
            self.inputs.new, self.inputs.anchor, 64, 192
        )
        loss = recovery.train_recovery_batch(
            parameters,
            optimizer,
            mixed,
            np.arange(64, dtype=np.int64),
            np.arange(192, dtype=np.int64) % len(self.inputs.anchor),
            trainable_layers="w3",
        )
        self.assertTrue(math_is_finite_nonnegative(loss))
        self.assertEqual(optimizer.step, 1)
        self.assertEqual(set(optimizer.first), {"w3"})
        self.assertTrue(np.array_equal(parameters["w1"], before["w1"]))
        self.assertTrue(np.array_equal(parameters["w2"], before["w2"]))
        self.assertFalse(np.array_equal(parameters["w3"], before["w3"]))

    def test_dual_gate_policies_and_runtime_hash_tie_break(self):
        reference = {
            "weighted_huber": 0.2,
            "sign_accuracy": 0.8,
            "correlation": 0.1,
        }
        thresholds = recovery._thresholds(reference)
        self.assertEqual(thresholds["minimum_sign_accuracy"], 0.795)
        self.assertAlmostEqual(thresholds["maximum_weighted_huber"], 0.204)
        self.assertTrue(
            recovery.new_gate_passes(
                recovery.V5_NONINFERIORITY,
                {**reference, "weighted_huber": 0.204, "sign_accuracy": 0.795},
                epoch_zero=reference,
                reference_thresholds=thresholds,
            )
        )
        self.assertTrue(
            recovery.new_gate_passes(
                recovery.EPOCH_ZERO_IMPROVEMENT,
                {**reference, "weighted_huber": 0.199},
                epoch_zero=reference,
                reference_thresholds=thresholds,
            )
        )
        self.assertFalse(
            recovery.new_gate_passes(
                recovery.EPOCH_ZERO_IMPROVEMENT,
                dict(reference),
                epoch_zero=reference,
                reference_thresholds=thresholds,
            )
        )
        self.assertLess(
            recovery.selection_key(reference, "0" * 64),
            recovery.selection_key(reference, "f" * 64),
        )

    def test_training_resume_recomputes_metrics_and_rejects_tamper(self):
        configuration = recovery.RecoveryConfiguration(
            "w3", 1e-10, 23, recovery.V5_NONINFERIORITY
        )
        output = self.root / "completed-arm"
        selected, report = recovery.run_recovery(
            self.inputs, configuration, output
        )
        self.assertEqual(report["schema"], recovery.RECOVERY_REPORT_SCHEMA)
        self.assertEqual(report["result"]["status"], "eligible-checkpoint-selected")
        self.assertEqual(report["result"]["selected_update"], 1)
        self.assertTrue(report["checkpoints"][0]["coverage"]["new"]["complete_coverage"])
        self.assertTrue(report["checkpoints"][0]["coverage"]["anchor"]["complete_coverage"])
        self.assertTrue(np.array_equal(selected["w1"], self.inputs.initial_parameters["w1"]))
        self.assertTrue(np.array_equal(selected["w2"], self.inputs.initial_parameters["w2"]))

        resumed, resumed_report = recovery.run_recovery(
            self.inputs, configuration, output, resume=True
        )
        self.assertEqual(resumed_report, report)
        self.assertEqual(
            training.runtime_bytes(resumed)[1], report["result"]["runtime"]
        )
        with self.assertRaisesRegex(ValueError, "schedule, references, or inputs"):
            recovery.run_recovery(
                self.inputs,
                dataclasses_replace(configuration, learning_rate=3e-6),
                output,
                resume=True,
            )

        receipt_path = output / recovery.RECEIPT_NAME
        receipt = json.loads(receipt_path.read_bytes())
        receipt["body_sha256"] = "0" * 64
        receipt_path.write_bytes(training.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "integrity failed"):
            recovery.run_recovery(
                self.inputs, configuration, output, resume=True
            )

    def test_full_replay_rejects_integrity_consistent_transcript_forgery(self):
        configuration = recovery.RecoveryConfiguration(
            "w3", 1e-10, 31, recovery.V5_NONINFERIORITY
        )
        output = self.root / "forged-transcript-arm"
        recovery.run_recovery(self.inputs, configuration, output)
        report_path = output / recovery.REPORT_NAME
        receipt_path = output / recovery.RECEIPT_NAME
        report = json.loads(report_path.read_bytes())
        report["checkpoints"][0]["average_training_weighted_huber"] += 0.125
        report_payload = training.canonical_json_bytes(report)
        report_path.write_bytes(report_payload)
        receipt = json.loads(receipt_path.read_bytes())
        receipt["report"] = {
            "file": recovery.REPORT_NAME,
            "sha256": recovery._sha256(report_payload),
            "bytes": len(report_payload),
        }
        body = dict(receipt)
        body.pop("body_sha256")
        receipt["body_sha256"] = recovery._sha256(
            training.canonical_json_bytes(body)
        )
        receipt_path.write_bytes(training.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "full replay transcript disagrees"):
            recovery.run_recovery(
                self.inputs, configuration, output, resume=True
            )

    def test_malformed_integrity_consistent_report_fails_closed(self):
        configuration = recovery.RecoveryConfiguration(
            "w3", 1e-10, 37, recovery.V5_NONINFERIORITY
        )
        output = self.root / "malformed-transcript-arm"
        recovery.run_recovery(self.inputs, configuration, output)
        report_path = output / recovery.REPORT_NAME
        receipt_path = output / recovery.RECEIPT_NAME
        report = json.loads(report_path.read_bytes())
        report["checkpoints"][0]["retention_gate"] = []
        report_payload = training.canonical_json_bytes(report)
        report_path.write_bytes(report_payload)
        receipt = json.loads(receipt_path.read_bytes())
        receipt["report"] = {
            "file": recovery.REPORT_NAME,
            "sha256": recovery._sha256(report_payload),
            "bytes": len(report_payload),
        }
        body = dict(receipt)
        body.pop("body_sha256")
        receipt["body_sha256"] = recovery._sha256(
            training.canonical_json_bytes(body)
        )
        receipt_path.write_bytes(training.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "retention gate is invalid"):
            recovery.run_recovery(
                self.inputs, configuration, output, resume=True
            )

    def test_partial_publication_is_deterministically_reconstructed(self):
        configuration = recovery.RecoveryConfiguration(
            "w3", 1e-10, 41, recovery.V5_NONINFERIORITY
        )
        complete = self.root / "partial-source-arm"
        _, expected = recovery.run_recovery(
            self.inputs, configuration, complete
        )
        partial = self.root / "partial-reconstructed-arm"
        partial.mkdir()
        shutil.copyfile(
            complete / recovery.RUNTIME_NAME,
            partial / recovery.RUNTIME_NAME,
        )
        selected, reconstructed = recovery.run_recovery(
            self.inputs, configuration, partial, resume=True
        )
        self.assertEqual(reconstructed, expected)
        self.assertTrue((partial / recovery.REPORT_NAME).is_file())
        self.assertTrue((partial / recovery.RECEIPT_NAME).is_file())
        self.assertEqual(
            training.runtime_bytes(selected)[1], reconstructed["result"]["runtime"]
        )

    def test_integrity_consistent_extra_receipt_field_is_rejected(self):
        configuration = recovery.RecoveryConfiguration(
            "w3", 1e-10, 43, recovery.V5_NONINFERIORITY
        )
        output = self.root / "extra-receipt-field-arm"
        recovery.run_recovery(self.inputs, configuration, output)
        receipt_path = output / recovery.RECEIPT_NAME
        receipt = json.loads(receipt_path.read_bytes())
        receipt["forged_extra"] = "not-bound-by-the-schema"
        body = dict(receipt)
        body.pop("body_sha256")
        receipt["body_sha256"] = recovery._sha256(
            training.canonical_json_bytes(body)
        )
        receipt_path.write_bytes(training.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "receipt schema is invalid"):
            recovery.run_recovery(
                self.inputs, configuration, output, resume=True
            )

    def test_residual_fallback_is_separate_v2_and_freezes_the_base(self):
        with self.assertRaisesRegex(ValueError, "1e-4, 3e-4, or 1e-3"):
            recovery.ResidualRecoveryConfiguration(3e-6, 29).normalized()
        zero = training.initialize_residual_adapter(
            self.inputs.initial_parameters
        )
        base_prediction, _ = training.forward(
            self.inputs.initial_parameters,
            self.inputs.selection.active_rows(range(len(self.inputs.selection))),
        )
        zero_prediction, _ = training.forward(
            zero,
            self.inputs.selection.active_rows(range(len(self.inputs.selection))),
        )
        self.assertTrue(np.array_equal(base_prediction, zero_prediction))

        configuration = recovery.ResidualRecoveryConfiguration(1e-4, 29)
        output = self.root / "residual-arm"
        selected, report = recovery.run_residual_recovery(
            self.inputs, configuration, output
        )
        self.assertEqual(
            report["schema"], recovery.RESIDUAL_RECOVERY_REPORT_SCHEMA
        )
        self.assertEqual(report["result"]["runtime"]["runtime_version"], 2)
        self.assertEqual(report["result"]["runtime"]["residual_rank"], 16)
        self.assertEqual(
            set(selected),
            set((*training.BASE_PARAMETER_NAMES, *training.RESIDUAL_PARAMETER_NAMES)),
        )
        for name in training.BASE_PARAMETER_NAMES:
            self.assertTrue(
                np.array_equal(selected[name], self.inputs.initial_parameters[name])
            )
        self.assertEqual(
            report["configuration"]["optimizer"]["trainable_parameters"],
            list(training.RESIDUAL_PARAMETER_NAMES),
        )
        self.assertTrue((output / recovery.RESIDUAL_RUNTIME_NAME).is_file())
        receipt = json.loads(
            (output / recovery.RESIDUAL_RECEIPT_NAME).read_bytes()
        )
        self.assertEqual(
            receipt["schema"], recovery.RESIDUAL_RECOVERY_RECEIPT_SCHEMA
        )
        resumed, resumed_report = recovery.run_residual_recovery(
            self.inputs, configuration, output, resume=True
        )
        self.assertEqual(resumed_report, report)
        self.assertEqual(
            training.runtime_bytes(resumed)[1], report["result"]["runtime"]
        )

    def test_v6_joint_patience_starts_at_full_anchor_coverage(self):
        active = [
            self.inputs.anchor.active_row(row % len(self.inputs.anchor))
            for row in range(2_000)
        ]
        large_anchor = recovery._readonly_dataset(training.Dataset.from_active(
            active,
            np.resize(self.inputs.anchor.targets, 2_000),
            np.resize(self.inputs.anchor.weights, 2_000),
            tuple(f"large-anchor:{row}" for row in range(2_000)),
        ))
        inputs = dataclasses_replace(self.inputs, anchor=large_anchor)

        def fixed_metrics(_parameters, dataset, batch_size=4_096):
            del _parameters, batch_size
            return {
                "samples": len(dataset),
                "weighted_huber": 0.2,
                "sign_accuracy": 0.8,
                "correlation": 0.1,
                "mae": 0.3,
                "prediction_mean": 0.0,
            }

        with recovery._CACHE_LOCK:
            recovery._METRIC_CACHE.clear()
        with (
            mock.patch.object(recovery.training, "metrics", side_effect=fixed_metrics),
            mock.patch.object(recovery, "train_recovery_batch", return_value=0.1),
        ):
            _selected, report = recovery._train_v6_joint(
                inputs, recovery.V6JointConfiguration(53)
            )
        with recovery._CACHE_LOCK:
            recovery._METRIC_CACHE.clear()
        self.assertEqual(
            report["result"]["anchor_coverage_complete_epoch"], 11
        )
        self.assertEqual(report["result"]["epochs_completed"], 19)
        self.assertFalse(report["checkpoints"][9]["coverage"]["anchor"]
                         ["complete_permutations"])
        self.assertEqual(
            report["checkpoints"][10]["coverage"]["anchor"]
            ["complete_permutations"],
            1,
        )

    def test_v6_joint_artifacts_and_full_replay_resume_are_separate(self):
        def fixed_metrics(_parameters, dataset, batch_size=4_096):
            del _parameters, batch_size
            return {
                "samples": len(dataset),
                "weighted_huber": 0.2,
                "sign_accuracy": 0.8,
                "correlation": 0.1,
                "mae": 0.3,
                "prediction_mean": 0.0,
            }

        with recovery._CACHE_LOCK:
            recovery._METRIC_CACHE.clear()
        configuration = recovery.V6JointConfiguration(59)
        output = self.root / "v6-joint-arm"
        with (
            mock.patch.object(recovery.training, "metrics", side_effect=fixed_metrics),
            mock.patch.object(recovery, "train_recovery_batch", return_value=0.1),
        ):
            selected, report = recovery.run_v6_joint(
                self.inputs, configuration, output
            )
            resumed, resumed_report = recovery.run_v6_joint(
                self.inputs, configuration, output, resume=True
            )
        with recovery._CACHE_LOCK:
            recovery._METRIC_CACHE.clear()
        self.assertEqual(report["schema"], recovery.V6_JOINT_REPORT_SCHEMA)
        self.assertEqual(report["result"]["epochs_completed"], 9)
        self.assertEqual(report["result"]["selected_epoch"], 0)
        self.assertEqual(resumed_report, report)
        self.assertEqual(
            training.runtime_bytes(selected)[0], training.runtime_bytes(resumed)[0]
        )
        self.assertTrue((output / recovery.V6_JOINT_RUNTIME_NAME).is_file())
        receipt = json.loads((output / recovery.V6_JOINT_RECEIPT_NAME).read_bytes())
        self.assertEqual(receipt["schema"], recovery.V6_JOINT_RECEIPT_SCHEMA)


def math_is_finite_nonnegative(value):
    return isinstance(value, float) and np.isfinite(value) and value >= 0.0


def dataclasses_replace(value, **changes):
    # Kept local so the optional-NumPy import skip remains simple.
    import dataclasses

    return dataclasses.replace(value, **changes)


if __name__ == "__main__":
    unittest.main()
