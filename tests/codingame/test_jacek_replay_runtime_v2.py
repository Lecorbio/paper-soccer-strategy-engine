import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import numpy as np
    import jacek_replay_train as training
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    training = None


@unittest.skipIf(np is None, "research tests require requirements-research.txt")
class JacekReplayRuntimeV2Tests(unittest.TestCase):
    @staticmethod
    def _parity_parameters():
        base = {
            "w1": np.zeros(
                (training.features.INPUT_COUNT, training.HIDDEN_ONE),
                dtype=np.float32,
            ),
            "w2": np.zeros(
                (training.HIDDEN_ONE, training.HIDDEN_TWO), dtype=np.float32
            ),
            "w3": np.zeros(training.HIDDEN_TWO, dtype=np.float32),
        }
        base["w1"][5, 0] = np.float32(2.0)
        base["w2"][0, 0] = np.float32(0.5)
        base["w3"][0] = np.float32(0.25)
        parameters = training.initialize_residual_adapter(base)
        parameters["base_gain"][...] = np.float32(1.2)
        parameters["residual_bias"][...] = np.float32(0.1)
        parameters["adapter_a"].fill(0.0)
        parameters["adapter_a"][0, 0] = np.float32(-0.5)
        parameters["adapter_b"][0] = np.float32(3.0)
        return base, parameters

    @staticmethod
    def _mixed_fixture():
        datasets = {
            split: training.Dataset.from_active(
                tuple(
                    np.asarray(sample.active, dtype=np.int32)
                    for sample in samples
                ),
                np.asarray(
                    [sample.target for sample in samples], dtype=np.float32
                ),
                np.asarray(
                    [sample.weight for sample in samples], dtype=np.float32
                ),
                tuple(sample.group_id for sample in samples),
            )
            for split, samples in training.tiny_fixture_samples().items()
        }
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(4)),
            datasets["train"].targets[:4],
            datasets["train"].weights[:4],
            tuple(f"v2-new:{index}" for index in range(4)),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(4, 8)),
            datasets["train"].targets[4:8],
            datasets["train"].weights[4:8],
            tuple(f"v2-anchor:{index}" for index in range(4)),
        )
        return datasets, training.MixedTraining(new, anchor, 3, 1)

    def test_v1_pack_bytes_and_prediction_value_are_unchanged(self):
        path = ROOT / "models" / "jacek_replay_bfm_bootstrap.runtime"
        original = path.read_bytes()
        parameters, report = training.load_runtime(path)
        repacked, repacked_report = training.runtime_bytes(parameters)
        active = np.asarray(
            training.features.encode_active(training.features.ReplayState()),
            dtype=np.int32,
        )
        prediction, _ = training.forward(parameters, [active])

        self.assertEqual(repacked, original)
        self.assertEqual(repacked_report, report)
        self.assertNotIn("runtime_version", report)
        # Repacking is byte-exact above. The derived float32 reduction may
        # drift slightly across supported NumPy/compiler combinations.
        expected_prediction = struct.unpack(
            "<f", struct.pack("<f", 0.0007981307)
        )[0]
        self.assertAlmostEqual(
            float(prediction[0]), expected_prediction, delta=2e-9
        )

    def test_zero_adapter_is_deterministic_and_bit_exact(self):
        base = training.initialize(919)
        first = training.initialize_residual_adapter(base)
        second = training.initialize_residual_adapter(base)
        active = [
            np.asarray([1, 17, 316, 6200], dtype=np.int32),
            np.asarray([3, 44, 987], dtype=np.int32),
        ]
        base_prediction, _ = training.forward(base, active)
        adapted_prediction, _ = training.forward(first, active)

        self.assertTrue(np.array_equal(first["adapter_a"], second["adapter_a"]))
        self.assertTrue(np.any(first["adapter_a"]))
        self.assertFalse(np.any(first["adapter_b"]))
        self.assertEqual(base_prediction.tobytes(), adapted_prediction.tobytes())
        self.assertEqual(
            training.runtime_bytes(first)[0], training.runtime_bytes(second)[0]
        )

    def test_v2_round_trip_and_python_cpp_parity_golden(self):
        _, parameters = self._parity_parameters()
        active = [np.asarray([5], dtype=np.int32)]
        prediction, _ = training.forward(parameters, active)
        self.assertEqual(prediction.tobytes(), struct.pack("<I", 0x3F109D42))

        artifact, expected = training.runtime_bytes(parameters)
        self.assertEqual(artifact[:8], training.RUNTIME_V2_MAGIC)
        self.assertEqual(expected["runtime_version"], 2)
        self.assertEqual(expected["residual_rank"], 16)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "adapter.runtime"
            path.write_bytes(artifact)
            loaded, observed = training.load_runtime(path)
        loaded_prediction, _ = training.forward(loaded, active)
        self.assertEqual(expected, observed)
        self.assertEqual(training.runtime_bytes(loaded)[0], artifact)
        self.assertEqual(loaded_prediction.tobytes(), prediction.tobytes())

    def test_adapter_gradient_updates_leave_base_bit_exact(self):
        base, parameters = self._parity_parameters()
        frozen = {name: value.copy() for name, value in base.items()}
        initial_a = parameters["adapter_a"].copy()
        optimizer = training.AdamW(parameters, learning_rate=0.001, weight_decay=1e-5)
        active = [np.asarray([5], dtype=np.int32)]
        for _ in range(2):
            training._apply_training_batch(
                parameters,
                optimizer,
                active,
                np.asarray([0.9], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                huber_delta=0.25,
            )

        for name in training.BASE_PARAMETER_NAMES:
            self.assertTrue(np.array_equal(parameters[name], frozen[name]))
        self.assertFalse(np.array_equal(parameters["adapter_a"], initial_a))
        self.assertNotEqual(float(parameters["base_gain"]), 1.2)
        self.assertNotEqual(float(parameters["residual_bias"]), 0.1)

    def test_malformed_v2_is_rejected(self):
        _, parameters = self._parity_parameters()
        artifact, _ = training.runtime_bytes(parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "adapter.runtime"

            wrong_count = bytearray(artifact)
            struct.pack_into(
                "<Q", wrong_count, 48, training.RUNTIME_V2_WEIGHT_COUNT - 1
            )
            path.write_bytes(wrong_count)
            with self.assertRaisesRegex(ValueError, "header contract"):
                training.load_runtime(path)

            nonfinite = bytearray(artifact)
            gain_offset = training.RUNTIME_HEADER.size + training.WEIGHT_COUNT * 4
            struct.pack_into("<f", nonfinite, gain_offset, float("nan"))
            nonfinite[88:120] = hashlib.sha256(
                nonfinite[training.RUNTIME_HEADER.size :]
            ).digest()
            path.write_bytes(nonfinite)
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                training.load_runtime(path)

            path.write_bytes(artifact + b"x")
            with self.assertRaisesRegex(ValueError, "truncated or has trailing"):
                training.load_runtime(path)

    def test_v2_mixed_checkpoints_bind_contract_freeze_base_and_reject_tamper(self):
        datasets, mixed = self._mixed_fixture()
        initial = training.initialize_residual_adapter(training.initialize(1701))
        initial["base_gain"][...] = np.float32(0.97)
        initial["residual_bias"][...] = np.float32(0.02)
        initial["adapter_b"][0] = np.float32(0.01)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            initial_runtime = root / "v7-actor.runtime"
            checkpoints = root / "v7-seeds"
            initial_report = training.export_runtime(initial_runtime, initial)
            arguments = {
                "seeds": (1701, 1702, 1703),
                "epochs": 2,
                "patience": 1,
                "batch_size": 4,
                "mixed_training": mixed,
                "selection_validation": datasets["validation"],
                "retention_validation": datasets["test"],
                "initial_runtime": initial_runtime,
                "seed_checkpoint_directory": checkpoints,
                "input_shard_identities": ({"fixture": "v7-from-v2"},),
            }
            selected, report = training.train_three_seeds({}, **arguments)

            contract = training._residual_runtime_contract()
            self.assertEqual(report["initial_runtime"], initial_report)
            self.assertAlmostEqual(initial_report["base_gain"], 0.97, places=6)
            self.assertAlmostEqual(initial_report["residual_bias"], 0.02, places=6)
            self.assertEqual(report["initialization"]["runtime_version"], 2)
            self.assertEqual(
                report["initialization"]["payload_layout"],
                training.RUNTIME_V2_PAYLOAD_LAYOUT,
            )
            self.assertEqual(
                report["initialization"]["residual_adapter"],
                contract["residual_adapter"],
            )
            self.assertEqual(
                report["optimizer"]["trainable_parameters"],
                list(training.RESIDUAL_PARAMETER_NAMES),
            )
            self.assertEqual(
                report["optimizer"]["frozen_parameters"],
                list(training.BASE_PARAMETER_NAMES),
            )
            self.assertTrue(
                training._base_parameters_are_byte_identical(selected, initial)
            )
            for seed in arguments["seeds"]:
                checkpoint, checkpoint_report = training.load_runtime(
                    checkpoints / f"seed-{seed}.runtime"
                )
                receipt = json.loads(
                    (checkpoints / f"seed-{seed}.json").read_bytes()
                )
                self.assertTrue(
                    training._base_parameters_are_byte_identical(
                        checkpoint, initial
                    )
                )
                self.assertEqual(
                    checkpoint_report["base_payload_sha256"],
                    initial_report["base_payload_sha256"],
                )
                self.assertEqual(
                    receipt["configuration"]["architecture"]["runtime_version"],
                    2,
                )
                self.assertEqual(
                    receipt["configuration"]["optimizer"]["frozen_parameters"],
                    list(training.BASE_PARAMETER_NAMES),
                )

            resumed, resumed_report = training.train_three_seeds(
                {}, resume_seeds=True, **arguments
            )
            self.assertEqual(training.runtime_bytes(selected)[0],
                             training.runtime_bytes(resumed)[0])
            self.assertEqual(report, resumed_report)

            seed = arguments["seeds"][0]
            checkpoint_path = checkpoints / f"seed-{seed}.runtime"
            receipt_path = checkpoints / f"seed-{seed}.json"
            tampered, _ = training.load_runtime(checkpoint_path)
            tampered["w1"][0, 0] += np.float32(0.125)
            tampered_payload, tampered_report = training.runtime_bytes(tampered)
            checkpoint_path.write_bytes(tampered_payload)
            receipt = json.loads(receipt_path.read_bytes())
            receipt["checkpoint"] = {
                "file": checkpoint_path.name,
                **tampered_report,
            }
            receipt["training_report"]["checkpoint"] = tampered_report
            body = dict(receipt)
            body.pop("body_sha256")
            receipt["body_sha256"] = hashlib.sha256(
                training.canonical_json_bytes(body)
            ).hexdigest()
            receipt_path.write_bytes(training.canonical_json_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "frozen base parameters changed"):
                training.train_three_seeds({}, resume_seeds=True, **arguments)

    def test_v7_from_v2_cli_manifest_declares_residual_payload_and_freeze_mask(self):
        samples = training.tiny_fixture_samples()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            shards = root / "shards"
            new_manifest = training.write_csr_shard(
                shards, "train", samples["train"][:4]
            )[1]
            anchor_manifest = training.write_csr_shard(
                shards, "train", samples["train"][4:]
            )[1]
            validation_manifest = training.write_csr_shard(
                shards, "validation", samples["validation"]
            )[1]
            retention_manifest = training.write_csr_shard(
                shards, "validation", samples["test"]
            )[1]
            initial = training.initialize_residual_adapter(training.initialize(2701))
            actor = root / "qualified-v2.runtime"
            initial_report = training.export_runtime(actor, initial)
            output = root / "v7-model"
            checkpoints = root / "v7-checkpoints"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "jacek_replay_train.py"),
                    "--new-shard-manifest", str(new_manifest),
                    "--anchor-shard-manifest", str(anchor_manifest),
                    "--selection-validation-manifest", str(validation_manifest),
                    "--retention-validation-manifest", str(retention_manifest),
                    "--new-rows-per-batch", "2",
                    "--anchor-rows-per-batch", "2",
                    "--batch-size", "4",
                    "--initial-runtime", str(actor),
                    "--seeds", "2701,2702,2703",
                    "--epochs", "1",
                    "--patience", "1",
                    "--seed-checkpoint-directory", str(checkpoints),
                    "--output-directory", str(output),
                ],
                check=True,
                capture_output=True,
            )
            manifest = json.loads(
                (output / "jacek_replay_bfm.runtime.json").read_bytes()
            )
            selected, selected_report = training.load_runtime(
                output / "jacek_replay_bfm.runtime"
            )

            self.assertEqual(manifest["schema"], training.RETENTION_MODEL_MANIFEST_SCHEMA)
            self.assertEqual(manifest["architecture"]["runtime_version"], 2)
            self.assertEqual(
                manifest["architecture"]["biases"],
                {
                    "base_network": False,
                    "residual_output_parameter": "residual_bias",
                },
            )
            self.assertEqual(
                manifest["architecture"]["payload_layout"],
                training.RUNTIME_V2_PAYLOAD_LAYOUT,
            )
            self.assertEqual(
                manifest["architecture"]["residual_adapter"]["rank"], 16
            )
            self.assertEqual(
                manifest["architecture"]["parameter_update"],
                {
                    "trainable": list(training.RESIDUAL_PARAMETER_NAMES),
                    "frozen": list(training.BASE_PARAMETER_NAMES),
                },
            )
            self.assertEqual(manifest["runtime"], {
                "path": "jacek_replay_bfm.runtime", **selected_report
            })
            self.assertEqual(
                selected_report["base_payload_sha256"],
                initial_report["base_payload_sha256"],
            )
            self.assertTrue(
                training._base_parameters_are_byte_identical(selected, initial)
            )
            self.assertEqual(
                manifest["training"]["initialization"]["kind"],
                "exact-supplied-runtime-v2-adapter-only-all-seeds-order-only-v1",
            )


if __name__ == "__main__":
    unittest.main()
