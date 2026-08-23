import argparse
import hashlib
import inspect
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_pack as pack  # noqa: E402
try:
    import numpy as np
    import jacek_replay_train as training
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    training = None


@unittest.skipIf(np is None, "research tests require requirements-research.txt")
class JacekReplayTrainingTests(unittest.TestCase):
    @staticmethod
    def _tiny_datasets():
        return {
            split: training.Dataset.from_active(
                tuple(np.asarray(sample.active, dtype=np.int32) for sample in samples),
                np.asarray([sample.target for sample in samples], dtype=np.float32),
                np.asarray([sample.weight for sample in samples], dtype=np.float32),
                tuple(sample.group_id for sample in samples),
            )
            for split, samples in training.tiny_fixture_samples().items()
        }

    def test_protected_test_reveal_is_opt_in(self):
        parameter = inspect.signature(training.train_three_seeds).parameters[
            "reveal_test"
        ]
        self.assertIs(parameter.default, False)
        workers = inspect.signature(training.train_three_seeds).parameters[
            "seed_workers"
        ]
        self.assertEqual(workers.default, 1)

    def test_checked_bootstrap_is_loadable_and_explicitly_untrained(self):
        runtime_path = ROOT / "models" / "jacek_replay_bfm_bootstrap.runtime"
        manifest_path = runtime_path.with_suffix(".runtime.json")
        parameters, report = training.load_runtime(runtime_path)
        manifest = json.loads(manifest_path.read_bytes())
        self.assertEqual(manifest["status"], "bootstrap-not-trained-not-selected")
        self.assertEqual(manifest["runtime"]["artifact_sha256"], report["artifact_sha256"])
        self.assertEqual(
            report["artifact_sha256"],
            "02c7757443285cf6d10e971f25dee0869339d76e1d09bb9616ce5b83966cabf7",
        )
        source_paths = {
            "trainer": ROOT / "tools" / "jacek_replay_train.py",
            "corpus": ROOT / "tools" / "jacek_replay_corpus.py",
            "features": ROOT / "tools" / "jacek_replay_features.py",
        }
        for name, path in source_paths.items():
            self.assertEqual(
                manifest["tool_sha256"][name], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        active = np.asarray(
            training.features.encode_active(training.features.ReplayState()),
            dtype=np.int32,
        )
        prediction, _ = training.forward(parameters, [active])
        self.assertAlmostEqual(float(prediction[0]), 0.000798130698967725, places=10)

    def test_csr_shard_is_deterministic_and_strictly_typed(self):
        fixture = training.tiny_fixture_samples()["train"]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            first_npz, first_manifest, _ = training.write_csr_shard(
                path, "train", fixture, provenance={"fixture": "v1"}
            )
            first_bytes = first_npz.read_bytes()
            second_npz, second_manifest, _ = training.write_csr_shard(
                path, "train", fixture, provenance={"fixture": "v1"}
            )
            second_bytes = second_npz.read_bytes()
            shard = training.load_csr_shard(first_manifest)
        self.assertEqual(first_npz, second_npz)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(shard.indices.dtype, np.dtype("<u2"))
        self.assertEqual(shard.indptr.dtype, np.dtype("<i8"))
        self.assertEqual(shard.targets.dtype, np.dtype("<f4"))
        self.assertEqual(shard.group_ids.dtype, np.dtype("V32"))
        self.assertEqual(len(shard), len(fixture))

    def test_runtime_header_and_input_major_payload_are_frozen(self):
        parameters = {
            "w1": np.zeros((6301, 192), dtype=np.float32),
            "w2": np.zeros((192, 32), dtype=np.float32),
            "w3": np.zeros(32, dtype=np.float32),
        }
        parameters["w1"][1, 2] = 3.25
        artifact, report = training.runtime_bytes(parameters)
        self.assertEqual(len(artifact), 4_864_000)
        self.assertEqual(report["weight_count"], 1_215_968)
        self.assertEqual(artifact[:8], b"JRBFM\0\0\x01")
        offset = training.RUNTIME_HEADER.size + (1 * 192 + 2) * 4
        self.assertEqual(struct.unpack_from("<f", artifact, offset)[0], 3.25)

    def test_weighted_huber_uses_weight_sum_normalization(self):
        loss, gradient = training._weighted_huber(
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 3.0], dtype=np.float32),
            0.25,
        )
        self.assertAlmostEqual(loss, 0.1640625)
        self.assertTrue(np.allclose(gradient, [0.0, 0.1875]))

    def test_metrics_streams_bounded_batches(self):
        samples = training.tiny_fixture_samples()["train"]
        dataset = training.Dataset.from_active(
            tuple(np.asarray(sample.active, dtype=np.int32) for sample in samples),
            np.asarray([sample.target for sample in samples], dtype=np.float32),
            np.ones(len(samples), dtype=np.float32),
            tuple(sample.group_id for sample in samples),
        )
        self.assertEqual(dataset.indices.dtype, np.dtype(np.uint16))
        self.assertEqual(dataset.group_ids.dtype, np.dtype(np.uint64))
        self.assertEqual(len(dataset.indptr), len(samples) + 1)
        original = training.forward
        observed = []

        def recording_forward(parameters, active):
            observed.append(len(active))
            return original(parameters, active)

        training.forward = recording_forward
        try:
            result = training.metrics(training.initialize(3), dataset, batch_size=3)
        finally:
            training.forward = original
        self.assertEqual(result["samples"], 8)
        self.assertEqual(observed, [3, 3, 2])

    def test_trainer_rejects_overlap_reintroduced_by_multiple_shards(self):
        sample = training.tiny_fixture_samples()["train"][0]
        reflected = corpus.LabeledSample(
            training.features.reflect_active(sample.active),
            sample.target,
            sample.weight,
            "another-root",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            train_manifest = training.write_csr_shard(
                root, "train", [sample]
            )[1]
            validation_manifest = training.write_csr_shard(
                root, "validation", [reflected]
            )[1]
            shards = [
                training.load_csr_shard(train_manifest),
                training.load_csr_shard(validation_manifest),
            ]
        with self.assertRaisesRegex(ValueError, "overlap crosses splits"):
            training.validate_shard_collection(shards)

    def test_trainer_allows_independent_same_state_across_round_shards(self):
        sample = training.tiny_fixture_samples()["train"][0]
        extra = training.tiny_fixture_samples()["train"][1]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifests = [
                training.write_csr_shard(root, "train", [sample])[1],
                training.write_csr_shard(root, "train", [sample, extra])[1],
            ]
            shards = [training.load_csr_shard(path) for path in manifests]
        training.validate_shard_collection(shards)
        self.assertEqual(len(training.combine_shards(shards)), 3)

    def test_teacher_jsonl_packs_to_three_content_addressed_shards(self):
        groups = (("root:train", "train"), ("root:val", "validation"), ("root:test", "test"))
        prefixes = ([], [{"player_id": 0, "action": "0"}], [
            {"player_id": 0, "action": "0"},
            {"player_id": 1, "action": "0"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            roots_path = root / "roots.json"
            roots_manifest = {
                "schema": corpus.ROOT_SCHEMA,
                "feature_schema": training.features.FEATURE_SCHEMA,
                "tool_sha256": {
                    "normalizer": "1" * 64,
                    "features": "2" * 64,
                },
                "exclusion_boundary": {"read_before_candidate_sources": True},
                "accepted": [
                    {"group_id": group, "split": split}
                    for group, split in groups
                ],
            }
            roots_manifest["body_sha256"] = corpus.sha256_bytes(
                corpus.canonical_json_bytes(roots_manifest)
            )
            roots_path.write_bytes(corpus.canonical_json_bytes(roots_manifest))
            teacher_path = root / "teacher.jsonl"
            lines = []
            for (group, _), prefix in zip(groups, prefixes):
                lines.append(corpus.canonical_json_bytes({
                    "schema": corpus.TEACHER_SCHEMA,
                    "group_id": group,
                    "source": "fixture",
                    "winner": 0,
                    "prefix": prefix,
                    "mover": len(prefix) % 2,
                    "root_score": 100.0,
                    "completed_depth": 2,
                    "nodes": 32_000,
                }))
            teacher_path.write_bytes(b"".join(lines))
            report = pack.pack_teacher_rows(
                roots_path=roots_path,
                teacher_paths=[teacher_path],
                output_directory=root / "shards",
            )
            self.assertEqual(set(report["tool_sha256"]), {"pack", "corpus", "features"})
            self.assertEqual(
                report["same_orientation_rows_aggregated"],
                {"train": 1, "validation": 1, "test": 1},
            )
            shards = {
                split: training.load_csr_shard(pathlib.Path(metadata["manifest"]))
                for split, metadata in report["shards"].items()
            }
            streaming_report = pack.pack_teacher_rows_streaming(
                roots_path=roots_path,
                teacher_paths=[teacher_path],
                output_directory=root / "streaming-shards",
            )
            streaming_shards = {
                split: training.load_csr_shard(pathlib.Path(metadata["manifest"]))
                for split, metadata in streaming_report["shards"].items()
            }
            repeated_streaming = pack.pack_teacher_rows_streaming(
                roots_path=roots_path,
                teacher_paths=[teacher_path],
                output_directory=root / "streaming-shards-repeat",
            )
            current_teacher = root / "teacher-current.jsonl"
            current_rows = (
                ("root:val", []),  # Crosses the prior train initial state.
                ("root:val", prefixes[1]),  # Same prior validation split.
                ("root:train", [{"player_id": 0, "action": "1"}]),
                ("root:test", prefixes[2]),
            )
            current_teacher.write_bytes(b"".join(
                corpus.canonical_json_bytes({
                    "schema": corpus.TEACHER_SCHEMA,
                    "group_id": group,
                    "source": "fixture-current",
                    "winner": 0,
                    "prefix": prefix,
                    "mover": len(prefix) % 2,
                    "root_score": 100.0,
                    "completed_depth": 2,
                    "nodes": 32_000,
                })
                for group, prefix in current_rows
            ))
            prior_manifests = [
                pathlib.Path(streaming_report["shards"][split]["manifest"])
                for split in ("train", "validation", "test")
            ]
            cumulative_report = pack.pack_teacher_rows_streaming(
                roots_path=roots_path,
                teacher_paths=[current_teacher],
                output_directory=root / "current-shards",
                prior_shard_manifests=prior_manifests,
            )
            cumulative_shards = [
                training.load_csr_shard(pathlib.Path(
                    cumulative_report["shards"][split]["manifest"]
                ))
                for split in ("train", "validation", "test")
            ]
            training.validate_shard_collection(
                [*streaming_shards.values(), *cumulative_shards]
            )
        self.assertEqual(set(shards), {"train", "validation", "test"})
        # All three fixture boundaries are horizontally symmetric, so their
        # reflection rows collapse to one canonical row per split.
        self.assertTrue(all(len(shard) == 1 for shard in shards.values()))
        self.assertTrue(all(float(shard.weights[0]) == 2.0 for shard in shards.values()))
        self.assertEqual(
            streaming_report["packing"], "sqlite-streaming-bounded-memory-v1"
        )
        self.assertEqual(
            cumulative_report["prior_cross_split_rows_removed"],
            {"train": 0, "validation": 2, "test": 0},
        )
        self.assertEqual(len(cumulative_report["prior_shards"]), 3)
        self.assertEqual(
            {
                split: streaming_report["shards"][split]["sha256"]
                for split in ("train", "validation", "test")
            },
            {
                split: repeated_streaming["shards"][split]["sha256"]
                for split in ("train", "validation", "test")
            },
        )
        for split in ("train", "validation", "test"):
            expected = {
                tuple(shards[split].active(row)): (
                    float(shards[split].targets[row]),
                    float(shards[split].weights[row]),
                )
                for row in range(len(shards[split]))
            }
            observed = {
                tuple(streaming_shards[split].active(row)): (
                    float(streaming_shards[split].targets[row]),
                    float(streaming_shards[split].weights[row]),
                )
                for row in range(len(streaming_shards[split]))
            }
            self.assertEqual(expected, observed)

    def test_streaming_round_can_emit_empty_delta_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            roots = {
                "schema": corpus.ROOT_SCHEMA,
                "feature_schema": training.features.FEATURE_SCHEMA,
                "tool_sha256": {
                    "normalizer": "1" * 64,
                    "features": "2" * 64,
                },
                "exclusion_boundary": {"read_before_candidate_sources": True},
                "accepted": [{"group_id": "root:train", "split": "train"}],
            }
            roots["body_sha256"] = corpus.sha256_bytes(
                corpus.canonical_json_bytes(roots)
            )
            roots_path = root / "roots.json"
            roots_path.write_bytes(corpus.canonical_json_bytes(roots))
            teacher_path = root / "teacher.jsonl"
            teacher_path.write_bytes(corpus.canonical_json_bytes({
                "schema": corpus.TEACHER_SCHEMA,
                "group_id": "root:train",
                "source": "fixture",
                "winner": 0,
                "prefix": [],
                "mover": 0,
                "root_score": 0.0,
                "completed_depth": 1,
                "nodes": 1,
            }))
            report = pack.pack_teacher_rows_streaming(
                roots_path=roots_path,
                teacher_paths=[teacher_path],
                output_directory=root / "shards",
            )
            validation = training.load_csr_shard(pathlib.Path(
                report["shards"]["validation"]["manifest"]
            ))
        self.assertEqual(len(validation), 0)
        self.assertEqual(len(training.combine_shards([validation])), 0)

    def test_runtime_round_trip_and_corruption_rejection(self):
        parameters = training.initialize(7)
        artifact, expected = training.runtime_bytes(parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.runtime"
            path.write_bytes(artifact)
            loaded, observed = training.load_runtime(path)
            self.assertEqual(expected, observed)
            self.assertTrue(np.array_equal(parameters["w1"], loaded["w1"]))

            path.write_bytes(artifact + b"x")
            with self.assertRaisesRegex(ValueError, "truncated or has trailing"):
                training.load_runtime(path)

            corrupt = bytearray(artifact)
            corrupt[120] = 1
            path.write_bytes(corrupt)
            with self.assertRaisesRegex(ValueError, "header contract"):
                training.load_runtime(path)

            nonfinite = bytearray(artifact)
            nonfinite[training.RUNTIME_HEADER.size : training.RUNTIME_HEADER.size + 4] = struct.pack("<f", float("nan"))
            nonfinite[88:120] = hashlib.sha256(nonfinite[128:]).digest()
            path.write_bytes(nonfinite)
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                training.load_runtime(path)

    def test_tiny_fixture_runs_exactly_three_seed_selection(self):
        selected, report = training.train_three_seeds(
            self._tiny_datasets(),
            seeds=(11, 12, 13),
            epochs=1,
            patience=1,
            batch_size=8,
            reveal_test=True,
        )
        self.assertEqual(selected["w1"].shape, (6301, 192))
        self.assertEqual(report["seeds"], [11, 12, 13])
        self.assertTrue(report["test_revealed_after_selection"])
        self.assertEqual(
            sum("test" in candidate for candidate in report["seed_reports"]), 1
        )
        self.assertTrue(
            all("checkpoint" in candidate for candidate in report["seed_reports"])
        )
        self.assertEqual(report["optimizer"]["name"], "adamw")

    def test_two_seed_workers_are_byte_identical_to_sequential_training(self):
        datasets = self._tiny_datasets()
        arguments = {
            "seeds": (13, 11, 12),
            "epochs": 2,
            "patience": 2,
            "batch_size": 4,
            "reveal_test": True,
        }
        sequential, sequential_report = training.train_three_seeds(
            datasets, seed_workers=1, **arguments
        )
        parallel, parallel_report = training.train_three_seeds(
            datasets, seed_workers=2, **arguments
        )
        self.assertEqual(
            training.runtime_bytes(sequential)[0], training.runtime_bytes(parallel)[0]
        )
        self.assertEqual(
            training.canonical_json_bytes(sequential_report),
            training.canonical_json_bytes(parallel_report),
        )
        self.assertEqual(
            [item["seed"] for item in parallel_report["seed_reports"]],
            [13, 11, 12],
        )
        self.assertEqual(
            sum("test" in item for item in parallel_report["seed_reports"]), 1
        )

    def test_cli_parallel_runtime_and_manifest_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sequential = root / "sequential"
            parallel = root / "parallel"
            sequential_checkpoints = root / "sequential-seeds"
            parallel_checkpoints = root / "parallel-seeds"
            command = [
                sys.executable,
                str(ROOT / "tools" / "jacek_replay_train.py"),
                "--tiny-fixture",
            ]
            subprocess.run(
                [
                    *command,
                    "--output-directory",
                    str(sequential),
                    "--seed-workers",
                    "1",
                    "--seed-checkpoint-directory",
                    str(sequential_checkpoints),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    *command,
                    "--output-directory",
                    str(parallel),
                    "--seed-workers",
                    "2",
                    "--seed-checkpoint-directory",
                    str(parallel_checkpoints),
                ],
                check=True,
                capture_output=True,
            )
            for name in (
                "jacek_replay_bfm.runtime",
                "jacek_replay_bfm.runtime.json",
            ):
                self.assertEqual(
                    (sequential / name).read_bytes(), (parallel / name).read_bytes()
                )
            checkpoint_bytes = {
                path.name: path.read_bytes() for path in parallel_checkpoints.iterdir()
            }
            checkpoint_mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in parallel_checkpoints.iterdir()
            }
            subprocess.run(
                [
                    *command,
                    "--output-directory",
                    str(parallel),
                    "--seed-workers",
                    "2",
                    "--seed-checkpoint-directory",
                    str(parallel_checkpoints),
                    "--resume-seeds",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                checkpoint_bytes,
                {
                    path.name: path.read_bytes()
                    for path in parallel_checkpoints.iterdir()
                },
            )
            self.assertEqual(
                checkpoint_mtimes,
                {
                    path.name: path.stat().st_mtime_ns
                    for path in parallel_checkpoints.iterdir()
                },
            )

    def test_interrupted_seed_set_resumes_only_missing_seeds(self):
        datasets = self._tiny_datasets()
        arguments = {
            "seeds": (11, 12, 13),
            "epochs": 2,
            "patience": 2,
            "batch_size": 4,
            "reveal_test": True,
            "input_shard_identities": ({"fixture": "resume-v1"},),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = pathlib.Path(directory) / "seeds"
            initial, initial_report = training.train_three_seeds(
                datasets,
                seed_workers=1,
                seed_checkpoint_directory=checkpoints,
                **arguments,
            )
            preserved_path = checkpoints / "seed-11.json"
            preserved_bytes = preserved_path.read_bytes()
            preserved_mtime = preserved_path.stat().st_mtime_ns
            for seed in (12, 13):
                (checkpoints / f"seed-{seed}.runtime").unlink()
                (checkpoints / f"seed-{seed}.json").unlink()

            trained_seeds = []
            original = training.train_seed

            def recording_train_seed(all_datasets, seed, **kwargs):
                trained_seeds.append(seed)
                return original(all_datasets, seed, **kwargs)

            training.train_seed = recording_train_seed
            try:
                resumed, resumed_report = training.train_three_seeds(
                    datasets,
                    seed_workers=1,
                    seed_checkpoint_directory=checkpoints,
                    resume_seeds=True,
                    **arguments,
                )
            finally:
                training.train_seed = original

            self.assertEqual(trained_seeds, [12, 13])
            self.assertEqual(preserved_bytes, preserved_path.read_bytes())
            self.assertEqual(preserved_mtime, preserved_path.stat().st_mtime_ns)
            self.assertEqual(
                training.runtime_bytes(initial)[0], training.runtime_bytes(resumed)[0]
            )
            self.assertEqual(
                training.canonical_json_bytes(initial_report),
                training.canonical_json_bytes(resumed_report),
            )
            self.assertEqual(len(initial_report["seed_checkpoints"]), 3)
            for seed in (11, 12, 13):
                receipt = json.loads((checkpoints / f"seed-{seed}.json").read_bytes())
                self.assertNotIn("test", receipt["training_report"])

    def test_seed_resume_rejects_stale_corrupt_and_incomplete_receipts(self):
        datasets = self._tiny_datasets()
        arguments = {
            "seeds": (11, 12, 13),
            "epochs": 1,
            "patience": 1,
            "batch_size": 8,
            "seed_checkpoint_directory": None,
            "input_shard_identities": ({"fixture": "corruption-v1"},),
        }
        with self.assertRaisesRegex(ValueError, "requires seed_checkpoint_directory"):
            training.train_three_seeds(datasets, resume_seeds=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoints = pathlib.Path(directory) / "seeds"
            arguments["seed_checkpoint_directory"] = checkpoints
            training.train_three_seeds(datasets, **arguments)
            receipt_path = checkpoints / "seed-11.json"
            runtime_path = checkpoints / "seed-11.runtime"
            receipt_bytes = receipt_path.read_bytes()
            runtime_bytes = runtime_path.read_bytes()

            receipt = json.loads(receipt_bytes)
            receipt["training_report"]["best_epoch"] = 999
            receipt_path.write_bytes(training.canonical_json_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "receipt integrity failed"):
                training.train_three_seeds(datasets, resume_seeds=True, **arguments)
            receipt_path.write_bytes(receipt_bytes)

            corrupt_runtime = bytearray(runtime_bytes)
            corrupt_runtime[-1] ^= 1
            runtime_path.write_bytes(corrupt_runtime)
            with self.assertRaisesRegex(ValueError, "parameters are corrupt"):
                training.train_three_seeds(datasets, resume_seeds=True, **arguments)
            runtime_path.write_bytes(runtime_bytes)

            with self.assertRaisesRegex(ValueError, "configuration is stale"):
                training.train_three_seeds(
                    datasets,
                    resume_seeds=True,
                    **{**arguments, "epochs": 2},
                )

            changed = self._tiny_datasets()
            changed["train"].targets[0] += np.float32(0.01)
            with self.assertRaisesRegex(ValueError, "inputs is stale"):
                training.train_three_seeds(
                    changed, resume_seeds=True, **arguments
                )

            receipt_path.unlink()
            with self.assertRaisesRegex(ValueError, "checkpoint is incomplete"):
                training.train_three_seeds(datasets, resume_seeds=True, **arguments)

    def test_seed_worker_count_must_be_positive_integer(self):
        datasets = self._tiny_datasets()
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    training.train_three_seeds(datasets, seed_workers=invalid)
        for invalid in ("0", "-2", "many"):
            with self.subTest(cli_value=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    training._parse_seed_workers(invalid)

    def test_parallel_worker_failure_is_contextual_and_fail_closed(self):
        datasets = self._tiny_datasets()
        datasets["train"] = training.Dataset(
            np.asarray([0], dtype=np.int64),
            np.asarray([], dtype=np.uint16),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1], dtype=np.uint64),
        )
        with self.assertRaisesRegex(RuntimeError, "training worker for seed") as raised:
            training.train_three_seeds(
                datasets,
                seeds=(11, 12, 13),
                epochs=1,
                patience=1,
                batch_size=1,
                seed_workers=2,
            )
        self.assertIsNotNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
