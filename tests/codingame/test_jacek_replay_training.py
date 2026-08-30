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
        # NumPy's float32 reduction may differ by one ULP across supported
        # compiler/platform combinations. The runtime bytes and manifest hash
        # above remain exact; keep the value regression narrowly bounded.
        self.assertAlmostEqual(
            float(prediction[0]), 0.000798130698967725, delta=2e-9
        )

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

    def test_mixed_batches_have_exact_quotas_and_deterministic_cycling(self):
        fixture = self._tiny_datasets()["train"]
        new = training.Dataset.from_active(
            fixture.active_rows(range(5)),
            fixture.targets[:5],
            fixture.weights[:5],
            tuple(f"new:{index}" for index in range(5)),
        )
        anchor = training.Dataset.from_active(
            fixture.active_rows(range(3)),
            fixture.targets[:3],
            fixture.weights[:3],
            tuple(f"anchor:{index}" for index in range(3)),
        )
        mixed = training.MixedTraining(new, anchor, 3, 1)
        first = training.mixed_epoch_batches(
            mixed, batch_size=4, seed=17, epoch=1
        )
        second = training.mixed_epoch_batches(
            mixed, batch_size=4, seed=17, epoch=1
        )
        self.assertEqual(len(first), 2)
        self.assertTrue(all(len(rows) == 3 and len(anchors) == 1
                            for rows, anchors in first))
        self.assertEqual(
            [[rows.tolist(), anchors.tolist()] for rows, anchors in first],
            [[rows.tolist(), anchors.tolist()] for rows, anchors in second],
        )
        self.assertEqual(set(np.concatenate([rows for rows, _ in first])), set(range(5)))
        self.assertEqual(len(np.concatenate([rows for _, rows in first])), 2)

        third = training.mixed_epoch_batches(
            mixed, batch_size=4, seed=17, epoch=2
        )
        self.assertEqual(
            set(np.concatenate([rows for rows, _ in third])), set(range(5))
        )
        anchor_stream = np.concatenate(
            [anchors for _, anchors in (*first, *third)]
        )
        # The anchor cursor crosses the epoch boundary without replacement.
        self.assertEqual(set(anchor_stream[:3]), set(range(3)))
        coverage = training.mixed_epoch_coverage(
            mixed, batch_size=4, epoch=2
        )
        self.assertEqual(coverage["new"]["complete_epoch_permutations"], 2)
        self.assertEqual(coverage["new"]["padding_rows_per_epoch"], 1)
        self.assertEqual(coverage["anchor"]["complete_permutations"], 1)

        # Targets differ between matched arms, but batch row order must not.
        other_new = training.Dataset(
            new.indptr,
            new.indices,
            -new.targets,
            new.weights,
            new.group_ids,
        )
        other = training.mixed_epoch_batches(
            training.MixedTraining(other_new, anchor, 3, 1),
            batch_size=4,
            seed=17,
            epoch=1,
        )
        self.assertEqual(
            [[rows.tolist(), anchors.tolist()] for rows, anchors in first],
            [[rows.tolist(), anchors.tolist()] for rows, anchors in other],
        )

    def test_mixed_training_uses_explicit_common_selection_validation(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(4)),
            datasets["train"].targets[:4],
            datasets["train"].weights[:4],
            tuple(f"new:{index}" for index in range(4)),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(4, 8)),
            datasets["train"].targets[4:8],
            datasets["train"].weights[4:8],
            tuple(f"anchor:{index}" for index in range(4)),
        )
        validation = datasets["validation"]
        with tempfile.TemporaryDirectory() as directory:
            initial_runtime = pathlib.Path(directory) / "initial.runtime"
            initial = training.initialize(99)
            training.export_runtime(initial_runtime, initial)
            selected, report = training.train_three_seeds(
                {},
                seeds=(31, 32, 33),
                epochs=1,
                patience=1,
                batch_size=4,
                mixed_training=training.MixedTraining(new, anchor, 2, 2),
                selection_validation=validation,
                retention_validation=datasets["test"],
                initial_runtime=initial_runtime,
            )
        self.assertEqual(selected["w1"].shape, (6301, 192))
        self.assertEqual(
            report["batching"],
            {
                "kind": "deterministic-continuous-two-stream-coverage-v2",
                "new_rows_per_batch": 2,
                "anchor_rows_per_batch": 2,
                "epoch_length": "ceil-new-rows/new-quota-batches",
                "row_order": "new-then-anchor",
                "new_stream": "fresh-complete-permutation-each-epoch-with-padding",
                "anchor_cross_epoch": (
                    "continuous-no-repeat-until-permutation-complete"
                ),
            },
        )
        self.assertEqual(
            report["selection_validation"]["kind"],
            "explicit-common-adjudicator",
        )
        self.assertTrue(all(
            seed_report["validation"]["samples"] == len(validation)
            for seed_report in report["seed_reports"]
        ))
        self.assertTrue(all(
            seed_report["retention"]["samples"] == len(datasets["test"])
            for seed_report in report["seed_reports"]
        ))
        self.assertTrue(all(
            seed_report["anchor_training"]["samples"] == len(anchor)
            for seed_report in report["seed_reports"]
        ))

    def test_source_normalized_loss_is_invariant_to_cross_source_weight_scale(self):
        new = np.asarray([1.0, 3.0], dtype=np.float32)
        anchor = np.asarray([1.0, 16_184.0], dtype=np.float32)
        first = training._source_normalized_weights(
            new,
            anchor,
            new_loss_coefficient=0.5,
            anchor_loss_coefficient=0.5,
        )
        second = training._source_normalized_weights(
            new,
            anchor * np.float32(100.0),
            new_loss_coefficient=0.5,
            anchor_loss_coefficient=0.5,
        )
        self.assertTrue(np.allclose(first, second))
        self.assertAlmostEqual(float(np.sum(first[:2])), 0.5, places=7)
        self.assertAlmostEqual(float(np.sum(first[2:])), 0.5, places=7)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            training._source_normalized_weights(
                new,
                anchor,
                new_loss_coefficient=0.6,
                anchor_loss_coefficient=0.5,
            )

    def test_64_192_quotas_are_exact_and_anchor_cursor_is_continuous(self):
        def empty_dataset(rows):
            return training.Dataset(
                np.zeros(rows + 1, dtype=np.int64),
                np.asarray([], dtype=np.uint16),
                np.zeros(rows, dtype=np.float32),
                np.ones(rows, dtype=np.float32),
                np.arange(rows, dtype=np.uint64),
            )

        mixed = training.MixedTraining(empty_dataset(70), empty_dataset(500), 64, 192)
        first = training.mixed_epoch_batches(
            mixed, batch_size=256, seed=123, epoch=1
        )
        second = training.mixed_epoch_batches(
            mixed, batch_size=256, seed=123, epoch=2
        )
        self.assertEqual(len(first), 2)
        self.assertTrue(all(
            len(new) == 64 and len(anchor) == 192
            for new, anchor in (*first, *second)
        ))
        self.assertEqual(set(np.concatenate([new for new, _ in first])), set(range(70)))
        self.assertEqual(set(np.concatenate([new for new, _ in second])), set(range(70)))
        anchor_rows = np.concatenate([
            rows for _, rows in (*first, *second)
        ])
        self.assertEqual(len(set(anchor_rows[:500])), 500)

    def test_mixed_seed_workers_train_through_full_anchor_coverage(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(4)),
            datasets["train"].targets[:4],
            datasets["train"].weights[:4],
            tuple(f"new:{index}" for index in range(4)),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(4, 8)),
            datasets["train"].targets[4:8],
            datasets["train"].weights[4:8],
            tuple(f"anchor:{index}" for index in range(4)),
        )
        with tempfile.TemporaryDirectory() as directory:
            initial_runtime = pathlib.Path(directory) / "actor.runtime"
            actor = training.initialize(777)
            training.export_runtime(initial_runtime, actor)
            selected, report = training.train_three_seeds(
                {},
                seeds=(41, 42, 43),
                epochs=2,
                patience=1,
                batch_size=4,
                mixed_training=training.MixedTraining(new, anchor, 3, 1),
                selection_validation=datasets["validation"],
                retention_validation=datasets["test"],
                initial_runtime=initial_runtime,
            )
            parallel, parallel_report = training.train_three_seeds(
                {},
                seeds=(41, 42, 43),
                epochs=2,
                patience=1,
                batch_size=4,
                mixed_training=training.MixedTraining(new, anchor, 3, 1),
                selection_validation=datasets["validation"],
                retention_validation=datasets["test"],
                initial_runtime=initial_runtime,
                seed_workers=2,
            )
        self.assertEqual(
            training.runtime_bytes(selected)[0], training.runtime_bytes(parallel)[0]
        )
        self.assertEqual(report, parallel_report)
        self.assertEqual(report["initial_runtime"]["artifact_sha256"],
                         training.runtime_bytes(actor)[1]["artifact_sha256"])
        for seed_report in report["seed_reports"]:
            self.assertEqual(seed_report["anchor_coverage_complete_epoch"], 2)
            self.assertGreaterEqual(len(seed_report["history"]), 2)
            self.assertGreaterEqual(
                seed_report["history"][-1]["coverage"]["anchor"][
                    "complete_permutations"
                ],
                1,
            )
            self.assertEqual(
                seed_report["initial"]["runtime"], report["initial_runtime"]
            )

    def test_equal_epoch_zero_key_skips_retention_and_falls_back(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(2)),
            datasets["train"].targets[:2],
            datasets["train"].weights[:2],
            ("new:0", "new:1"),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(2, 4)),
            datasets["train"].targets[2:4],
            datasets["train"].weights[2:4],
            ("anchor:0", "anchor:1"),
        )
        mixed = training.MixedTraining(new, anchor, 2, 2)
        retention = datasets["test"]
        initial = training.initialize(456)
        runtime_report = training.runtime_bytes(initial)[1]
        original = training.train_mixed_batch
        original_metrics = training.metrics
        shared_initial_metrics = {
            "validation": original_metrics(initial, datasets["validation"]),
            "retention": original_metrics(initial, retention),
        }
        retention_metric_calls = []

        def no_update(*_args, **_kwargs):
            return 0.0

        def recording_metrics(parameters, dataset, batch_size=4096):
            if dataset is retention:
                retention_metric_calls.append(1)
            return original_metrics(parameters, dataset, batch_size)

        training.train_mixed_batch = no_update
        training.metrics = recording_metrics
        try:
            selected, report = training.train_seed(
                {
                    "train": training.concatenate_datasets((new, anchor)),
                    "validation": datasets["validation"],
                },
                999,
                epochs=1,
                patience=1,
                batch_size=4,
                learning_rate=0.001,
                weight_decay=0.0,
                mixed_training=mixed,
                initial_parameters=initial,
                initial_runtime_report=runtime_report,
                initial_metrics=shared_initial_metrics,
                retention_validation=retention,
            )
        finally:
            training.train_mixed_batch = original
            training.metrics = original_metrics
        self.assertFalse(report["history"][0]["eligible"])
        self.assertIsNone(report["history"][0]["retention"])
        self.assertEqual(
            report["history"][0]["retention_status"],
            "not-evaluated-adjudicator-cannot-beat-current-best",
        )
        self.assertEqual(report["best_epoch"], 0)
        self.assertEqual(report["selection"], "exact-initial-runtime-fallback")
        self.assertEqual(training.runtime_bytes(selected)[0],
                         training.runtime_bytes(initial)[0])
        self.assertEqual(retention_metric_calls, [])

    def test_precoverage_epoch_is_preserved_while_training_reaches_coverage(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(4)),
            datasets["train"].targets[:4],
            datasets["train"].weights[:4],
            tuple(f"new:{index}" for index in range(4)),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(4, 8)),
            datasets["train"].targets[4:8],
            datasets["train"].weights[4:8],
            tuple(f"anchor:{index}" for index in range(4)),
        )
        retention = datasets["test"]
        mixed = training.MixedTraining(new, anchor, 3, 1)
        actor = {
            "w1": np.zeros((6301, 192), dtype=np.float32),
            "w2": np.zeros((192, 32), dtype=np.float32),
            "w3": np.zeros(32, dtype=np.float32),
        }
        runtime_report = training.runtime_bytes(actor)[1]
        original_batch = training.train_mixed_batch
        original_metrics = training.metrics
        batch_calls = []

        def epoch_marker(parameters, *_args, **_kwargs):
            batch_calls.append(1)
            epoch = (len(batch_calls) + 1) // 2
            parameters["w3"][0] = np.float32(0.1 * epoch)
            return 0.1

        def controlled_metrics(parameters, dataset, batch_size=4096):
            del batch_size
            marker = float(parameters["w3"][0])
            if dataset is datasets["validation"]:
                huber = 0.3 if marker == 0.0 else (0.1 if marker < 0.15 else 0.2)
            else:
                huber = 0.1
            return {
                "samples": len(dataset),
                "weighted_huber": huber,
                "sign_accuracy": 0.9,
                "correlation": 0.7,
                "mae": huber,
                "prediction_mean": 0.0,
            }

        training.train_mixed_batch = epoch_marker
        training.metrics = controlled_metrics
        try:
            selected, report = training.train_seed(
                {
                    "train": training.concatenate_datasets((new, anchor)),
                    "validation": datasets["validation"],
                },
                88,
                epochs=2,
                patience=1,
                batch_size=4,
                learning_rate=0.001,
                weight_decay=0.0,
                mixed_training=mixed,
                retention_validation=retention,
                initial_parameters=actor,
                initial_runtime_report=runtime_report,
            )
        finally:
            training.train_mixed_batch = original_batch
            training.metrics = original_metrics
        self.assertAlmostEqual(float(selected["w3"][0]), 0.1, places=7)
        self.assertEqual(report["best_epoch"], 1)
        self.assertTrue(report["history"][0]["eligible"])
        self.assertEqual(
            report["history"][0]["coverage"]["anchor"]["complete_permutations"],
            0,
        )
        self.assertEqual(len(batch_calls), 4)
        self.assertGreaterEqual(
            report["history"][-1]["coverage"]["anchor"]["complete_permutations"],
            1,
        )
        self.assertIsNone(report["history"][1]["retention"])
        self.assertEqual(
            report["history"][1]["retention_status"],
            "not-evaluated-adjudicator-cannot-beat-current-best",
        )

    def test_target_metadata_is_derived_from_shard_provenance(self):
        search_policy = corpus.target_policy_for_schema(
            corpus.SEARCH_TEACHER_SCHEMA
        )
        rank4_policy = corpus.target_policy_for_schema(corpus.TEACHER_SCHEMA)
        rank4_v2_policy = corpus.target_policy_for_schema(
            corpus.RANK4_TEACHER_SCHEMA
        )
        manifests = [
            {"provenance": {"target_policy": search_policy}},
            {"provenance": {"target_policy": rank4_policy}},
            {"provenance": {"target_policy": rank4_v2_policy}},
            {"provenance": {}},
        ]
        metadata = training.target_metadata_from_shard_provenance(
            manifests,
            ("new", "anchor", "rank4-new", "selection-validation"),
        )
        self.assertEqual(len(metadata["declared_policies"]), 3)
        self.assertEqual(metadata["undeclared_roles"], ["selection-validation"])
        policies = {
            row["policy"]["teacher_schema"]: row["roles"]
            for row in metadata["declared_policies"]
        }
        self.assertEqual(policies[corpus.SEARCH_TEACHER_SCHEMA], ["new"])
        self.assertEqual(policies[corpus.TEACHER_SCHEMA], ["anchor"])
        self.assertEqual(
            policies[corpus.RANK4_TEACHER_SCHEMA], ["rank4-new"]
        )

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

    def test_retention_validation_rejects_group_and_symmetric_feature_overlap(self):
        samples = training.tiny_fixture_samples()
        retention_sample = samples["validation"][0]
        retention = training.Dataset.from_active(
            (retention_sample.active,),
            np.asarray([retention_sample.target], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            ("retention-root",),
        )
        reflected = training.Dataset.from_active(
            (training.features.reflect_active(retention_sample.active),),
            np.asarray([retention_sample.target], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            ("different-root",),
        )
        with self.assertRaisesRegex(ValueError, "feature overlap"):
            training.validate_retention_validation_independence(
                retention, {"new-adjudicator": reflected}
            )
        distinct_sample = samples["test"][0]
        repeated_group = training.Dataset.from_active(
            (distinct_sample.active,),
            np.asarray([distinct_sample.target], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            ("retention-root",),
        )
        with self.assertRaisesRegex(ValueError, "root group overlaps"):
            training.validate_retention_validation_independence(
                retention, {"anchor-training": repeated_group}
            )
        distinct = training.Dataset.from_active(
            (distinct_sample.active,),
            np.asarray([distinct_sample.target], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            ("independent-root",),
        )
        training.validate_retention_validation_independence(
            retention, {"new-adjudicator": distinct}
        )
        training.validate_retention_validation_independence(
            retention,
            {
                "anchor-training": repeated_group,
                "new-adjudicator": distinct,
            },
            prevalidated_cross_split_roles=("anchor-training",),
        )
        with self.assertRaisesRegex(ValueError, "not comparison inputs"):
            training.validate_retention_validation_independence(
                retention,
                {"new-adjudicator": distinct},
                prevalidated_cross_split_roles=("anchor-training",),
            )

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
                report["target_policies"],
                [corpus.target_policy_for_schema(corpus.TEACHER_SCHEMA)],
            )
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
            legacy_manifest = json.loads(
                (parallel / "jacek_replay_bfm.runtime.json").read_bytes()
            )
            self.assertEqual(legacy_manifest["schema"], training.MODEL_MANIFEST_SCHEMA)
            self.assertNotIn("initialization", legacy_manifest["training"])
            for seed in training.FIXED_SEEDS:
                checkpoint_receipt = json.loads(
                    (parallel_checkpoints / f"seed-{seed}.json").read_bytes()
                )
                self.assertEqual(
                    checkpoint_receipt["schema"], training.SEED_CHECKPOINT_SCHEMA
                )
                self.assertNotIn("initial_runtime", checkpoint_receipt["inputs"])
                self.assertNotIn(
                    "retention_selection", checkpoint_receipt["configuration"]
                )
                self.assertNotIn("update", checkpoint_receipt["configuration"])
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

    def test_mixed_resume_binds_initial_runtime_monitor_and_loss_policy(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(4)),
            datasets["train"].targets[:4],
            datasets["train"].weights[:4],
            tuple(f"new:{index}" for index in range(4)),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(4, 8)),
            datasets["train"].targets[4:8],
            datasets["train"].weights[4:8],
            tuple(f"anchor:{index}" for index in range(4)),
        )
        mixed = training.MixedTraining(new, anchor, 3, 1)
        retention = datasets["test"]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            initial_runtime = root / "actor.runtime"
            other_runtime = root / "other.runtime"
            training.export_runtime(initial_runtime, training.initialize(901))
            training.export_runtime(other_runtime, training.initialize(902))
            checkpoints = root / "seeds"
            arguments = {
                "seeds": (51, 52, 53),
                "epochs": 2,
                "patience": 1,
                "batch_size": 4,
                "mixed_training": mixed,
                "selection_validation": datasets["validation"],
                "retention_validation": retention,
                "initial_runtime": initial_runtime,
                "seed_checkpoint_directory": checkpoints,
                "input_shard_identities": ({"fixture": "mixed-resume-v2"},),
            }
            first, first_report = training.train_three_seeds({}, **arguments)
            before = {
                path.name: path.read_bytes() for path in checkpoints.iterdir()
            }
            resumed, resumed_report = training.train_three_seeds(
                {}, resume_seeds=True, **arguments
            )
            self.assertEqual(training.runtime_bytes(first)[0],
                             training.runtime_bytes(resumed)[0])
            self.assertEqual(first_report, resumed_report)
            self.assertEqual(
                before,
                {path.name: path.read_bytes() for path in checkpoints.iterdir()},
            )
            receipt = json.loads((checkpoints / "seed-51.json").read_bytes())
            self.assertEqual(
                receipt["schema"], training.RETENTION_SEED_CHECKPOINT_SCHEMA
            )
            self.assertEqual(
                receipt["inputs"]["initial_runtime"]["artifact_sha256"],
                training.load_runtime(initial_runtime)[1]["artifact_sha256"],
            )
            self.assertEqual(
                receipt["inputs"]["retention_monitor"]["samples"], len(retention)
            )
            self.assertEqual(
                receipt["inputs"]["anchor_training_monitor"]["samples"],
                len(anchor),
            )
            self.assertEqual(
                receipt["configuration"]["loss"],
                {
                    "name": "separately-normalized-weighted-huber",
                    "delta": 0.25,
                    "new_coefficient": 0.5,
                    "anchor_coefficient": 0.5,
                },
            )
            wrong_schema = dict(receipt)
            wrong_schema["schema"] = training.SEED_CHECKPOINT_SCHEMA
            (checkpoints / "seed-51.json").write_bytes(
                training.canonical_json_bytes(wrong_schema)
            )
            with self.assertRaisesRegex(ValueError, "receipt schema is invalid"):
                training.train_three_seeds({}, resume_seeds=True, **arguments)
            (checkpoints / "seed-51.json").write_bytes(before["seed-51.json"])
            tampered_metrics = json.loads(before["seed-51.json"])
            tampered_metrics["training_report"]["initial"]["validation"][
                "weighted_huber"
            ] = 0.0
            tampered_metrics["training_report"]["validation"][
                "weighted_huber"
            ] = 0.0
            body = dict(tampered_metrics)
            body.pop("body_sha256")
            tampered_metrics["body_sha256"] = hashlib.sha256(
                training.canonical_json_bytes(body)
            ).hexdigest()
            (checkpoints / "seed-51.json").write_bytes(
                training.canonical_json_bytes(tampered_metrics)
            )
            with self.assertRaisesRegex(ValueError, "retention report is stale"):
                training.train_three_seeds({}, resume_seeds=True, **arguments)
            (checkpoints / "seed-51.json").write_bytes(before["seed-51.json"])
            tampered_final = json.loads(before["seed-51.json"])
            tampered_final["training_report"]["training"][
                "weighted_huber"
            ] = 0.0
            body = dict(tampered_final)
            body.pop("body_sha256")
            tampered_final["body_sha256"] = hashlib.sha256(
                training.canonical_json_bytes(body)
            ).hexdigest()
            (checkpoints / "seed-51.json").write_bytes(
                training.canonical_json_bytes(tampered_final)
            )
            with self.assertRaisesRegex(ValueError, "recomputed metrics disagree"):
                training.train_three_seeds({}, resume_seeds=True, **arguments)
            (checkpoints / "seed-51.json").write_bytes(before["seed-51.json"])
            with self.assertRaisesRegex(ValueError, "inputs is stale"):
                training.train_three_seeds(
                    {},
                    resume_seeds=True,
                    **{**arguments, "initial_runtime": other_runtime},
                )
            with self.assertRaisesRegex(ValueError, "configuration is stale"):
                training.train_three_seeds(
                    {},
                    resume_seeds=True,
                    **{
                        **arguments,
                        "new_loss_coefficient": 0.6,
                        "anchor_loss_coefficient": 0.4,
                    },
                )

    def test_mixed_cli_emits_only_v2_model_and_checkpoint_schemas(self):
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
            actor = root / "actor.runtime"
            training.export_runtime(actor, training.initialize(7001))
            output = root / "model"
            checkpoints = root / "seed-checkpoints"
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
                    "--seeds", "71,72,73",
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
            self.assertEqual(
                manifest["schema"], training.RETENTION_MODEL_MANIFEST_SCHEMA
            )
            self.assertEqual(
                manifest["training"]["initialization"]["kind"],
                "exact-supplied-runtime-all-seeds-order-only-v1",
            )
            self.assertEqual(
                manifest["training"]["retention_selection"]["dataset"]["samples"],
                len(samples["test"]),
            )
            self.assertTrue(
                manifest["target"]["loss"].startswith(
                    "separately-normalized-weighted-huber-delta-0.25"
                )
            )
            for seed in (71, 72, 73):
                receipt = json.loads(
                    (checkpoints / f"seed-{seed}.json").read_bytes()
                )
                self.assertEqual(
                    receipt["schema"], training.RETENTION_SEED_CHECKPOINT_SCHEMA
                )
                self.assertIn("retention_selection", receipt["configuration"])
                self.assertEqual(
                    receipt["inputs"]["retention_monitor"]["samples"],
                    len(samples["test"]),
                )

    def test_resume_recomputes_feasible_epoch_zero_selection_minimum(self):
        datasets = self._tiny_datasets()
        new = training.Dataset.from_active(
            datasets["train"].active_rows(range(2)),
            datasets["train"].targets[:2],
            datasets["train"].weights[:2],
            ("new:0", "new:1"),
        )
        anchor = training.Dataset.from_active(
            datasets["train"].active_rows(range(2, 4)),
            datasets["train"].targets[2:4],
            datasets["train"].weights[2:4],
            ("anchor:0", "anchor:1"),
        )
        mixed = training.MixedTraining(new, anchor, 2, 2)
        retention = datasets["test"]
        zero = {
            "w1": np.zeros((6301, 192), dtype=np.float32),
            "w2": np.zeros((192, 32), dtype=np.float32),
            "w3": np.zeros(32, dtype=np.float32),
        }
        original_batch = training.train_mixed_batch
        original_metrics = training.metrics

        def improving_batch(parameters, *_args, **_kwargs):
            parameters["w3"][0] = np.float32(0.1)
            return 0.1

        def controlled_metrics(parameters, dataset, batch_size=4096):
            del batch_size
            trained = float(parameters["w3"][0]) > 0.0
            if dataset is datasets["validation"]:
                huber, sign, correlation = (
                    (0.1, 0.9, 0.7) if trained else (0.2, 0.8, 0.5)
                )
            elif dataset is retention:
                huber, sign, correlation = (
                    (0.1005, 0.899, 0.7) if trained else (0.1, 0.9, 0.7)
                )
            else:
                huber, sign, correlation = (0.1, 0.9, 0.7)
            return {
                "samples": len(dataset),
                "weighted_huber": huber,
                "sign_accuracy": sign,
                "correlation": correlation,
                "mae": huber,
                "prediction_mean": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            initial_runtime = root / "actor.runtime"
            checkpoints = root / "seeds"
            training.export_runtime(initial_runtime, zero)
            arguments = {
                "seeds": (61, 62, 63),
                "epochs": 1,
                "patience": 1,
                "batch_size": 4,
                "mixed_training": mixed,
                "selection_validation": datasets["validation"],
                "retention_validation": retention,
                "initial_runtime": initial_runtime,
                "seed_checkpoint_directory": checkpoints,
            }
            training.train_mixed_batch = improving_batch
            training.metrics = controlled_metrics
            try:
                selected, report = training.train_three_seeds({}, **arguments)
            finally:
                training.train_mixed_batch = original_batch
                training.metrics = original_metrics
            self.assertAlmostEqual(float(selected["w3"][0]), 0.1, places=7)
            self.assertTrue(all(
                item["best_epoch"] == 1
                and item["selection"] == "feasible-trained-epoch"
                for item in report["seed_reports"]
            ))
            training.metrics = controlled_metrics
            try:
                resumed, resumed_report = training.train_three_seeds(
                    {}, resume_seeds=True, **arguments
                )
            finally:
                training.metrics = original_metrics
            self.assertEqual(
                training.runtime_bytes(selected)[0],
                training.runtime_bytes(resumed)[0],
            )
            self.assertEqual(report, resumed_report)
            receipt_path = checkpoints / "seed-61.json"
            original_receipt = receipt_path.read_bytes()
            receipt = json.loads(original_receipt)
            receipt["training_report"]["best_epoch"] = 0
            body = dict(receipt)
            body.pop("body_sha256")
            receipt["body_sha256"] = hashlib.sha256(
                training.canonical_json_bytes(body)
            ).hexdigest()
            receipt_path.write_bytes(training.canonical_json_bytes(receipt))
            training.metrics = controlled_metrics
            try:
                with self.assertRaisesRegex(ValueError, "selected epoch is stale"):
                    training.train_three_seeds({}, resume_seeds=True, **arguments)
            finally:
                training.metrics = original_metrics

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

            wrong_schema = json.loads(receipt_bytes)
            wrong_schema["schema"] = training.RETENTION_SEED_CHECKPOINT_SCHEMA
            receipt_path.write_bytes(training.canonical_json_bytes(wrong_schema))
            with self.assertRaisesRegex(ValueError, "receipt schema is invalid"):
                training.train_three_seeds(datasets, resume_seeds=True, **arguments)
            receipt_path.write_bytes(receipt_bytes)

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
