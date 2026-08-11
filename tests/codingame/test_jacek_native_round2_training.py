import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import jacek_native_corpus as round1
import jacek_native_corpus_round2 as corpus
import jacek_native_workflow_round2 as workflow
from tests.codingame import test_jacek_native_training as fixtures


ROUND2_EXPORTER_PATH = (
    ROOT / "submissions" / "codingame" / "tools" /
    "generate_jacek_native_model_round2.py"
)


def round2_record(game, winner):
    record = fixtures.game_record(game, winner)
    record.update({
        "schema": corpus.GAME_SCHEMA,
        "seed": str(50_000 + game),
        "game": game,
        "shard_index": 0,
        "shard_count": 1,
        "split_group": f"native-round2:{50_000 + game}:{game}",
    })
    generator = record["generator"]
    generator.update({
        "schema": corpus.GENERATOR_SCHEMA,
        "checkpoint_color_schedule": corpus.COLOR_SCHEDULE,
        "opening_pair_index": game // 2,
        "opening_seed": str(90_000 + game // 2),
        "reanalysis": {
            "selection": corpus.REANALYSIS_SELECTION,
            "samples_per_game": 0,
            "work": 0,
            "verification_work": 0,
            "teacher": None,
        },
    })
    first = {
        "model_sha256": "1" * 64,
        "packed_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
    }
    second = {
        "model_sha256": "4" * 64,
        "packed_sha256": "5" * 64,
        "artifact_sha256": "6" * 64,
    }
    generator["models"] = (
        {"player_one": first, "player_two": second}
        if game % 2 == 0
        else {"player_one": second, "player_two": first}
    )
    return record


def compiler_identity():
    compiler_path = pathlib.Path(
        shutil.which("c++") or shutil.which("clang++")
    ).resolve()
    version = subprocess.run(
        [str(compiler_path), "--version"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return compiler_path, {
        "executable": compiler_path.name,
        "sha256": hashlib.sha256(compiler_path.read_bytes()).hexdigest(),
        "version": version,
        "version_sha256": hashlib.sha256(version.encode()).hexdigest(),
    }


def write_round2_run(directory, records, seed=2026082101):
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / corpus.ARCHIVED_BINARY_NAME
    binary.write_bytes(b"round-two archived self-play fixture\n")
    _, compiler = compiler_identity()
    sources = [{
        "path": path,
        "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
    } for path in corpus.BUILD_SOURCE_PATHS]
    producer = hashlib.sha256(json.dumps(
        [[entry["path"], entry["sha256"]] for entry in sources],
        separators=(",", ":"),
    ).encode()).hexdigest()
    build = {
        "schema": corpus.BUILD_PROVENANCE_SCHEMA,
        "binary": {
            "path": corpus.ARCHIVED_BINARY_NAME,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "compiler": compiler,
        "build_argv": list(corpus.CANONICAL_BUILD_ARGV),
        "producer_sha256": producer,
        "sources": sources,
    }
    build_raw = corpus._canonical_json_bytes(build)
    (directory / corpus.BUILD_PROVENANCE_NAME).write_bytes(build_raw)
    build_digest = hashlib.sha256(build_raw).hexdigest()
    checkpoint_root = directory / "checkpoints"
    checkpoint_root.mkdir()
    runtime_source = ROOT / "models" / "jacek_native_untrained_seed.runtime"
    checkpoint_identity = workflow.runtime_identity(runtime_source)
    checkpoint_manifest = {}
    for role in ("player_one", "player_two"):
        runtime = checkpoint_root / f"{role}.runtime"
        shutil.copyfile(runtime_source, runtime)
        checkpoint_manifest[role] = {
            "name": "untrained",
            "runtime": f"checkpoints/{role}.runtime",
            **checkpoint_identity,
        }
    for record in records:
        record["generator"]["producer_sha256"] = producer
        record["generator"]["build_provenance_sha256"] = build_digest
        record["generator"]["models"] = {
            "player_one": checkpoint_identity,
            "player_two": checkpoint_identity,
        }
    shard = directory / "shard-00-of-01.jsonl"
    shard.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = {
        "schema": corpus.RUN_SCHEMA,
        "run_id": "fixture",
        "producer_sha256": producer,
        "build_provenance": {
            "path": corpus.BUILD_PROVENANCE_NAME,
            "sha256": build_digest,
        },
        "binary": build["binary"],
        "checkpoints": checkpoint_manifest,
        "config": {
            "games": len(records),
            "seed": seed,
            "work": 32,
            "samples_per_game": 1,
            "reanalysis_samples_per_game": 0,
            "shards": 1,
            "parallel": 1,
            "temperature": 3.0,
            "temperature_turns": 12,
            "temperature_schedule":
                "absolute-complete-turn-index-before-cutoff/v1",
            "opening_depths": [0],
            "opening_schema":
                "deterministic-procedural-complete-turn-prefix/v1",
            "checkpoint_color_schedule": corpus.COLOR_SCHEDULE,
            "reanalysis_selection": corpus.REANALYSIS_SELECTION,
            "reanalysis_work": 0,
            "verification_work": 0,
            "max_complete_turns": 384,
        },
        "shard_outputs": [{
            "shard": 0,
            "path": shard.name,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "bytes": shard.stat().st_size,
            "stderr_sha256": "0" * 64,
        }],
    }
    (directory / corpus.MANIFEST_NAME).write_bytes(
        corpus._canonical_json_bytes(manifest)
    )
    return shard


def write_round1_run(directory, records, seed=73194721):
    directory.mkdir(parents=True, exist_ok=True)
    build, build_digest = fixtures.write_build_provenance(directory)
    for record in records:
        record["shard_index"] = 0
        record["shard_count"] = 1
        record["generator"]["producer_sha256"] = build["producer_sha256"]
        record["generator"]["build_provenance_sha256"] = build_digest
    shard = directory / "shard-00-of-01.jsonl"
    shard.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = {
        "schema": "papersoccer.jace-native-selfplay-run/invalid",
        "producer_sha256": build["producer_sha256"],
        "build_provenance": {
            "path": round1.BUILD_PROVENANCE_NAME,
            "sha256": build_digest,
        },
        "binary": build["binary"],
        "config": {"games": len(records), "seed": seed},
        "shard_outputs": [{
            "shard": 0,
            "path": shard.name,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "bytes": shard.stat().st_size,
        }],
    }
    manifest["schema"] = "papersoccer.jacek-native-selfplay-run/v1"
    (directory / corpus.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return shard


class JacekNativeRound2CorpusTest(unittest.TestCase):
    def test_mixed_outcome_restart_group_remains_atomic(self):
        games = [
            types.SimpleNamespace(split_group="restart-live-game", winner=0),
            types.SimpleNamespace(split_group="restart-live-game", winner=1),
            types.SimpleNamespace(split_group="mixed-two", winner=0),
            types.SimpleNamespace(split_group="mixed-two", winner=1),
            types.SimpleNamespace(split_group="mixed-three", winner=0),
            types.SimpleNamespace(split_group="mixed-three", winner=1),
        ]
        assignment = corpus.assign_splits(games)
        self.assertIn(assignment["restart-live-game"], {
            "train", "validation", "test"
        })
        self.assertEqual(len(assignment), 3)

    def test_planned_work_exhaustion_can_be_stable(self):
        record = round2_record(0, 0)
        teacher = {
            "model_sha256": "7" * 64,
            "packed_sha256": "8" * 64,
            "artifact_sha256": "9" * 64,
        }
        record["generator"]["reanalysis"] = {
            "selection": corpus.REANALYSIS_SELECTION,
            "samples_per_game": 1,
            "work": corpus.TEACHER_WORK,
            "verification_work": corpus.VERIFICATION_WORK,
            "teacher": teacher,
        }
        record["samples"][0]["reanalysis"] = {
            "selection_reason": "hard",
            "value": 0.25,
            "work": corpus.TEACHER_WORK,
            "verification_work": corpus.VERIFICATION_WORK,
            "operational_interruption": False,
            "primary_planned_work_exhaustion": True,
            "verification_planned_work_exhaustion": True,
            "primary_generator_sampling_truncations": 3,
            "verification_generator_sampling_truncations": 7,
            "primary_proof_sampling_truncations": 5,
            "verification_proof_sampling_truncations": 11,
            "action_stable": True,
            "value_delta": 0.05,
            "stable": True,
            "exact": False,
        }
        game = corpus.validate_record(record)
        self.assertEqual(game.samples[0].auxiliary_value, 0.25)
        self.assertIn(
            corpus.NativeModelArtifact(
                artifact_sha256=teacher["artifact_sha256"],
                model_sha256=teacher["model_sha256"],
                packed_sha256=teacher["packed_sha256"],
            ),
            game.model_artifacts,
        )

        corrupted = copy.deepcopy(record)
        corrupted["samples"][0]["reanalysis"].update({
            "operational_interruption": True,
            "stable": True,
        })
        with self.assertRaisesRegex(ValueError, "stability classification"):
            corpus.validate_record(corrupted)

    def test_exact_outcome_overrides_operational_interruption(self):
        record = round2_record(0, 0)
        record["generator"]["reanalysis"] = {
            "selection": corpus.REANALYSIS_SELECTION,
            "samples_per_game": 1,
            "work": corpus.TEACHER_WORK,
            "verification_work": corpus.VERIFICATION_WORK,
            "teacher": {
                "model_sha256": "7" * 64,
                "packed_sha256": "8" * 64,
                "artifact_sha256": "9" * 64,
            },
        }
        record["samples"][0]["reanalysis"] = {
            "selection_reason": "uncertain",
            "value": 1.0,
            "work": corpus.TEACHER_WORK,
            "verification_work": corpus.VERIFICATION_WORK,
            "operational_interruption": True,
            "primary_planned_work_exhaustion": True,
            "verification_planned_work_exhaustion": False,
            "primary_generator_sampling_truncations": 1,
            "verification_generator_sampling_truncations": 2,
            "primary_proof_sampling_truncations": 3,
            "verification_proof_sampling_truncations": 4,
            "action_stable": False,
            "value_delta": 2.0,
            "stable": True,
            "exact": True,
        }
        game = corpus.validate_record(record)
        self.assertTrue(game.samples[0].exact)
        self.assertEqual(game.samples[0].auxiliary_value, 1.0)

    def test_strict_manifest_checks_shard_binary_and_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            shard = write_round2_run(
                directory, [round2_record(0, 0), round2_record(1, 1)]
            )
            games, _, lineage = corpus.load_games([shard])
            self.assertEqual(len(games), 2)
            self.assertEqual(len(lineage["strict_current"]), 1)
            shard.write_text(shard.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "shard identity is stale"):
                corpus.load_games([shard])

        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            shard = write_round2_run(
                directory, [round2_record(0, 0), round2_record(1, 1)]
            )
            (directory / corpus.ARCHIVED_BINARY_NAME).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "archived binary is stale"):
                corpus.load_games([shard])

        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            shard = write_round2_run(
                directory, [round2_record(0, 0), round2_record(1, 1)]
            )
            (directory / "checkpoints/player_one.runtime").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint is stale"):
                corpus.load_games([shard])

    def test_cumulative_round1_lineage_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current = write_round2_run(
                root / "current",
                [round2_record(0, 0), round2_record(1, 1)],
            )
            archived = write_round1_run(
                root / "archived",
                [fixtures.game_record(20, 0), fixtures.game_record(21, 1)],
            )
            games, _, lineage = corpus.load_games([current], [archived])
            self.assertEqual(len(games), 4)
            self.assertEqual(len(lineage["strict_current"]), 1)
            self.assertEqual(len(lineage["archived_round1"]), 1)
            (archived.parent / round1.ARCHIVED_BINARY_NAME).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "round-one binary is stale"):
                corpus.load_games([current], [archived])

    def test_round2_run_seeds_must_be_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = write_round2_run(
                root / "first",
                [round2_record(0, 0), round2_record(1, 1)], seed=77,
            )
            second_records = [round2_record(0, 0), round2_record(1, 1)]
            for index, record in enumerate(second_records):
                record["seed"] = str(60_000 + index)
                record["split_group"] = f"native-round2:{60_000 + index}:{index}"
            second = write_round2_run(root / "second", second_records, seed=77)
            with self.assertRaisesRegex(ValueError, "seeds must be unique"):
                corpus.load_games([first, second])


class JacekNativeRound2WorkflowTest(unittest.TestCase):
    def test_source_contract_is_additive_and_round1_is_unchanged(self):
        self.assertEqual(
            corpus.BUILD_SOURCE_PATHS[0],
            "tools/jacek_native_selfplay_round2.cpp",
        )
        self.assertEqual(
            corpus.BUILD_SOURCE_PATHS[1:], round1.BUILD_SOURCE_PATHS
        )
        self.assertEqual(
            workflow.producer_sha256(),
            hashlib.sha256(json.dumps(
                [[entry["path"], entry["sha256"]]
                 for entry in workflow.source_contract()],
                separators=(",", ":"),
            ).encode()).hexdigest(),
        )

    def test_four_member_league_plan_has_unique_deterministic_seeds(self):
        members = {
            name: pathlib.Path(f"{name}.runtime")
            for name in ("current", "seed12", "seed11", "untrained")
        }
        first = workflow.league_pairings(members, "current", 123)
        second = workflow.league_pairings(dict(reversed(list(members.items()))),
                                          "current", 123)
        self.assertEqual(first, second)
        self.assertEqual([item[0] for item in first], [
            "current", "seed11", "seed12", "untrained"
        ])
        self.assertEqual(len({item[2] for item in first}), 4)

    def test_runtime_identity_is_file_backed(self):
        runtime = ROOT / "models" / "jacek_native_untrained_seed.runtime"
        identity = workflow.runtime_identity(runtime)
        self.assertEqual(identity["artifact_sha256"], hashlib.sha256(
            runtime.read_bytes()
        ).hexdigest())
        self.assertTrue(all(len(value) == 64 for value in identity.values()))

    def test_wall_clock_measurements_do_not_change_manifest_identity(self):
        base = {
            "shard": 0,
            "path": "shard-00-of-01.jsonl",
            "sha256": "a" * 64,
            "bytes": 17,
            "stderr_sha256": "b" * 64,
        }
        first = workflow.stable_shard_reports([
            {**base, "elapsed_seconds": 1.25}
        ])
        second = workflow.stable_shard_reports([
            {**base, "elapsed_seconds": 99.5}
        ])
        self.assertEqual(first, second)
        self.assertNotIn("elapsed_seconds", first[0])


@unittest.skipIf(np is None, "round-two trainer tests require NumPy")
class JacekNativeRound2TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "jacek_native_round2_trainer_under_test",
            TOOLS / "train_jacek_native_round2.py",
        )
        cls.trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.trainer
        spec.loader.exec_module(cls.trainer)
        exporter_spec = importlib.util.spec_from_file_location(
            "jacek_native_round2_exporter_under_test", ROUND2_EXPORTER_PATH,
        )
        cls.exporter = importlib.util.module_from_spec(exporter_spec)
        sys.modules[exporter_spec.name] = cls.exporter
        exporter_spec.loader.exec_module(cls.exporter)

    def dataset(self):
        active = tuple(
            np.asarray([index], dtype=np.int32) for index in (0, 1, 2, 3)
        )
        return self.trainer.Dataset(
            active=active,
            outcome=np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            auxiliary=np.zeros(4, dtype=np.float32),
            auxiliary_mask=np.zeros(4, dtype=bool),
            exact_mask=np.zeros(4, dtype=bool),
            game_keys=("a", "b", "c", "d"),
            turn=np.asarray([0, 15, 25, 45], dtype=np.int32),
        )

    def test_turn_binned_calibration_and_w1_coverage_are_reported(self):
        parameters = self.trainer.initialize(7)
        metrics = self.trainer.metrics(parameters, self.dataset())
        calibration = metrics["turn_calibration"]
        self.assertEqual(
            [calibration[name]["samples"] for name, _, _
             in self.trainer.TURN_BINS],
            [1, 1, 1, 1],
        )
        for value in calibration.values():
            self.assertIsNotNone(value["prediction_std"])
            self.assertIsNotNone(value["prediction_min"])
            self.assertIsNotNone(value["prediction_max"])
            self.assertEqual(
                set(value["prediction_quantiles"]),
                {"p05", "p25", "p50", "p75", "p95"},
            )
        coverage = self.trainer.quantized_w1_coverage(parameters)
        self.assertEqual(coverage["all"]["rows"], corpus.INPUT_COUNT)
        self.assertEqual(sum(coverage["levels"].values()),
                         corpus.INPUT_COUNT * self.trainer.HIDDEN_ONE)

    def test_robust_fixed_scale_rejects_a_single_max_outlier_and_roundtrips(self):
        parameters = self.trainer.initialize(19)
        pattern = np.where(
            np.arange(parameters["w1"].size).reshape(parameters["w1"].shape)
            % 2,
            np.float32(-0.02),
            np.float32(0.02),
        )
        parameters["w1"] = pattern.astype(np.float32)
        parameters["w1"][0, 0] = np.float32(100.0)
        dynamic_scale = self.trainer.quantize(parameters)[1]["w1"]
        candidates = self.trainer._robust_scale_candidates(parameters["w1"])
        self.assertLess(candidates[-1], dynamic_scale / 1_000.0)

        first_scales, first, first_report = self.trainer.select_fixed_scales(
            parameters, self.dataset()
        )
        second_scales, second, second_report = self.trainer.select_fixed_scales(
            parameters, self.dataset()
        )
        self.assertEqual(first_report, second_report)
        self.assertEqual(
            self.trainer._scale_payload(first_scales),
            self.trainer._scale_payload(second_scales),
        )
        for name in ("w1", "w2", "w3"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertTrue(first_report["max_abs_is_not_a_scale_candidate"])
        self.assertEqual(
            self.trainer.quantized_w1_coverage(first)["all"]["coverage"],
            1.0,
        )
        _, _, exported = self.trainer.quantize(first)
        for name in ("w1", "w2", "w3"):
            np.testing.assert_array_equal(first[name], exported[name])

    def test_provisional_order_honors_exact_override_combined_target(self):
        candidates = [self.trainer.initialize(1), self.trainer.initialize(2)]
        reports = [{
            "seed": 1,
            "quantized_metrics": {"validation": {
                "outcome_mse": 0.1,
                "combined_target_mse": 0.4,
            }},
        }, {
            "seed": 2,
            "quantized_metrics": {"validation": {
                "outcome_mse": 0.2,
                "combined_target_mse": 0.3,
            }},
        }]
        arguments = argparse.Namespace(
            auxiliary_weight=0.25, batch_size=256, epochs=50,
            patience=8, learning_rate=0.001, weight_decay=1e-5,
            qat_epochs=4,
        )
        model = self.trainer.build_report(
            candidates, reports, {"corpus_sha256": "a" * 64}, arguments
        )
        self.assertEqual(model["training"]["provisional_seed"], 2)
        self.assertIn("combined-target", model["training"]["selection"])

    def test_dataset_release_uses_uint16_sparse_indices(self):
        samples = [corpus.NativeSample(
            game_key="memory", split_group="memory", turn=0, player=0,
            active=(0, 315, 1155), outcome=1.0,
            auxiliary_value=None, exact=False, symmetry="identity",
        )]
        dataset = self.trainer.dataset_from_samples(samples, release=True)
        self.assertEqual(dataset.active[0].dtype, np.uint16)
        self.assertIsNone(samples[0])

    def test_qat_tie_retains_pre_qat_best(self):
        datasets = {name: self.dataset()
                    for name in ("train", "validation", "test")}
        with mock.patch.object(self.trainer, "train_batch", return_value=0.0):
            _, report = self.trainer.train_seed(
                datasets, 11, 1, 1, 4, 0.001, 1e-5, 0.25, 2
            )
        self.assertTrue(report["qat_selection"]["pre_qat_retained"])
        self.assertEqual(report["qat_selection"]["selected_qat_epoch"], 0)
        self.assertTrue(report["qat_selection"]["fixed_scale_qat"])
        qat_history = [item for item in report["history"] if "qat_epoch" in item]
        self.assertEqual(len(qat_history), 2)
        self.assertTrue(all(
            item["fixed_scales"] == report["qat_selection"]["fixed_scales"]
            for item in qat_history
        ))
        self.assertIn(
            "turn_calibration", report["quantized_metrics"]["validation"]
        )

    def test_model_selection_is_explicitly_provisional(self):
        candidates = [self.trainer.initialize(1), self.trainer.initialize(2)]
        reports = [{
            "seed": seed,
            "quantized_metrics": {
                "validation": {
                    "outcome_mse": loss,
                    "combined_target_mse": loss,
                }
            },
        } for seed, loss in ((1, 0.2), (2, 0.3))]
        arguments = argparse.Namespace(
            auxiliary_weight=0.25, batch_size=256, epochs=50,
            patience=8, learning_rate=0.001, weight_decay=1e-5,
            qat_epochs=4,
        )
        model = self.trainer.build_report(
            candidates, reports, {"corpus_sha256": "a" * 64}, arguments
        )
        self.assertIsNone(model["training"]["chosen_seed"])
        self.assertEqual(model["training"]["provisional_seed"], 1)
        self.assertEqual(
            model["training"]["external_actual_clock_selection"]["status"],
            "pending",
        )
        self.assertIn("external", model["training"]["selection"])
        self.assertTrue(all(
            len(checkpoint["checkpoint_sha256"]) == 64
            for checkpoint in model["checkpoints"]
        ))

    def test_cumulative_seed_and_native_lineage_has_explicit_mode(self):
        seed_runtime = self.trainer.round1_trainer.UNTRAINED_SEED_RUNTIME
        seed_lines = seed_runtime.read_text(encoding="utf-8").splitlines()
        seed = corpus.NativeModelArtifact(
            artifact_sha256=hashlib.sha256(seed_runtime.read_bytes()).hexdigest(),
            model_sha256=seed_lines[3],
            packed_sha256=seed_lines[4],
        )
        native = corpus.NativeModelArtifact(
            artifact_sha256="7" * 64,
            model_sha256="8" * 64,
            packed_sha256="9" * 64,
        )
        report = self.trainer.checkpoint_provenance([
            types.SimpleNamespace(model_artifacts=(seed, native))
        ])
        self.assertEqual(
            report["mode"], "cumulative-native-runtime-models/v2"
        )

    def exportable_model(self):
        candidates = [self.trainer.initialize(1), self.trainer.initialize(2)]
        reports = [{
            "seed": seed,
            "quantized_metrics": {
                "validation": {
                    "outcome_mse": loss,
                    "combined_target_mse": loss,
                }
            },
        } for seed, loss in ((1, 0.2), (2, 0.3))]
        source_digest = "a" * 64
        source_sha256 = {f"sha256:{source_digest}": source_digest}
        artifact = {
            "artifact_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "packed_sha256": "d" * 64,
        }
        corpus_report = {
            "source_sha256": source_sha256,
            "corpus_sha256": hashlib.sha256(json.dumps(
                sorted(source_sha256.items()), separators=(",", ":")
            ).encode()).hexdigest(),
            "corpus_validator_sha256": hashlib.sha256(
                pathlib.Path(corpus.__file__).read_bytes()
            ).hexdigest(),
            "restart_corpus_validator_sha256": hashlib.sha256(
                pathlib.Path(
                    self.trainer.restart_contract.__file__
                ).read_bytes()
            ).hexdigest(),
            "augmentation": {
                "reflection": True,
                "rotation": "player-two-canonicalization-in-feature-encoder",
                "grouping": "whole-game-before-augmentation",
            },
            "lineage": {
                "strict_current": [{
                    "manifest_sha256": "1" * 64,
                    "build_provenance_sha256": "2" * 64,
                    "binary_sha256": "3" * 64,
                    "shard_sha256": ["4" * 64],
                    "games": 8,
                    "seed": 17,
                }],
                "archived_round1": [],
                "live_restart_round2": [],
            },
            "generation": {
                "checkpoint_provenance": {
                    "mode": "native-runtime-models/v1",
                    "artifacts": [artifact],
                },
                "model_artifact_sha256": [artifact["artifact_sha256"]],
            },
        }
        arguments = argparse.Namespace(
            auxiliary_weight=0.25, batch_size=256, epochs=50,
            patience=8, learning_rate=0.001, weight_decay=1e-5,
            qat_epochs=4,
        )
        return self.trainer.build_report(
            candidates, reports, corpus_report, arguments
        )

    def test_round2_exporter_selects_every_identified_seed(self):
        model = self.exportable_model()
        header, metadata = self.exporter.render(model, "e" * 64, seed=2)
        runtime = self.exporter.render_runtime(model, "e" * 64, seed=2)
        self.assertEqual(metadata["training_seed"], 2)
        self.assertIn("kTrainingSeed = 2ULL", header)
        self.assertEqual(runtime.splitlines()[3], "e" * 64)
        self.assertEqual(
            hashlib.sha256(
                base64.b64decode(runtime.splitlines()[6])
            ).hexdigest(),
            runtime.splitlines()[4],
        )

    def test_round2_exporter_rejects_implicit_provisional_seed(self):
        model = self.exportable_model()
        with self.assertRaisesRegex(ValueError, "explicit seed"):
            self.exporter.render_runtime(model, "e" * 64)

    def test_round2_exporter_rejects_stale_semantic_provenance(self):
        model = self.exportable_model()
        model["provenance"]["trainer_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trainer SHA-256"):
            self.exporter.render_runtime(model, "e" * 64, seed=1)

        model = self.exportable_model()
        model["checkpoints"][0]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            self.exporter.render_runtime(model, "e" * 64, seed=1)

    def test_explicit_restart_paths_merge_without_observed_labels(self):
        artifact = corpus.NativeModelArtifact(
            artifact_sha256="1" * 64,
            model_sha256="2" * 64,
            packed_sha256="3" * 64,
        )
        sample = corpus.NativeSample(
            game_key="sample", split_group="sample", turn=0, player=0,
            active=(0,), outcome=1.0, auxiliary_value=None, exact=False,
            symmetry="identity",
        )

        def native_game(key, group, winner):
            return corpus.NativeGame(
                key=key, split_group=group, seed=1, game=0,
                shard_index=0, winner=winner, samples=(sample,),
                producer_sha256="4" * 64,
                build_provenance_sha256="5" * 64,
                model_artifacts=(artifact,), search_stats={"searches": 1},
                opening_depth=0, temperature_turns=12,
                transcript_sha256="6" * 64, build_contract={"fixture": True},
            )

        base = native_game("base", "base", 0)
        restart = native_game("restart", "live-source", 1)
        splits = {
            name: [sample] for name in ("train", "validation", "test")
        }
        assignments = {"base": "train", "live-source": "test"}
        restart_lineage = {
            "manifest_sha256": "7" * 64,
            "build_provenance_sha256": "8" * 64,
            "binary_sha256": "9" * 64,
            "collector_tsv_sha256": "a" * 64,
            "arena_manifest_sha256": "b" * 64,
            "asserted_source_sha256": "c" * 64,
            "exclusion_registry_sha256": "d" * 64,
            "source_binding_status": "asserted-not-api-verified",
            "games": 1,
            "selected_prefixes": 1,
        }
        base_lineage = {"strict_current": [], "archived_round1": []}
        with (
            mock.patch.object(
                corpus, "load_games",
                return_value=([base], {"sha256:" + "e" * 64: "e" * 64},
                              base_lineage),
            ),
            mock.patch.object(
                self.trainer.restart_contract, "load_games",
                return_value=([restart], {"sha256:" + "f" * 64: "f" * 64},
                              restart_lineage),
            ) as restart_loader,
            mock.patch.object(
                corpus, "prepare_splits",
                return_value=(splits, {name: 0 for name in splits}, assignments),
            ),
            mock.patch.object(corpus, "build_contracts", return_value=[]),
        ):
            _, report = self.trainer.load_datasets(
                [pathlib.Path("current/shard.jsonl")],
                restart_round2_paths=[pathlib.Path("restart/shard.jsonl")],
            )
        restart_loader.assert_called_once()
        self.assertEqual(report["observed_move_policy_labels"], 0)
        self.assertEqual(
            report["lineage"]["live_restart_round2"], [restart_lineage]
        )
        self.assertEqual(report["games"], 2)


if __name__ == "__main__":
    unittest.main()
