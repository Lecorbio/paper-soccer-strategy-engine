import base64
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "jacek_native_bfm"
    / "check_purity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "jacek_native_bfm_purity_under_test", CHECKER_PATH
)
purity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = purity
SPEC.loader.exec_module(purity)


class JacekNativeBfmPurityTest(unittest.TestCase):
    def fixture(self, temporary, source="int native_search();\n", **config_updates):
        root = pathlib.Path(temporary)
        bot = root / "submissions/codingame/bots/jacek_native_bfm"
        bot.mkdir(parents=True)
        production = bot / "bot.cpp"
        production.write_text(source, encoding="utf-8")
        (bot / "sources.txt").write_text(
            "submissions/codingame/bots/jacek_native_bfm/bot.cpp\n",
            encoding="utf-8",
        )
        dependencies = list(purity.EXPECTED_PURITY_DEPENDENCIES)
        semantic_dependencies = list(purity.EXPECTED_SEMANTIC_DEPENDENCIES)
        trainer = root / "tools/train_jacek_native.py"
        trainer.parent.mkdir(parents=True)
        trainer.write_text("def train_native():\n    return None\n", encoding="utf-8")
        for relative in dependencies[1:]:
            dependency = root / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            if dependency != trainer:
                dependency.write_text("// native dependency\n", encoding="utf-8")
        corpus = root / semantic_dependencies[0]
        corpus.write_text(
            "FORBIDDEN_PROVENANCE = (\n"
            "    'rank_4', 'rank-4', 'rank4', 'replay-book', 'replay_book',\n"
            "    'alpha-beta-teacher', 'alpha_beta_teacher',\n"
            ")\n"
            "def _check_purity(record, line_number):\n"
            "    return None\n"
            "def validate_record(record, line_number=1):\n"
            "    _check_purity(record, line_number)\n"
            "    forbidden = set(record) & {\n"
            "        'policy', 'policy_target', 'teacher_move',\n"
            "        'expert_move', 'rank4_value',\n"
            "    }\n"
            "    return forbidden\n",
            encoding="utf-8",
        )
        seed_generator = root / "tools/generate_jacek_native_seed.py"
        seed_generator.write_text(
            "def deterministic_native_seed():\n    return 17\n",
            encoding="utf-8",
        )
        seed_payload = b"\0\0"
        seed_packed_sha = hashlib.sha256(seed_payload).hexdigest()
        seed_descriptor = {
            "schema": "papersoccer.jacek-native-untrained-seed/v1",
            "model_schema": "jacek_native_model/v1",
            "feature_schema": (
                "canonical-edges316-onehot-true-turn-distance105x8-v1"
            ),
            "training": None,
            "incumbent_dependencies": False,
            "protected_data": False,
            "generator_sha256": hashlib.sha256(
                seed_generator.read_bytes()
            ).hexdigest(),
            "weights": {
                "counts": {"w1": 1, "w2": 1, "w3": 1},
                "scales": {"w1": 0.01, "w2": 0.05, "w3": 0.05},
                "packed_sha256": seed_packed_sha,
            },
        }
        seed_descriptor_path = root / "models/jacek_native_untrained_seed.json"
        seed_descriptor_text = json.dumps(
            seed_descriptor, sort_keys=True, separators=(",", ":")
        ) + "\n"
        seed_descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        seed_descriptor_path.write_text(seed_descriptor_text, encoding="utf-8")
        seed_runtime = (
            "papersoccer.jacek-native-runtime-model/v1\n"
            "jacek_native_model/v1\n"
            "canonical-edges316-onehot-true-turn-distance105x8-v1\n"
            f"{hashlib.sha256(seed_descriptor_text.encode()).hexdigest()}\n"
            f"{seed_packed_sha}\n"
            "0.01 0.05 0.05\n"
            f"{base64.b64encode(seed_payload).decode()}\n"
        )
        seed_runtime_path = root / "models/jacek_native_untrained_seed.runtime"
        seed_runtime_path.write_text(seed_runtime, encoding="utf-8")
        seed_identity = {
            "artifact_sha256": hashlib.sha256(seed_runtime.encode()).hexdigest(),
            "model_sha256": hashlib.sha256(
                seed_descriptor_text.encode()
            ).hexdigest(),
            "packed_sha256": seed_packed_sha,
        }
        build_sources = [
            {
                "path": relative,
                "sha256": hashlib.sha256(relative.encode()).hexdigest(),
            }
            for relative in purity.BUILD_SOURCE_PATHS
        ]
        build_producer = hashlib.sha256(json.dumps(
            [[entry["path"], entry["sha256"]] for entry in build_sources],
            separators=(",", ":"),
        ).encode()).hexdigest()
        compiler_version = "fixture-cxx 1.0"
        build_contract = {
            "schema": purity.BUILD_PROVENANCE_SCHEMA,
            "binary": {
                "path": "selfplay-binary",
                "sha256": hashlib.sha256(b"fixture binary").hexdigest(),
            },
            "compiler": {
                "executable": "fixture-cxx",
                "sha256": hashlib.sha256(b"fixture compiler").hexdigest(),
                "version": compiler_version,
                "version_sha256": hashlib.sha256(
                    compiler_version.encode()
                ).hexdigest(),
            },
            "build_argv": list(purity.CANONICAL_BUILD_ARGV),
            "producer_sha256": build_producer,
            "sources": build_sources,
        }
        build_sha = hashlib.sha256((json.dumps(
            build_contract, sort_keys=True, separators=(",", ":")
        ) + "\n").encode()).hexdigest()
        model = {
            "schema": "jacek_native_model/v1",
            "feature_schema": (
                "canonical-edges316-onehot-true-turn-distance105x8-v1"
            ),
            "target": {
                "primary": "mover-relative-final-outcome",
                "policy_target": None,
            },
            "provenance": {
                "incumbent_labels": False,
                "protected_data": False,
                "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
                "corpus_validator_sha256": hashlib.sha256(
                    corpus.read_bytes()
                ).hexdigest(),
                "corpus_sha256": hashlib.sha256(b"fixture corpus").hexdigest(),
                "source_sha256": {
                    "fixture.jsonl": hashlib.sha256(b"fixture shard").hexdigest()
                },
                "generation": {
                    "producer_sha256": [build_producer],
                    "build_provenance_sha256": [build_sha],
                    "build_contracts": [{
                        "sha256": build_sha,
                        "contract": build_contract,
                    }],
                    "model_artifact_sha256": [
                        seed_identity["artifact_sha256"]
                    ],
                    "checkpoint_provenance": {
                        "mode": "untrained-seed-bootstrap/v1",
                        "artifacts": [seed_identity],
                    },
                },
            },
        }
        model_path = root / dependencies[0]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model), encoding="utf-8")

        config = {
            "schema": "papersoccer.codingame-submission.v1",
            "sources": "sources.txt",
            "output": "submission.cpp",
            "source_limit": 94_999,
            "allowed_local_includes": [],
            "purity_dependencies": dependencies,
            "purity_semantic_dependencies": semantic_dependencies,
        }
        config.update(config_updates)
        (bot / "submission.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (bot / "submission.cpp").write_text(source, encoding="utf-8")
        return root, bot

    def bind_native_checkpoint(self, root, bot):
        trainer = root / "tools/train_jacek_native.py"
        corpus = root / purity.EXPECTED_SEMANTIC_DEPENDENCIES[0]
        seed_runtime_path = root / "models/jacek_native_untrained_seed.runtime"
        seed_runtime_lines = seed_runtime_path.read_text(
            encoding="utf-8"
        ).splitlines()
        seed_identity = {
            "artifact_sha256": hashlib.sha256(
                seed_runtime_path.read_bytes()
            ).hexdigest(),
            "model_sha256": seed_runtime_lines[3],
            "packed_sha256": seed_runtime_lines[4],
        }
        active_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
        active_generation = json.loads(
            active_path.read_text(encoding="utf-8")
        )["provenance"]["generation"]
        build_generation = {
            field: json.loads(json.dumps(active_generation[field]))
            for field in (
                "producer_sha256",
                "build_provenance_sha256",
                "build_contracts",
            )
        }
        shapes = purity.MODEL_SHAPES
        weights = {
            name: {
                "shape": list(shape),
                "values": [0] * (shape[0] * shape[1] if len(shape) == 2 else shape[0]),
            }
            for name, shape in shapes.items()
        }
        quantization = {
            "bits": 3,
            "minimum": -3,
            "maximum": 3,
            "scheme": "symmetric-per-layer-round-to-nearest",
            "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
            "scales": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
            "weights": weights,
        }
        checkpoint_model = {
            "schema": purity.MODEL_SCHEMA,
            "feature_schema": purity.FEATURE_SCHEMA,
            "architecture": {
                "inputs": 1156,
                "hidden_one": 32,
                "hidden_two": 32,
                "outputs": 1,
                "biases": False,
                "hidden_one_activation":
                    "square-nonnegative-leaky-0.01-negative",
                "hidden_two_activation": "leaky-relu-0.01",
                "output_activation": "tanh",
            },
            "rules": {
                "width": 8,
                "height": 10,
                "goal_rule": "own-goals-allowed",
                "blocked_rule": "mover-loses",
            },
            "target": {
                "primary": "mover-relative-final-outcome",
                "auxiliary": "stable-native-bfm-reanalysis",
                "auxiliary_weight": 0.25,
                "policy_target": None,
            },
            "provenance": {
                "incumbent_labels": False,
                "protected_data": False,
                "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
                "corpus_validator_sha256": hashlib.sha256(
                    corpus.read_bytes()
                ).hexdigest(),
                "corpus_sha256": hashlib.sha256(
                    b"prior native corpus"
                ).hexdigest(),
                "source_sha256": {
                    "sha256:fixture": hashlib.sha256(
                        b"prior native shard"
                    ).hexdigest()
                },
                "generation": {
                    **build_generation,
                    "model_artifact_sha256": [
                        seed_identity["artifact_sha256"]
                    ],
                    "checkpoint_provenance": {
                        "mode": "untrained-seed-bootstrap/v1",
                        "artifacts": [seed_identity],
                    },
                },
            },
            "checkpoints": [{
                "seed": 23,
                "quantization": quantization,
            }],
        }
        model_relative = "models/jacek_native_prior_checkpoint.json"
        runtime_relative = "models/jacek_native_prior_checkpoint.runtime"
        model_path = root / model_relative
        model_text = json.dumps(
            checkpoint_model, sort_keys=True, separators=(",", ":")
        ) + "\n"
        model_path.write_text(model_text, encoding="utf-8")
        model_sha = hashlib.sha256(model_text.encode()).hexdigest()
        weight_count = sum(
            shape[0] * shape[1] if len(shape) == 2 else shape[0]
            for shape in shapes.values()
        )
        payload = bytes((weight_count * 3 + 7) // 8)
        packed_sha = hashlib.sha256(payload).hexdigest()
        runtime = (
            f"{purity.RUNTIME_SCHEMA}\n"
            f"{purity.MODEL_SCHEMA}\n"
            f"{purity.FEATURE_SCHEMA}\n"
            f"{model_sha}\n"
            f"{packed_sha}\n"
            "1 1 1\n"
            f"{base64.b64encode(payload).decode()}\n"
        )
        runtime_path = root / runtime_relative
        runtime_path.write_text(runtime, encoding="utf-8")
        identity = {
            "artifact_sha256": hashlib.sha256(runtime.encode()).hexdigest(),
            "model_sha256": model_sha,
            "packed_sha256": packed_sha,
        }

        config_path = bot / "submission.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["native_checkpoint_provenance"] = [{
            "model": model_relative,
            "runtime": runtime_relative,
        }]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        active = json.loads(active_path.read_text(encoding="utf-8"))
        active["provenance"]["generation"].update({
            "model_artifact_sha256": [identity["artifact_sha256"]],
            "checkpoint_provenance": {
                "mode": "native-runtime-models/v1",
                "artifacts": [identity],
            },
        })
        active_path.write_text(json.dumps(active), encoding="utf-8")
        return model_path, runtime_path, identity

    def round2_fixture(self, temporary):
        root, bot = self.fixture(temporary)
        config_path = bot / "submission.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["purity_dependencies"] = list(purity.ROUND2_PURITY_DEPENDENCIES)
        config["purity_semantic_dependencies"] = list(
            purity.ROUND2_SEMANTIC_DEPENDENCIES
        )
        config_path.write_text(json.dumps(config), encoding="utf-8")

        for relative in purity.ROUND2_PURITY_DEPENDENCIES[1:]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("# native round-two dependency\n", encoding="utf-8")
        round2_corpus = root / purity.ROUND2_SEMANTIC_DEPENDENCIES[0]
        round2_corpus.write_text(
            "import jacek_native_corpus as round1\n"
            "def validate_record(record, line_number=1):\n"
            "    round1._check_purity(record, line_number)\n"
            "    return round1.validate_record(record, line_number)\n",
            encoding="utf-8",
        )
        restart_corpus = root / purity.ROUND2_SEMANTIC_DEPENDENCIES[1]
        restart_corpus.write_text(
            "import jacek_native_corpus as round1\n"
            "import jacek_native_corpus_round2 as round2\n"
            "OBSERVED_USAGE = 'state-construction-only'\n"
            "def validate_record(record, manifest, collector, selected, "
            "line_number=1):\n"
            "    round1._check_purity(record, line_number)\n"
            "    return round2.validate_record(record, line_number)\n",
            encoding="utf-8",
        )
        trainer = root / "tools/train_jacek_native_round2.py"
        trainer.write_text("def train_round_two():\n    return None\n", encoding="utf-8")
        model_path = root / purity.ROUND2_PURITY_DEPENDENCIES[0]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(
            (root / purity.EXPECTED_PURITY_DEPENDENCIES[0]).read_bytes()
        )
        model = json.loads(model_path.read_text(encoding="utf-8"))
        seed_runtime = root / "models/jacek_native_untrained_seed.runtime"
        seed_lines = seed_runtime.read_text(encoding="utf-8").splitlines()
        seed_identity = {
            "artifact_sha256": hashlib.sha256(seed_runtime.read_bytes()).hexdigest(),
            "model_sha256": seed_lines[3],
            "packed_sha256": seed_lines[4],
        }
        shapes = purity.MODEL_SHAPES
        quantization = {
            "bits": 3,
            "minimum": -3,
            "maximum": 3,
            "scheme": "symmetric-per-layer-round-to-nearest",
            "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
            "scales": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
            "weights": {
                name: {
                    "shape": list(shape),
                    "values": [0] * (
                        shape[0] * shape[1] if len(shape) == 2 else shape[0]
                    ),
                }
                for name, shape in shapes.items()
            },
        }
        checkpoint_payload = {"seed": 31, "model": {}, "quantization": quantization}
        checkpoint = {
            **checkpoint_payload,
            "checkpoint_sha256": hashlib.sha256(json.dumps(
                checkpoint_payload, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n").hexdigest(),
        }
        source_digest = hashlib.sha256(b"round two shard").hexdigest()
        generation = model["provenance"]["generation"]
        build_sha = generation["build_provenance_sha256"][0]
        build_binary_sha = generation["build_contracts"][0]["contract"][
            "binary"
        ]["sha256"]
        model.update({
            "architecture": {
                "inputs": 1156, "hidden_one": 32, "hidden_two": 32,
                "outputs": 1, "biases": False,
                "hidden_one_activation":
                    "square-nonnegative-leaky-0.01-negative",
                "hidden_two_activation": "leaky-relu-0.01",
                "output_activation": "tanh",
            },
            "rules": {
                "width": 8, "height": 10,
                "goal_rule": "own-goals-allowed",
                "blocked_rule": "mover-loses",
            },
            "target": {
                "primary": "mover-relative-final-outcome",
                "auxiliary": "stable-native-bfm-reanalysis",
                "auxiliary_weight": 0.25,
                "policy_target": None,
            },
            "checkpoints": [checkpoint],
            "training": {
                "seeds": [31], "chosen_seed": None, "provisional_seed": 31,
                "external_actual_clock_selection": {
                    "required": True, "status": "pending",
                    "criterion": "native-actual-clock-match-strength",
                    "eligible_seed_order": [31],
                },
            },
        })
        model["provenance"].update({
            "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
            "corpus_validator_sha256": hashlib.sha256(
                round2_corpus.read_bytes()).hexdigest(),
            "restart_corpus_validator_sha256": hashlib.sha256(
                restart_corpus.read_bytes()).hexdigest(),
            "source_sha256": {f"sha256:{source_digest}": source_digest},
            "corpus_sha256": hashlib.sha256(json.dumps(
                [[f"sha256:{source_digest}", source_digest]],
                separators=(",", ":"),
            ).encode()).hexdigest(),
            "lineage": {
                "strict_current": [{
                    "manifest_sha256": "1" * 64,
                    "build_provenance_sha256": build_sha,
                    "binary_sha256": build_binary_sha,
                    "shard_sha256": [source_digest],
                    "games": 8,
                    "seed": 17,
                }],
                "archived_round1": [],
                "live_restart_round2": [],
            },
        })
        generation.update({
            "model_artifact_sha256": [seed_identity["artifact_sha256"]],
            "checkpoint_provenance": {
                "mode": "untrained-seed-bootstrap/v1",
                "artifacts": [seed_identity],
            },
        })
        model_path.write_text(json.dumps(model), encoding="utf-8")
        return root, bot, model_path

    def test_clean_native_source_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_round1_header_guard_rejects_stale_deployment_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            bot = pathlib.Path(temporary) / "jacek_native_bfm"
            bot.mkdir()
            header = bot / "jacek_native_model.hpp"
            exporter_path = (
                ROOT
                / "submissions/codingame/tools/"
                "generate_jacek_native_model.py"
            )
            exporter_spec = importlib.util.spec_from_file_location(
                "jacek_native_round1_exporter_test", exporter_path
            )
            exporter = importlib.util.module_from_spec(exporter_spec)
            exporter_spec.loader.exec_module(exporter)
            model_path = ROOT / purity.ROUND1_PURITY_DEPENDENCIES[0]
            model_raw = model_path.read_bytes()
            canonical, _ = exporter.render(
                json.loads(model_raw), hashlib.sha256(model_raw).hexdigest()
            )
            header.write_text(canonical, encoding="utf-8")
            purity._validate_round1_header(ROOT, bot, [header.resolve()])
            header.write_text(
                canonical + "// stale round-two bytes\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "round-one model header"):
                purity._validate_round1_header(ROOT, bot, [header.resolve()])

    def test_round2_purity_branch_accepts_native_reanalysis_teacher_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot, _ = self.round2_fixture(temporary)
            (root / "tools/jacek_native_workflow_round2.py").write_text(
                "teacher = 'native reanalysis checkpoint'\n", encoding="utf-8"
            )
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_round2_purity_requires_immutable_pending_selection_and_restart_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot, model_path = self.round2_fixture(temporary)
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["training"]["chosen_seed"] = 31
            model["training"]["external_actual_clock_selection"]["status"] = "complete"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actual-clock selection"):
                purity.purity_violations(bot, root)

        with tempfile.TemporaryDirectory() as temporary:
            root, bot, model_path = self.round2_fixture(temporary)
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["provenance"]["restart_corpus_validator_sha256"] = "0" * 64
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "restart corpus-validator"):
                purity.purity_violations(bot, root)

    def test_deployment_routes_pending_model_through_activation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot, model_path = self.round2_fixture(temporary)
            deployment_path = root / purity.ROUND2_DEPLOYMENT_PATH
            deployment_path.write_text("{}\n", encoding="utf-8")
            selection_path = root / purity.ROUND2_SELECTION_PATH
            runtime_path = root / purity.ROUND2_RUNTIME_PATH
            selection_path.write_text("{}\n", encoding="utf-8")
            runtime_path.write_text("selected runtime\n", encoding="utf-8")
            header = "#pragma once\n// selected pending model\n"
            (bot / "jacek_native_model.hpp").write_text(
                header, encoding="utf-8"
            )
            activation = mock.Mock()
            activation.load_deployment.return_value = {
                "model_path": model_path.resolve(),
                "selection_path": selection_path.resolve(),
                "runtime_path": runtime_path.resolve(),
                "checkpoint_paths": [],
                "baseline_model_path": (
                    root / "models/jacek_native_bootstrap_model.json"
                ).resolve(),
                "baseline_runtime_path": (
                    root / "models/jacek_native_untrained_seed.runtime"
                ).resolve(),
                "evidence_paths": [],
            }
            activation.render_deployment.return_value = (header, {})
            with mock.patch.object(
                purity, "_load_round2_activation", return_value=activation
            ):
                self.assertEqual(purity.purity_violations(bot, root), [])
            activation.load_deployment.assert_called_once_with(
                deployment_path.resolve(), root.resolve()
            )

    def test_incumbent_and_label_dependencies_are_rejected(self):
        forbidden = {
            "rank_4": "rank-4 dependency",
            "rank4": "rank-4 dependency",
            "replay_book": "replay-book dependency",
            "replay_value_model": "replay-value dependency",
            "teacher_residual": "teacher-residual dependency",
            "alpha_beta": "alpha-beta dependency",
            "teacher": "teacher-label dependency",
        }
        for token, expected in forbidden.items():
            with self.subTest(token=token), tempfile.TemporaryDirectory() as temporary:
                root, bot = self.fixture(temporary, source=f"int {token};\n")
                self.assertTrue(
                    any(
                        expected in violation
                        for violation in purity.purity_violations(bot, root)
                    )
                )

    def test_comparison_harness_is_outside_the_production_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            (bot / "comparison_gate.cpp").write_text(
                '#include "../rank_4/bot.cpp"\n', encoding="utf-8"
            )
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_transitive_training_dependency_is_scanned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            (root / "tools/jacek_native_workflow.py").write_text(
                "rank4 = 'forbidden dependency'\n", encoding="utf-8"
            )
            self.assertTrue(
                any(
                    "tools/jacek_native_workflow.py: rank-4 dependency"
                    in violation
                    for violation in purity.purity_violations(bot, root)
                )
            )

    def test_model_provenance_flags_and_trainer_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["provenance"]["incumbent_labels"] = True
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "permits incumbent labels"):
                purity.purity_violations(bot, root)

        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            (root / "tools/train_jacek_native.py").write_text(
                "def train_native():\n    return 7\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "trainer SHA-256 is stale"):
                purity.purity_violations(bot, root)

        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            seed_path = root / "models/jacek_native_untrained_seed.json"
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            seed["protected_data"] = True
            seed_path.write_text(json.dumps(seed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "permits protected data"):
                purity.purity_violations(bot, root)

    def test_model_build_provenance_hash_and_producer_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["provenance"]["generation"]["build_contracts"][0][
                "contract"
            ]["binary"]["sha256"] = "b" * 64
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract SHA-256 is stale"):
                purity.purity_violations(bot, root)

        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            generation = model["provenance"]["generation"]
            item = generation["build_contracts"][0]
            contract = item["contract"]
            contract["producer_sha256"] = "c" * 64
            item["sha256"] = hashlib.sha256((json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            ) + "\n").encode()).hexdigest()
            generation["build_provenance_sha256"] = [item["sha256"]]
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "producer SHA-256"):
                purity.purity_violations(bot, root)

    def test_bootstrap_requires_the_exact_untrained_seed_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            artifact = model["provenance"]["generation"][
                "checkpoint_provenance"
            ]["artifacts"][0]
            artifact["packed_sha256"] = "b" * 64
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact untrained seed"):
                purity.purity_violations(bot, root)

    def test_native_checkpoint_requires_file_backed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            generation = model["provenance"]["generation"]
            generation["checkpoint_provenance"]["mode"] = (
                "native-runtime-models/v1"
            )
            arbitrary = {
                "artifact_sha256": "b" * 64,
                "model_sha256": "c" * 64,
                "packed_sha256": "d" * 64,
            }
            generation["checkpoint_provenance"]["artifacts"] = [arbitrary]
            generation["model_artifact_sha256"] = [
                arbitrary["artifact_sha256"]
            ]
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file-backed model/runtime"):
                purity.purity_violations(bot, root)

    def test_native_checkpoint_cannot_self_reference_the_active_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            config_path = bot / "submission.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["native_checkpoint_provenance"] = [{
                "model": purity.EXPECTED_PURITY_DEPENDENCIES[0],
                "runtime": "models/jacek_native_untrained_seed.runtime",
            }]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not self-reference"):
                purity.purity_violations(bot, root)

    def test_file_backed_native_checkpoint_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            self.bind_native_checkpoint(root, bot)
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_one_native_model_can_back_multiple_checkpoint_runtimes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path, first_runtime_path, _ = self.bind_native_checkpoint(
                root, bot
            )
            model = json.loads(model_path.read_text(encoding="utf-8"))
            second = json.loads(json.dumps(model["checkpoints"][0]))
            second["seed"] = 24
            second["quantization"]["weights"]["w3"]["values"][0] = 1
            model["checkpoints"].append(second)
            model_text = json.dumps(
                model, sort_keys=True, separators=(",", ":")
            ) + "\n"
            model_path.write_text(model_text, encoding="utf-8")
            model_sha = hashlib.sha256(model_text.encode()).hexdigest()
            rendered = [
                purity._checkpoint_runtime_bytes(model, model_sha, checkpoint)
                for checkpoint in model["checkpoints"]
            ]
            first_runtime_path.write_bytes(rendered[0][0])
            second_runtime_path = (
                root / "models/jacek_native_prior_checkpoint_24.runtime"
            )
            second_runtime_path.write_bytes(rendered[1][0])

            config_path = bot / "submission.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["native_checkpoint_provenance"].append({
                "model": model_path.relative_to(root).as_posix(),
                "runtime": second_runtime_path.relative_to(root).as_posix(),
            })
            config_path.write_text(json.dumps(config), encoding="utf-8")
            identities = sorted(
                [metadata for _, metadata in rendered],
                key=lambda value: (
                    value["artifact_sha256"],
                    value["model_sha256"],
                    value["packed_sha256"],
                ),
            )
            active_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            active = json.loads(active_path.read_text(encoding="utf-8"))
            generation = active["provenance"]["generation"]
            generation["model_artifact_sha256"] = [
                value["artifact_sha256"] for value in identities
            ]
            generation["checkpoint_provenance"]["artifacts"] = identities
            active_path.write_text(json.dumps(active), encoding="utf-8")
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_native_checkpoint_lineage_can_chain_to_the_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            first_model_path, _, first_identity = self.bind_native_checkpoint(
                root, bot
            )
            second_model = json.loads(
                first_model_path.read_text(encoding="utf-8")
            )
            second_model["provenance"]["generation"].update({
                "model_artifact_sha256": [
                    first_identity["artifact_sha256"]
                ],
                "checkpoint_provenance": {
                    "mode": "native-runtime-models/v1",
                    "artifacts": [first_identity],
                },
            })
            second_model["checkpoints"][0]["seed"] = 31
            second_model["checkpoints"][0]["quantization"]["weights"]["w3"][
                "values"
            ][0] = 1
            second_model_path = root / "models/jacek_native_round_two.json"
            second_model_text = json.dumps(
                second_model, sort_keys=True, separators=(",", ":")
            ) + "\n"
            second_model_path.write_text(second_model_text, encoding="utf-8")
            second_model_sha = hashlib.sha256(
                second_model_text.encode()
            ).hexdigest()
            runtime, second_identity = purity._checkpoint_runtime_bytes(
                second_model,
                second_model_sha,
                second_model["checkpoints"][0],
            )
            second_runtime_path = root / "models/jacek_native_round_two.runtime"
            second_runtime_path.write_bytes(runtime)

            config_path = bot / "submission.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["native_checkpoint_provenance"].append({
                "model": second_model_path.relative_to(root).as_posix(),
                "runtime": second_runtime_path.relative_to(root).as_posix(),
            })
            config_path.write_text(json.dumps(config), encoding="utf-8")
            active_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            active = json.loads(active_path.read_text(encoding="utf-8"))
            generation = active["provenance"]["generation"]
            generation["model_artifact_sha256"] = [
                second_identity["artifact_sha256"]
            ]
            generation["checkpoint_provenance"]["artifacts"] = [
                second_identity
            ]
            active_path.write_text(json.dumps(active), encoding="utf-8")
            self.assertEqual(purity.purity_violations(bot, root), [])

    def test_tampered_native_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            _, runtime_path, _ = self.bind_native_checkpoint(root, bot)
            runtime_path.write_text(
                runtime_path.read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not generated by"):
                purity.purity_violations(bot, root)

    def test_declared_native_identity_must_match_the_backing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            self.bind_native_checkpoint(root, bot)
            active_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            active = json.loads(active_path.read_text(encoding="utf-8"))
            generation = active["provenance"]["generation"]
            generation["checkpoint_provenance"]["artifacts"][0][
                "artifact_sha256"
            ] = "c" * 64
            generation["model_artifact_sha256"] = ["c" * 64]
            active_path.write_text(json.dumps(active), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match file-backed"):
                purity.purity_violations(bot, root)

    def test_native_checkpoint_model_rejects_incumbent_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            model_path, _, _ = self.bind_native_checkpoint(root, bot)
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["provenance"]["incumbent_labels"] = True
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "permits incumbent labels"):
                purity.purity_violations(bot, root)

    def test_corpus_guard_is_checked_semantically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            corpus = root / purity.EXPECTED_SEMANTIC_DEPENDENCIES[0]
            corpus.write_text(
                "FORBIDDEN_PROVENANCE = ('rank4',)\n"
                "def _check_purity(record, line_number):\n"
                "    return None\n"
                "def validate_record(record, line_number=1):\n"
                "    return record\n",
                encoding="utf-8",
            )
            model_path = root / purity.EXPECTED_PURITY_DEPENDENCIES[0]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["provenance"]["corpus_validator_sha256"] = hashlib.sha256(
                corpus.read_bytes()
            ).hexdigest()
            model_path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance guard is incomplete"):
                purity.purity_violations(bot, root)

    def test_source_limit_must_preserve_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary, source_limit=100_000)
            with self.assertRaisesRegex(ValueError, "source_limit must be exactly 94999"):
                purity.purity_violations(bot, root)

    def test_escaping_source_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, bot = self.fixture(temporary)
            (bot / "sources.txt").write_text("../outside.cpp\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes its allowed directory"):
                purity.purity_violations(bot, root)


if __name__ == "__main__":
    unittest.main()
