import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import unittest
from unittest import mock

try:
    import numpy as np  # noqa: F401
except ModuleNotFoundError:
    np = None

from tests.codingame.test_jacek_native_round2_selection import (
    JacekNativeRound2SelectionTest as SelectionFixture,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
ACTIVATION_PATH = (
    ROOT
    / "submissions/codingame/tools/jacek_native_round2_activation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "jacek_native_round2_activation_under_test", ACTIVATION_PATH
)
activation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = activation
SPEC.loader.exec_module(activation)


@unittest.skipIf(np is None, "round-two activation tests require NumPy")
class JacekNativeRound2ActivationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SelectionFixture.setUpClass()

    def setUp(self):
        self.fixture = SelectionFixture("test_profiles_are_frozen_and_training_model_has_no_chosen_seed")
        self.fixture.setUp()
        self.directory = self.fixture.directory
        for relative in (
            "models/jacek_native_untrained_seed.json",
            "models/jacek_native_untrained_seed.runtime",
            "tools/generate_jacek_native_seed.py",
        ):
            destination = self.directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.fixture.baseline_model = (
            self.directory / "models/jacek_native_bootstrap_model.json"
        )
        shutil.copyfile(
            ROOT / "models/jacek_native_bootstrap_model.json",
            self.fixture.baseline_model,
        )
        tracked_baseline_runtime = (
            self.directory / "models/jacek_native_bootstrap_seed_20260813.runtime"
        )
        tracked_baseline_runtime.write_bytes(
            self.fixture.baseline_runtime.read_bytes()
        )
        self.fixture.baseline_runtime = tracked_baseline_runtime
        seed_runtime = (
            self.directory / "models/jacek_native_untrained_seed.runtime"
        )
        seed_lines = seed_runtime.read_text(encoding="utf-8").splitlines()
        seed_identity = {
            "artifact_sha256": hashlib.sha256(
                seed_runtime.read_bytes()
            ).hexdigest(),
            "model_sha256": seed_lines[3],
            "packed_sha256": seed_lines[4],
        }
        generation = self.fixture.model["provenance"]["generation"]
        generation["checkpoint_provenance"] = {
            "mode": "untrained-seed-bootstrap/v1",
            "artifacts": [seed_identity],
        }
        generation["model_artifact_sha256"] = [
            seed_identity["artifact_sha256"]
        ]
        self.fixture.model_path = (
            self.directory / "models/jacek_native_round2_candidate.json"
        )
        self.fixture._write_model()
        self.fixture._write_runtimes()
        self.fixture.reports = (
            self.directory
            / "models/jacek_native_round2_gate_evidence/promotion"
        )
        self.paths = self.fixture._all_reports()
        self.sidecar_path = (
            self.directory / "models/jacek_native_round2_selection.json"
        )
        self.sidecar = self.fixture.selection.finalize_selection(
            model_path=self.fixture.model_path,
            baseline_model=self.fixture.baseline_model,
            baseline_seed=self.fixture.baseline_seed,
            baseline_runtime=self.fixture.baseline_runtime,
            report_paths=self.paths,
            output=self.sidecar_path,
        )
        seed = self.sidecar["selected"]["seed"]
        self.runtime_path = self._install_selected_runtime(seed)
        self.deployment_path = self.directory / "deployment.json"

    def tearDown(self):
        self.fixture.tearDown()

    def _install_selected_runtime(self, seed):
        installed = (
            self.directory / "models/jacek_native_round2_selected.runtime"
        )
        installed.write_bytes(self.fixture.candidate_runtimes[seed].read_bytes())
        return installed

    def create(self, checkpoint_pairs=()):
        return activation.create_deployment(
            model_path=self.fixture.model_path,
            selection_path=self.sidecar_path,
            runtime_path=self.runtime_path,
            baseline_model=self.fixture.baseline_model,
            baseline_seed=self.fixture.baseline_seed,
            baseline_runtime=self.fixture.baseline_runtime,
            report_paths=self.paths,
            checkpoint_pairs=checkpoint_pairs,
            output=self.deployment_path,
            repository_root=self.directory,
        )

    def test_deployment_binds_exact_pending_model_selection_and_runtime(self):
        descriptor = self.create()
        validated = activation.load_deployment(
            self.deployment_path, self.directory
        )
        header, metadata = activation.render_deployment(validated)
        direct, direct_metadata = activation.round2_exporter.render(
            self.fixture.model,
            hashlib.sha256(self.fixture.model_path.read_bytes()).hexdigest(),
            self.sidecar["selected"]["seed"],
        )
        self.assertEqual(header, direct)
        self.assertEqual(
            metadata["packed_sha256"], direct_metadata["packed_sha256"]
        )
        self.assertEqual(
            metadata["selection_payload_sha256"],
            self.sidecar["selection_payload_sha256"],
        )
        self.assertEqual(descriptor["selected_seed"], metadata["training_seed"])
        self.assertIsNone(self.fixture.model["training"]["chosen_seed"])
        self.assertEqual(
            self.fixture.model["training"]["external_actual_clock_selection"][
                "status"
            ],
            "pending",
        )

    def test_import_origins_and_canonical_baseline_are_fail_closed(self):
        with mock.patch.object(
            activation.round2_exporter,
            "__file__",
            str(self.directory / "shadow-exporter.py"),
        ), self.assertRaisesRegex(ImportError, "module origin mismatch"):
            activation._validate_module_origins()

        alternate = self.directory / "models/alternate-bootstrap.json"
        alternate.write_bytes(self.fixture.baseline_model.read_bytes())
        with self.assertRaisesRegex(
            activation.ActivationError, "frozen canonical bootstrap"
        ):
            activation.create_deployment(
                model_path=self.fixture.model_path,
                selection_path=self.sidecar_path,
                runtime_path=self.runtime_path,
                baseline_model=alternate,
                baseline_seed=self.fixture.baseline_seed,
                baseline_runtime=self.fixture.baseline_runtime,
                report_paths=self.paths,
                checkpoint_pairs=(),
                output=self.deployment_path,
                repository_root=self.directory,
            )
        with self.assertRaisesRegex(
            activation.ActivationError, "frozen canonical bootstrap"
        ):
            activation.create_deployment(
                model_path=self.fixture.model_path,
                selection_path=self.sidecar_path,
                runtime_path=self.runtime_path,
                baseline_model=self.fixture.baseline_model,
                baseline_seed=self.fixture.baseline_seed - 1,
                baseline_runtime=self.fixture.baseline_runtime,
                report_paths=self.paths,
                checkpoint_pairs=(),
                output=self.deployment_path,
                repository_root=self.directory,
            )

    def test_exploratory_deployment_is_explicitly_nonpromoting(self):
        self.sidecar_path.unlink()
        self.fixture.reports = (
            self.directory
            / "models/jacek_native_round2_gate_evidence/exploratory"
        )
        paths = self.fixture._all_reports({
            101: {"screen": (260, 260), "decisive": (54, 54)},
            102: {"screen": (263, 262), "decisive": (55, 55)},
        })
        self.sidecar = self.fixture.selection.finalize_exploratory_selection(
            model_path=self.fixture.model_path,
            baseline_model=self.fixture.baseline_model,
            baseline_seed=self.fixture.baseline_seed,
            baseline_runtime=self.fixture.baseline_runtime,
            report_paths=paths,
            output=self.sidecar_path,
        )
        self.runtime_path = self._install_selected_runtime(
            self.sidecar["selected"]["seed"]
        )
        self.paths = paths
        descriptor = self.create()
        self.assertEqual(descriptor["decision"]["kind"], "exploratory")
        self.assertFalse(descriptor["decision"]["promotion_eligible"])
        self.assertTrue(descriptor["decision"]["threshold_shortfalls"])

    def test_rehashed_decision_tamper_is_rejected(self):
        sidecar = json.loads(self.sidecar_path.read_text())
        sidecar["decision"]["promotion_eligible"] = False
        sidecar["selection_payload_sha256"] = (
            self.fixture.selection._selection_payload_hash(sidecar)
        )
        self.sidecar_path.write_bytes(
            self.fixture.selection.canonical_json_bytes(sidecar)
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "deterministic frozen gate evidence"
        ):
            self.create()

    def test_rehashed_selected_seed_and_ranking_tamper_is_rejected(self):
        sidecar = json.loads(self.sidecar_path.read_text())
        rows = sidecar["ranking"]["passing_seeds"]
        rows[0], rows[1] = rows[1], rows[0]
        sidecar["selected"] = {
            **sidecar["selected"],
            **{
                key: value
                for key, value in next(
                    report for report in self.sidecar["reports"]
                    if report["seed"] == rows[0]["seed"]
                ).items()
                if key == "seed"
            },
        }
        sidecar["selection_payload_sha256"] = (
            self.fixture.selection._selection_payload_hash(sidecar)
        )
        self.sidecar_path.write_bytes(
            self.fixture.selection.canonical_json_bytes(sidecar)
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "deterministic frozen gate evidence"
        ):
            self.create()

    def test_native_ancestry_requires_exact_file_backed_checkpoint(self):
        baseline_raw = self.fixture.baseline_model.read_bytes()
        baseline_model = json.loads(baseline_raw)
        runtime = self.fixture.selection.round1_exporter.render_runtime(
            baseline_model,
            hashlib.sha256(baseline_raw).hexdigest(),
            self.fixture.baseline_seed,
        ).encode()
        lines = runtime.decode("utf-8").splitlines()
        identity = {
            "artifact_sha256": hashlib.sha256(runtime).hexdigest(),
            "model_sha256": lines[3],
            "packed_sha256": lines[4],
        }
        generation = self.fixture.model["provenance"]["generation"]
        generation["checkpoint_provenance"] = {
            "mode": "native-runtime-models/v1",
            "artifacts": [identity],
        }
        generation["model_artifact_sha256"] = [identity["artifact_sha256"]]
        self.fixture._write_model()
        self.fixture._write_runtimes()
        self.fixture.reports = (
            self.directory
            / "models/jacek_native_round2_gate_evidence/native"
        )
        self.paths = self.fixture._all_reports()
        self.sidecar_path.unlink()
        self.sidecar = self.fixture.selection.finalize_selection(
            model_path=self.fixture.model_path,
            baseline_model=self.fixture.baseline_model,
            baseline_seed=self.fixture.baseline_seed,
            baseline_runtime=self.fixture.baseline_runtime,
            report_paths=self.paths,
            output=self.sidecar_path,
        )
        self.runtime_path = self._install_selected_runtime(
            self.sidecar["selected"]["seed"]
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "file-backed declarations"
        ):
            self.create()
        descriptor = self.create([(
            self.fixture.baseline_model,
            self.fixture.baseline_runtime,
        )])
        self.assertEqual(len(descriptor["native_checkpoint_provenance"]), 1)
        validated = activation.load_deployment(
            self.deployment_path, self.directory
        )
        self.assertEqual(len(validated["checkpoint_paths"]), 1)

    def test_runtime_or_descriptor_identity_tamper_is_rejected(self):
        self.create()
        self.runtime_path.write_bytes(self.runtime_path.read_bytes() + b"x")
        with self.assertRaisesRegex(
            activation.ActivationError, "runtime bytes are stale"
        ):
            activation.load_deployment(self.deployment_path, self.directory)

    def test_archived_gate_evidence_tamper_is_rejected(self):
        self.create()
        self.paths[0].write_bytes(self.paths[0].read_bytes() + b"x")
        with self.assertRaisesRegex(
            activation.ActivationError, "gate report bytes are stale"
        ):
            activation.load_deployment(self.deployment_path, self.directory)

    def test_descriptor_refuses_path_escape_and_overwrite(self):
        self.create()
        with self.assertRaisesRegex(activation.ActivationError, "overwrite"):
            self.create()
        descriptor = json.loads(self.deployment_path.read_text())
        descriptor["model"]["path"] = "../outside.json"
        self.deployment_path.write_bytes(
            activation._canonical_json_bytes(descriptor)
        )
        with self.assertRaisesRegex(activation.ActivationError, "escapes"):
            activation.load_deployment(self.deployment_path, self.directory)


if __name__ == "__main__":
    unittest.main()
