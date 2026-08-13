import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
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

    def _untrained_seed_identity(self):
        runtime = (
            self.directory / "models/jacek_native_untrained_seed.runtime"
        ).read_bytes()
        lines = runtime.decode("utf-8").splitlines()
        return {
            "artifact_sha256": hashlib.sha256(runtime).hexdigest(),
            "model_sha256": lines[3],
            "packed_sha256": lines[4],
        }

    def _install_native_baseline(self):
        active_seeds = self.fixture.seeds
        try:
            self.fixture.seeds = [201, 202]
            parent = self.fixture._model()
        finally:
            self.fixture.seeds = active_seeds
        seed_identity = self._untrained_seed_identity()
        parent_generation = parent["provenance"]["generation"]
        parent_generation["checkpoint_provenance"] = {
            "mode": "untrained-seed-bootstrap/v1",
            "artifacts": [seed_identity],
        }
        parent_generation["model_artifact_sha256"] = [
            seed_identity["artifact_sha256"]
        ]
        parent_model = self.directory / "models/native-parent.json"
        parent_model.write_bytes(
            self.fixture.selection.canonical_json_bytes(parent)
        )
        parent_seed = 201
        parent_runtime = self.directory / "models/native-parent.runtime"
        parent_runtime.write_text(
            self.fixture.selection.round2_exporter.render_runtime(
                parent,
                hashlib.sha256(parent_model.read_bytes()).hexdigest(),
                parent_seed,
            ),
            encoding="utf-8",
        )
        lines = parent_runtime.read_text(encoding="utf-8").splitlines()
        parent_identity = {
            "artifact_sha256": hashlib.sha256(
                parent_runtime.read_bytes()
            ).hexdigest(),
            "model_sha256": lines[3],
            "packed_sha256": lines[4],
        }
        generation = self.fixture.model["provenance"]["generation"]
        generation["checkpoint_provenance"] = {
            "mode": "native-runtime-models/v1",
            "artifacts": [parent_identity],
        }
        generation["model_artifact_sha256"] = [
            parent_identity["artifact_sha256"]
        ]
        self.fixture._write_model()
        self.fixture._write_runtimes()
        self.fixture.baseline_model = parent_model
        self.fixture.baseline_seed = parent_seed
        self.fixture.baseline_runtime = parent_runtime
        self.fixture.reports = (
            self.directory
            / "models/jacek_native_round2_gate_evidence/native-baseline"
        )
        self.paths = self.fixture._all_reports()
        self.sidecar_path.unlink()
        self.sidecar = self.fixture.selection.finalize_selection(
            model_path=self.fixture.model_path,
            baseline_model=parent_model,
            baseline_seed=parent_seed,
            baseline_runtime=parent_runtime,
            report_paths=self.paths,
            output=self.sidecar_path,
        )
        self.runtime_path = self._install_selected_runtime(
            self.sidecar["selected"]["seed"]
        )
        return parent_model, parent_seed, parent_runtime

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

    def test_retained_native_baseline_is_recursively_bound(self):
        parent_model, parent_seed, parent_runtime = (
            self._install_native_baseline()
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "file-backed declarations"
        ):
            self.create()
        descriptor = self.create([(parent_model, parent_runtime)])
        self.assertEqual(descriptor["baseline"]["seed"], parent_seed)
        self.assertEqual(descriptor["baseline"]["exporter"]["kind"], "round2")
        self.assertEqual(
            descriptor["baseline"]["checkpoint_sha256"],
            self.fixture.selection._baseline_identity(
                parent_model, parent_seed, parent_runtime
            )["checkpoint_sha256"],
        )
        loaded = activation.load_deployment(
            self.deployment_path, self.directory
        )
        self.assertEqual(
            loaded["baseline_model_path"], parent_model.resolve()
        )

        tampered = json.loads(self.deployment_path.read_text())
        tampered["baseline"]["seed"] = parent_seed + 1
        self.deployment_path.write_bytes(
            activation._canonical_json_bytes(tampered)
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "exact retained native checkpoint"
        ):
            activation.load_deployment(self.deployment_path, self.directory)

    def test_recursive_checkpoint_cycle_is_rejected(self):
        identity_a = {
            "artifact_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "packed_sha256": "c" * 64,
        }
        identity_b = {
            "artifact_sha256": "d" * 64,
            "model_sha256": "e" * 64,
            "packed_sha256": "f" * 64,
        }

        def model_with_parent(parent):
            return {"provenance": {"generation": {
                "checkpoint_provenance": {
                    "mode": "native-runtime-models/v1",
                    "artifacts": [parent],
                },
                "model_artifact_sha256": [parent["artifact_sha256"]],
            }}}

        model_a = self.directory / "models/cycle-a.json"
        runtime_a = self.directory / "models/cycle-a.runtime"
        model_b = self.directory / "models/cycle-b.json"
        runtime_b = self.directory / "models/cycle-b.runtime"
        model_a.write_bytes(b"cycle-a\n")
        model_b.write_bytes(b"cycle-b\n")
        metadata = {
            model_a.resolve(): {
                "identity": identity_a,
                "model": model_with_parent(identity_b),
            },
            model_b.resolve(): {
                "identity": identity_b,
                "model": model_with_parent(identity_a),
            },
        }

        def checkpoint_identity(model_path, _runtime_path, *_retained_evidence):
            return {
                **metadata[model_path.resolve()],
                "seed": 1,
                "checkpoint_sha256": "1" * 64,
                "exporter": "round2",
                "exporter_sha256": "2" * 64,
            }

        active = model_with_parent(identity_a)
        with (
            mock.patch.object(
                activation, "_untrained_seed_identity",
                return_value={
                    "artifact_sha256": "0" * 64,
                    "model_sha256": "1" * 64,
                    "packed_sha256": "2" * 64,
                },
            ),
            mock.patch.object(
                activation, "_checkpoint_model_identity",
                side_effect=checkpoint_identity,
            ),
            self.assertRaisesRegex(
                activation.ActivationError, "contains a cycle"
            ),
        ):
            activation._validate_checkpoint_ancestry(
                self.directory,
                self.fixture.model_path,
                active,
                [(model_a, runtime_a), (model_b, runtime_b)],
            )

    def test_historical_deployed_checkpoint_identity_is_recognized_exactly(self):
        model = self.directory / "models/historical-native.json"
        runtime = self.directory / "models/historical-native.runtime"
        selection = self.directory / "models/historical-selection.json"
        deployment = self.directory / "models/historical-deployment.json"
        model.write_bytes(
            (ROOT / "models/jacek_native_history62_champion.json").read_bytes()
        )
        runtime.write_bytes(
            (ROOT / "models/jacek_native_history62_champion.runtime").read_bytes()
        )
        selection.write_bytes(
            (ROOT / "models/jacek_native_history62_selection.json").read_bytes()
        )
        deployment.write_bytes(
            (ROOT / "models/jacek_native_history62_deployment.json").read_bytes()
        )
        identity = activation._checkpoint_model_identity(
            model, runtime, selection, deployment
        )
        self.assertEqual(identity["seed"], 20260822)
        self.assertEqual(identity["exporter"], "round2")
        self.assertEqual(
            identity["identity"]["artifact_sha256"],
            "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3",
        )
        runtime.write_bytes(runtime.read_bytes() + b"x")
        with self.assertRaisesRegex(
            activation.ActivationError, "exact unique retained model export"
        ):
            activation._checkpoint_model_identity(
                model, runtime, selection, deployment
            )

        runtime.write_bytes(
            (ROOT / "models/jacek_native_round2_selected.runtime").read_bytes()
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "exact unique retained model export"
        ):
            activation._checkpoint_model_identity(model, runtime)

        _, historical_model = activation.selection_tool._load_round2_model(model)
        sibling_runtime = self.directory / "models/historical-seed23.runtime"
        sibling_runtime.write_bytes(
            activation.selection_tool._render_historical_round2_runtime(
                historical_model,
                hashlib.sha256(model.read_bytes()).hexdigest(),
                20260823,
            )
        )
        sibling = activation._checkpoint_model_identity(
            model, sibling_runtime, selection, deployment
        )
        self.assertEqual(sibling["seed"], 20260823)
        self.assertEqual(
            sibling["checkpoint_sha256"],
            "2db8a2d469efb98d8746dc6c694ba8f31bfbd40f57c3944d9480971148d9602f",
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "exact retained native checkpoint"
        ):
            activation._prevalidate_deployment_baseline(
                self.directory,
                model,
                20260823,
                sibling_runtime,
                selection,
                deployment,
            )

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

    def test_retained_baseline_descriptors_must_stay_under_models(self):
        self.create()
        outside_selection = self.directory / "outside-selection.json"
        outside_deployment = self.directory / "outside-deployment.json"
        outside_selection.write_bytes(self.sidecar_path.read_bytes())
        outside_deployment.write_bytes(self.deployment_path.read_bytes())
        descriptor = json.loads(self.deployment_path.read_text())
        descriptor["baseline"]["retained_evidence"] = {
            "selection": activation._file_identity(
                self.directory, outside_selection, "outside selection"
            ),
            "deployment": activation._file_identity(
                self.directory, outside_deployment, "outside deployment"
            ),
        }
        self.deployment_path.write_bytes(
            activation._canonical_json_bytes(descriptor)
        )
        with self.assertRaisesRegex(
            activation.ActivationError, "must be installed under models"
        ):
            activation.load_deployment(self.deployment_path, self.directory)

    def test_install_is_atomic_compare_and_swap_and_supports_rollback(self):
        self.create()
        candidate = self.directory / "models/iteration-deployment.json"
        candidate.write_bytes(self.deployment_path.read_bytes())
        canonical = (
            self.directory / "models/jacek_native_round2_deployment.json"
        )
        original = b'{"old":"pointer"}\n'
        canonical.write_bytes(original)
        old_sha = hashlib.sha256(original).hexdigest()
        with self.assertRaisesRegex(
            activation.ActivationError, "changed before install"
        ):
            activation.install_deployment(
                candidate, "0" * 64, canonical, self.directory
            )
        self.assertEqual(canonical.read_bytes(), original)

        installed_sha = activation.install_deployment(
            candidate, old_sha, canonical, self.directory
        )
        self.assertEqual(canonical.read_bytes(), candidate.read_bytes())
        self.assertEqual(
            installed_sha, hashlib.sha256(candidate.read_bytes()).hexdigest()
        )

        rollback = self.directory / "models/rollback-deployment.json"
        rollback.write_bytes(candidate.read_bytes())
        current_sha = hashlib.sha256(canonical.read_bytes()).hexdigest()
        activation.install_deployment(
            rollback, current_sha, canonical, self.directory
        )
        self.assertEqual(canonical.read_bytes(), rollback.read_bytes())
        self.assertEqual(
            list(canonical.parent.glob(".jacek_native_round2_deployment.json.*.install")),
            [],
        )

    def test_install_can_compare_and_swap_back_to_distinct_validated_bytes(self):
        models = self.directory / "models"
        canonical = models / "jacek_native_round2_deployment.json"
        previous = models / "previous-deployment.json"
        candidate = models / "candidate-deployment.json"
        previous.write_bytes(b'{"generation":"previous"}\n')
        candidate.write_bytes(b'{"generation":"candidate"}\n')
        canonical.write_bytes(previous.read_bytes())

        def validated(path, _root):
            return {"deployment_bytes": pathlib.Path(path).read_bytes()}

        with mock.patch.object(
            activation, "load_deployment", side_effect=validated
        ):
            activation.install_deployment(
                candidate,
                hashlib.sha256(previous.read_bytes()).hexdigest(),
                canonical,
                self.directory,
            )
            self.assertEqual(canonical.read_bytes(), candidate.read_bytes())
            activation.install_deployment(
                previous,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
                canonical,
                self.directory,
            )
        self.assertEqual(canonical.read_bytes(), previous.read_bytes())


class JacekNativeHistoricalActiveCompatibilityTest(unittest.TestCase):
    REPORTS = tuple(
        ROOT / "models/jacek_native_round2_gate_evidence" / name
        for name in (
            "b44c1cec78c5c86421ea693af329662451ac9301665dd8fd3db0997721e3854a.json",
            "2baee0a80f18b045357aef2686d58bacc89a1cf147e3699b1538be3e6c788ee2.json",
            "28138eab2218e7a574dc4b7379fd19f8219a5efdbca5f107f507540f0761a98d.json",
            "41c8dc7c19f3ccacaf74de7e318fee129a4eedc7735ce5d0d2e940f2e3e9a983.json",
            "93ac04d0d020015d270738785b75225960d88280b4a4c62bd438d47f549db938.json",
            "662122aba29156f3400c34f0b4dc4b25068c9f05b5027583f8eb8efdbfe73f19.json",
        )
    )

    def validate(self, reports=None, baseline_seed=20260813):
        return activation.validate_selection(
            ROOT / "models/jacek_native_history62_champion.json",
            ROOT / "models/jacek_native_history62_selection.json",
            baseline_model=ROOT / "models/jacek_native_bootstrap_model.json",
            baseline_seed=baseline_seed,
            baseline_runtime=(
                ROOT
                / f"models/jacek_native_bootstrap_seed_{baseline_seed}.runtime"
            ),
            report_paths=self.REPORTS if reports is None else reports,
            repository_root=ROOT,
        )

    def copy_archive(self, root):
        relatives = [
            "models/jacek_native_history62_champion.json",
            "models/jacek_native_history62_champion.runtime",
            "models/jacek_native_history62_selection.json",
            "models/jacek_native_history62_deployment.json",
            "models/jacek_native_bootstrap_model.json",
            "models/jacek_native_bootstrap_seed_20260811.runtime",
            "models/jacek_native_bootstrap_seed_20260812.runtime",
            "models/jacek_native_bootstrap_seed_20260813.runtime",
            "models/jacek_native_untrained_seed.json",
            "models/jacek_native_untrained_seed.runtime",
            "tools/generate_jacek_native_seed.py",
        ]
        for report in self.REPORTS:
            relative = report.relative_to(ROOT)
            relatives.append(relative.as_posix())
            payload = json.loads(report.read_text())
            relatives.append((relative.parent / payload["stdout"]["path"]).as_posix())
        for relative in relatives:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def test_exact_history62_archive_renders_the_frozen_header_and_runtime(self):
        validated = self.validate()
        validated["deployment_sha256"] = "0" * 64
        header, metadata = activation.render_deployment(validated)
        self.assertTrue(validated["historical_active"])
        self.assertEqual(
            hashlib.sha256(validated["runtime_bytes"]).hexdigest(),
            activation.HISTORICAL_ACTIVE_RUNTIME_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(header.encode()).hexdigest(),
            activation.HISTORICAL_ACTIVE_HEADER_SHA256,
        )
        self.assertEqual(metadata["training_seed"], 20260822)

    def test_exact_archive_creates_and_reloads_one_v2_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.copy_archive(root)
            reports = [root / path.relative_to(ROOT) for path in self.REPORTS]
            model = root / "models/jacek_native_history62_champion.json"
            bootstrap = root / "models/jacek_native_bootstrap_model.json"
            output = root / "models/history62-reactivation-v2.json"
            descriptor = activation.create_deployment(
                model_path=model,
                selection_path=root / "models/jacek_native_history62_selection.json",
                runtime_path=root / "models/jacek_native_history62_champion.runtime",
                baseline_model=bootstrap,
                baseline_seed=20260813,
                baseline_runtime=(
                    root / "models/jacek_native_bootstrap_seed_20260813.runtime"
                ),
                report_paths=reports,
                checkpoint_pairs=[
                    (
                        bootstrap,
                        root / f"models/jacek_native_bootstrap_seed_{seed}.runtime",
                    )
                    for seed in (20260811, 20260812, 20260813)
                ],
                output=output,
                repository_root=root,
            )
            self.assertEqual(descriptor["selected_seed"], 20260822)
            loaded = activation.load_deployment(output, root)
            self.assertTrue(loaded["historical_active"])
            self.assertEqual(len(loaded["checkpoint_paths"]), 3)

    def test_historical_active_path_rejects_noncanonical_baseline_and_gate_set(self):
        with self.assertRaisesRegex(
            activation.ActivationError, "canonical bootstrap baseline"
        ):
            self.validate(baseline_seed=20260812)
        with self.assertRaisesRegex(
            activation.ActivationError, "(report set|incomplete)"
        ):
            self.validate(reports=self.REPORTS[:-1])

    def test_historical_active_render_rejects_runtime_identity_tamper(self):
        validated = self.validate()
        validated["deployment_sha256"] = "0" * 64
        validated["runtime_bytes"] += b"x"
        with self.assertRaisesRegex(
            activation.ActivationError, "render identity is stale"
        ):
            activation.render_deployment(validated)

    def test_arbitrary_legacy_model_is_not_admitted_as_historical_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = json.loads(
                (ROOT / "models/jacek_native_history62_champion.json").read_text()
            )
            model["provenance"]["trainer_sha256"] = "0" * 64
            path = pathlib.Path(temporary) / "legacy.json"
            path.write_bytes(activation._canonical_json_bytes(model))
            with self.assertRaises(activation.ActivationError):
                activation.validate_selection(
                    path,
                    ROOT / "models/jacek_native_history62_selection.json",
                    baseline_model=ROOT / "models/jacek_native_bootstrap_model.json",
                    baseline_seed=20260813,
                    baseline_runtime=(
                        ROOT / "models/jacek_native_bootstrap_seed_20260813.runtime"
                    ),
                    report_paths=self.REPORTS,
                    repository_root=ROOT,
                )

    def test_exact_predecessor_tool_hashes_are_admitted_but_unknowns_fail(self):
        deployment = (
            ROOT
            / "models/jacek_native_round3_rank1_2026083111_deployment.json"
        )
        # The archived round-three gate correctly becomes stale when bot.cpp's
        # search constants change.  Isolate this test to the predecessor-tool
        # compatibility boundary instead of bypassing that deeper provenance
        # check or pretending the old gate was executed by the current bot.
        with mock.patch.object(
            activation,
            "_validate_file_identity",
            side_effect=activation.ActivationError(
                "past activation-tool compatibility check"
            ),
        ), self.assertRaisesRegex(
            activation.ActivationError,
            "past activation-tool compatibility check",
        ):
            activation.load_deployment(deployment, ROOT)
        descriptor = json.loads(deployment.read_text())
        descriptor["activation_tool_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            tampered = pathlib.Path(temporary) / "tampered-deployment.json"
            tampered.write_bytes(activation._canonical_json_bytes(descriptor))
            with self.assertRaisesRegex(
                activation.ActivationError, "activation-tool SHA-256 is stale"
            ):
                activation.load_deployment(tampered, ROOT)
        descriptor = json.loads(deployment.read_text())
        descriptor["round2_exporter_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            tampered = pathlib.Path(temporary) / "tampered-exporter.json"
            tampered.write_bytes(activation._canonical_json_bytes(descriptor))
            with self.assertRaisesRegex(
                activation.ActivationError,
                "selector/exporter identity is stale",
            ):
                activation.load_deployment(tampered, ROOT)


if __name__ == "__main__":
    unittest.main()
