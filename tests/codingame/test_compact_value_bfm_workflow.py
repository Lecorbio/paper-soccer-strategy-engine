from __future__ import annotations

import contextlib
import io
import importlib.util
import inspect
import json
import pathlib
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "compact_value_bfm_workflow.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


class CompactValueBfmWorkflowTest(unittest.TestCase):
    def make_bundle(self, root: pathlib.Path) -> pathlib.Path:
        bundle = root / "input-bundle"
        bundle.mkdir()
        routes = {
            "canonical_prior_manifests": [],
            "opening_exclusions": [],
            "pilot_search_manifests": [],
            "full_search_manifests": [],
            "pilot_rank4_manifests": [],
            "full_rank4_manifests": [],
        }
        artifacts = []

        def add(role: str, relative: str) -> str:
            path = bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{role}\n".encode()
            path.write_bytes(payload)
            artifacts.append(workflow.ImportArtifact(
                role=role,
                source=path,
                relative_path=relative,
                sha256=workflow.sha256_bytes(payload),
                bytes=len(payload),
            ))
            return relative

        for index in range(9):
            routes["canonical_prior_manifests"].append(
                add(f"canonical-{index}", f"canonical/{index}.json")
            )
        for index in range(7):
            routes["opening_exclusions"].append(
                add(f"exclusion-{index}", f"openings/{index}.tsv")
            )
        for key in (
            "pilot_search_manifests", "full_search_manifests",
            "pilot_rank4_manifests", "full_rank4_manifests",
        ):
            for index in range(3):
                routes[key].append(add(f"{key}-{index}", f"new/{key}/{index}.json"))
        policy = {
            "source_campaign_id": workflow.SOURCE_CAMPAIGN_ID,
            "explicit_allowlist": True,
            "source_campaign_scanned": False,
            "protected_path_markers_rejected_before_access": True,
            "sealed_final_accessed": False,
            "blind_labels_accessed": False,
            "runtime_uses_source_paths": False,
            "git_required_after_freeze": False,
            "protected_tests_locked": True,
            "external_upload": False,
            "replace_rank4": False,
            "rank1_claim": False,
        }
        manifest = workflow._bundle_manifest(artifacts, routes, policy)
        manifest_path = bundle / "bundle-manifest.json"
        manifest_path.write_bytes(workflow.canonical_json_bytes(manifest))
        return manifest_path

    def test_forbidden_marker_is_rejected_before_resolve_or_stat(self):
        with mock.patch.object(
            pathlib.Path, "resolve", side_effect=AssertionError("resolved decoy")
        ):
            with self.assertRaisesRegex(workflow.WorkflowError, "protected path"):
                workflow.checked_root("/tmp/sealed_final/decoy.tsv")
            with self.assertRaisesRegex(workflow.WorkflowError, "protected path"):
                workflow.checked_root("/tmp/blind-label/decoy.tsv")

    def test_bundle_path_rejects_absolute_traversal_and_markers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            for value in ("../escape", "/absolute", "x/sealed-final.tsv",
                          "x/blind_label.tsv"):
                with self.subTest(value=value):
                    with self.assertRaises(workflow.WorkflowError):
                        workflow.safe_bundle_path(root, value, "test route")

    def test_verify_is_git_and_old_worktree_independent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = self.make_bundle(root)
            with mock.patch("subprocess.run", side_effect=AssertionError("Git used")):
                verified = workflow.verify_bundle(manifest)
            self.assertEqual(verified["campaign_id"], workflow.CAMPAIGN_ID)
            self.assertFalse(verified["policy"]["runtime_uses_source_paths"])
            self.assertTrue(verified["policy"]["protected_tests_locked"])

    def test_tampered_or_unregistered_bundle_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = self.make_bundle(root)
            target = manifest.parent / "canonical/0.json"
            target.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(workflow.WorkflowError, "changed"):
                workflow.verify_bundle(manifest)
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = self.make_bundle(root)
            (manifest.parent / "decoy.txt").write_text("unregistered")
            with self.assertRaisesRegex(workflow.WorkflowError, "unregistered"):
                workflow.verify_bundle(manifest)

    def test_frozen_counts_and_exact_order_are_constant(self):
        self.assertEqual(
            workflow.EXPECTED_NEW_ROWS,
            {"train": 241_365, "validation": 29_418, "test": 30_706},
        )
        self.assertEqual(sum(workflow.EXPECTED_CANONICAL_ROWS.values()), 1_228_970)
        self.assertEqual(len(workflow.CANONICAL_MANIFEST_SHA256), 9)
        self.assertEqual(
            workflow.CANONICAL_SPLITS,
            ("train", "validation", "test") * 3,
        )
        self.assertEqual(workflow.EXPECTED_ADJUDICATOR_ROWS, 8_000)

    def test_source_import_never_enumerates_the_campaign(self):
        source = inspect.getsource(workflow.collect_imports)
        for forbidden in (".glob(", ".rglob(", ".iterdir(", "os.walk("):
            self.assertNotIn(forbidden, source)

    def test_body_hash_and_manifest_tampering_fail(self):
        body = {"schema": "fixture", "value": 1}
        digest = workflow.sha256_bytes(workflow.canonical_json_bytes(body))
        value = dict(body, body_sha256=digest)
        workflow.verify_body_hash(value, expected=digest, label="fixture")
        value["value"] = 2
        with self.assertRaisesRegex(workflow.WorkflowError, "body SHA-256"):
            workflow.verify_body_hash(value, expected=digest, label="fixture")

    def source_sizes(self):
        return {
            architecture: {
                "architecture": architecture,
                "complete_source_ascii_characters": size,
                "eligible": True,
                "limit": 95_000,
            }
            for architecture, size in (
                ("compact-8x8", 81_434),
                ("source-neutral-8x16", 81_482),
                ("capacity-12x8", 94_453),
            )
        }

    def test_source_measurement_uses_exporters_without_overwriting_active_files(self):
        components = workflow._lazy_runtime_components()
        before = workflow._active_output_snapshots()
        with tempfile.TemporaryDirectory() as raw:
            measured = workflow.measure_architecture_source_sizes(
                components, "1" * 64, pathlib.Path(raw)
            )
        self.assertEqual(tuple(measured), workflow.ARCHITECTURE_ORDER)
        self.assertTrue(all(row["eligible"] for row in measured.values()))
        self.assertEqual(before, workflow._active_output_snapshots())
        self.assertLessEqual(
            measured["capacity-12x8"]["complete_source_ascii_characters"],
            95_000,
        )

    def test_family_roster_is_six_deployable_candidates_and_one_control(self):
        specs = workflow.family_campaign_specs(self.source_sizes())
        self.assertEqual(len(specs), 7)
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "compact-8x8--search-target",
                "compact-8x8--teacher-assisted",
                "source-neutral-8x16--search-target",
                "source-neutral-8x16--teacher-assisted",
                "capacity-12x8--search-target",
                "capacity-12x8--teacher-assisted",
                "compact-8x8--rank4-control",
            ],
        )
        rejected = self.source_sizes()
        rejected["capacity-12x8"] = {
            **rejected["capacity-12x8"], "eligible": False,
        }
        self.assertNotIn(
            "capacity-12x8--search-target",
            [spec.name for spec in workflow.family_campaign_specs(rejected)],
        )

    def test_worker_executor_caps_at_two_and_preserves_spec_order(self):
        specs = workflow.family_campaign_specs(self.source_sizes())
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def runner(spec):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.01)
            with lock:
                state["active"] -= 1
            return {"name": spec.name}

        results = workflow._execute_campaigns(
            specs, workers=2, runner=runner
        )
        self.assertLessEqual(state["maximum"], 2)
        self.assertEqual(
            [result["name"] for result in results],
            [spec.name for spec in specs],
        )
        with self.assertRaisesRegex(workflow.WorkflowError, "one or two"):
            workflow._execute_campaigns(specs, workers=3, runner=runner)

    def test_training_subprocess_is_single_threaded_and_resume_aware(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output = root / "run"
            spec = workflow.CampaignSpec(
                "compact-8x8", "teacher-assisted", 81_434
            )
            selection = (
                output / "campaigns" / spec.name / "selections" /
                "selection.json"
            )
            selection.parent.mkdir(parents=True)
            selection.write_text("fixture")
            completed = types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"selection": str(selection)}),
                stderr="",
            )
            with mock.patch.object(
                workflow.subprocess, "run", return_value=completed
            ) as run:
                result = workflow._run_training_process(
                    spec,
                    bundle_manifest=root / "bundle.json",
                    output_directory=output,
                    input_audit=root / "audit.json",
                    sidecar_index=root / "sidecars.json",
                    resume=True,
                )
            arguments = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertIn("--resume", arguments)
            self.assertIn("--sidecar-index", arguments)
            for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            ):
                self.assertEqual(environment[name], "1")
            self.assertEqual(result["selection_path"], selection.resolve())

    def test_mocked_run_writes_stable_hashed_receipt_and_requires_resume(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            bundle_manifest = root / "bundle-manifest.json"
            bundle_manifest.write_text("{}\n")
            output = root / "output"
            fake_bundle = types.SimpleNamespace(
                body_sha256="b" * 64,
                manifest={"campaign_id": workflow.CAMPAIGN_ID},
            )
            frozen_bundle = types.SimpleNamespace(
                load=mock.Mock(return_value=fake_bundle)
            )
            components = types.SimpleNamespace(
                trainer=types.SimpleNamespace(FrozenBundle=frozen_bundle)
            )
            prerequisite = {
                "input_audit_path": root / "audit.json",
                "sidecar_index_path": root / "sidecars.json",
                "input_audit_reference": {"body_sha256": "a" * 64},
                "sidecar_reference": {"body_sha256": "c" * 64},
            }
            tools = {"fixture": {"sha256": "d" * 64}}
            specs = workflow.family_campaign_specs(self.source_sizes())

            def execute(current, **_kwargs):
                return [{"spec": spec} for spec in current]

            def campaign(_components, _bundle, _output, process):
                spec = process["spec"]
                return {
                    "name": spec.name,
                    "campaign_output": f"campaigns/{spec.name}",
                    "protected_tests_opened": False,
                }

            with (
                mock.patch.object(workflow, "_lazy_runtime_components", return_value=components),
                mock.patch.object(workflow, "_tool_bindings", return_value=tools),
                mock.patch.object(workflow, "_ensure_run_prerequisites", return_value=prerequisite),
                mock.patch.object(workflow, "measure_architecture_source_sizes", return_value=self.source_sizes()),
                mock.patch.object(workflow, "_execute_campaigns", side_effect=execute),
                mock.patch.object(workflow, "_campaign_result", side_effect=campaign),
                mock.patch.object(workflow, "verify_family_run", return_value={"verified": True}),
            ):
                result = workflow.run_family(
                    bundle_manifest, output, workers=2
                )
                self.assertEqual(result, {"verified": True})
                start = json.loads(
                    (output / "run-state/run-start.json").read_text()
                )
                reference = json.loads(
                    (output / "run-state/run-reference.json").read_text()
                )
                workflow._verify_hashed_document(
                    start, schema=workflow.RUN_START_SCHEMA, label="start"
                )
                workflow._verify_hashed_document(
                    reference,
                    schema=workflow.RUN_REFERENCE_SCHEMA,
                    label="reference",
                )
                self.assertFalse(reference["protected_tests_opened"])
                with self.assertRaisesRegex(
                    workflow.WorkflowError, "requires --resume"
                ):
                    workflow.run_family(bundle_manifest, output)
                self.assertEqual(
                    workflow.run_family(
                        bundle_manifest, output, resume=True
                    ),
                    {"verified": True},
                )

    def test_verify_cli_optionally_validates_run_reference(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            with (
                mock.patch.object(workflow, "verify_bundle", return_value={"bundle": True}),
                mock.patch.object(workflow, "verify_family_run", return_value={"run": True}) as verify_run,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                status = workflow.main([
                    "verify",
                    "--bundle-manifest", str(root / "bundle.json"),
                    "--run-output-directory", str(root / "run"),
                    "--run-reference", str(root / "run/run-state/run-reference.json"),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(output.getvalue()),
                {"bundle": {"bundle": True}, "run": {"run": True}},
            )
            verify_run.assert_called_once()

    def test_run_source_has_no_git_or_old_worktree_dependency(self):
        source = inspect.getsource(workflow.run_family)
        self.assertNotIn("git", source.lower())
        self.assertNotIn("bb37", source)


if __name__ == "__main__":
    unittest.main()
