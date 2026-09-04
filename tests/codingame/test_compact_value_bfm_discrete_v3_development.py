import base64
import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock


from submissions.codingame.bots.compact_value_bfm import (
    discrete_v3_development_runner as v3runner,
)
from submissions.codingame.bots.compact_value_bfm import (
    test_development_runner as base_test,
)
from tools import compact_value_bfm_discrete_v3_development as development


q = development.qualification
maintained = v3runner.maintained


def runtime(root, name, hidden_one, hidden_two, arm):
    counts = {
        "w1": 6301 * hidden_one,
        "w2": hidden_one * hidden_two,
        "w3": hidden_two,
    }
    counts["total"] = sum(counts.values())
    payload = bytes((counts["total"] * 3 + 7) // 8)
    document = maintained.body_hashed({
        "schema": maintained.export_model.RUNTIME_SCHEMA,
        "feature_schema": maintained.export_model.FEATURE_SCHEMA,
        "architecture": {
            "name": name,
            "dimensions": [6301, hidden_one, hidden_two, 1],
            "biases": False,
            "activations": maintained.export_model.ACTIVATIONS,
            "payload_layout": maintained.export_model.LAYOUT,
        },
        "quantization": {
            **maintained.export_model.QUANTIZATION,
            "scales": {"w1": 0.125, "w2": 0.125, "w3": 0.125},
            "weight_counts": counts,
            "packed_byte_count": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        },
        "selection": {
            "arm": arm, "seed": 20260907,
            "float_epoch": 1, "qat_epoch": 0,
            "source_bundle_body_sha256": "1" * 64,
        },
    })
    raw = maintained.canonical_json_bytes(document)
    path = root / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def sealed(path, schema, **body):
    q.write_sealed(path, {"schema": schema, **body})
    return path


class SyntheticContext:
    def __init__(self, root):
        self.root = root.resolve()
        self.banks = base_test.make_banks(self.root)
        self.compiler = {
            "command": "injected", "executable": "injected",
            "version_sha256": "0" * 64,
        }
        self.candidate_runtime = runtime(
            self.root / "candidate", "capacity-12x8", 12, 8, "search-target"
        )
        self.candidate_source = self.root / "candidate/submission.cpp"
        self.candidate_source.write_bytes(b"int main(){return 0;}\n")
        runtime_document = q.load_sealed(self.candidate_runtime)
        self.candidate_selection = sealed(
            self.root / "candidate/selection.json",
            development.adapter.v3.SELECTION_SCHEMA,
            namespace=development.NAMESPACE,
            architecture="capacity-12x8",
            source_export={
                "runtime_sha256": q.sha256_file(self.candidate_runtime),
                "runtime_body_sha256": runtime_document["body_sha256"],
                "model_header_sha256": "2" * 64,
                "source_sha256": q.sha256_file(self.candidate_source),
                "source_ascii_bytes": self.candidate_source.stat().st_size,
                "source_limit_exclusive": 95_000,
            },
        )
        candidate_selection_document = q.load_sealed(self.candidate_selection)
        self.adapter_plan_path = self.root / "adapter-plan-v2.json"
        sealed(
            self.adapter_plan_path, development.adapter.ADAPTER_PLAN_SCHEMA,
            namespace=development.NAMESPACE,
        )
        diagnostic = sealed(
            self.root / "diagnostic.json", development.adapter.ADAPTER_REPORT_SCHEMA,
            namespace=development.NAMESPACE,
        )
        diagnostic_reference = sealed(
            self.root / "diagnostic-reference.json",
            development.adapter.ADAPTER_REPORT_REFERENCE_SCHEMA,
            namespace=development.NAMESPACE,
        )
        claim = sealed(
            self.root / "claim.json", development.adapter.EVALUATION_CLAIM_SCHEMA,
            namespace=development.NAMESPACE,
        )
        completion = sealed(
            self.root / "completion.json",
            development.adapter.EVALUATION_COMPLETION_SCHEMA,
            namespace=development.NAMESPACE,
        )
        candidate = {
            "candidate_id": development.CANDIDATE_ID,
            "architecture": development.CAPACITY_ARCHITECTURE,
            "runtime_architecture": "capacity-12x8",
            "dimensions": [6301, 12, 8, 1],
            "target": "search-target",
            "selection": q.artifact_reference(self.candidate_selection),
            "runtime": development._regular(self.candidate_runtime),
            "generated_source": development._regular(self.candidate_source),
            "source_export": dict(candidate_selection_document["source_export"]),
        }
        self.handoff_path = self.root / "adapter-handoff-v2.json"
        sealed(
            self.handoff_path, development.adapter.HANDOFF_SCHEMA,
            namespace=development.NAMESPACE,
            campaign_id=development.CAMPAIGN_ID,
            adapter_plan=q.artifact_reference(self.adapter_plan_path),
            evaluation_claim=q.artifact_reference(self.root / "claim.json"),
            evaluation_completion=q.artifact_reference(self.root / "completion.json"),
            fresh_report_reference=q.artifact_reference(
                self.root / "diagnostic-reference.json"
            ),
            fresh_report=q.artifact_reference(self.root / "diagnostic.json"),
            v3_selection=q.artifact_reference(self.candidate_selection),
            candidate=candidate,
            development_contract={
                "mode": "discrete-v3-post-holdout",
                "required_output_schema": development.FINALIST_SCHEMA,
                "fresh_position_symmetry_exclusion_audit": {
                    "evidence_schema": development.exclusions.RECEIPT_SCHEMA,
                },
            },
            policy={
                "fresh_protected_tests_opened": True,
                "old_protected_tests_accessed": False,
                "development_screen_required": True,
                "development_selected": False,
                "rank4_final_bank_generation_authorized": False,
                "rank4_gate_authorized": False,
                "upload_authorized": False,
            },
        )
        self.handoff = q.load_sealed(self.handoff_path)
        self.adapter_context = {
            "handoff": self.handoff,
            "handoff_path": self.handoff_path,
            "adapter_plan_path": self.adapter_plan_path,
            "adapter_plan": q.load_sealed(self.adapter_plan_path),
            "candidate": candidate,
        }

        self.exclusion_plan_path = self.root / "exclusion-plan.json"
        sealed(
            self.exclusion_plan_path, development.exclusions.PLAN_SCHEMA,
            namespace=development.NAMESPACE,
        )
        self.exclusion_plan = q.load_sealed(self.exclusion_plan_path)
        fingerprint = sealed(
            self.root / "fingerprints.json", development.exclusions.FINGERPRINT_SCHEMA,
            namespace=development.NAMESPACE,
        )
        self.exclusion_receipt_path = self.root / "exclusion-receipt.json"
        sealed(
            self.exclusion_receipt_path, development.exclusions.RECEIPT_SCHEMA,
            namespace=development.NAMESPACE,
            references={
                "protected_canonical_fingerprints": q.artifact_reference(
                    self.root / "fingerprints.json"
                ),
            },
            verdict={"development_games_authorized": True},
        )
        self.exclusion_receipt = q.load_sealed(self.exclusion_receipt_path)
        self.exclusion_context = {
            "receipt": self.exclusion_receipt,
            "plan": self.exclusion_plan,
            "protected_fingerprint_path": self.root / "fingerprints.json",
            "development_bank_records": {
                stage: development._regular(path)
                for stage, path in self.banks.items()
            },
            "development_ready": True,
        }

        self.control_selection_path = base_test.make_selection(
            self.root / "control", "compact-8x8", "rank4-control", False
        )
        control_selection, control_runtime = maintained._selection_runtime(
            self.control_selection_path
        )
        self.control_runtime = control_runtime
        self.control_source = self.root / "control/control.cpp"
        self.control_source.write_bytes(b"int main(){return 1;}\n")
        self.control_context = {
            "selection": control_selection,
            "selection_path": self.control_selection_path,
            "runtime_path": control_runtime,
            "runtime_declaration": dict(control_selection["runtime"]),
            "runtime": development._regular(control_runtime),
            "selection_record": development._sealed_record(
                self.control_selection_path, maintained.SELECTION_SCHEMA
            ),
            "rendered_source": {
                "bytes": self.control_source.stat().st_size,
                "sha256": q.sha256_file(self.control_source),
            },
        }
        self.plan_path = self.root / "development-v3/plan.json"

    def adapter_validator(self, path, output_root):
        if path.resolve() != self.handoff_path or output_root != self.root:
            raise development.DevelopmentError("synthetic adapter route changed")
        return self.adapter_context

    def exclusion_validator(self, plan, receipt, output_root):
        if (
            plan.resolve() != self.exclusion_plan_path
            or receipt.resolve() != self.exclusion_receipt_path
            or output_root != self.root
        ):
            raise development.DevelopmentError("synthetic exclusion route changed")
        value = dict(self.exclusion_context)
        value["development_bank_records"] = {
            stage: development._regular(path)
            for stage, path in self.banks.items()
        }
        return value

    def control_validator(self, path):
        if path.resolve() != self.control_selection_path.resolve():
            raise development.DevelopmentError("synthetic control route changed")
        return self.control_context

    def compiler_identity(self):
        return self.compiler

    def prepare(self):
        with mock.patch.object(
            development.adapter.v3, "canonical_v3_root", return_value=self.root
        ):
            return development.prepare_plan(
                self.root,
                adapter_handoff_path=self.handoff_path,
                exclusion_plan_path=self.exclusion_plan_path,
                exclusion_receipt_path=self.exclusion_receipt_path,
                rank4_control_selection_path=self.control_selection_path,
                created_at_utc="2026-09-01T13:00:00Z",
                adapter_validator=self.adapter_validator,
                exclusion_validator=self.exclusion_validator,
                control_validator=self.control_validator,
                compiler_identity=self.compiler_identity,
            )

    def plan_loader(self, path, *, output_root):
        return development.validate_plan(
            path, output_root=output_root,
            adapter_validator=self.adapter_validator,
            exclusion_validator=self.exclusion_validator,
            control_validator=self.control_validator,
            compiler_identity=self.compiler_identity,
        )

    def candidates(self, plan):
        run_root = pathlib.Path(plan["outputs"]["development_root"])
        binaries = pathlib.Path(plan["outputs"]["binaries"])
        binaries.mkdir(parents=True, exist_ok=True)
        candidate_bytes = b"candidate-binary"
        control_bytes = b"control-binary"
        candidate_binary = binaries / (
            hashlib.sha256(candidate_bytes).hexdigest() + ".rank4-gate"
        )
        control_binary = binaries / (
            hashlib.sha256(control_bytes).hexdigest() + ".rank4-gate"
        )
        candidate_binary.write_bytes(candidate_bytes)
        control_binary.write_bytes(control_bytes)
        candidate_selection = q.load_sealed(self.candidate_selection)
        control_selection = q.load_sealed(self.control_selection_path)
        return [
            maintained.Candidate(
                candidate_id=development.CANDIDATE_ID,
                architecture=development.CAPACITY_ARCHITECTURE,
                target="search-target",
                selection_path=self.candidate_selection,
                selection_sha256=q.sha256_file(self.candidate_selection),
                selection_body_sha256=candidate_selection["body_sha256"],
                runtime_path=self.candidate_runtime,
                runtime_sha256=q.sha256_file(self.candidate_runtime),
                deployment_eligible=True,
                source_path=self.candidate_source,
                source_sha256=q.sha256_file(self.candidate_source),
                source_bytes=self.candidate_source.stat().st_size,
                binary_path=candidate_binary,
                binary_sha256=q.sha256_file(candidate_binary),
            ),
            maintained.Candidate(
                candidate_id=development.CONTROL_ID,
                architecture=development.CONTROL_ARCHITECTURE,
                target=maintained.campaign.CONTROL_TARGET,
                selection_path=self.control_selection_path,
                selection_sha256=q.sha256_file(self.control_selection_path),
                selection_body_sha256=control_selection["body_sha256"],
                runtime_path=self.control_runtime,
                runtime_sha256=q.sha256_file(self.control_runtime),
                deployment_eligible=False,
                source_path=self.control_source,
                source_sha256=q.sha256_file(self.control_source),
                source_bytes=self.control_source.stat().st_size,
                binary_path=control_binary,
                binary_sha256=q.sha256_file(control_binary),
            ),
        ]

    def runner(self, *, resume=False, gate_executor=None, candidate_builder=None):
        self.prepare()
        return v3runner.DiscreteV3DevelopmentRunner(
            plan_path=self.plan_path,
            output_root=self.root,
            resume=resume,
            compiler=mock.Mock(side_effect=AssertionError("compiler unexpectedly ran")),
            gate_executor=gate_executor,
            compiler_identity=self.compiler,
            plan_loader=self.plan_loader,
            candidate_builder=candidate_builder or self.candidates,
        )


class V3FakeCampaign(base_test.FakeCampaign):
    def __init__(self, context):
        super().__init__()
        self.context = context

    def outcome(self, spec):
        stage = spec["stage"]
        identifier = spec["candidate_id"]
        candidate = development.CANDIDATE_ID
        if stage == "model_screen":
            return ((130, 65, 65, 9.0) if identifier == candidate
                    else (140, 70, 70, 7.0))
        if stage == "tuple_screen":
            if identifier == f"{candidate}:c0.80-f0.5-l1":
                return 160, 80, 80, 9.0
            if identifier == f"{candidate}:c0.95-f0.5-l1":
                return 159, 79, 80, 10.0
            ordinal = list(maintained.campaign.TUPLE_ROSTER).index(tuple(spec["tuple"]))
            wins = 120 - ordinal
            return wins, wins // 2, wins - wins // 2, 11.0
        if stage == "tuple_confirmation":
            if identifier == f"{candidate}:c0.80-f0.5-l1":
                return 302, 151, 151, 9.0
            return 300, 150, 150, 10.0
        return super().outcome(spec)

    def __call__(self, candidate, bank, spec):
        result = super().__call__(candidate, bank, spec)
        planned = (
            self.context.plan["candidate"]
            if candidate.candidate_id == development.CANDIDATE_ID
            else self.context.plan["rank4_control"]
        )
        result["bindings"]["candidate_runtime_body_sha256"] = planned[
            "runtime_identity"
        ]["body_sha256"]
        result["bindings"]["candidate_payload_sha256"] = planned[
            "runtime_identity"
        ]["payload_sha256"]
        return result


class DiscreteV3DevelopmentTest(unittest.TestCase):
    def patches(self):
        return (
            mock.patch.object(
                v3runner.maintained.openings, "validate_bank",
                side_effect=base_test.fake_validate_bank,
            ),
            mock.patch.object(
                v3runner.maintained, "paired_bootstrap_lower", return_value=0.01,
            ),
            mock.patch.object(
                development.maintained, "paired_bootstrap_lower", return_value=0.01,
            ),
        )

    def test_real_control_relative_runtime_is_normalized_to_absolute_record(self):
        selection = q.load_sealed(
            development.RANK4_CONTROL_SELECTION_PATH,
            maintained.SELECTION_SCHEMA,
        )
        self.assertEqual(
            selection["runtime"]["path"],
            "quantized-runtimes/"
            "41661543c6314c378368298ebe15ef0008c465d6c2ed157b993552df81455d84."
            "runtime.json",
        )
        details = development._default_control_validator(
            development.RANK4_CONTROL_SELECTION_PATH
        )
        runtime_path = pathlib.Path(details["runtime"]["path"])
        self.assertTrue(runtime_path.is_absolute())
        self.assertEqual(runtime_path, pathlib.Path(details["runtime_path"]))
        self.assertEqual(details["runtime_declaration"], selection["runtime"])
        self.assertEqual(
            details["runtime"]["sha256"],
            development.RANK4_CONTROL_RUNTIME_SHA256,
        )

    def test_plan_binds_handoff_exclusion_control_banks_and_powerless_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            plan_path = context.prepare()
            plan = context.plan_loader(plan_path, output_root=context.root)
            context.plan = plan
            self.assertEqual(plan["candidate"]["architecture"], "6301-12-8-1")
            self.assertTrue(plan["exclusion"]["development_ready"])
            self.assertEqual(set(plan["banks"]), set(development.STAGE_ORDER))
            self.assertFalse(plan["policy"]["final_bank_generation_authorized"])
            self.assertFalse(plan["policy"]["rank4_gate_authorized"])
            self.assertFalse(plan["policy"]["upload_authorized"])
            self.assertEqual(
                plan["request_ancestry"]["adapter_handoff_sha256"],
                q.sha256_file(context.handoff_path),
            )

    def test_prerequisite_failure_prevents_plan_compiler_and_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.exclusion_context["development_ready"] = False
            with self.assertRaises(development.DevelopmentError):
                context.prepare()
            self.assertFalse(context.plan_path.exists())
            self.assertFalse((context.root / "development-v3").exists())

        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            foreign = context.root / "development-v3/requests/foreign"
            foreign.parent.mkdir(parents=True)
            foreign.write_bytes(b"foreign")
            with self.assertRaisesRegex(
                development.DevelopmentError, "predates"
            ):
                context.prepare()
            self.assertFalse(context.plan_path.exists())

    def test_runner_executes_exact_adaptive_sequence_and_seals_finalist(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(
                context.plan_path, output_root=context.root
            )
            fake = V3FakeCampaign(context)
            active = (
                q.sha256_file(v3runner.HERE / "model.hpp"),
                q.sha256_file(v3runner.HERE / "submission.cpp"),
            )
            runner = context.runner(gate_executor=fake)
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                value = runner.execute()
            self.assertEqual(fake.calls, 18)
            self.assertEqual(value["finalist"]["schema"], development.FINALIST_SCHEMA)
            self.assertEqual(value["finalist"]["candidate"]["candidate_id"], development.CANDIDATE_ID)
            self.assertTrue(value["finalist"]["fresh_protected_tests_opened"])
            self.assertFalse(value["finalist"]["final_bank_generation_authorized"])
            self.assertEqual(
                active,
                (q.sha256_file(v3runner.HERE / "model.hpp"),
                 q.sha256_file(v3runner.HERE / "submission.cpp")),
            )
            requests = list((context.root / "development-v3/requests").glob("*.request.json"))
            self.assertEqual(len(requests), 18)
            for path in requests:
                request = q.load_sealed(path, development.REQUEST_SCHEMA)
                self.assertEqual(request["ancestry"], context.plan["request_ancestry"])
                self.assertIn("binary_sha256", request["candidate"])
                self.assertEqual(
                    request["expected_configuration"]["rank4_nodes"], 3_000_000
                )

    def test_full_resume_reuses_finalist_without_compiler_or_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            fake = V3FakeCampaign(context)
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                first = context.runner(gate_executor=fake).execute()
            gate = mock.Mock(side_effect=AssertionError("gate reran"))
            builder = mock.Mock(side_effect=AssertionError("candidate compile reran"))
            resumed = context.runner(
                resume=True, gate_executor=gate, candidate_builder=builder
            )
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                second = resumed.execute()
            self.assertEqual(first["finalist"], second["finalist"])
            gate.assert_not_called()
            builder.assert_not_called()

    def test_request_only_partial_run_resumes_and_tampered_reference_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            failing = mock.Mock(side_effect=RuntimeError("synthetic crash"))
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(RuntimeError):
                    context.runner(gate_executor=failing).execute()
            self.assertEqual(
                len(list((context.root / "development-v3/requests").glob("*.request.json"))),
                1,
            )
            forbidden_gate = mock.Mock(
                side_effect=AssertionError("non-resume gate ran")
            )
            with self.assertRaisesRegex(v3runner.RunnerError, "requires --resume"):
                context.runner(gate_executor=forbidden_gate).execute()
            forbidden_gate.assert_not_called()
            fake = V3FakeCampaign(context)
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                context.runner(resume=True, gate_executor=fake).execute()
            reference = next((context.root / "development-v3/run-references-v3").glob("*.json"))
            reference.write_bytes(b"tampered")
            finalist_ref = context.root / "development-v3/finalist-reference.json"
            finalist = q.load_sealed(finalist_ref, development.FINALIST_REFERENCE_SCHEMA)
            pathlib.Path(finalist["finalist"]["path"]).unlink()
            finalist_ref.unlink()
            (context.root / "development-v3/development-result.json").unlink()
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                with self.assertRaises((development.DevelopmentError, ValueError)):
                    context.runner(
                        resume=True,
                        gate_executor=mock.Mock(
                            side_effect=AssertionError("gate should not rerun")
                        ),
                    ).execute()

    def test_orphan_gate_output_is_promoted_without_rerunning_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            fake = V3FakeCampaign(context)

            def crash_after_gate(candidate, bank, spec):
                gate = fake(candidate, bank, spec)
                base_request = {
                    "schema": "papersoccer.compact-value-bfm-development-request.v1",
                    "candidate_id": spec["candidate_id"],
                    "model_candidate_id": candidate.candidate_id,
                    "selection_sha256": candidate.selection_sha256,
                    "selection_body_sha256": candidate.selection_body_sha256,
                    "runtime_sha256": candidate.runtime_sha256,
                    "candidate_source_sha256": candidate.source_sha256,
                    "binary_sha256": candidate.binary_sha256,
                    "rank4_source_sha256": maintained.RANK4_SHA256,
                    "bank_sha256": bank.sha256,
                    "bank_manifest_sha256": bank.manifest_sha256,
                    "stage": spec["stage"], "mode": spec["mode"],
                    "tuple": list(spec["tuple"]), "work": dict(spec["work"]),
                    "pairs": bank.pairs,
                }
                request_sha = maintained.sha256_bytes(
                    maintained.canonical_json_bytes(base_request)
                )
                path = context.root / f"development-v3/scratch/{request_sha}.gate.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(maintained.canonical_json_bytes(gate))
                raise RuntimeError("crash after gate output")

            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(RuntimeError):
                    context.runner(gate_executor=crash_after_gate).execute()
            resumed_gate = V3FakeCampaign(context)
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                value = context.runner(
                    resume=True, gate_executor=resumed_gate
                ).execute()
            # Seventeen remaining games run; the orphaned first output is promoted.
            self.assertEqual(resumed_gate.calls, 17)
            self.assertEqual(value["finalist"]["actual_clock"]["wins"], 211)

    def test_missing_outer_reference_is_recovered_without_gate_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            fake = V3FakeCampaign(context)
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                first = context.runner(gate_executor=fake).execute()
            outer = next((context.root / "development-v3/run-references-v3").glob("*.json"))
            outer.unlink()
            finalist_ref = context.root / "development-v3/finalist-reference.json"
            finalist = q.load_sealed(finalist_ref, development.FINALIST_REFERENCE_SCHEMA)
            pathlib.Path(finalist["finalist"]["path"]).unlink()
            finalist_ref.unlink()
            (context.root / "development-v3/development-result.json").unlink()
            gate = mock.Mock(side_effect=AssertionError("gate reran"))
            patches = self.patches()
            with patches[0], patches[1], patches[2]:
                second = context.runner(resume=True, gate_executor=gate).execute()
            gate.assert_not_called()
            self.assertEqual(first["finalist"]["tuple"], second["finalist"]["tuple"])
            self.assertTrue(outer.exists())

    def test_request_core_receipt_binary_and_finalist_tamper_fail_closed(self):
        targets = ("request", "core", "config", "binary", "finalist")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                context = SyntheticContext(pathlib.Path(temporary))
                context.prepare()
                context.plan = context.plan_loader(
                    context.plan_path, output_root=context.root
                )
                fake = V3FakeCampaign(context)
                patches = self.patches()
                with patches[0], patches[1], patches[2]:
                    value = context.runner(gate_executor=fake).execute()
                if target == "request":
                    path = next((context.root / "development-v3/requests").glob("*.json"))
                    path.write_bytes(b"tampered")
                elif target == "core":
                    path = next((context.root / "development-v3/receipts").glob("*.json"))
                    path.write_bytes(b"tampered")
                elif target == "config":
                    path = next((context.root / "development-v3/scratch").glob("*.gate.json"))
                    document = json.loads(path.read_bytes())
                    document["config"]["rank4_nodes"] = 1
                    path.write_bytes(maintained.canonical_json_bytes(document))
                elif target == "binary":
                    path = pathlib.Path(value["finalist"]["binary"]["path"])
                    path.write_bytes(path.read_bytes() + b"tampered")
                else:
                    path = pathlib.Path(value["finalist_path"])
                    path.write_bytes(b"tampered")
                with self.assertRaises((development.DevelopmentError, ValueError)):
                    development.validate_finalist(
                        context.root / "development-v3/finalist-reference.json",
                        plan_path=context.plan_path,
                        output_root=context.root,
                        plan_validator=context.plan_loader,
                    )

    def test_compile_reference_escape_key_name_and_identity_tamper_fail_closed(self):
        for target in ("escape", "key", "binary-name", "identity"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                context = SyntheticContext(pathlib.Path(temporary))
                context.prepare()
                context.plan = context.plan_loader(
                    context.plan_path, output_root=context.root
                )
                fake = V3FakeCampaign(context)
                patches = self.patches()
                with patches[0], patches[1], patches[2]:
                    context.runner(gate_executor=fake).execute()
                request_path = next(
                    (context.root / "development-v3/requests").glob("*.request.json")
                )
                request = q.load_sealed(request_path, development.REQUEST_SCHEMA)
                candidate = dict(request["candidate"])
                reference_value = dict(request["compile_reference"])
                if target == "escape":
                    original = pathlib.Path(reference_value["path"])
                    escaped = context.root / "escaped-reference.json"
                    escaped.write_bytes(original.read_bytes())
                    reference_value = development._sealed_record(
                        escaped, v3runner.BINARY_REFERENCE_SCHEMA
                    )
                elif target in {"key", "binary-name"}:
                    reference_path = pathlib.Path(reference_value["path"])
                    reference = q.load_sealed(
                        reference_path, v3runner.BINARY_REFERENCE_SCHEMA
                    )
                    reference.pop("body_sha256")
                    if target == "key":
                        reference["compile_key"] = "0" * 64
                    else:
                        wrong = reference_path.parent / "wrong-binary-name"
                        wrong.write_bytes(b"wrong-binary")
                        reference["binary"] = development._regular(wrong)
                        candidate["binary_path"] = str(wrong.resolve())
                        candidate["binary_sha256"] = q.sha256_file(wrong)
                        candidate["binary_bytes"] = wrong.stat().st_size
                    reference_path.write_bytes(
                        q.canonical_json_bytes(q.seal(reference))
                    )
                    reference_value = development._sealed_record(
                        reference_path, v3runner.BINARY_REFERENCE_SCHEMA
                    )
                else:
                    candidate["architecture"] = "6301-8-8-1"
                with self.assertRaises(development.DevelopmentError):
                    if target == "identity":
                        # The deep request validator, not merely a stale file
                        # hash, must reject a self-consistently sealed identity.
                        request["candidate"] = candidate
                        request.pop("body_sha256")
                        tampered_path, _ = development._write_content_addressed(
                            context.root / "development-v3/requests",
                            request, ".request.json",
                        )
                        outer = next(
                            (context.root / "development-v3/receipts-v3").glob(
                                "*.receipt.json"
                            )
                        )
                        receipt = q.load_sealed(outer, development.RECEIPT_SCHEMA)
                        receipt.pop("body_sha256")
                        receipt["request"] = development._sealed_record(
                            tampered_path, development.REQUEST_SCHEMA
                        )
                        receipt["request_sha256"] = q.sha256_file(tampered_path)
                        tampered_receipt, _ = development._write_content_addressed(
                            context.root / "development-v3/receipts-v3",
                            receipt, ".receipt.json",
                        )
                        development.validate_run_receipt(
                            development._sealed_record(
                                tampered_receipt, development.RECEIPT_SCHEMA
                            ),
                            context.plan,
                        )
                    else:
                        development._validate_compile_reference(
                            reference_value, plan=context.plan,
                            candidate=candidate,
                            development_plan_record=request["development_plan"],
                        )

    def test_plan_and_finalist_tamper_and_active_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            changed = context.banks["actual_clock"].read_bytes() + b" "
            context.banks["actual_clock"].write_bytes(changed)
            with self.assertRaises(development.DevelopmentError):
                context.plan_loader(context.plan_path, output_root=context.root)

        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            plan = q.load_sealed(context.plan_path, development.PLAN_SCHEMA)
            plan.pop("body_sha256")
            plan["tools"]["qualification_tool"]["sha256"] = "0" * 64
            context.plan_path.write_bytes(
                q.canonical_json_bytes(q.seal(plan))
            )
            with self.assertRaises(development.DevelopmentError):
                context.plan_loader(context.plan_path, output_root=context.root)

        with tempfile.TemporaryDirectory() as temporary:
            context = SyntheticContext(pathlib.Path(temporary))
            context.prepare()
            context.plan = context.plan_loader(context.plan_path, output_root=context.root)
            fake = V3FakeCampaign(context)
            runner = context.runner(gate_executor=fake)
            lock = context.root / "development-v3/development.lock"
            with v3runner._exclusive_lock(lock):
                with self.assertRaises(v3runner.RunnerError):
                    runner.execute()

    def test_every_planned_execution_output_redirect_fails_before_games(self):
        route_names = (
            "requests", "receipts", "references", "base_receipts",
            "base_references", "binaries", "sources", "gate_banks",
            "scratch", "result", "finalists", "finalist_reference",
        )
        for route_name in route_names:
            with self.subTest(route=route_name), tempfile.TemporaryDirectory() as temporary:
                context = SyntheticContext(pathlib.Path(temporary))
                context.prepare()
                context.plan = context.plan_loader(
                    context.plan_path, output_root=context.root
                )
                run_root = context.root / "development-v3"
                routes = {
                    "requests": pathlib.Path(context.plan["outputs"]["requests"]),
                    "receipts": pathlib.Path(context.plan["outputs"]["receipts"]),
                    "references": pathlib.Path(context.plan["outputs"]["references"]),
                    "base_receipts": pathlib.Path(context.plan["outputs"]["base_receipts"]),
                    "base_references": pathlib.Path(context.plan["outputs"]["base_references"]),
                    "binaries": pathlib.Path(context.plan["outputs"]["binaries"]),
                    "sources": run_root / "candidate-sources",
                    "gate_banks": run_root / "gate-banks",
                    "scratch": run_root / "scratch",
                    "result": pathlib.Path(context.plan["outputs"]["result"]),
                    "finalists": pathlib.Path(context.plan["outputs"]["finalists"]),
                    "finalist_reference": pathlib.Path(
                        context.plan["outputs"]["finalist_reference"]
                    ),
                }
                route = routes[route_name]
                route.parent.mkdir(parents=True, exist_ok=True)
                target = context.root / f"redirect-target-{route_name}"
                if route_name in {"result", "finalist_reference"}:
                    target.write_bytes(b"redirect")
                else:
                    target.mkdir()
                route.symlink_to(target)
                gate = mock.Mock(side_effect=AssertionError("gate ran"))
                with self.assertRaises(v3runner.RunnerError):
                    context.runner(gate_executor=gate).execute()
                gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
