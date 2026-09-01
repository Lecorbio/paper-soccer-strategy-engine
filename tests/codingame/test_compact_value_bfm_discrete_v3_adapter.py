import copy
import hashlib
import importlib.util
import pathlib
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_discrete_v3_adapter.py"
SPEC = importlib.util.spec_from_file_location(
    "compact_value_bfm_discrete_v3_adapter", TOOL
)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(adapter)
q = adapter.qualification


def dummy_record(root, label):
    return {
        "path": str((root / f"not-opened-{label}.json").resolve()),
        "bytes": 1,
        "sha256": hashlib.sha256(label.encode("ascii")).hexdigest(),
    }


def overwrite_sealed(path, changes):
    value = q.load_sealed(path)
    value.pop("body_sha256")
    value.update(copy.deepcopy(changes))
    path.write_bytes(q.canonical_json_bytes(q.seal(value)))


class AdapterFixture:
    def __init__(self, root, *, materialization_changes=None):
        self.root = root.resolve()
        self.runtime_path = self.root / "training/quantized-runtimes/runtime.json"
        self.runtime_path.parent.mkdir(parents=True)
        self.runtime_path.write_bytes(b'{"runtime":"fixture"}\n')
        self.prior_path = self.root / "training/quantized-runtimes/prior.json"
        self.prior_path.write_bytes(b'{"runtime":"prior"}\n')
        self.source_path = self.root / "fine-tune/generated-sources/source.cpp"
        self.source_path.parent.mkdir(parents=True)
        self.source_path.write_bytes(b"int main(){return 0;}\n")
        runtime = adapter._regular_record(self.runtime_path)
        source = adapter._regular_record(self.source_path)
        prior_plan_record = {
            **adapter._regular_record(self.prior_path),
            "resolved_path": str(self.prior_path.resolve()),
            "executable": False,
        }
        self.configuration = {"minimum_samples_per_report": 20_000}
        self.plan_path = self.root / "discrete-v3-plan.json"
        tool_record = adapter._regular_record(TOOL)
        self.plan = q.write_sealed(self.plan_path, {
            "schema": adapter.v3.PLAN_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "fresh_protected_holdout": self.configuration,
            "training": {"prior_compact_runtime": prior_plan_record},
            "tools": {
                name: tool_record for name in adapter.TOOL_CLOSURE_KEYS
            },
        })
        self.selection_path = self.root / "selections/selection.json"
        self.selection_path.parent.mkdir(parents=True)
        self.selection = q.write_sealed(self.selection_path, {
            "schema": adapter.v3.SELECTION_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "architecture": adapter.EXPECTED_RUNTIME_ARCHITECTURE,
            "source_export": {
                "runtime_sha256": runtime["sha256"],
                "runtime_body_sha256": "1" * 64,
                "model_header_sha256": "2" * 64,
                "source_sha256": source["sha256"],
                "source_ascii_bytes": source["bytes"],
                "source_limit_exclusive": 95_000,
            },
            "offline_gate": {
                "passed": True,
                "status": "offline-evaluator-qualified-not-game-gated",
                "errors": [],
            },
        })
        self.selection_reference_path = self.root / "selection-reference.json"
        q.write_sealed(self.selection_reference_path, {
            "schema": adapter.v3.SELECTION_REFERENCE_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "selection": q.artifact_reference(
                self.selection_path, adapter.v3.SELECTION_SCHEMA
            ),
        })
        self.outcome_path = self.root / "governance/02-outcome.json"
        self.outcome_path.parent.mkdir(parents=True)
        q.write_sealed(self.outcome_path, {
            "schema": adapter.v3.OUTCOME_SCHEMA,
            "namespace": adapter.NAMESPACE,
        })

        selection_reference = q.artifact_reference(
            self.selection_path, adapter.v3.SELECTION_SCHEMA
        )
        holdout = self.root / "fresh-holdout"
        holdout.mkdir()
        self.materialization_claim_path = holdout / "00-materialization-claim.json"
        q.write_sealed(self.materialization_claim_path, {
            "schema": adapter.FRESH_MATERIALIZATION_CLAIM_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "campaign_id": f"{adapter.v3.SUCCESSOR_CAMPAIGN_ID}-holdout",
            "status": "fresh-protected-holdout-materialization-claimed-once",
            "successor_plan": q.artifact_reference(
                self.plan_path, adapter.v3.PLAN_SCHEMA
            ),
            "immutable_selection": selection_reference,
            "selected_runtime": runtime,
            "prior_runtime": adapter._regular_record(self.prior_path),
            "configuration": self.configuration,
            "selection_may_change": False,
            "old_protected_tests_permitted": False,
            "materialization_attempts_authorized": 1,
            "exclusive_process_lock": str(
                (holdout / "materialization.lock").resolve()
            ),
            "claimed_at_utc": "2026-09-01T11:00:00Z",
        })
        self.samples = {
            "search": 64_000, "rank4": 64_000, "canonical": 32_000,
        }
        self.metrics = {
            label: {
                "samples": count,
                "weighted_huber": 0.05,
                "objective_weighted_huber": 0.05,
                "sign_accuracy": 0.86,
                "correlation": 0.7,
                "mae": 0.2,
                "prediction_mean": 0.01,
            }
            for label, count in self.samples.items()
        }
        self.materialization_path = holdout / "materialization-receipt.json"
        materialization = {
            "schema": adapter.FRESH_MATERIALIZATION_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "campaign_id": f"{adapter.v3.SUCCESSOR_CAMPAIGN_ID}-holdout",
            "status": "fresh-protected-holdout-materialized-once",
            "claim": q.artifact_reference(
                self.materialization_claim_path,
                adapter.FRESH_MATERIALIZATION_CLAIM_SCHEMA,
            ),
            "immutable_selection": selection_reference,
            "game_plan": {}, "game_plan_tsv": {}, "game_plan_rows": 3_200,
            "fresh_roots": {}, "fresh_roots_tsv": {}, "fresh_opening_bank": {},
            "games": {}, "games_manifest": {}, "positions": {},
            "positions_manifest": dummy_record(self.root, "positions"),
            "hard_positions": {}, "search_labels": {}, "rank4_labels": {},
            "canonical_labels": {}, "canonical_label_rows": 64_000,
            "packing_priors": [],
            # Intentionally absent: injected tests prove adapter orchestration
            # never opens these protected fixture routes.
            "test_shards": {
                label: dummy_record(self.root, label)
                for label in ("search", "rank4", "canonical")
            },
            "test_samples": self.samples,
            "group_isolation": {"passed": True},
            "split_isolation": {"passed": True},
            "stage_receipts": [],
            "selection_changed": False,
            "old_protected_tests_accessed": False,
            "fresh_protected_tests_opened": True,
        }
        if materialization_changes:
            materialization.update(copy.deepcopy(materialization_changes))
        q.write_sealed(self.materialization_path, materialization)
        self.v1_plan_path = self.root / "development-adapter/adapter-plan.json"
        self.v1_plan_path.parent.mkdir(parents=True, exist_ok=True)
        v1_tool = dummy_record(self.root, "v1-adapter")
        v1_tests = dummy_record(self.root, "v1-tests")
        q.write_sealed(self.v1_plan_path, {
            "schema": adapter.V1_ADAPTER_PLAN_SCHEMA,
            "namespace": adapter.NAMESPACE,
            "campaign_id": adapter.v3.SUCCESSOR_CAMPAIGN_ID,
            "status": adapter.ADAPTER_PLAN_STATUS,
            "candidate": {
                "architecture": "6301-8-8-1",
                "runtime_architecture": adapter.EXPECTED_RUNTIME_ARCHITECTURE,
            },
            "tool_closure": {
                "adapter": v1_tool,
                "adapter_tests": v1_tests,
            },
        })
        self.retirement_path = self.root / "development-adapter/v1-retirement.json"
        self.adapter_plan_path = (
            self.root / "development-adapter/adapter-plan-v2.json"
        )
        self.evaluation_paths = adapter._evaluation_paths(self.root)
        self.output = self.root / "development-adapter/handoff-v2.json"

    def candidate_loader(self, plan_path, output_root):
        if plan_path.resolve() != self.plan_path or output_root != self.root:
            raise AssertionError("fixture candidate loader received a foreign route")
        selection = q.load_sealed(
            self.selection_path, adapter.v3.SELECTION_SCHEMA
        )
        return {
            "output_root": self.root,
            "plan_path": self.plan_path,
            "plan": self.plan,
            "plan_reference": q.artifact_reference(
                self.plan_path, adapter.v3.PLAN_SCHEMA
            ),
            "selection_reference_path": self.selection_reference_path,
            "selection_reference": q.artifact_reference(
                self.selection_reference_path,
                adapter.v3.SELECTION_REFERENCE_SCHEMA,
            ),
            "selection_path": self.selection_path,
            "selection": selection,
            "selection_artifact": q.artifact_reference(
                self.selection_path, adapter.v3.SELECTION_SCHEMA
            ),
            "outcome_reference": q.artifact_reference(
                self.outcome_path, adapter.v3.OUTCOME_SCHEMA
            ),
            "runtime_path": self.runtime_path,
            "runtime": adapter._regular_record(self.runtime_path),
            "architecture": {
                "runtime_name": adapter.EXPECTED_RUNTIME_ARCHITECTURE,
                "dimensions": list(adapter.EXPECTED_DIMENSIONS),
                "campaign_name": adapter.EXPECTED_CAMPAIGN_ARCHITECTURE,
            },
            "source_path": self.source_path,
            "source": adapter._regular_record(self.source_path),
        }

    def v1_loader(self, output_root):
        if output_root != self.root:
            raise AssertionError("fixture v1 loader received a foreign root")
        plan = q.load_sealed(self.v1_plan_path, adapter.V1_ADAPTER_PLAN_SCHEMA)
        return {
            "path": self.v1_plan_path,
            "plan": plan,
            "artifact": q.artifact_reference(
                self.v1_plan_path, adapter.V1_ADAPTER_PLAN_SCHEMA
            ),
            "tool": dict(plan["tool_closure"]["adapter"]),
            "tests": dict(plan["tool_closure"]["adapter_tests"]),
            "declared_architecture": "6301-8-8-1",
            "runtime_architecture": adapter.EXPECTED_RUNTIME_ARCHITECTURE,
            "dimensions": list(adapter.EXPECTED_DIMENSIONS),
            "derived_architecture": adapter.EXPECTED_CAMPAIGN_ARCHITECTURE,
        }

    def retirement_validator(self, path, *, output_root):
        return adapter.validate_v1_retirement(
            path, output_root=output_root, v1_plan_loader=self.v1_loader
        )

    def retire(self):
        return adapter.retire_v1(
            self.retirement_path,
            output_root=self.root,
            retired_at_utc="2026-09-01T11:58:00Z",
            v1_plan_loader=self.v1_loader,
        )

    def prepare(self):
        self.retire()
        return adapter.prepare_adapter(
            self.adapter_plan_path,
            plan_path=self.plan_path,
            output_root=self.root,
            retirement_path=self.retirement_path,
            planned_at_utc="2026-09-01T11:59:00Z",
            candidate_loader=self.candidate_loader,
            retirement_validator=self.retirement_validator,
        )

    def diagnostic(self, _candidate=None, _materialization=None):
        return {"samples": self.samples, "metrics": self.metrics}

    def evaluate(self, evaluator=None):
        self.prepare()
        return adapter.evaluate_adapter(
            adapter_plan_path=self.adapter_plan_path,
            plan_path=self.plan_path,
            output_root=self.root,
            evaluator=evaluator or self.diagnostic,
            candidate_loader=self.candidate_loader,
            retirement_validator=self.retirement_validator,
            clock=lambda: "2026-09-01T12:00:00Z",
        )

    def create(self):
        self.evaluate()
        return adapter.create_handoff(
            self.output,
            adapter_plan_path=self.adapter_plan_path,
            plan_path=self.plan_path,
            output_root=self.root,
            evaluation_completion_path=self.evaluation_paths["completion"],
            created_at_utc="2026-09-01T12:01:00Z",
            candidate_loader=self.candidate_loader,
            retirement_validator=self.retirement_validator,
        )


class DiscreteV3AdapterTest(unittest.TestCase):
    def test_retire_v1_seals_exact_defect_without_touching_v1_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            before = fixture.v1_plan_path.read_bytes()
            receipt = fixture.retire()
            self.assertEqual(set(receipt), adapter.V1_RETIREMENT_FIELDS)
            self.assertEqual(receipt["status"], adapter.V1_RETIREMENT_STATUS)
            self.assertEqual(
                receipt["defect"]["incorrect_declared_architecture"],
                "6301-8-8-1",
            )
            self.assertEqual(
                receipt["defect"]["runtime_dimensions"],
                list(adapter.EXPECTED_DIMENSIONS),
            )
            self.assertFalse(receipt["adapter_evaluation_started"])
            self.assertFalse(receipt["protected_metrics_opened"])
            self.assertEqual(fixture.v1_plan_path.read_bytes(), before)
            self.assertFalse(fixture.adapter_plan_path.exists())
            self.assertFalse(fixture.evaluation_paths["root"].exists())

    def test_retire_v1_rejects_any_v1_evaluation_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            v1_lock = fixture.root / "development-adapter/evaluation.lock"
            v1_lock.touch()
            with self.assertRaisesRegex(adapter.AdapterError, "v1 evaluation"):
                fixture.retire()
            self.assertFalse(fixture.retirement_path.exists())

    def test_prepare_seals_exact_pre_metric_contract_and_tool_test_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            plan = fixture.prepare()
            self.assertEqual(set(plan), adapter.ADAPTER_PLAN_FIELDS)
            self.assertEqual(plan["status"], adapter.ADAPTER_PLAN_STATUS)
            self.assertEqual(plan["schema"], adapter.ADAPTER_PLAN_SCHEMA)
            self.assertEqual(
                plan["candidate"]["architecture"],
                adapter.EXPECTED_CAMPAIGN_ARCHITECTURE,
            )
            self.assertEqual(
                plan["candidate"]["dimensions"],
                list(adapter.EXPECTED_DIMENSIONS),
            )
            self.assertEqual(
                plan["v1_retirement"],
                q.artifact_reference(
                    fixture.retirement_path, adapter.V1_RETIREMENT_SCHEMA
                ),
            )
            self.assertNotEqual(fixture.v1_plan_path, fixture.adapter_plan_path)
            self.assertEqual(
                fixture.evaluation_paths["root"].name, "evaluation-v2"
            )
            self.assertEqual(
                fixture.evaluation_paths["lock"].name, "evaluation-v2.lock"
            )
            self.assertEqual(
                q.load_sealed(fixture.v1_plan_path)["schema"],
                adapter.V1_ADAPTER_PLAN_SCHEMA,
            )
            self.assertFalse(plan["policy"]["fresh_protected_tests_opened"])
            evidence = plan["expected_fresh_evidence"]
            self.assertFalse(evidence["metric_dependent_branch_authorized"])
            self.assertFalse(evidence["legacy_evaluation_outputs_accepted"])
            self.assertTrue(
                evidence["no_evaluation_outputs_observed_at_prepare"]
            )
            self.assertNotIn("metrics", plan)
            self.assertEqual(
                plan["tool_closure"]["adapter"]["sha256"],
                adapter._sha256_file(TOOL),
            )
            self.assertEqual(
                plan["tool_closure"]["adapter_tests"]["sha256"],
                adapter._sha256_file(pathlib.Path(__file__)),
            )

    def test_v2_prepare_rejects_non_runtime_derived_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.retire()

            def bad_loader(plan_path, output_root):
                candidate = dict(fixture.candidate_loader(plan_path, output_root))
                candidate["architecture"] = {
                    "runtime_name": adapter.EXPECTED_RUNTIME_ARCHITECTURE,
                    "dimensions": list(adapter.EXPECTED_DIMENSIONS),
                    "campaign_name": "6301-8-8-1",
                }
                return candidate

            with self.assertRaisesRegex(adapter.AdapterError, "runtime-derived"):
                adapter.prepare_adapter(
                    fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    retirement_path=fixture.retirement_path,
                    planned_at_utc="2026-09-01T11:59:00Z",
                    candidate_loader=bad_loader,
                    retirement_validator=fixture.retirement_validator,
                )
            self.assertFalse(fixture.adapter_plan_path.exists())

    def test_v2_prepare_requires_retirement_and_rejects_late_v1_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            with self.assertRaises(adapter.AdapterError):
                adapter.prepare_adapter(
                    fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    retirement_path=fixture.retirement_path,
                    planned_at_utc="2026-09-01T11:59:00Z",
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )
            fixture.retire()
            v1_handoff = fixture.root / "development-adapter/handoff.json"
            v1_handoff.write_bytes(b"late-v1")
            with self.assertRaisesRegex(adapter.AdapterError, "v1 evaluation"):
                adapter.prepare_adapter(
                    fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    retirement_path=fixture.retirement_path,
                    planned_at_utc="2026-09-01T11:59:00Z",
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )
            self.assertFalse(fixture.adapter_plan_path.exists())

    def test_prepare_rejects_preexisting_hidden_legacy_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            hidden = fixture.root / "fresh-holdout/reports/hidden.json"
            hidden.parent.mkdir()
            hidden.write_bytes(b"hidden")
            with self.assertRaisesRegex(adapter.AdapterError, "direct/legacy"):
                fixture.prepare()
            self.assertFalse(fixture.adapter_plan_path.exists())

    def test_prepare_rejects_any_preexisting_legacy_reports_route(self):
        for route_kind in ("regular-file", "empty-directory"):
            with self.subTest(route_kind=route_kind), tempfile.TemporaryDirectory() as temporary:
                fixture = AdapterFixture(pathlib.Path(temporary))
                reports = fixture.root / "fresh-holdout/reports"
                reports.parent.mkdir(parents=True, exist_ok=True)
                if route_kind == "regular-file":
                    reports.write_bytes(b"hidden legacy bytes")
                else:
                    reports.mkdir()
                with self.assertRaisesRegex(adapter.AdapterError, "direct/legacy"):
                    fixture.prepare()
                self.assertFalse(fixture.adapter_plan_path.exists())

    def test_source_reproduction_requires_exact_payload_and_export_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runtime = root / "runtime.json"
            source = root / "source.cpp"
            runtime.write_bytes(b"runtime\n")
            payload = b"int main(){return 0;}\n"
            source.write_bytes(payload)
            model = types.SimpleNamespace(
                render_header=lambda _path: (
                    b"header", {
                        "body_sha256": "1" * 64,
                        "header_sha256": "2" * 64,
                    },
                )
            )
            submission = types.SimpleNamespace(
                render=lambda **_kwargs: (root / "unused.cpp", payload)
            )
            expected = {
                "runtime_sha256": adapter._sha256_file(runtime),
                "runtime_body_sha256": "1" * 64,
                "model_header_sha256": "2" * 64,
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "source_ascii_bytes": len(payload),
                "source_limit_exclusive": 95_000,
            }

            def run(expected_export):
                modules = iter((model, submission))
                with (
                    mock.patch.object(
                        adapter.v3.iteration,
                        "validate_maintained_python_tool_closure",
                        return_value={},
                    ),
                    mock.patch.object(
                        adapter.v3.iteration, "_verify_file_record",
                        side_effect=[root / "model.py", root / "submission.py"],
                    ),
                    mock.patch.object(
                        adapter.v3.iteration, "_load_module",
                        side_effect=lambda *_args: next(modules),
                    ),
                ):
                    return adapter._rerender_source(
                        runtime_path=runtime, source_path=source,
                        plan={
                            "tools": {
                                "model_exporter": {}, "submission_exporter": {},
                            }
                        },
                        expected_export=expected_export,
                    )

            run(expected)
            with self.assertRaisesRegex(adapter.AdapterError, "reproduces exactly"):
                run({**expected, "source_ascii_bytes": 1})

    def test_evaluate_seals_claim_before_injected_protected_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            observed = []

            def evaluator(_candidate, _materialization):
                observed.append(q.load_sealed(
                    fixture.evaluation_paths["claim"],
                    adapter.EVALUATION_CLAIM_SCHEMA,
                )["status"])
                self.assertFalse(
                    fixture.evaluation_paths["report_reference"].exists()
                )
                return fixture.diagnostic()

            completed = fixture.evaluate(evaluator)
            self.assertEqual(observed, [adapter.EVALUATION_CLAIM_STATUS])
            self.assertEqual(
                completed["completion"]["status"],
                adapter.EVALUATION_COMPLETION_STATUS,
            )

    def test_seals_powerless_adapter_owned_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            handoff = fixture.create()
            self.assertEqual(set(handoff), adapter.HANDOFF_FIELDS)
            self.assertEqual(handoff["status"], adapter.HANDOFF_STATUS)
            self.assertEqual(
                handoff["candidate"]["candidate_id"], adapter.CANDIDATE_ID
            )
            self.assertEqual(
                handoff["candidate"]["architecture"],
                adapter.EXPECTED_CAMPAIGN_ARCHITECTURE,
            )
            diagnostic = handoff["diagnostic_evidence"]
            self.assertEqual(
                diagnostic["classification"],
                "diagnostic-only-no-pass-fail-verdict",
            )
            self.assertIn("metrics", diagnostic)
            self.assertNotIn("passed", diagnostic)
            self.assertNotIn("acceptance_verdict", diagnostic)
            self.assertEqual(
                handoff["evaluation_claim"]["path"],
                str(fixture.evaluation_paths["claim"].resolve()),
            )
            self.assertEqual(
                handoff["evaluation_completion"]["path"],
                str(fixture.evaluation_paths["completion"].resolve()),
            )
            policy = handoff["policy"]
            self.assertFalse(policy["development_selected"])
            self.assertFalse(policy["rank4_final_bank_generation_authorized"])
            self.assertFalse(policy["rank4_gate_authorized"])
            self.assertFalse(policy["upload_authorized"])
            contract = handoff["development_contract"]
            self.assertEqual(contract["mode"], "discrete-v3-post-holdout")
            self.assertEqual(
                contract["required_output_schema"],
                "papersoccer.compact-value-bfm."
                "discrete-v3-post-holdout-finalist.v1",
            )
            exclusion = contract["fresh_position_symmetry_exclusion_audit"]
            self.assertEqual(
                exclusion["evidence_schema"],
                "papersoccer.compact-value-bfm."
                "discrete-v3-fresh-position-exclusion-audit.v1",
            )
            self.assertEqual(
                exclusion["equivalences"],
                ["exact", "rotate", "reflect", "rotate_reflect"],
            )
            validated = adapter.validate_handoff(
                fixture.output,
                adapter_plan_path=fixture.adapter_plan_path,
                plan_path=fixture.plan_path,
                output_root=fixture.root,
                evaluation_completion_path=fixture.evaluation_paths["completion"],
                candidate_loader=fixture.candidate_loader,
                retirement_validator=fixture.retirement_validator,
            )
            self.assertEqual(validated, handoff)

    def test_evaluate_rejects_wrong_samples_without_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            wrong = dict(fixture.samples)
            wrong["search"] -= 1
            with self.assertRaisesRegex(adapter.AdapterError, "sample/metric roster"):
                fixture.evaluate(lambda *_args: {
                    "samples": wrong, "metrics": fixture.metrics,
                })
            self.assertTrue(fixture.evaluation_paths["claim"].exists())
            self.assertFalse(fixture.evaluation_paths["completion"].exists())

    def test_evaluate_rejects_materialization_without_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(
                pathlib.Path(temporary),
                materialization_changes={"split_isolation": {"passed": False}},
            )
            with self.assertRaisesRegex(adapter.AdapterError, "materialization"):
                fixture.evaluate()
            self.assertFalse(fixture.evaluation_paths["claim"].exists())

    def test_direct_evaluation_reference_is_never_acceptable_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.prepare()
            direct = fixture.root / "fresh-holdout/report-reference.json"
            direct.write_bytes(b"direct")
            with self.assertRaisesRegex(adapter.AdapterError, "direct/legacy"):
                fixture.evaluate()
            self.assertFalse(fixture.evaluation_paths["claim"].exists())

    def test_racing_direct_evaluation_poisons_attempt_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            hidden = fixture.root / "fresh-holdout/reports/racing.json"

            def racing(_candidate, _materialization):
                hidden.parent.mkdir()
                hidden.write_bytes(b"racing")
                return fixture.diagnostic()

            with self.assertRaisesRegex(adapter.AdapterError, "direct/legacy"):
                fixture.evaluate(racing)
            self.assertTrue(fixture.evaluation_paths["claim"].exists())
            self.assertFalse(fixture.evaluation_paths["completion"].exists())
            hidden.unlink()
            hidden.parent.rmdir()
            with self.assertRaisesRegex(adapter.AdapterError, "not resumable"):
                fixture.evaluate()

    def test_preexisting_adapter_report_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.prepare()
            hidden = fixture.evaluation_paths["reports"] / "foreign.json"
            hidden.parent.mkdir(parents=True)
            hidden.write_bytes(b"foreign")
            with self.assertRaisesRegex(adapter.AdapterError, "partial or foreign"):
                fixture.evaluate()
            self.assertFalse(fixture.evaluation_paths["claim"].exists())

    def test_empty_partial_adapter_evaluation_root_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.prepare()
            fixture.evaluation_paths["root"].mkdir(parents=True)
            with self.assertRaisesRegex(adapter.AdapterError, "not resumable"):
                fixture.evaluate()
            self.assertFalse(fixture.evaluation_paths["claim"].exists())

    def test_claim_and_completion_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.evaluate()
            overwrite_sealed(
                fixture.evaluation_paths["claim"], {"upload_authorized": True}
            )
            with self.assertRaisesRegex(adapter.AdapterError, "claim content changed"):
                adapter.validate_evaluation_completion(
                    fixture.evaluation_paths["completion"],
                    adapter_plan_path=fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.evaluate()
            overwrite_sealed(
                fixture.evaluation_paths["completion"],
                {"upload_authorized": True},
            )
            with self.assertRaisesRegex(adapter.AdapterError, "completion content changed"):
                adapter.validate_evaluation_completion(
                    fixture.evaluation_paths["completion"],
                    adapter_plan_path=fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )

    def test_repeat_evaluate_is_idempotent_only_after_full_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            calls = []

            def evaluator(*_args):
                calls.append(1)
                return fixture.diagnostic()

            first = fixture.evaluate(evaluator)
            second = fixture.evaluate(evaluator)
            self.assertEqual(calls, [1])
            self.assertEqual(first, second)

    def test_direct_evaluation_after_completion_invalidates_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.evaluate()
            hidden = fixture.root / "fresh-holdout/reports/late.json"
            hidden.parent.mkdir()
            hidden.write_bytes(b"late")
            with self.assertRaisesRegex(adapter.AdapterError, "direct/legacy"):
                adapter.validate_evaluation_completion(
                    fixture.evaluation_paths["completion"],
                    adapter_plan_path=fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )

    def test_active_adapter_lock_rejects_second_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.prepare()
            with adapter._exclusive_adapter_lock(fixture.evaluation_paths["lock"]):
                with self.assertRaisesRegex(adapter.AdapterError, "is active"):
                    fixture.evaluate()

    def test_stale_lock_without_completion_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.prepare()
            fixture.evaluation_paths["lock"].touch()
            with self.assertRaisesRegex(adapter.AdapterError, "lock is not resumable"):
                fixture.evaluate()
            self.assertFalse(fixture.evaluation_paths["claim"].exists())

    def test_validation_detects_candidate_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.create()
            fixture.source_path.write_bytes(b"int main(){return 1;}\n")
            with self.assertRaises(adapter.AdapterError):
                adapter.validate_handoff(
                    fixture.output,
                    adapter_plan_path=fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    evaluation_completion_path=fixture.evaluation_paths["completion"],
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )

    def test_rejects_noncanonical_handoff_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AdapterFixture(pathlib.Path(temporary))
            fixture.evaluate()
            with self.assertRaisesRegex(adapter.AdapterError, "path is not canonical"):
                adapter.create_handoff(
                    fixture.root / "handoff.json",
                    adapter_plan_path=fixture.adapter_plan_path,
                    plan_path=fixture.plan_path,
                    output_root=fixture.root,
                    evaluation_completion_path=fixture.evaluation_paths["completion"],
                    created_at_utc="2026-09-01T12:01:00Z",
                    candidate_loader=fixture.candidate_loader,
                    retirement_validator=fixture.retirement_validator,
                )


if __name__ == "__main__":
    unittest.main()
