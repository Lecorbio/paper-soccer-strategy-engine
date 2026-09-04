from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/compact_value_bfm_rank4_teacher_dual_final.py"
SPEC = importlib.util.spec_from_file_location("rank4_teacher_dual_final_tests", TOOL)
assert SPEC is not None and SPEC.loader is not None
dual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dual)
q = dual.qualification


def write(path: pathlib.Path, schema: str, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    q.write_sealed(path, {"schema": schema, **fields})
    return path


def timing_receipt():
    samples = []
    for count in (1, 2, 10):
        for color in (0, 1):
            for replica in range(count):
                samples.append({
                    "process_count": count,
                    "color": color,
                    "replica": replica,
                    "first_ms": 100.0 + count,
                    "later_max_ms": 20.0 + color,
                    "stdout_sha256": "1" * 64,
                    "stderr_sha256": "2" * 64,
                })
    return {
        "schema": dual.deployment_preflight.maintained.TIMING_SCHEMA,
        "probe_sha256": "3" * 64,
        "first_limit_exclusive_ms": 900.0,
        "later_limit_exclusive_ms": 180.0,
        "samples": samples,
    }


def strict_summary(*, wins=527, color0=267, color1=260, failures=0):
    failure_map = {
        name: (failures if index == 0 else 0)
        for index, name in enumerate(q.FAILURE_CATEGORIES)
    }
    return {
        "games": 1_000,
        "candidate_wins": wins,
        "candidate_color_wins": {"0": color0, "1": color1},
        "failures": failure_map,
        "maximum_turns": 320,
        "timing": {"first_max_ms": 800.0, "later_max_ms": 155.0},
        "uncontended_timing": {"first_max_ms": 101.0, "later_max_ms": 21.0},
    }


def clean_process_audit(**changes):
    value = {
        "auditor_pid": 123,
        "process_nice": 0,
        "logical_cpu_count": 8,
        "load_average": {
            "one_minute": 1.0,
            "five_minutes": 1.5,
            "fifteen_minutes": 2.0,
        },
        "one_minute_load_limit_exclusive": 8.0,
        "competing_actual_clock_processes": [],
        "ps_stdout_sha256": "a" * 64,
        "ps_stderr_sha256": "b" * 64,
    }
    value.update(changes)
    return value


class PrepareFixture:
    def __init__(self, root: pathlib.Path, *, attempt: int = 0):
        self.root = root
        self.output = root / "execution"
        self.source = root / "candidate.cpp"
        self.source.write_text("int main(){return 0;}\n", encoding="ascii")
        self.runtime = root / "candidate.runtime.json"
        q.write_sealed(self.runtime, {
            "schema": "papersoccer.compact-value-bfm-runtime.v1",
            "architecture": {
                "dimensions": [6301, 12, 8, 1], "biases": False,
            },
            "quantization": {"payload_sha256": "4" * 64},
        })
        self.gate = root / "rank4-gate"
        self.gate.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
        self.gate.chmod(0o755)
        self.campaign = write(
            root / "campaign.json", dual.challenger.PLAN_SCHEMA,
            test_only_nonproduction=True,
            production_allowlist_enforced=False,
        )
        self.authorization_path = write(
            root / "authorization.json",
            dual.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
        )
        self.preflight_path = write(
            root / "preflight.json", dual.deployment_preflight.REFERENCE_SCHEMA
        )
        self.release_path = write(
            root / "release.json", dual.challenger.RELEASE_EVIDENCE_SCHEMA
        )
        self.ci_path = write(root / "ci.json", dual.upload.CI_SCHEMA)
        self.exclusion = write(
            root / "excluded.json", "test.fingerprint-exclusion.v1",
            fingerprints=["5" * 64],
        )
        self.commit = "a" * 40
        self.configuration = dual.deployment.deployment_configuration(
            ["0.95", "0.5", "1"], "default",
            dual.deployment.PROFILE_ROSTER["default"],
        )
        self.authorization = {
            "attempt": attempt,
            "created_at_utc": "2026-09-03T23:59:00Z",
            "candidate": {
                "source": dual._record(self.source, ascii_required=True),
                "runtime": dual._record(self.runtime),
                "architecture": {
                    "id": dual.ARCHITECTURE,
                    "dimensions": [6301, 12, 8, 1], "biases": False,
                    "outputs": 1, "head": "scalar-value-only",
                    "policy_head": False,
                    "runtime_body_sha256": json.loads(
                        self.runtime.read_bytes()
                    )["body_sha256"],
                    "payload_sha256": "4" * 64,
                },
            },
            "required_exclusion_sha256": [q.sha256_file(self.exclusion)],
            "release_evidence": {
                **dual._record(self.release_path),
                "schema": dual.challenger.RELEASE_EVIDENCE_SCHEMA,
                "body_sha256": q.load_sealed(
                    self.release_path, dual.challenger.RELEASE_EVIDENCE_SCHEMA
                )["body_sha256"],
            },
        }

    def authorization_validator(self, _authorization, _campaign):
        return {"authorization": self.authorization, "context": {}}

    def preflight_validator(self, _path):
        source = dual._record(self.source, ascii_required=True)
        runtime = dual._record(self.runtime, ascii_required=True)
        gate = dual._record(self.gate, executable=True)
        command = {
            "passed": True,
            "argv": [
                "/usr/bin/clang++",
                f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{source["path"]}"',
            ],
        }
        return {
            "reference": {"gate": gate},
            "receipt": {
                "commands": {"compile_rank4_gate": command},
                "binaries": {"rank4_gate": gate},
            },
            "plan": {
                "inputs": {
                    "repository": str(self.root),
                    "tools": {"clang": {
                        "path": "/usr/bin/clang++", "sha256": "6" * 64,
                    }},
                },
            },
            "candidate_commit": self.commit,
            "candidate": source,
            "runtime": runtime,
            "derivation": {"configuration": self.configuration},
            "timing": timing_receipt(),
            "ci": q.artifact_reference(self.ci_path, dual.upload.CI_SCHEMA),
        }

    def ci_validator(self, _path, expected_head):
        return {"head_sha": expected_head, "conclusion": "success"}

    def prepare(
        self, *, generalized_release: bool = False,
        output_root: pathlib.Path | None = None,
    ):
        return dual.prepare_execution(
            authorization_path=self.authorization_path,
            campaign_plan_path=self.campaign,
            output_root=self.output if output_root is None else output_root,
            deployment_preflight_path=(
                None if generalized_release else self.preflight_path
            ),
            release_evidence_path=(
                self.release_path if generalized_release else None
            ),
            ci_path=self.ci_path,
            rank4_source=ROOT / "submissions/codingame/bots/rank_4/submission.cpp",
            exclusion_paths=[self.exclusion],
            created_at_utc="2026-09-04T00:00:00Z",
            authorization_validator=self.authorization_validator,
            preflight_validator=self.preflight_validator,
            ci_validator=self.ci_validator,
            allow_injected_test_evidence=True,
        )


class DualFinalExecutionTests(unittest.TestCase):
    def test_release_adapter_uses_generated_source_not_deployed_source(self):
        selected = pathlib.Path("/frozen/generated.cpp")
        deployed = pathlib.Path("/release/deployed.cpp")
        observed = []

        def adapt(_path, **kwargs):
            observed.append(("adapt", kwargs["candidate_source"]))
            return {
                "candidate": {"path": str(deployed)},
                "timing": timing_receipt(),
                "uncontended_timing": {
                    "workers": 1,
                    "colors": [0, 1],
                    "first_max_ms": 101.0,
                    "later_max_ms": 21.0,
                    "first_limit_exclusive_ms": 900.0,
                    "later_limit_exclusive_ms": 180.0,
                },
            }

        release = __import__(
            "tools.compact_value_bfm_rank4_teacher_release", fromlist=["*"]
        )
        authorization = {
            "attempt": 5,
            "generated_source": {
                "route": "artifacts/generated.cpp", "bytes": 1,
                "sha256": "1" * 64,
            },
            "candidate": {
                "runtime": {"path": "/runtime"},
                "source": {"path": str(deployed)},
            },
        }
        with mock.patch.object(
            release, "dual_final_preflight_state", side_effect=adapt
        ), mock.patch.object(
            dual.challenger, "validate_campaign", return_value={"plan": {}}
        ), mock.patch.object(
            dual.challenger, "_resolve_campaign_artifact", return_value=selected
        ):
            state = dual._default_release_preflight_validator(
                pathlib.Path("/release-evidence.json"),
                campaign_plan_path=pathlib.Path("/campaign.json"),
                authorization=authorization,
            )
        self.assertEqual(state["candidate"]["path"], str(deployed))
        self.assertEqual(
            state["uncontended_timing"],
            {"first_max_ms": 101.0, "later_max_ms": 21.0},
        )
        self.assertEqual(observed, [("adapt", selected)])

    def test_authorized_exclusion_paths_are_discovered_from_full_ledger(self):
        records = {
            "protected": {
                "route": "artifacts/protected.json", "bytes": 1,
                "sha256": "1" * 64,
            },
            "live": {
                "route": "artifacts/live.json", "bytes": 1,
                "sha256": "2" * 64,
            },
            "development": {
                "path": "/development.json", "bytes": 1,
                "sha256": "3" * 64,
            },
            "dynamic": {
                "path": "/dynamic.json", "bytes": 1,
                "sha256": "4" * 64,
                "schema": dual.challenger.DYNAMIC_EXCLUSION_SCHEMA,
                "body_sha256": "5" * 64,
                "classification": "protected-final",
                "fingerprint_count": 500,
            },
        }
        context = {
            "inputs": {
                "protected_exclusions": {"protected": records["protected"]},
                "live_exclusions": {"live": records["live"]},
            },
            "plan": {},
        }
        entries = [{
            "event": "attempt-outcome-recorded",
            "development_exclusion": records["development"],
        }]
        def resolve(record, **_kwargs):
            return pathlib.Path(
                record.get("path", "/bundle/" + str(record.get("route")))
            )

        with mock.patch.object(dual.challenger, "load_ledger", return_value=entries), \
                mock.patch.object(
                    dual.challenger, "_cumulative_dynamic_exclusions",
                    return_value=[records["dynamic"]],
                ), mock.patch.object(
                    dual.challenger, "_resolve_campaign_artifact",
                    side_effect=resolve,
                ) as resolve_artifact, mock.patch.object(
                    dual.challenger, "_verify_dynamic_exclusion_record",
                    return_value=pathlib.Path("/dynamic.json"),
                ):
            paths = dual._discover_authorized_exclusions(
                {"context": context},
                {"required_exclusion_sha256": [
                    record["sha256"] for record in records.values()
                ]},
            )
        self.assertEqual(
            {str(path) for path in paths},
            {
                "/bundle/artifacts/protected.json",
                "/bundle/artifacts/live.json",
                "/development.json",
                "/dynamic.json",
            },
        )
        self.assertEqual(resolve_artifact.call_count, 3)

    def test_prepare_supports_attempt_zero_and_later_candidate(self):
        for attempt in (0, 7):
            with self.subTest(attempt=attempt), tempfile.TemporaryDirectory() as temporary:
                fixture = PrepareFixture(pathlib.Path(temporary), attempt=attempt)
                plan_path = fixture.prepare(generalized_release=attempt > 0)
                state = dual.validate_execution_plan(
                    plan_path,
                    authorization_validator=fixture.authorization_validator,
                    preflight_validator=fixture.preflight_validator,
                    ci_validator=fixture.ci_validator,
                    allow_injected_test_evidence=True,
                )
                self.assertEqual(state["plan"]["attempt"], attempt)
                self.assertEqual(
                    state["plan"]["candidate"]["source"]["sha256"],
                    q.sha256_file(fixture.source),
                )
                self.assertTrue(state["plan"]["compile_binding"]["candidate_embedded"])
                self.assertNotEqual(
                    state["plan"]["gate"]["path"],
                    state["plan"]["gate_source"]["path"],
                )
                self.assertEqual(
                    state["plan"]["gate"]["sha256"],
                    state["plan"]["gate_source"]["sha256"],
                )
                self.assertEqual(
                    pathlib.Path(state["plan"]["gate"]["path"]).stat().st_mode
                    & 0o222,
                    0,
                )
                self.assertEqual(
                    state["plan"]["preflight_kind"],
                    "rank4-teacher-release" if attempt > 0
                    else "discrete-v3-deployment-preflight",
                )
                self.assertEqual(state["plan"]["gate_contract"]["workers_per_gate"], 4)
                self.assertEqual(state["plan"]["gate_contract"]["shards_per_gate"], 100)
                self.assertEqual(
                    state["plan"]["root"],
                    str(fixture.authorization_path.resolve().parent / "execution"),
                )
                self.assertTrue(
                    state["plan"]["execution_identity"]["one_execution_root"]
                )
                self.assertEqual(
                    state["plan"]["campaign_heavy_stage_lock"],
                    str(root := fixture.campaign.resolve().parent / ".rank4-teacher-heavy-stage.lock"),
                )
                self.assertEqual(root.parent, fixture.root.resolve())

    def test_prepare_accepts_plain_authorized_source_record_but_requires_ascii(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            fixture.authorization["candidate"]["source"] = dual._record(
                fixture.source
            )
            self.assertNotIn(
                "ascii", fixture.authorization["candidate"]["source"]
            )
            plan_path = fixture.prepare()
            self.assertEqual(fixture.prepare(), plan_path)
            state = dual.validate_execution_plan(
                plan_path,
                authorization_validator=fixture.authorization_validator,
                preflight_validator=fixture.preflight_validator,
                ci_validator=fixture.ci_validator,
                allow_injected_test_evidence=True,
            )
            self.assertNotIn("ascii", state["plan"]["candidate"]["source"])

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            fixture.source.write_bytes(b"int main(){}\n// \xff\n")
            fixture.authorization["candidate"]["source"] = dual._record(
                fixture.source
            )
            with self.assertRaisesRegex(
                dual.DualFinalError,
                "authorized candidate source.*not ASCII",
            ):
                fixture.prepare()

    def test_generalized_release_detailed_timing_is_compacted_in_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary), attempt=1)
            release = __import__(
                "tools.compact_value_bfm_rank4_teacher_release", fromlist=["*"]
            )
            adapted = fixture.preflight_validator(fixture.release_path)
            detailed = release._uncontended_timing(adapted["timing"])
            compact = dual._uncontended_timing(adapted["timing"])
            self.assertEqual(
                set(detailed),
                {
                    "workers", "colors", "first_max_ms", "later_max_ms",
                    "first_limit_exclusive_ms", "later_limit_exclusive_ms",
                },
            )
            self.assertEqual(
                compact,
                {
                    "first_max_ms": detailed["first_max_ms"],
                    "later_max_ms": detailed["later_max_ms"],
                },
            )
            adapted["uncontended_timing"] = detailed
            fixture.authorization["generated_source"] = dual._record(
                fixture.source
            )

            def normalized_preflight(path):
                with mock.patch.object(
                    dual.challenger, "validate_campaign",
                    return_value={"plan": {}},
                ), mock.patch.object(
                    dual.challenger, "_resolve_campaign_artifact",
                    return_value=fixture.source.resolve(),
                ), mock.patch.object(
                    release, "dual_final_preflight_state", return_value=adapted,
                ):
                    return dual._default_release_preflight_validator(
                        path,
                        campaign_plan_path=fixture.campaign,
                        authorization=fixture.authorization,
                    )

            fixture.preflight_validator = normalized_preflight
            plan_path = fixture.prepare(generalized_release=True)

            plan = q.load_sealed(plan_path, dual.PLAN_SCHEMA)
            self.assertEqual(plan["uncontended_timing"], compact)
            self.assertEqual(adapted["uncontended_timing"], detailed)

    def test_authorization_has_one_deterministic_execution_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(
                dual.DualFinalError, "deterministic campaign-attempt execution root"
            ):
                fixture.prepare(output_root=fixture.root / "another-execution")
            plan_path = fixture.prepare()
            plan = q.load_sealed(plan_path, dual.PLAN_SCHEMA)
            self.assertEqual(
                plan["execution_identity"],
                dual._execution_identity(
                    campaign_plan_path=fixture.campaign,
                    authorization_path=fixture.authorization_path,
                    attempt=0, root=fixture.output,
                ),
            )

    def test_production_rejects_all_injected_prepare_and_run_callables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            production_context = {
                "inputs": {"production_allowlist_enforced": True},
                "plan": {"outputs": {"dual_final": str(root / "dual-final")}},
            }
            noop = lambda *_args, **_kwargs: {}
            with mock.patch.object(
                dual.challenger, "validate_campaign",
                return_value=production_context,
            ), self.assertRaises(dual.DualFinalError) as caught:
                dual.prepare_execution(
                    authorization_path=root / "authorization.json",
                    campaign_plan_path=campaign,
                    output_root=root / "execution",
                    deployment_preflight_path=root / "preflight.json",
                    ci_path=root / "ci.json", rank4_source=root / "rank4.cpp",
                    exclusion_paths=[], created_at_utc="2026-09-04T00:00:00Z",
                    authorization_validator=noop,
                    preflight_validator=noop, ci_validator=noop,
                    fingerprint_loader=noop,
                    allow_injected_test_evidence=True,
                )
            for name in (
                "authorization_validator", "preflight_validator",
                "ci_validator", "fingerprint_loader",
            ):
                self.assertIn(name, str(caught.exception))

            plan_path = write(
                root / "execution-plan.json", dual.PLAN_SCHEMA,
                production=True,
            )
            with self.assertRaises(dual.DualFinalError) as caught:
                dual.materialize_banks(
                    plan_path, claimed_at_utc="2026-09-04T00:00:00Z",
                    entropy=noop, bank_generator=noop, state_validator=noop,
                    governance_preparer=noop,
                    allow_injected_test_evidence=True,
                )
            for name in (
                "entropy", "bank_generator", "state_validator",
                "governance_preparer",
            ):
                self.assertIn(name, str(caught.exception))

            with self.assertRaises(dual.DualFinalError) as caught:
                dual.run_dual_final(
                    plan_path, launched_at_utc="2026-09-04T00:00:00Z",
                    runner=noop, result_validator=noop, clock=noop,
                    state_validator=noop, result_recorder=noop,
                    dual_completer=noop, executor_factory=noop,
                    process_auditor=noop,
                    allow_injected_test_evidence=True,
                )
            for name in (
                "runner", "result_validator", "clock", "state_validator",
                "result_recorder", "dual_completer", "executor_factory",
                "process_auditor",
            ):
                self.assertIn(name, str(caught.exception))

    def test_nonproduction_injection_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = write(
                pathlib.Path(temporary) / "execution-plan.json",
                dual.PLAN_SCHEMA, production=False,
            )
            with self.assertRaisesRegex(
                dual.DualFinalError, "without explicit test opt-in"
            ):
                dual.materialize_banks(
                    plan_path, claimed_at_utc="2026-09-04T00:00:00Z",
                    state_validator=lambda _path: {},
                )
            forged_state = {
                "path": plan_path,
                "plan": {
                    **q.load_sealed(plan_path, dual.PLAN_SCHEMA),
                    "production": True,
                    "root": str(pathlib.Path(temporary) / "production-root"),
                },
            }
            with self.assertRaisesRegex(
                dual.DualFinalError, "another execution identity"
            ):
                dual.materialize_banks(
                    plan_path, claimed_at_utc="2026-09-04T00:00:00Z",
                    state_validator=lambda _path: forged_state,
                    allow_injected_test_evidence=True,
                )

    def test_prepare_rejects_incomplete_exclusions_and_candidate_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            fixture.authorization["required_exclusion_sha256"].append("7" * 64)
            with self.assertRaisesRegex(dual.DualFinalError, "every authorized"):
                fixture.prepare()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            plan = fixture.prepare()
            fixture.source.write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(dual.DualFinalError, "candidate source.*changed"):
                dual.validate_execution_plan(
                    plan,
                    authorization_validator=fixture.authorization_validator,
                    preflight_validator=fixture.preflight_validator,
                    ci_validator=fixture.ci_validator,
                    allow_injected_test_evidence=True,
                )

    def test_materialization_orders_a_then_b_and_requires_disjoint_fresh_banks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_file = write(root / "plan.json", dual.PLAN_SCHEMA)
            authorization = write(
                root / "authorization.json",
                dual.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
            )
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            dual_plan = write(root / "dual-plan.json", dual.challenger.DUAL_FINAL_SCHEMA)
            dual_reference = write(
                root / "governance-reference.json",
                dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA,
                dual_final_plan=q.artifact_reference(
                    dual_plan, dual.challenger.DUAL_FINAL_SCHEMA
                ),
            )
            state = {
                "path": plan_file,
                "plan": {
                    "root": str(root), "attempt": 2,
                    "execution_identity": {
                        "campaign_plan_body_sha256": q.load_sealed(
                            campaign, dual.challenger.PLAN_SCHEMA
                        )["body_sha256"],
                        "authorization_body_sha256": q.load_sealed(
                            authorization,
                            dual.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
                        )["body_sha256"],
                        "attempt": 2, "root": str(root),
                        "one_execution_root": True,
                    },
                    "authorization": q.artifact_reference(
                        authorization, dual.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
                    ),
                    "campaign_plan": q.artifact_reference(
                        campaign, dual.challenger.PLAN_SCHEMA
                    ),
                },
            }
            plan_file.write_bytes(q.canonical_json_bytes(q.seal({
                "schema": dual.PLAN_SCHEMA, **state["plan"],
            })))
            state["plan"] = q.load_sealed(plan_file, dual.PLAN_SCHEMA)
            first_bank_path = root / "gate-a.bank"
            second_bank_path = root / "gate-b.bank"
            first_bank_path.write_bytes(b"a")
            second_bank_path.write_bytes(b"b")
            first_record = dual._record(first_bank_path)
            second_record = dual._record(second_bank_path)
            first_exclusion = write(
                root / "a-exclusion.json", dual.FINGERPRINT_EXCLUSION_SCHEMA
            )
            second_exclusion = write(
                root / "b-exclusion.json", dual.FINGERPRINT_EXCLUSION_SCHEMA
            )
            bank_a = {
                "openings": [{"fingerprints": {"canonical": "1" * 64}}],
            }
            bank_b = {
                "openings": [{"fingerprints": {"canonical": "2" * 64}}],
            }
            states = {
                "gate-a": {
                    "seed": {"seed_256_hex": "01" * 32}, "bank": bank_a,
                    "bank_path": first_bank_path,
                    "fingerprint_exclusion_path": first_exclusion,
                    "path": write(
                        root / "gate-a.receipt.json", dual.BANK_RECEIPT_SCHEMA
                    ),
                    "receipt": {
                        "protected_bank": first_record,
                        "gate_bank": first_record,
                        "exclusion_sources": [],
                    },
                },
                "gate-b": {
                    "seed": {"seed_256_hex": "02" * 32}, "bank": bank_b,
                    "bank_path": second_bank_path,
                    "fingerprint_exclusion_path": second_exclusion,
                    "path": write(
                        root / "gate-b.receipt.json", dual.BANK_RECEIPT_SCHEMA
                    ),
                    "receipt": {
                        "protected_bank": second_record,
                        "gate_bank": second_record,
                        "exclusion_sources": [first_record],
                    },
                },
            }
            order = []

            def materialize(_state, *, gate_id, **_kwargs):
                order.append(gate_id)
                return root / f"{gate_id}.receipt"

            def prepare(*_args, **kwargs):
                self.assertEqual(kwargs["bank_a"], first_bank_path)
                self.assertEqual(kwargs["bank_b"], second_bank_path)
                return dual_reference

            with mock.patch.object(dual, "_materialize_one", side_effect=materialize), \
                    mock.patch.object(
                        dual, "validate_bank_receipt",
                        side_effect=lambda _state, *, gate_id: states[gate_id],
                    ):
                result = dual.materialize_banks(
                    plan_file, claimed_at_utc="2026-09-04T00:01:00Z",
                    state_validator=lambda _path: state,
                    governance_preparer=prepare,
                    allow_injected_test_evidence=True,
                )
            self.assertEqual(result, dual_reference)
            self.assertEqual(order, ["gate-a", "gate-b"])
            prepared = q.load_sealed(root / "prepared.json", dual.PREPARED_SCHEMA)
            self.assertEqual(set(prepared["fingerprint_exclusions"]), {"gate-a", "gate-b"})
            self.assertEqual(prepared["games_launched"], 0)

            states["gate-b"]["seed"] = {"seed_256_hex": "01" * 32}
            (root / "prepared.json").unlink()
            with mock.patch.object(dual, "_materialize_one", side_effect=materialize), \
                    mock.patch.object(
                        dual, "validate_bank_receipt",
                        side_effect=lambda _state, *, gate_id: states[gate_id],
                    ), self.assertRaisesRegex(dual.DualFinalError, "reused entropy"):
                dual.materialize_banks(
                    plan_file, claimed_at_utc="2026-09-04T00:01:00Z",
                    state_validator=lambda _path: state,
                    governance_preparer=prepare,
                    allow_injected_test_evidence=True,
                )
            states["gate-b"]["seed"] = {"seed_256_hex": "02" * 32}
            states["gate-b"]["bank"] = bank_a
            with mock.patch.object(dual, "_materialize_one", side_effect=materialize), \
                    mock.patch.object(
                        dual, "validate_bank_receipt",
                        side_effect=lambda _state, *, gate_id: states[gate_id],
                    ), self.assertRaisesRegex(dual.DualFinalError, "overlap by symmetry"):
                dual.materialize_banks(
                    plan_file, claimed_at_utc="2026-09-04T00:01:00Z",
                    state_validator=lambda _path: state,
                    governance_preparer=prepare,
                    allow_injected_test_evidence=True,
                )

    def test_started_bank_without_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            gate_root = root / "gates/gate-a"
            gate_root.mkdir(parents=True)
            write(gate_root / "bank-claim.json", dual.BANK_CLAIM_SCHEMA)
            state = {"path": root / "plan.json", "plan": {"root": str(root)}}
            with self.assertRaisesRegex(dual.DualFinalError, "retry forbidden"):
                dual._materialize_one(
                    state, gate_id="gate-a",
                    claimed_at_utc="2026-09-04T00:00:00Z",
                    entropy=lambda _size: b"x" * 32,
                    bank_generator=lambda **_kwargs: [],
                    allow_injected_test_evidence=True,
                )

    def test_abandonment_recovers_fingerprints_when_bank_receipt_was_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            plan_path = fixture.prepare()
            state = dual.validate_execution_plan(
                plan_path,
                authorization_validator=fixture.authorization_validator,
                preflight_validator=fixture.preflight_validator,
                ci_validator=fixture.ci_validator,
                allow_injected_test_evidence=True,
            )
            dual._materialize_one(
                state, gate_id="gate-a",
                claimed_at_utc="2026-09-04T00:01:00Z",
                entropy=lambda _size: b"x" * 32,
                bank_generator=dual.openings.generate_openings,
                allow_injected_test_evidence=True,
            )
            gate_root = pathlib.Path(state["plan"]["root"]) / "gates/gate-a"
            (gate_root / "bank-receipt.json").unlink()
            (gate_root / "fingerprint-exclusion.json").unlink()
            derived = dual._derive_protected_abortion(
                state, ensure_exclusions=True
            )
            self.assertEqual(derived["protected_stage"], "bank-materialization")
            self.assertEqual(derived["gate_id"], "gate-a")
            self.assertEqual(len(derived["fingerprint_exclusions"]), 1)
            exclusion_path = pathlib.Path(
                derived["fingerprint_exclusions"][0]["path"]
            )
            exclusion = q.load_sealed(
                exclusion_path, dual.FINGERPRINT_EXCLUSION_SCHEMA
            )
            self.assertEqual(exclusion["fingerprint_count"], 500)
            self.assertFalse(exclusion["contains_transcripts"])

    def test_prepared_resume_binds_the_exact_recorded_banks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            dual_reference = write(
                root / "dual-reference.json",
                dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA,
            )
            state = {
                "path": plan_path,
                "plan": {
                    "root": str(root), "attempt": 3,
                    "campaign_plan": q.artifact_reference(
                        campaign, dual.challenger.PLAN_SCHEMA
                    ),
                    "execution_identity": {"attempt": 3},
                },
            }
            bank_states = {}
            exclusions = {}
            expected_banks = {}
            governance_gates = []
            for index, gate_id in enumerate(dual.GATE_IDS):
                protected = root / f"{gate_id}.protected"
                protected.write_bytes(bytes([index + 1]))
                gate_bank = root / f"{gate_id}.tsv"
                gate_bank.write_bytes(bytes([index + 11]))
                receipt_path = write(
                    root / f"{gate_id}.receipt.json", dual.BANK_RECEIPT_SCHEMA
                )
                fingerprint = write(
                    root / f"{gate_id}.fingerprints.json",
                    dual.FINGERPRINT_EXCLUSION_SCHEMA,
                )
                receipt = {
                    "protected_bank": dual._record(protected),
                    "gate_bank": dual._record(gate_bank),
                }
                bank_states[gate_id] = {
                    "path": receipt_path, "receipt": receipt,
                    "fingerprint_exclusion_path": fingerprint,
                }
                exclusions[gate_id] = q.artifact_reference(
                    fingerprint, dual.FINGERPRINT_EXCLUSION_SCHEMA
                )
                expected_banks[gate_id] = {
                    "bank_receipt": q.artifact_reference(
                        receipt_path, dual.BANK_RECEIPT_SCHEMA
                    ),
                    **receipt,
                }
                governance_gates.append({
                    "gate_id": gate_id, "bank": receipt["protected_bank"]
                })
            prepared_path = write(
                root / "prepared.json", dual.PREPARED_SCHEMA,
                namespace=dual.NAMESPACE, campaign_id=dual.CAMPAIGN_ID,
                attempt=3,
                execution_plan=q.artifact_reference(plan_path, dual.PLAN_SCHEMA),
                execution_identity={"attempt": 3},
                dual_final_reference=q.artifact_reference(
                    dual_reference, dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA
                ),
                fingerprint_exclusions=exclusions, banks=expected_banks,
                candidate_unchanged=True, independent_banks=True,
                gate_b_excludes_gate_a=True, games_launched=0,
            )
            dual_state = {"plan": {"gates": governance_gates}}
            with mock.patch.object(
                dual, "validate_bank_receipt",
                side_effect=lambda _state, *, gate_id: bank_states[gate_id],
            ), mock.patch.object(
                dual.challenger, "validate_dual_final", return_value=dual_state,
            ):
                self.assertEqual(dual._load_prepared(state)[0], dual_reference.resolve())
                forged = q.load_sealed(prepared_path, dual.PREPARED_SCHEMA)
                forged = {key: value for key, value in forged.items() if key != "body_sha256"}
                forged["banks"] = {
                    **forged["banks"],
                    "gate-b": {
                        **forged["banks"]["gate-b"],
                        "protected_bank": forged["banks"]["gate-a"]["protected_bank"],
                    },
                }
                prepared_path.write_bytes(q.canonical_json_bytes(q.seal(forged)))
                with self.assertRaisesRegex(
                    dual.DualFinalError, "prepared receipt changed"
                ):
                    dual._load_prepared(state)

    def test_shared_heavy_stage_lock_and_sealed_prelaunch_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            execution = root / "execution"
            execution.mkdir()
            plan_path = write(execution / "execution-plan.json", dual.PLAN_SCHEMA)
            gate = root / "rank4-gate"
            gate.write_text("gate", encoding="ascii")
            receipt_path = write(
                execution / "gate-a.receipt.json", dual.BANK_RECEIPT_SCHEMA
            )
            state = {
                "path": plan_path,
                "plan": {
                    "root": str(execution), "attempt": 2,
                    "campaign_plan": q.artifact_reference(
                        campaign, dual.challenger.PLAN_SCHEMA
                    ),
                    "campaign_heavy_stage_lock": str(
                        root / ".rank4-teacher-heavy-stage.lock"
                    ),
                    "gate": {"path": str(gate)},
                },
            }
            bank_state = {"path": receipt_path}
            with dual._exclusive_heavy_stage_lock(state) as lock:
                with self.assertRaisesRegex(
                    dual.DualFinalError, "another campaign heavy stage"
                ):
                    with dual._exclusive_heavy_stage_lock(state):
                        pass
                audit_path = dual._seal_prelaunch_audit(
                    state, bank_state=bank_state, gate_id="gate-a",
                    lock_evidence=lock,
                    process_auditor=lambda _binary: clean_process_audit(),
                    clock=lambda: "2026-09-04T00:00:00Z",
                )
            audit = dual._validate_prelaunch_audit(
                audit_path, state=state, bank_state=bank_state,
                gate_id="gate-a",
            )
            self.assertEqual(audit["workers"], 4)
            self.assertEqual(audit["threads_per_worker"], 1)
            self.assertEqual(
                audit["campaign_heavy_stage_lock"]["path"],
                str((root / ".rank4-teacher-heavy-stage.lock").resolve()),
            )

    def test_prelaunch_audit_rejects_nice_load_and_competing_gate_processes(self):
        for changed in (
            {"process_nice": 1},
            {"load_average": {
                "one_minute": 8.0, "five_minutes": 1.0,
                "fifteen_minutes": 1.0,
            }},
            {"competing_actual_clock_processes": [{"pid": 999}]},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                dual.DualFinalError, "not clean"
            ):
                dual._validate_process_audit(clean_process_audit(**changed))

        process = __import__("subprocess").CompletedProcess(
            [], 0,
            stdout=(
                "99999 1 0 python compact_value_bfm_discrete_v3_recovery_runner.py\n"
            ),
            stderr="",
        )
        with mock.patch.object(dual.subprocess, "run", return_value=process), \
                mock.patch.object(dual.os, "getpriority", return_value=0), \
                mock.patch.object(dual.os, "getloadavg", return_value=(1, 1, 1)), \
                mock.patch.object(dual.os, "cpu_count", return_value=8):
            observed = dual._default_prelaunch_process_audit(pathlib.Path("/gate"))
        self.assertEqual(
            [item["pid"] for item in observed["competing_actual_clock_processes"]],
            [99999],
        )
        with self.assertRaisesRegex(dual.DualFinalError, "not clean"):
            dual._validate_process_audit(observed)

    def test_started_shard_without_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            claim = root / "gates/gate-a/ledger/claims/shard-000.json"
            write(claim, q.SHARD_CLAIM_SCHEMA)
            state = {"path": root / "plan.json", "plan": {"root": str(root)}}
            with self.assertRaisesRegex(q.SpentShardError, "retry forbidden"):
                dual._audit_shards(
                    state, {"binding_path": root / "binding.json"},
                    gate_id="gate-a", result_validator=lambda *_a, **_k: {},
                )

    def test_spent_shard_derives_metric_free_abort_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            binding_path = write(
                root / "gate-binding.json", q.GATE_BINDING_SCHEMA,
                namespace=q.NAMESPACE,
                candidate={"sha256": "1" * 64},
                opponent={"sha256": "2" * 64},
                harness={"sha256": "3" * 64},
                bank={"sha256": "4" * 64},
            )
            binding = q.load_sealed(binding_path, q.GATE_BINDING_SCHEMA)
            state = {
                "path": plan_path,
                "fingerprints": set(),
                "plan": {
                    "root": str(root), "attempt": 1,
                    "uncontended_timing": {},
                },
            }
            write(root / "prepared.json", dual.PREPARED_SCHEMA)
            ledger = root / "gates/gate-a/ledger"
            q.start_final_shard(
                ledger, binding_path=binding_path, index=0,
                started_at_utc="2026-09-04T00:00:00Z",
            )
            raw = ledger / "raw/shard-000.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(b"partial")
            bank_state = {
                "binding_path": binding_path,
                "binding": binding,
            }
            with (
                mock.patch.object(
                    dual, "_materialized_fingerprint_exclusions",
                    return_value=[],
                ),
                mock.patch.object(
                    dual, "validate_bank_receipt", return_value=bank_state
                ),
            ):
                derived = dual._derive_protected_abortion(
                    state, ensure_exclusions=False
                )
            self.assertEqual(derived["protected_stage"], "shard-execution")
            self.assertEqual(derived["gate_id"], "gate-a")
            self.assertEqual(derived["shard_index"], 0)
            self.assertIsNotNone(derived["spent_claim"])
            self.assertEqual(derived["partial_raw"], dual._record(raw))

    def test_protected_abort_receipt_is_idempotent_and_calls_governance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            execution_root = root / "execution"
            execution_root.mkdir()
            source_binding = write(
                root / "source-binding.json", q.SOURCE_BINDING_SCHEMA
            )
            gate = root / "gate"
            gate.write_text("#!/bin/sh\n", encoding="ascii")
            plan_path = execution_root / "execution-plan.json"
            plan = {
                "schema": dual.PLAN_SCHEMA,
                "root": str(execution_root),
                "attempt": 2,
                "production": False,
                "campaign_plan": q.artifact_reference(
                    campaign, dual.challenger.PLAN_SCHEMA
                ),
                "campaign_heavy_stage_lock": str(
                    root / ".rank4-teacher-heavy-stage.lock"
                ),
                "execution_identity": {"attempt": 2},
                "source_binding": q.artifact_reference(
                    source_binding, q.SOURCE_BINDING_SCHEMA
                ),
                "candidate": {
                    "runtime": {"sha256": "1" * 64},
                    "source": {"sha256": "2" * 64},
                },
                "candidate_commit": "3" * 40,
                "gate": {"path": str(gate.resolve())},
            }
            q.write_sealed(plan_path, plan)
            loaded = q.load_sealed(plan_path, dual.PLAN_SCHEMA)
            state = {"path": plan_path, "plan": loaded}
            derived = {
                "protected_stage": "shard-execution",
                "gate_id": "gate-a", "shard_index": 0,
                "spent_claim": None,
                "partial_raw": None, "invalid_receipt": None,
                "fingerprint_exclusions": [],
            }
            calls = []
            def record(*args, **kwargs):
                calls.append((args, kwargs))
                return {"event": "protected-stage-aborted"}
            state_loader = lambda _path: state
            auditor = lambda _gate: clean_process_audit()
            with mock.patch.object(
                dual, "_derive_protected_abortion", return_value=derived
            ):
                first = dual.abandon_protected_stage(
                    plan_path, aborted_at_utc="2026-09-04T00:00:01Z",
                    state_validator=state_loader,
                    governance_recorder=record,
                    process_auditor=auditor,
                    allow_injected_test_evidence=True,
                )
                second = dual.abandon_protected_stage(
                    plan_path, aborted_at_utc="2026-09-04T00:00:02Z",
                    state_validator=state_loader,
                    governance_recorder=record,
                    process_auditor=auditor,
                    allow_injected_test_evidence=True,
                )
            self.assertEqual(first, second)
            receipt = q.load_sealed(first, dual.PROTECTED_STAGE_ABORTION_SCHEMA)
            self.assertFalse(receipt["partial_metrics_read"])
            self.assertFalse(receipt["retry_authorized"])
            self.assertTrue(receipt["candidate_rejected"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                calls[1][1]["created_at_utc"], receipt["aborted_at_utc"]
            )

    def test_protected_abandonment_rejects_an_active_gate_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PrepareFixture(pathlib.Path(temporary))
            plan_path = fixture.prepare()
            state = dual.validate_execution_plan(
                plan_path,
                authorization_validator=fixture.authorization_validator,
                preflight_validator=fixture.preflight_validator,
                ci_validator=fixture.ci_validator,
                allow_injected_test_evidence=True,
            )
            lock_path = pathlib.Path(
                state["plan"]["campaign_heavy_stage_lock"]
            )
            descriptor = dual.os.open(
                lock_path, dual.os.O_CREAT | dual.os.O_RDWR, 0o600
            )
            dual.fcntl.flock(
                descriptor, dual.fcntl.LOCK_EX | dual.fcntl.LOCK_NB
            )
            derive = mock.Mock()
            try:
                with (
                    mock.patch.object(
                        dual, "_derive_protected_abortion", derive
                    ),
                    self.assertRaisesRegex(
                        dual.DualFinalError,
                        "another campaign heavy stage is active",
                    ),
                ):
                    dual.abandon_protected_stage(
                        plan_path,
                        aborted_at_utc="2026-09-04T00:00:01Z",
                        state_validator=lambda _path: state,
                        governance_recorder=lambda *_args, **_kwargs: {},
                        allow_injected_test_evidence=True,
                    )
            finally:
                dual.fcntl.flock(descriptor, dual.fcntl.LOCK_UN)
                dual.os.close(descriptor)
            derive.assert_not_called()
            self.assertFalse(
                (pathlib.Path(state["plan"]["root"])
                 / "protected-stage-aborted.json").exists()
            )
            active_process = clean_process_audit(
                competing_actual_clock_processes=[{
                    "pid": 999, "ppid": 1, "nice": 0,
                    "command_sha256": "f" * 64,
                }]
            )
            derive = mock.Mock()
            with (
                mock.patch.object(dual, "_derive_protected_abortion", derive),
                self.assertRaisesRegex(dual.DualFinalError, "not clean"),
            ):
                dual.abandon_protected_stage(
                    plan_path,
                    aborted_at_utc="2026-09-04T00:00:02Z",
                    state_validator=lambda _path: state,
                    governance_recorder=lambda *_args, **_kwargs: {},
                    process_auditor=lambda _gate: active_process,
                    allow_injected_test_evidence=True,
                )
            derive.assert_not_called()
            self.assertFalse(
                (pathlib.Path(state["plan"]["root"])
                 / "protected-stage-aborted.json").exists()
            )

    def test_shard_audit_rejects_redirected_evidence_and_raw_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            configuration = dual.deployment.deployment_configuration(
                ["0.95", "0.5", "1"], "default",
                dual.deployment.PROFILE_ROSTER["default"],
            )
            gate_path = root / "frozen-gate"
            gate_path.write_text("#!/bin/sh\n", encoding="ascii")
            gate_path.chmod(0o500)
            plan = {
                "root": str(root), "attempt": 1,
                "candidate_search_profile": "standard-v1",
                "candidate": {"source": {"sha256": "1" * 64}},
                "runtime_identity": {
                    "runtime_body_sha256": "2" * 64,
                    "payload_sha256": "3" * 64,
                },
                "configuration": configuration,
                "gate": dual._record(gate_path, executable=True),
            }
            state = {"path": plan_path, "plan": plan}
            ledger = root / "gates/gate-a/ledger"
            for name in ("claims", "receipts", "raw", "raw-evidence"):
                (ledger / name).mkdir(parents=True, exist_ok=True)
            binding_path = write(
                root / "binding.json", q.GATE_BINDING_SCHEMA,
                namespace=q.NAMESPACE,
                candidate={"sha256": "1" * 64},
                opponent={"sha256": "4" * 64},
                harness={"sha256": "5" * 64},
                bank={"sha256": "6" * 64},
            )
            binding = q.load_sealed(binding_path, q.GATE_BINDING_SCHEMA)
            bank_receipt_path = write(
                root / "bank-receipt.json", dual.BANK_RECEIPT_SCHEMA
            )
            bank_state = {
                "path": bank_receipt_path, "binding_path": binding_path,
                "binding": binding,
                "receipt": {"gate_bank": {"sha256": "7" * 64}},
            }
            prelaunch = write(
                ledger / "prelaunch-audits/audit-000.json",
                dual.PRELAUNCH_AUDIT_SCHEMA,
            )
            raw_path = ledger / "raw/shard-000.json"
            raw_path.write_text("{}", encoding="ascii")
            games = [
                {
                    "pair_index": pair, "candidate_player": color,
                    "failure": None, "winner": color, "turns": 1,
                    "candidate": {
                        "maximum_first_ms": 1.0,
                        "maximum_later_ms": 1.0,
                    },
                }
                for pair in range(5) for color in (0, 1)
            ]
            document = {
                "bindings": {
                    "candidate_runtime_body_sha256": "2" * 64,
                    "candidate_payload_sha256": "3" * 64,
                },
                "config": dual._expected_gate_configuration(plan, pair_offset=0),
                "games": games,
            }
            normalized = dual._adapt_result(
                raw_path, plan=plan, bank=bank_state["receipt"], index=0,
                result_validator=lambda *_args, **_kwargs: document,
            )
            q.start_final_shard(
                ledger, binding_path=binding_path, index=0,
                started_at_utc="2026-09-04T00:00:01Z",
            )
            evidence_path = write(
                ledger / "raw-evidence/shard-000.json",
                dual.RAW_EVIDENCE_SCHEMA,
                namespace=dual.NAMESPACE, campaign_id=dual.CAMPAIGN_ID,
                attempt=1, gate_id="gate-a",
                execution_plan=q.artifact_reference(plan_path, dual.PLAN_SCHEMA),
                bank_receipt=q.artifact_reference(
                    bank_receipt_path, dual.BANK_RECEIPT_SCHEMA
                ),
                prelaunch_audit=q.artifact_reference(
                    prelaunch, dual.PRELAUNCH_AUDIT_SCHEMA
                ),
                gate_executable=plan["gate"],
                shard_index=0,
                actual_clock_configuration=dual._expected_gate_configuration(
                    plan, pair_offset=0
                ),
                raw_gate_result=dual._record(raw_path),
                normalized_games_sha256=q.sha256_bytes(
                    q.canonical_json_bytes(normalized)
                ),
            )
            receipt = q.record_shard_receipt(
                ledger, binding_path=binding_path, index=0, games=normalized,
                completed_at_utc="2026-09-04T00:00:02Z",
                evidence=q.artifact_reference(
                    evidence_path, dual.RAW_EVIDENCE_SCHEMA
                ),
            )
            validator = lambda *_args, **_kwargs: document
            with mock.patch.object(
                dual, "_validate_prelaunch_audit",
                return_value={"audited_at_utc": "2026-09-04T00:00:00Z"},
            ), mock.patch.object(
                dual, "_verify_frozen_gate", return_value=gate_path,
            ):
                self.assertEqual(
                    dual._audit_shards(
                        state, bank_state, gate_id="gate-a",
                        result_validator=validator,
                    ),
                    list(range(1, 100)),
                )
                alternate = write(
                    root / "alternate-evidence.json", dual.RAW_EVIDENCE_SCHEMA
                )
                forged = {
                    key: value for key, value in receipt.items()
                    if key != "body_sha256"
                }
                forged["evidence"] = q.artifact_reference(
                    alternate, dual.RAW_EVIDENCE_SCHEMA
                )
                receipt_path = ledger / "receipts/shard-000.json"
                receipt_path.write_bytes(q.canonical_json_bytes(q.seal(forged)))
                with self.assertRaisesRegex(
                    dual.DualFinalError, "evidence route changed"
                ):
                    dual._audit_shards(
                        state, bank_state, gate_id="gate-a",
                        result_validator=validator,
                    )

    def test_sanitized_bank_exclusion_contains_only_canonical_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            bank_path = root / "bank.json"
            bank_path.write_bytes(b"protected bank identity")
            runtime_sha = "a" * 64
            source_sha = "b" * 64
            state = {"plan": {
                "root": str(root), "attempt": 3,
                "candidate": {
                    "runtime": {"sha256": runtime_sha},
                    "source": {"sha256": source_sha},
                },
            }}
            canonical = [
                hashlib.sha256(f"state-{index}".encode()).hexdigest()
                for index in range(500)
            ]
            bank = {
                "seed_hex": "01" * 32,
                "seed_receipt": {"path": "/sealed/seed", "sha256": "c" * 64},
                "openings": [
                    {"fingerprints": {
                        "exact": value, "rotate": value,
                        "reflect": value, "rotate_reflect": value,
                        "canonical": value,
                    }} for value in canonical
                ],
            }
            path = dual._write_fingerprint_exclusion(
                state, gate_id="gate-a", bank_path=bank_path, bank=bank,
            )
            value = q.load_sealed(path, dual.FINGERPRINT_EXCLUSION_SCHEMA)
            self.assertEqual(value["fingerprints"], sorted(canonical))
            self.assertEqual(value["fingerprint_count"], 500)
            self.assertFalse(value["contains_transcripts"])
            self.assertFalse(value["contains_metrics"])
            self.assertFalse(value["contains_labels"])
            self.assertNotIn("openings", value)
            governance = dual.challenger._write_bank_dynamic_exclusion(
                root=root / "governance", attempt=3, gate_id="gate-a",
                bank=bank, bank_record=dual._record(bank_path),
                candidate=state["plan"]["candidate"],
            )
            self.assertEqual(governance["sha256"], q.sha256_file(path))

    def test_effective_candidate_profile_can_be_standard_under_intervention(self):
        preflight = {
            "derivation": {"source": {
                "search_throughput_profile": "state-evaluation-cache-v1",
                "candidate_search_profile": "standard-v1",
            }}
        }
        self.assertEqual(
            dual._preflight_candidate_search_profile(preflight),
            "standard-v1",
        )
        del preflight["derivation"]["source"]["candidate_search_profile"]
        with self.assertRaisesRegex(
            dual.DualFinalError, "effective candidate search profile"
        ):
            dual._preflight_candidate_search_profile(preflight)

    def _run_fixture(
        self, root: pathlib.Path, *, pass_a: bool, pass_b: bool,
        clock_values=None,
    ):
        plan_path = write(
            root / "execution-plan.json", dual.PLAN_SCHEMA, production=False
        )
        campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
        dual_reference = write(
            root / "dual-reference.json", dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA
        )
        plan = {
            "root": str(root), "attempt": 4,
            "production": False,
            "execution_identity": {"attempt": 4, "root": str(root)},
            "campaign_heavy_stage_lock": str(
                (root / ".rank4-teacher-heavy-stage.lock").resolve()
            ),
            "campaign_plan": q.artifact_reference(campaign, dual.challenger.PLAN_SCHEMA),
            "candidate": {
                "runtime": {"sha256": "8" * 64},
                "source": {"sha256": "9" * 64},
            },
        }
        state = {"path": plan_path, "plan": plan}
        plan_path.write_bytes(q.canonical_json_bytes(q.seal({
            "schema": dual.PLAN_SCHEMA, **plan,
        })))
        state["plan"] = q.load_sealed(plan_path, dual.PLAN_SCHEMA)
        plan = state["plan"]
        execution_order = []
        gate_launches = {}

        def execute(_state, *, gate_id, **kwargs):
            execution_order.append(f"run:{gate_id}")
            gate_launches[gate_id] = {
                "launched_at_utc": kwargs["launched_at_utc"],
                "not_before_utc": kwargs["not_before_utc"],
            }
            summary = strict_summary(
                wins=527 if (pass_a if gate_id == "gate-a" else pass_b) else 526,
                color0=267 if (pass_a if gate_id == "gate-a" else pass_b) else 266,
                color1=260,
            )
            verdict = q.strict_gate_verdict(summary)
            path = root / f"gates/{gate_id}/ledger/aggregate.json"
            write(
                path, q.FINAL_AGGREGATE_SCHEMA,
                binding={}, completed_at_utc="2026-09-04T00:02:00Z",
                summary=summary, verdict=verdict,
                status="rank4-qualified" if verdict["passed"] else "final-gate-failed",
            )
            return path

        def evidence(_state, *, gate_id, **_kwargs):
            execution_order.append(f"evidence:{gate_id}")
            return write(
                root / f"{gate_id}.evidence.json",
                dual.challenger.FINAL_GATE_EVIDENCE_SCHEMA,
            )

        def record(_reference, *, gate_id, **_kwargs):
            execution_order.append(f"record:{gate_id}")
            return write(
                root / f"{gate_id}.result.json",
                dual.challenger.FINAL_RESULT_SCHEMA,
            )

        def complete(*_args, **_kwargs):
            execution_order.append("complete")
            return write(
                root / "dual-qualified.json",
                dual.challenger.DUAL_QUALIFICATION_SCHEMA,
            )

        def seal_audit(_state, *, gate_id, **_kwargs):
            execution_order.append(f"audit:{gate_id}")
            return write(
                root / f"gates/{gate_id}/ledger/prelaunch-audits/audit-000.json",
                dual.PRELAUNCH_AUDIT_SCHEMA,
            )

        with mock.patch.object(dual, "_load_prepared", return_value=(dual_reference, {})), \
                mock.patch.object(dual.challenger, "validate_dual_final"), \
                mock.patch.object(dual, "validate_bank_receipt", return_value={}), \
                mock.patch.object(dual, "_seal_prelaunch_audit", side_effect=seal_audit), \
                mock.patch.object(dual, "_execute_gate", side_effect=execute), \
                mock.patch.object(dual, "_deep_evidence", side_effect=evidence):
            times = iter(clock_values or (
                "2026-09-04T00:02:00Z",
                "2026-09-04T00:02:01Z",
                "2026-09-04T00:03:00Z",
            ))
            result = dual.run_dual_final(
                plan_path, launched_at_utc="2026-09-04T00:00:00Z",
                state_validator=lambda _path: state,
                result_recorder=record, dual_completer=complete,
                clock=lambda: next(times),
                allow_injected_test_evidence=True,
            )
        return result, execution_order, gate_launches, state, dual_reference

    def test_gate_b_runs_only_after_gate_a_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            failed, order, launches, _state, _dual_reference = self._run_fixture(
                pathlib.Path(temporary), pass_a=False, pass_b=True
            )
            self.assertEqual(failed["status"], "gate-a-failed")
            self.assertEqual(order, [
                "audit:gate-a", "run:gate-a", "evidence:gate-a", "record:gate-a",
            ])
            self.assertEqual(set(launches), {"gate-a"})
        with tempfile.TemporaryDirectory() as temporary:
            passed, order, launches, _state, _dual_reference = self._run_fixture(
                pathlib.Path(temporary), pass_a=True, pass_b=True
            )
            self.assertEqual(passed["status"], "two-gates-passed")
            self.assertEqual(order, [
                "audit:gate-a", "run:gate-a", "evidence:gate-a", "record:gate-a",
                "audit:gate-b", "run:gate-b", "evidence:gate-b",
                "record:gate-b", "complete",
            ])
            self.assertEqual(
                launches,
                {
                    "gate-a": {
                        "launched_at_utc": "2026-09-04T00:02:00Z",
                        "not_before_utc": "2026-09-04T00:00:00Z",
                    },
                    "gate-b": {
                        "launched_at_utc": "2026-09-04T00:02:01Z",
                        "not_before_utc": "2026-09-04T00:02:00Z",
                    },
                },
            )
            receipt = q.load_sealed(
                passed["receipt"], dual.RUN_RECEIPT_SCHEMA
            )
            self.assertEqual(
                set(receipt["gate_prelaunch_audits"]), set(dual.GATE_IDS)
            )
            self.assertTrue(
                receipt["campaign_heavy_stage_lock"]["held_exclusively"]
            )
            self.assertEqual(
                receipt["execution_identity"],
                {"attempt": 4, "root": str(pathlib.Path(temporary))},
            )

    def test_gate_b_internal_launch_cannot_predate_gate_a_completion(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            dual.DualFinalError, "gate-b internally captured launch predates"
        ):
            self._run_fixture(
                pathlib.Path(temporary), pass_a=True, pass_b=True,
                clock_values=(
                    "2026-09-04T00:01:00Z",
                    "2026-09-04T00:01:59Z",
                ),
            )

    def test_execution_receipt_revalidates_exact_governance_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run, _order, _launches, state, dual_reference = self._run_fixture(
                root, pass_a=True, pass_b=False
            )
            receipt = q.load_sealed(run["receipt"], dual.RUN_RECEIPT_SCHEMA)
            result_paths = {
                gate_id: pathlib.Path(record["path"])
                for gate_id, record in receipt["gate_results"].items()
            }
            evidence_paths = {
                gate_id: pathlib.Path(record["path"])
                for gate_id, record in receipt["gate_evidence"].items()
            }
            dual_state = {
                "context": {"plan": {}},
                "plan": {},
                "path": root / "dual-plan.json",
            }

            def validate_result(path, *, gate_id, **_kwargs):
                if path.resolve() != result_paths[gate_id].resolve():
                    raise ValueError("substituted result")
                return {
                    "candidate": state["plan"]["candidate"],
                    "source_evidence": dual._record(evidence_paths[gate_id]),
                    "completed_at_utc": "2026-09-04T00:02:00Z",
                    "passed": gate_id == "gate-a",
                }

            ledger = [
                {
                    "event": "final-gate-recorded", "attempt": 4,
                    "gate_id": gate_id,
                    "result": dual.challenger._sealed_record(
                        result_paths[gate_id], dual.challenger.FINAL_RESULT_SCHEMA
                    ),
                }
                for gate_id in dual.GATE_IDS
            ]
            with mock.patch.object(
                dual, "_load_prepared", return_value=(dual_reference, {})
            ), mock.patch.object(
                dual.challenger, "validate_dual_final", return_value=dual_state
            ), mock.patch.object(
                dual.challenger, "validate_final_result", side_effect=validate_result
            ) as deep_result, mock.patch.object(
                dual, "validate_gate_evidence",
                side_effect=lambda path, **_kwargs: {
                    "verdict": {"passed": path == evidence_paths["gate-a"]}
                },
            ), mock.patch.object(
                dual, "validate_bank_receipt", return_value={"path": root}
            ), mock.patch.object(
                dual, "_validate_prelaunch_audit", return_value={"valid": True}
            ), mock.patch.object(
                dual.challenger, "load_ledger", return_value=ledger
            ):
                checked = dual.validate_execution_receipt(
                    run["receipt"], state_validator=lambda _path: state,
                    allow_injected_test_evidence=True,
                )
                self.assertEqual(checked["status"], "gate-b-failed")
                self.assertEqual(deep_result.call_count, 2)

                alternate = write(
                    root / "alternate-result.json",
                    dual.challenger.FINAL_RESULT_SCHEMA,
                )
                forged = {
                    key: value for key, value in receipt.items()
                    if key != "body_sha256"
                }
                forged["gate_results"] = {
                    **forged["gate_results"],
                    "gate-b": q.artifact_reference(
                        alternate, dual.challenger.FINAL_RESULT_SCHEMA
                    ),
                }
                run["receipt"].write_bytes(
                    q.canonical_json_bytes(q.seal(forged))
                )
                with self.assertRaisesRegex(
                    dual.DualFinalError, "governance result failed deep validation"
                ):
                    dual.validate_execution_receipt(
                        run["receipt"], state_validator=lambda _path: state,
                        allow_injected_test_evidence=True,
                    )

    def test_consumption_and_deep_validation_enforce_launch_lower_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            receipt_path = write(
                root / "bank-receipt.json", dual.BANK_RECEIPT_SCHEMA
            )
            fingerprint_path = write(
                root / "fingerprint.json", dual.FINGERPRINT_EXCLUSION_SCHEMA
            )
            binding_path = write(
                root / "binding.json", q.GATE_BINDING_SCHEMA, bank={}
            )
            state = {
                "path": plan_path,
                "plan": {
                    "root": str(root), "attempt": 1,
                    "uncontended_timing": {},
                },
            }
            bank_state = {
                "path": receipt_path,
                "fingerprint_exclusion_path": fingerprint_path,
                "binding_path": binding_path,
                "binding": {"bank": {}},
            }
            audit_path = write(
                root / "gates/gate-b/ledger/prelaunch-audits/audit-000.json",
                dual.PRELAUNCH_AUDIT_SCHEMA,
            )
            audit = {"audited_at_utc": "2026-09-04T00:00:30Z"}
            with mock.patch.object(
                dual, "_validate_prelaunch_audit", return_value=audit
            ):
                dual._consume_gate(
                    state, bank_state, gate_id="gate-b",
                    launched_at_utc="2026-09-04T00:01:00Z",
                    prelaunch_audit_path=audit_path,
                )
                with self.assertRaisesRegex(
                    dual.DualFinalError, "predates its authorized predecessor"
                ):
                    dual._consume_gate(
                        state, bank_state, gate_id="gate-b",
                        launched_at_utc="2026-09-04T00:03:00Z",
                        prelaunch_audit_path=audit_path,
                        not_before_utc="2026-09-04T00:02:00Z",
                    )
            with mock.patch.object(
                dual, "validate_bank_receipt", return_value=bank_state
            ), mock.patch.object(
                dual, "_validate_maintained_aggregate",
                return_value={
                    "verdict": {"passed": True},
                    "completed_at_utc": "2026-09-04T00:02:00Z",
                },
            ), mock.patch.object(
                dual, "_validate_prelaunch_audit", return_value=audit,
            ), self.assertRaisesRegex(
                dual.DualFinalError, "does not follow passing gate-a completion"
            ):
                dual._validate_consumption(state, bank_state, gate_id="gate-b")

    def test_strict_thresholds_are_527_260_and_zero_failures(self):
        self.assertTrue(q.strict_gate_verdict(strict_summary())["passed"])
        for summary in (
            strict_summary(wins=526, color0=266, color1=260),
            strict_summary(wins=527, color0=268, color1=259),
            strict_summary(failures=1),
        ):
            self.assertFalse(q.strict_gate_verdict(summary)["passed"])

    def test_maintained_aggregate_is_recomputed_from_all_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            binding_path = write(
                root / "gate-binding.json", q.GATE_BINDING_SCHEMA
            )
            bank_state = {"binding_path": binding_path}
            all_games = []
            color_seen = {0: 0, 1: 0}
            for pair in range(500):
                for color in (0, 1):
                    candidate_win = color_seen[color] < (267 if color == 0 else 260)
                    color_seen[color] += 1
                    all_games.append({
                        "pair_index": pair, "candidate_color": color,
                        "candidate_win": candidate_win, "turns": 320,
                        "failure": None, "first_ms": 800.0,
                        "later_max_ms": 155.0,
                    })

            def receipt(_path, *, index, **_kwargs):
                begin = index * 10
                return {"games": all_games[begin:begin + 10]}

            summary = strict_summary()
            aggregate = write(
                root / "ledger/aggregate.json", q.FINAL_AGGREGATE_SCHEMA,
                namespace=dual.NAMESPACE,
                binding=q.artifact_reference(binding_path, q.GATE_BINDING_SCHEMA),
                completed_at_utc="2026-09-04T00:00:00Z", summary=summary,
                verdict=q.strict_gate_verdict(summary), status="rank4-qualified",
            )
            with mock.patch.object(q, "validate_shard_receipt", side_effect=receipt):
                value = dual._validate_maintained_aggregate(
                    aggregate, bank_state=bank_state,
                    uncontended_timing=summary["uncontended_timing"],
                )
            self.assertTrue(value["verdict"]["passed"])

            bad_summary = {**summary, "candidate_wins": 528}
            bad = write(
                root / "bad/aggregate.json", q.FINAL_AGGREGATE_SCHEMA,
                namespace=dual.NAMESPACE,
                binding=q.artifact_reference(binding_path, q.GATE_BINDING_SCHEMA),
                completed_at_utc="2026-09-04T00:00:00Z", summary=bad_summary,
                verdict=q.strict_gate_verdict(bad_summary), status="rank4-qualified",
            )
            with mock.patch.object(q, "validate_shard_receipt", side_effect=receipt), \
                    self.assertRaisesRegex(dual.DualFinalError, "aggregate changed"):
                dual._validate_maintained_aggregate(
                    bad, bank_state=bank_state,
                    uncontended_timing=summary["uncontended_timing"],
                )

    def test_execute_gate_uses_exactly_four_workers_and_100_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            gate_path = root / "rank4-gate"
            gate_path.write_text("#!/bin/sh\n", encoding="ascii")
            gate_path.chmod(0o500)
            plan = {
                "root": str(root), "attempt": 1,
                "production": False,
                "candidate_search_profile": "standard-v1",
                "configuration": dual.deployment.deployment_configuration(
                    ["0.95", "0.5", "1"], "default",
                    dual.deployment.PROFILE_ROSTER["default"],
                ),
                "candidate": {
                    "source": {"path": "/candidate", "sha256": "1" * 64},
                },
                "rank4": {"path": "/rank4"},
                "gate": dual._record(gate_path, executable=True),
                "repository": str(root),
                "uncontended_timing": {
                    "first_max_ms": 101.0, "later_max_ms": 21.0,
                },
            }
            state = {"path": plan_path, "plan": plan}
            bank_receipt_path = write(
                root / "bank-receipt.json", dual.BANK_RECEIPT_SCHEMA
            )
            bank_state = {
                "binding_path": root / "binding.json",
                "path": bank_receipt_path,
                "receipt": {"gate_bank": {"path": "/bank", "sha256": "2" * 64}},
            }
            audit_path = write(
                root / "gates/gate-a/ledger/prelaunch-audits/audit-000.json",
                dual.PRELAUNCH_AUDIT_SCHEMA,
            )
            worker_counts = []
            specs = []

            def executor_factory(*, max_workers):
                worker_counts.append(max_workers)
                return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

            def runner(spec):
                specs.append(spec)
                return {"index": spec["index"]}

            def aggregate(ledger, **_kwargs):
                return write(
                    ledger / "aggregate.json", q.FINAL_AGGREGATE_SCHEMA,
                    binding={}, completed_at_utc="2026-09-04T00:00:00Z",
                    summary=strict_summary(),
                    verdict=q.strict_gate_verdict(strict_summary()),
                    status="rank4-qualified",
                )

            games = lambda index: [
                {
                    "pair_index": index * 5 + pair,
                    "candidate_color": color, "candidate_win": True,
                    "turns": 1, "failure": None,
                    "first_ms": 1.0, "later_max_ms": 1.0,
                }
                for pair in range(5) for color in (0, 1)
            ]
            with mock.patch.object(dual, "validate_bank_receipt", return_value=bank_state), \
                    mock.patch.object(dual, "_consume_gate"), \
                    mock.patch.object(dual, "_audit_shards", side_effect=[list(range(20, 100)), []]), \
                    mock.patch.object(q, "start_final_shard"), \
                    mock.patch.object(q, "record_shard_receipt"), \
                    mock.patch.object(q, "aggregate_final", side_effect=aggregate), \
                    mock.patch.object(dual, "_validate_maintained_aggregate", return_value={}), \
                    mock.patch.object(
                        dual, "_validate_prelaunch_audit",
                        return_value={
                            "audited_at_utc": "2026-09-04T00:00:00Z",
                            "campaign_heavy_stage_lock": {},
                        },
                    ), \
                    mock.patch.object(dual, "_verify_lock_identity"), \
                    mock.patch.object(
                        dual, "_verify_frozen_gate", return_value=gate_path,
                    ), \
                    mock.patch.object(
                        dual, "_adapt_result",
                        side_effect=lambda _raw, *, index, **_kwargs: games(index),
                    ):
                dual._execute_gate(
                    state, gate_id="gate-a",
                    launched_at_utc="2026-09-04T00:00:00Z",
                    runner=runner, result_validator=lambda *_a, **_k: {},
                    clock=lambda: "2026-09-04T00:00:00Z",
                    prelaunch_audit_path=audit_path,
                    executor_factory=executor_factory,
                    allow_injected_test_evidence=True,
                )
            self.assertEqual(worker_counts, [4])
            self.assertEqual(dual.SHARDS, 100)
            self.assertEqual(len(specs), 80)
            self.assertEqual({spec["index"] for spec in specs}, set(range(20, 100)))
            self.assertTrue(all(spec["workers"] == 4 for spec in specs))
            self.assertTrue(all(spec["threads_per_worker"] == 1 for spec in specs))

    def test_deep_evidence_exposes_and_binds_full_100_shard_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
            plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
            dual_plan_path = write(root / "dual-plan.json", dual.challenger.DUAL_FINAL_SCHEMA)
            dual_reference = write(
                root / "dual-reference.json", dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA
            )
            bank_path = root / "protected-bank.json"
            bank_path.write_bytes(b"protected")
            bank_record = dual._record(bank_path)
            bank_receipt_path = write(
                root / "bank-receipt.json", dual.BANK_RECEIPT_SCHEMA
            )
            fingerprint_path = write(
                root / "fingerprint-exclusion.json", dual.FINGERPRINT_EXCLUSION_SCHEMA
            )
            other_fingerprint = write(
                root / "other-fingerprint-exclusion.json",
                dual.FINGERPRINT_EXCLUSION_SCHEMA,
            )
            binding_path = write(
                root / "gate-binding.json", q.GATE_BINDING_SCHEMA,
                bank={"path": "/bank-adapter", "sha256": "1" * 64},
            )
            candidate = {
                "runtime": {"sha256": "2" * 64},
                "source": {"sha256": "3" * 64},
            }
            configuration = {"tuple": ["0.95", "0.5", "1"]}
            timing = {"first_max_ms": 101.0, "later_max_ms": 21.0}
            plan = {
                "root": str(root), "attempt": 1,
                "candidate_search_profile": "standard-v1",
                "campaign_plan": q.artifact_reference(campaign, dual.challenger.PLAN_SCHEMA),
                "candidate": candidate, "candidate_commit": "a" * 40,
                "runtime_identity": {
                    "architecture": dual.ARCHITECTURE,
                    "runtime_body_sha256": "4" * 64,
                    "payload_sha256": "5" * 64,
                },
                "configuration": configuration,
                "deployment_preflight": {"path": "/preflight", "sha256": "6" * 64},
                "preflight_kind": "rank4-teacher-release",
                "preflight_schema": dual.challenger.RELEASE_EVIDENCE_SCHEMA,
                "compile_binding": {"candidate_embedded": True},
                "uncontended_timing": timing,
                "ci": {"path": "/ci", "sha256": "7" * 64},
            }
            state = {"path": plan_path, "plan": plan}
            prepared = {
                "fingerprint_exclusions": {
                    "gate-a": q.artifact_reference(
                        fingerprint_path, dual.FINGERPRINT_EXCLUSION_SCHEMA
                    ),
                    "gate-b": q.artifact_reference(
                        other_fingerprint, dual.FINGERPRINT_EXCLUSION_SCHEMA
                    ),
                },
            }
            write(root / "prepared.json", dual.PREPARED_SCHEMA)
            dual_state = {
                "path": dual_plan_path,
                "plan": {
                    "gates": [{"gate_id": "gate-a", "bank": bank_record}],
                    "dynamic_exclusions": [
                        {"sha256": record["sha256"]}
                        for record in prepared["fingerprint_exclusions"].values()
                    ],
                },
            }
            binding = q.load_sealed(binding_path, q.GATE_BINDING_SCHEMA)
            bank_state = {
                "path": bank_receipt_path,
                "receipt": {"protected_bank": bank_record},
                "binding_path": binding_path, "binding": binding,
                "fingerprint_exclusion_path": fingerprint_path,
            }
            ledger = root / "gates/gate-a/ledger"
            prelaunch_path = write(
                ledger / "prelaunch-audits/audit-000.json",
                dual.PRELAUNCH_AUDIT_SCHEMA,
            )
            summary = strict_summary()
            maintained_path = write(
                ledger / "aggregate.json", q.FINAL_AGGREGATE_SCHEMA,
                namespace=dual.NAMESPACE, binding={},
                completed_at_utc="2026-09-04T00:00:00Z",
                summary=summary, verdict=q.strict_gate_verdict(summary),
                status="rank4-qualified",
            )
            normalized_path = write(
                ledger / "governance-aggregate.json",
                dual.NORMALIZED_AGGREGATE_SCHEMA,
                namespace=dual.NAMESPACE, campaign_id=dual.CAMPAIGN_ID,
                attempt=1, gate_id="gate-a",
                candidate_source_sha256="3" * 64,
                candidate_runtime_sha256="2" * 64,
                bank_sha256=bank_record["sha256"], workers=4,
                threads_per_worker=1,
                maintained_aggregate=q.artifact_reference(
                    maintained_path, q.FINAL_AGGREGATE_SCHEMA
                ),
                summary=summary, verdict=q.strict_gate_verdict(summary),
                completed_at_utc="2026-09-04T00:00:00Z",
            )
            consumption_path = write(
                ledger / "consumption.json", dual.CONSUMPTION_SCHEMA,
                namespace=dual.NAMESPACE, campaign_id=dual.CAMPAIGN_ID,
                attempt=1, gate_id="gate-a",
                status="gate-bank-consumed-at-launch",
                launched_at_utc="2026-09-04T00:00:00Z",
                execution_plan=q.artifact_reference(plan_path, dual.PLAN_SCHEMA),
                bank_receipt=q.artifact_reference(
                    bank_receipt_path, dual.BANK_RECEIPT_SCHEMA
                ),
                fingerprint_exclusion=q.artifact_reference(
                    fingerprint_path, dual.FINGERPRINT_EXCLUSION_SCHEMA
                ),
                gate_binding=q.artifact_reference(
                    binding_path, q.GATE_BINDING_SCHEMA
                ),
                initial_prelaunch_audit=q.artifact_reference(
                    prelaunch_path, dual.PRELAUNCH_AUDIT_SCHEMA
                ),
                workers=4, threads_per_worker=1,
                one_launch_only=True, retry_authorized=False,
                upload_authorized=False,
            )
            primitive_path = write(
                ledger / "bank-consumed.json", dual.PRIMITIVE_CONSUMPTION_SCHEMA,
                namespace=dual.NAMESPACE,
                binding_sha256=q.sha256_file(binding_path), bank=binding["bank"],
                consumed_at_utc="2026-09-04T00:00:00Z",
            )
            del normalized_path, consumption_path, primitive_path
            for index in range(100):
                claim_path = write(
                    ledger / "claims" / f"shard-{index:03d}.json",
                    q.SHARD_CLAIM_SCHEMA,
                )
                raw_path = ledger / "raw" / f"shard-{index:03d}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps({"index": index}), encoding="ascii")
                raw_evidence_path = write(
                    ledger / "raw-evidence" / f"shard-{index:03d}.json",
                    dual.RAW_EVIDENCE_SCHEMA,
                    raw_gate_result=dual._record(raw_path),
                    prelaunch_audit=q.artifact_reference(
                        prelaunch_path, dual.PRELAUNCH_AUDIT_SCHEMA
                    ),
                )
                write(
                    ledger / "receipts" / f"shard-{index:03d}.json",
                    q.SHARD_RECEIPT_SCHEMA,
                    evidence=q.artifact_reference(
                        raw_evidence_path, dual.RAW_EVIDENCE_SCHEMA
                    ),
                )
                self.assertTrue(claim_path.is_file())

            receipt_loader = lambda path, **_kwargs: q.load_sealed(
                path, q.SHARD_RECEIPT_SCHEMA
            )
            search_activation = {
                "schema": (
                    dual.gate_support.SEARCH_PROFILE_ACTIVATION_AGGREGATE_SCHEMA
                ),
                "candidate_search_profile": "standard-v1",
                "document_count": 100,
                "candidate_decisions": 1_000,
                "search_intervention": {},
                "requirements": {"all_intervention_counters_zero": True},
                "exercised": True,
                "body_sha256": "8" * 64,
            }
            validator = dual.validate_gate_evidence
            common_patches = (
                mock.patch.object(dual, "_load_prepared", return_value=(dual_reference, prepared)),
                mock.patch.object(
                    dual.challenger, "validate_dual_final", return_value=dual_state
                ),
                mock.patch.object(dual, "validate_bank_receipt", return_value=bank_state),
                mock.patch.object(dual, "_audit_shards", return_value=[]),
                mock.patch.object(
                    dual, "_validate_maintained_aggregate",
                    return_value=q.load_sealed(maintained_path, q.FINAL_AGGREGATE_SCHEMA),
                ),
                mock.patch.object(dual, "_normalized_aggregate", return_value=(
                    ledger / "governance-aggregate.json"
                )),
                mock.patch.object(q, "validate_shard_receipt", side_effect=receipt_loader),
                mock.patch.object(dual, "validate_execution_plan", return_value=state),
                mock.patch.object(
                    dual, "_aggregate_raw_search_profile",
                    return_value=search_activation,
                ),
                mock.patch.object(
                    dual, "_validate_prelaunch_audit",
                    return_value={"audited_at_utc": "2026-09-04T00:00:00Z"},
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], \
                    common_patches[3], common_patches[4], common_patches[5], \
                    common_patches[6], common_patches[7], common_patches[8], \
                    common_patches[9], \
                    mock.patch.object(dual, "validate_gate_evidence"):
                evidence_path = dual._deep_evidence(
                    state, dual_reference=dual_reference, gate_id="gate-a",
                    aggregate_path=maintained_path,
                    result_validator=lambda *_a, **_k: {},
                    allow_injected_test_evidence=True,
                )
            evidence = q.load_sealed(
                evidence_path, dual.challenger.FINAL_GATE_EVIDENCE_SCHEMA
            )
            self.assertEqual(len(evidence["claims"]), 100)
            self.assertEqual(len(evidence["receipts"]), 100)
            self.assertEqual(len(evidence["raw_gate_results"]), 100)

            with mock.patch.object(dual, "_load_prepared", return_value=(dual_reference, prepared)), \
                    mock.patch.object(
                        dual.challenger, "validate_dual_final", return_value=dual_state
                    ), mock.patch.object(
                        dual, "validate_bank_receipt", return_value=bank_state
                    ), mock.patch.object(dual, "_audit_shards", return_value=[]), \
                    mock.patch.object(
                        dual, "_validate_maintained_aggregate",
                        return_value=q.load_sealed(
                            maintained_path, q.FINAL_AGGREGATE_SCHEMA
                        ),
                    ), mock.patch.object(
                        q, "validate_shard_receipt", side_effect=receipt_loader
                    ), mock.patch.object(
                        dual, "validate_execution_plan", return_value=state
                    ), mock.patch.object(
                        dual, "_aggregate_raw_search_profile",
                        return_value=search_activation,
                    ), mock.patch.object(
                        dual, "_validate_prelaunch_audit",
                        return_value={
                            "audited_at_utc": "2026-09-04T00:00:00Z"
                        },
                    ):
                checked = validator(
                    evidence_path, state=state, dual_reference=dual_reference,
                    result_validator=lambda *_a, **_k: {},
                    allow_injected_test_evidence=True,
                )
                self.assertEqual(checked["candidate"]["source_sha256"], "3" * 64)
                tampered = {key: value for key, value in evidence.items() if key != "body_sha256"}
                tampered["candidate"] = {
                    **tampered["candidate"], "source_sha256": "f" * 64,
                }
                tampered_path = write(
                    root / "tampered-evidence.json",
                    tampered.pop("schema"), **tampered,
                )
                with self.assertRaisesRegex(dual.DualFinalError, "closure changed"):
                    validator(
                        tampered_path, state=state,
                        dual_reference=dual_reference,
                        result_validator=lambda *_a, **_k: {},
                        allow_injected_test_evidence=True,
                    )


if __name__ == "__main__":
    unittest.main()
