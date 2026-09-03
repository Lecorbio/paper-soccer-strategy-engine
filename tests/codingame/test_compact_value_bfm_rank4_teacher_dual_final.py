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
        self.campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
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

    def prepare(self, *, generalized_release: bool = False):
        return dual.prepare_execution(
            authorization_path=self.authorization_path,
            campaign_plan_path=self.campaign,
            output_root=self.output,
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
        )


class DualFinalExecutionTests(unittest.TestCase):
    def test_release_adapter_uses_generated_source_not_deployed_source(self):
        selected = pathlib.Path("/frozen/generated.cpp")
        deployed = pathlib.Path("/release/deployed.cpp")
        observed = []

        def adapt(_path, **kwargs):
            observed.append(("adapt", kwargs["candidate_source"]))
            return {"candidate": {"path": str(deployed)}}

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
                )
                self.assertEqual(state["plan"]["attempt"], attempt)
                self.assertEqual(
                    state["plan"]["candidate"]["source"]["sha256"],
                    q.sha256_file(fixture.source),
                )
                self.assertTrue(state["plan"]["compile_binding"]["candidate_embedded"])
                self.assertEqual(
                    state["plan"]["preflight_kind"],
                    "rank4-teacher-release" if attempt > 0
                    else "discrete-v3-deployment-preflight",
                )
                self.assertEqual(state["plan"]["gate_contract"]["workers_per_gate"], 4)
                self.assertEqual(state["plan"]["gate_contract"]["shards_per_gate"], 100)

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
                    "authorization": q.artifact_reference(
                        authorization, dual.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
                    ),
                    "campaign_plan": q.artifact_reference(
                        campaign, dual.challenger.PLAN_SCHEMA
                    ),
                },
            }
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
                    "receipt": {
                        "protected_bank": first_record,
                        "exclusion_sources": [],
                    },
                },
                "gate-b": {
                    "seed": {"seed_256_hex": "02" * 32}, "bank": bank_b,
                    "bank_path": second_bank_path,
                    "fingerprint_exclusion_path": second_exclusion,
                    "receipt": {
                        "protected_bank": second_record,
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
                )

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

    def _run_fixture(self, root: pathlib.Path, *, pass_a: bool, pass_b: bool):
        plan_path = write(root / "execution-plan.json", dual.PLAN_SCHEMA)
        campaign = write(root / "campaign.json", dual.challenger.PLAN_SCHEMA)
        dual_reference = write(
            root / "dual-reference.json", dual.challenger.DUAL_FINAL_REFERENCE_SCHEMA
        )
        plan = {
            "root": str(root), "attempt": 4,
            "campaign_plan": q.artifact_reference(campaign, dual.challenger.PLAN_SCHEMA),
            "candidate": {
                "runtime": {"sha256": "8" * 64},
                "source": {"sha256": "9" * 64},
            },
        }
        state = {"path": plan_path, "plan": plan}
        execution_order = []

        def execute(_state, *, gate_id, **_kwargs):
            execution_order.append(f"run:{gate_id}")
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

        with mock.patch.object(dual, "_load_prepared", return_value=(dual_reference, {})), \
                mock.patch.object(dual.challenger, "validate_dual_final"), \
                mock.patch.object(dual, "_execute_gate", side_effect=execute), \
                mock.patch.object(dual, "_deep_evidence", side_effect=evidence):
            result = dual.run_dual_final(
                plan_path, launched_at_utc="2026-09-04T00:02:00Z",
                state_validator=lambda _path: state,
                result_recorder=record, dual_completer=complete,
                clock=lambda: "2026-09-04T00:03:00Z",
            )
        return result, execution_order

    def test_gate_b_runs_only_after_gate_a_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            failed, order = self._run_fixture(
                pathlib.Path(temporary), pass_a=False, pass_b=True
            )
            self.assertEqual(failed["status"], "gate-a-failed")
            self.assertEqual(order, ["run:gate-a", "evidence:gate-a", "record:gate-a"])
        with tempfile.TemporaryDirectory() as temporary:
            passed, order = self._run_fixture(
                pathlib.Path(temporary), pass_a=True, pass_b=True
            )
            self.assertEqual(passed["status"], "two-gates-passed")
            self.assertEqual(order, [
                "run:gate-a", "evidence:gate-a", "record:gate-a",
                "run:gate-b", "evidence:gate-b", "record:gate-b", "complete",
            ])

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
            plan = {
                "root": str(root), "attempt": 1,
                "configuration": dual.deployment.deployment_configuration(
                    ["0.95", "0.5", "1"], "default",
                    dual.deployment.PROFILE_ROSTER["default"],
                ),
                "candidate": {
                    "source": {"path": "/candidate", "sha256": "1" * 64},
                },
                "rank4": {"path": "/rank4"},
                "gate": {"path": "/gate"},
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
                        dual, "_adapt_result",
                        side_effect=lambda _raw, *, index, **_kwargs: games(index),
                    ):
                dual._execute_gate(
                    state, gate_id="gate-a",
                    launched_at_utc="2026-09-04T00:00:00Z",
                    runner=runner, result_validator=lambda *_a, **_k: {},
                    clock=lambda: "2026-09-04T00:00:00Z",
                    executor_factory=executor_factory,
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
            )
            with common_patches[0], common_patches[1], common_patches[2], \
                    common_patches[3], common_patches[4], common_patches[5], \
                    common_patches[6], common_patches[7], \
                    mock.patch.object(dual, "validate_gate_evidence"):
                evidence_path = dual._deep_evidence(
                    state, dual_reference=dual_reference, gate_id="gate-a",
                    aggregate_path=maintained_path,
                    result_validator=lambda *_a, **_k: {},
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
                    ):
                checked = validator(
                    evidence_path, state=state, dual_reference=dual_reference,
                    result_validator=lambda *_a, **_k: {},
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
                    )


if __name__ == "__main__":
    unittest.main()
