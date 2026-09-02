import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/compact_value_bfm_discrete_v3_release.py"
SPEC = importlib.util.spec_from_file_location("compact_v3_release_tested", TOOL)
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)
q = release.qualification
COMMIT = "c" * 40
AGENT = 701
SUBMISSION = 801
RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"


def gh_payload(head=COMMIT, conclusion="success"):
    jobs = [
        {
            "name": name,
            "status": "completed",
            "conclusion": conclusion if index == 0 else "success",
            "databaseId": 100 + index,
            "url": (
                f"{release.upload_primitives.RUN_URL_PREFIX}123/job/{100 + index}"
            ),
        }
        for index, name in enumerate(release.upload_primitives.JOB_NAMES)
    ]
    jobs.append({
        "name": "deploy", "status": "completed", "conclusion": "skipped"
    })
    return {
        "databaseId": 123,
        "workflowDatabaseId": release.upload_primitives.WORKFLOW_DATABASE_ID,
        "attempt": 1,
        "name": "CI and Pages",
        "workflowName": "CI and Pages",
        "event": "workflow_dispatch",
        "headBranch": "compact-value-bfm",
        "headSha": head,
        "status": "completed",
        "conclusion": conclusion,
        "url": f"{release.upload_primitives.RUN_URL_PREFIX}123",
        "jobs": jobs,
    }


def strict_summary():
    return {
        "games": 1_000,
        "candidate_wins": 527,
        "candidate_color_wins": {"0": 260, "1": 267},
        "failures": {name: 0 for name in q.FAILURE_CATEGORIES},
        "maximum_turns": 320,
        "timing": {"first_max_ms": 999.0, "later_max_ms": 199.0},
        "uncontended_timing": {
            "first_max_ms": 899.0, "later_max_ms": 179.0,
        },
    }


class Fixture:
    def __init__(self, root):
        self.root = root.resolve()
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.source = self.repository / "submission.cpp"
        self.generated = self.root / "generated.cpp"
        base_source = b"""\
inline constexpr std::size_t kRootPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr double kExploration = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr double kFinalVisitWeight = 1.0;
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
"""
        self.generated.write_bytes(base_source)
        self.selected_tuple = ["0.95", "0.75", "1"]
        self.profile = "heavy"
        self.profile_work = release.final_bridge.deployment.PROFILE_ROSTER[
            self.profile
        ]
        self.source.write_bytes(release.final_bridge.deployment.derive_source(
            base_source, search_tuple=self.selected_tuple,
            profile=self.profile, work=self.profile_work,
        ))
        self.derivation = release.final_bridge.deployment.attest_derivation(
            base_source, self.source.read_bytes(),
            search_tuple=self.selected_tuple, profile=self.profile,
            work=self.profile_work,
        )
        self.manifest_value = release.final_bridge.deployment.create_manifest(
            base_source, self.source.read_bytes(),
            search_tuple=self.selected_tuple, profile=self.profile,
            work=self.profile_work,
        )
        self.manifest = self.root / "deployment.json"
        self.manifest.write_text(
            json.dumps(
                self.manifest_value, sort_keys=True, separators=(",", ":")
            ), encoding="ascii",
        )
        self.runtime = self.root / "runtime.json"
        self.runtime.write_text('{"runtime":true}\n', encoding="ascii")

        self.source_binding = self.root / "source-binding.json"
        q.create_source_binding(
            self.source_binding, candidate_source=self.source,
            candidate_commit=COMMIT, rank4_source=RANK4,
            opponent_source=RANK4,
        )
        source_binding = q.load_sealed(
            self.source_binding, q.SOURCE_BINDING_SCHEMA
        )
        self.bank_adapter = self.root / "bank-adapter.json"
        q.write_sealed(self.bank_adapter, {
            "schema": q.FINAL_BANK_SCHEMA,
            "namespace": q.NAMESPACE,
            "source_binding": q.artifact_reference(
                self.source_binding, q.SOURCE_BINDING_SCHEMA
            ),
            "candidate_commit": COMMIT,
            "candidate_sha256": source_binding["candidate"]["sha256"],
            "rank4_sha256": q.RANK4_SHA256,
            "opening_count": 500,
        })
        self.harness = self.root / "gate"
        self.harness.write_text("gate\n", encoding="ascii")
        self.binding = self.root / "gate-binding.json"
        q.create_gate_binding(
            self.binding, source_binding_path=self.source_binding,
            bank_path=self.bank_adapter, harness_path=self.harness,
        )

        self.final_root = self.root / release.final_bridge.BRIDGE_DIRECTORY
        self.ledger = self.final_root / "ledger"
        self.ledger.mkdir(parents=True)
        self.plan = self.final_root / "plan.json"
        q.write_sealed(self.plan, {
            "schema": release.final_bridge.PLAN_SCHEMA,
            "namespace": q.NAMESPACE,
            "inputs": {
                "candidate": release._record(self.source, ascii_required=True),
                "generated_source": release._record(
                    self.generated, ascii_required=True
                ),
                "deployment_derivation": self.derivation,
                "deployment_manifest": release._record(
                    self.manifest, ascii_required=True
                ),
                "deployment_manifest_body_sha256": self.manifest_value[
                    "body_sha256"
                ],
                "tuple": self.selected_tuple,
                "profile": self.profile,
                "profile_work": self.profile_work,
            },
            "configuration": {"deployment": self.derivation["configuration"]},
            "paths": {"ledger": str(self.ledger)},
        })
        self.bank_receipt = self.final_root / "bank-receipt.json"
        q.write_sealed(self.bank_receipt, {
            "schema": release.final_bridge.BANK_RECEIPT_SCHEMA,
            "namespace": q.NAMESPACE,
            "gate_binding": q.artifact_reference(
                self.binding, q.GATE_BINDING_SCHEMA
            ),
        })
        self.aggregate = self.ledger / "aggregate.json"
        summary = strict_summary()
        q.write_sealed(self.aggregate, {
            "schema": q.FINAL_AGGREGATE_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "rank4-qualified",
            "binding": q.artifact_reference(self.binding, q.GATE_BINDING_SCHEMA),
            "completed_at_utc": "2026-09-01T10:00:00Z",
            "summary": summary,
            "verdict": q.strict_gate_verdict(summary),
        })
        self.consumption = self.ledger / "consumption.json"
        q.write_sealed(self.consumption, {
            "schema": release.final_bridge.CONSUMPTION_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "v3-final-bank-consumed-at-launch",
            "launched_at_utc": "2026-09-01T09:00:00Z",
            "plan": q.artifact_reference(
                self.plan, release.final_bridge.PLAN_SCHEMA
            ),
            "bank_receipt": q.artifact_reference(
                self.bank_receipt, release.final_bridge.BANK_RECEIPT_SCHEMA
            ),
            "one_launch_only": True,
            "upload_authorized": False,
        })
        self.preflight_claim = self.root / "deployment-preflight-claim.json"
        q.write_sealed(self.preflight_claim, {
            "schema": release.final_bridge.deployment_preflight.CLAIM_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "deployment-preflight-claimed-before-execution",
            "claimed_at_utc": "2026-09-01T08:00:00Z",
        })
        self.preflight_receipt = self.root / "deployment-preflight-receipt.json"
        q.write_sealed(self.preflight_receipt, {
            "schema": release.final_bridge.deployment_preflight.RECEIPT_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "deployment-preflight-passed",
            "claim": q.artifact_reference(
                self.preflight_claim,
                release.final_bridge.deployment_preflight.CLAIM_SCHEMA,
            ),
        })
        self.preflight = self.root / "preflight.json"
        q.write_sealed(self.preflight, {
            "schema": release.final_bridge.deployment_preflight.REFERENCE_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "deployment-preflight-passed-awaiting-final",
            "receipt": q.artifact_reference(
                self.preflight_receipt,
                release.final_bridge.deployment_preflight.RECEIPT_SCHEMA,
            ),
        })

        def sealed(name, schema):
            path = self.root / name
            q.write_sealed(path, {"schema": schema, "namespace": q.NAMESPACE})
            return path

        self.development_plan = sealed(
            "development-plan.json", release.development.PLAN_SCHEMA
        )
        self.finalist_reference = sealed(
            "finalist-reference.json", release.development.FINALIST_REFERENCE_SCHEMA
        )
        self.finalist = sealed("finalist.json", release.development.FINALIST_SCHEMA)
        self.evaluation = sealed(
            "evaluation.json", release.adapter.EVALUATION_COMPLETION_SCHEMA
        )
        self.adapter_plan = sealed(
            "adapter-plan.json", release.adapter.ADAPTER_PLAN_SCHEMA
        )
        self.v3_plan = sealed("v3-plan.json", release.adapter.v3.PLAN_SCHEMA)
        self.handoff = self.root / "handoff.json"
        q.write_sealed(self.handoff, {
            "schema": release.adapter.HANDOFF_SCHEMA,
            "namespace": q.NAMESPACE,
            "adapter_plan": q.artifact_reference(
                self.adapter_plan, release.adapter.ADAPTER_PLAN_SCHEMA
            ),
            "v3_plan": q.artifact_reference(
                self.v3_plan, release.adapter.v3.PLAN_SCHEMA
            ),
            "evaluation_completion": q.artifact_reference(
                self.evaluation, release.adapter.EVALUATION_COMPLETION_SCHEMA
            ),
        })
        self.exclusion = sealed(
            "exclusion.json", release.exclusions.RECEIPT_SCHEMA
        )
        self.qualified = self.ledger / "v3-qualified-inputs.json"
        candidate = release._record(self.source, ascii_required=True)
        runtime = release._record(self.runtime, ascii_required=True)
        q.write_sealed(self.qualified, {
            "schema": release.final_bridge.QUALIFIED_SCHEMA,
            "namespace": q.NAMESPACE,
            "campaign_id": release.CAMPAIGN_ID,
            "status": "v3-rank4-qualified-awaiting-green-ci",
            "candidate_commit": COMMIT,
            "candidate": candidate,
            "runtime": runtime,
            "deployment_derivation": self.derivation,
            "deployment_manifest": release._record(
                self.manifest, ascii_required=True
            ),
            "deployment_manifest_body_sha256": self.manifest_value[
                "body_sha256"
            ],
            "development_plan": q.artifact_reference(
                self.development_plan, release.development.PLAN_SCHEMA
            ),
            "finalist_reference": q.artifact_reference(
                self.finalist_reference,
                release.development.FINALIST_REFERENCE_SCHEMA,
            ),
            "finalist": release.development._sealed_record(
                self.finalist, release.development.FINALIST_SCHEMA
            ),
            "handoff": release.development._sealed_record(
                self.handoff, release.adapter.HANDOFF_SCHEMA
            ),
            "evaluation_completion": q.artifact_reference(
                self.evaluation, release.adapter.EVALUATION_COMPLETION_SCHEMA
            ),
            "exclusion_receipt": release.development._sealed_record(
                self.exclusion, release.exclusions.RECEIPT_SCHEMA
            ),
            "plan": q.artifact_reference(self.plan, release.final_bridge.PLAN_SCHEMA),
            "bank_receipt": q.artifact_reference(
                self.bank_receipt, release.final_bridge.BANK_RECEIPT_SCHEMA
            ),
            "aggregate": q.artifact_reference(
                self.aggregate, q.FINAL_AGGREGATE_SCHEMA
            ),
            "preflight": q.artifact_reference(
                self.preflight,
                release.final_bridge.deployment_preflight.REFERENCE_SCHEMA,
            ),
            "strict_thresholds": q.strict_gate_verdict(summary)["thresholds"],
            "uploads_authorized": 0,
            "rank4_replacement_authorized": False,
        })
        self.chain = {
            "qualified": q.load_sealed(
                self.qualified, release.final_bridge.QUALIFIED_SCHEMA
            ),
            "qualified_path": self.qualified,
            "plan": q.load_sealed(self.plan, release.final_bridge.PLAN_SCHEMA),
            "plan_path": self.plan,
            "bank_receipt": q.load_sealed(
                self.bank_receipt, release.final_bridge.BANK_RECEIPT_SCHEMA
            ),
            "bank_receipt_path": self.bank_receipt,
            "aggregate": q.load_sealed(self.aggregate, q.FINAL_AGGREGATE_SCHEMA),
            "aggregate_path": self.aggregate,
            "consumption": q.load_sealed(
                self.consumption, release.final_bridge.CONSUMPTION_SCHEMA
            ),
            "consumption_path": self.consumption,
            "candidate_path": self.source,
            "runtime_path": self.runtime,
            "preflight_path": self.preflight,
            "binding_path": self.binding,
            "git": {"commit": COMMIT, "tracked_clean": True,
                    "branch": "compact-value-bfm"},
            "raw_shards": [
                {"index": index, "raw_sha256": f"{index:064x}",
                 "evidence_sha256": f"{index + 1:064x}",
                 "receipt_sha256": f"{index + 2:064x}"}
                for index in range(100)
            ],
        }

        self.registry = self.root / "live-exclusions.json"
        self.registry.write_bytes(release.live.canonical_json_bytes({
            "schema": release.live.EXCLUSION_SCHEMA,
            "records": [],
        }))

    def validator(self, campaign_root, qualified_path, repository):
        if campaign_root.resolve() != self.root or qualified_path.resolve() != self.qualified:
            raise release.ReleaseError("synthetic qualified route changed")
        if repository.resolve() != self.repository:
            raise release.ReleaseError("synthetic repository route changed")
        release._validate_qualified_records(
            self.chain["qualified"], campaign_root=self.root,
            handoff_validator=lambda path, **_kwargs: q.load_sealed(
                path, release.adapter.HANDOFF_SCHEMA
            ),
        )
        release._validate_deployment_binding(
            self.chain["qualified"], self.chain["plan"]
        )
        return dict(self.chain)

    def prereqs(self):
        release.freeze_live_exclusions(
            self.root, registry_path=self.registry,
            frozen_at_utc="2026-09-01T11:00:00Z",
        )
        release.seal_ci_evidence(
            self.root, gh_payload=gh_payload(), expected_head=COMMIT,
            fetched_at_utc="2026-09-01T11:30:00Z",
        )

    def authorize(self):
        self.prereqs()
        root = self.root / release.RELEASE_DIRECTORY
        return release.authorize_release(
            self.root, qualified_path=self.qualified,
            ci_evidence_path=root / "github-ci.json",
            live_exclusion_binding_path=root / "live-exclusion-binding.json",
            repository=self.repository,
            authorized_at_utc="2026-09-01T12:00:00Z",
            qualified_validator=self.validator,
        )

    def state_kwargs(self):
        return {
            "campaign_root": self.root,
            "repository": self.repository,
            "qualified_validator": self.validator,
        }

    def ready(self):
        self.authorize()
        release.fresh_editor(
            **self.state_kwargs(), session_id="fresh-editor-1",
            opened_at_utc="2026-09-01T12:01:00Z",
        )
        copied = self.root / "editor-copy.cpp"
        copied.write_bytes(self.source.read_bytes())
        release.attest_copyback(
            **self.state_kwargs(), generated_source=self.source,
            copied_back_source=copied,
            created_at_utc="2026-09-01T12:02:00Z",
        )
        release.record_play(
            **self.state_kwargs(), legal_stdout=True, expected_telemetry=True,
            created_at_utc="2026-09-01T12:03:00Z",
        )


class ReleaseBridgeTest(unittest.TestCase):
    def test_release_requires_the_finalist_configured_source_derivative(self):
        base = b"""\
inline constexpr std::size_t kRootPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr double kExploration = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr double kFinalVisitWeight = 1.0;
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
"""
        work = release.final_bridge.deployment.PROFILE_ROSTER["heavy"]
        selected_tuple = ["0.95", "0.75", "1"]
        deployed = release.final_bridge.deployment.derive_source(
            base, search_tuple=selected_tuple, profile="heavy", work=work
        )
        derivation = release.final_bridge.deployment.attest_derivation(
            base, deployed, search_tuple=selected_tuple,
            profile="heavy", work=work,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate_path = root / "candidate.cpp"
            candidate_path.write_bytes(deployed)
            manifest_value = release.final_bridge.deployment.create_manifest(
                base, deployed, search_tuple=selected_tuple,
                profile="heavy", work=work,
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    manifest_value, sort_keys=True, separators=(",", ":")
                ), encoding="ascii",
            )
            candidate_record = release._record(
                candidate_path, ascii_required=True
            )
            manifest_record = release._record(
                manifest_path, ascii_required=True
            )
            plan = {
                "inputs": {
                    "tuple": selected_tuple,
                    "profile": "heavy",
                    "profile_work": work,
                    "generated_source": derivation["base_source"],
                    "candidate": candidate_record,
                    "deployment_derivation": derivation,
                    "deployment_manifest": manifest_record,
                    "deployment_manifest_body_sha256": manifest_value[
                        "body_sha256"
                    ],
                },
                "configuration": {"deployment": derivation["configuration"]},
            }
            qualified = {
                "candidate": candidate_record,
                "deployment_manifest": manifest_record,
                "deployment_manifest_body_sha256": manifest_value["body_sha256"],
            }
            self.assertEqual(
                release._validate_deployment_binding(qualified, plan),
                derivation["configuration"],
            )
            wrong = {**qualified, "candidate": derivation["base_source"]}
            with self.assertRaisesRegex(release.ReleaseError, "derivative"):
                release._validate_deployment_binding(wrong, plan)

    def test_real_five_field_qualified_handoff_record_is_required(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            qualified = fixture.chain["qualified"]
            self.assertEqual(set(qualified["handoff"]), {
                "path", "bytes", "sha256", "body_sha256", "schema",
            })
            records = release._validate_qualified_records(
                qualified,
                campaign_root=fixture.root,
                handoff_validator=lambda path, **_kwargs: q.load_sealed(
                    path, release.adapter.HANDOFF_SCHEMA
                ),
            )
            self.assertEqual(records["handoff"], fixture.handoff.resolve())
            flattened = dict(qualified)
            flattened["handoff"] = q.artifact_reference(
                fixture.handoff, release.adapter.HANDOFF_SCHEMA
            )
            with self.assertRaises(Exception):
                release._validate_qualified_records(
                    flattened,
                    campaign_root=fixture.root,
                    handoff_validator=lambda path, **_kwargs: q.load_sealed(
                        path, release.adapter.HANDOFF_SCHEMA
                    ),
                )

    def test_green_ci_and_exact_qualified_chain_authorize_once(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            authorization = fixture.authorize()
            self.assertEqual(authorization["uploads_authorized"], 1)
            self.assertFalse(authorization["rank4_replacement_authorized"])
            self.assertEqual(
                authorization["ci"]["repository"],
                release.upload_primitives.REPOSITORY_SLUG,
            )
            self.assertEqual(
                authorization["ci"]["workflow_database_id"],
                release.upload_primitives.WORKFLOW_DATABASE_ID,
            )
            self.assertEqual(authorization["ci"]["attempt"], 1)
            self.assertEqual(authorization, fixture.authorize())
            inputs = q.load_sealed(
                fixture.root / release.RELEASE_DIRECTORY / "authorization-inputs.json",
                release.AUTH_INPUT_SCHEMA,
            )
            self.assertEqual(inputs["raw_shards"]["count"], 100)
            self.assertEqual(inputs["live_games_required"], 90)
            self.assertFalse(inputs["second_upload_authorized"])

    def test_authorization_only_interruption_resumes_without_new_authority(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.authorize()
            root = fixture.root / release.RELEASE_DIRECTORY
            authorization = q.load_sealed(
                root / "one-upload-authorization.json", q.UPLOAD_AUTH_SCHEMA
            )
            (root / "authorization-inputs.json").unlink()
            resumed = fixture.authorize()
            self.assertEqual(resumed, authorization)
            self.assertTrue((root / "authorization-inputs.json").is_file())

    def test_authorization_tamper_cannot_grant_second_upload(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.authorize()
            path = (
                fixture.root / release.RELEASE_DIRECTORY
                / "authorization-inputs.json"
            )
            value = q.load_sealed(path, release.AUTH_INPUT_SCHEMA)
            value.pop("body_sha256")
            value["second_upload_authorized"] = True
            path.write_bytes(q.canonical_json_bytes(q.seal(value)))
            with self.assertRaises(release.ReleaseError):
                release.validate_release_authorization(
                    fixture.root, repository=fixture.repository,
                    qualified_validator=fixture.validator,
                )

    def test_failed_or_wrong_head_ci_rejects_authorization(self):
        for payload in (gh_payload(head="d" * 40), gh_payload(conclusion="failure")):
            with self.subTest(payload=payload["headSha"]), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(pathlib.Path(raw))
                release.freeze_live_exclusions(
                    fixture.root, registry_path=fixture.registry,
                    frozen_at_utc="2026-09-01T11:00:00Z",
                )
                with self.assertRaises(Exception):
                    release.seal_ci_evidence(
                        fixture.root, gh_payload=payload, expected_head=COMMIT,
                        fetched_at_utc="2026-09-01T11:30:00Z",
                    )

    def test_wrong_workflow_attempt_and_repository_ci_provenance_rejects(self):
        mutations = (
            ("workflowDatabaseId", release.upload_primitives.WORKFLOW_DATABASE_ID + 1),
            ("workflowDatabaseId", True),
            ("attempt", 2),
            ("attempt", True),
            ("url", "https://github.com/other/repository/actions/runs/123"),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(pathlib.Path(raw))
                payload = gh_payload()
                payload[field] = value
                if field == "workflowDatabaseId":
                    self.assertEqual(payload["workflowName"], "CI and Pages")
                release.freeze_live_exclusions(
                    fixture.root, registry_path=fixture.registry,
                    frozen_at_utc="2026-09-01T11:00:00Z",
                )
                with self.assertRaises(Exception):
                    release.seal_ci_evidence(
                        fixture.root, gh_payload=payload, expected_head=COMMIT,
                        fetched_at_utc="2026-09-01T11:30:00Z",
                    )
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            payload = gh_payload()
            payload["jobs"][0]["url"] = (
                "https://github.com/other/repository/actions/runs/123/job/1"
            )
            release.freeze_live_exclusions(
                fixture.root, registry_path=fixture.registry,
                frozen_at_utc="2026-09-01T11:00:00Z",
            )
            with self.assertRaises(Exception):
                release.seal_ci_evidence(
                    fixture.root, gh_payload=payload, expected_head=COMMIT,
                    fetched_at_utc="2026-09-01T11:30:00Z",
                )

    def test_incomplete_qualified_context_and_late_exclusion_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.prereqs()
            root = fixture.root / release.RELEASE_DIRECTORY
            with self.assertRaisesRegex(release.ReleaseError, "incomplete"):
                release.authorize_release(
                    fixture.root, qualified_path=fixture.qualified,
                    ci_evidence_path=root / "github-ci.json",
                    live_exclusion_binding_path=root / "live-exclusion-binding.json",
                    repository=fixture.repository,
                    authorized_at_utc="2026-09-01T12:00:00Z",
                    qualified_validator=lambda *_args: {},
                )

        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            release.freeze_live_exclusions(
                fixture.root, registry_path=fixture.registry,
                frozen_at_utc="2026-09-01T12:01:00Z",
            )
            release.seal_ci_evidence(
                fixture.root, gh_payload=gh_payload(), expected_head=COMMIT,
                fetched_at_utc="2026-09-01T11:30:00Z",
            )
            root = fixture.root / release.RELEASE_DIRECTORY
            with self.assertRaisesRegex(release.ReleaseError, "predates"):
                release.authorize_release(
                    fixture.root, qualified_path=fixture.qualified,
                    ci_evidence_path=root / "github-ci.json",
                    live_exclusion_binding_path=root / "live-exclusion-binding.json",
                    repository=fixture.repository,
                    authorized_at_utc="2026-09-01T12:00:00Z",
                    qualified_validator=fixture.validator,
                )

    def test_copyback_and_play_order_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.authorize()
            with self.assertRaises(Exception):
                release.record_play(
                    **fixture.state_kwargs(), legal_stdout=True,
                    expected_telemetry=True,
                    created_at_utc="2026-09-01T12:01:00Z",
                )
            release.fresh_editor(
                **fixture.state_kwargs(), session_id="fresh",
                opened_at_utc="2026-09-01T12:01:00Z",
            )
            bad = fixture.root / "bad-copy.cpp"
            bad.write_text("different\n", encoding="ascii")
            with self.assertRaises(Exception):
                release.attest_copyback(
                    **fixture.state_kwargs(), generated_source=fixture.source,
                    copied_back_source=bad,
                    created_at_utc="2026-09-01T12:02:00Z",
                )
            with self.assertRaisesRegex(release.ReleaseError, "qualified"):
                release.attest_copyback(
                    **fixture.state_kwargs(), generated_source=fixture.generated,
                    copied_back_source=fixture.generated,
                    created_at_utc="2026-09-01T12:02:00Z",
                )

    def test_failed_play_is_terminal(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.authorize()
            release.fresh_editor(
                **fixture.state_kwargs(), session_id="fresh",
                opened_at_utc="2026-09-01T12:01:00Z",
            )
            copied = fixture.root / "copy.cpp"
            copied.write_bytes(fixture.source.read_bytes())
            release.attest_copyback(
                **fixture.state_kwargs(), generated_source=fixture.source,
                copied_back_source=copied,
                created_at_utc="2026-09-01T12:02:00Z",
            )
            release.record_play(
                **fixture.state_kwargs(), legal_stdout=False,
                expected_telemetry=True,
                created_at_utc="2026-09-01T12:03:00Z",
            )
            with self.assertRaisesRegex(release.ReleaseError, "forbids"):
                release.start_submit(
                    **fixture.state_kwargs(), started_at_utc="2026-09-01T12:04:00Z"
                )

    def test_ambiguous_submit_never_allows_second_click_and_needs_unique_resolution(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.ready()
            release.start_submit(
                **fixture.state_kwargs(), started_at_utc="2026-09-01T12:04:00Z"
            )
            release.record_ambiguous(
                **fixture.state_kwargs(), observed_at_utc="2026-09-01T12:05:00Z",
                evidence={"network": "uncertain"},
            )
            with self.assertRaises(Exception):
                release.start_submit(
                    **fixture.state_kwargs(), started_at_utc="2026-09-01T12:06:00Z"
                )
            with self.assertRaises(Exception):
                release.attest_submission(
                    **fixture.state_kwargs(), agent_id=AGENT,
                    submission_id=SUBMISSION,
                    submitted_at_utc="2026-09-01T12:06:00Z",
                )
            result = release.attest_submission(
                **fixture.state_kwargs(), agent_id=AGENT,
                submission_id=SUBMISSION,
                submitted_at_utc="2026-09-01T12:06:00Z",
                ambiguity_resolution={
                    "matching_submissions": 1,
                    "agent_id": AGENT,
                    "submission_id": SUBMISSION,
                },
            )
            self.assertEqual(result["submit_clicks"], 1)
            self.assertEqual(result, release.attest_submission(
                **fixture.state_kwargs(), agent_id=AGENT,
                submission_id=SUBMISSION,
                submitted_at_utc="2026-09-01T12:07:00Z",
                ambiguity_resolution={
                    "matching_submissions": 1,
                    "agent_id": AGENT,
                    "submission_id": SUBMISSION,
                },
            ))

    def test_completion_requires_exact_bound_90_game_window(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.ready()
            release.start_submit(
                **fixture.state_kwargs(), started_at_utc="2026-09-01T12:04:00Z"
            )
            release.attest_submission(
                **fixture.state_kwargs(), agent_id=AGENT,
                submission_id=SUBMISSION,
                submitted_at_utc="2026-09-01T12:05:00Z",
            )
            data_root = fixture.root / release.RELEASE_DIRECTORY / "live-window"
            data_root.mkdir()
            reference = data_root / "live-window.reference.json"
            attestation = (
                fixture.root / release.RELEASE_DIRECTORY
                / "upload/05-submission-attested.json"
            )
            receipt_path, receipt = release.live.write_content_addressed(
                data_root / "window-receipts", {
                    "schema": release.live.WINDOW_RECEIPT_SCHEMA,
                    "summary": {
                        "status": "complete-accepted-diagnostic",
                        "focus_operational_failure_games": 0,
                        "opponent_operational_failure_games": 0,
                        "opponent_failure_games_counted_as_strength_wins": 0,
                    },
                },
            )
            release.live.write_sealed(reference, {
                "schema": release.live.WINDOW_REFERENCE_SCHEMA,
                "namespace": q.NAMESPACE,
                "receipt": {
                    "path": str(receipt_path.resolve()),
                    "sha256": release.live.sha256_file(receipt_path),
                    "body_sha256": receipt["body_sha256"],
                },
                "status": "complete-accepted-diagnostic",
                "exact_games": 90,
                "training_eligible": False,
                "rollback_authorized": False,
                "second_upload_authorized": False,
            })

            def complete(_path, *, data_root):
                return {
                    "status": "complete-accepted-diagnostic",
                    "exact_games": 90,
                    "training_eligible": False,
                    "rollback_authorized": False,
                    "second_upload_authorized": False,
                    "submission_attestation": release.live.artifact_reference(
                        attestation, q.UPLOAD_EVENT_SCHEMA
                    ),
                }

            result = release.verify_completion(
                **fixture.state_kwargs(), verified_at_utc="2026-09-01T13:00:00Z",
                live_verifier=complete,
            )
            self.assertEqual(result["uploads_completed"], 1)
            self.assertEqual(result["live_games"], 90)
            self.assertFalse(result["second_upload_authorized"])
            self.assertEqual(result, release.verify_completion(
                **fixture.state_kwargs(), verified_at_utc="2026-09-01T14:00:00Z",
                live_verifier=complete,
            ))

            def incomplete(_path, *, data_root):
                return {**complete(_path, data_root=data_root), "exact_games": 89}

            with self.assertRaises(release.ReleaseError):
                release.validate_completion(
                    **fixture.state_kwargs(), live_verifier=incomplete
                )

            publication = release.publish_release(
                **fixture.state_kwargs(), live_verifier=complete
            )
            self.assertEqual(publication["upload"]["count"], 1)
            self.assertEqual(publication["live"]["games"], 90)
            rendered = q.canonical_json_bytes(publication).decode("ascii")
            for forbidden in ("canonical_sha256", "game_ids", "transcript", "seed_hex"):
                self.assertNotIn(forbidden, rendered)

    def test_second_submission_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            fixture.ready()
            release.start_submit(
                **fixture.state_kwargs(), started_at_utc="2026-09-01T12:04:00Z"
            )
            release.attest_submission(
                **fixture.state_kwargs(), agent_id=AGENT,
                submission_id=SUBMISSION,
                submitted_at_utc="2026-09-01T12:05:00Z",
            )
            with self.assertRaisesRegex(release.ReleaseError, "another identity"):
                release.attest_submission(
                    **fixture.state_kwargs(), agent_id=AGENT,
                    submission_id=SUBMISSION + 1,
                    submitted_at_utc="2026-09-01T12:06:00Z",
                )

    def test_release_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            external = fixture.root / "external"
            external.mkdir()
            (fixture.root / release.RELEASE_DIRECTORY).symlink_to(
                external, target_is_directory=True
            )
            with self.assertRaises(release.ReleaseError):
                release.freeze_live_exclusions(
                    fixture.root, registry_path=fixture.registry,
                    frozen_at_utc="2026-09-01T11:00:00Z",
                )

    def test_foreign_upload_state_rejects_before_authorization(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            foreign = fixture.ledger / "upload-authorization"
            foreign.mkdir()
            fixture.prereqs()
            root = fixture.root / release.RELEASE_DIRECTORY
            with self.assertRaisesRegex(release.ReleaseError, "exactly-one"):
                release.authorize_release(
                    fixture.root, qualified_path=fixture.qualified,
                    ci_evidence_path=root / "github-ci.json",
                    live_exclusion_binding_path=root / "live-exclusion-binding.json",
                    repository=fixture.repository,
                    authorized_at_utc="2026-09-01T12:00:00Z",
                    qualified_validator=fixture.validator,
                )

    def test_every_fixed_release_route_rejects_symlink_and_irregular_types(self):
        file_routes = (
            "authorization-inputs.json", "one-upload-authorization.json",
            "github-ci.json", "live-exclusion-binding.json",
            "00-fresh-editor.json", "upload/00-prepared.json",
            "upload/01-editor-copyback.json", "upload/02-play.json",
            "upload/03-submit-started.json", "upload/04-submit-ambiguous.json",
            "upload/05-submission-attested.json", "completion.json",
            "publication.json", "live-window/live-window.reference.json",
        )
        directory_routes = ("upload", "live-window")
        for relative in file_routes:
            with self.subTest(kind="file-symlink", route=relative), \
                    tempfile.TemporaryDirectory() as raw:
                campaign = pathlib.Path(raw).resolve()
                root = release.release_root(campaign, create=True)
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                external = campaign / "external-file"
                external.write_text("external\n", encoding="ascii")
                path.symlink_to(external)
                with self.assertRaises(release.ReleaseError):
                    release._fixed_file(root, relative, relative)
            with self.subTest(kind="file-is-directory", route=relative), \
                    tempfile.TemporaryDirectory() as raw:
                campaign = pathlib.Path(raw).resolve()
                root = release.release_root(campaign, create=True)
                path = root / relative
                path.mkdir(parents=True)
                with self.assertRaises(release.ReleaseError):
                    release._fixed_file(root, relative, relative)
        for relative in directory_routes:
            with self.subTest(kind="directory-symlink", route=relative), \
                    tempfile.TemporaryDirectory() as raw:
                campaign = pathlib.Path(raw).resolve()
                root = release.release_root(campaign, create=True)
                external = campaign / "external-directory"
                external.mkdir()
                (root / relative).symlink_to(external, target_is_directory=True)
                with self.assertRaises(release.ReleaseError):
                    release._fixed_directory(
                        root, relative, create=False, label=relative
                    )
            with self.subTest(kind="directory-is-file", route=relative), \
                    tempfile.TemporaryDirectory() as raw:
                campaign = pathlib.Path(raw).resolve()
                root = release.release_root(campaign, create=True)
                (root / relative).write_text("irregular\n", encoding="ascii")
                with self.assertRaises(release.ReleaseError):
                    release._fixed_directory(
                        root, relative, create=False, label=relative
                    )

        for ancestor in ("upload", "live-window"):
            with self.subTest(kind="ancestor-symlink", route=ancestor), \
                    tempfile.TemporaryDirectory() as raw:
                campaign = pathlib.Path(raw).resolve()
                root = release.release_root(campaign, create=True)
                external = campaign / "external-ancestor"
                external.mkdir()
                (root / ancestor).symlink_to(external, target_is_directory=True)
                with self.assertRaises(release.ReleaseError):
                    release._fixed_output(root, f"{ancestor}/child.json")

    def test_flat_upload_ledger_rejects_every_unexpected_child_everywhere(self):
        def prepare_submission(fixture):
            fixture.ready()
            release.start_submit(
                **fixture.state_kwargs(), started_at_utc="2026-09-01T12:04:00Z"
            )
            release.attest_submission(
                **fixture.state_kwargs(), agent_id=AGENT,
                submission_id=SUBMISSION,
                submitted_at_utc="2026-09-01T12:05:00Z",
            )

        def install_live(fixture):
            data_root = fixture.root / release.RELEASE_DIRECTORY / "live-window"
            data_root.mkdir()
            attestation = (
                fixture.root / release.RELEASE_DIRECTORY
                / "upload/05-submission-attested.json"
            )
            receipt_path, receipt = release.live.write_content_addressed(
                data_root / "window-receipts", {
                    "schema": release.live.WINDOW_RECEIPT_SCHEMA,
                    "summary": {
                        "status": "complete-accepted-diagnostic",
                        "focus_operational_failure_games": 0,
                        "opponent_operational_failure_games": 0,
                        "opponent_failure_games_counted_as_strength_wins": 0,
                    },
                },
            )
            reference = data_root / "live-window.reference.json"
            release.live.write_sealed(reference, {
                "schema": release.live.WINDOW_REFERENCE_SCHEMA,
                "namespace": q.NAMESPACE,
                "receipt": {
                    "path": str(receipt_path.resolve()),
                    "sha256": release.live.sha256_file(receipt_path),
                    "body_sha256": receipt["body_sha256"],
                },
                "status": "complete-accepted-diagnostic",
                "exact_games": 90,
                "training_eligible": False,
                "rollback_authorized": False,
                "second_upload_authorized": False,
            })

            def verifier(_path, *, data_root):
                return {
                    "status": "complete-accepted-diagnostic",
                    "exact_games": 90,
                    "training_eligible": False,
                    "rollback_authorized": False,
                    "second_upload_authorized": False,
                    "submission_attestation": release.live.artifact_reference(
                        attestation, q.UPLOAD_EVENT_SCHEMA
                    ),
                }

            return verifier

        def inject(kind, fixture):
            ledger = fixture.root / release.RELEASE_DIRECTORY / "upload"
            path = ledger / f"foreign-{kind}"
            if kind == "directory":
                path.mkdir()
            elif kind == "non-json-file":
                path.write_bytes(b"not json\n")
            elif kind == "special-fifo":
                os.mkfifo(path)
            else:
                external = fixture.root / "external-event"
                external.write_text("external\n", encoding="ascii")
                path.symlink_to(external)

        for operation in ("attestation", "completion", "publication"):
            for kind in ("directory", "non-json-file", "special-fifo", "symlink"):
                with self.subTest(operation=operation, kind=kind), \
                        tempfile.TemporaryDirectory() as raw:
                    fixture = Fixture(pathlib.Path(raw))
                    prepare_submission(fixture)
                    verifier = install_live(fixture)
                    if operation == "publication":
                        release.verify_completion(
                            **fixture.state_kwargs(),
                            verified_at_utc="2026-09-01T13:00:00Z",
                            live_verifier=verifier,
                        )
                    state = release.validate_release_authorization(
                        fixture.root, repository=fixture.repository,
                        qualified_validator=fixture.validator,
                    )
                    inject(kind, fixture)
                    with self.assertRaises(release.ReleaseError):
                        if operation == "attestation":
                            release._submission_attestation(state)
                        elif operation == "completion":
                            release.verify_completion(
                                **fixture.state_kwargs(),
                                verified_at_utc="2026-09-01T13:00:00Z",
                                live_verifier=verifier,
                            )
                        else:
                            release.publish_release(
                                **fixture.state_kwargs(), live_verifier=verifier
                            )


if __name__ == "__main__":
    unittest.main()
