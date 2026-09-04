import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_upload.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_upload", TOOL)
upload = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(upload)
q = upload.qualification
final = upload.final_tools
campaign = final.campaign
preflight = upload.preflight_tools
RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
COMMIT = "e" * 40


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def gh_run(head=COMMIT):
    jobs = [
        {"name": name, "status": "completed", "conclusion": "success",
         "databaseId": index + 10,
         "url": f"{upload.RUN_URL_PREFIX}12345/job/{index + 10}"}
        for index, name in enumerate(upload.JOB_NAMES)
    ]
    jobs.append({"name": "deploy", "status": "completed", "conclusion": "skipped"})
    return {
        "databaseId": 12345,
        "workflowDatabaseId": upload.WORKFLOW_DATABASE_ID,
        "attempt": 1,
        "name": "CI and Pages",
        "workflowName": "CI and Pages",
        "event": "workflow_dispatch",
        "headBranch": "compact-value-bfm",
        "headSha": head,
        "status": "completed",
        "conclusion": "success",
        "url": f"{upload.RUN_URL_PREFIX}12345",
        "jobs": jobs,
    }


def engine():
    return {
        "decisions": 0, "deadline_stops": 0, "soft_overruns": 0,
        "headroom_failures": 0, "hard_timeouts": 0, "work": 0,
        "generated_children": 0, "evaluated_children": 0,
        "maximum_first_ms": 800.0, "maximum_later_ms": 155.0,
        "times_ms": [],
    }


def gate_document(index, candidate_sha, bank_sha):
    games = []
    for pair in range(index * 5, index * 5 + 5):
        for color in (0, 1):
            won = pair < (260 if color == 0 else 267)
            games.append({
                "opening_id": f"opening-{pair}", "pair_index": pair,
                "candidate_player": color,
                "winner": color if won else 1 - color,
                "turns": 100, "failure": None,
                "candidate": engine(), "rank4": engine(),
            })
    wins = sum(game["winner"] == game["candidate_player"] for game in games)
    color0 = sum(game["winner"] == 0 and game["candidate_player"] == 0
                 for game in games)
    color1 = sum(game["winner"] == 1 and game["candidate_player"] == 1
                 for game in games)
    merged = engine()
    return {
        "schema": upload.gate_support.LEGACY_RESULT_SCHEMA,
        "bindings": {
            "candidate_source_sha256": candidate_sha,
            "candidate_source_bytes": 10,
            "candidate_runtime_body_sha256": digest("body"),
            "candidate_payload_sha256": digest("payload"),
            "rank4_source_sha256": q.RANK4_SHA256,
            "rank4_source_bytes": q.RANK4_BYTES,
            "opponent_sha256": q.RANK4_SHA256,
            "bank_sha256": bank_sha, "bank_bytes": 10,
        },
        "config": {
            "mode": "actual-clock", "pair_offset": index * 5,
            "pair_count": 5, "candidate_clocks_ms": [800, 155],
            "rank4_clocks_ms": [800, 165], "max_turns": 320,
            "minimum_candidate_wins": -1, "minimum_wins_per_color": -1,
        },
        "games": games,
        "result": {
            "games": 10, "candidate_wins": wins,
            "candidate_wins_player0": color0,
            "candidate_wins_player1": color1,
            "rank4_wins": 10 - wins, "failures": 0, "unfinished": 0,
            "failure_categories": {}, "candidate": merged,
            "rank4": merged, "passed": True,
        },
    }


def content_addressed(root, payload):
    value = q.seal(payload)
    raw = q.canonical_json_bytes(value)
    path = root / f"{q.sha256_bytes(raw)}.json"
    path.write_bytes(raw)
    return path


class FinalFixture:
    def __init__(self, root):
        self.root = root
        self.ledger = root / "ledger"
        self.candidate = root / "submission.cpp"
        self.candidate.write_text("int main(){return 0;}\n", encoding="ascii")
        self.source_binding = root / "source-binding.json"
        q.create_source_binding(
            self.source_binding, candidate_source=self.candidate,
            candidate_commit=COMMIT, rank4_source=RANK4,
            opponent_source=RANK4,
        )
        source = q.load_sealed(self.source_binding, q.SOURCE_BINDING_SCHEMA)
        self.adapter = root / "bank-adapter.json"
        q.write_sealed(self.adapter, {
            "schema": q.FINAL_BANK_SCHEMA, "namespace": q.NAMESPACE,
            "source_binding": q.artifact_reference(
                self.source_binding, q.SOURCE_BINDING_SCHEMA
            ), "candidate_commit": COMMIT,
            "candidate_sha256": source["candidate"]["sha256"],
            "rank4_sha256": q.RANK4_SHA256, "opening_count": 500,
        })
        self.gate = root / "rank4-gate"
        self.gate.write_text("gate")
        self.binding = root / "gate-binding.json"
        q.create_gate_binding(
            self.binding, source_binding_path=self.source_binding,
            bank_path=self.adapter, harness_path=self.gate,
        )
        self.gate_bank = root / "gate-bank.tsv"
        self.gate_bank.write_text("opening_id\ttranscript\n")
        self.runtime = root / "runtime.json"
        self.runtime.write_text(json.dumps({
            "body_sha256": digest("body"),
            "quantization": {"payload_sha256": digest("payload")},
        }) + "\n")
        self.preflight = self._preflight(source)
        self.plan = root / "final-plan.json"
        q.write_sealed(self.plan, {
            "schema": final.PLAN_SCHEMA, "namespace": q.NAMESPACE,
            "candidate_commit": COMMIT, "candidate": source["candidate"],
            "rank4": source["rank4"],
            "gate_binding": q.artifact_reference(
                self.binding, q.GATE_BINDING_SCHEMA
            ),
            "gate_bank": {"path": str(self.gate_bank.resolve()),
                          "sha256": q.sha256_file(self.gate_bank)},
            "protected_bank": {"path": "/protected", "sha256": digest("protected")},
            "runtime": {"path": str(self.runtime.resolve()),
                        "bytes": self.runtime.stat().st_size,
                        "sha256": q.sha256_file(self.runtime), "ascii": True},
            "rank4_gate": {"path": str(self.gate.resolve()),
                           "bytes": self.gate.stat().st_size,
                           "sha256": q.sha256_file(self.gate)},
            "preflight": q.artifact_reference(self.preflight),
            "selection": {"path": "/selection", "sha256": digest("selection")},
        })
        self.consumption = self.ledger / "bank-consumed-at-launch.json"
        q.write_sealed(self.consumption, {
            "schema": final.CONSUMPTION_SCHEMA, "namespace": q.NAMESPACE,
            "status": "bank-consumed-at-launch", "launched_at_utc": "2026-08-31T10:00:00Z",
            "plan": q.artifact_reference(self.plan, final.PLAN_SCHEMA),
            "protected_bank": q.load_sealed(self.plan)["protected_bank"],
            "gate_bank": q.load_sealed(self.plan)["gate_bank"],
            "gate_binding": q.load_sealed(self.plan)["gate_binding"],
            "one_launch_only": True,
        })
        self._shards()
        q.aggregate_final(
            self.ledger, binding_path=self.binding,
            uncontended_timing={"first_max_ms": 800.0, "later_max_ms": 155.0},
            completed_at_utc="2026-08-31T11:00:00Z",
        )
        self.qualified = self.ledger / "rank4-qualified-inputs.json"
        q.write_sealed(self.qualified, {
            "schema": final.QUALIFIED_INPUT_SCHEMA, "namespace": q.NAMESPACE,
            "status": "rank4-qualified-awaiting-green-ci",
            "candidate_commit": COMMIT, "candidate": source["candidate"],
            "selection": q.load_sealed(self.plan)["selection"],
            "preflight": q.artifact_reference(self.preflight),
            "final_plan": q.artifact_reference(self.plan, final.PLAN_SCHEMA),
            "aggregate": q.artifact_reference(
                self.ledger / "aggregate.json", q.FINAL_AGGREGATE_SCHEMA
            ), "one_upload_authorization_requires_green_ci": True,
            "rank4_replacement_authorized": False,
        })
        self.ci = upload.authorization_directory(self.ledger) / "github-ci.json"
        upload.seal_ci_evidence(
            self.ci, gh_payload=gh_run(), expected_head=COMMIT,
            fetched_at_utc="2026-08-31T12:00:00Z",
        )

    def _preflight(self, source):
        samples = []
        for count in (1, 2, 10):
            for color in (0, 1):
                for replica in range(count):
                    samples.append({"process_count": count, "color": color,
                                    "replica": replica, "first_ms": 800.0,
                                    "later_max_ms": 155.0})
        return content_addressed(self.root, {
            "schema": preflight.RECEIPT_SCHEMA, "namespace": q.NAMESPACE,
            "status": "passed",
            "inputs_before": {"candidate_commit": COMMIT,
                              "candidate": source["candidate"]},
            "inputs_after": {"candidate_commit": COMMIT,
                             "candidate": source["candidate"]},
            "checks": {"all": "passed"},
            "timing": {"schema": preflight.TIMING_SCHEMA,
                       "probe_sha256": digest("probe"),
                       "first_limit_exclusive_ms": 900.0,
                       "later_limit_exclusive_ms": 180.0,
                       "samples": samples},
            "parity": {"schema": preflight.PARITY_SCHEMA,
                       "states": 4096, "feature_states": 4096,
                       "features_sha256": digest("features"),
                       "cpp_sha256": digest("cpp"),
                       "scalar_sha256": digest("scalar"),
                       "maximum_absolute_error": 0.000001,
                       "all_finite": True},
            "protected_banks_accessed": [], "git_writes": 0, "uploads": 0,
        })

    def _shards(self):
        plan = q.load_sealed(self.plan)
        candidate_sha = q.sha256_file(self.candidate)
        bank_sha = q.sha256_file(self.gate_bank)
        # Raw results do not require the final plan to be materialized yet.
        for index in range(100):
            q.start_final_shard(
                self.ledger, binding_path=self.binding, index=index,
                started_at_utc="2026-08-31T10:00:00Z",
            )
            raw = self.ledger / "raw" / f"shard-{index:03d}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(gate_document(index, candidate_sha, bank_sha)))
            document = upload.gate_support.validate_result(
                raw, expected_bank_sha256=bank_sha,
                expected_candidate_sha256=candidate_sha,
                allow_legacy_attempt_zero=True,
            )
            games = []
            for game in document["games"]:
                games.append({
                    "pair_index": game["pair_index"],
                    "candidate_color": game["candidate_player"],
                    "candidate_win": game["winner"] == game["candidate_player"],
                    "turns": game["turns"], "failure": None,
                    "first_ms": 800.0, "later_max_ms": 155.0,
                })
            evidence = self.ledger / "raw-evidence" / f"shard-{index:03d}.json"
            q.write_sealed(evidence, {
                "schema": final.RAW_SHARD_EVIDENCE_SCHEMA,
                "namespace": q.NAMESPACE,
                "plan": q.artifact_reference(self.plan, final.PLAN_SCHEMA),
                "rank4_gate": plan["rank4_gate"],
                "shard_index": index,
                "raw_gate_result": {
                    "path": str(raw.resolve()), "bytes": raw.stat().st_size,
                    "sha256": q.sha256_file(raw),
                },
                "normalized_games_sha256": q.sha256_bytes(
                    q.canonical_json_bytes(games)
                ),
                "gate_result_validated_before_normalization": True,
            })
            q.record_shard_receipt(
                self.ledger, binding_path=self.binding, index=index,
                games=games, completed_at_utc="2026-08-31T10:01:00Z",
                evidence=q.artifact_reference(
                    evidence, final.RAW_SHARD_EVIDENCE_SCHEMA
                ),
            )

    def authorize(self):
        with mock.patch.object(
            upload.preflight_tools,
            "validate_preflight_receipt",
            return_value={},
        ):
            return upload.authorize_upload(
                self.ledger, qualified_inputs_path=self.qualified,
                final_plan_path=self.plan, consumption_path=self.consumption,
                preflight_path=self.preflight, ci_evidence_path=self.ci,
                authorized_at_utc="2026-08-31T12:01:00Z",
            )


class GitHubEvidenceTest(unittest.TestCase):
    def test_real_job_names_map_to_exact_five_job_ids(self):
        normalized = upload.validate_gh_run(gh_run(), expected_head=COMMIT)
        self.assertEqual(tuple(normalized["jobs"]), upload.REQUIRED_JOB_IDS)
        self.assertEqual(normalized["head_branch"], "compact-value-bfm")
        self.assertEqual(normalized["repository"], upload.REPOSITORY_SLUG)
        self.assertEqual(
            normalized["workflow_database_id"], upload.WORKFLOW_DATABASE_ID
        )
        self.assertEqual(normalized["attempt"], 1)

    def test_wrong_event_branch_head_or_failed_job_is_rejected(self):
        for field, value in (
            ("event", "push"), ("headBranch", "main"),
            ("headSha", "0" * 40),
        ):
            payload = gh_run()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(upload.UploadError):
                upload.validate_gh_run(payload, expected_head=COMMIT)
        payload = gh_run()
        payload["jobs"][0]["conclusion"] = "failure"
        with self.assertRaisesRegex(upload.UploadError, "did not pass"):
            upload.validate_gh_run(payload, expected_head=COMMIT)

    def test_bool_run_id_is_rejected(self):
        payload = gh_run()
        payload["databaseId"] = True
        with self.assertRaises(upload.UploadError):
            upload.validate_gh_run(payload, expected_head=COMMIT)

    def test_wrong_or_bool_workflow_identity_and_attempt_are_rejected(self):
        for field, value in (
            ("workflowDatabaseId", upload.WORKFLOW_DATABASE_ID + 1),
            ("workflowDatabaseId", True),
            ("attempt", 2),
            ("attempt", True),
        ):
            payload = gh_run()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(
                upload.UploadError
            ):
                upload.validate_gh_run(payload, expected_head=COMMIT)

    def test_same_name_wrong_workflow_and_wrong_repository_urls_are_rejected(self):
        payload = gh_run()
        payload["workflowDatabaseId"] += 1
        self.assertEqual(payload["workflowName"], upload.WORKFLOW_NAME)
        with self.assertRaises(upload.UploadError):
            upload.validate_gh_run(payload, expected_head=COMMIT)
        for target in ("run", "job"):
            payload = gh_run()
            if target == "run":
                payload["url"] = "https://github.com/other/repository/actions/runs/12345"
            else:
                payload["jobs"][0]["url"] = (
                    "https://github.com/other/repository/actions/runs/12345/job/10"
                )
            with self.subTest(target=target), self.assertRaises(upload.UploadError):
                upload.validate_gh_run(payload, expected_head=COMMIT)

    def test_fetch_requests_authoritative_workflow_fields_and_repository(self):
        payload = gh_run()
        completed = upload.subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps(payload).encode("ascii"), stderr=b"",
        )
        with mock.patch.object(upload.subprocess, "run", return_value=completed) as run:
            self.assertEqual(upload.fetch_gh_run(12345), payload)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:4], ["gh", "run", "view", "12345"])
        self.assertEqual(argv[argv.index("--repo") + 1], upload.REPOSITORY_SLUG)
        fields = argv[argv.index("--json") + 1].split(",")
        self.assertIn("workflowDatabaseId", fields)
        self.assertIn("attempt", fields)

    def test_sealed_evidence_revalidates_repository_workflow_and_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "ci.json"
            evidence = upload.seal_ci_evidence(
                path, gh_payload=gh_run(), expected_head=COMMIT,
                fetched_at_utc="2026-09-02T01:00:00Z",
            )
            self.assertEqual(
                upload.validate_ci_evidence(path, expected_head=COMMIT), evidence
            )
            for field, value in (
                ("repository", "other/repository"),
                ("workflow_database_id", True),
                ("attempt", 2),
            ):
                changed = dict(evidence)
                changed.pop("body_sha256")
                changed[field] = value
                path.write_bytes(q.canonical_json_bytes(q.seal(changed)))
                with self.subTest(field=field), self.assertRaises(upload.UploadError):
                    upload.validate_ci_evidence(path, expected_head=COMMIT)
                path.write_bytes(q.canonical_json_bytes(evidence))


class AuthorizationChainTest(unittest.TestCase):
    def test_terminal_integrity_marker_forbids_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalFixture(pathlib.Path(temporary))
            blocker = (
                fixture.ledger / "iteration-governance" / "iteration" /
                "02-integrity-failure.json"
            )
            q.write_sealed(blocker, {
                "schema": (
                    "papersoccer.compact-value-bfm."
                    "iteration-integrity-failure.v1"
                ),
                "namespace": q.NAMESPACE,
                "status": "terminal-precompletion-integrity-failure",
                "upload_authorized": False,
            })
            with self.assertRaisesRegex(
                upload.UploadError, "terminal integrity blocker"
            ):
                fixture.authorize()

    def test_authorization_revalidates_all_100_raw_bound_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalFixture(pathlib.Path(temporary))
            authorization = fixture.authorize()
            self.assertEqual(authorization["uploads_authorized"], 1)
            inputs = q.load_sealed(
                upload.authorization_directory(fixture.ledger) /
                "authorization-inputs.json", upload.AUTH_INPUT_SCHEMA
            )
            self.assertEqual(len(inputs["raw_shards"]), 100)

    def test_missing_or_tampered_raw_shard_rejects_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalFixture(pathlib.Path(temporary))
            raw = fixture.ledger / "raw/shard-009.json"
            raw.unlink()
            with self.assertRaisesRegex(upload.UploadError, "absent|changed"):
                fixture.authorize()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalFixture(pathlib.Path(temporary))
            raw = fixture.ledger / "raw/shard-009.json"
            document = json.loads(raw.read_text())
            document["games"][0]["winner"] = 1 - document["games"][0]["winner"]
            raw.write_text(json.dumps(document))
            with self.assertRaises(Exception):
                fixture.authorize()


class EventFixture:
    def __init__(self, root):
        self.root = root
        self.ledger = root / "ledger"
        self.source = root / "submission.cpp"
        self.source.write_text("int main(){}\n")
        raw = self.source.read_bytes()
        directory = upload.authorization_directory(self.ledger)
        self.authorization = directory / "one-upload-authorization.json"
        q.write_sealed(self.authorization, {
            "schema": q.UPLOAD_AUTH_SCHEMA, "namespace": q.NAMESPACE,
            "uploads_authorized": 1, "rank4_replacement_authorized": False,
            "candidate_commit": COMMIT,
            "candidate": {"path": str(self.source.resolve()), "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(), "ascii": True},
            "binding": {"path": "/binding", "sha256": digest("binding")},
            "aggregate": {"path": "/aggregate", "sha256": digest("aggregate")},
            "ci": {"conclusion": "success"},
            "upload_ledger_root": str(directory.resolve()),
        })
        q.write_sealed(directory / "authorization-inputs.json", {
            "schema": upload.AUTH_INPUT_SCHEMA, "namespace": q.NAMESPACE,
            "status": "one-upload-authorized",
            "authorized_at_utc": "2026-08-31T12:00:00Z",
            "authorization_directory": str(directory),
            "authorization": q.artifact_reference(
                self.authorization, q.UPLOAD_AUTH_SCHEMA
            ), "uploads_authorized": 1,
        })

    def ready(self):
        upload.fresh_editor(
            self.ledger, session_id="fresh-session",
            opened_at_utc="2026-08-31T13:00:00Z",
        )
        copied = self.root / "copy.cpp"
        copied.write_bytes(self.source.read_bytes())
        upload.attest_copyback(
            self.ledger, generated_source=self.source,
            copied_back_source=copied,
            created_at_utc="2026-08-31T13:01:00Z",
        )
        upload.record_play(
            self.ledger, legal_stdout=True, expected_telemetry=True,
            created_at_utc="2026-08-31T13:02:00Z",
        )


class UploadEventTest(unittest.TestCase):
    def test_fixed_directory_orders_copyback_play_submit_and_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EventFixture(pathlib.Path(temporary))
            fixture.ready()
            upload.start_submit(
                fixture.ledger, started_at_utc="2026-08-31T13:03:00Z"
            )
            attestation = upload.attest_submission(
                fixture.ledger, agent_id=701, submission_id=801,
                submitted_at_utc="2026-08-31T13:04:00Z",
            )
            self.assertEqual(attestation["submit_clicks"], 1)
            other = fixture.root / "other-ledger"
            with self.assertRaises(Exception):
                upload.fresh_editor(
                    other, session_id="second",
                    opened_at_utc="2026-08-31T13:05:00Z",
                )

    def test_failed_play_is_terminal_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EventFixture(pathlib.Path(temporary))
            upload.fresh_editor(
                fixture.ledger, session_id="fresh",
                opened_at_utc="2026-08-31T13:00:00Z",
            )
            copy = fixture.root / "copy.cpp"
            copy.write_bytes(fixture.source.read_bytes())
            upload.attest_copyback(
                fixture.ledger, generated_source=fixture.source,
                copied_back_source=copy,
                created_at_utc="2026-08-31T13:01:00Z",
            )
            upload.record_play(
                fixture.ledger, legal_stdout=False, expected_telemetry=True,
                created_at_utc="2026-08-31T13:02:00Z",
            )
            with self.assertRaisesRegex(upload.UploadError, "cannot be retried"):
                upload.record_play(
                    fixture.ledger, legal_stdout=True,
                    expected_telemetry=True,
                    created_at_utc="2026-08-31T13:03:00Z",
                )
            with self.assertRaisesRegex(upload.UploadError, "forbids Submit"):
                upload.start_submit(
                    fixture.ledger, started_at_utc="2026-08-31T13:04:00Z"
                )

    def test_ambiguous_submit_requires_unique_bound_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EventFixture(pathlib.Path(temporary))
            fixture.ready()
            upload.start_submit(
                fixture.ledger, started_at_utc="2026-08-31T13:03:00Z"
            )
            upload.record_ambiguous(
                fixture.ledger, observed_at_utc="2026-08-31T13:04:00Z",
                evidence={"network": "uncertain"},
            )
            with self.assertRaises(Exception):
                upload.attest_submission(
                    fixture.ledger, agent_id=701, submission_id=801,
                    submitted_at_utc="2026-08-31T13:05:00Z",
                )
            result = upload.attest_submission(
                fixture.ledger, agent_id=701, submission_id=801,
                submitted_at_utc="2026-08-31T13:05:00Z",
                ambiguity_resolution={"matching_submissions": 1,
                                      "agent_id": 701, "submission_id": 801},
            )
            self.assertEqual(result["status"], "submission-attested")


class CompletionTest(unittest.TestCase):
    def test_completion_accepts_direct_rank4_teacher_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            ledger = root / "direct-upload"
            ledger.mkdir()
            candidate = root / "submission.cpp"
            candidate.write_text("int main(){}\n", encoding="ascii")
            authorization_path = ledger / "one-upload-authorization.json"
            q.write_sealed(authorization_path, {
                "schema": q.UPLOAD_AUTH_SCHEMA,
                "namespace": q.NAMESPACE,
                "uploads_authorized": 1,
                "candidate_commit": COMMIT,
                "candidate": {
                    "path": str(candidate.resolve()),
                    "sha256": q.sha256_file(candidate),
                    "bytes": candidate.stat().st_size,
                    "ascii": True,
                },
            })
            authorization = q.load_sealed(
                authorization_path, q.UPLOAD_AUTH_SCHEMA
            )
            ci_path = root / "ci.json"
            ci_path.write_text("{}\n", encoding="ascii")
            dual_path = root / "dual.json"
            dual_path.write_text("{}\n", encoding="ascii")
            dual_reference = {
                "path": str(dual_path.resolve()),
                "sha256": q.sha256_file(dual_path),
            }
            inputs = {
                "schema": upload.RANK4_TEACHER_UPLOAD_INPUT_SCHEMA,
                "ci": {
                    "path": str(ci_path.resolve()),
                    "sha256": q.sha256_file(ci_path),
                },
                "dual_qualification": dual_reference,
            }
            q.write_sealed(ledger / "authorization-inputs.json", inputs)
            attestation_path = ledger / "upload/05-submission-attested.json"
            q.write_sealed(attestation_path, {
                "schema": q.UPLOAD_EVENT_SCHEMA,
                "namespace": q.NAMESPACE,
                "status": "submission-attested",
                "submit_clicks": 1,
                "authorization": q.artifact_reference(
                    authorization_path, q.UPLOAD_AUTH_SCHEMA
                ),
                "candidate_commit": COMMIT,
                "source_sha256": q.sha256_file(candidate),
                "source_bytes": candidate.stat().st_size,
                "agent_id": 701,
                "submission_id": 801,
                "submitted_at_utc": "2026-08-31T13:00:00Z",
                "ambiguity_resolution": None,
            })
            receipt_path, _receipt = upload.live_tools.write_content_addressed(
                root / "window-receipts", {
                    "schema": upload.live_tools.WINDOW_RECEIPT_SCHEMA,
                    "submission_attestation":
                        upload.live_tools.artifact_reference(
                            attestation_path, q.UPLOAD_EVENT_SCHEMA
                        ),
                },
            )
            live_ref = root / "live.reference.json"
            live_ref.write_text("reference", encoding="ascii")
            ci = {
                "run_id": 12345,
                "fetched_at_utc": "2026-08-31T12:30:00Z",
            }

            with mock.patch.object(
                upload, "_authorization",
                return_value=(authorization_path, authorization, inputs),
            ), mock.patch.object(
                upload, "validate_qualified_chain"
            ) as legacy_chain, mock.patch.object(
                upload, "validate_ci_evidence", return_value=ci
            ) as validate_ci:
                result = upload.verify_completion(
                    ledger,
                    live_reference_path=live_ref,
                    live_data_root=root,
                    live_verifier=lambda path, data_root: {
                        "status": "complete-accepted-diagnostic",
                        "exact_games": 90,
                        "training_eligible": False,
                        "rollback_authorized": False,
                        "second_upload_authorized": False,
                        "receipt": upload.live_tools.artifact_reference(
                            receipt_path,
                            upload.live_tools.WINDOW_RECEIPT_SCHEMA,
                        ),
                    },
                    verified_at_utc="2026-08-31T14:00:00Z",
                )

            legacy_chain.assert_not_called()
            validate_ci.assert_called_once_with(
                ci_path.resolve(), expected_head=COMMIT
            )
            self.assertEqual(result["strict_final"], dual_reference)

    def test_completion_accepts_clean_or_own_failure_diagnostic_only(self):
        for status in (
            "complete-accepted-diagnostic",
            "complete-rejected-focus-operational-failure",
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                fixture = FinalFixture(pathlib.Path(temporary))
                fixture.authorize()
                directory = upload.authorization_directory(fixture.ledger)
                attestation_path = directory / "upload/05-submission-attested.json"
                q.write_sealed(attestation_path, {
                    "schema": q.UPLOAD_EVENT_SCHEMA, "namespace": q.NAMESPACE,
                    "status": "submission-attested", "submit_clicks": 1,
                    "authorization": q.artifact_reference(
                        directory / "one-upload-authorization.json",
                        q.UPLOAD_AUTH_SCHEMA,
                    ), "candidate_commit": COMMIT,
                    "source_sha256": q.sha256_file(fixture.candidate),
                    "source_bytes": fixture.candidate.stat().st_size,
                    "agent_id": 701, "submission_id": 801,
                    "submitted_at_utc": "2026-08-31T13:00:00Z",
                    "ambiguity_resolution": None,
                })
                receipt_path, _receipt = upload.live_tools.write_content_addressed(
                    fixture.root / "window-receipts", {
                        "schema": upload.live_tools.WINDOW_RECEIPT_SCHEMA,
                        "submission_attestation":
                            upload.live_tools.artifact_reference(
                                attestation_path, q.UPLOAD_EVENT_SCHEMA
                            ),
                    },
                )
                live_ref = fixture.root / "live.reference.json"
                live_ref.write_text("reference")
                with mock.patch.object(
                    upload.preflight_tools,
                    "validate_preflight_receipt",
                    return_value={},
                ):
                    result = upload.verify_completion(
                        fixture.ledger, live_reference_path=live_ref,
                        live_data_root=fixture.root,
                        live_verifier=lambda path, data_root: {
                            "status": status, "exact_games": 90,
                            "training_eligible": False,
                            "rollback_authorized": False,
                            "second_upload_authorized": False,
                            "receipt": upload.live_tools.artifact_reference(
                                receipt_path,
                                upload.live_tools.WINDOW_RECEIPT_SCHEMA,
                            ),
                        },
                        verified_at_utc="2026-08-31T14:00:00Z",
                    )
                self.assertEqual(result["status"], "complete")

    def test_incomplete_live_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalFixture(pathlib.Path(temporary))
            fixture.authorize()
            directory = upload.authorization_directory(fixture.ledger)
            q.write_sealed(directory / "upload/05-submission-attested.json", {
                "schema": q.UPLOAD_EVENT_SCHEMA, "namespace": q.NAMESPACE,
                "status": "submission-attested", "submit_clicks": 1,
                "authorization": q.artifact_reference(
                    directory / "one-upload-authorization.json", q.UPLOAD_AUTH_SCHEMA
                ), "candidate_commit": COMMIT,
                "source_sha256": q.sha256_file(fixture.candidate),
                "source_bytes": fixture.candidate.stat().st_size,
                "agent_id": 701, "submission_id": 801,
                "submitted_at_utc": "2026-08-31T13:00:00Z",
                "ambiguity_resolution": None,
            })
            live_ref = fixture.root / "live.reference.json"
            live_ref.write_text("reference")
            with mock.patch.object(
                upload.preflight_tools,
                "validate_preflight_receipt",
                return_value={},
            ), self.assertRaisesRegex(upload.UploadError, "not a complete"):
                upload.verify_completion(
                    fixture.ledger, live_reference_path=live_ref,
                    live_data_root=fixture.root,
                    live_verifier=lambda path, data_root: {
                        "status": "waiting", "exact_games": 89,
                        "training_eligible": False,
                        "rollback_authorized": False,
                        "second_upload_authorized": False,
                    }, verified_at_utc="2026-08-31T14:00:00Z",
                )


if __name__ == "__main__":
    unittest.main()
