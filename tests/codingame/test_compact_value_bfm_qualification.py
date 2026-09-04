import copy
import hashlib
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_qualification.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_qualification", TOOL)
q = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(q)


COMMIT = "a" * 40
RANK4 = ROOT / "submissions" / "codingame" / "bots" / "rank_4" / "submission.cpp"


def base_summary():
    return {
        "games": 1000,
        "candidate_wins": 527,
        "candidate_color_wins": {"0": 260, "1": 267},
        "failures": {name: 0 for name in q.FAILURE_CATEGORIES},
        "maximum_turns": 320,
        "timing": {"first_max_ms": 999.0, "later_max_ms": 199.0},
        "uncontended_timing": {
            "first_max_ms": 899.0, "later_max_ms": 179.0,
        },
    }


def base8_12(value):
    digits = []
    for _ in range(12):
        digits.append(str(value & 7))
        value >>= 3
    return "".join(reversed(digits))


def openings():
    result = []
    for index in range(500):
        fingerprints = sorted({
            hashlib.sha256(f"opening-{index}-symmetry-{symmetry}".encode()).hexdigest()
            for symmetry in range(4)
        })
        flat = base8_12(index)
        result.append({
            "opening_id": f"final-{index:03d}",
            "transcript": f"{flat[0]}/{flat[1:]}",
            "primitive_plies": len(flat),
            "fingerprint": fingerprints[0],
            "symmetry_fingerprints": fingerprints,
        })
    return result


def battles(count, *, agent=701, submission=801, done=True):
    return [
        {
            "gameId": 10_000 + index,
            "done": done,
            "players": [
                {"playerAgentId": agent, "submissionId": submission, "position": 0},
                {"playerAgentId": 20_000 + index, "submissionId": 30_000 + index,
                 "position": 1},
            ],
        }
        for index in range(count)
    ]


class Fixture:
    def __init__(self, root):
        self.root = root
        self.candidate = root / "candidate.cpp"
        self.candidate.write_text("// compact candidate\nint main(){return 0;}\n", encoding="ascii")
        self.source_binding = root / "source-binding.json"
        q.create_source_binding(
            self.source_binding,
            candidate_source=self.candidate,
            candidate_commit=COMMIT,
            rank4_source=RANK4,
            opponent_source=RANK4,
        )
        self.bank = root / "protected-bank.json"
        q.create_final_bank(
            self.bank,
            source_binding_path=self.source_binding,
            openings=openings(),
            seed_factory=lambda size: b"z" * size,
        )
        self.harness = root / "harness.py"
        self.harness.write_text("# harness\n", encoding="ascii")
        self.binding = root / "gate-binding.json"
        q.create_gate_binding(
            self.binding,
            source_binding_path=self.source_binding,
            bank_path=self.bank,
            harness_path=self.harness,
        )

    def aggregate(self, path=None):
        path = path or self.root / "aggregate-pass.json"
        return q.write_sealed(path, {
            "schema": q.FINAL_AGGREGATE_SCHEMA,
            "namespace": q.NAMESPACE,
            "binding": q.artifact_reference(self.binding, q.GATE_BINDING_SCHEMA),
            "completed_at_utc": "2026-08-31T12:00:00Z",
            "summary": base_summary(),
            "verdict": q.strict_gate_verdict(base_summary()),
            "status": "rank4-qualified",
        })

    def authorization(self):
        aggregate = self.root / "aggregate-pass.json"
        self.aggregate(aggregate)
        path = self.root / "one-upload-authorization.json"
        q.create_upload_authorization(
            path,
            binding_path=self.binding,
            aggregate_path=aggregate,
            ci_record={
                "run_id": 123,
                "head_sha": COMMIT,
                "conclusion": "success",
                "workflow": "CI and Pages",
                "workflow_file": "pages.yml",
                "event": "workflow_dispatch",
                "head_branch": "compact-value-bfm",
                "head_ref": "refs/heads/compact-value-bfm",
                "url": "https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/runs/123",
                "jobs": {
                    "replay-training-contract": "success",
                    "leaderboard-contract": "success",
                    "test-gcc": "success",
                    "test-clang": "success",
                    "test-sanitizers": "success",
                },
            },
        )
        return path


class StrictGateBoundaryTest(unittest.TestCase):
    def test_exact_527_with_260_and_267_passes(self):
        self.assertTrue(q.strict_gate_verdict(base_summary())["passed"])

    def test_526_fails(self):
        summary = base_summary()
        summary["candidate_wins"] = 526
        self.assertIn("candidate_wins", q.strict_gate_verdict(summary)["errors"])

    def test_527_with_259_in_one_color_fails(self):
        summary = base_summary()
        summary["candidate_color_wins"] = {"0": 259, "1": 268}
        verdict = q.strict_gate_verdict(summary)
        self.assertFalse(verdict["passed"])
        self.assertIn("candidate_color_0", verdict["errors"])

    def test_limits_are_strict_and_failures_are_zero_only(self):
        for section, key, value in (
            ("timing", "first_max_ms", 1000.0),
            ("timing", "later_max_ms", 200.0),
            ("uncontended_timing", "first_max_ms", 900.0),
            ("uncontended_timing", "later_max_ms", 180.0),
        ):
            with self.subTest(section=section, key=key):
                summary = base_summary()
                summary[section][key] = value
                self.assertFalse(q.strict_gate_verdict(summary)["passed"])
        summary = base_summary()
        summary["failures"]["timeout"] = 1
        self.assertFalse(q.strict_gate_verdict(summary)["passed"])
        summary = base_summary()
        summary["maximum_turns"] = 321
        self.assertFalse(q.strict_gate_verdict(summary)["passed"])


class BindingAndBankTest(unittest.TestCase):
    def test_exact_rank4_candidate_bank_and_opponent_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            binding = q.load_sealed(fixture.binding, q.GATE_BINDING_SCHEMA)
            self.assertEqual(binding["rank4"]["sha256"], q.RANK4_SHA256)
            self.assertEqual(binding["opponent"]["sha256"], q.RANK4_SHA256)
            bank = q.load_sealed(fixture.bank, q.FINAL_BANK_SCHEMA)
            self.assertEqual(bank["seed_256_hex"], (b"z" * 32).hex())
            self.assertEqual(len(bank["openings"]), 500)
            self.assertEqual(binding["bank"]["sha256"], q.sha256_file(fixture.bank))

    def test_wrong_opponent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate = root / "candidate.cpp"
            candidate.write_text("int main(){}\n")
            wrong = root / "wrong.cpp"
            wrong.write_text("int main(){return 1;}\n")
            with self.assertRaisesRegex(q.QualificationError, "opponent"):
                q.create_source_binding(
                    root / "binding.json",
                    candidate_source=candidate,
                    candidate_commit=COMMIT,
                    rank4_source=RANK4,
                    opponent_source=wrong,
                )

    def test_excluded_or_repeated_symmetry_is_rejected_before_entropy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate = root / "candidate.cpp"
            candidate.write_text("int main(){}\n")
            source = root / "source.json"
            q.create_source_binding(
                source, candidate_source=candidate, candidate_commit=COMMIT,
                rank4_source=RANK4, opponent_source=RANK4,
            )
            rows = openings()
            called = []
            with self.assertRaisesRegex(q.QualificationError, "excluded symmetry"):
                q.create_final_bank(
                    root / "bank.json", source_binding_path=source,
                    openings=rows,
                    excluded_fingerprints=[rows[0]["symmetry_fingerprints"][1]],
                    seed_factory=lambda size: called.append(size) or b"x" * size,
                )
            self.assertEqual(called, [])
            rows = openings()
            rows[1]["symmetry_fingerprints"] = sorted(set(
                rows[1]["symmetry_fingerprints"] +
                [rows[0]["symmetry_fingerprints"][0]]
            ))
            with self.assertRaisesRegex(q.QualificationError, "overlap by symmetry"):
                q.create_final_bank(
                    root / "bank2.json", source_binding_path=source,
                    openings=rows, seed_factory=lambda size: b"x" * size,
                )


class FinalShardLedgerTest(unittest.TestCase):
    @staticmethod
    def shard_games(index):
        games = []
        for pair in range(index * 5, index * 5 + 5):
            for color in (0, 1):
                games.append({
                    "pair_index": pair,
                    "candidate_color": color,
                    "candidate_win": pair < (260 if color == 0 else 267),
                    "turns": 100,
                    "failure": None,
                    "first_ms": 800.0,
                    "later_max_ms": 155.0,
                })
        return games

    @staticmethod
    def evidence(root, index):
        path = root / "evidence" / f"shard-{index:03d}.json"
        q.write_sealed(path, {
            "schema": "fixture.raw-gate-evidence.v1",
            "shard_index": index,
        })
        return q.artifact_reference(path)

    def test_spent_claim_without_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            final = fixture.root / "final"
            first = q.start_final_shard(
                final, binding_path=fixture.binding, index=0,
                started_at_utc="2026-08-31T12:00:00Z",
            )
            self.assertEqual(first["status"], "started")
            with self.assertRaises(q.SpentShardError):
                q.start_final_shard(
                    final, binding_path=fixture.binding, index=0,
                    started_at_utc="2026-08-31T12:01:00Z",
                )

    def test_completed_receipt_is_reused_not_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            final = fixture.root / "final"
            q.start_final_shard(
                final, binding_path=fixture.binding, index=7,
                started_at_utc="2026-08-31T12:00:00Z",
            )
            q.record_shard_receipt(
                final, binding_path=fixture.binding, index=7,
                games=self.shard_games(7), completed_at_utc="2026-08-31T12:01:00Z",
                evidence=self.evidence(final, 7),
            )
            resumed = q.start_final_shard(
                final, binding_path=fixture.binding, index=7,
                started_at_utc="2026-08-31T12:02:00Z",
            )
            self.assertEqual(resumed["status"], "complete-reused")

    def test_all_100_five_pair_receipts_aggregate_to_exact_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            final = fixture.root / "final"
            for index in range(100):
                q.start_final_shard(
                    final, binding_path=fixture.binding, index=index,
                    started_at_utc=f"2026-08-31T12:{index // 60:02d}:{index % 60:02d}Z",
                )
                q.record_shard_receipt(
                    final, binding_path=fixture.binding, index=index,
                    games=self.shard_games(index),
                    completed_at_utc=f"2026-08-31T13:{index // 60:02d}:{index % 60:02d}Z",
                    evidence=self.evidence(final, index),
                )
            aggregate = q.aggregate_final(
                final, binding_path=fixture.binding,
                uncontended_timing={"first_max_ms": 899.0, "later_max_ms": 179.0},
                completed_at_utc="2026-08-31T14:00:00Z",
            )
            self.assertEqual(aggregate["summary"]["games"], 1000)
            self.assertEqual(aggregate["summary"]["candidate_wins"], 527)
            self.assertEqual(
                aggregate["summary"]["candidate_color_wins"], {"0": 260, "1": 267}
            )
            self.assertEqual(aggregate["status"], "rank4-qualified")


class UploadLedgerTest(unittest.TestCase):
    def make_ready(self, fixture):
        authorization = fixture.authorization()
        q.prepare_upload(
            fixture.root, authorization_path=authorization,
            created_at_utc="2026-08-31T15:00:00Z",
            fresh_editor=True,
        )
        copyback = fixture.root / "copyback.cpp"
        copyback.write_bytes(fixture.candidate.read_bytes())
        q.attest_editor_copyback(
            fixture.root, authorization_path=authorization,
            generated_source=fixture.candidate, copied_back_source=copyback,
            created_at_utc="2026-08-31T15:01:00Z",
        )
        q.record_play(
            fixture.root, authorization_path=authorization,
            legal_stdout=True, expected_telemetry=True,
            created_at_utc="2026-08-31T15:02:00Z",
        )
        return authorization

    def test_copyback_play_and_single_submit_happy_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            authorization = self.make_ready(fixture)
            q.start_submit(
                fixture.root, authorization_path=authorization,
                started_at_utc="2026-08-31T15:03:00Z",
            )
            final = q.attest_submission(
                fixture.root, authorization_path=authorization,
                agent_id=7001, submission_id=8001,
                submitted_at_utc="2026-08-31T15:04:00Z",
            )
            self.assertEqual(final["submit_clicks"], 1)
            self.assertEqual(final["source_sha256"], q.sha256_file(fixture.candidate))
            with self.assertRaisesRegex(q.QualificationError, "never click again"):
                q.start_submit(
                    fixture.root, authorization_path=authorization,
                    started_at_utc="2026-08-31T15:05:00Z",
                )

    def test_ambiguous_submit_requires_unique_history_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            authorization = self.make_ready(fixture)
            q.start_submit(
                fixture.root, authorization_path=authorization,
                started_at_utc="2026-08-31T15:03:00Z",
            )
            q.record_submit_ambiguous(
                fixture.root, authorization_path=authorization,
                observed_at_utc="2026-08-31T15:04:00Z",
                evidence={"network": "uncertain"},
            )
            with self.assertRaisesRegex(q.QualificationError, "history/API"):
                q.attest_submission(
                    fixture.root, authorization_path=authorization,
                    agent_id=7002, submission_id=8002,
                    submitted_at_utc="2026-08-31T15:05:00Z",
                )
            with self.assertRaisesRegex(q.QualificationError, "not unique"):
                q.attest_submission(
                    fixture.root, authorization_path=authorization,
                    agent_id=7002, submission_id=8002,
                    submitted_at_utc="2026-08-31T15:05:00Z",
                    ambiguity_resolution={
                        "matching_submissions": 2,
                        "agent_id": 7002, "submission_id": 8002,
                    },
                )
            final = q.attest_submission(
                fixture.root, authorization_path=authorization,
                agent_id=7002, submission_id=8002,
                submitted_at_utc="2026-08-31T15:05:00Z",
                ambiguity_resolution={
                    "matching_submissions": 1,
                    "agent_id": 7002, "submission_id": 8002,
                    "history_checked": True,
                },
            )
            self.assertEqual(final["status"], "submission-attested")

    def test_failed_play_blocks_submit_and_copyback_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            authorization = fixture.authorization()
            q.prepare_upload(
                fixture.root, authorization_path=authorization,
                created_at_utc="2026-08-31T15:00:00Z",
                fresh_editor=True,
            )
            bad = fixture.root / "bad.cpp"
            bad.write_text("different\n")
            with self.assertRaisesRegex(q.QualificationError, "differs"):
                q.attest_editor_copyback(
                    fixture.root, authorization_path=authorization,
                    generated_source=fixture.candidate, copied_back_source=bad,
                    created_at_utc="2026-08-31T15:01:00Z",
                )
            good = fixture.root / "good.cpp"
            good.write_bytes(fixture.candidate.read_bytes())
            q.attest_editor_copyback(
                fixture.root, authorization_path=authorization,
                generated_source=fixture.candidate, copied_back_source=good,
                created_at_utc="2026-08-31T15:01:00Z",
            )
            q.record_play(
                fixture.root, authorization_path=authorization,
                legal_stdout=False, expected_telemetry=True,
                created_at_utc="2026-08-31T15:02:00Z",
            )
            with self.assertRaises(q.QualificationError):
                q.start_submit(
                    fixture.root, authorization_path=authorization,
                    started_at_utc="2026-08-31T15:03:00Z",
                )


class ExactLiveWindowTest(unittest.TestCase):
    def test_89_waits_90_is_ready_and_91_is_rejected(self):
        report = q.classify_matching_window(battles(89), agent_id=701, submission_id=801)
        self.assertEqual(report["status"], "waiting")
        self.assertEqual(report["complete_games"], 89)
        report = q.classify_matching_window(battles(90), agent_id=701, submission_id=801)
        self.assertTrue(report["collector_permitted"])
        with self.assertRaisesRegex(q.QualificationError, "exceeds exactly 90"):
            q.classify_matching_window(battles(91), agent_id=701, submission_id=801)

    def test_other_submission_is_ignored_and_detail_fields_fail_closed(self):
        rows = battles(90)
        older = battles(5, submission=800)
        for index, row in enumerate(older):
            row["gameId"] = 50_000 + index
        rows.extend(older)
        self.assertTrue(q.classify_matching_window(
            rows, agent_id=701, submission_id=801
        )["collector_permitted"])

        class Exploding(dict):
            def items(self):
                raise AssertionError("forbidden value was consumed")

        row = battles(1)[0]
        row["frames"] = Exploding()
        with self.assertRaisesRegex(q.QualificationError, "forbidden"):
            q.classify_matching_window([row], agent_id=701, submission_id=801)

    def test_complete_window_binds_source_identity_and_own_failure_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            authorization = UploadLedgerTest().make_ready(fixture)
            q.start_submit(
                fixture.root, authorization_path=authorization,
                started_at_utc="2026-08-31T15:03:00Z",
            )
            attestation = fixture.root / "upload" / "05-submission-attested.json"
            q.attest_submission(
                fixture.root, authorization_path=authorization,
                agent_id=701, submission_id=801,
                submitted_at_utc="2026-08-31T15:04:00Z",
            )
            ids = [10_000 + index for index in range(90)]
            receipt = q.finalize_live_window(
                fixture.root / "live-window.json",
                battles=battles(90),
                submission_attestation_path=attestation,
                collector_manifest={
                    "agent_id": 701,
                    "submission_id": 801,
                    "source_sha256": q.sha256_file(fixture.candidate),
                    "repository_commit": COMMIT,
                    "game_ids": ids,
                    "focus_operational_failures": 1,
                    "opponent_operational_failures": 7,
                },
            )
            self.assertEqual(receipt["status"], "complete-rejected-own-failure")
            self.assertFalse(receipt["opponent_failures_count_as_strength_wins"])
            self.assertFalse(receipt["training_eligible"])


if __name__ == "__main__":
    unittest.main()
