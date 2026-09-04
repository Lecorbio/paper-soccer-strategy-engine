import hashlib
import importlib.util
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_final.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_final", TOOL)
final = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(final)
q = final.qualification
campaign = final.campaign
opening_tools = final.opening_tools
preflight = final.preflight_tools

RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
COMMIT = "d" * 40
EXAMPLE = "5/2/2/0/1/4/1/17/6/0/75"


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def write_tsv(path, index):
    path.write_text(
        "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
        "# rules=8x10;own-goals-allowed;mover-loses\n"
        "# classification=development\n# seed=1\n"
        "# minimum-physical-plies=12\n"
        "opening_id\ttranscript\tstate_identity\n"
        f"copied-{index}\t{EXAMPLE}\topaque:{index}\n",
        encoding="ascii",
    )
    return path


def content_addressed(root, payload):
    value = q.seal(payload)
    raw = q.canonical_json_bytes(value)
    path = root / f"{q.sha256_bytes(raw)}.json"
    path.write_bytes(raw)
    return path


class FreezeFixture:
    def __init__(self, root):
        self.root = root
        self.candidate = root / "candidate.cpp"
        self.candidate.write_text("int main(){return 0;}\n", encoding="ascii")
        self.runtime = root / "selected.runtime.json"
        self.runtime.write_text('{"selected":true}\n', encoding="ascii")
        candidate = {
            "path": str(self.candidate.resolve()),
            "bytes": self.candidate.stat().st_size,
            "sha256": q.sha256_file(self.candidate),
            "ascii": True,
            "bootstrap_zero": False,
        }
        rank4 = {
            "path": str(RANK4.resolve()), "bytes": q.RANK4_BYTES,
            "sha256": q.RANK4_SHA256, "ascii": True,
        }
        runtime_sha = q.sha256_file(self.runtime)
        self.selection = root / "selection.json"
        q.write_sealed(self.selection, {
            "schema": campaign.SELECTION_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "immutable-development-selected-not-tests-opened",
            "selection_immutable": True,
            "protected_tests_opened": False,
            "model": {"candidate_id": "selected", "architecture": "6301-8-8-1",
                      "target": "search-target", "artifact_sha256": runtime_sha,
                      "source_bytes": candidate["bytes"]},
            "tuple": ["0.95", "0.5", "1"],
            "profile": "default",
            "profile_work": campaign.PROFILE_ROSTER["default"],
            "actual_clock": {"candidate_id": "selected", "wins": 211},
        })
        self.test_auth = root / "test-auth.json"
        q.write_sealed(self.test_auth, {
            "schema": campaign.TEST_AUTH_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "protected-tests-authorized-once",
            "selection": q.artifact_reference(
                self.selection, campaign.SELECTION_SCHEMA
            ),
            "artifact": {"path": str(self.runtime.resolve()),
                         "bytes": self.runtime.stat().st_size,
                         "sha256": runtime_sha},
            "selection_may_change": False,
            "tests_may_only_diagnose": True,
        })
        timing = []
        for count in (1, 2, 10):
            for color in (0, 1):
                for replica in range(count):
                    timing.append({"process_count": count, "color": color,
                                   "replica": replica, "first_ms": 800.0,
                                   "later_max_ms": 155.0})
        inputs = {
            "candidate_commit": COMMIT,
            "candidate": candidate,
            "rank4": rank4,
            "runtime": {"path": str(self.runtime.resolve()),
                        "bytes": self.runtime.stat().st_size,
                        "sha256": runtime_sha, "ascii": True},
        }
        self.gate = root / "preflight-rank4-gate"
        self.gate.write_text("source-specific-gate", encoding="ascii")
        self.gate.chmod(0o700)
        gate_record = {
            "path": str(self.gate.resolve()),
            "bytes": self.gate.stat().st_size,
            "sha256": q.sha256_file(self.gate),
            "executable": True,
        }
        self.preflight = content_addressed(root, {
            "schema": preflight.RECEIPT_SCHEMA,
            "namespace": q.NAMESPACE,
            "status": "passed",
            "claim": {},
            "plan": {},
            "inputs_before": inputs,
            "inputs_after": inputs,
            "checks": {"all-local-preflight": "passed"},
            "timing": {"schema": preflight.TIMING_SCHEMA,
                       "samples": timing,
                       "first_limit_exclusive_ms": 900.0,
                       "later_limit_exclusive_ms": 180.0},
            "panels": {
                "clang-release": {
                    "binaries": {
                        "papersoccer_codingame_compact_value_bfm_rank4_gate":
                            gate_record,
                    }
                }
            },
            "protected_banks_accessed": [], "git_writes": 0, "uploads": 0,
        })

    def freeze(self, git=None):
        with mock.patch.object(
            final.preflight_tools,
            "validate_preflight_receipt",
            return_value={},
        ):
            return final.freeze_candidate(
                self.root / "freeze", repository=self.root,
                selection_path=self.selection,
                protected_test_authorization_path=self.test_auth,
                preflight_receipt_path=self.preflight,
                candidate_source=self.candidate, rank4_source=RANK4,
                runtime_path=self.runtime, frozen_at_utc="2026-08-31T22:00:00Z",
                git_verifier=git or (
                    lambda repository, source, commit: {
                        "commit": commit, "tracked_clean": True,
                        "source_path": "candidate.cpp", "committed_bytes_equal": True,
                    }
                ),
            )


class PreparedFixture:
    def __init__(self, root):
        self.root = root
        freeze = FreezeFixture(root)
        self.freeze = freeze.freeze()
        self.freeze_path = root / "freeze/freeze.json"
        self.copied = [write_tsv(root / f"bank-{index:03d}.tsv", index)
                       for index in range(7)]
        generated = opening_tools.generate_development_banks(
            root / "development", exclusion_paths=self.copied
        )
        self.development = [generated[stage]
                            for stage in opening_tools.DEVELOPMENT_ORDER]
        self.gate = freeze.gate
        self.plan = final.prepare_protected_final(
            root / "protected", freeze_path=self.freeze_path,
            copied_exclusion_paths=self.copied,
            development_bank_paths=self.development,
            rank4_gate_executable=self.gate,
            created_at_utc="2026-08-31T22:01:00Z",
            entropy=lambda count: b"f" * count,
        )
        self.plan_path = root / "protected/final-plan.json"


class ShardFixture:
    def __init__(self, root):
        self.root = root
        self.candidate = root / "candidate.cpp"
        self.candidate.write_text("int main(){return 0;}\n", encoding="ascii")
        self.source = root / "source.json"
        q.create_source_binding(
            self.source, candidate_source=self.candidate,
            candidate_commit=COMMIT, rank4_source=RANK4,
            opponent_source=RANK4,
        )
        self.adapter = root / "adapter.json"
        source = q.load_sealed(self.source, q.SOURCE_BINDING_SCHEMA)
        q.write_sealed(self.adapter, {
            "schema": q.FINAL_BANK_SCHEMA, "namespace": q.NAMESPACE,
            "source_binding": q.artifact_reference(
                self.source, q.SOURCE_BINDING_SCHEMA
            ),
            "candidate_commit": COMMIT,
            "candidate_sha256": source["candidate"]["sha256"],
            "rank4_sha256": q.RANK4_SHA256, "opening_count": 500,
        })
        self.gate = root / "gate"
        self.gate.write_text("gate")
        self.gate.chmod(0o700)
        self.binding = root / "binding.json"
        q.create_gate_binding(
            self.binding, source_binding_path=self.source,
            bank_path=self.adapter, harness_path=self.gate,
        )
        self.gate_bank = root / "bank.tsv"
        self.gate_bank.write_text("opening_id\ttranscript\n")
        self.selection = root / "selection.json"
        q.write_sealed(self.selection, {
            "schema": campaign.SELECTION_SCHEMA, "namespace": q.NAMESPACE,
            "tuple": ["0.95", "0.5", "1"],
            "profile_work": campaign.PROFILE_ROSTER["default"],
        })
        self.plan_path = root / "plan.json"
        self.plan = q.write_sealed(self.plan_path, {
            "schema": final.PLAN_SCHEMA, "namespace": q.NAMESPACE,
            "candidate_commit": COMMIT, "candidate": source["candidate"],
            "rank4": source["rank4"],
            "selection": q.artifact_reference(
                self.selection, campaign.SELECTION_SCHEMA
            ),
            "preflight": {"path": "/preflight", "sha256": digest("preflight")},
            "protected_bank": {"path": "/protected", "sha256": digest("protected")},
            "gate_bank": {"path": str(self.gate_bank.resolve()),
                          "sha256": q.sha256_file(self.gate_bank)},
            "gate_binding": q.artifact_reference(
                self.binding, q.GATE_BINDING_SCHEMA
            ),
            "rank4_gate": {"path": str(self.gate.resolve()),
                           "bytes": self.gate.stat().st_size,
                           "sha256": q.sha256_file(self.gate)},
            "uncontended_timing": {"first_max_ms": 800.0,
                                   "later_max_ms": 155.0},
        })

    @staticmethod
    def games(index, wins=527):
        result = []
        for pair in range(index * 5, index * 5 + 5):
            for color in (0, 1):
                if wins == 527:
                    won = pair < (260 if color == 0 else 267)
                else:
                    won = pair < (260 if color == 0 else 266)
                result.append({
                    "pair_index": pair, "candidate_color": color,
                    "candidate_win": won, "turns": 100, "failure": None,
                    "first_ms": 800.0, "later_max_ms": 155.0,
                })
        return result


class FreezeAndPrepareTest(unittest.TestCase):
    def test_freeze_binds_selection_preflight_source_rank4_and_clean_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FreezeFixture(pathlib.Path(temporary))
            frozen = fixture.freeze()
            self.assertEqual(frozen["status"],
                             "candidate-source-clean-commit-frozen")
            self.assertEqual(frozen["candidate_commit"], COMMIT)
            self.assertEqual(frozen["rank4"]["sha256"], q.RANK4_SHA256)
            self.assertTrue(frozen["git"]["tracked_clean"])

    def test_dirty_git_or_runtime_mismatch_rejects_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FreezeFixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(final.FinalError, "clean Git"):
                fixture.freeze(git=lambda repository, source, commit: {
                    "commit": commit, "tracked_clean": False
                })
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FreezeFixture(pathlib.Path(temporary))
            fixture.runtime.write_text("changed\n")
            with self.assertRaisesRegex(final.FinalError, "runtime identity"):
                fixture.freeze()

    def test_prepare_validates_13_exclusions_and_remains_unconsumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PreparedFixture(pathlib.Path(temporary))
            plan = fixture.plan
            self.assertEqual(len(plan["exclusions"]), 13)
            self.assertEqual(plan["status"], "protected-final-ready-unconsumed")
            bank = opening_tools.validate_bank(
                pathlib.Path(plan["protected_bank"]["path"])
            )
            self.assertEqual(bank["opening_count"], 500)
            self.assertEqual(
                final.gate_support.validate_bank(
                    pathlib.Path(plan["gate_bank"]["path"])
                )["openings"].__len__(), 500
            )
            self.assertFalse((fixture.root / "ledger/bank-consumed-at-launch.json").exists())


class ConsumptionAndShardTest(unittest.TestCase):
    def test_consumption_marker_is_atomic_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            ledger = fixture.root / "ledger"
            first = final.consume_bank_at_launch(
                ledger, plan_path=fixture.plan_path,
                launched_at_utc="2026-08-31T23:00:00Z",
            )
            second = final.consume_bank_at_launch(
                ledger, plan_path=fixture.plan_path,
                launched_at_utc="2026-08-31T23:01:00Z",
            )
            self.assertEqual(first["body_sha256"], second["body_sha256"])
            body = {key: value for key, value in fixture.plan.items()
                    if key != "body_sha256"}
            body["gate_bank"] = {"path": "/other", "sha256": digest("other")}
            other = fixture.root / "other-plan.json"
            q.write_sealed(other, body)
            with self.assertRaisesRegex(final.FinalError, "another identity"):
                final.consume_bank_at_launch(
                    ledger, plan_path=other,
                    launched_at_utc="2026-08-31T23:02:00Z",
                )

    def test_exact_100_shards_run_on_at_most_four_workers_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            ledger = fixture.root / "ledger"
            lock = threading.Lock()
            active = maximum = calls = 0

            def runner(spec):
                nonlocal active, maximum, calls
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    calls += 1
                time.sleep(0.001)
                with lock:
                    active -= 1
                return spec["index"]

            aggregate = final.run_final_shards(
                ledger, plan_path=fixture.plan_path, repository=fixture.root,
                maximum_workers=4, runner=runner,
                adapter=lambda raw, plan, index: fixture.games(index),
                clock=lambda: "2026-08-31T23:00:00Z",
            )
            self.assertEqual(calls, 100)
            self.assertLessEqual(maximum, 4)
            self.assertEqual(aggregate["status"], "rank4-qualified")
            self.assertTrue((ledger / "rank4-qualified-inputs.json").is_file())
            final.run_final_shards(
                ledger, plan_path=fixture.plan_path, repository=fixture.root,
                maximum_workers=4,
                runner=lambda spec: self.fail("completed shard was rerun"),
                adapter=lambda raw, plan, index: fixture.games(index),
                clock=lambda: "2026-09-01T00:15:00Z",
            )

    def test_started_missing_or_invalid_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            ledger = fixture.root / "ledger"
            q.start_final_shard(
                ledger, binding_path=fixture.binding, index=0,
                started_at_utc="2026-08-31T23:00:00Z",
            )
            with self.assertRaises(q.SpentShardError):
                final.run_final_shards(
                    ledger, plan_path=fixture.plan_path,
                    repository=fixture.root, runner=lambda spec: None,
                    adapter=lambda raw, plan, index: fixture.games(index),
                )

    def test_worker_limit_and_526_failure_do_not_emit_qualified_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(final.FinalError, "exactly four"):
                final.run_final_shards(
                    fixture.root / "ledger", plan_path=fixture.plan_path,
                    repository=fixture.root, maximum_workers=5,
                    runner=lambda spec: spec["index"],
                    adapter=lambda raw, plan, index: fixture.games(index),
                )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            ledger = fixture.root / "ledger"
            aggregate = final.run_final_shards(
                ledger, plan_path=fixture.plan_path, repository=fixture.root,
                runner=lambda spec: spec["index"],
                adapter=lambda raw, plan, index: fixture.games(index, wins=526),
                clock=lambda: "2026-08-31T23:00:00Z",
            )
            self.assertEqual(aggregate["status"], "final-gate-failed")
            self.assertFalse((ledger / "rank4-qualified-inputs.json").exists())

    def test_gate_command_uses_five_pair_offsets_and_selected_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ShardFixture(pathlib.Path(temporary))
            command = final.gate_command(
                fixture.plan, 37, fixture.root / "out.json"
            )
            self.assertEqual(command[command.index("--pair-offset") + 1], "185")
            self.assertEqual(command[command.index("--pair-count") + 1], "5")
            self.assertEqual(command[command.index("--mode") + 1], "actual-clock")
            self.assertEqual(command[command.index("--candidate-nodes") + 1], "80000")
            self.assertEqual(command[command.index("--max-turns") + 1], "320")


if __name__ == "__main__":
    unittest.main()
