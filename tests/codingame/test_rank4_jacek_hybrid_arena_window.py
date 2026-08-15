import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "rank_4_jacek_hybrid"
    / "arena_window.py"
)
SPEC = importlib.util.spec_from_file_location("rank4_jacek_hybrid_arena_window", MODULE_PATH)
arena = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)


AGENT_ID = 7_001
SUBMISSION_ID = 8_002
COMMIT = "a" * 40
T0 = "2026-08-13T19:15:07Z"
PLAY = "2026-08-13T20:00:00Z"
UPLOAD = "2026-08-13T20:01:00Z"
CREATED = "2026-08-13T20:02:00Z"
DEADLINE = "2026-08-15T07:15:07Z"


def canonical_write(path, value):
    content = arena.canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return arena.sha256_bytes(content)


def battle(game_id, *, done=True, players=2, submission_id=SUBMISSION_ID):
    rows = [
        {
            "playerAgentId": AGENT_ID,
            "position": 0,
            "submissionId": submission_id,
        }
    ]
    if players == 2:
        rows.append(
            {
                "playerAgentId": 90_000 + game_id,
                "position": 1,
                "submissionId": 100_000 + game_id,
            }
        )
    return {"done": done, "gameId": game_id, "players": rows}


def detail(game_id, *, frame_player=0, stderr=None, stdout="0"):
    frame = {"agentId": frame_player, "stdout": stdout}
    if stderr is not None:
        frame["stderr"] = stderr
    return {
        "agents": [
            {"agentId": AGENT_ID, "index": 0},
            {"agentId": 90_000 + game_id, "index": 1},
        ],
        "frames": [frame],
        "gameId": game_id,
        "ranks": [0, 1],
    }


class CampaignFixture:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.repository = root / "repository"
        self.repository.mkdir()
        self.output = self.repository / "results" / "rank_4_jacek_hybrid" / "arena"
        registry = {
            "records": [],
            "schema": arena.EXCLUSION_SCHEMA,
            "selection": "synthetic ID-only unit-test registry",
            "sources": [],
        }
        registry_content = arena.canonical_json_bytes(registry)
        self.registry_sha = arena.sha256_bytes(registry_content)
        self.registry = self.output / "exclusions" / f"{self.registry_sha}.json"
        self.registry.parent.mkdir(parents=True)
        self.registry.write_bytes(registry_content)
        self.campaign = self.repository / "results" / "rank_4_jacek_hybrid" / "campaign.json"
        campaign = {
            "arena_exclusions": {
                "path": self.registry.relative_to(self.repository).as_posix(),
                "schema": arena.EXCLUSION_SCHEMA,
                "sha256": self.registry_sha,
            },
            "campaign_id": "synthetic-hybrid-campaign",
            "schema": "papersoccer.rank4-jacek-hybrid-campaign.v1",
            "time_boundary": {
                "deadline_utc": DEADLINE,
                "goal_created_at_utc": T0,
            },
        }
        canonical_write(self.campaign, campaign)
        self.plan_sha, self.plan_path, self.plan = arena.create_plan(
            campaign_path=self.campaign,
            planned_at_utc="2026-08-13T19:20:00Z",
            output_root=self.output,
            repository=self.repository,
        )
        self.source = self.repository / "submission.cpp"
        self.copyback = self.repository / "copyback.cpp"
        self.source.write_bytes(b"int main(){return 0;}\n")
        self.copyback.write_bytes(self.source.read_bytes())

        def fake_git(repository, source, commit):
            return {
                "commit": commit,
                "source_path": source.relative_to(repository).as_posix(),
                "status": "tracked-committed-clean",
            }

        self.attestation_sha, self.attestation_path, self.attestation = (
            arena.create_attestation(
                plan_path=self.plan_path,
                plan_sha256=self.plan_sha,
                window_id=arena.VALIDATION_WINDOW_ID,
                generated_source=self.source,
                copied_back_source=self.copyback,
                repository=self.repository,
                repository_commit=COMMIT,
                agent_id=AGENT_ID,
                submission_id=SUBMISSION_ID,
                play_checked_at_utc=PLAY,
                uploaded_at_utc=UPLOAD,
                created_at_utc=CREATED,
                preflight={key: True for key in arena.PREFLIGHT_KEYS},
                play_stdout_legal=True,
                play_telemetry_ok=True,
                output_root=self.output,
                git_verifier=fake_git,
            )
        )

    def synthetic_manifest(self, *, clean=True):
        data_root = self.output
        generic_source = data_root / "source_payloads" / f"{self.attestation['source']['sha256']}.source"
        generic_source.parent.mkdir(parents=True, exist_ok=True)
        generic_source.write_bytes(self.source.read_bytes())
        generic_registry = data_root / "exclusions" / f"{self.registry_sha}.json"
        generic_registry.parent.mkdir(parents=True, exist_ok=True)
        if not generic_registry.exists():
            generic_registry.write_bytes(self.registry.read_bytes())
        stored = []
        for game_id in range(1, 91):
            replay_payload = {"gameId": game_id, "synthetic": True}
            replay_content = arena.canonical_json_bytes(replay_payload)
            replay_hash = arena.sha256_bytes(replay_content)
            raw_path = data_root / "raw" / "game-detail-v1" / str(game_id) / f"{replay_hash}.json"
            normalized_path = data_root / "normalized" / "game-detail-v1" / str(game_id) / f"{replay_hash}.json"
            replay_path = data_root / "replay_payloads" / str(game_id) / f"{replay_hash}.json"
            for payload_path in (raw_path, normalized_path, replay_path):
                payload_path.parent.mkdir(parents=True, exist_ok=True)
                payload_path.write_bytes(replay_content)
            record = {
                "acquisition": {
                    "normalized_path": normalized_path.relative_to(self.repository).as_posix(),
                    "normalized_sha256": replay_hash,
                    "raw_path": raw_path.relative_to(self.repository).as_posix(),
                    "raw_sha256": replay_hash,
                    "replay_payload_path": replay_path.relative_to(self.repository).as_posix(),
                },
                "focus": {
                    "agent_id": AGENT_ID,
                    "submission_id": SUBMISSION_ID,
                },
                "game_id": game_id,
                "operational": {
                    "classification": "clean" if clean else "operationally-terminated",
                    "focus_status": "ok",
                },
                "replay": {
                    "rules_validation": {
                        "status": "terminal-valid" if clean else "incomplete"
                    }
                },
                "schema": arena.GENERIC_GAME_SCHEMA,
                "source_sha256": self.attestation["source"]["sha256"],
                "status": "accepted",
            }
            content = arena.canonical_json_bytes(record)
            digest = arena.sha256_bytes(content)
            record_path = data_root / "game_records" / str(game_id) / f"{digest}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_bytes(content)
            stored.append(
                {
                    "record": record,
                    "record_path": record_path.relative_to(self.repository).as_posix(),
                    "record_sha256": digest,
                }
            )
        manifest = {
            "binding": {
                "agent_id": AGENT_ID,
                "asserted_submission_id": SUBMISSION_ID,
                "collector_sha256": arena.sha256_file(arena.GENERIC_COLLECTOR),
                "repository_commit": COMMIT,
                "schema": arena.GENERIC_BINDING_SCHEMA,
                "source": {
                    "archived_path": generic_source.relative_to(self.repository).as_posix(),
                    "bytes": len(self.source.read_bytes()),
                    "characters": len(self.source.read_bytes()),
                    "sha256": self.attestation["source"]["sha256"],
                },
            },
            "coverage": {
                "accepted_games": 90,
                "battle_window_games": 90,
                "expected_games": 90,
                "focus_operational_failures": 0,
                "full_window_accounted": True,
            },
            "collector_sha256": arena.sha256_file(arena.GENERIC_COLLECTOR),
            "exclusion_registry": {
                "path": generic_registry.relative_to(self.repository).as_posix(),
                "sha256": self.registry_sha,
            },
            "games": stored,
            "run_id": arena.VALIDATION_WINDOW_ID,
            "schema": arena.GENERIC_BATCH_SCHEMA,
        }
        content = arena.canonical_json_bytes(manifest)
        digest = arena.sha256_bytes(content)
        path = data_root / "manifests" / str(AGENT_ID) / str(SUBMISSION_ID) / self.attestation["source"]["sha256"] / f"{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "manifest": manifest,
            "manifest_path": path,
            "manifest_sha256": digest,
            "result": {
                "coverage": manifest["coverage"],
                "manifest_path": path.relative_to(self.repository).as_posix(),
                "manifest_sha256": digest,
                "run_id": arena.VALIDATION_WINDOW_ID,
                "schema": arena.GENERIC_BATCH_SCHEMA,
            },
        }


class ArenaWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CampaignFixture(pathlib.Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_preregisters_only_validation_and_optional_rollback(self):
        plan = self.fixture.plan
        self.assertFalse(plan["results_observed_before_assignment"])
        self.assertEqual(
            [item["window_id"] for item in plan["windows"]],
            [arena.VALIDATION_WINDOW_ID, arena.ROLLBACK_WINDOW_ID],
        )
        self.assertEqual(plan["windows"][0]["expected_games"], 90)
        self.assertEqual(plan["windows"][0]["role"], "arena-validation")
        self.assertFalse(plan["windows"][0]["training_eligible"])
        self.assertTrue(plan["windows"][0]["training_forbidden"])
        self.assertTrue(plan["windows"][1]["optional"])
        self.assertEqual(plan["windows"][1]["role"], "rollback-accounting")

    def test_attestation_is_exact_ascii_and_discloses_api_limit(self):
        attestation = self.fixture.attestation
        self.assertEqual(
            attestation["source"]["sha256"],
            attestation["editor_copyback"]["sha256"],
        )
        self.assertEqual(attestation["source"]["bytes"], attestation["source"]["characters"])
        self.assertEqual(attestation["git"]["status"], "tracked-committed-clean")
        self.assertEqual(attestation["upload_bytes_disclosure"], arena.DISCLOSURE)
        self.assertFalse(attestation["editor_copyback"]["api_readable"])

    def test_rollback_contract_accepts_only_exact_h62_identity(self):
        rollback = copy.deepcopy(self.fixture.attestation)
        rollback["window"] = arena.window_from_plan(
            self.fixture.plan, arena.ROLLBACK_WINDOW_ID
        )
        rollback["source"]["sha256"] = arena.SAFE_H62_SHA256
        rollback["source"]["bytes"] = arena.SAFE_H62_BYTES
        rollback["source"]["characters"] = arena.SAFE_H62_BYTES
        rollback["editor_copyback"]["sha256"] = arena.SAFE_H62_SHA256
        rollback["editor_copyback"]["bytes"] = arena.SAFE_H62_BYTES
        rollback["editor_copyback"]["characters"] = arena.SAFE_H62_BYTES
        arena.validate_attestation(
            rollback,
            self.fixture.plan,
            repository=self.fixture.repository,
            verify_files=False,
        )

        wrong_size = copy.deepcopy(rollback)
        wrong_size["source"]["bytes"] -= 1
        wrong_size["source"]["characters"] -= 1
        wrong_size["editor_copyback"]["bytes"] -= 1
        wrong_size["editor_copyback"]["characters"] -= 1
        with self.assertRaisesRegex(arena.ArenaWindowError, "exact safe H62"):
            arena.validate_attestation(
                wrong_size,
                self.fixture.plan,
                repository=self.fixture.repository,
                verify_files=False,
            )

        wrong_hash = copy.deepcopy(rollback)
        wrong_hash["source"]["sha256"] = "0" * 64
        wrong_hash["editor_copyback"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(arena.ArenaWindowError, "exact safe H62"):
            arena.validate_attestation(
                wrong_hash,
                self.fixture.plan,
                repository=self.fixture.repository,
                verify_files=False,
            )

    def test_attestation_rejects_copyback_difference_and_failed_gate(self):
        self.fixture.copyback.write_bytes(b"different\n")
        with self.assertRaisesRegex(arena.ArenaWindowError, "byte-identical"):
            arena.create_attestation(
                plan_path=self.fixture.plan_path,
                plan_sha256=self.fixture.plan_sha,
                window_id=arena.VALIDATION_WINDOW_ID,
                generated_source=self.fixture.source,
                copied_back_source=self.fixture.copyback,
                repository=self.fixture.repository,
                repository_commit=COMMIT,
                agent_id=AGENT_ID,
                submission_id=SUBMISSION_ID,
                play_checked_at_utc=PLAY,
                uploaded_at_utc=UPLOAD,
                created_at_utc=CREATED,
                preflight={key: True for key in arena.PREFLIGHT_KEYS},
                play_stdout_legal=True,
                play_telemetry_ok=True,
                output_root=self.fixture.output,
                git_verifier=lambda *args: {},
            )

    def test_attestation_rejects_generated_source_as_copyback_alias(self):
        with self.assertRaisesRegex(arena.ArenaWindowError, "distinct retained file"):
            arena.create_attestation(
                plan_path=self.fixture.plan_path,
                plan_sha256=self.fixture.plan_sha,
                window_id=arena.VALIDATION_WINDOW_ID,
                generated_source=self.fixture.source,
                copied_back_source=self.fixture.source,
                repository=self.fixture.repository,
                repository_commit=COMMIT,
                agent_id=AGENT_ID,
                submission_id=SUBMISSION_ID,
                play_checked_at_utc=PLAY,
                uploaded_at_utc=UPLOAD,
                created_at_utc=CREATED,
                preflight={key: True for key in arena.PREFLIGHT_KEYS},
                play_stdout_legal=True,
                play_telemetry_ok=True,
                output_root=self.fixture.output,
                git_verifier=lambda *args: {},
            )

    def test_attestation_propagates_dirty_or_untracked_git_refusal(self):
        def refuse(*args):
            raise arena.ArenaWindowError("tracked worktree is not clean")

        with self.assertRaisesRegex(arena.ArenaWindowError, "not clean"):
            arena.create_attestation(
                plan_path=self.fixture.plan_path,
                plan_sha256=self.fixture.plan_sha,
                window_id=arena.VALIDATION_WINDOW_ID,
                generated_source=self.fixture.source,
                copied_back_source=self.fixture.copyback,
                repository=self.fixture.repository,
                repository_commit=COMMIT,
                agent_id=AGENT_ID,
                submission_id=SUBMISSION_ID,
                play_checked_at_utc=PLAY,
                uploaded_at_utc=UPLOAD,
                created_at_utc=CREATED,
                preflight={key: True for key in arena.PREFLIGHT_KEYS},
                play_stdout_legal=True,
                play_telemetry_ok=True,
                output_root=self.fixture.output,
                git_verifier=refuse,
            )

    def test_git_attestation_checks_head_clean_tracked_and_committed_bytes(self):
        calls = []

        class Completed:
            def __init__(self, returncode=0, stdout=b""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = b""

        def runner(command, **kwargs):
            calls.append(command)
            arguments = tuple(command[1:])
            if arguments == ("rev-parse", "--verify", "HEAD"):
                return Completed(stdout=COMMIT + "\n")
            if arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
                return Completed(stdout="")
            if arguments[:2] == ("ls-files", "--error-unmatch"):
                return Completed(stdout=b"submission.cpp\n")
            if arguments == ("show", f"{COMMIT}:submission.cpp"):
                return Completed(stdout=self.fixture.source.read_bytes())
            return Completed(returncode=1)

        binding = arena.verify_git_source(
            self.fixture.repository,
            self.fixture.source,
            COMMIT,
            runner=runner,
        )
        self.assertEqual(binding["status"], "tracked-committed-clean")
        self.assertEqual(binding["source_path"], "submission.cpp")
        self.assertEqual(len(calls), 4)

    def test_git_attestation_rejects_tracked_dirty_status(self):
        class Completed:
            returncode = 0
            stderr = ""

            def __init__(self, stdout):
                self.stdout = stdout

        def runner(command, **kwargs):
            if command[1] == "rev-parse":
                return Completed(COMMIT + "\n")
            return Completed(" M submission.cpp\n")

        with self.assertRaisesRegex(arena.ArenaWindowError, "not clean"):
            arena.verify_git_source(
                self.fixture.repository,
                self.fixture.source,
                COMMIT,
                runner=runner,
            )

    def test_metadata_exact90_pending_and_overfull_are_distinct(self):
        exact = arena.classify_metadata(
            [battle(game_id) for game_id in range(1, 91)],
            agent_id=AGENT_ID,
            submission_id=SUBMISSION_ID,
        )
        self.assertTrue(exact["ready"])
        pending = arena.classify_metadata(
            [battle(game_id) for game_id in range(1, 90)]
            + [battle(90, done=False, players=1)],
            agent_id=AGENT_ID,
            submission_id=SUBMISSION_ID,
        )
        self.assertFalse(pending["ready"])
        self.assertEqual(pending["pending_games"], 1)
        overfull = arena.classify_metadata(
            [battle(game_id) for game_id in range(1, 92)],
            agent_id=AGENT_ID,
            submission_id=SUBMISSION_ID,
        )
        self.assertTrue(overfull["overfull"])

    def test_metadata_rejects_embedded_detail_without_consuming_it(self):
        class Exploding(dict):
            def items(self):
                raise AssertionError("detail value was consumed")

        row = battle(1)
        row["frames"] = Exploding()
        with self.assertRaisesRegex(arena.ArenaWindowError, "embedded"):
            arena.classify_metadata(
                [row], agent_id=AGENT_ID, submission_id=SUBMISSION_ID
            )

    def test_focus_timeout_and_malformed_output_are_sanitized(self):
        timeout = arena.inspect_focus_detail(
            detail(1, stderr="Agent $0 timed out"),
            game_id=1,
            focus_agent_id=AGENT_ID,
            rules_validator=lambda turns, winner: {"status": "terminal-valid"},
        )
        self.assertEqual(timeout, {"category": "timeout", "focus_failure": True, "game_id": 1})
        malformed = arena.inspect_focus_detail(
            detail(2, stdout="not-an-action"),
            game_id=2,
            focus_agent_id=AGENT_ID,
            rules_validator=lambda turns, winner: {"status": "terminal-valid"},
        )
        self.assertTrue(malformed["focus_failure"])
        self.assertEqual(malformed["category"], "malformed-transcript")
        self.assertNotIn("evidence", timeout)

    def test_scoped_opponent_timeout_does_not_claim_focus_failure(self):
        opponent = detail(3, frame_player=1, stderr="Agent $1 timed out", stdout="")
        result = arena.inspect_focus_detail(
            opponent,
            game_id=3,
            focus_agent_id=AGENT_ID,
            rules_validator=lambda turns, winner: {"status": "incomplete", "failing_player_id": None},
        )
        self.assertFalse(result["focus_failure"])
        self.assertIsNone(result["category"])

    def test_watch_returns_42_before_collection_on_focus_failure(self):
        runner_calls = []
        details = []

        def fetch_detail(game_id):
            details.append(game_id)
            return {"synthetic": game_id}

        code, result = arena.watch_collect(
            plan_path=self.fixture.plan_path,
            plan_sha256=self.fixture.plan_sha,
            attestation_path=self.fixture.attestation_path,
            attestation_sha256=self.fixture.attestation_sha,
            exclusion_registry=self.fixture.registry,
            exclusion_sha256=self.fixture.registry_sha,
            data_root=self.fixture.output,
            repository=self.fixture.repository,
            fetch_battles=lambda: [battle(1)],
            fetch_detail=fetch_detail,
            detail_inspector=lambda payload, **kwargs: {
                "category": "crash",
                "focus_failure": True,
                "game_id": kwargs["game_id"],
            },
            runner=lambda *args, **kwargs: runner_calls.append((args, kwargs)),
            timeout_seconds=10,
        )
        self.assertEqual(code, arena.FAILURE_EXIT)
        self.assertEqual(result["status"], "focus-operational-failure")
        self.assertEqual(result["category"], "crash")
        self.assertTrue(result["rollback_required"])
        self.assertEqual(details, [1])
        self.assertEqual(runner_calls, [])
        forbidden = {"frames", "transcript", "stdout", "opponent", "result"}
        self.assertTrue(forbidden.isdisjoint(result))
        failure_path = arena.resolve_path(
            result["failure_receipt_path"], self.fixture.repository
        )
        self.assertEqual(
            arena.check_artifact(failure_path, repository=self.fixture.repository)[
                "status"
            ],
            "ok",
        )

    def test_pre90_timeout_reveals_only_progress_and_never_collects(self):
        runner_calls = []
        progress = []
        monotonic = iter([0.0, 0.0])
        code, result = arena.watch_collect(
            plan_path=self.fixture.plan_path,
            plan_sha256=self.fixture.plan_sha,
            attestation_path=self.fixture.attestation_path,
            attestation_sha256=self.fixture.attestation_sha,
            exclusion_registry=self.fixture.registry,
            exclusion_sha256=self.fixture.registry_sha,
            data_root=self.fixture.output,
            repository=self.fixture.repository,
            poll_seconds=0,
            timeout_seconds=0,
            fetch_battles=lambda: [battle(game_id) for game_id in range(1, 90)],
            fetch_detail=lambda game_id: {"gameId": game_id, "synthetic": True},
            detail_inspector=lambda payload, **kwargs: {
                "category": None,
                "focus_failure": False,
                "game_id": kwargs["game_id"],
            },
            runner=lambda *args, **kwargs: runner_calls.append((args, kwargs)),
            progress=progress.append,
            monotonic=lambda: next(monotonic),
        )
        self.assertEqual(code, 2)
        self.assertEqual(runner_calls, [])
        self.assertEqual(
            progress,
            [
                {
                    "complete_games": 89,
                    "expected_games": 90,
                    "focus_failure": False,
                    "pending_games": 0,
                    "status": "waiting",
                }
            ],
        )
        forbidden = {
            "action",
            "frames",
            "opponent",
            "outcome",
            "rank",
            "result",
            "score",
            "stderr",
            "stdout",
            "transcript",
            "turns",
            "winner",
        }
        self.assertTrue(forbidden.isdisjoint(result))
        self.assertTrue(forbidden.isdisjoint(progress[0]))

    def test_watch_refuses_91_without_detail_or_collector(self):
        calls = []
        with self.assertRaisesRegex(arena.ArenaWindowError, "exceeds exactly 90"):
            arena.watch_collect(
                plan_path=self.fixture.plan_path,
                plan_sha256=self.fixture.plan_sha,
                attestation_path=self.fixture.attestation_path,
                attestation_sha256=self.fixture.attestation_sha,
                exclusion_registry=self.fixture.registry,
                exclusion_sha256=self.fixture.registry_sha,
                data_root=self.fixture.output,
                repository=self.fixture.repository,
                fetch_battles=lambda: [battle(game_id) for game_id in range(1, 92)],
                fetch_detail=lambda game_id: calls.append(("detail", game_id)),
                runner=lambda *args, **kwargs: calls.append(("runner", args, kwargs)),
            )
        self.assertEqual(calls, [])

    def _collect_exact_with_fake_subprocess(self):
        synthetic = self.fixture.synthetic_manifest()
        runner_calls = []

        class Completed:
            returncode = 0
            stderr = ""
            stdout = json.dumps(synthetic["result"])

        def runner(command, **kwargs):
            runner_calls.append((command, kwargs))
            return Completed()

        def verifier(result, **kwargs):
            self.assertEqual(result, synthetic["result"])
            self.assertEqual(kwargs["expected_game_ids"], list(range(1, 91)))
            return {
                "manifest": synthetic["manifest"],
                "manifest_path": synthetic["manifest_path"],
                "manifest_sha256": synthetic["manifest_sha256"],
            }

        progress = []
        code, result = arena.watch_collect(
            plan_path=self.fixture.plan_path,
            plan_sha256=self.fixture.plan_sha,
            attestation_path=self.fixture.attestation_path,
            attestation_sha256=self.fixture.attestation_sha,
            exclusion_registry=self.fixture.registry,
            exclusion_sha256=self.fixture.registry_sha,
            data_root=self.fixture.output,
            repository=self.fixture.repository,
            fetch_battles=lambda: [battle(game_id) for game_id in range(1, 91)],
            fetch_detail=lambda game_id: {"synthetic": game_id},
            detail_inspector=lambda payload, **kwargs: {
                "category": None,
                "focus_failure": False,
                "game_id": kwargs["game_id"],
            },
            runner=runner,
            collector_verifier=verifier,
            progress=progress.append,
        )
        return code, result, runner_calls, progress

    def test_exact90_invokes_immutable_generic_collector_once(self):
        code, result, runner_calls, progress = self._collect_exact_with_fake_subprocess()
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "exact-window-collected")
        self.assertFalse(result["training_eligible"])
        self.assertEqual(len(runner_calls), 1)
        command, kwargs = runner_calls[0]
        self.assertEqual(pathlib.Path(command[1]).name, "collect_arena_batch.py")
        bindings = {
            "--agent-id": str(AGENT_ID),
            "--submission-id": str(SUBMISSION_ID),
            "--expected-games": "90",
            "--run-id": arena.VALIDATION_WINDOW_ID,
            "--source-sha256": self.fixture.attestation["source"]["sha256"],
            "--repository-commit": COMMIT,
            "--exclusion-registry-sha256": self.fixture.registry_sha,
        }
        for flag, expected in bindings.items():
            self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertEqual(command.count(flag), 1)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(
            progress,
            [
                {
                    "complete_games": 90,
                    "expected_games": 90,
                    "focus_failure": False,
                    "pending_games": 0,
                    "status": "ready",
                }
            ],
        )

    def test_real_archive_verifier_checks_all_90_record_bindings(self):
        synthetic = self.fixture.synthetic_manifest()
        verified = arena.verify_collector_result(
            synthetic["result"],
            repository=self.fixture.repository,
            data_root=self.fixture.output,
            attestation=self.fixture.attestation,
            exclusion_sha256=self.fixture.registry_sha,
            expected_game_ids=list(range(1, 91)),
        )
        self.assertEqual(verified["manifest_sha256"], synthetic["manifest_sha256"])
        record_path = pathlib.Path(
            synthetic["manifest"]["games"][0]["record_path"]
        )
        (self.fixture.repository / record_path).write_text("{}\n")
        with self.assertRaisesRegex(arena.ArenaWindowError, "record archive"):
            arena.verify_collector_result(
                synthetic["result"],
                repository=self.fixture.repository,
                data_root=self.fixture.output,
                attestation=self.fixture.attestation,
                exclusion_sha256=self.fixture.registry_sha,
                expected_game_ids=list(range(1, 91)),
            )

    def test_derive_is_validation_only_with_zero_training_surfaces(self):
        code, result, _, _ = self._collect_exact_with_fake_subprocess()
        self.assertEqual(code, 0)
        receipt_path = arena.resolve_path(
            result["collection_receipt_path"], self.fixture.repository
        )
        digest, path, derivation = arena.derive_validation(
            plan_path=self.fixture.plan_path,
            plan_sha256=self.fixture.plan_sha,
            attestation_path=self.fixture.attestation_path,
            attestation_sha256=self.fixture.attestation_sha,
            collection_receipt_path=receipt_path,
            collection_receipt_sha256=result["collection_receipt_sha256"],
            output_root=self.fixture.output,
            repository=self.fixture.repository,
        )
        self.assertEqual(path.stem, digest)
        self.assertFalse(derivation["training_eligible"])
        self.assertTrue(derivation["training_forbidden"])
        self.assertEqual(
            derivation["fresh_arena_usage"],
            {
                "action_ranking_games": 0,
                "policy_rows": 0,
                "training_games": 0,
                "validation_games": 90,
                "value_rows": 0,
            },
        )
        self.assertEqual(len(derivation["records"]), 90)
        self.assertTrue(all("outcome" not in item and "action" not in item for item in derivation["records"]))
        self.assertEqual(
            arena.check_artifact(path, repository=self.fixture.repository)["status"],
            "ok",
        )

    def test_check_detects_content_address_tampering(self):
        checked = arena.check_artifact(
            self.fixture.plan_path, repository=self.fixture.repository
        )
        self.assertEqual(checked["status"], "ok")
        self.fixture.plan_path.write_text("{}\n")
        with self.assertRaisesRegex(arena.ArenaWindowError, "content-addressed"):
            arena.check_artifact(
                self.fixture.plan_path, repository=self.fixture.repository
            )


if __name__ == "__main__":
    unittest.main()
