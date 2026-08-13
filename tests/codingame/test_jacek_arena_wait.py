import importlib.util
import io
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
SPEC = importlib.util.spec_from_file_location(
    "jacek_arena_wait_for_window", BOT / "wait_for_arena_window.py"
)
wait_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(wait_gate)


AGENT_ID = 7001
SUBMISSION_ID = 8002


def battle(
    game_id,
    *,
    submission_id=SUBMISSION_ID,
    done=True,
    player_count=2,
):
    players = [
        {
            "playerAgentId": AGENT_ID,
            "submissionId": submission_id,
            "position": 0,
        }
    ]
    if player_count == 2:
        players.append(
            {
                "playerAgentId": 900000 + int(game_id),
                "submissionId": 100000 + int(game_id),
                "position": 1,
            }
        )
    return {"gameId": game_id, "done": done, "players": players}


def complete_window(count=90):
    return [battle(index + 1) for index in range(count)]


class ArenaWindowWaitTests(unittest.TestCase):
    def classify(self, battles):
        return wait_gate.classify_battle_metadata(
            battles, agent_id=AGENT_ID, submission_id=SUBMISSION_ID
        )

    def collector_command(self):
        return [
            "python3",
            "submissions/codingame/tools/collect_arena_batch.py",
            "--agent-id",
            str(AGENT_ID),
            "--submission-id",
            str(SUBMISSION_ID),
            "--expected-games",
            "90",
        ]

    def test_one_player_not_done_record_remains_pending(self):
        rows = complete_window(89)
        rows.append(battle(90, done=False, player_count=1))
        report = self.classify(rows)
        self.assertEqual(report["complete_two_player_count"], 89)
        self.assertEqual(report["pending_count"], 1)
        self.assertEqual(
            report["pending"][0]["reasons"], ["not_done", "player_count_1"]
        )
        self.assertFalse(report["collector_permitted"])
        self.assertEqual(report["detail_requests"], 0)

    def test_two_player_not_done_record_remains_pending(self):
        rows = complete_window(89)
        rows.append(battle(90, done=False))
        report = self.classify(rows)
        self.assertEqual(report["complete_two_player_count"], 89)
        self.assertEqual(report["pending_count"], 1)
        self.assertEqual(report["pending"][0]["reasons"], ["not_done"])
        self.assertFalse(report["collector_permitted"])

    def test_exact_90_permits_and_invokes_collector_once(self):
        report = self.classify(complete_window())
        calls = []

        class Completed:
            returncode = 0

        def runner(command, *, check):
            calls.append((command, check))
            return Completed()

        returncode = wait_gate.run_collector_if_ready(
            report, self.collector_command(), runner=runner
        )
        self.assertTrue(report["collector_permitted"])
        self.assertEqual(returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][1])

    def test_91_complete_games_refuses_collector_without_calling_runner(self):
        report = self.classify(complete_window(91))
        calls = []
        with self.assertRaises(wait_gate.CollectorRefused):
            wait_gate.run_collector_if_ready(
                report,
                self.collector_command(),
                runner=lambda *args, **kwargs: calls.append((args, kwargs)),
            )
        self.assertTrue(report["overfull"])
        self.assertFalse(report["collector_permitted"])
        self.assertEqual(calls, [])

    def test_old_submission_metadata_is_ignored(self):
        rows = complete_window()
        rows.extend(
            battle(100 + index, submission_id=SUBMISSION_ID - 1)
            for index in range(3)
        )
        report = self.classify(rows)
        self.assertEqual(report["complete_two_player_count"], 90)
        self.assertEqual(report["ignored_other_submission_count"], 3)
        self.assertTrue(report["collector_permitted"])

    def test_forbidden_game_detail_key_is_rejected_without_visiting_value(self):
        class ExplodingValue(dict):
            def items(self):
                raise AssertionError("detail value was consumed")

        row = battle(1)
        row["frames"] = ExplodingValue()
        with self.assertRaisesRegex(wait_gate.MetadataError, "game-detail field"):
            self.classify([row])

    def test_mismatched_or_missing_collector_bindings_are_rejected(self):
        wrong = self.collector_command()
        wrong[wrong.index(str(SUBMISSION_ID))] = str(SUBMISSION_ID + 1)
        with self.assertRaisesRegex(wait_gate.CollectorRefused, "submission-id"):
            wait_gate.validate_collector_command(
                wrong, agent_id=AGENT_ID, submission_id=SUBMISSION_ID
            )
        missing_expected = self.collector_command()[:-2]
        with self.assertRaisesRegex(wait_gate.CollectorRefused, "expected-games"):
            wait_gate.validate_collector_command(
                missing_expected, agent_id=AGENT_ID, submission_id=SUBMISSION_ID
            )

        not_really_invoked = [
            "python3",
            "-c",
            "collect_arena_batch.py",
            "--agent-id",
            str(AGENT_ID),
            "--submission-id",
            str(SUBMISSION_ID),
            "--expected-games",
            "90",
        ]
        with self.assertRaisesRegex(wait_gate.CollectorRefused, "executed script"):
            wait_gate.validate_collector_command(
                not_really_invoked, agent_id=AGENT_ID, submission_id=SUBMISSION_ID
            )

    def test_forged_ready_counts_without_exact_ids_are_rejected(self):
        report = self.classify(complete_window())
        report["complete_game_ids"] = report["complete_game_ids"][:-1]
        calls = []
        with self.assertRaises(wait_gate.CollectorRefused):
            wait_gate.run_collector_if_ready(
                report,
                self.collector_command(),
                runner=lambda *args, **kwargs: calls.append((args, kwargs)),
            )
        self.assertEqual(calls, [])

    def test_fetch_uses_only_battle_list_metadata_endpoint(self):
        requested = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.close()

        def opener(request, *, timeout):
            requested.append((request.full_url, request.data, timeout))
            return Response(json.dumps(complete_window()).encode("utf-8"))

        rows = wait_gate.fetch_battle_metadata(
            AGENT_ID, timeout_seconds=7.0, opener=opener
        )
        report = self.classify(rows)
        self.assertEqual(
            requested,
            [
                (
                    wait_gate.BATTLE_LIST_URL,
                    json.dumps([AGENT_ID, None], separators=(",", ":")).encode(
                        "ascii"
                    ),
                    7.0,
                )
            ],
        )
        self.assertNotIn("findByGameId", requested[0][0])
        self.assertEqual(report["detail_requests"], 0)

    def test_wait_does_not_return_on_pending_snapshot(self):
        pending = complete_window(89) + [battle(90, done=False, player_count=1)]
        snapshots = iter([pending, complete_window()])
        sleeps = []
        clock = iter([0.0, 0.0])
        report = wait_gate.wait_for_exact_window(
            lambda: next(snapshots),
            agent_id=AGENT_ID,
            submission_id=SUBMISSION_ID,
            poll_seconds=0.0,
            timeout_seconds=10.0,
            monotonic=lambda: next(clock),
            sleeper=sleeps.append,
        )
        self.assertTrue(report["collector_permitted"])
        self.assertEqual(sleeps, [0.0])


if __name__ == "__main__":
    unittest.main()
