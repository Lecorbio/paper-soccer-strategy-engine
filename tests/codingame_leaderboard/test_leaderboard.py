from __future__ import annotations

import os
import json
import math
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from benchmarks.codingame_leaderboard import leaderboard as subject


REPOSITORY = Path(__file__).resolve().parents[2]


def entrant(bot_id: str) -> dict:
    return {
        "id": bot_id,
        "displayName": bot_id.upper(),
        "submissionSha256": bot_id.rjust(64, "0")[-64:],
        "documentationUrl": f"https://example.test/{bot_id}",
        "aliases": [],
    }


def native_match(first: str, second: str, winner: str, *, forfeit: str | None = None) -> dict:
    loser = second if winner == first else first
    outcome = {"winnerId": winner, "loserId": loser, "reason": "goal", "forfeit": None}
    if forfeit is not None:
        outcome["forfeit"] = {
            "botId": forfeit,
            "classification": "timeout",
            "detail": "decision deadline exceeded",
        }
        outcome["reason"] = "forfeit"
    accepted_action = {
        "turn": 0, "botId": first, "player": 0, "opponentAction": "-",
        "action": "0", "accepted": True, "durationMicros": 1,
        "deadlineMillis": 1000, "failureClassification": None,
        "moves": [{
            "direction": 0, "from": {"x": 4, "y": 6}, "to": {"x": 4, "y": 5},
            "extraTurn": False,
            "statusAfter": (
                ("player_0_wins" if winner == first else "player_1_wins")
                if forfeit is None
                else "ongoing"
            ),
        }],
    }
    if forfeit is None:
        actions = [accepted_action]
    elif forfeit == first:
        actions = [{
            "turn": 0, "botId": first, "player": 0, "opponentAction": "-",
            "action": None, "accepted": False, "durationMicros": 1_000_000,
            "deadlineMillis": 1000, "failureClassification": "timeout", "moves": [],
        }]
    else:
        actions = [accepted_action, {
            "turn": 1, "botId": second, "player": 1, "opponentAction": "0",
            "action": None, "accepted": False, "durationMicros": 1_000_000,
            "deadlineMillis": 1000, "failureClassification": "timeout", "moves": [],
        }]
    player_one_durations = [action["durationMicros"] for action in actions if action["player"] == 0]
    player_two_durations = [action["durationMicros"] for action in actions if action["player"] == 1]
    return {
        "schema": subject.MATCH_SCHEMA,
        "participants": {
            "playerOne": {
                "id": first,
                "player": 0,
                "executable": subject._expected_executable_name(first),
            },
            "playerTwo": {
                "id": second,
                "player": 1,
                "executable": subject._expected_executable_name(second),
            },
        },
        "rules": {
            "width": 8, "height": 10,
            "goalRule": "OwnGoalsAllowed", "blockedRule": "MoverLoses",
        },
        "timeouts": {"firstMillis": 1000, "laterMillis": 200},
        "actions": actions,
        "outcome": outcome,
        "timings": {
            "totalMicros": sum(player_one_durations + player_two_durations),
            "playerOne": {
                "decisions": len(player_one_durations), "totalMicros": sum(player_one_durations),
                "maxMicros": max(player_one_durations, default=0),
            },
            "playerTwo": {
                "decisions": len(player_two_durations), "totalMicros": sum(player_two_durations),
                "maxMicros": max(player_two_durations, default=0),
            },
        },
        "provenance": {"refereeVersion": "test"},
    }


def make_executable(path: Path, source: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(source), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class RosterTests(unittest.TestCase):
    def test_checked_in_roster_covers_every_registered_artifact(self) -> None:
        roster = subject.load_roster()
        entrants = subject.validate_roster(roster, REPOSITORY)
        self.assertEqual(len(entrants), 20)
        rank4 = next(item for item in entrants if item["id"] == "rank_4")
        self.assertEqual([alias["id"] for alias in rank4["aliases"]], ["selfplay_nn_v2"])
        self.assertEqual(
            subject.sha256_file(REPOSITORY / rank4["submissionPath"]),
            subject.sha256_file(REPOSITORY / rank4["aliases"][0]["submissionPath"]),
        )

    def test_stale_submission_hash_is_rejected(self) -> None:
        roster = subject.load_roster()
        roster["entrants"][0]["submissionSha256"] = "0" * 64
        with self.assertRaisesRegex(subject.ContractError, "hash is stale"):
            subject.validate_roster(roster, REPOSITORY)


class SplitMixTests(unittest.TestCase):
    def test_reference_values(self) -> None:
        rng = subject.SplitMix64(0)
        self.assertEqual(
            [rng.next_u64() for _ in range(4)],
            [
                0xE220A8397B1DCDAF,
                0x6E789E6AA1B965F4,
                0x06C45D188009454F,
                0xF88BB8A8724C81EC,
            ],
        )


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ids = [item["id"] for item in subject.validate_roster(subject.load_roster(), REPOSITORY)]
        cls.schedule = subject.build_schedule(cls.ids)

    def test_frozen_schedule_hash_and_prefix(self) -> None:
        self.assertEqual(
            subject.schedule_sha256(self.schedule),
            "85e498f1a7c2996b72ef05c9140ad71ff8a577e1056d1da3cbe65445370b6b54",
        )
        self.assertEqual(
            (self.schedule[0]["playerOneId"], self.schedule[0]["playerTwoId"]),
            ("conservative_frontier_proof", "selfplay_nn"),
        )

    def test_balance_contract(self) -> None:
        subject.validate_schedule(self.schedule, self.ids)
        games = Counter()
        player_one = Counter()
        pairs = Counter()
        for game in self.schedule:
            games.update((game["playerOneId"], game["playerTwoId"]))
            player_one[game["playerOneId"]] += 1
            pairs[tuple(sorted((game["playerOneId"], game["playerTwoId"])))] += 1
        self.assertEqual(set(games.values()), {90})
        self.assertEqual(set(player_one.values()), {45})
        self.assertEqual(Counter(pairs.values()), {4: 120, 6: 70})

    def test_every_adjacent_block_is_color_swapped(self) -> None:
        for index in range(0, len(self.schedule), 2):
            first, second = self.schedule[index : index + 2]
            self.assertEqual(first["blockId"], second["blockId"])
            self.assertEqual(first["playerOneId"], second["playerTwoId"])
            self.assertEqual(first["playerTwoId"], second["playerOneId"])

    def test_seed_changes_schedule(self) -> None:
        self.assertNotEqual(self.schedule, subject.build_schedule(self.ids, subject.SCHEDULE_SEED + 1))


class TrueSkillTests(unittest.TestCase):
    def test_first_update_matches_golden_classic_trueskill_vector(self) -> None:
        winner, loser = subject.update_ratings(subject.Rating(), subject.Rating())
        self.assertAlmostEqual(winner.mu, 29.205473176557785, places=12)
        self.assertAlmostEqual(winner.sigma, 7.194816484813345, places=12)
        self.assertAlmostEqual(loser.mu, 20.794526823442215, places=12)
        self.assertAlmostEqual(loser.sigma, 7.194816484813345, places=12)
        self.assertAlmostEqual(winner.conservative_score, 7.621023722117748, places=12)

    def test_upset_moves_ratings_more_than_expected_result(self) -> None:
        strong = subject.Rating(35.0, 4.0)
        weak = subject.Rating(15.0, 4.0)
        expected_winner, _ = subject.update_ratings(strong, weak)
        upset_winner, _ = subject.update_ratings(weak, strong)
        self.assertGreater(upset_winner.mu - weak.mu, expected_winner.mu - strong.mu)

    def test_forfeit_is_rated_as_loss_and_recorded(self) -> None:
        entrants = [entrant("a"), entrant("b")]
        games = [
            {
                "playerOneId": "a",
                "playerTwoId": "b",
                "winnerId": "a",
                "forfeitId": "b",
            }
        ]
        standings, head = subject.rate_games(entrants, games)
        self.assertEqual(standings[0]["id"], "a")
        self.assertEqual(next(row for row in standings if row["id"] == "b")["forfeits"], 1)
        self.assertEqual(len(head), 2)
        self.assertNotIn(None, [row["score"] for row in head])

    def test_exact_rating_tie_uses_canonical_id(self) -> None:
        standings, _ = subject.rate_games([entrant("z"), entrant("a")], [])
        self.assertEqual([row["id"] for row in standings], ["a", "z"])

    def test_standings_comparison_accepts_only_tiny_platform_float_drift(self) -> None:
        standings, _ = subject.rate_games(
            [entrant("a"), entrant("b")],
            [{
                "playerOneId": "a",
                "playerTwoId": "b",
                "winnerId": "a",
                "forfeitId": None,
            }],
        )
        drifted = [dict(row) for row in standings]
        for field in subject.STANDING_FLOAT_FIELDS:
            drifted[0][field] = math.nextafter(drifted[0][field], math.inf)
        self.assertTrue(subject._standings_match(drifted, standings))

        for field in subject.STANDING_FLOAT_FIELDS:
            changed = [dict(row) for row in standings]
            changed[0][field] += 1e-8
            self.assertFalse(subject._standings_match(changed, standings))

        nonfinite = [dict(row) for row in standings]
        nonfinite[0]["score"] = math.inf
        self.assertFalse(subject._standings_match(nonfinite, standings))

        changed_win_rate = [dict(row) for row in standings]
        changed_win_rate[0]["winRate"] = math.nextafter(
            changed_win_rate[0]["winRate"], math.inf
        )
        self.assertFalse(subject._standings_match(changed_win_rate, standings))

    def test_standings_comparison_keeps_rank_and_records_exact(self) -> None:
        standings, _ = subject.rate_games([entrant("a"), entrant("b")], [])
        changed_rank = [dict(row) for row in standings]
        changed_rank[0]["rank"] = 2
        self.assertFalse(subject._standings_match(changed_rank, standings))

        changed_record = [dict(row) for row in standings]
        changed_record[0]["wins"] = 1
        self.assertFalse(subject._standings_match(changed_record, standings))


class NativeMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduled = {"playerOneId": "alpha", "playerTwoId": "beta"}

    def test_normalizes_winner_and_forfeit(self) -> None:
        match = native_match("alpha", "beta", "alpha", forfeit="beta")
        self.assertEqual(subject.normalize_native_match(match, self.scheduled), ("alpha", "beta"))

    def test_rejects_participant_mismatch(self) -> None:
        with self.assertRaisesRegex(subject.ContractError, "participants"):
            subject.normalize_native_match(native_match("alpha", "other", "alpha"), self.scheduled)

    def test_rejects_participant_executable_target_mismatch(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["participants"]["playerOne"]["executable"] = "/tmp/unreviewed-bot"
        with self.assertRaisesRegex(subject.ContractError, "executable differs"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_forfeit_charged_to_winner(self) -> None:
        with self.assertRaisesRegex(subject.ContractError, "forfeit"):
            subject.normalize_native_match(
                native_match("alpha", "beta", "alpha", forfeit="alpha"), self.scheduled
            )

    def test_rejects_empty_transcript(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["actions"] = []
        with self.assertRaisesRegex(subject.ContractError, "actions"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_empty_provenance(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["provenance"] = {}
        with self.assertRaisesRegex(subject.ContractError, "provenance"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_unversioned_extra_native_field(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["debugLog"] = "not part of v1"
        with self.assertRaisesRegex(subject.ContractError, "native match fields differ from v1"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_direction_coordinate_disagreement(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["actions"][0]["moves"][0]["to"] = {"x": 5, "y": 5}
        with self.assertRaisesRegex(subject.ContractError, "encoded direction"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_action_timing_summary_disagreement(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["timings"]["playerOne"]["maxMicros"] = 9
        with self.assertRaisesRegex(subject.ContractError, "timing maximum differs"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_accepted_action_after_deadline(self) -> None:
        match = native_match("alpha", "beta", "alpha")
        match["actions"][0]["durationMicros"] = 1_000_001
        match["timings"]["totalMicros"] = 1_000_001
        match["timings"]["playerOne"]["totalMicros"] = 1_000_001
        match["timings"]["playerOne"]["maxMicros"] = 1_000_001
        with self.assertRaisesRegex(subject.ContractError, "exceeds its decision deadline"):
            subject.normalize_native_match(match, self.scheduled)

    def test_rejects_forfeit_with_committed_move(self) -> None:
        match = native_match("alpha", "beta", "beta", forfeit="alpha")
        match["actions"][0]["moves"] = [{
            "direction": 0,
            "from": {"x": 4, "y": 6},
            "to": {"x": 4, "y": 5},
            "extraTurn": False,
            "statusAfter": "ongoing",
        }]
        with self.assertRaisesRegex(subject.ContractError, "must not commit any moves"):
            subject.normalize_native_match(match, self.scheduled)

    def test_preserves_malformed_rejected_action_as_forfeit_evidence(self) -> None:
        match = native_match("alpha", "beta", "beta", forfeit="alpha")
        action = match["actions"][0]
        action["action"] = "x"
        action["durationMicros"] = 1
        action["failureClassification"] = "invalid-character"
        match["outcome"]["forfeit"]["classification"] = "invalid-character"
        match["outcome"]["forfeit"]["detail"] = "response rejected atomically"
        match["timings"]["totalMicros"] = 1
        match["timings"]["playerOne"]["totalMicros"] = 1
        match["timings"]["playerOne"]["maxMicros"] = 1
        self.assertEqual(subject.normalize_native_match(match, self.scheduled), ("beta", "alpha"))

    def test_rejects_invalid_calendar_timestamp(self) -> None:
        for invalid in ("2026-02-30T00:00:00Z", "2026-08-13T00:00:00+00:00", "todayZ"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(subject.ContractError):
                    subject._parse_utc_timestamp(invalid, "generatedAtUtc")

    def test_rejects_infrastructure_classification(self) -> None:
        match = native_match("alpha", "beta", "beta", forfeit="alpha")
        match["actions"][-1]["failureClassification"] = "infrastructure-error"
        match["outcome"]["forfeit"]["classification"] = "infrastructure-error"
        with self.assertRaisesRegex(subject.ContractError, "not an allowed bot failure"):
            subject.normalize_native_match(match, self.scheduled)

    def test_authoritative_replay_checks_natural_winner_and_forfeit_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            referee = Path(name) / "referee.py"
            make_executable(
                referee,
                r'''
                import json, sys
                args = sys.argv[1:]
                transcript = args[args.index("--validate-transcript") + 1]
                if "--expected-winner" in args:
                    winner = int(args[args.index("--expected-winner") + 1])
                    terminal = True
                else:
                    assert "--allow-incomplete" in args
                    winner = None
                    terminal = False
                print(json.dumps({
                    "schema": "papersoccer.codingame-transcript-validation.v1",
                    "terminal": terminal,
                    "winnerPlayer": winner,
                    "terminalReason": "goal" if terminal else None,
                    "acceptedActionCount": 0 if not transcript else len(transcript.split("/")),
                    "edgeCount": len(transcript.replace("/", "")),
                }))
                ''',
            )
            subject.validate_match_replay(
                referee, native_match("alpha", "beta", "alpha"), self.scheduled
            )
            subject.validate_match_replay(
                referee,
                native_match("alpha", "beta", "alpha", forfeit="beta"),
                self.scheduled,
            )

    def test_authoritative_replay_rejects_wrong_edge_count(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            referee = Path(name) / "referee.py"
            make_executable(referee, 'print(\'{"schema":"papersoccer.codingame-transcript-validation.v1","terminal":true,"winnerPlayer":0,"terminalReason":"goal","acceptedActionCount":1,"edgeCount":9}\')\n')
            with self.assertRaisesRegex(subject.ContractError, "edge count"):
                subject.validate_match_replay(
                    referee, native_match("alpha", "beta", "alpha"), self.scheduled
                )

    def test_authoritative_replay_rejects_wrong_terminal_reason(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            referee = Path(name) / "referee.py"
            make_executable(referee, 'print(\'{"schema":"papersoccer.codingame-transcript-validation.v1","terminal":true,"winnerPlayer":0,"terminalReason":"blocked_mover","acceptedActionCount":1,"edgeCount":1}\')\n')
            with self.assertRaisesRegex(subject.ContractError, "terminal reason differs"):
                subject.validate_match_replay(
                    referee, native_match("alpha", "beta", "alpha"), self.scheduled
                )


class RefereeRunnerTests(unittest.TestCase):
    def test_runner_uses_exact_flags_and_clean_environment(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee.py"
            first = root / "first"
            second = root / "second"
            make_executable(first, "pass\n")
            make_executable(second, "pass\n")
            make_executable(
                referee,
                r'''
                import json, os, sys
                args = sys.argv[1:]
                # Parent variables must not leak into the referee.
                assert "PAPERSOCCER_TEST_SECRET" not in os.environ
                first_id = args[args.index("--player-one-id") + 1]
                second_id = args[args.index("--player-two-id") + 1]
                assert args[args.index("--first-timeout-ms") + 1] == "1000"
                assert args[args.index("--later-timeout-ms") + 1] == "200"
                print(json.dumps({
                    "schema": "papersoccer.codingame-match.v1",
                    "participants": {
                        "playerOne": {"id": first_id, "player": 0, "executable": "papersoccer_codingame_" + first_id + "_submission"},
                        "playerTwo": {"id": second_id, "player": 1, "executable": "papersoccer_codingame_" + second_id + "_submission"},
                    },
                    "rules": {"width": 8, "height": 10, "goalRule": "OwnGoalsAllowed", "blockedRule": "MoverLoses"},
                    "timeouts": {"firstMillis": 1000, "laterMillis": 200},
                    "actions": [{
                        "turn": 0, "botId": first_id, "player": 0, "opponentAction": "-",
                        "action": "0", "accepted": True, "durationMicros": 1,
                        "deadlineMillis": 1000, "failureClassification": None,
                        "moves": [{"direction": 0, "from": {"x": 4, "y": 6}, "to": {"x": 4, "y": 5}, "extraTurn": False, "statusAfter": "player_0_wins"}],
                    }],
                    "timings": {"totalMicros": 1,
                        "playerOne": {"decisions": 1, "totalMicros": 1, "maxMicros": 1},
                        "playerTwo": {"decisions": 0, "totalMicros": 0, "maxMicros": 0}},
                    "provenance": {"refereeVersion": "test"},
                    "outcome": {"winnerId": first_id, "loserId": second_id, "reason": "goal", "forfeit": None},
                }))
                ''',
            )
            scheduled = {"playerOneId": "a", "playerTwoId": "b"}
            os.environ["PAPERSOCCER_TEST_SECRET"] = "must-not-leak"
            try:
                match = subject.run_native_match(referee, first, second, scheduled)
            finally:
                del os.environ["PAPERSOCCER_TEST_SECRET"]
            self.assertEqual(match["outcome"]["winnerId"], "a")

    @unittest.skipUnless(hasattr(os, "killpg") and Path("/proc").exists(), "requires POSIX /proc")
    def test_watchdog_kills_referee_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee.py"
            first = root / "first"
            second = root / "second"
            pid_file = root / "child.pid"
            make_executable(first, "pass\n")
            make_executable(second, "pass\n")
            make_executable(
                referee,
                f'''
                import subprocess, time
                child = subprocess.Popen(["/bin/sh", "-c", "echo $$ > {pid_file}; exec sleep 60"])
                time.sleep(60)
                ''',
            )
            with self.assertRaisesRegex(subject.InfrastructureError, "infrastructure timeout"):
                subject.run_native_match(
                    referee,
                    first,
                    second,
                    {"playerOneId": "a", "playerTwoId": "b"},
                    infrastructure_timeout_seconds=0.15,
                )
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            # A killed, unreaped grandchild can briefly be a zombie; it must not be running.
            if Path(f"/proc/{child_pid}/stat").exists():
                state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                self.assertEqual(state, "Z")

    def test_nonzero_referee_exit_is_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee.py"
            first = root / "first"
            second = root / "second"
            make_executable(first, "pass\n")
            make_executable(second, "pass\n")
            make_executable(referee, "raise SystemExit(7)\n")
            with self.assertRaisesRegex(subject.InfrastructureError, "exited 7"):
                subject.run_native_match(
                    referee, first, second, {"playerOneId": "a", "playerTwoId": "b"}
                )

    def test_live_output_limit_aborts_referee(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee.py"
            first = root / "first"
            second = root / "second"
            make_executable(first, "pass\n")
            make_executable(second, "pass\n")
            make_executable(
                referee,
                "import sys, time\nsys.stdout.write('x' * 4096)\nsys.stdout.flush()\ntime.sleep(30)\n",
            )
            with self.assertRaisesRegex(subject.InfrastructureError, "output exceeded"):
                subject.run_native_match(
                    referee,
                    first,
                    second,
                    {"playerOneId": "a", "playerTwoId": "b"},
                    output_limit_bytes=1024,
                    infrastructure_timeout_seconds=2.0,
                )

    def test_output_limit_is_a_hard_file_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee.py"
            first = root / "first"
            second = root / "second"
            make_executable(first, "pass\n")
            make_executable(second, "pass\n")
            make_executable(
                referee,
                "import os\nos.write(1, b'x' * (32 * 1024 * 1024))\n",
            )
            with self.assertRaisesRegex(subject.InfrastructureError, "output exceeded"):
                subject.run_native_match(
                    referee,
                    first,
                    second,
                    {"playerOneId": "a", "playerTwoId": "b"},
                    output_limit_bytes=1024,
                    infrastructure_timeout_seconds=2.0,
                )


class FingerprintTests(unittest.TestCase):
    def test_contract_digest_includes_transcript_validation_cli_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name)
            main = repository / "src/codingame/main.cpp"
            main.parent.mkdir(parents=True)
            main.write_text("first transcript contract\n", encoding="utf-8")
            before = subject._contract_source_digest(repository)
            main.write_text("changed transcript contract\n", encoding="utf-8")
            after = subject._contract_source_digest(repository)
            self.assertNotEqual(before, after)

    def test_own_outputs_are_excluded_from_source_digest_and_dirty_flag(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name)
            os.system(f"git -C {repository} init -q")
            tracked = repository / "tracked.txt"
            tracked.write_text("source", encoding="utf-8")
            os.system(f"git -C {repository} add tracked.txt")
            os.system(
                f"git -C {repository} -c user.name=test -c user.email=test@example.test commit -qm initial"
            )
            checkpoint = repository / "checkpoint.json"
            before = subject._source_tree_digest(repository, (checkpoint,))
            checkpoint.write_text("generated", encoding="utf-8")
            after = subject._source_tree_digest(repository, (checkpoint,))
            self.assertEqual(before, after)

    def test_provenance_rejects_referee_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee"
            make_executable(referee, "pass\n")
            provenance = {
                "sourceCommit": "a" * 40,
                "sourceTreeSha256": "b" * 64,
                "sourceTreeDirty": False,
                "contractSourceSha256": "c" * 64,
                "referee": {
                    "schema": subject.MATCH_SCHEMA,
                    "version": "v1",
                    "sha256": "d" * 64,
                },
                "executables": {"alpha": "e" * 64},
                "environment": {"os": "test", "cpu": "test", "compiler": "test"},
            }
            with self.assertRaisesRegex(subject.ContractError, "referee hash differs"):
                subject._validate_tournament_provenance(
                    provenance,
                    ["alpha"],
                    referee,
                    root,
                    require_current_sources=False,
                    require_referee_hash=True,
                )

    def test_cross_platform_replay_referee_may_have_a_different_hash(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee"
            make_executable(referee, "pass\n")
            provenance = {
                "sourceCommit": "a" * 40,
                "sourceTreeSha256": "b" * 64,
                "sourceTreeDirty": False,
                "contractSourceSha256": "c" * 64,
                "referee": {
                    "schema": subject.MATCH_SCHEMA,
                    "version": "v1",
                    "sha256": "d" * 64,
                },
                "executables": {"alpha": "e" * 64},
                "environment": {"os": "test", "cpu": "test", "compiler": "test"},
            }
            subject._validate_tournament_provenance(
                provenance,
                ["alpha"],
                referee,
                root,
                require_current_sources=False,
                require_referee_hash=False,
            )

    def test_provenance_rejects_incomplete_executable_hash_map(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            referee = root / "referee"
            make_executable(referee, "pass\n")
            provenance = {
                "sourceCommit": "a" * 40,
                "sourceTreeSha256": "b" * 64,
                "sourceTreeDirty": False,
                "contractSourceSha256": "c" * 64,
                "referee": {
                    "schema": subject.MATCH_SCHEMA,
                    "version": "v1",
                    "sha256": subject.sha256_file(referee),
                },
                "executables": {"alpha": "e" * 64},
                "environment": {"os": "test", "cpu": "test", "compiler": "test"},
            }
            with self.assertRaisesRegex(subject.ContractError, "exactly cover"):
                subject._validate_tournament_provenance(
                    provenance,
                    ["alpha", "beta"],
                    referee,
                    root,
                    require_current_sources=False,
                    require_referee_hash=True,
                )


class MatchSchemaTests(unittest.TestCase):
    def test_action_syntax_is_conditional_on_acceptance(self) -> None:
        schema = json.loads(
            (REPOSITORY / "benchmarks/codingame_leaderboard/match.schema.json").read_text(
                encoding="utf-8"
            )
        )
        action = schema["$defs"]["action"]
        self.assertNotIn("pattern", action["properties"]["action"])
        conditional = action["allOf"][0]
        self.assertEqual(conditional["if"]["properties"]["accepted"], {"const": True})
        self.assertEqual(conditional["then"]["properties"]["action"]["pattern"], "^[0-7]+$")
        self.assertEqual(
            conditional["else"]["properties"]["action"]["type"], ["string", "null"]
        )
        self.assertEqual(conditional["else"]["properties"]["moves"]["maxItems"], 0)

    def test_forfeit_classification_enums_match_the_frozen_python_contract(self) -> None:
        schema = json.loads(
            (REPOSITORY / "benchmarks/codingame_leaderboard/match.schema.json").read_text(
                encoding="utf-8"
            )
        )
        action = schema["$defs"]["action"]
        rejected = action["allOf"][0]["else"]["properties"]["failureClassification"]["enum"]
        outcome = schema["$defs"]["forfeit"]["properties"]["classification"]["enum"]
        self.assertEqual(set(rejected), subject.BOT_FORFEIT_CLASSIFICATIONS)
        self.assertEqual(set(outcome), subject.BOT_FORFEIT_CLASSIFICATIONS)


class SummaryTests(unittest.TestCase):
    def test_exact_web_contract_and_global(self) -> None:
        standings, head = subject.rate_games(
            [entrant("a"), entrant("b")],
            [{"playerOneId": "a", "playerTwoId": "b", "winnerId": "a", "forfeitId": None}],
        )
        artifact = {
            "id": "test",
            "generatedAtUtc": "2026-08-13T00:00:00Z",
            "contract": {
                "roster": {"entrants": [entrant("a"), entrant("b")]},
                "schedule": {
                    "games": 1,
                    "gamesPerEntrant": 1,
                    "playerOneGamesPerEntrant": 1,
                    "playerTwoGamesPerEntrant": 0,
                    "seed": subject.SCHEDULE_SEED,
                },
                "rules": {"label": "rules"},
                "rating": {"label": "score"},
            },
            "provenance": {
                "sourceCommit": "abc",
                "environment": {"os": "os", "cpu": "cpu", "compiler": "compiler"},
            },
            "standings": standings,
            "headToHead": head,
        }
        summary = subject.build_summary(artifact)
        self.assertEqual(
            set(summary), {"schema", "tournament", "standings", "headToHead"}
        )
        expected_tournament_fields = {
            "id", "generatedAtUtc", "entrantCount", "gameCount", "gamesPerEntrant",
            "playerOneGamesPerEntrant", "playerTwoGamesPerEntrant", "rulesLabel",
            "scoringLabel", "scheduleSeed", "sourceCommit", "environment", "rawResultsUrl",
        }
        self.assertEqual(set(summary["tournament"]), expected_tournament_fields)
        script = subject.render_snapshot(artifact).decode("utf-8")
        self.assertTrue(script.startswith("// Generated"))
        self.assertIn("globalThis.PAPERSOCCER_CODINGAME_LEADERBOARD_RESULTS = ", script)
        self.assertNotIn('"match"', script)


class TinyEndToEndTests(unittest.TestCase):
    @staticmethod
    def _schedule() -> list[dict]:
        games = []
        for block, (first, second) in enumerate(
            (("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")), start=1
        ):
            for leg, (player_one, player_two) in enumerate(
                ((first, second), (second, first)), start=1
            ):
                games.append({
                    "id": f"game-{len(games) + 1:04d}",
                    "blockId": f"block-{block:03d}",
                    "stage": "fixture-round-robin",
                    "stageIndex": 1,
                    "leg": leg,
                    "playerOneId": player_one,
                    "playerTwoId": player_two,
                })
        return games

    def test_three_bot_runner_checkpoint_resume_rating_publish_and_check(self) -> None:
        """Exercise the real orchestration boundary with a deterministic fake referee."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repository = root / "repository"
            build_dir = root / "build"
            artifact_dir = root / "artifacts"
            roster_path = repository / "benchmarks/codingame_leaderboard/roster.json"
            referee = build_dir / "fake-referee"
            checkpoint_path = artifact_dir / "checkpoint.json"
            output_path = artifact_dir / "tournament.json"
            snapshot_path = artifact_dir / "leaderboard-results.js"
            invocation_log = root / "referee-invocations.jsonl"
            roster_path.parent.mkdir(parents=True)
            build_dir.mkdir()

            registered_ids = ["alpha", "beta", "gamma", "selfplay_nn_v2"]
            (repository / "CMakeLists.txt").write_text(
                "set(PAPERSOCCER_CODINGAME_BOTS\n  "
                + "\n  ".join(registered_ids)
                + "\n)\n",
                encoding="utf-8",
            )
            submission_hashes = {}
            for bot_id in registered_ids:
                bot_root = repository / f"submissions/codingame/bots/{bot_id}"
                bot_root.mkdir(parents=True)
                source = (
                    "// reviewed alpha fixture\n"
                    if bot_id in {"alpha", "selfplay_nn_v2"}
                    else f"// reviewed {bot_id} fixture\n"
                )
                submission = bot_root / "submission.cpp"
                submission.write_text(source, encoding="utf-8")
                submission_hashes[bot_id] = subject.sha256_file(submission)

            entrants = []
            for bot_id in ("alpha", "beta", "gamma"):
                aliases = []
                if bot_id == "alpha":
                    aliases.append({
                        "id": "selfplay_nn_v2",
                        "submissionPath": (
                            "submissions/codingame/bots/selfplay_nn_v2/submission.cpp"
                        ),
                        "submissionSha256": submission_hashes["selfplay_nn_v2"],
                    })
                entrants.append({
                    "id": bot_id,
                    "displayName": bot_id.title(),
                    "submissionPath": f"submissions/codingame/bots/{bot_id}/submission.cpp",
                    "submissionSha256": submission_hashes[bot_id],
                    "documentationUrl": f"https://example.test/{bot_id}",
                    "executableTarget": subject._expected_executable_name(bot_id),
                    "aliases": aliases,
                })
            roster_path.write_text(
                json.dumps({
                    "schema": subject.ROSTER_SCHEMA,
                    "cmakeRegistry": {
                        "path": "CMakeLists.txt",
                        "variable": "PAPERSOCCER_CODINGAME_BOTS",
                    },
                    "entrants": entrants,
                }),
                encoding="utf-8",
            )

            for entrant_data in entrants:
                make_executable(build_dir / entrant_data["executableTarget"], "pass\n")

            make_executable(
                referee,
                f'''
                import json, os, sys

                args = sys.argv[1:]
                replay = "--validate-transcript" in args
                with open({str(invocation_log)!r}, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{
                        "mode": "replay" if replay else "match",
                        "cwd": os.getcwd(),
                        "secretPresent": "PAPERSOCCER_TEST_SECRET" in os.environ,
                    }}) + "\\n")

                if "PAPERSOCCER_TEST_SECRET" in os.environ:
                    raise SystemExit(91)

                if replay:
                    transcript = args[args.index("--validate-transcript") + 1]
                    expected = None
                    if "--expected-winner" in args:
                        expected = int(args[args.index("--expected-winner") + 1])
                    print(json.dumps({{
                        "schema": "papersoccer.codingame-transcript-validation.v1",
                        "terminal": expected is not None,
                        "winnerPlayer": expected,
                        "terminalReason": "goal" if expected is not None else None,
                        "acceptedActionCount": 0 if not transcript else len(transcript.split("/")),
                        "edgeCount": len(transcript.replace("/", "")),
                    }}))
                    raise SystemExit(0)

                first_id = args[args.index("--player-one-id") + 1]
                second_id = args[args.index("--player-two-id") + 1]
                assert args[args.index("--first-timeout-ms") + 1] == "1000"
                assert args[args.index("--later-timeout-ms") + 1] == "200"
                first_target = "papersoccer_codingame_" + first_id + "_submission"
                second_target = "papersoccer_codingame_" + second_id + "_submission"
                is_forfeit = first_id == "gamma" and second_id == "alpha"
                winner = "alpha" if "alpha" in (first_id, second_id) else "gamma"
                loser = second_id if winner == first_id else first_id
                winner_player = 0 if winner == first_id else 1
                if is_forfeit:
                    actions = [{{
                        "turn": 0, "botId": first_id, "player": 0,
                        "opponentAction": "-", "action": None, "accepted": False,
                        "durationMicros": 1000000, "deadlineMillis": 1000,
                        "failureClassification": "timeout", "moves": [],
                    }}]
                    outcome = {{
                        "winnerId": winner, "loserId": loser, "reason": "forfeit",
                        "forfeit": {{"botId": loser, "classification": "timeout",
                                    "detail": "deterministic fixture timeout"}},
                    }}
                    player_one_timing = {{"decisions": 1, "totalMicros": 1000000,
                                         "maxMicros": 1000000}}
                else:
                    actions = [{{
                        "turn": 0, "botId": first_id, "player": 0,
                        "opponentAction": "-", "action": "0", "accepted": True,
                        "durationMicros": 1, "deadlineMillis": 1000,
                        "failureClassification": None,
                        "moves": [{{
                            "direction": 0, "from": {{"x": 4, "y": 6}},
                            "to": {{"x": 4, "y": 5}}, "extraTurn": False,
                            "statusAfter": "player_" + str(winner_player) + "_wins",
                        }}],
                    }}]
                    outcome = {{
                        "winnerId": winner, "loserId": loser, "reason": "goal",
                        "forfeit": None,
                    }}
                    player_one_timing = {{"decisions": 1, "totalMicros": 1, "maxMicros": 1}}

                print(json.dumps({{
                    "schema": "papersoccer.codingame-match.v1",
                    "participants": {{
                        "playerOne": {{"id": first_id, "player": 0,
                                      "executable": first_target}},
                        "playerTwo": {{"id": second_id, "player": 1,
                                      "executable": second_target}},
                    }},
                    "rules": {{"width": 8, "height": 10,
                              "goalRule": "OwnGoalsAllowed", "blockedRule": "MoverLoses"}},
                    "timeouts": {{"firstMillis": 1000, "laterMillis": 200}},
                    "actions": actions,
                    "outcome": outcome,
                    "timings": {{
                        "totalMicros": player_one_timing["totalMicros"],
                        "playerOne": player_one_timing,
                        "playerTwo": {{"decisions": 0, "totalMicros": 0, "maxMicros": 0}},
                    }},
                    "provenance": {{"refereeVersion": "fixture-v1"}},
                }}))
                ''',
            )

            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git", "-C", str(repository), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture",
                ],
                check=True,
            )

            schedule = self._schedule()
            original_validate_schedule = subject.validate_schedule
            original_contract = subject._contract

            def validate_fixture_schedule(games, ids, *, full_contract=True):
                self.assertEqual(list(games), schedule)
                self.assertEqual(set(ids), {"alpha", "beta", "gamma"})
                original_validate_schedule(games, ids, full_contract=False)

            def fixture_contract(roster, path, games):
                contract = original_contract(roster, path, games)
                contract["schedule"].update({
                    "games": 6,
                    "gamesPerEntrant": 4,
                    "playerOneGamesPerEntrant": 2,
                    "playerTwoGamesPerEntrant": 2,
                })
                return contract

            patches = (
                mock.patch.multiple(
                    subject,
                    EXPECTED_ENTRANTS=3,
                    EXPECTED_REGISTERED_BOTS=4,
                    EXPECTED_GAMES=6,
                ),
                mock.patch.object(subject, "build_schedule", return_value=schedule),
                mock.patch.object(
                    subject, "validate_schedule", side_effect=validate_fixture_schedule
                ),
                mock.patch.object(subject, "_contract", side_effect=fixture_contract),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                os.environ["PAPERSOCCER_TEST_SECRET"] = "must-not-leak"
                try:
                    first_result = subject.run_tournament(
                        repository=repository,
                        roster_path=roster_path,
                        referee=referee,
                        build_dir=build_dir,
                        output_path=output_path,
                        checkpoint_path=checkpoint_path,
                        stop_after=2,
                        generated_at_utc="2026-08-13T12:34:56Z",
                        infrastructure_timeout_seconds=2.0,
                    )
                    self.assertIsNone(first_result)
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                    self.assertEqual(len(checkpoint["games"]), 2)
                    self.assertFalse(output_path.exists())

                    alpha_executable = build_dir / subject._expected_executable_name("alpha")
                    original_executable = alpha_executable.read_bytes()
                    alpha_executable.write_bytes(original_executable + b"# changed\n")
                    with self.assertRaisesRegex(subject.ContractError, "fingerprint differs"):
                        subject.run_tournament(
                            repository=repository,
                            roster_path=roster_path,
                            referee=referee,
                            build_dir=build_dir,
                            output_path=output_path,
                            checkpoint_path=checkpoint_path,
                            resume=True,
                            infrastructure_timeout_seconds=2.0,
                        )
                    alpha_executable.write_bytes(original_executable)
                    alpha_executable.chmod(alpha_executable.stat().st_mode | stat.S_IXUSR)

                    artifact = subject.run_tournament(
                        repository=repository,
                        roster_path=roster_path,
                        referee=referee,
                        build_dir=build_dir,
                        output_path=output_path,
                        checkpoint_path=checkpoint_path,
                        resume=True,
                        infrastructure_timeout_seconds=2.0,
                    )
                finally:
                    del os.environ["PAPERSOCCER_TEST_SECRET"]

                self.assertIsNotNone(artifact)
                self.assertEqual(len(artifact["games"]), 6)
                self.assertEqual(
                    [row["id"] for row in artifact["standings"]],
                    ["alpha", "gamma", "beta"],
                )
                gamma = next(row for row in artifact["standings"] if row["id"] == "gamma")
                self.assertEqual(gamma["forfeits"], 1)
                self.assertEqual(artifact["contract"]["schedule"]["gamesPerEntrant"], 4)
                self.assertEqual(
                    json.loads(checkpoint_path.read_text(encoding="utf-8"))["games"],
                    artifact["games"],
                )

                subject.publish_snapshot(
                    output_path,
                    snapshot_path,
                    repository=repository,
                    roster_path=roster_path,
                    referee=referee,
                )
                subject.publish_snapshot(
                    output_path,
                    snapshot_path,
                    repository=repository,
                    roster_path=roster_path,
                    referee=referee,
                    check=True,
                )
                snapshot = snapshot_path.read_text(encoding="utf-8")
                self.assertIn(subject.SUMMARY_SCHEMA, snapshot)
                self.assertNotIn('"actions"', snapshot)
                snapshot_path.write_text(snapshot + "// stale\n", encoding="utf-8")
                with self.assertRaisesRegex(subject.ContractError, "snapshot is stale"):
                    subject.publish_snapshot(
                        output_path,
                        snapshot_path,
                        repository=repository,
                        roster_path=roster_path,
                        referee=referee,
                        check=True,
                    )

            invocations = [
                json.loads(line)
                for line in invocation_log.read_text(encoding="utf-8").splitlines()
            ]
            matches = [invocation for invocation in invocations if invocation["mode"] == "match"]
            replays = [invocation for invocation in invocations if invocation["mode"] == "replay"]
            self.assertEqual(len(matches), 6)
            self.assertGreaterEqual(len(replays), 18)
            self.assertTrue(all(not invocation["secretPresent"] for invocation in invocations))
            self.assertTrue(all("papersoccer-leaderboard-" in item["cwd"] for item in matches))


if __name__ == "__main__":
    unittest.main()
