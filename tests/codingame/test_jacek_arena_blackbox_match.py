import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
SPEC = importlib.util.spec_from_file_location(
    "jacek_arena_blackbox_match", BOT / "blackbox_match_gate.py"
)
match_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = match_gate
SPEC.loader.exec_module(match_gate)


class RulesTests(unittest.TestCase):
    def test_exact_public_topology(self):
        self.assertEqual(len(match_gate.TOPOLOGY.coordinates), 105)
        self.assertEqual(len(match_gate.TOPOLOGY.edges), 316)
        self.assertEqual(
            match_gate.TOPOLOGY.coordinates[match_gate.TOPOLOGY.initial_ball],
            (4, 6),
        )

    def test_ordinary_move_changes_turn(self):
        state = match_gate.initial_state()
        match_gate.apply_complete_action(state, "0")
        self.assertEqual(match_gate.TOPOLOGY.coordinates[state.ball], (4, 5))
        self.assertEqual(state.to_move, 1)
        self.assertIsNone(state.winner)

    def test_boundary_rebound_must_be_completed(self):
        state = match_gate.State(ball=match_gate.TOPOLOGY.vertex(1, 6))
        before = state.clone()
        with self.assertRaisesRegex(
            match_gate.RuleViolation, "mandatory-rebound-omitted"
        ):
            match_gate.apply_complete_action(state, "6")
        self.assertEqual(state, before)
        match_gate.apply_complete_action(state, "61")
        self.assertEqual(match_gate.TOPOLOGY.coordinates[state.ball], (1, 5))
        self.assertEqual(state.to_move, 1)

    def test_exact_goal_and_own_goal(self):
        attack = match_gate.State(
            ball=match_gate.TOPOLOGY.vertex(4, 1), to_move=0
        )
        match_gate.apply_complete_action(attack, "0")
        self.assertEqual(attack.winner, 0)

        own_goal = match_gate.State(
            ball=match_gate.TOPOLOGY.vertex(4, 11), to_move=0
        )
        match_gate.apply_complete_action(own_goal, "4")
        self.assertEqual(own_goal.winner, 1)

    def test_blocked_destination_loses_for_mover(self):
        source = match_gate.TOPOLOGY.vertex(4, 6)
        destination = match_gate.TOPOLOGY.vertex(4, 5)
        incoming = match_gate.TOPOLOGY.edge(source, destination)
        used = set(match_gate.TOPOLOGY.incident_edges[destination]) - {incoming}
        state = match_gate.State(used=used, ball=source, to_move=0)
        match_gate.apply_complete_action(state, "0")
        self.assertEqual(state.winner, 1)

    def test_overlong_action_is_atomic(self):
        state = match_gate.initial_state()
        before = state.clone()
        with self.assertRaisesRegex(
            match_gate.RuleViolation, "overlong-complete-turn"
        ):
            match_gate.apply_complete_action(state, "00")
        self.assertEqual(state, before)


class FakeBot:
    def __init__(self, executable, label, action="0"):
        self.executable = executable
        self.label = label
        self.action = action
        self.player = None
        self.requests = []
        self.closed = False

    def initialize(self, player):
        self.player = player

    def request(self, previous_action, timeout_ms):
        self.requests.append((previous_action, timeout_ms))
        return self.action

    def close(self):
        self.closed = True


class RefereeTests(unittest.TestCase):
    def test_protocol_uses_required_two_line_turn_framing(self):
        bot = object.__new__(match_gate.ProtocolBot)
        writes = []
        bot._write = writes.append
        bot._read_line = lambda timeout_ms: b"0"
        self.assertEqual(bot.request(None, 100.0), "0")
        self.assertEqual(bot.request("123", 100.0), "0")
        self.assertEqual(writes, [b"0\n-\n", b"3\n123\n"])

    def test_complete_black_box_protocol_game(self):
        bots = []

        def factory(executable, label):
            bot = FakeBot(executable, label)
            bots.append(bot)
            return bot

        result = match_gate.play_game(
            0,
            candidate_executable=pathlib.Path("candidate"),
            h62_executable=pathlib.Path("h62"),
            candidate_player=0,
            first_timeout_ms=100.0,
            later_timeout_ms=100.0,
            process_factory=factory,
        )
        self.assertTrue(result["clean"])
        self.assertEqual(result["winner_player"], 0)
        self.assertEqual(result["candidate_result"], "win")
        self.assertEqual(result["physical_edges"], 6)
        self.assertEqual(result["decisions_by_player"], [3, 3])
        self.assertEqual(bots[0].requests[0][0], None)
        self.assertEqual(bots[1].requests[0][0], "0")
        self.assertTrue(all(bot.closed for bot in bots))

    def test_invalid_candidate_output_is_an_operational_failure(self):
        def factory(executable, label):
            return FakeBot(executable, label, "x" if label == "candidate" else "0")

        result = match_gate.play_game(
            0,
            candidate_executable=pathlib.Path("candidate"),
            h62_executable=pathlib.Path("h62"),
            candidate_player=0,
            first_timeout_ms=100.0,
            later_timeout_ms=100.0,
            process_factory=factory,
        )
        self.assertFalse(result["clean"])
        self.assertEqual(result["candidate_result"], "operational-loss")
        self.assertEqual(result["failure"]["side"], "candidate")
        self.assertEqual(result["failure"]["category"], "invalid-output")

    def test_report_retains_no_actions_or_opponent_streams(self):
        def factory(executable, label):
            return FakeBot(executable, label)

        games = [
            match_gate.play_game(
                index,
                candidate_executable=pathlib.Path("candidate"),
                h62_executable=pathlib.Path("h62"),
                candidate_player=index,
                first_timeout_ms=100.0,
                later_timeout_ms=100.0,
                process_factory=factory,
            )
            for index in (0, 1)
        ]
        identity = {
            "bytes": 1,
            "sha256": "0" * 64,
            "source_content_emitted": False,
        }
        report = match_gate.summarize_matches(
            games,
            requested_games=2,
            profile="custom",
            profile_exact=True,
            workers=1,
            seed=0,
            first_timeout_ms=100.0,
            later_timeout_ms=100.0,
            candidate_source=identity,
            h62_source=identity,
            builds=[],
            compiler_identity={},
        )
        forbidden_keys = {
            "transcript", "stdout", "stderr", "opponent_action", "actions"
        }

        def assert_no_stream_content(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for child in value.values():
                    assert_no_stream_content(child)
            elif isinstance(value, list):
                for child in value:
                    assert_no_stream_content(child)

        assert_no_stream_content(report)
        match_gate._canonical_json(report).decode("ascii")
        self.assertEqual(report["results"]["candidate_wins"], 1)
        self.assertEqual(report["results"]["candidate_losses"], 1)
        self.assertFalse(report["gate"]["beats_h62"])

    def test_odd_game_count_is_rejected_before_process_launch(self):
        with self.assertRaisesRegex(match_gate.GateError, "even"):
            match_gate.run_games(
                games=3,
                workers=1,
                seed=0,
                candidate_executable=pathlib.Path("candidate"),
                h62_executable=pathlib.Path("h62"),
                first_timeout_ms=100.0,
                later_timeout_ms=100.0,
            )


if __name__ == "__main__":
    unittest.main()
