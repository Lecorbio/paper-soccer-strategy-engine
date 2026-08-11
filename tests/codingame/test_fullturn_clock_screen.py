#!/usr/bin/env python3

import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "rank_4_fullturn_bfm"
    / "run_clock_screen.py"
)
SPEC = importlib.util.spec_from_file_location("fullturn_clock_screen", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FullTurnClockScreenTest(unittest.TestCase):
    def test_parses_one_all_games_summary(self):
        fields = MODULE.parse_summary(
            "opening=x\n"
            "summary batch=all games=24 candidate=13 reference=11 "
            "unfinished=0 candidate_operational_timeouts=0 "
            "reference_operational_timeouts=0 candidate_max_first_ms=810.5 "
            "candidate_max_later_ms=169.5 reference_max_first_ms=805.0 "
            "reference_max_later_ms=168.0\n"
        )
        self.assertEqual(fields["candidate"], "13")
        self.assertEqual(fields["games"], "24")

    def test_safety_evaluation_is_clock_and_operation_based(self):
        fields = {
            "games": "24",
            "unfinished": "0",
            "candidate_operational_timeouts": "0",
            "reference_operational_timeouts": "0",
            "candidate_max_first_ms": "899.0",
            "candidate_max_later_ms": "179.0",
            "reference_max_first_ms": "850.0",
            "reference_max_later_ms": "170.0",
        }
        self.assertEqual(
            MODULE.evaluate_summary(
                fields,
                expected_games=24,
                first_headroom_ms=900.0,
                later_headroom_ms=180.0,
            ),
            [],
        )
        fields["candidate_max_later_ms"] = "180.0"
        self.assertIn(
            "candidate exceeded later-response headroom",
            MODULE.evaluate_summary(
                fields,
                expected_games=24,
                first_headroom_ms=900.0,
                later_headroom_ms=180.0,
            ),
        )
        fields["candidate_max_later_ms"] = "nan"
        with self.assertRaises(ValueError):
            MODULE.evaluate_summary(
                fields,
                expected_games=24,
                first_headroom_ms=900.0,
                later_headroom_ms=180.0,
            )

    def test_opening_turns_require_unique_nonnegative_values(self):
        self.assertEqual(MODULE.parse_opening_turns("0,1,2,7"), (0, 1, 2, 7))
        with self.assertRaises(Exception):
            MODULE.parse_opening_turns("0,1,1")
        with self.assertRaises(Exception):
            MODULE.parse_opening_turns("0,-1")

    def test_command_records_search_hypothesis_parameters(self):
        arguments = types.SimpleNamespace(
            gate=pathlib.Path("gate"),
            pairs_per_depth=1,
            candidate_work=3_000_000,
            reference_nodes=3_000_000,
            max_turns=200,
            batch_start=0,
            batch_count=1,
            opening_turns=(0, 1),
            first_ms=800,
            later_ms=165,
            candidate_max_actions=64,
            candidate_nonroot_actions=32,
            candidate_exploration=0.125,
            candidate_fpu=-0.25,
            candidate_final_visit_weight=0.0,
            candidate_replay_blend=100,
            candidate_residual_weight=0,
            candidate_root_only=True,
        )
        command = MODULE.build_command(arguments)
        self.assertEqual(
            command[command.index("--candidate-max-actions") + 1], "64"
        )
        self.assertEqual(
            command[command.index("--candidate-nonroot-actions") + 1], "32"
        )
        self.assertEqual(
            command[command.index("--candidate-exploration") + 1], "0.125"
        )
        self.assertEqual(command[command.index("--candidate-fpu") + 1], "-0.25")
        self.assertEqual(
            command[command.index("--candidate-final-visit-weight") + 1], "0.0"
        )
        self.assertEqual(
            command[command.index("--candidate-replay-blend") + 1], "100"
        )
        self.assertEqual(
            command[command.index("--candidate-residual-weight") + 1], "0"
        )
        self.assertEqual(
            command[command.index("--candidate-root-only") + 1], "1"
        )


if __name__ == "__main__":
    unittest.main()
