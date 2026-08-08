import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "submissions/codingame/tools/promotion_gate.py"
SPEC = importlib.util.spec_from_file_location("promotion_gate", MODULE_PATH)
promotion_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion_gate)


class PromotionStatisticsTest(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertEqual(promotion_gate.quantile([0.0, 1.0], 0.5), 0.5)

    def test_bootstrap_resamples_source_games_and_is_deterministic(self):
        pairs = [
            {"stratum": "a", "candidate_pair_score": 0.5},
            {"stratum": "a", "candidate_pair_score": 0.5},
            {"stratum": "b", "candidate_pair_score": 0.5},
        ]
        for index, pair in enumerate(pairs):
            pair["source_game_id"] = index
            pair["opening_id"] = f"opening-{index}"
        first = promotion_gate.source_game_cluster_bootstrap(pairs, 100, 7)
        second = promotion_gate.source_game_cluster_bootstrap(pairs, 100, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["lower"], 0.5)
        self.assertEqual(first["upper"], 0.5)

    def test_bootstrap_does_not_treat_positions_from_one_game_as_independent(self):
        pairs = [
            {
                "opening_id": f"opening-{index}",
                "source_game_id": 11 if index < 3 else 22,
                "candidate_pair_score": 1.0 if index < 3 else 0.0,
            }
            for index in range(4)
        ]
        result = promotion_gate.source_game_cluster_bootstrap(pairs, 1000, 19)
        self.assertEqual(result["opening_pairs"], 4)
        self.assertEqual(result["source_game_clusters"], 2)
        self.assertEqual(result["estimate"], 0.5)

    def test_independent_replay_rejects_an_overlong_complete_turn(self):
        row = {
            "transcript": "00",
            "winner_player_id": "0",
        }
        with self.assertRaises(promotion_gate.UsageError):
            promotion_gate.independently_reconstruct_bank_state(row)

    def test_multi_budget_profiles_require_every_budget(self):
        identity = {"candidate": "candidate", "runner": "runner"}
        passing = {
            "node_budget": 30000,
            "passed": True,
            "requirements": [{"id": "score", "passed": True}],
        }
        failing = {
            "node_budget": 100000,
            "passed": False,
            "requirements": [{"id": "score", "passed": False}],
        }
        report = promotion_gate.combine_budget_profiles(
            "test", identity, [passing, failing]
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["reason_codes"], ["nodes_100000:score"])

    def test_timing_report_recomputes_all_cases_and_limits(self):
        config = {
            "fresh_process_samples": 1,
            "shell_cases": ["shell"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        }
        samples = [
            {"sample": 0, "case": "initial-player-0", "player": 0,
             "first_ms": 650.0, "later_ms": 130.0},
            {"sample": 0, "case": "initial-player-1", "player": 1,
             "first_ms": 651.0, "later_ms": 131.0},
            {"sample": 0, "case": "shell", "later_ms": 132.0},
        ]
        passing = promotion_gate.make_timing_report(
            "bot", "candidate", "manifest", "timing", samples, config
        )
        self.assertTrue(passing["passed"])
        samples[-1]["later_ms"] = 205.0
        failing = promotion_gate.make_timing_report(
            "bot", "candidate", "manifest", "timing", samples, config
        )
        self.assertFalse(failing["passed"])
        self.assertIn("later_max", failing["reason_codes"])

    def test_neutral_completed_control_is_explicitly_rejected(self):
        identity = {"candidate": "same", "incumbent": "same"}
        game_zero = {"candidate_player": 0, "winner": 0}
        game_one = {"candidate_player": 1, "winner": 0}
        pairs = [
            {
                "opening_id": f"opening-{index}",
                "source_game_id": index + 1,
                "stratum": "shell",
                "candidate_pair_score": 0.5,
                "games": [game_zero, game_one],
            }
            for index in range(4)
        ]
        manifest = {
            "banks": {"development.tsv": {"records": 4}},
            "stages": {
                "development": {
                    "bank": "development.tsv",
                    "minimum_mean": 0.52,
                    "require_more_wins_than_incumbent": True,
                }
            },
            "statistics": {"resamples": 100, "seed": 17},
        }
        shard = {
            "schema": "papersoccer.codingame-promotion-shard.v1",
            "identity": identity,
            "pairs": pairs,
            "operational": {field: 0 for field in promotion_gate.OPERATIONAL_FIELDS},
        }
        report = promotion_gate.aggregate_stage(
            manifest, "development", [shard], identity
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["confidence_interval"]["lower"], 0.5)
        self.assertIn("cluster_mean_score", report["reason_codes"])
        self.assertIn("candidate_game_wins", report["reason_codes"])


if __name__ == "__main__":
    unittest.main()
