"""Regression checks for sparse paired-window inference and all own failures."""
import unittest
from tools import rank_two_live_eight_assessment_v2 as assessment
from tools import rank_two_live_eight_v1 as retained

PRIMARY = [dict(user_id=i) for i in range(1, 7)]


def window(index, arm, missing=()):
    games = []
    for opponent in range(1, 7):
        for color in (0, 1):
            if (opponent, color) in missing:
                continue
            for _ in range(7):
                games.append(dict(game_id=index*1000+len(games), won=arm=="C", color=color,
                    opponent_user_id=opponent, opponent_submission_id=100+opponent,
                    clean=True, own_failure=False, opponent_failure=False))
    while len(games) < 90:
        games.append(dict(game_id=index*1000+len(games), won=arm=="C", color=0,
            opponent_user_id=999, opponent_submission_id=999,
            clean=True, own_failure=False, opponent_failure=False))
    return dict(complete=True, arm=arm, games=games)


class LiveAssessmentValidityTests(unittest.TestCase):
    def test_complementary_missing_cells_never_report_conditional_interval(self):
        windows = [window(1,"B",((1,0),)), window(2,"C"),
                   window(3,"C",((2,1),)), window(4,"B")]
        old = retained.score_windows(windows, PRIMARY, 512, 11)
        self.assertTrue(old["sufficient"])
        self.assertLess(old["eligible_bootstrap_repetitions"], 512)
        self.assertIsNotNone(old["interval95"])
        new = assessment.score_windows(windows, PRIMARY, 512, 11)
        self.assertEqual(new["point"], old["point"])
        self.assertIsNone(new["interval95"])
        self.assertFalse(new["confidence_interval_usable"])
        self.assertFalse(new["live_gate_passed"])

    def test_complete_support_preserves_full_draw_inference(self):
        windows = [window(1,"B"),window(2,"C"),window(3,"C"),window(4,"B")]
        old = retained.score_windows(windows, PRIMARY, 512, 11)
        new = assessment.score_windows(windows, PRIMARY, 512, 11)
        self.assertEqual(new["point"], old["point"])
        self.assertEqual(new["interval95"], old["interval95"])
        self.assertEqual(new["eligible_bootstrap_repetitions"], 512)
        self.assertTrue(new["confidence_interval_usable"])
        self.assertTrue(new["live_gate_passed"])

    def test_supplemental_own_failure_blocks_otherwise_passing_gate(self):
        windows = [window(1,"B"),window(2,"C"),window(3,"C"),window(4,"B")]
        windows[1]["supplemental_games"] = [dict(game_id=99999,own_failure=True)]
        result = assessment.score_windows(windows, PRIMARY, 512, 11)
        self.assertEqual(result["own_failures"], 1)
        self.assertFalse(result["live_gate_passed"])

    def test_duplicate_games_remain_rejected(self):
        windows = [window(1,"B"),window(2,"C"),window(1,"C"),window(4,"B")]
        with self.assertRaisesRegex(ValueError,"duplicate attested game"):
            assessment.score_windows(windows, PRIMARY, 512, 11)


if __name__ == "__main__":
    unittest.main()
