import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "submissions/codingame/bots/rank_4_jacek_hybrid/replay_tactical_audit.cpp"
)
BOOK = ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp"


class ReplayTacticalAuditContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = AUDIT.read_text(encoding="ascii")
        cls.book = BOOK.read_text(encoding="ascii")

    def test_tracked_replay_registry_has_exact_activation_cardinality(self):
        rows = re.findall(
            r'\{([01]),\s*(\d+),\s*"([0-7/]+)"\}', self.book
        )
        self.assertEqual(len(rows), 24)
        eligible = []
        for player, first_turn, transcript in rows:
            actions = transcript.split("/")
            eligible.extend(
                (int(player), "/".join(actions[:turn]), actions[turn])
                for turn in range(int(first_turn), len(actions))
                if turn % 2 == int(player)
            )
        self.assertEqual(len(eligible), 512)
        unique = {(player, prefix): action for player, prefix, action in eligible}
        self.assertEqual(len(unique), 511)
        for player, prefix, action in eligible:
            self.assertEqual(unique[(player, prefix)], action)

    def test_five_level_ordering_is_literal_and_mover_relative(self):
        expected = {
            "ImmediateLoss": "-2",
            "OpponentComponentWin": "-1",
            "OpponentComponentUnknown": "0",
            "OpponentComponentLoss": "1",
            "ImmediateWin": "2",
        }
        for name, value in expected.items():
            self.assertRegex(
                self.audit,
                rf"{name}\s*=\s*{re.escape(value)}",
            )
        self.assertIn("*winning_player == mover", self.audit)
        self.assertIn("after.to_move == ps::opponent(mover)", self.audit)

    def test_caps_and_displacement_fail_closed(self):
        self.assertIn("constexpr std::uint64_t kDefaultStateCap = 2'000'000", self.audit)
        self.assertIn("constexpr std::uint64_t kDefaultGlobalStateCap = 100'000'000", self.audit)
        self.assertRegex(self.audit, r"if \(capped != 0\) \{\s*return 2;")
        self.assertRegex(self.audit, r"if \(displaced != 0\) \{\s*return 1;")
        self.assertIn("correction_category != maximum", self.audit)

    def test_audit_includes_only_the_hybrid_engine_and_replay_book(self):
        self.assertIn('#include "bot.cpp"', self.audit)
        forbidden = (
            "validation_d",
            "final_d",
            "protected",
            "matches.json",
            "jacek_native_bfm",
            "rank_4/replay_book",
        )
        for token in forbidden:
            self.assertNotIn(token, self.audit)


if __name__ == "__main__":
    unittest.main()
