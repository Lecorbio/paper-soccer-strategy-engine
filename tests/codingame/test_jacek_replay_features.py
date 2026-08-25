import hashlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_features as features  # noqa: E402


class JacekReplayFeatureTests(unittest.TestCase):
    @staticmethod
    def active_sha256(active):
        packed = b"".join(index.to_bytes(2, "little") for index in active)
        return hashlib.sha256(packed).hexdigest()

    def test_frozen_dimensions_and_initial_features(self):
        self.assertEqual(features.INPUT_COUNT, 6301)
        self.assertEqual(len(features.POINTS), 105)
        self.assertEqual(len(features.EDGES), 316)
        active = features.encode_active(features.ReplayState())
        self.assertEqual(len(active), 105)
        self.assertEqual(features.validate_active(active), active)
        self.assertEqual(
            self.active_sha256(active),
            "1ee1c7cf987fb1ca8d2bc707c4511b93fdf96216e6cf2a7917a18da6a2bec4d4",
        )

    def test_vertex_category_contract(self):
        self.assertEqual(features.vertex_category(0, 0), 0)
        self.assertEqual(features.vertex_category(0, 8), 7)
        self.assertEqual(features.vertex_category(6, 8), 55)
        self.assertEqual(features.vertex_category(7, 8), 56)
        self.assertEqual(features.vertex_category(1_000_000, 0), 56)

    def test_rotation_and_reflection_are_involutions(self):
        state = features.ReplayState()
        features.apply_complete_turn(state, 0, "0")
        active = features.encode_active(state)
        self.assertEqual(
            self.active_sha256(active),
            "4b39d1752979d5c819cd8d4c4a022f56a0a0eb3227bd658c9fadf190dedf719f",
        )
        self.assertEqual(
            features.reflect_active(features.reflect_active(active)), active
        )
        self.assertEqual(
            features.rotate_active(features.rotate_active(active)), active
        )
        self.assertEqual(
            features.encode_active(state, reflected=True),
            features.reflect_active(active),
        )

    def test_horizontal_reflection_matches_mirrored_game_geometry(self):
        """Reflection is x -> WIDTH-x, including directions and used edges."""

        original = features.ReplayState()
        mirrored = features.ReplayState()
        direction_reflection = str.maketrans("01234567", "07654321")
        canonical_segment = lambda first, second: tuple(  # noqa: E731
            sorted((first, second), key=lambda point: (point[1], point[0]))
        )
        prefix = ("0", "0", "3", "0", "61", "0")
        for action in prefix:
            mover = original.to_move
            self.assertEqual(mirrored.to_move, mover)
            features.apply_complete_turn(original, mover, action)
            features.apply_complete_turn(
                mirrored, mover, action.translate(direction_reflection)
            )
            self.assertEqual(
                mirrored.ball, features.reflect_point(original.ball)
            )
            self.assertEqual(
                mirrored.used_segments,
                {
                    canonical_segment(
                        features.reflect_point(first),
                        features.reflect_point(second),
                    )
                    for first, second in original.used_segments
                },
            )
            self.assertEqual(
                mirrored.visit_count,
                {
                    features.reflect_point(point): count
                    for point, count in original.visit_count.items()
                },
            )
            self.assertEqual(
                features.encode_active(mirrored),
                features.reflect_active(features.encode_active(original)),
            )

    def test_player_two_is_canonicalized_by_rotation(self):
        player_two_state = features.ReplayState()
        features.apply_complete_turn(player_two_state, 0, "0")
        self.assertEqual(player_two_state.to_move, 1)
        active = features.encode_active(player_two_state)
        self.assertEqual(features.validate_active(active), active)
        self.assertIn(features.ROTATED_EDGES[features.EDGE_INDEX[((4, 5), (4, 6))]], active)

    def test_complete_turn_validation_rejects_early_end(self):
        state = features.ReplayState(
            ball=(1, 5), to_move=0, visit_count={(4, 6): 1, (1, 5): 1}
        )
        with self.assertRaisesRegex(ValueError, "before rebound chain"):
            features.apply_complete_turn(state, 0, "6")

    def test_known_short_game_replays_to_player_one_goal(self):
        turns = (
            (0, "0"),
            (1, "0"),
            (0, "3"),
            (1, "0"),
            (0, "61"),
            (1, "0"),
            (0, "07"),
        )
        state = features.ReplayState()
        for player, action in turns:
            features.apply_complete_turn(state, player, action)
        self.assertEqual(state.winner, 0)


if __name__ == "__main__":
    unittest.main()
