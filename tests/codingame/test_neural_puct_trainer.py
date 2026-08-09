import dataclasses
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    ROOT / "submissions/codingame/bots/neural_puct/train_neural_puct.py"
)


@unittest.skipIf(np is None, "neural trainer tests require NumPy")
class NeuralPuctTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "neural_puct_trainer_under_test", TRAINER_PATH
        )
        cls.trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.trainer
        spec.loader.exec_module(cls.trainer)

    def soft_record(self, seed, game, probabilities, turns=("0",)):
        targets = []
        for action in turns:
            targets.append(
                [
                    {
                        "probabilities": probabilities,
                        "total_visits": 4,
                        "fallback": False,
                    }
                    for _ in action
                ]
            )
        return {
            "schema": "papersoccer.selfplay.v1",
            "teacher": "rank_5",
            "seed": seed,
            "game": game,
            "winner": 0,
            "teacher_start_turn": 0,
            "turns": list(turns),
            "policy_target_schema": "canonical-primitive-root-visits-v1",
            "policy_targets": targets,
        }

    def test_selfplay_groups_exact_transcripts_before_split(self):
        selected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        alternate = [0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        records = [
            self.soft_record(11, 0, selected),
            self.soft_record(12, 1, selected),
            self.soft_record(13, 2, alternate),
            {
                "schema": "papersoccer.selfplay.v1",
                "teacher": "rank_5",
                "seed": 14,
                "game": 3,
                "winner": 1,
                "teacher_start_turn": 0,
                "turns": ["1"],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "selfplay.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            games, teachers, _, schemas = self.trainer.load_selfplay([path])

        self.assertEqual(teachers, ["rank_5"])
        self.assertEqual(schemas, ["canonical-primitive-root-visits-v1"])
        self.assertEqual(len(games), 3)
        grouped = [game for game in games if game.turns == ((0, "0"),)]
        self.assertEqual(len(grouped), 2)
        self.assertEqual({game.split_group for game in grouped}, {grouped[0].split_group})
        self.assertEqual(sorted(game.duplicate_count for game in grouped), [1, 2])
        splits = self.trainer.stratified_splits(games)
        self.assertEqual({splits[game.key] for game in grouped}, {splits[grouped[0].key]})
        report = self.trainer.selfplay_preprocessing_report(games)
        self.assertEqual(report["selfplay_raw_records"], 4)
        self.assertEqual(report["selfplay_retained_payload_records"], 3)
        self.assertEqual(report["selfplay_trajectory_groups"], 2)
        self.assertEqual(report["selfplay_exact_duplicate_records_collapsed"], 1)
        self.assertEqual(report["selfplay_trajectory_payload_conflict_groups"], 1)
        self.assertIsNone(next(game for game in games if game.turns == ((0, "1"),)).policy_targets)

    def test_preteacher_primitives_are_replayed_but_not_sampled(self):
        games, _ = self.trainer.load_public_games(
            [*self.trainer.DEFAULT_ELITE, self.trainer.DEFAULT_JACEK]
        )
        selected = None
        baseline = None
        prefix_turns = None
        for game in games:
            try:
                candidate = self.trainer.replay_game(game)
            except ValueError:
                continue
            for turns in range(1, min(10, len(game.turns))):
                if sum(len(action) for _, action in game.turns[:turns]) > turns:
                    selected = game
                    baseline = candidate
                    prefix_turns = turns
                    break
            if selected is not None:
                break
        self.assertIsNotNone(selected)
        prefix_primitives = sum(
            len(action) for _, action in selected.turns[:prefix_turns]
        )
        prefixed = dataclasses.replace(selected, policy_start_turn=prefix_turns)
        samples = self.trainer.replay_game(prefixed)
        self.assertEqual(len(samples), len(baseline) - prefix_primitives * 2)
        self.assertEqual(
            self.trainer.preteacher_primitive_count(prefixed), prefix_primitives
        )
        unchanged = self.trainer.replay_game(
            dataclasses.replace(selected, policy_start_turn=0)
        )
        self.assertEqual(len(unchanged), len(baseline))
        self.assertEqual(
            [sample.state_key for sample in unchanged],
            [sample.state_key for sample in baseline],
        )

    def sample(self, state_key, policy, value):
        target = np.zeros(8, dtype=np.float32)
        target[policy] = 1.0
        return self.trainer.Sample(
            features=np.zeros(1, dtype=np.float32),
            action_features=np.zeros(
                (8, self.trainer.ACTION_FEATURE_COUNT), dtype=np.float32
            ),
            legal=np.ones(8, dtype=bool),
            policy=policy,
            policy_target=target,
            value=value,
            has_policy=True,
            game_key=f"game-{state_key!r}-{policy}-{value}",
            focus_agent_id=-1,
            source="neural-selfplay",
            state_key=state_key,
        )

    def test_ordered_overlap_purge_keeps_within_split_conflicts(self):
        train = [self.sample(b"a", 0, 1.0), self.sample(b"a", 1, -1.0)]
        validation = [
            self.sample(b"a", 2, 1.0),
            self.sample(b"b", 0, 1.0),
            self.sample(b"b", 1, -1.0),
        ]
        test = [
            self.sample(b"a", 0, 1.0),
            self.sample(b"b", 0, 1.0),
            self.sample(b"c", 0, 1.0),
        ]
        buckets = {"train": train, "validation": validation, "test": test}
        removed = self.trainer.purge_held_out_overlaps(buckets)
        self.assertEqual(removed, {"validation": 1, "test": 2})
        self.assertEqual(buckets["train"], train)
        self.assertEqual([sample.state_key for sample in buckets["validation"]], [b"b", b"b"])
        self.assertEqual([sample.state_key for sample in buckets["test"]], [b"c"])

    def test_public_split_assignment_is_unchanged(self):
        games = [
            self.trainer.Game(
                key=f"public:{index}",
                game_id=index,
                source="elite",
                focus_agent_id=7,
                focus_player=0,
                winner=0,
                turns=((0, "0"),),
            )
            for index in range(20)
        ]
        expected = {}
        ordered = sorted(
            games,
            key=lambda game: (
                self.trainer.hashlib.sha256(game.key.encode()).digest(),
                game.key,
            ),
        )
        for index, game in enumerate(ordered):
            expected[game.key] = (
                "train" if index < 16 else "validation" if index < 18 else "test"
            )
        self.assertEqual(self.trainer.stratified_splits(games), expected)

    def test_live_whole_game_group_is_global_across_label_sources(self):
        games = []
        for index in range(10):
            for suffix, agent in (("expert", 100 + index), ("relabel", 999)):
                games.append(
                    self.trainer.Game(
                        key=f"live:{index}:{suffix}",
                        game_id=index,
                        source=suffix,
                        focus_agent_id=agent,
                        focus_player=0,
                        winner=0,
                        turns=((0, "0"),),
                        split_group=f"codingame-live:{index}",
                        split_scope="global",
                    )
                )
        splits = self.trainer.stratified_splits(games)
        for index in range(10):
            self.assertEqual(
                splits[f"live:{index}:expert"], splits[f"live:{index}:relabel"]
            )
        self.assertEqual(
            {split: list(splits.values()).count(split) for split in set(splits.values())},
            {"train": 16, "validation": 2, "test": 2},
        )

    def test_live_source_mass_is_frozen_and_unlabelled_values_stay_zero(self):
        samples = [
            self.sample(b"a", 0, 1.0),
            self.sample(b"b", 1, -1.0),
            self.sample(b"c", 2, 0.25),
            self.sample(b"d", 3, -0.25),
        ]
        for sample in samples[:2]:
            sample.source_group = "anchor"
        for sample in samples[2:]:
            sample.source_group = "live"
        samples[-1].has_value = False
        samples[-1].value_mass = 0.0
        self.trainer.assign_weights(samples, 1.0, 0.25)
        policy = {
            group: sum(
                sample.policy_weight
                for sample in samples
                if sample.source_group == group
            )
            for group in ("anchor", "live")
        }
        value = {
            group: sum(
                sample.value_weight
                for sample in samples
                if sample.source_group == group
            )
            for group in ("anchor", "live")
        }
        self.assertAlmostEqual(policy["live"] / sum(policy.values()), 0.25)
        self.assertAlmostEqual(value["live"] / sum(value.values()), 0.25)
        self.assertEqual(samples[-1].value_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
