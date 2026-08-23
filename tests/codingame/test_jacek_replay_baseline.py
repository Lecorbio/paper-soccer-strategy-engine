import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_features as features  # noqa: E402
from tests.codingame.test_jacek_replay_workflow import make_round  # noqa: E402
try:
    import numpy as np
    import jacek_replay_baseline as baseline  # noqa: E402
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    baseline = None


@unittest.skipIf(np is None, "research tests require requirements-research.txt")
class JacekReplayBaselineTests(unittest.TestCase):
    def test_diagnostic_checkpoint_is_deterministic_and_strictly_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            original = (
                baseline.replay_features.INPUT_COUNT,
                baseline.training.HIDDEN_ONE,
                baseline.training.HIDDEN_TWO,
                baseline.training.FEATURE_SCHEMA_HASH,
            )
            with baseline.baseline_runtime_contract():
                parameters = baseline.training.initialize(20260823)
                first = directory / "first.runtime"
                second = directory / "second.runtime"
                first_report = baseline.training.export_runtime(first, parameters)
                second_report = baseline.training.export_runtime(second, parameters)
                loaded, loaded_report = baseline.training.load_runtime(first)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertEqual(first_report, second_report)
                self.assertEqual(first_report, loaded_report)
                self.assertEqual(
                    loaded["w1"].shape, (baseline.BASELINE_INPUTS, 32)
                )
                self.assertEqual(
                    first_report["feature_schema_sha256"],
                    baseline.BASELINE_FEATURE_SCHEMA_HASH.hex(),
                )
                damaged = bytearray(first.read_bytes())
                damaged[-1] ^= 1
                first.write_bytes(damaged)
                with self.assertRaisesRegex(ValueError, "payload SHA-256"):
                    baseline.training.load_runtime(first)
            self.assertEqual(
                (
                    baseline.replay_features.INPUT_COUNT,
                    baseline.training.HIDDEN_ONE,
                    baseline.training.HIDDEN_TWO,
                    baseline.training.FEATURE_SCHEMA_HASH,
                ),
                original,
            )

    def test_baseline_training_contract_and_selected_metrics_are_frozen(self):
        metrics = {
            "samples": 8,
            "weighted_huber": 0.2,
            "sign_accuracy": 0.75,
            "correlation": 0.4,
            "mae": 0.3,
            "prediction_mean": 0.0,
        }
        selection = {
            **baseline.fixed_selection_contract(),
            "chosen_seed": 20260823,
            "seed_reports": [
                {"seed": seed, "validation": dict(metrics)}
                for seed in baseline.training.FIXED_SEEDS
            ],
        }
        baseline.validate_selection(selection, metrics)
        selection["optimizer"]["epochs"] = 1
        with self.assertRaisesRegex(ValueError, "optimizer"):
            baseline.validate_selection(selection, metrics)

    def test_joint_categories_collapse_to_distance_only_schema(self):
        active = np.asarray(features.encode_active(features.ReplayState()))
        collapsed = baseline.collapse(active)
        self.assertEqual(len(collapsed), 105)
        self.assertTrue(np.all(collapsed >= 316))
        self.assertLess(int(collapsed[-1]), baseline.BASELINE_INPUTS)

        modified = active.copy()
        first = next(index for index, value in enumerate(modified) if value >= 316)
        vertex, _ = divmod(int(modified[first]) - 316, 57)
        modified[first] = 316 + vertex * 57 + 56
        collapsed = baseline.collapse(modified)
        self.assertIn(316 + vertex * 8 + 7, collapsed)

    def test_candidate_must_bind_the_same_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            pack = {
                "schema": "papersoccer.jacek-replay-pack-report.v1",
                "shards": {
                    split: {"sha256": character * 64}
                    for split, character in (
                        ("train", "a"),
                        ("validation", "b"),
                        ("test", "c"),
                    )
                },
            }
            pack_path = directory / "pack.json"
            pack_path.write_text(json.dumps(pack))
            candidate = {
                "schema": "papersoccer.jacek-replay-bfm-model.v1",
                "status": "canonical-campaign-candidate-not-game-gated",
                "architecture": {"dimensions": [6301, 192, 32, 1]},
                "campaign_contract": {
                    "eligible": True,
                    "round": 2,
                    "continuation_games": 10_000,
                    "prior_rounds": 2,
                    "test_revealed": True,
                },
                "source_shards": [],
            }
            candidate_path = directory / "candidate.json"
            candidate_path.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError, "same shards"):
                baseline.validate_candidate_binding(
                    candidate_path,
                    [pack_path],
                    [pack],
                    directory / "workflow.json",
                    {"entries": []},
                )

    def test_final_candidate_requires_exact_canonical_workflow_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            round0 = make_round(directory, 0)
            round1 = make_round(directory, 1, round0)
            round2 = make_round(directory, 2, round1)
            pack_reports = [
                json.loads(path.read_bytes()) for path in round2["pack_paths"]
            ]
            _, binding = baseline.validate_candidate_binding(
                round2["model_path"],
                round2["pack_paths"],
                pack_reports,
                round2["receipt_path"],
                round2["validation"],
            )
            self.assertEqual(len(binding["canonical_workflow_entries"]), 3)

            candidate = json.loads(round2["model_path"].read_bytes())
            candidate["campaign_contract"]["canonical_ancestry"] = []
            round2["model_path"].write_text(json.dumps(candidate, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "canonical workflow lineage"):
                baseline.validate_candidate_binding(
                    round2["model_path"],
                    round2["pack_paths"],
                    pack_reports,
                    round2["receipt_path"],
                    round2["validation"],
                )


if __name__ == "__main__":
    unittest.main()
