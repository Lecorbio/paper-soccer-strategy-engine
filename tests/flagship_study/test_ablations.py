from __future__ import annotations

import unittest

from benchmarks.flagship_study import ablations
from benchmarks.flagship_study.prepare_manifest import (
    config_alpha_beta,
    config_mcts,
)


COMPARISONS = {
    "mcts": [
        ["mcts-1000", "mcts-2000"],
        ["mcts-2000", "mcts-4000"],
        ["mcts-1000", "mcts-4000"],
    ],
    "alpha_beta": [
        ["alpha-beta-20k", "alpha-beta-50k"],
        ["alpha-beta-50k", "alpha-beta-100k"],
        ["alpha-beta-20k", "alpha-beta-100k"],
    ],
    "jacek_inspired": [
        ["jacek-20k", "jacek-50k"],
        ["jacek-50k", "jacek-100k"],
        ["jacek-20k", "jacek-100k"],
    ],
    "equal_budget_evaluator": [
        ["alpha-beta-20k", "jacek-20k"],
        ["alpha-beta-50k", "jacek-50k"],
        ["alpha-beta-100k", "jacek-100k"],
    ],
}


def _manifest() -> dict:
    return {
        "configurations": (
            [config_mcts(value) for value in (1000, 2000, 4000)]
            + [config_alpha_beta(value, neural=False)
               for value in (20_000, 50_000, 100_000)]
            + [config_alpha_beta(value, neural=True)
               for value in (20_000, 50_000, 100_000)]
        ),
        "statistics": {
            "ablations": {
                "practical_gain_threshold": 0.01,
                "comparison_unit": "aligned_color_swapped_opening_pair",
                "bootstrap_method": "paired_difference_percentile",
                "bootstrap_resamples": 200,
                "stratify_by": "opening_depth",
                "comparisons": COMPARISONS,
            },
        },
        "openings": {"depths": [4, 8]},
        "samples": {
            "development": {"color_swapped_pairs_per_depth_matchup": 2},
            "validation": {"color_swapped_pairs_per_depth_matchup": 2},
        },
        "seeds": {
            "analysis": {"development": "7001", "validation": "7002"},
        },
    }


def _curated(phase: str) -> dict:
    values = {
        "mcts-1000": 0.0,
        "mcts-2000": 0.5,
        "mcts-4000": 0.5,
        "alpha-beta-20k": 0.5,
        "alpha-beta-50k": 0.0,
        "alpha-beta-100k": 0.0,
        "jacek-20k": 0.5,
        "jacek-50k": 0.5,
        "jacek-100k": 0.5,
    }
    opening_ids = [
        f"{phase}-d4-a", f"{phase}-d4-b",
        f"{phase}-d8-a", f"{phase}-d8-b",
    ]
    return {
        "phase": phase,
        "paired_scores": {
            identifier: {
                "phase": phase,
                "bot_id": identifier,
                "opponent_config_id": "rank5-fixed-50k",
                "opening_ids": list(opening_ids),
                "opening_depths": [4, 4, 8, 8],
                "scores": [score] * 4,
            }
            for identifier, score in values.items()
        },
    }


class PreregisteredAblationTests(unittest.TestCase):
    def test_aligned_pair_bootstraps_and_classifications_are_deterministic(self) -> None:
        manifest = _manifest()
        development = _curated("development")
        validation = _curated("validation")

        first = ablations.compute(manifest, development, validation)
        second = ablations.compute(manifest, development, validation)

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], ablations.SCHEMA)
        self.assertEqual(first["source_phases"], ["development", "validation"])
        mcts = first["scaling"]["mcts"]
        self.assertEqual(mcts[0]["validation_classification"],
                         "supported_practical_gain")
        self.assertEqual(mcts[1]["validation_classification"],
                         "supported_no_practical_gain")
        hand = first["scaling"]["alpha_beta"]
        self.assertEqual(hand[0]["validation_classification"],
                         "supported_regression")
        evaluator = first["equal_budget_evaluator"]
        self.assertEqual(evaluator[0]["validation_classification"],
                         "practical_equivalence_supported")
        self.assertEqual(evaluator[1]["validation_classification"],
                         "neural_materially_stronger")
        for family in first["scaling"].values():
            for comparison in family:
                for phase in ("development", "validation"):
                    metrics = comparison["phases"][phase]
                    self.assertEqual(metrics["pairs"], 4)
                    self.assertEqual(
                        metrics["pair_difference_bootstrap_95"]["resamples"], 200
                    )

    def test_misaligned_or_duplicate_opening_pairs_are_rejected(self) -> None:
        validation = _curated("validation")
        validation["paired_scores"]["mcts-1000"]["opening_ids"][0] = "different"
        with self.assertRaisesRegex(ablations.AblationError, "not aligned"):
            ablations.compute(_manifest(), _curated("development"), validation)

        duplicated = _curated("validation")
        payload = duplicated["paired_scores"]["mcts-1000"]
        payload["opening_ids"][1] = payload["opening_ids"][0]
        with self.assertRaisesRegex(ablations.AblationError, "duplicate"):
            ablations.compute(_manifest(), _curated("development"), duplicated)

    def test_classification_uses_interval_not_only_point_delta(self) -> None:
        self.assertEqual(
            ablations._scaling_classification(0.005, 0.030, 0.01),
            "unresolved_at_1pp",
        )
        self.assertEqual(
            ablations._scaling_classification(-0.020, 0.009, 0.01),
            "supported_no_practical_gain",
        )
        self.assertEqual(
            ablations._evaluator_classification(-0.009, 0.009, 0.01),
            "practical_equivalence_supported",
        )
        self.assertEqual(
            ablations._evaluator_classification(-0.020, 0.005, 0.01),
            "unresolved_at_1pp",
        )


if __name__ == "__main__":
    unittest.main()
