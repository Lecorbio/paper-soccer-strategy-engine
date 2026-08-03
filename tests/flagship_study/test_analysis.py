from __future__ import annotations

import json
import math
import unittest

from benchmarks.flagship_study.analysis import (
    AnalysisError,
    CalibrationMapping,
    ConvergenceError,
    DisconnectedComparisonError,
    PairedComparison,
    SeparationError,
    apply_calibration,
    bootstrap_bradley_terry,
    calibration_metrics,
    classify_pareto,
    depth_stratified_pair_bootstrap,
    evaluate_calibration,
    fit_bradley_terry,
    fit_logistic_calibration,
    orient_player_one_probability,
    orient_player_one_score,
    pair_clustered_calibration_metrics,
    pair_score,
    summarize_pair_outcomes,
)


def _ordered_bt_pairs() -> list[PairedComparison]:
    """Connected data with finite estimates ordered Alpha > ... > Rank5."""

    bots = ("Alpha", "Beta", "Jacek", "Rank5")
    pairs: list[PairedComparison] = []
    index = 0
    for left_index, left in enumerate(bots):
        for right in bots[left_index + 1:]:
            for depth in (4, 8):
                # The stronger bot wins 6/8 games. Splits keep the directed
                # win graph strongly connected and the MLE finite.
                for outcomes in ((1, 1), (1, 1), (1, 0), (1, 0)):
                    pairs.append(PairedComparison(
                        pair_id=f"bt-{index}",
                        opening_depth=depth,
                        bot_a=left,
                        bot_b=right,
                        outcomes_for_a=outcomes,
                    ))
                    index += 1
    return pairs


def _calibration_observations(
    bot_id: str = "Jacek",
) -> list[dict[str, object]]:
    values = (
        (-2.0, 0),
        (-1.5, 0),
        (-1.0, 1),
        (-0.5, 0),
        (0.0, 0),
        (0.2, 1),
        (0.5, 0),
        (1.0, 1),
        (1.5, 1),
        (2.0, 1),
    )
    return [
        {
            "phase": "validation",
            "bot_id": bot_id,
            "raw_score": score,
            "outcome": outcome,
            "player_to_move": 1,
            "score_perspective": "player_one",
            "score_kind": "signed",
            "completed_depth": 6,
        }
        for score, outcome in values
    ]


class PairOutcomeTests(unittest.TestCase):
    def test_pair_summary_uses_two_binary_games_without_draws(self) -> None:
        pairs = [
            PairedComparison("p0", 4, "candidate", "opponent", (1, 1)),
            PairedComparison("p1", 8, "candidate", "opponent", (1, 0)),
            PairedComparison("p2", 12, "candidate", "opponent", (0, 0)),
        ]

        summary = summarize_pair_outcomes(pairs)

        self.assertEqual(summary["game_wins"], 3)
        self.assertEqual(summary["game_losses"], 3)
        self.assertEqual(summary["truncations"], 0)
        self.assertEqual(summary["pairs_won_2_0"], 1)
        self.assertEqual(summary["pairs_split_1_1"], 1)
        self.assertEqual(summary["pairs_lost_0_2"], 1)
        self.assertEqual(summary["mean_pair_score"], 0.5)
        self.assertEqual(pair_score((1, 0)), 0.5)

    def test_pair_validation_rejects_wrong_size_nonbinary_and_truncation(self) -> None:
        with self.assertRaisesRegex(AnalysisError, "exactly two"):
            pair_score((1,))
        with self.assertRaisesRegex(AnalysisError, "integer 0 or 1"):
            pair_score((1, 2))
        with self.assertRaisesRegex(AnalysisError, "truncation"):
            summarize_pair_outcomes([
                PairedComparison("p0", 4, "a", "b", (1, 0), truncated=True)
            ])

    def test_depth_stratified_pair_bootstrap_is_deterministic(self) -> None:
        pairs = [
            PairedComparison("p0", 4, "a", "b", (1, 1)),
            PairedComparison("p1", 4, "a", "b", (0, 0)),
            PairedComparison("p2", 8, "a", "b", (1, 0)),
            PairedComparison("p3", 8, "a", "b", (1, 1)),
        ]

        first = depth_stratified_pair_bootstrap(pairs, seed=7331)
        second = depth_stratified_pair_bootstrap(pairs, seed=7331)

        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(first["opening_depths"], [4, 8])
        self.assertLessEqual(first["lower"], 0.625)
        self.assertGreaterEqual(first["upper"], 0.625)


class BradleyTerryTests(unittest.TestCase):
    def test_known_ordering_and_sum_to_zero(self) -> None:
        fit = fit_bradley_terry(_ordered_bt_pairs())

        abilities = fit.abilities
        self.assertGreater(abilities["Alpha"], abilities["Beta"])
        self.assertGreater(abilities["Beta"], abilities["Jacek"])
        self.assertGreater(abilities["Jacek"], abilities["Rank5"])
        self.assertAlmostEqual(sum(abilities.values()), 0.0, places=12)
        self.assertTrue(fit.converged)

    def test_disconnected_and_separated_graphs_fail_explicitly(self) -> None:
        disconnected = [
            PairedComparison("ab", 4, "A", "B", (1, 0)),
            PairedComparison("cd", 4, "C", "D", (1, 0)),
        ]
        with self.assertRaises(DisconnectedComparisonError):
            fit_bradley_terry(disconnected)

        separated = []
        index = 0
        bots = ("A", "B", "C", "D")
        for left_index, left in enumerate(bots):
            for right in bots[left_index + 1:]:
                separated.append(PairedComparison(
                    f"sep-{index}", 4, left, right, (1, 1)
                ))
                index += 1
        with self.assertRaises(SeparationError):
            fit_bradley_terry(separated)

    def test_pair_clustered_bt_bootstrap_is_deterministic(self) -> None:
        pairs = _ordered_bt_pairs()

        first = bootstrap_bradley_terry(
            pairs, seed=991, resamples=200
        )
        second = bootstrap_bradley_terry(
            pairs, seed=991, resamples=200
        )

        self.assertEqual(first, second)
        self.assertEqual(first["opening_depths"], [4, 8])
        self.assertEqual(first["resamples"], 200)
        self.assertEqual(first["successful_resamples"], 200)
        self.assertEqual(len(first["strata"]), 12)
        for bot_id, interval in first["intervals"].items():
            self.assertLessEqual(interval["lower"], interval["estimate"], bot_id)
            self.assertGreaterEqual(interval["upper"], interval["estimate"], bot_id)

    def test_bootstrap_refit_separation_is_counted_and_thresholded(self) -> None:
        pairs = []
        index = 0
        bots = ("A", "B", "C", "D")
        for left_index, left in enumerate(bots):
            for right in bots[left_index + 1:]:
                pairs.append(PairedComparison(
                    f"balanced-{index}", 4, left, right, (1, 1)
                ))
                index += 1
                pairs.append(PairedComparison(
                    f"balanced-{index}", 4, left, right, (0, 0)
                ))
                index += 1

        with self.assertRaisesRegex(ConvergenceError, "too few finite"):
            bootstrap_bradley_terry(pairs, seed=1, resamples=100)
        accepted = bootstrap_bradley_terry(
            pairs,
            seed=1,
            resamples=100,
            minimum_success_fraction=0.8,
        )
        self.assertEqual(accepted["successful_resamples"], 82)
        self.assertEqual(accepted["failed_resamples"]["separation"], 18)


class CalibrationTests(unittest.TestCase):
    def test_player_one_orientation_helpers(self) -> None:
        self.assertEqual(orient_player_one_score(17.0, 1), 17.0)
        self.assertEqual(orient_player_one_score(17.0, 2), -17.0)
        self.assertAlmostEqual(orient_player_one_probability(0.8, 1), 0.8)
        self.assertAlmostEqual(orient_player_one_probability(0.8, 2), 0.2)

    def test_fit_orients_player_two_scores_before_standardization(self) -> None:
        player_one_rows = _calibration_observations()
        mixed_rows = []
        for index, row in enumerate(player_one_rows):
            transformed = dict(row)
            if index % 2:
                transformed["player_to_move"] = 2
                transformed["raw_score"] = -float(transformed["raw_score"])
            mixed_rows.append(transformed)

        player_one_fit = fit_logistic_calibration(
            player_one_rows, phase="validation"
        )
        mixed_fit = fit_logistic_calibration(mixed_rows, phase="validation")

        self.assertEqual(player_one_fit, mixed_fit)

    def test_calibration_fit_rejects_test_leakage(self) -> None:
        test_rows = _calibration_observations()
        for row in test_rows:
            row["phase"] = "test"

        with self.assertRaisesRegex(AnalysisError, "validation"):
            fit_logistic_calibration(test_rows, phase="test")
        with self.assertRaisesRegex(AnalysisError, "expected validation"):
            fit_logistic_calibration(test_rows, phase="validation")

    def test_cached_rank5_truncations_and_invalid_depths_are_excluded(self) -> None:
        clean = _calibration_observations("Rank5")
        baseline = fit_logistic_calibration(clean, phase="validation")
        augmented = list(clean) + [
            {
                "phase": "validation",
                "cached_continuation": True,
                # Deliberately omit score/outcome: filtering must happen first.
            },
            {"phase": "validation", "truncated": True},
            {
                "phase": "validation",
                "completed_depth": 0,
                "cached_continuation": False,
            },
        ]

        fitted = fit_logistic_calibration(augmented, phase="validation")

        self.assertEqual(fitted.sample_count, baseline.sample_count)
        self.assertEqual(fitted.score_mean, baseline.score_mean)
        self.assertEqual(fitted.score_scale, baseline.score_scale)
        self.assertEqual(fitted.intercept, baseline.intercept)
        self.assertEqual(fitted.slope, baseline.slope)
        self.assertEqual(fitted.excluded_cached_continuations, 1)
        self.assertEqual(fitted.excluded_truncations, 1)
        self.assertEqual(fitted.excluded_invalid_depths, 1)

    def test_logistic_separation_is_rejected(self) -> None:
        observations = []
        for score, outcome in ((-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1)):
            observations.append({
                "phase": "validation",
                "bot_id": "bot",
                "raw_score": score,
                "outcome": outcome,
                "player_to_move": 1,
                "completed_depth": 1,
            })
        with self.assertRaises(SeparationError):
            fit_logistic_calibration(observations, phase="validation")

    def test_frozen_mapping_round_trip_and_metrics(self) -> None:
        mapping = fit_logistic_calibration(
            _calibration_observations(), phase="validation"
        )
        restored = CalibrationMapping.from_dict(mapping.to_dict())

        json.dumps(mapping.to_dict(), sort_keys=True)
        self.assertEqual(
            apply_calibration(mapping, (-1.0, 0.0, 1.0)),
            apply_calibration(restored.to_dict(), (-1.0, 0.0, 1.0)),
        )
        metrics = calibration_metrics((0.1, 0.9), (0, 1))
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["brier_score"], 0.01)
        self.assertAlmostEqual(metrics["log_loss"], -math.log(0.9))
        self.assertEqual(len(metrics["reliability_bins"]), 10)
        self.assertEqual(sum(value["count"] for value in metrics["reliability_bins"]), 2)

    def test_pair_clustered_calibration_bootstrap_is_deterministic(self) -> None:
        probabilities = (0.15, 0.25, 0.75, 0.85, 0.35, 0.65, 0.45, 0.55)
        outcomes = (0, 0, 1, 1, 0, 1, 1, 0)
        clusters = ("p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4")
        strata = ("m1-d4",) * 4 + ("m1-d8",) * 4

        first = pair_clustered_calibration_metrics(
            probabilities, outcomes, clusters, strata,
            seed=991, resamples=200, minimum_bin_successful_resamples=1,
        )
        second = pair_clustered_calibration_metrics(
            probabilities, outcomes, clusters, strata,
            seed=991, resamples=200, minimum_bin_successful_resamples=1,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["samples"], 8)
        self.assertEqual(first["pair_clusters"], 4)
        bootstrap = first["pair_cluster_bootstrap_95"]
        self.assertEqual(bootstrap["method"], "pair_cluster_percentile_stratified")
        self.assertEqual(bootstrap["successful_resamples"], 200)
        self.assertEqual(
            sum(value["pair_clusters"] for value in first["reliability_bins"]), 8
        )
        populated = [
            value for value in first["reliability_bins"] if value["count"]
        ]
        self.assertTrue(all(
            value["observed_frequency_pair_bootstrap_95"] is not None
            for value in populated
        ))

    def test_pair_clustered_calibration_rejects_cross_stratum_cluster(self) -> None:
        with self.assertRaisesRegex(AnalysisError, "crosses strata"):
            pair_clustered_calibration_metrics(
                (0.2, 0.8), (0, 1), ("same", "same"), ("d4", "d8"),
                seed=1, resamples=2, minimum_bin_successful_resamples=1,
            )

    def test_test_metrics_exclude_cached_rank5_prediction(self) -> None:
        mapping = fit_logistic_calibration(
            _calibration_observations("Rank5"), phase="validation"
        )
        observations = [
            {
                "phase": "test",
                "bot_id": "Rank5",
                "raw_score": -0.5,
                "outcome": 0,
                "player_to_move": 1,
                "completed_depth": 6,
            },
            {
                "phase": "test",
                "bot_id": "Rank5",
                "raw_score": 0.5,
                "outcome": 1,
                "player_to_move": 1,
                "completed_depth": 6,
            },
            {"phase": "test", "cached_continuation": True},
        ]

        metrics = evaluate_calibration(mapping, observations, phase="test")

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["excluded"]["cached_continuations"], 1)


class ParetoTests(unittest.TestCase):
    def test_pareto_classification_and_test_leakage_guard(self) -> None:
        points = [
            {
                "id": "fast",
                "p95_ms": 10.0,
                "strength": 0.5,
                "strength_phases": ["development", "validation"],
                "latency_phase": "validation",
                "gate_eligible": True,
            },
            {
                "id": "slow-weak",
                "p95_ms": 20.0,
                "strength": 0.4,
                "strength_phases": ["validation"],
                "latency_phase": "validation",
                "gate_eligible": True,
            },
            {
                "id": "slow-strong",
                "p95_ms": 20.0,
                "strength": 0.7,
                "strength_phases": ["validation"],
                "latency_phase": "validation",
                "gate_eligible": True,
            },
        ]

        classified = {point["id"]: point for point in classify_pareto(points)}

        self.assertTrue(classified["fast"]["pareto_optimal"])
        self.assertFalse(classified["slow-weak"]["pareto_optimal"])
        self.assertTrue(classified["slow-strong"]["pareto_optimal"])
        points.append({
            "id": "ineligible-strongest", "p95_ms": 60.0, "strength": 1.0,
            "strength_phases": ["validation"], "latency_phase": "validation",
            "gate_eligible": False,
        })
        classified = {point["id"]: point for point in classify_pareto(points)}
        self.assertFalse(classified["ineligible-strongest"]["pareto_optimal"])
        self.assertTrue(classified["ineligible-strongest"]["unconstrained_pareto_optimal"])
        self.assertTrue(classified["slow-strong"]["constrained_pareto_optimal"])
        points[0]["strength_phases"] = ["test"]
        with self.assertRaisesRegex(AnalysisError, "test strength"):
            classify_pareto(points)


if __name__ == "__main__":
    unittest.main()
