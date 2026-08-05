from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

from benchmarks.flagship_study import web_summary


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_HASH = "a" * 64

ENTRANTS = {
    "mcts": "mcts-1000",
    "alpha_beta": "alpha-beta-50k",
    "jacek_inspired": "jacek-20k",
    "rank5_derived": "rank5-fixed-50k",
}
LABELS = {
    "mcts": "Tactical MctsBot",
    "alpha_beta": "Hand-evaluated AlphaBetaBot",
    "jacek_inspired": "Neural alpha-beta (JacekInspiredBot)",
    "rank5_derived": "Rank5DerivedBot — fixed 50k demo profile",
}


def _manifest() -> dict:
    configurations = []
    for family, config_id in ENTRANTS.items():
        configurations.append({
            "id": config_id,
            "family": family,
            "public_label": LABELS[family],
            "settings": (
                {"iterations": 1000}
                if family == "mcts"
                else {"max_nodes": 50000 if family != "jacek_inspired" else 20000}
            ),
        })
    return {
        "schema_version": web_summary.MANIFEST_SCHEMA,
        "study": {
            "id": "flagship-test",
            "title": "Flagship test",
            "frozen": True,
            "rank5_disclaimer": "Rank5 is adapted to the demo rules.",
        },
        "configurations": configurations,
        "candidate_grids": {
            family: [config_id] for family, config_id in ENTRANTS.items()
        },
        "latency_protocol": {"gate_ms": 50},
    }


def _selection() -> dict:
    rows = []
    values = {
        "mcts": (0.2, 36.0),
        "alpha_beta": (0.43, 24.0),
        "jacek_inspired": (0.56, 35.0),
        "rank5_derived": (0.5, 31.0),
    }
    for family, config_id in ENTRANTS.items():
        strength, latency = values[family]
        reference = family == "rank5_derived"
        rows.append({
            "id": config_id,
            "family": family,
            "fixed": reference,
            "gate_eligible": True,
            "selected": True,
            "constrained_pareto_optimal": family != "mcts",
            "validation_strength": strength,
            "validation_strength_pair_bootstrap_95": (
                None if reference else {"lower": strength - 0.04, "upper": strength + 0.04}
            ),
            "validation_strength_pairs": None if reference else 20,
            "validation_p95_ms": latency,
            "validation_latency_decisions": 1000,
            "strength_definition": (
                "defined common-opponent reference level" if reference else None
            ),
        })
    return {
        "schema_version": web_summary.SELECTION_SCHEMA,
        "manifest_sha256": MANIFEST_HASH,
        "source_phase": "validation",
        "test_authorized": True,
        "selected_configurations": {
            family: ENTRANTS[family] for family in web_summary.TUNABLE_FAMILIES
        },
        "fixed_rank5_configuration": ENTRANTS["rank5_derived"],
        "validation_pareto": rows,
    }


def _matchup(
    matchup_id: str,
    left: str,
    right: str,
    *,
    unresolved: bool = False,
) -> dict:
    if unresolved:
        score, lower, upper = 0.5, 0.25, 0.75
        classification, stronger = "statistically_unresolved", None
        left_wins = right_wins = 1
    else:
        score, lower, upper = 1.0, 0.75, 1.0
        classification, stronger = "stronger", left
        left_wins, right_wins = 2, 0
    return {
        "left_config_id": left,
        "right_config_id": right,
        "mean_pair_score": score,
        "pair_bootstrap_95": {"lower": lower, "upper": upper},
        "pairs": 1,
        "games": 2,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "truncations": 0,
        "conclusion": {
            "classification": classification,
            "stronger_config_id": stronger,
        },
    }


def _test_data() -> dict:
    mcts = ENTRANTS["mcts"]
    hand = ENTRANTS["alpha_beta"]
    neural = ENTRANTS["jacek_inspired"]
    rank5 = ENTRANTS["rank5_derived"]
    matchups = {
        "mcts-hand": _matchup("mcts-hand", hand, mcts),
        "mcts-neural": _matchup("mcts-neural", neural, mcts),
        "mcts-rank5": _matchup("mcts-rank5", rank5, mcts),
        "hand-neural": _matchup("hand-neural", neural, hand),
        "hand-rank5": _matchup("hand-rank5", rank5, hand),
        "neural-rank5": _matchup("neural-rank5", neural, rank5, unresolved=True),
    }
    bt_values = {
        mcts: (-1.1, -1.2, -1.0),
        hand: (0.1, 0.0, 0.2),
        neural: (0.55, 0.45, 0.65),
        rank5: (0.45, 0.35, 0.55),
    }
    calibration = {
        config_id: {
            "brier_score": 0.2,
            "log_loss": 0.5,
            "samples": 100,
            "decision_count": 101,
            "excluded": {"truncations": 0},
        }
        for config_id in ENTRANTS.values()
    }
    games = [{"game_id": index} for index in range(12)]
    return {
        "schema_version": web_summary.CURATED_SCHEMA,
        "manifest_sha256": MANIFEST_HASH,
        "phase": "test",
        "analysis_complete": True,
        "completeness": {
            "expected_games": 12,
            "completed_games": 12,
            "unique_game_ids": 12,
            "operationally_valid": True,
            "truncations": 0,
        },
        "binary_games": games,
        "sample_sizes": {"games": 12, "pairs": 6, "opening_depths": [4, 8]},
        "configurations": {config_id: {} for config_id in ENTRANTS.values()},
        "bradley_terry": {
            "point_fit": {
                "converged": True,
                "bot_ids": list(ENTRANTS.values()),
                "abilities": {
                    config_id: estimate
                    for config_id, (estimate, _, _) in bt_values.items()
                },
            },
            "resamples": 100,
            "successful_resamples": 100,
            "intervals": {
                config_id: {"estimate": estimate, "lower": lower, "upper": upper}
                for config_id, (estimate, lower, upper) in bt_values.items()
            },
        },
        "matchups": matchups,
        "calibration": calibration,
    }


class WebSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.selection = _selection()
        self.test = _test_data()

    def build(self) -> dict:
        return web_summary.build_summary(
            self.manifest,
            self.selection,
            self.test,
            manifest_hash=MANIFEST_HASH,
        )

    def test_builds_compact_performance_only_contract(self) -> None:
        summary = self.build()

        self.assertEqual(summary["schema"], web_summary.SUMMARY_SCHEMA)
        self.assertEqual(summary["study"]["entrantCount"], 4)
        self.assertEqual(summary["study"]["games"], 12)
        self.assertEqual(summary["study"]["pairs"], 6)
        self.assertEqual(
            summary["study"]["headline"],
            "Neural alpha-beta has the highest strength estimate, while its matchup "
            "with Rank5Derived remains statistically unresolved.",
        )
        self.assertEqual(summary["entrants"][0]["id"], ENTRANTS["jacek_inspired"])
        self.assertTrue(
            next(
                entrant for entrant in summary["entrants"]
                if entrant["id"] == ENTRANTS["rank5_derived"]
            )["validation"]["strengthIsReference"]
        )
        unresolved = next(
            matchup for matchup in summary["matchups"]
            if matchup["classification"] == "statistically_unresolved"
        )
        self.assertIsNone(unresolved["strongerId"])

        serialized = json.dumps(summary).lower()
        for forbidden in (
            "binary_games", "execution_environment", "observed_at", "timestamp",
            "sha256", "raw_shard", "outcomes",
        ):
            self.assertNotIn(forbidden, serialized)
        for link in summary["links"].values():
            self.assertTrue(link.startswith("https://github.com/"))
        self.assertEqual(set(summary["links"]), {"report"})

    def test_render_is_deterministic_classic_script(self) -> None:
        first = web_summary.render_summary(self.build())
        second = web_summary.render_summary(self.build())

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("globalThis.PaperSoccerBenchmarkResults = {"))
        self.assertTrue(first.endswith(";\n"))
        self.assertNotIn("NaN", first)

    def test_rejects_mismatched_provenance(self) -> None:
        for artifact in (self.selection, self.test):
            with self.subTest(artifact=artifact["schema_version"]):
                original = artifact["manifest_sha256"]
                artifact["manifest_sha256"] = "b" * 64
                with self.assertRaisesRegex(web_summary.SummaryError, "provenance"):
                    self.build()
                artifact["manifest_sha256"] = original

    def test_rejects_incomplete_analysis_and_missing_entrant(self) -> None:
        incomplete = copy.deepcopy(self.test)
        incomplete["analysis_complete"] = False
        with self.assertRaisesRegex(web_summary.SummaryError, "not complete"):
            web_summary.build_summary(
                self.manifest, self.selection, incomplete, manifest_hash=MANIFEST_HASH
            )

        self.test["configurations"].pop(ENTRANTS["mcts"])
        with self.assertRaisesRegex(web_summary.SummaryError, "absent from test"):
            self.build()

    def test_rejects_bad_intervals(self) -> None:
        self.test["bradley_terry"]["intervals"][ENTRANTS["jacek_inspired"]][
            "upper"
        ] = 0.5
        with self.assertRaisesRegex(web_summary.SummaryError, "contain its estimate"):
            self.build()

    def test_rejects_bradley_terry_estimate_that_disagrees_with_point_fit(self) -> None:
        self.test["bradley_terry"]["intervals"][ENTRANTS["jacek_inspired"]][
            "estimate"
        ] = 0.5
        with self.assertRaisesRegex(web_summary.SummaryError, "point-fit ability"):
            self.build()

    def test_headline_tracks_the_validated_results(self) -> None:
        rank5 = ENTRANTS["rank5_derived"]
        mcts = ENTRANTS["mcts"]
        point_fit = self.test["bradley_terry"]["point_fit"]["abilities"]
        intervals = self.test["bradley_terry"]["intervals"]
        point_fit[rank5] = 0.7
        intervals[rank5] = {"estimate": 0.7, "lower": 0.6, "upper": 0.8}
        point_fit[mcts] = -1.35
        intervals[mcts] = {"estimate": -1.35, "lower": -1.45, "upper": -1.25}

        summary = self.build()

        self.assertEqual(summary["entrants"][0]["id"], rank5)
        self.assertEqual(
            summary["study"]["headline"],
            "Rank5Derived has the highest strength estimate, while its matchup with "
            "Neural alpha-beta remains statistically unresolved.",
        )

    def test_equal_strengths_have_a_deterministic_id_tiebreak(self) -> None:
        values = {
            ENTRANTS["mcts"]: (-0.95, -1.05, -0.85),
            ENTRANTS["alpha_beta"]: (0.2, 0.1, 0.3),
            ENTRANTS["jacek_inspired"]: (0.55, 0.45, 0.65),
            ENTRANTS["rank5_derived"]: (0.2, 0.1, 0.3),
        }
        for config_id, (estimate, lower, upper) in values.items():
            self.test["bradley_terry"]["point_fit"]["abilities"][config_id] = estimate
            self.test["bradley_terry"]["intervals"][config_id] = {
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
            }

        summary = self.build()

        self.assertEqual(
            [entrant["id"] for entrant in summary["entrants"]],
            [
                ENTRANTS["jacek_inspired"],
                ENTRANTS["alpha_beta"],
                ENTRANTS["rank5_derived"],
                ENTRANTS["mcts"],
            ],
        )

    def test_rejects_validation_reference_not_anchored_to_fixed_rank5(self) -> None:
        rank5 = next(
            row
            for row in self.selection["validation_pareto"]
            if row["id"] == ENTRANTS["rank5_derived"]
        )
        rank5["strength_definition"] = None
        rank5["validation_strength_pair_bootstrap_95"] = {"lower": 0.45, "upper": 0.55}
        rank5["validation_strength_pairs"] = 20

        with self.assertRaisesRegex(web_summary.SummaryError, "fixed Rank5"):
            self.build()

    def test_rejects_matchup_score_and_study_total_mismatches(self) -> None:
        bad_score = copy.deepcopy(self.test)
        row = bad_score["matchups"]["mcts-hand"]
        row["mean_pair_score"] = 0.75
        row["pair_bootstrap_95"] = {"lower": 0.6, "upper": 0.9}
        with self.assertRaisesRegex(web_summary.SummaryError, "score does not match"):
            web_summary.build_summary(
                self.manifest,
                self.selection,
                bad_score,
                manifest_hash=MANIFEST_HASH,
            )

        bad_total = copy.deepcopy(self.test)
        row = bad_total["matchups"]["mcts-hand"]
        row["pairs"] = 2
        row["games"] = 4
        row["left_wins"] = 4
        row["right_wins"] = 0
        with self.assertRaisesRegex(web_summary.SummaryError, "study totals"):
            web_summary.build_summary(
                self.manifest,
                self.selection,
                bad_total,
                manifest_hash=MANIFEST_HASH,
            )

    def test_rejects_invalid_calibration_metrics(self) -> None:
        config_id = ENTRANTS["mcts"]
        cases = (
            ("brier_score", 1.01, "Brier score"),
            ("log_loss", -0.01, "log loss"),
            ("samples", 102, "decision count"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                test_data = copy.deepcopy(self.test)
                test_data["calibration"][config_id][field] = value
                with self.assertRaisesRegex(web_summary.SummaryError, message):
                    web_summary.build_summary(
                        self.manifest,
                        self.selection,
                        test_data,
                        manifest_hash=MANIFEST_HASH,
                    )

    def test_rejects_truncations_at_every_published_analysis_layer(self) -> None:
        cases = []
        completeness = copy.deepcopy(self.test)
        completeness["completeness"]["truncations"] = 1
        cases.append(completeness)

        matchup = copy.deepcopy(self.test)
        matchup["matchups"]["mcts-hand"]["truncations"] = 1
        cases.append(matchup)

        calibration = copy.deepcopy(self.test)
        calibration["calibration"][ENTRANTS["mcts"]]["excluded"]["truncations"] = 1
        cases.append(calibration)

        for test_data in cases:
            with self.subTest(), self.assertRaisesRegex(web_summary.SummaryError, "truncation"):
                web_summary.build_summary(
                    self.manifest,
                    self.selection,
                    test_data,
                    manifest_hash=MANIFEST_HASH,
                )

    def test_write_and_check_cli_use_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest_path = root / "manifest.json"
            selection_path = root / "selection.json"
            test_path = root / "test.json"
            output_path = root / "benchmark-results.js"

            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.selection["manifest_sha256"] = actual_hash
            self.test["manifest_sha256"] = actual_hash
            selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
            test_path.write_text(json.dumps(self.test), encoding="utf-8")
            arguments = [
                "--manifest", str(manifest_path),
                "--selection-lock", str(selection_path),
                "--test-data", str(test_path),
                "--output", str(output_path),
            ]

            self.assertEqual(web_summary.main(["--write", *arguments]), 0)
            expected = web_summary.render_summary(
                web_summary.generate_summary(manifest_path, selection_path, test_path)
            )
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected)
            self.assertEqual(web_summary.main(["--check", *arguments]), 0)

            output_path.write_text(expected + "// stale\n", encoding="utf-8")
            self.assertEqual(web_summary.main(["--check", *arguments]), 1)

    def test_checked_in_snapshot_is_byte_fresh(self) -> None:
        expected = web_summary.render_summary(web_summary.generate_summary())
        actual = web_summary.DEFAULT_OUTPUT.read_text(encoding="utf-8")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
