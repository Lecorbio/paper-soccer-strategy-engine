from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from benchmarks.flagship_study import release_summary


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDY_ROOT = REPOSITORY_ROOT / "benchmarks/flagship_study"


class ReleaseSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((STUDY_ROOT / "manifest.json").read_text())
        cls.selection = json.loads((STUDY_ROOT / "selection_lock.json").read_text())
        cls.test_data = json.loads((STUDY_ROOT / "data/test.json").read_text())
        cls.source_hashes = {
            "manifest": release_summary._sha256_file(STUDY_ROOT / "manifest.json"),
            "selection_lock": release_summary._sha256_file(
                STUDY_ROOT / "selection_lock.json"
            ),
            "test_data": release_summary._sha256_file(STUDY_ROOT / "data/test.json"),
        }

    def build(self, *, test_data: dict | None = None) -> dict:
        return release_summary.build_release_summary(
            self.manifest,
            self.selection,
            self.test_data if test_data is None else test_data,
            source_hashes=self.source_hashes,
        )

    def test_compact_contract_contains_release_evidence(self) -> None:
        summary = self.build()

        self.assertEqual(summary["schema_version"], release_summary.SCHEMA)
        self.assertEqual(
            summary["test"],
            {
                "games": 4800,
                "pairs": 2400,
                "truncations": 0,
                "opening_depths": [4, 8, 12, 20],
                "bootstrap_resamples": 10000,
            },
        )
        self.assertEqual(
            [row["config_id"] for row in summary["locked_configurations"]],
            ["mcts-1000", "alpha-beta-50k", "jacek-20k", "rank5-fixed-50k"],
        )
        rank5 = summary["locked_configurations"][-1]
        self.assertEqual(rank5["validation_p95_ms"], 31.383417)
        self.assertEqual(rank5["all_edge_p95_ms"], 27.236708)
        self.assertEqual(
            summary["provenance"]["rank5_submission"]["sha256"],
            "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29",
        )
        self.assertEqual(
            summary["provenance"]["neural_model"]["sha256"],
            "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084",
        )
        neural_rank5 = next(
            row
            for row in summary["pairwise_results"]
            if row["matchup_id"] == "test-jacek-vs-rank5"
        )
        self.assertEqual(neural_rank5["classification"], "statistically_unresolved")
        self.assertEqual(
            (neural_rank5["ci_lower"], neural_rank5["ci_upper"]),
            (0.4825, 0.545),
        )

    def test_checked_in_outputs_are_byte_identical_and_compact(self) -> None:
        expected = release_summary.generate_release_files()

        self.assertEqual(set(expected), set(release_summary.OUTPUT_FILES))
        for name, content in expected.items():
            self.assertEqual(content, (STUDY_ROOT / "summary" / name).read_text())
        combined = "\n".join(expected.values())
        for decision_level_field in (
            '"binary_games"',
            '"calibration_observations"',
            '"game_id"',
        ):
            self.assertNotIn(decision_level_field, combined)

    def test_refuses_truncated_or_wrong_provenance_results(self) -> None:
        truncated = dict(self.test_data)
        truncated["completeness"] = dict(self.test_data["completeness"])
        truncated["completeness"]["truncations"] = 1
        with self.assertRaisesRegex(release_summary.ReleaseSummaryError, "truncations"):
            self.build(test_data=truncated)

        mismatched = dict(self.test_data)
        mismatched["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(release_summary.ReleaseSummaryError, "manifest"):
            self.build(test_data=mismatched)

    def test_write_check_and_stale_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            common = [
                "--manifest", str(STUDY_ROOT / "manifest.json"),
                "--selection-lock", str(STUDY_ROOT / "selection_lock.json"),
                "--test-data", str(STUDY_ROOT / "data/test.json"),
                "--output-dir", str(output),
            ]
            self.assertEqual(release_summary.main(["--write", *common]), 0)
            self.assertEqual(release_summary.main(["--check", *common]), 0)
            (output / "pairwise.csv").write_text("stale\n", encoding="utf-8")
            with mock.patch("sys.stderr"):
                self.assertEqual(release_summary.main(["--check", *common]), 1)


if __name__ == "__main__":
    unittest.main()
