from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RECORDER_PATH = (
    ROOT / "tools" / "record_rank4_jacek_hybrid_proof_scope_clock.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rank4_jacek_hybrid_proof_scope_recorder", RECORDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recorder
SPEC.loader.exec_module(recorder)


def preliminary_fields() -> dict[str, str]:
    """The qualification-relevant fields from the discarded 1-vs-0 run."""

    return {
        "bank": "all",
        "games": "76",
        "candidate_wins": "38",
        "reference_wins": "38",
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": "23/15/0/0/38",
        "candidate_p1": "15/23/0/0/38",
        "candidate_invocations": "1172",
        "candidate_searches": "1172",
        "candidate_illegal": "0",
        "candidate_operational": "0",
        "candidate_exceptions": "0",
        "candidate_hard_timeouts": "0",
        "candidate_first_ms_max": "800.179",
        "candidate_later_ms_max": "165.189",
        "reference_invocations": "1174",
        "reference_searches": "1174",
        "reference_illegal": "0",
        "reference_operational": "0",
        "reference_exceptions": "0",
        "reference_hard_timeouts": "0",
        "reference_first_ms_max": "800.199",
        "reference_later_ms_max": "165.191",
        "candidate_proof_rebound": "1172/33/3",
        "candidate_proof_root": "1172/33/3",
        "candidate_proof_leaf": "0/0/0",
        "candidate_proof_ply1": "0/0/0/0",
        "candidate_proof_ply2": "0/0/0/0",
        "reference_proof_rebound": "0/0/0",
        "reference_proof_root": "0/0/0",
        "reference_proof_leaf": "0/0/0",
        "reference_proof_ply1": "0/0/0/0",
        "reference_proof_ply2": "0/0/0/0",
    }


def output_line(fields: dict[str, str]) -> str:
    return "summary " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


class ProofScopeRecorderTest(unittest.TestCase):
    def test_parse_fields_rejects_duplicate_and_malformed_tokens(self) -> None:
        invalid_lines = (
            "summary games=76 games=76",
            "summary games",
            "summary =76",
            "summary games=",
            "summary games=7=6",
        )
        for line in invalid_lines:
            with self.subTest(line=line), self.assertRaises(ValueError):
                recorder.parse_fields(line)

    def test_valid_preliminary_stdout_fields_are_accepted(self) -> None:
        fields = recorder.parse_fields(output_line(preliminary_fields()))
        details = recorder.validate_summary(fields, "all", 1, 0)
        self.assertEqual(details["colors"], [(23, 15, 0, 0, 38),
                                              (15, 23, 0, 0, 38)])
        self.assertEqual(details["proof"]["candidate"]["root"],
                         (1172, 33, 3))
        self.assertEqual(details["proof"]["reference"]["root"], (0, 0, 0))

    def test_aggregate_and_color_accounting_mismatches_are_rejected(self) -> None:
        aggregate = preliminary_fields()
        aggregate["candidate_wins"] = "39"
        with self.assertRaisesRegex(ValueError, "aggregate game accounting"):
            recorder.validate_summary(aggregate, "all", 1, 0)

        colors = preliminary_fields()
        colors["candidate_p0"] = "22/16/0/0/38"
        with self.assertRaisesRegex(ValueError, "color sums"):
            recorder.validate_summary(colors, "all", 1, 0)

    def test_unfinished_or_failed_games_are_not_qualification_clean(self) -> None:
        unfinished = preliminary_fields()
        unfinished.update(
            candidate_wins="37",
            unfinished="1",
            candidate_p0="22/15/1/0/38",
        )
        with self.assertRaisesRegex(ValueError, "unfinished|failed|clean"):
            recorder.validate_summary(unfinished, "all", 1, 0)

        failed = preliminary_fields()
        failed.update(
            reference_wins="37",
            failed="1",
            candidate_p1="15/22/0/1/38",
        )
        with self.assertRaisesRegex(ValueError, "unfinished|failed|clean"):
            recorder.validate_summary(failed, "all", 1, 0)

    def test_same_runtime_headroom_thresholds_are_strict(self) -> None:
        for engine, phase, limit in (
            ("candidate", "first", "990.000"),
            ("candidate", "later", "198.000"),
            ("reference", "first", "990.000"),
            ("reference", "later", "198.000"),
        ):
            fields = preliminary_fields()
            fields[f"{engine}_{phase}_ms_max"] = limit
            with self.subTest(engine=engine, phase=phase), \
                    self.assertRaisesRegex(ValueError, "lacks headroom"):
                recorder.validate_summary(fields, "all", 1, 0)

    def test_disabled_scope_leakage_and_missing_enabled_work_are_rejected(self) -> None:
        leaked = preliminary_fields()
        leaked["candidate_proof_leaf"] = "1/0/0"
        with self.assertRaisesRegex(ValueError, "disabled candidate leaf"):
            recorder.validate_summary(leaked, "all", 1, 0)

        ignored = preliminary_fields()
        ignored["candidate_proof_root"] = "0/0/0"
        with self.assertRaisesRegex(ValueError, "enabled candidate root"):
            recorder.validate_summary(ignored, "all", 1, 0)

    def test_proof_hits_and_cutoffs_cannot_exceed_their_probe_accounting(self) -> None:
        excessive_hits = preliminary_fields()
        excessive_hits["candidate_proof_root"] = "2/3/0"
        with self.assertRaisesRegex(ValueError, "hit count exceeds probes"):
            recorder.validate_summary(excessive_hits, "all", 1, 0)

        bad_cutoffs = preliminary_fields()
        bad_cutoffs["candidate_proof_ply1"] = "5/1/1/1"
        with self.assertRaisesRegex(ValueError, "cutoff accounting"):
            recorder.validate_summary(bad_cutoffs, "all", 5, 0)


if __name__ == "__main__":
    unittest.main()
