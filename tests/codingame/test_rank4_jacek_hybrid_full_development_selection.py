import copy
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "build_rank4_jacek_hybrid_full_development_selection.py"
SPEC = importlib.util.spec_from_file_location("hybrid_full_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selection
SPEC.loader.exec_module(selection)


class FullDevelopmentSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = selection.load_report(*selection.EXPECTED_REPORTS[0])
        cls.rank4 = selection.load_report(*selection.EXPECTED_REPORTS[1])

    def test_real_evidence_selects_mask7_without_final_qualification(self):
        manifest = selection.build_selection()
        self.assertEqual(
            manifest["decision"]["status"], "selected-for-operational-activation"
        )
        self.assertEqual(manifest["decision"]["selected_exact_proof_mask"], 7)
        self.assertFalse(manifest["decision"]["final_qualification"])
        self.assertFalse(manifest["final_qualification"])
        self.assertEqual(
            [(item["candidate_wins"], item["reference_wins"])
             for item in manifest["reports"]],
            [(166, 140), (169, 137)],
        )
        self.assertTrue(
            manifest["evidence_policy"]["validation_or_final_banks_read"] is False
        )

    def test_report_files_are_exact_canonical_content_addresses(self):
        for engine, digest in selection.EXPECTED_REPORTS:
            report = selection.load_report(engine, digest)
            path = selection.REPORT_DIRECTORY / f"{digest}.json"
            self.assertEqual(selection.sha256(path.read_bytes()), digest)
            self.assertEqual(selection.canonical_json(report), path.read_bytes())

    def test_exact_frozen_plan_source_gate_and_head_are_bound(self):
        manifest = selection.build_selection()
        self.assertEqual(
            manifest["frozen_plan"]["sha256"], selection.EXPECTED_PLAN_SHA256
        )
        self.assertEqual(
            manifest["evaluated_identity"]["git_head"], selection.EXPECTED_HEAD
        )
        self.assertEqual(
            manifest["evaluated_identity"]["generated_source"]["sha256"],
            "6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f",
        )
        self.assertEqual(
            manifest["evaluated_identity"]["comparison_gate_source"]["sha256"],
            "d872e1720511d3045ac6890d566d795323e056db7ddf74cf365ce2f436f45b80",
        )

    def test_process_or_operational_failure_is_rejected(self):
        report = copy.deepcopy(self.control)
        report["returncode"] = 1
        with self.assertRaisesRegex(ValueError, "process status"):
            selection.validate_report(report, "hybrid-control")
        report = copy.deepcopy(self.control)
        report["parsed"]["aggregate"]["candidate_operational"] = "1"
        with self.assertRaisesRegex(ValueError, "operational|counter"):
            selection.validate_report(report, "hybrid-control")

    def test_exact_game_and_both_color_thresholds_are_revalidated(self):
        report = copy.deepcopy(self.control)
        report["parsed"]["aggregate"]["candidate_wins"] = "159"
        report["parsed"]["aggregate"]["reference_wins"] = "147"
        report["parsed"]["aggregate"]["candidate_p0"] = "76/77/0/0/153"
        with self.assertRaisesRegex(ValueError, "accounting|threshold|color sums"):
            selection.validate_report(report, "hybrid-control")
        report = copy.deepcopy(self.control)
        report["parsed"]["banks"][0]["games"] = "77"
        with self.assertRaisesRegex(ValueError, "game accounting|games"):
            selection.validate_report(report, "hybrid-control")

    def test_proof_and_timing_corruption_are_rejected(self):
        report = copy.deepcopy(self.control)
        report["parsed"]["aggregate"]["candidate_proof_ply2"] = "1/0/0/0"
        with self.assertRaisesRegex(ValueError, "ply2|proof"):
            selection.validate_report(report, "hybrid-control")
        report = copy.deepcopy(self.control)
        report["parsed"]["aggregate"]["candidate_later_ms_p99"] = "199.0"
        with self.assertRaisesRegex(ValueError, "timing|headroom"):
            selection.validate_report(report, "hybrid-control")

    def test_common_input_drift_and_prerequisite_drift_are_rejected(self):
        report = copy.deepcopy(self.rank4)
        report["inputs_after"] = copy.deepcopy(report["inputs_after"])
        report["inputs_after"]["CMakeLists.txt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "unstable"):
            selection.validate_report(report, "rank4")
        report = copy.deepcopy(self.rank4)
        report["inputs_before"] = copy.deepcopy(report["inputs_before"])
        report["inputs_before"]["extra"] = {"sha256": "1" * 64}
        with self.assertRaisesRegex(ValueError, "share exact inputs"):
            selection.validate_sequence(self.control, report)
        report = copy.deepcopy(self.rank4)
        report["accepted_control_prerequisite"]["sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            selection.validate_sequence(self.control, report)
        expected = self.rank4["accepted_control_prerequisite"]
        self.assertEqual(expected["sha256"], selection.EXPECTED_REPORTS[0][1])
        self.assertEqual(expected["ended_utc"], self.control["ended_utc"])
        self.assertLessEqual(
            selection.parse_utc(self.control["ended_utc"]),
            selection.parse_utc(self.rank4["started_utc"]),
        )

    def test_overlap_and_parsed_stdout_drift_are_rejected(self):
        report = copy.deepcopy(self.rank4)
        report["started_utc"] = self.control["started_utc"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            selection.validate_sequence(self.control, report)
        report = copy.deepcopy(self.control)
        report["parsed"]["aggregate"]["candidate_wins"] = "165"
        with self.assertRaisesRegex(ValueError, "serialized lines"):
            selection.validate_serialized_gate_output(report)

    def test_writer_is_canonical_and_content_addressed(self):
        payload = selection.build_selection()
        original = selection.OUTPUT_DIRECTORY
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            selection.OUTPUT_DIRECTORY = pathlib.Path(directory)
            try:
                path, digest = selection.write_selection(payload)
            finally:
                selection.OUTPUT_DIRECTORY = original
            self.assertEqual(path.name, f"{digest}.json")
            self.assertEqual(selection.sha256(path.read_bytes()), digest)
            self.assertEqual(path.read_bytes(), selection.canonical_json(payload))


if __name__ == "__main__":
    unittest.main()
