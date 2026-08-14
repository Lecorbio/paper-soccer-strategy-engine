import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = (
    ROOT / "tools/record_rank4_jacek_hybrid_null_fastpath_clock.py"
)
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("null_fastpath_recorder", RECORDER_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def summary_fields(label: str) -> dict[str, str]:
    fields = {
        "bank": label,
        "games": "76",
        "candidate_wins": "38",
        "reference_wins": "38",
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": "19/19/0/0/38",
        "candidate_p1": "19/19/0/0/38",
    }
    for engine in ("candidate", "reference"):
        fields.update({
            f"{engine}_invocations": "100",
            f"{engine}_searches": "100",
            f"{engine}_illegal": "0",
            f"{engine}_operational": "0",
            f"{engine}_exceptions": "0",
            f"{engine}_hard_timeouts": "0",
            f"{engine}_soft_overruns": "80",
            f"{engine}_nodes": "20000",
            f"{engine}_nodes_avg": "200.000",
            f"{engine}_nodes_p99": "300",
            f"{engine}_nodes_max": "400",
            f"{engine}_depth_avg": "4.000",
            f"{engine}_depth_max": "5",
            f"{engine}_attempted_depth_avg": "5.000",
            f"{engine}_attempted_depth_max": "6",
            f"{engine}_exhaustions": "80",
            f"{engine}_first_ms_p99": "800.100",
            f"{engine}_first_ms_max": "800.200",
            f"{engine}_later_ms_p99": "165.100",
            f"{engine}_later_ms_max": "165.200",
            f"{engine}_proof_rebound": "1110/254/3",
            f"{engine}_proof_root": "100/2/1",
            f"{engine}_proof_leaf": "1000/250/2",
            f"{engine}_proof_ply1": "10/2/0/2",
            f"{engine}_proof_ply2": "0/0/0/0",
        })
    fields["reference_nodes_avg"] = "190.000"
    fields["reference_depth_avg"] = "3.900"
    return fields


def line(prefix: str, fields: dict[str, str]) -> str:
    return prefix + " " + " ".join(f"{key}={value}" for key, value in fields.items())


def valid_stdout() -> str:
    bank = summary_fields("0")
    aggregate = dict(bank)
    aggregate["bank"] = "all"
    configuration = recorder.expected_configuration()
    return "\n".join((
        line("bank_summary", bank),
        line("summary", aggregate),
        line("configuration", configuration),
    )) + "\n"


class NullFastpathRecorderTest(unittest.TestCase):
    def test_frozen_command_is_development_d20_mask7_only(self):
        command = recorder.command_for_gate()
        joined = " ".join(command)
        self.assertIn("development_d20.tsv", joined)
        self.assertNotIn("validation", joined.lower())
        self.assertNotIn("final", joined.lower())
        self.assertEqual(command.count("7"), 2)
        self.assertEqual(recorder.expected_configuration()[
            "reference_exact_proof_mask"], "7")
        self.assertEqual(recorder.expected_configuration()[
            "reference_engine"], "rank4")

    def test_valid_output_passes_structure_and_thresholds(self):
        parsed = recorder.validate_gate_stdout(valid_stdout())
        self.assertEqual(recorder.selection_errors(parsed["aggregate"]), [])

    def test_below_overall_threshold_is_rejected(self):
        stdout = valid_stdout().replace(
            "candidate_wins=38 reference_wins=38",
            "candidate_wins=37 reference_wins=39",
        ).replace(
            "candidate_p0=19/19/0/0/38",
            "candidate_p0=18/20/0/0/38",
        )
        parsed = recorder.validate_gate_stdout(stdout)
        self.assertIn("fewer than 38", " ".join(
            recorder.selection_errors(parsed["aggregate"])))

    def test_below_one_color_threshold_is_rejected(self):
        stdout = valid_stdout().replace(
            "candidate_p0=19/19/0/0/38 candidate_p1=19/19/0/0/38",
            "candidate_p0=18/20/0/0/38 candidate_p1=20/18/0/0/38",
        )
        parsed = recorder.validate_gate_stdout(stdout)
        self.assertIn("physical color 0", " ".join(
            recorder.selection_errors(parsed["aggregate"])))

    def test_progress_requires_depth_or_throughput(self):
        stdout = valid_stdout().replace(
            "candidate_nodes_avg=200.000", "candidate_nodes_avg=189.000"
        ).replace(
            "candidate_depth_avg=4.000", "candidate_depth_avg=3.800"
        )
        parsed = recorder.validate_gate_stdout(stdout)
        self.assertIn("lower completed depth", " ".join(
            recorder.selection_errors(parsed["aggregate"])))

    def test_nonlower_depth_alone_satisfies_progress(self):
        stdout = valid_stdout().replace(
            "candidate_nodes_avg=200.000", "candidate_nodes_avg=189.000"
        ).replace(
            "candidate_depth_avg=4.000", "candidate_depth_avg=3.900"
        )
        parsed = recorder.validate_gate_stdout(stdout)
        self.assertEqual(recorder.selection_errors(parsed["aggregate"]), [])

    def test_timing_headroom_is_strict(self):
        with self.assertRaisesRegex(ValueError, "timing"):
            recorder.validate_gate_stdout(
                valid_stdout().replace(
                    "candidate_first_ms_p99=800.100",
                    "candidate_first_ms_p99=900.000",
                )
            )

    def test_failure_is_rejected(self):
        broken = valid_stdout().replace(
            "candidate_wins=38 reference_wins=38 unfinished=0 failed=0",
            "candidate_wins=37 reference_wins=38 unfinished=0 failed=1",
        ).replace(
            "candidate_p0=19/19/0/0/38",
            "candidate_p0=18/19/0/1/38",
        )
        with self.assertRaisesRegex(ValueError, "unfinished or failed"):
            recorder.validate_gate_stdout(broken)

    def test_rebound_scope_sum_is_strict(self):
        with self.assertRaisesRegex(ValueError, "rebound/scope"):
            recorder.validate_gate_stdout(
                valid_stdout().replace(
                    "candidate_proof_rebound=1110/254/3",
                    "candidate_proof_rebound=1110/253/3",
                )
            )

    def test_single_bank_and_aggregate_must_match(self):
        lines = valid_stdout().splitlines()
        lines[1] = lines[1].replace(
            "candidate_nodes=20000", "candidate_nodes=20001", 1
        )
        with self.assertRaisesRegex(ValueError, "summaries differ"):
            recorder.validate_gate_stdout("\n".join(lines) + "\n")

    def test_extra_stdout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            recorder.validate_gate_stdout(valid_stdout() + "noise\n")

    def test_content_addressed_attempt_prevents_retry(self):
        head = "a" * 40
        payload = {
            "schema": recorder.SCHEMA,
            "attempt_id": recorder.attempt_id(head),
        }
        raw = recorder.canonical_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / f"{digest}.json").write_bytes(raw)
            (output / "not-content-addressed.json").write_bytes(raw)
            self.assertEqual(
                recorder.matching_attempts(head, output),
                [output / f"{digest}.json"],
            )

    def test_content_addressed_persistence_reloads_exact_canonical_bytes(self):
        payload = {"schema": "synthetic", "value": [3, 2, 1]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path, digest = recorder.persist_content_addressed_report(
                output, payload, 123
            )
            self.assertEqual(path.name, f"{digest}.json")
            self.assertEqual(path.read_bytes(), recorder.canonical_json(payload))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertFalse((output / f".{digest}.123.tmp").exists())

    def test_v1_and_v2_rejections_do_not_count_as_the_one_v3_attempt(self):
        head = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for schema in (
                "rank4-jacek-hybrid-null-fastpath-clock-v1",
                "rank4-jacek-hybrid-null-fastpath-clock-v2",
            ):
                payload = {
                    "schema": schema,
                    "attempt_id": recorder.attempt_id(head),
                }
                raw = recorder.canonical_json(payload)
                digest = hashlib.sha256(raw).hexdigest()
                (output / f"{digest}.json").write_bytes(raw)
            self.assertEqual(recorder.matching_attempts(head, output), [])

    def test_v3_attempt_identity_binds_all_amendment_prerequisites(self):
        key = recorder.attempt_key("a" * 40)
        self.assertEqual(key["schema"], recorder.SCHEMA)
        self.assertEqual(
            key["original_plan_sha256"],
            recorder.EXPECTED_ORIGINAL_PLAN_SHA256,
        )
        self.assertEqual(
            key["audit_receipt_sha256"],
            recorder.EXPECTED_AUDIT_RECEIPT_SHA256,
        )
        self.assertEqual(
            key["plan_v2_sha256"], recorder.EXPECTED_PLAN_V2_SHA256,
        )
        self.assertEqual(
            key["contamination_receipt_sha256"],
            recorder.EXPECTED_CONTAMINATION_RECEIPT_SHA256,
        )
        self.assertEqual(
            key["rejected_v1_attempt_sha256"],
            recorder.EXPECTED_REJECTED_V1_ATTEMPT_SHA256,
        )
        self.assertEqual(
            key["rejected_v2_attempt_sha256"],
            recorder.EXPECTED_REJECTED_V2_ATTEMPT_SHA256,
        )

    def test_candidate_and_archive_exact_identities(self):
        identities = {
            str(path.relative_to(ROOT)): recorder.common.file_identity(path)
            for path in (
                recorder.CANDIDATE_BOT,
                recorder.CANDIDATE_SOURCE,
                recorder.CANDIDATE_TEST,
                recorder.CONTROL_BOT,
                recorder.CONTROL_SOURCE,
                recorder.BANK,
                recorder.ORIGINAL_PLAN,
                recorder.PLAN_V2,
                recorder.PLAN,
                recorder.AUDIT_RECEIPT,
                recorder.CONTAMINATION_RECEIPT,
                recorder.REJECTED_V1_ATTEMPT,
                recorder.REJECTED_V2_ATTEMPT,
                recorder.CONTROL_MANIFEST,
            )
        }
        recorder.validate_exact_file_identities(identities)
        bindings = recorder.require_origin_and_archive_bindings()
        self.assertEqual(
            bindings["plan_v2"]["candidate"]["exact_proof_mask"], 7
        )
        self.assertEqual(
            bindings["plan_v3"]["candidate"]["exact_proof_mask"], 7
        )
        self.assertEqual(
            bindings["audit_receipt"]["technical_audit"]["status"], "pass"
        )
        self.assertFalse(
            bindings["rejected_v1_attempt"]["development_ablation_acceptable"]
        )
        self.assertFalse(
            bindings["rejected_v2_attempt"]["development_ablation_acceptable"]
        )
        self.assertEqual(
            bindings["control_manifest"]["source_commit"],
            recorder.CONTROL_SOURCE_COMMIT,
        )

    def test_process_table_parser_is_strict(self):
        parsed = recorder.parse_process_table(
            "  10  1 /bin/zsh\n  11  10 python recorder.py\n"
        )
        self.assertEqual([item["pid"] for item in parsed], [10, 11])
        with self.assertRaisesRegex(ValueError, "malformed"):
            recorder.parse_process_table("10 only-two-fields\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            recorder.parse_process_table("10 1 a\n10 1 b\n")

    def test_process_preflight_allows_self_and_ancestors_only(self):
        clean = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 10, "ppid": 1,
             "command": "zsh record_rank4_jacek_hybrid wrapper"},
            {"pid": 11, "ppid": 10,
             "command": "python record_rank4_jacek_hybrid_null_fastpath"},
            {"pid": 12, "ppid": 1, "command": "unrelated build"},
        ], 11)
        self.assertTrue(clean["clean"])
        self.assertEqual(clean["allowed_ancestor_pids"], [1, 10, 11])

        conflict = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 11, "ppid": 1,
             "command": "python record_rank4_jacek_hybrid_null_fastpath"},
            {"pid": 99, "ppid": 1,
             "command": "/tmp/rank4-proof-cache/gate_tt"},
        ], 11)
        self.assertFalse(conflict["clean"])
        self.assertEqual([item["pid"] for item in conflict["conflicts"]], [99])

    def test_process_preflight_rejects_missing_self(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            recorder.process_preflight_from_table([
                {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            ], 11)

    def test_default_gate_keeps_rank4_mask_rejection(self):
        generic = (recorder.BOT_DIRECTORY / "comparison_gate.cpp").read_text()
        driver = (
            recorder.BOT_DIRECTORY / "comparison_gate_null_fastpath.cpp"
        ).read_text()
        self.assertIn(
            "#if !defined(PAPER_SOCCER_GATE_RANK4_SLOT_HAS_EXACT_PROOF)",
            generic,
        )
        self.assertIn(
            "#define PAPER_SOCCER_GATE_RANK4_SLOT_HAS_EXACT_PROOF", driver
        )


if __name__ == "__main__":
    unittest.main()
