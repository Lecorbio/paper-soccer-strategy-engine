import hashlib
import importlib.util
import json
import datetime as dt
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = ROOT / "tools/record_rank4_jacek_hybrid_heldout_qualification.py"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("heldout_qualification", RECORDER_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


STAGE_SWEEPS = {
    "validation": ((3, 1, 10), (3, 1, 9), (3, 1, 9), (3, 2, 8)),
    "final": ((4, 1, 22), (3, 1, 23), (3, 1, 22), (3, 2, 21)),
}


def output_line(prefix: str, fields: dict[str, str]) -> str:
    return prefix + " " + " ".join(f"{key}={value}" for key, value in fields.items())


def bank_fields(label: str, record: dict, sweeps: tuple[int, int, int]) -> dict[str, str]:
    candidate_sweeps, reference_sweeps, split_pairs = sweeps
    candidate_wins = 2 * candidate_sweeps + split_pairs
    reference_wins = 2 * reference_sweeps + split_pairs
    p0_wins = (candidate_wins + 1) // 2
    p1_wins = candidate_wins - p0_wins
    pairs = record["opening_pairs"]
    fields = {
        "bank": label,
        "games": str(record["games"]),
        "candidate_wins": str(candidate_wins),
        "reference_wins": str(reference_wins),
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": f"{p0_wins}/{pairs-p0_wins}/0/0/{pairs}",
        "candidate_p1": f"{p1_wins}/{pairs-p1_wins}/0/0/{pairs}",
        "candidate_sweeps": str(candidate_sweeps),
        "reference_sweeps": str(reference_sweeps),
        "split_pairs": str(split_pairs),
        "unresolved_pairs": "0",
    }
    for engine in ("candidate", "reference"):
        fields.update({
            f"{engine}_invocations": "100",
            f"{engine}_searches": "100",
            f"{engine}_illegal": "0",
            f"{engine}_operational": "0",
            f"{engine}_exceptions": "0",
            f"{engine}_hard_timeouts": "0",
            f"{engine}_soft_overruns": "20",
            f"{engine}_nodes": "10000",
            f"{engine}_nodes_avg": "100.000",
            f"{engine}_nodes_p99": "200",
            f"{engine}_nodes_max": "300",
            f"{engine}_depth_avg": "4.000",
            f"{engine}_depth_max": "5",
            f"{engine}_attempted_depth_avg": "5.000",
            f"{engine}_attempted_depth_max": "6",
            f"{engine}_exhaustions": "50",
            f"{engine}_first_ms_p99": "800.100",
            f"{engine}_first_ms_max": "800.200",
            f"{engine}_later_ms_p99": "165.100",
            f"{engine}_later_ms_max": "165.200",
        })
    fields.update({
        "candidate_proof_root": "10/2/1",
        "candidate_proof_leaf": "20/3/1",
        "candidate_proof_ply1": "5/1/1/2",
        "candidate_proof_ply2": "0/0/0/0",
        "candidate_proof_rebound": "35/6/3",
        "reference_proof_root": "0/0/0",
        "reference_proof_leaf": "0/0/0",
        "reference_proof_ply1": "0/0/0/0",
        "reference_proof_ply2": "0/0/0/0",
        "reference_proof_rebound": "0/0/0",
    })
    return fields


def aggregate_fields(banks: list[dict[str, str]]) -> dict[str, str]:
    result = dict(banks[0])
    result["bank"] = "all"
    for key in ("games", "candidate_wins", "reference_wins", "unfinished", "failed"):
        result[key] = str(sum(int(bank[key]) for bank in banks))
    for key in recorder.PAIR_FIELDS:
        result[key] = str(sum(int(bank[key]) for bank in banks))
    for color in range(2):
        components = [tuple(int(value) for value in bank[f"candidate_p{color}"].split("/")) for bank in banks]
        result[f"candidate_p{color}"] = "/".join(
            str(sum(item[index] for item in components)) for index in range(5)
        )
    for engine in ("candidate", "reference"):
        for suffix in recorder.development.ENGINE_ADDITIVE_FIELDS:
            key = f"{engine}_{suffix}"
            result[key] = str(sum(int(bank[key]) for bank in banks))
        for suffix in recorder.development.ENGINE_INTEGER_MAX_FIELDS:
            key = f"{engine}_{suffix}"
            result[key] = str(max(int(bank[key]) for bank in banks))
        for phase in recorder.development.TIMING_PHASES:
            key = f"{engine}_{phase}_ms_max"
            result[key] = format(max(float(bank[key]) for bank in banks), ".3f")
        for scope in ("rebound", "root", "leaf", "ply1", "ply2"):
            key = f"{engine}_proof_{scope}"
            parsed = [tuple(int(value) for value in bank[key].split("/")) for bank in banks]
            result[key] = "/".join(
                str(sum(item[index] for item in parsed))
                for index in range(len(parsed[0]))
            )
    return result


def valid_stdout(plan: dict, stage: str) -> str:
    banks = [
        bank_fields(str(index), record, STAGE_SWEEPS[stage][index])
        for index, record in enumerate(plan["banks"][stage])
    ]
    aggregate = aggregate_fields(banks)
    return "\n".join([
        *(output_line("bank_summary", bank) for bank in banks),
        output_line("summary", aggregate),
        output_line("configuration", recorder.expected_configuration(plan, stage)),
    ]) + "\n"


class HeldoutQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = recorder.validate_plan()

    def test_plan_registry_and_thresholds_are_exact(self):
        self.assertEqual(sum(item["games"] for item in self.plan["banks"]["validation"]), 106)
        self.assertEqual(sum(item["games"] for item in self.plan["banks"]["final"]), 212)
        self.assertEqual(self.plan["paired_sweep_test"]["equivalent_rational_max_inclusive"], "1/40")
        dependency_labels = {
            str(path.relative_to(ROOT)) for path in recorder.tracked_dependencies()
        }
        self.assertIn(
            "tests/codingame/test_rank4_jacek_hybrid_heldout_qualification.py",
            dependency_labels,
        )
        self.assertIn(
            "tests/codingame/test_rank4_jacek_hybrid_final_source_preflight.py",
            dependency_labels,
        )
        for item in (
            *self.plan["banks"]["validation"], *self.plan["banks"]["final"],
        ):
            self.assertNotIn(item["path"], dependency_labels)

    def test_plan_validation_never_reads_a_tsv(self):
        original = Path.read_bytes

        def guarded(path: Path):
            if path.suffix == ".tsv":
                raise AssertionError(f"sealed TSV read: {path}")
            return original(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            recorder.validate_plan()

    def test_validation_command_cannot_address_final(self):
        command = recorder.command_for_stage(self.plan, "validation")
        joined = " ".join(command)
        self.assertIn("validation_d04.tsv", joined)
        self.assertNotIn("final_d", joined)
        self.assertEqual(command.count("7"), 1)
        self.assertEqual(command.count("0"), 1)
        self.assertNotIn("--retain-transcripts", command)

    def test_final_uses_test_gate_role_and_same_configuration(self):
        command = recorder.command_for_stage(self.plan, "final")
        self.assertEqual(command[command.index("--expected-role") + 1], "test")
        configuration = recorder.expected_configuration(self.plan, "final")
        self.assertEqual(configuration["candidate_clock"], "800/165")
        self.assertEqual(configuration["reference_clock"], "800/165")
        self.assertEqual(configuration["operational_clock"], "1000/200")
        self.assertEqual(configuration["replay_corrections"], "disabled")
        self.assertEqual(configuration["transcripts"], "not-retained")

    def test_valid_stage_outputs_reconcile_explicit_sweeps(self):
        validation = recorder.validate_stage_stdout(
            self.plan, "validation", valid_stdout(self.plan, "validation")
        )
        final = recorder.validate_stage_stdout(
            self.plan, "final", valid_stdout(self.plan, "final")
        )
        self.assertEqual(validation["sweeps"], {
            "candidate_sweeps": 12, "reference_sweeps": 5,
            "split_pairs": 36, "unresolved_pairs": 0,
        })
        self.assertEqual(final["aggregate"]["candidate_wins"], "114")
        self.assertEqual(recorder.stage_threshold_errors(
            "validation", validation["aggregate"]), [])
        self.assertEqual(recorder.stage_threshold_errors(
            "final", final["aggregate"]), [])

    def test_sweeps_are_not_inferred_from_aggregate_wins(self):
        broken = valid_stdout(self.plan, "validation").replace(
            "candidate_sweeps=3", "candidate_sweeps=2", 1
        )
        with self.assertRaisesRegex(ValueError, "sweep|wins"):
            recorder.validate_stage_stdout(self.plan, "validation", broken)
        missing = valid_stdout(self.plan, "validation").replace(
            " candidate_sweeps=3", "", 1
        )
        with self.assertRaisesRegex(ValueError, "field set"):
            recorder.validate_stage_stdout(self.plan, "validation", missing)
        extra = valid_stdout(self.plan, "validation").replace(
            "bank_summary ", "bank_summary extra_secret=forbidden ", 1
        )
        with self.assertRaisesRegex(ValueError, "field set"):
            recorder.validate_stage_stdout(self.plan, "validation", extra)

    def test_unresolved_pair_and_nonzero_failure_are_rejected(self):
        unresolved = valid_stdout(self.plan, "validation").replace(
            "unresolved_pairs=0", "unresolved_pairs=1", 1
        )
        with self.assertRaises(ValueError):
            recorder.validate_stage_stdout(self.plan, "validation", unresolved)
        failed = valid_stdout(self.plan, "validation").replace(
            "candidate_wins=16 reference_wins=12 unfinished=0 failed=0",
            "candidate_wins=15 reference_wins=12 unfinished=0 failed=1",
            1,
        )
        with self.assertRaisesRegex(ValueError, "unfinished|failed"):
            recorder.validate_stage_stdout(self.plan, "validation", failed)

    def test_timing_thresholds_are_strict_for_both_engines(self):
        cases = (
            ("candidate_first_ms_p99=800.100", "candidate_first_ms_p99=900.000"),
            ("reference_later_ms_max=165.200", "reference_later_ms_max=198.000"),
            ("reference_first_ms_p99=800.100", "reference_first_ms_p99=nan"),
        )
        for old, new in cases:
            with self.subTest(new=new), self.assertRaises(ValueError):
                recorder.validate_stage_stdout(
                    self.plan, "validation",
                    valid_stdout(self.plan, "validation").replace(old, new),
                )

    def test_proof_accounting_requires_mask7_and_zero_ply2(self):
        broken = valid_stdout(self.plan, "validation").replace(
            "candidate_proof_ply2=0/0/0/0",
            "candidate_proof_ply2=1/0/0/0", 1,
        )
        with self.assertRaisesRegex(ValueError, "disabled candidate ply2"):
            recorder.validate_stage_stdout(self.plan, "validation", broken)
        broken = valid_stdout(self.plan, "validation").replace(
            "candidate_proof_rebound=35/6/3",
            "candidate_proof_rebound=35/5/3", 1,
        )
        with self.assertRaisesRegex(ValueError, "rebound/scope"):
            recorder.validate_stage_stdout(self.plan, "validation", broken)

    def test_exact_sign_test_uses_rational_one_sided_tail(self):
        passing = recorder.exact_one_sided_sign_test(25, 10)
        failing = recorder.exact_one_sided_sign_test(20, 15)
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertLessEqual(
            passing["p_numerator"] * 40, passing["p_denominator"]
        )

    def test_pooled_gate_is_conjunctive(self):
        validation_parsed = recorder.validate_stage_stdout(
            self.plan, "validation", valid_stdout(self.plan, "validation")
        )
        final_parsed = recorder.validate_stage_stdout(
            self.plan, "final", valid_stdout(self.plan, "final")
        )
        validation_report = {"parsed": validation_parsed}
        final_report = {"parsed": final_parsed}
        pooled = recorder.pooled_evaluation(validation_report, final_report)
        self.assertTrue(pooled["acceptable"])
        self.assertEqual(pooled["candidate_wins"], 174)
        self.assertEqual(pooled["candidate_wins_by_physical_color"], [89, 85])
        self.assertEqual(pooled["sweeps"]["candidate_sweeps"], 25)
        self.assertEqual(pooled["sweeps"]["reference_sweeps"], 10)

        final_parsed["aggregate"]["candidate_sweeps"] = "28"
        final_parsed["aggregate"]["reference_sweeps"] = "20"
        final_parsed["aggregate"]["split_pairs"] = "58"
        rejected = recorder.pooled_evaluation(validation_report, final_report)
        self.assertFalse(rejected["acceptable"])

    def test_atomic_claim_is_spent_before_bank_access_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "OUTPUT", Path(directory)
        ):
            path, payload = recorder.create_stage_claim(
                "a" * 64, "validation", "b" * 64
            )
            self.assertTrue(path.is_file())
            self.assertTrue(payload["claim_precedes_first_stage_bank_byte"])
            with self.assertRaisesRegex(ValueError, "spent"):
                recorder.create_stage_claim(
                    "a" * 64, "validation", "b" * 64
                )

    def test_old_report_without_provenance_is_rejected(self):
        binding = {
            "candidate_qualification_id": "a" * 64,
            "dependency_identities": {},
        }
        with self.assertRaisesRegex(ValueError, "identity"):
            recorder.validate_persisted_stage_report(
                {
                    "schema": recorder.REPORT_SCHEMA,
                    "candidate_qualification_id": "a" * 64,
                    "binding_sha256": "b" * 64,
                    "stage": "validation",
                    "stage_acceptable": True,
                },
                binding, "b" * 64, self.plan, "validation",
            )

    def test_complete_report_revalidates_and_unknown_fields_are_rejected(self):
        stdout = valid_stdout(self.plan, "validation")
        parsed = recorder.validate_stage_stdout(self.plan, "validation", stdout)
        identifier, binding_sha256 = "a" * 64, "b" * 64
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "ROOT", Path(directory)
        ), mock.patch.object(recorder, "OUTPUT", Path(directory) / "out"):
            producer_label = recorder.identity_label(recorder.RECORDER)
            producer = {
                "path": producer_label, "bytes": 1, "sha256": "c" * 64,
                "ascii": True, "mode": "0444", "executable": True,
            }
            compiler_records = {
                "clang": {"family": "Clang"},
                "gnu": {"family": "GNU"},
            }
            host = {"sha256": "d" * 64}
            runtime = {"python_version": "synthetic"}
            binding = {
                "candidate_qualification_id": identifier,
                "candidate_commit": "e" * 40,
                "dependency_identities": {producer_label: producer},
                "compiler_records": compiler_records,
                "host": host,
                "runtime": runtime,
            }
            inputs = {producer_label: producer}
            for record in self.plan["banks"]["validation"]:
                inputs[record["path"]] = {
                    "path": record["path"], "bytes": record["bytes"],
                    "sha256": record["sha256"], "ascii": True,
                    "mode": "0444", "executable": False,
                }
            claim_path, claim = recorder.create_stage_claim(
                identifier, "validation", binding_sha256
            )
            claimed = recorder.parse_utc(claim["claimed_utc"])
            started = (claimed + dt.timedelta(seconds=1)).isoformat()
            ended = (claimed + dt.timedelta(seconds=2)).isoformat()
            git = {
                "head": "e" * 40,
                "author_utc": "2026-08-14T09:00:00+00:00",
                "committer_utc": "2026-08-14T09:00:01+00:00",
                "tracked_status": "",
            }
            process = {
                "clean": True, "self_pid": 1,
                "allowed_ancestor_pids": [1], "observed_process_count": 1,
                "conflicts": [], "markers": list(recorder.PROCESS_MARKERS),
                "checked_utc": claim["claimed_utc"],
                "command": ["/bin/ps", "-axo", "pid=,ppid=,command="],
            }
            command = recorder.command_for_stage(self.plan, "validation")
            report = {
                "schema": recorder.REPORT_SCHEMA,
                "campaign_id": recorder.CAMPAIGN_ID,
                "campaign_t0_utc": recorder.CAMPAIGN_T0_UTC,
                "classification": "untouched-validation-one-shot-qualification-stage",
                "final_qualification": False,
                "producer": producer,
                "candidate_qualification_id": identifier,
                "binding_sha256": binding_sha256,
                "stage": "validation",
                "claim": {**claim, "path": recorder.identity_label(claim_path)},
                "started_utc": started, "ended_utc": ended,
                "elapsed_monotonic_ns": 1_000_000_000,
                "command_argv": command,
                "command_shell": recorder.shlex.join(command),
                "cwd": str(recorder.ROOT),
                "timeout_seconds": recorder.STAGE_TIMEOUT_SECONDS["validation"],
                "environment": recorder.preflight.environment_record(),
                "host_before": host, "host_after": host,
                "runtime": runtime, "returncode": 0, "timed_out": False,
                "os_error_class": None,
                "stdout": recorder._stream_evidence(stdout),
                "stderr": recorder._stream_evidence(""),
                "process_preflight": process,
                "git_before": git, "git_after": git,
                "inputs_before": inputs, "inputs_after": inputs,
                "stable_inputs": True,
                "compiler_records_before": compiler_records,
                "compiler_records_after": compiler_records,
                "stable_compilers": True,
                "accessed_bank_paths": [
                    item["path"] for item in self.plan["banks"]["validation"]
                ],
                "parsed": parsed, "validation_codes": [],
                "threshold_errors": [], "stage_acceptable": True,
                "replay_corrections": "disabled",
                "transcripts": "not-retained",
            }
            with mock.patch.object(
                recorder, "_path_from_label", return_value=Path(directory) / "dep"
            ), mock.patch.object(recorder, "identities", return_value=inputs):
                recorder.validate_persisted_stage_report(
                    report, binding, binding_sha256, self.plan, "validation"
                )
                malformed_cases = []
                false_returncode = copy.deepcopy(report)
                false_returncode["returncode"] = False
                malformed_cases.append(false_returncode)
                bool_elapsed = copy.deepcopy(report)
                bool_elapsed["elapsed_monotonic_ns"] = True
                malformed_cases.append(bool_elapsed)
                bool_stream_bytes = copy.deepcopy(report)
                bool_stream_bytes["stderr"]["bytes"] = False
                malformed_cases.append(bool_stream_bytes)
                bool_process_pid = copy.deepcopy(report)
                bool_process_pid["process_preflight"]["self_pid"] = True
                bool_process_pid["process_preflight"][
                    "allowed_ancestor_pids"
                ] = [True]
                malformed_cases.append(bool_process_pid)
                for malformed in malformed_cases:
                    with self.subTest(field=malformed), self.assertRaises(ValueError):
                        recorder.validate_persisted_stage_report(
                            malformed, binding, binding_sha256,
                            self.plan, "validation",
                        )
            report["extra_secret"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "identity"):
                recorder.validate_persisted_stage_report(
                    report, binding, binding_sha256, self.plan, "validation"
                )

    def test_rejected_validation_requires_absent_final_claim_and_reports(self):
        identifier = "a" * 64
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "OUTPUT", Path(directory) / "out"
        ):
            recorder.require_final_attempt_unopened(identifier)
            recorder.create_stage_claim(identifier, "final", "b" * 64)
            with self.assertRaisesRegex(ValueError, "FINAL claim"):
                recorder.require_final_attempt_unopened(identifier)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "OUTPUT", Path(directory) / "out"
        ):
            reports = recorder.OUTPUT / "reports/final"
            reports.mkdir(parents=True)
            (reports / ("c" * 64 + ".json")).write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "FINAL report"):
                recorder.require_final_attempt_unopened(identifier)

    def test_content_addressed_report_round_trip_is_exact(self):
        payload = {"schema": "synthetic", "values": [3, 2, 1]}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "ROOT", Path(directory)
        ):
            path, digest = recorder.persist_content_addressed(
                Path(directory), payload
            )
            self.assertEqual(path.name, f"{digest}.json")
            self.assertEqual(path.read_bytes(), recorder.canonical_json(payload))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            loaded, loaded_digest = recorder.load_canonical_content_addressed(
                path, "synthetic", Path(directory)
            )
            self.assertEqual(loaded, payload)
            self.assertEqual(loaded_digest, digest)

    def test_process_preflight_rejects_competing_clock_job(self):
        clean = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 10, "ppid": 1, "command": "zsh heldout wrapper"},
            {"pid": 11, "ppid": 10, "command": "python rank4-jacek-hybrid"},
        ], 11)
        self.assertTrue(clean["clean"])
        conflict = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 11, "ppid": 1, "command": "python rank4-jacek-hybrid"},
            {"pid": 99, "ppid": 1,
             "command": "papersoccer_codingame_rank_4 competing"},
        ], 11)
        self.assertFalse(conflict["clean"])

    def test_source_orders_claim_before_bank_paths_and_final_after_validation(self):
        source = RECORDER_PATH.read_text(encoding="utf-8")
        run_stage_source = source[source.index("def run_stage("):source.index(
            "def find_decisions(")]
        self.assertLess(
            run_stage_source.index("claim, claim_payload = create_stage_claim"),
            run_stage_source.index("bank_paths = [ROOT / item"),
        )
        run_source = source[source.index("def run_qualification("):source.index(
            "def main()")]
        self.assertLess(
            run_source.index('if not validation[2]["stage_acceptable"]'),
            run_source.index('final = _one_report_or_none'),
        )
        self.assertIn("/tmp/rank4-hybrid-prototype-benchmark.lock", source)

    def test_cli_has_no_arbitrary_path_arguments(self):
        source = RECORDER_PATH.read_text(encoding="ascii")
        self.assertNotIn('add_argument("--preflight"', source)
        self.assertNotIn('add_argument("--binding"', source)
        self.assertIn("fixed_preflight_receipt", source)
        self.assertIn("fixed_binding_path", source)

    def test_sealed_path_is_rejected_before_any_stat_or_resolve(self):
        forbidden = ROOT / "results/rank_4_jacek_hybrid/openings/final_d04.tsv"
        with mock.patch.object(
            Path, "is_symlink", side_effect=AssertionError("stat attempted")
        ):
            with self.assertRaisesRegex(ValueError, "forbidden|sealed"):
                recorder.guard_read_path(forbidden)

    def test_decision_is_recomputed_and_rejected_validation_never_loads_final(self):
        producer = {
            "path": recorder.identity_label(recorder.RECORDER),
            "bytes": 1, "sha256": "c" * 64, "ascii": True,
            "mode": "0444", "executable": True,
        }
        binding = {
            "candidate_qualification_id": "a" * 64,
            "dependency_identities": {
                recorder.identity_label(recorder.RECORDER): producer,
            },
        }
        validation_report = {
            "stage_acceptable": False,
            "ended_utc": "2026-08-14T10:00:00+00:00",
        }
        validation = (
            recorder.OUTPUT / "reports/validation" / ("c" * 64 + ".json"),
            "c" * 64,
            validation_report,
        )
        created = "2026-08-14T10:00:01+00:00"
        decision = recorder.decision_payload(
            binding, "b" * 64, validation, None, created
        )
        self.assertFalse(decision["arena_authorization"])
        self.assertIsNone(decision["final_report"])
        with mock.patch.object(
            recorder, "_load_decision_report", return_value=validation
        ) as loader:
            recorder.validate_persisted_decision(
                decision, binding, "b" * 64, self.plan
            )
            loader.assert_called_once()
        forged = dict(decision)
        forged["arena_authorization"] = True
        with mock.patch.object(
            recorder, "_load_decision_report", return_value=validation
        ), self.assertRaisesRegex(ValueError, "recomputed"):
            recorder.validate_persisted_decision(
                forged, binding, "b" * 64, self.plan
            )

    def test_decision_report_path_cannot_redirect_to_a_bank(self):
        reference = {
            "stage": "validation",
            "path": "results/rank_4_jacek_hybrid/openings/validation_d04.tsv",
            "sha256": "c" * 64,
            "acceptable": False,
        }
        with mock.patch.object(
            recorder, "load_canonical_content_addressed",
            side_effect=AssertionError("unexpected read"),
        ), self.assertRaisesRegex(ValueError, "not fixed"):
            recorder._load_decision_report(
                reference, "validation",
                {"candidate_qualification_id": "a" * 64},
                "b" * 64, self.plan,
            )

    def test_malformed_existing_decision_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            recorder, "OUTPUT", Path(directory)
        ), mock.patch.object(recorder, "ROOT", Path(directory)):
            registry = Path(directory) / "decisions"
            registry.mkdir()
            (registry / ("a" * 64 + ".json")).write_text(
                '{"schema":"rank4-jacek-hybrid-heldout-decision-v1"}\n',
                encoding="ascii",
            )
            with self.assertRaises(ValueError):
                recorder.find_decisions("b" * 64, "c" * 64)


if __name__ == "__main__":
    unittest.main()
