#!/usr/bin/env python3
"""Synthetic/static tests for the DEVELOPMENT-only mask-3 removal recorder."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = ROOT / "tools/record_rank4_jacek_hybrid_mask3_removal_clock.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("mask3_removal_clock_tested", RECORDER_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


CHECKED = "2026-08-14T12:20:00+00:00"
CLAIMED = "2026-08-14T12:20:01+00:00"
STARTED = "2026-08-14T12:20:02+00:00"
ENDED = "2026-08-14T12:20:03+00:00"
REPORTED = "2026-08-14T12:20:04+00:00"


def process_record(checked: str = CHECKED) -> dict[str, object]:
    return {
        "checked_utc": checked,
        "clean": True,
        "conflicts": [],
        "observed_process_count": 100,
        "tool": {
            "ascii": False,
            "bytes": recorder.PS_BYTES,
            "executable": True,
            "mode": "4755",
            "path": str(recorder.PS),
            "sha256": recorder.PS_SHA256,
        },
    }


def evidence(*, checked: str = CHECKED) -> dict[str, object]:
    return {
        "plan_sha256": recorder.PLAN_SHA256,
        "admin": {"admin_commit": "b" * 40},
        "historical": {"fixed": "c" * 64},
        "sources": {"source": {"sha256": "d" * 64}},
        "closure": {"closure_sha256": "e" * 64},
        "host": {"sha256": "f" * 64},
        "runtime": {"python_version": "fixed"},
        "host_inspector": {"sha256": "1" * 64},
        "environment_sha256": "2" * 64,
        "qualification_key": {"schema": "synthetic"},
        "candidate_identity": "a" * 64,
        "process": process_record(checked),
    }


def proof_fields(engine: str, mask: int, multiplier: int) -> dict[str, str]:
    result: dict[str, str] = {}
    sums = [0, 0, 0]
    for scope, bit in (("root", 1), ("leaf", 2), ("ply1", 4), ("ply2", 8)):
        if mask & bit:
            counters = [10 * multiplier, multiplier, multiplier]
        else:
            counters = [0, 0, 0]
        sums = [left + right for left, right in zip(sums, counters)]
        if scope in ("ply1", "ply2"):
            result[f"{engine}_proof_{scope}"] = "/".join(
                map(str, (*counters, counters[1] + counters[2]))
            )
        else:
            result[f"{engine}_proof_{scope}"] = "/".join(map(str, counters))
    result[f"{engine}_proof_rebound"] = "/".join(map(str, sums))
    return result


def engine_fields(engine: str, games: int, mask: int, multiplier: int) -> dict[str, str]:
    return {
        f"{engine}_invocations": str(games),
        f"{engine}_searches": str(games),
        f"{engine}_illegal": "0",
        f"{engine}_operational": "0",
        f"{engine}_exceptions": "0",
        f"{engine}_hard_timeouts": "0",
        f"{engine}_soft_overruns": "0",
        f"{engine}_nodes": str(games * 100),
        f"{engine}_nodes_avg": "100.000",
        f"{engine}_nodes_p99": "100",
        f"{engine}_nodes_max": "100",
        f"{engine}_depth_avg": "5.000",
        f"{engine}_depth_max": "5",
        f"{engine}_attempted_depth_avg": "6.000",
        f"{engine}_attempted_depth_max": "6",
        f"{engine}_exhaustions": "0",
        f"{engine}_first_ms_p99": "10.000",
        f"{engine}_first_ms_max": "20.000",
        f"{engine}_later_ms_p99": "5.000",
        f"{engine}_later_ms_max": "10.000",
        **proof_fields(engine, mask, multiplier),
    }


def summary_fields(stage: str, index: int | None) -> dict[str, str]:
    spec = recorder.stage_spec(stage)
    if index is None:
        bank = "all"
        games = 306
        candidate_wins = 160
        reference_wins = 146
        candidate_color = 80
        reference_color = 73
        multiplier = 4
    else:
        games = recorder.BANKS[index][4]
        bank = str(index)
        candidate_wins = 40
        reference_wins = games - candidate_wins
        candidate_color = 20
        reference_color = reference_wins // 2
        multiplier = 1
    fields = {
        "bank": bank,
        "games": str(games),
        "candidate_wins": str(candidate_wins),
        "reference_wins": str(reference_wins),
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": f"{candidate_color}/{reference_color}/0/0/{games // 2}",
        "candidate_p1": f"{candidate_color}/{reference_color}/0/0/{games // 2}",
        **engine_fields("candidate", games, 3, multiplier),
        **engine_fields(
            "reference", games, int(spec["reference_exact_proof_mask"]), multiplier
        ),
    }
    assert frozenset(fields) == recorder.EXPECTED_SUMMARY_FIELDS
    return fields


def fields_line(prefix: str, fields: dict[str, str]) -> str:
    return prefix + " " + " ".join(f"{key}={fields[key]}" for key in sorted(fields))


def valid_stdout(stage: str) -> bytes:
    lines = [
        fields_line("bank_summary", summary_fields(stage, index))
        for index in range(4)
    ]
    lines.append(fields_line("summary", summary_fields(stage, None)))
    lines.append(fields_line("configuration", recorder.configuration_expected(stage)))
    return ("\n".join(lines) + "\n").encode("ascii")


def execution_result(stage: str, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "started_utc": STARTED,
        "ended_utc": ENDED,
        "elapsed_monotonic_ns": 1_000_000_000,
        "returncode": 0,
        "timed_out": False,
        "os_error_class": None,
        "stdout": valid_stdout(stage),
        "stderr": b"",
    }
    result.update(overrides)
    return result


@contextmanager
def runtime_registries():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mask3-recorder-test-", dir=build) as name:
        output = Path(name)
        replacements = {
            "OUTPUT": output,
            "LOCK": output / ".recorder.lock",
            "CLAIMS": output / "claims",
            "EXECUTIONS": output / "executions",
            "REPORTS": output / "reports",
            "DECISIONS": output / "decisions",
        }
        with mock.patch.multiple(recorder, **replacements):
            recorder.prepare_runtime_directories()
            yield output


def create_claim(stage: str, current: dict[str, object], prior=()):
    with mock.patch.object(recorder, "utc_now", return_value=CLAIMED):
        return recorder.create_claim(current, stage, list(prior))


def create_execution(stage: str, current: dict[str, object], claim, **overrides):
    return recorder.persist_execution(
        current, stage, claim, execution_result(stage, **overrides)
    )


def create_report(stage: str, current: dict[str, object], claim, execution,
                  *, after=None, postflight_error=None, created=REPORTED):
    if after is None and postflight_error is None:
        after = recorder.stable_evidence(current)
    payload = recorder.report_payload(
        current, claim, execution, after,
        postflight_error=postflight_error, created_utc=created,
    )
    path, digest = recorder.persist_content_addressed(recorder.REPORTS, payload)
    return path, digest, payload


class Mask3RemovalClockTests(unittest.TestCase):
    def test_plan_is_canonical_and_binds_three_stage_protocol(self):
        plan = recorder.load_plan()
        self.assertEqual(recorder.canonical_json(plan), recorder.PLAN.read_bytes())
        self.assertEqual(
            [item["stage"] for item in plan["stages"]], list(recorder.STAGE_NAMES)
        )
        self.assertEqual(plan["thresholds_each_stage"]["exact_games"], 306)
        self.assertEqual(plan["thresholds_each_stage"]["candidate_wins_min"], 160)
        self.assertEqual(
            plan["one_shot_policy"]["exclusive_locks"],
            [
                str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                for path in recorder.CAMPAIGN_LOCKS
            ],
        )
        self.assertEqual(
            plan["binary_provenance"]["host_runtime"]["frozen_host_sha256"],
            "1a7f59560af8acc4bc4533679ffc1fe83a835bf979928bb47909c7cbffbed30c",
        )
        self.assertEqual(plan["execution_environment"]["values"],
                         recorder.gate_environment())
        self.assertEqual(
            plan["execution_environment"]["sha256"],
            recorder.sha256_bytes(recorder.canonical_json(recorder.gate_environment())),
        )
        self.assertIn("clock-rollback", plan["decision_policy"]["wall_clock_rollback"])
        self.assertIn("single-link regular files", plan["one_shot_policy"]["durability"])
        self.assertIn("O_RDONLY|O_NOFOLLOW", plan["one_shot_policy"]["durability"])
        self.assertIn("interrupted-postflight", plan["one_shot_policy"][
            "postflight_failure"])
        self.assertIn("GIT_OPTIONAL_LOCKS=0", plan["evidence_boundary"][
            "git_read_policy"])
        self.assertEqual(
            plan["producer_bootstrap_identities"][
                "tools/record_rank4_jacek_hybrid_full_development_clock.py"
            ]["sha256"],
            "58e72685151f86009b5a682c49363d3dc3ae11a151d15b88906130c4505f251e",
        )

    def test_commands_are_exact_development_only_stage_commands(self):
        expected = (("hybrid-control", "7"), ("hybrid-control", "0"),
                    ("rank4", "0"))
        for stage, (reference, reference_mask) in zip(recorder.STAGE_NAMES, expected):
            command = recorder.command_for(stage)
            self.assertEqual(command[0], str(recorder.GATE))
            self.assertEqual(command[command.index("--reference-engine") + 1], reference)
            self.assertEqual(
                command[command.index("--reference-exact-proof-mask") + 1],
                reference_mask,
            )
            self.assertEqual(
                command[command.index("--candidate-exact-proof-mask") + 1], "3"
            )
            banks = [command[index + 1] for index, token in enumerate(command)
                     if token == "--bank"]
            self.assertEqual(banks, [str(path) for path in recorder.BANK_PATHS])
            self.assertTrue(all("development_" in path for path in banks))

    def test_protected_banks_and_build_commands_are_not_addressable(self):
        source = RECORDER_PATH.read_text()
        for forbidden in (
            "validation_d04.tsv", "validation_d08.tsv", "validation_d12.tsv",
            "validation_d20.tsv", "final_d04.tsv", "final_d08.tsv",
            "final_d12.tsv", "final_d20.tsv", "cmake --build",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("subprocess.run([\"cmake\"", source)

    def test_live_frozen_binary_host_and_historical_evidence(self):
        if recorder.sys.platform != "darwin" or not recorder.GATE.is_file():
            self.skipTest("exact frozen macOS DEVELOPMENT binary is local-only")
        host_runtime = recorder.host_runtime_identity()
        self.assertEqual(
            host_runtime["host"]["sha256"],
            "1a7f59560af8acc4bc4533679ffc1fe83a835bf979928bb47909c7cbffbed30c",
        )
        self.assertEqual(
            recorder.file_identity(recorder.GATE)["sha256"], recorder.GATE_SHA256
        )
        history = recorder.validate_historical_evidence()
        self.assertEqual(history["closed_heldout_decision"], recorder.HISTORICAL_JSON[
            "closed_heldout_decision"][1])
        with self.assertRaisesRegex(
            ValueError,
            r"preserved closure identity drift: .*comparison_gate\.cpp",
        ):
            recorder.preserved_binary_closure()

        binding = recorder.load_tracked_canonical_json(
            recorder.FROZEN_BINDING,
            recorder.FROZEN_BINDING_SHA256,
            live_mode=0o444,
        )
        receipt = recorder.load_tracked_canonical_json(
            recorder.PREFLIGHT_RECEIPT,
            recorder.PREFLIGHT_RECEIPT_SHA256,
        )
        frozen_identities = dict(binding["dependency_identities"])
        frozen_identities.update(
            receipt["builds"]["clang_release"]["build_artifacts"]
        )
        comparison_gate_source = recorder.ROOT / (
            "submissions/codingame/bots/rank_4_jacek_hybrid/"
            "comparison_gate.cpp"
        )
        comparison_gate_label = recorder.path_label(comparison_gate_source)
        historical_comparison_gate = frozen_identities[comparison_gate_label]
        live_file_identity = recorder.file_identity

        def historical_source_identity(path: Path, **kwargs) -> dict:
            if path == comparison_gate_source:
                return copy.deepcopy(historical_comparison_gate)
            return live_file_identity(path, **kwargs)

        with mock.patch.object(
            recorder, "file_identity", side_effect=historical_source_identity
        ):
            closure = recorder.preserved_binary_closure()
        self.assertEqual(closure["entries"], 1056)
        self.assertEqual(
            closure["closure_sha256"], recorder.EXPECTED_CLOSURE_SHA256
        )
        self.assertEqual(closure["binary"]["sha256"], recorder.GATE_SHA256)
        with mock.patch.object(recorder.platform, "machine", return_value="drifted"):
            with self.assertRaisesRegex(ValueError, "host/runtime"):
                recorder.host_runtime_identity()

    def test_otool_exception_remains_exact_hash_mode_and_size_bound(self):
        wrong = {
            "ascii": False, "bytes": recorder.OTOOL_BYTES,
            "executable": True, "mode": "0644", "path": str(recorder.OTOOL),
            "sha256": recorder.OTOOL_SHA256,
        }
        with mock.patch.object(recorder, "file_identity", return_value=wrong), \
             mock.patch.object(recorder.subprocess, "run") as run, \
             self.assertRaisesRegex(ValueError, "tool identity"):
            recorder.runtime_linkage()
        run.assert_not_called()

    def test_normal_gate_schema_matches_both_immutable_reports(self):
        for digest in (
            "8f7aa959b54843baad13333e3023d43c852be1e11296bba0e5b3ac8524aa1fa9",
            "cd259e7053467a01a87d0b79c88b2fb036eb9273c57d82a0b893df8738b21cf1",
        ):
            path = ROOT / f"results/rank_4_jacek_hybrid/gates/full_development_clock/{digest}.json"
            payload = json.loads(path.read_text())
            parsed = payload["parsed"]
            for fields in (*parsed["banks"], parsed["aggregate"]):
                self.assertEqual(frozenset(fields), recorder.EXPECTED_SUMMARY_FIELDS)
            self.assertEqual(
                frozenset(parsed["configuration"]),
                recorder.EXPECTED_CONFIGURATION_FIELDS,
            )
        self.assertEqual(len(recorder.EXPECTED_SUMMARY_FIELDS), 58)
        self.assertFalse({"candidate_sweeps", "reference_sweeps", "split_pairs",
                          "unresolved_pairs"} & recorder.EXPECTED_SUMMARY_FIELDS)

    def test_producer_bootstrap_modules_are_exact_c807_head_blobs(self):
        plan = json.loads(recorder.PLAN.read_text())
        identities = recorder.producer_bootstrap_identities(plan)
        self.assertEqual(set(identities), set(recorder.PRODUCER_BOOTSTRAP_IDENTITIES))
        for label, identity in identities.items():
            self.assertEqual(
                identity["sha256"],
                recorder.PRODUCER_BOOTSTRAP_IDENTITIES[label]["sha256"],
            )
            self.assertEqual(identity["mode"], "0644")

    def test_valid_six_line_output_parses_for_all_stages(self):
        for stage in recorder.STAGE_NAMES:
            stream = recorder._stdout_receipt(valid_stdout(stage), stage)
            self.assertTrue(stream["retained"])
            execution = {
                "stdout": stream,
                "stderr": recorder._stream_metadata(b""),
                "returncode": 0,
                "timed_out": False,
                "os_error_class": None,
            }
            parsed, validation, thresholds = recorder.parse_execution_output(
                stage, execution
            )
            self.assertEqual(validation, [])
            self.assertEqual(thresholds, [])
            self.assertEqual(parsed["aggregate"]["candidate_wins"], "160")

    def test_stdout_retention_rejects_any_nonexact_shape(self):
        stage = recorder.STAGE_NAMES[0]
        valid = valid_stdout(stage)
        variants = (
            valid + b"\n",
            valid.replace(b"\n", b"\r\n", 1),
            valid.replace(b"games=78", b"games=78\x00", 1),
            valid[:-1],
            valid.replace(b"games=78", b"unknown=1 games=78", 1),
            valid.replace(b" games=78", b"", 1),
            valid + b"\xff",
        )
        for raw in variants:
            with self.subTest(raw=raw[-20:]):
                receipt = recorder._stdout_receipt(raw, stage)
                self.assertFalse(receipt["retained"])
                self.assertNotIn("text", receipt)

    def test_invalid_timeout_and_stderr_streams_reject_without_raw_retention(self):
        stage = recorder.STAGE_NAMES[0]
        cases = (
            {"stdout": b"bad\n", "stderr": b""},
            {"stdout": valid_stdout(stage), "stderr": b"secret diagnostic"},
        )
        current = evidence()
        with runtime_registries():
            claim = create_claim(stage, current)
            for case in cases:
                execution = create_execution(stage, current, claim, **case)
                parsed, validation, _ = recorder.parse_execution_output(stage, execution[2])
                if case["stdout"] == b"bad\n":
                    self.assertTrue(validation)
                    self.assertEqual(parsed["banks"], [])
                self.assertNotIn("text", execution[2]["stderr"])
                if case["stderr"]:
                    report = recorder.report_payload(
                        current, claim, execution, recorder.stable_evidence(current),
                        created_utc=REPORTED,
                    )
                    self.assertFalse(report["development_selection_acceptable"])
                    self.assertIn("gate stderr is not empty", report["process_errors"])
                execution[0].unlink()
            timeout = create_execution(
                stage, current, claim, stdout=b"partial", stderr=b"timeout",
                timed_out=True, returncode=None, elapsed_monotonic_ns=3_600_000_000_000,
                ended_utc="2026-08-14T13:20:02+00:00",
            )
            report = recorder.report_payload(
                current, claim, timeout, recorder.stable_evidence(current),
                created_utc="2026-08-14T13:20:03+00:00",
            )
            self.assertFalse(report["development_selection_acceptable"])
            self.assertFalse(timeout[2]["stdout"]["retained"])

    def test_exact_work_timing_and_proof_accounting_fail_closed(self):
        stage = recorder.STAGE_NAMES[0]
        banks = [summary_fields(stage, index) for index in range(4)]
        aggregate = summary_fields(stage, None)
        recorder.validate_full_summaries(banks, aggregate, stage)
        mutations = (
            ("candidate_nodes_avg", "nan"),
            ("candidate_nodes_avg", "99.000"),
            ("candidate_nodes_p99", "101"),
            ("candidate_depth_avg", "6.000"),
            ("candidate_attempted_depth_avg", "4.000"),
            ("candidate_soft_overruns", "79"),
            ("candidate_exhaustions", "79"),
            ("candidate_first_ms_max", "990.000"),
            ("candidate_later_ms_max", "198.000"),
            ("reference_first_ms_max", "990.000"),
            ("reference_later_ms_max", "198.000"),
            ("candidate_first_ms_p99", "21.000"),
            ("reference_later_ms_p99", "nan"),
            ("candidate_proof_ply2", "1/0/0/0"),
        )
        for key, value in mutations:
            changed = copy.deepcopy(banks)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                recorder.validate_full_summaries(changed, aggregate, stage)
        for engine in ("candidate", "reference"):
            over_budget_banks = copy.deepcopy(banks)
            over_budget_aggregate = copy.deepcopy(aggregate)
            for fields in over_budget_banks:
                games = int(fields["games"])
                fields[f"{engine}_nodes"] = str(games * 3_000_001)
                fields[f"{engine}_nodes_avg"] = "3000001.000"
                fields[f"{engine}_nodes_p99"] = "3000001"
                fields[f"{engine}_nodes_max"] = "3000001"
            over_budget_aggregate[f"{engine}_nodes"] = str(306 * 3_000_001)
            over_budget_aggregate[f"{engine}_nodes_avg"] = "3000001.000"
            over_budget_aggregate[f"{engine}_nodes_p99"] = "3000001"
            over_budget_aggregate[f"{engine}_nodes_max"] = "3000001"
            with self.subTest(engine=engine, invariant="budget"), \
                 self.assertRaisesRegex(ValueError, "configured budget"):
                recorder.validate_full_summaries(
                    over_budget_banks, over_budget_aggregate, stage
                )
        impossible_banks = copy.deepcopy(banks)
        impossible_aggregate = copy.deepcopy(aggregate)
        impossible_banks[0]["candidate_nodes_max"] = "7801"
        impossible_aggregate["candidate_nodes_max"] = "7801"
        with self.assertRaisesRegex(ValueError, "total or configured"):
            recorder.validate_full_summaries(
                impossible_banks, impossible_aggregate, stage
            )

    def test_selection_thresholds_are_exact_160_and_77_per_color(self):
        aggregate = summary_fields(recorder.STAGE_NAMES[0], None)
        self.assertEqual(recorder.old_full.selection_threshold_errors(aggregate), [])
        aggregate["candidate_wins"] = "159"
        self.assertTrue(recorder.old_full.selection_threshold_errors(aggregate))
        aggregate = summary_fields(recorder.STAGE_NAMES[0], None)
        aggregate["candidate_p0"] = "76/77/0/0/153"
        self.assertTrue(recorder.old_full.selection_threshold_errors(aggregate))

    def test_claim_is_exclusive_durable_and_has_ordered_process_record(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            self.assertEqual(stat.S_IMODE(claim[0].stat().st_mode), 0o444)
            self.assertEqual(claim[2]["preclaim_process"], current["process"])
            with self.assertRaises(FileExistsError):
                create_claim(stage, current)
            bad = evidence(checked="2026-08-14T12:18:00+00:00")
            bad["candidate_identity"] = "3" * 64
            with self.assertRaises(ValueError):
                create_claim(stage, bad)
            self.assertFalse(recorder.claim_path(bad["candidate_identity"], stage).exists())

    def test_claim_precedes_gate_and_last_check_is_persisted(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        events: list[str] = []
        claim = (ROOT / "build/claim.json", "1" * 64,
                 {"stage": stage, "preclaim_process": current["process"]})
        execution = (ROOT / "build/execution.json", "2" * 64, {"stage": stage})
        report = (ROOT / "build/report.json", "3" * 64,
                  {"stage": stage, "development_selection_acceptable": True})
        refreshed = copy.deepcopy(current)
        with mock.patch.object(recorder, "require_before_deadline",
                               side_effect=lambda: events.append("deadline")), \
             mock.patch.object(recorder, "prepare_evidence",
                               side_effect=lambda: (events.append("process"), refreshed)[1]), \
             mock.patch.object(recorder, "create_claim",
                               side_effect=lambda *_: (events.append("claim"), claim)[1]), \
             mock.patch.object(recorder, "execute_gate",
                               side_effect=lambda *_: (events.append("gate"), {})[1]), \
             mock.patch.object(recorder, "persist_execution",
                               side_effect=lambda *_: (events.append("execution"), execution)[1]), \
             mock.patch.object(recorder, "persist_report",
                               side_effect=lambda *_, **__: (
                                   events.append("report"), report
                               )[1]):
            result = recorder.run_stage(
                current, stage, [], {"claims": {}, "executions": {}, "reports": {}}
            )
        self.assertEqual(result, report)
        self.assertEqual(events, ["deadline", "process", "claim", "gate",
                                  "execution", "report"])

    def test_execution_receipt_structure_and_semantics_are_separated(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            loaded, digest = recorder.load_execution(execution[0], current, claim)
            self.assertEqual(loaded, execution[2])
            self.assertEqual(digest, execution[1])
            structural_corruptions = (
                {**execution[2], "timed_out": 0},
                {**execution[2], "extra": True},
                {**execution[2], "started_utc": 7},
            )
            for payload in structural_corruptions:
                execution[0].chmod(0o644)
                execution[0].write_bytes(recorder.canonical_json(payload))
                execution[0].chmod(0o444)
                with self.subTest(keys=set(payload)), self.assertRaises(ValueError):
                    recorder.load_execution(execution[0], current, claim)

            semantic_cases = (
                ({"returncode": None}, REPORTED,
                 "outcome tuple", True, True),
                ({"elapsed_monotonic_ns": 20_000_000_000}, REPORTED,
                 "wall/monotonic", False, True),
                ({"started_utc": "2026-08-14T12:20:00+00:00"}, REPORTED,
                 "chronology", False, True),
                ({"ended_utc": "2026-08-14T12:20:01+00:00"}, REPORTED,
                 "chronology", False, True),
                ({}, "2026-08-14T12:20:02+00:00",
                 "chronology", False, True),
                ({"started_utc": "2026-08-14T12:13:42+00:00",
                  "ended_utc": "2026-08-14T12:13:43+00:00"}, REPORTED,
                 "campaign interval", False, False),
            )
            for overrides, reported, expected_error, chronology, within in semantic_cases:
                payload = {**execution[2], **overrides}
                execution[0].chmod(0o644)
                execution[0].write_bytes(recorder.canonical_json(payload))
                execution[0].chmod(0o444)
                loaded, digest = recorder.load_execution(
                    execution[0], current, claim
                )
                semantic_execution = (execution[0], digest, loaded)
                report = recorder.report_payload(
                    current, claim, semantic_execution,
                    recorder.stable_evidence(current), created_utc=reported,
                )
                with self.subTest(error=expected_error):
                    self.assertFalse(report["development_selection_acceptable"])
                    self.assertTrue(any(
                        expected_error in error for error in report["process_errors"]
                    ))
                    self.assertIs(report["stage_chronology_valid"], chronology)
                    self.assertIs(report["stage_within_campaign_interval"], within)

    def test_recovered_execution_rollback_becomes_terminal_without_rerun(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(
                stage, current, claim,
                started_utc="2026-08-14T12:20:00+00:00",
            )
            state = recorder._scan_state(current)
            with mock.patch.object(recorder, "execute_gate") as gate, \
                 mock.patch.object(recorder, "prepare_evidence") as refresh, \
                 mock.patch.object(recorder, "utc_now", return_value=REPORTED):
                report = recorder.run_stage(current, stage, [], state)
            gate.assert_not_called()
            refresh.assert_not_called()
            self.assertFalse(report[2]["stage_chronology_valid"])
            self.assertFalse(report[2]["development_selection_acceptable"])
            state = recorder._scan_state(current)
            with mock.patch.object(
                recorder, "utc_now", return_value="2026-08-14T12:20:05+00:00"
            ):
                decision = recorder.finalize_decision(current, state["reports"])
            self.assertEqual(
                decision[2]["status"],
                "terminal-development-clock-rollback-rejection",
            )
            self.assertFalse(decision[2]["selected_for_source_activation_testing"])

    def test_execution_receipt_crash_resume_never_reruns_gate(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            state = {"claims": {stage: claim}, "executions": {stage: execution},
                     "reports": {}}
            with mock.patch.object(recorder, "execute_gate") as gate, \
                 mock.patch.object(recorder, "prepare_evidence") as refresh:
                report = recorder.run_stage(current, stage, [], state)
            gate.assert_not_called()
            refresh.assert_not_called()
            self.assertEqual(
                report[2]["postflight_error"]["class"],
                "InterruptedPostflightRecovery",
            )
            self.assertFalse(report[2]["development_selection_acceptable"])
            self.assertFalse(report[2]["stable_evidence"])

    def test_report_replays_stable_evidence_and_exact_timestamps(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            report = create_report(stage, current, claim, execution)
            replayed = recorder.validate_report(
                report[0], current, {stage: claim}, {stage: execution}
            )
            self.assertEqual(replayed, report)
            self.assertTrue(report[2]["development_selection_acceptable"])
            self.assertEqual(report[2]["preclaim_process"], claim[2]["preclaim_process"])

    def test_postflight_drift_is_durable_terminal_and_replays_after_restoration(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            after = copy.deepcopy(recorder.stable_evidence(current))
            after["environment_sha256"] = "9" * 64
            report = create_report(stage, current, claim, execution, after=after)
            self.assertFalse(report[2]["stable_evidence"])
            self.assertFalse(report[2]["development_selection_acceptable"])
            state = recorder._scan_state(current)
            self.assertEqual(state["reports"][stage], report)

    def test_postflight_exception_is_digest_only_and_cannot_be_laundered(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            with mock.patch.object(
                recorder, "prepare_evidence", side_effect=ValueError("sensitive detail")
            ):
                report = recorder.persist_report(current, claim, execution)
            marker = report[2]["postflight_error"]
            self.assertEqual(set(marker), {
                "class", "message_bytes", "message_sha256", "retained"
            })
            self.assertNotIn("sensitive detail", json.dumps(report[2]))
            self.assertFalse(report[2]["development_selection_acceptable"])
            self.assertEqual(recorder._scan_state(current)["reports"][stage], report)

    def test_later_claim_requires_exact_prior_chain_and_postdated_process(self):
        first, second = recorder.STAGE_NAMES[:2]
        current = evidence()
        with runtime_registries():
            claim1 = create_claim(first, current)
            execution1 = create_execution(first, current, claim1)
            report1 = create_report(first, current, claim1, execution1)
            later = evidence(checked="2026-08-14T12:20:05+00:00")
            later["candidate_identity"] = current["candidate_identity"]
            with mock.patch.object(recorder, "utc_now",
                                   return_value="2026-08-14T12:20:06+00:00"):
                claim2 = recorder.create_claim(later, second, [report1])
            state = recorder._scan_state(later)
            self.assertEqual(state["claims"][second], claim2)
            claim2[0].unlink()
            stale = evidence(checked="2026-08-14T12:20:03+00:00")
            stale["candidate_identity"] = current["candidate_identity"]
            with mock.patch.object(recorder, "utc_now",
                                   return_value="2026-08-14T12:20:06+00:00"), \
                 self.assertRaises(ValueError):
                recorder.create_claim(stale, second, [report1])

    def test_first_rejection_and_spent_claim_are_terminal_without_retry(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        rejected = (
            ROOT / "build/rejected.json", "5" * 64,
            {"schema": recorder.REPORT_SCHEMA, "stage": stage,
             "development_selection_acceptable": False, "created_utc": REPORTED,
             "stage_chronology_valid": True,
             "stage_within_campaign_interval": True},
        )
        decision = recorder.decision_payload(current, {stage: rejected},
                                             created_utc="2026-08-14T12:20:05+00:00")
        self.assertEqual(decision["status"], "terminal-development-rejection")
        self.assertFalse(decision["retry_authorized"])
        spent = (
            ROOT / "build/spent.json", "6" * 64,
            {"schema": recorder.CLAIM_SCHEMA, "stage": stage,
             "claimed_utc": CLAIMED},
        )
        decision = recorder.decision_payload(
            current, {}, spent_claim=spent,
            created_utc="2026-08-14T12:20:05+00:00",
        )
        self.assertEqual(decision["status"], "blocked-spent-stage-without-execution")
        self.assertFalse(decision["retry_authorized"])

    def test_all_three_passes_select_only_source_activation_testing(self):
        current = evidence()
        reports = {}
        for index, stage in enumerate(recorder.STAGE_NAMES):
            reports[stage] = (
                ROOT / f"build/report-{index}.json", str(index + 1) * 64,
                {"schema": recorder.REPORT_SCHEMA, "stage": stage,
                 "development_selection_acceptable": True,
                 "created_utc": f"2026-08-14T12:20:0{4 + index}+00:00",
                 "stage_chronology_valid": True,
                 "stage_within_campaign_interval": True},
            )
        decision = recorder.decision_payload(
            current, reports, created_utc="2026-08-14T12:20:08+00:00"
        )
        self.assertTrue(decision["selected_for_source_activation_testing"])
        self.assertFalse(decision["heldout_qualification"])
        self.assertFalse(decision["fresh_bank_campaign_authorization"])
        self.assertFalse(decision["arena_authorization"])
        self.assertEqual(decision["protected_bank_files_accessed"], [])

    def test_deadline_is_fail_closed_for_report_and_decision(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries():
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            late = recorder.report_payload(
                current, claim, execution, recorder.stable_evidence(current),
                created_utc="2026-08-15T07:15:08+00:00",
            )
            self.assertFalse(late["development_selection_acceptable"])
            self.assertIn("stage evidence exceeded the campaign interval",
                          late["process_errors"])
        reports = {
            stage_name: (
                ROOT / f"build/{stage_name}.json", str(index + 1) * 64,
                {"schema": recorder.REPORT_SCHEMA, "stage": stage_name,
                 "development_selection_acceptable": True,
                 "created_utc": REPORTED,
                 "stage_chronology_valid": True,
                 "stage_within_campaign_interval": True},
            )
            for index, stage_name in enumerate(recorder.STAGE_NAMES)
        }
        decision = recorder.decision_payload(
            current, reports, created_utc="2026-08-15T07:15:08+00:00"
        )
        self.assertEqual(decision["status"], "terminal-development-deadline-rejection")

    def test_admin_commit_timestamps_must_be_in_campaign_interval(self):
        head = "7" * 40
        def fake_git(*arguments, binary=False):
            if arguments[:2] == ("rev-parse", "HEAD"):
                return head
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{head} {recorder.ADMIN_PARENT_COMMIT}"
            if arguments[:2] == ("diff", "--name-only"):
                return "\n".join(recorder.ADMIN_CHANGED_PATHS)
            if arguments[:2] == ("diff", "--cached"):
                return ""
            if arguments[:3] == ("rev-list", "--left-right", "--count"):
                return "0 0"
            if arguments[:2] == ("show", "-s"):
                return "2026-08-14T12:00:00+00:00\n2026-08-14T12:20:00+00:00"
            raise AssertionError(arguments)
        with mock.patch.object(recorder, "git_run", side_effect=fake_git), \
             self.assertRaisesRegex(ValueError, "timestamp"):
            recorder.require_admin_commit({
                "source_identities": {},
                "producer_bootstrap_identities": {},
            })

    def test_admin_git_checks_use_the_exact_nonbank_bounded_pathspec(self):
        plan = json.loads(recorder.PLAN.read_text())
        head = "7" * 40
        calls = []

        def fake_git(*arguments, binary=False):
            calls.append(arguments)
            if arguments == ("rev-parse", "HEAD"):
                return head
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{head} {recorder.ADMIN_PARENT_COMMIT}"
            if arguments[:2] == ("diff", "--name-only"):
                return "\n".join(recorder.ADMIN_CHANGED_PATHS)
            if arguments[:2] == ("diff", "--cached"):
                return ""
            if arguments[:3] == ("rev-list", "--left-right", "--count"):
                return "0 0"
            if arguments[:2] == ("show", "-s"):
                return "2026-08-14T12:20:00+00:00\n2026-08-14T12:20:01+00:00"
            if arguments[0] == "ls-tree":
                label = arguments[-1]
                return f"100644 blob {'8' * 40}\t{label}"
            if arguments[:2] == ("ls-files", "--stage"):
                label = arguments[-1]
                return f"100644 {'8' * 40} 0\t{label}"
            if arguments[0] == "show" and binary:
                return b"x"
            raise AssertionError(arguments)

        identity = {
            "ascii": True, "bytes": 1, "executable": False,
            "mode": "0644", "path": "synthetic",
            "sha256": recorder.sha256_bytes(b"x"),
        }
        with mock.patch.object(recorder, "git_run", side_effect=fake_git), \
             mock.patch.object(recorder, "file_identity", return_value=identity):
            admin = recorder.require_admin_commit(plan)
        expected = recorder.bounded_git_paths(plan)
        self.assertEqual(admin["bounded_git_paths"], list(expected))
        cached = next(call for call in calls if call[:2] == ("diff", "--cached"))
        self.assertEqual(cached[cached.index("--") + 1:], expected)
        self.assertFalse(any(call[0] == "status" for call in calls))
        self.assertTrue(all(
            not path.endswith(".tsv") and "openings" not in Path(path).parts
            for path in expected
        ))

    def test_process_snapshot_detects_generic_legacy_recorder_or_gate(self):
        identity = {
            "ascii": False, "bytes": recorder.PS_BYTES,
            "executable": True, "mode": "4755",
            "path": str(recorder.PS), "sha256": recorder.PS_SHA256,
        }
        outputs = (
            "999991 1 python tools/record_rank4_jacek_hybrid_old_clock.py\n",
            f"999992 1 {recorder.GATE.name} --profile clock\n",
        )
        for output in outputs:
            completed = subprocess.CompletedProcess([], 0, output, "")
            with mock.patch.object(recorder, "file_identity", return_value=identity), \
                 mock.patch.object(recorder.subprocess, "run", return_value=completed), \
                 self.assertRaises(ValueError):
                recorder.process_snapshot()

    def test_all_six_locks_are_held_and_released_in_reverse_order(self):
        current = evidence()
        decision = (ROOT / "build/decision.json", "7" * 64, {"status": "done"})
        opened: list[Path] = []
        closed: list[int] = []
        events: list[str] = []
        def open_lock(path):
            opened.append(path)
            return 100 + len(opened)
        state = {"claims": {}, "executions": {}, "reports": {},
                 "decisions": [decision]}
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock", side_effect=open_lock), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories"), \
             mock.patch.object(
                 recorder, "repair_registry_durability",
                 side_effect=lambda *_: events.append("repair"),
             ), \
             mock.patch.object(
                 recorder, "prepare_evidence",
                 side_effect=lambda: (events.append("evidence"), current)[1],
             ), \
             mock.patch.object(
                 recorder, "_scan_state",
                 side_effect=lambda *_: (events.append("scan"), state)[1],
             ), \
             mock.patch.object(recorder, "validate_decision"), \
             mock.patch.object(recorder.os, "close", side_effect=closed.append):
            self.assertEqual(recorder.run_campaign(), decision)
        self.assertEqual(opened, list(recorder.CAMPAIGN_LOCKS))
        self.assertEqual(closed, list(reversed(range(101, 107))))
        self.assertEqual(events, ["evidence", "repair", "scan"])

    def test_busy_lock_closes_prior_descriptors_before_any_evidence(self):
        calls = [101, BlockingIOError()]
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock", side_effect=calls), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories") as prepare, \
             mock.patch.object(recorder, "prepare_evidence") as evidence_call, \
             mock.patch.object(recorder.os, "close") as close, \
             self.assertRaises(ValueError):
            recorder.run_campaign()
        close.assert_called_once_with(101)
        prepare.assert_not_called()
        evidence_call.assert_not_called()

    def test_earlier_lock_path_swap_fails_before_any_evidence(self):
        build = ROOT / "build"
        with tempfile.TemporaryDirectory(
            prefix="mask3-lock-swap-test-", dir=build
        ) as name:
            directory = Path(name)
            first = directory / "first.lock"
            second = directory / "second.lock"
            replacement = directory / "replacement"
            displaced = directory / "displaced-first.lock"
            replacement.write_bytes(b"")
            replacement.chmod(0o644)
            real_open = recorder.open_exclusive_lock

            def open_and_swap(path):
                descriptor = real_open(path)
                if path == second:
                    os.replace(first, displaced)
                    os.replace(replacement, first)
                return descriptor

            with mock.patch.object(recorder, "CAMPAIGN_LOCKS", (first, second)), \
                 mock.patch.object(recorder, "validate_output_topology"), \
                 mock.patch.object(
                     recorder, "open_exclusive_lock", side_effect=open_and_swap
                 ), \
                 mock.patch.object(recorder, "prepare_runtime_directories") as prepare, \
                 mock.patch.object(recorder, "prepare_evidence") as evidence_call, \
                 self.assertRaisesRegex(ValueError, "pathname"):
                recorder.run_campaign()
            prepare.assert_not_called()
            evidence_call.assert_not_called()

    def test_new_registry_and_every_exclusive_record_are_directory_fsynced(self):
        build = ROOT / "build"
        with tempfile.TemporaryDirectory(prefix="mask3-fsync-test-", dir=build) as name:
            output = Path(name)
            replacements = {
                "OUTPUT": output,
                "LOCK": output / ".recorder.lock",
                "CLAIMS": output / "claims",
                "EXECUTIONS": output / "executions",
                "REPORTS": output / "reports",
                "DECISIONS": output / "decisions",
            }
            with mock.patch.multiple(recorder, **replacements), \
                 mock.patch.object(recorder, "fsync_directory") as sync_directory:
                recorder.prepare_runtime_directories()
                self.assertEqual(
                    sync_directory.call_args_list,
                    [mock.call(output), mock.call(output), mock.call(output),
                     mock.call(output)],
                )
                sync_directory.reset_mock()
                recorder.prepare_runtime_directories()
                self.assertEqual(
                    sync_directory.call_args_list,
                    [mock.call(output), mock.call(output), mock.call(output),
                     mock.call(output)],
                )
                current = evidence()
                sync_directory.reset_mock()
                with mock.patch.object(recorder.os, "fsync", wraps=os.fsync) as sync_file:
                    claim = create_claim(recorder.STAGE_NAMES[0], current)
                    execution = create_execution(
                        recorder.STAGE_NAMES[0], current, claim
                    )
                    report = create_report(
                        recorder.STAGE_NAMES[0], current, claim, execution
                    )
                    decision = recorder.persist_content_addressed(
                        recorder.DECISIONS, {"schema": "synthetic-decision"}
                    )
                self.assertEqual(
                    sync_directory.call_args_list,
                    [mock.call(recorder.CLAIMS), mock.call(recorder.EXECUTIONS),
                     mock.call(recorder.REPORTS), mock.call(recorder.DECISIONS)],
                )
                self.assertGreaterEqual(sync_file.call_count, 4)
                sync_directory.reset_mock()
                with mock.patch.object(
                    recorder, "fsync_regular_record",
                    wraps=recorder.fsync_regular_record,
                ) as repair:
                    recorder.repair_registry_durability(current)
                    repaired = {call.args[0] for call in repair.call_args_list}
                    self.assertEqual(
                        repaired,
                        {claim[0], execution[0], report[0], decision[0]},
                    )
                    repair.reset_mock()
                    recorder.persist_content_addressed(recorder.REPORTS, report[2])
                    repair.assert_called_once()
                    self.assertEqual(repair.call_args.args, (report[0],))
                    self.assertEqual(
                        repair.call_args.kwargs["expected_inode"],
                        recorder._inode_key(os.lstat(report[0])),
                    )
                    foreign = recorder.REPORTS / "foreign.json"
                    foreign.write_text("foreign")
                    repair.reset_mock()
                    with self.assertRaises(ValueError):
                        recorder.repair_registry_durability(current)
                    repair.assert_not_called()

    def test_campaign_orders_all_stages_and_stops_at_first_rejection(self):
        current = evidence()
        accepted_reports = {
            stage: (
                ROOT / f"build/{stage}.json", str(index + 1) * 64,
                {"schema": recorder.REPORT_SCHEMA, "stage": stage,
                 "development_selection_acceptable": True,
                 "created_utc": f"2026-08-14T12:20:0{4 + index}+00:00",
                 "stage_chronology_valid": True,
                 "stage_within_campaign_interval": True},
            )
            for index, stage in enumerate(recorder.STAGE_NAMES)
        }
        empty = {"claims": {}, "executions": {}, "reports": {}, "decisions": []}
        progressive = [empty]
        for index, stage in enumerate(recorder.STAGE_NAMES):
            progressive.append({
                "claims": {}, "executions": {}, "decisions": [],
                "reports": {
                    name: accepted_reports[name]
                    for name in recorder.STAGE_NAMES[:index + 1]
                },
            })
        final = (ROOT / "build/final-decision.json", "8" * 64, {"status": "done"})
        opened = iter(range(201, 207))
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock",
                               side_effect=lambda _path: next(opened)), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories"), \
             mock.patch.object(recorder, "repair_registry_durability"), \
             mock.patch.object(recorder, "prepare_evidence", return_value=current), \
             mock.patch.object(recorder, "_scan_state", side_effect=progressive), \
             mock.patch.object(
                 recorder, "run_stage",
                 side_effect=[accepted_reports[name] for name in recorder.STAGE_NAMES],
             ) as run_stage, \
             mock.patch.object(recorder, "finalize_decision", return_value=final), \
             mock.patch.object(recorder, "execute_gate") as gate, \
             mock.patch.object(recorder.os, "close"):
            self.assertEqual(recorder.run_campaign(), final)
        self.assertEqual(
            [call.args[1] for call in run_stage.call_args_list],
            list(recorder.STAGE_NAMES),
        )
        gate.assert_not_called()

        rejected = copy.deepcopy(accepted_reports[recorder.STAGE_NAMES[0]])
        rejected[2]["development_selection_acceptable"] = False
        rejected_state = {
            "claims": {}, "executions": {}, "decisions": [],
            "reports": {recorder.STAGE_NAMES[0]: rejected},
        }
        opened = iter(range(301, 307))
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock",
                               side_effect=lambda _path: next(opened)), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories"), \
             mock.patch.object(recorder, "repair_registry_durability"), \
             mock.patch.object(recorder, "prepare_evidence", return_value=current), \
             mock.patch.object(recorder, "_scan_state",
                               side_effect=[empty, rejected_state]), \
             mock.patch.object(recorder, "run_stage", return_value=rejected) as run_stage, \
             mock.patch.object(recorder, "finalize_decision", return_value=final), \
             mock.patch.object(recorder, "execute_gate") as gate, \
             mock.patch.object(recorder.os, "close"):
            recorder.run_campaign()
        run_stage.assert_called_once()
        self.assertEqual(run_stage.call_args.args[1], recorder.STAGE_NAMES[0])
        gate.assert_not_called()

    def test_campaign_recovers_claim_only_and_report_without_another_gate(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        claim = (
            ROOT / "build/spent-stage.json", "4" * 64,
            {"schema": recorder.CLAIM_SCHEMA, "stage": stage,
             "claimed_utc": CLAIMED},
        )
        spent_state = {
            "claims": {stage: claim}, "executions": {}, "reports": {},
            "decisions": [],
        }
        decision = (ROOT / "build/decision.json", "5" * 64, {"status": "spent"})
        opened = iter(range(401, 407))
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock",
                               side_effect=lambda _path: next(opened)), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories"), \
             mock.patch.object(recorder, "repair_registry_durability"), \
             mock.patch.object(recorder, "prepare_evidence", return_value=current), \
             mock.patch.object(recorder, "_scan_state", return_value=spent_state), \
             mock.patch.object(recorder, "finalize_decision", return_value=decision) as persist, \
             mock.patch.object(recorder, "execute_gate") as gate, \
             mock.patch.object(recorder.os, "close"):
            self.assertEqual(recorder.run_campaign(), decision)
        persist.assert_called_once_with(current, {}, spent_claim=claim)
        gate.assert_not_called()

        report = (
            ROOT / "build/rejected-stage.json", "6" * 64,
            {"schema": recorder.REPORT_SCHEMA, "stage": stage,
             "development_selection_acceptable": False, "created_utc": REPORTED,
             "stage_chronology_valid": True,
             "stage_within_campaign_interval": True},
        )
        report_state = {
            "claims": {}, "executions": {}, "reports": {stage: report},
            "decisions": [],
        }
        opened = iter(range(501, 507))
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "open_exclusive_lock",
                               side_effect=lambda _path: next(opened)), \
             mock.patch.object(recorder, "validate_held_lock"), \
             mock.patch.object(recorder, "prepare_runtime_directories"), \
             mock.patch.object(recorder, "repair_registry_durability"), \
             mock.patch.object(recorder, "prepare_evidence", return_value=current), \
             mock.patch.object(recorder, "_scan_state", return_value=report_state), \
             mock.patch.object(recorder, "finalize_decision", return_value=decision), \
             mock.patch.object(recorder, "execute_gate") as gate, \
             mock.patch.object(recorder.os, "close"):
            self.assertEqual(recorder.run_campaign(), decision)
        gate.assert_not_called()

    def test_decision_semantics_replay_and_multiple_decisions_fail_closed(self):
        current = evidence()
        reports = {
            stage: (
                ROOT / f"build/{stage}.json", str(index + 1) * 64,
                {"schema": recorder.REPORT_SCHEMA, "stage": stage,
                 "development_selection_acceptable": True,
                 "created_utc": f"2026-08-14T12:20:0{4 + index}+00:00",
                 "stage_chronology_valid": True,
                 "stage_within_campaign_interval": True},
            )
            for index, stage in enumerate(recorder.STAGE_NAMES)
        }
        with runtime_registries():
            with mock.patch.object(recorder, "utc_now",
                                   return_value="2026-08-14T12:20:08+00:00"):
                decision = recorder.persist_decision(current, reports)
            state = {"claims": {}, "executions": {}, "reports": reports,
                     "decisions": [decision]}
            recorder.validate_decision(decision, current, state)
            decision[0].unlink()
            for variant in ("one", "two"):
                recorder.persist_content_addressed(recorder.DECISIONS, {
                    "schema": recorder.DECISION_SCHEMA,
                    "campaign_id": recorder.CAMPAIGN_ID,
                    "candidate_identity": current["candidate_identity"],
                    "variant": variant,
                })
            with self.assertRaisesRegex(ValueError, "multiple"):
                recorder._scan_state(current)

    def test_backward_clock_decision_is_rescanned_terminal_and_never_success(self):
        reports = {}
        prior = []
        current = evidence(checked="2026-08-14T12:30:00+00:00")
        with runtime_registries():
            for index, stage in enumerate(recorder.STAGE_NAMES):
                second = index * 5
                checked = f"2026-08-14T12:30:{second:02d}+00:00"
                claimed = f"2026-08-14T12:30:{second + 1:02d}+00:00"
                started = f"2026-08-14T12:30:{second + 2:02d}+00:00"
                ended = f"2026-08-14T12:30:{second + 3:02d}+00:00"
                reported = f"2026-08-14T12:30:{second + 4:02d}+00:00"
                current = evidence(checked=checked)
                with mock.patch.object(recorder, "utc_now", return_value=claimed):
                    claim = recorder.create_claim(current, stage, prior)
                execution = create_execution(
                    stage, current, claim, started_utc=started, ended_utc=ended
                )
                report = create_report(
                    stage, current, claim, execution, created=reported
                )
                reports[stage] = report
                prior.append(report)
            with mock.patch.object(
                recorder, "utc_now", return_value="2026-08-14T12:30:02+00:00"
            ):
                decision = recorder.finalize_decision(current, reports)
            self.assertEqual(
                decision[2]["status"],
                "terminal-development-clock-rollback-rejection",
            )
            self.assertFalse(decision[2]["decision_chronology_valid"])
            self.assertFalse(decision[2]["selected_for_source_activation_testing"])
            state = recorder._scan_state(current)
            self.assertEqual(state["decisions"], [decision])
            recorder.validate_decision(decision, current, state)
            with mock.patch.object(recorder, "run_campaign", return_value=decision), \
                 mock.patch.object(recorder.sys, "argv", [str(RECORDER_PATH), "run"]), \
                 mock.patch.object(recorder.sys, "stdout", new_callable=io.StringIO):
                self.assertEqual(recorder.main(), 1)

    def test_main_sanitizes_preclaim_subprocess_errors(self):
        error = subprocess.TimeoutExpired(["/usr/bin/otool"], 60)
        with mock.patch.object(recorder, "audit_campaign", side_effect=error), \
             mock.patch.object(recorder.sys, "argv", [str(RECORDER_PATH), "audit"]), \
             mock.patch.object(recorder.sys, "stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(recorder.main(), 2)
        self.assertIn("timed out", stderr.getvalue())

    def test_main_run_output_is_exact_and_success_requires_selection(self):
        path = ROOT / "build/synthetic-mask3-decision.json"
        digest = "7" * 64
        for selected, status, expected_return in (
            (True, "selected-for-source-activation-testing-only", 0),
            (False, "terminal-development-rejection", 1),
            (False, "terminal-development-clock-rollback-rejection", 1),
            (False, "terminal-development-deadline-rejection", 1),
        ):
            decision = {
                "status": status,
                "selected_for_source_activation_testing": selected,
            }
            with mock.patch.object(
                recorder, "run_campaign", return_value=(path, digest, decision)
            ), mock.patch.object(
                recorder.sys, "argv", [str(RECORDER_PATH), "run"]
            ), mock.patch.object(
                recorder.sys, "stdout", new_callable=io.StringIO
            ) as stdout, mock.patch.object(
                recorder.sys, "stderr", new_callable=io.StringIO
            ) as stderr:
                result = recorder.main()
            self.assertEqual(result, expected_return)
            self.assertEqual(
                stdout.getvalue(),
                f"build/synthetic-mask3-decision.json\nsha256={digest}\n"
                f"status={status}\n",
            )
            self.assertEqual(stderr.getvalue(), "")

        with mock.patch.object(
            recorder, "run_campaign", side_effect=ValueError("synthetic failure")
        ), mock.patch.object(
            recorder.sys, "argv", [str(RECORDER_PATH), "run"]
        ), mock.patch.object(
            recorder.sys, "stdout", new_callable=io.StringIO
        ) as stdout, mock.patch.object(
            recorder.sys, "stderr", new_callable=io.StringIO
        ) as stderr:
            self.assertEqual(recorder.main(), 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "synthetic failure\n")

    def test_audit_path_is_write_free_and_git_disables_optional_locks(self):
        current = evidence()
        state = {
            "claims": {}, "executions": {}, "reports": {}, "decisions": [],
        }
        with mock.patch.object(recorder, "validate_output_topology"), \
             mock.patch.object(recorder, "prepare_evidence", return_value=current), \
             mock.patch.object(recorder, "_scan_state", return_value=state), \
             mock.patch.object(recorder, "prepare_runtime_directories") as mkdirs, \
             mock.patch.object(recorder, "repair_registry_durability") as repair, \
             mock.patch.object(recorder, "write_exclusive") as write_record, \
             mock.patch.object(recorder, "open_exclusive_lock") as lock, \
             mock.patch.object(recorder, "execute_gate") as gate, \
             mock.patch.object(recorder.os, "fsync") as file_sync, \
             mock.patch.object(recorder, "fsync_directory") as directory_sync:
            audited = recorder.audit_campaign()
        self.assertEqual(audited["decision_count"], 0)
        for forbidden in (
            mkdirs, repair, write_record, lock, gate, file_sync, directory_sync,
        ):
            forbidden.assert_not_called()
        self.assertTrue(recorder.sys.dont_write_bytecode)

        completed = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(
            recorder.subprocess, "run", return_value=completed
        ) as run:
            recorder.git_run("status", "--porcelain")
        self.assertEqual(run.call_args.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
        self.assertNotIn(
            'git_run("status", "--porcelain"', RECORDER_PATH.read_text()
        )

    def test_foreign_and_symlink_registry_entries_fail_closed(self):
        current = evidence()
        with runtime_registries():
            foreign = recorder.CLAIMS / "foreign.txt"
            foreign.write_text("x")
            with self.assertRaises(ValueError):
                recorder._registry_files(recorder.CLAIMS, recorder.SHA256_RE)
            foreign.unlink()
            target = recorder.CLAIMS / "target"
            target.write_text("x")
            link = recorder.CLAIMS / ("a" * 64 + ".json")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                recorder._registry_files(
                    recorder.CLAIMS, recorder.re.compile(r"[0-9a-f]{64}\.json")
                )
            link.unlink()
            target.unlink()
            hardlink_target = recorder.OUTPUT / "hardlink-target"
            hardlink_target.write_text("protected stand-in")
            hardlink = recorder.CLAIMS / (
                f"{current['candidate_identity']}.{recorder.STAGE_NAMES[0]}.json"
            )
            os.link(hardlink_target, hardlink)
            with mock.patch.object(recorder, "fsync_regular_record") as repair, \
                 self.assertRaises(ValueError):
                recorder.repair_registry_durability(current)
            repair.assert_not_called()
            hardlink.unlink()
            hardlink_target.unlink()
            payload = {"schema": "collision"}
            digest = recorder.sha256_bytes(recorder.canonical_json(payload))
            collision_target = recorder.CLAIMS / "collision-target"
            collision_target.write_bytes(recorder.canonical_json(payload))
            collision = recorder.REPORTS / f"{digest}.json"
            collision.symlink_to(collision_target)
            with self.assertRaisesRegex(ValueError, "immutable single-link"):
                recorder.persist_content_addressed(recorder.REPORTS, payload)

    def test_registry_read_is_nofollow_inode_pinned_and_reports_read_once(self):
        current = evidence()
        stage = recorder.STAGE_NAMES[0]
        with runtime_registries() as output:
            claim = create_claim(stage, current)
            execution = create_execution(stage, current, claim)
            report = create_report(stage, current, claim, execution)
            with mock.patch.object(
                recorder, "read_regular_record",
                wraps=recorder.read_regular_record,
            ) as read_record, mock.patch.object(
                recorder, "validate_report", wraps=recorder.validate_report,
            ) as validate_report:
                state = recorder._scan_state(current)
            self.assertEqual(state["reports"][stage], report)
            report_reads = [
                call for call in read_record.call_args_list
                if call.args and call.args[0] == report[0]
            ]
            self.assertEqual(len(report_reads), 1)
            validate_report.assert_called_once()
            self.assertEqual(
                validate_report.call_args.kwargs["raw"],
                recorder.canonical_json(report[2]),
            )

            pattern = recorder.re.compile(
                rf"{current['candidate_identity']}\.{stage}\.json"
            )
            [(claim_path, captured_inode)] = recorder._registry_snapshot(
                recorder.CLAIMS, pattern
            )
            stand_in = output / "protected-stand-in"
            stand_in.write_bytes(b"protected content must not be read")
            stand_in.chmod(0o444)
            claim_path.unlink()
            os.replace(stand_in, claim_path)
            with mock.patch.object(
                recorder.os, "read", wraps=os.read
            ) as descriptor_read, self.assertRaisesRegex(
                ValueError, "immutable single-link"
            ):
                recorder.read_regular_record(
                    claim_path, expected_inode=captured_inode
                )
            descriptor_read.assert_not_called()

    def test_existing_content_swap_is_rejected_before_any_target_read(self):
        with runtime_registries() as output:
            payload = {"schema": "synthetic-content-addressed-record"}
            path, _ = recorder.persist_content_addressed(
                recorder.REPORTS, payload
            )
            stand_in = output / "protected-content-stand-in"
            stand_in.write_bytes(b"protected content must not be read")
            stand_in.chmod(0o444)
            real_fsync = recorder.fsync_regular_record

            def swap_before_repair(record_path, *, expected_inode=None):
                record_path.unlink()
                os.replace(stand_in, record_path)
                return real_fsync(
                    record_path, expected_inode=expected_inode
                )

            with mock.patch.object(
                recorder, "fsync_regular_record",
                side_effect=swap_before_repair,
            ), mock.patch.object(
                recorder.os, "read", wraps=os.read
            ) as descriptor_read, self.assertRaisesRegex(
                ValueError, "immutable single-link"
            ):
                recorder.persist_content_addressed(recorder.REPORTS, payload)
            descriptor_read.assert_not_called()
            self.assertEqual(path.stat().st_size, len(b"protected content must not be read"))

    def test_fixed_input_reads_are_nofollow_and_resolved_tsv_aliases_reject(self):
        with runtime_registries() as output:
            target = output / "protected-stand-in.tsv"
            target.write_bytes(b"synthetic protected content")
            alias = output / "apparently-safe-header.h"
            alias.symlink_to(target)
            with mock.patch.object(
                recorder.os, "read", wraps=os.read
            ) as descriptor_read, self.assertRaises(ValueError):
                recorder.read_regular_file(alias)
            descriptor_read.assert_not_called()
            with mock.patch.object(recorder, "git_run") as git, \
                 mock.patch.object(recorder.os, "read", wraps=os.read) as read, \
                 self.assertRaises(ValueError):
                recorder.load_tracked_exact_bytes(
                    alias, recorder.sha256_bytes(b"synthetic protected content")
                )
            read.assert_not_called()
            git.assert_not_called()
            with self.assertRaisesRegex(ValueError, "resolved compiler dependency"):
                recorder._resolved_dependency(alias)
            alias.unlink()
            os.link(target, alias)
            with mock.patch.object(
                recorder.os, "read", wraps=os.read
            ) as descriptor_read, self.assertRaises(ValueError):
                recorder.read_regular_file(alias)
            descriptor_read.assert_not_called()

    def test_new_lock_has_fixed_mode_under_restrictive_umask_and_hardlinks_fail(self):
        with runtime_registries() as output:
            lock = output / "restrictive-umask.lock"
            previous_umask = os.umask(0o077)
            try:
                descriptor = recorder.open_exclusive_lock(lock)
            finally:
                os.umask(previous_umask)
            try:
                self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o644)
                self.assertEqual(os.fstat(descriptor).st_nlink, 1)
            finally:
                os.close(descriptor)

            target = output / "unrelated-lock-target"
            target.write_bytes(b"")
            target.chmod(0o644)
            linked_lock = output / "hardlinked.lock"
            os.link(target, linked_lock)
            with self.assertRaisesRegex(ValueError, "regular file"):
                recorder.open_exclusive_lock(linked_lock)

    def test_tracked_evidence_live_mode_is_checked_before_content_read(self):
        with runtime_registries() as output:
            path = output / "immutable-evidence.json"
            raw = recorder.canonical_json({"schema": "synthetic"})
            path.write_bytes(raw)
            path.chmod(0o444)
            digest = recorder.sha256_bytes(raw)
            with mock.patch.object(recorder, "require_git_regular_blob"), \
                 mock.patch.object(recorder, "require_index_regular_blob"), \
                 mock.patch.object(recorder, "git_run", return_value=raw):
                self.assertEqual(
                    recorder.load_tracked_exact_bytes(
                        path, digest, live_mode=0o444
                    ),
                    raw,
                )
            path.chmod(0o644)
            with mock.patch.object(recorder, "git_run") as git, \
                 mock.patch.object(recorder.os, "read", wraps=os.read) as read, \
                 self.assertRaises(ValueError):
                recorder.load_tracked_exact_bytes(
                    path, digest, live_mode=0o444
                )
            read.assert_not_called()
            git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
