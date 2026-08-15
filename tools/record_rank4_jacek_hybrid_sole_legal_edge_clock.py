#!/usr/bin/env python3
"""Record the one frozen DEVELOPMENT d20 sole-legal-edge clock ablation.

There are no bank, clock, mask, or retry knobs.  The dedicated comparison
target maps its generic ``rank4`` slot to the archived pre-fastpath engine.
Validation/final banks cannot be addressed by this recorder.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import record_rank4_jacek_hybrid_null_fastpath_clock as base


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/sole_legal_edge_clock"
PLAN = OUTPUT / "plan.json"
ROLLBACK = ROOT / (
    "results/rank_4_jacek_hybrid/gates/null_fastpath_clock/rollback/"
    "33c1146b41c5bfd2be07283309d6614872ef9bc17ff7619089cdb35d03b74a62.json"
)
DECISION = ROOT / (
    "results/rank_4_jacek_hybrid/gates/null_fastpath_clock/selection/"
    "27de96bac5b2ea6c43613ee8b9f5c64f16a33505bfcfde1872d33e3b3c2268bb.json"
)
PROTOTYPE = OUTPUT / "prototype"
PROTOTYPE_FILES = (
    PROTOTYPE / "RESULTS.md",
    PROTOTYPE / "microbench.cpp",
    PROTOTYPE / "parity_harness.cpp",
    PROTOTYPE / "run_microbench.py",
)
LOCK = ROOT / "build/rank4-jacek-hybrid-sole-legal-edge-clock.lock"
GATE_TARGET = base.GATE_TARGET
GATE = base.GATE
BANK = base.BANK
RUN_TIMEOUT_SECONDS = base.RUN_TIMEOUT_SECONDS

SCHEMA = "rank4-jacek-hybrid-sole-legal-edge-clock-v1"
PLAN_SCHEMA = "rank4-jacek-hybrid-sole-legal-edge-clock-plan-v1"
CLASSIFICATION = (
    "development-sole-legal-edge-clock-ablation-not-final-qualification"
)
CAMPAIGN_ID = base.CAMPAIGN_ID
CAMPAIGN_T0_UTC = base.CAMPAIGN_T0_UTC

ROLLBACK_COMMIT = "e6fc14b6adc7ca8d1d0ff515b9519da60cc0c217"
CONTROL_SOURCE_COMMIT = base.CONTROL_SOURCE_COMMIT
EXPECTED_CANDIDATE_BOT_SHA256 = (
    "16a4358680cfc69e830136d4e0c2e6e45371139a02ca09ecb9bf1f9e239d3b2b"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "d18c49c7cc149d8b48a69a03ebb13dd4fc49ae8927c1324515ba1ae197822b15"
)
EXPECTED_CANDIDATE_TEST_SHA256 = (
    "0823299900cf0d31730c73ccb91a3a55c7a7ef351949e583a40a7c66a43f5e88"
)
EXPECTED_CONTROL_BOT_SHA256 = base.EXPECTED_CONTROL_BOT_SHA256
EXPECTED_CONTROL_SOURCE_SHA256 = base.EXPECTED_CONTROL_SOURCE_SHA256
EXPECTED_BANK_SHA256 = base.EXPECTED_BANK_SHA256
EXPECTED_PLAN_SHA256 = (
    "04cd689b6d5907646fd11cdb35a206719e08134507b768d0fa08e767952aec4b"
)
EXPECTED_ROLLBACK_SHA256 = (
    "33c1146b41c5bfd2be07283309d6614872ef9bc17ff7619089cdb35d03b74a62"
)
EXPECTED_DECISION_SHA256 = (
    "27de96bac5b2ea6c43613ee8b9f5c64f16a33505bfcfde1872d33e3b3c2268bb"
)
EXPECTED_CONTROL_MANIFEST_SHA256 = base.EXPECTED_MANIFEST_SHA256
EXPECTED_PROTOTYPE = {
    "RESULTS.md": ("7c7e72fd09a678f2b377009a694d49547d4460e190362db7ef8835a8c62793a4", 2966),
    "microbench.cpp": ("2007e0daee4282ee6924aa43bbb8ffec3844a6ab845d2db3e5e79c6634837510", 2628),
    "parity_harness.cpp": ("12d4db655376df1c3e3556615a1f7562137539448ecd8eb58dab6767bae9842f", 5945),
    "run_microbench.py": ("48c3c5a468bb086c6dc55d2d0bc529741ee3eaedbba8f753afa7af9503dbaa21", 1836),
}

BOT_DIRECTORY = base.BOT_DIRECTORY
CANDIDATE_BOT = base.CANDIDATE_BOT
CANDIDATE_SOURCE = base.CANDIDATE_SOURCE
CANDIDATE_TEST = base.CANDIDATE_TEST
CONTROL_MANIFEST = base.CONTROL_MANIFEST
CONTROL_BOT = base.CONTROL_BOT
CONTROL_SOURCE = base.CONTROL_SOURCE

TRACKED_INPUTS = (
    RECORDER,
    Path(base.__file__).resolve(),
    ROOT / "tools/record_rank4_jacek_hybrid_proof_scope_clock.py",
    ROOT / "CMakeLists.txt",
    PLAN,
    ROLLBACK,
    DECISION,
    *PROTOTYPE_FILES,
    CONTROL_MANIFEST,
    CONTROL_BOT,
    CONTROL_SOURCE,
    CANDIDATE_BOT,
    CANDIDATE_SOURCE,
    CANDIDATE_TEST,
    BOT_DIRECTORY / "replay_book.hpp",
    BOT_DIRECTORY / "replay_value_model.hpp",
    BOT_DIRECTORY / "teacher_residual_model.hpp",
    BOT_DIRECTORY / "comparison_gate.cpp",
    BOT_DIRECTORY / "comparison_gate_engine.hpp",
    BOT_DIRECTORY / "comparison_gate_hybrid.cpp",
    BOT_DIRECTORY / "comparison_gate_null_fastpath.cpp",
    BOT_DIRECTORY / "comparison_gate_prefastpath.cpp",
    ROOT / "src/bots/mcts_internal.hpp",
    ROOT / "src/opening_bank/opening_bank.cpp",
    ROOT / "src/opening_bank/opening_bank_internal.hpp",
    ROOT / "src/core/rules.cpp",
    ROOT / "src/core/geometry.cpp",
    ROOT / "include/papersoccer/rules.hpp",
    ROOT / "include/papersoccer/geometry.hpp",
    ROOT / "include/papersoccer/types.hpp",
    BANK,
    ROOT / "build/CMakeCache.txt",
    ROOT / f"build/CMakeFiles/{GATE_TARGET}.dir/flags.make",
    ROOT / f"build/CMakeFiles/{GATE_TARGET}.dir/link.txt",
    GATE,
)

canonical_json = base.canonical_json
sha256_bytes = base.sha256_bytes
persist_content_addressed_report = base.persist_content_addressed_report
expected_configuration = base.expected_configuration
command_for_gate = base.command_for_gate
validate_gate_stdout = base.validate_gate_stdout
selection_errors = base.selection_errors
parse_process_table = base.parse_process_table
process_preflight_from_table = base.process_preflight_from_table


def validate_canonical_json_file(path: Path, digest: str) -> Any:
    raw = path.read_bytes()
    if sha256_bytes(raw) != digest:
        raise ValueError(f"frozen JSON SHA-256 mismatch: {path}")
    payload = json.loads(raw)
    if canonical_json(payload) != raw:
        raise ValueError(f"frozen JSON is not canonical: {path}")
    return payload


def validate_exact_json_file(path: Path, digest: str) -> Any:
    raw = path.read_bytes()
    if sha256_bytes(raw) != digest:
        raise ValueError(f"frozen JSON SHA-256 mismatch: {path}")
    return json.loads(raw)


def validate_plan_and_lineage() -> dict[str, Any]:
    plan = validate_canonical_json_file(PLAN, EXPECTED_PLAN_SHA256)
    rollback = validate_exact_json_file(ROLLBACK, EXPECTED_ROLLBACK_SHA256)
    decision = validate_exact_json_file(DECISION, EXPECTED_DECISION_SHA256)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("final_qualification") is not False:
        raise ValueError("sole-edge plan schema/classification mismatch")
    if plan.get("candidate") != {
        "base_commit": ROLLBACK_COMMIT,
        "bot_bytes": 63_350,
        "bot_sha256": EXPECTED_CANDIDATE_BOT_SHA256,
        "change": "when exactly one legal slot exists, return its canonical move with score zero before heuristic and mobility scoring",
        "exact_proof_mask": 7,
        "source_bytes": 94_527,
        "source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "test_bytes": 40_103,
        "test_sha256": EXPECTED_CANDIDATE_TEST_SHA256,
    }:
        raise ValueError("sole-edge plan candidate binding mismatch")
    reference = plan.get("reference", {})
    if (reference.get("archived_bot_sha256") != EXPECTED_CONTROL_BOT_SHA256 or
            reference.get("archived_source_sha256") != EXPECTED_CONTROL_SOURCE_SHA256 or
            reference.get("source_commit") != CONTROL_SOURCE_COMMIT or
            reference.get("exact_proof_mask") != 7 or
            reference.get("function") != "choose_prefastpath" or
            reference.get("slot_echo") != "rank4"):
        raise ValueError("sole-edge plan reference binding mismatch")
    lineage = plan.get("lineage", {})
    if (lineage.get("pre_singleton_rollback", {}).get("manifest_sha256") !=
            EXPECTED_ROLLBACK_SHA256 or
            lineage.get("null_fastpath_decision", {}).get("sha256") !=
            EXPECTED_DECISION_SHA256):
        raise ValueError("sole-edge plan lineage binding mismatch")
    policy = plan.get("evidence_policy", {})
    if (policy.get("attempts_per_exact_committed_head_and_inputs") != 1 or
            policy.get("candidate_files_must_equal_head_blobs") is not True or
            policy.get("clean_process_preflight_required") is not True or
            policy.get("clean_tracked_tree_required") is not True or
            policy.get("validation_or_final_banks_forbidden") is not True):
        raise ValueError("sole-edge evidence policy mismatch")
    thresholds = plan.get("thresholds", {})
    if thresholds != {
        "candidate_first_ms_max_lt": 990,
        "candidate_first_ms_p99_lt": 900,
        "candidate_later_ms_max_lt": 198,
        "candidate_later_ms_p99_lt": 180,
        "candidate_wins_by_color_min": [19, 19],
        "candidate_wins_min": 38,
        "failures_max": 0,
        "progress": "candidate_depth_avg >= reference_depth_avg OR candidate_nodes_avg > reference_nodes_avg",
        "reference_first_ms_max_lt": 990,
        "reference_first_ms_p99_lt": 900,
        "reference_later_ms_max_lt": 198,
        "reference_later_ms_p99_lt": 180,
        "unfinished_max": 0,
    }:
        raise ValueError("sole-edge plan threshold mismatch")
    if (rollback.get("schema") != "rank4-jacek-hybrid-null-fastpath-rollback-v1" or
            rollback.get("rollback_base", {}).get("algorithm") !=
            "archived-prefastpath-hybrid" or
            rollback.get("rollback_base", {}).get("operational_exact_proof_mask") != 7):
        raise ValueError("rollback manifest semantic binding mismatch")
    artifacts = {item.get("role"): item for item in rollback.get("production_artifacts", [])}
    if (artifacts.get("engine-source", {}).get("sha256") !=
            "34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b" or
            artifacts.get("upload-source", {}).get("sha256") !=
            "2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7" or
            rollback.get("verification", {}).get("fixed_work_parity", {}).get("status") != "passed"):
        raise ValueError("pre-singleton rollback identity/parity mismatch")
    if (decision.get("decision", {}).get("status") !=
            "null-fastpath-rejected-mandatory-revert" or
            decision.get("decision", {}).get("selected_algorithm") !=
            "archived-prefastpath-hybrid"):
        raise ValueError("null-fastpath decision lineage mismatch")
    return {"plan": plan, "rollback": rollback, "decision": decision}


def require_committed_bindings(head: str) -> dict[str, Any]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ROLLBACK_COMMIT, head],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("pre-singleton rollback commit is not an ancestor of HEAD")
    for relative, live in (
        ("submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp", CANDIDATE_BOT),
        ("submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp", CANDIDATE_SOURCE),
        ("submissions/codingame/bots/rank_4_jacek_hybrid/submission_test.cpp", CANDIDATE_TEST),
    ):
        if base.git_blob(head, relative) != live.read_bytes():
            raise ValueError(f"candidate is not the exact committed HEAD blob: {relative}")
    if base.git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp",
    ) != CONTROL_BOT.read_bytes():
        raise ValueError("archived reference bot differs from its source commit")
    if base.git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp",
    ) != CONTROL_SOURCE.read_bytes():
        raise ValueError("archived reference source differs from its source commit")
    return validate_plan_and_lineage()


def validate_exact_file_identities(identities: dict[str, dict[str, Any]]) -> None:
    expected = {
        CANDIDATE_BOT: (EXPECTED_CANDIDATE_BOT_SHA256, 63_350),
        CANDIDATE_SOURCE: (EXPECTED_CANDIDATE_SOURCE_SHA256, 94_527),
        CANDIDATE_TEST: (EXPECTED_CANDIDATE_TEST_SHA256, 40_103),
        CONTROL_BOT: (EXPECTED_CONTROL_BOT_SHA256, 62_777),
        CONTROL_SOURCE: (EXPECTED_CONTROL_SOURCE_SHA256, 94_004),
        BANK: (EXPECTED_BANK_SHA256, 13_150),
        PLAN: (EXPECTED_PLAN_SHA256, 4_509),
        ROLLBACK: (EXPECTED_ROLLBACK_SHA256, None),
        DECISION: (EXPECTED_DECISION_SHA256, None),
        CONTROL_MANIFEST: (EXPECTED_CONTROL_MANIFEST_SHA256, None),
    }
    for path in PROTOTYPE_FILES:
        digest, size = EXPECTED_PROTOTYPE[path.name]
        expected[path] = (digest, size)
    for path, (digest, size) in expected.items():
        key = str(path.relative_to(ROOT))
        identity = identities.get(key)
        if identity is None or identity.get("sha256") != digest:
            raise ValueError(f"exact input SHA-256 mismatch: {key}")
        if size is not None and identity.get("bytes") != size:
            raise ValueError(f"exact input byte count mismatch: {key}")
        if identity.get("ascii") is not True:
            raise ValueError(f"exact input ASCII mismatch: {key}")
    if identities[str(CANDIDATE_SOURCE.relative_to(ROOT))]["bytes"] > 99_999:
        raise ValueError("candidate exceeds CodinGame source limit")


def attempt_key(head: str) -> dict[str, str]:
    return {
        "bank_sha256": EXPECTED_BANK_SHA256,
        "candidate_bot_sha256": EXPECTED_CANDIDATE_BOT_SHA256,
        "candidate_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "candidate_test_sha256": EXPECTED_CANDIDATE_TEST_SHA256,
        "control_bot_sha256": EXPECTED_CONTROL_BOT_SHA256,
        "control_source_sha256": EXPECTED_CONTROL_SOURCE_SHA256,
        "decision_sha256": EXPECTED_DECISION_SHA256,
        "head": head,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "rollback_sha256": EXPECTED_ROLLBACK_SHA256,
        "schema": SCHEMA,
    }


def attempt_id(head: str) -> str:
    return sha256_bytes(canonical_json(attempt_key(head)))


def matching_attempts(head: str, output: Path = OUTPUT) -> list[Path]:
    expected = attempt_id(head)
    matches: list[Path] = []
    for path in sorted(output.glob("*.json")):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if (path.stem == sha256_bytes(raw) and isinstance(payload, dict) and
                canonical_json(payload) == raw and payload.get("schema") == SCHEMA and
                payload.get("attempt_id") == expected):
            matches.append(path)
    return matches


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another sole-edge recorder owns the lock", file=sys.stderr)
            return 2
        tracked_before = base.common.git_text("status", "--porcelain", "--untracked-files=no")
        full_before = base.common.git_text("status", "--porcelain")
        head_before = base.common.git_text("rev-parse", "HEAD")
        if tracked_before:
            print("tracked or staged files differ from HEAD", file=sys.stderr)
            return 2
        try:
            base.common.require_repository_inputs_tracked(TRACKED_INPUTS)
            bindings = require_committed_bindings(head_before)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(str(error), file=sys.stderr)
            return 2
        build = subprocess.run(
            ["cmake", "--build", "build", "--parallel", "2", "--target", GATE_TARGET],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        if build.returncode != 0 or build.stderr:
            print("fresh sole-edge comparison gate build failed", file=sys.stderr)
            return 2
        missing = [str(path) for path in TRACKED_INPUTS if not path.is_file()]
        if missing:
            print("missing inputs: " + ", ".join(missing), file=sys.stderr)
            return 2
        before = {
            str(path.relative_to(ROOT)): base.common.file_identity(path)
            for path in TRACKED_INPUTS
        }
        try:
            validate_exact_file_identities(before)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        generated_check = subprocess.run(
            ["node", "submissions/codingame/tools/generate_submission.mjs",
             "rank_4_jacek_hybrid", "--check"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if generated_check.returncode != 0 or generated_check.stderr:
            print("generated source current-check failed", file=sys.stderr)
            return 2
        if matching_attempts(head_before):
            print("an attempt already exists for this exact committed HEAD and inputs", file=sys.stderr)
            return 2
        try:
            preflight = base.require_clean_process_preflight()
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        command = command_for_gate()
        started = base.utc_now()
        monotonic_started = time.monotonic_ns()
        timed_out = False
        os_error: str | None = None
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False, timeout=RUN_TIMEOUT_SECONDS,
            )
            returncode: int | None = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out, returncode = True, None
            stdout, stderr = error.stdout or "", error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
        except OSError as error:
            os_error, returncode, stdout, stderr = f"{type(error).__name__}: {error}", None, "", ""
        elapsed_ns = time.monotonic_ns() - monotonic_started
        ended = base.utc_now()
        after = {
            str(path.relative_to(ROOT)): base.common.file_identity(path)
            for path in TRACKED_INPUTS
        }
        tracked_after = base.common.git_text("status", "--porcelain", "--untracked-files=no")
        full_after = base.common.git_text("status", "--porcelain")
        head_after = base.common.git_text("rev-parse", "HEAD")
        validation_errors: list[str] = []
        threshold_errors: list[str] = []
        parsed: dict[str, Any] = {}
        try:
            parsed = validate_gate_stdout(stdout)
            threshold_errors = selection_errors(parsed["aggregate"])
        except (ValueError, OverflowError) as error:
            validation_errors.append(str(error))
        stable_inputs = before == after
        acceptable = (
            returncode == 0 and not timed_out and os_error is None and stderr == "" and
            stable_inputs and head_before == head_after and tracked_before == "" and
            tracked_after == "" and not validation_errors and not threshold_errors
        )
        report = {
            "schema": SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "campaign_t0_utc": CAMPAIGN_T0_UTC,
            "classification": CLASSIFICATION,
            "final_qualification": False,
            "attempt_id": attempt_id(head_before),
            "attempt_key": attempt_key(head_before),
            "frozen_plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": EXPECTED_PLAN_SHA256},
            "lineage": bindings,
            "reference_slot": {"echo": "rank4", "semantics": "archived-prefastpath-hybrid", "function": "choose_prefastpath"},
            "candidate_exact_proof_mask": 7,
            "reference_exact_proof_mask": 7,
            "process_preflight": preflight,
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_monotonic_ns": elapsed_ns,
            "command_argv": command,
            "command_shell": shlex.join(command),
            "cwd": str(ROOT),
            "returncode": returncode,
            "timed_out": timed_out,
            "os_error": os_error,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "git": {"head_before": head_before, "head_after": head_after,
                    "tracked_status_before": tracked_before, "tracked_status_after": tracked_after,
                    "full_status_before": full_before, "full_status_after": full_after},
            "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
            "fresh_gate_build": {"target": GATE_TARGET, "returncode": build.returncode,
                                 "stdout": build.stdout, "stderr": build.stderr},
            "generated_source_check": {"returncode": generated_check.returncode,
                                       "stdout": generated_check.stdout, "stderr": generated_check.stderr},
            "inputs_sha256": sha256_bytes(canonical_json(before)),
            "inputs_before": before,
            "inputs_after": after,
            "stable_inputs": stable_inputs,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed,
            "validation_errors": validation_errors,
            "threshold_errors": threshold_errors,
            "development_ablation_acceptable": acceptable,
        }
        final_path, digest = persist_content_addressed_report(OUTPUT, report, os.getpid())
        print(final_path.relative_to(ROOT))
        print(f"sha256={digest}")
        print(f"development_ablation_acceptable={str(acceptable).lower()}")
        return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
