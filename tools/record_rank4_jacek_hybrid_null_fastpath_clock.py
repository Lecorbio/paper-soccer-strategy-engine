#!/usr/bin/env python3
"""Record the one frozen DEVELOPMENT d20 null-fastpath clock ablation.

The dedicated executable maps its generic ``rank4`` reference slot to the
content-addressed pre-fastpath hybrid from commit 2a37f20.  This recorder has
no bank, clock, mask, or retry knobs and cannot address validation/final data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import record_rank4_jacek_hybrid_proof_scope_clock as common


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/null_fastpath_clock"
ORIGINAL_PLAN = OUTPUT / "plan.json"
PLAN = OUTPUT / "plan_v2.json"
AUDIT_RECEIPT = OUTPUT / "static_audit_receipt.json"
REJECTED_ATTEMPT = OUTPUT / (
    "74576b4a30d48f4e596ea0f52ab9c5498e1b1da15b323f848dcc1d603dc710ab.json"
)
LOCK = ROOT / "build/rank4-jacek-hybrid-null-fastpath-clock.lock"
GATE_TARGET = (
    "papersoccer_codingame_rank_4_jacek_hybrid_"
    "null_fastpath_comparison_gate"
)
GATE = ROOT / "build" / GATE_TARGET
BANK = ROOT / "results/rank_4_jacek_hybrid/openings/development_d20.tsv"
RUN_TIMEOUT_SECONDS = 1_800

SCHEMA = "rank4-jacek-hybrid-null-fastpath-clock-v2"
CLASSIFICATION = (
    "development-null-fastpath-clock-ablation-not-final-qualification"
)
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"

CANDIDATE_ORIGIN_COMMIT = "cf02c80fc2f30d0c0941030ffb18e775501d9e17"
CONTROL_SOURCE_COMMIT = "2a37f20727da0e90e71a5d0b71a44cd0ae87ffbd"
EXPECTED_CANDIDATE_BOT_SHA256 = (
    "8853e4a9ac980b3ce1958faf515f9531094d4a174e823c48d7016c0f6387e9a3"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "6459cec3c3e2552dc74fd1eb61eb0b154662c61004c62608478d17ccaf3f50c5"
)
EXPECTED_CANDIDATE_SOURCE_BYTES = 94_574
EXPECTED_CANDIDATE_TEST_SHA256 = (
    "046db19a6f2b5c0c55e92c331ba353e6e398a9f9bafc7dc42ced7d264f590372"
)
EXPECTED_CONTROL_BOT_SHA256 = (
    "c8412600f0b90610660f24a02828e77f67cc7d78bdf775da11628b3355274215"
)
EXPECTED_CONTROL_SOURCE_SHA256 = (
    "6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f"
)
EXPECTED_CONTROL_SOURCE_BYTES = 94_004
EXPECTED_BANK_SHA256 = (
    "2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4"
)
EXPECTED_ORIGINAL_PLAN_SHA256 = (
    "29d0a718747078a46f71e3f4466cfba96377fa7b9a19967fd4684933a29bdbf6"
)
EXPECTED_PLAN_SHA256 = (
    "0db7f944df34b857a3930896869abe7b9c16a2f1ccd9cd0b90e16650fa6b5a9b"
)
EXPECTED_AUDIT_RECEIPT_SHA256 = (
    "4245451b27985365494a0fc69f2c562efda4fe5a1c720dafe71cf0807e0e8fe6"
)
EXPECTED_REJECTED_ATTEMPT_SHA256 = (
    "74576b4a30d48f4e596ea0f52ab9c5498e1b1da15b323f848dcc1d603dc710ab"
)
EXPECTED_MANIFEST_SHA256 = (
    "d94204c4d314332e439e38de774d0e110c73910b961bd0f0152252b7404ae772"
)

BOT_DIRECTORY = ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid"
CANDIDATE_BOT = BOT_DIRECTORY / "bot.cpp"
CANDIDATE_SOURCE = BOT_DIRECTORY / "submission.cpp"
CANDIDATE_TEST = BOT_DIRECTORY / "submission_test.cpp"
CONTROL_DIRECTORY = (
    ROOT / "results/rank_4_jacek_hybrid/controls/null_fastpath_pre_2a37f20"
)
CONTROL_BOT = CONTROL_DIRECTORY / (
    "bot.c8412600f0b90610660f24a02828e77f67cc7d78bdf775da11628b3355274215.cpp"
)
CONTROL_SOURCE = CONTROL_DIRECTORY / (
    "submission.6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f.cpp"
)
CONTROL_MANIFEST = CONTROL_DIRECTORY / "manifest.json"

COMMIT_BOUND_DEPENDENCIES = (
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_value_model.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/teacher_residual_model.hpp",
    "src/bots/mcts_internal.hpp",
)

TRACKED_INPUTS = (
    RECORDER,
    ROOT / "tools/record_rank4_jacek_hybrid_proof_scope_clock.py",
    ROOT / "CMakeLists.txt",
    ORIGINAL_PLAN,
    PLAN,
    AUDIT_RECEIPT,
    REJECTED_ATTEMPT,
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


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read frozen git blob: {commit}:{path}")
    return result.stdout


def expected_configuration() -> dict[str, str]:
    return {
        "profile": "clock",
        # This label is the generic gate slot. The dedicated executable binds
        # it to choose_prefastpath, not to the public Rank-4 engine.
        "reference_engine": "rank4",
        "bank_count": "1",
        "expected_role": "development",
        "bank_validation": (
            "schema,header,role,depth,seed,replay,state-sha256,"
            "canonical-sha256,disjoint"
        ),
        "max_turns": "320",
        "expected_depths": "20",
        "expected_seeds": "4624785204876369057",
        "expected_sha256": EXPECTED_BANK_SHA256,
        "candidate_nodes": "3000000",
        "reference_nodes": "3000000",
        "candidate_clock": "800/165",
        "reference_clock": "800/165",
        "operational_clock": "1000/200",
        "candidate_exact_proof_mask": "7",
        "reference_exact_proof_mask": "7",
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }


def command_for_gate() -> list[str]:
    return [
        str(GATE),
        "--profile", "clock",
        "--reference-engine", "rank4",
        "--bank", str(BANK),
        "--expected-role", "development",
        "--expected-depths", "20",
        "--expected-seeds", "4624785204876369057",
        "--expected-sha256", EXPECTED_BANK_SHA256,
        "--max-turns", "320",
        "--candidate-nodes", "3000000",
        "--reference-nodes", "3000000",
        "--candidate-first-ms", "800",
        "--candidate-later-ms", "165",
        "--reference-first-ms", "800",
        "--reference-later-ms", "165",
        "--operational-first-ms", "1000",
        "--operational-later-ms", "200",
        "--candidate-exact-proof-mask", "7",
        "--reference-exact-proof-mask", "7",
    ]


def finite_nonnegative(fields: dict[str, str], key: str) -> float:
    value = common.exact_float(fields, key)
    if not math.isfinite(value):
        raise ValueError(f"field is not finite: {key}")
    return value


def validate_rebound_accounting(fields: dict[str, str]) -> None:
    for engine in ("candidate", "reference"):
        rebound = common.parse_proof(fields, engine, "rebound")
        scopes = [0, 0, 0]
        for scope in ("root", "leaf", "ply1", "ply2"):
            counters = common.parse_proof(fields, engine, scope)
            for index in range(3):
                scopes[index] += counters[index]
        if rebound != tuple(scopes):
            raise ValueError(f"{engine} rebound/scope counters mismatch")


def validate_timing(fields: dict[str, str]) -> None:
    for engine in ("candidate", "reference"):
        first_p99 = finite_nonnegative(fields, f"{engine}_first_ms_p99")
        first_max = finite_nonnegative(fields, f"{engine}_first_ms_max")
        later_p99 = finite_nonnegative(fields, f"{engine}_later_ms_p99")
        later_max = finite_nonnegative(fields, f"{engine}_later_ms_max")
        if first_p99 > first_max or later_p99 > later_max:
            raise ValueError(f"{engine} timing p99 exceeds maximum")
        if first_p99 >= 900.0 or first_max >= 990.0:
            raise ValueError(f"{engine} first timing gate failed")
        if later_p99 >= 180.0 or later_max >= 198.0:
            raise ValueError(f"{engine} later timing gate failed")


def validate_gate_stdout(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    bank_lines = [line for line in lines if line.startswith("bank_summary ")]
    summary_lines = [line for line in lines if line.startswith("summary ")]
    config_lines = [line for line in lines if line.startswith("configuration ")]
    if (len(lines) != 3 or len(bank_lines) != 1 or
            len(summary_lines) != 1 or len(config_lines) != 1):
        raise ValueError("gate stdout is not exactly three expected lines")

    bank = common.parse_fields(bank_lines[0])
    aggregate = common.parse_fields(summary_lines[0])
    configuration = common.parse_fields(config_lines[0])
    if configuration != expected_configuration():
        raise ValueError("complete configuration echo mismatch")

    common.validate_summary(
        bank, "0", 7, 7, expected_games=76, expected_color_games=38
    )
    common.validate_summary(
        aggregate, "all", 7, 7,
        expected_games=76, expected_color_games=38,
    )
    validate_rebound_accounting(bank)
    validate_rebound_accounting(aggregate)
    validate_timing(bank)
    validate_timing(aggregate)

    bank_without_label = {key: value for key, value in bank.items()
                          if key != "bank"}
    aggregate_without_label = {key: value for key, value in aggregate.items()
                               if key != "bank"}
    if bank_without_label != aggregate_without_label:
        raise ValueError("single-bank and aggregate summaries differ")

    return {
        "bank_line": bank_lines[0],
        "bank": bank,
        "aggregate_line": summary_lines[0],
        "aggregate": aggregate,
        "configuration_line": config_lines[0],
        "configuration": configuration,
    }


def selection_errors(aggregate: dict[str, str]) -> list[str]:
    errors: list[str] = []
    if common.exact_int(aggregate, "candidate_wins") < 38:
        errors.append("candidate has fewer than 38 wins")
    for color in range(2):
        wins = common.parse_color(aggregate, f"candidate_p{color}")[0]
        if wins < 19:
            errors.append(
                f"candidate has fewer than 19 wins as physical color {color}"
            )

    candidate_depth = finite_nonnegative(aggregate, "candidate_depth_avg")
    reference_depth = finite_nonnegative(aggregate, "reference_depth_avg")
    candidate_nodes = finite_nonnegative(aggregate, "candidate_nodes_avg")
    reference_nodes = finite_nonnegative(aggregate, "reference_nodes_avg")
    if candidate_depth < reference_depth and candidate_nodes <= reference_nodes:
        errors.append(
            "candidate has lower completed depth without improved node throughput"
        )
    return errors


def attempt_key(head: str) -> dict[str, str]:
    return {
        "head": head,
        "schema": SCHEMA,
        "original_plan_sha256": EXPECTED_ORIGINAL_PLAN_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "audit_receipt_sha256": EXPECTED_AUDIT_RECEIPT_SHA256,
        "rejected_attempt_sha256": EXPECTED_REJECTED_ATTEMPT_SHA256,
        "candidate_bot_sha256": EXPECTED_CANDIDATE_BOT_SHA256,
        "candidate_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "control_bot_sha256": EXPECTED_CONTROL_BOT_SHA256,
        "control_source_sha256": EXPECTED_CONTROL_SOURCE_SHA256,
        "bank_sha256": EXPECTED_BANK_SHA256,
    }


def attempt_id(head: str) -> str:
    return sha256_bytes(canonical_json(attempt_key(head)))


def matching_attempts(head: str, output: Path = OUTPUT) -> list[Path]:
    expected = attempt_id(head)
    matches: list[Path] = []
    for path in sorted(output.glob("*.json")):
        try:
            raw = path.read_bytes()
            digest = sha256_bytes(raw)
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if (path.stem == digest and isinstance(payload, dict) and
                canonical_json(payload) == raw and
                payload.get("schema") == SCHEMA and
                payload.get("attempt_id") == expected):
            matches.append(path)
    return matches


def validate_canonical_json_file(path: Path, expected_sha256: str) -> Any:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen JSON SHA-256 mismatch: {path}")
    payload = json.loads(raw)
    if canonical_json(payload) != raw:
        raise ValueError(f"frozen JSON is not canonical: {path}")
    return payload


def require_origin_and_archive_bindings() -> dict[str, Any]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CANDIDATE_ORIGIN_COMMIT,
         "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("candidate origin commit is not an ancestor of HEAD")

    candidate_paths = {
        "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp":
            CANDIDATE_BOT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp":
            CANDIDATE_SOURCE,
        "submissions/codingame/bots/rank_4_jacek_hybrid/submission_test.cpp":
            CANDIDATE_TEST,
    }
    for source_path, live_path in candidate_paths.items():
        if git_blob(CANDIDATE_ORIGIN_COMMIT, source_path) != live_path.read_bytes():
            raise ValueError(f"candidate changed since origin commit: {source_path}")

    if git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp",
    ) != CONTROL_BOT.read_bytes():
        raise ValueError("archived control bot differs from source commit")
    if git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp",
    ) != CONTROL_SOURCE.read_bytes():
        raise ValueError("archived control source differs from source commit")

    for dependency in COMMIT_BOUND_DEPENDENCIES:
        live = (ROOT / dependency).read_bytes()
        if git_blob(CANDIDATE_ORIGIN_COMMIT, dependency) != live:
            raise ValueError(f"candidate dependency changed: {dependency}")
        if git_blob(CONTROL_SOURCE_COMMIT, dependency) != live:
            raise ValueError(f"control dependency changed: {dependency}")

    original_plan = validate_canonical_json_file(
        ORIGINAL_PLAN, EXPECTED_ORIGINAL_PLAN_SHA256
    )
    plan = validate_canonical_json_file(PLAN, EXPECTED_PLAN_SHA256)
    audit_receipt = validate_canonical_json_file(
        AUDIT_RECEIPT, EXPECTED_AUDIT_RECEIPT_SHA256
    )
    rejected_attempt = validate_canonical_json_file(
        REJECTED_ATTEMPT, EXPECTED_REJECTED_ATTEMPT_SHA256
    )
    manifest = validate_canonical_json_file(
        CONTROL_MANIFEST, EXPECTED_MANIFEST_SHA256
    )
    if plan.get("schema") != "rank4-jacek-hybrid-null-fastpath-plan-v2":
        raise ValueError("amended plan schema mismatch")
    amendment = plan.get("amendment", {})
    if (amendment.get("previous_plan", {}).get("sha256") !=
            EXPECTED_ORIGINAL_PLAN_SHA256 or
            amendment.get("audit_receipt", {}).get("sha256") !=
            EXPECTED_AUDIT_RECEIPT_SHA256 or
            amendment.get("premature_attempt", {}).get("sha256") !=
            EXPECTED_REJECTED_ATTEMPT_SHA256 or
            amendment.get("gameplay_semantics_changed") is not False or
            amendment.get("technical_gate_semantics_changed") is not False):
        raise ValueError("amended plan prerequisite binding mismatch")
    for key in ("bank", "candidate", "command", "reference", "thresholds"):
        if plan.get(key) != original_plan.get(key):
            raise ValueError(f"amended plan changed frozen semantics: {key}")
    if (audit_receipt.get("technical_audit", {}).get("status") != "pass" or
            audit_receipt.get("premature_attempt", {}).get("sha256") !=
            EXPECTED_REJECTED_ATTEMPT_SHA256 or
            audit_receipt.get("premature_attempt", {}).get(
                "gameplay_result_accepted") is not False):
        raise ValueError("static audit receipt prerequisite mismatch")
    if (rejected_attempt.get("schema") !=
            "rank4-jacek-hybrid-null-fastpath-clock-v1" or
            rejected_attempt.get("returncode") != -15 or
            rejected_attempt.get("development_ablation_acceptable") is not False or
            rejected_attempt.get("stdout") != "" or
            rejected_attempt.get("stderr") != ""):
        raise ValueError("premature rejected-attempt prerequisite mismatch")
    return {
        "original_plan": original_plan,
        "plan_v2": plan,
        "audit_receipt": audit_receipt,
        "rejected_attempt": rejected_attempt,
        "control_manifest": manifest,
    }


def validate_exact_file_identities(
    identities: dict[str, dict[str, Any]],
) -> None:
    expected = {
        str(CANDIDATE_BOT.relative_to(ROOT)):
            (EXPECTED_CANDIDATE_BOT_SHA256, 63_587, True),
        str(CANDIDATE_SOURCE.relative_to(ROOT)):
            (EXPECTED_CANDIDATE_SOURCE_SHA256,
             EXPECTED_CANDIDATE_SOURCE_BYTES, True),
        str(CANDIDATE_TEST.relative_to(ROOT)):
            (EXPECTED_CANDIDATE_TEST_SHA256, 41_408, True),
        str(CONTROL_BOT.relative_to(ROOT)):
            (EXPECTED_CONTROL_BOT_SHA256, 62_777, True),
        str(CONTROL_SOURCE.relative_to(ROOT)):
            (EXPECTED_CONTROL_SOURCE_SHA256,
             EXPECTED_CONTROL_SOURCE_BYTES, True),
        str(BANK.relative_to(ROOT)): (EXPECTED_BANK_SHA256, 13_150, True),
        str(ORIGINAL_PLAN.relative_to(ROOT)):
            (EXPECTED_ORIGINAL_PLAN_SHA256, None, True),
        str(PLAN.relative_to(ROOT)): (EXPECTED_PLAN_SHA256, None, True),
        str(AUDIT_RECEIPT.relative_to(ROOT)):
            (EXPECTED_AUDIT_RECEIPT_SHA256, None, True),
        str(REJECTED_ATTEMPT.relative_to(ROOT)):
            (EXPECTED_REJECTED_ATTEMPT_SHA256, None, True),
        str(CONTROL_MANIFEST.relative_to(ROOT)):
            (EXPECTED_MANIFEST_SHA256, None, True),
    }
    for path, (digest, size, ascii_expected) in expected.items():
        identity = identities.get(path)
        if identity is None or identity.get("sha256") != digest:
            raise ValueError(f"exact input SHA-256 mismatch: {path}")
        if size is not None and identity.get("bytes") != size:
            raise ValueError(f"exact input byte count mismatch: {path}")
        if identity.get("ascii") is not ascii_expected:
            raise ValueError(f"exact input ASCII mismatch: {path}")
    source = identities[str(CANDIDATE_SOURCE.relative_to(ROOT))]
    if source["bytes"] > 99_999:
        raise ValueError("candidate exceeds CodinGame source limit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another null-fastpath recorder owns the lock", file=sys.stderr)
            return 2

        tracked_before = common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_before = common.git_text("status", "--porcelain")
        head_before = common.git_text("rev-parse", "HEAD")
        if tracked_before:
            print("tracked or staged files differ from HEAD", file=sys.stderr)
            return 2
        try:
            common.require_repository_inputs_tracked(TRACKED_INPUTS)
            bindings = require_origin_and_archive_bindings()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(str(error), file=sys.stderr)
            return 2

        build = subprocess.run(
            ["cmake", "--build", "build", "--parallel", "2", "--target",
             GATE_TARGET], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if build.returncode != 0 or build.stderr:
            print("fresh null-fastpath gate build failed", file=sys.stderr)
            if build.stderr:
                print(build.stderr, file=sys.stderr, end="")
            return 2

        missing = [str(path) for path in TRACKED_INPUTS if not path.is_file()]
        if missing:
            print("missing inputs: " + ", ".join(missing), file=sys.stderr)
            return 2
        before = {
            str(path.relative_to(ROOT)): common.file_identity(path)
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

        prior = matching_attempts(head_before)
        if prior:
            print(
                "an attempt already exists for this exact HEAD and frozen "
                "ablation; retries are forbidden", file=sys.stderr,
            )
            return 2

        command = command_for_gate()
        started = utc_now()
        monotonic_started = time.monotonic_ns()
        timed_out = False
        os_error: str | None = None
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
                timeout=RUN_TIMEOUT_SECONDS,
            )
            returncode: int | None = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = None
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
        except OSError as error:
            os_error = f"{type(error).__name__}: {error}"
            returncode = None
            stdout = ""
            stderr = ""
        monotonic_ended = time.monotonic_ns()
        ended = utc_now()

        after = {
            str(path.relative_to(ROOT)): common.file_identity(path)
            for path in TRACKED_INPUTS
        }
        stable_inputs = before == after
        tracked_after = common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_after = common.git_text("status", "--porcelain")
        head_after = common.git_text("rev-parse", "HEAD")

        validation_errors: list[str] = []
        threshold_errors: list[str] = []
        parsed: dict[str, Any] = {}
        try:
            parsed = validate_gate_stdout(stdout)
            threshold_errors = selection_errors(parsed["aggregate"])
        except (ValueError, OverflowError) as error:
            validation_errors.append(str(error))

        acceptable = (
            returncode == 0 and not timed_out and os_error is None and
            stderr == "" and stable_inputs and
            head_before == head_after and tracked_before == "" and
            tracked_after == "" and not validation_errors and
            not threshold_errors
        )
        input_digest = sha256_bytes(canonical_json(before))
        report = {
            "schema": SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "campaign_t0_utc": CAMPAIGN_T0_UTC,
            "classification": CLASSIFICATION,
            "final_qualification": False,
            "attempt_id": attempt_id(head_before),
            "attempt_key": attempt_key(head_before),
            "candidate_origin_commit": CANDIDATE_ORIGIN_COMMIT,
            "control_source_commit": CONTROL_SOURCE_COMMIT,
            "reference_slot": {
                "echo": "rank4",
                "semantics": "archived-prefastpath-hybrid",
                "function": "choose_prefastpath",
            },
            "candidate_exact_proof_mask": 7,
            "reference_exact_proof_mask": 7,
            "frozen_plan": {
                "path": str(PLAN.relative_to(ROOT)),
                "sha256": EXPECTED_PLAN_SHA256,
            },
            "amendment_prerequisites": {
                "original_plan": {
                    "path": str(ORIGINAL_PLAN.relative_to(ROOT)),
                    "sha256": EXPECTED_ORIGINAL_PLAN_SHA256,
                },
                "audit_receipt": {
                    "path": str(AUDIT_RECEIPT.relative_to(ROOT)),
                    "sha256": EXPECTED_AUDIT_RECEIPT_SHA256,
                },
                "rejected_attempt": {
                    "path": str(REJECTED_ATTEMPT.relative_to(ROOT)),
                    "sha256": EXPECTED_REJECTED_ATTEMPT_SHA256,
                    "accepted_gameplay_result": False,
                },
            },
            "origin_bindings": bindings,
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_monotonic_ns": monotonic_ended - monotonic_started,
            "command_argv": command,
            "command_shell": shlex.join(command),
            "cwd": str(ROOT),
            "returncode": returncode,
            "timed_out": timed_out,
            "os_error": os_error,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "git": {
                "head_before": head_before,
                "head_after": head_after,
                "tracked_status_before": tracked_before,
                "tracked_status_after": tracked_after,
                "full_status_before": full_before,
                "full_status_after": full_after,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cmake_version": subprocess.run(
                    ["cmake", "--version"], text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=True,
                ).stdout.splitlines()[0],
                "cxx_version": subprocess.run(
                    ["/usr/bin/c++", "--version"], text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                ).stdout.splitlines()[0],
            },
            "fresh_gate_build": {
                "target": GATE_TARGET,
                "returncode": build.returncode,
                "stdout": build.stdout,
                "stderr": build.stderr,
            },
            "generated_source_check": {
                "returncode": generated_check.returncode,
                "stdout": generated_check.stdout,
                "stderr": generated_check.stderr,
            },
            "inputs_sha256": input_digest,
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
        raw = canonical_json(report)
        digest = sha256_bytes(raw)
        final_path = OUTPUT / f"{digest}.json"
        temporary = OUTPUT / f".{digest}.{os.getpid()}.tmp"
        temporary.write_bytes(raw)
        os.replace(temporary, final_path)
        print(final_path.relative_to(ROOT))
        print(f"sha256={digest}")
        print(f"development_ablation_acceptable={str(acceptable).lower()}")
        return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
