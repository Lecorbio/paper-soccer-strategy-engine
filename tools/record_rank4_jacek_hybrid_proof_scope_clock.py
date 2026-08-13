#!/usr/bin/env python3
"""Run one preregistered Jacek-hybrid proof-scope clock comparison.

The recorder intentionally has no knobs for banks, clocks, or work limits.  This
keeps the four proof-scope ablations on the frozen DEVELOPMENT d20 bank directly
comparable and prevents an accidental read of validation or final openings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import platform
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
GATE = ROOT / "build/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate"
GATE_TARGET = "papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate"
BANK = ROOT / "results/rank_4_jacek_hybrid/openings/development_d20.tsv"
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock"
LOCK = ROOT / "build/rank4-jacek-hybrid-proof-scope-clock.lock"
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"
RUN_TIMEOUT_SECONDS = 1_800
EXPECTED_SOURCE_SHA256 = (
    "6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f"
)
EXPECTED_SOURCE_BYTES = 94_004
EXPECTED_BANK_SHA256 = (
    "2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4"
)
ALLOWED_MASK_PAIRS = frozenset(((1, 0), (3, 1), (7, 3), (15, 7)))
TRACKED_INPUTS = (
    RECORDER,
    ROOT / "CMakeLists.txt",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/replay_value_model.hpp",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/teacher_residual_model.hpp",
    ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp",
    ROOT
    / "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_engine.hpp",
    ROOT
    / "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_hybrid.cpp",
    ROOT
    / "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_rank4.cpp",
    ROOT / "submissions/codingame/bots/rank_4/bot.cpp",
    ROOT / "submissions/codingame/bots/rank_4/replay_book.hpp",
    ROOT / "submissions/codingame/bots/rank_4/replay_value_model.hpp",
    ROOT / "submissions/codingame/bots/rank_4/teacher_residual_model.hpp",
    ROOT / "src/bots/mcts_internal.hpp",
    ROOT / "src/opening_bank/opening_bank.cpp",
    ROOT / "src/opening_bank/opening_bank_internal.hpp",
    ROOT / "src/core/rules.cpp",
    ROOT / "src/core/geometry.cpp",
    ROOT / "include/papersoccer/rules.hpp",
    ROOT / "include/papersoccer/geometry.hpp",
    ROOT / "include/papersoccer/types.hpp",
    ROOT / "build/CMakeCache.txt",
    ROOT
    / "build/CMakeFiles/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate.dir/flags.make",
    ROOT
    / "build/CMakeFiles/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate.dir/link.txt",
    BANK,
    GATE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ascii": all(byte < 128 for byte in data),
    }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if token.count("=") != 1:
            raise ValueError(f"malformed output token: {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError(f"duplicate/empty output token: {token!r}")
        fields[key] = value
    return fields


def exact_int(fields: dict[str, str], key: str) -> int:
    value = fields.get(key)
    if value is None or not value.isdigit():
        raise ValueError(f"missing/non-integer field: {key}")
    return int(value)


def exact_float(fields: dict[str, str], key: str) -> float:
    value = fields.get(key)
    if value is None:
        raise ValueError(f"missing field: {key}")
    parsed = float(value)
    if not (parsed >= 0.0):
        raise ValueError(f"invalid nonnegative field: {key}")
    return parsed


def parse_color(fields: dict[str, str], key: str) -> tuple[int, int, int, int, int]:
    value = fields.get(key, "")
    parts = value.split("/")
    if len(parts) != 5 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid color accounting: {key}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def parse_proof(fields: dict[str, str], engine: str, scope: str) -> tuple[int, ...]:
    value = fields.get(f"{engine}_proof_{scope}", "")
    parts = value.split("/")
    expected = 4 if scope in ("ply1", "ply2") else 3
    if len(parts) != expected or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid proof counter: {engine}/{scope}")
    return tuple(int(part) for part in parts)


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def require_repository_inputs_tracked(paths: tuple[Path, ...]) -> None:
    build_root = ROOT / "build"
    for path in paths:
        if path == GATE or path == BANK or path.is_relative_to(build_root):
            continue
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT))],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"recorder input is not tracked at HEAD: {path}")


def validate_summary(fields: dict[str, str], expected_bank: str,
                     candidate_mask: int, reference_mask: int) -> dict[str, Any]:
    if fields.get("bank") != expected_bank:
        raise ValueError("summary bank label mismatch")
    games = exact_int(fields, "games")
    candidate_wins = exact_int(fields, "candidate_wins")
    reference_wins = exact_int(fields, "reference_wins")
    unfinished = exact_int(fields, "unfinished")
    failed = exact_int(fields, "failed")
    if games != 76 or candidate_wins + reference_wins + unfinished + failed != games:
        raise ValueError("aggregate game accounting mismatch")
    if unfinished != 0 or failed != 0:
        raise ValueError("unfinished or failed games are not acceptable")
    colors = [parse_color(fields, f"candidate_p{color}") for color in range(2)]
    for color in colors:
        if sum(color[:4]) != color[4] or color[4] != 38:
            raise ValueError("per-color accounting mismatch")
    if (sum(color[0] for color in colors) != candidate_wins or
            sum(color[1] for color in colors) != reference_wins or
            sum(color[2] for color in colors) != unfinished or
            sum(color[3] for color in colors) != failed):
        raise ValueError("color sums do not match aggregate")
    for engine in ("candidate", "reference"):
        invocations = exact_int(fields, f"{engine}_invocations")
        searches = exact_int(fields, f"{engine}_searches")
        if invocations <= 0 or searches != invocations:
            raise ValueError(f"{engine} invocation/search accounting mismatch")
        for suffix in ("illegal", "operational", "exceptions", "hard_timeouts"):
            if exact_int(fields, f"{engine}_{suffix}") != 0:
                raise ValueError(f"{engine} {suffix} is nonzero")
        if exact_float(fields, f"{engine}_first_ms_max") >= 990.0:
            raise ValueError(f"{engine} first clock lacks headroom")
        if exact_float(fields, f"{engine}_later_ms_max") >= 198.0:
            raise ValueError(f"{engine} later clock lacks headroom")

    scope_bits = {"root": 1, "leaf": 2, "ply1": 4, "ply2": 8}
    proof: dict[str, dict[str, tuple[int, ...]]] = {}
    for engine, mask in (("candidate", candidate_mask), ("reference", reference_mask)):
        proof[engine] = {}
        for scope, bit in scope_bits.items():
            counters = parse_proof(fields, engine, scope)
            proof[engine][scope] = counters
            if mask & bit:
                if counters[0] <= 0:
                    raise ValueError(f"enabled {engine} {scope} scope did no work")
            elif any(counters):
                raise ValueError(f"disabled {engine} {scope} scope leaked work")
            if any(value > counters[0] for value in counters[1:3]):
                raise ValueError(f"{engine} {scope} hit count exceeds probes")
            if len(counters) == 4 and counters[3] != counters[1] + counters[2]:
                raise ValueError(f"{engine} {scope} cutoff accounting mismatch")
    return {"colors": colors, "proof": proof}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mask", required=True, type=int)
    parser.add_argument("--reference-mask", required=True, type=int)
    args = parser.parse_args()
    if not 0 <= args.candidate_mask <= 15:
        parser.error("candidate mask must be in 0..15")
    if not 0 <= args.reference_mask <= 15:
        parser.error("reference mask must be in 0..15")
    mask_pair = (args.candidate_mask, args.reference_mask)
    if mask_pair not in ALLOWED_MASK_PAIRS:
        parser.error(
            "mask pair must be one of 1/0, 3/1, 7/3, or 15/7"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another proof-scope recorder owns the lock", file=sys.stderr)
            return 2

        tracked_status_before = git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_before = git_text("status", "--porcelain")
        head_before = git_text("rev-parse", "HEAD")
        if tracked_status_before:
            print("tracked or staged files differ from HEAD", file=sys.stderr)
            return 2
        try:
            require_repository_inputs_tracked(TRACKED_INPUTS)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        build = subprocess.run(
            ["cmake", "--build", "build", "--parallel", "2", "--target",
             GATE_TARGET],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if build.returncode != 0 or build.stderr:
            print("fresh comparison-gate build failed", file=sys.stderr)
            if build.stderr:
                print(build.stderr, file=sys.stderr, end="")
            return 2

        missing = [str(path) for path in TRACKED_INPUTS if not path.is_file()]
        if missing:
            print("missing inputs: " + ", ".join(missing), file=sys.stderr)
            return 2
        before = {
            str(path.relative_to(ROOT)): file_identity(path)
            for path in TRACKED_INPUTS
        }
        source_identity = before[
            "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp"
        ]
        if (source_identity["sha256"] != EXPECTED_SOURCE_SHA256 or
                source_identity["bytes"] != EXPECTED_SOURCE_BYTES or
                not source_identity["ascii"] or
                source_identity["bytes"] > 99_999):
            print("exact generated source identity mismatch", file=sys.stderr)
            return 2
        if before[str(BANK.relative_to(ROOT))]["sha256"] != EXPECTED_BANK_SHA256:
            print("frozen DEVELOPMENT d20 bank SHA-256 mismatch", file=sys.stderr)
            return 2

        generated_check = subprocess.run(
            ["node", "submissions/codingame/tools/generate_submission.mjs",
             "rank_4_jacek_hybrid", "--check"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if generated_check.returncode != 0 or generated_check.stderr:
            print("generated source current-check failed", file=sys.stderr)
            return 2

        command = [
            str(GATE),
            "--profile",
            "clock",
            "--reference-engine",
            "hybrid-control",
            "--bank",
            str(BANK),
            "--expected-role",
            "development",
            "--expected-depths",
            "20",
            "--expected-seeds",
            "4624785204876369057",
            "--expected-sha256",
            EXPECTED_BANK_SHA256,
            "--max-turns",
            "320",
            "--candidate-nodes",
            "3000000",
            "--reference-nodes",
            "3000000",
            "--candidate-first-ms",
            "800",
            "--candidate-later-ms",
            "165",
            "--reference-first-ms",
            "800",
            "--reference-later-ms",
            "165",
            "--operational-first-ms",
            "1000",
            "--operational-later-ms",
            "200",
            "--candidate-exact-proof-mask",
            str(args.candidate_mask),
            "--reference-exact-proof-mask",
            str(args.reference_mask),
        ]
        started = utc_now()
        monotonic_started = time.monotonic_ns()
        timed_out = False
        os_error: str | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
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
            str(path.relative_to(ROOT)): file_identity(path)
            for path in TRACKED_INPUTS
        }
        stable_inputs = before == after

        tracked_status_after = git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_after = git_text("status", "--porcelain")
        head_after = git_text("rev-parse", "HEAD")
        stdout_lines = [line for line in stdout.splitlines() if line.strip()]
        aggregate_lines = [line for line in stdout_lines if line.startswith("summary ")]
        configuration_lines = [
            line for line in stdout_lines if line.startswith("configuration ")
        ]
        bank_lines = [line for line in stdout_lines if line.startswith("bank_summary ")]
        validation_errors: list[str] = []
        aggregate: dict[str, str] = {}
        bank_summary: dict[str, str] = {}
        configuration: dict[str, str] = {}
        detailed_validation: dict[str, Any] = {}
        expected_configuration = {
            "profile": "clock",
            "reference_engine": "hybrid-control",
            "bank_count": "1",
            "expected_role": "development",
            "bank_validation":
                "schema,header,role,depth,seed,replay,state-sha256,canonical-sha256,disjoint",
            "max_turns": "320",
            "expected_depths": "20",
            "expected_seeds": "4624785204876369057",
            "expected_sha256": EXPECTED_BANK_SHA256,
            "candidate_nodes": "3000000",
            "reference_nodes": "3000000",
            "candidate_clock": "800/165",
            "reference_clock": "800/165",
            "operational_clock": "1000/200",
            "candidate_exact_proof_mask": str(args.candidate_mask),
            "reference_exact_proof_mask": str(args.reference_mask),
            "openings": "preregistered-public-rules",
            "replay_corrections": "disabled",
            "transcripts": "not-retained",
        }
        try:
            if len(stdout_lines) != 3 or len(aggregate_lines) != 1 or \
                    len(configuration_lines) != 1 or len(bank_lines) != 1:
                raise ValueError("gate stdout is not exactly three expected lines")
            aggregate = parse_fields(aggregate_lines[0])
            bank_summary = parse_fields(bank_lines[0])
            configuration = parse_fields(configuration_lines[0])
            if configuration != expected_configuration:
                raise ValueError("complete configuration echo mismatch")
            detailed_validation = validate_summary(
                aggregate, "all", args.candidate_mask, args.reference_mask
            )
            validate_summary(
                bank_summary, "0", args.candidate_mask, args.reference_mask
            )
            aggregate_without_bank = dict(aggregate)
            bank_without_bank = dict(bank_summary)
            aggregate_without_bank.pop("bank", None)
            bank_without_bank.pop("bank", None)
            if aggregate_without_bank != bank_without_bank:
                raise ValueError("single bank and aggregate summaries differ")
            for engine in ("candidate", "reference"):
                rebound = parse_proof(aggregate, engine, "rebound")
                scope_sums = [0, 0, 0]
                for scope in ("root", "leaf", "ply1", "ply2"):
                    counters = parse_proof(aggregate, engine, scope)
                    for index in range(3):
                        scope_sums[index] += counters[index]
                if tuple(scope_sums) != rebound:
                    raise ValueError(f"{engine} aggregate proof counters mismatch")
        except (ValueError, OverflowError) as error:
            validation_errors.append(str(error))

        development_ablation_acceptable = (
            returncode == 0
            and not timed_out
            and os_error is None
            and stderr == ""
            and stable_inputs
            and head_before == head_after
            and tracked_status_before == ""
            and tracked_status_after == ""
            and not validation_errors
        )

        report = {
            "schema": "rank4-jacek-hybrid-proof-scope-clock-v2",
            "campaign_id": CAMPAIGN_ID,
            "campaign_t0_utc": CAMPAIGN_T0_UTC,
            "classification": (
                "development-proof-scope-ablation-not-final-qualification"
            ),
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_monotonic_ns": monotonic_ended - monotonic_started,
            "candidate_exact_proof_mask": args.candidate_mask,
            "reference_exact_proof_mask": args.reference_mask,
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
                "tracked_status_before": tracked_status_before,
                "tracked_status_after": tracked_status_after,
                "full_status_before": full_status_before,
                "full_status_after": full_status_after,
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
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True,
                ).stdout.splitlines()[0],
            },
            "generated_source_check": {
                "returncode": generated_check.returncode,
                "stdout": generated_check.stdout,
                "stderr": generated_check.stderr,
            },
            "fresh_gate_build": {
                "target": GATE_TARGET,
                "returncode": build.returncode,
                "stdout": build.stdout,
                "stderr": build.stderr,
            },
            "inputs_before": before,
            "inputs_after": after,
            "stable_inputs": stable_inputs,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": {
                "aggregate_line": aggregate_lines[-1] if len(aggregate_lines) == 1 else None,
                "aggregate": aggregate,
                "bank_lines": bank_lines,
                "configuration_line": (
                    configuration_lines[-1] if len(configuration_lines) == 1 else None
                ),
                "configuration": configuration,
                "expected_configuration": expected_configuration,
                "validation_errors": validation_errors,
                "detailed_validation": detailed_validation,
            },
            "development_ablation_acceptable": development_ablation_acceptable,
            "final_qualification": False,
        }
        canonical = json.dumps(
            report, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii") + b"\n"
        report_sha256 = hashlib.sha256(canonical).hexdigest()
        destination = OUTPUT / f"{report_sha256}.json"
        temporary = OUTPUT / f".{report_sha256}.{os.getpid()}.tmp"
        temporary.write_bytes(canonical)
        os.replace(temporary, destination)
        persisted = destination.read_bytes()
        if persisted != canonical or hashlib.sha256(persisted).hexdigest() != report_sha256:
            print("persisted report failed byte/hash verification", file=sys.stderr)
            return 2

        print(f"report={destination.relative_to(ROOT)}")
        print(f"report_sha256={report_sha256}")
        if aggregate_lines:
            print(aggregate_lines[-1])
        print(f"stable_inputs={str(stable_inputs).lower()}")
        print(
            "development_ablation_acceptable="
            f"{str(development_ablation_acceptable).lower()}"
        )
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if validation_errors:
            print("; ".join(validation_errors), file=sys.stderr)
        return 0 if development_ablation_acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
