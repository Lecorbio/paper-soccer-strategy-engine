#!/usr/bin/env python3
"""Produce the fixed final-source preflight for held-out qualification.

This program has no path, compiler, target, test, bank, or environment knobs.
It builds the committed finalist with independently discovered Clang and GNU
compilers, runs the fixed release and sanitizer test panels, and executes one
small DEVELOPMENT-only comparison-gate contract probe.  It never addresses a
VALIDATION or FINAL bank.  Successful evidence is canonical,
content-addressed JSON under the fixed held-out preflight directory.

The qualification binder treats this tracked producer and its exact command
plan as part of the candidate identity.  Merely writing a JSON document with
"passed" strings is therefore insufficient: every command record, compiler
blob, build artifact, source identity, environment, and host identity is
revalidated by the binder.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = Path(__file__).resolve()
QUALIFICATION_RECORDER = (
    ROOT / "tools/record_rank4_jacek_hybrid_heldout_qualification.py"
)
PRODUCER_TEST = (
    ROOT / "tests/codingame/"
    "test_rank4_jacek_hybrid_final_source_preflight.py"
)
QUALIFICATION_TEST = (
    ROOT / "tests/codingame/"
    "test_rank4_jacek_hybrid_heldout_qualification.py"
)
PLAN = ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification/PLAN.json"
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification/preflight"
RECEIPTS = OUTPUT / "receipts"
CLAIMS = OUTPUT / "claims"
LOCK = ROOT / "build/rank4-jacek-hybrid-heldout-preflight.lock"
BENCHMARK_LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
BUILD_ROOT = ROOT / "build/rank4-jacek-hybrid-heldout-preflight"
CLANG_BUILD = BUILD_ROOT / "clang-release"
GNU_BUILD = BUILD_ROOT / "gnu-release"
SANITIZER_BUILD = BUILD_ROOT / "clang-sanitized"
TEMPORARY_DIRECTORY = BUILD_ROOT / "tmp"

SCHEMA = "rank4-jacek-hybrid-final-source-preflight-v2"
CLAIM_SCHEMA = "rank4-jacek-hybrid-final-source-preflight-claim-v1"
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"
ORDINARY_GATE_TARGET = (
    "papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate"
)
GATE_TARGET = (
    "papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate"
)
FINAL_GATE = CLANG_BUILD / GATE_TARGET
BOT = ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid"
SOURCE = BOT / "submission.cpp"
ENGINE = BOT / "bot.cpp"
SOURCE_TEST = BOT / "submission_test.cpp"
POSITION_KEY_TEST = BOT / "position_key_cache_test.cpp"
ORDINARY_GATE_SOURCE = BOT / "comparison_gate.cpp"
HELDOUT_GATE_WRAPPER = BOT / "comparison_gate_heldout.cpp"
ORDINARY_GATE_BASELINE_SHA256 = (
    "3d50c0f1e4b6a96d95f24774ce1fc664c2d27b9dd3d93601d6c8547a111230d8"
)
ORDINARY_GATE_BASELINE_BYTES = 40_095
DEVELOPMENT_CONTRACT_BANK = (
    ROOT / "results/rank_4_jacek_hybrid/openings/development_d04.tsv"
)
DEVELOPMENT_CONTRACT_BANK_SHA256 = (
    "984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc"
)
DEVELOPMENT_CONTRACT_BANK_SEED = "18128950407139886133"
DEVELOPMENT_CONTRACT_BANK_GAMES = 78
DEVELOPMENT_CONTRACT_BANK_PAIRS = 39

REQUIRED_CHECKS = (
    "generated_source_current",
    "generated_source_ascii_and_at_most_99999_bytes",
    "generated_source_has_no_local_includes",
    "clang_release_compile_and_focused_tests",
    "gnu_release_compile_and_focused_tests",
    "address_sanitizer_focused_tests",
    "undefined_behavior_sanitizer_focused_tests",
    "submission_tests",
    "rank4_compatibility_parity",
    "position_key_cache_regression_if_present",
    "protocol_regression",
    "cheap_replay_tactical_audit",
    "full_replay_tactical_audit",
    "source_purity_and_no_replay_corrections",
    "ordinary_comparison_gate_output_isolation",
    "comparison_gate_paired_sweep_output_contract",
)

PAIR_FIELDS = (
    "candidate_sweeps", "reference_sweeps", "split_pairs",
    "unresolved_pairs",
)
SUMMARY_FIELDS = (
    "bank", "games", "candidate_wins", "reference_wins", "unfinished",
    "failed", "candidate_p0", "candidate_p1", *PAIR_FIELDS,
    "candidate_invocations", "candidate_searches", "candidate_illegal",
    "candidate_operational", "candidate_exceptions",
    "candidate_hard_timeouts", "candidate_soft_overruns", "candidate_nodes",
    "candidate_nodes_avg", "candidate_nodes_p99", "candidate_nodes_max",
    "candidate_depth_avg", "candidate_depth_max",
    "candidate_attempted_depth_avg", "candidate_attempted_depth_max",
    "candidate_exhaustions", "candidate_first_ms_p99",
    "candidate_first_ms_max", "candidate_later_ms_p99",
    "candidate_later_ms_max", "reference_invocations",
    "reference_searches", "reference_illegal", "reference_operational",
    "reference_exceptions", "reference_hard_timeouts",
    "reference_soft_overruns", "reference_nodes", "reference_nodes_avg",
    "reference_nodes_p99", "reference_nodes_max", "reference_depth_avg",
    "reference_depth_max", "reference_attempted_depth_avg",
    "reference_attempted_depth_max", "reference_exhaustions",
    "reference_first_ms_p99", "reference_first_ms_max",
    "reference_later_ms_p99", "reference_later_ms_max",
    "candidate_proof_rebound", "candidate_proof_root",
    "candidate_proof_leaf", "candidate_proof_ply1", "candidate_proof_ply2",
    "reference_proof_rebound", "reference_proof_root",
    "reference_proof_leaf", "reference_proof_ply1", "reference_proof_ply2",
)
CONFIGURATION_FIELDS = (
    "profile", "reference_engine", "bank_count", "expected_role",
    "bank_validation", "max_turns", "expected_depths", "expected_seeds",
    "expected_sha256", "candidate_nodes", "reference_nodes",
    "candidate_clock", "reference_clock", "operational_clock",
    "candidate_exact_proof_mask", "reference_exact_proof_mask", "openings",
    "replay_corrections", "transcripts",
)

BASE_TARGETS = (
    "papersoccer_codingame_rank_4_jacek_hybrid_submission",
    "papersoccer_codingame_rank_4_jacek_hybrid_submission_test",
    "papersoccer_codingame_rank_4_jacek_hybrid_parity_test",
    "papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit",
    ORDINARY_GATE_TARGET,
    GATE_TARGET,
)
POSITION_KEY_TARGET = (
    "papersoccer_codingame_rank_4_jacek_hybrid_position_key_cache_test"
)
VERSIONED_TOOLS = {
    "cmake": ("cmake", "--version"),
    "ctest": ("ctest", "--version"),
    "git": ("git", "--version"),
    "node": ("node", "--version"),
}
BASE_EXPECTED_RELEASE_TESTS = (
    "submission_test", "submission_current", "protocol_smoke_test",
    "parity_test", "replay_tactical_audit_cheap",
    "replay_tactical_audit_full", "comparison_gate_self_test",
    "heldout_comparison_gate_self_test",
)
BASE_EXPECTED_SANITIZER_TESTS = (
    "submission_test", "protocol_smoke_test", "parity_test",
    "replay_tactical_audit_cheap", "comparison_gate_self_test",
    "heldout_comparison_gate_self_test",
)

TRACKED_INPUTS = (
    PRODUCER,
    QUALIFICATION_RECORDER,
    PRODUCER_TEST,
    QUALIFICATION_TEST,
    PLAN,
    ROOT / "CMakeLists.txt",
    ROOT / "submissions/codingame/tools/generate_submission.mjs",
    ROOT / "submissions/codingame/tools/protocol_smoke_test.mjs",
    BOT / "submission.json",
    BOT / "sources.txt",
    ENGINE,
    SOURCE,
    SOURCE_TEST,
    BOT / "parity_test.cpp",
    BOT / "replay_tactical_audit.cpp",
    BOT / "replay_book.hpp",
    BOT / "replay_value_model.hpp",
    BOT / "teacher_residual_model.hpp",
    BOT / "generate_replay_book.mjs",
    BOT / "generate_replay_value_header.mjs",
    BOT / "generate_teacher_residual_header.mjs",
    ORDINARY_GATE_SOURCE,
    HELDOUT_GATE_WRAPPER,
    BOT / "comparison_gate_engine.hpp",
    BOT / "comparison_gate_hybrid.cpp",
    BOT / "comparison_gate_rank4.cpp",
    ROOT / "submissions/codingame/bots/rank_4/bot.cpp",
    ROOT / "submissions/codingame/bots/rank_4/submission.cpp",
    ROOT / "src/bots/mcts_internal.hpp",
    ROOT / "src/opening_bank/opening_bank.cpp",
    ROOT / "src/opening_bank/opening_bank_internal.hpp",
    ROOT / "src/core/rules.cpp",
    ROOT / "src/core/geometry.cpp",
    ROOT / "include/papersoccer/rules.hpp",
    ROOT / "include/papersoccer/geometry.hpp",
    ROOT / "include/papersoccer/types.hpp",
    DEVELOPMENT_CONTRACT_BANK,
)
OPTIONAL_TRACKED_INPUTS = (BOT / "mcts_internal.hpp", POSITION_KEY_TEST)

PROCESS_MARKERS = (
    "rank_4_jacek_hybrid", "rank4-jacek-hybrid",
    "rank4-hybrid-prototype-benchmark", "record_rank4_jacek_hybrid",
    "papersoccer_codingame_rank_4", "development_d04.tsv",
    "development_d08.tsv", "development_d12.tsv", "development_d20.tsv",
    "validation_d04.tsv", "validation_d08.tsv", "validation_d12.tsv",
    "validation_d20.tsv", "final_d04.tsv", "final_d08.tsv",
    "final_d12.tsv", "final_d20.tsv",
)

EXTERNAL_EXECUTABLE_ROOTS = tuple(Path(value) for value in (
    "/Applications", "/Library", "/System", "/bin", "/lib", "/lib64",
    "/opt", "/sbin", "/usr",
))
FIXED_PATH_DIRECTORIES = tuple(Path(value) for value in (
    "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
    "/usr/sbin", "/sbin",
))
SEALED_BANK_NAME = re.compile(
    r"(?:validation|final)_d[0-9]+\.tsv", re.IGNORECASE
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            exact_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def is_exact_int(value: Any, *, minimum: int | None = None) -> bool:
    if type(value) is not int:
        return False
    return minimum is None or value >= minimum


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _under(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def external_executable_path(path: Path) -> Path:
    """Resolve a fixed-PATH executable without ever treating repo data as code."""
    lexical = _lexical_absolute(path)
    root = _lexical_absolute(ROOT)
    if (lexical.suffix.lower() == ".tsv" or
            SEALED_BANK_NAME.fullmatch(lexical.name) or
            lexical == root or lexical.is_relative_to(root) or
            not _under(lexical, EXTERNAL_EXECUTABLE_ROOTS)):
        raise ValueError(f"external executable path is forbidden: {lexical}")
    resolved = lexical.resolve(strict=True)
    if (resolved.suffix.lower() == ".tsv" or
            SEALED_BANK_NAME.fullmatch(resolved.name) or
            resolved == root or resolved.is_relative_to(root) or
            not _under(resolved, EXTERNAL_EXECUTABLE_ROOTS)):
        raise ValueError(f"resolved external executable path is forbidden: {resolved}")
    metadata = os.lstat(resolved)
    if (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or
            not os.access(resolved, os.X_OK)):
        raise ValueError(f"external tool is not an executable regular file: {resolved}")
    return resolved


def fixed_named_executable(name: str) -> Path | None:
    if (not name or name in (".", "..") or "/" in name or "\\" in name):
        raise ValueError("fixed executable name is invalid")
    for directory in FIXED_PATH_DIRECTORIES:
        candidate = directory / name
        try:
            return external_executable_path(candidate)
        except FileNotFoundError:
            continue
    return None


def fixed_tool_path(label: str) -> Path:
    try:
        name = VERSIONED_TOOLS[label][0]
    except KeyError as error:
        raise ValueError(f"unknown fixed tool: {label}") from error
    found = fixed_named_executable(name)
    if found is None:
        raise ValueError(f"fixed preflight tool is absent: {name}")
    return found


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks a timezone")
    return parsed.astimezone(dt.timezone.utc)


def require_after_t0(value: str, label: str) -> None:
    if parse_utc(value) < parse_utc(CAMPAIGN_T0_UTC):
        raise ValueError(f"{label} predates campaign T0")


def identity_label(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        return str(resolved)


def file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"identity path is not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    mode = stat.S_IMODE(path.stat().st_mode)
    return {
        "path": identity_label(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "ascii": all(byte < 128 for byte in raw),
        "mode": format(mode, "04o"),
        "executable": os.access(path, os.X_OK),
    }


def identities(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    unique = sorted({path.resolve(strict=True) for path in paths})
    return {identity_label(path): file_identity(path) for path in unique}


def optional_regular_file_exists(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"optional fixed input is a forbidden symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"optional fixed input is not a regular file: {path}")
    return True


def tracked_inputs() -> tuple[Path, ...]:
    optional = tuple(
        path for path in OPTIONAL_TRACKED_INPUTS
        if optional_regular_file_exists(path)
    )
    return (*TRACKED_INPUTS, *optional)


def fixed_targets() -> tuple[str, ...]:
    optional = (POSITION_KEY_TARGET,) if optional_regular_file_exists(
        POSITION_KEY_TEST
    ) else ()
    return (*BASE_TARGETS, *optional)


def expected_test_suffixes(sanitizers: bool) -> tuple[str, ...]:
    base = (
        BASE_EXPECTED_SANITIZER_TESTS if sanitizers
        else BASE_EXPECTED_RELEASE_TESTS
    )
    optional = ("position_key_cache_test",) if optional_regular_file_exists(
        POSITION_KEY_TEST
    ) else ()
    return (*base, *optional)


def test_regex(sanitizers: bool) -> str:
    suffixes = expected_test_suffixes(sanitizers)
    return (
        "^papersoccer_codingame_rank_4_jacek_hybrid_(" +
        "|".join(suffixes) + ")$"
    )


def sanitized_environment() -> dict[str, str]:
    fixed_path = ":".join(str(path) for path in FIXED_PATH_DIRECTORIES)
    detect_leaks = "0" if platform.system() == "Darwin" else "1"
    return {
        "ASAN_OPTIONS": (
            f"abort_on_error=1:detect_leaks={detect_leaks}:halt_on_error=1"
        ),
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": fixed_path,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(TEMPORARY_DIRECTORY),
        "TZ": "UTC",
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def environment_record() -> dict[str, Any]:
    values = sanitized_environment()
    system = platform.system()
    leak_policy = {
        "platform": system,
        "detect_leaks": system != "Darwin",
        "reason": (
            "AppleClang-ASan-leak-detection-disabled-pre-main-runtime-limitation"
            if system == "Darwin" else
            "ASan-leak-detection-enabled"
        ),
    }
    payload = {"values": values, "asan_leak_detection": leak_policy}
    return {**payload, "sha256": sha256_bytes(canonical_json(payload))}


def host_identity() -> dict[str, Any]:
    python = file_identity(external_executable_path(Path(sys.executable)))
    uname = platform.uname()
    cpu_model = platform.processor().strip()
    sysctl_path = external_executable_path(Path("/usr/sbin/sysctl")) \
        if platform.system() == "Darwin" else None
    if sysctl_path is not None:
        result = subprocess.run(
            [str(sysctl_path), "-n", "machdep.cpu.brand_string"],
            env=sanitized_environment(), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if result.returncode == 0 and not result.stderr:
            cpu_model = result.stdout.strip()
    payload = {
        "node": uname.node,
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "python_executable": python,
    }
    return {**payload, "sha256": sha256_bytes(canonical_json(payload))}


def git_text(*arguments: str) -> str:
    return subprocess.run(
        [str(fixed_tool_path("git")), *arguments], cwd=ROOT,
        env=sanitized_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def git_blob(head: str, path: Path) -> bytes:
    relative = str(path.relative_to(ROOT))
    completed = subprocess.run(
        [str(fixed_tool_path("git")), "show", f"{head}:{relative}"], cwd=ROOT,
        env=sanitized_environment(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"tracked preflight input is absent at HEAD: {relative}")
    return completed.stdout


def require_clean_tracked_head() -> dict[str, Any]:
    tracked = git_text("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise ValueError("tracked or staged files differ from HEAD")
    head = git_text("rev-parse", "HEAD")
    author_utc = git_text("show", "-s", "--format=%aI", head)
    committer_utc = git_text("show", "-s", "--format=%cI", head)
    require_after_t0(author_utc, "candidate author time")
    require_after_t0(committer_utc, "candidate commit time")
    for path in tracked_inputs():
        if path.is_symlink() or git_blob(head, path) != path.read_bytes():
            raise ValueError(f"preflight input differs from committed HEAD: {path}")
    return {
        "head": head,
        "author_utc": author_utc,
        "committer_utc": committer_utc,
        "tracked_status": "",
    }


def parse_process_table(stdout: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if not parts:
            continue
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("malformed process table")
        pid = int(parts[0])
        if pid <= 0 or pid in seen:
            raise ValueError("invalid or duplicate process PID")
        seen.add(pid)
        result.append({"pid": pid, "ppid": int(parts[1]), "command": parts[2]})
    if not result:
        raise ValueError("empty process table")
    return result


def process_preflight_from_table(
    processes: list[dict[str, Any]], self_pid: int,
) -> dict[str, Any]:
    by_pid = {item["pid"]: item for item in processes}
    if self_pid not in by_pid:
        raise ValueError("preflight producer PID absent from process table")
    allowed = {self_pid}
    parent = by_pid[self_pid]["ppid"]
    while parent > 0 and parent in by_pid and parent not in allowed:
        allowed.add(parent)
        parent = by_pid[parent]["ppid"]
    matching = [
        item for item in processes
        if any(marker in item["command"].lower() for marker in PROCESS_MARKERS)
    ]
    conflicts = [item for item in matching if item["pid"] not in allowed]
    return {
        "clean": not conflicts,
        "self_pid": self_pid,
        "allowed_ancestor_pids": sorted(allowed),
        "observed_process_count": len(processes),
        "conflicts": conflicts,
        "markers": list(PROCESS_MARKERS),
    }


def process_table_command() -> list[str]:
    return [
        str(external_executable_path(Path("/bin/ps"))),
        "-axo", "pid=,ppid=,command=",
    ]


def require_clean_processes() -> dict[str, Any]:
    command = process_table_command()
    completed = subprocess.run(
        command, env=sanitized_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("process-table preflight failed")
    evidence = process_preflight_from_table(
        parse_process_table(completed.stdout), os.getpid()
    )
    evidence["checked_utc"] = utc_now()
    evidence["command"] = command
    if not evidence["clean"]:
        raise ValueError("competing campaign process is active")
    return evidence


def ensure_directory_durable(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"directory ancestor is not a real directory: {cursor}")
        break
    for directory in reversed(missing):
        directory.mkdir()
        descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def fixed_directory_exists(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"fixed registry is not a real directory: {path}")
    return True


def fixed_registry_files(path: Path, pattern: str) -> list[Path]:
    if not fixed_directory_exists(path):
        return []
    files: list[Path] = []
    for entry in sorted(path.iterdir()):
        if (entry.is_symlink() or not entry.is_file() or
                not re.fullmatch(pattern, entry.name)):
            raise ValueError(f"fixed registry contains an invalid entry: {entry}")
        files.append(entry)
    return files


def fsync_directory_and_parent(directory: Path) -> None:
    for durable in (directory, directory.parent):
        descriptor = os.open(durable, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def open_lock(path: Path):
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "a+b")


def preflight_claim_path(head: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise ValueError("preflight candidate commit is not a hex object ID")
    return CLAIMS / f"{head}.json"


def create_preflight_claim(
    head: str, plan_sha256: str, environment: dict[str, Any],
    host: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = preflight_claim_path(head)
    ensure_directory_durable(path.parent)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise OSError("fixed preflight-claim registry is invalid")
    claimed_utc = utc_now()
    require_after_t0(claimed_utc, "preflight claim")
    payload = {
        "schema": CLAIM_SCHEMA,
        "candidate_commit": head,
        "plan_sha256": plan_sha256,
        "producer_sha256": file_identity(PRODUCER)["sha256"],
        "environment_sha256": environment["sha256"],
        "host_sha256": host["sha256"],
        "claimed_utc": claimed_utc,
        "one_shot": True,
        "claim_precedes_first_build_or_test_command": True,
    }
    raw = canonical_json(payload)
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise ValueError("preflight claim is spent; retry forbidden") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory_and_parent(path.parent)
    if path.read_bytes() != raw:
        raise OSError("preflight claim readback failed")
    return path, payload


def validate_preflight_claim(
    embedded: dict[str, Any], head: str, plan_sha256: str,
    environment: dict[str, Any], host: dict[str, Any],
) -> None:
    expected_path = preflight_claim_path(head)
    claim = dict(embedded)
    label = claim.pop("path", "")
    if label != identity_label(expected_path):
        raise ValueError("preflight claim path mismatch")
    if expected_path.is_symlink() or not expected_path.is_file():
        raise ValueError("durable preflight claim is absent")
    raw = expected_path.read_bytes()
    expected_keys = {
        "schema", "candidate_commit", "plan_sha256", "producer_sha256",
        "environment_sha256", "host_sha256", "claimed_utc", "one_shot",
        "claim_precedes_first_build_or_test_command",
    }
    if (set(claim) != expected_keys or canonical_json(claim) != raw or
            claim.get("schema") != CLAIM_SCHEMA or
            claim.get("candidate_commit") != head or
            claim.get("plan_sha256") != plan_sha256 or
            claim.get("producer_sha256") != file_identity(PRODUCER)["sha256"] or
            claim.get("environment_sha256") != environment["sha256"] or
            claim.get("host_sha256") != host["sha256"] or
            claim.get("one_shot") is not True or
            claim.get("claim_precedes_first_build_or_test_command") is not True):
        raise ValueError("preflight claim binding mismatch")
    require_after_t0(claim["claimed_utc"], "preflight claim")


def persist_content_addressed(payload: dict[str, Any]) -> tuple[Path, str]:
    raw = canonical_json(payload)
    digest = sha256_bytes(raw)
    ensure_directory_durable(RECEIPTS)
    if RECEIPTS.is_symlink() or not RECEIPTS.is_dir():
        raise OSError("fixed preflight receipt registry is invalid")
    destination = RECEIPTS / f"{digest}.json"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file() or \
                destination.read_bytes() != raw:
            raise OSError("preflight content-address collision")
        return destination, digest
    temporary = RECEIPTS / f".{digest}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory_and_parent(RECEIPTS)
    except BaseException:
        raise
    persisted = destination.read_bytes()
    if (destination.stem != digest or persisted != raw or
            sha256_bytes(persisted) != digest or
            canonical_json(json.loads(persisted)) != persisted):
        raise OSError("preflight receipt readback failed")
    return destination, digest


def _stream_record(data: str) -> dict[str, Any]:
    raw = data.encode("utf-8")
    return {
        "retained": False,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "empty": raw == b"",
    }


def run_fixed_command(
    argv: list[str], timeout_seconds: int,
    required_stdout_markers: Iterable[str] = (),
) -> tuple[dict[str, Any], str]:
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    timed_out = False
    os_error: str | None = None
    try:
        completed = subprocess.run(
            argv, cwd=ROOT, env=sanitized_environment(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=timeout_seconds, start_new_session=True,
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
        returncode, stdout, stderr = None, "", ""
        os_error = type(error).__name__
    ended_ns = time.monotonic_ns()
    ended_utc = utc_now()
    markers = list(required_stdout_markers)
    marker_status = {marker: marker in stdout for marker in markers}
    passed = (
        returncode == 0 and not timed_out and os_error is None and
        stderr == "" and all(marker_status.values())
    )
    record = {
        "argv": argv,
        "cwd": str(ROOT),
        "environment_sha256": environment_record()["sha256"],
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "elapsed_monotonic_ns": ended_ns - started_ns,
        "timeout_seconds": timeout_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "os_error_class": os_error,
        "stdout": _stream_record(stdout),
        "stderr": _stream_record(stderr),
        "required_stdout_markers": marker_status,
        "passed": passed,
    }
    return record, stdout


def tool_identities() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for label, (name, version_flag) in VERSIONED_TOOLS.items():
        path = fixed_tool_path(label)
        identity = file_identity(path)
        command, stdout = run_fixed_command(
            [str(path), version_flag], 30
        )
        if not command["passed"] or not stdout.strip():
            raise ValueError(f"fixed preflight tool version failed: {name}")
        records[label] = {
            "executable": identity,
            "version_argv": [str(path), version_flag],
            "version_first_line": stdout.splitlines()[0],
            "version_stdout_sha256": command["stdout"]["sha256"],
            "version_stdout_bytes": command["stdout"]["bytes"],
            "version_stderr_empty": True,
        }
    ps_path = external_executable_path(Path("/bin/ps"))
    records["ps"] = {"executable": file_identity(ps_path)}
    records["runtime_linkage"] = {
        "executable": file_identity(runtime_linkage_tool())
    }
    return records


def runtime_linkage_tool() -> Path:
    if platform.system() == "Darwin":
        path = external_executable_path(Path("/usr/bin/otool"))
    elif platform.system() == "Linux":
        found = fixed_named_executable("ldd")
        if found is None:
            raise ValueError("fixed ldd runtime-linkage tool is absent")
        path = found
    else:
        raise ValueError("unsupported runtime-linkage platform")
    identity = file_identity(path)
    if identity["executable"] is not True:
        raise ValueError("runtime-linkage tool is not executable")
    return path


def _safe_materialized_runtime_path(raw: str) -> Path | None:
    if not raw.startswith("/"):
        return None
    lexical = Path(os.path.abspath(raw))
    if (lexical.suffix.lower() == ".tsv" or re.fullmatch(
            r"(?:validation|final)_d[0-9]+\.tsv", lexical.name.lower()
    )):
        raise ValueError("runtime linkage attempted to address a held-out bank")
    allowed = tuple(Path(value) for value in (
        "/Applications", "/Library", "/System", "/lib", "/lib64",
        "/opt", "/usr",
    ))
    if not any(lexical == root or lexical.is_relative_to(root) for root in allowed):
        raise ValueError("runtime linkage contains a non-system absolute path")
    if not lexical.exists():
        return None
    if lexical.is_symlink():
        resolved = lexical.resolve(strict=True)
        if not any(
            resolved == root or resolved.is_relative_to(root) for root in allowed
        ):
            raise ValueError("runtime linkage symlink escapes system roots")
        return resolved
    return lexical


def runtime_linkage_snapshot(binary: Path) -> dict[str, Any]:
    tool = runtime_linkage_tool()
    argv = [str(tool), "-L", str(binary)] if platform.system() == "Darwin" \
        else [str(tool), str(binary)]
    completed = subprocess.run(
        argv, cwd=ROOT, env=sanitized_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=60, start_new_session=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("runtime-linkage inspection failed")
    names: set[str] = set()
    lines = completed.stdout.splitlines()
    if platform.system() == "Darwin":
        for line in lines[1:]:
            value = line.strip().split(maxsplit=1)
            if value:
                names.add(value[0])
    else:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if " => " in stripped:
                value = stripped.split(" => ", 1)[1].split(maxsplit=1)[0]
            else:
                value = stripped.split(maxsplit=1)[0]
            names.add(value)
    materialized = {
        path for name in names
        if (path := _safe_materialized_runtime_path(name)) is not None
    }
    normalized = "\n".join(sorted(names)) + ("\n" if names else "")
    return {
        "tool": file_identity(tool),
        "argv": argv,
        "environment_sha256": environment_record()["sha256"],
        "raw_stdout_retained": False,
        "normalized_dependency_output": _stream_record(normalized),
        "stderr": _stream_record(completed.stderr),
        "returncode": completed.returncode,
        "dependency_names": sorted(names),
        "materialized_dependencies": identities(materialized),
        "passed": True,
    }


def _compiler_family_matches(family: str, version: str) -> bool:
    lowered = version.lower()
    if family == "clang":
        return "clang" in lowered
    if family == "gnu":
        return ("gcc" in lowered or "g++" in lowered) and "clang" not in lowered
    raise ValueError("unknown compiler family")


def compiler_candidates(family: str) -> tuple[str, ...]:
    if family == "clang":
        return ("clang++",)
    if family == "gnu":
        return ("g++-15", "g++-14", "g++-13", "g++-12", "g++")
    raise ValueError("unknown compiler family")


def discover_compiler(family: str) -> tuple[Path, dict[str, Any]]:
    for name in compiler_candidates(family):
        found = fixed_named_executable(name)
        if found is None:
            continue
        path = found
        identity = file_identity(path)
        if not identity["executable"]:
            continue
        record, stdout = run_fixed_command([str(path), "--version"], 30)
        if record["passed"] and _compiler_family_matches(family, stdout):
            first_line = stdout.splitlines()[0] if stdout.splitlines() else ""
            return path, {
                "family": "Clang" if family == "clang" else "GNU",
                "executable": identity,
                "version_argv": [str(path), "--version"],
                "version_first_line": first_line,
                "version_stdout_sha256": record["stdout"]["sha256"],
                "version_stdout_bytes": record["stdout"]["bytes"],
                "version_stderr_empty": True,
            }
    raise ValueError(f"no executable {family} compiler passed family validation")


def configure_command(
    build: Path, compiler: Path, sanitizers: bool,
) -> list[str]:
    return [
        str(fixed_tool_path("cmake")), "-S", str(ROOT), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DPAPERSOCCER_ENABLE_SANITIZERS={'ON' if sanitizers else 'OFF'}",
    ]


def build_command(build: Path) -> list[str]:
    return [
        str(fixed_tool_path("cmake")), "--build", str(build),
        "--parallel", "2", "--target",
        *fixed_targets(),
    ]


def ctest_command(build: Path, sanitizers: bool) -> list[str]:
    return [
        str(fixed_tool_path("ctest")), "--test-dir", str(build),
        "--output-on-failure",
        "--timeout", "300", "-j", "1", "-R", test_regex(sanitizers),
    ]


def expected_test_markers(sanitizers: bool) -> tuple[str, ...]:
    suffixes = expected_test_suffixes(sanitizers)
    return tuple(
        f"papersoccer_codingame_rank_4_jacek_hybrid_{suffix}"
        for suffix in suffixes
    ) + ("100% tests passed, 0 tests failed",)


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if token.count("=") != 1:
            raise ValueError("malformed comparison-gate token")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError("duplicate or empty comparison-gate token")
        fields[key] = value
    return fields


def exact_int(fields: dict[str, str], key: str) -> int:
    value = fields.get(key, "")
    if not value.isdigit():
        raise ValueError(f"missing or invalid integer field: {key}")
    return int(value)


def validate_gate_contract(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 3 or stdout != "\n".join(lines) + "\n":
        raise ValueError("gate contract output is not exactly three lines")
    if (not lines[0].startswith("bank_summary ") or
            not lines[1].startswith("summary ") or
            not lines[2].startswith("configuration ")):
        raise ValueError("gate contract line order mismatch")
    bank = parse_fields(lines[0])
    aggregate = parse_fields(lines[1])
    configuration = parse_fields(lines[2])
    if set(bank) != set(SUMMARY_FIELDS) or set(aggregate) != set(SUMMARY_FIELDS):
        raise ValueError("gate contract summary field set mismatch")
    if set(configuration) != set(CONFIGURATION_FIELDS):
        raise ValueError("gate contract configuration field set mismatch")
    expected_configuration = {
        "profile": "nodes", "reference_engine": "rank4", "bank_count": "1",
        "expected_role": "development",
        "bank_validation": (
            "schema,header,role,depth,seed,replay,state-sha256,"
            "canonical-sha256,disjoint"
        ),
        "max_turns": "320", "expected_depths": "4",
        "expected_seeds": DEVELOPMENT_CONTRACT_BANK_SEED,
        "expected_sha256": DEVELOPMENT_CONTRACT_BANK_SHA256,
        "candidate_nodes": "500", "reference_nodes": "500",
        "candidate_clock": "800/165", "reference_clock": "800/165",
        "operational_clock": "1000/200",
        "candidate_exact_proof_mask": "7",
        "reference_exact_proof_mask": "0",
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled", "transcripts": "not-retained",
    }
    if configuration != expected_configuration:
        raise ValueError("gate contract configuration mismatch")
    if bank.get("bank") != "0" or aggregate.get("bank") != "all":
        raise ValueError("gate contract bank label mismatch")
    bank_without_label = {key: value for key, value in bank.items() if key != "bank"}
    aggregate_without_label = {
        key: value for key, value in aggregate.items() if key != "bank"
    }
    if bank_without_label != aggregate_without_label:
        raise ValueError("gate contract bank and aggregate differ")
    if exact_int(aggregate, "games") != DEVELOPMENT_CONTRACT_BANK_GAMES:
        raise ValueError("gate contract game count mismatch")
    candidate = exact_int(aggregate, "candidate_sweeps")
    reference = exact_int(aggregate, "reference_sweeps")
    split = exact_int(aggregate, "split_pairs")
    unresolved = exact_int(aggregate, "unresolved_pairs")
    if candidate + reference + split + unresolved != DEVELOPMENT_CONTRACT_BANK_PAIRS:
        raise ValueError("gate contract paired-opening accounting mismatch")
    if exact_int(aggregate, "candidate_wins") != 2 * candidate + split:
        raise ValueError("gate contract candidate sweep accounting mismatch")
    if exact_int(aggregate, "reference_wins") != 2 * reference + split:
        raise ValueError("gate contract reference sweep accounting mismatch")
    if unresolved != 0 or exact_int(aggregate, "unfinished") != 0 or \
            exact_int(aggregate, "failed") != 0:
        raise ValueError("gate contract did not resolve every game/pair")
    return {
        "stdout": _stream_record(stdout),
        "bank_field_names": sorted(bank),
        "aggregate_field_names": sorted(aggregate),
        "configuration": configuration,
        "paired_sweep_fields": list(PAIR_FIELDS),
        "aggregate_wins_used_to_infer_sweeps": False,
        "opening_pairs": DEVELOPMENT_CONTRACT_BANK_PAIRS,
        "unresolved_pairs": unresolved,
        "passed": True,
    }


def gate_contract_command() -> list[str]:
    return [
        str(FINAL_GATE), "--profile", "nodes", "--reference-engine", "rank4",
        "--bank", str(DEVELOPMENT_CONTRACT_BANK),
        "--expected-role", "development", "--expected-depths", "4",
        "--expected-seeds", DEVELOPMENT_CONTRACT_BANK_SEED,
        "--expected-sha256", DEVELOPMENT_CONTRACT_BANK_SHA256,
        "--max-turns", "320", "--candidate-nodes", "500",
        "--reference-nodes", "500", "--candidate-first-ms", "800",
        "--candidate-later-ms", "165", "--reference-first-ms", "800",
        "--reference-later-ms", "165", "--operational-first-ms", "1000",
        "--operational-later-ms", "200", "--candidate-exact-proof-mask", "7",
        "--reference-exact-proof-mask", "0",
    ]


def ordinary_gate_projection() -> bytes:
    marker = "#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)"
    output: list[str] = []
    inside = False
    keep_ordinary_branch = False
    for line in ORDINARY_GATE_SOURCE.read_text(encoding="ascii").splitlines(
            keepends=True):
        stripped = line.strip()
        if stripped == marker:
            if inside:
                raise ValueError("nested held-out gate macro block")
            inside = True
            keep_ordinary_branch = False
        elif inside and stripped == "#else":
            keep_ordinary_branch = True
        elif inside and stripped == "#endif":
            inside = False
            keep_ordinary_branch = False
        elif not inside or keep_ordinary_branch:
            output.append(line)
    if inside:
        raise ValueError("unterminated held-out gate macro block")
    return "".join(output).encode("ascii")


def heldout_gate_isolation_checks() -> dict[str, Any]:
    projection = ordinary_gate_projection()
    wrapper = HELDOUT_GATE_WRAPPER.read_bytes()
    expected_wrapper = (
        b"#define PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING 1\n"
        b'#include "comparison_gate.cpp"\n'
    )
    if (len(projection) != ORDINARY_GATE_BASELINE_BYTES or
            sha256_bytes(projection) != ORDINARY_GATE_BASELINE_SHA256):
        raise ValueError("ordinary comparison-gate projection changed")
    if wrapper != expected_wrapper:
        raise ValueError("dedicated held-out gate wrapper changed")
    return {
        "ordinary_projection_bytes": len(projection),
        "ordinary_projection_sha256": sha256_bytes(projection),
        "ordinary_baseline_bytes": ORDINARY_GATE_BASELINE_BYTES,
        "ordinary_baseline_sha256": ORDINARY_GATE_BASELINE_SHA256,
        "heldout_wrapper": file_identity(HELDOUT_GATE_WRAPPER),
        "ordinary_output_path_isolated": True,
        "passed": True,
    }


def source_checks() -> dict[str, Any]:
    source = file_identity(SOURCE)
    raw = SOURCE.read_bytes()
    text = raw.decode("ascii")
    local_includes = re.findall(r'^\s*#\s*include\s*"[^"]+"', text, re.MULTILINE)
    if source["bytes"] > 99_999 or source["ascii"] is not True:
        raise ValueError("generated source violates ASCII/size limit")
    if local_includes:
        raise ValueError("generated source contains local includes")
    runner_text = QUALIFICATION_RECORDER.read_text(encoding="ascii")
    forbidden_runner_fragments = (
        "--retain-transcripts", '"replay_corrections": "enabled"',
        '"retained": True',
    )
    required_runner_fragments = (
        '"replay_corrections": "disabled"',
        '"transcripts": "not-retained"',
        '"policy": "digest-only-no-raw-stream"',
    )
    if (any(fragment in runner_text for fragment in forbidden_runner_fragments) or
            any(fragment not in runner_text for fragment in required_runner_fragments)):
        raise ValueError("qualification runner violates purity/stream policy")
    return {
        "generated_source": {**source, "source_limit": 99_999},
        "local_include_count": len(local_includes),
        "heldout_gate_isolation": heldout_gate_isolation_checks(),
        "source_purity": {
            "qualification_recorder": file_identity(QUALIFICATION_RECORDER),
            "replay_corrections_in_qualification_runner": False,
            "transcripts_retained_by_qualification_runner": False,
            "raw_process_streams_persisted_by_qualification_runner": False,
            "passed": True,
        },
    }


def prepare_build_directory(path: Path) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_relative_to(BUILD_ROOT):
            raise ValueError("refusing to replace non-fixed preflight build path")
        if not path.is_dir():
            raise ValueError("fixed preflight build path is not a directory")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build_artifact_paths(build: Path) -> tuple[Path, ...]:
    paths: set[Path] = {
        build / "CMakeCache.txt",
        build / "compile_commands.json",
        build / "CTestTestfile.cmake",
        build / "libpapersoccer_core.a",
        build / "libpapersoccer_opening_bank_support.a",
        *(build / target for target in fixed_targets()),
    }
    for target in fixed_targets():
        directory = build / f"CMakeFiles/{target}.dir"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"fixed target build directory is absent: {target}")
        paths.update(directory.rglob("*.o"))
        paths.update(directory.rglob("*.o.d"))
        for name in ("flags.make", "link.txt"):
            candidate = directory / name
            if not candidate.is_symlink() and candidate.is_file():
                paths.add(candidate)
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"fixed build artifact is absent: {path}")
    return tuple(sorted(paths))


def run_build_panel(
    name: str, build: Path, compiler: Path, compiler_family: str,
    sanitizers: bool,
) -> dict[str, Any]:
    prepare_build_directory(build)
    configure, _ = run_fixed_command(
        configure_command(build, compiler, sanitizers), 600
    )
    if not configure["passed"]:
        raise ValueError(f"{name} configure failed")
    compile_result, _ = run_fixed_command(build_command(build), 3_600)
    if not compile_result["passed"]:
        raise ValueError(f"{name} compile failed")
    tests, _ = run_fixed_command(
        ctest_command(build, sanitizers), 2_400,
        expected_test_markers(sanitizers),
    )
    if not tests["passed"]:
        raise ValueError(f"{name} focused tests failed")
    cache_path = build / "CMakeCache.txt"
    metadata_candidates = [
        entry / "CMakeCXXCompiler.cmake"
        for entry in (build / "CMakeFiles").iterdir()
        if (not entry.is_symlink() and entry.is_dir() and
            re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", entry.name))
    ]
    if len(metadata_candidates) != 1:
        raise ValueError(f"{name} compiler metadata closure mismatch")
    compiler_metadata_path = metadata_candidates[0]
    flags_path = build / f"CMakeFiles/{GATE_TARGET}.dir/flags.make"
    link_path = build / f"CMakeFiles/{GATE_TARGET}.dir/link.txt"
    cache_text = cache_path.read_text(encoding="utf-8")
    compiler_text = compiler_metadata_path.read_text(encoding="utf-8")
    flags_text = flags_path.read_text(encoding="utf-8")
    link_text = link_path.read_text(encoding="utf-8")
    expected_sanitizer = "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=ON" if sanitizers \
        else "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF"
    if expected_sanitizer not in cache_text:
        raise ValueError(f"{name} sanitizer cache flag mismatch")
    expected_family = f'set(CMAKE_CXX_COMPILER_ID "{compiler_family}")'
    if expected_family not in compiler_text or str(compiler) not in compiler_text:
        raise ValueError(f"{name} compiler metadata mismatch")
    address = "-fsanitize=address,undefined" in flags_text and \
        "-fsanitize=address,undefined" in link_text
    undefined_behavior = address and "-fno-sanitize-recover=all" in flags_text
    if sanitizers != address or sanitizers != undefined_behavior:
        raise ValueError(f"{name} sanitizer instrumentation mismatch")
    return {
        "name": name,
        "sanitizers_enabled": sanitizers,
        "configure": configure,
        "compile": compile_result,
        "tests": tests,
        "cmake_cache": file_identity(cache_path),
        "compiler_metadata": file_identity(compiler_metadata_path),
        "instrumentation": {
            "address": address,
            "undefined_behavior": undefined_behavior,
            "flags": file_identity(flags_path),
            "link": file_identity(link_path),
        },
        "build_artifacts": identities(build_artifact_paths(build)),
        "passed": True,
    }


def validate_command_record(
    record: dict[str, Any], argv: list[str], timeout: int,
    required_markers: Iterable[str] = (),
) -> None:
    expected_keys = {
        "argv", "cwd", "environment_sha256", "started_utc", "ended_utc",
        "elapsed_monotonic_ns", "timeout_seconds", "returncode", "timed_out",
        "os_error_class", "stdout", "stderr", "required_stdout_markers",
        "passed",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("preflight command record schema mismatch")
    markers = list(required_markers)
    if (not exact_json_equal(record["argv"], argv) or
            record["cwd"] != str(ROOT) or
            record["environment_sha256"] != environment_record()["sha256"] or
            not is_exact_int(record["timeout_seconds"], minimum=1) or
            record["timeout_seconds"] != timeout or
            not is_exact_int(record["returncode"], minimum=0) or
            record["returncode"] != 0 or
            record["timed_out"] is not False or
            record["os_error_class"] is not None or
            record["passed"] is not True or
            not exact_json_equal(
                record["required_stdout_markers"],
                {name: True for name in markers},
            )):
        raise ValueError("preflight command did not pass the fixed invocation")
    require_after_t0(record["started_utc"], "preflight command start")
    require_after_t0(record["ended_utc"], "preflight command end")
    if (parse_utc(record["started_utc"]) > parse_utc(record["ended_utc"]) or
            not is_exact_int(record["elapsed_monotonic_ns"], minimum=1) or
            record["elapsed_monotonic_ns"] <= 0 or
            record["elapsed_monotonic_ns"] >=
            (timeout + 60) * 1_000_000_000):
        raise ValueError("preflight command timing record mismatch")
    wall_seconds = (
        parse_utc(record["ended_utc"]) - parse_utc(record["started_utc"])
    ).total_seconds()
    monotonic_seconds = record["elapsed_monotonic_ns"] / 1_000_000_000
    if wall_seconds < 0 or abs(wall_seconds - monotonic_seconds) > 30:
        raise ValueError("preflight command wall/monotonic timing mismatch")
    for stream_name in ("stdout", "stderr"):
        stream = record[stream_name]
        if (not isinstance(stream, dict) or set(stream) != {
                "retained", "bytes", "sha256", "empty"
        } or stream["retained"] is not False or
                not is_exact_int(stream["bytes"], minimum=0) or
                not isinstance(stream["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", stream["sha256"]) or
                type(stream.get("empty")) is not bool or
                stream.get("empty") is not (stream["bytes"] == 0)):
            raise ValueError("preflight stream evidence mismatch")
        if stream["empty"] and stream["sha256"] != sha256_bytes(b""):
            raise ValueError("empty preflight stream digest mismatch")
    if record["stderr"]["empty"] is not True or record["stderr"]["bytes"] != 0:
        raise ValueError("preflight command stderr was nonempty")


def validate_passed_receipt(
    receipt: dict[str, Any], digest: str, expected_head: str,
    expected_plan_sha256: str,
) -> None:
    plan_identity = file_identity(PLAN)
    if plan_identity["sha256"] != expected_plan_sha256:
        raise ValueError("live held-out plan hash differs from frozen identity")
    expected_keys = {
        "schema", "status", "campaign_id", "campaign_t0_utc",
        "classification", "final_qualification", "producer",
        "candidate_commit", "candidate_commit_times", "plan",
        "created_utc", "claim", "environment", "host", "git",
        "process_preflight", "tool_identities_before",
        "tool_identities_after",
        "tracked_inputs_before", "tracked_inputs_after", "source_checks",
        "generator_check", "compilers", "builds", "comparison_gate",
        "checks", "heldout_bank_files_accessed",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("preflight receipt schema fields mismatch")
    if (not re.fullmatch(r"[0-9a-f]{64}", digest) or
            sha256_bytes(canonical_json(receipt)) != digest):
        raise ValueError("preflight receipt content address mismatch")
    if (receipt["schema"] != SCHEMA or receipt["status"] != "passed" or
            receipt["campaign_id"] != CAMPAIGN_ID or
            receipt["campaign_t0_utc"] != CAMPAIGN_T0_UTC or
            receipt["classification"] !=
            "final-source-preflight-before-heldout-access" or
            receipt["final_qualification"] is not False or
            receipt["candidate_commit"] != expected_head or
            not exact_json_equal(receipt["plan"], plan_identity) or
            not exact_json_equal(receipt["environment"], environment_record()) or
            not exact_json_equal(receipt["host"], host_identity()) or
            not exact_json_equal(receipt["heldout_bank_files_accessed"], [])):
        raise ValueError("preflight receipt top-level binding mismatch")
    if not exact_json_equal(receipt["producer"], file_identity(PRODUCER)):
        raise ValueError("preflight producer identity mismatch")
    require_after_t0(receipt["created_utc"], "preflight receipt")
    validate_preflight_claim(
        receipt["claim"], expected_head, expected_plan_sha256,
        receipt["environment"], receipt["host"],
    )
    if parse_utc(receipt["claim"]["claimed_utc"]) > parse_utc(
            receipt["created_utc"]):
        raise ValueError("preflight receipt predates its claim")
    expected_commit_times = {
        "author_utc": git_text("show", "-s", "--format=%aI", expected_head),
        "committer_utc": git_text("show", "-s", "--format=%cI", expected_head),
    }
    for value in expected_commit_times.values():
        require_after_t0(value, "preflight candidate commit")
    if not exact_json_equal(receipt["candidate_commit_times"], expected_commit_times):
        raise ValueError("preflight candidate commit times mismatch")
    expected_git = {
        "head": expected_head,
        **expected_commit_times,
        "tracked_status": "",
    }
    if not exact_json_equal(
            receipt["git"], {"before": expected_git, "after": expected_git}):
        raise ValueError("preflight git provenance mismatch")
    for position in ("before", "after"):
        process = receipt["process_preflight"].get(position, {})
        ancestors = process.get("allowed_ancestor_pids", []) if isinstance(
            process, dict
        ) else []
        expected_process_keys = {
            "clean", "self_pid", "allowed_ancestor_pids",
            "observed_process_count", "conflicts", "markers", "checked_utc",
            "command",
        }
        if (not isinstance(process, dict) or set(process) != expected_process_keys or
                process.get("clean") is not True or process.get("conflicts") != [] or
                process.get("markers") != list(PROCESS_MARKERS) or
                not exact_json_equal(
                    process.get("command"), process_table_command()
                ) or not is_exact_int(process.get("self_pid"), minimum=1) or
                not isinstance(ancestors, list) or
                ancestors != sorted(set(ancestors)) or
                any(not is_exact_int(pid, minimum=1) for pid in ancestors) or
                process["self_pid"] not in ancestors or
                not is_exact_int(
                    process.get("observed_process_count"), minimum=1
                ) or
                process["observed_process_count"] < len(ancestors)):
            raise ValueError("preflight process-isolation evidence mismatch")
        require_after_t0(process["checked_utc"], "preflight process check")
    process_before = receipt["process_preflight"]["before"]
    process_after = receipt["process_preflight"]["after"]
    if (process_before["self_pid"] != process_after["self_pid"] or
            parse_utc(receipt["claim"]["claimed_utc"]) >
            parse_utc(process_before["checked_utc"]) or
            parse_utc(process_before["checked_utc"]) >
            parse_utc(process_after["checked_utc"]) or
            parse_utc(process_after["checked_utc"]) >
            parse_utc(receipt["created_utc"])):
        raise ValueError("preflight process checks are not sequential")
    if (not exact_json_equal(
            receipt["tool_identities_before"],
            receipt["tool_identities_after"],
        ) or not exact_json_equal(
            receipt["tool_identities_after"], tool_identities()
        )):
        raise ValueError("preflight tool identity changed")
    checks = receipt["checks"]
    if (not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS) or
            any(checks[name] != {"status": "passed"} for name in REQUIRED_CHECKS)):
        raise ValueError("preflight check registry mismatch")
    if not exact_json_equal(
            receipt["tracked_inputs_before"], receipt["tracked_inputs_after"]):
        raise ValueError("preflight tracked inputs changed")
    expected_tracked = identities(tracked_inputs())
    if not exact_json_equal(receipt["tracked_inputs_before"], expected_tracked):
        raise ValueError("preflight tracked input closure mismatch")
    source = receipt["source_checks"]
    if (not isinstance(source, dict) or set(source) != {
            "generated_source", "local_include_count",
            "heldout_gate_isolation", "source_purity"
            } or not exact_json_equal(source.get("generated_source"), {
                **file_identity(SOURCE), "source_limit": 99_999,
            }) or not is_exact_int(source.get("local_include_count"), minimum=0) or
            source.get("local_include_count") != 0 or
            not exact_json_equal(
                source.get("heldout_gate_isolation"),
                heldout_gate_isolation_checks(),
            ) or
            not exact_json_equal(source.get("source_purity"), {
                "qualification_recorder": file_identity(
                    QUALIFICATION_RECORDER
                ),
                "replay_corrections_in_qualification_runner": False,
                "transcripts_retained_by_qualification_runner": False,
                "raw_process_streams_persisted_by_qualification_runner": False,
                "passed": True,
            })):
        raise ValueError("preflight source checks mismatch")
    generator_argv = [
        str(fixed_tool_path("node")),
        "submissions/codingame/tools/generate_submission.mjs",
        "rank_4_jacek_hybrid", "--check",
    ]
    validate_command_record(
        receipt["generator_check"], generator_argv, 300,
        ("rank_4_jacek_hybrid submission is current",),
    )
    ordered_commands = [receipt["generator_check"]]
    discovered: dict[str, tuple[Path, dict[str, Any]]] = {
        family: discover_compiler(family) for family in ("clang", "gnu")
    }
    for family, expected_name in (("clang", "Clang"), ("gnu", "GNU")):
        compiler = receipt["compilers"].get(family, {})
        path, live = discovered[family]
        if (set(compiler) != {"family", "before", "after", "stable"} or
                compiler.get("family") != expected_name or
                not exact_json_equal(compiler.get("before"), live) or
                not exact_json_equal(compiler.get("after"), live) or
                compiler.get("stable") is not True):
            raise ValueError(f"preflight compiler identity mismatch: {family}")
        if not live["executable"]["executable"]:
            raise ValueError(f"preflight compiler is not executable: {family}")
        if str(path) != live["executable"]["path"]:
            raise ValueError(f"preflight compiler path mismatch: {family}")
    build_specs = {
        "clang_release": (CLANG_BUILD, discovered["clang"][0], "Clang", False),
        "gnu_release": (GNU_BUILD, discovered["gnu"][0], "GNU", False),
        "clang_sanitized": (
            SANITIZER_BUILD, discovered["clang"][0], "Clang", True
        ),
    }
    if set(receipt["builds"]) != set(build_specs):
        raise ValueError("preflight build panel set mismatch")
    for name, (build, compiler, compiler_family, sanitizers) in build_specs.items():
        panel = receipt["builds"][name]
        if (set(panel) != {
                "name", "sanitizers_enabled", "configure", "compile", "tests",
                "cmake_cache", "compiler_metadata", "instrumentation",
                "build_artifacts", "passed",
        } or panel["name"] != name or
                panel["sanitizers_enabled"] is not sanitizers or
                panel["passed"] is not True or
                not exact_json_equal(
                    panel["cmake_cache"], file_identity(build / "CMakeCache.txt")
                )):
            raise ValueError(f"preflight build panel metadata mismatch: {name}")
        metadata_candidates = [
            entry / "CMakeCXXCompiler.cmake"
            for entry in (build / "CMakeFiles").iterdir()
            if (not entry.is_symlink() and entry.is_dir() and
                re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", entry.name))
        ]
        if len(metadata_candidates) != 1:
            raise ValueError(f"preflight compiler metadata closure mismatch: {name}")
        if not exact_json_equal(
                panel["compiler_metadata"], file_identity(metadata_candidates[0])):
            raise ValueError(f"preflight compiler metadata identity mismatch: {name}")
        if not exact_json_equal(
                panel["build_artifacts"], identities(build_artifact_paths(build))):
            raise ValueError(f"preflight build-artifact closure mismatch: {name}")
        flags_path = build / f"CMakeFiles/{GATE_TARGET}.dir/flags.make"
        link_path = build / f"CMakeFiles/{GATE_TARGET}.dir/link.txt"
        instrumentation = panel["instrumentation"]
        if not exact_json_equal(instrumentation, {
            "address": sanitizers,
            "undefined_behavior": sanitizers,
            "flags": file_identity(flags_path),
            "link": file_identity(link_path),
        }):
            raise ValueError(f"preflight sanitizer evidence mismatch: {name}")
        metadata_text = metadata_candidates[0].read_text(encoding="utf-8")
        if f'set(CMAKE_CXX_COMPILER_ID "{compiler_family}")' not in metadata_text:
            raise ValueError(f"preflight compiler family evidence mismatch: {name}")
        validate_command_record(
            panel["configure"], configure_command(build, compiler, sanitizers), 600
        )
        validate_command_record(panel["compile"], build_command(build), 3_600)
        validate_command_record(
            panel["tests"], ctest_command(build, sanitizers), 2_400,
            expected_test_markers(sanitizers),
        )
        ordered_commands.extend((
            panel["configure"], panel["compile"], panel["tests"],
        ))
    gate = receipt["comparison_gate"]
    development_identity = file_identity(DEVELOPMENT_CONTRACT_BANK)
    if development_identity["sha256"] != DEVELOPMENT_CONTRACT_BANK_SHA256:
        raise ValueError("development contract bank hash changed")
    if (set(gate) != {
            "binary", "runtime_linkage", "development_contract_bank",
            "command", "contract",
    } or
            not exact_json_equal(gate["binary"], file_identity(FINAL_GATE)) or
            not exact_json_equal(
                gate["runtime_linkage"], runtime_linkage_snapshot(FINAL_GATE)
            ) or not exact_json_equal(
                gate["development_contract_bank"], development_identity
            )):
        raise ValueError("preflight comparison gate identity mismatch")
    validate_command_record(gate["command"], gate_contract_command(), 600)
    ordered_commands.append(gate["command"])
    contract = gate["contract"]
    expected_contract_configuration = {
        "profile": "nodes", "reference_engine": "rank4", "bank_count": "1",
        "expected_role": "development",
        "bank_validation": (
            "schema,header,role,depth,seed,replay,state-sha256,"
            "canonical-sha256,disjoint"
        ),
        "max_turns": "320", "expected_depths": "4",
        "expected_seeds": DEVELOPMENT_CONTRACT_BANK_SEED,
        "expected_sha256": DEVELOPMENT_CONTRACT_BANK_SHA256,
        "candidate_nodes": "500", "reference_nodes": "500",
        "candidate_clock": "800/165", "reference_clock": "800/165",
        "operational_clock": "1000/200",
        "candidate_exact_proof_mask": "7",
        "reference_exact_proof_mask": "0",
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled", "transcripts": "not-retained",
    }
    if (not isinstance(contract, dict) or set(contract) != {
            "stdout", "bank_field_names", "aggregate_field_names",
            "configuration", "paired_sweep_fields",
            "aggregate_wins_used_to_infer_sweeps", "opening_pairs",
            "unresolved_pairs", "passed",
    } or contract.get("passed") is not True or
            not exact_json_equal(
                contract.get("stdout"), gate["command"]["stdout"]
            ) or not exact_json_equal(
                contract.get("configuration"), expected_contract_configuration
            ) or not exact_json_equal(
                contract.get("paired_sweep_fields"), list(PAIR_FIELDS)
            ) or
            contract.get("aggregate_wins_used_to_infer_sweeps") is not False or
            not exact_json_equal(
                contract.get("bank_field_names"), sorted(SUMMARY_FIELDS)
            ) or not exact_json_equal(
                contract.get("aggregate_field_names"), sorted(SUMMARY_FIELDS)
            ) or not is_exact_int(contract.get("opening_pairs"), minimum=0) or
            contract.get("opening_pairs") != DEVELOPMENT_CONTRACT_BANK_PAIRS or
            not is_exact_int(contract.get("unresolved_pairs"), minimum=0) or
            contract.get("unresolved_pairs") != 0):
        raise ValueError("preflight paired-sweep contract mismatch")
    previous = parse_utc(receipt["claim"]["claimed_utc"])
    for command in ordered_commands:
        started = parse_utc(command["started_utc"])
        ended = parse_utc(command["ended_utc"])
        if started < previous:
            raise ValueError("preflight commands overlap or are out of order")
        previous = ended
    if previous > parse_utc(process_after["checked_utc"]):
        raise ValueError("post-preflight process check predates a command")


def existing_receipts_for_head(head: str) -> list[Path]:
    if not fixed_directory_exists(RECEIPTS):
        return []
    matches: list[Path] = []
    for path in fixed_registry_files(RECEIPTS, r"[0-9a-f]{64}\.json"):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (path.stem != sha256_bytes(raw) or canonical_json(payload) != raw or
                not isinstance(payload, dict)):
            raise ValueError("invalid content-addressed preflight receipt")
        if payload.get("candidate_commit") != head:
            raise ValueError("foreign preflight receipt violates single-finalist policy")
        matches.append(path)
    return matches


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.parse_args()
    ensure_directory_durable(BUILD_ROOT)
    ensure_directory_durable(TEMPORARY_DIRECTORY)
    ensure_directory_durable(LOCK.parent)
    with open_lock(LOCK) as private_handle, open_lock(
            BENCHMARK_LOCK) as shared_handle:
        try:
            fcntl.flock(private_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(shared_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("held-out preflight lock is busy", file=sys.stderr)
            return 2
        try:
            git = require_clean_tracked_head()
            claim_files = fixed_registry_files(
                CLAIMS, r"[0-9a-f]{40,64}\.json"
            )
            expected_claim_path = preflight_claim_path(git["head"])
            if any(path != expected_claim_path for path in claim_files):
                raise ValueError(
                    "foreign preflight claim violates single-finalist policy"
                )
            existing = existing_receipts_for_head(git["head"])
            if len(existing) > 1:
                raise ValueError("multiple preflight receipts exist for current HEAD")
            if existing:
                raw = existing[0].read_bytes()
                payload = json.loads(raw)
                validate_passed_receipt(
                    payload, existing[0].stem, git["head"], file_identity(PLAN)["sha256"]
                )
                print(existing[0].relative_to(ROOT))
                print(f"sha256={existing[0].stem}")
                return 0
            environment = environment_record()
            host_before = host_identity()
            plan_sha256 = file_identity(PLAN)["sha256"]
            claim_path = preflight_claim_path(git["head"])
            if os.path.lexists(claim_path):
                raise ValueError(
                    "preflight claim is spent without a valid receipt; retry forbidden"
                )
            claim_path, claim = create_preflight_claim(
                git["head"], plan_sha256, environment, host_before
            )
            process_before = require_clean_processes()
            tools_before = tool_identities()
            tracked_before = identities(tracked_inputs())
            source = source_checks()
            generator, _ = run_fixed_command(
                [str(fixed_tool_path("node")),
                 "submissions/codingame/tools/generate_submission.mjs",
                 "rank_4_jacek_hybrid", "--check"],
                300, ("rank_4_jacek_hybrid submission is current",),
            )
            if not generator["passed"]:
                raise ValueError("generated source current check failed")
            clang_path, clang_before = discover_compiler("clang")
            gnu_path, gnu_before = discover_compiler("gnu")
            builds = {
                "clang_release": run_build_panel(
                    "clang_release", CLANG_BUILD, clang_path, "Clang", False
                ),
                "gnu_release": run_build_panel(
                    "gnu_release", GNU_BUILD, gnu_path, "GNU", False
                ),
                "clang_sanitized": run_build_panel(
                    "clang_sanitized", SANITIZER_BUILD, clang_path, "Clang", True
                ),
            }
            gate_command, gate_stdout = run_fixed_command(
                gate_contract_command(), 600
            )
            if not gate_command["passed"]:
                raise ValueError("comparison gate contract command failed")
            gate_contract = validate_gate_contract(gate_stdout)
            clang_after = discover_compiler("clang")[1]
            gnu_after = discover_compiler("gnu")[1]
            if clang_before != clang_after or gnu_before != gnu_after:
                raise ValueError("compiler changed during preflight")
            tracked_after = identities(tracked_inputs())
            if tracked_before != tracked_after:
                raise ValueError("tracked input changed during preflight")
            if host_before != host_identity():
                raise ValueError("host identity changed during preflight")
            process_after = require_clean_processes()
            tools_after = tool_identities()
            if tools_before != tools_after:
                raise ValueError("preflight tool identity changed during execution")
            git_after = require_clean_tracked_head()
            if git_after != git:
                raise ValueError("git identity changed during preflight")
            created_utc = utc_now()
            checks = {name: {"status": "passed"} for name in REQUIRED_CHECKS}
            receipt = {
                "schema": SCHEMA,
                "status": "passed",
                "campaign_id": CAMPAIGN_ID,
                "campaign_t0_utc": CAMPAIGN_T0_UTC,
                "classification": "final-source-preflight-before-heldout-access",
                "final_qualification": False,
                "producer": file_identity(PRODUCER),
                "candidate_commit": git["head"],
                "candidate_commit_times": {
                    "author_utc": git["author_utc"],
                    "committer_utc": git["committer_utc"],
                },
                "plan": file_identity(PLAN),
                "created_utc": created_utc,
                "claim": {**claim, "path": identity_label(claim_path)},
                "environment": environment,
                "host": host_before,
                "git": {"before": git, "after": git_after},
                "process_preflight": {
                    "before": process_before, "after": process_after,
                },
                "tool_identities_before": tools_before,
                "tool_identities_after": tools_after,
                "tracked_inputs_before": tracked_before,
                "tracked_inputs_after": tracked_after,
                "source_checks": source,
                "generator_check": generator,
                "compilers": {
                    "clang": {
                        "family": "Clang", "before": clang_before,
                        "after": clang_after, "stable": True,
                    },
                    "gnu": {
                        "family": "GNU", "before": gnu_before,
                        "after": gnu_after, "stable": True,
                    },
                },
                "builds": builds,
                "comparison_gate": {
                    "binary": file_identity(FINAL_GATE),
                    "runtime_linkage": runtime_linkage_snapshot(FINAL_GATE),
                    "development_contract_bank":
                        file_identity(DEVELOPMENT_CONTRACT_BANK),
                    "command": gate_command,
                    "contract": gate_contract,
                },
                "checks": checks,
                "heldout_bank_files_accessed": [],
            }
            path, digest = persist_content_addressed(receipt)
            validate_passed_receipt(
                receipt, digest, git["head"], file_identity(PLAN)["sha256"]
            )
        except (
            KeyError, OSError, TypeError, UnicodeError, ValueError,
            json.JSONDecodeError, subprocess.SubprocessError,
        ) as error:
            print(str(error), file=sys.stderr)
            return 2
    print(path.relative_to(ROOT))
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
