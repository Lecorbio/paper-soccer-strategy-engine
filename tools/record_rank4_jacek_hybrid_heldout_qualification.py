#!/usr/bin/env python3
"""Bind and run the one-shot Rank-4/Jacek held-out qualification.

``bind`` freezes one committed finalist, its preflight receipt, compilers,
dependencies, build metadata, and comparison binary without touching any
VALIDATION or FINAL TSV. ``run`` then enforces a validation-first state machine.
It claims a stage before reading its first bank byte, never retries a spent
claim, and cannot address FINAL until the unchanged binding has a persisted,
accepted VALIDATION report.

The dedicated held-out comparison binary emits explicit paired-sweep counters
without changing the ordinary development gate. Binding remains blocked until
the fixed preflight executes and proves that exact output contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

import record_rank4_jacek_hybrid_final_source_preflight as preflight
import record_rank4_jacek_hybrid_full_development_clock as development
import record_rank4_jacek_hybrid_proof_scope_clock as common


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification"
PLAN = OUTPUT / "PLAN.json"
CAMPAIGN = ROOT / "results/rank_4_jacek_hybrid/campaign.json"
PLAN_SHA256 = "3b463b8bc4f9c34d7a9c320b07165a6563c167e14e004f3d970a9b77497408c5"
CAMPAIGN_SHA256 = "aed2a52f7a59c2b1988b5c365c23b57f8ec41fbfb50927211655a8565df63fa7"

GATE_TARGET = preflight.GATE_TARGET
GATE = preflight.FINAL_GATE
BUILD_ROOT = preflight.CLANG_BUILD
TARGET_DIRECTORY = BUILD_ROOT / f"CMakeFiles/{GATE_TARGET}.dir"
CANDIDATE_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_hybrid.cpp.o.d"
)
REFERENCE_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_rank4.cpp.o.d"
)
DRIVER_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_heldout.cpp.o.d"
)
DEPFILES = (CANDIDATE_DEPFILE, REFERENCE_DEPFILE, DRIVER_DEPFILE)
OBJECT_FILES = tuple(path.with_suffix("") for path in DEPFILES)
CORE_TARGET_DIRECTORY = BUILD_ROOT / "CMakeFiles/papersoccer_core.dir"
OPENING_TARGET_DIRECTORY = (
    BUILD_ROOT / "CMakeFiles/papersoccer_opening_bank_support.dir"
)
LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
PRIVATE_LOCK = ROOT / "build/rank4-jacek-hybrid-heldout-qualification.lock"
BIND_LOCK = ROOT / "build/rank4-jacek-hybrid-heldout-binding.lock"
STAGE_TIMEOUT_SECONDS = {"validation": 14_400, "final": 28_800}

PLAN_SCHEMA = "rank4-jacek-hybrid-heldout-qualification-plan-v3"
BINDING_SCHEMA = "rank4-jacek-hybrid-heldout-binding-v3"
PREFLIGHT_SCHEMA = preflight.SCHEMA
REPORT_SCHEMA = "rank4-jacek-hybrid-heldout-stage-report-v2"
DECISION_SCHEMA = "rank4-jacek-hybrid-heldout-decision-v2"
CLAIM_SCHEMA = "rank4-jacek-hybrid-heldout-stage-claim-v1"
BIND_CLAIM_SCHEMA = "rank4-jacek-hybrid-heldout-binding-claim-v1"
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"

PAIR_FIELDS = preflight.PAIR_FIELDS
REQUIRED_PREFLIGHT_CHECKS = preflight.REQUIRED_CHECKS
SUMMARY_FIELDS = preflight.SUMMARY_FIELDS
CONFIGURATION_FIELDS = preflight.CONFIGURATION_FIELDS

BOT = ROOT / "submissions/codingame/bots/rank_4_jacek_hybrid"
RANK4 = ROOT / "submissions/codingame/bots/rank_4"
SOURCE_PATH = BOT / "submission.cpp"
ENGINE_PATH = BOT / "bot.cpp"
TEST_PATH = BOT / "submission_test.cpp"
REFERENCE_SOURCE_PATH = RANK4 / "submission.cpp"
RECORDER_TEST = ROOT / (
    "tests/codingame/test_rank4_jacek_hybrid_heldout_qualification.py"
)
PREFLIGHT_PRODUCER = preflight.PRODUCER
PREFLIGHT_TEST = preflight.PRODUCER_TEST
PREFLIGHT_RECEIPTS = preflight.RECEIPTS

# No held-out bank path appears here. Binding hashes only these source/build
# dependencies. Stage bank paths are obtained from frozen PLAN metadata only
# after the corresponding atomic claim is durable.
TRACKED_DEPENDENCIES = (
    RECORDER,
    RECORDER_TEST,
    PREFLIGHT_PRODUCER,
    PREFLIGHT_TEST,
    PLAN,
    preflight.PREDECESSOR_CLAIM,
    preflight.PREDECESSOR_FAILURE,
    CAMPAIGN,
    ROOT / "CMakeLists.txt",
    ROOT / "tools/record_rank4_jacek_hybrid_proof_scope_clock.py",
    ROOT / "tools/record_rank4_jacek_hybrid_full_development_clock.py",
    ROOT / "submissions/codingame/tools/generate_submission.mjs",
    ROOT / "submissions/codingame/tools/protocol_smoke_test.mjs",
    BOT / "submission.json",
    BOT / "sources.txt",
    ENGINE_PATH,
    SOURCE_PATH,
    TEST_PATH,
    BOT / "parity_test.cpp",
    BOT / "replay_tactical_audit.cpp",
    BOT / "replay_book.hpp",
    BOT / "replay_value_model.hpp",
    BOT / "teacher_residual_model.hpp",
    BOT / "generate_replay_book.mjs",
    BOT / "generate_replay_value_header.mjs",
    BOT / "generate_teacher_residual_header.mjs",
    BOT / "comparison_gate.cpp",
    BOT / "comparison_gate_heldout.cpp",
    BOT / "comparison_gate_engine.hpp",
    BOT / "comparison_gate_hybrid.cpp",
    BOT / "comparison_gate_rank4.cpp",
    RANK4 / "bot.cpp",
    REFERENCE_SOURCE_PATH,
    RANK4 / "replay_book.hpp",
    RANK4 / "replay_value_model.hpp",
    RANK4 / "teacher_residual_model.hpp",
    ROOT / "src/bots/mcts_internal.hpp",
    ROOT / "src/opening_bank/opening_bank.cpp",
    ROOT / "src/opening_bank/opening_bank_internal.hpp",
    ROOT / "src/core/rules.cpp",
    ROOT / "src/core/geometry.cpp",
    ROOT / "include/papersoccer/rules.hpp",
    ROOT / "include/papersoccer/geometry.hpp",
    ROOT / "include/papersoccer/types.hpp",
)
OPTIONAL_TRACKED_DEPENDENCIES = (
    BOT / "mcts_internal.hpp", preflight.POSITION_KEY_TEST,
)
BUILD_DEPENDENCIES = (
    BUILD_ROOT / "CMakeCache.txt",
    BUILD_ROOT / "compile_commands.json",
    TARGET_DIRECTORY / "flags.make",
    TARGET_DIRECTORY / "link.txt",
    *DEPFILES,
    *OBJECT_FILES,
    BUILD_ROOT / "libpapersoccer_opening_bank_support.a",
    BUILD_ROOT / "libpapersoccer_core.a",
    GATE,
)

PROCESS_MARKERS = (
    "rank_4_jacek_hybrid",
    "rank4-jacek-hybrid",
    "rank4-hybrid-prototype-benchmark",
    "papersoccer_codingame_rank_4",
    "record_rank4_jacek_hybrid",
    "validation_d04.tsv",
    "validation_d08.tsv",
    "validation_d12.tsv",
    "validation_d20.tsv",
    "final_d04.tsv",
    "final_d08.tsv",
    "final_d12.tsv",
    "final_d20.tsv",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(dt.timezone.utc)


def require_after_t0(value: str, label: str) -> None:
    if parse_utc(value) < parse_utc(CAMPAIGN_T0_UTC):
        raise ValueError(f"{label} predates campaign T0")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_sealed_bank_path(path: Path) -> bool:
    return bool(re.fullmatch(
        r"(?:validation|final)_d[0-9]+\.tsv", path.name.lower()
    ))


def guard_read_path(
    path: Path, *, allowed_stage_banks: Iterable[Path] = (),
    allow_external: bool = False,
) -> Path:
    """Reject sealed/arbitrary paths before a stat, resolve, or content read."""
    lexical = _lexical_absolute(path)
    allowed_lexical = {_lexical_absolute(item) for item in allowed_stage_banks}
    if lexical.suffix.lower() == ".tsv" and lexical not in allowed_lexical:
        raise ValueError(f"TSV path is forbidden before its stage claim: {lexical}")
    if _is_sealed_bank_path(lexical) and lexical not in allowed_lexical:
        raise ValueError(f"sealed bank path is forbidden before claim: {lexical}")
    root = _lexical_absolute(ROOT)
    root_resolved = root.resolve(strict=True)
    in_root = lexical == root or lexical.is_relative_to(root)
    if not in_root:
        permitted_external_roots = tuple(
            Path(item) for item in (
                "/Applications", "/Library", "/System", "/lib", "/lib64",
                "/opt", "/usr",
            )
        )
        if not allow_external or not any(
            lexical == prefix or lexical.is_relative_to(prefix)
            for prefix in permitted_external_roots
        ):
            raise ValueError(f"external dependency path is not whitelisted: {lexical}")
    if path.is_symlink():
        raise ValueError(f"symlink input path is forbidden: {path}")
    resolved = lexical.resolve(strict=True)
    if resolved.suffix.lower() == ".tsv" and resolved not in allowed_lexical:
        raise ValueError(f"resolved TSV path is forbidden before claim: {resolved}")
    if _is_sealed_bank_path(resolved) and resolved not in allowed_lexical:
        raise ValueError(f"resolved sealed bank path is forbidden: {resolved}")
    resolved_in_root = (
        resolved == root_resolved or resolved.is_relative_to(root_resolved)
    )
    if in_root and not resolved_in_root:
        raise ValueError(f"repository path resolves outside the repository: {lexical}")
    if not in_root and not any(
        resolved == prefix or resolved.is_relative_to(prefix)
        for prefix in permitted_external_roots
    ):
        raise ValueError(f"external dependency resolves outside system roots: {resolved}")
    return resolved


def identity_label(path: Path) -> str:
    resolved = _lexical_absolute(path)
    try:
        return str(resolved.relative_to(_lexical_absolute(ROOT)))
    except ValueError:
        return str(resolved)


def file_identity(
    path: Path, *, allowed_stage_banks: Iterable[Path] = (),
    allow_external: bool = False,
) -> dict[str, Any]:
    resolved = guard_read_path(
        path, allowed_stage_banks=allowed_stage_banks,
        allow_external=allow_external,
    )
    if not resolved.is_file():
        raise ValueError(f"identity path is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    return {
        "path": identity_label(resolved),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "ascii": all(byte < 128 for byte in raw),
        "mode": format(stat.S_IMODE(resolved.stat().st_mode), "04o"),
        "executable": os.access(resolved, os.X_OK),
    }


def identities(
    paths: Iterable[Path], *, allowed_stage_banks: Iterable[Path] = (),
    allow_external: bool = False,
) -> dict[str, dict[str, Any]]:
    unique = sorted({_lexical_absolute(path) for path in paths})
    return {
        identity_label(path): file_identity(
            path, allowed_stage_banks=allowed_stage_banks,
            allow_external=allow_external,
        )
        for path in unique
    }


def parse_make_depfile(path: Path) -> set[Path]:
    fixed_depfile = guard_read_path(path)
    text = fixed_depfile.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise ValueError(f"malformed compiler depfile: {fixed_depfile}")
    _, raw_dependencies = text.split(":", 1)
    dependencies: set[Path] = set()
    for token in shlex.split(raw_dependencies):
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = ROOT / dependency
        dependencies.add(guard_read_path(dependency, allow_external=True))
    if not dependencies:
        raise ValueError(f"empty compiler depfile: {fixed_depfile}")
    return dependencies


def library_depfiles() -> tuple[Path, ...]:
    result: list[Path] = []
    for directory in (CORE_TARGET_DIRECTORY, OPENING_TARGET_DIRECTORY):
        fixed_directory = _lexical_absolute(directory)
        if not fixed_directory_exists(fixed_directory):
            raise ValueError(f"fixed library target directory is absent: {directory}")
        matches = sorted(fixed_directory.rglob("*.o.d"))
        if not matches:
            raise ValueError(f"fixed library target has no compiler depfiles: {directory}")
        for path in matches:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid library compiler depfile: {path}")
            result.append(path)
    return tuple(result)


def collect_binding_paths() -> tuple[list[Path], dict[str, Any]]:
    candidate = parse_make_depfile(CANDIDATE_DEPFILE)
    reference = parse_make_depfile(REFERENCE_DEPFILE)
    driver = parse_make_depfile(DRIVER_DEPFILE)
    library_files = library_depfiles()
    library_dependencies: set[Path] = set()
    for depfile in library_files:
        library_dependencies |= parse_make_depfile(depfile)
    shared_header = guard_read_path(ROOT / "src/bots/mcts_internal.hpp")
    private_path = BOT / "mcts_internal.hpp"
    private_header = guard_read_path(private_path) if \
        preflight.optional_regular_file_exists(private_path) else None
    candidate_uses_private = private_header is not None and private_header in candidate
    candidate_uses_shared = shared_header in candidate
    if private_header is not None:
        if not candidate_uses_private or candidate_uses_shared:
            raise ValueError("candidate gate does not isolate its private header")
    elif not candidate_uses_shared:
        raise ValueError("candidate gate lacks the shared fallback header")
    if (shared_header not in reference or
            (private_header is not None and private_header in reference)):
        raise ValueError("Rank-4 gate depfile header routing is not isolated")
    link_text = guard_read_path(TARGET_DIRECTORY / "link.txt").read_text(
        encoding="utf-8"
    )
    for library in (
        BUILD_ROOT / "libpapersoccer_opening_bank_support.a",
        BUILD_ROOT / "libpapersoccer_core.a",
    ):
        if str(library) not in link_text and library.name not in link_text:
            raise ValueError(f"gate link command omits frozen library: {library.name}")
    library_objects = tuple(path.with_suffix("") for path in library_files)
    paths = {
        *(_lexical_absolute(path) for path in tracked_dependencies()),
        *(_lexical_absolute(path) for path in BUILD_DEPENDENCIES),
        *library_files, *library_objects,
        *candidate, *reference, *driver, *library_dependencies,
    }
    for path in sorted(paths):
        fixed = guard_read_path(path, allow_external=True)
        if not fixed.is_file():
            raise ValueError(f"missing transitive binding dependency: {fixed}")
    routing = {
        "private_header_present": private_header is not None,
        "candidate_private_header": candidate_uses_private,
        "candidate_shared_header": candidate_uses_shared,
        "reference_shared_header": True,
        "reference_private_header": False,
        "candidate_dependency_count": len(candidate),
        "reference_dependency_count": len(reference),
        "driver_dependency_count": len(driver),
        "library_depfile_count": len(library_files),
        "library_dependency_count": len(library_dependencies),
        "complete_dependency_count": len(paths),
    }
    return sorted(paths), routing


def tracked_dependencies() -> tuple[Path, ...]:
    optional = tuple(
        path for path in OPTIONAL_TRACKED_DEPENDENCIES
        if preflight.optional_regular_file_exists(path)
    )
    return (*TRACKED_DEPENDENCIES, *optional)


def _require_fixed_json_path(path: Path, directory: Path) -> Path:
    lexical = _lexical_absolute(path)
    expected_directory = _lexical_absolute(directory)
    if (lexical.parent != expected_directory or
            not re.fullmatch(r"[0-9a-f]{64}\.json", lexical.name) or
            lexical.is_symlink()):
        raise ValueError(f"JSON path is outside its fixed registry: {path}")
    return guard_read_path(lexical)


def load_canonical_content_addressed(
    path: Path, schema: str, directory: Path,
) -> tuple[dict[str, Any], str]:
    fixed = _require_fixed_json_path(path, directory)
    raw = fixed.read_bytes()
    digest = sha256_bytes(raw)
    payload = json.loads(raw)
    if (fixed.stem != digest or not isinstance(payload, dict) or
            canonical_json(payload) != raw):
        raise ValueError(f"not canonical content-addressed JSON: {fixed}")
    if payload.get("schema") != schema:
        raise ValueError(f"schema mismatch: {fixed}")
    return payload, digest


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
        for durable in (directory.parent, directory):
            descriptor = os.open(durable, os.O_RDONLY)
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


def persist_content_addressed(directory: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    raw = canonical_json(payload)
    digest = sha256_bytes(raw)
    ensure_directory_durable(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("fixed content-addressed registry is invalid")
    destination = directory / f"{digest}.json"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file() or \
                destination.read_bytes() != raw:
            raise OSError("content-address collision or noncanonical duplicate")
        load_canonical_content_addressed(destination, payload["schema"], directory)
        return destination, digest
    temporary = directory / f".{digest}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    fsync_directory_and_parent(directory)
    persisted = destination.read_bytes()
    if (destination.stem != digest or persisted != raw or
            sha256_bytes(persisted) != digest or
            canonical_json(json.loads(persisted)) != persisted):
        raise OSError("content-addressed persistence readback failed")
    return destination, digest


def git_text(*arguments: str) -> str:
    return subprocess.run(
        [str(preflight.fixed_tool_path("git")), *arguments], cwd=ROOT,
        env=preflight.sanitized_environment(),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def git_blob(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        [str(preflight.fixed_tool_path("git")), "show", f"{commit}:{relative}"],
        cwd=ROOT,
        env=preflight.sanitized_environment(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot read committed blob: {relative}")
    return completed.stdout


def require_clean_tracked_tree() -> dict[str, str]:
    tracked = git_text("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise ValueError("tracked or staged files differ from HEAD")
    head = git_text("rev-parse", "HEAD")
    author_utc = git_text("show", "-s", "--format=%aI", head)
    committer_utc = git_text("show", "-s", "--format=%cI", head)
    require_after_t0(author_utc, "candidate author time")
    require_after_t0(committer_utc, "candidate commit time")
    return {
        "head": head,
        "author_utc": author_utc,
        "committer_utc": committer_utc,
        "tracked_status": tracked,
    }


def validate_git_state(state: dict[str, Any], expected_head: str) -> None:
    if (not isinstance(state, dict) or set(state) != {
            "head", "author_utc", "committer_utc", "tracked_status"
    } or state.get("head") != expected_head or
            state.get("tracked_status") != ""):
        raise ValueError("persisted clean tracked git state mismatch")
    require_after_t0(state["author_utc"], "candidate author time")
    require_after_t0(state["committer_utc"], "candidate commit time")


def require_tracked_head_paths(paths: Iterable[Path], head: str) -> None:
    for path in paths:
        lexical = _lexical_absolute(path)
        if not lexical.is_relative_to(_lexical_absolute(ROOT)):
            continue
        if lexical.is_relative_to(_lexical_absolute(ROOT / "build")):
            continue
        guard_read_path(lexical)
        relative = str(lexical.relative_to(_lexical_absolute(ROOT)))
        listed = subprocess.run(
            [str(preflight.fixed_tool_path("git")), "ls-files",
             "--error-unmatch", "--", relative],
            cwd=ROOT, env=preflight.sanitized_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        if listed.returncode != 0:
            raise ValueError(f"binding dependency is not tracked: {relative}")
        if git_blob(head, relative) != lexical.read_bytes():
            raise ValueError(f"live dependency differs from HEAD: {relative}")


def parse_process_table(stdout: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_line in stdout.splitlines():
        parts = raw_line.strip().split(maxsplit=2)
        if not parts:
            continue
        if (len(parts) != 3 or not parts[0].isdigit() or
                not parts[1].isdigit()):
            raise ValueError("malformed ps row")
        pid = int(parts[0])
        if pid <= 0 or pid in seen:
            raise ValueError("invalid or duplicate ps PID")
        seen.add(pid)
        result.append({"pid": pid, "ppid": int(parts[1]), "command": parts[2]})
    if not result:
        raise ValueError("empty ps process table")
    return result


def process_preflight_from_table(processes: list[dict[str, Any]], self_pid: int) -> dict[str, Any]:
    by_pid = {item["pid"]: item for item in processes}
    if self_pid not in by_pid:
        raise ValueError("recorder PID absent from process table")
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


def require_clean_processes() -> dict[str, Any]:
    command = preflight.process_table_command()
    completed = subprocess.run(
        command, env=preflight.sanitized_environment(), text=True,
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


def _read_exact_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    fixed = guard_read_path(path)
    raw = fixed.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen JSON hash mismatch: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"frozen JSON root is not an object: {path}")
    return payload


def validate_plan() -> dict[str, Any]:
    plan = _read_exact_json(PLAN, PLAN_SHA256)
    campaign = _read_exact_json(CAMPAIGN, CAMPAIGN_SHA256)
    if (plan.get("schema") != PLAN_SCHEMA or
            plan.get("status") != "preregistered-unbound" or
            plan.get("classification") !=
            "single-finalist-untouched-validation-then-final-qualification"):
        raise ValueError("held-out plan identity/classification mismatch")
    if not preflight.exact_json_equal(plan, preflight.validate_successor_plan()):
        raise ValueError("held-out technical-successor amendment mismatch")
    if (plan.get("campaign", {}).get("campaign_id") != CAMPAIGN_ID or
            plan.get("campaign", {}).get("campaign_manifest_sha256") !=
            CAMPAIGN_SHA256 or
            plan.get("campaign", {}).get("t0_utc") != CAMPAIGN_T0_UTC or
            campaign.get("campaign_id") != CAMPAIGN_ID or
            campaign.get("time_boundary", {}).get("goal_created_at_utc") !=
            CAMPAIGN_T0_UTC):
        raise ValueError("campaign binding mismatch")
    require_after_t0(
        campaign["time_boundary"]["preregistered_at_utc"],
        "campaign preregistration",
    )
    configuration = plan.get("configuration", {})
    if not preflight.exact_json_equal(configuration, {
        "profile": "clock", "reference_engine": "rank4",
        "maximum_turns": 320, "candidate_nodes": 3_000_000,
        "reference_nodes": 3_000_000,
        "candidate_clock_ms": [800, 165],
        "reference_clock_ms": [800, 165],
        "operational_hard_clock_ms": [1000, 200],
        "candidate_exact_proof_mask": 7, "reference_exact_proof_mask": 0,
        "opening_rules": "preregistered-public-codingame-rules",
        "replay_corrections": "disabled", "transcripts": "not-retained",
    }):
        raise ValueError("held-out configuration mismatch")
    stages = plan.get("stage_order", [])
    if (not isinstance(stages, list) or [item.get("stage") for item in stages]
            != ["validation", "final", "pooled"]):
        raise ValueError("held-out stage order mismatch")
    expected_stage = {
        "validation": (106, 53, 54, [26, 26]),
        "final": (212, 106, 108, [53, 53]),
        "pooled": (318, 159, 174, [83, 83]),
    }
    for item in stages:
        games, pairs, wins, colors = expected_stage[item["stage"]]
        if (not preflight.exact_json_equal(item.get("games"), games) or
                not preflight.exact_json_equal(
                    item.get("opening_pairs"), pairs
                ) or not preflight.exact_json_equal(
                    item.get("candidate_wins_min"), wins
                ) or not preflight.exact_json_equal(
                    item.get("candidate_wins_by_physical_color_min"), colors
                )):
            raise ValueError(f"held-out threshold mismatch: {item['stage']}")
    sweep = plan.get("paired_sweep_test", {})
    if (sweep.get("scope") != "pooled-validation-plus-final" or
            sweep.get("unresolved_pairs_max") != 0 or
            sweep.get("equivalent_rational_max_inclusive") != "1/40" or
            sweep.get("aggregate_wins_must_not_be_used_to_infer_sweeps") is not True):
        raise ValueError("paired-sweep policy mismatch")
    if plan.get("required_final_source_preflight_checks") != list(
            REQUIRED_PREFLIGHT_CHECKS):
        raise ValueError("preflight check registry mismatch")

    assignments = {
        item["path"]: item
        for item in campaign.get("procedural_openings", {}).get("assignments", [])
    }
    banks = plan.get("banks", {})
    for stage in ("validation", "final"):
        records = banks.get(stage, [])
        if len(records) != 4:
            raise ValueError(f"wrong {stage} bank count")
        for record in records:
            assignment = assignments.get(record.get("path"))
            if assignment is None:
                raise ValueError("bank missing from campaign metadata")
            if (record.get("depth") != assignment.get("depth") or
                    record.get("seed") != assignment.get("header_seed") or
                    record.get("bytes") != assignment.get("bytes") or
                    record.get("sha256") != assignment.get("sha256") or
                    record.get("opening_pairs") != assignment.get("pairs") or
                    record.get("games") != assignment.get("color_swapped_games") or
                    record.get("logical_role") != assignment.get("role")):
                raise ValueError("bank/campaign metadata mismatch")
        if sum(item["games"] for item in records) != expected_stage[stage][0]:
            raise ValueError(f"wrong {stage} game total")
    return plan


def fixed_preflight_receipt(
    head: str, dependency_identities: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any], str]:
    if not fixed_directory_exists(PREFLIGHT_RECEIPTS):
        raise ValueError("fixed final-source preflight receipt directory is absent")
    paths = fixed_registry_files(PREFLIGHT_RECEIPTS, r"[0-9a-f]{64}\.json")
    if len(paths) != 1:
        raise ValueError("single finalist requires exactly one preflight receipt")
    matches: list[tuple[Path, dict[str, Any], str]] = []
    for path in paths:
        receipt, digest = load_canonical_content_addressed(
            path, PREFLIGHT_SCHEMA, PREFLIGHT_RECEIPTS
        )
        if receipt.get("candidate_commit") == head:
            matches.append((path, receipt, digest))
    if len(matches) != 1:
        raise ValueError("current HEAD requires exactly one fixed preflight receipt")
    path, receipt, digest = matches[0]
    preflight.validate_passed_receipt(receipt, digest, head, PLAN_SHA256)
    source = dependency_identities.get(identity_label(SOURCE_PATH))
    if (source is None or receipt["source_checks"]["generated_source"] != {
            **source, "source_limit": 99_999,
    }):
        raise ValueError("preflight generated-source binding mismatch")
    if not preflight.exact_json_equal(
            receipt["comparison_gate"]["binary"],
            dependency_identities.get(identity_label(GATE)),
    ):
        raise ValueError("preflight comparison-gate binary binding mismatch")
    if receipt["comparison_gate"]["contract"].get(
            "paired_sweep_fields") != list(PAIR_FIELDS):
        raise ValueError("preflight comparison gate lacks paired-sweep fields")
    return path, receipt, digest


def qualification_key(
    plan: dict[str, Any], head: str,
    dependency_identities: dict[str, dict[str, Any]],
    preflight_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": BINDING_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": PLAN_SHA256,
        "candidate_commit": head,
        "candidate_engine_sha256": dependency_identities[
            identity_label(ENGINE_PATH)]["sha256"],
        "candidate_source_sha256": dependency_identities[
            identity_label(SOURCE_PATH)]["sha256"],
        "complete_dependency_sha256": sha256_bytes(
            canonical_json(dependency_identities)
        ),
        "preflight_receipt_sha256": preflight_sha256,
        "environment_sha256": preflight.environment_record()["sha256"],
        "host_sha256": preflight.host_identity()["sha256"],
        "configuration": plan["configuration"],
        "bank_sha256": {
            stage: [item["sha256"] for item in plan["banks"][stage]]
            for stage in ("validation", "final")
        },
    }


def candidate_qualification_id(key: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(key))


def require_no_prior_binding_attempt() -> None:
    binding_directory = OUTPUT / "bindings"
    if fixed_directory_exists(binding_directory):
        paths = fixed_registry_files(
            binding_directory, r"[0-9a-f]{64}\.json"
        )
        for path in paths:
            load_canonical_content_addressed(
                path, BINDING_SCHEMA, binding_directory
            )
        if paths:
            raise ValueError("single-finalist campaign already has a binding")
    claim_directory = OUTPUT / "binding_claims"
    if fixed_directory_exists(claim_directory):
        paths = fixed_registry_files(
            claim_directory, r"[0-9a-f]{64}\.json"
        )
        if paths:
            raise ValueError("single-finalist binding claim is already spent")


def binding_claim_path(identifier: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("binding identifier is not a lowercase SHA-256")
    return OUTPUT / "binding_claims" / f"{identifier}.json"


def create_binding_claim(
    identifier: str, key: dict[str, Any], preflight_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = binding_claim_path(identifier)
    ensure_directory_durable(path.parent)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise OSError("fixed binding-claim registry is invalid")
    created_utc = utc_now()
    require_after_t0(created_utc, "binding claim")
    payload = {
        "schema": BIND_CLAIM_SCHEMA,
        "candidate_qualification_id": identifier,
        "qualification_key_sha256": sha256_bytes(canonical_json(key)),
        "preflight_receipt_sha256": preflight_sha256,
        "claimed_utc": created_utc,
        "one_binding_for_exact_identity": True,
    }
    raw = canonical_json(payload)
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise ValueError("binding claim is already spent") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory_and_parent(path.parent)
    if path.read_bytes() != raw:
        raise OSError("binding claim readback failed")
    return path, payload


def validate_binding_claim(
    path: Path, identifier: str, key: dict[str, Any],
    preflight_sha256: str,
) -> dict[str, Any]:
    expected = binding_claim_path(identifier)
    if _lexical_absolute(path) != _lexical_absolute(expected) or path.is_symlink():
        raise ValueError("binding claim path mismatch")
    raw = guard_read_path(path).read_bytes()
    payload = json.loads(raw)
    expected_payload_keys = {
        "schema", "candidate_qualification_id", "qualification_key_sha256",
        "preflight_receipt_sha256", "claimed_utc",
        "one_binding_for_exact_identity",
    }
    if (not isinstance(payload, dict) or set(payload) != expected_payload_keys or
            canonical_json(payload) != raw or
            payload.get("schema") != BIND_CLAIM_SCHEMA or
            payload.get("candidate_qualification_id") != identifier or
            payload.get("qualification_key_sha256") !=
            sha256_bytes(canonical_json(key)) or
            payload.get("preflight_receipt_sha256") != preflight_sha256 or
            payload.get("one_binding_for_exact_identity") is not True):
        raise ValueError("binding claim binding mismatch")
    require_after_t0(payload["claimed_utc"], "binding claim")
    return payload


def _create_binding_locked() -> tuple[Path, str]:
    plan = validate_plan()  # metadata only; does not touch a bank path
    git = require_clean_tracked_tree()
    head = git["head"]
    process = require_clean_processes()
    all_dependencies, dependency_routing = collect_binding_paths()
    require_tracked_head_paths(all_dependencies, head)
    before = identities(all_dependencies, allow_external=True)
    source = before[identity_label(SOURCE_PATH)]
    if source["ascii"] is not True or source["bytes"] > 99_999:
        raise ValueError("candidate generated source violates contest limit")
    receipt_path, receipt, receipt_sha256 = fixed_preflight_receipt(head, before)
    compiler_records_before = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    for name in ("clang", "gnu"):
        recorded = receipt["compilers"][name]
        if (not preflight.exact_json_equal(
                compiler_records_before[name], recorded["before"]
            ) or not preflight.exact_json_equal(
                recorded["before"], recorded["after"]
            ) or
                recorded.get("stable") is not True):
            raise ValueError(f"compiler changed since preflight: {name}")
    compiler_identities = {
        name: receipt["compilers"][name]["before"]["executable"]
        for name in ("clang", "gnu")
    }
    tool_identities = {
        name: record["executable"]
        for name, record in receipt["tool_identities_after"].items()
    }
    runtime_identities = tuple(
        receipt["comparison_gate"]["runtime_linkage"]
        ["materialized_dependencies"].values()
    )
    bound_identities = dict(before)
    for identity in (
        *compiler_identities.values(), *tool_identities.values(),
        *runtime_identities,
    ):
        bound_identities[identity["path"]] = identity
    after = identities(all_dependencies, allow_external=True)
    after_bound = dict(after)
    compiler_records_after = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    if not preflight.exact_json_equal(
            compiler_records_before, compiler_records_after):
        raise ValueError("compiler changed while binding was frozen")
    for name, identity in compiler_identities.items():
        live_identity = compiler_records_after[name]["executable"]
        after_bound[live_identity["path"]] = live_identity
    for identity in tool_identities.values():
        after_bound[identity["path"]] = file_identity(
            Path(identity["path"]), allow_external=True
        )
    for identity in runtime_identities:
        after_bound[identity["path"]] = file_identity(
            Path(identity["path"]), allow_external=True
        )
    if not preflight.exact_json_equal(bound_identities, after_bound):
        raise ValueError("binding dependencies changed while being frozen")
    dependency_routing["complete_dependency_count_with_compilers"] = len(
        bound_identities
    )
    key = qualification_key(plan, head, bound_identities, receipt_sha256)
    identifier = candidate_qualification_id(key)
    require_no_prior_binding_attempt()
    claim_path_value, claim = create_binding_claim(
        identifier, key, receipt_sha256
    )
    created_utc = utc_now()
    require_after_t0(created_utc, "binding creation")
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "frozen-unopened-heldout",
        "created_utc": created_utc,
        "campaign_id": CAMPAIGN_ID,
        "campaign_t0_utc": CAMPAIGN_T0_UTC,
        "plan": {"path": identity_label(PLAN), "sha256": PLAN_SHA256},
        "campaign_manifest": {
            "path": identity_label(CAMPAIGN), "sha256": CAMPAIGN_SHA256,
        },
        "candidate_qualification_id": identifier,
        "qualification_key": key,
        "binding_claim": {
            **claim, "path": identity_label(claim_path_value),
        },
        "candidate_commit": head,
        "configuration": plan["configuration"],
        "bank_registry_from_campaign_metadata_only": plan["banks"],
        "dependency_identities": bound_identities,
        "dependency_routing": dependency_routing,
        "preflight_receipt": file_identity(receipt_path),
        "preflight_summary": {
            "status": receipt["status"],
            "checks": receipt["checks"],
            "technical_successor": receipt["technical_successor"],
            "compilers": receipt["compilers"],
            "comparison_gate": receipt["comparison_gate"],
            "builds": receipt["builds"],
        },
        "compiler_identities": compiler_identities,
        "tool_identities": tool_identities,
        "compiler_records": compiler_records_after,
        "environment": preflight.environment_record(),
        "host": preflight.host_identity(),
        "runtime": {
            "python_version": sys.version,
            "python_executable": file_identity(
                preflight.external_executable_path(Path(sys.executable)),
                allow_external=True,
            ),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "process_preflight": process,
        "git": git,
        "bank_files_accessed": [],
    }
    return persist_content_addressed(OUTPUT / "bindings", binding)


def create_binding() -> tuple[Path, str]:
    ensure_directory_durable(BIND_LOCK.parent)
    with open_lock(BIND_LOCK) as bind_handle, open_lock(LOCK) as shared_handle:
        try:
            fcntl.flock(bind_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(shared_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another binding/build/benchmark job owns the lock") from error
        return _create_binding_locked()


def stage_records(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in ("validation", "final"):
        raise ValueError("stage must be validation or final")
    records = plan["banks"][stage]
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("frozen stage bank registry is malformed")
    return records


def expected_configuration(plan: dict[str, Any], stage: str) -> dict[str, str]:
    records = stage_records(plan, stage)
    configuration = plan["configuration"]
    return {
        "profile": "clock",
        "reference_engine": "rank4",
        "bank_count": "4",
        "expected_role": records[0]["gate_role"],
        "bank_validation": (
            "schema,header,role,depth,seed,replay,state-sha256,"
            "canonical-sha256,disjoint"
        ),
        "max_turns": str(configuration["maximum_turns"]),
        "expected_depths": ",".join(str(item["depth"]) for item in records),
        "expected_seeds": ",".join(item["seed"] for item in records),
        "expected_sha256": ",".join(item["sha256"] for item in records),
        "candidate_nodes": str(configuration["candidate_nodes"]),
        "reference_nodes": str(configuration["reference_nodes"]),
        "candidate_clock": "/".join(
            str(value) for value in configuration["candidate_clock_ms"]
        ),
        "reference_clock": "/".join(
            str(value) for value in configuration["reference_clock_ms"]
        ),
        "operational_clock": "/".join(
            str(value) for value in configuration["operational_hard_clock_ms"]
        ),
        "candidate_exact_proof_mask": str(
            configuration["candidate_exact_proof_mask"]
        ),
        "reference_exact_proof_mask": str(
            configuration["reference_exact_proof_mask"]
        ),
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }


def command_for_stage(plan: dict[str, Any], stage: str) -> list[str]:
    records = stage_records(plan, stage)
    configuration = plan["configuration"]
    command = [
        str(GATE), "--profile", "clock", "--reference-engine", "rank4",
    ]
    for record in records:
        command.extend(("--bank", str(ROOT / record["path"])))
    command.extend((
        "--expected-role", records[0]["gate_role"],
        "--expected-depths", ",".join(str(item["depth"]) for item in records),
        "--expected-seeds", ",".join(item["seed"] for item in records),
        "--expected-sha256", ",".join(item["sha256"] for item in records),
        "--max-turns", str(configuration["maximum_turns"]),
        "--candidate-nodes", str(configuration["candidate_nodes"]),
        "--reference-nodes", str(configuration["reference_nodes"]),
        "--candidate-first-ms", str(configuration["candidate_clock_ms"][0]),
        "--candidate-later-ms", str(configuration["candidate_clock_ms"][1]),
        "--reference-first-ms", str(configuration["reference_clock_ms"][0]),
        "--reference-later-ms", str(configuration["reference_clock_ms"][1]),
        "--operational-first-ms",
        str(configuration["operational_hard_clock_ms"][0]),
        "--operational-later-ms",
        str(configuration["operational_hard_clock_ms"][1]),
        "--candidate-exact-proof-mask",
        str(configuration["candidate_exact_proof_mask"]),
        "--reference-exact-proof-mask",
        str(configuration["reference_exact_proof_mask"]),
    ))
    return command


def finite_nonnegative(fields: dict[str, str], key: str) -> float:
    value = common.exact_float(fields, key)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"field is not finite and nonnegative: {key}")
    return value


def validate_strict_timing(fields: dict[str, str]) -> None:
    for engine in ("candidate", "reference"):
        first_p99 = finite_nonnegative(fields, f"{engine}_first_ms_p99")
        first_max = finite_nonnegative(fields, f"{engine}_first_ms_max")
        later_p99 = finite_nonnegative(fields, f"{engine}_later_ms_p99")
        later_max = finite_nonnegative(fields, f"{engine}_later_ms_max")
        if first_p99 > first_max or later_p99 > later_max:
            raise ValueError(f"{engine} timing p99 exceeds maximum")
        if first_p99 >= 900.0 or first_max >= 990.0:
            raise ValueError(f"{engine} first timing threshold failed")
        if later_p99 >= 180.0 or later_max >= 198.0:
            raise ValueError(f"{engine} later timing threshold failed")


def sweep_counters(fields: dict[str, str]) -> dict[str, int]:
    return {name: common.exact_int(fields, name) for name in PAIR_FIELDS}


def validate_sweep_accounting(
    fields: dict[str, str], opening_pairs: int,
) -> dict[str, int]:
    counters = sweep_counters(fields)
    candidate = counters["candidate_sweeps"]
    reference = counters["reference_sweeps"]
    split = counters["split_pairs"]
    unresolved = counters["unresolved_pairs"]
    if candidate + reference + split + unresolved != opening_pairs:
        raise ValueError("paired-sweep opening accounting mismatch")
    if unresolved != 0:
        raise ValueError("unresolved paired sweep is nonzero")
    if common.exact_int(fields, "candidate_wins") != 2 * candidate + split:
        raise ValueError("candidate wins do not reconcile with explicit sweeps")
    if common.exact_int(fields, "reference_wins") != 2 * reference + split:
        raise ValueError("reference wins do not reconcile with explicit sweeps")
    return counters


def validate_stage_stdout(
    plan: dict[str, Any], stage: str, stdout: str,
) -> dict[str, Any]:
    records = stage_records(plan, stage)
    lines = [line for line in stdout.splitlines() if line.strip()]
    bank_lines = [line for line in lines if line.startswith("bank_summary ")]
    summary_lines = [line for line in lines if line.startswith("summary ")]
    configuration_lines = [
        line for line in lines if line.startswith("configuration ")
    ]
    if (len(lines) != len(records) + 2 or len(bank_lines) != len(records) or
            len(summary_lines) != 1 or len(configuration_lines) != 1):
        raise ValueError("gate stdout does not contain only frozen summary lines")
    expected_lines = [*bank_lines, summary_lines[0], configuration_lines[0]]
    if lines != expected_lines or stdout != "\n".join(expected_lines) + "\n":
        raise ValueError("gate stdout line order or final newline differs")
    banks = [common.parse_fields(line) for line in bank_lines]
    aggregate = common.parse_fields(summary_lines[0])
    configuration = common.parse_fields(configuration_lines[0])
    for fields in (*banks, aggregate):
        if set(fields) != set(SUMMARY_FIELDS):
            raise ValueError("gate summary field set is not exact")
    if set(configuration) != set(CONFIGURATION_FIELDS):
        raise ValueError("gate configuration field set is not exact")
    if configuration != expected_configuration(plan, stage):
        raise ValueError("complete configuration echo mismatch")
    candidate_mask = plan["configuration"]["candidate_exact_proof_mask"]
    reference_mask = plan["configuration"]["reference_exact_proof_mask"]
    for index, (bank, record) in enumerate(zip(banks, records)):
        common.validate_summary(
            bank, str(index), candidate_mask, reference_mask,
            expected_games=record["games"],
            expected_color_games=record["opening_pairs"],
        )
        development.validate_rebound_identity(bank)
        validate_strict_timing(bank)
        validate_sweep_accounting(bank, record["opening_pairs"])
    total_games = sum(item["games"] for item in records)
    total_pairs = sum(item["opening_pairs"] for item in records)
    common.validate_summary(
        aggregate, "all", candidate_mask, reference_mask,
        expected_games=total_games, expected_color_games=total_pairs,
    )
    development.validate_rebound_identity(aggregate)
    validate_strict_timing(aggregate)
    aggregate_sweeps = validate_sweep_accounting(aggregate, total_pairs)
    development.validate_bank_aggregate_consistency(banks, aggregate)
    for name in PAIR_FIELDS:
        if aggregate_sweeps[name] != sum(
            sweep_counters(bank)[name] for bank in banks
        ):
            raise ValueError(f"bank/aggregate sweep mismatch: {name}")
    return {
        "stage": stage,
        "bank_lines": bank_lines,
        "banks": banks,
        "aggregate_line": summary_lines[0],
        "aggregate": aggregate,
        "configuration_line": configuration_lines[0],
        "configuration": configuration,
        "sweeps": aggregate_sweeps,
    }


def _stage_threshold(stage: str) -> tuple[int, tuple[int, int]]:
    if stage == "validation":
        return 54, (26, 26)
    if stage == "final":
        return 108, (53, 53)
    raise ValueError("invalid held-out stage")


def stage_threshold_errors(stage: str, aggregate: dict[str, str]) -> list[str]:
    wins_minimum, color_minimum = _stage_threshold(stage)
    errors: list[str] = []
    if common.exact_int(aggregate, "candidate_wins") < wins_minimum:
        errors.append(f"candidate has fewer than {wins_minimum} {stage} wins")
    for color in range(2):
        wins = common.parse_color(aggregate, f"candidate_p{color}")[0]
        if wins < color_minimum[color]:
            errors.append(
                f"candidate has fewer than {color_minimum[color]} wins "
                f"as physical color {color} on {stage}"
            )
    return errors


def exact_one_sided_sign_test(candidate_sweeps: int, reference_sweeps: int) -> dict[str, Any]:
    if candidate_sweeps < 0 or reference_sweeps < 0:
        raise ValueError("negative sweep count")
    sample = candidate_sweeps + reference_sweeps
    if sample == 0:
        probability = Fraction(1, 1)
    else:
        numerator = sum(
            math.comb(sample, successes)
            for successes in range(candidate_sweeps, sample + 1)
        )
        probability = Fraction(numerator, 2 ** sample)
    return {
        "candidate_sweeps": candidate_sweeps,
        "reference_sweeps": reference_sweeps,
        "decisive_sweep_pairs": sample,
        "tail": "P[X>=candidate_sweeps|X~Binomial(C+R,0.5)]",
        "p_numerator": probability.numerator,
        "p_denominator": probability.denominator,
        "p_decimal": format(float(probability), ".18g"),
        "threshold_numerator": 1,
        "threshold_denominator": 40,
        "passed": probability <= Fraction(1, 40),
    }


def pooled_evaluation(
    validation_report: dict[str, Any], final_report: dict[str, Any],
) -> dict[str, Any]:
    aggregates = [
        validation_report["parsed"]["aggregate"],
        final_report["parsed"]["aggregate"],
    ]
    candidate_wins = sum(common.exact_int(item, "candidate_wins") for item in aggregates)
    reference_wins = sum(common.exact_int(item, "reference_wins") for item in aggregates)
    color_wins = [
        sum(common.parse_color(item, f"candidate_p{color}")[0] for item in aggregates)
        for color in range(2)
    ]
    sweeps = {
        name: sum(common.exact_int(item, name) for item in aggregates)
        for name in PAIR_FIELDS
    }
    errors: list[str] = []
    if candidate_wins < 174:
        errors.append("candidate has fewer than 174 pooled wins")
    for color, wins in enumerate(color_wins):
        if wins < 83:
            errors.append(f"candidate has fewer than 83 pooled wins as color {color}")
    if sum(sweeps.values()) != 159 or sweeps["unresolved_pairs"] != 0:
        errors.append("pooled sweep accounting is not exactly 159 resolved pairs")
    if candidate_wins != 2 * sweeps["candidate_sweeps"] + sweeps["split_pairs"]:
        errors.append("pooled candidate wins do not reconcile with sweeps")
    if reference_wins != 2 * sweeps["reference_sweeps"] + sweeps["split_pairs"]:
        errors.append("pooled reference wins do not reconcile with sweeps")
    sign_test = exact_one_sided_sign_test(
        sweeps["candidate_sweeps"], sweeps["reference_sweeps"]
    )
    if not sign_test["passed"]:
        errors.append("pooled exact one-sided paired-sweep sign test exceeds 0.025")
    return {
        "games": 318,
        "opening_pairs": 159,
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "candidate_wins_by_physical_color": color_wins,
        "sweeps": sweeps,
        "sign_test": sign_test,
        "errors": errors,
        "acceptable": not errors,
    }


def _path_from_label(
    label: str, expected_labels: Iterable[str], *, allow_external: bool = False,
) -> Path:
    if label not in set(expected_labels):
        raise ValueError("untrusted identity label is not in the live fixed closure")
    path = Path(label)
    candidate = path if path.is_absolute() else ROOT / path
    return guard_read_path(candidate, allow_external=allow_external)


def validate_process_preflight(evidence: dict[str, Any]) -> None:
    expected_keys = {
        "clean", "self_pid", "allowed_ancestor_pids", "observed_process_count",
        "conflicts", "markers", "checked_utc", "command",
    }
    ancestors = evidence.get("allowed_ancestor_pids", []) if isinstance(
        evidence, dict
    ) else []
    if (not isinstance(evidence, dict) or set(evidence) != expected_keys or
            evidence.get("clean") is not True or evidence.get("conflicts") != [] or
            evidence.get("markers") != list(PROCESS_MARKERS) or
            not preflight.exact_json_equal(
                evidence.get("command"), preflight.process_table_command()
            ) or not preflight.is_exact_int(
                evidence.get("self_pid"), minimum=1
            ) or
            not isinstance(ancestors, list) or ancestors != sorted(set(ancestors)) or
            any(not preflight.is_exact_int(pid, minimum=1) for pid in ancestors) or
            evidence["self_pid"] not in ancestors or
            not preflight.is_exact_int(
                evidence.get("observed_process_count"), minimum=1
            ) or
            evidence["observed_process_count"] < len(ancestors)):
        raise ValueError("clean process-preflight provenance mismatch")
    require_after_t0(evidence["checked_utc"], "process preflight")


def fixed_binding_path() -> Path:
    directory = OUTPUT / "bindings"
    if not fixed_directory_exists(directory):
        raise ValueError("fixed binding registry is absent")
    paths = fixed_registry_files(directory, r"[0-9a-f]{64}\.json")
    if len(paths) != 1:
        raise ValueError("held-out qualification requires exactly one frozen binding")
    _require_fixed_json_path(paths[0], directory)
    return paths[0]


def load_and_validate_binding(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]:
    fixed_path = fixed_binding_path()
    if _lexical_absolute(path) != _lexical_absolute(fixed_path):
        raise ValueError("binding path is not the sole fixed registry entry")
    binding_directory = OUTPUT / "bindings"
    binding, binding_sha256 = load_canonical_content_addressed(
        fixed_path, BINDING_SCHEMA, binding_directory
    )
    plan = validate_plan()
    expected_keys = {
        "schema", "status", "created_utc", "campaign_id", "campaign_t0_utc",
        "plan", "campaign_manifest", "candidate_qualification_id",
        "qualification_key", "binding_claim", "candidate_commit",
        "configuration", "bank_registry_from_campaign_metadata_only",
        "dependency_identities", "dependency_routing", "preflight_receipt",
        "preflight_summary", "compiler_identities", "compiler_records",
        "tool_identities", "environment", "host", "runtime",
        "process_preflight", "git",
        "bank_files_accessed",
    }
    if (set(binding) != expected_keys or
            binding.get("status") != "frozen-unopened-heldout" or
            binding.get("campaign_id") != CAMPAIGN_ID or
            binding.get("campaign_t0_utc") != CAMPAIGN_T0_UTC or
            binding.get("plan", {}).get("sha256") != PLAN_SHA256 or
            binding.get("plan", {}).get("path") != identity_label(PLAN) or
            binding.get("campaign_manifest") != {
                "path": identity_label(CAMPAIGN), "sha256": CAMPAIGN_SHA256,
            } or
            not preflight.exact_json_equal(
                binding.get("configuration"), plan["configuration"]
            ) or not preflight.exact_json_equal(
                binding.get("bank_registry_from_campaign_metadata_only"),
                plan["banks"],
            ) or not preflight.exact_json_equal(
                binding.get("bank_files_accessed"), []
            )):
        raise ValueError("binding does not match frozen plan")
    require_after_t0(binding["created_utc"], "binding creation")
    git = require_clean_tracked_tree()
    if git["head"] != binding.get("candidate_commit"):
        raise ValueError("binding candidate commit is not current HEAD")
    validate_git_state(binding.get("git", {}), git["head"])
    if not preflight.exact_json_equal(binding["git"], git):
        raise ValueError("bound candidate git provenance changed")
    binding_paths, dependency_routing = collect_binding_paths()
    require_tracked_head_paths(binding_paths, git["head"])
    live = identities(binding_paths, allow_external=True)
    receipt_path, receipt, receipt_sha256 = fixed_preflight_receipt(
        git["head"], live
    )
    compiler_records = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    if not preflight.exact_json_equal(
            compiler_records, binding.get("compiler_records")):
        raise ValueError("bound compiler record changed")
    for name in ("clang", "gnu"):
        identity = compiler_records[name]["executable"]
        live[identity["path"]] = identity
    expected_tools = {
        name: record["executable"]
        for name, record in receipt["tool_identities_after"].items()
    }
    if not preflight.exact_json_equal(
            binding.get("tool_identities"), expected_tools):
        raise ValueError("bound fixed-tool identity changed")
    for identity in expected_tools.values():
        live[identity["path"]] = file_identity(
            Path(identity["path"]), allow_external=True
        )
    for identity in receipt["comparison_gate"]["runtime_linkage"][
            "materialized_dependencies"].values():
        live[identity["path"]] = file_identity(
            Path(identity["path"]), allow_external=True
        )
    dependency_routing["complete_dependency_count_with_compilers"] = len(live)
    if not preflight.exact_json_equal(
            live, binding.get("dependency_identities")):
        raise ValueError("live dependency/build/binary identities differ from binding")
    if not preflight.exact_json_equal(
            dependency_routing, binding.get("dependency_routing")):
        raise ValueError("compiler dependency routing differs from binding")
    key = qualification_key(plan, git["head"], live, receipt_sha256)
    identifier = candidate_qualification_id(key)
    if (not preflight.exact_json_equal(
            key, binding.get("qualification_key")) or identifier !=
            binding.get("candidate_qualification_id")):
        raise ValueError("candidate qualification identity mismatch")
    receipt_identity = binding.get("preflight_receipt", {})
    if not preflight.exact_json_equal(
            file_identity(receipt_path), receipt_identity):
        raise ValueError("final-source preflight receipt identity changed")
    expected_preflight_summary = {
        "status": receipt["status"],
        "checks": receipt["checks"],
        "technical_successor": receipt["technical_successor"],
        "compilers": receipt["compilers"],
        "comparison_gate": receipt["comparison_gate"],
        "builds": receipt["builds"],
    }
    if not preflight.exact_json_equal(
            binding.get("preflight_summary"), expected_preflight_summary):
        raise ValueError("bound preflight summary changed")
    compiler_identities = binding.get("compiler_identities", {})
    for name in ("clang", "gnu"):
        recorded = compiler_identities.get(name)
        if not preflight.exact_json_equal(
                recorded, compiler_records[name]["executable"]):
            raise ValueError(f"bound compiler identity changed: {name}")
    if (not preflight.exact_json_equal(
            binding.get("environment"), preflight.environment_record()
        ) or not preflight.exact_json_equal(
            binding.get("host"), preflight.host_identity()
        )):
        raise ValueError("bound environment or host changed")
    expected_runtime = {
        "python_version": sys.version,
        "python_executable": file_identity(
            preflight.external_executable_path(Path(sys.executable)),
            allow_external=True,
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if not preflight.exact_json_equal(binding.get("runtime"), expected_runtime):
        raise ValueError("bound runtime changed")
    validate_process_preflight(binding.get("process_preflight", {}))
    claim = dict(binding.get("binding_claim", {}))
    claim_label = claim.pop("path", "")
    expected_claim = binding_claim_path(identifier)
    if claim_label != identity_label(expected_claim):
        raise ValueError("binding claim label mismatch")
    persisted_claim = validate_binding_claim(
        expected_claim, identifier, key, receipt_sha256
    )
    if not preflight.exact_json_equal(claim, persisted_claim):
        raise ValueError("embedded binding claim differs from durable claim")
    created_time = parse_utc(binding["created_utc"])
    if (parse_utc(persisted_claim["claimed_utc"]) > created_time or
            parse_utc(receipt["created_utc"]) > created_time or
            parse_utc(binding["process_preflight"]["checked_utc"]) >
            created_time):
        raise ValueError("binding provenance timestamps are out of order")
    return binding, binding_sha256, plan, git


def claim_path(identifier: str, stage: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("stage-claim identifier is not a lowercase SHA-256")
    if stage not in ("validation", "final"):
        raise ValueError("stage claim must be validation or final")
    return OUTPUT / "claims" / f"{identifier}.{stage}.json"


def validate_stage_claim_registry(identifier: str) -> None:
    directory = OUTPUT / "claims"
    files = fixed_registry_files(
        directory, r"[0-9a-f]{64}\.(?:validation|final)\.json"
    )
    allowed = {
        claim_path(identifier, "validation"),
        claim_path(identifier, "final"),
    }
    if any(path not in allowed for path in files):
        raise ValueError("foreign claim exists in single-finalist registry")


def create_stage_claim(
    identifier: str, stage: str, binding_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = claim_path(identifier, stage)
    ensure_directory_durable(path.parent)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise OSError("fixed stage-claim registry is invalid")
    claimed_utc = utc_now()
    require_after_t0(claimed_utc, f"{stage} claim")
    payload = {
        "schema": CLAIM_SCHEMA,
        "candidate_qualification_id": identifier,
        "stage": stage,
        "binding_sha256": binding_sha256,
        "claimed_utc": claimed_utc,
        "one_shot": True,
        "claim_precedes_first_stage_bank_byte": True,
    }
    raw = canonical_json(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ValueError(f"{stage} attempt claim is already spent") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The exclusive pathname remains as a spent claim even if persistence
        # was interrupted. It must never be removed for a retry.
        raise
    fsync_directory_and_parent(path.parent)
    if path.read_bytes() != raw:
        raise OSError("atomic stage-claim readback failed")
    return path, payload


def validate_stage_claim(
    identifier: str, stage: str, binding_sha256: str,
    embedded: dict[str, Any],
) -> None:
    expected_path = claim_path(identifier, stage)
    claim = dict(embedded)
    label = claim.pop("path", "")
    if label != identity_label(expected_path):
        raise ValueError("persisted stage claim path mismatch")
    raw = guard_read_path(expected_path).read_bytes()
    expected_keys = {
        "schema", "candidate_qualification_id", "stage", "binding_sha256",
        "claimed_utc", "one_shot", "claim_precedes_first_stage_bank_byte",
    }
    if (set(claim) != expected_keys or canonical_json(claim) != raw or
            claim.get("schema") != CLAIM_SCHEMA or
            claim.get("candidate_qualification_id") != identifier or
            claim.get("stage") != stage or
            claim.get("binding_sha256") != binding_sha256 or
            claim.get("one_shot") is not True or
            claim.get("claim_precedes_first_stage_bank_byte") is not True):
        raise ValueError("persisted stage claim binding mismatch")
    require_after_t0(claim["claimed_utc"], f"{stage} claim")


def validate_stage_bank_identities(
    plan: dict[str, Any], stage: str,
    stage_identities: dict[str, dict[str, Any]],
) -> None:
    for record in stage_records(plan, stage):
        identity = stage_identities.get(record["path"])
        if (identity is None or identity.get("sha256") != record["sha256"] or
                not preflight.is_exact_int(identity.get("bytes"), minimum=0) or
                not preflight.exact_json_equal(
                    identity.get("bytes"), record["bytes"]
                ) or
                identity.get("ascii") is not True):
            raise ValueError(f"{stage} bank identity mismatch: {record['path']}")


def stage_report_matches(
    report: dict[str, Any], identifier: str, stage: str,
    binding_sha256: str,
) -> bool:
    return (
        report.get("schema") == REPORT_SCHEMA and
        report.get("candidate_qualification_id") == identifier and
        report.get("stage") == stage and
        report.get("binding_sha256") == binding_sha256
    )


def find_stage_reports(
    identifier: str, stage: str, binding_sha256: str,
) -> list[tuple[Path, str, dict[str, Any]]]:
    matches: list[tuple[Path, str, dict[str, Any]]] = []
    directory = OUTPUT / "reports" / stage
    if not fixed_directory_exists(directory):
        return []
    for path in fixed_registry_files(directory, r"[0-9a-f]{64}\.json"):
        report, digest = load_canonical_content_addressed(
            path, REPORT_SCHEMA, directory
        )
        if not stage_report_matches(report, identifier, stage, binding_sha256):
            raise ValueError(f"foreign {stage} report exists in single-finalist registry")
        matches.append((path, digest, report))
    return matches


def require_final_attempt_unopened(identifier: str) -> None:
    """Prove a rejected VALIDATION has no durable FINAL attempt artifacts."""
    if os.path.lexists(claim_path(identifier, "final")):
        raise ValueError("rejected validation coexists with a FINAL claim")
    directory = OUTPUT / "reports" / "final"
    if fixed_registry_files(directory, r"[0-9a-f]{64}\.json"):
        raise ValueError("rejected validation coexists with a FINAL report")


def validate_persisted_stage_report(
    report: dict[str, Any], binding: dict[str, Any], binding_sha256: str,
    plan: dict[str, Any], stage: str,
) -> None:
    identifier = binding["candidate_qualification_id"]
    expected_keys = {
        "schema", "campaign_id", "campaign_t0_utc", "classification",
        "final_qualification", "producer", "candidate_qualification_id",
        "binding_sha256", "stage", "claim", "started_utc", "ended_utc",
        "elapsed_monotonic_ns", "command_argv", "command_shell", "cwd",
        "timeout_seconds", "environment", "host_before", "host_after",
        "runtime", "returncode", "timed_out", "os_error_class", "stdout",
        "stderr", "process_preflight", "git_before", "git_after",
        "inputs_before", "inputs_after", "stable_inputs",
        "compiler_records_before", "compiler_records_after",
        "stable_compilers", "accessed_bank_paths", "parsed",
        "validation_codes", "threshold_errors", "stage_acceptable",
        "replay_corrections", "transcripts",
    }
    if set(report) != expected_keys or not stage_report_matches(
            report, identifier, stage, binding_sha256):
        raise ValueError("persisted stage report identity mismatch")
    if (report.get("campaign_id") != CAMPAIGN_ID or
            report.get("campaign_t0_utc") != CAMPAIGN_T0_UTC or
            report.get("classification") !=
            f"untouched-{stage}-one-shot-qualification-stage" or
            report.get("final_qualification") is not False or
            not preflight.exact_json_equal(
                report.get("producer"), binding["dependency_identities"].get(
                    identity_label(RECORDER)
                )
            ) or report.get("replay_corrections") != "disabled" or
            report.get("transcripts") != "not-retained"):
        raise ValueError("persisted stage provenance mismatch")
    validate_stage_claim(
        identifier, stage, binding_sha256, report.get("claim", {})
    )
    require_after_t0(report["started_utc"], f"{stage} start")
    require_after_t0(report["ended_utc"], f"{stage} end")
    claim_time = parse_utc(report["claim"]["claimed_utc"])
    start_time = parse_utc(report["started_utc"])
    end_time = parse_utc(report["ended_utc"])
    if (claim_time > start_time or start_time > end_time or
            not preflight.is_exact_int(
                report["elapsed_monotonic_ns"], minimum=1
            ) or
            report["elapsed_monotonic_ns"] <= 0 or
            report["elapsed_monotonic_ns"] >=
            (STAGE_TIMEOUT_SECONDS[stage] + 300) * 1_000_000_000):
        raise ValueError("persisted stage timestamps are inconsistent")
    wall_seconds = (end_time - start_time).total_seconds()
    monotonic_seconds = report["elapsed_monotonic_ns"] / 1_000_000_000
    if wall_seconds < 0 or abs(wall_seconds - monotonic_seconds) > 60:
        raise ValueError("persisted stage wall/monotonic timing mismatch")
    expected_command = command_for_stage(plan, stage)
    if (not preflight.exact_json_equal(
            report.get("command_argv"), expected_command
        ) or
            report.get("command_shell") != shlex.join(expected_command) or
            report.get("cwd") != str(ROOT) or
            report.get("timeout_seconds") != STAGE_TIMEOUT_SECONDS[stage] or
            not preflight.exact_json_equal(
                report.get("environment"), preflight.environment_record()
            ) or not preflight.exact_json_equal(
                report.get("runtime"), binding["runtime"]
            )):
        raise ValueError("persisted stage command/environment/host mismatch")
    validate_process_preflight(report.get("process_preflight", {}))
    if parse_utc(report["process_preflight"]["checked_utc"]) > claim_time:
        raise ValueError("process preflight occurred after the atomic stage claim")
    allowed_codes = {
        "input_identity_before", "process_execution", "input_identity_after",
        "stdout_contract", "tracked_state_after", "compiler_changed",
        "host_changed",
    }
    validation_codes = report.get("validation_codes")
    if (not isinstance(validation_codes, list) or
            len(validation_codes) != len(set(validation_codes)) or
            any(code not in allowed_codes for code in validation_codes)):
        raise ValueError("persisted validation-code registry mismatch")
    for stream_name in ("stdout", "stderr"):
        stream = report.get(stream_name, {})
        if (not isinstance(stream, dict) or set(stream) != {
                "retained", "bytes", "sha256", "empty", "policy"
        } or stream.get("retained") is not False or
                not preflight.is_exact_int(stream.get("bytes"), minimum=0) or
                not isinstance(stream.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", stream["sha256"]) or
                type(stream.get("empty")) is not bool or
                stream.get("empty") is not (stream["bytes"] == 0) or
                stream.get("policy") != "digest-only-no-raw-stream"):
            raise ValueError(f"persisted {stream_name} evidence mismatch")
        if stream["empty"] and stream["sha256"] != sha256_bytes(b""):
            raise ValueError(f"empty persisted {stream_name} digest mismatch")
    parsed = report.get("parsed", {})
    expected_threshold_errors: list[str] = []
    if parsed:
        reconstructed_stdout = "\n".join([
            *parsed.get("bank_lines", []),
            str(parsed.get("aggregate_line", "")),
            str(parsed.get("configuration_line", "")),
        ]) + "\n"
        reparsed = validate_stage_stdout(plan, stage, reconstructed_stdout)
        if not preflight.exact_json_equal(reparsed, parsed):
            raise ValueError("persisted parsed stage payload is not reproducible")
        encoded = reconstructed_stdout.encode("utf-8")
        if (report["stdout"]["bytes"] != len(encoded) or
                report["stdout"]["sha256"] != sha256_bytes(encoded) or
                report["stdout"]["empty"] is not False):
            raise ValueError("persisted stdout digest differs from parsed summaries")
        expected_threshold_errors = stage_threshold_errors(
            stage, parsed["aggregate"]
        )
        if "stdout_contract" in validation_codes:
            raise ValueError("valid parsed stdout has a stdout-contract rejection")
    elif "stdout_contract" not in validation_codes:
        raise ValueError("empty parsed payload lacks stdout-contract rejection")
    if not preflight.exact_json_equal(
            report.get("threshold_errors"), expected_threshold_errors):
        raise ValueError("persisted stage threshold result mismatch")
    before = report.get("inputs_before")
    after = report.get("inputs_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("persisted stage input maps are malformed")
    expected_labels = set(binding["dependency_identities"])
    expected_labels.update(item["path"] for item in stage_records(plan, stage))
    if before and set(before) != expected_labels:
        raise ValueError("persisted stage input map is incomplete or overbroad")
    for label, identity in binding["dependency_identities"].items():
        if before and not preflight.exact_json_equal(before.get(label), identity):
            raise ValueError(f"persisted bound dependency mismatch: {label}")
    if before:
        validate_stage_bank_identities(plan, stage, before)
        if "input_identity_before" in validation_codes:
            raise ValueError("complete before-input map has a rejection code")
    elif "input_identity_before" not in validation_codes:
        raise ValueError("missing before-input map lacks a rejection code")
    if after and set(after) != expected_labels:
        raise ValueError("persisted after-input map is incomplete or overbroad")
    if after:
        validate_stage_bank_identities(plan, stage, after)
        if "input_identity_after" in validation_codes:
            raise ValueError("complete after-input map has a rejection code")
    elif "input_identity_after" not in validation_codes:
        raise ValueError("missing after-input map lacks a rejection code")
    expected_banks = [item["path"] for item in stage_records(plan, stage)]
    if report.get("accessed_bank_paths") != expected_banks:
        raise ValueError("persisted accessed-bank registry mismatch")
    if stage == "validation" and any("final_d" in path for path in before):
        raise ValueError("validation report accessed FINAL")
    git_before = report.get("git_before", {})
    validate_git_state(git_before, binding["candidate_commit"])
    git_after = report.get("git_after")
    if "tracked_state_after" in validation_codes:
        if git_after is not None:
            raise ValueError("failed post-stage git check retained untrusted state")
    else:
        validate_git_state(git_after, binding["candidate_commit"])
    compiler_before = report.get("compiler_records_before")
    compiler_after = report.get("compiler_records_after")
    stable_compilers = (
        preflight.exact_json_equal(compiler_before, compiler_after) and
        preflight.exact_json_equal(
            compiler_after, binding["compiler_records"]
        )
    )
    if report.get("stable_compilers") is not stable_compilers:
        raise ValueError("persisted stable-compiler result mismatch")
    if stable_compilers == ("compiler_changed" in validation_codes):
        raise ValueError("persisted compiler rejection code mismatch")
    if not stable_compilers and (compiler_before != {} or compiler_after != {}):
        raise ValueError("failed compiler check retained untrusted details")
    stable_host = (
        preflight.exact_json_equal(
            report.get("host_before"), report.get("host_after")
        ) and preflight.exact_json_equal(
            report.get("host_after"), binding["host"]
        )
    )
    if stable_host == ("host_changed" in validation_codes):
        raise ValueError("persisted host rejection code mismatch")
    if not stable_host and (
            report.get("host_before") != {} or report.get("host_after") != {}):
        raise ValueError("failed host check retained untrusted details")
    stable_inputs = bool(before) and preflight.exact_json_equal(before, after)
    if report.get("stable_inputs") is not stable_inputs:
        raise ValueError("persisted stable-input result mismatch")
    process_success = (
        report.get("returncode") == 0 and report.get("timed_out") is False and
        report.get("os_error_class") is None and
        report["stderr"]["empty"] is True
    )
    if (not isinstance(report.get("timed_out"), bool) or
            not (report.get("returncode") is None or
                 preflight.is_exact_int(report.get("returncode"))) or
            not (report.get("os_error_class") is None or
                 (isinstance(report.get("os_error_class"), str) and
                  re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",
                               report["os_error_class"])))):
        raise ValueError("persisted process result is malformed")
    if process_success == ("process_execution" in validation_codes):
        raise ValueError("persisted process-execution code mismatch")
    acceptable = (
        process_success and stable_inputs and not validation_codes and
        not expected_threshold_errors and bool(parsed)
    )
    if report.get("stage_acceptable") is not acceptable:
        raise ValueError("persisted stage acceptance was not recomputed exactly")
    if acceptable and report["stderr"]["bytes"] != 0:
        raise ValueError("accepted stage has nonempty stderr")
    # A durable claim now permits re-reading exactly this stage's banks.  This
    # independently revalidates every persisted input before a decision is
    # reused or FINAL is considered.
    if before:
        bound_paths = [
            _path_from_label(
                label,
                binding["dependency_identities"], allow_external=True,
            )
            for label in binding["dependency_identities"]
        ]
        bank_paths = [ROOT / item["path"] for item in stage_records(plan, stage)]
        live = identities(
            (*bound_paths, *bank_paths),
            allowed_stage_banks=bank_paths, allow_external=True,
        )
        if not preflight.exact_json_equal(live, after):
            raise ValueError("persisted stage inputs differ from live claimed inputs")


def _stream_evidence(data: str) -> dict[str, Any]:
    raw = data.encode("utf-8")
    return {
        "retained": False,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "empty": raw == b"",
        "policy": "digest-only-no-raw-stream",
    }


def _run_process(command: list[str], timeout: int) -> dict[str, Any]:
    timed_out = False
    os_error: str | None = None
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=preflight.sanitized_environment(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=timeout, start_new_session=True,
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
    return {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "os_error": os_error,
    }


def run_stage(
    binding: dict[str, Any], binding_sha256: str, plan: dict[str, Any],
    stage: str, process_preflight: dict[str, Any], git_before: dict[str, str],
) -> tuple[Path, str, dict[str, Any]]:
    identifier = binding["candidate_qualification_id"]
    claim, claim_payload = create_stage_claim(identifier, stage, binding_sha256)
    # The claim above is the last operation before stage bank paths are turned
    # into files and read. FINAL reaches this line only after accepted VALIDATION.
    bank_paths = [ROOT / item["path"] for item in stage_records(plan, stage)]
    dependency_paths: tuple[Path, ...] = ()
    command = command_for_stage(plan, stage)
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    parsed: dict[str, Any] = {}
    validation_codes: list[str] = []
    threshold_errors: list[str] = []
    process_result = {
        "returncode": None, "stdout": "", "stderr": "",
        "timed_out": False, "os_error": None,
    }
    def reject(code: str) -> None:
        if code not in validation_codes:
            validation_codes.append(code)

    try:
        bound_paths = [
            _path_from_label(
                label,
                binding["dependency_identities"], allow_external=True,
            )
            for label in binding["dependency_identities"]
        ]
        dependency_paths = (*bound_paths, *bank_paths)
        before = identities(
            dependency_paths, allowed_stage_banks=bank_paths,
            allow_external=True,
        )
        validate_stage_bank_identities(plan, stage, before)
    except (OSError, ValueError):
        before = {}
        reject("input_identity_before")
    try:
        compiler_records_before = {
            name: preflight.discover_compiler(name)[1]
            for name in ("clang", "gnu")
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        compiler_records_before = {}
        reject("compiler_changed")
    try:
        host_before = preflight.host_identity()
    except (OSError, ValueError, subprocess.SubprocessError):
        host_before = {}
        reject("host_changed")
    if not validation_codes:
        process_result = _run_process(command, STAGE_TIMEOUT_SECONDS[stage])
        if (process_result["returncode"] != 0 or process_result["timed_out"] or
                process_result["os_error"] is not None or
                process_result["stderr"] != ""):
            reject("process_execution")
    else:
        reject("process_execution")
    try:
        if before and dependency_paths:
            after = identities(
                dependency_paths, allowed_stage_banks=bank_paths,
                allow_external=True,
            )
            validate_stage_bank_identities(plan, stage, after)
        else:
            raise ValueError("before-input identity unavailable")
    except (OSError, ValueError):
        after = {}
        reject("input_identity_after")
    try:
        if process_result["stdout"]:
            parsed = validate_stage_stdout(plan, stage, process_result["stdout"])
            threshold_errors = stage_threshold_errors(stage, parsed["aggregate"])
        else:
            raise ValueError("empty stdout")
    except (ValueError, OverflowError):
        reject("stdout_contract")
        parsed = {}
        threshold_errors = []
    try:
        compiler_records_after = {
            name: preflight.discover_compiler(name)[1]
            for name in ("clang", "gnu")
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        compiler_records_after = {}
    if (not preflight.exact_json_equal(
            compiler_records_before, compiler_records_after
        ) or not preflight.exact_json_equal(
            compiler_records_after, binding["compiler_records"]
        )):
        reject("compiler_changed")
        compiler_records_before = {}
        compiler_records_after = {}
    try:
        host_after = preflight.host_identity()
    except (OSError, ValueError, subprocess.SubprocessError):
        host_after = {}
    if (not preflight.exact_json_equal(host_before, host_after) or
            not preflight.exact_json_equal(host_after, binding["host"])):
        reject("host_changed")
        host_before = {}
        host_after = {}
    try:
        git_after = require_clean_tracked_tree()
    except (OSError, ValueError, subprocess.SubprocessError):
        reject("tracked_state_after")
        git_after = None
    ended_ns = time.monotonic_ns()
    ended_utc = utc_now()
    stable_inputs = bool(before) and preflight.exact_json_equal(before, after)
    acceptable = (
        process_result["returncode"] == 0 and
        process_result["timed_out"] is False and
        process_result["os_error"] is None and
        process_result["stderr"] == "" and
        stable_inputs and isinstance(git_after, dict) and
        git_after["head"] == binding["candidate_commit"] and
        not validation_codes and not threshold_errors and bool(parsed) and
        preflight.exact_json_equal(
            compiler_records_before, compiler_records_after
        ) and preflight.exact_json_equal(
            compiler_records_after, binding["compiler_records"]
        ) and preflight.exact_json_equal(host_before, host_after) and
        preflight.exact_json_equal(host_after, binding["host"])
    )
    report = {
        "schema": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "campaign_t0_utc": CAMPAIGN_T0_UTC,
        "classification": f"untouched-{stage}-one-shot-qualification-stage",
        "final_qualification": False,
        "producer": binding["dependency_identities"][identity_label(RECORDER)],
        "candidate_qualification_id": identifier,
        "binding_sha256": binding_sha256,
        "stage": stage,
        "claim": {**claim_payload, "path": identity_label(claim)},
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "elapsed_monotonic_ns": ended_ns - started_ns,
        "command_argv": command,
        "command_shell": shlex.join(command),
        "cwd": str(ROOT),
        "timeout_seconds": STAGE_TIMEOUT_SECONDS[stage],
        "environment": preflight.environment_record(),
        "host_before": host_before,
        "host_after": host_after,
        "runtime": binding["runtime"],
        "returncode": process_result["returncode"],
        "timed_out": process_result["timed_out"],
        "os_error_class": process_result["os_error"],
        "stdout": _stream_evidence(process_result["stdout"]),
        "stderr": _stream_evidence(process_result["stderr"]),
        "process_preflight": process_preflight,
        "git_before": git_before,
        "git_after": git_after,
        "inputs_before": before,
        "inputs_after": after,
        "stable_inputs": stable_inputs,
        "compiler_records_before": compiler_records_before,
        "compiler_records_after": compiler_records_after,
        "stable_compilers": compiler_records_before == compiler_records_after ==
        binding["compiler_records"],
        "accessed_bank_paths": [item["path"] for item in stage_records(plan, stage)],
        "parsed": parsed,
        "validation_codes": validation_codes,
        "threshold_errors": threshold_errors,
        "stage_acceptable": acceptable,
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }
    path, digest = persist_content_addressed(OUTPUT / "reports" / stage, report)
    persisted, _ = load_canonical_content_addressed(
        path, REPORT_SCHEMA, OUTPUT / "reports" / stage
    )
    if not preflight.exact_json_equal(persisted, report):
        raise OSError("stage report semantic readback failed")
    return path, digest, report


def find_decisions(
    identifier: str, binding_sha256: str,
) -> list[tuple[Path, str, dict[str, Any]]]:
    matches: list[tuple[Path, str, dict[str, Any]]] = []
    directory = OUTPUT / "decisions"
    if not fixed_directory_exists(directory):
        return []
    for path in fixed_registry_files(directory, r"[0-9a-f]{64}\.json"):
        decision, digest = load_canonical_content_addressed(
            path, DECISION_SCHEMA, directory
        )
        if (decision.get("candidate_qualification_id") != identifier or
                decision.get("binding_sha256") != binding_sha256):
            raise ValueError("foreign decision exists in single-finalist registry")
        matches.append((path, digest, decision))
    return matches


def _report_reference(
    stage: str, report: tuple[Path, str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "path": identity_label(report[0]),
        "sha256": report[1],
        "acceptable": report[2]["stage_acceptable"],
    }


def _load_decision_report(
    reference: dict[str, Any], stage: str, binding: dict[str, Any],
    binding_sha256: str, plan: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    if (not isinstance(reference, dict) or set(reference) != {
            "stage", "path", "sha256", "acceptable"
    } or reference.get("stage") != stage or
            not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", "")) or
            not isinstance(reference.get("acceptable"), bool)):
        raise ValueError(f"decision {stage} report reference is malformed")
    directory = OUTPUT / "reports" / stage
    path = directory / f"{reference['sha256']}.json"
    if reference["path"] != identity_label(path):
        raise ValueError(f"decision {stage} report path is not fixed")
    report, digest = load_canonical_content_addressed(
        path, REPORT_SCHEMA, directory
    )
    validate_persisted_stage_report(
        report, binding, binding_sha256, plan, stage
    )
    if reference["acceptable"] is not report["stage_acceptable"]:
        raise ValueError(f"decision {stage} acceptance differs from report")
    return path, digest, report


def stage_decision_errors(stage: str, report: dict[str, Any]) -> list[str]:
    errors = [
        f"{stage} evidence failure: {code}"
        for code in report.get("validation_codes", [])
    ]
    errors.extend(
        f"{stage} threshold failure: {message}"
        for message in report.get("threshold_errors", [])
    )
    if report.get("stable_inputs") is False and not any(
            "input_identity" in code
            for code in report.get("validation_codes", [])):
        errors.append(f"{stage} evidence failure: inputs changed during stage")
    if not errors:
        errors.append(f"{stage} stage failed its frozen acceptance conjunction")
    return errors


def decision_payload(
    binding: dict[str, Any], binding_sha256: str,
    validation: tuple[Path, str, dict[str, Any]],
    final: tuple[Path, str, dict[str, Any]] | None,
    created_utc: str,
) -> dict[str, Any]:
    validation_ref = _report_reference("validation", validation)
    final_ref: dict[str, Any] | None = None
    pooled: dict[str, Any] | None = None
    errors: list[str] = []
    if not validation[2]["stage_acceptable"]:
        if final is not None:
            raise ValueError("rejected validation cannot reference FINAL")
        status = "rejected-at-validation-final-unopened"
        errors.append("validation stage failed; FINAL remained unopened")
        errors.extend(stage_decision_errors("validation", validation[2]))
    elif final is None:
        raise ValueError("accepted validation requires a final report")
    else:
        final_ref = _report_reference("final", final)
        if not final[2]["stage_acceptable"]:
            status = "rejected-at-final"
            errors.append("final stage failed")
            errors.extend(stage_decision_errors("final", final[2]))
        else:
            pooled = pooled_evaluation(validation[2], final[2])
            errors.extend(pooled["errors"])
            status = (
                "accepted-heldout-qualification" if pooled["acceptable"]
                else "rejected-at-pooled-qualification"
            )
    acceptable = status == "accepted-heldout-qualification"
    return {
        "schema": DECISION_SCHEMA,
        "created_utc": created_utc,
        "campaign_id": CAMPAIGN_ID,
        "campaign_t0_utc": CAMPAIGN_T0_UTC,
        "classification": "single-finalist-untouched-heldout-decision",
        "final_qualification": False,
        "producer": binding["dependency_identities"][identity_label(RECORDER)],
        "candidate_qualification_id": binding["candidate_qualification_id"],
        "binding_sha256": binding_sha256,
        "status": status,
        "validation_report": validation_ref,
        "final_report": final_ref,
        "pooled": pooled,
        "errors": errors,
        "heldout_qualification_acceptable": acceptable,
        "arena_authorization": acceptable,
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }


def validate_persisted_decision(
    decision: dict[str, Any], binding: dict[str, Any],
    binding_sha256: str, plan: dict[str, Any],
) -> None:
    expected_keys = {
        "schema", "created_utc", "campaign_id", "campaign_t0_utc",
        "classification", "final_qualification", "producer",
        "candidate_qualification_id", "binding_sha256", "status",
        "validation_report",
        "final_report", "pooled", "errors",
        "heldout_qualification_acceptable", "arena_authorization",
        "replay_corrections", "transcripts",
    }
    if (not isinstance(decision, dict) or set(decision) != expected_keys or
            decision.get("schema") != DECISION_SCHEMA or
            decision.get("campaign_id") != CAMPAIGN_ID or
            decision.get("campaign_t0_utc") != CAMPAIGN_T0_UTC or
            decision.get("classification") !=
            "single-finalist-untouched-heldout-decision" or
            decision.get("final_qualification") is not False or
            decision.get("producer") !=
            binding["dependency_identities"].get(identity_label(RECORDER)) or
            decision.get("candidate_qualification_id") !=
            binding["candidate_qualification_id"] or
            decision.get("binding_sha256") != binding_sha256 or
            decision.get("replay_corrections") != "disabled" or
            decision.get("transcripts") != "not-retained"):
        raise ValueError("persisted held-out decision provenance mismatch")
    require_after_t0(decision["created_utc"], "held-out decision")
    validation = _load_decision_report(
        decision["validation_report"], "validation", binding,
        binding_sha256, plan,
    )
    if parse_utc(decision["created_utc"]) < parse_utc(
            validation[2]["ended_utc"]):
        raise ValueError("decision predates validation report")
    # This branch intentionally neither constructs nor reads a FINAL report or
    # bank path. A failed VALIDATION remains the hard data-boundary stop.
    if validation[2]["stage_acceptable"]:
        if decision["final_report"] is None:
            raise ValueError("accepted validation decision lacks FINAL report")
        final = _load_decision_report(
            decision["final_report"], "final", binding,
            binding_sha256, plan,
        )
        if parse_utc(final[2]["started_utc"]) < parse_utc(
                validation[2]["ended_utc"]):
            raise ValueError("FINAL overlaps or predates VALIDATION")
        if parse_utc(decision["created_utc"]) < parse_utc(final[2]["ended_utc"]):
            raise ValueError("decision predates FINAL report")
    else:
        if decision["final_report"] is not None:
            raise ValueError("rejected validation decision addresses FINAL")
        require_final_attempt_unopened(binding["candidate_qualification_id"])
        final = None
    expected = decision_payload(
        binding, binding_sha256, validation, final, decision["created_utc"]
    )
    if not preflight.exact_json_equal(decision, expected):
        raise ValueError("persisted decision semantics were not recomputed exactly")


def persist_decision(
    binding: dict[str, Any], binding_sha256: str,
    validation: tuple[Path, str, dict[str, Any]],
    final: tuple[Path, str, dict[str, Any]] | None,
) -> tuple[Path, str, dict[str, Any]]:
    identifier = binding["candidate_qualification_id"]
    if find_decisions(identifier, binding_sha256):
        raise ValueError("qualification decision already exists")
    validate_persisted_stage_report(
        validation[2], binding, binding_sha256, validate_plan(), "validation"
    )
    if final is not None:
        validate_persisted_stage_report(
            final[2], binding, binding_sha256, validate_plan(), "final"
        )
    elif not validation[2]["stage_acceptable"]:
        require_final_attempt_unopened(binding["candidate_qualification_id"])
    created_utc = utc_now()
    require_after_t0(created_utc, "held-out decision")
    decision = decision_payload(
        binding, binding_sha256, validation, final, created_utc
    )
    path, digest = persist_content_addressed(OUTPUT / "decisions", decision)
    persisted, persisted_digest = load_canonical_content_addressed(
        path, DECISION_SCHEMA, OUTPUT / "decisions"
    )
    if persisted_digest != digest:
        raise OSError("held-out decision digest readback mismatch")
    validate_persisted_decision(
        persisted, binding, binding_sha256, validate_plan()
    )
    return path, digest, decision


def _one_report_or_none(
    identifier: str, stage: str, binding_sha256: str,
) -> tuple[Path, str, dict[str, Any]] | None:
    reports = find_stage_reports(identifier, stage, binding_sha256)
    if len(reports) > 1:
        raise ValueError(f"multiple {stage} reports exist for one-shot identity")
    return reports[0] if reports else None


def run_qualification() -> tuple[Path, str, dict[str, Any]]:
    ensure_directory_durable(PRIVATE_LOCK.parent)
    with open_lock(PRIVATE_LOCK) as private_handle, open_lock(LOCK) as shared_handle:
        try:
            fcntl.flock(private_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(shared_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another campaign benchmark/clock job owns the lock") from error
        binding_path = fixed_binding_path()
        binding, binding_sha256, plan, git = load_and_validate_binding(binding_path)
        identifier = binding["candidate_qualification_id"]
        validate_stage_claim_registry(identifier)
        decisions = find_decisions(identifier, binding_sha256)
        if len(decisions) > 1:
            raise ValueError("multiple qualification decisions exist")
        if decisions:
            validate_persisted_decision(
                decisions[0][2], binding, binding_sha256, plan
            )
            return decisions[0]

        validation = _one_report_or_none(identifier, "validation", binding_sha256)
        if validation is None:
            if os.path.lexists(claim_path(identifier, "validation")):
                raise ValueError("validation claim is spent without a report; retry forbidden")
            # No stage bank is read by either check.
            binding, binding_sha256, plan, git = load_and_validate_binding(
                binding_path
            )
            process = require_clean_processes()
            validation = run_stage(
                binding, binding_sha256, plan, "validation", process, git
            )
        validate_persisted_stage_report(
            validation[2], binding, binding_sha256, plan, "validation"
        )
        if not validation[2]["stage_acceptable"]:
            # This branch constructs no FINAL command/path and reads no FINAL file.
            return persist_decision(binding, binding_sha256, validation, None)
        final = _one_report_or_none(identifier, "final", binding_sha256)
        if final is None:
            if os.path.lexists(claim_path(identifier, "final")):
                raise ValueError("final claim is spent without a report; retry forbidden")
            # Revalidate the unchanged binding and process isolation only after
            # accepted VALIDATION is durable, still before the FINAL claim/read.
            binding, binding_sha256, plan, git = load_and_validate_binding(
                binding_path
            )
            process = require_clean_processes()
            final = run_stage(
                binding, binding_sha256, plan, "final", process, git
            )
        validate_persisted_stage_report(
            final[2], binding, binding_sha256, plan, "final"
        )
        return persist_decision(binding, binding_sha256, validation, final)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("bind")
    subparsers.add_parser("run")
    arguments = parser.parse_args()
    try:
        if arguments.action == "bind":
            path, digest = create_binding()
            acceptable = True
        else:
            path, digest, decision = run_qualification()
            acceptable = bool(decision["heldout_qualification_acceptable"])
    except (
        KeyError, OSError, TypeError, UnicodeError, ValueError,
        json.JSONDecodeError, subprocess.SubprocessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(path.relative_to(ROOT))
    print(f"sha256={digest}")
    if arguments.action == "run":
        print(f"heldout_qualification_acceptable={str(acceptable).lower()}")
    return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
