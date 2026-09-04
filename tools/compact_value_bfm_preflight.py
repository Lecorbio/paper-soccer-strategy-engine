#!/usr/bin/env python3
"""Source-bound, one-shot local preflight for compact_value_bfm.

The real ``run`` command creates fresh GCC, Clang, and Clang sanitizer builds.
Pure validators are intentionally separate so unit tests can supply receipts
without compiling or running the 4,096-state and timing suites.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _qualification_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_qualification.py")
    name = "compact_value_bfm_preflight_qualification"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qualification primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _qualification_module()
PreflightError = base.QualificationError

NAMESPACE = "compact_value_bfm"
PLAN_SCHEMA = "papersoccer.compact-value-bfm.preflight-plan.v1"
CLAIM_SCHEMA = "papersoccer.compact-value-bfm.preflight-claim.v1"
COMMAND_SCHEMA = "papersoccer.compact-value-bfm.command-receipt.v1"
RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.preflight-receipt.v1"
PARITY_SCHEMA = "papersoccer.compact-value-bfm.inference-parity.v1"
TIMING_SCHEMA = "papersoccer.compact-value-bfm.process-timing.v1"

RANK4_SHA256 = base.RANK4_SHA256
RANK4_BYTES = base.RANK4_BYTES
SOURCE_LIMIT_EXCLUSIVE = 95_000
PARITY_STATES = 4_096
PARITY_MAX_ERROR = 2e-6
PROCESS_COUNTS = (1, 2, 10)
FIRST_LIMIT_MS = 900.0
LATER_LIMIT_MS = 180.0
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")

BOT_RELATIVE = pathlib.Path("submissions/codingame/bots/compact_value_bfm")
RANK4_RELATIVE = pathlib.Path("submissions/codingame/bots/rank_4/submission.cpp")
COMPACT_TEST_RELATIVE = pathlib.Path("tests/codingame")
COMPACT_TEST_PATTERN = "test_compact_value_bfm*.py"
SOURCE_CLOSURE = (
    pathlib.Path("CMakeLists.txt"),
    BOT_RELATIVE / "submission.json",
    BOT_RELATIVE / "sources.txt",
    BOT_RELATIVE / "model.hpp",
    BOT_RELATIVE / "engine.hpp",
    BOT_RELATIVE / "engine.cpp",
    BOT_RELATIVE / "bot.cpp",
    BOT_RELATIVE / "submission.cpp",
    BOT_RELATIVE / "submission_test.cpp",
    BOT_RELATIVE / "comparison_gate.cpp",
    BOT_RELATIVE / "timing_probe.cpp",
    BOT_RELATIVE / "feature_probe.cpp",
    BOT_RELATIVE / "inference_probe.cpp",
    BOT_RELATIVE / "feature_parity.py",
    BOT_RELATIVE / "search_trace_probe.cpp",
    BOT_RELATIVE / "search_variant_parity.py",
    BOT_RELATIVE / "state_evaluation_cache_parity.py",
    BOT_RELATIVE / "progressive_widening_invariance.py",
    BOT_RELATIVE / "subtree_reuse_invariance.py",
    BOT_RELATIVE / "search_profile_exclusion.py",
    BOT_RELATIVE / "source_compaction_parity.py",
    BOT_RELATIVE / "rank4_gate.cpp",
    BOT_RELATIVE / "rank4_gate_support.py",
    BOT_RELATIVE / "test_rank4_gate_support.py",
    BOT_RELATIVE / "export_model.py",
    BOT_RELATIVE / "export_submission.py",
    BOT_RELATIVE / "test_exporters.py",
    pathlib.Path("submissions/codingame/tools/protocol_smoke_test.mjs"),
    pathlib.Path("tools/compact_value_bfm_preflight.py"),
    RANK4_RELATIVE,
)

SEARCH_TRACE_BASES = (
    "baseline",
    "no_feature_sort_only",
    "single_pass_selection_only",
    "combined",
)
SEARCH_TRACE_PROFILES = (
    "state_evaluation_cache_v1",
    "progressive_widening_v1",
    "subtree_reuse_v1",
)
SEARCH_TRACE_TARGETS = tuple(
    "papersoccer_codingame_compact_value_bfm_search_trace_" + suffix
    for base in SEARCH_TRACE_BASES
    for suffix in (
        base,
        *(f"{base}_{profile}" for profile in SEARCH_TRACE_PROFILES),
    )
) + ("papersoccer_codingame_compact_value_bfm_search_trace_modular",)
SEARCH_PARITY_TESTS = (
    "papersoccer_codingame_compact_value_bfm_search_variant_parity",
    *(
        f"papersoccer_codingame_compact_value_bfm_{kind}_{base}"
        for base in SEARCH_TRACE_BASES
        for kind in (
            "state_evaluation_cache_parity",
            "progressive_widening_invariance",
            "subtree_reuse_invariance",
        )
    ),
    "papersoccer_codingame_compact_value_bfm_profile_exclusion",
    "papersoccer_codingame_compact_value_bfm_source_compaction_parity",
)

BUILD_TARGETS = (
    "papersoccer_codingame_compact_value_bfm_submission",
    "papersoccer_codingame_compact_value_bfm_submission_test",
    "papersoccer_codingame_compact_value_bfm_timing_probe",
    "papersoccer_codingame_compact_value_bfm_comparison_gate",
    "papersoccer_codingame_compact_value_bfm_feature_probe",
    "papersoccer_codingame_compact_value_bfm_inference_probe",
    "papersoccer_codingame_compact_value_bfm_rank4_gate",
) + SEARCH_TRACE_TARGETS
RELEASE_TESTS = (
    "papersoccer_codingame_compact_value_bfm_submission_test",
    "papersoccer_codingame_compact_value_bfm_timing_probe",
    "papersoccer_codingame_compact_value_bfm_timing_probe_player1",
    "papersoccer_codingame_compact_value_bfm_comparison_gate_smoke",
    "papersoccer_codingame_compact_value_bfm_exporter_tests",
    "papersoccer_codingame_compact_value_bfm_feature_parity",
    "papersoccer_codingame_compact_value_bfm_submission_current",
    "papersoccer_codingame_compact_value_bfm_protocol_smoke_test",
    "papersoccer_codingame_compact_value_bfm_rank4_gate_self_test",
    "papersoccer_codingame_compact_value_bfm_rank4_gate_support_tests",
) + SEARCH_PARITY_TESTS
SANITIZER_TESTS = tuple(
    test for test in RELEASE_TESTS if "timing_probe" not in test
)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PreflightError(f"{field} must be a lowercase SHA-256")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise PreflightError("candidate commit must be a full lowercase Git SHA-1")
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PreflightError(f"{field} must be finite and nonnegative")
    return result


def _regular_file(
    path: pathlib.Path, *, ascii_required: bool = False,
    allow_symlink: bool = False,
) -> dict[str, Any]:
    if (path.is_symlink() and not allow_symlink) or not path.is_file():
        raise PreflightError(f"required regular file is absent: {path}")
    raw = path.read_bytes()
    record = {
        "path": str(path.absolute()),
        "bytes": len(raw),
        "sha256": base.sha256_bytes(raw),
    }
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise PreflightError(f"file is not ASCII: {path}") from error
        record["ascii"] = True
    return record


def _executable(path: pathlib.Path, family: str) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise PreflightError(f"{family} executable is invalid: {path}")
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, check=False,
    )
    version = completed.stdout + completed.stderr
    if completed.returncode != 0 or not version:
        raise PreflightError(f"{family} version command failed")
    first_line = version.splitlines()[0].decode("utf-8", errors="replace")
    lower = first_line.casefold()
    if family == "GNU" and not any(marker in lower for marker in ("gcc", "g++", "gnu")):
        raise PreflightError("configured GCC compiler is not GNU")
    if family == "Clang" and "clang" not in lower:
        raise PreflightError("configured Clang compiler is not Clang")
    return {
        **_regular_file(path),
        "family": family,
        "version_first_line": first_line,
        "version_sha256": base.sha256_bytes(version),
    }


def _versioned_tool(
    path: pathlib.Path, label: str, version_arguments: Sequence[str] = ("--version",),
) -> dict[str, Any]:
    lexical = path.absolute()
    if not lexical.is_file() or not os.access(lexical, os.X_OK):
        raise PreflightError(f"required tool is invalid: {label}")
    completed = subprocess.run(
        [str(lexical), *version_arguments], capture_output=True, check=False,
    )
    raw = completed.stdout + completed.stderr
    if completed.returncode != 0 or not raw:
        raise PreflightError(f"tool version command failed: {label}")
    return {
        **_regular_file(lexical, allow_symlink=True),
        "version_sha256": base.sha256_bytes(raw),
        "version_first_line": raw.splitlines()[0].decode("utf-8", errors="replace"),
    }


def _git(
    repository: pathlib.Path, *arguments: str,
    git_path: pathlib.Path | None = None,
) -> bytes:
    executable = str((git_path or pathlib.Path("git")).absolute()) \
        if git_path is not None else "git"
    completed = subprocess.run(
        [executable, *arguments], cwd=repository, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise PreflightError(f"Git read failed: {' '.join(arguments[:2])}")
    return completed.stdout


def snapshot_inputs(
    repository: pathlib.Path, *, runtime_path: pathlib.Path,
    python_path: pathlib.Path, gcc_path: pathlib.Path, clang_path: pathlib.Path,
    cmake_path: pathlib.Path | None = None,
    ctest_path: pathlib.Path | None = None,
    node_path: pathlib.Path | None = None,
    git_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    repository = repository.resolve()
    cmake_path = cmake_path or _discover("cmake")
    ctest_path = ctest_path or _discover("ctest")
    node_path = node_path or _discover("node")
    git_path = git_path or _discover("git")
    head = _git(repository, "rev-parse", "HEAD", git_path=git_path).decode().strip()
    _commit(head)
    if _git(
        repository, "status", "--porcelain=v1", "--untracked-files=no",
        git_path=git_path,
    ).strip():
        raise PreflightError("tracked worktree must be clean")
    sources = {}
    for relative in SOURCE_CLOSURE:
        path = repository / relative
        record = _regular_file(path, ascii_required=path.suffix in {".cpp", ".hpp", ".py", ".mjs"})
        committed = _git(
            repository, "show", f"{head}:{relative.as_posix()}",
            git_path=git_path,
        )
        if committed != path.read_bytes():
            raise PreflightError(f"preflight source differs from committed HEAD: {relative}")
        sources[relative.as_posix()] = record
    candidate = sources[(BOT_RELATIVE / "submission.cpp").as_posix()]
    rank4 = sources[RANK4_RELATIVE.as_posix()]
    candidate["bootstrap_zero"] = (
        b"bootstrap-zero-not-qualified" in
        (repository / BOT_RELATIVE / "submission.cpp").read_bytes()
    )
    if not 0 < candidate["bytes"] < SOURCE_LIMIT_EXCLUSIVE:
        raise PreflightError("candidate source is not strictly below 95,000 ASCII bytes")
    if rank4["sha256"] != RANK4_SHA256 or rank4["bytes"] != RANK4_BYTES:
        raise PreflightError("maintained Rank-4 source identity changed")
    expected_python = (repository / ".venv/bin/python").absolute()
    if python_path.absolute() != expected_python:
        raise PreflightError("Python interpreter is not this worktree's .venv/bin/python")
    return {
        "candidate_commit": head,
        "candidate": candidate,
        "rank4": rank4,
        "runtime": _regular_file(runtime_path, ascii_required=True),
        "sources": dict(sorted(sources.items())),
        "tools": {
            "python": _regular_file(python_path, allow_symlink=True),
            "gcc": _executable(gcc_path, "GNU"),
            "clang": _executable(clang_path, "Clang"),
            "cmake": _versioned_tool(cmake_path, "cmake"),
            "ctest": _versioned_tool(ctest_path, "ctest"),
            "node": _versioned_tool(node_path, "node"),
            "git": _versioned_tool(git_path, "git"),
        },
    }


def validate_input_snapshot(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise PreflightError("input snapshot is not an object")
    _commit(inputs.get("candidate_commit"))
    candidate = inputs.get("candidate")
    rank4 = inputs.get("rank4")
    runtime = inputs.get("runtime")
    if (not isinstance(candidate, dict) or candidate.get("ascii") is not True
            or candidate.get("bootstrap_zero") is not False
            or not isinstance(candidate.get("bytes"), int)
            or not 0 < candidate["bytes"] < SOURCE_LIMIT_EXCLUSIVE):
        raise PreflightError("candidate snapshot is not selected ASCII source below 95k")
    _sha(candidate.get("sha256"), "candidate SHA-256")
    if (not isinstance(rank4, dict) or rank4.get("ascii") is not True
            or rank4.get("sha256") != RANK4_SHA256
            or rank4.get("bytes") != RANK4_BYTES):
        raise PreflightError("input snapshot does not bind exact maintained Rank-4")
    if not isinstance(runtime, dict) or runtime.get("bytes", 0) <= 0:
        raise PreflightError("runtime snapshot is absent")
    _sha(runtime.get("sha256"), "runtime SHA-256")
    sources = inputs.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        path.as_posix() for path in SOURCE_CLOSURE
    }:
        raise PreflightError("source-closure snapshot is incomplete")
    if (sources[(BOT_RELATIVE / "submission.cpp").as_posix()] != candidate
            or sources[RANK4_RELATIVE.as_posix()] != rank4):
        raise PreflightError("source closure contradicts candidate/Rank-4 records")
    for relative, record in sources.items():
        if not isinstance(record, dict):
            raise PreflightError(f"source record is malformed: {relative}")
        _sha(record.get("sha256"), f"source {relative} SHA-256")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise PreflightError(f"source record has invalid size: {relative}")
    tools = inputs.get("tools")
    if not isinstance(tools, dict) or set(tools) != {
        "python", "gcc", "clang", "cmake", "ctest", "node", "git"
    }:
        raise PreflightError("tool snapshot is incomplete")
    for name, record in tools.items():
        if not isinstance(record, dict):
            raise PreflightError(f"tool record is malformed: {name}")
        _sha(record.get("sha256"), f"tool {name} SHA-256")
    if tools["gcc"].get("family") != "GNU" or tools["clang"].get("family") != "Clang":
        raise PreflightError("compiler family bindings are not exact")
    return dict(inputs)


def _test_regex(names: Sequence[str]) -> str:
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def create_plan(
    repository: pathlib.Path, *, build_root: pathlib.Path,
    python_path: pathlib.Path, gcc_path: pathlib.Path, clang_path: pathlib.Path,
    runtime_path: pathlib.Path,
    cmake_path: pathlib.Path | None = None,
    ctest_path: pathlib.Path | None = None,
    node_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    repository = repository.resolve()
    build_root = build_root.resolve()
    python_path = python_path.absolute()
    gcc_path = gcc_path.resolve()
    clang_path = clang_path.resolve()
    runtime_path = runtime_path.resolve()
    cmake = str((cmake_path or _discover("cmake")).absolute())
    ctest = str((ctest_path or _discover("ctest")).absolute())
    node = str((node_path or _discover("node")).absolute())
    panels = {}
    for name, compiler, family, sanitized in (
        ("gcc-release", gcc_path, "GNU", False),
        ("clang-release", clang_path, "Clang", False),
        ("clang-sanitized", clang_path, "Clang", True),
    ):
        build = build_root / name
        tests = SANITIZER_TESTS if sanitized else RELEASE_TESTS
        panels[name] = {
            "family": family,
            "sanitized": sanitized,
            "build": str(build),
            "configure": [
                cmake, "-S", str(repository), "-B", str(build),
                "-DCMAKE_BUILD_TYPE:STRING=" + ("Debug" if sanitized else "Release"),
                "-DCMAKE_CXX_COMPILER:FILEPATH=" + str(compiler),
                "-DPython3_EXECUTABLE:FILEPATH=" + str(python_path),
                "-DPAPERSOCCER_ENABLE_SANITIZERS:BOOL=" + ("ON" if sanitized else "OFF"),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
            ],
            "compile": [
                cmake, "--build", str(build), "--parallel", "2", "--target",
                *BUILD_TARGETS,
            ],
            "ctest": [
                ctest, "--test-dir", str(build), "--output-on-failure",
                "--timeout", "300", "-j", "1", "-R", _test_regex(tests),
            ],
            "expected_tests": list(tests),
        }
    clang_build = build_root / "clang-release"
    python = str(python_path)
    bot = repository / BOT_RELATIVE
    plan = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "repository": str(repository),
        "build_root": str(build_root),
        "python": str(python_path),
        "runtime": str(runtime_path),
        "panels": panels,
        "commands": {
            "python_discovery": [
                python, "-m", "unittest", "discover",
                "-s", str(repository / COMPACT_TEST_RELATIVE),
                "-p", COMPACT_TEST_PATTERN,
            ],
            "model_exporter_current": [
                python, str(bot / "export_model.py"), "--runtime", str(runtime_path),
                "--output", str(bot / "model.hpp"), "--check",
            ],
            "submission_exporter_current": [
                python, str(bot / "export_submission.py"), "--check",
            ],
            "submission_measure": [
                python, str(bot / "export_submission.py"), "--measure",
            ],
            "native_compact": [
                str(clang_build / "papersoccer_codingame_compact_value_bfm_submission_test")
            ],
            "frontier": [
                str(clang_build / "papersoccer_codingame_compact_value_bfm_comparison_gate")
            ],
            "protocol_end_to_end": [
                node, str(repository / "submissions/codingame/tools/protocol_smoke_test.mjs"),
                str(clang_build / "papersoccer_codingame_compact_value_bfm_submission"),
            ],
            "feature_parity": [
                python, str(bot / "feature_parity.py"), "--probe",
                str(clang_build / "papersoccer_codingame_compact_value_bfm_feature_probe"),
                "--states", str(PARITY_STATES),
            ],
        },
        "inference_probe": str(
            clang_build / "papersoccer_codingame_compact_value_bfm_inference_probe"
        ),
        "timing_probe": str(
            clang_build / "papersoccer_codingame_compact_value_bfm_timing_probe"
        ),
        "thresholds": {
            "source_bytes_exclusive": SOURCE_LIMIT_EXCLUSIVE,
            "parity_states_minimum": PARITY_STATES,
            "inference_error_maximum": PARITY_MAX_ERROR,
            "process_counts": list(PROCESS_COUNTS),
            "first_ms_exclusive": FIRST_LIMIT_MS,
            "later_ms_exclusive": LATER_LIMIT_MS,
        },
    }
    return base.seal(plan)


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    base.validate_seal(plan)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("namespace") != NAMESPACE:
        raise PreflightError("preflight plan schema/namespace is invalid")
    python = plan.get("python")
    repository = plan.get("repository")
    if not isinstance(python, str) or not isinstance(repository, str):
        raise PreflightError("preflight repository/Python binding is invalid")
    panels = plan.get("panels")
    if not isinstance(panels, dict) or set(panels) != {
        "gcc-release", "clang-release", "clang-sanitized"
    }:
        raise PreflightError("preflight panel roster is invalid")
    for name, panel in panels.items():
        typed = f"-DPython3_EXECUTABLE:FILEPATH={python}"
        if panel.get("configure", []).count(typed) != 1:
            raise PreflightError(f"{name} lacks the exact typed Python binding")
        if panel.get("sanitized") is not (name == "clang-sanitized"):
            raise PreflightError(f"{name} sanitizer classification is invalid")
        if name == "gcc-release" and panel.get("family") != "GNU":
            raise PreflightError("GCC Release panel is not GNU")
        if name != "gcc-release" and panel.get("family") != "Clang":
            raise PreflightError(f"{name} is not Clang")
    if plan.get("thresholds") != {
        "source_bytes_exclusive": SOURCE_LIMIT_EXCLUSIVE,
        "parity_states_minimum": PARITY_STATES,
        "inference_error_maximum": PARITY_MAX_ERROR,
        "process_counts": list(PROCESS_COUNTS),
        "first_ms_exclusive": FIRST_LIMIT_MS,
        "later_ms_exclusive": LATER_LIMIT_MS,
    }:
        raise PreflightError("preflight thresholds changed")
    commands = plan.get("commands")
    expected_discovery = [
        python, "-m", "unittest", "discover",
        "-s", str(pathlib.Path(repository) / COMPACT_TEST_RELATIVE),
        "-p", COMPACT_TEST_PATTERN,
    ]
    if (not isinstance(commands, Mapping)
            or commands.get("python_discovery") != expected_discovery):
        raise PreflightError("compact Python test discovery route changed")
    return dict(plan)


def validate_plan_input_bindings(
    plan: Mapping[str, Any], inputs: Mapping[str, Any]
) -> None:
    tools = inputs["tools"]
    if plan["python"] != tools["python"]["path"]:
        raise PreflightError("plan Python path differs from hashed interpreter")
    for name, panel in plan["panels"].items():
        compiler = tools["gcc" if name == "gcc-release" else "clang"]["path"]
        expected_compiler = f"-DCMAKE_CXX_COMPILER:FILEPATH={compiler}"
        if (panel["configure"][0] != tools["cmake"]["path"]
                or panel["compile"][0] != tools["cmake"]["path"]
                or panel["ctest"][0] != tools["ctest"]["path"]
                or expected_compiler not in panel["configure"]):
            raise PreflightError(f"{name} plan differs from hashed tool identities")
    if plan["commands"]["protocol_end_to_end"][0] != tools["node"]["path"]:
        raise PreflightError("protocol plan differs from hashed Node executable")


def _stream_record(value: bytes) -> dict[str, Any]:
    return {"bytes": len(value), "sha256": base.sha256_bytes(value)}


def run_command(
    label: str, argv: Sequence[str], *, cwd: pathlib.Path,
    required_markers: Sequence[str] = (), timeout_seconds: int = 3_600,
) -> tuple[dict[str, Any], bytes, bytes]:
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, capture_output=True, check=False,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        timed_out = True
    combined = stdout + b"\n" + stderr
    marker_status = {
        marker: marker.encode("utf-8") in combined for marker in required_markers
    }
    receipt = base.seal({
        "schema": COMMAND_SCHEMA,
        "label": label,
        "argv": list(argv),
        "cwd": str(cwd.resolve()),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_ns": time.monotonic_ns() - started,
        "stdout": _stream_record(stdout),
        "stderr": _stream_record(stderr),
        "required_markers": marker_status,
        "passed": returncode == 0 and not timed_out and all(marker_status.values()),
    })
    return receipt, stdout, stderr


def validate_command_receipt(
    receipt: Mapping[str, Any], *, label: str, argv: Sequence[str],
    required_markers: Sequence[str] = (),
) -> dict[str, Any]:
    base.validate_seal(receipt)
    if (receipt.get("schema") != COMMAND_SCHEMA or receipt.get("label") != label
            or receipt.get("argv") != list(argv) or receipt.get("passed") is not True
            or receipt.get("returncode") != 0 or receipt.get("timed_out") is not False
            or receipt.get("elapsed_ns", 0) <= 0):
        raise PreflightError(f"command receipt failed or changed: {label}")
    markers = receipt.get("required_markers")
    if markers != {marker: True for marker in required_markers}:
        raise PreflightError(f"command marker receipt changed: {label}")
    for stream in (receipt.get("stdout"), receipt.get("stderr")):
        if (not isinstance(stream, dict)
                or not isinstance(stream.get("bytes"), int) or stream["bytes"] < 0
                or SHA256_RE.fullmatch(str(stream.get("sha256"))) is None):
            raise PreflightError(f"command stream receipt is invalid: {label}")
    return dict(receipt)


def validate_cache_text(text: str, *, python_path: pathlib.Path, sanitized: bool) -> dict[str, Any]:
    expected_python = f"Python3_EXECUTABLE:FILEPATH={python_path.absolute()}"
    expected_sanitizer = (
        "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=ON" if sanitized
        else "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF"
    )
    if expected_python not in text.splitlines():
        raise PreflightError("CMakeCache Python FILEPATH does not equal worktree interpreter")
    if expected_sanitizer not in text.splitlines():
        raise PreflightError("CMakeCache sanitizer option is wrong")
    return {
        "python_entry": expected_python,
        "sanitizer_entry": expected_sanitizer,
        "python_equal": True,
        "checked_before_ctest": True,
    }


def validate_panel(
    panel: Mapping[str, Any], *, planned: Mapping[str, Any],
    python_path: pathlib.Path,
) -> dict[str, Any]:
    name = panel.get("name")
    if name not in {"gcc-release", "clang-release", "clang-sanitized"}:
        raise PreflightError("build panel name is invalid")
    if planned.get("build") != panel.get("build_path"):
        raise PreflightError(f"{name} build path changed")
    validate_command_receipt(
        panel.get("configure", {}), label=f"{name}:configure",
        argv=planned["configure"],
    )
    validate_command_receipt(
        panel.get("compile", {}), label=f"{name}:compile",
        argv=planned["compile"],
    )
    expected_tests = planned["expected_tests"]
    validate_command_receipt(
        panel.get("ctest", {}), label=f"{name}:ctest", argv=planned["ctest"],
        required_markers=[*expected_tests, "100% tests passed, 0 tests failed"],
    )
    cache = panel.get("cache")
    if (not isinstance(cache, dict) or cache.get("python_entry") !=
            f"Python3_EXECUTABLE:FILEPATH={python_path.absolute()}"
            or cache.get("python_equal") is not True
            or cache.get("checked_before_ctest") is not True):
        raise PreflightError(f"{name} cache was not checked before CTest")
    instrumentation = panel.get("instrumentation")
    expected_sanitized = name == "clang-sanitized"
    if instrumentation != {
        "address": expected_sanitized,
        "undefined_behavior": expected_sanitized,
    }:
        raise PreflightError(f"{name} sanitizer instrumentation is invalid")
    compiler = panel.get("compiler")
    expected_family = "GNU" if name == "gcc-release" else "Clang"
    if not isinstance(compiler, dict) or compiler.get("family") != expected_family:
        raise PreflightError(f"{name} compiler identity is invalid")
    _sha(compiler.get("sha256"), f"{name} compiler SHA-256")
    binaries = panel.get("binaries")
    if not isinstance(binaries, dict) or set(binaries) != set(BUILD_TARGETS):
        raise PreflightError(f"{name} binary roster is incomplete")
    for target, record in binaries.items():
        if (
            not isinstance(record, dict)
            or record.get("executable") is not True
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
        ):
            raise PreflightError(f"{name} binary record is invalid: {target}")
        _sha(record.get("sha256"), f"{name} binary {target} SHA-256")
    return dict(panel)


def validate_exporter_record(record: Mapping[str, Any], *, inputs: Mapping[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate")
    if (not isinstance(candidate, dict) or candidate.get("ascii") is not True
            or candidate.get("sha256") != inputs["candidate"]["sha256"]
            or candidate.get("bytes") != inputs["candidate"]["bytes"]
            or not 0 < candidate["bytes"] < SOURCE_LIMIT_EXCLUSIVE
            or record.get("runtime_sha256") != inputs["runtime"]["sha256"]):
        raise PreflightError("exporter freshness/source identity check failed")
    for name in (
        "model_exporter_current", "submission_exporter_current", "submission_measure"
    ):
        if record.get("commands", {}).get(name, {}).get("passed") is not True:
            raise PreflightError(f"exporter command did not pass: {name}")
    if record.get("fresh") is not True or record.get("under_95k") is not True:
        raise PreflightError("exporter freshness/size status failed")
    return dict(record)


def validate_parity_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(receipt, Mapping) or receipt.get("schema") != PARITY_SCHEMA
            or receipt.get("states") < PARITY_STATES
            or receipt.get("feature_states") < PARITY_STATES):
        raise PreflightError("feature/inference parity covers fewer than 4,096 states")
    error = _finite_nonnegative(receipt.get("maximum_absolute_error"), "parity error")
    if error >= PARITY_MAX_ERROR:
        raise PreflightError("C++ versus scalar float32 inference error is not below 2e-6")
    for name in ("features_sha256", "cpp_sha256", "scalar_sha256"):
        _sha(receipt.get(name), name)
    if receipt.get("all_finite") is not True:
        raise PreflightError("parity produced a nonfinite value")
    return dict(receipt)


def validate_timing_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != TIMING_SCHEMA:
        raise PreflightError("timing receipt schema is invalid")
    samples = receipt.get("samples")
    if not isinstance(samples, list):
        raise PreflightError("timing samples are absent")
    coverage: dict[tuple[int, int], set[int]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise PreflightError("timing sample is malformed")
        process_count = sample.get("process_count")
        color = sample.get("color")
        replica = sample.get("replica")
        if process_count not in PROCESS_COUNTS or color not in (0, 1):
            raise PreflightError("timing process/color identity is invalid")
        if (isinstance(replica, bool) or not isinstance(replica, int)
                or not 0 <= replica < process_count):
            raise PreflightError("timing replica identity is invalid")
        key = (process_count, color)
        coverage.setdefault(key, set())
        if replica in coverage[key]:
            raise PreflightError("timing sample repeats a replica")
        coverage[key].add(replica)
        first = _finite_nonnegative(sample.get("first_ms"), "timing first")
        later = _finite_nonnegative(sample.get("later_max_ms"), "timing later")
        if first >= FIRST_LIMIT_MS or later >= LATER_LIMIT_MS:
            raise PreflightError("timing threshold is not strictly below 900/180 ms")
    expected = {
        (count, color): set(range(count))
        for count in PROCESS_COUNTS for color in (0, 1)
    }
    if coverage != expected:
        raise PreflightError("timing lacks exact 1/2/10-process both-color coverage")
    if receipt.get("first_limit_exclusive_ms") != FIRST_LIMIT_MS or \
            receipt.get("later_limit_exclusive_ms") != LATER_LIMIT_MS:
        raise PreflightError("timing receipt thresholds changed")
    return dict(receipt)


def write_content_addressed(directory: pathlib.Path, payload: Mapping[str, Any]) -> pathlib.Path:
    artifact = base.seal(payload)
    raw = base.canonical_json_bytes(artifact)
    path = directory / f"{base.sha256_bytes(raw)}.json"
    base.atomic_write_once(path, raw)
    return path


def claim_path(output_root: pathlib.Path) -> pathlib.Path:
    return output_root / "claim.json"


def create_claim(
    output_root: pathlib.Path, *, plan: Mapping[str, Any],
    inputs: Mapping[str, Any], claimed_at_utc: str,
) -> dict[str, Any]:
    validate_plan(plan)
    validate_input_snapshot(inputs)
    validate_plan_input_bindings(plan, inputs)
    path = claim_path(output_root)
    if path.exists():
        raise PreflightError("preflight claim is already spent")
    return base.write_sealed(path, {
        "schema": CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "one_shot": True,
        "claim_precedes_build_test_or_parity_command": True,
        "claimed_at_utc": claimed_at_utc,
        "candidate_commit": _commit(inputs.get("candidate_commit")),
        "candidate_sha256": _sha(inputs.get("candidate", {}).get("sha256"), "candidate SHA-256"),
        "rank4_sha256": _sha(inputs.get("rank4", {}).get("sha256"), "Rank-4 SHA-256"),
        "plan_body_sha256": plan["body_sha256"],
        "inputs_sha256": base.sha256_bytes(base.canonical_json_bytes(dict(inputs))),
    })


def validate_preflight_receipt(
    receipt: Mapping[str, Any], *, claim: Mapping[str, Any],
    plan: Mapping[str, Any], inputs: Mapping[str, Any],
) -> dict[str, Any]:
    base.validate_seal(receipt)
    base.validate_seal(claim)
    validate_plan(plan)
    validate_input_snapshot(inputs)
    validate_plan_input_bindings(plan, inputs)
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("namespace") != NAMESPACE
            or receipt.get("status") != "passed"
            or receipt.get("claim") != claim
            or receipt.get("plan") != plan
            or receipt.get("claim_body_sha256") != claim["body_sha256"]
            or receipt.get("plan_body_sha256") != plan["body_sha256"]
            or receipt.get("inputs_before") != inputs
            or receipt.get("inputs_after") != inputs
            or receipt.get("protected_banks_accessed") != []
            or receipt.get("git_writes") != 0
            or receipt.get("uploads") != 0):
        raise PreflightError("preflight receipt top-level binding is invalid")
    panels = receipt.get("panels")
    if not isinstance(panels, dict) or set(panels) != set(plan["panels"]):
        raise PreflightError("preflight receipt panel roster is invalid")
    python = pathlib.Path(plan["python"])
    for name in plan["panels"]:
        validate_panel(panels[name], planned=plan["panels"][name], python_path=python)
    commands = receipt.get("commands")
    required_markers = {
        "python_discovery": ("OK",),
        "model_exporter_current": ("compact model header current",),
        "submission_exporter_current": ("compact submission current",),
        "submission_measure": ('"eligible": true',),
        "native_compact": ("compact_value_bfm submission tests passed",),
        "frontier": ("compact_value_bfm comparison smoke passed",),
        "protocol_end_to_end": ("Player 0 and Player 1 protocol smoke tests passed",),
        "feature_parity": ("compact feature parity passed states=4096",),
    }
    if not isinstance(commands, dict) or set(commands) != set(required_markers):
        raise PreflightError("preflight direct-command roster is invalid")
    for name, markers in required_markers.items():
        validate_command_receipt(
            commands[name], label=name, argv=plan["commands"][name],
            required_markers=markers,
        )
    validate_exporter_record(receipt.get("exporter", {}), inputs=inputs)
    validate_parity_receipt(receipt.get("parity", {}))
    validate_timing_receipt(receipt.get("timing", {}))
    checks = receipt.get("checks")
    required_checks = {
        "fresh_gcc_release", "fresh_clang_release", "fresh_clang_asan_ubsan",
        "cmakecache_python_before_ctest", "complete_python_discovery",
        "native_compact", "frontier", "protocol", "end_to_end",
        "exporter_fresh_ascii_under_95k", "feature_inference_parity_4096",
        "timing_1_2_10_both_colors",
    }
    if checks != {name: "passed" for name in sorted(required_checks)}:
        raise PreflightError("preflight check registry is incomplete")
    return dict(receipt)


def _import_file(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot import preflight helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unpack_weights(payload: bytes, count: int) -> list[int]:
    weights = []
    for index in range(count):
        bit = index * 3
        window = payload[bit // 8]
        if bit % 8 > 5 and bit // 8 + 1 < len(payload):
            window |= payload[bit // 8 + 1] << 8
        code = (window >> (bit % 8)) & 7
        value = code - 8 if code & 4 else code
        if value == -4:
            raise PreflightError("inference parity runtime contains code 100")
        weights.append(value)
    return weights


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _first_activation(value: float) -> float:
    return _f32(_f32(0.01) * value) if value < 0.0 else _f32(value * value)


def _second_activation(value: float) -> float:
    return _f32(_f32(0.01) * value) if value < 0.0 else value


def _fast_tanh(value: float) -> float:
    value = _f32(value)
    if value < _f32(-4.95):
        return -1.0
    if value > _f32(4.95):
        return 1.0
    square = _f32(value * value)
    numerator_inner = _f32(_f32(378.0) + square)
    numerator_inner = _f32(_f32(17325.0) + _f32(square * numerator_inner))
    numerator = _f32(value * _f32(_f32(135135.0) + _f32(square * numerator_inner)))
    denominator_inner = _f32(_f32(3150.0) + _f32(_f32(28.0) * square))
    denominator_inner = _f32(_f32(62370.0) + _f32(square * denominator_inner))
    denominator = _f32(_f32(135135.0) + _f32(square * denominator_inner))
    return _f32(numerator / denominator)


def scalar_inference(
    features: Sequence[int], *, weights: Sequence[int], hidden_one: int,
    hidden_two: int, scale_one: float, scale_two: float, scale_three: float,
) -> float:
    first = [0] * hidden_one
    for active in features:
        offset = active * hidden_one
        for hidden in range(hidden_one):
            first[hidden] += weights[offset + hidden]
    activated = [
        _first_activation(_f32(_f32(value) * scale_one)) for value in first
    ]
    offset_two = 6301 * hidden_one
    second = [0.0] * hidden_two
    for input_index in range(hidden_one):
        for hidden in range(hidden_two):
            scaled = _f32(activated[input_index] * scale_two)
            term = _f32(scaled * _f32(weights[
                offset_two + input_index * hidden_two + hidden
            ]))
            second[hidden] = _f32(second[hidden] + term)
    second = [_second_activation(value) for value in second]
    offset_three = offset_two + hidden_one * hidden_two
    output = 0.0
    for hidden in range(hidden_two):
        scaled = _f32(second[hidden] * scale_three)
        term = _f32(scaled * _f32(weights[offset_three + hidden]))
        output = _f32(output + term)
    return _fast_tanh(output)


def run_inference_parity(
    *, repository: pathlib.Path, runtime_path: pathlib.Path,
    probe_path: pathlib.Path, states: int = PARITY_STATES,
) -> dict[str, Any]:
    bot = repository / BOT_RELATIVE
    export_model = _import_file(bot / "export_model.py", "compact_preflight_export_model")
    feature_parity = _import_file(
        bot / "feature_parity.py", "compact_preflight_feature_parity"
    )
    runtime, packed, metadata = export_model.validate_runtime(runtime_path)
    _transcripts, feature_rows = feature_parity.fixtures(states)
    input_text = "\n".join(
        ",".join(str(value) for value in features) for features in feature_rows
    ) + "\n"
    completed = subprocess.run(
        [str(probe_path.resolve())], input=input_text.encode("ascii"),
        capture_output=True, check=False, timeout=300,
    )
    if completed.returncode != 0:
        raise PreflightError("inference probe failed")
    lines = completed.stdout.decode("ascii").splitlines()
    if len(lines) != states or any(re.fullmatch(r"[0-9a-f]{8}", line) is None for line in lines):
        raise PreflightError("inference probe returned malformed coverage")
    counts = metadata["counts"]
    weights = _unpack_weights(packed, counts["total"])
    scales = runtime["quantization"]["scales"]
    scalar_values = []
    cpp_values = []
    maximum_error = 0.0
    for line, features in zip(lines, feature_rows, strict=True):
        cpp = struct.unpack("<f", int(line, 16).to_bytes(4, "little"))[0]
        scalar = scalar_inference(
            features, weights=weights,
            hidden_one=metadata["hidden_one"], hidden_two=metadata["hidden_two"],
            scale_one=_f32(scales["w1"]), scale_two=_f32(scales["w2"]),
            scale_three=_f32(scales["w3"]),
        )
        if not math.isfinite(cpp) or not math.isfinite(scalar):
            raise PreflightError("inference parity produced nonfinite values")
        maximum_error = max(maximum_error, abs(cpp - scalar))
        cpp_values.append(struct.pack("<f", cpp))
        scalar_values.append(struct.pack("<f", scalar))
    feature_digest = hashlib.sha256()
    for features in feature_rows:
        feature_digest.update(len(features).to_bytes(2, "little"))
        for value in features:
            feature_digest.update(int(value).to_bytes(2, "little"))
    receipt = {
        "schema": PARITY_SCHEMA,
        "states": states,
        "feature_states": len(feature_rows),
        "features_sha256": feature_digest.hexdigest(),
        "cpp_sha256": base.sha256_bytes(b"".join(cpp_values)),
        "scalar_sha256": base.sha256_bytes(b"".join(scalar_values)),
        "maximum_absolute_error": maximum_error,
        "all_finite": True,
        "runtime_sha256": base.sha256_file(runtime_path),
        "probe_sha256": base.sha256_file(probe_path),
    }
    validate_parity_receipt(receipt)
    return receipt


TIMING_LINE = re.compile(
    r"(?P<label>player[01]_(?:first|later|later_initial)) "
    r"budget_ms=(?P<budget>[0-9]+) elapsed_us=(?P<elapsed>[0-9]+) "
)


def _one_timing(probe: pathlib.Path, color: int, process_count: int, replica: int) -> dict[str, Any]:
    completed = subprocess.run(
        [str(probe.resolve()), str(color)], capture_output=True, check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise PreflightError("timing probe process failed")
    text = completed.stdout.decode("ascii")
    matches = list(TIMING_LINE.finditer(text))
    expected_labels = {
        f"player{color}_first", f"player{color}_later",
        f"player{color}_later_initial",
    }
    if {match.group("label") for match in matches} != expected_labels:
        raise PreflightError("timing probe output is incomplete")
    first = next(
        int(match.group("elapsed")) / 1000.0 for match in matches
        if match.group("label").endswith("_first")
    )
    later = max(
        int(match.group("elapsed")) / 1000.0 for match in matches
        if "later" in match.group("label")
    )
    return {
        "process_count": process_count,
        "color": color,
        "replica": replica,
        "first_ms": first,
        "later_max_ms": later,
        "stdout_sha256": base.sha256_bytes(completed.stdout),
        "stderr_sha256": base.sha256_bytes(completed.stderr),
    }


def run_timing_suite(probe_path: pathlib.Path) -> dict[str, Any]:
    samples = []
    for process_count in PROCESS_COUNTS:
        for color in (0, 1):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=process_count
            ) as executor:
                futures = [
                    executor.submit(
                        _one_timing, probe_path, color, process_count, replica
                    )
                    for replica in range(process_count)
                ]
                samples.extend(future.result() for future in futures)
    receipt = {
        "schema": TIMING_SCHEMA,
        "probe_sha256": base.sha256_file(probe_path),
        "first_limit_exclusive_ms": FIRST_LIMIT_MS,
        "later_limit_exclusive_ms": LATER_LIMIT_MS,
        "samples": sorted(
            samples,
            key=lambda item: (item["process_count"], item["color"], item["replica"]),
        ),
    }
    validate_timing_receipt(receipt)
    return receipt


def _panel_instrumentation(build: pathlib.Path, sanitized: bool) -> dict[str, bool]:
    compile_commands = (build / "compile_commands.json").read_text(encoding="utf-8")
    link_paths = list((build / "CMakeFiles").glob(
        "papersoccer_codingame_compact_value_bfm_submission_test.dir/link.txt"
    ))
    link_text = link_paths[0].read_text(encoding="utf-8") if len(link_paths) == 1 else ""
    address = "-fsanitize=address,undefined" in compile_commands and \
        "-fsanitize=address,undefined" in link_text
    undefined = address and "-fno-sanitize-recover=all" in compile_commands
    if address is not sanitized or undefined is not sanitized:
        raise PreflightError("sanitizer instrumentation differs from panel plan")
    return {"address": address, "undefined_behavior": undefined}


def _run_panel(
    name: str, planned: Mapping[str, Any], *, repository: pathlib.Path,
    python_path: pathlib.Path, compiler: Mapping[str, Any],
) -> dict[str, Any]:
    build = pathlib.Path(planned["build"])
    if build.exists() or build.is_symlink():
        raise PreflightError(f"fresh build path already exists: {build}")
    configure, _, _ = run_command(
        f"{name}:configure", planned["configure"], cwd=repository,
    )
    validate_command_receipt(
        configure, label=f"{name}:configure", argv=planned["configure"]
    )
    compile_receipt, _, _ = run_command(
        f"{name}:compile", planned["compile"], cwd=repository,
    )
    validate_command_receipt(
        compile_receipt, label=f"{name}:compile", argv=planned["compile"]
    )
    cache_path = build / "CMakeCache.txt"
    cache_text = cache_path.read_text(encoding="utf-8")
    cache = validate_cache_text(
        cache_text, python_path=python_path, sanitized=planned["sanitized"]
    )
    markers = [*planned["expected_tests"], "100% tests passed, 0 tests failed"]
    ctest, _, _ = run_command(
        f"{name}:ctest", planned["ctest"], cwd=repository,
        required_markers=markers,
    )
    panel = {
        "name": name,
        "build_path": planned["build"],
        "compiler": dict(compiler),
        "configure": configure,
        "compile": compile_receipt,
        "cache": cache,
        "ctest": ctest,
        "instrumentation": _panel_instrumentation(build, planned["sanitized"]),
        "binaries": {},
    }
    for target in BUILD_TARGETS:
        binary = build / target
        if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
            raise PreflightError(f"built executable is missing or nonexecutable: {target}")
        panel["binaries"][target] = {
            **_regular_file(binary),
            "executable": True,
        }
    validate_panel(panel, planned=planned, python_path=python_path)
    return panel


DIRECT_MARKERS = {
    "python_discovery": ("OK",),
    "model_exporter_current": ("compact model header current",),
    "submission_exporter_current": ("compact submission current",),
    "submission_measure": ('"eligible": true',),
    "native_compact": ("compact_value_bfm submission tests passed",),
    "frontier": ("compact_value_bfm comparison smoke passed",),
    "protocol_end_to_end": ("Player 0 and Player 1 protocol smoke tests passed",),
    "feature_parity": ("compact feature parity passed states=4096",),
}


def _run_direct_commands(plan: Mapping[str, Any], repository: pathlib.Path) -> dict[str, Any]:
    receipts = {}
    for name, markers in DIRECT_MARKERS.items():
        timeout = 3_600 if name == "python_discovery" else 300
        receipt, _, _ = run_command(
            name, plan["commands"][name], cwd=repository,
            required_markers=markers, timeout_seconds=timeout,
        )
        validate_command_receipt(
            receipt, label=name, argv=plan["commands"][name],
            required_markers=markers,
        )
        receipts[name] = receipt
    return receipts


def _exporter_record(
    *, repository: pathlib.Path, runtime_path: pathlib.Path,
    inputs: Mapping[str, Any], commands: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_path = repository / BOT_RELATIVE / "submission.cpp"
    candidate = _regular_file(candidate_path, ascii_required=True)
    record = {
        "candidate": candidate,
        "runtime_sha256": base.sha256_file(runtime_path),
        "fresh": candidate == inputs["candidate"],
        "under_95k": candidate["bytes"] < SOURCE_LIMIT_EXCLUSIVE,
        "commands": {
            name: commands[name] for name in (
                "model_exporter_current", "submission_exporter_current",
                "submission_measure",
            )
        },
    }
    validate_exporter_record(record, inputs=inputs)
    return record


def _receipt_files(output_root: pathlib.Path) -> list[pathlib.Path]:
    directory = output_root / "receipts"
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"))
    if any(SHA256_RE.fullmatch(path.stem) is None for path in files):
        raise PreflightError("preflight receipt registry contains a foreign entry")
    return files


def _load_content_addressed(path: pathlib.Path) -> dict[str, Any]:
    value = base.load_sealed(path, RECEIPT_SCHEMA)
    if base.sha256_file(path) != path.stem:
        raise PreflightError("preflight receipt filename is not content-addressed")
    return value


def run_preflight(
    *, repository: pathlib.Path, output_root: pathlib.Path,
    build_root: pathlib.Path, runtime_path: pathlib.Path,
    python_path: pathlib.Path, gcc_path: pathlib.Path, clang_path: pathlib.Path,
    claimed_at_utc: str,
    cmake_path: pathlib.Path | None = None,
    ctest_path: pathlib.Path | None = None,
    node_path: pathlib.Path | None = None,
    git_path: pathlib.Path | None = None,
) -> pathlib.Path:
    repository = repository.resolve()
    output_root = output_root.resolve()
    build_root = build_root.resolve()
    inputs = snapshot_inputs(
        repository, runtime_path=runtime_path, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, cmake_path=cmake_path,
        ctest_path=ctest_path, node_path=node_path, git_path=git_path,
    )
    plan = create_plan(
        repository, build_root=build_root, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, runtime_path=runtime_path,
        cmake_path=cmake_path, ctest_path=ctest_path, node_path=node_path,
    )
    existing = _receipt_files(output_root)
    claim_file = claim_path(output_root)
    if existing:
        if len(existing) != 1 or not claim_file.exists():
            raise PreflightError("preflight receipt registry is not singular and claimed")
        claim = base.load_sealed(claim_file, CLAIM_SCHEMA)
        receipt = _load_content_addressed(existing[0])
        validate_preflight_receipt(
            receipt, claim=claim, plan=plan, inputs=inputs
        )
        return existing[0]
    if claim_file.exists():
        raise PreflightError("preflight claim is spent without a valid receipt")
    if build_root.exists() or build_root.is_symlink():
        raise PreflightError("fresh preflight build root already exists")
    claim = create_claim(
        output_root, plan=plan, inputs=inputs, claimed_at_utc=claimed_at_utc
    )
    panels = {
        name: _run_panel(
            name, planned, repository=repository,
            python_path=python_path,
            compiler=inputs["tools"]["gcc" if name == "gcc-release" else "clang"],
        )
        for name, planned in plan["panels"].items()
    }
    commands = _run_direct_commands(plan, repository)
    parity = run_inference_parity(
        repository=repository, runtime_path=runtime_path,
        probe_path=pathlib.Path(plan["inference_probe"]), states=PARITY_STATES,
    )
    timing = run_timing_suite(pathlib.Path(plan["timing_probe"]))
    inputs_after = snapshot_inputs(
        repository, runtime_path=runtime_path, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, cmake_path=cmake_path,
        ctest_path=ctest_path, node_path=node_path, git_path=git_path,
    )
    if inputs_after != inputs:
        raise PreflightError("source/tool inputs changed during preflight")
    checks = {
        name: "passed" for name in sorted({
            "fresh_gcc_release", "fresh_clang_release", "fresh_clang_asan_ubsan",
            "cmakecache_python_before_ctest", "complete_python_discovery",
            "native_compact", "frontier", "protocol", "end_to_end",
            "exporter_fresh_ascii_under_95k", "feature_inference_parity_4096",
            "timing_1_2_10_both_colors",
        })
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "status": "passed",
        "claim_body_sha256": claim["body_sha256"],
        "plan_body_sha256": plan["body_sha256"],
        "claim": claim,
        "plan": plan,
        "inputs_before": inputs,
        "inputs_after": inputs_after,
        "panels": panels,
        "commands": commands,
        "exporter": _exporter_record(
            repository=repository, runtime_path=runtime_path,
            inputs=inputs, commands=commands,
        ),
        "parity": parity,
        "timing": timing,
        "checks": checks,
        "protected_banks_accessed": [],
        "git_writes": 0,
        "uploads": 0,
    }
    path = write_content_addressed(output_root / "receipts", receipt)
    validate_preflight_receipt(
        base.load_sealed(path, RECEIPT_SCHEMA), claim=claim,
        plan=plan, inputs=inputs,
    )
    return path


def _discover(name: str) -> pathlib.Path:
    found = shutil.which(name)
    if found is None:
        raise PreflightError(f"required executable is absent: {name}")
    return pathlib.Path(found)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repository", type=pathlib.Path, default=pathlib.Path(__file__).parents[1])
    run.add_argument("--output-root", type=pathlib.Path, required=True)
    run.add_argument("--build-root", type=pathlib.Path, required=True)
    run.add_argument("--runtime", type=pathlib.Path, required=True)
    run.add_argument("--python", type=pathlib.Path)
    run.add_argument("--gcc", type=pathlib.Path)
    run.add_argument("--clang", type=pathlib.Path)
    run.add_argument("--cmake", type=pathlib.Path)
    run.add_argument("--ctest", type=pathlib.Path)
    run.add_argument("--node", type=pathlib.Path)
    run.add_argument("--git", type=pathlib.Path)
    run.add_argument("--claimed-at-utc", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--claim", type=pathlib.Path, required=True)
    validate.add_argument("--receipt", type=pathlib.Path, required=True)
    validate.add_argument("--plan", type=pathlib.Path, required=True)
    validate.add_argument("--inputs", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            repository = args.repository.resolve()
            path = run_preflight(
                repository=repository,
                output_root=args.output_root,
                build_root=args.build_root,
                runtime_path=args.runtime,
                python_path=(args.python or repository / ".venv/bin/python"),
                gcc_path=(args.gcc or _discover("g++")),
                clang_path=(args.clang or _discover("clang++")),
                cmake_path=(args.cmake or _discover("cmake")),
                ctest_path=(args.ctest or _discover("ctest")),
                node_path=(args.node or _discover("node")),
                git_path=(args.git or _discover("git")),
                claimed_at_utc=args.claimed_at_utc,
            )
            result = {"status": "passed", "receipt": str(path),
                      "sha256": base.sha256_file(path)}
        else:
            claim = base.load_sealed(args.claim, CLAIM_SCHEMA)
            receipt = _load_content_addressed(args.receipt)
            plan = base.load_sealed(args.plan, PLAN_SCHEMA)
            inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
            validate_preflight_receipt(
                receipt, claim=claim, plan=plan, inputs=inputs
            )
            result = {"status": "passed", "receipt": str(args.receipt)}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (PreflightError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact preflight failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
