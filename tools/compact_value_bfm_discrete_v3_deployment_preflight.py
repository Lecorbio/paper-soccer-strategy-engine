#!/usr/bin/env python3
"""Post-development preflight for an exact discrete-v3 deployment derivative.

The maintained preflight remains the authority for the frozen base source and
its full build/test closure.  This bridge deeply validates that receipt, then
tests a distinct cleanly committed source whose bytes are the exact seven-slot
deployment derivative.  It never rewrites the canonical exporter inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
BOT = REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
TEST_PATH = (
    REPOSITORY
    / "tests/codingame/test_compact_value_bfm_discrete_v3_deployment_preflight.py"
)
DIRECTORY = "discrete-v3-deployment-preflight"


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load deployment-preflight dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


maintained = _load(
    HERE / "compact_value_bfm_preflight.py", "compact_v3_deploy_maintained_preflight"
)
deployment = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment.py",
    "compact_v3_deploy_source_derivation",
)
base = maintained.base
CANDIDATE_RELATIVE = deployment.CANDIDATE_RELATIVE
MANIFEST_RELATIVE = deployment.MANIFEST_RELATIVE


class DeploymentPreflightError(ValueError):
    pass


NAMESPACE = maintained.NAMESPACE
PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-deployment-preflight-plan.v1"
CLAIM_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-deployment-preflight-claim.v1"
RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-deployment-preflight-receipt.v1"
)
REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-deployment-preflight-reference.v1"
)

HARNESS_SOURCES = {
    "submission_test.cpp": BOT / "submission_test.cpp",
    "timing_probe.cpp": BOT / "timing_probe.cpp",
    "inference_probe.cpp": BOT / "inference_probe.cpp",
}
SOURCE_INPUTS = {
    "rank4": REPOSITORY / "submissions/codingame/bots/rank_4/submission.cpp",
    "gate_source": BOT / "rank4_gate.cpp",
    "protocol_smoke": REPOSITORY
    / "submissions/codingame/tools/protocol_smoke_test.mjs",
    "export_model": BOT / "export_model.py",
    "feature_parity": BOT / "feature_parity.py",
    **{name.removesuffix(".cpp"): path for name, path in HARNESS_SOURCES.items()},
}
BINARY_NAMES = (
    "candidate", "candidate_gcc", "submission_test", "submission_test_gcc",
    "submission_test_sanitized",
    "timing_probe", "inference_probe", "rank4_gate", "configuration_probe",
)
COMMAND_NAMES = (
    "compile_candidate", "compile_candidate_gcc", "compile_submission_test",
    "compile_submission_test_gcc",
    "compile_submission_test_sanitized", "compile_timing_probe",
    "compile_inference_probe", "compile_rank4_gate",
    "compile_configuration_probe", "native_test", "gcc_native_test",
    "sanitized_test",
    "protocol", "rank4_gate_self_test", "configuration_probe",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DeploymentPreflightError(f"{label} is not UTC text")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeploymentPreflightError(f"{label} is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise DeploymentPreflightError(f"{label} is not UTC")
    return value


def _record(path: pathlib.Path, *, ascii_required: bool = False,
            executable: bool = False, allow_symlink: bool = False) -> dict[str, Any]:
    if (
        (path.is_symlink() and not allow_symlink)
        or not path.is_file()
        or (executable and not os.access(path, os.X_OK))
    ):
        raise DeploymentPreflightError(f"required regular file is absent: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise DeploymentPreflightError(f"required file is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **({"ascii": True} if ascii_required else {}),
        **({"executable": True} if executable else {}),
    }


def _verify_record(value: Any, label: str, *, ascii_required: bool = False,
                   executable: bool = False, allow_symlink: bool = False) -> pathlib.Path:
    if not isinstance(value, Mapping):
        raise DeploymentPreflightError(f"{label} record is absent")
    path = pathlib.Path(str(value.get("path", "")))
    if dict(value) != _record(
        path, ascii_required=ascii_required, executable=executable,
        allow_symlink=allow_symlink,
    ):
        raise DeploymentPreflightError(f"{label} changed")
    return path.resolve()


def _compiler_record(path: pathlib.Path, family: str) -> dict[str, Any]:
    record = _record(path, executable=True, allow_symlink=True)
    completed = subprocess.run(
        [record["path"], "--version"], capture_output=True, check=False,
        timeout=30,
    )
    try:
        text = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise DeploymentPreflightError("compiler version is not ASCII") from error
    lowered = text.lower()
    matches = (
        family == "GNU" and completed.returncode == 0
        and ("free software foundation" in lowered or "gcc" in lowered)
    ) or (
        family == "Clang" and completed.returncode == 0 and "clang" in lowered
    )
    if not matches:
        raise DeploymentPreflightError(f"compiler is not exact {family}")
    return {
        **record, "family": family,
        "version_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _reference(path: pathlib.Path, schema: str) -> dict[str, Any]:
    value = base.load_sealed(path, schema)
    return {"path": str(path.resolve()), "sha256": base.sha256_file(path)}


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise DeploymentPreflightError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file() or dict(value) != _reference(path, schema):
        raise DeploymentPreflightError(f"{label} reference changed")
    return path.resolve()


def _directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise DeploymentPreflightError(f"unsafe deployment-preflight directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise DeploymentPreflightError(f"deployment-preflight directory is absent: {path}")
    return path.resolve()


def _output(path: pathlib.Path) -> pathlib.Path:
    parent = _directory(path.parent, create=True)
    result = parent / path.name
    if result.is_symlink() or (result.exists() and not result.is_file()):
        raise DeploymentPreflightError(f"unsafe deployment-preflight output: {result}")
    return result


def _git_identity(repository: pathlib.Path, candidate_source: pathlib.Path) -> dict[str, Any]:
    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=repository, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise DeploymentPreflightError(
                f"deployment Git verification failed: {' '.join(arguments[:2])}"
            )
        return completed.stdout

    repository = repository.resolve()
    expected = repository / CANDIDATE_RELATIVE
    manifest_path = repository / MANIFEST_RELATIVE
    if (
        candidate_source.resolve() != expected.resolve()
        or candidate_source.is_symlink() or not candidate_source.is_file()
    ):
        raise DeploymentPreflightError(
            "deployment candidate is not the fixed distinct tracked source"
        )
    if candidate_source.resolve() == (repository / maintained.BOT_RELATIVE / "submission.cpp").resolve():
        raise DeploymentPreflightError("deployment candidate aliases canonical submission")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DeploymentPreflightError("fixed deployment manifest is absent or redirected")
    try:
        manifest = deployment.verify_manifest_file(manifest_path, candidate_source)
    except Exception as error:
        raise DeploymentPreflightError("fixed deployment manifest is invalid") from error
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    if not len(commit) == 40 or any(c not in "0123456789abcdef" for c in commit):
        raise DeploymentPreflightError("deployment commit is malformed")
    if git("status", "--porcelain=v1", "--untracked-files=no").strip():
        raise DeploymentPreflightError("deployment repository has tracked changes")
    relative = CANDIDATE_RELATIVE.as_posix()
    manifest_relative = MANIFEST_RELATIVE.as_posix()
    for label, path, tracked in (
        ("candidate", candidate_source, relative),
        ("manifest", manifest_path, manifest_relative),
    ):
        git("ls-files", "--error-unmatch", "--", tracked)
        if git("show", f"{commit}:{tracked}") != path.read_bytes():
            raise DeploymentPreflightError(
                f"deployment {label} differs from committed bytes"
            )
    return {
        "commit": commit, "tracked_clean": True, "source_path": relative,
        "manifest_path": manifest_relative,
        "manifest_sha256": base.sha256_file(manifest_path),
        "manifest_body_sha256": manifest["body_sha256"],
        "committed_bytes_equal": True,
    }


def _load_base_preflight(
    path: pathlib.Path, *, generated_source: pathlib.Path,
    runtime_path: pathlib.Path,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not path.name.endswith(".json"):
        raise DeploymentPreflightError("maintained base preflight is absent")
    if path.stem != base.sha256_file(path):
        raise DeploymentPreflightError("maintained base preflight is not content-addressed")
    receipt = base.load_sealed(path, maintained.RECEIPT_SCHEMA)
    try:
        maintained.validate_preflight_receipt(
            receipt, claim=receipt["claim"], plan=receipt["plan"],
            inputs=receipt["inputs_before"],
        )
    except Exception as error:
        raise DeploymentPreflightError("maintained base preflight is invalid") from error
    base_record = _record(generated_source, ascii_required=True)
    runtime_record = _record(runtime_path, ascii_required=True)
    maintained_candidate = receipt["inputs_before"]["candidate"]
    maintained_runtime = receipt["inputs_before"]["runtime"]
    maintained_tool = receipt["inputs_before"]["sources"].get(
        "tools/compact_value_bfm_preflight.py"
    )
    if (
        maintained_candidate.get("sha256") != base_record["sha256"]
        or maintained_candidate.get("bytes") != base_record["bytes"]
        or maintained_candidate.get("ascii") is not True
        or maintained_candidate.get("bootstrap_zero") is not False
        or maintained_runtime.get("sha256") != runtime_record["sha256"]
        or maintained_runtime.get("bytes") != runtime_record["bytes"]
        or not isinstance(maintained_tool, Mapping)
        or maintained_tool.get("sha256")
        != base.sha256_file(pathlib.Path(maintained.__file__).resolve())
    ):
        raise DeploymentPreflightError(
            "maintained base preflight differs from finalist base/runtime/tool"
        )
    return receipt


def _tool_closure() -> dict[str, Any]:
    return {
        "deployment_preflight": _record(pathlib.Path(__file__).resolve()),
        "deployment_preflight_tests": _record(TEST_PATH),
        "deployment_derivation": _record(pathlib.Path(deployment.__file__).resolve()),
        "deployment_derivation_tests": _record(
            REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_deployment.py"
        ),
        "maintained_preflight": _record(pathlib.Path(maintained.__file__).resolve()),
    }


def _configuration_probe_source() -> bytes:
    return b"""\
#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"
#include <iomanip>
#include <iostream>
int main() {
  namespace cv = compact_value_bfm;
  std::cout << std::setprecision(17)
            << "{\\\"candidate_c\\\":" << cv::kExploration
            << ",\\\"candidate_fpu\\\":" << cv::kFirstPlayUrgency
            << ",\\\"candidate_lambda\\\":" << cv::kFinalVisitWeight
            << ",\\\"candidate_actions\\\":" << cv::kMaximumActions
            << ",\\\"candidate_root_partial_paths\\\":" << cv::kRootPartialPaths
            << ",\\\"candidate_nonroot_partial_paths\\\":" << cv::kNonrootPartialPaths
            << ",\\\"candidate_nodes\\\":" << cv::kProductionTreeNodes
            << ",\\\"candidate_expansions\\\":" << cv::kMaximumExpansions
            << ",\\\"candidate_shuffle_seed\\\":"
            << cv::GeneratorConfig{}.shuffle_seed << "}\\n";
}
"""


def _snapshot(
    *, base_preflight_path: pathlib.Path, generated_source: pathlib.Path,
    candidate_source: pathlib.Path, runtime_path: pathlib.Path,
    repository: pathlib.Path, source_repository: pathlib.Path,
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
    python_path: pathlib.Path, gcc_path: pathlib.Path,
    clang_path: pathlib.Path, node_path: pathlib.Path,
) -> dict[str, Any]:
    if source_repository.resolve() != REPOSITORY.resolve():
        raise DeploymentPreflightError(
            "deployment source repository differs from the bound tool checkout"
        )
    _load_base_preflight(
        base_preflight_path, generated_source=generated_source,
        runtime_path=runtime_path,
    )
    git = _git_identity(repository, candidate_source)
    manifest_path = repository.resolve() / MANIFEST_RELATIVE
    generated = _record(generated_source, ascii_required=True)
    candidate = _record(candidate_source, ascii_required=True)
    runtime = _record(runtime_path, ascii_required=True)
    try:
        derivation = deployment.attest_derivation(
            generated_source.read_bytes(), candidate_source.read_bytes(),
            search_tuple=search_tuple, profile=profile, work=work,
        )
    except Exception as error:
        raise DeploymentPreflightError("deployment derivation is invalid") from error
    if derivation["base_source"] != {
        key: generated[key] for key in ("bytes", "sha256", "ascii")
    } or derivation["deployed_source"] != {
        key: candidate[key] for key in ("bytes", "sha256", "ascii")
    }:
        raise DeploymentPreflightError("deployment derivation/file records disagree")
    manifest_value = deployment.verify_manifest_file(manifest_path, candidate_source)
    if (
        manifest_value.get("base_source") != derivation["base_source"]
        or manifest_value.get("deployed_source") != derivation["deployed_source"]
        or manifest_value.get("configuration") != derivation["configuration"]
        or manifest_value.get("algorithm") != derivation["algorithm"]
    ):
        raise DeploymentPreflightError("deployment manifest differs from derivation")
    sources = {
        name: _record(path, ascii_required=True)
        for name, path in SOURCE_INPUTS.items()
    }
    if (
        sources["rank4"]["sha256"] != maintained.RANK4_SHA256
        or sources["rank4"]["bytes"] != maintained.RANK4_BYTES
    ):
        raise DeploymentPreflightError("maintained Rank-4 identity changed")
    return {
        "base_preflight": _reference(base_preflight_path, maintained.RECEIPT_SCHEMA),
        "generated_source": generated,
        "candidate": candidate,
        "runtime": runtime,
        "repository": str(repository.resolve()),
        "source_repository": str(source_repository.resolve()),
        "git": git,
        "manifest": _record(manifest_path, ascii_required=True),
        "manifest_body_sha256": manifest_value["body_sha256"],
        "derivation": derivation,
        "configuration": derivation["configuration"],
        "sources": sources,
        "tools": {
            "python": _record(python_path, executable=True, allow_symlink=True),
            "gcc": _compiler_record(gcc_path, "GNU"),
            "clang": _compiler_record(clang_path, "Clang"),
            "node": _record(node_path, executable=True, allow_symlink=True),
        },
        "tool_closure": _tool_closure(),
    }


def _commands(inputs: Mapping[str, Any], build: pathlib.Path) -> dict[str, list[str]]:
    clang = inputs["tools"]["clang"]["path"]
    gcc = inputs["tools"]["gcc"]["path"]
    node = inputs["tools"]["node"]["path"]
    harness = build / "harness"
    binaries = build / "binaries"
    common = [clang, "-std=c++20", "-O3", "-DNDEBUG"]
    gcc_common = [gcc, "-std=c++20", "-O3", "-DNDEBUG"]
    candidate = inputs["candidate"]["path"]
    gate = inputs["sources"]["gate_source"]["path"]
    macro = f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{candidate}"'
    return {
        "compile_candidate": [*common, candidate, "-o", str(binaries / "candidate")],
        "compile_candidate_gcc": [
            *gcc_common, candidate, "-o", str(binaries / "candidate_gcc")
        ],
        "compile_submission_test": [
            *common, str(harness / "submission_test.cpp"), "-o",
            str(binaries / "submission_test"),
        ],
        "compile_submission_test_gcc": [
            *gcc_common, str(harness / "submission_test.cpp"), "-o",
            str(binaries / "submission_test_gcc"),
        ],
        "compile_submission_test_sanitized": [
            clang, "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", str(harness / "submission_test.cpp"),
            "-o", str(binaries / "submission_test_sanitized"),
        ],
        "compile_timing_probe": [
            *common, str(harness / "timing_probe.cpp"), "-o",
            str(binaries / "timing_probe"),
        ],
        "compile_inference_probe": [
            *common, str(harness / "inference_probe.cpp"), "-o",
            str(binaries / "inference_probe"),
        ],
        "compile_rank4_gate": [
            *common, macro, gate, "-o", str(binaries / "rank4_gate"),
        ],
        "compile_configuration_probe": [
            *common, str(harness / "configuration_probe.cpp"), "-o",
            str(binaries / "configuration_probe"),
        ],
        "native_test": [str(binaries / "submission_test")],
        "gcc_native_test": [str(binaries / "submission_test_gcc")],
        "sanitized_test": [str(binaries / "submission_test_sanitized")],
        "protocol": [
            node, inputs["sources"]["protocol_smoke"]["path"],
            str(binaries / "candidate"),
        ],
        "rank4_gate_self_test": [str(binaries / "rank4_gate"), "--self-test"],
        "configuration_probe": [str(binaries / "configuration_probe")],
    }


def _plan_body(inputs: Mapping[str, Any], root: pathlib.Path,
               planned_at_utc: str) -> dict[str, Any]:
    build = root / "build"
    return {
        "schema": PLAN_SCHEMA, "namespace": NAMESPACE,
        "status": "deployment-preflight-planned-unclaimed",
        "planned_at_utc": _utc(planned_at_utc, "deployment-preflight plan time"),
        "inputs": dict(inputs),
        "paths": {
            "root": str(root), "build": str(build),
            "claim": str(root / "claim.json"),
            "receipts": str(root / "receipts"),
            "reference": str(root / "reference.json"),
        },
        "commands": _commands(inputs, build),
        "configuration_probe_source": {
            "bytes": len(_configuration_probe_source()),
            "sha256": hashlib.sha256(_configuration_probe_source()).hexdigest(),
        },
        "thresholds": {
            "source_bytes_exclusive": 95_000,
            "parity_states_minimum": maintained.PARITY_STATES,
            "inference_error_maximum": maintained.PARITY_MAX_ERROR,
            "process_counts": list(maintained.PROCESS_COUNTS),
            "first_ms_exclusive": maintained.FIRST_LIMIT_MS,
            "later_ms_exclusive": maintained.LATER_LIMIT_MS,
        },
        "policy": {
            "canonical_exporter_inputs_modified": False,
            "protected_banks_accessed": False,
            "git_writes": 0, "uploads": 0,
        },
    }


def prepare(
    *, output_root: pathlib.Path, base_preflight_path: pathlib.Path,
    generated_source: pathlib.Path, candidate_source: pathlib.Path,
    runtime_path: pathlib.Path, repository: pathlib.Path,
    source_repository: pathlib.Path, search_tuple: Sequence[Any], profile: Any,
    work: Mapping[str, Any], python_path: pathlib.Path, gcc_path: pathlib.Path,
    clang_path: pathlib.Path, node_path: pathlib.Path, planned_at_utc: str,
) -> pathlib.Path:
    root = _directory(output_root / DIRECTORY, create=True)
    plan_path = _output(root / "plan.json")
    inputs = _snapshot(
        base_preflight_path=base_preflight_path,
        generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
    )
    expected = base.seal(_plan_body(inputs, root, planned_at_utc))
    if plan_path.exists():
        if base.load_sealed(plan_path, PLAN_SCHEMA) != expected:
            raise DeploymentPreflightError("existing deployment-preflight plan changed")
    else:
        if any((root / name).exists() or (root / name).is_symlink() for name in (
            "claim.json", "receipts", "reference.json", "build"
        )):
            raise DeploymentPreflightError("execution output predates deployment plan")
        base.write_sealed(plan_path, {
            key: value for key, value in expected.items() if key != "body_sha256"
        })
    validate_plan(
        plan_path, base_preflight_path=base_preflight_path,
        generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
    )
    return plan_path


def validate_plan(
    path: pathlib.Path, *, base_preflight_path: pathlib.Path,
    generated_source: pathlib.Path, candidate_source: pathlib.Path,
    runtime_path: pathlib.Path, repository: pathlib.Path,
    source_repository: pathlib.Path, search_tuple: Sequence[Any], profile: Any,
    work: Mapping[str, Any], python_path: pathlib.Path, gcc_path: pathlib.Path,
    clang_path: pathlib.Path, node_path: pathlib.Path,
) -> dict[str, Any]:
    plan = base.load_sealed(path, PLAN_SCHEMA)
    root = path.parent.resolve()
    if path.is_symlink() or path.resolve() != root / "plan.json":
        raise DeploymentPreflightError("deployment-preflight plan route changed")
    inputs = _snapshot(
        base_preflight_path=base_preflight_path,
        generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
    )
    expected = base.seal(_plan_body(
        inputs, root, str(plan.get("planned_at_utc", ""))
    ))
    if plan != expected:
        raise DeploymentPreflightError("deployment-preflight plan changed")
    return plan


CommandRunner = Callable[..., tuple[Mapping[str, Any], bytes, bytes]]
ParityRunner = Callable[..., Mapping[str, Any]]
TimingRunner = Callable[[pathlib.Path], Mapping[str, Any]]


def _copy_harness(plan: Mapping[str, Any]) -> dict[str, Any]:
    build = pathlib.Path(plan["paths"]["build"])
    harness = _directory(build / "harness", create=True)
    _directory(build / "binaries", create=True)
    candidate = pathlib.Path(plan["inputs"]["candidate"]["path"])
    copies = {"submission.cpp": candidate, **HARNESS_SOURCES}
    for name, source in copies.items():
        destination = harness / name
        if destination.exists() or destination.is_symlink():
            raise DeploymentPreflightError("deployment harness was not pristine")
        shutil.copyfile(source, destination)
    probe = harness / "configuration_probe.cpp"
    probe.write_bytes(_configuration_probe_source())
    if any(path.is_symlink() or not path.is_file() for path in harness.iterdir()):
        raise DeploymentPreflightError("deployment harness contains an irregular entry")
    records = {
        path.name: _record(path, ascii_required=True)
        for path in sorted(harness.iterdir())
    }
    expected = {
        "submission.cpp": plan["inputs"]["candidate"],
        "submission_test.cpp": plan["inputs"]["sources"]["submission_test"],
        "timing_probe.cpp": plan["inputs"]["sources"]["timing_probe"],
        "inference_probe.cpp": plan["inputs"]["sources"]["inference_probe"],
    }
    for name, source in expected.items():
        if any(records[name][key] != source[key] for key in ("bytes", "sha256", "ascii")):
            raise DeploymentPreflightError(f"deployment harness copy changed: {name}")
    probe = records["configuration_probe.cpp"]
    if (
        probe["bytes"] != plan["configuration_probe_source"]["bytes"]
        or probe["sha256"] != plan["configuration_probe_source"]["sha256"]
    ):
        raise DeploymentPreflightError("configuration probe source changed")
    return records


def _run_commands(
    plan: Mapping[str, Any], *, command_runner: CommandRunner,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    markers = {
        "native_test": ("compact_value_bfm submission tests passed",),
        "gcc_native_test": ("compact_value_bfm submission tests passed",),
        "sanitized_test": ("compact_value_bfm submission tests passed",),
        "protocol": ("Player 0 and Player 1 protocol smoke tests passed",),
        "rank4_gate_self_test": ("compact_value_bfm Rank-4 gate self-test passed",),
    }
    receipts: dict[str, Any] = {}
    outputs: dict[str, bytes] = {}
    repository = pathlib.Path(plan["inputs"]["repository"])
    for name in COMMAND_NAMES:
        receipt, stdout, _stderr = command_runner(
            name, plan["commands"][name], cwd=repository,
            required_markers=markers.get(name, ()), timeout_seconds=3_600,
        )
        maintained.validate_command_receipt(
            receipt, label=name, argv=plan["commands"][name],
            required_markers=markers.get(name, ()),
        )
        receipts[name] = dict(receipt)
        outputs[name] = stdout
    return receipts, outputs


def _binary_records(plan: Mapping[str, Any]) -> dict[str, Any]:
    root = pathlib.Path(plan["paths"]["build"]) / "binaries"
    return {
        name: _record(root / name, executable=True) for name in BINARY_NAMES
    }


def _parse_configuration(raw: bytes, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentPreflightError("configuration probe output is malformed") from error
    projected = {
        key: expected[key] for key in (
            "candidate_c", "candidate_fpu", "candidate_lambda",
            "candidate_actions", "candidate_root_partial_paths",
            "candidate_nonroot_partial_paths", "candidate_nodes",
            "candidate_expansions", "candidate_shuffle_seed",
        )
    }
    if value != projected:
        raise DeploymentPreflightError("compiled deployment configuration differs")
    return value


def _receipt_files(root: pathlib.Path) -> list[pathlib.Path]:
    directory = root / "receipts"
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"))
    if any(path.stem != base.sha256_file(path) for path in files):
        raise DeploymentPreflightError("deployment receipt registry is not content-addressed")
    return files


def run_preflight(
    *, plan_path: pathlib.Path, base_preflight_path: pathlib.Path,
    generated_source: pathlib.Path, candidate_source: pathlib.Path,
    runtime_path: pathlib.Path, repository: pathlib.Path,
    source_repository: pathlib.Path, search_tuple: Sequence[Any], profile: Any,
    work: Mapping[str, Any], python_path: pathlib.Path, gcc_path: pathlib.Path,
    clang_path: pathlib.Path, node_path: pathlib.Path, claimed_at_utc: str,
    command_runner: CommandRunner = maintained.run_command,
    parity_runner: ParityRunner = maintained.run_inference_parity,
    timing_runner: TimingRunner = maintained.run_timing_suite,
) -> pathlib.Path:
    plan = validate_plan(
        plan_path, base_preflight_path=base_preflight_path,
        generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
    )
    root = plan_path.parent.resolve()
    reference_path = _output(root / "reference.json")
    if reference_path.exists():
        validate_reference(
            reference_path, generated_source=generated_source,
            candidate_source=candidate_source, runtime_path=runtime_path,
            repository=repository, source_repository=source_repository,
            search_tuple=search_tuple, profile=profile, work=work,
        )
        return reference_path
    claim_path = _output(root / "claim.json")
    receipts = _receipt_files(root)
    if claim_path.exists() or receipts:
        raise DeploymentPreflightError(
            "deployment-preflight claim is spent without a complete reference"
        )
    if pathlib.Path(plan["paths"]["build"]).exists():
        raise DeploymentPreflightError("deployment-preflight build is not fresh")
    claim = base.write_sealed(claim_path, {
        "schema": CLAIM_SCHEMA, "namespace": NAMESPACE,
        "status": "deployment-preflight-claimed-before-execution",
        "claimed_at_utc": _utc(claimed_at_utc, "deployment-preflight claim time"),
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate_sha256": plan["inputs"]["candidate"]["sha256"],
        "one_shot": True,
    })
    harness = _copy_harness(plan)
    commands, outputs = _run_commands(plan, command_runner=command_runner)
    binaries = _binary_records(plan)
    configuration = _parse_configuration(
        outputs["configuration_probe"], plan["inputs"]["configuration"]
    )
    parity = dict(parity_runner(
        repository=source_repository, runtime_path=runtime_path,
        probe_path=pathlib.Path(binaries["inference_probe"]["path"]),
        states=maintained.PARITY_STATES,
    ))
    timing = dict(timing_runner(pathlib.Path(binaries["timing_probe"]["path"])))
    maintained.validate_parity_receipt(parity)
    maintained.validate_timing_receipt(timing)
    inputs_after = _snapshot(
        base_preflight_path=base_preflight_path,
        generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
    )
    if inputs_after != plan["inputs"]:
        raise DeploymentPreflightError("deployment inputs changed during preflight")
    checks = {
        name: "passed" for name in sorted({
            "maintained_base_preflight_deep", "exact_seven_slot_derivation",
            "distinct_clean_committed_candidate", "clang_candidate",
            "gcc_clang_native_and_sanitized", "source_specific_rank4_gate",
            "protocol_both_colors", "compiled_configuration_exact",
            "feature_inference_parity_4096", "timing_1_2_10_both_colors",
            "inputs_unchanged", "canonical_exporter_inputs_untouched",
        })
    }
    body = {
        "schema": RECEIPT_SCHEMA, "namespace": NAMESPACE,
        "status": "deployment-preflight-passed",
        "completed_at_utc": utc_now(),
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "claim": _reference(claim_path, CLAIM_SCHEMA),
        "inputs": plan["inputs"],
        "base_preflight": plan["inputs"]["base_preflight"],
        "derivation": plan["inputs"]["derivation"],
        "commands": commands, "harness": harness, "binaries": binaries,
        "configuration_probe": configuration,
        "parity": parity, "timing": timing, "checks": checks,
        "protected_banks_accessed": [], "git_writes": 0, "uploads": 0,
    }
    sealed = base.seal(body)
    raw = base.canonical_json_bytes(sealed)
    receipt_path = _output(root / "receipts" / f"{base.sha256_bytes(raw)}.json")
    base.atomic_write_once(receipt_path, raw)
    base.write_sealed(reference_path, {
        "schema": REFERENCE_SCHEMA, "namespace": NAMESPACE,
        "status": "deployment-preflight-passed-awaiting-final",
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "receipt": _reference(receipt_path, RECEIPT_SCHEMA),
        "candidate": plan["inputs"]["candidate"],
        "generated_source": plan["inputs"]["generated_source"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "gate": binaries["rank4_gate"],
        "upload_authorized": False,
    })
    validate_reference(
        reference_path, generated_source=generated_source,
        candidate_source=candidate_source, runtime_path=runtime_path,
        repository=repository, source_repository=source_repository,
        search_tuple=search_tuple, profile=profile, work=work,
    )
    return reference_path


def validate_reference(
    path: pathlib.Path, *, generated_source: pathlib.Path,
    candidate_source: pathlib.Path, runtime_path: pathlib.Path,
    repository: pathlib.Path, source_repository: pathlib.Path,
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
) -> dict[str, Any]:
    reference = base.load_sealed(path, REFERENCE_SCHEMA)
    plan_path = _verify_reference(reference.get("plan"), PLAN_SCHEMA, "deployment plan")
    plan = base.load_sealed(plan_path, PLAN_SCHEMA)
    if (
        path.is_symlink() or path.resolve()
        != pathlib.Path(plan["paths"]["reference"]).resolve()
    ):
        raise DeploymentPreflightError("deployment-preflight reference route changed")
    tools = plan["inputs"]["tools"]
    validate_plan(
        plan_path, base_preflight_path=pathlib.Path(
            plan["inputs"]["base_preflight"]["path"]
        ), generated_source=generated_source, candidate_source=candidate_source,
        runtime_path=runtime_path, repository=repository,
        source_repository=source_repository, search_tuple=search_tuple,
        profile=profile, work=work,
        python_path=pathlib.Path(tools["python"]["path"]),
        gcc_path=pathlib.Path(tools["gcc"]["path"]),
        clang_path=pathlib.Path(tools["clang"]["path"]),
        node_path=pathlib.Path(tools["node"]["path"]),
    )
    receipt_path = _verify_reference(
        reference.get("receipt"), RECEIPT_SCHEMA, "deployment receipt"
    )
    receipt = base.load_sealed(receipt_path, RECEIPT_SCHEMA)
    claim_path = _verify_reference(receipt.get("claim"), CLAIM_SCHEMA, "deployment claim")
    claim = base.load_sealed(claim_path, CLAIM_SCHEMA)
    if (
        set(claim) != {
            "schema", "namespace", "status", "claimed_at_utc", "plan",
            "candidate_commit", "candidate_sha256", "one_shot", "body_sha256",
        }
        or claim.get("namespace") != NAMESPACE
        or claim.get("status") != "deployment-preflight-claimed-before-execution"
        or claim.get("candidate_commit") != plan["inputs"]["git"]["commit"]
        or claim.get("candidate_sha256") != plan["inputs"]["candidate"]["sha256"]
        or claim.get("one_shot") is not True
    ):
        raise DeploymentPreflightError("deployment claim contract changed")
    _utc(claim.get("claimed_at_utc"), "deployment-preflight claim time")
    if set(receipt) != {
        "schema", "namespace", "status", "completed_at_utc", "plan", "claim",
        "inputs", "base_preflight", "derivation", "commands", "harness",
        "binaries", "configuration_probe", "parity", "timing", "checks",
        "protected_banks_accessed", "git_writes", "uploads", "body_sha256",
    } or receipt.get("namespace") != NAMESPACE:
        raise DeploymentPreflightError("deployment receipt field contract changed")
    _utc(receipt.get("completed_at_utc"), "deployment-preflight completion time")
    if (
        receipt.get("status") != "deployment-preflight-passed"
        or receipt.get("plan") != _reference(plan_path, PLAN_SCHEMA)
        or receipt.get("inputs") != plan["inputs"]
        or receipt.get("base_preflight") != plan["inputs"]["base_preflight"]
        or receipt.get("derivation") != plan["inputs"]["derivation"]
        or claim.get("plan") != _reference(plan_path, PLAN_SCHEMA)
        or claim.get("candidate_sha256") != plan["inputs"]["candidate"]["sha256"]
        or receipt.get("protected_banks_accessed") != []
        or receipt.get("git_writes") != 0 or receipt.get("uploads") != 0
    ):
        raise DeploymentPreflightError("deployment receipt ancestry changed")
    harness = receipt.get("harness")
    expected_harness_names = {
        "submission.cpp", "submission_test.cpp", "timing_probe.cpp",
        "inference_probe.cpp", "configuration_probe.cpp",
    }
    if not isinstance(harness, Mapping) or set(harness) != expected_harness_names:
        raise DeploymentPreflightError("deployment harness roster changed")
    for name in expected_harness_names:
        _verify_record(harness[name], f"deployment harness {name}", ascii_required=True)
    source_bindings = {
        "submission.cpp": plan["inputs"]["candidate"],
        "submission_test.cpp": plan["inputs"]["sources"]["submission_test"],
        "timing_probe.cpp": plan["inputs"]["sources"]["timing_probe"],
        "inference_probe.cpp": plan["inputs"]["sources"]["inference_probe"],
    }
    for name, source in source_bindings.items():
        if any(harness[name][key] != source[key] for key in ("bytes", "sha256", "ascii")):
            raise DeploymentPreflightError(f"deployment harness binding changed: {name}")
    if (
        harness["configuration_probe.cpp"]["bytes"]
        != plan["configuration_probe_source"]["bytes"]
        or harness["configuration_probe.cpp"]["sha256"]
        != plan["configuration_probe_source"]["sha256"]
    ):
        raise DeploymentPreflightError("configuration probe harness changed")
    commands = receipt.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(COMMAND_NAMES):
        raise DeploymentPreflightError("deployment command roster changed")
    markers = {
        "native_test": ("compact_value_bfm submission tests passed",),
        "gcc_native_test": ("compact_value_bfm submission tests passed",),
        "sanitized_test": ("compact_value_bfm submission tests passed",),
        "protocol": ("Player 0 and Player 1 protocol smoke tests passed",),
        "rank4_gate_self_test": ("compact_value_bfm Rank-4 gate self-test passed",),
    }
    for name in COMMAND_NAMES:
        maintained.validate_command_receipt(
            commands[name], label=name, argv=plan["commands"][name],
            required_markers=markers.get(name, ()),
        )
        if commands[name].get("cwd") != plan["inputs"]["repository"]:
            raise DeploymentPreflightError(f"deployment command cwd changed: {name}")
    if (
        any(
            plan["commands"][name][0]
            != plan["inputs"]["tools"]["gcc"]["path"]
            for name in ("compile_candidate_gcc", "compile_submission_test_gcc")
        )
        or any(
            plan["commands"][name][0]
            != plan["inputs"]["tools"]["clang"]["path"]
            for name in COMMAND_NAMES if name.startswith("compile_")
            and not name.endswith("_gcc")
        )
    ):
        raise DeploymentPreflightError("deployment compiler command binding changed")
    binaries = receipt.get("binaries")
    if not isinstance(binaries, Mapping) or set(binaries) != set(BINARY_NAMES):
        raise DeploymentPreflightError("deployment binary roster changed")
    for name in BINARY_NAMES:
        _verify_record(binaries[name], f"deployment binary {name}", executable=True)
    config_binary = pathlib.Path(binaries["configuration_probe"]["path"])
    checked = subprocess.run(
        [str(config_binary)], capture_output=True, check=False, timeout=30
    )
    if checked.returncode != 0 or _parse_configuration(
        checked.stdout, plan["inputs"]["configuration"]
    ) != receipt.get("configuration_probe"):
        raise DeploymentPreflightError("deployment configuration recheck failed")
    parity = receipt.get("parity", {})
    timing = receipt.get("timing", {})
    maintained.validate_parity_receipt(parity)
    maintained.validate_timing_receipt(timing)
    if (
        parity.get("runtime_sha256") != plan["inputs"]["runtime"]["sha256"]
        or parity.get("probe_sha256") != binaries["inference_probe"]["sha256"]
        or timing.get("probe_sha256") != binaries["timing_probe"]["sha256"]
    ):
        raise DeploymentPreflightError("deployment parity/timing binary binding changed")
    expected_checks = {
        name: "passed" for name in sorted({
            "maintained_base_preflight_deep", "exact_seven_slot_derivation",
            "distinct_clean_committed_candidate", "clang_candidate",
            "gcc_clang_native_and_sanitized", "source_specific_rank4_gate",
            "protocol_both_colors", "compiled_configuration_exact",
            "feature_inference_parity_4096", "timing_1_2_10_both_colors",
            "inputs_unchanged", "canonical_exporter_inputs_untouched",
        })
    }
    expected_reference = base.seal({
        "schema": REFERENCE_SCHEMA, "namespace": NAMESPACE,
        "status": "deployment-preflight-passed-awaiting-final",
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "receipt": _reference(receipt_path, RECEIPT_SCHEMA),
        "candidate": plan["inputs"]["candidate"],
        "generated_source": plan["inputs"]["generated_source"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "gate": binaries["rank4_gate"], "upload_authorized": False,
    })
    if receipt.get("checks") != expected_checks or reference != expected_reference:
        raise DeploymentPreflightError("deployment receipt/reference content changed")
    return {
        "reference": reference, "reference_path": path.resolve(),
        "receipt": receipt, "receipt_path": receipt_path,
        "plan": plan, "plan_path": plan_path,
        "gate_path": pathlib.Path(binaries["rank4_gate"]["path"]).resolve(),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate": plan["inputs"]["candidate"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "timing": receipt["timing"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
        command.add_argument("--base-preflight", type=pathlib.Path, required=True)
        command.add_argument("--generated-source", type=pathlib.Path, required=True)
        command.add_argument("--candidate-source", type=pathlib.Path, required=True)
        command.add_argument("--runtime", type=pathlib.Path, required=True)
        command.add_argument("--repository", type=pathlib.Path, required=True)
        command.add_argument("--source-repository", type=pathlib.Path, default=REPOSITORY)
        command.add_argument("--tuple", nargs=3, required=True)
        command.add_argument("--profile", choices=tuple(deployment.PROFILE_ROSTER), required=True)
        command.add_argument("--python", type=pathlib.Path, required=True)
        command.add_argument("--gcc", type=pathlib.Path, required=True)
        command.add_argument("--clang", type=pathlib.Path, required=True)
        command.add_argument("--node", type=pathlib.Path, required=True)
    commands.choices["prepare"].add_argument("--planned-at-utc", default=utc_now())
    commands.choices["run"].add_argument("--plan", type=pathlib.Path, required=True)
    commands.choices["run"].add_argument("--claimed-at-utc", default=utc_now())
    args = parser.parse_args(argv)
    common = {
        "output_root": args.output_root,
        "base_preflight_path": args.base_preflight,
        "generated_source": args.generated_source,
        "candidate_source": args.candidate_source,
        "runtime_path": args.runtime, "repository": args.repository,
        "source_repository": args.source_repository,
        "search_tuple": args.tuple, "profile": args.profile,
        "work": deployment.PROFILE_ROSTER[args.profile],
        "python_path": args.python, "gcc_path": args.gcc,
        "clang_path": args.clang,
        "node_path": args.node,
    }
    try:
        if args.command == "prepare":
            result = prepare(**common, planned_at_utc=args.planned_at_utc)
        else:
            common.pop("output_root")
            result = run_preflight(
                **common, plan_path=args.plan, claimed_at_utc=args.claimed_at_utc
            )
        print(json.dumps({"path": str(result), "sha256": base.sha256_file(result)}, sort_keys=True))
        return 0
    except (DeploymentPreflightError, OSError, json.JSONDecodeError) as error:
        print(f"deployment preflight failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
