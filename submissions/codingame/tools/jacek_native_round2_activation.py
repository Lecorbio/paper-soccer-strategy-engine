#!/usr/bin/env python3
"""Activate one immutable Jacek-native round-two selection.

The trained model remains pending (`chosen_seed: null`).  A selection sidecar
names the tested seed and runtime; this tool binds those immutable files in a
small deployment descriptor and is the only production header-generation path
for round two.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import fcntl
import hashlib
import json
import math
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


TOOL_DIRECTORY = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
REPOSITORY_TOOLS = ROOT / "tools"
for directory in (TOOL_DIRECTORY, REPOSITORY_TOOLS):
    if str(directory) in sys.path:
        sys.path.remove(str(directory))
sys.path[:0] = [str(TOOL_DIRECTORY), str(REPOSITORY_TOOLS)]

import generate_jacek_native_model_round2 as round2_exporter  # noqa: E402
import jacek_native_round2_selection as selection_tool  # noqa: E402


def _validate_module_origins() -> None:
    expected_modules = (
        (
            round2_exporter,
            TOOL_DIRECTORY / "generate_jacek_native_model_round2.py",
        ),
        (
            round2_exporter.round1_exporter,
            TOOL_DIRECTORY / "generate_jacek_native_model.py",
        ),
        (
            selection_tool,
            REPOSITORY_TOOLS / "jacek_native_round2_selection.py",
        ),
        (
            selection_tool.round1_exporter,
            TOOL_DIRECTORY / "generate_jacek_native_model.py",
        ),
        (
            selection_tool.round2_exporter,
            TOOL_DIRECTORY / "generate_jacek_native_model_round2.py",
        ),
    )
    for module, expected_path in expected_modules:
        expected_path = expected_path.resolve()
        actual_path = pathlib.Path(getattr(module, "__file__", "")).resolve()
        if actual_path != expected_path:
            raise ImportError(
                f"native activation module origin mismatch: {actual_path} != "
                f"{expected_path}"
            )


_validate_module_origins()


DEPLOYMENT_SCHEMA = "papersoccer.jacek-native-round2-deployment/v2"
DEFAULT_DEPLOYMENT = ROOT / "models/jacek_native_round2_deployment.json"
DEFAULT_HEADER = (
    ROOT / "submissions/codingame/bots/jacek_native_bfm/"
    "jacek_native_model.hpp"
)
SELECTION_SCHEMAS = {
    selection_tool.SELECTION_SCHEMA: ("promotion", True),
    selection_tool.EXPLORATORY_SELECTION_SCHEMA: ("exploratory", False),
}
EVIDENCE_DIRECTORY = pathlib.PurePosixPath(
    "models/jacek_native_round2_gate_evidence"
)
MODEL_ARTIFACT_DIRECTORY = pathlib.PurePosixPath("models")
UNTRAINED_DESCRIPTOR = pathlib.PurePosixPath(
    "models/jacek_native_untrained_seed.json"
)
UNTRAINED_RUNTIME = pathlib.PurePosixPath(
    "models/jacek_native_untrained_seed.runtime"
)
UNTRAINED_GENERATOR = pathlib.PurePosixPath(
    "tools/generate_jacek_native_seed.py"
)
CANONICAL_BASELINE_MODEL = pathlib.PurePosixPath(
    "models/jacek_native_bootstrap_model.json"
)
CANONICAL_BASELINE_RUNTIME = pathlib.PurePosixPath(
    "models/jacek_native_bootstrap_seed_20260813.runtime"
)
CANONICAL_BASELINE_SEED = 20260813
CANONICAL_BASELINE_MODEL_SHA256 = (
    "19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc"
)
CANONICAL_BASELINE_RUNTIME_SHA256 = (
    "877ee8d0afdb20cf3466bee4c09f654d33c6ac4ecc230b8022f570a31e60f93d"
)

# The prior v2 activation tools remain valid for descriptors they already
# created.  These are exact predecessors, not a general stale-tool bypass.
# Their semantics are strict subsets of this implementation and the rest of
# every descriptor (model, selection, runtime, evidence and ancestry) is still
# revalidated below.
COMPATIBLE_ACTIVATION_TOOL_SHA256 = frozenset({
    "3792d2fcc3b18ce9814949443bbf2c5941525828ffa3153c3e9c0e31c306bf6f",
    "2dd153dad38698e698e3cc89a7cbbacfc47c6ead86ac27ba2e8da03937c110f4",
})

# The last exporter emitted the immutable history-62 and round-three
# deployment descriptors before phase-weight profiles became explicit.  It is
# admitted only by this exact digest; its frozen target contract is validated
# by that archived descriptor's remaining content-addressed identities.
COMPATIBLE_ROUND2_EXPORTER_SHA256 = frozenset({
    "a4a24311c9d3abd839008971afc2e2d3389b84bb7365878c3e30c721573e7968",
})

# History 62 predates the phase-weighted round-two trainer now used by the
# exporter.  It may be reactivated only through this complete, content-addressed
# archive.  These identities deliberately name one model, selected seed,
# runtime, selection, old deployment, generated header and full gate window.
HISTORICAL_ACTIVE_MODEL_SHA256 = (
    "b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14"
)
HISTORICAL_ACTIVE_MODEL = pathlib.PurePosixPath(
    "models/jacek_native_history62_champion.json"
)
HISTORICAL_ACTIVE_SEED = 20260822
HISTORICAL_ACTIVE_RUNTIME = pathlib.PurePosixPath(
    "models/jacek_native_history62_champion.runtime"
)
HISTORICAL_ACTIVE_RUNTIME_SHA256 = (
    "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3"
)
HISTORICAL_ACTIVE_SELECTION_SHA256 = (
    "5597e4228850cd44aac4adc5f11e3d6533e5528e3e04c51700d2f04b2cbe2cef"
)
HISTORICAL_ACTIVE_SELECTION = pathlib.PurePosixPath(
    "models/jacek_native_history62_selection.json"
)
HISTORICAL_ACTIVE_DEPLOYMENT = pathlib.PurePosixPath(
    "models/jacek_native_history62_deployment.json"
)
HISTORICAL_ACTIVE_DEPLOYMENT_SHA256 = (
    "88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f"
)
HISTORICAL_ACTIVE_HEADER_SHA256 = (
    "3c1a8ef97f6dc14b9eed64679bd698939380db6fb72181d0b45d1aea74bd3458"
)
HISTORICAL_ACTIVE_GATE_EVIDENCE = frozenset({
    (
        "b44c1cec78c5c86421ea693af329662451ac9301665dd8fd3db0997721e3854a",
        "56791ba0eaed563c9774b0b8eac3e070f27aa842320367d8f2bbc4a943b450d1",
    ),
    (
        "2baee0a80f18b045357aef2686d58bacc89a1cf147e3699b1538be3e6c788ee2",
        "8eeadd7083536d26e87d0971422767cb4d7ee025af1b1df146439acd1b1886e8",
    ),
    (
        "28138eab2218e7a574dc4b7379fd19f8219a5efdbca5f107f507540f0761a98d",
        "4e956927891e51b6380d63b10e4bcd487b7201c595c21e821cf8f490c7e34091",
    ),
    (
        "41c8dc7c19f3ccacaf74de7e318fee129a4eedc7735ce5d0d2e940f2e3e9a983",
        "bbdc5056c6aeb1d3e51362a6303b384179a758c628d5d70a9e3cf74bb25cd62c",
    ),
    (
        "93ac04d0d020015d270738785b75225960d88280b4a4c62bd438d47f549db938",
        "a23a6390b3e32e6fe858bcae6a2f524cb6635c53cc602d972de0c4e8d01a7f1d",
    ),
    (
        "662122aba29156f3400c34f0b4dc4b25068c9f05b5027583f8eb8efdbfe73f19",
        "ac5b58aca9ae8635de89489e498da52e08adb9f1010072926b763a6d7740dd67",
    ),
})


class ActivationError(ValueError):
    """An immutable selection or deployment binding is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return selection_tool.canonical_json_bytes(value)


def _load_canonical(path: pathlib.Path, label: str) -> tuple[bytes, Any]:
    try:
        raw = path.read_bytes()
        value = selection_tool._strict_json(raw, label)
    except (OSError, selection_tool.SelectionError) as error:
        raise ActivationError(f"cannot validate {label}: {path}") from error
    return raw, value


def _contained(root: pathlib.Path, path: pathlib.Path, label: str) -> pathlib.Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ActivationError(f"{label} escapes the repository: {path}") from error
    return resolved


def _relative(root: pathlib.Path, path: pathlib.Path, label: str) -> str:
    return _contained(root, path, label).relative_to(root.resolve()).as_posix()


def _resolve_relative(root: pathlib.Path, value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or pathlib.PurePath(value).is_absolute():
        raise ActivationError(f"{label} must be a repository-relative path")
    return _contained(root, root / value, label)


def validate_selection(
    model_path: pathlib.Path,
    selection_path: pathlib.Path,
    *,
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    report_paths: Sequence[pathlib.Path],
    baseline_selection: pathlib.Path | None = None,
    baseline_deployment: pathlib.Path | None = None,
    repository_root: pathlib.Path = ROOT,
) -> dict[str, Any]:
    try:
        model_raw, model = selection_tool._load_round2_model(model_path)
    except selection_tool.SelectionError as error:
        raise ActivationError(
            f"cannot validate round-two model: {model_path}"
        ) from error
    sidecar_raw, sidecar = _load_canonical(selection_path, "selection sidecar")
    if not isinstance(model, Mapping) or not isinstance(sidecar, Mapping):
        raise ActivationError("model and selection roots must be objects")
    model_sha = _sha256(model_raw)
    if model_sha == HISTORICAL_ACTIVE_MODEL_SHA256:
        return _validate_historical_active_selection(
            repository_root=repository_root,
            model_path=model_path,
            model_raw=model_raw,
            model=model,
            selection_path=selection_path,
            sidecar_raw=sidecar_raw,
            sidecar=sidecar,
            baseline_model=baseline_model,
            baseline_seed=baseline_seed,
            baseline_runtime=baseline_runtime,
            baseline_selection=baseline_selection,
            baseline_deployment=baseline_deployment,
            report_paths=report_paths,
        )
    if sidecar.get("schema") not in SELECTION_SCHEMAS:
        raise ActivationError("selection sidecar schema is not deployable")
    try:
        expected = selection_tool.finalize_selection(
            model_path=model_path,
            baseline_model=baseline_model,
            baseline_seed=baseline_seed,
            baseline_runtime=baseline_runtime,
            baseline_selection=baseline_selection,
            baseline_deployment=baseline_deployment,
            report_paths=report_paths,
            output=None,
            exploratory=(
                sidecar.get("schema")
                == selection_tool.EXPLORATORY_SELECTION_SCHEMA
            ),
        )
    except (OSError, selection_tool.SelectionError) as error:
        raise ActivationError(
            "selection cannot be reproduced from frozen gate evidence"
        ) from error
    if sidecar_raw != _canonical_json_bytes(expected):
        raise ActivationError(
            "selection does not match deterministic frozen gate evidence"
        )

    selected = expected["selected"]
    runtime_raw = round2_exporter.render_runtime(
        model, model_sha, selected["seed"]
    ).encode()

    return {
        "decision": dict(expected["decision"]),
        "model": model,
        "model_bytes": model_raw,
        "model_sha256": model_sha,
        "runtime_bytes": runtime_raw,
        "selected": dict(selected),
        "selection": expected,
        "selection_bytes": sidecar_raw,
        "selection_sha256": _sha256(sidecar_raw),
        "historical_active": False,
        "historical_deployment_path": None,
    }


def _file_identity(
    root: pathlib.Path, path: pathlib.Path, label: str
) -> dict[str, Any]:
    resolved = _contained(root, path, label)
    raw = resolved.read_bytes()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(raw),
        "bytes": len(raw),
    }


def _validate_file_identity(
    root: pathlib.Path, identity: object, label: str
) -> pathlib.Path:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256", "bytes"}
        or not _valid_sha256(identity.get("sha256"))
        or isinstance(identity.get("bytes"), bool)
        or not isinstance(identity.get("bytes"), int)
        or identity["bytes"] <= 0
    ):
        raise ActivationError(f"{label} file identity is malformed")
    path = _resolve_relative(root, identity["path"], label)
    raw = path.read_bytes()
    if _sha256(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
        raise ActivationError(f"{label} bytes are stale")
    return path


def _is_canonical_bootstrap_baseline(
    root: pathlib.Path,
    model_path: pathlib.Path,
    seed: int,
    runtime_path: pathlib.Path,
) -> bool:
    root = root.resolve()
    return (
        _relative(root, model_path, "baseline model")
        == CANONICAL_BASELINE_MODEL.as_posix()
        and _relative(root, runtime_path, "baseline runtime")
        == CANONICAL_BASELINE_RUNTIME.as_posix()
        and seed == CANONICAL_BASELINE_SEED
        and _sha256(model_path.read_bytes()) == CANONICAL_BASELINE_MODEL_SHA256
        and _sha256(runtime_path.read_bytes())
        == CANONICAL_BASELINE_RUNTIME_SHA256
    )


def _validate_historical_active_gate_evidence(
    root: pathlib.Path,
    sidecar: Mapping[str, Any],
    report_paths: Sequence[pathlib.Path],
) -> None:
    reports = sidecar.get("reports")
    if not isinstance(reports, list):
        raise ActivationError("historical active selection has no gate evidence")
    declared: set[tuple[str, str]] = set()
    for report in reports:
        if not isinstance(report, Mapping):
            raise ActivationError("historical active gate declaration is malformed")
        report_sha = report.get("report_sha256")
        stdout_sha = report.get("stdout_sha256")
        if not _valid_sha256(report_sha) or not _valid_sha256(stdout_sha):
            raise ActivationError("historical active gate identity is malformed")
        declared.add((report_sha, stdout_sha))
    if (
        declared != HISTORICAL_ACTIVE_GATE_EVIDENCE
        or len(reports) != len(HISTORICAL_ACTIVE_GATE_EVIDENCE)
    ):
        raise ActivationError("historical active gate declarations are not frozen")

    try:
        canonical_reports = selection_tool._report_paths(report_paths)
    except selection_tool.SelectionError as error:
        raise ActivationError("historical active gate report set is invalid") from error
    observed: set[tuple[str, str]] = set()
    for report_path in canonical_reports:
        report_path = _require_evidence_path(root, report_path, "gate report")
        report_raw, report = _load_canonical(report_path, "gate report")
        report_sha = _sha256(report_raw)
        expected = dict(HISTORICAL_ACTIVE_GATE_EVIDENCE).get(report_sha)
        stdout = report.get("stdout") if isinstance(report, Mapping) else None
        stdout_name = stdout.get("path") if isinstance(stdout, Mapping) else None
        if (
            expected is None
            or not isinstance(stdout_name, str)
            or pathlib.PurePath(stdout_name).name != stdout_name
        ):
            raise ActivationError("historical active gate report is not frozen")
        stdout_path = _require_evidence_path(
            root, report_path.parent / stdout_name, "gate stdout"
        )
        stdout_raw = stdout_path.read_bytes()
        if (
            _sha256(stdout_raw) != expected
            or stdout.get("sha256") != expected
            or stdout.get("bytes") != len(stdout_raw)
        ):
            raise ActivationError("historical active gate stdout is not frozen")
        observed.add((report_sha, expected))
    if (
        observed != HISTORICAL_ACTIVE_GATE_EVIDENCE
        or len(canonical_reports) != len(HISTORICAL_ACTIVE_GATE_EVIDENCE)
    ):
        raise ActivationError("historical active gate evidence is incomplete")


def _validate_historical_active_selection(
    *,
    repository_root: pathlib.Path,
    model_path: pathlib.Path,
    model_raw: bytes,
    model: Mapping[str, Any],
    selection_path: pathlib.Path,
    sidecar_raw: bytes,
    sidecar: Mapping[str, Any],
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    baseline_selection: pathlib.Path | None,
    baseline_deployment: pathlib.Path | None,
    report_paths: Sequence[pathlib.Path],
) -> dict[str, Any]:
    root = repository_root.resolve()
    if (
        _sha256(model_raw) != HISTORICAL_ACTIVE_MODEL_SHA256
        or _sha256(sidecar_raw) != HISTORICAL_ACTIVE_SELECTION_SHA256
        or _relative(root, model_path, "historical active model")
        != HISTORICAL_ACTIVE_MODEL.as_posix()
        or _relative(root, selection_path, "historical active selection")
        != HISTORICAL_ACTIVE_SELECTION.as_posix()
        or sidecar.get("schema") != selection_tool.SELECTION_SCHEMA
    ):
        raise ActivationError("historical active model/selection identity is stale")
    if (
        baseline_selection is not None
        or baseline_deployment is not None
        or not _is_canonical_bootstrap_baseline(
            root, baseline_model, baseline_seed, baseline_runtime
        )
    ):
        raise ActivationError(
            "historical active model requires the frozen canonical bootstrap baseline"
        )

    runtime_path = _require_model_artifact_path(
        root,
        root / HISTORICAL_ACTIVE_RUNTIME,
        "historical active runtime",
        ".runtime",
    )
    deployment_path = _require_model_artifact_path(
        root,
        root / HISTORICAL_ACTIVE_DEPLOYMENT,
        "historical active deployment",
        ".json",
    )
    runtime_raw = runtime_path.read_bytes()
    if (
        _sha256(runtime_raw) != HISTORICAL_ACTIVE_RUNTIME_SHA256
        or _sha256(deployment_path.read_bytes())
        != HISTORICAL_ACTIVE_DEPLOYMENT_SHA256
    ):
        raise ActivationError("historical active archive bytes are stale")
    try:
        identity = selection_tool._baseline_identity(
            model_path,
            HISTORICAL_ACTIVE_SEED,
            runtime_path,
            selection_path,
            deployment_path,
        )
    except (OSError, selection_tool.SelectionError) as error:
        raise ActivationError(
            "historical active archive is not the exact retained deployment"
        ) from error
    selected = sidecar.get("selected")
    decision = sidecar.get("decision")
    if (
        not isinstance(selected, Mapping)
        or selected.get("seed") != HISTORICAL_ACTIVE_SEED
        or selected.get("checkpoint_sha256") != identity["checkpoint_sha256"]
        or selected.get("runtime_sha256") != HISTORICAL_ACTIVE_RUNTIME_SHA256
        or selected.get("model_sha256") != HISTORICAL_ACTIVE_MODEL_SHA256
        or decision != {
            "kind": "promotion",
            "promotion_eligible": True,
            "threshold_shortfalls": [],
        }
    ):
        raise ActivationError("historical active selection contradicts its archive")
    _validate_historical_active_gate_evidence(root, sidecar, report_paths)
    return {
        "decision": dict(decision),
        "model": model,
        "model_bytes": model_raw,
        "model_sha256": HISTORICAL_ACTIVE_MODEL_SHA256,
        "runtime_bytes": runtime_raw,
        "selected": dict(selected),
        "selection": dict(sidecar),
        "selection_bytes": sidecar_raw,
        "selection_sha256": HISTORICAL_ACTIVE_SELECTION_SHA256,
        "historical_active": True,
        "historical_deployment_path": deployment_path,
    }


def _require_evidence_path(
    root: pathlib.Path, path: pathlib.Path, label: str
) -> pathlib.Path:
    resolved = _contained(root, path, label)
    relative = pathlib.PurePosixPath(
        resolved.relative_to(root.resolve()).as_posix()
    )
    try:
        relative.relative_to(EVIDENCE_DIRECTORY)
    except ValueError as error:
        raise ActivationError(
            f"{label} must be installed under {EVIDENCE_DIRECTORY}"
        ) from error
    return resolved


def _require_model_artifact_path(
    root: pathlib.Path, path: pathlib.Path, label: str, suffix: str
) -> pathlib.Path:
    resolved = _contained(root, path, label)
    relative = pathlib.PurePosixPath(
        resolved.relative_to(root.resolve()).as_posix()
    )
    try:
        relative.relative_to(MODEL_ARTIFACT_DIRECTORY)
    except ValueError as error:
        raise ActivationError(
            f"{label} must be installed under {MODEL_ARTIFACT_DIRECTORY}"
        ) from error
    if resolved.suffix != suffix:
        raise ActivationError(f"{label} must use the {suffix} suffix")
    return resolved


def _gate_evidence_identities(
    root: pathlib.Path, report_paths: Sequence[pathlib.Path]
) -> tuple[list[dict[str, Any]], list[pathlib.Path]]:
    try:
        reports = selection_tool._report_paths(report_paths)
    except selection_tool.SelectionError as error:
        raise ActivationError("gate report set is invalid") from error
    entries: list[dict[str, Any]] = []
    for report_path in reports:
        report_path = _require_evidence_path(root, report_path, "gate report")
        _, report = _load_canonical(report_path, "gate report")
        candidate = report.get("candidate") if isinstance(report, Mapping) else None
        profile = report.get("profile") if isinstance(report, Mapping) else None
        stdout = report.get("stdout") if isinstance(report, Mapping) else None
        seed = candidate.get("seed") if isinstance(candidate, Mapping) else None
        profile_name = profile.get("name") if isinstance(profile, Mapping) else None
        stdout_name = stdout.get("path") if isinstance(stdout, Mapping) else None
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or profile_name not in selection_tool.PROFILES
            or not isinstance(stdout_name, str)
            or pathlib.PurePath(stdout_name).name != stdout_name
        ):
            raise ActivationError("gate evidence index is malformed")
        stdout_path = _require_evidence_path(
            root, report_path.parent / stdout_name, "gate stdout"
        )
        entries.append({
            "seed": seed,
            "profile": profile_name,
            "report": _file_identity(root, report_path, "gate report"),
            "stdout": _file_identity(root, stdout_path, "gate stdout"),
        })
    entries.sort(key=lambda entry: (entry["seed"], entry["profile"]))
    keys = [(entry["seed"], entry["profile"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ActivationError("gate evidence declarations are duplicated")
    return entries, [
        _resolve_relative(root, entry["report"]["path"], "gate report")
        for entry in entries
    ]


def _load_gate_evidence(
    root: pathlib.Path, value: object
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    if not isinstance(value, list) or not value:
        raise ActivationError("deployment gate evidence is missing")
    report_paths: list[pathlib.Path] = []
    all_paths: list[pathlib.Path] = []
    keys: list[tuple[int, str]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {
            "seed", "profile", "report", "stdout"
        }:
            raise ActivationError("deployment gate evidence is malformed")
        seed = entry.get("seed")
        profile = entry.get("profile")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or profile not in selection_tool.PROFILES
        ):
            raise ActivationError("deployment gate evidence key is malformed")
        report_path = _validate_file_identity(
            root, entry.get("report"), "gate report"
        )
        stdout_path = _validate_file_identity(
            root, entry.get("stdout"), "gate stdout"
        )
        _require_evidence_path(root, report_path, "gate report")
        _require_evidence_path(root, stdout_path, "gate stdout")
        _, report = _load_canonical(report_path, "gate report")
        expected_candidate = (
            report.get("candidate") if isinstance(report, Mapping) else None
        )
        expected_profile = (
            report.get("profile") if isinstance(report, Mapping) else None
        )
        expected_stdout = (
            report.get("stdout") if isinstance(report, Mapping) else None
        )
        if (
            not isinstance(report, Mapping)
            or not isinstance(expected_candidate, Mapping)
            or expected_candidate.get("seed") != seed
            or not isinstance(expected_profile, Mapping)
            or expected_profile.get("name") != profile
            or not isinstance(expected_stdout, Mapping)
            or report_path.parent / expected_stdout.get("path", "")
            != stdout_path
            or expected_stdout.get("sha256") != entry["stdout"]["sha256"]
            or expected_stdout.get("bytes") != entry["stdout"]["bytes"]
        ):
            raise ActivationError("gate report/stdout binding is stale")
        keys.append((seed, profile))
        report_paths.append(report_path)
        all_paths.extend((report_path, stdout_path))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ActivationError("gate evidence declarations are not canonical")
    return report_paths, all_paths


def _runtime_identity(raw: bytes, label: str) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ActivationError(f"{label} is not UTF-8") from error
    if (
        len(lines) != 7
        or lines[:3] != [
            "papersoccer.jacek-native-runtime-model/v1",
            round2_exporter.MODEL_SCHEMA,
            round2_exporter.FEATURE_SCHEMA,
        ]
        or not _valid_sha256(lines[3])
        or not _valid_sha256(lines[4])
    ):
        raise ActivationError(f"{label} runtime metadata is malformed")
    return {
        "artifact_sha256": _sha256(raw),
        "model_sha256": lines[3],
        "packed_sha256": lines[4],
    }


def _checkpoint_provenance(
    model: Mapping[str, Any], label: str
) -> tuple[str, list[dict[str, str]]]:
    generation = model.get("provenance", {}).get("generation")
    checkpoint = (
        generation.get("checkpoint_provenance")
        if isinstance(generation, Mapping) else None
    )
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"mode", "artifacts"}
    ):
        raise ActivationError(f"{label} checkpoint provenance is malformed")
    mode = checkpoint.get("mode")
    if mode not in {
        "untrained-seed-bootstrap/v1",
        "native-runtime-models/v1",
        "cumulative-native-runtime-models/v2",
    }:
        raise ActivationError(f"{label} checkpoint provenance mode is invalid")
    artifacts = checkpoint.get("artifacts")
    normalized: list[dict[str, str]] = []
    if not isinstance(artifacts, list) or not artifacts:
        raise ActivationError(f"{label} checkpoint provenance is empty")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {
                "artifact_sha256", "model_sha256", "packed_sha256"
            }
            or not all(_valid_sha256(artifact.get(field)) for field in artifact)
        ):
            raise ActivationError(f"{label} checkpoint identity is malformed")
        normalized.append(dict(artifact))
    key = lambda item: (
        item["artifact_sha256"], item["model_sha256"], item["packed_sha256"]
    )
    if normalized != sorted(normalized, key=key) or len({
        item["artifact_sha256"] for item in normalized
    }) != len(normalized):
        raise ActivationError(
            f"{label} checkpoint identities are not canonical/unique"
        )
    if generation.get("model_artifact_sha256") != [
        item["artifact_sha256"] for item in normalized
    ]:
        raise ActivationError(
            f"{label} checkpoint artifact summary is incomplete"
        )
    return mode, normalized


def _untrained_seed_identity(root: pathlib.Path) -> dict[str, str]:
    descriptor_path = root / UNTRAINED_DESCRIPTOR
    runtime_path = root / UNTRAINED_RUNTIME
    generator_path = root / UNTRAINED_GENERATOR
    descriptor_raw, descriptor = _load_canonical(
        descriptor_path, "untrained seed descriptor"
    )
    runtime_raw = runtime_path.read_bytes()
    identity = _runtime_identity(runtime_raw, "untrained seed")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("schema")
        != "papersoccer.jacek-native-untrained-seed/v1"
        or descriptor.get("model_schema") != round2_exporter.MODEL_SCHEMA
        or descriptor.get("feature_schema") != round2_exporter.FEATURE_SCHEMA
        or descriptor.get("training") is not None
        or descriptor.get("incumbent_dependencies") is not False
        or descriptor.get("protected_data") is not False
        or descriptor.get("generator_sha256")
        != _sha256(generator_path.read_bytes())
        or identity["model_sha256"] != _sha256(descriptor_raw)
    ):
        raise ActivationError("untrained seed descriptor/runtime is stale")
    lines = runtime_raw.decode("utf-8").splitlines()
    try:
        payload = base64.b64decode(lines[6], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ActivationError("untrained seed payload is invalid") from error
    weights = descriptor.get("weights")
    counts = weights.get("counts") if isinstance(weights, Mapping) else None
    scales = weights.get("scales") if isinstance(weights, Mapping) else None
    try:
        runtime_scales = [float(value) for value in lines[5].split()]
        descriptor_scales = [
            float(scales[name]) for name in ("w1", "w2", "w3")
        ]
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise ActivationError("untrained seed scales are invalid") from error
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"w1", "w2", "w3"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in counts.values()
        )
        or len(payload) != (sum(counts.values()) * 3 + 7) // 8
        or _sha256(payload) != identity["packed_sha256"]
        or weights.get("packed_sha256") != identity["packed_sha256"]
        or runtime_scales != descriptor_scales
        or len(runtime_scales) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in runtime_scales)
    ):
        raise ActivationError("untrained seed runtime payload is stale")
    return identity


def _checkpoint_model_identity(
    model_path: pathlib.Path,
    runtime_path: pathlib.Path,
    baseline_selection: pathlib.Path | None = None,
    baseline_deployment: pathlib.Path | None = None,
) -> dict[str, Any]:
    try:
        _, model, _, _ = selection_tool._baseline_model(model_path)
    except selection_tool.SelectionError as error:
        raise ActivationError(
            f"cannot validate checkpoint model: {model_path}"
        ) from error
    if not isinstance(model, Mapping):
        raise ActivationError("checkpoint model root is not an object")
    runtime_raw = runtime_path.read_bytes()
    checkpoints = model.get("checkpoints")
    retained = [
        checkpoint for checkpoint in checkpoints or []
        if isinstance(checkpoint, Mapping)
    ]
    matches: list[dict[str, Any]] = []
    for checkpoint in retained:
        seed = checkpoint.get("seed")
        try:
            identity = selection_tool._baseline_identity(
                model_path,
                seed,
                runtime_path,
                baseline_selection,
                baseline_deployment,
                require_deployed_seed=False,
            )
        except (OSError, selection_tool.SelectionError):
            continue
        if identity["runtime_sha256"] == _sha256(runtime_raw):
            matches.append(identity)
    if len(matches) != 1:
        raise ActivationError(
            "checkpoint runtime is not one exact unique retained model export"
        )
    selected = matches[0]
    return {
        "identity": _runtime_identity(runtime_raw, "checkpoint"),
        "model": model,
        "seed": selected["seed"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "exporter": selected["exporter"],
        "exporter_sha256": selected["exporter_sha256"],
    }


def _validate_checkpoint_ancestry(
    root: pathlib.Path,
    active_model_path: pathlib.Path,
    active_model: Mapping[str, Any],
    checkpoint_pairs: Sequence[tuple[pathlib.Path, pathlib.Path]],
    baseline_model: pathlib.Path | None = None,
    baseline_runtime: pathlib.Path | None = None,
    baseline_selection: pathlib.Path | None = None,
    baseline_deployment: pathlib.Path | None = None,
) -> dict[tuple[pathlib.Path, pathlib.Path], dict[str, Any]]:
    root = root.resolve()
    seed_identity = _untrained_seed_identity(root)
    active_mode, active_artifacts = _checkpoint_provenance(
        active_model, "active round-two model"
    )
    declarations: dict[str, dict[str, Any]] = {}
    pair_declarations: dict[
        tuple[pathlib.Path, pathlib.Path], dict[str, Any]
    ] = {}
    seen_pairs: set[tuple[pathlib.Path, pathlib.Path]] = set()
    seen_runtimes: set[pathlib.Path] = set()
    try:
        active_model_sha256 = _sha256(active_model_path.read_bytes())
    except OSError as error:
        raise ActivationError("active round-two model cannot be read") from error
    for index, (model_path, runtime_path) in enumerate(checkpoint_pairs):
        model_path = _contained(root, model_path, f"checkpoint {index} model")
        runtime_path = _contained(root, runtime_path, f"checkpoint {index} runtime")
        try:
            same_active_model = (
                model_path == active_model_path.resolve()
                or _sha256(model_path.read_bytes()) == active_model_sha256
            )
        except OSError as error:
            raise ActivationError(
                "checkpoint ancestry model cannot be read"
            ) from error
        if same_active_model:
            raise ActivationError(
                "checkpoint ancestry must not self-reference the active model"
            )
        pair = (model_path, runtime_path)
        if pair in seen_pairs or runtime_path in seen_runtimes:
            raise ActivationError("checkpoint file declarations are duplicated")
        seen_pairs.add(pair)
        seen_runtimes.add(runtime_path)
        historical_evidence = (
            (baseline_selection, baseline_deployment)
            if baseline_model is not None
            and model_path == baseline_model.resolve()
            else (None, None)
        )
        metadata = _checkpoint_model_identity(
            model_path, runtime_path, *historical_evidence
        )
        identity = metadata["identity"]
        artifact_sha = identity["artifact_sha256"]
        if artifact_sha in declarations:
            raise ActivationError("checkpoint runtime artifacts are duplicated")
        declaration = {
            **metadata,
            "model_path": model_path,
            "runtime_path": runtime_path,
        }
        declarations[artifact_sha] = declaration
        pair_declarations[pair] = declaration

    if active_mode == "untrained-seed-bootstrap/v1":
        if active_artifacts != [seed_identity] or declarations:
            raise ActivationError(
                "bootstrap ancestry must contain only the exact untrained seed"
            )
        return pair_declarations
    cumulative = active_mode == "cumulative-native-runtime-models/v2"
    if active_mode != "native-runtime-models/v1" and not cumulative:
        raise ActivationError("unsupported active checkpoint provenance mode")
    if not cumulative and seed_identity in active_artifacts:
        raise ActivationError("native ancestry must not mix the untrained seed")
    if cumulative and (
        seed_identity not in active_artifacts or len(active_artifacts) < 2
    ):
        raise ActivationError(
            "cumulative ancestry requires the seed root and a native checkpoint"
        )
    if not declarations:
        raise ActivationError(
            "native checkpoint ancestry requires file-backed declarations"
        )

    reachable: set[str] = set()
    visiting: set[str] = set()

    def validate_lineage(identity: dict[str, str]) -> None:
        if identity == seed_identity:
            return
        artifact_sha = identity["artifact_sha256"]
        declaration = declarations.get(artifact_sha)
        if declaration is None or declaration["identity"] != identity:
            raise ActivationError(
                "checkpoint artifacts do not match file-backed ancestry"
            )
        if artifact_sha in visiting:
            raise ActivationError("checkpoint ancestry contains a cycle")
        if artifact_sha in reachable:
            return
        visiting.add(artifact_sha)
        parent_model = declaration["model"]
        parent_path = declaration["model_path"]
        parent_mode, parents = _checkpoint_provenance(
            parent_model,
            f"checkpoint model {_relative(root, parent_path, 'checkpoint model')}",
        )
        if parent_mode == "untrained-seed-bootstrap/v1":
            if parents != [seed_identity]:
                raise ActivationError(
                    "checkpoint bootstrap ancestry is not the exact seed root"
                )
        elif parent_mode in {
            "native-runtime-models/v1", "cumulative-native-runtime-models/v2"
        }:
            parent_cumulative = (
                parent_mode == "cumulative-native-runtime-models/v2"
            )
            if seed_identity in parents and not parent_cumulative:
                raise ActivationError(
                    "native checkpoint ancestry illegally mixes the seed root"
                )
            if parent_cumulative and seed_identity not in parents:
                raise ActivationError(
                    "cumulative checkpoint ancestry omits its seed root"
                )
            if parent_cumulative and len(parents) < 2:
                raise ActivationError(
                    "cumulative checkpoint ancestry has no native checkpoint"
                )
            for parent in parents:
                validate_lineage(parent)
        else:
            raise ActivationError("unsupported checkpoint ancestry mode")
        visiting.remove(artifact_sha)
        reachable.add(artifact_sha)

    for artifact in active_artifacts:
        validate_lineage(artifact)
    if reachable != set(declarations):
        raise ActivationError(
            "checkpoint ancestry contains unused file declarations"
        )
    return pair_declarations


def _validate_deployment_baseline(
    root: pathlib.Path,
    active_model_path: pathlib.Path,
    active_model: Mapping[str, Any],
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    baseline_selection: pathlib.Path | None,
    baseline_deployment: pathlib.Path | None,
    declarations: Mapping[
        tuple[pathlib.Path, pathlib.Path], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    root = root.resolve()
    active_model_path = active_model_path.resolve()
    baseline_model = baseline_model.resolve()
    baseline_runtime = baseline_runtime.resolve()
    if baseline_model == active_model_path or _sha256(
        baseline_model.read_bytes()
    ) == _sha256(active_model_path.read_bytes()):
        raise ActivationError("active model cannot be its own deployment baseline")
    try:
        selected_identity = selection_tool._baseline_identity(
            baseline_model,
            baseline_seed,
            baseline_runtime,
            baseline_selection,
            baseline_deployment,
        )
    except (OSError, selection_tool.SelectionError) as error:
        raise ActivationError(
            "deployment baseline is not an exact retained checkpoint"
        ) from error

    if _is_canonical_bootstrap_baseline(
        root, baseline_model, baseline_seed, baseline_runtime
    ):
        if selected_identity.get("exporter") != "round1":
            raise ActivationError(
                "frozen canonical bootstrap uses the wrong exporter"
            )
        return selected_identity

    if selected_identity.get("exporter") != "round2":
        raise ActivationError(
            "deployment baseline is not the frozen canonical bootstrap or "
            "a retained native checkpoint"
        )
    declaration = declarations.get((baseline_model, baseline_runtime))
    if declaration is None:
        raise ActivationError(
            "native deployment baseline is missing from recursive checkpoint "
            "ancestry"
        )
    identity = declaration["identity"]
    if (
        declaration["model_path"] == active_model_path
        or declaration["seed"] != baseline_seed
        or declaration["checkpoint_sha256"]
        != selected_identity["checkpoint_sha256"]
        or declaration["exporter"] != selected_identity["exporter"]
        or declaration["exporter_sha256"]
        != selected_identity["exporter_sha256"]
        or identity["model_sha256"] != selected_identity["model_sha256"]
        or identity["artifact_sha256"] != selected_identity["runtime_sha256"]
        or identity["packed_sha256"] != selected_identity["packed_sha256"]
    ):
        raise ActivationError(
            "native deployment baseline disagrees with recursive checkpoint "
            "ancestry"
        )
    active_mode, active_artifacts = _checkpoint_provenance(
        active_model, "active round-two model"
    )
    if active_mode not in {
        "native-runtime-models/v1", "cumulative-native-runtime-models/v2"
    } or identity not in active_artifacts:
        raise ActivationError(
            "native deployment baseline is absent from active checkpoint ancestry"
        )
    return selected_identity


def _prevalidate_deployment_baseline(
    root: pathlib.Path,
    model_path: pathlib.Path,
    seed: int,
    runtime_path: pathlib.Path,
    selection_path: pathlib.Path | None = None,
    deployment_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    try:
        identity = selection_tool._baseline_identity(
            model_path, seed, runtime_path, selection_path, deployment_path
        )
    except (OSError, selection_tool.SelectionError) as error:
        raise ActivationError(
            "deployment baseline is not the frozen canonical bootstrap or an "
            "exact retained native checkpoint"
        ) from error
    if identity["exporter"] == "round1" and not _is_canonical_bootstrap_baseline(
        root, model_path, seed, runtime_path
    ):
        raise ActivationError(
            "deployment baseline is not the frozen canonical bootstrap"
        )
    return identity


def create_deployment(
    *,
    model_path: pathlib.Path,
    selection_path: pathlib.Path,
    runtime_path: pathlib.Path,
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    report_paths: Sequence[pathlib.Path],
    checkpoint_pairs: Sequence[tuple[pathlib.Path, pathlib.Path]],
    output: pathlib.Path,
    repository_root: pathlib.Path = ROOT,
    baseline_selection: pathlib.Path | None = None,
    baseline_deployment: pathlib.Path | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output = _contained(repository_root, output, "deployment output")
    if output.exists():
        raise ActivationError("refusing to overwrite immutable deployment")
    model_path = _require_model_artifact_path(
        repository_root, model_path, "round-two model", ".json"
    )
    selection_path = _require_model_artifact_path(
        repository_root, selection_path, "selection sidecar", ".json"
    )
    runtime_path = _require_model_artifact_path(
        repository_root, runtime_path, "selected runtime", ".runtime"
    )
    baseline_model = _require_model_artifact_path(
        repository_root, baseline_model, "baseline model", ".json"
    )
    baseline_runtime = _require_model_artifact_path(
        repository_root, baseline_runtime, "baseline runtime", ".runtime"
    )
    if (baseline_selection is None) != (baseline_deployment is None):
        raise ActivationError(
            "baseline selection and deployment descriptors must be supplied together"
        )
    if baseline_selection is not None and baseline_deployment is not None:
        baseline_selection = _require_model_artifact_path(
            repository_root, baseline_selection, "baseline selection", ".json"
        )
        baseline_deployment = _require_model_artifact_path(
            repository_root, baseline_deployment, "baseline deployment", ".json"
        )
    if (
        isinstance(baseline_seed, bool)
        or not isinstance(baseline_seed, int)
        or baseline_seed < 0
    ):
        raise ActivationError("baseline seed is invalid")
    _prevalidate_deployment_baseline(
        repository_root,
        baseline_model,
        baseline_seed,
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
    )
    evidence, canonical_reports = _gate_evidence_identities(
        repository_root, report_paths
    )
    validated = validate_selection(
        model_path,
        selection_path,
        baseline_model=baseline_model,
        baseline_seed=baseline_seed,
        baseline_runtime=baseline_runtime,
        baseline_selection=baseline_selection,
        baseline_deployment=baseline_deployment,
        report_paths=canonical_reports,
        repository_root=repository_root,
    )
    runtime_raw = _contained(
        repository_root, runtime_path, "selected runtime"
    ).read_bytes()
    if runtime_raw != validated["runtime_bytes"]:
        raise ActivationError("installed runtime is not the selected tested runtime")
    if validated["historical_active"] and _relative(
        repository_root, runtime_path, "historical active runtime"
    ) != HISTORICAL_ACTIVE_RUNTIME.as_posix():
        raise ActivationError("historical active runtime path is not frozen")
    normalized_pairs = [
        (
            _require_model_artifact_path(
                repository_root, model, "checkpoint model", ".json"
            ),
            _require_model_artifact_path(
                repository_root, runtime, "checkpoint runtime", ".runtime"
            ),
        )
        for model, runtime in checkpoint_pairs
    ]
    declarations = _validate_checkpoint_ancestry(
        repository_root,
        model_path,
        validated["model"],
        normalized_pairs,
        baseline_model,
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
    )
    baseline_identity = _validate_deployment_baseline(
        repository_root,
        model_path,
        validated["model"],
        baseline_model,
        baseline_seed,
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
        declarations,
    )
    checkpoints = []
    for model, runtime in normalized_pairs:
        checkpoints.append({
            "model": _file_identity(
                repository_root, model, "checkpoint model"
            ),
            "runtime": _file_identity(
                repository_root, runtime, "checkpoint runtime"
            ),
        })
    checkpoints.sort(key=lambda item: (
        item["model"]["path"], item["runtime"]["path"]
    ))
    if len({
        (item["model"]["path"], item["runtime"]["path"])
        for item in checkpoints
    }) != len(checkpoints):
        raise ActivationError("checkpoint provenance declarations are duplicated")
    descriptor = {
        "schema": DEPLOYMENT_SCHEMA,
        "decision": validated["decision"],
        "model": _file_identity(repository_root, model_path, "round-two model"),
        "selection": {
            **_file_identity(
                repository_root, selection_path, "selection sidecar"
            ),
            "payload_sha256": validated["selection"][
                "selection_payload_sha256"
            ],
        },
        "runtime": {
            **_file_identity(
                repository_root, runtime_path, "selected runtime"
            ),
            "packed_sha256": validated["selected"]["packed_sha256"],
        },
        "baseline": {
            "seed": baseline_seed,
            "checkpoint_sha256": baseline_identity["checkpoint_sha256"],
            "exporter": {
                "kind": baseline_identity["exporter"],
                "sha256": baseline_identity["exporter_sha256"],
            },
            "model": _file_identity(
                repository_root, baseline_model, "baseline model"
            ),
            "runtime": _file_identity(
                repository_root, baseline_runtime, "baseline runtime"
            ),
            "retained_evidence": (
                {
                    "selection": _file_identity(
                        repository_root,
                        baseline_selection,
                        "baseline selection",
                    ),
                    "deployment": _file_identity(
                        repository_root,
                        baseline_deployment,
                        "baseline deployment",
                    ),
                }
                if baseline_selection is not None
                and baseline_deployment is not None
                else None
            ),
        },
        "gate_evidence": evidence,
        "selected_seed": validated["selected"]["seed"],
        "native_checkpoint_provenance": checkpoints,
        "activation_tool_sha256": _sha256(pathlib.Path(__file__).read_bytes()),
        "selection_tool_sha256": _sha256(
            pathlib.Path(selection_tool.__file__).read_bytes()
        ),
        "round2_exporter_sha256": _sha256(
            pathlib.Path(round2_exporter.__file__).read_bytes()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(_canonical_json_bytes(descriptor))
    except FileExistsError as error:
        raise ActivationError("refusing to overwrite immutable deployment") from error
    return descriptor


def load_deployment(
    deployment_path: pathlib.Path,
    repository_root: pathlib.Path = ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    descriptor_raw, descriptor = _load_canonical(
        deployment_path, "round-two deployment"
    )
    expected_keys = {
        "schema", "decision", "model", "selection", "runtime", "baseline",
        "gate_evidence", "selected_seed", "native_checkpoint_provenance",
        "activation_tool_sha256", "selection_tool_sha256",
        "round2_exporter_sha256",
    }
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != expected_keys
        or descriptor.get("schema") != DEPLOYMENT_SCHEMA
    ):
        raise ActivationError("deployment descriptor schema is not frozen")
    activation_tool_sha256 = descriptor.get("activation_tool_sha256")
    if activation_tool_sha256 not in {
        _sha256(pathlib.Path(__file__).read_bytes()),
        *COMPATIBLE_ACTIVATION_TOOL_SHA256,
    }:
        raise ActivationError("deployment activation-tool SHA-256 is stale")
    round2_exporter_sha256 = descriptor.get("round2_exporter_sha256")
    if descriptor.get("selection_tool_sha256") != _sha256(
        pathlib.Path(selection_tool.__file__).read_bytes()
    ) or round2_exporter_sha256 not in {
        _sha256(pathlib.Path(round2_exporter.__file__).read_bytes()),
        *COMPATIBLE_ROUND2_EXPORTER_SHA256,
    }:
        raise ActivationError("deployment selector/exporter identity is stale")

    resolved: dict[str, pathlib.Path] = {}
    for name in ("model", "selection", "runtime"):
        identity = descriptor.get(name)
        extra = {"payload_sha256"} if name == "selection" else (
            {"packed_sha256"} if name == "runtime" else set()
        )
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"path", "sha256", "bytes"} | extra
            or not _valid_sha256(identity.get("sha256"))
            or any(not _valid_sha256(identity.get(field)) for field in extra)
            or isinstance(identity.get("bytes"), bool)
            or not isinstance(identity.get("bytes"), int)
            or identity["bytes"] <= 0
        ):
            raise ActivationError(f"deployment {name} identity is malformed")
        path = _resolve_relative(repository_root, identity["path"], name)
        raw = path.read_bytes()
        if _sha256(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
            raise ActivationError(f"deployment {name} bytes are stale")
        resolved[name] = path
    _require_model_artifact_path(
        repository_root, resolved["model"], "round-two model", ".json"
    )
    _require_model_artifact_path(
        repository_root, resolved["selection"], "selection sidecar", ".json"
    )
    _require_model_artifact_path(
        repository_root, resolved["runtime"], "selected runtime", ".runtime"
    )

    baseline = descriptor.get("baseline")
    if (
        not isinstance(baseline, Mapping)
        or set(baseline) != {
            "seed", "checkpoint_sha256", "exporter", "model", "runtime",
            "retained_evidence",
        }
        or isinstance(baseline.get("seed"), bool)
        or not isinstance(baseline.get("seed"), int)
        or baseline["seed"] < 0
        or not _valid_sha256(baseline.get("checkpoint_sha256"))
    ):
        raise ActivationError("deployment baseline identity is malformed")
    baseline_exporter = baseline.get("exporter")
    if (
        not isinstance(baseline_exporter, Mapping)
        or set(baseline_exporter) != {"kind", "sha256"}
        or baseline_exporter.get("kind") not in {"round1", "round2"}
        or not _valid_sha256(baseline_exporter.get("sha256"))
    ):
        raise ActivationError("deployment baseline exporter is malformed")
    baseline_model = _validate_file_identity(
        repository_root, baseline.get("model"), "baseline model"
    )
    baseline_runtime = _validate_file_identity(
        repository_root, baseline.get("runtime"), "baseline runtime"
    )
    retained_evidence = baseline.get("retained_evidence")
    baseline_selection: pathlib.Path | None = None
    baseline_deployment: pathlib.Path | None = None
    if retained_evidence is not None:
        if (
            not isinstance(retained_evidence, Mapping)
            or set(retained_evidence) != {"selection", "deployment"}
        ):
            raise ActivationError(
                "deployment baseline retained evidence is malformed"
            )
        baseline_selection = _validate_file_identity(
            repository_root,
            retained_evidence.get("selection"),
            "baseline selection",
        )
        baseline_deployment = _validate_file_identity(
            repository_root,
            retained_evidence.get("deployment"),
            "baseline deployment",
        )
        baseline_selection = _require_model_artifact_path(
            repository_root,
            baseline_selection,
            "baseline selection",
            ".json",
        )
        baseline_deployment = _require_model_artifact_path(
            repository_root,
            baseline_deployment,
            "baseline deployment",
            ".json",
        )
    _require_model_artifact_path(
        repository_root, baseline_model, "baseline model", ".json"
    )
    _require_model_artifact_path(
        repository_root, baseline_runtime, "baseline runtime", ".runtime"
    )
    _prevalidate_deployment_baseline(
        repository_root,
        baseline_model,
        baseline["seed"],
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
    )
    report_paths, evidence_paths = _load_gate_evidence(
        repository_root, descriptor.get("gate_evidence")
    )
    validated = validate_selection(
        resolved["model"],
        resolved["selection"],
        baseline_model=baseline_model,
        baseline_seed=baseline["seed"],
        baseline_runtime=baseline_runtime,
        baseline_selection=baseline_selection,
        baseline_deployment=baseline_deployment,
        report_paths=report_paths,
        repository_root=repository_root,
    )
    if validated["historical_active"] and _relative(
        repository_root, resolved["runtime"], "historical active runtime"
    ) != HISTORICAL_ACTIVE_RUNTIME.as_posix():
        raise ActivationError("historical active runtime path is not frozen")
    if (
        descriptor.get("decision") != validated["decision"]
        or descriptor.get("selected_seed") != validated["selected"]["seed"]
        or descriptor["selection"]["payload_sha256"]
        != validated["selection"]["selection_payload_sha256"]
        or descriptor["runtime"]["packed_sha256"]
        != validated["selected"]["packed_sha256"]
        or resolved["runtime"].read_bytes() != validated["runtime_bytes"]
    ):
        raise ActivationError("deployment contradicts the immutable selection")

    checkpoints = descriptor.get("native_checkpoint_provenance")
    if not isinstance(checkpoints, list):
        raise ActivationError("deployment checkpoint provenance is missing")
    normalized = []
    seen: set[tuple[str, str]] = set()
    for entry in checkpoints:
        if not isinstance(entry, Mapping) or set(entry) != {"model", "runtime"}:
            raise ActivationError("deployment checkpoint entry is malformed")
        checked = {}
        for name in ("model", "runtime"):
            identity = entry.get(name)
            if (
                not isinstance(identity, Mapping)
                or set(identity) != {"path", "sha256", "bytes"}
                or not _valid_sha256(identity.get("sha256"))
                or isinstance(identity.get("bytes"), bool)
                or not isinstance(identity.get("bytes"), int)
                or identity["bytes"] <= 0
            ):
                raise ActivationError("checkpoint file identity is malformed")
            path = _resolve_relative(
                repository_root, identity["path"], f"checkpoint {name}"
            )
            raw = path.read_bytes()
            if _sha256(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
                raise ActivationError("checkpoint file bytes are stale")
            checked[name] = path
        _require_model_artifact_path(
            repository_root, checked["model"], "checkpoint model", ".json"
        )
        _require_model_artifact_path(
            repository_root, checked["runtime"], "checkpoint runtime", ".runtime"
        )
        key = (entry["model"]["path"], entry["runtime"]["path"])
        if key in seen:
            raise ActivationError("checkpoint declarations are duplicated")
        seen.add(key)
        normalized.append((key, checked))
    if [item[0] for item in normalized] != sorted(item[0] for item in normalized):
        raise ActivationError("checkpoint declarations are not canonical")
    checkpoint_pairs = [
        (item[1]["model"], item[1]["runtime"]) for item in normalized
    ]
    declarations = _validate_checkpoint_ancestry(
        repository_root,
        resolved["model"],
        validated["model"],
        checkpoint_pairs,
        baseline_model,
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
    )
    baseline_identity = _validate_deployment_baseline(
        repository_root,
        resolved["model"],
        validated["model"],
        baseline_model,
        baseline["seed"],
        baseline_runtime,
        baseline_selection,
        baseline_deployment,
        declarations,
    )
    if (
        baseline["checkpoint_sha256"]
        != baseline_identity["checkpoint_sha256"]
        or baseline_exporter["kind"] != baseline_identity["exporter"]
        or baseline_exporter["sha256"]
        != baseline_identity["exporter_sha256"]
    ):
        raise ActivationError(
            "deployment baseline checkpoint/exporter binding is stale"
        )

    return {
        **validated,
        "checkpoint_paths": [item[1] for item in normalized],
        "baseline_model_path": baseline_model,
        "baseline_runtime_path": baseline_runtime,
        "baseline_selection_path": baseline_selection,
        "baseline_deployment_path": baseline_deployment,
        "evidence_paths": evidence_paths,
        "deployment": descriptor,
        "deployment_bytes": descriptor_raw,
        "deployment_path": deployment_path.resolve(),
        "deployment_sha256": _sha256(descriptor_raw),
        "model_path": resolved["model"],
        "runtime_path": resolved["runtime"],
        "selection_path": resolved["selection"],
    }


def _render_historical_active_header(
    validated: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if (
        validated.get("model_sha256") != HISTORICAL_ACTIVE_MODEL_SHA256
        or validated.get("selection_sha256")
        != HISTORICAL_ACTIVE_SELECTION_SHA256
        or validated.get("selected", {}).get("seed") != HISTORICAL_ACTIVE_SEED
        or validated.get("selected", {}).get("runtime_sha256")
        != HISTORICAL_ACTIVE_RUNTIME_SHA256
        or validated.get("runtime_bytes") is None
        or _sha256(validated["runtime_bytes"])
        != HISTORICAL_ACTIVE_RUNTIME_SHA256
    ):
        raise ActivationError("historical active render identity is stale")
    compatible = copy.deepcopy(validated["model"])
    provenance = compatible.get("provenance")
    if not isinstance(provenance, dict):
        raise ActivationError("historical active model provenance is malformed")
    round1 = round2_exporter.round1_exporter
    provenance["trainer_sha256"] = _sha256(
        pathlib.Path(round1.TRAINER).read_bytes()
    )
    provenance["corpus_validator_sha256"] = _sha256(
        pathlib.Path(round1.CORPUS_VALIDATOR).read_bytes()
    )
    try:
        header, metadata = round1.render(
            compatible,
            HISTORICAL_ACTIVE_MODEL_SHA256,
            HISTORICAL_ACTIVE_SEED,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ActivationError("historical active header cannot be rendered") from error
    if (
        _sha256(header.encode()) != HISTORICAL_ACTIVE_HEADER_SHA256
        or metadata.get("packed_sha256")
        != validated["selected"].get("packed_sha256")
    ):
        raise ActivationError("historical active generated header is not frozen")
    return header, metadata


def render_deployment(validated: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if validated.get("historical_active") is True:
        header, metadata = _render_historical_active_header(validated)
    else:
        header, metadata = round2_exporter.render(
            validated["model"],
            validated["model_sha256"],
            validated["selected"]["seed"],
        )
    return header, {
        **metadata,
        "deployment_sha256": validated["deployment_sha256"],
        "selection_sha256": validated["selection_sha256"],
        "selection_payload_sha256": validated["selection"][
            "selection_payload_sha256"
        ],
        "runtime_sha256": validated["selected"]["runtime_sha256"],
        "selection_kind": validated["decision"]["kind"],
        "promotion_eligible": validated["decision"]["promotion_eligible"],
    }


def _write_or_check(path: pathlib.Path, raw: bytes, check: bool, label: str) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != raw:
            raise ActivationError(f"{label} is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def install_deployment(
    candidate: pathlib.Path,
    expected_current_sha256: str,
    destination: pathlib.Path = DEFAULT_DEPLOYMENT,
    repository_root: pathlib.Path = ROOT,
) -> str:
    """Atomically CAS the one mutable activation pointer after validation."""
    repository_root = repository_root.resolve()
    destination = _contained(repository_root, destination, "deployment pointer")
    canonical = repository_root / "models/jacek_native_round2_deployment.json"
    if destination != canonical:
        raise ActivationError("deployment install destination is not canonical")
    if not _valid_sha256(expected_current_sha256):
        raise ActivationError("expected current deployment SHA-256 is invalid")
    candidate = _require_model_artifact_path(
        repository_root, candidate, "candidate deployment", ".json"
    )
    candidate_raw = load_deployment(candidate, repository_root)["deployment_bytes"]
    lock_path = repository_root / "build/.jacek-native-deployment-install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    temporary = destination.with_name(
        f".{destination.name}.{_sha256(candidate_raw)}.install"
    )
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current_raw = destination.read_bytes()
        except OSError as error:
            raise ActivationError(
                "current canonical deployment cannot be read"
            ) from error
        if _sha256(current_raw) != expected_current_sha256:
            raise ActivationError(
                "current canonical deployment changed before install"
            )
        with temporary.open("xb") as stream:
            stream.write(candidate_raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)
    return _sha256(candidate_raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="bind an immutable deployment")
    create.add_argument("--model", type=pathlib.Path, required=True)
    create.add_argument("--selection", type=pathlib.Path, required=True)
    create.add_argument("--runtime", type=pathlib.Path, required=True)
    create.add_argument("--baseline-model", type=pathlib.Path, required=True)
    create.add_argument("--baseline-seed", type=int, required=True)
    create.add_argument("--baseline-runtime", type=pathlib.Path, required=True)
    create.add_argument("--baseline-selection", type=pathlib.Path)
    create.add_argument("--baseline-deployment", type=pathlib.Path)
    create.add_argument(
        "--reports", nargs="+", type=pathlib.Path, required=True
    )
    create.add_argument(
        "--checkpoint", nargs=2, action="append", default=[],
        metavar=("MODEL", "RUNTIME"), type=pathlib.Path,
    )
    create.add_argument("--output", type=pathlib.Path, default=DEFAULT_DEPLOYMENT)

    validate = commands.add_parser("validate", help="validate a deployment")
    validate.add_argument(
        "--deployment", type=pathlib.Path, default=DEFAULT_DEPLOYMENT
    )

    generate = commands.add_parser(
        "generate", help="generate the exact selected production header"
    )
    generate.add_argument(
        "--deployment", type=pathlib.Path, default=DEFAULT_DEPLOYMENT
    )
    generate.add_argument("--output", type=pathlib.Path, default=DEFAULT_HEADER)
    generate.add_argument("--runtime-output", type=pathlib.Path)
    generate.add_argument("--check", action="store_true")
    generate.add_argument("--metadata", action="store_true")
    install = commands.add_parser(
        "install", help="atomically switch the canonical deployment pointer"
    )
    install.add_argument("--deployment", type=pathlib.Path, required=True)
    install.add_argument("--expected-current-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            descriptor = create_deployment(
                model_path=arguments.model,
                selection_path=arguments.selection,
                runtime_path=arguments.runtime,
                baseline_model=arguments.baseline_model,
                baseline_seed=arguments.baseline_seed,
                baseline_runtime=arguments.baseline_runtime,
                baseline_selection=arguments.baseline_selection,
                baseline_deployment=arguments.baseline_deployment,
                report_paths=arguments.reports,
                checkpoint_pairs=[tuple(pair) for pair in arguments.checkpoint],
                output=arguments.output,
            )
            print(json.dumps({
                "output": str(arguments.output),
                "selected_seed": descriptor["selected_seed"],
                "selection_kind": descriptor["decision"]["kind"],
            }, indent=2, sort_keys=True))
            return 0

        if arguments.command == "install":
            installed_sha = install_deployment(
                arguments.deployment, arguments.expected_current_sha256
            )
            print(json.dumps({
                "deployment": str(DEFAULT_DEPLOYMENT),
                "deployment_sha256": installed_sha,
            }, indent=2, sort_keys=True))
            return 0

        validated = load_deployment(arguments.deployment)
        if arguments.command == "validate":
            print(json.dumps({
                "deployment_sha256": validated["deployment_sha256"],
                "promotion_eligible": validated["decision"][
                    "promotion_eligible"
                ],
                "runtime_sha256": validated["selected"]["runtime_sha256"],
                "selected_seed": validated["selected"]["seed"],
                "selection_kind": validated["decision"]["kind"],
            }, indent=2, sort_keys=True))
            return 0

        header, metadata = render_deployment(validated)
        _write_or_check(
            arguments.output, header.encode(), arguments.check, "model header"
        )
        if arguments.runtime_output is not None:
            _write_or_check(
                arguments.runtime_output,
                validated["runtime_bytes"],
                arguments.check,
                "runtime checkpoint",
            )
        if arguments.metadata:
            print(json.dumps(metadata, indent=2, sort_keys=True))
        elif not arguments.check:
            print(f"wrote {arguments.output}")
        return 0
    except (
        ActivationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"round-two activation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
