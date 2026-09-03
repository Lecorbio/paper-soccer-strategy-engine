#!/usr/bin/env python3
"""Govern a fixed-architecture Rank-4-teacher challenger campaign.

This tool freezes inputs, materializes deterministic pilot/full schedules,
maintains an append-only attempt ledger, prepares/validates two independent
strict final gates, and binds an existing upload plus clean exact-90 live
diagnostic.  It never trains a model, starts a game process, accesses a network,
schedules recurring work, uploads a submission, or replaces Rank 4.

Long-running producers are deliberately external.  Invoke the progress command
explicitly after durable producer checkpoints; there is no recurring automation
or implicit retry.  Logical game plans have ten shards, but at most eight
single-thread workers may run concurrently.  Timing/final gates remain subject
to their stricter one/four-worker contracts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
TEST_PATH = (
    REPOSITORY
    / "tests/codingame/test_compact_value_bfm_rank4_teacher_challenger.py"
)
DOC_PATH = REPOSITORY / "docs/compact-value-bfm-rank4-teacher-challenger.md"
RANK4_PATH = REPOSITORY / "submissions/codingame/bots/rank_4/submission.cpp"
EXPORT_MODEL_PATH = (
    REPOSITORY / "submissions/codingame/bots/compact_value_bfm/export_model.py"
)
PILOT_PIPELINE_PATH = HERE / "compact_value_bfm_pilot_pipeline.py"
TRAINER_PATH = HERE / "compact_value_bfm_train.py"
LIVE_WINDOW_PATH = (
    REPOSITORY
    / "submissions/codingame/bots/compact_value_bfm/live_window.py"
)


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load challenger dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py",
    "rank4_teacher_challenger_qualification",
)
openings = _load(
    HERE / "compact_value_bfm_openings.py",
    "rank4_teacher_challenger_openings",
)
export_model = _load(
    EXPORT_MODEL_PATH,
    "rank4_teacher_challenger_export_model",
)


class ChallengerError(ValueError):
    """A challenger artifact or transition violates the frozen campaign."""


NAMESPACE = "compact_value_bfm"
CAMPAIGN_ID = "compact-value-bfm-rank4-teacher-challenger-v1"
ARCHITECTURE = "6301-12-8-1"
DIMENSIONS = [6301, 12, 8, 1]
ATTEMPT_ZERO = 0
SOURCE_LIMIT = 95_000
MINIMUM_COMPARABLE_VALIDATION_GROUPS = 100
MINIMUM_COMPARABLE_VALIDATION_FRACTION = 0.80

ATTEMPT_ONE_INPUT_ALLOWLIST = {
    "initial_float_checkpoint": {
        "bytes": 303_534,
        "sha256": "0dbe279295bcfeb80392e10ff9b04f9bd5c0eb4e1ea45c974b6a92d266f37729",
    },
    "prior_runtime": {
        "bytes": 26_338,
        "sha256": "e4d814db15d5fc9ec99ebc16adc6be73290f0d4137f50206d3519902caf656b7",
    },
    "roots_tsv": {
        "bytes": 17_262,
        "sha256": "deab1da276fcf6eb3b837eadcf88794d60a22ebc8eb09443a56179e183eb7631",
    },
    "roots_manifest": {
        "bytes": 138_227,
        "sha256": "27e3029445c4bff3df0a97f1b159b7433e4ae252e99da3ebad0c12a35aa90926",
    },
}

ALLOWLIST = {
    "teacher_runtime": {
        "bytes": 4_864_000,
        "sha256": "f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75",
    },
    "teacher_manifest": {
        "bytes": 73_824,
        "sha256": "9222b43d46d4e8ae3e7211f429fa306688a2917c7795f26e7514d7a41314ac95",
    },
    "attempt_zero_runtime": {
        "bytes": 38_960,
        "sha256": "130c6ef1d2311a76c7a94fd144a805aa22477a32bced59a8079021e4293ea336",
    },
    "attempt_zero_source": {
        "bytes": 94_834,
        "sha256": "f5e67d699be19c3d495673c04ee2453570391c59e5f7be2a779198ce98b2d621",
    },
    "mixed_six_exclusion": {
        "bytes": 8_857,
        "sha256": "16bd3dae03897b807635e5ecabd2e2638bddf9d1c34408dc4494f36592cb1b8e",
        "body_sha256": "d78acf1e0bef5b81e4056b2d03a31a8374de32377e0691d3f3a089df7ef23d80",
    },
    "fresh_exclusion_receipt": {
        "bytes": 4_075,
        "sha256": "62a137bbe56b8db28b0d8b4c76e6ca79eb696443afa7274152dc233e2ed12c88",
        "body_sha256": "33ba81d7b6f80dfef407f4a0b40e6cebdc32708e636f69bd91ddd8f2cce99c2f",
    },
    "attempt_zero_recovery_plan": {
        "bytes": 29_840,
        "sha256": "cc895408aba607a0d75b01c0f0f6d2d5b26556d172af71fea13f96dac3146128",
        "body_sha256": "c2da279ab6d2deaa3b2e7b253886120da1ba5c12540dd2c0d900c2447a44ca95",
    },
}
TRAINING_BUNDLE_ALLOWLIST = {
    "bytes": 16_878,
    "sha256": "58e4d8ca648e52d2df31d27f13faa805d45e7c4e0c4b87f43b146118b768c742",
    "body_sha256": "56b9c1e6dd75e49298f677b73da6e8e4890f618c8d0f5daa252ec1248cbecd3a",
}

INPUT_SCHEMA = "papersoccer.compact-value-bfm.rank4-teacher-challenger-inputs.v1"
INPUT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-input-reference.v1"
)
PLAN_SCHEMA = "papersoccer.compact-value-bfm.rank4-teacher-challenger-plan.v1"
PHASE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-phase-plan.v1"
)
PHASE_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-phase-reference.v1"
)
LEDGER_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-attempt-event.v1"
)
DUAL_FINAL_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-dual-final-plan.v1"
)
DUAL_FINAL_AUTHORIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-authorization.v1"
)
DUAL_FINAL_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-dual-final-reference.v1"
)
FINAL_RESULT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-final-gate-result.v1"
)
DUAL_QUALIFICATION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-dual-qualified.v1"
)
PHASE_OUTCOME_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "phase-outcome-evidence.v1"
)
DEVELOPMENT_EXCLUSION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "development-fingerprints.v1"
)
ATTRIBUTION_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "attribution-evidence.v1"
)
FINAL_GATE_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "final-gate-evidence.v1"
)
ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "additional-upload-authorization.v1"
)
CAMPAIGN_COMPLETION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-completion.v1"
)
BUILD_MANIFEST_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-build-manifest.v1"
)
DYNAMIC_EXCLUSION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "fingerprint-exclusion.v1"
)
LIVE_FINGERPRINT_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.verified-live-canonical-fingerprints.v1"
)
RELEASE_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-release-evidence.v1"
)
RECOVERY_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-plan.v1"
)
RECOVERY_RESULT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-result.v1"
)
RECOVERY_FINALIST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-finalist.v1"
)
RECOVERY_FINALIST_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-"
    "finalist-reference.v1"
)
RECOVERY_JOURNAL_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-"
    "journal-event.v1"
)

PILOT_QUOTAS = {
    "student-p1-vs-rank4": 500,
    "student-p2-vs-rank4": 500,
    "student-selfplay": 500,
    "student-p1-vs-prior-incumbent": 125,
    "student-p2-vs-prior-incumbent": 125,
    "incumbent-p1-vs-rank4": 125,
    "incumbent-p2-vs-rank4": 125,
}
FULL_QUOTAS = {name: count * 5 for name, count in PILOT_QUOTAS.items()}
PHASE_QUOTAS = {"pilot": PILOT_QUOTAS, "full": FULL_QUOTAS}
PHASE_TOTALS = {name: sum(quotas.values()) for name, quotas in PHASE_QUOTAS.items()}
EXECUTABLE_ACTOR_MODES = {
    "incumbent-selfplay", "incumbent-p1-vs-rank4",
    "incumbent-p2-vs-rank4", "incumbent-p1-vs-jacek-nn",
    "incumbent-p2-vs-jacek-nn", "incumbent-p1-vs-runner-up",
    "incumbent-p2-vs-runner-up", "student-selfplay",
    "student-p1-vs-rank4", "student-p2-vs-rank4",
    "student-p1-vs-jacek-nn", "student-p2-vs-jacek-nn",
    "student-p1-vs-prior-incumbent", "student-p2-vs-prior-incumbent",
}
if PHASE_TOTALS != {"pilot": 2_000, "full": 10_000}:
    raise RuntimeError("challenger phase quotas are internally inconsistent")
if any(not set(quotas) <= EXECUTABLE_ACTOR_MODES for quotas in PHASE_QUOTAS.values()):
    raise RuntimeError("challenger phase uses a non-executable continuation actor mode")

RESOURCE_LIMITS = {
    "logical_game_shards": 10,
    "maximum_concurrent_game_workers": 8,
    "threads_per_worker": 1,
    "maximum_concurrent_training_seeds": 2,
    "uncontended_timing_workers": 1,
    "strict_final_workers": 4,
    "memory_limit_mib_per_worker": 4_096,
    "nice": 0,
    "thread_environment": {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    },
    "recurring_automation": False,
    "explicit_resume_required": True,
}

FINAL_THRESHOLDS = {
    "pairs": 500,
    "games": 1_000,
    "candidate_wins_min": 527,
    "candidate_color_wins_min": 260,
    "failures": 0,
    "independent_gates": 2,
    "candidate_must_be_unchanged": True,
}

LEDGER_EVENTS = {
    "attempt-opened",
    "attempt-zero-result-recorded",
    "phase-plan-materialized",
    "progress-recorded",
    "attempt-outcome-recorded",
    "dual-final-authorized",
    "dual-final-prepared",
    "final-gate-recorded",
    "dual-final-qualified",
    "upload-attested",
    "live-window-recorded",
    "additional-upload-authorized",
    "campaign-complete",
}
TERMINAL_EVENTS = {"campaign-complete"}
LABEL_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

ATTRIBUTION_INTERVENTIONS = {
    "quantization-gap": "quantization-gap-qat-scales",
    "teacher-ranking-gap": "teacher-ranking-gap-hard-state-density",
    "search-throughput-gap": "search-throughput-gap-caching",
}

BUILD_BINARY_ROLES = {
    "continuation_producer",
    "action_teacher",
    "rank4_position_teacher",
    "rank4_gate",
}


def canonical_json_bytes(value: Any) -> bytes:
    return qualification.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return qualification.sha256_bytes(value)


def sha256_file(path: pathlib.Path) -> str:
    return qualification.sha256_file(path)


def _production_allowlist_for_role(role: str) -> Mapping[str, Any] | None:
    if role in ALLOWLIST:
        return ALLOWLIST[role]
    if role == "attempt_one_initial_checkpoint":
        role = "initial_float_checkpoint"
    return ATTEMPT_ONE_INPUT_ALLOWLIST.get(role)


def utc(value: Any, label: str) -> str:
    qualification._utc(value, label)
    return str(value)


def _regular(path: pathlib.Path, *, ascii_required: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ChallengerError(f"{path} is absent, irregular, or redirected")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ChallengerError(f"{path} is not ASCII") from error
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _verify_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ChallengerError(f"{label} record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _regular(path) != dict(value):
        raise ChallengerError(f"{label} bytes changed")
    return path.resolve()


def _record_metadata(value: Any, label: str) -> dict[str, Any]:
    """Validate a provenance record without dereferencing its source path."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "bytes", "sha256"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
        or SHA256_RE.fullmatch(str(value.get("sha256", ""))) is None
    ):
        raise ChallengerError(f"{label} provenance record is malformed")
    return dict(value)


def _sealed_metadata(value: Any, schema: str, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "bytes", "sha256", "schema", "body_sha256"}
        or value.get("schema") != schema
        or SHA256_RE.fullmatch(str(value.get("body_sha256", ""))) is None
    ):
        raise ChallengerError(f"{label} sealed provenance is malformed")
    _record_metadata(
        {key: value[key] for key in ("path", "bytes", "sha256")}, label
    )
    return dict(value)


def _sealed_record(path: pathlib.Path, schema: str) -> dict[str, Any]:
    value = qualification.load_sealed(path, schema)
    return {
        **_regular(path),
        "schema": schema,
        "body_sha256": value["body_sha256"],
    }


def _verify_sealed_record(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "bytes", "sha256", "schema", "body_sha256",
    } or value.get("schema") != schema:
        raise ChallengerError(f"{label} sealed record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _sealed_record(path, schema) != dict(value):
        raise ChallengerError(f"{label} sealed bytes changed")
    return path.resolve()


def _safe_root(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    absolute = path.absolute()
    if os.path.lexists(absolute):
        if absolute.is_symlink() or not absolute.is_dir() or absolute != absolute.resolve():
            raise ChallengerError("campaign root is redirected or irregular")
    elif create:
        absolute.mkdir(parents=True, mode=0o700)
    else:
        raise ChallengerError("campaign root does not exist")
    return absolute.resolve()


def _output_paths(root: pathlib.Path) -> dict[str, str]:
    return {
        "root": str(root),
        "input_directory": str(root / "inputs"),
        "input_reference": str(root / "input-reference.json"),
        "plan": str(root / "campaign-plan.json"),
        "phase_plans": str(root / "phase-plans"),
        "ledger": str(root / "attempt-ledger/events"),
        "dual_final": str(root / "dual-final"),
        "completion": str(root / "campaign-complete.json"),
    }


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    artifact = qualification.seal(body)
    raw = canonical_json_bytes(artifact)
    path = directory / f"{sha256_bytes(raw)}{suffix}"
    qualification.atomic_write_once(path, raw)
    return path, artifact


def _architecture(runtime_path: pathlib.Path) -> dict[str, Any]:
    try:
        runtime, _payload, metadata = export_model.validate_runtime(runtime_path)
    except Exception as error:
        raise ChallengerError("candidate runtime failed compact validation") from error
    architecture = runtime.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("dimensions") != DIMENSIONS
        or architecture.get("biases") is not False
        or metadata.get("hidden_one") != 12
        or metadata.get("hidden_two") != 8
        or any("policy" in str(key).lower() for key in runtime)
    ):
        raise ChallengerError("challenger must remain fixed 6301-12-8-1 value-only")
    return {
        "id": ARCHITECTURE,
        "dimensions": DIMENSIONS,
        "biases": False,
        "outputs": 1,
        "head": "scalar-value-only",
        "policy_head": False,
        "runtime_body_sha256": runtime["body_sha256"],
        "payload_sha256": metadata["payload_sha256"],
    }


def _named_records(
    values: Mapping[str, pathlib.Path], label: str, *, allow_empty: bool = False,
) -> dict[str, Any]:
    if (not values and not allow_empty) or any(LABEL_RE.fullmatch(name) is None for name in values):
        raise ChallengerError(f"{label} names are absent or malformed")
    records = {name: _regular(path) for name, path in sorted(values.items())}
    paths = [record["path"] for record in records.values()]
    if len(paths) != len(set(paths)):
        raise ChallengerError(f"{label} paths are repeated")
    return records


def _bundle_suffix(path: pathlib.Path) -> str:
    suffix = "".join(path.suffixes[-2:]) or ".bin"
    suffix = re.sub(r"[^A-Za-z0-9._-]", "_", suffix)
    return suffix if suffix.startswith(".") else f".{suffix}"


def _copy_to_bundle(
    source: Mapping[str, Any], *, input_directory: pathlib.Path,
) -> dict[str, Any]:
    source_path = _verify_record(source, "bundle source")
    route = pathlib.PurePosixPath(
        "artifacts", f"{source['sha256']}{_bundle_suffix(source_path)}"
    )
    target = input_directory / pathlib.Path(route)
    raw = source_path.read_bytes()
    qualification.atomic_write_once(target, raw)
    if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
        raise ChallengerError("content-addressed input bundle copy changed")
    return {"route": route.as_posix(), "bytes": len(raw), "sha256": source["sha256"]}


def _copy_record_to_route(
    source: Mapping[str, Any], *, input_directory: pathlib.Path,
    route: pathlib.PurePosixPath,
) -> dict[str, Any]:
    source_path = _verify_record(source, "bundle source")
    if route.is_absolute() or ".." in route.parts or route.parts[:1] != ("artifacts",):
        raise ChallengerError("content-addressed bundle route escaped")
    target = input_directory / pathlib.Path(route)
    _atomic_copy_once(
        source_path, target, size=int(source["bytes"]), digest=str(source["sha256"])
    )
    return {
        "route": route.as_posix(),
        "bytes": int(source["bytes"]),
        "sha256": str(source["sha256"]),
    }


def _training_manifest_dependencies(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Return the adjacent file closure of a supported named training manifest."""

    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, Mapping) or value.get("schema") != "papersoccer.jacek-replay-csr-shard.v1":
        return {}
    npz_name = value.get("npz")
    npz_sha = value.get("npz_sha256")
    if (
        not isinstance(npz_name, str)
        or pathlib.PurePosixPath(npz_name).name != npz_name
        or npz_name != f"{npz_sha}.npz"
        or SHA256_RE.fullmatch(str(npz_sha)) is None
    ):
        raise ChallengerError("named training shard has a malformed NPZ dependency")
    dependency = path.parent / npz_name
    record = _regular(dependency)
    if record["sha256"] != npz_sha:
        raise ChallengerError("named training shard NPZ dependency changed")
    return {"npz": record}


def _copy_named_training_inputs(
    records: Mapping[str, Mapping[str, Any]], *, input_directory: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    copied: dict[str, Any] = {}
    dependencies: dict[str, Any] = {}
    source_dependencies: dict[str, Any] = {}
    for name, record in records.items():
        source = _verify_record(record, f"training input {name}")
        subtree = pathlib.PurePosixPath(
            "artifacts", "training-inputs", f"{name}-{record['sha256']}"
        )
        copied[name] = _copy_record_to_route(
            record, input_directory=input_directory, route=subtree / source.name
        )
        discovered = _training_manifest_dependencies(source)
        source_dependencies[name] = discovered
        dependencies[name] = {
            dependency_name: _copy_record_to_route(
                dependency,
                input_directory=input_directory,
                route=subtree / pathlib.Path(str(dependency["path"])).name,
            )
            for dependency_name, dependency in discovered.items()
        }
    return copied, dependencies, source_dependencies


def _verify_bundle_record(
    value: Any, *, input_directory: pathlib.Path, label: str,
) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"route", "bytes", "sha256"}:
        raise ChallengerError(f"{label} bundle record is malformed")
    route = pathlib.PurePosixPath(str(value.get("route", "")))
    if route.is_absolute() or ".." in route.parts or route.parts[:1] != ("artifacts",):
        raise ChallengerError(f"{label} bundle route escaped")
    path = input_directory / pathlib.Path(route)
    regular = _regular(path)
    if regular["bytes"] != value["bytes"] or regular["sha256"] != value["sha256"]:
        raise ChallengerError(f"{label} bundled bytes changed")
    return path.resolve()


def _bundle_named(
    records: Mapping[str, Mapping[str, Any]], *, input_directory: pathlib.Path,
) -> dict[str, Any]:
    return {
        name: _copy_to_bundle(record, input_directory=input_directory)
        for name, record in records.items()
    }


def _repository_identity() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
        text=True, capture_output=True, check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ChallengerError("cannot freeze repository commit")
    return {"root": str(REPOSITORY.resolve()), "commit": head}


def _compiler_identity() -> dict[str, Any]:
    executable = pathlib.Path(
        shutil.which(os.environ.get("CXX", "c++")) or ""
    ).resolve()
    record = _regular(executable)
    version = subprocess.run(
        [str(executable), "--version"], text=True, capture_output=True, check=False,
    )
    target = subprocess.run(
        [str(executable), "-dumpmachine"], text=True, capture_output=True, check=False,
    )
    if version.returncode != 0 or target.returncode != 0 or not target.stdout.strip():
        raise ChallengerError("cannot freeze C++ compiler identity")
    return {
        **record,
        "version_text": version.stdout,
        "version_sha256": sha256_bytes(version.stdout.encode("utf-8")),
        "target": target.stdout.strip(),
    }


def _audit_tool_paths() -> dict[str, pathlib.Path]:
    """Frozen source closure needed to audit or rebuild the campaign runners."""

    return {
        "campaign_tool": pathlib.Path(__file__).resolve(),
        "campaign_test": TEST_PATH,
        "qualification": pathlib.Path(qualification.__file__).resolve(),
        "openings": pathlib.Path(openings.__file__).resolve(),
        "model_exporter": EXPORT_MODEL_PATH,
        "pilot_pipeline": PILOT_PIPELINE_PATH,
        "teacher_training": HERE / "compact_value_bfm_teacher_training.py",
        "dual_final_runner": HERE / "compact_value_bfm_rank4_teacher_dual_final.py",
        "release_bridge": HERE / "compact_value_bfm_rank4_teacher_release.py",
        "release_bridge_test": (
            REPOSITORY
            / "tests/codingame/test_compact_value_bfm_rank4_teacher_release.py"
        ),
        "compact_trainer": TRAINER_PATH,
        "replay_trainer": HERE / "jacek_replay_train.py",
        "replay_corpus": HERE / "jacek_replay_corpus.py",
        "replay_features": HERE / "jacek_replay_features.py",
        "replay_pack": HERE / "jacek_replay_pack.py",
        "continuation_producer": HERE / "jacek_replay_continuations.cpp",
        "continuation_interface": HERE / "jacek_replay_continuations_internal.hpp",
        "action_teacher": HERE / "jacek_replay_bfm_search_teacher.cpp",
        "action_teacher_interface": HERE / "jacek_replay_bfm_search_teacher_internal.hpp",
        "rank4_position_teacher": HERE / "jacek_replay_rank4_position_teacher.cpp",
        "compact_engine": (
            REPOSITORY / "submissions/codingame/bots/compact_value_bfm/engine.cpp"
        ),
        "submission_exporter": (
            REPOSITORY
            / "submissions/codingame/bots/compact_value_bfm/export_submission.py"
        ),
        "rank4_gate": (
            REPOSITORY / "submissions/codingame/bots/compact_value_bfm/rank4_gate.cpp"
        ),
        "rank4_gate_support": (
            REPOSITORY
            / "submissions/codingame/bots/compact_value_bfm/rank4_gate_support.py"
        ),
        "bot_interface": REPOSITORY / "include/papersoccer/bot.hpp",
        "jacek_bfm_engine": (
            REPOSITORY / "src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp"
        ),
        "cmake": REPOSITORY / "CMakeLists.txt",
        "live_window_verifier": LIVE_WINDOW_PATH,
        "campaign_documentation": DOC_PATH,
    }


def _campaign_source_paths() -> dict[str, pathlib.Path]:
    """Return the complete source closure used by campaign producers/runners.

    Results, generated binaries, and protected artifacts are deliberately not
    included.  The closure contains every C++/Python source under the build and
    training subsystems, plus the build/configuration files that select them.
    """

    candidates: set[pathlib.Path] = {
        REPOSITORY / "CMakeLists.txt",
        REPOSITORY / "requirements-research.txt",
    }
    # These actor sources are textually included by the frozen continuation and
    # Rank-4 position-teacher producers.  They live outside the ordinary
    # include/src/tools and compact-submission trees, so an explicit roster is
    # required to keep the build manifest a complete producer-source closure.
    candidates.update(
        REPOSITORY / relative
        for relative in (
            "submissions/codingame/bots/rank_4/replay_book.hpp",
            "submissions/codingame/bots/rank_4/replay_value_model.hpp",
            "submissions/codingame/bots/rank_4/teacher_residual_model.hpp",
            "submissions/codingame/bots/rank_4/bot.cpp",
            "submissions/codingame/bots/jacek_nn/replay_book.hpp",
            "submissions/codingame/bots/jacek_nn/replay_value_model.hpp",
            "submissions/codingame/bots/jacek_nn/teacher_residual_model.hpp",
            "submissions/codingame/bots/jacek_nn/bot.cpp",
        )
    )
    for root, patterns in (
        (REPOSITORY / "include", ("*.hpp", "*.h")),
        (REPOSITORY / "src", ("*.cpp", "*.hpp", "*.h")),
        (REPOSITORY / "tools", ("*.py", "*.cpp", "*.hpp", "*.h")),
        (
            REPOSITORY / "submissions/codingame/bots/compact_value_bfm",
            ("*.py", "*.cpp", "*.hpp", "*.h", "*.json", "*.txt"),
        ),
        (REPOSITORY / "tests/codingame", ("*.py", "*.cpp", "*.hpp", "*.h")),
        (REPOSITORY / "docs", ("*.md",)),
        (REPOSITORY / ".github", ("*.yml", "*.yaml")),
    ):
        for pattern in patterns:
            candidates.update(root.rglob(pattern))
    candidates.add(RANK4_PATH)
    records: dict[str, pathlib.Path] = {}
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
        except ValueError as error:
            raise ChallengerError("campaign source closure escaped repository") from error
        records[relative] = path.resolve()
    required = {path.resolve() for path in _audit_tool_paths().values()}
    if not required.issubset(set(records.values())):
        raise ChallengerError("campaign source closure omits an audited tool")
    return records


def _build_binary_record(path: pathlib.Path) -> dict[str, Any]:
    record = _regular(path)
    if not os.access(path, os.X_OK):
        raise ChallengerError(f"campaign build binary is not executable: {path}")
    return {**record, "executable": True}


def create_build_manifest(
    output: pathlib.Path, *, binaries: Mapping[str, pathlib.Path],
    created_at_utc: str,
) -> pathlib.Path:
    """Seal a clean-checkout source/compiler/binary closure before freezing."""

    if set(binaries) != BUILD_BINARY_ROLES:
        raise ChallengerError("build manifest requires the exact producer binary roster")
    repository = _repository_identity()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY, text=True, capture_output=True, check=False,
    )
    if dirty.returncode != 0 or dirty.stdout:
        raise ChallengerError("build manifest requires a clean tracked checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *list(_campaign_source_paths())],
        cwd=REPOSITORY, text=True, capture_output=True, check=False,
    )
    if tracked.returncode != 0:
        raise ChallengerError("build source closure contains an untracked file")
    source_records = {
        relative: _regular(path)
        for relative, path in _campaign_source_paths().items()
    }
    body = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "clean-source-compiler-binaries-frozen",
        "created_at_utc": utc(created_at_utc, "build manifest timestamp"),
        "repository": repository,
        "source_closure": source_records,
        "compiler": _compiler_identity(),
        "binaries": {
            name: _build_binary_record(path)
            for name, path in sorted(binaries.items())
        },
        "build_contract": {
            "system": "cmake",
            "configuration": "Release",
            "language_standard": "c++20",
            "sources_clean": True,
            "binaries_built_after_source_freeze": True,
        },
    }
    qualification.write_sealed(output, body)
    _validate_build_manifest(output, production=True)
    return output.resolve()


def _validate_build_manifest(
    path: pathlib.Path, *, production: bool,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, BUILD_MANIFEST_SCHEMA)
    sources = value.get("source_closure")
    binaries = value.get("binaries")
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("status") != "clean-source-compiler-binaries-frozen"
        or not isinstance(sources, Mapping)
        or not sources
        or not isinstance(binaries, Mapping)
        or set(binaries) != BUILD_BINARY_ROLES
        or value.get("build_contract") != {
            "system": "cmake",
            "configuration": "Release",
            "language_standard": "c++20",
            "sources_clean": True,
            "binaries_built_after_source_freeze": True,
        }
    ):
        raise ChallengerError("campaign build manifest contract changed")
    utc(value.get("created_at_utc"), "build manifest timestamp")
    for relative, record in sources.items():
        route = _safe_relative(relative, "build source route")
        if route.as_posix() != relative:
            raise ChallengerError("build source route normalization changed")
        _verify_record(record, f"build source {relative}")
    for name, record in binaries.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256", "executable"}
            or record.get("executable") is not True
        ):
            raise ChallengerError(f"build binary record changed: {name}")
        binary = _verify_record(
            {key: record[key] for key in ("path", "bytes", "sha256")},
            f"build binary {name}",
        )
        if not os.access(binary, os.X_OK):
            raise ChallengerError(f"build binary lost executable status: {name}")
    compiler = value.get("compiler")
    if not isinstance(compiler, Mapping):
        raise ChallengerError("build compiler identity is absent")
    _record_metadata(
        {key: compiler[key] for key in ("path", "bytes", "sha256")},
        "build compiler",
    )
    if production:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPOSITORY, text=True, capture_output=True, check=False,
        )
        if dirty.returncode != 0 or dirty.stdout:
            raise ChallengerError("production freeze requires a clean source checkout")
        expected = {
            relative: _regular(source)
            for relative, source in _campaign_source_paths().items()
        }
        if sources != expected or value.get("repository") != _repository_identity():
            raise ChallengerError("build manifest is not the exact current source closure")
        current_compiler = _compiler_identity()
        if compiler != current_compiler:
            raise ChallengerError("build manifest compiler is not current")
    return value


def _copy_build_bundle(
    manifest_path: pathlib.Path, *, input_directory: pathlib.Path,
    production: bool,
) -> dict[str, Any]:
    value = _validate_build_manifest(manifest_path, production=production)
    sources = {
        relative: _copy_record_to_route(
            record,
            input_directory=input_directory,
            route=pathlib.PurePosixPath("artifacts", "source-closure", relative),
        )
        for relative, record in value["source_closure"].items()
    }
    binaries = {}
    for name, record in value["binaries"].items():
        copied = _copy_to_bundle(
            {key: record[key] for key in ("path", "bytes", "sha256")},
            input_directory=input_directory,
        )
        os.chmod(input_directory / pathlib.Path(copied["route"]), 0o555)
        binaries[name] = copied
    manifest = _copy_to_bundle(_regular(manifest_path), input_directory=input_directory)
    return {
        "manifest": {**manifest, "schema": BUILD_MANIFEST_SCHEMA,
                     "body_sha256": value["body_sha256"]},
        "repository": value["repository"],
        "compiler": value["compiler"],
        "sources": sources,
        "binaries": binaries,
        "build_contract": value["build_contract"],
    }


def _validate_copied_build_bundle(
    value: Any, *, input_directory: pathlib.Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "manifest", "repository", "compiler", "sources", "binaries",
        "build_contract",
    }:
        raise ChallengerError("copied build bundle field roster changed")
    manifest_record = value["manifest"]
    if not isinstance(manifest_record, Mapping) or set(manifest_record) != {
        "route", "bytes", "sha256", "schema", "body_sha256",
    }:
        raise ChallengerError("copied build manifest record changed")
    manifest_path = _verify_bundle_record(
        {key: manifest_record[key] for key in ("route", "bytes", "sha256")},
        input_directory=input_directory,
        label="copied build manifest",
    )
    manifest = qualification.load_sealed(manifest_path, BUILD_MANIFEST_SCHEMA)
    if manifest_record["body_sha256"] != manifest["body_sha256"]:
        raise ChallengerError("copied build manifest body changed")
    sources = value["sources"]
    binaries = value["binaries"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(manifest.get("source_closure", {}))
        or not isinstance(binaries, Mapping)
        or set(binaries) != BUILD_BINARY_ROLES
    ):
        raise ChallengerError("copied build source/binary roster changed")
    for relative, record in sources.items():
        source = _verify_bundle_record(
            record, input_directory=input_directory,
            label=f"copied build source {relative}",
        )
        if (
            pathlib.PurePosixPath(record["route"])
            != pathlib.PurePosixPath("artifacts", "source-closure", relative)
            or record["bytes"] != manifest["source_closure"][relative]["bytes"]
            or record["sha256"] != manifest["source_closure"][relative]["sha256"]
            or source.is_symlink()
        ):
            raise ChallengerError("copied build source binding changed")
    for name, record in binaries.items():
        binary_path = _verify_bundle_record(
            record, input_directory=input_directory,
            label=f"copied build binary {name}",
        )
        declared = manifest["binaries"][name]
        if (
            record["bytes"] != declared["bytes"]
            or record["sha256"] != declared["sha256"]
            or declared.get("executable") is not True
            or not os.access(binary_path, os.X_OK)
        ):
            raise ChallengerError("copied build binary binding changed")
    if (
        value.get("repository") != manifest.get("repository")
        or value.get("compiler") != manifest.get("compiler")
        or value.get("build_contract") != manifest.get("build_contract")
    ):
        raise ChallengerError("copied build provenance changed")
    return dict(value)


def _compiler_contract(value: Any, *, input_directory: pathlib.Path) -> None:
    if not isinstance(value, Mapping) or set(value) != {"identity", "bundle"}:
        raise ChallengerError("frozen compiler contract is malformed")
    identity = value["identity"]
    if not isinstance(identity, Mapping) or set(identity) != {
        "path", "bytes", "sha256", "version_text", "version_sha256", "target",
    }:
        raise ChallengerError("frozen compiler identity is malformed")
    _record_metadata(
        {key: identity[key] for key in ("path", "bytes", "sha256")},
        "compiler",
    )
    if (
        not isinstance(identity.get("version_text"), str)
        or sha256_bytes(identity["version_text"].encode("utf-8"))
        != identity.get("version_sha256")
        or not isinstance(identity.get("target"), str)
        or not identity["target"]
    ):
        raise ChallengerError("frozen compiler version identity changed")
    bundled = _verify_bundle_record(
        value["bundle"], input_directory=input_directory, label="bundled compiler"
    )
    if (
        bundled.stat().st_size != identity["bytes"]
        or sha256_file(bundled) != identity["sha256"]
    ):
        raise ChallengerError("bundled compiler differs from its frozen identity")


def _safe_relative(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ChallengerError(f"{label} route is absent")
    route = pathlib.PurePosixPath(value)
    if route.is_absolute() or ".." in route.parts or str(route) != value:
        raise ChallengerError(f"{label} route escaped its bundle")
    return route


def _atomic_copy_once(
    source: pathlib.Path, target: pathlib.Path, *, size: int, digest: str,
) -> None:
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != size or sha256_file(target) != digest:
            raise ChallengerError("immutable training-bundle copy changed")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != size or sha256_file(temporary) != digest:
            raise ChallengerError("training-bundle source changed during copy")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.stat().st_size != size or sha256_file(target) != digest:
                raise ChallengerError("training-bundle copy collided")
    finally:
        temporary.unlink(missing_ok=True)


def _copy_training_bundle(
    manifest_path: pathlib.Path, *, input_directory: pathlib.Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ChallengerError("training bundle manifest is absent or redirected")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChallengerError("training bundle manifest is invalid JSON") from error
    body = dict(manifest) if isinstance(manifest, Mapping) else {}
    claimed = body.pop("body_sha256", None)
    if (
        not isinstance(manifest, Mapping)
        or raw != canonical_json_bytes(manifest)
        or manifest.get("schema") != "papersoccer.compact-value-bfm-input-bundle.v1"
        or manifest.get("campaign_id") != "compact-value-bfm-20260831-v1"
        or manifest.get("feature_schema") != export_model.FEATURE_SCHEMA
        or claimed != sha256_bytes(canonical_json_bytes(body))
        or manifest.get("policy", {}).get("protected_tests_locked") is not True
        or manifest.get("policy", {}).get("runtime_uses_source_paths") is not False
        or manifest.get("policy", {}).get("git_required_after_freeze") is not False
        or manifest.get("protected_splits")
        != ["search:test", "rank4:test", "canonical:test"]
    ):
        raise ChallengerError("training bundle manifest contract changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ChallengerError("training bundle has no artifact roster")
    source_root = manifest_path.parent
    destination_root = input_directory / "training-bundle"
    seen_roles: set[str] = set()
    seen_routes: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    protected: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"role", "relative_path", "sha256", "bytes"}
            or not isinstance(record.get("role"), str)
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
        ):
            raise ChallengerError("training bundle artifact record changed")
        route = _safe_relative(record["relative_path"], "training artifact")
        if record["role"] in seen_roles or route.as_posix() in seen_routes:
            raise ChallengerError("training bundle artifact roster is repeated")
        source = source_root / pathlib.Path(route)
        if source.is_symlink() or not source.is_file() or source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise ChallengerError("training bundle source artifact changed")
        target = destination_root / pathlib.Path(route)
        _atomic_copy_once(
            source, target, size=record["bytes"], digest=record["sha256"]
        )
        bundled = {
            "route": f"training-bundle/{route.as_posix()}",
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "role": record["role"],
        }
        records[route.as_posix()] = bundled
        if "-test-" in record["role"]:
            protected[record["role"]] = bundled
        seen_roles.add(record["role"])
        seen_routes.add(route.as_posix())
    for relative, record in sorted(records.items()):
        if not relative.endswith(".json"):
            continue
        try:
            child = json.loads(
                (destination_root / pathlib.Path(relative)).read_bytes()
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(child, Mapping) or child.get("schema") != "papersoccer.jacek-replay-csr-shard.v1":
            continue
        dependency_name = child.get("npz")
        dependency_sha = child.get("npz_sha256")
        if (
            not isinstance(dependency_name, str)
            or pathlib.PurePosixPath(dependency_name).name != dependency_name
            or dependency_name != f"{dependency_sha}.npz"
            or SHA256_RE.fullmatch(str(dependency_sha)) is None
        ):
            raise ChallengerError("training bundle shard dependency is malformed")
        dependency_route = (
            pathlib.PurePosixPath(relative).parent / dependency_name
        ).as_posix()
        dependency = records.get(dependency_route)
        if dependency is None or dependency.get("sha256") != dependency_sha:
            raise ChallengerError("training bundle shard dependency is absent")
    routes = manifest.get("routes")
    if not isinstance(routes, Mapping):
        raise ChallengerError("training bundle routes are absent")
    declared_test_routes: set[str] = set()
    canonical_routes = routes.get("canonical_splits")
    if isinstance(canonical_routes, Mapping):
        declared_test_routes.update(str(value) for value in canonical_routes.get("test", []))
    for name in (
        "pilot_search_manifests", "full_search_manifests",
        "pilot_rank4_manifests", "full_rank4_manifests",
    ):
        values = routes.get(name)
        if isinstance(values, list) and len(values) == 3:
            declared_test_routes.add(str(values[2]))
    for relative in sorted(declared_test_routes):
        record = records.get(relative)
        if record is None:
            raise ChallengerError("declared protected test route is absent")
        protected[record["role"]] = record
        manifest_file = destination_root / relative
        try:
            child = json.loads(manifest_file.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        npz = child.get("npz") if isinstance(child, Mapping) else None
        if isinstance(npz, str):
            dependency = pathlib.PurePosixPath(relative).parent / npz
            dependency_record = records.get(dependency.as_posix())
            if dependency_record is None:
                raise ChallengerError("protected test manifest dependency is absent")
            protected[dependency_record["role"]] = dependency_record

    def route_list(name: str, *, first_two: bool = False) -> list[dict[str, Any]]:
        values = routes.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise ChallengerError(f"training bundle route changed: {name}")
        selected = values[:2] if first_two else values
        return [records[_safe_relative(value, name).as_posix()] for value in selected]

    canonical = routes.get("canonical_splits")
    if not isinstance(canonical, Mapping):
        raise ChallengerError("canonical training routes are absent")
    exposed = {
        "canonical_train": [records[_safe_relative(value, "canonical train").as_posix()] for value in canonical.get("train", [])],
        "canonical_validation": [records[_safe_relative(value, "canonical validation").as_posix()] for value in canonical.get("validation", [])],
        "pilot_search_train_validation": route_list("pilot_search_manifests", first_two=True),
        "full_search_train_validation": route_list("full_search_manifests", first_two=True),
        "pilot_rank4_train_validation": route_list("pilot_rank4_manifests", first_two=True),
        "full_rank4_train_validation": route_list("full_rank4_manifests", first_two=True),
        "common_adjudicator": records[_safe_relative(routes.get("common_adjudicator_manifest"), "common adjudicator").as_posix()],
    }
    if len(exposed["canonical_train"]) != 3 or len(exposed["canonical_validation"]) != 3:
        raise ChallengerError("canonical train/validation routes changed")
    manifest_target = destination_root / "bundle-manifest.json"
    qualification.atomic_write_once(manifest_target, raw)
    return {
        "manifest": {
            "route": "training-bundle/bundle-manifest.json",
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "body_sha256": claimed,
            "schema": manifest["schema"],
        },
        "artifact_count": len(records),
        "artifact_bytes": sum(record["bytes"] for record in records.values()),
        "artifacts": [records[route] for route in sorted(records)],
        "protected_test_artifacts": [protected[role] for role in sorted(protected)],
        "protected_test_count": len(protected),
        "exposed_routes": exposed,
        "source_manifest": _regular(manifest_path),
    }


def _validate_copied_training_bundle(
    value: Any, *, input_directory: pathlib.Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "manifest", "artifact_count", "artifact_bytes", "artifacts",
        "protected_test_artifacts", "protected_test_count", "exposed_routes",
    }:
        raise ChallengerError("copied training bundle record changed")
    manifest_record = value["manifest"]
    if not isinstance(manifest_record, Mapping) or set(manifest_record) != {
        "route", "bytes", "sha256", "body_sha256", "schema",
    }:
        raise ChallengerError("copied training bundle manifest record changed")
    route = _safe_relative(manifest_record["route"], "copied bundle manifest")
    manifest_path = input_directory / pathlib.Path(route)
    if (
        manifest_path.is_symlink() or not manifest_path.is_file()
        or manifest_path.stat().st_size != manifest_record["bytes"]
        or sha256_file(manifest_path) != manifest_record["sha256"]
    ):
        raise ChallengerError("copied training bundle manifest bytes changed")
    manifest = json.loads(manifest_path.read_bytes())
    body = dict(manifest)
    claimed = body.pop("body_sha256", None)
    if (
        manifest.get("schema") != "papersoccer.compact-value-bfm-input-bundle.v1"
        or manifest.get("campaign_id") != "compact-value-bfm-20260831-v1"
        or manifest.get("feature_schema") != export_model.FEATURE_SCHEMA
        or claimed != manifest_record["body_sha256"]
        or claimed != sha256_bytes(canonical_json_bytes(body))
        or manifest.get("policy", {}).get("protected_tests_locked") is not True
        or manifest.get("policy", {}).get("runtime_uses_source_paths") is not False
        or manifest.get("policy", {}).get("git_required_after_freeze") is not False
        or manifest.get("protected_splits")
        != ["search:test", "rank4:test", "canonical:test"]
    ):
        raise ChallengerError("copied training bundle manifest body changed")
    declared = {
        record["relative_path"]: record for record in manifest.get("artifacts", [])
        if isinstance(record, Mapping)
    }
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(declared) != len(artifacts):
        raise ChallengerError("copied training bundle artifact count changed")
    by_relative = {}
    for record in artifacts:
        if not isinstance(record, Mapping) or set(record) != {
            "route", "bytes", "sha256", "role",
        }:
            raise ChallengerError("copied training artifact record changed")
        prefix = "training-bundle/"
        if not str(record["route"]).startswith(prefix):
            raise ChallengerError("copied training artifact escaped its subtree")
        relative = str(record["route"])[len(prefix):]
        source_record = declared.get(relative)
        if source_record != {
            "role": record["role"], "relative_path": relative,
            "sha256": record["sha256"], "bytes": record["bytes"],
        }:
            raise ChallengerError("copied training artifact differs from manifest")
        target = input_directory / pathlib.Path(_safe_relative(record["route"], "copied training artifact"))
        if target.is_symlink() or not target.is_file() or target.stat().st_size != record["bytes"] or sha256_file(target) != record["sha256"]:
            raise ChallengerError("copied training artifact bytes changed")
        by_relative[relative] = record
    for relative in sorted(by_relative):
        if not relative.endswith(".json"):
            continue
        try:
            child = json.loads(
                (input_directory / "training-bundle" / relative).read_bytes()
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(child, Mapping) or child.get("schema") != "papersoccer.jacek-replay-csr-shard.v1":
            continue
        dependency_name = child.get("npz")
        dependency_sha = child.get("npz_sha256")
        if (
            not isinstance(dependency_name, str)
            or pathlib.PurePosixPath(dependency_name).name != dependency_name
            or dependency_name != f"{dependency_sha}.npz"
            or SHA256_RE.fullmatch(str(dependency_sha)) is None
        ):
            raise ChallengerError("copied training shard dependency is malformed")
        dependency_route = (
            pathlib.PurePosixPath(relative).parent / dependency_name
        ).as_posix()
        dependency = by_relative.get(dependency_route)
        if dependency is None or dependency.get("sha256") != dependency_sha:
            raise ChallengerError("copied training shard dependency is absent")
    protected_by_role = {
        record["role"]: record
        for record in artifacts if "-test-" in str(record["role"])
    }
    manifest_routes = manifest.get("routes", {})
    declared_test_routes = set()
    canonical_routes = manifest_routes.get("canonical_splits", {})
    if isinstance(canonical_routes, Mapping):
        declared_test_routes.update(str(value) for value in canonical_routes.get("test", []))
    for name in (
        "pilot_search_manifests", "full_search_manifests",
        "pilot_rank4_manifests", "full_rank4_manifests",
    ):
        values = manifest_routes.get(name)
        if isinstance(values, list) and len(values) == 3:
            declared_test_routes.add(str(values[2]))
    for relative in declared_test_routes:
        record = by_relative.get(relative)
        if record is None:
            raise ChallengerError("copied protected test route is absent")
        protected_by_role[record["role"]] = record
        try:
            child = json.loads(
                (input_directory / pathlib.Path(record["route"])).read_bytes()
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        npz = child.get("npz") if isinstance(child, Mapping) else None
        if isinstance(npz, str):
            dependency = (pathlib.PurePosixPath(relative).parent / npz).as_posix()
            dependency_record = by_relative.get(dependency)
            if dependency_record is None:
                raise ChallengerError("copied protected test dependency is absent")
            protected_by_role[dependency_record["role"]] = dependency_record
    protected = [protected_by_role[role] for role in sorted(protected_by_role)]
    if (
        value["artifact_count"] != len(artifacts)
        or value["artifact_bytes"] != sum(record["bytes"] for record in artifacts)
        or value["protected_test_count"] != len(protected)
        or value["protected_test_artifacts"]
        != protected
    ):
        raise ChallengerError("protected training-bundle classification changed")
    exposed = value.get("exposed_routes")
    if not isinstance(exposed, Mapping) or set(exposed) != {
        "canonical_train", "canonical_validation",
        "pilot_search_train_validation", "full_search_train_validation",
        "pilot_rank4_train_validation", "full_rank4_train_validation",
        "common_adjudicator",
    }:
        raise ChallengerError("training bundle exposed routes changed")
    def records_for(values: Sequence[str]) -> list[dict[str, Any]]:
        return [by_relative[_safe_relative(item, "exposed route").as_posix()] for item in values]

    expected_exposed = {
        "canonical_train": records_for(manifest_routes["canonical_splits"]["train"]),
        "canonical_validation": records_for(manifest_routes["canonical_splits"]["validation"]),
        "pilot_search_train_validation": records_for(manifest_routes["pilot_search_manifests"][:2]),
        "full_search_train_validation": records_for(manifest_routes["full_search_manifests"][:2]),
        "pilot_rank4_train_validation": records_for(manifest_routes["pilot_rank4_manifests"][:2]),
        "full_rank4_train_validation": records_for(manifest_routes["full_rank4_manifests"][:2]),
        "common_adjudicator": by_relative[_safe_relative(
            manifest_routes["common_adjudicator_manifest"], "common adjudicator"
        ).as_posix()],
    }
    if exposed != expected_exposed:
        raise ChallengerError("training bundle runner routes differ from manifest")
    expected_files = {
        pathlib.Path("bundle-manifest.json"),
        *(pathlib.Path(relative) for relative in by_relative),
    }
    training_root = input_directory / "training-bundle"
    actual_files = {
        path.relative_to(training_root)
        for path in training_root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files or any(
        path.is_symlink() for path in training_root.rglob("*")
    ):
        raise ChallengerError("copied training bundle tree contains extras or redirects")
    protected_routes = {record["route"] for record in protected}
    exposed_records = []
    for item in exposed.values():
        exposed_records.extend(item if isinstance(item, list) else [item])
    if any(
        not isinstance(record, Mapping)
        or record.get("route") in protected_routes
        or record.get("route", "").removeprefix("training-bundle/") not in by_relative
        for record in exposed_records
    ):
        raise ChallengerError("protected test artifact entered exposed training routes")
    return dict(value)


def freeze_campaign(
    *,
    output_root: pathlib.Path,
    candidate_runtime: pathlib.Path,
    candidate_source: pathlib.Path,
    rank4_source: pathlib.Path,
    teacher_runtime: pathlib.Path,
    teacher_manifest: pathlib.Path,
    mixed_six_exclusion: pathlib.Path,
    fresh_exclusion_receipt: pathlib.Path,
    attempt_zero_recovery_plan: pathlib.Path,
    training_bundle_manifest: pathlib.Path,
    attempt_one_initial_checkpoint: pathlib.Path,
    prior_runtime: pathlib.Path,
    roots_tsv: pathlib.Path,
    roots_manifest: pathlib.Path,
    build_manifest: pathlib.Path,
    training_inputs: Mapping[str, pathlib.Path],
    protected_exclusions: Mapping[str, pathlib.Path],
    live_exclusions: Mapping[str, pathlib.Path],
    created_at_utc: str,
    allow_unlisted_test_inputs: bool = False,
) -> pathlib.Path:
    created = utc(created_at_utc, "campaign freeze timestamp")
    try:
        recovery_plan_source = _sealed_record(
            attempt_zero_recovery_plan, RECOVERY_PLAN_SCHEMA
        )
    except Exception as error:
        raise ChallengerError("attempt-zero recovery plan is not sealed") from error
    root = _safe_root(output_root, create=True)
    paths = _output_paths(root)
    plan_path = pathlib.Path(paths["plan"])
    if plan_path.exists():
        existing = validate_campaign(plan_path)
        provenance = existing["inputs"]["source_provenance"]
        provided_allowlisted = {
            "teacher_runtime": _regular(teacher_runtime),
            "teacher_manifest": _regular(teacher_manifest),
            "attempt_zero_runtime": _regular(candidate_runtime),
            "attempt_zero_source": _regular(candidate_source, ascii_required=True),
            "mixed_six_exclusion": _regular(mixed_six_exclusion),
            "fresh_exclusion_receipt": _regular(fresh_exclusion_receipt),
            "attempt_zero_recovery_plan": _regular(attempt_zero_recovery_plan),
            "attempt_one_initial_checkpoint": _regular(
                attempt_one_initial_checkpoint
            ),
            "prior_runtime": _regular(prior_runtime),
            "roots_tsv": _regular(roots_tsv),
            "roots_manifest": _regular(roots_manifest),
            "build_manifest": _regular(build_manifest),
        }
        provided_training = _named_records(training_inputs, "training input")
        provided_training_dependencies = {
            name: _training_manifest_dependencies(
                pathlib.Path(str(record["path"]))
            )
            for name, record in provided_training.items()
        }
        provided_protected = _named_records({
            **protected_exclusions,
            "mixed-six": mixed_six_exclusion,
            "fresh-exclusion-receipt": fresh_exclusion_receipt,
        }, "protected exclusion")
        provided_live = _named_records(
            live_exclusions, "live exclusion", allow_empty=True
        )
        if (
            provenance.get("allowlisted_inputs") != provided_allowlisted
            or provenance.get("training_inputs") != provided_training
            or provenance.get("training_input_dependencies")
            != provided_training_dependencies
            or provenance.get("protected_exclusions") != provided_protected
            or provenance.get("live_exclusions") != provided_live
            or provenance.get("rank4_teacher") != _regular(rank4_source, ascii_required=True)
            or provenance.get("training_bundle_manifest") != _regular(training_bundle_manifest)
            or provenance.get("build_manifest") != _regular(build_manifest)
            or provenance.get("attempt_zero_recovery_plan") != recovery_plan_source
            or existing["plan"].get("created_at_utc") != created
        ):
            raise ChallengerError("existing freeze was invoked with different inputs")
        return plan_path
    if any(root.iterdir()):
        raise ChallengerError("campaign root predates its immutable plan")
    architecture = _architecture(candidate_runtime)
    source = _regular(candidate_source, ascii_required=True)
    if not 0 < source["bytes"] < SOURCE_LIMIT:
        raise ChallengerError("candidate source violates the 95KB limit")
    rank4 = _regular(rank4_source, ascii_required=True)
    if (
        rank4["sha256"] != qualification.RANK4_SHA256
        or rank4["bytes"] != qualification.RANK4_BYTES
    ):
        raise ChallengerError("Rank-4 source is not the maintained exact source")
    external_allowlisted = {
        "teacher_runtime": _regular(teacher_runtime),
        "teacher_manifest": _regular(teacher_manifest),
        "attempt_zero_runtime": _regular(candidate_runtime),
        "attempt_zero_source": source,
        "mixed_six_exclusion": _regular(mixed_six_exclusion),
        "fresh_exclusion_receipt": _regular(fresh_exclusion_receipt),
        "attempt_zero_recovery_plan": _regular(attempt_zero_recovery_plan),
        "attempt_one_initial_checkpoint": _regular(
            attempt_one_initial_checkpoint
        ),
        "prior_runtime": _regular(prior_runtime),
        "roots_tsv": _regular(roots_tsv),
        "roots_manifest": _regular(roots_manifest),
        "build_manifest": _regular(build_manifest),
    }
    allowlist_matches = True
    for role, record in external_allowlisted.items():
        expected = _production_allowlist_for_role(role)
        if role == "build_manifest":
            continue
        if expected is None:
            allowlist_matches = False
            continue
        if any(record.get(key) != value for key, value in expected.items() if key != "body_sha256"):
            allowlist_matches = False
    for role, schema in (
        ("mixed_six_exclusion", "papersoccer.compact-value-bfm.discrete-v3-development-recovery-mixed-six-exclusion.v1"),
        ("fresh_exclusion_receipt", "papersoccer.compact-value-bfm.discrete-v3-fresh-position-exclusion-audit.v1"),
        ("attempt_zero_recovery_plan", "papersoccer.compact-value-bfm.discrete-v3-development-recovery-plan.v1"),
    ):
        try:
            sealed = qualification.load_sealed(pathlib.Path(external_allowlisted[role]["path"]), schema)
        except Exception:
            allowlist_matches = False
        else:
            if sealed.get("body_sha256") != ALLOWLIST[role]["body_sha256"]:
                allowlist_matches = False
    training_bundle_source = _regular(training_bundle_manifest)
    if any(
        training_bundle_source.get(key) != value
        for key, value in TRAINING_BUNDLE_ALLOWLIST.items()
        if key != "body_sha256"
    ):
        allowlist_matches = False
    try:
        teacher = json.loads(pathlib.Path(external_allowlisted["teacher_manifest"]["path"]).read_bytes())
    except (OSError, json.JSONDecodeError):
        allowlist_matches = False
    else:
        runtime_binding = teacher.get("runtime") if isinstance(teacher, Mapping) else None
        if (
            not isinstance(runtime_binding, Mapping)
            or runtime_binding.get("artifact_sha256")
            != external_allowlisted["teacher_runtime"]["sha256"]
            or runtime_binding.get("bytes") != external_allowlisted["teacher_runtime"]["bytes"]
        ):
            allowlist_matches = False
    if not allowlist_matches and not allow_unlisted_test_inputs:
        raise ChallengerError("campaign inputs are outside the explicit production allowlist")
    external_training = _named_records(training_inputs, "training input")
    external_protected = _named_records({
        **protected_exclusions,
        "mixed-six": mixed_six_exclusion,
        "fresh-exclusion-receipt": fresh_exclusion_receipt,
    }, "protected exclusion")
    external_live = _named_records(
        live_exclusions, "live exclusion", allow_empty=True
    )
    training_identities = {(v["path"], v["sha256"]) for v in external_training.values()}
    exclusion_identities = {
        (v["path"], v["sha256"])
        for v in [*external_protected.values(), *external_live.values()]
    }
    if training_identities & exclusion_identities:
        raise ChallengerError("protected/live exclusions entered training inputs")
    input_directory = pathlib.Path(paths["input_directory"])
    allowlisted = {
        role: _copy_to_bundle(record, input_directory=input_directory)
        for role, record in external_allowlisted.items()
    }
    rank4_bundle = _copy_to_bundle(rank4, input_directory=input_directory)
    training, training_dependencies, external_training_dependencies = (
        _copy_named_training_inputs(
            external_training, input_directory=input_directory
        )
    )
    protected = _bundle_named(external_protected, input_directory=input_directory)
    live = _bundle_named(external_live, input_directory=input_directory)
    external_tools = {
        name: _regular(path) for name, path in _audit_tool_paths().items()
    }
    tool_bundle = _bundle_named(external_tools, input_directory=input_directory)
    compiler_identity = _compiler_identity()
    compiler_source = {
        key: compiler_identity[key] for key in ("path", "bytes", "sha256")
    }
    compiler_bundle = _copy_to_bundle(
        compiler_source, input_directory=input_directory
    )
    build_bundle = _copy_build_bundle(
        build_manifest,
        input_directory=input_directory,
        production=not allow_unlisted_test_inputs,
    )
    training_bundle = _copy_training_bundle(
        training_bundle_manifest, input_directory=input_directory
    )
    inputs_body = {
        "schema": INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "content-addressed-inputs-frozen",
        "created_at_utc": created,
        "candidate": {
            "runtime": allowlisted["attempt_zero_runtime"],
            "source": allowlisted["attempt_zero_source"],
            "architecture": architecture,
        },
        "rank4_teacher": rank4_bundle,
        "teacher": {
            "runtime": allowlisted["teacher_runtime"],
            "manifest": allowlisted["teacher_manifest"],
        },
        "attempt_one_inputs": {
            "initial_float_checkpoint": allowlisted[
                "attempt_one_initial_checkpoint"
            ],
            "prior_runtime": allowlisted["prior_runtime"],
            "roots_tsv": allowlisted["roots_tsv"],
            "roots_manifest": allowlisted["roots_manifest"],
            "student_runtime": allowlisted["attempt_zero_runtime"],
            "build_manifest": allowlisted["build_manifest"],
            "producer_binaries": build_bundle["binaries"],
        },
        "allowlisted_inputs": allowlisted,
        "production_allowlist_enforced": allowlist_matches,
        "training_inputs": training,
        "training_input_dependencies": training_dependencies,
        "training_bundle": {
            key: value for key, value in training_bundle.items()
            if key != "source_manifest"
        },
        "protected_exclusions": protected,
        "live_exclusions": live,
        "live_exclusion_state": {
            "count": len(live),
            "status": (
                "frozen-live-exclusions"
                if live else "no-live-evidence-exists"
            ),
        },
        "attempt_zero": {
            "status": "external-recovery-running-awaiting-result",
            "recovery_plan": allowlisted["attempt_zero_recovery_plan"],
            "recovery_plan_binding": {
                key: recovery_plan_source[key]
                for key in ("bytes", "sha256", "schema", "body_sha256")
            },
            "result_recorded": False,
        },
        "audit_tool_bundle": tool_bundle,
        "compiler_bundle": compiler_bundle,
        "build_bundle": build_bundle,
        "source_provenance": {
            "allowlisted_inputs": external_allowlisted,
            "training_inputs": external_training,
            "training_input_dependencies": external_training_dependencies,
            "protected_exclusions": external_protected,
            "live_exclusions": external_live,
            "tools": external_tools,
            "rank4_teacher": rank4,
            "training_bundle_manifest": training_bundle_source,
            "build_manifest": _regular(build_manifest),
            "attempt_zero_recovery_plan": recovery_plan_source,
            "compiler": compiler_identity,
        },
        "exclusion_policy": {
            "protected_labels_available_to_training": False,
            "live_games_available_to_training": False,
            "training_bundle_test_artifacts_locked": True,
            "all_exclusions_immutable": True,
        },
    }
    inputs_path, _inputs = _write_content_addressed(
        pathlib.Path(paths["input_directory"]), inputs_body, ".inputs.json"
    )
    qualification.write_sealed(pathlib.Path(paths["input_reference"]), {
        "schema": INPUT_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "inputs": _sealed_record(inputs_path, INPUT_SCHEMA),
    })
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "external-recovery-running-awaiting-result",
        "created_at_utc": created,
        "inputs": _sealed_record(inputs_path, INPUT_SCHEMA),
        "input_reference": _sealed_record(
            pathlib.Path(paths["input_reference"]), INPUT_REFERENCE_SCHEMA
        ),
        "architecture": architecture,
        "attempt_policy": {
            "initial_attempt": ATTEMPT_ZERO,
            "maximum_attempts": None,
            "goal_persists_until_success": True,
            "attempt_zero_state": "external-recovery-running",
            "implicit_retry": False,
            "new_attempt_requires_ledger_route": True,
        },
        "phases": {
            "pilot": {"games": 2_000, "quota_multiplier": 1, "quotas": PILOT_QUOTAS},
            "full": {"games": 10_000, "quota_multiplier": 5, "quotas": FULL_QUOTAS},
        },
        "adaptation": {
            "routing": "deterministic-from-terminal-phase-and-attempt-outcomes",
            "no_progress_limit": 2,
            "strength_delta_pp_min": 1.5,
            "teacher_regret_reduction_fraction_min": 0.10,
            "unchanged_progress_polls_never_abandon_phase": True,
            "architecture_may_change": False,
            "policy_head_may_be_added": False,
            "automatic_launch": False,
        },
        "dual_final": {
            "gate_ids": ["gate-a", "gate-b"],
            "thresholds": FINAL_THRESHOLDS,
            "candidate_authorization_precedes_banks": True,
            "banks_must_be_disjoint": True,
            "gate_b_excludes_gate_a": True,
            "candidate_identity_shared": True,
            "gate_result_reuse": False,
        },
        "completion": {
            "dual_qualification_is_terminal": False,
            "existing_submission_attestation_required": True,
            "default_uploads": 1,
            "live_games": 90,
            "clean_focus_operational_failures": 0,
            "additional_upload_requires_explicit_authorization": True,
            "terminal_event": "campaign-complete",
        },
        "resources": RESOURCE_LIMITS,
        "repository": _repository_identity(),
        "compiler": {
            "identity": compiler_identity,
            "bundle": compiler_bundle,
        },
        "build": build_bundle,
        "tools": {
            name: {"source": external_tools[name], "bundle": tool_bundle[name]}
            for name in external_tools
        },
        "outputs": paths,
        "policy": {
            "games_launched_by_this_tool": 0,
            "protected_final_generated": False,
            "rank4_replacement_authorized": False,
            "upload_authorized": False,
            "recurring_automation_assumed": False,
            "production_allowlist_enforced": allowlist_matches,
            "attempt_zero_external_recovery_running": True,
        },
    }
    qualification.write_sealed(plan_path, body)
    plan = validate_campaign(plan_path)
    _append_event(
        plan,
        attempt=ATTEMPT_ZERO,
        event="attempt-opened",
        created_at_utc=created,
        fields={
            "parent_attempt": None,
            "route": "await-attempt-zero-result",
            "architecture": ARCHITECTURE,
            "policy_head": False,
            "hypothesis": "external discrete-v3 recovery may already satisfy Rank-4 qualification",
            "intervention": "external-recovery",
            "attribution_receipt": None,
            "attempt_inputs": {
                "student_runtime": _inputs["candidate"]["runtime"],
                "prior_runtime": _inputs["attempt_one_inputs"]["prior_runtime"],
                "initial_float_checkpoint": _inputs["attempt_one_inputs"][
                    "initial_float_checkpoint"
                ],
                "roots_tsv": _inputs["attempt_one_inputs"]["roots_tsv"],
                "roots_manifest": _inputs["attempt_one_inputs"]["roots_manifest"],
                "build_manifest": _inputs["attempt_one_inputs"]["build_manifest"],
                "producer_binaries": _inputs["attempt_one_inputs"][
                    "producer_binaries"
                ],
            },
            "dynamic_exclusions": [],
        },
    )
    return plan_path


def validate_campaign(path: pathlib.Path) -> dict[str, Any]:
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    root = pathlib.Path(str(plan.get("outputs", {}).get("root", "")))
    if path.is_symlink() or not path.is_file() or path.resolve() != root / "campaign-plan.json":
        raise ChallengerError("campaign plan path changed")
    if _output_paths(_safe_root(root, create=False)) != plan.get("outputs"):
        raise ChallengerError("campaign output routes changed")
    inputs_path = _verify_sealed_record(plan.get("inputs"), INPUT_SCHEMA, "campaign inputs")
    if (
        inputs_path.parent != pathlib.Path(plan["outputs"]["input_directory"])
        or inputs_path.name != f"{sha256_file(inputs_path)}.inputs.json"
    ):
        raise ChallengerError("campaign inputs are not content addressed in the bundle")
    inputs = qualification.load_sealed(inputs_path, INPUT_SCHEMA)
    reference_path = _verify_sealed_record(
        plan.get("input_reference"), INPUT_REFERENCE_SCHEMA,
        "campaign input reference",
    )
    reference = qualification.load_sealed(reference_path, INPUT_REFERENCE_SCHEMA)
    if reference.get("inputs") != plan.get("inputs"):
        raise ChallengerError("campaign input reference changed")
    input_directory = inputs_path.parent
    runtime_path = _verify_bundle_record(
        inputs["candidate"]["runtime"], input_directory=input_directory,
        label="attempt-zero runtime",
    )
    architecture = _architecture(runtime_path)
    if set(inputs) != {
        "schema", "namespace", "campaign_id", "status", "created_at_utc",
        "candidate", "rank4_teacher", "teacher", "allowlisted_inputs",
        "production_allowlist_enforced", "training_inputs",
        "training_input_dependencies", "training_bundle",
        "protected_exclusions", "live_exclusions", "live_exclusion_state",
        "attempt_zero", "attempt_one_inputs", "audit_tool_bundle",
        "compiler_bundle", "build_bundle", "source_provenance",
        "exclusion_policy",
        "body_sha256",
    }:
        raise ChallengerError("campaign input field roster changed")
    if set(plan) != {
        "schema", "namespace", "campaign_id", "status", "created_at_utc",
        "inputs", "input_reference", "architecture", "attempt_policy",
        "phases", "adaptation", "dual_final", "resources", "tools",
        "completion", "repository", "compiler", "build", "outputs", "policy",
        "body_sha256",
    }:
        raise ChallengerError("campaign plan field roster changed")
    bundled_groups = (
        inputs["allowlisted_inputs"], inputs["training_inputs"],
        inputs["protected_exclusions"], inputs["live_exclusions"],
        inputs["audit_tool_bundle"],
    )
    _validate_copied_training_bundle(
        inputs["training_bundle"], input_directory=input_directory
    )
    _validate_copied_build_bundle(
        inputs["build_bundle"], input_directory=input_directory
    )
    if plan.get("build") != inputs.get("build_bundle"):
        raise ChallengerError("campaign plan/build bundle binding changed")
    for group in bundled_groups:
        if not isinstance(group, Mapping):
            raise ChallengerError("frozen input bundle mapping changed")
        for record in group.values():
            _verify_bundle_record(
                record, input_directory=input_directory, label="frozen input"
            )
    dependencies = inputs.get("training_input_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        inputs["training_inputs"]
    ):
        raise ChallengerError("named training dependency roster changed")
    dependency_records = []
    for name, group in dependencies.items():
        if not isinstance(group, Mapping) or set(group) - {"npz"}:
            raise ChallengerError("named training dependency group changed")
        primary = _verify_bundle_record(
            inputs["training_inputs"][name], input_directory=input_directory,
            label=f"training input {name}",
        )
        for dependency in group.values():
            dependency_path = _verify_bundle_record(
                dependency, input_directory=input_directory,
                label=f"training input dependency {name}",
            )
            dependency_records.append(dependency)
            if dependency_path.parent != primary.parent:
                raise ChallengerError("named training dependency lost adjacency")
        discovered = _training_manifest_dependencies(primary)
        if {
            dependency_name: {
                "route": str(pathlib.PurePosixPath(inputs["training_inputs"][name]["route"]).parent / pathlib.Path(record["path"]).name),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for dependency_name, record in discovered.items()
        } != group:
            raise ChallengerError("named training manifest closure changed")
    _compiler_contract(plan.get("compiler"), input_directory=input_directory)
    if inputs.get("compiler_bundle") != plan["compiler"]["bundle"]:
        raise ChallengerError("compiler input/plan bundle binding changed")
    source_path = _verify_bundle_record(
        inputs["candidate"]["source"], input_directory=input_directory,
        label="attempt-zero source",
    )
    try:
        source_path.read_bytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise ChallengerError("bundled candidate source is not ASCII") from error
    candidate_source = inputs["candidate"]["source"]
    _verify_bundle_record(
        inputs["rank4_teacher"], input_directory=input_directory,
        label="Rank-4 teacher",
    )
    expected_artifact_routes = {
        pathlib.Path(record["route"]).relative_to("artifacts")
        for record in [
            inputs["rank4_teacher"],
            inputs["compiler_bundle"],
            inputs["build_bundle"]["manifest"],
            *inputs["build_bundle"]["sources"].values(),
            *inputs["build_bundle"]["binaries"].values(),
            *dependency_records,
            *(record for group in bundled_groups for record in group.values()),
        ]
        if str(record["route"]).startswith("artifacts/")
    }
    artifact_root = input_directory / "artifacts"
    actual_artifact_routes = {
        path.relative_to(artifact_root)
        for path in artifact_root.rglob("*") if path.is_file()
    }
    if actual_artifact_routes != expected_artifact_routes or any(
        path.is_symlink() for path in artifact_root.rglob("*")
    ):
        raise ChallengerError("content-addressed artifact bundle roster changed")
    if (
        candidate_source != inputs["candidate"]["source"]
        or not 0 < candidate_source["bytes"] < SOURCE_LIMIT
        or inputs["rank4_teacher"]["sha256"] != qualification.RANK4_SHA256
        or inputs["rank4_teacher"]["bytes"] != qualification.RANK4_BYTES
    ):
        raise ChallengerError("candidate/Rank-4 source binding changed")
    attempt_one = inputs.get("attempt_one_inputs")
    if (
        not isinstance(attempt_one, Mapping)
        or set(attempt_one) != {
            "initial_float_checkpoint", "prior_runtime", "roots_tsv",
            "roots_manifest", "student_runtime", "build_manifest",
            "producer_binaries",
        }
        or attempt_one.get("initial_float_checkpoint")
        != inputs["allowlisted_inputs"].get("attempt_one_initial_checkpoint")
        or attempt_one.get("prior_runtime")
        != inputs["allowlisted_inputs"].get("prior_runtime")
        or attempt_one.get("roots_tsv")
        != inputs["allowlisted_inputs"].get("roots_tsv")
        or attempt_one.get("roots_manifest")
        != inputs["allowlisted_inputs"].get("roots_manifest")
        or attempt_one.get("student_runtime")
        != inputs["allowlisted_inputs"].get("attempt_zero_runtime")
        or attempt_one.get("build_manifest")
        != inputs["allowlisted_inputs"].get("build_manifest")
        or attempt_one.get("producer_binaries")
        != inputs["build_bundle"].get("binaries")
    ):
        raise ChallengerError("attempt-one frozen input binding changed")
    training_identities = {
        (record["route"], record["sha256"])
        for record in inputs["training_inputs"].values()
    }
    excluded_identities = {
        (record["route"], record["sha256"])
        for record in [
            *inputs["protected_exclusions"].values(),
            *inputs["live_exclusions"].values(),
        ]
    }
    if training_identities & excluded_identities:
        raise ChallengerError("protected/live exclusions entered frozen training inputs")
    if (
        plan.get("namespace") != NAMESPACE
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("status") != "external-recovery-running-awaiting-result"
        or plan.get("architecture") != architecture
        or architecture.get("id") != ARCHITECTURE
        or architecture.get("policy_head") is not False
        or plan.get("phases") != {
            "pilot": {"games": 2_000, "quota_multiplier": 1, "quotas": PILOT_QUOTAS},
            "full": {"games": 10_000, "quota_multiplier": 5, "quotas": FULL_QUOTAS},
        }
        or plan.get("attempt_policy") != {
            "initial_attempt": 0,
            "maximum_attempts": None,
            "goal_persists_until_success": True,
            "attempt_zero_state": "external-recovery-running",
            "implicit_retry": False,
            "new_attempt_requires_ledger_route": True,
        }
        or plan.get("adaptation") != {
            "routing": "deterministic-from-terminal-phase-and-attempt-outcomes",
            "no_progress_limit": 2,
            "strength_delta_pp_min": 1.5,
            "teacher_regret_reduction_fraction_min": 0.10,
            "unchanged_progress_polls_never_abandon_phase": True,
            "architecture_may_change": False,
            "policy_head_may_be_added": False,
            "automatic_launch": False,
        }
        or plan.get("dual_final") != {
            "gate_ids": ["gate-a", "gate-b"],
            "thresholds": FINAL_THRESHOLDS,
            "candidate_authorization_precedes_banks": True,
            "banks_must_be_disjoint": True,
            "gate_b_excludes_gate_a": True,
            "candidate_identity_shared": True,
            "gate_result_reuse": False,
        }
        or plan.get("completion") != {
            "dual_qualification_is_terminal": False,
            "existing_submission_attestation_required": True,
            "default_uploads": 1,
            "live_games": 90,
            "clean_focus_operational_failures": 0,
            "additional_upload_requires_explicit_authorization": True,
            "terminal_event": "campaign-complete",
        }
        or plan.get("resources") != RESOURCE_LIMITS
        or plan.get("policy") != {
            "games_launched_by_this_tool": 0,
            "protected_final_generated": False,
            "rank4_replacement_authorized": False,
            "upload_authorized": False,
            "recurring_automation_assumed": False,
            "production_allowlist_enforced": inputs.get("production_allowlist_enforced"),
            "attempt_zero_external_recovery_running": True,
        }
        or plan.get("policy", {}).get("production_allowlist_enforced")
        is not inputs.get("production_allowlist_enforced")
    ):
        raise ChallengerError("campaign plan contract changed")
    bundled_recovery_plan_record = inputs["allowlisted_inputs"].get(
        "attempt_zero_recovery_plan"
    )
    bundled_recovery_plan_path = _verify_bundle_record(
        bundled_recovery_plan_record,
        input_directory=input_directory,
        label="attempt-zero recovery plan",
    )
    bundled_recovery_plan = qualification.load_sealed(
        bundled_recovery_plan_path, RECOVERY_PLAN_SCHEMA
    )
    expected_recovery_binding = {
        "bytes": bundled_recovery_plan_record["bytes"],
        "sha256": bundled_recovery_plan_record["sha256"],
        "schema": RECOVERY_PLAN_SCHEMA,
        "body_sha256": bundled_recovery_plan["body_sha256"],
    }
    if (
        inputs.get("namespace") != NAMESPACE
        or inputs.get("campaign_id") != CAMPAIGN_ID
        or inputs.get("status") != "content-addressed-inputs-frozen"
        or inputs.get("candidate") != {
            "runtime": inputs["allowlisted_inputs"]["attempt_zero_runtime"],
            "source": inputs["allowlisted_inputs"]["attempt_zero_source"],
            "architecture": architecture,
        }
        or inputs.get("teacher") != {
            "runtime": inputs["allowlisted_inputs"]["teacher_runtime"],
            "manifest": inputs["allowlisted_inputs"]["teacher_manifest"],
        }
        or inputs.get("attempt_zero") != {
            "status": "external-recovery-running-awaiting-result",
            "recovery_plan": inputs["allowlisted_inputs"]["attempt_zero_recovery_plan"],
            "recovery_plan_binding": expected_recovery_binding,
            "result_recorded": False,
        }
        or inputs.get("protected_exclusions", {}).get("mixed-six")
        != inputs["allowlisted_inputs"]["mixed_six_exclusion"]
        or inputs.get("protected_exclusions", {}).get("fresh-exclusion-receipt")
        != inputs["allowlisted_inputs"]["fresh_exclusion_receipt"]
        or inputs.get("exclusion_policy") != {
            "protected_labels_available_to_training": False,
            "live_games_available_to_training": False,
            "training_bundle_test_artifacts_locked": True,
            "all_exclusions_immutable": True,
        }
        or not inputs.get("training_inputs")
        or not inputs.get("protected_exclusions")
        or inputs.get("live_exclusion_state") != {
            "count": len(inputs.get("live_exclusions", {})),
            "status": (
                "frozen-live-exclusions"
                if inputs.get("live_exclusions") else "no-live-evidence-exists"
            ),
        }
    ):
        raise ChallengerError("campaign frozen input contract changed")
    if inputs.get("production_allowlist_enforced") is True:
        for role, expected in {
            **ALLOWLIST, **ATTEMPT_ONE_INPUT_ALLOWLIST,
        }.items():
            frozen_role = (
                "attempt_one_initial_checkpoint"
                if role == "initial_float_checkpoint"
                else role
            )
            record = inputs["allowlisted_inputs"].get(frozen_role)
            if not isinstance(record, Mapping) or any(
                record.get(key) != value
                for key, value in expected.items() if key != "body_sha256"
            ):
                raise ChallengerError("production allowlist binding changed")
        if any(
            inputs["training_bundle"]["manifest"].get(key) != value
            for key, value in TRAINING_BUNDLE_ALLOWLIST.items()
        ):
            raise ChallengerError("production training bundle allowlist changed")
        for role, schema in (
            ("mixed_six_exclusion", "papersoccer.compact-value-bfm.discrete-v3-development-recovery-mixed-six-exclusion.v1"),
            ("fresh_exclusion_receipt", "papersoccer.compact-value-bfm.discrete-v3-fresh-position-exclusion-audit.v1"),
            ("attempt_zero_recovery_plan", "papersoccer.compact-value-bfm.discrete-v3-development-recovery-plan.v1"),
        ):
            bundled_path = _verify_bundle_record(
                inputs["allowlisted_inputs"][role],
                input_directory=input_directory, label=role,
            )
            sealed = qualification.load_sealed(bundled_path, schema)
            if sealed.get("body_sha256") != ALLOWLIST[role]["body_sha256"]:
                raise ChallengerError("production sealed allowlist body changed")
        teacher_manifest_path = _verify_bundle_record(
            inputs["allowlisted_inputs"]["teacher_manifest"],
            input_directory=input_directory, label="teacher manifest",
        )
        teacher_manifest = json.loads(teacher_manifest_path.read_bytes())
        teacher_runtime = teacher_manifest.get("runtime")
        if (
            not isinstance(teacher_runtime, Mapping)
            or teacher_runtime.get("artifact_sha256")
            != inputs["allowlisted_inputs"]["teacher_runtime"]["sha256"]
        ):
            raise ChallengerError("teacher manifest/runtime bundle binding changed")
    expected_tools = set(_audit_tool_paths())
    if set(plan.get("tools", {})) != expected_tools:
        raise ChallengerError("campaign tool closure changed")
    for name in sorted(expected_tools):
        value = plan["tools"][name]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"source", "bundle"}
            or value["bundle"] != inputs["audit_tool_bundle"][name]
        ):
            raise ChallengerError(f"campaign tool changed: {name}")
        source = _record_metadata(value["source"], f"campaign tool {name}")
        bundled = _verify_bundle_record(
            value["bundle"], input_directory=input_directory,
            label=f"bundled campaign tool {name}",
        )
        if bundled.stat().st_size != source["bytes"] or sha256_file(bundled) != source["sha256"]:
            raise ChallengerError(f"campaign tool/bundle differ: {name}")
    repository = plan.get("repository")
    if (
        not isinstance(repository, Mapping)
        or set(repository) != {"root", "commit"}
        or not isinstance(repository.get("root"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("commit", ""))) is None
    ):
        raise ChallengerError("repository provenance changed")
    return {"plan": plan, "path": path.resolve(), "root": root, "inputs": inputs}


def _ledger_path(plan: Mapping[str, Any]) -> pathlib.Path:
    return pathlib.Path(plan["outputs"]["ledger"])


def _completed_no_improvement_streak(
    entries: Sequence[Mapping[str, Any]],
) -> int:
    """Count terminal trained attempts since the last attribution intervention."""

    for entry in reversed(entries):
        if (
            entry.get("event") == "attempt-outcome-recorded"
            and entry.get("admitted") is False
        ):
            if entry.get("adaptation_route") == "open-next-attempt-attribution-adaptation":
                return 0
            value = entry.get("consecutive_no_improvement")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
            raise ChallengerError("completed-attempt improvement streak changed")
        if entry.get("event") == "final-gate-recorded" and entry.get("passed") is False:
            return 0
    return 0


def _resolve_campaign_artifact(
    record: Any, *, plan: Mapping[str, Any], label: str,
) -> pathlib.Path:
    if isinstance(record, Mapping) and set(record) == {"route", "bytes", "sha256"}:
        return _verify_bundle_record(
            record,
            input_directory=pathlib.Path(plan["outputs"]["input_directory"]),
            label=label,
        )
    return _verify_record(record, label)


def verify_phase_build_source_closure(
    *, required_sources: Sequence[str],
    campaign_context: Mapping[str, Any] | None = None,
    phase_context: Mapping[str, Any] | None = None,
    stored_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and revalidate the code/build closure used by a phase.

    Preparation supplies ``campaign_context`` and ``phase_context`` so a
    bundle ``route`` (attempt one) or ledger-local ``path`` (later attempts)
    can be resolved.  Resume/load paths supply ``stored_closure``; its absolute
    manifest and producer records are then checked again.  In both modes the
    current files that will actually execute must match the source metadata in
    the sealed build manifest and the current HEAD must remain the manifest's
    commit.  Original paths embedded in the manifest are provenance only and
    are never dereferenced, which keeps copied campaign bundles self-contained.
    """

    normalized_routes: list[str] = []
    for raw in required_sources:
        route = _safe_relative(raw, "required build source")
        if route.as_posix() != raw:
            raise ChallengerError("required build source route is not normalized")
        normalized_routes.append(raw)
    if not normalized_routes or len(set(normalized_routes)) != len(normalized_routes):
        raise ChallengerError("required build source roster is empty or duplicated")
    normalized_routes.sort()

    bundled_sources: Mapping[str, Any] | None = None
    phase_build_record: Mapping[str, Any] | None = None
    if stored_closure is None:
        if not isinstance(campaign_context, Mapping) or not isinstance(
            phase_context, Mapping
        ):
            raise ChallengerError("phase build closure requires campaign and phase")
        plan = campaign_context.get("plan")
        phase = phase_context.get("phase")
        if not isinstance(plan, Mapping) or not isinstance(phase, Mapping):
            raise ChallengerError("phase build closure context is malformed")
        attempt_inputs = phase.get("attempt_inputs")
        producer_records = phase.get("producer_binaries")
        if (
            not isinstance(attempt_inputs, Mapping)
            or not isinstance(producer_records, Mapping)
            or set(producer_records) != BUILD_BINARY_ROLES
        ):
            raise ChallengerError("phase build closure inputs are incomplete")
        phase_build_record = attempt_inputs.get("build_manifest")
        manifest_path = _resolve_campaign_artifact(
            phase_build_record, plan=plan, label="phase build manifest"
        )
        resolved_producers = {
            role: _resolve_campaign_artifact(
                record, plan=plan, label=f"phase producer {role}"
            )
            for role, record in producer_records.items()
        }
        if isinstance(phase_build_record, Mapping) and "route" in phase_build_record:
            build_bundle = campaign_context.get("inputs", {}).get("build_bundle")
            if not isinstance(build_bundle, Mapping):
                raise ChallengerError("bundled phase build closure is absent")
            bundle_manifest = build_bundle.get("manifest")
            if (
                not isinstance(bundle_manifest, Mapping)
                or phase_build_record.get("bytes") != bundle_manifest.get("bytes")
                or phase_build_record.get("sha256") != bundle_manifest.get("sha256")
            ):
                raise ChallengerError("phase build manifest differs from frozen bundle")
            sources = build_bundle.get("sources")
            if not isinstance(sources, Mapping):
                raise ChallengerError("bundled build source closure is absent")
            bundled_sources = sources
    else:
        if not isinstance(stored_closure, Mapping) or set(stored_closure) != {
            "manifest", "repository_commit", "sources", "sources_sha256",
            "compiler", "producer_binaries", "closure_sha256",
        }:
            raise ChallengerError("stored build source closure field roster changed")
        manifest_path = _verify_sealed_record(
            stored_closure.get("manifest"), BUILD_MANIFEST_SCHEMA,
            "stored phase build manifest",
        )
        stored_producers = stored_closure.get("producer_binaries")
        if not isinstance(stored_producers, Mapping) or set(
            stored_producers
        ) != BUILD_BINARY_ROLES:
            raise ChallengerError("stored producer binary closure changed")
        resolved_producers = {
            role: _verify_record(record, f"stored phase producer {role}")
            for role, record in stored_producers.items()
        }

    manifest = qualification.load_sealed(manifest_path, BUILD_MANIFEST_SCHEMA)
    sources = manifest.get("source_closure")
    binaries = manifest.get("binaries")
    repository = manifest.get("repository")
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("status") != "clean-source-compiler-binaries-frozen"
        or not isinstance(sources, Mapping)
        or not isinstance(binaries, Mapping)
        or set(binaries) != BUILD_BINARY_ROLES
        or not isinstance(repository, Mapping)
        or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("commit", ""))) is None
    ):
        raise ChallengerError("phase build manifest contract changed")
    current_repository = _repository_identity()
    if repository.get("commit") != current_repository["commit"]:
        raise ChallengerError("current HEAD differs from phase build manifest")
    current_compiler = _compiler_identity()
    if manifest.get("compiler") != current_compiler:
        raise ChallengerError("current compiler differs from phase build manifest")

    normalized_sources: dict[str, dict[str, Any]] = {}
    for relative in normalized_routes:
        declared = sources.get(relative)
        metadata = _record_metadata(declared, f"build source {relative}")
        current_path = (REPOSITORY / pathlib.PurePosixPath(relative)).resolve()
        try:
            current_path.relative_to(REPOSITORY.resolve())
        except ValueError as error:
            raise ChallengerError("required phase source escaped repository") from error
        current = _regular(current_path)
        if (
            current["bytes"] != metadata["bytes"]
            or current["sha256"] != metadata["sha256"]
        ):
            raise ChallengerError(f"current phase source differs from build: {relative}")
        if bundled_sources is not None:
            bundled = bundled_sources.get(relative)
            if not isinstance(bundled, Mapping):
                raise ChallengerError(f"frozen build bundle omits source: {relative}")
            frozen_path = _verify_bundle_record(
                bundled,
                input_directory=pathlib.Path(
                    campaign_context["plan"]["outputs"]["input_directory"]
                ),
                label=f"frozen build source {relative}",
            )
            frozen = _regular(frozen_path)
            if (
                frozen["bytes"] != metadata["bytes"]
                or frozen["sha256"] != metadata["sha256"]
            ):
                raise ChallengerError(f"frozen build source differs: {relative}")
        normalized_sources[relative] = {
            "bytes": metadata["bytes"], "sha256": metadata["sha256"]
        }

    normalized_producers: dict[str, dict[str, Any]] = {}
    for role, producer_path in sorted(resolved_producers.items()):
        actual = _regular(producer_path)
        declared = binaries.get(role)
        if (
            not isinstance(declared, Mapping)
            or actual["bytes"] != declared.get("bytes")
            or actual["sha256"] != declared.get("sha256")
            or declared.get("executable") is not True
            or not os.access(producer_path, os.X_OK)
        ):
            raise ChallengerError(f"phase producer differs from build manifest: {role}")
        normalized_producers[role] = actual

    sources_sha256 = sha256_bytes(canonical_json_bytes(normalized_sources))
    normalized = {
        "manifest": _sealed_record(manifest_path, BUILD_MANIFEST_SCHEMA),
        "repository_commit": current_repository["commit"],
        "sources": normalized_sources,
        "sources_sha256": sources_sha256,
        "compiler": current_compiler,
        "producer_binaries": normalized_producers,
    }
    normalized["closure_sha256"] = sha256_bytes(canonical_json_bytes(normalized))
    if stored_closure is not None and dict(stored_closure) != normalized:
        raise ChallengerError("stored build source closure differs from current build")
    if phase_build_record is not None and (
        phase_build_record.get("bytes") != normalized["manifest"]["bytes"]
        or phase_build_record.get("sha256") != normalized["manifest"]["sha256"]
    ):
        raise ChallengerError("phase build-manifest binding changed")
    return normalized


def validate_build_source_closure_evidence(
    value: Any, *, required_sources: Sequence[str],
) -> dict[str, Any]:
    """Deep-check an archived closure without consulting the current checkout.

    Phase execution uses :func:`verify_phase_build_source_closure`, which also
    compares current HEAD/tool bytes.  Outcome evidence must remain auditable
    after an intentional later promotion commit, so this archival validator
    instead checks the sealed manifest, exact required roster, normalized
    hashes, frozen producers, compiler identity, and closure digest.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "manifest", "repository_commit", "sources", "sources_sha256",
        "compiler", "producer_binaries", "closure_sha256",
    }:
        raise ChallengerError("build source closure evidence field roster changed")
    routes = []
    for raw in required_sources:
        route = _safe_relative(raw, "build source evidence route")
        if route.as_posix() != raw:
            raise ChallengerError("build source evidence route is not normalized")
        routes.append(raw)
    if not routes or len(set(routes)) != len(routes):
        raise ChallengerError("build source evidence roster is empty or duplicated")
    routes.sort()
    manifest_path = _verify_sealed_record(
        value.get("manifest"), BUILD_MANIFEST_SCHEMA,
        "build source closure manifest",
    )
    manifest = qualification.load_sealed(manifest_path, BUILD_MANIFEST_SCHEMA)
    manifest_sources = manifest.get("source_closure")
    sources = value.get("sources")
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(manifest_sources, Mapping)
        or not isinstance(sources, Mapping)
        or set(sources) != set(routes)
        or value.get("repository_commit")
        != manifest.get("repository", {}).get("commit")
        or value.get("compiler") != manifest.get("compiler")
    ):
        raise ChallengerError("build source closure evidence changed")
    normalized_sources = {}
    for relative in routes:
        record = sources[relative]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"bytes", "sha256"}
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ChallengerError(f"build source evidence is malformed: {relative}")
        declared = _record_metadata(
            manifest_sources.get(relative), f"manifest build source {relative}"
        )
        normalized = {"bytes": declared["bytes"], "sha256": declared["sha256"]}
        if dict(record) != normalized:
            raise ChallengerError(f"build source evidence differs: {relative}")
        normalized_sources[relative] = normalized
    if value.get("sources_sha256") != sha256_bytes(
        canonical_json_bytes(normalized_sources)
    ):
        raise ChallengerError("build source evidence aggregate changed")
    producers = value.get("producer_binaries")
    manifest_binaries = manifest.get("binaries")
    if (
        not isinstance(producers, Mapping)
        or set(producers) != BUILD_BINARY_ROLES
        or not isinstance(manifest_binaries, Mapping)
        or set(manifest_binaries) != BUILD_BINARY_ROLES
    ):
        raise ChallengerError("build producer evidence roster changed")
    for role, record in producers.items():
        producer = _verify_record(record, f"archived phase producer {role}")
        declared = manifest_binaries[role]
        if (
            record.get("bytes") != declared.get("bytes")
            or record.get("sha256") != declared.get("sha256")
            or declared.get("executable") is not True
            or not os.access(producer, os.X_OK)
        ):
            raise ChallengerError(f"archived phase producer differs: {role}")
    body = {key: value[key] for key in value if key != "closure_sha256"}
    if value.get("closure_sha256") != sha256_bytes(canonical_json_bytes(body)):
        raise ChallengerError("build source closure evidence digest changed")
    return dict(value)


def _validate_attempt_inputs(
    value: Any, *, plan: Mapping[str, Any], require_student_12x8: bool,
) -> dict[str, Any]:
    expected = {
        "student_runtime", "prior_runtime", "initial_float_checkpoint",
        "roots_tsv", "roots_manifest", "build_manifest", "producer_binaries",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ChallengerError("attempt input binding roster changed")
    resolved = {
        name: _resolve_campaign_artifact(
            record, plan=plan, label=f"attempt input {name}"
        )
        for name, record in value.items() if name != "producer_binaries"
    }
    producers = value.get("producer_binaries")
    if not isinstance(producers, Mapping) or set(producers) != BUILD_BINARY_ROLES:
        raise ChallengerError("attempt producer binary roster changed")
    for name, record in producers.items():
        binary = _resolve_campaign_artifact(
            record, plan=plan, label=f"attempt producer binary {name}"
        )
        if not os.access(binary, os.X_OK):
            raise ChallengerError(f"attempt producer binary is not executable: {name}")
    if require_student_12x8:
        _architecture(resolved["student_runtime"])
    try:
        export_model.validate_runtime(resolved["prior_runtime"])
    except Exception as error:
        raise ChallengerError("attempt prior runtime failed compact validation") from error
    if (
        resolved["initial_float_checkpoint"].name
        != f"{sha256_file(resolved['initial_float_checkpoint'])}.float.npz"
    ):
        raise ChallengerError("attempt initial checkpoint is not content addressed")
    build = qualification.load_sealed(
        resolved["build_manifest"], BUILD_MANIFEST_SCHEMA
    )
    if build.get("campaign_id") != CAMPAIGN_ID:
        raise ChallengerError("attempt build manifest belongs to another campaign")
    if any(
        record.get("sha256") != build.get("binaries", {}).get(name, {}).get("sha256")
        or record.get("bytes") != build.get("binaries", {}).get(name, {}).get("bytes")
        for name, record in producers.items()
    ):
        raise ChallengerError("attempt producer binaries differ from build manifest")
    # The replay-root parser performs semantic validation later.  Here the
    # manifest must at least bind the exact TSV bytes used by the phase.
    try:
        roots = json.loads(resolved["roots_manifest"].read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChallengerError("attempt roots manifest is invalid JSON") from error
    if not isinstance(roots, Mapping):
        raise ChallengerError("attempt roots manifest is malformed")
    source_record = roots.get("source_roots")
    output_sha = roots.get("output_sha256")
    tsv_sha = sha256_file(resolved["roots_tsv"])
    if (
        isinstance(source_record, Mapping)
        and source_record.get("sha256") not in {None, tsv_sha}
    ) or (isinstance(output_sha, str) and output_sha != tsv_sha):
        raise ChallengerError("attempt roots manifest/TSV binding changed")
    return dict(value)


def _load_live_fingerprint_evidence(
    value: Any,
) -> tuple[pathlib.Path, dict[str, Any]]:
    path = _verify_sealed_record(
        value, LIVE_FINGERPRINT_EVIDENCE_SCHEMA,
        "trusted live fingerprint evidence",
    )
    evidence = qualification.load_sealed(
        path, LIVE_FINGERPRINT_EVIDENCE_SCHEMA
    )
    fingerprints = evidence.get("fingerprints")
    game_ids = evidence.get("game_ids")
    source = evidence.get("source_identity")
    collector = evidence.get("collector_manifest")
    reference = evidence.get("live_window_reference")
    receipt = evidence.get("live_window_receipt")
    expected_fields = {
        "schema", "namespace", "status", "live_window_reference",
        "live_window_receipt", "collector_manifest", "source_identity",
        "exact_games", "game_ids", "game_ids_sha256", "canonicalization",
        "boundary_count", "fingerprints", "fingerprint_count",
        "fingerprints_sha256", "contains_transcripts", "contains_metrics",
        "contains_labels", "training_eligible", "body_sha256",
    }
    if (
        set(evidence) != expected_fields
        or evidence.get("namespace") != NAMESPACE
        or evidence.get("status") != "verified-live-canonical-fingerprints"
        or evidence.get("exact_games") != 90
        or not isinstance(game_ids, list)
        or game_ids != sorted(set(game_ids))
        or len(game_ids) != 90
        or any(
            isinstance(game_id, bool) or not isinstance(game_id, int)
            or game_id <= 0 for game_id in game_ids
        )
        or evidence.get("game_ids_sha256")
        != sha256_bytes(canonical_json_bytes(game_ids))
        or not isinstance(fingerprints, list) or not fingerprints
        or fingerprints != sorted(set(fingerprints))
        or any(SHA256_RE.fullmatch(str(item)) is None for item in fingerprints)
        or evidence.get("fingerprint_count") != len(fingerprints)
        or evidence.get("fingerprints_sha256")
        != sha256_bytes(canonical_json_bytes(fingerprints))
        or isinstance(evidence.get("boundary_count"), bool)
        or not isinstance(evidence.get("boundary_count"), int)
        or evidence["boundary_count"] < len(fingerprints)
        or evidence.get("canonicalization")
        != "minimum(exact,rotate180,reflect,rotate180-reflect)"
        or evidence.get("contains_transcripts") is not False
        or evidence.get("contains_metrics") is not False
        or evidence.get("contains_labels") is not False
        or evidence.get("training_eligible") is not False
        or not isinstance(source, Mapping)
        or set(source) != {
            "agent_id", "submission_id", "repository_commit",
            "source_sha256", "source_bytes",
        }
        or isinstance(source.get("agent_id"), bool)
        or not isinstance(source.get("agent_id"), int)
        or source["agent_id"] <= 0
        or isinstance(source.get("submission_id"), bool)
        or not isinstance(source.get("submission_id"), int)
        or source["submission_id"] <= 0
        or qualification.COMMIT_RE.fullmatch(
            str(source.get("repository_commit", ""))
        ) is None
        or SHA256_RE.fullmatch(str(source.get("source_sha256", ""))) is None
        or isinstance(source.get("source_bytes"), bool)
        or not isinstance(source.get("source_bytes"), int)
        or not 0 < source["source_bytes"] < SOURCE_LIMIT
        or not isinstance(reference, Mapping)
        or set(reference) != {"path", "sha256", "body_sha256"}
        or not isinstance(receipt, Mapping)
        or set(receipt) != {"path", "sha256", "body_sha256"}
        or any(
            not isinstance(record.get("path"), str) or not record["path"]
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            or SHA256_RE.fullmatch(str(record.get("body_sha256", ""))) is None
            for record in (reference, receipt)
        )
        or not isinstance(collector, Mapping)
        or set(collector) != {
            "path", "sha256", "schema", "collector_sha256",
            "accepted_records_sha256",
        }
        or not isinstance(collector.get("path"), str)
        or not collector["path"]
        or collector.get("schema")
        != "papersoccer.codingame-arena-batch.v1"
        or any(
            SHA256_RE.fullmatch(str(collector.get(field, ""))) is None
            for field in ("sha256", "collector_sha256", "accepted_records_sha256")
        )
    ):
        raise ChallengerError("trusted live fingerprint evidence changed")
    return path, evidence


def validate_dynamic_exclusion(path: pathlib.Path) -> dict[str, Any]:
    value = qualification.load_sealed(path, DYNAMIC_EXCLUSION_SCHEMA)
    fingerprints = value.get("fingerprints")
    origin = value.get("origin")
    classification = value.get("classification")
    protected = classification == "protected-final-canonical-fingerprints"
    live = classification == "live-diagnostic-canonical-fingerprints"
    expected_domain = (
        "protected-final-opening-canonical-state"
        if protected else "live-game-canonical-state"
    )
    expected_fields = {
            "schema", "namespace", "campaign_id", "attempt", "gate_id",
            "classification", "domain", "origin", "canonicalization",
            "fingerprints", "fingerprint_count", "contains_transcripts",
            "contains_metrics", "contains_labels", "training_eligible",
            "required_for_all_later_development_and_protected_banks",
            "body_sha256",
    }
    if live:
        expected_fields.add("live_fingerprint_evidence")
    if (
        set(value) != expected_fields
        or value.get("namespace") != NAMESPACE
        or value.get("campaign_id") != CAMPAIGN_ID
        or not (protected or live)
        or value.get("domain") != expected_domain
        or value.get("canonicalization")
        != "minimum(exact,rotate180,reflect,rotate180-reflect)"
        or not isinstance(origin, Mapping)
        or not isinstance(value.get("attempt"), int)
        or isinstance(value.get("attempt"), bool)
        or value["attempt"] < 0
        or not isinstance(value.get("gate_id"), str)
        or not value["gate_id"]
        or not isinstance(fingerprints, list)
        or not fingerprints
        or fingerprints != sorted(set(fingerprints))
        or value.get("fingerprint_count") != len(fingerprints)
        or any(SHA256_RE.fullmatch(str(item)) is None for item in fingerprints)
        or value.get("contains_labels") is not False
        or value.get("contains_metrics") is not False
        or value.get("contains_transcripts") is not False
        or value.get("training_eligible") is not False
        or value.get("required_for_all_later_development_and_protected_banks")
        is not True
    ):
        raise ChallengerError("sanitized dynamic exclusion changed")
    if protected and (
        value["gate_id"] not in {"gate-a", "gate-b"}
        or value["fingerprint_count"] != 500
        or set(origin) != {
            "candidate_source_sha256", "candidate_runtime_sha256",
            "protected_bank_sha256", "seed_sha256",
        }
        or any(SHA256_RE.fullmatch(str(item)) is None for item in origin.values())
    ):
        raise ChallengerError("protected-final sanitized exclusion changed")
    if live and (
        not value["gate_id"].startswith("live-upload-")
        or set(origin) != {
            "candidate_source_sha256", "live_receipt_sha256",
            "game_ids_sha256",
        }
        or any(SHA256_RE.fullmatch(str(item)) is None for item in origin.values())
    ):
        raise ChallengerError("live sanitized exclusion changed")
    if live:
        _evidence_path, evidence = _load_live_fingerprint_evidence(
            value.get("live_fingerprint_evidence")
        )
        if (
            value.get("fingerprints") != evidence["fingerprints"]
            or origin.get("candidate_source_sha256")
            != evidence["source_identity"]["source_sha256"]
            or origin.get("live_receipt_sha256")
            != evidence["live_window_receipt"]["sha256"]
            or origin.get("game_ids_sha256") != evidence["game_ids_sha256"]
        ):
            raise ChallengerError(
                "live sanitized exclusion differs from trusted fingerprint evidence"
            )
    return value


def _dynamic_exclusion_record(path: pathlib.Path) -> dict[str, Any]:
    value = validate_dynamic_exclusion(path)
    return {
        **_regular(path),
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "body_sha256": value["body_sha256"],
        "classification": value["classification"],
        "fingerprint_count": value["fingerprint_count"],
    }


def _verify_dynamic_exclusion_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "bytes", "sha256", "schema", "body_sha256",
        "classification", "fingerprint_count",
    }:
        raise ChallengerError(f"{label} dynamic exclusion record changed")
    path = pathlib.Path(str(value.get("path", "")))
    if _dynamic_exclusion_record(path) != dict(value):
        raise ChallengerError(f"{label} dynamic exclusion bytes changed")
    return path.resolve()


LiveFingerprintExtractor = Callable[
    [pathlib.Path, pathlib.Path], Mapping[str, Any]
]


def _default_live_fingerprint_extractor(
    reference_path: pathlib.Path, data_root: pathlib.Path,
) -> Mapping[str, Any]:
    module = _load(LIVE_WINDOW_PATH, "rank4_teacher_challenger_live_fingerprints")
    return module.extract_verified_live_fingerprints(
        reference_path, data_root=data_root
    )


def _extract_live_fingerprint_evidence(
    reference_path: pathlib.Path, data_root: pathlib.Path, *,
    extractor: LiveFingerprintExtractor,
    allow_injected_test_evidence: bool,
) -> dict[str, Any]:
    if (
        extractor is not _default_live_fingerprint_extractor
        and allow_injected_test_evidence is not True
    ):
        raise ChallengerError(
            "injected live fingerprint evidence is forbidden in production"
        )
    try:
        evidence = dict(extractor(reference_path, data_root))
        qualification.validate_seal(evidence)
    except Exception as error:
        raise ChallengerError("trusted live fingerprint extraction failed") from error
    if evidence.get("schema") != LIVE_FINGERPRINT_EVIDENCE_SCHEMA:
        raise ChallengerError("trusted live fingerprint extractor schema changed")
    return evidence


def _publish_live_fingerprint_evidence(
    output: pathlib.Path, evidence: Mapping[str, Any],
) -> pathlib.Path:
    raw = canonical_json_bytes(dict(evidence))
    digest = sha256_bytes(raw)
    path = (
        output.parent / "live-fingerprint-evidence"
        / f"{digest}.json"
    )
    qualification.atomic_write_once(path, raw)
    _load_live_fingerprint_evidence(
        _sealed_record(path, LIVE_FINGERPRINT_EVIDENCE_SCHEMA)
    )
    return path.resolve()


def _validate_live_dynamic_match(
    dynamic_path: pathlib.Path, *, candidate_source_sha256: str,
    attempt: int, upload_ordinal: int,
    live_reference: pathlib.Path | None = None,
    live_data_root: pathlib.Path | None = None,
    extractor: LiveFingerprintExtractor = _default_live_fingerprint_extractor,
    allow_injected_test_evidence: bool = False,
) -> dict[str, Any]:
    dynamic = validate_dynamic_exclusion(dynamic_path)
    _evidence_path, evidence = _load_live_fingerprint_evidence(
        dynamic["live_fingerprint_evidence"]
    )
    module = _load(LIVE_WINDOW_PATH, "rank4_teacher_challenger_live_fingerprint_paths")
    reference_path = (
        module.resolve_path(evidence["live_window_reference"]["path"])
        if live_reference is None else live_reference.resolve()
    )
    data_root = (
        reference_path.parent
        if live_data_root is None else live_data_root.resolve()
    )
    extracted = _extract_live_fingerprint_evidence(
        reference_path, data_root, extractor=extractor,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    if (
        extracted != evidence
        or SHA256_RE.fullmatch(candidate_source_sha256) is None
        or dynamic.get("attempt") != attempt
        or dynamic.get("gate_id") != f"live-upload-{upload_ordinal}"
        or dynamic.get("origin", {}).get("candidate_source_sha256")
        != candidate_source_sha256
    ):
        raise ChallengerError(
            "live sanitized exclusion is not the trusted rejected window"
        )
    return dynamic


def materialize_live_dynamic_exclusion(
    output: pathlib.Path, *, attempt: int, upload_ordinal: int,
    candidate_source_sha256: str, live_reference: pathlib.Path,
    live_data_root: pathlib.Path,
    fingerprint_extractor: LiveFingerprintExtractor = (
        _default_live_fingerprint_extractor
    ),
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    """Extract and publish only trusted canonical hashes from one live window."""

    evidence = _extract_live_fingerprint_evidence(
        live_reference, live_data_root, extractor=fingerprint_extractor,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    evidence_path = _publish_live_fingerprint_evidence(output, evidence)
    _published_path, published_evidence = _load_live_fingerprint_evidence(
        _sealed_record(evidence_path, LIVE_FINGERPRINT_EVIDENCE_SCHEMA)
    )
    if published_evidence != evidence:
        raise ChallengerError("published live fingerprint evidence changed")
    evidence = published_evidence
    values = evidence["fingerprints"]
    if (
        isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0
        or isinstance(upload_ordinal, bool)
        or not isinstance(upload_ordinal, int) or upload_ordinal <= 0
        or SHA256_RE.fullmatch(candidate_source_sha256) is None
        or evidence["source_identity"]["source_sha256"]
        != candidate_source_sha256
    ):
        raise ChallengerError("rejected-live fingerprint material is invalid")
    body = {
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "gate_id": f"live-upload-{upload_ordinal}",
        "classification": "live-diagnostic-canonical-fingerprints",
        "domain": "live-game-canonical-state",
        "origin": {
            "candidate_source_sha256": candidate_source_sha256,
            "live_receipt_sha256": evidence["live_window_receipt"]["sha256"],
            "game_ids_sha256": evidence["game_ids_sha256"],
        },
        "live_fingerprint_evidence": _sealed_record(
            evidence_path, LIVE_FINGERPRINT_EVIDENCE_SCHEMA
        ),
        "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
        "fingerprints": values,
        "fingerprint_count": len(values),
        "contains_transcripts": False,
        "contains_metrics": False,
        "contains_labels": False,
        "training_eligible": False,
        "required_for_all_later_development_and_protected_banks": True,
    }
    qualification.write_sealed(output, body)
    validate_dynamic_exclusion(output)
    return output.resolve()


def _cumulative_dynamic_exclusions(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        candidates: Sequence[Any] = ()
        if entry.get("event") == "final-gate-recorded" and entry.get("passed") is False:
            candidates = entry.get("dynamic_exclusions", ())
        elif entry.get("event") == "live-window-recorded" and entry.get("passed") is False:
            item = entry.get("dynamic_exclusion")
            candidates = () if item is None else (item,)
        for record in candidates:
            path = _verify_dynamic_exclusion_record(
                record, "cumulative"
            )
            normalized = _dynamic_exclusion_record(path)
            records[str(normalized["sha256"])] = normalized
    return [records[digest] for digest in sorted(records)]


def load_ledger(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = _ledger_path(plan)
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ChallengerError("attempt ledger is redirected or irregular")
    entries = []
    previous = "0" * 64
    active_attempt = -1
    materialized: dict[int, set[str]] = {}
    progress: dict[tuple[int, str], dict[str, int]] = {}
    final_results: dict[int, set[str]] = {}
    dual_authorized: set[int] = set()
    dual_prepared: set[int] = set()
    upload_count = 0
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ChallengerError("attempt ledger contains an irregular entry")
        value = qualification.load_sealed(path, LEDGER_SCHEMA)
        sequence = len(entries)
        if (
            path.name != f"{sequence:06d}-{value['body_sha256']}.json"
            or value.get("sequence") != sequence
            or value.get("previous_sha256") != previous
            or value.get("campaign_plan")
            != _sealed_record(pathlib.Path(plan["outputs"]["plan"]), PLAN_SCHEMA)
            or value.get("event") not in LEDGER_EVENTS
            or isinstance(value.get("attempt"), bool)
            or not isinstance(value.get("attempt"), int)
            or value["attempt"] < 0
        ):
            raise ChallengerError("attempt ledger chain changed")
        utc(value.get("created_at_utc"), "attempt event timestamp")
        if entries and entries[-1]["event"] in TERMINAL_EVENTS:
            raise ChallengerError("attempt ledger continues after terminal event")
        event = value["event"]
        attempt = value["attempt"]
        if event == "attempt-opened":
            if attempt != active_attempt + 1:
                raise ChallengerError("attempt numbers are not monotonic successors")
            if active_attempt >= 0 and (
                not entries
                or entries[-1].get("adaptation_route") not in {
                    "open-next-attempt-teacher-refresh",
                    "open-next-attempt-same-contract",
                    "open-next-attempt-targeted-retry",
                    "open-next-attempt-attribution-adaptation",
                    "open-next-attempt-protected-rejection",
                    "open-next-attempt-explicit-after-live-failure",
                }
            ):
                raise ChallengerError("new attempt lacks its deterministic route")
            active_attempt = attempt
            materialized[attempt] = set()
            final_results[attempt] = set()
            if (
                value.get("architecture") != ARCHITECTURE
                or value.get("policy_head") is not False
                or not isinstance(value.get("hypothesis"), str)
                or not value["hypothesis"].strip()
                or not isinstance(value.get("intervention"), str)
                or (
                    attempt == 0
                    and (
                        value.get("parent_attempt") is not None
                        or value.get("route") != "await-attempt-zero-result"
                        or value.get("intervention") != "external-recovery"
                        or value.get("attribution_receipt") is not None
                    )
                )
                or (
                    attempt > 0
                    and (
                        value.get("parent_attempt") != attempt - 1
                        or value.get("parent_event_sha256")
                        != entries[-1]["body_sha256"]
                        or value.get("route")
                        != entries[-1].get("adaptation_route")
                    )
                )
            ):
                raise ChallengerError("attempt-opened event changed")
            _validate_attempt_inputs(
                value.get("attempt_inputs"), plan=plan,
                require_student_12x8=True,
            )
            expected_dynamic = _cumulative_dynamic_exclusions(entries)
            if value.get("dynamic_exclusions") != expected_dynamic:
                raise ChallengerError(
                    "attempt-opened dynamic exclusion accumulation changed"
                )
            for ordinal, record in enumerate(expected_dynamic):
                _verify_dynamic_exclusion_record(
                    record, f"attempt dynamic exclusion {ordinal}"
                )
            if attempt > 0:
                route = str(value["route"])
                expected = {
                    "open-next-attempt-teacher-refresh": "teacher-refresh",
                    "open-next-attempt-same-contract": "same-contract",
                    "open-next-attempt-targeted-retry": "targeted-retry",
                    "open-next-attempt-protected-rejection": "protected-rejection-clean-restart",
                    "open-next-attempt-explicit-after-live-failure": "explicit-post-live-restart",
                }
                if route == "open-next-attempt-attribution-adaptation":
                    receipt_path = _verify_sealed_record(
                        value.get("attribution_receipt"),
                        ATTRIBUTION_EVIDENCE_SCHEMA,
                        "attempt attribution receipt",
                    )
                    receipt = qualification.load_sealed(
                        receipt_path, ATTRIBUTION_EVIDENCE_SCHEMA
                    )
                    classification = receipt.get("classification")
                    if (
                        classification not in ATTRIBUTION_INTERVENTIONS
                        or receipt.get("completed_no_improvement_attempts") != 2
                        or receipt.get("next_attempt") != attempt
                        or receipt.get("selected_intervention")
                        != value.get("intervention")
                        or value.get("intervention")
                        != ATTRIBUTION_INTERVENTIONS[classification]
                    ):
                        raise ChallengerError("attempt attribution selection changed")
                elif (
                    value.get("attribution_receipt") is not None
                    or value.get("intervention") != expected.get(route)
                ):
                    raise ChallengerError("attempt intervention route changed")
        elif attempt != active_attempt:
            raise ChallengerError("ledger event does not belong to the active attempt")
        if event == "phase-plan-materialized":
            phase = value.get("phase")
            opened_event = next(
                item for item in entries
                if item.get("event") == "attempt-opened"
                and item.get("attempt") == attempt
            )
            if (
                phase not in PHASE_QUOTAS
                or phase in materialized[attempt]
                or value.get("games") != PHASE_TOTALS[phase]
                or value.get("quotas") != PHASE_QUOTAS[phase]
                or value.get("games_launched") != 0
                or value.get("dynamic_exclusions")
                != opened_event.get("dynamic_exclusions")
            ):
                raise ChallengerError("phase materialization ledger event changed")
            _verify_sealed_record(
                value.get("phase_reference"), PHASE_REFERENCE_SCHEMA,
                "phase materialization reference",
            )
            expected_phase_inputs = dict(opened_event["attempt_inputs"])
            if phase == "full":
                prior_pilot = entries[-1] if entries else {}
                if (
                    prior_pilot.get("event") != "attempt-outcome-recorded"
                    or prior_pilot.get("phase") != "pilot"
                    or prior_pilot.get("admitted") is not True
                ):
                    raise ChallengerError("full phase lacks admitted pilot inputs")
                expected_phase_inputs["student_runtime"] = prior_pilot["candidate"][
                    "runtime"
                ]
            if value.get("attempt_inputs") != expected_phase_inputs:
                raise ChallengerError("phase materialization inputs changed")
            materialized[attempt].add(str(phase))
        elif event == "progress-recorded":
            phase = value.get("phase")
            if phase not in materialized[attempt]:
                raise ChallengerError("progress predates phase materialization")
            if any(
                item["event"] == "attempt-outcome-recorded"
                and item["attempt"] == attempt and item.get("phase") == phase
                for item in entries
            ):
                raise ChallengerError("progress continues after terminal phase outcome")
            progress_evidence = value.get("progress_evidence")
            production = qualification.load_sealed(
                pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
            ).get("production_allowlist_enforced") is True
            if production and not isinstance(progress_evidence, Mapping):
                raise ChallengerError("production progress lacks pipeline evidence")
            if progress_evidence is not None:
                if not isinstance(progress_evidence, Mapping) or set(
                    progress_evidence
                ) != {
                    "pipeline_plan", "games_receipt", "games_manifest",
                    "positions_receipt", "positions_manifest",
                    "finalized_receipt",
                }:
                    raise ChallengerError("progress evidence roster changed")
                _verify_record(
                    progress_evidence["pipeline_plan"], "progress pipeline plan"
                )
                for evidence_name in (
                    "games_receipt", "games_manifest", "positions_receipt",
                    "positions_manifest", "finalized_receipt",
                ):
                    record = progress_evidence[evidence_name]
                    if record is not None:
                        _verify_record(record, f"progress {evidence_name}")
            key = (attempt, str(phase))
            prior = progress.get(key, {
                "completed_games": 0,
                "completed_quotas": {
                    name: 0 for name in PHASE_QUOTAS[str(phase)]
                },
                "accepted_positions": 0,
                "progress_events": 0,
                "no_progress_events": 0,
                "consecutive_no_progress": 0,
            })
            completed = value.get("completed_games")
            completed_quotas = value.get("completed_quotas")
            accepted = value.get("accepted_positions")
            if (
                any(isinstance(item, bool) or not isinstance(item, int) for item in (completed, accepted))
                or completed < prior["completed_games"]
                or completed > PHASE_TOTALS[str(phase)]
                or accepted < prior["accepted_positions"]
                or not isinstance(completed_quotas, Mapping)
                or set(completed_quotas) != set(PHASE_QUOTAS[str(phase)])
                or any(
                    isinstance(count, bool) or not isinstance(count, int)
                    or count < prior["completed_quotas"][name]
                    or count > PHASE_QUOTAS[str(phase)][name]
                    for name, count in completed_quotas.items()
                )
                or sum(completed_quotas.values()) != completed
            ):
                raise ChallengerError("ledger progress counters are invalid")
            game_delta = completed - prior["completed_games"]
            position_delta = accepted - prior["accepted_positions"]
            made = game_delta + position_delta > 0
            consecutive = 0 if made else prior["consecutive_no_progress"] + 1
            expected_route = deterministic_route(
                phase=str(phase), completed_games=completed,
                target_games=PHASE_TOTALS[str(phase)],
                progress_delta=game_delta + position_delta,
                pipeline_complete=(
                    progress_evidence is None
                    or progress_evidence.get("finalized_receipt") is not None
                ),
            )
            expected = {
                "game_delta": game_delta,
                "position_delta": position_delta,
                "made_progress": made,
                "progress_events": prior["progress_events"] + int(made),
                "no_progress_events": prior["no_progress_events"] + int(not made),
                "consecutive_no_progress": consecutive,
                "adaptation_route": expected_route,
                "automatic_action": False,
            }
            if any(value.get(name) != expected_value for name, expected_value in expected.items()):
                raise ChallengerError("ledger progress/adaptation derivation changed")
            progress[key] = {
                "completed_games": completed,
                "completed_quotas": dict(completed_quotas),
                "accepted_positions": accepted,
                "progress_events": expected["progress_events"],
                "no_progress_events": expected["no_progress_events"],
                "consecutive_no_progress": consecutive,
            }
        elif event == "attempt-zero-result-recorded":
            if (
                attempt != 0
                or len(entries) != 1
                or value.get("passed") not in {True, False}
                or value.get("adaptation_route")
                != (
                    "prepare-dual-final"
                    if value.get("passed") is True
                    else "open-next-attempt-teacher-refresh"
                )
                or value.get("automatic_action") is not False
                or value.get("external_recovery_plan")
                != qualification.load_sealed(
                    pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
                )["attempt_zero"]["recovery_plan"]
            ):
                raise ChallengerError("attempt-zero result routing changed")
            candidate_identity = value.get("candidate_identity")
            if value.get("passed") is True:
                if (
                    not isinstance(candidate_identity, Mapping)
                    or set(candidate_identity) != {"runtime_sha256", "source_sha256"}
                    or any(re.fullmatch(r"[0-9a-f]{64}", str(item)) is None for item in candidate_identity.values())
                ):
                    raise ChallengerError("passing attempt-zero candidate identity changed")
            elif candidate_identity is not None:
                raise ChallengerError("rejected attempt-zero unexpectedly names a candidate")
            result_record = value.get("result")
            if not isinstance(result_record, Mapping):
                raise ChallengerError("attempt-zero result record is absent")
            result_path = _verify_sealed_record(
                result_record, str(result_record.get("schema", "")),
                "attempt-zero result",
            )
            try:
                result_path.relative_to(pathlib.Path(plan["outputs"]["root"]))
            except ValueError as error:
                raise ChallengerError("attempt-zero result escaped campaign bundle") from error
            result_value = qualification.load_sealed(result_path)
            expected_plan = qualification.load_sealed(
                pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
            )["attempt_zero"]["recovery_plan_binding"]
            recovery_plan_record = result_value.get("recovery_plan")
            if (
                not isinstance(recovery_plan_record, Mapping)
                or recovery_plan_record.get("bytes") != expected_plan["bytes"]
                or recovery_plan_record.get("sha256") != expected_plan["sha256"]
                or recovery_plan_record.get("schema") != RECOVERY_PLAN_SCHEMA
                or recovery_plan_record.get("body_sha256")
                != expected_plan["body_sha256"]
            ):
                raise ChallengerError("attempt-zero result lost its recovery-plan binding")
            if value.get("passed") is True:
                finalist_path = _verify_sealed_record(
                    value.get("referenced_finalist"), RECOVERY_FINALIST_SCHEMA,
                    "copied attempt-zero finalist",
                )
                recovery_result_path = _verify_sealed_record(
                    value.get("referenced_recovery_result"),
                    RECOVERY_RESULT_SCHEMA,
                    "copied attempt-zero recovery result",
                )
                finalist = qualification.load_sealed(
                    finalist_path, RECOVERY_FINALIST_SCHEMA
                )
                recovery_result = qualification.load_sealed(
                    recovery_result_path, RECOVERY_RESULT_SCHEMA
                )
                if (
                    result_value.get("schema") != RECOVERY_FINALIST_REFERENCE_SCHEMA
                    or finalist.get("recovery_plan") != recovery_plan_record
                    or recovery_result.get("recovery_plan") != recovery_plan_record
                    or finalist.get("recovery_result", {}).get("sha256")
                    != value["referenced_recovery_result"]["sha256"]
                ):
                    raise ChallengerError("copied attempt-zero evidence closure changed")
            elif (
                result_value.get("schema") != RECOVERY_JOURNAL_SCHEMA
                or result_value.get("event") != "terminal-failure"
                or result_value.get("no_retry") is not True
                or value.get("referenced_finalist") is not None
                or value.get("referenced_recovery_result") is not None
            ):
                raise ChallengerError("rejected attempt-zero evidence changed")
        elif event == "attempt-outcome-recorded":
            phase = value.get("phase")
            prior = entries[-1] if entries else {}
            if (
                attempt <= 0
                or phase not in PHASE_QUOTAS
                or prior.get("event") != "progress-recorded"
                or prior.get("phase") != phase
                or prior.get("completed_games") != PHASE_TOTALS[str(phase)]
                or prior.get("adaptation_route") != "record-attempt-outcome"
                or not isinstance(value.get("metrics"), Mapping)
            ):
                raise ChallengerError("attempt outcome chronology changed")
            admitted = _phase_admission(str(phase), value["metrics"])
            strength = value.get("strength_delta_pp")
            regret = value.get("teacher_regret_reduction_fraction")
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) for item in (strength, regret)
            ):
                raise ChallengerError("attempt improvement metrics changed")
            if (
                value["metrics"].get("strength_delta_pp") != float(strength)
                or value["metrics"].get("teacher_regret_reduction_fraction")
                != float(regret)
            ):
                raise ChallengerError("attempt improvement metrics lost evidence binding")
            phase_improved = float(strength) >= 1.5 or float(regret) >= 0.10
            attempt_improved = phase_improved or any(
                item["event"] == "attempt-outcome-recorded"
                and item["attempt"] == attempt and item.get("improved") is True
                for item in entries
            )
            prior_no_improvement = _completed_no_improvement_streak(entries)
            consecutive = (
                prior_no_improvement if admitted else
                0 if attempt_improved else prior_no_improvement + 1
            )
            route = (
                ("materialize-full" if phase == "pilot" else "prepare-dual-final")
                if admitted else
                "open-next-attempt-targeted-retry" if attempt_improved else
                "open-next-attempt-attribution-adaptation" if consecutive == 2 else
                "open-next-attempt-same-contract"
            )
            if (
                value.get("admitted") is not admitted
                or value.get("phase_improved") is not phase_improved
                or value.get("improved") is not attempt_improved
                or value.get("consecutive_no_improvement") != consecutive
                or value.get("adaptation_route") != route
                or value.get("automatic_action") is not False
            ):
                raise ChallengerError("attempt outcome/adaptation derivation changed")
            candidate = value.get("candidate")
            if not isinstance(candidate, Mapping):
                raise ChallengerError("attempt outcome candidate is absent")
            runtime_path = _verify_record(candidate.get("runtime"), "attempt outcome runtime")
            source_path = _verify_record(candidate.get("source"), "attempt outcome source")
            if (
                candidate.get("architecture") != _architecture(runtime_path)
                or source_path.is_symlink()
                or not 0 < source_path.stat().st_size < SOURCE_LIMIT
            ):
                raise ChallengerError("attempt outcome candidate changed")
            receipt_path = _verify_record(
                value.get("outcome_receipt"), "attempt outcome receipt"
            )
            receipt_value = qualification.load_sealed(
                receipt_path, PHASE_OUTCOME_EVIDENCE_SCHEMA
            )
            exclusion_path = _verify_sealed_record(
                value.get("development_exclusion"), DEVELOPMENT_EXCLUSION_SCHEMA,
                "attempt development exclusion",
            )
            exclusion = _validate_development_exclusion(
                exclusion_path, attempt=attempt, phase=str(phase)
            )
            phase_reference = (
                pathlib.Path(plan["outputs"]["phase_plans"])
                / f"attempt-{attempt:03d}/{phase}/phase-reference.json"
            )
            phase_state = validate_phase_reference(
                phase_reference, plan, ledger_entries=entries
            )
            if (
                value.get("outcome_receipt_schema")
                != PHASE_OUTCOME_EVIDENCE_SCHEMA
                or receipt_value.get("campaign_id") != CAMPAIGN_ID
                or receipt_value.get("attempt") != attempt
                or receipt_value.get("phase") != phase
                or receipt_value.get("status") != "complete"
                or receipt_value.get("phase_reference")
                != _sealed_record(phase_reference, PHASE_REFERENCE_SCHEMA)
                or receipt_value.get("schedule") != _regular(phase_state["schedule"])
                or receipt_value.get("candidate") != {
                    "runtime_sha256": candidate["runtime"]["sha256"],
                    "source_sha256": candidate["source"]["sha256"],
                }
                or receipt_value.get("completed_games") != PHASE_TOTALS[str(phase)]
                or receipt_value.get("completed_quotas") != PHASE_QUOTAS[str(phase)]
                or receipt_value.get("metrics_sha256")
                != sha256_bytes(canonical_json_bytes(dict(value["metrics"])))
                or receipt_value.get("development_exclusion", {}).get("sha256")
                != value["development_exclusion"]["sha256"]
                or receipt_value.get("development_exclusion", {}).get("body_sha256")
                != exclusion["body_sha256"]
                or receipt_value.get("protected_or_live_metrics_read") is not False
                or receipt_value.get("all_games_finished") is not True
            ):
                raise ChallengerError("attempt outcome receipt schema changed")
            admission_record = value.get("admission_receipt")
            admission_path = None
            if admission_record is not None:
                admission_path = _verify_sealed_record(
                    admission_record,
                    "papersoccer.compact-value-bfm.rank4-teacher-phase-admission.v1",
                    "attempt admission receipt",
                )
            _validate_phase_outcome_evidence(
                receipt_path,
                context={
                    "plan": plan,
                    "inputs": qualification.load_sealed(
                        pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
                    ),
                },
                attempt=attempt,
                phase=str(phase),
                candidate=candidate,
                metrics=value["metrics"],
                development_exclusion=exclusion_path,
                admission_receipt=admission_path,
                ledger_entries=entries,
            )
            for bound_path in (
                runtime_path, source_path, receipt_path, exclusion_path,
            ):
                try:
                    bound_path.relative_to(pathlib.Path(plan["outputs"]["root"]))
                except ValueError as error:
                    raise ChallengerError("attempt outcome artifact escaped campaign") from error
        elif event == "dual-final-authorized":
            if (
                not entries
                or entries[-1].get("adaptation_route") != "prepare-dual-final"
                or attempt in dual_authorized
            ):
                raise ChallengerError("dual final lacks completed candidate-freeze route")
            _verify_sealed_record(
                value.get("authorization"), DUAL_FINAL_AUTHORIZATION_SCHEMA,
                "dual-final authorization ledger record",
            )
            if value.get("banks_materialized") != 0 or value.get("games_launched") != 0:
                raise ChallengerError("dual-final authorization opened evidence early")
            dual_authorized.add(attempt)
        elif event == "dual-final-prepared":
            if (
                not entries or entries[-1].get("event") != "dual-final-authorized"
                or attempt not in dual_authorized
            ):
                raise ChallengerError("dual final lacks prior candidate authorization")
            if attempt in dual_prepared:
                raise ChallengerError("dual final was prepared more than once")
            exclusions = value.get("dynamic_exclusions")
            if not isinstance(exclusions, list) or len(exclusions) != 2:
                raise ChallengerError("dual final sanitized exclusions are absent")
            for ordinal, record in enumerate(exclusions):
                exclusion_path = _verify_dynamic_exclusion_record(
                    record, f"prepared dynamic exclusion {ordinal}"
                )
                exclusion = validate_dynamic_exclusion(exclusion_path)
                if (
                    exclusion.get("attempt") != attempt
                    or exclusion.get("gate_id") != ("gate-a", "gate-b")[ordinal]
                ):
                    raise ChallengerError("prepared dynamic exclusion identity changed")
            dual_prepared.add(attempt)
        elif event == "final-gate-recorded":
            gate_id = value.get("gate_id")
            if (
                attempt not in dual_prepared
                or gate_id not in {"gate-a", "gate-b"}
                or gate_id in final_results[attempt]
                or (gate_id == "gate-b" and final_results[attempt] != {"gate-a"})
            ):
                raise ChallengerError("final gate ledger roster changed")
            result_path = _verify_sealed_record(
                value.get("result"), FINAL_RESULT_SCHEMA,
                f"{gate_id} ledger result",
            )
            result = qualification.load_sealed(result_path, FINAL_RESULT_SCHEMA)
            passed = result.get("passed") is True
            expected_route = (
                "run-gate-b" if gate_id == "gate-a" and passed else
                "complete-dual-final" if passed else
                "open-next-attempt-protected-rejection"
            )
            evidence_path = _verify_record(
                value.get("evidence"), f"{gate_id} ledger evidence"
            )
            if (
                value.get("passed") is not passed
                or value.get("adaptation_route") != expected_route
                or value.get("automatic_action") is not False
                or result.get("evidence") != value.get("evidence")
                or sha256_file(evidence_path) != value["evidence"]["sha256"]
                or value.get("dynamic_exclusions")
                != (
                    entries[-1].get("dynamic_exclusions", [])
                    if not passed and entries[-1].get("event") == "dual-final-prepared"
                    else (
                        next(
                            item.get("dynamic_exclusions", [])
                            for item in reversed(entries)
                            if item.get("event") == "dual-final-prepared"
                            and item.get("attempt") == attempt
                        )
                        if not passed else []
                    )
                )
            ):
                raise ChallengerError("final gate result route changed")
            if gate_id == "gate-b":
                prior_gate = entries[-1]
                if (
                    prior_gate.get("event") != "final-gate-recorded"
                    or prior_gate.get("gate_id") != "gate-a"
                    or prior_gate.get("passed") is not True
                ):
                    raise ChallengerError("gate B did not follow passing Gate A")
            final_results[attempt].add(str(gate_id))
        elif event == "dual-final-qualified":
            if (
                final_results[attempt] != {"gate-a", "gate-b"}
                or not entries
                or entries[-1].get("event") != "final-gate-recorded"
                or entries[-1].get("gate_id") != "gate-b"
                or entries[-1].get("passed") is not True
            ):
                raise ChallengerError("dual qualification predates two passing final gates")
            qualification_path = _verify_sealed_record(
                value.get("qualification"), DUAL_QUALIFICATION_SCHEMA,
                "dual qualification ledger record",
            )
            qualified = qualification.load_sealed(
                qualification_path, DUAL_QUALIFICATION_SCHEMA
            )
            if (
                value.get("candidate_unchanged") is not True
                or value.get("gates_passed") != 2
                or qualified.get("attempt") != attempt
                or qualified.get("candidate_unchanged") is not True
                or qualified.get("independent_banks") is not True
            ):
                raise ChallengerError("dual qualification ledger evidence changed")
        elif event == "upload-attested":
            prior = entries[-1] if entries else {}
            attestation_path = _verify_sealed_record(
                value.get("submission_attestation"),
                qualification.UPLOAD_EVENT_SCHEMA,
                "copied submission attestation",
            )
            authorization_path = _verify_sealed_record(
                value.get("upload_authorization"),
                qualification.UPLOAD_AUTH_SCHEMA,
                "copied upload authorization",
            )
            attestation = qualification.load_sealed(
                attestation_path, qualification.UPLOAD_EVENT_SCHEMA
            )
            authorization = qualification.load_sealed(
                authorization_path, qualification.UPLOAD_AUTH_SCHEMA
            )
            upload_count += 1
            if (
                prior.get("event") != "dual-final-qualified"
                or value.get("qualification") != prior.get("qualification")
                or value.get("candidate")
                != qualification.load_sealed(
                    pathlib.Path(prior["qualification"]["path"]),
                    DUAL_QUALIFICATION_SCHEMA,
                ).get("candidate")
                or value.get("upload_ordinal") != upload_count
                or value.get("submit_clicks") != 1
                or value.get("adaptation_route") != "record-live-window"
                or value.get("automatic_action") is not False
                or attestation.get("status") != "submission-attested"
                or attestation.get("submit_clicks") != 1
                or attestation.get("agent_id") != value.get("agent_id")
                or attestation.get("submission_id") != value.get("submission_id")
                or attestation.get("candidate_commit")
                != value.get("candidate_commit")
                or attestation.get("source_sha256")
                != value.get("candidate", {}).get("source", {}).get("sha256")
                or authorization.get("candidate", {}).get("sha256")
                != attestation.get("source_sha256")
                or authorization.get("uploads_authorized") != 1
                or value.get("source_submission_attestation", {}).get("sha256")
                != value["submission_attestation"]["sha256"]
            ):
                raise ChallengerError("upload attestation ledger transition changed")
        elif event == "live-window-recorded":
            prior = entries[-1] if entries else {}
            _verify_record(value.get("live_reference"), "copied live reference")
            _verify_record(value.get("live_receipt"), "copied live receipt")
            passed = value.get("passed") is True
            if (
                prior.get("event") != "upload-attested"
                or value.get("upload_ordinal") != prior.get("upload_ordinal")
                or value.get("submission_id") != prior.get("submission_id")
                or value.get("candidate") != prior.get("candidate")
                or value.get("exact_games") != 90
                or value.get("training_eligible") is not False
                or value.get("automatic_action") is not False
                or (passed and value.get("dynamic_exclusion") is not None)
                or (not passed and value.get("dynamic_exclusion") is None)
                or value.get("adaptation_route")
                != (
                    "complete-campaign" if passed
                    else "await-explicit-additional-upload-authorization"
                )
                or (
                    passed and (
                        value.get("status") != "complete-accepted-diagnostic"
                        or value.get("focus_operational_failure_games") != 0
                    )
                )
            ):
                raise ChallengerError("live-window ledger transition changed")
            if not passed:
                exclusion_path = _verify_dynamic_exclusion_record(
                    value["dynamic_exclusion"], "rejected live dynamic exclusion"
                )
                exclusion = validate_dynamic_exclusion(exclusion_path)
                if (
                    exclusion.get("attempt") != attempt
                    or exclusion.get("gate_id")
                    != f"live-upload-{value['upload_ordinal']}"
                ):
                    raise ChallengerError("rejected live dynamic exclusion identity changed")
        elif event == "additional-upload-authorized":
            prior = entries[-1] if entries else {}
            _verify_sealed_record(
                value.get("authorization"),
                ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA,
                "additional upload authorization",
            )
            if (
                prior.get("event") != "live-window-recorded"
                or prior.get("passed") is not False
                or value.get("next_attempt") != attempt + 1
                or value.get("next_upload_ordinal") != upload_count + 1
                or value.get("consumed") is not False
                or value.get("adaptation_route")
                != "open-next-attempt-explicit-after-live-failure"
                or value.get("automatic_action") is not False
            ):
                raise ChallengerError("additional upload authorization route changed")
        elif event == "campaign-complete":
            prior = entries[-1] if entries else {}
            completion_path = _verify_sealed_record(
                value.get("completion"), CAMPAIGN_COMPLETION_SCHEMA,
                "campaign completion",
            )
            completion = qualification.load_sealed(
                completion_path, CAMPAIGN_COMPLETION_SCHEMA
            )
            if (
                prior.get("event") != "live-window-recorded"
                or prior.get("passed") is not True
                or value.get("candidate") != prior.get("candidate")
                or value.get("strict_final_gates") != 2
                or value.get("exact_live_games") != 90
                or value.get("focus_operational_failure_games") != 0
                or value.get("uploads_completed") != upload_count
                or value.get("goal_achieved") is not True
                or completion.get("candidate") != prior.get("candidate")
                or completion.get("proof", {}).get("strict_final_gates") != 2
                or completion.get("proof", {}).get("exact_live_games") != 90
                or completion.get("proof", {}).get(
                    "focus_operational_failure_games"
                ) != 0
            ):
                raise ChallengerError("campaign completion ledger evidence changed")
        entries.append(value)
        previous = value["body_sha256"]
    return entries


def _append_event(
    context: Mapping[str, Any], *, attempt: int, event: str,
    created_at_utc: str, fields: Mapping[str, Any],
) -> dict[str, Any]:
    if event not in LEDGER_EVENTS:
        raise ChallengerError("unknown attempt ledger event")
    created = utc(created_at_utc, "attempt event timestamp")
    plan = context["plan"] if "plan" in context else context
    entries = load_ledger(plan)
    if entries and entries[-1]["event"] in TERMINAL_EVENTS:
        raise ChallengerError("attempt ledger is terminal")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ChallengerError("attempt number is outside the frozen range")
    active = max((entry["attempt"] for entry in entries), default=-1)
    if (
        (event == "attempt-opened" and attempt != active + 1)
        or (event != "attempt-opened" and attempt != active)
    ):
        raise ChallengerError("event does not belong to the monotonic active attempt")
    body = {
        "schema": LEDGER_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "campaign_plan": _sealed_record(
            pathlib.Path(plan["outputs"]["plan"]), PLAN_SCHEMA
        ),
        "sequence": len(entries),
        "previous_sha256": entries[-1]["body_sha256"] if entries else "0" * 64,
        "attempt": attempt,
        "event": event,
        "created_at_utc": created,
        **dict(fields),
    }
    artifact = qualification.seal(body)
    raw = canonical_json_bytes(artifact)
    path = _ledger_path(plan) / f"{len(entries):06d}-{artifact['body_sha256']}.json"
    qualification.atomic_write_once(path, raw)
    return artifact


def _attempt_open(entries: Sequence[Mapping[str, Any]], attempt: int) -> bool:
    return any(
        entry["event"] == "attempt-opened" and entry["attempt"] == attempt
        for entry in entries
    )


def _phase_rows(plan_sha: str, attempt: int, phase: str) -> list[dict[str, Any]]:
    quotas = PHASE_QUOTAS[phase]
    slots = [
        (mode, ordinal)
        for mode, count in quotas.items()
        for ordinal in range(count)
    ]
    domain = f"{plan_sha}:{attempt}:{phase}:rank4-teacher-challenger-v1"
    slots.sort(
        key=lambda item: (
            hashlib.sha256(f"{domain}:{item[0]}:{item[1]}".encode("ascii")).digest(),
            item,
        )
    )
    rows = []
    for game_ordinal, (mode, quota_ordinal) in enumerate(slots):
        digest = hashlib.sha256(
            f"{domain}:game:{game_ordinal}:{mode}:{quota_ordinal}".encode("ascii")
        ).digest()
        rows.append({
            "game_ordinal": game_ordinal,
            "game_id": f"attempt-{attempt:03d}-{phase}-{game_ordinal:05d}",
            "actor_mode": mode,
            "quota_ordinal": quota_ordinal,
            "base_seed": int.from_bytes(digest[:8], "big"),
            "worker": game_ordinal % RESOURCE_LIMITS["logical_game_shards"],
        })
    return rows


def _render_phase_tsv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = ["game_ordinal\tactor_mode\tbase_seed"]
    lines.extend(
        "\t".join(str(row[key]) for key in (
            "game_ordinal", "actor_mode", "base_seed"
        ))
        for row in rows
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def materialize_phase(
    plan_path: pathlib.Path, *, attempt: int, phase: str, created_at_utc: str,
) -> pathlib.Path:
    context = validate_campaign(plan_path)
    if phase not in PHASE_QUOTAS:
        raise ChallengerError("phase must be pilot or full")
    if attempt == ATTEMPT_ZERO:
        raise ChallengerError("attempt zero is external and cannot materialize a new phase")
    entries = load_ledger(context["plan"])
    if not _attempt_open(entries, attempt) or attempt != max(entry["attempt"] for entry in entries):
        raise ChallengerError("phase belongs to an unopened attempt")
    if phase == "full" and not (
        entries[-1].get("event") == "attempt-outcome-recorded"
        and entries[-1].get("attempt") == attempt
        and entries[-1].get("phase") == "pilot"
        and entries[-1].get("admitted") is True
        and entries[-1].get("adaptation_route") == "materialize-full"
    ):
        raise ChallengerError("pilot admission did not authorize the full phase")
    opened = next(
        entry for entry in entries
        if entry.get("event") == "attempt-opened" and entry.get("attempt") == attempt
    )
    attempt_inputs = dict(opened["attempt_inputs"])
    if phase == "full":
        attempt_inputs["student_runtime"] = entries[-1]["candidate"]["runtime"]
    _validate_attempt_inputs(
        attempt_inputs, plan=context["plan"], require_student_12x8=True
    )
    dynamic_exclusions = list(opened["dynamic_exclusions"])
    directory = pathlib.Path(context["plan"]["outputs"]["phase_plans"]) / f"attempt-{attempt:03d}" / phase
    reference_path = directory / "phase-reference.json"
    if reference_path.exists():
        reference = qualification.load_sealed(reference_path, PHASE_REFERENCE_SCHEMA)
        _verify_sealed_record(reference["phase_plan"], PHASE_SCHEMA, "phase plan")
        _verify_record(reference["schedule"], "phase schedule")
        return reference_path
    rows = _phase_rows(sha256_file(plan_path), attempt, phase)
    quotas = Counter(row["actor_mode"] for row in rows)
    workers = Counter(row["worker"] for row in rows)
    if (
        dict(quotas) != PHASE_QUOTAS[phase]
        or len(rows) != PHASE_TOTALS[phase]
        or set(workers) != set(range(10))
        or len(set(row["game_id"] for row in rows)) != len(rows)
        or len(set(row["base_seed"] for row in rows)) != len(rows)
    ):
        raise ChallengerError("deterministic phase materialization changed")
    body = {
        "schema": PHASE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "campaign_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "attempt": attempt,
        "phase": phase,
        "status": "materialized-not-launched",
        "created_at_utc": utc(created_at_utc, "phase materialization timestamp"),
        "games": len(rows),
        "quota_multiplier": 1 if phase == "pilot" else 5,
        "quotas": PHASE_QUOTAS[phase],
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "architecture": context["plan"]["architecture"],
        "actor_bindings": {
            "incumbent": context["inputs"]["teacher"],
            "incumbent_role": "accepted-f7bdb201-rank4-teacher",
            "student_runtime": attempt_inputs["student_runtime"],
            "prior_incumbent": attempt_inputs["prior_runtime"],
        },
        "attempt_inputs": attempt_inputs,
        "producer_binaries": attempt_inputs["producer_binaries"],
        "dynamic_exclusions": dynamic_exclusions,
        "protected_exclusions": context["inputs"]["protected_exclusions"],
        "locked_training_bundle_tests": context["inputs"]["training_bundle"]["protected_test_artifacts"],
        "live_exclusions": context["inputs"]["live_exclusions"],
        "resources": RESOURCE_LIMITS,
        "launch_authorized": True,
        "automatic_launch": False,
    }
    phase_path, _phase = _write_content_addressed(directory, body, ".phase-plan.json")
    schedule = _render_phase_tsv(rows)
    schedule_path = directory / f"{sha256_bytes(schedule)}.tsv"
    qualification.atomic_write_once(schedule_path, schedule)
    qualification.write_sealed(reference_path, {
        "schema": PHASE_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "phase": phase,
        "phase_plan": _sealed_record(phase_path, PHASE_SCHEMA),
        "schedule": _regular(schedule_path),
    })
    _append_event(
        context,
        attempt=attempt,
        event="phase-plan-materialized",
        created_at_utc=created_at_utc,
        fields={
            "phase": phase,
            "games": len(rows),
            "quotas": PHASE_QUOTAS[phase],
            "phase_reference": _sealed_record(reference_path, PHASE_REFERENCE_SCHEMA),
            "attempt_inputs": attempt_inputs,
            "dynamic_exclusions": dynamic_exclusions,
            "games_launched": 0,
        },
    )
    return reference_path


def validate_phase_reference(
    path: pathlib.Path, plan: Mapping[str, Any], *,
    ledger_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    reference = qualification.load_sealed(path, PHASE_REFERENCE_SCHEMA)
    phase_path = _verify_sealed_record(reference.get("phase_plan"), PHASE_SCHEMA, "phase plan")
    schedule_path = _verify_record(reference.get("schedule"), "phase schedule")
    phase = qualification.load_sealed(phase_path, PHASE_SCHEMA)
    name = phase.get("phase")
    attempt = phase.get("attempt")
    if name not in PHASE_QUOTAS or not isinstance(attempt, int):
        raise ChallengerError("phase identity changed")
    expected_directory = pathlib.Path(plan["outputs"]["phase_plans"]) / f"attempt-{attempt:03d}" / name
    if (
        path.is_symlink()
        or path.resolve() != expected_directory / "phase-reference.json"
        or phase_path.parent != expected_directory
        or phase_path.name != f"{sha256_file(phase_path)}.phase-plan.json"
        or schedule_path.parent != expected_directory
        or schedule_path.name != f"{sha256_file(schedule_path)}.tsv"
        or set(reference) != {
            "schema", "namespace", "campaign_id", "attempt", "phase",
            "phase_plan", "schedule", "body_sha256",
        }
        or set(phase) != {
            "schema", "namespace", "campaign_id", "campaign_plan", "attempt",
            "phase", "status", "created_at_utc", "games", "quota_multiplier",
            "quotas", "rows", "rows_sha256", "architecture", "actor_bindings",
            "attempt_inputs", "producer_binaries", "dynamic_exclusions",
            "protected_exclusions", "locked_training_bundle_tests",
            "live_exclusions", "resources", "launch_authorized",
            "automatic_launch", "body_sha256",
        }
    ):
        raise ChallengerError("phase plan/reference path or fields changed")
    expected_rows = _phase_rows(sha256_file(pathlib.Path(plan["outputs"]["plan"])), attempt, name)
    ledger = list(ledger_entries) if ledger_entries is not None else load_ledger(plan)
    opened = next(
        entry for entry in ledger
        if entry.get("event") == "attempt-opened" and entry.get("attempt") == attempt
    )
    expected_attempt_inputs = dict(opened["attempt_inputs"])
    if name == "full":
        pilot = next(
            entry for entry in ledger
            if entry.get("event") == "attempt-outcome-recorded"
            and entry.get("attempt") == attempt
            and entry.get("phase") == "pilot"
            and entry.get("admitted") is True
        )
        expected_attempt_inputs["student_runtime"] = pilot["candidate"]["runtime"]
    _validate_attempt_inputs(
        phase.get("attempt_inputs"), plan=plan, require_student_12x8=True
    )
    if (
        phase.get("campaign_plan") != _sealed_record(pathlib.Path(plan["outputs"]["plan"]), PLAN_SCHEMA)
        or phase.get("rows") != expected_rows
        or phase.get("rows_sha256") != sha256_bytes(canonical_json_bytes(expected_rows))
        or phase.get("quotas") != PHASE_QUOTAS[name]
        or phase.get("games") != PHASE_TOTALS[name]
        or phase.get("status") != "materialized-not-launched"
        or phase.get("quota_multiplier") != (1 if name == "pilot" else 5)
        or phase.get("architecture") != plan["architecture"]
        or phase.get("attempt_inputs") != expected_attempt_inputs
        or phase.get("dynamic_exclusions") != opened.get("dynamic_exclusions")
        or phase.get("producer_binaries")
        != expected_attempt_inputs["producer_binaries"]
        or phase.get("protected_exclusions")
        != qualification.load_sealed(
            pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
        )["protected_exclusions"]
        or phase.get("live_exclusions")
        != qualification.load_sealed(
            pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
        )["live_exclusions"]
        or phase.get("resources") != RESOURCE_LIMITS
        or phase.get("launch_authorized") is not True
        or phase.get("automatic_launch") is not False
        or phase.get("actor_bindings") != {
            "incumbent": qualification.load_sealed(
                pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
            )["teacher"],
            "incumbent_role": "accepted-f7bdb201-rank4-teacher",
            "student_runtime": expected_attempt_inputs["student_runtime"],
            "prior_incumbent": expected_attempt_inputs["prior_runtime"],
        }
        or phase.get("locked_training_bundle_tests")
        != qualification.load_sealed(
            pathlib.Path(plan["inputs"]["path"]), INPUT_SCHEMA
        )["training_bundle"]["protected_test_artifacts"]
        or schedule_path.read_bytes() != _render_phase_tsv(expected_rows)
    ):
        raise ChallengerError("phase plan/schedule changed")
    utc(phase.get("created_at_utc"), "phase plan timestamp")
    return {"reference": reference, "phase": phase, "path": phase_path, "schedule": schedule_path}


def _latest_progress(entries: Sequence[Mapping[str, Any]], attempt: int, phase: str) -> dict[str, Any]:
    matches = [
        entry for entry in entries
        if entry["event"] == "progress-recorded"
        and entry["attempt"] == attempt and entry.get("phase") == phase
    ]
    if not matches:
        return {
            "completed_games": 0,
            "completed_quotas": {name: 0 for name in PHASE_QUOTAS[phase]},
            "accepted_positions": 0,
            "progress_events": 0,
            "no_progress_events": 0,
            "consecutive_no_progress": 0,
            "adaptation_route": None,
        }
    return {key: matches[-1][key] for key in (
        "completed_games", "completed_quotas", "accepted_positions", "progress_events",
        "no_progress_events", "consecutive_no_progress",
        "adaptation_route",
    )}


def deterministic_route(
    *, phase: str, completed_games: int, target_games: int,
    progress_delta: int, pipeline_complete: bool = True,
) -> str:
    if completed_games == target_games and pipeline_complete:
        return "record-attempt-outcome"
    if progress_delta > 0:
        return "continue-current-phase"
    return "explicit-resume-current-phase"


def _derive_pipeline_progress(
    context: Mapping[str, Any], *, attempt: int, phase: str,
    pipeline_plan_path: pathlib.Path,
) -> tuple[int, dict[str, int], int, dict[str, Any]]:
    try:
        from tools import compact_value_bfm_pilot_pipeline as phase_pipeline

        pipeline_plan = phase_pipeline.load_pipeline(pipeline_plan_path.resolve())
    except Exception as error:
        raise ChallengerError("phase progress pipeline failed validation") from error
    phase_reference = (
        pathlib.Path(context["plan"]["outputs"]["phase_plans"])
        / f"attempt-{attempt:03d}/{phase}/phase-reference.json"
    )
    if (
        pipeline_plan.get("campaign_id") != CAMPAIGN_ID
        or pipeline_plan.get("attempt") != attempt
        or pipeline_plan.get("phase") != phase
        or pipeline_plan.get("campaign_plan") != _regular(
            pathlib.Path(context["plan"]["outputs"]["plan"])
        )
        or pipeline_plan.get("phase_reference") != _regular(phase_reference)
    ):
        raise ChallengerError("phase progress pipeline belongs to another phase")
    completed_games = 0
    completed_quotas = {name: 0 for name in PHASE_QUOTAS[phase]}
    accepted_positions = 0
    evidence: dict[str, Any] = {
        "pipeline_plan": _regular(pipeline_plan_path.resolve()),
        "games_receipt": None,
        "games_manifest": None,
        "positions_receipt": None,
        "positions_manifest": None,
        "finalized_receipt": None,
    }
    games_receipt_path = (
        pathlib.Path(pipeline_plan["outputs"]["receipts"]) / "01-games.json"
    )
    if games_receipt_path.exists():
        receipt = phase_pipeline._load_sealed(
            games_receipt_path, phase_pipeline.STAGE_RECEIPT_SCHEMA,
            "phase games progress receipt",
        )
        manifest_path = pathlib.Path(pipeline_plan["outputs"]["games_manifest"])
        manifest = phase_pipeline._load_sealed(
            manifest_path, phase_pipeline.GAME_MANIFEST_SCHEMA,
            "phase games progress manifest",
        )
        rows = manifest.get("rows")
        if (
            receipt.get("pipeline_body_sha256") != pipeline_plan["body_sha256"]
            or receipt.get("stage") != "01-games"
            or not isinstance(rows, list)
        ):
            raise ChallengerError("phase games progress evidence changed")
        completed_games = len(rows)
        counts = Counter(str(row.get("actor_mode")) for row in rows)
        if set(counts) - set(completed_quotas):
            raise ChallengerError("phase games progress has an unknown actor mode")
        completed_quotas.update(counts)
        evidence["games_receipt"] = _regular(games_receipt_path)
        evidence["games_manifest"] = _regular(manifest_path)
    positions_receipt_path = (
        pathlib.Path(pipeline_plan["outputs"]["receipts"]) / "02-positions.json"
    )
    if positions_receipt_path.exists():
        receipt = phase_pipeline._load_sealed(
            positions_receipt_path, phase_pipeline.STAGE_RECEIPT_SCHEMA,
            "phase positions progress receipt",
        )
        manifest_path = pathlib.Path(pipeline_plan["outputs"]["positions_manifest"])
        manifest = phase_pipeline._load_sealed(
            manifest_path, phase_pipeline.POSITION_MANIFEST_SCHEMA,
            "phase positions progress manifest",
        )
        if (
            receipt.get("pipeline_body_sha256") != pipeline_plan["body_sha256"]
            or receipt.get("stage") != "02-positions"
            or manifest.get("pipeline_body_sha256") != pipeline_plan["body_sha256"]
            or isinstance(manifest.get("positions"), bool)
            or not isinstance(manifest.get("positions"), int)
        ):
            raise ChallengerError("phase position progress evidence changed")
        accepted_positions = int(manifest["positions"])
        evidence["positions_receipt"] = _regular(positions_receipt_path)
        evidence["positions_manifest"] = _regular(manifest_path)
    final_receipt_path = (
        pathlib.Path(pipeline_plan["outputs"]["receipts"])
        / "07-finalize-labels.json"
    )
    if final_receipt_path.exists():
        final_receipt = phase_pipeline._load_sealed(
            final_receipt_path, phase_pipeline.STAGE_RECEIPT_SCHEMA,
            "phase finalized progress receipt",
        )
        if (
            final_receipt.get("pipeline_body_sha256")
            != pipeline_plan["body_sha256"]
            or final_receipt.get("stage") != "07-finalize-labels"
            or final_receipt.get("details", {}).get("protected_tests_opened")
            is not False
        ):
            raise ChallengerError("phase finalized progress evidence changed")
        evidence["finalized_receipt"] = _regular(final_receipt_path)
    return completed_games, completed_quotas, accepted_positions, evidence


def record_progress(
    plan_path: pathlib.Path, *, attempt: int, phase: str,
    completed_games: int | None = None,
    completed_quotas: Mapping[str, int] | None = None,
    accepted_positions: int | None = None,
    pipeline_plan_path: pathlib.Path | None = None,
    created_at_utc: str,
) -> dict[str, Any]:
    context = validate_campaign(plan_path)
    if attempt == ATTEMPT_ZERO:
        raise ChallengerError("attempt zero progress belongs to the external recovery")
    if phase not in PHASE_QUOTAS:
        raise ChallengerError("progress phase changed")
    phase_reference = (
        pathlib.Path(context["plan"]["outputs"]["phase_plans"])
        / f"attempt-{attempt:03d}/{phase}/phase-reference.json"
    )
    validate_phase_reference(phase_reference, context["plan"])
    entries = load_ledger(context["plan"])
    progress_evidence = None
    if pipeline_plan_path is not None:
        (
            completed_games, completed_quotas, accepted_positions,
            progress_evidence,
        ) = _derive_pipeline_progress(
            context, attempt=attempt, phase=phase,
            pipeline_plan_path=pipeline_plan_path,
        )
    elif context["inputs"].get("production_allowlist_enforced") is True:
        raise ChallengerError("production progress must derive from pipeline receipts")
    previous = _latest_progress(entries, attempt, phase)
    if (
        previous.get("adaptation_route") == "record-attempt-outcome"
        or any(
            entry["event"] == "attempt-outcome-recorded"
            and entry["attempt"] == attempt and entry.get("phase") == phase
            for entry in entries
        )
    ):
        raise ChallengerError("terminal phase progress is already sealed")
    for value, name in ((completed_games, "completed games"), (accepted_positions, "accepted positions")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ChallengerError(f"{name} is invalid")
    if (
        completed_games < previous["completed_games"]
        or completed_games > PHASE_TOTALS[phase]
        or accepted_positions < previous["accepted_positions"]
        or not isinstance(completed_quotas, Mapping)
        or set(completed_quotas) != set(PHASE_QUOTAS[phase])
        or any(
            isinstance(count, bool) or not isinstance(count, int)
            or count < previous["completed_quotas"][name]
            or count > PHASE_QUOTAS[phase][name]
            for name, count in completed_quotas.items()
        )
        or sum(completed_quotas.values()) != completed_games
    ):
        raise ChallengerError("progress quotas regressed, disagree, or exceeded the phase")
    game_delta = completed_games - previous["completed_games"]
    position_delta = accepted_positions - previous["accepted_positions"]
    work_delta = game_delta + position_delta
    made_progress = work_delta > 0
    progress_events = previous["progress_events"] + int(made_progress)
    no_progress_events = previous["no_progress_events"] + int(not made_progress)
    consecutive = 0 if made_progress else previous["consecutive_no_progress"] + 1
    route = deterministic_route(
        phase=phase,
        completed_games=completed_games,
        target_games=PHASE_TOTALS[phase],
        progress_delta=work_delta,
        pipeline_complete=(
            progress_evidence is None
            or progress_evidence.get("finalized_receipt") is not None
        ),
    )
    return _append_event(
        context,
        attempt=attempt,
        event="progress-recorded",
        created_at_utc=created_at_utc,
        fields={
            "phase": phase,
            "completed_games": completed_games,
            "completed_quotas": dict(completed_quotas),
            "accepted_positions": accepted_positions,
            "game_delta": game_delta,
            "position_delta": position_delta,
            "made_progress": made_progress,
            "progress_events": progress_events,
            "no_progress_events": no_progress_events,
            "consecutive_no_progress": consecutive,
            "adaptation_route": route,
            "automatic_action": False,
            "progress_evidence": progress_evidence,
        },
    )


def record_attempt_zero_result(
    plan_path: pathlib.Path, *, result_path: pathlib.Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Record the immutable external attempt-zero outcome without launching it."""

    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    if len(entries) != 1 or entries[0].get("route") != "await-attempt-zero-result":
        raise ChallengerError("attempt-zero result was already recorded or ledger changed")
    try:
        value = qualification.load_sealed(result_path)
    except Exception as error:
        raise ChallengerError("attempt-zero result is not a sealed artifact") from error
    expected_plan = context["inputs"]["attempt_zero"]["recovery_plan_binding"]

    def recovery_plan_matches(record: Any) -> bool:
        return bool(
            isinstance(record, Mapping)
            and record.get("bytes") == expected_plan["bytes"]
            and record.get("sha256") == expected_plan["sha256"]
            and record.get("schema") == RECOVERY_PLAN_SCHEMA
            and record.get("body_sha256") == expected_plan["body_sha256"]
        )

    passed = bool(
        value.get("schema") == RECOVERY_FINALIST_REFERENCE_SCHEMA
        and value.get("complete") is True
        and recovery_plan_matches(value.get("recovery_plan"))
    )
    rejected = bool(
        value.get("schema") == RECOVERY_JOURNAL_SCHEMA
        and value.get("event") == "terminal-failure"
        and value.get("no_retry") is True
        and recovery_plan_matches(value.get("recovery_plan"))
    )
    if passed == rejected:
        raise ChallengerError("attempt-zero result has no unambiguous pass/reject outcome")
    candidate_identity = None
    finalist_path: pathlib.Path | None = None
    recovery_result_path: pathlib.Path | None = None
    finalist: Mapping[str, Any] | None = None
    recovery_result: Mapping[str, Any] | None = None
    if passed:
        finalist_record = value.get("finalist")
        recovery_result_record = value.get("recovery_result")
        finalist_path = _verify_sealed_record(
            finalist_record, RECOVERY_FINALIST_SCHEMA,
            "attempt-zero recovery finalist",
        )
        recovery_result_path = _verify_sealed_record(
            recovery_result_record, RECOVERY_RESULT_SCHEMA,
            "attempt-zero recovery result",
        )
        finalist = qualification.load_sealed(
            finalist_path, RECOVERY_FINALIST_SCHEMA
        )
        recovery_result = qualification.load_sealed(
            recovery_result_path, RECOVERY_RESULT_SCHEMA
        )
        if (
            finalist.get("recovery_plan") != value.get("recovery_plan")
            or finalist.get("recovery_result") != recovery_result_record
            or recovery_result.get("recovery_plan") != value.get("recovery_plan")
        ):
            raise ChallengerError("attempt-zero finalist/result lineage changed")
        candidate = finalist.get("candidate")
        if isinstance(candidate, Mapping):
            candidate_identity = {
                "runtime_sha256": candidate.get("runtime", {}).get("sha256"),
                "source_sha256": candidate.get("generated_source", {}).get("sha256"),
            }
        if (
            not isinstance(candidate_identity, Mapping)
            or set(candidate_identity) != {"runtime_sha256", "source_sha256"}
            or any(
                not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in candidate_identity.values()
            )
        ):
            raise ChallengerError("passing attempt-zero result omits candidate identity")
        frozen_candidate = context["inputs"]["candidate"]
        if candidate_identity != {
            "runtime_sha256": frozen_candidate["runtime"]["sha256"],
            "source_sha256": frozen_candidate["source"]["sha256"],
        }:
            raise ChallengerError("attempt-zero finalist changed the frozen candidate")
    artifact_root = (
        pathlib.Path(context["root"])
        / "attempt-ledger/artifacts/attempt-000"
    )
    copied_result = _copy_attempt_artifact(
        _regular(result_path),
        root=artifact_root,
    )
    copied_finalist = None
    copied_recovery_result = None
    if finalist_path is not None and recovery_result_path is not None:
        copied_finalist_record = _copy_attempt_artifact(
            _regular(finalist_path), root=artifact_root
        )
        copied_result_record = _copy_attempt_artifact(
            _regular(recovery_result_path), root=artifact_root
        )
        copied_finalist = {
            **copied_finalist_record,
            "schema": RECOVERY_FINALIST_SCHEMA,
            "body_sha256": finalist["body_sha256"],
        }
        copied_recovery_result = {
            **copied_result_record,
            "schema": RECOVERY_RESULT_SCHEMA,
            "body_sha256": recovery_result["body_sha256"],
        }
    return _append_event(
        context,
        attempt=ATTEMPT_ZERO,
        event="attempt-zero-result-recorded",
        created_at_utc=created_at_utc,
        fields={
            "external_recovery_plan": context["inputs"]["attempt_zero"]["recovery_plan"],
            "result": {
                **copied_result,
                "schema": str(value["schema"]),
                "body_sha256": value["body_sha256"],
            },
            "referenced_finalist": copied_finalist,
            "referenced_recovery_result": copied_recovery_result,
            "passed": passed,
            "candidate_identity": candidate_identity,
            "adaptation_route": (
                "prepare-dual-final"
                if passed else "open-next-attempt-teacher-refresh"
            ),
            "automatic_action": False,
        },
    )


def _zero_failures(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == set(qualification.FAILURE_CATEGORIES)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item == 0
            for item in value.values()
        )
    )


def _phase_admission(phase: str, metrics: Mapping[str, Any]) -> bool:
    validation_groups = metrics.get("ranking_validation_groups")
    comparable_groups = metrics.get("comparable_exhaustive_validation_groups")
    comparable_fraction = metrics.get("comparable_exhaustive_validation_fraction")
    coverage = bool(
        isinstance(validation_groups, int)
        and not isinstance(validation_groups, bool)
        and isinstance(comparable_groups, int)
        and not isinstance(comparable_groups, bool)
        and validation_groups >= MINIMUM_COMPARABLE_VALIDATION_GROUPS
        and comparable_groups >= MINIMUM_COMPARABLE_VALIDATION_GROUPS
        and comparable_groups <= validation_groups
        and isinstance(comparable_fraction, (int, float))
        and not isinstance(comparable_fraction, bool)
        and math.isfinite(float(comparable_fraction))
        and float(comparable_fraction)
        == comparable_groups / validation_groups
        and float(comparable_fraction) >= MINIMUM_COMPARABLE_VALIDATION_FRACTION
    )
    if not coverage:
        return False
    if phase == "pilot":
        screen = metrics.get("rank4_screen")
        regret = metrics.get("mean_teacher_regret_reduction_fraction")
        flip = metrics.get("quantized_action_flip_rate")
        control_flip = metrics.get("scalar_control_action_flip_rate")
        return bool(
            metrics.get("canonical_retention_passed") is True
            and metrics.get("candidate_quantized") is True
            and metrics.get("evaluation_classification")
            == "unseen-root-unprotected"
            and isinstance(regret, (int, float)) and not isinstance(regret, bool)
            and math.isfinite(float(regret)) and 0.10 <= float(regret) <= 1.0
            and isinstance(flip, (int, float)) and not isinstance(flip, bool)
            and isinstance(control_flip, (int, float)) and not isinstance(control_flip, bool)
            and math.isfinite(float(flip)) and math.isfinite(float(control_flip))
            and 0.0 <= float(flip) <= 1.0
            and 0.0 <= float(control_flip) <= 1.0
            and float(flip) <= float(control_flip) + 0.005
            and isinstance(screen, Mapping)
            and screen.get("classification") == "fresh-unprotected"
            and screen.get("pairs") == 100
            and screen.get("games") == 200
            and isinstance(screen.get("candidate_wins"), int)
            and not isinstance(screen.get("candidate_wins"), bool)
            and screen["candidate_wins"] >= 105
            and _zero_failures(screen.get("failures"))
        )
    if phase == "full":
        clock = metrics.get("actual_clock")
        if not isinstance(clock, Mapping):
            return False
        colors = clock.get("candidate_color_wins")
        lower = clock.get("paired_lower_95")
        confidence = (
            isinstance(lower, (int, float)) and not isinstance(lower, bool)
            and math.isfinite(float(lower)) and 0.5 < float(lower) <= 1.0
        )
        return bool(
            clock.get("classification") == "fresh-unprotected"
            and clock.get("pairs") == 500
            and clock.get("games") == 1_000
            and isinstance(clock.get("candidate_wins"), int)
            and not isinstance(clock.get("candidate_wins"), bool)
            and clock["candidate_wins"] >= 550
            and isinstance(colors, Mapping) and set(colors) == {"0", "1"}
            and all(
                isinstance(colors[color], int) and not isinstance(colors[color], bool)
                and colors[color] >= 265 for color in ("0", "1")
            )
            and sum(colors.values()) == clock["candidate_wins"]
            and confidence
            and _zero_failures(clock.get("failures"))
        )
    raise ChallengerError("unknown terminal phase outcome")


def _copy_attempt_artifact(
    record: Mapping[str, Any], *, root: pathlib.Path,
) -> dict[str, Any]:
    source = _verify_record(record, "attempt artifact source")
    target = root / f"{record['sha256']}{_bundle_suffix(source)}"
    qualification.atomic_write_once(target, source.read_bytes())
    return _regular(target)


def _validate_development_exclusion(
    path: pathlib.Path, *, attempt: int, phase: str,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, DEVELOPMENT_EXCLUSION_SCHEMA)
    fingerprints = value.get("fingerprints")
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != attempt
        or value.get("phase") != phase
        or value.get("classification") != "unprotected-development-fingerprints"
        or not isinstance(fingerprints, list)
        or not fingerprints
        or fingerprints != sorted(set(fingerprints))
        or any(SHA256_RE.fullmatch(str(item)) is None for item in fingerprints)
        or value.get("protected_or_live_data_included") is not False
    ):
        raise ChallengerError("development exclusion evidence changed")
    return value


def _validate_phase_outcome_evidence(
    path: pathlib.Path, *, context: Mapping[str, Any], attempt: int,
    phase: str, candidate: Mapping[str, Any], metrics: Mapping[str, Any],
    development_exclusion: pathlib.Path,
    admission_receipt: pathlib.Path | None = None,
    ledger_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, PHASE_OUTCOME_EVIDENCE_SCHEMA)
    phase_reference = (
        pathlib.Path(context["plan"]["outputs"]["phase_plans"])
        / f"attempt-{attempt:03d}/{phase}/phase-reference.json"
    )
    phase_state = validate_phase_reference(
        phase_reference, context["plan"], ledger_entries=ledger_entries
    )
    expected_candidate = {
        "runtime_sha256": candidate["runtime"]["sha256"],
        "source_sha256": candidate["source"]["sha256"],
    }
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != attempt
        or value.get("phase") != phase
        or value.get("status") != "complete"
        or value.get("phase_reference")
        != _sealed_record(phase_reference, PHASE_REFERENCE_SCHEMA)
        or value.get("schedule") != _regular(phase_state["schedule"])
        or value.get("candidate") != expected_candidate
        or value.get("completed_games") != PHASE_TOTALS[phase]
        or value.get("completed_quotas") != PHASE_QUOTAS[phase]
        or value.get("metrics_sha256")
        != sha256_bytes(canonical_json_bytes(dict(metrics)))
        or value.get("development_exclusion", {}).get("sha256")
        != sha256_file(development_exclusion)
        or value.get("development_exclusion", {}).get("body_sha256")
        != qualification.load_sealed(
            development_exclusion, DEVELOPMENT_EXCLUSION_SCHEMA
        )["body_sha256"]
        or value.get("protected_or_live_metrics_read") is not False
        or value.get("all_games_finished") is not True
    ):
        raise ChallengerError("phase outcome evidence is not schedule/candidate bound")
    closure = value.get("evidence_closure")
    production = context["inputs"].get("production_allowlist_enforced") is True
    if closure is None and not production:
        return value
    if not isinstance(closure, Mapping) or set(closure) != {
        "training_plan", "pipeline_plan", "finalized_pipeline_receipt",
        "training_selection", "gate_plan", "gate_results",
        "selected_candidate", "input_audit_sha256",
        "build_source_closure_sha256", "protected_tests_opened",
    }:
        raise ChallengerError("phase outcome evidence closure roster changed")
    effective_entries = (
        list(ledger_entries)
        if ledger_entries is not None else load_ledger(context["plan"])
    )
    latest_progress = next((
        entry for entry in reversed(effective_entries)
        if entry.get("event") == "progress-recorded"
        and entry.get("attempt") == attempt and entry.get("phase") == phase
    ), None)
    if production and (
        not isinstance(latest_progress, Mapping)
        or latest_progress.get("completed_games") != PHASE_TOTALS[phase]
        or latest_progress.get("completed_quotas") != PHASE_QUOTAS[phase]
        or latest_progress.get("progress_evidence", {}).get(
            "pipeline_plan", {}
        ).get("sha256") != closure.get("pipeline_plan", {}).get("sha256")
        or latest_progress.get("progress_evidence", {}).get("games_receipt") is None
        or latest_progress.get("progress_evidence", {}).get("positions_receipt") is None
        or latest_progress.get("progress_evidence", {}).get("finalized_receipt")
        is None
    ):
        raise ChallengerError("terminal phase progress is not pipeline-receipt derived")
    sealed_roles = {
        "training_plan": (
            "papersoccer.compact-value-bfm.rank4-teacher-training-plan.v1"
        ),
        "finalized_pipeline_receipt": (
            "papersoccer.compact-value-bfm-teacher-phase-stage-receipt.v1"
        ),
        "training_selection": (
            "papersoccer.compact-value-bfm.rank4-teacher-training-selection.v1"
        ),
        "gate_plan": "papersoccer.compact-value-bfm.rank4-teacher-gate-plan.v1",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for role, schema in sealed_roles.items():
        record = closure.get(role)
        record_path = _verify_sealed_record(record, schema, f"phase {role}")
        loaded[role] = qualification.load_sealed(record_path, schema)
    pipeline_record = closure.get("pipeline_plan")
    pipeline_path = _verify_record(pipeline_record, "phase pipeline plan")
    pipeline_value = qualification.load_sealed(
        pipeline_path, "papersoccer.compact-value-bfm-teacher-phase-pipeline.v1"
    )
    training_plan = loaded["training_plan"]
    final_receipt = loaded["finalized_pipeline_receipt"]
    selection = loaded["training_selection"]
    gate_plan = loaded["gate_plan"]
    selected = closure.get("selected_candidate")
    gate_results = closure.get("gate_results")
    if (
        closure.get("protected_tests_opened") is not False
        or not isinstance(selected, Mapping)
        or not isinstance(gate_results, Mapping)
        or selected.get("runtime", {}).get("sha256")
        != expected_candidate["runtime_sha256"]
        or selected.get("source", {}).get("sha256")
        != expected_candidate["source_sha256"]
        or training_plan.get("campaign_id") != CAMPAIGN_ID
        or training_plan.get("attempt") != attempt
        or training_plan.get("phase") != phase
        or training_plan.get("phase_reference") != _regular(phase_reference)
        or training_plan.get("pipeline_plan") != pipeline_record
        or training_plan.get("final_pipeline_receipt", {}).get("sha256")
        != closure["finalized_pipeline_receipt"]["sha256"]
        or pipeline_value.get("campaign_id") != CAMPAIGN_ID
        or pipeline_value.get("attempt") != attempt
        or pipeline_value.get("phase") != phase
        or pipeline_value.get("phase_reference") != _regular(phase_reference)
        or final_receipt.get("pipeline_body_sha256")
        != pipeline_value.get("body_sha256")
        or final_receipt.get("stage") != "07-finalize-labels"
        or final_receipt.get("details", {}).get("protected_tests_opened") is not False
        or selection.get("plan_body_sha256") != training_plan.get("body_sha256")
        or selection.get("selected_model") is None
        or gate_plan.get("selection", {}).get("sha256")
        != closure["training_selection"]["sha256"]
        or gate_plan.get("plan_body_sha256") != training_plan.get("body_sha256")
        or set(gate_results) != {
            str(item.get("variant")) for item in gate_plan.get("requests", [])
            if isinstance(item, Mapping)
        }
        or closure.get("input_audit_sha256")
        != sha256_bytes(canonical_json_bytes(training_plan.get("input_audit")))
        or closure.get("build_source_closure_sha256")
        != training_plan.get("build_source_closure", {}).get("closure_sha256")
    ):
        raise ChallengerError("phase outcome deep evidence binding changed")
    for variant, record in gate_results.items():
        _verify_record(record, f"phase gate result {variant}")
    if admission_receipt is None:
        if production:
            raise ChallengerError("production phase outcome requires admission receipt")
    else:
        try:
            from tools import compact_value_bfm_teacher_training as teacher_training

            admission = teacher_training.load_phase_admission(
                admission_receipt.resolve()
            )
        except Exception as error:
            raise ChallengerError("teacher-training admission failed validation") from error
        if (
            admission.get("campaign_id") != CAMPAIGN_ID
            or admission.get("attempt") != attempt
            or admission.get("phase") != phase
            or admission.get("metrics") != dict(metrics)
            or admission.get("selected_candidate") != selected
            or admission.get("results") != gate_results
            or admission.get("phase_outcome_evidence", {}).get("sha256")
            != sha256_file(path)
            or admission.get("development_exclusion", {}).get("sha256")
            != sha256_file(development_exclusion)
        ):
            raise ChallengerError("teacher-training admission/evidence disagree")
    return value


def record_attempt_outcome(
    plan_path: pathlib.Path, *, attempt: int, phase: str,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    outcome_receipt: pathlib.Path, development_exclusion: pathlib.Path,
    metrics: Mapping[str, Any],
    strength_delta_pp: float, teacher_regret_reduction_fraction: float,
    created_at_utc: str,
    admission_receipt: pathlib.Path | None = None,
) -> dict[str, Any]:
    context = validate_campaign(plan_path)
    if attempt <= 0 or phase not in PHASE_QUOTAS:
        raise ChallengerError("trained attempt outcome identity changed")
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    if (
        latest.get("attempt") != attempt
        or latest.get("event") != "progress-recorded"
        or latest.get("phase") != phase
        or latest.get("completed_games") != PHASE_TOTALS[phase]
        or latest.get("adaptation_route") != "record-attempt-outcome"
    ):
        raise ChallengerError("attempt outcome predates terminal phase progress")
    for value, label in (
        (strength_delta_pp, "strength delta"),
        (teacher_regret_reduction_fraction, "teacher regret reduction"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ChallengerError(f"{label} is invalid")
    if (
        metrics.get("strength_delta_pp") != float(strength_delta_pp)
        or metrics.get("teacher_regret_reduction_fraction")
        != float(teacher_regret_reduction_fraction)
    ):
        raise ChallengerError("attempt improvement metrics disagree with evidence")
    architecture = _architecture(candidate_runtime)
    source = _regular(candidate_source, ascii_required=True)
    if not 0 < source["bytes"] < SOURCE_LIMIT:
        raise ChallengerError("attempt candidate source violates 95KB")
    artifact_root = pathlib.Path(context["root"]) / "attempt-ledger/artifacts" / f"attempt-{attempt:03d}"
    candidate = {
        "runtime": _copy_attempt_artifact(_regular(candidate_runtime), root=artifact_root),
        "source": _copy_attempt_artifact(source, root=artifact_root),
        "architecture": architecture,
    }
    _validate_development_exclusion(
        development_exclusion, attempt=attempt, phase=phase
    )
    receipt_value = _validate_phase_outcome_evidence(
        outcome_receipt, context=context, attempt=attempt, phase=phase,
        candidate=candidate, metrics=metrics,
        development_exclusion=development_exclusion,
        admission_receipt=admission_receipt,
    )
    receipt = _copy_attempt_artifact(_regular(outcome_receipt), root=artifact_root)
    copied_admission = None
    if admission_receipt is not None:
        admission_value = qualification.load_sealed(
            admission_receipt,
            "papersoccer.compact-value-bfm.rank4-teacher-phase-admission.v1",
        )
        copied_admission = {
            **_copy_attempt_artifact(_regular(admission_receipt), root=artifact_root),
            "schema": admission_value["schema"],
            "body_sha256": admission_value["body_sha256"],
        }
    copied_exclusion = _copy_attempt_artifact(
        _regular(development_exclusion), root=artifact_root
    )
    admitted = _phase_admission(phase, metrics)
    phase_improved = (
        float(strength_delta_pp) >= 1.5
        or float(teacher_regret_reduction_fraction) >= 0.10
    )
    attempt_improved = phase_improved or any(
        entry["event"] == "attempt-outcome-recorded"
        and entry["attempt"] == attempt and entry.get("improved") is True
        for entry in entries
    )
    prior_no_improvement = _completed_no_improvement_streak(entries)
    consecutive = (
        prior_no_improvement if admitted else
        0 if attempt_improved else prior_no_improvement + 1
    )
    if admitted:
        route = "materialize-full" if phase == "pilot" else "prepare-dual-final"
    elif attempt_improved:
        route = "open-next-attempt-targeted-retry"
    elif consecutive == 2:
        route = "open-next-attempt-attribution-adaptation"
    else:
        route = "open-next-attempt-same-contract"
    return _append_event(
        context, attempt=attempt, event="attempt-outcome-recorded",
        created_at_utc=created_at_utc,
        fields={
            "phase": phase,
            "candidate": candidate,
            "outcome_receipt": receipt,
            "outcome_receipt_schema": PHASE_OUTCOME_EVIDENCE_SCHEMA,
            "admission_receipt": copied_admission,
            "development_exclusion": {
                **copied_exclusion,
                "schema": DEVELOPMENT_EXCLUSION_SCHEMA,
                "body_sha256": qualification.load_sealed(
                    development_exclusion, DEVELOPMENT_EXCLUSION_SCHEMA
                )["body_sha256"],
            },
            "metrics": dict(metrics),
            "strength_delta_pp": float(strength_delta_pp),
            "teacher_regret_reduction_fraction": float(teacher_regret_reduction_fraction),
            "admitted": admitted,
            "phase_improved": phase_improved,
            "improved": attempt_improved,
            "consecutive_no_improvement": consecutive,
            "adaptation_route": route,
            "automatic_action": False,
        },
    )


def open_next_attempt(
    plan_path: pathlib.Path, *, attempt: int, hypothesis: str,
    intervention: str, created_at_utc: str,
    attribution_receipt: pathlib.Path | None = None,
    student_runtime: pathlib.Path | None = None,
    prior_runtime: pathlib.Path | None = None,
    initial_float_checkpoint: pathlib.Path | None = None,
    roots_tsv: pathlib.Path | None = None,
    roots_manifest: pathlib.Path | None = None,
    build_manifest: pathlib.Path | None = None,
) -> dict[str, Any]:
    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    current = max(entry["attempt"] for entry in entries)
    if isinstance(attempt, bool) or attempt != current + 1:
        raise ChallengerError("next attempt is not the exact successor")
    latest = entries[-1]
    if latest.get("adaptation_route") not in {
        "open-next-attempt-teacher-refresh", "open-next-attempt-same-contract",
        "open-next-attempt-targeted-retry",
        "open-next-attempt-attribution-adaptation",
        "open-next-attempt-protected-rejection",
        "open-next-attempt-explicit-after-live-failure",
    }:
        raise ChallengerError("ledger did not authorize a next attempt")
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 500:
        raise ChallengerError("next attempt hypothesis is absent or too long")
    expected_interventions = {
        "open-next-attempt-teacher-refresh": "teacher-refresh",
        "open-next-attempt-same-contract": "same-contract",
        "open-next-attempt-targeted-retry": "targeted-retry",
        "open-next-attempt-protected-rejection": "protected-rejection-clean-restart",
        "open-next-attempt-explicit-after-live-failure": "explicit-post-live-restart",
    }
    copied_attribution = None
    if latest["adaptation_route"] == "open-next-attempt-attribution-adaptation":
        if attribution_receipt is None:
            raise ChallengerError("attribution adaptation requires its sealed audit")
        attribution = qualification.load_sealed(
            attribution_receipt, ATTRIBUTION_EVIDENCE_SCHEMA
        )
        classification = attribution.get("classification")
        if (
            attribution.get("campaign_id") != CAMPAIGN_ID
            or attribution.get("completed_no_improvement_attempts") != 2
            or classification not in ATTRIBUTION_INTERVENTIONS
            or attribution.get("selected_intervention")
            != ATTRIBUTION_INTERVENTIONS[classification]
            or attribution.get("next_attempt") != attempt
            or attribution.get("protected_or_live_data_read") is not False
            or intervention != ATTRIBUTION_INTERVENTIONS[classification]
        ):
            raise ChallengerError("attribution audit/intervention changed")
        copied = _copy_attempt_artifact(
            _regular(attribution_receipt),
            root=pathlib.Path(context["root"]) / "attempt-ledger/artifacts"
            / f"attempt-{attempt:03d}/attribution",
        )
        copied_attribution = {
            **copied,
            "schema": ATTRIBUTION_EVIDENCE_SCHEMA,
            "body_sha256": attribution["body_sha256"],
        }
    else:
        if attribution_receipt is not None:
            raise ChallengerError("attribution audit supplied outside attribution route")
        if intervention != expected_interventions[latest["adaptation_route"]]:
            raise ChallengerError("next attempt intervention disagrees with its route")

    frozen = context["inputs"]["attempt_one_inputs"]
    prior_candidates = [
        entry.get("candidate", {}).get("runtime")
        for entry in reversed(entries)
        if isinstance(entry.get("candidate"), Mapping)
        and isinstance(entry.get("candidate", {}).get("runtime"), Mapping)
    ]
    default_student = (
        frozen["student_runtime"]
        if attempt == 1 or not prior_candidates
        else prior_candidates[0]
    )
    overrides = {
        "student_runtime": student_runtime,
        "prior_runtime": prior_runtime,
        "initial_float_checkpoint": initial_float_checkpoint,
        "roots_tsv": roots_tsv,
        "roots_manifest": roots_manifest,
        "build_manifest": build_manifest,
    }
    defaults = {
        "student_runtime": default_student,
        "prior_runtime": frozen["prior_runtime"],
        "initial_float_checkpoint": frozen["initial_float_checkpoint"],
        "roots_tsv": frozen["roots_tsv"],
        "roots_manifest": frozen["roots_manifest"],
        "build_manifest": frozen["build_manifest"],
    }
    attempt_root = (
        pathlib.Path(context["root"]) / "attempt-ledger/artifacts"
        / f"attempt-{attempt:03d}/inputs"
    )
    attempt_inputs: dict[str, Any] = {}
    for name in defaults:
        supplied = overrides[name]
        if supplied is None:
            attempt_inputs[name] = defaults[name]
            continue
        record = _regular(supplied)
        if name == "build_manifest":
            _validate_build_manifest(supplied, production=False)
        copied = _copy_attempt_artifact(record, root=attempt_root)
        attempt_inputs[name] = copied
    if build_manifest is None:
        attempt_inputs["producer_binaries"] = frozen["producer_binaries"]
    else:
        build_value = _validate_build_manifest(build_manifest, production=False)
        copied_binaries = {}
        for name, record in build_value["binaries"].items():
            copied = _copy_attempt_artifact(
                {key: record[key] for key in ("path", "bytes", "sha256")},
                root=attempt_root / "binaries",
            )
            os.chmod(pathlib.Path(copied["path"]), 0o555)
            copied_binaries[name] = copied
        attempt_inputs["producer_binaries"] = copied_binaries
    _validate_attempt_inputs(
        attempt_inputs, plan=context["plan"], require_student_12x8=True
    )
    dynamic_exclusions = _cumulative_dynamic_exclusions(entries)
    return _append_event(
        context,
        attempt=attempt,
        event="attempt-opened",
        created_at_utc=created_at_utc,
        fields={
            "parent_attempt": current,
            "parent_event_sha256": latest["body_sha256"],
            "route": latest["adaptation_route"],
            "architecture": ARCHITECTURE,
            "policy_head": False,
            "hypothesis": hypothesis.strip(),
            "intervention": intervention,
            "attribution_receipt": copied_attribution,
            "attempt_inputs": attempt_inputs,
            "dynamic_exclusions": dynamic_exclusions,
        },
    )


def _bank_fingerprints(bank: Mapping[str, Any]) -> set[str]:
    return {
        str(value)
        for opening in bank.get("openings", [])
        for value in opening.get("fingerprints", {}).values()
    }


def _write_bank_dynamic_exclusion(
    *, root: pathlib.Path, attempt: int, gate_id: str,
    bank: Mapping[str, Any], bank_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    values = sorted({
        str(opening.get("fingerprints", {}).get("canonical"))
        for opening in bank.get("openings", [])
        if isinstance(opening, Mapping)
    })
    seed = bank.get("seed_receipt")
    if (
        gate_id not in {"gate-a", "gate-b"}
        or len(values) != 500
        or any(SHA256_RE.fullmatch(value) is None for value in values)
        or not isinstance(seed, Mapping)
        or SHA256_RE.fullmatch(str(seed.get("sha256", ""))) is None
    ):
        raise ChallengerError("protected bank cannot produce sanitized fingerprints")
    body = {
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "gate_id": gate_id,
        "classification": "protected-final-canonical-fingerprints",
        "domain": "protected-final-opening-canonical-state",
        "origin": {
            "candidate_source_sha256": candidate["source"]["sha256"],
            "candidate_runtime_sha256": candidate["runtime"]["sha256"],
            "protected_bank_sha256": bank_record["sha256"],
            "seed_sha256": seed["sha256"],
        },
        "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
        "fingerprints": values,
        "fingerprint_count": len(values),
        "contains_transcripts": False,
        "contains_metrics": False,
        "contains_labels": False,
        "training_eligible": False,
        "required_for_all_later_development_and_protected_banks": True,
    }
    path = root / "sanitized-exclusions" / f"{gate_id}.json"
    qualification.write_sealed(path, body)
    return _dynamic_exclusion_record(path)


BankValidator = Callable[[pathlib.Path], Mapping[str, Any]]


def _required_final_exclusion_hashes(
    context: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
) -> list[str]:
    records = [
        *context["inputs"]["protected_exclusions"].values(),
        *context["inputs"]["live_exclusions"].values(),
        *(
            entry["development_exclusion"]
            for entry in entries
            if entry.get("event") == "attempt-outcome-recorded"
        ),
        *_cumulative_dynamic_exclusions(entries),
    ]
    return sorted({str(record["sha256"]) for record in records})


def authorize_dual_final(
    plan_path: pathlib.Path, *, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    created_at_utc: str,
    release_evidence_path: pathlib.Path | None = None,
    deployed_source: pathlib.Path | None = None,
) -> pathlib.Path:
    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    root = pathlib.Path(context["plan"]["outputs"]["dual_final"]) / f"attempt-{attempt:03d}"
    authorization_path = root / "dual-final-authorization.json"
    if (
        latest.get("event") == "dual-final-authorized"
        and latest.get("attempt") == attempt
        and authorization_path.exists()
    ):
        _validate_dual_final_authorization(
            authorization_path, context=context, attempt=attempt
        )
        return authorization_path
    authorized_from_external_zero = (
        attempt == 0
        and latest.get("event") == "attempt-zero-result-recorded"
        and latest.get("passed") is True
        and latest.get("adaptation_route") == "prepare-dual-final"
    )
    authorized_from_full = (
        attempt > 0
        and latest.get("attempt") == attempt
        and latest.get("event") == "attempt-outcome-recorded"
        and latest.get("phase") == "full"
        and latest.get("admitted") is True
        and latest.get("adaptation_route") == "prepare-dual-final"
    )
    if not (authorized_from_external_zero or authorized_from_full):
        raise ChallengerError("full progress did not authorize dual final")
    architecture = _architecture(candidate_runtime)
    supplied_runtime = _regular(candidate_runtime)
    supplied_source = _regular(candidate_source, ascii_required=True)
    if not 0 < supplied_source["bytes"] < SOURCE_LIMIT:
        raise ChallengerError("final candidate violates source limit")
    production = context["inputs"].get("production_allowlist_enforced") is True
    release_evidence = None
    release_candidate = None
    if release_evidence_path is not None:
        try:
            from tools import compact_value_bfm_rank4_teacher_release as release_bridge

            release_evidence = release_bridge.validate_release_evidence(
                release_evidence_path,
                campaign_plan_path=plan_path,
                attempt=attempt,
                candidate_runtime=candidate_runtime,
                candidate_source=candidate_source,
            )
        except Exception as error:
            raise ChallengerError("candidate release evidence failed validation") from error
        release_candidate = release_evidence.get("candidate")
        if not isinstance(release_candidate, Mapping):
            raise ChallengerError("release evidence omits selected candidate")
    elif production:
        raise ChallengerError("production dual final requires release evidence")
    released_source_path = candidate_source.resolve()
    if release_candidate is not None:
        released_record = release_candidate.get("source")
        if not isinstance(released_record, Mapping) or not {
            "path", "bytes", "sha256"
        }.issubset(released_record):
            raise ChallengerError("release evidence deployed source is malformed")
        released_source_path = _verify_record(
            {key: released_record[key] for key in ("path", "bytes", "sha256")},
            "released deployed source",
        )
        if deployed_source is not None and _regular(
            deployed_source, ascii_required=True
        )["sha256"] != sha256_file(released_source_path):
            raise ChallengerError("supplied deployed source differs from release evidence")
    if authorized_from_external_zero:
        expected_candidate = latest.get("candidate_identity")
        if (
            not isinstance(expected_candidate, Mapping)
            or supplied_runtime["sha256"] != expected_candidate.get("runtime_sha256")
            or (
                release_candidate is None
                and supplied_source["sha256"] != expected_candidate.get("source_sha256")
            )
            or (
                release_candidate is not None
                and release_candidate.get("generated_source", {}).get("sha256")
                != expected_candidate.get("source_sha256")
            )
        ):
            raise ChallengerError("attempt-zero candidate changed before dual final")
        canonical_runtime = _verify_bundle_record(
            context["inputs"]["candidate"]["runtime"],
            input_directory=pathlib.Path(context["plan"]["outputs"]["input_directory"]),
            label="attempt-zero final runtime",
        )
        canonical_source = (
            released_source_path
            if release_candidate is not None
            else _verify_bundle_record(
                context["inputs"]["candidate"]["source"],
                input_directory=pathlib.Path(
                    context["plan"]["outputs"]["input_directory"]
                ),
                label="attempt-zero final source",
            )
        )
    else:
        expected_candidate = latest.get("candidate")
        if (
            not isinstance(expected_candidate, Mapping)
            or supplied_runtime["sha256"]
            != expected_candidate.get("runtime", {}).get("sha256")
            or (
                release_candidate is None
                and supplied_source["sha256"]
                != expected_candidate.get("source", {}).get("sha256")
            )
            or (
                release_candidate is not None
                and (
                    release_candidate.get("generated_source", {}).get("sha256")
                    != expected_candidate.get("source", {}).get("sha256")
                    or release_candidate.get("generated_source", {}).get("bytes")
                    != expected_candidate.get("source", {}).get("bytes")
                )
            )
        ):
            raise ChallengerError("admitted full candidate changed before dual final")
        canonical_runtime = _verify_record(
            expected_candidate["runtime"], "admitted final runtime"
        )
        canonical_source = (
            released_source_path
            if release_candidate is not None
            else _verify_record(expected_candidate["source"], "admitted final source")
        )
    candidate = {
        "runtime": _regular(canonical_runtime),
        "source": _regular(canonical_source, ascii_required=True),
        "architecture": architecture,
    }
    if not 0 < candidate["source"]["bytes"] < SOURCE_LIMIT:
        raise ChallengerError("released deployed source violates source limit")
    if release_candidate is not None and (
        release_candidate.get("runtime", {}).get("sha256")
        != candidate["runtime"]["sha256"]
        or release_candidate.get("source", {}).get("sha256")
        != candidate["source"]["sha256"]
        or release_candidate.get("architecture") != architecture
    ):
        raise ChallengerError("released candidate differs from dual-final candidate")
    if authorization_path.exists():
        existing = qualification.load_sealed(
            authorization_path, DUAL_FINAL_AUTHORIZATION_SCHEMA
        )
        if existing.get("candidate") != candidate:
            raise ChallengerError("dual-final authorization candidate changed")
        return authorization_path
    if root.exists() and any(root.iterdir()):
        raise ChallengerError("dual-final artifacts predate candidate authorization")
    body = {
        "schema": DUAL_FINAL_AUTHORIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "campaign_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "attempt": attempt,
        "status": "candidate-frozen-before-protected-bank-materialization",
        "created_at_utc": utc(created_at_utc, "dual-final authorization timestamp"),
        "basis_event_sha256": latest["body_sha256"],
        "candidate": candidate,
        "generated_source": (
            context["inputs"]["candidate"]["source"]
            if authorized_from_external_zero else latest["candidate"]["source"]
        ),
        "release_evidence": (
            None if release_evidence_path is None
            else _sealed_record(release_evidence_path, RELEASE_EVIDENCE_SCHEMA)
        ),
        "required_exclusion_sha256": _required_final_exclusion_hashes(
            context, entries
        ),
        "gate_contract": {
            "gate_ids": ["gate-a", "gate-b"],
            "pairs_per_gate": 500,
            "games_per_gate": 1_000,
            "workers_per_gate": 4,
            "gates_concurrent": False,
            "gate_b_excludes_gate_a": True,
        },
        "thresholds": FINAL_THRESHOLDS,
        "bank_materializations_authorized": 2,
        "entropy_draws_authorized": 2,
        "candidate_change_authorized": False,
        "automatic_launch": False,
        "uploads_authorized": 0,
    }
    qualification.write_sealed(authorization_path, body)
    _append_event(
        context, attempt=attempt, event="dual-final-authorized",
        created_at_utc=created_at_utc,
        fields={
            "authorization": _sealed_record(
                authorization_path, DUAL_FINAL_AUTHORIZATION_SCHEMA
            ),
            "candidate": candidate,
            "banks_materialized": 0,
            "games_launched": 0,
        },
    )
    return authorization_path


def _validate_dual_final_authorization(
    path: pathlib.Path, *, context: Mapping[str, Any], attempt: int,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, DUAL_FINAL_AUTHORIZATION_SCHEMA)
    entries = load_ledger(context["plan"])
    event = next((
        entry for entry in entries
        if entry.get("event") == "dual-final-authorized"
        and entry.get("attempt") == attempt
    ), None)
    event_index = entries.index(event) if event in entries else -1
    basis_sha256 = (
        entries[event_index - 1]["body_sha256"] if event_index > 0 else None
    )
    basis_event = entries[event_index - 1] if event_index > 0 else {}
    expected_generated_source = (
        context["inputs"]["candidate"]["source"]
        if attempt == 0 else basis_event.get("candidate", {}).get("source")
    )
    if (
        path.is_symlink()
        or path.resolve() != pathlib.Path(context["plan"]["outputs"]["dual_final"]) / f"attempt-{attempt:03d}/dual-final-authorization.json"
        or event is None
        or event.get("authorization")
        != _sealed_record(path, DUAL_FINAL_AUTHORIZATION_SCHEMA)
        or event.get("candidate") != value.get("candidate")
        or value.get("campaign_plan")
        != _sealed_record(pathlib.Path(context["plan"]["outputs"]["plan"]), PLAN_SCHEMA)
        or value.get("attempt") != attempt
        or value.get("status")
        != "candidate-frozen-before-protected-bank-materialization"
        or value.get("basis_event_sha256") != basis_sha256
        or value.get("generated_source") != expected_generated_source
        or value.get("required_exclusion_sha256")
        != _required_final_exclusion_hashes(context, entries[:event_index])
        or value.get("gate_contract") != {
            "gate_ids": ["gate-a", "gate-b"], "pairs_per_gate": 500,
            "games_per_gate": 1_000, "workers_per_gate": 4,
            "gates_concurrent": False, "gate_b_excludes_gate_a": True,
        }
        or value.get("thresholds") != FINAL_THRESHOLDS
        or value.get("bank_materializations_authorized") != 2
        or value.get("entropy_draws_authorized") != 2
        or value.get("candidate_change_authorized") is not False
        or value.get("automatic_launch") is not False
        or value.get("uploads_authorized") != 0
    ):
        raise ChallengerError("dual-final authorization changed")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ChallengerError("dual-final authorization candidate is absent")
    runtime_path = _verify_record(candidate.get("runtime"), "authorized runtime")
    source_path = _verify_record(candidate.get("source"), "authorized source")
    generated_source_path = _resolve_campaign_artifact(
        value.get("generated_source"), plan=context["plan"],
        label="authorized generated source",
    )
    if (
        candidate.get("architecture") != _architecture(runtime_path)
        or not 0 < source_path.stat().st_size < SOURCE_LIMIT
    ):
        raise ChallengerError("authorized candidate runtime/source contract changed")
    release_record = value.get("release_evidence")
    production = context["inputs"].get("production_allowlist_enforced") is True
    if release_record is None:
        if production:
            raise ChallengerError("production authorization lost release evidence")
    else:
        release_path = _verify_sealed_record(
            release_record, RELEASE_EVIDENCE_SCHEMA, "candidate release evidence"
        )
        try:
            from tools import compact_value_bfm_rank4_teacher_release as release_bridge

            release_bridge.validate_release_evidence(
                release_path,
                campaign_plan_path=pathlib.Path(context["plan"]["outputs"]["plan"]),
                attempt=attempt,
                candidate_runtime=runtime_path,
                candidate_source=generated_source_path,
            )
        except Exception as error:
            raise ChallengerError("authorized release evidence failed revalidation") from error
    return value


def _validate_authorized_bank(
    bank: Mapping[str, Any], *, record: Mapping[str, Any],
    authorization: Mapping[str, Any], gate_id: str,
) -> dict[str, Any]:
    source_reference = bank.get("source_binding")
    seed_reference = bank.get("seed_receipt")
    if not isinstance(source_reference, Mapping) or not isinstance(seed_reference, Mapping):
        raise ChallengerError(f"{gate_id} protected bank lacks source/seed binding")
    source_path = pathlib.Path(str(source_reference.get("path", "")))
    if (
        source_path.is_symlink() or not source_path.is_file()
        or sha256_file(source_path) != source_reference.get("sha256")
    ):
        raise ChallengerError(f"{gate_id} source binding changed")
    source = qualification.load_sealed(
        source_path, qualification.SOURCE_BINDING_SCHEMA
    )
    qualification.validate_source_binding(source)
    candidate = authorization["candidate"]
    if (
        source.get("candidate", {}).get("sha256")
        != candidate["source"]["sha256"]
        or source.get("candidate", {}).get("bytes")
        != candidate["source"]["bytes"]
        or source.get("rank4", {}).get("sha256") != qualification.RANK4_SHA256
        or source.get("opponent", {}).get("sha256") != qualification.RANK4_SHA256
    ):
        raise ChallengerError(f"{gate_id} bank belongs to another candidate")
    seed_path = pathlib.Path(str(seed_reference.get("path", "")))
    if (
        seed_path.is_symlink() or not seed_path.is_file()
        or sha256_file(seed_path) != seed_reference.get("sha256")
    ):
        raise ChallengerError(f"{gate_id} protected seed receipt changed")
    seed = qualification.load_sealed(seed_path, openings.SEED_SCHEMA)
    if (
        seed.get("status") != "protected-seed-frozen-before-bank-generation"
        or seed.get("source_binding") != source_reference
        or seed.get("candidate_sha256") != candidate["source"]["sha256"]
        or qualification._utc(seed.get("created_at_utc"), f"{gate_id} seed time")
        < qualification._utc(
            authorization.get("created_at_utc"), "dual-final authorization time"
        )
        or bank.get("seed_receipt") != seed_reference
        or bank.get("seed_hex") != seed.get("seed_256_hex")
        or record.get("sha256") is None
    ):
        raise ChallengerError(f"{gate_id} bank predates or contradicts authorization")
    return {"source": source, "seed": seed}


def prepare_dual_final(
    authorization_path: pathlib.Path, *, plan_path: pathlib.Path,
    bank_a: pathlib.Path, bank_b: pathlib.Path, created_at_utc: str,
    bank_validator: BankValidator = openings.validate_bank,
) -> pathlib.Path:
    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    try:
        authorization_header = qualification.load_sealed(
            authorization_path, DUAL_FINAL_AUTHORIZATION_SCHEMA
        )
    except Exception as error:
        raise ChallengerError("dual-final authorization is invalid") from error
    attempt = authorization_header.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ChallengerError("dual-final authorization attempt changed")
    root = pathlib.Path(context["plan"]["outputs"]["dual_final"]) / f"attempt-{attempt:03d}"
    reference_path = root / "dual-final-reference.json"
    if (
        latest.get("event") == "dual-final-prepared"
        and latest.get("attempt") == attempt
        and reference_path.exists()
    ):
        validate_dual_final(
            reference_path, plan_path=plan_path, bank_validator=bank_validator
        )
        return reference_path
    if latest.get("event") != "dual-final-authorized":
        raise ChallengerError("protected banks lack prior candidate authorization")
    if latest.get("attempt") != attempt:
        raise ChallengerError("protected banks belong to another authorized attempt")
    authorization = _validate_dual_final_authorization(
        authorization_path, context=context, attempt=attempt
    )
    candidate = dict(authorization["candidate"])
    documents = [dict(bank_validator(path)) for path in (bank_a, bank_b)]
    records = [_regular(path) for path in (bank_a, bank_b)]
    if (
        records[0]["sha256"] == records[1]["sha256"]
        or records[0]["path"] == records[1]["path"]
        or any(
            bank.get("classification") != "protected-final"
            or bank.get("opening_count") != 500
            for bank in documents
        )
        or _bank_fingerprints(documents[0]) & _bank_fingerprints(documents[1])
    ):
        raise ChallengerError("dual final banks are not independent protected 500-pair banks")
    bank_bindings = [
        _validate_authorized_bank(
            bank, record=record, authorization=authorization, gate_id=gate_id
        )
        for gate_id, bank, record in zip(
            ("gate-a", "gate-b"), documents, records, strict=True
        )
    ]
    if (
        bank_bindings[0]["seed"]["seed_256_hex"]
        == bank_bindings[1]["seed"]["seed_256_hex"]
        or documents[0].get("source_binding")
        != documents[1].get("source_binding")
    ):
        raise ChallengerError("dual final banks reused entropy or source identity")
    required_exclusion_hashes = set(authorization["required_exclusion_sha256"])
    for index, bank in enumerate(documents):
        observed = {
            str(record.get("sha256"))
            for record in bank.get("exclusion_sources", [])
            if isinstance(record, Mapping)
        }
        if not required_exclusion_hashes.issubset(observed):
            raise ChallengerError("dual final bank omits protected/live exclusions")
        if index == 1 and records[0]["sha256"] not in observed:
            raise ChallengerError("gate B bank does not exclude all Gate A fingerprints")
    dynamic_exclusions = [
        _write_bank_dynamic_exclusion(
            root=root, attempt=attempt, gate_id=gate_id,
            bank=bank, bank_record=record, candidate=candidate,
        )
        for gate_id, bank, record in zip(
            ("gate-a", "gate-b"), documents, records, strict=True
        )
    ]
    if reference_path.exists():
        validate_dual_final(reference_path, plan_path=plan_path, bank_validator=bank_validator)
        return reference_path
    body = {
        "schema": DUAL_FINAL_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "campaign_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "authorization": _sealed_record(
            authorization_path, DUAL_FINAL_AUTHORIZATION_SCHEMA
        ),
        "attempt": attempt,
        "status": "two-independent-final-gates-prepared-not-run",
        "created_at_utc": utc(created_at_utc, "dual final preparation timestamp"),
        "candidate": candidate,
        "gates": [
            {
                "gate_id": gate_id,
                "bank": record,
                "pairs": 500,
                "games": 1_000,
                "result_path": str(root / f"{gate_id}.result.json"),
            }
            for gate_id, record in zip(("gate-a", "gate-b"), records, strict=True)
        ],
        "dynamic_exclusions": dynamic_exclusions,
        "thresholds": FINAL_THRESHOLDS,
        "resources": {
            "workers_per_gate": 4,
            "gates_concurrent": False,
            "threads_per_worker": 1,
        },
        "candidate_change_between_gates_authorized": False,
        "result_reuse_between_gates_authorized": False,
        "automatic_launch": False,
        "rank4_replacement_authorized": False,
        "upload_authorized": False,
    }
    dual_path, _dual = _write_content_addressed(root, body, ".dual-final-plan.json")
    qualification.write_sealed(reference_path, {
        "schema": DUAL_FINAL_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "dual_final_plan": _sealed_record(dual_path, DUAL_FINAL_SCHEMA),
    })
    _append_event(
        context,
        attempt=attempt,
        event="dual-final-prepared",
        created_at_utc=created_at_utc,
        fields={
            "dual_final_reference": _sealed_record(reference_path, DUAL_FINAL_REFERENCE_SCHEMA),
            "candidate": candidate,
            "gate_ids": ["gate-a", "gate-b"],
            "dynamic_exclusions": dynamic_exclusions,
            "games_launched": 0,
        },
    )
    return reference_path


def validate_dual_final(
    reference_path: pathlib.Path, *, plan_path: pathlib.Path,
    bank_validator: BankValidator = openings.validate_bank,
) -> dict[str, Any]:
    context = validate_campaign(plan_path)
    expected_root = pathlib.Path(context["plan"]["outputs"]["dual_final"])
    reference = qualification.load_sealed(reference_path, DUAL_FINAL_REFERENCE_SCHEMA)
    dual_path = _verify_sealed_record(reference.get("dual_final_plan"), DUAL_FINAL_SCHEMA, "dual final plan")
    dual = qualification.load_sealed(dual_path, DUAL_FINAL_SCHEMA)
    attempt = dual.get("attempt")
    expected_attempt_root = expected_root / f"attempt-{attempt:03d}" if isinstance(attempt, int) else expected_root
    if (
        reference_path.is_symlink()
        or reference_path.resolve() != expected_attempt_root / "dual-final-reference.json"
        or dual_path.parent != expected_attempt_root
        or dual_path.name != f"{sha256_file(dual_path)}.dual-final-plan.json"
    ):
        raise ChallengerError("dual final paths changed")
    candidate = dual.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ChallengerError("dual final candidate is absent")
    architecture = _architecture(pathlib.Path(candidate["runtime"]["path"]))
    runtime_path = _verify_record(candidate.get("runtime"), "dual final runtime")
    source_path = _verify_record(candidate.get("source"), "dual final source")
    try:
        source_path.read_bytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise ChallengerError("dual final source is not ASCII") from error
    if (
        candidate.get("architecture") != architecture
        or runtime_path != pathlib.Path(candidate["runtime"]["path"]).resolve()
        or not 0 < source_path.stat().st_size < SOURCE_LIMIT
    ):
        raise ChallengerError("dual final candidate architecture changed")
    authorization_path = _verify_sealed_record(
        dual.get("authorization"), DUAL_FINAL_AUTHORIZATION_SCHEMA,
        "dual-final authorization",
    )
    authorization = _validate_dual_final_authorization(
        authorization_path, context=context, attempt=int(attempt)
    )
    if authorization.get("candidate") != candidate:
        raise ChallengerError("dual final changed the authorized candidate")
    gates = dual.get("gates")
    if not isinstance(gates, list) or [gate.get("gate_id") for gate in gates] != ["gate-a", "gate-b"]:
        raise ChallengerError("dual final gate roster changed")
    documents = []
    bank_records = []
    for gate in gates:
        bank_path = _verify_record(gate.get("bank"), f"{gate.get('gate_id')} bank")
        document = dict(bank_validator(bank_path))
        if (
            set(gate) != {"gate_id", "bank", "pairs", "games", "result_path"}
            or document.get("classification") != "protected-final"
            or document.get("opening_count") != 500
            or gate.get("pairs") != 500
            or gate.get("games") != 1_000
            or gate.get("result_path")
            != str(expected_attempt_root / f"{gate['gate_id']}.result.json")
        ):
            raise ChallengerError("dual final gate bank/count changed")
        _validate_authorized_bank(
            document, record=gate["bank"], authorization=authorization,
            gate_id=str(gate["gate_id"]),
        )
        documents.append(document)
        bank_records.append(gate["bank"])
    if _bank_fingerprints(documents[0]) & _bank_fingerprints(documents[1]):
        raise ChallengerError("dual final banks overlap")
    dynamic_exclusions = dual.get("dynamic_exclusions")
    if not isinstance(dynamic_exclusions, list) or len(dynamic_exclusions) != 2:
        raise ChallengerError("dual final sanitized exclusion roster changed")
    for index, record in enumerate(dynamic_exclusions):
        path = _verify_dynamic_exclusion_record(
            record, f"dual final gate {index}"
        )
        exclusion = validate_dynamic_exclusion(path)
        if (
            exclusion.get("attempt") != attempt
            or exclusion.get("gate_id") != ("gate-a", "gate-b")[index]
            or exclusion.get("origin", {}).get("protected_bank_sha256")
            != bank_records[index]["sha256"]
            or exclusion.get("origin", {}).get("candidate_source_sha256")
            != candidate["source"]["sha256"]
            or exclusion.get("origin", {}).get("candidate_runtime_sha256")
            != candidate["runtime"]["sha256"]
            or exclusion.get("origin", {}).get("seed_sha256")
            != documents[index]["seed_receipt"]["sha256"]
            or exclusion.get("fingerprints")
            != sorted({
                str(opening["fingerprints"]["canonical"])
                for opening in documents[index]["openings"]
            })
        ):
            raise ChallengerError("dual final sanitized fingerprints disagree with bank")
    required_exclusion_hashes = set(authorization["required_exclusion_sha256"])
    if any(
        not required_exclusion_hashes.issubset({
            str(record.get("sha256"))
            for record in bank.get("exclusion_sources", [])
            if isinstance(record, Mapping)
        })
        for bank in documents
    ):
        raise ChallengerError("dual final bank exclusion binding changed")
    second_exclusions = {
        str(record.get("sha256"))
        for record in documents[1].get("exclusion_sources", [])
        if isinstance(record, Mapping)
    }
    if bank_records[0]["sha256"] not in second_exclusions:
        raise ChallengerError("gate B bank lost its Gate A exclusion")
    if (
        set(dual) != {
            "schema", "namespace", "campaign_id", "campaign_plan", "attempt",
            "status", "created_at_utc", "authorization", "candidate", "gates", "thresholds",
            "dynamic_exclusions",
            "resources", "candidate_change_between_gates_authorized",
            "result_reuse_between_gates_authorized", "automatic_launch",
            "rank4_replacement_authorized", "upload_authorized", "body_sha256",
        }
        or dual.get("campaign_plan") != _sealed_record(plan_path, PLAN_SCHEMA)
        or dual.get("authorization")
        != _sealed_record(authorization_path, DUAL_FINAL_AUTHORIZATION_SCHEMA)
        or dual.get("thresholds") != FINAL_THRESHOLDS
        or dual.get("candidate_change_between_gates_authorized") is not False
        or dual.get("result_reuse_between_gates_authorized") is not False
        or dual.get("automatic_launch") is not False
    ):
        raise ChallengerError("dual final policy changed")
    return {"context": context, "reference": reference, "plan": dual, "path": dual_path, "banks": documents}


def _validate_final_gate_evidence(
    path: pathlib.Path, *, dual: Mapping[str, Any], dual_path: pathlib.Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    value = qualification.load_sealed(path, FINAL_GATE_EVIDENCE_SCHEMA)
    aggregate_record = value.get("aggregate")
    aggregate_path = _verify_record(aggregate_record, "final gate aggregate")
    aggregate = qualification.load_sealed(aggregate_path)
    candidate = dual["candidate"]
    campaign_plan_path = pathlib.Path(dual["campaign_plan"]["path"])
    campaign = validate_campaign(campaign_plan_path)
    production = campaign["inputs"].get("production_allowlist_enforced") is True
    bridge_schema = value.get("bridge_schema")
    if bridge_schema is not None:
        try:
            from tools import compact_value_bfm_rank4_teacher_dual_final as dual_runner

            if bridge_schema != dual_runner.DEEP_GATE_EVIDENCE_SCHEMA:
                raise ChallengerError("final gate bridge schema changed")
            deep = dual_runner.validate_governance_evidence(
                path,
                campaign_plan_path=campaign_plan_path,
                dual_reference=dual_path.parent / "dual-final-reference.json",
            )
        except Exception as error:
            raise ChallengerError("deep final-gate evidence failed validation") from error
        if deep.get("summary") != value.get("summary"):
            raise ChallengerError("deep final-gate summary changed")
    elif production:
        raise ChallengerError("production final gate lacks deep maintained evidence")
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != dual["attempt"]
        or value.get("gate_id") != gate["gate_id"]
        or value.get("status") != "complete"
        or value.get("dual_final_plan")
        != _sealed_record(dual_path, DUAL_FINAL_SCHEMA)
        or value.get("candidate") != {
            "runtime_sha256": candidate["runtime"]["sha256"],
            "source_sha256": candidate["source"]["sha256"],
        }
        or value.get("bank") != {
            "sha256": gate["bank"]["sha256"],
            "bytes": gate["bank"]["bytes"],
        }
        or value.get("pairs") != 500
        or value.get("games") != 1_000
        or value.get("workers") != 4
        or value.get("threads_per_worker") != 1
        or value.get("all_shards_complete") is not True
        or value.get("summary") != aggregate.get("summary")
        or aggregate.get("candidate_source_sha256")
        != candidate["source"]["sha256"]
        or aggregate.get("bank_sha256") != gate["bank"]["sha256"]
        or aggregate.get("workers") != 4
    ):
        raise ChallengerError("final gate evidence is not candidate/bank/aggregate bound")
    return value


def record_final_result(
    dual_reference: pathlib.Path, *, plan_path: pathlib.Path,
    gate_id: str, evidence_path: pathlib.Path, completed_at_utc: str,
    bank_validator: BankValidator = openings.validate_bank,
) -> pathlib.Path:
    state = validate_dual_final(dual_reference, plan_path=plan_path, bank_validator=bank_validator)
    gates = {gate["gate_id"]: gate for gate in state["plan"]["gates"]}
    if gate_id not in gates:
        raise ChallengerError("unknown final gate id")
    gate = gates[gate_id]
    output = pathlib.Path(gate["result_path"])
    if output.exists():
        validate_final_result(
            output, dual=state["plan"], gate_id=gate_id, require_pass=False
        )
        entries = load_ledger(state["context"]["plan"])
        if not any(
            entry.get("event") == "final-gate-recorded"
            and entry.get("gate_id") == gate_id
            and entry.get("result") == _sealed_record(output, FINAL_RESULT_SCHEMA)
            for entry in entries
        ):
            raise ChallengerError("final result exists without its ledger event")
        return output
    entries = load_ledger(state["context"]["plan"])
    latest = entries[-1]
    if (
        (gate_id == "gate-a" and latest.get("event") != "dual-final-prepared")
        or (
            gate_id == "gate-b"
            and (
                latest.get("event") != "final-gate-recorded"
                or latest.get("gate_id") != "gate-a"
                or latest.get("passed") is not True
                or latest.get("adaptation_route") != "run-gate-b"
            )
        )
    ):
        raise ChallengerError("final gates must execute sequentially A then B")
    evidence = _validate_final_gate_evidence(
        evidence_path, dual=state["plan"], dual_path=state["path"], gate=gate
    )
    if any(
        entry.get("evidence", {}).get("sha256") == sha256_file(evidence_path)
        for entry in entries if entry.get("event") == "final-gate-recorded"
    ):
        raise ChallengerError("final gate evidence was reused")
    summary = evidence["summary"]
    verdict = qualification.strict_gate_verdict(summary)
    artifact_root = output.parent / "evidence"
    copied_evidence = _copy_attempt_artifact(
        _regular(evidence_path), root=artifact_root
    )
    copied_aggregate = _copy_attempt_artifact(
        evidence["aggregate"], root=artifact_root
    )
    body = {
        "schema": FINAL_RESULT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": state["plan"]["attempt"],
        "gate_id": gate_id,
        "dual_final_plan": _sealed_record(state["path"], DUAL_FINAL_SCHEMA),
        "candidate": dict(state["plan"]["candidate"]),
        "bank": dict(gate["bank"]),
        "summary": dict(summary),
        "evidence": copied_evidence,
        "source_evidence": _regular(evidence_path),
        "aggregate": copied_aggregate,
        "verdict": verdict,
        "completed_at_utc": utc(completed_at_utc, "final gate completion timestamp"),
        "passed": bool(verdict["passed"]),
        "rank4_replacement_authorized": False,
        "upload_authorized": False,
    }
    if output.exists():
        if qualification.load_sealed(output, FINAL_RESULT_SCHEMA) != qualification.seal(body):
            raise ChallengerError("immutable final result changed")
        return output
    qualification.write_sealed(output, body)
    _append_event(
        state["context"],
        attempt=state["plan"]["attempt"],
        event="final-gate-recorded",
        created_at_utc=completed_at_utc,
        fields={
            "gate_id": gate_id,
            "result": _sealed_record(output, FINAL_RESULT_SCHEMA),
            "evidence": copied_evidence,
            "passed": bool(verdict["passed"]),
            "dynamic_exclusions": (
                [] if verdict["passed"]
                else list(state["plan"]["dynamic_exclusions"])
            ),
            "adaptation_route": (
                "run-gate-b" if gate_id == "gate-a" and verdict["passed"] else
                "complete-dual-final" if verdict["passed"] else
                "open-next-attempt-protected-rejection"
            ),
            "automatic_action": False,
        },
    )
    return output


def validate_final_result(
    path: pathlib.Path, *, dual: Mapping[str, Any], gate_id: str,
    require_pass: bool = True,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, FINAL_RESULT_SCHEMA)
    gate = next((item for item in dual["gates"] if item["gate_id"] == gate_id), None)
    verdict = qualification.strict_gate_verdict(value.get("summary", {}))
    evidence_path = _verify_record(value.get("evidence"), f"{gate_id} evidence")
    source_evidence_path = _verify_record(
        value.get("source_evidence"), f"{gate_id} source evidence"
    )
    aggregate_path = _verify_record(value.get("aggregate"), f"{gate_id} aggregate")
    evidence = qualification.load_sealed(
        evidence_path, FINAL_GATE_EVIDENCE_SCHEMA
    )
    aggregate = qualification.load_sealed(aggregate_path)
    if (
        gate is None
        or path.is_symlink()
        or path.resolve() != pathlib.Path(gate["result_path"])
        or set(value) != {
            "schema", "namespace", "campaign_id", "attempt", "gate_id",
            "dual_final_plan", "candidate", "bank", "summary", "verdict",
            "evidence", "aggregate",
            "source_evidence",
            "completed_at_utc", "passed", "rank4_replacement_authorized",
            "upload_authorized", "body_sha256",
        }
        or value.get("gate_id") != gate_id
        or value.get("candidate") != dual["candidate"]
        or value.get("bank") != gate["bank"]
        or value.get("verdict") != verdict
        or evidence.get("summary") != value.get("summary")
        or sha256_file(source_evidence_path) != sha256_file(evidence_path)
        or evidence.get("aggregate", {}).get("sha256") != value["aggregate"]["sha256"]
        or aggregate.get("summary") != value.get("summary")
        or value.get("passed") is not bool(verdict["passed"])
        or (require_pass and not verdict["passed"])
        or value.get("rank4_replacement_authorized") is not False
        or value.get("upload_authorized") is not False
    ):
        raise ChallengerError(f"{gate_id} final result changed or failed")
    _validate_final_gate_evidence(
        source_evidence_path,
        dual=dual,
        dual_path=pathlib.Path(value["dual_final_plan"]["path"]),
        gate=gate,
    )
    utc(value.get("completed_at_utc"), f"{gate_id} completion timestamp")
    return value


def complete_dual_final(
    dual_reference: pathlib.Path, *, plan_path: pathlib.Path,
    result_a: pathlib.Path, result_b: pathlib.Path,
    completed_at_utc: str,
    bank_validator: BankValidator = openings.validate_bank,
) -> pathlib.Path:
    state = validate_dual_final(dual_reference, plan_path=plan_path, bank_validator=bank_validator)
    first = validate_final_result(result_a, dual=state["plan"], gate_id="gate-a")
    second = validate_final_result(result_b, dual=state["plan"], gate_id="gate-b")
    if result_a.resolve() == result_b.resolve() or first["candidate"] != second["candidate"]:
        raise ChallengerError("dual final reused a result or changed the candidate")
    entries = load_ledger(state["context"]["plan"])
    recorded = {
        entry.get("gate_id"): entry.get("result")
        for entry in entries
        if entry["event"] == "final-gate-recorded"
        and entry["attempt"] == state["plan"]["attempt"]
    }
    if recorded != {
        "gate-a": _sealed_record(result_a, FINAL_RESULT_SCHEMA),
        "gate-b": _sealed_record(result_b, FINAL_RESULT_SCHEMA),
    }:
        raise ChallengerError("dual final ledger does not bind both independent results")
    output = dual_reference.parent / "dual-qualified.json"
    body = {
        "schema": DUAL_QUALIFICATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": state["plan"]["attempt"],
        "status": "two-independent-strict-final-gates-passed",
        "dual_final_plan": _sealed_record(state["path"], DUAL_FINAL_SCHEMA),
        "candidate": dict(first["candidate"]),
        "gate_results": [
            _sealed_record(result_a, FINAL_RESULT_SCHEMA),
            _sealed_record(result_b, FINAL_RESULT_SCHEMA),
        ],
        "thresholds": FINAL_THRESHOLDS,
        "completed_at_utc": utc(completed_at_utc, "dual final completion timestamp"),
        "candidate_unchanged": True,
        "independent_banks": True,
        "rank4_replacement_authorized": False,
        "upload_authorized": False,
    }
    if output.exists():
        if qualification.load_sealed(output, DUAL_QUALIFICATION_SCHEMA) != qualification.seal(body):
            raise ChallengerError("dual final qualification changed")
        return output
    qualification.write_sealed(output, body)
    _append_event(
        state["context"],
        attempt=state["plan"]["attempt"],
        event="dual-final-qualified",
        created_at_utc=completed_at_utc,
        fields={
            "qualification": _sealed_record(output, DUAL_QUALIFICATION_SCHEMA),
            "candidate_unchanged": True,
            "gates_passed": 2,
            "rank4_replacement_authorized": False,
            "upload_authorized": False,
        },
    )
    return output


def record_upload_attestation(
    plan_path: pathlib.Path, *, submission_attestation: pathlib.Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Bind one already-completed CodinGame upload; never performs the upload."""

    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    if latest.get("event") == "upload-attested":
        if latest.get("source_submission_attestation") != _sealed_record(
            submission_attestation, qualification.UPLOAD_EVENT_SCHEMA
        ):
            raise ChallengerError("existing upload attestation uses another submission")
        return dict(latest)
    if latest.get("event") != "dual-final-qualified":
        raise ChallengerError("upload attestation requires the latest dual qualification")
    qualified_path = _verify_sealed_record(
        latest.get("qualification"), DUAL_QUALIFICATION_SCHEMA,
        "upload dual qualification",
    )
    qualified = qualification.load_sealed(
        qualified_path, DUAL_QUALIFICATION_SCHEMA
    )
    try:
        attestation = qualification.load_sealed(
            submission_attestation, qualification.UPLOAD_EVENT_SCHEMA
        )
    except Exception as error:
        raise ChallengerError("submission attestation is invalid") from error
    authorization_reference = attestation.get("authorization")
    if not isinstance(authorization_reference, Mapping):
        raise ChallengerError("submission attestation omits upload authorization")
    authorization_path = pathlib.Path(str(authorization_reference.get("path", "")))
    try:
        authorization = qualification._load_authorization(authorization_path)
    except Exception as error:
        raise ChallengerError("upload authorization is invalid") from error
    candidate = qualified["candidate"]["source"]
    prior_uploads = [
        entry for entry in entries if entry.get("event") == "upload-attested"
    ]
    upload_ordinal = len(prior_uploads) + 1
    extra_authorizations = [
        entry for entry in entries
        if entry.get("event") == "additional-upload-authorized"
    ]
    if (
        submission_attestation.is_symlink()
        or submission_attestation.resolve()
        != authorization_path.parent / "upload/05-submission-attested.json"
        or attestation.get("status") != "submission-attested"
        or attestation.get("submit_clicks") != 1
        or not isinstance(attestation.get("agent_id"), int)
        or isinstance(attestation.get("agent_id"), bool)
        or attestation["agent_id"] <= 0
        or not isinstance(attestation.get("submission_id"), int)
        or isinstance(attestation.get("submission_id"), bool)
        or attestation["submission_id"] <= 0
        or authorization_reference
        != qualification.artifact_reference(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        )
        or authorization.get("uploads_authorized") != 1
        or authorization.get("rank4_replacement_authorized") is not False
        or authorization.get("candidate", {}).get("sha256") != candidate["sha256"]
        or authorization.get("candidate", {}).get("bytes") != candidate["bytes"]
        or attestation.get("source_sha256") != candidate["sha256"]
        or attestation.get("source_bytes") != candidate["bytes"]
        or attestation.get("candidate_commit") != authorization.get("candidate_commit")
        or any(
            entry.get("submission_id") == attestation["submission_id"]
            for entry in prior_uploads
        )
        or (
            upload_ordinal > 1
            and not any(
                entry.get("next_upload_ordinal") == upload_ordinal
                and entry.get("consumed") is False
                for entry in extra_authorizations
            )
        )
    ):
        raise ChallengerError("submission attestation is not unique/candidate authorized")
    artifact_root = (
        pathlib.Path(context["root"]) / "attempt-ledger/artifacts"
        / f"attempt-{latest['attempt']:03d}/upload-{upload_ordinal:02d}"
    )
    copied_attestation_record = _copy_attempt_artifact(
        _regular(submission_attestation), root=artifact_root
    )
    copied_authorization_record = _copy_attempt_artifact(
        _regular(authorization_path), root=artifact_root
    )
    event = _append_event(
        context, attempt=int(latest["attempt"]), event="upload-attested",
        created_at_utc=created_at_utc,
        fields={
            "qualification": latest["qualification"],
            "candidate": qualified["candidate"],
            "submission_attestation": {
                **copied_attestation_record,
                "schema": qualification.UPLOAD_EVENT_SCHEMA,
                "body_sha256": attestation["body_sha256"],
            },
            "upload_authorization": {
                **copied_authorization_record,
                "schema": qualification.UPLOAD_AUTH_SCHEMA,
                "body_sha256": authorization["body_sha256"],
            },
            "source_submission_attestation": _sealed_record(
                submission_attestation, qualification.UPLOAD_EVENT_SCHEMA
            ),
            "agent_id": attestation["agent_id"],
            "submission_id": attestation["submission_id"],
            "candidate_commit": attestation["candidate_commit"],
            "upload_ordinal": upload_ordinal,
            "submit_clicks": 1,
            "adaptation_route": "record-live-window",
            "automatic_action": False,
        },
    )
    return event


def _default_live_window_validator(
    reference_path: pathlib.Path, data_root: pathlib.Path,
) -> Mapping[str, Any]:
    module = _load(LIVE_WINDOW_PATH, "rank4_teacher_challenger_live_window")
    return module.verify_window_reference(reference_path, data_root=data_root)


def record_live_window(
    plan_path: pathlib.Path, *, live_reference: pathlib.Path,
    live_data_root: pathlib.Path, created_at_utc: str,
    dynamic_exclusion_path: pathlib.Path | None = None,
    live_validator: Callable[[pathlib.Path, pathlib.Path], Mapping[str, Any]] = (
        _default_live_window_validator
    ),
    live_fingerprint_extractor: LiveFingerprintExtractor = (
        _default_live_fingerprint_extractor
    ),
    allow_injected_test_evidence: bool = False,
) -> dict[str, Any]:
    """Validate and bind an existing exact-90 live diagnostic; never polls it."""

    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    if latest.get("event") == "live-window-recorded":
        if latest.get("source_live_reference") != _sealed_record(
            live_reference, str(qualification.load_sealed(live_reference)["schema"])
        ):
            raise ChallengerError("existing live window uses another reference")
        if latest.get("passed") is False:
            if dynamic_exclusion_path is None:
                raise ChallengerError(
                    "rejected live window lost its trusted dynamic exclusion"
                )
            existing_dynamic = _verify_dynamic_exclusion_record(
                latest.get("dynamic_exclusion"), "recorded live exclusion"
            )
            if sha256_file(existing_dynamic) != sha256_file(dynamic_exclusion_path):
                raise ChallengerError("existing live window uses another exclusion")
            _validate_live_dynamic_match(
                existing_dynamic,
                candidate_source_sha256=latest["candidate"]["source"]["sha256"],
                attempt=int(latest["attempt"]),
                upload_ordinal=int(latest["upload_ordinal"]),
                live_reference=live_reference, live_data_root=live_data_root,
                extractor=live_fingerprint_extractor,
                allow_injected_test_evidence=allow_injected_test_evidence,
            )
        return dict(latest)
    if (
        latest.get("event") != "upload-attested"
        or latest.get("adaptation_route") != "record-live-window"
    ):
        raise ChallengerError("live window requires the latest upload attestation")
    try:
        validated_reference = dict(live_validator(live_reference, live_data_root))
    except Exception as error:
        raise ChallengerError("live-window reference failed full validation") from error
    try:
        reference = qualification.load_sealed(live_reference)
        receipt_reference = reference["receipt"]
        receipt_path = pathlib.Path(str(receipt_reference["path"])).resolve()
        receipt = qualification.load_sealed(receipt_path)
    except Exception as error:
        raise ChallengerError("live-window receipt closure is invalid") from error
    result = receipt if isinstance(receipt.get("summary"), Mapping) else validated_reference
    if (
        validated_reference.get("status") != result.get("summary", {}).get("status")
        or validated_reference.get("exact_games") != 90
    ):
        raise ChallengerError("live-window reference/receipt status changed")
    source_attestation = latest["source_submission_attestation"]
    identity = result.get("identity")
    summary = result.get("summary")
    live_status = summary.get("status") if isinstance(summary, Mapping) else None
    clean = bool(
        result.get("exact_games") == 90
        and live_status == "complete-accepted-diagnostic"
        and result.get("training_eligible") is False
        and result.get("rollback_authorized") is False
        and result.get("second_upload_authorized") is False
        and isinstance(result.get("game_ids"), list)
        and len(result["game_ids"]) == 90
        and len(set(result["game_ids"])) == 90
        and isinstance(identity, Mapping)
        and identity.get("source_sha256")
        == latest["candidate"]["source"]["sha256"]
        and identity.get("source_bytes")
        == latest["candidate"]["source"]["bytes"]
        and identity.get("agent_id") == latest["agent_id"]
        and identity.get("submission_id") == latest["submission_id"]
        and identity.get("repository_commit") == latest["candidate_commit"]
        and isinstance(summary, Mapping)
        and summary.get("focus_operational_failures") == []
        and summary.get("focus_operational_failure_games") == 0
        and receipt.get("submission_attestation", {}).get("sha256")
        == source_attestation["sha256"]
    )
    if result.get("exact_games") != 90:
        raise ChallengerError("live diagnostic is not exactly 90 games")
    dynamic_exclusion = None
    if clean:
        if dynamic_exclusion_path is not None:
            raise ChallengerError("passing live window cannot add a retry exclusion")
    else:
        if dynamic_exclusion_path is None:
            raise ChallengerError(
                "rejected live window requires sanitized dynamic fingerprints"
            )
        exclusion = _validate_live_dynamic_match(
            dynamic_exclusion_path,
            candidate_source_sha256=latest["candidate"]["source"]["sha256"],
            attempt=int(latest["attempt"]),
            upload_ordinal=int(latest["upload_ordinal"]),
            live_reference=live_reference, live_data_root=live_data_root,
            extractor=live_fingerprint_extractor,
            allow_injected_test_evidence=allow_injected_test_evidence,
        )
        game_ids_sha256 = sha256_bytes(canonical_json_bytes(sorted(result["game_ids"])))
        if (
            exclusion.get("classification")
            != "live-diagnostic-canonical-fingerprints"
            or exclusion.get("attempt") != latest["attempt"]
            or exclusion.get("gate_id")
            != f"live-upload-{latest['upload_ordinal']}"
            or exclusion.get("origin") != {
                "candidate_source_sha256": latest["candidate"]["source"]["sha256"],
                "live_receipt_sha256": sha256_file(receipt_path),
                "game_ids_sha256": game_ids_sha256,
            }
        ):
            raise ChallengerError("live sanitized exclusion disagrees with rejected window")
    artifact_root = (
        pathlib.Path(context["root"]) / "attempt-ledger/artifacts"
        / f"attempt-{latest['attempt']:03d}/live"
    )
    copied_reference = _copy_attempt_artifact(
        _regular(live_reference), root=artifact_root
    )
    copied_receipt = _copy_attempt_artifact(
        _regular(receipt_path), root=artifact_root
    )
    if dynamic_exclusion_path is not None:
        copied_dynamic = _copy_attempt_artifact(
            _regular(dynamic_exclusion_path), root=artifact_root / "exclusions"
        )
        copied_dynamic_path = pathlib.Path(copied_dynamic["path"])
        # Preserve the sealed payload exactly; the copied record is what all
        # future attempts bind, never the live transcript/metric closure.
        dynamic_exclusion = _dynamic_exclusion_record(copied_dynamic_path)
    return _append_event(
        context, attempt=int(latest["attempt"]), event="live-window-recorded",
        created_at_utc=created_at_utc,
        fields={
            "upload_ordinal": latest["upload_ordinal"],
            "submission_id": latest["submission_id"],
            "candidate": latest["candidate"],
            "source_live_reference": _sealed_record(
                live_reference, str(reference["schema"])
            ),
            "live_reference": copied_reference,
            "live_receipt": copied_receipt,
            "dynamic_exclusion": dynamic_exclusion,
            "status": live_status,
            "exact_games": 90,
            "focus_operational_failure_games": (
                summary.get("focus_operational_failure_games")
                if isinstance(summary, Mapping) else None
            ),
            "passed": clean,
            "adaptation_route": (
                "complete-campaign" if clean
                else "await-explicit-additional-upload-authorization"
            ),
            "automatic_action": False,
            "training_eligible": False,
        },
    )


def authorize_additional_upload(
    plan_path: pathlib.Path, *, authorization_path: pathlib.Path,
    created_at_utc: str,
    live_fingerprint_extractor: LiveFingerprintExtractor = (
        _default_live_fingerprint_extractor
    ),
    allow_injected_test_evidence: bool = False,
) -> dict[str, Any]:
    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    if (
        latest.get("event") != "live-window-recorded"
        or latest.get("passed") is not False
        or latest.get("adaptation_route")
        != "await-explicit-additional-upload-authorization"
    ):
        raise ChallengerError("additional upload authorization lacks a rejected live window")
    dynamic_path = _verify_dynamic_exclusion_record(
        latest.get("dynamic_exclusion"), "rejected live dynamic exclusion"
    )
    _validate_live_dynamic_match(
        dynamic_path,
        candidate_source_sha256=latest["candidate"]["source"]["sha256"],
        attempt=int(latest["attempt"]),
        upload_ordinal=int(latest["upload_ordinal"]),
        extractor=live_fingerprint_extractor,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    authorization = qualification.load_sealed(
        authorization_path, ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA
    )
    next_attempt = int(latest["attempt"]) + 1
    next_upload = int(latest["upload_ordinal"]) + 1
    if (
        authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("previous_attempt") != latest["attempt"]
        or authorization.get("next_attempt") != next_attempt
        or authorization.get("next_upload_ordinal") != next_upload
        or authorization.get("rejected_live_reference")
        != latest["source_live_reference"]
        or authorization.get("rejected_live_dynamic_exclusion")
        != latest["dynamic_exclusion"]
        or authorization.get("explicit_user_authorization") is not True
        or authorization.get("attempt_openings_authorized") != 1
        or authorization.get("additional_uploads_authorized") != 1
        or authorization.get("protected_or_live_data_training_allowed") is not False
        or authorization.get("automatic_action") is not False
    ):
        raise ChallengerError("additional upload authorization changed")
    artifact_root = (
        pathlib.Path(context["root"]) / "attempt-ledger/artifacts"
        / f"attempt-{latest['attempt']:03d}/additional-upload"
    )
    copied = _copy_attempt_artifact(_regular(authorization_path), root=artifact_root)
    return _append_event(
        context, attempt=int(latest["attempt"]),
        event="additional-upload-authorized", created_at_utc=created_at_utc,
        fields={
            "authorization": {
                **copied,
                "schema": ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA,
                "body_sha256": authorization["body_sha256"],
            },
            "next_attempt": next_attempt,
            "next_upload_ordinal": next_upload,
            "consumed": False,
            "adaptation_route": "open-next-attempt-explicit-after-live-failure",
            "automatic_action": False,
        },
    )


def complete_campaign(
    plan_path: pathlib.Path, *, completed_at_utc: str,
) -> pathlib.Path:
    context = validate_campaign(plan_path)
    entries = load_ledger(context["plan"])
    latest = entries[-1]
    output = pathlib.Path(context["plan"]["outputs"]["completion"])
    if latest.get("event") == "campaign-complete":
        _verify_sealed_record(
            latest.get("completion"), CAMPAIGN_COMPLETION_SCHEMA,
            "campaign completion",
        )
        return output
    if (
        latest.get("event") != "live-window-recorded"
        or latest.get("passed") is not True
        or latest.get("adaptation_route") != "complete-campaign"
        or latest.get("exact_games") != 90
        or latest.get("focus_operational_failure_games") != 0
    ):
        raise ChallengerError("campaign completion requires a clean exact-90 live window")
    qualified = next(
        entry for entry in reversed(entries)
        if entry.get("event") == "dual-final-qualified"
        and entry.get("attempt") == latest["attempt"]
    )
    upload = next(
        entry for entry in reversed(entries)
        if entry.get("event") == "upload-attested"
        and entry.get("attempt") == latest["attempt"]
    )
    body = {
        "schema": CAMPAIGN_COMPLETION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "rank4-teacher-challenger-complete",
        "campaign_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "attempt": latest["attempt"],
        "candidate": latest["candidate"],
        "dual_qualification": qualified["qualification"],
        "upload_attestation": upload["submission_attestation"],
        "live_reference": latest["live_reference"],
        "completed_at_utc": utc(completed_at_utc, "campaign completion timestamp"),
        "proof": {
            "strict_final_gates": 2,
            "candidate_unchanged": True,
            "exact_live_games": 90,
            "focus_operational_failure_games": 0,
            "upload_ordinal": latest["upload_ordinal"],
        },
        "next_attempt_authorized": False,
        "additional_upload_authorized": False,
    }
    if output.exists():
        if qualification.load_sealed(output, CAMPAIGN_COMPLETION_SCHEMA) != qualification.seal(body):
            raise ChallengerError("campaign completion changed")
        return output
    qualification.write_sealed(output, body)
    _append_event(
        context, attempt=int(latest["attempt"]), event="campaign-complete",
        created_at_utc=completed_at_utc,
        fields={
            "completion": _sealed_record(output, CAMPAIGN_COMPLETION_SCHEMA),
            "candidate": latest["candidate"],
            "strict_final_gates": 2,
            "exact_live_games": 90,
            "focus_operational_failure_games": 0,
            "uploads_completed": latest["upload_ordinal"],
            "goal_achieved": True,
        },
    )
    return output


def _parse_named(values: Sequence[str]) -> dict[str, pathlib.Path]:
    result = {}
    for raw in values:
        if "=" not in raw:
            raise ChallengerError("named path must be NAME=PATH")
        name, path = raw.split("=", 1)
        if name in result:
            raise ChallengerError("named path is repeated")
        result[name] = pathlib.Path(path)
    return result


def _load_summary(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ChallengerError("cannot read final summary") from error
    if not isinstance(value, Mapping):
        raise ChallengerError("final summary must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Resource policy: at most 8 concurrent single-thread generation/label "
            "workers and 2 training seeds; "
            "1 for uncontended timing; exactly 4 per strict final gate. Commands "
            "never create recurring automations; resume explicitly from the ledger."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    live_fingerprints = commands.add_parser("sanitize-live-fingerprints")
    live_fingerprints.add_argument("--output", type=pathlib.Path, required=True)
    live_fingerprints.add_argument("--attempt", type=int, required=True)
    live_fingerprints.add_argument("--upload-ordinal", type=int, required=True)
    live_fingerprints.add_argument("--candidate-source-sha256", required=True)
    live_fingerprints.add_argument("--live-reference", type=pathlib.Path, required=True)
    live_fingerprints.add_argument("--live-data-root", type=pathlib.Path, required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--output", type=pathlib.Path, required=True)
    build.add_argument("--binary", action="append", required=True)
    build.add_argument("--created-at-utc", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output-root", type=pathlib.Path, required=True)
    freeze.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    freeze.add_argument("--candidate-source", type=pathlib.Path, required=True)
    freeze.add_argument("--rank4-source", type=pathlib.Path, default=RANK4_PATH)
    freeze.add_argument("--teacher-runtime", type=pathlib.Path, required=True)
    freeze.add_argument("--teacher-manifest", type=pathlib.Path, required=True)
    freeze.add_argument("--mixed-six-exclusion", type=pathlib.Path, required=True)
    freeze.add_argument("--fresh-exclusion-receipt", type=pathlib.Path, required=True)
    freeze.add_argument("--attempt-zero-recovery-plan", type=pathlib.Path, required=True)
    freeze.add_argument("--training-bundle-manifest", type=pathlib.Path, required=True)
    freeze.add_argument("--attempt-one-initial-checkpoint", type=pathlib.Path, required=True)
    freeze.add_argument("--prior-runtime", type=pathlib.Path, required=True)
    freeze.add_argument("--roots-tsv", type=pathlib.Path, required=True)
    freeze.add_argument("--roots-manifest", type=pathlib.Path, required=True)
    freeze.add_argument("--build-manifest", type=pathlib.Path, required=True)
    freeze.add_argument("--training-input", action="append", required=True)
    freeze.add_argument("--protected-exclusion", action="append", default=[])
    freeze.add_argument("--live-exclusion", action="append", default=[])
    freeze.add_argument("--created-at-utc", required=True)
    phase = commands.add_parser("materialize-phase")
    phase.add_argument("--plan", type=pathlib.Path, required=True)
    phase.add_argument("--attempt", type=int, required=True)
    phase.add_argument("--phase", choices=tuple(PHASE_QUOTAS), required=True)
    phase.add_argument("--created-at-utc", required=True)
    progress = commands.add_parser("record-progress")
    progress.add_argument("--plan", type=pathlib.Path, required=True)
    progress.add_argument("--attempt", type=int, required=True)
    progress.add_argument("--phase", choices=tuple(PHASE_QUOTAS), required=True)
    progress.add_argument("--pipeline-plan", type=pathlib.Path, required=True)
    progress.add_argument("--created-at-utc", required=True)
    zero_result = commands.add_parser("record-attempt-zero-result")
    zero_result.add_argument("--plan", type=pathlib.Path, required=True)
    zero_result.add_argument("--result", type=pathlib.Path, required=True)
    zero_result.add_argument("--created-at-utc", required=True)
    outcome = commands.add_parser("record-attempt-outcome")
    outcome.add_argument("--plan", type=pathlib.Path, required=True)
    outcome.add_argument("--attempt", type=int, required=True)
    outcome.add_argument("--phase", choices=tuple(PHASE_QUOTAS), required=True)
    outcome.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    outcome.add_argument("--candidate-source", type=pathlib.Path, required=True)
    outcome.add_argument("--outcome-receipt", type=pathlib.Path, required=True)
    outcome.add_argument("--admission-receipt", type=pathlib.Path, required=True)
    outcome.add_argument("--development-exclusion", type=pathlib.Path, required=True)
    outcome.add_argument("--metrics", type=pathlib.Path, required=True)
    outcome.add_argument("--strength-delta-pp", type=float, required=True)
    outcome.add_argument("--teacher-regret-reduction-fraction", type=float, required=True)
    outcome.add_argument("--created-at-utc", required=True)
    next_attempt = commands.add_parser("open-attempt")
    next_attempt.add_argument("--plan", type=pathlib.Path, required=True)
    next_attempt.add_argument("--attempt", type=int, required=True)
    next_attempt.add_argument("--hypothesis", required=True)
    next_attempt.add_argument("--intervention", required=True)
    next_attempt.add_argument("--attribution-receipt", type=pathlib.Path)
    next_attempt.add_argument("--student-runtime", type=pathlib.Path)
    next_attempt.add_argument("--prior-runtime", type=pathlib.Path)
    next_attempt.add_argument("--initial-float-checkpoint", type=pathlib.Path)
    next_attempt.add_argument("--roots-tsv", type=pathlib.Path)
    next_attempt.add_argument("--roots-manifest", type=pathlib.Path)
    next_attempt.add_argument("--build-manifest", type=pathlib.Path)
    next_attempt.add_argument("--created-at-utc", required=True)
    dual_authorize = commands.add_parser("authorize-dual-final")
    dual_authorize.add_argument("--plan", type=pathlib.Path, required=True)
    dual_authorize.add_argument("--attempt", type=int, required=True)
    dual_authorize.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    dual_authorize.add_argument("--candidate-source", type=pathlib.Path, required=True)
    dual_authorize.add_argument("--release-evidence", type=pathlib.Path, required=True)
    dual_authorize.add_argument("--deployed-source", type=pathlib.Path, required=True)
    dual_authorize.add_argument("--created-at-utc", required=True)
    dual = commands.add_parser("prepare-dual-final")
    dual.add_argument("--plan", type=pathlib.Path, required=True)
    dual.add_argument("--authorization", type=pathlib.Path, required=True)
    dual.add_argument("--bank-a", type=pathlib.Path, required=True)
    dual.add_argument("--bank-b", type=pathlib.Path, required=True)
    dual.add_argument("--created-at-utc", required=True)
    result = commands.add_parser("record-final-result")
    result.add_argument("--plan", type=pathlib.Path, required=True)
    result.add_argument("--dual-reference", type=pathlib.Path, required=True)
    result.add_argument("--gate-id", choices=("gate-a", "gate-b"), required=True)
    result.add_argument("--evidence", type=pathlib.Path, required=True)
    result.add_argument("--completed-at-utc", required=True)
    complete = commands.add_parser("complete-dual-final")
    complete.add_argument("--plan", type=pathlib.Path, required=True)
    complete.add_argument("--dual-reference", type=pathlib.Path, required=True)
    complete.add_argument("--result-a", type=pathlib.Path, required=True)
    complete.add_argument("--result-b", type=pathlib.Path, required=True)
    complete.add_argument("--completed-at-utc", required=True)
    upload = commands.add_parser("record-upload-attestation")
    upload.add_argument("--plan", type=pathlib.Path, required=True)
    upload.add_argument("--submission-attestation", type=pathlib.Path, required=True)
    upload.add_argument("--created-at-utc", required=True)
    live = commands.add_parser("record-live-window")
    live.add_argument("--plan", type=pathlib.Path, required=True)
    live.add_argument("--live-reference", type=pathlib.Path, required=True)
    live.add_argument("--live-data-root", type=pathlib.Path, required=True)
    live.add_argument("--dynamic-exclusion", type=pathlib.Path)
    live.add_argument("--created-at-utc", required=True)
    extra = commands.add_parser("authorize-additional-upload")
    extra.add_argument("--plan", type=pathlib.Path, required=True)
    extra.add_argument("--authorization", type=pathlib.Path, required=True)
    extra.add_argument("--created-at-utc", required=True)
    finish = commands.add_parser("complete-campaign")
    finish.add_argument("--plan", type=pathlib.Path, required=True)
    finish.add_argument("--completed-at-utc", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "sanitize-live-fingerprints":
            output = materialize_live_dynamic_exclusion(
                arguments.output,
                attempt=arguments.attempt,
                upload_ordinal=arguments.upload_ordinal,
                candidate_source_sha256=arguments.candidate_source_sha256,
                live_reference=arguments.live_reference,
                live_data_root=arguments.live_data_root,
            )
        elif arguments.command == "build-manifest":
            output = create_build_manifest(
                arguments.output,
                binaries=_parse_named(arguments.binary),
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.command == "freeze":
            output = freeze_campaign(
                output_root=arguments.output_root,
                candidate_runtime=arguments.candidate_runtime,
                candidate_source=arguments.candidate_source,
                rank4_source=arguments.rank4_source,
                teacher_runtime=arguments.teacher_runtime,
                teacher_manifest=arguments.teacher_manifest,
                mixed_six_exclusion=arguments.mixed_six_exclusion,
                fresh_exclusion_receipt=arguments.fresh_exclusion_receipt,
                attempt_zero_recovery_plan=arguments.attempt_zero_recovery_plan,
                training_bundle_manifest=arguments.training_bundle_manifest,
                attempt_one_initial_checkpoint=(
                    arguments.attempt_one_initial_checkpoint
                ),
                prior_runtime=arguments.prior_runtime,
                roots_tsv=arguments.roots_tsv,
                roots_manifest=arguments.roots_manifest,
                build_manifest=arguments.build_manifest,
                training_inputs=_parse_named(arguments.training_input),
                protected_exclusions=_parse_named(arguments.protected_exclusion),
                live_exclusions=_parse_named(arguments.live_exclusion),
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.command == "materialize-phase":
            output = materialize_phase(
                arguments.plan, attempt=arguments.attempt, phase=arguments.phase,
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.command == "record-progress":
            event = record_progress(
                arguments.plan, attempt=arguments.attempt, phase=arguments.phase,
                pipeline_plan_path=arguments.pipeline_plan,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "route": event["adaptation_route"]}, sort_keys=True))
            return 0
        elif arguments.command == "record-attempt-zero-result":
            event = record_attempt_zero_result(
                arguments.plan, result_path=arguments.result,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "route": event["adaptation_route"]}, sort_keys=True))
            return 0
        elif arguments.command == "record-attempt-outcome":
            event = record_attempt_outcome(
                arguments.plan, attempt=arguments.attempt, phase=arguments.phase,
                candidate_runtime=arguments.candidate_runtime,
                candidate_source=arguments.candidate_source,
                outcome_receipt=arguments.outcome_receipt,
                admission_receipt=arguments.admission_receipt,
                development_exclusion=arguments.development_exclusion,
                metrics=_load_summary(arguments.metrics),
                strength_delta_pp=arguments.strength_delta_pp,
                teacher_regret_reduction_fraction=arguments.teacher_regret_reduction_fraction,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "admitted": event["admitted"], "route": event["adaptation_route"]}, sort_keys=True))
            return 0
        elif arguments.command == "open-attempt":
            event = open_next_attempt(
                arguments.plan, attempt=arguments.attempt,
                hypothesis=arguments.hypothesis,
                intervention=arguments.intervention,
                attribution_receipt=arguments.attribution_receipt,
                student_runtime=arguments.student_runtime,
                prior_runtime=arguments.prior_runtime,
                initial_float_checkpoint=arguments.initial_float_checkpoint,
                roots_tsv=arguments.roots_tsv,
                roots_manifest=arguments.roots_manifest,
                build_manifest=arguments.build_manifest,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "attempt": event["attempt"]}, sort_keys=True))
            return 0
        elif arguments.command == "authorize-dual-final":
            output = authorize_dual_final(
                arguments.plan, attempt=arguments.attempt,
                candidate_runtime=arguments.candidate_runtime,
                candidate_source=arguments.candidate_source,
                release_evidence_path=arguments.release_evidence,
                deployed_source=arguments.deployed_source,
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.command == "prepare-dual-final":
            output = prepare_dual_final(
                arguments.authorization, plan_path=arguments.plan,
                bank_a=arguments.bank_a, bank_b=arguments.bank_b,
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.command == "record-final-result":
            output = record_final_result(
                arguments.dual_reference, plan_path=arguments.plan,
                gate_id=arguments.gate_id, evidence_path=arguments.evidence,
                completed_at_utc=arguments.completed_at_utc,
            )
        elif arguments.command == "complete-dual-final":
            output = complete_dual_final(
                arguments.dual_reference, plan_path=arguments.plan,
                result_a=arguments.result_a, result_b=arguments.result_b,
                completed_at_utc=arguments.completed_at_utc,
            )
        elif arguments.command == "record-upload-attestation":
            event = record_upload_attestation(
                arguments.plan,
                submission_attestation=arguments.submission_attestation,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "upload_ordinal": event["upload_ordinal"]}, sort_keys=True))
            return 0
        elif arguments.command == "record-live-window":
            event = record_live_window(
                arguments.plan, live_reference=arguments.live_reference,
                live_data_root=arguments.live_data_root,
                dynamic_exclusion_path=arguments.dynamic_exclusion,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "passed": event["passed"]}, sort_keys=True))
            return 0
        elif arguments.command == "authorize-additional-upload":
            event = authorize_additional_upload(
                arguments.plan, authorization_path=arguments.authorization,
                created_at_utc=arguments.created_at_utc,
            )
            print(json.dumps({"event": event["event"], "next_attempt": event["next_attempt"]}, sort_keys=True))
            return 0
        elif arguments.command == "complete-campaign":
            output = complete_campaign(
                arguments.plan, completed_at_utc=arguments.completed_at_utc
            )
        else:
            context = validate_campaign(arguments.plan)
            entries = load_ledger(context["plan"])
            print(json.dumps({
                "status": context["plan"]["status"],
                "ledger_events": len(entries),
                "latest_event": entries[-1]["event"] if entries else None,
            }, sort_keys=True))
            return 0
    except (ChallengerError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
