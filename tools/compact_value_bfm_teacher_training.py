#!/usr/bin/env python3
"""Receipt-backed training and Rank-4 screening for the teacher challenger.

This adapter is the bridge between a finalized pilot/full label pipeline and
``compact_value_bfm_train``.  It deliberately keeps the frozen 6301x12x8x1
value-only runtime contract: the rich complete-turn labels affect the loss,
not the deployed interface or architecture.

The operations are explicit and resumable:

``prepare``
    Validate the challenger campaign/phase and finalized scalar/successor
    artifacts, copy the exact content-addressed float initialization, and seal
    an extended train/validation/symmetry-isolation audit.
``train``
    Run the fixed three seeds with two seed workers for the scalar control and
    ranking arms, export/compact every selected runtime, and freeze the model
    choice before any Rank-4 screening bank is read.
``prepare-gate``
    Bind a fresh unprotected bank and compile the independently testable
    phase-specific pilot screen or complete full search A/B roster around the
    already-selected model.
``run-gate``
    Execute each whole-bank variant serially with one nice-zero, single-thread
    process and seal the prelaunch/no-retry evidence.
``select-full-search`` / ``prepare-full-qualification``
    Freeze the full-round A/B winner before opening a separate, disjoint
    qualification bank; that plan is then executed with ``run-gate`` again.
``admit``
    Consume the sealed gate execution, choose the search variant only after
    model selection, and emit governance-compatible phase evidence.

No operation opens a protected model-validation split, schedules recurring
work, uploads a source, or changes Rank 4.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import functools
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
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# This CLI imports NumPy before loading the maintained trainer.  Re-exec before
# that import so Accelerate/OpenBLAS/MKL see the same one-thread contract that
# the trainer later validates and records for its concurrent seed workers.
_TRAINING_NATIVE_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_TRAINING_NATIVE_THREAD_PREIMPORT_MARKER = (
    "PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"
)
if __name__ == "__main__" and (
    os.environ.get(_TRAINING_NATIVE_THREAD_PREIMPORT_MARKER) != "1"
    or any(
        os.environ.get(name) != value
        for name, value in _TRAINING_NATIVE_THREAD_ENVIRONMENT.items()
    )
):
    _training_environment = dict(os.environ)
    _training_environment.update(_TRAINING_NATIVE_THREAD_ENVIRONMENT)
    _training_environment[_TRAINING_NATIVE_THREAD_PREIMPORT_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
        _training_environment,
    )

import numpy as np


REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import compact_value_bfm_pilot_pipeline as pipeline
from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import compact_value_bfm_train as trainer
from tools import jacek_replay_corpus as corpus
from tools import jacek_replay_features as features


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load teacher-training dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


COMPACT_DIRECTORY = (
    REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
)
RANK4_SOURCE = REPOSITORY / "submissions/codingame/bots/rank_4/submission.cpp"
GATE_SOURCE = COMPACT_DIRECTORY / "rank4_gate.cpp"
MODEL_EXPORTER_PATH = COMPACT_DIRECTORY / "export_model.py"
SOURCE_EXPORTER_PATH = COMPACT_DIRECTORY / "export_submission.py"
GATE_SUPPORT_PATH = COMPACT_DIRECTORY / "rank4_gate_support.py"
SOURCE_CONFIG = COMPACT_DIRECTORY / "submission.json"
SOURCE_MANIFEST = COMPACT_DIRECTORY / "sources.txt"

model_exporter = _load(MODEL_EXPORTER_PATH, "teacher_training_model_exporter")
source_exporter = _load(SOURCE_EXPORTER_PATH, "teacher_training_source_exporter")
gate_support = _load(GATE_SUPPORT_PATH, "teacher_training_gate_support")


class TeacherTrainingError(ValueError):
    """A training, source, gate, or resume binding is inconsistent."""


PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-training-plan.v1"
)
SELECTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-training-selection.v1"
)
SELECTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-training-selection-reference.v1"
)
GATE_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-plan.v1"
)
GATE_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-reference.v1"
)
GATE_EXECUTION_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-execution-claim.v1"
)
GATE_VARIANT_EXECUTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-variant-execution.v1"
)
GATE_EXECUTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-execution.v1"
)
GATE_EXECUTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-execution-reference.v1"
)
GATE_ABANDONMENT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-abandonment.v1"
)
FULL_SEARCH_SELECTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-search-selection.v1"
)
FULL_SEARCH_SELECTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-search-selection-reference.v1"
)
FULL_QUALIFICATION_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-qualification-plan.v1"
)
FULL_QUALIFICATION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-qualification-reference.v1"
)
SCREEN_REQUEST_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-screen-request.v1"
)
ADMISSION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-phase-admission.v1"
)
ADMISSION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-phase-admission-reference.v1"
)
DEVELOPMENT_FINGERPRINT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-development-fingerprints.v1"
)
DEVELOPMENT_BANK_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-development-bank-claim.v1"
)
DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-development-bank-seed-claim.v1"
)

ARCHITECTURE = trainer.ARCHITECTURES["capacity-12x8"]
ARM = trainer.ARMS["search-target"]
SOURCE_LIMIT_EXCLUSIVE = 95_000
SOURCE_RESERVE_TARGET = 2_000
SOURCE_MAXIMUM_FOR_TARGET = SOURCE_LIMIT_EXCLUSIVE - SOURCE_RESERVE_TARGET
SEED_WORKERS = 2
PILOT_WEIGHTS = (0.0, 0.10, 0.25)
PILOT_PAIRS = 100
FULL_PAIRS = 500
PILOT_MINIMUM_WINS = 105
FULL_MINIMUM_WINS = 550
FULL_MINIMUM_COLOR_WINS = 265
ATTEMPT_ONE_INITIAL_CHECKPOINT_SHA256 = (
    "0dbe279295bcfeb80392e10ff9b04f9bd5c0eb4e1ea45c974b6a92d266f37729"
)

SEARCH_VARIANTS: dict[str, tuple[str, ...]] = {
    "baseline": (
        "COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT",
        "COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT",
    ),
    "no-feature-sort-only": (
        "COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT",
    ),
    "single-pass-selection-only": (
        "COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT",
    ),
    "combined": (),
}
SEARCH_VARIANT_ORDER = tuple(SEARCH_VARIANTS)

# The first round's four search variants remain the complete, unchanged
# ``standard-v1`` roster.  A throughput intervention is always an additional
# macro layered independently onto *each* of those four standard bases.  This
# keeps the original feature-sort/descendant-selection A/B interpretable while
# giving every intervention arm an exact like-for-like control.
SEARCH_THROUGHPUT_PROFILE_MACROS: dict[str, str | None] = {
    "standard-v1": None,
    "state-evaluation-cache-v1": "COMPACT_VALUE_BFM_STATE_EVALUATION_CACHE_V1",
    "progressive-widening-v1": "COMPACT_VALUE_BFM_PROGRESSIVE_WIDENING_V1",
    "subtree-reuse-v1": "COMPACT_VALUE_BFM_SUBTREE_REUSE_V1",
}
SEARCH_THROUGHPUT_PROFILE_ORDER = tuple(SEARCH_THROUGHPUT_PROFILE_MACROS)
TREATMENT_SEPARATOR = "--"


def _search_throughput_profile(
    adaptation_contract: Mapping[str, Any],
) -> str:
    try:
        normalized = challenger._validated_adaptation_contract(
            adaptation_contract
        )
    except Exception as error:
        raise TeacherTrainingError(
            "search-throughput adaptation contract is invalid"
        ) from error
    profile = normalized["search_throughput_profile"]
    if profile not in SEARCH_THROUGHPUT_PROFILE_MACROS:
        raise TeacherTrainingError("search-throughput profile is unsupported")
    return profile


def _treatment_variant(base_variant: str, profile: str) -> str:
    if base_variant not in SEARCH_VARIANTS or profile == "standard-v1":
        raise TeacherTrainingError("search treatment variant is invalid")
    if profile not in SEARCH_THROUGHPUT_PROFILE_MACROS:
        raise TeacherTrainingError("search treatment profile is unsupported")
    return f"{base_variant}{TREATMENT_SEPARATOR}{profile}"


def active_search_variants(profile: str) -> dict[str, tuple[str, ...]]:
    """Return the exact ordered gate roster for one immutable profile.

    ``standard-v1`` deliberately returns only the historical four variants.
    Intervention profiles append one treatment for every corresponding base;
    no treatment is allowed to stand in for or omit its control.
    """

    if profile not in SEARCH_THROUGHPUT_PROFILE_MACROS:
        raise TeacherTrainingError("search-throughput profile is unsupported")
    variants = dict(SEARCH_VARIANTS)
    macro = SEARCH_THROUGHPUT_PROFILE_MACROS[profile]
    if macro is None:
        return variants
    for base in SEARCH_VARIANT_ORDER:
        variants[_treatment_variant(base, profile)] = (*SEARCH_VARIANTS[base], macro)
    return variants


def phase_gate_variants(
    phase: str, profile: str,
) -> dict[str, tuple[str, ...]]:
    """Return the plan-aligned development-gate roster for one phase.

    Pilot screens isolate model strength: standard attempts use only the
    reference baseline, while a throughput intervention adds only its paired
    baseline treatment.  Full search A/B retains the complete 4/8-arm roster.
    """

    active = active_search_variants(profile)
    if phase == "full":
        return active
    if phase != "pilot":
        raise TeacherTrainingError("development-gate phase is unsupported")
    pilot = {"baseline": active["baseline"]}
    if profile != "standard-v1":
        treatment = _treatment_variant("baseline", profile)
        pilot[treatment] = active[treatment]
    return pilot


def _expected_gate_game_volume(
    variants: Sequence[str], *, pairs_per_variant: int,
) -> dict[str, object]:
    if (
        not variants
        or pairs_per_variant <= 0
        or len(set(variants)) != len(variants)
    ):
        raise TeacherTrainingError("development-gate game volume is invalid")
    return {
        "variant_count": len(variants),
        "pairs_per_variant": pairs_per_variant,
        "games_per_variant": 2 * pairs_per_variant,
        "total_pairs": len(variants) * pairs_per_variant,
        "total_games": len(variants) * 2 * pairs_per_variant,
        "each_pair_balanced_by_candidate_color": True,
    }


def _search_variant_metadata(profile: str, variant: str) -> dict[str, object]:
    variants = active_search_variants(profile)
    if variant not in variants:
        raise TeacherTrainingError("search variant is outside the active profile")
    suffix = f"{TREATMENT_SEPARATOR}{profile}"
    treatment = profile != "standard-v1" and variant.endswith(suffix)
    base = variant.removesuffix(suffix) if treatment else variant
    if base not in SEARCH_VARIANTS:
        raise TeacherTrainingError("search variant has no standard base")
    return {
        "search_throughput_profile": profile,
        "candidate_search_profile": profile if treatment else "standard-v1",
        "variant": variant,
        "standard_base_variant": base,
        "is_treatment": treatment,
        "profile_compile_macro": (
            SEARCH_THROUGHPUT_PROFILE_MACROS[profile] if treatment else None
        ),
        "compile_time_macros": list(variants[variant]),
    }

GATE_CONFIGURATION = {
    "mode": "actual-clock",
    "pair_offset": 0,
    "candidate_c": 0.95,
    "candidate_fpu": 0.5,
    "candidate_lambda": 1.0,
    "candidate_actions": 250,
    "candidate_root_partial_paths": 4_000,
    "candidate_nonroot_partial_paths": 512,
    "candidate_nodes": 80_000,
    "candidate_expansions": 2_000_000,
    "candidate_shuffle_seed": 1,
    "candidate_clocks_ms": [800, 155],
    "rank4_nodes": 3_000_000,
    "rank4_clocks_ms": [800, 165],
    "max_turns": 320,
}
GATE_WORKERS = 1
GATE_THREADS_PER_WORKER = 1
GATE_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
GATE_PROCESS_TIMEOUT_SECONDS = 86_400

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _production_execution_base(campaign: Mapping[str, Any]) -> pathlib.Path:
    root = campaign.get("root")
    if root is None:
        root = campaign.get("plan", {}).get("outputs", {}).get("root")
    if not isinstance(root, (str, os.PathLike)):
        raise TeacherTrainingError("campaign has no production execution root")
    return pathlib.Path(root).resolve() / pipeline.PRODUCTION_EXECUTION_DIRECTORY


def _guard_test_hooks(
    plan: Mapping[str, Any], *, hooks_used: bool,
    allow_injected_test_evidence: bool,
) -> None:
    if not hooks_used:
        return
    authority = plan.get("execution_authority")
    production = bool(
        isinstance(authority, Mapping)
        and authority.get("production_allowlist_enforced") is True
    )
    if production or allow_injected_test_evidence is not True:
        raise TeacherTrainingError(
            "injected teacher-training hooks are explicit nonproduction test evidence only"
        )


@contextlib.contextmanager
def _heavy_stage_lease(plan: Mapping[str, Any]):
    authority = plan.get("execution_authority")
    lock_value = (
        authority.get("heavy_stage_lock")
        if isinstance(authority, Mapping) else None
    )
    if lock_value is None:
        yield
        return
    lock_path = pathlib.Path(str(lock_value))
    if not lock_path.is_absolute() or lock_path.name != (
        ".rank4-teacher-heavy-stage.lock"
    ):
        raise TeacherTrainingError("teacher-training heavy-stage lock route changed")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise TeacherTrainingError(
                "another Rank-4 campaign heavy stage is active"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _heavy_stage(function):
    @functools.wraps(function)
    def guarded(plan_path: pathlib.Path, *args, **kwargs):
        plan = load_training_plan(plan_path.resolve())
        with _heavy_stage_lease(plan):
            return function(plan_path, *args, **kwargs)
    return guarded


def required_build_sources() -> tuple[str, ...]:
    """Exact mutable source roster needed to prepare/train/screen a model."""

    manifest_members: list[str] = []
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        relative = line.strip()
        if not relative or relative.startswith("#"):
            continue
        route = pathlib.PurePosixPath(relative)
        if route.is_absolute() or ".." in route.parts or route.as_posix() != relative:
            raise TeacherTrainingError("submission source manifest route is unsafe")
        manifest_members.append(relative)
    if not manifest_members or len(set(manifest_members)) != len(manifest_members):
        raise TeacherTrainingError("submission source manifest is empty or duplicated")
    required = set(pipeline.PIPELINE_REQUIRED_BUILD_SOURCES)
    required.update(manifest_members)
    required.update({
        "tools/compact_value_bfm_teacher_training.py",
        "submissions/codingame/bots/compact_value_bfm/export_submission.py",
        "submissions/codingame/bots/compact_value_bfm/rank4_gate_support.py",
        "submissions/codingame/bots/compact_value_bfm/rank4_gate.cpp",
        "submissions/codingame/bots/compact_value_bfm/inference_probe.cpp",
        "submissions/codingame/bots/compact_value_bfm/submission.json",
        "submissions/codingame/bots/compact_value_bfm/sources.txt",
        "submissions/codingame/bots/rank_4/submission.cpp",
    })
    return tuple(sorted(required))


def canonical_json_bytes(value: object) -> bytes:
    return trainer.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record(path: pathlib.Path, *, ascii_required: bool = False) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TeacherTrainingError(f"required regular file is absent: {path}")
    resolved = path.resolve()
    payload = resolved.read_bytes()
    if ascii_required:
        try:
            payload.decode("ascii")
        except UnicodeDecodeError as error:
            raise TeacherTrainingError(f"required ASCII file is not ASCII: {path}") from error
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _validate_record(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise TeacherTrainingError(f"{label} file record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _record(path) != dict(value):
        raise TeacherTrainingError(f"{label} bytes changed")
    return path.resolve()


def _sealed(body: Mapping[str, object]) -> dict[str, object]:
    return challenger.qualification.seal(dict(body))


def _validate_sealed(value: object, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise TeacherTrainingError(f"{label} schema changed")
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        not isinstance(claimed, str)
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise TeacherTrainingError(f"{label} body SHA-256 mismatch")
    return value


def _load_sealed(path: pathlib.Path, schema: str, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeacherTrainingError(f"{label} is unreadable") from error
    if payload != canonical_json_bytes(value):
        raise TeacherTrainingError(f"{label} is not canonical JSON")
    return _validate_sealed(value, schema, label)


def _write_once(path: pathlib.Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise TeacherTrainingError(f"immutable output changed: {path}")
        return
    challenger.qualification.atomic_write_once(path, payload)
    os.chmod(path, 0o555 if executable else 0o444)


def _write_sealed(path: pathlib.Path, body: Mapping[str, object]) -> dict[str, object]:
    document = _sealed(body)
    _write_once(path, canonical_json_bytes(document))
    return document


def _write_content_addressed(
    directory: pathlib.Path, payload: bytes, suffix: str, *, executable: bool = False,
) -> pathlib.Path:
    path = directory / f"{sha256_bytes(payload)}{suffix}"
    _write_once(path, payload, executable=executable)
    return path


def _safe_child(root: pathlib.Path, relative: object, label: str) -> pathlib.Path:
    if not isinstance(relative, str) or not relative or pathlib.PurePath(relative).is_absolute():
        raise TeacherTrainingError(f"{label} route is invalid")
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise TeacherTrainingError(f"{label} escapes its output root") from error
    return path


def _relative(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise TeacherTrainingError("artifact escaped its phase output root") from error


def _tool_closure() -> dict[str, object]:
    sources = []
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        relative = line.strip()
        if not relative or relative.startswith("#"):
            continue
        sources.append(_record(REPOSITORY / relative))
    return {
        "adapter": _record(pathlib.Path(__file__)),
        "trainer": _record(pathlib.Path(trainer.__file__)),
        "pipeline": _record(pathlib.Path(pipeline.__file__)),
        "model_exporter": _record(MODEL_EXPORTER_PATH),
        "source_exporter": _record(SOURCE_EXPORTER_PATH),
        "source_config": _record(SOURCE_CONFIG),
        "source_manifest": _record(SOURCE_MANIFEST),
        "source_inputs": sources,
        "gate_source": _record(GATE_SOURCE),
        "gate_support": _record(GATE_SUPPORT_PATH),
        "rank4_source": _record(RANK4_SOURCE),
    }


def _validate_tool_closure(value: object) -> None:
    expected = {
        "adapter", "trainer", "pipeline", "model_exporter", "source_exporter",
        "source_config", "source_manifest", "source_inputs", "gate_source",
        "gate_support", "rank4_source",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TeacherTrainingError("teacher-training tool closure changed")
    for label in expected - {"source_inputs"}:
        _validate_record(value[label], label)
    inputs = value["source_inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise TeacherTrainingError("submission source closure is empty")
    for index, record in enumerate(inputs):
        _validate_record(record, f"source input {index}")


def _validate_pipeline_build_subset(
    training_closure: Mapping[str, Any], pipeline_closure: object,
) -> None:
    if not isinstance(pipeline_closure, Mapping):
        raise TeacherTrainingError("label pipeline has no frozen build source closure")
    if (
        training_closure.get("manifest") != pipeline_closure.get("manifest")
        or training_closure.get("repository_commit")
        != pipeline_closure.get("repository_commit")
        or training_closure.get("compiler") != pipeline_closure.get("compiler")
        or training_closure.get("producer_binaries")
        != pipeline_closure.get("producer_binaries")
        or training_closure.get("source_validation_mode")
        != pipeline_closure.get("source_validation_mode")
        or training_closure.get("allowed_current_drift_routes")
        != pipeline_closure.get("allowed_current_drift_routes")
    ):
        raise TeacherTrainingError("training and label-pipeline builds differ")
    training_sources = training_closure.get("sources")
    pipeline_sources = pipeline_closure.get("sources")
    if (
        not isinstance(training_sources, Mapping)
        or not isinstance(pipeline_sources, Mapping)
        or set(pipeline_sources) != set(pipeline.PIPELINE_REQUIRED_BUILD_SOURCES)
        or any(
            training_sources.get(relative) != record
            for relative, record in pipeline_sources.items()
        )
    ):
        raise TeacherTrainingError(
            "training does not extend the exact pipeline source closure"
        )


def _revalidate_stored_build_source_closure(
    value: object, *, pipeline_closure: object | None = None,
) -> dict[str, Any]:
    try:
        closure = challenger.verify_phase_build_source_closure(
            required_sources=required_build_sources(),
            stored_closure=value if isinstance(value, Mapping) else None,
        )
    except Exception as error:
        raise TeacherTrainingError(
            "training code differs from its frozen build source closure"
        ) from error
    if pipeline_closure is not None:
        _validate_pipeline_build_subset(closure, pipeline_closure)
    return closure


def _phase_paths(root: pathlib.Path) -> dict[str, str]:
    return {
        "root": str(root),
        "plan": str(root / "training-plan.json"),
        "checkpoint_directory": str(root / "inputs"),
        "runs": str(root / "runs"),
        "sources": str(root / "sources"),
        "source_verification": str(root / "source-verification"),
        "selections": str(root / "selections"),
        "selection_reference": str(root / "selection-reference.json"),
        "gate_sources": str(root / "gates/sources"),
        "gate_binaries": str(root / "gates/binaries"),
        "gate_requests": str(root / "gates/requests"),
        "gate_banks": str(root / "gates/banks"),
        "gate_plans": str(root / "gates/plans"),
        "gate_reference": str(root / "gate-reference.json"),
        "gate_executions": str(root / "gates/executions"),
        "gate_execution_reference": str(root / "gate-execution-reference.json"),
        "full_search_selections": str(root / "full-search/selections"),
        "full_search_selection_reference": str(
            root / "full-search-selection-reference.json"
        ),
        "full_qualification_plans": str(root / "full-qualification/plans"),
        "full_qualification_plan_reference": str(
            root / "full-qualification-plan-reference.json"
        ),
        "full_qualification_banks": str(root / "full-qualification/banks"),
        "full_qualification_executions": str(
            root / "full-qualification/executions"
        ),
        "full_qualification_execution_reference": str(
            root / "full-qualification-execution-reference.json"
        ),
        "admissions": str(root / "admissions"),
        "admission_reference": str(root / "admission-reference.json"),
        "development_fingerprints": str(root / "development-fingerprints"),
        "phase_evidence": str(root / "phase-outcome-evidence.json"),
    }


class _ExternalShardView:
    """Minimal immutable bundle view accepted by the maintained shard loader."""

    def __init__(self, manifest: pathlib.Path, npz: pathlib.Path) -> None:
        if manifest.parent.resolve() != npz.parent.resolve():
            raise TeacherTrainingError("external shard manifest/NPZ are not adjacent")
        self.root = manifest.parent.resolve()
        self.protected_routes: set[str] = set()
        self.records = {
            manifest.name: {
                "relative_path": manifest.name,
                **{key: value for key, value in _record(manifest).items() if key != "path"},
            },
            npz.name: {
                "relative_path": npz.name,
                **{key: value for key, value in _record(npz).items() if key != "path"},
            },
        }

    def is_protected(self, relative: str) -> bool:
        del relative
        return False

    def artifact_path(
        self, relative: object, *, allow_protected: bool = False,
        protected_context: bool = False,
    ) -> pathlib.Path:
        del allow_protected
        if protected_context:
            raise TeacherTrainingError("external shard cannot be protected")
        if not isinstance(relative, str) or pathlib.PurePath(relative).name != relative:
            raise TeacherTrainingError("external shard route is invalid")
        record = self.records.get(relative)
        if record is None:
            raise TeacherTrainingError("external shard route is unregistered")
        path = (self.root / relative).resolve()
        if _record(path)["sha256"] != record["sha256"]:
            raise TeacherTrainingError("external shard bytes changed")
        return path


def _load_shard_reference(
    path: pathlib.Path, *, pipeline_body_sha256: str, split: str,
) -> tuple[trainer.Dataset, dict[str, object]]:
    reference = _load_sealed(
        path, pipeline.SHARD_REFERENCE_SCHEMA, f"external {split} shard reference"
    )
    if (
        reference.get("pipeline_body_sha256") != pipeline_body_sha256
        or reference.get("split") != split
        or reference.get("shard_schema") != trainer.SHARD_SCHEMA
        or reference.get("protected_tests_opened") is not False
    ):
        raise TeacherTrainingError(f"external {split} shard reference binding changed")
    manifest = _validate_record(reference.get("manifest"), f"external {split} manifest")
    npz = _validate_record(reference.get("npz"), f"external {split} NPZ")
    try:
        manifest_document = json.loads(manifest.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeacherTrainingError(f"external {split} manifest is unreadable") from error
    view = _ExternalShardView(manifest, npz)
    dataset = trainer.load_shard(view, manifest.name)
    if dataset.split != split:
        raise TeacherTrainingError(f"external {split} shard has the wrong split")
    return dataset, {
        "reference": _record(path),
        "reference_body_sha256": reference["body_sha256"],
        "manifest": _record(manifest),
        "npz": _record(npz),
        "dataset": trainer.dataset_identity(dataset),
        "provenance": manifest_document.get("provenance"),
    }


def _root_hashes(dataset: trainer.Dataset) -> set[str]:
    return {bytes(value).hex() for value in dataset.group_ids}


def _ranking_root_hashes(
    groups: Sequence[trainer.CompleteTurnGroup], split: str,
) -> set[str]:
    result: set[str] = set()
    for group in groups:
        source = group.evidence.get("source_binding")
        if not isinstance(source, Mapping) or source.get("split") != split:
            raise TeacherTrainingError("successor ranking source split changed")
        root = source.get("root_group_id")
        if not isinstance(root, str) or not root:
            raise TeacherTrainingError("successor ranking root binding is absent")
        result.add(hashlib.sha256(root.encode("utf-8")).hexdigest())
    return result


def _feature_maps() -> tuple[np.ndarray, ...]:
    def feature_map(vertex_map: Sequence[int], edge_map: Sequence[int]) -> np.ndarray:
        result = np.empty(trainer.INPUT_COUNT, dtype=np.uint16)
        result[: trainer.EDGE_COUNT] = np.asarray(edge_map, dtype=np.uint16)
        for vertex in range(trainer.VERTEX_COUNT):
            begin = trainer.EDGE_COUNT + vertex * trainer.VERTEX_CATEGORIES
            destination = (
                trainer.EDGE_COUNT
                + int(vertex_map[vertex]) * trainer.VERTEX_CATEGORIES
            )
            result[begin : begin + trainer.VERTEX_CATEGORIES] = np.arange(
                destination,
                destination + trainer.VERTEX_CATEGORIES,
                dtype=np.uint16,
            )
        return result

    reflected = feature_map(features.REFLECTED_VERTICES, features.REFLECTED_EDGES)
    rotated = feature_map(features.ROTATED_VERTICES, features.ROTATED_EDGES)
    return reflected, rotated, reflected[rotated]


_FEATURE_MAPS = _feature_maps()


def _feature_fingerprint(active: Sequence[int] | np.ndarray) -> str:
    normalized = np.asarray(active, dtype="<u2")
    variants = [normalized.tobytes(order="C")]
    for transform in _FEATURE_MAPS:
        variants.append(
            np.sort(transform[normalized]).astype("<u2", copy=False).tobytes(order="C")
        )
    return hashlib.sha256(min(variants)).hexdigest()


def _dataset_fingerprints(dataset: trainer.Dataset) -> set[str]:
    return {_feature_fingerprint(dataset.active_row(row)) for row in range(len(dataset))}


def _ranking_fingerprints(
    groups: Sequence[trainer.CompleteTurnGroup],
) -> set[str]:
    return {
        _feature_fingerprint(successor.active)
        for group in groups
        for successor in group.successors
    }


def _extended_isolation(
    *, external_train: trainer.Dataset, external_validation: trainer.Dataset,
    anchor: trainer.Dataset, common: trainer.Dataset,
    canonical_validation: trainer.Dataset,
    rankings: trainer.SuccessorRankingLabels,
    expected_root_hashes: Mapping[str, set[str]] | None = None,
) -> dict[str, object]:
    """Reject root and four-way feature leakage including ranked successors."""

    scalar_train_roots = _root_hashes(external_train)
    scalar_validation_roots = _root_hashes(external_validation)
    ranking_train_roots = _ranking_root_hashes(rankings.train, "train")
    ranking_validation_roots = _ranking_root_hashes(rankings.validation, "validation")
    expected_train_roots = (
        scalar_train_roots
        if expected_root_hashes is None
        else expected_root_hashes["train"]
    )
    expected_validation_roots = (
        scalar_validation_roots
        if expected_root_hashes is None
        else expected_root_hashes["validation"]
    )
    if ranking_train_roots != expected_train_roots:
        raise TeacherTrainingError("train successor labels do not bind every scalar root")
    if ranking_validation_roots != expected_validation_roots:
        raise TeacherTrainingError(
            "validation successor labels do not bind every scalar root"
        )
    train_roots = scalar_train_roots | expected_train_roots | _root_hashes(anchor)
    validation_roots = (
        scalar_validation_roots
        | expected_validation_roots
        | _root_hashes(common)
        | _root_hashes(canonical_validation)
    )
    if train_roots & validation_roots:
        raise TeacherTrainingError("external/root anchor train-validation overlap")

    external_validation_fingerprints = _dataset_fingerprints(external_validation)
    ranking_validation_fingerprints = _ranking_fingerprints(rankings.validation)
    novel_validation = (
        external_validation_fingerprints | ranking_validation_fingerprints
    )
    canonical_validation_fingerprints = (
        _dataset_fingerprints(common) | _dataset_fingerprints(canonical_validation)
    )
    all_validation = novel_validation | canonical_validation_fingerprints

    external_train_fingerprints = _dataset_fingerprints(external_train)
    ranking_train_fingerprints = _ranking_fingerprints(rankings.train)
    if (external_train_fingerprints | ranking_train_fingerprints) & all_validation:
        raise TeacherTrainingError(
            "external train/successor exact-rotate-reflect validation overlap"
        )
    anchor_fingerprints = _dataset_fingerprints(anchor)
    if anchor_fingerprints & novel_validation:
        raise TeacherTrainingError(
            "canonical anchor overlaps external/ranking validation"
        )
    return {
        "policy": (
            "whole-root-and-minimum-sha256-over-exact+rotate+reflect+"
            "rotate-reflect-including-complete-turn-successors"
        ),
        "external_train_rows": len(external_train),
        "external_validation_rows": len(external_validation),
        "anchor_rows": len(anchor),
        "common_adjudicator_rows": len(common),
        "canonical_validation_rows": len(canonical_validation),
        "ranking_train_groups": len(rankings.train),
        "ranking_validation_groups": len(rankings.validation),
        "ranking_train_successors": sum(len(group.successors) for group in rankings.train),
        "ranking_validation_successors": sum(
            len(group.successors) for group in rankings.validation
        ),
        "train_root_groups": len(train_roots),
        "validation_root_groups": len(validation_roots),
        "protected_tests_opened": False,
        "passed": True,
    }


def _state_for_group(group: trainer.CompleteTurnGroup) -> features.ReplayState:
    source = group.evidence.get("source_binding")
    if not isinstance(source, Mapping):
        raise TeacherTrainingError("successor group source binding is absent")
    try:
        return corpus._prefix_state(source.get("prefix"))
    except (TypeError, ValueError) as error:
        raise TeacherTrainingError("successor group parent prefix is invalid") from error


def _state_fingerprints_for_rankings(
    rankings: trainer.SuccessorRankingLabels,
) -> set[str]:
    result: set[str] = set()
    for group in (*rankings.train, *rankings.validation):
        parent = _state_for_group(group)
        result.add(challenger.openings.state_fingerprints(parent)["canonical"])
        for successor in group.successors:
            transcript = successor.evidence.get("transcript")
            if not isinstance(transcript, str):
                raise TeacherTrainingError("successor transcript is absent")
            state = features.ReplayState(
                ball=parent.ball,
                to_move=parent.to_move,
                winner=parent.winner,
                used_segments=set(parent.used_segments),
                visit_count=dict(parent.visit_count),
            )
            try:
                physical = corpus._canonical_transcript_for_physical(
                    transcript, parent.to_move
                )
                features.apply_complete_turn(state, parent.to_move, physical)
            except ValueError as error:
                raise TeacherTrainingError("successor transcript no longer replays") from error
            # Frozen opening exclusions contain only playable non-terminal
            # states.  A solved terminal successor cannot collide with one.
            if state.winner is None:
                result.add(challenger.openings.state_fingerprints(state)["canonical"])
    return result


def _exclusion_audit(
    pipeline_plan: Mapping[str, Any], *, external_train: trainer.Dataset,
    external_validation: trainer.Dataset,
    rankings: trainer.SuccessorRankingLabels,
) -> dict[str, object]:
    exclusions = pipeline._exclusion_context(pipeline_plan)
    feature_values = (
        _dataset_fingerprints(external_train)
        | _dataset_fingerprints(external_validation)
        | _ranking_fingerprints(rankings.train)
        | _ranking_fingerprints(rankings.validation)
    )
    state_values = _state_fingerprints_for_rankings(rankings)
    intersections: dict[str, int] = {}
    for role, values in exclusions["by_role"].items():
        domain = exclusions["domains"][role]
        candidate = (
            feature_values
            if domain == pipeline.FEATURE_FINGERPRINT_DOMAIN
            else state_values
        )
        intersections[role] = len(candidate & values)
    if any(intersections.values()):
        raise TeacherTrainingError(
            "external scalar/successor labels intersect a frozen exclusion"
        )
    return {
        "policy": "all-external-parent-and-successor-states-vs-frozen-exclusions",
        "source_counts": exclusions["counts"],
        "candidate_feature_fingerprints": len(feature_values),
        "candidate_state_fingerprints": len(state_values),
        "intersections": intersections,
        "protected_or_live_labels_read": False,
        "passed": True,
    }


def _finalized_pipeline(
    pipeline_plan_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], pathlib.Path]:
    plan = pipeline.load_pipeline(pipeline_plan_path.resolve())
    receipt_path = pathlib.Path(plan["outputs"]["receipts"]) / "07-finalize-labels.json"
    receipt = _load_sealed(
        receipt_path, pipeline.STAGE_RECEIPT_SCHEMA, "finalized label-pipeline receipt"
    )
    if (
        receipt.get("pipeline_body_sha256") != plan["body_sha256"]
        or receipt.get("attempt") != plan["attempt"]
        or receipt.get("phase") != plan["phase"]
        or receipt.get("stage") != "07-finalize-labels"
        or receipt.get("details", {}).get("protected_tests_opened") is not False
    ):
        raise TeacherTrainingError("label pipeline is not finalized for this phase")
    outputs = receipt.get("outputs")
    required = {
        "successor_labels", "train_shard_reference",
        "validation_shard_reference", "scalar_manifest",
    }
    if not isinstance(outputs, Mapping) or not required.issubset(outputs):
        raise TeacherTrainingError("finalized label receipt omits training artifacts")
    for name, record in outputs.items():
        _validate_record(record, f"finalized pipeline {name}")
    return plan, receipt, receipt_path


def _copied_bundle_path(campaign_context: Mapping[str, Any]) -> pathlib.Path:
    record = campaign_context.get("inputs", {}).get("training_bundle", {}).get("manifest")
    if not isinstance(record, Mapping):
        raise TeacherTrainingError("challenger campaign has no copied training bundle")
    try:
        return challenger._bundle_path(campaign_context, record)
    except Exception as error:
        raise TeacherTrainingError("copied training bundle binding changed") from error


def _phase_input_path(
    campaign_context: Mapping[str, Any], phase_context: Mapping[str, Any],
    name: str,
) -> tuple[pathlib.Path, Mapping[str, Any]]:
    record = phase_context.get("phase", {}).get("attempt_inputs", {}).get(name)
    if not isinstance(record, Mapping):
        raise TeacherTrainingError(f"phase plan has no bound {name}")
    try:
        if "route" in record:
            path = challenger._bundle_path(campaign_context, record)
        else:
            path = challenger._verify_record(record, f"phase {name}")
    except Exception as error:
        raise TeacherTrainingError(f"phase-bound {name} changed") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise TeacherTrainingError(f"phase-bound {name} bytes changed")
    return path.resolve(), record


def _load_core_inputs(
    bundle: trainer.FrozenBundle,
) -> tuple[trainer.Dataset, trainer.Dataset, trainer.Dataset, dict[str, tuple[str, ...]]]:
    anchor_routes = bundle.canonical_routes("train")
    canonical_validation_routes = bundle.canonical_routes("validation")
    common_route = bundle.common_adjudicator_route()
    anchor = trainer.concatenate_datasets(
        [trainer.load_shard(bundle, route) for route in anchor_routes], split="train"
    )
    common = trainer.load_shard(bundle, common_route)
    canonical_validation = trainer.concatenate_datasets(
        [trainer.load_shard(bundle, route) for route in canonical_validation_routes],
        split="validation",
    )
    if common.split != "validation":
        raise TeacherTrainingError("copied common adjudicator split changed")
    return anchor, common, canonical_validation, {
        "anchor": tuple(anchor_routes),
        "common_adjudicator": (common_route,),
        "canonical_validation": tuple(canonical_validation_routes),
    }


def _build_training_inputs(
    *, pipeline_plan: Mapping[str, Any], final_receipt: Mapping[str, Any],
    bundle: trainer.FrozenBundle,
) -> tuple[trainer.TrainingInputs, trainer.Dataset, dict[str, object]]:
    outputs = final_receipt["outputs"]
    external_train, train_binding = _load_shard_reference(
        _validate_record(outputs["train_shard_reference"], "train shard reference"),
        pipeline_body_sha256=str(pipeline_plan["body_sha256"]),
        split="train",
    )
    external_validation, validation_binding = _load_shard_reference(
        _validate_record(
            outputs["validation_shard_reference"], "validation shard reference"
        ),
        pipeline_body_sha256=str(pipeline_plan["body_sha256"]),
        split="validation",
    )
    labels_path = _validate_record(outputs["successor_labels"], "successor labels")
    rankings = trainer.load_successor_ranking_labels(labels_path, bundle)
    roots_path = pathlib.Path(pipeline_plan["outputs"]["root_assignments"])
    roots = _load_sealed(roots_path, corpus.ROOT_SCHEMA, "phase root assignments")
    accepted_roots = roots.get("accepted")
    if not isinstance(accepted_roots, list) or not accepted_roots:
        raise TeacherTrainingError("phase root assignments are empty")
    expected_root_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in accepted_roots:
        if (
            not isinstance(row, Mapping)
            or row.get("split") not in expected_root_hashes
            or not isinstance(row.get("group_id"), str)
            or not row["group_id"]
        ):
            raise TeacherTrainingError("phase root assignment is malformed")
        expected_root_hashes[str(row["split"])].add(
            hashlib.sha256(str(row["group_id"]).encode()).hexdigest()
        )
    if not all(expected_root_hashes.values()):
        raise TeacherTrainingError("phase root assignments require both splits")
    packing_roots = _validate_record(
        pipeline_plan["inputs"]["filtered_roots_manifest"],
        "filtered packing root assignments",
    )
    roots_sha = sha256_file(packing_roots)
    for binding in (train_binding, validation_binding):
        provenance = binding.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("roots_manifest_sha256") != roots_sha
        ):
            raise TeacherTrainingError("external CSR is not bound to phase root assignments")
    anchor, common, canonical_validation, source_routes = _load_core_inputs(bundle)
    isolation = _extended_isolation(
        external_train=external_train,
        external_validation=external_validation,
        anchor=anchor,
        common=common,
        canonical_validation=canonical_validation,
        rankings=rankings,
        expected_root_hashes=expected_root_hashes,
    )
    exclusion = _exclusion_audit(
        pipeline_plan,
        external_train=external_train,
        external_validation=external_validation,
        rankings=rankings,
    )
    audit: dict[str, object] = {
        "schema": "papersoccer.compact-value-bfm.rank4-teacher-input-audit.v1",
        "source_bundle_body_sha256": bundle.body_sha256,
        "pipeline_body_sha256": pipeline_plan["body_sha256"],
        "external": {
            "train": train_binding,
            "validation": validation_binding,
            "successor_labels": {
                "file": _record(labels_path),
                "body_sha256": rankings.body_sha256,
                "train_groups": len(rankings.train),
                "validation_groups": len(rankings.validation),
            },
            "root_assignments": {
                "file": _record(roots_path),
                "body_sha256": roots["body_sha256"],
                "packing_roots": _record(packing_roots),
                "train_roots": len(expected_root_hashes["train"]),
                "validation_roots": len(expected_root_hashes["validation"]),
            },
        },
        "core": {
            "anchor": trainer.dataset_identity(anchor),
            "common_adjudicator": trainer.dataset_identity(common),
            "canonical_validation": trainer.dataset_identity(canonical_validation),
        },
        "root_and_symmetry_isolation": isolation,
        "frozen_exclusion_audit": exclusion,
        "protected_tests_opened": False,
        "passed": True,
    }
    inputs = trainer.TrainingInputs(
        new=external_train,
        anchor=anchor,
        common_adjudicator=common,
        canonical_validation=canonical_validation,
        source_routes={
            "new": (str(train_binding["manifest"]["path"]),),
            "external_validation": (str(validation_binding["manifest"]["path"]),),
            **source_routes,
        },
        paired_row_validation={
            "policy": "external-phase-train-and-validation-bound-before-training",
            "train_manifest_sha256": train_binding["manifest"]["sha256"],
            "validation_manifest_sha256": validation_binding["manifest"]["sha256"],
            "passed": True,
        },
        split_isolation=isolation,
        input_audit=audit,
        successor_rankings=rankings,
    )
    return inputs, external_validation, audit


def _validate_pilot_admission_for_full(
    path: pathlib.Path, *, campaign_id: str, attempt: int,
) -> dict[str, Any]:
    value = load_phase_admission(path)
    selected = value.get("selected_candidate")
    search_throughput_profile = _search_throughput_profile(
        value.get("adaptation_contract")
    )
    variants = active_search_variants(search_throughput_profile)
    if (
        value.get("campaign_id") != campaign_id
        or value.get("attempt") != attempt
        or value.get("phase") != "pilot"
        or value.get("admitted") is not True
        or not isinstance(selected, Mapping)
        or value.get("search_throughput_profile")
        != search_throughput_profile
        or selected.get("search_throughput_profile")
        != search_throughput_profile
        or selected.get("candidate_search_profile")
        != _search_variant_metadata(
            search_throughput_profile, selected["search_variant"]
        )["candidate_search_profile"]
        or selected.get("search_variant") not in variants
        or selected.get("compile_time_macros")
        != list(variants[selected["search_variant"]])
        or float(selected.get("ranking_weight", -1.0)) not in (0.10, 0.25)
        or selected.get("offline_eligible") is not True
        or selected.get("diagnostic_only") is not False
        or selected.get("source_is_default_for_variant") is not True
    ):
        raise TeacherTrainingError("full phase requires an admitted pilot candidate")
    return value


def _phase_adaptation_contract(
    phase_context: Mapping[str, Any],
) -> dict[str, str]:
    """Copy the exact immutable phase adaptation contract into evidence."""

    phase = phase_context.get("phase")
    adaptation = phase.get("adaptation_contract") if isinstance(
        phase, Mapping
    ) else None
    try:
        return challenger._validated_adaptation_contract(adaptation)
    except Exception as error:
        raise TeacherTrainingError(
            "phase adaptation contract is absent or invalid"
        ) from error


def _phase_qat_profile(
    phase_context: Mapping[str, Any],
) -> tuple[str, dict[str, object]]:
    """Resolve QAT exclusively from the immutable phase adaptation contract."""

    adaptation = _phase_adaptation_contract(phase_context)
    name = adaptation["qat_profile"]
    try:
        profile = trainer.resolve_qat_profile(name)
        contract = trainer.qat_profile_contract(profile)
    except trainer.TrainingError as error:
        raise TeacherTrainingError(
            "phase adaptation contract has no registered QAT profile"
        ) from error
    return profile.name, contract


def _validated_hard_state_density(
    value: object, *, teacher_ranking_profile: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "policy", "hard_teacher_ranking_profile", "hard_group_multiplier",
        "unique_comparable_groups", "hard_unique_groups",
        "scheduled_group_entries", "hard_scheduled_entries",
        "density_increased",
    }:
        raise TeacherTrainingError("hard-state density evidence is malformed")
    counts = (
        value.get("unique_comparable_groups"),
        value.get("hard_unique_groups"),
        value.get("scheduled_group_entries"),
        value.get("hard_scheduled_entries"),
    )
    if (
        value.get("policy") != "deterministic-expanded-ranking-schedule-v1"
        or value.get("hard_teacher_ranking_profile")
        != pipeline.HARD_5PCT_2M_TEACHER_RANKING_PROFILE
        or value.get("hard_group_multiplier")
        != trainer.HARD_STATE_DENSITY_MULTIPLIER
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        )
    ):
        raise TeacherTrainingError("hard-state density registry changed")
    unique, hard, scheduled, hard_scheduled = map(int, counts)
    if hard > unique or hard_scheduled != hard * trainer.HARD_STATE_DENSITY_MULTIPLIER:
        raise TeacherTrainingError("hard-state density counts changed")
    if teacher_ranking_profile == pipeline.STANDARD_TEACHER_RANKING_PROFILE:
        passed = (
            hard == 0
            and hard_scheduled == 0
            and scheduled == unique
            and value.get("density_increased") is False
        )
    elif teacher_ranking_profile == pipeline.HARD_5PCT_2M_TEACHER_RANKING_PROFILE:
        passed = (
            hard > 0
            and scheduled == unique + hard * (
                trainer.HARD_STATE_DENSITY_MULTIPLIER - 1
            )
            and value.get("density_increased") is True
        )
    else:
        raise TeacherTrainingError("teacher-ranking density profile is unsupported")
    if not passed:
        raise TeacherTrainingError(
            "hard-state density does not match the phase adaptation"
        )
    return dict(value)


def _record_subset(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or not {"path", "bytes", "sha256"}.issubset(value):
        raise TeacherTrainingError(f"{label} record is malformed")
    return _validate_record(
        {key: value[key] for key in ("path", "bytes", "sha256")}, label
    )


def load_phase_admission(path: pathlib.Path) -> dict[str, Any]:
    value = _load_sealed(path.resolve(), ADMISSION_SCHEMA, "phase admission")
    expected = {
        "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
        "gate_plan", "gate_plan_body_sha256", "training_selection",
        "gate_execution", "gate_execution_body_sha256",
        "injected_test_results",
        "full_search_selection", "full_qualification_plan",
        "full_qualification_execution", "qualification_result",
        "finalized_pipeline_receipt", "results", "summaries",
        "selected_candidate", "metrics", "strength_delta_pp",
        "teacher_regret_reduction_fraction", "development_exclusion",
        "phase_outcome_evidence", "admitted", "next_route",
        "protected_or_live_metrics_read", "qat_profile",
        "qat_profile_contract", "adaptation_contract",
        "search_throughput_profile", "active_search_variant_roster",
        "candidate_search_profile",
        "body_sha256",
    }
    selected = value.get("selected_candidate")
    try:
        adaptation_contract = challenger._validated_adaptation_contract(
            value.get("adaptation_contract")
        )
        search_throughput_profile = _search_throughput_profile(
            adaptation_contract
        )
        active_variants = active_search_variants(search_throughput_profile)
    except Exception as error:
        raise TeacherTrainingError(
            "phase admission adaptation contract changed"
        ) from error
    selected_variant = (
        selected.get("search_variant") if isinstance(selected, Mapping) else None
    )
    selected_metadata = (
        _search_variant_metadata(search_throughput_profile, selected_variant)
        if selected_variant in active_variants
        else None
    )
    active_roster = value.get("active_search_variant_roster")
    expected_active_roster = list(phase_gate_variants(
        str(value.get("phase")), search_throughput_profile
    ))
    if (
        set(value) != expected
        or value.get("phase") not in {"pilot", "full"}
        or isinstance(value.get("attempt"), bool)
        or not isinstance(value.get("attempt"), int)
        or value["attempt"] <= 0
        or not isinstance(value.get("campaign_id"), str)
        or SHA256_RE.fullmatch(str(value.get("plan_body_sha256"))) is None
        or not isinstance(value.get("results"), Mapping)
        or not isinstance(value.get("summaries"), Mapping)
        or set(value["results"]) != set(value["summaries"])
        or not isinstance(selected, Mapping)
        or value.get("search_throughput_profile")
        != search_throughput_profile
        or not isinstance(active_roster, list)
        or active_roster != expected_active_roster
        or selected_variant not in active_roster
        or selected_metadata is None
        or selected.get("compile_time_macros")
        != selected_metadata["compile_time_macros"]
        or selected.get("search_throughput_profile")
        != search_throughput_profile
        or selected.get("candidate_search_profile")
        != selected_metadata["candidate_search_profile"]
        or selected.get("standard_base_variant")
        != selected_metadata["standard_base_variant"]
        or selected.get("search_treatment")
        is not selected_metadata["is_treatment"]
        or selected.get("source_is_default_for_variant") is not True
        or not isinstance(selected.get("offline_eligible"), bool)
        or not isinstance(selected.get("diagnostic_only"), bool)
        or selected.get("diagnostic_only") is selected.get("offline_eligible")
        or selected.get("qat_profile") != value.get("qat_profile")
        or selected.get("qat_profile_contract")
        != value.get("qat_profile_contract")
        or selected.get("adaptation_contract")
        != value.get("adaptation_contract")
        or value.get("protected_or_live_metrics_read") is not False
        or not isinstance(value.get("admitted"), bool)
    ):
        raise TeacherTrainingError("phase admission contract changed")
    if value["phase"] == "pilot":
        metrics_value = value.get("metrics")
        search_selection = (
            metrics_value.get("search_ab")
            if isinstance(metrics_value, Mapping)
            else None
        )
        if not isinstance(search_selection, Mapping):
            raise TeacherTrainingError("phase admission search selection is absent")
        if search_throughput_profile == "standard-v1":
            if list(value["results"]) != ["baseline"]:
                raise TeacherTrainingError("standard-v1 admission roster changed")
        elif selected_metadata["is_treatment"]:
            treatment_comparisons = search_selection.get(
                "treatment_comparisons"
            )
            treatment = (
                treatment_comparisons.get(selected_variant)
                if isinstance(treatment_comparisons, Mapping)
                else None
            )
            checks = treatment.get("checks") if isinstance(treatment, Mapping) else None
            if (
                not isinstance(checks, Mapping)
                or not checks
                or not all(check is True for check in checks.values())
                or treatment.get("retained") is not True
                or selected_variant
                not in search_selection.get("retained_variants", [])
            ):
                raise TeacherTrainingError(
                    "phase admission retained an unclean search treatment"
                )
    gate_plan_path = _record_subset(value["gate_plan"], "admission gate plan")
    gate_plan = _load_sealed(gate_plan_path, GATE_PLAN_SCHEMA, "admission gate plan")
    if (
        gate_plan["body_sha256"] != value.get("gate_plan_body_sha256")
        or gate_plan.get("search_throughput_profile")
        != search_throughput_profile
        or gate_plan.get("active_search_variant_roster") != active_roster
        or set(value["results"]) != set(active_roster)
    ):
        raise TeacherTrainingError("phase admission gate plan body changed")
    _record_subset(value["training_selection"], "admission training selection")
    _record_subset(
        value["finalized_pipeline_receipt"], "admission finalized pipeline receipt"
    )
    _record_subset(value["development_exclusion"], "admission development exclusion")
    evidence_path = _record_subset(
        value["phase_outcome_evidence"], "admission phase evidence"
    )
    evidence = _load_sealed(
        evidence_path, challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA, "phase evidence"
    )
    evidence_closure = evidence.get("evidence_closure")
    if not isinstance(evidence_closure, Mapping) or set(evidence_closure) != {
        "training_plan", "pipeline_plan", "finalized_pipeline_receipt",
        "training_selection", "gate_plan", "gate_results",
        "gate_execution", "full_search_selection", "full_qualification_plan",
        "full_qualification_execution", "qualification_result",
        "selected_candidate", "input_audit_sha256",
        "build_source_closure_sha256", "protected_tests_opened",
    }:
        raise TeacherTrainingError("phase admission evidence closure is absent")
    training_plan_path = _record_subset(
        evidence_closure.get("training_plan"), "admission training plan"
    )
    validated_training_plan = _load_sealed(
        training_plan_path, PLAN_SCHEMA, "admission training plan"
    )
    _validate_admission_gate_execution(
        value,
        training_plan=validated_training_plan,
        gate_plan_path=gate_plan_path,
    )
    qualification_evidence = _validate_admission_full_qualification(
        value, training_plan=validated_training_plan
    )
    qat_profile = value.get("qat_profile")
    try:
        qat_profile_contract = trainer.validate_qat_profile_contract(
            value.get("qat_profile_contract"),
            expected_name=(qat_profile if isinstance(qat_profile, str) else ""),
        )
    except trainer.TrainingError as error:
        raise TeacherTrainingError("phase admission QAT profile changed") from error
    selection_path = _record_subset(
        evidence_closure.get("training_selection"),
        "admission training selection",
    )
    validated_selection = load_training_selection(
        validated_training_plan, selection_path
    )
    validated_search_evidence = _validate_admission_search_evidence(
        value,
        training_plan=validated_training_plan,
        gate_plan_path=gate_plan_path,
    )
    validated_gate_plan = validated_search_evidence["gate_plan"]
    if validated_gate_plan != gate_plan:
        raise TeacherTrainingError("phase admission gate evidence changed")
    plan_training = validated_training_plan.get("training")
    selection_model = validated_selection.get("selected_model")
    pilot_binding = validated_training_plan.get("pilot_admission")
    full_search_prior_clean = bool(
        value["phase"] != "full"
        or (
            isinstance(pilot_binding, Mapping)
            and pilot_binding.get("search_variant") in active_variants
            and pilot_binding.get("compile_time_macros") == list(
                active_variants[pilot_binding["search_variant"]]
            )
            and pilot_binding.get("search_throughput_profile")
            == search_throughput_profile
            and pilot_binding.get("candidate_search_profile")
            == _search_variant_metadata(
                search_throughput_profile,
                pilot_binding["search_variant"],
            )["candidate_search_profile"]
        )
    )
    if isinstance(selection_model, Mapping):
        try:
            trainer.validate_qat_execution_evidence(
                selection_model.get("qat_execution_evidence"),
                expected_profile=(
                    qat_profile if isinstance(qat_profile, str) else ""
                ),
            )
        except trainer.TrainingError as error:
            raise TeacherTrainingError(
                "phase admission QAT execution evidence changed"
            ) from error
    selection_arms = validated_selection.get("arms")
    selected_model_arms = [
        arm for arm in selection_arms
        if isinstance(arm, Mapping)
        and arm.get("ranking_weight") == selection_model.get("ranking_weight")
    ] if isinstance(selection_arms, list) and isinstance(
        selection_model, Mapping
    ) else []
    scalar_control_arms = [
        arm for arm in selection_arms
        if isinstance(arm, Mapping) and arm.get("ranking_weight") == 0.0
    ] if isinstance(selection_arms, list) else []
    metrics_value = value.get("metrics")
    selected_gate_summary = (
        qualification_evidence.get("summary")
        if value["phase"] == "full" and isinstance(qualification_evidence, Mapping)
        else validated_search_evidence.get("summaries", {}).get(selected_variant)
    )
    if (
        len(selected_model_arms) != 1
        or len(scalar_control_arms) != 1
        or not isinstance(metrics_value, Mapping)
        or not isinstance(selected_gate_summary, Mapping)
        or not isinstance(selected_gate_summary.get("candidate_wins"), int)
        or isinstance(selected_gate_summary.get("candidate_wins"), bool)
    ):
        raise TeacherTrainingError("phase outcome metrics cannot be reconstructed")
    selected_model_arm = selected_model_arms[0]
    scalar_control_arm = scalar_control_arms[0]
    denominator = 2.0 * (
        PILOT_PAIRS if value["phase"] == "pilot" else FULL_PAIRS
    )
    rank4_win_rate = selected_gate_summary["candidate_wins"] / denominator
    rank4_absolute_margin_pp = 100.0 * (rank4_win_rate - 0.5)
    if (
        metrics_value.get("canonical_retention_passed")
        is not selected_model_arm.get("offline_gate_passed")
        or metrics_value.get("offline_model_eligible")
        is not selection_model.get("offline_eligible")
        or metrics_value.get("diagnostic_only")
        is not selection_model.get("diagnostic_only")
        or metrics_value.get("quantized_action_flip_rate")
        != selected_model_arm.get("metrics", {}).get(
            "float_vs_quantized_action_flip_rate"
        )
        or metrics_value.get("scalar_control_action_flip_rate")
        != scalar_control_arm.get("metrics", {}).get(
            "float_vs_quantized_action_flip_rate"
        )
        or metrics_value.get("rank4_win_rate") != rank4_win_rate
        or metrics_value.get("rank4_absolute_margin_pp")
        != rank4_absolute_margin_pp
        or metrics_value.get("strength_delta_pp") != rank4_absolute_margin_pp
        or value.get("strength_delta_pp") != rank4_absolute_margin_pp
        or (
            value["phase"] == "pilot"
            and metrics_value.get("development_gate_game_volume")
            != validated_gate_plan.get("expected_game_volume")
        )
        or (
            value["phase"] == "full"
            and (
                metrics_value.get("search_ab_game_volume")
                != validated_gate_plan.get("expected_game_volume")
                or metrics_value.get("qualification_game_volume")
                != qualification_evidence.get("qualification_plan", {}).get(
                    "expected_game_volume"
                )
            )
        )
    ):
        raise TeacherTrainingError("phase outcome metrics changed")
    recomputed_admitted = _recomputed_phase_admission(
        phase=str(value["phase"]),
        selection=validated_selection,
        search_evidence=validated_search_evidence,
        qualification_evidence=qualification_evidence,
    )
    expected_next_route = (
        "materialize-full"
        if recomputed_admitted and value["phase"] == "pilot"
        else "prepare-dual-final"
        if recomputed_admitted
        else "open-next-leakage-isolated-attempt"
    )
    if (
        value.get("admitted") is not recomputed_admitted
        or value.get("next_route") != expected_next_route
    ):
        raise TeacherTrainingError(
            "phase admission decision differs from sealed evidence"
        )
    try:
        challenger.validate_build_source_closure_evidence(
            validated_training_plan.get("build_source_closure"),
            required_sources=required_build_sources(),
        )
    except Exception as error:
        raise TeacherTrainingError(
            "phase admission build source closure changed"
        ) from error
    candidate = evidence.get("candidate")
    if (
        evidence.get("campaign_id") != value["campaign_id"]
        or evidence.get("attempt") != value["attempt"]
        or evidence.get("phase") != value["phase"]
        or evidence.get("metrics_sha256")
        != sha256_bytes(canonical_json_bytes(value["metrics"]))
        or evidence.get("qat_profile") != qat_profile
        or evidence.get("qat_profile_contract") != qat_profile_contract
        or evidence.get("adaptation_contract")
        != adaptation_contract
        or evidence.get("search_throughput_profile")
        != search_throughput_profile
        or evidence.get("candidate_search_profile")
        != selected_metadata["candidate_search_profile"]
        or evidence.get("gate_execution") != value.get("gate_execution")
        or evidence.get("full_search_selection")
        != value.get("full_search_selection")
        or evidence.get("full_qualification_plan")
        != value.get("full_qualification_plan")
        or evidence.get("full_qualification_execution")
        != value.get("full_qualification_execution")
        or evidence.get("qualification_result")
        != value.get("qualification_result")
        or evidence.get("injected_test_results")
        is not value.get("injected_test_results")
        or evidence.get("active_search_variant_roster") != active_roster
        or value.get("candidate_search_profile")
        != selected_metadata["candidate_search_profile"]
        or not full_search_prior_clean
        or not isinstance(plan_training, Mapping)
        or validated_training_plan.get("adaptation_contract")
        != adaptation_contract
        or validated_training_plan.get("search_throughput_profile")
        != search_throughput_profile
        or validated_training_plan.get("search_variants")
        != {
            name: list(macros)
            for name, macros in active_variants.items()
        }
        or plan_training.get("qat_profile") != qat_profile
        or plan_training.get("qat_profile_contract") != qat_profile_contract
        or not isinstance(selection_model, Mapping)
        or selection_model.get("qat_profile") != qat_profile
        or selection_model.get("qat_profile_contract") != qat_profile_contract
        or selection_model.get("adaptation_contract")
        != adaptation_contract
        or validated_selection.get("adaptation_contract")
        != adaptation_contract
        or selection_model.get("search_throughput_profile")
        != search_throughput_profile
        or selection_model.get("active_search_variant_roster")
        != list(active_variants)
        or selected.get("offline_eligible")
        is not selection_model.get("offline_eligible")
        or selected.get("diagnostic_only")
        is not selection_model.get("diagnostic_only")
        or validated_selection.get("search_throughput_profile")
        != search_throughput_profile
        or validated_selection.get("active_search_variant_roster")
        != list(active_variants)
        or selected.get("qat_execution_evidence_sha256")
        != sha256_bytes(canonical_json_bytes(
            selection_model.get("qat_execution_evidence")
        ))
        or selected.get("hard_state_density")
        != selection_model.get("hard_state_density")
        or candidate
        != {
            "runtime_sha256": selected.get("runtime", {}).get("sha256"),
            "source_sha256": selected.get("source", {}).get("sha256"),
        }
        or evidence_closure.get("selected_candidate") != selected
        or evidence_closure.get("gate_results") != value["results"]
        or evidence_closure.get("gate_execution")
        != value.get("gate_execution")
        or evidence_closure.get("full_search_selection")
        != value.get("full_search_selection")
        or evidence_closure.get("full_qualification_plan")
        != value.get("full_qualification_plan")
        or evidence_closure.get("full_qualification_execution")
        != value.get("full_qualification_execution")
        or evidence_closure.get("qualification_result")
        != value.get("qualification_result")
        or evidence_closure.get("build_source_closure_sha256")
        != validated_training_plan.get("build_source_closure", {}).get(
            "closure_sha256"
        )
        or evidence_closure.get("training_plan", {}).get("body_sha256")
        != validated_training_plan.get("body_sha256")
        or evidence.get("protected_or_live_metrics_read") is not False
    ):
        raise TeacherTrainingError("phase admission evidence closure changed")
    for variant, record in value["results"].items():
        if variant not in active_variants:
            raise TeacherTrainingError("phase admission has an unknown result variant")
        _record_subset(record, f"{variant} admission result")
    _record_subset(selected.get("runtime"), "admission candidate runtime")
    source = _record_subset(selected.get("source"), "admission candidate source")
    _record_subset(selected.get("binary"), "admission candidate binary")
    if not 0 < source.stat().st_size < SOURCE_LIMIT_EXCLUSIVE:
        raise TeacherTrainingError("admission candidate source violates 95KB")
    return value


def prepare_training(
    *, campaign_plan: pathlib.Path, phase_reference: pathlib.Path,
    pipeline_plan: pathlib.Path,
    output_root: pathlib.Path, created_at_utc: str,
    pilot_admission: pathlib.Path | None = None,
) -> pathlib.Path:
    campaign_context = challenger.validate_campaign(campaign_plan.resolve())
    phase_context = challenger.validate_phase_reference(
        phase_reference.resolve(), campaign_context["plan"]
    )
    phase = str(phase_context["phase"]["phase"])
    attempt = int(phase_context["phase"]["attempt"])
    production = (
        campaign_context["inputs"].get("production_allowlist_enforced") is True
    )
    expected_output_base = (
        _production_execution_base(campaign_context) if production else None
    )
    if production and output_root.resolve() != expected_output_base:
        raise TeacherTrainingError(
            "production teacher-training output root is not campaign-derived"
        )
    adaptation_contract = _phase_adaptation_contract(phase_context)
    qat_profile, qat_profile_contract = _phase_qat_profile(phase_context)
    search_throughput_profile = _search_throughput_profile(adaptation_contract)
    search_variants = active_search_variants(search_throughput_profile)
    pipeline_context, final_receipt, final_receipt_path = _finalized_pipeline(
        pipeline_plan
    )
    if (
        pipeline_context.get("campaign_id") != campaign_context["plan"]["campaign_id"]
        or pipeline_context.get("attempt") != attempt
        or pipeline_context.get("phase") != phase
        or pipeline_context.get("campaign_plan") != _record(campaign_plan.resolve())
        or pipeline_context.get("phase_reference") != _record(phase_reference.resolve())
        or pipeline_context.get("adaptation_contract") != adaptation_contract
    ):
        raise TeacherTrainingError("campaign, phase, and label pipeline disagree")
    phase_bound_paths = {
        name: _phase_input_path(campaign_context, phase_context, name)[0]
        for name in (
            "student_runtime", "prior_runtime", "roots_tsv", "roots_manifest"
        )
    }
    pipeline_inputs = pipeline_context.get("inputs", {})
    expected_pipeline_records = {
        "student_runtime": _record(phase_bound_paths["student_runtime"]),
        "prior_runtime": _record(phase_bound_paths["prior_runtime"]),
        "source_roots_tsv": _record(phase_bound_paths["roots_tsv"]),
        "source_roots_manifest": _record(phase_bound_paths["roots_manifest"]),
    }
    if any(
        pipeline_inputs.get(name) != record
        for name, record in expected_pipeline_records.items()
    ):
        raise TeacherTrainingError(
            "label pipeline does not use the phase-bound student/prior/roots"
        )
    try:
        build_source_closure = challenger.verify_phase_build_source_closure(
            required_sources=required_build_sources(),
            campaign_context=campaign_context,
            phase_context=phase_context,
        )
    except Exception as error:
        raise TeacherTrainingError(
            "training code differs from its frozen build source closure"
        ) from error
    _validate_pipeline_build_subset(
        build_source_closure, pipeline_context.get("build_source_closure")
    )

    bundle_path = _copied_bundle_path(campaign_context)
    bundle = trainer.FrozenBundle.load(bundle_path)
    if pipeline_context.get("source_bundle_body_sha256") != bundle.body_sha256:
        raise TeacherTrainingError("pipeline and copied FrozenBundle identities differ")

    checkpoint, checkpoint_phase_record = _phase_input_path(
        campaign_context, phase_context, "initial_float_checkpoint"
    )
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.name != f"{sha256_file(checkpoint)}.float.npz"
    ):
        raise TeacherTrainingError(
            "phase-bound 12x8 checkpoint must be an exact content-addressed regular file"
        )
    if attempt == 1 and sha256_file(checkpoint) != ATTEMPT_ONE_INITIAL_CHECKPOINT_SHA256:
        raise TeacherTrainingError(
            "attempt one must initialize from the authoritative successor checkpoint"
        )
    trainer.load_float_checkpoint(checkpoint, ARCHITECTURE)

    selected_pilot = None
    if phase == "pilot":
        if pilot_admission is not None:
            raise TeacherTrainingError("pilot preparation cannot consume pilot admission")
        weights = PILOT_WEIGHTS
    elif phase == "full":
        if pilot_admission is None:
            raise TeacherTrainingError("full preparation requires the admitted pilot")
        selected_pilot = _validate_pilot_admission_for_full(
            pilot_admission,
            campaign_id=str(campaign_context["plan"]["campaign_id"]),
            attempt=attempt,
        )
        if selected_pilot.get("qat_profile") != qat_profile:
            raise TeacherTrainingError(
                "full phase QAT profile differs from its admitted pilot"
            )
        if selected_pilot.get("adaptation_contract") != adaptation_contract:
            raise TeacherTrainingError(
                "full phase adaptation differs from its admitted pilot"
            )
        if (
            selected_pilot.get("search_throughput_profile")
            != search_throughput_profile
        ):
            raise TeacherTrainingError(
                "full phase search-throughput profile differs from its admitted pilot"
            )
        ranking_weight = float(selected_pilot["selected_candidate"]["ranking_weight"])
        weights = (0.0, ranking_weight)
    else:
        raise TeacherTrainingError("teacher training accepts only pilot or full phases")

    inputs, _external_validation, input_audit = _build_training_inputs(
        pipeline_plan=pipeline_context,
        final_receipt=final_receipt,
        bundle=bundle,
    )
    del inputs
    root = (
        output_root.resolve()
        / f"attempt-{attempt:03d}"
        / phase
        / "teacher-training"
    )
    paths = _phase_paths(root)
    checkpoint_copy = pathlib.Path(paths["checkpoint_directory"]) / checkpoint.name
    _write_once(checkpoint_copy, checkpoint.read_bytes())
    trainer.load_float_checkpoint(checkpoint_copy, ARCHITECTURE)

    body: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "campaign_id": campaign_context["plan"]["campaign_id"],
        "attempt": attempt,
        "phase": phase,
        "adaptation_contract": adaptation_contract,
        "search_throughput_profile": search_throughput_profile,
        "created_at_utc": challenger.utc(created_at_utc, "training-plan timestamp"),
        "campaign_plan": _record(campaign_plan.resolve()),
        "phase_reference": _record(phase_reference.resolve()),
        "pipeline_plan": _record(pipeline_plan.resolve()),
        "pipeline_body_sha256": pipeline_context["body_sha256"],
        "final_pipeline_receipt": _record(final_receipt_path),
        "final_pipeline_receipt_body_sha256": final_receipt["body_sha256"],
        "source_bundle": {
            "manifest": _record(bundle_path),
            "body_sha256": bundle.body_sha256,
        },
        "initial_checkpoint": _record(checkpoint_copy),
        "initial_checkpoint_phase_record": dict(checkpoint_phase_record),
        "architecture": {
            "name": ARCHITECTURE.name,
            "dimensions": list(ARCHITECTURE.dimensions),
            "biases": False,
            "activations": list(trainer.ACTIVATIONS),
            "quantization_bits": trainer.QUANTIZATION_BITS,
            "policy_head": False,
        },
        "training": {
            "ranking_weights": list(weights),
            "fixed_seeds": list(trainer.FIXED_SEEDS),
            "seed_workers": SEED_WORKERS,
            "new_rows_per_batch": trainer.NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": trainer.ANCHOR_ROWS_PER_BATCH,
            "float_warmup_epochs": trainer.RANKING_FLOAT_EPOCHS,
            "float_learning_rate": trainer.RANKING_FLOAT_LEARNING_RATE,
            "qat_epochs": trainer.QAT_EPOCHS,
            "qat_profile": qat_profile,
            "qat_profile_contract": qat_profile_contract,
            "native_thread_environment": dict(
                trainer.NATIVE_THREAD_ENVIRONMENT
            ),
            "ranking_epoch_schedule": (
                "balanced-full-weighted-pool-permutation-per-epoch-v1"
            ),
            "ranking_group_microbatch_objective": (
                "mean-of-gap-normalized-group-losses"
            ),
            "ranking_lambda_application": "once-after-group-mean",
            "protected_tests_opened": False,
        },
        "pilot_admission": (
            None
            if pilot_admission is None
            else {
                "file": _record(pilot_admission.resolve()),
                "body_sha256": selected_pilot["body_sha256"],
                "ranking_weight": selected_pilot["selected_candidate"][
                    "ranking_weight"
                ],
                "search_variant": selected_pilot["selected_candidate"][
                    "search_variant"
                ],
                "compile_time_macros": selected_pilot["selected_candidate"][
                    "compile_time_macros"
                ],
                "search_throughput_profile": selected_pilot[
                    "search_throughput_profile"
                ],
                "candidate_search_profile": selected_pilot[
                    "candidate_search_profile"
                ],
                "qat_profile": selected_pilot["qat_profile"],
                "adaptation_contract": selected_pilot[
                    "adaptation_contract"
                ],
            }
        ),
        "input_audit": input_audit,
        "source_policy": {
            "limit_exclusive": SOURCE_LIMIT_EXCLUSIVE,
            "reserve_target": SOURCE_RESERVE_TARGET,
            "maximum_for_reserve_target": SOURCE_MAXIMUM_FOR_TARGET,
            "deterministic_compactor_required": True,
        },
        "search_variants": {
            name: list(macros) for name, macros in search_variants.items()
        },
        "build_source_closure": build_source_closure,
        "execution_authority": {
            "production_allowlist_enforced": production,
            "campaign_derived_output_base": (
                str(expected_output_base)
                if expected_output_base is not None else None
            ),
            "pipeline_execution_authority": pipeline_context[
                "execution_authority"
            ],
            "build_source_closure_sha256": build_source_closure[
                "closure_sha256"
            ],
            "injected_test_evidence_authorized": False,
            "heavy_stage_lock": (
                str(pathlib.Path(campaign_context["plan"]["outputs"]["root"])
                    .resolve() / ".rank4-teacher-heavy-stage.lock")
                if production else None
            ),
        },
        "outputs": paths,
        "tools": _tool_closure(),
    }
    plan_path = pathlib.Path(paths["plan"])
    _write_sealed(plan_path, body)
    load_training_plan(plan_path, revalidate_inputs=False)
    return plan_path


def load_training_plan(
    path: pathlib.Path, *, revalidate_inputs: bool = True,
) -> dict[str, Any]:
    plan = _load_sealed(path.resolve(), PLAN_SCHEMA, "teacher-training plan")
    root = path.parent.resolve()
    search_throughput_profile = _search_throughput_profile(
        plan.get("adaptation_contract")
    )
    expected_search_variants = active_search_variants(
        search_throughput_profile
    )
    if plan.get("outputs") != _phase_paths(root):
        raise TeacherTrainingError("teacher-training output routes changed")
    authority = plan.get("execution_authority")
    if (
        not isinstance(authority, Mapping)
        or set(authority) != {
            "production_allowlist_enforced", "campaign_derived_output_base",
            "pipeline_execution_authority", "build_source_closure_sha256",
            "injected_test_evidence_authorized", "heavy_stage_lock",
        }
        or not isinstance(authority.get("production_allowlist_enforced"), bool)
        or authority.get("build_source_closure_sha256")
        != plan.get("build_source_closure", {}).get("closure_sha256")
        or authority.get("injected_test_evidence_authorized") is not False
    ):
        raise TeacherTrainingError("teacher-training execution authority changed")
    if (
        plan.get("phase") not in {"pilot", "full"}
        or plan.get("architecture")
        != {
            "name": "capacity-12x8",
            "dimensions": [6301, 12, 8, 1],
            "biases": False,
            "activations": list(trainer.ACTIVATIONS),
            "quantization_bits": 3,
            "policy_head": False,
        }
        or plan.get("source_policy")
        != {
            "limit_exclusive": SOURCE_LIMIT_EXCLUSIVE,
            "reserve_target": SOURCE_RESERVE_TARGET,
            "maximum_for_reserve_target": SOURCE_MAXIMUM_FOR_TARGET,
            "deterministic_compactor_required": True,
        }
        or plan.get("search_throughput_profile")
        != search_throughput_profile
        or plan.get("search_variants")
        != {
            name: list(macros)
            for name, macros in expected_search_variants.items()
        }
    ):
        raise TeacherTrainingError("teacher-training architecture/source policy changed")
    expected_weights = list(PILOT_WEIGHTS)
    pilot = plan.get("pilot_admission")
    if plan["phase"] == "full":
        if not isinstance(pilot, Mapping):
            raise TeacherTrainingError("full plan lost its pilot admission")
        admission_path = _validate_record(pilot.get("file"), "pilot admission")
        admission = _validate_pilot_admission_for_full(
            admission_path,
            campaign_id=str(plan["campaign_id"]),
            attempt=int(plan["attempt"]),
        )
        if (
            pilot.get("body_sha256") != admission["body_sha256"]
            or pilot.get("ranking_weight")
            != admission["selected_candidate"]["ranking_weight"]
            or pilot.get("search_variant")
            != admission["selected_candidate"]["search_variant"]
            or pilot.get("compile_time_macros")
            != admission["selected_candidate"]["compile_time_macros"]
            or pilot.get("search_throughput_profile")
            != admission.get("search_throughput_profile")
            or pilot.get("candidate_search_profile")
            != admission.get("candidate_search_profile")
            or pilot.get("qat_profile") != admission.get("qat_profile")
            or pilot.get("adaptation_contract")
            != admission.get("adaptation_contract")
        ):
            raise TeacherTrainingError("full plan pilot admission binding changed")
        expected_weights = [0.0, float(pilot["ranking_weight"])]
    elif pilot is not None:
        raise TeacherTrainingError("pilot plan unexpectedly binds an admission")
    training = plan.get("training")
    training_qat_profile = training.get("qat_profile") if isinstance(
        training, Mapping
    ) else None
    try:
        expected_qat_contract = trainer.validate_qat_profile_contract(
            training.get("qat_profile_contract") if isinstance(
                training, Mapping
            ) else None,
            expected_name=(
                training_qat_profile
                if isinstance(training_qat_profile, str)
                else ""
            ),
        )
    except trainer.TrainingError as error:
        raise TeacherTrainingError("teacher-training QAT profile changed") from error
    if (
        not isinstance(training, Mapping)
        or training.get("ranking_weights") != expected_weights
        or training.get("fixed_seeds") != list(trainer.FIXED_SEEDS)
        or training.get("seed_workers") != 2
        or training.get("new_rows_per_batch") != 64
        or training.get("anchor_rows_per_batch") != 192
        or training.get("float_warmup_epochs") != 1
        or training.get("float_learning_rate") != 0.00006
        or training.get("qat_epochs") != 4
        or training.get("qat_profile_contract") != expected_qat_contract
        or training.get("native_thread_environment")
        != trainer.NATIVE_THREAD_ENVIRONMENT
        or training.get("ranking_epoch_schedule")
        != "balanced-full-weighted-pool-permutation-per-epoch-v1"
        or training.get("ranking_group_microbatch_objective")
        != "mean-of-gap-normalized-group-losses"
        or training.get("ranking_lambda_application")
        != "once-after-group-mean"
        or training.get("protected_tests_opened") is not False
    ):
        raise TeacherTrainingError("teacher-training schedule changed")
    for label in (
        "campaign_plan", "phase_reference", "pipeline_plan",
        "final_pipeline_receipt", "initial_checkpoint",
    ):
        _validate_record(plan.get(label), label)
    campaign_context = challenger.validate_campaign(
        _validate_record(plan["campaign_plan"], "campaign plan")
    )
    phase_context = challenger.validate_phase_reference(
        _validate_record(plan["phase_reference"], "phase reference"),
        campaign_context["plan"],
    )
    phase_qat_profile, phase_qat_contract = _phase_qat_profile(phase_context)
    phase_adaptation_contract = _phase_adaptation_contract(phase_context)
    if (
        plan.get("adaptation_contract") != phase_adaptation_contract
        or search_throughput_profile
        != phase_adaptation_contract["search_throughput_profile"]
        or training_qat_profile != phase_qat_profile
        or expected_qat_contract != phase_qat_contract
    ):
        raise TeacherTrainingError(
            "teacher-training QAT profile differs from phase adaptation"
        )
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
    if pipeline_plan.get("adaptation_contract") != phase_adaptation_contract:
        raise TeacherTrainingError(
            "teacher-training pipeline differs from phase adaptation"
        )
    if authority.get("pipeline_execution_authority") != pipeline_plan.get(
        "execution_authority"
    ):
        raise TeacherTrainingError("teacher/pipeline execution authority changed")
    production = (
        campaign_context["inputs"].get("production_allowlist_enforced") is True
    )
    if authority.get("production_allowlist_enforced") is not production:
        raise TeacherTrainingError("teacher-training production authority changed")
    if production:
        expected_base = _production_execution_base(campaign_context)
        expected_root = (
            expected_base / f"attempt-{int(plan['attempt']):03d}"
            / str(plan["phase"]) / "teacher-training"
        )
        if (
            authority.get("campaign_derived_output_base") != str(expected_base)
            or pathlib.Path(plan["outputs"]["root"]).resolve() != expected_root
            or pipeline_plan.get("execution_authority", {}).get(
                "production_allowlist_enforced"
            ) is not True
            or authority.get("heavy_stage_lock") != str(
                pathlib.Path(campaign_context["plan"]["outputs"]["root"])
                .resolve() / ".rank4-teacher-heavy-stage.lock"
            )
        ):
            raise TeacherTrainingError(
                "production teacher-training execution root changed"
            )
    elif (
        authority.get("campaign_derived_output_base") is not None
        or authority.get("heavy_stage_lock") is not None
    ):
        raise TeacherTrainingError("nonproduction training claims a production root")
    _revalidate_stored_build_source_closure(
        plan.get("build_source_closure"),
        pipeline_closure=pipeline_plan.get("build_source_closure"),
    )
    try:
        current_build_source_closure = (
            challenger.verify_phase_build_source_closure(
                required_sources=required_build_sources(),
                campaign_context=campaign_context,
                phase_context=phase_context,
            )
        )
    except Exception as error:
        raise TeacherTrainingError(
            "training code differs from its frozen build source closure"
        ) from error
    if current_build_source_closure != plan.get("build_source_closure"):
        raise TeacherTrainingError("teacher-training build source closure changed")
    _validate_pipeline_build_subset(
        current_build_source_closure, pipeline_plan.get("build_source_closure")
    )
    if pipeline_plan.get("body_sha256") != plan.get("pipeline_body_sha256"):
        raise TeacherTrainingError("training pipeline plan binding changed")
    phase_checkpoint, phase_checkpoint_record = _phase_input_path(
        campaign_context, phase_context, "initial_float_checkpoint"
    )
    if (
        plan.get("initial_checkpoint_phase_record") != dict(phase_checkpoint_record)
        or sha256_file(phase_checkpoint) != plan["initial_checkpoint"]["sha256"]
        or (
            plan["attempt"] == 1
            and plan["initial_checkpoint"]["sha256"]
            != ATTEMPT_ONE_INITIAL_CHECKPOINT_SHA256
        )
    ):
        raise TeacherTrainingError("phase-bound initial checkpoint changed")
    bundle_record = plan.get("source_bundle")
    if not isinstance(bundle_record, Mapping):
        raise TeacherTrainingError("source bundle record is absent")
    bundle_path = _validate_record(bundle_record.get("manifest"), "source bundle")
    bundle = trainer.FrozenBundle.load(bundle_path)
    if bundle.body_sha256 != bundle_record.get("body_sha256"):
        raise TeacherTrainingError("source bundle body changed")
    trainer.load_float_checkpoint(
        _validate_record(plan["initial_checkpoint"], "initial checkpoint"), ARCHITECTURE
    )
    _validate_tool_closure(plan.get("tools"))
    if revalidate_inputs:
        final_receipt = _load_sealed(
            _validate_record(plan["final_pipeline_receipt"], "pipeline receipt"),
            pipeline.STAGE_RECEIPT_SCHEMA,
            "final pipeline receipt",
        )
        if (
            pipeline_plan.get("body_sha256") != plan.get("pipeline_body_sha256")
            or final_receipt.get("body_sha256")
            != plan.get("final_pipeline_receipt_body_sha256")
            or final_receipt.get("pipeline_body_sha256")
            != pipeline_plan.get("body_sha256")
            or final_receipt.get("stage") != "07-finalize-labels"
        ):
            raise TeacherTrainingError("finalized pipeline binding changed")
        _inputs, _external_validation, audit = _build_training_inputs(
            pipeline_plan=pipeline_plan, final_receipt=final_receipt, bundle=bundle
        )
        if audit != plan.get("input_audit"):
            raise TeacherTrainingError("recomputed training input audit changed")
    return plan


def training_context(
    plan: Mapping[str, Any],
) -> tuple[trainer.FrozenBundle, trainer.TrainingInputs, trainer.Dataset]:
    bundle = trainer.FrozenBundle.load(
        _validate_record(plan["source_bundle"]["manifest"], "source bundle")
    )
    if bundle.body_sha256 != plan["source_bundle"]["body_sha256"]:
        raise TeacherTrainingError("training source bundle identity changed")
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
    _revalidate_stored_build_source_closure(
        plan.get("build_source_closure"),
        pipeline_closure=pipeline_plan.get("build_source_closure"),
    )
    final_receipt = _load_sealed(
        _validate_record(plan["final_pipeline_receipt"], "pipeline receipt"),
        pipeline.STAGE_RECEIPT_SCHEMA,
        "final pipeline receipt",
    )
    if (
        pipeline_plan.get("body_sha256") != plan.get("pipeline_body_sha256")
        or final_receipt.get("body_sha256")
        != plan.get("final_pipeline_receipt_body_sha256")
        or final_receipt.get("pipeline_body_sha256")
        != pipeline_plan.get("body_sha256")
        or final_receipt.get("stage") != "07-finalize-labels"
    ):
        raise TeacherTrainingError("finalized pipeline binding changed")
    inputs, external_validation, audit = _build_training_inputs(
        pipeline_plan=pipeline_plan, final_receipt=final_receipt, bundle=bundle
    )
    if audit != plan["input_audit"]:
        raise TeacherTrainingError("training inputs changed after plan preparation")
    return bundle, inputs, external_validation


def _weight_slug(weight: float) -> str:
    return {0.0: "lambda-000", 0.10: "lambda-010", 0.25: "lambda-025"}[float(weight)]


def _artifact_from_receipt(
    root: pathlib.Path, record: object, label: str,
) -> pathlib.Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        raise TeacherTrainingError(f"{label} record is malformed")
    path = _safe_child(root, record.get("path"), label)
    actual = _record(path)
    if (
        actual["sha256"] != record.get("sha256")
        or actual["bytes"] != record.get("bytes")
    ):
        raise TeacherTrainingError(f"{label} bytes changed")
    return path


def _runtime_source(runtime: pathlib.Path) -> bytes:
    try:
        document, _payload, metadata = model_exporter.validate_runtime(runtime)
        header, rendered = model_exporter.render_header(runtime)
        _default, source = source_exporter.render(model_header=header)
    except Exception as error:
        raise TeacherTrainingError("selected runtime source export failed") from error
    architecture = document.get("architecture", {})
    if (
        architecture.get("name") != "capacity-12x8"
        or architecture.get("dimensions") != [6301, 12, 8, 1]
        or document.get("selection", {}).get("arm") != "search-target"
        or metadata.get("file_sha256") != sha256_file(runtime)
        or rendered.get("file_sha256") != sha256_file(runtime)
    ):
        raise TeacherTrainingError("selected runtime changed the deployment contract")
    try:
        decoded = source.decode("ascii")
        compacted = source_exporter.compact_cpp_code(decoded).encode("ascii")
    except (UnicodeDecodeError, ValueError) as error:
        raise TeacherTrainingError("generated source is not deterministic compact ASCII") from error
    if compacted != source:
        raise TeacherTrainingError("generated source compactor is not idempotent")
    return source


def _source_record(
    *, runtime: pathlib.Path, output_directory: pathlib.Path,
    renderer: Callable[[pathlib.Path], bytes],
) -> dict[str, object]:
    payload = renderer(runtime)
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise TeacherTrainingError("rendered candidate source is not ASCII") from error
    if not 0 < len(payload) < SOURCE_LIMIT_EXCLUSIVE:
        raise TeacherTrainingError("rendered candidate source is not below 95,000 ASCII bytes")
    reserve = SOURCE_LIMIT_EXCLUSIVE - len(payload)
    if reserve < SOURCE_RESERVE_TARGET:
        raise TeacherTrainingError("rendered candidate source misses the 2KB reserve target")
    source = _write_content_addressed(output_directory, payload, ".submission.cpp")
    return {
        **_record(source, ascii_required=True),
        "ascii_bytes": len(payload),
        "limit_exclusive": SOURCE_LIMIT_EXCLUSIVE,
        "reserve": reserve,
        "reserve_target": SOURCE_RESERVE_TARGET,
        "reserve_target_met": True,
        "compactor": _record(SOURCE_EXPORTER_PATH),
    }


def _compile_cpp(
    source: pathlib.Path, output: pathlib.Path, *, compiler: str | None = None,
) -> None:
    command = shutil.which(compiler or os.environ.get("CXX", "c++"))
    if command is None:
        raise TeacherTrainingError("C++ compiler is unavailable")
    completed = subprocess.run(
        [command, "-std=c++20", "-O3", str(source), "-o", str(output)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise TeacherTrainingError(f"candidate source compilation failed: {completed.stderr}")


def _verify_exported_source(
    runtime: pathlib.Path, source: pathlib.Path, dataset: trainer.Dataset,
    output_directory: pathlib.Path,
) -> Mapping[str, object]:
    """Compile the standalone bot and bit-compare its C++ scalar inference."""

    architecture, quantized, _selection, _document = trainer.load_runtime(runtime)
    if architecture.name != ARCHITECTURE.name:
        raise TeacherTrainingError("source verification runtime changed architecture")
    rows = min(4_096, len(dataset))
    if rows <= 0:
        raise TeacherTrainingError("source verification parity corpus is empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    compiler = _compiler_identity()
    standalone_temporary = output_directory / f".{sha256_file(source)}.bot.tmp"
    probe_source_temporary = output_directory / f".{sha256_file(source)}.probe.cpp"
    probe_binary_temporary = output_directory / f".{sha256_file(source)}.probe.tmp"
    try:
        _compile_cpp(source, standalone_temporary)
        standalone = _write_content_addressed(
            output_directory,
            standalone_temporary.read_bytes(),
            ".submission-binary",
            executable=True,
        )
        probe_text = (COMPACT_DIRECTORY / "inference_probe.cpp").read_text(
            encoding="utf-8"
        )
        marker = '#include "submission.cpp"'
        if probe_text.count(marker) != 1:
            raise TeacherTrainingError("maintained inference probe include changed")
        include_path = str(source.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        probe_text = probe_text.replace(marker, f'#include "{include_path}"')
        probe_source_temporary.write_text(probe_text, encoding="utf-8")
        _compile_cpp(probe_source_temporary, probe_binary_temporary)
        lines = []
        expected: list[np.float32] = []
        for row in range(rows):
            active = dataset.active_row(row)
            lines.append(",".join(str(int(value)) for value in active))
            value = trainer.scalar_quantized_forward(quantized, architecture, active)
            expected.append(np.float32(value))
        completed = subprocess.run(
            [str(probe_binary_temporary)],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        observed_hex = completed.stdout.splitlines()
        try:
            observed = np.asarray(
                [
                    struct.unpack("<f", struct.pack("<I", int(value, 16)))[0]
                    for value in observed_hex
                ],
                dtype=np.float32,
            )
        except (ValueError, struct.error) as error:
            raise TeacherTrainingError("exported C++ inference probe output is malformed") from error
        expected_array = np.asarray(expected, dtype=np.float32)
        difference = (
            math.inf
            if completed.returncode != 0 or observed.shape != expected_array.shape
            else float(np.max(np.abs(observed - expected_array)))
        )
        tolerance = 2e-6
        if not math.isfinite(difference) or difference > tolerance:
            raise TeacherTrainingError(
                f"exported C++ inference parity differs by {difference:.9g}"
            )
        probe_binary = _write_content_addressed(
            output_directory,
            probe_binary_temporary.read_bytes(),
            ".inference-probe",
            executable=True,
        )
    finally:
        standalone_temporary.unlink(missing_ok=True)
        probe_source_temporary.unlink(missing_ok=True)
        probe_binary_temporary.unlink(missing_ok=True)
    return {
        "compiled": True,
        "standalone_binary": _record(standalone),
        "inference_probe_binary": _record(probe_binary),
        "states": rows,
        "comparison": "scalar-float32-hex-vs-maintained-python-scalar",
        "maximum_absolute_error": difference,
        "tolerance": tolerance,
        "mismatches": 0,
        "compiler": compiler,
        "passed": True,
    }


def _validate_source_verification(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "compiled", "standalone_binary", "inference_probe_binary", "states",
        "comparison", "maximum_absolute_error", "tolerance", "mismatches",
        "compiler", "passed",
    }:
        raise TeacherTrainingError("source compile/parity evidence is malformed")
    difference = value.get("maximum_absolute_error")
    tolerance = value.get("tolerance")
    if (
        value.get("compiled") is not True
        or value.get("passed") is not True
        or value.get("states") != 4_096
        or value.get("comparison")
        != "scalar-float32-hex-vs-maintained-python-scalar"
        or value.get("mismatches") != 0
        or isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or float(tolerance) != 2e-6
        or float(difference) > float(tolerance)
        or not isinstance(value.get("compiler"), Mapping)
    ):
        raise TeacherTrainingError("source compile/parity evidence failed")
    _validate_record(value.get("standalone_binary"), "standalone candidate binary")
    _validate_record(value.get("inference_probe_binary"), "inference probe binary")
    return dict(value)


RosterRunner = Callable[
    [
        trainer.FrozenBundle,
        trainer.TrainingInputs,
        trainer.Architecture,
        trainer.Arm,
        pathlib.Path,
        float,
        pathlib.Path,
        str,
        bool,
    ],
    Sequence[Mapping[str, Any]],
]
SourceVerifier = Callable[
    [pathlib.Path, pathlib.Path, trainer.Dataset, pathlib.Path],
    Mapping[str, object],
]


def _run_seed_roster(
    bundle: trainer.FrozenBundle, inputs: trainer.TrainingInputs,
    architecture: trainer.Architecture, arm: trainer.Arm,
    output_directory: pathlib.Path, ranking_weight: float,
    initial_checkpoint: pathlib.Path, qat_profile: str, resume: bool,
) -> Sequence[Mapping[str, Any]]:
    return trainer._train_seed_roster(
        bundle,
        inputs,
        architecture,
        arm,
        output_directory,
        seed_workers=SEED_WORKERS,
        sidecar_index=None,
        ranking_weight=ranking_weight,
        initial_checkpoint=initial_checkpoint,
        qat_profile=qat_profile,
        resume=resume,
    )


def _validate_seed_roster(
    receipts: Sequence[Mapping[str, Any]], *, ranking_weight: float,
    qat_profile: str, teacher_ranking_profile: str,
) -> list[dict[str, Any]]:
    profile_contract = trainer.qat_profile_contract(qat_profile)
    if (
        len(receipts) != len(trainer.FIXED_SEEDS)
        or [receipt.get("seed") for receipt in receipts] != list(trainer.FIXED_SEEDS)
    ):
        raise TeacherTrainingError("training did not return the exact fixed seed roster")
    normalized: list[dict[str, Any]] = []
    native_executions: list[dict[str, object]] = []
    for receipt in receipts:
        successor = receipt.get("successor_ranking")
        if (
            receipt.get("architecture") != ARCHITECTURE.name
            or receipt.get("arm") != ARM.name
            or not isinstance(successor, Mapping)
            or successor.get("labels_present") is not True
            or float(successor.get("loss_weight", -1.0)) != ranking_weight
            or receipt.get("qat_profile") != qat_profile
            or receipt.get("qat_profile_contract") != profile_contract
            or not isinstance(receipt.get("offline_gate"), Mapping)
            or not isinstance(receipt.get("quantized_validation"), Mapping)
            or not isinstance(receipt.get("float_checkpoint"), Mapping)
            or not isinstance(receipt.get("quantized_runtime"), Mapping)
        ):
            raise TeacherTrainingError("seed receipt changed its ranking/runtime contract")
        try:
            trainer.validate_qat_execution_evidence(
                receipt.get("quantized_training"),
                expected_profile=qat_profile,
            )
            native_execution = trainer.validate_native_thread_execution(
                receipt.get("native_thread_execution")
            )
            schedule_execution = trainer.validate_successor_schedule_execution(
                receipt.get("float_training"), receipt.get("quantized_training"),
                seed=receipt.get("seed"),
            )
            if successor.get("schedule_execution") != schedule_execution:
                raise trainer.TrainingError(
                    "seed receipt successor schedule summary changed"
                )
        except trainer.TrainingError as error:
            raise TeacherTrainingError(
                "seed receipt QAT/native-thread execution evidence changed"
            ) from error
        _validated_hard_state_density(
            successor.get("hard_state_density"),
            teacher_ranking_profile=teacher_ranking_profile,
        )
        native_executions.append(native_execution)
        normalized.append(dict(receipt))
    if any(value != native_executions[0] for value in native_executions[1:]):
        raise TeacherTrainingError(
            "concurrent seed receipts disagree on native-thread execution"
        )
    return normalized


def _selected_seed(
    receipts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    passing = [
        receipt
        for receipt in receipts
        if receipt.get("offline_gate", {}).get("passed") is True
    ]
    pool = passing or list(receipts)
    try:
        return min(pool, key=trainer._receipt_selection_key)
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherTrainingError("seed validation metrics cannot be ranked") from error


def _ranking_metrics(receipt: Mapping[str, Any]) -> dict[str, float | int]:
    metrics = receipt.get("quantized_validation", {}).get("successor_ranking")
    if not isinstance(metrics, Mapping):
        raise TeacherTrainingError("selected seed lacks quantized ranking metrics")
    required = (
        "mean_teacher_regret", "top1_agreement",
        "float_vs_quantized_action_flip_rate",
    )
    result: dict[str, float | int] = {}
    for name in required:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TeacherTrainingError(f"selected seed ranking metric {name} is invalid")
        result[name] = float(value)
    groups = metrics.get("groups")
    comparable = metrics.get("comparable_groups")
    if (
        isinstance(groups, bool)
        or not isinstance(groups, int)
        or groups <= 0
        or isinstance(comparable, bool)
        or not isinstance(comparable, int)
        or not 0 <= comparable <= groups
    ):
        raise TeacherTrainingError("selected seed ranking coverage is invalid")
    result["ranking_validation_groups"] = groups
    result["comparable_exhaustive_validation_groups"] = comparable
    result["comparable_exhaustive_validation_fraction"] = comparable / groups
    if (
        result["mean_teacher_regret"] < 0.0
        or not 0.0 <= result["top1_agreement"] <= 1.0
        or not 0.0 <= result["float_vs_quantized_action_flip_rate"] <= 1.0
    ):
        raise TeacherTrainingError("selected seed ranking metrics are out of range")
    return result


def _regret_reduction(control: float, candidate: float) -> float:
    if control <= 0.0:
        return 0.0
    return (control - candidate) / control


def _model_selection(arms: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    by_weight = {float(arm["ranking_weight"]): arm for arm in arms}
    if 0.0 not in by_weight or len(by_weight) != len(arms):
        raise TeacherTrainingError("training arms omit or duplicate the scalar control")
    control = by_weight[0.0]
    control_metrics = control["metrics"]
    control_coverage_passed = bool(
        int(control_metrics["comparable_exhaustive_validation_groups"]) >= 100
        and float(control_metrics["comparable_exhaustive_validation_fraction"])
        >= 0.80
    )
    eligible = []
    comparisons = []
    for weight in sorted(set(by_weight) - {0.0}):
        arm = by_weight[weight]
        metrics = arm["metrics"]
        reduction = _regret_reduction(
            float(control_metrics["mean_teacher_regret"]),
            float(metrics["mean_teacher_regret"]),
        )
        passed = bool(
            arm["offline_gate_passed"]
            and control_coverage_passed
            and int(metrics["comparable_exhaustive_validation_groups"]) >= 100
            and float(metrics["comparable_exhaustive_validation_fraction"])
            >= 0.80
            and reduction >= 0.10
            and float(metrics["float_vs_quantized_action_flip_rate"])
            <= float(control_metrics["float_vs_quantized_action_flip_rate"]) + 0.005
            and arm["source"]["reserve_target_met"] is True
        )
        comparison = {
            "ranking_weight": weight,
            "mean_teacher_regret_reduction_fraction": reduction,
            "candidate_flip_rate": metrics[
                "float_vs_quantized_action_flip_rate"
            ],
            "scalar_control_flip_rate": control_metrics[
                "float_vs_quantized_action_flip_rate"
            ],
            "canonical_retention_passed": arm["offline_gate_passed"],
            "source_reserve_target_met": arm["source"]["reserve_target_met"],
            "ranking_validation_groups": metrics["ranking_validation_groups"],
            "comparable_exhaustive_validation_groups": metrics[
                "comparable_exhaustive_validation_groups"
            ],
            "comparable_exhaustive_validation_fraction": metrics[
                "comparable_exhaustive_validation_fraction"
            ],
            "scalar_control_coverage_passed": control_coverage_passed,
            "eligible": passed,
        }
        comparisons.append(comparison)
        if passed:
            eligible.append((arm, comparison))
    if not eligible:
        diagnostic = min(
            (by_weight[weight] for weight in sorted(set(by_weight) - {0.0})),
            key=lambda arm: (
                float(arm["metrics"]["mean_teacher_regret"]),
                -float(arm["metrics"]["top1_agreement"]),
                float(arm["metrics"]["float_vs_quantized_action_flip_rate"]),
                float(arm["ranking_weight"]),
                int(arm["seed"]),
            ),
        )
        return {
            "status": "offline-rejected-before-rank4-screen",
            "control_ranking_weight": 0.0,
            "comparisons": comparisons,
            "selected_ranking_weight": None,
            "diagnostic_ranking_weight": diagnostic["ranking_weight"],
            "diagnostic_seed": diagnostic["seed"],
        }
    selected_arm, selected_comparison = min(
        eligible,
        key=lambda item: (
            float(item[0]["metrics"]["mean_teacher_regret"]),
            -float(item[0]["metrics"]["top1_agreement"]),
            float(item[0]["metrics"]["float_vs_quantized_action_flip_rate"]),
            float(item[0]["ranking_weight"]),
        ),
    )
    return {
        "status": "model-selected-before-rank4-screen",
        "control_ranking_weight": 0.0,
        "comparisons": comparisons,
        "selected_ranking_weight": selected_arm["ranking_weight"],
        "diagnostic_ranking_weight": None,
        "diagnostic_seed": None,
        "selected_seed": selected_arm["seed"],
        "selected_mean_teacher_regret_reduction_fraction": selected_comparison[
            "mean_teacher_regret_reduction_fraction"
        ],
    }


def _selection_reference_path(plan: Mapping[str, Any]) -> pathlib.Path:
    return pathlib.Path(plan["outputs"]["selection_reference"])


def _load_reference(
    path: pathlib.Path, *, schema: str, receipt_schema: str,
    expected_plan_body_sha256: str, label: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    reference = _load_sealed(path, schema, f"{label} reference")
    if reference.get("plan_body_sha256") != expected_plan_body_sha256:
        raise TeacherTrainingError(f"{label} reference uses another plan")
    receipt_path = _validate_record(reference.get("receipt"), f"{label} receipt")
    receipt = _load_sealed(receipt_path, receipt_schema, label)
    if (
        receipt.get("plan_body_sha256") != expected_plan_body_sha256
        or reference.get("receipt_body_sha256") != receipt.get("body_sha256")
    ):
        raise TeacherTrainingError(f"{label} receipt uses another plan")
    return receipt_path, receipt


@_heavy_stage
def run_training(
    plan_path: pathlib.Path, *, resume: bool = False,
    roster_runner: RosterRunner | None = None,
    renderer: Callable[[pathlib.Path], bytes] | None = None,
    source_verifier: SourceVerifier | None = None,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    _guard_test_hooks(
        plan,
        hooks_used=any(
            hook is not None
            for hook in (roster_runner, renderer, source_verifier)
        ),
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    reference_path = _selection_reference_path(plan)
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("training selection is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=SELECTION_REFERENCE_SCHEMA,
            receipt_schema=SELECTION_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="training selection",
        )
        load_training_selection(plan, receipt_path)
        return receipt_path

    bundle, inputs, _external_validation = training_context(plan)
    initial_checkpoint = _validate_record(
        plan["initial_checkpoint"], "initial float checkpoint"
    )
    run_roster = roster_runner or _run_seed_roster
    render_source = renderer or _runtime_source
    verify_source = source_verifier or _verify_exported_source
    qat_profile = str(plan["training"]["qat_profile"])
    qat_profile_contract = trainer.validate_qat_profile_contract(
        plan["training"]["qat_profile_contract"],
        expected_name=qat_profile,
    )
    arms = []
    for weight_value in plan["training"]["ranking_weights"]:
        weight = float(weight_value)
        arm_root = pathlib.Path(plan["outputs"]["runs"]) / _weight_slug(weight)
        receipts = _validate_seed_roster(
            run_roster(
                bundle,
                inputs,
                ARCHITECTURE,
                ARM,
                arm_root,
                weight,
                initial_checkpoint,
                qat_profile,
                resume,
            ),
            ranking_weight=weight,
            qat_profile=qat_profile,
            teacher_ranking_profile=plan["adaptation_contract"][
                "teacher_ranking_profile"
            ],
        )
        selected = _selected_seed(receipts)
        runtime = _artifact_from_receipt(
            arm_root, selected["quantized_runtime"], "selected quantized runtime"
        )
        checkpoint = _artifact_from_receipt(
            arm_root, selected["float_checkpoint"], "selected float checkpoint"
        )
        source = _source_record(
            runtime=runtime,
            output_directory=pathlib.Path(plan["outputs"]["sources"]),
            renderer=render_source,
        )
        source_path = _validate_record(
            {key: source[key] for key in ("path", "bytes", "sha256")},
            "selected source",
        )
        source_verification = _validate_source_verification(verify_source(
            runtime,
            source_path,
            inputs.common_adjudicator,
            pathlib.Path(plan["outputs"]["source_verification"])
            / _weight_slug(weight),
        ))
        receipt_records = []
        for receipt in receipts:
            payload = canonical_json_bytes(dict(receipt))
            copied = _write_content_addressed(
                pathlib.Path(plan["outputs"]["selections"]) / "seed-evidence",
                payload,
                ".seed-receipt.json",
            )
            receipt_records.append({
                "seed": receipt["seed"],
                "receipt": _record(copied),
                "body_sha256": receipt.get("body_sha256"),
                "offline_gate_passed": receipt["offline_gate"].get("passed") is True,
            })
        successor = selected.get("successor_ranking", {})
        density = _validated_hard_state_density(
            successor.get("hard_state_density"),
            teacher_ranking_profile=plan["adaptation_contract"][
                "teacher_ranking_profile"
            ],
        )
        arms.append({
            "ranking_weight": weight,
            "seed": selected["seed"],
            "adaptation_contract": plan["adaptation_contract"],
            "qat_profile": qat_profile,
            "qat_profile_contract": qat_profile_contract,
            "qat_execution_evidence": selected["quantized_training"],
            "hard_state_density": density,
            "offline_gate_passed": selected["offline_gate"].get("passed") is True,
            "runtime": _record(runtime),
            "float_checkpoint": _record(checkpoint),
            "source": source,
            "source_compile_and_parity": source_verification,
            "metrics": _ranking_metrics(selected),
            "float_validation": selected["float_validation"],
            "quantized_validation": selected["quantized_validation"],
            "inference_parity": selected.get("inference_parity"),
            "float_per_layer_update_evidence": successor.get(
                "float_per_layer_update_evidence"
            ),
            "qat_per_layer_update_evidence": successor.get(
                "qat_per_layer_update_evidence"
            ),
            "seed_receipts": receipt_records,
            "seed_selection_order": [
                receipt["seed"]
                for receipt in sorted(receipts, key=trainer._receipt_selection_key)
            ],
        })

    model_selection = _model_selection(arms)
    selected_weight = (
        model_selection["selected_ranking_weight"]
        if model_selection["selected_ranking_weight"] is not None
        else model_selection["diagnostic_ranking_weight"]
    )
    selected_model = None
    if selected_weight is not None:
        selected_arm = next(
            arm
            for arm in arms
            if arm["ranking_weight"] == selected_weight
        )
        offline_eligible = model_selection["selected_ranking_weight"] is not None
        selected_model = {
            "ranking_weight": selected_arm["ranking_weight"],
            "seed": selected_arm["seed"],
            "adaptation_contract": selected_arm["adaptation_contract"],
            "search_throughput_profile": plan[
                "search_throughput_profile"
            ],
            "active_search_variant_roster": list(plan["search_variants"]),
            "qat_profile": selected_arm["qat_profile"],
            "qat_profile_contract": selected_arm["qat_profile_contract"],
            "qat_execution_evidence": selected_arm[
                "qat_execution_evidence"
            ],
            "hard_state_density": selected_arm["hard_state_density"],
            "runtime": selected_arm["runtime"],
            "float_checkpoint": selected_arm["float_checkpoint"],
            "source": selected_arm["source"],
            "metrics": selected_arm["metrics"],
            "offline_eligible": offline_eligible,
            "diagnostic_only": not offline_eligible,
            "selected_before_rank4_bank_read": True,
        }
    body: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "adaptation_contract": plan["adaptation_contract"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "active_search_variant_roster": list(plan["search_variants"]),
        "plan_body_sha256": plan["body_sha256"],
        "source_bundle_body_sha256": plan["source_bundle"]["body_sha256"],
        "initial_checkpoint": plan["initial_checkpoint"],
        "training_policy": plan["training"],
        "execution_authority": plan["execution_authority"],
        "input_audit": plan["input_audit"],
        "arms": arms,
        "model_selection": model_selection,
        "selected_model": selected_model,
        "rank4_screen_bank_read": False,
        "protected_tests_opened": False,
    }
    document = _sealed(body)
    payload = canonical_json_bytes(document)
    receipt_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["selections"]),
        payload,
        ".training-selection.json",
    )
    _write_sealed(
        reference_path,
        {
            "schema": SELECTION_REFERENCE_SCHEMA,
            "plan_body_sha256": plan["body_sha256"],
            "receipt": _record(receipt_path),
            "receipt_body_sha256": document["body_sha256"],
        },
    )
    return receipt_path.resolve()


def _reconstructed_training_arm(
    plan: Mapping[str, Any], arm: Mapping[str, Any], *,
    bundle: trainer.FrozenBundle, inputs: trainer.TrainingInputs,
    qat_profile: str,
) -> dict[str, Any]:
    """Deep-load copied seed evidence and rebuild every seed-derived arm field."""

    weight = float(arm["ranking_weight"])
    arm_root = pathlib.Path(plan["outputs"]["runs"]) / _weight_slug(weight)
    initial_checkpoint = _validate_record(
        plan["initial_checkpoint"], "selection initial checkpoint"
    )
    records = arm.get("seed_receipts")
    if (
        not isinstance(records, list)
        or len(records) != len(trainer.FIXED_SEEDS)
    ):
        raise TeacherTrainingError("training arm seed evidence is absent")
    receipts: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    for expected_seed, record in zip(trainer.FIXED_SEEDS, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "seed", "receipt", "body_sha256", "offline_gate_passed",
        }:
            raise TeacherTrainingError("copied seed evidence roster changed")
        receipt_path = _validate_record(
            record.get("receipt"), f"copied seed {expected_seed} receipt"
        )
        receipt = _load_sealed(
            receipt_path, trainer.SEED_RECEIPT_SCHEMA,
            f"copied seed {expected_seed} receipt",
        )
        if (
            record.get("seed") != expected_seed
            or receipt.get("seed") != expected_seed
            or record.get("body_sha256") != receipt.get("body_sha256")
            or record.get("offline_gate_passed")
            is not (receipt.get("offline_gate", {}).get("passed") is True)
            or receipt_path.name
            != f"{sha256_file(receipt_path)}.seed-receipt.json"
        ):
            raise TeacherTrainingError("copied seed receipt identity changed")
        try:
            expected_binding = trainer.training_binding(
                bundle, inputs, ARCHITECTURE, ARM, expected_seed, None,
                weight, initial_checkpoint, qat_profile,
            )
        except trainer.TrainingError as error:
            raise TeacherTrainingError(
                "copied seed training binding cannot be reconstructed"
            ) from error
        if receipt.get("binding") != expected_binding:
            raise TeacherTrainingError("copied seed training binding changed")
        _artifact_from_receipt(
            arm_root, receipt.get("quantized_runtime"),
            f"seed {expected_seed} quantized runtime",
        )
        _artifact_from_receipt(
            arm_root, receipt.get("float_checkpoint"),
            f"seed {expected_seed} float checkpoint",
        )
        receipts.append(receipt)
        normalized_records.append({
            "seed": expected_seed,
            "receipt": _record(receipt_path),
            "body_sha256": receipt["body_sha256"],
            "offline_gate_passed": receipt["offline_gate"].get("passed") is True,
        })
    receipts = _validate_seed_roster(
        receipts,
        ranking_weight=weight,
        qat_profile=qat_profile,
        teacher_ranking_profile=plan["adaptation_contract"][
            "teacher_ranking_profile"
        ],
    )
    selected = _selected_seed(receipts)
    runtime = _artifact_from_receipt(
        arm_root, selected["quantized_runtime"], "selected quantized runtime"
    )
    checkpoint = _artifact_from_receipt(
        arm_root, selected["float_checkpoint"], "selected float checkpoint"
    )
    source = arm.get("source")
    if not isinstance(source, Mapping):
        raise TeacherTrainingError("arm source record is absent")
    source_path = _validate_record(
        {key: source[key] for key in ("path", "bytes", "sha256")},
        "selected source",
    )
    rendered = _runtime_source(runtime)
    expected_source = {
        **_record(source_path, ascii_required=True),
        "ascii_bytes": len(rendered),
        "limit_exclusive": SOURCE_LIMIT_EXCLUSIVE,
        "reserve": SOURCE_LIMIT_EXCLUSIVE - len(rendered),
        "reserve_target": SOURCE_RESERVE_TARGET,
        "reserve_target_met": (
            SOURCE_LIMIT_EXCLUSIVE - len(rendered) >= SOURCE_RESERVE_TARGET
        ),
        "compactor": _record(SOURCE_EXPORTER_PATH),
    }
    if source_path.read_bytes() != rendered or dict(source) != expected_source:
        raise TeacherTrainingError("selected source differs from its seed runtime")
    source_verification = _validate_source_verification(
        arm.get("source_compile_and_parity")
    )
    successor = selected.get("successor_ranking", {})
    density = _validated_hard_state_density(
        successor.get("hard_state_density"),
        teacher_ranking_profile=plan["adaptation_contract"][
            "teacher_ranking_profile"
        ],
    )
    return {
        "ranking_weight": weight,
        "seed": selected["seed"],
        "adaptation_contract": plan["adaptation_contract"],
        "qat_profile": qat_profile,
        "qat_profile_contract": trainer.qat_profile_contract(qat_profile),
        "qat_execution_evidence": selected["quantized_training"],
        "hard_state_density": density,
        "offline_gate_passed": selected["offline_gate"].get("passed") is True,
        "runtime": _record(runtime),
        "float_checkpoint": _record(checkpoint),
        "source": expected_source,
        "source_compile_and_parity": source_verification,
        "metrics": _ranking_metrics(selected),
        "float_validation": selected["float_validation"],
        "quantized_validation": selected["quantized_validation"],
        "inference_parity": selected.get("inference_parity"),
        "float_per_layer_update_evidence": successor.get(
            "float_per_layer_update_evidence"
        ),
        "qat_per_layer_update_evidence": successor.get(
            "qat_per_layer_update_evidence"
        ),
        "seed_receipts": normalized_records,
        "seed_selection_order": [
            receipt["seed"]
            for receipt in sorted(receipts, key=trainer._receipt_selection_key)
        ],
    }


def load_training_selection(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    value = _load_sealed(path.resolve(), SELECTION_SCHEMA, "training selection")
    selected = value.get("selected_model")
    if (
        set(value) != {
            "schema", "campaign_id", "attempt", "phase",
            "adaptation_contract", "search_throughput_profile",
            "active_search_variant_roster", "plan_body_sha256",
            "source_bundle_body_sha256", "initial_checkpoint",
            "training_policy", "input_audit", "arms", "model_selection",
            "execution_authority",
            "selected_model", "rank4_screen_bank_read",
            "protected_tests_opened", "body_sha256",
        }
        or value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("adaptation_contract") != plan["adaptation_contract"]
        or value.get("search_throughput_profile")
        != plan["search_throughput_profile"]
        or value.get("active_search_variant_roster")
        != list(plan["search_variants"])
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("source_bundle_body_sha256")
        != plan["source_bundle"]["body_sha256"]
        or value.get("initial_checkpoint") != plan["initial_checkpoint"]
        or value.get("training_policy") != plan["training"]
        or value.get("execution_authority") != plan["execution_authority"]
        or value.get("input_audit") != plan["input_audit"]
        or value.get("rank4_screen_bank_read") is not False
        or value.get("protected_tests_opened") is not False
        or not isinstance(value.get("arms"), list)
        or value.get("model_selection") != _model_selection(value["arms"])
    ):
        raise TeacherTrainingError("training selection binding changed")
    qat_profile = str(plan["training"]["qat_profile"])
    qat_profile_contract = trainer.validate_qat_profile_contract(
        plan["training"]["qat_profile_contract"],
        expected_name=qat_profile,
    )
    bundle, inputs, _external_validation = training_context(plan)
    for arm in value["arms"]:
        if (
            arm.get("seed") not in trainer.FIXED_SEEDS
            or not isinstance(arm.get("ranking_weight"), (int, float))
            or float(arm["ranking_weight"]) not in trainer.RANKING_LOSS_WEIGHTS
            or not isinstance(arm.get("metrics"), Mapping)
            or not isinstance(arm.get("seed_receipts"), list)
            or [item.get("seed") for item in arm["seed_receipts"]]
            != list(trainer.FIXED_SEEDS)
            or arm.get("qat_profile") != qat_profile
            or arm.get("qat_profile_contract") != qat_profile_contract
            or arm.get("adaptation_contract") != plan["adaptation_contract"]
        ):
            raise TeacherTrainingError("training arm/seed roster changed")
        try:
            trainer.validate_qat_execution_evidence(
                arm.get("qat_execution_evidence"),
                expected_profile=qat_profile,
            )
        except trainer.TrainingError as error:
            raise TeacherTrainingError(
                "training arm QAT execution evidence changed"
            ) from error
        _validated_hard_state_density(
            arm.get("hard_state_density"),
            teacher_ranking_profile=plan["adaptation_contract"][
                "teacher_ranking_profile"
            ],
        )
        _validate_record(arm.get("runtime"), "arm runtime")
        _validate_record(arm.get("float_checkpoint"), "arm float checkpoint")
        source = arm.get("source")
        if not isinstance(source, Mapping):
            raise TeacherTrainingError("arm source record is absent")
        _validate_record(
            {key: source[key] for key in ("path", "bytes", "sha256")},
            "arm source",
        )
        _validate_source_verification(arm.get("source_compile_and_parity"))
        for seed in arm["seed_receipts"]:
            _validate_record(seed.get("receipt"), "copied seed receipt")
        reconstructed = _reconstructed_training_arm(
            plan, arm, bundle=bundle, inputs=inputs, qat_profile=qat_profile
        )
        if dict(arm) != reconstructed:
            raise TeacherTrainingError(
                "training arm differs from its copied seed receipts"
            )
    if [float(arm["ranking_weight"]) for arm in value["arms"]] != [
        float(weight) for weight in plan["training"]["ranking_weights"]
    ]:
        raise TeacherTrainingError("training selection arm roster changed")
    model_selection = value["model_selection"]
    selected_weight = model_selection["selected_ranking_weight"]
    offline_eligible = selected_weight is not None
    expected_weight = (
        selected_weight
        if offline_eligible
        else model_selection.get("diagnostic_ranking_weight")
    )
    expected_seed = model_selection.get(
        "selected_seed" if offline_eligible else "diagnostic_seed"
    )
    selected_fields = {
        "ranking_weight", "seed", "adaptation_contract",
        "search_throughput_profile", "active_search_variant_roster",
        "qat_profile", "qat_profile_contract", "qat_execution_evidence",
        "hard_state_density", "runtime", "float_checkpoint", "source",
        "metrics", "offline_eligible", "diagnostic_only",
        "selected_before_rank4_bank_read",
    }
    if not isinstance(selected, Mapping) or set(selected) != selected_fields:
        raise TeacherTrainingError("model selection lost its screened candidate")
    matching = [
        arm
        for arm in value["arms"]
        if arm["ranking_weight"] == expected_weight
    ]
    if (
        expected_weight is None
        or expected_seed is None
        or len(matching) != 1
        or selected.get("ranking_weight") != expected_weight
        or selected.get("seed") != expected_seed
        or selected.get("offline_eligible") is not offline_eligible
        or selected.get("diagnostic_only") is not (not offline_eligible)
        or selected.get("selected_before_rank4_bank_read") is not True
        or any(
            selected.get(name) != matching[0].get(name)
            for name in (
                "seed", "runtime", "float_checkpoint", "source", "metrics",
                "qat_profile", "qat_profile_contract",
                "qat_execution_evidence",
                "hard_state_density",
                "adaptation_contract",
            )
        )
        or selected.get("search_throughput_profile")
        != plan["search_throughput_profile"]
        or selected.get("active_search_variant_roster")
        != list(plan["search_variants"])
    ):
        raise TeacherTrainingError("selected model disagrees with its training arm")
    return value


def _bank_input(
    bank_path: pathlib.Path, *, phase: str, output_directory: pathlib.Path,
) -> tuple[dict[str, Any], pathlib.Path, dict[str, object]]:
    try:
        bank = challenger.openings.validate_bank(bank_path.resolve())
    except Exception as error:
        raise TeacherTrainingError("Rank-4 screen bank did not validate") from error
    expected_pairs = PILOT_PAIRS if phase == "pilot" else FULL_PAIRS
    openings = bank.get("openings")
    if (
        bank.get("classification") != "unprotected-development"
        or not isinstance(openings, list)
        or len(openings) != expected_pairs
        or bank.get("opening_count") != expected_pairs
        or bank.get("campaign_binding", {}).get("pairs") != expected_pairs
    ):
        raise TeacherTrainingError(
            f"{phase} gate requires exactly {expected_pairs} fresh unprotected pairs"
        )
    payload = (
        "# papersoccer.compact-value-bfm-opening-bank.v1\n"
        "opening_id\ttranscript\n"
        + "".join(
            f"{row['opening_id']}\t{row['transcript']}\n" for row in openings
        )
    ).encode("ascii")
    tsv = _write_content_addressed(output_directory, payload, ".opening-bank.tsv")
    try:
        validated = gate_support.validate_bank(tsv)
    except Exception as error:
        raise TeacherTrainingError("rendered gate TSV did not validate") from error
    if len(validated.get("openings", [])) != expected_pairs:
        raise TeacherTrainingError("rendered gate TSV changed its pair count")
    return bank, tsv, {
        "manifest": _record(bank_path.resolve()),
        "manifest_body_sha256": bank["body_sha256"],
        "gate_tsv": _record(tsv),
        "pairs": expected_pairs,
        "classification": "fresh-unprotected",
    }


def _development_generation_exclusions(
    plan: Mapping[str, Any], *, inputs: trainer.TrainingInputs,
    external_validation: trainer.Dataset,
    extra_state_fingerprints: set[str] | None = None,
    extra_feature_fingerprints: set[str] | None = None,
) -> tuple[set[str], set[str], dict[str, object]]:
    rankings = inputs.successor_rankings
    if rankings is None:
        raise TeacherTrainingError("development-bank generation requires rankings")
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
    exclusions = pipeline._exclusion_context(pipeline_plan)
    state_values = (
        _state_fingerprints_for_rankings(rankings)
        | _phase_game_state_fingerprints(plan)
    )
    feature_values = (
        _dataset_fingerprints(inputs.new)
        | _dataset_fingerprints(inputs.anchor)
        | _dataset_fingerprints(inputs.common_adjudicator)
        | _dataset_fingerprints(inputs.canonical_validation)
        | _dataset_fingerprints(external_validation)
        | _ranking_fingerprints(rankings.train)
        | _ranking_fingerprints(rankings.validation)
    )
    state_values.update(extra_state_fingerprints or ())
    feature_values.update(extra_feature_fingerprints or ())
    for role, values in exclusions["by_role"].items():
        if exclusions["domains"][role] == pipeline.FEATURE_FINGERPRINT_DOMAIN:
            feature_values.update(values)
        else:
            state_values.update(values)
    evidence = {
        "state_fingerprint_count": len(state_values),
        "state_fingerprints_sha256": sha256_bytes(
            canonical_json_bytes(sorted(state_values))
        ),
        "feature_fingerprint_count": len(feature_values),
        "feature_fingerprints_sha256": sha256_bytes(
            canonical_json_bytes(sorted(feature_values))
        ),
        "sources": exclusions["sources"],
        "protected_or_live_data_read_as_fingerprints_only": True,
    }
    return state_values, feature_values, evidence


def _generate_production_development_bank(
    plan: Mapping[str, Any], *, selection_path: pathlib.Path,
    selection_body_sha256: str, purpose: str, count: int,
    output_directory: pathlib.Path, inputs: trainer.TrainingInputs,
    external_validation: trainer.Dataset,
    extra_state_fingerprints: set[str] | None = None,
    extra_feature_fingerprints: set[str] | None = None,
) -> tuple[pathlib.Path, dict[str, object]]:
    """Generate the sole development bank after a sealed model selection."""

    if purpose not in {"pilot-screen", "full-search-ab", "full-qualification"}:
        raise TeacherTrainingError("production development-bank purpose changed")
    state_exclusions, feature_exclusions, exclusion_evidence = (
        _development_generation_exclusions(
            plan, inputs=inputs, external_validation=external_validation,
            extra_state_fingerprints=extra_state_fingerprints,
            extra_feature_fingerprints=extra_feature_fingerprints,
        )
    )
    selection = _record(selection_path.resolve())
    seed_material = canonical_json_bytes({
        "domain": "rank4-teacher-post-selection-development-bank-v1",
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "purpose": purpose,
        "selection_body_sha256": selection_body_sha256,
        "exclusions": exclusion_evidence,
    })
    base_seed = hashlib.sha256(seed_material).digest()
    output_directory.mkdir(parents=True, exist_ok=True)
    claim_path = output_directory / "development-bank-seed-claim.json"
    existing_claim = None
    if claim_path.exists():
        if claim_path.is_symlink() or not claim_path.is_file():
            raise TeacherTrainingError("development-bank seed claim is irregular")
        existing_claim = _load_sealed(
            claim_path, DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
            "development-bank seed claim",
        )
        claimed_at_utc = str(existing_claim.get("claimed_at_utc", ""))
    else:
        claimed_at_utc = utc_now()
    claim_body = {
        "schema": DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "purpose": purpose,
        "plan_body_sha256": plan["body_sha256"],
        "selection": selection,
        "selection_body_sha256": selection_body_sha256,
        "claimed_at_utc": challenger.utc(
            claimed_at_utc, "development-bank seed claim time"
        ),
        "base_seed_256_hex": base_seed.hex(),
        "seed_derivation": "sha256-canonical-post-selection-domain-v1",
        "exclusions": exclusion_evidence,
        "count": count,
        "output_directory": str(output_directory.resolve()),
        "production_execution_authority": plan["execution_authority"],
        "bank_generated": False,
        "one_bank_only": True,
        "protected_tests_opened": False,
    }
    expected_claim = _sealed(claim_body)
    if existing_claim is not None:
        if existing_claim != expected_claim:
            raise TeacherTrainingError("development-bank seed claim changed")
    else:
        _write_sealed(claim_path, claim_body)

    stage = "rank4_teacher_" + purpose.replace("-", "_")
    generated = None
    selected_seed = None
    for ordinal in range(64):
        seed = hashlib.sha256(base_seed + ordinal.to_bytes(4, "big")).digest()
        candidate = challenger.openings.generate_openings(
            stage=stage, count=count, seed=seed,
            excluded_fingerprints=set(state_exclusions),
        )
        candidate_features = {
            _feature_fingerprint(
                features.encode_active(
                    challenger.openings.replay_transcript(row["transcript"])[0]
                )
            )
            for row in candidate
        }
        if (
            len(candidate_features) == count
            and not candidate_features & feature_exclusions
        ):
            generated = candidate
            selected_seed = seed
            break
    if generated is None or selected_seed is None:
        raise TeacherTrainingError(
            "deterministic development-bank generation exhausted feature isolation"
        )
    bank_path = challenger.openings.write_bank(
        output_directory / "opening-bank",
        challenger.openings.bank_payload(
            stage=stage,
            classification="unprotected-development",
            seed=selected_seed,
            exclusions={
                "body_sha256": sha256_bytes(canonical_json_bytes(
                    exclusion_evidence
                )),
                "sources": [{
                    "selection_sha256": selection["sha256"],
                    "seed_claim_sha256": sha256_file(claim_path),
                    "state_fingerprints_sha256": exclusion_evidence[
                        "state_fingerprints_sha256"
                    ],
                    "feature_fingerprints_sha256": exclusion_evidence[
                        "feature_fingerprints_sha256"
                    ],
                }],
            },
            openings=generated,
        ),
    )
    challenger.openings.validate_bank(bank_path)
    return bank_path.resolve(), {
        **_record(claim_path),
        "schema": DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
        "body_sha256": expected_claim["body_sha256"],
    }


def _development_bank_claim(
    plan: Mapping[str, Any], *, selection_path: pathlib.Path,
    selection_schema: str, selection_body_sha256: str,
    bank_path: pathlib.Path, purpose: str, claim_path: pathlib.Path,
    seed_claim: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Claim exactly one post-selection development bank at a fixed route."""

    if purpose not in {"pilot-screen", "full-search-ab", "full-qualification"}:
        raise TeacherTrainingError("development-bank claim purpose is invalid")
    if claim_path.is_symlink() or (
        claim_path.exists() and not claim_path.is_file()
    ):
        raise TeacherTrainingError("development-bank claim route is irregular")
    selection_record = _record(selection_path.resolve())
    if seed_claim is not None:
        if not isinstance(seed_claim, Mapping) or not {
            "path", "bytes", "sha256", "schema", "body_sha256"
        }.issubset(seed_claim):
            raise TeacherTrainingError("development-bank seed claim is malformed")
        seed_path = _validate_record(
            {key: seed_claim[key] for key in ("path", "bytes", "sha256")},
            "development-bank seed claim",
        )
        seed_document = _load_sealed(
            seed_path, DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
            "development-bank seed claim",
        )
        if (
            seed_claim.get("schema") != DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA
            or seed_claim.get("body_sha256") != seed_document["body_sha256"]
        ):
            raise TeacherTrainingError("development-bank seed claim changed")
    try:
        bank_document = challenger.openings.validate_bank(bank_path.resolve())
    except Exception as error:
        raise TeacherTrainingError(
            "development bank failed validation before its one-shot claim"
        ) from error
    bank_record = _record(bank_path.resolve())
    if seed_claim is not None:
        try:
            base_seed = bytes.fromhex(str(seed_document["base_seed_256_hex"]))
            selected_seed = bytes.fromhex(str(bank_document["seed_hex"]))
        except (KeyError, TypeError, ValueError) as error:
            raise TeacherTrainingError(
                "development bank lost deterministic seed ancestry"
            ) from error
        if (
            len(base_seed) != 32
            or len(selected_seed) != 32
            or seed_document.get("campaign_id") != plan["campaign_id"]
            or seed_document.get("attempt") != plan["attempt"]
            or seed_document.get("phase") != plan["phase"]
            or seed_document.get("purpose") != purpose
            or seed_document.get("plan_body_sha256") != plan["body_sha256"]
            or seed_document.get("selection") != selection_record
            or seed_document.get("selection_body_sha256")
            != selection_body_sha256
            or seed_document.get("production_execution_authority")
            != plan["execution_authority"]
            or seed_document.get("count") != bank_document.get("opening_count")
            or seed_document.get("bank_generated") is not False
            or seed_document.get("one_bank_only") is not True
            or seed_document.get("protected_tests_opened") is not False
            or not any(
                hashlib.sha256(
                    base_seed + ordinal.to_bytes(4, "big")
                ).digest() == selected_seed
                for ordinal in range(64)
            )
        ):
            raise TeacherTrainingError(
                "development bank deterministic seed ancestry changed"
            )
    frozen = plan.get("build_source_closure")
    if not isinstance(frozen, Mapping):
        raise TeacherTrainingError("development-bank claim lost build evidence")
    if claim_path.exists():
        existing = _load_sealed(
            claim_path, DEVELOPMENT_BANK_CLAIM_SCHEMA,
            "development-bank claim",
        )
        claimed_at = str(existing.get("claimed_at_utc", ""))
    else:
        claimed_at = utc_now()
    body = {
        "schema": DEVELOPMENT_BANK_CLAIM_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "purpose": purpose,
        "plan_body_sha256": plan["body_sha256"],
        "selection": selection_record,
        "selection_schema": selection_schema,
        "selection_body_sha256": selection_body_sha256,
        "bank": bank_record,
        "seed_claim": None if seed_claim is None else dict(seed_claim),
        "claimed_at_utc": challenger.utc(
            claimed_at, "development-bank claim time"
        ),
        "claim_route": str(claim_path.resolve()),
        "production_execution_authority": plan["execution_authority"],
        "frozen_execution_sources": {
            "manifest": frozen["manifest"],
            "repository_commit": frozen["repository_commit"],
            "closure_sha256": frozen["closure_sha256"],
        },
        "model_selected_before_bank_read": True,
        "one_bank_only": True,
        "retry_with_another_bank_authorized": False,
        "protected_tests_opened": False,
    }
    expected = _sealed(body)
    if claim_path.exists():
        if existing != expected:
            raise TeacherTrainingError(
                "development-bank claim was reused with another bank"
            )
    else:
        _write_sealed(claim_path, body)
    document = _load_sealed(
        claim_path, DEVELOPMENT_BANK_CLAIM_SCHEMA,
        "development-bank claim",
    )
    return {
        **_record(claim_path),
        "schema": DEVELOPMENT_BANK_CLAIM_SCHEMA,
        "body_sha256": document["body_sha256"],
    }


def _validate_development_bank_claim(
    plan: Mapping[str, Any], value: object, *, selection_path: pathlib.Path,
    selection_schema: str, selection_body_sha256: str,
    bank_path: pathlib.Path, purpose: str, claim_path: pathlib.Path,
    seed_claim: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TeacherTrainingError("development-bank claim record is absent")
    if not claim_path.is_file() or claim_path.is_symlink():
        raise TeacherTrainingError("development-bank claim is absent or redirected")
    stored_claim = _load_sealed(
        claim_path, DEVELOPMENT_BANK_CLAIM_SCHEMA, "development-bank claim"
    )
    if seed_claim is None and stored_claim.get("seed_claim") is not None:
        seed_claim = stored_claim["seed_claim"]
    actual = _development_bank_claim(
        plan, selection_path=selection_path,
        selection_schema=selection_schema,
        selection_body_sha256=selection_body_sha256,
        bank_path=bank_path, purpose=purpose, claim_path=claim_path,
        seed_claim=seed_claim,
    )
    if dict(value) != actual:
        raise TeacherTrainingError("development-bank claim binding changed")
    return actual


def _bank_fingerprints(bank: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    state_values: set[str] = set()
    feature_values: set[str] = set()
    for opening in bank["openings"]:
        state, _plies = challenger.openings.replay_transcript(opening["transcript"])
        state_values.add(challenger.openings.state_fingerprints(state)["canonical"])
        feature_values.add(_feature_fingerprint(features.encode_active(state)))
    if len(state_values) != len(bank["openings"]):
        raise TeacherTrainingError("screen bank repeats a canonical state")
    return state_values, feature_values


def _gate_freshness_audit(
    *, plan: Mapping[str, Any], bank: Mapping[str, Any],
    inputs: trainer.TrainingInputs, external_validation: trainer.Dataset,
    phase_game_fingerprints: set[str],
) -> dict[str, object]:
    bank_states, bank_features = _bank_fingerprints(bank)
    rankings = inputs.successor_rankings
    if rankings is None:
        raise TeacherTrainingError("gate freshness requires successor labels")
    training_features = (
        _dataset_fingerprints(inputs.new)
        | _dataset_fingerprints(inputs.anchor)
        | _dataset_fingerprints(inputs.common_adjudicator)
        | _dataset_fingerprints(inputs.canonical_validation)
        | _dataset_fingerprints(external_validation)
        | _ranking_fingerprints(rankings.train)
        | _ranking_fingerprints(rankings.validation)
    )
    training_states = _state_fingerprints_for_rankings(rankings)
    if (
        bank_features & training_features
        or bank_states & training_states
        or bank_states & phase_game_fingerprints
    ):
        raise TeacherTrainingError("screen bank intersects this phase's training data")
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
    exclusions = pipeline._exclusion_context(pipeline_plan)
    intersections: dict[str, int] = {}
    for role, values in exclusions["by_role"].items():
        candidates = (
            bank_features
            if exclusions["domains"][role] == pipeline.FEATURE_FINGERPRINT_DOMAIN
            else bank_states
        )
        intersections[role] = len(candidates & values)
    if any(intersections.values()):
        raise TeacherTrainingError("screen bank intersects a frozen campaign exclusion")
    return {
        "policy": (
            "bank-vs-external-scalar+complete-turn-successors+all-frozen-"
            "feature-or-state-domain-exclusions"
        ),
        "bank_state_fingerprints": len(bank_states),
        "bank_feature_fingerprints": len(bank_features),
        "training_feature_fingerprints": len(training_features),
        "training_state_fingerprints": len(training_states),
        "phase_game_state_fingerprints": len(phase_game_fingerprints),
        "frozen_exclusion_intersections": intersections,
        "protected_or_live_metrics_read": False,
        "passed": True,
    }


def _phase_game_state_fingerprints(plan: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
    games_manifest = _load_sealed(
        pathlib.Path(pipeline_plan["outputs"]["games_manifest"]),
        pipeline.GAME_MANIFEST_SCHEMA,
        "phase games manifest",
    )
    if (
        games_manifest.get("pipeline_body_sha256") != pipeline_plan["body_sha256"]
        or games_manifest.get("attempt") != plan["attempt"]
        or games_manifest.get("phase") != plan["phase"]
    ):
        raise TeacherTrainingError("phase game manifest binding changed")
    for game in games_manifest.get("rows", []):
        if not isinstance(game, Mapping) or not isinstance(game.get("transcript"), str):
            raise TeacherTrainingError("phase game manifest contains a malformed row")
        state = features.ReplayState()
        for action in game["transcript"].split("/"):
            if state.winner is None:
                result.add(challenger.openings.state_fingerprints(state)["canonical"])
            try:
                features.apply_complete_turn(state, state.to_move, action)
            except ValueError as error:
                raise TeacherTrainingError("phase game transcript no longer replays") from error
        if state.winner != game.get("winner"):
            raise TeacherTrainingError("phase game transcript winner changed")
    return result


def _development_state_fingerprints(
    bank: Mapping[str, Any], rankings: trainer.SuccessorRankingLabels,
    phase_game_fingerprints: set[str],
) -> set[str]:
    return (
        set(phase_game_fingerprints)
        | _state_fingerprints_for_rankings(rankings)
        | _bank_fingerprints(bank)[0]
    )


def _compiler_identity(command: str | None = None) -> dict[str, object]:
    requested = command or os.environ.get("CXX", "c++")
    executable = shutil.which(requested)
    if executable is None:
        raise TeacherTrainingError(f"C++ compiler is unavailable: {requested}")
    resolved = pathlib.Path(executable).resolve()
    completed = subprocess.run(
        [str(resolved), "--version"], capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise TeacherTrainingError("C++ compiler identity command failed")
    version = completed.stdout + completed.stderr
    return {
        "requested": requested,
        "executable": _record(resolved),
        "version_sha256": sha256_bytes(version),
        "flags": ["-std=c++20", "-O3"],
    }


Compiler = Callable[[pathlib.Path, pathlib.Path, pathlib.Path], None]


def _compile_gate(
    gate_source: pathlib.Path, candidate_source: pathlib.Path, output: pathlib.Path,
) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise TeacherTrainingError("C++ compiler is unavailable")
    include = (
        f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{candidate_source.resolve()}"'
    )
    completed = subprocess.run(
        [compiler, "-std=c++20", "-O3", include, str(gate_source), "-o", str(output)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise TeacherTrainingError(
            f"Rank-4 gate compilation failed: {completed.stderr}"
        )


def _variant_source(base: bytes, macros: Sequence[str]) -> bytes:
    try:
        decoded = base.decode("ascii")
    except UnicodeDecodeError as error:
        raise TeacherTrainingError("base submission source is not ASCII") from error
    prefix = "".join(f"#define {macro} 1\n" for macro in macros)
    try:
        payload = source_exporter.compact_cpp_code(prefix + decoded).encode("ascii")
    except ValueError as error:
        raise TeacherTrainingError("search-variant source compaction failed") from error
    if not 0 < len(payload) < SOURCE_LIMIT_EXCLUSIVE:
        raise TeacherTrainingError("search-variant source violates 95KB")
    if SOURCE_LIMIT_EXCLUSIVE - len(payload) < SOURCE_RESERVE_TARGET:
        raise TeacherTrainingError("search-variant source misses the reserve target")
    return payload


def _compiled_binary(
    *, source: pathlib.Path, output_directory: pathlib.Path,
    compiler: Compiler, compiler_identity: Mapping[str, object],
) -> tuple[pathlib.Path, dict[str, object]]:
    compile_key = sha256_bytes(canonical_json_bytes({
        "gate_source": _record(GATE_SOURCE),
        "candidate_source": _record(source),
        "rank4_source": _record(RANK4_SOURCE),
        "compiler": dict(compiler_identity),
        "flags": ["-std=c++20", "-O3"],
    }))
    reference_path = output_directory / f"{compile_key}.binary-reference.json"
    if reference_path.exists():
        try:
            reference = json.loads(reference_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise TeacherTrainingError("gate binary reference is unreadable") from error
        binary = _validate_record(reference.get("binary"), "resumed gate binary")
        if reference.get("compile_key") != compile_key:
            raise TeacherTrainingError("resumed gate binary uses another compile binding")
        return binary, reference
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{compile_key}.", dir=output_directory
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    temporary.unlink()
    try:
        compiler(GATE_SOURCE, source, temporary)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise TeacherTrainingError("compiler omitted the Rank-4 gate binary")
        payload = temporary.read_bytes()
        binary = _write_content_addressed(
            output_directory, payload, ".rank4-gate", executable=True
        )
    finally:
        temporary.unlink(missing_ok=True)
    reference = {
        "schema": "papersoccer.compact-value-bfm.rank4-teacher-gate-binary.v1",
        "compile_key": compile_key,
        "candidate_source": _record(source),
        "binary": _record(binary),
        "compiler": dict(compiler_identity),
    }
    _write_once(reference_path, canonical_json_bytes(reference))
    return binary, reference


def _gate_arguments(
    *, binary: pathlib.Path, bank: pathlib.Path, source: pathlib.Path,
    pairs: int, minimum_wins: int, minimum_color_wins: int,
) -> list[str]:
    return [
        str(binary),
        "--bank", str(bank),
        "--expected-bank-sha256", sha256_file(bank),
        "--candidate-source", str(source),
        "--expected-candidate-sha256", sha256_file(source),
        "--rank4-source", str(RANK4_SOURCE),
        "--pair-offset", "0",
        "--pair-count", str(pairs),
        "--mode", "actual-clock",
        "--candidate-c", "0.95",
        "--candidate-fpu", "0.5",
        "--candidate-lambda", "1.0",
        "--candidate-actions", "250",
        "--candidate-root-partial-paths", "4000",
        "--candidate-nonroot-partial-paths", "512",
        "--candidate-nodes", "80000",
        "--candidate-expansions", "2000000",
        "--candidate-seed", "1",
        "--rank4-nodes", "3000000",
        "--max-turns", "320",
        "--minimum-candidate-wins", str(minimum_wins),
        "--minimum-wins-per-color", str(minimum_color_wins),
        "--output", "RESULT_PATH",
    ]


def _search_ab_policy(
    *, phase: str, profile: str, variants: Sequence[str],
    pilot_prior_variant: str | None,
) -> dict[str, object]:
    active = active_search_variants(profile)
    expected = phase_gate_variants(phase, profile)
    treatment_bases = (
        SEARCH_VARIANT_ORDER if phase == "full" else ("baseline",)
    )
    if (
        tuple(variants) != tuple(expected)
        or (
            phase == "pilot" and pilot_prior_variant is not None
        )
        or (
            phase == "full"
            and pilot_prior_variant not in active
        )
    ):
        raise TeacherTrainingError("search A/B policy variant roster is invalid")
    pairs = PILOT_PAIRS if phase == "pilot" else FULL_PAIRS
    return {
        "phase": phase,
        "roster_policy": (
            "single-baseline-model-screen"
            if phase == "pilot" and profile == "standard-v1"
            else "paired-baseline-intervention-screen"
            if phase == "pilot"
            else "complete-full-model-search-ab"
        ),
        "trained_model_is_fixed_across_variants": True,
        "variants": list(variants),
        "variant_macros": {
            name: list(active[name]) for name in variants
        },
        "search_throughput_profile": profile,
        "standard_variant_roster": list(SEARCH_VARIANT_ORDER),
        "phase_standard_variant_roster": [
            name for name in SEARCH_VARIANT_ORDER if name in expected
        ],
        "treatment_pairs": (
            {
                base: _treatment_variant(base, profile)
                for base in treatment_bases
            }
            if profile != "standard-v1"
            else {}
        ),
        "treatment_retention_requirements": {
            "identical_model": True,
            "identical_bank_and_pair_roster": True,
            "zero_failures": True,
            "source_clean": True,
            "parity_clean": True,
            "timing_clean": True,
            "base_control_profile_clean": True,
            "intervention_activated": True,
            "strictly_better_paired_wins": True,
            "strictly_more_candidate_wins": True,
        },
        "baseline": "legacy-feature-sort+legacy-descendant-sort",
        "no-feature-sort-only": "optimized-features+legacy-descendant-sort",
        "single-pass-selection-only": "legacy-features+single-pass-selection",
        "combined": "optimized-features+single-pass-selection",
        "expected_game_volume": _expected_gate_game_volume(
            variants, pairs_per_variant=pairs
        ),
        "full_reruns_complete_active_roster": phase == "full",
        "pilot_uses_complete_active_roster": False,
        "pilot_prior_variant": pilot_prior_variant,
        "pilot_prior_macros": (
            None
            if pilot_prior_variant is None
            else list(active[pilot_prior_variant])
        ),
        "pilot_prior_used_only_for_ties": pilot_prior_variant is not None,
    }


def _gate_execution_policy(variants: Sequence[str]) -> dict[str, object]:
    return {
        "workers": GATE_WORKERS,
        "threads_per_worker": GATE_THREADS_PER_WORKER,
        "whole_bank_processes": True,
        "variants_serial": True,
        "variant_order": list(variants),
        "process_nice": 0,
        "no_competing_rank4_gate_processes": True,
        "prelaunch_claim_required": True,
        "retry_authorized": False,
        "thread_environment": dict(GATE_THREAD_ENVIRONMENT),
    }


@_heavy_stage
def prepare_gate(
    plan_path: pathlib.Path, *, selection_path: pathlib.Path,
    bank_path: pathlib.Path | None = None, resume: bool = False,
    compiler: Compiler | None = None,
    compiler_identity: Mapping[str, object] | None = None,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    _guard_test_hooks(
        plan,
        hooks_used=compiler is not None or compiler_identity is not None,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    selection = load_training_selection(plan, selection_path.resolve())
    if (
        selection.get("model_selection", {}).get("status") not in {
            "model-selected-before-rank4-screen",
            "offline-rejected-before-rank4-screen",
        }
        or not isinstance(selection.get("selected_model"), Mapping)
    ):
        raise TeacherTrainingError("no trained diagnostic model can be screened")
    if sha256_file(RANK4_SOURCE) != gate_support.RANK4_SHA256:
        raise TeacherTrainingError("maintained Rank-4 source identity changed")
    reference_path = pathlib.Path(plan["outputs"]["gate_reference"])
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("gate plan is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=GATE_REFERENCE_SCHEMA,
            receipt_schema=GATE_PLAN_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="gate plan",
        )
        validated = load_gate_plan(plan, receipt_path)
        if (
            validated.get("selection") != _record(selection_path.resolve())
            or (
                bank_path is not None
                and validated.get("bank", {}).get("manifest")
                != _record(bank_path.resolve())
            )
            or receipt != validated
        ):
            raise TeacherTrainingError("resumed gate inputs differ from the frozen plan")
        return receipt_path

    bundle, inputs, external_validation = training_context(plan)
    del bundle
    purpose = "pilot-screen" if plan["phase"] == "pilot" else "full-search-ab"
    production = plan["execution_authority"]["production_allowlist_enforced"]
    seed_claim = None
    if production:
        generated_directory = pathlib.Path(plan["outputs"]["gate_banks"]) / "generated"
        seed_claim_path = generated_directory / "development-bank-seed-claim.json"
        if bank_path is not None and not seed_claim_path.exists():
            raise TeacherTrainingError(
                "production development bank must be generated after selection"
            )
        generated_bank, seed_claim = _generate_production_development_bank(
            plan,
            selection_path=selection_path.resolve(),
            selection_body_sha256=selection["body_sha256"],
            purpose=purpose,
            count=(PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS),
            output_directory=generated_directory,
            inputs=inputs,
            external_validation=external_validation,
        )
        if bank_path is not None and bank_path.resolve() != generated_bank:
            raise TeacherTrainingError(
                "supplied production bank differs from deterministic generation"
            )
        effective_bank_path = generated_bank
    else:
        if bank_path is None:
            raise TeacherTrainingError("nonproduction gate requires an explicit test bank")
        effective_bank_path = bank_path.resolve()
    bank_claim_path = (
        pathlib.Path(plan["outputs"]["root"]) / "gates/development-bank-claim.json"
    )
    bank_claim = _development_bank_claim(
        plan,
        selection_path=selection_path.resolve(),
        selection_schema=SELECTION_SCHEMA,
        selection_body_sha256=selection["body_sha256"],
        bank_path=effective_bank_path,
        purpose=purpose,
        claim_path=bank_claim_path,
        seed_claim=seed_claim,
    )
    bank, gate_bank, bank_record = _bank_input(
        effective_bank_path,
        phase=str(plan["phase"]),
        output_directory=pathlib.Path(plan["outputs"]["gate_banks"]),
    )
    phase_game_fingerprints = _phase_game_state_fingerprints(plan)
    freshness = _gate_freshness_audit(
        plan=plan,
        bank=bank,
        inputs=inputs,
        external_validation=external_validation,
        phase_game_fingerprints=phase_game_fingerprints,
    )
    rankings = inputs.successor_rankings
    assert rankings is not None
    development_fingerprints = sorted(
        _development_state_fingerprints(
            bank, rankings, phase_game_fingerprints
        )
    )
    development_body = {
        "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "classification": "unprotected-development-fingerprints",
        "canonicalization": "minimum-sha256-over-exact+rotate+reflect+rotate-reflect",
        "fingerprints": development_fingerprints,
        "fingerprint_count": len(development_fingerprints),
        "includes_phase_games": True,
        "includes_teacher_successors": True,
        "includes_rank4_gate_bank": True,
        "protected_or_live_data_included": False,
    }
    development_document = _sealed(development_body)
    development_payload = canonical_json_bytes(development_document)
    development_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["development_fingerprints"]),
        development_payload,
        ".development-fingerprints.json",
    )

    selected = selection["selected_model"]
    runtime = _validate_record(selected["runtime"], "selected model runtime")
    base_source = _validate_record(
        {key: selected["source"][key] for key in ("path", "bytes", "sha256")},
        "selected model source",
    )
    runtime_document, _runtime_payload, runtime_metadata = (
        model_exporter.validate_runtime(runtime)
    )
    if runtime_document.get("architecture", {}).get("dimensions") != [6301, 12, 8, 1]:
        raise TeacherTrainingError("screened runtime changed architecture")
    search_throughput_profile = str(plan["search_throughput_profile"])
    active_variants = phase_gate_variants(
        str(plan["phase"]), search_throughput_profile
    )
    pilot_prior_variant = None
    if plan["phase"] == "full":
        pilot_prior_variant = plan["pilot_admission"]["search_variant"]
        if (
            pilot_prior_variant not in active_variants
            or plan["pilot_admission"].get("compile_time_macros")
            != list(active_variants[pilot_prior_variant])
            or plan["pilot_admission"].get("search_throughput_profile")
            != search_throughput_profile
        ):
            raise TeacherTrainingError(
                "full gate changed its pilot search prior"
            )
    variants = tuple(active_variants)

    identity = dict(compiler_identity or _compiler_identity())
    compile_function = compiler or _compile_gate
    pairs = PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS
    minimum_wins = (
        PILOT_MINIMUM_WINS if plan["phase"] == "pilot" else -1
    )
    minimum_color_wins = -1
    requests = []
    for variant in variants:
        macros = active_variants[variant]
        variant_metadata = _search_variant_metadata(
            search_throughput_profile, variant
        )
        source_payload = _variant_source(base_source.read_bytes(), macros)
        source = _write_content_addressed(
            pathlib.Path(plan["outputs"]["gate_sources"]),
            source_payload,
            f".{variant}.submission.cpp",
        )
        binary, binary_reference = _compiled_binary(
            source=source,
            output_directory=pathlib.Path(plan["outputs"]["gate_binaries"]),
            compiler=compile_function,
            compiler_identity=identity,
        )
        request_body = {
            "schema": SCREEN_REQUEST_SCHEMA,
            "campaign_id": plan["campaign_id"],
            "attempt": plan["attempt"],
            "phase": plan["phase"],
            "gate_purpose": (
                "pilot-screen" if plan["phase"] == "pilot" else "full-search-ab"
            ),
            "plan_body_sha256": plan["body_sha256"],
            "training_selection": _record(selection_path.resolve()),
            "training_selection_body_sha256": selection["body_sha256"],
            "development_bank_claim": bank_claim,
            "execution_authority": plan["execution_authority"],
            "model_selected_before_bank_read": True,
            "ranking_weight": selected["ranking_weight"],
            "seed": selected["seed"],
            "runtime": _record(runtime),
            "runtime_body_sha256": runtime_metadata["body_sha256"],
            "runtime_payload_sha256": runtime_document["quantization"][
                "payload_sha256"
            ],
            "search_throughput_profile": search_throughput_profile,
            "search_variant": variant,
            "search_variant_metadata": variant_metadata,
            "compile_time_macros": list(macros),
            "macros_embedded_at_source_start": True,
            "candidate_source": _record(source, ascii_required=True),
            "base_model_source": _record(base_source, ascii_required=True),
            "source_is_default_for_variant": True,
            "source_reserve": SOURCE_LIMIT_EXCLUSIVE - source.stat().st_size,
            "compiler": identity,
            "binary": _record(binary),
            "binary_reference": binary_reference,
            "rank4_source": _record(RANK4_SOURCE),
            "bank": bank_record,
            "freshness_audit": freshness,
            "configuration": {
                **GATE_CONFIGURATION,
                "candidate_search_profile": variant_metadata[
                    "candidate_search_profile"
                ],
                "pair_count": pairs,
                "minimum_candidate_wins": minimum_wins,
                "minimum_wins_per_color": minimum_color_wins,
            },
            "argv": _gate_arguments(
                binary=binary,
                bank=gate_bank,
                source=source,
                pairs=pairs,
                minimum_wins=minimum_wins,
                minimum_color_wins=minimum_color_wins,
            ),
            "protected_tests_opened": False,
        }
        request_document = _sealed(request_body)
        request_payload = canonical_json_bytes(request_document)
        request_path = _write_content_addressed(
            pathlib.Path(plan["outputs"]["gate_requests"]),
            request_payload,
            f".{variant}.screen-request.json",
        )
        requests.append({
            "variant": variant,
            "macros": list(macros),
            "search_variant_metadata": variant_metadata,
            "request": _record(request_path),
            "request_body_sha256": request_document["body_sha256"],
            "source": _record(source, ascii_required=True),
            "binary": _record(binary),
        })
    body = {
        "schema": GATE_PLAN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "plan_body_sha256": plan["body_sha256"],
        "selection": _record(selection_path.resolve()),
        "selection_body_sha256": selection["body_sha256"],
        "development_bank_claim": bank_claim,
        "execution_authority": plan["execution_authority"],
        "model_selected_before_bank_read": True,
        "ranking_weight": selected["ranking_weight"],
        "runtime": _record(runtime),
        "bank": bank_record,
        "freshness_audit": freshness,
        "search_throughput_profile": search_throughput_profile,
        "active_search_variant_roster": list(variants),
        "expected_game_volume": _expected_gate_game_volume(
            variants, pairs_per_variant=pairs
        ),
        "development_exclusion": {
            **_record(development_path),
            "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "body_sha256": development_document["body_sha256"],
        },
        "search_ab_policy": _search_ab_policy(
            phase=str(plan["phase"]),
            profile=search_throughput_profile,
            variants=variants,
            pilot_prior_variant=pilot_prior_variant,
        ),
        "execution_policy": _gate_execution_policy(variants),
        "execution_outputs": {
            "root": plan["outputs"]["gate_executions"],
            "reference": plan["outputs"]["gate_execution_reference"],
        },
        "requests": requests,
        "protected_tests_opened": False,
    }
    document = _sealed(body)
    payload = canonical_json_bytes(document)
    receipt_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["gate_plans"]),
        payload,
        ".gate-plan.json",
    )
    _write_sealed(
        reference_path,
        {
            "schema": GATE_REFERENCE_SCHEMA,
            "plan_body_sha256": plan["body_sha256"],
            "receipt": _record(receipt_path),
            "receipt_body_sha256": document["body_sha256"],
        },
    )
    return receipt_path.resolve()


def load_gate_plan(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    value = _load_sealed(path.resolve(), GATE_PLAN_SCHEMA, "teacher gate plan")
    search_throughput_profile = _search_throughput_profile(
        plan.get("adaptation_contract")
    )
    active_variants = phase_gate_variants(
        str(plan["phase"]), search_throughput_profile
    )
    pilot_prior_variant = (
        None
        if plan["phase"] == "pilot"
        else plan["pilot_admission"]["search_variant"]
    )
    expected_variants = list(active_variants)
    requests = value.get("requests")
    if (
        value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("model_selected_before_bank_read") is not True
        or value.get("execution_authority") != plan["execution_authority"]
        or value.get("protected_tests_opened") is not False
        or value.get("search_throughput_profile")
        != search_throughput_profile
        or value.get("active_search_variant_roster") != expected_variants
        or value.get("expected_game_volume")
        != _expected_gate_game_volume(
            expected_variants,
            pairs_per_variant=(
                PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS
            ),
        )
        or value.get("search_ab_policy")
        != _search_ab_policy(
            phase=str(plan["phase"]),
            profile=search_throughput_profile,
            variants=expected_variants,
            pilot_prior_variant=pilot_prior_variant,
        )
        or value.get("execution_policy")
        != _gate_execution_policy(expected_variants)
        or value.get("execution_outputs") != {
            "root": plan["outputs"]["gate_executions"],
            "reference": plan["outputs"]["gate_execution_reference"],
        }
        or not isinstance(requests, list)
        or [request.get("variant") for request in requests] != expected_variants
    ):
        raise TeacherTrainingError("teacher gate plan binding changed")
    _validate_record(value.get("selection"), "gate-plan training selection")
    _validate_record(value.get("runtime"), "gate-plan runtime")
    bank = value.get("bank")
    if not isinstance(bank, Mapping):
        raise TeacherTrainingError("gate-plan bank binding is absent")
    _validate_record(bank.get("manifest"), "gate bank manifest")
    _validate_record(bank.get("gate_tsv"), "gate bank TSV")
    purpose = "pilot-screen" if plan["phase"] == "pilot" else "full-search-ab"
    bank_claim_path = (
        pathlib.Path(plan["outputs"]["root"]) / "gates/development-bank-claim.json"
    )
    _validate_development_bank_claim(
        plan, value.get("development_bank_claim"),
        selection_path=_validate_record(value["selection"], "gate selection"),
        selection_schema=SELECTION_SCHEMA,
        selection_body_sha256=str(value["selection_body_sha256"]),
        bank_path=_validate_record(bank["manifest"], "gate bank manifest"),
        purpose=purpose, claim_path=bank_claim_path,
    )
    exclusion = value.get("development_exclusion")
    if not isinstance(exclusion, Mapping):
        raise TeacherTrainingError("gate-plan development exclusion is absent")
    exclusion_path = _validate_record(
        {key: exclusion[key] for key in ("path", "bytes", "sha256")},
        "development exclusion",
    )
    exclusion_value = _load_sealed(
        exclusion_path,
        challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
        "development exclusion",
    )
    if exclusion.get("body_sha256") != exclusion_value["body_sha256"]:
        raise TeacherTrainingError("development exclusion body changed")
    for request_record in requests:
        variant = request_record["variant"]
        metadata = _search_variant_metadata(search_throughput_profile, variant)
        if (
            request_record.get("macros") != list(active_variants[variant])
            or request_record.get("search_variant_metadata") != metadata
        ):
            raise TeacherTrainingError("gate request macro roster changed")
        request_path = _validate_record(
            request_record.get("request"), f"{variant} gate request"
        )
        request = _load_sealed(
            request_path, SCREEN_REQUEST_SCHEMA, f"{variant} gate request"
        )
        if (
            request.get("plan_body_sha256") != plan["body_sha256"]
            or request.get("gate_purpose") != (
                "pilot-screen" if plan["phase"] == "pilot" else "full-search-ab"
            )
            or request.get("search_throughput_profile")
            != search_throughput_profile
            or request.get("search_variant") != variant
            or request.get("search_variant_metadata") != metadata
            or request.get("configuration", {}).get(
                "candidate_search_profile"
            ) != metadata["candidate_search_profile"]
            or request.get("compile_time_macros")
            != list(active_variants[variant])
            or request.get("model_selected_before_bank_read") is not True
            or request.get("source_is_default_for_variant") is not True
            or request.get("protected_tests_opened") is not False
            or request_record.get("request_body_sha256")
            != request["body_sha256"]
            or request_record.get("source") != request.get("candidate_source")
            or request_record.get("binary") != request.get("binary")
            or request.get("training_selection") != value.get("selection")
            or request.get("training_selection_body_sha256")
            != value.get("selection_body_sha256")
            or request.get("development_bank_claim")
            != value.get("development_bank_claim")
            or request.get("execution_authority")
            != value.get("execution_authority")
            or request.get("runtime") != value.get("runtime")
            or request.get("bank") != value.get("bank")
            or request.get("configuration")
            != _expected_gate_configuration(
                str(plan["phase"]),
                candidate_search_profile=str(
                    metadata["candidate_search_profile"]
                ),
            )
            or request.get("rank4_source", {}).get("sha256")
            != gate_support.RANK4_SHA256
        ):
            raise TeacherTrainingError(f"{variant} gate request binding changed")
        candidate_source = _validate_record(
            request.get("candidate_source"), f"{variant} source"
        )
        base_source = _validate_record(
            request.get("base_model_source"), f"{variant} base source"
        )
        if candidate_source.read_bytes() != _variant_source(
            base_source.read_bytes(), active_variants[variant]
        ):
            raise TeacherTrainingError(f"{variant} source does not encode its macros")
        _validate_record(request.get("binary"), f"{variant} gate binary")
    return value


def _expected_gate_configuration(
    phase: str, *, candidate_search_profile: str,
    qualification: bool = False,
) -> dict[str, object]:
    pairs = PILOT_PAIRS if phase == "pilot" else FULL_PAIRS
    return {
        **GATE_CONFIGURATION,
        "candidate_search_profile": candidate_search_profile,
        "pair_count": pairs,
        "minimum_candidate_wins": (
            FULL_MINIMUM_WINS
            if qualification
            else PILOT_MINIMUM_WINS
            if phase == "pilot"
            else -1
        ),
        "minimum_wins_per_color": (
            FULL_MINIMUM_COLOR_WINS if qualification else -1
        ),
    }


def _validate_gate_result(
    request: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    expected_bank = request["bank"]["gate_tsv"]["sha256"]
    expected_source = request["candidate_source"]["sha256"]
    expected_profile = str(request["search_variant_metadata"][
        "candidate_search_profile"
    ])
    try:
        document = gate_support.validate_result(
            path.resolve(),
            expected_bank_sha256=expected_bank,
            expected_candidate_sha256=expected_source,
            expected_candidate_search_profile=expected_profile,
        )
    except Exception as error:
        raise TeacherTrainingError(
            f"{request['search_variant']} Rank-4 result did not validate"
        ) from error
    expected = _expected_gate_configuration(
        str(request["phase"]),
        candidate_search_profile=expected_profile,
        qualification=request.get("gate_purpose") == "full-qualification",
    )
    configuration = document.get("config")
    if not isinstance(configuration, Mapping) or any(
        configuration.get(key) != value for key, value in expected.items()
    ):
        raise TeacherTrainingError("Rank-4 result used a different gate configuration")
    bindings = document.get("bindings", {})
    if (
        bindings.get("candidate_source_sha256") != expected_source
        or bindings.get("bank_sha256") != expected_bank
        or bindings.get("rank4_source_sha256") != gate_support.RANK4_SHA256
        or bindings.get("candidate_runtime_body_sha256")
        != request["runtime_body_sha256"]
        or bindings.get("candidate_payload_sha256")
        != request["runtime_payload_sha256"]
    ):
        raise TeacherTrainingError("Rank-4 result runtime/source/bank binding changed")
    return document


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc_instant(value: object, label: str) -> dt.datetime:
    text = challenger.utc(value, label)
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def _gate_command(request: Mapping[str, Any], output: pathlib.Path) -> list[str]:
    argv = request.get("argv")
    if (
        not isinstance(argv, list)
        or any(not isinstance(item, str) or not item for item in argv)
        or argv.count("RESULT_PATH") != 1
    ):
        raise TeacherTrainingError("gate request argv is malformed")
    return [str(output) if item == "RESULT_PATH" else item for item in argv]


ProcessAuditor = Callable[[pathlib.Path], Mapping[str, Any]]
GateProcessRunner = Callable[
    [Sequence[str], pathlib.Path, Mapping[str, str], pathlib.Path],
    Mapping[str, Any],
]


def _default_gate_process_audit(binary: pathlib.Path) -> Mapping[str, Any]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,nice=,command="],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise TeacherTrainingError("cannot audit competing Rank-4 gate processes")
    matches = []
    for line in completed.stdout.splitlines():
        pieces = line.strip().split(None, 2)
        if len(pieces) != 3:
            continue
        pid_text, nice_text, command = pieces
        executable = command.split(None, 1)[0]
        name = pathlib.Path(executable).name
        if (
            name.endswith(".rank4-gate")
            or "compact_value_bfm_rank4_gate" in name
            or pathlib.Path(executable).resolve() == binary.resolve()
        ):
            matches.append({
                "pid": int(pid_text),
                "nice": int(nice_text),
                "executable": executable,
            })
    try:
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError) as error:
        raise TeacherTrainingError("cannot verify development-gate process nice") from error
    return {
        "process_nice": current_nice,
        "competing_rank4_gate_processes": matches,
        "ps_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "ps_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }


def _default_gate_process_runner(
    command: Sequence[str], repository: pathlib.Path,
    environment: Mapping[str, str], output: pathlib.Path,
) -> Mapping[str, Any]:
    del output
    process = subprocess.Popen(
        list(command),
        cwd=repository,
        env=dict(environment),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        process_nice = os.getpriority(os.PRIO_PROCESS, process.pid)
        if process_nice != 0:
            process.terminate()
            process.communicate(timeout=30)
            raise TeacherTrainingError("development-gate child process nice is not zero")
        _stdout, stderr = process.communicate(timeout=GATE_PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise TeacherTrainingError("development-gate process timed out") from error
    return {
        "pid": process.pid,
        "process_nice": process_nice,
        "returncode": process.returncode,
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha256_bytes(stderr),
        "stdout_policy": "discarded-duplicate-of-raw-result",
    }


def _validate_prelaunch_audit(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "process_nice", "competing_rank4_gate_processes",
            "ps_stdout_sha256", "ps_stderr_sha256",
        }
        or value.get("process_nice") != 0
        or value.get("competing_rank4_gate_processes") != []
        or SHA256_RE.fullmatch(str(value.get("ps_stdout_sha256"))) is None
        or SHA256_RE.fullmatch(str(value.get("ps_stderr_sha256"))) is None
    ):
        raise TeacherTrainingError("development-gate prelaunch audit is not clean")
    return dict(value)


def _variant_execution_directory(
    request_plan: Mapping[str, Any], variant: str,
) -> pathlib.Path:
    return pathlib.Path(request_plan["execution_outputs"]["root"]) / variant


def _validate_variant_execution(
    path: pathlib.Path, *, plan: Mapping[str, Any],
    gate_plan: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_sealed(
        path.resolve(), GATE_VARIANT_EXECUTION_SCHEMA,
        "development-gate variant execution",
    )
    variant = str(request["search_variant"])
    directory = _variant_execution_directory(gate_plan, variant).resolve()
    if path.resolve() != directory / "receipt.json":
        raise TeacherTrainingError("development-gate receipt route changed")
    claim_path = _record_subset(value.get("claim"), "development-gate claim")
    claim = _load_sealed(
        claim_path, GATE_EXECUTION_CLAIM_SCHEMA, "development-gate claim"
    )
    raw_path = _record_subset(value.get("raw_result"), "development-gate raw result")
    request_path = _validate_record(
        next(item for item in gate_plan["requests"] if item["variant"] == variant)[
            "request"
        ],
        f"{variant} execution request",
    )
    if (
        set(value) != {
            "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
            "gate_plan_body_sha256", "variant", "claim", "request",
            "raw_result", "profile_activation", "execution", "status",
            "retry_authorized", "execution_authority", "body_sha256",
        }
        or value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("gate_plan_body_sha256") != gate_plan["body_sha256"]
        or value.get("execution_authority") != plan["execution_authority"]
        or value.get("variant") != variant
        or value.get("request") != _record(request_path)
        or claim_path != directory / "claim.json"
        or raw_path != directory / "raw-result.json"
    ):
        raise TeacherTrainingError("development-gate variant binding changed")
    document = _validate_gate_result(request, raw_path)
    expected_activation = _profile_activation_evidence(
        document,
        expected_profile=str(
            request["search_variant_metadata"]["candidate_search_profile"]
        ),
    )
    command = _gate_command(request, raw_path)
    execution = value.get("execution")
    if (
        set(claim) != {
            "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
            "gate_plan", "gate_plan_body_sha256", "variant", "request",
            "request_body_sha256", "claimed_at_utc", "worker", "prelaunch_audit",
            "no_retry", "execution_authority", "body_sha256",
        }
        or claim.get("campaign_id") != plan["campaign_id"]
        or claim.get("attempt") != plan["attempt"]
        or claim.get("phase") != plan["phase"]
        or claim.get("plan_body_sha256") != plan["body_sha256"]
        or claim.get("gate_plan") != _record(pathlib.Path(gate_plan["path"]))
        or claim.get("gate_plan_body_sha256") != gate_plan["body_sha256"]
        or claim.get("variant") != variant
        or claim.get("request") != _record(
            request_path
        )
        or claim.get("request_body_sha256") != request["body_sha256"]
        or claim.get("execution_authority") != plan["execution_authority"]
        or claim.get("worker") != {
            "workers": 1,
            "threads_per_worker": 1,
            "whole_bank_process": True,
            "process_nice": 0,
            "thread_environment": GATE_THREAD_ENVIRONMENT,
        }
        or claim.get("no_retry") is not True
    ):
        raise TeacherTrainingError("development-gate claim binding changed")
    _validate_prelaunch_audit(claim.get("prelaunch_audit"))
    claimed = _utc_instant(claim.get("claimed_at_utc"), "gate claim time")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != {
            "launched_at_utc", "finished_at_utc", "command", "environment",
            "workers", "threads_per_worker", "whole_bank_process",
            "variants_serial", "process_nice", "pid", "returncode",
            "stderr_bytes", "stderr_sha256", "stdout_policy",
        }
        or execution.get("command") != command
        or execution.get("environment") != GATE_THREAD_ENVIRONMENT
        or execution.get("workers") != 1
        or execution.get("threads_per_worker") != 1
        or execution.get("whole_bank_process") is not True
        or execution.get("variants_serial") is not True
        or execution.get("process_nice") != 0
        or isinstance(execution.get("pid"), bool)
        or not isinstance(execution.get("pid"), int)
        or execution["pid"] <= 0
        or execution.get("returncode")
        != (0 if document["result"]["passed"] is True else 2)
        or execution.get("stderr_bytes") != 0
        or execution.get("stderr_sha256") != sha256_bytes(b"")
        or execution.get("stdout_policy")
        != "discarded-duplicate-of-raw-result"
        or value.get("profile_activation") != expected_activation
        or value.get("retry_authorized") is not False
        or value.get("status") != "complete-no-retry"
    ):
        raise TeacherTrainingError("development-gate execution evidence changed")
    launched = _utc_instant(execution.get("launched_at_utc"), "gate launch time")
    finished = _utc_instant(execution.get("finished_at_utc"), "gate finish time")
    if not claimed <= launched <= finished:
        raise TeacherTrainingError("development-gate chronology changed")
    return value


def _load_gate_request_plan(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_bytes()).get("schema")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
        raise TeacherTrainingError("gate request plan is unreadable") from error
    if schema == GATE_PLAN_SCHEMA:
        return load_gate_plan(plan, path)
    if schema == FULL_QUALIFICATION_PLAN_SCHEMA:
        return load_full_qualification_plan(plan, path)
    raise TeacherTrainingError("gate request plan schema is unsupported")


def load_gate_execution(
    plan: Mapping[str, Any], gate_plan_path: pathlib.Path,
    path: pathlib.Path,
) -> dict[str, Any]:
    gate_plan = _load_gate_request_plan(plan, gate_plan_path.resolve())
    gate_plan_with_path = {**gate_plan, "path": str(gate_plan_path.resolve())}
    value = _load_sealed(
        path.resolve(), GATE_EXECUTION_SCHEMA, "development-gate execution"
    )
    expected_variants = gate_plan["active_search_variant_roster"]
    receipts = value.get("variant_receipts")
    if (
        set(value) != {
            "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
            "gate_plan", "gate_plan_body_sha256", "execution_policy",
            "variant_order", "variant_receipts", "status", "retry_authorized",
            "execution_authority", "body_sha256",
        }
        or value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("gate_plan") != _record(gate_plan_path.resolve())
        or value.get("gate_plan_body_sha256") != gate_plan["body_sha256"]
        or value.get("execution_authority") != plan["execution_authority"]
        or value.get("execution_policy") != gate_plan["execution_policy"]
        or value.get("variant_order") != expected_variants
        or not isinstance(receipts, Mapping)
        or set(receipts) != set(expected_variants)
        or value.get("status") != "complete-serial-one-worker-no-retry"
        or value.get("retry_authorized") is not False
    ):
        raise TeacherTrainingError("development-gate execution binding changed")
    previous_finished: dt.datetime | None = None
    plan_ready = (
        _utc_instant(
            gate_plan["prepared_at_utc"], "qualification plan preparation time"
        )
        if gate_plan.get("schema") == FULL_QUALIFICATION_PLAN_SCHEMA
        else None
    )
    for variant in expected_variants:
        receipt_path = _record_subset(
            receipts[variant], f"{variant} execution receipt"
        )
        receipt = _validate_variant_execution(
            receipt_path,
            plan=plan,
            gate_plan=gate_plan_with_path,
            request=_load_sealed(
                _validate_record(
                    next(item for item in gate_plan["requests"] if item["variant"] == variant)[
                        "request"
                    ],
                    f"{variant} request",
                ),
                SCREEN_REQUEST_SCHEMA,
                f"{variant} request",
            ),
        )
        claim = _load_sealed(
            _record_subset(receipt["claim"], f"{variant} claim"),
            GATE_EXECUTION_CLAIM_SCHEMA,
            f"{variant} claim",
        )
        claimed = _utc_instant(claim["claimed_at_utc"], f"{variant} claim time")
        finished = _utc_instant(
            receipt["execution"]["finished_at_utc"], f"{variant} finish time"
        )
        if previous_finished is not None and claimed < previous_finished:
            raise TeacherTrainingError("development-gate variants overlapped")
        if plan_ready is not None and claimed < plan_ready:
            raise TeacherTrainingError("qualification execution predates its plan")
        previous_finished = finished
    return value


def _optional_artifact_record(
    path: pathlib.Path, *, label: str,
) -> dict[str, object] | None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise TeacherTrainingError(f"{label} is redirected or irregular")
    return _record(path) if path.is_file() else None


def _validate_abandoned_gate_claim(
    path: pathlib.Path, *, plan: Mapping[str, Any],
    gate_plan_path: pathlib.Path, gate_plan: Mapping[str, Any],
    request_path: pathlib.Path, request: Mapping[str, Any], variant: str,
) -> dict[str, Any]:
    claim = _load_sealed(
        path, GATE_EXECUTION_CLAIM_SCHEMA, "abandoned development-gate claim"
    )
    expected_worker = {
        "workers": 1,
        "threads_per_worker": 1,
        "whole_bank_process": True,
        "process_nice": 0,
        "thread_environment": GATE_THREAD_ENVIRONMENT,
    }
    if (
        path.resolve()
        != (_variant_execution_directory(gate_plan, variant) / "claim.json").resolve()
        or set(claim) != {
            "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
            "gate_plan", "gate_plan_body_sha256", "variant", "request",
            "request_body_sha256", "claimed_at_utc", "worker",
            "prelaunch_audit", "no_retry", "execution_authority",
            "body_sha256",
        }
        or claim.get("campaign_id") != plan["campaign_id"]
        or claim.get("attempt") != plan["attempt"]
        or claim.get("phase") != plan["phase"]
        or claim.get("plan_body_sha256") != plan["body_sha256"]
        or claim.get("gate_plan") != _record(gate_plan_path.resolve())
        or claim.get("gate_plan_body_sha256") != gate_plan["body_sha256"]
        or claim.get("variant") != variant
        or claim.get("request") != _record(request_path)
        or claim.get("request_body_sha256") != request["body_sha256"]
        or claim.get("worker") != expected_worker
        or claim.get("execution_authority") != plan["execution_authority"]
        or claim.get("no_retry") is not True
    ):
        raise TeacherTrainingError("abandoned development-gate claim changed")
    _validate_prelaunch_audit(claim.get("prelaunch_audit"))
    _utc_instant(claim.get("claimed_at_utc"), "abandoned gate claim time")
    return claim


def _abandonment_request(
    gate_plan: Mapping[str, Any], variant: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    records = [
        item for item in gate_plan["requests"]
        if item.get("variant") == variant
    ]
    if len(records) != 1:
        raise TeacherTrainingError("abandoned gate variant request is ambiguous")
    request_path = _validate_record(
        records[0].get("request"), f"{variant} abandoned gate request"
    )
    request = _load_sealed(
        request_path, SCREEN_REQUEST_SCHEMA, f"{variant} abandoned gate request"
    )
    return request_path, request


def validate_gate_abandonment(
    plan: Mapping[str, Any], gate_plan_path: pathlib.Path,
    path: pathlib.Path,
) -> dict[str, Any]:
    """Validate a terminal, metric-free receipt for one spent development bank."""

    gate_plan_path = gate_plan_path.resolve()
    gate_plan = _load_gate_request_plan(plan, gate_plan_path)
    value = _load_sealed(
        path.resolve(), GATE_ABANDONMENT_SCHEMA,
        "development-gate abandonment",
    )
    variant = str(value.get("variant", ""))
    if variant not in gate_plan["active_search_variant_roster"]:
        raise TeacherTrainingError("abandoned development-gate variant changed")
    directory = _variant_execution_directory(gate_plan, variant).resolve()
    if path.resolve() != directory / "aborted.json":
        raise TeacherTrainingError("development-gate abandonment route changed")
    request_path, request = _abandonment_request(gate_plan, variant)
    claim_path = directory / "claim.json"
    raw_path = directory / "raw-result.json"
    receipt_path = directory / "receipt.json"
    claim_record = value.get("claim")
    if claim_path.is_file():
        claim = _validate_abandoned_gate_claim(
            claim_path, plan=plan, gate_plan_path=gate_plan_path,
            gate_plan=gate_plan, request_path=request_path,
            request=request, variant=variant,
        )
        expected_claim = _record(claim_path)
    elif claim_path.exists() or claim_path.is_symlink():
        raise TeacherTrainingError("abandoned development-gate claim is irregular")
    else:
        claim = None
        expected_claim = None
    expected_raw = _optional_artifact_record(
        raw_path, label="abandoned development-gate raw result"
    )
    valid_receipt = False
    if receipt_path.is_file():
        try:
            _validate_variant_execution(
                receipt_path,
                plan=plan,
                gate_plan={**gate_plan, "path": str(gate_plan_path)},
                request=request,
            )
        except (TeacherTrainingError, OSError, ValueError):
            pass
        else:
            valid_receipt = True
    elif receipt_path.exists() or receipt_path.is_symlink():
        raise TeacherTrainingError("abandoned development-gate receipt is irregular")
    if valid_receipt:
        raise TeacherTrainingError(
            "completed development-gate execution cannot be abandoned"
        )
    expected_invalid_receipt = _optional_artifact_record(
        receipt_path, label="invalid development-gate receipt"
    )
    if expected_claim is None and expected_raw is None:
        raise TeacherTrainingError(
            "development-gate abandonment requires a spent claim or raw result"
        )
    development = gate_plan.get("development_exclusion")
    if not isinstance(development, Mapping):
        raise TeacherTrainingError("abandoned gate lost its development exclusion")
    development_path = _record_subset(
        development, "abandoned development exclusion"
    )
    development_value = _load_sealed(
        development_path, challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
        "abandoned development exclusion",
    )
    expected_candidate = {
        "runtime": dict(request["runtime"]),
        "source": dict(request["candidate_source"]),
    }
    expected_fields = {
        "schema", "campaign_id", "attempt", "phase", "plan_body_sha256",
        "training_plan", "gate_plan", "gate_plan_body_sha256", "variant", "request",
        "claim", "raw_result", "invalid_receipt", "bank", "candidate",
        "development_exclusion", "execution_authority",
        "build_source_closure", "abandonment_quiescence_audit",
        "status", "failure_class",
        "partial_metrics_read", "improvement_counted", "retry_authorized",
        "abandoned_at_utc", "body_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("training_plan")
        != _record(pathlib.Path(plan["outputs"]["plan"]))
        or value.get("gate_plan") != _record(gate_plan_path)
        or value.get("gate_plan_body_sha256") != gate_plan["body_sha256"]
        or value.get("request") != _record(request_path)
        or claim_record != expected_claim
        or value.get("raw_result") != expected_raw
        or value.get("invalid_receipt") != expected_invalid_receipt
        or value.get("bank") != gate_plan.get("bank")
        or value.get("candidate") != expected_candidate
        or value.get("development_exclusion") != development
        or development.get("body_sha256") != development_value["body_sha256"]
        or value.get("execution_authority") != plan["execution_authority"]
        or value.get("build_source_closure") != plan.get("build_source_closure")
        or _validate_prelaunch_audit(
            value.get("abandonment_quiescence_audit")
        ) != value.get("abandonment_quiescence_audit")
        or value.get("status") != "aborted-spent-no-retry"
        or value.get("failure_class") != "infrastructure-interruption"
        or value.get("partial_metrics_read") is not False
        or value.get("improvement_counted") is not False
        or value.get("retry_authorized") is not False
    ):
        raise TeacherTrainingError("development-gate abandonment binding changed")
    abandoned = _utc_instant(
        value.get("abandoned_at_utc"), "development-gate abandonment time"
    )
    if claim is not None and abandoned < _utc_instant(
        claim["claimed_at_utc"], "abandoned gate claim time"
    ):
        raise TeacherTrainingError("development-gate abandonment predates its claim")
    return value


@_heavy_stage
def abandon_gate_execution(
    plan_path: pathlib.Path, *, gate_plan_path: pathlib.Path,
    variant: str, abandoned_at_utc: str,
    process_auditor: ProcessAuditor = _default_gate_process_audit,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    """Seal an interrupted one-shot development gate without retrying it."""

    plan = load_training_plan(plan_path.resolve())
    _guard_test_hooks(
        plan,
        hooks_used=process_auditor is not _default_gate_process_audit,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    gate_plan_path = gate_plan_path.resolve()
    gate_plan = _load_gate_request_plan(plan, gate_plan_path)
    if variant not in gate_plan["active_search_variant_roster"]:
        raise TeacherTrainingError("unknown development-gate variant to abandon")
    directory = _variant_execution_directory(gate_plan, variant)
    output = directory / "aborted.json"
    if output.exists():
        validate_gate_abandonment(plan, gate_plan_path, output)
        return output.resolve()
    request_path, request = _abandonment_request(gate_plan, variant)
    claim_path = directory / "claim.json"
    raw_path = directory / "raw-result.json"
    receipt_path = directory / "receipt.json"
    claim_record = None
    if claim_path.is_file():
        _validate_abandoned_gate_claim(
            claim_path, plan=plan, gate_plan_path=gate_plan_path,
            gate_plan=gate_plan, request_path=request_path,
            request=request, variant=variant,
        )
        claim_record = _record(claim_path)
    elif claim_path.exists() or claim_path.is_symlink():
        raise TeacherTrainingError("abandoned development-gate claim is irregular")
    raw_record = _optional_artifact_record(
        raw_path, label="abandoned development-gate raw result"
    )
    if claim_record is None and raw_record is None:
        raise TeacherTrainingError(
            "development-gate abandonment requires a spent claim or raw result"
        )
    if receipt_path.is_file():
        try:
            _validate_variant_execution(
                receipt_path,
                plan=plan,
                gate_plan={**gate_plan, "path": str(gate_plan_path)},
                request=request,
            )
        except (TeacherTrainingError, OSError, ValueError):
            invalid_receipt = _record(receipt_path)
        else:
            raise TeacherTrainingError(
                "completed development-gate execution cannot be abandoned"
            )
    else:
        invalid_receipt = _optional_artifact_record(
            receipt_path, label="invalid development-gate receipt"
        )
    quiescence_audit = _validate_prelaunch_audit(
        process_auditor(
            _validate_record(request["binary"], f"{variant} gate binary")
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_sealed(output, {
        "schema": GATE_ABANDONMENT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "plan_body_sha256": plan["body_sha256"],
        "training_plan": _record(pathlib.Path(plan["outputs"]["plan"])),
        "gate_plan": _record(gate_plan_path),
        "gate_plan_body_sha256": gate_plan["body_sha256"],
        "variant": variant,
        "request": _record(request_path),
        "claim": claim_record,
        "raw_result": raw_record,
        "invalid_receipt": invalid_receipt,
        "bank": gate_plan["bank"],
        "candidate": {
            "runtime": dict(request["runtime"]),
            "source": dict(request["candidate_source"]),
        },
        "development_exclusion": gate_plan["development_exclusion"],
        "execution_authority": plan["execution_authority"],
        "build_source_closure": plan.get("build_source_closure"),
        "abandonment_quiescence_audit": quiescence_audit,
        "status": "aborted-spent-no-retry",
        "failure_class": "infrastructure-interruption",
        "partial_metrics_read": False,
        "improvement_counted": False,
        "retry_authorized": False,
        "abandoned_at_utc": challenger.utc(
            abandoned_at_utc, "development-gate abandonment time"
        ),
    })
    validate_gate_abandonment(plan, gate_plan_path, output)
    return output.resolve()


@_heavy_stage
def run_gate_execution(
    plan_path: pathlib.Path, *, gate_plan_path: pathlib.Path,
    resume: bool = False,
    runner: GateProcessRunner = _default_gate_process_runner,
    process_auditor: ProcessAuditor = _default_gate_process_audit,
    clock: Callable[[], str] = utc_now,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    _guard_test_hooks(
        plan,
        hooks_used=(
            runner is not _default_gate_process_runner
            or process_auditor is not _default_gate_process_audit
        ),
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    gate_plan = _load_gate_request_plan(plan, gate_plan_path.resolve())
    reference_path = pathlib.Path(gate_plan["execution_outputs"]["reference"])
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("development-gate execution is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=GATE_EXECUTION_REFERENCE_SCHEMA,
            receipt_schema=GATE_EXECUTION_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="development-gate execution",
        )
        if load_gate_execution(plan, gate_plan_path, receipt_path) != receipt:
            raise TeacherTrainingError("resumed development-gate execution changed")
        return receipt_path
    execution_root = pathlib.Path(gate_plan["execution_outputs"]["root"])
    if execution_root.exists() and not execution_root.is_dir():
        raise TeacherTrainingError("development-gate execution root is irregular")
    requests = {
        item["variant"]: _load_sealed(
            _validate_record(item["request"], f"{item['variant']} request"),
            SCREEN_REQUEST_SCHEMA,
            f"{item['variant']} request",
        )
        for item in gate_plan["requests"]
    }
    receipts: dict[str, dict[str, object]] = {}
    previous_finished: dt.datetime | None = None
    plan_ready = (
        _utc_instant(
            gate_plan["prepared_at_utc"], "qualification plan preparation time"
        )
        if gate_plan.get("schema") == FULL_QUALIFICATION_PLAN_SCHEMA
        else None
    )
    for variant in gate_plan["active_search_variant_roster"]:
        request = requests[variant]
        directory = _variant_execution_directory(gate_plan, variant)
        claim_path = directory / "claim.json"
        raw_path = directory / "raw-result.json"
        receipt_path = directory / "receipt.json"
        abandonment_path = directory / "aborted.json"
        if abandonment_path.exists() or abandonment_path.is_symlink():
            validate_gate_abandonment(
                plan, gate_plan_path.resolve(), abandonment_path
            )
            raise TeacherTrainingError(
                "abandoned development-gate variant is spent and cannot retry"
            )
        if receipt_path.exists():
            if not resume:
                raise TeacherTrainingError("development-gate variant already executed")
            gate_plan_with_path = {**gate_plan, "path": str(gate_plan_path.resolve())}
            receipt = _validate_variant_execution(
                receipt_path,
                plan=plan,
                gate_plan=gate_plan_with_path,
                request=request,
            )
            claim = _load_sealed(
                _record_subset(receipt["claim"], f"{variant} claim"),
                GATE_EXECUTION_CLAIM_SCHEMA,
                f"{variant} claim",
            )
            claimed = _utc_instant(claim["claimed_at_utc"], f"{variant} claim time")
            if previous_finished is not None and claimed < previous_finished:
                raise TeacherTrainingError("development-gate variants overlapped")
            previous_finished = _utc_instant(
                receipt["execution"]["finished_at_utc"], f"{variant} finish time"
            )
            receipts[variant] = _record(receipt_path)
            continue
        if claim_path.exists() or raw_path.exists():
            raise TeacherTrainingError(
                "claimed development-gate variant is terminal and cannot retry"
            )
        audit = _validate_prelaunch_audit(
            process_auditor(_validate_record(request["binary"], f"{variant} binary"))
        )
        claimed_at = challenger.utc(clock(), f"{variant} claim time")
        claimed_instant = _utc_instant(claimed_at, f"{variant} claim time")
        if previous_finished is not None and claimed_instant < previous_finished:
            raise TeacherTrainingError("development-gate claim overlaps prior variant")
        if plan_ready is not None and claimed_instant < plan_ready:
            raise TeacherTrainingError("qualification claim predates its plan")
        directory.mkdir(parents=True, exist_ok=True)
        request_path = _validate_record(
            next(item for item in gate_plan["requests"] if item["variant"] == variant)[
                "request"
            ],
            f"{variant} request",
        )
        claim = _write_sealed(claim_path, {
            "schema": GATE_EXECUTION_CLAIM_SCHEMA,
            "campaign_id": plan["campaign_id"],
            "attempt": plan["attempt"],
            "phase": plan["phase"],
            "plan_body_sha256": plan["body_sha256"],
            "gate_plan": _record(gate_plan_path.resolve()),
            "gate_plan_body_sha256": gate_plan["body_sha256"],
            "variant": variant,
            "request": _record(request_path),
            "request_body_sha256": request["body_sha256"],
            "claimed_at_utc": claimed_at,
            "worker": {
                "workers": 1,
                "threads_per_worker": 1,
                "whole_bank_process": True,
                "process_nice": 0,
                "thread_environment": GATE_THREAD_ENVIRONMENT,
            },
            "prelaunch_audit": audit,
            "execution_authority": plan["execution_authority"],
            "no_retry": True,
        })
        command = _gate_command(request, raw_path.resolve())
        environment = dict(os.environ)
        environment.update(GATE_THREAD_ENVIRONMENT)
        launched_at = challenger.utc(clock(), f"{variant} launch time")
        run = dict(runner(command, REPOSITORY, environment, raw_path.resolve()))
        finished_at = challenger.utc(clock(), f"{variant} finish time")
        if not raw_path.is_file():
            raise TeacherTrainingError("development-gate process omitted raw result")
        document = _validate_gate_result(request, raw_path)
        expected_run = {
            "pid", "process_nice", "returncode", "stderr_bytes",
            "stderr_sha256", "stdout_policy",
        }
        if set(run) != expected_run:
            raise TeacherTrainingError("development-gate runner evidence is malformed")
        receipt = _write_sealed(receipt_path, {
            "schema": GATE_VARIANT_EXECUTION_SCHEMA,
            "campaign_id": plan["campaign_id"],
            "attempt": plan["attempt"],
            "phase": plan["phase"],
            "plan_body_sha256": plan["body_sha256"],
            "gate_plan_body_sha256": gate_plan["body_sha256"],
            "variant": variant,
            "claim": _record(claim_path),
            "request": _record(request_path),
            "raw_result": _record(raw_path),
            "profile_activation": _profile_activation_evidence(
                document,
                expected_profile=str(
                    request["search_variant_metadata"]["candidate_search_profile"]
                ),
            ),
            "execution_authority": plan["execution_authority"],
            "execution": {
                "launched_at_utc": launched_at,
                "finished_at_utc": finished_at,
                "command": command,
                "environment": GATE_THREAD_ENVIRONMENT,
                "workers": 1,
                "threads_per_worker": 1,
                "whole_bank_process": True,
                "variants_serial": True,
                **run,
            },
            "status": "complete-no-retry",
            "retry_authorized": False,
        })
        gate_plan_with_path = {**gate_plan, "path": str(gate_plan_path.resolve())}
        _validate_variant_execution(
            receipt_path,
            plan=plan,
            gate_plan=gate_plan_with_path,
            request=request,
        )
        previous_finished = _utc_instant(finished_at, f"{variant} finish time")
        receipts[variant] = _record(receipt_path)
    body = {
        "schema": GATE_EXECUTION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "plan_body_sha256": plan["body_sha256"],
        "gate_plan": _record(gate_plan_path.resolve()),
        "gate_plan_body_sha256": gate_plan["body_sha256"],
        "execution_policy": gate_plan["execution_policy"],
        "variant_order": gate_plan["active_search_variant_roster"],
        "variant_receipts": receipts,
        "execution_authority": plan["execution_authority"],
        "status": "complete-serial-one-worker-no-retry",
        "retry_authorized": False,
    }
    document = _sealed(body)
    receipt_path = _write_content_addressed(
        execution_root, canonical_json_bytes(document), ".gate-execution.json"
    )
    load_gate_execution(plan, gate_plan_path, receipt_path)
    _write_sealed(reference_path, {
        "schema": GATE_EXECUTION_REFERENCE_SCHEMA,
        "plan_body_sha256": plan["body_sha256"],
        "receipt": _record(receipt_path),
        "receipt_body_sha256": document["body_sha256"],
    })
    return receipt_path.resolve()


def _execution_result_paths(
    execution: Mapping[str, Any], *, plan: Mapping[str, Any],
) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for variant in execution["variant_order"]:
        receipt_path = _record_subset(
            execution["variant_receipts"][variant], f"{variant} execution receipt"
        )
        receipt = _load_sealed(
            receipt_path,
            GATE_VARIANT_EXECUTION_SCHEMA,
            f"{variant} execution receipt",
        )
        result[variant] = _record_subset(
            receipt["raw_result"], f"{variant} raw result"
        )
    if list(result) != list(execution["variant_order"]):
        raise TeacherTrainingError("development-gate result order changed")
    del plan
    return result


def _validate_admission_gate_execution(
    value: Mapping[str, Any], *, training_plan: Mapping[str, Any],
    gate_plan_path: pathlib.Path,
) -> dict[str, Any] | None:
    injected = value.get("injected_test_results")
    execution_record = value.get("gate_execution")
    if injected is True:
        campaign_path = _record_subset(
            training_plan.get("campaign_plan"), "injected-result campaign plan"
        )
        campaign = challenger.validate_campaign(campaign_path)
        if (
            campaign.get("inputs", {}).get("production_allowlist_enforced") is True
            or execution_record is not None
            or value.get("gate_execution_body_sha256") is not None
        ):
            raise TeacherTrainingError("injected gate results are test-only")
        return None
    if injected is not False:
        raise TeacherTrainingError("gate execution injection policy changed")
    execution_path = _record_subset(
        execution_record, "admission gate execution"
    )
    execution = load_gate_execution(
        training_plan, gate_plan_path, execution_path
    )
    if (
        execution_record.get("schema") != GATE_EXECUTION_SCHEMA
        or execution_record.get("body_sha256") != execution["body_sha256"]
        or value.get("gate_execution_body_sha256") != execution["body_sha256"]
    ):
        raise TeacherTrainingError("admission gate execution record changed")
    execution_results = _execution_result_paths(
        execution, plan=training_plan
    )
    admission_results = value.get("results")
    if (
        not isinstance(admission_results, Mapping)
        or set(admission_results) != set(execution_results)
        or any(
            admission_results[variant].get("sha256")
            != sha256_file(execution_results[variant])
            for variant in execution_results
        )
    ):
        raise TeacherTrainingError("admission results differ from gate execution")
    return execution


def _validate_admission_full_qualification(
    value: Mapping[str, Any], *, training_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    fields = (
        "full_search_selection", "full_qualification_plan",
        "full_qualification_execution", "qualification_result",
    )
    if value.get("phase") == "pilot":
        if any(value.get(field) is not None for field in fields):
            raise TeacherTrainingError("pilot admission contains full qualification")
        return None
    selection_path = _record_subset(
        value.get("full_search_selection"), "admission full search selection"
    )
    selection = load_full_search_selection(training_plan, selection_path)
    qualification_plan_path = _record_subset(
        value.get("full_qualification_plan"), "admission qualification plan"
    )
    qualification_plan = load_full_qualification_plan(
        training_plan, qualification_plan_path
    )
    execution_path = _record_subset(
        value.get("full_qualification_execution"),
        "admission qualification execution",
    )
    execution = load_gate_execution(
        training_plan, qualification_plan_path, execution_path
    )
    result_paths = _execution_result_paths(execution, plan=training_plan)
    candidate = selection["selected_candidate"]
    variant = candidate["search_variant"]
    if (
        list(result_paths) != [variant]
        or qualification_plan.get("selected_candidate") != candidate
        or value.get("selected_candidate") != candidate
        or value.get("full_search_selection", {}).get("body_sha256")
        != selection["body_sha256"]
        or value.get("gate_execution", {}).get("sha256")
        != selection["search_ab_execution"]["sha256"]
        or value.get("full_qualification_plan", {}).get("body_sha256")
        != qualification_plan["body_sha256"]
        or value.get("full_qualification_execution", {}).get("body_sha256")
        != execution["body_sha256"]
        or value.get("qualification_result", {}).get("sha256")
        != sha256_file(result_paths[variant])
        or not isinstance(value.get("results"), Mapping)
        or set(value["results"]) != set(selection["search_ab_results"])
        or any(
            value["results"][name].get("sha256")
            != selection["search_ab_results"][name].get("sha256")
            for name in selection["search_ab_results"]
        )
        or value.get("development_exclusion")
        != qualification_plan["development_exclusion"]
    ):
        raise TeacherTrainingError("full qualification admission chain changed")
    request = _load_sealed(
        _validate_record(
            qualification_plan["requests"][0]["request"],
            "admission qualification request",
        ),
        SCREEN_REQUEST_SCHEMA,
        "admission qualification request",
    )
    document = _validate_gate_result(request, result_paths[variant])
    return {
        "selection": selection,
        "qualification_plan": qualification_plan,
        "execution": execution,
        "request": request,
        "document": document,
        "summary": _result_summary(document),
    }


def _failure_counts(document: Mapping[str, Any]) -> dict[str, int]:
    output = {name: 0 for name in challenger.qualification.FAILURE_CATEGORIES}
    categories = document.get("result", {}).get("failure_categories", {})
    if not isinstance(categories, Mapping):
        raise TeacherTrainingError("Rank-4 result failure categories are malformed")
    mapping = {
        "candidate_illegal": "illegal",
        "rank4_illegal": "illegal",
        "unfinished": "unfinished",
        "candidate_timeout": "timeout",
        "rank4_timeout": "timeout",
        "candidate_exception": "crash",
        "rank4_exception": "crash",
        "lockstep_mismatch": "crash",
        "candidate_malformed": "malformed",
        "rank4_malformed": "malformed",
    }
    for source, count in categories.items():
        if source not in mapping or isinstance(count, bool) or not isinstance(count, int):
            raise TeacherTrainingError("Rank-4 result has an unknown failure category")
        output[mapping[source]] += count
    return output


def _result_summary(document: Mapping[str, Any]) -> dict[str, object]:
    result = document["result"]
    return {
        "pairs": int(document["config"]["pair_count"]),
        "games": int(result["games"]),
        "candidate_wins": int(result["candidate_wins"]),
        "candidate_color_wins": {
            "0": int(result["candidate_wins_player0"]),
            "1": int(result["candidate_wins_player1"]),
        },
        "failures": _failure_counts(document),
        "operational_failures": int(result["failures"]),
        "unfinished": int(result["unfinished"]),
    }


def _pair_scores(document: Mapping[str, Any]) -> dict[int, int]:
    scores: dict[int, int] = {}
    for game in document["games"]:
        pair = int(game["pair_index"])
        scores.setdefault(pair, 0)
        if game.get("failure") is None and game.get("winner") == game.get(
            "candidate_player"
        ):
            scores[pair] += 1
    return scores


def _actual_clock_timing_clean(document: Mapping[str, Any]) -> bool:
    configuration = document.get("config")
    result = document.get("result")
    candidate = result.get("candidate") if isinstance(result, Mapping) else None
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("mode") != "actual-clock"
        or not isinstance(candidate, Mapping)
        or candidate.get("soft_overruns") != 0
        or candidate.get("headroom_failures") != 0
        or candidate.get("hard_timeouts") != 0
    ):
        return False
    times = candidate.get("times_ms")
    return bool(
        isinstance(times, list)
        and candidate.get("decisions") == len(times)
        and len(times) > 0
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in times
        )
    )


def _operational_parity_clean(document: Mapping[str, Any]) -> bool:
    result = document.get("result")
    categories = (
        result.get("failure_categories") if isinstance(result, Mapping) else None
    )
    if not isinstance(categories, Mapping):
        return False
    parity_failures = {
        "candidate_exception", "candidate_malformed", "candidate_illegal",
        "lockstep_mismatch",
    }
    return all(int(categories.get(name, 0)) == 0 for name in parity_failures)


def _parity_evidence(document: Mapping[str, Any]) -> dict[str, int]:
    result = document["result"]
    categories = result["failure_categories"]
    return {
        name: int(categories.get(name, 0))
        for name in (
            "candidate_exception", "candidate_malformed", "candidate_illegal",
            "lockstep_mismatch",
        )
    }


def _timing_evidence(document: Mapping[str, Any]) -> dict[str, object]:
    candidate = document["result"]["candidate"]
    times = candidate["times_ms"]
    return {
        "mode": document["config"]["mode"],
        "decisions": candidate["decisions"],
        "soft_overruns": candidate["soft_overruns"],
        "headroom_failures": candidate["headroom_failures"],
        "hard_timeouts": candidate["hard_timeouts"],
        "maximum_first_ms": candidate.get("maximum_first_ms"),
        "maximum_later_ms": candidate.get("maximum_later_ms"),
        "times_sha256": sha256_bytes(canonical_json_bytes(times)),
    }


def _request_source_clean(
    request: Mapping[str, Any], *, profile: str, variant: str,
) -> bool:
    active = active_search_variants(profile)
    metadata = _search_variant_metadata(profile, variant)
    source = request.get("candidate_source")
    return bool(
        isinstance(source, Mapping)
        and request.get("search_throughput_profile") == profile
        and request.get("search_variant") == variant
        and request.get("search_variant_metadata") == metadata
        and request.get("compile_time_macros") == list(active[variant])
        and request.get("macros_embedded_at_source_start") is True
        and request.get("source_is_default_for_variant") is True
        and isinstance(request.get("source_reserve"), int)
        and not isinstance(request.get("source_reserve"), bool)
        and request["source_reserve"] >= SOURCE_RESERVE_TARGET
        and isinstance(source.get("bytes"), int)
        and 0 < source["bytes"] < SOURCE_LIMIT_EXCLUSIVE
    )


def _same_treatment_inputs(
    base: Mapping[str, Any], treatment: Mapping[str, Any],
) -> tuple[bool, bool]:
    model_fields = (
        "training_selection", "training_selection_body_sha256",
        "ranking_weight", "seed", "runtime", "runtime_body_sha256",
        "runtime_payload_sha256", "base_model_source",
    )
    same_model = all(base.get(name) == treatment.get(name) for name in model_fields)
    base_configuration = base.get("configuration")
    treatment_configuration = treatment.get("configuration")
    same_execution_configuration = bool(
        isinstance(base_configuration, Mapping)
        and isinstance(treatment_configuration, Mapping)
        and {
            key: value for key, value in base_configuration.items()
            if key != "candidate_search_profile"
        } == {
            key: value for key, value in treatment_configuration.items()
            if key != "candidate_search_profile"
        }
    )
    same_bank = (
        base.get("bank") == treatment.get("bank")
        and same_execution_configuration
    )
    return same_model, same_bank


def _profile_activation_evidence(
    document: Mapping[str, Any], *, expected_profile: str,
) -> dict[str, object]:
    try:
        return dict(gate_support.require_search_profile_exercised(
            document, expected_profile=expected_profile
        ))
    except Exception as error:
        return {
            "schema": gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA,
            "candidate_search_profile": expected_profile,
            "exercised": False,
            "rejection": str(error),
        }


def _treatment_ab_evidence(
    documents: Mapping[str, Mapping[str, Any]],
    requests: Mapping[str, Mapping[str, Any]], *, profile: str,
    phase: str = "full",
) -> dict[str, object]:
    active = phase_gate_variants(phase, profile)
    bases = SEARCH_VARIANT_ORDER if phase == "full" else ("baseline",)
    if profile == "standard-v1":
        if set(documents) != set(active) or set(requests) != set(active):
            raise TeacherTrainingError("standard paired A/B roster changed")
        return {}
    if set(documents) != set(active) or set(requests) != set(active):
        raise TeacherTrainingError("treatment paired A/B roster is incomplete")
    result: dict[str, object] = {}
    for base in bases:
        treatment = _treatment_variant(base, profile)
        base_scores = _pair_scores(documents[base])
        treatment_scores = _pair_scores(documents[treatment])
        same_pair_roster = base_scores.keys() == treatment_scores.keys()
        if not same_pair_roster:
            raise TeacherTrainingError(
                f"{treatment} changed its corresponding base pair roster"
            )
        deltas = [
            treatment_scores[pair] - base_scores[pair]
            for pair in sorted(base_scores)
        ]
        base_summary = _result_summary(documents[base])
        treatment_summary = _result_summary(documents[treatment])
        base_activation = _profile_activation_evidence(
            documents[base], expected_profile="standard-v1"
        )
        treatment_activation = _profile_activation_evidence(
            documents[treatment], expected_profile=profile
        )
        same_model, same_bank = _same_treatment_inputs(
            requests[base], requests[treatment]
        )
        candidate_win_delta = (
            int(treatment_summary["candidate_wins"])
            - int(base_summary["candidate_wins"])
        )
        better_pairs = sum(delta > 0 for delta in deltas)
        worse_pairs = sum(delta < 0 for delta in deltas)
        checks = {
            "identical_model": same_model,
            "identical_bank_and_configuration": same_bank,
            "identical_pair_roster": same_pair_roster,
            "base_zero_failures": _zero_failures(base_summary),
            "treatment_zero_failures": _zero_failures(treatment_summary),
            "source_clean": (
                _request_source_clean(requests[base], profile=profile, variant=base)
                and _request_source_clean(
                    requests[treatment], profile=profile, variant=treatment
                )
            ),
            "parity_clean": (
                _operational_parity_clean(documents[base])
                and _operational_parity_clean(documents[treatment])
            ),
            "timing_clean": (
                _actual_clock_timing_clean(documents[base])
                and _actual_clock_timing_clean(documents[treatment])
            ),
            "base_control_profile_clean": (
                base_activation.get("exercised") is True
            ),
            "intervention_activated": (
                treatment_activation.get("exercised") is True
            ),
            "strictly_better_paired_wins": sum(deltas) > 0,
            "strictly_more_candidate_wins": candidate_win_delta > 0,
        }
        result[treatment] = {
            "search_throughput_profile": profile,
            "profile_compile_macro": SEARCH_THROUGHPUT_PROFILE_MACROS[profile],
            "standard_base_variant": base,
            "treatment_variant": treatment,
            "pairs": len(deltas),
            "candidate_win_delta": candidate_win_delta,
            "paired_candidate_win_delta": sum(deltas),
            "better_pairs": better_pairs,
            "equal_pairs": sum(delta == 0 for delta in deltas),
            "worse_pairs": worse_pairs,
            "input_bindings": {
                "runtime_body_sha256": requests[base][
                    "runtime_body_sha256"
                ],
                "runtime_payload_sha256": requests[base][
                    "runtime_payload_sha256"
                ],
                "bank": requests[base]["bank"],
                "execution_configuration": {
                    key: value
                    for key, value in requests[base]["configuration"].items()
                    if key != "candidate_search_profile"
                },
                "candidate_search_profiles": {
                    "base": requests[base]["configuration"].get(
                        "candidate_search_profile"
                    ),
                    "treatment": requests[treatment]["configuration"].get(
                        "candidate_search_profile"
                    ),
                },
            },
            "source_evidence": {
                "base": {
                    "source": requests[base]["candidate_source"],
                    "compile_time_macros": requests[base][
                        "compile_time_macros"
                    ],
                    "source_reserve": requests[base]["source_reserve"],
                },
                "treatment": {
                    "source": requests[treatment]["candidate_source"],
                    "compile_time_macros": requests[treatment][
                        "compile_time_macros"
                    ],
                    "source_reserve": requests[treatment]["source_reserve"],
                },
            },
            "parity_evidence": {
                "base": _parity_evidence(documents[base]),
                "treatment": _parity_evidence(documents[treatment]),
            },
            "timing_evidence": {
                "base": _timing_evidence(documents[base]),
                "treatment": _timing_evidence(documents[treatment]),
            },
            "base_control_activation": base_activation,
            "intervention_activation": treatment_activation,
            "checks": checks,
            "pairwise_supported": all(checks.values()),
        }
    return result


def _standard_variant_cleanliness(
    documents: Mapping[str, Mapping[str, Any]],
    requests: Mapping[str, Mapping[str, Any]], *, profile: str,
    phase: str = "full",
) -> dict[str, dict[str, object]]:
    expected = (
        SEARCH_VARIANT_ORDER if phase == "full" else ("baseline",)
    )
    if not set(expected) <= set(documents) or not set(expected) <= set(requests):
        raise TeacherTrainingError("standard search cleanliness roster is incomplete")
    result: dict[str, dict[str, object]] = {}
    for variant in expected:
        source_clean = _request_source_clean(
            requests[variant], profile=profile, variant=variant
        )
        timing_clean = _actual_clock_timing_clean(documents[variant])
        result[variant] = {
            "variant": variant,
            "role": "control" if variant == "baseline" else "search-change",
            "source_clean": source_clean,
            "timing_clean": timing_clean,
            "retention_clean": (
                True if variant == "baseline" else source_clean and timing_clean
            ),
            "source_evidence": {
                "source": requests[variant]["candidate_source"],
                "compile_time_macros": requests[variant][
                    "compile_time_macros"
                ],
                "source_reserve": requests[variant]["source_reserve"],
                "binary": requests[variant]["binary"],
            },
            "timing_evidence": _timing_evidence(documents[variant]),
        }
    return result


def _validate_standard_variant_cleanliness(
    value: Mapping[str, Mapping[str, Any]] | None,
    *, phase: str = "full",
) -> dict[str, dict[str, Any]]:
    expected = SEARCH_VARIANT_ORDER if phase == "full" else ("baseline",)
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise TeacherTrainingError("standard search cleanliness roster is incomplete")
    normalized: dict[str, dict[str, Any]] = {}
    for variant in expected:
        raw = value[variant]
        if not isinstance(raw, Mapping):
            raise TeacherTrainingError("standard search cleanliness is malformed")
        evidence = dict(raw)
        source_clean = evidence.get("source_clean")
        timing_clean = evidence.get("timing_clean")
        expected_retention = bool(
            variant == "baseline"
            or (source_clean is True and timing_clean is True)
        )
        if (
            evidence.get("variant") != variant
            or evidence.get("role")
            != ("control" if variant == "baseline" else "search-change")
            or not isinstance(source_clean, bool)
            or not isinstance(timing_clean, bool)
            or evidence.get("retention_clean") is not expected_retention
            or not isinstance(evidence.get("source_evidence"), Mapping)
            or not isinstance(evidence.get("timing_evidence"), Mapping)
        ):
            raise TeacherTrainingError("standard search cleanliness changed")
        normalized[variant] = evidence
    return normalized


def _paired_ab_evidence(
    documents: Mapping[str, Mapping[str, Any]], *,
    search_throughput_profile: str = "standard-v1",
    requests: Mapping[str, Mapping[str, Any]] | None = None,
    phase: str = "full",
) -> dict[str, object]:
    active = phase_gate_variants(phase, search_throughput_profile)
    if set(documents) != set(active):
        raise TeacherTrainingError("paired A/B documents are incomplete")
    baseline = _pair_scores(documents["baseline"])
    result: dict[str, object] = {}
    for name in active:
        scores = _pair_scores(documents[name])
        if scores.keys() != baseline.keys():
            raise TeacherTrainingError("paired A/B opening roster changed")
        deltas = [scores[pair] - baseline[pair] for pair in sorted(baseline)]
        result[name] = {
            "pairs": len(deltas),
            "candidate_win_delta": sum(deltas),
            "better_pairs": sum(delta > 0 for delta in deltas),
            "equal_pairs": sum(delta == 0 for delta in deltas),
            "worse_pairs": sum(delta < 0 for delta in deltas),
            "pair_score_domain": "candidate-wins-in-two-color-pair",
        }
    standard_evidence = {
        "baseline": "baseline",
        "same_pair_roster": True,
        "variants": result,
    }
    if search_throughput_profile == "standard-v1":
        # This is intentionally byte-for-byte the historical evidence shape.
        return standard_evidence
    if requests is None:
        raise TeacherTrainingError("treatment paired A/B requests are absent")
    return {
        **standard_evidence,
        "search_throughput_profile": search_throughput_profile,
        "active_variant_roster": list(active),
        "expected_game_volume": _expected_gate_game_volume(
            tuple(active),
            pairs_per_variant=(PILOT_PAIRS if phase == "pilot" else FULL_PAIRS),
        ),
        "treatment_comparisons": _treatment_ab_evidence(
            documents, requests, profile=search_throughput_profile,
            phase=phase,
        ),
    }


def _select_standard_pilot_search_variant(
    summaries: Mapping[str, Mapping[str, Any]],
    *, variant_cleanliness: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Select only search changes independently supported by paired results."""

    if set(summaries) != set(SEARCH_VARIANTS):
        raise TeacherTrainingError("pilot search A/B result roster is incomplete")
    baseline = summaries["baseline"]
    cleanliness = _validate_standard_variant_cleanliness(variant_cleanliness)
    baseline_wins = int(baseline["candidate_wins"])
    operational = {
        name: (
            int(summary.get("operational_failures", -1)) == 0
            and int(summary.get("unfinished", -1)) == 0
        )
        for name, summary in summaries.items()
    }
    feature_improved = bool(
        operational["no-feature-sort-only"]
        and cleanliness["no-feature-sort-only"]["retention_clean"] is True
        and int(summaries["no-feature-sort-only"]["candidate_wins"]) > baseline_wins
    )
    descendant_improved = bool(
        operational["single-pass-selection-only"]
        and cleanliness["single-pass-selection-only"]["retention_clean"] is True
        and int(summaries["single-pass-selection-only"]["candidate_wins"])
        > baseline_wins
    )
    combined_supported = bool(
        feature_improved
        and descendant_improved
        and operational["combined"]
        and cleanliness["combined"]["retention_clean"] is True
        and int(summaries["combined"]["candidate_wins"])
        >= max(
            int(summaries["no-feature-sort-only"]["candidate_wins"]),
            int(summaries["single-pass-selection-only"]["candidate_wins"]),
        )
    )
    retained = []
    if operational["baseline"]:
        retained.append("baseline")
    if feature_improved:
        retained.append("no-feature-sort-only")
    if descendant_improved:
        retained.append("single-pass-selection-only")
    if combined_supported:
        retained.append("combined")
    eligible = [
        name
        for name in retained
        if int(summaries[name]["candidate_wins"]) >= PILOT_MINIMUM_WINS
    ]
    preference = {
        "combined": 0,
        "no-feature-sort-only": 1,
        "single-pass-selection-only": 2,
        "baseline": 3,
    }
    selection_pool = eligible or retained or ["baseline"]
    selected = min(
        selection_pool,
        key=lambda name: (
            -int(summaries[name]["candidate_wins"]), preference[name]
        ),
    )
    return {
        "baseline_wins": baseline_wins,
        "paired_win_deltas": {
            name: int(summary["candidate_wins"]) - baseline_wins
            for name, summary in summaries.items()
        },
        "independent_changes": {
            "no_feature_sort_improved": feature_improved,
            "single_pass_selection_improved": descendant_improved,
            "combined_supported_by_both_individual_arms": combined_supported,
        },
        "variant_cleanliness": cleanliness,
        "retained_variants": retained,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "selected_variant_passed_screen": selected in eligible,
        "selection_order": "wins-desc-then-combined-feature-descendant-baseline",
    }


def _select_complete_search_variant(
    summaries: Mapping[str, Mapping[str, Any]], *,
    search_throughput_profile: str = "standard-v1",
    treatment_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    variant_cleanliness: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Select the complete full-round 4/8-arm search A/B roster."""

    active = active_search_variants(search_throughput_profile)
    if set(summaries) != set(active):
        raise TeacherTrainingError("pilot search A/B result roster is incomplete")
    standard = _select_standard_pilot_search_variant({
        name: summaries[name] for name in SEARCH_VARIANT_ORDER
    }, variant_cleanliness=variant_cleanliness)
    if search_throughput_profile == "standard-v1":
        if treatment_evidence not in (None, {}):
            raise TeacherTrainingError(
                "standard-v1 cannot consume treatment evidence"
            )
        # Preserve the original four-arm selection result exactly.
        return standard

    expected_treatments = {
        _treatment_variant(base, search_throughput_profile)
        for base in SEARCH_VARIANT_ORDER
    }
    if not isinstance(treatment_evidence, Mapping) or set(
        treatment_evidence
    ) != expected_treatments:
        raise TeacherTrainingError("search treatment evidence roster is incomplete")
    required_checks = {
        "identical_model", "identical_bank_and_configuration",
        "identical_pair_roster", "base_zero_failures",
        "treatment_zero_failures", "source_clean", "parity_clean",
        "timing_clean", "base_control_profile_clean",
        "intervention_activated", "strictly_better_paired_wins",
        "strictly_more_candidate_wins",
    }
    normalized_evidence: dict[str, dict[str, Any]] = {}
    retained_treatments: list[str] = []
    for base in SEARCH_VARIANT_ORDER:
        treatment = _treatment_variant(base, search_throughput_profile)
        raw_evidence = treatment_evidence[treatment]
        if not isinstance(raw_evidence, Mapping):
            raise TeacherTrainingError("search treatment evidence is malformed")
        evidence = dict(raw_evidence)
        checks = evidence.get("checks")
        paired_delta = evidence.get("paired_candidate_win_delta")
        better_pairs = evidence.get("better_pairs")
        worse_pairs = evidence.get("worse_pairs")
        numeric_evidence_valid = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (paired_delta, better_pairs, worse_pairs)
        )
        pairwise_supported = bool(
            isinstance(checks, Mapping)
            and set(checks) == required_checks
            and all(value is True for value in checks.values())
            and numeric_evidence_valid
            and _zero_failures(summaries[base])
            and _zero_failures(summaries[treatment])
            and int(summaries[treatment]["candidate_wins"])
            > int(summaries[base]["candidate_wins"])
            and paired_delta > 0
        )
        base_independently_retained = base in standard["retained_variants"]
        expected_retained = pairwise_supported and base_independently_retained
        if (
            evidence.get("search_throughput_profile")
            != search_throughput_profile
            or evidence.get("standard_base_variant") != base
            or evidence.get("treatment_variant") != treatment
            or evidence.get("profile_compile_macro")
            != SEARCH_THROUGHPUT_PROFILE_MACROS[search_throughput_profile]
            or evidence.get("candidate_win_delta")
            != int(summaries[treatment]["candidate_wins"])
            - int(summaries[base]["candidate_wins"])
            or evidence.get("pairwise_supported") is not pairwise_supported
        ):
            raise TeacherTrainingError("search treatment evidence changed")
        evidence["base_independently_retained"] = base_independently_retained
        evidence["retained"] = expected_retained
        normalized_evidence[treatment] = evidence
        if expected_retained:
            retained_treatments.append(treatment)

    retained = [*standard["retained_variants"], *retained_treatments]
    eligible = [
        name
        for name in retained
        if int(summaries[name]["candidate_wins"]) >= PILOT_MINIMUM_WINS
    ]
    base_preference = {
        "combined": 0,
        "no-feature-sort-only": 1,
        "single-pass-selection-only": 2,
        "baseline": 3,
    }
    preference = {
        **base_preference,
        **{
            _treatment_variant(base, search_throughput_profile): 4 + order
            for base, order in base_preference.items()
        },
    }
    selection_pool = eligible or retained or ["baseline"]
    selected = min(
        selection_pool,
        key=lambda name: (
            -int(summaries[name]["candidate_wins"]), preference[name]
        ),
    )
    return {
        "baseline_wins": standard["baseline_wins"],
        "paired_win_deltas": {
            name: int(summary["candidate_wins"])
            - int(summaries["baseline"]["candidate_wins"])
            for name, summary in summaries.items()
        },
        "independent_changes": {
            **standard["independent_changes"],
            "throughput_treatments_improved": {
                base: _treatment_variant(base, search_throughput_profile)
                in retained_treatments
                for base in SEARCH_VARIANT_ORDER
            },
        },
        "search_throughput_profile": search_throughput_profile,
        "active_variant_roster": list(active),
        "standard_selection": standard,
        "treatment_comparisons": normalized_evidence,
        "retained_variants": retained,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "selected_variant_passed_screen": selected in eligible,
        "selection_order": (
            "wins-desc-then-standard-complexity-before-treatment-complexity"
        ),
    }


def select_pilot_search_variant(
    summaries: Mapping[str, Mapping[str, Any]], *,
    search_throughput_profile: str = "standard-v1",
    treatment_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    variant_cleanliness: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Screen one baseline, optionally with its paired intervention treatment."""

    active = phase_gate_variants("pilot", search_throughput_profile)
    if set(summaries) != set(active):
        raise TeacherTrainingError("pilot screen result roster is incomplete")
    cleanliness = _validate_standard_variant_cleanliness(
        variant_cleanliness, phase="pilot"
    )
    baseline = summaries["baseline"]
    baseline_retained = bool(
        _zero_failures(baseline)
        and cleanliness["baseline"]["source_clean"] is True
        and cleanliness["baseline"]["timing_clean"] is True
    )
    baseline_eligible = bool(
        baseline_retained
        and int(baseline["candidate_wins"]) >= PILOT_MINIMUM_WINS
    )
    retained = ["baseline"] if baseline_retained else []
    eligible = ["baseline"] if baseline_eligible else []
    normalized_evidence: dict[str, dict[str, Any]] = {}
    retained_treatment = None

    if search_throughput_profile == "standard-v1":
        if treatment_evidence not in (None, {}):
            raise TeacherTrainingError(
                "standard pilot cannot consume treatment evidence"
            )
    else:
        treatment = _treatment_variant("baseline", search_throughput_profile)
        if (
            not isinstance(treatment_evidence, Mapping)
            or set(treatment_evidence) != {treatment}
            or not isinstance(treatment_evidence[treatment], Mapping)
        ):
            raise TeacherTrainingError(
                "pilot intervention evidence roster is incomplete"
            )
        evidence = dict(treatment_evidence[treatment])
        checks = evidence.get("checks")
        required_checks = {
            "identical_model", "identical_bank_and_configuration",
            "identical_pair_roster", "base_zero_failures",
            "treatment_zero_failures", "source_clean", "parity_clean",
            "timing_clean", "base_control_profile_clean",
            "intervention_activated", "strictly_better_paired_wins",
            "strictly_more_candidate_wins",
        }
        paired_delta = evidence.get("paired_candidate_win_delta")
        better_pairs = evidence.get("better_pairs")
        worse_pairs = evidence.get("worse_pairs")
        numeric_evidence_valid = all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (paired_delta, better_pairs, worse_pairs)
        )
        pairwise_supported = bool(
            isinstance(checks, Mapping)
            and set(checks) == required_checks
            and all(item is True for item in checks.values())
            and numeric_evidence_valid
            and _zero_failures(baseline)
            and _zero_failures(summaries[treatment])
            and int(summaries[treatment]["candidate_wins"])
            > int(baseline["candidate_wins"])
            and paired_delta > 0
        )
        treatment_retained = pairwise_supported and baseline_retained
        if (
            evidence.get("search_throughput_profile")
            != search_throughput_profile
            or evidence.get("standard_base_variant") != "baseline"
            or evidence.get("treatment_variant") != treatment
            or evidence.get("profile_compile_macro")
            != SEARCH_THROUGHPUT_PROFILE_MACROS[search_throughput_profile]
            or evidence.get("candidate_win_delta")
            != int(summaries[treatment]["candidate_wins"])
            - int(baseline["candidate_wins"])
            or evidence.get("pairwise_supported") is not pairwise_supported
        ):
            raise TeacherTrainingError("pilot intervention evidence changed")
        evidence["base_independently_retained"] = baseline_retained
        evidence["retained"] = treatment_retained
        normalized_evidence[treatment] = evidence
        if treatment_retained:
            retained.append(treatment)
            retained_treatment = treatment
            if int(summaries[treatment]["candidate_wins"]) >= PILOT_MINIMUM_WINS:
                eligible.append(treatment)

    selection_pool = eligible or retained or ["baseline"]
    selected = min(
        selection_pool,
        key=lambda name: (
            -int(summaries[name]["candidate_wins"]),
            0 if name == "baseline" else 1,
        ),
    )
    return {
        "phase": "pilot",
        "roster_policy": (
            "single-baseline-model-screen"
            if search_throughput_profile == "standard-v1"
            else "paired-baseline-intervention-screen"
        ),
        "search_throughput_profile": search_throughput_profile,
        "active_variant_roster": list(active),
        "expected_game_volume": _expected_gate_game_volume(
            tuple(active), pairs_per_variant=PILOT_PAIRS
        ),
        "baseline_wins": int(baseline["candidate_wins"]),
        "paired_win_deltas": {
            name: int(summary["candidate_wins"])
            - int(baseline["candidate_wins"])
            for name, summary in summaries.items()
        },
        "variant_cleanliness": cleanliness,
        "treatment_comparisons": normalized_evidence,
        "retained_treatment": retained_treatment,
        "retained_variants": retained,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "selected_variant_passed_screen": selected in eligible,
        "control_fallback": selected == "baseline",
        "selection_order": "wins-desc-with-clean-baseline-control-fallback",
    }


def select_full_search_variant(
    summaries: Mapping[str, Mapping[str, Any]], *,
    documents: Mapping[str, Mapping[str, Any]],
    requests: Mapping[str, Mapping[str, Any]],
    search_throughput_profile: str,
    treatment_evidence: Mapping[str, Mapping[str, Any]] | None,
    variant_cleanliness: Mapping[str, Mapping[str, Any]],
    pilot_prior_variant: str,
) -> dict[str, object]:
    """Freeze the full-model A/B winner before qualification data is opened."""

    active = active_search_variants(search_throughput_profile)
    if (
        set(summaries) != set(active)
        or set(documents) != set(active)
        or set(requests) != set(active)
        or pilot_prior_variant not in active
    ):
        raise TeacherTrainingError("full search A/B roster/prior is incomplete")
    retention = _select_complete_search_variant(
        summaries,
        search_throughput_profile=search_throughput_profile,
        treatment_evidence=treatment_evidence,
        variant_cleanliness=variant_cleanliness,
    )
    retained = list(retention["retained_variants"])
    retention_evidence = {
        key: value
        for key, value in retention.items()
        if key not in {
            "eligible_variants", "selected_variant",
            "selected_variant_passed_screen", "selection_order",
        }
    }
    base_preference = {
        "combined": 0,
        "no-feature-sort-only": 1,
        "single-pass-selection-only": 2,
        "baseline": 3,
    }
    preference = dict(base_preference)
    if search_throughput_profile != "standard-v1":
        preference.update({
            _treatment_variant(base, search_throughput_profile): 4 + order
            for base, order in base_preference.items()
        })
    selection_pool = retained or ["baseline"]
    selected = min(
        selection_pool,
        key=lambda variant: (
            -int(summaries[variant]["candidate_wins"]),
            0 if variant == pilot_prior_variant else 1,
            preference[variant],
        ),
    )
    return {
        "phase": "full",
        "search_throughput_profile": search_throughput_profile,
        "active_variant_roster": list(active),
        "expected_game_volume": _expected_gate_game_volume(
            tuple(active), pairs_per_variant=FULL_PAIRS
        ),
        "pilot_prior": {
            "variant": pilot_prior_variant,
            "compile_time_macros": list(active[pilot_prior_variant]),
            "role": "equal-win-tie-context-only",
        },
        "independent_ab_retention": retention_evidence,
        "retained_variants": retained,
        "selected_variant": selected,
        "selected_variant_frozen_before_qualification": True,
        "qualification_bank_read": False,
        "selection_order": (
            "ab-wins-desc-then-pilot-prior-tie-then-standard-complexity"
        ),
    }


def _full_search_selection_body(
    plan: Mapping[str, Any], *, gate_plan_path: pathlib.Path,
    execution_path: pathlib.Path, selected_at_utc: str,
) -> dict[str, object]:
    if plan.get("phase") != "full":
        raise TeacherTrainingError("full search selection requires the full phase")
    gate_plan = load_gate_plan(plan, gate_plan_path.resolve())
    execution = load_gate_execution(plan, gate_plan_path.resolve(), execution_path.resolve())
    selected_at = _utc_instant(selected_at_utc, "full search selection time")
    last_variant = execution["variant_order"][-1]
    last_receipt = _load_sealed(
        _record_subset(
            execution["variant_receipts"][last_variant],
            "last search A/B execution receipt",
        ),
        GATE_VARIANT_EXECUTION_SCHEMA,
        "last search A/B execution receipt",
    )
    if selected_at < _utc_instant(
        last_receipt["execution"]["finished_at_utc"],
        "last search A/B finish time",
    ):
        raise TeacherTrainingError("full search selection predates A/B completion")
    result_paths = _execution_result_paths(execution, plan=plan)
    requests = {
        item["variant"]: _load_sealed(
            _validate_record(item["request"], f"{item['variant']} request"),
            SCREEN_REQUEST_SCHEMA,
            f"{item['variant']} request",
        )
        for item in gate_plan["requests"]
    }
    documents = {
        variant: _validate_gate_result(requests[variant], result_paths[variant])
        for variant in gate_plan["active_search_variant_roster"]
    }
    pair_rosters = [
        {
            (int(game["pair_index"]), int(game["candidate_player"]))
            for game in document["games"]
        }
        for document in documents.values()
    ]
    if not pair_rosters or any(roster != pair_rosters[0] for roster in pair_rosters[1:]):
        raise TeacherTrainingError("full search A/B pair roster changed")
    summaries = {
        variant: _result_summary(document)
        for variant, document in documents.items()
    }
    paired = _paired_ab_evidence(
        documents,
        search_throughput_profile=plan["search_throughput_profile"],
        requests=requests,
    )
    search_selection = select_full_search_variant(
        summaries,
        documents=documents,
        requests=requests,
        search_throughput_profile=plan["search_throughput_profile"],
        treatment_evidence=paired.get("treatment_comparisons"),
        variant_cleanliness=_standard_variant_cleanliness(
            documents, requests, profile=plan["search_throughput_profile"]
        ),
        pilot_prior_variant=plan["pilot_admission"]["search_variant"],
    )
    search_selection["paired_evidence"] = paired
    variant = str(search_selection["selected_variant"])
    metadata = _search_variant_metadata(plan["search_throughput_profile"], variant)
    request = requests[variant]
    training_selection = load_training_selection(
        plan, _validate_record(gate_plan["selection"], "full training selection")
    )
    model = training_selection["selected_model"]
    candidate = {
        "ranking_weight": model["ranking_weight"],
        "seed": model["seed"],
        "adaptation_contract": model["adaptation_contract"],
        "qat_profile": model["qat_profile"],
        "qat_profile_contract": model["qat_profile_contract"],
        "qat_execution_evidence_sha256": sha256_bytes(canonical_json_bytes(
            model["qat_execution_evidence"]
        )),
        "hard_state_density": model["hard_state_density"],
        "runtime": model["runtime"],
        "offline_eligible": model["offline_eligible"],
        "diagnostic_only": model["diagnostic_only"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "candidate_search_profile": metadata["candidate_search_profile"],
        "search_variant": variant,
        "standard_base_variant": metadata["standard_base_variant"],
        "search_treatment": metadata["is_treatment"],
        "compile_time_macros": metadata["compile_time_macros"],
        "source": request["candidate_source"],
        "binary": request["binary"],
        "source_is_default_for_variant": True,
    }
    return {
        "schema": FULL_SEARCH_SELECTION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": "full",
        "selected_at_utc": selected_at_utc,
        "plan_body_sha256": plan["body_sha256"],
        "adaptation_contract": plan["adaptation_contract"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "active_search_variant_roster": gate_plan[
            "active_search_variant_roster"
        ],
        "training_selection": gate_plan["selection"],
        "training_selection_body_sha256": gate_plan["selection_body_sha256"],
        "search_ab_gate_plan": {
            **_record(gate_plan_path.resolve()),
            "schema": GATE_PLAN_SCHEMA,
            "body_sha256": gate_plan["body_sha256"],
        },
        "search_ab_execution": {
            **_record(execution_path.resolve()),
            "schema": GATE_EXECUTION_SCHEMA,
            "body_sha256": execution["body_sha256"],
        },
        "search_ab_results": {
            variant: _record(path) for variant, path in result_paths.items()
        },
        "summaries": summaries,
        "search_ab": search_selection,
        "search_ab_game_volume": gate_plan["expected_game_volume"],
        "selected_candidate": candidate,
        "selected_before_qualification_bank_read": True,
        "qualification_bank_read": False,
        "protected_tests_opened": False,
    }


def freeze_full_search_selection(
    plan_path: pathlib.Path, *, gate_plan_path: pathlib.Path,
    execution_path: pathlib.Path, resume: bool = False,
    clock: Callable[[], str] = utc_now,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    reference_path = pathlib.Path(
        plan["outputs"]["full_search_selection_reference"]
    )
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("full search selection is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=FULL_SEARCH_SELECTION_REFERENCE_SCHEMA,
            receipt_schema=FULL_SEARCH_SELECTION_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="full search selection",
        )
        expected = _sealed(_full_search_selection_body(
            plan,
            gate_plan_path=gate_plan_path.resolve(),
            execution_path=execution_path.resolve(),
            selected_at_utc=str(receipt.get("selected_at_utc", "")),
        ))
        if receipt != expected or load_full_search_selection(plan, receipt_path) != receipt:
            raise TeacherTrainingError("resumed full search selection changed")
        return receipt_path
    body = _full_search_selection_body(
        plan,
        gate_plan_path=gate_plan_path.resolve(),
        execution_path=execution_path.resolve(),
        selected_at_utc=challenger.utc(clock(), "full search selection time"),
    )
    document = _sealed(body)
    payload = canonical_json_bytes(document)
    receipt_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["full_search_selections"]),
        payload,
        ".full-search-selection.json",
    )
    load_full_search_selection(plan, receipt_path)
    _write_sealed(reference_path, {
        "schema": FULL_SEARCH_SELECTION_REFERENCE_SCHEMA,
        "plan_body_sha256": plan["body_sha256"],
        "receipt": _record(receipt_path),
        "receipt_body_sha256": document["body_sha256"],
    })
    return receipt_path.resolve()


def load_full_search_selection(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    value = _load_sealed(
        path.resolve(), FULL_SEARCH_SELECTION_SCHEMA, "full search selection"
    )
    gate_plan_path = _record_subset(
        value.get("search_ab_gate_plan"), "full search A/B gate plan"
    )
    execution_path = _record_subset(
        value.get("search_ab_execution"), "full search A/B execution"
    )
    expected = _sealed(_full_search_selection_body(
        plan,
        gate_plan_path=gate_plan_path,
        execution_path=execution_path,
        selected_at_utc=str(value.get("selected_at_utc", "")),
    ))
    if value != expected:
        raise TeacherTrainingError("full search selection evidence changed")
    return value


def _full_qualification_body(
    plan: Mapping[str, Any], *, selection_path: pathlib.Path,
    bank_path: pathlib.Path | None, prepared_at_utc: str,
) -> dict[str, object]:
    if plan.get("phase") != "full":
        raise TeacherTrainingError("qualification is restricted to the full phase")
    selection = load_full_search_selection(plan, selection_path.resolve())
    if (
        selection.get("selected_before_qualification_bank_read") is not True
        or selection.get("qualification_bank_read") is not False
    ):
        raise TeacherTrainingError("search variant was not frozen before qualification")
    prepared_at = _utc_instant(prepared_at_utc, "full qualification preparation time")
    if prepared_at < _utc_instant(
        selection["selected_at_utc"], "full search selection time"
    ):
        raise TeacherTrainingError("qualification predates search selection")
    ab_gate_path = _record_subset(
        selection["search_ab_gate_plan"], "qualification A/B gate plan"
    )
    ab_gate = load_gate_plan(plan, ab_gate_path)
    ab_bank_path = _validate_record(
        ab_gate["bank"]["manifest"], "search A/B bank manifest"
    )
    ab_bank = challenger.openings.validate_bank(ab_bank_path)
    ab_states, ab_features = _bank_fingerprints(ab_bank)

    bundle, inputs, external_validation = training_context(plan)
    del bundle
    production = plan["execution_authority"]["production_allowlist_enforced"]
    qualification_claim_path = (
        pathlib.Path(plan["outputs"]["full_qualification_banks"]).parent
        / "development-bank-claim.json"
    )
    seed_claim = None
    if production:
        generated_directory = (
            pathlib.Path(plan["outputs"]["full_qualification_banks"])
            / "generated"
        )
        seed_claim_path = generated_directory / "development-bank-seed-claim.json"
        if bank_path is not None and qualification_claim_path.is_file():
            effective_bank_path = bank_path.resolve()
            seed_document = _load_sealed(
                seed_claim_path, DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
                "qualification development-bank seed claim",
            )
            seed_claim = {
                **_record(seed_claim_path),
                "schema": DEVELOPMENT_BANK_SEED_CLAIM_SCHEMA,
                "body_sha256": seed_document["body_sha256"],
            }
        else:
            if bank_path is not None:
                raise TeacherTrainingError(
                    "production qualification bank must be generated after search freeze"
                )
            generated_bank, seed_claim = _generate_production_development_bank(
                plan,
                selection_path=selection_path.resolve(),
                selection_body_sha256=selection["body_sha256"],
                purpose="full-qualification", count=FULL_PAIRS,
                output_directory=generated_directory,
                inputs=inputs, external_validation=external_validation,
                extra_state_fingerprints=ab_states,
                extra_feature_fingerprints=ab_features,
            )
            effective_bank_path = generated_bank
    else:
        if bank_path is None:
            raise TeacherTrainingError(
                "nonproduction qualification requires an explicit test bank"
            )
        effective_bank_path = bank_path.resolve()
    qualification_bank_claim = _development_bank_claim(
        plan,
        selection_path=selection_path.resolve(),
        selection_schema=FULL_SEARCH_SELECTION_SCHEMA,
        selection_body_sha256=selection["body_sha256"],
        bank_path=effective_bank_path,
        purpose="full-qualification",
        claim_path=qualification_claim_path,
        seed_claim=seed_claim,
    )
    bank, gate_bank, bank_record = _bank_input(
        effective_bank_path,
        phase="full",
        output_directory=pathlib.Path(plan["outputs"]["full_qualification_banks"]),
    )
    qualification_states, qualification_features = _bank_fingerprints(bank)
    disjoint = {
        "canonical_state_intersection": len(ab_states & qualification_states),
        "canonical_feature_intersection": len(ab_features & qualification_features),
        "search_ab_bank_sha256": ab_gate["bank"]["manifest"]["sha256"],
        "qualification_bank_sha256": bank_record["manifest"]["sha256"],
        "passed": not (
            ab_states & qualification_states
            or ab_features & qualification_features
        ),
    }
    if disjoint["passed"] is not True:
        raise TeacherTrainingError(
            "full qualification bank intersects the search A/B bank"
        )
    phase_game_fingerprints = _phase_game_state_fingerprints(plan)
    freshness = _gate_freshness_audit(
        plan=plan,
        bank=bank,
        inputs=inputs,
        external_validation=external_validation,
        phase_game_fingerprints=phase_game_fingerprints,
    )
    rankings = inputs.successor_rankings
    assert rankings is not None
    development_fingerprints = sorted(
        _development_state_fingerprints(
            bank, rankings, phase_game_fingerprints
        ) | ab_states
    )
    development_document = _sealed({
        "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": "full",
        "prepared_at_utc": prepared_at_utc,
        "classification": "unprotected-development-fingerprints",
        "canonicalization": "minimum-sha256-over-exact+rotate+reflect+rotate-reflect",
        "fingerprints": development_fingerprints,
        "fingerprint_count": len(development_fingerprints),
        "includes_phase_games": True,
        "includes_teacher_successors": True,
        "includes_rank4_gate_bank": True,
        "includes_search_ab_bank": True,
        "includes_post_selection_qualification_bank": True,
        "protected_or_live_data_included": False,
    })
    development_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["development_fingerprints"]),
        canonical_json_bytes(development_document),
        ".development-fingerprints.json",
    )
    candidate = selection["selected_candidate"]
    variant = str(candidate["search_variant"])
    ab_request_record = next(
        item for item in ab_gate["requests"] if item["variant"] == variant
    )
    ab_request = _load_sealed(
        _validate_record(ab_request_record["request"], "selected A/B request"),
        SCREEN_REQUEST_SCHEMA,
        "selected A/B request",
    )
    source = _validate_record(candidate["source"], "qualified candidate source")
    binary = _validate_record(candidate["binary"], "qualified candidate binary")
    configuration = _expected_gate_configuration(
        "full",
        candidate_search_profile=str(candidate["candidate_search_profile"]),
        qualification=True,
    )
    request_body = {
        "schema": SCREEN_REQUEST_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": "full",
        "gate_purpose": "full-qualification",
        "plan_body_sha256": plan["body_sha256"],
        "full_search_selection": {
            **_record(selection_path.resolve()),
            "schema": FULL_SEARCH_SELECTION_SCHEMA,
            "body_sha256": selection["body_sha256"],
        },
        "development_bank_claim": qualification_bank_claim,
        "execution_authority": plan["execution_authority"],
        "model_selected_before_bank_read": True,
        "ranking_weight": candidate["ranking_weight"],
        "seed": candidate["seed"],
        "runtime": candidate["runtime"],
        "runtime_body_sha256": ab_request["runtime_body_sha256"],
        "runtime_payload_sha256": ab_request["runtime_payload_sha256"],
        "search_throughput_profile": candidate["search_throughput_profile"],
        "search_variant": variant,
        "search_variant_metadata": _search_variant_metadata(
            candidate["search_throughput_profile"], variant
        ),
        "compile_time_macros": candidate["compile_time_macros"],
        "macros_embedded_at_source_start": True,
        "candidate_source": _record(source, ascii_required=True),
        "base_model_source": ab_request["base_model_source"],
        "source_is_default_for_variant": True,
        "source_reserve": SOURCE_LIMIT_EXCLUSIVE - source.stat().st_size,
        "compiler": ab_request["compiler"],
        "binary": _record(binary),
        "binary_reference": ab_request["binary_reference"],
        "rank4_source": _record(RANK4_SOURCE),
        "bank": bank_record,
        "freshness_audit": freshness,
        "search_ab_bank_disjoint": disjoint,
        "configuration": configuration,
        "argv": _gate_arguments(
            binary=binary,
            bank=gate_bank,
            source=source,
            pairs=FULL_PAIRS,
            minimum_wins=FULL_MINIMUM_WINS,
            minimum_color_wins=FULL_MINIMUM_COLOR_WINS,
        ),
        "protected_tests_opened": False,
    }
    request_document = _sealed(request_body)
    request_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["gate_requests"]),
        canonical_json_bytes(request_document),
        f".{variant}.full-qualification-request.json",
    )
    variants = [variant]
    return {
        "schema": FULL_QUALIFICATION_PLAN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": "full",
        "plan_body_sha256": plan["body_sha256"],
        "prepared_at_utc": prepared_at_utc,
        "full_search_selection": {
            **_record(selection_path.resolve()),
            "schema": FULL_SEARCH_SELECTION_SCHEMA,
            "body_sha256": selection["body_sha256"],
        },
        "development_bank_claim": qualification_bank_claim,
        "execution_authority": plan["execution_authority"],
        "selected_candidate": candidate,
        "qualification_bank_opened_after_selection": True,
        "search_ab_bank": ab_gate["bank"],
        "bank": bank_record,
        "bank_disjointness": disjoint,
        "freshness_audit": freshness,
        "development_exclusion": {
            **_record(development_path),
            "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "body_sha256": development_document["body_sha256"],
        },
        "active_search_variant_roster": variants,
        "expected_game_volume": _expected_gate_game_volume(
            variants, pairs_per_variant=FULL_PAIRS
        ),
        "execution_policy": _gate_execution_policy(variants),
        "execution_outputs": {
            "root": plan["outputs"]["full_qualification_executions"],
            "reference": plan["outputs"][
                "full_qualification_execution_reference"
            ],
        },
        "requests": [{
            "variant": variant,
            "macros": candidate["compile_time_macros"],
            "search_variant_metadata": request_body["search_variant_metadata"],
            "request": _record(request_path),
            "request_body_sha256": request_document["body_sha256"],
            "source": request_body["candidate_source"],
            "binary": request_body["binary"],
        }],
        "protected_tests_opened": False,
    }


@_heavy_stage
def prepare_full_qualification(
    plan_path: pathlib.Path, *, selection_path: pathlib.Path,
    bank_path: pathlib.Path | None = None, resume: bool = False,
    clock: Callable[[], str] = utc_now,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    reference_path = pathlib.Path(
        plan["outputs"]["full_qualification_plan_reference"]
    )
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("full qualification plan is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=FULL_QUALIFICATION_REFERENCE_SCHEMA,
            receipt_schema=FULL_QUALIFICATION_PLAN_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="full qualification plan",
        )
        if (
            receipt.get("full_search_selection", {}).get("sha256")
            != sha256_file(selection_path.resolve())
            or (
                bank_path is not None
                and receipt.get("bank", {}).get("manifest")
                != _record(bank_path.resolve())
            )
            or load_full_qualification_plan(plan, receipt_path) != receipt
        ):
            raise TeacherTrainingError("resumed full qualification plan changed")
        return receipt_path
    body = _full_qualification_body(
        plan,
        selection_path=selection_path.resolve(),
        bank_path=(None if bank_path is None else bank_path.resolve()),
        prepared_at_utc=challenger.utc(
            clock(), "full qualification preparation time"
        ),
    )
    document = _sealed(body)
    receipt_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["full_qualification_plans"]),
        canonical_json_bytes(document),
        ".full-qualification-plan.json",
    )
    load_full_qualification_plan(plan, receipt_path)
    _write_sealed(reference_path, {
        "schema": FULL_QUALIFICATION_REFERENCE_SCHEMA,
        "plan_body_sha256": plan["body_sha256"],
        "receipt": _record(receipt_path),
        "receipt_body_sha256": document["body_sha256"],
    })
    return receipt_path.resolve()


def load_full_qualification_plan(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    value = _load_sealed(
        path.resolve(), FULL_QUALIFICATION_PLAN_SCHEMA,
        "full qualification plan",
    )
    selection_path = _record_subset(
        value.get("full_search_selection"), "qualification search selection"
    )
    bank_path = _validate_record(
        value.get("bank", {}).get("manifest"), "qualification bank manifest"
    )
    expected = _sealed(_full_qualification_body(
        plan,
        selection_path=selection_path,
        bank_path=bank_path,
        prepared_at_utc=str(value.get("prepared_at_utc", "")),
    ))
    if value != expected:
        raise TeacherTrainingError("full qualification plan evidence changed")
    return value


def paired_bootstrap_lower_95(
    document: Mapping[str, Any], *, seed_material: str, samples: int = 20_000,
) -> float:
    scores = list(_pair_scores(document).values())
    if len(scores) != FULL_PAIRS or any(score not in (0, 1, 2) for score in scores):
        raise TeacherTrainingError("full paired bootstrap requires exactly 500 pairs")
    counts = np.bincount(np.asarray(scores, dtype=np.int8), minlength=3)
    probabilities = counts.astype(np.float64) / float(len(scores))
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("ascii")).digest()[:8], "little")
    random_source = np.random.default_rng(seed)
    draws = random_source.multinomial(len(scores), probabilities, size=samples)
    rates = (draws[:, 1] + 2 * draws[:, 2]) / float(2 * len(scores))
    return float(np.quantile(rates, 0.025, method="lower"))


def _zero_failures(summary: Mapping[str, Any]) -> bool:
    failures = summary.get("failures")
    return bool(
        isinstance(failures, Mapping)
        and set(failures) == set(challenger.qualification.FAILURE_CATEGORIES)
        and all(value == 0 for value in failures.values())
        and summary.get("operational_failures") == 0
        and summary.get("unfinished") == 0
    )


def _validate_admission_search_evidence(
    value: Mapping[str, Any], *, training_plan: Mapping[str, Any],
    gate_plan_path: pathlib.Path,
) -> dict[str, Any]:
    """Rebuild the selected search decision from sealed requests/results."""

    gate_plan = load_gate_plan(training_plan, gate_plan_path)
    requests = {
        record["variant"]: _load_sealed(
            _validate_record(
                record.get("request"), f"{record.get('variant')} gate request"
            ),
            SCREEN_REQUEST_SCHEMA,
            f"{record.get('variant')} gate request",
        )
        for record in gate_plan["requests"]
    }
    results = value.get("results")
    if not isinstance(results, Mapping) or set(results) != set(requests):
        raise TeacherTrainingError("admission search result roster changed")
    documents: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, object]] = {}
    pair_roster: set[tuple[int, int]] | None = None
    for variant, request in requests.items():
        result_path = _record_subset(results[variant], f"{variant} result")
        document = _validate_gate_result(request, result_path)
        roster = {
            (int(game["pair_index"]), int(game["candidate_player"]))
            for game in document["games"]
        }
        if pair_roster is None:
            pair_roster = roster
        elif pair_roster != roster:
            raise TeacherTrainingError(
                "admission search results changed their paired roster"
            )
        documents[variant] = document
        summaries[variant] = _result_summary(document)
    if value.get("summaries") != summaries:
        raise TeacherTrainingError("admission search summaries changed")

    selected = value.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise TeacherTrainingError("admission selected candidate is absent")
    selected_variant = selected.get("search_variant")
    selected_request = requests.get(selected_variant)
    if (
        not isinstance(selected_request, Mapping)
        or selected.get("runtime") != selected_request.get("runtime")
        or selected.get("source") != selected_request.get("candidate_source")
        or selected.get("binary") != selected_request.get("binary")
        or selected.get("compile_time_macros")
        != selected_request.get("compile_time_macros")
        or selected.get("search_throughput_profile")
        != selected_request.get("search_throughput_profile")
        or selected.get("candidate_search_profile")
        != selected_request.get("search_variant_metadata", {}).get(
            "candidate_search_profile"
        )
    ):
        raise TeacherTrainingError(
            "admission candidate differs from its exact gate request"
        )

    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TeacherTrainingError("admission search metrics are absent")
    profile = str(training_plan["search_throughput_profile"])
    paired = _paired_ab_evidence(
        documents,
        search_throughput_profile=profile,
        requests=requests,
        phase=str(training_plan["phase"]),
    )
    variant_cleanliness = _standard_variant_cleanliness(
        documents, requests, profile=profile,
        phase=str(training_plan["phase"]),
    )
    if training_plan["phase"] == "pilot":
        selection = select_pilot_search_variant(
            summaries,
            search_throughput_profile=profile,
            treatment_evidence=paired.get("treatment_comparisons"),
            variant_cleanliness=variant_cleanliness,
        )
        selection["paired_evidence"] = paired
        if (
            metrics.get("search_ab") != selection
            or selection.get("selected_variant") != selected_variant
        ):
            raise TeacherTrainingError(
                "admission search selection cannot be reproduced"
            )
    else:
        pilot_prior = training_plan.get("pilot_admission")
        if not isinstance(pilot_prior, Mapping):
            raise TeacherTrainingError("full admission lost its pilot search prior")
        selection = select_full_search_variant(
            summaries,
            documents=documents,
            requests=requests,
            search_throughput_profile=profile,
            treatment_evidence=paired.get("treatment_comparisons"),
            variant_cleanliness=variant_cleanliness,
            pilot_prior_variant=str(pilot_prior.get("search_variant")),
        )
        selection["paired_evidence"] = paired
        if (
            metrics.get("search_ab") != selection
            or selection.get("selected_variant") != selected_variant
        ):
            raise TeacherTrainingError(
                "full admission search selection cannot be reproduced"
            )
    return {
        "gate_plan": gate_plan,
        "requests": requests,
        "documents": documents,
        "summaries": summaries,
        "selected_variant": selected_variant,
        "search_selection": selection,
    }


def pilot_admission_passes(
    *, summary: Mapping[str, Any], canonical_retention_passed: bool,
    regret_reduction: float, candidate_flip_rate: float,
    scalar_control_flip_rate: float,
    ranking_validation_groups: int,
    comparable_exhaustive_validation_groups: int,
    comparable_exhaustive_validation_fraction: float,
) -> bool:
    return bool(
        summary.get("pairs") == PILOT_PAIRS
        and summary.get("games") == 2 * PILOT_PAIRS
        and isinstance(summary.get("candidate_wins"), int)
        and not isinstance(summary.get("candidate_wins"), bool)
        and summary["candidate_wins"] >= PILOT_MINIMUM_WINS
        and _zero_failures(summary)
        and canonical_retention_passed
        and ranking_validation_groups >= 100
        and comparable_exhaustive_validation_groups >= 100
        and math.isfinite(comparable_exhaustive_validation_fraction)
        and comparable_exhaustive_validation_fraction >= 0.80
        and math.isfinite(regret_reduction)
        and regret_reduction >= 0.10
        and math.isfinite(candidate_flip_rate)
        and math.isfinite(scalar_control_flip_rate)
        and 0.0 <= candidate_flip_rate <= scalar_control_flip_rate + 0.005
    )


def full_admission_passes(
    *, summary: Mapping[str, Any], paired_lower_95: float,
    canonical_retention_passed: bool,
    ranking_validation_groups: int,
    comparable_exhaustive_validation_groups: int,
    comparable_exhaustive_validation_fraction: float,
) -> bool:
    colors = summary.get("candidate_color_wins")
    return bool(
        summary.get("pairs") == FULL_PAIRS
        and summary.get("games") == 2 * FULL_PAIRS
        and isinstance(summary.get("candidate_wins"), int)
        and not isinstance(summary.get("candidate_wins"), bool)
        and summary["candidate_wins"] >= FULL_MINIMUM_WINS
        and isinstance(colors, Mapping)
        and set(colors) == {"0", "1"}
        and all(
            isinstance(colors[color], int)
            and not isinstance(colors[color], bool)
            and colors[color] >= FULL_MINIMUM_COLOR_WINS
            for color in ("0", "1")
        )
        and sum(colors.values()) == summary["candidate_wins"]
        and math.isfinite(paired_lower_95)
        and paired_lower_95 > 0.5
        and _zero_failures(summary)
        and canonical_retention_passed
        and ranking_validation_groups >= 100
        and comparable_exhaustive_validation_groups >= 100
        and math.isfinite(comparable_exhaustive_validation_fraction)
        and comparable_exhaustive_validation_fraction >= 0.80
    )


def _recomputed_phase_admission(
    *, phase: str, selection: Mapping[str, Any],
    search_evidence: Mapping[str, Any],
    qualification_evidence: Mapping[str, Any] | None = None,
) -> bool:
    """Derive admission only from the sealed model and actual gate evidence."""

    selected_model = selection.get("selected_model")
    arms = selection.get("arms")
    model_selection = selection.get("model_selection")
    selected_variant = search_evidence.get("selected_variant")
    summaries = search_evidence.get("summaries")
    requests = search_evidence.get("requests")
    documents = search_evidence.get("documents")
    if (
        phase not in {"pilot", "full"}
        or not isinstance(selected_model, Mapping)
        or not isinstance(arms, list)
        or not isinstance(model_selection, Mapping)
        or not isinstance(selected_variant, str)
        or not isinstance(summaries, Mapping)
        or not isinstance(requests, Mapping)
        or not isinstance(documents, Mapping)
    ):
        raise TeacherTrainingError(
            "phase admission cannot be reconstructed from sealed evidence"
        )
    selected_arms = [
        arm for arm in arms
        if isinstance(arm, Mapping)
        and arm.get("ranking_weight") == selected_model.get("ranking_weight")
    ]
    control_arms = [
        arm for arm in arms
        if isinstance(arm, Mapping) and arm.get("ranking_weight") == 0.0
    ]
    comparisons = [
        item for item in model_selection.get("comparisons", [])
        if isinstance(item, Mapping)
        and item.get("ranking_weight") == selected_model.get("ranking_weight")
    ]
    if (
        len(selected_arms) != 1
        or len(control_arms) != 1
        or len(comparisons) != 1
        or selected_variant not in summaries
        or selected_variant not in requests
        or selected_variant not in documents
    ):
        raise TeacherTrainingError(
            "phase admission selected evidence is ambiguous"
        )
    selected_arm = selected_arms[0]
    control_arm = control_arms[0]
    comparison = comparisons[0]
    offline_eligible = selected_model.get("offline_eligible")
    diagnostic_only = selected_model.get("diagnostic_only")
    if (
        not isinstance(offline_eligible, bool)
        or not isinstance(diagnostic_only, bool)
        or diagnostic_only is offline_eligible
    ):
        raise TeacherTrainingError(
            "phase admission model eligibility evidence is malformed"
        )
    summary = summaries[selected_variant]
    if not isinstance(summary, Mapping):
        raise TeacherTrainingError("phase admission selected summary is malformed")
    try:
        if phase == "pilot":
            selection_evidence = search_evidence.get("search_selection")
            return bool(
                offline_eligible
                and isinstance(selection_evidence, Mapping)
                and selection_evidence.get("selected_variant_passed_screen")
                is True
                and pilot_admission_passes(
                summary=summary,
                canonical_retention_passed=(
                    selected_arm.get("offline_gate_passed") is True
                ),
                regret_reduction=float(
                    comparison["mean_teacher_regret_reduction_fraction"]
                ),
                candidate_flip_rate=float(
                    selected_arm["metrics"][
                        "float_vs_quantized_action_flip_rate"
                    ]
                ),
                scalar_control_flip_rate=float(
                    control_arm["metrics"][
                        "float_vs_quantized_action_flip_rate"
                    ]
                ),
                ranking_validation_groups=int(
                    selected_arm["metrics"]["ranking_validation_groups"]
                ),
                comparable_exhaustive_validation_groups=int(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_groups"
                    ]
                ),
                comparable_exhaustive_validation_fraction=float(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_fraction"
                    ]
                ),
                )
            )
        if not isinstance(qualification_evidence, Mapping):
            raise TeacherTrainingError("full qualification evidence is absent")
        request = qualification_evidence.get("request")
        document = qualification_evidence.get("document")
        summary = qualification_evidence.get("summary")
        if (
            not isinstance(request, Mapping)
            or not isinstance(document, Mapping)
            or not isinstance(summary, Mapping)
        ):
            raise TeacherTrainingError(
                "full phase gate evidence is malformed"
            )
        lower = paired_bootstrap_lower_95(
            document, seed_material=str(request["body_sha256"])
        )
        return bool(offline_eligible and full_admission_passes(
            summary=summary,
            paired_lower_95=lower,
            canonical_retention_passed=(
                selected_arm.get("offline_gate_passed") is True
            ),
            ranking_validation_groups=int(
                selected_arm["metrics"]["ranking_validation_groups"]
            ),
            comparable_exhaustive_validation_groups=int(
                selected_arm["metrics"][
                    "comparable_exhaustive_validation_groups"
                ]
            ),
            comparable_exhaustive_validation_fraction=float(
                selected_arm["metrics"][
                    "comparable_exhaustive_validation_fraction"
                ]
            ),
        ))
    except TeacherTrainingError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherTrainingError(
            "phase admission model/gate evidence is malformed"
        ) from error


def _parse_result_arguments(values: Sequence[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    known_variants = {
        variant
        for profile in SEARCH_THROUGHPUT_PROFILE_ORDER
        for variant in active_search_variants(profile)
    }
    for value in values:
        name, separator, path = value.partition("=")
        if (
            not separator
            or name not in known_variants
            or name in result
            or not path
        ):
            raise TeacherTrainingError(
                "gate results must be unique SEARCH_VARIANT=/absolute/result.json values"
            )
        result[name] = pathlib.Path(path)
    return result


def admit_phase(
    plan_path: pathlib.Path, *, gate_plan_path: pathlib.Path,
    execution_path: pathlib.Path | None = None,
    full_search_selection_path: pathlib.Path | None = None,
    qualification_plan_path: pathlib.Path | None = None,
    qualification_execution_path: pathlib.Path | None = None,
    result_paths: Mapping[str, pathlib.Path] | None = None,
    allow_injected_test_results: bool = False,
    resume: bool = False,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    gate_plan = load_gate_plan(plan, gate_plan_path.resolve())
    full_search_selection = None
    qualification_plan = None
    qualification_execution = None
    qualification_request = None
    qualification_document = None
    qualification_result_path = None
    if plan["phase"] == "full":
        if any(path is None for path in (
            full_search_selection_path,
            qualification_plan_path,
            qualification_execution_path,
        )):
            raise TeacherTrainingError(
                "full admission requires search selection and fresh qualification"
            )
        full_search_selection_path = full_search_selection_path.resolve()
        qualification_plan_path = qualification_plan_path.resolve()
        qualification_execution_path = qualification_execution_path.resolve()
        full_search_selection = load_full_search_selection(
            plan, full_search_selection_path
        )
        qualification_plan = load_full_qualification_plan(
            plan, qualification_plan_path
        )
        qualification_execution = load_gate_execution(
            plan, qualification_plan_path, qualification_execution_path
        )
        if (
            qualification_plan["full_search_selection"]["sha256"]
            != sha256_file(full_search_selection_path)
            or qualification_plan["selected_candidate"]
            != full_search_selection["selected_candidate"]
        ):
            raise TeacherTrainingError("full qualification changed search selection")
        qualification_paths = _execution_result_paths(
            qualification_execution, plan=plan
        )
        if list(qualification_paths) != [
            full_search_selection["selected_candidate"]["search_variant"]
        ]:
            raise TeacherTrainingError("full qualification result roster changed")
        qualification_result_path = next(iter(qualification_paths.values()))
        qualification_request = _load_sealed(
            _validate_record(
                qualification_plan["requests"][0]["request"],
                "full qualification request",
            ),
            SCREEN_REQUEST_SCHEMA,
            "full qualification request",
        )
        qualification_document = _validate_gate_result(
            qualification_request, qualification_result_path
        )
    elif any(path is not None for path in (
        full_search_selection_path,
        qualification_plan_path,
        qualification_execution_path,
    )):
        raise TeacherTrainingError("pilot admission cannot consume full qualification")
    supplied_results = dict(result_paths or {})
    if execution_path is not None:
        if supplied_results or allow_injected_test_results:
            raise TeacherTrainingError(
                "gate execution cannot be combined with injected results"
            )
        execution_path = execution_path.resolve()
        gate_execution = load_gate_execution(
            plan, gate_plan_path.resolve(), execution_path
        )
        supplied_results = _execution_result_paths(gate_execution, plan=plan)
        gate_execution_record: dict[str, object] | None = {
            **_record(execution_path),
            "schema": GATE_EXECUTION_SCHEMA,
            "body_sha256": gate_execution["body_sha256"],
        }
        injected_test_results = False
    else:
        if not allow_injected_test_results or not supplied_results:
            raise TeacherTrainingError(
                "admission requires a sealed serial gate execution"
            )
        campaign = challenger.validate_campaign(
            _validate_record(plan["campaign_plan"], "campaign plan")
        )
        if campaign.get("inputs", {}).get("production_allowlist_enforced") is True:
            raise TeacherTrainingError("injected gate results are test-only")
        gate_execution = None
        gate_execution_record = None
        injected_test_results = True
    full_search_selection_record = (
        None
        if full_search_selection is None
        else {
            **_record(full_search_selection_path),
            "schema": FULL_SEARCH_SELECTION_SCHEMA,
            "body_sha256": full_search_selection["body_sha256"],
        }
    )
    qualification_plan_record = (
        None
        if qualification_plan is None
        else {
            **_record(qualification_plan_path),
            "schema": FULL_QUALIFICATION_PLAN_SCHEMA,
            "body_sha256": qualification_plan["body_sha256"],
        }
    )
    qualification_execution_record = (
        None
        if qualification_execution is None
        else {
            **_record(qualification_execution_path),
            "schema": GATE_EXECUTION_SCHEMA,
            "body_sha256": qualification_execution["body_sha256"],
        }
    )
    if (
        full_search_selection is not None
        and gate_execution_record != full_search_selection["search_ab_execution"]
    ):
        raise TeacherTrainingError(
            "full admission uses a different search A/B execution"
        )
    reference_path = pathlib.Path(plan["outputs"]["admission_reference"])
    if reference_path.exists():
        if not resume:
            raise TeacherTrainingError("phase admission is complete; use --resume")
        receipt_path, receipt = _load_reference(
            reference_path,
            schema=ADMISSION_REFERENCE_SCHEMA,
            receipt_schema=ADMISSION_SCHEMA,
            expected_plan_body_sha256=str(plan["body_sha256"]),
            label="phase admission",
        )
        if load_phase_admission(receipt_path) != receipt:
            raise TeacherTrainingError("resumed phase admission changed")
        if (
            receipt.get("gate_execution") != gate_execution_record
            or receipt.get("injected_test_results") is not injected_test_results
            or receipt.get("full_search_selection")
            != full_search_selection_record
            or receipt.get("full_qualification_plan")
            != qualification_plan_record
            or receipt.get("full_qualification_execution")
            != qualification_execution_record
        ):
            raise TeacherTrainingError("resumed gate execution differs")
        if supplied_results and receipt.get("results") != {
            name: _record(path.resolve()) for name, path in sorted(supplied_results.items())
        }:
            # Stored results are copied into the phase artifact tree, so compare
            # their bytes rather than their source paths.
            expected_hashes = {
                name: sha256_file(path.resolve())
                for name, path in supplied_results.items()
            }
            observed_hashes = {
                name: record.get("sha256")
                for name, record in receipt.get("results", {}).items()
            }
            if expected_hashes != observed_hashes:
                raise TeacherTrainingError(
                    "resumed admission results differ from the frozen receipt"
                )
        return receipt_path

    requests = {
        record["variant"]: _load_sealed(
            _validate_record(record["request"], f"{record['variant']} request"),
            SCREEN_REQUEST_SCHEMA,
            f"{record['variant']} request",
        )
        for record in gate_plan["requests"]
    }
    if set(supplied_results) != set(requests):
        raise TeacherTrainingError("gate result roster differs from the frozen requests")
    documents: dict[str, dict[str, Any]] = {}
    result_records: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, object]] = {}
    pair_rosters = None
    for variant in requests:
        result_path = supplied_results[variant].resolve()
        document = _validate_gate_result(requests[variant], result_path)
        roster = {
            (int(game["pair_index"]), int(game["candidate_player"]))
            for game in document["games"]
        }
        if pair_rosters is None:
            pair_rosters = roster
        elif roster != pair_rosters:
            raise TeacherTrainingError("search variants do not share the exact pair roster")
        copied = _write_content_addressed(
            pathlib.Path(plan["outputs"]["admissions"]) / "gate-results",
            result_path.read_bytes(),
            f".{variant}.gate.json",
        )
        documents[variant] = document
        result_records[variant] = _record(copied)
        summaries[variant] = _result_summary(document)

    qualification_result_record = None
    if qualification_result_path is not None:
        copied_qualification = _write_content_addressed(
            pathlib.Path(plan["outputs"]["admissions"]) / "qualification-result",
            qualification_result_path.read_bytes(),
            ".full-qualification.gate.json",
        )
        qualification_result_record = _record(copied_qualification)

    selection = load_training_selection(
        plan, _validate_record(gate_plan["selection"], "training selection")
    )
    selected_model = selection["selected_model"]
    selected_arm = next(
        arm
        for arm in selection["arms"]
        if arm["ranking_weight"] == selected_model["ranking_weight"]
    )
    control_arm = next(
        arm for arm in selection["arms"] if arm["ranking_weight"] == 0.0
    )
    comparison = next(
        item
        for item in selection["model_selection"]["comparisons"]
        if item["ranking_weight"] == selected_model["ranking_weight"]
    )

    if plan["phase"] == "pilot":
        paired_evidence = _paired_ab_evidence(
            documents,
            search_throughput_profile=plan["search_throughput_profile"],
            requests=requests,
            phase="pilot",
        )
        search_selection = select_pilot_search_variant(
            summaries,
            search_throughput_profile=plan["search_throughput_profile"],
            treatment_evidence=paired_evidence.get("treatment_comparisons"),
            variant_cleanliness=_standard_variant_cleanliness(
                documents,
                requests,
                profile=plan["search_throughput_profile"],
                phase="pilot",
            ),
        )
        search_selection["paired_evidence"] = paired_evidence
        selected_variant = search_selection["selected_variant"]
        selected_summary = None if selected_variant is None else summaries[selected_variant]
        admitted = bool(
            selected_model["offline_eligible"] is True
            and selected_model["diagnostic_only"] is False
            and search_selection["selected_variant_passed_screen"] is True
            and pilot_admission_passes(
                summary=selected_summary,
                canonical_retention_passed=(
                    selected_arm["offline_gate_passed"] is True
                ),
                regret_reduction=float(
                    comparison["mean_teacher_regret_reduction_fraction"]
                ),
                candidate_flip_rate=float(
                    selected_arm["metrics"][
                        "float_vs_quantized_action_flip_rate"
                    ]
                ),
                scalar_control_flip_rate=float(
                    control_arm["metrics"][
                        "float_vs_quantized_action_flip_rate"
                    ]
                ),
                ranking_validation_groups=int(
                    selected_arm["metrics"]["ranking_validation_groups"]
                ),
                comparable_exhaustive_validation_groups=int(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_groups"
                    ]
                ),
                comparable_exhaustive_validation_fraction=float(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_fraction"
                    ]
                ),
            )
        )
        gate_metrics = (
            {
                "classification": "fresh-unprotected",
                "pairs": PILOT_PAIRS,
                "games": 2 * PILOT_PAIRS,
                "candidate_wins": 0,
                "failures": {
                    name: 0
                    for name in challenger.qualification.FAILURE_CATEGORIES
                },
            }
            if selected_summary is None
            else {
                "classification": "fresh-unprotected",
                "pairs": selected_summary["pairs"],
                "games": selected_summary["games"],
                "candidate_wins": selected_summary["candidate_wins"],
                "failures": selected_summary["failures"],
            }
        )
        metrics: dict[str, object] = {
            "canonical_retention_passed": selected_arm["offline_gate_passed"],
            "offline_model_eligible": selected_model["offline_eligible"],
            "diagnostic_only": selected_model["diagnostic_only"],
            "candidate_quantized": True,
            "evaluation_classification": "unseen-root-unprotected",
            "mean_teacher_regret_reduction_fraction": comparison[
                "mean_teacher_regret_reduction_fraction"
            ],
            "quantized_action_flip_rate": selected_arm["metrics"][
                "float_vs_quantized_action_flip_rate"
            ],
            "scalar_control_action_flip_rate": control_arm["metrics"][
                "float_vs_quantized_action_flip_rate"
            ],
            "ranking_validation_groups": selected_arm["metrics"][
                "ranking_validation_groups"
            ],
            "comparable_exhaustive_validation_groups": selected_arm[
                "metrics"
            ]["comparable_exhaustive_validation_groups"],
            "comparable_exhaustive_validation_fraction": selected_arm[
                "metrics"
            ]["comparable_exhaustive_validation_fraction"],
            "rank4_screen": gate_metrics,
            "search_ab": search_selection,
            "development_gate_game_volume": gate_plan[
                "expected_game_volume"
            ],
        }
    else:
        paired_evidence = _paired_ab_evidence(
            documents,
            search_throughput_profile=plan["search_throughput_profile"],
            requests=requests,
        )
        search_selection = select_full_search_variant(
            summaries,
            documents=documents,
            requests=requests,
            search_throughput_profile=plan["search_throughput_profile"],
            treatment_evidence=paired_evidence.get("treatment_comparisons"),
            variant_cleanliness=_standard_variant_cleanliness(
                documents,
                requests,
                profile=plan["search_throughput_profile"],
            ),
            pilot_prior_variant=plan["pilot_admission"]["search_variant"],
        )
        search_selection["paired_evidence"] = paired_evidence
        selected_variant = search_selection["selected_variant"]
        if (
            full_search_selection is None
            or full_search_selection.get("search_ab") != search_selection
            or full_search_selection.get("selected_candidate", {}).get(
                "search_variant"
            ) != selected_variant
            or qualification_document is None
            or qualification_request is None
        ):
            raise TeacherTrainingError(
                "full search selection/qualification evidence changed"
            )
        selected_summary = _result_summary(qualification_document)
        lower = paired_bootstrap_lower_95(
            qualification_document,
            seed_material=qualification_request["body_sha256"],
        )
        admitted = bool(
            selected_model["offline_eligible"] is True
            and selected_model["diagnostic_only"] is False
            and full_admission_passes(
                summary=selected_summary,
                paired_lower_95=lower,
                canonical_retention_passed=(
                    selected_arm["offline_gate_passed"] is True
                ),
                ranking_validation_groups=int(
                    selected_arm["metrics"]["ranking_validation_groups"]
                ),
                comparable_exhaustive_validation_groups=int(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_groups"
                    ]
                ),
                comparable_exhaustive_validation_fraction=float(
                    selected_arm["metrics"][
                        "comparable_exhaustive_validation_fraction"
                    ]
                ),
            )
        )
        metrics = {
            "actual_clock": {
                "classification": "fresh-unprotected",
                "pairs": selected_summary["pairs"],
                "games": selected_summary["games"],
                "candidate_wins": selected_summary["candidate_wins"],
                "candidate_color_wins": selected_summary["candidate_color_wins"],
                "paired_lower_95": lower,
                "paired_lower_95_method": (
                    "deterministic-20000-cluster-bootstrap-2.5-percentile"
                ),
                "failures": selected_summary["failures"],
            },
            "search_ab": search_selection,
            "search_ab_game_volume": gate_plan["expected_game_volume"],
            "qualification_game_volume": qualification_plan[
                "expected_game_volume"
            ],
            "full_search_selection_body_sha256": full_search_selection[
                "body_sha256"
            ],
            "qualification_bank_selected_after_search_freeze": True,
            "pilot_search_prior": search_selection["pilot_prior"],
            "search_throughput_profile": plan["search_throughput_profile"],
            "canonical_retention_passed": selected_arm["offline_gate_passed"],
            "offline_model_eligible": selected_model["offline_eligible"],
            "diagnostic_only": selected_model["diagnostic_only"],
            "quantized_action_flip_rate": selected_arm["metrics"][
                "float_vs_quantized_action_flip_rate"
            ],
            "scalar_control_action_flip_rate": control_arm["metrics"][
                "float_vs_quantized_action_flip_rate"
            ],
            "ranking_validation_groups": selected_arm["metrics"][
                "ranking_validation_groups"
            ],
            "comparable_exhaustive_validation_groups": selected_arm[
                "metrics"
            ]["comparable_exhaustive_validation_groups"],
            "comparable_exhaustive_validation_fraction": selected_arm[
                "metrics"
            ]["comparable_exhaustive_validation_fraction"],
        }

    selected_request = None if selected_variant is None else requests[selected_variant]
    selected_variant_metadata = (
        None
        if selected_variant is None
        else _search_variant_metadata(
            plan["search_throughput_profile"], selected_variant
        )
    )
    candidate_search_profile = (
        None
        if selected_variant_metadata is None
        else selected_variant_metadata["candidate_search_profile"]
    )
    wins = 0 if selected_summary is None else int(selected_summary["candidate_wins"])
    rank4_win_rate = wins / (2.0 * (
        PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS
    ))
    rank4_absolute_margin_pp = 100.0 * (rank4_win_rate - 0.5)
    # Retained for consumers of the v1 admission shape.  This is an absolute
    # Rank-4 margin, never an attempt-to-attempt improvement measurement.
    strength_delta_pp = rank4_absolute_margin_pp
    regret_reduction = float(comparison["mean_teacher_regret_reduction_fraction"])
    metrics.update({
        "strength_delta_pp": strength_delta_pp,
        "rank4_win_rate": rank4_win_rate,
        "rank4_absolute_margin_pp": rank4_absolute_margin_pp,
        "teacher_regret_reduction_fraction": regret_reduction,
        "qat_profile": plan["training"]["qat_profile"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "candidate_search_profile": candidate_search_profile,
        "active_search_variant_roster": gate_plan[
            "active_search_variant_roster"
        ],
        "selected_search_variant": selected_variant,
    })

    selected_candidate = None
    if selected_request is not None:
        assert selected_variant_metadata is not None
        selected_candidate = {
            "ranking_weight": selected_model["ranking_weight"],
            "seed": selected_model["seed"],
            "adaptation_contract": selected_model["adaptation_contract"],
            "qat_profile": selected_model["qat_profile"],
            "qat_profile_contract": selected_model["qat_profile_contract"],
            "qat_execution_evidence_sha256": sha256_bytes(
                canonical_json_bytes(
                    selected_model["qat_execution_evidence"]
                )
            ),
            "hard_state_density": selected_model["hard_state_density"],
            "runtime": selected_model["runtime"],
            "offline_eligible": selected_model["offline_eligible"],
            "diagnostic_only": selected_model["diagnostic_only"],
            "search_throughput_profile": plan["search_throughput_profile"],
            "candidate_search_profile": candidate_search_profile,
            "search_variant": selected_variant,
            "standard_base_variant": selected_variant_metadata[
                "standard_base_variant"
            ],
            "search_treatment": selected_variant_metadata["is_treatment"],
            "compile_time_macros": selected_variant_metadata[
                "compile_time_macros"
            ],
            "source": selected_request["candidate_source"],
            "binary": selected_request["binary"],
            "source_is_default_for_variant": True,
        }
    if (
        full_search_selection is not None
        and selected_candidate != full_search_selection["selected_candidate"]
    ):
        raise TeacherTrainingError("full admission changed the frozen search candidate")

    phase_reference = _validate_record(plan["phase_reference"], "phase reference")
    campaign_plan = _validate_record(plan["campaign_plan"], "campaign plan")
    campaign_context = challenger.validate_campaign(campaign_plan)
    phase_context = challenger.validate_phase_reference(
        phase_reference, campaign_context["plan"]
    )
    development = (
        gate_plan["development_exclusion"]
        if qualification_plan is None
        else qualification_plan["development_exclusion"]
    )
    development_path = _validate_record(
        {key: development[key] for key in ("path", "bytes", "sha256")},
        "development exclusion",
    )
    evidence_body = {
        "schema": challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "status": "complete",
        "phase_reference": challenger._sealed_record(
            phase_reference, challenger.PHASE_REFERENCE_SCHEMA
        ),
        "schedule": _record(pathlib.Path(phase_context["schedule"])),
        "candidate": (
            None
            if selected_candidate is None
            else {
                "runtime_sha256": selected_candidate["runtime"]["sha256"],
                "source_sha256": selected_candidate["source"]["sha256"],
            }
        ),
        "completed_games": challenger.PHASE_TOTALS[plan["phase"]],
        "completed_quotas": challenger.PHASE_QUOTAS[plan["phase"]],
        "metrics_sha256": sha256_bytes(canonical_json_bytes(metrics)),
        "development_exclusion": challenger._sealed_record(
            development_path, challenger.DEVELOPMENT_EXCLUSION_SCHEMA
        ),
        "protected_or_live_metrics_read": False,
        "all_games_finished": True,
        "qat_profile": plan["training"]["qat_profile"],
        "qat_profile_contract": plan["training"]["qat_profile_contract"],
        "adaptation_contract": plan["adaptation_contract"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "candidate_search_profile": candidate_search_profile,
        "gate_execution": gate_execution_record,
        "full_search_selection": full_search_selection_record,
        "full_qualification_plan": qualification_plan_record,
        "full_qualification_execution": qualification_execution_record,
        "qualification_result": qualification_result_record,
        "injected_test_results": injected_test_results,
        "active_search_variant_roster": gate_plan[
            "active_search_variant_roster"
        ],
        "evidence_closure": {
            "training_plan": {
                **_record(plan_path.resolve()),
                "schema": PLAN_SCHEMA,
                "body_sha256": plan["body_sha256"],
            },
            "pipeline_plan": plan["pipeline_plan"],
            "finalized_pipeline_receipt": {
                **plan["final_pipeline_receipt"],
                "schema": pipeline.STAGE_RECEIPT_SCHEMA,
                "body_sha256": plan["final_pipeline_receipt_body_sha256"],
            },
            "training_selection": {
                **gate_plan["selection"],
                "schema": SELECTION_SCHEMA,
                "body_sha256": gate_plan["selection_body_sha256"],
            },
            "gate_plan": {
                **_record(gate_plan_path.resolve()),
                "schema": GATE_PLAN_SCHEMA,
                "body_sha256": gate_plan["body_sha256"],
            },
            "gate_results": result_records,
            "gate_execution": gate_execution_record,
            "full_search_selection": full_search_selection_record,
            "full_qualification_plan": qualification_plan_record,
            "full_qualification_execution": qualification_execution_record,
            "qualification_result": qualification_result_record,
            "selected_candidate": selected_candidate,
            "input_audit_sha256": sha256_bytes(
                canonical_json_bytes(plan["input_audit"])
            ),
            "build_source_closure_sha256": plan["build_source_closure"][
                "closure_sha256"
            ],
            "protected_tests_opened": False,
        },
    }
    # A diagnostic-only model is still an actual, screened candidate and needs
    # outcome evidence for attribution/loss reuse; its explicit eligibility
    # flags keep it from being admitted or promoted.
    phase_evidence = None
    if selected_candidate is not None:
        phase_evidence_path = pathlib.Path(plan["outputs"]["phase_evidence"])
        phase_evidence_document = _write_sealed(phase_evidence_path, evidence_body)
        phase_evidence = {
            **_record(phase_evidence_path),
            "schema": challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
            "body_sha256": phase_evidence_document["body_sha256"],
        }

    body = {
        "schema": ADMISSION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "plan_body_sha256": plan["body_sha256"],
        "gate_plan": _record(gate_plan_path.resolve()),
        "gate_plan_body_sha256": gate_plan["body_sha256"],
        "gate_execution": gate_execution_record,
        "gate_execution_body_sha256": (
            None if gate_execution is None else gate_execution["body_sha256"]
        ),
        "injected_test_results": injected_test_results,
        "full_search_selection": full_search_selection_record,
        "full_qualification_plan": qualification_plan_record,
        "full_qualification_execution": qualification_execution_record,
        "qualification_result": qualification_result_record,
        "training_selection": gate_plan["selection"],
        "finalized_pipeline_receipt": {
            **plan["final_pipeline_receipt"],
            "schema": pipeline.STAGE_RECEIPT_SCHEMA,
            "body_sha256": plan["final_pipeline_receipt_body_sha256"],
        },
        "results": result_records,
        "summaries": summaries,
        "selected_candidate": selected_candidate,
        "metrics": metrics,
        "strength_delta_pp": strength_delta_pp,
        "teacher_regret_reduction_fraction": regret_reduction,
        "qat_profile": plan["training"]["qat_profile"],
        "qat_profile_contract": plan["training"]["qat_profile_contract"],
        "adaptation_contract": plan["adaptation_contract"],
        "search_throughput_profile": plan["search_throughput_profile"],
        "candidate_search_profile": candidate_search_profile,
        "active_search_variant_roster": gate_plan[
            "active_search_variant_roster"
        ],
        "development_exclusion": development,
        "phase_outcome_evidence": phase_evidence,
        "admitted": admitted,
        "next_route": (
            "materialize-full"
            if admitted and plan["phase"] == "pilot"
            else "prepare-dual-final"
            if admitted
            else "open-next-leakage-isolated-attempt"
        ),
        "protected_or_live_metrics_read": False,
    }
    document = _sealed(body)
    payload = canonical_json_bytes(document)
    receipt_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["admissions"]),
        payload,
        ".phase-admission.json",
    )
    _write_sealed(
        reference_path,
        {
            "schema": ADMISSION_REFERENCE_SCHEMA,
            "plan_body_sha256": plan["body_sha256"],
            "receipt": _record(receipt_path),
            "receipt_body_sha256": document["body_sha256"],
        },
    )
    return receipt_path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--campaign-plan", type=pathlib.Path, required=True)
    prepare.add_argument("--phase-reference", type=pathlib.Path, required=True)
    prepare.add_argument("--pipeline-plan", type=pathlib.Path, required=True)
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--created-at-utc", required=True)
    prepare.add_argument("--pilot-admission", type=pathlib.Path)

    train = commands.add_parser("train")
    train.add_argument("--plan", type=pathlib.Path, required=True)
    train.add_argument("--resume", action="store_true")

    gate = commands.add_parser("prepare-gate")
    gate.add_argument("--plan", type=pathlib.Path, required=True)
    gate.add_argument("--selection", type=pathlib.Path, required=True)
    gate.add_argument(
        "--bank", type=pathlib.Path,
        help="nonproduction test bank; production generates its sole bank",
    )
    gate.add_argument("--resume", action="store_true")

    run_gate = commands.add_parser("run-gate")
    run_gate.add_argument("--plan", type=pathlib.Path, required=True)
    run_gate.add_argument("--gate-plan", type=pathlib.Path, required=True)
    run_gate.add_argument("--resume", action="store_true")

    abandon_gate = commands.add_parser("abandon-gate-execution")
    abandon_gate.add_argument("--plan", type=pathlib.Path, required=True)
    abandon_gate.add_argument("--gate-plan", type=pathlib.Path, required=True)
    abandon_gate.add_argument("--variant", required=True)
    abandon_gate.add_argument("--abandoned-at-utc", required=True)

    select_search = commands.add_parser("select-full-search")
    select_search.add_argument("--plan", type=pathlib.Path, required=True)
    select_search.add_argument("--gate-plan", type=pathlib.Path, required=True)
    select_search.add_argument("--execution", type=pathlib.Path, required=True)
    select_search.add_argument("--resume", action="store_true")

    qualify = commands.add_parser("prepare-full-qualification")
    qualify.add_argument("--plan", type=pathlib.Path, required=True)
    qualify.add_argument("--selection", type=pathlib.Path, required=True)
    qualify.add_argument(
        "--bank", type=pathlib.Path,
        help="nonproduction test bank; production generates its sole bank",
    )
    qualify.add_argument("--resume", action="store_true")

    admit = commands.add_parser("admit")
    admit.add_argument("--plan", type=pathlib.Path, required=True)
    admit.add_argument("--gate-plan", type=pathlib.Path, required=True)
    admit.add_argument("--execution", type=pathlib.Path, required=True)
    admit.add_argument("--full-search-selection", type=pathlib.Path)
    admit.add_argument("--qualification-plan", type=pathlib.Path)
    admit.add_argument("--qualification-execution", type=pathlib.Path)
    admit.add_argument("--resume", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=pathlib.Path, required=True)
    verify.add_argument("--selection", type=pathlib.Path)
    verify.add_argument("--gate-plan", type=pathlib.Path)
    verify.add_argument("--gate-execution", type=pathlib.Path)
    verify.add_argument("--admission", type=pathlib.Path)

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            output = prepare_training(
                campaign_plan=arguments.campaign_plan,
                phase_reference=arguments.phase_reference,
                pipeline_plan=arguments.pipeline_plan,
                output_root=arguments.output_root,
                created_at_utc=arguments.created_at_utc,
                pilot_admission=arguments.pilot_admission,
            )
            result: object = {"training_plan": str(output)}
        elif arguments.command == "train":
            output = run_training(arguments.plan, resume=arguments.resume)
            result = {"training_selection": str(output)}
        elif arguments.command == "prepare-gate":
            output = prepare_gate(
                arguments.plan,
                selection_path=arguments.selection,
                bank_path=arguments.bank,
                resume=arguments.resume,
            )
            result = {"gate_plan": str(output)}
        elif arguments.command == "run-gate":
            output = run_gate_execution(
                arguments.plan,
                gate_plan_path=arguments.gate_plan,
                resume=arguments.resume,
            )
            result = {"gate_execution": str(output)}
        elif arguments.command == "abandon-gate-execution":
            output = abandon_gate_execution(
                arguments.plan,
                gate_plan_path=arguments.gate_plan,
                variant=arguments.variant,
                abandoned_at_utc=arguments.abandoned_at_utc,
            )
            result = {"gate_abandonment": str(output)}
        elif arguments.command == "select-full-search":
            output = freeze_full_search_selection(
                arguments.plan,
                gate_plan_path=arguments.gate_plan,
                execution_path=arguments.execution,
                resume=arguments.resume,
            )
            result = {"full_search_selection": str(output)}
        elif arguments.command == "prepare-full-qualification":
            output = prepare_full_qualification(
                arguments.plan,
                selection_path=arguments.selection,
                bank_path=arguments.bank,
                resume=arguments.resume,
            )
            result = {"full_qualification_plan": str(output)}
        elif arguments.command == "admit":
            output = admit_phase(
                arguments.plan,
                gate_plan_path=arguments.gate_plan,
                execution_path=arguments.execution,
                full_search_selection_path=arguments.full_search_selection,
                qualification_plan_path=arguments.qualification_plan,
                qualification_execution_path=arguments.qualification_execution,
                resume=arguments.resume,
            )
            result = {"phase_admission": str(output)}
        else:
            plan = load_training_plan(arguments.plan)
            verified: dict[str, object] = {
                "plan_body_sha256": plan["body_sha256"],
                "phase": plan["phase"],
            }
            if arguments.selection:
                selection = load_training_selection(plan, arguments.selection)
                verified["selection_body_sha256"] = selection["body_sha256"]
            if arguments.gate_plan:
                gate_plan = load_gate_plan(plan, arguments.gate_plan)
                verified["gate_plan_body_sha256"] = gate_plan["body_sha256"]
            if arguments.gate_execution:
                if not arguments.gate_plan:
                    raise TeacherTrainingError(
                        "gate execution verification requires --gate-plan"
                    )
                execution = load_gate_execution(
                    plan, arguments.gate_plan, arguments.gate_execution
                )
                verified["gate_execution_body_sha256"] = execution[
                    "body_sha256"
                ]
            if arguments.admission:
                admission = _load_sealed(
                    arguments.admission, ADMISSION_SCHEMA, "phase admission"
                )
                if admission.get("plan_body_sha256") != plan["body_sha256"]:
                    raise TeacherTrainingError("phase admission uses another plan")
                verified["admission_body_sha256"] = admission["body_sha256"]
                verified["admitted"] = admission["admitted"]
            result = verified
    except (TeacherTrainingError, trainer.TrainingError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
