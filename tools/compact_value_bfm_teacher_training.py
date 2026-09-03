#!/usr/bin/env python3
"""Receipt-backed training and Rank-4 screening for the teacher challenger.

This adapter is the bridge between a finalized pilot/full label pipeline and
``compact_value_bfm_train``.  It deliberately keeps the frozen 6301x12x8x1
value-only runtime contract: the rich complete-turn labels affect the loss,
not the deployed interface or architecture.

The four operations are explicit and resumable:

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
    reference/optimized search variants around the already-selected model.
``admit``
    Validate exact Rank-4 gate results, choose the search variant only after
    model selection, and emit governance-compatible phase evidence.

No operation opens a protected model-validation split, schedules recurring
work, uploads a source, or changes Rank 4.
"""

from __future__ import annotations

import argparse
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

SHA256_RE = re.compile(r"[0-9a-f]{64}")


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
    if (
        value.get("campaign_id") != campaign_id
        or value.get("attempt") != attempt
        or value.get("phase") != "pilot"
        or value.get("admitted") is not True
        or not isinstance(selected, Mapping)
        or selected.get("search_variant") not in SEARCH_VARIANTS
        or float(selected.get("ranking_weight", -1.0)) not in (0.10, 0.25)
        or selected.get("source_is_default_for_variant") is not True
    ):
        raise TeacherTrainingError("full phase requires an admitted pilot candidate")
    return value


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
        "finalized_pipeline_receipt", "results", "summaries",
        "selected_candidate", "metrics", "strength_delta_pp",
        "teacher_regret_reduction_fraction", "development_exclusion",
        "phase_outcome_evidence", "admitted", "next_route",
        "protected_or_live_metrics_read", "body_sha256",
    }
    selected = value.get("selected_candidate")
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
        or selected.get("search_variant") not in SEARCH_VARIANTS
        or selected.get("compile_time_macros")
        != list(SEARCH_VARIANTS[selected["search_variant"]])
        or selected.get("source_is_default_for_variant") is not True
        or value.get("protected_or_live_metrics_read") is not False
        or not isinstance(value.get("admitted"), bool)
    ):
        raise TeacherTrainingError("phase admission contract changed")
    gate_plan_path = _record_subset(value["gate_plan"], "admission gate plan")
    gate_plan = _load_sealed(gate_plan_path, GATE_PLAN_SCHEMA, "admission gate plan")
    if gate_plan["body_sha256"] != value.get("gate_plan_body_sha256"):
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
        or candidate
        != {
            "runtime_sha256": selected.get("runtime", {}).get("sha256"),
            "source_sha256": selected.get("source", {}).get("sha256"),
        }
        or evidence_closure.get("selected_candidate") != selected
        or evidence_closure.get("gate_results") != value["results"]
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
        if variant not in SEARCH_VARIANTS:
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
    pipeline_context, final_receipt, final_receipt_path = _finalized_pipeline(
        pipeline_plan
    )
    if (
        pipeline_context.get("campaign_id") != campaign_context["plan"]["campaign_id"]
        or pipeline_context.get("attempt") != attempt
        or pipeline_context.get("phase") != phase
        or pipeline_context.get("campaign_plan") != _record(campaign_plan.resolve())
        or pipeline_context.get("phase_reference") != _record(phase_reference.resolve())
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
            name: list(macros) for name, macros in SEARCH_VARIANTS.items()
        },
        "build_source_closure": build_source_closure,
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
    if plan.get("outputs") != _phase_paths(root):
        raise TeacherTrainingError("teacher-training output routes changed")
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
        or plan.get("search_variants")
        != {name: list(macros) for name, macros in SEARCH_VARIANTS.items()}
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
        ):
            raise TeacherTrainingError("full plan pilot admission binding changed")
        expected_weights = [0.0, float(pilot["ranking_weight"])]
    elif pilot is not None:
        raise TeacherTrainingError("pilot plan unexpectedly binds an admission")
    training = plan.get("training")
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
    pipeline_plan = pipeline.load_pipeline(
        _validate_record(plan["pipeline_plan"], "pipeline plan")
    )
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
    initial_checkpoint: pathlib.Path, resume: bool,
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
        resume=resume,
    )


def _validate_seed_roster(
    receipts: Sequence[Mapping[str, Any]], *, ranking_weight: float,
) -> list[dict[str, Any]]:
    if (
        len(receipts) != len(trainer.FIXED_SEEDS)
        or [receipt.get("seed") for receipt in receipts] != list(trainer.FIXED_SEEDS)
    ):
        raise TeacherTrainingError("training did not return the exact fixed seed roster")
    normalized: list[dict[str, Any]] = []
    for receipt in receipts:
        successor = receipt.get("successor_ranking")
        if (
            receipt.get("architecture") != ARCHITECTURE.name
            or receipt.get("arm") != ARM.name
            or not isinstance(successor, Mapping)
            or successor.get("labels_present") is not True
            or float(successor.get("loss_weight", -1.0)) != ranking_weight
            or not isinstance(receipt.get("offline_gate"), Mapping)
            or not isinstance(receipt.get("quantized_validation"), Mapping)
            or not isinstance(receipt.get("float_checkpoint"), Mapping)
            or not isinstance(receipt.get("quantized_runtime"), Mapping)
        ):
            raise TeacherTrainingError("seed receipt changed its ranking/runtime contract")
        normalized.append(dict(receipt))
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
        return {
            "status": "offline-rejected-before-rank4-screen",
            "control_ranking_weight": 0.0,
            "comparisons": comparisons,
            "selected_ranking_weight": None,
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


def run_training(
    plan_path: pathlib.Path, *, resume: bool = False,
    roster_runner: RosterRunner | None = None,
    renderer: Callable[[pathlib.Path], bytes] | None = None,
    source_verifier: SourceVerifier | None = None,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
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
                resume,
            ),
            ranking_weight=weight,
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
        arms.append({
            "ranking_weight": weight,
            "seed": selected["seed"],
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
    selected_model = None
    if model_selection["selected_ranking_weight"] is not None:
        selected_arm = next(
            arm
            for arm in arms
            if arm["ranking_weight"] == model_selection["selected_ranking_weight"]
        )
        selected_model = {
            "ranking_weight": selected_arm["ranking_weight"],
            "seed": selected_arm["seed"],
            "runtime": selected_arm["runtime"],
            "float_checkpoint": selected_arm["float_checkpoint"],
            "source": selected_arm["source"],
            "metrics": selected_arm["metrics"],
            "selected_before_rank4_bank_read": True,
        }
    body: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "attempt": plan["attempt"],
        "phase": plan["phase"],
        "plan_body_sha256": plan["body_sha256"],
        "source_bundle_body_sha256": plan["source_bundle"]["body_sha256"],
        "initial_checkpoint": plan["initial_checkpoint"],
        "training_policy": plan["training"],
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


def load_training_selection(
    plan: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    value = _load_sealed(path.resolve(), SELECTION_SCHEMA, "training selection")
    selected = value.get("selected_model")
    if (
        value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("source_bundle_body_sha256")
        != plan["source_bundle"]["body_sha256"]
        or value.get("initial_checkpoint") != plan["initial_checkpoint"]
        or value.get("training_policy") != plan["training"]
        or value.get("input_audit") != plan["input_audit"]
        or value.get("rank4_screen_bank_read") is not False
        or value.get("protected_tests_opened") is not False
        or not isinstance(value.get("arms"), list)
        or value.get("model_selection") != _model_selection(value["arms"])
    ):
        raise TeacherTrainingError("training selection binding changed")
    for arm in value["arms"]:
        if (
            arm.get("seed") not in trainer.FIXED_SEEDS
            or not isinstance(arm.get("ranking_weight"), (int, float))
            or float(arm["ranking_weight"]) not in trainer.RANKING_LOSS_WEIGHTS
            or not isinstance(arm.get("metrics"), Mapping)
            or not isinstance(arm.get("seed_receipts"), list)
            or [item.get("seed") for item in arm["seed_receipts"]]
            != list(trainer.FIXED_SEEDS)
        ):
            raise TeacherTrainingError("training arm/seed roster changed")
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
    if [float(arm["ranking_weight"]) for arm in value["arms"]] != [
        float(weight) for weight in plan["training"]["ranking_weights"]
    ]:
        raise TeacherTrainingError("training selection arm roster changed")
    if value["model_selection"]["selected_ranking_weight"] is None:
        if selected is not None:
            raise TeacherTrainingError("rejected model selection retained a candidate")
    elif not isinstance(selected, Mapping):
        raise TeacherTrainingError("eligible model selection lost its candidate")
    else:
        matching = [
            arm
            for arm in value["arms"]
            if arm["ranking_weight"] == selected.get("ranking_weight")
        ]
        if (
            len(matching) != 1
            or selected.get("selected_before_rank4_bank_read") is not True
            or any(
                selected.get(name) != matching[0].get(name)
                for name in (
                    "seed", "runtime", "float_checkpoint", "source", "metrics"
                )
            )
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


def prepare_gate(
    plan_path: pathlib.Path, *, selection_path: pathlib.Path,
    bank_path: pathlib.Path, resume: bool = False,
    compiler: Compiler | None = None,
    compiler_identity: Mapping[str, object] | None = None,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    selection = load_training_selection(plan, selection_path.resolve())
    if selection.get("model_selection", {}).get("status") != (
        "model-selected-before-rank4-screen"
    ):
        raise TeacherTrainingError("no offline-eligible trained model can be screened")
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
            or validated.get("bank", {}).get("manifest")
            != _record(bank_path.resolve())
            or receipt != validated
        ):
            raise TeacherTrainingError("resumed gate inputs differ from the frozen plan")
        return receipt_path

    bundle, inputs, external_validation = training_context(plan)
    del bundle
    bank, gate_bank, bank_record = _bank_input(
        bank_path,
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
    inherited_variant = None
    if plan["phase"] == "pilot":
        variants = SEARCH_VARIANT_ORDER
    else:
        inherited_variant = plan["pilot_admission"]["search_variant"]
        variants = (inherited_variant,)

    identity = dict(compiler_identity or _compiler_identity())
    compile_function = compiler or _compile_gate
    pairs = PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS
    minimum_wins = (
        PILOT_MINIMUM_WINS if plan["phase"] == "pilot" else FULL_MINIMUM_WINS
    )
    minimum_color_wins = -1 if plan["phase"] == "pilot" else FULL_MINIMUM_COLOR_WINS
    requests = []
    for variant in variants:
        macros = SEARCH_VARIANTS[variant]
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
            "plan_body_sha256": plan["body_sha256"],
            "training_selection": _record(selection_path.resolve()),
            "training_selection_body_sha256": selection["body_sha256"],
            "model_selected_before_bank_read": True,
            "ranking_weight": selected["ranking_weight"],
            "seed": selected["seed"],
            "runtime": _record(runtime),
            "runtime_body_sha256": runtime_metadata["body_sha256"],
            "runtime_payload_sha256": runtime_document["quantization"][
                "payload_sha256"
            ],
            "search_variant": variant,
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
        "model_selected_before_bank_read": True,
        "ranking_weight": selected["ranking_weight"],
        "runtime": _record(runtime),
        "bank": bank_record,
        "freshness_audit": freshness,
        "development_exclusion": {
            **_record(development_path),
            "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "body_sha256": development_document["body_sha256"],
        },
        "search_ab_policy": {
            "trained_model_is_fixed_across_variants": True,
            "variants": list(variants),
            "variant_macros": {
                name: list(SEARCH_VARIANTS[name]) for name in variants
            },
            "baseline": "legacy-feature-sort+legacy-descendant-sort",
            "no-feature-sort-only": "optimized-features+legacy-descendant-sort",
            "single-pass-selection-only": "legacy-features+single-pass-selection",
            "combined": "optimized-features+single-pass-selection",
            "full_inherits_pilot_variant": inherited_variant,
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
    expected_variants = (
        list(SEARCH_VARIANT_ORDER)
        if plan["phase"] == "pilot"
        else [plan["pilot_admission"]["search_variant"]]
    )
    requests = value.get("requests")
    if (
        value.get("campaign_id") != plan["campaign_id"]
        or value.get("attempt") != plan["attempt"]
        or value.get("phase") != plan["phase"]
        or value.get("plan_body_sha256") != plan["body_sha256"]
        or value.get("model_selected_before_bank_read") is not True
        or value.get("protected_tests_opened") is not False
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
        if request_record.get("macros") != list(SEARCH_VARIANTS[variant]):
            raise TeacherTrainingError("gate request macro roster changed")
        request_path = _validate_record(
            request_record.get("request"), f"{variant} gate request"
        )
        request = _load_sealed(
            request_path, SCREEN_REQUEST_SCHEMA, f"{variant} gate request"
        )
        if (
            request.get("plan_body_sha256") != plan["body_sha256"]
            or request.get("search_variant") != variant
            or request.get("compile_time_macros")
            != list(SEARCH_VARIANTS[variant])
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
            or request.get("runtime") != value.get("runtime")
            or request.get("bank") != value.get("bank")
            or request.get("configuration")
            != _expected_gate_configuration(str(plan["phase"]))
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
            base_source.read_bytes(), SEARCH_VARIANTS[variant]
        ):
            raise TeacherTrainingError(f"{variant} source does not encode its macros")
        _validate_record(request.get("binary"), f"{variant} gate binary")
    return value


def _expected_gate_configuration(phase: str) -> dict[str, object]:
    pairs = PILOT_PAIRS if phase == "pilot" else FULL_PAIRS
    return {
        **GATE_CONFIGURATION,
        "pair_count": pairs,
        "minimum_candidate_wins": (
            PILOT_MINIMUM_WINS if phase == "pilot" else FULL_MINIMUM_WINS
        ),
        "minimum_wins_per_color": (
            -1 if phase == "pilot" else FULL_MINIMUM_COLOR_WINS
        ),
    }


def _validate_gate_result(
    request: Mapping[str, Any], path: pathlib.Path,
) -> dict[str, Any]:
    expected_bank = request["bank"]["gate_tsv"]["sha256"]
    expected_source = request["candidate_source"]["sha256"]
    try:
        document = gate_support.validate_result(
            path.resolve(),
            expected_bank_sha256=expected_bank,
            expected_candidate_sha256=expected_source,
        )
    except Exception as error:
        raise TeacherTrainingError(
            f"{request['search_variant']} Rank-4 result did not validate"
        ) from error
    expected = _expected_gate_configuration(str(request["phase"]))
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


def _paired_ab_evidence(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    if set(documents) != set(SEARCH_VARIANTS):
        raise TeacherTrainingError("paired A/B documents are incomplete")
    baseline = _pair_scores(documents["baseline"])
    result: dict[str, object] = {}
    for name in SEARCH_VARIANT_ORDER:
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
    return {
        "baseline": "baseline",
        "same_pair_roster": True,
        "variants": result,
    }


def select_pilot_search_variant(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Select only search changes independently supported by paired results."""

    if set(summaries) != set(SEARCH_VARIANTS):
        raise TeacherTrainingError("pilot search A/B result roster is incomplete")
    baseline = summaries["baseline"]
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
        and int(summaries["no-feature-sort-only"]["candidate_wins"]) > baseline_wins
    )
    descendant_improved = bool(
        operational["single-pass-selection-only"]
        and int(summaries["single-pass-selection-only"]["candidate_wins"])
        > baseline_wins
    )
    combined_supported = bool(
        feature_improved
        and descendant_improved
        and operational["combined"]
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
        "retained_variants": retained,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "selected_variant_passed_screen": selected in eligible,
        "selection_order": "wins-desc-then-combined-feature-descendant-baseline",
    }


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


def _parse_result_arguments(values: Sequence[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in SEARCH_VARIANTS or name in result or not path:
            raise TeacherTrainingError(
                "gate results must be unique SEARCH_VARIANT=/absolute/result.json values"
            )
        result[name] = pathlib.Path(path)
    return result


def admit_phase(
    plan_path: pathlib.Path, *, gate_plan_path: pathlib.Path,
    result_paths: Mapping[str, pathlib.Path], resume: bool = False,
) -> pathlib.Path:
    plan = load_training_plan(plan_path.resolve())
    gate_plan = load_gate_plan(plan, gate_plan_path.resolve())
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
        if result_paths and receipt.get("results") != {
            name: _record(path.resolve()) for name, path in sorted(result_paths.items())
        }:
            # Stored results are copied into the phase artifact tree, so compare
            # their bytes rather than their source paths.
            expected_hashes = {
                name: sha256_file(path.resolve())
                for name, path in result_paths.items()
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
    if set(result_paths) != set(requests):
        raise TeacherTrainingError("gate result roster differs from the frozen requests")
    documents: dict[str, dict[str, Any]] = {}
    result_records: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, object]] = {}
    pair_rosters = None
    for variant in requests:
        result_path = result_paths[variant].resolve()
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
        search_selection = select_pilot_search_variant(summaries)
        search_selection["paired_evidence"] = _paired_ab_evidence(documents)
        selected_variant = search_selection["selected_variant"]
        selected_summary = None if selected_variant is None else summaries[selected_variant]
        admitted = pilot_admission_passes(
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
        }
    else:
        selected_variant = next(iter(summaries))
        selected_summary = summaries[selected_variant]
        lower = paired_bootstrap_lower_95(
            documents[selected_variant],
            seed_material=requests[selected_variant]["body_sha256"],
        )
        admitted = full_admission_passes(
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
            "inherited_pilot_search_variant": selected_variant,
            "canonical_retention_passed": selected_arm["offline_gate_passed"],
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
    wins = 0 if selected_summary is None else int(selected_summary["candidate_wins"])
    strength_delta_pp = 100.0 * (wins / (2.0 * (
        PILOT_PAIRS if plan["phase"] == "pilot" else FULL_PAIRS
    )) - 0.5)
    regret_reduction = float(comparison["mean_teacher_regret_reduction_fraction"])
    metrics.update({
        "strength_delta_pp": strength_delta_pp,
        "teacher_regret_reduction_fraction": regret_reduction,
    })

    selected_candidate = None
    if selected_request is not None:
        selected_candidate = {
            "ranking_weight": selected_model["ranking_weight"],
            "seed": selected_model["seed"],
            "runtime": selected_model["runtime"],
            "search_variant": selected_variant,
            "compile_time_macros": list(SEARCH_VARIANTS[selected_variant]),
            "source": selected_request["candidate_source"],
            "binary": selected_request["binary"],
            "source_is_default_for_variant": True,
        }

    phase_reference = _validate_record(plan["phase_reference"], "phase reference")
    campaign_plan = _validate_record(plan["campaign_plan"], "campaign plan")
    campaign_context = challenger.validate_campaign(campaign_plan)
    phase_context = challenger.validate_phase_reference(
        phase_reference, campaign_context["plan"]
    )
    development = gate_plan["development_exclusion"]
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
    # Governance accepts outcome evidence only for an actual candidate.  A
    # rejected model/search selection remains fully auditable in the admission
    # receipt but cannot masquerade as recordable phase evidence.
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
    gate.add_argument("--bank", type=pathlib.Path, required=True)
    gate.add_argument("--resume", action="store_true")

    admit = commands.add_parser("admit")
    admit.add_argument("--plan", type=pathlib.Path, required=True)
    admit.add_argument("--gate-plan", type=pathlib.Path, required=True)
    admit.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="repeat once per request variant",
    )
    admit.add_argument("--resume", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=pathlib.Path, required=True)
    verify.add_argument("--selection", type=pathlib.Path)
    verify.add_argument("--gate-plan", type=pathlib.Path)
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
        elif arguments.command == "admit":
            output = admit_phase(
                arguments.plan,
                gate_plan_path=arguments.gate_plan,
                result_paths=_parse_result_arguments(arguments.result),
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
