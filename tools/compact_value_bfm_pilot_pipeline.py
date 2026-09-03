#!/usr/bin/env python3
"""Resumable pilot/full data and complete-turn label adapter.

The adapter consumes a frozen Rank-4-teacher challenger phase, preserves
its exact game schedule, and drives the already-built continuation and teacher
executables.  It never trains, evaluates a protected split, or uploads.
Long-running commands are explicit and receipt-backed; ``--resume`` reuses only
byte-identical inputs and outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import jacek_replay_corpus as corpus
from tools import jacek_replay_features as features
from tools import jacek_replay_pack as replay_pack


class PilotPipelineError(ValueError):
    """A pilot input, producer result, or resume receipt is inconsistent."""


PIPELINE_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-pipeline.v1"
GAME_PLAN_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-selfsearch-plan.v1"
GAME_MANIFEST_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-games.v1"
POSITION_MANIFEST_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-positions.v1"
HARD_SELECTION_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-hard-selection.v1"
SCALAR_SAMPLE_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-scalar-sample.v1"
SCALAR_PACK_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-scalar-pack.v1"
SHARD_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-shard-reference.v1"
STAGE_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-stage-receipt.v1"
FINGERPRINT_SET_SCHEMA = "papersoccer.compact-value-bfm-pilot-fingerprint-set.v1"
FOUR_WAY_CANONICALIZATION = (
    "minimum-sha256-over-exact+rotate+reflect+rotate-reflect"
)
FEATURE_FINGERPRINT_DOMAIN = "canonical-sparse-active-u16le-v1"
STATE_FINGERPRINT_DOMAIN = "canonical-opening-state-serialization-v1"

GAME_WORKERS_MAX = 8
POSITIONS_PER_GAME_MAX = 20
HARD_NUMERATOR = 1
HARD_DENOMINATOR = 4
SHALLOW_TREE_NODES = 64_000
DEEP_TREE_NODES = 500_000
RANK4_TREE_NODES = 32_000
LABEL_GAMES_PER_CHUNK = 25
COMPACT_GAME_MODES = frozenset({
    "student-selfplay",
    "student-p1-vs-rank4",
    "student-p2-vs-rank4",
    "student-p1-vs-prior-incumbent",
    "student-p2-vs-prior-incumbent",
})
INCUMBENT_GAME_MODES = frozenset({
    "incumbent-p1-vs-rank4",
    "incumbent-p2-vs-rank4",
})
PILOT_GAME_MODES = COMPACT_GAME_MODES | INCUMBENT_GAME_MODES
THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
# Every mutable Python module used by this adapter plus the complete CMake
# source closure of its three native producers.  A phase cannot start or resume
# unless these bytes still equal the clean build manifest committed at freeze.
PIPELINE_REQUIRED_BUILD_SOURCES = tuple(sorted({
    "CMakeLists.txt",
    "requirements-research.txt",
    "tools/compact_value_bfm_pilot_pipeline.py",
    "tools/compact_value_bfm_rank4_teacher_challenger.py",
    "tools/compact_value_bfm_qualification.py",
    "tools/compact_value_bfm_openings.py",
    "tools/compact_value_bfm_train.py",
    "tools/jacek_replay_corpus.py",
    "tools/jacek_replay_features.py",
    "tools/jacek_replay_pack.py",
    "tools/jacek_replay_train.py",
    "submissions/codingame/bots/compact_value_bfm/export_model.py",
    "include/papersoccer/bot.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/bots/bot.cpp",
    "src/bots/mcts_internal.hpp",
    "src/bots/jacek_replay_bfm/features.cpp",
    "src/bots/jacek_replay_bfm/model.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "tools/jacek_replay_continuations.cpp",
    "tools/jacek_replay_continuations_internal.hpp",
    "tools/compact_value_bfm_runtime_loader.hpp",
    "tools/compact_value_bfm_runtime_loader.cpp",
    "tools/jacek_replay_bfm_search_teacher.cpp",
    "tools/jacek_replay_bfm_search_teacher_internal.hpp",
    "tools/jacek_replay_rank4_position_teacher.cpp",
    "submissions/codingame/bots/compact_value_bfm/model.hpp",
    "submissions/codingame/bots/compact_value_bfm/engine.hpp",
    "submissions/codingame/bots/compact_value_bfm/engine.cpp",
    "submissions/codingame/bots/rank_4/replay_book.hpp",
    "submissions/codingame/bots/rank_4/replay_value_model.hpp",
    "submissions/codingame/bots/rank_4/teacher_residual_model.hpp",
    "submissions/codingame/bots/rank_4/bot.cpp",
    "submissions/codingame/bots/jacek_nn/replay_book.hpp",
    "submissions/codingame/bots/jacek_nn/replay_value_model.hpp",
    "submissions/codingame/bots/jacek_nn/teacher_residual_model.hpp",
    "submissions/codingame/bots/jacek_nn/bot.cpp",
}))
# The producers link the shared engine library.  Include its whole C++ header
# and implementation tree so a transitive header/source edit cannot evade the
# explicit entry-point list above.
PIPELINE_REQUIRED_BUILD_SOURCES = tuple(sorted(
    set(PIPELINE_REQUIRED_BUILD_SOURCES)
    | {
        path.relative_to(REPOSITORY).as_posix()
        for root in (REPOSITORY / "include", REPOSITORY / "src")
        for pattern in ("*.hpp", "*.h", "*.cpp")
        for path in root.rglob(pattern)
        if path.is_file() and not path.is_symlink()
    }
))
POSITION_HEADER = (
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
)
GAME_HEADER = "group_id\tsource\twinner\ttranscript"


def canonical_json_bytes(value: object) -> bytes:
    return challenger.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record(path: pathlib.Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise PilotPipelineError(f"required regular file is absent: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_record(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise PilotPipelineError(f"{label} file record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _record(path) != dict(value):
        raise PilotPipelineError(f"{label} bytes changed")
    return path.resolve()


def _seal(body: Mapping[str, object]) -> dict[str, object]:
    return challenger.qualification.seal(dict(body))


def _validate_sealed(value: object, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise PilotPipelineError(f"{label} schema changed")
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256_bytes(canonical_json_bytes(body)):
        raise PilotPipelineError(f"{label} body SHA-256 mismatch")
    return value


def _write_once(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PilotPipelineError(f"immutable output changed: {path}")
        return
    challenger.qualification.atomic_write_once(path, payload)


def _write_sealed(path: pathlib.Path, body: Mapping[str, object]) -> dict[str, object]:
    document = _seal(body)
    _write_once(path, canonical_json_bytes(document))
    return document


def _load_sealed(path: pathlib.Path, schema: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotPipelineError(f"{label} is unreadable") from error
    return _validate_sealed(value, schema, label)


def _bundle_path(context: Mapping[str, Any], record: Mapping[str, object]) -> pathlib.Path:
    if "route" in record:
        root = pathlib.Path(context["plan"]["outputs"]["input_directory"])
        path = root / pathlib.Path(str(record["route"]))
    else:
        path = pathlib.Path(str(record.get("path", "")))
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise PilotPipelineError("challenger bundle record changed")
    return path.resolve()


def _phase_context(
    campaign_plan: pathlib.Path,
    phase_reference: pathlib.Path,
    campaign_context: Mapping[str, Any] | None,
    phase_context: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = (
        dict(campaign_context)
        if campaign_context is not None
        else challenger.validate_campaign(campaign_plan.resolve())
    )
    phase = (
        dict(phase_context)
        if phase_context is not None
        else challenger.validate_phase_reference(phase_reference.resolve(), campaign["plan"])
    )
    phase_name = phase.get("phase", {}).get("phase")
    if phase_name not in challenger.PHASE_TOTALS:
        raise PilotPipelineError("the adapter requires a pilot or full phase reference")
    rows = phase["phase"].get("rows")
    expected_games = challenger.PHASE_TOTALS[phase_name]
    if (
        not isinstance(rows, list)
        or len(rows) != expected_games
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise PilotPipelineError(
            f"{phase_name} phase does not contain exactly {expected_games:,} games"
        )
    if Counter(str(row.get("actor_mode")) for row in rows) != Counter(
        challenger.PHASE_QUOTAS[phase_name]
    ):
        raise PilotPipelineError(f"{phase_name} phase actor quotas changed")
    ordinals = [row.get("game_ordinal") for row in rows]
    seeds = [row.get("base_seed") for row in rows]
    game_ids = [row.get("game_id") for row in rows]
    workers = [row.get("worker") for row in rows]
    if (
        ordinals != list(range(expected_games))
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < 1 << 64
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
        or any(
            not isinstance(game_id, str)
            or not game_id
            or not game_id.isascii()
            for game_id in game_ids
        )
        or len(set(game_ids)) != len(game_ids)
        or any(
            isinstance(worker, bool)
            or not isinstance(worker, int)
            or not 0 <= worker < challenger.RESOURCE_LIMITS["logical_game_shards"]
            for worker in workers
        )
    ):
        raise PilotPipelineError("phase rows cannot execute as a native schedule")
    return campaign, phase


def _render_selfsearch_plan(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = ["game_ordinal\tactor_mode\tbase_seed"]
    previous = -1
    for row in rows:
        game_ordinal = row.get("game_ordinal")
        if (
            isinstance(game_ordinal, bool)
            or not isinstance(game_ordinal, int)
            or game_ordinal <= previous
            or not isinstance(row.get("actor_mode"), str)
            or isinstance(row.get("base_seed"), bool)
            or not isinstance(row.get("base_seed"), int)
        ):
            raise PilotPipelineError("challenger phase row is malformed")
        previous = game_ordinal
        lines.append(f"{game_ordinal}\t{row['actor_mode']}\t{row['base_seed']}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _pipeline_paths(root: pathlib.Path) -> dict[str, str]:
    return {
        "root": str(root),
        "plan": str(root / "pipeline-plan.json"),
        "selfsearch_plan": str(root / "selfsearch-plan.tsv"),
        "filtered_roots": str(root / "roots/train-validation-roots.tsv"),
        "filtered_roots_manifest": str(root / "roots/train-validation-roots.json"),
        "games": str(root / "games/games.tsv"),
        "games_manifest": str(root / "games/games.manifest.json"),
        "positions": str(root / "positions/positions.tsv"),
        "positions_manifest": str(root / "positions/positions.manifest.json"),
        "root_assignments": str(root / "positions/roots.json"),
        "shallow_actions": str(root / "labels/shallow-actions.jsonl"),
        "rank4_labels": str(root / "labels/rank4.jsonl"),
        "hard_positions": str(root / "positions/hard.tsv"),
        "hard_report": str(root / "positions/hard-selection.json"),
        "deep_actions": str(root / "labels/deep-actions.jsonl"),
        "merged_actions": str(root / "labels/merged-actions.jsonl"),
        "successor_labels": str(root / "labels/successor-labels.json"),
        "scalar_samples": str(root / "scalar/scalar-samples.jsonl"),
        "scalar_manifest": str(root / "scalar/scalar-pack.json"),
        "scalar_shards": str(root / "scalar/shards"),
        "scalar_train_reference": str(root / "scalar/train-shard-reference.json"),
        "scalar_validation_reference": str(
            root / "scalar/validation-shard-reference.json"
        ),
        "receipts": str(root / "receipts"),
    }


def _filter_roots(
    *, roots_tsv: pathlib.Path, roots_manifest: pathlib.Path,
    output_tsv: pathlib.Path, output_manifest: pathlib.Path,
) -> tuple[dict[str, object], dict[str, str]]:
    manifest = replay_pack.load_roots(roots_manifest.resolve())
    assignments = replay_pack.frozen_assignments(manifest)
    accepted_by_group = {
        str(record["group_id"]): record for record in manifest["accepted"]
    }
    lines = roots_tsv.read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["group_id\tsource\twinner\ttranscript"]:
        raise PilotPipelineError("root TSV has the wrong schema")
    retained = [lines[0]]
    retained_groups = []
    seen = set()
    rejected_test = 0
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4 or not fields[0] or fields[0] in seen:
            raise PilotPipelineError("root TSV has a malformed or duplicate row")
        seen.add(fields[0])
        split = assignments.get(fields[0])
        frozen = accepted_by_group.get(fields[0], {})
        frozen_turns = frozen.get("turns")
        if not isinstance(frozen_turns, list) or any(
            not isinstance(turn, Mapping)
            or turn.get("player_id") != index % 2
            or not isinstance(turn.get("action"), str)
            for index, turn in enumerate(frozen_turns)
        ):
            raise PilotPipelineError("root manifest lacks exact replay lineage")
        frozen_transcript = "/".join(str(turn["action"]) for turn in frozen_turns)
        if (
            frozen.get("source") != fields[1]
            or frozen.get("winner") != int(fields[2])
            or frozen_transcript != fields[3]
        ):
            raise PilotPipelineError("root TSV differs from its frozen manifest")
        classification = str(frozen.get("classification", "")).casefold()
        if (
            split == "test"
            or frozen.get("protected") is True
            or classification.startswith("protected")
        ):
            rejected_test += 1
            continue
        if split not in {"train", "validation"}:
            raise PilotPipelineError("root TSV contains an unfrozen root group")
        retained.append(line)
        retained_groups.append({"group_id": fields[0], "split": split})
    if not retained_groups or not {item["split"] for item in retained_groups} == {
        "train", "validation"
    }:
        raise PilotPipelineError("filtered roots require nonempty train and validation")
    payload = ("\n".join(retained) + "\n").encode("utf-8")
    _write_once(output_tsv, payload)
    body: dict[str, object] = {
        "schema": corpus.ROOT_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "tool_sha256": {
            "normalizer": sha256_file(pathlib.Path(__file__)),
            "features": sha256_file(pathlib.Path(features.__file__)),
        },
        "exclusion_boundary": {"read_before_candidate_sources": True},
        "source_roots": _record(roots_tsv),
        "source_manifest": _record(roots_manifest),
        "accepted": retained_groups,
        "excluded": {"test_or_protected_roots": rejected_test},
        "counts": {
            "source": len(lines) - 1,
            "retained": len(retained_groups),
            "train": sum(item["split"] == "train" for item in retained_groups),
            "validation": sum(
                item["split"] == "validation" for item in retained_groups
            ),
            "test_or_protected_rejected": rejected_test,
        },
        "output_sha256": sha256_bytes(payload),
    }
    body["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    _write_once(output_manifest, canonical_json_bytes(body))
    replay_pack.load_roots(output_manifest)
    return body, {item["group_id"]: item["split"] for item in retained_groups}


def _exclusion_inputs(
    campaign: Mapping[str, Any],
    dynamic_exclusions: Sequence[Mapping[str, object]] = (),
) -> dict[str, dict[str, object]]:
    inputs = campaign["inputs"]
    sources: dict[str, dict[str, object]] = {}
    aliases = {
        "mixed-development-fingerprints": "mixed-development",
        "prior-train-fingerprints": "prior-train",
        "prior-validation-fingerprints": "prior-validation",
        "protected-fingerprints": "protected",
        "protected-canonical-fingerprints": "protected",
        "fresh-protected-fingerprints": "protected:fresh-protected-fingerprints",
    }
    for section in ("training_inputs", "protected_exclusions"):
        for name, record in inputs.get(section, {}).items():
            role = aliases.get(name)
            if (
                role is None
                and section == "protected_exclusions"
                and isinstance(name, str)
                and "fingerprint" in name.casefold()
            ):
                role = f"protected:{name}"
            if role is None:
                continue
            resolved = _bundle_path(campaign, record)
            normalized = _record(resolved)
            if role in sources and sources[role] != normalized:
                if role == "protected":
                    role = f"protected:{name}"
                if role in sources and sources[role] != normalized:
                    raise PilotPipelineError(
                        f"multiple frozen {role} exclusion sources"
                    )
            sources[role] = normalized
    live = inputs.get("live_exclusions")
    if not isinstance(live, Mapping):
        raise PilotPipelineError("campaign live exclusions are malformed")
    for name, record in sorted(live.items()):
        if not isinstance(name, str) or not name:
            raise PilotPipelineError("campaign live exclusion name is malformed")
        role = f"live:{name}"
        sources[role] = _record(_bundle_path(campaign, record))
    for ordinal, record in enumerate(dynamic_exclusions):
        try:
            path = challenger._verify_dynamic_exclusion_record(
                record, f"phase dynamic exclusion {ordinal}"
            )
            exclusion = challenger.validate_dynamic_exclusion(path)
        except Exception as error:
            raise PilotPipelineError("phase dynamic exclusion is invalid") from error
        role = (
            f"dynamic:{exclusion['classification']}:"
            f"{exclusion['body_sha256']}"
        )
        if role in sources:
            raise PilotPipelineError("phase dynamic exclusion is duplicated")
        sources[role] = _record(path)
    required = {"mixed-development", "prior-train", "prior-validation"}
    if not required.issubset(sources) or not any(
        role == "protected" or role.startswith("protected:") for role in sources
    ):
        raise PilotPipelineError(
            "campaign inputs must freeze mixed, prior train/validation, and protected fingerprints"
        )
    return sources


def _prior_shard_inputs(
    campaign: Mapping[str, Any],
) -> dict[str, list[dict[str, object]]]:
    exposed = campaign["inputs"].get("training_bundle", {}).get("exposed_routes", {})
    if isinstance(exposed, Mapping):
        routed = {
            "train": exposed.get("canonical_train"),
            "validation": exposed.get("canonical_validation"),
        }
        if all(
            isinstance(records, list) and records
            for records in routed.values()
        ):
            return {
                split: [
                    _record(_bundle_path(campaign, record))
                    for record in records
                ]
                for split, records in routed.items()
            }
    aliases = {
        "prior-train-manifest": "train",
        "prior-train-shard-manifest": "train",
        "prior-validation-manifest": "validation",
        "prior-validation-shard-manifest": "validation",
    }
    result: dict[str, list[dict[str, object]]] = {
        "train": [], "validation": []
    }
    for name, record in campaign["inputs"].get("training_inputs", {}).items():
        split = aliases.get(name)
        if split is None and isinstance(name, str):
            if name.startswith("prior-train-manifest-"):
                split = "train"
            elif name.startswith("prior-validation-manifest-"):
                split = "validation"
        if split is None:
            continue
        normalized = _record(_bundle_path(campaign, record))
        if normalized in result[split]:
            raise PilotPipelineError(f"prior {split} shard manifest is repeated")
        result[split].append(normalized)
    if any(not records for records in result.values()):
        raise PilotPipelineError("campaign inputs must freeze prior train/validation shards")
    return result


def _training_bundle_identity(
    campaign: Mapping[str, Any],
) -> tuple[dict[str, object], str]:
    inputs = campaign["inputs"]
    manifest_record = inputs.get("training_bundle", {}).get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise PilotPipelineError("campaign training bundle manifest is absent")
    path = _bundle_path(campaign, manifest_record)
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotPipelineError("campaign training bundle manifest is unreadable") from error
    body = dict(document) if isinstance(document, Mapping) else {}
    claimed = body.pop("body_sha256", None)
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != "papersoccer.compact-value-bfm-input-bundle.v1"
        or payload != canonical_json_bytes(document)
        or not isinstance(claimed, str)
        or claimed != sha256_bytes(canonical_json_bytes(body))
        or manifest_record.get("body_sha256") != claimed
    ):
        raise PilotPipelineError("campaign training bundle identity changed")
    return _record(path), claimed


def prepare_pipeline(
    *,
    campaign_plan: pathlib.Path,
    phase_reference: pathlib.Path,
    output_root: pathlib.Path,
    student_runtime: pathlib.Path | None = None,
    roots_tsv: pathlib.Path | None = None,
    roots_manifest: pathlib.Path | None = None,
    game_producer: pathlib.Path | None = None,
    action_teacher: pathlib.Path | None = None,
    rank4_teacher: pathlib.Path | None = None,
    created_at_utc: str,
    campaign_context: Mapping[str, Any] | None = None,
    phase_context: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    campaign, phase = _phase_context(
        campaign_plan, phase_reference, campaign_context, phase_context
    )
    phase_name = str(phase["phase"]["phase"])
    attempt = int(phase["phase"]["attempt"])
    root = output_root.resolve() / f"attempt-{attempt:03d}" / phase_name
    root.mkdir(parents=True, exist_ok=True)
    paths = _pipeline_paths(root)
    rows = phase["phase"]["rows"]
    inputs = campaign["inputs"]
    teacher_runtime = _bundle_path(campaign, inputs["teacher"]["runtime"])
    rank4_source = _bundle_path(campaign, inputs["rank4_teacher"])
    phase_inputs = phase["phase"].get("attempt_inputs")
    producer_bindings = phase["phase"].get("producer_binaries")
    dynamic_exclusions = phase["phase"].get("dynamic_exclusions")
    if (
        not isinstance(phase_inputs, Mapping)
        or set(phase_inputs) != {
            "student_runtime", "prior_runtime", "initial_float_checkpoint",
            "roots_tsv", "roots_manifest", "build_manifest",
            "producer_binaries",
        }
        or not isinstance(producer_bindings, Mapping)
        or set(producer_bindings) != challenger.BUILD_BINARY_ROLES
        or not isinstance(dynamic_exclusions, list)
        or phase_inputs.get("producer_binaries") != producer_bindings
    ):
        raise PilotPipelineError("phase lacks frozen attempt/build/exclusion inputs")
    student = _bundle_path(campaign, phase_inputs["student_runtime"])
    prior_runtime = _bundle_path(campaign, phase_inputs["prior_runtime"])
    bound_roots_tsv = _bundle_path(campaign, phase_inputs["roots_tsv"])
    bound_roots_manifest = _bundle_path(campaign, phase_inputs["roots_manifest"])
    bound_game_producer = _bundle_path(
        campaign, producer_bindings["continuation_producer"]
    )
    bound_action_teacher = _bundle_path(campaign, producer_bindings["action_teacher"])
    bound_rank4_teacher = _bundle_path(
        campaign, producer_bindings["rank4_position_teacher"]
    )
    bound_rank4_gate = _bundle_path(campaign, producer_bindings["rank4_gate"])
    provided = {
        "student_runtime": student_runtime,
        "roots_tsv": roots_tsv,
        "roots_manifest": roots_manifest,
        "game_producer": game_producer,
        "action_teacher": action_teacher,
        "rank4_teacher": rank4_teacher,
    }
    bound = {
        "student_runtime": student,
        "roots_tsv": bound_roots_tsv,
        "roots_manifest": bound_roots_manifest,
        "game_producer": bound_game_producer,
        "action_teacher": bound_action_teacher,
        "rank4_teacher": bound_rank4_teacher,
    }
    for name, supplied in provided.items():
        if supplied is not None and supplied.resolve() != bound[name].resolve():
            raise PilotPipelineError(f"{name} override differs from frozen phase input")
    # The challenger validator is the architecture authority and is deliberately
    # reused instead of inventing a second compact-runtime interpretation.
    if challenger._architecture(student).get("id") != challenger.ARCHITECTURE:
        raise PilotPipelineError("student runtime changed the frozen architecture")
    for path in (
        bound_roots_tsv, bound_roots_manifest, bound_game_producer,
        bound_action_teacher, bound_rank4_teacher,
        bound_rank4_gate,
    ):
        _record(path)
    try:
        build_source_closure = challenger.verify_phase_build_source_closure(
            required_sources=PIPELINE_REQUIRED_BUILD_SOURCES,
            campaign_context=campaign,
            phase_context=phase,
        )
    except Exception as error:
        raise PilotPipelineError(
            "phase code differs from its frozen build source closure"
        ) from error
    exclusion_sources = _exclusion_inputs(campaign, dynamic_exclusions)
    prior_shards = _prior_shard_inputs(campaign)
    training_bundle_manifest, source_bundle = _training_bundle_identity(campaign)
    campaign_input = inputs.get("body_sha256")
    if (
        not isinstance(campaign_input, str)
        or len(campaign_input) != 64
        or any(character not in "0123456789abcdef" for character in campaign_input)
    ):
        raise PilotPipelineError("challenger input bundle has no body identity")
    # Do not publish even intermediate preparation artifacts until every
    # external binding has passed validation.
    schedule = _render_selfsearch_plan(rows)
    filtered_manifest, root_splits = _filter_roots(
        roots_tsv=bound_roots_tsv, roots_manifest=bound_roots_manifest,
        output_tsv=pathlib.Path(paths["filtered_roots"]),
        output_manifest=pathlib.Path(paths["filtered_roots_manifest"]),
    )
    _write_once(pathlib.Path(paths["selfsearch_plan"]), schedule)
    body = {
        "schema": PIPELINE_SCHEMA,
        "campaign_id": phase["phase"]["campaign_id"],
        "attempt": attempt,
        "phase": phase_name,
        "created_at_utc": created_at_utc,
        "campaign_plan": _record(campaign_plan),
        "phase_reference": _record(phase_reference),
        "phase_plan": _record(pathlib.Path(phase["path"])),
        "source_bundle_body_sha256": source_bundle,
        "campaign_input_body_sha256": campaign_input,
        "inputs": {
            "accepted_teacher_runtime": _record(teacher_runtime),
            "rank4_source": _record(rank4_source),
            "prior_runtime": _record(prior_runtime),
            "student_runtime": _record(student),
            "source_roots_tsv": _record(bound_roots_tsv),
            "source_roots_manifest": _record(bound_roots_manifest),
            "filtered_roots_tsv": _record(pathlib.Path(paths["filtered_roots"])),
            "filtered_roots_manifest": _record(
                pathlib.Path(paths["filtered_roots_manifest"])
            ),
            "exclusion_sources": exclusion_sources,
            "prior_shard_manifests": prior_shards,
            "training_bundle_manifest": training_bundle_manifest,
        },
        "producers": {
            "games": _record(bound_game_producer),
            "action_teacher": _record(bound_action_teacher),
            "rank4_teacher": _record(bound_rank4_teacher),
            "rank4_gate": _record(bound_rank4_gate),
        },
        "build_source_closure": build_source_closure,
        "game_plan": {
            "schema": GAME_PLAN_SCHEMA,
            "phase": phase_name,
            "games": len(rows),
            "rows": [
                {
                    "game_ordinal": row["game_ordinal"],
                    "game_id": row["game_id"],
                    "actor_mode": row["actor_mode"],
                    "base_seed": row["base_seed"],
                    "logical_shard": row["worker"],
                }
                for row in rows
            ],
            "schedule": _record(pathlib.Path(paths["selfsearch_plan"])),
            "root_splits": root_splits,
            "filtered_root_counts": filtered_manifest["counts"],
        },
        "phase_input_binding": {
            "attempt_inputs": dict(phase_inputs),
            "producer_binaries": dict(producer_bindings),
            "dynamic_exclusions": list(dynamic_exclusions),
        },
        "policy": {
            "game_workers_max": GAME_WORKERS_MAX,
            "threads_per_worker": 1,
            "positions_per_game_max": POSITIONS_PER_GAME_MAX,
            "position_splits": ["train", "validation"],
            "hard_fraction": [HARD_NUMERATOR, HARD_DENOMINATOR],
            "shallow_tree_nodes": SHALLOW_TREE_NODES,
            "deep_tree_nodes": DEEP_TREE_NODES,
            "rank4_tree_nodes": RANK4_TREE_NODES,
            "protected_tests_opened": False,
            "thread_environment": THREAD_ENVIRONMENT,
        },
        "outputs": paths,
        "tool": _record(pathlib.Path(__file__)),
    }
    plan_path = pathlib.Path(paths["plan"])
    _write_sealed(plan_path, body)
    load_pipeline(plan_path)
    return plan_path


def load_pipeline(path: pathlib.Path) -> dict[str, Any]:
    plan = _load_sealed(path.resolve(), PIPELINE_SCHEMA, "pilot pipeline")
    outputs = plan.get("outputs")
    if not isinstance(outputs, dict) or outputs != _pipeline_paths(path.parent.resolve()):
        raise PilotPipelineError("pilot pipeline output routing changed")
    for label in ("campaign_plan", "phase_reference", "phase_plan", "tool"):
        _validate_record(plan.get(label), label)
    for section in ("inputs", "producers"):
        values = plan.get(section)
        if not isinstance(values, dict):
            raise PilotPipelineError(f"pipeline {section} are missing")
        for label, record in values.items():
            if section == "inputs" and label == "exclusion_sources":
                if not isinstance(record, dict):
                    raise PilotPipelineError(f"pipeline {label} are missing")
                for role, source in record.items():
                    _validate_record(source, f"{label}.{role}")
            elif section == "inputs" and label == "prior_shard_manifests":
                if (
                    not isinstance(record, dict)
                    or set(record) != {"train", "validation"}
                ):
                    raise PilotPipelineError("pipeline prior shards are missing")
                for split, sources in record.items():
                    if not isinstance(sources, list) or not sources:
                        raise PilotPipelineError(
                            f"pipeline prior {split} shards are missing"
                        )
                    for ordinal, source in enumerate(sources):
                        _validate_record(
                            source, f"prior_shard_manifests.{split}.{ordinal}"
                        )
            else:
                _validate_record(record, f"{section}.{label}")
    try:
        challenger.verify_phase_build_source_closure(
            required_sources=PIPELINE_REQUIRED_BUILD_SOURCES,
            stored_closure=plan.get("build_source_closure"),
        )
    except Exception as error:
        raise PilotPipelineError(
            "pipeline code differs from its frozen build source closure"
        ) from error
    schedule_path = _validate_record(
        plan.get("game_plan", {}).get("schedule"), "game schedule"
    )
    policy = plan.get("policy")
    rows = plan.get("game_plan", {}).get("rows")
    phase_binding = plan.get("phase_input_binding")
    if (
        plan.get("phase") not in challenger.PHASE_TOTALS
        or any(
            not isinstance(plan.get(field), str)
            or len(plan[field]) != 64
            or any(character not in "0123456789abcdef" for character in plan[field])
            for field in (
                "source_bundle_body_sha256", "campaign_input_body_sha256"
            )
        )
        or not isinstance(policy, dict)
        or policy.get("game_workers_max") != 8
        or policy.get("threads_per_worker") != 1
        or policy.get("hard_fraction") != [1, 4]
        or policy.get("protected_tests_opened") is not False
        or not isinstance(rows, list)
        or len(rows) != challenger.PHASE_TOTALS[plan["phase"]]
        or any(not isinstance(row, Mapping) for row in rows)
        or plan.get("game_plan", {}).get("phase") != plan["phase"]
        or plan.get("game_plan", {}).get("games") != len(rows)
        or Counter(str(row.get("actor_mode")) for row in rows)
        != Counter(challenger.PHASE_QUOTAS[plan["phase"]])
        or not isinstance(phase_binding, Mapping)
        or set(phase_binding) != {
            "attempt_inputs", "producer_binaries", "dynamic_exclusions"
        }
        or not isinstance(phase_binding.get("dynamic_exclusions"), list)
    ):
        raise PilotPipelineError("teacher phase pipeline policy changed")
    if schedule_path.read_bytes() != _render_selfsearch_plan(rows):
        raise PilotPipelineError("teacher phase rows and native schedule diverged")
    attempt_inputs = phase_binding["attempt_inputs"]
    producer_binaries = phase_binding["producer_binaries"]
    if (
        not isinstance(attempt_inputs, Mapping)
        or not isinstance(producer_binaries, Mapping)
        or set(producer_binaries) != challenger.BUILD_BINARY_ROLES
    ):
        raise PilotPipelineError("pipeline phase input binding is malformed")
    expected_hashes = {
        "student_runtime": plan["inputs"]["student_runtime"]["sha256"],
        "prior_runtime": plan["inputs"]["prior_runtime"]["sha256"],
        "roots_tsv": plan["inputs"]["source_roots_tsv"]["sha256"],
        "roots_manifest": plan["inputs"]["source_roots_manifest"]["sha256"],
        "build_manifest": plan["build_source_closure"]["manifest"]["sha256"],
        "continuation_producer": plan["producers"]["games"]["sha256"],
        "action_teacher": plan["producers"]["action_teacher"]["sha256"],
        "rank4_position_teacher": plan["producers"]["rank4_teacher"]["sha256"],
        "rank4_gate": plan["producers"]["rank4_gate"]["sha256"],
    }
    observed_hashes = {
        "student_runtime": attempt_inputs.get("student_runtime", {}).get("sha256"),
        "prior_runtime": attempt_inputs.get("prior_runtime", {}).get("sha256"),
        "roots_tsv": attempt_inputs.get("roots_tsv", {}).get("sha256"),
        "roots_manifest": attempt_inputs.get("roots_manifest", {}).get("sha256"),
        "build_manifest": attempt_inputs.get("build_manifest", {}).get("sha256"),
        "continuation_producer": producer_binaries.get(
            "continuation_producer", {}
        ).get("sha256"),
        "action_teacher": producer_binaries.get("action_teacher", {}).get("sha256"),
        "rank4_position_teacher": producer_binaries.get(
            "rank4_position_teacher", {}
        ).get("sha256"),
        "rank4_gate": producer_binaries.get("rank4_gate", {}).get("sha256"),
    }
    if observed_hashes != expected_hashes:
        raise PilotPipelineError("pipeline artifacts differ from frozen phase bindings")
    dynamic_hashes = {
        record.get("sha256") for record in phase_binding["dynamic_exclusions"]
        if isinstance(record, Mapping)
    }
    frozen_dynamic_hashes = {
        record.get("sha256")
        for role, record in plan["inputs"]["exclusion_sources"].items()
        if str(role).startswith("dynamic:")
    }
    if dynamic_hashes != frozen_dynamic_hashes or len(dynamic_hashes) != len(
        phase_binding["dynamic_exclusions"]
    ):
        raise PilotPipelineError("pipeline dynamic exclusion binding changed")
    return plan


def _stage_receipt_path(plan: Mapping[str, Any], stage: str) -> pathlib.Path:
    return pathlib.Path(plan["outputs"]["receipts"]) / f"{stage}.json"


def _reuse_stage(
    plan: Mapping[str, Any], stage: str, inputs: Mapping[str, object], resume: bool
) -> dict[str, Any] | None:
    path = _stage_receipt_path(plan, stage)
    if not path.exists():
        return None
    if not resume:
        raise PilotPipelineError(f"stage {stage} is complete; use --resume")
    receipt = _load_sealed(path, STAGE_RECEIPT_SCHEMA, f"{stage} receipt")
    if (
        receipt.get("pipeline_body_sha256") != plan["body_sha256"]
        or receipt.get("attempt") != plan["attempt"]
        or receipt.get("phase") != plan["phase"]
        or receipt.get("inputs") != dict(inputs)
    ):
        raise PilotPipelineError(f"stage {stage} receipt inputs changed")
    for label, record in receipt.get("outputs", {}).items():
        _validate_record(record, f"{stage}.{label}")
    return receipt


def _finish_stage(
    plan: Mapping[str, Any],
    stage: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, pathlib.Path],
    details: Mapping[str, object],
) -> dict[str, object]:
    return _write_sealed(
        _stage_receipt_path(plan, stage),
        {
            "schema": STAGE_RECEIPT_SCHEMA,
            "pipeline_body_sha256": plan["body_sha256"],
            "attempt": plan["attempt"],
            "phase": plan["phase"],
            "stage": stage,
            "inputs": dict(inputs),
            "outputs": {name: _record(path) for name, path in outputs.items()},
            "details": dict(details),
        },
    )


GameProducer = Callable[[Sequence[str], pathlib.Path, pathlib.Path, Mapping[str, str]], None]
LabelProducer = Callable[[Sequence[str], pathlib.Path, pathlib.Path, Mapping[str, str]], None]


def _subprocess_producer(
    command: Sequence[str], input_path: pathlib.Path, output_path: pathlib.Path,
    environment: Mapping[str, str],
) -> None:
    with input_path.open("rb") as source, output_path.open("wb") as destination:
        completed = subprocess.run(
            list(command), stdin=source, stdout=destination,
            stderr=subprocess.PIPE, env={**os.environ, **environment}, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"producer exited {completed.returncode}: "
            + completed.stderr.decode("utf-8", "replace")
        )


def _subprocess_game_producer(
    command: Sequence[str], input_path: pathlib.Path, output_path: pathlib.Path,
    environment: Mapping[str, str],
) -> None:
    del input_path, output_path
    completed = subprocess.run(
        list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env={**os.environ, **environment}, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"game producer exited {completed.returncode}: "
            + completed.stderr.decode("utf-8", "replace")
        )


def _render_game_chunk(rows: Sequence[Mapping[str, object]]) -> bytes:
    return _render_selfsearch_plan(rows)


def _game_execution_profile(row: Mapping[str, object]) -> str:
    """Return the continuation-generator-compatible actor backend.

    The existing native continuation generator deliberately rejects a compact
    runtime invocation if even one plan row has no student actor.  A phase's
    logical shards mix those rows with accepted-teacher-vs-Rank-4 rows, so a
    logical shard must be executed as two profile-homogeneous plan chunks.
    """

    mode = row.get("actor_mode")
    if mode in COMPACT_GAME_MODES:
        return "compact"
    if mode in INCUMBENT_GAME_MODES:
        return "incumbent"
    raise PilotPipelineError(f"teacher phase has unsupported actor mode: {mode!r}")


def _parse_game_chunk(
    rows: Sequence[Mapping[str, object]], games: pathlib.Path,
    manifest_path: pathlib.Path, *, input_path: pathlib.Path,
    plan: Mapping[str, Any], profile: str,
) -> list[dict[str, object]]:
    lines = games.read_text(encoding="utf-8").splitlines()
    if lines[:1] != [GAME_HEADER] or len(lines) != len(rows) + 1:
        raise PilotPipelineError("game producer output has the wrong row count")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PilotPipelineError("game producer manifest is unreadable") from error
    produced = manifest.get("rows") if isinstance(manifest, dict) else None
    configuration = manifest.get("configuration") if isinstance(manifest, dict) else None
    bindings = manifest.get("bindings") if isinstance(manifest, dict) else None
    compact = profile == "compact"
    if (
        not isinstance(produced, list)
        or len(produced) != len(rows)
        or manifest.get("schema") != "papersoccer.jacek-selfsearch-games.v1"
        or manifest.get("campaign_id") != plan["campaign_id"]
        or manifest.get("requested_games") != len(rows)
        or manifest.get("successful_games") != len(rows)
        or not isinstance(configuration, Mapping)
        or not isinstance(bindings, Mapping)
        or configuration.get("bfm_tree_nodes") != 8_000
        or configuration.get("rank4_nodes") != 16_000
        or configuration.get("jacek_nn_nodes") != 64_000
        or configuration.get("exploration") != 0.5
        or configuration.get("fpu") != 0.5
        or bindings.get("roots_sha256")
        != plan["inputs"]["filtered_roots_tsv"]["sha256"]
        or bindings.get("plan_sha256") != sha256_file(input_path)
        or bindings.get("output_sha256") != sha256_file(games)
        or bindings.get("incumbent_model_sha256")
        != plan["inputs"]["accepted_teacher_runtime"]["sha256"]
        or bindings.get("runner_up_model_sha256")
        != plan["inputs"]["accepted_teacher_runtime"]["sha256"]
        or (
            compact
            and (
                configuration.get("actor_backend")
                != "compact-value-bfm-runtime-v1"
                or configuration.get("minimum_post_prefix_turns") != 20
                or bindings.get("compact_student_runtime_sha256")
                != plan["inputs"]["student_runtime"]["sha256"]
                or bindings.get("compact_prior_runtime_sha256")
                != plan["inputs"]["prior_runtime"]["sha256"]
            )
        )
        or (
            not compact
            and (
                "actor_backend" in configuration
                or any(str(key).startswith("compact_") for key in bindings)
            )
        )
    ):
        raise PilotPipelineError("game producer manifest rows are incomplete")
    normalized = []
    for planned, line, evidence in zip(rows, lines[1:], produced, strict=True):
        fields = line.split("\t")
        if len(fields) != 4 or fields[2] not in {"0", "1"} or not fields[3]:
            raise PilotPipelineError("game producer emitted a malformed game")
        if (
            not isinstance(evidence, dict)
            or evidence.get("game_ordinal") != planned["game_ordinal"]
            or evidence.get("base_seed") != planned["base_seed"]
            or evidence.get("actor_mode") != planned["actor_mode"]
            or evidence.get("root_group_id") != fields[0]
            or evidence.get("winner") != int(fields[2])
            or evidence.get("transcript_sha256")
            != sha256_bytes(fields[3].encode("ascii"))
            or not isinstance(evidence.get("prefix_turns"), int)
            or not 0 <= evidence["prefix_turns"] < len(fields[3].split("/"))
        ):
            raise PilotPipelineError("game producer manifest disagrees with its TSV")
        normalized.append(
            {
                "game_ordinal": planned["game_ordinal"],
                "game_id": planned["game_id"],
                "actor_mode": planned["actor_mode"],
                "base_seed": planned["base_seed"],
                "root_group_id": fields[0],
                "source": fields[1],
                "winner": int(fields[2]),
                "transcript": fields[3],
                "prefix_turns": evidence["prefix_turns"],
            }
        )
    return normalized


def run_game_chunks(
    plan_path: pathlib.Path, *, workers: int = 8, resume: bool = False,
    producer: GameProducer | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    expected_games = challenger.PHASE_TOTALS[plan["phase"]]
    if isinstance(workers, bool) or not 1 <= workers <= GAME_WORKERS_MAX:
        raise PilotPipelineError("game workers must be in 1..8")
    stage_inputs = {
        "phase_plan": plan["phase_plan"],
        "schedule": plan["game_plan"]["schedule"],
        "roots": plan["inputs"]["filtered_roots_tsv"],
        "student": plan["inputs"]["student_runtime"],
        "prior": plan["inputs"]["prior_runtime"],
        "teacher": plan["inputs"]["accepted_teacher_runtime"],
        "producer": plan["producers"]["games"],
        "workers": workers,
    }
    reused = _reuse_stage(plan, "01-games", stage_inputs, resume)
    if reused is not None:
        return reused
    rows = plan["game_plan"]["rows"]
    shards: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        logical_shard = int(row["logical_shard"])
        shards[(logical_shard, _game_execution_profile(row))].append(row)
    if {logical_shard for logical_shard, _profile in shards} != set(range(10)):
        raise PilotPipelineError("teacher-phase logical game shard roster changed")
    if {str(row["actor_mode"]) for row in rows} != PILOT_GAME_MODES:
        raise PilotPipelineError("teacher-phase actor-mode roster changed")
    root = pathlib.Path(plan["outputs"]["root"]) / "games/chunks"
    runner = _subprocess_game_producer if producer is None else producer
    environment = dict(THREAD_ENVIRONMENT)
    allowed_roots = set(plan["game_plan"]["root_splits"])

    def parse_chunk(
        shard_rows: Sequence[Mapping[str, object]],
        games_path: pathlib.Path,
        manifest_path: pathlib.Path,
        input_path: pathlib.Path,
        profile: str,
    ) -> list[dict[str, object]]:
        normalized = _parse_game_chunk(
            shard_rows, games_path, manifest_path,
            input_path=input_path, plan=plan, profile=profile,
        )
        if any(
            str(row["root_group_id"]) not in allowed_roots
            or row["source"] != plan["campaign_id"]
            for row in normalized
        ):
            raise PilotPipelineError(
                "game producer changed its campaign or frozen root roster"
            )
        return normalized

    def execute(
        item: tuple[tuple[int, str], list[dict[str, object]]]
    ) -> dict[str, object]:
        (ordinal, profile), shard_rows = item
        directory = root / f"shard-{ordinal:02d}-{profile}"
        input_path = directory / "plan.tsv"
        games_path = directory / "games.tsv"
        manifest_path = directory / "manifest.json"
        receipt_path = directory / "receipt.json"
        _write_once(input_path, _render_game_chunk(shard_rows))
        inputs = {
            "plan": _record(input_path),
            "execution_profile": profile,
            "roots": plan["inputs"]["filtered_roots_tsv"],
            "student": plan["inputs"]["student_runtime"],
            "prior": plan["inputs"]["prior_runtime"],
            "teacher": plan["inputs"]["accepted_teacher_runtime"],
            "producer": plan["producers"]["games"],
        }
        if receipt_path.exists():
            if not resume:
                raise PilotPipelineError("game shard is complete; use --resume")
            receipt = _load_sealed(receipt_path, STAGE_RECEIPT_SCHEMA, "game shard")
            if (
                receipt.get("pipeline_body_sha256") != plan["body_sha256"]
                or receipt.get("attempt") != plan["attempt"]
                or receipt.get("phase") != plan["phase"]
                or receipt.get("inputs") != inputs
            ):
                raise PilotPipelineError("game shard receipt inputs changed")
            parse_chunk(shard_rows, games_path, manifest_path, input_path, profile)
            return receipt
        if games_path.exists() or manifest_path.exists():
            raise PilotPipelineError("game shard has output without a receipt")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = pathlib.Path(tempfile.mkdtemp(dir=directory, prefix=".running-"))
        try:
            staged_games = temporary / "games.tsv"
            staged_manifest = temporary / "manifest.json"
            command = [
                str(_validate_record(plan["producers"]["games"], "game producer")),
                "--input", str(_validate_record(plan["inputs"]["filtered_roots_tsv"], "roots")),
                "--output", str(staged_games), "--manifest", str(staged_manifest),
                "--model", str(_validate_record(plan["inputs"]["accepted_teacher_runtime"], "teacher")),
                "--runner-up-model", str(_validate_record(plan["inputs"]["accepted_teacher_runtime"], "teacher")),
                "--selfsearch-plan", str(input_path), "--campaign-id", plan["campaign_id"],
                "--games", str(len(shard_rows)), "--actor-nodes", "16000",
                "--candidate-tree-nodes", "8000", "--jacek-nn-nodes", "64000",
                "--candidate-exploration", "0.5", "--candidate-fpu", "0.5",
            ]
            if profile == "compact":
                command.extend([
                    "--compact-student-runtime",
                    str(_validate_record(plan["inputs"]["student_runtime"], "student")),
                    "--compact-prior-runtime",
                    str(_validate_record(plan["inputs"]["prior_runtime"], "prior")),
                ])
            runner(command, input_path, staged_games, environment)
            # Existing producer writes its manifest via --manifest rather than stdout.
            if not staged_manifest.exists():
                raise PilotPipelineError("game producer omitted its manifest")
            normalized = parse_chunk(
                shard_rows, staged_games, staged_manifest, input_path, profile
            )
            os.replace(staged_games, games_path)
            os.replace(staged_manifest, manifest_path)
            receipt = _write_sealed(
                receipt_path,
                {
                    "schema": STAGE_RECEIPT_SCHEMA,
                    "pipeline_body_sha256": plan["body_sha256"],
                    "attempt": plan["attempt"],
                    "phase": plan["phase"],
                    "stage": f"01-games-shard-{ordinal:02d}-{profile}",
                    "inputs": inputs,
                    "outputs": {"games": _record(games_path), "manifest": _record(manifest_path)},
                    "details": {
                        "rows": len(normalized),
                        "worker_threads": 1,
                        "execution_profile": profile,
                    },
                },
            )
            return receipt
        finally:
            for child in temporary.iterdir() if temporary.exists() else ():
                child.unlink(missing_ok=True)
            temporary.rmdir() if temporary.exists() else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(execute, sorted(shards.items())))
    normalized_rows: list[dict[str, object]] = []
    shard_records = []
    for (ordinal, profile), shard_rows in sorted(shards.items()):
        directory = root / f"shard-{ordinal:02d}-{profile}"
        normalized_rows.extend(
            parse_chunk(
                shard_rows,
                directory / "games.tsv",
                directory / "manifest.json",
                directory / "plan.tsv",
                profile,
            )
        )
        shard_records.append(_record(directory / "receipt.json"))
    normalized_rows.sort(key=lambda row: int(row["game_ordinal"]))
    if [row["game_ordinal"] for row in normalized_rows] != list(range(expected_games)):
        raise PilotPipelineError("merged game chunks do not cover the frozen phase schedule")
    games_path = pathlib.Path(plan["outputs"]["games"])
    manifest_path = pathlib.Path(plan["outputs"]["games_manifest"])
    game_payload = (GAME_HEADER + "\n" + "\n".join(
        f"{row['root_group_id']}\t{row['source']}\t{row['winner']}\t{row['transcript']}"
        for row in normalized_rows
    ) + "\n").encode("utf-8")
    _write_once(games_path, game_payload)
    manifest = _write_sealed(
        manifest_path,
        {
            "schema": GAME_MANIFEST_SCHEMA,
            "pipeline_body_sha256": plan["body_sha256"],
            "phase": plan["phase"],
            "attempt": plan["attempt"],
            "games": expected_games,
            "shards": shard_records,
            "rows": normalized_rows,
            "games_sha256": sha256_bytes(game_payload),
        },
    )
    return _finish_stage(
        plan, "01-games", stage_inputs,
        {"games": games_path, "manifest": manifest_path},
        {
            "games": manifest["games"],
            "logical_shards": 10,
            "execution_chunks": len(shards),
            "workers": workers,
        },
    )


def _phase_splits(plan: Mapping[str, Any], rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    del rows
    splits = plan.get("game_plan", {}).get("root_splits")
    if not isinstance(splits, dict) or not splits or any(
        not isinstance(group, str) or split not in {"train", "validation"}
        for group, split in splits.items()
    ):
        raise PilotPipelineError("filtered root split roster is malformed")
    return {str(group): str(split) for group, split in splits.items()}


def _fingerprint_values(path: pathlib.Path, role: str) -> set[str]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotPipelineError(f"{role} fingerprint source is unreadable") from error
    if not isinstance(value, dict):
        raise PilotPipelineError(f"{role} fingerprint source is not an object")
    schema = value.get("schema")
    result: set[str] = set()
    if schema == FINGERPRINT_SET_SCHEMA:
        body = dict(value)
        claimed = body.pop("body_sha256", None)
        if claimed != sha256_bytes(canonical_json_bytes(body)):
            raise PilotPipelineError(f"{role} fingerprint-set body changed")
        fingerprints = value.get("fingerprints")
        source_records = value.get("sources")
        if (
            set(value) != {
                "schema", "classification", "canonicalization",
                "fingerprint_domain", "sources", "fingerprints",
                "fingerprint_count", "source_paths_followed",
                "contains_labels", "contains_metrics", "contains_transcripts",
                "body_sha256",
            }
            or not isinstance(source_records, list)
            or any(
                not isinstance(record, Mapping)
                or set(record) != {
                    "path", "bytes", "sha256", "canonical_fingerprint_count"
                }
                or not isinstance(record.get("path"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] < 0
                or not isinstance(record.get("canonical_fingerprint_count"), int)
                or record["canonical_fingerprint_count"] < 1
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in record["sha256"]
                )
                for record in source_records
            )
            or not isinstance(fingerprints, list)
            or any(not isinstance(item, str) for item in fingerprints)
            or fingerprints != sorted(set(fingerprints))
            or value.get("fingerprint_count") != len(fingerprints)
            or value.get("canonicalization") != FOUR_WAY_CANONICALIZATION
            or value.get("fingerprint_domain") != FEATURE_FINGERPRINT_DOMAIN
            or value.get("source_paths_followed") is not False
            or value.get("contains_labels") is not False
            or value.get("contains_metrics") is not False
            or value.get("contains_transcripts") is not False
            or (
                role in {"mixed-development", "prior-train", "prior-validation"}
                and value.get("classification") != role
            )
        ):
            raise PilotPipelineError(f"{role} fingerprint set is malformed")
        result.update(str(item) for item in fingerprints)
    elif schema == "papersoccer.compact-value-bfm.discrete-v3-protected-canonical-fingerprints.v1":
        body = dict(value)
        claimed = body.pop("body_sha256", None)
        rows = value.get("rows")
        if (
            claimed != sha256_bytes(canonical_json_bytes(body))
            or not isinstance(rows, list)
            or len(rows) != value.get("position_count")
            or value.get("canonicalization")
            != "minimum-sha256-over-exact+rotate+reflect+rotate_reflect"
            or value.get("contains_labels") is not False
            or value.get("contains_metrics") is not False
            or value.get("contains_transcripts") is not False
        ):
            raise PilotPipelineError("protected fingerprint-only rows are absent")
        result.update(str(row.get("canonical_sha256")) for row in rows if isinstance(row, dict))
        if len(result) != value.get("unique_canonical_count"):
            raise PilotPipelineError("protected canonical fingerprint count changed")
    elif schema == challenger.DYNAMIC_EXCLUSION_SCHEMA:
        try:
            dynamic = challenger.validate_dynamic_exclusion(path)
        except Exception as error:
            raise PilotPipelineError(
                f"{role} dynamic fingerprint exclusion changed"
            ) from error
        result.update(str(item) for item in dynamic["fingerprints"])
    else:
        raise PilotPipelineError(f"{role} is not a fingerprint-only recognized schema")
    if not result or any(
        len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        for item in result
    ):
        raise PilotPipelineError(f"{role} fingerprints are empty or malformed")
    return result


def _fingerprint_domain(path: pathlib.Path, role: str) -> str:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotPipelineError(f"{role} fingerprint source is unreadable") from error
    if not isinstance(value, Mapping):
        raise PilotPipelineError(f"{role} fingerprint source is malformed")
    if value.get("schema") == FINGERPRINT_SET_SCHEMA:
        return FEATURE_FINGERPRINT_DOMAIN
    if value.get("schema") == (
        "papersoccer.compact-value-bfm.discrete-v3-protected-canonical-fingerprints.v1"
    ):
        return STATE_FINGERPRINT_DOMAIN
    if value.get("schema") == challenger.DYNAMIC_EXCLUSION_SCHEMA:
        return STATE_FINGERPRINT_DOMAIN
    raise PilotPipelineError(f"{role} fingerprint domain is unknown")


def _inline_source_fingerprints(
    path: pathlib.Path, classification: str
) -> set[str]:
    """Read fingerprints embedded in an explicit local source, never references."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PilotPipelineError("fingerprint-union source is unreadable") from error
    if raw.startswith(b"# papersoccer.jacek-replay-bfm-opening-bank.v1\n"):
        if classification != "mixed-development":
            raise PilotPipelineError(
                "historical opening exclusions may only feed mixed development"
            )
        try:
            historical = challenger.openings.load_exclusion_bank(path)
        except Exception as error:
            raise PilotPipelineError(
                "historical opening exclusion failed validation"
            ) from error
        result = set()
        for opening in historical["openings"]:
            state, _primitive_plies = challenger.openings.replay_transcript(
                opening["transcript"]
            )
            result.add(
                corpus.canonical_feature_fingerprint(
                    features.encode_active(state)
                ).hex()
            )
        return result
    try:
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotPipelineError("fingerprint-union source is unreadable") from error
    if not isinstance(value, dict):
        raise PilotPipelineError("fingerprint-union source is not an object")
    if value.get("schema") == FINGERPRINT_SET_SCHEMA:
        source_classification = value.get("classification")
        allowed = (
            isinstance(source_classification, str)
            and (
                source_classification == classification
                or (
                    classification == "mixed-development"
                    and source_classification.startswith("development")
                )
            )
        )
        if not allowed:
            raise PilotPipelineError(
                "fingerprint-set source classification crosses an exclusion boundary"
            )
        return _fingerprint_values(path, "fingerprint-union source")
    schema = value.get("schema")
    if schema == "papersoccer.jacek-replay-csr-shard.v1":
        try:
            from tools import jacek_replay_train as replay_train
        except ModuleNotFoundError as error:
            if error.name == "numpy":
                raise RuntimeError(
                    "CSR fingerprint materialization requires requirements-research.txt"
                ) from error
            raise
        shard = replay_train.load_csr_shard(path)
        expected_split = {
            "prior-train": "train",
            "prior-validation": "validation",
        }.get(classification)
        if shard.split == "test" or (
            expected_split is not None and shard.split != expected_split
        ):
            raise PilotPipelineError(
                "CSR fingerprint source is protected or has the wrong split"
            )
        return {
            corpus.canonical_feature_fingerprint(shard.active(row)).hex()
            for row in range(len(shard))
        }
    if not isinstance(schema, str) or "opening-bank" not in schema:
        raise PilotPipelineError(
            "fingerprint-union sources must be inline opening banks or fingerprint sets"
        )
    bank = challenger.openings.validate_bank(path)
    if (
        classification != "mixed-development"
        or bank.get("classification") != "unprotected-development"
    ):
        raise PilotPipelineError(
            "opening banks may only feed the unprotected mixed-development set"
        )
    result = set()
    for opening in bank["openings"]:
        state, _primitive_plies = challenger.openings.replay_transcript(
            opening.get("transcript")
        )
        result.add(
            corpus.canonical_feature_fingerprint(
                features.encode_active(state)
            ).hex()
        )
    if not result or any(
        len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        for item in result
    ):
        raise PilotPipelineError("opening bank has malformed canonical fingerprints")
    return result


def materialize_fingerprint_set(
    *,
    output_directory: pathlib.Path,
    classification: str,
    sources: Sequence[pathlib.Path],
) -> pathlib.Path:
    """Create a self-contained, content-addressed canonical fingerprint union."""

    if classification not in {
        "mixed-development", "prior-train", "prior-validation"
    } or not sources:
        raise PilotPipelineError("fingerprint-set classification or sources are invalid")
    records = []
    fingerprints: set[str] = set()
    resolved_sources = [path.resolve() for path in sources]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise PilotPipelineError("fingerprint-set sources are duplicated")
    for path in sorted(resolved_sources):
        source_record = _record(path)
        source_fingerprints = _inline_source_fingerprints(path, classification)
        records.append({
            **source_record,
            "canonical_fingerprint_count": len(source_fingerprints),
        })
        fingerprints.update(source_fingerprints)
    body: dict[str, object] = {
        "schema": FINGERPRINT_SET_SCHEMA,
        "classification": classification,
        "canonicalization": FOUR_WAY_CANONICALIZATION,
        "fingerprint_domain": FEATURE_FINGERPRINT_DOMAIN,
        "sources": records,
        "fingerprints": sorted(fingerprints),
        "fingerprint_count": len(fingerprints),
        "source_paths_followed": False,
        "contains_labels": False,
        "contains_metrics": False,
        "contains_transcripts": False,
    }
    body["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    payload = canonical_json_bytes(body)
    output = output_directory.resolve() / (
        f"{sha256_bytes(payload)}.fingerprint-set.json"
    )
    _write_once(output, payload)
    _fingerprint_values(output, classification)
    return output


def _exclusion_context(plan: Mapping[str, Any]) -> dict[str, object]:
    by_role: dict[str, set[str]] = {}
    domains: dict[str, str] = {}
    sources = plan["inputs"]["exclusion_sources"]
    expected = {"mixed-development", "prior-train", "prior-validation"}
    if (
        not isinstance(sources, Mapping)
        or not expected.issubset(sources)
        or not any(
            role == "protected" or str(role).startswith("protected:")
            for role in sources
        )
    ):
        raise PilotPipelineError("frozen fingerprint exclusion roster changed")
    for role in sorted(sources):
        path = _validate_record(sources[role], f"{role} exclusions")
        by_role[role] = _fingerprint_values(path, role)
        domains[role] = _fingerprint_domain(path, role)
    union_by_domain: dict[str, set[str]] = defaultdict(set)
    cross_source = 0
    for role in by_role:
        domain = domains[role]
        cross_source += len(union_by_domain[domain] & by_role[role])
        union_by_domain[domain].update(by_role[role])
    return {
        "by_role": by_role,
        "domains": domains,
        "union_by_domain": dict(union_by_domain),
        "sources": sources,
        "counts": {role: len(values) for role, values in by_role.items()},
        "cross_source_intersection_count": cross_source,
    }


def materialize_positions(plan_path: pathlib.Path, *, resume: bool = False) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    games_path = pathlib.Path(plan["outputs"]["games"])
    games_manifest_path = pathlib.Path(plan["outputs"]["games_manifest"])
    inputs = {
        "games": _record(games_path),
        "manifest": _record(games_manifest_path),
        "filtered_roots": plan["inputs"]["filtered_roots_manifest"],
        "exclusion_sources": plan["inputs"]["exclusion_sources"],
    }
    reused = _reuse_stage(plan, "02-positions", inputs, resume)
    if reused is not None:
        return reused
    manifest = _load_sealed(games_manifest_path, GAME_MANIFEST_SCHEMA, "pilot games")
    rows = manifest["rows"]
    expected_games = challenger.PHASE_TOTALS[plan["phase"]]
    if (
        manifest.get("pipeline_body_sha256") != plan["body_sha256"]
        or manifest.get("phase") != plan["phase"]
        or manifest.get("attempt") != plan["attempt"]
        or manifest.get("games") != expected_games
        or not isinstance(rows, list)
        or len(rows) != expected_games
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("game_ordinal") for row in rows]
        != list(range(expected_games))
    ):
        raise PilotPipelineError("game manifest does not cover its frozen phase")
    assignments = _phase_splits(plan, rows)
    exclusions = _exclusion_context(plan)
    excluded_by_role = exclusions["by_role"]
    exclusion_domains = exclusions["domains"]
    output_lines = [POSITION_HEADER]
    position_rows = []
    skipped = []
    seen_fingerprints: set[str] = set()
    fingerprints_by_split: dict[str, set[str]] = {
        "train": set(), "validation": set()
    }
    external_intersections = Counter()
    candidate_duplicate_intersections = 0
    stratum_names = ("opening", "middle", "late", "decisive")
    for game in rows:
        actions = str(game["transcript"]).split("/")
        prefix_turns = int(game["prefix_turns"])
        if not 0 <= prefix_turns < len(actions):
            raise PilotPipelineError("generated game prefix boundary is invalid")
        root_group_id = str(game["root_group_id"])
        if root_group_id not in assignments:
            raise PilotPipelineError("game producer used a root outside the filtered roster")
        split = assignments[root_group_id]
        state = features.ReplayState()
        prefix: list[str] = []
        candidates: dict[str, list[dict[str, object]]] = {
            name: [] for name in stratum_names
        }
        candidate_canonicals: set[str] = set()
        suffix_count = len(actions) - prefix_turns
        for turn, action in enumerate(actions):
            if turn >= prefix_turns:
                relative = turn - prefix_turns
                position_stratum = stratum_names[
                    min(3, relative * 4 // suffix_count)
                ]
                prefix_text = "/".join(prefix)
                fingerprints = challenger.openings.state_fingerprints(state)
                canonical = fingerprints["canonical"]
                feature_canonical = corpus.canonical_feature_fingerprint(
                    features.encode_active(state)
                ).hex()
                candidate_by_domain = {
                    STATE_FINGERPRINT_DOMAIN: canonical,
                    FEATURE_FINGERPRINT_DOMAIN: feature_canonical,
                }
                matched_roles = [
                    role
                    for role, values in excluded_by_role.items()
                    if candidate_by_domain[exclusion_domains[role]] in values
                ]
                if matched_roles:
                    for role in matched_roles:
                        external_intersections[role] += 1
                elif canonical in seen_fingerprints or canonical in candidate_canonicals:
                    candidate_duplicate_intersections += 1
                else:
                    digest = sha256_bytes(
                        canonical_json_bytes(
                            {
                                "pipeline": plan["body_sha256"],
                                "game_id": game["game_id"],
                                "root_group_id": root_group_id,
                                "turn": turn,
                                "prefix_sha256": sha256_bytes(
                                    prefix_text.encode()
                                ),
                                "canonical_fingerprint": canonical,
                            }
                        )
                    )
                    candidates[position_stratum].append({
                        "position_id": f"position:{digest}",
                        "prefix": prefix_text,
                        "turn": turn,
                        "mover": state.to_move,
                        "canonical_fingerprint": canonical,
                        "canonical_feature_fingerprint": feature_canonical,
                        "position_stratum": position_stratum,
                    })
                    candidate_canonicals.add(canonical)
            features.apply_complete_turn(state, state.to_move, action)
            prefix.append(action)
        if state.winner != game["winner"]:
            raise PilotPipelineError("generated game transcript has the wrong winner")
        per_stratum = min(5, *(len(candidates[name]) for name in stratum_names))
        if per_stratum == 0:
            skipped.append(game["game_id"])
            continue
        selected = []
        for name in stratum_names:
            bucket = candidates[name]
            selected.extend(
                bucket[index * len(bucket) // per_stratum]
                for index in range(per_stratum)
            )
        selected.sort(key=lambda item: int(item["turn"]))
        source = f"challenger-{plan['phase']}:{game['actor_mode']}"
        for candidate in selected:
            position_id = str(candidate["position_id"])
            output_lines.append("\t".join((
                    position_id, root_group_id, str(game["game_id"]), source, split,
                    str(game["winner"]), str(candidate["mover"]),
                    str(candidate["prefix"]),
            )))
            position_rows.append({
                "position_id": position_id,
                "root_group_id": root_group_id,
                "game_id": game["game_id"],
                "game_ordinal": game["game_ordinal"],
                "actor_mode": game["actor_mode"],
                "game_stratum": [game["actor_mode"], game["winner"]],
                "position_stratum": candidate["position_stratum"],
                "turn": candidate["turn"],
                "split": split,
                "winner": game["winner"],
                "mover": candidate["mover"],
                "canonical_fingerprint": candidate["canonical_fingerprint"],
                "canonical_feature_fingerprint": candidate[
                    "canonical_feature_fingerprint"
                ],
            })
            seen_fingerprints.add(str(candidate["canonical_fingerprint"]))
            fingerprints_by_split[split].add(
                str(candidate["canonical_fingerprint"])
            )
    by_game = Counter(str(row["game_id"]) for row in position_rows)
    by_root = Counter(str(row["root_group_id"]) for row in position_rows)
    if any(count > POSITIONS_PER_GAME_MAX or count % 4 for count in by_game.values()):
        raise PilotPipelineError("position freeze violated the per-game exact-quarter contract")
    train_validation_intersection = len(
        fingerprints_by_split["train"] & fingerprints_by_split["validation"]
    )
    if train_validation_intersection:
        raise PilotPipelineError("train and validation positions overlap by four-way symmetry")
    positions_payload = ("\n".join(output_lines) + "\n").encode("utf-8")
    positions_path = pathlib.Path(plan["outputs"]["positions"])
    _write_once(positions_path, positions_payload)
    positions_manifest_path = pathlib.Path(plan["outputs"]["positions_manifest"])
    positions_manifest = _write_sealed(
        positions_manifest_path,
        {
            "schema": POSITION_MANIFEST_SCHEMA,
            "pipeline_body_sha256": plan["body_sha256"],
            "games": inputs,
            "maximum_positions_per_game": POSITIONS_PER_GAME_MAX,
            "positions": len(position_rows),
            "games_with_positions": len(by_game),
            "skipped_short_games": skipped,
            "split_policy": "inherit-frozen-whole-root-group-train-validation-v1",
            "split_counts": dict(Counter(str(row["split"]) for row in position_rows)),
            "position_stratum_counts": dict(
                Counter(str(row["position_stratum"]) for row in position_rows)
            ),
            "exclusion_audit": {
                "algorithm": "minimum-sha256-over-exact+rotate+reflect+rotate-reflect",
                "sources": exclusions["sources"],
                "source_fingerprint_domains": exclusions["domains"],
                "source_fingerprint_counts": exclusions["counts"],
                "source_cross_intersection_count": exclusions[
                    "cross_source_intersection_count"
                ],
                "candidate_external_intersections": dict(external_intersections),
                "candidate_duplicate_intersections": candidate_duplicate_intersections,
                "train_validation_intersection_count": train_validation_intersection,
                "protected_labels_metrics_transcripts_read": False,
            },
            "rows": position_rows,
            "positions_sha256": sha256_bytes(positions_payload),
        },
    )
    roots = []
    for root_group_id, split in sorted(assignments.items()):
        if root_group_id in by_root:
            roots.append({"group_id": root_group_id, "split": split})
    roots_body: dict[str, object] = {
        "schema": corpus.ROOT_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "tool_sha256": {
            "normalizer": sha256_file(pathlib.Path(__file__)),
            "features": sha256_file(pathlib.Path(features.__file__)),
        },
        "exclusion_boundary": {"read_before_candidate_sources": True},
        "accepted": roots,
        "excluded": [],
        "counts": {"accepted": len(roots)},
    }
    roots_body["body_sha256"] = sha256_bytes(canonical_json_bytes(roots_body))
    roots_path = pathlib.Path(plan["outputs"]["root_assignments"])
    _write_once(roots_path, canonical_json_bytes(roots_body))
    return _finish_stage(
        plan, "02-positions", inputs,
        {"positions": positions_path, "manifest": positions_manifest_path, "roots": roots_path},
        {"positions": positions_manifest["positions"], "maximum_per_game": 20},
    )


def _position_rows(path: pathlib.Path) -> tuple[list[str], dict[str, list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[:1] != [POSITION_HEADER] or len(lines) < 2:
        raise PilotPipelineError("position TSV has the wrong schema")
    rows: list[str] = []
    by_game: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 8 or not fields[0] or fields[0] in seen:
            raise PilotPipelineError("position TSV has a malformed or duplicate row")
        seen.add(fields[0])
        rows.append(line)
        by_game[fields[2]].append(line)
    return rows, by_game


def _run_label_chunks(
    plan: Mapping[str, Any], *, stage: str, positions: pathlib.Path,
    output: pathlib.Path, kind: str, nodes: int, workers: int, resume: bool,
    producer: LabelProducer | None,
) -> dict[str, object]:
    if isinstance(workers, bool) or not 1 <= workers <= GAME_WORKERS_MAX:
        raise PilotPipelineError("label workers must be in 1..8")
    rows, by_game = _position_rows(positions)
    game_groups = list(by_game.values())
    chunks = [
        [row for group in game_groups[start : start + LABEL_GAMES_PER_CHUNK] for row in group]
        for start in range(0, len(game_groups), LABEL_GAMES_PER_CHUNK)
    ]
    producer_key = "action_teacher" if kind == "action" else "rank4_teacher"
    stage_inputs = {
        "positions": _record(positions),
        "producer": plan["producers"][producer_key],
        "teacher_runtime": plan["inputs"]["accepted_teacher_runtime"] if kind == "action" else None,
        "nodes": nodes,
        "workers": workers,
    }
    reused = _reuse_stage(plan, stage, stage_inputs, resume)
    if reused is not None:
        return reused
    runner = _subprocess_producer if producer is None else producer
    root = output.parent / f"{stage}-chunks"

    def execute(item: tuple[int, list[str]]) -> dict[str, object]:
        ordinal, chunk_rows = item
        directory = root / f"chunk-{ordinal:04d}"
        input_path = directory / "positions.tsv"
        labels_path = directory / "labels.jsonl"
        receipt_path = directory / "receipt.json"
        _write_once(input_path, (POSITION_HEADER + "\n" + "\n".join(chunk_rows) + "\n").encode())
        inputs = {"positions": _record(input_path), **stage_inputs}
        if receipt_path.exists():
            if not resume:
                raise PilotPipelineError("label chunk is complete; use --resume")
            receipt = _load_sealed(receipt_path, STAGE_RECEIPT_SCHEMA, "label chunk")
            if (
                receipt.get("pipeline_body_sha256") != plan["body_sha256"]
                or receipt.get("attempt") != plan["attempt"]
                or receipt.get("phase") != plan["phase"]
                or receipt.get("inputs") != inputs
            ):
                raise PilotPipelineError("label chunk receipt inputs changed")
            _validate_label_chunk(kind, nodes, input_path, labels_path, plan)
            return receipt
        if labels_path.exists():
            raise PilotPipelineError("label chunk has output without a receipt")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ".labels.partial"
        if kind == "action":
            teacher = _validate_record(plan["inputs"]["accepted_teacher_runtime"], "teacher")
            command = [
                str(_validate_record(plan["producers"][producer_key], "action teacher")),
                "--model", str(teacher), "--model-sha256", sha256_file(teacher),
                "--campaign-id", plan["campaign_id"], "--tree-nodes", str(nodes),
                "--time-ms", "0", "--max-actions", "250", "--max-partial-paths", "50000",
                "--exploration", "0.5", "--fpu", "0.5", "--emit-action-groups",
                "--source-bundle-body-sha256", plan["source_bundle_body_sha256"],
            ]
        else:
            command = [
                str(_validate_record(plan["producers"][producer_key], "Rank-4 teacher")),
                "--campaign-id", plan["campaign_id"], "--nodes", str(nodes), "--time-ms", "0",
            ]
        runner(command, input_path, temporary, THREAD_ENVIRONMENT)
        _validate_label_chunk(kind, nodes, input_path, temporary, plan)
        os.replace(temporary, labels_path)
        return _write_sealed(
            receipt_path,
            {
                "schema": STAGE_RECEIPT_SCHEMA,
                "pipeline_body_sha256": plan["body_sha256"],
                "attempt": plan["attempt"],
                "phase": plan["phase"],
                "stage": f"{stage}-chunk-{ordinal:04d}",
                "inputs": inputs,
                "outputs": {"labels": _record(labels_path)},
                "details": {"rows": len(chunk_rows), "worker_threads": 1},
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(execute, enumerate(chunks)))
    ordered_files = [
        root / f"chunk-{ordinal:04d}/labels.jsonl" for ordinal in range(len(chunks))
    ]
    payload = b"".join(path.read_bytes() for path in ordered_files)
    _write_once(output, payload)
    _validate_label_chunk(kind, nodes, positions, output, plan)
    return _finish_stage(
        plan, stage, stage_inputs, {"labels": output},
        {"rows": len(rows), "chunks": len(receipts), "workers": workers, "nodes": nodes},
    )


def _validate_label_chunk(
    kind: str, nodes: int, positions: pathlib.Path, labels: pathlib.Path,
    plan: Mapping[str, Any],
) -> None:
    position_rows, _ = _position_rows(positions)
    expected = [row.split("\t")[0] for row in position_rows]
    by_id = {row.split("\t")[0]: row.split("\t") for row in position_rows}
    observed = []
    for line_number, line in enumerate(labels.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if kind == "action":
                row = corpus.validate_complete_turn_action_group(row)
                source = row["group"]["source_binding"]
                position_id = source["position_id"]
                if (
                    row["source_bundle_body_sha256"] != plan["source_bundle_body_sha256"]
                    or row["teacher"]["artifact_sha256"]
                    != plan["inputs"]["accepted_teacher_runtime"]["sha256"]
                    or row["group"]["work_budget"]["max_tree_nodes"] != nodes
                ):
                    raise ValueError("action teacher binding changed")
            else:
                corpus.sample_from_teacher_row(row)
                position_id = row["position_id"]
                source = row
                if (
                    row.get("search_config", {}).get("max_nodes") != nodes
                    or row.get("teacher", {}).get("source_sha256")
                    != plan["inputs"]["rank4_source"]["sha256"]
                ):
                    raise ValueError("Rank-4 teacher work changed")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise PilotPipelineError(f"label line {line_number} is invalid: {error}") from error
        fields = by_id.get(str(position_id))
        if fields is None:
            raise PilotPipelineError("label position is outside its chunk")
        expected_prefix = [
            {"player_id": index % 2, "action": action}
            for index, action in enumerate(fields[7].split("/") if fields[7] else [])
        ]
        if (
            source.get("root_group_id") != fields[1]
            or source.get("group_id") != fields[2]
            or source.get("source") != fields[3]
            or source.get("split") != fields[4]
            or source.get("winner") != int(fields[5])
            or (source.get("mover", row.get("group", {}).get("parent_mover"))) != int(fields[6])
            or source.get("prefix") != expected_prefix
        ):
            raise PilotPipelineError("label lineage changed")
        observed.append(str(position_id))
    if observed != expected:
        raise PilotPipelineError("labels do not exactly cover positions in order")


def run_shallow_action_labels(
    plan_path: pathlib.Path, *, workers: int = 8, resume: bool = False,
    producer: LabelProducer | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    return _run_label_chunks(
        plan, stage="03-shallow-actions",
        positions=pathlib.Path(plan["outputs"]["positions"]),
        output=pathlib.Path(plan["outputs"]["shallow_actions"]),
        kind="action", nodes=SHALLOW_TREE_NODES, workers=workers,
        resume=resume, producer=producer,
    )


def run_rank4_labels(
    plan_path: pathlib.Path, *, workers: int = 8, resume: bool = False,
    producer: LabelProducer | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    return _run_label_chunks(
        plan, stage="04-rank4-labels",
        positions=pathlib.Path(plan["outputs"]["positions"]),
        output=pathlib.Path(plan["outputs"]["rank4_labels"]),
        kind="rank4", nodes=RANK4_TREE_NODES, workers=workers,
        resume=resume, producer=producer,
    )


def _rank4_value(row: Mapping[str, object]) -> float:
    corpus.sample_from_teacher_row(dict(row))
    mover = int(row["mover"])
    proven = row.get("proven_winner")
    if proven is not None:
        return 1.0 if int(proven) == mover else -1.0
    return (1.0 if mover == 0 else -1.0) * math.tanh(float(row["root_score"]) / 12_000.0)


def _student_predictor(runtime: pathlib.Path) -> Callable[[Mapping[str, Any]], list[float]]:
    try:
        from tools import compact_value_bfm_train as trainer
    except ModuleNotFoundError as error:
        raise RuntimeError("student action scoring requires the research NumPy environment") from error
    architecture, quantized, _selection, _document = trainer.load_runtime(runtime)

    def predict(group: Mapping[str, Any]) -> list[float]:
        return [
            float(trainer.scalar_quantized_forward(quantized, architecture, successor["active"]))
            for successor in group["successors"]
        ]

    return predict


def action_regret(
    group: Mapping[str, Any], student_values: Sequence[float]
) -> dict[str, object]:
    successors = group.get("successors")
    parent = group.get("parent_mover")
    if not isinstance(successors, list) or not successors or parent not in (0, 1):
        raise PilotPipelineError("action-regret group is malformed")
    if len(student_values) != len(successors) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) for value in student_values
    ):
        raise PilotPipelineError("student successor values are malformed")

    def parent_value(value: float, mover: int) -> float:
        return value if mover == parent else -value

    teacher = [
        parent_value(float(successor["teacher_value"]), int(successor["value_mover"]))
        for successor in successors
    ]
    student = [
        parent_value(float(value), int(successor["value_mover"]))
        for value, successor in zip(student_values, successors, strict=True)
    ]
    teacher_best = min(
        range(len(successors)), key=lambda index: (-teacher[index], successors[index]["successor_id"])
    )
    student_best = min(
        range(len(successors)), key=lambda index: (-student[index], successors[index]["successor_id"])
    )
    regret = max(0.0, teacher[teacher_best] - teacher[student_best])
    return {
        "regret": regret,
        "teacher_best_successor_id": successors[teacher_best]["successor_id"],
        "student_best_successor_id": successors[student_best]["successor_id"],
        "action_disagreement": teacher_best != student_best,
        "successors_exhaustive": group.get("successors_exhaustive") is True,
    }


def select_hard_positions(
    plan_path: pathlib.Path, *, resume: bool = False,
    predictor: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    positions = pathlib.Path(plan["outputs"]["positions"])
    positions_manifest_path = pathlib.Path(plan["outputs"]["positions_manifest"])
    shallow = pathlib.Path(plan["outputs"]["shallow_actions"])
    rank4_path = pathlib.Path(plan["outputs"]["rank4_labels"])
    inputs = {
        "positions": _record(positions),
        "positions_manifest": _record(positions_manifest_path),
        "shallow": _record(shallow), "rank4": _record(rank4_path),
        "student": plan["inputs"]["student_runtime"],
    }
    reused = _reuse_stage(plan, "05-hard-selection", inputs, resume)
    if reused is not None:
        return reused
    position_rows, by_game = _position_rows(positions)
    row_by_id = {row.split("\t")[0]: row for row in position_rows}
    position_manifest = _load_sealed(
        positions_manifest_path, POSITION_MANIFEST_SCHEMA, "pilot positions"
    )
    position_strata = {
        str(row["position_id"]): str(row["position_stratum"])
        for row in position_manifest["rows"]
    }
    if set(position_strata) != set(row_by_id):
        raise PilotPipelineError("position stratum manifest does not cover the TSV")
    action_rows = corpus.load_complete_turn_action_groups((shallow,))
    actions_by_position = {
        str(row["group"]["source_binding"]["position_id"]): row for row in action_rows
    }
    rank4_rows = {}
    for line in rank4_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        corpus.sample_from_teacher_row(row)
        rank4_rows[str(row["position_id"])] = row
    if set(actions_by_position) != set(row_by_id) or set(rank4_rows) != set(row_by_id):
        raise PilotPipelineError("hard-selection labels do not cover the frozen positions")
    score_student = predictor or _student_predictor(
        _validate_record(plan["inputs"]["student_runtime"], "student runtime")
    )
    selected: set[str] = set()
    evidence = []
    nonexhaustive_fill = 0
    selected_position_strata = Counter()
    selected_tactical = Counter()
    for game_id, group_rows in by_game.items():
        if len(group_rows) % 4:
            raise PilotPipelineError("per-game positions cannot satisfy an exact 25%")
        scored = []
        for line in group_rows:
            fields = line.split("\t")
            position_id, winner, mover = fields[0], int(fields[5]), int(fields[6])
            action = actions_by_position[position_id]
            group = action["group"]
            regret = action_regret(group, score_student(group))
            search_value = float(group["root_value"])
            rank4_value = _rank4_value(rank4_rows[position_id])
            outcome = 1.0 if winner == mover else -1.0
            search_rank4_disagree = (search_value >= 0.0) != (rank4_value >= 0.0)
            outcome_disagree = (search_value >= 0.0) != (outcome >= 0.0) or (
                (rank4_value >= 0.0) != (outcome >= 0.0)
            )
            key = (
                int(not regret["successors_exhaustive"]),
                -float(regret["regret"]),
                -int(regret["action_disagreement"]),
                -int(search_rank4_disagree),
                -abs(search_value - rank4_value),
                -int(outcome_disagree),
                min(abs(search_value), abs(rank4_value)),
                position_id,
            )
            scored.append((key, position_id, {
                **regret,
                "position_id": position_id,
                "game_id": game_id,
                "search_value": search_value,
                "rank4_value": rank4_value,
                "outcome": outcome,
                "search_rank4_sign_disagreement": search_rank4_disagree,
                "teacher_outcome_disagreement": outcome_disagree,
                "position_stratum": position_strata[position_id],
            }))
        count = len(scored) // 4
        ordered = sorted(scored)
        chosen = ordered[:count]
        nonexhaustive_fill += sum(
            not bool(item[2]["successors_exhaustive"]) for item in chosen
        )
        for _key, _position_id, item in chosen:
            selected_position_strata[str(item["position_stratum"])] += 1
            selected_tactical["teacher_student_action_disagreement"] += int(
                bool(item["action_disagreement"])
            )
            selected_tactical["positive_action_regret"] += int(
                float(item["regret"]) > 0.0
            )
            selected_tactical["search_rank4_sign_disagreement"] += int(
                bool(item["search_rank4_sign_disagreement"])
            )
            selected_tactical["teacher_outcome_disagreement"] += int(
                bool(item["teacher_outcome_disagreement"])
            )
        selected.update(item[1] for item in chosen)
        evidence.extend({
            **item[2],
            "hard_rank_within_game": rank,
            "selected_for_deep_label": rank < count,
        } for rank, item in enumerate(ordered))
    output_rows = [row_by_id[position_id] for position_id in row_by_id if position_id in selected]
    output_payload = (POSITION_HEADER + "\n" + "\n".join(output_rows) + "\n").encode()
    hard_path = pathlib.Path(plan["outputs"]["hard_positions"])
    _write_once(hard_path, output_payload)
    report_path = pathlib.Path(plan["outputs"]["hard_report"])
    report = _write_sealed(
        report_path,
        {
            "schema": HARD_SELECTION_SCHEMA,
            "pipeline_body_sha256": plan["body_sha256"],
            "inputs": inputs,
            "fraction": [1, 4],
            "positions": len(position_rows),
            "selected": len(selected),
            "games": len(by_game),
            "score_order": [
                "exhaustive-first-nonexhaustive-deterministic-fill-only",
                "teacher-student-action-regret-desc",
                "action-disagreement", "search-rank4-sign-disagreement",
                "search-rank4-absolute-gap", "teacher-outcome-disagreement",
                "minimum-absolute-confidence", "position-id",
            ],
            "rows": evidence,
            "nonexhaustive_fill": nonexhaustive_fill,
            "selected_position_strata": dict(selected_position_strata),
            "selected_tactical": dict(selected_tactical),
            "output_sha256": sha256_bytes(output_payload),
        },
    )
    return _finish_stage(
        plan, "05-hard-selection", inputs,
        {"positions": hard_path, "report": report_path},
        {"positions": len(position_rows), "selected": report["selected"], "fraction": [1, 4]},
    )


def run_deep_action_labels(
    plan_path: pathlib.Path, *, workers: int = 8, resume: bool = False,
    producer: LabelProducer | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    return _run_label_chunks(
        plan, stage="06-deep-actions",
        positions=pathlib.Path(plan["outputs"]["hard_positions"]),
        output=pathlib.Path(plan["outputs"]["deep_actions"]),
        kind="action", nodes=DEEP_TREE_NODES, workers=workers,
        resume=resume, producer=producer,
    )


def _scalar_sample_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        direct, reflected = corpus.sample_from_teacher_row(dict(row))
        split = row["split"]
        for is_reflected, sample in ((False, direct), (True, reflected)):
            output.append({
                "schema": SCALAR_SAMPLE_SCHEMA,
                "split": split,
                "active": list(sample.active),
                "target": sample.target,
                "weight": sample.weight,
                "root_group_id": sample.group_id,
                "reflected": is_reflected,
                "lineage": [dataclasses.asdict(lineage) for lineage in sample.lineages],
            })
    return output


StandardPacker = Callable[
    [Mapping[str, Any], pathlib.Path, pathlib.Path], Mapping[str, pathlib.Path]
]


def _standard_train_validation_pack(
    plan: Mapping[str, Any], merged_path: pathlib.Path,
    output_directory: pathlib.Path,
) -> Mapping[str, pathlib.Path]:
    try:
        from tools import jacek_replay_train as replay_train
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise RuntimeError(
                "standard CSR packing requires requirements-research.txt"
            ) from error
        raise
    roots_path = _validate_record(
        plan["inputs"]["filtered_roots_manifest"], "filtered roots"
    )
    assignments = replay_pack.frozen_assignments(replay_pack.load_roots(roots_path))
    samples = corpus.load_teacher_rows((merged_path,))
    retained, removed, aggregated = corpus.split_and_purge_samples(samples, assignments)
    if retained["test"] or not retained["train"] or not retained["validation"]:
        raise PilotPipelineError(
            "pilot scalar packing requires nonempty train/validation and no test rows"
        )
    prior_fingerprints: dict[bytes, str] = {}
    prior_records = []
    for split in ("train", "validation"):
        for ordinal, manifest_record in enumerate(
            plan["inputs"]["prior_shard_manifests"][split]
        ):
            manifest_path = _validate_record(
                manifest_record, f"prior {split} shard {ordinal}"
            )
            shard = replay_train.load_csr_shard(manifest_path)
            if shard.split != split:
                raise PilotPipelineError("prior shard split changed")
            for row in range(len(shard)):
                fingerprint = corpus.canonical_feature_fingerprint(shard.active(row))
                prior = prior_fingerprints.setdefault(fingerprint, split)
                if prior != split:
                    raise PilotPipelineError("prior shards overlap across splits")
            prior_records.append(_record(manifest_path))
    intersections = {"train": 0, "validation": 0}
    for split in ("train", "validation"):
        for sample in retained[split]:
            fingerprint = corpus.canonical_feature_fingerprint(sample.active)
            if fingerprint in prior_fingerprints:
                intersections[split] += 1
    if any(intersections.values()):
        raise PilotPipelineError("new scalar rows overlap prior train/validation shards")
    output_directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, pathlib.Path] = {}
    for split in ("train", "validation"):
        npz_path, manifest_path, _manifest = replay_train.write_csr_shard(
            output_directory,
            split,
            retained[split],
            provenance={
                "roots_manifest_sha256": sha256_file(roots_path),
                "teacher_jsonl_sha256": [
                    {"name": merged_path.name, "sha256": sha256_file(merged_path)}
                ],
                "target_policies": [
                    corpus.target_policy_for_schema(
                        corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA
                    )
                ],
                "tool_sha256": {
                    "pack": sha256_file(pathlib.Path(replay_pack.__file__)),
                    "corpus": sha256_file(pathlib.Path(corpus.__file__)),
                    "features": sha256_file(pathlib.Path(features.__file__)),
                    "adapter": sha256_file(pathlib.Path(__file__)),
                },
                "reflection_augmentation": True,
                "packing": "pilot-train-validation-existing-csr-writer-v1",
                "prior_shards": prior_records,
                "prior_intersections": intersections,
                "cross_split_canonical_rows_removed": removed,
                "same_orientation_rows_aggregated": aggregated,
                "protected_tests_opened": False,
            },
        )
        result[f"{split}_manifest"] = manifest_path
        result[f"{split}_npz"] = npz_path
    return result


def finalize_labels(
    plan_path: pathlib.Path, *, resume: bool = False,
    standard_packer: StandardPacker | None = None,
) -> dict[str, object]:
    plan = load_pipeline(plan_path)
    shallow_path = pathlib.Path(plan["outputs"]["shallow_actions"])
    deep_path = pathlib.Path(plan["outputs"]["deep_actions"])
    positions_path = pathlib.Path(plan["outputs"]["positions"])
    hard_path = pathlib.Path(plan["outputs"]["hard_positions"])
    hard_report_path = pathlib.Path(plan["outputs"]["hard_report"])
    inputs = {
        "shallow": _record(shallow_path), "deep": _record(deep_path),
        "positions": _record(positions_path),
        "hard_positions": _record(hard_path),
        "hard_report": _record(hard_report_path),
        "roots": _record(pathlib.Path(plan["outputs"]["root_assignments"])),
        "pack_tool": _record(pathlib.Path(replay_pack.__file__)),
        "prior_shard_manifests": plan["inputs"]["prior_shard_manifests"],
    }
    reused = _reuse_stage(plan, "07-finalize-labels", inputs, resume)
    if reused is not None:
        return reused
    shallow = corpus.load_complete_turn_action_groups((shallow_path,))
    deep = corpus.load_complete_turn_action_groups((deep_path,))
    position_rows, positions_by_game = _position_rows(positions_path)
    hard_rows, hard_by_game = _position_rows(hard_path)
    expected_all = [row.split("\t")[0] for row in position_rows]
    expected_hard = [row.split("\t")[0] for row in hard_rows]
    shallow_by_position = {
        str(row["group"]["source_binding"]["position_id"]): row
        for row in shallow
    }
    deep_by_position = {
        str(row["group"]["source_binding"]["position_id"]): row
        for row in deep
    }
    if (
        set(shallow_by_position) != set(expected_all)
        or set(deep_by_position) != set(expected_hard)
        or any(
            row["group"]["work_budget"]["max_tree_nodes"] != SHALLOW_TREE_NODES
            for row in shallow
        )
        or any(
            row["group"]["work_budget"]["max_tree_nodes"] != DEEP_TREE_NODES
            for row in deep
        )
        or set(hard_by_game) != set(positions_by_game)
        or any(
            len(positions_by_game[game_id]) % 4
            or len(hard_by_game[game_id]) * 4 != len(positions_by_game[game_id])
            for game_id in positions_by_game
        )
    ):
        raise PilotPipelineError(
            "shallow coverage or exact per-game deep replacement changed"
        )
    merged = corpus.merge_complete_turn_action_groups(shallow, deep)
    merged_by_position = {
        str(row["group"]["source_binding"]["position_id"]): row
        for row in merged
    }
    if set(merged_by_position) != set(expected_all) or any(
        merged_by_position[position_id]["group"]["work_budget"]["max_tree_nodes"]
        != (DEEP_TREE_NODES if position_id in deep_by_position else SHALLOW_TREE_NODES)
        for position_id in expected_all
    ):
        raise PilotPipelineError("deep labels did not replace exactly the hard rows")
    merged_path = pathlib.Path(plan["outputs"]["merged_actions"])
    merged_payload = b"".join(canonical_json_bytes(row) for row in merged)
    _write_once(merged_path, merged_payload)
    aggregate = corpus.build_complete_turn_successor_labels(merged)
    aggregate_payload = canonical_json_bytes(aggregate)
    aggregate_path = pathlib.Path(plan["outputs"]["successor_labels"])
    content_path = aggregate_path.with_name(
        f"{sha256_bytes(aggregate_payload)}.successor-labels.json"
    )
    _write_once(content_path, aggregate_payload)
    _write_once(aggregate_path, aggregate_payload)
    scalar_rows = _scalar_sample_rows(merged)
    scalar_payload = b"".join(canonical_json_bytes(row) for row in scalar_rows)
    scalar_path = pathlib.Path(plan["outputs"]["scalar_samples"])
    _write_once(scalar_path, scalar_payload)
    packer = standard_packer or _standard_train_validation_pack
    packed = dict(
        packer(
            plan,
            merged_path,
            pathlib.Path(plan["outputs"]["scalar_shards"]),
        )
    )
    if set(packed) != {
        "train_manifest", "train_npz", "validation_manifest", "validation_npz"
    }:
        raise PilotPipelineError("standard scalar packer returned the wrong artifacts")
    for label, path in packed.items():
        _record(path)
    references = {}
    for split in ("train", "validation"):
        reference_path = pathlib.Path(
            plan["outputs"][f"scalar_{split}_reference"]
        )
        references[split] = reference_path
        _write_sealed(
            reference_path,
            {
                "schema": SHARD_REFERENCE_SCHEMA,
                "pipeline_body_sha256": plan["body_sha256"],
                "split": split,
                "manifest": _record(packed[f"{split}_manifest"]),
                "npz": _record(packed[f"{split}_npz"]),
                "shard_schema": "papersoccer.jacek-replay-csr-shard.v1",
                "protected_tests_opened": False,
            },
        )
    scalar_manifest_path = pathlib.Path(plan["outputs"]["scalar_manifest"])
    scalar_manifest = _write_sealed(
        scalar_manifest_path,
        {
            "schema": SCALAR_PACK_SCHEMA,
            "pipeline_body_sha256": plan["body_sha256"],
            "source_rows": _record(merged_path),
            "root_assignments": inputs["roots"],
            "target_policy": corpus.target_policy_for_schema(
                corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA
            ),
            "reflection_augmentation": True,
            "counts": dict(Counter(str(row["split"]) for row in scalar_rows)),
            "samples": _record(scalar_path),
            "standard_shards": {
                split: _record(path) for split, path in references.items()
            },
        },
    )
    return _finish_stage(
        plan, "07-finalize-labels", inputs,
        {
            "merged": merged_path, "successor_labels": content_path,
            "successor_pointer": aggregate_path, "scalar_samples": scalar_path,
            "scalar_manifest": scalar_manifest_path,
            "train_shard_reference": references["train"],
            "validation_shard_reference": references["validation"],
        },
        {
            "groups": len(merged), "scalar_samples": len(scalar_rows),
            "deep_replacements": len(deep), "protected_tests_opened": False,
            "scalar_manifest_body_sha256": scalar_manifest["body_sha256"],
        },
    )


def run_pipeline(
    plan_path: pathlib.Path, *, workers: int = 8, resume: bool = False,
) -> dict[str, object]:
    run_game_chunks(plan_path, workers=workers, resume=resume)
    materialize_positions(plan_path, resume=resume)
    run_shallow_action_labels(plan_path, workers=workers, resume=resume)
    run_rank4_labels(plan_path, workers=workers, resume=resume)
    select_hard_positions(plan_path, resume=resume)
    run_deep_action_labels(plan_path, workers=workers, resume=resume)
    return finalize_labels(plan_path, resume=resume)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprints = commands.add_parser("build-fingerprint-set")
    fingerprints.add_argument("--output-directory", type=pathlib.Path, required=True)
    fingerprints.add_argument("--classification", required=True)
    fingerprints.add_argument(
        "--source", type=pathlib.Path, action="append", required=True
    )
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--campaign-plan", type=pathlib.Path, required=True)
    prepare.add_argument("--phase-reference", type=pathlib.Path, required=True)
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--student-runtime", type=pathlib.Path)
    prepare.add_argument("--roots-tsv", type=pathlib.Path)
    prepare.add_argument("--roots-manifest", type=pathlib.Path)
    prepare.add_argument("--game-producer", type=pathlib.Path)
    prepare.add_argument("--action-teacher", type=pathlib.Path)
    prepare.add_argument("--rank4-teacher", type=pathlib.Path)
    prepare.add_argument("--created-at-utc", required=True)
    for name in (
        "run-games", "freeze-positions", "run-shallow-labels",
        "run-rank4-labels", "select-hard", "run-deep-labels", "finalize", "run",
    ):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--resume", action="store_true")
        if name in {"run-games", "run-shallow-labels", "run-rank4-labels", "run-deep-labels", "run"}:
            command.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args(argv)
    if arguments.command == "build-fingerprint-set":
        output = materialize_fingerprint_set(
            output_directory=arguments.output_directory,
            classification=arguments.classification,
            sources=arguments.source,
        )
    elif arguments.command == "prepare":
        output = prepare_pipeline(
            campaign_plan=arguments.campaign_plan,
            phase_reference=arguments.phase_reference,
            output_root=arguments.output_root,
            student_runtime=arguments.student_runtime,
            roots_tsv=arguments.roots_tsv,
            roots_manifest=arguments.roots_manifest,
            game_producer=arguments.game_producer,
            action_teacher=arguments.action_teacher,
            rank4_teacher=arguments.rank4_teacher,
            created_at_utc=arguments.created_at_utc,
        )
    elif arguments.command == "run-games":
        output = run_game_chunks(arguments.plan, workers=arguments.workers, resume=arguments.resume)
    elif arguments.command == "freeze-positions":
        output = materialize_positions(arguments.plan, resume=arguments.resume)
    elif arguments.command == "run-shallow-labels":
        output = run_shallow_action_labels(arguments.plan, workers=arguments.workers, resume=arguments.resume)
    elif arguments.command == "run-rank4-labels":
        output = run_rank4_labels(arguments.plan, workers=arguments.workers, resume=arguments.resume)
    elif arguments.command == "select-hard":
        output = select_hard_positions(arguments.plan, resume=arguments.resume)
    elif arguments.command == "run-deep-labels":
        output = run_deep_action_labels(arguments.plan, workers=arguments.workers, resume=arguments.resume)
    elif arguments.command == "finalize":
        output = finalize_labels(arguments.plan, resume=arguments.resume)
    else:
        output = run_pipeline(arguments.plan, workers=arguments.workers, resume=arguments.resume)
    if isinstance(output, dict):
        rendered = output
    else:
        rendered = {
            "fingerprint_set" if arguments.command == "build-fingerprint-set" else "plan":
            str(output)
        }
    print(json.dumps(rendered, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
