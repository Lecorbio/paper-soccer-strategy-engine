#!/usr/bin/env python3
"""Freeze, label, pack, and evaluate a blind replay-BFM retention holdout.

This tool deliberately produces a shard schema that the training loader does
not accept.  Its three stages are ordered by explicit receipt dependencies:

* ``freeze`` selects complete candidate root groups after training inputs are
  frozen, without opening any teacher labels or selected model;
* ``pack`` opens fixed-work Rank-4 labels only after a model-selection receipt
  exists; and
* ``evaluate`` binds the selected runtime to paired, root-clustered
  noninferiority evidence.

Game and position generation remain the responsibility of the existing
continuation/position producers.  This module consumes their position TSV and
reuses the canonical feature, teacher-row, shard, and runtime parsers.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

import jacek_replay_corpus as corpus
import jacek_replay_features as features
import jacek_replay_train as training


FREEZE_SCHEMA = "papersoccer.jacek-retention-holdout-freeze.v1"
SHARD_SCHEMA = "papersoccer.jacek-retention-holdout-shard.v1"
EVIDENCE_SCHEMA = "papersoccer.jacek-retention-noninferiority.v1"
POSITION_HEADER = (
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
)
TEACHER_SPLIT = "validation"
ROWS_PER_GROUP = 20
RANK4_FIXED_NODES = 400_000
RANK4_FIXED_CONFIGURATION = {
    "max_nodes": RANK4_FIXED_NODES,
    "max_time_ms": 0,
    "max_turn_depth": 32,
    "replay_value_blend_percent": 15,
    "teacher_residual_weight_percent": 100,
}


@dataclasses.dataclass(frozen=True)
class FreezeSpec:
    profile: str
    groups: int
    selection_seed: int
    rows_per_group: int = ROWS_PER_GROUP
    source_quotas: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        if (
            not self.profile
            or isinstance(self.groups, bool)
            or self.groups <= 0
            or isinstance(self.selection_seed, bool)
            or self.selection_seed < 0
            or self.selection_seed >= 1 << 64
            or self.rows_per_group != ROWS_PER_GROUP
        ):
            raise ValueError("retention freeze specification is invalid")
        quota_names = [name for name, _count in self.source_quotas]
        if self.source_quotas and (
            len(set(quota_names)) != len(quota_names)
            or any(not name or count <= 0 for name, count in self.source_quotas)
            or sum(count for _name, count in self.source_quotas) != self.groups
        ):
            raise ValueError("retention source quotas are invalid")


PILOT_SOURCE_QUOTAS = (
    ("rank4-vs-rank4", 200),
    ("round0-selfplay", 100),
    ("round0-p1-vs-rank4", 50),
    ("round0-p2-vs-rank4", 50),
    ("round1-selfplay", 100),
    ("round1-p1-vs-rank4", 50),
    ("round1-p2-vs-rank4", 50),
)
FULL_SOURCE_QUOTAS = tuple(
    (source, count * 2) for source, count in PILOT_SOURCE_QUOTAS
)
PILOT_SPEC = FreezeSpec(
    "pilot", 600, 2026082611, source_quotas=PILOT_SOURCE_QUOTAS
)
FULL_SPEC = FreezeSpec(
    "full", 1_200, 2026082617, source_quotas=FULL_SOURCE_QUOTAS
)
PROFILES = {spec.profile: spec for spec in (PILOT_SPEC, FULL_SPEC)}


@dataclasses.dataclass(frozen=True)
class NoninferiorityThresholds:
    sign_accuracy_margin: float = 0.005
    weighted_huber_multiplier: float = 1.02
    confidence: float = 0.95
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 2026082623

    def validate(self) -> None:
        if (
            not math.isfinite(self.sign_accuracy_margin)
            or not 0.0 <= self.sign_accuracy_margin < 1.0
            or not math.isfinite(self.weighted_huber_multiplier)
            or self.weighted_huber_multiplier < 1.0
            or not math.isfinite(self.confidence)
            or not 0.5 < self.confidence < 1.0
            or isinstance(self.bootstrap_replicates, bool)
            or self.bootstrap_replicates < 1_000
            or isinstance(self.bootstrap_seed, bool)
            or not 0 <= self.bootstrap_seed < 1 << 64
        ):
            raise ValueError("retention noninferiority thresholds are invalid")

    def record(self) -> dict[str, int | float | str]:
        self.validate()
        return {
            "sign_accuracy_margin": self.sign_accuracy_margin,
            "weighted_huber_multiplier": self.weighted_huber_multiplier,
            "confidence": self.confidence,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "interval": "one-sided-root-cluster-percentile-nearest-rank",
        }


FROZEN_THRESHOLDS = NoninferiorityThresholds()


@dataclasses.dataclass(frozen=True)
class PositionRow:
    position_id: str
    root_group_id: str
    group_id: str
    source: str
    split: str
    winner: int
    mover: int
    prefix: str
    prefix_records: tuple[dict[str, object], ...]
    active: tuple[int, ...]
    canonical_fingerprint: bytes

    def render(self) -> str:
        return "\t".join(
            (
                self.position_id,
                self.root_group_id,
                self.group_id,
                self.source,
                self.split,
                str(self.winner),
                str(self.mover),
                self.prefix,
            )
        )


@dataclasses.dataclass(frozen=True)
class HoldoutShard:
    manifest: dict
    indptr: np.ndarray
    indices: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    root_group_ids: np.ndarray
    position_ids: np.ndarray
    canonical_fingerprints: np.ndarray
    orientations: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def active(self, row: int) -> np.ndarray:
        return self.indices[self.indptr[row] : self.indptr[row + 1]]


def canonical_json_bytes(value: object) -> bytes:
    return corpus.canonical_json_bytes(value)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_snapshot(path: pathlib.Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"retention input is not a file: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lines": len(payload.splitlines()),
    }


def _validate_snapshot(record: object, label: str) -> None:
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or artifact_snapshot(pathlib.Path(record["path"])) != record
    ):
        raise ValueError(f"{label} artifact binding is stale")


def _validate_snapshot_list(records: object, label: str) -> None:
    if not isinstance(records, list):
        raise ValueError(f"{label} artifact list is malformed")
    for index, record in enumerate(records):
        _validate_snapshot(record, f"{label} {index}")


def _validate_producer_identity(record: object) -> None:
    current = _producer_identity()
    if record != current:
        raise ValueError("retention producer identity changed after freeze")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_body_hash(value: Mapping[str, object], label: str) -> None:
    supplied = value.get("body_sha256")
    body = dict(value)
    body.pop("body_sha256", None)
    if (
        not _valid_sha256(supplied)
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != supplied
    ):
        raise ValueError(f"{label} body hash is invalid")


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as destination:
            temporary = pathlib.Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_content_addressed(
    directory: pathlib.Path, suffix: str, payload: bytes
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{hashlib.sha256(payload).hexdigest()}{suffix}"
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"content-addressed retention artifact conflicts: {target}")
        return target
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=f".{target.name}.", delete=False
        ) as destination:
            temporary = pathlib.Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise RuntimeError(
                    f"content-addressed retention artifact raced: {target}"
                )
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _producer_identity() -> dict[str, object]:
    paths = {
        "retention": pathlib.Path(__file__).resolve(),
        "corpus": pathlib.Path(corpus.__file__).resolve(),
        "features": pathlib.Path(features.__file__).resolve(),
        "training_runtime": pathlib.Path(training.__file__).resolve(),
    }
    return {name: artifact_snapshot(path) for name, path in paths.items()}


def _prefix_state(prefix: str) -> tuple[features.ReplayState, tuple[dict, ...]]:
    state = features.ReplayState()
    records: list[dict[str, object]] = []
    if prefix:
        for action in prefix.split("/"):
            if not action:
                raise ValueError("retention position prefix has an empty action")
            mover = state.to_move
            features.apply_complete_turn(state, mover, action)
            records.append({"player_id": mover, "action": action})
    if state.winner is not None:
        raise ValueError("retention position prefix is terminal")
    return state, tuple(records)


def load_position_rows(
    path: pathlib.Path, *, required_split: str | None = None
) -> list[PositionRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != POSITION_HEADER:
        raise ValueError("retention position TSV header is invalid")
    result: list[PositionRow] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError(f"retention position line {line_number} is malformed")
        position_id, root_group_id, group_id, source, split = fields[:5]
        if any(not value or "\t" in value for value in fields[:5]):
            raise ValueError(f"retention position line {line_number} has an empty identity")
        if position_id in seen_ids:
            raise ValueError("retention positions repeat a position_id")
        if split not in {"train", "validation", "test"} or (
            required_split is not None and split != required_split
        ):
            raise ValueError("retention position split is invalid")
        try:
            winner, mover = int(fields[5]), int(fields[6])
        except ValueError as error:
            raise ValueError("retention winner/mover is invalid") from error
        if winner not in (0, 1) or mover not in (0, 1):
            raise ValueError("retention winner/mover must be zero or one")
        state, prefix_records = _prefix_state(fields[7])
        if state.to_move != mover:
            raise ValueError("retention position mover disagrees with its prefix")
        active = features.encode_active(state)
        result.append(
            PositionRow(
                position_id,
                root_group_id,
                group_id,
                source,
                split,
                winner,
                mover,
                fields[7],
                prefix_records,
                active,
                corpus.canonical_feature_fingerprint(active),
            )
        )
        seen_ids.add(position_id)
    if not result:
        raise ValueError("retention position TSV is empty")
    return result


def _root_groups_from_manifest(path: pathlib.Path) -> set[str]:
    value = _load_json(path, "retention exclusion root manifest")
    schema = value.get("schema")
    _verify_body_hash(value, "retention exclusion root manifest")
    if schema == corpus.ROOT_SCHEMA:
        records = value.get("accepted")
        if not isinstance(records, list):
            raise ValueError("canonical root manifest has no accepted groups")
        groups = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(
                record.get("root_group_id", record.get("group_id")), str
            ):
                raise ValueError("canonical root manifest group is malformed")
            groups.add(str(record.get("root_group_id", record.get("group_id"))))
        return groups
    if schema == FREEZE_SCHEMA:
        records = value.get("selection", {}).get("groups")
        if not isinstance(records, list):
            raise ValueError("retention freeze manifest has no groups")
        groups = {
            str(record.get("root_group_id"))
            for record in records
            if isinstance(record, dict) and isinstance(record.get("root_group_id"), str)
        }
        if len(groups) != len(records):
            raise ValueError("retention freeze manifest group set is malformed")
        return groups
    raise ValueError("unsupported retention exclusion root manifest")


def _fingerprints_from_shards(paths: Sequence[pathlib.Path]) -> set[bytes]:
    result: set[bytes] = set()
    for path in paths:
        shard = training.load_csr_shard(path)
        for row in range(len(shard)):
            result.add(
                corpus.canonical_feature_fingerprint(shard.active(row).tolist())
            )
    return result


def _exclusion_sets(
    *,
    shard_manifests: Sequence[pathlib.Path],
    position_tsvs: Sequence[pathlib.Path],
    root_manifests: Sequence[pathlib.Path],
) -> tuple[set[str], set[bytes]]:
    groups: set[str] = set()
    fingerprints = _fingerprints_from_shards(shard_manifests)
    for path in position_tsvs:
        rows = load_position_rows(path)
        groups.update(row.root_group_id for row in rows)
        fingerprints.update(row.canonical_fingerprint for row in rows)
    for path in root_manifests:
        groups.update(_root_groups_from_manifest(path))
    return groups, fingerprints


def _selection_key(spec: FreezeSpec, root_group_id: str) -> tuple[bytes, str]:
    material = (
        b"papersoccer-retention-group-selection-v1\0"
        + spec.selection_seed.to_bytes(8, "little")
        + root_group_id.encode("utf-8")
    )
    return hashlib.sha256(material).digest(), root_group_id


def freeze_candidate_groups(
    *,
    candidate_positions: pathlib.Path,
    training_input_receipt: pathlib.Path,
    campaign_id: str,
    spec: FreezeSpec,
    excluded_shard_manifests: Sequence[pathlib.Path] = (),
    excluded_position_tsvs: Sequence[pathlib.Path] = (),
    excluded_root_manifests: Sequence[pathlib.Path] = (),
    precomputed_excluded_groups: set[str] | None = None,
    precomputed_excluded_fingerprints: set[bytes] | None = None,
) -> tuple[bytes, dict]:
    """Select exact complete groups; a single overlapping row rejects its group."""

    spec.validate()
    if not campaign_id:
        raise ValueError("retention campaign_id must be nonempty")
    training_receipt = artifact_snapshot(training_input_receipt)
    candidates = load_position_rows(candidate_positions, required_split=TEACHER_SPLIT)

    if (
        (precomputed_excluded_groups is None)
        != (precomputed_excluded_fingerprints is None)
    ):
        raise ValueError("retention precomputed exclusion sets are incomplete")
    if precomputed_excluded_groups is None:
        excluded_groups, excluded_fingerprints = _exclusion_sets(
            shard_manifests=excluded_shard_manifests,
            position_tsvs=excluded_position_tsvs,
            root_manifests=excluded_root_manifests,
        )
    else:
        excluded_groups = set(precomputed_excluded_groups)
        excluded_fingerprints = set(precomputed_excluded_fingerprints or ())
        if not excluded_groups or not excluded_fingerprints:
            raise ValueError("retention precomputed exclusion sets are empty")

    grouped: dict[str, list[PositionRow]] = defaultdict(list)
    for row in candidates:
        grouped[row.root_group_id].append(row)

    selected: list[tuple[str, list[PositionRow]]] = []
    selected_fingerprints: set[bytes] = set()
    rejected: list[dict[str, object]] = []
    considered = 0
    source_quotas = dict(spec.source_quotas)
    selected_source_counts: Counter[str] = Counter()
    for root_group_id in sorted(grouped, key=lambda group: _selection_key(spec, group)):
        if len(selected) == spec.groups:
            break
        considered += 1
        rows = sorted(grouped[root_group_id], key=lambda row: row.position_id)
        fingerprints = {row.canonical_fingerprint for row in rows}
        sources = {row.source for row in rows}
        source = next(iter(sources)) if len(sources) == 1 else None
        reason: str | None = None
        if len(rows) != spec.rows_per_group:
            reason = "not-exact-rows-per-root-group"
        elif len({row.group_id for row in rows}) != 1:
            reason = "root-group-spans-multiple-generated-games"
        elif source is None:
            reason = "root-group-spans-multiple-sources"
        elif source_quotas and source not in source_quotas:
            reason = "source-outside-frozen-quotas"
        elif source_quotas and selected_source_counts[source] >= source_quotas[source]:
            reason = "source-quota-already-filled"
        elif len(fingerprints) != spec.rows_per_group:
            reason = "within-root-canonical-fingerprint-overlap"
        elif root_group_id in excluded_groups:
            reason = "excluded-root-group-overlap"
        elif fingerprints & excluded_fingerprints:
            reason = "excluded-canonical-fingerprint-overlap"
        elif fingerprints & selected_fingerprints:
            reason = "selected-canonical-fingerprint-overlap"
        if reason is not None:
            rejected.append(
                {
                    "root_group_id": root_group_id,
                    "rows": len(rows),
                    "reason": reason,
                }
            )
            continue
        selected.append((root_group_id, rows))
        selected_fingerprints.update(fingerprints)
        selected_source_counts[str(source)] += 1

    if len(selected) != spec.groups:
        reasons = Counter(str(record["reason"]) for record in rejected)
        raise ValueError(
            "retention candidate pool cannot fill its exact group quota: "
            f"selected={len(selected)} required={spec.groups} rejected={dict(reasons)}"
        )
    if source_quotas and dict(selected_source_counts) != source_quotas:
        raise RuntimeError("retention freeze did not satisfy exact source quotas")

    frozen_rows = [row for _group, rows in selected for row in rows]
    if (
        len(frozen_rows) != spec.groups * spec.rows_per_group
        or len({row.position_id for row in frozen_rows}) != len(frozen_rows)
        or len(selected_fingerprints) != len(frozen_rows)
    ):
        raise RuntimeError("retention freeze lost exact position coverage")
    payload = (
        POSITION_HEADER + "\n" + "\n".join(row.render() for row in frozen_rows) + "\n"
    ).encode("utf-8")
    group_records = [
        {
            "root_group_id": root_group_id,
            "generated_group_id": rows[0].group_id,
            "source": rows[0].source,
            "rows": len(rows),
            "position_ids_sha256": hashlib.sha256(
                "\n".join(row.position_id for row in rows).encode("utf-8")
            ).hexdigest(),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(row.canonical_fingerprint for row in rows))
            ).hexdigest(),
        }
        for root_group_id, rows in selected
    ]
    manifest: dict[str, object] = {
        "schema": FREEZE_SCHEMA,
        "campaign_id": campaign_id,
        "profile": spec.profile,
        "feature_schema": features.FEATURE_SCHEMA,
        "role": f"retention-{spec.profile}",
        "teacher_split": TEACHER_SPLIT,
        "training_eligible": False,
        "configuration": {
            "groups": spec.groups,
            "rows_per_group": spec.rows_per_group,
            "selection_seed": spec.selection_seed,
            "selection": "sha256-seeded-whole-root-groups-v1",
            "source_quotas": source_quotas,
            "overlap_identity": "rotate-and-reflect-canonical-feature-fingerprint",
            "partial_group_policy": "reject-whole-candidate-group-before-freeze",
            "post_freeze_drop_policy": "forbidden",
        },
        "timing": {
            "training_inputs_frozen_by": training_receipt,
            "teacher_labels_opened": False,
            "selected_model_opened": False,
            "required_reveal_order": (
                "freeze-before-model-selection;labels-after-model-selection;"
                "metrics-after-selected-runtime-binding"
            ),
        },
        "inputs": {
            "candidate_positions": artifact_snapshot(candidate_positions),
            "excluded_shards": [
                artifact_snapshot(path) for path in excluded_shard_manifests
            ],
            "excluded_positions": [
                artifact_snapshot(path) for path in excluded_position_tsvs
            ],
            "excluded_roots": [
                artifact_snapshot(path) for path in excluded_root_manifests
            ],
        },
        "exclusion_universe": {
            "root_groups": len(excluded_groups),
            "canonical_fingerprints": len(excluded_fingerprints),
            "root_group_ids_sha256": hashlib.sha256(
                "\n".join(sorted(excluded_groups)).encode("utf-8")
            ).hexdigest(),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(excluded_fingerprints))
            ).hexdigest(),
        },
        "selection": {
            "candidate_root_groups": len(grouped),
            "considered_root_groups": considered,
            "selected_root_groups": len(selected),
            "selected_positions": len(frozen_rows),
            "unconsidered_root_groups": len(grouped) - considered,
            "rejected": rejected,
            "rejection_counts": dict(
                sorted(Counter(str(row["reason"]) for row in rejected).items())
            ),
            "groups": group_records,
            "selected_source_counts": dict(sorted(selected_source_counts.items())),
            "root_group_ids_sha256": hashlib.sha256(
                "\n".join(root for root, _rows in selected).encode("utf-8")
            ).hexdigest(),
            "position_ids_sha256": hashlib.sha256(
                "\n".join(row.position_id for row in frozen_rows).encode("utf-8")
            ).hexdigest(),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(selected_fingerprints))
            ).hexdigest(),
        },
        "output": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "lines": len(payload.splitlines()),
        },
        "producer": _producer_identity(),
    }
    manifest["body_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return payload, manifest


def write_freeze(
    positions_path: pathlib.Path,
    manifest_path: pathlib.Path,
    payload: bytes,
    manifest: Mapping[str, object],
) -> None:
    if hashlib.sha256(payload).hexdigest() != manifest.get("output", {}).get("sha256"):
        raise ValueError("retention freeze output hash is inconsistent")
    _atomic_write(positions_path, payload)
    _atomic_write(manifest_path, canonical_json_bytes(dict(manifest)))


def load_freeze(
    positions_path: pathlib.Path, manifest_path: pathlib.Path
) -> tuple[dict, list[PositionRow]]:
    manifest = _load_json(manifest_path, "retention freeze manifest")
    if (
        manifest.get("schema") != FREEZE_SCHEMA
        or manifest.get("feature_schema") != features.FEATURE_SCHEMA
        or manifest.get("training_eligible") is not False
        or manifest.get("teacher_split") != TEACHER_SPLIT
    ):
        raise ValueError("retention freeze manifest identity is invalid")
    _verify_body_hash(manifest, "retention freeze manifest")
    timing = manifest.get("timing")
    inputs = manifest.get("inputs")
    if not isinstance(timing, dict) or not isinstance(inputs, dict):
        raise ValueError("retention freeze provenance is missing")
    _validate_snapshot(
        timing.get("training_inputs_frozen_by"), "retention training-input receipt"
    )
    _validate_snapshot(inputs.get("candidate_positions"), "retention candidate positions")
    _validate_snapshot_list(inputs.get("excluded_shards"), "retention excluded shards")
    _validate_snapshot_list(
        inputs.get("excluded_positions"), "retention excluded positions"
    )
    _validate_snapshot_list(inputs.get("excluded_roots"), "retention excluded roots")
    _validate_producer_identity(manifest.get("producer"))
    excluded_shard_paths = [
        pathlib.Path(record["path"]) for record in inputs["excluded_shards"]
    ]
    excluded_position_paths = [
        pathlib.Path(record["path"]) for record in inputs["excluded_positions"]
    ]
    excluded_root_paths = [
        pathlib.Path(record["path"]) for record in inputs["excluded_roots"]
    ]
    excluded_groups, excluded_fingerprints = _exclusion_sets(
        shard_manifests=excluded_shard_paths,
        position_tsvs=excluded_position_paths,
        root_manifests=excluded_root_paths,
    )
    exclusion = manifest.get("exclusion_universe")
    if (
        not isinstance(exclusion, dict)
        or exclusion.get("root_groups") != len(excluded_groups)
        or exclusion.get("canonical_fingerprints") != len(excluded_fingerprints)
        or exclusion.get("root_group_ids_sha256")
        != hashlib.sha256(
            "\n".join(sorted(excluded_groups)).encode("utf-8")
        ).hexdigest()
        or exclusion.get("canonical_fingerprints_sha256")
        != hashlib.sha256(b"".join(sorted(excluded_fingerprints))).hexdigest()
    ):
        raise ValueError("retention exclusion universe is stale")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("sha256") != sha256_file(positions_path)
        or output.get("bytes") != positions_path.stat().st_size
        or output.get("lines") != len(positions_path.read_bytes().splitlines())
    ):
        raise ValueError("retention frozen positions differ from their manifest")
    rows = load_position_rows(positions_path, required_split=TEACHER_SPLIT)
    configuration = manifest.get("configuration", {})
    groups = manifest.get("selection", {}).get("groups")
    expected_groups = configuration.get("groups")
    rows_per_group = configuration.get("rows_per_group")
    if (
        not isinstance(expected_groups, int)
        or expected_groups <= 0
        or rows_per_group != ROWS_PER_GROUP
        or not isinstance(groups, list)
        or len(groups) != expected_groups
        or len(rows) != expected_groups * ROWS_PER_GROUP
        or len({row.root_group_id for row in rows}) != expected_groups
        or any(
            count != ROWS_PER_GROUP
            for count in Counter(row.root_group_id for row in rows).values()
        )
        or len({row.canonical_fingerprint for row in rows}) != len(rows)
    ):
        raise ValueError("retention freeze exact group contract is invalid")
    if manifest.get("selection", {}).get("position_ids_sha256") != hashlib.sha256(
        "\n".join(row.position_id for row in rows).encode("utf-8")
    ).hexdigest() or manifest.get("selection", {}).get(
        "canonical_fingerprints_sha256"
    ) != hashlib.sha256(
        b"".join(sorted(row.canonical_fingerprint for row in rows))
    ).hexdigest():
        raise ValueError("retention freeze identities are stale")
    selected_root_ids = []
    rows_by_root = defaultdict(list)
    for row in rows:
        rows_by_root[row.root_group_id].append(row)
    for record in groups:
        bound_rows = (
            rows_by_root.get(str(record.get("root_group_id")), [])
            if isinstance(record, dict)
            else []
        )
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("root_group_id"), str)
            or record.get("rows") != ROWS_PER_GROUP
            or not bound_rows
            or record.get("source") != bound_rows[0].source
        ):
            raise ValueError("retention freeze group record is malformed")
        selected_root_ids.append(record["root_group_id"])
    row_root_ids = list(dict.fromkeys(row.root_group_id for row in rows))
    if selected_root_ids != row_root_ids or manifest.get("selection", {}).get(
        "root_group_ids_sha256"
    ) != hashlib.sha256("\n".join(row_root_ids).encode("utf-8")).hexdigest():
        raise ValueError("retention freeze root group identities are stale")
    if set(row_root_ids) & excluded_groups or {
        row.canonical_fingerprint for row in rows
    } & excluded_fingerprints:
        raise ValueError("retention freeze overlaps its exclusion universe")
    source_quotas = configuration.get("source_quotas")
    actual_source_counts = dict(
        sorted(Counter(group_rows[0].source for group_rows in rows_by_root.values()).items())
    )
    if source_quotas and (
        not isinstance(source_quotas, dict)
        or source_quotas != actual_source_counts
        or manifest.get("selection", {}).get("selected_source_counts")
        != actual_source_counts
    ):
        raise ValueError("retention freeze source quotas are stale")
    return manifest, rows


def _load_rank4_labels(path: pathlib.Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        if not raw.strip():
            raise ValueError("retention labels contain a blank line")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"retention label line {line_number} is invalid") from error
        if not isinstance(value, dict) or raw != canonical_json_bytes(value):
            raise ValueError("retention labels must be canonical one-row JSONL")
        position_id = value.get("position_id")
        if not isinstance(position_id, str) or not position_id or position_id in result:
            raise ValueError("retention labels repeat or omit position_id")
        result[position_id] = value
    if not result:
        raise ValueError("retention labels are empty")
    return result


def _label_lineage_matches(row: PositionRow, label: Mapping[str, object]) -> bool:
    return bool(
        label.get("position_id") == row.position_id
        and label.get("root_group_id") == row.root_group_id
        and label.get("group_id") == row.group_id
        and label.get("source") == row.source
        and label.get("split") == row.split
        and label.get("winner") == row.winner
        and label.get("mover") == row.mover
        and label.get("prefix") == list(row.prefix_records)
    )


def _holdout_arrays(
    rows: Sequence[PositionRow], labels: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, np.ndarray], dict[str, int], dict]:
    active_rows: list[tuple[int, ...]] = []
    targets: list[float] = []
    weights: list[float] = []
    roots: list[bytes] = []
    positions: list[bytes] = []
    fingerprints: list[bytes] = []
    orientations: list[int] = []
    termination_counts: Counter[str] = Counter()
    common_configuration: dict | None = None
    common_teacher: dict | None = None
    for row in rows:
        label = labels[row.position_id]
        if not _label_lineage_matches(row, label):
            raise ValueError(f"Rank-4 retention label lineage differs: {row.position_id}")
        configuration = label.get("search_config")
        teacher = label.get("teacher")
        if (
            not isinstance(configuration, dict)
            or configuration != RANK4_FIXED_CONFIGURATION
            or not isinstance(teacher, dict)
        ):
            raise ValueError("Rank-4 retention label is not the frozen fixed-work profile")
        if common_configuration is None:
            common_configuration = dict(configuration)
            common_teacher = dict(teacher)
        elif configuration != common_configuration or teacher != common_teacher:
            raise ValueError("Rank-4 retention labels use different teachers/configurations")
        samples = corpus.sample_from_teacher_row(label)
        if (
            len(samples) != 2
            or samples[0].active != row.active
            or samples[1].active != features.reflect_active(row.active)
            or samples[0].group_id != row.root_group_id
            or samples[1].group_id != row.root_group_id
        ):
            raise ValueError("Rank-4 retention label feature lineage is stale")
        reason = str(label.get("search_stats", {}).get("termination_reason"))
        termination_counts[reason] += 1
        for orientation, sample in enumerate(samples):
            active_rows.append(sample.active)
            targets.append(float(sample.target))
            weights.append(float(sample.weight))
            roots.append(hashlib.sha256(row.root_group_id.encode("utf-8")).digest())
            positions.append(hashlib.sha256(row.position_id.encode("utf-8")).digest())
            fingerprints.append(row.canonical_fingerprint)
            orientations.append(orientation)
    if common_configuration is None or common_teacher is None:
        raise ValueError("Rank-4 retention labels are empty")
    indptr = np.zeros(len(active_rows) + 1, dtype="<i8")
    for index, active in enumerate(active_rows):
        indptr[index + 1] = indptr[index] + len(active)
    arrays = {
        "indptr": indptr,
        "indices": np.fromiter(
            (feature for active in active_rows for feature in active),
            dtype="<u2",
            count=int(indptr[-1]),
        ),
        "targets": np.asarray(targets, dtype="<f4"),
        "weights": np.asarray(weights, dtype="<f4"),
        "root_group_ids": np.asarray(roots, dtype="V32"),
        "position_ids": np.asarray(positions, dtype="V32"),
        "canonical_fingerprints": np.asarray(fingerprints, dtype="V32"),
        "orientations": np.asarray(orientations, dtype="u1"),
    }
    return arrays, dict(sorted(termination_counts.items())), {
        "teacher": common_teacher,
        "search_config": common_configuration,
    }


def pack_holdout(
    *,
    frozen_positions: pathlib.Path,
    freeze_manifest: pathlib.Path,
    labels: pathlib.Path,
    selection_receipt: pathlib.Path,
    teacher_source_sha256: str,
    output_directory: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, dict]:
    """Validate every label and publish a retention-only reflected CSR shard."""

    if not _valid_sha256(teacher_source_sha256):
        raise ValueError("Rank-4 retention teacher source SHA-256 is invalid")
    freeze, rows = load_freeze(frozen_positions, freeze_manifest)
    # This receipt is deliberately opened and frozen before labels are read.
    selection_snapshot = artifact_snapshot(selection_receipt)
    training_snapshot = freeze.get("timing", {}).get("training_inputs_frozen_by")
    if training_snapshot == selection_snapshot:
        raise ValueError("retention selection receipt cannot equal its input freeze receipt")
    label_rows = _load_rank4_labels(labels)
    expected_ids = {row.position_id for row in rows}
    if set(label_rows) != expected_ids:
        missing = len(expected_ids - set(label_rows))
        extra = len(set(label_rows) - expected_ids)
        raise ValueError(
            f"Rank-4 retention labels do not exactly cover frozen positions: "
            f"missing={missing} extra={extra}"
        )
    for value in label_rows.values():
        if (
            value.get("schema") != corpus.RANK4_TEACHER_SCHEMA
            or value.get("campaign_id") != freeze.get("campaign_id")
            or value.get("teacher", {}).get("source_sha256")
            != teacher_source_sha256
        ):
            raise ValueError("Rank-4 retention label identity is invalid")
    arrays, termination_counts, teacher_configuration = _holdout_arrays(
        rows, label_rows
    )
    expected_samples = len(rows) * 2
    if len(arrays["targets"]) != expected_samples or sum(
        termination_counts.values()
    ) != len(rows):
        raise RuntimeError("retention packing did not preserve every frozen label")

    npz_payload = training.deterministic_npz(arrays)
    npz_path = _write_content_addressed(output_directory, ".npz", npz_payload)
    manifest: dict[str, object] = {
        "schema": SHARD_SCHEMA,
        "profile": freeze.get("profile"),
        "campaign_id": freeze.get("campaign_id"),
        "feature_schema": features.FEATURE_SCHEMA,
        "role": freeze.get("role"),
        "teacher_split": TEACHER_SPLIT,
        "training_eligible": False,
        "training_loader_compatible": False,
        "npz": npz_path.name,
        "npz_sha256": hashlib.sha256(npz_payload).hexdigest(),
        "base_positions": len(rows),
        "samples": expected_samples,
        "root_groups": freeze.get("configuration", {}).get("groups"),
        "rows_per_group": ROWS_PER_GROUP,
        "reflection_rows_per_position": 2,
        "active_features": int(arrays["indices"].shape[0]),
        "termination_counts": termination_counts,
        "teacher_configuration": teacher_configuration,
        "target_policy": corpus.target_policy_for_schema(
            corpus.RANK4_TEACHER_SCHEMA
        ),
        "array_contract": {
            "indptr": "little-endian-int64[n+1]",
            "indices": "little-endian-uint16[nnz]",
            "targets": "little-endian-float32[n]",
            "weights": "little-endian-float32[n]",
            "root_group_ids": "raw-sha256-32bytes[n]",
            "position_ids": "raw-sha256-32bytes[n]",
            "canonical_fingerprints": "raw-sha256-32bytes[n]",
            "orientations": "uint8[n];0=original,1=reflection",
        },
        "inputs": {
            "freeze_manifest": artifact_snapshot(freeze_manifest),
            "frozen_positions": artifact_snapshot(frozen_positions),
            "labels": artifact_snapshot(labels),
        },
        "reveal": {
            "policy": "labels-opened-only-after-model-selection-receipt",
            "selection_receipt": selection_snapshot,
            "training_input_receipt": training_snapshot,
        },
        "producer": _producer_identity(),
    }
    manifest["body_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = _write_content_addressed(
        output_directory, ".json", manifest_payload
    )
    return npz_path, manifest_path, manifest


def load_holdout_shard(manifest_path: pathlib.Path) -> HoldoutShard:
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("retention holdout manifest is invalid JSON") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("feature_schema") != features.FEATURE_SCHEMA
        or manifest.get("training_eligible") is not False
        or manifest.get("training_loader_compatible") is not False
        or manifest.get("teacher_split") != TEACHER_SPLIT
        or manifest_path.stem != hashlib.sha256(manifest_bytes).hexdigest()
        or manifest_bytes != canonical_json_bytes(manifest)
    ):
        raise ValueError("retention holdout manifest identity is invalid")
    _verify_body_hash(manifest, "retention holdout manifest")
    inputs = manifest.get("inputs")
    reveal = manifest.get("reveal")
    if not isinstance(inputs, dict) or not isinstance(reveal, dict):
        raise ValueError("retention holdout provenance is missing")
    for name in ("freeze_manifest", "frozen_positions", "labels"):
        _validate_snapshot(inputs.get(name), f"retention holdout {name}")
    _validate_snapshot(
        reveal.get("selection_receipt"), "retention holdout selection receipt"
    )
    _validate_snapshot(
        reveal.get("training_input_receipt"),
        "retention holdout training-input receipt",
    )
    _validate_producer_identity(manifest.get("producer"))
    npz_name, expected_sha = manifest.get("npz"), manifest.get("npz_sha256")
    if (
        not isinstance(npz_name, str)
        or pathlib.PurePath(npz_name).name != npz_name
        or not _valid_sha256(expected_sha)
    ):
        raise ValueError("retention holdout NPZ identity is invalid")
    npz_path = manifest_path.parent / npz_name
    if (
        not npz_path.is_file()
        or npz_path.stem != expected_sha
        or sha256_file(npz_path) != expected_sha
    ):
        raise ValueError("retention holdout NPZ hash is invalid")
    expected_arrays = {
        "indptr",
        "indices",
        "targets",
        "weights",
        "root_group_ids",
        "position_ids",
        "canonical_fingerprints",
        "orientations",
    }
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise ValueError("retention holdout arrays are incomplete")
            arrays = {name: archive[name].copy() for name in expected_arrays}
    except (EOFError, OSError, ValueError) as error:
        raise ValueError("retention holdout NPZ is invalid") from error
    targets = arrays["targets"]
    count = int(targets.shape[0]) if targets.ndim == 1 else -1
    if (
        count <= 0
        or arrays["indptr"].dtype != np.dtype("<i8")
        or arrays["indices"].dtype != np.dtype("<u2")
        or arrays["targets"].dtype != np.dtype("<f4")
        or arrays["weights"].dtype != np.dtype("<f4")
        or arrays["root_group_ids"].dtype != np.dtype("V32")
        or arrays["position_ids"].dtype != np.dtype("V32")
        or arrays["canonical_fingerprints"].dtype != np.dtype("V32")
        or arrays["orientations"].dtype != np.dtype("u1")
        or arrays["indptr"].ndim != 1
        or arrays["indices"].ndim != 1
        or arrays["indptr"].shape != (count + 1,)
        or any(
            arrays[name].shape != (count,)
            for name in (
                "weights",
                "root_group_ids",
                "position_ids",
                "canonical_fingerprints",
                "orientations",
            )
        )
        or int(arrays["indptr"][0]) != 0
        or int(arrays["indptr"][-1]) != len(arrays["indices"])
        or np.any(arrays["indptr"][1:] < arrays["indptr"][:-1])
        or np.any(arrays["indices"] >= features.INPUT_COUNT)
        or not np.all(np.isfinite(targets))
        or np.any(np.abs(targets) > 1.0)
        or not np.all(np.isfinite(arrays["weights"]))
        or np.any(arrays["weights"] <= 0.0)
    ):
        raise ValueError("retention holdout array contract is invalid")
    shard = HoldoutShard(
        manifest,
        arrays["indptr"],
        arrays["indices"],
        targets,
        arrays["weights"],
        arrays["root_group_ids"],
        arrays["position_ids"],
        arrays["canonical_fingerprints"],
        arrays["orientations"],
    )
    base_positions = manifest.get("base_positions")
    root_groups = manifest.get("root_groups")
    if (
        not isinstance(base_positions, int)
        or base_positions <= 0
        or len(shard) != base_positions * 2
        or manifest.get("samples") != len(shard)
        or manifest.get("active_features") != len(shard.indices)
        or manifest.get("reflection_rows_per_position") != 2
        or manifest.get("rows_per_group") != ROWS_PER_GROUP
        or not isinstance(root_groups, int)
        or root_groups <= 0
        or base_positions != root_groups * ROWS_PER_GROUP
    ):
        raise ValueError("retention holdout exact counts are invalid")
    position_counts = Counter(bytes(value) for value in shard.position_ids)
    fingerprint_counts = Counter(bytes(value) for value in shard.canonical_fingerprints)
    root_counts = Counter(bytes(value) for value in shard.root_group_ids)
    if (
        len(position_counts) != base_positions
        or set(position_counts.values()) != {2}
        or len(fingerprint_counts) != base_positions
        or set(fingerprint_counts.values()) != {2}
        or len(root_counts) != root_groups
        or set(root_counts.values()) != {ROWS_PER_GROUP * 2}
    ):
        raise ValueError("retention holdout group/fingerprint counts are invalid")
    for base in range(base_positions):
        first, second = 2 * base, 2 * base + 1
        first_active = features.validate_active(shard.active(first).tolist())
        second_active = features.validate_active(shard.active(second).tolist())
        fingerprint = corpus.canonical_feature_fingerprint(first_active)
        if (
            int(shard.orientations[first]) != 0
            or int(shard.orientations[second]) != 1
            or bytes(shard.position_ids[first]) != bytes(shard.position_ids[second])
            or bytes(shard.root_group_ids[first]) != bytes(shard.root_group_ids[second])
            or bytes(shard.canonical_fingerprints[first]) != fingerprint
            or bytes(shard.canonical_fingerprints[second]) != fingerprint
            or second_active != features.reflect_active(first_active)
            or shard.targets[first] != shard.targets[second]
            or shard.weights[first] != shard.weights[second]
        ):
            raise ValueError("retention reflection pair contract is invalid")
    return shard


def _predictions(
    parameters: Mapping[str, np.ndarray], shard: HoldoutShard, batch_size: int = 4096
) -> np.ndarray:
    output = np.empty(len(shard), dtype=np.float32)
    for start in range(0, len(shard), batch_size):
        stop = min(start + batch_size, len(shard))
        output[start:stop], _ = training.forward(
            parameters,
            tuple(shard.active(row) for row in range(start, stop)),
        )
    return output


def _point_metrics(
    predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> dict[str, float | int]:
    difference = predictions - targets
    # Use the trainer's canonical normalization so this point gate is exactly
    # comparable to the retained 110,004-row anchor gate.
    weighted_huber, _gradient = training._weighted_huber(
        predictions, targets, weights, 0.25
    )
    sign_accuracy = float(
        np.mean((predictions >= 0.0) == (targets >= 0.0))
    )
    if len(targets) > 1 and np.std(predictions) > 0 and np.std(targets) > 0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    return {
        "samples": len(targets),
        "weighted_huber": weighted_huber,
        "sign_accuracy": sign_accuracy,
        "correlation": correlation,
        "mae": float(np.mean(np.abs(difference))),
        "prediction_mean": float(np.mean(predictions)),
    }


def _normalized_group_ids(values: Sequence[object] | np.ndarray) -> list[bytes]:
    result = []
    for value in values:
        if isinstance(value, np.void):
            result.append(bytes(value))
        elif isinstance(value, bytes):
            result.append(value)
        elif isinstance(value, str):
            result.append(hashlib.sha256(value.encode("utf-8")).digest())
        else:
            raise ValueError("retention root group identity is invalid")
    return result


def noninferiority_evidence(
    *,
    actor_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    root_group_ids: Sequence[object] | np.ndarray,
    thresholds: NoninferiorityThresholds = FROZEN_THRESHOLDS,
) -> dict:
    """Return frozen point and paired root-cluster NI evidence."""

    thresholds.validate()
    actor_predictions = np.asarray(actor_predictions, dtype=np.float32)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    count = len(targets)
    if (
        count <= 0
        or actor_predictions.shape != (count,)
        or candidate_predictions.shape != (count,)
        or weights.shape != (count,)
        or len(root_group_ids) != count
        or not np.all(np.isfinite(actor_predictions))
        or not np.all(np.isfinite(candidate_predictions))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("retention noninferiority inputs are invalid")
    groups = _normalized_group_ids(root_group_ids)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("retention NI requires at least two root groups")
    group_index = {group: index for index, group in enumerate(unique_groups)}
    inverse = np.fromiter(
        (group_index[group] for group in groups), dtype=np.int64, count=count
    )
    group_count = len(unique_groups)

    actor_correct = (actor_predictions >= 0.0) == (targets >= 0.0)
    candidate_correct = (candidate_predictions >= 0.0) == (targets >= 0.0)
    actor_difference = np.abs(actor_predictions - targets)
    candidate_difference = np.abs(candidate_predictions - targets)
    actor_loss = np.where(
        actor_difference <= 0.25,
        0.5 * actor_difference * actor_difference,
        0.25 * (actor_difference - 0.125),
    )
    candidate_loss = np.where(
        candidate_difference <= 0.25,
        0.5 * candidate_difference * candidate_difference,
        0.25 * (candidate_difference - 0.125),
    )
    group_rows = np.bincount(inverse, minlength=group_count).astype(np.float64)
    group_actor_correct = np.bincount(
        inverse, weights=actor_correct.astype(np.float64), minlength=group_count
    )
    group_candidate_correct = np.bincount(
        inverse, weights=candidate_correct.astype(np.float64), minlength=group_count
    )
    group_actor_loss = np.bincount(
        inverse, weights=(weights * actor_loss).astype(np.float64), minlength=group_count
    )
    group_candidate_loss = np.bincount(
        inverse,
        weights=(weights * candidate_loss).astype(np.float64),
        minlength=group_count,
    )
    if float(np.sum(group_actor_loss)) <= 0.0:
        raise ValueError("retention actor Huber loss is zero; ratio NI is undefined")

    replicates = thresholds.bootstrap_replicates
    sign_distribution = np.empty(replicates, dtype="<f8")
    huber_distribution = np.empty(replicates, dtype="<f8")
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    offset = 0
    batch_size = min(256, replicates)
    while offset < replicates:
        take = min(batch_size, replicates - offset)
        sampled = rng.integers(
            0, group_count, size=(take, group_count), dtype=np.int64
        )
        sampled_rows = group_rows[sampled].sum(axis=1)
        sign_distribution[offset : offset + take] = (
            group_candidate_correct[sampled].sum(axis=1)
            - group_actor_correct[sampled].sum(axis=1)
        ) / sampled_rows
        actor_loss_sum = group_actor_loss[sampled].sum(axis=1)
        if np.any(actor_loss_sum <= 0.0):
            raise ValueError("retention bootstrap sampled zero actor Huber loss")
        huber_distribution[offset : offset + take] = (
            group_candidate_loss[sampled].sum(axis=1) / actor_loss_sum
        )
        offset += take
    sign_distribution.sort()
    huber_distribution.sort()
    alpha = 1.0 - thresholds.confidence
    lower_index = min(
        replicates - 1, max(0, math.ceil(alpha * replicates) - 1)
    )
    upper_index = min(
        replicates - 1,
        max(0, math.ceil(thresholds.confidence * replicates) - 1),
    )
    sign_lower = float(sign_distribution[lower_index])
    huber_upper = float(huber_distribution[upper_index])

    actor_metrics = _point_metrics(actor_predictions, targets, weights)
    candidate_metrics = _point_metrics(candidate_predictions, targets, weights)
    actor_huber = float(actor_metrics["weighted_huber"])
    candidate_huber = float(candidate_metrics["weighted_huber"])
    point_sign_delta = float(candidate_metrics["sign_accuracy"]) - float(
        actor_metrics["sign_accuracy"]
    )
    point_huber_ratio = candidate_huber / actor_huber
    gates = {
        "point_sign": point_sign_delta >= -thresholds.sign_accuracy_margin,
        "point_huber": point_huber_ratio <= thresholds.weighted_huber_multiplier,
        "cluster_sign": sign_lower >= -thresholds.sign_accuracy_margin,
        "cluster_huber": huber_upper <= thresholds.weighted_huber_multiplier,
    }
    distribution_hash = hashlib.sha256(
        b"papersoccer-retention-bootstrap-v1\0"
        + sign_distribution.tobytes(order="C")
        + huber_distribution.tobytes(order="C")
    ).hexdigest()
    return {
        "thresholds": thresholds.record(),
        "root_groups": group_count,
        "actor": actor_metrics,
        "candidate": candidate_metrics,
        "point": {
            "sign_accuracy_delta": point_sign_delta,
            "weighted_huber_ratio": point_huber_ratio,
        },
        "root_cluster": {
            "sign_accuracy_delta_lower_bound": sign_lower,
            "weighted_huber_ratio_upper_bound": huber_upper,
            "lower_order_index": lower_index,
            "upper_order_index": upper_index,
            "distribution_sha256": distribution_hash,
        },
        "gates": {**gates, "pass": all(gates.values())},
    }


def _json_contains(value: object, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_json_contains(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_json_contains(item, expected) for item in value)
    return False


def evaluate_holdout(
    *,
    shard_manifest: pathlib.Path,
    actor_runtime: pathlib.Path,
    candidate_runtime: pathlib.Path,
    selection_receipt: pathlib.Path,
    output: pathlib.Path | None = None,
) -> dict:
    """Bind one selected candidate to conjunctive point and cluster NI gates."""

    selection_snapshot = artifact_snapshot(selection_receipt)
    selection = _load_json(selection_receipt, "retention model-selection receipt")
    candidate_snapshot = artifact_snapshot(candidate_runtime)
    if not _json_contains(selection, str(candidate_snapshot["sha256"])):
        raise ValueError("selection receipt does not bind the candidate runtime SHA-256")
    shard = load_holdout_shard(shard_manifest)
    if shard.manifest.get("reveal", {}).get("selection_receipt") != selection_snapshot:
        raise ValueError("retention shard and evaluation use different selection receipts")
    actor_parameters, actor_report = training.load_runtime(actor_runtime)
    candidate_parameters, candidate_report = training.load_runtime(candidate_runtime)
    evidence = noninferiority_evidence(
        actor_predictions=_predictions(actor_parameters, shard),
        candidate_predictions=_predictions(candidate_parameters, shard),
        targets=shard.targets,
        weights=shard.weights,
        root_group_ids=shard.root_group_ids,
    )
    report: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "profile": shard.manifest.get("profile"),
        "campaign_id": shard.manifest.get("campaign_id"),
        "role": shard.manifest.get("role"),
        "training_eligible": False,
        "inputs": {
            "shard_manifest": artifact_snapshot(shard_manifest),
            "actor_runtime": artifact_snapshot(actor_runtime),
            "candidate_runtime": candidate_snapshot,
            "selection_receipt": selection_snapshot,
        },
        "runtime_reports": {
            "actor": actor_report,
            "candidate": candidate_report,
        },
        "reveal": {
            "policy": "metrics-computed-only-after-selected-runtime-binding",
            "labels_reveal": shard.manifest.get("reveal"),
        },
        "noninferiority": evidence,
        "pass": evidence["gates"]["pass"],
        "producer": _producer_identity(),
    }
    report["body_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    if output is not None:
        _atomic_write(output, canonical_json_bytes(report))
    return report


def _profile(value: str) -> FreezeSpec:
    try:
        return PROFILES[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError("profile must be pilot or full") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--profile", type=_profile, required=True)
    freeze.add_argument("--campaign-id", required=True)
    freeze.add_argument("--candidate-positions", type=pathlib.Path, required=True)
    freeze.add_argument("--training-input-receipt", type=pathlib.Path, required=True)
    freeze.add_argument(
        "--exclude-shard-manifest", action="append", type=pathlib.Path, default=[]
    )
    freeze.add_argument(
        "--exclude-positions", action="append", type=pathlib.Path, default=[]
    )
    freeze.add_argument(
        "--exclude-root-manifest", action="append", type=pathlib.Path, default=[]
    )
    freeze.add_argument("--output-positions", type=pathlib.Path, required=True)
    freeze.add_argument("--output-manifest", type=pathlib.Path, required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--frozen-positions", type=pathlib.Path, required=True)
    pack.add_argument("--freeze-manifest", type=pathlib.Path, required=True)
    pack.add_argument("--labels", type=pathlib.Path, required=True)
    pack.add_argument("--selection-receipt", type=pathlib.Path, required=True)
    pack.add_argument("--teacher-source-sha256", required=True)
    pack.add_argument("--output-directory", type=pathlib.Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--shard-manifest", type=pathlib.Path, required=True)
    evaluate.add_argument("--actor-runtime", type=pathlib.Path, required=True)
    evaluate.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    evaluate.add_argument("--selection-receipt", type=pathlib.Path, required=True)
    evaluate.add_argument("--output", type=pathlib.Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "freeze":
        payload, manifest = freeze_candidate_groups(
            candidate_positions=arguments.candidate_positions,
            training_input_receipt=arguments.training_input_receipt,
            campaign_id=arguments.campaign_id,
            spec=arguments.profile,
            excluded_shard_manifests=arguments.exclude_shard_manifest,
            excluded_position_tsvs=arguments.exclude_positions,
            excluded_root_manifests=arguments.exclude_root_manifest,
        )
        write_freeze(
            arguments.output_positions,
            arguments.output_manifest,
            payload,
            manifest,
        )
        print(json.dumps(manifest["selection"], indent=2, sort_keys=True))
    elif arguments.command == "pack":
        _npz, manifest_path, manifest = pack_holdout(
            frozen_positions=arguments.frozen_positions,
            freeze_manifest=arguments.freeze_manifest,
            labels=arguments.labels,
            selection_receipt=arguments.selection_receipt,
            teacher_source_sha256=arguments.teacher_source_sha256,
            output_directory=arguments.output_directory,
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "samples": manifest["samples"],
                    "termination_counts": manifest["termination_counts"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        report = evaluate_holdout(
            shard_manifest=arguments.shard_manifest,
            actor_runtime=arguments.actor_runtime,
            candidate_runtime=arguments.candidate_runtime,
            selection_receipt=arguments.selection_receipt,
            output=arguments.output,
        )
        print(json.dumps(report["noninferiority"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
