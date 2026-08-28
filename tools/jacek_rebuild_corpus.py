#!/usr/bin/env python3
"""Freeze and verify the provenance-bound replay-rebuild corpus.

The rebuild corpus is deliberately separate from the campaign packer.  It
binds the immutable canonical train/validation/test shards, materializes one
deduplicated train shard for each new-data teacher, and materializes the
deduplicated adjudicator validation shard.  Deduplication is over the replay
feature schema's rotate/reflection equivalence class and always considers v6
before v5.

The protected canonical test shards are present in the evidence manifest, but
are never returned by a training interface.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402
import jacek_replay_train as training  # noqa: E402


MANIFEST_SCHEMA = "papersoccer.jacek-rebuild-corpus.v1"
DEDUPLICATION_SCHEMA = "papersoccer.jacek-rebuild-deduplication.v1"
PRODUCER_SCHEMA = "papersoccer.jacek-rebuild-corpus-producer.v1"

EXPECTED_CANONICAL_COUNTS = {
    "train": 997_914,
    "validation": 110_004,
    "test": 121_052,
}
CAMPAIGN_PRECEDENCE = ("v6", "v5")
ROLE_ORDER = ("train", "validation", "test")

_INPUT_SPECS = {
    "canonical_train": ("canonical", "canonical", "train"),
    "canonical_validation": ("canonical", "canonical", "validation"),
    "canonical_test": ("canonical", "canonical", "test"),
    "v5_search_train": ("v5", "search", "train"),
    "v6_search_train": ("v6", "search", "train"),
    "v5_rank4_train": ("v5", "rank4", "train"),
    "v6_rank4_train": ("v6", "rank4", "train"),
    "v5_adjudicator_validation": ("v5", "adjudicator", "validation"),
    "v6_adjudicator_validation": ("v6", "adjudicator", "validation"),
}
_DEDUP_INPUTS = {
    "search": ("v6_search_train", "v5_search_train"),
    "rank4": ("v6_rank4_train", "v5_rank4_train"),
    "adjudicator": (
        "v6_adjudicator_validation",
        "v5_adjudicator_validation",
    ),
}
_DEDUP_ROLES = {
    "search": "train",
    "rank4": "train",
    "adjudicator": "validation",
}


def canonical_json_bytes(value: object) -> bytes:
    return corpus.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _producer_identity() -> dict[str, object]:
    return {
        "schema": PRODUCER_SCHEMA,
        "tool_sha256": {
            "rebuild_corpus": sha256_file(pathlib.Path(__file__)),
            "replay_corpus": sha256_file(pathlib.Path(corpus.__file__)),
            "replay_features": sha256_file(pathlib.Path(features.__file__)),
            "replay_train": sha256_file(pathlib.Path(training.__file__)),
        },
    }


def _relative_path(path: pathlib.Path, base: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _resolve_bound_path(
    base: pathlib.Path, supplied: object, label: str
) -> pathlib.Path:
    if not isinstance(supplied, str) or not supplied or "\x00" in supplied:
        raise ValueError(f"{label} path is invalid")
    if pathlib.PurePath(supplied).is_absolute():
        raise ValueError(f"{label} path must be relative to the corpus manifest")
    resolved = (base / supplied).resolve()
    if _relative_path(resolved, base) != pathlib.PurePath(supplied).as_posix():
        raise ValueError(f"{label} path is not normalized")
    return resolved


def _read_canonical_json(path: pathlib.Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON: {path}")
    return value, raw


def _source_identity_and_shard(
    manifest_path: pathlib.Path,
    base: pathlib.Path,
    *,
    campaign: str,
    channel: str,
    role: str,
) -> tuple[dict[str, object], training.SparseShard]:
    manifest_path = manifest_path.resolve()
    manifest, raw = _read_canonical_json(manifest_path, "source shard manifest")
    shard = training.load_csr_shard(manifest_path)
    if shard.split != role:
        raise ValueError(
            f"{campaign} {channel} source must use split {role}, not {shard.split}"
        )
    npz_name = manifest.get("npz")
    if not isinstance(npz_name, str):
        raise ValueError("source shard has no NPZ path")
    npz_path = (manifest_path.parent / npz_name).resolve()
    identity = {
        "campaign": campaign,
        "channel": channel,
        "role": role,
        "manifest_path": _relative_path(manifest_path, base),
        "manifest_sha256": sha256_bytes(raw),
        "npz_path": _relative_path(npz_path, base),
        "npz_sha256": shard.npz_sha256,
        "rows": len(shard),
        "active_features": int(shard.indices.shape[0]),
        "source_provenance_sha256": sha256_bytes(
            canonical_json_bytes(manifest.get("provenance", {}))
        ),
    }
    return identity, shard


def _source_identity(
    manifest_path: pathlib.Path,
    base: pathlib.Path,
    *,
    campaign: str,
    channel: str,
    role: str,
) -> dict[str, object]:
    identity, shard = _source_identity_and_shard(
        manifest_path,
        base,
        campaign=campaign,
        channel=channel,
        role=role,
    )
    del shard
    return identity


def _sealed_source_identity(
    manifest_path: pathlib.Path,
    base: pathlib.Path,
    *,
    campaign: str,
    channel: str,
    role: str,
) -> dict[str, object]:
    """Bind a protected shard without decoding its target or weight arrays."""

    manifest_path = manifest_path.resolve()
    manifest, raw = _read_canonical_json(
        manifest_path, "sealed source shard manifest"
    )
    npz_name = manifest.get("npz")
    npz_sha256 = manifest.get("npz_sha256")
    rows = manifest.get("samples")
    active_features = manifest.get("active_features")
    if (
        manifest.get("schema") != training.SHARD_SCHEMA
        or manifest.get("split") != role
        or not isinstance(npz_name, str)
        or not isinstance(npz_sha256, str)
        or sha256_file(manifest_path.parent / npz_name) != npz_sha256
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or isinstance(active_features, bool)
        or not isinstance(active_features, int)
        or active_features <= 0
    ):
        raise ValueError("sealed source shard identity is invalid")
    npz_path = (manifest_path.parent / npz_name).resolve()
    return {
        "campaign": campaign,
        "channel": channel,
        "role": role,
        "manifest_path": _relative_path(manifest_path, base),
        "manifest_sha256": sha256_bytes(raw),
        "npz_path": _relative_path(npz_path, base),
        "npz_sha256": npz_sha256,
        "rows": rows,
        "active_features": active_features,
        "source_provenance_sha256": sha256_bytes(
            canonical_json_bytes(manifest.get("provenance", {}))
        ),
    }


def _load_bound_source(
    base: pathlib.Path,
    identity: Mapping[str, object],
    *,
    campaign: str,
    channel: str,
    role: str,
) -> training.SparseShard:
    expected_keys = {
        "campaign",
        "channel",
        "role",
        "manifest_path",
        "manifest_sha256",
        "npz_path",
        "npz_sha256",
        "rows",
        "active_features",
        "source_provenance_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_keys:
        raise ValueError("source shard identity fields are not frozen")
    path = _resolve_bound_path(base, identity.get("manifest_path"), "source manifest")
    observed, shard = _source_identity_and_shard(
        path,
        base,
        campaign=campaign,
        channel=channel,
        role=role,
    )
    if observed != dict(identity):
        del shard
        raise ValueError("source shard identity or counts changed")
    return shard


def _bind_inputs(
    base: pathlib.Path,
    supplied: Mapping[str, Sequence[pathlib.Path]],
) -> dict[str, list[dict[str, object]]]:
    if set(supplied) != set(_INPUT_SPECS):
        missing = sorted(set(_INPUT_SPECS) - set(supplied))
        extra = sorted(set(supplied) - set(_INPUT_SPECS))
        raise ValueError(
            f"rebuild corpus input keys differ; missing={missing}, extra={extra}"
        )
    result: dict[str, list[dict[str, object]]] = {}
    for key, (campaign, channel, role) in _INPUT_SPECS.items():
        paths = tuple(pathlib.Path(path) for path in supplied[key])
        if not paths:
            raise ValueError(f"rebuild corpus input {key} is empty")
        identity_loader = (
            _sealed_source_identity if key == "canonical_test" else _source_identity
        )
        identities = [
            identity_loader(
                path, base, campaign=campaign, channel=channel, role=role
            )
            for path in paths
        ]
        identities.sort(
            key=lambda item: (
                str(item["manifest_sha256"]), str(item["manifest_path"])
            )
        )
        unique_paths = {str(item["manifest_path"]) for item in identities}
        if len(unique_paths) != len(identities):
            raise ValueError(f"rebuild corpus input {key} repeats a source manifest")
        result[key] = identities
    return result


def _reload_inputs(
    base: pathlib.Path, value: object
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(value, dict) or set(value) != set(_INPUT_SPECS):
        raise ValueError("rebuild corpus input bindings are incomplete")
    result: dict[str, list[dict[str, object]]] = {}
    for key in _INPUT_SPECS:
        identities = value.get(key)
        if not isinstance(identities, list) or not identities:
            raise ValueError(f"rebuild corpus input {key} is empty")
        normalized: list[dict[str, object]] = []
        for identity in identities:
            if not isinstance(identity, dict):
                raise ValueError(f"rebuild corpus input {key} identity is invalid")
            _resolve_bound_path(
                base, identity.get("manifest_path"), f"rebuild corpus input {key}"
            )
            normalized.append(dict(identity))
        if normalized != sorted(
            normalized,
            key=lambda item: (
                str(item["manifest_sha256"]), str(item["manifest_path"])
            ),
        ):
            raise ValueError(f"rebuild corpus input {key} is not canonically ordered")
        if len({str(item["manifest_path"]) for item in normalized}) != len(normalized):
            raise ValueError(f"rebuild corpus input {key} repeats a source manifest")
        result[key] = normalized
    return result


def _normalized_expected_counts(
    supplied: Mapping[str, int] | None,
) -> dict[str, int]:
    counts = dict(EXPECTED_CANONICAL_COUNTS if supplied is None else supplied)
    if set(counts) != set(ROLE_ORDER) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError(
            "canonical count contract must bind positive train/validation/test"
        )
    return {role: counts[role] for role in ROLE_ORDER}


def _validate_canonical_counts(
    inputs: Mapping[str, Sequence[Mapping[str, object]]],
    expected: Mapping[str, int],
) -> None:
    for role in ROLE_ORDER:
        observed = sum(
            int(identity["rows"]) for identity in inputs[f"canonical_{role}"]
        )
        if observed != expected[role]:
            raise ValueError(
                f"canonical {role} row count is {observed}, expected {expected[role]}"
            )


@dataclasses.dataclass(frozen=True)
class _ChosenRow:
    active: np.ndarray
    target: np.float32
    weight: np.float32
    group_id: bytes


@dataclasses.dataclass(frozen=True)
class _DeduplicationResult:
    channel: str
    role: str
    arrays: dict[str, np.ndarray]
    selection: dict[str, object]
    selection_sha256: str

    def as_shard(self) -> training.SparseShard:
        return training.SparseShard(
            self.arrays["indptr"],
            self.arrays["indices"],
            self.arrays["targets"],
            self.arrays["weights"],
            self.arrays["group_ids"],
            self.role,
        )


def _deduplicate_channel(
    base: pathlib.Path,
    channel: str,
    inputs: Mapping[str, Sequence[Mapping[str, object]]],
) -> _DeduplicationResult:
    role = _DEDUP_ROLES[channel]
    chosen: dict[bytes, _ChosenRow] = {}
    source_reports: list[dict[str, object]] = []
    input_rows = 0
    for input_key in _DEDUP_INPUTS[channel]:
        campaign, expected_channel, expected_role = _INPUT_SPECS[input_key]
        if expected_channel != channel or expected_role != role:
            raise RuntimeError(
                "internal rebuild deduplication specification is inconsistent"
            )
        for identity in inputs[input_key]:
            shard = _load_bound_source(
                base,
                identity,
                campaign=campaign,
                channel=channel,
                role=role,
            )
            selected_indices: list[int] = []
            input_rows += len(shard)
            for row in range(len(shard)):
                active = shard.active(row)
                fingerprint = corpus.canonical_feature_fingerprint(active.tolist())
                if fingerprint in chosen:
                    continue
                chosen[fingerprint] = _ChosenRow(
                    active.copy(),
                    np.float32(shard.targets[row]),
                    np.float32(shard.weights[row]),
                    bytes(shard.group_ids[row]),
                )
                selected_indices.append(row)
            source_reports.append(
                {
                    "input_key": input_key,
                    "campaign": campaign,
                    "manifest_path": identity["manifest_path"],
                    "manifest_sha256": identity["manifest_sha256"],
                    "input_rows": len(shard),
                    "retained_rows": len(selected_indices),
                    "discarded_rows": len(shard) - len(selected_indices),
                    "retained_row_indices": selected_indices,
                }
            )
            del shard

    ordered = sorted(chosen.items())
    if not ordered:
        raise ValueError(f"deduplicated {channel} corpus is empty")
    indptr = np.zeros(len(ordered) + 1, dtype="<i8")
    for row, (_fingerprint, selected) in enumerate(ordered):
        indptr[row + 1] = indptr[row] + int(selected.active.shape[0])
    indices = np.empty(int(indptr[-1]), dtype="<u2")
    targets = np.empty(len(ordered), dtype="<f4")
    weights = np.empty(len(ordered), dtype="<f4")
    group_ids = np.empty(len(ordered), dtype="V32")
    for row, (_fingerprint, selected) in enumerate(ordered):
        first, last = int(indptr[row]), int(indptr[row + 1])
        indices[first:last] = selected.active
        targets[row] = selected.target
        weights[row] = selected.weight
        group_ids[row] = selected.group_id
    selection = {
        "schema": DEDUPLICATION_SCHEMA,
        "channel": channel,
        "role": role,
        "campaign_precedence": list(CAMPAIGN_PRECEDENCE),
        "equivalence": "canonical-rotate-reflection-feature-fingerprint",
        "tie_break": "manifest-sha256-then-manifest-path-then-row-index",
        "input_rows": input_rows,
        "retained_rows": len(ordered),
        "discarded_rows": input_rows - len(ordered),
        "canonical_fingerprints_sha256": sha256_bytes(
            b"".join(fingerprint for fingerprint, _selected in ordered)
        ),
        "sources": source_reports,
    }
    return _DeduplicationResult(
        channel,
        role,
        {
            "group_ids": group_ids,
            "indices": indices,
            "indptr": indptr,
            "targets": targets,
            "weights": weights,
        },
        selection,
        sha256_bytes(canonical_json_bytes(selection)),
    )


def _generated_manifest(
    result: _DeduplicationResult,
    *,
    npz_name: str,
    npz_sha256: str,
    producer: Mapping[str, object],
) -> dict[str, object]:
    arrays = result.arrays
    return {
        "schema": training.SHARD_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "split": result.role,
        "npz": npz_name,
        "npz_sha256": npz_sha256,
        "samples": int(arrays["targets"].shape[0]),
        "active_features": int(arrays["indices"].shape[0]),
        "array_contract": {
            "indptr": "little-endian-int64[n+1]",
            "indices": "little-endian-uint16[nnz]",
            "targets": "little-endian-float32[n]",
            "weights": "little-endian-float32[n]",
            "group_ids": "raw-sha256-32bytes[n]",
        },
        "provenance": {
            "schema": DEDUPLICATION_SCHEMA,
            "channel": result.channel,
            "role": result.role,
            "campaign_precedence": list(CAMPAIGN_PRECEDENCE),
            "selection_sha256": result.selection_sha256,
            "canonical_fingerprints_sha256": result.selection[
                "canonical_fingerprints_sha256"
            ],
            "producer": dict(producer),
        },
    }


def _write_deduplicated_shard(
    base: pathlib.Path,
    result: _DeduplicationResult,
    producer: Mapping[str, object],
) -> dict[str, object]:
    directory = base / "deduplicated" / result.channel
    payload = training.deterministic_npz(result.arrays)
    npz_sha = sha256_bytes(payload)
    npz_path = directory / f"{npz_sha}.npz"
    training._write_once(npz_path, payload)
    manifest = _generated_manifest(
        result,
        npz_name=npz_path.name,
        npz_sha256=npz_sha,
        producer=producer,
    )
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = directory / f"{sha256_bytes(manifest_payload)}.json"
    training._write_once(manifest_path, manifest_payload)
    return _source_identity(
        manifest_path,
        base,
        campaign="rebuild",
        channel=result.channel,
        role=result.role,
    )


def _load_generated_shard(
    base: pathlib.Path,
    identity: object,
    result: _DeduplicationResult,
    producer: Mapping[str, object],
) -> training.SparseShard:
    if not isinstance(identity, Mapping):
        raise ValueError(f"deduplicated {result.channel} shard identity is missing")
    shard = _load_bound_source(
        base,
        identity,
        campaign="rebuild",
        channel=result.channel,
        role=result.role,
    )
    path = _resolve_bound_path(
        base, identity.get("manifest_path"), "generated manifest"
    )
    manifest, _raw = _read_canonical_json(path, "generated shard manifest")
    payload = training.deterministic_npz(result.arrays)
    npz_sha = sha256_bytes(payload)
    expected_manifest = _generated_manifest(
        result,
        npz_name=f"{npz_sha}.npz",
        npz_sha256=npz_sha,
        producer=producer,
    )
    if manifest != expected_manifest:
        raise ValueError(f"deduplicated {result.channel} shard provenance changed")
    for name in ("indptr", "indices", "targets", "weights", "group_ids"):
        if not np.array_equal(getattr(shard, name), result.arrays[name]):
            raise ValueError(f"deduplicated {result.channel} shard rows changed")
    return shard


def _ingest_identity_batch(
    connection: sqlite3.Connection,
    table: str,
    role_index: int,
    values: Sequence[bytes],
) -> None:
    connection.executemany(
        f"INSERT INTO {table}(value, role) VALUES (?, ?) "
        "ON CONFLICT(value) DO UPDATE SET role = "
        "CASE WHEN role = excluded.role THEN role ELSE -1 END",
        ((value, role_index) for value in values),
    )


def _ingest_shard_for_leakage(
    connection: sqlite3.Connection,
    role_index: int,
    shard: training.SparseShard,
) -> None:
    fingerprints: list[bytes] = []
    groups: list[bytes] = []
    for row in range(len(shard)):
        fingerprints.append(
            corpus.canonical_feature_fingerprint(shard.active(row).tolist())
        )
        groups.append(bytes(shard.group_ids[row]))
        if len(fingerprints) == 4096:
            _ingest_identity_batch(connection, "fingerprints", role_index, fingerprints)
            _ingest_identity_batch(connection, "root_groups", role_index, groups)
            fingerprints.clear()
            groups.clear()
    if fingerprints:
        _ingest_identity_batch(connection, "fingerprints", role_index, fingerprints)
        _ingest_identity_batch(connection, "root_groups", role_index, groups)


def _leakage_report(
    base: pathlib.Path,
    inputs: Mapping[str, Sequence[Mapping[str, object]]],
    deduplicated: Mapping[str, _DeduplicationResult],
) -> dict[str, object]:
    role_rows = {role: 0 for role in ROLE_ORDER}
    with tempfile.TemporaryDirectory(prefix=".jacek-rebuild-leakage.") as temporary:
        database = pathlib.Path(temporary) / "identities.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                "CREATE TABLE fingerprints(value BLOB PRIMARY KEY, role INTEGER)"
            )
            connection.execute(
                "CREATE TABLE root_groups(value BLOB PRIMARY KEY, role INTEGER)"
            )
            for role_index, role in enumerate(ROLE_ORDER):
                campaign, channel, expected_role = _INPUT_SPECS[f"canonical_{role}"]
                for identity in inputs[f"canonical_{role}"]:
                    if role == "test":
                        # Canonical test states and targets stay sealed until
                        # final qualification.  Their manifest/NPZ hashes and
                        # counts are bound here; canonical workflow receipts
                        # already prove original split isolation.
                        observed = _sealed_source_identity(
                            _resolve_bound_path(
                                base,
                                identity.get("manifest_path"),
                                "sealed source manifest",
                            ),
                            base,
                            campaign=campaign,
                            channel=channel,
                            role=expected_role,
                        )
                        if observed != dict(identity):
                            raise ValueError("sealed canonical test identity changed")
                        role_rows[role] += int(identity["rows"])
                    else:
                        shard = _load_bound_source(
                            base,
                            identity,
                            campaign=campaign,
                            channel=channel,
                            role=expected_role,
                        )
                        role_rows[role] += len(shard)
                        _ingest_shard_for_leakage(connection, role_index, shard)
                        del shard
                for channel_name, result in deduplicated.items():
                    if result.role != role:
                        continue
                    shard = result.as_shard()
                    role_rows[role] += len(shard)
                    _ingest_shard_for_leakage(connection, role_index, shard)
                connection.commit()
            fingerprint_conflict = connection.execute(
                "SELECT hex(value) FROM fingerprints WHERE role = -1 LIMIT 1"
            ).fetchone()
            if fingerprint_conflict is not None:
                raise ValueError(
                    "canonical feature fingerprint leaks across train/validation/test "
                    f"roles: {str(fingerprint_conflict[0]).lower()}"
                )
            group_conflict = connection.execute(
                "SELECT hex(value) FROM root_groups WHERE role = -1 LIMIT 1"
            ).fetchone()
            if group_conflict is not None:
                raise ValueError(
                    "root-group identity leaks across train/validation/test roles: "
                    f"{str(group_conflict[0]).lower()}"
                )
            roles = {}
            for role_index, role in enumerate(ROLE_ORDER):
                roles[role] = (
                    {
                        "rows_bound": role_rows[role],
                        "sealed": True,
                        "isolation_evidence": "canonical-workflow-provenance",
                        "arrays_decoded": False,
                    }
                    if role == "test"
                    else {
                        "rows_scanned": role_rows[role],
                        "unique_canonical_fingerprints": int(
                            connection.execute(
                                "SELECT COUNT(*) FROM fingerprints WHERE role = ?",
                                (role_index,),
                            ).fetchone()[0]
                        ),
                        "unique_root_groups": int(
                            connection.execute(
                                "SELECT COUNT(*) FROM root_groups WHERE role = ?",
                                (role_index,),
                            ).fetchone()[0]
                        ),
                    }
                )
        finally:
            connection.close()
    return {
        "policy": (
            "train-validation-fingerprint-and-root-isolation;"
            "canonical-test-sealed-by-upstream-workflow-provenance"
        ),
        "roles": roles,
        "verified": True,
    }


def _manifest_paths(
    identities: Sequence[Mapping[str, object]],
) -> list[str]:
    return [str(identity["manifest_path"]) for identity in identities]


def _interfaces(
    inputs: Mapping[str, Sequence[Mapping[str, object]]],
    generated: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    canonical_train = _manifest_paths(inputs["canonical_train"])
    canonical_validation = _manifest_paths(inputs["canonical_validation"])
    adjudicator = [str(generated["adjudicator"]["manifest_path"])]
    return {
        channel: {
            "training_manifest_paths": [str(generated[channel]["manifest_path"])],
            "anchor_manifest_paths": canonical_train,
            "validation_manifest_paths": adjudicator,
            "retention_validation_manifest_paths": canonical_validation,
        }
        for channel in ("search", "rank4")
    }


def _build_manifest(
    *,
    expected_counts: Mapping[str, int],
    inputs: Mapping[str, Sequence[Mapping[str, object]]],
    results: Mapping[str, _DeduplicationResult],
    generated: Mapping[str, Mapping[str, object]],
    leakage: Mapping[str, object],
    producer: Mapping[str, object],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "canonical_count_contract": dict(expected_counts),
        "inputs": {key: list(inputs[key]) for key in _INPUT_SPECS},
        "deduplicated": {
            channel: {
                "selection": results[channel].selection,
                "selection_sha256": results[channel].selection_sha256,
                "shard": dict(generated[channel]),
            }
            for channel in ("search", "rank4", "adjudicator")
        },
        "interfaces": _interfaces(inputs, generated),
        "protected_test": {
            "manifest_paths": _manifest_paths(inputs["canonical_test"]),
            "rows": expected_counts["test"],
            "selection_eligible": False,
            "training_eligible": False,
        },
        "leakage": dict(leakage),
        "producer": dict(producer),
    }
    manifest["body_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _verify_body_sha256(manifest: Mapping[str, object]) -> None:
    body_sha = manifest.get("body_sha256")
    if not _valid_sha256(body_sha):
        raise ValueError("rebuild corpus manifest has no valid body SHA-256")
    body = dict(manifest)
    del body["body_sha256"]
    if sha256_bytes(canonical_json_bytes(body)) != body_sha:
        raise ValueError("rebuild corpus manifest body SHA-256 mismatch")


@dataclasses.dataclass(frozen=True)
class RebuildCorpus:
    """A fully revalidated rebuild manifest and its safe path interfaces."""

    manifest_path: pathlib.Path
    manifest: dict[str, object]

    def _paths(self, channel: str, field: str) -> tuple[pathlib.Path, ...]:
        if channel not in {"search", "rank4"}:
            raise ValueError("training channel must be search or rank4")
        interfaces = self.manifest["interfaces"]
        assert isinstance(interfaces, dict)
        interface = interfaces[channel]
        assert isinstance(interface, dict)
        return tuple(
            (self.manifest_path.parent / value).resolve()
            for value in interface[field]
        )

    def training_manifest_paths(self, channel: str) -> tuple[pathlib.Path, ...]:
        return self._paths(channel, "training_manifest_paths")

    def anchor_manifest_paths(self, channel: str) -> tuple[pathlib.Path, ...]:
        return self._paths(channel, "anchor_manifest_paths")

    def validation_manifest_paths(self, channel: str) -> tuple[pathlib.Path, ...]:
        return self._paths(channel, "validation_manifest_paths")

    def retention_validation_manifest_paths(
        self, channel: str
    ) -> tuple[pathlib.Path, ...]:
        return self._paths(channel, "retention_validation_manifest_paths")

    @property
    def protected_test_manifest_paths(self) -> tuple[pathlib.Path, ...]:
        protected = self.manifest["protected_test"]
        assert isinstance(protected, dict)
        return tuple(
            (self.manifest_path.parent / value).resolve()
            for value in protected["manifest_paths"]
        )


def freeze_rebuild_corpus(
    output_directory: pathlib.Path,
    *,
    canonical_train: Sequence[pathlib.Path],
    canonical_validation: Sequence[pathlib.Path],
    canonical_test: Sequence[pathlib.Path],
    v5_search_train: Sequence[pathlib.Path],
    v6_search_train: Sequence[pathlib.Path],
    v5_rank4_train: Sequence[pathlib.Path],
    v6_rank4_train: Sequence[pathlib.Path],
    v5_adjudicator_validation: Sequence[pathlib.Path],
    v6_adjudicator_validation: Sequence[pathlib.Path],
    expected_canonical_counts: Mapping[str, int] | None = None,
) -> tuple[pathlib.Path, dict[str, object]]:
    """Freeze a content-addressed corpus manifest and deduplicated CSR shards."""

    base = pathlib.Path(output_directory).resolve()
    base.mkdir(parents=True, exist_ok=True)
    producer = _producer_identity()
    supplied = {
        "canonical_train": canonical_train,
        "canonical_validation": canonical_validation,
        "canonical_test": canonical_test,
        "v5_search_train": v5_search_train,
        "v6_search_train": v6_search_train,
        "v5_rank4_train": v5_rank4_train,
        "v6_rank4_train": v6_rank4_train,
        "v5_adjudicator_validation": v5_adjudicator_validation,
        "v6_adjudicator_validation": v6_adjudicator_validation,
    }
    expected = _normalized_expected_counts(expected_canonical_counts)
    inputs = _bind_inputs(base, supplied)
    _validate_canonical_counts(inputs, expected)
    results = {
        channel: _deduplicate_channel(base, channel, inputs)
        for channel in ("search", "rank4", "adjudicator")
    }
    leakage = _leakage_report(base, inputs, results)
    if _producer_identity() != producer:
        raise ValueError("rebuild corpus producer changed while evidence was frozen")
    generated = {
        channel: _write_deduplicated_shard(base, results[channel], producer)
        for channel in ("search", "rank4", "adjudicator")
    }
    manifest = _build_manifest(
        expected_counts=expected,
        inputs=inputs,
        results=results,
        generated=generated,
        leakage=leakage,
        producer=producer,
    )
    payload = canonical_json_bytes(manifest)
    manifest_path = base / f"{sha256_bytes(payload)}.json"
    training._write_once(manifest_path, payload)
    return manifest_path, manifest


def validate_rebuild_manifest(
    manifest_path: pathlib.Path,
    *,
    expected_canonical_counts: Mapping[str, int] | None = None,
) -> RebuildCorpus:
    """Reload every bound artifact and fail closed on drift or leakage."""

    manifest_path = pathlib.Path(manifest_path).resolve()
    manifest, raw = _read_canonical_json(manifest_path, "rebuild corpus manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("feature_schema") != features.FEATURE_SCHEMA
        or manifest_path.suffix != ".json"
        or manifest_path.stem != sha256_bytes(raw)
    ):
        raise ValueError("rebuild corpus manifest identity is invalid")
    _verify_body_sha256(manifest)
    producer = _producer_identity()
    if manifest.get("producer") != producer:
        raise ValueError("rebuild corpus producer provenance changed")
    expected = _normalized_expected_counts(expected_canonical_counts)
    if manifest.get("canonical_count_contract") != expected:
        raise ValueError("rebuild corpus canonical count contract changed")
    base = manifest_path.parent
    inputs = _reload_inputs(base, manifest.get("inputs"))
    _validate_canonical_counts(inputs, expected)
    results = {
        channel: _deduplicate_channel(base, channel, inputs)
        for channel in ("search", "rank4", "adjudicator")
    }
    recorded_dedup = manifest.get("deduplicated")
    if not isinstance(recorded_dedup, dict) or set(recorded_dedup) != set(results):
        raise ValueError("rebuild corpus deduplicated shard bindings are incomplete")
    generated: dict[str, dict[str, object]] = {}
    for channel, result in results.items():
        record = recorded_dedup.get(channel)
        if (
            not isinstance(record, dict)
            or set(record) != {"selection", "selection_sha256", "shard"}
            or record.get("selection") != result.selection
            or record.get("selection_sha256") != result.selection_sha256
        ):
            raise ValueError(f"deduplicated {channel} row selection changed")
        shard_identity = record.get("shard")
        shard = _load_generated_shard(base, shard_identity, result, producer)
        del shard
        assert isinstance(shard_identity, Mapping)
        generated[channel] = dict(shard_identity)
    leakage = _leakage_report(base, inputs, results)
    expected_manifest = _build_manifest(
        expected_counts=expected,
        inputs=inputs,
        results=results,
        generated=generated,
        leakage=leakage,
        producer=producer,
    )
    if manifest != expected_manifest:
        raise ValueError("rebuild corpus manifest semantics changed")

    protected_paths = set(expected_manifest["protected_test"]["manifest_paths"])
    for interface in expected_manifest["interfaces"].values():
        exposed = set(interface["training_manifest_paths"])
        exposed.update(interface["anchor_manifest_paths"])
        exposed.update(interface["validation_manifest_paths"])
        exposed.update(interface["retention_validation_manifest_paths"])
        if protected_paths.intersection(exposed):
            raise ValueError("canonical test shard is exposed by a training interface")
    return RebuildCorpus(manifest_path, manifest)


# Short aliases are useful to workflow callers while keeping the artifact name
# explicit in diagnostics and command help.
freeze_corpus = freeze_rebuild_corpus
load_rebuild_manifest = validate_rebuild_manifest
validate_manifest = validate_rebuild_manifest


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    for option in (
        "canonical-train",
        "canonical-validation",
        "canonical-test",
        "v5-search-train",
        "v6-search-train",
        "v5-rank4-train",
        "v6-rank4-train",
        "v5-adjudicator-validation",
        "v6-adjudicator-validation",
    ):
        parser.add_argument(
            f"--{option}",
            action="append",
            required=True,
            type=pathlib.Path,
            metavar="SHARD_MANIFEST",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze rebuild evidence")
    freeze.add_argument("--output", required=True, type=pathlib.Path)
    _add_input_arguments(freeze)
    validate = subparsers.add_parser("validate", help="revalidate frozen evidence")
    validate.add_argument("manifest", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        path, manifest = freeze_rebuild_corpus(
            arguments.output,
            canonical_train=arguments.canonical_train,
            canonical_validation=arguments.canonical_validation,
            canonical_test=arguments.canonical_test,
            v5_search_train=arguments.v5_search_train,
            v6_search_train=arguments.v6_search_train,
            v5_rank4_train=arguments.v5_rank4_train,
            v6_rank4_train=arguments.v6_rank4_train,
            v5_adjudicator_validation=arguments.v5_adjudicator_validation,
            v6_adjudicator_validation=arguments.v6_adjudicator_validation,
        )
        report: dict[str, Any] = {
            "manifest": str(path),
            "manifest_sha256": path.stem,
            "retained_rows": {
                channel: manifest["deduplicated"][channel]["selection"][
                    "retained_rows"
                ]
                for channel in ("search", "rank4", "adjudicator")
            },
            "status": "frozen",
        }
    else:
        loaded = validate_rebuild_manifest(arguments.manifest)
        report = {
            "manifest": str(loaded.manifest_path),
            "manifest_sha256": loaded.manifest_path.stem,
            "status": "valid",
        }
    sys.stdout.buffer.write(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
