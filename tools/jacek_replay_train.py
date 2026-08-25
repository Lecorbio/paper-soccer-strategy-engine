#!/usr/bin/env python3
"""Pack sparse shards, train, and export the Jacek replay BFM value model.

The production architecture is fixed at 6301 -> 192 -> 32 -> 1 with no
biases.  The first layer consumes active indices directly, so training and
native inference never materialize dense 6301-element input vectors.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import math
import multiprocessing
import os
import pathlib
import struct
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402


HIDDEN_ONE = 192
HIDDEN_TWO = 32
OUTPUT_COUNT = 1
LEAKY_SLOPE = np.float32(0.01)
WEIGHT_COUNT = (
    features.INPUT_COUNT * HIDDEN_ONE
    + HIDDEN_ONE * HIDDEN_TWO
    + HIDDEN_TWO
)
FIXED_SEEDS = (20260823, 20260824, 20260825)

SHARD_SCHEMA = "papersoccer.jacek-replay-csr-shard.v1"
MODEL_MANIFEST_SCHEMA = "papersoccer.jacek-replay-bfm-model.v1"
SEED_CHECKPOINT_SCHEMA = "papersoccer.jacek-replay-bfm-seed-checkpoint.v1"
RUNTIME_MAGIC = b"JRBFM\0\0\x01"
RUNTIME_VERSION = 1
RUNTIME_HEADER = struct.Struct("<8s9IfQ32s32s8s")
FEATURE_SCHEMA_HASH = hashlib.sha256(features.FEATURE_SCHEMA.encode("utf-8")).digest()
ACTIVATION_IDS = (1, 2, 3)


def canonical_json_bytes(value: object) -> bytes:
    return corpus.canonical_json_bytes(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, value, allow_pickle=False)
    return output.getvalue()


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return a byte-stable, uncompressed NPZ with fixed ZIP metadata."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _array_bytes(arrays[name]))
    return output.getvalue()


def _write_once(path: pathlib.Path, payload: bytes) -> None:
    """Atomically create a content-addressed artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-addressed path conflicts: {path}")
        return
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"content-addressed path raced: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class SparseShard:
    indptr: np.ndarray
    indices: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    group_ids: np.ndarray
    split: str
    npz_sha256: str = ""

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def active(self, row: int) -> np.ndarray:
        return self.indices[self.indptr[row] : self.indptr[row + 1]]


def _samples_to_arrays(samples: Sequence[corpus.LabeledSample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot write an empty sparse shard")
    normalized = [features.validate_active(sample.active) for sample in samples]
    indptr = np.zeros(len(samples) + 1, dtype="<i8")
    for index, active in enumerate(normalized):
        indptr[index + 1] = indptr[index] + len(active)
    indices = np.fromiter(
        (feature for active in normalized for feature in active),
        dtype="<u2",
        count=int(indptr[-1]),
    )
    targets = np.asarray([sample.target for sample in samples], dtype="<f4")
    weights = np.asarray([sample.weight for sample in samples], dtype="<f4")
    group_ids = np.asarray(
        [hashlib.sha256(sample.group_id.encode()).digest() for sample in samples],
        dtype="V32",
    )
    if not np.all(np.isfinite(targets)) or np.any(np.abs(targets) > 1.0):
        raise ValueError("shard targets must be finite and in [-1, 1]")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("shard weights must be finite and positive")
    return {
        "group_ids": group_ids,
        "indices": indices,
        "indptr": indptr,
        "targets": targets,
        "weights": weights,
    }


def write_csr_shard(
    directory: pathlib.Path,
    split: str,
    samples: Sequence[corpus.LabeledSample],
    *,
    provenance: Mapping[str, object] | None = None,
) -> tuple[pathlib.Path, pathlib.Path, dict]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    arrays = _samples_to_arrays(samples)
    payload = deterministic_npz(arrays)
    digest = _sha256(payload)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{digest}.npz"
    _write_once(npz_path, payload)
    manifest = {
        "schema": SHARD_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "split": split,
        "npz": npz_path.name,
        "npz_sha256": digest,
        "samples": len(samples),
        "active_features": int(arrays["indices"].shape[0]),
        "array_contract": {
            "indptr": "little-endian-int64[n+1]",
            "indices": "little-endian-uint16[nnz]",
            "targets": "little-endian-float32[n]",
            "weights": "little-endian-float32[n]",
            "group_ids": "raw-sha256-32bytes[n]",
        },
        "provenance": dict(provenance or {}),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = directory / f"{_sha256(manifest_bytes)}.json"
    _write_once(manifest_path, manifest_bytes)
    return npz_path, manifest_path, manifest


def load_csr_shard(manifest_path: pathlib.Path) -> SparseShard:
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid shard manifest: {manifest_path}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("feature_schema") != features.FEATURE_SCHEMA
        or manifest.get("split") not in {"train", "validation", "test"}
    ):
        raise ValueError("unexpected sparse shard manifest")
    if manifest_path.suffix != ".json" or _sha256(manifest_bytes) != manifest_path.stem:
        raise ValueError("sparse shard manifest is not content addressed")
    npz_name, expected_hash = manifest.get("npz"), manifest.get("npz_sha256")
    if (
        not isinstance(npz_name, str)
        or pathlib.PurePath(npz_name).name != npz_name
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise ValueError("invalid sparse shard identity")
    npz_path = manifest_path.parent / npz_name
    if _sha256_file(npz_path) != expected_hash or npz_path.stem != expected_hash:
        raise ValueError("sparse shard SHA-256 mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_arrays = {"indptr", "indices", "targets", "weights", "group_ids"}
            if set(archive.files) != expected_arrays:
                raise ValueError("sparse shard arrays are not frozen")
            arrays = {name: archive[name].copy() for name in expected_arrays}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("invalid sparse shard NPZ") from error
    indptr = arrays["indptr"]
    indices = arrays["indices"]
    targets = arrays["targets"]
    weights = arrays["weights"]
    group_ids = arrays["group_ids"]
    count = int(targets.shape[0]) if targets.ndim == 1 else -1
    if (
        indptr.dtype != np.dtype("<i8")
        or indices.dtype != np.dtype("<u2")
        or targets.dtype != np.dtype("<f4")
        or weights.dtype != np.dtype("<f4")
        or group_ids.dtype != np.dtype("V32")
        or indptr.ndim != 1
        or indices.ndim != 1
        or weights.ndim != 1
        or group_ids.ndim != 1
        or count < 0
        or indptr.shape != (count + 1,)
        or weights.shape != (count,)
        or group_ids.shape != (count,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != indices.shape[0]
        or np.any(indptr[1:] < indptr[:-1])
        or np.any(indices >= features.INPUT_COUNT)
        or not np.all(np.isfinite(targets))
        or np.any(np.abs(targets) > 1.0)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("sparse shard array contract violation")
    shard = SparseShard(
        indptr, indices, targets, weights, group_ids, manifest["split"], expected_hash
    )
    if manifest.get("samples") != len(shard) or manifest.get("active_features") != len(indices):
        raise ValueError("sparse shard manifest counts disagree")
    for row in range(len(shard)):
        features.validate_active(shard.active(row).tolist())
    return shard


@dataclasses.dataclass(frozen=True)
class Dataset:
    indptr: np.ndarray
    indices: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    group_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def active_row(self, row: int) -> np.ndarray:
        return self.indices[self.indptr[row] : self.indptr[row + 1]]

    def active_rows(self, rows: Sequence[int] | np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self.active_row(int(row)) for row in rows)

    @classmethod
    def from_active(
        cls,
        active: Sequence[Sequence[int] | np.ndarray],
        targets: np.ndarray,
        weights: np.ndarray,
        group_ids: Sequence[str] | np.ndarray,
    ) -> "Dataset":
        indptr = np.zeros(len(active) + 1, dtype=np.int64)
        for row, values in enumerate(active):
            indptr[row + 1] = indptr[row] + len(values)
        indices = np.fromiter(
            (int(index) for values in active for index in values),
            dtype=np.uint16,
            count=int(indptr[-1]),
        )
        return cls(
            indptr,
            indices,
            np.asarray(targets, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
            np.fromiter(
                (
                    int.from_bytes(
                        hashlib.sha256(str(group).encode()).digest()[:8], "little"
                    )
                    for group in group_ids
                ),
                dtype=np.uint64,
                count=len(group_ids),
            ),
        )


def concatenate_datasets(datasets: Sequence[Dataset]) -> Dataset:
    """Concatenate datasets in caller order without changing row identity."""

    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("dataset concatenation requires nonempty inputs")
    total_rows = sum(len(dataset) for dataset in datasets)
    indptr = np.empty(total_rows + 1, dtype=np.int64)
    indptr[0] = 0
    row_offset = 0
    active_offset = 0
    for dataset in datasets:
        rows = len(dataset)
        indptr[row_offset + 1 : row_offset + rows + 1] = (
            dataset.indptr[1:] + active_offset
        )
        row_offset += rows
        active_offset += len(dataset.indices)
    return Dataset(
        indptr,
        np.concatenate([dataset.indices for dataset in datasets]).astype(
            np.uint16, copy=False
        ),
        np.concatenate([dataset.targets for dataset in datasets]).astype(
            np.float32, copy=False
        ),
        np.concatenate([dataset.weights for dataset in datasets]).astype(
            np.float32, copy=False
        ),
        np.concatenate([dataset.group_ids for dataset in datasets]).astype(
            np.uint64, copy=False
        ),
    )


@dataclasses.dataclass(frozen=True)
class MixedTraining:
    new: Dataset
    anchor: Dataset
    new_rows_per_batch: int
    anchor_rows_per_batch: int

    def validate(self, batch_size: int) -> None:
        if len(self.new) == 0 or len(self.anchor) == 0:
            raise ValueError("mixed training streams must be nonempty")
        for value, label in (
            (self.new_rows_per_batch, "new_rows_per_batch"),
            (self.anchor_rows_per_batch, "anchor_rows_per_batch"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.new_rows_per_batch + self.anchor_rows_per_batch != batch_size:
            raise ValueError("mixed training row quotas must sum to batch_size")


def _cycled_rows(
    count: int, total: int, *, seed: int, epoch: int, stream: str
) -> np.ndarray:
    output = np.empty(total, dtype=np.int64)
    offset = 0
    cycle = 0
    while offset < total:
        material = f"{seed}:{epoch}:{stream}:{cycle}".encode("utf-8")
        cycle_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        order = np.random.default_rng(cycle_seed).permutation(count)
        take = min(count, total - offset)
        output[offset : offset + take] = order[:take]
        offset += take
        cycle += 1
    return output


def mixed_epoch_batches(
    training: MixedTraining, *, batch_size: int, seed: int, epoch: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return exact, deterministic two-stream batches for one epoch.

    An epoch visits every new row once before deterministic padding.  The
    canonical anchor is sampled from independently seeded permutations and
    cycles only when the requested anchor quota exhausts it.  This prevents a
    much larger anchor corpus from multiplying the amount of new-data work in
    every epoch while preserving the exact frozen batch composition.
    """

    training.validate(batch_size)
    if epoch <= 0:
        raise ValueError("mixed training epoch must be positive")
    batch_count = math.ceil(
        len(training.new) / training.new_rows_per_batch
    )
    new_rows = _cycled_rows(
        len(training.new),
        batch_count * training.new_rows_per_batch,
        seed=seed,
        epoch=epoch,
        stream="new",
    )
    anchor_rows = _cycled_rows(
        len(training.anchor),
        batch_count * training.anchor_rows_per_batch,
        seed=seed,
        epoch=epoch,
        stream="anchor",
    )
    return tuple(
        (
            new_rows[
                index * training.new_rows_per_batch :
                (index + 1) * training.new_rows_per_batch
            ],
            anchor_rows[
                index * training.anchor_rows_per_batch :
                (index + 1) * training.anchor_rows_per_batch
            ],
        )
        for index in range(batch_count)
    )


def _dataset_identity(dataset: Dataset) -> dict:
    digest = hashlib.sha256()
    arrays = {
        "group_ids": dataset.group_ids,
        "indices": dataset.indices,
        "indptr": dataset.indptr,
        "targets": dataset.targets,
        "weights": dataset.weights,
    }
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
        metadata = {
            "name": name,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
        digest.update(canonical_json_bytes(metadata))
        digest.update(memoryview(contiguous).cast("B"))
    return {
        "samples": len(dataset),
        "active_features": int(dataset.indices.shape[0]),
        "sha256": digest.hexdigest(),
    }


def _dataset_identities(datasets: Mapping[str, Dataset]) -> dict[str, dict]:
    return {
        split: _dataset_identity(dataset)
        for split, dataset in sorted(datasets.items())
    }


def _shard_identities(manifest_paths: Sequence[pathlib.Path]) -> list[dict]:
    identities = []
    for ordinal, path in enumerate(manifest_paths):
        payload = path.read_bytes()
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid shard manifest identity: {path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema") != SHARD_SCHEMA:
            raise ValueError(f"unexpected shard manifest identity: {path}")
        identities.append(
            {
                "ordinal": ordinal,
                "manifest_sha256": _sha256(payload),
                "npz_sha256": manifest.get("npz_sha256"),
                "split": manifest.get("split"),
                "samples": manifest.get("samples"),
                "active_features": manifest.get("active_features"),
            }
        )
    return identities


def combine_shards(shards: Sequence[SparseShard]) -> Dataset:
    if not shards:
        raise ValueError("dataset needs at least one shard")
    split = shards[0].split
    if any(shard.split != split for shard in shards):
        raise ValueError("cannot combine shards from different splits")
    if len(shards) == 1:
        shard = shards[0]
        group_ids = np.fromiter(
            (
                int.from_bytes(bytes(group)[:8], "little")
                for group in shard.group_ids
            ),
            dtype=np.uint64,
            count=len(shard),
        )
        return Dataset(
            shard.indptr,
            shard.indices,
            shard.targets,
            shard.weights,
            group_ids,
        )
    total_rows = sum(len(shard) for shard in shards)
    indptr = np.empty(total_rows + 1, dtype=np.int64)
    indptr[0] = 0
    row_offset = 0
    active_offset = 0
    for shard in shards:
        rows = len(shard)
        indptr[row_offset + 1 : row_offset + rows + 1] = (
            shard.indptr[1:] + active_offset
        )
        row_offset += rows
        active_offset += len(shard.indices)
    indices = np.concatenate([shard.indices for shard in shards])
    targets = np.concatenate([shard.targets for shard in shards]).astype(np.float32)
    weights = np.concatenate([shard.weights for shard in shards]).astype(np.float32)
    group_ids = np.fromiter(
        (
            int.from_bytes(bytes(group)[:8], "little")
            for shard in shards
            for group in shard.group_ids
        ),
        dtype=np.uint64,
        count=total_rows,
    )
    return Dataset(indptr, indices, targets, weights, group_ids)


def validate_shard_collection(shards: Sequence[SparseShard]) -> None:
    """Fail closed if separately packed shards reintroduce split leakage."""

    group_splits: dict[str, str] = {}
    fingerprints: dict[bytes, str] = {}
    for shard in shards:
        for row in range(len(shard)):
            group = bytes(shard.group_ids[row]).hex()
            previous_split = group_splits.setdefault(group, shard.split)
            if previous_split != shard.split:
                raise ValueError(f"root group crosses splits: {group}")
            fingerprint = corpus.canonical_feature_fingerprint(
                shard.active(row).tolist()
            )
            previous_fingerprint_split = fingerprints.get(fingerprint)
            if previous_fingerprint_split is not None:
                if previous_fingerprint_split != shard.split:
                    raise ValueError(
                        "exact/rotated/reflected feature overlap crosses splits"
                    )
            else:
                fingerprints[fingerprint] = shard.split


def initialize(seed: int) -> dict[str, np.ndarray]:
    if seed < 0 or seed >= 1 << 64:
        raise ValueError("seed must fit uint64")
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(
            0.0, math.sqrt(1.0 / features.INPUT_COUNT),
            (features.INPUT_COUNT, HIDDEN_ONE),
        ).astype(np.float32),
        "w2": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_ONE), (HIDDEN_ONE, HIDDEN_TWO)
        ).astype(np.float32),
        "w3": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_TWO), (HIDDEN_TWO,)
        ).astype(np.float32),
    }


def _hidden_one(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value * value, LEAKY_SLOPE * value).astype(np.float32)


def _hidden_one_derivative(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 2.0 * value, LEAKY_SLOPE).astype(np.float32)


def _leaky(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value, LEAKY_SLOPE * value).astype(np.float32)


def _leaky_derivative(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 1.0, LEAKY_SLOPE).astype(np.float32)


def forward(
    parameters: Mapping[str, np.ndarray], active: Sequence[np.ndarray]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    first_pre = np.empty((len(active), HIDDEN_ONE), dtype=np.float32)
    for row, indices in enumerate(active):
        first_pre[row] = parameters["w1"][indices].sum(axis=0)
    first = _hidden_one(first_pre)
    second_pre = first @ parameters["w2"]
    second = _leaky(second_pre)
    output_pre = second @ parameters["w3"]
    output = np.tanh(output_pre).astype(np.float32)
    return output, (first_pre, first, second_pre, second, output_pre)


def _weighted_huber(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    difference = predictions - targets
    absolute = np.abs(difference)
    losses = np.where(
        absolute <= delta,
        0.5 * difference * difference,
        delta * (absolute - 0.5 * delta),
    )
    denominator = max(float(np.sum(weights)), np.finfo(np.float32).tiny)
    loss = float(np.sum(weights * losses) / denominator)
    gradient = weights * np.clip(difference, -delta, delta) / denominator
    return loss, gradient.astype(np.float32)


class AdamW:
    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.learning_rate = np.float32(learning_rate)
        self.weight_decay = np.float32(weight_decay)
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step = 0

    def update(
        self,
        parameters: Mapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        self.step += 1
        correction_one = 1.0 - 0.9**self.step
        correction_two = 1.0 - 0.999**self.step
        for name, parameter in parameters.items():
            gradient = gradients[name]
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradient * gradient
            first = self.first[name] / correction_one
            second = self.second[name] / correction_two
            parameter *= 1.0 - self.learning_rate * self.weight_decay
            parameter -= self.learning_rate * first / (np.sqrt(second) + 1e-8)


def _apply_training_batch(
    parameters: Mapping[str, np.ndarray],
    optimizer: AdamW,
    active: Sequence[np.ndarray],
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    huber_delta: float,
) -> float:
    prediction, cache = forward(parameters, active)
    first_pre, first, second_pre, second, output_pre = cache
    loss, output_gradient = _weighted_huber(
        prediction,
        targets,
        weights,
        huber_delta,
    )
    output_pre_gradient = output_gradient * (1.0 - np.tanh(output_pre) ** 2)
    gradients: dict[str, np.ndarray] = {
        "w3": second.T @ output_pre_gradient,
    }
    second_gradient = output_pre_gradient[:, None] * parameters["w3"][None, :]
    second_pre_gradient = second_gradient * _leaky_derivative(second_pre)
    gradients["w2"] = first.T @ second_pre_gradient
    first_gradient = second_pre_gradient @ parameters["w2"].T
    first_pre_gradient = first_gradient * _hidden_one_derivative(first_pre)
    gradients["w1"] = np.zeros_like(parameters["w1"])
    for row, indices in enumerate(active):
        np.add.at(gradients["w1"], indices, first_pre_gradient[row])
    squared_norm = sum(float(np.sum(value * value)) for value in gradients.values())
    norm = math.sqrt(squared_norm)
    if norm > 5.0:
        scale = np.float32(5.0 / norm)
        for gradient in gradients.values():
            gradient *= scale
    optimizer.update(parameters, gradients)
    return loss


def train_batch(
    parameters: Mapping[str, np.ndarray],
    optimizer: AdamW,
    dataset: Dataset,
    rows: np.ndarray,
    *,
    huber_delta: float,
) -> float:
    return _apply_training_batch(
        parameters,
        optimizer,
        dataset.active_rows(rows),
        dataset.targets[rows],
        dataset.weights[rows],
        huber_delta=huber_delta,
    )


def train_mixed_batch(
    parameters: Mapping[str, np.ndarray],
    optimizer: AdamW,
    training: MixedTraining,
    new_rows: np.ndarray,
    anchor_rows: np.ndarray,
    *,
    huber_delta: float,
) -> float:
    if (
        len(new_rows) != training.new_rows_per_batch
        or len(anchor_rows) != training.anchor_rows_per_batch
    ):
        raise ValueError("mixed training batch does not match its exact row quotas")
    active = (
        *training.new.active_rows(new_rows),
        *training.anchor.active_rows(anchor_rows),
    )
    targets = np.concatenate(
        (training.new.targets[new_rows], training.anchor.targets[anchor_rows])
    )
    weights = np.concatenate(
        (training.new.weights[new_rows], training.anchor.weights[anchor_rows])
    )
    return _apply_training_batch(
        parameters,
        optimizer,
        active,
        targets,
        weights,
        huber_delta=huber_delta,
    )


def metrics(
    parameters: Mapping[str, np.ndarray], dataset: Dataset, batch_size: int = 4_096
) -> dict:
    if batch_size <= 0:
        raise ValueError("metrics batch size must be positive")
    predictions = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        predictions[start:stop], _ = forward(
            parameters, dataset.active_rows(range(start, stop))
        )
    loss, _ = _weighted_huber(predictions, dataset.targets, dataset.weights, 0.25)
    sign_accuracy = float(np.mean((predictions >= 0.0) == (dataset.targets >= 0.0)))
    if len(dataset) > 1 and np.std(predictions) > 0.0 and np.std(dataset.targets) > 0.0:
        correlation = float(np.corrcoef(predictions, dataset.targets)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    return {
        "samples": len(dataset),
        "weighted_huber": loss,
        "sign_accuracy": sign_accuracy,
        "correlation": correlation,
        "mae": float(np.mean(np.abs(predictions - dataset.targets))),
        "prediction_mean": float(np.mean(predictions)),
    }


def train_seed(
    datasets: Mapping[str, Dataset],
    seed: int,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mixed_training: MixedTraining | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    parameters = initialize(seed)
    optimizer = AdamW(parameters, learning_rate, weight_decay)
    rng = np.random.default_rng(seed ^ 0xD1B54A32D192ED03)
    if mixed_training is not None:
        mixed_training.validate(batch_size)
    best: dict[str, np.ndarray] | None = None
    best_epoch = 0
    best_key = (float("inf"), 0.0, 0.0)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        if mixed_training is None:
            order = rng.permutation(len(datasets["train"]))
            for start in range(0, len(order), batch_size):
                losses.append(
                    train_batch(
                        parameters,
                        optimizer,
                        datasets["train"],
                        order[start : start + batch_size],
                        huber_delta=0.25,
                    )
                )
        else:
            for new_rows, anchor_rows in mixed_epoch_batches(
                mixed_training, batch_size=batch_size, seed=seed, epoch=epoch
            ):
                losses.append(
                    train_mixed_batch(
                        parameters,
                        optimizer,
                        mixed_training,
                        new_rows,
                        anchor_rows,
                        huber_delta=0.25,
                    )
                )
        validation = metrics(parameters, datasets["validation"])
        key = (
            validation["weighted_huber"],
            -validation["sign_accuracy"],
            -validation["correlation"],
        )
        history.append(
            {
                "epoch": epoch,
                "training_weighted_huber": float(np.mean(losses)),
                "validation": validation,
            }
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best = {name: value.copy() for name, value in parameters.items()}
        elif epoch - best_epoch >= patience:
            break
    if best is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    return best, {
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "training": metrics(best, datasets["train"]),
        "validation": metrics(best, datasets["validation"]),
    }


def _validate_parameters(parameters: Mapping[str, np.ndarray]) -> None:
    expected = {
        "w1": (features.INPUT_COUNT, HIDDEN_ONE),
        "w2": (HIDDEN_ONE, HIDDEN_TWO),
        "w3": (HIDDEN_TWO,),
    }
    if set(parameters) != set(expected):
        raise ValueError("runtime parameters must contain exactly w1, w2, and w3")
    for name, shape in expected.items():
        value = parameters[name]
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid or nonfinite runtime tensor {name}")


def runtime_bytes(parameters: Mapping[str, np.ndarray]) -> tuple[bytes, dict]:
    _validate_parameters(parameters)
    payload = b"".join(
        np.asarray(parameters[name], dtype="<f4", order="C").tobytes(order="C")
        for name in ("w1", "w2", "w3")
    )
    if len(payload) != WEIGHT_COUNT * 4:
        raise RuntimeError("runtime payload weight count is inconsistent")
    payload_hash = hashlib.sha256(payload).digest()
    header = RUNTIME_HEADER.pack(
        RUNTIME_MAGIC,
        RUNTIME_HEADER.size,
        RUNTIME_VERSION,
        features.INPUT_COUNT,
        HIDDEN_ONE,
        HIDDEN_TWO,
        OUTPUT_COUNT,
        *ACTIVATION_IDS,
        float(LEAKY_SLOPE),
        WEIGHT_COUNT,
        FEATURE_SCHEMA_HASH,
        payload_hash,
        b"\0" * 8,
    )
    artifact = header + payload
    return artifact, {
        "artifact_sha256": _sha256(artifact),
        "payload_sha256": payload_hash.hex(),
        "feature_schema_sha256": FEATURE_SCHEMA_HASH.hex(),
        "bytes": len(artifact),
        "weight_count": WEIGHT_COUNT,
    }


def export_runtime(path: pathlib.Path, parameters: Mapping[str, np.ndarray]) -> dict:
    artifact, report = runtime_bytes(parameters)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            output.write(artifact)
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def load_runtime(path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict]:
    artifact = path.read_bytes()
    if len(artifact) < RUNTIME_HEADER.size:
        raise ValueError("runtime is shorter than its fixed header")
    fields = RUNTIME_HEADER.unpack(artifact[: RUNTIME_HEADER.size])
    (
        magic,
        header_size,
        version,
        inputs,
        hidden_one,
        hidden_two,
        outputs,
        activation_one,
        activation_two,
        activation_out,
        slope,
        weight_count,
        schema_hash,
        payload_hash,
        reserved,
    ) = fields
    expected = (
        magic == RUNTIME_MAGIC
        and header_size == RUNTIME_HEADER.size
        and version == RUNTIME_VERSION
        and (inputs, hidden_one, hidden_two, outputs)
        == (features.INPUT_COUNT, HIDDEN_ONE, HIDDEN_TWO, OUTPUT_COUNT)
        and (activation_one, activation_two, activation_out) == ACTIVATION_IDS
        and struct.pack("<f", slope) == struct.pack("<f", float(LEAKY_SLOPE))
        and weight_count == WEIGHT_COUNT
        and schema_hash == FEATURE_SCHEMA_HASH
        and reserved == b"\0" * 8
    )
    if not expected:
        raise ValueError("runtime header contract mismatch")
    payload = artifact[RUNTIME_HEADER.size :]
    if len(payload) != WEIGHT_COUNT * 4:
        raise ValueError("runtime is truncated or has trailing bytes")
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("runtime payload SHA-256 mismatch")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.all(np.isfinite(values)):
        raise ValueError("runtime contains NaN or infinity")
    first_end = features.INPUT_COUNT * HIDDEN_ONE
    second_end = first_end + HIDDEN_ONE * HIDDEN_TWO
    parameters = {
        "w1": values[:first_end].reshape(features.INPUT_COUNT, HIDDEN_ONE).copy(),
        "w2": values[first_end:second_end].reshape(HIDDEN_ONE, HIDDEN_TWO).copy(),
        "w3": values[second_end:].copy(),
    }
    return parameters, {
        "artifact_sha256": _sha256(artifact),
        "payload_sha256": payload_hash.hex(),
        "feature_schema_sha256": schema_hash.hex(),
        "bytes": len(artifact),
        "weight_count": weight_count,
    }


def _training_producer_identity() -> dict[str, str]:
    return {
        "trainer_sha256": _sha256_file(pathlib.Path(__file__)),
        "features_sha256": _sha256_file(pathlib.Path(features.__file__)),
        "corpus_sha256": _sha256_file(pathlib.Path(corpus.__file__)),
    }


def _seed_training_configuration(
    training_arguments: Mapping[str, int | float | str],
) -> dict:
    new_rows = int(training_arguments.get("new_rows_per_batch", 0))
    anchor_rows = int(training_arguments.get("anchor_rows_per_batch", 0))
    if (new_rows == 0) != (anchor_rows == 0):
        raise ValueError("mixed batch configuration is incomplete")
    return {
        "architecture": {
            "dimensions": [
                features.INPUT_COUNT,
                HIDDEN_ONE,
                HIDDEN_TWO,
                OUTPUT_COUNT,
            ],
            "biases": False,
            "activations": ["square-leaky-0.01", "leaky-relu-0.01", "tanh"],
        },
        "optimizer": {
            "name": "adamw",
            "epochs": int(training_arguments["epochs"]),
            "patience": int(training_arguments["patience"]),
            "batch_size": int(training_arguments["batch_size"]),
            "learning_rate": float(training_arguments["learning_rate"]),
            "weight_decay": float(training_arguments["weight_decay"]),
            "gradient_norm_clip": 5.0,
        },
        "loss": {"name": "weighted-huber", "delta": 0.25},
        "metrics_batch_size": 4_096,
        "augmentation": "reflection rows inherit root game split",
        "batching": (
            {
                "kind": "deterministic-two-stream-cycling-v1",
                "new_rows_per_batch": new_rows,
                "anchor_rows_per_batch": anchor_rows,
                "epoch_length": "new-stream-covered-once-anchor-sampled",
                "row_order": "new-then-anchor",
            }
            if new_rows
            else {"kind": "single-stream-permutation-v1"}
        ),
        "selection_validation": training_arguments.get(
            "selection_validation", "legacy-validation-split"
        ),
    }


def _seed_checkpoint_paths(
    directory: pathlib.Path, seed: int
) -> tuple[pathlib.Path, pathlib.Path]:
    stem = f"seed-{seed}"
    return directory / f"{stem}.runtime", directory / f"{stem}.json"


def _seed_checkpoint_publication(
    seed: int,
    checkpoint_path: pathlib.Path,
    receipt_path: pathlib.Path,
    runtime_report: Mapping[str, object],
    receipt_payload: bytes,
) -> dict:
    return {
        "seed": seed,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": runtime_report["artifact_sha256"],
        "receipt": receipt_path.name,
        "receipt_sha256": _sha256(receipt_payload),
    }


def _metric_report_is_valid(value: object, expected_samples: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "samples",
        "weighted_huber",
        "sign_accuracy",
        "correlation",
        "mae",
        "prediction_mean",
    }:
        return False
    if value.get("samples") != expected_samples:
        return False
    numbers = [
        value.get("weighted_huber"),
        value.get("sign_accuracy"),
        value.get("correlation"),
        value.get("mae"),
        value.get("prediction_mean"),
    ]
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        for number in numbers
    ):
        return False
    return (
        value["weighted_huber"] >= 0.0
        and 0.0 <= value["sign_accuracy"] <= 1.0
        and -1.000_000_1 <= value["correlation"] <= 1.000_000_1
        and value["mae"] >= 0.0
    )


def _validate_seed_report(
    report: object,
    seed: int,
    configuration: Mapping[str, object],
    inputs: Mapping[str, object],
    runtime_report: Mapping[str, object],
) -> dict:
    if not isinstance(report, dict) or set(report) != {
        "seed",
        "best_epoch",
        "history",
        "training",
        "validation",
        "checkpoint",
    }:
        raise ValueError(f"seed {seed} checkpoint training report is invalid")
    epochs = configuration["optimizer"]["epochs"]
    best_epoch = report.get("best_epoch")
    history = report.get("history")
    if (
        report.get("seed") != seed
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch <= 0
        or best_epoch > epochs
        or not isinstance(history, list)
        or not history
        or len(history) > epochs
        or report.get("checkpoint") != runtime_report
    ):
        raise ValueError(f"seed {seed} checkpoint training report is stale or corrupt")
    dataset_inputs = inputs["datasets"]
    if not _metric_report_is_valid(
        report.get("training"), dataset_inputs["train"]["samples"]
    ) or not _metric_report_is_valid(
        report.get("validation"), dataset_inputs["validation"]["samples"]
    ):
        raise ValueError(f"seed {seed} checkpoint metrics are stale or corrupt")
    for expected_epoch, row in enumerate(history, 1):
        if (
            not isinstance(row, dict)
            or set(row) != {"epoch", "training_weighted_huber", "validation"}
            or row.get("epoch") != expected_epoch
            or isinstance(row.get("training_weighted_huber"), bool)
            or not isinstance(row.get("training_weighted_huber"), (int, float))
            or not math.isfinite(row["training_weighted_huber"])
            or row["training_weighted_huber"] < 0.0
            or not _metric_report_is_valid(
                row.get("validation"), dataset_inputs["validation"]["samples"]
            )
        ):
            raise ValueError(f"seed {seed} checkpoint history is stale or corrupt")
    if best_epoch > len(history):
        raise ValueError(f"seed {seed} checkpoint best epoch is not in its history")
    return report


def _publish_seed_checkpoint(
    directory: pathlib.Path,
    seed: int,
    candidate: Mapping[str, np.ndarray],
    report: dict,
    receipt_binding: Mapping[str, object],
) -> dict:
    if receipt_binding.get("producer") != _training_producer_identity():
        raise RuntimeError("seed checkpoint producer changed during training")
    checkpoint_path, receipt_path = _seed_checkpoint_paths(directory, seed)
    checkpoint_payload, runtime_report = runtime_bytes(candidate)
    if report.get("checkpoint") != runtime_report:
        raise RuntimeError("seed report checkpoint identity changed before publication")
    body = {
        "schema": SEED_CHECKPOINT_SCHEMA,
        "seed": seed,
        "configuration": receipt_binding["configuration"],
        "inputs": receipt_binding["inputs"],
        "producer": receipt_binding["producer"],
        "checkpoint": {"file": checkpoint_path.name, **runtime_report},
        "training_report": report,
    }
    receipt = {
        **body,
        "body_sha256": _sha256(canonical_json_bytes(body)),
    }
    receipt_payload = canonical_json_bytes(receipt)
    _write_once(checkpoint_path, checkpoint_payload)
    _write_once(receipt_path, receipt_payload)
    return _seed_checkpoint_publication(
        seed,
        checkpoint_path,
        receipt_path,
        runtime_report,
        receipt_payload,
    )


def _load_seed_checkpoint(
    directory: pathlib.Path,
    seed: int,
    receipt_binding: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict, dict]:
    checkpoint_path, receipt_path = _seed_checkpoint_paths(directory, seed)
    try:
        receipt_payload = receipt_path.read_bytes()
        receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"seed {seed} checkpoint receipt is invalid") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != SEED_CHECKPOINT_SCHEMA:
        raise ValueError(f"seed {seed} checkpoint receipt schema is invalid")
    try:
        canonical_receipt = canonical_json_bytes(receipt)
    except (TypeError, ValueError) as error:
        raise ValueError(f"seed {seed} checkpoint receipt is nonfinite") from error
    if receipt_payload != canonical_receipt:
        raise ValueError(f"seed {seed} checkpoint receipt is not canonical")
    body = dict(receipt)
    body_sha256 = body.pop("body_sha256", None)
    if (
        not isinstance(body_sha256, str)
        or body_sha256 != _sha256(canonical_json_bytes(body))
    ):
        raise ValueError(f"seed {seed} checkpoint receipt integrity failed")
    expected = {
        "seed": seed,
        "configuration": receipt_binding["configuration"],
        "inputs": receipt_binding["inputs"],
        "producer": receipt_binding["producer"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"seed {seed} checkpoint receipt {field} is stale")
    try:
        candidate, runtime_report = load_runtime(checkpoint_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"seed {seed} checkpoint parameters are corrupt") from error
    if receipt.get("checkpoint") != {"file": checkpoint_path.name, **runtime_report}:
        raise ValueError(f"seed {seed} checkpoint hash is stale or corrupt")
    report = _validate_seed_report(
        receipt.get("training_report"),
        seed,
        receipt_binding["configuration"],
        receipt_binding["inputs"],
        runtime_report,
    )
    publication = _seed_checkpoint_publication(
        seed,
        checkpoint_path,
        receipt_path,
        runtime_report,
        receipt_payload,
    )
    return candidate, report, publication


def _train_seed_and_report(
    datasets: Mapping[str, Dataset],
    seed: int,
    training_arguments: Mapping[str, int | float | str],
    mixed_training: MixedTraining | None = None,
    checkpoint_directory: pathlib.Path | None = None,
    receipt_binding: Mapping[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], dict, dict | None]:
    if receipt_binding is not None and (
        receipt_binding.get("producer") != _training_producer_identity()
        or receipt_binding.get("configuration")
        != _seed_training_configuration(training_arguments)
    ):
        raise RuntimeError("seed checkpoint producer or configuration changed")
    candidate, report = train_seed(
        datasets,
        seed,
        epochs=int(training_arguments["epochs"]),
        patience=int(training_arguments["patience"]),
        batch_size=int(training_arguments["batch_size"]),
        learning_rate=float(training_arguments["learning_rate"]),
        weight_decay=float(training_arguments["weight_decay"]),
        mixed_training=mixed_training,
    )
    report["checkpoint"] = runtime_bytes(candidate)[1]
    publication = None
    if checkpoint_directory is not None:
        if receipt_binding is None:
            raise RuntimeError("seed checkpoint publication has no receipt binding")
        publication = _publish_seed_checkpoint(
            checkpoint_directory, seed, candidate, report, receipt_binding
        )
    return candidate, report, publication


_SEED_WORKER_DATASETS: Mapping[str, Dataset] | None = None
_SEED_WORKER_ARGUMENTS: Mapping[str, int | float | str] | None = None
_SEED_WORKER_MIXED_TRAINING: MixedTraining | None = None
_SEED_WORKER_CHECKPOINT_DIRECTORY: pathlib.Path | None = None
_SEED_WORKER_RECEIPT_BINDING: Mapping[str, object] | None = None


def _initialize_seed_worker(
    datasets: Mapping[str, Dataset],
    training_arguments: Mapping[str, int | float | str],
    mixed_training: MixedTraining | None,
    checkpoint_directory: pathlib.Path | None,
    receipt_binding: Mapping[str, object] | None,
) -> None:
    """Install one read-only dataset copy per long-lived worker process."""

    global _SEED_WORKER_DATASETS, _SEED_WORKER_ARGUMENTS
    global _SEED_WORKER_MIXED_TRAINING
    global _SEED_WORKER_CHECKPOINT_DIRECTORY, _SEED_WORKER_RECEIPT_BINDING
    _SEED_WORKER_DATASETS = datasets
    _SEED_WORKER_ARGUMENTS = training_arguments
    _SEED_WORKER_MIXED_TRAINING = mixed_training
    _SEED_WORKER_CHECKPOINT_DIRECTORY = checkpoint_directory
    _SEED_WORKER_RECEIPT_BINDING = receipt_binding


def _train_seed_worker(
    seed: int,
) -> tuple[dict[str, np.ndarray], dict, dict | None]:
    if _SEED_WORKER_DATASETS is None or _SEED_WORKER_ARGUMENTS is None:
        raise RuntimeError("seed training worker was not initialized")
    return _train_seed_and_report(
        _SEED_WORKER_DATASETS,
        seed,
        _SEED_WORKER_ARGUMENTS,
        _SEED_WORKER_MIXED_TRAINING,
        _SEED_WORKER_CHECKPOINT_DIRECTORY,
        _SEED_WORKER_RECEIPT_BINDING,
    )


def _train_seeds_parallel(
    datasets: Mapping[str, Dataset],
    seeds: Sequence[int],
    training_arguments: Mapping[str, int | float | str],
    mixed_training: MixedTraining | None,
    seed_workers: int,
    checkpoint_directory: pathlib.Path | None,
    receipt_binding: Mapping[str, object] | None,
) -> list[tuple[dict[str, np.ndarray], dict, dict | None]]:
    """Train independent seeds while returning results in input seed order."""

    worker_count = min(seed_workers, len(seeds))
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_seed_worker,
        initargs=(
            datasets,
            training_arguments,
            mixed_training,
            checkpoint_directory,
            receipt_binding,
        ),
    )
    futures: dict[concurrent.futures.Future, tuple[int, int]] = {}
    results: list[
        tuple[dict[str, np.ndarray], dict, dict | None] | None
    ] = [None] * len(seeds)
    try:
        for index, seed in enumerate(seeds):
            try:
                future = executor.submit(_train_seed_worker, seed)
            except Exception as error:
                raise RuntimeError(
                    f"could not start training worker for seed {seed}"
                ) from error
            futures[future] = (index, seed)
        for future in concurrent.futures.as_completed(futures):
            index, seed = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError(f"training worker for seed {seed} failed") from error
    except BaseException:
        for pending in futures:
            pending.cancel()
        terminate_workers = getattr(executor, "terminate_workers", None)
        if callable(terminate_workers):
            terminate_workers()
        else:
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    if any(result is None for result in results):
        raise RuntimeError("parallel seed training returned an incomplete result set")
    return [result for result in results if result is not None]


def _checkpoint_receipt_binding(
    datasets: Mapping[str, Dataset],
    training_arguments: Mapping[str, int | float | str],
    input_shard_identities: Sequence[Mapping[str, object]],
    mixed_training: MixedTraining | None = None,
) -> dict:
    try:
        normalized_shards = json.loads(
            canonical_json_bytes([dict(identity) for identity in input_shard_identities])
        )
    except (TypeError, ValueError) as error:
        raise ValueError("input shard identities must be finite JSON objects") from error
    if not isinstance(normalized_shards, list) or any(
        not isinstance(identity, dict) for identity in normalized_shards
    ):
        raise ValueError("input shard identities must be JSON objects")
    return {
        "configuration": _seed_training_configuration(training_arguments),
        "inputs": {
            "feature_schema": features.FEATURE_SCHEMA,
            "datasets": _dataset_identities(datasets),
            "training_streams": (
                {
                    "new": _dataset_identity(mixed_training.new),
                    "anchor": _dataset_identity(mixed_training.anchor),
                }
                if mixed_training is not None
                else None
            ),
            "shards": normalized_shards,
        },
        "producer": _training_producer_identity(),
    }


def _prepare_seed_checkpoints(
    directory: pathlib.Path,
    seeds: Sequence[int],
    resume: bool,
    receipt_binding: Mapping[str, object],
) -> dict[int, tuple[dict[str, np.ndarray], dict, dict]]:
    if directory.exists() and not directory.is_dir():
        raise ValueError("seed checkpoint path exists but is not a directory")
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {
        path.name
        for seed in seeds
        for path in _seed_checkpoint_paths(directory, seed)
    }
    unexpected = sorted(
        path.name for path in directory.iterdir() if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(
            "seed checkpoint directory contains unexpected entries: "
            + ", ".join(unexpected)
        )
    completed = {}
    for seed in seeds:
        checkpoint_path, receipt_path = _seed_checkpoint_paths(directory, seed)
        checkpoint_exists = checkpoint_path.exists()
        receipt_exists = receipt_path.exists()
        if checkpoint_exists != receipt_exists:
            raise ValueError(f"seed {seed} checkpoint is incomplete")
        if not checkpoint_exists:
            continue
        if not resume:
            raise ValueError(
                f"seed {seed} checkpoint already exists; use --resume-seeds or "
                "a fresh checkpoint directory"
            )
        completed[seed] = _load_seed_checkpoint(directory, seed, receipt_binding)
    return completed


def train_three_seeds(
    datasets: Mapping[str, Dataset],
    *,
    seeds: Sequence[int] = FIXED_SEEDS,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    reveal_test: bool = False,
    seed_workers: int = 1,
    seed_checkpoint_directory: pathlib.Path | None = None,
    resume_seeds: bool = False,
    input_shard_identities: Sequence[Mapping[str, object]] = (),
    mixed_training: MixedTraining | None = None,
    selection_validation: Dataset | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("the frozen training campaign requires exactly three seeds")
    if (
        isinstance(seed_workers, bool)
        or not isinstance(seed_workers, int)
        or seed_workers <= 0
    ):
        raise ValueError("seed_workers must be a positive integer")
    datasets = dict(datasets)
    if selection_validation is not None:
        if len(selection_validation) == 0:
            raise ValueError("selection validation dataset must be nonempty")
        datasets["validation"] = selection_validation
    if mixed_training is not None:
        mixed_training.validate(batch_size)
        datasets["train"] = concatenate_datasets(
            (mixed_training.new, mixed_training.anchor)
        )
        if "validation" not in datasets or len(datasets["validation"]) == 0:
            raise ValueError(
                "mixed training requires a separate nonempty selection validation"
            )
        if reveal_test and ("test" not in datasets or len(datasets["test"]) == 0):
            raise ValueError("test reveal requires a nonempty test dataset")
    elif set(datasets) != {"train", "validation", "test"} or any(
        len(dataset) == 0 for dataset in datasets.values()
    ):
        raise ValueError("training, validation, and test datasets must be nonempty")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if any(seed < 0 or seed >= 1 << 64 for seed in normalized_seeds):
        raise ValueError("training seeds must fit uint64")
    if resume_seeds and seed_checkpoint_directory is None:
        raise ValueError("resume_seeds requires seed_checkpoint_directory")
    training_arguments: dict[str, int | float | str] = {
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "new_rows_per_batch": (
            mixed_training.new_rows_per_batch if mixed_training is not None else 0
        ),
        "anchor_rows_per_batch": (
            mixed_training.anchor_rows_per_batch if mixed_training is not None else 0
        ),
        "selection_validation": (
            "explicit-common-adjudicator"
            if selection_validation is not None
            else "dataset-validation-split"
        ),
    }
    receipt_binding = None
    completed: dict[
        int, tuple[dict[str, np.ndarray], dict, dict | None]
    ] = {}
    if seed_checkpoint_directory is not None:
        seed_checkpoint_directory = seed_checkpoint_directory.resolve()
        receipt_binding = _checkpoint_receipt_binding(
            datasets, training_arguments, input_shard_identities, mixed_training
        )
        completed = _prepare_seed_checkpoints(
            seed_checkpoint_directory,
            normalized_seeds,
            resume_seeds,
            receipt_binding,
        )
    pending_seeds = [seed for seed in normalized_seeds if seed not in completed]
    if pending_seeds:
        if seed_workers == 1:
            trained = [
                _train_seed_and_report(
                    datasets,
                    seed,
                    training_arguments,
                    mixed_training,
                    seed_checkpoint_directory,
                    receipt_binding,
                )
                for seed in pending_seeds
            ]
        else:
            trained = _train_seeds_parallel(
                datasets,
                pending_seeds,
                training_arguments,
                mixed_training,
                seed_workers,
                seed_checkpoint_directory,
                receipt_binding,
            )
        completed.update(zip(pending_seeds, trained))
    seed_results = [completed[seed] for seed in normalized_seeds]
    candidates = [result[0] for result in seed_results]
    reports = [result[1] for result in seed_results]
    chosen_index = min(
        range(3),
        key=lambda index: (
            reports[index]["validation"]["weighted_huber"],
            -reports[index]["validation"]["sign_accuracy"],
            -reports[index]["validation"]["correlation"],
            reports[index]["seed"],
        ),
    )
    if reveal_test:
        # The protected test split is intentionally evaluated only after the
        # validation-only choice is immutable, and only for a final campaign.
        reports[chosen_index]["test"] = metrics(
            candidates[chosen_index], datasets["test"]
        )
    selection_report = {
        "seeds": list(normalized_seeds),
        "seed_reports": reports,
        "chosen_seed": reports[chosen_index]["seed"],
        "selection": (
            "minimum validation weighted Huber; then maximum sign accuracy; "
            "then maximum correlation; then seed"
        ),
        "test_revealed_after_selection": reveal_test,
        "optimizer": {
            "name": "adamw",
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_norm_clip": 5.0,
        },
        "loss": {"name": "weighted-huber", "delta": 0.25},
        "augmentation": "reflection rows inherit root game split",
        "batching": _seed_training_configuration(training_arguments)["batching"],
        "selection_validation": {
            "kind": training_arguments["selection_validation"],
            "dataset": _dataset_identity(datasets["validation"]),
        },
    }
    if seed_checkpoint_directory is not None:
        publications = [result[2] for result in seed_results]
        if any(publication is None for publication in publications):
            raise RuntimeError("seed checkpoint publication set is incomplete")
        selection_report["seed_checkpoints"] = publications
    return candidates[chosen_index], selection_report


def tiny_fixture_samples() -> dict[str, list[corpus.LabeledSample]]:
    """Build a tiny deterministic schema fixture without external game data."""

    base = list(features.encode_active(features.ReplayState()))
    rows: dict[str, list[corpus.LabeledSample]] = {
        "train": [], "validation": [], "test": []
    }
    splits = ("train",) * 8 + ("validation",) * 2 + ("test",) * 2
    for index, split in enumerate(splits):
        active = base.copy()
        first_vertex = next(
            position for position, value in enumerate(active) if value >= features.EDGE_COUNT
        )
        relative = active[first_vertex] - features.EDGE_COUNT
        vertex, category = divmod(relative, features.VERTEX_CATEGORIES)
        active[first_vertex] = (
            features.EDGE_COUNT
            + vertex * features.VERTEX_CATEGORIES
            + (category + index) % features.VERTEX_CATEGORIES
        )
        active.sort()
        target = np.float32(-0.9 + index * (1.8 / 11.0)).item()
        rows[split].append(
            corpus.LabeledSample(tuple(active), target, 1.0, f"tiny:{index:02d}")
        )
    return rows


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(seed < 0 or seed >= 1 << 64 for seed in seeds):
        raise argparse.ArgumentTypeError("exactly three unique uint64 seeds are required")
    return seeds


def _parse_seed_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed workers must be an integer") from error
    if workers <= 0:
        raise argparse.ArgumentTypeError("seed workers must be positive")
    return workers


def _declared_target_policies(manifest: Mapping[str, object]) -> list[dict]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return []
    declared = provenance.get("target_policies")
    if declared is None:
        single = provenance.get("target_policy")
        declared = [] if single is None else [single]
    if not isinstance(declared, list) or any(
        not isinstance(policy, dict) for policy in declared
    ):
        raise ValueError("shard target policy provenance is malformed")
    policies = []
    for policy in declared:
        if (
            policy.get("schema") != corpus.TARGET_POLICY_SCHEMA
            or policy.get("teacher_schema")
            not in {corpus.TEACHER_SCHEMA, corpus.SEARCH_TEACHER_SCHEMA}
            or policy.get("mixture")
            != {
                "teacher_weight": 0.75,
                "outcome_weight": 0.25,
                "outcome_frame": "mover-relative-terminal-winner",
            }
            or not isinstance(policy.get("teacher_value"), dict)
        ):
            raise ValueError("shard target policy provenance is invalid")
        policies.append(json.loads(canonical_json_bytes(policy)))
    return policies


def target_metadata_from_shard_provenance(
    manifests: Sequence[Mapping[str, object]], roles: Sequence[str]
) -> dict:
    if len(manifests) != len(roles):
        raise ValueError("source shard roles do not align with manifests")
    declarations: dict[bytes, dict[str, object]] = {}
    undeclared_roles: set[str] = set()
    for manifest, role in zip(manifests, roles, strict=True):
        policies = _declared_target_policies(manifest)
        if not policies:
            undeclared_roles.add(role)
        for policy in policies:
            key = canonical_json_bytes(policy)
            record = declarations.setdefault(
                key, {"roles": set(), "policy": policy}
            )
            record["roles"].add(role)
    bound_policies = [
        {
            "roles": sorted(record["roles"]),
            "policy": record["policy"],
        }
        for _, record in sorted(declarations.items())
    ]
    return {
        "policy_provenance": "source-shard-manifest-provenance",
        "declared_policies": bound_policies,
        "undeclared_roles": sorted(undeclared_roles),
        "loss": "weighted-huber-delta-0.25",
        "policy_head": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_manifests", nargs="*", type=pathlib.Path)
    parser.add_argument("--pack-report", type=pathlib.Path)
    parser.add_argument(
        "--new-shard-manifest", action="append", type=pathlib.Path, default=[]
    )
    parser.add_argument(
        "--anchor-shard-manifest", action="append", type=pathlib.Path, default=[]
    )
    parser.add_argument(
        "--selection-validation-manifest",
        action="append",
        type=pathlib.Path,
        default=[],
    )
    parser.add_argument("--new-rows-per-batch", type=int)
    parser.add_argument("--anchor-rows-per-batch", type=int)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--tiny-fixture", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--seeds", type=_parse_seeds, default=FIXED_SEEDS)
    parser.add_argument("--seed-workers", type=_parse_seed_workers, default=1)
    parser.add_argument("--seed-checkpoint-directory", type=pathlib.Path)
    parser.add_argument(
        "--resume-seeds",
        action="store_true",
        help="validate and reuse completed per-seed checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--reveal-test",
        action="store_true",
        help="evaluate the protected test split after final validation selection",
    )
    arguments = parser.parse_args()
    mixed_values = (
        arguments.new_shard_manifest,
        arguments.anchor_shard_manifest,
        arguments.selection_validation_manifest,
        arguments.new_rows_per_batch,
        arguments.anchor_rows_per_batch,
    )
    mixed_requested = any(
        bool(value) if isinstance(value, list) else value is not None
        for value in mixed_values
    )
    if mixed_requested and (
        not arguments.new_shard_manifest
        or not arguments.anchor_shard_manifest
        or not arguments.selection_validation_manifest
        or arguments.new_rows_per_batch is None
        or arguments.anchor_rows_per_batch is None
    ):
        parser.error(
            "mixed training requires new, anchor, and selection-validation "
            "manifests plus both per-batch row quotas"
        )
    if mixed_requested and (
        arguments.shard_manifests
        or arguments.pack_report
        or arguments.tiny_fixture
        or arguments.bootstrap_only
    ):
        parser.error("mixed training cannot be combined with legacy training inputs")
    if mixed_requested and arguments.reveal_test:
        parser.error("mixed training has no protected test input to reveal")
    if mixed_requested and (
        arguments.new_rows_per_batch <= 0
        or arguments.anchor_rows_per_batch <= 0
        or arguments.new_rows_per_batch + arguments.anchor_rows_per_batch
        != arguments.batch_size
    ):
        parser.error("mixed per-batch row quotas must be positive and sum to batch size")
    if arguments.bootstrap_only and (
        arguments.tiny_fixture
        or arguments.shard_manifests
        or arguments.pack_report
        or arguments.seed_checkpoint_directory
        or arguments.resume_seeds
        or mixed_requested
    ):
        parser.error("--bootstrap-only does not accept training or checkpoint inputs")
    if arguments.resume_seeds and arguments.seed_checkpoint_directory is None:
        parser.error("--resume-seeds requires --seed-checkpoint-directory")
    training_inputs = sum(
        (
            arguments.tiny_fixture,
            bool(arguments.shard_manifests),
            bool(arguments.pack_report),
            mixed_requested,
        )
    )
    if not arguments.bootstrap_only and training_inputs != 1:
        parser.error(
            "choose exactly one of --tiny-fixture, --pack-report, or shard manifests"
        )
    if arguments.epochs <= 0 or arguments.patience <= 0 or arguments.batch_size <= 0:
        parser.error("epochs, patience, and batch size must be positive")
    if (
        not math.isfinite(arguments.learning_rate)
        or arguments.learning_rate <= 0.0
        or not math.isfinite(arguments.weight_decay)
        or arguments.weight_decay < 0.0
    ):
        parser.error("learning rate and weight decay are invalid")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.bootstrap_only:
        selected = initialize(arguments.seeds[0])
        training_report = {
            "kind": "deterministic-finite-bootstrap-not-trained-not-selected",
            "seed": arguments.seeds[0],
        }
        source_manifests: list[dict] = []
        runtime_name = "jacek_replay_bfm_bootstrap.runtime"
    else:
        mixed_training = None
        selection_validation = None
        source_roles: list[str]
        if arguments.tiny_fixture:
            fixture = tiny_fixture_samples()
            shard_directory = arguments.output_directory / "tiny-shards"
            manifests = [
                write_csr_shard(
                    shard_directory,
                    split,
                    fixture[split],
                    provenance={"fixture": "deterministic-tiny-v1"},
                )[1]
                for split in ("train", "validation", "test")
            ]
            source_roles = ["legacy"] * len(manifests)
        elif arguments.pack_report:
            try:
                pack_report = json.loads(arguments.pack_report.read_bytes())
                if pack_report.get("schema") != "papersoccer.jacek-replay-pack-report.v1":
                    raise ValueError("unexpected pack-report schema")
                shard_records = pack_report["shards"]
                manifests = [
                    pathlib.Path(shard_records[split]["manifest"])
                    for split in ("train", "validation", "test")
                ]
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                parser.error(f"invalid --pack-report: {error}")
            source_roles = ["legacy"] * len(manifests)
        elif mixed_requested:
            new_manifests = arguments.new_shard_manifest
            anchor_manifests = arguments.anchor_shard_manifest
            validation_manifests = arguments.selection_validation_manifest
            manifests = [*new_manifests, *anchor_manifests, *validation_manifests]
            source_roles = (
                ["new"] * len(new_manifests)
                + ["anchor"] * len(anchor_manifests)
                + ["selection-validation"] * len(validation_manifests)
            )
        else:
            manifests = arguments.shard_manifests
            source_roles = ["legacy"] * len(manifests)
        loaded = [load_csr_shard(path) for path in manifests]
        validate_shard_collection(loaded)
        if mixed_requested:
            new_count = len(arguments.new_shard_manifest)
            anchor_count = len(arguments.anchor_shard_manifest)
            new_shards = loaded[:new_count]
            anchor_shards = loaded[new_count : new_count + anchor_count]
            validation_shards = loaded[new_count + anchor_count :]
            if any(shard.split != "train" for shard in (*new_shards, *anchor_shards)):
                parser.error("new and anchor shard manifests must have split=train")
            if any(shard.split != "validation" for shard in validation_shards):
                parser.error("selection validation manifests must have split=validation")
            new_dataset = combine_shards(new_shards)
            anchor_dataset = combine_shards(anchor_shards)
            selection_validation = combine_shards(validation_shards)
            mixed_training = MixedTraining(
                new_dataset,
                anchor_dataset,
                arguments.new_rows_per_batch,
                arguments.anchor_rows_per_batch,
            )
            datasets = {"validation": selection_validation}
        else:
            datasets = {
                split: combine_shards(
                    [shard for shard in loaded if shard.split == split]
                )
                for split in ("train", "validation", "test")
            }
        del loaded
        source_manifests = [json.loads(path.read_bytes()) for path in manifests]
        selected, training_report = train_three_seeds(
            datasets,
            seeds=arguments.seeds,
            epochs=1 if arguments.tiny_fixture and arguments.epochs == 50 else arguments.epochs,
            patience=arguments.patience,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            reveal_test=arguments.reveal_test,
            seed_workers=arguments.seed_workers,
            seed_checkpoint_directory=arguments.seed_checkpoint_directory,
            resume_seeds=arguments.resume_seeds,
            input_shard_identities=_shard_identities(manifests),
            mixed_training=mixed_training,
            selection_validation=selection_validation,
        )
        runtime_name = "jacek_replay_bfm.runtime"

    runtime_path = arguments.output_directory / runtime_name
    runtime_report = export_runtime(runtime_path, selected)
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "status": (
            "bootstrap-not-trained-not-selected"
            if arguments.bootstrap_only
            else "research-candidate-not-game-gated"
        ),
        "feature_schema": features.FEATURE_SCHEMA,
        "architecture": {
            "dimensions": [features.INPUT_COUNT, HIDDEN_ONE, HIDDEN_TWO, OUTPUT_COUNT],
            "biases": False,
            "activations": ["square-leaky-0.01", "leaky-relu-0.01", "tanh"],
            "payload_layout": "w1-input-major,w2-input-major,w3",
        },
        "target": target_metadata_from_shard_provenance(
            source_manifests, source_roles
        ) if not arguments.bootstrap_only else {
            "policy_provenance": "bootstrap-has-no-training-targets",
            "declared_policies": [],
            "undeclared_roles": [],
            "loss": "weighted-huber-delta-0.25",
            "policy_head": False,
        },
        "runtime": {"path": runtime_name, **runtime_report},
        "training": training_report,
        "source_shards": source_manifests,
        "tool_sha256": {
            "trainer": _sha256(pathlib.Path(__file__).read_bytes()),
            "corpus": _sha256(pathlib.Path(corpus.__file__).read_bytes()),
            "features": _sha256(pathlib.Path(features.__file__).read_bytes()),
        },
    }
    manifest_path = arguments.output_directory / f"{runtime_name}.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({"manifest": str(manifest_path), **runtime_report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
