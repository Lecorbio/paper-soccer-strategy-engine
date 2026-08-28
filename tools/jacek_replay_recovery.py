#!/usr/bin/env python3
"""Train one evidence-gated Jacek replay recovery arm.

This is deliberately separate from :mod:`jacek_replay_train`: campaign
training keeps its existing semantics while rebuild experiments bind two
independent reference runtimes, exact input manifests, a layer freeze mask,
and a fixed two-stream schedule.  A process may prepare the immutable inputs
once and reuse them for every learning-rate/layer/seed arm.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import threading
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_train as training  # noqa: E402


RECOVERY_REPORT_SCHEMA = "papersoccer.jacek-replay-recovery-report.v1"
RECOVERY_RECEIPT_SCHEMA = "papersoccer.jacek-replay-recovery-receipt.v1"
RESIDUAL_RECOVERY_REPORT_SCHEMA = (
    "papersoccer.jacek-replay-residual-recovery-report.v1"
)
RESIDUAL_RECOVERY_RECEIPT_SCHEMA = (
    "papersoccer.jacek-replay-residual-recovery-receipt.v1"
)
V6_JOINT_REPORT_SCHEMA = "papersoccer.jacek-replay-v6-joint-report.v1"
V6_JOINT_RECEIPT_SCHEMA = "papersoccer.jacek-replay-v6-joint-receipt.v1"

V5_NONINFERIORITY = "v5-recovery-noninferiority"
EPOCH_ZERO_IMPROVEMENT = "epoch-zero-improvement"
SELECTION_POLICIES = (V5_NONINFERIORITY, EPOCH_ZERO_IMPROVEMENT)

NEW_ROWS_PER_BATCH = 64
ANCHOR_ROWS_PER_BATCH = 192
BATCH_SIZE = NEW_ROWS_PER_BATCH + ANCHOR_ROWS_PER_BATCH
NEW_LOSS_COEFFICIENT = 0.25
ANCHOR_LOSS_COEFFICIENT = 0.75
HUBER_DELTA = 0.25
CHECKPOINT_INTERVAL = 782
MAX_ANCHOR_PASSES = 2
WEIGHT_DECAY = 1e-5
GRADIENT_NORM_CLIP = 5.0
SIGN_TOLERANCE = 0.005
HUBER_RATIO = 1.02

RUNTIME_NAME = "recovery.runtime"
REPORT_NAME = "recovery.report.json"
RECEIPT_NAME = "recovery.receipt.json"
RESIDUAL_RUNTIME_NAME = "residual-recovery.runtime"
RESIDUAL_REPORT_NAME = "residual-recovery.report.json"
RESIDUAL_RECEIPT_NAME = "residual-recovery.receipt.json"
V6_JOINT_RUNTIME_NAME = "v6-joint.runtime"
V6_JOINT_REPORT_NAME = "v6-joint.report.json"
V6_JOINT_RECEIPT_NAME = "v6-joint.receipt.json"
RESIDUAL_LEARNING_RATES = (1e-4, 3e-4, 1e-3)
V6_JOINT_BASE_LEARNING_RATE = 6e-5
V6_JOINT_REFERENCE_NEW_SAMPLES = 50_000
V6_JOINT_EPOCHS = 50
V6_JOINT_PATIENCE = 8
_RECEIPT_FIELDS = {
    "schema",
    "configuration",
    "inputs",
    "producer",
    "runtime",
    "report",
    "body_sha256",
}

_TRAINABLE_NAMES = {
    "w3": ("w3",),
    "w2-w3": ("w2", "w3"),
    "all": ("w1", "w2", "w3"),
}
_TRAINABLE_ALIASES = {"w2+w3": "w2-w3"}
_POLICY_ALIASES = {
    "v5-recovery": V5_NONINFERIORITY,
    "other-base": EPOCH_ZERO_IMPROVEMENT,
}

_CACHE_LOCK = threading.RLock()
_DATASET_CACHE: dict[str, "PreparedRecoveryDatasets"] = {}
_SHARD_CACHE: dict[tuple[str, str], training.SparseShard] = {}
_RUNTIME_CACHE: dict[str, tuple[Mapping[str, np.ndarray], bytes]] = {}
_METRIC_CACHE: dict[tuple[str, str], bytes] = {}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Copy an array onto an immutable bytes buffer.

    Merely clearing NumPy's WRITEABLE flag is reversible when an array owns a
    mutable allocation.  A bytes-backed view cannot be made writeable again,
    which is required because worker threads share these cached inputs.
    """

    contiguous = np.ascontiguousarray(value)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _readonly_parameters(
    parameters: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return MappingProxyType(
        {name: _readonly_array(value) for name, value in parameters.items()}
    )


def _readonly_dataset(dataset: training.Dataset) -> training.Dataset:
    return training.Dataset(
        _readonly_array(dataset.indptr),
        _readonly_array(dataset.indices),
        _readonly_array(dataset.targets),
        _readonly_array(dataset.weights),
        _readonly_array(dataset.group_ids),
    )


@dataclasses.dataclass(frozen=True)
class PreparedRecoveryDatasets:
    """One immutable, provenance-bound four-role evidence bundle."""

    new: training.Dataset
    anchor: training.Dataset
    selection: training.Dataset
    retention: training.Dataset
    identity_json: bytes

    def receipt_identity(self) -> dict:
        return json.loads(self.identity_json)


@dataclasses.dataclass(frozen=True)
class RecoveryInputs:
    """Validated immutable datasets and runtimes shared by recovery arms."""

    new: training.Dataset
    anchor: training.Dataset
    selection: training.Dataset
    retention: training.Dataset
    initial_parameters: Mapping[str, np.ndarray]
    retention_reference_parameters: Mapping[str, np.ndarray]
    new_reference_parameters: Mapping[str, np.ndarray]
    identity_json: bytes

    def receipt_identity(self) -> dict:
        return json.loads(self.identity_json)


@dataclasses.dataclass(frozen=True)
class RecoveryConfiguration:
    """The only per-arm choices; all safety policy is fixed by this module."""

    trainable_layers: str
    learning_rate: float
    seed: int
    selection_policy: str = V5_NONINFERIORITY

    def normalized(self) -> "RecoveryConfiguration":
        layers = _TRAINABLE_ALIASES.get(self.trainable_layers, self.trainable_layers)
        policy = _POLICY_ALIASES.get(self.selection_policy, self.selection_policy)
        if layers not in _TRAINABLE_NAMES:
            raise ValueError("trainable_layers must be w3, w2-w3, or all")
        if policy not in SELECTION_POLICIES:
            raise ValueError(
                "selection_policy must be v5-recovery-noninferiority or "
                "epoch-zero-improvement"
            )
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or float(self.learning_rate) <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 1 << 64
        ):
            raise ValueError("seed must fit uint64")
        return RecoveryConfiguration(layers, float(self.learning_rate), self.seed, policy)


@dataclasses.dataclass(frozen=True)
class ResidualRecoveryConfiguration:
    """Fixed adapter-only fallback arm over a v1 base runtime."""

    learning_rate: float
    seed: int

    def normalized(self) -> "ResidualRecoveryConfiguration":
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or float(self.learning_rate) not in RESIDUAL_LEARNING_RATES
        ):
            raise ValueError(
                "residual learning_rate must be 1e-4, 3e-4, or 1e-3"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 1 << 64
        ):
            raise ValueError("seed must fit uint64")
        return ResidualRecoveryConfiguration(float(self.learning_rate), self.seed)


@dataclasses.dataclass(frozen=True)
class V6JointConfiguration:
    """The fixed retention-safe v6 joint recipe; only row-order seed varies."""

    seed: int

    def normalized(self) -> "V6JointConfiguration":
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 1 << 64
        ):
            raise ValueError("seed must fit uint64")
        return V6JointConfiguration(self.seed)


def _manifest_probe(path: pathlib.Path) -> dict:
    """Re-hash a manifest and its NPZ before every cache lookup."""

    path = path.resolve()
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid recovery shard manifest: {path}") from error
    manifest_sha256 = _sha256(payload)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != training.SHARD_SCHEMA
        or path.suffix != ".json"
        or path.stem != manifest_sha256
    ):
        raise ValueError(f"recovery shard manifest is not content addressed: {path}")
    npz_name = manifest.get("npz")
    npz_sha256 = manifest.get("npz_sha256")
    if (
        not isinstance(npz_name, str)
        or pathlib.PurePath(npz_name).name != npz_name
        or not isinstance(npz_sha256, str)
        or len(npz_sha256) != 64
    ):
        raise ValueError(f"invalid recovery shard identity: {path}")
    npz_path = path.parent / npz_name
    if npz_path.stem != npz_sha256 or _sha256_file(npz_path) != npz_sha256:
        raise ValueError(f"recovery shard NPZ SHA-256 mismatch: {path}")
    return {
        "manifest_sha256": manifest_sha256,
        "npz_sha256": npz_sha256,
        "split": manifest.get("split"),
        "samples": manifest.get("samples"),
        "active_features": manifest.get("active_features"),
    }


def _normalize_manifest_paths(
    paths: Sequence[pathlib.Path | str], role: str
) -> tuple[pathlib.Path, ...]:
    normalized = tuple(pathlib.Path(path).resolve() for path in paths)
    if not normalized:
        raise ValueError(f"{role} requires at least one shard manifest")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{role} repeats a shard manifest")
    return normalized


def _validate_role_isolation(
    shards: Mapping[str, Sequence[training.SparseShard]],
) -> None:
    group_roles: dict[bytes, str] = {}
    fingerprint_roles: dict[bytes, str] = {}
    for role in ("new", "anchor", "selection", "retention"):
        for shard in shards[role]:
            for row in range(len(shard)):
                group = bytes(shard.group_ids[row])
                previous_role = group_roles.setdefault(group, role)
                if previous_role != role:
                    raise ValueError(
                        f"recovery root group crosses {previous_role} and {role}"
                    )
                fingerprint = corpus.canonical_feature_fingerprint(
                    shard.active(row).tolist()
                )
                previous_role = fingerprint_roles.get(fingerprint)
                if previous_role is None:
                    fingerprint_roles[fingerprint] = role
                elif previous_role == role and role in {"new", "selection"}:
                    raise ValueError(
                        f"recovery {role} repeats a canonical fingerprint"
                    )
                elif previous_role != role:
                    raise ValueError(
                        "recovery exact/rotated/reflected fingerprint crosses "
                        f"{previous_role} and {role}"
                    )


def _load_shard_cached(
    path: pathlib.Path, probe: Mapping[str, object]
) -> training.SparseShard:
    key = (str(probe["manifest_sha256"]), str(probe["npz_sha256"]))
    with _CACHE_LOCK:
        cached = _SHARD_CACHE.get(key)
    if cached is not None:
        return cached
    shard = training.load_csr_shard(path)
    for name in ("indptr", "indices", "targets", "weights", "group_ids"):
        getattr(shard, name).setflags(write=False)
    with _CACHE_LOCK:
        previous = _SHARD_CACHE.setdefault(key, shard)
    return previous


def _prepare_datasets(
    manifests: Mapping[str, tuple[pathlib.Path, ...]],
    *, prevalidated_role_isolation: bool,
) -> PreparedRecoveryDatasets:
    probes = {
        role: [
            {"ordinal": ordinal, **_manifest_probe(path)}
            for ordinal, path in enumerate(paths)
        ]
        for role, paths in manifests.items()
    }
    for role, role_probes in probes.items():
        manifest_hashes = [item["manifest_sha256"] for item in role_probes]
        npz_hashes = [item["npz_sha256"] for item in role_probes]
        if (
            len(set(manifest_hashes)) != len(manifest_hashes)
            or len(set(npz_hashes)) != len(npz_hashes)
        ):
            raise ValueError(
                f"{role} repeats content-identical manifest or NPZ evidence"
            )
    cache_material = {
        "manifests": {
            role: [
                {
                    "manifest_sha256": item["manifest_sha256"],
                    "npz_sha256": item["npz_sha256"],
                }
                for item in probes[role]
            ]
            for role in sorted(probes)
        },
        "prevalidated_role_isolation": prevalidated_role_isolation,
    }
    cache_key = _sha256(training.canonical_json_bytes(cache_material))
    with _CACHE_LOCK:
        cached = _DATASET_CACHE.get(cache_key)
        if cached is not None:
            return cached

        loaded = {
            role: tuple(
                _load_shard_cached(path, probes[role][ordinal])
                for ordinal, path in enumerate(paths)
            )
            for role, paths in manifests.items()
        }
        expected_splits = {
            "new": "train",
            "anchor": "train",
            "selection": "validation",
            "retention": "validation",
        }
        for role, role_shards in loaded.items():
            if any(shard.split != expected_splits[role] for shard in role_shards):
                raise ValueError(
                    f"{role} manifests must use split {expected_splits[role]}"
                )
        if not prevalidated_role_isolation:
            _validate_role_isolation(loaded)
        datasets = {
            role: _readonly_dataset(training.combine_shards(role_shards))
            for role, role_shards in loaded.items()
        }
        identity = {
            "feature_schema": training.features.FEATURE_SCHEMA,
            "role_isolation": (
                "deep-rebuild-corpus-receipt"
                if prevalidated_role_isolation
                else "recomputed-from-sparse-rows"
            ),
            "manifests": probes,
            "datasets": {
                role: training._dataset_identity(dataset)
                for role, dataset in sorted(datasets.items())
            },
        }
        prepared = PreparedRecoveryDatasets(
            datasets["new"],
            datasets["anchor"],
            datasets["selection"],
            datasets["retention"],
            training.canonical_json_bytes(identity),
        )
        _DATASET_CACHE[cache_key] = prepared
        return prepared


def _load_runtime_cached(
    path: pathlib.Path | str,
) -> tuple[Mapping[str, np.ndarray], dict]:
    path = pathlib.Path(path).resolve()
    try:
        artifact_sha256 = _sha256_file(path)
    except OSError as error:
        raise ValueError(f"cannot read recovery runtime: {path}") from error
    with _CACHE_LOCK:
        cached = _RUNTIME_CACHE.get(artifact_sha256)
        if cached is not None:
            parameters, report_json = cached
            return parameters, json.loads(report_json)
        parameters, report = training.load_runtime(path)
        if report["artifact_sha256"] != artifact_sha256:
            raise ValueError(f"recovery runtime changed while loading: {path}")
        readonly = _readonly_parameters(parameters)
        report_json = training.canonical_json_bytes(report)
        _RUNTIME_CACHE[artifact_sha256] = (readonly, report_json)
        return readonly, json.loads(report_json)


def prepare_recovery_datasets(
    *,
    new_manifests: Sequence[pathlib.Path | str],
    anchor_manifests: Sequence[pathlib.Path | str],
    selection_manifests: Sequence[pathlib.Path | str],
    retention_manifests: Sequence[pathlib.Path | str],
    prevalidated_role_isolation: bool = False,
) -> PreparedRecoveryDatasets:
    """Validate, freeze, and cache the four exact recovery dataset roles.

    Manifest and NPZ hashes are recomputed before a cached bundle is reused.
    Call this once per search/Rank-4 channel, then bind every base/runtime trio
    with :func:`bind_recovery_runtimes` without touching the large NPZs again.
    """

    manifests = {
        "new": _normalize_manifest_paths(new_manifests, "new"),
        "anchor": _normalize_manifest_paths(anchor_manifests, "anchor"),
        "selection": _normalize_manifest_paths(selection_manifests, "selection"),
        "retention": _normalize_manifest_paths(retention_manifests, "retention"),
    }
    return _prepare_datasets(
        manifests,
        prevalidated_role_isolation=prevalidated_role_isolation,
    )


def bind_recovery_runtimes(
    datasets: PreparedRecoveryDatasets,
    *,
    initial_runtime: pathlib.Path | str,
    retention_reference_runtime: pathlib.Path | str,
    new_reference_runtime: pathlib.Path | str,
) -> RecoveryInputs:
    """Bind three validated runtimes without re-reading dataset artifacts."""

    if not isinstance(datasets, PreparedRecoveryDatasets):
        raise TypeError("datasets must be prepared by prepare_recovery_datasets")
    initial_parameters, initial_report = _load_runtime_cached(initial_runtime)
    retention_parameters, retention_report = _load_runtime_cached(
        retention_reference_runtime
    )
    new_parameters, new_report = _load_runtime_cached(new_reference_runtime)
    expected_v1 = set(training.BASE_PARAMETER_NAMES)
    for role, parameters in (
        ("initial", initial_parameters),
        ("retention reference", retention_parameters),
        ("new-data reference", new_parameters),
    ):
        if set(parameters) != expected_v1:
            raise ValueError(
                f"{role} must be a v1 base runtime for recovery training"
            )
    dataset_identity = datasets.receipt_identity()
    identity = {
        **dataset_identity,
        "runtimes": {
            "initial": initial_report,
            "retention_reference": retention_report,
            "new_reference": new_report,
        },
    }
    return RecoveryInputs(
        datasets.new,
        datasets.anchor,
        datasets.selection,
        datasets.retention,
        initial_parameters,
        retention_parameters,
        new_parameters,
        training.canonical_json_bytes(identity),
    )


def prepare_recovery_inputs(
    *,
    initial_runtime: pathlib.Path | str,
    retention_reference_runtime: pathlib.Path | str,
    new_reference_runtime: pathlib.Path | str,
    new_manifests: Sequence[pathlib.Path | str],
    anchor_manifests: Sequence[pathlib.Path | str],
    selection_manifests: Sequence[pathlib.Path | str],
    retention_manifests: Sequence[pathlib.Path | str],
) -> RecoveryInputs:
    """Convenience wrapper for one-off dataset preparation and runtime binding."""

    datasets = prepare_recovery_datasets(
        new_manifests=new_manifests,
        anchor_manifests=anchor_manifests,
        selection_manifests=selection_manifests,
        retention_manifests=retention_manifests,
    )
    return bind_recovery_runtimes(
        datasets,
        initial_runtime=initial_runtime,
        retention_reference_runtime=retention_reference_runtime,
        new_reference_runtime=new_reference_runtime,
    )


def clear_recovery_input_cache() -> None:
    """Clear process-local immutable caches (primarily for isolation tests)."""

    with _CACHE_LOCK:
        _DATASET_CACHE.clear()
        _SHARD_CACHE.clear()
        _RUNTIME_CACHE.clear()
        _METRIC_CACHE.clear()


def train_recovery_batch(
    parameters: Mapping[str, np.ndarray],
    optimizer: training.AdamW,
    mixed: training.MixedTraining,
    new_rows: np.ndarray,
    anchor_rows: np.ndarray,
    *,
    trainable_layers: str,
) -> float:
    """Apply one fixed 64/192, 0.25/0.75 recovery optimizer step."""

    trainable_layers = _TRAINABLE_ALIASES.get(trainable_layers, trainable_layers)
    if trainable_layers not in _TRAINABLE_NAMES:
        raise ValueError("trainable_layers must be w3, w2-w3, or all")
    if set(parameters) != set(training.BASE_PARAMETER_NAMES):
        raise ValueError("train_recovery_batch accepts only v1 base parameters")
    mixed.validate(BATCH_SIZE)
    if len(new_rows) != NEW_ROWS_PER_BATCH or len(anchor_rows) != ANCHOR_ROWS_PER_BATCH:
        raise ValueError("recovery batch must contain exactly 64 new and 192 anchor rows")

    active = (*mixed.new.active_rows(new_rows), *mixed.anchor.active_rows(anchor_rows))
    targets = np.concatenate(
        (mixed.new.targets[new_rows], mixed.anchor.targets[anchor_rows])
    )
    weights = training._source_normalized_weights(
        mixed.new.weights[new_rows],
        mixed.anchor.weights[anchor_rows],
        new_loss_coefficient=NEW_LOSS_COEFFICIENT,
        anchor_loss_coefficient=ANCHOR_LOSS_COEFFICIENT,
    )
    prediction, cache = training.forward(parameters, active)
    first_pre, first, second_pre, second, output_pre = cache
    loss, output_gradient = training._weighted_huber(
        prediction, targets, weights, HUBER_DELTA
    )
    output_pre_gradient = output_gradient * (1.0 - np.tanh(output_pre) ** 2)

    names = _TRAINABLE_NAMES[trainable_layers]
    gradients: dict[str, np.ndarray] = {
        "w3": second.T @ output_pre_gradient,
    }
    if "w2" in names or "w1" in names:
        second_gradient = output_pre_gradient[:, None] * parameters["w3"][None, :]
        second_pre_gradient = second_gradient * training._leaky_derivative(second_pre)
        if "w2" in names:
            gradients["w2"] = first.T @ second_pre_gradient
        if "w1" in names:
            first_gradient = second_pre_gradient @ parameters["w2"].T
            first_pre_gradient = first_gradient * training._hidden_one_derivative(
                first_pre
            )
            gradients["w1"] = np.zeros_like(parameters["w1"])
            for row, indices in enumerate(active):
                np.add.at(gradients["w1"], indices, first_pre_gradient[row])
    gradients = {name: gradients[name] for name in names}
    norm = math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
    if norm > GRADIENT_NORM_CLIP:
        scale = np.float32(GRADIENT_NORM_CLIP / norm)
        for gradient in gradients.values():
            gradient *= scale
    optimizer.update({name: parameters[name] for name in names}, gradients)
    return loss


def _metric_is_valid(value: object, samples: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "samples",
        "weighted_huber",
        "sign_accuracy",
        "correlation",
        "mae",
        "prediction_mean",
    } or value.get("samples") != samples:
        return False
    numbers = [
        value.get("weighted_huber"),
        value.get("sign_accuracy"),
        value.get("correlation"),
        value.get("mae"),
        value.get("prediction_mean"),
    ]
    return all(
        not isinstance(number, bool)
        and isinstance(number, (int, float))
        and math.isfinite(float(number))
        for number in numbers
    ) and (
        value["weighted_huber"] >= 0.0
        and value["mae"] >= 0.0
        and 0.0 <= value["sign_accuracy"] <= 1.0
        and -1.000_000_1 <= value["correlation"] <= 1.000_000_1
    )


def selection_key(
    metric_report: Mapping[str, float], runtime_hash: str = ""
) -> tuple[float, float, float, str]:
    """Frozen checkpoint order: Huber, sign, correlation, runtime hash."""

    return (
        float(metric_report["weighted_huber"]),
        -float(metric_report["sign_accuracy"]),
        -float(metric_report["correlation"]),
        runtime_hash,
    )


def _thresholds(reference: Mapping[str, float]) -> dict[str, float]:
    return {
        "minimum_sign_accuracy": max(
            0.0, float(reference["sign_accuracy"]) - SIGN_TOLERANCE
        ),
        "maximum_weighted_huber": float(reference["weighted_huber"]) * HUBER_RATIO,
        "sign_tolerance": SIGN_TOLERANCE,
        "huber_ratio": HUBER_RATIO,
    }


def _passes_thresholds(
    candidate: Mapping[str, float], thresholds: Mapping[str, float]
) -> bool:
    return bool(
        float(candidate["sign_accuracy"])
        >= float(thresholds["minimum_sign_accuracy"])
        and float(candidate["weighted_huber"])
        <= float(thresholds["maximum_weighted_huber"])
    )


def new_gate_passes(
    policy: str,
    candidate: Mapping[str, float],
    *,
    epoch_zero: Mapping[str, float],
    reference_thresholds: Mapping[str, float],
) -> bool:
    """Evaluate the v5-reference or non-v5 epoch-zero new-data rule."""

    policy = _POLICY_ALIASES.get(policy, policy)
    if policy == V5_NONINFERIORITY:
        return _passes_thresholds(candidate, reference_thresholds)
    if policy == EPOCH_ZERO_IMPROVEMENT:
        return selection_key(candidate)[:3] < selection_key(epoch_zero)[:3]
    raise ValueError("unknown recovery selection policy")


def _coverage(update: int, inputs: RecoveryInputs) -> dict:
    result = {}
    for role, quota, count in (
        ("new", NEW_ROWS_PER_BATCH, len(inputs.new)),
        ("anchor", ANCHOR_ROWS_PER_BATCH, len(inputs.anchor)),
    ):
        seen = update * quota
        result[role] = {
            "dataset_rows": count,
            "rows_seen": seen,
            "complete_permutations": seen // count,
            "permutation_offset": seen % count,
            "complete_coverage": seen >= count,
        }
    return result


def _schedule(inputs: RecoveryInputs) -> tuple[int, list[int]]:
    max_updates = (MAX_ANCHOR_PASSES * len(inputs.anchor)) // ANCHOR_ROWS_PER_BATCH
    minimum_eligible = max(
        math.ceil(len(inputs.new) / NEW_ROWS_PER_BATCH),
        math.ceil(len(inputs.anchor) / ANCHOR_ROWS_PER_BATCH),
    )
    if max_updates <= 0:
        raise ValueError("anchor corpus is too small for one fixed recovery batch")
    if minimum_eligible > max_updates:
        raise ValueError(
            "fixed two-anchor-pass budget cannot cover both recovery streams"
        )
    checkpoints = list(range(CHECKPOINT_INTERVAL, max_updates + 1, CHECKPOINT_INTERVAL))
    if not checkpoints or checkpoints[-1] != max_updates:
        checkpoints.append(max_updates)
    return max_updates, checkpoints


def _configuration_report(
    inputs: RecoveryInputs, configuration: RecoveryConfiguration
) -> dict:
    configuration = configuration.normalized()
    max_updates, checkpoints = _schedule(inputs)
    return {
        "architecture": {
            "dimensions": [
                training.features.INPUT_COUNT,
                training.HIDDEN_ONE,
                training.HIDDEN_TWO,
                training.OUTPUT_COUNT,
            ],
            "biases": False,
            "runtime_version": 1,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": configuration.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_clip": GRADIENT_NORM_CLIP,
            "seed": configuration.seed,
            "trainable_layers": configuration.trainable_layers,
        },
        "batching": {
            "kind": "deterministic-continuous-two-stream-recovery-v1",
            "batch_size": BATCH_SIZE,
            "new_rows_per_batch": NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
            "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
            "checkpoint_updates": checkpoints,
            "maximum_updates": max_updates,
            "maximum_anchor_passes": MAX_ANCHOR_PASSES,
            "eligibility": "complete-new-and-anchor-coverage",
        },
        "loss": {
            "name": "separately-normalized-weighted-huber",
            "delta": HUBER_DELTA,
            "new_coefficient": NEW_LOSS_COEFFICIENT,
            "anchor_coefficient": ANCHOR_LOSS_COEFFICIENT,
        },
        "metrics_batch_size": 4_096,
        "resume_validation": {
            "kind": "full-deterministic-training-replay-and-byte-comparison",
            "cost": "approximately-one-complete-arm-training-run",
            "checkpoint_artifacts_persisted": False,
        },
        "selection": {
            "policy": configuration.selection_policy,
            "retention_reference": "canonical-retention-runtime",
            "retention_sign_tolerance": SIGN_TOLERANCE,
            "retention_huber_ratio": HUBER_RATIO,
            "order": "huber-sign-correlation-runtime-sha256",
            "fallback": "exact-initial-runtime-not-eligible",
        },
    }


def _residual_configuration_report(
    inputs: RecoveryInputs, configuration: ResidualRecoveryConfiguration
) -> dict:
    configuration = configuration.normalized()
    max_updates, checkpoints = _schedule(inputs)
    return {
        "architecture": {
            "base_dimensions": [
                training.features.INPUT_COUNT,
                training.HIDDEN_ONE,
                training.HIDDEN_TWO,
                training.OUTPUT_COUNT,
            ],
            "biases": False,
            "runtime_version": training.RUNTIME_V2_VERSION,
            "residual": {
                "source": "first-hidden-layer",
                "rank": training.RESIDUAL_RANK,
                "formula": (
                    "tanh(base_gain*base_logit+residual_bias+"
                    "leaky(h1*adapter_a)*adapter_b)"
                ),
                "initialization_seed": training.RESIDUAL_INITIALIZATION_SEED,
                "zero_adapter_prediction_equivalent_to_v1": True,
            },
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": configuration.learning_rate,
            "allowed_learning_rates": list(RESIDUAL_LEARNING_RATES),
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_clip": GRADIENT_NORM_CLIP,
            "seed": configuration.seed,
            "trainable_parameters": list(training.RESIDUAL_PARAMETER_NAMES),
            "frozen_parameters": list(training.BASE_PARAMETER_NAMES),
        },
        "batching": {
            "kind": "deterministic-continuous-two-stream-residual-recovery-v1",
            "batch_size": BATCH_SIZE,
            "new_rows_per_batch": NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
            "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
            "checkpoint_updates": checkpoints,
            "maximum_updates": max_updates,
            "maximum_anchor_passes": MAX_ANCHOR_PASSES,
            "eligibility": "complete-new-and-anchor-coverage",
        },
        "loss": {
            "name": "separately-normalized-weighted-huber",
            "delta": HUBER_DELTA,
            "new_coefficient": NEW_LOSS_COEFFICIENT,
            "anchor_coefficient": ANCHOR_LOSS_COEFFICIENT,
        },
        "metrics_batch_size": 4_096,
        "resume_validation": {
            "kind": "full-deterministic-training-replay-and-byte-comparison",
            "cost": "approximately-one-complete-arm-training-run",
            "checkpoint_artifacts_persisted": False,
        },
        "selection": {
            "policy": V5_NONINFERIORITY,
            "retention_reference": "canonical-retention-runtime",
            "retention_sign_tolerance": SIGN_TOLERANCE,
            "retention_huber_ratio": HUBER_RATIO,
            "order": "huber-sign-correlation-runtime-sha256",
            "fallback": "deterministic-zero-adapter-v2-not-eligible",
        },
    }


def v6_joint_learning_rate_policy(actual_new_samples: int) -> dict:
    """Return the exact capped inverse-update learning-rate scale from v6."""

    if (
        isinstance(actual_new_samples, bool)
        or not isinstance(actual_new_samples, int)
        or actual_new_samples <= 0
    ):
        raise ValueError("actual_new_samples must be a positive integer")
    reference_steps = math.ceil(
        V6_JOINT_REFERENCE_NEW_SAMPLES / NEW_ROWS_PER_BATCH
    )
    actual_steps = math.ceil(actual_new_samples / NEW_ROWS_PER_BATCH)
    scale = min(1.0, reference_steps / actual_steps)
    effective = V6_JOINT_BASE_LEARNING_RATE * scale
    return {
        "kind": "inverse-new-train-optimizer-steps-capped-v1",
        "base_learning_rate": V6_JOINT_BASE_LEARNING_RATE,
        "reference_new_samples": V6_JOINT_REFERENCE_NEW_SAMPLES,
        "actual_new_samples": actual_new_samples,
        "new_rows_per_batch": NEW_ROWS_PER_BATCH,
        "reference_optimizer_steps": reference_steps,
        "actual_optimizer_steps": actual_steps,
        "scale": scale,
        "effective_learning_rate": effective,
    }


def _v6_joint_configuration_report(
    inputs: RecoveryInputs, configuration: V6JointConfiguration
) -> dict:
    configuration = configuration.normalized()
    batches_per_epoch = math.ceil(len(inputs.new) / NEW_ROWS_PER_BATCH)
    anchor_rows_per_epoch = batches_per_epoch * ANCHOR_ROWS_PER_BATCH
    coverage_epoch = math.ceil(len(inputs.anchor) / anchor_rows_per_epoch)
    if coverage_epoch > V6_JOINT_EPOCHS:
        raise ValueError(
            "v6 joint 50-epoch schedule cannot complete anchor coverage"
        )
    return {
        "architecture": {
            "dimensions": [
                training.features.INPUT_COUNT,
                training.HIDDEN_ONE,
                training.HIDDEN_TWO,
                training.OUTPUT_COUNT,
            ],
            "biases": False,
            "runtime_version": training.RUNTIME_VERSION,
        },
        "optimizer": {
            "name": "adamw",
            "maximum_epochs": V6_JOINT_EPOCHS,
            "patience": V6_JOINT_PATIENCE,
            "patience_starts_after_complete_anchor_coverage": True,
            "learning_rate_policy": v6_joint_learning_rate_policy(len(inputs.new)),
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_clip": GRADIENT_NORM_CLIP,
            "seed": configuration.seed,
            "trainable_layers": "all",
        },
        "batching": {
            "kind": "deterministic-continuous-two-stream-coverage-v2",
            "batch_size": BATCH_SIZE,
            "new_rows_per_batch": NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
            "batches_per_epoch": batches_per_epoch,
            "epoch_length": "ceil-new-rows/new-quota-batches",
            "new_stream": "fresh-complete-permutation-each-epoch-with-padding",
            "anchor_cross_epoch": "continuous-no-repeat-until-permutation-complete",
            "anchor_rows_per_epoch": anchor_rows_per_epoch,
            "anchor_coverage_complete_epoch": coverage_epoch,
        },
        "loss": {
            "name": "separately-normalized-weighted-huber",
            "delta": HUBER_DELTA,
            "new_coefficient": NEW_LOSS_COEFFICIENT,
            "anchor_coefficient": ANCHOR_LOSS_COEFFICIENT,
        },
        "metrics_batch_size": 4_096,
        "resume_validation": {
            "kind": "full-deterministic-training-replay-and-byte-comparison",
            "cost": "approximately-one-complete-joint-training-run",
            "checkpoint_artifacts_persisted": False,
        },
        "selection": {
            "policy": EPOCH_ZERO_IMPROVEMENT,
            "retention_reference": "canonical-incumbent-runtime",
            "retention_sign_tolerance": SIGN_TOLERANCE,
            "retention_huber_ratio": HUBER_RATIO,
            "order": "huber-sign-correlation-runtime-sha256",
            "fallback": "exact-initial-runtime-not-eligible",
        },
    }


def _producer_identity() -> dict[str, str]:
    return {
        "recovery_trainer_sha256": _sha256_file(pathlib.Path(__file__)),
        "base_trainer_sha256": _sha256_file(pathlib.Path(training.__file__)),
        "corpus_sha256": _sha256_file(pathlib.Path(corpus.__file__)),
        "features_sha256": _sha256_file(pathlib.Path(training.features.__file__)),
    }


def _cached_metrics(
    parameters: Mapping[str, np.ndarray],
    runtime_identity: Mapping[str, object],
    dataset: training.Dataset,
    dataset_identity: Mapping[str, object],
) -> dict:
    key = (str(runtime_identity["artifact_sha256"]), str(dataset_identity["sha256"]))
    with _CACHE_LOCK:
        payload = _METRIC_CACHE.get(key)
    if payload is not None:
        return json.loads(payload)
    measured = training.metrics(parameters, dataset)
    payload = training.canonical_json_bytes(measured)
    with _CACHE_LOCK:
        previous = _METRIC_CACHE.setdefault(key, payload)
    return json.loads(previous)


def _reference_evidence(inputs: RecoveryInputs) -> tuple[dict, dict]:
    identity = inputs.receipt_identity()
    datasets = identity["datasets"]
    runtimes = identity["runtimes"]
    initial_selection = _cached_metrics(
        inputs.initial_parameters,
        runtimes["initial"],
        inputs.selection,
        datasets["selection"],
    )
    initial_retention = _cached_metrics(
        inputs.initial_parameters,
        runtimes["initial"],
        inputs.retention,
        datasets["retention"],
    )
    retention_reference = _cached_metrics(
        inputs.retention_reference_parameters,
        runtimes["retention_reference"],
        inputs.retention,
        datasets["retention"],
    )
    new_reference = _cached_metrics(
        inputs.new_reference_parameters,
        runtimes["new_reference"],
        inputs.selection,
        datasets["selection"],
    )
    evidence = {
        "epoch_zero": {
            "runtime": runtimes["initial"],
            "selection": initial_selection,
            "retention": initial_retention,
        },
        "retention_reference": {
            "runtime": runtimes["retention_reference"],
            "retention": retention_reference,
        },
        "new_reference": {
            "runtime": runtimes["new_reference"],
            "selection": new_reference,
        },
    }
    thresholds = {
        "retention": _thresholds(retention_reference),
        "new_reference": _thresholds(new_reference),
    }
    return evidence, thresholds


def _residual_reference_evidence(
    inputs: RecoveryInputs,
    zero_adapter: Mapping[str, np.ndarray],
) -> tuple[dict, dict]:
    evidence, thresholds = _reference_evidence(inputs)
    zero_runtime = training.runtime_bytes(zero_adapter)[1]
    identity = inputs.receipt_identity()
    zero_selection = _cached_metrics(
        zero_adapter,
        zero_runtime,
        inputs.selection,
        identity["datasets"]["selection"],
    )
    zero_retention = _cached_metrics(
        zero_adapter,
        zero_runtime,
        inputs.retention,
        identity["datasets"]["retention"],
    )
    if (
        zero_selection != evidence["epoch_zero"]["selection"]
        or zero_retention != evidence["epoch_zero"]["retention"]
    ):
        raise RuntimeError("zero residual adapter is not prediction-equivalent to v1")
    evidence["epoch_zero"] = {
        "runtime": zero_runtime,
        "base_runtime": identity["runtimes"]["initial"],
        "selection": zero_selection,
        "retention": zero_retention,
    }
    return evidence, thresholds


def _frozen_layers(configuration: RecoveryConfiguration) -> tuple[str, ...]:
    trainable = set(_TRAINABLE_NAMES[configuration.normalized().trainable_layers])
    return tuple(name for name in ("w1", "w2", "w3") if name not in trainable)


def _frozen_layers_match(
    candidate: Mapping[str, np.ndarray],
    initial: Mapping[str, np.ndarray],
    configuration: RecoveryConfiguration,
) -> bool:
    return all(
        np.array_equal(candidate[name], initial[name])
        for name in _frozen_layers(configuration)
    )


def _train(
    inputs: RecoveryInputs, configuration: RecoveryConfiguration
) -> tuple[dict[str, np.ndarray], dict]:
    configuration = configuration.normalized()
    configuration_report = _configuration_report(inputs, configuration)
    max_updates = configuration_report["batching"]["maximum_updates"]
    checkpoint_updates = set(configuration_report["batching"]["checkpoint_updates"])
    references, thresholds = _reference_evidence(inputs)

    parameters = {
        name: np.asarray(value, dtype=np.float32).copy()
        for name, value in inputs.initial_parameters.items()
    }
    names = _TRAINABLE_NAMES[configuration.trainable_layers]
    optimizer = training.AdamW(
        {name: parameters[name] for name in names},
        configuration.learning_rate,
        WEIGHT_DECAY,
    )
    mixed = training.MixedTraining(
        inputs.new,
        inputs.anchor,
        NEW_ROWS_PER_BATCH,
        ANCHOR_ROWS_PER_BATCH,
    )
    # Local compatibility use of the base trainer's pure deterministic stream.
    new_order = training._continuous_rows(
        len(inputs.new),
        max_updates * NEW_ROWS_PER_BATCH,
        seed=configuration.seed,
        stream="recovery:new",
    )
    anchor_order = training._continuous_rows(
        len(inputs.anchor),
        max_updates * ANCHOR_ROWS_PER_BATCH,
        seed=configuration.seed,
        stream="recovery:anchor",
    )
    history: list[dict] = []
    interval_losses: list[float] = []
    best: dict[str, np.ndarray] | None = None
    best_row: dict | None = None
    for update in range(1, max_updates + 1):
        new_start = (update - 1) * NEW_ROWS_PER_BATCH
        anchor_start = (update - 1) * ANCHOR_ROWS_PER_BATCH
        interval_losses.append(
            train_recovery_batch(
                parameters,
                optimizer,
                mixed,
                new_order[new_start : new_start + NEW_ROWS_PER_BATCH],
                anchor_order[anchor_start : anchor_start + ANCHOR_ROWS_PER_BATCH],
                trainable_layers=configuration.trainable_layers,
            )
        )
        if update not in checkpoint_updates:
            continue
        runtime_report = training.runtime_bytes(parameters)[1]
        selection = training.metrics(parameters, inputs.selection)
        coverage = _coverage(update, inputs)
        coverage_complete = bool(
            coverage["new"]["complete_coverage"]
            and coverage["anchor"]["complete_coverage"]
        )
        new_passed = new_gate_passes(
            configuration.selection_policy,
            selection,
            epoch_zero=references["epoch_zero"]["selection"],
            reference_thresholds=thresholds["new_reference"],
        )
        retention = None
        retention_passed = None
        if coverage_complete and new_passed:
            retention = training.metrics(parameters, inputs.retention)
            retention_passed = _passes_thresholds(retention, thresholds["retention"])
            retention_status = "evaluated"
        elif not coverage_complete:
            retention_status = "not-evaluated-incomplete-coverage"
        else:
            retention_status = "not-evaluated-new-data-gate-failed"
        eligible = bool(coverage_complete and new_passed and retention_passed)
        row = {
            "update": update,
            "average_training_weighted_huber": float(np.mean(interval_losses)),
            "coverage": coverage,
            "runtime": runtime_report,
            "selection": selection,
            "new_gate": {
                "policy": configuration.selection_policy,
                "passed": new_passed,
            },
            "retention_gate": {
                "status": retention_status,
                "metrics": retention,
                "passed": retention_passed,
            },
            "eligible": eligible,
        }
        interval_losses.clear()
        history.append(row)
        if eligible and (
            best_row is None
            or selection_key(selection, runtime_report["artifact_sha256"])
            < selection_key(
                best_row["selection"], best_row["runtime"]["artifact_sha256"]
            )
        ):
            best = {name: value.copy() for name, value in parameters.items()}
            best_row = row

    if best is None:
        selected = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in inputs.initial_parameters.items()
        }
        selected_update = 0
        status = "no-eligible-checkpoint"
        selected_selection = references["epoch_zero"]["selection"]
        selected_retention = references["epoch_zero"]["retention"]
        eligible = False
    else:
        selected = best
        if best_row is None:
            raise RuntimeError("eligible recovery candidate has no history row")
        selected_update = best_row["update"]
        status = "eligible-checkpoint-selected"
        selected_selection = best_row["selection"]
        selected_retention = best_row["retention_gate"]["metrics"]
        eligible = True
    if not _frozen_layers_match(selected, inputs.initial_parameters, configuration):
        raise RuntimeError("recovery optimizer modified a frozen layer")
    selected_runtime = training.runtime_bytes(selected)[1]
    report = {
        "schema": RECOVERY_REPORT_SCHEMA,
        "configuration": configuration_report,
        "inputs": inputs.receipt_identity(),
        "producer": _producer_identity(),
        "references": references,
        "thresholds": thresholds,
        "checkpoints": history,
        "result": {
            "status": status,
            "selected_update": selected_update,
            "eligible": eligible,
            "runtime": selected_runtime,
            "selection": selected_selection,
            "retention": selected_retention,
            "frozen_layers": list(_frozen_layers(configuration)),
            "frozen_layers_verified": True,
        },
    }
    return selected, report


def _residual_base_matches(
    candidate: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray]
) -> bool:
    return all(
        np.array_equal(candidate[name], base[name])
        for name in training.BASE_PARAMETER_NAMES
    )


def _train_residual(
    inputs: RecoveryInputs, configuration: ResidualRecoveryConfiguration
) -> tuple[dict[str, np.ndarray], dict]:
    configuration = configuration.normalized()
    configuration_report = _residual_configuration_report(inputs, configuration)
    max_updates = configuration_report["batching"]["maximum_updates"]
    checkpoint_updates = set(configuration_report["batching"]["checkpoint_updates"])
    zero_adapter = training.initialize_residual_adapter(inputs.initial_parameters)
    references, thresholds = _residual_reference_evidence(inputs, zero_adapter)
    parameters = {name: value.copy() for name, value in zero_adapter.items()}
    optimizer = training.AdamW(
        {
            name: parameters[name]
            for name in training.RESIDUAL_PARAMETER_NAMES
        },
        configuration.learning_rate,
        WEIGHT_DECAY,
    )
    mixed = training.MixedTraining(
        inputs.new,
        inputs.anchor,
        NEW_ROWS_PER_BATCH,
        ANCHOR_ROWS_PER_BATCH,
    )
    new_order = training._continuous_rows(
        len(inputs.new),
        max_updates * NEW_ROWS_PER_BATCH,
        seed=configuration.seed,
        stream="residual-recovery:new",
    )
    anchor_order = training._continuous_rows(
        len(inputs.anchor),
        max_updates * ANCHOR_ROWS_PER_BATCH,
        seed=configuration.seed,
        stream="residual-recovery:anchor",
    )
    history: list[dict] = []
    interval_losses: list[float] = []
    best: dict[str, np.ndarray] | None = None
    best_row: dict | None = None
    for update in range(1, max_updates + 1):
        new_start = (update - 1) * NEW_ROWS_PER_BATCH
        anchor_start = (update - 1) * ANCHOR_ROWS_PER_BATCH
        interval_losses.append(
            training.train_mixed_batch(
                parameters,
                optimizer,
                mixed,
                new_order[new_start : new_start + NEW_ROWS_PER_BATCH],
                anchor_order[
                    anchor_start : anchor_start + ANCHOR_ROWS_PER_BATCH
                ],
                huber_delta=HUBER_DELTA,
                new_loss_coefficient=NEW_LOSS_COEFFICIENT,
                anchor_loss_coefficient=ANCHOR_LOSS_COEFFICIENT,
            )
        )
        if update not in checkpoint_updates:
            continue
        if not _residual_base_matches(parameters, inputs.initial_parameters):
            raise RuntimeError("residual optimizer modified the frozen v1 base")
        runtime_report = training.runtime_bytes(parameters)[1]
        selection = training.metrics(parameters, inputs.selection)
        coverage = _coverage(update, inputs)
        coverage_complete = bool(
            coverage["new"]["complete_coverage"]
            and coverage["anchor"]["complete_coverage"]
        )
        new_passed = new_gate_passes(
            V5_NONINFERIORITY,
            selection,
            epoch_zero=references["epoch_zero"]["selection"],
            reference_thresholds=thresholds["new_reference"],
        )
        retention = None
        retention_passed = None
        if coverage_complete and new_passed:
            retention = training.metrics(parameters, inputs.retention)
            retention_passed = _passes_thresholds(retention, thresholds["retention"])
            retention_status = "evaluated"
        elif not coverage_complete:
            retention_status = "not-evaluated-incomplete-coverage"
        else:
            retention_status = "not-evaluated-new-data-gate-failed"
        eligible = bool(coverage_complete and new_passed and retention_passed)
        row = {
            "update": update,
            "average_training_weighted_huber": float(np.mean(interval_losses)),
            "coverage": coverage,
            "runtime": runtime_report,
            "selection": selection,
            "new_gate": {
                "policy": V5_NONINFERIORITY,
                "passed": new_passed,
            },
            "retention_gate": {
                "status": retention_status,
                "metrics": retention,
                "passed": retention_passed,
            },
            "eligible": eligible,
        }
        interval_losses.clear()
        history.append(row)
        if eligible and (
            best_row is None
            or selection_key(selection, runtime_report["artifact_sha256"])
            < selection_key(
                best_row["selection"], best_row["runtime"]["artifact_sha256"]
            )
        ):
            best = {name: value.copy() for name, value in parameters.items()}
            best_row = row

    if best is None:
        selected = {name: value.copy() for name, value in zero_adapter.items()}
        selected_update = 0
        status = "no-eligible-checkpoint"
        selected_selection = references["epoch_zero"]["selection"]
        selected_retention = references["epoch_zero"]["retention"]
        eligible = False
    else:
        selected = best
        if best_row is None:
            raise RuntimeError("eligible residual candidate has no history row")
        selected_update = best_row["update"]
        status = "eligible-checkpoint-selected"
        selected_selection = best_row["selection"]
        selected_retention = best_row["retention_gate"]["metrics"]
        eligible = True
    if not _residual_base_matches(selected, inputs.initial_parameters):
        raise RuntimeError("selected residual runtime modified the frozen v1 base")
    selected_runtime = training.runtime_bytes(selected)[1]
    report = {
        "schema": RESIDUAL_RECOVERY_REPORT_SCHEMA,
        "configuration": configuration_report,
        "inputs": inputs.receipt_identity(),
        "producer": _producer_identity(),
        "references": references,
        "thresholds": thresholds,
        "checkpoints": history,
        "result": {
            "status": status,
            "selected_update": selected_update,
            "eligible": eligible,
            "runtime": selected_runtime,
            "selection": selected_selection,
            "retention": selected_retention,
            "frozen_layers": list(training.BASE_PARAMETER_NAMES),
            "frozen_layers_verified": True,
        },
    }
    return selected, report


def _train_v6_joint(
    inputs: RecoveryInputs, configuration: V6JointConfiguration
) -> tuple[dict[str, np.ndarray], dict]:
    configuration = configuration.normalized()
    configuration_report = _v6_joint_configuration_report(inputs, configuration)
    coverage_epoch = configuration_report["batching"][
        "anchor_coverage_complete_epoch"
    ]
    learning_rate = configuration_report["optimizer"]["learning_rate_policy"][
        "effective_learning_rate"
    ]
    references, thresholds = _reference_evidence(inputs)
    parameters = {
        name: np.asarray(value, dtype=np.float32).copy()
        for name, value in inputs.initial_parameters.items()
    }
    optimizer = training.AdamW(parameters, learning_rate, WEIGHT_DECAY)
    mixed = training.MixedTraining(
        inputs.new,
        inputs.anchor,
        NEW_ROWS_PER_BATCH,
        ANCHOR_ROWS_PER_BATCH,
    )
    epoch_zero_key = selection_key(references["epoch_zero"]["selection"])[:3]
    best_key: tuple[float, float, float, str] | None = None
    best: dict[str, np.ndarray] | None = None
    best_row: dict | None = None
    history: list[dict] = []
    last_progress_epoch = coverage_epoch
    for epoch in range(1, V6_JOINT_EPOCHS + 1):
        losses = []
        for new_rows, anchor_rows in training.mixed_epoch_batches(
            mixed,
            batch_size=BATCH_SIZE,
            seed=configuration.seed,
            epoch=epoch,
        ):
            losses.append(
                train_recovery_batch(
                    parameters,
                    optimizer,
                    mixed,
                    new_rows,
                    anchor_rows,
                    trainable_layers="all",
                )
            )
        runtime_report = training.runtime_bytes(parameters)[1]
        selection = training.metrics(parameters, inputs.selection)
        key = selection_key(selection)[:3]
        candidate_key = selection_key(
            selection, runtime_report["artifact_sha256"]
        )
        coverage = training.mixed_epoch_coverage(
            mixed, batch_size=BATCH_SIZE, epoch=epoch
        )
        coverage_complete = coverage["anchor"]["complete_permutations"] >= 1
        improves_epoch_zero = key < epoch_zero_key
        can_beat_best = bool(
            improves_epoch_zero
            and (best_key is None or candidate_key < best_key)
        )
        retention = None
        retention_passed = None
        if not coverage_complete:
            retention_status = "not-evaluated-incomplete-anchor-coverage"
        elif not improves_epoch_zero:
            retention_status = "not-evaluated-epoch-zero-not-improved"
        elif not can_beat_best:
            retention_status = "not-evaluated-cannot-beat-current-best"
        else:
            retention = training.metrics(parameters, inputs.retention)
            retention_passed = _passes_thresholds(retention, thresholds["retention"])
            retention_status = "evaluated"
        eligible = bool(
            coverage_complete
            and improves_epoch_zero
            and can_beat_best
            and retention_passed
        )
        row = {
            "epoch": epoch,
            "average_training_weighted_huber": float(np.mean(losses)),
            "coverage": coverage,
            "runtime": runtime_report,
            "selection": selection,
            "new_gate": {
                "policy": EPOCH_ZERO_IMPROVEMENT,
                "passed": improves_epoch_zero,
            },
            "selection_candidate": can_beat_best,
            "retention_gate": {
                "status": retention_status,
                "metrics": retention,
                "passed": retention_passed,
            },
            "eligible": eligible,
        }
        history.append(row)
        if eligible:
            best = {name: value.copy() for name, value in parameters.items()}
            best_row = row
            best_key = candidate_key
            last_progress_epoch = epoch
        if (
            coverage_complete
            and epoch - last_progress_epoch >= V6_JOINT_PATIENCE
        ):
            break

    if best is None:
        selected = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in inputs.initial_parameters.items()
        }
        selected_epoch = 0
        status = "no-eligible-checkpoint"
        selected_selection = references["epoch_zero"]["selection"]
        selected_retention = references["epoch_zero"]["retention"]
        eligible = False
    else:
        selected = best
        if best_row is None:
            raise RuntimeError("eligible v6 joint candidate has no history row")
        selected_epoch = best_row["epoch"]
        status = "eligible-checkpoint-selected"
        selected_selection = best_row["selection"]
        selected_retention = best_row["retention_gate"]["metrics"]
        eligible = True
    selected_runtime = training.runtime_bytes(selected)[1]
    report = {
        "schema": V6_JOINT_REPORT_SCHEMA,
        "configuration": configuration_report,
        "inputs": inputs.receipt_identity(),
        "producer": _producer_identity(),
        "references": references,
        "thresholds": {"retention": thresholds["retention"]},
        "checkpoints": history,
        "result": {
            "status": status,
            "selected_epoch": selected_epoch,
            "eligible": eligible,
            "epochs_completed": len(history),
            "runtime": selected_runtime,
            "selection": selected_selection,
            "retention": selected_retention,
            "anchor_coverage_complete_epoch": coverage_epoch,
        },
    }
    return selected, report


def _expected_checkpoint_updates(configuration_report: Mapping[str, object]) -> list[int]:
    maximum = int(configuration_report["batching"]["maximum_updates"])
    expected = list(range(CHECKPOINT_INTERVAL, maximum + 1, CHECKPOINT_INTERVAL))
    if not expected or expected[-1] != maximum:
        expected.append(maximum)
    return expected


def _runtime_report_is_valid(value: object, *, residual: bool) -> bool:
    fields = {
        "artifact_sha256",
        "payload_sha256",
        "feature_schema_sha256",
        "bytes",
        "weight_count",
    }
    if residual:
        fields.update(
            {
                "runtime_version",
                "residual_rank",
                "payload_layout",
                "base_payload_sha256",
                "base_gain",
                "residual_bias",
            }
        )
    if not isinstance(value, dict) or set(value) != fields:
        return False
    hashes = (
        value.get("artifact_sha256"),
        value.get("payload_sha256"),
        value.get("feature_schema_sha256"),
    )
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in hashes
    ):
        return False
    expected_weights = (
        training.RUNTIME_V2_WEIGHT_COUNT if residual else training.WEIGHT_COUNT
    )
    if (
        value.get("weight_count") != expected_weights
        or value.get("bytes") != training.RUNTIME_HEADER.size + expected_weights * 4
    ):
        return False
    return not residual or (
        value.get("runtime_version") == training.RUNTIME_V2_VERSION
        and value.get("residual_rank") == training.RESIDUAL_RANK
        and value.get("payload_layout") == training.RUNTIME_V2_PAYLOAD_LAYOUT
        and isinstance(value.get("base_payload_sha256"), str)
        and len(value["base_payload_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in value["base_payload_sha256"]
        )
        and not isinstance(value.get("base_gain"), bool)
        and isinstance(value.get("base_gain"), (int, float))
        and math.isfinite(float(value["base_gain"]))
        and not isinstance(value.get("residual_bias"), bool)
        and isinstance(value.get("residual_bias"), (int, float))
        and math.isfinite(float(value["residual_bias"]))
    )


def _validate_report_common(
    report: object,
    inputs: RecoveryInputs,
    selected: Mapping[str, np.ndarray],
    runtime_report: Mapping[str, object],
    *,
    expected_schema: str,
    expected_configuration: Mapping[str, object],
    references: Mapping[str, object],
    thresholds: Mapping[str, object],
    selection_policy: str,
    frozen_layers: Sequence[str],
    frozen_parameters_match: bool,
    fallback_runtime: Mapping[str, object],
    residual: bool,
) -> dict:
    expected_fields = {
        "schema",
        "configuration",
        "inputs",
        "producer",
        "references",
        "thresholds",
        "checkpoints",
        "result",
    }
    if not isinstance(report, dict) or set(report) != expected_fields:
        raise ValueError("recovery report shape is invalid")
    if (
        report["schema"] != expected_schema
        or report["configuration"] != expected_configuration
        or report["inputs"] != inputs.receipt_identity()
        or report["producer"] != _producer_identity()
    ):
        raise ValueError("recovery report binding is stale or corrupt")
    if report["references"] != references or report["thresholds"] != thresholds:
        raise ValueError("recovery report reference metrics are stale or corrupt")

    checkpoints = report["checkpoints"]
    expected_updates = _expected_checkpoint_updates(expected_configuration)
    if (
        not isinstance(checkpoints, list)
        or any(not isinstance(row, dict) for row in checkpoints)
        or [row["update"] if "update" in row else None for row in checkpoints]
        != expected_updates
    ):
        raise ValueError("recovery checkpoint schedule is stale or corrupt")
    eligible_rows = []
    for row in checkpoints:
        if not isinstance(row, dict) or set(row) != {
            "update",
            "average_training_weighted_huber",
            "coverage",
            "runtime",
            "selection",
            "new_gate",
            "retention_gate",
            "eligible",
        }:
            raise ValueError("recovery checkpoint row is invalid")
        update = row["update"]
        average_loss = row["average_training_weighted_huber"]
        if (
            isinstance(average_loss, bool)
            or not isinstance(average_loss, (int, float))
            or not math.isfinite(float(average_loss))
            or float(average_loss) < 0.0
            or row["coverage"] != _coverage(update, inputs)
            or not _metric_is_valid(row["selection"], len(inputs.selection))
            or not _runtime_report_is_valid(row["runtime"], residual=residual)
        ):
            raise ValueError("recovery checkpoint evidence is stale or corrupt")
        coverage_complete = bool(
            row["coverage"]["new"]["complete_coverage"]
            and row["coverage"]["anchor"]["complete_coverage"]
        )
        expected_new_pass = new_gate_passes(
            selection_policy,
            row["selection"],
            epoch_zero=references["epoch_zero"]["selection"],
            reference_thresholds=thresholds["new_reference"],
        )
        if row["new_gate"] != {
            "policy": selection_policy,
            "passed": expected_new_pass,
        }:
            raise ValueError("recovery new-data gate is stale or corrupt")
        retention_gate = row["retention_gate"]
        if not isinstance(retention_gate, dict) or set(retention_gate) != {
            "status",
            "metrics",
            "passed",
        }:
            raise ValueError("recovery retention gate is invalid")
        if coverage_complete and expected_new_pass:
            expected_status = "evaluated"
            if not _metric_is_valid(retention_gate.get("metrics"), len(inputs.retention)):
                raise ValueError("recovery retention metric is invalid")
            expected_retention_pass = _passes_thresholds(
                retention_gate["metrics"], thresholds["retention"]
            )
        else:
            expected_status = (
                "not-evaluated-incomplete-coverage"
                if not coverage_complete
                else "not-evaluated-new-data-gate-failed"
            )
            expected_retention_pass = None
            if retention_gate.get("metrics") is not None:
                raise ValueError("recovery retention metric was evaluated out of policy")
        expected_retention_gate = {
            "status": expected_status,
            "metrics": retention_gate.get("metrics"),
            "passed": expected_retention_pass,
        }
        expected_eligible = bool(
            coverage_complete and expected_new_pass and expected_retention_pass
        )
        if retention_gate != expected_retention_gate or row["eligible"] is not expected_eligible:
            raise ValueError("recovery eligibility evidence is stale or corrupt")
        if expected_eligible:
            eligible_rows.append(row)

    result = report["result"]
    if not isinstance(result, dict) or set(result) != {
        "status",
        "selected_update",
        "eligible",
        "runtime",
        "selection",
        "retention",
        "frozen_layers",
        "frozen_layers_verified",
    } or result["runtime"] != runtime_report:
        raise ValueError("recovery result is stale or corrupt")
    best_row = min(
        eligible_rows,
        key=lambda row: selection_key(
            row["selection"], row["runtime"]["artifact_sha256"]
        ),
        default=None,
    )
    if best_row is None:
        expected_result = {
            "status": "no-eligible-checkpoint",
            "selected_update": 0,
            "eligible": False,
            "runtime": fallback_runtime,
            "selection": references["epoch_zero"]["selection"],
            "retention": references["epoch_zero"]["retention"],
            "frozen_layers": list(frozen_layers),
            "frozen_layers_verified": True,
        }
    else:
        expected_result = {
            "status": "eligible-checkpoint-selected",
            "selected_update": best_row["update"],
            "eligible": True,
            "runtime": best_row["runtime"],
            "selection": best_row["selection"],
            "retention": best_row["retention_gate"]["metrics"],
            "frozen_layers": list(frozen_layers),
            "frozen_layers_verified": True,
        }
    if result != expected_result:
        raise ValueError("recovery deterministic selection is stale or corrupt")
    recomputed_selection = training.metrics(selected, inputs.selection)
    recomputed_retention = training.metrics(selected, inputs.retention)
    if (
        result["selection"] != recomputed_selection
        or result["retention"] != recomputed_retention
        or not frozen_parameters_match
    ):
        raise ValueError("recovery selected runtime recomputed metrics disagree")
    return report


def _validate_report(
    report: object,
    inputs: RecoveryInputs,
    configuration: RecoveryConfiguration,
    selected: Mapping[str, np.ndarray],
    runtime_report: Mapping[str, object],
) -> dict:
    configuration = configuration.normalized()
    references, thresholds = _reference_evidence(inputs)
    return _validate_report_common(
        report,
        inputs,
        selected,
        runtime_report,
        expected_schema=RECOVERY_REPORT_SCHEMA,
        expected_configuration=_configuration_report(inputs, configuration),
        references=references,
        thresholds=thresholds,
        selection_policy=configuration.selection_policy,
        frozen_layers=_frozen_layers(configuration),
        frozen_parameters_match=_frozen_layers_match(
            selected, inputs.initial_parameters, configuration
        ),
        fallback_runtime=inputs.receipt_identity()["runtimes"]["initial"],
        residual=False,
    )


def _validate_residual_report(
    report: object,
    inputs: RecoveryInputs,
    configuration: ResidualRecoveryConfiguration,
    selected: Mapping[str, np.ndarray],
    runtime_report: Mapping[str, object],
) -> dict:
    configuration = configuration.normalized()
    zero_adapter = training.initialize_residual_adapter(inputs.initial_parameters)
    references, thresholds = _residual_reference_evidence(inputs, zero_adapter)
    return _validate_report_common(
        report,
        inputs,
        selected,
        runtime_report,
        expected_schema=RESIDUAL_RECOVERY_REPORT_SCHEMA,
        expected_configuration=_residual_configuration_report(inputs, configuration),
        references=references,
        thresholds=thresholds,
        selection_policy=V5_NONINFERIORITY,
        frozen_layers=training.BASE_PARAMETER_NAMES,
        frozen_parameters_match=_residual_base_matches(
            selected, inputs.initial_parameters
        ),
        fallback_runtime=training.runtime_bytes(zero_adapter)[1],
        residual=True,
    )


def _require_full_replay_match(
    persisted_parameters: Mapping[str, np.ndarray],
    persisted_report: Mapping[str, object],
    replayed_parameters: Mapping[str, np.ndarray],
    replayed_report: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Reject any transcript or selected-byte difference after full replay."""

    if persisted_report != replayed_report:
        raise ValueError(f"{label} full replay transcript disagrees")
    persisted_bytes = training.runtime_bytes(persisted_parameters)[0]
    replayed_bytes = training.runtime_bytes(replayed_parameters)[0]
    if persisted_bytes != replayed_bytes:
        raise ValueError(f"{label} full replay selected runtime bytes disagree")


def _publish_result(
    output_directory: pathlib.Path,
    selected: Mapping[str, np.ndarray],
    report: Mapping[str, object],
    *,
    runtime_name: str,
    report_name: str,
    receipt_name: str,
    receipt_schema: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    runtime_path = output_directory / runtime_name
    report_path = output_directory / report_name
    receipt_path = output_directory / receipt_name
    runtime_payload, runtime_report = training.runtime_bytes(selected)
    if report["result"]["runtime"] != runtime_report:
        raise RuntimeError("recovery runtime identity changed before publication")
    report_payload = training.canonical_json_bytes(report)
    body = {
        "schema": receipt_schema,
        "configuration": report["configuration"],
        "inputs": report["inputs"],
        "producer": report["producer"],
        "runtime": {"file": runtime_name, **runtime_report},
        "report": {
            "file": report_name,
            "sha256": _sha256(report_payload),
            "bytes": len(report_payload),
        },
    }
    receipt = {**body, "body_sha256": _sha256(training.canonical_json_bytes(body))}
    training._write_once(runtime_path, runtime_payload)
    training._write_once(report_path, report_payload)
    training._write_once(receipt_path, training.canonical_json_bytes(receipt))


def _publish(
    output_directory: pathlib.Path,
    selected: Mapping[str, np.ndarray],
    report: Mapping[str, object],
) -> None:
    _publish_result(
        output_directory,
        selected,
        report,
        runtime_name=RUNTIME_NAME,
        report_name=REPORT_NAME,
        receipt_name=RECEIPT_NAME,
        receipt_schema=RECOVERY_RECEIPT_SCHEMA,
    )


def _publish_residual(
    output_directory: pathlib.Path,
    selected: Mapping[str, np.ndarray],
    report: Mapping[str, object],
) -> None:
    _publish_result(
        output_directory,
        selected,
        report,
        runtime_name=RESIDUAL_RUNTIME_NAME,
        report_name=RESIDUAL_REPORT_NAME,
        receipt_name=RESIDUAL_RECEIPT_NAME,
        receipt_schema=RESIDUAL_RECOVERY_RECEIPT_SCHEMA,
    )


def _publish_v6_joint(
    output_directory: pathlib.Path,
    selected: Mapping[str, np.ndarray],
    report: Mapping[str, object],
) -> None:
    _publish_result(
        output_directory,
        selected,
        report,
        runtime_name=V6_JOINT_RUNTIME_NAME,
        report_name=V6_JOINT_REPORT_NAME,
        receipt_name=V6_JOINT_RECEIPT_NAME,
        receipt_schema=V6_JOINT_RECEIPT_SCHEMA,
    )


def load_recovery_result(
    inputs: RecoveryInputs,
    configuration: RecoveryConfiguration,
    output_directory: pathlib.Path | str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate a completed arm by replaying its full training schedule.

    No intermediate model matrices are persisted.  Consequently strict resume
    costs approximately one complete arm run, but every metric, checkpoint
    runtime hash, interval loss, decision, and selected byte is recomputed.
    """

    output_directory = pathlib.Path(output_directory).resolve()
    runtime_path = output_directory / RUNTIME_NAME
    report_path = output_directory / REPORT_NAME
    receipt_path = output_directory / RECEIPT_NAME
    try:
        report_payload = report_path.read_bytes()
        receipt_payload = receipt_path.read_bytes()
        report = json.loads(report_payload)
        receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("recovery output is incomplete or invalid") from error
    if report_payload != training.canonical_json_bytes(report):
        raise ValueError("recovery report is not canonical")
    if receipt_payload != training.canonical_json_bytes(receipt):
        raise ValueError("recovery receipt is not canonical")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema") != RECOVERY_RECEIPT_SCHEMA
    ):
        raise ValueError("recovery receipt schema is invalid")
    body = dict(receipt)
    body_sha256 = body.pop("body_sha256", None)
    if body_sha256 != _sha256(training.canonical_json_bytes(body)):
        raise ValueError("recovery receipt integrity failed")
    expected_configuration = _configuration_report(inputs, configuration)
    expected_inputs = inputs.receipt_identity()
    expected_producer = _producer_identity()
    if (
        receipt.get("configuration") != expected_configuration
        or receipt.get("inputs") != expected_inputs
        or receipt.get("producer") != expected_producer
    ):
        raise ValueError("recovery receipt schedule, references, or inputs are stale")
    expected_report = {
        "file": REPORT_NAME,
        "sha256": _sha256(report_payload),
        "bytes": len(report_payload),
    }
    if receipt.get("report") != expected_report:
        raise ValueError("recovery report identity is stale or corrupt")
    try:
        selected, runtime_report = training.load_runtime(runtime_path)
    except (OSError, ValueError) as error:
        raise ValueError("recovery selected runtime is invalid") from error
    if receipt.get("runtime") != {"file": RUNTIME_NAME, **runtime_report}:
        raise ValueError("recovery runtime identity is stale or corrupt")
    validated = _validate_report(
        report, inputs, configuration.normalized(), selected, runtime_report
    )
    replayed, replayed_report = _train(inputs, configuration.normalized())
    _require_full_replay_match(
        selected,
        validated,
        replayed,
        replayed_report,
        label="recovery",
    )
    return selected, validated


def load_residual_recovery_result(
    inputs: RecoveryInputs,
    configuration: ResidualRecoveryConfiguration,
    output_directory: pathlib.Path | str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate a residual arm by replaying its full adapter schedule.

    This intentionally costs approximately one complete residual arm run so
    all checkpoint evidence can be recomputed without persisting every model.
    """

    configuration = configuration.normalized()
    output_directory = pathlib.Path(output_directory).resolve()
    runtime_path = output_directory / RESIDUAL_RUNTIME_NAME
    report_path = output_directory / RESIDUAL_REPORT_NAME
    receipt_path = output_directory / RESIDUAL_RECEIPT_NAME
    try:
        report_payload = report_path.read_bytes()
        receipt_payload = receipt_path.read_bytes()
        report = json.loads(report_payload)
        receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("residual recovery output is incomplete or invalid") from error
    if report_payload != training.canonical_json_bytes(report):
        raise ValueError("residual recovery report is not canonical")
    if receipt_payload != training.canonical_json_bytes(receipt):
        raise ValueError("residual recovery receipt is not canonical")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema") != RESIDUAL_RECOVERY_RECEIPT_SCHEMA
    ):
        raise ValueError("residual recovery receipt schema is invalid")
    body = dict(receipt)
    body_sha256 = body.pop("body_sha256", None)
    if body_sha256 != _sha256(training.canonical_json_bytes(body)):
        raise ValueError("residual recovery receipt integrity failed")
    expected_configuration = _residual_configuration_report(inputs, configuration)
    if (
        receipt.get("configuration") != expected_configuration
        or receipt.get("inputs") != inputs.receipt_identity()
        or receipt.get("producer") != _producer_identity()
    ):
        raise ValueError(
            "residual recovery receipt schedule, references, or inputs are stale"
        )
    expected_report = {
        "file": RESIDUAL_REPORT_NAME,
        "sha256": _sha256(report_payload),
        "bytes": len(report_payload),
    }
    if receipt.get("report") != expected_report:
        raise ValueError("residual recovery report identity is stale or corrupt")
    try:
        selected, runtime_report = training.load_runtime(runtime_path)
    except (OSError, ValueError) as error:
        raise ValueError("residual recovery selected runtime is invalid") from error
    if (
        set(selected)
        != set((*training.BASE_PARAMETER_NAMES, *training.RESIDUAL_PARAMETER_NAMES))
        or receipt.get("runtime")
        != {"file": RESIDUAL_RUNTIME_NAME, **runtime_report}
    ):
        raise ValueError("residual recovery runtime identity is stale or corrupt")
    validated = _validate_residual_report(
        report, inputs, configuration, selected, runtime_report
    )
    replayed, replayed_report = _train_residual(inputs, configuration)
    _require_full_replay_match(
        selected,
        validated,
        replayed,
        replayed_report,
        label="residual recovery",
    )
    return selected, validated


def load_v6_joint_result(
    inputs: RecoveryInputs,
    configuration: V6JointConfiguration,
    output_directory: pathlib.Path | str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate a v6 joint arm through a complete deterministic replay."""

    configuration = configuration.normalized()
    output_directory = pathlib.Path(output_directory).resolve()
    runtime_path = output_directory / V6_JOINT_RUNTIME_NAME
    report_path = output_directory / V6_JOINT_REPORT_NAME
    receipt_path = output_directory / V6_JOINT_RECEIPT_NAME
    try:
        report_payload = report_path.read_bytes()
        receipt_payload = receipt_path.read_bytes()
        report = json.loads(report_payload)
        receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v6 joint output is incomplete or invalid") from error
    if report_payload != training.canonical_json_bytes(report):
        raise ValueError("v6 joint report is not canonical")
    if receipt_payload != training.canonical_json_bytes(receipt):
        raise ValueError("v6 joint receipt is not canonical")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema") != V6_JOINT_RECEIPT_SCHEMA
    ):
        raise ValueError("v6 joint receipt schema is invalid")
    body = dict(receipt)
    body_sha256 = body.pop("body_sha256", None)
    if body_sha256 != _sha256(training.canonical_json_bytes(body)):
        raise ValueError("v6 joint receipt integrity failed")
    expected_configuration = _v6_joint_configuration_report(inputs, configuration)
    if (
        receipt.get("configuration") != expected_configuration
        or receipt.get("inputs") != inputs.receipt_identity()
        or receipt.get("producer") != _producer_identity()
    ):
        raise ValueError("v6 joint receipt schedule, references, or inputs are stale")
    expected_report = {
        "file": V6_JOINT_REPORT_NAME,
        "sha256": _sha256(report_payload),
        "bytes": len(report_payload),
    }
    if receipt.get("report") != expected_report:
        raise ValueError("v6 joint report identity is stale or corrupt")
    try:
        selected, runtime_report = training.load_runtime(runtime_path)
    except (OSError, ValueError) as error:
        raise ValueError("v6 joint selected runtime is invalid") from error
    if (
        set(selected) != set(training.BASE_PARAMETER_NAMES)
        or receipt.get("runtime")
        != {"file": V6_JOINT_RUNTIME_NAME, **runtime_report}
    ):
        raise ValueError("v6 joint runtime identity is stale or corrupt")
    replayed, replayed_report = _train_v6_joint(inputs, configuration)
    _require_full_replay_match(
        selected,
        report,
        replayed,
        replayed_report,
        label="v6 joint",
    )
    return selected, report


def run_recovery(
    inputs: RecoveryInputs,
    configuration: RecoveryConfiguration,
    output_directory: pathlib.Path | str,
    *,
    resume: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train or strictly resume one recovery arm."""

    if not isinstance(inputs, RecoveryInputs):
        raise TypeError("inputs must be prepared by prepare_recovery_inputs")
    configuration = configuration.normalized()
    output_directory = pathlib.Path(output_directory).resolve()
    paths = tuple(
        output_directory / name for name in (RUNTIME_NAME, REPORT_NAME, RECEIPT_NAME)
    )
    existing = [path.exists() for path in paths]
    if any(existing):
        if not all(existing):
            if not resume:
                raise ValueError(
                    "recovery output is partial; pass resume=True for deterministic "
                    "reconstruction"
                )
            selected, report = _train(inputs, configuration)
            _publish(output_directory, selected, report)
            return selected, report
        if not resume:
            raise ValueError("recovery output exists; pass resume=True to validate it")
        return load_recovery_result(inputs, configuration, output_directory)
    selected, report = _train(inputs, configuration)
    _publish(output_directory, selected, report)
    return selected, report


def run_residual_recovery(
    inputs: RecoveryInputs,
    configuration: ResidualRecoveryConfiguration,
    output_directory: pathlib.Path | str,
    *,
    resume: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train or strictly resume one rank-16 adapter-only fallback arm."""

    if not isinstance(inputs, RecoveryInputs):
        raise TypeError("inputs must be prepared by prepare_recovery_inputs")
    configuration = configuration.normalized()
    output_directory = pathlib.Path(output_directory).resolve()
    paths = tuple(
        output_directory / name
        for name in (
            RESIDUAL_RUNTIME_NAME,
            RESIDUAL_REPORT_NAME,
            RESIDUAL_RECEIPT_NAME,
        )
    )
    existing = [path.exists() for path in paths]
    if any(existing):
        if not all(existing):
            if not resume:
                raise ValueError(
                    "residual recovery output is partial; pass resume=True for "
                    "deterministic reconstruction"
                )
            selected, report = _train_residual(inputs, configuration)
            _publish_residual(output_directory, selected, report)
            return selected, report
        if not resume:
            raise ValueError(
                "residual recovery output exists; pass resume=True to validate it"
            )
        return load_residual_recovery_result(inputs, configuration, output_directory)
    selected, report = _train_residual(inputs, configuration)
    _publish_residual(output_directory, selected, report)
    return selected, report


def run_v6_joint(
    inputs: RecoveryInputs,
    configuration: V6JointConfiguration,
    output_directory: pathlib.Path | str,
    *,
    resume: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train or fully replay-resume one exact v6 joint-recipe arm."""

    if not isinstance(inputs, RecoveryInputs):
        raise TypeError("inputs must be prepared by prepare_recovery_inputs")
    configuration = configuration.normalized()
    output_directory = pathlib.Path(output_directory).resolve()
    paths = tuple(
        output_directory / name
        for name in (
            V6_JOINT_RUNTIME_NAME,
            V6_JOINT_REPORT_NAME,
            V6_JOINT_RECEIPT_NAME,
        )
    )
    existing = [path.exists() for path in paths]
    if any(existing):
        if not all(existing):
            if not resume:
                raise ValueError(
                    "v6 joint output is partial; pass resume=True for "
                    "deterministic reconstruction"
                )
            selected, report = _train_v6_joint(inputs, configuration)
            _publish_v6_joint(output_directory, selected, report)
            return selected, report
        if not resume:
            raise ValueError(
                "v6 joint output exists; pass resume=True to validate it"
            )
        return load_v6_joint_result(inputs, configuration, output_directory)
    selected, report = _train_v6_joint(inputs, configuration)
    _publish_v6_joint(output_directory, selected, report)
    return selected, report


def _parse_uint64(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= parsed < 1 << 64:
        raise argparse.ArgumentTypeError("seed must fit uint64")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-runtime", type=pathlib.Path, required=True)
    parser.add_argument(
        "--retention-reference-runtime", type=pathlib.Path, required=True
    )
    parser.add_argument("--new-reference-runtime", type=pathlib.Path, required=True)
    parser.add_argument(
        "--new-manifest", "--new-shard-manifest", action="append", type=pathlib.Path,
        required=True, dest="new_manifests",
    )
    parser.add_argument(
        "--anchor-manifest", "--anchor-shard-manifest", action="append",
        type=pathlib.Path, required=True, dest="anchor_manifests",
    )
    parser.add_argument(
        "--selection-manifest", "--selection-validation-manifest", action="append",
        type=pathlib.Path, required=True, dest="selection_manifests",
    )
    parser.add_argument(
        "--retention-manifest", "--retention-validation-manifest", action="append",
        type=pathlib.Path, required=True, dest="retention_manifests",
    )
    parser.add_argument(
        "--selection-policy", choices=SELECTION_POLICIES
    )
    parser.add_argument(
        "--trainable-layers", choices=("w3", "w2-w3", "w2+w3", "all")
    )
    parser.add_argument(
        "--residual-adapter",
        action="store_true",
        help="run the separate v2 rank-16 adapter-only fallback",
    )
    parser.add_argument(
        "--v6-joint",
        action="store_true",
        help="run the fixed epoch-based v6 joint recipe",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=_parse_uint64, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.residual_adapter and arguments.v6_joint:
        parser.error("choose only one of --residual-adapter or --v6-joint")
    if arguments.v6_joint:
        if arguments.trainable_layers is not None or arguments.learning_rate is not None:
            parser.error("--v6-joint does not accept layer or learning-rate overrides")
        if arguments.selection_policy not in {None, EPOCH_ZERO_IMPROVEMENT}:
            parser.error("v6 joint training uses epoch-zero-improvement")
    elif arguments.residual_adapter:
        if arguments.trainable_layers is not None:
            parser.error("--residual-adapter does not accept --trainable-layers")
        if arguments.learning_rate is None:
            parser.error("--residual-adapter requires --learning-rate")
        if arguments.selection_policy not in {None, V5_NONINFERIORITY}:
            parser.error("residual recovery uses v5-recovery-noninferiority")
    else:
        if arguments.trainable_layers is None or arguments.learning_rate is None:
            parser.error("v1 recovery requires --trainable-layers and --learning-rate")
    selection_policy = arguments.selection_policy or V5_NONINFERIORITY
    try:
        inputs = prepare_recovery_inputs(
            initial_runtime=arguments.initial_runtime,
            retention_reference_runtime=arguments.retention_reference_runtime,
            new_reference_runtime=arguments.new_reference_runtime,
            new_manifests=arguments.new_manifests,
            anchor_manifests=arguments.anchor_manifests,
            selection_manifests=arguments.selection_manifests,
            retention_manifests=arguments.retention_manifests,
        )
        if arguments.v6_joint:
            _, report = run_v6_joint(
                inputs,
                V6JointConfiguration(arguments.seed),
                arguments.output_directory,
                resume=arguments.resume,
            )
        elif arguments.residual_adapter:
            _, report = run_residual_recovery(
                inputs,
                ResidualRecoveryConfiguration(
                    arguments.learning_rate,
                    arguments.seed,
                ),
                arguments.output_directory,
                resume=arguments.resume,
            )
        else:
            _, report = run_recovery(
                inputs,
                RecoveryConfiguration(
                    arguments.trainable_layers,
                    arguments.learning_rate,
                    arguments.seed,
                    selection_policy,
                ),
                arguments.output_directory,
                resume=arguments.resume,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report["result"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
