#!/usr/bin/env python3
"""Train and publish the standalone compact value-only BFM family.

The module deliberately does not import the large-teacher campaign wrapper.
It consumes only a frozen ``compact-value-bfm`` input bundle, keeps every test
route closed until a separately validated immutable selection is supplied,
and publishes content-addressed float checkpoints, signed-three-bit runtimes,
and body-hashed receipts.

The training implementation is small enough to audit directly.  All models
are bias-free sparse networks with the deployment activation/order contract::

    6301 -> H1 -> H2 -> 1
    square/leaky-0.01 -> leaky-ReLU-0.01 -> fast-tanh-rational-v1

No policy target or policy head is accepted anywhere in this file.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import struct
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

# Numerical runtimes commonly snapshot these values when NumPy is imported.
# Direct CLI launches therefore re-exec once with the exact one-thread contract
# instead of changing process-global settings after worker threads exist.
NATIVE_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
NATIVE_THREAD_PREIMPORT_MARKER = (
    "PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"
)


def _reexec_cli_with_native_thread_contract() -> None:
    if __name__ != "__main__" or (
        os.environ.get(NATIVE_THREAD_PREIMPORT_MARKER) == "1"
        and all(
            os.environ.get(name) == value
            for name, value in NATIVE_THREAD_ENVIRONMENT.items()
        )
    ):
        return
    environment = dict(os.environ)
    environment.update(NATIVE_THREAD_ENVIRONMENT)
    environment[NATIVE_THREAD_PREIMPORT_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


_reexec_cli_with_native_thread_contract()
NATIVE_THREAD_ENVIRONMENT_AT_NUMPY_IMPORT = {
    name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT
}
NATIVE_THREAD_PREIMPORT_MARKER_AT_NUMPY_IMPORT = os.environ.get(
    NATIVE_THREAD_PREIMPORT_MARKER
)

import numpy as np


TOOL_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOL_DIRECTORY))


CAMPAIGN_ID = "compact-value-bfm-20260831-v1"
BUNDLE_SCHEMA = "papersoccer.compact-value-bfm-input-bundle.v1"
SHARD_SCHEMA = "papersoccer.jacek-replay-csr-shard.v1"
SIDECAR_SCHEMA = "papersoccer.compact-value-bfm-teacher-sidecar.v1"
SIDECAR_INDEX_SCHEMA = "papersoccer.compact-value-bfm-sidecar-index.v1"
SUCCESSOR_LABEL_SCHEMA = (
    "papersoccer.compact-value-bfm-complete-turn-successor-labels.v1"
)
SUCCESSOR_STORE_SCHEMA = "papersoccer.compact-value-bfm-ranking-store.v2"
INPUT_AUDIT_SCHEMA = "papersoccer.compact-value-bfm-input-audit.v1"
RUNTIME_SCHEMA = "papersoccer.compact-value-bfm-runtime.v1"
SEED_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm-seed-receipt.v1"
SEED_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm-seed-reference.v1"
SELECTION_SCHEMA = "papersoccer.compact-value-bfm-selection.v1"
PROTECTED_REPORT_SCHEMA = "papersoccer.compact-value-bfm-protected-report.v1"

FEATURE_SCHEMA = (
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree"
)
INPUT_COUNT = 6_301
EDGE_COUNT = 316
VERTEX_COUNT = 105
VERTEX_CATEGORIES = 57
OUTPUT_COUNT = 1
LEAKY_SLOPE = np.float32(0.01)
HUBER_DELTA = np.float32(0.25)

FIXED_SEEDS = (20260907, 20260908, 20260909)
BATCH_SIZE = 256
NEW_ROWS_PER_BATCH = 64
ANCHOR_ROWS_PER_BATCH = 192
MAX_FLOAT_EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 5.0
QAT_EPOCHS = 4
QAT_LEARNING_RATE = 0.00025
RANKING_LOSS_WEIGHTS = (0.0, 0.10, 0.25)
RANKING_PAIR_CAP = 8
HARD_TEACHER_RANKING_PROFILE = "hardest-5pct-2m-v1"
HARD_STATE_DENSITY_MULTIPLIER = 8
RANKING_FLOAT_EPOCHS = 1
RANKING_FLOAT_LEARNING_RATE = 0.00006

QUANTIZATION_BITS = 3
QUANTIZATION_MINIMUM = -3
QUANTIZATION_MAXIMUM = 3
PACKING = "signed-three-bit-twos-complement-lsb-first"
PAYLOAD_LAYOUT = "w1-input-major,w2-input-major,w3"
ACTIVATIONS = (
    "square-leaky-0.01",
    "leaky-relu-0.01",
    "fast-tanh-rational-v1",
)

ROBUST_SCALE_QUANTILES = (
    ("p800", 800, 1_000),
    ("p900", 900, 1_000),
    ("p950", 950, 1_000),
    ("p975", 975, 1_000),
    ("p990", 990, 1_000),
    ("p995", 995, 1_000),
)
SCALE_SEARCH_PASSES = 2

QAT_PROFILE_SCHEMA = "papersoccer.compact-value-bfm-qat-profile.v1"
NATIVE_THREAD_EXECUTION_SCHEMA = (
    "papersoccer.compact-value-bfm-native-thread-execution.v1"
)
STANDARD_QAT_PROFILE = "standard-v1"
REFINED_ADAPTIVE_SCALES_QAT_PROFILE = "refined-adaptive-scales-v1"


@dataclasses.dataclass(frozen=True)
class QATProfile:
    """A closed, receipt-bindable fake-quantization/scale-search recipe."""

    name: str
    scale_quantiles: tuple[tuple[str, int, int], ...]
    coordinate_search_passes: int
    local_refinement_multipliers: tuple[tuple[str, int, int], ...]
    local_refinement_passes: int
    adapt_scales_after_each_epoch: bool
    adaptive_quantile_names: tuple[str, ...]
    adaptive_coordinate_passes: int


REFINED_SCALE_QUANTILES = (
    ("p700", 700, 1_000),
    ("p750", 750, 1_000),
    ("p800", 800, 1_000),
    ("p850", 850, 1_000),
    ("p875", 875, 1_000),
    ("p900", 900, 1_000),
    ("p925", 925, 1_000),
    ("p950", 950, 1_000),
    ("p965", 965, 1_000),
    ("p975", 975, 1_000),
    ("p985", 985, 1_000),
    ("p990", 990, 1_000),
    ("p995", 995, 1_000),
    ("p998", 998, 1_000),
)
REFINED_SCALE_MULTIPLIERS = (
    ("m900", 900, 1_000),
    ("m950", 950, 1_000),
    ("m1000", 1_000, 1_000),
    ("m1050", 1_050, 1_000),
    ("m1100", 1_100, 1_000),
)

QAT_PROFILES = {
    STANDARD_QAT_PROFILE: QATProfile(
        name=STANDARD_QAT_PROFILE,
        scale_quantiles=ROBUST_SCALE_QUANTILES,
        coordinate_search_passes=SCALE_SEARCH_PASSES,
        local_refinement_multipliers=(),
        local_refinement_passes=0,
        adapt_scales_after_each_epoch=False,
        adaptive_quantile_names=(),
        adaptive_coordinate_passes=0,
    ),
    REFINED_ADAPTIVE_SCALES_QAT_PROFILE: QATProfile(
        name=REFINED_ADAPTIVE_SCALES_QAT_PROFILE,
        scale_quantiles=REFINED_SCALE_QUANTILES,
        coordinate_search_passes=3,
        local_refinement_multipliers=REFINED_SCALE_MULTIPLIERS,
        local_refinement_passes=1,
        adapt_scales_after_each_epoch=True,
        adaptive_quantile_names=("p900", "p975", "p995", "p998"),
        adaptive_coordinate_passes=1,
    ),
}

COMMON_MINIMUM_SIGN = 0.8475
COMMON_MAXIMUM_HUBER = 0.0560
CANONICAL_MINIMUM_SIGN = 0.8613
CANONICAL_MAXIMUM_HUBER = 0.0551
MAXIMUM_SIGN_LOSS = 0.005
MAXIMUM_HUBER_RATIO = 1.02
CAPACITY_SOURCE_LIMIT = 95_000

FORBIDDEN_PATH_MARKERS = (
    "sealed-final",
    "sealed_final",
    "blind-label",
    "blind_label",
)


class TrainingError(ValueError):
    """A frozen input, model artifact, or training receipt is invalid."""


def _native_thread_controllers(values: object) -> list[dict[str, object]]:
    if not isinstance(values, list):
        raise TrainingError("native thread-controller inventory is malformed")
    controllers = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TrainingError("native thread-controller entry is malformed")
        threads = value.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
            raise TrainingError("native numerical runtime is not limited to one thread")
        controllers.append({
            "user_api": value.get("user_api"),
            "internal_api": value.get("internal_api"),
            "prefix": value.get("prefix"),
            "version": value.get("version"),
            "num_threads": threads,
        })
    return sorted(
        controllers,
        key=lambda item: tuple(str(item[name]) for name in (
            "user_api", "internal_api", "prefix", "version", "num_threads"
        )),
    )


def validate_native_thread_execution(value: object) -> dict[str, object]:
    expected_environment = dict(NATIVE_THREAD_ENVIRONMENT)
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema", "native_threads_per_seed_maximum",
            "environment_required", "environment_at_numpy_import",
            "environment_at_worker_launch", "environment_precedes_numpy_import",
            "preimport_bootstrap_marker",
            "limiter_scope", "threadpoolctl_available", "threadpoolctl_version",
            "threadpool_controllers",
        }
        or value.get("schema") != NATIVE_THREAD_EXECUTION_SCHEMA
        or value.get("native_threads_per_seed_maximum") != 1
        or value.get("environment_required") != expected_environment
        or value.get("environment_at_numpy_import") != expected_environment
        or value.get("environment_at_worker_launch") != expected_environment
        or value.get("environment_precedes_numpy_import") is not True
        or value.get("preimport_bootstrap_marker") != "1"
        or value.get("limiter_scope")
        != "outer-roster-established-before-seed-workers"
        or not isinstance(value.get("threadpoolctl_available"), bool)
        or (
            value.get("threadpoolctl_available") is True
            and not isinstance(value.get("threadpoolctl_version"), str)
        )
        or (
            value.get("threadpoolctl_available") is False
            and value.get("threadpoolctl_version") is not None
        )
    ):
        raise TrainingError("native one-thread execution evidence is malformed")
    controllers = _native_thread_controllers(value.get("threadpool_controllers"))
    if controllers != value.get("threadpool_controllers"):
        raise TrainingError("native thread-controller evidence is not canonical")
    return dict(value)


@contextlib.contextmanager
def native_thread_execution_scope():
    """Limit native kernels once outside the concurrent seed worker pool."""

    expected = dict(NATIVE_THREAD_ENVIRONMENT)
    imported = dict(NATIVE_THREAD_ENVIRONMENT_AT_NUMPY_IMPORT)
    current = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT}
    marker = os.environ.get(NATIVE_THREAD_PREIMPORT_MARKER)
    if (
        imported != expected
        or current != expected
        or NATIVE_THREAD_PREIMPORT_MARKER_AT_NUMPY_IMPORT != "1"
        or marker != "1"
    ):
        raise TrainingError(
            "training requires all BLAS/OpenMP limits to equal one before NumPy import; "
            "launch the trainer CLI so it can re-exec with the frozen environment"
        )
    try:
        import threadpoolctl  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        evidence = {
            "schema": NATIVE_THREAD_EXECUTION_SCHEMA,
            "native_threads_per_seed_maximum": 1,
            "environment_required": expected,
            "environment_at_numpy_import": imported,
            "environment_at_worker_launch": current,
            "environment_precedes_numpy_import": True,
            "preimport_bootstrap_marker": marker,
            "limiter_scope": "outer-roster-established-before-seed-workers",
            "threadpoolctl_available": False,
            "threadpoolctl_version": None,
            "threadpool_controllers": [],
        }
        yield validate_native_thread_execution(evidence)
        return
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            controllers = _native_thread_controllers(threadpoolctl.threadpool_info())
            evidence = {
                "schema": NATIVE_THREAD_EXECUTION_SCHEMA,
                "native_threads_per_seed_maximum": 1,
                "environment_required": expected,
                "environment_at_numpy_import": imported,
                "environment_at_worker_launch": current,
                "environment_precedes_numpy_import": True,
                "preimport_bootstrap_marker": marker,
                "limiter_scope": "outer-roster-established-before-seed-workers",
                "threadpoolctl_available": True,
                "threadpoolctl_version": str(threadpoolctl.__version__),
                "threadpool_controllers": controllers,
            }
            yield validate_native_thread_execution(evidence)
    except TrainingError:
        raise
    except Exception as error:
        raise TrainingError("threadpoolctl could not enforce the one-thread limit") from error


@dataclasses.dataclass(frozen=True)
class Architecture:
    name: str
    hidden_one: int
    hidden_two: int
    deployment_class: str

    @property
    def dimensions(self) -> tuple[int, int, int, int]:
        return (INPUT_COUNT, self.hidden_one, self.hidden_two, OUTPUT_COUNT)

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "w1": (INPUT_COUNT, self.hidden_one),
            "w2": (self.hidden_one, self.hidden_two),
            "w3": (self.hidden_two,),
        }

    @property
    def weight_counts(self) -> dict[str, int]:
        counts = {
            name: math.prod(shape) for name, shape in self.shapes.items()
        }
        return {**counts, "total": sum(counts.values())}


ARCHITECTURES = {
    "compact-8x8": Architecture(
        "compact-8x8", 8, 8, "primary"
    ),
    "source-neutral-8x16": Architecture(
        "source-neutral-8x16", 8, 16, "source-neutral-fallback"
    ),
    "capacity-12x8": Architecture(
        "capacity-12x8", 12, 8, "capacity-source-size-conditional"
    ),
}


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    new_source: str
    teacher_assisted: bool
    deployment_eligible: bool


ARMS = {
    "search-target": Arm("search-target", "search", False, True),
    "teacher-assisted": Arm("teacher-assisted", "search", True, True),
    "rank4-control": Arm("rank4-control", "rank4", False, False),
}


def architecture_deployment_eligible(
    architecture: Architecture | str,
    generated_source_ascii_bytes: int | None = None,
) -> bool:
    """Apply the capacity fallback's hard generated-source eligibility rule."""

    if isinstance(architecture, str):
        try:
            architecture = ARCHITECTURES[architecture]
        except KeyError as error:
            raise TrainingError("unknown compact architecture") from error
    if architecture.deployment_class != "capacity-source-size-conditional":
        return True
    return bool(
        type(generated_source_ascii_bytes) is int
        and 0 < generated_source_ascii_bytes <= CAPACITY_SOURCE_LIMIT
    )


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def body_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def verify_body_hash(
    value: Mapping[str, object], *, schema: str, label: str
) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or not isinstance(claimed, str)
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise TrainingError(f"{label} body SHA-256 is invalid")


def resolve_qat_profile(value: str | QATProfile) -> QATProfile:
    """Return only a canonical registered profile; copied variants fail closed."""

    if isinstance(value, QATProfile):
        registered = QAT_PROFILES.get(value.name)
        if registered != value:
            raise TrainingError("QAT profile differs from its registered definition")
        return registered
    if not isinstance(value, str) or value not in QAT_PROFILES:
        raise TrainingError(
            "QAT profile must be standard-v1 or refined-adaptive-scales-v1"
        )
    return QAT_PROFILES[value]


def qat_profile_contract(value: str | QATProfile) -> dict[str, object]:
    """Build the exact body-hashed recipe sealed into plans and receipts."""

    profile = resolve_qat_profile(value)
    return body_hashed({
        "schema": QAT_PROFILE_SCHEMA,
        "qat_profile": profile.name,
        "quantization": {
            "bits": QUANTIZATION_BITS,
            "minimum": QUANTIZATION_MINIMUM,
            "maximum": QUANTIZATION_MAXIMUM,
            "scheme": "symmetric-signed-three-bit-per-layer-fixed-scale",
            "fake_quantized_layers": ["w1", "w2", "w3"],
            "straight_through_master_weights": True,
        },
        "schedule": {
            "float_warmup_epochs": RANKING_FLOAT_EPOCHS,
            "qat_epochs": QAT_EPOCHS,
            "qat_learning_rate": QAT_LEARNING_RATE,
            "all_layers_trainable_each_qat_epoch": True,
        },
        "scale_selection": {
            "lower_rank_quantiles": [
                {
                    "name": name,
                    "numerator": numerator,
                    "denominator": denominator,
                }
                for name, numerator, denominator in profile.scale_quantiles
            ],
            "coordinate_search_passes": profile.coordinate_search_passes,
            "local_refinement_multipliers": [
                {
                    "name": name,
                    "numerator": numerator,
                    "denominator": denominator,
                }
                for name, numerator, denominator
                in profile.local_refinement_multipliers
            ],
            "local_refinement_passes": profile.local_refinement_passes,
            "adapt_scales_after_each_qat_epoch": (
                profile.adapt_scales_after_each_epoch
            ),
            "adaptive_quantile_names": list(profile.adaptive_quantile_names),
            "adaptive_coordinate_passes": (
                profile.adaptive_coordinate_passes
            ),
            "validation_objective": (
                "existing-validation-key-then-lower-scale"
                if profile.name == STANDARD_QAT_PROFILE
                else (
                    "float-quantized-action-flip-then-teacher-regret-then-"
                    "existing-validation-key-then-lower-scale"
                )
            ),
        },
    })


def validate_qat_profile_contract(
    value: object, *, expected_name: str | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingError("QAT profile contract is absent")
    name = value.get("qat_profile")
    profile = resolve_qat_profile(name if isinstance(name, str) else "")
    if expected_name is not None and profile.name != expected_name:
        raise TrainingError("QAT profile contract names another profile")
    expected = qat_profile_contract(profile)
    if dict(value) != expected:
        raise TrainingError("QAT profile contract differs from the registry")
    return expected


def _reject_path_markers(raw: os.PathLike[str] | str, label: str) -> None:
    text = os.fspath(raw)
    if any(marker in text.lower() for marker in FORBIDDEN_PATH_MARKERS):
        raise TrainingError(f"{label} contains a protected path marker")


def _safe_relative(raw: object, label: str) -> str:
    if not isinstance(raw, str):
        raise TrainingError(f"{label} is not a relative path")
    _reject_path_markers(raw, label)
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise TrainingError(f"{label} is unsafe")
    return relative.as_posix()


def _atomic_write_once(path: pathlib.Path, payload: bytes) -> None:
    """Create an immutable artifact; an existing unequal file is an error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise TrainingError(f"immutable artifact conflicts: {path}")
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
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise TrainingError(f"immutable artifact raced: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_content_addressed(
    directory: pathlib.Path, payload: bytes, suffix: str
) -> pathlib.Path:
    digest = sha256_bytes(payload)
    path = directory / f"{digest}{suffix}"
    _atomic_write_once(path, payload)
    return path


def _write_stable_reference(path: pathlib.Path, value: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise TrainingError(f"stable reference changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_canonical_json(path: pathlib.Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"could not read {label}") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise TrainingError(f"{label} is not canonical JSON")
    return payload, value


class FrozenBundle:
    """A frozen-bundle view which never probes protected files preselection."""

    def __init__(
        self,
        manifest_path: pathlib.Path,
        manifest_payload: bytes,
        manifest: Mapping[str, Any],
    ) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.manifest_payload = manifest_payload
        self.manifest = dict(manifest)
        self.body_sha256 = str(manifest["body_sha256"])
        self.routes = dict(manifest["routes"])
        self.records = {
            str(record["relative_path"]): dict(record)
            for record in manifest["artifacts"]
        }
        self.protected_routes = self._protected_route_set()

    @classmethod
    def load(cls, manifest_path: pathlib.Path) -> "FrozenBundle":
        _reject_path_markers(manifest_path, "bundle manifest")
        manifest_path = manifest_path.resolve()
        _reject_path_markers(manifest_path, "resolved bundle manifest")
        payload, manifest = _load_canonical_json(
            manifest_path, "compact input bundle"
        )
        verify_body_hash(manifest, schema=BUNDLE_SCHEMA, label="compact input bundle")
        if (
            manifest.get("campaign_id") != CAMPAIGN_ID
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or not isinstance(manifest.get("routes"), dict)
            or not isinstance(manifest.get("artifacts"), list)
            or manifest.get("policy", {}).get("protected_tests_locked") is not True
            or manifest.get("policy", {}).get("runtime_uses_source_paths") is not False
            or manifest.get("policy", {}).get("git_required_after_freeze") is not False
        ):
            raise TrainingError("compact input bundle policy is invalid")
        records = manifest["artifacts"]
        seen_roles: set[str] = set()
        seen_paths: set[str] = set()
        for record in records:
            if (
                not isinstance(record, dict)
                or set(record) != {"role", "relative_path", "sha256", "bytes"}
                or not isinstance(record.get("role"), str)
                or not valid_sha256(record.get("sha256"))
                or type(record.get("bytes")) is not int
                or record["bytes"] < 0
            ):
                raise TrainingError("compact bundle artifact registry is malformed")
            relative = _safe_relative(record.get("relative_path"), "artifact route")
            if record["role"] in seen_roles or relative in seen_paths:
                raise TrainingError("compact bundle artifact registry is not unique")
            seen_roles.add(record["role"])
            seen_paths.add(relative)
        return cls(manifest_path, payload, manifest)

    def _protected_route_set(self) -> set[str]:
        result: set[str] = set()
        for key in (
            "pilot_search_manifests",
            "full_search_manifests",
            "pilot_rank4_manifests",
            "full_rank4_manifests",
        ):
            values = self.routes.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise TrainingError(f"bundle route {key} is incomplete")
            result.add(_safe_relative(values[2], f"{key} test route"))
        canonical = self.routes.get("canonical_splits")
        if not isinstance(canonical, dict):
            raise TrainingError("bundle canonical split routes are missing")
        for split in ("train", "validation", "test"):
            values = canonical.get(split)
            if not isinstance(values, list) or len(values) != 3:
                raise TrainingError(f"canonical {split} routes are incomplete")
        result.update(
            _safe_relative(value, "canonical test route")
            for value in canonical["test"]
        )
        declared = self.manifest.get("protected_splits")
        if declared != ["search:test", "rank4:test", "canonical:test"]:
            raise TrainingError("bundle protected split declaration changed")
        return result

    def is_protected(self, relative: str) -> bool:
        return _safe_relative(relative, "bundle route") in self.protected_routes

    def artifact_path(
        self,
        relative: object,
        *,
        allow_protected: bool = False,
        protected_context: bool = False,
    ) -> pathlib.Path:
        relative_text = _safe_relative(relative, "bundle artifact")
        if (relative_text in self.protected_routes or protected_context) and not allow_protected:
            # Intentionally precedes Path construction, resolve, stat, open, and hash.
            raise TrainingError("protected test artifact is locked before selection")
        record = self.records.get(relative_text)
        if record is None:
            raise TrainingError("bundle route has no registered artifact")
        unresolved = self.root / relative_text
        if unresolved.is_symlink():
            raise TrainingError("frozen bundle artifact became a symlink")
        path = unresolved.resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise TrainingError("bundle route escapes its frozen root") from error
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise TrainingError(f"frozen bundle artifact changed: {relative_text}")
        return path

    def arm_train_routes(self, arm: Arm | str) -> tuple[str, str]:
        if isinstance(arm, str):
            try:
                arm = ARMS[arm]
            except KeyError as error:
                raise TrainingError("unknown training arm") from error
        prefix = "search" if arm.new_source == "search" else "rank4"
        return (
            _safe_relative(
                self.routes[f"pilot_{prefix}_manifests"][0], "pilot train route"
            ),
            _safe_relative(
                self.routes[f"full_{prefix}_manifests"][0], "full train route"
            ),
        )

    def canonical_routes(self, split: str) -> tuple[str, ...]:
        if split not in {"train", "validation", "test"}:
            raise TrainingError("invalid canonical split")
        return tuple(
            _safe_relative(value, f"canonical {split} route")
            for value in self.routes["canonical_splits"][split]
        )

    def common_adjudicator_route(self) -> str:
        return _safe_relative(
            self.routes.get("common_adjudicator_manifest"),
            "common adjudicator route",
        )

    def sidecar_role(self, relative: str) -> str:
        relative = _safe_relative(relative, "teacher sidecar source")
        train_routes = {
            *self.arm_train_routes("search-target"),
            *self.arm_train_routes("rank4-control"),
            *self.canonical_routes("train"),
        }
        if relative in train_routes:
            return "train"
        if relative == self.common_adjudicator_route():
            return "common-adjudicator"
        if relative in set(self.canonical_routes("validation")):
            return "canonical-validation"
        raise TrainingError(
            "teacher predictions are limited to train, common adjudicator, "
            "and canonical validation"
        )


@dataclasses.dataclass(frozen=True)
class Dataset:
    indptr: np.ndarray
    indices: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    group_ids: np.ndarray
    split: str
    source_manifest_sha256: str
    source_npz_sha256: str
    source_route: str = ""
    teacher_predictions: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def active_row(self, row: int) -> np.ndarray:
        return self.indices[self.indptr[row] : self.indptr[row + 1]]

    def active_rows(self, rows: Iterable[int]) -> tuple[np.ndarray, ...]:
        return tuple(self.active_row(int(row)) for row in rows)


@dataclasses.dataclass(frozen=True)
class CompleteTurnSuccessor:
    successor_id: str
    active: np.ndarray
    teacher_value: float
    value_mover: int
    evidence: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class CompleteTurnGroup:
    group_id: str
    parent_mover: int
    successors: tuple[CompleteTurnSuccessor, ...]
    successors_exhaustive: bool = True
    evidence: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SuccessorRankingLabels:
    train: tuple[CompleteTurnGroup, ...]
    validation: tuple[CompleteTurnGroup, ...]
    teacher: Mapping[str, object]
    source_bundle_body_sha256: str
    artifact_sha256: str
    body_sha256: str
    artifact_schema: str = SUCCESSOR_LABEL_SCHEMA


def _ranking_weight(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError("successor ranking loss weight is not numeric")
    normalized = float(value)
    if normalized not in RANKING_LOSS_WEIGHTS:
        raise TrainingError(
            "successor ranking loss weight must be exactly 0, 0.10, or 0.25"
        )
    return normalized


def validate_successor_label_document(
    value: object,
    *,
    source_bundle_body_sha256: str,
    artifact_sha256: str = "0" * 64,
) -> SuccessorRankingLabels:
    """Validate the corpus-owned rich aggregate and project its training core."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "feature_schema", "source_bundle_body_sha256", "teacher",
        "ranking", "splits", "protected_tests_opened", "body_sha256",
    }:
        raise TrainingError("successor label document field roster changed")
    verify_body_hash(
        value, schema=SUCCESSOR_LABEL_SCHEMA, label="successor label document"
    )
    teacher = value.get("teacher")
    ranking = value.get("ranking")
    splits = value.get("splits")
    if (
        value.get("feature_schema") != FEATURE_SCHEMA
        or value.get("source_bundle_body_sha256") != source_bundle_body_sha256
        or not isinstance(teacher, Mapping)
        or set(teacher) != {
            "kind", "artifact_sha256", "payload_sha256",
            "feature_schema_sha256", "source_sha256",
        }
        or teacher.get("kind") != "jacek_replay_bfm_search"
        or any(
            not valid_sha256(teacher.get(name))
            for name in set(teacher) - {"kind"}
        )
        or ranking != {
            "complete_turn_boundaries": True,
            "teacher_value_frame": "explicit-mover-relative",
            "successor_aliases": "canonical-boundary-state",
            "best_tie_break": "successor-id-ascending",
        }
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "validation"}
        or value.get("protected_tests_opened") is not False
        or not valid_sha256(artifact_sha256)
    ):
        raise TrainingError("successor label document binding changed")
    try:
        import jacek_replay_corpus as action_corpus
    except ImportError as error:
        raise TrainingError("complete-turn action corpus validator is unavailable") from error
    if action_corpus.COMPLETE_TURN_SUCCESSOR_LABELS_SCHEMA != SUCCESSOR_LABEL_SCHEMA:
        raise TrainingError("trainer and corpus successor schemas disagree")
    try:
        validated_document = action_corpus.validate_complete_turn_successor_labels(
            dict(value)
        )
    except (TypeError, ValueError) as error:
        raise TrainingError("rich successor-label aggregate validation failed") from error
    if validated_document != dict(value):
        raise TrainingError("successor-label aggregate normalization changed content")

    observed_groups: set[str] = set()
    normalized: dict[str, tuple[CompleteTurnGroup, ...]] = {}
    total_groups = 0
    for split in ("train", "validation"):
        rows = splits.get(split)
        if not isinstance(rows, list):
            raise TrainingError("successor label split is malformed")
        group_ids = [
            row.get("group_id") if isinstance(row, Mapping) else None
            for row in rows
        ]
        if (
            any(not valid_sha256(group_id) for group_id in group_ids)
            or group_ids != sorted(group_ids)
            or len(set(group_ids)) != len(group_ids)
        ):
            raise TrainingError("successor label groups are not unique sorted IDs")
        groups: list[CompleteTurnGroup] = []
        for group_value in rows:
            group_id = str(group_value["group_id"])
            if group_id in observed_groups:
                raise TrainingError("successor label group crosses frozen splits")
            row = {
                "schema": action_corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA,
                "feature_schema": value["feature_schema"],
                "source_bundle_body_sha256": value[
                    "source_bundle_body_sha256"
                ],
                "teacher": dict(teacher),
                "ranking": dict(ranking),
                "split": split,
                "group": dict(group_value),
            }
            try:
                validated_row = action_corpus.validate_complete_turn_action_group(row)
            except (TypeError, ValueError) as error:
                raise TrainingError("rich complete-turn group validation failed") from error
            group = validated_row["group"]
            normalized_successors = tuple(CompleteTurnSuccessor(
                successor_id=str(successor["successor_id"]),
                active=np.asarray(successor["active"], dtype="<u2"),
                teacher_value=float(successor["teacher_value"]),
                value_mover=int(successor["value_mover"]),
                evidence={
                    key: value
                    for key, value in successor.items()
                    if key not in {
                        "successor_id", "active", "teacher_value", "value_mover"
                    }
                },
            ) for successor in group["successors"])
            groups.append(CompleteTurnGroup(
                group_id=group_id,
                parent_mover=int(group["parent_mover"]),
                successors=normalized_successors,
                successors_exhaustive=bool(group["successors_exhaustive"]),
                evidence={
                    key: item
                    for key, item in group.items()
                    if key not in {
                        "group_id", "parent_mover", "successors",
                        "successors_exhaustive",
                    }
                },
            ))
            observed_groups.add(group_id)
            total_groups += 1
        normalized[split] = tuple(groups)
    if total_groups == 0:
        raise TrainingError("successor label document contains no groups")
    return SuccessorRankingLabels(
        train=normalized["train"],
        validation=normalized["validation"],
        teacher=dict(teacher),
        source_bundle_body_sha256=source_bundle_body_sha256,
        artifact_sha256=artifact_sha256,
        body_sha256=str(value["body_sha256"]),
    )


def load_successor_ranking_labels(
    path: pathlib.Path, bundle: FrozenBundle,
) -> SuccessorRankingLabels:
    if path.is_symlink() or not path.is_file():
        raise TrainingError("successor label document is absent or redirected")
    payload, value = _load_canonical_json(path, "successor label document")
    digest = sha256_bytes(payload)
    if path.name != f"{digest}.successor-labels.json":
        raise TrainingError("successor label document is not content addressed")
    labels = validate_successor_label_document(
        value,
        source_bundle_body_sha256=bundle.body_sha256,
        artifact_sha256=digest,
    )
    teacher_core = {
        name: labels.teacher[name]
        for name in (
            "artifact_sha256", "payload_sha256", "feature_schema_sha256"
        )
    }
    if teacher_core != _validate_teacher_identity(bundle, teacher_core):
        raise TrainingError("successor label teacher binding changed")
    return labels


def _validate_active_rows(indptr: np.ndarray, indices: np.ndarray) -> None:
    expected_vertices = np.arange(VERTEX_COUNT, dtype=np.int64)
    for row in range(len(indptr) - 1):
        active = indices[indptr[row] : indptr[row + 1]]
        if (
            len(active) < VERTEX_COUNT
            or np.any(active[1:] <= active[:-1])
            or int(active[0]) < 0
            or int(active[-1]) >= INPUT_COUNT
        ):
            raise TrainingError("sparse active-index row is invalid")
        categories = active[active >= EDGE_COUNT].astype(np.int64) - EDGE_COUNT
        if (
            len(categories) != VERTEX_COUNT
            or not np.array_equal(categories // VERTEX_CATEGORIES, expected_vertices)
        ):
            raise TrainingError("sparse row does not select one category per vertex")


def load_shard(
    bundle: FrozenBundle,
    relative: str,
    *,
    allow_protected: bool = False,
) -> Dataset:
    relative = _safe_relative(relative, "shard manifest route")
    protected = bundle.is_protected(relative)
    manifest_path = bundle.artifact_path(
        relative,
        allow_protected=allow_protected,
        protected_context=protected,
    )
    manifest_payload, manifest = _load_canonical_json(
        manifest_path, "sparse shard manifest"
    )
    manifest_sha = sha256_bytes(manifest_payload)
    if (
        manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("split") not in {"train", "validation", "test"}
        or manifest_path.name != f"{manifest_sha}.json"
        or not isinstance(manifest.get("npz"), str)
        or not isinstance(manifest.get("npz_sha256"), str)
        or manifest["npz"] != f"{manifest['npz_sha256']}.npz"
    ):
        raise TrainingError("sparse shard manifest contract is invalid")
    parent = pathlib.PurePosixPath(relative).parent
    npz_relative = (parent / str(manifest["npz"])).as_posix()
    npz_path = bundle.artifact_path(
        npz_relative,
        allow_protected=allow_protected,
        protected_context=protected,
    )
    if sha256_file(npz_path) != manifest["npz_sha256"]:
        raise TrainingError("sparse shard NPZ hash changed")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            expected = {"indptr", "indices", "targets", "weights", "group_ids"}
            if set(archive.files) != expected:
                raise TrainingError("sparse shard NPZ arrays changed")
            arrays = {name: archive[name].copy() for name in expected}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError("sparse shard NPZ is corrupt") from error
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
        or count < 1
        or indptr.shape != (count + 1,)
        or weights.shape != (count,)
        or group_ids.shape != (count,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != len(indices)
        or np.any(indptr[1:] < indptr[:-1])
        or np.any(indices >= INPUT_COUNT)
        or not np.all(np.isfinite(targets))
        or np.any(np.abs(targets) > 1.0)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or manifest.get("samples") != count
        or manifest.get("active_features") != len(indices)
    ):
        raise TrainingError("sparse shard array contract is invalid")
    _validate_active_rows(indptr, indices)
    return Dataset(
        indptr=indptr,
        indices=indices,
        targets=targets,
        weights=weights,
        group_ids=group_ids,
        split=str(manifest["split"]),
        source_manifest_sha256=manifest_sha,
        source_npz_sha256=str(manifest["npz_sha256"]),
        source_route=relative,
    )


def concatenate_datasets(datasets: Sequence[Dataset], *, split: str) -> Dataset:
    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise TrainingError("dataset concatenation requires nonempty inputs")
    if any(dataset.split != split for dataset in datasets):
        raise TrainingError("dataset concatenation crosses frozen splits")
    teacher_presence = [dataset.teacher_predictions is not None for dataset in datasets]
    if any(teacher_presence) and not all(teacher_presence):
        raise TrainingError("teacher prediction coverage is incomplete")
    total_rows = sum(len(dataset) for dataset in datasets)
    indptr = np.empty(total_rows + 1, dtype="<i8")
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
    manifest_binding = sha256_bytes(
        canonical_json_bytes([dataset.source_manifest_sha256 for dataset in datasets])
    )
    npz_binding = sha256_bytes(
        canonical_json_bytes([dataset.source_npz_sha256 for dataset in datasets])
    )
    teacher = None
    if all(teacher_presence):
        teacher = np.concatenate(
            [np.asarray(dataset.teacher_predictions, dtype="<f4") for dataset in datasets]
        ).astype("<f4", copy=False)
    return Dataset(
        indptr=indptr,
        indices=np.concatenate([dataset.indices for dataset in datasets]).astype(
            "<u2", copy=False
        ),
        targets=np.concatenate([dataset.targets for dataset in datasets]).astype(
            "<f4", copy=False
        ),
        weights=np.concatenate([dataset.weights for dataset in datasets]).astype(
            "<f4", copy=False
        ),
        group_ids=np.concatenate([dataset.group_ids for dataset in datasets]).astype(
            "V32", copy=False
        ),
        split=split,
        source_manifest_sha256=manifest_binding,
        source_npz_sha256=npz_binding,
        source_route="+".join(dataset.source_route for dataset in datasets),
        teacher_predictions=teacher,
    )


def dataset_identity(dataset: Dataset) -> dict[str, object]:
    digest = hashlib.sha256()
    for name, value in sorted(
        {
            "group_ids": dataset.group_ids,
            "indices": dataset.indices,
            "indptr": dataset.indptr,
            "targets": dataset.targets,
            "weights": dataset.weights,
        }.items()
    ):
        array = np.asarray(value)
        contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
        digest.update(canonical_json_bytes({
            "name": name,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }))
        digest.update(contiguous.tobytes(order="C"))
    return {
        "samples": len(dataset),
        "active_features": int(len(dataset.indices)),
        "sha256": digest.hexdigest(),
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "source_npz_sha256": dataset.source_npz_sha256,
    }


def _stream_cycle(count: int, *, seed: int, stream: str, cycle: int) -> np.ndarray:
    if count <= 0 or cycle < 0 or not stream:
        raise TrainingError("training stream arguments are invalid")
    material = f"{seed}:{stream}:{cycle}".encode("ascii")
    cycle_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(cycle_seed).permutation(count)


def _continuous_rows(
    count: int,
    total: int,
    *,
    seed: int,
    stream: str,
    start: int = 0,
) -> np.ndarray:
    if count <= 0 or total < 0 or start < 0:
        raise TrainingError("continuous row stream arguments are invalid")
    output = np.empty(total, dtype=np.int64)
    offset = 0
    cycle, cycle_offset = divmod(start, count)
    while offset < total:
        order = _stream_cycle(count, seed=seed, stream=stream, cycle=cycle)
        take = min(count - cycle_offset, total - offset)
        output[offset : offset + take] = order[cycle_offset : cycle_offset + take]
        offset += take
        cycle += 1
        cycle_offset = 0
    return output


def mixed_epoch_schedule(
    new_count: int,
    anchor_count: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact paired 64-new/192-anchor row streams for an epoch."""

    if epoch <= 0 or new_count <= 0 or anchor_count <= 0:
        raise TrainingError("mixed epoch counts must be positive")
    batch_count = math.ceil(new_count / NEW_ROWS_PER_BATCH)
    new_total = batch_count * NEW_ROWS_PER_BATCH
    anchor_total = batch_count * ANCHOR_ROWS_PER_BATCH
    new_rows = _continuous_rows(
        new_count,
        new_total,
        seed=seed,
        stream=f"new:epoch:{epoch}",
    )
    anchor_rows = _continuous_rows(
        anchor_count,
        anchor_total,
        seed=seed,
        stream="anchor",
        start=(epoch - 1) * anchor_total,
    )
    return new_rows, anchor_rows


def mixed_epoch_batches(
    new_count: int,
    anchor_count: int,
    *,
    seed: int,
    epoch: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    new_rows, anchor_rows = mixed_epoch_schedule(
        new_count, anchor_count, seed=seed, epoch=epoch
    )
    batch_count = len(new_rows) // NEW_ROWS_PER_BATCH
    for batch in range(batch_count):
        yield (
            new_rows[
                batch * NEW_ROWS_PER_BATCH : (batch + 1) * NEW_ROWS_PER_BATCH
            ],
            anchor_rows[
                batch * ANCHOR_ROWS_PER_BATCH :
                (batch + 1) * ANCHOR_ROWS_PER_BATCH
            ],
        )


def successor_ranking_epoch_schedule(
    group_count: int,
    batch_count: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    """Partition one full weighted-pool permutation across scalar batches."""

    if (
        group_count <= 0
        or batch_count <= 0
        or group_count < batch_count
        or epoch <= 0
    ):
        raise TrainingError("successor ranking schedule arguments are invalid")
    order = _stream_cycle(
        group_count,
        seed=seed,
        stream="successor-ranking-weighted-pool",
        cycle=epoch - 1,
    )
    smaller, larger = divmod(group_count, batch_count)
    result = []
    offset = 0
    for batch in range(batch_count):
        size = smaller + int(batch < larger)
        result.append(order[offset : offset + size])
        offset += size
    if (
        offset != group_count
        or any(len(indices) == 0 for indices in result)
        or max(map(len, result)) - min(map(len, result)) > 1
        or sorted(int(index) for indices in result for index in indices)
        != list(range(group_count))
    ):
        raise TrainingError("successor ranking schedule lost weighted pool entries")
    return tuple(result)


def mixed_epoch_coverage(new_count: int, anchor_count: int, epoch: int) -> dict[str, Any]:
    if epoch <= 0:
        raise TrainingError("coverage epoch must be positive")
    batch_count = math.ceil(new_count / NEW_ROWS_PER_BATCH)
    new_rows = batch_count * NEW_ROWS_PER_BATCH
    anchor_rows = batch_count * ANCHOR_ROWS_PER_BATCH
    return {
        "new": {
            "dataset_rows": new_count,
            "rows_per_epoch": new_rows,
            "cumulative_rows": new_rows * epoch,
            "complete_epoch_permutations": epoch,
            "padding_rows_per_epoch": new_rows - new_count,
        },
        "anchor": {
            "dataset_rows": anchor_count,
            "rows_per_epoch": anchor_rows,
            "cumulative_rows": anchor_rows * epoch,
            "complete_permutations": anchor_rows * epoch // anchor_count,
            "permutation_offset": anchor_rows * epoch % anchor_count,
        },
    }


def anchor_coverage_complete_epoch(new_count: int, anchor_count: int) -> int:
    rows = mixed_epoch_coverage(new_count, anchor_count, 1)["anchor"]["rows_per_epoch"]
    return math.ceil(anchor_count / rows)


def independently_normalized_mixed_weights(
    new_weights: np.ndarray, anchor_weights: np.ndarray
) -> np.ndarray:
    """Normalize sources independently and apply their immutable .25/.75 shares."""

    new_weights = np.asarray(new_weights, dtype=np.float32)
    anchor_weights = np.asarray(anchor_weights, dtype=np.float32)
    if (
        new_weights.shape != (NEW_ROWS_PER_BATCH,)
        or anchor_weights.shape != (ANCHOR_ROWS_PER_BATCH,)
        or not np.all(np.isfinite(new_weights))
        or not np.all(np.isfinite(anchor_weights))
        or np.any(new_weights <= 0.0)
        or np.any(anchor_weights <= 0.0)
    ):
        raise TrainingError("mixed batch weights violate the 64/192 contract")
    normalized_new = new_weights * np.float32(
        0.25 / float(np.sum(new_weights, dtype=np.float64))
    )
    normalized_anchor = anchor_weights * np.float32(
        0.75 / float(np.sum(anchor_weights, dtype=np.float64))
    )
    result = np.concatenate((normalized_new, normalized_anchor)).astype(
        np.float32, copy=False
    )
    if not math.isclose(float(np.sum(result[:64], dtype=np.float64)), 0.25,
                        rel_tol=0.0, abs_tol=2e-7):
        raise TrainingError("new-source batch normalization drifted")
    if not math.isclose(float(np.sum(result[64:], dtype=np.float64)), 0.75,
                        rel_tol=0.0, abs_tol=2e-7):
        raise TrainingError("anchor-source batch normalization drifted")
    return result


def _validate_parameters(
    parameters: Mapping[str, np.ndarray], architecture: Architecture
) -> dict[str, np.ndarray]:
    if set(parameters) != {"w1", "w2", "w3"}:
        raise TrainingError("compact model must contain exactly w1, w2, and w3")
    normalized: dict[str, np.ndarray] = {}
    for name, shape in architecture.shapes.items():
        value = np.asarray(parameters[name], dtype=np.float32)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise TrainingError(f"compact parameter {name} is invalid")
        normalized[name] = value
    return normalized


def initialize_parameters(
    architecture: Architecture | str, seed: int
) -> dict[str, np.ndarray]:
    if isinstance(architecture, str):
        try:
            architecture = ARCHITECTURES[architecture]
        except KeyError as error:
            raise TrainingError("unknown compact architecture") from error
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise TrainingError("training seed must fit uint64")
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(
            0.0, 0.02, architecture.shapes["w1"]
        ).astype(np.float32),
        "w2": rng.normal(
            0.0,
            math.sqrt(1.0 / architecture.hidden_one),
            architecture.shapes["w2"],
        ).astype(np.float32),
        "w3": rng.normal(
            0.0,
            math.sqrt(1.0 / architecture.hidden_two),
            architecture.shapes["w3"],
        ).astype(np.float32),
    }


def first_activation(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value >= 0.0, value * value, LEAKY_SLOPE * value
    ).astype(np.float32)


def first_activation_derivative(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value >= 0.0, np.float32(2.0) * value, LEAKY_SLOPE
    ).astype(np.float32)


def second_activation(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value >= 0.0, value, LEAKY_SLOPE * value
    ).astype(np.float32)


def second_activation_derivative(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(value >= 0.0, 1.0, LEAKY_SLOPE).astype(np.float32)


def fast_tanh(value: np.ndarray) -> np.ndarray:
    """Deployment's fixed rational tanh in the same expression order."""

    value = np.asarray(value, dtype=np.float32)
    clipped = np.clip(value, np.float32(-4.95), np.float32(4.95)).astype(
        np.float32
    )
    square = clipped * clipped
    numerator = clipped * (
        np.float32(135135.0)
        + square
        * (
            np.float32(17325.0)
            + square * (np.float32(378.0) + square)
        )
    )
    denominator = np.float32(135135.0) + square * (
        np.float32(62370.0)
        + square * (np.float32(3150.0) + np.float32(28.0) * square)
    )
    result = numerator / denominator
    return np.where(
        value < np.float32(-4.95),
        np.float32(-1.0),
        np.where(value > np.float32(4.95), np.float32(1.0), result),
    ).astype(np.float32)


def fast_tanh_derivative(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    clipped = np.clip(value, np.float32(-4.95), np.float32(4.95)).astype(
        np.float32
    )
    square = clipped * clipped
    numerator = clipped * (
        np.float32(135135.0)
        + square
        * (
            np.float32(17325.0)
            + square * (np.float32(378.0) + square)
        )
    )
    denominator = np.float32(135135.0) + square * (
        np.float32(62370.0)
        + square * (np.float32(3150.0) + np.float32(28.0) * square)
    )
    numerator_derivative = np.float32(135135.0) + square * (
        np.float32(51975.0)
        + square * (np.float32(1890.0) + np.float32(7.0) * square)
    )
    denominator_derivative = np.float32(2.0) * clipped * (
        np.float32(62370.0)
        + square * (np.float32(6300.0) + np.float32(84.0) * square)
    )
    derivative = (
        numerator_derivative * denominator
        - numerator * denominator_derivative
    ) / (denominator * denominator)
    return np.where(np.abs(value) > np.float32(4.95), 0.0, derivative).astype(
        np.float32
    )


def _fast_tanh_scalar(value: np.float32) -> np.float32:
    if value < np.float32(-4.95):
        return np.float32(-1.0)
    if value > np.float32(4.95):
        return np.float32(1.0)
    square = np.float32(value * value)
    numerator = np.float32(
        value
        * np.float32(
            np.float32(135135.0)
            + np.float32(
                square
                * np.float32(
                    np.float32(17325.0)
                    + np.float32(
                        square * np.float32(np.float32(378.0) + square)
                    )
                )
            )
        )
    )
    denominator = np.float32(
        np.float32(135135.0)
        + np.float32(
            square
            * np.float32(
                np.float32(62370.0)
                + np.float32(
                    square
                    * np.float32(
                        np.float32(3150.0) + np.float32(28.0) * square
                    )
                )
            )
        )
    )
    return np.float32(numerator / denominator)


@dataclasses.dataclass(frozen=True)
class QuantizedWeights:
    integer: dict[str, np.ndarray]
    scales: dict[str, np.float32]

    def effective(self) -> dict[str, np.ndarray]:
        return {
            name: self.integer[name].astype(np.float32) * self.scales[name]
            for name in ("w1", "w2", "w3")
        }


def quantize_fixed(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    scales: Mapping[str, object],
) -> QuantizedWeights:
    parameters = _validate_parameters(parameters, architecture)
    integer: dict[str, np.ndarray] = {}
    normalized_scales: dict[str, np.float32] = {}
    for name in ("w1", "w2", "w3"):
        try:
            scale = np.float32(scales[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise TrainingError(f"fixed scale {name} is invalid") from error
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise TrainingError(f"fixed scale {name} must be finite and positive")
        values = np.clip(
            np.rint(parameters[name] / scale),
            QUANTIZATION_MINIMUM,
            QUANTIZATION_MAXIMUM,
        ).astype(np.int8)
        if np.any(values == -4) or np.any(values < -3) or np.any(values > 3):
            raise TrainingError("fixed quantizer emitted a forbidden code")
        integer[name] = values
        normalized_scales[name] = scale
    return QuantizedWeights(integer, normalized_scales)


def robust_scale_candidates(
    value: np.ndarray,
    *,
    quantiles: Sequence[tuple[str, int, int]] = ROBUST_SCALE_QUANTILES,
) -> tuple[np.float32, ...]:
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise TrainingError("scale search received a nonfinite tensor")
    ordered = np.sort(np.abs(value).reshape(-1))
    if not ordered.size:
        return (np.float32(1.0),)
    result: list[np.float32] = []
    for _name, numerator, denominator in quantiles:
        index = ((ordered.size - 1) * numerator) // denominator
        threshold = float(ordered[index])
        if not math.isfinite(threshold) or threshold <= 0.0:
            continue
        scale = np.float32(threshold / QUANTIZATION_MAXIMUM)
        if scale > 0.0 and all(scale != prior for prior in result):
            result.append(scale)
    if not result:
        positive = ordered[ordered > 0.0]
        return (
            np.float32(
                float(positive[0]) / QUANTIZATION_MAXIMUM
                if positive.size
                else 1.0
            ),
        )
    return tuple(result)


def _refined_scale_candidates(
    base: np.float32,
    multipliers: Sequence[tuple[str, int, int]],
) -> tuple[np.float32, ...]:
    result: list[np.float32] = []
    for _name, numerator, denominator in multipliers:
        candidate = np.float32(float(base) * numerator / denominator)
        if (
            math.isfinite(float(candidate))
            and candidate > 0.0
            and all(candidate != prior for prior in result)
        ):
            result.append(candidate)
    if not result:
        raise TrainingError("scale refinement produced no positive candidate")
    return tuple(result)


def forward(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    active: Sequence[np.ndarray],
    *,
    quantized: QuantizedWeights | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    parameters = _validate_parameters(parameters, architecture)
    if not active:
        raise TrainingError("forward pass requires at least one row")
    if quantized is None:
        effective = parameters
        first_pre = np.empty((len(active), architecture.hidden_one), dtype=np.float32)
        for row, indices in enumerate(active):
            first_pre[row] = np.sum(
                effective["w1"][indices], axis=0, dtype=np.float32
            )
    else:
        effective = quantized.effective()
        _validate_parameters(effective, architecture)
        first_pre = np.empty((len(active), architecture.hidden_one), dtype=np.float32)
        for row, indices in enumerate(active):
            accumulator = np.sum(
                quantized.integer["w1"][indices], axis=0, dtype=np.int32
            )
            first_pre[row] = accumulator.astype(np.float32) * quantized.scales["w1"]
    first = first_activation(first_pre)
    second_pre = np.asarray(first @ effective["w2"], dtype=np.float32)
    second = second_activation(second_pre)
    output_pre = np.asarray(second @ effective["w3"], dtype=np.float32)
    output = fast_tanh(output_pre)
    if not np.all(np.isfinite(output)):
        raise TrainingError("compact inference produced a nonfinite value")
    return output, (first_pre, first, second_pre, second, output_pre)


def scalar_quantized_forward(
    quantized: QuantizedWeights,
    architecture: Architecture,
    active: Sequence[int] | np.ndarray,
) -> np.float32:
    """Scalar float32 deployment order with W1 integer accumulation once."""

    indices = np.asarray(active, dtype=np.uint16)
    q1 = quantized.integer["w1"]
    q2 = quantized.integer["w2"]
    q3 = quantized.integer["w3"]
    if q1.shape != architecture.shapes["w1"]:
        raise TrainingError("quantized scalar evaluator has a wrong architecture")
    first = np.empty(architecture.hidden_one, dtype=np.float32)
    for hidden in range(architecture.hidden_one):
        accumulator = 0
        for feature in indices:
            accumulator += int(q1[int(feature), hidden])
        value = np.float32(np.int32(accumulator) * quantized.scales["w1"])
        first[hidden] = (
            np.float32(value * value)
            if value >= 0.0
            else np.float32(LEAKY_SLOPE * value)
        )
    second = np.empty(architecture.hidden_two, dtype=np.float32)
    for output in range(architecture.hidden_two):
        total = np.float32(0.0)
        for hidden in range(architecture.hidden_one):
            term = np.float32(
                np.float32(first[hidden] * quantized.scales["w2"])
                * np.float32(q2[hidden, output])
            )
            total = np.float32(total + term)
        second[output] = (
            total if total >= 0.0 else np.float32(LEAKY_SLOPE * total)
        )
    total = np.float32(0.0)
    for hidden in range(architecture.hidden_two):
        term = np.float32(
            np.float32(second[hidden] * quantized.scales["w3"])
            * np.float32(q3[hidden])
        )
        total = np.float32(total + term)
    return _fast_tanh_scalar(total)


def pack_signed_three_bit(values: Sequence[int] | np.ndarray) -> bytes:
    output = bytearray()
    accumulator = 0
    available = 0
    for raw in values:
        value = int(raw)
        if isinstance(raw, (bool, np.bool_)) or not -3 <= value <= 3:
            raise TrainingError("signed-three-bit payload contains a forbidden value")
        encoded = value & 0b111
        if encoded == 0b100:
            raise TrainingError("signed-three-bit payload contains forbidden code 100")
        accumulator |= encoded << available
        available += QUANTIZATION_BITS
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_signed_three_bit(payload: bytes, count: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TrainingError("signed-three-bit count is invalid")
    expected = (count * QUANTIZATION_BITS + 7) // 8
    if len(payload) != expected:
        raise TrainingError(
            f"packed payload has {len(payload)} bytes; expected {expected}"
        )
    result = np.empty(count, dtype=np.int8)
    accumulator = 0
    available = 0
    source_index = 0
    for index in range(count):
        while available < QUANTIZATION_BITS:
            if source_index >= len(payload):
                raise TrainingError("packed payload is truncated")
            accumulator |= payload[source_index] << available
            source_index += 1
            available += 8
        encoded = accumulator & 0b111
        accumulator >>= QUANTIZATION_BITS
        available -= QUANTIZATION_BITS
        if encoded == 0b100:
            raise TrainingError("packed payload contains forbidden code 100 (-4)")
        result[index] = encoded - 8 if encoded & 0b100 else encoded
    if source_index != len(payload) or accumulator != 0:
        raise TrainingError("packed payload has nonzero padding or trailing bytes")
    return result


def _flatten_quantized(
    quantized: QuantizedWeights, architecture: Architecture
) -> np.ndarray:
    pieces = []
    for name in ("w1", "w2", "w3"):
        value = np.asarray(quantized.integer[name], dtype=np.int8)
        if value.shape != architecture.shapes[name]:
            raise TrainingError(f"quantized tensor {name} has a wrong shape")
        if np.any(value < -3) or np.any(value > 3) or np.any(value == -4):
            raise TrainingError(f"quantized tensor {name} has a forbidden value")
        pieces.append(value.reshape(-1, order="C"))
    return np.concatenate(pieces).astype(np.int8, copy=False)


def runtime_document(
    architecture: Architecture,
    quantized: QuantizedWeights,
    *,
    arm: Arm | str,
    seed: int,
    float_epoch: int,
    qat_epoch: int,
    source_bundle_body_sha256: str,
) -> dict[str, object]:
    if isinstance(arm, str):
        try:
            arm = ARMS[arm]
        except KeyError as error:
            raise TrainingError("unknown runtime arm") from error
    flat = _flatten_quantized(quantized, architecture)
    packed = pack_signed_three_bit(flat)
    counts = architecture.weight_counts
    expected_bytes = (counts["total"] * QUANTIZATION_BITS + 7) // 8
    if len(packed) != expected_bytes:
        raise TrainingError("internal packed runtime length mismatch")
    scales = {
        name: float(np.float32(quantized.scales[name]))
        for name in ("w1", "w2", "w3")
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise TrainingError("runtime quantization scales are invalid")
    body: dict[str, object] = {
        "schema": RUNTIME_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "architecture": {
            "name": architecture.name,
            "dimensions": list(architecture.dimensions),
            "biases": False,
            "activations": list(ACTIVATIONS),
            "payload_layout": PAYLOAD_LAYOUT,
        },
        "quantization": {
            "bits": QUANTIZATION_BITS,
            "minimum": QUANTIZATION_MINIMUM,
            "maximum": QUANTIZATION_MAXIMUM,
            "scheme": "symmetric-signed-three-bit-per-layer-fixed-scale",
            "packing": PACKING,
            "scales": scales,
            "weight_counts": counts,
            "packed_byte_count": len(packed),
            "payload_sha256": sha256_bytes(packed),
            "payload_base64": base64.b64encode(packed).decode("ascii"),
        },
        "selection": {
            "arm": arm.name,
            "seed": seed,
            "float_epoch": float_epoch,
            "qat_epoch": qat_epoch,
            "source_bundle_body_sha256": source_bundle_body_sha256,
        },
    }
    return body_hashed(body)


def validate_runtime_document(
    value: Mapping[str, object],
) -> tuple[Architecture, QuantizedWeights, dict[str, object]]:
    verify_body_hash(value, schema=RUNTIME_SCHEMA, label="compact runtime")
    if set(value) != {
        "schema",
        "feature_schema",
        "architecture",
        "quantization",
        "selection",
        "body_sha256",
    } or value.get("feature_schema") != FEATURE_SCHEMA:
        raise TrainingError("compact runtime feature schema changed")
    architecture_value = value.get("architecture")
    if not isinstance(architecture_value, dict):
        raise TrainingError("compact runtime architecture is missing")
    name = architecture_value.get("name")
    if not isinstance(name, str) or name not in ARCHITECTURES:
        raise TrainingError("compact runtime architecture name is invalid")
    architecture = ARCHITECTURES[name]
    if architecture_value != {
        "name": architecture.name,
        "dimensions": list(architecture.dimensions),
        "biases": False,
        "activations": list(ACTIVATIONS),
        "payload_layout": PAYLOAD_LAYOUT,
    }:
        raise TrainingError("compact runtime architecture contract changed")
    quantization = value.get("quantization")
    if not isinstance(quantization, dict):
        raise TrainingError("compact runtime quantization is missing")
    if set(quantization) != {
        "bits",
        "minimum",
        "maximum",
        "scheme",
        "packing",
        "scales",
        "weight_counts",
        "packed_byte_count",
        "payload_sha256",
        "payload_base64",
    }:
        raise TrainingError("compact runtime quantization fields changed")
    expected_static = {
        "bits": QUANTIZATION_BITS,
        "minimum": QUANTIZATION_MINIMUM,
        "maximum": QUANTIZATION_MAXIMUM,
        "scheme": "symmetric-signed-three-bit-per-layer-fixed-scale",
        "packing": PACKING,
        "weight_counts": architecture.weight_counts,
    }
    if any(quantization.get(key) != expected for key, expected in expected_static.items()):
        raise TrainingError("compact runtime quantization contract changed")
    scales_raw = quantization.get("scales")
    if not isinstance(scales_raw, dict) or set(scales_raw) != {"w1", "w2", "w3"}:
        raise TrainingError("compact runtime scales are incomplete")
    scales: dict[str, np.float32] = {}
    for key in ("w1", "w2", "w3"):
        raw = scales_raw[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TrainingError("compact runtime scale is not numeric")
        scale = np.float32(raw)
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise TrainingError("compact runtime scale is invalid")
        # Runtime JSON must preserve the exact float32 value used by inference.
        if float(scale) != float(raw):
            raise TrainingError("compact runtime scale is not canonical float32")
        scales[key] = scale
    encoded = quantization.get("payload_base64")
    if not isinstance(encoded, str) or not encoded.isascii():
        raise TrainingError("compact runtime payload is not ASCII base64")
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TrainingError("compact runtime payload base64 is invalid") from error
    total = architecture.weight_counts["total"]
    expected_bytes = (total * QUANTIZATION_BITS + 7) // 8
    if (
        quantization.get("packed_byte_count") != expected_bytes
        or len(packed) != expected_bytes
        or quantization.get("payload_sha256") != sha256_bytes(packed)
    ):
        raise TrainingError("compact runtime payload length or hash changed")
    flat = unpack_signed_three_bit(packed, total)
    integer: dict[str, np.ndarray] = {}
    offset = 0
    for tensor in ("w1", "w2", "w3"):
        count = architecture.weight_counts[tensor]
        integer[tensor] = flat[offset : offset + count].reshape(
            architecture.shapes[tensor], order="C"
        ).copy()
        offset += count
    selection = value.get("selection")
    if (
        not isinstance(selection, dict)
        or set(selection) != {
            "arm",
            "seed",
            "float_epoch",
            "qat_epoch",
            "source_bundle_body_sha256",
        }
        or selection.get("arm") not in ARMS
        or selection.get("seed") not in FIXED_SEEDS
        or isinstance(selection.get("float_epoch"), bool)
        or not isinstance(selection.get("float_epoch"), int)
        or not 1 <= selection["float_epoch"] <= MAX_FLOAT_EPOCHS
        or isinstance(selection.get("qat_epoch"), bool)
        or not isinstance(selection.get("qat_epoch"), int)
        or not 0 <= selection["qat_epoch"] <= QAT_EPOCHS
        or not valid_sha256(selection.get("source_bundle_body_sha256"))
    ):
        raise TrainingError("compact runtime selection binding is invalid")
    return architecture, QuantizedWeights(integer, scales), dict(selection)


def load_runtime(
    path: pathlib.Path,
) -> tuple[Architecture, QuantizedWeights, dict[str, object], dict[str, Any]]:
    payload, value = _load_canonical_json(path, "compact quantized runtime")
    if path.name != f"{sha256_bytes(payload)}.runtime.json":
        raise TrainingError("compact runtime is not content addressed")
    architecture, quantized, selection = validate_runtime_document(value)
    return architecture, quantized, selection, value


def write_runtime(
    output_directory: pathlib.Path,
    architecture: Architecture,
    quantized: QuantizedWeights,
    **selection: object,
) -> pathlib.Path:
    document = runtime_document(
        architecture, quantized, **selection  # type: ignore[arg-type]
    )
    payload = canonical_json_bytes(document)
    path = _write_content_addressed(output_directory, payload, ".runtime.json")
    # Refuse publication unless the exact bytes round-trip all strict checks.
    load_runtime(path)
    return path


def assert_quantized_inference_parity(
    quantized: QuantizedWeights,
    architecture: Architecture,
    dataset: Dataset,
    *,
    maximum_rows: int = 4_096,
    tolerance: float = 2e-6,
) -> dict[str, object]:
    rows = min(len(dataset), maximum_rows)
    if rows <= 0:
        raise TrainingError("inference parity dataset is empty")
    placeholder = quantized.effective()
    vector, _ = forward(
        placeholder,
        architecture,
        dataset.active_rows(range(rows)),
        quantized=quantized,
    )
    scalar = np.asarray(
        [
            scalar_quantized_forward(quantized, architecture, dataset.active_row(row))
            for row in range(rows)
        ],
        dtype=np.float32,
    )
    difference = float(np.max(np.abs(vector - scalar)))
    if not math.isfinite(difference) or difference > tolerance:
        raise TrainingError(
            f"vector/scalar compact inference differs by {difference:.9g}"
        )
    return {"states": rows, "maximum_absolute_error": difference, "tolerance": tolerance}


class AcceptedTeacherPredictor:
    """Read the copied accepted float teacher and produce ordered float32 values."""

    def __init__(self, bundle: FrozenBundle) -> None:
        runtime_relative = _safe_relative(
            bundle.routes.get("teacher_runtime"), "teacher runtime route"
        )
        manifest_relative = _safe_relative(
            bundle.routes.get("teacher_manifest"), "teacher manifest route"
        )
        self.runtime_path = bundle.artifact_path(runtime_relative)
        manifest_path = bundle.artifact_path(manifest_relative)
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingError("accepted teacher manifest is invalid") from error
        expected_architecture = {
            "dimensions": [INPUT_COUNT, 192, 32, 1],
            "biases": False,
            "activations": ["square-leaky-0.01", "leaky-relu-0.01", "tanh"],
            "payload_layout": "w1-input-major,w2-input-major,w3",
        }
        runtime_record = bundle.records[runtime_relative]
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "papersoccer.jacek-replay-bfm-model.v2"
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("architecture") != expected_architecture
            or manifest.get("runtime", {}).get("artifact_sha256")
            != runtime_record["sha256"]
        ):
            raise TrainingError("accepted teacher manifest contract changed")
        try:
            import jacek_replay_train as large_teacher

            self.parameters, runtime_report = large_teacher.load_runtime(
                self.runtime_path
            )
        except (ImportError, OSError, ValueError) as error:
            raise TrainingError("accepted teacher runtime could not be loaded") from error
        if runtime_report.get("artifact_sha256") != runtime_record["sha256"]:
            raise TrainingError("accepted teacher runtime identity changed")
        self.runtime_identity = {
            "artifact_sha256": runtime_report["artifact_sha256"],
            "payload_sha256": runtime_report["payload_sha256"],
            "feature_schema_sha256": runtime_report["feature_schema_sha256"],
        }
        self._module = large_teacher

    def __call__(self, dataset: Dataset, batch_size: int = 4_096) -> np.ndarray:
        predictions = np.empty(len(dataset), dtype="<f4")
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            values, _ = self._module.forward(
                self.parameters, dataset.active_rows(range(start, stop))
            )
            predictions[start:stop] = np.asarray(values, dtype="<f4")
        if (
            not np.all(np.isfinite(predictions))
            or np.any(np.abs(predictions) > 1.0)
        ):
            raise TrainingError("accepted teacher produced an invalid prediction")
        return predictions


def _prediction_payload(predictions: np.ndarray) -> bytes:
    value = np.asarray(predictions, dtype="<f4")
    if value.ndim != 1 or not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1.0):
        raise TrainingError("teacher prediction payload is invalid")
    return value.tobytes(order="C")


def _validate_teacher_identity(
    bundle: FrozenBundle, identity: Mapping[str, object]
) -> dict[str, object]:
    if set(identity) != {
        "artifact_sha256", "payload_sha256", "feature_schema_sha256"
    } or any(not valid_sha256(identity.get(name)) for name in identity):
        raise TrainingError("teacher prediction identity is invalid")
    runtime_route = bundle.routes.get("teacher_runtime")
    if runtime_route is not None:
        relative = _safe_relative(runtime_route, "teacher runtime route")
        record = bundle.records.get(relative)
        if record is None or identity.get("artifact_sha256") != record["sha256"]:
            raise TrainingError("teacher prediction runtime is not the accepted copy")
    return dict(identity)


def generate_teacher_sidecar(
    bundle: FrozenBundle,
    source_route: str,
    output_directory: pathlib.Path,
    predictor: Callable[[Dataset], np.ndarray],
    teacher_identity: Mapping[str, object],
) -> pathlib.Path:
    """Generate one allowlisted, row-order-bound teacher prediction sidecar."""

    role = bundle.sidecar_role(source_route)
    teacher_identity = _validate_teacher_identity(bundle, teacher_identity)
    dataset = load_shard(bundle, source_route)
    if role == "train" and dataset.split != "train":
        raise TrainingError("teacher train sidecar source has the wrong split")
    if role in {"common-adjudicator", "canonical-validation"} and dataset.split != "validation":
        raise TrainingError("teacher validation sidecar source has the wrong split")
    predictions = np.asarray(predictor(dataset), dtype="<f4")
    if predictions.shape != (len(dataset),):
        raise TrainingError("teacher sidecar prediction count changed")
    payload = _prediction_payload(predictions)
    prediction_path = _write_content_addressed(
        output_directory, payload, ".predictions.f32"
    )
    body: dict[str, object] = {
        "schema": SIDECAR_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "classification": role,
        "source": {
            "route": source_route,
            "manifest_sha256": dataset.source_manifest_sha256,
            "npz_sha256": dataset.source_npz_sha256,
            "dataset": dataset_identity(dataset),
            "split": dataset.split,
        },
        "teacher": teacher_identity,
        "predictions": {
            "file": prediction_path.name,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "count": len(dataset),
            "dtype": "little-endian-float32[n]",
        },
        "protected_test_predictions": False,
    }
    document = body_hashed(body)
    path = _write_content_addressed(
        output_directory, canonical_json_bytes(document), ".sidecar.json"
    )
    load_teacher_sidecar(bundle, dataset, path)
    return path


def load_teacher_sidecar(
    bundle: FrozenBundle,
    dataset: Dataset,
    sidecar_path: pathlib.Path,
) -> np.ndarray:
    payload, sidecar = _load_canonical_json(sidecar_path, "teacher sidecar")
    if sidecar_path.name != f"{sha256_bytes(payload)}.sidecar.json":
        raise TrainingError("teacher sidecar is not content addressed")
    verify_body_hash(sidecar, schema=SIDECAR_SCHEMA, label="teacher sidecar")
    role = bundle.sidecar_role(dataset.source_route)
    source = sidecar.get("source")
    prediction = sidecar.get("predictions")
    if (
        sidecar.get("campaign_id") != CAMPAIGN_ID
        or sidecar.get("classification") != role
        or sidecar.get("protected_test_predictions") is not False
        or not isinstance(source, dict)
        or source.get("route") != dataset.source_route
        or source.get("manifest_sha256") != dataset.source_manifest_sha256
        or source.get("npz_sha256") != dataset.source_npz_sha256
        or source.get("dataset") != dataset_identity(dataset)
        or source.get("split") != dataset.split
        or not isinstance(sidecar.get("teacher"), dict)
        or not isinstance(prediction, dict)
        or prediction.get("count") != len(dataset)
        or prediction.get("bytes") != len(dataset) * 4
        or prediction.get("dtype") != "little-endian-float32[n]"
    ):
        raise TrainingError("teacher sidecar binding is stale")
    _validate_teacher_identity(bundle, sidecar["teacher"])
    relative = _safe_relative(prediction.get("file"), "sidecar prediction file")
    prediction_path = (sidecar_path.parent / relative).resolve()
    try:
        prediction_path.relative_to(sidecar_path.parent.resolve())
    except ValueError as error:
        raise TrainingError("teacher sidecar prediction escapes its directory") from error
    raw = prediction_path.read_bytes()
    if (
        len(raw) != prediction["bytes"]
        or sha256_bytes(raw) != prediction.get("sha256")
        or prediction_path.name != f"{prediction['sha256']}.predictions.f32"
    ):
        raise TrainingError("teacher prediction payload changed")
    values = np.frombuffer(raw, dtype="<f4").copy()
    if (
        values.shape != (len(dataset),)
        or not np.all(np.isfinite(values))
        or np.any(np.abs(values) > 1.0)
    ):
        raise TrainingError("teacher prediction payload values are invalid")
    return values


def attach_teacher_sidecar(
    bundle: FrozenBundle, dataset: Dataset, sidecar_path: pathlib.Path
) -> Dataset:
    predictions = load_teacher_sidecar(bundle, dataset, sidecar_path)
    return dataclasses.replace(dataset, teacher_predictions=predictions)


def default_teacher_sidecar_routes(bundle: FrozenBundle) -> tuple[str, ...]:
    """Only routes needed by the deployable teacher-assisted Search arm."""

    return (
        *bundle.arm_train_routes("search-target"),
        *bundle.canonical_routes("train"),
        bundle.common_adjudicator_route(),
        *bundle.canonical_routes("validation"),
    )


def generate_teacher_sidecars(
    bundle: FrozenBundle,
    output_directory: pathlib.Path,
    *,
    predictor: Callable[[Dataset], np.ndarray] | None = None,
    teacher_identity: Mapping[str, object] | None = None,
) -> pathlib.Path:
    if predictor is None:
        accepted = AcceptedTeacherPredictor(bundle)
        predictor = accepted
        teacher_identity = accepted.runtime_identity
    if teacher_identity is None:
        raise TrainingError("teacher sidecar generation requires teacher identity")
    teacher_identity = _validate_teacher_identity(bundle, teacher_identity)
    entries = []
    for route in default_teacher_sidecar_routes(bundle):
        sidecar = generate_teacher_sidecar(
            bundle,
            route,
            output_directory,
            predictor,
            teacher_identity,
        )
        entries.append({
            "source_route": route,
            "sidecar": sidecar.name,
            "sha256": sha256_file(sidecar),
        })
    body: dict[str, object] = {
        "schema": SIDECAR_INDEX_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_bundle_body_sha256": bundle.body_sha256,
        "teacher": teacher_identity,
        "entries": entries,
        "allowed_classifications": [
            "train", "common-adjudicator", "canonical-validation"
        ],
        "protected_test_predictions": False,
    }
    document = body_hashed(body)
    path = _write_content_addressed(
        output_directory, canonical_json_bytes(document), ".sidecar-index.json"
    )
    load_sidecar_index(bundle, path)
    return path


def load_sidecar_index(
    bundle: FrozenBundle, index_path: pathlib.Path
) -> dict[str, pathlib.Path]:
    payload, index = _load_canonical_json(index_path, "teacher sidecar index")
    if index_path.name != f"{sha256_bytes(payload)}.sidecar-index.json":
        raise TrainingError("teacher sidecar index is not content addressed")
    verify_body_hash(index, schema=SIDECAR_INDEX_SCHEMA, label="sidecar index")
    entries = index.get("entries")
    if (
        index.get("campaign_id") != CAMPAIGN_ID
        or index.get("source_bundle_body_sha256") != bundle.body_sha256
        or index.get("protected_test_predictions") is not False
        or index.get("allowed_classifications")
        != ["train", "common-adjudicator", "canonical-validation"]
        or not isinstance(index.get("teacher"), dict)
        or not isinstance(entries, list)
    ):
        raise TrainingError("teacher sidecar index policy changed")
    expected_teacher = _validate_teacher_identity(bundle, index["teacher"])
    result: dict[str, pathlib.Path] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source_route", "sidecar", "sha256"
        }:
            raise TrainingError("teacher sidecar index entry is malformed")
        route = _safe_relative(entry["source_route"], "sidecar source route")
        bundle.sidecar_role(route)
        if route in result:
            raise TrainingError("teacher sidecar index repeats a source")
        sidecar_name = _safe_relative(entry["sidecar"], "sidecar file")
        sidecar_path = (index_path.parent / sidecar_name).resolve()
        try:
            sidecar_path.relative_to(index_path.parent.resolve())
        except ValueError as error:
            raise TrainingError("sidecar index entry escapes its directory") from error
        if (
            not sidecar_path.is_file()
            or sha256_file(sidecar_path) != entry["sha256"]
        ):
            raise TrainingError("teacher sidecar index entry changed")
        _sidecar_payload, sidecar = _load_canonical_json(
            sidecar_path, "indexed teacher sidecar"
        )
        verify_body_hash(
            sidecar, schema=SIDECAR_SCHEMA, label="indexed teacher sidecar"
        )
        if sidecar.get("teacher") != expected_teacher:
            raise TrainingError("teacher identity changed across sidecars")
        result[route] = sidecar_path
    if set(result) != set(default_teacher_sidecar_routes(bundle)):
        raise TrainingError("teacher sidecar index coverage is incomplete")
    return result


@dataclasses.dataclass(frozen=True)
class TrainingInputs:
    new: Dataset
    anchor: Dataset
    common_adjudicator: Dataset
    canonical_validation: Dataset
    source_routes: dict[str, tuple[str, ...]]
    paired_row_validation: dict[str, object] = dataclasses.field(
        default_factory=dict
    )
    split_isolation: dict[str, object] = dataclasses.field(default_factory=dict)
    input_audit: dict[str, object] = dataclasses.field(default_factory=dict)
    successor_rankings: SuccessorRankingLabels | None = None


def _array_identity(value: np.ndarray) -> str:
    value = np.asarray(value)
    contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_matched_train_rows(bundle: FrozenBundle) -> dict[str, object]:
    """Bind Search and Rank-4 control to byte-identical rows and weights."""

    search_routes = bundle.arm_train_routes("search-target")
    rank4_routes = bundle.arm_train_routes("rank4-control")
    pairs = []
    for phase, search_route, rank4_route in zip(
        ("pilot", "full"), search_routes, rank4_routes, strict=True
    ):
        search = load_shard(bundle, search_route)
        rank4 = load_shard(bundle, rank4_route)
        arrays = {
            "indptr": (search.indptr, rank4.indptr),
            "indices": (search.indices, rank4.indices),
            "group_ids": (search.group_ids, rank4.group_ids),
            "weights": (search.weights, rank4.weights),
        }
        if (
            len(search) != len(rank4)
            or any(not np.array_equal(first, second)
                   for first, second in arrays.values())
        ):
            raise TrainingError(
                f"matched Search/Rank-4 {phase} train row schedule changed"
            )
        pairs.append({
            "phase": phase,
            "samples": len(search),
            "search_manifest_sha256": search.source_manifest_sha256,
            "rank4_manifest_sha256": rank4.source_manifest_sha256,
            "byte_equal": {
                name: _array_identity(first)
                for name, (first, _second) in arrays.items()
            },
            "targets_intentionally_not_part_of_pairing": True,
        })
    return {
        "policy": "byte-identical-indptr-indices-group_ids-weights",
        "pairs": pairs,
        "total_samples": sum(int(pair["samples"]) for pair in pairs),
        "passed": True,
    }


def validate_unprotected_split_isolation(
    new: Dataset,
    anchor: Dataset,
    common_adjudicator: Dataset,
    canonical_validation: Dataset,
) -> dict[str, object]:
    """Reject group or exact/rotated/reflected train-validation leakage."""

    try:
        import jacek_replay_features as features
    except ImportError as error:
        raise TrainingError("split-isolation validator is unavailable") from error

    def feature_map(
        vertex_map: Sequence[int], edge_map: Sequence[int]
    ) -> np.ndarray:
        result = np.empty(INPUT_COUNT, dtype=np.uint16)
        result[:EDGE_COUNT] = np.asarray(edge_map, dtype=np.uint16)
        for vertex in range(VERTEX_COUNT):
            begin = EDGE_COUNT + vertex * VERTEX_CATEGORIES
            destination = EDGE_COUNT + int(vertex_map[vertex]) * VERTEX_CATEGORIES
            result[begin : begin + VERTEX_CATEGORIES] = np.arange(
                destination,
                destination + VERTEX_CATEGORIES,
                dtype=np.uint16,
            )
        return result

    reflected = feature_map(features.REFLECTED_VERTICES, features.REFLECTED_EDGES)
    rotated = feature_map(features.ROTATED_VERTICES, features.ROTATED_EDGES)
    reflected_rotated = reflected[rotated]
    maps = (reflected, rotated, reflected_rotated)

    def fingerprint(active: np.ndarray) -> bytes:
        # ``load_shard`` already performed the full row schema validation.
        # Mapping and sorting in NumPy avoids repeating the reference helper's
        # expensive 105x57 Python membership check for 1.36 million rows.
        identity = np.asarray(active, dtype="<u2").tobytes(order="C")
        variants = [identity]
        for transform in maps:
            transformed = np.sort(transform[active]).astype("<u2", copy=False)
            variants.append(transformed.tobytes(order="C"))
        return hashlib.sha256(min(variants)).digest()

    train_datasets = (new, anchor)
    validation_datasets = (common_adjudicator, canonical_validation)
    train_groups = {
        bytes(group)
        for dataset in train_datasets
        for group in dataset.group_ids
    }
    validation_groups = {
        bytes(group)
        for dataset in validation_datasets
        for group in dataset.group_ids
    }
    if train_groups.intersection(validation_groups):
        raise TrainingError("unprotected train/validation root group overlap")
    train_fingerprints: set[bytes] = set()
    for dataset in train_datasets:
        for row in range(len(dataset)):
            train_fingerprints.add(fingerprint(dataset.active_row(row)))
    validation_fingerprints: set[bytes] = set()
    for dataset in validation_datasets:
        for row in range(len(dataset)):
            canonical = fingerprint(dataset.active_row(row))
            if canonical in train_fingerprints:
                raise TrainingError(
                    "unprotected train/validation exact/rotate/reflect overlap"
                )
            validation_fingerprints.add(canonical)
    return {
        "policy": (
            "group-id-and-canonical-exact-rotate-reflect-"
            "rotated-reflect-train-vs-validation"
        ),
        "train_rows": sum(len(dataset) for dataset in train_datasets),
        "validation_rows": sum(len(dataset) for dataset in validation_datasets),
        "train_group_ids": len(train_groups),
        "validation_group_ids": len(validation_groups),
        "train_canonical_fingerprints": len(train_fingerprints),
        "validation_canonical_fingerprints": len(validation_fingerprints),
        "protected_tests_opened": False,
        "passed": True,
    }


def _input_audit_routes(bundle: FrozenBundle) -> dict[str, list[str]]:
    return {
        "search_train": list(bundle.arm_train_routes("search-target")),
        "rank4_train": list(bundle.arm_train_routes("rank4-control")),
        "canonical_train": list(bundle.canonical_routes("train")),
        "common_adjudicator": [bundle.common_adjudicator_route()],
        "canonical_validation": list(bundle.canonical_routes("validation")),
    }


def _input_audit_artifact_bindings(
    bundle: FrozenBundle, routes: Mapping[str, Sequence[str]]
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for role, values in routes.items():
        rows = []
        for relative in values:
            record = bundle.records.get(relative)
            if record is None:
                raise TrainingError("input-audit route has no bundle record")
            rows.append({
                "relative_path": relative,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            })
        result[role] = rows
    return result


def generate_input_audit(
    bundle: FrozenBundle, output_directory: pathlib.Path
) -> pathlib.Path:
    """Audit immutable row pairing and split isolation exactly once per bundle."""

    routes = _input_audit_routes(bundle)
    paired = validate_matched_train_rows(bundle)
    search = concatenate_datasets(
        [load_shard(bundle, route) for route in routes["search_train"]],
        split="train",
    )
    anchor = concatenate_datasets(
        [load_shard(bundle, route) for route in routes["canonical_train"]],
        split="train",
    )
    common = load_shard(bundle, routes["common_adjudicator"][0])
    canonical_validation = concatenate_datasets(
        [
            load_shard(bundle, route)
            for route in routes["canonical_validation"]
        ],
        split="validation",
    )
    isolation = validate_unprotected_split_isolation(
        search, anchor, common, canonical_validation
    )
    body: dict[str, object] = {
        "schema": INPUT_AUDIT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_bundle_body_sha256": bundle.body_sha256,
        "routes": routes,
        "artifact_bindings": _input_audit_artifact_bindings(bundle, routes),
        "datasets": {
            "search_train": dataset_identity(search),
            "canonical_train": dataset_identity(anchor),
            "common_adjudicator": dataset_identity(common),
            "canonical_validation": dataset_identity(canonical_validation),
        },
        "paired_row_validation": paired,
        "split_isolation": isolation,
        "protected_tests_opened": False,
        "runtime_source_paths_used": False,
    }
    document = body_hashed(body)
    path = _write_content_addressed(
        output_directory,
        canonical_json_bytes(document),
        ".input-audit.json",
    )
    validate_input_audit(bundle, path)
    return path


def validate_input_audit(
    bundle: FrozenBundle, audit_path: pathlib.Path
) -> dict[str, Any]:
    payload, audit = _load_canonical_json(audit_path, "compact input audit")
    if audit_path.name != f"{sha256_bytes(payload)}.input-audit.json":
        raise TrainingError("compact input audit is not content addressed")
    verify_body_hash(audit, schema=INPUT_AUDIT_SCHEMA, label="compact input audit")
    expected_routes = _input_audit_routes(bundle)
    if (
        set(audit) != {
            "schema", "campaign_id", "source_bundle_body_sha256", "routes",
            "artifact_bindings", "datasets", "paired_row_validation",
            "split_isolation", "protected_tests_opened",
            "runtime_source_paths_used", "body_sha256",
        }
        or audit.get("campaign_id") != CAMPAIGN_ID
        or audit.get("source_bundle_body_sha256") != bundle.body_sha256
        or audit.get("routes") != expected_routes
        or audit.get("artifact_bindings")
        != _input_audit_artifact_bindings(bundle, expected_routes)
        or audit.get("protected_tests_opened") is not False
        or audit.get("runtime_source_paths_used") is not False
        or audit.get("paired_row_validation", {}).get("passed") is not True
        or audit.get("split_isolation", {}).get("passed") is not True
        or not isinstance(audit.get("datasets"), dict)
    ):
        raise TrainingError("compact input audit binding changed")
    return audit


def load_training_inputs(
    bundle: FrozenBundle,
    arm: Arm | str,
    *,
    sidecar_index: pathlib.Path | None = None,
    input_audit: pathlib.Path | None = None,
    successor_labels: pathlib.Path | None = None,
) -> TrainingInputs:
    if isinstance(arm, str):
        try:
            arm = ARMS[arm]
        except KeyError as error:
            raise TrainingError("unknown training arm") from error
    if input_audit is None:
        raise TrainingError("training requires the immutable bundle-level input audit")
    audit = validate_input_audit(bundle, input_audit)
    sidecars: dict[str, pathlib.Path] = {}
    if arm.teacher_assisted:
        if sidecar_index is None:
            raise TrainingError("teacher-assisted arm requires the frozen sidecar index")
        sidecars = load_sidecar_index(bundle, sidecar_index)
    elif sidecar_index is not None:
        raise TrainingError("non-teacher arm must not consume teacher sidecars")

    def loaded(route: str) -> Dataset:
        dataset = load_shard(bundle, route)
        if arm.teacher_assisted:
            dataset = attach_teacher_sidecar(bundle, dataset, sidecars[route])
        return dataset

    new_routes = bundle.arm_train_routes(arm)
    anchor_routes = bundle.canonical_routes("train")
    common_route = bundle.common_adjudicator_route()
    canonical_validation_routes = bundle.canonical_routes("validation")
    inputs = TrainingInputs(
        new=concatenate_datasets(
            [loaded(route) for route in new_routes], split="train"
        ),
        anchor=concatenate_datasets(
            [loaded(route) for route in anchor_routes], split="train"
        ),
        common_adjudicator=loaded(common_route),
        canonical_validation=concatenate_datasets(
            [loaded(route) for route in canonical_validation_routes],
            split="validation",
        ),
        source_routes={
            "new": tuple(new_routes),
            "anchor": tuple(anchor_routes),
            "common_adjudicator": (common_route,),
            "canonical_validation": tuple(canonical_validation_routes),
        },
        paired_row_validation=dict(audit["paired_row_validation"]),
        split_isolation=dict(audit["split_isolation"]),
        input_audit={
            "file": input_audit.name,
            "sha256": sha256_file(input_audit),
            "body_sha256": audit["body_sha256"],
        },
        successor_rankings=(
            None
            if successor_labels is None
            else load_successor_ranking_labels(successor_labels, bundle)
        ),
    )
    expected = bundle.manifest.get("row_counts", {})
    if (
        len(inputs.new) != expected.get(
            "search" if arm.new_source == "search" else "rank4_control", {}
        ).get("train")
        or len(inputs.anchor) != expected.get("canonical", {}).get("train")
        or len(inputs.common_adjudicator) != expected.get("common_adjudicator")
        or len(inputs.canonical_validation)
        != expected.get("canonical", {}).get("validation")
    ):
        raise TrainingError("training input row counts changed")
    if inputs.common_adjudicator.split != "validation":
        raise TrainingError("common adjudicator split changed")
    return inputs


def _weighted_huber_loss_gradient(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float = float(HUBER_DELTA),
) -> tuple[float, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if (
        predictions.shape != targets.shape
        or predictions.shape != weights.shape
        or predictions.ndim != 1
        or len(predictions) == 0
        or not np.all(np.isfinite(predictions))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isfinite(delta)
        or delta <= 0.0
    ):
        raise TrainingError("weighted Huber inputs are invalid")
    difference = predictions - targets
    absolute = np.abs(difference)
    delta_value = np.float32(delta)
    losses = np.where(
        absolute <= delta_value,
        np.float32(0.5) * difference * difference,
        delta_value * (absolute - np.float32(0.5) * delta_value),
    ).astype(np.float32)
    denominator = max(
        float(np.sum(weights, dtype=np.float64)), np.finfo(np.float32).tiny
    )
    loss = float(np.sum(weights * losses, dtype=np.float64) / denominator)
    gradient = (
        weights * np.clip(difference, -delta_value, delta_value)
        / np.float32(denominator)
    ).astype(np.float32)
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise TrainingError("weighted Huber produced a nonfinite result")
    return loss, gradient


def arm_loss_gradient(
    arm: Arm | str,
    predictions: np.ndarray,
    stored_targets: np.ndarray,
    weights: np.ndarray,
    teacher_predictions: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, float]]:
    if isinstance(arm, str):
        try:
            arm = ARMS[arm]
        except KeyError as error:
            raise TrainingError("unknown loss arm") from error
    stored_loss, stored_gradient = _weighted_huber_loss_gradient(
        predictions, stored_targets, weights
    )
    if not arm.teacher_assisted:
        if teacher_predictions is not None:
            raise TrainingError("non-teacher arm received a teacher loss target")
        return stored_loss, stored_gradient, {
            "stored_target_weighted_huber": stored_loss,
            "objective_weighted_huber": stored_loss,
        }
    if teacher_predictions is None:
        raise TrainingError("teacher-assisted arm is missing its teacher target")
    teacher_loss, teacher_gradient = _weighted_huber_loss_gradient(
        predictions, teacher_predictions, weights
    )
    objective = 0.5 * stored_loss + 0.5 * teacher_loss
    gradient = np.float32(0.5) * stored_gradient + np.float32(0.5) * teacher_gradient
    return objective, gradient.astype(np.float32), {
        "stored_target_weighted_huber": stored_loss,
        "teacher_prediction_weighted_huber": teacher_loss,
        "objective_weighted_huber": objective,
        "stored_target_loss_share": 0.5,
        "teacher_prediction_loss_share": 0.5,
    }


def _parent_frame_sign(parent_mover: int, value_mover: int) -> np.float32:
    if (
        isinstance(parent_mover, bool)
        or isinstance(value_mover, bool)
        or parent_mover not in (0, 1)
        or value_mover not in (0, 1)
    ):
        raise TrainingError("successor ranking mover perspective is invalid")
    return np.float32(1.0 if parent_mover == value_mover else -1.0)


def _parent_frame_values(
    group: CompleteTurnGroup, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if (
        values.shape != (len(group.successors),)
        or not np.all(np.isfinite(values))
    ):
        raise TrainingError("successor ranking predictions are invalid")
    signs = np.asarray([
        _parent_frame_sign(group.parent_mover, successor.value_mover)
        for successor in group.successors
    ], dtype=np.float32)
    return (values * signs).astype(np.float32), signs


def _teacher_parent_values(group: CompleteTurnGroup) -> np.ndarray:
    values = np.asarray(
        [successor.teacher_value for successor in group.successors],
        dtype=np.float32,
    )
    parent, _signs = _parent_frame_values(group, values)
    return parent


def _deterministic_best(group: CompleteTurnGroup, values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(group.successors),) or not np.all(np.isfinite(values)):
        raise TrainingError("successor ranking best-action values are invalid")
    return min(
        range(len(group.successors)),
        key=lambda index: (
            -float(values[index]), group.successors[index].successor_id
        ),
    )


def _ranking_pairs(
    group: CompleteTurnGroup,
    *,
    pair_cap: int = RANKING_PAIR_CAP,
) -> tuple[int, tuple[int, ...], np.ndarray]:
    if isinstance(pair_cap, bool) or pair_cap != RANKING_PAIR_CAP:
        raise TrainingError(
            f"successor ranking pair cap must be exactly {RANKING_PAIR_CAP}"
        )
    teacher = _teacher_parent_values(group)
    best = _deterministic_best(group, teacher)
    if not group.successors_exhaustive:
        return best, (), np.asarray([], dtype=np.float32)
    candidates = []
    for index, successor in enumerate(group.successors):
        if index == best:
            continue
        gap = float(teacher[best] - teacher[index])
        if gap > 0.0:
            candidates.append((index, gap, successor.successor_id))
    candidates.sort(key=lambda row: (-row[1], row[2]))
    selected = tuple(row[0] for row in candidates[:pair_cap])
    gaps = np.asarray(
        [float(teacher[best] - teacher[index]) for index in selected],
        dtype=np.float32,
    )
    return best, selected, gaps


def _comparable_ranking_groups(
    groups: Sequence[CompleteTurnGroup],
) -> tuple[CompleteTurnGroup, ...]:
    return tuple(
        group
        for group in groups
        if group.successors_exhaustive and bool(_ranking_pairs(group)[1])
    )


def _density_weighted_ranking_groups(
    groups: Sequence[CompleteTurnGroup],
) -> tuple[tuple[CompleteTurnGroup, ...], dict[str, object]]:
    comparable = _comparable_ranking_groups(groups)
    expanded: list[CompleteTurnGroup] = []
    hard = 0
    for group in comparable:
        profile = _ranking_group_profile(group)
        if profile not in {"standard-v1", HARD_TEACHER_RANKING_PROFILE}:
            raise TrainingError("ranking group has an unknown density profile")
        multiplier = (
            HARD_STATE_DENSITY_MULTIPLIER
            if profile == HARD_TEACHER_RANKING_PROFILE else 1
        )
        hard += int(profile == HARD_TEACHER_RANKING_PROFILE)
        expanded.extend([group] * multiplier)
    return tuple(expanded), {
        "policy": "deterministic-expanded-ranking-schedule-v1",
        "hard_teacher_ranking_profile": HARD_TEACHER_RANKING_PROFILE,
        "hard_group_multiplier": HARD_STATE_DENSITY_MULTIPLIER,
        "unique_comparable_groups": len(comparable),
        "hard_unique_groups": hard,
        "scheduled_group_entries": len(expanded),
        "hard_scheduled_entries": hard * HARD_STATE_DENSITY_MULTIPLIER,
        "density_increased": hard > 0 and len(expanded) > len(comparable),
    }


def _ranking_group_profile(group: CompleteTurnGroup) -> str:
    work = group.evidence.get("work_budget")
    profile = (
        work.get("teacher_ranking_profile", "standard-v1")
        if isinstance(work, Mapping) else "standard-v1"
    )
    if profile not in {"standard-v1", HARD_TEACHER_RANKING_PROFILE}:
        raise TrainingError("ranking group has an unknown density profile")
    return str(profile)


def ranking_schedule_coverage(
    groups: Sequence[CompleteTurnGroup],
    schedule: Sequence[np.ndarray],
    *,
    epoch: int,
) -> dict[str, object]:
    """Prove one balanced, lossless pass over the weighted ranking pool."""

    if not groups or not schedule or epoch <= 0:
        raise TrainingError("ranking schedule coverage inputs are invalid")
    flattened = [int(index) for batch in schedule for index in batch]
    sizes = [len(batch) for batch in schedule]
    if (
        any(size <= 0 for size in sizes)
        or max(sizes) - min(sizes) > 1
        or sorted(flattened) != list(range(len(groups)))
    ):
        raise TrainingError("ranking schedule does not cover its weighted pool once")
    hard_entries = sum(
        _ranking_group_profile(groups[index]) == HARD_TEACHER_RANKING_PROFILE
        for index in flattened
    )
    unique_groups = {group.group_id for group in groups}
    executed_unique = {groups[index].group_id for index in flattened}
    report = {
        "policy": "balanced-full-weighted-pool-permutation-per-epoch-v1",
        "epoch": epoch,
        "scalar_batches": len(schedule),
        "weighted_pool_entries": len(groups),
        "executed_weighted_entries": len(flattened),
        "unique_groups": len(unique_groups),
        "executed_unique_groups": len(executed_unique),
        "hard_weighted_entries": sum(
            _ranking_group_profile(group) == HARD_TEACHER_RANKING_PROFILE
            for group in groups
        ),
        "executed_hard_weighted_entries": hard_entries,
        "complete_weighted_pool_permutations": 1,
        "dropped_weighted_entries": 0,
        "minimum_groups_per_scalar_batch": min(sizes),
        "maximum_groups_per_scalar_batch": max(sizes),
        "balanced_microbatches": max(sizes) - min(sizes) <= 1,
        "schedule_sha256": sha256_bytes(canonical_json_bytes([
            [int(index) for index in batch] for batch in schedule
        ])),
    }
    if (
        report["executed_weighted_entries"] != report["weighted_pool_entries"]
        or report["executed_unique_groups"] != report["unique_groups"]
        or report["executed_hard_weighted_entries"]
        != report["hard_weighted_entries"]
    ):
        raise TrainingError("ranking schedule coverage evidence is incomplete")
    return report


def validate_ranking_schedule_coverage(
    value: object,
    *,
    density: Mapping[str, object],
    epoch: int,
    scalar_batches: int,
    seed: int,
) -> dict[str, object]:
    expected_fields = {
        "policy", "epoch", "scalar_batches", "weighted_pool_entries",
        "executed_weighted_entries", "unique_groups", "executed_unique_groups",
        "hard_weighted_entries", "executed_hard_weighted_entries",
        "complete_weighted_pool_permutations", "dropped_weighted_entries",
        "minimum_groups_per_scalar_batch", "maximum_groups_per_scalar_batch",
        "balanced_microbatches", "schedule_sha256",
    }
    weighted = density.get("scheduled_group_entries")
    unique = density.get("unique_comparable_groups")
    hard = density.get("hard_scheduled_entries")
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (weighted, unique, hard, epoch, scalar_batches, seed)
        )
        or scalar_batches <= 0
        or weighted < scalar_batches
        or seed not in FIXED_SEEDS
    ):
        raise TrainingError("ranking schedule coverage receipt is malformed")
    smaller, remainder = divmod(int(weighted), scalar_batches)
    larger = smaller + int(remainder > 0)
    expected_schedule = successor_ranking_epoch_schedule(
        int(weighted), scalar_batches, seed=seed, epoch=epoch
    )
    expected_schedule_sha256 = sha256_bytes(canonical_json_bytes([
        [int(index) for index in batch] for batch in expected_schedule
    ]))
    if (
        value.get("policy")
        != "balanced-full-weighted-pool-permutation-per-epoch-v1"
        or value.get("epoch") != epoch
        or value.get("scalar_batches") != scalar_batches
        or value.get("weighted_pool_entries") != weighted
        or value.get("executed_weighted_entries") != weighted
        or value.get("unique_groups") != unique
        or value.get("executed_unique_groups") != unique
        or value.get("hard_weighted_entries") != hard
        or value.get("executed_hard_weighted_entries") != hard
        or value.get("complete_weighted_pool_permutations") != 1
        or value.get("dropped_weighted_entries") != 0
        or value.get("minimum_groups_per_scalar_batch") != smaller
        or value.get("maximum_groups_per_scalar_batch") != larger
        or value.get("balanced_microbatches") is not True
        or value.get("schedule_sha256") != expected_schedule_sha256
    ):
        raise TrainingError("ranking schedule did not cover its weighted pool exactly")
    return dict(value)


def pairwise_successor_ranking_loss_gradient(
    group: CompleteTurnGroup,
    predictions: np.ndarray,
    *,
    pair_cap: int = RANKING_PAIR_CAP,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Gap-weighted logistic best-vs-other loss in the parent's frame."""

    parent_predictions, signs = _parent_frame_values(group, predictions)
    best, alternatives, gaps = _ranking_pairs(group, pair_cap=pair_cap)
    gradient_parent = np.zeros(len(group.successors), dtype=np.float32)
    if not alternatives:
        return 0.0, gradient_parent, {
            "group_id": group.group_id,
            "teacher_best_successor_id": group.successors[best].successor_id,
            "pair_count": 0,
            "selected_successor_ids": [],
            "gap_weighting": "teacher-gap-normalized",
            "pair_cap": pair_cap,
            "successors_exhaustive": group.successors_exhaustive,
            "skipped_nonexhaustive": not group.successors_exhaustive,
        }
    denominator = float(np.sum(gaps, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise TrainingError("successor ranking teacher gaps are invalid")
    gap_weights = (gaps / np.float32(denominator)).astype(np.float32)
    loss = 0.0
    for weight, other in zip(gap_weights, alternatives, strict=True):
        margin = float(parent_predictions[best] - parent_predictions[other])
        pair_loss = float(np.logaddexp(0.0, -margin))
        derivative = -1.0 / (1.0 + math.exp(margin))
        loss += float(weight) * pair_loss
        gradient_parent[best] += np.float32(float(weight) * derivative)
        gradient_parent[other] -= np.float32(float(weight) * derivative)
    gradient = (gradient_parent * signs).astype(np.float32)
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise TrainingError("successor ranking loss produced a nonfinite result")
    return loss, gradient, {
        "group_id": group.group_id,
        "teacher_best_successor_id": group.successors[best].successor_id,
        "pair_count": len(alternatives),
        "selected_successor_ids": [
            group.successors[index].successor_id for index in alternatives
        ],
        "teacher_gaps": [float(value) for value in gaps],
        "normalized_gap_weights": [float(value) for value in gap_weights],
        "gap_weighting": "teacher-gap-normalized",
        "pair_cap": pair_cap,
        "successors_exhaustive": True,
        "skipped_nonexhaustive": False,
    }


def ranking_microbatch_loss_gradient(
    groups: Sequence[CompleteTurnGroup],
    predictions: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Average normalized group objectives without changing the external lambda."""

    if not groups:
        raise TrainingError("successor ranking microbatch is empty")
    expected_predictions = sum(len(group.successors) for group in groups)
    predictions = np.asarray(predictions, dtype=np.float32)
    if (
        predictions.shape != (expected_predictions,)
        or not np.all(np.isfinite(predictions))
    ):
        raise TrainingError("successor ranking microbatch predictions are invalid")
    output_gradient = np.zeros(expected_predictions, dtype=np.float32)
    losses = []
    pairs = 0
    offset = 0
    scale = np.float32(1.0 / len(groups))
    for group in groups:
        stop = offset + len(group.successors)
        loss, gradient, report = pairwise_successor_ranking_loss_gradient(
            group, predictions[offset:stop]
        )
        if (
            report.get("successors_exhaustive") is not True
            or report.get("skipped_nonexhaustive") is not False
            or int(report.get("pair_count", 0)) <= 0
        ):
            raise TrainingError(
                "successor ranking microbatch contains an excluded zero-pair group"
            )
        output_gradient[offset:stop] = gradient * scale
        losses.append(loss)
        pairs += int(report["pair_count"])
        offset = stop
    loss = float(np.mean(losses, dtype=np.float64))
    if (
        offset != expected_predictions
        or not math.isfinite(loss)
        or not np.all(np.isfinite(output_gradient))
    ):
        raise TrainingError("successor ranking microbatch objective is invalid")
    return loss, output_gradient, {
        "groups": len(groups),
        "successors": expected_predictions,
        "pairs": pairs,
        "group_objective": "mean-of-gap-normalized-pairwise-losses",
        "lambda_application": "once-after-group-mean",
    }


class AdamW:
    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0.0
            or not math.isfinite(weight_decay)
            or weight_decay < 0.0
        ):
            raise TrainingError("AdamW configuration is invalid")
        self.learning_rate = np.float32(learning_rate)
        self.weight_decay = np.float32(weight_decay)
        self.first = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in parameters.items()
        }
        self.second = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in parameters.items()
        }
        self.step = 0

    def update(
        self,
        parameters: Mapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        if set(parameters) != set(self.first) or set(gradients) != set(self.first):
            raise TrainingError("AdamW parameter roster changed")
        self.step += 1
        correction_one = np.float32(1.0 - 0.9**self.step)
        correction_two = np.float32(1.0 - 0.999**self.step)
        for name in sorted(parameters):
            gradient = np.asarray(gradients[name], dtype=np.float32)
            if gradient.shape != parameters[name].shape or not np.all(np.isfinite(gradient)):
                raise TrainingError("AdamW gradient is invalid")
            self.first[name] = (
                np.float32(0.9) * self.first[name]
                + np.float32(0.1) * gradient
            ).astype(np.float32)
            self.second[name] = (
                np.float32(0.999) * self.second[name]
                + np.float32(0.001) * gradient * gradient
            ).astype(np.float32)
            first = self.first[name] / correction_one
            second = self.second[name] / correction_two
            parameters[name] *= np.float32(
                1.0 - self.learning_rate * self.weight_decay
            )
            parameters[name] -= self.learning_rate * first / (
                np.sqrt(second).astype(np.float32) + np.float32(1e-8)
            )
            if not np.all(np.isfinite(parameters[name])):
                raise TrainingError("AdamW update produced a nonfinite parameter")


def _network_gradients(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    active: Sequence[np.ndarray],
    cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_gradient: np.ndarray,
    effective: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output_gradient = np.asarray(output_gradient, dtype=np.float32)
    first_pre, first, second_pre, second, output_pre = cache
    if output_gradient.shape != output_pre.shape:
        raise TrainingError("compact output gradient shape changed")
    output_pre_gradient = output_gradient * fast_tanh_derivative(output_pre)
    gradients: dict[str, np.ndarray] = {
        "w3": np.asarray(second.T @ output_pre_gradient, dtype=np.float32),
    }
    second_gradient = (
        output_pre_gradient[:, None] * effective["w3"][None, :]
    ).astype(np.float32)
    second_pre_gradient = (
        second_gradient * second_activation_derivative(second_pre)
    )
    gradients["w2"] = np.asarray(first.T @ second_pre_gradient, dtype=np.float32)
    first_gradient = np.asarray(
        second_pre_gradient @ effective["w2"].T, dtype=np.float32
    )
    first_pre_gradient = (
        first_gradient * first_activation_derivative(first_pre)
    )
    gradients["w1"] = np.zeros_like(parameters["w1"], dtype=np.float32)
    for row, indices in enumerate(active):
        np.add.at(gradients["w1"], indices, first_pre_gradient[row])
    return gradients


def _train_mixed_batch(
    parameters: dict[str, np.ndarray],
    architecture: Architecture,
    arm: Arm,
    optimizer: AdamW,
    inputs: TrainingInputs,
    new_rows: np.ndarray,
    anchor_rows: np.ndarray,
    *,
    fixed_scales: Mapping[str, object] | None = None,
    ranking_group: CompleteTurnGroup | None = None,
    ranking_groups: Sequence[CompleteTurnGroup] | None = None,
    ranking_weight: float = 0.0,
) -> float:
    if (
        new_rows.shape != (NEW_ROWS_PER_BATCH,)
        or anchor_rows.shape != (ANCHOR_ROWS_PER_BATCH,)
    ):
        raise TrainingError("training batch does not have exactly 64/192 rows")
    active = (
        *inputs.new.active_rows(new_rows),
        *inputs.anchor.active_rows(anchor_rows),
    )
    targets = np.concatenate(
        (inputs.new.targets[new_rows], inputs.anchor.targets[anchor_rows])
    ).astype(np.float32, copy=False)
    weights = independently_normalized_mixed_weights(
        inputs.new.weights[new_rows], inputs.anchor.weights[anchor_rows]
    )
    teacher = None
    if arm.teacher_assisted:
        if (
            inputs.new.teacher_predictions is None
            or inputs.anchor.teacher_predictions is None
        ):
            raise TrainingError("teacher-assisted training sidecars are incomplete")
        teacher = np.concatenate((
            inputs.new.teacher_predictions[new_rows],
            inputs.anchor.teacher_predictions[anchor_rows],
        )).astype(np.float32, copy=False)
    quantized = (
        quantize_fixed(parameters, architecture, fixed_scales)
        if fixed_scales is not None
        else None
    )
    ranking_weight = _ranking_weight(ranking_weight)
    if ranking_group is not None and ranking_groups is not None:
        raise TrainingError("ranking group and microbatch cannot both be supplied")
    ranking_microbatch = tuple(
        (ranking_group,) if ranking_group is not None else (ranking_groups or ())
    )
    if bool(ranking_microbatch) != (ranking_weight > 0.0):
        raise TrainingError(
            "positive successor ranking weight requires one nonempty group microbatch"
        )
    predictions, cache = forward(
        parameters, architecture, active, quantized=quantized
    )
    loss, output_gradient, _ = arm_loss_gradient(
        arm, predictions, targets, weights, teacher
    )
    effective = quantized.effective() if quantized is not None else parameters
    gradients = _network_gradients(
        parameters,
        architecture,
        active,
        cache,
        output_gradient,
        effective,
    )
    objective = loss
    if ranking_microbatch:
        ranking_active = tuple(
            successor.active
            for group in ranking_microbatch
            for successor in group.successors
        )
        ranking_predictions, ranking_cache = forward(
            parameters,
            architecture,
            ranking_active,
            quantized=quantized,
        )
        ranking_loss, ranking_output_gradient, _ranking_report = (
            ranking_microbatch_loss_gradient(
                ranking_microbatch, ranking_predictions
            )
        )
        ranking_gradients = _network_gradients(
            parameters,
            architecture,
            ranking_active,
            ranking_cache,
            ranking_output_gradient * np.float32(ranking_weight),
            effective,
        )
        for name in gradients:
            gradients[name] += ranking_gradients[name]
        objective += ranking_weight * ranking_loss
    norm = math.sqrt(
        sum(
            float(np.sum(value * value, dtype=np.float64))
            for value in gradients.values()
        )
    )
    if not math.isfinite(norm):
        raise TrainingError("compact gradient norm is nonfinite")
    if norm > GRADIENT_CLIP:
        scale = np.float32(GRADIENT_CLIP / norm)
        for gradient in gradients.values():
            gradient *= scale
    optimizer.update(parameters, gradients)
    return float(objective)


def predict_dataset(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    dataset: Dataset,
    *,
    quantized: QuantizedWeights | None = None,
    batch_size: int = 4_096,
) -> np.ndarray:
    if batch_size <= 0 or len(dataset) <= 0:
        raise TrainingError("metric prediction arguments are invalid")
    predictions = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        predictions[start:stop], _ = forward(
            parameters,
            architecture,
            dataset.active_rows(range(start, stop)),
            quantized=quantized,
        )
    return predictions


def successor_ranking_metrics(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    groups: Sequence[CompleteTurnGroup],
    *,
    quantized: QuantizedWeights | None = None,
) -> dict[str, float | int | bool]:
    if not groups:
        raise TrainingError("successor ranking metrics require nonempty groups")
    agreements = 0
    regrets = []
    losses = []
    pair_count = 0
    flips = 0
    comparable_groups = 0
    singleton_groups = 0
    skipped_nonexhaustive_groups = 0
    skipped_tied_groups = 0
    for group in groups:
        if len(group.successors) == 1:
            singleton_groups += 1
        if not group.successors_exhaustive:
            skipped_nonexhaustive_groups += 1
            continue
        _best, alternatives, _gaps = _ranking_pairs(group)
        if not alternatives:
            skipped_tied_groups += int(len(group.successors) > 1)
            continue
        comparable_groups += 1
        active = tuple(successor.active for successor in group.successors)
        float_raw, _float_cache = forward(
            parameters, architecture, active, quantized=None
        )
        evaluated_raw = float_raw
        if quantized is not None:
            evaluated_raw, _quantized_cache = forward(
                parameters, architecture, active, quantized=quantized
            )
        teacher_parent = _teacher_parent_values(group)
        float_parent, _float_signs = _parent_frame_values(group, float_raw)
        evaluated_parent, _evaluated_signs = _parent_frame_values(
            group, evaluated_raw
        )
        teacher_best = _deterministic_best(group, teacher_parent)
        float_best = _deterministic_best(group, float_parent)
        evaluated_best = _deterministic_best(group, evaluated_parent)
        agreements += int(evaluated_best == teacher_best)
        flips += int(evaluated_best != float_best)
        regrets.append(float(
            teacher_parent[teacher_best] - teacher_parent[evaluated_best]
        ))
        loss, _gradient, report = pairwise_successor_ranking_loss_gradient(
            group, evaluated_raw
        )
        losses.append(loss)
        pair_count += int(report["pair_count"])
    group_count = len(groups)
    denominator = max(1, comparable_groups)
    report: dict[str, float | int | bool] = {
        "groups": group_count,
        "comparable_groups": comparable_groups,
        "singleton_groups": singleton_groups,
        "skipped_nonexhaustive_groups": skipped_nonexhaustive_groups,
        "skipped_tied_groups": skipped_tied_groups,
        "pairs": pair_count,
        "top1_agreement": float(agreements / denominator),
        "mean_teacher_regret": (
            0.0 if not regrets else float(np.mean(regrets, dtype=np.float64))
        ),
        "pairwise_loss": (
            0.0 if not losses else float(np.mean(losses, dtype=np.float64))
        ),
        "float_vs_quantized_action_flips": flips,
        "float_vs_quantized_action_flip_rate": float(flips / denominator),
        "quantized_comparison": quantized is not None,
        "pair_cap": RANKING_PAIR_CAP,
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in report.values()
    ):
        raise TrainingError("successor ranking metric is nonfinite")
    return report


def metrics_from_predictions(
    predictions: np.ndarray, dataset: Dataset, arm: Arm | str
) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.shape != (len(dataset),):
        raise TrainingError("metric predictions have a wrong shape")
    stored_loss, _ = _weighted_huber_loss_gradient(
        predictions, dataset.targets, dataset.weights
    )
    if isinstance(arm, str):
        arm = ARMS[arm]
    teacher_loss = None
    objective_loss = stored_loss
    if arm.teacher_assisted:
        if dataset.teacher_predictions is None:
            raise TrainingError("teacher-assisted metrics lack their sidecar")
        teacher_loss, _ = _weighted_huber_loss_gradient(
            predictions, dataset.teacher_predictions, dataset.weights
        )
        objective_loss = 0.5 * stored_loss + 0.5 * teacher_loss
    sign = float(np.mean(
        (predictions >= 0.0) == (dataset.targets >= 0.0)
    ))
    if len(dataset) > 1 and np.std(predictions) > 0.0 and np.std(dataset.targets) > 0.0:
        correlation = float(np.corrcoef(predictions, dataset.targets)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    report: dict[str, float | int] = {
        "samples": len(dataset),
        "weighted_huber": stored_loss,
        "objective_weighted_huber": objective_loss,
        "sign_accuracy": sign,
        "correlation": correlation,
        "mae": float(np.mean(np.abs(predictions - dataset.targets))),
        "prediction_mean": float(np.mean(predictions)),
    }
    if teacher_loss is not None:
        report["teacher_prediction_weighted_huber"] = teacher_loss
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in report.values()
    ):
        raise TrainingError("metric report contains a nonfinite value")
    return report


def evaluate_validation_pair(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    inputs: TrainingInputs,
    arm: Arm,
    *,
    quantized: QuantizedWeights | None = None,
    ranking_weight: float = 0.0,
) -> dict[str, dict[str, Any]]:
    ranking_weight = _ranking_weight(ranking_weight)
    if ranking_weight > 0.0 and inputs.successor_rankings is None:
        raise TrainingError("successor ranking labels are required by the loss weight")
    if (
        ranking_weight > 0.0
        and inputs.successor_rankings is not None
        and not _comparable_ranking_groups(inputs.successor_rankings.validation)
    ):
        raise TrainingError("positive ranking loss has no comparable validation groups")
    report: dict[str, dict[str, Any]] = {
        "common_adjudicator": metrics_from_predictions(
            predict_dataset(
                parameters,
                architecture,
                inputs.common_adjudicator,
                quantized=quantized,
            ),
            inputs.common_adjudicator,
            arm,
        ),
        "canonical_validation": metrics_from_predictions(
            predict_dataset(
                parameters,
                architecture,
                inputs.canonical_validation,
                quantized=quantized,
            ),
            inputs.canonical_validation,
            arm,
        ),
    }
    if inputs.successor_rankings is not None:
        report["successor_ranking"] = {
            **successor_ranking_metrics(
                parameters,
                architecture,
                inputs.successor_rankings.validation,
                quantized=quantized,
            ),
            "loss_weight": ranking_weight,
        }
    return report


def _validation_key(
    report: Mapping[str, Mapping[str, float | int]]
) -> tuple[float, ...]:
    common = report["common_adjudicator"]
    canonical = report["canonical_validation"]
    result = (
        float(common["objective_weighted_huber"]),
        float(canonical["objective_weighted_huber"]),
        -float(common["sign_accuracy"]),
        -float(canonical["sign_accuracy"]),
    )
    ranking = report.get("successor_ranking")
    if ranking is None or float(ranking.get("loss_weight", 0.0)) == 0.0:
        return result
    return (
        *result,
        float(ranking["mean_teacher_regret"]),
        -float(ranking["top1_agreement"]),
        float(ranking["float_vs_quantized_action_flip_rate"]),
        float(ranking["pairwise_loss"]),
    )


def _qat_validation_key(
    report: Mapping[str, Mapping[str, float | int]],
    profile: QATProfile,
) -> tuple[float, ...]:
    """Keep standard ordering exact; target action flips in the refined arm."""

    profile = resolve_qat_profile(profile)
    base = _validation_key(report)
    if profile.name == STANDARD_QAT_PROFILE:
        return base
    ranking = report.get("successor_ranking")
    if not isinstance(ranking, Mapping):
        raise TrainingError(
            "refined adaptive QAT requires successor-ranking validation metrics"
        )
    try:
        result = (
            float(ranking["float_vs_quantized_action_flip_rate"]),
            float(ranking["mean_teacher_regret"]),
            -float(ranking["top1_agreement"]),
            *base,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingError(
            "refined adaptive QAT ranking metrics are incomplete"
        ) from error
    if any(not math.isfinite(value) for value in result):
        raise TrainingError("refined adaptive QAT ranking metrics are nonfinite")
    return result


@dataclasses.dataclass(frozen=True)
class FloatTrainingResult:
    parameters: dict[str, np.ndarray]
    epoch: int
    metrics: dict[str, dict[str, float | int]]
    report: dict[str, object]


def _parameter_update_evidence(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    if set(before) != {"w1", "w2", "w3"} or set(after) != set(before):
        raise TrainingError("per-layer update evidence tensor roster changed")
    report: dict[str, dict[str, object]] = {}
    for name in ("w1", "w2", "w3"):
        first = np.asarray(before[name], dtype="<f4")
        last = np.asarray(after[name], dtype="<f4")
        if first.shape != last.shape or not np.all(np.isfinite(last)):
            raise TrainingError("per-layer update evidence shape/value changed")
        delta = np.asarray(last - first, dtype=np.float32)
        report[name] = {
            "parameters": int(first.size),
            "changed_parameters": int(np.count_nonzero(first != last)),
            "changed": bool(np.any(first != last)),
            "l2_delta": float(np.linalg.norm(delta.astype(np.float64))),
            "maximum_absolute_delta": float(
                np.max(np.abs(delta)) if delta.size else 0.0
            ),
            "before_sha256": sha256_bytes(first.tobytes(order="C")),
            "after_sha256": sha256_bytes(last.tobytes(order="C")),
        }
    return report


def _parameter_identity(
    parameters: Mapping[str, np.ndarray], architecture: Architecture,
) -> dict[str, object]:
    normalized = _validate_parameters(parameters, architecture)
    layers = {}
    for name in ("w1", "w2", "w3"):
        value = np.asarray(normalized[name], dtype="<f4")
        layers[name] = {
            "shape": list(value.shape),
            "dtype": "little-endian-float32",
            "sha256": sha256_bytes(value.tobytes(order="C")),
        }
    return {
        "architecture": architecture.name,
        "dimensions": list(architecture.dimensions),
        "layers": layers,
    }


def train_float_seed(
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    seed: int,
    *,
    maximum_epochs: int = MAX_FLOAT_EPOCHS,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    ranking_weight: float = 0.0,
    initial_parameters: Mapping[str, np.ndarray] | None = None,
) -> FloatTrainingResult:
    if seed not in FIXED_SEEDS:
        raise TrainingError("compact training requires one of the three fixed seeds")
    if maximum_epochs <= 0 or maximum_epochs > MAX_FLOAT_EPOCHS or patience <= 0:
        raise TrainingError("float training epoch configuration is invalid")
    ranking_weight = _ranking_weight(ranking_weight)
    successor_mode = inputs.successor_rankings is not None
    if successor_mode:
        if (
            architecture.name != "capacity-12x8"
            or maximum_epochs != RANKING_FLOAT_EPOCHS
            or not math.isfinite(learning_rate)
            or not 0.0 < learning_rate <= RANKING_FLOAT_LEARNING_RATE
            or initial_parameters is None
        ):
            raise TrainingError(
                "successor ranking requires bound 12x8 initialization, one "
                "float epoch, and learning rate at most 6e-5"
            )
    elif initial_parameters is not None:
        raise TrainingError("legacy scalar training cannot inject an initial checkpoint")
    all_ranking_groups = (
        () if inputs.successor_rankings is None else inputs.successor_rankings.train
    )
    ranking_groups, density_report = _density_weighted_ranking_groups(
        all_ranking_groups
    )
    if ranking_weight > 0.0 and not ranking_groups:
        raise TrainingError("positive ranking loss has no training groups")
    coverage_epoch = anchor_coverage_complete_epoch(len(inputs.new), len(inputs.anchor))
    if not successor_mode and maximum_epochs < coverage_epoch:
        raise TrainingError("float training cannot cover the complete anchor stream")
    parameters = (
        initialize_parameters(architecture, seed)
        if initial_parameters is None
        else {
            name: value.copy()
            for name, value in _validate_parameters(
                initial_parameters, architecture
            ).items()
        }
    )
    starting_parameters = {
        name: value.copy() for name, value in parameters.items()
    }
    optimizer = AdamW(
        parameters, learning_rate=learning_rate, weight_decay=weight_decay
    )
    best_parameters: dict[str, np.ndarray] | None = None
    best_metrics: dict[str, dict[str, float | int]] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    last_progress_epoch = coverage_epoch
    history: list[dict[str, object]] = []
    for epoch in range(1, maximum_epochs + 1):
        losses = []
        batch_count = math.ceil(len(inputs.new) / NEW_ROWS_PER_BATCH)
        ranking_schedule = (
            None
            if ranking_weight == 0.0
            else successor_ranking_epoch_schedule(
                len(ranking_groups), batch_count, seed=seed, epoch=epoch
            )
        )
        ranking_coverage = (
            None
            if ranking_schedule is None
            else ranking_schedule_coverage(
                ranking_groups, ranking_schedule, epoch=epoch
            )
        )
        for batch_index, (new_rows, anchor_rows) in enumerate(mixed_epoch_batches(
            len(inputs.new), len(inputs.anchor), seed=seed, epoch=epoch
        )):
            losses.append(_train_mixed_batch(
                parameters,
                architecture,
                arm,
                optimizer,
                inputs,
                new_rows,
                anchor_rows,
                ranking_groups=(
                    None
                    if ranking_schedule is None
                    else tuple(
                        ranking_groups[int(index)]
                        for index in ranking_schedule[batch_index]
                    )
                ),
                ranking_weight=ranking_weight,
            ))
        validation = evaluate_validation_pair(
            parameters,
            architecture,
            inputs,
            arm,
            ranking_weight=ranking_weight,
        )
        coverage = mixed_epoch_coverage(len(inputs.new), len(inputs.anchor), epoch)
        complete = (
            True
            if successor_mode
            else coverage["anchor"]["complete_permutations"] >= 1
        )
        key = _validation_key(validation)
        eligible = complete and (best_key is None or key < best_key)
        history.append({
            "epoch": epoch,
            "training_objective_weighted_huber": float(np.mean(losses)),
            "validation": validation,
            "coverage": coverage,
            "ranking_schedule_coverage": ranking_coverage,
            "eligible": eligible,
        })
        if eligible:
            best_key = key
            best_epoch = epoch
            best_parameters = {
                name: value.copy() for name, value in parameters.items()
            }
            best_metrics = validation
            last_progress_epoch = epoch
        if complete and epoch - last_progress_epoch >= patience:
            break
    minimum_epoch = 1 if successor_mode else coverage_epoch
    if best_parameters is None or best_metrics is None or best_epoch < minimum_epoch:
        raise TrainingError("float training produced no selectable checkpoint")
    training_report: dict[str, object] = {
        "seed": seed,
        "best_float_epoch": best_epoch,
        "anchor_coverage_complete_epoch": coverage_epoch,
        "history": history,
        "optimizer": {
                "name": "adamw",
                "batch_size": BATCH_SIZE,
                "maximum_epochs": maximum_epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_norm_clip": GRADIENT_CLIP,
        },
        "batching": {
            "new_rows_per_batch": NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
            "new_loss_share": 0.25,
            "anchor_loss_share": 0.75,
            "sources_normalized_separately": True,
            "anchor_stream": "continuous-no-repeat-until-permutation-complete",
        },
        "validation": best_metrics,
    }
    if successor_mode:
        training_report.update({
            "selected_epoch_anchor_coverage": mixed_epoch_coverage(
                len(inputs.new), len(inputs.anchor), best_epoch
            ),
            "initialization": {
                "kind": "frozen-float-checkpoint",
                "seed_affects": "row-order-only",
                "parameters": _parameter_identity(
                    starting_parameters, architecture
                ),
            },
            "successor_ranking": {
                "labels_present": True,
                "loss_active": ranking_weight > 0.0,
                "loss_weight": ranking_weight,
                "composition": "scalar-loss-plus-lambda-ranking-loss",
                "group_microbatch_objective": (
                    "mean-of-gap-normalized-group-losses"
                ),
                "ranking_lambda_application": "once-after-group-mean",
                "epoch_schedule": (
                    "balanced-full-weighted-pool-permutation-per-epoch-v1"
                ),
                "pair_cap": RANKING_PAIR_CAP,
                "gap_weighting": "teacher-gap-normalized",
                "train_groups": len(all_ranking_groups),
                "comparable_train_groups": density_report[
                    "unique_comparable_groups"
                ],
                "hard_state_density": density_report,
                "weighted_group_entries_per_epoch": len(ranking_groups),
                "full_weighted_pool_coverage_each_active_epoch": (
                    ranking_weight > 0.0
                ),
                "selected_epoch_schedule_coverage": history[
                    best_epoch - 1
                ]["ranking_schedule_coverage"],
                "skipped_nonexhaustive_train_groups": sum(
                    not group.successors_exhaustive
                    for group in all_ranking_groups
                ),
                "skipped_zero_pair_train_groups": sum(
                    group.successors_exhaustive and not bool(
                        _ranking_pairs(group)[1]
                    )
                    for group in all_ranking_groups
                ),
                "validation_groups": (
                    len(inputs.successor_rankings.validation)
                ),
                "float_warmup_epochs": RANKING_FLOAT_EPOCHS,
                "float_learning_rate": learning_rate,
                "legacy_full_anchor_pass_required": False,
            },
            "per_layer_update_evidence": _parameter_update_evidence(
                starting_parameters, best_parameters
            ),
        })
    return FloatTrainingResult(
        parameters=best_parameters,
        epoch=best_epoch,
        metrics=best_metrics,
        report=training_report,
    )


def select_fixed_scales(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    inputs: TrainingInputs,
    arm: Arm,
    *,
    ranking_weight: float = 0.0,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
) -> tuple[QuantizedWeights, dict[str, object]]:
    parameters = _validate_parameters(parameters, architecture)
    profile = resolve_qat_profile(qat_profile)
    candidates = {
        name: robust_scale_candidates(
            parameters[name], quantiles=profile.scale_quantiles
        )
        for name in ("w1", "w2", "w3")
    }
    requested = {name: values[-1] for name, values in candidates.items()}
    trials: list[dict[str, object]] = []
    for search_pass in range(1, profile.coordinate_search_passes + 1):
        for name in ("w1", "w2", "w3"):
            best: tuple[tuple[float, ...], np.float32] | None = None
            for candidate in candidates[name]:
                trial_scales = dict(requested)
                trial_scales[name] = candidate
                quantized = quantize_fixed(parameters, architecture, trial_scales)
                metrics = evaluate_validation_pair(
                    parameters,
                    architecture,
                    inputs,
                    arm,
                    quantized=quantized,
                    ranking_weight=ranking_weight,
                )
                key = (*_qat_validation_key(metrics, profile), float(candidate))
                trials.append({
                    "pass": search_pass,
                    "layer": name,
                    "requested_scale": float(candidate),
                    "scales": {
                        layer: float(quantized.scales[layer])
                        for layer in ("w1", "w2", "w3")
                    },
                    "validation": metrics,
                })
                if best is None or key < best[0]:
                    best = (key, candidate)
            assert best is not None
            requested[name] = best[1]
    refinement_trials = 0
    for refinement_pass in range(1, profile.local_refinement_passes + 1):
        for name in ("w1", "w2", "w3"):
            best = None
            refined = _refined_scale_candidates(
                requested[name], profile.local_refinement_multipliers
            )
            for candidate in refined:
                trial_scales = dict(requested)
                trial_scales[name] = candidate
                quantized = quantize_fixed(parameters, architecture, trial_scales)
                metrics = evaluate_validation_pair(
                    parameters,
                    architecture,
                    inputs,
                    arm,
                    quantized=quantized,
                    ranking_weight=ranking_weight,
                )
                key = (*_qat_validation_key(metrics, profile), float(candidate))
                trials.append({
                    "stage": "local-refinement",
                    "refinement_pass": refinement_pass,
                    "layer": name,
                    "requested_scale": float(candidate),
                    "scales": {
                        layer: float(quantized.scales[layer])
                        for layer in ("w1", "w2", "w3")
                    },
                    "validation": metrics,
                })
                refinement_trials += 1
                if best is None or key < best[0]:
                    best = (key, candidate)
            assert best is not None
            requested[name] = best[1]
    selected = quantize_fixed(parameters, architecture, requested)
    selected_metrics = evaluate_validation_pair(
        parameters,
        architecture,
        inputs,
        arm,
        quantized=selected,
        ranking_weight=ranking_weight,
    )
    return selected, {
        "scheme": (
            "fixed-symmetric-3bit-validation-coordinate-search-"
            "lower-rank-robust-quantiles/v1"
            if profile.name == STANDARD_QAT_PROFILE
            else "refined-adaptive-symmetric-3bit-validation-scale-search/v1"
        ),
        "qat_profile": profile.name,
        "qat_profile_contract": qat_profile_contract(profile),
        "passes": profile.coordinate_search_passes,
        "local_refinement_passes": profile.local_refinement_passes,
        "local_refinement_trials": refinement_trials,
        "maximum_candidate_quantile": (
            f"{profile.scale_quantiles[-1][0]}-lower-rank"
        ),
        "max_abs_is_not_a_scale_candidate": True,
        "candidates": {
            name: [float(value) for value in values]
            for name, values in candidates.items()
        },
        "selected_scales": {
            name: float(selected.scales[name]) for name in ("w1", "w2", "w3")
        },
        "selected_validation": selected_metrics,
        "trials": trials,
    }


def _adapt_fixed_scales(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    inputs: TrainingInputs,
    arm: Arm,
    starting_scales: Mapping[str, object],
    profile: QATProfile,
    *,
    qat_epoch: int,
    ranking_weight: float,
) -> tuple[QuantizedWeights, dict[str, object]]:
    """Locally reselect scales after one QAT epoch from current master weights."""

    parameters = _validate_parameters(parameters, architecture)
    profile = resolve_qat_profile(profile)
    if not profile.adapt_scales_after_each_epoch:
        raise TrainingError("fixed QAT profile cannot perform adaptive reselection")
    if not 1 <= qat_epoch <= QAT_EPOCHS:
        raise TrainingError("adaptive scale epoch is outside the QAT schedule")
    starting = quantize_fixed(parameters, architecture, starting_scales)
    requested = dict(starting.scales)
    named_quantiles = {
        name: (name, numerator, denominator)
        for name, numerator, denominator in profile.scale_quantiles
    }
    trials: list[dict[str, object]] = []
    candidate_evidence: dict[str, list[float]] = {}
    for search_pass in range(1, profile.adaptive_coordinate_passes + 1):
        for name in ("w1", "w2", "w3"):
            candidates = list(_refined_scale_candidates(
                requested[name], profile.local_refinement_multipliers
            ))
            for quantile_name in profile.adaptive_quantile_names:
                candidate = robust_scale_candidates(
                    parameters[name],
                    quantiles=(named_quantiles[quantile_name],),
                )[0]
                if all(candidate != prior for prior in candidates):
                    candidates.append(candidate)
            candidate_evidence[name] = [float(value) for value in candidates]
            best: tuple[tuple[float, ...], np.float32] | None = None
            for candidate in candidates:
                trial_scales = dict(requested)
                trial_scales[name] = candidate
                quantized = quantize_fixed(parameters, architecture, trial_scales)
                metrics = evaluate_validation_pair(
                    parameters,
                    architecture,
                    inputs,
                    arm,
                    quantized=quantized,
                    ranking_weight=ranking_weight,
                )
                key = (*_qat_validation_key(metrics, profile), float(candidate))
                trials.append({
                    "pass": search_pass,
                    "layer": name,
                    "requested_scale": float(candidate),
                    "scales": {
                        layer: float(quantized.scales[layer])
                        for layer in ("w1", "w2", "w3")
                    },
                    "validation": metrics,
                })
                if best is None or key < best[0]:
                    best = (key, candidate)
            assert best is not None
            requested[name] = best[1]
    selected = quantize_fixed(parameters, architecture, requested)
    selected_metrics = evaluate_validation_pair(
        parameters,
        architecture,
        inputs,
        arm,
        quantized=selected,
        ranking_weight=ranking_weight,
    )
    before = {
        name: float(np.float32(starting_scales[name]))
        for name in ("w1", "w2", "w3")
    }
    after = {
        name: float(selected.scales[name]) for name in ("w1", "w2", "w3")
    }
    return selected, {
        "scheme": "post-epoch-local-plus-current-weight-quantile-reselection/v1",
        "qat_profile": profile.name,
        "qat_epoch": qat_epoch,
        "starting_scales": before,
        "candidates": candidate_evidence,
        "passes": profile.adaptive_coordinate_passes,
        "selected_scales": after,
        "changed_layers": [
            name for name in ("w1", "w2", "w3")
            if before[name] != after[name]
        ],
        "selection_changed": before != after,
        "selected_validation": selected_metrics,
        "trials": trials,
    }


@dataclasses.dataclass(frozen=True)
class QuantizedTrainingResult:
    quantized: QuantizedWeights
    qat_epoch: int
    metrics: dict[str, dict[str, float | int]]
    report: dict[str, object]


def _quantized_update_evidence(
    before: QuantizedWeights,
    after: QuantizedWeights,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name in ("w1", "w2", "w3"):
        first = np.asarray(before.integer[name], dtype=np.int8)
        last = np.asarray(after.integer[name], dtype=np.int8)
        if first.shape != last.shape:
            raise TrainingError("quantized update evidence tensor shape changed")
        report[name] = {
            "codes": int(first.size),
            "changed_codes": int(np.count_nonzero(first != last)),
            "changed": bool(np.any(first != last)),
            "before_sha256": sha256_bytes(first.tobytes(order="C")),
            "after_sha256": sha256_bytes(last.tobytes(order="C")),
            "scale": float(after.scales[name]),
        }
    return report


def run_fixed_scale_qat(
    float_result: FloatTrainingResult,
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    seed: int,
    *,
    qat_epochs: int = QAT_EPOCHS,
    ranking_weight: float = 0.0,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
) -> QuantizedTrainingResult:
    if qat_epochs != QAT_EPOCHS:
        raise TrainingError("compact deployment requires exactly four QAT epochs")
    profile = resolve_qat_profile(qat_profile)
    if profile.name == REFINED_ADAPTIVE_SCALES_QAT_PROFILE and (
        architecture.name != "capacity-12x8"
        or inputs.successor_rankings is None
    ):
        raise TrainingError(
            "refined adaptive QAT requires successor-labeled capacity-12x8"
        )
    ranking_weight = _ranking_weight(ranking_weight)
    all_ranking_groups = (
        () if inputs.successor_rankings is None else inputs.successor_rankings.train
    )
    ranking_groups, density_report = _density_weighted_ranking_groups(
        all_ranking_groups
    )
    if ranking_weight > 0.0 and not ranking_groups:
        raise TrainingError("positive ranking loss has no QAT training groups")
    pre_qat, scale_report = select_fixed_scales(
        float_result.parameters,
        architecture,
        inputs,
        arm,
        ranking_weight=ranking_weight,
        qat_profile=profile,
    )
    selected: QuantizedWeights | None = None
    selected_epoch = 0
    pre_qat_metrics = evaluate_validation_pair(
        float_result.parameters,
        architecture,
        inputs,
        arm,
        quantized=pre_qat,
        ranking_weight=ranking_weight,
    )
    selected_metrics: dict[str, dict[str, float | int]] | None = None
    selected_key: tuple[float, ...] | None = None
    fixed_scales = dict(pre_qat.scales)
    master = {
        name: value.copy() for name, value in float_result.parameters.items()
    }
    optimizer = AdamW(
        master, learning_rate=QAT_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    executed_batches = 0
    for qat_epoch in range(1, qat_epochs + 1):
        epoch_starting_parameters = {
            name: value.copy() for name, value in master.items()
        }
        schedule_epoch = (
            RANKING_FLOAT_EPOCHS
            if inputs.successor_rankings is not None
            else MAX_FLOAT_EPOCHS
        ) + qat_epoch
        batch_count = math.ceil(len(inputs.new) / NEW_ROWS_PER_BATCH)
        ranking_schedule = (
            None
            if ranking_weight == 0.0
            else successor_ranking_epoch_schedule(
                len(ranking_groups),
                batch_count,
                seed=seed,
                epoch=schedule_epoch,
            )
        )
        ranking_coverage = (
            None
            if ranking_schedule is None
            else ranking_schedule_coverage(
                ranking_groups, ranking_schedule, epoch=schedule_epoch
            )
        )
        for batch_index, (new_rows, anchor_rows) in enumerate(mixed_epoch_batches(
            len(inputs.new),
            len(inputs.anchor),
            seed=seed,
            epoch=schedule_epoch,
        )):
            _train_mixed_batch(
                master,
                architecture,
                arm,
                optimizer,
                inputs,
                new_rows,
                anchor_rows,
                fixed_scales=fixed_scales,
                ranking_groups=(
                    None
                    if ranking_schedule is None
                    else tuple(
                        ranking_groups[int(index)]
                        for index in ranking_schedule[batch_index]
                    )
                ),
                ranking_weight=ranking_weight,
            )
            executed_batches += 1
        applied_scales = {
            name: float(fixed_scales[name]) for name in ("w1", "w2", "w3")
        }
        adaptive_scale_search = None
        if profile.adapt_scales_after_each_epoch:
            candidate, adaptive_scale_search = _adapt_fixed_scales(
                master,
                architecture,
                inputs,
                arm,
                fixed_scales,
                profile,
                qat_epoch=qat_epoch,
                ranking_weight=ranking_weight,
            )
            metrics = adaptive_scale_search["selected_validation"]
            fixed_scales = dict(candidate.scales)
        else:
            candidate = quantize_fixed(master, architecture, fixed_scales)
            metrics = evaluate_validation_pair(
                master,
                architecture,
                inputs,
                arm,
                quantized=candidate,
                ranking_weight=ranking_weight,
            )
        key = _qat_validation_key(metrics, profile)
        history.append({
            "qat_epoch": qat_epoch,
            "schedule_epoch": schedule_epoch,
            "fixed_scales": applied_scales,
            "candidate_scales": {
                name: float(candidate.scales[name])
                for name in ("w1", "w2", "w3")
            },
            "adaptive_scale_search": adaptive_scale_search,
            "ranking_schedule_coverage": ranking_coverage,
            "fake_quantization": {
                "bits": QUANTIZATION_BITS,
                "layers": ["w1", "w2", "w3"],
                "scales_applied_to_every_batch": applied_scales,
                "batches": batch_count,
                "optimizer_steps_after_epoch": executed_batches,
                "all_layers_trainable": True,
                "master_parameter_updates": _parameter_update_evidence(
                    epoch_starting_parameters, master
                ),
            },
            "validation": metrics,
        })
        # QAT epoch zero is diagnostic only.  Strict comparison keeps the
        # earlier trained QAT epoch on an exact validation tie.
        if selected_key is None or key < selected_key:
            selected = candidate
            selected_epoch = qat_epoch
            selected_metrics = metrics
            selected_key = key
    if selected is None or selected_metrics is None or selected_epoch == 0:
        raise TrainingError("QAT produced no selectable trained epoch")
    selected_per_layer_update = _quantized_update_evidence(pre_qat, selected)
    qat_report: dict[str, object] = {
        "qat_profile": profile.name,
        "qat_profile_contract": qat_profile_contract(profile),
        "qat_epochs": qat_epochs,
        "learning_rate": QAT_LEARNING_RATE,
        "fixed_scale_qat": not profile.adapt_scales_after_each_epoch,
        "adaptive_scale_qat": profile.adapt_scales_after_each_epoch,
        "all_layer_fake_three_bit_qat": True,
        "selected_qat_epoch": selected_epoch,
        "selected_scales": {
            name: float(selected.scales[name])
            for name in ("w1", "w2", "w3")
        },
        "pre_qat_validation": pre_qat_metrics,
        "pre_qat_retained": False,
        "tie_break": "prefer-earlier-qat-epoch-on-exact-tie",
        "scale_search": scale_report,
        "history": history,
        "executed_qat_epochs": [
            int(item["qat_epoch"]) for item in history
        ],
        "optimizer_steps": executed_batches,
        "final_master_per_layer_update_evidence": _parameter_update_evidence(
            float_result.parameters, master
        ),
        "applied_scale_trajectory": [
            {
                "qat_epoch": int(item["qat_epoch"]),
                "training_scales": dict(item["fixed_scales"]),
                "candidate_scales": dict(item["candidate_scales"]),
                "adapted_after_epoch": (
                    item["adaptive_scale_search"] is not None
                ),
            }
            for item in history
        ],
        "selected_validation": selected_metrics,
        "selected_per_layer_qat_evidence": selected_per_layer_update,
    }
    if inputs.successor_rankings is not None:
        qat_report.update({
            "successor_ranking": {
                "labels_present": True,
                "loss_active": ranking_weight > 0.0,
                "loss_weight": ranking_weight,
                "composition": "scalar-loss-plus-lambda-ranking-loss",
                "group_microbatch_objective": (
                    "mean-of-gap-normalized-group-losses"
                ),
                "ranking_lambda_application": "once-after-group-mean",
                "epoch_schedule": (
                    "balanced-full-weighted-pool-permutation-per-epoch-v1"
                ),
                "pair_cap": RANKING_PAIR_CAP,
                "gap_weighting": "teacher-gap-normalized",
                "train_groups": len(all_ranking_groups),
                "comparable_train_groups": density_report[
                    "unique_comparable_groups"
                ],
                "hard_state_density": density_report,
                "weighted_group_entries_per_epoch": len(ranking_groups),
                "full_weighted_pool_coverage_each_active_epoch": (
                    ranking_weight > 0.0
                ),
                "selected_epoch_schedule_coverage": history[
                    selected_epoch - 1
                ]["ranking_schedule_coverage"],
                "skipped_nonexhaustive_train_groups": sum(
                    not group.successors_exhaustive
                    for group in all_ranking_groups
                ),
                "skipped_zero_pair_train_groups": sum(
                    group.successors_exhaustive and not bool(
                        _ranking_pairs(group)[1]
                    )
                    for group in all_ranking_groups
                ),
            },
            "per_layer_update_evidence": selected_per_layer_update,
        })
    validate_qat_execution_evidence(qat_report, expected_profile=profile.name)
    if inputs.successor_rankings is not None:
        validate_successor_schedule_execution(
            float_result.report, qat_report, seed=seed
        )
    return QuantizedTrainingResult(
        quantized=selected,
        qat_epoch=selected_epoch,
        metrics=selected_metrics,
        report=qat_report,
    )


def validate_qat_execution_evidence(
    value: object, *, expected_profile: str,
) -> dict[str, object]:
    """Validate the exact four-epoch, all-layer fake-quantization evidence."""

    if not isinstance(value, Mapping):
        raise TrainingError("QAT execution evidence is absent")
    profile = resolve_qat_profile(expected_profile)
    validate_qat_profile_contract(
        value.get("qat_profile_contract"), expected_name=profile.name
    )
    history = value.get("history")
    trajectory = value.get("applied_scale_trajectory")
    scale_search = value.get("scale_search")

    def scales(record: object, label: str) -> dict[str, float]:
        if not isinstance(record, Mapping) or set(record) != {"w1", "w2", "w3"}:
            raise TrainingError(f"{label} scale evidence is incomplete")
        normalized: dict[str, float] = {}
        for name in ("w1", "w2", "w3"):
            raw = record[name]
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) <= 0.0
                or float(np.float32(raw)) != float(raw)
            ):
                raise TrainingError(f"{label} scale evidence is invalid")
            normalized[name] = float(raw)
        return normalized

    if not isinstance(scale_search, Mapping):
        raise TrainingError("QAT initial scale-search evidence is absent")
    validate_qat_profile_contract(
        scale_search.get("qat_profile_contract"), expected_name=profile.name
    )
    initial_scales = scales(
        scale_search.get("selected_scales"), "QAT initial"
    )
    selected_scales = scales(value.get("selected_scales"), "QAT selected")
    selected_epoch = value.get("selected_qat_epoch")
    final_updates = value.get("final_master_per_layer_update_evidence")
    selected_updates = value.get("selected_per_layer_qat_evidence")
    if (
        value.get("qat_profile") != profile.name
        or value.get("qat_epochs") != QAT_EPOCHS
        or value.get("learning_rate") != QAT_LEARNING_RATE
        or value.get("fixed_scale_qat")
        is not (not profile.adapt_scales_after_each_epoch)
        or value.get("adaptive_scale_qat")
        is not profile.adapt_scales_after_each_epoch
        or value.get("all_layer_fake_three_bit_qat") is not True
        or value.get("executed_qat_epochs") != [1, 2, 3, 4]
        or not isinstance(history, list)
        or len(history) != QAT_EPOCHS
        or not isinstance(trajectory, list)
        or len(trajectory) != QAT_EPOCHS
        or scale_search.get("qat_profile") != profile.name
        or scale_search.get("passes") != profile.coordinate_search_passes
        or scale_search.get("local_refinement_passes")
        != profile.local_refinement_passes
        or not isinstance(scale_search.get("trials"), list)
        or not scale_search["trials"]
        or (
            profile.local_refinement_passes == 0
            and scale_search.get("local_refinement_trials") != 0
        )
        or (
            profile.local_refinement_passes > 0
            and (
                not isinstance(scale_search.get("local_refinement_trials"), int)
                or scale_search["local_refinement_trials"] <= 0
            )
        )
        or isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= QAT_EPOCHS
        or value.get("pre_qat_retained") is not False
        or value.get("tie_break")
        != "prefer-earlier-qat-epoch-on-exact-tie"
        or not isinstance(value.get("pre_qat_validation"), Mapping)
        or not isinstance(final_updates, Mapping)
        or set(final_updates) != {"w1", "w2", "w3"}
        or not isinstance(selected_updates, Mapping)
        or set(selected_updates) != {"w1", "w2", "w3"}
    ):
        raise TrainingError("QAT execution schedule/profile evidence changed")

    for name in ("w1", "w2", "w3"):
        update = selected_updates[name]
        if (
            not isinstance(update, Mapping)
            or set(update) != {
                "codes", "changed_codes", "changed", "before_sha256",
                "after_sha256", "scale",
            }
            or isinstance(update.get("codes"), bool)
            or not isinstance(update.get("codes"), int)
            or update["codes"] <= 0
            or isinstance(update.get("changed_codes"), bool)
            or not isinstance(update.get("changed_codes"), int)
            or not 0 <= update["changed_codes"] <= update["codes"]
            or update.get("changed") is not (update["changed_codes"] > 0)
            or not valid_sha256(update.get("before_sha256"))
            or not valid_sha256(update.get("after_sha256"))
            or update.get("scale") != selected_scales[name]
        ):
            raise TrainingError("QAT selected per-layer evidence changed")
    expected_step = 0
    next_training_scales = initial_scales
    selected_epoch_scales = initial_scales
    for index, (epoch, applied) in enumerate(zip(history, trajectory), start=1):
        if not isinstance(epoch, Mapping) or not isinstance(applied, Mapping):
            raise TrainingError("QAT epoch evidence is malformed")
        fake = epoch.get("fake_quantization")
        batches = fake.get("batches") if isinstance(fake, Mapping) else None
        applied_scales = fake.get("scales_applied_to_every_batch") if isinstance(
            fake, Mapping
        ) else None
        updates = fake.get("master_parameter_updates") if isinstance(
            fake, Mapping
        ) else None
        if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
            raise TrainingError("QAT epoch batch evidence is invalid")
        expected_step += batches
        epoch_scales = scales(epoch.get("fixed_scales"), "QAT epoch training")
        candidate_scales = scales(
            epoch.get("candidate_scales"), "QAT epoch candidate"
        )
        fake_scales = scales(applied_scales, "QAT fake-quantization")
        if (
            epoch.get("qat_epoch") != index
            or fake.get("bits") != QUANTIZATION_BITS
            or fake.get("layers") != ["w1", "w2", "w3"]
            or fake.get("all_layers_trainable") is not True
            or fake.get("optimizer_steps_after_epoch") != expected_step
            or fake_scales != epoch_scales
            or epoch_scales != next_training_scales
            or not isinstance(updates, Mapping)
            or set(updates) != {"w1", "w2", "w3"}
            or applied.get("qat_epoch") != index
            or applied.get("training_scales") != epoch_scales
            or applied.get("candidate_scales") != candidate_scales
            or applied.get("adapted_after_epoch")
            is not profile.adapt_scales_after_each_epoch
            or (epoch.get("adaptive_scale_search") is not None)
            is not profile.adapt_scales_after_each_epoch
        ):
            raise TrainingError("QAT all-layer scale/application evidence changed")
        adaptive = epoch.get("adaptive_scale_search")
        if profile.adapt_scales_after_each_epoch:
            if (
                not isinstance(adaptive, Mapping)
                or adaptive.get("qat_profile") != profile.name
                or adaptive.get("qat_epoch") != index
                or adaptive.get("starting_scales") != epoch_scales
                or adaptive.get("selected_scales") != candidate_scales
                or adaptive.get("passes") != profile.adaptive_coordinate_passes
                or not isinstance(adaptive.get("trials"), list)
                or not adaptive["trials"]
            ):
                raise TrainingError("QAT adaptive scale evidence changed")
        elif candidate_scales != epoch_scales:
            raise TrainingError("standard-v1 changed scales during QAT")
        next_training_scales = candidate_scales
        if selected_epoch == index:
            selected_epoch_scales = candidate_scales
    if value.get("optimizer_steps") != expected_step:
        raise TrainingError("QAT optimizer step evidence changed")
    if selected_scales != selected_epoch_scales:
        raise TrainingError("QAT selected scale evidence changed")
    if value.get("selected_validation") != history[selected_epoch - 1].get(
        "validation"
    ):
        raise TrainingError("QAT selected validation evidence changed")
    successor = value.get("successor_ranking")
    if isinstance(successor, Mapping) and value.get(
        "per_layer_update_evidence"
    ) != selected_updates:
        raise TrainingError("QAT successor selected-layer evidence changed")
    return dict(value)


def validate_successor_schedule_execution(
    float_training: object, quantized_training: object,
    *, seed: int,
) -> dict[str, object]:
    """Validate full weighted-pool coverage across warm-up and all QAT epochs."""

    if not isinstance(float_training, Mapping) or not isinstance(
        quantized_training, Mapping
    ):
        raise TrainingError("successor schedule execution evidence is absent")
    float_successor = float_training.get("successor_ranking")
    qat_successor = quantized_training.get("successor_ranking")
    float_history = float_training.get("history")
    qat_history = quantized_training.get("history")
    if (
        seed not in FIXED_SEEDS
        or not isinstance(float_successor, Mapping)
        or not isinstance(qat_successor, Mapping)
        or not isinstance(float_history, list)
        or len(float_history) != RANKING_FLOAT_EPOCHS
        or not isinstance(qat_history, list)
        or len(qat_history) != QAT_EPOCHS
        or float_successor.get("hard_state_density")
        != qat_successor.get("hard_state_density")
        or float_successor.get("loss_active")
        is not qat_successor.get("loss_active")
        or float_successor.get("loss_weight") != qat_successor.get("loss_weight")
        or any(
            successor.get("group_microbatch_objective")
            != "mean-of-gap-normalized-group-losses"
            or successor.get("ranking_lambda_application")
            != "once-after-group-mean"
            or successor.get("epoch_schedule")
            != "balanced-full-weighted-pool-permutation-per-epoch-v1"
            for successor in (float_successor, qat_successor)
        )
    ):
        raise TrainingError("successor warm-up/QAT schedule binding changed")
    density = float_successor["hard_state_density"]
    loss_active = float_successor.get("loss_active")
    weighted_entries = density.get("scheduled_group_entries") if isinstance(
        density, Mapping
    ) else None
    if (
        not isinstance(loss_active, bool)
        or isinstance(weighted_entries, bool)
        or not isinstance(weighted_entries, int)
        or weighted_entries < 0
        or float_successor.get("weighted_group_entries_per_epoch")
        != weighted_entries
        or qat_successor.get("weighted_group_entries_per_epoch")
        != weighted_entries
    ):
        raise TrainingError("successor weighted-pool schedule evidence changed")
    active = loss_active
    if float_successor.get("full_weighted_pool_coverage_each_active_epoch") is not active:
        raise TrainingError("successor float coverage policy changed")
    if qat_successor.get("full_weighted_pool_coverage_each_active_epoch") is not active:
        raise TrainingError("successor QAT coverage policy changed")

    reports = []
    for expected_epoch, item in enumerate(float_history, start=1):
        if not isinstance(item, Mapping):
            raise TrainingError("successor float history is malformed")
        scalar = item.get("coverage", {}).get("new") if isinstance(
            item.get("coverage"), Mapping
        ) else None
        rows = scalar.get("rows_per_epoch") if isinstance(scalar, Mapping) else None
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise TrainingError("successor float scalar batch evidence is malformed")
        report = item.get("ranking_schedule_coverage")
        if active:
            reports.append(validate_ranking_schedule_coverage(
                report,
                density=density,
                epoch=expected_epoch,
                scalar_batches=rows // NEW_ROWS_PER_BATCH,
                seed=seed,
            ))
        elif report is not None:
            raise TrainingError("inactive ranking loss executed a float group schedule")
    for qat_epoch, item in enumerate(qat_history, start=1):
        if not isinstance(item, Mapping):
            raise TrainingError("successor QAT history is malformed")
        fake = item.get("fake_quantization")
        batches = fake.get("batches") if isinstance(fake, Mapping) else None
        schedule_epoch = item.get("schedule_epoch")
        expected_schedule_epoch = RANKING_FLOAT_EPOCHS + qat_epoch
        if (
            isinstance(batches, bool)
            or not isinstance(batches, int)
            or batches <= 0
            or schedule_epoch != expected_schedule_epoch
        ):
            raise TrainingError("successor QAT scalar schedule is discontinuous")
        report = item.get("ranking_schedule_coverage")
        if active:
            reports.append(validate_ranking_schedule_coverage(
                report,
                density=density,
                epoch=expected_schedule_epoch,
                scalar_batches=batches,
                seed=seed,
            ))
        elif report is not None:
            raise TrainingError("inactive ranking loss executed a QAT group schedule")
    selected_float = float_successor.get("selected_epoch_schedule_coverage")
    selected_qat = qat_successor.get("selected_epoch_schedule_coverage")
    if active:
        float_epoch = float_training.get("best_float_epoch")
        qat_epoch = quantized_training.get("selected_qat_epoch")
        if (
            isinstance(float_epoch, bool)
            or not isinstance(float_epoch, int)
            or not 1 <= float_epoch <= len(float_history)
            or isinstance(qat_epoch, bool)
            or not isinstance(qat_epoch, int)
            or not 1 <= qat_epoch <= len(qat_history)
            or selected_float
            != float_history[float_epoch - 1].get("ranking_schedule_coverage")
            or selected_qat
            != qat_history[qat_epoch - 1].get("ranking_schedule_coverage")
        ):
            raise TrainingError("selected successor schedule coverage changed")
    elif selected_float is not None or selected_qat is not None:
        raise TrainingError("inactive ranking loss selected group coverage")
    return {
        "loss_active": active,
        "float_epochs": len(float_history),
        "qat_epochs": len(qat_history),
        "validated_full_pool_reports": len(reports),
        "weighted_entries_per_active_epoch": (
            density.get("scheduled_group_entries") if active else 0
        ),
    }


def _finite_metric_report(report: Mapping[str, object]) -> bool:
    for value in report.values():
        if isinstance(value, Mapping):
            if not _finite_metric_report(value):
                return False
        elif isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def offline_advancement_gate(
    float_metrics: Mapping[str, Mapping[str, float | int]],
    quantized_metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, object]:
    errors: list[str] = []
    if not _finite_metric_report(float_metrics) or not _finite_metric_report(
        quantized_metrics
    ):
        errors.append("metric report contains NaN or infinity")
    for name, minimum_sign, maximum_huber in (
        ("common_adjudicator", COMMON_MINIMUM_SIGN, COMMON_MAXIMUM_HUBER),
        ("canonical_validation", CANONICAL_MINIMUM_SIGN, CANONICAL_MAXIMUM_HUBER),
    ):
        float_report = float_metrics[name]
        quantized = quantized_metrics[name]
        sign = float(quantized["sign_accuracy"])
        huber = float(quantized["weighted_huber"])
        float_sign = float(float_report["sign_accuracy"])
        float_huber = float(float_report["weighted_huber"])
        if sign < minimum_sign:
            errors.append(f"{name} sign accuracy is below {minimum_sign}")
        if huber > maximum_huber:
            errors.append(f"{name} weighted Huber exceeds {maximum_huber}")
        if not (float_sign - sign < MAXIMUM_SIGN_LOSS):
            errors.append(f"{name} quantized sign loss is not below .005")
        if huber > float_huber * MAXIMUM_HUBER_RATIO:
            errors.append(f"{name} quantized Huber exceeds 1.02x float")
    passed = not errors
    return {
        "passed": passed,
        "status": (
            "offline-evaluator-qualified-not-game-gated"
            if passed
            else "offline-evaluator-rejected"
        ),
        "errors": errors,
        "thresholds": {
            "common_adjudicator": {
                "minimum_sign_accuracy": COMMON_MINIMUM_SIGN,
                "maximum_weighted_huber": COMMON_MAXIMUM_HUBER,
            },
            "canonical_validation": {
                "minimum_sign_accuracy": CANONICAL_MINIMUM_SIGN,
                "maximum_weighted_huber": CANONICAL_MAXIMUM_HUBER,
            },
            "relative_to_float": {
                "maximum_sign_loss_exclusive": MAXIMUM_SIGN_LOSS,
                "maximum_huber_ratio_inclusive": MAXIMUM_HUBER_RATIO,
            },
        },
    }


def _array_npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output, np.asarray(value), allow_pickle=False
    )
    return output.getvalue()


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            archive.writestr(info, _array_npy_bytes(arrays[name]))
    return output.getvalue()


def write_float_checkpoint(
    output_directory: pathlib.Path,
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
) -> pathlib.Path:
    normalized = _validate_parameters(parameters, architecture)
    payload = deterministic_npz({
        name: np.asarray(normalized[name], dtype="<f4")
        for name in ("w1", "w2", "w3")
    })
    path = _write_content_addressed(output_directory, payload, ".float.npz")
    load_float_checkpoint(path, architecture)
    return path


def load_float_checkpoint(
    path: pathlib.Path, architecture: Architecture
) -> dict[str, np.ndarray]:
    if not path.name.endswith(".float.npz"):
        raise TrainingError("float checkpoint suffix is invalid")
    expected_sha = path.name.removesuffix(".float.npz")
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise TrainingError("float checkpoint is not content addressed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"w1", "w2", "w3"}:
                raise TrainingError("float checkpoint tensor roster changed")
            parameters = {
                name: archive[name].copy() for name in ("w1", "w2", "w3")
            }
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError("float checkpoint is corrupt") from error
    for name, value in parameters.items():
        if value.dtype != np.dtype("<f4"):
            raise TrainingError(f"float checkpoint {name} dtype changed")
    _validate_parameters(parameters, architecture)
    return parameters


def _bound_initial_checkpoint(
    path: pathlib.Path, architecture: Architecture,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise TrainingError("initial float checkpoint is absent or redirected")
    parameters = load_float_checkpoint(path, architecture)
    return parameters, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "parameters": _parameter_identity(parameters, architecture),
    }


def training_binding(
    bundle: FrozenBundle,
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    seed: int,
    sidecar_index: pathlib.Path | None,
    ranking_weight: float = 0.0,
    initial_checkpoint: pathlib.Path | None = None,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
) -> dict[str, object]:
    ranking_weight = _ranking_weight(ranking_weight)
    profile = resolve_qat_profile(qat_profile)
    sidecar = None
    if sidecar_index is not None:
        sidecar = {
            "file_sha256": sha256_file(sidecar_index),
            "body_sha256": _load_canonical_json(
                sidecar_index, "teacher sidecar index"
            )[1]["body_sha256"],
        }
    body: dict[str, object] = {
        "schema": "papersoccer.compact-value-bfm-training-binding.v1",
        "campaign_id": CAMPAIGN_ID,
        "source_bundle_body_sha256": bundle.body_sha256,
        "architecture": {
            "name": architecture.name,
            "dimensions": list(architecture.dimensions),
            "biases": False,
            "activations": list(ACTIVATIONS),
        },
        "arm": dataclasses.asdict(arm),
        "seed": seed,
        "datasets": {
            "new": dataset_identity(inputs.new),
            "anchor": dataset_identity(inputs.anchor),
            "common_adjudicator": dataset_identity(inputs.common_adjudicator),
            "canonical_validation": dataset_identity(inputs.canonical_validation),
        },
        # JSON has no tuple type.  Normalize the in-memory route tuples before
        # hashing so the just-written receipt compares equal after reload and
        # later seeds can advance instead of falsely reporting binding drift.
        "source_routes": {
            name: list(routes)
            for name, routes in sorted(inputs.source_routes.items())
        },
        "paired_row_validation": inputs.paired_row_validation,
        "split_isolation": inputs.split_isolation,
        "input_audit": inputs.input_audit,
        "teacher_sidecar_index": sidecar,
        "settings": {
            "seeds": list(FIXED_SEEDS),
            "batch_size": BATCH_SIZE,
            "new_rows_per_batch": NEW_ROWS_PER_BATCH,
            "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
            "new_loss_share": 0.25,
            "anchor_loss_share": 0.75,
            "maximum_float_epochs": MAX_FLOAT_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "qat_epochs": QAT_EPOCHS,
            "qat_learning_rate": QAT_LEARNING_RATE,
            "qat_profile": profile.name,
            "qat_profile_contract": qat_profile_contract(profile),
        },
    }
    labels = inputs.successor_rankings
    if ranking_weight > 0.0 and labels is None:
        raise TrainingError("positive ranking loss has no bound label artifact")
    if (labels is None) != (initial_checkpoint is None):
        raise TrainingError(
            "successor labels require exactly one frozen initial checkpoint"
        )
    if labels is not None:
        if architecture.name != "capacity-12x8":
            raise TrainingError("successor ranking requires capacity-12x8")
        assert initial_checkpoint is not None
        _initial_parameters, checkpoint = _bound_initial_checkpoint(
            initial_checkpoint, architecture
        )
        if labels.artifact_schema not in {SUCCESSOR_LABEL_SCHEMA, SUCCESSOR_STORE_SCHEMA}:
            raise TrainingError("unrecognized successor artifact schema")
        body["successor_ranking"] = {
            "schema": labels.artifact_schema,
            "artifact_sha256": labels.artifact_sha256,
            "body_sha256": labels.body_sha256,
            "source_bundle_body_sha256": labels.source_bundle_body_sha256,
            "teacher": dict(labels.teacher),
            "train_groups": len(labels.train),
            "validation_groups": len(labels.validation),
            "comparable_train_groups": len(
                _comparable_ranking_groups(labels.train)
            ),
            "comparable_validation_groups": len(
                _comparable_ranking_groups(labels.validation)
            ),
            "skipped_nonexhaustive_groups": sum(
                not group.successors_exhaustive
                for group in (*labels.train, *labels.validation)
            ),
            "loss_weight": ranking_weight,
            "composition": "scalar-loss-plus-lambda-ranking-loss",
            "group_microbatch_objective": "mean-of-gap-normalized-group-losses",
            "ranking_lambda_application": "once-after-group-mean",
            "epoch_schedule": (
                "balanced-full-weighted-pool-permutation-per-epoch-v1"
            ),
            "allowed_loss_weights": list(RANKING_LOSS_WEIGHTS),
            "pair_cap": RANKING_PAIR_CAP,
            "gap_weighting": "teacher-gap-normalized",
            "runtime_architecture_changed": False,
            "initial_checkpoint": checkpoint,
            "float_warmup": {
                "epochs": RANKING_FLOAT_EPOCHS,
                "learning_rate": RANKING_FLOAT_LEARNING_RATE,
                "seeds_affect_row_order_only": True,
                "legacy_full_anchor_pass_required": False,
            },
        }
    elif profile.name != STANDARD_QAT_PROFILE:
        raise TrainingError(
            "refined adaptive QAT requires successor-labeled capacity-12x8"
        )
    return body_hashed(body)


def _seed_reference_path(
    output_directory: pathlib.Path,
    architecture: Architecture,
    arm: Arm,
    seed: int,
) -> pathlib.Path:
    return (
        output_directory
        / "seed-references"
        / architecture.name
        / arm.name
        / f"seed-{seed}.json"
    )


def _relative_artifact_path(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise TrainingError("training artifact escaped its output root") from error


def _output_artifact(
    output_directory: pathlib.Path,
    relative: object,
    *,
    expected_sha256: object,
    label: str,
) -> pathlib.Path:
    relative_text = _safe_relative(relative, label)
    path = (output_directory / relative_text).resolve()
    try:
        path.relative_to(output_directory.resolve())
    except ValueError as error:
        raise TrainingError(f"{label} escapes the output root") from error
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise TrainingError(f"{label} content binding changed")
    return path


def _load_seed_receipt_from_reference(
    output_directory: pathlib.Path,
    reference_path: pathlib.Path,
    expected_binding: Mapping[str, object],
) -> dict[str, Any]:
    _payload, reference = _load_canonical_json(
        reference_path, "compact seed reference"
    )
    verify_body_hash(
        reference, schema=SEED_REFERENCE_SCHEMA, label="compact seed reference"
    )
    receipt_path = _output_artifact(
        output_directory,
        reference.get("receipt"),
        expected_sha256=reference.get("receipt_sha256"),
        label="seed receipt",
    )
    receipt_payload, receipt = _load_canonical_json(
        receipt_path, "compact seed receipt"
    )
    if receipt_path.name != f"{sha256_bytes(receipt_payload)}.seed-receipt.json":
        raise TrainingError("compact seed receipt is not content addressed")
    verify_body_hash(receipt, schema=SEED_RECEIPT_SCHEMA, label="compact seed receipt")
    validate_native_thread_execution(receipt.get("native_thread_execution"))
    if receipt.get("successor_ranking", {}).get("labels_present") is True:
        schedule_execution = validate_successor_schedule_execution(
            receipt.get("float_training"), receipt.get("quantized_training"),
            seed=receipt.get("seed"),
        )
        if receipt["successor_ranking"].get(
            "schedule_execution"
        ) != schedule_execution:
            raise TrainingError("compact seed successor schedule summary changed")
    if receipt.get("binding") != expected_binding:
        raise TrainingError("compact seed resume binding changed")
    settings = expected_binding.get("settings")
    profile_name = settings.get("qat_profile") if isinstance(
        settings, Mapping
    ) else None
    if not isinstance(profile_name, str):
        raise TrainingError("compact seed binding lost its QAT profile")
    profile_contract = validate_qat_profile_contract(
        settings.get("qat_profile_contract"), expected_name=profile_name
    )
    if (
        receipt.get("qat_profile") != profile_name
        or receipt.get("qat_profile_contract") != profile_contract
    ):
        raise TrainingError("compact seed receipt QAT profile changed")
    validate_qat_execution_evidence(
        receipt.get("quantized_training"), expected_profile=profile_name
    )
    architecture_name = receipt.get("architecture")
    if architecture_name not in ARCHITECTURES:
        raise TrainingError("compact seed receipt architecture changed")
    checkpoint = receipt.get("float_checkpoint")
    runtime = receipt.get("quantized_runtime")
    if not isinstance(checkpoint, dict) or not isinstance(runtime, dict):
        raise TrainingError("compact seed receipt artifact binding is incomplete")
    checkpoint_path = _output_artifact(
        output_directory,
        checkpoint.get("path"),
        expected_sha256=checkpoint.get("sha256"),
        label="float checkpoint",
    )
    load_float_checkpoint(checkpoint_path, ARCHITECTURES[architecture_name])
    runtime_path = _output_artifact(
        output_directory,
        runtime.get("path"),
        expected_sha256=runtime.get("sha256"),
        label="quantized runtime",
    )
    loaded_architecture, _quantized, selection, _document = load_runtime(runtime_path)
    if (
        loaded_architecture.name != architecture_name
        or selection.get("arm") != receipt.get("arm")
        or selection.get("seed") != receipt.get("seed")
    ):
        raise TrainingError("compact seed runtime disagrees with its receipt")
    return receipt


def train_seed_candidate(
    bundle: FrozenBundle,
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    seed: int,
    output_directory: pathlib.Path,
    *,
    sidecar_index: pathlib.Path | None = None,
    ranking_weight: float = 0.0,
    initial_checkpoint: pathlib.Path | None = None,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
    resume: bool = False,
    _native_thread_execution: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if _native_thread_execution is None:
        with native_thread_execution_scope() as execution:
            return train_seed_candidate(
                bundle,
                inputs,
                architecture,
                arm,
                seed,
                output_directory,
                sidecar_index=sidecar_index,
                ranking_weight=ranking_weight,
                initial_checkpoint=initial_checkpoint,
                qat_profile=qat_profile,
                resume=resume,
                _native_thread_execution=execution,
            )
    native_execution = validate_native_thread_execution(
        _native_thread_execution
    )
    profile = resolve_qat_profile(qat_profile)
    binding = training_binding(
        bundle,
        inputs,
        architecture,
        arm,
        seed,
        sidecar_index,
        ranking_weight,
        initial_checkpoint,
        profile,
    )
    reference_path = _seed_reference_path(
        output_directory, architecture, arm, seed
    )
    if reference_path.exists():
        if not resume:
            raise TrainingError("completed seed exists; use --resume")
        receipt = _load_seed_receipt_from_reference(
            output_directory, reference_path, binding
        )
        if receipt.get("native_thread_execution") != native_execution:
            raise TrainingError("compact seed resume native-thread contract changed")
        return receipt
    # No intermediate optimizer/epoch state is persisted.  Any orphaned
    # content-addressed files left by interruption are harmless; without the
    # final reference this seed always restarts deterministically at epoch zero.
    float_arguments: dict[str, Any] = {"ranking_weight": ranking_weight}
    if inputs.successor_rankings is not None:
        if initial_checkpoint is None:
            raise TrainingError("successor ranking initial checkpoint is absent")
        initial_parameters, _checkpoint = _bound_initial_checkpoint(
            initial_checkpoint, architecture
        )
        float_arguments.update({
            "maximum_epochs": RANKING_FLOAT_EPOCHS,
            "patience": 1,
            "learning_rate": RANKING_FLOAT_LEARNING_RATE,
            "initial_parameters": initial_parameters,
        })
    float_result = train_float_seed(
        inputs, architecture, arm, seed, **float_arguments
    )
    quantized_result = run_fixed_scale_qat(
        float_result,
        inputs,
        architecture,
        arm,
        seed,
        ranking_weight=ranking_weight,
        qat_profile=profile,
    )
    gate = offline_advancement_gate(
        float_result.metrics, quantized_result.metrics
    )
    checkpoint_path = write_float_checkpoint(
        output_directory / "float-checkpoints",
        float_result.parameters,
        architecture,
    )
    runtime_path = write_runtime(
        output_directory / "quantized-runtimes",
        architecture,
        quantized_result.quantized,
        arm=arm,
        seed=seed,
        float_epoch=float_result.epoch,
        qat_epoch=quantized_result.qat_epoch,
        source_bundle_body_sha256=bundle.body_sha256,
    )
    if len(inputs.common_adjudicator) < 4_096:
        raise TrainingError("common adjudicator has fewer than 4,096 parity states")
    parity = assert_quantized_inference_parity(
        quantized_result.quantized,
        architecture,
        inputs.common_adjudicator,
        maximum_rows=4_096,
    )
    body: dict[str, object] = {
        "schema": SEED_RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "binding": binding,
        "architecture": architecture.name,
        "arm": arm.name,
        "seed": seed,
        "native_thread_execution": native_execution,
        "qat_profile": profile.name,
        "qat_profile_contract": qat_profile_contract(profile),
        "float_checkpoint": {
            "path": _relative_artifact_path(checkpoint_path, output_directory),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "quantized_runtime": {
            "path": _relative_artifact_path(runtime_path, output_directory),
            "sha256": sha256_file(runtime_path),
            "bytes": runtime_path.stat().st_size,
        },
        "float_training": float_result.report,
        "quantized_training": quantized_result.report,
        "float_validation": float_result.metrics,
        "quantized_validation": quantized_result.metrics,
        "offline_gate": gate,
        "inference_parity": parity,
        "status": gate["status"],
        "deployment_eligible": arm.deployment_eligible,
        "protected_tests_opened": False,
        "resume_policy": "completed-receipt-reused;interrupted-seed-restarts-epoch-zero",
    }
    if inputs.successor_rankings is not None:
        body["successor_ranking"] = {
            "labels_present": True,
            "loss_active": ranking_weight > 0.0,
            "loss_weight": _ranking_weight(ranking_weight),
            "float_validation": float_result.metrics["successor_ranking"],
            "quantized_validation": quantized_result.metrics[
                "successor_ranking"
            ],
            "float_per_layer_update_evidence": float_result.report[
                "per_layer_update_evidence"
            ],
            "qat_per_layer_update_evidence": quantized_result.report[
                "per_layer_update_evidence"
            ],
            "hard_state_density": float_result.report[
                "successor_ranking"
            ]["hard_state_density"],
        }
        if (
            quantized_result.report["successor_ranking"][
                "hard_state_density"
            ] != body["successor_ranking"]["hard_state_density"]
        ):
            raise TrainingError("float/QAT hard-state density policies differ")
        body["successor_ranking"]["schedule_execution"] = (
            validate_successor_schedule_execution(
                float_result.report, quantized_result.report, seed=seed
            )
        )
    receipt_document = body_hashed(body)
    receipt_path = _write_content_addressed(
        output_directory / "seed-receipts",
        canonical_json_bytes(receipt_document),
        ".seed-receipt.json",
    )
    reference = body_hashed({
        "schema": SEED_REFERENCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "binding_body_sha256": binding["body_sha256"],
        "receipt": _relative_artifact_path(receipt_path, output_directory),
        "receipt_sha256": sha256_file(receipt_path),
    })
    _write_stable_reference(reference_path, reference)
    return _load_seed_receipt_from_reference(
        output_directory, reference_path, binding
    )


def _receipt_selection_key(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = receipt["quantized_validation"]
    return (*_validation_key(metrics), float(receipt["seed"]))


def _seed_worker_count(value: int, *, successor_mode: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise TrainingError("seed workers must be exactly 1 or 2")
    if successor_mode and value != 2:
        raise TrainingError("successor-label training requires exactly two seed workers")
    return value


def _train_seed_roster(
    bundle: FrozenBundle,
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    output_directory: pathlib.Path,
    *,
    seed_workers: int,
    sidecar_index: pathlib.Path | None,
    ranking_weight: float,
    initial_checkpoint: pathlib.Path | None,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
    resume: bool,
) -> list[dict[str, Any]]:
    """Run independent seeds with shared read-only inputs and stable ordering."""

    workers = _seed_worker_count(
        seed_workers, successor_mode=inputs.successor_rankings is not None
    )

    with native_thread_execution_scope() as native_execution:
        def one(seed: int) -> dict[str, Any]:
            return train_seed_candidate(
                bundle,
                inputs,
                architecture,
                arm,
                seed,
                output_directory,
                sidecar_index=sidecar_index,
                ranking_weight=ranking_weight,
                initial_checkpoint=initial_checkpoint,
                qat_profile=qat_profile,
                resume=resume,
                _native_thread_execution=native_execution,
            )

        if workers == 1:
            return [one(seed) for seed in FIXED_SEEDS]
        ordered: list[dict[str, Any] | None] = [None] * len(FIXED_SEEDS)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="compact-value-bfm-seed",
        ) as executor:
            futures = {
                executor.submit(one, seed): index
                for index, seed in enumerate(FIXED_SEEDS)
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    ordered[futures[future]] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if any(receipt is None for receipt in ordered):
            raise TrainingError("seed worker pool returned an incomplete receipt roster")
        return [receipt for receipt in ordered if receipt is not None]


def train_arm_campaign(
    bundle: FrozenBundle,
    architecture: Architecture,
    arm: Arm,
    output_directory: pathlib.Path,
    *,
    sidecar_index: pathlib.Path | None = None,
    successor_labels: pathlib.Path | None = None,
    ranking_weight: float = 0.0,
    initial_checkpoint: pathlib.Path | None = None,
    input_audit: pathlib.Path | None = None,
    seed_workers: int = 1,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
    resume: bool = False,
    generated_source_ascii_bytes: int | None = None,
) -> pathlib.Path:
    inputs = load_training_inputs(
        bundle,
        arm,
        sidecar_index=sidecar_index,
        input_audit=input_audit,
        successor_labels=successor_labels,
    )
    ranking_weight = _ranking_weight(ranking_weight)
    profile = resolve_qat_profile(qat_profile)
    if successor_labels is None and ranking_weight > 0.0:
        raise TrainingError(
            "positive ranking weight requires successor labels"
        )
    if (successor_labels is None) != (initial_checkpoint is None):
        raise TrainingError(
            "successor labels require --initial-checkpoint and legacy mode forbids it"
        )
    seed_workers = _seed_worker_count(
        seed_workers, successor_mode=inputs.successor_rankings is not None
    )
    receipts = _train_seed_roster(
        bundle,
        inputs,
        architecture,
        arm,
        output_directory,
        seed_workers=seed_workers,
        sidecar_index=sidecar_index,
        ranking_weight=ranking_weight,
        initial_checkpoint=initial_checkpoint,
        qat_profile=profile,
        resume=resume,
    )
    passing = [
        receipt for receipt in receipts
        if receipt.get("offline_gate", {}).get("passed") is True
    ]
    pool = passing or receipts
    chosen = min(pool, key=_receipt_selection_key)
    chosen_reference_path = _seed_reference_path(
        output_directory, architecture, arm, int(chosen["seed"])
    )
    _reference_payload, chosen_reference = _load_canonical_json(
        chosen_reference_path, "selected seed reference"
    )
    selected_receipt_path = _output_artifact(
        output_directory,
        chosen_reference.get("receipt"),
        expected_sha256=chosen_reference.get("receipt_sha256"),
        label="selected seed receipt",
    )
    runtime_record = chosen["quantized_runtime"]
    source_size_eligible = architecture_deployment_eligible(
        architecture, generated_source_ascii_bytes
    )
    deployment_eligible = bool(
        arm.deployment_eligible
        and chosen["offline_gate"]["passed"]
        and source_size_eligible
    )
    body: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_bundle_body_sha256": bundle.body_sha256,
        "architecture": architecture.name,
        "arm": arm.name,
        "seed": chosen["seed"],
        "float_epoch": chosen["float_training"]["best_float_epoch"],
        "qat_epoch": chosen["quantized_training"]["selected_qat_epoch"],
        "qat_profile": profile.name,
        "qat_profile_contract": qat_profile_contract(profile),
        "qat_execution_evidence": chosen["quantized_training"],
        "scales": chosen["quantized_training"]["selected_scales"],
        "runtime": runtime_record,
        "selected_seed_receipt": {
            "path": _relative_artifact_path(
                selected_receipt_path, output_directory
            ),
            "sha256": sha256_file(selected_receipt_path),
            "body_sha256": chosen["body_sha256"],
        },
        "selected_seed_receipt_body_sha256": chosen["body_sha256"],
        "selected_seed_receipt_sha256": sha256_bytes(canonical_json_bytes(chosen)),
        "seed_receipt_body_sha256": [receipt["body_sha256"] for receipt in receipts],
        "ranking": [receipt["seed"] for receipt in sorted(receipts, key=_receipt_selection_key)],
        "float_validation": chosen["float_validation"],
        "quantized_validation": chosen["quantized_validation"],
        "offline_gate": chosen["offline_gate"],
        "status": chosen["status"],
        "deployment_eligible": deployment_eligible,
        "rank4_control_never_deployment_eligible": arm.name == "rank4-control",
        "source_size_eligibility": {
            "generated_source_ascii_bytes": generated_source_ascii_bytes,
            "maximum_ascii_bytes": CAPACITY_SOURCE_LIMIT,
            "passed": source_size_eligible,
            "conditional": architecture.name == "capacity-12x8",
        },
        "immutable_before_protected_test": True,
        "protected_tests_opened": False,
        "game_gated": False,
        "policy_head": False,
        "seed_execution_policy": {
            "seed_workers": seed_workers,
            "maximum_seed_workers": 2,
            "worker_model": "shared-read-only-input-thread-pool",
            "receipt_order": "fixed-seed-order",
            "selection_order": "validation-key-then-seed",
            "resume": "per-seed-content-addressed-reference",
            "per_seed_numerical_binding_includes_worker_count": False,
        },
    }
    if chosen.get("successor_ranking", {}).get("labels_present") is True:
        body["successor_ranking"] = dict(chosen["successor_ranking"])
    document = body_hashed(body)
    path = _write_content_addressed(
        output_directory / "selections",
        canonical_json_bytes(document),
        ".selection.json",
    )
    validate_selection(path, output_directory, bundle)
    return path


def validate_selection(
    selection_path: pathlib.Path,
    artifact_root: pathlib.Path,
    bundle: FrozenBundle,
) -> dict[str, Any]:
    payload, selection = _load_canonical_json(
        selection_path, "compact immutable selection"
    )
    if selection_path.name != f"{sha256_bytes(payload)}.selection.json":
        raise TrainingError("compact selection is not content addressed")
    verify_body_hash(selection, schema=SELECTION_SCHEMA, label="compact selection")
    runtime = selection.get("runtime")
    selected_receipt = selection.get("selected_seed_receipt")
    expected_fields = {
        "schema", "campaign_id", "source_bundle_body_sha256",
        "architecture", "arm", "seed", "float_epoch", "qat_epoch",
        "qat_profile", "qat_profile_contract", "qat_execution_evidence",
        "scales", "runtime", "selected_seed_receipt",
        "selected_seed_receipt_body_sha256", "selected_seed_receipt_sha256",
        "seed_receipt_body_sha256", "ranking", "float_validation",
        "quantized_validation", "offline_gate", "status",
        "deployment_eligible", "rank4_control_never_deployment_eligible",
        "source_size_eligibility", "immutable_before_protected_test",
        "protected_tests_opened", "game_gated", "policy_head",
        "seed_execution_policy", "body_sha256",
    }
    if "successor_ranking" in selection:
        expected_fields.add("successor_ranking")
    if (
        set(selection) != expected_fields
        or selection.get("campaign_id") != CAMPAIGN_ID
        or selection.get("source_bundle_body_sha256") != bundle.body_sha256
        or selection.get("architecture") not in ARCHITECTURES
        or selection.get("arm") not in ARMS
        or selection.get("seed") not in FIXED_SEEDS
        or selection.get("immutable_before_protected_test") is not True
        or selection.get("protected_tests_opened") is not False
        or selection.get("game_gated") is not False
        or selection.get("policy_head") is not False
        or not isinstance(runtime, dict)
        or set(runtime) != {"path", "sha256", "bytes"}
        or not isinstance(selected_receipt, dict)
        or set(selected_receipt) != {"path", "sha256", "body_sha256"}
    ):
        raise TrainingError("compact immutable selection contract changed")
    profile_name = selection.get("qat_profile")
    if not isinstance(profile_name, str):
        raise TrainingError("compact selection lost its QAT profile")
    profile_contract = validate_qat_profile_contract(
        selection.get("qat_profile_contract"), expected_name=profile_name
    )
    validate_qat_execution_evidence(
        selection.get("qat_execution_evidence"),
        expected_profile=profile_name,
    )
    seed_policy = selection.get("seed_execution_policy")
    seed_workers = seed_policy.get("seed_workers") if isinstance(
        seed_policy, Mapping
    ) else None
    if (
        not isinstance(seed_policy, Mapping)
        or set(seed_policy) != {
            "seed_workers", "maximum_seed_workers", "worker_model",
            "receipt_order", "selection_order", "resume",
            "per_seed_numerical_binding_includes_worker_count",
        }
        or isinstance(seed_workers, bool)
        or seed_workers not in (1, 2)
        or seed_policy.get("maximum_seed_workers") != 2
        or seed_policy.get("worker_model")
        != "shared-read-only-input-thread-pool"
        or seed_policy.get("receipt_order") != "fixed-seed-order"
        or seed_policy.get("selection_order") != "validation-key-then-seed"
        or seed_policy.get("resume")
        != "per-seed-content-addressed-reference"
        or seed_policy.get("per_seed_numerical_binding_includes_worker_count")
        is not False
        or ("successor_ranking" in selection and seed_workers != 2)
    ):
        raise TrainingError("compact seed execution policy changed")
    runtime_path = _output_artifact(
        artifact_root,
        runtime.get("path"),
        expected_sha256=runtime.get("sha256"),
        label="selected runtime",
    )
    architecture, _quantized, runtime_selection, _document = load_runtime(runtime_path)
    if runtime_path.stat().st_size != runtime.get("bytes"):
        raise TrainingError("selected runtime byte count changed")
    receipt_path = _output_artifact(
        artifact_root,
        selected_receipt["path"],
        expected_sha256=selected_receipt["sha256"],
        label="selection seed receipt",
    )
    receipt_payload, receipt = _load_canonical_json(
        receipt_path, "selection seed receipt"
    )
    verify_body_hash(
        receipt, schema=SEED_RECEIPT_SCHEMA, label="selection seed receipt"
    )
    validate_native_thread_execution(receipt.get("native_thread_execution"))
    if receipt.get("successor_ranking", {}).get("labels_present") is True:
        schedule_execution = validate_successor_schedule_execution(
            receipt.get("float_training"), receipt.get("quantized_training"),
            seed=receipt.get("seed"),
        )
        if receipt["successor_ranking"].get(
            "schedule_execution"
        ) != schedule_execution:
            raise TrainingError("selection seed successor schedule summary changed")
    if (
        architecture.name != selection["architecture"]
        or runtime_selection.get("arm") != selection["arm"]
        or runtime_selection.get("seed") != selection["seed"]
        or runtime_selection.get("float_epoch") != selection.get("float_epoch")
        or runtime_selection.get("qat_epoch") != selection.get("qat_epoch")
        or runtime_selection.get("source_bundle_body_sha256") != bundle.body_sha256
        or selected_receipt["body_sha256"] != receipt.get("body_sha256")
        or selection.get("selected_seed_receipt_body_sha256")
        != receipt.get("body_sha256")
        or selection.get("selected_seed_receipt_sha256")
        != sha256_bytes(receipt_payload)
        or receipt.get("architecture") != selection["architecture"]
        or receipt.get("arm") != selection["arm"]
        or receipt.get("seed") != selection["seed"]
        or receipt.get("float_validation") != selection["float_validation"]
        or receipt.get("quantized_validation") != selection["quantized_validation"]
        or receipt.get("offline_gate") != selection["offline_gate"]
        or receipt.get("qat_profile") != profile_name
        or receipt.get("qat_profile_contract") != profile_contract
        or receipt.get("quantized_training")
        != selection.get("qat_execution_evidence")
        or selection.get("successor_ranking")
        != (
            receipt.get("successor_ranking")
            if receipt.get("successor_ranking", {}).get("labels_present") is True
            else None
        )
    ):
        raise TrainingError("selected runtime identity disagrees with selection")
    runtime_scales = {
        name: float(_quantized.scales[name]) for name in ("w1", "w2", "w3")
    }
    if selection.get("scales") != runtime_scales:
        raise TrainingError("selected quantization scales changed")
    if selection["arm"] == "rank4-control" and selection.get("deployment_eligible"):
        raise TrainingError("matched Rank-4 control became deployment eligible")
    return selection


def evaluate_protected_tests(
    bundle: FrozenBundle,
    selection_path: pathlib.Path,
    artifact_root: pathlib.Path,
    output_directory: pathlib.Path,
) -> pathlib.Path:
    """Open tests only after immutable selection; never alter that selection."""

    selection = validate_selection(selection_path, artifact_root, bundle)
    selection_sha = sha256_file(selection_path)
    runtime_record = selection["runtime"]
    runtime_path = _output_artifact(
        artifact_root,
        runtime_record["path"],
        expected_sha256=runtime_record["sha256"],
        label="selected runtime",
    )
    architecture, quantized, _runtime_selection, _runtime = load_runtime(runtime_path)
    arm = ARMS[str(selection["arm"])]
    prefix = "search" if arm.new_source == "search" else "rank4"
    new_test_routes = (
        _safe_relative(bundle.routes[f"pilot_{prefix}_manifests"][2], "pilot test"),
        _safe_relative(bundle.routes[f"full_{prefix}_manifests"][2], "full test"),
    )
    new_test = concatenate_datasets(
        [load_shard(bundle, route, allow_protected=True) for route in new_test_routes],
        split="test",
    )
    canonical_test = concatenate_datasets(
        [
            load_shard(bundle, route, allow_protected=True)
            for route in bundle.canonical_routes("test")
        ],
        split="test",
    )
    # Protected tests are evaluated only against their stored targets.  No
    # teacher sidecar exists or may be generated for either dataset.
    diagnostic_arm = ARMS["search-target"]
    effective = quantized.effective()
    metrics = {
        "new_test": metrics_from_predictions(
            predict_dataset(
                effective, architecture, new_test, quantized=quantized
            ),
            new_test,
            diagnostic_arm,
        ),
        "canonical_test": metrics_from_predictions(
            predict_dataset(
                effective, architecture, canonical_test, quantized=quantized
            ),
            canonical_test,
            diagnostic_arm,
        ),
    }
    body: dict[str, object] = {
        "schema": PROTECTED_REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "selection_sha256": selection_sha,
        "selection_body_sha256": selection["body_sha256"],
        "runtime_sha256": runtime_record["sha256"],
        "metrics": metrics,
        "diagnostic_only": True,
        "selection_changed": False,
        "deployment_decision_changed": False,
    }
    document = body_hashed(body)
    return _write_content_addressed(
        output_directory,
        canonical_json_bytes(document),
        ".protected-report.json",
    )


def _parse_architecture(value: str) -> Architecture:
    try:
        return ARCHITECTURES[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            f"architecture must be one of {', '.join(ARCHITECTURES)}"
        ) from error


def _parse_arm(value: str) -> Arm:
    try:
        return ARMS[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            f"arm must be one of {', '.join(ARMS)}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-runtime")
    verify.add_argument("--runtime", type=pathlib.Path, required=True)

    sidecars = commands.add_parser("generate-sidecars")
    sidecars.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    sidecars.add_argument("--output-directory", type=pathlib.Path, required=True)

    audit = commands.add_parser("audit-inputs")
    audit.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    audit.add_argument("--output-directory", type=pathlib.Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    train.add_argument("--output-directory", type=pathlib.Path, required=True)
    train.add_argument("--architecture", type=_parse_architecture, required=True)
    train.add_argument("--arm", type=_parse_arm, required=True)
    train.add_argument("--input-audit", type=pathlib.Path, required=True)
    train.add_argument("--sidecar-index", type=pathlib.Path)
    train.add_argument("--successor-labels", type=pathlib.Path)
    train.add_argument("--initial-checkpoint", type=pathlib.Path)
    train.add_argument(
        "--ranking-weight",
        type=float,
        choices=RANKING_LOSS_WEIGHTS,
        default=0.0,
    )
    train.add_argument("--seed-workers", type=int, choices=(1, 2), default=1)
    train.add_argument(
        "--qat-profile",
        choices=tuple(QAT_PROFILES),
        default=STANDARD_QAT_PROFILE,
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument("--generated-source-ascii-bytes", type=int)

    post = commands.add_parser("post-selection-test")
    post.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    post.add_argument("--selection", type=pathlib.Path, required=True)
    post.add_argument("--artifact-root", type=pathlib.Path, required=True)
    post.add_argument("--output-directory", type=pathlib.Path, required=True)

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "verify-runtime":
            architecture, quantized, selection, document = load_runtime(
                arguments.runtime
            )
            result: object = {
                "schema": document["schema"],
                "architecture": architecture.name,
                "weight_counts": architecture.weight_counts,
                "scales": {
                    name: float(quantized.scales[name])
                    for name in ("w1", "w2", "w3")
                },
                "selection": selection,
                "runtime_sha256": sha256_file(arguments.runtime),
            }
        else:
            bundle = FrozenBundle.load(arguments.bundle_manifest)
            if arguments.command == "generate-sidecars":
                result = {
                    "sidecar_index": str(generate_teacher_sidecars(
                        bundle, arguments.output_directory
                    ))
                }
            elif arguments.command == "audit-inputs":
                result = {
                    "input_audit": str(generate_input_audit(
                        bundle, arguments.output_directory
                    ))
                }
            elif arguments.command == "train":
                result = {
                    "selection": str(train_arm_campaign(
                        bundle,
                        arguments.architecture,
                        arguments.arm,
                        arguments.output_directory,
                        sidecar_index=arguments.sidecar_index,
                        successor_labels=arguments.successor_labels,
                        ranking_weight=arguments.ranking_weight,
                        initial_checkpoint=arguments.initial_checkpoint,
                        seed_workers=arguments.seed_workers,
                        qat_profile=arguments.qat_profile,
                        input_audit=arguments.input_audit,
                        resume=arguments.resume,
                        generated_source_ascii_bytes=(
                            arguments.generated_source_ascii_bytes
                        ),
                    ))
                }
            else:
                result = {
                    "protected_report": str(evaluate_protected_tests(
                        bundle,
                        arguments.selection,
                        arguments.artifact_root,
                        arguments.output_directory,
                    ))
                }
    except (OSError, TrainingError) as error:
        parser.exit(1, f"compact value-BFM trainer failure: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
