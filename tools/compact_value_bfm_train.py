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
import copy
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
import weakref
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
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

_TRAINER_SOURCE_AT_IMPORT = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


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
CHANNEL_RUNTIME_SCHEMA = "papersoccer.compact-value-bfm-runtime.v2"
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
PARALLEL_NATIVE_EXECUTION_SCHEMA = "papersoccer.compact-value-bfm-native-thread-execution.v2"
KERNEL_EXECUTION_SCHEMA = "papersoccer.compact-value-bfm-native-kernel-execution.v1"

STANDARD_QAT_PROFILE = "standard-v1"
REFINED_ADAPTIVE_SCALES_QAT_PROFILE = "refined-adaptive-scales-v1"
RETENTION_FIRST_LOW_RATE_QAT_PROFILE = "retention-first-low-rate-v1"
CHANNEL_PREDICTION_QAT_PROFILE = "channel-prediction-qat-v1"
STUDENT_RIVALS_RETENTION_PROFILE = "student-rivals-retention-v1"
WARMUP_CONSISTENCY_QAT_PROFILE = "warmup-consistency-qat-v1"
STUDENT_PAIR_PROFILES = (STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE)
CONSISTENCY_TRAINING_OBJECTIVE = "mean-over-scalar-batches-of-label-loss-plus-lambda-student-rival-loss-plus-frozen-warmup-consistency-huber"
STUDENT_RIVAL_PAIR_POLICY = "current-student-top8-positive-gap-v1"
STUDENT_TRAINING_OBJECTIVE = "mean-over-scalar-batches-of-scalar-loss-plus-lambda-current-student-rival-group-mean"
CHANNEL_SCALE_COUNTS = {"w1": 12, "w2": 8, "w3": 1}


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
    qat_learning_rate: float = QAT_LEARNING_RATE


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
    RETENTION_FIRST_LOW_RATE_QAT_PROFILE: QATProfile(
        name=RETENTION_FIRST_LOW_RATE_QAT_PROFILE,
        scale_quantiles=REFINED_SCALE_QUANTILES,
        coordinate_search_passes=3,
        local_refinement_multipliers=REFINED_SCALE_MULTIPLIERS,
        local_refinement_passes=1,
        adapt_scales_after_each_epoch=True,
        adaptive_quantile_names=("p900", "p975", "p995", "p998"),
        adaptive_coordinate_passes=1,
        qat_learning_rate=0.0000625,
    ),
    STUDENT_RIVALS_RETENTION_PROFILE: QATProfile(
        name=STUDENT_RIVALS_RETENTION_PROFILE,
        scale_quantiles=REFINED_SCALE_QUANTILES,
        coordinate_search_passes=3,
        local_refinement_multipliers=REFINED_SCALE_MULTIPLIERS,
        local_refinement_passes=1,
        adapt_scales_after_each_epoch=True,
        adaptive_quantile_names=("p900", "p975", "p995", "p998"),
        adaptive_coordinate_passes=1,
        qat_learning_rate=0.0000625,
    ),
    WARMUP_CONSISTENCY_QAT_PROFILE: QATProfile(
        name=WARMUP_CONSISTENCY_QAT_PROFILE,
        scale_quantiles=REFINED_SCALE_QUANTILES,
        coordinate_search_passes=3,
        local_refinement_multipliers=REFINED_SCALE_MULTIPLIERS,
        local_refinement_passes=1,
        adapt_scales_after_each_epoch=True,
        adaptive_quantile_names=("p900", "p975", "p995", "p998"),
        adaptive_coordinate_passes=1,
        qat_learning_rate=0.0000625,
    ),
    CHANNEL_PREDICTION_QAT_PROFILE: QATProfile(
        name=CHANNEL_PREDICTION_QAT_PROFILE,
        scale_quantiles=REFINED_SCALE_QUANTILES,
        coordinate_search_passes=2,
        local_refinement_multipliers=(),
        local_refinement_passes=0,
        adapt_scales_after_each_epoch=True,
        adaptive_quantile_names=(),
        adaptive_coordinate_passes=2,
        qat_learning_rate=0.0000625,
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


def validation_prediction_settings(workers=1, seed_concurrency=1):
    if type(workers) is not int or workers not in (1, 2, 4):
        raise TrainingError("validation workers must be exactly1,2or4")
    if type(seed_concurrency) is not int or seed_concurrency not in (1, 2, 4) or workers * seed_concurrency > 4:
        raise TrainingError("combined active seed/prediction concurrency exceeds4 or is unsupported")
    if workers == 1:
        return None
    from tools import compact_value_bfm_prediction_pool as prediction
    return {"policy": prediction.policy(workers, seed_concurrency),
        "sources": prediction._source_records(sys.modules[__name__])}


def native_kernel_evidence(value):
    value = validate_native_thread_execution(value)
    if value["schema"] != NATIVE_THREAD_EXECUTION_SCHEMA:
        raise TrainingError("kernel evidence requires the coordinator's native limiter")
    result = dict(value)
    result["schema"] = KERNEL_EXECUTION_SCHEMA
    result.pop("native_threads_per_seed_maximum")
    result["native_threads_per_kernel_maximum"] = 1
    result["limiter_scope"] = "per-process-native-kernel-limit"
    return result


def validate_native_kernel_evidence(value):
    if not isinstance(value, Mapping) or value.get("schema") != KERNEL_EXECUTION_SCHEMA or value.get("native_threads_per_kernel_maximum") != 1 or value.get("limiter_scope") != "per-process-native-kernel-limit":
        raise TrainingError("native kernel execution evidence changed")
    legacy = dict(value)
    legacy["schema"] = NATIVE_THREAD_EXECUTION_SCHEMA
    legacy.pop("native_threads_per_kernel_maximum")
    legacy["native_threads_per_seed_maximum"] = 1
    legacy["limiter_scope"] = "outer-roster-established-before-seed-workers"
    validate_native_thread_execution(legacy)
    return dict(value)


def prediction_native_execution(base, settings):
    if settings is None:
        if base.get("schema") != NATIVE_THREAD_EXECUTION_SCHEMA:
            raise TrainingError("serial execution cannot claim prediction helper use")
        return validate_native_thread_execution(base)
    from tools import compact_value_bfm_prediction_pool as prediction
    prediction.validate_policy(settings["policy"])
    return {"schema": PARALLEL_NATIVE_EXECUTION_SCHEMA,
        "coordinator_kernel": native_kernel_evidence(base),
        "prediction_policy": settings["policy"],
        "maximum_active_numerical_streams": settings["policy"]["maximum_active_numerical_streams"],
        "parent_numerics_wait_for_all_helpers": True}


def validate_native_thread_execution(value: object) -> dict[str, object]:
    expected_environment = dict(NATIVE_THREAD_ENVIRONMENT)
    if isinstance(value, Mapping) and value.get("schema") == PARALLEL_NATIVE_EXECUTION_SCHEMA:
        from tools import compact_value_bfm_prediction_pool as prediction
        if set(value) != {"schema", "coordinator_kernel", "prediction_policy", "maximum_active_numerical_streams", "parent_numerics_wait_for_all_helpers"}:
            raise TrainingError("parallel native execution envelope changed")
        policy = prediction.validate_policy(value["prediction_policy"])
        validate_native_kernel_evidence(value["coordinator_kernel"])
        if value["maximum_active_numerical_streams"] != policy["maximum_active_numerical_streams"] or value["parent_numerics_wait_for_all_helpers"] is not True:
            raise TrainingError("parallel native aggregate concurrency changed")
        return dict(value)

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
            "QAT profile must be standard-v1, refined-adaptive-scales-v1 "
            "or retention-first-low-rate-v1 or channel-prediction-qat-v1 "
            "or student-rivals-retention-v1 or warmup-consistency-qat-v1"
        )
    return QAT_PROFILES[value]


def qat_profile_contract(value: str | QATProfile) -> dict[str, object]:
    """Build the exact body-hashed recipe sealed into plans and receipts."""

    profile = resolve_qat_profile(value)
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE:
        return _channel_qat_profile_contract()
    body = {
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
            "qat_learning_rate": profile.qat_learning_rate,
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
    }
    if profile.name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE):
        body["scale_selection"]["validation_objective"] = (
            "retention-feasibility-then-normalized-violation-sum-then-"
            "refined-ranking-key-then-lower-scale"
        )
        body["scale_selection"]["retention_policy"] = {
            "reference": "frozen-pre-QAT-float-validation",
            "predicate": "existing-offline-advancement-gate",
            "failure_score": "sum-eight-positive-normalized-violations",
            "pool_order": ["common_adjudicator", "canonical_validation"],
            "components_per_pool": [
                "max(0,(minimum_sign-sign)/minimum_sign)",
                "max(0,(huber-maximum_huber)/maximum_huber)",
                "max(0,(float_sign-sign-maximum_sign_loss)/maximum_sign_loss)",
                "max(0,(huber-float_huber*maximum_huber_ratio)/"
                "max(float_huber*maximum_huber_ratio,maximum_huber))",
            ],
            "sum": "math.fsum",
            "absolute_sign_denominator": "pool-minimum-sign-accuracy",
            "absolute_huber_denominator": "pool-maximum-weighted-huber",
            "relative_sign_denominator": MAXIMUM_SIGN_LOSS,
            "relative_huber_denominator": (
                "max-frozen-float-huber-times-ratio-and-pool-maximum-huber"
            ),
            "strict_sign_boundary": "failure-even-when-normalized-excess-is-zero",
            "nonfinite_reference_or_score": "reject",
            "gate_thresholds": {
                "common_minimum_sign": COMMON_MINIMUM_SIGN,
                "common_maximum_huber": COMMON_MAXIMUM_HUBER,
                "canonical_minimum_sign": CANONICAL_MINIMUM_SIGN,
                "canonical_maximum_huber": CANONICAL_MAXIMUM_HUBER,
                "maximum_sign_loss_exclusive": MAXIMUM_SIGN_LOSS,
                "maximum_huber_ratio_inclusive": MAXIMUM_HUBER_RATIO,
            },
        }
    if profile.name in STUDENT_PAIR_PROFILES:
        body["training_pair_policy"] = training_pair_policy_contract(profile)
        body["schedule"]["float_warmup_learning_rate"] = RANKING_FLOAT_LEARNING_RATE
    if profile.name == WARMUP_CONSISTENCY_QAT_PROFILE:
        body["warmup_consistency"] = warmup_consistency_contract()
    return body_hashed(body)


def training_pair_policy_contract(value: str | QATProfile) -> dict[str, object] | None:
    """New-only training policy; historical profile bodies never gain fields."""
    profile = resolve_qat_profile(value)
    if profile.name not in STUDENT_PAIR_PROFILES:
        return None
    return body_hashed({
        "schema": "papersoccer.compact-value-bfm-training-pair-policy.v1",
        "name": STUDENT_RIVAL_PAIR_POLICY,
        "base_profile": RETENTION_FIRST_LOW_RATE_QAT_PROFILE,
        "applies_to": ["float-warmup", "qat-1", "qat-2", "qat-3", "qat-4"],
        "teacher_best": "unchanged-float32-parent-frame-max-then-successor-ID-index",
        "eligible_rivals": "strict-positive-float32-teacher-gap;exhaustive-groups-only",
        "selection_order": "current-student-parent-prediction-descending;successor-ID;index",
        "pair_cap": RANKING_PAIR_CAP,
        "accumulation_order": "teacher-gap-descending;successor-ID;index",
        "student_prediction_source": "same-current-forward-used-for-training-gradient",
        "reselection": "every-active-ranking-group-evaluation;no-stale-selection-cache",
        "loss": "unchanged-logaddexp-gap-normalized-group-mean;lambda-before-backprop",
        "inactive_lambda_zero": "no-ranking-forward-or-pair-selection;unchanged-scalar-math",
        "validation_pair_policy": "unchanged-static-teacher-worst-eight",
        "training_objective_report": STUDENT_TRAINING_OBJECTIVE,
        "evidence": "per-epoch-counts-and-ordered-pair-prediction-and-batch-master-SHA256",
        "digest_encoding": "SHA256-of-LF-terminated-canonical-JSON-events-in-execution-order",
        "prediction_digest_dtype": "little-endian-float32-raw-current-forward-values",
        "evidence_replay_limit": "stream-digests-attest-reviewed-execution;no-raw-prediction-trace-or-independent-forward-replay",
        "automatic_training_or_gate_waiver": False,
    })


def _validate_training_pair_policy(value):
    if value is not None and (type(value) is not str or value != STUDENT_RIVAL_PAIR_POLICY):
        raise TrainingError("unknown training pair policy")
    return value


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


@dataclasses.dataclass(frozen=True)
class ChannelQuantizedWeights:
    """Explicit v2 output-channel scales; legacy scalar descriptors stay v1."""

    integer: Mapping[str, np.ndarray]
    scales: Mapping[str, np.ndarray]

    def __post_init__(self):
        if set(self.integer) != set(CHANNEL_SCALE_COUNTS) or set(self.scales) != set(CHANNEL_SCALE_COUNTS):
            raise TrainingError("channel quantization tensor roster changed")
        architecture = ARCHITECTURES["capacity-12x8"]
        codes, scales = {}, {}
        for name, count in CHANNEL_SCALE_COUNTS.items():
            code, scale = np.asarray(self.integer[name]), np.asarray(self.scales[name])
            if code.dtype != np.dtype("int8") or code.shape != architecture.shapes[name] or np.any(code < -3) or np.any(code > 3):
                raise TrainingError("channel quantization code dtype/shape/range changed")
            if scale.dtype != np.dtype("float32") or scale.shape != (count,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise TrainingError("channel scale dtype/shape/value changed")
            codes[name], scales[name] = code.copy(), scale.copy()
            codes[name].flags.writeable = False; scales[name].flags.writeable = False
        object.__setattr__(self, "integer", MappingProxyType(codes)); object.__setattr__(self, "scales", MappingProxyType(scales))
        if not all(np.all(np.isfinite(value)) for value in self.effective().values()):
            raise TrainingError("channel effective weights are nonfinite")

    def effective(self):
        return {name: self.integer[name].astype(np.float32) * self.scales[name] for name in CHANNEL_SCALE_COUNTS}


def _normalize_channel_scales(scales, *, canonical=False):
    if not isinstance(scales, Mapping) or set(scales) != set(CHANNEL_SCALE_COUNTS):
        raise TrainingError("channel scales are incomplete")
    result = {}
    for name, count in CHANNEL_SCALE_COUNTS.items():
        values = scales[name]
        if not isinstance(values, (list, tuple, np.ndarray)) or np.asarray(values).shape != (count,):
            raise TrainingError("channel scale axis/count changed")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            raise TrainingError("channel scale is not numeric")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise TrainingError("channel scales must be finite positive float32")
        if canonical and any(float(a) != float(b) for a, b in zip(array, values, strict=True)):
            raise TrainingError("channel scale is not canonical float32")
        result[name] = array.copy()
    return result


def quantize_channels(parameters, architecture, scales):
    if architecture.name != "capacity-12x8":
        raise TrainingError("channel quantization requires capacity-12x8")
    parameters = _validate_parameters(parameters, architecture)
    scales = _normalize_channel_scales(scales)
    with np.errstate(over="ignore"):
        integer = {name: np.clip(np.rint(parameters[name] / scales[name]), -3, 3).astype(np.int8) for name in CHANNEL_SCALE_COUNTS}
    return ChannelQuantizedWeights(integer, scales)


def _channel_scale_document(value):
    scales = value.scales if isinstance(value, ChannelQuantizedWeights) else _normalize_channel_scales(value, canonical=True)
    return {name: [float(scale) for scale in scales[name]] for name in CHANNEL_SCALE_COUNTS}


def scalar_channel_quantized_forward(quantized, architecture, active):
    if not isinstance(quantized, ChannelQuantizedWeights) or architecture.name != "capacity-12x8":
        raise TrainingError("channel scalar inference requires its v2 descriptor")
    indices = np.asarray(active, dtype=np.uint16)
    first = np.empty(12, dtype=np.float32)
    for output in range(12):
        accumulator = sum(int(quantized.integer["w1"][int(index), output]) for index in indices)
        value = np.float32(np.int32(accumulator) * quantized.scales["w1"][output])
        first[output] = np.float32(value * value) if value >= 0 else np.float32(LEAKY_SLOPE * value)
    second = np.empty(8, dtype=np.float32)
    for output in range(8):
        total = np.float32(0)
        for hidden in range(12):
            term = np.float32(np.float32(first[hidden] * quantized.scales["w2"][output]) * np.float32(quantized.integer["w2"][hidden, output]))
            total = np.float32(total + term)
        second[output] = total if total >= 0 else np.float32(LEAKY_SLOPE * total)
    total = np.float32(0)
    for hidden in range(8):
        term = np.float32(np.float32(second[hidden] * quantized.scales["w3"][0]) * np.float32(quantized.integer["w3"][hidden]))
        total = np.float32(total + term)
    return _fast_tanh_scalar(total)


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

    if isinstance(quantized, ChannelQuantizedWeights):
        return scalar_channel_quantized_forward(quantized, architecture, active)
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


def _channel_runtime_document(
    architecture: Architecture,
    quantized: ChannelQuantizedWeights,
    *,
    arm: Arm | str,
    seed: int,
    float_epoch: int,
    qat_epoch: int,
    source_bundle_body_sha256: str,
    qat_profile: str,
    qat_evidence_sha256: str,
) -> dict[str, object]:
    if (architecture.name != "capacity-12x8" or qat_profile != CHANNEL_PREDICTION_QAT_PROFILE
            or not valid_sha256(qat_evidence_sha256) or type(float_epoch) is not int or float_epoch != 1
            or type(seed) is not int or seed not in FIXED_SEEDS or type(qat_epoch) is not int or not 0 <= qat_epoch <= QAT_EPOCHS):
        raise TrainingError("v2 runtime lost its channel profile/training evidence")
    quantized = ChannelQuantizedWeights(quantized.integer, quantized.scales)
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
    scales = _channel_scale_document(quantized)
    body: dict[str, object] = {
        "schema": CHANNEL_RUNTIME_SCHEMA,
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
            "scheme": "symmetric-signed-three-bit-per-output-channel-fixed-scale",
            "granularity": "per-output-channel",
            "scale_axis": "output",
            "scale_counts": dict(CHANNEL_SCALE_COUNTS),
            "packing": PACKING,
            "scales": scales,
            "weight_counts": counts,
            "packed_byte_count": len(packed),
            "payload_sha256": sha256_bytes(packed),
            "payload_base64": base64.b64encode(packed).decode("ascii"),
        },
        "selection": {
            "qat_profile": qat_profile,
            "qat_evidence_sha256": qat_evidence_sha256,
            "arm": arm.name,
            "seed": seed,
            "float_epoch": float_epoch,
            "qat_epoch": qat_epoch,
            "source_bundle_body_sha256": source_bundle_body_sha256,
        },
    }
    return body_hashed(body)


def _validate_channel_runtime_document(
    value: Mapping[str, object],
) -> tuple[Architecture, QuantizedWeights, dict[str, object]]:
    verify_body_hash(value, schema=CHANNEL_RUNTIME_SCHEMA, label="compact runtime")
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
    if name != "capacity-12x8":
        raise TrainingError("v2 runtime requires capacity-12x8")
    if (architecture_value.get("biases") is not False
            or not isinstance(architecture_value.get("dimensions"), list)
            or any(type(value) is not int for value in architecture_value["dimensions"])):
        raise TrainingError("v2 architecture dimensions/bias flag have wrong types")
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
        "granularity", "scale_axis", "scale_counts",
        "weight_counts",
        "packed_byte_count",
        "payload_sha256",
        "payload_base64",
    }:
        raise TrainingError("compact runtime quantization fields changed")
    if (any(type(quantization.get(key)) is not int for key in ("bits", "minimum", "maximum", "packed_byte_count"))
            or not isinstance(quantization.get("scale_counts"), dict)
            or any(type(value) is not int for value in quantization["scale_counts"].values())
            or not isinstance(quantization.get("weight_counts"), dict)
            or any(type(value) is not int for value in quantization["weight_counts"].values())):
        raise TrainingError("v2 quantization counts must be exact integers")
    expected_static = {
        "bits": QUANTIZATION_BITS,
        "minimum": QUANTIZATION_MINIMUM,
        "maximum": QUANTIZATION_MAXIMUM,
        "scheme": "symmetric-signed-three-bit-per-output-channel-fixed-scale",
        "granularity": "per-output-channel",
        "scale_axis": "output",
        "scale_counts": dict(CHANNEL_SCALE_COUNTS),
        "packing": PACKING,
        "weight_counts": architecture.weight_counts,
    }
    if any(quantization.get(key) != expected for key, expected in expected_static.items()):
        raise TrainingError("compact runtime quantization contract changed")
    scales = _normalize_channel_scales(quantization.get("scales"), canonical=True)
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
            "arm", "qat_profile", "qat_evidence_sha256",
            "seed",
            "float_epoch",
            "qat_epoch",
            "source_bundle_body_sha256",
        }
        or selection.get("qat_profile") != CHANNEL_PREDICTION_QAT_PROFILE
        or not valid_sha256(selection.get("qat_evidence_sha256"))
        or selection.get("float_epoch") != 1
        or type(selection.get("seed")) is not int
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
    return architecture, ChannelQuantizedWeights(integer, scales), dict(selection)



def runtime_document(
    architecture: Architecture,
    quantized: QuantizedWeights,
    *,
    arm: Arm | str,
    seed: int,
    float_epoch: int,
    qat_epoch: int,
    source_bundle_body_sha256: str,
    qat_profile: str | None = None,
    qat_evidence_sha256: str | None = None,
) -> dict[str, object]:
    if isinstance(quantized, ChannelQuantizedWeights):
        return _channel_runtime_document(architecture, quantized, arm=arm, seed=seed, float_epoch=float_epoch,
            qat_epoch=qat_epoch, source_bundle_body_sha256=source_bundle_body_sha256,
            qat_profile=qat_profile, qat_evidence_sha256=qat_evidence_sha256)
    if qat_profile is not None or qat_evidence_sha256 is not None:
        raise TrainingError("v1 scalar runtime cannot carry v2 channel evidence")
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
    if isinstance(value, Mapping) and value.get("schema") == CHANNEL_RUNTIME_SCHEMA:
        return _validate_channel_runtime_document(value)
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
    # Inspect IDs only for exact maxima. Ascending indices retain the original
    # first-index decision when IDs also tie; empty input still reaches min().
    maxima = np.flatnonzero(values == np.max(values)) if values.size else ()
    return int(min(
        maxima,
        key=lambda index: group.successors[int(index)].successor_id,
    ))


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


def _student_ranking_pairs(group, parent_predictions, *, pair_cap=RANKING_PAIR_CAP):
    """Choose by current student score, then accumulate in canonical teacher order."""
    if isinstance(pair_cap, bool) or pair_cap != RANKING_PAIR_CAP:
        raise TrainingError("student rival pair cap must be exactly eight")
    scores = np.asarray(parent_predictions, dtype=np.float32)
    if scores.shape != (len(group.successors),) or not np.all(np.isfinite(scores)):
        raise TrainingError("student rival parent predictions are invalid")
    teacher = _teacher_parent_values(group)
    best = _deterministic_best(group, teacher)
    if not group.successors_exhaustive:
        return best, (), np.asarray([], dtype=np.float32)
    candidates = []
    for index, successor in enumerate(group.successors):
        gap = float(teacher[best] - teacher[index])
        if gap > 0.0:
            candidates.append((index, gap, successor.successor_id))
    candidates.sort(key=lambda row: (-float(scores[row[0]]), row[2], row[0]))
    chosen = candidates[:pair_cap]
    chosen.sort(key=lambda row: (-row[1], row[2], row[0]))
    alternatives = tuple(row[0] for row in chosen)
    gaps = np.asarray([row[1] for row in chosen], dtype=np.float32)
    current = _deterministic_best(group, scores)
    if float(teacher[best] - teacher[current]) > 0.0 and current not in alternatives:
        raise TrainingError("student rival selection omitted its regrettable argmax")
    return best, alternatives, gaps


def student_rival_loss_gradient(
    group: CompleteTurnGroup,
    predictions: np.ndarray,
    *,
    pair_cap: int = RANKING_PAIR_CAP,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Training-only rival choice; arithmetic and canonical reduction match legacy."""

    parent_predictions, signs = _parent_frame_values(group, predictions)
    best, alternatives, gaps = _student_ranking_pairs(group, parent_predictions, pair_cap=pair_cap)
    gradient_parent = np.zeros(len(group.successors), dtype=np.float32)
    if not alternatives:
        return 0.0, gradient_parent, {
            "group_id": group.group_id,
            "training_pair_policy": STUDENT_RIVAL_PAIR_POLICY,
            "teacher_best_successor_index": best,
            "selected_successor_indices": list(alternatives),
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
        "training_pair_policy": STUDENT_RIVAL_PAIR_POLICY,
        "teacher_best_successor_index": best,
        "selected_successor_indices": list(alternatives),
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


class _StudentRivalEpochEvidence:
    """Constant retained state; hash current decisions without retaining predictions."""
    def __init__(self, *, phase, seed, schedule_epoch, ranking_weight, scalar_batches):
        self.phase, self.seed, self.epoch = phase, seed, schedule_epoch
        self.weight = _ranking_weight(ranking_weight)
        self.scalar_batches = scalar_batches
        self.batch = -1
        self.batch_digest = hashlib.sha256()
        self.pair_digest = hashlib.sha256()
        self.counts = dict(group_evaluations=0, selected_pairs=0,
            changed_static_subset_groups=0, regrettable_argmax_groups=0,
            regrettable_argmax_included_groups=0)

    def begin_batch(self, batch, parameters, architecture):
        if batch != self.batch + 1 or batch >= self.scalar_batches:
            raise TrainingError("student rival batch evidence is discontinuous")
        self.batch = batch
        self.batch_digest.update(canonical_json_bytes({"batch": batch,
            "parameters": _parameter_identity(parameters, architecture)}) + b"\n")

    def observe(self, group, predictions, report):
        if self.weight == 0.0 or self.batch < 0:
            raise TrainingError("inactive student rival policy selected pairs")
        scores, _ = _parent_frame_values(group, predictions)
        teacher = _teacher_parent_values(group)
        best, static, _ = _ranking_pairs(group)
        chosen = tuple(report["selected_successor_indices"])
        current = _deterministic_best(group, scores)
        regret = float(teacher[best] - teacher[current])
        if (report["training_pair_policy"] != STUDENT_RIVAL_PAIR_POLICY
                or report["teacher_best_successor_index"] != best or not chosen
                or len(chosen) != report["pair_count"]
                or (regret > 0.0 and current not in chosen)):
            raise TrainingError("student rival execution invariant failed")
        event = {"group_ordinal": self.counts["group_evaluations"], "batch": self.batch,
            "group_id": group.group_id, "teacher_best_index": best,
            "teacher_best_id": group.successors[best].successor_id,
            "selected_indices": list(chosen), "selected_ids": report["selected_successor_ids"],
            "teacher_gaps": report["teacher_gaps"], "current_argmax_index": current,
            "current_argmax_id": group.successors[current].successor_id,
            "regret": regret, "current_predictions_sha256": sha256_bytes(
                np.asarray(predictions, dtype="<f4").tobytes())}
        self.pair_digest.update(canonical_json_bytes(event) + b"\n")
        self.counts["group_evaluations"] += 1
        self.counts["selected_pairs"] += len(chosen)
        self.counts["changed_static_subset_groups"] += int(set(chosen) != set(static))
        self.counts["regrettable_argmax_groups"] += int(regret > 0.0)
        self.counts["regrettable_argmax_included_groups"] += int(regret > 0.0 and current in chosen)

    def finish(self):
        if self.batch + 1 != self.scalar_batches:
            raise TrainingError("student rival batch evidence is incomplete")
        return body_hashed({"schema": "papersoccer.compact-value-bfm-student-rival-epoch.v1",
            "policy": STUDENT_RIVAL_PAIR_POLICY, "phase": self.phase, "seed": self.seed,
            "schedule_epoch": self.epoch, "loss_weight": self.weight,
            "active": self.weight > 0.0, "scalar_batches": self.scalar_batches,
            "executed_scalar_batches": self.batch + 1, **self.counts,
            "missed_regrettable_argmax_groups": 0,
            "batch_master_sha256": self.batch_digest.hexdigest(),
            "ordered_pair_prediction_sha256": self.pair_digest.hexdigest(),
            "raw_prediction_trace_retained": False})


def ranking_microbatch_loss_gradient(
    groups: Sequence[CompleteTurnGroup],
    predictions: np.ndarray,
    *, training_pair_policy: str | None = None,
    pair_evidence: _StudentRivalEpochEvidence | None = None,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Average normalized group objectives without changing the external lambda."""

    policy = _validate_training_pair_policy(training_pair_policy)
    if pair_evidence is not None and policy is None:
        raise TrainingError("legacy ranking cannot record student pair evidence")
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
        loss_function = (pairwise_successor_ranking_loss_gradient if policy is None else student_rival_loss_gradient)
        loss, gradient, report = loss_function(
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
        if pair_evidence is not None:
            pair_evidence.observe(group, predictions[offset:stop], report)
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
    # This helper expects active/cache from the same forward call, which has
    # already accessed every row. Preserve all dense reductions and retained
    # scatter order; omit only exact zeros. NaN/Inf rows remain selected.
    for row in np.flatnonzero(np.any(first_pre_gradient != 0, axis=1)):
        np.add.at(gradients["w1"], active[row], first_pre_gradient[row])
    return gradients


def warmup_consistency_contract():
    return body_hashed({
        "schema": "papersoccer.compact-value-bfm-warmup-consistency-policy.v1",
        "base_profile": STUDENT_RIVALS_RETENTION_PROFILE,
        "phase": "qat-only", "coefficient": 1.0, "huber_delta": float(HUBER_DELTA),
        "teacher": "own-frozen-genuine-float-warmup;stop-gradient",
        "inputs": "existing-64-new-192-anchor-TRAIN-scalar-batch-only",
        "weights": "existing-separately-normalized-25percent-new-75percent-anchor",
        "formula": "labels-plus-lambda-ranking-plus-weighted-huber-to-frozen-warmup",
        "derivative": "add-once-to-scalar-output-gradient-before-existing-backprop-clip-adamw",
        "unchanged": ["warmup", "labels", "ranking", "epochs", "learning-rate", "per-layer-3bit", "scale-and-epoch-selection", "gates"],
        "optional_audit_control": "when-bound-match-warmup-checkpoint-metrics-and-initial-QAT-scales-codes-before-optimizer",
        "trace": "per-epoch-counts-loss-components-and-ordered-batch-row-prediction-derivative-hashes",
        "replay_limit": "source-bound-execution-digests;no-independent-forward-or-gradient-replay",
    })


def _validate_warmup_consistency_policy(value):
    verify_body_hash(value, schema="papersoccer.compact-value-bfm-warmup-consistency-policy.v1", label="consistency policy")
    if value != warmup_consistency_contract():
        raise TrainingError("warmup consistency policy changed")


def _consistency_binding_identity(binding):
    successor = binding.get("successor_ranking", {})
    return {
        "source_bundle_body_sha256": binding.get("source_bundle_body_sha256"),
        "architecture": binding.get("architecture"), "arm": binding.get("arm"),
        "seed": binding.get("seed"), "datasets": binding.get("datasets"),
        "source_routes": binding.get("source_routes"),
        "ranking": {name: successor.get(name) for name in (
            "schema", "artifact_sha256", "body_sha256", "source_bundle_body_sha256",
            "teacher", "train_groups", "validation_groups", "loss_weight")},
        "initial_parameters": successor.get("initial_checkpoint", {}).get("parameters"),
    }


def build_warmup_consistency_expectation(control_receipt_path: pathlib.Path):
    """Read one completed no-penalty control; never run a warmup or prediction."""
    path = pathlib.Path(control_receipt_path).absolute()
    if path.resolve() != path or not path.is_file():
        raise TrainingError("consistency control receipt is absent or redirected")
    payload, receipt = _load_canonical_json(path, "consistency no-penalty control")
    verify_body_hash(receipt, schema=SEED_RECEIPT_SCHEMA, label="consistency control")
    if path.name != sha256_bytes(payload) + ".seed-receipt.json":
        raise TrainingError("consistency control is not content addressed")
    profile = receipt.get("qat_profile")
    weight = _ranking_weight(receipt.get("successor_ranking", {}).get("loss_weight"))
    if profile != STUDENT_RIVALS_RETENTION_PROFILE and not (
            profile == RETENTION_FIRST_LOW_RATE_QAT_PROFILE and weight == 0.0):
        raise TrainingError("consistency requires a no-penalty student-rival or inactive scalar control")
    architecture = ARCHITECTURES[receipt["architecture"]]
    if architecture.name != "capacity-12x8" or receipt.get("arm") != "search-target":
        raise TrainingError("consistency control architecture/arm changed")
    completed = _load_seed_receipt_from_reference(path.parent.parent,
        _seed_reference_path(path.parent.parent, architecture, ARMS["search-target"], receipt["seed"]), receipt["binding"])
    if canonical_json_bytes(completed) != payload:
        raise TrainingError("consistency control differs from its completed seed reference")
    report = receipt["quantized_training"]
    validate_qat_execution_evidence(report, expected_profile=profile,
        float_validation_reference=receipt["float_validation"])
    validate_successor_schedule_execution(receipt["float_training"], report, seed=receipt["seed"])
    checkpoint = receipt["float_checkpoint"]
    checkpoint_path = _output_artifact(path.parent.parent, checkpoint["path"],
        expected_sha256=checkpoint["sha256"], label="consistency warmup control")
    parameters = load_float_checkpoint(checkpoint_path, architecture)
    scales = report["scale_search"]["selected_scales"]
    if scales != report["history"][0]["fixed_scales"]:
        raise TrainingError("consistency control initial scales changed")
    quantized = quantize_fixed(parameters, architecture, scales)
    warmup = receipt["float_training"]
    if (warmup.get("best_float_epoch") != 1 or len(warmup.get("history", [])) != 1
            or warmup.get("optimizer", {}).get("learning_rate") != RANKING_FLOAT_LEARNING_RATE
            or warmup.get("validation") != receipt["float_validation"]
            or any(warmup.get("per_layer_update_evidence", {}).get(name, {}).get("after_sha256")
                != sha256_bytes(parameters[name].tobytes()) for name in parameters)):
        raise TrainingError("consistency control warmup evidence changed")
    return body_hashed({
        "schema": "papersoccer.compact-value-bfm-warmup-consistency-expectation.v1",
        "control_receipt": {"path": str(path), "sha256": sha256_bytes(payload), "bytes": len(payload)},
        "control_profile": profile, "input_binding": _consistency_binding_identity(receipt["binding"]),
        "warmup": {"parameters": _parameter_identity(parameters, architecture),
            "checkpoint_sha256": checkpoint["sha256"], "checkpoint_bytes": checkpoint["bytes"],
            "validation": receipt["float_validation"], "optimizer": warmup["optimizer"],
            "batching": warmup["batching"], "epoch": 1},
        "initial_quantization": {"scales": dict(scales),
            "code_sha256": {name: sha256_bytes(v.tobytes()) for name, v in quantized.integer.items()},
            "validation": report["pre_qat_validation"]},
    })


def _validate_consistency_expectation(value, *, binding=None, reread_control=False):
    if not isinstance(value, Mapping):
        raise TrainingError("warmup consistency requires its no-penalty control expectation")
    verify_body_hash(value, schema="papersoccer.compact-value-bfm-warmup-consistency-expectation.v1", label="consistency expectation")
    if set(value) != {"schema", "control_receipt", "control_profile", "input_binding", "warmup", "initial_quantization", "body_sha256"}:
        raise TrainingError("consistency expectation fields changed")
    control = value["control_receipt"]
    if (not isinstance(control, Mapping) or set(control) != {"path", "sha256", "bytes"}
            or not valid_sha256(control.get("sha256")) or type(control.get("bytes")) is not int or control["bytes"] <= 0
            or not isinstance(control.get("path"), str) or not pathlib.Path(control["path"]).is_absolute()):
        raise TrainingError("consistency control record is invalid")
    if binding is not None and value["input_binding"] != _consistency_binding_identity(binding):
        raise TrainingError("consistency no-penalty control input/seed/initialization binding differs")
    if reread_control and build_warmup_consistency_expectation(pathlib.Path(control["path"])) != value:
        raise TrainingError("consistency no-penalty control or expected warmup/grid changed")
    # Copy so a caller cannot relabel the running teacher or expectation in place.
    return json.loads(canonical_json_bytes(value))


@dataclasses.dataclass(frozen=True, init=False)
class _FrozenWarmupTeacher:
    _payloads: tuple[bytes, ...]
    _identity_bytes: bytes
    architecture: Architecture

    def __init__(self, parameters, architecture):
        normalized = _validate_parameters(parameters, architecture)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "_payloads", tuple(np.asarray(normalized[name], dtype="<f4").tobytes(order="C") for name in ("w1", "w2", "w3")))
        object.__setattr__(self, "_identity_bytes", canonical_json_bytes(_parameter_identity(normalized, architecture)))
        self.verify()

    @property
    def parameters(self):
        shapes = tuple(self.architecture.shapes[name] for name in ("w1", "w2", "w3"))
        return {name: np.frombuffer(payload, dtype="<f4").reshape(shape)
            for name, payload, shape in zip(("w1", "w2", "w3"), self._payloads, shapes)}

    @property
    def identity(self):
        return json.loads(self._identity_bytes)

    def verify(self):
        if (type(self._identity_bytes) is not bytes or type(self._payloads) is not tuple
                or len(self._payloads) != 3 or any(type(v) is not bytes for v in self._payloads)
                or canonical_json_bytes(_parameter_identity(self.parameters, self.architecture)) != self._identity_bytes):
            raise TrainingError("frozen warmup consistency teacher changed")

    def predict(self, active, architecture):
        self.verify()
        if architecture != self.architecture:
            raise TrainingError("consistency teacher architecture changed")
        prediction, _ = forward(self.parameters, architecture, active)
        self.verify()
        return np.frombuffer(np.asarray(prediction, dtype="<f4").tobytes(), dtype="<f4")


def _check_consistency_warmup(float_result, inputs, architecture, arm, seed, weight, expectation):
    expected = _validate_consistency_expectation(expectation, reread_control=True)
    ib = expected["input_binding"]
    labels = inputs.successor_rankings
    actual_datasets = {name: dataset_identity(getattr(inputs, name)) for name in (
        "new", "anchor", "common_adjudicator", "canonical_validation")}
    if (ib["seed"] != seed or ib["architecture"]["name"] != architecture.name or ib["arm"] != dataclasses.asdict(arm)
            or ib["datasets"] != actual_datasets or ib["ranking"]["loss_weight"] != weight
            or labels is None or ib["ranking"]["body_sha256"] != labels.body_sha256
            or ib["initial_parameters"] != float_result.report.get("initialization", {}).get("parameters")):
        raise TrainingError("consistency actual warmup/input/seed differs from control")
    warmup = expected["warmup"]
    payload = deterministic_npz({name: np.asarray(value, dtype="<f4") for name, value in float_result.parameters.items()})
    if (float_result.epoch != 1 or _parameter_identity(float_result.parameters, architecture) != warmup["parameters"]
            or sha256_bytes(payload) != warmup["checkpoint_sha256"] or len(payload) != warmup["checkpoint_bytes"]
            or float_result.metrics != warmup["validation"] or float_result.report.get("optimizer") != warmup["optimizer"]
            or float_result.report.get("batching") != warmup["batching"]):
        raise TrainingError("consistency warmup checkpoint/metrics differ before QAT")
    return expected


def _check_consistency_initial_grid(quantized, metrics, expectation):
    actual = {"scales": {name: float(value) for name, value in quantized.scales.items()},
        "code_sha256": {name: sha256_bytes(v.tobytes()) for name, v in quantized.integer.items()}, "validation": metrics}
    if actual != expectation["initial_quantization"]:
        raise TrainingError("consistency initial QAT scales/codes/metrics differ before optimizer")
    return body_hashed({"schema": "papersoccer.compact-value-bfm-warmup-consistency-pre-update.v1",
        "expectation_sha256": expectation["body_sha256"], "warmup": expectation["warmup"],
        "initial_quantization": actual, "optimizer_steps": 0, "checked_before_optimizer_construction": True})


class _WarmupConsistencyEpochEvidence:
    def __init__(self, teacher, *, seed, qat_epoch, input_identity, expectation_sha256):
        self.teacher = teacher
        self.seed = seed
        self.qat_epoch = qat_epoch
        self.input_identity = json.loads(canonical_json_bytes(input_identity))
        self.expectation_sha256 = expectation_sha256
        self.digest = hashlib.sha256()
        self.losses = []
        self.nonzero_derivative_batches = 0
        self.nonzero_teacher_disagreement_rows = 0
        self.derivative_l2_sum = 0.0
        self.teacher.verify()
        self.reference_identity = self.teacher.identity

    def record(self, *, parameters, architecture, new_rows, anchor_rows, active,
               predictions, targets, weights, derivative, label_loss, consistency_loss, weighted_ranking_loss):
        self.teacher.verify()
        if self.teacher.identity != self.reference_identity:
            raise TrainingError("consistency epoch teacher identity changed")
        components = (float(label_loss), float(consistency_loss), float(weighted_ranking_loss))
        if any(not math.isfinite(v) or v < 0 for v in components) or not np.all(np.isfinite(derivative)):
            raise TrainingError("consistency loss/gradient is nonfinite")
        self.nonzero_derivative_batches += int(np.any(derivative != 0))
        self.nonzero_teacher_disagreement_rows += int(np.count_nonzero(predictions != targets))
        norm = math.sqrt(float(np.sum(np.asarray(derivative, dtype=np.float64) ** 2, dtype=np.float64)))
        self.derivative_l2_sum += norm
        event = {"batch": len(self.losses), "seed": self.seed, "qat_epoch": self.qat_epoch,
            "reference": self.reference_identity, "master": _parameter_identity(parameters, architecture),
            "new_rows_sha256": _array_identity(np.asarray(new_rows, dtype="<i8")),
            "anchor_rows_sha256": _array_identity(np.asarray(anchor_rows, dtype="<i8")),
            "active_features_sha256": sha256_bytes(canonical_json_bytes([list(map(int, row)) for row in active])),
            "quantized_prediction_sha256": _array_identity(predictions), "teacher_prediction_sha256": _array_identity(targets),
            "mixed_weights_sha256": _array_identity(weights), "consistency_output_gradient_sha256": _array_identity(derivative),
            "label_loss": components[0], "consistency_loss": components[1], "weighted_ranking_loss": components[2]}
        self.digest.update(canonical_json_bytes(event))
        self.losses.append(components)

    def finish(self):
        self.teacher.verify()
        count = len(self.losses)
        if not count:
            raise TrainingError("consistency epoch performed no batches")
        means = {name: float(np.mean([row[index] for row in self.losses]))
            for index, name in enumerate(("label_loss", "consistency_loss", "weighted_ranking_loss"))}
        means["total_objective"] = float(np.mean([(row[0] + row[1]) + row[2] for row in self.losses]))
        return body_hashed({"schema": "papersoccer.compact-value-bfm-warmup-consistency-epoch.v1",
            "policy": warmup_consistency_contract(), "seed": self.seed, "qat_epoch": self.qat_epoch,
            "schedule_epoch": self.qat_epoch + 1, "expectation_sha256": self.expectation_sha256,
            "reference_before": self.reference_identity, "reference_after": self.teacher.identity,
            "training_inputs": self.input_identity, "batches": count, "new_rows": count * 64, "anchor_rows": count * 192,
            "teacher_forward_calls": count, "consistency_gradient_additions": count,
            "nonzero_derivative_batches": self.nonzero_derivative_batches,
            "nonzero_teacher_disagreement_rows": self.nonzero_teacher_disagreement_rows,
            "consistency_output_gradient_l2_sum": self.derivative_l2_sum,
            "mean_loss_components": means, "ordered_batch_prediction_gradient_sha256": self.digest.hexdigest(),
            "reference_optimizer_steps": 0, "heldout_reference_predictions": False,
            "raw_prediction_trace_retained": False, "independent_gradient_replay_claimed": False})


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
    quantization_granularity: str = "per-layer",
    ranking_group: CompleteTurnGroup | None = None,
    ranking_groups: Sequence[CompleteTurnGroup] | None = None,
    ranking_weight: float = 0.0,
    training_pair_policy: str | None = None,
    pair_evidence: _StudentRivalEpochEvidence | None = None,
    warmup_teacher: _FrozenWarmupTeacher | None = None,
    consistency_evidence: _WarmupConsistencyEpochEvidence | None = None,
) -> float:
    _validate_training_pair_policy(training_pair_policy)
    if warmup_teacher is not None or consistency_evidence is not None:
        if (not isinstance(warmup_teacher, _FrozenWarmupTeacher) or fixed_scales is None
                or quantization_granularity != "per-layer" or training_pair_policy != STUDENT_RIVAL_PAIR_POLICY
                or not isinstance(consistency_evidence, _WarmupConsistencyEpochEvidence)
                or consistency_evidence.teacher is not warmup_teacher):
            raise TrainingError("warmup consistency requires its QAT-only per-layer teacher/evidence context")
        if inputs.new.split != "train" or inputs.anchor.split != "train":
            raise TrainingError("warmup consistency teacher requires TRAIN scalar sources")
        warmup_teacher.verify()
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
    if quantization_granularity not in {"per-layer", "per-output-channel"}:
        raise TrainingError("unknown training quantization granularity")
    quantized = (
        (quantize_channels(parameters, architecture, fixed_scales)
         if quantization_granularity == "per-output-channel" else quantize_fixed(parameters, architecture, fixed_scales))
        if fixed_scales is not None else None
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
    label_loss = loss
    consistency_loss = 0.0
    teacher_predictions = consistency_derivative = None
    if warmup_teacher is not None:
        teacher_predictions = warmup_teacher.predict(active, architecture)
        consistency_loss, consistency_derivative = _weighted_huber_loss_gradient(
            predictions, teacher_predictions, weights)
        output_gradient = output_gradient + consistency_derivative
        loss += consistency_loss
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
    weighted_ranking_loss = 0.0
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
                ranking_microbatch, ranking_predictions,
                **({"training_pair_policy": training_pair_policy, "pair_evidence": pair_evidence}
                   if training_pair_policy is not None else {}),
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
        weighted_ranking_loss = ranking_weight * ranking_loss
        objective += weighted_ranking_loss
    if consistency_evidence is not None:
        consistency_evidence.record(parameters=parameters, architecture=architecture,
            new_rows=new_rows, anchor_rows=anchor_rows, active=active, predictions=predictions,
            targets=teacher_predictions, weights=weights, derivative=consistency_derivative,
            label_loss=label_loss, consistency_loss=consistency_loss, weighted_ranking_loss=weighted_ranking_loss)
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


def _float_ranking_group_structure(groups):
    """Small guards for exact group order and immutable mapped backing arrays."""
    result = []
    stores = {}
    owners = {}
    containers = {}
    for group in groups:
        successors = group.successors
        kind = type(successors)
        if kind not in containers:
            module = sys.modules.get(kind.__module__)
            maintained = (
                module is not None
                and kind.__module__ in {
                    "tools.compact_value_bfm_ranking_store", "compact_value_bfm_ranking_store"
                }
                and pathlib.Path(getattr(module, "__file__", "")).resolve()
                    == TOOL_DIRECTORY / "compact_value_bfm_ranking_store.py"
                and getattr(module, "MappedSuccessors", None) is kind
            )
            containers[kind] = module if maintained else None
        module = containers[kind]
        store = getattr(successors, "store", None)
        mapped = (
            module is not None
            and getattr(module, "RankingStore", None) is type(store)
        )
        backing = None
        if mapped:
            store_id = id(store)
            if store_id not in stores:
                arrays = tuple(getattr(store, name) for name in ("metadata", "indices", "transcripts"))
                readonly = all(isinstance(value, np.memmap) and value.mode == "r"
                               and not value.flags.writeable for value in arrays)
                signatures = tuple((id(value), repr(value.dtype.descr), value.shape, value.strides,
                    bool(value.flags.writeable), getattr(value, "mode", None),
                    str(getattr(value, "filename", "")), getattr(value, "offset", None),
                    id(getattr(value, "_mmap", None)), int(value.ctypes.data)) for value in arrays)
                stores[store_id] = (readonly, str(store.index), store.document.get("body_sha256"), signatures)
                owners[store_id] = (weakref.ref(store), *(weakref.ref(value) for value in arrays))
            mapped = stores[store_id][0]
            backing = (store_id, successors.begin, successors.end, successors.root_termination)
        result.append((group.group_id, group.parent_mover, group.successors_exhaustive,
                       id(successors), len(successors), mapped, backing))
    return tuple(result), stores, owners


def _eager_float_ranking_group_content(group):
    """Check mutable/eager content only after the normal comparability checks.

    No feature or prediction tensors are retained. Production mapped labels use
    the read-only backing-array guards instead of hashing every feature here.
    """
    digest = hashlib.sha256()
    for successor in group.successors:
        digest.update(canonical_json_bytes({
            "successor_id": successor.successor_id,
            "teacher_value": float(successor.teacher_value),
            "value_mover": successor.value_mover,
            "active_sha256": _array_identity(successor.active),
        }))
    return digest.hexdigest()


class _FloatRankingDecisionCache:
    """Private to one scale search with fixed parameters and immutable groups.

    Store only exact best indices and small identity/content guards. Weak group
    references prevent object-ID reuse without retaining feature/store objects.
    """
    __slots__ = ("architecture", "parameters", "groups", "structure", "owners", "best", "comparable", "content")

    def __init__(self, parameters, architecture, groups):
        self.architecture = architecture
        self.parameters = _parameter_identity(parameters, architecture)
        self.groups = tuple(weakref.ref(group) for group in groups)
        structure, stores, self.owners = _float_ranking_group_structure(groups)
        self.structure = structure, stores
        self.best = [None] * len(groups)
        self.comparable = [None] * len(groups)
        self.content = [None] * len(groups)

    def validate(self, parameters, architecture, groups):
        structure, stores, owners = _float_ranking_group_structure(groups)
        owners_changed = self.owners.keys() != owners.keys() or any(
            old() is None or old() is not new()
            for key in self.owners if key in owners
            for old, new in zip(self.owners[key], owners[key])
        )
        if (architecture != self.architecture
                or _parameter_identity(parameters, architecture) != self.parameters
                or len(groups) != len(self.groups)
                or any(reference() is not group for reference, group in zip(self.groups, groups))
                or (structure, stores) != self.structure or owners_changed):
            raise TrainingError("float-ranking cache parameters, architecture or group sequence changed")

    @property
    def immutable_mapped_groups(self):
        return bool(self.groups) and all(row[5] for row in self.structure[0])

    def skipped(self, index):
        if self.comparable[index] is True:
            raise TrainingError("float-ranking cache group comparability changed")
        self.comparable[index] = False

    def decision(self, index, group):
        if self.comparable[index] is False:
            raise TrainingError("float-ranking cache group comparability changed")
        self.comparable[index] = True
        if not self.structure[0][index][5]:
            content = _eager_float_ranking_group_content(group)
            if self.content[index] is not None and self.content[index] != content:
                raise TrainingError("float-ranking cache eager group content changed")
            self.content[index] = content
        return self.best[index]


def _new_float_ranking_decision_cache(parameters, architecture, inputs):
    labels = getattr(inputs, "successor_rankings", None)
    return None if labels is None else _FloatRankingDecisionCache(parameters, architecture, labels.validation)


def successor_ranking_metrics(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    groups: Sequence[CompleteTurnGroup],
    *,
    quantized: QuantizedWeights | None = None,
    _float_best_cache: _FloatRankingDecisionCache | None = None,
) -> dict[str, float | int | bool]:
    if not groups:
        raise TrainingError("successor ranking metrics require nonempty groups")
    if _float_best_cache is not None:
        if quantized is None:
            raise TrainingError("float-ranking decision cache requires quantized validation")
        _float_best_cache.validate(parameters, architecture, groups)
    agreements = 0
    regrets = []
    losses = []
    pair_count = 0
    flips = 0
    comparable_groups = 0
    singleton_groups = 0
    skipped_nonexhaustive_groups = 0
    skipped_tied_groups = 0
    for group_index, group in enumerate(groups):
        if len(group.successors) == 1:
            singleton_groups += 1
        if not group.successors_exhaustive:
            skipped_nonexhaustive_groups += 1
            if _float_best_cache is not None:
                _float_best_cache.skipped(group_index)
            continue
        _best, alternatives, _gaps = _ranking_pairs(group)
        if not alternatives:
            skipped_tied_groups += int(len(group.successors) > 1)
            if _float_best_cache is not None:
                _float_best_cache.skipped(group_index)
            continue
        comparable_groups += 1
        active = tuple(successor.active for successor in group.successors)
        cached_best = None if _float_best_cache is None else _float_best_cache.decision(group_index, group)
        if cached_best is None:
            float_raw, _float_cache = forward(
                parameters, architecture, active, quantized=None
            )
        if quantized is not None:
            evaluated_raw, _quantized_cache = forward(
                parameters, architecture, active, quantized=quantized
            )
        else:
            evaluated_raw = float_raw
        teacher_parent = _teacher_parent_values(group)
        if cached_best is None:
            float_parent, _float_signs = _parent_frame_values(group, float_raw)
        evaluated_parent, _evaluated_signs = _parent_frame_values(
            group, evaluated_raw
        )
        teacher_best = _deterministic_best(group, teacher_parent)
        if cached_best is None:
            float_best = _deterministic_best(group, float_parent)
            if _float_best_cache is not None:
                _float_best_cache.best[group_index] = float_best
        else:
            float_best = cached_best
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
    _float_best_cache: _FloatRankingDecisionCache | None = None,
    _prediction_executor=None,
) -> dict[str, dict[str, Any]]:
    prediction = predict_dataset if _prediction_executor is None else _prediction_executor.predict
    ranking_weight = _ranking_weight(ranking_weight)
    if _float_best_cache is not None and inputs.successor_rankings is None:
        raise TrainingError("float-ranking decision cache requires successor groups")
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
            prediction(
                parameters,
                architecture,
                inputs.common_adjudicator,
                quantized=quantized,
            ),
            inputs.common_adjudicator,
            arm,
        ),
        "canonical_validation": metrics_from_predictions(
            prediction(
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
                _float_best_cache=_float_best_cache,
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


@dataclasses.dataclass(frozen=True)
class _FrozenRetentionReference:
    """Immutable scalar report bytes, fixed before any QAT optimizer step."""

    payload: bytes

    def __post_init__(self):
        try:
            report = json.loads(self.payload)
            if (
                not isinstance(report, Mapping)
                or canonical_json_bytes(report) != self.payload
                or not _finite_metric_report(report)
            ):
                raise ValueError("noncanonical or nonfinite reference")
            for name in ("common_adjudicator", "canonical_validation"):
                for key in ("sign_accuracy", "weighted_huber"):
                    if not math.isfinite(float(report[name][key])):
                        raise ValueError("nonfinite reference metric")
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingError("retention reference requires finite float validation") from error

    def metrics(self):
        return json.loads(self.payload)

    def document(self):
        return {
            "schema": "papersoccer.compact-value-bfm-frozen-retention-reference.v1",
            "float_validation_sha256": sha256_bytes(self.payload),
            "float_validation": self.metrics(),
        }


def _retention_reference(profile, value):
    if profile.name not in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE):
        if value is not None:
            raise TrainingError("float retention reference requires retention-first QAT")
        return None
    if isinstance(value, _FrozenRetentionReference):
        return value
    if not isinstance(value, Mapping):
        raise TrainingError("retention-first QAT requires frozen pre-QAT float validation")
    try:
        return _FrozenRetentionReference(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise TrainingError("retention reference requires finite float validation") from error


def _retention_violation_key(report, reference):
    """Use unchanged gate feasibility, then finite normalized positive deficits."""
    if not isinstance(report, Mapping) or not _finite_metric_report(report):
        raise TrainingError("retention-first QAT requires finite validation metrics")
    frozen = reference.metrics()
    try:
        passed = offline_advancement_gate(frozen, report)["passed"]
        violations = []
        for name, minimum_sign, maximum_huber in (
            ("common_adjudicator", COMMON_MINIMUM_SIGN, COMMON_MAXIMUM_HUBER),
            ("canonical_validation", CANONICAL_MINIMUM_SIGN, CANONICAL_MAXIMUM_HUBER),
        ):
            candidate, baseline = report[name], frozen[name]
            sign = float(candidate["sign_accuracy"])
            huber = float(candidate["weighted_huber"])
            float_sign = float(baseline["sign_accuracy"])
            float_huber = float(baseline["weighted_huber"])
            relative_huber_cap = float_huber * MAXIMUM_HUBER_RATIO
            components = (
                (minimum_sign - sign) / minimum_sign,
                (huber - maximum_huber) / maximum_huber,
                (float_sign - sign - MAXIMUM_SIGN_LOSS) / MAXIMUM_SIGN_LOSS,
                (huber - relative_huber_cap) / max(relative_huber_cap, maximum_huber),
            )
            if not all(math.isfinite(value) for value in (*components, relative_huber_cap)):
                raise ValueError("nonfinite normalized retention violation")
            violations.extend(max(0.0, value) for value in components)
        score = math.fsum(violations)
        if not math.isfinite(score):
            raise ValueError("nonfinite retention violation sum")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TrainingError("retention-first QAT violation metrics are invalid") from error
    # Strict sign loss uses the gate result, even at a zero-excess boundary.
    return (0.0, 0.0) if passed else (1.0, score)


def _qat_validation_key(
    report: Mapping[str, Mapping[str, float | int]],
    profile: QATProfile,
    *, float_validation_reference: object = None,
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
    if profile.name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE):
        reference = _retention_reference(profile, float_validation_reference)
        return (*_retention_violation_key(report, reference), *result)
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
    training_pair_policy: str | None = None,
    _prediction_executor=None,
) -> FloatTrainingResult:
    prediction_arguments = {} if _prediction_executor is None else {"_prediction_executor": _prediction_executor}
    if seed not in FIXED_SEEDS:
        raise TrainingError("compact training requires one of the three fixed seeds")
    if maximum_epochs <= 0 or maximum_epochs > MAX_FLOAT_EPOCHS or patience <= 0:
        raise TrainingError("float training epoch configuration is invalid")
    policy = _validate_training_pair_policy(training_pair_policy)
    if policy is not None and (inputs.successor_rankings is None or learning_rate != RANKING_FLOAT_LEARNING_RATE
            or weight_decay != WEIGHT_DECAY or patience != 1 or type(patience) is not int):
        raise TrainingError("student rivals require the exact successor warmup recipe")
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
        pair_evidence = None if policy is None else _StudentRivalEpochEvidence(
            phase="float-warmup", seed=seed, schedule_epoch=epoch,
            ranking_weight=ranking_weight, scalar_batches=batch_count)
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
            if pair_evidence is not None:
                pair_evidence.begin_batch(batch_index, parameters, architecture)
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
                **({"training_pair_policy": policy, "pair_evidence": pair_evidence} if policy is not None else {}),
            ))
        validation = evaluate_validation_pair(
            parameters,
            architecture,
            inputs,
            arm,
            ranking_weight=ranking_weight,
            **prediction_arguments,
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
            **({"training_objective_weighted_huber": float(np.mean(losses))}
               if policy is None else {"training_total_objective": float(np.mean(losses)),
                   "training_total_objective_definition": STUDENT_TRAINING_OBJECTIVE}),
            "validation": validation,
            "coverage": coverage,
            "ranking_schedule_coverage": ranking_coverage,
            "eligible": eligible,
            **({"training_pair_selection": pair_evidence.finish()} if pair_evidence is not None else {}),
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
    if policy is not None:
        training_report["training_pair_policy"] = training_pair_policy_contract(STUDENT_RIVALS_RETENTION_PROFILE)
    return FloatTrainingResult(
        parameters=best_parameters,
        epoch=best_epoch,
        metrics=best_metrics,
        report=training_report,
    )


CHANNEL_QAT_EXECUTION_SCHEMA = "papersoccer.compact-value-bfm-channel-qat-execution.v1"
CHANNEL_FIXTURE_POLICY = {
    "new_rows": 1024, "anchor_rows": 3072, "rows": 4096,
    "generator": "PCG64", "seed": 20260908, "replace": False,
    "draw_order": ["new", "anchor"], "within_pool_order": "ascending",
    "fixture_order": ["new", "anchor"], "labels_used_for_calibration": False,
}
CHANNEL_CALIBRATION_POLICY = {
    "sweeps": 2, "coordinate_events": 42,
    "order": "w1-output0..11,w2-output0..7,w3-output0",
    "candidates": "incumbent-first;current-master-robust14-lower-rank-quantiles;maxabs-div3;exact-float32-dedup",
    "objective": "sum(float64(weights)*(float64(candidate)-float64(frozen-pre-QAT-float))**2)/sum(float64(weights))",
    "tie_break": "first-incumbent-on-exact-tie", "updates": "greedy-immediate",
    "early_stop": False, "epsilon": None, "batch_size": 4096,
    "maximum_prediction_rows": 672, "prediction_dtype": "little-endian-float32",
    "evidence": "streamed-memory-mapped-candidate-matrix;unused-rows-zero",
    "maximum_prediction_payload_bytes": 11010048,
    "heldout_used_for_calibration": False,
    "replay_limit": "artifact replay verifies objectives, choices and code states; inference replay is separate",
}


def _channel_qat_profile_contract():
    retention = qat_profile_contract(RETENTION_FIRST_LOW_RATE_QAT_PROFILE)["scale_selection"]["retention_policy"]
    return body_hashed({
        "schema": "papersoccer.compact-value-bfm-qat-profile.v2",
        "qat_profile": CHANNEL_PREDICTION_QAT_PROFILE,
        "quantization": {"bits": 3, "minimum": -3, "maximum": 3,
            "scheme": "symmetric-signed-three-bit-per-output-channel-fixed-scale",
            "granularity": "per-output-channel", "scale_axis": "output", "scale_counts": dict(CHANNEL_SCALE_COUNTS),
            "runtime_schema": CHANNEL_RUNTIME_SCHEMA, "fake_quantized_layers": ["w1", "w2", "w3"],
            "straight_through_master_weights": True},
        "schedule": {"float_warmup_epochs": 1, "float_warmup_learning_rate": RANKING_FLOAT_LEARNING_RATE,
            "qat_epochs": 4, "qat_learning_rate": .0000625, "all_layers_trainable_each_qat_epoch": True},
        "training_fixture": copy.deepcopy(CHANNEL_FIXTURE_POLICY),
        "scale_selection": {"lower_rank_quantiles": [{"name": n, "numerator": a, "denominator": b} for n, a, b in REFINED_SCALE_QUANTILES],
            "initialization": "per-output-weight-MSE-projection;float32-effective-weights;float64-MSE;first-candidate-ties",
            "all_zero_scale": 1., "positive_underflow_scale": "smallest-positive-float32",
            "initial_calibration": copy.deepcopy(CHANNEL_CALIBRATION_POLICY),
            "after_each_qat_epoch": copy.deepcopy(CHANNEL_CALIBRATION_POLICY),
            "target_reference": "frozen-pre-QAT-float-predictions-on-fixed-training-fixture",
            "current_master_codes_recomputed_for_every_candidate": True},
        "epoch_selection": {"validation_objective": "retention-feasibility-then-normalized-violation-sum-then-refined-ranking-key",
            "retention_policy": retention, "exact_ties": "earlier-trained-epoch", "pre-QAT-selectable": False},
    })


class _ChannelTrainingFixture:
    __slots__ = ("indptr", "indices", "weights")

    def __init__(self, active, weights):
        self.indptr = np.zeros(len(active) + 1, dtype="<i8")
        self.indptr[1:] = np.cumsum([len(row) for row in active], dtype=np.int64)
        self.indices = np.concatenate(active).astype("<u2", copy=True)
        self.weights = np.asarray(weights, dtype="<f4").copy()
        if self.weights.shape != (4096,) or not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise TrainingError("channel fixture requires4096 positive finite training weights")
        for array in (self.indptr, self.indices, self.weights): array.flags.writeable = False

    def __len__(self): return len(self.weights)

    def active_rows(self, rows):
        return tuple(self.indices[self.indptr[i]:self.indptr[i + 1]] for i in rows)


def _channel_sample_indices(new_count, anchor_count):
    if type(new_count) is not int or type(anchor_count) is not int or new_count < 1024 or anchor_count < 3072:
        raise TrainingError("channel calibration requires1024new and3072filtered-anchor training rows")
    generator = np.random.Generator(np.random.PCG64(20260908))
    return {"new": np.sort(generator.choice(new_count, 1024, replace=False)).astype("<i8"),
            "anchor": np.sort(generator.choice(anchor_count, 3072, replace=False)).astype("<i8")}


def _channel_artifact(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _channel_read_artifact(value, suffix):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"} or not valid_sha256(value["sha256"]):
        raise TrainingError("channel calibration artifact binding is malformed")
    path = pathlib.Path(value["path"])
    _reject_path_markers(path, "channel calibration artifact")
    if path.resolve() != path or not path.is_file() or path.name != value["sha256"] + suffix or _channel_artifact(path) != dict(value):
        raise TrainingError("channel calibration artifact path/bytes changed")
    return path


def _channel_fixture(inputs, directory):
    # Only the two already audited training datasets are touched here. The
    # returned object has no labels, Dataset references or held-out attributes.
    if inputs.new.split != "train" or inputs.anchor.split != "train":
        raise TrainingError("channel calibration fixture crossed training splits")
    indices = _channel_sample_indices(len(inputs.new), len(inputs.anchor))
    fixture = _ChannelTrainingFixture((*inputs.new.active_rows(indices["new"]), *inputs.anchor.active_rows(indices["anchor"])),
        np.concatenate((inputs.new.weights[indices["new"]], inputs.anchor.weights[indices["anchor"]])))
    identity = body_hashed({"schema": "papersoccer.compact-value-bfm-channel-fixture.v1", "policy": copy.deepcopy(CHANNEL_FIXTURE_POLICY),
        "datasets": {"new": dataset_identity(inputs.new), "anchor": dataset_identity(inputs.anchor)},
        "sampled_indices": {name: list(map(int, values)) for name, values in indices.items()},
        "arrays": {name: _array_identity(getattr(fixture, name)) for name in ("indptr", "indices", "weights")}})
    buffer = io.BytesIO()
    np.savez(buffer, indptr=fixture.indptr, indices=fixture.indices, weights=fixture.weights,
             new_indices=indices["new"], anchor_indices=indices["anchor"])
    artifact = _write_content_addressed(directory, buffer.getvalue(), ".channel-fixture.npz")
    return fixture, {"identity": identity, "artifact": _channel_artifact(artifact)}


def _channel_prediction_reference(parameters, architecture, fixture, fixture_document, directory):
    fixture_arrays = {name: _array_identity(getattr(fixture, name)) for name in ("indptr", "indices", "weights")}
    if fixture_arrays != fixture_document["identity"]["arrays"]:
        raise TrainingError("channel reference fixture differs from its audited identity")
    prediction = predict_dataset(parameters, architecture, fixture, batch_size=4096)
    prediction.flags.writeable = False
    path = _write_content_addressed(directory, _array_npy_bytes(prediction), ".channel-reference.npy")
    document = body_hashed({"schema": "papersoccer.compact-value-bfm-channel-prediction-reference.v1",
        "fixture_identity_sha256": fixture_document["identity"]["body_sha256"],
        "fixture_array_sha256": fixture_arrays,
        "pre_qat_parameters": _parameter_identity(parameters, architecture),
        "prediction_array_sha256": _array_identity(prediction), "prediction": _channel_artifact(path),
        "frozen_across_initial_and_four_epoch_calibrations": True})
    return prediction, document


def _channel_candidates(values):
    result = []
    for scale in (*robust_scale_candidates(values, quantiles=REFINED_SCALE_QUANTILES), np.float32(float(np.max(np.abs(values))) / 3)):
        if np.isfinite(scale) and scale > 0 and all(scale != prior for prior in result): result.append(np.float32(scale))
    if not result: result = [np.nextafter(np.float32(0), np.float32(1)) if np.any(values != 0) else np.float32(1)]
    return tuple(result)


def _channel_coordinate_candidates(values, incumbent):
    result = []
    for scale in (np.float32(incumbent), *_channel_candidates(values)):
        if all(scale != previous for previous in result): result.append(scale)
    return tuple(result)


def _channel_weight_projection(parameters, architecture):
    parameters = _validate_parameters(parameters, architecture); scales = {}; reports = []
    for name, count in CHANNEL_SCALE_COUNTS.items():
        scales[name] = np.empty(count, dtype=np.float32)
        for channel in range(count):
            values = parameters[name] if name == "w3" else parameters[name][:, channel]
            trials = []
            for ordinal, scale in enumerate(_channel_candidates(values)):
                with np.errstate(over="ignore"):
                    codes = np.clip(np.rint(values / scale), -3, 3).astype(np.int8)
                    effective = codes.astype(np.float32) * scale
                delta = values.astype(np.float64) - effective.astype(np.float64)
                error = float(np.mean(delta * delta, dtype=np.float64))
                if not math.isfinite(error): raise TrainingError("channel weight projection error is nonfinite")
                trials.append({"ordinal": ordinal, "scale": float(scale), "weight_mse": error})
            selected = min(trials, key=lambda row: (row["weight_mse"], row["ordinal"]))
            scales[name][channel] = np.float32(selected["scale"])
            reports.append({"layer": name, "channel": channel, "trials": trials, "selected_ordinal": selected["ordinal"]})
    return quantize_channels(parameters, architecture, scales), reports


def _channel_prediction_objective(candidate, reference, weights):
    if (candidate.dtype != np.dtype("float32") or reference.dtype != np.dtype("float32")
            or candidate.shape != (4096,) or reference.shape != candidate.shape or weights.shape != candidate.shape
            or not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(weights)) or np.any(weights <= 0)):
        raise TrainingError("channel prediction objective requires finite4096-row evidence")
    delta = candidate.astype(np.float64) - reference.astype(np.float64); weight = weights.astype(np.float64)
    result = float(np.sum(weight * (delta * delta), dtype=np.float64) / np.sum(weight, dtype=np.float64))
    if not math.isfinite(result): raise TrainingError("channel prediction objective is nonfinite")
    return result


class _ChannelPredictionWriter:
    """Bounded disk matrix; no list of candidate prediction tensors is kept."""
    def __init__(self, directory):
        self.directory = pathlib.Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        file = tempfile.NamedTemporaryFile(dir=self.directory, prefix=".channel-", suffix=".npy", delete=False)
        self.path = pathlib.Path(file.name); file.close()
        self.matrix = np.lib.format.open_memmap(self.path, mode="w+", dtype="<f4", shape=(672, 4096))
        self.matrix[:] = 0; self.rows = 0

    def append(self, prediction):
        if self.rows >= 672 or prediction.shape != (4096,) or prediction.dtype != np.dtype("float32"):
            raise TrainingError("channel prediction audit exceeded its fixed bound")
        self.matrix[self.rows] = prediction; self.rows += 1
        return self.rows - 1

    def finish(self):
        self.matrix.flush(); self.matrix._mmap.close(); self.matrix = None
        digest = sha256_file(self.path); target = self.directory / (digest + ".channel-predictions.npy")
        os.chmod(self.path, 0o444)
        try: os.link(self.path, target)
        except FileExistsError:
            if sha256_file(target) != digest: raise TrainingError("channel prediction artifact collision")
        self.path.unlink()
        return _channel_artifact(target)

    def close(self):
        if self.matrix is not None:
            self.matrix._mmap.close(); self.matrix = None
        self.path.unlink(missing_ok=True)


def _channel_calibrate(parameters, architecture, fixture, reference, reference_document, starting_scales, directory, *, epoch):
    parameters = _validate_parameters(parameters, architecture)
    verify_body_hash(reference_document, schema="papersoccer.compact-value-bfm-channel-prediction-reference.v1", label="channel prediction reference")
    if (reference.flags.writeable or reference.dtype != np.dtype("float32") or reference.shape != (4096,)
            or not np.all(np.isfinite(reference)) or _array_identity(reference) != reference_document.get("prediction_array_sha256")
            or {name: _array_identity(getattr(fixture, name)) for name in ("indptr", "indices", "weights")} != reference_document.get("fixture_array_sha256")):
        raise TrainingError("channel calibration target/fixture differs from the frozen reference")
    before = _parameter_identity(parameters, architecture); reference_hash = _array_identity(reference)
    current = quantize_channels(parameters, architecture, starting_scales); starting = current
    writer = _ChannelPredictionWriter(directory); history = []; previous_score = None; previous_prediction = None
    try:
        for sweep in (1, 2):
            for name, count in CHANNEL_SCALE_COUNTS.items():
                for channel in range(count):
                    values = parameters[name] if name == "w3" else parameters[name][:, channel]
                    incumbent = current.scales[name][channel]; trials = []; best = None
                    for ordinal, scale in enumerate(_channel_coordinate_candidates(values, incumbent)):
                        scales = {key: value.copy() for key, value in current.scales.items()}; scales[name][channel] = scale
                        quantized = quantize_channels(parameters, architecture, scales)
                        prediction = predict_dataset(parameters, architecture, fixture, quantized=quantized, batch_size=4096)
                        objective = _channel_prediction_objective(prediction, reference, fixture.weights)
                        row = writer.append(prediction)
                        if ordinal == 0 and previous_score is not None and (objective != previous_score or prediction.tobytes() != previous_prediction.tobytes()):
                            raise TrainingError("channel incumbent prediction continuity changed")
                        trials.append({"ordinal": ordinal, "scale": float(scale), "prediction_row": row,
                            "scale_state_sha256": sha256_bytes(canonical_json_bytes(_channel_scale_document(quantized))), "objective": objective})
                        if best is None or objective < best[0]: best = (objective, ordinal, quantized, prediction)
                    current = best[2]; previous_score = best[0]; previous_prediction = best[3]
                    history.append({"sweep": sweep, "layer": name, "channel": channel, "incumbent_scale": float(incumbent),
                        "trials": trials, "selected_ordinal": best[1], "selected_scale": float(current.scales[name][channel]), "selected_objective": best[0]})
        artifact = writer.finish()
    finally: writer.close()
    if _parameter_identity(parameters, architecture) != before or _array_identity(reference) != reference_hash:
        raise TrainingError("channel calibration changed masters or frozen target predictions")
    master = write_float_checkpoint(pathlib.Path(directory), parameters, architecture)
    return current, {"schema": "papersoccer.compact-value-bfm-channel-calibration.v1", "qat_epoch": epoch,
        "policy": copy.deepcopy(CHANNEL_CALIBRATION_POLICY), "training_reference": reference_document,
        "master_parameters": before, "master_checkpoint": _channel_artifact(master),
        "starting_scales": _channel_scale_document(starting), "selected_scales": _channel_scale_document(current),
        "starting_code_sha256": {name: sha256_bytes(value.tobytes()) for name, value in starting.integer.items()},
        "selected_code_sha256": {name: sha256_bytes(value.tobytes()) for name, value in current.integer.items()},
        "trials": history, "prediction_rows": writer.rows, "prediction_matrix": artifact,
        "initial_objective": history[0]["trials"][0]["objective"], "selected_objective": previous_score,
        "parameters_unchanged_by_calibration": True, "heldout_read_by_calibration": False}


def select_fixed_scales(
    parameters: Mapping[str, np.ndarray],
    architecture: Architecture,
    inputs: TrainingInputs,
    arm: Arm,
    *,
    ranking_weight: float = 0.0,
    qat_profile: str | QATProfile = STANDARD_QAT_PROFILE,
    float_validation_reference: object = None,
    _prediction_executor=None,
) -> tuple[QuantizedWeights, dict[str, object]]:
    prediction_arguments = {} if _prediction_executor is None else {"_prediction_executor": _prediction_executor}
    parameters = _validate_parameters(parameters, architecture)
    profile = resolve_qat_profile(qat_profile)
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE:
        raise TrainingError("channel profile requires training-only channel calibration, not scalar heldout scale search")
    reference = _retention_reference(profile, float_validation_reference)
    selection_arguments = {} if reference is None else {"float_validation_reference": reference}
    float_best_cache = _new_float_ranking_decision_cache(parameters, architecture, inputs)
    cache_arguments = {} if float_best_cache is None else {"_float_best_cache": float_best_cache}
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
                    **cache_arguments,
                    **prediction_arguments,
                )
                key = (*_qat_validation_key(metrics, profile, **selection_arguments), float(candidate))
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
                    **cache_arguments,
                    **prediction_arguments,
                )
                key = (*_qat_validation_key(metrics, profile, **selection_arguments), float(candidate))
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
        **cache_arguments,
        **prediction_arguments,
    )
    report = {
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
    if reference is not None:
        report["retention_reference"] = reference.document()
    return selected, report


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
    float_validation_reference: object = None,
    _prediction_executor=None,
) -> tuple[QuantizedWeights, dict[str, object]]:
    """Locally reselect scales after one QAT epoch from current master weights."""
    prediction_arguments = {} if _prediction_executor is None else {"_prediction_executor": _prediction_executor}

    parameters = _validate_parameters(parameters, architecture)
    profile = resolve_qat_profile(profile)
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE:
        raise TrainingError("channel profile requires training-only channel calibration, not scalar heldout scale search")
    reference = _retention_reference(profile, float_validation_reference)
    selection_arguments = {} if reference is None else {"float_validation_reference": reference}
    if not profile.adapt_scales_after_each_epoch:
        raise TrainingError("fixed QAT profile cannot perform adaptive reselection")
    if not 1 <= qat_epoch <= QAT_EPOCHS:
        raise TrainingError("adaptive scale epoch is outside the QAT schedule")
    # Each call follows a different QAT master state; never share decisions
    # with the initial search, another epoch, or another seed.
    float_best_cache = _new_float_ranking_decision_cache(parameters, architecture, inputs)
    cache_arguments = {} if float_best_cache is None else {"_float_best_cache": float_best_cache}
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
                    **cache_arguments,
                    **prediction_arguments,
                )
                key = (*_qat_validation_key(metrics, profile, **selection_arguments), float(candidate))
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
        **cache_arguments,
        **prediction_arguments,
    )
    before = {
        name: float(np.float32(starting_scales[name]))
        for name in ("w1", "w2", "w3")
    }
    after = {
        name: float(selected.scales[name]) for name in ("w1", "w2", "w3")
    }
    report = {
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
    if reference is not None:
        report["retention_reference"] = reference.document()
    return selected, report


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
    if isinstance(before, ChannelQuantizedWeights) or isinstance(after, ChannelQuantizedWeights):
        if not isinstance(before, ChannelQuantizedWeights) or not isinstance(after, ChannelQuantizedWeights):
            raise TrainingError("quantized update evidence mixed scalar/channel types")
        return {name: {"granularity": "per-output-channel", "codes": int(after.integer[name].size),
            "changed_codes": int(np.count_nonzero(before.integer[name] != after.integer[name])),
            "changed": bool(np.any(before.integer[name] != after.integer[name])),
            "before_sha256": sha256_bytes(before.integer[name].tobytes()),
            "after_sha256": sha256_bytes(after.integer[name].tobytes()),
            "output_scales": _channel_scale_document(after)[name]} for name in CHANNEL_SCALE_COUNTS}
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


def _run_channel_prediction_qat(
    float_result: FloatTrainingResult,
    inputs: TrainingInputs,
    architecture: Architecture,
    arm: Arm,
    seed: int,
    *,
    qat_epochs: int = QAT_EPOCHS,
    ranking_weight: float = 0.0,
    qat_profile: str | QATProfile = CHANNEL_PREDICTION_QAT_PROFILE,
    calibration_directory: pathlib.Path,
    original_parameters: Mapping[str, np.ndarray],
) -> QuantizedTrainingResult:
    if qat_epochs != QAT_EPOCHS:
        raise TrainingError("compact deployment requires exactly four QAT epochs")
    profile = resolve_qat_profile(qat_profile)
    if profile.name != STANDARD_QAT_PROFILE and (
        architecture.name != "capacity-12x8"
        or inputs.successor_rankings is None
    ):
        raise TrainingError(
            "refined adaptive QAT requires successor-labeled capacity-12x8"
        )
    reference = _retention_reference(
        profile,
        float_result.metrics,
    )
    selection_arguments = {} if reference is None else {"float_validation_reference": reference}
    ranking_weight = _ranking_weight(ranking_weight)
    all_ranking_groups = (
        () if inputs.successor_rankings is None else inputs.successor_rankings.train
    )
    ranking_groups, density_report = _density_weighted_ranking_groups(
        all_ranking_groups
    )
    if ranking_weight > 0.0 and not ranking_groups:
        raise TrainingError("positive ranking loss has no QAT training groups")
    if (float_result.epoch != 1 or float_result.report.get("best_float_epoch") != 1
            or len(float_result.report.get("history", [])) != 1
            or float_result.report.get("optimizer", {}).get("maximum_epochs") != 1
            or float_result.report.get("optimizer", {}).get("learning_rate") != RANKING_FLOAT_LEARNING_RATE
            or float_result.report.get("optimizer", {}).get("weight_decay") != WEIGHT_DECAY
            or float_result.report.get("validation") != float_result.metrics):
        raise TrainingError("channel QAT requires the unchanged one-epoch float warmup")
    directory = pathlib.Path(calibration_directory).resolve()
    _reject_path_markers(directory, "channel calibration output")
    directory.mkdir(parents=True, exist_ok=True)
    original_parameters = _validate_parameters(original_parameters, architecture)
    if float_result.report.get("initialization", {}).get("parameters") != _parameter_identity(original_parameters, architecture):
        raise TrainingError("channel QAT original initialization differs from frozen warmup")
    fixture, fixture_document = _channel_fixture(inputs, directory)
    frozen_prediction, prediction_document = _channel_prediction_reference(
        float_result.parameters, architecture, fixture, fixture_document, directory)
    projected, weight_projection = _channel_weight_projection(float_result.parameters, architecture)
    pre_qat, scale_report = _channel_calibrate(float_result.parameters, architecture, fixture,
        frozen_prediction, prediction_document, projected.scales, directory, epoch=0)
    original_checkpoint = write_float_checkpoint(directory, original_parameters, architecture)
    selected: ChannelQuantizedWeights | None = None
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
        master, learning_rate=profile.qat_learning_rate, weight_decay=WEIGHT_DECAY
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
                quantization_granularity="per-output-channel",
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
        applied_scales = _channel_scale_document(fixed_scales)
        candidate, adaptive_scale_search = _channel_calibrate(master, architecture, fixture,
            frozen_prediction, prediction_document, fixed_scales, directory, epoch=qat_epoch)
        fixed_scales = dict(candidate.scales)
        metrics = evaluate_validation_pair(master, architecture, inputs, arm,
            quantized=candidate, ranking_weight=ranking_weight)
        key = _qat_validation_key(metrics, profile, **selection_arguments)
        history.append({
            "qat_epoch": qat_epoch,
            "schedule_epoch": schedule_epoch,
            "fixed_scales": applied_scales,
            "candidate_scales": _channel_scale_document(candidate),
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
        "schema": CHANNEL_QAT_EXECUTION_SCHEMA,
        "quantization_granularity": "per-output-channel",
        "float_warmup_evidence_sha256": sha256_bytes(canonical_json_bytes(float_result.report)),
        "training_fixture": fixture_document,
        "training_prediction_reference": prediction_document,
        "weight_projection": weight_projection,
        "original_initialization_checkpoint": _channel_artifact(original_checkpoint),
        "original_initialization_code_evidence": _quantized_update_evidence(
            quantize_channels(original_parameters, architecture, selected.scales), selected),
        "qat_profile": profile.name,
        "qat_profile_contract": qat_profile_contract(profile),
        "qat_epochs": qat_epochs,
        "learning_rate": profile.qat_learning_rate,
        "fixed_scale_qat": not profile.adapt_scales_after_each_epoch,
        "adaptive_scale_qat": profile.adapt_scales_after_each_epoch,
        "all_layer_fake_three_bit_qat": True,
        "selected_qat_epoch": selected_epoch,
        "selected_scales": _channel_scale_document(selected),
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
    if reference is not None:
        qat_report["retention_reference"] = reference.document()
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
    validate_qat_execution_evidence(qat_report, expected_profile=profile.name, **selection_arguments)
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
    calibration_directory: pathlib.Path | None = None,
    original_parameters: Mapping[str, np.ndarray] | None = None,
    _prediction_executor=None,
    warmup_consistency_expectation: Mapping[str, object] | None = None,
) -> QuantizedTrainingResult:
    prediction_arguments = {} if _prediction_executor is None else {"_prediction_executor": _prediction_executor}
    if qat_epochs != QAT_EPOCHS:
        raise TrainingError("compact deployment requires exactly four QAT epochs")
    profile = resolve_qat_profile(qat_profile)
    consistency = profile.name == WARMUP_CONSISTENCY_QAT_PROFILE
    if consistency and (inputs.new.split != "train" or inputs.anchor.split != "train" or arm.name != "search-target"):
        raise TrainingError("warmup consistency requires search-target TRAIN scalar sources")
    if not consistency and warmup_consistency_expectation is not None:
        raise TrainingError("consistency control/callback cannot enter another QAT profile")
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE and _prediction_executor is not None:
        raise TrainingError("parallel validation currently rejects channel QAT before work")
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE:
        if calibration_directory is None or original_parameters is None:
            raise TrainingError("channel QAT requires an audit directory and original float parameters")
        return _run_channel_prediction_qat(float_result, inputs, architecture, arm, seed, qat_epochs=qat_epochs,
            ranking_weight=ranking_weight, qat_profile=profile, calibration_directory=calibration_directory, original_parameters=original_parameters)
    if calibration_directory is not None or original_parameters is not None:
        raise TrainingError("legacy QAT cannot accept channel-only audit/initialization parameters")
    if profile.name != STANDARD_QAT_PROFILE and (
        architecture.name != "capacity-12x8"
        or inputs.successor_rankings is None
    ):
        raise TrainingError(
            "refined adaptive QAT requires successor-labeled capacity-12x8"
        )
    policy = STUDENT_RIVAL_PAIR_POLICY if profile.name in STUDENT_PAIR_PROFILES else None
    if policy is not None:
        _validate_student_float_evidence(float_result.report, seed=seed, ranking_weight=ranking_weight)
        if (float_result.metrics != float_result.report.get("validation") or any(
                sha256_bytes(value.tobytes()) != float_result.report.get("per_layer_update_evidence", {}).get(name, {}).get("after_sha256")
                for name, value in _validate_parameters(float_result.parameters, architecture).items())):
            raise TrainingError("student rival QAT warmup parameters/validation changed")
    elif "training_pair_policy" in float_result.report:
        raise TrainingError("student rival warmup cannot enter a legacy QAT profile")
    reference = _retention_reference(
        profile,
        float_result.metrics if profile.name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE) else None,
    )
    selection_arguments = {} if reference is None else {"float_validation_reference": reference}
    ranking_weight = _ranking_weight(ranking_weight)
    all_ranking_groups = (
        () if inputs.successor_rankings is None else inputs.successor_rankings.train
    )
    ranking_groups, density_report = _density_weighted_ranking_groups(
        all_ranking_groups
    )
    if ranking_weight > 0.0 and not ranking_groups:
        raise TrainingError("positive ranking loss has no QAT training groups")
    expectation = None
    teacher = _FrozenWarmupTeacher(float_result.parameters, architecture) if consistency else None
    if consistency and warmup_consistency_expectation is not None:
        expectation = _check_consistency_warmup(float_result, inputs, architecture, arm, seed, ranking_weight, warmup_consistency_expectation)
    pre_qat, scale_report = select_fixed_scales(
        float_result.parameters,
        architecture,
        inputs,
        arm,
        ranking_weight=ranking_weight,
        qat_profile=profile,
        **selection_arguments,
        **prediction_arguments,
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
        **prediction_arguments,
    )
    consistency_reference = pre_update_check = None
    if consistency:
        teacher.verify()
        training_identities = {name: dataset_identity(getattr(inputs, name)) for name in ("new", "anchor")}
        checkpoint_payload = deterministic_npz(teacher.parameters)
        consistency_reference = body_hashed({
            "schema": "papersoccer.compact-value-bfm-frozen-warmup-teacher.v1",
            "parameters": teacher.identity, "checkpoint_sha256": sha256_bytes(checkpoint_payload),
            "checkpoint_bytes": len(checkpoint_payload), "training_inputs": training_identities,
            "float_validation_sha256": sha256_bytes(canonical_json_bytes(float_result.metrics)),
            "float_report_sha256": sha256_bytes(canonical_json_bytes(float_result.report))})
        if expectation is not None:
            pre_update_check = _check_consistency_initial_grid(pre_qat, pre_qat_metrics, expectation)
        teacher.verify()
        if _parameter_identity(float_result.parameters, architecture) != teacher.identity:
            raise TrainingError("consistency warmup changed before optimizer construction")
    selected_metrics: dict[str, dict[str, float | int]] | None = None
    selected_key: tuple[float, ...] | None = None
    fixed_scales = dict(pre_qat.scales)
    master = {
        name: value.copy() for name, value in float_result.parameters.items()
    }
    optimizer = AdamW(
        master, learning_rate=profile.qat_learning_rate, weight_decay=WEIGHT_DECAY
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
        pair_evidence = None if policy is None else _StudentRivalEpochEvidence(
            phase="qat", seed=seed, schedule_epoch=schedule_epoch,
            ranking_weight=ranking_weight, scalar_batches=batch_count)
        consistency_evidence = None if teacher is None else _WarmupConsistencyEpochEvidence(
            teacher, seed=seed, qat_epoch=qat_epoch, input_identity=consistency_reference["training_inputs"],
            expectation_sha256=None if expectation is None else expectation["body_sha256"])
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
        training_objectives = [] if policy is not None else None
        for batch_index, (new_rows, anchor_rows) in enumerate(mixed_epoch_batches(
            len(inputs.new),
            len(inputs.anchor),
            seed=seed,
            epoch=schedule_epoch,
        )):
            if pair_evidence is not None:
                pair_evidence.begin_batch(batch_index, master, architecture)
            objective = _train_mixed_batch(
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
                **({"training_pair_policy": policy, "pair_evidence": pair_evidence} if policy is not None else {}),
                **({"warmup_teacher": teacher, "consistency_evidence": consistency_evidence} if consistency else {}),
            )
            if training_objectives is not None:
                training_objectives.append(objective)
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
                **selection_arguments,
                **prediction_arguments,
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
                **prediction_arguments,
            )
        key = _qat_validation_key(metrics, profile, **selection_arguments)
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
            **({"training_total_objective": float(np.mean(training_objectives)),
                "training_total_objective_definition": (CONSISTENCY_TRAINING_OBJECTIVE if consistency else STUDENT_TRAINING_OBJECTIVE)}
               if training_objectives is not None else {}),
            **({"training_pair_selection": pair_evidence.finish()} if pair_evidence is not None else {}),
            **({"warmup_consistency": consistency_evidence.finish()} if consistency_evidence is not None else {}),
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
        "learning_rate": profile.qat_learning_rate,
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
    if reference is not None:
        qat_report["retention_reference"] = reference.document()
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
    if policy is not None:
        qat_report["training_pair_policy"] = training_pair_policy_contract(profile)
        qat_report["training_pair_seed"] = seed
        qat_report["float_warmup_report_sha256"] = sha256_bytes(canonical_json_bytes(float_result.report))
    if consistency:
        teacher.verify()
        qat_report["warmup_consistency"] = {"policy": warmup_consistency_contract(),
            "reference": consistency_reference, "expectation": expectation,
            "pre_update_control_check": pre_update_check, "reference_after": teacher.identity}
    validate_qat_execution_evidence(qat_report, expected_profile=profile.name, **selection_arguments)
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


def _validate_channel_fixture(document):
    if not isinstance(document, Mapping) or set(document) != {"identity", "artifact"}:
        raise TrainingError("channel fixture evidence is absent")
    identity = document["identity"]
    verify_body_hash(identity, schema="papersoccer.compact-value-bfm-channel-fixture.v1", label="channel fixture")
    if identity.get("policy") != CHANNEL_FIXTURE_POLICY or set(identity.get("datasets", {})) != {"new", "anchor"}:
        raise TrainingError("channel fixture training identity/policy changed")
    indices = _channel_sample_indices(identity["datasets"]["new"]["samples"], identity["datasets"]["anchor"]["samples"])
    if identity.get("sampled_indices") != {name: values.tolist() for name, values in indices.items()}:
        raise TrainingError("channel fixture PCG64 sample changed")
    with np.load(_channel_read_artifact(document["artifact"], ".channel-fixture.npz"), allow_pickle=False) as archive:
        if set(archive.files) != {"indptr", "indices", "weights", "new_indices", "anchor_indices"}:
            raise TrainingError("channel fixture contains unexpected arrays or labels")
        arrays = {name: archive[name].copy() for name in archive.files}
    if (arrays["indptr"].dtype != np.dtype("<i8") or arrays["indptr"].shape != (4097,)
            or arrays["indices"].dtype != np.dtype("<u2") or arrays["indices"].ndim != 1
            or arrays["indptr"][0] != 0 or arrays["indptr"][-1] != len(arrays["indices"])
            or np.any(np.diff(arrays["indptr"]) < 0) or arrays["weights"].dtype != np.dtype("<f4")
            or arrays["weights"].shape != (4096,) or not np.all(np.isfinite(arrays["weights"])) or np.any(arrays["weights"] <= 0)
            or identity.get("arrays") != {name: _array_identity(arrays[name]) for name in ("indptr", "indices", "weights")}):
        raise TrainingError("channel fixture array evidence changed")
    for name in ("new", "anchor"):
        if arrays[name + "_indices"].dtype != np.dtype("<i8") or not np.array_equal(arrays[name + "_indices"], indices[name]):
            raise TrainingError("channel fixture sampled index bytes changed")
    return arrays["weights"]


def _validate_channel_calibration(stage, reference_document, reference, weights, *, epoch):
    if (not isinstance(stage, Mapping) or stage.get("schema") != "papersoccer.compact-value-bfm-channel-calibration.v1"
            or type(stage.get("qat_epoch")) is not int or stage["qat_epoch"] != epoch
            or stage.get("policy") != CHANNEL_CALIBRATION_POLICY or stage.get("training_reference") != reference_document
            or stage.get("parameters_unchanged_by_calibration") is not True or stage.get("heldout_read_by_calibration") is not False):
        raise TrainingError("channel scale stage/reference/policy changed")
    architecture = ARCHITECTURES["capacity-12x8"]
    parameters = load_float_checkpoint(_channel_read_artifact(stage["master_checkpoint"], ".float.npz"), architecture)
    if stage.get("master_parameters") != _parameter_identity(parameters, architecture):
        raise TrainingError("channel current-master identity changed")
    current = _normalize_channel_scales(stage.get("starting_scales"), canonical=True)
    starting = quantize_channels(parameters, architecture, current)
    if stage.get("starting_code_sha256") != {name: sha256_bytes(value.tobytes()) for name, value in starting.integer.items()}:
        raise TrainingError("channel incumbent codes do not use current masters")
    matrix = np.load(_channel_read_artifact(stage["prediction_matrix"], ".channel-predictions.npy"), mmap_mode="r", allow_pickle=False)
    count = stage.get("prediction_rows"); history = stage.get("trials")
    order = [(sweep, name, channel) for sweep in (1, 2) for name, n in CHANNEL_SCALE_COUNTS.items() for channel in range(n)]
    if (matrix.dtype != np.dtype("<f4") or matrix.shape != (672, 4096) or type(count) is not int or not 42 <= count <= 672
            or not np.all(np.isfinite(matrix[:count])) or np.any(matrix[count:] != 0)
            or not isinstance(history, list) or len(history) != 42
            or any(not isinstance(row, Mapping) or type(row.get("sweep")) is not int or type(row.get("channel")) is not int for row in history)
            or [(row["sweep"], row["layer"], row["channel"]) for row in history] != order):
        raise TrainingError("channel prediction matrix/two-sweep evidence changed")
    offset = 0; last_score = None; last_row = None
    for event in history:
        name, channel = event["layer"], event["channel"]
        values = parameters[name] if name == "w3" else parameters[name][:, channel]
        candidates = _channel_coordinate_candidates(values, current[name][channel]); trials = event.get("trials")
        if (not isinstance(trials, list) or len(trials) != len(candidates)
                or event.get("incumbent_scale") != float(current[name][channel])
                or any(type(row.get("ordinal")) is not int or type(row.get("prediction_row")) is not int or isinstance(row.get("scale"), bool) for row in trials)
                or [row["ordinal"] for row in trials] != list(range(len(candidates)))
                or [row["scale"] for row in trials] != list(map(float, candidates))
                or [row["prediction_row"] for row in trials] != list(range(offset, offset + len(candidates)))):
            raise TrainingError("channel incumbent-first candidate/prediction order changed")
        for trial, candidate in zip(trials, candidates, strict=True):
            state = _channel_scale_document(current); state[name][channel] = float(candidate)
            if trial.get("scale_state_sha256") != sha256_bytes(canonical_json_bytes(state)):
                raise TrainingError("channel prediction lost its full scale-state binding")
            score = _channel_prediction_objective(matrix[trial["prediction_row"]], reference, weights)
            if isinstance(trial.get("objective"), bool) or trial.get("objective") != score:
                raise TrainingError("channel candidate objective differs from saved predictions")
        if last_score is not None and (trials[0]["objective"] != last_score or matrix[offset].tobytes() != matrix[last_row].tobytes()):
            raise TrainingError("channel incumbent prediction continuity changed")
        selected = min(trials, key=lambda row: (row["objective"], row["ordinal"]))
        if (type(event.get("selected_ordinal")) is not int or event["selected_ordinal"] != selected["ordinal"]
                or event.get("selected_scale") != selected["scale"] or event.get("selected_objective") != selected["objective"]):
            raise TrainingError("channel exact first-incumbent choice changed")
        current[name][channel] = np.float32(selected["scale"]); last_score = selected["objective"]
        last_row = selected["prediction_row"]; offset += len(trials)
    selected = quantize_channels(parameters, architecture, current)
    if (offset != count or stage.get("selected_scales") != _channel_scale_document(selected)
            or stage.get("selected_code_sha256") != {name: sha256_bytes(value.tobytes()) for name, value in selected.integer.items()}
            or stage.get("initial_objective") != history[0]["trials"][0]["objective"] or stage.get("selected_objective") != last_score):
        raise TrainingError("channel scale/code/objective trajectory changed")
    return parameters, selected


def _validate_channel_qat_execution(value, *, float_validation_reference):
    profile = resolve_qat_profile(CHANNEL_PREDICTION_QAT_PROFILE)
    validate_qat_profile_contract(value.get("qat_profile_contract"), expected_name=profile.name)
    reference = _retention_reference(profile, float_validation_reference)
    if (value.get("schema") != CHANNEL_QAT_EXECUTION_SCHEMA or value.get("qat_profile") != profile.name
            or value.get("quantization_granularity") != "per-output-channel" or not valid_sha256(value.get("float_warmup_evidence_sha256"))
            or type(value.get("qat_epochs")) is not int or value.get("qat_epochs") != 4
            or value.get("learning_rate") != .0000625 or value.get("fixed_scale_qat") is not False or value.get("adaptive_scale_qat") is not True
            or value.get("all_layer_fake_three_bit_qat") is not True or value.get("pre_qat_retained") is not False
            or value.get("tie_break") != "prefer-earlier-qat-epoch-on-exact-tie"
            or value.get("executed_qat_epochs") != [1, 2, 3, 4] or any(type(epoch) is not int for epoch in value["executed_qat_epochs"])
            or value.get("retention_reference") != reference.document()):
        raise TrainingError("channel QAT mandatory schedule/profile/retention reference changed")
    fixture = value.get("training_fixture"); weights = _validate_channel_fixture(fixture)
    prediction_document = value.get("training_prediction_reference")
    verify_body_hash(prediction_document, schema="papersoccer.compact-value-bfm-channel-prediction-reference.v1", label="channel prediction reference")
    if (prediction_document.get("fixture_identity_sha256") != fixture["identity"]["body_sha256"]
            or prediction_document.get("fixture_array_sha256") != fixture["identity"]["arrays"]
            or prediction_document.get("frozen_across_initial_and_four_epoch_calibrations") is not True):
        raise TrainingError("channel target reference lost its fixed fixture")
    prediction = np.load(_channel_read_artifact(prediction_document["prediction"], ".channel-reference.npy"), mmap_mode="r", allow_pickle=False)
    if prediction.dtype != np.dtype("<f4") or prediction.shape != (4096,) or _array_identity(prediction) != prediction_document.get("prediction_array_sha256"):
        raise TrainingError("channel frozen prediction target bytes changed")
    warmup, initial_quantized = _validate_channel_calibration(value["scale_search"], prediction_document, prediction, weights, epoch=0)
    if prediction_document.get("pre_qat_parameters") != _parameter_identity(warmup, ARCHITECTURES["capacity-12x8"]):
        raise TrainingError("channel targets no longer bind the pre-QAT float master")
    projected, projection_report = _channel_weight_projection(warmup, ARCHITECTURES["capacity-12x8"])
    if value.get("weight_projection") != projection_report or value["scale_search"]["starting_scales"] != _channel_scale_document(projected):
        raise TrainingError("channel QAT initial weight projection changed")
    history = value.get("history"); trajectory = value.get("applied_scale_trajectory")
    if not isinstance(history, list) or len(history) != 4 or not isinstance(trajectory, list) or len(trajectory) != 4:
        raise TrainingError("channel QAT must contain four complete trained epochs")
    expected_batches = math.ceil(fixture["identity"]["datasets"]["new"]["samples"] / NEW_ROWS_PER_BATCH)
    last_parameters = warmup; scales = _channel_scale_document(initial_quantized); states = []; step = 0
    for epoch, (row, applied) in enumerate(zip(history, trajectory, strict=True), start=1):
        parameters, quantized = _validate_channel_calibration(row["adaptive_scale_search"], prediction_document, prediction, weights, epoch=epoch)
        candidate_scales = _channel_scale_document(quantized); fake = row.get("fake_quantization", {}); step += expected_batches
        if (type(row.get("qat_epoch")) is not int or row["qat_epoch"] != epoch or row.get("schedule_epoch") != 1 + epoch
                or row.get("fixed_scales") != scales or row.get("candidate_scales") != candidate_scales
                or row["adaptive_scale_search"]["starting_scales"] != scales
                or fake.get("bits") != 3 or fake.get("layers") != ["w1", "w2", "w3"] or fake.get("all_layers_trainable") is not True
                or type(fake.get("batches")) is not int or type(fake.get("optimizer_steps_after_epoch")) is not int
                or type(row.get("schedule_epoch")) is not int or type(applied.get("qat_epoch")) is not int
                or fake.get("batches") != expected_batches or fake.get("optimizer_steps_after_epoch") != step
                or fake.get("scales_applied_to_every_batch") != scales
                or fake.get("master_parameter_updates") != _parameter_update_evidence(last_parameters, parameters)
                or applied != {"qat_epoch": epoch, "training_scales": scales, "candidate_scales": candidate_scales, "adapted_after_epoch": True}
                or not isinstance(row.get("validation"), Mapping) or not _finite_metric_report(row["validation"])):
            raise TrainingError("channel epoch optimizer/master/scale evidence is discontinuous")
        states.append(quantized); last_parameters = parameters; scales = candidate_scales
    selected_epoch = value.get("selected_qat_epoch")
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= 4:
        raise TrainingError("channel QAT selected an untrained epoch")
    selected_index = min(range(4), key=lambda index: (*_qat_validation_key(history[index]["validation"], profile, float_validation_reference=reference), index))
    selected = states[selected_index]
    original = load_float_checkpoint(_channel_read_artifact(value["original_initialization_checkpoint"], ".float.npz"), ARCHITECTURES["capacity-12x8"])
    original_quantized = quantize_channels(original, ARCHITECTURES["capacity-12x8"], selected.scales)
    if (selected_epoch != selected_index + 1 or value.get("selected_scales") != _channel_scale_document(selected)
            or value.get("selected_validation") != history[selected_index]["validation"] or type(value.get("optimizer_steps")) is not int or value.get("optimizer_steps") != step
            or value.get("selected_per_layer_qat_evidence") != _quantized_update_evidence(initial_quantized, selected)
            or value.get("per_layer_update_evidence") != _quantized_update_evidence(initial_quantized, selected)
            or value.get("original_initialization_code_evidence") != _quantized_update_evidence(original_quantized, selected)
            or value.get("final_master_per_layer_update_evidence") != _parameter_update_evidence(warmup, last_parameters)):
        raise TrainingError("channel QAT selection/code/update evidence changed")
    return dict(value)


def _validate_student_epoch_evidence(value, *, phase, seed, epoch, weight, batches, groups):
    if not isinstance(value, Mapping):
        raise TrainingError("student rival epoch evidence is absent")
    verify_body_hash(value, schema="papersoccer.compact-value-bfm-student-rival-epoch.v1", label="student rival epoch")
    count_fields = ("group_evaluations", "selected_pairs", "changed_static_subset_groups",
        "regrettable_argmax_groups", "regrettable_argmax_included_groups", "missed_regrettable_argmax_groups")
    expected_fields = {"schema", "policy", "phase", "seed", "schedule_epoch", "loss_weight", "active",
        "scalar_batches", "executed_scalar_batches", *count_fields, "batch_master_sha256",
        "ordered_pair_prediction_sha256", "raw_prediction_trace_retained", "body_sha256"}
    active = weight > 0.0
    if (set(value) != expected_fields or value["policy"] != STUDENT_RIVAL_PAIR_POLICY
            or value["phase"] != phase or type(value["seed"]) is not int or value["seed"] != seed
            or type(value["schedule_epoch"]) is not int or value["schedule_epoch"] != epoch
            or type(value["loss_weight"]) is not float or value["loss_weight"] != weight
            or value["active"] is not active or type(value["scalar_batches"]) is not int or value["scalar_batches"] != batches
            or type(value["executed_scalar_batches"]) is not int or value["executed_scalar_batches"] != batches
            or any(type(value[k]) is not int or value[k] < 0 for k in count_fields)
            or value["group_evaluations"] != (groups if active else 0)
            or value["missed_regrettable_argmax_groups"] != 0
            or value["regrettable_argmax_included_groups"] != value["regrettable_argmax_groups"]
            or value["regrettable_argmax_groups"] > value["group_evaluations"]
            or value["changed_static_subset_groups"] > value["group_evaluations"]
            or not value["group_evaluations"] <= value["selected_pairs"] <= RANKING_PAIR_CAP * value["group_evaluations"]
            or (active and value["group_evaluations"] <= 0)
            or value["raw_prediction_trace_retained"] is not False
            or not valid_sha256(value["batch_master_sha256"])
            or value["batch_master_sha256"] == hashlib.sha256().hexdigest()
            or not valid_sha256(value["ordered_pair_prediction_sha256"])
            or ((value["ordered_pair_prediction_sha256"] == hashlib.sha256().hexdigest()) is active)):
        raise TrainingError("student rival epoch counts/policy/identity changed")
    return dict(value)


def _validate_student_training_objective(epoch, *, definition=STUDENT_TRAINING_OBJECTIVE):
    if ("training_objective_weighted_huber" in epoch
            or type(epoch.get("training_total_objective")) is not float
            or not math.isfinite(epoch["training_total_objective"])
            or epoch.get("training_total_objective_definition") != definition):
        raise TrainingError("student rival total training objective label/value changed")


def _validate_student_float_evidence(value, *, seed, ranking_weight):
    policy = training_pair_policy_contract(STUDENT_RIVALS_RETENTION_PROFILE)
    optimizer = {"name": "adamw", "batch_size": BATCH_SIZE, "maximum_epochs": 1, "patience": 1,
        "learning_rate": RANKING_FLOAT_LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_norm_clip": GRADIENT_CLIP}
    batching = {"new_rows_per_batch": NEW_ROWS_PER_BATCH, "anchor_rows_per_batch": ANCHOR_ROWS_PER_BATCH,
        "new_loss_share": .25, "anchor_loss_share": .75, "sources_normalized_separately": True,
        "anchor_stream": "continuous-no-repeat-until-permutation-complete"}
    if (not isinstance(value, Mapping) or value.get("training_pair_policy") != policy
            or type(seed) is not int or seed not in FIXED_SEEDS or type(value.get("seed")) is not int or value["seed"] != seed
            or type(value.get("best_float_epoch")) is not int or value["best_float_epoch"] != 1
            or not isinstance(value.get("history"), list) or len(value["history"]) != 1
            or value.get("optimizer") != optimizer or value.get("batching") != batching
            or any(type(value["optimizer"][key]) is not int for key in ("batch_size", "maximum_epochs", "patience"))
            or value.get("initialization", {}).get("kind") != "frozen-float-checkpoint"
            or value["initialization"].get("seed_affects") != "row-order-only"):
        raise TrainingError("student rivals require a matching new-policy warmup")
    successor = value.get("successor_ranking", {})
    weight = _ranking_weight(ranking_weight)
    if type(successor.get("loss_weight")) is not float or successor["loss_weight"] != weight or successor.get("loss_active") is not (weight > 0.0):
        raise TrainingError("student rival warmup loss binding changed")
    item = value["history"][0]
    _validate_student_training_objective(item)
    rows = item.get("coverage", {}).get("new", {}).get("rows_per_epoch")
    groups = successor.get("weighted_group_entries_per_epoch")
    if type(rows) is not int or rows <= 0 or rows % NEW_ROWS_PER_BATCH or type(groups) is not int or groups < 0:
        raise TrainingError("student rival warmup schedule is absent")
    return _validate_student_epoch_evidence(item.get("training_pair_selection"), phase="float-warmup",
        seed=seed, epoch=1, weight=weight, batches=rows // NEW_ROWS_PER_BATCH, groups=groups)


def _validate_student_qat_evidence(value):
    policy = training_pair_policy_contract(STUDENT_RIVALS_RETENTION_PROFILE)
    if (value.get("training_pair_policy") != policy or value.get("qat_profile") not in STUDENT_PAIR_PROFILES
            or type(value.get("training_pair_seed")) is not int or value["training_pair_seed"] not in FIXED_SEEDS
            or not valid_sha256(value.get("float_warmup_report_sha256"))
            or not isinstance(value.get("history"), list) or len(value["history"]) != QAT_EPOCHS):
        raise TrainingError("student rival QAT/warmup policy binding changed")
    successor = value.get("successor_ranking", {})
    weight = _ranking_weight(successor.get("loss_weight"))
    groups = successor.get("weighted_group_entries_per_epoch")
    if type(groups) is not int or groups < 0 or successor.get("loss_active") is not (weight > 0.0):
        raise TrainingError("student rival QAT schedule is absent")
    reports = []
    for epoch, item in enumerate(value["history"], start=2):
        _validate_student_training_objective(item, definition=(CONSISTENCY_TRAINING_OBJECTIVE
            if value["qat_profile"] == WARMUP_CONSISTENCY_QAT_PROFILE else STUDENT_TRAINING_OBJECTIVE))
        batches = item.get("fake_quantization", {}).get("batches")
        if type(batches) is not int or batches <= 0:
            raise TrainingError("student rival QAT batch count is invalid")
        reports.append(_validate_student_epoch_evidence(item.get("training_pair_selection"),
            phase="qat", seed=value["training_pair_seed"], epoch=epoch, weight=weight, batches=batches, groups=groups))
    return reports


def validate_student_rival_execution(float_training, quantized_training, *, seed, expected_binding=None):
    """Cross-bind new-only warmup and QAT evidence; stream hashes are not replay proofs."""
    qat = _validate_student_qat_evidence(quantized_training)
    weight = _ranking_weight(quantized_training["successor_ranking"]["loss_weight"])
    warmup = _validate_student_float_evidence(float_training, seed=seed, ranking_weight=weight)
    if (quantized_training["training_pair_seed"] != seed
            or quantized_training["float_warmup_report_sha256"] != sha256_bytes(canonical_json_bytes(float_training))):
        raise TrainingError("student rival QAT transplanted another warmup report")
    if expected_binding is not None:
        settings = expected_binding.get("settings", {})
        successor = expected_binding.get("successor_ranking", {})
        policy = training_pair_policy_contract(STUDENT_RIVALS_RETENTION_PROFILE)
        if (expected_binding.get("seed") != seed or settings.get("qat_profile") != quantized_training["qat_profile"]
                or settings.get("qat_profile_contract") != qat_profile_contract(quantized_training["qat_profile"])
                or settings.get("training_pair_policy") != policy or successor.get("training_pair_policy") != policy
                or successor.get("loss_weight") != weight
                or float_training.get("initialization", {}).get("parameters") != successor.get("initial_checkpoint", {}).get("parameters")
                or successor.get("float_warmup", {}).get("epochs") != 1
                or successor.get("float_warmup", {}).get("learning_rate") != RANKING_FLOAT_LEARNING_RATE):
            raise TrainingError("student rival outer warmup/policy/initialization binding changed")
    epochs = [warmup, *qat]
    return body_hashed({"schema": "papersoccer.compact-value-bfm-student-rival-execution.v1",
        "training_pair_policy": training_pair_policy_contract(STUDENT_RIVALS_RETENTION_PROFILE),
        "seed": seed, "loss_weight": weight, "validated_epoch_reports": len(epochs),
        "group_evaluations": sum(v["group_evaluations"] for v in epochs),
        "selected_pairs": sum(v["selected_pairs"] for v in epochs),
        "changed_static_subset_groups": sum(v["changed_static_subset_groups"] for v in epochs),
        "regrettable_argmax_groups": sum(v["regrettable_argmax_groups"] for v in epochs),
        "missed_regrettable_argmax_groups": 0,
        "epoch_evidence_body_sha256": [v["body_sha256"] for v in epochs],
        "raw_prediction_trace_retained": False,
        "independent_numerical_replay_claimed": False})


def _validate_warmup_consistency_qat(value):
    evidence = value.get("warmup_consistency")
    if (not isinstance(evidence, Mapping) or set(evidence) != {
            "policy", "reference", "expectation", "pre_update_control_check", "reference_after"}
            or evidence["policy"] != warmup_consistency_contract()):
        raise TrainingError("warmup consistency QAT evidence is absent or relabeled")
    _validate_warmup_consistency_policy(evidence["policy"])
    reference = evidence["reference"]
    verify_body_hash(reference, schema="papersoccer.compact-value-bfm-frozen-warmup-teacher.v1", label="frozen warmup teacher")
    if (set(reference) != {"schema", "parameters", "checkpoint_sha256", "checkpoint_bytes", "training_inputs",
            "float_validation_sha256", "float_report_sha256", "body_sha256"}
            or evidence["reference_after"] != reference["parameters"]
            or reference["float_report_sha256"] != value.get("float_warmup_report_sha256")
            or reference["float_validation_sha256"] != value.get("retention_reference", {}).get("float_validation_sha256")
            or type(reference["checkpoint_bytes"]) is not int or reference["checkpoint_bytes"] <= 0
            or not valid_sha256(reference["checkpoint_sha256"])
            or set(reference["training_inputs"]) != {"new", "anchor"}):
        raise TrainingError("warmup consistency frozen reference identity changed")
    architecture = ARCHITECTURES["capacity-12x8"]
    params = reference["parameters"]
    if (params.get("architecture") != architecture.name or params.get("dimensions") != list(architecture.dimensions)
            or set(params.get("layers", {})) != {"w1", "w2", "w3"}):
        raise TrainingError("warmup consistency reference architecture changed")
    for name, shape in architecture.shapes.items():
        layer = params["layers"][name]
        if (layer.get("shape") != list(shape) or layer.get("dtype") != "little-endian-float32"
                or not valid_sha256(layer.get("sha256"))
                or value.get("final_master_per_layer_update_evidence", {}).get(name, {}).get("before_sha256") != layer["sha256"]):
            raise TrainingError("warmup consistency reference differs from original QAT masters")
    expected = evidence["expectation"]
    if expected is not None:
        expected = _validate_consistency_expectation(expected)
        warmup = expected["warmup"]
        check = evidence["pre_update_control_check"]
        verify_body_hash(check, schema="papersoccer.compact-value-bfm-warmup-consistency-pre-update.v1", label="consistency pre-update check")
        if (set(check) != {"schema", "expectation_sha256", "warmup", "initial_quantization", "optimizer_steps", "checked_before_optimizer_construction", "body_sha256"}
                or check["expectation_sha256"] != expected["body_sha256"] or check["warmup"] != warmup
                or check["initial_quantization"] != expected["initial_quantization"]
                or type(check["optimizer_steps"]) is not int or check["optimizer_steps"] != 0
                or check["checked_before_optimizer_construction"] is not True
                or reference["parameters"] != warmup["parameters"]
                or reference["checkpoint_sha256"] != warmup["checkpoint_sha256"]
                or reference["checkpoint_bytes"] != warmup["checkpoint_bytes"]
                or reference["float_validation_sha256"] != sha256_bytes(canonical_json_bytes(warmup["validation"]))
                or check["initial_quantization"]["scales"] != value["scale_search"]["selected_scales"]
                or check["initial_quantization"]["validation"] != value["pre_qat_validation"]
                or {name: expected["input_binding"]["datasets"][name] for name in ("new", "anchor")} != reference["training_inputs"]):
            raise TrainingError("consistency control-to-warmup/initial-grid proof changed")
    elif evidence["pre_update_control_check"] is not None:
        raise TrainingError("consistency pre-update control check has no bound expectation")
    reports = []
    fields = {"schema", "policy", "seed", "qat_epoch", "schedule_epoch", "expectation_sha256", "reference_before", "reference_after",
        "training_inputs", "batches", "new_rows", "anchor_rows", "teacher_forward_calls", "consistency_gradient_additions",
        "nonzero_derivative_batches", "nonzero_teacher_disagreement_rows", "consistency_output_gradient_l2_sum", "mean_loss_components",
        "ordered_batch_prediction_gradient_sha256", "reference_optimizer_steps", "heldout_reference_predictions", "raw_prediction_trace_retained",
        "independent_gradient_replay_claimed", "body_sha256"}
    weight = value["successor_ranking"]["loss_weight"]
    for index, item in enumerate(value["history"], start=1):
        report = item.get("warmup_consistency")
        verify_body_hash(report, schema="papersoccer.compact-value-bfm-warmup-consistency-epoch.v1", label="consistency epoch")
        batches = item["fake_quantization"]["batches"]
        _validate_warmup_consistency_policy(report["policy"])
        if (set(report) != fields or report["policy"] != warmup_consistency_contract()
                or type(report["seed"]) is not int or report["seed"] != value["training_pair_seed"]
                or type(report["qat_epoch"]) is not int or report["qat_epoch"] != index
                or type(report["schedule_epoch"]) is not int or report["schedule_epoch"] != index + 1
                or report["expectation_sha256"] != (None if expected is None else expected["body_sha256"])
                or report["reference_before"] != params or report["reference_after"] != params
                or report["training_inputs"] != reference["training_inputs"]
                or any(type(report[k]) is not int or report[k] != v for k, v in {
                    "batches": batches, "new_rows": batches * 64, "anchor_rows": batches * 192,
                    "teacher_forward_calls": batches, "consistency_gradient_additions": batches, "reference_optimizer_steps": 0}.items())
                or type(report["nonzero_derivative_batches"]) is not int or not 0 <= report["nonzero_derivative_batches"] <= batches
                or type(report["nonzero_teacher_disagreement_rows"]) is not int or not 0 <= report["nonzero_teacher_disagreement_rows"] <= batches * 256
                or not valid_sha256(report["ordered_batch_prediction_gradient_sha256"])
                or report["ordered_batch_prediction_gradient_sha256"] == hashlib.sha256().hexdigest()
                or any(report[k] is not False for k in ("heldout_reference_predictions", "raw_prediction_trace_retained", "independent_gradient_replay_claimed"))):
            raise TrainingError("consistency epoch phase/rows/reference/use evidence changed")
        norm = report["consistency_output_gradient_l2_sum"]
        components = report["mean_loss_components"]
        if (type(norm) is not float or not math.isfinite(norm) or norm < 0
                or (norm > 0) != (report["nonzero_derivative_batches"] > 0)
                or report["nonzero_derivative_batches"] > report["nonzero_teacher_disagreement_rows"]
                or not isinstance(components, Mapping) or set(components) != {"label_loss", "consistency_loss", "weighted_ranking_loss", "total_objective"}
                or any(type(v) is not float or not math.isfinite(v) or v < 0 for v in components.values())
                or (weight == 0.0 and components["weighted_ranking_loss"] != 0.0)
                or not math.isclose(components["total_objective"], math.fsum(components[k] for k in ("label_loss", "consistency_loss", "weighted_ranking_loss")), rel_tol=1e-12, abs_tol=1e-15)
                or item.get("training_total_objective") != components["total_objective"]
                or item.get("training_total_objective_definition") != CONSISTENCY_TRAINING_OBJECTIVE):
            raise TrainingError("consistency component losses/derivative use changed")
        reports.append(report)
    return reports


def summarize_warmup_consistency_execution(float_training, quantized_training, *, expected_binding=None):
    if quantized_training.get("qat_profile") != WARMUP_CONSISTENCY_QAT_PROFILE:
        raise TrainingError("warmup consistency summary requires its own QAT profile")
    reports = _validate_warmup_consistency_qat(quantized_training)
    evidence = quantized_training["warmup_consistency"]
    reference = evidence["reference"]
    seed = quantized_training["training_pair_seed"]
    validate_student_rival_execution(float_training, quantized_training, seed=seed, expected_binding=expected_binding)
    if (reference["float_report_sha256"] != sha256_bytes(canonical_json_bytes(float_training))
            or reference["float_validation_sha256"] != sha256_bytes(canonical_json_bytes(float_training["validation"]))
            or any(float_training.get("per_layer_update_evidence", {}).get(name, {}).get("after_sha256") != layer["sha256"]
                for name, layer in reference["parameters"]["layers"].items())):
        raise TrainingError("consistency teacher differs from its genuine warmup")
    if expected_binding is not None:
        settings = expected_binding["settings"]
        if (settings.get("warmup_consistency") != warmup_consistency_contract()
                or settings.get("warmup_consistency_expectation") != evidence["expectation"]
                or reference["training_inputs"] != {name: expected_binding["datasets"][name] for name in ("new", "anchor")}):
            raise TrainingError("consistency outer input/policy/control binding changed")
        if evidence["expectation"] is not None:
            _validate_consistency_expectation(evidence["expectation"], binding=expected_binding)
    return body_hashed({"schema": "papersoccer.compact-value-bfm-warmup-consistency-execution.v1",
        "qat_profile": WARMUP_CONSISTENCY_QAT_PROFILE, "seed": seed, "coefficient": 1.0,
        "reference": reference, "control_expectation_sha256": None if evidence["expectation"] is None else evidence["expectation"]["body_sha256"],
        "control_checked_before_optimizer": evidence["pre_update_control_check"] is not None,
        "validated_qat_epochs": QAT_EPOCHS, "teacher_forward_calls": sum(v["teacher_forward_calls"] for v in reports),
        "consistency_gradient_additions": sum(v["consistency_gradient_additions"] for v in reports),
        "nonzero_derivative_batches": sum(v["nonzero_derivative_batches"] for v in reports),
        "nonzero_teacher_disagreement_rows": sum(v["nonzero_teacher_disagreement_rows"] for v in reports),
        "epoch_evidence_body_sha256": [v["body_sha256"] for v in reports],
        "reference_optimizer_steps": 0, "raw_prediction_trace_retained": False,
        "independent_gradient_replay_claimed": False, "per_term_parameter_attribution_measured": False})


def _validate_consistency_receipt_artifacts(receipt, binding, warmup_parameters):
    report = receipt["quantized_training"]
    summarize_warmup_consistency_execution(receipt["float_training"], report, expected_binding=binding)
    reference = report["warmup_consistency"]["reference"]
    artifact = receipt["float_checkpoint"]
    if (reference["parameters"] != _parameter_identity(warmup_parameters, ARCHITECTURES[receipt["architecture"]])
            or reference["checkpoint_sha256"] != artifact["sha256"] or reference["checkpoint_bytes"] != artifact["bytes"]):
        raise TrainingError("consistency teacher checkpoint differs from owned warmup artifact")
    expectation = report["warmup_consistency"]["expectation"]
    if expectation is not None:
        _validate_consistency_expectation(expectation, binding=binding, reread_control=True)
        quantized = quantize_fixed(warmup_parameters, ARCHITECTURES[receipt["architecture"]], report["scale_search"]["selected_scales"])
        _check_consistency_initial_grid(quantized, report["pre_qat_validation"], expectation)


def validate_qat_execution_evidence(
    value: object, *, expected_profile: str,
    float_validation_reference: object = None,
) -> dict[str, object]:
    """Validate the exact four-epoch, all-layer fake-quantization evidence."""

    if not isinstance(value, Mapping):
        raise TrainingError("QAT execution evidence is absent")
    profile = resolve_qat_profile(expected_profile)
    if profile.name == CHANNEL_PREDICTION_QAT_PROFILE:
        return _validate_channel_qat_execution(value, float_validation_reference=float_validation_reference)
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
    reference = None
    if profile.name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE):
        document = value.get("retention_reference")
        if not isinstance(document, Mapping):
            raise TrainingError("QAT frozen retention reference is absent")
        reference = _retention_reference(profile, document.get("float_validation"))
        if document != reference.document() or scale_search.get("retention_reference") != document:
            raise TrainingError("QAT frozen retention reference changed")
        if float_validation_reference is not None and reference != _retention_reference(profile, float_validation_reference):
            raise TrainingError("QAT retention reference differs from frozen float validation")

    def retention_search(search, *, adaptive):
        if reference is None:
            return
        if search.get("retention_reference") != reference.document():
            raise TrainingError("QAT adaptive retention reference changed")
        if not isinstance(search.get("trials"), list) or not search["trials"] or any(
            not isinstance(trial, Mapping) for trial in search["trials"]
        ):
            raise TrainingError("QAT retention search trials are malformed")
        grouped = []
        for trial in search["trials"]:
            identity = (trial.get("stage", "coordinate"),
                trial.get("pass", trial.get("refinement_pass")), trial.get("layer"))
            if not grouped or grouped[-1][0] != identity:
                grouped.append((identity, []))
            grouped[-1][1].append(trial)
        expected = [("coordinate", index, name)
            for index in range(1, (profile.adaptive_coordinate_passes if adaptive else profile.coordinate_search_passes) + 1)
            for name in ("w1", "w2", "w3")]
        if not adaptive:
            expected += [("local-refinement", index, name)
                for index in range(1, profile.local_refinement_passes + 1)
                for name in ("w1", "w2", "w3")]
        if [group[0] for group in grouped] != expected:
            raise TrainingError("QAT retention search order changed")
        state = search["starting_scales"] if adaptive else {name: values[-1] for name, values in search["candidates"].items()}
        for (_stage, _pass, layer), trials in grouped:
            if any(any(trial["scales"][name] != state[name] for name in state if name != layer) for trial in trials):
                raise TrainingError("QAT retention coordinate trajectory changed")
            winner = min(trials, key=lambda trial: (
                *_qat_validation_key(trial["validation"], profile, float_validation_reference=reference),
                float(trial["requested_scale"])))
            state = winner["scales"]
        if state != search["selected_scales"]:
            raise TrainingError("QAT retention scale choice differs from its frozen objective")
        if winner["validation"] != search.get("selected_validation"):
            raise TrainingError("QAT retention selected metrics differ from the winning scale trial")

    retention_search(scale_search, adaptive=False)
    if reference is not None and scale_search.get("selected_validation") != value.get("pre_qat_validation"):
        raise TrainingError("QAT retention pre-QAT metrics differ from initial scale selection")
    selected_scales = scales(value.get("selected_scales"), "QAT selected")
    selected_epoch = value.get("selected_qat_epoch")
    final_updates = value.get("final_master_per_layer_update_evidence")
    selected_updates = value.get("selected_per_layer_qat_evidence")
    if (
        value.get("qat_profile") != profile.name
        or value.get("qat_epochs") != QAT_EPOCHS
        or value.get("learning_rate") != profile.qat_learning_rate
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
            retention_search(adaptive, adaptive=True)
            if reference is not None and adaptive.get("selected_validation") != epoch.get("validation"):
                raise TrainingError("QAT retention epoch metrics differ from adaptive scale selection")
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
    if reference is not None:
        winner = min(history, key=lambda epoch: _qat_validation_key(
            epoch["validation"], profile, float_validation_reference=reference))
        if winner["qat_epoch"] != selected_epoch:
            raise TrainingError("QAT retention epoch choice differs from its frozen objective")
    successor = value.get("successor_ranking")
    if isinstance(successor, Mapping) and value.get(
        "per_layer_update_evidence"
    ) != selected_updates:
        raise TrainingError("QAT successor selected-layer evidence changed")
    if profile.name in STUDENT_PAIR_PROFILES:
        _validate_student_qat_evidence(value)
    if profile.name == WARMUP_CONSISTENCY_QAT_PROFILE:
        _validate_warmup_consistency_qat(value)
    elif "warmup_consistency" in value or any("warmup_consistency" in item for item in history):
        raise TrainingError("consistency evidence cannot enter another QAT profile")
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
    pair_execution = None
    if quantized_training.get("qat_profile") in STUDENT_PAIR_PROFILES:
        pair_execution = validate_student_rival_execution(float_training, quantized_training, seed=seed)
    elif "training_pair_policy" in float_training or "training_pair_policy" in quantized_training:
        raise TrainingError("student rival warmup cannot be relabeled as a legacy recipe")
    return {
        **({"training_pair_execution": pair_execution} if pair_execution is not None else {}),
        **({"warmup_consistency_execution": summarize_warmup_consistency_execution(float_training, quantized_training)}
           if quantized_training.get("qat_profile") == WARMUP_CONSISTENCY_QAT_PROFILE else {}),
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
    *, validation_workers: int = 1, seed_concurrency: int = 1,
    warmup_consistency_expectation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    prediction_settings = validation_prediction_settings(validation_workers, seed_concurrency)
    if validation_workers > 1 and resolve_qat_profile(qat_profile).name == CHANNEL_PREDICTION_QAT_PROFILE:
        raise TrainingError("parallel validation currently rejects channel QAT before work")
    ranking_weight = _ranking_weight(ranking_weight)
    profile = resolve_qat_profile(qat_profile)
    if profile.name == WARMUP_CONSISTENCY_QAT_PROFILE and (inputs.new.split != "train" or inputs.anchor.split != "train" or arm.name != "search-target"):
        raise TrainingError("warmup consistency requires search-target TRAIN scalar sources")
    if warmup_consistency_expectation is not None and profile.name != WARMUP_CONSISTENCY_QAT_PROFILE:
        raise TrainingError("consistency audit control cannot enter another profile binding")
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
            "qat_learning_rate": profile.qat_learning_rate,
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
    if profile.name in STUDENT_PAIR_PROFILES:
        body["settings"]["training_pair_policy"] = training_pair_policy_contract(profile)
        body["successor_ranking"]["training_pair_policy"] = training_pair_policy_contract(profile)
    if profile.name == WARMUP_CONSISTENCY_QAT_PROFILE:
        body["settings"]["warmup_consistency"] = warmup_consistency_contract()
        if warmup_consistency_expectation is not None:
            body["settings"]["warmup_consistency_expectation"] = _validate_consistency_expectation(
                warmup_consistency_expectation, binding=body, reread_control=True)
    if prediction_settings is not None:
        from tools import compact_value_bfm_prediction_pool as prediction
        prediction_settings["features"] = {name: prediction._features(getattr(inputs, name), np)
            for name in ("common_adjudicator", "canonical_validation")}
        body["settings"]["validation_prediction"] = prediction_settings
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


def _channel_artifact_scope_preflight(report, output_directory):
    root = pathlib.Path(output_directory).resolve()
    if not isinstance(report, Mapping):
        raise TrainingError("channel QAT evidence is absent")
    def paths(value):
        if isinstance(value, Mapping):
            if set(value) == {"path", "sha256", "bytes"}:
                if not isinstance(value["path"], str):
                    raise TrainingError("channel calibration artifact path is not a string")
                path = pathlib.Path(value["path"])
                _reject_path_markers(path, "channel calibration artifact")
                if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
                    raise TrainingError("channel calibration artifact escaped its seed output")
            else:
                for item in value.values(): paths(item)
        elif isinstance(value, list):
            for item in value: paths(item)
    paths(report)


def _validate_channel_receipt_links(receipt, expected_binding, output_directory, warmup_parameters, quantized, runtime_selection, runtime_document_value):
    report = receipt["quantized_training"]
    _channel_artifact_scope_preflight(report, output_directory)
    fixture_identity = report["training_fixture"]["identity"]
    expected_datasets = {name: expected_binding["datasets"][name] for name in ("new", "anchor")}
    if (fixture_identity["datasets"] != expected_datasets
            or report["training_prediction_reference"]["pre_qat_parameters"] != _parameter_identity(warmup_parameters, ARCHITECTURES["capacity-12x8"])
            or runtime_document_value.get("schema") != CHANNEL_RUNTIME_SCHEMA
            or runtime_selection.get("qat_profile") != CHANNEL_PREDICTION_QAT_PROFILE
            or runtime_selection.get("qat_evidence_sha256") != sha256_bytes(canonical_json_bytes(report))
            or _channel_scale_document(quantized) != report["selected_scales"]
            or {name: sha256_bytes(value.tobytes()) for name, value in quantized.integer.items()}
                != report["history"][report["selected_qat_epoch"] - 1]["adaptive_scale_search"]["selected_code_sha256"]
            or runtime_selection.get("qat_epoch") != report["selected_qat_epoch"] or runtime_selection.get("float_epoch") != 1
            or runtime_selection.get("source_bundle_body_sha256") != expected_binding.get("source_bundle_body_sha256")):
        raise TrainingError("channel receipt/runtime transplanted another fixture, warmup or QAT evidence")
    original = load_float_checkpoint(_channel_read_artifact(report["original_initialization_checkpoint"], ".float.npz"), ARCHITECTURES["capacity-12x8"])
    initial = expected_binding.get("successor_ranking", {}).get("initial_checkpoint", {})
    if (_parameter_identity(original, ARCHITECTURES["capacity-12x8"]) != initial.get("parameters")
            or report.get("float_warmup_evidence_sha256") != sha256_bytes(canonical_json_bytes(receipt.get("float_training")))
            or receipt.get("float_training", {}).get("per_layer_update_evidence") != _parameter_update_evidence(original, warmup_parameters)
            or receipt.get("float_training", {}).get("optimizer", {}).get("learning_rate") != RANKING_FLOAT_LEARNING_RATE
            or receipt.get("float_training", {}).get("seed") != expected_binding.get("seed")
            or receipt.get("seed") != expected_binding.get("seed")
            or receipt.get("float_training", {}).get("validation") != receipt.get("float_validation")):
        raise TrainingError("channel receipt changed the outer frozen initialization/warmup evidence")
    if receipt.get("quantized_validation") != report["selected_validation"] or receipt.get("offline_gate") != offline_advancement_gate(receipt["float_validation"], receipt["quantized_validation"]):
        raise TrainingError("channel receipt metric/gate outcome differs from selected QAT evidence")


def _validate_student_receipt_artifacts(receipt, binding, warmup_parameters, quantized, selection, document):
    report = receipt["quantized_training"]
    float_report = receipt["float_training"]
    updates = float_report.get("per_layer_update_evidence", {})
    initial = binding["successor_ranking"]["initial_checkpoint"]["parameters"]
    if (document.get("schema") != RUNTIME_SCHEMA or selection.get("float_epoch") != 1
            or selection.get("qat_epoch") != report["selected_qat_epoch"]
            or selection.get("source_bundle_body_sha256") != binding.get("source_bundle_body_sha256")
            or float_report.get("validation") != receipt.get("float_validation")
            or receipt.get("quantized_validation") != report.get("selected_validation")
            or receipt.get("offline_gate") != offline_advancement_gate(receipt["float_validation"], receipt["quantized_validation"])):
        raise TrainingError("student rival runtime/metric/epoch binding changed")
    for name in ("w1", "w2", "w3"):
        if (updates.get(name, {}).get("before_sha256") != initial["layers"][name]["sha256"]
                or updates.get(name, {}).get("after_sha256") != sha256_bytes(warmup_parameters[name].tobytes())
                or quantized.scales[name] != report["selected_scales"][name]
                or sha256_bytes(quantized.integer[name].tobytes()) != report["selected_per_layer_qat_evidence"][name]["after_sha256"]):
            raise TrainingError("student rival warmup/runtime codes changed")
    if report.get("qat_profile") == WARMUP_CONSISTENCY_QAT_PROFILE:
        _validate_consistency_receipt_artifacts(receipt, binding, warmup_parameters)


def prediction_seed_execution_policy(seed_workers, settings):
    from tools import compact_value_bfm_prediction_pool as prediction
    pol = prediction.validate_policy(settings["policy"])
    if seed_workers not in (1, 2) or pol["seed_concurrency"] != seed_workers:
        raise TrainingError("prediction policy differs from actual thread seed roster")
    return {"schema": "papersoccer.compact-value-bfm-seed-execution.v2",
        "seed_workers": seed_workers, "maximum_seed_workers": 2,
        "worker_model": "shared-read-only-input-thread-pool" if seed_workers == 2 else "serial-seed-roster",
        "receipt_order": "fixed-seed-order", "selection_order": "validation-key-then-seed",
        "resume": "completed-reference-only;interrupted-parallel-seed-is-spent",
        "per_seed_execution_binding_includes_worker_count": True,
        "validation_prediction_policy": pol,
        "maximum_active_numerical_streams": pol["maximum_active_numerical_streams"]}


def validate_validation_prediction_receipt(receipt, expected_binding, artifact_root):
    settings = expected_binding.get("settings", {}).get("validation_prediction")
    evidence = receipt.get("validation_prediction_execution")
    native = receipt.get("native_thread_execution", {})
    if settings is None:
        if evidence is not None or native.get("schema") == PARALLEL_NATIVE_EXECUTION_SCHEMA:
            raise TrainingError("serial seed cannot claim unbound prediction helpers")
        return
    from tools import compact_value_bfm_prediction_pool as prediction
    prediction.validate_policy(settings["policy"])
    if (native.get("schema") != PARALLEL_NATIVE_EXECUTION_SCHEMA
            or native.get("prediction_policy") != settings["policy"]):
        raise TrainingError("prediction helper native envelope disagrees with seed binding")
    prediction.validate_execution(evidence, settings, artifact_root=artifact_root,
        dataset_identities=expected_binding["datasets"], trainer=sys.modules[__name__])


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
    validate_validation_prediction_receipt(receipt, expected_binding, output_directory)
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
        or settings.get("qat_learning_rate") != profile_contract["schedule"]["qat_learning_rate"]
        or (profile_name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE)
            and not isinstance(receipt.get("float_validation"), Mapping))
    ):
        raise TrainingError("compact seed receipt QAT profile changed")
    if profile_name == CHANNEL_PREDICTION_QAT_PROFILE:
        _channel_artifact_scope_preflight(receipt.get("quantized_training"), output_directory)
    validate_qat_execution_evidence(
        receipt.get("quantized_training"), expected_profile=profile_name,
        **({"float_validation_reference": receipt.get("float_validation")}
            if profile_name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE) else {}),
    )
    if profile_name in STUDENT_PAIR_PROFILES:
        validate_student_rival_execution(receipt["float_training"], receipt["quantized_training"],
            seed=receipt["seed"], expected_binding=expected_binding)
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
    warmup_parameters = load_float_checkpoint(checkpoint_path, ARCHITECTURES[architecture_name])
    runtime_path = _output_artifact(
        output_directory,
        runtime.get("path"),
        expected_sha256=runtime.get("sha256"),
        label="quantized runtime",
    )
    loaded_architecture, _quantized, selection, _document = load_runtime(runtime_path)
    if profile_name != CHANNEL_PREDICTION_QAT_PROFILE and _document.get("schema") != RUNTIME_SCHEMA:
        raise TrainingError("legacy seed receipt cannot bind a channel runtime")
    if profile_name == CHANNEL_PREDICTION_QAT_PROFILE:
        _validate_channel_receipt_links(receipt, expected_binding, output_directory, warmup_parameters, _quantized, selection, _document)
    if profile_name in STUDENT_PAIR_PROFILES:
        _validate_student_receipt_artifacts(receipt, expected_binding, warmup_parameters, _quantized, selection, _document)
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
    validation_workers: int = 1,
    _seed_concurrency: int = 1,
    warmup_consistency_expectation: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if warmup_consistency_expectation is not None and resolve_qat_profile(qat_profile).name != WARMUP_CONSISTENCY_QAT_PROFILE:
        raise TrainingError("consistency audit control cannot enter another seed profile")
    prediction_settings = validation_prediction_settings(validation_workers, _seed_concurrency)
    if validation_workers > 1 and resolve_qat_profile(qat_profile).name == CHANNEL_PREDICTION_QAT_PROFILE:
        raise TrainingError("parallel validation currently rejects channel QAT before any work")
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
                validation_workers=validation_workers, _seed_concurrency=_seed_concurrency,
                warmup_consistency_expectation=warmup_consistency_expectation,
            )
    native_execution = validate_native_thread_execution(
        _native_thread_execution
    )
    native_execution = prediction_native_execution(native_execution, prediction_settings)
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
        validation_workers=validation_workers, seed_concurrency=_seed_concurrency,
        warmup_consistency_expectation=warmup_consistency_expectation,
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
    if profile.name in STUDENT_PAIR_PROFILES:
        float_arguments["training_pair_policy"] = STUDENT_RIVAL_PAIR_POLICY
    prediction_evidence = None
    manager = contextlib.nullcontext(None)
    if prediction_settings is not None:
        from tools import compact_value_bfm_prediction_pool as prediction
        manager = prediction.ValidationPredictionPool(
            {"common_adjudicator": inputs.common_adjudicator, "canonical_validation": inputs.canonical_validation},
            output_directory / "validation-prediction" / f"seed-{seed}", validation_workers,
            seed_concurrency=_seed_concurrency, trainer=sys.modules[__name__])
    with manager as predictor:
        prediction_arguments = {} if predictor is None else {"_prediction_executor": predictor}
        float_result = train_float_seed(
            inputs, architecture, arm, seed, **float_arguments, **prediction_arguments
        )
        quantized_result = run_fixed_scale_qat(
            float_result,
            inputs,
            architecture,
            arm,
            seed,
            ranking_weight=ranking_weight,
            qat_profile=profile,
            **({"warmup_consistency_expectation": warmup_consistency_expectation}
               if profile.name == WARMUP_CONSISTENCY_QAT_PROFILE else {}),
            **({"calibration_directory": output_directory / "channel-calibration" / f"seed-{seed}",
                "original_parameters": float_arguments["initial_parameters"]}
               if profile.name == CHANNEL_PREDICTION_QAT_PROFILE else {}),
            **prediction_arguments,
        )
    if prediction_settings is not None:
        prediction_evidence = manager.evidence()
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
        **({"qat_profile": profile.name, "qat_evidence_sha256": sha256_bytes(canonical_json_bytes(quantized_result.report))}
           if profile.name == CHANNEL_PREDICTION_QAT_PROFILE else {}),
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
    if prediction_evidence is not None:
        body["validation_prediction_execution"] = prediction_evidence
        body["resume_policy"] = "completed-receipt-reused;interrupted-parallel-seed-is-spent"
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
    validation_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run independent seeds with shared read-only inputs and stable ordering."""

    validation_prediction_settings(validation_workers, seed_workers)
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
                validation_workers=validation_workers, _seed_concurrency=workers,
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
    validation_workers: int = 1,
) -> pathlib.Path:
    prediction_settings = validation_prediction_settings(validation_workers, seed_workers)
    if validation_workers > 1 and resolve_qat_profile(qat_profile).name == CHANNEL_PREDICTION_QAT_PROFILE:
        raise TrainingError("parallel validation currently rejects channel QAT before any work")
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
        validation_workers=validation_workers,
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
    if prediction_settings is not None:
        body["validation_prediction"] = chosen["binding"]["settings"]["validation_prediction"]
        body["seed_execution_policy"] = prediction_seed_execution_policy(seed_workers, body["validation_prediction"])
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
    if "validation_prediction" in selection:
        expected_fields.add("validation_prediction")
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
    if profile_name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE) and not isinstance(selection.get("float_validation"), Mapping):
        raise TrainingError("compact selection frozen float validation is absent")
    if profile_name == CHANNEL_PREDICTION_QAT_PROFILE:
        _channel_artifact_scope_preflight(selection.get("qat_execution_evidence"), artifact_root)
    validate_qat_execution_evidence(
        selection.get("qat_execution_evidence"),
        expected_profile=profile_name,
        **({"float_validation_reference": selection.get("float_validation")}
            if profile_name in (RETENTION_FIRST_LOW_RATE_QAT_PROFILE, CHANNEL_PREDICTION_QAT_PROFILE, STUDENT_RIVALS_RETENTION_PROFILE, WARMUP_CONSISTENCY_QAT_PROFILE) else {}),
    )
    seed_policy = selection.get("seed_execution_policy")
    seed_workers = seed_policy.get("seed_workers") if isinstance(
        seed_policy, Mapping
    ) else None
    prediction_settings = selection.get("validation_prediction")
    if prediction_settings is not None:
        if (seed_policy != prediction_seed_execution_policy(seed_workers, prediction_settings)
                or ("successor_ranking" in selection and seed_workers != 2)):
            raise TrainingError("parallel seed execution policy changed")
    else:
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
    validate_validation_prediction_receipt(receipt, receipt.get("binding", {}), artifact_root)
    if receipt.get("binding", {}).get("settings", {}).get("validation_prediction") != prediction_settings:
        raise TrainingError("selection prediction execution binding changed")
    if receipt.get("successor_ranking", {}).get("labels_present") is True:
        schedule_execution = validate_successor_schedule_execution(
            receipt.get("float_training"), receipt.get("quantized_training"),
            seed=receipt.get("seed"),
        )
        if receipt["successor_ranking"].get(
            "schedule_execution"
        ) != schedule_execution:
            raise TrainingError("selection seed successor schedule summary changed")
    if profile_name in STUDENT_PAIR_PROFILES:
        validate_student_rival_execution(receipt["float_training"], receipt["quantized_training"],
            seed=receipt["seed"], expected_binding=receipt["binding"])
        checkpoint = receipt["float_checkpoint"]
        warmup_path = _output_artifact(artifact_root, checkpoint["path"], expected_sha256=checkpoint["sha256"], label="student rival warmup")
        _validate_student_receipt_artifacts(receipt, receipt["binding"], load_float_checkpoint(warmup_path, architecture),
            _quantized, runtime_selection, _document)
    if profile_name != CHANNEL_PREDICTION_QAT_PROFILE and _document.get("schema") != RUNTIME_SCHEMA:
        raise TrainingError("legacy immutable selection cannot bind a channel runtime")
    if profile_name == CHANNEL_PREDICTION_QAT_PROFILE:
        checkpoint = receipt["float_checkpoint"]
        warmup_path = _output_artifact(artifact_root, checkpoint["path"], expected_sha256=checkpoint["sha256"], label="channel warmup")
        _validate_channel_receipt_links(receipt, receipt["binding"], artifact_root,
            load_float_checkpoint(warmup_path, architecture), _quantized, runtime_selection, _document)
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
    train.add_argument("--validation-workers", type=int, choices=(1, 2, 4), default=1)
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
                "scales": (_channel_scale_document(quantized) if isinstance(quantized, ChannelQuantizedWeights) else {
                    name: float(quantized.scales[name])
                    for name in ("w1", "w2", "w3")
                }),
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
                        validation_workers=arguments.validation_workers,
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
