#!/usr/bin/env python3
"""One-shot bounded W3 discrete repair after quantization-v2 rejection.

This development-only campaign freezes v2 W1/W2 and searches the complete
current±1 neighborhood of the eight signed-three-bit W3 codes across a fixed
positive float32 scale grid.  It uses only the already-open unprotected
validation pair.  Protected tests remain inaccessible until one immutable
offline-passing selection exists.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load(
    HERE / "compact_value_bfm_quantization_v2.py", "compact_discrete_v3_predecessor"
)
v1 = v2.v1
qualification = v2.qualification
compact = v2.compact
iteration = v2.iteration
selfsearch = v2.selfsearch
large_training = v2.large_training


V3Error = qualification.QualificationError
SuccessorError = V3Error
SuccessorError = V3Error
NAMESPACE = v2.NAMESPACE
SUCCESSOR_CAMPAIGN_ID = "compact-value-bfm-discrete-v3-20260901-v1"
V2_CAMPAIGN_ID = v2.SUCCESSOR_CAMPAIGN_ID
SOURCE_PLAN_SCHEMA = v2.SOURCE_PLAN_SCHEMA
TRAINING_INPUT_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-training-input.v1"
AUTHORIZATION_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-authorization.v1"
PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-plan.v1"
EXECUTION_CLAIM_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-claim.v1"
SEARCH_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-search.v1"
SELECTION_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-selection.v1"
SELECTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-selection-reference.v1"
)
OUTCOME_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-outcome.v1"

V2_PLAN_FILE = "78dca783486002fb738dad6e2442221d28a9b1ff68a08f58eab9b0ca1d56dc47"
V2_PLAN_BODY = "78891c08e6508a45f576514037e6c25f2df38e119c325ba4fe5cb257b6c16bc5"
V2_SELECTION_FILE = (
    "7a77304ba31e96d174e5f7893053519e48f3d0adf48ec97f658efefe5ef4d001"
)
V2_SELECTION_BODY = (
    "191063a5263158e89a556053f79806dc17ba5d4508c32a565ed87510734f3e8c"
)
V2_OUTCOME_FILE = "30a98c99927b7ee4b3e2d097d5fd4d112ff93a69824e80cdd1eee1a3c46c17a3"
V2_OUTCOME_BODY = "d06c1ba1e80f67e041836cc59596d4aa96928344c679439feaa507488e74810c"
V2_INDEX_FILE = "6865fb90d7f35b26a53f1af5f66053c4a75f40e7147a12844d40be7b8701946a"
V2_INDEX_BODY = "8d564154d63c6d2a062280e2960555c2309b88b076832e1f6a6042d62cef8c14"
V2_RUNTIME_FILE = "02cceff8e577497d52d2be6438bddac4a459eacad60aba63ce56ab1885e6b9db"
V2_RUNTIME_BODY = "233d3a1ea36bc43cf655483d57f97c4082f207d64cc9f1edc80246820421483c"

ARCHITECTURE = v2.ARCHITECTURE
SEED = v2.SEED
FLOAT_EPOCH = v2.FLOAT_EPOCH
QAT_EPOCH = 1
BASE_W3_CODES = (0, 3, -1, 1, -3, -2, 0, -1)
CODE_OPTIONS = tuple(
    tuple(range(max(-3, value - 1), min(3, value + 1) + 1))
    for value in BASE_W3_CODES
)
CODE_VECTOR_COUNT = math.prod(len(options) for options in CODE_OPTIONS)
SCALE_TICK_MIN = 100
SCALE_TICK_MAX = 2_400
SCALE_TICK_VALUE = 0.00005
SCALE_COUNT = SCALE_TICK_MAX - SCALE_TICK_MIN + 1
TOTAL_GRID_CANDIDATES = CODE_VECTOR_COUNT * SCALE_COUNT
EXPECTED_SIGN_FEASIBLE_CODES = 123
EXPECTED_COMMON_SURVIVORS = 7_636
EXPECTED_PASSING_CANDIDATES = 404
EXPECTED_WINNER_CODES = (0, 2, -1, 1, -2, -1, 0, 0)
EXPECTED_WINNER_SCALE = float(np.float32(0.0728))
BLAS_ENVIRONMENT = v2.BLAS_ENVIRONMENT
_RUN_LOCK_FD: int | None = None

_canonical_json = v2._canonical_json
_declared_record = v2._declared_record
retired_protected_paths = v2.retired_protected_paths


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_v2_root() -> pathlib.Path:
    return (
        REPOSITORY / "results" / "compact_value_bfm" / V2_CAMPAIGN_ID
    ).resolve()


def canonical_v1_root() -> pathlib.Path:
    return v2.canonical_v1_root()


def canonical_v3_root() -> pathlib.Path:
    return (
        REPOSITORY / "results" / "compact_value_bfm" / SUCCESSOR_CAMPAIGN_ID
    ).resolve()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V3Error(f"v3 artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_record(record: object, label: str) -> pathlib.Path:
    return v2._verify_record(record, label)


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    return v2._write_content_addressed(directory, body, suffix)


def _v2_paths(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "plan": root / "quantization-v2-plan.json",
        "selection": root / "selections" / (
            f"{V2_SELECTION_FILE}.quantization-v2-selection.json"
        ),
        "outcome": root / "governance" / "02-outcome.json",
        "candidate_index": root / "candidate-index.json",
        "runtime": root / "training" / "quantized-runtimes" / (
            f"{V2_RUNTIME_FILE}.runtime.json"
        ),
    }


def _validate_v2(root: pathlib.Path) -> dict[str, Any]:
    root = root.resolve()
    if root != canonical_v2_root():
        raise V3Error("v3 predecessor root is not canonical v2")
    paths = _v2_paths(root)
    expected = {
        "plan": V2_PLAN_FILE,
        "selection": V2_SELECTION_FILE,
        "outcome": V2_OUTCOME_FILE,
        "candidate_index": V2_INDEX_FILE,
        "runtime": V2_RUNTIME_FILE,
    }
    for name, digest in expected.items():
        if paths[name].is_symlink() or not paths[name].is_file():
            raise V3Error(f"v2 {name} is not a regular file")
        if _sha256_file(paths[name]) != digest:
            raise V3Error(f"v2 {name} changed")
    plan = v2.load_plan(paths["plan"], output_root=root)
    selection_path = v2._selection_reference(
        root / "selection-reference.json", plan=plan, output_root=root
    )
    if selection_path != paths["selection"]:
        raise V3Error("v2 selected path changed")
    selection = qualification.load_sealed(paths["selection"], v2.SELECTION_SCHEMA)
    outcome = qualification.load_sealed(paths["outcome"], v2.OUTCOME_SCHEMA)
    candidate_index = qualification.load_sealed(
        paths["candidate_index"], v2.CANDIDATE_INDEX_SCHEMA
    )
    architecture, quantized, runtime_selection, runtime = compact.load_runtime(
        paths["runtime"]
    )
    if (
        plan.get("body_sha256") != V2_PLAN_BODY
        or selection.get("body_sha256") != V2_SELECTION_BODY
        or outcome.get("body_sha256") != V2_OUTCOME_BODY
        or candidate_index.get("body_sha256") != V2_INDEX_BODY
        or runtime.get("body_sha256") != V2_RUNTIME_BODY
        or selection.get("offline_gate", {}).get("passed") is not False
        or selection.get("selection_immutable") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("old_protected_tests_accessed") is not False
        or outcome.get("status") != "quantization-v2-terminal-offline-rejection"
        or outcome.get("fresh_protected_tests_authorized") is not False
        or outcome.get("v3_authorized") is not False
        or architecture.name != ARCHITECTURE
        or runtime_selection.get("qat_epoch") != QAT_EPOCH
        or tuple(int(value) for value in quantized.integer["w3"]) != BASE_W3_CODES
    ):
        raise V3Error("v2 discrete-repair boundary changed")
    if (root / "fresh-holdout" / "materialization-receipt.json").exists():
        raise V3Error("v2 fresh protected holdout was materialized")
    return {
        "paths": paths,
        "plan": plan,
        "selection": selection,
        "outcome": outcome,
        "candidate_index": candidate_index,
        "architecture": architecture,
        "quantized": quantized,
    }


def _authorization_body(predecessor: Mapping[str, Any], authorized_at_utc: str) -> dict[str, Any]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "one-discrete-v3-authorized",
        "authorized_at_utc": authorized_at_utc,
        "authorization_basis": "user-explicitly-overrode-v2-no-v3-and-requested-finish",
        "v2_outcome": qualification.artifact_reference(
            predecessor["paths"]["outcome"], v2.OUTCOME_SCHEMA
        ),
        "v2_runtime": _record(predecessor["paths"]["runtime"]),
        "w1_changes_authorized": False,
        "w2_changes_authorized": False,
        "w3_local_code_and_scale_search_authorized": True,
        "new_training_games_authorized": 0,
        "new_training_labels_authorized": 0,
        "preplan_unprotected_development_diagnostics": {
            "coarse_grid_runs": 1,
            "dense_full_grid_runs": 4,
            "artifacts_persisted": False,
            "protected_tests_opened": False,
            "purpose": "algorithm-design-and-reproducibility-debugging",
        },
        "canonical_postplan_search_executions_authorized": 1,
        "fresh_protected_holdout_materializations_authorized_after_pass": 1,
        "fresh_evaluation_games_and_labels_authorized_only_after_pass": True,
        "v4_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


def _search_roster_body() -> dict[str, Any]:
    return {
        "base_w3_codes": list(BASE_W3_CODES),
        "code_options": [list(values) for values in CODE_OPTIONS],
        "code_order": "lexicographic-itertools-product",
        "code_vector_count": CODE_VECTOR_COUNT,
        "scale_tick_min": SCALE_TICK_MIN,
        "scale_tick_max": SCALE_TICK_MAX,
        "scale_tick_value": SCALE_TICK_VALUE,
        "scale_count": SCALE_COUNT,
        "scale_float_contract": "float32(integer-tick*0.00005)",
        "total_grid_candidates": TOTAL_GRID_CANDIDATES,
        "positive_scale_sign_pruning": True,
        "common_huber_pruning_before_canonical": True,
        "expected_sign_feasible_codes": EXPECTED_SIGN_FEASIBLE_CODES,
        "expected_common_survivors": EXPECTED_COMMON_SURVIVORS,
        "expected_passing_candidates": EXPECTED_PASSING_CANDIDATES,
        "selection_key": "unchanged-gate-feasibility-key-plus-grid-ordinal",
        "preplan_unprotected_development_diagnostics": {
            "coarse_grid_runs": 1,
            "dense_full_grid_runs": 4,
            "artifacts_persisted": False,
            "protected_tests_opened": False,
        },
        "w1_frozen": True,
        "w2_frozen": True,
    }


def prepare(
    *, v2_root: pathlib.Path, output_root: pathlib.Path,
    holdout_tool: pathlib.Path, authorized_at_utc: str,
) -> pathlib.Path:
    qualification._utc(authorized_at_utc, "v3 authorization timestamp")
    output_root = output_root.resolve()
    if output_root != canonical_v3_root():
        raise V3Error(f"v3 output must be canonical: {canonical_v3_root()}")
    expected_holdout = (HERE / "compact_value_bfm_discrete_v3_holdout.py").resolve()
    if (
        holdout_tool.is_symlink()
        or not holdout_tool.is_file()
        or holdout_tool.resolve() != expected_holdout
    ):
        raise V3Error("v3 holdout wrapper is not maintained exact file")
    predecessor = _validate_v2(v2_root.resolve())
    authorization_path = output_root / "governance" / "00-authorization.json"
    if authorization_path.exists():
        authorization = qualification.load_sealed(
            authorization_path, AUTHORIZATION_SCHEMA
        )
        expected = qualification.seal(_authorization_body(
            predecessor, str(authorization.get("authorized_at_utc"))
        ))
        if authorization != expected:
            raise V3Error("existing v3 authorization changed")
    else:
        authorization = qualification.write_sealed(
            authorization_path,
            _authorization_body(predecessor, authorized_at_utc),
        )
    v2_plan = predecessor["plan"]
    tools = dict(v2_plan["tools"])
    tools.update({
        "discrete_v3": _record(pathlib.Path(__file__).resolve()),
        "discrete_v3_holdout": _record(holdout_tool.resolve()),
    })
    fresh = dict(v2_plan["fresh_protected_holdout"])
    fresh.update({
        "campaign_id": f"{SUCCESSOR_CAMPAIGN_ID}-holdout",
        "game_plan_seed": 8_950_116_866_532_575_368,
        "materialized": False,
        "selection_may_change_after_results": False,
        "old_protected_routes_permitted": False,
        "tool": tools["discrete_v3_holdout"],
    })
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "discrete-v3-planned-protected-tests-unmaterialized",
        "authorization": qualification.artifact_reference(
            authorization_path, AUTHORIZATION_SCHEMA
        ),
        "predecessor": {
            "campaign_id": V2_CAMPAIGN_ID,
            "plan": qualification.artifact_reference(
                predecessor["paths"]["plan"], v2.PLAN_SCHEMA
            ),
            "selection": qualification.artifact_reference(
                predecessor["paths"]["selection"], v2.SELECTION_SCHEMA
            ),
            "outcome": qualification.artifact_reference(
                predecessor["paths"]["outcome"], v2.OUTCOME_SCHEMA
            ),
            "candidate_index": qualification.artifact_reference(
                predecessor["paths"]["candidate_index"],
                v2.CANDIDATE_INDEX_SCHEMA,
            ),
            "runtime": _record(predecessor["paths"]["runtime"]),
        },
        "training": {
            "architecture": ARCHITECTURE,
            "seed": SEED,
            "float_epoch": FLOAT_EPOCH,
            "qat_epoch": QAT_EPOCH,
            "float_metrics": predecessor["selection"]["float_validation"],
            "source_bundle_body_sha256": v1.SOURCE_BUNDLE_BODY,
            "source_bundle_manifest": dict(v2_plan["training"]["source_bundle_manifest"]),
            "safe_routes": dict(v2_plan["training"]["safe_routes"]),
            "safe_input_artifacts": dict(v2_plan["training"]["safe_input_artifacts"]),
            "roots_manifest": dict(v2_plan["training"]["roots_manifest"]),
            "roots_tsv": dict(v2_plan["training"]["roots_tsv"]),
            "prior_compact_runtime": dict(v2_plan["training"]["prior_compact_runtime"]),
            "search_teacher_runtime": dict(v2_plan["training"]["search_teacher_runtime"]),
            "new_train_manifest": dict(v2_plan["training"]["new_train_manifest"]),
            "new_train_npz": dict(v2_plan["training"]["new_train_npz"]),
            "retained_train_rows": v2_plan["training"]["retained_train_rows"],
            "split_isolation": dict(v2_plan["training"]["split_isolation"]),
        },
        "search_roster": _search_roster_body(),
        "fresh_protected_holdout": fresh,
        "source": {"plan": dict(v2_plan["source"]["plan"])},
        "tools": tools,
        "policy": {
            "float_weights_changed": False,
            "w1_changed": False,
            "w2_changed": False,
            "new_training_games": 0,
            "new_training_labels": 0,
            "protected_tests_opened": False,
            "canonical_postplan_search_executions": 1,
            "v4_authorized": False,
            "upload_authorized": False,
        },
    }
    plan_path = output_root / "discrete-v3-plan.json"
    qualification.write_sealed(plan_path, body)
    qualification.write_sealed(output_root / "training-input.json", {
        "schema": TRAINING_INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": qualification.load_sealed(plan_path, PLAN_SCHEMA)[
            "body_sha256"
        ],
        "predecessor_v2_plan": qualification.artifact_reference(
            predecessor["paths"]["plan"], v2.PLAN_SCHEMA
        ),
        "new_train_manifest": dict(v2_plan["training"]["new_train_manifest"]),
        "new_train_npz": dict(v2_plan["training"]["new_train_npz"]),
        "retained_train_rows": v2_plan["training"]["retained_train_rows"],
        "split_isolation": dict(v2_plan["training"]["split_isolation"]),
        "new_training_games_generated": 0,
        "new_training_labels_generated": 0,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": False,
        "eligible_for_discrete_search": True,
    })
    load_plan(plan_path, output_root=output_root)
    return plan_path


def load_plan(plan_path: pathlib.Path, *, output_root: pathlib.Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if (
        output_root != canonical_v3_root()
        or plan_path != output_root / "discrete-v3-plan.json"
        or plan_path.is_symlink()
        or not plan_path.is_file()
    ):
        raise V3Error("v3 plan path/root is not canonical")
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    predecessor = _validate_v2(canonical_v2_root())
    authorization_path = output_root / "governance" / "00-authorization.json"
    authorization_record = plan.get("authorization")
    if (
        not isinstance(authorization_record, Mapping)
        or pathlib.Path(str(authorization_record.get("path", "")))
        != authorization_path
        or authorization_path.is_symlink()
        or not authorization_path.is_file()
        or _sha256_file(authorization_path) != authorization_record.get("sha256")
    ):
        raise V3Error("v3 authorization binding changed")
    authorization = qualification.load_sealed(
        authorization_path, AUTHORIZATION_SCHEMA
    )
    if authorization != qualification.seal(_authorization_body(
        predecessor, str(authorization.get("authorized_at_utc"))
    )):
        raise V3Error("v3 authorization changed")
    if (
        plan.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or plan.get("status")
        != "discrete-v3-planned-protected-tests-unmaterialized"
        or plan.get("policy") != {
            "float_weights_changed": False,
            "w1_changed": False,
            "w2_changed": False,
            "new_training_games": 0,
            "new_training_labels": 0,
            "protected_tests_opened": False,
            "canonical_postplan_search_executions": 1,
            "v4_authorized": False,
            "upload_authorized": False,
        }
    ):
        raise V3Error("v3 plan policy changed")
    if plan.get("search_roster") != _search_roster_body():
        raise V3Error("v3 search roster changed")
    v2_plan = predecessor["plan"]
    expected_fresh = dict(v2_plan["fresh_protected_holdout"])
    expected_fresh.update({
        "campaign_id": f"{SUCCESSOR_CAMPAIGN_ID}-holdout",
        "game_plan_seed": 8_950_116_866_532_575_368,
        "materialized": False,
        "selection_may_change_after_results": False,
        "old_protected_routes_permitted": False,
        "tool": plan.get("tools", {}).get("discrete_v3_holdout"),
    })
    if plan.get("fresh_protected_holdout") != expected_fresh:
        raise V3Error("v3 fresh-holdout plan changed")
    if plan.get("source") != {"plan": dict(v2_plan["source"]["plan"])}:
        raise V3Error("v3 source binding changed")
    expected_training = {
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "float_epoch": FLOAT_EPOCH,
        "qat_epoch": QAT_EPOCH,
        "float_metrics": predecessor["selection"]["float_validation"],
        "source_bundle_body_sha256": v1.SOURCE_BUNDLE_BODY,
        "source_bundle_manifest": v2_plan["training"]["source_bundle_manifest"],
        "safe_routes": v2_plan["training"]["safe_routes"],
        "safe_input_artifacts": v2_plan["training"]["safe_input_artifacts"],
        "roots_manifest": v2_plan["training"]["roots_manifest"],
        "roots_tsv": v2_plan["training"]["roots_tsv"],
        "prior_compact_runtime": v2_plan["training"]["prior_compact_runtime"],
        "search_teacher_runtime": v2_plan["training"]["search_teacher_runtime"],
        "new_train_manifest": v2_plan["training"]["new_train_manifest"],
        "new_train_npz": v2_plan["training"]["new_train_npz"],
        "retained_train_rows": v2_plan["training"]["retained_train_rows"],
        "split_isolation": v2_plan["training"]["split_isolation"],
    }
    if plan.get("training") != expected_training:
        raise V3Error("v3 training binding changed")
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise V3Error("v3 tool closure missing")
    for name, record in v2_plan["tools"].items():
        if tools.get(name) != record:
            raise V3Error(f"v3 changed v2 tool {name}")
    expected_new = {
        "discrete_v3": pathlib.Path(__file__).resolve(),
        "discrete_v3_holdout": (HERE / "compact_value_bfm_discrete_v3_holdout.py").resolve(),
    }
    for name, expected in expected_new.items():
        record = tools.get(name)
        if not isinstance(record, Mapping) or pathlib.Path(str(record.get("path", ""))) != expected:
            raise V3Error(f"v3 redirects tool {name}")
    for name, record in tools.items():
        _verify_record(record, f"v3 tool {name}")
    training_path = output_root / "training-input.json"
    if training_path.is_symlink() or not training_path.is_file():
        raise V3Error("v3 training input absent")
    training = qualification.load_sealed(training_path, TRAINING_INPUT_SCHEMA)
    if (
        training.get("plan_body_sha256") != plan["body_sha256"]
        or training.get("new_train_manifest", {}).get("sha256")
        != v2.TRAIN_MANIFEST_SHA256
        or training.get("new_train_npz", {}).get("sha256") != v2.TRAIN_NPZ_SHA256
        or training.get("eligible_for_discrete_search") is not True
        or training.get("old_protected_tests_accessed") is not False
        or training.get("fresh_protected_tests_opened") is not False
    ):
        raise V3Error("v3 training-input receipt changed")
    return plan


def _acquire_lock(output_root: pathlib.Path) -> pathlib.Path:
    global _RUN_LOCK_FD
    path = output_root / "discrete-v3.lock"
    if _RUN_LOCK_FD is not None:
        return path
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise V3Error("another discrete-v3 execution is active") from error
    _RUN_LOCK_FD = descriptor
    atexit.register(_release_lock)
    return path


def _release_lock() -> None:
    global _RUN_LOCK_FD
    descriptor = _RUN_LOCK_FD
    if descriptor is None:
        return
    _RUN_LOCK_FD = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_blas() -> None:
    wrong = [name for name in v2.BLAS_ENVIRONMENT if os.environ.get(name) != "1"]
    if wrong:
        raise V3Error("v3 requires BLAS=1: " + ", ".join(wrong))


def _second_activations(dataset: Any, quantized: Any, architecture: Any) -> np.ndarray:
    output = np.empty((len(dataset), architecture.hidden_two), dtype=np.float32)
    effective = quantized.effective()
    for begin in range(0, len(dataset), 4_096):
        end = min(begin + 4_096, len(dataset))
        _prediction, cache = compact.forward(
            effective, architecture, dataset.active_rows(range(begin, end)),
            quantized=quantized,
        )
        output[begin:end] = cache[3]
    return output


def _sign_accuracy(projection: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((projection >= 0.0) == (targets >= 0.0)))


def _huber_for_scales(
    projection: np.ndarray, scales: np.ndarray, dataset: Any,
) -> np.ndarray:
    result = np.empty(len(scales), dtype=np.float64)
    denominator = max(
        float(np.sum(dataset.weights, dtype=np.float64)), np.finfo(np.float32).tiny
    )
    delta = np.float32(compact.HUBER_DELTA)
    for begin in range(0, len(scales), 64):
        selected = scales[begin : begin + 64]
        predictions = compact.fast_tanh(
            projection[:, None] * selected[None, :]
        )
        difference = predictions - dataset.targets[:, None]
        absolute = np.abs(difference)
        losses = np.where(
            absolute <= delta,
            np.float32(0.5) * difference * difference,
            delta * (absolute - np.float32(0.5) * delta),
        ).astype(np.float32)
        result[begin : begin + len(selected)] = np.sum(
            dataset.weights[:, None] * losses, axis=0, dtype=np.float64
        ) / denominator
    return result


def _grid_scales() -> np.ndarray:
    return np.asarray(
        [np.float32(tick * SCALE_TICK_VALUE)
         for tick in range(SCALE_TICK_MIN, SCALE_TICK_MAX + 1)],
        dtype=np.float32,
    )


def _candidate_key(
    float_metrics: Mapping[str, Mapping[str, float | int]],
    metrics: Mapping[str, Mapping[str, float | int]], ordinal: int,
) -> tuple[float, ...]:
    return (*iteration.gate_feasibility_key(compact, float_metrics, metrics), float(ordinal))


def _load_inputs(plan: Mapping[str, Any]) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    v1_root = canonical_v1_root()
    v1_plan = v1.load_plan(v1_root / "successor-plan.json", output_root=v1_root)
    _receipt, new_dataset = v1._load_training_input(v1_plan, v1_root)
    inputs = v1._base_inputs(v1_plan, new_dataset)
    predecessor = _validate_v2(canonical_v2_root())
    architecture, quantized, _selection, _runtime = compact.load_runtime(
        predecessor["paths"]["runtime"]
    )
    if inputs.split_isolation != plan["training"]["split_isolation"]:
        raise V3Error("v3 split isolation changed")
    return inputs, architecture, quantized, predecessor


def run(*, plan_path: pathlib.Path, output_root: pathlib.Path) -> pathlib.Path:
    _require_blas()
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    _acquire_lock(output_root)
    reference_path = output_root / "selection-reference.json"
    existing = _selection_reference(reference_path, plan=plan, output_root=output_root)
    if existing is not None:
        return existing
    claim_path = output_root / "governance" / "01-execution-claim.json"
    expected_claim = {
        "schema": EXECUTION_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "discrete-v3-execution-claimed-once",
        "plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
        "exclusive_process_lock": str((output_root / "discrete-v3.lock").resolve()),
        "canonical_postplan_executions_authorized": 1,
        "preplan_diagnostics_do_not_claim_blind_validation": True,
        "new_training_games": 0,
        "new_training_labels": 0,
        "protected_tests_opened": False,
        "v4_authorized": False,
    }
    if claim_path.exists():
        claim = qualification.load_sealed(claim_path, EXECUTION_CLAIM_SCHEMA)
        if any(claim.get(field) != value for field, value in expected_claim.items()):
            raise V3Error("v3 execution claim changed")
    else:
        claim = qualification.write_sealed(claim_path, {
            **expected_claim, "claimed_at_utc": utc_now(),
        })
    inputs, architecture, base_quantized, predecessor = _load_inputs(plan)
    float_metrics = plan["training"]["float_metrics"]
    common_h2 = _second_activations(
        inputs.common_adjudicator, base_quantized, architecture
    )
    canonical_h2 = _second_activations(
        inputs.canonical_validation, base_quantized, architecture
    )
    code_vectors = np.asarray(
        list(itertools.product(*CODE_OPTIONS)), dtype=np.int8
    )
    if code_vectors.shape != (CODE_VECTOR_COUNT, 8):
        raise V3Error("v3 code-vector roster changed")
    scales = _grid_scales()
    common_sign_floor = max(
        compact.COMMON_MINIMUM_SIGN,
        float(float_metrics["common_adjudicator"]["sign_accuracy"])
        - compact.MAXIMUM_SIGN_LOSS + 1e-15,
    )
    canonical_sign_floor = max(
        compact.CANONICAL_MINIMUM_SIGN,
        float(float_metrics["canonical_validation"]["sign_accuracy"])
        - compact.MAXIMUM_SIGN_LOSS + 1e-15,
    )
    common_huber_limit = min(
        compact.COMMON_MAXIMUM_HUBER,
        float(float_metrics["common_adjudicator"]["weighted_huber"])
        * compact.MAXIMUM_HUBER_RATIO,
    )
    canonical_huber_limit = min(
        compact.CANONICAL_MAXIMUM_HUBER,
        float(float_metrics["canonical_validation"]["weighted_huber"])
        * compact.MAXIMUM_HUBER_RATIO,
    )
    sign_feasible_codes = 0
    common_survivors = 0
    passing = []
    arm = compact.ARMS["search-target"]
    for code_index, codes in enumerate(code_vectors):
        common_projection = np.asarray(common_h2 @ codes.astype(np.float32), dtype=np.float32)
        canonical_projection = np.asarray(
            canonical_h2 @ codes.astype(np.float32), dtype=np.float32
        )
        common_sign = _sign_accuracy(common_projection, inputs.common_adjudicator.targets)
        canonical_sign = _sign_accuracy(
            canonical_projection, inputs.canonical_validation.targets
        )
        if common_sign < common_sign_floor or canonical_sign < canonical_sign_floor:
            continue
        sign_feasible_codes += 1
        common_hubers = _huber_for_scales(
            common_projection, scales, inputs.common_adjudicator
        )
        scale_indices = np.flatnonzero(common_hubers <= common_huber_limit)
        common_survivors += len(scale_indices)
        if not len(scale_indices):
            continue
        canonical_hubers = _huber_for_scales(
            canonical_projection, scales[scale_indices], inputs.canonical_validation
        )
        passing_indices = scale_indices[
            canonical_hubers <= canonical_huber_limit
        ]
        for scale_index in passing_indices:
            scale = np.float32(scales[int(scale_index)])
            common_predictions = compact.fast_tanh(common_projection * scale)
            canonical_predictions = compact.fast_tanh(canonical_projection * scale)
            metrics = {
                "common_adjudicator": compact.metrics_from_predictions(
                    common_predictions, inputs.common_adjudicator, arm
                ),
                "canonical_validation": compact.metrics_from_predictions(
                    canonical_predictions, inputs.canonical_validation, arm
                ),
            }
            gate = compact.offline_advancement_gate(float_metrics, metrics)
            ordinal = code_index * SCALE_COUNT + int(scale_index)
            if gate["passed"]:
                passing.append({
                    "ordinal": ordinal,
                    "code_index": code_index,
                    "scale_index": int(scale_index),
                    "w3_codes": [int(value) for value in codes],
                    "w3_scale": float(scale),
                    "metrics": metrics,
                    "offline_gate": gate,
                    "selection_key": list(_candidate_key(
                        float_metrics, metrics, ordinal
                    )),
                })
    if (
        sign_feasible_codes != EXPECTED_SIGN_FEASIBLE_CODES
        or common_survivors != EXPECTED_COMMON_SURVIVORS
        or len(passing) != EXPECTED_PASSING_CANDIDATES
    ):
        raise V3Error(
            "v3 deterministic pruning/pass counts changed: "
            f"sign={sign_feasible_codes},common={common_survivors},"
            f"passing={len(passing)}"
        )
    selected = min(passing, key=lambda row: tuple(row["selection_key"]))
    if (
        tuple(selected["w3_codes"]) != EXPECTED_WINNER_CODES
        or selected["w3_scale"] != EXPECTED_WINNER_SCALE
    ):
        raise V3Error("v3 deterministic winner changed")
    search_receipt_path = output_root / "search-receipt.json"
    integer = {
        name: value.copy() for name, value in base_quantized.integer.items()
    }
    integer["w3"] = np.asarray(selected["w3_codes"], dtype=np.int8)
    quantized = compact.QuantizedWeights(
        integer,
        {
            "w1": base_quantized.scales["w1"],
            "w2": base_quantized.scales["w2"],
            "w3": np.float32(selected["w3_scale"]),
        },
    )
    exact_metrics = compact.evaluate_validation_pair(
        quantized.effective(), architecture, inputs, arm, quantized=quantized
    )
    exact_gate = compact.offline_advancement_gate(float_metrics, exact_metrics)
    if exact_gate.get("passed") is not True:
        raise V3Error("v3 exact maintained evaluator rejected the search winner")
    parity = {
        **compact.assert_quantized_inference_parity(
            quantized, architecture, inputs.common_adjudicator,
            maximum_rows=4_096,
        ),
        "passed": True,
    }
    exact_selection_key = list(_candidate_key(
        float_metrics, exact_metrics, int(selected["ordinal"])
    ))
    qualification.write_sealed(search_receipt_path, {
        "schema": SEARCH_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "execution_claim": qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        ),
        "code_vector_count": CODE_VECTOR_COUNT,
        "scale_count": SCALE_COUNT,
        "total_grid_candidates": TOTAL_GRID_CANDIDATES,
        "sign_feasible_codes": sign_feasible_codes,
        "common_survivors": common_survivors,
        "passing_candidates": passing,
        "passing_candidate_count": len(passing),
        "selected_ordinal": selected["ordinal"],
        "selected_exact_metrics": exact_metrics,
        "selected_exact_gate": exact_gate,
        "selected_exact_selection_key": exact_selection_key,
        "protected_tests_opened": False,
    })
    runtime = compact.write_runtime(
        output_root / "training" / "quantized-runtimes",
        architecture, quantized, arm="search-target", seed=SEED,
        float_epoch=FLOAT_EPOCH, qat_epoch=QAT_EPOCH,
        source_bundle_body_sha256=v1.SOURCE_BUNDLE_BODY,
    )
    generated_source, source_export = iteration.render_iteration_source(
        runtime=runtime, plan=plan, output_root=output_root
    )
    selection_body = {
        "schema": SELECTION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": exact_gate["status"],
        "plan_body_sha256": plan["body_sha256"],
        "execution_claim": qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        ),
        "predecessor_runtime": dict(plan["predecessor"]["runtime"]),
        "training_input": qualification.artifact_reference(
            output_root / "training-input.json", TRAINING_INPUT_SCHEMA
        ),
        "search_receipt": qualification.artifact_reference(
            search_receipt_path, SEARCH_RECEIPT_SCHEMA
        ),
        "architecture": ARCHITECTURE,
        "arm": "search-target",
        "seed": SEED,
        "float_epoch": FLOAT_EPOCH,
        "qat_epoch": QAT_EPOCH,
        "candidate_ordinal": selected["ordinal"],
        "w3_codes": selected["w3_codes"],
        "scales": {
            name: float(quantized.scales[name]) for name in ("w1", "w2", "w3")
        },
        "float_validation": float_metrics,
        "quantized_validation": exact_metrics,
        "offline_gate": exact_gate,
        "selection_key": exact_selection_key,
        "runtime": _record(runtime),
        "generated_source": _record(generated_source),
        "source_export": source_export,
        "inference_parity": parity,
        "float_weights_changed": False,
        "w1_changed": False,
        "w2_changed": False,
        "new_training_games_generated": 0,
        "new_training_labels_generated": 0,
        "selection_immutable": True,
        "selection_may_change_after_fresh_protected_tests": False,
        "old_protected_tests_accessed": False,
        "old_protected_tests_permanently_excluded": True,
        "fresh_protected_tests_opened": False,
        "fresh_protected_tests_authorized": exact_gate["passed"],
        "game_gated": False,
        "upload_authorized": False,
        "v4_authorized": False,
    }
    selection_path, _selection = _write_content_addressed(
        output_root / "selections", selection_body, ".discrete-v3-selection.json"
    )
    outcome_path = output_root / "governance" / "02-outcome.json"
    qualification.write_sealed(outcome_path, {
        "schema": OUTCOME_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": (
            "discrete-v3-offline-qualified-awaiting-fresh-tests"
            if exact_gate["passed"]
            else "discrete-v3-terminal-offline-rejection"
        ),
        "selection": qualification.artifact_reference(
            selection_path, SELECTION_SCHEMA
        ),
        "offline_gate": exact_gate,
        "fresh_protected_tests_authorized": exact_gate["passed"],
        "fresh_protected_tests_opened": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
        "v4_authorized": False,
    })
    qualification.write_sealed(reference_path, {
        "schema": SELECTION_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "selection": _record(selection_path),
        "outcome": qualification.artifact_reference(outcome_path, OUTCOME_SCHEMA),
        "offline_gate_passed": exact_gate["passed"],
        "fresh_protected_tests_opened": False,
        "selection_immutable": True,
    })
    validated = _selection_reference(
        reference_path, plan=plan, output_root=output_root
    )
    if validated != selection_path:
        raise V3Error("v3 selection failed immediate validation")
    return selection_path


def _validate_selection_closure(
    selection_path: pathlib.Path, *, plan: Mapping[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    if (
        selection_path.parent != output_root / "selections"
        or not selection_path.name.endswith(".discrete-v3-selection.json")
        or selection_path.is_symlink()
        or not selection_path.is_file()
        or _sha256_file(selection_path)
        != selection_path.name.removesuffix(".discrete-v3-selection.json")
    ):
        raise V3Error("v3 selection path/content address changed")
    selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
    claim_path = output_root / "governance" / "01-execution-claim.json"
    search_path = output_root / "search-receipt.json"
    if (
        claim_path.is_symlink() or not claim_path.is_file()
        or search_path.is_symlink() or not search_path.is_file()
    ):
        raise V3Error("v3 claim/search receipt is absent or redirected")
    claim = qualification.load_sealed(claim_path, EXECUTION_CLAIM_SCHEMA)
    search = qualification.load_sealed(search_path, SEARCH_RECEIPT_SCHEMA)
    gate = compact.offline_advancement_gate(
        plan["training"]["float_metrics"], selection.get("quantized_validation", {})
    )
    if (
        selection.get("namespace") != NAMESPACE
        or selection.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or selection.get("status") != gate.get("status")
        or selection.get("plan_body_sha256") != plan["body_sha256"]
        or selection.get("architecture") != ARCHITECTURE
        or selection.get("arm") != "search-target"
        or selection.get("seed") != SEED
        or selection.get("float_epoch") != FLOAT_EPOCH
        or selection.get("qat_epoch") != QAT_EPOCH
        or selection.get("predecessor_runtime") != plan["predecessor"]["runtime"]
        or selection.get("training_input") != qualification.artifact_reference(
            output_root / "training-input.json", TRAINING_INPUT_SCHEMA
        )
        or selection.get("float_validation") != plan["training"]["float_metrics"]
        or selection.get("offline_gate") != gate
        or gate.get("passed") is not True
        or selection.get("w3_codes") != list(EXPECTED_WINNER_CODES)
        or selection.get("scales", {}).get("w3") != EXPECTED_WINNER_SCALE
        or selection.get("execution_claim") != qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        )
        or selection.get("search_receipt") != qualification.artifact_reference(
            search_path, SEARCH_RECEIPT_SCHEMA
        )
        or selection.get("selection_immutable") is not True
        or selection.get("float_weights_changed") is not False
        or selection.get("w1_changed") is not False
        or selection.get("w2_changed") is not False
        or selection.get("new_training_games_generated") != 0
        or selection.get("new_training_labels_generated") != 0
        or selection.get("old_protected_tests_accessed") is not False
        or selection.get("old_protected_tests_permanently_excluded") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("fresh_protected_tests_authorized") is not True
        or selection.get("game_gated") is not False
        or selection.get("upload_authorized") is not False
        or selection.get("v4_authorized") is not False
    ):
        raise V3Error("v3 selection policy changed")
    if (
        claim.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or claim.get("status") != "discrete-v3-execution-claimed-once"
        or claim.get("plan") != qualification.artifact_reference(
            output_root / "discrete-v3-plan.json", PLAN_SCHEMA
        )
        or claim.get("exclusive_process_lock")
        != str((output_root / "discrete-v3.lock").resolve())
        or claim.get("canonical_postplan_executions_authorized") != 1
        or claim.get("preplan_diagnostics_do_not_claim_blind_validation") is not True
        or claim.get("new_training_games") != 0
        or claim.get("new_training_labels") != 0
        or claim.get("protected_tests_opened") is not False
        or claim.get("v4_authorized") is not False
    ):
        raise V3Error("v3 execution claim changed")
    passing = search.get("passing_candidates")
    if (
        search.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or search.get("plan_body_sha256") != plan["body_sha256"]
        or search.get("execution_claim") != qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        )
        or search.get("code_vector_count") != CODE_VECTOR_COUNT
        or search.get("scale_count") != SCALE_COUNT
        or search.get("total_grid_candidates") != TOTAL_GRID_CANDIDATES
        or search.get("sign_feasible_codes") != EXPECTED_SIGN_FEASIBLE_CODES
        or search.get("common_survivors") != EXPECTED_COMMON_SURVIVORS
        or search.get("passing_candidate_count") != EXPECTED_PASSING_CANDIDATES
        or not isinstance(passing, list)
        or len(passing) != EXPECTED_PASSING_CANDIDATES
        or search.get("selected_ordinal") != selection.get("candidate_ordinal")
        or search.get("selected_exact_metrics")
        != selection.get("quantized_validation")
        or search.get("selected_exact_gate") != selection.get("offline_gate")
        or search.get("selected_exact_selection_key")
        != selection.get("selection_key")
        or search.get("protected_tests_opened") is not False
    ):
        raise V3Error("v3 search receipt changed")
    code_vectors = np.asarray(
        list(itertools.product(*CODE_OPTIONS)), dtype=np.int8
    )
    scales = _grid_scales()
    seen_ordinals: set[int] = set()
    for row in passing:
        if not isinstance(row, Mapping):
            raise V3Error("v3 passing candidate is malformed")
        metrics = row.get("metrics")
        ordinal = row.get("ordinal")
        if not isinstance(metrics, Mapping) or type(ordinal) is not int:
            raise V3Error("v3 passing candidate identity is malformed")
        if (
            not 0 <= ordinal < TOTAL_GRID_CANDIDATES
            or ordinal in seen_ordinals
            or row.get("code_index") != ordinal // SCALE_COUNT
            or row.get("scale_index") != ordinal % SCALE_COUNT
            or row.get("w3_codes") != [
                int(value) for value in code_vectors[ordinal // SCALE_COUNT]
            ]
            or row.get("w3_scale")
            != float(scales[ordinal % SCALE_COUNT])
        ):
            raise V3Error("v3 passing candidate grid identity changed")
        seen_ordinals.add(ordinal)
        expected_gate = compact.offline_advancement_gate(
            plan["training"]["float_metrics"], metrics
        )
        expected_key = list(_candidate_key(
            plan["training"]["float_metrics"], metrics, ordinal
        ))
        if (
            expected_gate.get("passed") is not True
            or row.get("offline_gate") != expected_gate
            or row.get("selection_key") != expected_key
        ):
            raise V3Error("v3 passing candidate gate/key changed")
    expected_winner = min(passing, key=lambda row: tuple(row["selection_key"]))
    if (
        expected_winner.get("ordinal") != selection.get("candidate_ordinal")
        or expected_winner.get("w3_codes") != list(EXPECTED_WINNER_CODES)
        or expected_winner.get("w3_scale") != EXPECTED_WINNER_SCALE
    ):
        raise V3Error("v3 selection is not the exact grid argmin")
    runtime_path = _verify_record(selection.get("runtime"), "v3 runtime")
    source_path = _verify_record(selection.get("generated_source"), "v3 source")
    if (
        runtime_path.parent != output_root / "training" / "quantized-runtimes"
        or source_path.parent != output_root / "fine-tune" / "generated-sources"
    ):
        raise V3Error("v3 deployment path changed")
    architecture, quantized, runtime_selection, _runtime = compact.load_runtime(runtime_path)
    predecessor = _validate_v2(canonical_v2_root())
    base_quantized = predecessor["quantized"]
    if (
        architecture.name != ARCHITECTURE
        or runtime_selection.get("arm") != "search-target"
        or runtime_selection.get("seed") != SEED
        or runtime_selection.get("float_epoch") != FLOAT_EPOCH
        or runtime_selection.get("qat_epoch") != QAT_EPOCH
        or runtime_selection.get("source_bundle_body_sha256") != v1.SOURCE_BUNDLE_BODY
        or not np.array_equal(
            quantized.integer["w1"], base_quantized.integer["w1"]
        )
        or not np.array_equal(
            quantized.integer["w2"], base_quantized.integer["w2"]
        )
        or float(quantized.scales["w1"]) != float(base_quantized.scales["w1"])
        or float(quantized.scales["w2"]) != float(base_quantized.scales["w2"])
        or tuple(int(value) for value in quantized.integer["w3"])
        != EXPECTED_WINNER_CODES
        or float(quantized.scales["w3"]) != EXPECTED_WINNER_SCALE
        or selection.get("scales") != {
            name: float(quantized.scales[name]) for name in ("w1", "w2", "w3")
        }
        or selection.get("source_export", {}).get("runtime_sha256")
        != selection["runtime"]["sha256"]
        or selection.get("source_export", {}).get("source_sha256")
        != selection["generated_source"]["sha256"]
        or selection.get("source_export", {}).get("source_ascii_bytes")
        != selection["generated_source"]["bytes"]
        or selection.get("source_export", {}).get("source_limit_exclusive")
        != 95_000
        or selection.get("inference_parity", {}).get("passed") is not True
        or not 0 < int(selection["generated_source"]["bytes"]) < 95_000
    ):
        raise V3Error("v3 runtime/source closure changed")
    inputs, exact_architecture, _base, _predecessor = _load_inputs(plan)
    exact_metrics = compact.evaluate_validation_pair(
        quantized.effective(), exact_architecture, inputs,
        compact.ARMS["search-target"], quantized=quantized,
    )
    exact_gate = compact.offline_advancement_gate(
        plan["training"]["float_metrics"], exact_metrics
    )
    exact_parity = {
        **compact.assert_quantized_inference_parity(
            quantized, exact_architecture, inputs.common_adjudicator,
            maximum_rows=4_096,
        ),
        "passed": True,
    }
    if (
        selection.get("quantized_validation") != exact_metrics
        or selection.get("offline_gate") != exact_gate
        or exact_gate.get("passed") is not True
        or selection.get("inference_parity") != exact_parity
        or selection.get("selection_key") != list(_candidate_key(
            plan["training"]["float_metrics"], exact_metrics,
            int(selection["candidate_ordinal"]),
        ))
    ):
        raise V3Error("v3 exact metrics/parity closure changed")
    return selection


def _validate_outcome(
    *, plan: Mapping[str, Any], output_root: pathlib.Path,
    selection_path: pathlib.Path,
) -> dict[str, Any]:
    path = output_root / "governance" / "02-outcome.json"
    if path.is_symlink() or not path.is_file():
        raise V3Error("v3 outcome absent")
    outcome = qualification.load_sealed(path, OUTCOME_SCHEMA)
    selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
    if (
        outcome.get("namespace") != NAMESPACE
        or outcome.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or outcome.get("status") != "discrete-v3-offline-qualified-awaiting-fresh-tests"
        or outcome.get("selection") != qualification.artifact_reference(
            selection_path, SELECTION_SCHEMA
        )
        or outcome.get("offline_gate") != selection["offline_gate"]
        or outcome.get("fresh_protected_tests_authorized") is not True
        or outcome.get("fresh_protected_tests_opened") is not False
        or outcome.get("rank4_gate_authorized") is not False
        or outcome.get("upload_authorized") is not False
        or outcome.get("v4_authorized") is not False
        or plan.get("policy", {}).get("v4_authorized") is not False
    ):
        raise V3Error("v3 outcome changed")
    return outcome


def _selection_reference(
    path: pathlib.Path, *, plan: Mapping[str, Any], output_root: pathlib.Path,
) -> pathlib.Path | None:
    if not path.exists():
        return None
    outcome_path = output_root / "governance" / "02-outcome.json"
    if (
        path != output_root / "selection-reference.json"
        or path.is_symlink()
        or outcome_path.is_symlink()
        or not outcome_path.is_file()
    ):
        raise V3Error("v3 reference/outcome path changed")
    reference = qualification.load_sealed(path, SELECTION_REFERENCE_SCHEMA)
    record = reference.get("selection")
    if (
        reference.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or reference.get("plan_body_sha256") != plan["body_sha256"]
        or reference.get("outcome") != qualification.artifact_reference(
            outcome_path, OUTCOME_SCHEMA
        )
        or reference.get("offline_gate_passed") is not True
        or reference.get("fresh_protected_tests_opened") is not False
        or reference.get("selection_immutable") is not True
        or not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
    ):
        raise V3Error("v3 selection reference changed")
    selection_path = pathlib.Path(record["path"])
    _verify_record(record, "v3 selection")
    _validate_selection_closure(selection_path, plan=plan, output_root=output_root)
    _validate_outcome(plan=plan, output_root=output_root, selection_path=selection_path)
    return selection_path


def verify(*, plan_path: pathlib.Path, output_root: pathlib.Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    selection = _selection_reference(
        output_root / "selection-reference.json", plan=plan, output_root=output_root
    )
    result: dict[str, Any] = {
        "schema": "papersoccer.compact-value-bfm.discrete-v3-status.v1",
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "plan_valid": True,
        "selection_complete": selection is not None,
        "protected_tests_opened": False,
    }
    if selection is not None:
        value = qualification.load_sealed(selection, SELECTION_SCHEMA)
        result.update({
            "selection": _record(selection),
            "offline_gate": value["offline_gate"],
            "w3_codes": value["w3_codes"],
            "w3_scale": value["scales"]["w3"],
        })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--v2-root", type=pathlib.Path, required=True)
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--holdout-tool", type=pathlib.Path, required=True)
    p.add_argument("--authorized-at-utc", default=utc_now())
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare(
                v2_root=args.v2_root, output_root=args.output_root,
                holdout_tool=args.holdout_tool,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "run":
            result = run(plan_path=args.plan, output_root=args.output_root)
        else:
            result = verify(plan_path=args.plan, output_root=args.output_root)
        if isinstance(result, pathlib.Path):
            result = {"path": str(result.resolve()), "sha256": _sha256_file(result)}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        V3Error, compact.TrainingError, iteration.IterationError,
        OSError, ValueError, json.JSONDecodeError,
    ) as error:
        print(f"compact discrete-v3 failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
