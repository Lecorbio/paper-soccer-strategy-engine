#!/usr/bin/env python3
"""One-shot quantization-only recovery for the rejected clean successor.

The immutable v1 float checkpoint already passes absolute development floors.
This campaign changes neither float weights nor data: it runs one finite,
precommitted 3-bit QAT/scale roster and publishes at most one immutable v2
selection.  This quantization phase generates no training game or teacher
label.  A separately bound one-shot fresh evaluation holdout is permitted only
after an immutable offline pass; upload remains forbidden.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load quantization-v2 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load(HERE / "compact_value_bfm_successor.py", "compact_quant_v2_predecessor")
qualification = v1.qualification
compact = v1.compact
iteration = v1.iteration


V2Error = qualification.QualificationError
SuccessorError = V2Error
NAMESPACE = v1.NAMESPACE
SUCCESSOR_CAMPAIGN_ID = "compact-value-bfm-quantization-v2-20260901-v1"
V1_CAMPAIGN_ID = v1.SUCCESSOR_CAMPAIGN_ID
V1_PLAN_FILE_SHA256 = (
    "88e5fd1b543da15694d4f897a6f94e738e3d687be0dac91b1f6106a66a4b31b7"
)
V1_PLAN_BODY_SHA256 = (
    "9488731809b2f80014ec6a1901b6c4f2ac732acf9add1aa41abf880b1057e256"
)
V1_TRAINING_FILE_SHA256 = (
    "4b24d6025b73643da77366c2ecc9656672f0ff8ff1358594a5f2ae84f416c8fe"
)
V1_TRAINING_BODY_SHA256 = (
    "c97855ec60a6bc96b094c1daead6249592d5c54eb10dbde216e2b5e653c5e1f5"
)
V1_REPACK_FILE_SHA256 = (
    "be1eae304032ddcfc9891ec35488a14fa3b3f2ea83b491addd7ba08f105b9931"
)
V1_REPACK_BODY_SHA256 = (
    "3a5d684ddb6192f082ac50a39cf0fdce75556adf0669ba7687dcf476c6630319"
)
V1_REJECTION_FILE_SHA256 = (
    "4be880cdab37d7b1e283abd2f843ee06435b9aec67b7c088cc9f531ba62a3d99"
)
V1_REJECTION_BODY_SHA256 = (
    "3a301eb0687a0cbd7ee9ed99c5c85b291a29a27bc0447b881f9681dfae3564d0"
)
V1_SELECTION_FILE_SHA256 = (
    "6f625e31ca862a0c1d5122ef29b924d4f7d630232075619f458c7f15d927e38a"
)
V1_SELECTION_BODY_SHA256 = (
    "5747d11cebbd30c7ac0a65627529ad752084582d8d55a00cf5b12787bf774e6d"
)
V1_FLOAT_SHA256 = (
    "0dbe279295bcfeb80392e10ff9b04f9bd5c0eb4e1ea45c974b6a92d266f37729"
)
TRAIN_MANIFEST_SHA256 = (
    "7b00d7028c689d87005b070f34a89fc3e72b14c1057e8eeb2b276946af047c10"
)
TRAIN_NPZ_SHA256 = (
    "997c63a4799383e9d40ed2ab09e57f7a07f4149538f7fe201be4d20a5fbbc0f9"
)

AUTHORIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.quantization-v2-authorization.v1"
)
PLAN_SCHEMA = "papersoccer.compact-value-bfm.quantization-v2-plan.v1"
EXECUTION_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.quantization-v2-execution-claim.v1"
)
TRAINING_INPUT_SCHEMA = (
    "papersoccer.compact-value-bfm.quantization-v2-training-input.v1"
)
TRAJECTORY_SCHEMA = "papersoccer.compact-value-bfm.quantization-v2-trajectory.v1"
CALIBRATION_SCHEMA = "papersoccer.compact-value-bfm.quantization-v2-calibration.v1"
CANDIDATE_INDEX_SCHEMA = (
    "papersoccer.compact-value-bfm.quantization-v2-candidate-index.v1"
)
SELECTION_SCHEMA = "papersoccer.compact-value-bfm.quantization-v2-selection.v1"
SELECTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.quantization-v2-selection-reference.v1"
)
OUTCOME_SCHEMA = "papersoccer.compact-value-bfm.quantization-v2-outcome.v1"

ARCHITECTURE = v1.ARCHITECTURE
SEED = v1.SEED
FLOAT_EPOCH = 2
QAT_EPOCHS = 4
SCHEDULE_EPOCHS = (51, 52, 53, 54)
QAT_LEARNING_RATES = (0.0000075, 0.000015, 0.00003)
WEIGHT_DECAY = 0.00001
BASE_SCALES = {
    "w1": 0.0801650658249855,
    "w2": 0.16067029535770416,
    "w3": 0.04906708374619484,
}
SCALE_FACTORS = {
    "w1": (0.875, 0.9375, 1.0, 1.0625, 1.125),
    "w2": (0.875, 0.9375, 1.0, 1.0625, 1.125),
    "w3": (0.90625, 0.9375, 0.96875, 1.0, 1.03125, 1.0625, 1.09375),
}
CALIBRATION_PASSES = 2
FIXED_CANDIDATES = 1 + len(QAT_LEARNING_RATES) * QAT_EPOCHS
CALIBRATION_CANDIDATES = len(QAT_LEARNING_RATES) * CALIBRATION_PASSES * sum(
    len(values) for values in SCALE_FACTORS.values()
)
TOTAL_CANDIDATES = FIXED_CANDIDATES + CALIBRATION_CANDIDATES
BLAS_ENVIRONMENT = v1.BLAS_ENVIRONMENT
SOURCE_PLAN_SCHEMA = v1.SOURCE_PLAN_SCHEMA
selfsearch = v1.selfsearch
large_training = v1.large_training
_canonical_json = v1._canonical_json
_declared_record = v1._declared_record
retired_protected_paths = v1.retired_protected_paths
_RUN_LOCK_FD: int | None = None


def _acquire_run_lock(output_root: pathlib.Path) -> pathlib.Path:
    global _RUN_LOCK_FD
    path = output_root / "quantization-v2.lock"
    if _RUN_LOCK_FD is not None:
        return path
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise V2Error("another quantization-v2 execution is active") from error
    _RUN_LOCK_FD = descriptor
    atexit.register(_release_run_lock)
    return path


def _release_run_lock() -> None:
    global _RUN_LOCK_FD
    descriptor = _RUN_LOCK_FD
    if descriptor is None:
        return
    _RUN_LOCK_FD = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_v1_root() -> pathlib.Path:
    return (
        REPOSITORY / "results" / "compact_value_bfm" / V1_CAMPAIGN_ID
    ).resolve()


def canonical_v2_root() -> pathlib.Path:
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
        raise V2Error(f"v2 artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_record(record: object, label: str) -> pathlib.Path:
    return v1._verify_record(record, label)


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    return v1._write_content_addressed(directory, body, suffix)


def _v1_paths(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "plan": root / "successor-plan.json",
        "training_input": root / "training-input.json",
        "repack": root / "global-repack" / "repack-receipt.json",
        "rejection": root / "governance" / "02-offline-rejection.json",
        "selection": (
            root / "selections"
            / f"{V1_SELECTION_FILE_SHA256}.successor-selection.json"
        ),
        "float_checkpoint": (
            root / "training" / "float-checkpoints"
            / f"{V1_FLOAT_SHA256}.float.npz"
        ),
    }


def _validate_v1(root: pathlib.Path) -> dict[str, Any]:
    if root.resolve() != canonical_v1_root():
        raise V2Error("v2 predecessor root is not canonical")
    paths = _v1_paths(root.resolve())
    expected_files = {
        "plan": V1_PLAN_FILE_SHA256,
        "training_input": V1_TRAINING_FILE_SHA256,
        "repack": V1_REPACK_FILE_SHA256,
        "rejection": V1_REJECTION_FILE_SHA256,
        "selection": V1_SELECTION_FILE_SHA256,
        "float_checkpoint": V1_FLOAT_SHA256,
    }
    for name, digest in expected_files.items():
        if paths[name].is_symlink() or not paths[name].is_file():
            raise V2Error(f"v1 {name} is not a regular file")
        if _sha256_file(paths[name]) != digest:
            raise V2Error(f"v1 {name} changed")
    plan = v1.load_plan(paths["plan"], output_root=root)
    training = qualification.load_sealed(
        paths["training_input"], v1.TRAINING_INPUT_SCHEMA
    )
    repack = qualification.load_sealed(paths["repack"], v1.REPACK_SCHEMA)
    rejection = qualification.load_sealed(
        paths["rejection"],
        "papersoccer.compact-value-bfm.clean-successor-offline-rejection.v1",
    )
    selection = qualification.load_sealed(paths["selection"], v1.SELECTION_SCHEMA)
    if (
        plan.get("body_sha256") != V1_PLAN_BODY_SHA256
        or training.get("body_sha256") != V1_TRAINING_BODY_SHA256
        or repack.get("body_sha256") != V1_REPACK_BODY_SHA256
        or rejection.get("body_sha256") != V1_REJECTION_BODY_SHA256
        or selection.get("body_sha256") != V1_SELECTION_BODY_SHA256
        or selection.get("offline_gate", {}).get("passed") is not False
        or selection.get("selection_immutable") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("old_protected_tests_accessed") is not False
        or rejection.get("fresh_protected_tests_opened") is not False
        or rejection.get("rank4_gate_authorized") is not False
        or rejection.get("upload_authorized") is not False
        or training.get("eligible_for_training") is not True
        or training.get("new_train_manifest", {}).get("sha256")
        != TRAIN_MANIFEST_SHA256
        or training.get("new_train_npz", {}).get("sha256") != TRAIN_NPZ_SHA256
        or selection.get("float_checkpoint", {}).get("sha256") != V1_FLOAT_SHA256
    ):
        raise V2Error("v1 rejection/input boundary changed")
    if (root / "fresh-holdout" / "materialization-receipt.json").exists():
        raise V2Error("v1 fresh protected holdout was materialized")
    return {
        "paths": paths,
        "plan": plan,
        "training": training,
        "repack": repack,
        "rejection": rejection,
        "selection": selection,
    }


def _authorization_body(v1_state: Mapping[str, Any], authorized_at_utc: str) -> dict[str, Any]:
    paths = v1_state["paths"]
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "one-quantization-only-v2-authorized",
        "authorized_at_utc": authorized_at_utc,
        "authorization_basis": (
            "user-explicitly-authorized-one-quantization-only-v2-no-v3"
        ),
        "v1_rejection": qualification.artifact_reference(
            paths["rejection"],
            "papersoccer.compact-value-bfm.clean-successor-offline-rejection.v1",
        ),
        "v1_float_checkpoint": _record(paths["float_checkpoint"]),
        "v1_training_input": qualification.artifact_reference(
            paths["training_input"], v1.TRAINING_INPUT_SCHEMA
        ),
        "float_fine_tuning_authorized": False,
        "architecture_change_authorized": False,
        "new_training_games_authorized": 0,
        "new_training_labels_authorized": 0,
        "repack_authorized": False,
        "quantization_executions_authorized": 1,
        "fresh_protected_holdout_materializations_authorized_after_pass": 1,
        "fresh_evaluation_games_and_labels_authorized_only_after_pass": True,
        "v3_authorized": False,
        "old_or_fresh_protected_tests_authorized_preselection": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


def prepare(
    *, v1_root: pathlib.Path, output_root: pathlib.Path,
    holdout_tool: pathlib.Path, authorized_at_utc: str,
) -> pathlib.Path:
    qualification._utc(authorized_at_utc, "v2 authorization timestamp")
    output_root = output_root.resolve()
    if output_root != canonical_v2_root():
        raise V2Error(f"v2 output must be the canonical root: {canonical_v2_root()}")
    expected_holdout = (HERE / "compact_value_bfm_quantization_v2_holdout.py").resolve()
    if (
        holdout_tool.is_symlink()
        or not holdout_tool.is_file()
        or holdout_tool.resolve() != expected_holdout
    ):
        raise V2Error("v2 holdout wrapper is not the maintained exact file")
    predecessor = _validate_v1(v1_root.resolve())
    authorization_path = output_root / "governance" / "00-authorization.json"
    if authorization_path.exists():
        authorization = qualification.load_sealed(
            authorization_path, AUTHORIZATION_SCHEMA
        )
        expected = qualification.seal(_authorization_body(
            predecessor, str(authorization.get("authorized_at_utc"))
        ))
        if authorization != expected:
            raise V2Error("existing v2 authorization changed")
    else:
        authorization = qualification.write_sealed(
            authorization_path,
            _authorization_body(predecessor, authorized_at_utc),
        )
    v1_plan = predecessor["plan"]
    tools = dict(v1_plan["tools"])
    tools.update({
        "quantization_v2": _record(pathlib.Path(__file__).resolve()),
        "quantization_v2_holdout": _record(holdout_tool.resolve()),
    })
    fresh = dict(v1_plan["fresh_protected_holdout"])
    fresh.update({
        "campaign_id": f"{SUCCESSOR_CAMPAIGN_ID}-holdout",
        "game_plan_seed": 8_950_116_866_532_575_367,
        "materialized": False,
        "selection_may_change_after_results": False,
        "old_protected_routes_permitted": False,
        "tool": tools["quantization_v2_holdout"],
    })
    plan_path = output_root / "quantization-v2-plan.json"
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "quantization-v2-planned-protected-tests-unmaterialized",
        "authorization": qualification.artifact_reference(
            authorization_path, AUTHORIZATION_SCHEMA
        ),
        "predecessor": {
            "campaign_id": V1_CAMPAIGN_ID,
            "plan": qualification.artifact_reference(
                predecessor["paths"]["plan"], v1.PLAN_SCHEMA
            ),
            "training_input": qualification.artifact_reference(
                predecessor["paths"]["training_input"], v1.TRAINING_INPUT_SCHEMA
            ),
            "repack": qualification.artifact_reference(
                predecessor["paths"]["repack"], v1.REPACK_SCHEMA
            ),
            "rejection": qualification.artifact_reference(
                predecessor["paths"]["rejection"],
                "papersoccer.compact-value-bfm.clean-successor-offline-rejection.v1",
            ),
            "selection": qualification.artifact_reference(
                predecessor["paths"]["selection"], v1.SELECTION_SCHEMA
            ),
            "float_checkpoint": _record(predecessor["paths"]["float_checkpoint"]),
        },
        "training": {
            "architecture": ARCHITECTURE,
            "seed": SEED,
            "float_epoch": FLOAT_EPOCH,
            "float_metrics": predecessor["selection"]["float_validation"],
            "float_parameters_immutable": True,
            "source_bundle_body_sha256": v1.SOURCE_BUNDLE_BODY,
            "source_bundle_manifest": dict(v1_plan["training"]["source_bundle_manifest"]),
            "safe_routes": dict(v1_plan["training"]["safe_routes"]),
            "safe_input_artifacts": dict(v1_plan["training"]["safe_input_artifacts"]),
            "roots_manifest": dict(v1_plan["training"]["roots_manifest"]),
            "roots_tsv": dict(v1_plan["training"]["roots_tsv"]),
            "prior_compact_runtime": dict(
                v1_plan["training"]["prior_compact_runtime"]
            ),
            "search_teacher_runtime": dict(
                v1_plan["training"]["search_teacher_runtime"]
            ),
            "new_train_manifest": dict(
                predecessor["training"]["new_train_manifest"]
            ),
            "new_train_npz": dict(predecessor["training"]["new_train_npz"]),
            "retained_train_rows": predecessor["training"]["retained_train_rows"],
            "split_isolation": dict(predecessor["training"]["split_isolation"]),
        },
        "quantization_roster": {
            "bits": 3,
            "minimum": -3,
            "maximum": 3,
            "base_scales": BASE_SCALES,
            "qat_learning_rates": list(QAT_LEARNING_RATES),
            "qat_epochs_per_rate": QAT_EPOCHS,
            "schedule_epochs": list(SCHEDULE_EPOCHS),
            "qat_epoch_contract": (
                "maintained-runtime-and-exporter-truthful-inclusive-range-0-through-4"
            ),
            "weight_decay": WEIGHT_DECAY,
            "adamw_reset_per_trajectory": True,
            "batching": {"new_rows": 64, "anchor_rows": 192},
            "calibration_passes": CALIBRATION_PASSES,
            "calibration_layer_order": ["w1", "w2", "w3"],
            "scale_factors": {
                name: list(values) for name, values in SCALE_FACTORS.items()
            },
            "fixed_candidates": FIXED_CANDIDATES,
            "calibration_candidates": CALIBRATION_CANDIDATES,
            "total_candidates": TOTAL_CANDIDATES,
            "selection_key": (
                "existing-gate-feasibility-key-then-stable-candidate-ordinal"
            ),
            "float_gate_denominator_immutable": True,
        },
        "fresh_protected_holdout": fresh,
        "source": {
            "plan": dict(v1_plan["source"]["plan"]),
        },
        "tools": tools,
        "policy": {
            "float_fine_tuning": False,
            "architecture_change": False,
            "new_training_games": 0,
            "new_training_labels": 0,
            "repack": False,
            "protected_tests_opened": False,
            "selection_attempts": 1,
            "v3_authorized": False,
            "upload_authorized": False,
        },
    }
    qualification.write_sealed(plan_path, body)
    training_input_path = output_root / "training-input.json"
    qualification.write_sealed(training_input_path, {
        "schema": TRAINING_INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": qualification.load_sealed(plan_path, PLAN_SCHEMA)[
            "body_sha256"
        ],
        "predecessor_training_input": qualification.artifact_reference(
            predecessor["paths"]["training_input"], v1.TRAINING_INPUT_SCHEMA
        ),
        "new_train_manifest": dict(predecessor["training"]["new_train_manifest"]),
        "new_train_npz": dict(predecessor["training"]["new_train_npz"]),
        "retained_train_rows": predecessor["training"]["retained_train_rows"],
        "split_isolation": dict(predecessor["training"]["split_isolation"]),
        "sole_new_data_manifest_for_training": True,
        "new_training_games_generated": 0,
        "new_training_labels_generated": 0,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": False,
        "eligible_for_quantization_only": True,
    })
    load_plan(plan_path, output_root=output_root)
    return plan_path


def load_plan(plan_path: pathlib.Path, *, output_root: pathlib.Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_path = output_root / "quantization-v2-plan.json"
    if (
        output_root != canonical_v2_root()
        or plan_path != expected_path
        or plan_path.is_symlink()
        or not plan_path.is_file()
    ):
        raise V2Error("v2 plan path/root is not canonical")
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    predecessor = _validate_v1(canonical_v1_root())
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
        raise V2Error("v2 plan authorization binding changed")
    authorization = qualification.load_sealed(
        authorization_path, AUTHORIZATION_SCHEMA
    )
    expected_authorization = qualification.seal(_authorization_body(
        predecessor, str(authorization.get("authorized_at_utc"))
    ))
    roster = plan.get("quantization_roster")
    if (
        authorization != expected_authorization
        or plan.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or plan.get("status")
        != "quantization-v2-planned-protected-tests-unmaterialized"
        or not isinstance(roster, Mapping)
        or roster.get("bits") != 3
        or roster.get("minimum") != -3
        or roster.get("maximum") != 3
        or roster.get("base_scales") != BASE_SCALES
        or roster.get("qat_learning_rates") != list(QAT_LEARNING_RATES)
        or roster.get("qat_epochs_per_rate") != QAT_EPOCHS
        or roster.get("schedule_epochs") != list(SCHEDULE_EPOCHS)
        or roster.get("qat_epoch_contract")
        != "maintained-runtime-and-exporter-truthful-inclusive-range-0-through-4"
        or roster.get("scale_factors")
        != {name: list(values) for name, values in SCALE_FACTORS.items()}
        or roster.get("total_candidates") != TOTAL_CANDIDATES
        or plan.get("policy") != {
            "float_fine_tuning": False,
            "architecture_change": False,
            "new_training_games": 0,
            "new_training_labels": 0,
            "repack": False,
            "protected_tests_opened": False,
            "selection_attempts": 1,
            "v3_authorized": False,
            "upload_authorized": False,
        }
    ):
        raise V2Error("v2 plan policy/roster changed")
    expected_predecessor = {
        "campaign_id": V1_CAMPAIGN_ID,
        "plan": qualification.artifact_reference(
            predecessor["paths"]["plan"], v1.PLAN_SCHEMA
        ),
        "training_input": qualification.artifact_reference(
            predecessor["paths"]["training_input"], v1.TRAINING_INPUT_SCHEMA
        ),
        "repack": qualification.artifact_reference(
            predecessor["paths"]["repack"], v1.REPACK_SCHEMA
        ),
        "rejection": qualification.artifact_reference(
            predecessor["paths"]["rejection"],
            "papersoccer.compact-value-bfm.clean-successor-offline-rejection.v1",
        ),
        "selection": qualification.artifact_reference(
            predecessor["paths"]["selection"], v1.SELECTION_SCHEMA
        ),
        "float_checkpoint": _record(predecessor["paths"]["float_checkpoint"]),
    }
    if plan.get("predecessor") != expected_predecessor:
        raise V2Error("v2 predecessor binding changed")
    v1_plan = predecessor["plan"]
    expected_training_fields = {
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "float_epoch": FLOAT_EPOCH,
        "float_metrics": predecessor["selection"]["float_validation"],
        "float_parameters_immutable": True,
        "source_bundle_body_sha256": v1.SOURCE_BUNDLE_BODY,
        "source_bundle_manifest": v1_plan["training"]["source_bundle_manifest"],
        "safe_routes": v1_plan["training"]["safe_routes"],
        "safe_input_artifacts": v1_plan["training"]["safe_input_artifacts"],
        "roots_manifest": v1_plan["training"]["roots_manifest"],
        "roots_tsv": v1_plan["training"]["roots_tsv"],
        "prior_compact_runtime": v1_plan["training"]["prior_compact_runtime"],
        "search_teacher_runtime": v1_plan["training"]["search_teacher_runtime"],
        "new_train_manifest": predecessor["training"]["new_train_manifest"],
        "new_train_npz": predecessor["training"]["new_train_npz"],
        "retained_train_rows": predecessor["training"]["retained_train_rows"],
        "split_isolation": predecessor["training"]["split_isolation"],
    }
    if any(
        plan.get("training", {}).get(field) != value
        for field, value in expected_training_fields.items()
    ):
        raise V2Error("v2 training input binding changed")
    expected_fresh = dict(v1_plan["fresh_protected_holdout"])
    expected_fresh.update({
        "campaign_id": f"{SUCCESSOR_CAMPAIGN_ID}-holdout",
        "game_plan_seed": 8_950_116_866_532_575_367,
        "materialized": False,
        "selection_may_change_after_results": False,
        "old_protected_routes_permitted": False,
        "tool": plan.get("tools", {}).get("quantization_v2_holdout"),
    })
    if plan.get("fresh_protected_holdout") != expected_fresh:
        raise V2Error("v2 fresh-holdout plan changed")
    if plan.get("source") != {"plan": dict(v1_plan["source"]["plan"])}:
        raise V2Error("v2 source-plan binding changed")
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise V2Error("v2 tool closure is missing")
    for name, record in v1_plan["tools"].items():
        if tools.get(name) != record:
            raise V2Error(f"v2 changed v1 tool {name}")
    expected_new_paths = {
        "quantization_v2": pathlib.Path(__file__).resolve(),
        "quantization_v2_holdout": (
            HERE / "compact_value_bfm_quantization_v2_holdout.py"
        ).resolve(),
    }
    for name, expected in expected_new_paths.items():
        record = tools.get(name)
        if (
            not isinstance(record, Mapping)
            or pathlib.Path(str(record.get("path", ""))) != expected
        ):
            raise V2Error(f"v2 redirects tool {name}")
    for name, record in tools.items():
        _verify_record(record, f"v2 tool {name}")
    training_input_path = output_root / "training-input.json"
    if training_input_path.is_symlink() or not training_input_path.is_file():
        raise V2Error("v2 training input is absent")
    training_input = qualification.load_sealed(
        training_input_path, TRAINING_INPUT_SCHEMA
    )
    if (
        training_input.get("plan_body_sha256") != plan["body_sha256"]
        or training_input.get("new_train_manifest", {}).get("sha256")
        != TRAIN_MANIFEST_SHA256
        or training_input.get("new_train_npz", {}).get("sha256") != TRAIN_NPZ_SHA256
        or training_input.get("eligible_for_quantization_only") is not True
        or training_input.get("new_training_games_generated") != 0
        or training_input.get("new_training_labels_generated") != 0
        or training_input.get("old_protected_tests_accessed") is not False
        or training_input.get("fresh_protected_tests_opened") is not False
    ):
        raise V2Error("v2 training-input receipt changed")
    return plan


def _require_blas() -> None:
    wrong = [name for name in BLAS_ENVIRONMENT if os.environ.get(name) != "1"]
    if wrong:
        raise V2Error("v2 requires all BLAS thread variables set to one: " + ", ".join(wrong))


def _candidate_key(
    float_metrics: Mapping[str, Mapping[str, float | int]],
    metrics: Mapping[str, Mapping[str, float | int]], ordinal: int,
) -> tuple[float, ...]:
    return (
        *iteration.gate_feasibility_key(compact, float_metrics, metrics),
        float(ordinal),
    )


def _candidate_record(
    *, ordinal: int, kind: str, lr: float | None, qat_epoch: int,
    scales: Mapping[str, float | np.floating[Any]],
    metrics: Mapping[str, Mapping[str, float | int]],
    float_metrics: Mapping[str, Mapping[str, float | int]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate = compact.offline_advancement_gate(float_metrics, metrics)
    return {
        "ordinal": ordinal,
        "kind": kind,
        "learning_rate": lr,
        "qat_epoch": qat_epoch,
        "scales": {name: float(np.float32(scales[name])) for name in ("w1", "w2", "w3")},
        "metrics": dict(metrics),
        "offline_gate": gate,
        "selection_key": list(_candidate_key(float_metrics, metrics, ordinal)),
        "metadata": dict(metadata or {}),
    }


@dataclass
class Candidate:
    record: dict[str, Any]
    quantized: Any
    master: dict[str, np.ndarray]


def _load_inputs(
    plan: Mapping[str, Any], output_root: pathlib.Path,
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    v1_root = canonical_v1_root()
    v1_plan = v1.load_plan(v1_root / "successor-plan.json", output_root=v1_root)
    v1_training, new_dataset = v1._load_training_input(v1_plan, v1_root)
    inputs = v1._base_inputs(v1_plan, new_dataset)
    if inputs.split_isolation != plan["training"]["split_isolation"]:
        raise V2Error("v2 split isolation changed")
    architecture = compact.ARCHITECTURES[ARCHITECTURE]
    parameters = compact.load_float_checkpoint(
        canonical_v1_root() / "training" / "float-checkpoints"
        / f"{V1_FLOAT_SHA256}.float.npz",
        architecture,
    )
    return inputs, parameters, v1_training


def run(*, plan_path: pathlib.Path, output_root: pathlib.Path) -> pathlib.Path:
    _require_blas()
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    _acquire_run_lock(output_root)
    reference_path = output_root / "selection-reference.json"
    existing = _selection_reference(reference_path, plan=plan, output_root=output_root)
    if existing is not None:
        _validate_outcome(plan=plan, output_root=output_root, selection_path=existing)
        return existing
    claim_path = output_root / "governance" / "01-execution-claim.json"
    expected_claim = {
        "schema": EXECUTION_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "quantization-v2-execution-claimed-once",
        "plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
        "exclusive_process_lock": str((output_root / "quantization-v2.lock").resolve()),
        "executions_authorized": 1,
        "float_fine_tuning": False,
        "new_training_games": 0,
        "new_training_labels": 0,
        "fresh_protected_holdout_may_run_only_after_pass": True,
        "protected_tests_opened": False,
        "v3_authorized": False,
    }
    if claim_path.exists():
        claim = qualification.load_sealed(claim_path, EXECUTION_CLAIM_SCHEMA)
        if any(claim.get(field) != value for field, value in expected_claim.items()):
            raise V2Error("v2 execution claim changed")
    else:
        claim = qualification.write_sealed(claim_path, {
            **expected_claim,
            "claimed_at_utc": utc_now(),
        })
    inputs, parameters, _v1_training = _load_inputs(plan, output_root)
    architecture = compact.ARCHITECTURES[ARCHITECTURE]
    arm = compact.ARMS["search-target"]
    float_metrics = plan["training"]["float_metrics"]
    base_scales = {name: np.float32(BASE_SCALES[name]) for name in BASE_SCALES}
    candidates: list[Candidate] = []
    ordinal = 0
    pre = compact.quantize_fixed(parameters, architecture, base_scales)
    pre_metrics = compact.evaluate_validation_pair(
        parameters, architecture, inputs, arm, quantized=pre
    )
    pre_record = _candidate_record(
        ordinal=ordinal, kind="pre-qat", lr=None, qat_epoch=0,
        scales=pre.scales, metrics=pre_metrics, float_metrics=float_metrics,
    )
    candidates.append(Candidate(
        record=pre_record,
        quantized=pre,
        master={name: value.copy() for name, value in parameters.items()},
    ))
    ordinal += 1
    trajectory_best: list[Candidate] = []
    trajectory_receipts = []
    for lr_index, learning_rate in enumerate(QAT_LEARNING_RATES):
        master = {name: value.copy() for name, value in parameters.items()}
        optimizer = compact.AdamW(
            master, learning_rate=learning_rate, weight_decay=WEIGHT_DECAY
        )
        trajectory: list[Candidate] = []
        for qat_epoch, schedule_epoch in enumerate(SCHEDULE_EPOCHS, 1):
            losses = []
            for new_rows, anchor_rows in compact.mixed_epoch_batches(
                len(inputs.new), len(inputs.anchor), seed=SEED, epoch=schedule_epoch
            ):
                losses.append(compact._train_mixed_batch(
                    master, architecture, arm, optimizer, inputs,
                    new_rows, anchor_rows, fixed_scales=base_scales,
                ))
            quantized = compact.quantize_fixed(master, architecture, base_scales)
            metrics = compact.evaluate_validation_pair(
                master, architecture, inputs, arm, quantized=quantized
            )
            record = _candidate_record(
                ordinal=ordinal, kind="fixed-scale-qat", lr=learning_rate,
                qat_epoch=qat_epoch, scales=quantized.scales,
                metrics=metrics, float_metrics=float_metrics,
                metadata={
                    "lr_index": lr_index,
                    "schedule_epoch": schedule_epoch,
                    "training_loss_mean": float(np.mean(losses)),
                },
            )
            candidate = Candidate(
                record=record,
                quantized=quantized,
                master={name: value.copy() for name, value in master.items()},
            )
            candidates.append(candidate)
            trajectory.append(candidate)
            ordinal += 1
        best = min(trajectory, key=lambda item: tuple(item.record["selection_key"]))
        trajectory_best.append(best)
        receipt = qualification.write_sealed(
            output_root / "trajectories" / f"lr-{lr_index:02d}.json",
            {
                "schema": TRAJECTORY_SCHEMA,
                "namespace": NAMESPACE,
                "campaign_id": SUCCESSOR_CAMPAIGN_ID,
                "plan_body_sha256": plan["body_sha256"],
                "execution_claim": qualification.artifact_reference(
                    claim_path, EXECUTION_CLAIM_SCHEMA
                ),
                "lr_index": lr_index,
                "learning_rate": learning_rate,
                "candidates": [item.record for item in trajectory],
                "selected_ordinal_for_calibration": best.record["ordinal"],
                "protected_tests_opened": False,
            },
        )
        trajectory_receipts.append(receipt)
    calibration_receipts = []
    for trajectory_index, source_candidate in enumerate(trajectory_best):
        requested = dict(base_scales)
        trials = []
        master = source_candidate.master
        for search_pass in range(1, CALIBRATION_PASSES + 1):
            for layer in ("w1", "w2", "w3"):
                layer_trials: list[Candidate] = []
                for factor in SCALE_FACTORS[layer]:
                    trial_scales = dict(requested)
                    trial_scales[layer] = np.float32(BASE_SCALES[layer] * factor)
                    quantized = compact.quantize_fixed(
                        master, architecture, trial_scales
                    )
                    metrics = compact.evaluate_validation_pair(
                        master, architecture, inputs, arm, quantized=quantized
                    )
                    record = _candidate_record(
                        ordinal=ordinal, kind="coordinate-scale-calibration",
                        lr=float(source_candidate.record["learning_rate"]),
                        qat_epoch=int(source_candidate.record["qat_epoch"]),
                        scales=quantized.scales,
                        metrics=metrics, float_metrics=float_metrics,
                        metadata={
                            "trajectory_index": trajectory_index,
                            "source_qat_ordinal": source_candidate.record["ordinal"],
                            "pass": search_pass,
                            "layer": layer,
                            "factor": factor,
                        },
                    )
                    candidate = Candidate(
                        record=record, quantized=quantized,
                        master={name: value.copy() for name, value in master.items()},
                    )
                    candidates.append(candidate)
                    layer_trials.append(candidate)
                    trials.append(record)
                    ordinal += 1
                best_layer = min(
                    layer_trials, key=lambda item: tuple(item.record["selection_key"])
                )
                requested = {
                    name: np.float32(best_layer.record["scales"][name])
                    for name in ("w1", "w2", "w3")
                }
        receipt = qualification.write_sealed(
            output_root / "calibration" / f"trajectory-{trajectory_index:02d}.json",
            {
                "schema": CALIBRATION_SCHEMA,
                "namespace": NAMESPACE,
                "campaign_id": SUCCESSOR_CAMPAIGN_ID,
                "plan_body_sha256": plan["body_sha256"],
                "source_candidate": source_candidate.record,
                "trials": trials,
                "trial_count": len(trials),
                "protected_tests_opened": False,
            },
        )
        calibration_receipts.append(receipt)
    if ordinal != TOTAL_CANDIDATES or len(candidates) != TOTAL_CANDIDATES:
        raise V2Error("v2 candidate roster count changed")
    candidate_index_path = output_root / "candidate-index.json"
    qualification.write_sealed(candidate_index_path, {
        "schema": CANDIDATE_INDEX_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "execution_claim": qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        ),
        "candidates": [candidate.record for candidate in candidates],
        "candidate_count": len(candidates),
        "protected_tests_opened": False,
    })
    selected = min(candidates, key=lambda item: tuple(item.record["selection_key"]))
    gate = selected.record["offline_gate"]
    parity = {
        **compact.assert_quantized_inference_parity(
            selected.quantized, architecture, inputs.common_adjudicator,
            maximum_rows=4_096,
        ),
        "passed": True,
    }
    runtime = compact.write_runtime(
        output_root / "training" / "quantized-runtimes",
        architecture, selected.quantized,
        arm="search-target", seed=SEED, float_epoch=FLOAT_EPOCH,
        qat_epoch=int(selected.record["qat_epoch"]),
        source_bundle_body_sha256=v1.SOURCE_BUNDLE_BODY,
    )
    generated_source, source_export = iteration.render_iteration_source(
        runtime=runtime, plan=plan, output_root=output_root
    )
    body = {
        "schema": SELECTION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": gate["status"],
        "plan_body_sha256": plan["body_sha256"],
        "execution_claim": qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        ),
        "predecessor_float_checkpoint": dict(plan["predecessor"]["float_checkpoint"]),
        "training_input": qualification.artifact_reference(
            output_root / "training-input.json", TRAINING_INPUT_SCHEMA
        ),
        "architecture": ARCHITECTURE,
        "arm": "search-target",
        "seed": SEED,
        "float_epoch": FLOAT_EPOCH,
        "qat_learning_rate": selected.record["learning_rate"],
        "qat_epoch": selected.record["qat_epoch"],
        "candidate_ordinal": selected.record["ordinal"],
        "candidate_kind": selected.record["kind"],
        "scales": selected.record["scales"],
        "float_validation": float_metrics,
        "quantized_validation": selected.record["metrics"],
        "offline_gate": gate,
        "selection_key": selected.record["selection_key"],
        "candidate_counts": {
            "fixed": FIXED_CANDIDATES,
            "calibration": CALIBRATION_CANDIDATES,
            "total": TOTAL_CANDIDATES,
        },
        "candidate_index": qualification.artifact_reference(
            candidate_index_path, CANDIDATE_INDEX_SCHEMA
        ),
        "trajectory_receipts": [
            qualification.artifact_reference(
                output_root / "trajectories" / f"lr-{index:02d}.json",
                TRAJECTORY_SCHEMA,
            )
            for index in range(len(trajectory_receipts))
        ],
        "calibration_receipts": [
            qualification.artifact_reference(
                output_root / "calibration" / f"trajectory-{index:02d}.json",
                CALIBRATION_SCHEMA,
            )
            for index in range(len(calibration_receipts))
        ],
        "runtime": _record(runtime),
        "generated_source": _record(generated_source),
        "source_export": source_export,
        "inference_parity": parity,
        "float_parameters_changed": False,
        "new_training_games_generated": 0,
        "new_training_labels_generated": 0,
        "selection_immutable": True,
        "selection_may_change_after_fresh_protected_tests": False,
        "old_protected_tests_accessed": False,
        "old_protected_tests_permanently_excluded": True,
        "fresh_protected_tests_opened": False,
        "fresh_protected_tests_authorized": gate["passed"],
        "game_gated": False,
        "upload_authorized": False,
        "v3_authorized": False,
    }
    selection_path, _selection = _write_content_addressed(
        output_root / "selections", body, ".quantization-v2-selection.json"
    )
    outcome_path = output_root / "governance" / "02-outcome.json"
    qualification.write_sealed(outcome_path, {
        "schema": OUTCOME_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": (
            "quantization-v2-offline-qualified-awaiting-fresh-tests"
            if gate["passed"]
            else "quantization-v2-terminal-offline-rejection"
        ),
        "selection": qualification.artifact_reference(
            selection_path, SELECTION_SCHEMA
        ),
        "offline_gate": gate,
        "fresh_protected_tests_authorized": gate["passed"],
        "fresh_protected_tests_opened": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
        "v3_authorized": False,
    })
    qualification.write_sealed(reference_path, {
        "schema": SELECTION_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "selection": _record(selection_path),
        "offline_gate_passed": gate["passed"],
        "fresh_protected_tests_opened": False,
        "selection_immutable": True,
        "outcome": qualification.artifact_reference(outcome_path, OUTCOME_SCHEMA),
    })
    validated = _selection_reference(
        reference_path, plan=plan, output_root=output_root
    )
    if validated != selection_path:
        raise V2Error("v2 selection failed immediate validation")
    return selection_path


def _validate_selection_closure(
    selection_path: pathlib.Path, *, plan: Mapping[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    expected_directory = output_root.resolve() / "selections"
    if (
        selection_path.parent != expected_directory
        or not selection_path.name.endswith(".quantization-v2-selection.json")
        or selection_path.is_symlink()
        or not selection_path.is_file()
        or _sha256_file(selection_path)
        != selection_path.name.removesuffix(".quantization-v2-selection.json")
    ):
        raise V2Error("v2 selection path/content address changed")
    selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
    gate = selection.get("offline_gate")
    if (
        selection.get("namespace") != NAMESPACE
        or selection.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or selection.get("plan_body_sha256") != plan["body_sha256"]
        or selection.get("architecture") != ARCHITECTURE
        or selection.get("arm") != "search-target"
        or selection.get("seed") != SEED
        or selection.get("float_epoch") != FLOAT_EPOCH
        or selection.get("predecessor_float_checkpoint")
        != plan["predecessor"]["float_checkpoint"]
        or selection.get("training_input") != qualification.artifact_reference(
            output_root / "training-input.json", TRAINING_INPUT_SCHEMA
        )
        or selection.get("float_validation") != plan["training"]["float_metrics"]
        or not isinstance(gate, Mapping)
        or not isinstance(gate.get("passed"), bool)
        or selection.get("status") != gate.get("status")
        or selection.get("candidate_counts") != {
            "fixed": FIXED_CANDIDATES,
            "calibration": CALIBRATION_CANDIDATES,
            "total": TOTAL_CANDIDATES,
        }
        or selection.get("candidate_index") != qualification.artifact_reference(
            output_root / "candidate-index.json", CANDIDATE_INDEX_SCHEMA
        )
        or selection.get("selection_immutable") is not True
        or selection.get("float_parameters_changed") is not False
        or selection.get("new_training_games_generated") != 0
        or selection.get("new_training_labels_generated") != 0
        or selection.get("old_protected_tests_accessed") is not False
        or selection.get("old_protected_tests_permanently_excluded") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("fresh_protected_tests_authorized") is not gate.get("passed")
        or selection.get("game_gated") is not False
        or selection.get("upload_authorized") is not False
        or selection.get("v3_authorized") is not False
    ):
        raise V2Error("v2 selection policy changed")
    recomputed_gate = compact.offline_advancement_gate(
        plan["training"]["float_metrics"], selection["quantized_validation"]
    )
    if gate != recomputed_gate:
        raise V2Error("v2 selection offline gate was not recomputed exactly")
    claim_path = output_root / "governance" / "01-execution-claim.json"
    if claim_path.is_symlink() or not claim_path.is_file():
        raise V2Error("v2 execution claim is absent")
    claim = qualification.load_sealed(claim_path, EXECUTION_CLAIM_SCHEMA)
    if (
        selection.get("execution_claim") != qualification.artifact_reference(
            claim_path, EXECUTION_CLAIM_SCHEMA
        )
        or claim.get("plan") != qualification.artifact_reference(
            output_root / "quantization-v2-plan.json", PLAN_SCHEMA
        )
        or claim.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or claim.get("status") != "quantization-v2-execution-claimed-once"
        or claim.get("exclusive_process_lock")
        != str((output_root / "quantization-v2.lock").resolve())
        or claim.get("executions_authorized") != 1
        or claim.get("float_fine_tuning") is not False
        or claim.get("new_training_games") != 0
        or claim.get("new_training_labels") != 0
        or claim.get("fresh_protected_holdout_may_run_only_after_pass") is not True
        or claim.get("protected_tests_opened") is not False
        or claim.get("v3_authorized") is not False
    ):
        raise V2Error("v2 execution claim/selection binding changed")
    candidate_index_path = output_root / "candidate-index.json"
    if candidate_index_path.is_symlink() or not candidate_index_path.is_file():
        raise V2Error("v2 candidate index is not a regular file")
    candidate_index = qualification.load_sealed(
        candidate_index_path, CANDIDATE_INDEX_SCHEMA
    )
    candidate_rows = candidate_index.get("candidates")
    if (
        selection.get("candidate_index") != qualification.artifact_reference(
            candidate_index_path, CANDIDATE_INDEX_SCHEMA
        )
        or candidate_index.get("plan_body_sha256") != plan["body_sha256"]
        or candidate_index.get("execution_claim")
        != qualification.artifact_reference(claim_path, EXECUTION_CLAIM_SCHEMA)
        or candidate_index.get("candidate_count") != TOTAL_CANDIDATES
        or not isinstance(candidate_rows, list)
        or len(candidate_rows) != TOTAL_CANDIDATES
        or [row.get("ordinal") for row in candidate_rows if isinstance(row, Mapping)]
        != list(range(TOTAL_CANDIDATES))
        or candidate_index.get("protected_tests_opened") is not False
    ):
        raise V2Error("v2 candidate index changed")
    for ordinal, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise V2Error("v2 candidate index contains a malformed row")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise V2Error("v2 candidate metrics are missing")
        expected_gate = compact.offline_advancement_gate(
            plan["training"]["float_metrics"], metrics
        )
        expected_key = list(_candidate_key(
            plan["training"]["float_metrics"], metrics, ordinal
        ))
        if (
            row.get("offline_gate") != expected_gate
            or row.get("selection_key") != expected_key
        ):
            raise V2Error("v2 candidate gate/key changed")
    expected_selected = min(
        candidate_rows, key=lambda row: tuple(row["selection_key"])
    )
    if expected_selected.get("ordinal") != selection.get("candidate_ordinal"):
        raise V2Error("v2 selection is not the minimum precommitted candidate")
    selected_rows = [
        row for row in candidate_rows
        if row.get("ordinal") == selection.get("candidate_ordinal")
    ]
    if len(selected_rows) != 1:
        raise V2Error("v2 selected candidate is absent from its index")
    selected_row = selected_rows[0]
    if (
        selection.get("candidate_kind") != selected_row.get("kind")
        or selection.get("qat_learning_rate") != selected_row.get("learning_rate")
        or selection.get("qat_epoch") != selected_row.get("qat_epoch")
        or selection.get("scales") != selected_row.get("scales")
        or selection.get("quantized_validation") != selected_row.get("metrics")
        or selection.get("offline_gate") != selected_row.get("offline_gate")
        or selection.get("selection_key") != selected_row.get("selection_key")
    ):
        raise V2Error("v2 selected candidate/index disagree")
    for field, directory, schema in (
        ("trajectory_receipts", output_root / "trajectories", TRAJECTORY_SCHEMA),
        ("calibration_receipts", output_root / "calibration", CALIBRATION_SCHEMA),
    ):
        records = selection.get(field)
        if not isinstance(records, list) or len(records) != 3:
            raise V2Error(f"v2 {field} roster changed")
        for index, record in enumerate(records):
            expected = directory / (
                f"lr-{index:02d}.json"
                if field == "trajectory_receipts"
                else f"trajectory-{index:02d}.json"
            )
            if (
                not isinstance(record, Mapping)
                or pathlib.Path(str(record.get("path", ""))) != expected
                or expected.is_symlink()
                or not expected.is_file()
                or qualification.artifact_reference(expected, schema) != record
            ):
                raise V2Error(f"v2 {field} binding changed")
    runtime_record = selection.get("runtime")
    source_record = selection.get("generated_source")
    if not isinstance(runtime_record, Mapping) or not isinstance(source_record, Mapping):
        raise V2Error("v2 selection deployment artifacts are missing")
    runtime_path = pathlib.Path(str(runtime_record.get("path", "")))
    source_path = pathlib.Path(str(source_record.get("path", "")))
    if (
        runtime_path.parent != output_root / "training" / "quantized-runtimes"
        or source_path.parent != output_root / "fine-tune" / "generated-sources"
        or _verify_record(runtime_record, "v2 runtime") != runtime_path
        or _verify_record(source_record, "v2 source") != source_path
    ):
        raise V2Error("v2 selection deployment path changed")
    architecture, _quantized, runtime_selection, _runtime = compact.load_runtime(
        runtime_path
    )
    if (
        architecture.name != ARCHITECTURE
        or runtime_selection.get("arm") != "search-target"
        or runtime_selection.get("seed") != SEED
        or runtime_selection.get("float_epoch") != FLOAT_EPOCH
        or runtime_selection.get("qat_epoch") != selection.get("qat_epoch")
        or runtime_selection.get("source_bundle_body_sha256") != v1.SOURCE_BUNDLE_BODY
        or selection.get("source_export", {}).get("runtime_sha256")
        != runtime_record.get("sha256")
        or selection.get("source_export", {}).get("source_sha256")
        != source_record.get("sha256")
        or not 0 < int(source_record.get("bytes", 0)) < 95_000
        or selection.get("inference_parity", {}).get("passed") is not True
    ):
        raise V2Error("v2 runtime/source closure changed")
    return selection


def _selection_reference(
    path: pathlib.Path, *, plan: Mapping[str, Any], output_root: pathlib.Path,
) -> pathlib.Path | None:
    if not path.exists():
        return None
    if path != output_root.resolve() / "selection-reference.json" or path.is_symlink():
        raise V2Error("v2 selection reference path changed")
    outcome_path = output_root / "governance" / "02-outcome.json"
    if outcome_path.is_symlink() or not outcome_path.is_file():
        raise V2Error("v2 selection reference lacks a regular outcome")
    reference = qualification.load_sealed(path, SELECTION_REFERENCE_SCHEMA)
    record = reference.get("selection")
    if (
        reference.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or reference.get("plan_body_sha256") != plan["body_sha256"]
        or reference.get("selection_immutable") is not True
        or reference.get("fresh_protected_tests_opened") is not False
        or reference.get("outcome") != qualification.artifact_reference(
            outcome_path, OUTCOME_SCHEMA
        )
        or not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
    ):
        raise V2Error("v2 selection reference changed")
    selection_path = pathlib.Path(record["path"])
    if selection_path.parent != output_root / "selections":
        raise V2Error("v2 selection reference redirects selection")
    _verify_record(record, "v2 selection")
    selection = _validate_selection_closure(
        selection_path, plan=plan, output_root=output_root
    )
    if reference.get("offline_gate_passed") is not selection["offline_gate"]["passed"]:
        raise V2Error("v2 selection reference/gate disagree")
    _validate_outcome(
        plan=plan, output_root=output_root, selection_path=selection_path
    )
    return selection_path


def _validate_outcome(
    *, plan: Mapping[str, Any], output_root: pathlib.Path,
    selection_path: pathlib.Path,
) -> dict[str, Any]:
    path = output_root / "governance" / "02-outcome.json"
    if path.is_symlink() or not path.is_file():
        raise V2Error("v2 terminal outcome is absent")
    outcome = qualification.load_sealed(path, OUTCOME_SCHEMA)
    selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
    gate = selection["offline_gate"]
    expected_status = (
        "quantization-v2-offline-qualified-awaiting-fresh-tests"
        if gate["passed"]
        else "quantization-v2-terminal-offline-rejection"
    )
    if (
        outcome.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or outcome.get("status") != expected_status
        or outcome.get("selection") != qualification.artifact_reference(
            selection_path, SELECTION_SCHEMA
        )
        or outcome.get("offline_gate") != gate
        or outcome.get("fresh_protected_tests_authorized") is not gate["passed"]
        or outcome.get("fresh_protected_tests_opened") is not False
        or outcome.get("rank4_gate_authorized") is not False
        or outcome.get("upload_authorized") is not False
        or outcome.get("v3_authorized") is not False
        or plan.get("policy", {}).get("v3_authorized") is not False
    ):
        raise V2Error("v2 terminal outcome changed")
    return outcome


def verify(*, plan_path: pathlib.Path, output_root: pathlib.Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    selection_path = _selection_reference(
        output_root / "selection-reference.json",
        plan=plan,
        output_root=output_root,
    )
    result: dict[str, Any] = {
        "schema": "papersoccer.compact-value-bfm.quantization-v2-status.v1",
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "plan_valid": True,
        "selection_complete": selection_path is not None,
        "protected_tests_opened": False,
    }
    if selection_path is not None:
        selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
        _validate_outcome(
            plan=plan, output_root=output_root, selection_path=selection_path
        )
        result.update({
            "selection": _record(selection_path),
            "offline_gate": selection["offline_gate"],
            "candidate_ordinal": selection["candidate_ordinal"],
            "candidate_counts": selection["candidate_counts"],
        })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--v1-root", type=pathlib.Path, required=True)
    prepare_command.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare_command.add_argument("--holdout-tool", type=pathlib.Path, required=True)
    prepare_command.add_argument("--authorized-at-utc", default=utc_now())
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare(
                v1_root=args.v1_root,
                output_root=args.output_root,
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
        V2Error, compact.TrainingError, iteration.IterationError,
        OSError, ValueError, json.JSONDecodeError,
    ) as error:
        print(f"compact quantization-v2 failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
