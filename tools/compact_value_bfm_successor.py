#!/usr/bin/env python3
"""Clean successor for the sealed Compact Value-BFM on-policy corpus.

This tool never generates another training game or label.  It imports only the
ten worker outputs that were sealed before the protected-test incident,
globally repacks them against the unprotected validation sets, and performs the
already-fixed fine-tune/QAT/export selection.  Old protected-test routes are
never materialized by this tool; a separate, plan-bound tool owns fresh tests
after immutable selection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load successor dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_successor_qualification"
)
compact = _load(HERE / "compact_value_bfm_train.py", "compact_successor_training")
iteration = _load(
    HERE / "compact_value_bfm_iteration.py", "compact_successor_iteration"
)
selfsearch = _load(
    HERE / "jacek_selfsearch_workflow.py", "compact_successor_selfsearch"
)
large_training = _load(
    HERE / "jacek_replay_train.py", "compact_successor_large_training"
)


SuccessorError = qualification.QualificationError
NAMESPACE = "compact_value_bfm"
SOURCE_CAMPAIGN_ID = "compact-value-bfm-20260831-v1"
SUCCESSOR_CAMPAIGN_ID = "compact-value-bfm-successor-20260901-v1"
SOURCE_ITERATION_PLAN_BODY = (
    "134dd286b914e3720e3bca826d0df838cb3398d2835707a863b8e6c694e88484"
)
SOURCE_ITERATION_PLAN_FILE = (
    "4762f718342790a6033e33d551fb49ece468da3e44dc34cfa1ec6471f166a7e1"
)
SOURCE_CLAIM_BODY = (
    "742e95fcec676fc65c162d4a372b30726444a667326c598e6f0fe7ae6332e34e"
)
SOURCE_INCIDENT_BODY = (
    "5152ebe421ebc3e8e7d996b55ae0d33e70a74b00cd428cb896f088e6ca02154e"
)
SOURCE_INTEGRITY_BODY = (
    "ae20372c0d8a75a3e1c7c894b10e58941faa87785110b530b9690e2cc56f25ed"
)
SOURCE_INVALIDATED_BODY = (
    "16bd30bcbac352e6d5b552082c21e04572bf651e195ae71537b3377137eb2ce2"
)
SOURCE_BUNDLE_BODY = (
    "56b9c1e6dd75e49298f677b73da6e8e4890f618c8d0f5daa252ec1248cbecd3a"
)
SOURCE_BUNDLE_FILE = (
    "58e4d8ca648e52d2df31d27f13faa805d45e7c4e0c4b87f43b146118b768c742"
)

AUTHORIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.clean-successor-authorization.v1"
)
PLAN_SCHEMA = "papersoccer.compact-value-bfm.clean-successor-plan.v1"
CARRY_SCHEMA = "papersoccer.compact-value-bfm.clean-successor-carry-forward.v1"
REPACK_SCHEMA = "papersoccer.compact-value-bfm.clean-successor-repack.v1"
TRAINING_INPUT_SCHEMA = (
    "papersoccer.compact-value-bfm.clean-successor-training-input.v1"
)
SELECTION_SCHEMA = "papersoccer.compact-value-bfm.clean-successor-selection.v1"
SELECTION_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.clean-successor-selection-reference.v1"
)

SOURCE_PLAN_SCHEMA = "papersoccer.compact-value-bfm.iteration-plan.v1"
SOURCE_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.iteration-execution-claim.v1"
)
SOURCE_WORKER_SCHEMA = (
    "papersoccer.compact-value-bfm.iteration-worker-result.v1"
)
SOURCE_INTEGRITY_SCHEMA = (
    "papersoccer.compact-value-bfm.iteration-integrity-failure.v1"
)
SOURCE_INVALIDATED_SCHEMA = (
    "papersoccer.compact-value-bfm.precompletion-family-exhausted.v1"
)
SOURCE_STAGE_SCHEMA = "papersoccer.jacek-replay-bfm-stage-receipt.v1"

WORKERS = 10
GAMES = 10_000
POSITIONS = 200_000
DEEP_POSITIONS = 50_000
ROWS_PER_WORKER = 20_000
ARCHITECTURE = "capacity-12x8"
SEED = 20260909
LEARNING_RATE = 0.00006

PROTECTED_ROUTE_KEYS = (
    "pilot_search_manifests",
    "full_search_manifests",
    "pilot_rank4_manifests",
    "full_rank4_manifests",
)
BLAS_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def canonical_source_root() -> pathlib.Path:
    return (
        REPOSITORY / "results" / "compact_value_bfm" / SOURCE_CAMPAIGN_ID
    ).resolve()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SuccessorError(f"{label} is not a regular non-symlink file")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SuccessorError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SuccessorError(f"{label} is not a JSON object")
    return value


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_record(
    path: pathlib.Path, *, expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SuccessorError(f"required input is not a regular file: {path}")
    size = path.stat().st_size
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SuccessorError(f"required input hash changed: {path}")
    if expected_bytes is not None and size != expected_bytes:
        raise SuccessorError(f"required input byte count changed: {path}")
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest}


def _verify_record(record: object, label: str) -> pathlib.Path:
    if (
        not isinstance(record, Mapping)
        or not {"path", "bytes", "sha256"}.issubset(record)
        or not isinstance(record.get("path"), str)
        or type(record.get("bytes")) is not int
        or not isinstance(record.get("sha256"), str)
    ):
        raise SuccessorError(f"{label} record is malformed")
    path = pathlib.Path(record["path"])
    if path.is_symlink():
        if (
            record.get("executable") is not True
            or not isinstance(record.get("resolved_path"), str)
            or str(path.resolve()) != record["resolved_path"]
        ):
            raise SuccessorError(f"{label} is an untrusted symlink")
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256_file(path) != str(record["sha256"])
    ):
        raise SuccessorError(f"{label} changed")
    if (
        isinstance(record.get("resolved_path"), str)
        and str(path.resolve()) != record["resolved_path"]
    ):
        raise SuccessorError(f"{label} resolved path changed")
    return path


def _declared_record(record: object, label: str) -> pathlib.Path:
    if (
        not isinstance(record, Mapping)
        or not {"path", "bytes", "sha256"}.issubset(record)
        or not isinstance(record.get("path"), str)
        or type(record.get("bytes")) is not int
        or int(record["bytes"]) < 0
        or not isinstance(record.get("sha256"), str)
        or qualification.SHA256_RE.fullmatch(str(record["sha256"])) is None
    ):
        raise SuccessorError(f"{label} declaration is malformed")
    return pathlib.Path(str(record["path"]))


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    artifact = qualification.seal(body)
    payload = qualification.canonical_json_bytes(artifact)
    digest = qualification.sha256_bytes(payload)
    path = directory / f"{digest}{suffix}"
    qualification.atomic_write_once(path, payload)
    return path, artifact


def _source_paths(source_root: pathlib.Path) -> dict[str, pathlib.Path]:
    iteration_root = source_root / "on-policy-iteration"
    governance = source_root / "iteration-governance"
    return {
        "plan": iteration_root / "iteration-plan.json",
        "claim": iteration_root / "execution-claim.json",
        "integrity": governance / "iteration" / "02-integrity-failure.json",
        "invalidated": source_root / "compact-value-family-invalidated.json",
        "bundle": source_root / "input-bundle" / "bundle-manifest.json",
        "disabled_plist": (
            iteration_root / "launchagent" / "installed-plist.disabled"
        ),
    }


def _protected_routes(manifest: Mapping[str, Any]) -> set[str]:
    routes = manifest.get("routes")
    if not isinstance(routes, Mapping):
        raise SuccessorError("source bundle routes are malformed")
    protected: set[str] = set()
    for key in PROTECTED_ROUTE_KEYS:
        values = routes.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise SuccessorError(f"source bundle route {key} is malformed")
        if not isinstance(values[2], str):
            raise SuccessorError(f"source bundle protected route {key} is malformed")
        protected.add(values[2])
    canonical = routes.get("canonical_splits")
    if not isinstance(canonical, Mapping):
        raise SuccessorError("source canonical routes are malformed")
    tests = canonical.get("test")
    if not isinstance(tests, list) or len(tests) != 3 or not all(
        isinstance(value, str) for value in tests
    ):
        raise SuccessorError("source canonical protected routes are malformed")
    protected.update(tests)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SuccessorError("source bundle artifact registry is malformed")
    role_to_relative: dict[str, str] = {}
    relative_to_role: dict[str, str] = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise SuccessorError("source bundle artifact record is malformed")
        role, relative = record.get("role"), record.get("relative_path")
        if not isinstance(role, str) or not isinstance(relative, str):
            raise SuccessorError("source bundle artifact identity is malformed")
        role_to_relative[role] = relative
        relative_to_role[relative] = role
        if "-test-" in role or role.endswith("-test"):
            protected.add(relative)
    for manifest_route in tuple(protected):
        role = relative_to_role.get(manifest_route)
        if role is not None and role.endswith("-manifest"):
            npz_route = role_to_relative.get(role.removesuffix("-manifest") + "-npz")
            if npz_route is None:
                raise SuccessorError("protected manifest lacks a registered NPZ")
            protected.add(npz_route)
    return protected


def _safe_base_routes(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    routes = manifest.get("routes")
    if not isinstance(routes, Mapping):
        raise SuccessorError("source bundle routes are malformed")
    canonical = routes.get("canonical_splits")
    if not isinstance(canonical, Mapping):
        raise SuccessorError("source canonical routes are malformed")
    result = {
        "anchor": list(canonical.get("train", [])),
        "canonical_validation": list(canonical.get("validation", [])),
        "common_adjudicator": [routes.get("common_adjudicator_manifest")],
    }
    protected = _protected_routes(manifest)
    flattened = [value for values in result.values() for value in values]
    if (
        len(result["anchor"]) != 3
        or len(result["canonical_validation"]) != 3
        or len(result["common_adjudicator"]) != 1
        or not all(isinstance(value, str) and value for value in flattened)
        or any(value in protected for value in flattened)
    ):
        raise SuccessorError("successor base route allowlist is unsafe")
    return result


def retired_protected_paths(bundle_manifest_path: pathlib.Path) -> set[pathlib.Path]:
    manifest = _canonical_json(bundle_manifest_path, "source bundle manifest")
    return {
        bundle_manifest_path.parent / relative
        for relative in _protected_routes(manifest)
    }


def _bundle_artifact_records(
    bundle: Any, routes: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for role, relative_routes in routes.items():
        records = []
        for relative in relative_routes:
            protected = _protected_routes(bundle.manifest)
            if bundle.is_protected(relative) or relative in protected:
                raise SuccessorError("protected source route entered successor allowlist")
            manifest_path = bundle.artifact_path(relative)
            manifest = _canonical_json(manifest_path, f"{role} manifest")
            npz_name = manifest.get("npz")
            if not isinstance(npz_name, str) or pathlib.PurePath(npz_name).name != npz_name:
                raise SuccessorError(f"{role} manifest has an invalid NPZ route")
            npz_relative = (
                pathlib.PurePosixPath(relative).parent / npz_name
            ).as_posix()
            if bundle.is_protected(npz_relative) or npz_relative in protected:
                raise SuccessorError("protected source NPZ entered successor allowlist")
            npz_path = bundle.artifact_path(npz_relative)
            records.append({
                "route": relative,
                "manifest": _regular_file_record(manifest_path),
                "npz": _regular_file_record(npz_path),
                "split": manifest.get("split"),
                "samples": manifest.get("samples"),
            })
        output[role] = records
    return output


def _stage_receipt(
    source_root: pathlib.Path, worker_index: int, *,
    ordinal: int, stage: str, filename: str,
) -> tuple[pathlib.Path, dict[str, Any], dict[str, Any]]:
    worker_root = (
        source_root / "on-policy-iteration" / "workers"
        / f"worker-{worker_index:02d}-pipeline"
    )
    receipt_path = worker_root / "receipts" / f"{ordinal:02d}-{stage}.json"
    receipt = _canonical_json(receipt_path, f"{stage} merged-label stage receipt")
    if (
        receipt.get("schema") != SOURCE_STAGE_SCHEMA
        or receipt.get("stage") != stage
        or receipt.get("ordinal") != ordinal
        or receipt.get("result") != {"rows": ROWS_PER_WORKER}
        or receipt.get("environment", {}).get("worker_index") != worker_index
        or receipt.get("environment", {}).get("plan_body_sha256")
        != SOURCE_ITERATION_PLAN_BODY
    ):
        raise SuccessorError(f"worker {worker_index} {stage} receipt changed")
    label_record = receipt.get("outputs", {}).get("labels")
    expected_path = worker_root / "labels" / filename
    if (
        not isinstance(label_record, Mapping)
        or label_record.get("path") != str(expected_path.resolve())
        or label_record.get("kind") != "file"
        or label_record.get("lines") != ROWS_PER_WORKER
        or type(label_record.get("bytes")) is not int
        or not isinstance(label_record.get("sha256"), str)
    ):
        raise SuccessorError(f"worker {worker_index} {stage} label binding changed")
    labels = {
        "path": str(expected_path.resolve()),
        "bytes": int(label_record["bytes"]),
        "sha256": str(label_record["sha256"]),
        "lines": ROWS_PER_WORKER,
    }
    return receipt_path, receipt, labels


def _validate_source(source_root: pathlib.Path) -> dict[str, Any]:
    paths = _source_paths(source_root)
    expected_source_files = {
        "plan": SOURCE_ITERATION_PLAN_FILE,
        "claim": "5e4d7acd0aa19410e380b3d02173159701f96fce76e385d5adf24d8b4aa936e8",
        "integrity": "1884d7b34424e50c26ee21cbce890e346eb7ccb0b78156089d290e0a72154031",
        "invalidated": "a5c0afeee5f01df873206bafe26e13d4d04b5f3493f9f275cdd3e3854632fc44",
    }
    for name, expected_sha in expected_source_files.items():
        _regular_file_record(paths[name], expected_sha256=expected_sha)
    plan = qualification.load_sealed(paths["plan"], SOURCE_PLAN_SCHEMA)
    claim = qualification.load_sealed(paths["claim"], SOURCE_CLAIM_SCHEMA)
    integrity = qualification.load_sealed(
        paths["integrity"], SOURCE_INTEGRITY_SCHEMA
    )
    invalidated = qualification.load_sealed(
        paths["invalidated"], SOURCE_INVALIDATED_SCHEMA
    )
    if (
        plan.get("body_sha256") != SOURCE_ITERATION_PLAN_BODY
        or _sha256_file(paths["plan"]) != SOURCE_ITERATION_PLAN_FILE
        or claim.get("body_sha256") != SOURCE_CLAIM_BODY
        or integrity.get("body_sha256") != SOURCE_INTEGRITY_BODY
        or invalidated.get("body_sha256") != SOURCE_INVALIDATED_BODY
        or plan.get("protected_tests_opened") is not False
        or len(plan.get("workers", [])) != WORKERS
        or claim.get("plan", {}).get("path") != str(paths["plan"].resolve())
        or claim.get("plan", {}).get("sha256") != _sha256_file(paths["plan"])
        or claim.get("plan", {}).get("body_sha256") != SOURCE_ITERATION_PLAN_BODY
        or integrity.get("protected_tests_opened") is not True
        or integrity.get("protected_test_authorized") is not False
        or integrity.get("derived_artifacts_written") is not False
        or integrity.get("selection_valid") is not False
        or integrity.get("iteration_completed") is not False
        or integrity.get("iterations_remaining") != 0
        or integrity.get("recovery_authorized") is not False
        or invalidated.get("status") != "compact-value-family-invalidated"
        or invalidated.get("upload_authorized") is not False
        or invalidated.get("goal_complete") is not False
    ):
        raise SuccessorError("source invalidation/carry-forward boundary changed")
    incident_path = pathlib.Path(integrity.get("incident", {}).get("path", ""))
    expected_incident_path = (
        source_root / "iteration-governance" / "terminal"
        / "00-protected-test-integrity-incident.json"
    )
    if incident_path != expected_incident_path:
        raise SuccessorError("integrity receipt redirects the source incident")
    _regular_file_record(
        incident_path,
        expected_sha256=(
            "f23a51b9dc4b524aa6f1eb70c46e556fb2ba272cd8f7709b175b703b581d8476"
        ),
    )
    incident = qualification.load_sealed(
        incident_path,
        "papersoccer.compact-value-bfm.protected-test-integrity-incident.v1",
    )
    if (
        incident.get("body_sha256") != SOURCE_INCIDENT_BODY
        or _sha256_file(incident_path)
        != "f23a51b9dc4b524aa6f1eb70c46e556fb2ba272cd8f7709b175b703b581d8476"
        or incident.get("incident", {}).get("immutable_selection_existed") is not False
        or incident.get("incident", {}).get("protected_metrics_computed") is not False
        or incident.get("incident", {}).get("derived_model_artifacts_written") is not False
        or incident.get("iteration_state", {}).get("fine_tune_started") is not False
        or incident.get("iteration_state", {}).get("worker_outputs_preserved") is not True
    ):
        raise SuccessorError("incident does not permit pre-incident data carry-forward")
    if not paths["disabled_plist"].is_file():
        raise SuccessorError("source LaunchAgent is not durably disabled")

    workers = []
    totals = {"games": 0, "positions": 0, "deep_relabel_positions": 0}
    integrity_workers = integrity.get("worker_results")
    if not isinstance(integrity_workers, list) or len(integrity_workers) != WORKERS:
        raise SuccessorError("integrity receipt lacks ten source workers")
    for index, worker_plan in enumerate(plan["workers"]):
        result_path = pathlib.Path(worker_plan["result_path"])
        expected_result_path = (
            source_root / "on-policy-iteration" / "workers"
            / f"worker-{index:02d}.result.json"
        )
        if result_path != expected_result_path:
            raise SuccessorError(f"source worker {index} path was redirected")
        if result_path.is_symlink() or not result_path.is_file():
            raise SuccessorError(f"source worker {index} is not a regular file")
        result = qualification.load_sealed(result_path, SOURCE_WORKER_SCHEMA)
        expected_reference = integrity_workers[index]
        if (
            result.get("worker_index") != index
            or result.get("plan_body_sha256") != SOURCE_ITERATION_PLAN_BODY
            or result.get("games") != 1_000
            or result.get("positions") != 20_000
            or result.get("deep_relabel_positions") != 5_000
            or result.get("resumed") is not True
            or expected_reference.get("worker_index") != index
            or expected_reference.get("result")
            != qualification.artifact_reference(result_path, SOURCE_WORKER_SCHEMA)
        ):
            raise SuccessorError(f"source worker {index} changed")
        search_receipt_path, _search_receipt, search_labels = _stage_receipt(
            source_root, index,
            ordinal=8, stage="search-targets", filename="search-merged.jsonl",
        )
        rank4_receipt_path, _rank4_receipt, rank4_labels = _stage_receipt(
            source_root, index,
            ordinal=9, stage="rank4-targets", filename="rank4-merged.jsonl",
        )
        workers.append({
            "worker_index": index,
            "result": qualification.artifact_reference(
                result_path, SOURCE_WORKER_SCHEMA
            ),
            "search_target_receipt": _regular_file_record(search_receipt_path),
            "rank4_target_receipt": _regular_file_record(rank4_receipt_path),
            "search_labels": search_labels,
            "rank4_labels": rank4_labels,
            "games": result["games"],
            "positions": result["positions"],
            "deep_relabel_positions": result["deep_relabel_positions"],
            "game_identities_sha256": result["game_identities_sha256"],
        })
        for field in totals:
            totals[field] += int(result[field])
    if totals != {
        "games": GAMES,
        "positions": POSITIONS,
        "deep_relabel_positions": DEEP_POSITIONS,
    }:
        raise SuccessorError("source worker totals changed")
    return {
        "paths": paths,
        "plan": plan,
        "claim": claim,
        "integrity": integrity,
        "invalidated": invalidated,
        "workers": workers,
        "totals": totals,
    }


def _tool_record(path: pathlib.Path) -> dict[str, Any]:
    return _regular_file_record(path)


def _authorization_body(
    source: Mapping[str, Any], authorized_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "clean-successor-carry-forward-authorized",
        "authorized_at_utc": authorized_at_utc,
        "authorization_basis": (
            "user-explicitly-authorized-reuse-sealed-pre-incident-games-and-labels-"
            "with-fresh-protected-holdouts"
        ),
        "source_campaign_invalidated": qualification.artifact_reference(
            source["paths"]["invalidated"], SOURCE_INVALIDATED_SCHEMA
        ),
        "source_integrity_failure": qualification.artifact_reference(
            source["paths"]["integrity"], SOURCE_INTEGRITY_SCHEMA
        ),
        "permitted_carry_forward": [
            "ten-sealed-worker-results",
            "ten-search-merged-label-files",
            "ten-rank4-merged-label-files-as-audit-only",
            "unprotected-anchor-train",
            "unprotected-common-adjudicator",
            "unprotected-canonical-validation",
            "authorized-initial-float-checkpoint",
        ],
        "old_protected_tests_permitted": False,
        "old_protected_tests_must_never_be_accessed": True,
        "new_training_games_authorized": False,
        "new_training_labels_authorized": False,
        "global_repack_authorized": True,
        "training_authorized_after_isolation": True,
        "fresh_protected_tests_authorized_only_after_immutable_selection": True,
        "upload_authorized": False,
    }


def prepare(
    *, source_root: pathlib.Path, output_root: pathlib.Path,
    fresh_holdout_tool: pathlib.Path, canonical_teacher: pathlib.Path,
    authorized_at_utc: str,
) -> pathlib.Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root != canonical_source_root():
        raise SuccessorError("successor source root is not the canonical invalidated campaign")
    qualification._utc(authorized_at_utc, "successor authorization timestamp")
    source = _validate_source(source_root)
    expected_output_root = source_root.parent / SUCCESSOR_CAMPAIGN_ID
    if output_root != expected_output_root:
        raise SuccessorError(
            f"successor output must use the canonical sibling root: {expected_output_root}"
        )
    expected_holdout_tool = (HERE / "compact_value_bfm_fresh_holdout.py").resolve()
    if (
        fresh_holdout_tool.is_symlink()
        or not fresh_holdout_tool.is_file()
        or fresh_holdout_tool.resolve() != expected_holdout_tool
    ):
        raise SuccessorError("fresh protected-holdout tool is absent")
    expected_canonical_teacher = (
        source_root / "iteration-build" / "clang-release"
        / "papersoccer_jacek_replay_teacher"
    ).resolve()
    if (
        canonical_teacher.is_symlink()
        or not canonical_teacher.is_file()
        or not os.access(canonical_teacher, os.X_OK)
        or canonical_teacher.resolve() != expected_canonical_teacher
    ):
        raise SuccessorError("fresh canonical teacher executable is absent")

    authorization_path = output_root / "governance" / "00-authorization.json"

    original_plan = source["plan"]
    source_bundle = compact.FrozenBundle.load(source["paths"]["bundle"])
    source_bundle_manifest = source_bundle.manifest
    source_bundle_record = original_plan["inputs"]["bundle_manifest"]
    if pathlib.Path(str(source_bundle_record.get("path", ""))) != source["paths"]["bundle"]:
        raise SuccessorError("original plan redirects the source bundle")
    if (
        _verify_record(source_bundle_record, "original source bundle manifest")
        != source["paths"]["bundle"]
        or source_bundle_record.get("sha256") != SOURCE_BUNDLE_FILE
        or source_bundle.body_sha256 != SOURCE_BUNDLE_BODY
    ):
        raise SuccessorError("source bundle differs from the exact original plan")
    safe_routes = _safe_base_routes(source_bundle_manifest)
    input_records = _bundle_artifact_records(source_bundle, safe_routes)
    roots_path = source_bundle.artifact_path(
        source_bundle_manifest["routes"]["roots_manifest"]
    )
    predecessor_input_keys = (
        "input_audit", "float_checkpoint", "roots_manifest", "roots_tsv",
        "previous_compact_runtime", "search_teacher_runtime",
    )
    predecessor_inputs = {
        key: dict(original_plan["inputs"][key]) for key in predecessor_input_keys
    }
    predecessor_paths = {
        key: _verify_record(record, f"predecessor input {key}")
        for key, record in predecessor_inputs.items()
    }
    if predecessor_paths["roots_manifest"] != roots_path:
        raise SuccessorError("predecessor roots manifest differs from bundle route")
    tools = dict(original_plan["tools"])
    tools.update({
        "successor": _tool_record(pathlib.Path(__file__).resolve()),
        "fresh_holdout": _tool_record(fresh_holdout_tool.resolve()),
        "canonical_teacher": _tool_record(canonical_teacher.resolve()),
        "opening_generator": _tool_record(
            (HERE / "compact_value_bfm_openings.py").resolve()
        ),
    })
    if authorization_path.exists():
        existing_authorization = qualification.load_sealed(
            authorization_path, AUTHORIZATION_SCHEMA
        )
        authorization_body = _authorization_body(
            source, str(existing_authorization.get("authorized_at_utc"))
        )
        expected_authorization = qualification.seal(authorization_body)
        if existing_authorization != expected_authorization:
            raise SuccessorError("existing successor authorization changed")
        authorization = existing_authorization
    else:
        authorization = qualification.write_sealed(
            authorization_path,
            _authorization_body(source, authorized_at_utc),
        )
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "clean-successor-planned-protected-tests-unmaterialized",
        "authorization": qualification.artifact_reference(
            authorization_path, AUTHORIZATION_SCHEMA
        ),
        "source": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "plan": qualification.artifact_reference(
                source["paths"]["plan"], SOURCE_PLAN_SCHEMA
            ),
            "claim": qualification.artifact_reference(
                source["paths"]["claim"], SOURCE_CLAIM_SCHEMA
            ),
            "integrity_failure": qualification.artifact_reference(
                source["paths"]["integrity"], SOURCE_INTEGRITY_SCHEMA
            ),
            "invalidated": qualification.artifact_reference(
                source["paths"]["invalidated"], SOURCE_INVALIDATED_SCHEMA
            ),
            "workers": source["workers"],
            "totals": source["totals"],
        },
        "training": {
            "architecture": ARCHITECTURE,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "arm": "search-target",
            "initial_float_checkpoint": predecessor_inputs["float_checkpoint"],
            "source_bundle_manifest": dict(source_bundle_record),
            "source_input_audit": predecessor_inputs["input_audit"],
            "roots_manifest": predecessor_inputs["roots_manifest"],
            "roots_tsv": predecessor_inputs["roots_tsv"],
            "prior_compact_runtime": predecessor_inputs["previous_compact_runtime"],
            "search_teacher_runtime": predecessor_inputs["search_teacher_runtime"],
            "safe_routes": safe_routes,
            "safe_input_artifacts": input_records,
            "prior_validation_roles": [
                "canonical_validation", "common_adjudicator"
            ],
            "global_pack_input_rows": POSITIONS,
            "global_pack_reflection_rows": POSITIONS * 2,
            "qat_epochs": 4,
        },
        "fresh_protected_holdout": {
            "materialized": False,
            "may_generate_only_after_immutable_selection": True,
            "old_protected_routes_permitted": False,
            "selection_may_change_after_results": False,
            "tool": tools["fresh_holdout"],
            "campaign_id": f"{SUCCESSOR_CAMPAIGN_ID}-holdout",
            "game_plan_seed": 8_950_116_866_532_575_366,
            "fresh_root_openings": 3_200,
            "fresh_root_seed_policy": (
                "sha256(campaign-id,game-plan-seed,fresh-protected-root-openings-v1)"
            ),
            "fresh_root_group_ids_new": True,
            "quotas": {
                "student-selfplay": 1_600,
                "student-p1-vs-rank4": 320,
                "student-p2-vs-rank4": 320,
                "student-p1-vs-jacek-nn": 320,
                "student-p2-vs-jacek-nn": 320,
                "student-p1-vs-prior-incumbent": 160,
                "student-p2-vs-prior-incumbent": 160,
            },
            "games": 3_200,
            "positions_per_game": 20,
            "positions": 64_000,
            "hard_fraction": [1, 4],
            "hard_positions": 16_000,
            "game_workers": 10,
            "game_chunk_size": 25,
            "compact_tree_nodes": 8_000,
            "rank4_actor_nodes": 16_000,
            "jacek_nn_actor_nodes": 64_000,
            "exploration": 0.5,
            "fpu": 0.5,
            "search_shallow_nodes": 64_000,
            "search_deep_nodes": 500_000,
            "rank4_shallow_nodes": 32_000,
            "rank4_deep_nodes": 400_000,
            "canonical_nodes": 32_000,
            "canonical_deep_nodes": 400_000,
            "canonical_deep_percent": 10,
            "canonical_max_samples_per_game": 100,
            "label_workers": 10,
            "label_chunk_games": 25,
            "source_identities": dict(original_plan["source_identities"]),
            "canonical_teacher": tools["canonical_teacher"],
            "fresh_root_split": "test",
            "diagnostic_only": True,
            "minimum_samples_per_report": 20_000,
        },
        "tools": tools,
        "policy": {
            "source_worker_outputs_immutable": True,
            "new_games": 0,
            "new_labels": 0,
            "second_training_iteration": False,
            "old_protected_tests_opened_by_successor": False,
            "old_protected_tests_permanently_excluded": True,
            "fresh_protected_tests_opened": False,
            "upload_authorized": False,
        },
    }
    plan_path = output_root / "successor-plan.json"
    qualification.write_sealed(plan_path, body)
    load_plan(plan_path, output_root=output_root)
    return plan_path


def load_plan(
    plan_path: pathlib.Path, *, output_root: pathlib.Path,
) -> dict[str, Any]:
    expected_plan_path = output_root.resolve() / "successor-plan.json"
    if (
        plan_path != expected_plan_path
        or plan_path.is_symlink()
        or not plan_path.is_file()
    ):
        raise SuccessorError("successor plan path is not canonical")
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    if (
        plan.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or plan.get("status")
        != "clean-successor-planned-protected-tests-unmaterialized"
        or plan.get("source", {}).get("totals") != {
            "games": GAMES,
            "positions": POSITIONS,
            "deep_relabel_positions": DEEP_POSITIONS,
        }
        or plan.get("training", {}).get("architecture") != ARCHITECTURE
        or plan.get("training", {}).get("seed") != SEED
        or float(plan.get("training", {}).get("learning_rate", -1))
        != LEARNING_RATE
        or plan.get("policy", {}).get("old_protected_tests_permanently_excluded")
        is not True
        or plan.get("policy", {}).get("fresh_protected_tests_opened") is not False
        or plan.get("policy", {}).get("upload_authorized") is not False
    ):
        raise SuccessorError("successor plan policy changed")
    holdout = plan.get("fresh_protected_holdout")
    if (
        not isinstance(holdout, Mapping)
        or holdout.get("campaign_id") != f"{SUCCESSOR_CAMPAIGN_ID}-holdout"
        or holdout.get("materialized") is not False
        or holdout.get("may_generate_only_after_immutable_selection") is not True
        or holdout.get("old_protected_routes_permitted") is not False
        or holdout.get("selection_may_change_after_results") is not False
        or holdout.get("game_plan_seed") != 8_950_116_866_532_575_366
        or holdout.get("fresh_root_openings") != 3_200
        or holdout.get("fresh_root_group_ids_new") is not True
        or holdout.get("games") != 3_200
        or holdout.get("positions") != 64_000
        or holdout.get("hard_positions") != 16_000
        or holdout.get("fresh_root_split") != "test"
        or holdout.get("diagnostic_only") is not True
    ):
        raise SuccessorError("successor fresh-holdout plan changed")
    authorization_record = plan.get("authorization")
    expected_authorization_path = output_root.resolve() / "governance" / "00-authorization.json"
    if (
        not isinstance(authorization_record, Mapping)
        or pathlib.Path(str(authorization_record.get("path", "")))
        != expected_authorization_path
    ):
        raise SuccessorError("successor plan redirects its authorization")
    if expected_authorization_path.is_symlink() or not expected_authorization_path.is_file():
        raise SuccessorError("successor authorization is not a regular file")
    if _sha256_file(expected_authorization_path) != authorization_record.get("sha256"):
        raise SuccessorError("successor authorization hash changed")
    authorization = qualification.load_sealed(
        expected_authorization_path, AUTHORIZATION_SCHEMA
    )
    if (
        authorization.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or authorization.get("status") != "clean-successor-carry-forward-authorized"
        or authorization.get("old_protected_tests_permitted") is not False
        or authorization.get("old_protected_tests_must_never_be_accessed") is not True
        or authorization.get("new_training_games_authorized") is not False
        or authorization.get("new_training_labels_authorized") is not False
        or authorization.get("global_repack_authorized") is not True
        or authorization.get("training_authorized_after_isolation") is not True
        or authorization.get("fresh_protected_tests_authorized_only_after_immutable_selection")
        is not True
        or authorization.get("upload_authorized") is not False
    ):
        raise SuccessorError("successor authorization policy changed")

    source_root = canonical_source_root()
    source_state = _validate_source(source_root)
    source_paths = source_state["paths"]
    expected_authorization = qualification.seal(
        _authorization_body(
            {"paths": source_paths}, str(authorization.get("authorized_at_utc"))
        )
    )
    if authorization != expected_authorization:
        raise SuccessorError("successor authorization ancestry changed")
    source_plan_record = plan.get("source", {}).get("plan")
    if (
        not isinstance(source_plan_record, Mapping)
        or pathlib.Path(str(source_plan_record.get("path", ""))) != source_paths["plan"]
        or source_plan_record.get("sha256") != SOURCE_ITERATION_PLAN_FILE
    ):
        raise SuccessorError("successor plan redirects the predecessor plan")
    source_plan = source_state["plan"]
    expected_source_references = {
        "claim": qualification.artifact_reference(
            source_paths["claim"], SOURCE_CLAIM_SCHEMA
        ),
        "integrity_failure": qualification.artifact_reference(
            source_paths["integrity"], SOURCE_INTEGRITY_SCHEMA
        ),
        "invalidated": qualification.artifact_reference(
            source_paths["invalidated"], SOURCE_INVALIDATED_SCHEMA
        ),
    }
    if any(
        plan.get("source", {}).get(field) != reference
        for field, reference in expected_source_references.items()
    ):
        raise SuccessorError("successor predecessor references changed")
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise SuccessorError("successor tool closure is missing")
    for label, source_record in source_plan["tools"].items():
        if tools.get(label) != source_record:
            raise SuccessorError(f"successor changed predecessor tool {label}")
    predecessor_training_bindings = {
        "initial_float_checkpoint": "float_checkpoint",
        "source_bundle_manifest": "bundle_manifest",
        "source_input_audit": "input_audit",
        "roots_manifest": "roots_manifest",
        "roots_tsv": "roots_tsv",
        "prior_compact_runtime": "previous_compact_runtime",
        "search_teacher_runtime": "search_teacher_runtime",
    }
    for successor_field, predecessor_field in predecessor_training_bindings.items():
        if plan.get("training", {}).get(successor_field) != source_plan["inputs"][
            predecessor_field
        ]:
            raise SuccessorError(
                f"successor predecessor input binding changed: {successor_field}"
            )
    expected_new_tool_paths = {
        "successor": pathlib.Path(__file__).resolve(),
        "fresh_holdout": (HERE / "compact_value_bfm_fresh_holdout.py").resolve(),
        "canonical_teacher": (
            source_root / "iteration-build" / "clang-release"
            / "papersoccer_jacek_replay_teacher"
        ).resolve(),
        "opening_generator": (HERE / "compact_value_bfm_openings.py").resolve(),
    }
    for label, expected_path in expected_new_tool_paths.items():
        record = tools.get(label)
        if (
            not isinstance(record, Mapping)
            or pathlib.Path(str(record.get("path", ""))) != expected_path
        ):
            raise SuccessorError(f"successor redirects new tool {label}")
    for label, record in tools.items():
        _verify_record(record, f"successor tool {label}")
    for label in (
        "initial_float_checkpoint", "source_bundle_manifest",
        "source_input_audit", "roots_manifest", "roots_tsv",
        "prior_compact_runtime", "search_teacher_runtime",
    ):
        _verify_record(plan["training"][label], f"successor training {label}")
    source_bundle_path = _verify_record(
        plan["training"]["source_bundle_manifest"], "successor source bundle"
    )
    if (
        source_bundle_path != source_paths["bundle"]
        or plan["training"]["source_bundle_manifest"].get("sha256")
        != SOURCE_BUNDLE_FILE
    ):
        raise SuccessorError("successor source bundle binding changed")
    bundle = compact.FrozenBundle.load(source_bundle_path)
    if bundle.body_sha256 != SOURCE_BUNDLE_BODY:
        raise SuccessorError("successor source bundle body changed")
    expected_safe_routes = _safe_base_routes(bundle.manifest)
    if plan["training"].get("safe_routes") != expected_safe_routes:
        raise SuccessorError("successor safe route allowlist changed")
    expected_safe_artifacts = _bundle_artifact_records(bundle, expected_safe_routes)
    if plan["training"].get("safe_input_artifacts") != expected_safe_artifacts:
        raise SuccessorError("successor safe input artifact bindings changed")
    workers = plan.get("source", {}).get("workers")
    if not isinstance(workers, list) or len(workers) != WORKERS:
        raise SuccessorError("successor plan worker roster changed")
    for index, worker in enumerate(workers):
        if worker.get("worker_index") != index:
            raise SuccessorError("successor plan worker order changed")
        _verify_record(worker.get("search_target_receipt"), "Search target receipt")
        _verify_record(worker.get("rank4_target_receipt"), "Rank-4 target receipt")
        search_path = _declared_record(worker.get("search_labels"), "Search labels")
        rank4_path = _declared_record(worker.get("rank4_labels"), "Rank-4 labels")
        if (
            search_path.name != "search-merged.jsonl"
            or rank4_path.name != "rank4-merged.jsonl"
            or worker["search_labels"].get("lines") != ROWS_PER_WORKER
            or worker["rank4_labels"].get("lines") != ROWS_PER_WORKER
        ):
            raise SuccessorError("successor source label declaration changed")
    protected = _protected_routes(bundle.manifest)
    safe_routes = plan["training"].get("safe_routes", {})
    flattened = [
        value for values in safe_routes.values() for value in values
    ] if isinstance(safe_routes, Mapping) else []
    if not flattened or any(value in protected for value in flattened):
        raise SuccessorError("successor plan includes an old protected route")
    return plan


def _carry_execution_claim(
    plan_path: pathlib.Path, plan: Mapping[str, Any], output_root: pathlib.Path,
) -> dict[str, Any]:
    path = output_root / "governance" / "01-carry-forward-execution-claim.json"
    expected = {
        "schema": (
            "papersoccer.compact-value-bfm.clean-successor-"
            "carry-forward-execution-claim.v1"
        ),
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "status": "carry-forward-execution-claimed-once",
        "plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
        "execution_phase_tools": {
            "successor": dict(plan["tools"]["successor"]),
            "python": dict(plan["tools"]["python"]),
        },
        "generator_executables_permitted": [],
        "teacher_executables_permitted": [],
        "old_packed_shards_permitted": False,
        "old_protected_artifacts_permitted": False,
        "quarantine_permitted": False,
        "source_reads_permitted": [
            "ten-worker-results",
            "ten-search-target-receipts",
            "ten-rank4-target-receipts",
            "ten-search-merged-labels",
            "ten-rank4-merged-labels",
        ],
        "new_on_policy_games_authorized": 0,
        "new_teacher_labels_authorized": 0,
        "executions_authorized": 1,
    }
    if path.exists():
        claim = qualification.load_sealed(path, expected["schema"])
        if any(claim.get(field) != value for field, value in expected.items()):
            raise SuccessorError("carry-forward execution claim changed")
        return claim
    return qualification.write_sealed(path, {
        **expected,
        "claimed_at_utc": utc_now(),
    })


def _copy_declared_file(
    source_record: Mapping[str, Any], destination: pathlib.Path,
    *, label: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    source = _declared_record(source_record, label)
    lowered = source.as_posix().lower()
    if any(marker in lowered for marker in (
        "/quarantine/", "sealed-final", "sealed_final", "blind-label",
        "blind_label",
    )):
        raise SuccessorError(f"forbidden source path entered carry-forward: {source}")
    if source.is_symlink() or not source.is_file():
        raise SuccessorError(f"carry-forward source is not a regular file: {source}")
    stat = source.stat()
    expected_bytes = int(source_record["bytes"])
    expected_sha = str(source_record["sha256"])
    if stat.st_size != expected_bytes:
        raise SuccessorError(f"carry-forward source byte count changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        copied = _regular_file_record(
            destination,
            expected_sha256=expected_sha,
            expected_bytes=expected_bytes,
        )
        return copied, {
            "birthtime": float(getattr(stat, "st_birthtime", stat.st_ctime)),
            "mtime": float(stat.st_mtime),
        }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    temporary = pathlib.Path(temporary_name)
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with source.open("rb") as source_stream, os.fdopen(
            descriptor, "wb"
        ) as destination_stream:
            for chunk in iter(lambda: source_stream.read(8 * 1024 * 1024), b""):
                destination_stream.write(chunk)
                digest.update(chunk)
                copied_bytes += len(chunk)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        if copied_bytes != expected_bytes or digest.hexdigest() != expected_sha:
            raise SuccessorError(f"carry-forward source content changed: {source}")
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _regular_file_record(
        destination,
        expected_sha256=expected_sha,
        expected_bytes=expected_bytes,
    ), {
        "birthtime": float(getattr(stat, "st_birthtime", stat.st_ctime)),
        "mtime": float(stat.st_mtime),
    }


def _paired_label_audit(
    search_path: pathlib.Path, rank4_path: pathlib.Path,
) -> tuple[int, dict[str, int], str]:
    identity_digest = hashlib.sha256()
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    rows = 0
    with search_path.open("r", encoding="utf-8") as search_stream, rank4_path.open(
        "r", encoding="utf-8"
    ) as rank4_stream:
        while True:
            search_line = search_stream.readline()
            rank4_line = rank4_stream.readline()
            if not search_line and not rank4_line:
                break
            if not search_line or not rank4_line:
                raise SuccessorError("Search/Rank-4 carry-forward row counts differ")
            try:
                search = json.loads(search_line)
                rank4 = json.loads(rank4_line)
            except json.JSONDecodeError as error:
                raise SuccessorError("carry-forward label row is invalid JSON") from error
            if (
                search.get("schema") != selfsearch.SEARCH_TEACHER_SCHEMA
                or rank4.get("schema") != selfsearch.RANK4_TEACHER_SCHEMA
            ):
                raise SuccessorError("carry-forward teacher schema changed")
            selfsearch.corpus.sample_from_teacher_row(search)
            selfsearch.corpus.sample_from_teacher_row(rank4)
            identity_fields = (
                "position_id", "root_group_id", "group_id", "source", "split",
                "winner", "mover", "prefix",
            )
            search_identity = {field: search.get(field) for field in identity_fields}
            rank4_identity = {field: rank4.get(field) for field in identity_fields}
            if search_identity != rank4_identity:
                raise SuccessorError("Search/Rank-4 carry-forward identities differ")
            split = search.get("split")
            if split not in split_counts:
                raise SuccessorError("carry-forward label split changed")
            split_counts[split] += 1
            identity_digest.update(qualification.canonical_json_bytes(search_identity))
            rows += 1
    return rows, split_counts, identity_digest.hexdigest()


def _concatenate_labels(
    plan: Mapping[str, Any], output_root: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    directory = output_root / "carry-forward"
    labels_path = directory / "search-workers-00-through-09.jsonl"
    receipt_path = directory / "carry-forward-receipt.json"
    if receipt_path.exists():
        receipt = qualification.load_sealed(receipt_path, CARRY_SCHEMA)
        concatenated = receipt.get("concatenated_labels")
        if (
            receipt.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
            or receipt.get("plan_body_sha256") != plan["body_sha256"]
            or not isinstance(concatenated, Mapping)
            or pathlib.Path(str(concatenated.get("path", ""))) != labels_path
            or receipt.get("old_protected_tests_accessed") is not False
            or receipt.get("quarantine_accessed") is not False
        ):
            raise SuccessorError("carry-forward receipt changed")
        _verify_record(concatenated, "concatenated labels")
        return labels_path, receipt
    directory.mkdir(parents=True, exist_ok=True)
    copied_workers = []
    timing_values: list[float] = []
    aggregate_splits = {"train": 0, "validation": 0, "test": 0}
    for worker in plan["source"]["workers"]:
        index = int(worker["worker_index"])
        source_result = pathlib.Path(worker["result"]["path"])
        result_source_record = {
            "path": str(source_result),
            "bytes": source_result.stat().st_size,
            "sha256": worker["result"]["sha256"],
        }
        result_copy, result_times = _copy_declared_file(
            result_source_record,
            directory / "worker-results" / f"worker-{index:02d}.result.json",
            label="source worker result",
        )
        search_receipt_copy, search_receipt_times = _copy_declared_file(
            worker["search_target_receipt"],
            directory / "receipts" / f"worker-{index:02d}.search-targets.json",
            label="source Search target receipt",
        )
        rank4_receipt_copy, rank4_receipt_times = _copy_declared_file(
            worker["rank4_target_receipt"],
            directory / "receipts" / f"worker-{index:02d}.rank4-targets.json",
            label="source Rank-4 target receipt",
        )
        search_copy, search_times = _copy_declared_file(
            worker["search_labels"],
            directory / "labels" / "search" / f"worker-{index:02d}.jsonl",
            label="source Search merged labels",
        )
        rank4_copy, rank4_times = _copy_declared_file(
            worker["rank4_labels"],
            directory / "labels" / "rank4" / f"worker-{index:02d}.jsonl",
            label="source Rank-4 merged labels",
        )
        rows, splits, identity_sha = _paired_label_audit(
            pathlib.Path(search_copy["path"]), pathlib.Path(rank4_copy["path"])
        )
        if rows != ROWS_PER_WORKER:
            raise SuccessorError("carry-forward worker does not contain 20,000 rows")
        for split, count in splits.items():
            aggregate_splits[split] += count
        for times in (
            result_times, search_receipt_times, rank4_receipt_times,
            search_times, rank4_times,
        ):
            timing_values.extend((times["birthtime"], times["mtime"]))
        copied_workers.append({
            "worker_index": index,
            "worker_result": result_copy,
            "search_target_receipt": search_receipt_copy,
            "rank4_target_receipt": rank4_receipt_copy,
            "search_labels": search_copy,
            "rank4_labels": rank4_copy,
            "rows": rows,
            "split_counts": splits,
            "paired_position_identity_sha256": identity_sha,
        })
    if aggregate_splits != {
        "train": 159_660, "validation": 19_460, "test": 20_880,
    }:
        raise SuccessorError("carry-forward aggregate split counts changed")

    segments = []
    total_bytes = 0
    for worker in copied_workers:
        source_record = worker["search_labels"]
        begin = total_bytes
        total_bytes += int(source_record["bytes"])
        segments.append({
            "worker_index": worker["worker_index"],
            "source": dict(source_record),
            "byte_begin": begin,
            "byte_end_exclusive": total_bytes,
            "rows": ROWS_PER_WORKER,
            "sha256": source_record["sha256"],
        })
    expected_concatenated_sha = (
        "7b83653ac127bded5334596430c56685448e46a3f03eca409fb234db54ebeac8"
    )
    if total_bytes != 639_418_342:
        raise SuccessorError("successor concatenation source byte total changed")
    if labels_path.exists():
        _regular_file_record(
            labels_path,
            expected_sha256=expected_concatenated_sha,
            expected_bytes=total_bytes,
        )
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory, prefix=".search-workers.", suffix=".jsonl"
        )
        temporary = pathlib.Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for worker in copied_workers:
                    source_path = _verify_record(
                        worker["search_labels"], "worker Search labels"
                    )
                    with source_path.open("rb") as source:
                        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            if (
                temporary.stat().st_size != total_bytes
                or _sha256_file(temporary) != expected_concatenated_sha
            ):
                raise SuccessorError("independently regenerated concatenation changed")
            os.chmod(temporary, 0o444)
            os.replace(temporary, labels_path)
        finally:
            temporary.unlink(missing_ok=True)
    integrity_path = pathlib.Path(plan["source"]["integrity_failure"]["path"])
    integrity = qualification.load_sealed(integrity_path, SOURCE_INTEGRITY_SCHEMA)
    incident_path = pathlib.Path(integrity["incident"]["path"])
    incident = qualification.load_sealed(
        incident_path,
        "papersoccer.compact-value-bfm.protected-test-integrity-incident.v1",
    )
    incident_time = qualification._utc(
        incident["recorded_at_utc"], "source incident timestamp"
    ).timestamp()
    latest_source_time = max(timing_values)
    if latest_source_time >= incident_time:
        raise SuccessorError("carry-forward source does not predate the incident")
    receipt = qualification.write_sealed(receipt_path, {
        "schema": CARRY_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "policy": "worker-index-order-00-through-09-byte-exact-concatenation",
        "execution_claim": qualification.artifact_reference(
            output_root / "governance" / "01-carry-forward-execution-claim.json"
        ),
        "copied_workers": copied_workers,
        "segments": segments,
        "source_workers": WORKERS,
        "source_games": GAMES,
        "source_positions": POSITIONS,
        "source_deep_relabel_positions": DEEP_POSITIONS,
        "new_games_generated": 0,
        "new_labels_generated": 0,
        "source_split_counts": aggregate_splits,
        "source_latest_birth_or_mtime_unix": latest_source_time,
        "incident_recorded_unix": incident_time,
        "causal_precedence_seconds": incident_time - latest_source_time,
        "old_packed_shards_imported": False,
        "rank4_labels_retained_as_matched_audit_only": True,
        "quarantine_accessed": False,
        "concatenated_labels": {
            "path": str(labels_path.resolve()),
            "bytes": total_bytes,
            "sha256": expected_concatenated_sha,
        },
        "old_protected_tests_accessed": False,
    })
    return labels_path, receipt


def _prior_manifests(plan: Mapping[str, Any]) -> list[pathlib.Path]:
    artifacts = plan["training"]["safe_input_artifacts"]
    result = [
        _verify_record(record["manifest"], "canonical validation manifest")
        for record in artifacts["canonical_validation"]
    ]
    result.extend(
        _verify_record(record["manifest"], "common adjudicator manifest")
        for record in artifacts["common_adjudicator"]
    )
    if len(result) != 4:
        raise SuccessorError("successor requires exactly four prior validation manifests")
    return result


def _compact_dataset(manifest_path: pathlib.Path, expected_split: str) -> Any:
    manifest = _canonical_json(manifest_path, "successor sparse shard manifest")
    if manifest.get("split") != expected_split:
        raise SuccessorError("successor sparse shard split changed")
    shard = large_training.load_csr_shard(manifest_path)
    if shard.split != expected_split or len(shard) != manifest.get("samples"):
        raise SuccessorError("successor sparse shard identity changed")
    return compact.Dataset(
        indptr=shard.indptr,
        indices=shard.indices,
        targets=shard.targets,
        weights=shard.weights,
        group_ids=shard.group_ids,
        split=expected_split,
        source_manifest_sha256=_sha256_file(manifest_path),
        source_npz_sha256=shard.npz_sha256,
        source_route=str(manifest_path.resolve()),
    )


def _base_inputs(plan: Mapping[str, Any], new_dataset: Any) -> Any:
    bundle_path = _verify_record(
        plan["training"]["source_bundle_manifest"], "source bundle manifest"
    )
    bundle = compact.FrozenBundle.load(bundle_path)
    routes = plan["training"]["safe_routes"]
    for relative in [value for values in routes.values() for value in values]:
        if bundle.is_protected(relative):
            raise SuccessorError("old protected route reached successor input loader")
    anchor = compact.concatenate_datasets(
        [compact.load_shard(bundle, route) for route in routes["anchor"]],
        split="train",
    )
    common = compact.load_shard(bundle, routes["common_adjudicator"][0])
    canonical_validation = compact.concatenate_datasets(
        [
            compact.load_shard(bundle, route)
            for route in routes["canonical_validation"]
        ],
        split="validation",
    )
    isolation = compact.validate_unprotected_split_isolation(
        new_dataset, anchor, common, canonical_validation
    )
    return compact.TrainingInputs(
        new=new_dataset,
        anchor=anchor,
        common_adjudicator=common,
        canonical_validation=canonical_validation,
        source_routes={
            "new": (new_dataset.source_route,),
            "anchor": tuple(routes["anchor"]),
            "common_adjudicator": tuple(routes["common_adjudicator"]),
            "canonical_validation": tuple(routes["canonical_validation"]),
        },
        paired_row_validation={
            "policy": "source-family-pairing-prevalidated-before-incident",
            "source_input_audit": dict(plan["training"]["source_input_audit"]),
            "passed": True,
        },
        split_isolation=isolation,
        input_audit={
            "policy": "successor-global-repack-plus-runtime-isolation",
            "protected_tests_opened": False,
        },
    )


def repack(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
) -> pathlib.Path:
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    _carry_execution_claim(plan_path, plan, output_root)
    labels_path, carry = _concatenate_labels(plan, output_root)
    prior = _prior_manifests(plan)
    pack_directory = output_root / "global-repack" / "search"
    report = selfsearch.run_pack(
        python=_verify_record(plan["tools"]["python"], "plan Python"),
        pack_tool=_verify_record(plan["tools"]["pack_tool"], "plan packer"),
        roots=_verify_record(plan["training"]["roots_manifest"], "roots manifest"),
        labels=labels_path,
        output_directory=pack_directory,
        prior_manifests=prior,
    )
    if (
        report.get("input_samples_after_reflection") != POSITIONS * 2
        or report.get("packing") != "sqlite-streaming-bounded-memory-v1"
        or len(report.get("prior_shards", [])) != 4
        or report.get("teacher_jsonl_sha256") != [{
            "name": labels_path.name,
            "sha256": carry["concatenated_labels"]["sha256"],
        }]
    ):
        raise SuccessorError("global repack report changed")
    expected_counts = {
        "cross_split_canonical_rows_removed": {
            "train": 432, "validation": 5_270, "test": 4_764,
        },
        "prior_cross_split_rows_removed": {
            "train": 432, "validation": 0, "test": 16,
        },
        "same_orientation_rows_aggregated": {
            "train": 86_358, "validation": 6_706, "test": 6_820,
        },
    }
    if any(report.get(field) != value for field, value in expected_counts.items()):
        raise SuccessorError("global repack deterministic accounting changed")
    shards = report.get("shards")
    if not isinstance(shards, Mapping) or set(shards) != {"train", "validation", "test"}:
        raise SuccessorError("global repack shard roster changed")
    expected_samples = {"train": 232_530, "validation": 26_944, "test": 30_176}
    if any(
        shards[split].get("samples") != samples
        for split, samples in expected_samples.items()
    ):
        raise SuccessorError("global repack deterministic shard counts changed")
    accepted = {"train": 319_320, "validation": 38_920, "test": 41_760}
    for split in ("train", "validation", "test"):
        if accepted[split] != (
            expected_counts["cross_split_canonical_rows_removed"][split]
            + expected_counts["same_orientation_rows_aggregated"][split]
            + expected_samples[split]
        ):
            raise SuccessorError("global repack split accounting is inconsistent")
    train_manifest = pathlib.Path(shards["train"]["manifest"])
    new_dataset = _compact_dataset(train_manifest, "train")
    inputs = _base_inputs(plan, new_dataset)
    receipt_path = output_root / "global-repack" / "repack-receipt.json"
    receipt = qualification.write_sealed(receipt_path, {
        "schema": REPACK_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "carry_forward": qualification.artifact_reference(
            output_root / "carry-forward" / "carry-forward-receipt.json",
            CARRY_SCHEMA,
        ),
        "pack_report": _regular_file_record(pack_directory / "pack-report.json"),
        "prior_validation_manifests": [
            _regular_file_record(path) for path in prior
        ],
        "input_rows": POSITIONS,
        "input_rows_after_reflection": POSITIONS * 2,
        "train_manifest": _regular_file_record(train_manifest),
        "train_npz": _regular_file_record(
            train_manifest.parent / _canonical_json(
                train_manifest, "global train manifest"
            )["npz"]
        ),
        "retained_train_rows": len(new_dataset),
        "cross_split_canonical_rows_removed": report[
            "cross_split_canonical_rows_removed"
        ],
        "prior_cross_split_rows_removed": report[
            "prior_cross_split_rows_removed"
        ],
        "same_orientation_rows_aggregated": report[
            "same_orientation_rows_aggregated"
        ],
        "split_isolation": inputs.split_isolation,
        "training_derived_validation_and_test_shards_are_not_protected_tests": True,
        "old_worker_train_manifests_eligible_for_training": False,
        "old_protected_tests_accessed": False,
        "passed": True,
    })
    training_input_path = output_root / "training-input.json"
    qualification.write_sealed(training_input_path, {
        "schema": TRAINING_INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "repack": qualification.artifact_reference(receipt_path, REPACK_SCHEMA),
        "new_train_manifest": dict(receipt["train_manifest"]),
        "new_train_npz": dict(receipt["train_npz"]),
        "retained_train_rows": len(new_dataset),
        "source_games": GAMES,
        "source_positions": POSITIONS,
        "source_deep_relabel_positions": DEEP_POSITIONS,
        "split_isolation": inputs.split_isolation,
        "sole_new_data_manifest_for_training": True,
        "new_games_generated": 0,
        "new_labels_generated": 0,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": False,
        "eligible_for_training": True,
    })
    return training_input_path


def _require_single_thread_blas() -> None:
    wrong = [name for name in BLAS_ENVIRONMENT if os.environ.get(name) != "1"]
    if wrong:
        raise SuccessorError(
            "successor training requires all BLAS thread variables set to 1: "
            + ", ".join(wrong)
        )


def _load_training_input(
    plan: Mapping[str, Any], output_root: pathlib.Path,
) -> tuple[dict[str, Any], Any]:
    path = output_root / "training-input.json"
    receipt = qualification.load_sealed(path, TRAINING_INPUT_SCHEMA)
    if (
        receipt.get("plan_body_sha256") != plan["body_sha256"]
        or receipt.get("eligible_for_training") is not True
        or receipt.get("old_protected_tests_accessed") is not False
        or receipt.get("fresh_protected_tests_opened") is not False
        or receipt.get("new_games_generated") != 0
        or receipt.get("new_labels_generated") != 0
    ):
        raise SuccessorError("successor training input is ineligible")
    manifest = _verify_record(
        receipt["new_train_manifest"], "successor global train manifest"
    )
    dataset = _compact_dataset(manifest, "train")
    if len(dataset) != receipt.get("retained_train_rows"):
        raise SuccessorError("successor global train rows changed")
    return receipt, dataset


def _validate_selection_closure(
    selection_path: pathlib.Path, *, plan: Mapping[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    expected_directory = output_root.resolve() / "selections"
    if (
        selection_path.parent != expected_directory
        or not selection_path.name.endswith(".successor-selection.json")
        or selection_path.is_symlink()
        or not selection_path.is_file()
    ):
        raise SuccessorError("successor selection path is not canonical")
    expected_file_sha = selection_path.name.removesuffix(
        ".successor-selection.json"
    )
    if _sha256_file(selection_path) != expected_file_sha:
        raise SuccessorError("successor selection is not content addressed")
    selection = qualification.load_sealed(selection_path, SELECTION_SCHEMA)
    gate = selection.get("offline_gate")
    if (
        selection.get("namespace") != NAMESPACE
        or selection.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or selection.get("plan_body_sha256") != plan["body_sha256"]
        or selection.get("architecture") != ARCHITECTURE
        or selection.get("arm") != "search-target"
        or selection.get("seed") != SEED
        or float(selection.get("learning_rate", -1)) != LEARNING_RATE
        or not isinstance(gate, Mapping)
        or not isinstance(gate.get("passed"), bool)
        or selection.get("status") != gate.get("status")
        or selection.get("selection_immutable") is not True
        or selection.get("selection_may_change_after_fresh_protected_tests")
        is not False
        or selection.get("old_protected_tests_accessed") is not False
        or selection.get("old_protected_tests_permanently_excluded") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("fresh_protected_tests_authorized") is not gate.get("passed")
        or selection.get("game_gated") is not False
        or selection.get("upload_authorized") is not False
    ):
        raise SuccessorError("successor immutable selection changed")
    training_path = output_root / "training-input.json"
    expected_training_reference = qualification.artifact_reference(
        training_path, TRAINING_INPUT_SCHEMA
    )
    if selection.get("training_input") != expected_training_reference:
        raise SuccessorError("successor selection training input changed")
    artifact_specs = {
        "float_checkpoint": output_root / "training" / "float-checkpoints",
        "runtime": output_root / "training" / "quantized-runtimes",
        "generated_source": output_root / "fine-tune" / "generated-sources",
    }
    artifact_paths = {}
    for field, directory in artifact_specs.items():
        record = selection.get(field)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise SuccessorError(f"successor selection {field} binding is missing")
        candidate = pathlib.Path(record["path"])
        if candidate.parent != directory.resolve() or candidate.is_symlink():
            raise SuccessorError(f"successor selection {field} path changed")
        artifact_paths[field] = _verify_record(record, f"successor selection {field}")
    architecture = compact.ARCHITECTURES[ARCHITECTURE]
    compact.load_float_checkpoint(artifact_paths["float_checkpoint"], architecture)
    runtime_architecture, _quantized, runtime_selection, _runtime = compact.load_runtime(
        artifact_paths["runtime"]
    )
    if (
        runtime_architecture.name != ARCHITECTURE
        or runtime_selection.get("arm") != "search-target"
        or runtime_selection.get("seed") != SEED
        or runtime_selection.get("float_epoch") != selection.get("float_epoch")
        or runtime_selection.get("qat_epoch") != selection.get("qat_epoch")
        or runtime_selection.get("source_bundle_body_sha256") != SOURCE_BUNDLE_BODY
    ):
        raise SuccessorError("successor selected runtime provenance changed")
    export = selection.get("source_export")
    if (
        not isinstance(export, Mapping)
        or export.get("runtime_sha256") != selection["runtime"]["sha256"]
        or export.get("source_sha256") != selection["generated_source"]["sha256"]
        or export.get("source_ascii_bytes") != selection["generated_source"]["bytes"]
        or export.get("source_limit_exclusive") != 95_000
        or not 0 < int(export.get("source_ascii_bytes", 0)) < 95_000
    ):
        raise SuccessorError("successor source export closure changed")
    return selection


def _selection_reference(
    path: pathlib.Path, *, plan: Mapping[str, Any], output_root: pathlib.Path,
) -> pathlib.Path | None:
    if not path.exists():
        return None
    expected_reference_path = output_root.resolve() / "selection-reference.json"
    if path != expected_reference_path or path.is_symlink():
        raise SuccessorError("successor selection reference path changed")
    reference = qualification.load_sealed(path, SELECTION_REFERENCE_SCHEMA)
    if (
        reference.get("namespace") != NAMESPACE
        or reference.get("campaign_id") != SUCCESSOR_CAMPAIGN_ID
        or reference.get("plan_body_sha256") != plan["body_sha256"]
        or reference.get("selection_immutable") is not True
        or reference.get("fresh_protected_tests_opened") is not False
    ):
        raise SuccessorError("successor selection reference uses another plan")
    record = reference.get("selection")
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise SuccessorError("successor selection reference is malformed")
    selection_path = pathlib.Path(record["path"])
    if selection_path.parent != output_root.resolve() / "selections":
        raise SuccessorError("successor selection reference redirects its selection")
    if _verify_record(record, "successor selection") != selection_path:
        raise SuccessorError("successor selection reference changed")
    selection = _validate_selection_closure(
        selection_path, plan=plan, output_root=output_root
    )
    if reference.get("offline_gate_passed") is not selection["offline_gate"]["passed"]:
        raise SuccessorError("successor selection gate/reference disagree")
    return selection_path


def train(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
) -> pathlib.Path:
    _require_single_thread_blas()
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    reference_path = output_root / "selection-reference.json"
    existing = _selection_reference(
        reference_path, plan=plan, output_root=output_root
    )
    if existing is not None:
        return existing
    training_receipt, new_dataset = _load_training_input(plan, output_root)
    inputs = _base_inputs(plan, new_dataset)
    if inputs.split_isolation != training_receipt.get("split_isolation"):
        raise SuccessorError("successor split isolation changed before training")
    architecture = compact.ARCHITECTURES[ARCHITECTURE]
    parameters = compact.load_float_checkpoint(
        _verify_record(
            plan["training"]["initial_float_checkpoint"],
            "initial float checkpoint",
        ),
        architecture,
    )
    optimizer = compact.AdamW(
        parameters,
        learning_rate=LEARNING_RATE,
        weight_decay=compact.WEIGHT_DECAY,
    )
    arm = compact.ARMS["search-target"]
    coverage_epoch = compact.anchor_coverage_complete_epoch(
        len(inputs.new), len(inputs.anchor)
    )
    best = None
    best_metrics = None
    best_key = None
    best_epoch = 0
    last_progress = coverage_epoch
    history = []
    for epoch in range(1, compact.MAX_FLOAT_EPOCHS + 1):
        losses = []
        for new_rows, anchor_rows in compact.mixed_epoch_batches(
            len(inputs.new), len(inputs.anchor), seed=SEED, epoch=epoch
        ):
            losses.append(compact._train_mixed_batch(
                parameters, architecture, arm, optimizer, inputs,
                new_rows, anchor_rows,
            ))
        metrics = compact.evaluate_validation_pair(
            parameters, architecture, inputs, arm
        )
        key = compact._validation_key(metrics)
        complete = epoch >= coverage_epoch
        eligible = complete and (best_key is None or key < best_key)
        history.append({
            "epoch": epoch,
            "loss": float(sum(losses) / len(losses)),
            "validation": metrics,
            "eligible": eligible,
        })
        if eligible:
            best = {name: value.copy() for name, value in parameters.items()}
            best_metrics, best_key, best_epoch = metrics, key, epoch
            last_progress = epoch
        if complete and epoch - last_progress >= compact.PATIENCE:
            break
    if best is None or best_metrics is None:
        raise SuccessorError("successor fine-tuning produced no eligible checkpoint")
    float_result = compact.FloatTrainingResult(
        parameters=best,
        epoch=best_epoch,
        metrics=best_metrics,
        report={
            "best_float_epoch": best_epoch,
            "history": history,
            "initialization": "sealed-pre-incident-authorized-float-checkpoint",
            "learning_rate": LEARNING_RATE,
            "clean_successor": True,
        },
    )
    quantized = iteration.run_gate_aware_fixed_scale_qat(
        compact, float_result, inputs, architecture, arm, SEED
    )
    gate = compact.offline_advancement_gate(best_metrics, quantized.metrics)
    checkpoint = compact.write_float_checkpoint(
        output_root / "training" / "float-checkpoints", best, architecture
    )
    runtime = compact.write_runtime(
        output_root / "training" / "quantized-runtimes",
        architecture,
        quantized.quantized,
        arm="search-target",
        seed=SEED,
        float_epoch=best_epoch,
        qat_epoch=quantized.qat_epoch,
        source_bundle_body_sha256=SOURCE_BUNDLE_BODY,
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
        "training_input": qualification.artifact_reference(
            output_root / "training-input.json", TRAINING_INPUT_SCHEMA
        ),
        "architecture": ARCHITECTURE,
        "arm": "search-target",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "float_epoch": best_epoch,
        "qat_epoch": quantized.qat_epoch,
        "float_checkpoint": _regular_file_record(checkpoint),
        "runtime": _regular_file_record(runtime),
        "generated_source": _regular_file_record(generated_source),
        "source_export": source_export,
        "float_validation": best_metrics,
        "quantized_validation": quantized.metrics,
        "quantized_selection_policy": quantized.report[
            "iteration_selection_policy"
        ],
        "offline_gate": gate,
        "selection_immutable": True,
        "selection_may_change_after_fresh_protected_tests": False,
        "old_protected_tests_accessed": False,
        "old_protected_tests_permanently_excluded": True,
        "fresh_protected_tests_opened": False,
        "fresh_protected_tests_authorized": gate["passed"],
        "game_gated": False,
        "upload_authorized": False,
    }
    selection_path, _selection = _write_content_addressed(
        output_root / "selections", body, ".successor-selection.json"
    )
    qualification.write_sealed(reference_path, {
        "schema": SELECTION_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "selection": _regular_file_record(selection_path),
        "offline_gate_passed": gate["passed"],
        "fresh_protected_tests_opened": False,
        "selection_immutable": True,
    })
    validated_selection = _selection_reference(
        reference_path, plan=plan, output_root=output_root
    )
    if validated_selection != selection_path:
        raise SuccessorError("successor selection failed immediate closure validation")
    return selection_path


def verify(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan = load_plan(plan_path, output_root=output_root)
    result: dict[str, Any] = {
        "schema": "papersoccer.compact-value-bfm.clean-successor-status.v1",
        "campaign_id": SUCCESSOR_CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "plan_valid": True,
        "old_protected_tests_accessed": False,
    }
    training_input = output_root / "training-input.json"
    result["repack_complete"] = training_input.is_file()
    if training_input.is_file():
        receipt, dataset = _load_training_input(plan, output_root)
        result["retained_train_rows"] = len(dataset)
        result["split_isolation"] = receipt["split_isolation"]
    selection = _selection_reference(
        output_root / "selection-reference.json",
        plan=plan,
        output_root=output_root,
    )
    result["selection_complete"] = selection is not None
    if selection is not None:
        value = qualification.load_sealed(selection, SELECTION_SCHEMA)
        result["offline_gate"] = value["offline_gate"]
        result["selection"] = _regular_file_record(selection)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--source-root", type=pathlib.Path, required=True)
    prepare_command.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare_command.add_argument(
        "--fresh-holdout-tool", type=pathlib.Path, required=True
    )
    prepare_command.add_argument(
        "--canonical-teacher", type=pathlib.Path, required=True
    )
    prepare_command.add_argument("--authorized-at-utc", default=utc_now())
    for name in ("repack", "train", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare(
                source_root=args.source_root,
                output_root=args.output_root,
                fresh_holdout_tool=args.fresh_holdout_tool,
                canonical_teacher=args.canonical_teacher,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "repack":
            result = repack(plan_path=args.plan, output_root=args.output_root)
        elif args.command == "train":
            result = train(plan_path=args.plan, output_root=args.output_root)
        else:
            result = verify(plan_path=args.plan, output_root=args.output_root)
        if isinstance(result, pathlib.Path):
            result = {"path": str(result.resolve()), "sha256": _sha256_file(result)}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        SuccessorError, iteration.IterationError, compact.TrainingError,
        OSError, ValueError, json.JSONDecodeError,
    ) as error:
        print(f"compact successor failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
