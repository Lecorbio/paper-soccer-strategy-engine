#!/usr/bin/env python3
"""Fail-closed discrete-v3 bridge from finalist to strict Rank-4 qualification.

The bridge is deliberately separate from the maintained legacy final/upload
pipeline.  It precommits one immutable finalist and its complete v2 adapter
evaluation/handoff ancestry, the fresh-position exclusion audit, seven frozen
historical exclusions, all six selected development banks, a clean committed
source/runtime preflight, exact Rank-4, and the source-specific gate binary.

Only after that plan exists may ``materialize`` draw one secret seed and create
one 500-opening protected bank.  ``run`` consumes that bank once, executes the
maintained 100-shard/1,000-game gate, and emits a v3-qualified input only on the
maintained 527/260/zero-failure/timing verdict.  No command authorizes upload.
"""

from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
TEST_PATH = REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_final.py"


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 final dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_v3_final_qualification"
)
opening_tools = _load(
    HERE / "compact_value_bfm_openings.py", "compact_v3_final_openings"
)
campaign = _load(
    HERE / "compact_value_bfm_campaign.py", "compact_v3_final_campaign"
)
preflight_tools = _load(
    HERE / "compact_value_bfm_preflight.py", "compact_v3_final_preflight"
)
legacy_final = _load(
    HERE / "compact_value_bfm_final.py", "compact_v3_final_legacy_primitives"
)
adapter = _load(
    HERE / "compact_value_bfm_discrete_v3_adapter.py", "compact_v3_final_adapter"
)
exclusions = _load(
    HERE / "compact_value_bfm_discrete_v3_exclusions.py", "compact_v3_final_exclusions"
)
development = _load(
    HERE / "compact_value_bfm_discrete_v3_development.py",
    "compact_v3_final_development",
)
deployment = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment.py",
    "compact_v3_final_deployment",
)
deployment_preflight = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment_preflight.py",
    "compact_v3_final_deployment_preflight",
)
gate_support = legacy_final.gate_support


BridgeError = qualification.QualificationError
NAMESPACE = adapter.NAMESPACE
CAMPAIGN_ID = adapter.v3.SUCCESSOR_CAMPAIGN_ID

FINALIST_SCHEMA = development.FINALIST_SCHEMA
AUTHORIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-final-authorization.v1"
)
PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-final-plan.v1"
BANK_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-final-bank-claim.v1"
)
BANK_RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-final-bank-receipt.v1"
)
CONSUMPTION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-final-consumption.v1"
)
RAW_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-final-raw-evidence.v1"
)
QUALIFIED_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-rank4-qualified-inputs.v1"
)

BRIDGE_DIRECTORY = "discrete-v3-final"
HISTORICAL_ROOT = (
    REPOSITORY / "results/compact_value_bfm/compact-value-bfm-20260831-v1"
    / "input-bundle/opening-exclusions"
)
EXPECTED_HISTORICAL_SHA256 = (
    "fde89ddd2dfde2fea62804f17f304c8ef8f54bb2a3353f4d7820242fc604de6b",
    "98af9ff685391d93e6b0d18d2cc06fd98bc33900f4cbfee915e34d23ab8ba245",
    "d8aa66b887fd152c1682c5986e3d6fc868df6bf4db874e5a30b27ad8733b04cc",
    "593da0a7676fd12f37ee4a59460c4e9b7ed6a44c692eddbba7787ef7ece3a597",
    "593da0a7676fd12f37ee4a59460c4e9b7ed6a44c692eddbba7787ef7ece3a597",
    "ab81f04bf43bf5de4c3f57897b8cdc886438c1c1f51c86dc15d2bbf92e8bda4d",
    "dbc36b91ab2b7e937523a5bf59bd9e6225de6518546a3d6b82729fd6a6d5ca90",
)
STAGE_ORDER = tuple(opening_tools.DEVELOPMENT_ORDER)
STAGE_COUNTS = dict(opening_tools.DEVELOPMENT_COUNTS)


FinalistValidator = Callable[
    [pathlib.Path, pathlib.Path, pathlib.Path], Mapping[str, Any]
]
FingerprintLoader = Callable[..., frozenset[str] | set[str]]
PreflightValidator = Callable[..., Mapping[str, Any]]
HistoricalValidator = Callable[
    [Sequence[pathlib.Path]], Mapping[str, Any]
]
GitVerifier = Callable[[pathlib.Path, pathlib.Path, str], Mapping[str, Any]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc(value: Any, label: str) -> str:
    qualification._utc(value, label)
    return str(value)


def _record(path: pathlib.Path, *, ascii_required: bool = False,
            executable: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or (
        executable and not os.access(path, os.X_OK)
    ):
        raise BridgeError(f"required final artifact is absent or redirected: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise BridgeError(f"required final artifact is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        **({"ascii": True} if ascii_required else {}),
        **({"executable": True} if executable else {}),
    }


def _verify_record(value: Any, label: str, *, ascii_required: bool = False,
                   executable: bool = False) -> pathlib.Path:
    expected_keys = {"path", "bytes", "sha256"}
    if ascii_required:
        expected_keys.add("ascii")
    if executable:
        expected_keys.add("executable")
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise BridgeError(f"{label} record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _record(path, ascii_required=ascii_required, executable=executable) != dict(value):
        raise BridgeError(f"{label} changed")
    return path.resolve()


def _reference(path: pathlib.Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BridgeError(f"sealed final artifact is absent or redirected: {path}")
    return qualification.artifact_reference(path, schema)


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise BridgeError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if dict(value) != _reference(path, schema):
        raise BridgeError(f"{label} reference changed")
    return path.resolve()


def _safe_root(path: pathlib.Path) -> pathlib.Path:
    try:
        return exclusions._safe_output_root(path)
    except Exception as error:
        raise BridgeError(f"unsafe v3 final root: {error}") from error


def _safe_directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    try:
        return exclusions._safe_directory(path, create=create)
    except Exception as error:
        raise BridgeError(f"unsafe v3 final directory: {error}") from error


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    try:
        return exclusions._safe_output_file(path)
    except Exception as error:
        raise BridgeError(f"unsafe v3 final output: {error}") from error


def _write_content_addressed(directory: pathlib.Path, body: Mapping[str, Any],
                             suffix: str) -> pathlib.Path:
    directory = _safe_directory(directory, create=True)
    artifact = qualification.seal(body)
    raw = qualification.canonical_json_bytes(artifact)
    path = _safe_output(directory / f"{qualification.sha256_bytes(raw)}{suffix}")
    qualification.atomic_write_once(path, raw)
    return path


def _default_historical_validator(paths: Sequence[pathlib.Path]) -> Mapping[str, Any]:
    if len(paths) != 7:
        raise BridgeError("exactly seven historical exclusions are required")
    resolved = []
    for index, path in enumerate(paths):
        lexical = path.absolute()
        expected = (HISTORICAL_ROOT / f"bank-{index:03d}.tsv").resolve()
        if (
            lexical != lexical.resolve() or lexical.resolve() != expected
            or lexical.is_symlink() or not lexical.is_file()
            or qualification.sha256_file(lexical) != EXPECTED_HISTORICAL_SHA256[index]
        ):
            raise BridgeError("historical exclusion identity changed")
        resolved.append(lexical.resolve())
    loaded = opening_tools.load_all_exclusions(resolved)
    return {"paths": resolved, "loaded": loaded}


def _default_preflight_validator(
    *, preflight_path: pathlib.Path, candidate_source: pathlib.Path,
    generated_source: pathlib.Path, runtime_path: pathlib.Path,
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
    repository: pathlib.Path,
    gate_path: pathlib.Path, git_verifier: GitVerifier,
) -> Mapping[str, Any]:
    candidate = _record(candidate_source, ascii_required=True)
    runtime = _record(runtime_path, ascii_required=True)
    try:
        validated = deployment_preflight.validate_reference(
            preflight_path, generated_source=generated_source,
            candidate_source=candidate_source, runtime_path=runtime_path,
            repository=repository, source_repository=REPOSITORY,
            search_tuple=search_tuple, profile=profile, work=work,
        )
    except Exception as error:
        raise BridgeError("deployment-aware preflight did not validate") from error
    commit = validated.get("candidate_commit")
    if (
        validated.get("candidate") != candidate
        or validated.get("runtime") != runtime
        or validated.get("derivation", {}).get("configuration")
        != deployment.deployment_configuration(search_tuple, profile, work)
        or not isinstance(commit, str)
    ):
        raise BridgeError("deployment-aware preflight differs from finalist inputs")
    git = dict(git_verifier(repository, candidate_source, commit))
    if git.get("commit") != commit or git.get("tracked_clean") is not True:
        raise BridgeError("clean Git verifier did not bind final candidate")
    gate = _record(gate_path, executable=True)
    expected_gate = validated.get("reference", {}).get("gate")
    if expected_gate != gate:
        raise BridgeError("gate executable differs from source-specific preflight")
    timing_samples = validated["timing"]["samples"]
    return {
        "commit": commit, "candidate": candidate, "runtime": runtime,
        "preflight": _reference(
            preflight_path, deployment_preflight.REFERENCE_SCHEMA
        ),
        "manifest": dict(validated["plan"]["inputs"]["manifest"]),
        "manifest_body_sha256": validated["plan"]["inputs"]
        ["manifest_body_sha256"],
        "gate": gate, "git": git,
        "uncontended_timing": {
            "first_max_ms": max(row["first_ms"] for row in timing_samples),
            "later_max_ms": max(row["later_max_ms"] for row in timing_samples),
        },
    }


def _default_fingerprint_loader(**kwargs: Any) -> frozenset[str]:
    return exclusions._load_private_canonical_fingerprints(**kwargs)


def _default_finalist_validator(
    reference_path: pathlib.Path, development_plan_path: pathlib.Path,
    campaign_root: pathlib.Path,
) -> Mapping[str, Any]:
    validated = development.validate_finalist(
        reference_path, plan_path=development_plan_path,
        output_root=campaign_root,
    )
    finalist = validated.get("finalist")
    finalist_path = pathlib.Path(str(validated.get("path", "")))
    reference = validated.get("reference")
    plan = qualification.load_sealed(
        development_plan_path, development.PLAN_SCHEMA
    )
    if (
        not isinstance(finalist, Mapping) or not isinstance(reference, Mapping)
        or finalist_path.is_symlink() or not finalist_path.is_file()
        or finalist.get("schema") != development.FINALIST_SCHEMA
        or finalist.get("status")
        != "development-selected-awaiting-preflight-and-frozen-final"
        or finalist.get("development_plan")
        != development._sealed_record(
            development_plan_path, development.PLAN_SCHEMA
        )
        or reference.get("finalist")
        != development._sealed_record(finalist_path, development.FINALIST_SCHEMA)
        or finalist.get("fresh_protected_tests_opened") is not True
        or finalist.get("model_weights_immutable") is not True
        or finalist.get("search_configuration_immutable") is not True
        or finalist.get("development_selected") is not True
        or finalist.get("final_bank_generation_authorized") is not False
        or finalist.get("rank4_gate_authorized") is not False
        or finalist.get("upload_authorized") is not False
    ):
        raise BridgeError("canonical development finalist policy changed")
    candidate = finalist.get("candidate")
    plan_candidate = plan.get("candidate")
    handoff_record = finalist.get("adapter", {}).get("handoff")
    if (
        not isinstance(candidate, Mapping) or not isinstance(plan_candidate, Mapping)
        or not isinstance(handoff_record, Mapping)
        or candidate != plan_candidate
    ):
        raise BridgeError("development finalist candidate/plan ancestry changed")
    handoff_path = development._verify_sealed_record(
        handoff_record, adapter.HANDOFF_SCHEMA, "finalist adapter handoff"
    )
    handoff = qualification.load_sealed(handoff_path, adapter.HANDOFF_SCHEMA)
    handoff_candidate = handoff.get("candidate")
    # The finalist candidate and adapter candidate intentionally have different
    # outer shapes.  Bind their deployment identity field-by-field instead of
    # requiring an invalid flattened-object equality.
    if (
        not isinstance(handoff_candidate, Mapping)
        or candidate.get("candidate_id") != handoff_candidate.get("candidate_id")
        or candidate.get("architecture") != handoff_candidate.get("architecture")
        or candidate.get("target") != handoff_candidate.get("target")
        or candidate.get("selection") != handoff_candidate.get("selection")
        or candidate.get("runtime") != handoff_candidate.get("runtime")
        or candidate.get("generated_source")
        != handoff_candidate.get("generated_source")
        or candidate.get("source_export") != handoff_candidate.get("source_export")
    ):
        raise BridgeError("development finalist differs from adapter deployment identity")
    runtime_path = _verify_record(candidate.get("runtime"), "finalist runtime")
    source_path = _verify_record(
        candidate.get("generated_source"), "finalist generated source"
    )
    runtime = qualification.load_sealed(runtime_path)
    identity = candidate.get("runtime_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("body_sha256") != runtime.get("body_sha256")
        or identity.get("payload_sha256")
        != runtime.get("quantization", {}).get("payload_sha256")
        or candidate.get("source_export", {}).get("runtime_sha256")
        != qualification.sha256_file(runtime_path)
        or candidate.get("source_export", {}).get("source_sha256")
        != qualification.sha256_file(source_path)
    ):
        raise BridgeError("development finalist runtime/source identity changed")
    exclusion_plan = development._verify_sealed_record(
        finalist["exclusion"]["plan"], exclusions.PLAN_SCHEMA,
        "finalist exclusion plan",
    )
    exclusion_receipt = development._verify_sealed_record(
        finalist["exclusion"]["receipt"], exclusions.RECEIPT_SCHEMA,
        "finalist exclusion receipt",
    )
    excluded = exclusions.validate_receipt(
        exclusion_receipt, plan_path=exclusion_plan, output_root=campaign_root
    )
    banks = exclusions.require_development_roster(
        exclusion_receipt, plan_path=exclusion_plan, output_root=campaign_root
    )
    if finalist.get("banks") != {
        stage: dict(banks[stage]) for stage in STAGE_ORDER
    }:
        raise BridgeError("development finalist bank roster changed")
    tuple_value = finalist.get("tuple")
    profile = finalist.get("profile")
    work = finalist.get("profile_work")
    if (
        not isinstance(tuple_value, list)
        or tuple(str(value) for value in tuple_value) not in campaign.TUPLE_ROSTER
        or profile not in campaign.PROFILE_ROSTER
        or work != campaign.PROFILE_ROSTER[profile]
    ):
        raise BridgeError("finalist tuple/profile is outside frozen roster")
    actual = finalist.get("actual_clock")
    colors = actual.get("color_wins") if isinstance(actual, Mapping) else None
    if (
        not isinstance(actual, Mapping) or actual.get("pairs") != 200
        or actual.get("games") != 400 or actual.get("wins", -1) < 211
        or not isinstance(colors, Mapping) or colors.get("0", -1) < 104
        or colors.get("1", -1) < 104 or actual.get("failures") != 0
    ):
        raise BridgeError("finalist actual-clock development gate is not an exact pass")
    return {
        "reference": dict(reference), "finalist": dict(finalist),
        "finalist_path": finalist_path.resolve(), "development_plan": plan,
        "development_plan_path": development_plan_path.resolve(),
        "handoff": handoff, "handoff_path": handoff_path,
        "runtime_path": runtime_path, "source_path": source_path,
        "development_bank_records": banks,
        "exclusion_plan_path": exclusion_plan,
        "exclusion_receipt_path": exclusion_receipt,
        "exclusion_validation": excluded,
    }


def _normalized_inputs(
    *, campaign_root: pathlib.Path, development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path,
    preflight_path: pathlib.Path, candidate_source: pathlib.Path,
    historical_paths: Sequence[pathlib.Path], rank4_source: pathlib.Path,
    gate_path: pathlib.Path, repository: pathlib.Path,
    finalist_validator: FinalistValidator, preflight_validator: PreflightValidator,
    historical_validator: HistoricalValidator, git_verifier: GitVerifier,
) -> dict[str, Any]:
    try:
        finalist = dict(finalist_validator(
            finalist_reference_path, development_plan_path, campaign_root
        ))
    except Exception as error:
        raise BridgeError("canonical development finalist validation failed") from error
    required = {
        "reference", "finalist", "finalist_path", "development_plan",
        "development_plan_path", "handoff", "handoff_path",
        "runtime_path", "source_path",
        "development_bank_records", "exclusion_plan_path",
        "exclusion_receipt_path", "exclusion_validation",
    }
    if set(finalist) != required:
        raise BridgeError("finalist validator returned an incomplete context")
    generated = pathlib.Path(finalist["source_path"])
    runtime = pathlib.Path(finalist["runtime_path"])
    generated_record = _record(generated, ascii_required=True)
    candidate_record = _record(candidate_source, ascii_required=True)
    finalist_body = finalist["finalist"]
    if (
        deployment.TUPLE_ROSTER != tuple(campaign.TUPLE_ROSTER)
        or deployment.PROFILE_ROSTER != campaign.PROFILE_ROSTER
    ):
        raise BridgeError("deployment-source roster differs from development")
    try:
        deployment_derivation = deployment.attest_derivation(
            generated.read_bytes(), candidate_source.read_bytes(),
            search_tuple=finalist_body["tuple"],
            profile=finalist_body["profile"],
            work=finalist_body["profile_work"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BridgeError(
            "committed candidate is not the exact finalist-configured deployment source"
        ) from error
    if (
        deployment_derivation["base_source"]
        != {key: generated_record[key] for key in ("bytes", "sha256", "ascii")}
        or deployment_derivation["deployed_source"]
        != {key: candidate_record[key] for key in ("bytes", "sha256", "ascii")}
        or not 0 < candidate_record["bytes"] < 95_000
    ):
        raise BridgeError("deployment-source derivation record contradicts its files")
    rank4 = _record(rank4_source, ascii_required=True)
    if (
        rank4["sha256"] != qualification.RANK4_SHA256
        or rank4["bytes"] != qualification.RANK4_BYTES
    ):
        raise BridgeError("v3 final bridge Rank-4 identity changed")
    preflight = dict(preflight_validator(
        preflight_path=preflight_path, candidate_source=candidate_source,
        generated_source=generated, runtime_path=runtime,
        search_tuple=finalist_body["tuple"], profile=finalist_body["profile"],
        work=finalist_body["profile_work"], repository=repository, gate_path=gate_path,
        git_verifier=git_verifier,
    ))
    if set(preflight) != {
        "commit", "candidate", "runtime", "preflight", "manifest",
        "manifest_body_sha256", "gate", "git", "uncontended_timing",
    }:
        raise BridgeError("preflight validator returned an incomplete context")
    if preflight["candidate"] != candidate_record or preflight["runtime"] != _record(
        runtime, ascii_required=True
    ):
        raise BridgeError("preflight differs from finalist source/runtime")
    historical = dict(historical_validator(historical_paths))
    if set(historical) != {"paths", "loaded"}:
        raise BridgeError("historical exclusion validator returned incomplete context")
    banks = finalist["development_bank_records"]
    if not isinstance(banks, Mapping) or set(banks) != set(STAGE_ORDER):
        raise BridgeError("chosen six development banks are incomplete")
    for stage in STAGE_ORDER:
        path = pathlib.Path(str(banks[stage]["path"]))
        bank = opening_tools.validate_bank(path)
        if bank.get("stage") != stage or bank.get("opening_count") != STAGE_COUNTS[stage]:
            raise BridgeError("chosen development bank contract changed")
    return {
        "campaign_root": str(campaign_root),
        "development_plan": _reference(
            development_plan_path, development.PLAN_SCHEMA
        ),
        "finalist_reference": _reference(
            finalist_reference_path, development.FINALIST_REFERENCE_SCHEMA
        ),
        "finalist": dict(finalist["reference"]["finalist"]),
        "handoff": dict(finalist["finalist"]["adapter"]["handoff"]),
        "evaluation_completion": dict(
            finalist["finalist"]["adapter"]["evaluation_completion"]
        ),
        "exclusion_plan": dict(finalist["finalist"]["exclusion"]["plan"]),
        "exclusion_receipt": dict(finalist["finalist"]["exclusion"]["receipt"]),
        "candidate_commit": preflight["commit"],
        "candidate": candidate_record,
        "generated_source": generated_record,
        "deployment_derivation": deployment_derivation,
        "deployment_manifest": preflight["manifest"],
        "deployment_manifest_body_sha256": preflight["manifest_body_sha256"],
        "runtime": preflight["runtime"],
        "rank4": rank4,
        "preflight": preflight["preflight"],
        "gate": preflight["gate"],
        "git": preflight["git"],
        "uncontended_timing": dict(preflight["uncontended_timing"]),
        "development_banks": {
            stage: dict(banks[stage]) for stage in STAGE_ORDER
        },
        "historical_exclusions": [
            _record(path) for path in historical["paths"]
        ],
        "historical_body_sha256": historical["loaded"]["body_sha256"],
        "tuple": list(finalist_body["tuple"]),
        "profile": finalist_body["profile"],
        "profile_work": dict(finalist_body["profile_work"]),
        "actual_clock": dict(finalist_body["actual_clock"]),
    }


def _tool_closure() -> dict[str, Any]:
    return {
        "bridge": _record(pathlib.Path(__file__).resolve()),
        "bridge_tests": _record(TEST_PATH),
        "qualification": _record(pathlib.Path(qualification.__file__).resolve()),
        "openings": _record(pathlib.Path(opening_tools.__file__).resolve()),
        "preflight": _record(pathlib.Path(preflight_tools.__file__).resolve()),
        "legacy_final_primitives": _record(pathlib.Path(legacy_final.__file__).resolve()),
        "adapter_v2": _record(pathlib.Path(adapter.__file__).resolve()),
        "fresh_exclusions": _record(pathlib.Path(exclusions.__file__).resolve()),
        "development": _record(pathlib.Path(development.__file__).resolve()),
        "development_tests": _record(development.TEST_PATH),
        "development_runner": _record(
            REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
            / "discrete_v3_development_runner.py"
        ),
        "deployment_source": _record(
            pathlib.Path(deployment.__file__).resolve()
        ),
        "deployment_source_tests": _record(
            REPOSITORY
            / "tests/codingame/test_compact_value_bfm_discrete_v3_deployment.py"
        ),
        "deployment_preflight": _record(
            pathlib.Path(deployment_preflight.__file__).resolve()
        ),
        "deployment_preflight_tests": _record(
            deployment_preflight.TEST_PATH
        ),
        "gate_support": _record(pathlib.Path(gate_support.__file__).resolve()),
    }


def _authorization_body(inputs: Mapping[str, Any], authorized_at_utc: str) -> dict[str, Any]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "one-discrete-v3-strict-final-authorized",
        "authorized_at_utc": _utc(authorized_at_utc, "v3 final authorization time"),
        "development_plan": inputs["development_plan"],
        "finalist_reference": inputs["finalist_reference"],
        "finalist": inputs["finalist"],
        "handoff": inputs["handoff"],
        "evaluation_completion": inputs["evaluation_completion"],
        "exclusion_plan": inputs["exclusion_plan"],
        "exclusion_receipt": inputs["exclusion_receipt"],
        "candidate_commit": inputs["candidate_commit"],
        "candidate": inputs["candidate"],
        "generated_source": inputs["generated_source"],
        "deployment_derivation": inputs["deployment_derivation"],
        "deployment_manifest": inputs["deployment_manifest"],
        "deployment_manifest_body_sha256": inputs[
            "deployment_manifest_body_sha256"
        ],
        "runtime": inputs["runtime"],
        "rank4": inputs["rank4"],
        "preflight": inputs["preflight"],
        "source_binding": inputs["source_binding"],
        "development_banks": inputs["development_banks"],
        "historical_exclusions": inputs["historical_exclusions"],
        "tuple": inputs["tuple"],
        "profile": inputs["profile"],
        "profile_work": inputs["profile_work"],
        "tool_closure": _tool_closure(),
        "secret_bank_materializations_authorized": 1,
        "strict_rank4_gate_attempts_authorized": 1,
        "uploads_authorized": 0,
        "rank4_replacement_authorized": False,
    }


def _plan_body(inputs: Mapping[str, Any], authorization_path: pathlib.Path,
               planned_at_utc: str, bridge_root: pathlib.Path) -> dict[str, Any]:
    deployed = inputs["deployment_derivation"]["configuration"]
    return {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-strict-final-planned-bank-unmaterialized",
        "planned_at_utc": _utc(planned_at_utc, "v3 final plan time"),
        "authorization": _reference(authorization_path, AUTHORIZATION_SCHEMA),
        "inputs": dict(inputs),
        "thresholds": qualification.strict_gate_verdict({})["thresholds"],
        "configuration": {
            "openings": 500, "pairs": 500, "games": 1_000,
            "shards": 100, "pairs_per_shard": 5, "workers": 4,
            "candidate_actions": deployed["candidate_actions"],
            "candidate_expansions": deployed["candidate_expansions"],
            "candidate_seed": deployed["candidate_shuffle_seed"],
            "deployment": dict(deployed),
            "rank4_nodes": 3_000_000,
            "maximum_turns": 320,
        },
        "paths": {
            "bridge_root": str(bridge_root),
            "source_binding": str(bridge_root / "source-binding.json"),
            "bank_claim": str(bridge_root / "bank-claim.json"),
            "bank_receipt": str(bridge_root / "bank-receipt.json"),
            "bank_lock": str(bridge_root / "bank.lock"),
            "protected": str(bridge_root / "protected"),
            "ledger": str(bridge_root / "ledger"),
        },
        "tool_closure": _tool_closure(),
        "policy": {
            "fresh_private_fingerprints_serialized_publicly": False,
            "selection_may_change": False,
            "bank_materialized": False,
            "rank4_gate_authorized_after_bank": True,
            "upload_authorized": False,
        },
    }


def prepare(
    *, campaign_root: pathlib.Path, development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path,
    preflight_path: pathlib.Path, candidate_source: pathlib.Path,
    historical_paths: Sequence[pathlib.Path], rank4_source: pathlib.Path,
    gate_path: pathlib.Path, repository: pathlib.Path,
    authorized_at_utc: str, planned_at_utc: str,
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
) -> pathlib.Path:
    campaign_root = _safe_root(campaign_root)
    bridge_root = _safe_directory(campaign_root / BRIDGE_DIRECTORY, create=True)
    authorization_path = _safe_output(bridge_root / "authorization.json")
    plan_path = _safe_output(bridge_root / "plan.json")
    inputs = _normalized_inputs(
        campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path, candidate_source=candidate_source,
        historical_paths=historical_paths, rank4_source=rank4_source,
        gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator, preflight_validator=preflight_validator,
        historical_validator=historical_validator, git_verifier=git_verifier,
    )
    source_binding_path = _safe_output(bridge_root / "source-binding.json")
    if source_binding_path.exists():
        source_binding = qualification.load_sealed(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        )
        qualification.validate_source_binding(source_binding)
        if (
            source_binding.get("candidate_commit") != inputs["candidate_commit"]
            or source_binding.get("candidate", {}).get("sha256")
            != inputs["candidate"]["sha256"]
        ):
            raise BridgeError("existing v3 source binding changed")
    else:
        qualification.create_source_binding(
            source_binding_path,
            candidate_source=pathlib.Path(inputs["candidate"]["path"]),
            candidate_commit=inputs["candidate_commit"],
            rank4_source=pathlib.Path(inputs["rank4"]["path"]),
            opponent_source=pathlib.Path(inputs["rank4"]["path"]),
        )
    inputs["source_binding"] = _reference(
        source_binding_path, qualification.SOURCE_BINDING_SCHEMA
    )
    auth_body = _authorization_body(inputs, authorized_at_utc)
    if authorization_path.exists():
        if qualification.load_sealed(
            authorization_path, AUTHORIZATION_SCHEMA
        ) != qualification.seal(auth_body):
            raise BridgeError("existing v3 final authorization changed")
    else:
        qualification.write_sealed(authorization_path, auth_body)
    body = _plan_body(inputs, authorization_path, planned_at_utc, bridge_root)
    if plan_path.exists():
        if qualification.load_sealed(plan_path, PLAN_SCHEMA) != qualification.seal(body):
            raise BridgeError("existing v3 final plan changed")
    else:
        if any(os.path.lexists(bridge_root / name) for name in (
            "bank-claim.json", "bank-receipt.json", "bank.lock",
            "protected", "ledger"
        )):
            raise BridgeError("v3 final execution output predates its plan")
        qualification.write_sealed(plan_path, body)
    validate_plan(
        plan_path, campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path, candidate_source=candidate_source,
        historical_paths=historical_paths, rank4_source=rank4_source,
        gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator, preflight_validator=preflight_validator,
        historical_validator=historical_validator, git_verifier=git_verifier,
    )
    return plan_path


def validate_plan(
    path: pathlib.Path, *, campaign_root: pathlib.Path,
    development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path, preflight_path: pathlib.Path,
    candidate_source: pathlib.Path, historical_paths: Sequence[pathlib.Path],
    rank4_source: pathlib.Path, gate_path: pathlib.Path, repository: pathlib.Path,
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
) -> dict[str, Any]:
    campaign_root = _safe_root(campaign_root)
    bridge_root = _safe_directory(campaign_root / BRIDGE_DIRECTORY, create=False)
    expected_path = bridge_root / "plan.json"
    if path.absolute() != expected_path or path.is_symlink() or not path.is_file():
        raise BridgeError("v3 final plan path is not canonical")
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    inputs = _normalized_inputs(
        campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path, candidate_source=candidate_source,
        historical_paths=historical_paths, rank4_source=rank4_source,
        gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator, preflight_validator=preflight_validator,
        historical_validator=historical_validator, git_verifier=git_verifier,
    )
    source_binding_path = bridge_root / "source-binding.json"
    source_binding = qualification.load_sealed(
        source_binding_path, qualification.SOURCE_BINDING_SCHEMA
    )
    qualification.validate_source_binding(source_binding)
    if (
        source_binding.get("candidate_commit") != inputs["candidate_commit"]
        or source_binding.get("candidate", {}).get("sha256")
        != inputs["candidate"]["sha256"]
    ):
        raise BridgeError("v3 source binding differs from final inputs")
    inputs["source_binding"] = _reference(
        source_binding_path, qualification.SOURCE_BINDING_SCHEMA
    )
    authorization = bridge_root / "authorization.json"
    auth = qualification.load_sealed(authorization, AUTHORIZATION_SCHEMA)
    if auth != qualification.seal(_authorization_body(
        inputs, str(auth.get("authorized_at_utc"))
    )):
        raise BridgeError("v3 final authorization content changed")
    expected = qualification.seal(_plan_body(
        inputs, authorization, str(plan.get("planned_at_utc")), bridge_root
    ))
    if plan != expected:
        raise BridgeError("v3 final plan content changed")
    return plan


def _fresh_private_set(
    plan: Mapping[str, Any], *, campaign_root: pathlib.Path,
    fingerprint_loader: FingerprintLoader,
) -> frozenset[str]:
    inputs = plan["inputs"]
    values = frozenset(fingerprint_loader(
        receipt_path=pathlib.Path(inputs["exclusion_receipt"]["path"]),
        plan_path=pathlib.Path(inputs["exclusion_plan"]["path"]),
        output_root=campaign_root,
    ))
    if not values or any(
        not isinstance(value, str) or len(value) != 64 for value in values
    ):
        raise BridgeError("fresh private canonical fingerprint set is invalid")
    public = qualification.load_sealed(
        pathlib.Path(inputs["exclusion_receipt"]["path"]), exclusions.RECEIPT_SCHEMA
    )
    if (
        public.get("verdict", {}).get("development_games_authorized") is not True
        or public.get("intersection", {}).get("unique_canonical_count") != 0
        or public.get("counts", {}).get("fresh_unique_canonical") != len(values)
    ):
        raise BridgeError("fresh private fingerprints disagree with public exclusion audit")
    return values


def _development_variants(plan: Mapping[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    variants: set[str] = set()
    sources = []
    for stage in STAGE_ORDER:
        record = plan["inputs"]["development_banks"][stage]
        path = pathlib.Path(record["path"])
        bank = opening_tools.validate_bank(path)
        if (
            bank.get("stage") != stage
            or bank.get("classification") != "unprotected-development"
            or bank.get("opening_count") != STAGE_COUNTS[stage]
            or qualification.sha256_file(path) != record["sha256"]
        ):
            raise BridgeError("development bank changed before strict final")
        for opening in bank["openings"]:
            values = {
                value for name, value in opening["fingerprints"].items()
                if name != "canonical"
            }
            if values & variants:
                raise BridgeError("chosen development banks overlap by symmetry")
            variants.update(values)
        sources.append({
            "stage": stage, "path": path.name,
            "sha256": record["sha256"], "opening_count": STAGE_COUNTS[stage],
        })
    return variants, sources


def _union_exclusions(
    plan: Mapping[str, Any], *, campaign_root: pathlib.Path,
    fingerprint_loader: FingerprintLoader,
    historical_validator: HistoricalValidator,
) -> dict[str, Any]:
    historical_paths = [
        pathlib.Path(record["path"])
        for record in plan["inputs"]["historical_exclusions"]
    ]
    historical = dict(historical_validator(historical_paths))
    old = set(historical["loaded"]["fingerprints"])
    development, development_sources = _development_variants(plan)
    fresh = _fresh_private_set(
        plan, campaign_root=campaign_root, fingerprint_loader=fingerprint_loader
    )
    if old & development:
        raise BridgeError("development bank overlaps a historical exclusion")
    # Fresh values are canonical hashes.  Every candidate four-way variant set
    # contains its canonical minimum, so adding these hashes to the generator's
    # excluded set is sufficient and avoids serializing protected variants.
    union = old | development | set(fresh)
    material = {
        "historical_body_sha256": historical["loaded"]["body_sha256"],
        "historical_count": len(old),
        "development_banks": {
            stage: plan["inputs"]["development_banks"][stage]["sha256"]
            for stage in STAGE_ORDER
        },
        "development_variant_count": len(development),
        "fresh_payload": qualification.artifact_reference(
            pathlib.Path(
                qualification.load_sealed(
                    pathlib.Path(plan["inputs"]["exclusion_receipt"]["path"]),
                    exclusions.RECEIPT_SCHEMA,
                )["references"]["protected_canonical_fingerprints"]["path"]
            ),
            exclusions.FINGERPRINT_SCHEMA,
        ),
        "fresh_unique_canonical_count": len(fresh),
    }
    return {
        "fingerprints": union,
        "historical": old,
        "development": development,
        "fresh": fresh,
        "summary": material,
        "sources": [
            *historical["loaded"]["sources"],
            *development_sources,
            {"classification": "protected-fresh-canonical-reference",
             **material["fresh_payload"]},
        ],
        "body_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(material)
        ),
    }


def _bank_claim_body(plan_path: pathlib.Path, plan: Mapping[str, Any],
                     claimed_at_utc: str) -> dict[str, Any]:
    return {
        "schema": BANK_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-final-bank-materialization-claimed-once",
        "claimed_at_utc": _utc(claimed_at_utc, "v3 bank claim time"),
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "finalist": plan["inputs"]["finalist"],
        "source_binding": plan["inputs"]["source_binding"],
        "exclusion_receipt": plan["inputs"]["exclusion_receipt"],
        "entropy_draws_authorized": 1,
        "protected_banks_authorized": 1,
        "selection_may_change": False,
        "upload_authorized": False,
    }


def _materialize_bank_locked(
    *, plan_path: pathlib.Path, campaign_root: pathlib.Path,
    development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path, preflight_path: pathlib.Path,
    candidate_source: pathlib.Path, historical_paths: Sequence[pathlib.Path],
    rank4_source: pathlib.Path, gate_path: pathlib.Path, repository: pathlib.Path,
    claimed_at_utc: str, entropy: Callable[[int], bytes],
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    fingerprint_loader: FingerprintLoader = _default_fingerprint_loader,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
) -> pathlib.Path:
    campaign_root = _safe_root(campaign_root)
    plan = validate_plan(
        plan_path, campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path, candidate_source=candidate_source,
        historical_paths=historical_paths, rank4_source=rank4_source,
        gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator, preflight_validator=preflight_validator,
        historical_validator=historical_validator, git_verifier=git_verifier,
    )
    bridge_root = pathlib.Path(plan["paths"]["bridge_root"])
    protected_root = _safe_directory(pathlib.Path(plan["paths"]["protected"]), create=True)
    opening_bank_directory = _safe_directory(
        protected_root / "opening-bank", create=True
    )
    gate_bank_directory = _safe_directory(
        protected_root / "gate-bank", create=True
    )
    bank_adapter_path = _safe_output(bridge_root / "protected-bank-adapter.json")
    gate_binding_path = _safe_output(bridge_root / "gate-binding.json")
    claim_path = _safe_output(pathlib.Path(plan["paths"]["bank_claim"]))
    receipt_path = _safe_output(pathlib.Path(plan["paths"]["bank_receipt"]))
    if receipt_path.exists():
        validate_bank_receipt(
            receipt_path, plan_path=plan_path, campaign_root=campaign_root,
            development_plan_path=development_plan_path,
            finalist_reference_path=finalist_reference_path,
            preflight_path=preflight_path,
            candidate_source=candidate_source, historical_paths=historical_paths,
            rank4_source=rank4_source, gate_path=gate_path, repository=repository,
            finalist_validator=finalist_validator,
            preflight_validator=preflight_validator,
            historical_validator=historical_validator,
            fingerprint_loader=fingerprint_loader, git_verifier=git_verifier,
        )
        return receipt_path
    if claim_path.exists() or claim_path.is_symlink():
        raise BridgeError("v3 final bank claim is spent without a valid receipt")
    if (
        any(opening_bank_directory.iterdir())
        or any(gate_bank_directory.iterdir())
        or os.path.lexists(bank_adapter_path)
        or os.path.lexists(gate_binding_path)
    ):
        raise BridgeError(
            "v3 protected nested route is not pristine before bank claim"
        )
    qualification.write_sealed(
        claim_path, _bank_claim_body(plan_path, plan, claimed_at_utc)
    )
    union = _union_exclusions(
        plan, campaign_root=campaign_root, fingerprint_loader=fingerprint_loader,
        historical_validator=historical_validator,
    )
    seed = entropy(32)
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise BridgeError("v3 final entropy source did not return 256 bits")
    openings = opening_tools.generate_openings(
        stage="protected_final", count=500, seed=seed,
        excluded_fingerprints=set(union["fingerprints"]),
    )
    seen: set[str] = set()
    for opening in openings:
        variants = {
            value for name, value in opening["fingerprints"].items()
            if name != "canonical"
        }
        if (
            variants & union["historical"]
            or variants & union["development"]
            or opening["fingerprints"]["canonical"] in union["fresh"]
            or variants & seen
        ):
            raise BridgeError("generated final opening violates four-way exclusions")
        seen.update(variants)
    exclusion_view = {
        "body_sha256": union["body_sha256"],
        "sources": union["sources"],
    }
    bank_path = opening_tools.write_bank(
        opening_bank_directory,
        opening_tools.bank_payload(
            stage="protected_final", classification="protected-final",
            seed=seed, exclusions=exclusion_view, openings=openings,
            source_binding=plan["inputs"]["source_binding"],
            seed_receipt=_reference(claim_path, BANK_CLAIM_SCHEMA),
        ),
    )
    opening_tools.validate_bank(bank_path)
    gate_bank_path = legacy_final._materialize_gate_bank(
        gate_bank_directory, bank_path
    )
    qualification.write_sealed(bank_adapter_path, {
        "schema": qualification.FINAL_BANK_SCHEMA,
        "namespace": NAMESPACE,
        "classification": "fresh-protected-final",
        "source_binding": plan["inputs"]["source_binding"],
        "candidate_commit": plan["inputs"]["candidate_commit"],
        "candidate_sha256": plan["inputs"]["candidate"]["sha256"],
        "rank4_sha256": qualification.RANK4_SHA256,
        "opening_count": 500,
        "protected_bank": {
            "path": str(bank_path.resolve()), "sha256": qualification.sha256_file(bank_path)
        },
        "gate_bank": {
            "path": str(gate_bank_path.resolve()),
            "sha256": qualification.sha256_file(gate_bank_path),
        },
        "bank_claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "exclusion_body_sha256": union["body_sha256"],
    })
    qualification.create_gate_binding(
        gate_binding_path,
        source_binding_path=pathlib.Path(plan["inputs"]["source_binding"]["path"]),
        bank_path=bank_adapter_path, harness_path=pathlib.Path(plan["inputs"]["gate"]["path"]),
    )
    qualification.write_sealed(receipt_path, {
        "schema": BANK_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-final-bank-materialized-unconsumed",
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "source_binding": plan["inputs"]["source_binding"],
        "protected_bank": _record(bank_path),
        "gate_bank": _record(gate_bank_path),
        "bank_adapter": _reference(bank_adapter_path, qualification.FINAL_BANK_SCHEMA),
        "gate_binding": _reference(gate_binding_path, qualification.GATE_BINDING_SCHEMA),
        "seed_sha256": qualification.sha256_bytes(seed),
        "opening_count": 500,
        "exclusion_body_sha256": union["body_sha256"],
        "counts": {
            "historical_variants": len(union["historical"]),
            "development_variants": len(union["development"]),
            "fresh_unique_canonical": len(union["fresh"]),
            "final_openings": 500,
        },
        "four_way_overlap_count": 0,
        "bank_consumed": False,
        "upload_authorized": False,
    })
    validate_bank_receipt(
        receipt_path, plan_path=plan_path, campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path,
        candidate_source=candidate_source, historical_paths=historical_paths,
        rank4_source=rank4_source, gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator,
        preflight_validator=preflight_validator,
        historical_validator=historical_validator,
        fingerprint_loader=fingerprint_loader, git_verifier=git_verifier,
    )
    return receipt_path


def materialize_bank(
    *, plan_path: pathlib.Path, campaign_root: pathlib.Path,
    development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path, preflight_path: pathlib.Path,
    candidate_source: pathlib.Path, historical_paths: Sequence[pathlib.Path],
    rank4_source: pathlib.Path, gate_path: pathlib.Path, repository: pathlib.Path,
    claimed_at_utc: str, entropy: Callable[[int], bytes],
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    fingerprint_loader: FingerprintLoader = _default_fingerprint_loader,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
) -> pathlib.Path:
    campaign_root = _safe_root(campaign_root)
    bridge_root = _safe_directory(campaign_root / BRIDGE_DIRECTORY, create=False)
    lock_path = bridge_root / "bank.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise BridgeError("v3 final bank lock path is unsafe")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise BridgeError("another v3 final bank materialization is active") from error
        return _materialize_bank_locked(
            plan_path=plan_path, campaign_root=campaign_root,
            development_plan_path=development_plan_path,
            finalist_reference_path=finalist_reference_path,
            preflight_path=preflight_path,
            candidate_source=candidate_source, historical_paths=historical_paths,
            rank4_source=rank4_source, gate_path=gate_path, repository=repository,
            claimed_at_utc=claimed_at_utc, entropy=entropy,
            finalist_validator=finalist_validator,
            preflight_validator=preflight_validator,
            historical_validator=historical_validator,
            fingerprint_loader=fingerprint_loader, git_verifier=git_verifier,
        )
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def validate_bank_receipt(
    path: pathlib.Path, *, plan_path: pathlib.Path, campaign_root: pathlib.Path,
    development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path, preflight_path: pathlib.Path,
    candidate_source: pathlib.Path, historical_paths: Sequence[pathlib.Path],
    rank4_source: pathlib.Path, gate_path: pathlib.Path, repository: pathlib.Path,
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    fingerprint_loader: FingerprintLoader = _default_fingerprint_loader,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
) -> dict[str, Any]:
    campaign_root = _safe_root(campaign_root)
    plan = validate_plan(
        plan_path, campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path, candidate_source=candidate_source,
        historical_paths=historical_paths, rank4_source=rank4_source,
        gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator, preflight_validator=preflight_validator,
        historical_validator=historical_validator, git_verifier=git_verifier,
    )
    expected_path = pathlib.Path(plan["paths"]["bank_receipt"])
    if path.absolute() != expected_path or path.is_symlink() or not path.is_file():
        raise BridgeError("v3 bank receipt path changed")
    receipt = qualification.load_sealed(path, BANK_RECEIPT_SCHEMA)
    claim_path = pathlib.Path(plan["paths"]["bank_claim"])
    claim = qualification.load_sealed(claim_path, BANK_CLAIM_SCHEMA)
    expected_claim = qualification.seal(_bank_claim_body(
        plan_path, plan, str(claim.get("claimed_at_utc"))
    ))
    if claim != expected_claim:
        raise BridgeError("v3 final bank claim changed")
    protected_path = _verify_record(receipt.get("protected_bank"), "protected final bank")
    gate_bank_path = _verify_record(receipt.get("gate_bank"), "protected gate bank")
    bank = opening_tools.validate_bank(protected_path)
    gate_support.validate_bank(gate_bank_path)
    union = _union_exclusions(
        plan, campaign_root=campaign_root, fingerprint_loader=fingerprint_loader,
        historical_validator=historical_validator,
    )
    if (
        bank.get("classification") != "protected-final"
        or bank.get("opening_count") != 500
        or bank.get("source_binding") != plan["inputs"]["source_binding"]
        or bank.get("seed_receipt") != _reference(claim_path, BANK_CLAIM_SCHEMA)
        or bank.get("exclusions_body_sha256") != union["body_sha256"]
    ):
        raise BridgeError("protected final bank ancestry changed")
    seen: set[str] = set()
    for opening in bank["openings"]:
        variants = {
            value for name, value in opening["fingerprints"].items()
            if name != "canonical"
        }
        if (
            variants & union["historical"]
            or variants & union["development"]
            or opening["fingerprints"]["canonical"] in union["fresh"]
            or variants & seen
        ):
            raise BridgeError("protected final bank overlaps an exclusion")
        seen.update(variants)
    try:
        seed = bytes.fromhex(str(bank.get("seed_hex", "")))
    except ValueError as error:
        raise BridgeError("protected final seed is malformed") from error
    if len(seed) != 32:
        raise BridgeError("protected final seed is not 256 bits")
    bank_adapter_path = _verify_reference(
        receipt.get("bank_adapter"), qualification.FINAL_BANK_SCHEMA,
        "protected bank adapter",
    )
    gate_binding_path = _verify_reference(
        receipt.get("gate_binding"), qualification.GATE_BINDING_SCHEMA,
        "v3 final gate binding",
    )
    source = qualification.load_sealed(
        pathlib.Path(plan["inputs"]["source_binding"]["path"]),
        qualification.SOURCE_BINDING_SCHEMA,
    )
    qualification.validate_source_binding(source)
    binding = qualification.load_sealed(
        gate_binding_path, qualification.GATE_BINDING_SCHEMA
    )
    if (
        binding.get("candidate") != source["candidate"]
        or binding.get("rank4") != source["rank4"]
        or binding.get("bank")
        != _reference(bank_adapter_path, qualification.FINAL_BANK_SCHEMA)
        or binding.get("harness", {}).get("sha256") != plan["inputs"]["gate"]["sha256"]
    ):
        raise BridgeError("v3 final gate binding changed")
    expected = qualification.seal({
        "schema": BANK_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-final-bank-materialized-unconsumed",
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "source_binding": plan["inputs"]["source_binding"],
        "protected_bank": _record(protected_path),
        "gate_bank": _record(gate_bank_path),
        "bank_adapter": _reference(bank_adapter_path, qualification.FINAL_BANK_SCHEMA),
        "gate_binding": _reference(gate_binding_path, qualification.GATE_BINDING_SCHEMA),
        "seed_sha256": qualification.sha256_bytes(seed),
        "opening_count": 500,
        "exclusion_body_sha256": union["body_sha256"],
        "counts": {
            "historical_variants": len(union["historical"]),
            "development_variants": len(union["development"]),
            "fresh_unique_canonical": len(union["fresh"]),
            "final_openings": 500,
        },
        "four_way_overlap_count": 0,
        "bank_consumed": False,
        "upload_authorized": False,
    })
    if receipt != expected:
        raise BridgeError("v3 final bank receipt content changed")
    return {
        "receipt": receipt, "plan": plan, "bank": bank,
        "protected_bank_path": protected_path, "gate_bank_path": gate_bank_path,
        "gate_binding_path": gate_binding_path,
    }


def consume_bank(plan_path: pathlib.Path, plan: Mapping[str, Any],
                 bank_receipt_path: pathlib.Path, launched_at_utc: str) -> pathlib.Path:
    ledger = _safe_directory(pathlib.Path(plan["paths"]["ledger"]), create=True)
    path = _safe_output(ledger / "consumption.json")
    body = {
        "schema": CONSUMPTION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-final-bank-consumed-at-launch",
        "launched_at_utc": _utc(launched_at_utc, "v3 final launch time"),
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "bank_receipt": _reference(bank_receipt_path, BANK_RECEIPT_SCHEMA),
        "one_launch_only": True,
        "upload_authorized": False,
    }
    if path.exists():
        existing = qualification.load_sealed(path, CONSUMPTION_SCHEMA)
        static_existing = {k: v for k, v in existing.items()
                           if k not in {"body_sha256", "launched_at_utc"}}
        static_body = {k: v for k, v in body.items() if k != "launched_at_utc"}
        if static_existing != static_body:
            raise BridgeError("v3 final consumption marker changed")
    else:
        qualification.write_sealed(path, body)
    return path


def _deployment_configuration(plan: Mapping[str, Any]) -> dict[str, Any]:
    inputs = plan.get("inputs")
    planned = plan.get("configuration")
    if not isinstance(inputs, Mapping) or not isinstance(planned, Mapping):
        raise BridgeError("strict final deployment configuration is absent")
    try:
        expected = deployment.deployment_configuration(
            inputs["tuple"], inputs["profile"], inputs["profile_work"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BridgeError("strict final deployment configuration changed") from error
    derivation = inputs.get("deployment_derivation")
    candidate = inputs.get("candidate")
    generated = inputs.get("generated_source")
    if (
        not isinstance(derivation, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(generated, Mapping)
        or derivation.get("schema") != deployment.DERIVATION_SCHEMA
        or derivation.get("configuration") != expected
        or derivation.get("deployed_source")
        != {
            key: candidate.get(key) for key in ("bytes", "sha256", "ascii")
        }
        or derivation.get("base_source")
        != {
            key: generated.get(key) for key in ("bytes", "sha256", "ascii")
        }
        or planned.get("deployment") != expected
        or planned.get("candidate_actions") != expected["candidate_actions"]
        or planned.get("candidate_expansions") != expected["candidate_expansions"]
        or planned.get("candidate_seed") != expected["candidate_shuffle_seed"]
    ):
        raise BridgeError("strict final source/configuration binding changed")
    return expected


def _expected_gate_configuration(
    plan: Mapping[str, Any], *, pair_offset: int, pair_count: int,
) -> dict[str, Any]:
    configured = _deployment_configuration(plan)
    return {
        "mode": "actual-clock",
        "pair_offset": pair_offset,
        "pair_count": pair_count,
        "candidate_c": configured["candidate_c"],
        "candidate_fpu": configured["candidate_fpu"],
        "candidate_lambda": configured["candidate_lambda"],
        "candidate_actions": configured["candidate_actions"],
        "candidate_root_partial_paths": configured[
            "candidate_root_partial_paths"
        ],
        "candidate_nonroot_partial_paths": configured[
            "candidate_nonroot_partial_paths"
        ],
        "candidate_nodes": configured["candidate_nodes"],
        "candidate_expansions": configured["candidate_expansions"],
        "candidate_shuffle_seed": configured["candidate_shuffle_seed"],
        "candidate_clocks_ms": [800, 155],
        "rank4_nodes": plan["configuration"]["rank4_nodes"],
        "rank4_clocks_ms": [800, 165],
        "max_turns": plan["configuration"]["maximum_turns"],
        "minimum_candidate_wins": -1,
        "minimum_wins_per_color": -1,
    }


def gate_command(plan: Mapping[str, Any], bank: Mapping[str, Any], index: int,
                 output: pathlib.Path) -> list[str]:
    configured = _deployment_configuration(plan)
    search_tuple = configured["tuple"]
    return [
        plan["inputs"]["gate"]["path"],
        "--bank", bank["gate_bank"]["path"],
        "--expected-bank-sha256", bank["gate_bank"]["sha256"],
        "--candidate-source", plan["inputs"]["candidate"]["path"],
        "--expected-candidate-sha256", plan["inputs"]["candidate"]["sha256"],
        "--rank4-source", plan["inputs"]["rank4"]["path"],
        "--pair-offset", str(index * 5), "--pair-count", "5",
        "--mode", "actual-clock",
        "--candidate-c", str(search_tuple[0]),
        "--candidate-fpu", str(search_tuple[1]),
        "--candidate-lambda", str(search_tuple[2]),
        "--candidate-actions", str(configured["candidate_actions"]),
        "--candidate-root-partial-paths",
        str(configured["candidate_root_partial_paths"]),
        "--candidate-nonroot-partial-paths",
        str(configured["candidate_nonroot_partial_paths"]),
        "--candidate-nodes", str(configured["candidate_nodes"]),
        "--candidate-expansions", str(configured["candidate_expansions"]),
        "--candidate-seed", str(configured["candidate_shuffle_seed"]),
        "--rank4-nodes", str(plan["configuration"]["rank4_nodes"]),
        "--max-turns", str(plan["configuration"]["maximum_turns"]),
        "--output", str(output),
    ]


def run_gate_process(spec: Mapping[str, Any]) -> pathlib.Path:
    output = pathlib.Path(spec["raw_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        spec["command"], cwd=spec["repository"], stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, check=False, timeout=3_600,
    )
    if completed.returncode not in (0, 2) or not output.is_file():
        raise BridgeError("v3 Rank-4 gate shard failed without complete output")
    return output


def adapt_gate_result(raw: Any, *, plan: Mapping[str, Any],
                      bank: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    path = pathlib.Path(raw)
    document = gate_support.validate_result(
        path, expected_bank_sha256=bank["gate_bank"]["sha256"],
        expected_candidate_sha256=plan["inputs"]["candidate"]["sha256"],
    )
    runtime_path = pathlib.Path(plan["inputs"]["runtime"]["path"])
    if _record(runtime_path, ascii_required=True) != plan["inputs"]["runtime"]:
        raise BridgeError("v3 runtime changed during strict final")
    runtime = json.loads(runtime_path.read_bytes())
    if (
        document.get("bindings", {}).get("candidate_runtime_body_sha256")
        != runtime.get("body_sha256")
        or document.get("bindings", {}).get("candidate_payload_sha256")
        != runtime.get("quantization", {}).get("payload_sha256")
        or document.get("config") != _expected_gate_configuration(
            plan, pair_offset=index * 5, pair_count=5
        )
    ):
        raise BridgeError("v3 gate result runtime/configuration binding changed")
    games = []
    for game in document["games"]:
        failure = game["failure"]
        games.append({
            "pair_index": game["pair_index"],
            "candidate_color": game["candidate_player"],
            "candidate_win": failure is None
            and game["winner"] == game["candidate_player"],
            "turns": max(1, game["turns"]),
            "failure": None if failure is None else legacy_final.FAILURE_MAP[failure],
            "first_ms": game["candidate"]["maximum_first_ms"],
            "later_max_ms": game["candidate"]["maximum_later_ms"],
        })
    return games


def _audit_shards(
    ledger: pathlib.Path, binding_path: pathlib.Path, *,
    plan: Mapping[str, Any], bank: Mapping[str, Any],
    result_adapter: Callable[..., list[dict[str, Any]]],
) -> list[int]:
    missing = []
    for index in range(100):
        claim = ledger / "claims" / f"shard-{index:03d}.json"
        receipt = ledger / "receipts" / f"shard-{index:03d}.json"
        if claim.exists():
            if not receipt.exists():
                raise qualification.SpentShardError(
                    f"v3 final shard {index} is spent without receipt"
                )
            validated = qualification.validate_shard_receipt(
                receipt, binding_path=binding_path, index=index
            )
            evidence_ref = validated.get("evidence")
            if not isinstance(evidence_ref, Mapping):
                raise BridgeError("v3 shard raw evidence reference is absent")
            evidence_path = pathlib.Path(str(evidence_ref.get("path", "")))
            evidence = qualification.load_sealed(
                evidence_path, RAW_EVIDENCE_SCHEMA
            )
            if set(evidence) != {
                "schema", "namespace", "plan", "bank_receipt", "shard_index",
                "raw_gate_result", "normalized_games_sha256", "body_sha256",
            } or evidence.get("namespace") != NAMESPACE:
                raise BridgeError("v3 shard raw evidence field roster changed")
            evidence_plan_path = _verify_reference(
                evidence.get("plan"), PLAN_SCHEMA, "v3 raw evidence plan"
            )
            evidence_bank_path = _verify_reference(
                evidence.get("bank_receipt"), BANK_RECEIPT_SCHEMA,
                "v3 raw evidence bank receipt",
            )
            raw_record = evidence.get("raw_gate_result")
            if (
                evidence_ref != _reference(evidence_path, RAW_EVIDENCE_SCHEMA)
                or qualification.load_sealed(evidence_plan_path, PLAN_SCHEMA) != plan
                or qualification.load_sealed(
                    evidence_bank_path, BANK_RECEIPT_SCHEMA
                ) != bank
                or evidence.get("shard_index") != index
                or not isinstance(raw_record, Mapping)
            ):
                raise BridgeError("v3 shard raw evidence changed")
            raw_path = _verify_record(raw_record, "v3 raw gate result")
            adapted = result_adapter(
                raw_path, plan=plan, bank=bank, index=index
            )
            if (
                adapted != validated.get("games")
                or evidence.get("normalized_games_sha256")
                != qualification.sha256_bytes(
                    qualification.canonical_json_bytes(adapted)
                )
            ):
                raise BridgeError("v3 raw gate result differs from shard receipt")
        elif receipt.exists():
            raise BridgeError("v3 final shard receipt exists without claim")
        else:
            missing.append(index)
    return missing


def run_final(
    *, plan_path: pathlib.Path, bank_receipt_path: pathlib.Path,
    campaign_root: pathlib.Path, development_plan_path: pathlib.Path,
    finalist_reference_path: pathlib.Path,
    preflight_path: pathlib.Path, candidate_source: pathlib.Path,
    historical_paths: Sequence[pathlib.Path], rank4_source: pathlib.Path,
    gate_path: pathlib.Path, repository: pathlib.Path, launched_at_utc: str,
    maximum_workers: int = 4,
    finalist_validator: FinalistValidator = _default_finalist_validator,
    preflight_validator: PreflightValidator = _default_preflight_validator,
    historical_validator: HistoricalValidator = _default_historical_validator,
    fingerprint_loader: FingerprintLoader = _default_fingerprint_loader,
    git_verifier: GitVerifier = legacy_final.verify_clean_git,
    runner: Callable[[Mapping[str, Any]], Any] = run_gate_process,
    result_adapter: Callable[..., list[dict[str, Any]]] = adapt_gate_result,
    clock: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    if maximum_workers != 4:
        raise BridgeError("v3 strict final requires exactly four workers")
    bank_state = validate_bank_receipt(
        bank_receipt_path, plan_path=plan_path, campaign_root=campaign_root,
        development_plan_path=development_plan_path,
        finalist_reference_path=finalist_reference_path,
        preflight_path=preflight_path,
        candidate_source=candidate_source, historical_paths=historical_paths,
        rank4_source=rank4_source, gate_path=gate_path, repository=repository,
        finalist_validator=finalist_validator,
        preflight_validator=preflight_validator,
        historical_validator=historical_validator,
        fingerprint_loader=fingerprint_loader, git_verifier=git_verifier,
    )
    plan = bank_state["plan"]
    ledger = _safe_directory(pathlib.Path(plan["paths"]["ledger"]), create=True)
    for name in ("claims", "receipts", "raw", "raw-evidence"):
        _safe_directory(ledger / name, create=True)
    _safe_output(ledger / "aggregate.json")
    _safe_output(ledger / "v3-qualified-inputs.json")
    consume_bank(plan_path, plan, bank_receipt_path, launched_at_utc)
    binding_path = bank_state["gate_binding_path"]
    missing = _audit_shards(
        ledger, binding_path, plan=plan, bank=bank_state["receipt"],
        result_adapter=result_adapter,
    )
    aggregate_path = ledger / "aggregate.json"
    qualified_path = ledger / "v3-qualified-inputs.json"
    if aggregate_path.exists():
        aggregate = qualification.load_sealed(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        )
        if missing:
            raise BridgeError("v3 aggregate exists with missing shards")
        if aggregate.get("verdict") != qualification.strict_gate_verdict(
            aggregate.get("summary", {})
        ):
            raise BridgeError("v3 aggregate verdict changed")
        if aggregate["verdict"]["passed"]:
            validate_qualified(
                qualified_path, plan_path=plan_path,
                bank_receipt_path=bank_receipt_path,
                aggregate_path=aggregate_path,
            )
        elif qualified_path.exists():
            raise BridgeError("failed v3 aggregate has qualified inputs")
        return aggregate
    def one(index: int) -> int:
        qualification.start_final_shard(
            ledger, binding_path=binding_path, index=index,
            started_at_utc=clock(),
        )
        raw_path = ledger / "raw" / f"shard-{index:03d}.json"
        spec = {
            "index": index, "repository": str(repository.resolve()),
            "raw_output": str(raw_path),
            "command": gate_command(plan, bank_state["receipt"], index, raw_path),
        }
        raw = runner(spec)
        games = result_adapter(
            raw, plan=plan, bank=bank_state["receipt"], index=index
        )
        if isinstance(raw, (str, os.PathLike, pathlib.Path)) and pathlib.Path(raw).is_file():
            raw_file = pathlib.Path(raw)
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            qualification.atomic_write_once(
                raw_path,
                qualification.canonical_json_bytes({"injected_raw": raw}),
            )
            raw_file = raw_path
        evidence_path = ledger / "raw-evidence" / f"shard-{index:03d}.json"
        qualification.write_sealed(evidence_path, {
            "schema": RAW_EVIDENCE_SCHEMA,
            "namespace": NAMESPACE,
            "plan": _reference(plan_path, PLAN_SCHEMA),
            "bank_receipt": _reference(bank_receipt_path, BANK_RECEIPT_SCHEMA),
            "shard_index": index,
            "raw_gate_result": _record(raw_file),
            "normalized_games_sha256": qualification.sha256_bytes(
                qualification.canonical_json_bytes(games)
            ),
        })
        qualification.record_shard_receipt(
            ledger, binding_path=binding_path, index=index, games=games,
            completed_at_utc=clock(),
            evidence=_reference(evidence_path, RAW_EVIDENCE_SCHEMA),
        )
        return index

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, index) for index in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    aggregate = qualification.aggregate_final(
        ledger, binding_path=binding_path,
        uncontended_timing=plan["inputs"]["uncontended_timing"],
        completed_at_utc=clock(),
    )
    if aggregate["verdict"]["passed"]:
        qualification.write_sealed(qualified_path, {
            "schema": QUALIFIED_SCHEMA,
            "namespace": NAMESPACE,
            "campaign_id": CAMPAIGN_ID,
            "status": "v3-rank4-qualified-awaiting-green-ci",
            "candidate_commit": plan["inputs"]["candidate_commit"],
            "candidate": plan["inputs"]["candidate"],
            "runtime": plan["inputs"]["runtime"],
            "deployment_derivation": plan["inputs"]["deployment_derivation"],
            "deployment_manifest": plan["inputs"]["deployment_manifest"],
            "deployment_manifest_body_sha256": plan["inputs"]
            ["deployment_manifest_body_sha256"],
            "development_plan": plan["inputs"]["development_plan"],
            "finalist_reference": plan["inputs"]["finalist_reference"],
            "finalist": plan["inputs"]["finalist"],
            "handoff": plan["inputs"]["handoff"],
            "evaluation_completion": plan["inputs"]["evaluation_completion"],
            "exclusion_receipt": plan["inputs"]["exclusion_receipt"],
            "plan": _reference(plan_path, PLAN_SCHEMA),
            "bank_receipt": _reference(bank_receipt_path, BANK_RECEIPT_SCHEMA),
            "aggregate": _reference(
                aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
            ),
            "preflight": plan["inputs"]["preflight"],
            "strict_thresholds": aggregate["verdict"]["thresholds"],
            "uploads_authorized": 0,
            "rank4_replacement_authorized": False,
        })
        validate_qualified(
            qualified_path, plan_path=plan_path,
            bank_receipt_path=bank_receipt_path,
            aggregate_path=aggregate_path,
        )
    return aggregate


def validate_qualified(path: pathlib.Path, *, plan_path: pathlib.Path,
                       bank_receipt_path: pathlib.Path,
                       aggregate_path: pathlib.Path) -> dict[str, Any]:
    value = qualification.load_sealed(path, QUALIFIED_SCHEMA)
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    expected_ledger = pathlib.Path(plan["paths"]["ledger"])
    if (
        path.absolute() != expected_ledger / "v3-qualified-inputs.json"
        or aggregate_path.absolute() != expected_ledger / "aggregate.json"
        or bank_receipt_path.absolute()
        != pathlib.Path(plan["paths"]["bank_receipt"])
        or path.is_symlink() or aggregate_path.is_symlink()
    ):
        raise BridgeError("v3 qualified input path chain changed")
    aggregate = qualification.load_sealed(
        aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
    )
    bank_receipt = qualification.load_sealed(
        bank_receipt_path, BANK_RECEIPT_SCHEMA
    )
    _verify_record(plan["inputs"]["candidate"], "qualified candidate", ascii_required=True)
    _verify_record(plan["inputs"]["runtime"], "qualified runtime", ascii_required=True)
    manifest_path = _verify_record(
        plan["inputs"]["deployment_manifest"], "qualified deployment manifest",
        ascii_required=True,
    )
    manifest = deployment.verify_manifest_file(
        manifest_path, pathlib.Path(plan["inputs"]["candidate"]["path"])
    )
    if (
        manifest.get("body_sha256")
        != plan["inputs"]["deployment_manifest_body_sha256"]
        or manifest.get("base_source")
        != plan["inputs"]["deployment_derivation"]["base_source"]
        or manifest.get("deployed_source")
        != plan["inputs"]["deployment_derivation"]["deployed_source"]
        or manifest.get("configuration")
        != plan["inputs"]["deployment_derivation"]["configuration"]
    ):
        raise BridgeError("qualified deployment manifest changed")
    development._verify_sealed_record(
        plan["inputs"]["finalist"], development.FINALIST_SCHEMA,
        "qualified finalist",
    )
    _verify_reference(
        plan["inputs"]["finalist_reference"],
        development.FINALIST_REFERENCE_SCHEMA, "qualified finalist reference",
    )
    _verify_reference(
        plan["inputs"]["development_plan"], development.PLAN_SCHEMA,
        "qualified development plan",
    )
    _verify_reference(
        plan["inputs"]["evaluation_completion"], adapter.EVALUATION_COMPLETION_SCHEMA,
        "qualified evaluation completion",
    )
    development._verify_sealed_record(
        plan["inputs"]["exclusion_receipt"], exclusions.RECEIPT_SCHEMA,
        "qualified exclusion receipt",
    )
    expected = qualification.seal({
        "schema": QUALIFIED_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "v3-rank4-qualified-awaiting-green-ci",
        "candidate_commit": plan["inputs"]["candidate_commit"],
        "candidate": plan["inputs"]["candidate"],
        "runtime": plan["inputs"]["runtime"],
        "deployment_derivation": plan["inputs"]["deployment_derivation"],
        "deployment_manifest": plan["inputs"]["deployment_manifest"],
        "deployment_manifest_body_sha256": plan["inputs"]
        ["deployment_manifest_body_sha256"],
        "development_plan": plan["inputs"]["development_plan"],
        "finalist_reference": plan["inputs"]["finalist_reference"],
        "finalist": plan["inputs"]["finalist"],
        "handoff": plan["inputs"]["handoff"],
        "evaluation_completion": plan["inputs"]["evaluation_completion"],
        "exclusion_receipt": plan["inputs"]["exclusion_receipt"],
        "plan": _reference(plan_path, PLAN_SCHEMA),
        "bank_receipt": _reference(bank_receipt_path, BANK_RECEIPT_SCHEMA),
        "aggregate": _reference(aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA),
        "preflight": plan["inputs"]["preflight"],
        "strict_thresholds": aggregate["verdict"]["thresholds"],
        "uploads_authorized": 0,
        "rank4_replacement_authorized": False,
    })
    if (
        aggregate.get("verdict") != qualification.strict_gate_verdict(
            aggregate.get("summary", {})
        )
        or aggregate.get("binding") != bank_receipt.get("gate_binding")
        or aggregate.get("verdict", {}).get("passed") is not True
        or value != expected
    ):
        raise BridgeError("v3 qualified input chain changed")
    return value


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-root", type=pathlib.Path, required=True)
    parser.add_argument("--development-plan", type=pathlib.Path, required=True)
    parser.add_argument("--finalist-reference", type=pathlib.Path, required=True)
    parser.add_argument("--preflight", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-source", type=pathlib.Path, required=True)
    parser.add_argument("--historical", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--rank4-source", type=pathlib.Path, required=True)
    parser.add_argument("--gate", type=pathlib.Path, required=True)
    parser.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    _common_arguments(prepare_command)
    prepare_command.add_argument("--authorized-at-utc", default=utc_now())
    prepare_command.add_argument("--planned-at-utc", default=utc_now())
    materialize_command = commands.add_parser("materialize")
    _common_arguments(materialize_command)
    materialize_command.add_argument("--plan", type=pathlib.Path, required=True)
    materialize_command.add_argument("--claimed-at-utc", default=utc_now())
    run_command = commands.add_parser("run")
    _common_arguments(run_command)
    run_command.add_argument("--plan", type=pathlib.Path, required=True)
    run_command.add_argument("--bank-receipt", type=pathlib.Path, required=True)
    run_command.add_argument("--launched-at-utc", default=utc_now())
    run_command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    common = {
        "campaign_root": args.campaign_root,
        "development_plan_path": args.development_plan,
        "finalist_reference_path": args.finalist_reference,
        "preflight_path": args.preflight,
        "candidate_source": args.candidate_source,
        "historical_paths": args.historical,
        "rank4_source": args.rank4_source,
        "gate_path": args.gate,
        "repository": args.repository,
    }
    try:
        if args.command == "prepare":
            path = prepare(
                **common, authorized_at_utc=args.authorized_at_utc,
                planned_at_utc=args.planned_at_utc,
            )
            result: Any = {"plan": str(path), "sha256": qualification.sha256_file(path)}
        elif args.command == "materialize":
            path = materialize_bank(
                **common, plan_path=args.plan, claimed_at_utc=args.claimed_at_utc,
                entropy=__import__("secrets").token_bytes,
            )
            result = {"bank_receipt": str(path), "sha256": qualification.sha256_file(path)}
        else:
            result = run_final(
                **common, plan_path=args.plan,
                bank_receipt_path=args.bank_receipt,
                launched_at_utc=args.launched_at_utc,
                maximum_workers=args.workers,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (BridgeError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact discrete-v3 final failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
