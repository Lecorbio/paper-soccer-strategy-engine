#!/usr/bin/env python3
"""Authorize and verify the one-upload/live lifecycle for discrete-v3.

This is a standalone bridge.  It does not alter or reinterpret any of the
immutable training, adapter, development, exclusion, preflight, or strict
Rank-4 artifacts.  Authorization is possible only after independently
revalidating the complete v3-qualified chain, a clean committed branch HEAD,
and the exact green GitHub workflow.  The upload lifecycle is append-only and
records one Submit attempt; ambiguous state can only be resolved by a unique
history/API identity and can never authorize another click.  The final stage
delegates collection to the maintained exact-90 live-window implementation.

No command in this module performs a browser click, pushes Git, starts CI, or
uploads source.  Those external actions remain explicit operator steps between
the sealed lifecycle events.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
TEST_PATH = (
    REPOSITORY
    / "tests/codingame/test_compact_value_bfm_discrete_v3_release.py"
)


def _load(path: pathlib.Path, name: str) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 release dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_v3_release_qualification"
)
final_bridge = _load(
    HERE / "compact_value_bfm_discrete_v3_final.py", "compact_v3_release_final"
)
development = _load(
    HERE / "compact_value_bfm_discrete_v3_development.py",
    "compact_v3_release_development",
)
adapter = _load(
    HERE / "compact_value_bfm_discrete_v3_adapter.py", "compact_v3_release_adapter"
)
exclusions = _load(
    HERE / "compact_value_bfm_discrete_v3_exclusions.py",
    "compact_v3_release_exclusions",
)
upload_primitives = _load(
    HERE / "compact_value_bfm_upload.py", "compact_v3_release_upload_primitives"
)
live = _load(
    REPOSITORY / "submissions/codingame/bots/compact_value_bfm/live_window.py",
    "compact_v3_release_live_window",
)


ReleaseError = qualification.QualificationError
NAMESPACE = qualification.NAMESPACE
CAMPAIGN_ID = final_bridge.CAMPAIGN_ID
RELEASE_DIRECTORY = "discrete-v3-release"

AUTH_INPUT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-release-authorization-inputs.v1"
)
FRESH_EDITOR_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-release-fresh-editor.v1"
)
COMPLETION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-release-completion.v1"
)
PUBLICATION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-publication.v1"
)

QUALIFIED_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "candidate_commit",
    "candidate", "runtime", "deployment_derivation", "deployment_manifest",
    "deployment_manifest_body_sha256", "development_plan", "finalist_reference",
    "finalist", "handoff", "evaluation_completion", "exclusion_receipt",
    "plan", "bank_receipt", "aggregate", "preflight", "strict_thresholds",
    "uploads_authorized", "rank4_replacement_authorized", "body_sha256",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> dt.datetime:
    return upload_primitives.parse_utc(value, label)


def _safe_root(path: pathlib.Path) -> pathlib.Path:
    try:
        return exclusions._safe_output_root(path)
    except Exception as error:
        raise ReleaseError(f"unsafe discrete-v3 release root: {error}") from error


def _safe_directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    try:
        return exclusions._safe_directory(path, create=create)
    except Exception as error:
        raise ReleaseError(f"unsafe discrete-v3 release directory: {error}") from error


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    try:
        return exclusions._safe_output_file(path)
    except Exception as error:
        raise ReleaseError(f"unsafe discrete-v3 release output: {error}") from error


def release_root(campaign_root: pathlib.Path, *, create: bool) -> pathlib.Path:
    campaign_root = _safe_root(campaign_root)
    return _safe_directory(campaign_root / RELEASE_DIRECTORY, create=create)


def _fixed_output(root: pathlib.Path, relative: str) -> pathlib.Path:
    root = _safe_directory(root, create=False)
    path = _safe_output(root / relative)
    try:
        path.absolute().relative_to(root)
    except ValueError as error:
        raise ReleaseError(f"fixed release output escaped its root: {relative}") from error
    return path


def _fixed_file(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    path = _fixed_output(root, relative)
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"{label} is absent, redirected, or irregular")
    return path


def _fixed_directory(
    root: pathlib.Path, relative: str, *, create: bool, label: str
) -> pathlib.Path:
    root = _safe_directory(root, create=False)
    path = _safe_directory(root / relative, create=create)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReleaseError(f"{label} escaped the fixed release root") from error
    return path


def _validate_regular_tree(path: pathlib.Path, label: str) -> pathlib.Path:
    path = _safe_directory(path, create=True)
    for child in path.rglob("*"):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise ReleaseError(f"{label} contains a redirected/irregular node: {child}")
        try:
            child.resolve().relative_to(path)
        except ValueError as error:
            raise ReleaseError(f"{label} contains an external path: {child}") from error
    return path


def _reject_foreign_upload_state(campaign_root: pathlib.Path) -> None:
    campaign_root = campaign_root.resolve()
    final_ledger = campaign_root / final_bridge.BRIDGE_DIRECTORY / "ledger"
    paths = (
        campaign_root / "upload-authorization",
        campaign_root / "upload",
        campaign_root / "completion.json",
        final_ledger / "upload-authorization",
        final_ledger / "upload",
        final_ledger / "completion.json",
    )
    observed = [path for path in paths if path.exists() or path.is_symlink()]
    if observed:
        raise ReleaseError(
            "foreign/legacy upload state would violate exactly-one authorization: "
            + ", ".join(str(path) for path in observed)
        )


def _record(path: pathlib.Path, *, ascii_required: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"required release artifact is absent or redirected: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ReleaseError(f"release source is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        **({"ascii": True} if ascii_required else {}),
    }


def _verify_record(
    value: Any, label: str, *, ascii_required: bool = False
) -> pathlib.Path:
    keys = {"path", "bytes", "sha256"} | ({"ascii"} if ascii_required else set())
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReleaseError(f"{label} record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _record(path, ascii_required=ascii_required) != dict(value):
        raise ReleaseError(f"{label} changed")
    return path.resolve()


def _reference(path: pathlib.Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"sealed release artifact is absent or redirected: {path}")
    return qualification.artifact_reference(path, schema)


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ReleaseError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if dict(value) != _reference(path, schema):
        raise ReleaseError(f"{label} reference changed")
    return path.resolve()


def _tool_closure() -> dict[str, Any]:
    return {
        "release": _record(pathlib.Path(__file__).resolve()),
        "release_tests": _record(TEST_PATH),
        "qualification": _record(pathlib.Path(qualification.__file__).resolve()),
        "final_bridge": _record(pathlib.Path(final_bridge.__file__).resolve()),
        "final_bridge_tests": _record(final_bridge.TEST_PATH),
        "deployment_source": _record(
            pathlib.Path(final_bridge.deployment.__file__).resolve()
        ),
        "deployment_source_tests": _record(
            REPOSITORY
            / "tests/codingame/test_compact_value_bfm_discrete_v3_deployment.py"
        ),
        "deployment_preflight": _record(
            pathlib.Path(final_bridge.deployment_preflight.__file__).resolve()
        ),
        "deployment_preflight_tests": _record(
            final_bridge.deployment_preflight.TEST_PATH
        ),
        "development": _record(pathlib.Path(development.__file__).resolve()),
        "development_tests": _record(development.TEST_PATH),
        "adapter_v2": _record(pathlib.Path(adapter.__file__).resolve()),
        "fresh_exclusions": _record(pathlib.Path(exclusions.__file__).resolve()),
        "upload_primitives": _record(
            pathlib.Path(upload_primitives.__file__).resolve()
        ),
        "live_window": _record(pathlib.Path(live.__file__).resolve()),
        "generic_live_collector": _record(live.GENERIC_COLLECTOR_PATH),
    }


def verify_release_git(
    repository: pathlib.Path, candidate_source: pathlib.Path, commit: str
) -> dict[str, Any]:
    try:
        result = dict(
            final_bridge.legacy_final.verify_clean_git(
                repository, candidate_source, commit
            )
        )
    except Exception as error:
        raise ReleaseError("release Git HEAD/source is not clean and committed") from error
    completed = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repository,
        capture_output=True, check=False,
    )
    branch = completed.stdout.decode("ascii", errors="strict").strip()
    if completed.returncode != 0 or branch != upload_primitives.BRANCH:
        raise ReleaseError("release HEAD is not on the exact CI/upload branch")
    result["branch"] = branch
    return result


QualifiedValidator = Callable[..., Mapping[str, Any]]
GitVerifier = Callable[[pathlib.Path, pathlib.Path, str], Mapping[str, Any]]
HandoffValidator = Callable[..., Mapping[str, Any]]


def _validate_qualified_records(
    qualified: Mapping[str, Any], *, campaign_root: pathlib.Path,
    handoff_validator: HandoffValidator = adapter.validate_handoff,
) -> dict[str, pathlib.Path]:
    plan_path = _verify_reference(
        qualified.get("plan"), final_bridge.PLAN_SCHEMA, "strict-final plan"
    )
    bank_receipt_path = _verify_reference(
        qualified.get("bank_receipt"), final_bridge.BANK_RECEIPT_SCHEMA,
        "strict-final bank receipt",
    )
    aggregate_path = _verify_reference(
        qualified.get("aggregate"), qualification.FINAL_AGGREGATE_SCHEMA,
        "strict-final aggregate",
    )
    development_plan_path = _verify_reference(
        qualified.get("development_plan"), development.PLAN_SCHEMA,
        "development plan",
    )
    finalist_reference_path = _verify_reference(
        qualified.get("finalist_reference"), development.FINALIST_REFERENCE_SCHEMA,
        "development finalist reference",
    )
    finalist_path = development._verify_sealed_record(
        qualified.get("finalist"), development.FINALIST_SCHEMA,
        "development finalist",
    )
    handoff_path = development._verify_sealed_record(
        qualified.get("handoff"), adapter.HANDOFF_SCHEMA, "adapter v2 handoff"
    )
    evaluation_path = _verify_reference(
        qualified.get("evaluation_completion"),
        adapter.EVALUATION_COMPLETION_SCHEMA, "adapter evaluation completion",
    )
    exclusion_path = development._verify_sealed_record(
        qualified.get("exclusion_receipt"), exclusions.RECEIPT_SCHEMA,
        "fresh-position exclusion receipt",
    )
    handoff = qualification.load_sealed(handoff_path, adapter.HANDOFF_SCHEMA)
    adapter_plan_path = _verify_reference(
        handoff.get("adapter_plan"), adapter.ADAPTER_PLAN_SCHEMA,
        "adapter v2 plan",
    )
    v3_plan_path = _verify_reference(
        handoff.get("v3_plan"), adapter.v3.PLAN_SCHEMA, "v3 plan"
    )
    if handoff.get("evaluation_completion") != qualified.get(
        "evaluation_completion"
    ):
        raise ReleaseError("qualified input and handoff use different evaluations")
    try:
        validated = handoff_validator(
            handoff_path, adapter_plan_path=adapter_plan_path,
            plan_path=v3_plan_path, output_root=campaign_root,
            evaluation_completion_path=evaluation_path,
        )
    except Exception as error:
        raise ReleaseError("adapter v2 handoff ancestry validation failed") from error
    if dict(validated) != handoff:
        raise ReleaseError("adapter v2 handoff validator returned different content")
    return {
        "plan": plan_path,
        "bank_receipt": bank_receipt_path,
        "aggregate": aggregate_path,
        "development_plan": development_plan_path,
        "finalist_reference": finalist_reference_path,
        "finalist": finalist_path,
        "handoff": handoff_path,
        "evaluation": evaluation_path,
        "exclusion": exclusion_path,
    }


def _validate_deployment_binding(
    qualified: Mapping[str, Any], plan: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ReleaseError("strict-final plan has no deployment inputs")
    derivation = inputs.get("deployment_derivation")
    generated = inputs.get("generated_source")
    candidate = qualified.get("candidate")
    manifest_record = qualified.get("deployment_manifest")
    try:
        configured = final_bridge.deployment.deployment_configuration(
            inputs["tuple"], inputs["profile"], inputs["profile_work"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseError("strict-final deployment configuration is invalid") from error
    if (
        not isinstance(derivation, Mapping)
        or not isinstance(generated, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(manifest_record, Mapping)
        or manifest_record != inputs.get("deployment_manifest")
        or qualified.get("deployment_manifest_body_sha256")
        != inputs.get("deployment_manifest_body_sha256")
        or derivation.get("schema")
        != final_bridge.deployment.DERIVATION_SCHEMA
        or derivation.get("configuration") != configured
        or derivation.get("deployed_source")
        != {
            key: candidate.get(key) for key in ("bytes", "sha256", "ascii")
        }
        or derivation.get("base_source")
        != {
            key: generated.get(key) for key in ("bytes", "sha256", "ascii")
        }
        or plan.get("configuration", {}).get("deployment") != configured
    ):
        raise ReleaseError("qualified source is not the finalist deployment derivative")
    manifest_path = _verify_record(
        manifest_record, "qualified deployment manifest", ascii_required=True
    )
    candidate_path = pathlib.Path(str(candidate.get("path", "")))
    try:
        manifest = final_bridge.deployment.verify_manifest_file(
            manifest_path, candidate_path
        )
    except Exception as error:
        raise ReleaseError("qualified deployment manifest did not validate") from error
    if (
        manifest.get("body_sha256")
        != qualified.get("deployment_manifest_body_sha256")
        or manifest.get("base_source") != derivation.get("base_source")
        or manifest.get("deployed_source") != derivation.get("deployed_source")
        or manifest.get("configuration") != configured
    ):
        raise ReleaseError("qualified deployment manifest binding changed")
    return configured


def validate_qualified_chain(
    campaign_root: pathlib.Path, qualified_path: pathlib.Path,
    repository: pathlib.Path, *, git_verifier: GitVerifier = verify_release_git,
) -> dict[str, Any]:
    """Independently revalidate the complete strict-final v3 chain."""

    campaign_root = _safe_root(campaign_root)
    expected_qualified = (
        campaign_root / final_bridge.BRIDGE_DIRECTORY
        / "ledger/v3-qualified-inputs.json"
    )
    if (
        qualified_path.absolute() != expected_qualified
        or qualified_path.is_symlink() or not qualified_path.is_file()
    ):
        raise ReleaseError("v3 qualified input path is not canonical")
    qualified = qualification.load_sealed(
        qualified_path, final_bridge.QUALIFIED_SCHEMA
    )
    if (
        set(qualified) != QUALIFIED_FIELDS
        or qualified.get("namespace") != NAMESPACE
        or qualified.get("campaign_id") != CAMPAIGN_ID
        or qualified.get("status") != "v3-rank4-qualified-awaiting-green-ci"
        or qualified.get("uploads_authorized") != 0
        or qualified.get("rank4_replacement_authorized") is not False
    ):
        raise ReleaseError("v3 qualified field/policy contract changed")

    records = _validate_qualified_records(
        qualified, campaign_root=campaign_root
    )
    plan_path = records["plan"]
    bank_receipt_path = records["bank_receipt"]
    aggregate_path = records["aggregate"]
    development_plan_path = records["development_plan"]
    finalist_reference_path = records["finalist_reference"]

    plan = qualification.load_sealed(plan_path, final_bridge.PLAN_SCHEMA)
    if (
        qualified.get("candidate_commit") != plan.get("inputs", {}).get(
            "candidate_commit"
        )
        or qualified.get("candidate") != plan.get("inputs", {}).get("candidate")
        or qualified.get("runtime") != plan.get("inputs", {}).get("runtime")
        or qualified.get("deployment_derivation")
        != plan.get("inputs", {}).get("deployment_derivation")
        or qualified.get("deployment_manifest")
        != plan.get("inputs", {}).get("deployment_manifest")
        or qualified.get("deployment_manifest_body_sha256")
        != plan.get("inputs", {}).get("deployment_manifest_body_sha256")
        or qualified.get("preflight") != plan.get("inputs", {}).get("preflight")
    ):
        raise ReleaseError("qualified inputs differ from their strict-final plan")
    candidate_path = _verify_record(
        qualified["candidate"], "qualified candidate", ascii_required=True
    )
    runtime_path = _verify_record(
        qualified["runtime"], "qualified runtime", ascii_required=True
    )
    if not 0 < qualified["candidate"]["bytes"] < 95_000:
        raise ReleaseError("qualified source exceeds the CodinGame source limit")

    inputs = plan.get("inputs")
    paths = plan.get("paths")
    if not isinstance(inputs, Mapping) or not isinstance(paths, Mapping):
        raise ReleaseError("strict-final plan has no input/path closure")
    _validate_deployment_binding(qualified, plan)
    preflight_path = _verify_reference(
        qualified["preflight"],
        final_bridge.deployment_preflight.REFERENCE_SCHEMA,
        "source-bound preflight",
    )
    historical_paths = [
        _verify_record(record, f"historical exclusion {index}")
        for index, record in enumerate(inputs.get("historical_exclusions", []))
    ]
    if len(historical_paths) != 7:
        raise ReleaseError("strict-final plan does not bind seven historical banks")
    rank4_path = _verify_record(inputs.get("rank4"), "maintained Rank-4 source",
                                ascii_required=True)
    try:
        gate_path = final_bridge._verify_record(
            inputs.get("gate"), "strict-final gate executable", executable=True
        )
    except Exception as error:
        raise ReleaseError("strict-final gate executable changed") from error

    git = dict(git_verifier(repository, candidate_path, qualified["candidate_commit"]))
    if (
        git.get("commit") != qualified["candidate_commit"]
        or git.get("tracked_clean") is not True
    ):
        raise ReleaseError("release Git verifier did not bind clean HEAD")

    try:
        bank_state = final_bridge.validate_bank_receipt(
            bank_receipt_path, plan_path=plan_path, campaign_root=campaign_root,
            development_plan_path=development_plan_path,
            finalist_reference_path=finalist_reference_path,
            preflight_path=preflight_path, candidate_source=candidate_path,
            historical_paths=historical_paths, rank4_source=rank4_path,
            gate_path=gate_path, repository=repository, git_verifier=git_verifier,
        )
        final_bridge.validate_qualified(
            qualified_path, plan_path=plan_path,
            bank_receipt_path=bank_receipt_path, aggregate_path=aggregate_path,
        )
    except Exception as error:
        raise ReleaseError("deep v3 strict-final ancestry validation failed") from error

    binding_path = pathlib.Path(bank_state["gate_binding_path"])
    ledger = pathlib.Path(str(paths.get("ledger", "")))
    if ledger != aggregate_path.parent or ledger.is_symlink() or not ledger.is_dir():
        raise ReleaseError("strict-final ledger path changed")
    missing = final_bridge._audit_shards(
        ledger, binding_path, plan=plan, bank=bank_state["receipt"],
        result_adapter=final_bridge.adapt_gate_result,
    )
    if missing:
        raise ReleaseError("qualified strict final has missing raw-bound shards")
    aggregate = qualification.load_sealed(
        aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
    )
    if (
        aggregate.get("verdict")
        != qualification.strict_gate_verdict(aggregate.get("summary", {}))
        or aggregate.get("verdict", {}).get("passed") is not True
        or aggregate.get("status") != "rank4-qualified"
        or aggregate.get("binding")
        != _reference(binding_path, qualification.GATE_BINDING_SCHEMA)
    ):
        raise ReleaseError("v3 strict-final aggregate is not an exact pass")

    consumption_path = ledger / "consumption.json"
    consumption = qualification.load_sealed(
        consumption_path, final_bridge.CONSUMPTION_SCHEMA
    )
    if (
        consumption.get("status") != "v3-final-bank-consumed-at-launch"
        or consumption.get("plan")
        != _reference(plan_path, final_bridge.PLAN_SCHEMA)
        or consumption.get("bank_receipt")
        != _reference(bank_receipt_path, final_bridge.BANK_RECEIPT_SCHEMA)
        or consumption.get("one_launch_only") is not True
        or consumption.get("upload_authorized") is not False
    ):
        raise ReleaseError("strict-final one-launch consumption changed")

    raw_shards = []
    for index in range(100):
        receipt = ledger / "receipts" / f"shard-{index:03d}.json"
        evidence = ledger / "raw-evidence" / f"shard-{index:03d}.json"
        raw = ledger / "raw" / f"shard-{index:03d}.json"
        raw_shards.append({
            "index": index,
            "receipt_sha256": qualification.sha256_file(receipt),
            "evidence_sha256": qualification.sha256_file(evidence),
            "raw_sha256": qualification.sha256_file(raw),
        })
    return {
        "qualified": qualified,
        "qualified_path": qualified_path.resolve(),
        "plan": plan,
        "plan_path": plan_path,
        "bank_receipt": bank_state["receipt"],
        "bank_receipt_path": bank_receipt_path,
        "aggregate": aggregate,
        "aggregate_path": aggregate_path,
        "consumption": consumption,
        "consumption_path": consumption_path,
        "candidate_path": candidate_path,
        "runtime_path": runtime_path,
        "preflight_path": preflight_path,
        "binding_path": binding_path,
        "git": git,
        "raw_shards": raw_shards,
    }


def freeze_live_exclusions(
    campaign_root: pathlib.Path, *, registry_path: pathlib.Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    root = release_root(campaign_root, create=True)
    output = _fixed_output(root, "live-exclusion-binding.json")
    try:
        if output.exists():
            binding, _registry = live.validate_exclusion_binding(output)
            return binding
        return live.freeze_exclusion_binding(
            output, registry_path=registry_path, frozen_at_utc=frozen_at_utc
        )
    except Exception as error:
        raise ReleaseError("could not freeze the ID-only live exclusion registry") from error


def seal_ci_evidence(
    campaign_root: pathlib.Path, *, gh_payload: Mapping[str, Any],
    expected_head: str, fetched_at_utc: str,
) -> dict[str, Any]:
    root = release_root(campaign_root, create=True)
    path = _fixed_output(root, "github-ci.json")
    return upload_primitives.seal_ci_evidence(
        path, gh_payload=gh_payload, expected_head=expected_head,
        fetched_at_utc=fetched_at_utc,
    )


def _normalized_ci(ci: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": ci["run_id"],
        "repository": ci["repository"],
        "workflow_database_id": ci["workflow_database_id"],
        "attempt": ci["attempt"],
        "head_sha": ci["head_sha"],
        "conclusion": "success",
        "jobs": {
            job_id: "success" for job_id in upload_primitives.REQUIRED_JOB_IDS
        },
        "workflow": upload_primitives.WORKFLOW_NAME,
        "workflow_file": upload_primitives.WORKFLOW_FILE,
        "event": "workflow_dispatch",
        "head_branch": upload_primitives.BRANCH,
        "head_ref": f"refs/heads/{upload_primitives.BRANCH}",
        "url": ci["url"],
    }


def _raw_roster_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return qualification.sha256_bytes(qualification.canonical_json_bytes(list(rows)))


def _authorization_inputs_body(
    chain: Mapping[str, Any], *, authorization_path: pathlib.Path,
    ci_path: pathlib.Path, ci: Mapping[str, Any], exclusion_path: pathlib.Path,
    exclusion: Mapping[str, Any], authorized_at_utc: str,
    root: pathlib.Path,
) -> dict[str, Any]:
    return {
        "schema": AUTH_INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "one-discrete-v3-upload-authorized",
        "authorized_at_utc": authorized_at_utc,
        "release_root": str(root.resolve()),
        "authorization": _reference(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        ),
        "qualified": _reference(
            pathlib.Path(chain["qualified_path"]), final_bridge.QUALIFIED_SCHEMA
        ),
        "strict_final_plan": _reference(
            pathlib.Path(chain["plan_path"]), final_bridge.PLAN_SCHEMA
        ),
        "bank_receipt": _reference(
            pathlib.Path(chain["bank_receipt_path"]),
            final_bridge.BANK_RECEIPT_SCHEMA,
        ),
        "aggregate": _reference(
            pathlib.Path(chain["aggregate_path"]),
            qualification.FINAL_AGGREGATE_SCHEMA,
        ),
        "consumption": _reference(
            pathlib.Path(chain["consumption_path"]),
            final_bridge.CONSUMPTION_SCHEMA,
        ),
        "preflight": _reference(
            pathlib.Path(chain["preflight_path"]),
            final_bridge.deployment_preflight.REFERENCE_SCHEMA,
        ),
        "candidate_commit": chain["qualified"]["candidate_commit"],
        "candidate": dict(chain["qualified"]["candidate"]),
        "runtime": dict(chain["qualified"]["runtime"]),
        "deployment_derivation": dict(
            chain["qualified"]["deployment_derivation"]
        ),
        "deployment_manifest": dict(chain["qualified"]["deployment_manifest"]),
        "deployment_manifest_body_sha256": chain["qualified"][
            "deployment_manifest_body_sha256"
        ],
        "git": dict(chain["git"]),
        "ci": _reference(ci_path, upload_primitives.CI_SCHEMA),
        "ci_run_id": ci["run_id"],
        "live_exclusion_binding": live.artifact_reference(
            exclusion_path, live.EXCLUSION_BINDING_SCHEMA
        ),
        "live_exclusion_frozen_at_utc": exclusion["frozen_at_utc"],
        "raw_shards": {
            "count": len(chain["raw_shards"]),
            "sha256": _raw_roster_digest(chain["raw_shards"]),
        },
        "tool_closure": _tool_closure(),
        "uploads_authorized": 1,
        "submit_clicks_authorized": 1,
        "live_games_required": 90,
        "rollback_authorized": False,
        "second_upload_authorized": False,
    }


def authorize_release(
    campaign_root: pathlib.Path, *, qualified_path: pathlib.Path,
    ci_evidence_path: pathlib.Path, live_exclusion_binding_path: pathlib.Path,
    repository: pathlib.Path, authorized_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    root = release_root(campaign_root, create=True)
    _reject_foreign_upload_state(campaign_root)
    authorization_path = _fixed_output(root, "one-upload-authorization.json")
    inputs_path = _fixed_output(root, "authorization-inputs.json")
    if inputs_path.exists() and authorization_path.exists():
        return validate_release_authorization(
            campaign_root, repository=repository,
            qualified_validator=qualified_validator,
        )["authorization"]
    if inputs_path.exists() and not authorization_path.exists():
        raise ReleaseError("authorization inputs exist without one-upload authorization")
    if any((root / name).exists() or (root / name).is_symlink() for name in (
        "upload", "00-fresh-editor.json", "live-window", "completion.json",
        "publication.json",
    )):
        raise ReleaseError("release execution output predates authorization")
    chain = dict(qualified_validator(campaign_root, qualified_path, repository))
    required_chain = {
        "qualified", "qualified_path", "plan", "plan_path", "bank_receipt",
        "bank_receipt_path", "aggregate", "aggregate_path", "consumption",
        "consumption_path", "candidate_path", "runtime_path", "preflight_path",
        "binding_path", "git", "raw_shards",
    }
    if set(chain) != required_chain:
        raise ReleaseError("qualified validator returned an incomplete chain")
    if pathlib.Path(chain["qualified_path"]).resolve() != qualified_path.resolve():
        raise ReleaseError("qualified validator redirected its input")
    fixed_ci_path = _fixed_file(root, "github-ci.json", "green CI evidence")
    if (
        ci_evidence_path.absolute() != fixed_ci_path
        or ci_evidence_path.is_symlink()
    ):
        raise ReleaseError("CI evidence is outside the fixed release root")
    ci = upload_primitives.validate_ci_evidence(
        ci_evidence_path, expected_head=chain["qualified"]["candidate_commit"]
    )
    fixed_exclusion_path = _fixed_file(
        root, "live-exclusion-binding.json", "live exclusion binding"
    )
    if (
        live_exclusion_binding_path.absolute() != fixed_exclusion_path
        or live_exclusion_binding_path.is_symlink()
    ):
        raise ReleaseError("live exclusion binding is outside the fixed release root")
    try:
        exclusion, _registry = live.validate_exclusion_binding(
            live_exclusion_binding_path
        )
    except Exception as error:
        raise ReleaseError("live exclusion binding is invalid") from error
    authorized = _parse_utc(authorized_at_utc, "v3 upload authorization time")
    chronological = (
        (ci["fetched_at_utc"], "CI fetch time"),
        (chain["aggregate"].get("completed_at_utc"), "strict-final completion"),
        (chain["consumption"].get("launched_at_utc"), "strict-final launch"),
        (exclusion.get("frozen_at_utc"), "live exclusion freeze"),
    )
    if any(authorized < _parse_utc(value, label) for value, label in chronological):
        raise ReleaseError("upload authorization predates a required sealed input")
    preflight_reference = qualification.load_sealed(
        pathlib.Path(chain["preflight_path"]),
        final_bridge.deployment_preflight.REFERENCE_SCHEMA,
    )
    preflight_receipt_path = _verify_reference(
        preflight_reference.get("receipt"),
        final_bridge.deployment_preflight.RECEIPT_SCHEMA,
        "deployment preflight receipt",
    )
    preflight_receipt = qualification.load_sealed(
        preflight_receipt_path, final_bridge.deployment_preflight.RECEIPT_SCHEMA
    )
    preflight_claim_path = _verify_reference(
        preflight_receipt.get("claim"),
        final_bridge.deployment_preflight.CLAIM_SCHEMA,
        "deployment preflight claim",
    )
    preflight_claim = qualification.load_sealed(
        preflight_claim_path, final_bridge.deployment_preflight.CLAIM_SCHEMA
    )
    if authorized < _parse_utc(
        preflight_claim.get("claimed_at_utc"), "preflight claim time"
    ):
        raise ReleaseError("upload authorization predates preflight")

    normalized_ci = _normalized_ci(ci)
    authorization = qualification.create_upload_authorization(
        authorization_path,
        binding_path=pathlib.Path(chain["binding_path"]),
        aggregate_path=pathlib.Path(chain["aggregate_path"]),
        ci_record=normalized_ci,
    )
    body = _authorization_inputs_body(
        chain, authorization_path=authorization_path,
        ci_path=ci_evidence_path, ci=ci,
        exclusion_path=live_exclusion_binding_path, exclusion=exclusion,
        authorized_at_utc=authorized_at_utc, root=root,
    )
    qualification.write_sealed(inputs_path, body)
    validate_release_authorization(
        campaign_root, repository=repository,
        qualified_validator=qualified_validator,
    )
    return authorization


def validate_release_authorization(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    root = release_root(campaign_root, create=False)
    _reject_foreign_upload_state(campaign_root)
    inputs_path = _fixed_file(
        root, "authorization-inputs.json", "release authorization inputs"
    )
    authorization_path = _fixed_file(
        root, "one-upload-authorization.json", "one-upload authorization"
    )
    inputs = qualification.load_sealed(inputs_path, AUTH_INPUT_SCHEMA)
    authorization = qualification.load_sealed(
        authorization_path, qualification.UPLOAD_AUTH_SCHEMA
    )
    if (
        inputs.get("release_root") != str(root.resolve())
        or inputs.get("authorization")
        != _reference(authorization_path, qualification.UPLOAD_AUTH_SCHEMA)
        or inputs.get("uploads_authorized") != 1
        or inputs.get("submit_clicks_authorized") != 1
        or inputs.get("second_upload_authorized") is not False
        or authorization.get("uploads_authorized") != 1
        or authorization.get("rank4_replacement_authorized") is not False
    ):
        raise ReleaseError("fixed v3 upload authorization changed")
    qualified_path = _verify_reference(
        inputs.get("qualified"), final_bridge.QUALIFIED_SCHEMA, "v3 qualified input"
    )
    chain = dict(qualified_validator(campaign_root, qualified_path, repository))
    fixed_ci_path = _fixed_file(root, "github-ci.json", "green CI evidence")
    ci_path = _verify_reference(
        inputs.get("ci"), upload_primitives.CI_SCHEMA, "green CI evidence"
    )
    if ci_path != fixed_ci_path:
        raise ReleaseError("green CI evidence escaped its fixed route")
    ci = upload_primitives.validate_ci_evidence(
        ci_path, expected_head=chain["qualified"]["candidate_commit"]
    )
    exclusion_path = live.resolve_path(inputs["live_exclusion_binding"]["path"])
    if exclusion_path != _fixed_file(
        root, "live-exclusion-binding.json", "live exclusion binding"
    ):
        raise ReleaseError("live exclusion binding escaped its fixed route")
    if inputs.get("live_exclusion_binding") != live.artifact_reference(
        exclusion_path, live.EXCLUSION_BINDING_SCHEMA
    ):
        raise ReleaseError("live exclusion binding reference changed")
    exclusion, _registry = live.validate_exclusion_binding(exclusion_path)
    authorized = _parse_utc(
        inputs.get("authorized_at_utc"), "v3 upload authorization time"
    )
    chronological = (
        (ci.get("fetched_at_utc"), "CI fetch time"),
        (chain["aggregate"].get("completed_at_utc"), "strict-final completion"),
        (chain["consumption"].get("launched_at_utc"), "strict-final launch"),
        (exclusion.get("frozen_at_utc"), "live exclusion freeze"),
    )
    if any(authorized < _parse_utc(value, label) for value, label in chronological):
        raise ReleaseError("sealed upload authorization chronology changed")
    preflight_reference = qualification.load_sealed(
        pathlib.Path(chain["preflight_path"]),
        final_bridge.deployment_preflight.REFERENCE_SCHEMA,
    )
    preflight_receipt_path = _verify_reference(
        preflight_reference.get("receipt"),
        final_bridge.deployment_preflight.RECEIPT_SCHEMA,
        "deployment preflight receipt",
    )
    preflight_receipt = qualification.load_sealed(
        preflight_receipt_path, final_bridge.deployment_preflight.RECEIPT_SCHEMA
    )
    preflight_claim_path = _verify_reference(
        preflight_receipt.get("claim"),
        final_bridge.deployment_preflight.CLAIM_SCHEMA,
        "deployment preflight claim",
    )
    preflight_claim = qualification.load_sealed(
        preflight_claim_path, final_bridge.deployment_preflight.CLAIM_SCHEMA
    )
    if authorized < _parse_utc(
        preflight_claim.get("claimed_at_utc"), "preflight claim time"
    ):
        raise ReleaseError("sealed upload authorization predates preflight")
    expected_authorization = qualification.seal({
        "schema": qualification.UPLOAD_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "uploads_authorized": 1,
        "rank4_replacement_authorized": False,
        "candidate_commit": chain["qualified"]["candidate_commit"],
        "candidate": chain["qualified"]["candidate"],
        "binding": _reference(
            pathlib.Path(chain["binding_path"]), qualification.GATE_BINDING_SCHEMA
        ),
        "aggregate": _reference(
            pathlib.Path(chain["aggregate_path"]),
            qualification.FINAL_AGGREGATE_SCHEMA,
        ),
        "ci": _normalized_ci(ci),
        "upload_ledger_root": str(root.resolve()),
    })
    if authorization != expected_authorization:
        raise ReleaseError("one-upload authorization no longer matches v3 chain")
    expected_inputs = qualification.seal(_authorization_inputs_body(
        chain, authorization_path=authorization_path,
        ci_path=ci_path, ci=ci, exclusion_path=exclusion_path,
        exclusion=exclusion, authorized_at_utc=str(inputs["authorized_at_utc"]),
        root=root,
    ))
    if inputs != expected_inputs:
        raise ReleaseError("v3 release authorization inputs changed")
    return {
        "root": root, "inputs": inputs, "authorization": authorization,
        "authorization_path": authorization_path, "chain": chain,
        "ci": ci, "exclusion_path": exclusion_path,
    }


def _event(root: pathlib.Path, name: str, status: str) -> dict[str, Any]:
    path = _fixed_file(root, f"upload/{name}", f"upload event {name}")
    value = qualification.load_sealed(
        path, qualification.UPLOAD_EVENT_SCHEMA
    )
    if value.get("status") != status:
        raise ReleaseError(f"upload event has unexpected status: {name}")
    return value


def fresh_editor(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    session_id: str, opened_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    if not isinstance(session_id, str) or not session_id.strip():
        raise ReleaseError("fresh editor session ID is empty")
    opened = _parse_utc(opened_at_utc, "fresh editor time")
    if opened < _parse_utc(state["inputs"]["authorized_at_utc"], "authorization time"):
        raise ReleaseError("fresh editor predates upload authorization")
    _fixed_output(state["root"], "upload/00-prepared.json")
    _fixed_output(state["root"], "00-fresh-editor.json")
    qualification.prepare_upload(
        state["root"], authorization_path=state["authorization_path"],
        created_at_utc=opened_at_utc, fresh_editor=True,
    )
    return qualification.write_sealed(_fixed_output(
        state["root"], "00-fresh-editor.json"
    ), {
        "schema": FRESH_EDITOR_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "fresh-editor-opened",
        "session_id": session_id,
        "opened_at_utc": opened_at_utc,
        "authorization": _reference(
            state["authorization_path"], qualification.UPLOAD_AUTH_SCHEMA
        ),
        "fresh": True,
    })


def attest_copyback(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    generated_source: pathlib.Path, copied_back_source: pathlib.Path,
    created_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    editor = qualification.load_sealed(
        _fixed_file(
            state["root"], "00-fresh-editor.json", "fresh editor receipt"
        ), FRESH_EDITOR_SCHEMA
    )
    if _parse_utc(created_at_utc, "copy-back time") < _parse_utc(
        editor.get("opened_at_utc"), "fresh editor time"
    ):
        raise ReleaseError("copy-back predates fresh editor")
    if pathlib.Path(state["chain"]["candidate_path"]).resolve() != generated_source.resolve():
        raise ReleaseError("copy-back does not use the qualified generated source")
    _record(generated_source, ascii_required=True)
    _record(copied_back_source, ascii_required=True)
    _fixed_output(state["root"], "upload/01-editor-copyback.json")
    return qualification.attest_editor_copyback(
        state["root"], authorization_path=state["authorization_path"],
        generated_source=generated_source, copied_back_source=copied_back_source,
        created_at_utc=created_at_utc,
    )


def record_play(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    legal_stdout: bool, expected_telemetry: bool, created_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    path = _fixed_output(state["root"], "upload/02-play.json")
    if path.exists() or path.is_symlink():
        raise ReleaseError("Play has already been recorded and cannot be retried")
    return qualification.record_play(
        state["root"], authorization_path=state["authorization_path"],
        legal_stdout=legal_stdout, expected_telemetry=expected_telemetry,
        created_at_utc=created_at_utc,
    )


def start_submit(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    started_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    play = qualification.load_sealed(
        _fixed_file(state["root"], "upload/02-play.json", "Play event"),
        qualification.UPLOAD_EVENT_SCHEMA,
    )
    if play.get("status") == "play-failed":
        raise ReleaseError("failed Play permanently forbids Submit")
    _fixed_output(state["root"], "upload/03-submit-started.json")
    return qualification.start_submit(
        state["root"], authorization_path=state["authorization_path"],
        started_at_utc=started_at_utc,
    )


def record_ambiguous(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    observed_at_utc: str, evidence: Mapping[str, Any],
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    _fixed_output(state["root"], "upload/04-submit-ambiguous.json")
    return qualification.record_submit_ambiguous(
        state["root"], authorization_path=state["authorization_path"],
        observed_at_utc=observed_at_utc, evidence=evidence,
    )


def attest_submission(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    agent_id: int, submission_id: int, submitted_at_utc: str,
    ambiguity_resolution: Mapping[str, Any] | None = None,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    path = _fixed_output(state["root"], "upload/05-submission-attested.json")
    if path.exists():
        _path, existing = _submission_attestation(state)
        if (
            existing.get("agent_id") != agent_id
            or existing.get("submission_id") != submission_id
            or existing.get("candidate_commit")
            != state["authorization"]["candidate_commit"]
            or existing.get("source_sha256")
            != state["authorization"]["candidate"]["sha256"]
            or existing.get("submit_clicks") != 1
        ):
            raise ReleaseError("existing submission attestation uses another identity")
        return existing
    return qualification.attest_submission(
        state["root"], authorization_path=state["authorization_path"],
        agent_id=agent_id, submission_id=submission_id,
        submitted_at_utc=submitted_at_utc,
        ambiguity_resolution=ambiguity_resolution,
    )


def _submission_attestation(state: Mapping[str, Any]) -> tuple[pathlib.Path, dict[str, Any]]:
    root = pathlib.Path(state["root"])
    upload_root = _validate_regular_tree(
        _fixed_directory(root, "upload", create=False, label="upload event directory"),
        "upload event directory",
    )
    path = _fixed_file(
        root, "upload/05-submission-attested.json", "submission attestation"
    )
    attestation = qualification.load_sealed(path, qualification.UPLOAD_EVENT_SCHEMA)
    authorization_reference = _reference(
        pathlib.Path(state["authorization_path"]), qualification.UPLOAD_AUTH_SCHEMA
    )
    expected_names = {
        "00-prepared.json", "01-editor-copyback.json", "02-play.json",
        "03-submit-started.json", "05-submission-attested.json",
    }
    ambiguous_path = upload_root / "04-submit-ambiguous.json"
    if ambiguous_path.exists():
        expected_names.add("04-submit-ambiguous.json")
    children = list(upload_root.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ReleaseError(
            "upload event ledger contains a directory, special node, or symlink"
        )
    observed_names = {child.name for child in children}
    if observed_names != expected_names:
        raise ReleaseError("upload event roster is incomplete or contains extras")
    editor = qualification.load_sealed(
        _fixed_file(root, "00-fresh-editor.json", "fresh editor receipt"),
        FRESH_EDITOR_SCHEMA,
    )
    prepared = _event(root, "00-prepared.json", "prepared")
    copyback = _event(root, "01-editor-copyback.json", "editor-copyback-verified")
    play = _event(root, "02-play.json", "play-passed")
    started = _event(root, "03-submit-started.json", "submit-started")
    for event in (editor, prepared, copyback, play, started, attestation):
        if event.get("authorization") != authorization_reference:
            raise ReleaseError("upload event uses another authorization")
    if editor.get("status") != "fresh-editor-opened" or editor.get("fresh") is not True:
        raise ReleaseError("fresh editor receipt changed")
    editor_at = _parse_utc(editor.get("opened_at_utc"), "fresh editor time")
    prepared_at = _parse_utc(prepared.get("created_at_utc"), "upload preparation")
    copied_at = _parse_utc(copyback.get("created_at_utc"), "editor copy-back")
    played_at = _parse_utc(play.get("created_at_utc"), "Play time")
    started_at = _parse_utc(started.get("started_at_utc"), "Submit start")
    submitted_at = _parse_utc(attestation.get("submitted_at_utc"), "submission time")
    if not editor_at <= prepared_at <= copied_at <= played_at <= started_at <= submitted_at:
        raise ReleaseError("upload lifecycle chronology changed")
    if started.get("one_shot") is not True:
        raise ReleaseError("Submit start is not a one-shot claim")
    if ambiguous_path.exists():
        ambiguous = _event(root, "04-submit-ambiguous.json", "submit-ambiguous")
        resolution = attestation.get("ambiguity_resolution")
        if (
            ambiguous.get("authorization") != authorization_reference
            or ambiguous.get("submit_must_not_be_clicked_again") is not True
            or _parse_utc(
                ambiguous.get("observed_at_utc"), "ambiguous Submit observation"
            ) > submitted_at
            or not isinstance(resolution, Mapping)
            or resolution.get("matching_submissions") != 1
            or resolution.get("agent_id") != attestation.get("agent_id")
            or resolution.get("submission_id") != attestation.get("submission_id")
        ):
            raise ReleaseError("ambiguous Submit was not uniquely resolved")
    elif attestation.get("ambiguity_resolution") is not None:
        raise ReleaseError("submission has unrequested ambiguity resolution")
    attestations = []
    for child in upload_root.rglob("*.json"):
        event = qualification.load_sealed(child, qualification.UPLOAD_EVENT_SCHEMA)
        if event.get("status") == "submission-attested":
            attestations.append(child.resolve())
    if (
        attestations != [path.resolve()]
        or attestation.get("status") != "submission-attested"
        or attestation.get("submit_clicks") != 1
        or attestation.get("authorization")
        != authorization_reference
        or attestation.get("candidate_commit")
        != state["authorization"]["candidate_commit"]
        or attestation.get("source_sha256")
        != state["authorization"]["candidate"]["sha256"]
        or attestation.get("source_bytes")
        != state["authorization"]["candidate"]["bytes"]
    ):
        raise ReleaseError("release does not contain exactly one bound upload")
    return path, attestation


def watch_live_window(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
    **watch_arguments: Any,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    attestation_path, _attestation = _submission_attestation(state)
    data_root = _validate_regular_tree(
        _fixed_directory(
            pathlib.Path(state["root"]), "live-window", create=True,
            label="live-window output directory",
        ),
        "live-window output directory",
    )
    result = dict(live.watch_window(
        submission_attestation_path=attestation_path,
        exclusion_binding_path=pathlib.Path(state["exclusion_path"]),
        data_root=data_root,
        **watch_arguments,
    ))
    if result.get("status") != "waiting" and (
        result.get("exact_games") != 90
        or result.get("second_upload_authorized") is not False
        or result.get("rollback_authorized") is not False
        or result.get("training_eligible") is not False
    ):
        raise ReleaseError("live window returned a forbidden completion state")
    return result


def verify_completion(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    verified_at_utc: str,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
    live_verifier: Callable[..., Mapping[str, Any]] = live.verify_window_reference,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    root = pathlib.Path(state["root"])
    completion_path = _fixed_output(root, "completion.json")
    if completion_path.exists():
        return validate_completion(
            campaign_root, repository=repository,
            qualified_validator=qualified_validator, live_verifier=live_verifier,
        )
    attestation_path, attestation = _submission_attestation(state)
    if _parse_utc(verified_at_utc, "completion verification time") < _parse_utc(
        attestation.get("submitted_at_utc"), "submission time"
    ):
        raise ReleaseError("completion verification predates submission")
    data_root = _validate_regular_tree(
        _fixed_directory(
            root, "live-window", create=False, label="live-window output directory"
        ),
        "live-window output directory",
    )
    reference_path = _fixed_file(
        root, "live-window/live-window.reference.json", "live-window reference"
    )
    live_result = dict(live_verifier(reference_path, data_root=data_root))
    if (
        live_result.get("exact_games") != 90
        or live_result.get("status") not in {
            "complete-accepted-diagnostic",
            "complete-rejected-focus-operational-failure",
        }
        or live_result.get("training_eligible") is not False
        or live_result.get("rollback_authorized") is not False
        or live_result.get("second_upload_authorized") is not False
    ):
        raise ReleaseError("live window is not an exact permitted 90-game outcome")
    live_attestation_path, live_attestation = upload_primitives._live_submission_attestation(
        live_result, live_data_root=data_root
    )
    if (
        live_attestation_path.resolve() != attestation_path.resolve()
        or live_attestation.get("sha256") != qualification.sha256_file(attestation_path)
    ):
        raise ReleaseError("live window belongs to another submission")
    body = {
        "schema": COMPLETION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "complete",
        "verified_at_utc": verified_at_utc,
        "qualified": state["inputs"]["qualified"],
        "authorization": state["inputs"]["authorization"],
        "ci": state["inputs"]["ci"],
        "upload_attestation": _reference(
            attestation_path, qualification.UPLOAD_EVENT_SCHEMA
        ),
        "live_reference": {
            "path": str(reference_path.resolve()),
            "sha256": qualification.sha256_file(reference_path),
            "status": live_result["status"],
            "exact_games": 90,
        },
        "uploads_completed": 1,
        "live_games": 90,
        "diagnostic_only": True,
        "training_eligible": False,
        "rollback_authorized": False,
        "second_upload_authorized": False,
        "rank4_replaced": False,
        "rank1_claim": False,
    }
    result = qualification.write_sealed(completion_path, body)
    validate_completion(
        campaign_root, repository=repository,
        qualified_validator=qualified_validator, live_verifier=live_verifier,
    )
    return result


def validate_completion(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
    live_verifier: Callable[..., Mapping[str, Any]] = live.verify_window_reference,
) -> dict[str, Any]:
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    root = pathlib.Path(state["root"])
    path = _fixed_file(root, "completion.json", "release completion")
    value = qualification.load_sealed(path, COMPLETION_SCHEMA)
    attestation_path, _attestation = _submission_attestation(state)
    data_root = _validate_regular_tree(
        _fixed_directory(
            root, "live-window", create=False, label="live-window output directory"
        ),
        "live-window output directory",
    )
    reference_path = _fixed_file(
        root, "live-window/live-window.reference.json", "live-window reference"
    )
    live_result = dict(live_verifier(reference_path, data_root=data_root))
    live_attestation_path, live_attestation = upload_primitives._live_submission_attestation(
        live_result, live_data_root=data_root
    )
    if (
        live_attestation_path.resolve() != attestation_path.resolve()
        or live_attestation.get("sha256") != qualification.sha256_file(attestation_path)
    ):
        raise ReleaseError("completed live window belongs to another submission")
    _parse_utc(value.get("verified_at_utc"), "completion verification time")
    expected = qualification.seal({
        "schema": COMPLETION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "complete",
        "verified_at_utc": str(value.get("verified_at_utc")),
        "qualified": state["inputs"]["qualified"],
        "authorization": state["inputs"]["authorization"],
        "ci": state["inputs"]["ci"],
        "upload_attestation": _reference(
            attestation_path, qualification.UPLOAD_EVENT_SCHEMA
        ),
        "live_reference": {
            "path": str(reference_path.resolve()),
            "sha256": qualification.sha256_file(reference_path),
            "status": live_result.get("status"),
            "exact_games": 90,
        },
        "uploads_completed": 1,
        "live_games": 90,
        "diagnostic_only": True,
        "training_eligible": False,
        "rollback_authorized": False,
        "second_upload_authorized": False,
        "rank4_replaced": False,
        "rank1_claim": False,
    })
    if (
        live_result.get("exact_games") != 90
        or live_result.get("status") not in {
            "complete-accepted-diagnostic",
            "complete-rejected-focus-operational-failure",
        }
        or live_result.get("training_eligible") is not False
        or live_result.get("rollback_authorized") is not False
        or live_result.get("second_upload_authorized") is not False
        or value != expected
    ):
        raise ReleaseError("v3 release completion chain changed")
    return value


def publish_release(
    campaign_root: pathlib.Path, *, repository: pathlib.Path,
    qualified_validator: QualifiedValidator = validate_qualified_chain,
    live_verifier: Callable[..., Mapping[str, Any]] = live.verify_window_reference,
) -> dict[str, Any]:
    """Publish a privacy-safe aggregate after upload and all 90 games."""

    completion = validate_completion(
        campaign_root, repository=repository,
        qualified_validator=qualified_validator, live_verifier=live_verifier,
    )
    state = validate_release_authorization(
        campaign_root, repository=repository, qualified_validator=qualified_validator
    )
    root = pathlib.Path(state["root"])
    publication_path = _fixed_output(root, "publication.json")
    _fixed_file(root, "completion.json", "release completion")
    data_root = _validate_regular_tree(
        _fixed_directory(
            root, "live-window", create=False, label="live-window output directory"
        ),
        "live-window output directory",
    )
    reference_path = _fixed_file(
        root, "live-window/live-window.reference.json", "live-window reference"
    )
    reference = live.load_sealed(
        reference_path, live.WINDOW_REFERENCE_SCHEMA, "release live reference"
    )
    receipt_path = live.resolve_path(reference["receipt"]["path"])
    receipt = live.load_sealed(
        receipt_path, live.WINDOW_RECEIPT_SCHEMA, "release live receipt"
    )
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise ReleaseError("live receipt has no privacy-safe operational summary")
    aggregate_summary = state["chain"]["aggregate"].get("summary")
    verdict = state["chain"]["aggregate"].get("verdict")
    if not isinstance(aggregate_summary, Mapping) or not isinstance(verdict, Mapping):
        raise ReleaseError("strict-final publication summary is absent")
    attestation_path, attestation = _submission_attestation(state)
    body = {
        "schema": PUBLICATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "complete",
        "candidate": {
            "commit": state["authorization"]["candidate_commit"],
            "source_sha256": state["authorization"]["candidate"]["sha256"],
            "source_bytes": state["authorization"]["candidate"]["bytes"],
            "runtime_sha256": state["inputs"]["runtime"]["sha256"],
        },
        "strict_rank4": {
            "games": aggregate_summary.get("games"),
            "candidate_wins": aggregate_summary.get("candidate_wins"),
            "candidate_color_wins": dict(
                aggregate_summary.get("candidate_color_wins", {})
            ),
            "failures": dict(aggregate_summary.get("failures", {})),
            "maximum_turns": aggregate_summary.get("maximum_turns"),
            "timing": dict(aggregate_summary.get("timing", {})),
            "uncontended_timing": dict(
                aggregate_summary.get("uncontended_timing", {})
            ),
            "passed": verdict.get("passed") is True,
            "thresholds": dict(verdict.get("thresholds", {})),
        },
        "ci": {
            "run_id": state["ci"]["run_id"],
            "head_sha": state["ci"]["head_sha"],
            "conclusion": state["ci"]["conclusion"],
            "required_jobs": {
                job_id: "success" for job_id in upload_primitives.REQUIRED_JOB_IDS
            },
        },
        "upload": {
            "count": 1,
            "submit_clicks": attestation["submit_clicks"],
            "attestation_sha256": qualification.sha256_file(attestation_path),
        },
        "live": {
            "games": 90,
            "status": reference.get("status"),
            "focus_operational_failure_games": summary.get(
                "focus_operational_failure_games"
            ),
            "opponent_operational_failure_games": summary.get(
                "opponent_operational_failure_games"
            ),
            "opponent_failure_games_counted_as_strength_wins": 0,
            "diagnostic_only": True,
        },
        "claims": {
            "rank4_replaced": False,
            "rank1_claim": False,
            "training_eligible": False,
            "rollback_authorized": False,
            "second_upload_authorized": False,
        },
        "completion_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(completion)
        ),
        "private_payloads_serialized": False,
    }
    if publication_path.exists():
        existing = qualification.load_sealed(publication_path, PUBLICATION_SCHEMA)
        if existing != qualification.seal(body):
            raise ReleaseError("existing discrete-v3 publication changed")
        return existing
    return qualification.write_sealed(publication_path, body)


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ReleaseError(f"JSON input is not an object: {path}")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-root", type=pathlib.Path, required=True)
    parser.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-live-exclusions")
    _common(freeze)
    freeze.add_argument("--registry", type=pathlib.Path, required=True)
    freeze.add_argument("--frozen-at-utc", default=utc_now())

    ci = commands.add_parser("seal-ci")
    _common(ci)
    ci.add_argument("--run-id", type=int)
    ci.add_argument("--gh-json", type=pathlib.Path)
    ci.add_argument("--expected-head", required=True)
    ci.add_argument("--fetched-at-utc", default=utc_now())
    ci.add_argument("--gh", type=pathlib.Path, default=pathlib.Path("gh"))

    authorize = commands.add_parser("authorize")
    _common(authorize)
    authorize.add_argument("--qualified", type=pathlib.Path, required=True)
    authorize.add_argument("--ci", type=pathlib.Path, required=True)
    authorize.add_argument("--live-exclusion-binding", type=pathlib.Path, required=True)
    authorize.add_argument("--authorized-at-utc", default=utc_now())

    editor = commands.add_parser("fresh-editor")
    _common(editor)
    editor.add_argument("--session-id", required=True)
    editor.add_argument("--opened-at-utc", default=utc_now())

    copyback = commands.add_parser("copyback")
    _common(copyback)
    copyback.add_argument("--generated-source", type=pathlib.Path, required=True)
    copyback.add_argument("--copied-back-source", type=pathlib.Path, required=True)
    copyback.add_argument("--created-at-utc", default=utc_now())

    play = commands.add_parser("play")
    _common(play)
    play.add_argument("--legal-stdout", action="store_true")
    play.add_argument("--expected-telemetry", action="store_true")
    play.add_argument("--created-at-utc", default=utc_now())

    submit = commands.add_parser("submit-start")
    _common(submit)
    submit.add_argument("--started-at-utc", default=utc_now())

    ambiguous = commands.add_parser("submit-ambiguous")
    _common(ambiguous)
    ambiguous.add_argument("--evidence-json", type=pathlib.Path, required=True)
    ambiguous.add_argument("--observed-at-utc", default=utc_now())

    attest = commands.add_parser("submission-attest")
    _common(attest)
    attest.add_argument("--agent-id", type=int, required=True)
    attest.add_argument("--submission-id", type=int, required=True)
    attest.add_argument("--submitted-at-utc", default=utc_now())
    attest.add_argument("--ambiguity-resolution-json", type=pathlib.Path)

    watch = commands.add_parser("watch-live")
    _common(watch)
    watch.add_argument("--poll-seconds", type=float, default=10.0)
    watch.add_argument("--timeout-seconds", type=float, default=3_600.0)
    watch.add_argument("--workers", type=int, default=2)

    complete = commands.add_parser("verify-completion")
    _common(complete)
    complete.add_argument("--verified-at-utc", default=utc_now())

    publish = commands.add_parser("publish")
    _common(publish)

    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-live-exclusions":
            result = freeze_live_exclusions(
                args.campaign_root, registry_path=args.registry,
                frozen_at_utc=args.frozen_at_utc,
            )
        elif args.command == "seal-ci":
            if (args.run_id is None) == (args.gh_json is None):
                raise ReleaseError(
                    "seal-ci requires exactly one of --run-id/--gh-json"
                )
            payload = (
                _json(args.gh_json)
                if args.gh_json is not None
                else upload_primitives.fetch_gh_run(
                    args.run_id, gh_executable=args.gh
                )
            )
            result = seal_ci_evidence(
                args.campaign_root, gh_payload=payload,
                expected_head=args.expected_head,
                fetched_at_utc=args.fetched_at_utc,
            )
        elif args.command == "authorize":
            result = authorize_release(
                args.campaign_root, qualified_path=args.qualified,
                ci_evidence_path=args.ci,
                live_exclusion_binding_path=args.live_exclusion_binding,
                repository=args.repository,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "fresh-editor":
            result = fresh_editor(
                args.campaign_root, repository=args.repository,
                session_id=args.session_id, opened_at_utc=args.opened_at_utc,
            )
        elif args.command == "copyback":
            result = attest_copyback(
                args.campaign_root, repository=args.repository,
                generated_source=args.generated_source,
                copied_back_source=args.copied_back_source,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "play":
            result = record_play(
                args.campaign_root, repository=args.repository,
                legal_stdout=args.legal_stdout,
                expected_telemetry=args.expected_telemetry,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "submit-start":
            result = start_submit(
                args.campaign_root, repository=args.repository,
                started_at_utc=args.started_at_utc,
            )
        elif args.command == "submit-ambiguous":
            result = record_ambiguous(
                args.campaign_root, repository=args.repository,
                observed_at_utc=args.observed_at_utc,
                evidence=_json(args.evidence_json),
            )
        elif args.command == "submission-attest":
            resolution = None if args.ambiguity_resolution_json is None else _json(
                args.ambiguity_resolution_json
            )
            result = attest_submission(
                args.campaign_root, repository=args.repository,
                agent_id=args.agent_id, submission_id=args.submission_id,
                submitted_at_utc=args.submitted_at_utc,
                ambiguity_resolution=resolution,
            )
        elif args.command == "watch-live":
            result = watch_live_window(
                args.campaign_root, repository=args.repository,
                poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds,
                maximum_workers=args.workers,
            )
        elif args.command == "verify-completion":
            result = verify_completion(
                args.campaign_root, repository=args.repository,
                verified_at_utc=args.verified_at_utc,
            )
        else:
            result = publish_release(
                args.campaign_root, repository=args.repository,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (
        ReleaseError,
        upload_primitives.UploadError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"compact discrete-v3 release failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
