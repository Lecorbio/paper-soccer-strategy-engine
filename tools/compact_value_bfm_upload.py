#!/usr/bin/env python3
"""Complete and verify the single Compact Value-BFM upload lifecycle.

The maintained lifecycle accepts either its legacy strict-final authorization
directory or the Rank-4 teacher challenger's direct release authorization.  The
latter is lazily and recursively revalidated before every editor/Play/Submit
transition; schema compatibility never bypasses the dual-gate release proof.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upload helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_upload_qualification"
)
final_tools = _load(HERE / "compact_value_bfm_final.py", "compact_upload_final")
preflight_tools = _load(
    HERE / "compact_value_bfm_preflight.py", "compact_upload_preflight"
)
live_tools = _load(
    REPOSITORY / "submissions/codingame/bots/compact_value_bfm/live_window.py",
    "compact_upload_live",
)
gate_support = final_tools.gate_support
UploadError = qualification.QualificationError

NAMESPACE = "compact_value_bfm"
CI_SCHEMA = "papersoccer.compact-value-bfm.github-ci-evidence.v1"
AUTH_INPUT_SCHEMA = "papersoccer.compact-value-bfm.upload-authorization-inputs.v1"
FRESH_EDITOR_SCHEMA = "papersoccer.compact-value-bfm.fresh-editor.v1"
COMPLETION_SCHEMA = "papersoccer.compact-value-bfm.completion.v1"
RANK4_TEACHER_UPLOAD_INPUT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "one-upload-authorization-inputs.v1"
)
BRANCH = "compact-value-bfm"
WORKFLOW_FILE = "pages.yml"
WORKFLOW_NAME = "CI and Pages"
WORKFLOW_DATABASE_ID = 316333312
REPOSITORY_SLUG = "Lecorbio/paper-soccer-strategy-engine"
REPOSITORY_URL = f"https://github.com/{REPOSITORY_SLUG}"
RUN_URL_PREFIX = f"{REPOSITORY_URL}/actions/runs/"
TERMINAL_INTEGRITY_BLOCKERS = (
    pathlib.Path("iteration-governance/iteration/02-integrity-failure.json"),
    pathlib.Path("compact-value-family-invalidated.json"),
)

JOB_NAMES = {
    "Jacek replay training contracts": "replay-training-contract",
    "Leaderboard contracts": "leaderboard-contract",
    "GCC build and tests": "test-gcc",
    "Clang build and tests": "test-clang",
    "ASan and UBSan build and tests": "test-sanitizers",
}
REQUIRED_JOB_IDS = tuple(JOB_NAMES.values())


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise UploadError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise UploadError(f"{field} is invalid") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise UploadError(f"{field} must be UTC")
    return parsed


def authorization_directory(ledger_root: pathlib.Path) -> pathlib.Path:
    root = ledger_root.resolve()
    # The persistent Rank-4 challenger release bridge writes its already-deeply
    # validated authorization directly in its fixed upload root.  Detect that
    # route lazily so the maintained editor/Play/Submit lifecycle can consume
    # it without weakening or duplicating the legacy authorization path.
    direct = (
        root / "one-upload-authorization.json",
        root / "authorization-inputs.json",
    )
    if any(path.exists() or path.is_symlink() for path in direct):
        return root
    return root / "upload-authorization"


def reject_terminal_integrity_failure(ledger_root: pathlib.Path) -> None:
    root = ledger_root.resolve()
    for relative in TERMINAL_INTEGRITY_BLOCKERS:
        blocker = root / relative
        if blocker.exists() or blocker.is_symlink():
            raise UploadError(
                f"upload is forbidden by terminal integrity blocker: {blocker}"
            )


def _artifact(path: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    return qualification.artifact_reference(path, schema)


def _content_addressed(path: pathlib.Path, schema: str) -> dict[str, Any]:
    value = qualification.load_sealed(path, schema)
    if not path.name.endswith(".json") or path.name[:-5] != qualification.sha256_file(path):
        raise UploadError(f"artifact is not content-addressed: {path}")
    return value


def validate_gh_run(payload: Mapping[str, Any], *, expected_head: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise UploadError("gh run view result is not an object")
    if (not isinstance(expected_head, str)
            or qualification.COMMIT_RE.fullmatch(expected_head) is None):
        raise UploadError("expected GitHub head is not an exact commit")
    run_id = payload.get("databaseId")
    workflow_id = payload.get("workflowDatabaseId")
    attempt = payload.get("attempt")
    if (type(run_id) is not int or run_id <= 0
            or type(workflow_id) is not int
            or workflow_id != WORKFLOW_DATABASE_ID
            or type(attempt) is not int or attempt != 1
            or payload.get("workflowName") != WORKFLOW_NAME
            or payload.get("name") != WORKFLOW_NAME
            or payload.get("event") != "workflow_dispatch"
            or payload.get("headBranch") != BRANCH
            or payload.get("headSha") != expected_head
            or payload.get("status") != "completed"
            or payload.get("conclusion") != "success"
            or not isinstance(payload.get("url"), str)
            or payload["url"] != f"{RUN_URL_PREFIX}{run_id}"):
        raise UploadError("GitHub run does not bind pages.yml workflow_dispatch/head")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise UploadError("GitHub run jobs are absent")
    normalized: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, Mapping) or not isinstance(job.get("name"), str):
            raise UploadError("GitHub run contains a malformed job")
        name = job["name"]
        if name in JOB_NAMES:
            job_id = JOB_NAMES[name]
            if job_id in normalized:
                raise UploadError("GitHub run repeats a required job")
            database_id = job.get("databaseId")
            if (job.get("status") != "completed"
                    or job.get("conclusion") != "success"
                    or type(database_id) is not int or database_id <= 0
                    or not isinstance(job.get("url"), str)
                    or job["url"]
                    != f"{RUN_URL_PREFIX}{run_id}/job/{database_id}"):
                raise UploadError(f"required GitHub job did not pass: {job_id}")
            normalized[job_id] = {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "database_id": database_id,
                "url": job.get("url"),
            }
        elif job.get("conclusion") not in ("skipped", None):
            raise UploadError(f"unexpected non-skipped GitHub job: {name}")
    if set(normalized) != set(REQUIRED_JOB_IDS):
        raise UploadError("GitHub run does not contain the exact five required jobs")
    database_ids = [job["database_id"] for job in normalized.values()]
    if len(set(database_ids)) != len(REQUIRED_JOB_IDS):
        raise UploadError("GitHub run repeats an actual job database ID")
    return {
        "run_id": run_id,
        "repository": REPOSITORY_SLUG,
        "workflow_database_id": workflow_id,
        "attempt": attempt,
        "workflow_file": WORKFLOW_FILE,
        "workflow_name": WORKFLOW_NAME,
        "event": "workflow_dispatch",
        "head_branch": BRANCH,
        "head_sha": expected_head,
        "status": "completed",
        "conclusion": "success",
        "url": payload["url"],
        "jobs": {job_id: normalized[job_id] for job_id in REQUIRED_JOB_IDS},
    }


def seal_ci_evidence(
    output: pathlib.Path, *, gh_payload: Mapping[str, Any],
    expected_head: str, fetched_at_utc: str,
) -> dict[str, Any]:
    parse_utc(fetched_at_utc, "CI fetch time")
    normalized = validate_gh_run(gh_payload, expected_head=expected_head)
    return qualification.write_sealed(output, {
        "schema": CI_SCHEMA,
        "namespace": NAMESPACE,
        "fetched_at_utc": fetched_at_utc,
        "raw_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(dict(gh_payload))
        ),
        **normalized,
    })


def fetch_gh_run(
    run_id: int, *, gh_executable: pathlib.Path = pathlib.Path("gh"),
) -> dict[str, Any]:
    if type(run_id) is not int or run_id <= 0:
        raise UploadError("GitHub run ID must be a positive integer")
    fields = (
        "databaseId,name,workflowName,workflowDatabaseId,attempt,event,"
        "headBranch,headSha,status,"
        "conclusion,url,jobs"
    )
    completed = subprocess.run(
        [
            str(gh_executable), "run", "view", str(run_id),
            "--repo", REPOSITORY_SLUG, "--json", fields,
        ],
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise UploadError(
            "gh run view failed "
            f"(stderr_sha256={qualification.sha256_bytes(completed.stderr)})"
        )
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UploadError("gh run view returned invalid JSON") from error
    if not isinstance(value, dict):
        raise UploadError("gh run view did not return an object")
    return value


def validate_ci_evidence(path: pathlib.Path, *, expected_head: str) -> dict[str, Any]:
    evidence = qualification.load_sealed(path, CI_SCHEMA)
    jobs = evidence.get("jobs")
    if (evidence.get("namespace") != NAMESPACE
            or type(evidence.get("run_id")) is not int
            or evidence["run_id"] <= 0
            or evidence.get("repository") != REPOSITORY_SLUG
            or type(evidence.get("workflow_database_id")) is not int
            or evidence.get("workflow_database_id") != WORKFLOW_DATABASE_ID
            or type(evidence.get("attempt")) is not int
            or evidence.get("attempt") != 1
            or evidence.get("workflow_file") != WORKFLOW_FILE
            or evidence.get("workflow_name") != WORKFLOW_NAME
            or evidence.get("event") != "workflow_dispatch"
            or evidence.get("head_branch") != BRANCH
            or evidence.get("head_sha") != expected_head
            or evidence.get("status") != "completed"
            or evidence.get("conclusion") != "success"
            or not isinstance(evidence.get("url"), str)
            or evidence["url"] != f"{RUN_URL_PREFIX}{evidence['run_id']}"
            or qualification.SHA256_RE.fullmatch(
                str(evidence.get("raw_sha256"))
            ) is None
            or not isinstance(jobs, Mapping)
            or set(jobs) != set(REQUIRED_JOB_IDS)):
        raise UploadError("sealed CI evidence changed")
    database_ids = []
    expected_names = {job_id: name for name, job_id in JOB_NAMES.items()}
    for job_id in REQUIRED_JOB_IDS:
        job = jobs[job_id]
        if (not isinstance(job, Mapping)
                or set(job) != {
                    "name", "status", "conclusion", "database_id", "url"
                }
                or job.get("name") != expected_names[job_id]
                or job.get("status") != "completed"
                or job.get("conclusion") != "success"
                or type(job.get("database_id")) is not int
                or job["database_id"] <= 0
                or not isinstance(job.get("url"), str)
                or job["url"] != (
                    f"{RUN_URL_PREFIX}{evidence['run_id']}/job/"
                    f"{job['database_id']}"
                )):
            raise UploadError("sealed CI job evidence changed")
        database_ids.append(job["database_id"])
    if len(set(database_ids)) != len(REQUIRED_JOB_IDS):
        raise UploadError("sealed CI repeats an actual job database ID")
    parse_utc(evidence.get("fetched_at_utc"), "CI fetch time")
    return evidence


def _validate_preflight_receipt(
    path: pathlib.Path, *, candidate_commit: str, candidate_sha256: str,
) -> dict[str, Any]:
    receipt = _content_addressed(path, preflight_tools.RECEIPT_SCHEMA)
    before = receipt.get("inputs_before")
    checks = receipt.get("checks")
    if (receipt.get("status") != "passed" or receipt.get("inputs_after") != before
            or not isinstance(before, Mapping)
            or before.get("candidate_commit") != candidate_commit
            or (before.get("candidate") or {}).get("sha256") != candidate_sha256
            or not isinstance(checks, Mapping) or not checks
            or any(value != "passed" for value in checks.values())
            or receipt.get("protected_banks_accessed") != []
            or receipt.get("git_writes") != 0 or receipt.get("uploads") != 0):
        raise UploadError("preflight receipt is not a passing source-bound preflight")
    preflight_tools.validate_timing_receipt(receipt.get("timing", {}))
    preflight_tools.validate_parity_receipt(receipt.get("parity", {}))
    try:
        preflight_tools.validate_preflight_receipt(
            receipt,
            claim=receipt.get("claim", {}),
            plan=receipt.get("plan", {}),
            inputs=before,
        )
    except Exception as error:
        raise UploadError("full preflight chain validation failed") from error
    return receipt


def _raw_bound_shards(
    ledger_root: pathlib.Path, *, plan: Mapping[str, Any],
    plan_path: pathlib.Path, binding_path: pathlib.Path,
) -> list[dict[str, Any]]:
    result = []
    for index in range(100):
        receipt_path = ledger_root / "receipts" / f"shard-{index:03d}.json"
        raw_path = ledger_root / "raw" / f"shard-{index:03d}.json"
        if not raw_path.is_file():
            raise UploadError(f"raw Rank-4 shard result is absent: {index}")
        receipt = qualification.validate_shard_receipt(
            receipt_path, binding_path=binding_path, index=index
        )
        adapted = final_tools.adapt_gate_result(raw_path, plan=plan, index=index)
        if adapted != receipt.get("games"):
            raise UploadError(f"raw Rank-4 shard differs from normalized receipt: {index}")
        evidence_ref = receipt.get("evidence")
        if not isinstance(evidence_ref, Mapping):
            raise UploadError(f"raw Rank-4 shard evidence is absent: {index}")
        evidence_path = pathlib.Path(str(evidence_ref.get("path")))
        evidence = qualification.load_sealed(
            evidence_path, final_tools.RAW_SHARD_EVIDENCE_SCHEMA
        )
        raw_record = evidence.get("raw_gate_result")
        normalized_sha256 = qualification.sha256_bytes(
            qualification.canonical_json_bytes(receipt["games"])
        )
        if (dict(evidence_ref) != _artifact(
                evidence_path, final_tools.RAW_SHARD_EVIDENCE_SCHEMA
            )
                or evidence.get("namespace") != NAMESPACE
                or evidence.get("plan") != _artifact(
                    plan_path, final_tools.PLAN_SCHEMA
                )
                or evidence.get("rank4_gate") != plan.get("rank4_gate")
                or evidence.get("shard_index") != index
                or evidence.get("normalized_games_sha256") != normalized_sha256
                or evidence.get("gate_result_validated_before_normalization")
                is not True
                or not isinstance(raw_record, Mapping)
                or raw_record.get("path") != str(raw_path.resolve())
                or raw_record.get("sha256")
                != qualification.sha256_file(raw_path)
                or raw_record.get("bytes") != raw_path.stat().st_size):
            raise UploadError(f"raw Rank-4 shard evidence binding changed: {index}")
        result.append({
            "shard_index": index,
            "raw_sha256": qualification.sha256_file(raw_path),
            "raw_evidence_sha256": qualification.sha256_file(evidence_path),
            "receipt_sha256": qualification.sha256_file(receipt_path),
            "claim_sha256": receipt["claim"]["sha256"],
        })
    return result


def validate_qualified_chain(
    ledger_root: pathlib.Path, *, qualified_inputs_path: pathlib.Path,
    final_plan_path: pathlib.Path, consumption_path: pathlib.Path,
    preflight_path: pathlib.Path,
) -> dict[str, Any]:
    qualified = qualification.load_sealed(
        qualified_inputs_path, final_tools.QUALIFIED_INPUT_SCHEMA
    )
    plan = qualification.load_sealed(final_plan_path, final_tools.PLAN_SCHEMA)
    consumption = qualification.load_sealed(
        consumption_path, final_tools.CONSUMPTION_SCHEMA
    )
    if (qualified.get("status") != "rank4-qualified-awaiting-green-ci"
            or qualified.get("candidate_commit") != plan.get("candidate_commit")
            or qualified.get("candidate") != plan.get("candidate")
            or qualified.get("final_plan") != _artifact(
                final_plan_path, final_tools.PLAN_SCHEMA
            )
            or consumption.get("status") != "bank-consumed-at-launch"
            or consumption.get("plan") != _artifact(
                final_plan_path, final_tools.PLAN_SCHEMA
            )
            or consumption.get("protected_bank") != plan.get("protected_bank")
            or consumption.get("gate_bank") != plan.get("gate_bank")
            or consumption.get("gate_binding") != plan.get("gate_binding")):
        raise UploadError("qualified inputs/final plan/bank consumption are inconsistent")
    binding_path = pathlib.Path(plan["gate_binding"]["path"])
    aggregate_path = ledger_root / "aggregate.json"
    aggregate = qualification.load_sealed(
        aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
    )
    if (qualified.get("aggregate") != _artifact(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        ) or aggregate.get("binding") != _artifact(
            binding_path, qualification.GATE_BINDING_SCHEMA
        ) or aggregate.get("verdict") != qualification.strict_gate_verdict(
            aggregate.get("summary", {})
        ) or aggregate.get("verdict", {}).get("passed") is not True
            or aggregate.get("status") != "rank4-qualified"):
        raise UploadError("strict final aggregate is not Rank-4-qualified")
    if plan.get("preflight") != _artifact(preflight_path):
        raise UploadError("final plan does not bind the supplied preflight")
    preflight = _validate_preflight_receipt(
        preflight_path, candidate_commit=plan["candidate_commit"],
        candidate_sha256=plan["candidate"]["sha256"],
    )
    raw_shards = _raw_bound_shards(
        ledger_root, plan=plan, plan_path=final_plan_path,
        binding_path=binding_path
    )
    return {
        "qualified": qualified,
        "plan": plan,
        "consumption": consumption,
        "aggregate": aggregate,
        "preflight": preflight,
        "binding_path": binding_path,
        "raw_shards": raw_shards,
    }


def authorize_upload(
    ledger_root: pathlib.Path, *, qualified_inputs_path: pathlib.Path,
    final_plan_path: pathlib.Path, consumption_path: pathlib.Path,
    preflight_path: pathlib.Path, ci_evidence_path: pathlib.Path,
    authorized_at_utc: str,
) -> dict[str, Any]:
    reject_terminal_integrity_failure(ledger_root)
    authorized_at = parse_utc(authorized_at_utc, "upload authorization time")
    chain = validate_qualified_chain(
        ledger_root, qualified_inputs_path=qualified_inputs_path,
        final_plan_path=final_plan_path, consumption_path=consumption_path,
        preflight_path=preflight_path,
    )
    plan = chain["plan"]
    ci = validate_ci_evidence(
        ci_evidence_path, expected_head=plan["candidate_commit"]
    )
    directory = authorization_directory(ledger_root)
    if ci_evidence_path.resolve() != (directory / "github-ci.json").resolve():
        raise UploadError("green CI evidence is outside the fixed authorization directory")
    chronological_inputs = (
        (ci["fetched_at_utc"], "CI fetch time"),
        (chain["aggregate"].get("completed_at_utc"), "final completion time"),
        (chain["consumption"].get("launched_at_utc"), "bank launch time"),
    )
    if any(authorized_at < parse_utc(value, label)
           for value, label in chronological_inputs):
        raise UploadError("upload authorization predates a required sealed input")
    claim = chain["preflight"].get("claim")
    if (isinstance(claim, Mapping)
            and authorized_at < parse_utc(
                claim.get("claimed_at_utc"), "preflight claim time"
            )):
        raise UploadError("upload authorization predates preflight")
    authorization_path = directory / "one-upload-authorization.json"
    normalized_ci = {
        "run_id": ci["run_id"],
        "repository": ci["repository"],
        "workflow_database_id": ci["workflow_database_id"],
        "attempt": ci["attempt"],
        "head_sha": ci["head_sha"],
        "conclusion": "success",
        "jobs": {job_id: "success" for job_id in REQUIRED_JOB_IDS},
        "workflow": WORKFLOW_NAME,
        "workflow_file": WORKFLOW_FILE,
        "event": "workflow_dispatch",
        "head_branch": BRANCH,
        "head_ref": f"refs/heads/{BRANCH}",
        "url": ci["url"],
    }
    authorization = qualification.create_upload_authorization(
        authorization_path,
        binding_path=chain["binding_path"],
        aggregate_path=ledger_root / "aggregate.json",
        ci_record=normalized_ci,
    )
    qualification.write_sealed(directory / "authorization-inputs.json", {
        "schema": AUTH_INPUT_SCHEMA,
        "namespace": NAMESPACE,
        "status": "one-upload-authorized",
        "authorized_at_utc": authorized_at_utc,
        "authorization_directory": str(directory),
        "authorization": _artifact(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        ),
        "qualified_inputs": _artifact(
            qualified_inputs_path, final_tools.QUALIFIED_INPUT_SCHEMA
        ),
        "final_plan": _artifact(final_plan_path, final_tools.PLAN_SCHEMA),
        "bank_consumption": _artifact(
            consumption_path, final_tools.CONSUMPTION_SCHEMA
        ),
        "preflight": _artifact(preflight_path),
        "ci": _artifact(ci_evidence_path, CI_SCHEMA),
        "raw_shards": chain["raw_shards"],
        "uploads_authorized": 1,
    })
    return authorization


def _authorization(
    ledger_root: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any], dict[str, Any]]:
    directory = authorization_directory(ledger_root)
    inputs_path = directory / "authorization-inputs.json"
    path = directory / "one-upload-authorization.json"
    if inputs_path.is_symlink() or path.is_symlink():
        raise UploadError("upload authorization route is redirected")
    inputs = qualification.load_sealed(inputs_path)
    authorization = qualification.load_sealed(path, qualification.UPLOAD_AUTH_SCHEMA)
    if inputs.get("schema") == AUTH_INPUT_SCHEMA:
        if (inputs.get("authorization_directory") != str(directory)
                or inputs.get("authorization") != _artifact(
                    path, qualification.UPLOAD_AUTH_SCHEMA
                ) or inputs.get("uploads_authorized") != 1
                or inputs.get("status") != "one-upload-authorized"):
            raise UploadError("fixed upload authorization directory changed")
    elif inputs.get("schema") == RANK4_TEACHER_UPLOAD_INPUT_SCHEMA:
        try:
            release = _load(
                HERE / "compact_value_bfm_rank4_teacher_release.py",
                "compact_upload_rank4_teacher_release",
            )
            release_path = pathlib.Path(inputs["release_evidence"]["path"])
            release_value = qualification.load_sealed(
                release_path, release.RELEASE_EVIDENCE_SCHEMA
            )
            state = release.validate_upload_authorization(
                directory,
                release_evidence_path=release_path,
                campaign_plan_path=pathlib.Path(
                    release_value["campaign_plan"]["path"]
                ),
                attempt=int(inputs["attempt"]),
                candidate_runtime=pathlib.Path(
                    release_value["candidate"]["runtime"]["path"]
                ),
                candidate_source=pathlib.Path(
                    release_value["candidate"]["generated_source"]["path"]
                ),
                dual_qualified_path=pathlib.Path(
                    inputs["dual_qualification"]["path"]
                ),
            )
        except Exception as error:
            raise UploadError(
                "Rank-4 challenger upload authorization failed deep validation"
            ) from error
        if (
            pathlib.Path(state.get("authorization_path", "")).resolve()
            != path.resolve()
            or pathlib.Path(state.get("inputs_path", "")).resolve()
            != inputs_path.resolve()
            or state.get("authorization") != authorization
            or state.get("inputs") != inputs
            or inputs.get("status")
            != "exactly-one-upload-authorized-after-dual-qualification"
            or inputs.get("uploads_authorized") != 1
            or inputs.get("submit_clicks_authorized") != 1
            or inputs.get("second_upload_authorized") is not False
            or authorization.get("upload_ledger_root") != str(directory)
            or authorization.get("uploads_authorized") != 1
            or authorization.get("two_independent_rank4_gates_passed") is not True
        ):
            raise UploadError("Rank-4 challenger upload authorization changed")
    else:
        raise UploadError("unsupported upload authorization input schema")
    return path, authorization, inputs


def _event(root: pathlib.Path, name: str, status: str) -> dict[str, Any]:
    value = qualification.load_sealed(
        root / "upload" / name, qualification.UPLOAD_EVENT_SCHEMA
    )
    if value.get("status") != status:
        raise UploadError(f"upload event has unexpected status: {name}")
    return value


def fresh_editor(
    ledger_root: pathlib.Path, *, session_id: str, opened_at_utc: str,
) -> dict[str, Any]:
    parse_utc(opened_at_utc, "fresh editor time")
    if not isinstance(session_id, str) or not session_id.strip():
        raise UploadError("fresh editor session ID is empty")
    authorization_path, _authorization_value, inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    if parse_utc(opened_at_utc, "fresh editor time") < parse_utc(
        inputs.get("authorized_at_utc"), "upload authorization time"
    ):
        raise UploadError("fresh editor predates upload authorization")
    qualification.prepare_upload(
        directory, authorization_path=authorization_path,
        created_at_utc=opened_at_utc, fresh_editor=True,
    )
    return qualification.write_sealed(directory / "00-fresh-editor.json", {
        "schema": FRESH_EDITOR_SCHEMA,
        "namespace": NAMESPACE,
        "status": "fresh-editor-opened",
        "session_id": session_id,
        "opened_at_utc": opened_at_utc,
        "authorization": _artifact(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        ),
        "fresh": True,
    })


def attest_copyback(
    ledger_root: pathlib.Path, *, generated_source: pathlib.Path,
    copied_back_source: pathlib.Path, created_at_utc: str,
) -> dict[str, Any]:
    created = parse_utc(created_at_utc, "copy-back time")
    authorization_path, _, _inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    editor = qualification.load_sealed(
        directory / "00-fresh-editor.json", FRESH_EDITOR_SCHEMA
    )
    if created < parse_utc(editor["opened_at_utc"], "fresh editor time"):
        raise UploadError("copy-back predates fresh editor")
    return qualification.attest_editor_copyback(
        directory, authorization_path=authorization_path,
        generated_source=generated_source,
        copied_back_source=copied_back_source,
        created_at_utc=created_at_utc,
    )


def record_play(
    ledger_root: pathlib.Path, *, legal_stdout: bool,
    expected_telemetry: bool, created_at_utc: str,
) -> dict[str, Any]:
    created = parse_utc(created_at_utc, "Play time")
    authorization_path, _, _inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    if (directory / "upload/02-play.json").exists():
        raise UploadError("Play has already been recorded and cannot be retried")
    copyback = _event(directory, "01-editor-copyback.json", "editor-copyback-verified")
    if created < parse_utc(copyback["created_at_utc"], "copy-back time"):
        raise UploadError("Play predates editor copy-back")
    return qualification.record_play(
        directory, authorization_path=authorization_path,
        legal_stdout=legal_stdout, expected_telemetry=expected_telemetry,
        created_at_utc=created_at_utc,
    )


def start_submit(ledger_root: pathlib.Path, *, started_at_utc: str) -> dict[str, Any]:
    started = parse_utc(started_at_utc, "Submit start time")
    authorization_path, _, _inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    recorded_play = qualification.load_sealed(
        directory / "upload/02-play.json", qualification.UPLOAD_EVENT_SCHEMA
    )
    if recorded_play.get("status") == "play-failed":
        raise UploadError("failed Play permanently forbids Submit")
    play = _event(directory, "02-play.json", "play-passed")
    if started < parse_utc(play["created_at_utc"], "Play time"):
        raise UploadError("Submit predates Play")
    return qualification.start_submit(
        directory, authorization_path=authorization_path,
        started_at_utc=started_at_utc,
    )


def record_ambiguous(
    ledger_root: pathlib.Path, *, observed_at_utc: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    observed = parse_utc(observed_at_utc, "ambiguous Submit time")
    authorization_path, _, _inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    started = _event(directory, "03-submit-started.json", "submit-started")
    if observed < parse_utc(started["started_at_utc"], "Submit start time"):
        raise UploadError("ambiguous observation predates Submit")
    return qualification.record_submit_ambiguous(
        directory, authorization_path=authorization_path,
        observed_at_utc=observed_at_utc, evidence=evidence,
    )


def attest_submission(
    ledger_root: pathlib.Path, *, agent_id: int, submission_id: int,
    submitted_at_utc: str,
    ambiguity_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    submitted = parse_utc(submitted_at_utc, "submission time")
    authorization_path, _, _inputs = _authorization(ledger_root)
    directory = authorization_directory(ledger_root)
    started = _event(directory, "03-submit-started.json", "submit-started")
    floor = parse_utc(started["started_at_utc"], "Submit start time")
    ambiguous_path = directory / "upload/04-submit-ambiguous.json"
    if ambiguous_path.exists():
        ambiguous = _event(directory, "04-submit-ambiguous.json", "submit-ambiguous")
        floor = max(floor, parse_utc(
            ambiguous["observed_at_utc"], "ambiguous observation time"
        ))
    if submitted < floor:
        raise UploadError("submission attestation predates its Submit evidence")
    return qualification.attest_submission(
        directory, authorization_path=authorization_path,
        agent_id=agent_id, submission_id=submission_id,
        submitted_at_utc=submitted_at_utc,
        ambiguity_resolution=ambiguity_resolution,
    )


def _live_submission_attestation(
    live: Mapping[str, Any], *, live_data_root: pathlib.Path,
) -> tuple[pathlib.Path, Mapping[str, Any]]:
    reference = live.get("submission_attestation")
    if not isinstance(reference, Mapping):
        receipt_reference = live.get("receipt")
        if not isinstance(receipt_reference, Mapping):
            raise UploadError("live receipt does not expose its submission identity")
        receipt_path = live_tools.resolve_path(receipt_reference.get("path"))
        try:
            receipt_path.relative_to(live_data_root.resolve())
        except ValueError as error:
            raise UploadError("live receipt escaped its data root") from error
        receipt = live_tools.load_sealed(
            receipt_path, live_tools.WINDOW_RECEIPT_SCHEMA,
            "completion live-window receipt",
        )
        reference = receipt.get("submission_attestation")
    if not isinstance(reference, Mapping):
        raise UploadError("live receipt has no bound submission attestation")
    return live_tools.resolve_path(reference.get("path")), reference


def verify_completion(
    ledger_root: pathlib.Path, *, live_reference_path: pathlib.Path,
    live_data_root: pathlib.Path,
    live_verifier: Callable[..., Mapping[str, Any]] =
    live_tools.verify_window_reference,
    verified_at_utc: str,
) -> dict[str, Any]:
    verified = parse_utc(verified_at_utc, "completion verification time")
    directory = authorization_directory(ledger_root)
    authorization_path, authorization, inputs = _authorization(ledger_root)
    chain = validate_qualified_chain(
        ledger_root,
        qualified_inputs_path=pathlib.Path(inputs["qualified_inputs"]["path"]),
        final_plan_path=pathlib.Path(inputs["final_plan"]["path"]),
        consumption_path=pathlib.Path(inputs["bank_consumption"]["path"]),
        preflight_path=pathlib.Path(inputs["preflight"]["path"]),
    )
    ci = validate_ci_evidence(
        pathlib.Path(inputs["ci"]["path"]),
        expected_head=chain["plan"]["candidate_commit"],
    )
    attestation_path = directory / "upload/05-submission-attested.json"
    attestation = qualification.load_sealed(
        attestation_path, qualification.UPLOAD_EVENT_SCHEMA
    )
    attestations = []
    for event_path in (directory / "upload").rglob("*.json"):
        event = qualification.load_sealed(
            event_path, qualification.UPLOAD_EVENT_SCHEMA
        )
        if event.get("status") == "submission-attested":
            attestations.append(event_path.resolve())
    if (attestations != [attestation_path.resolve()]
            or attestation.get("status") != "submission-attested"
            or attestation.get("submit_clicks") != 1
            or attestation.get("authorization") != _artifact(
                authorization_path, qualification.UPLOAD_AUTH_SCHEMA
            ) or attestation.get("candidate_commit") != authorization["candidate_commit"]
            or attestation.get("source_sha256") != authorization["candidate"]["sha256"]
            or attestation.get("source_bytes") != authorization["candidate"]["bytes"]
            or type(attestation.get("agent_id")) is not int
            or attestation["agent_id"] <= 0
            or type(attestation.get("submission_id")) is not int
            or attestation["submission_id"] <= 0):
        raise UploadError("completion does not contain exactly one bound upload attestation")
    if verified < parse_utc(
        attestation.get("submitted_at_utc"), "submission attestation time"
    ):
        raise UploadError("completion verification predates the upload")
    live = dict(live_verifier(
        live_reference_path, data_root=live_data_root
    ))
    if (live.get("exact_games") != 90
            or live.get("status") not in {
                "complete-accepted-diagnostic",
                "complete-rejected-focus-operational-failure",
            }
            or live.get("training_eligible") is not False
            or live.get("rollback_authorized") is not False
            or live.get("second_upload_authorized") is not False):
        raise UploadError("live window is not a complete permitted diagnostic outcome")
    live_attestation_path, live_attestation = _live_submission_attestation(
        live, live_data_root=live_data_root
    )
    if (live_attestation_path.resolve() != attestation_path.resolve()
            or live_attestation.get("sha256")
            != qualification.sha256_file(attestation_path)):
        raise UploadError("90-game live receipt belongs to another upload")
    return qualification.write_sealed(ledger_root / "completion.json", {
        "schema": COMPLETION_SCHEMA,
        "namespace": NAMESPACE,
        "status": "complete",
        "verified_at_utc": verified_at_utc,
        "candidate_commit": authorization["candidate_commit"],
        "strict_final": inputs["qualified_inputs"],
        "ci": {"run_id": ci["run_id"], "sha256": inputs["ci"]["sha256"]},
        "upload_attestation": _artifact(
            attestation_path, qualification.UPLOAD_EVENT_SCHEMA
        ),
        "live_reference": {
            "path": str(live_reference_path.resolve()),
            "sha256": qualification.sha256_file(live_reference_path),
            "status": live["status"],
            "exact_games": 90,
        },
        "rank4_replaced": False,
        "rank1_claim": False,
    })


def _json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UploadError(f"JSON input is not an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ci = commands.add_parser("seal-ci")
    ci.add_argument("--ledger-root", type=pathlib.Path, required=True)
    ci.add_argument("--run-id", type=int)
    ci.add_argument("--gh-json", type=pathlib.Path)
    ci.add_argument("--head", required=True)
    ci.add_argument("--fetched-at-utc", default=utc_now())
    ci.add_argument("--gh", type=pathlib.Path, default=pathlib.Path("gh"))
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--ledger-root", type=pathlib.Path, required=True)
    authorize.add_argument("--qualified-inputs", type=pathlib.Path, required=True)
    authorize.add_argument("--final-plan", type=pathlib.Path, required=True)
    authorize.add_argument("--bank-consumption", type=pathlib.Path, required=True)
    authorize.add_argument("--preflight", type=pathlib.Path, required=True)
    authorize.add_argument("--ci", type=pathlib.Path, required=True)
    authorize.add_argument("--authorized-at-utc", default=utc_now())
    editor = commands.add_parser("fresh-editor")
    editor.add_argument("--ledger-root", type=pathlib.Path, required=True)
    editor.add_argument("--session-id", required=True)
    editor.add_argument("--opened-at-utc", default=utc_now())
    copyback = commands.add_parser("copyback")
    copyback.add_argument("--ledger-root", type=pathlib.Path, required=True)
    copyback.add_argument("--generated-source", type=pathlib.Path, required=True)
    copyback.add_argument("--copied-back-source", type=pathlib.Path, required=True)
    copyback.add_argument("--created-at-utc", default=utc_now())
    play = commands.add_parser("play")
    play.add_argument("--ledger-root", type=pathlib.Path, required=True)
    play.add_argument("--legal-stdout", action="store_true")
    play.add_argument("--expected-telemetry", action="store_true")
    play.add_argument("--created-at-utc", default=utc_now())
    submit = commands.add_parser("start-submit")
    submit.add_argument("--ledger-root", type=pathlib.Path, required=True)
    submit.add_argument("--started-at-utc", default=utc_now())
    ambiguous = commands.add_parser("ambiguous")
    ambiguous.add_argument("--ledger-root", type=pathlib.Path, required=True)
    ambiguous.add_argument("--evidence", type=pathlib.Path, required=True)
    ambiguous.add_argument("--observed-at-utc", default=utc_now())
    attest = commands.add_parser("attest")
    attest.add_argument("--ledger-root", type=pathlib.Path, required=True)
    attest.add_argument("--agent-id", type=int, required=True)
    attest.add_argument("--submission-id", type=int, required=True)
    attest.add_argument("--submitted-at-utc", default=utc_now())
    attest.add_argument("--ambiguity-resolution", type=pathlib.Path)
    verify = commands.add_parser("verify-completion")
    verify.add_argument("--ledger-root", type=pathlib.Path, required=True)
    verify.add_argument("--live-reference", type=pathlib.Path, required=True)
    verify.add_argument("--live-data-root", type=pathlib.Path, required=True)
    verify.add_argument("--verified-at-utc", default=utc_now())
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-ci":
            if (args.run_id is None) == (args.gh_json is None):
                raise UploadError("seal-ci requires exactly one of --run-id/--gh-json")
            payload = _json(args.gh_json) if args.gh_json else fetch_gh_run(
                args.run_id, gh_executable=args.gh
            )
            output = authorization_directory(args.ledger_root) / "github-ci.json"
            result = seal_ci_evidence(
                output, gh_payload=payload, expected_head=args.head,
                fetched_at_utc=args.fetched_at_utc,
            )
        elif args.command == "authorize":
            result = authorize_upload(
                args.ledger_root, qualified_inputs_path=args.qualified_inputs,
                final_plan_path=args.final_plan,
                consumption_path=args.bank_consumption,
                preflight_path=args.preflight, ci_evidence_path=args.ci,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "fresh-editor":
            result = fresh_editor(
                args.ledger_root, session_id=args.session_id,
                opened_at_utc=args.opened_at_utc,
            )
        elif args.command == "copyback":
            result = attest_copyback(
                args.ledger_root, generated_source=args.generated_source,
                copied_back_source=args.copied_back_source,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "play":
            result = record_play(
                args.ledger_root, legal_stdout=args.legal_stdout,
                expected_telemetry=args.expected_telemetry,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "start-submit":
            result = start_submit(
                args.ledger_root, started_at_utc=args.started_at_utc
            )
        elif args.command == "ambiguous":
            result = record_ambiguous(
                args.ledger_root, observed_at_utc=args.observed_at_utc,
                evidence=_json(args.evidence),
            )
        elif args.command == "attest":
            result = attest_submission(
                args.ledger_root, agent_id=args.agent_id,
                submission_id=args.submission_id,
                submitted_at_utc=args.submitted_at_utc,
                ambiguity_resolution=(
                    None if args.ambiguity_resolution is None
                    else _json(args.ambiguity_resolution)
                ),
            )
        else:
            result = verify_completion(
                args.ledger_root, live_reference_path=args.live_reference,
                live_data_root=args.live_data_root,
                verified_at_utc=args.verified_at_utc,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (UploadError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact upload failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
