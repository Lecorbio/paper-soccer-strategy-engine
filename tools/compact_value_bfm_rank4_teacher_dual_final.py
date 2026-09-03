#!/usr/bin/env python3
"""Execute the Rank-4 teacher challenger's two protected final gates.

This is the stateful execution bridge for the otherwise declarative challenger
governor.  It consumes one validated ``dual-final-authorization`` only after a
source-specific deployment preflight and exact green CI have bound the frozen
candidate.  It then:

* claims and materializes Gate A and Gate B exactly once, in that order;
* draws a separate 256-bit OS seed for each bank;
* includes every authorized exclusion in both banks and every Gate A symmetry
  fingerprint in Gate B's exclusion set;
* runs each gate as exactly 100 five-pair shards on exactly four workers;
* refuses to retry a bank or shard whose claim exists without its receipt; and
* adapts the maintained ``FINAL_AGGREGATE_SCHEMA`` into the challenger's final
  result input while preserving a recursively verifiable evidence closure.

No command uploads a submission or accesses CodinGame.  Protected banks and
actual-clock games are created only by explicit ``materialize`` and ``run``
commands; unit tests inject fake generators/runners and never create real
protected banks or play games.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import importlib.util
import json
import os
import pathlib
import secrets
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dual-final dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py",
    "rank4_teacher_dual_final_qualification",
)
openings = _load(
    HERE / "compact_value_bfm_openings.py",
    "rank4_teacher_dual_final_openings",
)
final_tools = _load(
    HERE / "compact_value_bfm_final.py",
    "rank4_teacher_dual_final_maintained",
)
deployment_preflight = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment_preflight.py",
    "rank4_teacher_dual_final_deployment_preflight",
)
deployment = deployment_preflight.deployment
upload = _load(
    HERE / "compact_value_bfm_upload.py",
    "rank4_teacher_dual_final_upload",
)
challenger = _load(
    HERE / "compact_value_bfm_rank4_teacher_challenger.py",
    "rank4_teacher_dual_final_governance",
)
gate_support = final_tools.gate_support


class DualFinalError(ValueError):
    """The dual-final execution or its evidence closure is invalid."""


NAMESPACE = challenger.NAMESPACE
CAMPAIGN_ID = challenger.CAMPAIGN_ID
ARCHITECTURE = challenger.ARCHITECTURE
GATE_IDS = ("gate-a", "gate-b")
SHARDS = 100
PAIRS_PER_SHARD = 5
WORKERS = 4
THREADS_PER_WORKER = 1

PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-execution-plan.v1"
)
BANK_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-bank-claim.v1"
)
BANK_RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-bank-receipt.v1"
)
CONSUMPTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-gate-consumption.v1"
)
PRIMITIVE_CONSUMPTION_SCHEMA = (
    "papersoccer.compact-value-bfm.bank-consumption.v1"
)
RAW_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-raw-shard-evidence.v1"
)
NORMALIZED_AGGREGATE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-normalized-aggregate.v1"
)
DEEP_GATE_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-deep-gate-evidence.v1"
)
RUN_RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-execution-receipt.v1"
)
PREPARED_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "dual-final-execution-prepared.v1"
)
FINGERPRINT_EXCLUSION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "fingerprint-exclusion.v1"
)

Record = dict[str, Any]
AuthorizationValidator = Callable[
    [pathlib.Path, pathlib.Path], Mapping[str, Any]
]
PreflightValidator = Callable[[pathlib.Path], Mapping[str, Any]]
CiValidator = Callable[[pathlib.Path, str], Mapping[str, Any]]
FingerprintLoader = Callable[[pathlib.Path], set[str]]
BankGenerator = Callable[..., list[dict[str, Any]]]
GateRunner = Callable[[Mapping[str, Any]], Any]
ResultValidator = Callable[..., Mapping[str, Any]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc(value: Any, label: str) -> str:
    try:
        qualification._utc(value, label)
    except Exception as error:
        raise DualFinalError(f"{label} is not a valid UTC timestamp") from error
    return str(value)


def _record(
    path: pathlib.Path, *, ascii_required: bool = False,
    executable: bool = False,
) -> Record:
    if path.is_symlink() or not path.is_file() or (
        executable and not os.access(path, os.X_OK)
    ):
        raise DualFinalError(f"required regular artifact is absent: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise DualFinalError(f"required artifact is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        **({"ascii": True} if ascii_required else {}),
        **({"executable": True} if executable else {}),
    }


def _verify_record(
    value: Any, label: str, *, ascii_required: bool = False,
    executable: bool = False,
) -> pathlib.Path:
    if not isinstance(value, Mapping):
        raise DualFinalError(f"{label} record is absent")
    path = pathlib.Path(str(value.get("path", "")))
    if dict(value) != _record(
        path, ascii_required=ascii_required, executable=executable
    ):
        raise DualFinalError(f"{label} changed")
    return path.resolve()


def _reference(path: pathlib.Path, schema: str | None = None) -> Record:
    if path.is_symlink() or not path.is_file():
        raise DualFinalError(f"referenced artifact is absent: {path}")
    return qualification.artifact_reference(path, schema)


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise DualFinalError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if dict(value) != _reference(path, schema):
        raise DualFinalError(f"{label} reference changed")
    return path.resolve()


def _safe_root(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    absolute = path.absolute()
    resolved = absolute.resolve()
    if (
        absolute == pathlib.Path(absolute.anchor)
        or resolved == pathlib.Path(resolved.anchor)
        or absolute.is_symlink()
    ):
        raise DualFinalError("unsafe dual-final output root")
    if create:
        absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir():
        raise DualFinalError("dual-final output root is absent")
    return resolved


def _write_sealed_once(path: pathlib.Path, body: Mapping[str, Any]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = qualification.seal(body)
    raw = qualification.canonical_json_bytes(sealed)
    qualification.atomic_write_once(path, raw)
    if qualification.load_sealed(path, str(body["schema"])) != sealed:
        raise DualFinalError(f"sealed artifact did not round-trip: {path}")
    return path


def _directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise DualFinalError(f"unsafe dual-final directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise DualFinalError(f"dual-final directory is absent: {path}")
    return path


def _default_authorization_validator(
    authorization_path: pathlib.Path, campaign_plan_path: pathlib.Path,
) -> Mapping[str, Any]:
    context = challenger.validate_campaign(campaign_plan_path)
    header = qualification.load_sealed(
        authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
    )
    attempt = header.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise DualFinalError("dual-final authorization attempt is invalid")
    try:
        value = challenger._validate_dual_final_authorization(
            authorization_path, context=context, attempt=attempt
        )
    except Exception as error:
        raise DualFinalError("challenger dual-final authorization did not validate") from error
    return {"authorization": dict(value), "context": context}


def _default_preflight_validator(path: pathlib.Path) -> Mapping[str, Any]:
    """Deeply validate a deployment preflight using its own frozen inputs."""

    reference = qualification.load_sealed(
        path, deployment_preflight.REFERENCE_SCHEMA
    )
    plan_path = _verify_reference(
        reference.get("plan"), deployment_preflight.PLAN_SCHEMA,
        "deployment preflight plan",
    )
    plan = qualification.load_sealed(plan_path, deployment_preflight.PLAN_SCHEMA)
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DualFinalError("deployment preflight inputs are absent")
    configuration = inputs.get("configuration")
    if not isinstance(configuration, Mapping):
        raise DualFinalError("deployment preflight configuration is absent")
    profile = configuration.get("profile")
    work = deployment.PROFILE_ROSTER.get(profile)
    try:
        state = deployment_preflight.validate_reference(
            path,
            generated_source=pathlib.Path(inputs["generated_source"]["path"]),
            candidate_source=pathlib.Path(inputs["candidate"]["path"]),
            runtime_path=pathlib.Path(inputs["runtime"]["path"]),
            repository=pathlib.Path(inputs["repository"]),
            source_repository=pathlib.Path(inputs["source_repository"]),
            search_tuple=configuration["tuple"],
            profile=profile,
            work=work,
        )
    except Exception as error:
        raise DualFinalError("source-specific deployment preflight did not validate") from error
    return dict(state)


def _default_release_preflight_validator(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Normalize the generalized macro/deployment release evidence.

    The release bridge owns validation of exporter base -> selected SEARCH_VARIANT
    -> seven-slot deployed source.  Its adapter returns the same execution-facing
    shape as the legacy discrete-v3 deployment preflight.
    """

    generated = authorization.get("generated_source")
    if not isinstance(generated, Mapping):
        raise DualFinalError("release authorization omits its generated source")
    try:
        campaign = challenger.validate_campaign(campaign_plan_path)
        selected_source = challenger._resolve_campaign_artifact(
            generated, plan=campaign["plan"],
            label="dual-final authorized generated source",
        )
    except Exception as error:
        raise DualFinalError("authorized generated source could not be resolved") from error
    try:
        from tools import compact_value_bfm_rank4_teacher_release as release

        adapter = getattr(release, "dual_final_preflight_state", None)
        if adapter is None:
            value = release.validate_release_evidence(
                path, campaign_plan_path=campaign_plan_path,
                attempt=int(authorization["attempt"]),
                candidate_runtime=pathlib.Path(
                    authorization["candidate"]["runtime"]["path"]
                ),
                candidate_source=selected_source,
            )
            state = {
                "reference": {"gate": value["gate"]},
                "candidate_commit": value["candidate_commit"],
                "candidate": value["candidate"]["source"],
                "runtime": value["candidate"]["runtime"],
                "derivation": {"configuration": value["configuration"]},
                "timing": value["timing"],
                "compile_binding": value["compile_binding"],
                "plan": {"inputs": {
                    "repository": value["repository"],
                    "tools": {"clang": value["compile_binding"]["compiler"]},
                }},
                "release_evidence": value,
            }
        else:
            state = adapter(
                path, campaign_plan_path=campaign_plan_path,
                attempt=int(authorization["attempt"]),
                candidate_runtime=pathlib.Path(
                    authorization["candidate"]["runtime"]["path"]
                ),
                candidate_source=selected_source,
            )
    except Exception as error:
        raise DualFinalError("generalized candidate release evidence did not validate") from error
    if not isinstance(state, Mapping):
        raise DualFinalError("release preflight adapter returned malformed state")
    return dict(state)


def _preflight_authority(
    *, deployment_preflight_path: pathlib.Path | None,
    release_evidence_path: pathlib.Path | None,
) -> tuple[str, str, pathlib.Path]:
    supplied = sum(item is not None for item in (
        deployment_preflight_path, release_evidence_path
    ))
    if supplied != 1:
        raise DualFinalError(
            "supply exactly one deployment preflight or generalized release evidence"
        )
    if release_evidence_path is not None:
        return (
            "rank4-teacher-release",
            challenger.RELEASE_EVIDENCE_SCHEMA,
            release_evidence_path,
        )
    assert deployment_preflight_path is not None
    return (
        "discrete-v3-deployment-preflight",
        deployment_preflight.REFERENCE_SCHEMA,
        deployment_preflight_path,
    )


def _default_ci_validator(path: pathlib.Path, candidate_commit: str) -> Mapping[str, Any]:
    try:
        return upload.validate_ci_evidence(path, expected_head=candidate_commit)
    except Exception as error:
        raise DualFinalError("exact five-job CI evidence did not validate") from error


def _runtime_identity(path: pathlib.Path) -> Record:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DualFinalError("candidate runtime is not JSON") from error
    architecture = value.get("architecture") if isinstance(value, Mapping) else None
    dimensions = architecture.get("dimensions") if isinstance(architecture, Mapping) else None
    if dimensions is None and isinstance(architecture, list):
        dimensions = architecture
    body = value.get("body_sha256") if isinstance(value, Mapping) else None
    quantization = value.get("quantization") if isinstance(value, Mapping) else None
    payload = quantization.get("payload_sha256") if isinstance(quantization, Mapping) else None
    if (
        dimensions != challenger.DIMENSIONS
        or qualification.SHA256_RE.fullmatch(str(body)) is None
        or qualification.SHA256_RE.fullmatch(str(payload)) is None
    ):
        raise DualFinalError("candidate runtime identity/architecture is invalid")
    return {
        "architecture": ARCHITECTURE,
        "runtime_body_sha256": body,
        "payload_sha256": payload,
    }


def _uncontended_timing(timing: Mapping[str, Any]) -> Record:
    try:
        deployment_preflight.maintained.validate_timing_receipt(timing)
    except Exception as error:
        raise DualFinalError("deployment preflight timing receipt is invalid") from error
    samples = [
        row for row in timing["samples"] if row.get("process_count") == 1
    ]
    if len(samples) != 2 or {row.get("color") for row in samples} != {0, 1}:
        raise DualFinalError("uncontended timing lacks exactly both colors")
    return {
        "first_max_ms": max(float(row["first_ms"]) for row in samples),
        "later_max_ms": max(float(row["later_max_ms"]) for row in samples),
    }


def _path_reference(
    mapping: Mapping[str, Any], *, relative_to: pathlib.Path,
) -> pathlib.Path | None:
    path_value = mapping.get("path")
    digest = mapping.get("sha256")
    if not isinstance(path_value, str) or qualification.SHA256_RE.fullmatch(
        str(digest)
    ) is None:
        return None
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = relative_to / path
    if path.suffix.lower() not in {".json", ".tsv"}:
        return None
    if path.is_symlink() or not path.is_file() or qualification.sha256_file(path) != digest:
        raise DualFinalError("exclusion artifact reference changed")
    return path.resolve()


def _extract_fingerprints(path: pathlib.Path) -> set[str]:
    """Extract fingerprints from an exclusion and all immutable references.

    Supported leaves are maintained opening banks, legacy TSV opening banks,
    challenger development fingerprint lists, and protected canonical-position
    rows.  JSON references are followed only when their path and SHA-256 match.
    """

    pending = [path.resolve()]
    visited: set[pathlib.Path] = set()
    fingerprints: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > 4_096:
            raise DualFinalError("exclusion reference closure is unexpectedly large")
        if current.is_symlink() or not current.is_file():
            raise DualFinalError("exclusion reference is absent or redirected")
        raw = current.read_bytes()
        if current.suffix.lower() == ".tsv":
            try:
                bank = openings.load_exclusion_bank(current)
            except Exception as error:
                raise DualFinalError("legacy opening exclusion is invalid") from error
            fingerprints.update(bank["fingerprints"])
            continue
        if current.suffix.lower() != ".json":
            # Non-JSON source/runtime/binary references remain identity-bound
            # by their parent record but cannot contain opening states.
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DualFinalError("exclusion closure contains an unsupported artifact") from error
        if not isinstance(value, Mapping):
            raise DualFinalError("exclusion JSON is not an object")
        schema = value.get("schema")
        if schema == openings.BANK_SCHEMA:
            bank = openings.validate_bank(current)
            fingerprints.update(
                fingerprint
                for row in bank["openings"]
                for fingerprint in row["fingerprints"].values()
            )
            # exclusion_sources in an opening bank are provenance labels.  Old
            # banks intentionally store their copied TSV basename, not a path
            # relative to the bank, so they must not be followed here.
            continue
        direct = value.get("fingerprints")
        if isinstance(direct, list):
            if any(qualification.SHA256_RE.fullmatch(str(item)) is None for item in direct):
                raise DualFinalError("exclusion fingerprint list is malformed")
            fingerprints.update(str(item) for item in direct)
            continue
        rows = value.get("rows")
        if isinstance(rows, list) and rows:
            canonical = [
                row.get("canonical_sha256") for row in rows
                if isinstance(row, Mapping) and "canonical_sha256" in row
            ]
            if canonical:
                if len(canonical) != len(rows) or any(
                    qualification.SHA256_RE.fullmatch(str(item)) is None
                    for item in canonical
                ):
                    raise DualFinalError("protected canonical fingerprint rows are malformed")
                fingerprints.update(str(item) for item in canonical)
                continue

        # Receipt/manifest schemas expose their actual state carriers through a
        # small number of named records.  Do not recursively walk unrelated
        # compiler, source, training, or historical provenance closures.
        records: list[Mapping[str, Any]] = []
        if schema == (
            "papersoccer.compact-value-bfm.discrete-v3-development-"
            "recovery-mixed-six-exclusion.v1"
        ):
            records.extend(
                item for item in value.get("historical_exclusions", [])
                if isinstance(item, Mapping)
            )
            records.extend(
                item["bank"] for item in value.get("selected_banks", [])
                if isinstance(item, Mapping) and isinstance(item.get("bank"), Mapping)
            )
            additional = value.get("additional_development_exclusions", {})
            spent = additional.get("spent_original_tuple_confirmation", {}) \
                if isinstance(additional, Mapping) else {}
            if isinstance(spent, Mapping) and isinstance(spent.get("bank"), Mapping):
                records.append(spent["bank"])
            protected = value.get("protected_fingerprint_source")
            if isinstance(protected, Mapping):
                records.append(protected)
        elif schema == (
            "papersoccer.compact-value-bfm.discrete-v3-fresh-position-"
            "exclusion-audit.v1"
        ):
            references = value.get("references", {})
            if isinstance(references, Mapping):
                development_banks = references.get("development_banks", {})
                if isinstance(development_banks, Mapping):
                    records.extend(
                        item for item in development_banks.values()
                        if isinstance(item, Mapping)
                    )
                protected = references.get("protected_canonical_fingerprints")
                if isinstance(protected, Mapping):
                    records.append(protected)
        for record in records:
            referenced = _path_reference(record, relative_to=current.parent)
            if referenced is not None and referenced not in visited:
                pending.append(referenced)
    return fingerprints


def _normalize_exclusions(
    paths: Sequence[pathlib.Path], required_sha256: Sequence[str],
    *, fingerprint_loader: FingerprintLoader,
) -> tuple[list[Record], set[str]]:
    records = [_record(path) for path in paths]
    observed = {record["sha256"] for record in records}
    required = set(required_sha256)
    if len(records) != len(observed) or observed != required:
        raise DualFinalError(
            "exclusion sources must be one distinct artifact for every authorized SHA-256"
        )
    fingerprints: set[str] = set()
    for record in records:
        fingerprints.update(fingerprint_loader(pathlib.Path(record["path"])))
    if not fingerprints:
        raise DualFinalError("complete exclusion union is empty")
    return sorted(records, key=lambda item: item["sha256"]), fingerprints


def _discover_authorized_exclusions(
    validated: Mapping[str, Any], authorization: Mapping[str, Any],
) -> list[pathlib.Path]:
    context = validated.get("context")
    if not isinstance(context, Mapping) or not isinstance(context.get("inputs"), Mapping):
        raise DualFinalError("authorized exclusion paths were not supplied or discoverable")
    try:
        entries = challenger.load_ledger(context["plan"])
        records: list[Mapping[str, Any]] = [
            *context["inputs"]["protected_exclusions"].values(),
            *context["inputs"]["live_exclusions"].values(),
            *(
                entry["development_exclusion"] for entry in entries
                if entry.get("event") == "attempt-outcome-recorded"
            ),
            *challenger._cumulative_dynamic_exclusions(entries),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise DualFinalError("could not discover authorized exclusion paths") from error
    by_sha: dict[str, pathlib.Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise DualFinalError("campaign ledger has a malformed exclusion record")
        try:
            if record.get("schema") == challenger.DYNAMIC_EXCLUSION_SCHEMA:
                path = challenger._verify_dynamic_exclusion_record(
                    record, "authorized dynamic exclusion"
                )
            elif set(record) == {"route", "bytes", "sha256"}:
                path = challenger._resolve_campaign_artifact(
                    record, plan=context["plan"],
                    label="authorized bundled exclusion",
                )
            elif {"path", "bytes", "sha256"}.issubset(record):
                path = challenger._resolve_campaign_artifact(
                    {key: record[key] for key in ("path", "bytes", "sha256")},
                    plan=context["plan"], label="authorized exclusion",
                )
            else:
                raise DualFinalError("campaign exclusion record has no resolvable route")
        except (OSError, ValueError) as error:
            raise DualFinalError("could not resolve an authorized exclusion") from error
        by_sha.setdefault(str(record.get("sha256")), path)
    required = set(authorization.get("required_exclusion_sha256", []))
    if set(by_sha) != required:
        missing = required - set(by_sha)
        if missing:
            raise DualFinalError("campaign ledger omits an authorized exclusion path")
        by_sha = {digest: by_sha[digest] for digest in required}
    return [by_sha[digest] for digest in sorted(required)]


def _preflight_compile_binding(state: Mapping[str, Any]) -> Record:
    direct = state.get("compile_binding")
    if isinstance(direct, Mapping):
        candidate = state.get("candidate")
        reference = state.get("reference")
        if (
            not isinstance(candidate, Mapping)
            or not isinstance(reference, Mapping)
            or direct.get("candidate_sha256") != candidate.get("sha256")
            or direct.get("candidate_embedded") is not True
            or direct.get("gate") != reference.get("gate")
            or not isinstance(direct.get("compiler"), Mapping)
            or qualification.SHA256_RE.fullmatch(
                str(direct.get("command_sha256"))
            ) is None
        ):
            raise DualFinalError("generalized release compile binding is incomplete")
        gate_path = _verify_record(
            direct["gate"], "release Rank-4 gate", executable=True
        )
        return {**dict(direct), "gate": _record(gate_path, executable=True)}
    receipt = state.get("receipt")
    plan = state.get("plan")
    if not isinstance(receipt, Mapping) or not isinstance(plan, Mapping):
        raise DualFinalError("deployment preflight lacks compile ancestry")
    command = receipt.get("commands", {}).get("compile_rank4_gate")
    gate = receipt.get("binaries", {}).get("rank4_gate")
    compiler = plan.get("inputs", {}).get("tools", {}).get("clang")
    candidate = state.get("candidate")
    if (
        not isinstance(command, Mapping)
        or not isinstance(gate, Mapping)
        or not isinstance(compiler, Mapping)
        or not isinstance(candidate, Mapping)
        or command.get("passed") is not True
        or command.get("argv") is None
        or gate != state.get("reference", {}).get("gate")
    ):
        raise DualFinalError("source-specific gate compile evidence is incomplete")
    argv = command["argv"]
    macro = f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{candidate["path"]}"'
    if not isinstance(argv, list) or macro not in argv:
        raise DualFinalError("gate compile did not embed the exact candidate path")
    gate_path = _verify_record(gate, "preflight Rank-4 gate", executable=True)
    return {
        "compiler": dict(compiler),
        "command_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(dict(command))
        ),
        "candidate_sha256": candidate["sha256"],
        "gate": _record(gate_path, executable=True),
        "candidate_embedded": True,
    }


def prepare_execution(
    *, authorization_path: pathlib.Path, campaign_plan_path: pathlib.Path,
    output_root: pathlib.Path,
    deployment_preflight_path: pathlib.Path | None = None,
    release_evidence_path: pathlib.Path | None = None,
    ci_path: pathlib.Path, rank4_source: pathlib.Path,
    exclusion_paths: Sequence[pathlib.Path], created_at_utc: str,
    authorization_validator: AuthorizationValidator = _default_authorization_validator,
    preflight_validator: PreflightValidator | None = None,
    ci_validator: CiValidator = _default_ci_validator,
    fingerprint_loader: FingerprintLoader = _extract_fingerprints,
) -> pathlib.Path:
    """Freeze all execution inputs before either protected bank is claimed."""

    validated = dict(authorization_validator(authorization_path, campaign_plan_path))
    authorization = validated.get("authorization")
    if not isinstance(authorization, Mapping):
        raise DualFinalError("authorization validator returned no authorization")
    if qualification._utc(
        created_at_utc, "execution plan time"
    ) < qualification._utc(
        authorization.get("created_at_utc"), "dual-final authorization time"
    ):
        raise DualFinalError("execution plan predates dual-final authorization")
    if not exclusion_paths:
        exclusion_paths = _discover_authorized_exclusions(validated, authorization)
    preflight_kind, preflight_schema, preflight_path = _preflight_authority(
        deployment_preflight_path=deployment_preflight_path,
        release_evidence_path=release_evidence_path,
    )
    if preflight_kind == "rank4-teacher-release":
        release_record = authorization.get("release_evidence")
        if (
            not isinstance(release_record, Mapping)
            or release_record.get("path") != str(preflight_path.resolve())
            or release_record.get("sha256") != qualification.sha256_file(preflight_path)
        ):
            raise DualFinalError("execution release evidence differs from authorization")
    if preflight_validator is None:
        if preflight_kind == "discrete-v3-deployment-preflight":
            preflight_validator = _default_preflight_validator
        else:
            preflight_validator = lambda candidate: _default_release_preflight_validator(
                candidate, campaign_plan_path=campaign_plan_path,
                authorization=authorization,
            )
    root = _safe_root(output_root, create=True)
    plan_path = root / "execution-plan.json"
    if plan_path.exists():
        existing = validate_execution_plan(
            plan_path, authorization_validator=authorization_validator,
            preflight_validator=preflight_validator, ci_validator=ci_validator,
            fingerprint_loader=fingerprint_loader,
        )
        supplied_exclusions = sorted(
            (_record(item) for item in exclusion_paths),
            key=lambda item: item["sha256"],
        )
        if (
            existing["plan"].get("campaign_plan")
            != _reference(campaign_plan_path, challenger.PLAN_SCHEMA)
            or existing["plan"].get("authorization")
            != _reference(
                authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
            )
            or existing["plan"].get("preflight_kind") != preflight_kind
            or existing["plan"].get("preflight_schema") != preflight_schema
            or existing["plan"].get("deployment_preflight")
            != _reference(preflight_path, preflight_schema)
            or existing["plan"].get("ci") != _reference(ci_path, upload.CI_SCHEMA)
            or existing["plan"].get("rank4")
            != _record(rank4_source, ascii_required=True)
            or existing["plan"].get("exclusion_sources") != supplied_exclusions
        ):
            raise DualFinalError("existing execution plan was invoked with other inputs")
        return plan_path
    if any(root.iterdir()):
        raise DualFinalError("dual-final execution outputs predate the execution plan")
    candidate = authorization.get("candidate")
    if not isinstance(candidate, Mapping):
        raise DualFinalError("authorization candidate is absent")
    source_path = _verify_record(
        candidate.get("source"), "authorized candidate source", ascii_required=True
    )
    runtime_path = _verify_record(
        candidate.get("runtime"), "authorized candidate runtime"
    )
    runtime_identity = _runtime_identity(runtime_path)
    expected_architecture = {
        "id": ARCHITECTURE,
        "dimensions": challenger.DIMENSIONS,
        "biases": False,
        "outputs": 1,
        "head": "scalar-value-only",
        "policy_head": False,
        "runtime_body_sha256": runtime_identity["runtime_body_sha256"],
        "payload_sha256": runtime_identity["payload_sha256"],
    }
    if (
        candidate.get("architecture") != expected_architecture
        or not 0 < candidate["source"]["bytes"] < challenger.SOURCE_LIMIT
    ):
        raise DualFinalError("authorized candidate architecture/source limit changed")
    preflight = dict(preflight_validator(preflight_path))
    if (
        preflight.get("candidate", {}).get("sha256")
        != candidate["source"]["sha256"]
        or preflight.get("candidate", {}).get("bytes")
        != candidate["source"]["bytes"]
        or preflight.get("runtime", {}).get("sha256")
        != candidate["runtime"]["sha256"]
        or preflight.get("runtime", {}).get("bytes")
        != candidate["runtime"]["bytes"]
    ):
        raise DualFinalError("deployment preflight belongs to another candidate")
    candidate_commit = preflight.get("candidate_commit")
    if qualification.COMMIT_RE.fullmatch(str(candidate_commit)) is None:
        raise DualFinalError("deployment preflight candidate commit is invalid")
    ci = dict(ci_validator(ci_path, str(candidate_commit)))
    if ci.get("head_sha") != candidate_commit or ci.get("conclusion") != "success":
        raise DualFinalError("CI evidence belongs to another candidate or did not pass")
    if (
        preflight_kind == "rank4-teacher-release"
        and preflight.get("ci") != _reference(ci_path, upload.CI_SCHEMA)
    ):
        raise DualFinalError("release evidence and supplied CI identity differ")
    rank4 = _record(rank4_source, ascii_required=True)
    if (
        rank4["sha256"] != qualification.RANK4_SHA256
        or rank4["bytes"] != qualification.RANK4_BYTES
    ):
        raise DualFinalError("maintained Rank-4 source identity changed")
    exclusions, fingerprints = _normalize_exclusions(
        exclusion_paths, authorization.get("required_exclusion_sha256", []),
        fingerprint_loader=fingerprint_loader,
    )
    source_binding_path = root / "source-binding.json"
    qualification.create_source_binding(
        source_binding_path, candidate_source=source_path,
        candidate_commit=str(candidate_commit), rank4_source=rank4_source,
        opponent_source=rank4_source,
    )
    compile_binding = _preflight_compile_binding(preflight)
    gate_path = _verify_record(
        compile_binding["gate"], "source-specific Rank-4 gate", executable=True
    )
    configuration = preflight.get("derivation", {}).get("configuration")
    if not isinstance(configuration, Mapping) or deployment.deployment_configuration(
        configuration.get("tuple", []), configuration.get("profile"),
        deployment.PROFILE_ROSTER.get(configuration.get("profile")),
    ) != configuration:
        raise DualFinalError("deployment search configuration changed")
    timing = _uncontended_timing(preflight.get("timing", {}))
    if preflight.get("uncontended_timing", timing) != timing:
        raise DualFinalError("preflight uncontended timing summary changed")
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "dual-final-execution-planned-banks-unclaimed",
        "created_at_utc": _utc(created_at_utc, "execution plan time"),
        "root": str(root),
        "campaign_plan": _reference(campaign_plan_path, challenger.PLAN_SCHEMA),
        "authorization": _reference(
            authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
        ),
        "attempt": authorization["attempt"],
        "candidate": dict(candidate),
        "candidate_commit": candidate_commit,
        "runtime_identity": runtime_identity,
        "rank4": rank4,
        "source_binding": _reference(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        ),
        "deployment_preflight": _reference(
            preflight_path, preflight_schema
        ),
        "preflight_kind": preflight_kind,
        "preflight_schema": preflight_schema,
        "ci": _reference(ci_path, upload.CI_SCHEMA),
        "compile_binding": compile_binding,
        "gate": _record(gate_path, executable=True),
        "repository": str(pathlib.Path(
            preflight["plan"]["inputs"]["repository"]
        ).resolve()),
        "configuration": dict(configuration),
        "uncontended_timing": timing,
        "exclusion_sources": exclusions,
        "exclusion_fingerprint_count": len(fingerprints),
        "exclusion_fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(fingerprints))
        ),
        "gate_contract": {
            "gate_ids": list(GATE_IDS), "pairs_per_gate": 500,
            "games_per_gate": 1_000, "shards_per_gate": SHARDS,
            "pairs_per_shard": PAIRS_PER_SHARD, "workers_per_gate": WORKERS,
            "threads_per_worker": THREADS_PER_WORKER,
            "gates_concurrent": False, "gate_b_excludes_gate_a": True,
        },
        "thresholds": challenger.FINAL_THRESHOLDS,
        "policy": {
            "bank_claim_before_entropy": True,
            "started_bank_retry_authorized": False,
            "started_shard_retry_authorized": False,
            "candidate_change_authorized": False,
            "upload_authorized": False,
        },
    }
    _write_sealed_once(plan_path, body)
    validate_execution_plan(
        plan_path, authorization_validator=authorization_validator,
        preflight_validator=preflight_validator, ci_validator=ci_validator,
        fingerprint_loader=fingerprint_loader,
    )
    return plan_path


def validate_execution_plan(
    path: pathlib.Path, *,
    authorization_validator: AuthorizationValidator = _default_authorization_validator,
    preflight_validator: PreflightValidator | None = None,
    ci_validator: CiValidator = _default_ci_validator,
    fingerprint_loader: FingerprintLoader = _extract_fingerprints,
) -> Record:
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    root = pathlib.Path(str(plan.get("root", "")))
    root = _safe_root(root, create=False)
    expected_fields = {
        "schema", "namespace", "campaign_id", "status", "created_at_utc",
        "root", "campaign_plan", "authorization", "attempt", "candidate",
        "candidate_commit", "runtime_identity", "rank4", "source_binding",
        "deployment_preflight", "preflight_kind", "preflight_schema", "ci",
        "compile_binding", "gate",
        "repository", "configuration", "uncontended_timing",
        "exclusion_sources", "exclusion_fingerprint_count",
        "exclusion_fingerprints_sha256", "gate_contract", "thresholds",
        "policy", "body_sha256",
    }
    if (
        set(plan) != expected_fields
        or plan.get("namespace") != NAMESPACE
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("status") != "dual-final-execution-planned-banks-unclaimed"
        or path.is_symlink()
        or path.resolve() != (root / "execution-plan.json").resolve()
    ):
        raise DualFinalError("execution plan route changed")
    _utc(plan.get("created_at_utc"), "execution plan time")
    campaign_plan = _verify_reference(
        plan.get("campaign_plan"), challenger.PLAN_SCHEMA, "campaign plan"
    )
    authorization_path = _verify_reference(
        plan.get("authorization"), challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
        "dual-final authorization",
    )
    validated = dict(authorization_validator(authorization_path, campaign_plan))
    authorization = validated.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or plan.get("attempt") != authorization.get("attempt")
        or plan.get("candidate") != authorization.get("candidate")
    ):
        raise DualFinalError("execution plan changed the authorized candidate")
    if qualification._utc(
        plan["created_at_utc"], "execution plan time"
    ) < qualification._utc(
        authorization.get("created_at_utc"), "dual-final authorization time"
    ):
        raise DualFinalError("execution plan predates authorization")
    source = _verify_record(
        plan["candidate"]["source"], "planned candidate source", ascii_required=True
    )
    runtime = _verify_record(plan["candidate"]["runtime"], "planned candidate runtime")
    if plan.get("runtime_identity") != _runtime_identity(runtime):
        raise DualFinalError("planned runtime identity changed")
    rank4 = _verify_record(plan.get("rank4"), "planned Rank-4", ascii_required=True)
    if qualification.sha256_file(rank4) != qualification.RANK4_SHA256:
        raise DualFinalError("planned Rank-4 identity changed")
    binding_path = _verify_reference(
        plan.get("source_binding"), qualification.SOURCE_BINDING_SCHEMA,
        "execution source binding",
    )
    binding = qualification.load_sealed(binding_path, qualification.SOURCE_BINDING_SCHEMA)
    qualification.validate_source_binding(binding)
    if (
        binding.get("candidate_commit") != plan.get("candidate_commit")
        or binding.get("candidate", {}).get("sha256") != qualification.sha256_file(source)
    ):
        raise DualFinalError("execution source binding changed")
    preflight_kind = plan.get("preflight_kind")
    preflight_schema = plan.get("preflight_schema")
    expected_authorities = {
        "discrete-v3-deployment-preflight": deployment_preflight.REFERENCE_SCHEMA,
        "rank4-teacher-release": challenger.RELEASE_EVIDENCE_SCHEMA,
    }
    if expected_authorities.get(preflight_kind) != preflight_schema:
        raise DualFinalError("execution preflight authority changed")
    preflight_path = _verify_reference(
        plan.get("deployment_preflight"), str(preflight_schema),
        "execution preflight authority",
    )
    if preflight_kind == "rank4-teacher-release":
        release_record = authorization.get("release_evidence")
        if (
            not isinstance(release_record, Mapping)
            or release_record.get("path") != str(preflight_path)
            or release_record.get("sha256") != qualification.sha256_file(preflight_path)
        ):
            raise DualFinalError("planned release evidence differs from authorization")
    if preflight_validator is None:
        if preflight_kind == "discrete-v3-deployment-preflight":
            preflight_validator = _default_preflight_validator
        else:
            preflight_validator = lambda candidate: _default_release_preflight_validator(
                candidate, campaign_plan_path=campaign_plan,
                authorization=authorization,
            )
    preflight = dict(preflight_validator(preflight_path))
    if (
        preflight.get("candidate_commit") != plan.get("candidate_commit")
        or preflight.get("candidate", {}).get("sha256")
        != plan["candidate"]["source"]["sha256"]
        or preflight.get("runtime", {}).get("sha256")
        != plan["candidate"]["runtime"]["sha256"]
        or plan.get("compile_binding") != _preflight_compile_binding(preflight)
        or plan.get("configuration")
        != preflight.get("derivation", {}).get("configuration")
        or plan.get("repository") != str(pathlib.Path(
            preflight["plan"]["inputs"]["repository"]
        ).resolve())
        or plan.get("uncontended_timing") != _uncontended_timing(preflight["timing"])
        or preflight.get(
            "uncontended_timing", plan.get("uncontended_timing")
        ) != plan.get("uncontended_timing")
    ):
        raise DualFinalError("execution preflight binding changed")
    gate = _verify_record(plan.get("gate"), "planned gate", executable=True)
    if plan["compile_binding"]["gate"] != _record(gate, executable=True):
        raise DualFinalError("planned gate differs from compiled gate")
    ci_path = _verify_reference(plan.get("ci"), upload.CI_SCHEMA, "green CI")
    ci = dict(ci_validator(ci_path, str(plan.get("candidate_commit"))))
    if ci.get("head_sha") != plan.get("candidate_commit"):
        raise DualFinalError("planned CI head changed")
    if (
        preflight_kind == "rank4-teacher-release"
        and preflight.get("ci") != _reference(ci_path, upload.CI_SCHEMA)
    ):
        raise DualFinalError("planned release/CI binding changed")
    exclusions, fingerprints = _normalize_exclusions(
        [pathlib.Path(item["path"]) for item in plan.get("exclusion_sources", [])],
        authorization.get("required_exclusion_sha256", []),
        fingerprint_loader=fingerprint_loader,
    )
    if (
        exclusions != plan.get("exclusion_sources")
        or len(fingerprints) != plan.get("exclusion_fingerprint_count")
        or qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(fingerprints))
        ) != plan.get("exclusion_fingerprints_sha256")
        or plan.get("gate_contract") != {
            "gate_ids": list(GATE_IDS), "pairs_per_gate": 500,
            "games_per_gate": 1_000, "shards_per_gate": SHARDS,
            "pairs_per_shard": PAIRS_PER_SHARD, "workers_per_gate": WORKERS,
            "threads_per_worker": THREADS_PER_WORKER,
            "gates_concurrent": False, "gate_b_excludes_gate_a": True,
        }
        or plan.get("thresholds") != challenger.FINAL_THRESHOLDS
        or plan.get("policy") != {
            "bank_claim_before_entropy": True,
            "started_bank_retry_authorized": False,
            "started_shard_retry_authorized": False,
            "candidate_change_authorized": False,
            "upload_authorized": False,
        }
    ):
        raise DualFinalError("execution plan exclusion/resource policy changed")
    return {
        "plan": plan, "path": path.resolve(), "root": root,
        "authorization": dict(authorization), "campaign_plan": campaign_plan,
        "preflight": preflight, "ci": ci, "fingerprints": fingerprints,
        "source_binding_path": binding_path,
    }


def _gate_root(plan: Mapping[str, Any], gate_id: str) -> pathlib.Path:
    if gate_id not in GATE_IDS:
        raise DualFinalError("unknown dual-final gate")
    root = pathlib.Path(plan["root"])
    gates = root / "gates"
    result = gates / gate_id
    if (
        gates.is_symlink() or result.is_symlink()
        or (gates.exists() and not gates.is_dir())
        or (result.exists() and not result.is_dir())
    ):
        raise DualFinalError("unsafe dual-final gate output route")
    return result


def _bank_claim_body(
    plan_path: pathlib.Path, plan: Mapping[str, Any], *, gate_id: str,
    claimed_at_utc: str, exclusion_sources: Sequence[Mapping[str, Any]],
    exclusion_fingerprints: set[str],
) -> Record:
    return {
        "schema": BANK_CLAIM_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": plan["attempt"],
        "gate_id": gate_id,
        "status": "protected-bank-claimed-before-entropy",
        "claimed_at_utc": _utc(claimed_at_utc, f"{gate_id} bank claim time"),
        "execution_plan": _reference(plan_path, PLAN_SCHEMA),
        "authorization": plan["authorization"],
        "candidate": {
            "runtime_sha256": plan["candidate"]["runtime"]["sha256"],
            "source_sha256": plan["candidate"]["source"]["sha256"],
        },
        "exclusion_sources": [dict(item) for item in exclusion_sources],
        "exclusion_fingerprint_count": len(exclusion_fingerprints),
        "exclusion_fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(exclusion_fingerprints))
        ),
        "entropy_draws_authorized": 1,
        "protected_banks_authorized": 1,
        "one_shot": True, "retry_authorized": False,
        "upload_authorized": False,
    }


def _opening_fingerprints(bank: Mapping[str, Any]) -> set[str]:
    return {
        str(value)
        for opening in bank.get("openings", [])
        for value in opening.get("fingerprints", {}).values()
    }


def _canonical_fingerprints(bank: Mapping[str, Any]) -> list[str]:
    values = sorted({
        str(opening.get("fingerprints", {}).get("canonical"))
        for opening in bank.get("openings", [])
    })
    if (
        len(values) != 500
        or any(qualification.SHA256_RE.fullmatch(value) is None for value in values)
    ):
        raise DualFinalError("protected bank canonical fingerprint roster is invalid")
    return values


def _write_fingerprint_exclusion(
    state: Mapping[str, Any], *, gate_id: str, bank_path: pathlib.Path,
    bank: Mapping[str, Any],
) -> pathlib.Path:
    """Publish the only final-bank artifact eligible for later exclusion use."""

    plan = state["plan"]
    values = _canonical_fingerprints(bank)
    path = _gate_root(plan, gate_id) / "fingerprint-exclusion.json"
    _write_sealed_once(path, {
        "schema": FINGERPRINT_EXCLUSION_SCHEMA,
        "namespace": NAMESPACE, "campaign_id": CAMPAIGN_ID,
        "attempt": plan["attempt"], "gate_id": gate_id,
        "classification": "protected-final-canonical-fingerprints",
        "domain": "protected-final-opening-canonical-state",
        "origin": {
            "candidate_source_sha256": plan["candidate"]["source"]["sha256"],
            "candidate_runtime_sha256": plan["candidate"]["runtime"]["sha256"],
            "protected_bank_sha256": qualification.sha256_file(bank_path),
            "seed_sha256": bank["seed_receipt"]["sha256"],
        },
        "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
        "fingerprints": values,
        "fingerprint_count": len(values),
        "contains_transcripts": False, "contains_metrics": False,
        "contains_labels": False, "training_eligible": False,
        "required_for_all_later_development_and_protected_banks": True,
    })
    validate_fingerprint_exclusion(
        path, state=state, gate_id=gate_id, bank_path=bank_path, bank=bank
    )
    return path


def validate_fingerprint_exclusion(
    path: pathlib.Path, *, state: Mapping[str, Any], gate_id: str,
    bank_path: pathlib.Path, bank: Mapping[str, Any],
) -> Record:
    value = qualification.load_sealed(path, FINGERPRINT_EXCLUSION_SCHEMA)
    plan = state["plan"]
    values = _canonical_fingerprints(bank)
    origin = value.get("origin")
    if (
        path.is_symlink()
        or path.resolve()
        != (_gate_root(plan, gate_id) / "fingerprint-exclusion.json").resolve()
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != plan["attempt"]
        or value.get("gate_id") != gate_id
        or value.get("classification")
        != "protected-final-canonical-fingerprints"
        or value.get("domain") != "protected-final-opening-canonical-state"
        or value.get("canonicalization")
        != "minimum(exact,rotate180,reflect,rotate180-reflect)"
        or not isinstance(origin, Mapping)
        or origin.get("candidate_source_sha256")
        != plan["candidate"]["source"]["sha256"]
        or origin.get("candidate_runtime_sha256")
        != plan["candidate"]["runtime"]["sha256"]
        or origin.get("protected_bank_sha256")
        != qualification.sha256_file(bank_path)
        or origin.get("seed_sha256")
        != bank.get("seed_receipt", {}).get("sha256")
        or value.get("fingerprints") != values
        or value.get("fingerprint_count") != 500
        or value.get("contains_transcripts") is not False
        or value.get("contains_metrics") is not False
        or value.get("contains_labels") is not False
        or value.get("training_eligible") is not False
        or value.get("required_for_all_later_development_and_protected_banks")
        is not True
        or set(value) != {
            "schema", "namespace", "campaign_id", "attempt", "gate_id",
            "classification", "domain", "origin", "canonicalization",
            "fingerprints", "fingerprint_count", "contains_transcripts",
            "contains_metrics", "contains_labels", "training_eligible",
            "required_for_all_later_development_and_protected_banks",
            "body_sha256",
        }
    ):
        raise DualFinalError(f"{gate_id} sanitized fingerprint exclusion changed")
    return value


def _materialize_one(
    state: Mapping[str, Any], *, gate_id: str, claimed_at_utc: str,
    entropy: Callable[[int], bytes], bank_generator: BankGenerator,
) -> pathlib.Path:
    plan = state["plan"]
    plan_path = pathlib.Path(state["path"])
    root = _gate_root(plan, gate_id)
    _directory(root, create=True)
    claim_path = root / "bank-claim.json"
    receipt_path = root / "bank-receipt.json"
    if receipt_path.exists():
        validate_bank_receipt(state, gate_id=gate_id)
        return receipt_path
    if claim_path.exists() or claim_path.is_symlink():
        raise DualFinalError(f"{gate_id} bank is spent without a receipt; retry forbidden")
    if any(root.iterdir()):
        raise DualFinalError(f"{gate_id} bank outputs predate its one-shot claim")
    if qualification._utc(
        claimed_at_utc, f"{gate_id} bank claim time"
    ) < qualification._utc(
        state["authorization"].get("created_at_utc"),
        "dual-final authorization time",
    ):
        raise DualFinalError(f"{gate_id} bank claim predates authorization")
    exclusions = set(state["fingerprints"])
    sources = [dict(item) for item in plan["exclusion_sources"]]
    if gate_id == "gate-b":
        first = validate_bank_receipt(state, gate_id="gate-a")
        first_bank = first["bank"]
        first_record = first["receipt"]["protected_bank"]
        exclusions.update(_opening_fingerprints(first_bank))
        sources.append(dict(first_record))
    claim = _bank_claim_body(
        plan_path, plan, gate_id=gate_id, claimed_at_utc=claimed_at_utc,
        exclusion_sources=sources, exclusion_fingerprints=exclusions,
    )
    _write_sealed_once(claim_path, claim)
    seed = entropy(32)
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise DualFinalError("protected bank entropy source did not return 256 bits")
    if gate_id == "gate-b":
        first_seed = validate_bank_receipt(state, gate_id="gate-a")["seed"]
        if seed.hex() == first_seed["seed_256_hex"]:
            raise DualFinalError("Gate B reused Gate A entropy")
    seed_path = root / "seed-receipt.json"
    _write_sealed_once(seed_path, {
        "schema": openings.SEED_SCHEMA, "namespace": NAMESPACE,
        "status": "protected-seed-frozen-before-bank-generation",
        "created_at_utc": _utc(claimed_at_utc, f"{gate_id} seed time"),
        "seed_256_hex": seed.hex(),
        "source_binding": plan["source_binding"],
        "clean_binding": plan["deployment_preflight"],
        "candidate_commit": plan["candidate_commit"],
        "candidate_sha256": plan["candidate"]["source"]["sha256"],
        "exclusions_body_sha256": claim["exclusion_fingerprints_sha256"],
        "exclusion_sources": sources,
        "exclusion_fingerprint_count": len(exclusions),
        "entropy_bits": 256, "bank_generated": False,
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
    })
    generated = bank_generator(
        stage="protected_final", count=500, seed=seed,
        excluded_fingerprints=set(exclusions),
    )
    if not isinstance(generated, list) or len(generated) != 500:
        raise DualFinalError("opening generator did not return exactly 500 openings")
    opening_directory = root / "opening-bank"
    _directory(opening_directory, create=True)
    bank_path = openings.write_bank(
        opening_directory,
        openings.bank_payload(
            stage="protected_final", classification="protected-final",
            seed=seed,
            exclusions={
                "body_sha256": claim["exclusion_fingerprints_sha256"],
                "sources": sources,
            },
            openings=generated, source_binding=plan["source_binding"],
            seed_receipt=_reference(seed_path, openings.SEED_SCHEMA),
        ),
    )
    bank = openings.validate_bank(bank_path)
    produced = _opening_fingerprints(bank)
    if produced & exclusions or len(bank["openings"]) != 500:
        raise DualFinalError(f"{gate_id} generated bank overlaps its exclusions")
    gate_bank_directory = root / "gate-bank"
    _directory(gate_bank_directory, create=True)
    gate_bank_path = final_tools._materialize_gate_bank(
        gate_bank_directory, bank_path
    )
    adapter_path = root / "bank-adapter.json"
    _write_sealed_once(adapter_path, {
        "schema": qualification.FINAL_BANK_SCHEMA,
        "namespace": NAMESPACE, "classification": "fresh-protected-final",
        "source_binding": plan["source_binding"],
        "candidate_commit": plan["candidate_commit"],
        "candidate_sha256": plan["candidate"]["source"]["sha256"],
        "rank4_sha256": qualification.RANK4_SHA256,
        "opening_count": 500,
        "protected_bank": _record(bank_path),
        "gate_bank": _record(gate_bank_path),
        "bank_claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "seed_receipt": _reference(seed_path, openings.SEED_SCHEMA),
    })
    gate_binding_path = root / "gate-binding.json"
    qualification.create_gate_binding(
        gate_binding_path,
        source_binding_path=state["source_binding_path"],
        bank_path=adapter_path,
        harness_path=pathlib.Path(plan["gate"]["path"]),
    )
    fingerprint_exclusion_path = _write_fingerprint_exclusion(
        state, gate_id=gate_id, bank_path=bank_path, bank=bank,
    )
    _write_sealed_once(receipt_path, {
        "schema": BANK_RECEIPT_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": plan["attempt"],
        "gate_id": gate_id,
        "status": "protected-bank-materialized-unconsumed",
        "execution_plan": _reference(plan_path, PLAN_SCHEMA),
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "seed_receipt": _reference(seed_path, openings.SEED_SCHEMA),
        "source_binding": plan["source_binding"],
        "candidate": {
            "runtime_sha256": plan["candidate"]["runtime"]["sha256"],
            "source_sha256": plan["candidate"]["source"]["sha256"],
        },
        "protected_bank": _record(bank_path),
        "gate_bank": _record(gate_bank_path),
        "bank_adapter": _reference(adapter_path, qualification.FINAL_BANK_SCHEMA),
        "gate_binding": _reference(
            gate_binding_path, qualification.GATE_BINDING_SCHEMA
        ),
        "fingerprint_exclusion": _reference(
            fingerprint_exclusion_path, FINGERPRINT_EXCLUSION_SCHEMA
        ),
        "opening_count": 500,
        "exclusion_sources": sources,
        "exclusion_fingerprint_count": len(exclusions),
        "exclusion_fingerprints_sha256": claim["exclusion_fingerprints_sha256"],
        "four_way_overlap_count": 0,
        "bank_consumed": False, "retry_authorized": False,
        "upload_authorized": False,
    })
    validate_bank_receipt(state, gate_id=gate_id)
    return receipt_path


def validate_bank_receipt(state: Mapping[str, Any], *, gate_id: str) -> Record:
    plan = state["plan"]
    root = _gate_root(plan, gate_id)
    receipt_path = root / "bank-receipt.json"
    if receipt_path.is_symlink():
        raise DualFinalError(f"{gate_id} bank receipt is redirected")
    receipt = qualification.load_sealed(receipt_path, BANK_RECEIPT_SCHEMA)
    claim_path = _verify_reference(
        receipt.get("claim"), BANK_CLAIM_SCHEMA, f"{gate_id} bank claim"
    )
    seed_path = _verify_reference(
        receipt.get("seed_receipt"), openings.SEED_SCHEMA,
        f"{gate_id} seed receipt",
    )
    seed = qualification.load_sealed(seed_path, openings.SEED_SCHEMA)
    bank_path = _verify_record(receipt.get("protected_bank"), f"{gate_id} bank")
    gate_bank_path = _verify_record(
        receipt.get("gate_bank"), f"{gate_id} gate bank"
    )
    bank = openings.validate_bank(bank_path)
    gate_support.validate_bank(gate_bank_path)
    adapter_path = _verify_reference(
        receipt.get("bank_adapter"), qualification.FINAL_BANK_SCHEMA,
        f"{gate_id} bank adapter",
    )
    adapter = qualification.load_sealed(
        adapter_path, qualification.FINAL_BANK_SCHEMA
    )
    binding_path = _verify_reference(
        receipt.get("gate_binding"), qualification.GATE_BINDING_SCHEMA,
        f"{gate_id} gate binding",
    )
    binding = qualification.load_sealed(
        binding_path, qualification.GATE_BINDING_SCHEMA
    )
    source_binding = qualification.load_sealed(
        state["source_binding_path"], qualification.SOURCE_BINDING_SCHEMA
    )
    qualification.validate_source_binding(source_binding)
    fingerprint_exclusion_path = _verify_reference(
        receipt.get("fingerprint_exclusion"), FINGERPRINT_EXCLUSION_SCHEMA,
        f"{gate_id} sanitized fingerprint exclusion",
    )
    validate_fingerprint_exclusion(
        fingerprint_exclusion_path, state=state, gate_id=gate_id,
        bank_path=bank_path, bank=bank,
    )
    sources = [dict(item) for item in plan["exclusion_sources"]]
    exclusions = set(state["fingerprints"])
    if gate_id == "gate-b":
        first = validate_bank_receipt(state, gate_id="gate-a")
        exclusions.update(_opening_fingerprints(first["bank"]))
        sources.append(dict(first["receipt"]["protected_bank"]))
    claim = qualification.load_sealed(claim_path, BANK_CLAIM_SCHEMA)
    expected_claim = qualification.seal(_bank_claim_body(
        pathlib.Path(state["path"]), plan, gate_id=gate_id,
        claimed_at_utc=str(claim.get("claimed_at_utc")),
        exclusion_sources=sources, exclusion_fingerprints=exclusions,
    ))
    if claim != expected_claim:
        raise DualFinalError(f"{gate_id} bank claim changed")
    produced = _opening_fingerprints(bank)
    expected_candidate = {
        "runtime_sha256": plan["candidate"]["runtime"]["sha256"],
        "source_sha256": plan["candidate"]["source"]["sha256"],
    }
    expected_seed = qualification.seal({
        "schema": openings.SEED_SCHEMA, "namespace": NAMESPACE,
        "status": "protected-seed-frozen-before-bank-generation",
        "created_at_utc": claim["claimed_at_utc"],
        "seed_256_hex": bank["seed_hex"],
        "source_binding": plan["source_binding"],
        "clean_binding": plan["deployment_preflight"],
        "candidate_commit": plan["candidate_commit"],
        "candidate_sha256": expected_candidate["source_sha256"],
        "exclusions_body_sha256": claim["exclusion_fingerprints_sha256"],
        "exclusion_sources": sources,
        "exclusion_fingerprint_count": len(exclusions),
        "entropy_bits": 256, "bank_generated": False,
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
    })
    expected_gate_raw = (
        "# papersoccer.compact-value-bfm-opening-bank.v1\n"
        "opening_id\ttranscript\n" + "".join(
            f"{opening['opening_id']}\t{opening['transcript']}\n"
            for opening in bank["openings"]
        )
    ).encode("ascii")
    expected_adapter = qualification.seal({
        "schema": qualification.FINAL_BANK_SCHEMA,
        "namespace": NAMESPACE, "classification": "fresh-protected-final",
        "source_binding": plan["source_binding"],
        "candidate_commit": plan["candidate_commit"],
        "candidate_sha256": expected_candidate["source_sha256"],
        "rank4_sha256": qualification.RANK4_SHA256,
        "opening_count": 500, "protected_bank": _record(bank_path),
        "gate_bank": _record(gate_bank_path),
        "bank_claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "seed_receipt": _reference(seed_path, openings.SEED_SCHEMA),
    })
    expected_binding = qualification.seal({
        "schema": qualification.GATE_BINDING_SCHEMA,
        "namespace": NAMESPACE,
        "candidate_commit": source_binding["candidate_commit"],
        "candidate": source_binding["candidate"],
        "rank4": source_binding["rank4"],
        "opponent": source_binding["opponent"],
        "source_binding": _reference(
            state["source_binding_path"], qualification.SOURCE_BINDING_SCHEMA
        ),
        "bank": _reference(adapter_path, qualification.FINAL_BANK_SCHEMA),
        "harness": {
            key: plan["gate"][key] for key in ("path", "bytes", "sha256")
        },
        "shards": SHARDS, "pairs_per_shard": PAIRS_PER_SHARD,
    })
    if (
        seed != expected_seed
        or adapter != expected_adapter
        or binding != expected_binding
        or gate_bank_path.read_bytes() != expected_gate_raw
        or bank.get("source_binding") != plan["source_binding"]
        or bank.get("seed_receipt") != _reference(seed_path, openings.SEED_SCHEMA)
        or bank.get("exclusion_sources") != sources
        or bank.get("exclusions_body_sha256")
        != claim["exclusion_fingerprints_sha256"]
        or produced & exclusions
        or binding.get("candidate", {}).get("sha256")
        != expected_candidate["source_sha256"]
        or binding.get("bank")
        != _reference(adapter_path, qualification.FINAL_BANK_SCHEMA)
        or binding.get("harness") != {
            key: plan["gate"][key] for key in ("path", "bytes", "sha256")
        }
    ):
        raise DualFinalError(f"{gate_id} protected bank ancestry changed")
    expected = qualification.seal({
        "schema": BANK_RECEIPT_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": plan["attempt"],
        "gate_id": gate_id,
        "status": "protected-bank-materialized-unconsumed",
        "execution_plan": _reference(pathlib.Path(state["path"]), PLAN_SCHEMA),
        "claim": _reference(claim_path, BANK_CLAIM_SCHEMA),
        "seed_receipt": _reference(seed_path, openings.SEED_SCHEMA),
        "source_binding": plan["source_binding"], "candidate": expected_candidate,
        "protected_bank": _record(bank_path),
        "gate_bank": _record(gate_bank_path),
        "bank_adapter": _reference(adapter_path, qualification.FINAL_BANK_SCHEMA),
        "gate_binding": _reference(binding_path, qualification.GATE_BINDING_SCHEMA),
        "fingerprint_exclusion": _reference(
            fingerprint_exclusion_path, FINGERPRINT_EXCLUSION_SCHEMA
        ),
        "opening_count": 500, "exclusion_sources": sources,
        "exclusion_fingerprint_count": len(exclusions),
        "exclusion_fingerprints_sha256": claim["exclusion_fingerprints_sha256"],
        "four_way_overlap_count": 0, "bank_consumed": False,
        "retry_authorized": False, "upload_authorized": False,
    })
    if receipt != expected:
        raise DualFinalError(f"{gate_id} bank receipt changed")
    return {
        "receipt": receipt, "path": receipt_path, "claim": claim,
        "seed": seed, "bank": bank, "bank_path": bank_path,
        "gate_bank_path": gate_bank_path, "binding": binding,
        "binding_path": binding_path,
        "fingerprint_exclusion_path": fingerprint_exclusion_path,
    }


def materialize_banks(
    plan_path: pathlib.Path, *, claimed_at_utc: str,
    entropy: Callable[[int], bytes] = secrets.token_bytes,
    bank_generator: BankGenerator = openings.generate_openings,
    state_validator: Callable[[pathlib.Path], Mapping[str, Any]] = validate_execution_plan,
    governance_preparer: Callable[..., pathlib.Path] = challenger.prepare_dual_final,
) -> pathlib.Path:
    """Materialize A then B and register both with challenger governance."""

    state = dict(state_validator(plan_path))
    root = pathlib.Path(state["plan"]["root"])
    lock_path = root / "bank-materialization.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise DualFinalError("another dual-final materialization is active") from error
        first_receipt = _materialize_one(
            state, gate_id="gate-a", claimed_at_utc=claimed_at_utc,
            entropy=entropy, bank_generator=bank_generator,
        )
        first = validate_bank_receipt(state, gate_id="gate-a")
        second_receipt = _materialize_one(
            state, gate_id="gate-b", claimed_at_utc=claimed_at_utc,
            entropy=entropy, bank_generator=bank_generator,
        )
        second = validate_bank_receipt(state, gate_id="gate-b")
        if first["seed"]["seed_256_hex"] == second["seed"]["seed_256_hex"]:
            raise DualFinalError("dual-final protected banks reused entropy")
        if _opening_fingerprints(first["bank"]) & _opening_fingerprints(second["bank"]):
            raise DualFinalError("dual-final protected banks overlap by symmetry")
        if first["receipt"]["protected_bank"]["sha256"] not in {
            item["sha256"] for item in second["receipt"]["exclusion_sources"]
        }:
            raise DualFinalError("Gate B receipt lost the Gate A exclusion")
        dual_reference = governance_preparer(
            pathlib.Path(state["plan"]["authorization"]["path"]),
            plan_path=pathlib.Path(state["plan"]["campaign_plan"]["path"]),
            bank_a=first["bank_path"], bank_b=second["bank_path"],
            created_at_utc=claimed_at_utc,
        )
        _write_sealed_once(root / "prepared.json", {
            "schema": PREPARED_SCHEMA,
            "namespace": NAMESPACE, "campaign_id": CAMPAIGN_ID,
            "attempt": state["plan"]["attempt"],
            "execution_plan": _reference(pathlib.Path(state["path"]), PLAN_SCHEMA),
            "dual_final_reference": _reference(
                dual_reference, challenger.DUAL_FINAL_REFERENCE_SCHEMA
            ),
            "fingerprint_exclusions": {
                "gate-a": _reference(
                    first["fingerprint_exclusion_path"],
                    FINGERPRINT_EXCLUSION_SCHEMA,
                ),
                "gate-b": _reference(
                    second["fingerprint_exclusion_path"],
                    FINGERPRINT_EXCLUSION_SCHEMA,
                ),
            },
            "candidate_unchanged": True, "independent_banks": True,
            "gate_b_excludes_gate_a": True, "games_launched": 0,
        })
        # Silence unused-variable false positives while keeping explicit proof
        # that both immutable receipts existed before governance preparation.
        del first_receipt, second_receipt
        return dual_reference
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_prepared(state: Mapping[str, Any]) -> tuple[pathlib.Path, Record]:
    path = pathlib.Path(state["plan"]["root"]) / "prepared.json"
    value = qualification.load_sealed(path, PREPARED_SCHEMA)
    dual_reference = _verify_reference(
        value.get("dual_final_reference"), challenger.DUAL_FINAL_REFERENCE_SCHEMA,
        "challenger dual-final reference",
    )
    expected_exclusions = {
        gate_id: _reference(
            validate_bank_receipt(state, gate_id=gate_id)[
                "fingerprint_exclusion_path"
            ],
            FINGERPRINT_EXCLUSION_SCHEMA,
        ) for gate_id in GATE_IDS
    }
    if (
        set(value) != {
            "schema", "namespace", "campaign_id", "attempt",
            "execution_plan", "dual_final_reference",
            "fingerprint_exclusions", "candidate_unchanged",
            "independent_banks", "gate_b_excludes_gate_a",
            "games_launched", "body_sha256",
        }
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("execution_plan")
        != _reference(pathlib.Path(state["path"]), PLAN_SCHEMA)
        or value.get("attempt") != state["plan"]["attempt"]
        or value.get("fingerprint_exclusions") != expected_exclusions
        or value.get("candidate_unchanged") is not True
        or value.get("independent_banks") is not True
        or value.get("gate_b_excludes_gate_a") is not True
        or value.get("games_launched") != 0
    ):
        raise DualFinalError("dual-final prepared receipt changed")
    return dual_reference, value


def _expected_gate_configuration(
    plan: Mapping[str, Any], *, pair_offset: int,
) -> Record:
    configured = plan["configuration"]
    return {
        "mode": "actual-clock", "pair_offset": pair_offset,
        "pair_count": PAIRS_PER_SHARD,
        "candidate_c": configured["candidate_c"],
        "candidate_fpu": configured["candidate_fpu"],
        "candidate_lambda": configured["candidate_lambda"],
        "candidate_actions": configured["candidate_actions"],
        "candidate_root_partial_paths": configured["candidate_root_partial_paths"],
        "candidate_nonroot_partial_paths": configured["candidate_nonroot_partial_paths"],
        "candidate_nodes": configured["candidate_nodes"],
        "candidate_expansions": configured["candidate_expansions"],
        "candidate_shuffle_seed": configured["candidate_shuffle_seed"],
        "candidate_clocks_ms": [800, 155], "rank4_nodes": 3_000_000,
        "rank4_clocks_ms": [800, 165], "max_turns": 320,
        "minimum_candidate_wins": -1, "minimum_wins_per_color": -1,
    }


def gate_command(
    plan: Mapping[str, Any], bank: Mapping[str, Any], index: int,
    output: pathlib.Path,
) -> list[str]:
    configuration = plan["configuration"]
    search_tuple = configuration["tuple"]
    return [
        plan["gate"]["path"], "--bank", bank["gate_bank"]["path"],
        "--expected-bank-sha256", bank["gate_bank"]["sha256"],
        "--candidate-source", plan["candidate"]["source"]["path"],
        "--expected-candidate-sha256", plan["candidate"]["source"]["sha256"],
        "--rank4-source", plan["rank4"]["path"],
        "--pair-offset", str(index * PAIRS_PER_SHARD),
        "--pair-count", str(PAIRS_PER_SHARD), "--mode", "actual-clock",
        "--candidate-c", str(search_tuple[0]),
        "--candidate-fpu", str(search_tuple[1]),
        "--candidate-lambda", str(search_tuple[2]),
        "--candidate-actions", str(configuration["candidate_actions"]),
        "--candidate-root-partial-paths",
        str(configuration["candidate_root_partial_paths"]),
        "--candidate-nonroot-partial-paths",
        str(configuration["candidate_nonroot_partial_paths"]),
        "--candidate-nodes", str(configuration["candidate_nodes"]),
        "--candidate-expansions", str(configuration["candidate_expansions"]),
        "--candidate-seed", str(configuration["candidate_shuffle_seed"]),
        "--rank4-nodes", "3000000", "--max-turns", "320",
        "--output", str(output),
    ]


def run_gate_process(spec: Mapping[str, Any]) -> pathlib.Path:
    output = pathlib.Path(spec["raw_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    completed = subprocess.run(
        spec["command"], cwd=spec["repository"], env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        check=False, timeout=3_600,
    )
    if completed.returncode not in (0, 2) or not output.is_file():
        raise DualFinalError("Rank-4 gate shard failed without complete output")
    return output


def _adapt_result(
    raw_path: pathlib.Path, *, plan: Mapping[str, Any],
    bank: Mapping[str, Any], index: int,
    result_validator: ResultValidator = gate_support.validate_result,
) -> list[Record]:
    document = dict(result_validator(
        raw_path, expected_bank_sha256=bank["gate_bank"]["sha256"],
        expected_candidate_sha256=plan["candidate"]["source"]["sha256"],
    ))
    bindings = document.get("bindings", {})
    if (
        bindings.get("candidate_runtime_body_sha256")
        != plan["runtime_identity"]["runtime_body_sha256"]
        or bindings.get("candidate_payload_sha256")
        != plan["runtime_identity"]["payload_sha256"]
        or document.get("config") != _expected_gate_configuration(
            plan, pair_offset=index * PAIRS_PER_SHARD
        )
    ):
        raise DualFinalError("raw gate result changed runtime or actual-clock config")
    games = []
    for game in document["games"]:
        failure = game["failure"]
        games.append({
            "pair_index": game["pair_index"],
            "candidate_color": game["candidate_player"],
            "candidate_win": failure is None
            and game["winner"] == game["candidate_player"],
            "turns": max(1, game["turns"]),
            "failure": None if failure is None else final_tools.FAILURE_MAP[failure],
            "first_ms": game["candidate"]["maximum_first_ms"],
            "later_max_ms": game["candidate"]["maximum_later_ms"],
        })
    return games


def _consume_gate(
    state: Mapping[str, Any], bank_state: Mapping[str, Any], *,
    gate_id: str, launched_at_utc: str,
) -> pathlib.Path:
    root = _gate_root(state["plan"], gate_id)
    ledger = root / "ledger"
    _directory(ledger, create=True)
    path = ledger / "consumption.json"
    body = {
        "schema": CONSUMPTION_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": state["plan"]["attempt"],
        "gate_id": gate_id, "status": "gate-bank-consumed-at-launch",
        "launched_at_utc": _utc(launched_at_utc, f"{gate_id} launch time"),
        "execution_plan": _reference(pathlib.Path(state["path"]), PLAN_SCHEMA),
        "bank_receipt": _reference(bank_state["path"], BANK_RECEIPT_SCHEMA),
        "fingerprint_exclusion": _reference(
            bank_state["fingerprint_exclusion_path"],
            FINGERPRINT_EXCLUSION_SCHEMA,
        ),
        "gate_binding": _reference(
            bank_state["binding_path"], qualification.GATE_BINDING_SCHEMA
        ),
        "workers": WORKERS, "threads_per_worker": THREADS_PER_WORKER,
        "one_launch_only": True, "retry_authorized": False,
        "upload_authorized": False,
    }
    if path.exists():
        existing = qualification.load_sealed(path, CONSUMPTION_SCHEMA)
        static_existing = {
            key: value for key, value in existing.items()
            if key not in {"body_sha256", "launched_at_utc"}
        }
        static_body = {
            key: value for key, value in body.items()
            if key != "launched_at_utc"
        }
        if static_existing != static_body:
            raise DualFinalError(f"{gate_id} consumption marker changed")
        consumed_at_utc = existing["launched_at_utc"]
    else:
        _write_sealed_once(path, body)
        consumed_at_utc = body["launched_at_utc"]
    # Prime the maintained primitive marker before the worker pool starts.
    # Otherwise four simultaneous first shard claims could race with different
    # second-resolution timestamps at a wall-clock boundary.
    primitive_path = ledger / "bank-consumed.json"
    primitive_body = {
        "schema": PRIMITIVE_CONSUMPTION_SCHEMA,
        "namespace": NAMESPACE,
        "binding_sha256": qualification.sha256_file(bank_state["binding_path"]),
        "bank": bank_state["binding"]["bank"],
        "consumed_at_utc": _utc(
            consumed_at_utc, f"{gate_id} primitive consumption time"
        ),
    }
    if primitive_path.exists():
        existing = qualification.load_sealed(
            primitive_path, PRIMITIVE_CONSUMPTION_SCHEMA
        )
        static_existing = {
            key: value for key, value in existing.items()
            if key not in {"body_sha256", "consumed_at_utc"}
        }
        static_body = {
            key: value for key, value in primitive_body.items()
            if key != "consumed_at_utc"
        }
        if static_existing != static_body:
            raise DualFinalError(f"{gate_id} primitive consumption changed")
    else:
        _write_sealed_once(primitive_path, primitive_body)
    return path


def _validate_consumption(
    state: Mapping[str, Any], bank_state: Mapping[str, Any], *, gate_id: str,
) -> tuple[pathlib.Path, Record, pathlib.Path, Record]:
    ledger = _gate_root(state["plan"], gate_id) / "ledger"
    path = ledger / "consumption.json"
    value = qualification.load_sealed(path, CONSUMPTION_SCHEMA)
    primitive_path = ledger / "bank-consumed.json"
    primitive = qualification.load_sealed(
        primitive_path, PRIMITIVE_CONSUMPTION_SCHEMA
    )
    if (
        set(value) != {
            "schema", "namespace", "campaign_id", "attempt", "gate_id",
            "status", "launched_at_utc", "execution_plan", "bank_receipt",
            "fingerprint_exclusion", "gate_binding", "workers",
            "threads_per_worker", "one_launch_only", "retry_authorized",
            "upload_authorized", "body_sha256",
        }
        or value.get("namespace") != NAMESPACE
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != state["plan"]["attempt"]
        or value.get("gate_id") != gate_id
        or value.get("status") != "gate-bank-consumed-at-launch"
        or value.get("execution_plan")
        != _reference(pathlib.Path(state["path"]), PLAN_SCHEMA)
        or value.get("bank_receipt")
        != _reference(bank_state["path"], BANK_RECEIPT_SCHEMA)
        or value.get("fingerprint_exclusion")
        != _reference(
            bank_state["fingerprint_exclusion_path"],
            FINGERPRINT_EXCLUSION_SCHEMA,
        )
        or value.get("gate_binding")
        != _reference(bank_state["binding_path"], qualification.GATE_BINDING_SCHEMA)
        or value.get("workers") != WORKERS
        or value.get("threads_per_worker") != THREADS_PER_WORKER
        or value.get("one_launch_only") is not True
        or value.get("retry_authorized") is not False
        or value.get("upload_authorized") is not False
    ):
        raise DualFinalError(f"{gate_id} consumption receipt changed")
    _utc(value.get("launched_at_utc"), f"{gate_id} launch time")
    if (
        set(primitive) != {
            "schema", "namespace", "binding_sha256", "bank",
            "consumed_at_utc", "body_sha256",
        }
        or primitive.get("namespace") != NAMESPACE
        or primitive.get("binding_sha256")
        != qualification.sha256_file(bank_state["binding_path"])
        or primitive.get("bank") != bank_state["binding"]["bank"]
        or primitive.get("consumed_at_utc") != value.get("launched_at_utc")
    ):
        raise DualFinalError(f"{gate_id} primitive consumption receipt changed")
    _utc(primitive.get("consumed_at_utc"), f"{gate_id} primitive launch time")
    return path, value, primitive_path, primitive


def _audit_shards(
    state: Mapping[str, Any], bank_state: Mapping[str, Any], *, gate_id: str,
    result_validator: ResultValidator,
) -> list[int]:
    ledger = _gate_root(state["plan"], gate_id) / "ledger"
    missing = []
    binding_path = bank_state["binding_path"]
    for index in range(SHARDS):
        claim_path = ledger / "claims" / f"shard-{index:03d}.json"
        receipt_path = ledger / "receipts" / f"shard-{index:03d}.json"
        if claim_path.exists():
            if not receipt_path.exists():
                raise qualification.SpentShardError(
                    f"{gate_id} shard {index} is spent without receipt; retry forbidden"
                )
            receipt = qualification.validate_shard_receipt(
                receipt_path, binding_path=binding_path, index=index
            )
            evidence_path = _verify_reference(
                receipt.get("evidence"), RAW_EVIDENCE_SCHEMA,
                f"{gate_id} shard {index} raw evidence",
            )
            evidence = qualification.load_sealed(
                evidence_path, RAW_EVIDENCE_SCHEMA
            )
            raw_path = _verify_record(
                evidence.get("raw_gate_result"),
                f"{gate_id} shard {index} raw result",
            )
            games = _adapt_result(
                raw_path, plan=state["plan"], bank=bank_state["receipt"],
                index=index, result_validator=result_validator,
            )
            if (
                evidence.get("execution_plan")
                != _reference(pathlib.Path(state["path"]), PLAN_SCHEMA)
                or evidence.get("bank_receipt")
                != _reference(bank_state["path"], BANK_RECEIPT_SCHEMA)
                or evidence.get("gate_id") != gate_id
                or evidence.get("shard_index") != index
                or evidence.get("actual_clock_configuration")
                != _expected_gate_configuration(
                    state["plan"], pair_offset=index * PAIRS_PER_SHARD
                )
                or evidence.get("normalized_games_sha256")
                != qualification.sha256_bytes(
                    qualification.canonical_json_bytes(games)
                )
                or receipt.get("games") != sorted(
                    games, key=lambda row: (row["pair_index"], row["candidate_color"])
                )
            ):
                raise DualFinalError(f"{gate_id} shard {index} evidence changed")
        elif receipt_path.exists():
            raise DualFinalError(f"{gate_id} shard receipt exists without claim")
        else:
            missing.append(index)
    return missing


def _execute_gate(
    state: Mapping[str, Any], *, gate_id: str, launched_at_utc: str,
    runner: GateRunner, result_validator: ResultValidator,
    clock: Callable[[], str],
    executor_factory: Callable[..., Any] = concurrent.futures.ThreadPoolExecutor,
) -> pathlib.Path:
    bank_state = validate_bank_receipt(state, gate_id=gate_id)
    root = _gate_root(state["plan"], gate_id)
    ledger = root / "ledger"
    _directory(ledger, create=True)
    for name in ("claims", "receipts", "raw", "raw-evidence"):
        _directory(ledger / name, create=True)
    _consume_gate(
        state, bank_state, gate_id=gate_id, launched_at_utc=launched_at_utc
    )
    missing = _audit_shards(
        state, bank_state, gate_id=gate_id,
        result_validator=result_validator,
    )
    aggregate_path = ledger / "aggregate.json"
    if aggregate_path.exists():
        if missing:
            raise DualFinalError(f"{gate_id} aggregate exists with missing shards")
        _validate_maintained_aggregate(
            aggregate_path, bank_state=bank_state,
            uncontended_timing=state["plan"]["uncontended_timing"],
        )
        return aggregate_path

    def one(index: int) -> int:
        qualification.start_final_shard(
            ledger, binding_path=bank_state["binding_path"], index=index,
            started_at_utc=clock(),
        )
        raw_path = ledger / "raw" / f"shard-{index:03d}.json"
        raw = runner({
            "gate_id": gate_id, "index": index,
            "repository": state["plan"]["repository"],
            "raw_output": str(raw_path),
            "command": gate_command(
                state["plan"], bank_state["receipt"], index, raw_path
            ),
            "workers": WORKERS, "threads_per_worker": THREADS_PER_WORKER,
        })
        if isinstance(raw, (str, os.PathLike, pathlib.Path)) and pathlib.Path(raw).is_file():
            raw_file = pathlib.Path(raw)
        else:
            qualification.atomic_write_once(
                raw_path,
                qualification.canonical_json_bytes({"injected_raw": raw}),
            )
            raw_file = raw_path
        games = _adapt_result(
            raw_file, plan=state["plan"], bank=bank_state["receipt"],
            index=index, result_validator=result_validator,
        )
        evidence_path = ledger / "raw-evidence" / f"shard-{index:03d}.json"
        _write_sealed_once(evidence_path, {
            "schema": RAW_EVIDENCE_SCHEMA, "namespace": NAMESPACE,
            "campaign_id": CAMPAIGN_ID, "attempt": state["plan"]["attempt"],
            "gate_id": gate_id,
            "execution_plan": _reference(pathlib.Path(state["path"]), PLAN_SCHEMA),
            "bank_receipt": _reference(bank_state["path"], BANK_RECEIPT_SCHEMA),
            "shard_index": index,
            "actual_clock_configuration": _expected_gate_configuration(
                state["plan"], pair_offset=index * PAIRS_PER_SHARD
            ),
            "raw_gate_result": _record(raw_file),
            "normalized_games_sha256": qualification.sha256_bytes(
                qualification.canonical_json_bytes(games)
            ),
        })
        qualification.record_shard_receipt(
            ledger, binding_path=bank_state["binding_path"], index=index,
            games=games, completed_at_utc=clock(),
            evidence=_reference(evidence_path, RAW_EVIDENCE_SCHEMA),
        )
        return index

    with executor_factory(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, index) for index in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    qualification.aggregate_final(
        ledger, binding_path=bank_state["binding_path"],
        uncontended_timing=state["plan"]["uncontended_timing"],
        completed_at_utc=clock(),
    )
    _audit_shards(
        state, bank_state, gate_id=gate_id,
        result_validator=result_validator,
    )
    _validate_maintained_aggregate(
        aggregate_path, bank_state=bank_state,
        uncontended_timing=state["plan"]["uncontended_timing"],
    )
    return aggregate_path


def _validate_maintained_aggregate(
    path: pathlib.Path, *, bank_state: Mapping[str, Any],
    uncontended_timing: Mapping[str, Any],
) -> Record:
    aggregate = qualification.load_sealed(
        path, qualification.FINAL_AGGREGATE_SCHEMA
    )
    games: list[Record] = []
    for index in range(SHARDS):
        receipt_path = path.parent / "receipts" / f"shard-{index:03d}.json"
        games.extend(qualification.validate_shard_receipt(
            receipt_path, binding_path=bank_state["binding_path"], index=index
        )["games"])
    failures = Counter(
        game["failure"] for game in games if game["failure"] is not None
    )
    reproduced = {
        "games": len(games),
        "candidate_wins": sum(game["candidate_win"] for game in games),
        "candidate_color_wins": {
            str(color): sum(
                game["candidate_win"] and game["candidate_color"] == color
                for game in games
            ) for color in (0, 1)
        },
        "failures": {
            name: failures[name] for name in qualification.FAILURE_CATEGORIES
        },
        "maximum_turns": max(game["turns"] for game in games),
        "timing": {
            "first_max_ms": max(game["first_ms"] for game in games),
            "later_max_ms": max(game["later_max_ms"] for game in games),
        },
        "uncontended_timing": dict(uncontended_timing),
    }
    if (
        set(aggregate) != {
            "schema", "namespace", "binding", "completed_at_utc", "summary",
            "verdict", "status", "body_sha256",
        }
        or path.resolve() != path.parent.resolve() / "aggregate.json"
        or aggregate.get("binding")
        != _reference(bank_state["binding_path"], qualification.GATE_BINDING_SCHEMA)
        or aggregate.get("summary", {}).get("uncontended_timing")
        != dict(uncontended_timing)
        or aggregate.get("summary") != reproduced
        or aggregate.get("verdict")
        != qualification.strict_gate_verdict(aggregate.get("summary", {}))
        or aggregate.get("status") not in {"rank4-qualified", "final-gate-failed"}
        or (aggregate.get("verdict", {}).get("passed") is True)
        != (aggregate.get("status") == "rank4-qualified")
    ):
        raise DualFinalError("maintained final aggregate changed")
    _utc(aggregate.get("completed_at_utc"), "maintained final aggregate time")
    return aggregate


def _normalized_aggregate(
    state: Mapping[str, Any], *, gate_id: str, aggregate_path: pathlib.Path,
) -> pathlib.Path:
    bank_state = validate_bank_receipt(state, gate_id=gate_id)
    aggregate = _validate_maintained_aggregate(
        aggregate_path, bank_state=bank_state,
        uncontended_timing=state["plan"]["uncontended_timing"],
    )
    output = aggregate_path.parent / "governance-aggregate.json"
    _write_sealed_once(output, {
        "schema": NORMALIZED_AGGREGATE_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": state["plan"]["attempt"],
        "gate_id": gate_id,
        "candidate_source_sha256": state["plan"]["candidate"]["source"]["sha256"],
        "candidate_runtime_sha256": state["plan"]["candidate"]["runtime"]["sha256"],
        "bank_sha256": bank_state["receipt"]["protected_bank"]["sha256"],
        "workers": WORKERS, "threads_per_worker": THREADS_PER_WORKER,
        "maintained_aggregate": _reference(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        ),
        "summary": dict(aggregate["summary"]),
        "verdict": dict(aggregate["verdict"]),
        "completed_at_utc": aggregate["completed_at_utc"],
    })
    return output


def _deep_evidence(
    state: Mapping[str, Any], *, dual_reference: pathlib.Path,
    gate_id: str, aggregate_path: pathlib.Path,
    result_validator: ResultValidator,
) -> pathlib.Path:
    prepared_reference, prepared = _load_prepared(state)
    if prepared_reference.resolve() != dual_reference.resolve():
        raise DualFinalError("prepared receipt references another dual final")
    dual_state = challenger.validate_dual_final(
        dual_reference,
        plan_path=pathlib.Path(state["plan"]["campaign_plan"]["path"]),
    )
    if {
        record["sha256"] for record in prepared["fingerprint_exclusions"].values()
    } != {
        record["sha256"] for record in dual_state["plan"].get(
            "dynamic_exclusions", []
        )
    }:
        raise DualFinalError("runner/governance sanitized exclusions differ")
    gate = next(item for item in dual_state["plan"]["gates"] if item["gate_id"] == gate_id)
    bank_state = validate_bank_receipt(state, gate_id=gate_id)
    if gate["bank"] != bank_state["receipt"]["protected_bank"]:
        raise DualFinalError("governance gate bank differs from execution bank")
    missing = _audit_shards(
        state, bank_state, gate_id=gate_id, result_validator=result_validator
    )
    if missing:
        raise DualFinalError(f"{gate_id} evidence has missing shards")
    maintained = _validate_maintained_aggregate(
        aggregate_path, bank_state=bank_state,
        uncontended_timing=state["plan"]["uncontended_timing"],
    )
    normalized_path = _normalized_aggregate(
        state, gate_id=gate_id, aggregate_path=aggregate_path
    )
    ledger = _gate_root(state["plan"], gate_id) / "ledger"
    (
        consumption_path, consumption, primitive_consumption_path,
        primitive_consumption,
    ) = _validate_consumption(
        state, bank_state, gate_id=gate_id
    )
    if (
        consumption.get("gate_id") != gate_id
        or consumption.get("workers") != WORKERS
        or consumption.get("threads_per_worker") != THREADS_PER_WORKER
        or consumption.get("one_launch_only") is not True
        or consumption.get("retry_authorized") is not False
        or primitive_consumption.get("binding_sha256")
        != qualification.sha256_file(bank_state["binding_path"])
        or primitive_consumption.get("bank") != bank_state["binding"]["bank"]
    ):
        raise DualFinalError(f"{gate_id} consumption evidence changed")
    claims = []
    receipts = []
    raw_evidence = []
    raw_results = []
    for index in range(SHARDS):
        claim_path = ledger / "claims" / f"shard-{index:03d}.json"
        receipt_path = ledger / "receipts" / f"shard-{index:03d}.json"
        receipt = qualification.validate_shard_receipt(
            receipt_path, binding_path=bank_state["binding_path"], index=index
        )
        evidence_path = _verify_reference(
            receipt["evidence"], RAW_EVIDENCE_SCHEMA,
            f"{gate_id} shard {index} evidence",
        )
        evidence = qualification.load_sealed(evidence_path, RAW_EVIDENCE_SCHEMA)
        raw_path = _verify_record(
            evidence["raw_gate_result"], f"{gate_id} shard {index} raw result"
        )
        claims.append(_reference(claim_path, qualification.SHARD_CLAIM_SCHEMA))
        receipts.append(_reference(receipt_path, qualification.SHARD_RECEIPT_SCHEMA))
        raw_evidence.append(_reference(evidence_path, RAW_EVIDENCE_SCHEMA))
        raw_results.append(_record(raw_path))
    output = ledger / "governance-gate-evidence.json"
    _write_sealed_once(output, {
        # The outer schema is intentionally the governor's accepted input.  The
        # bridge_schema field selects this stricter recursive contract.
        "schema": challenger.FINAL_GATE_EVIDENCE_SCHEMA,
        "bridge_schema": DEEP_GATE_EVIDENCE_SCHEMA,
        "namespace": NAMESPACE, "campaign_id": CAMPAIGN_ID,
        "attempt": state["plan"]["attempt"], "gate_id": gate_id,
        "status": "complete",
        "dual_final_plan": _reference(
            dual_state["path"], challenger.DUAL_FINAL_SCHEMA
        ),
        "execution_prepared": _reference(
            pathlib.Path(state["plan"]["root"]) / "prepared.json",
            PREPARED_SCHEMA,
        ),
        "all_final_bank_fingerprint_exclusions": dict(
            prepared["fingerprint_exclusions"]
        ),
        "execution_plan": _reference(pathlib.Path(state["path"]), PLAN_SCHEMA),
        "candidate": {
            "runtime_sha256": state["plan"]["candidate"]["runtime"]["sha256"],
            "source_sha256": state["plan"]["candidate"]["source"]["sha256"],
        },
        "candidate_commit": state["plan"]["candidate_commit"],
        "runtime_identity": dict(state["plan"]["runtime_identity"]),
        "bank": {
            "sha256": gate["bank"]["sha256"], "bytes": gate["bank"]["bytes"]
        },
        "bank_receipt": _reference(bank_state["path"], BANK_RECEIPT_SCHEMA),
        "fingerprint_exclusion": _reference(
            bank_state["fingerprint_exclusion_path"],
            FINGERPRINT_EXCLUSION_SCHEMA,
        ),
        "gate_binding": _reference(
            bank_state["binding_path"], qualification.GATE_BINDING_SCHEMA
        ),
        "pairs": 500, "games": 1_000, "workers": WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "shards": SHARDS, "pairs_per_shard": PAIRS_PER_SHARD,
        "actual_clock": True,
        "configuration": dict(state["plan"]["configuration"]),
        "all_shards_complete": True,
        "consumption": _reference(consumption_path, CONSUMPTION_SCHEMA),
        "primitive_consumption": _record(primitive_consumption_path),
        "claims": claims, "receipts": receipts,
        "raw_evidence": raw_evidence, "raw_gate_results": raw_results,
        "maintained_aggregate": _reference(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        ),
        # Current governance consumes this normalized aggregate.  Recursive
        # validation proves its summary came from the maintained aggregate.
        "aggregate": _record(normalized_path),
        "summary": dict(maintained["summary"]),
        "verdict": dict(maintained["verdict"]),
        "deployment_preflight": state["plan"]["deployment_preflight"],
        "preflight_kind": state["plan"]["preflight_kind"],
        "preflight_schema": state["plan"]["preflight_schema"],
        "compile_binding": dict(state["plan"]["compile_binding"]),
        "uncontended_timing": dict(state["plan"]["uncontended_timing"]),
        "ci": state["plan"]["ci"],
        "deep_validation": {
            "maintained_aggregate_schema": qualification.FINAL_AGGREGATE_SCHEMA,
            "claim_count": SHARDS, "receipt_count": SHARDS,
            "raw_evidence_count": SHARDS, "raw_result_count": SHARDS,
            "candidate_source_and_runtime_bound": True,
            "gate_binary_and_compiler_bound": True,
            "actual_clock_configuration_bound": True,
            "preflight_timing_and_ci_bound": True,
            "no_retry_or_reuse": True,
        },
    })
    validate_gate_evidence(
        output, state=state, dual_reference=dual_reference,
        result_validator=result_validator,
    )
    return output


def validate_gate_evidence(
    path: pathlib.Path, *, state: Mapping[str, Any],
    dual_reference: pathlib.Path, result_validator: ResultValidator = gate_support.validate_result,
) -> Record:
    """Recursively validate an evidence adapter accepted by governance."""

    value = qualification.load_sealed(
        path, challenger.FINAL_GATE_EVIDENCE_SCHEMA
    )
    gate_id = value.get("gate_id")
    if gate_id not in GATE_IDS:
        raise DualFinalError("deep gate evidence has unknown gate")
    prepared_reference, prepared = _load_prepared(state)
    if prepared_reference.resolve() != dual_reference.resolve():
        raise DualFinalError("deep evidence uses another prepared dual final")
    dual_state = challenger.validate_dual_final(
        dual_reference,
        plan_path=pathlib.Path(state["plan"]["campaign_plan"]["path"]),
    )
    if {
        record["sha256"] for record in prepared["fingerprint_exclusions"].values()
    } != {
        record["sha256"] for record in dual_state["plan"].get(
            "dynamic_exclusions", []
        )
    }:
        raise DualFinalError("deep evidence sanitized exclusions differ from governance")
    gate = next(item for item in dual_state["plan"]["gates"] if item["gate_id"] == gate_id)
    bank_state = validate_bank_receipt(state, gate_id=str(gate_id))
    maintained_path = _verify_reference(
        value.get("maintained_aggregate"), qualification.FINAL_AGGREGATE_SCHEMA,
        "maintained final aggregate",
    )
    maintained = _validate_maintained_aggregate(
        maintained_path, bank_state=bank_state,
        uncontended_timing=state["plan"]["uncontended_timing"],
    )
    normalized_path = _verify_record(
        value.get("aggregate"), "normalized governance aggregate"
    )
    normalized = qualification.load_sealed(
        normalized_path, NORMALIZED_AGGREGATE_SCHEMA
    )
    missing = _audit_shards(
        state, bank_state, gate_id=str(gate_id),
        result_validator=result_validator,
    )
    ledger = _gate_root(state["plan"], str(gate_id)) / "ledger"
    expected_rosters = {
        "claims": [
            _reference(
                ledger / "claims" / f"shard-{index:03d}.json",
                qualification.SHARD_CLAIM_SCHEMA,
            ) for index in range(SHARDS)
        ],
        "receipts": [
            _reference(
                ledger / "receipts" / f"shard-{index:03d}.json",
                qualification.SHARD_RECEIPT_SCHEMA,
            ) for index in range(SHARDS)
        ],
        "raw_evidence": [
            _reference(
                ledger / "raw-evidence" / f"shard-{index:03d}.json",
                RAW_EVIDENCE_SCHEMA,
            ) for index in range(SHARDS)
        ],
        "raw_gate_results": [
            _record(ledger / "raw" / f"shard-{index:03d}.json")
            for index in range(SHARDS)
        ],
    }
    consumption_path, consumption, primitive_path, primitive = (
        _validate_consumption(state, bank_state, gate_id=str(gate_id))
    )
    if (
        value.get("consumption")
        != _reference(consumption_path, CONSUMPTION_SCHEMA)
        or value.get("primitive_consumption") != _record(primitive_path)
    ):
        raise DualFinalError("deep evidence consumption references changed")
    deep = value.get("deep_validation")
    expected_deep = {
        "maintained_aggregate_schema": qualification.FINAL_AGGREGATE_SCHEMA,
        "claim_count": SHARDS, "receipt_count": SHARDS,
        "raw_evidence_count": SHARDS, "raw_result_count": SHARDS,
        "candidate_source_and_runtime_bound": True,
        "gate_binary_and_compiler_bound": True,
        "actual_clock_configuration_bound": True,
        "preflight_timing_and_ci_bound": True,
        "no_retry_or_reuse": True,
    }
    expected_fields = {
        "schema", "bridge_schema", "namespace", "campaign_id", "attempt",
        "gate_id", "status", "dual_final_plan", "execution_prepared",
        "all_final_bank_fingerprint_exclusions", "execution_plan", "candidate",
        "candidate_commit", "runtime_identity", "bank", "bank_receipt",
        "fingerprint_exclusion", "gate_binding", "pairs", "games", "workers",
        "threads_per_worker", "shards", "pairs_per_shard", "actual_clock",
        "configuration", "all_shards_complete", "consumption",
        "primitive_consumption", "claims", "receipts", "raw_evidence",
        "raw_gate_results", "maintained_aggregate", "aggregate", "summary",
        "verdict", "deployment_preflight", "compile_binding",
        "preflight_kind", "preflight_schema", "uncontended_timing", "ci",
        "deep_validation", "body_sha256",
    }
    if (
        missing
        or set(value) != expected_fields
        or value.get("bridge_schema") != DEEP_GATE_EVIDENCE_SCHEMA
        or value.get("namespace") != NAMESPACE
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != state["plan"]["attempt"]
        or value.get("status") != "complete"
        or value.get("dual_final_plan")
        != _reference(dual_state["path"], challenger.DUAL_FINAL_SCHEMA)
        or value.get("execution_prepared")
        != _reference(
            pathlib.Path(state["plan"]["root"]) / "prepared.json",
            PREPARED_SCHEMA,
        )
        or value.get("all_final_bank_fingerprint_exclusions")
        != prepared["fingerprint_exclusions"]
        or value.get("execution_plan")
        != _reference(pathlib.Path(state["path"]), PLAN_SCHEMA)
        or value.get("candidate") != {
            "runtime_sha256": state["plan"]["candidate"]["runtime"]["sha256"],
            "source_sha256": state["plan"]["candidate"]["source"]["sha256"],
        }
        or value.get("candidate_commit") != state["plan"]["candidate_commit"]
        or value.get("runtime_identity") != state["plan"]["runtime_identity"]
        or value.get("bank") != {
            "sha256": gate["bank"]["sha256"], "bytes": gate["bank"]["bytes"]
        }
        or value.get("bank_receipt")
        != _reference(bank_state["path"], BANK_RECEIPT_SCHEMA)
        or value.get("fingerprint_exclusion")
        != _reference(
            bank_state["fingerprint_exclusion_path"],
            FINGERPRINT_EXCLUSION_SCHEMA,
        )
        or value.get("gate_binding")
        != _reference(bank_state["binding_path"], qualification.GATE_BINDING_SCHEMA)
        or value.get("pairs") != 500 or value.get("games") != 1_000
        or value.get("workers") != WORKERS
        or value.get("threads_per_worker") != THREADS_PER_WORKER
        or value.get("shards") != SHARDS
        or value.get("pairs_per_shard") != PAIRS_PER_SHARD
        or value.get("actual_clock") is not True
        or value.get("configuration") != state["plan"]["configuration"]
        or value.get("all_shards_complete") is not True
        or any(value.get(name) != roster for name, roster in expected_rosters.items())
        or value.get("summary") != maintained["summary"]
        or value.get("verdict") != maintained["verdict"]
        or maintained_path.resolve() != ledger.resolve() / "aggregate.json"
        or normalized_path.resolve()
        != ledger.resolve() / "governance-aggregate.json"
        or set(normalized) != {
            "schema", "namespace", "campaign_id", "attempt", "gate_id",
            "candidate_source_sha256", "candidate_runtime_sha256",
            "bank_sha256", "workers", "threads_per_worker",
            "maintained_aggregate", "summary", "verdict",
            "completed_at_utc", "body_sha256",
        }
        or normalized.get("maintained_aggregate")
        != _reference(maintained_path, qualification.FINAL_AGGREGATE_SCHEMA)
        or normalized.get("namespace") != NAMESPACE
        or normalized.get("campaign_id") != CAMPAIGN_ID
        or normalized.get("attempt") != state["plan"]["attempt"]
        or normalized.get("gate_id") != gate_id
        or normalized.get("summary") != maintained["summary"]
        or normalized.get("verdict") != maintained["verdict"]
        or normalized.get("completed_at_utc") != maintained["completed_at_utc"]
        or normalized.get("candidate_source_sha256")
        != state["plan"]["candidate"]["source"]["sha256"]
        or normalized.get("candidate_runtime_sha256")
        != state["plan"]["candidate"]["runtime"]["sha256"]
        or normalized.get("bank_sha256") != gate["bank"]["sha256"]
        or normalized.get("workers") != WORKERS
        or normalized.get("threads_per_worker") != THREADS_PER_WORKER
        or value.get("deployment_preflight")
        != state["plan"]["deployment_preflight"]
        or value.get("preflight_kind") != state["plan"]["preflight_kind"]
        or value.get("preflight_schema") != state["plan"]["preflight_schema"]
        or value.get("compile_binding") != state["plan"]["compile_binding"]
        or value.get("uncontended_timing")
        != state["plan"]["uncontended_timing"]
        or value.get("ci") != state["plan"]["ci"]
        or deep != expected_deep
        or consumption.get("retry_authorized") is not False
        or consumption_path.resolve() != ledger.resolve() / "consumption.json"
        or primitive_path.resolve() != ledger.resolve() / "bank-consumed.json"
        or primitive.get("binding_sha256")
        != qualification.sha256_file(bank_state["binding_path"])
    ):
        raise DualFinalError("deep final-gate evidence closure changed")
    # Re-run the expensive source/preflight/CI/exclusion validators last; this
    # guarantees callers cannot use a copied evidence closure after inputs move.
    validate_execution_plan(pathlib.Path(state["path"]))
    return value


def validate_governance_evidence(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    dual_reference: pathlib.Path,
    result_validator: ResultValidator = gate_support.validate_result,
) -> Record:
    """Convenience entrypoint for the challenger governor's lazy import.

    The execution-plan reference is read from the evidence but is never trusted
    until its complete authorization/preflight/CI/exclusion closure validates.
    """

    header = qualification.load_sealed(
        path, challenger.FINAL_GATE_EVIDENCE_SCHEMA
    )
    execution_path = _verify_reference(
        header.get("execution_plan"), PLAN_SCHEMA, "deep execution plan"
    )
    state = validate_execution_plan(execution_path)
    if state["campaign_plan"].resolve() != campaign_plan_path.resolve():
        raise DualFinalError("deep evidence belongs to another campaign plan")
    return validate_gate_evidence(
        path, state=state, dual_reference=dual_reference,
        result_validator=result_validator,
    )


def run_dual_final(
    plan_path: pathlib.Path, *, launched_at_utc: str,
    runner: GateRunner = run_gate_process,
    result_validator: ResultValidator = gate_support.validate_result,
    clock: Callable[[], str] = utc_now,
    state_validator: Callable[[pathlib.Path], Mapping[str, Any]] = validate_execution_plan,
    result_recorder: Callable[..., pathlib.Path] = challenger.record_final_result,
    dual_completer: Callable[..., pathlib.Path] = challenger.complete_dual_final,
    executor_factory: Callable[..., Any] = concurrent.futures.ThreadPoolExecutor,
) -> Record:
    """Run A, and only after an exact pass run B; safely resume receipts."""

    state = dict(state_validator(plan_path))
    plan = state["plan"]
    dual_reference, _prepared = _load_prepared(state)
    challenger.validate_dual_final(
        dual_reference,
        plan_path=pathlib.Path(plan["campaign_plan"]["path"]),
    )
    results: dict[str, pathlib.Path] = {}
    evidence: dict[str, pathlib.Path] = {}
    for gate_id in GATE_IDS:
        if gate_id == "gate-b":
            first_aggregate = qualification.load_sealed(
                _gate_root(plan, "gate-a") / "ledger" / "aggregate.json",
                qualification.FINAL_AGGREGATE_SCHEMA,
            )
            if first_aggregate.get("verdict", {}).get("passed") is not True:
                break
        aggregate_path = _execute_gate(
            state, gate_id=gate_id, launched_at_utc=launched_at_utc,
            runner=runner, result_validator=result_validator, clock=clock,
            executor_factory=executor_factory,
        )
        evidence_path = _deep_evidence(
            state, dual_reference=dual_reference, gate_id=gate_id,
            aggregate_path=aggregate_path, result_validator=result_validator,
        )
        result = result_recorder(
            dual_reference,
            plan_path=pathlib.Path(plan["campaign_plan"]["path"]),
            gate_id=gate_id, evidence_path=evidence_path,
            completed_at_utc=qualification.load_sealed(
                aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
            )["completed_at_utc"],
        )
        results[gate_id] = result
        evidence[gate_id] = evidence_path
        aggregate = qualification.load_sealed(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        )
        if aggregate["verdict"]["passed"] is not True:
            break
    qualification_path: pathlib.Path | None = None
    if set(results) == set(GATE_IDS) and all(
        qualification.load_sealed(
            _gate_root(plan, gate_id) / "ledger" / "aggregate.json",
            qualification.FINAL_AGGREGATE_SCHEMA,
        )["verdict"]["passed"] is True
        for gate_id in GATE_IDS
    ):
        expected_qualification = dual_reference.parent / "dual-qualified.json"
        completion_time = clock()
        if expected_qualification.exists():
            completion_time = qualification.load_sealed(
                expected_qualification, challenger.DUAL_QUALIFICATION_SCHEMA
            )["completed_at_utc"]
        qualification_path = dual_completer(
            dual_reference,
            plan_path=pathlib.Path(plan["campaign_plan"]["path"]),
            result_a=results["gate-a"], result_b=results["gate-b"],
            completed_at_utc=completion_time,
        )
    output = pathlib.Path(plan["root"]) / "execution-receipt.json"
    body = {
        "schema": RUN_RECEIPT_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "attempt": plan["attempt"],
        "status": (
            "two-gates-passed" if qualification_path is not None
            else "gate-a-failed" if "gate-b" not in results
            else "gate-b-failed"
        ),
        "execution_plan": _reference(plan_path, PLAN_SCHEMA),
        "dual_final_reference": _reference(
            dual_reference, challenger.DUAL_FINAL_REFERENCE_SCHEMA
        ),
        "candidate": {
            "runtime_sha256": plan["candidate"]["runtime"]["sha256"],
            "source_sha256": plan["candidate"]["source"]["sha256"],
        },
        "gate_results": {
            gate_id: _reference(result, challenger.FINAL_RESULT_SCHEMA)
            for gate_id, result in results.items()
        },
        "gate_evidence": {
            gate_id: _reference(item, challenger.FINAL_GATE_EVIDENCE_SCHEMA)
            for gate_id, item in evidence.items()
        },
        "dual_qualification": (
            None if qualification_path is None else _reference(
                qualification_path, challenger.DUAL_QUALIFICATION_SCHEMA
            )
        ),
        "gate_b_launched_only_after_gate_a_pass": True,
        "workers_per_gate": WORKERS, "shards_per_gate": SHARDS,
        "upload_authorized": False,
    }
    if output.exists():
        existing = qualification.load_sealed(output, RUN_RECEIPT_SCHEMA)
        if existing != qualification.seal(body):
            raise DualFinalError("execution receipt changed on resume")
    else:
        _write_sealed_once(output, body)
    return {
        "receipt": output, "status": body["status"],
        "results": results, "evidence": evidence,
        "qualification": qualification_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--authorization", type=pathlib.Path, required=True)
    prepare.add_argument("--campaign-plan", type=pathlib.Path, required=True)
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    authority = prepare.add_mutually_exclusive_group(required=True)
    authority.add_argument("--deployment-preflight", type=pathlib.Path)
    authority.add_argument("--release-evidence", type=pathlib.Path)
    prepare.add_argument("--ci", type=pathlib.Path, required=True)
    prepare.add_argument("--rank4-source", type=pathlib.Path, required=True)
    prepare.add_argument("--exclusion", type=pathlib.Path, action="append", default=[])
    prepare.add_argument("--created-at-utc", default=utc_now())
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--plan", type=pathlib.Path, required=True)
    materialize.add_argument("--claimed-at-utc", default=utc_now())
    run = commands.add_parser("run")
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--launched-at-utc", default=utc_now())
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            path = prepare_execution(
                authorization_path=arguments.authorization,
                campaign_plan_path=arguments.campaign_plan,
                output_root=arguments.output_root,
                deployment_preflight_path=arguments.deployment_preflight,
                release_evidence_path=arguments.release_evidence,
                ci_path=arguments.ci, rank4_source=arguments.rank4_source,
                exclusion_paths=arguments.exclusion,
                created_at_utc=arguments.created_at_utc,
            )
            result: Any = {"plan": str(path), "sha256": qualification.sha256_file(path)}
        elif arguments.command == "materialize":
            path = materialize_banks(
                arguments.plan, claimed_at_utc=arguments.claimed_at_utc
            )
            result = {"dual_final_reference": str(path), "sha256": qualification.sha256_file(path)}
        elif arguments.command == "run":
            completed = run_dual_final(
                arguments.plan, launched_at_utc=arguments.launched_at_utc
            )
            result = {
                "receipt": str(completed["receipt"]),
                "status": completed["status"],
            }
        else:
            state = validate_execution_plan(arguments.plan)
            result = {
                "plan": str(arguments.plan.resolve()),
                "attempt": state["plan"]["attempt"],
                "candidate": state["plan"]["candidate"],
            }
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        DualFinalError, qualification.QualificationError, OSError,
        KeyError, TypeError, ValueError,
    ) as error:
        print(f"dual-final execution failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
