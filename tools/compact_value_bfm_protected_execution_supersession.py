#!/usr/bin/env python3
"""Seal a hash-only replay contract for a protected infrastructure failure.

This adapter is deliberately narrow.  It recognizes the post-Gate-A contract
mismatch where the dual runner emitted a recursively validated two-field
``dual_final_plan`` reference and the outer challenger expected the legacy
five-field sealed record.  It never carries a score, transcript, position, or
qualification into the successor campaign; only the two sanitized fingerprint
sets are exported as a protected exclusion.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


from tools import compact_value_bfm_qualification as qualification
from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import compact_value_bfm_rank4_teacher_dual_final as dual


SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "protected-execution-supersession.v1"
)
STATUS = "same-candidate-fresh-gates-after-governance-adapter-failure"


class SupersessionError(RuntimeError):
    pass


def _sealed(path: pathlib.Path, schema: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SupersessionError(f"{label} is absent or redirected")
    path = path.resolve()
    try:
        value = qualification.load_sealed(path, schema)
    except Exception as error:
        raise SupersessionError(f"{label} is not a valid sealed artifact") from error
    raw = path.read_bytes()
    return value, {
        "path": str(path), "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        "schema": schema, "body_sha256": value["body_sha256"],
    }


def _thin(path: pathlib.Path, schema: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, sealed = _sealed(path, schema, label)
    return value, {"path": sealed["path"], "sha256": sealed["sha256"]}


def _verify_thin(value: Any, schema: str, label: str) -> tuple[pathlib.Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise SupersessionError(f"{label} reference shape changed")
    path = pathlib.Path(str(value.get("path", "")))
    loaded, expected = _thin(path, schema, label)
    if dict(value) != expected:
        raise SupersessionError(f"{label} reference changed")
    return path.resolve(), loaded


def _verify_regular(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise SupersessionError(f"{label} record shape changed")
    path = pathlib.Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file():
        raise SupersessionError(f"{label} is absent or redirected")
    path = path.resolve()
    raw = path.read_bytes()
    expected = {
        "path": str(path), "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
    }
    if dict(value) != expected:
        raise SupersessionError(f"{label} bytes changed")
    return path


def _roster(
    directory: pathlib.Path, *, schema: str | None, regular: bool, label: str,
) -> tuple[list[dict[str, Any]], str]:
    if directory.is_symlink() or not directory.is_dir():
        raise SupersessionError(f"{label} directory is absent or redirected")
    paths = sorted(directory.iterdir())
    expected = [directory / f"shard-{index:03d}.json" for index in range(100)]
    if paths != expected:
        raise SupersessionError(f"{label} roster is not exactly 100 contiguous shards")
    records: list[dict[str, Any]] = []
    for path in paths:
        if regular:
            if path.is_symlink() or not path.is_file():
                raise SupersessionError(f"{label} contains an irregular shard")
            raw = path.read_bytes()
            records.append({
                "path": str(path.resolve()), "bytes": len(raw),
                "sha256": qualification.sha256_bytes(raw),
            })
        else:
            assert schema is not None
            _value, record = _thin(path, schema, f"{label} {path.name}")
            records.append(record)
    digest_rows = [
        {"name": pathlib.Path(record["path"]).name, **{
            key: record[key] for key in ("bytes", "sha256") if key in record
        }}
        for record in records
    ]
    return records, qualification.sha256_bytes(
        qualification.canonical_json_bytes(digest_rows)
    )


def _fingerprint_exclusion(
    record: Any, *, gate_id: str, candidate: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, value = _verify_thin(
        record, dual.FINGERPRINT_EXCLUSION_SCHEMA,
        f"{gate_id} fingerprint exclusion",
    )
    values = value.get("fingerprints")
    origin = value.get("origin")
    if (
        value.get("gate_id") != gate_id
        or value.get("classification")
        != "protected-final-canonical-fingerprints"
        or value.get("fingerprint_count") != 500
        or not isinstance(values, list) or len(values) != 500
        or len(set(values)) != 500
        or any(
            qualification.SHA256_RE.fullmatch(str(item)) is None
            for item in values
        )
        or not isinstance(origin, Mapping)
        or origin.get("candidate_source_sha256") != candidate["source_sha256"]
        or origin.get("candidate_runtime_sha256") != candidate["runtime_sha256"]
        or value.get("contains_transcripts") is not False
        or value.get("contains_metrics") is not False
        or value.get("contains_labels") is not False
        or value.get("training_eligible") is not False
        or value.get("required_for_all_later_development_and_protected_banks")
        is not True
    ):
        raise SupersessionError(f"{gate_id} fingerprint exclusion policy changed")
    _loaded, sealed = _sealed(
        path, dual.FINGERPRINT_EXCLUSION_SCHEMA,
        f"{gate_id} fingerprint exclusion",
    )
    return sealed, {"fingerprints": [str(item) for item in values], "origin": dict(origin)}


def _derive(execution_plan_path: pathlib.Path, *, created_at_utc: str) -> dict[str, Any]:
    qualification._utc(created_at_utc, "supersession timestamp")
    execution, execution_record = _sealed(
        execution_plan_path, dual.PLAN_SCHEMA, "superseded execution plan"
    )
    root = pathlib.Path(str(execution.get("root", ""))).resolve()
    if (
        execution_plan_path.resolve() != root / "execution-plan.json"
        or execution.get("production") is not True
        or execution.get("evidence_mode") != "production-default-callables"
        or execution.get("status")
        != "dual-final-execution-planned-banks-unclaimed"
    ):
        raise SupersessionError("superseded execution identity changed")
    campaign_path, _campaign = _verify_thin(
        execution.get("campaign_plan"), challenger.PLAN_SCHEMA,
        "superseded campaign plan",
    )
    authorization_path, authorization = _verify_thin(
        execution.get("authorization"), challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
        "superseded authorization",
    )
    candidate_record = execution.get("candidate")
    if not isinstance(candidate_record, Mapping):
        raise SupersessionError("superseded candidate is absent")
    candidate = {
        "runtime_sha256": str(candidate_record.get("runtime", {}).get("sha256", "")),
        "source_sha256": str(candidate_record.get("source", {}).get("sha256", "")),
    }
    if (
        any(qualification.SHA256_RE.fullmatch(value) is None for value in candidate.values())
        or authorization.get("candidate", {}).get("runtime", {}).get("sha256")
        != candidate["runtime_sha256"]
        or authorization.get("candidate", {}).get("source", {}).get("sha256")
        != candidate["source_sha256"]
    ):
        raise SupersessionError("superseded candidate changed after authorization")

    prepared_path = root / "prepared.json"
    prepared, prepared_record = _sealed(
        prepared_path, dual.PREPARED_SCHEMA, "superseded prepared receipt"
    )
    if (
        prepared.get("execution_plan")
        != {"path": execution_record["path"], "sha256": execution_record["sha256"]}
        or prepared.get("candidate_unchanged") is not True
        or prepared.get("independent_banks") is not True
        or prepared.get("gate_b_excludes_gate_a") is not True
        or prepared.get("games_launched") != 0
    ):
        raise SupersessionError("superseded prepared policy changed")
    dual_reference_path, dual_reference = _verify_thin(
        prepared.get("dual_final_reference"), challenger.DUAL_FINAL_REFERENCE_SCHEMA,
        "superseded dual-final reference",
    )
    dual_plan_record = dual_reference.get("dual_final_plan")
    if not isinstance(dual_plan_record, Mapping):
        raise SupersessionError("superseded dual-final plan reference is absent")
    dual_plan_path = pathlib.Path(str(dual_plan_record.get("path", "")))
    dual_plan, expected_dual_plan_record = _sealed(
        dual_plan_path, challenger.DUAL_FINAL_SCHEMA, "superseded dual-final plan"
    )
    if dict(dual_plan_record) != expected_dual_plan_record:
        raise SupersessionError("superseded dual-final sealed record changed")
    _authorization_value, authorization_record = _sealed(
        authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
        "superseded authorization",
    )
    if (
        dual_plan.get("candidate") != candidate_record
        or dual_plan.get("authorization") != authorization_record
    ):
        raise SupersessionError("superseded dual-final candidate/authorization changed")
    gates = {
        str(item.get("gate_id")): item
        for item in dual_plan.get("gates", []) if isinstance(item, Mapping)
    }
    if set(gates) != {"gate-a", "gate-b"}:
        raise SupersessionError("superseded gate roster changed")

    exclusions: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    origins: dict[str, Any] = {}
    for gate_id in ("gate-a", "gate-b"):
        record = prepared.get("fingerprint_exclusions", {}).get(gate_id)
        sealed, checked = _fingerprint_exclusion(
            record, gate_id=gate_id, candidate=candidate
        )
        if checked["origin"].get("protected_bank_sha256") != gates[gate_id].get(
            "bank", {}
        ).get("sha256"):
            raise SupersessionError(f"{gate_id} exclusion lost its protected bank")
        if fingerprints & set(checked["fingerprints"]):
            raise SupersessionError("superseded protected banks overlap by fingerprint")
        fingerprints.update(checked["fingerprints"])
        exclusions.append(sealed)
        origins[gate_id] = checked["origin"]
    if len(fingerprints) != 1_000:
        raise SupersessionError("supersession does not exclude exactly 1,000 fingerprints")

    gate_a_root = root / "gates/gate-a"
    ledger_a = gate_a_root / "ledger"
    evidence_path = ledger_a / "governance-gate-evidence.json"
    evidence, evidence_record = _sealed(
        evidence_path, challenger.FINAL_GATE_EVIDENCE_SCHEMA,
        "superseded Gate-A deep evidence",
    )
    thin_dual = {"path": str(dual_plan_path.resolve()), "sha256": qualification.sha256_file(dual_plan_path)}
    if (
        evidence.get("bridge_schema") != dual.DEEP_GATE_EVIDENCE_SCHEMA
        or evidence.get("gate_id") != "gate-a"
        or evidence.get("status") != "complete"
        or evidence.get("candidate") != candidate
        or evidence.get("bank") != {
            "sha256": gates["gate-a"]["bank"]["sha256"],
            "bytes": gates["gate-a"]["bank"]["bytes"],
        }
        or evidence.get("pairs") != 500 or evidence.get("games") != 1_000
        or evidence.get("workers") != 4
        or evidence.get("threads_per_worker") != 1
        or evidence.get("shards") != 100
        or evidence.get("all_shards_complete") is not True
        or evidence.get("dual_final_plan") != thin_dual
        or evidence.get("dual_final_plan") == expected_dual_plan_record
    ):
        raise SupersessionError("failure is not the exact thin-reference bridge mismatch")
    aggregate_path = _verify_regular(
        evidence.get("aggregate"), "superseded normalized aggregate"
    )
    aggregate = qualification.load_sealed(
        aggregate_path, dual.NORMALIZED_AGGREGATE_SCHEMA
    )
    if (
        aggregate.get("candidate_source_sha256") != candidate["source_sha256"]
        or aggregate.get("candidate_runtime_sha256") != candidate["runtime_sha256"]
        or aggregate.get("bank_sha256") != gates["gate-a"]["bank"]["sha256"]
        or aggregate.get("workers") != 4
        or aggregate.get("threads_per_worker") != 1
        or aggregate.get("summary") != evidence.get("summary")
    ):
        raise SupersessionError("superseded Gate-A aggregate binding changed")

    roster_specs = {
        "claims": (qualification.SHARD_CLAIM_SCHEMA, False),
        "receipts": (qualification.SHARD_RECEIPT_SCHEMA, False),
        "raw_evidence": (dual.RAW_EVIDENCE_SCHEMA, False),
        "raw_results": (None, True),
    }
    roster_fields = {
        "claims": "claims", "receipts": "receipts",
        "raw_evidence": "raw_evidence", "raw_results": "raw_gate_results",
    }
    directories = {
        "claims": ledger_a / "claims", "receipts": ledger_a / "receipts",
        "raw_evidence": ledger_a / "raw-evidence", "raw_results": ledger_a / "raw",
    }
    roster_summary: dict[str, Any] = {}
    for name, (schema, regular) in roster_specs.items():
        records, digest = _roster(
            directories[name], schema=schema, regular=regular,
            label=f"Gate-A {name}",
        )
        if evidence.get(roster_fields[name]) != records:
            raise SupersessionError(f"Gate-A {name} evidence roster changed")
        roster_summary[name] = {"count": 100, "roster_sha256": digest}

    if pathlib.Path(str(gates["gate-a"].get("result_path", ""))).exists():
        raise SupersessionError("Gate-A governance result already exists")
    if (dual_reference_path.parent / "dual-qualified.json").exists():
        raise SupersessionError("superseded execution already has a qualification")
    if (root / "execution-receipt.json").exists():
        raise SupersessionError("superseded execution already has a run receipt")
    ledger_b = root / "gates/gate-b/ledger"
    if ledger_b.exists() and any(ledger_b.rglob("*")):
        raise SupersessionError("Gate B was consumed or partially launched")

    return {
        "schema": SCHEMA,
        "namespace": challenger.NAMESPACE,
        "campaign_id": challenger.CAMPAIGN_ID,
        "status": STATUS,
        "created_at_utc": str(created_at_utc),
        "source": {
            "campaign_plan": _sealed(
                campaign_path, challenger.PLAN_SCHEMA,
                "superseded campaign plan",
            )[1],
            "execution_plan": execution_record,
            "authorization": _sealed(
                authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
                "superseded authorization",
            )[1],
            "prepared": prepared_record,
            "dual_final_reference": _sealed(
                dual_reference_path, challenger.DUAL_FINAL_REFERENCE_SCHEMA,
                "superseded dual-final reference",
            )[1],
            "dual_final_plan": expected_dual_plan_record,
            "gate_a_deep_evidence": evidence_record,
            "gate_a_normalized_aggregate": _sealed(
                aggregate_path, dual.NORMALIZED_AGGREGATE_SCHEMA,
                "superseded normalized aggregate",
            )[1],
        },
        "candidate": candidate,
        "failure": {
            "class": "post-aggregate-governance-adapter-contract-mismatch",
            "stage": "after-gate-a-deep-validation-before-governance-result",
            "field": "dual_final_plan",
            "producer_shape": ["path", "sha256"],
            "legacy_consumer_shape": [
                "body_sha256", "bytes", "path", "schema", "sha256",
            ],
            "path_and_sha256_match": True,
            "bot_or_game_attributable": False,
        },
        "source_state": {
            "gate_a_all_shards_complete": True,
            "gate_a_rosters": roster_summary,
            "gate_a_result_recorded": False,
            "gate_b_consumed_or_launched": False,
            "dual_qualification_recorded": False,
            "execution_receipt_recorded": False,
        },
        "fingerprint_exclusions": exclusions,
        "fingerprint_origins": origins,
        "fingerprints": sorted(fingerprints),
        "fingerprint_count": 1_000,
        "contains_transcripts": False,
        "contains_metrics": False,
        "contains_labels": False,
        "training_eligible": False,
        "policy": {
            "same_exact_candidate_required": True,
            "candidate_change_authorized": False,
            "source_result_reuse_authorized": False,
            "source_qualification_reuse_authorized": False,
            "training_or_model_selection_use_authorized": False,
            "fresh_disjoint_banks_required": True,
            "all_source_bank_fingerprints_excluded": True,
        },
    }


def create_receipt(
    execution_plan_path: pathlib.Path, *, output: pathlib.Path,
    created_at_utc: str,
) -> pathlib.Path:
    output = output.resolve()
    body = _derive(execution_plan_path, created_at_utc=created_at_utc)
    source_campaign_root = pathlib.Path(
        body["source"]["campaign_plan"]["path"]
    ).parent.resolve()
    try:
        output.relative_to(source_campaign_root)
    except ValueError:
        pass
    else:
        raise SupersessionError("supersession receipt must be outside the source campaign")
    expected = qualification.seal(body)
    if output.exists():
        if qualification.load_sealed(output, SCHEMA) != expected:
            raise SupersessionError("existing supersession receipt changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        qualification.write_sealed(output, body)
    validate_receipt(output)
    return output


def validate_receipt(
    path: pathlib.Path, *, candidate_source: pathlib.Path | None = None,
    candidate_runtime: pathlib.Path | None = None,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, SCHEMA)
    execution_record = value.get("source", {}).get("execution_plan")
    if not isinstance(execution_record, Mapping):
        raise SupersessionError("supersession source execution is absent")
    execution_path = pathlib.Path(str(execution_record.get("path", "")))
    _loaded, expected_record = _sealed(
        execution_path, dual.PLAN_SCHEMA, "superseded execution plan"
    )
    if dict(execution_record) != expected_record:
        raise SupersessionError("supersession source execution changed")
    expected = qualification.seal(_derive(
        execution_path, created_at_utc=str(value.get("created_at_utc", ""))
    ))
    if value != expected or path.is_symlink():
        raise SupersessionError("supersession receipt changed")
    for supplied, key in (
        (candidate_source, "source_sha256"),
        (candidate_runtime, "runtime_sha256"),
    ):
        if supplied is not None and qualification.sha256_file(
            supplied
        ) != value["candidate"][key]:
            raise SupersessionError("successor candidate differs from superseded candidate")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--execution-plan", type=pathlib.Path, required=True)
    create.add_argument("--output", type=pathlib.Path, required=True)
    create.add_argument("--created-at-utc", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=pathlib.Path, required=True)
    validate.add_argument("--candidate-source", type=pathlib.Path)
    validate.add_argument("--candidate-runtime", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        path = create_receipt(
            arguments.execution_plan, output=arguments.output,
            created_at_utc=arguments.created_at_utc,
        )
        print(path)
    else:
        value = validate_receipt(
            arguments.receipt, candidate_source=arguments.candidate_source,
            candidate_runtime=arguments.candidate_runtime,
        )
        print({
            "status": value["status"],
            "fingerprint_count": value["fingerprint_count"],
            "candidate": value["candidate"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
