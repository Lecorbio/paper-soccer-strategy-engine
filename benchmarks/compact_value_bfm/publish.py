#!/usr/bin/env python3
"""Publish and verify a compact, transcript-free Value-BFM evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EVIDENCE_SCHEMA = "papersoccer.compact-value-bfm.public-evidence.v1"
NAMESPACE = "compact_value_bfm"
RANK4_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
ROLE_SCHEMAS = {
    "runtime": {"papersoccer.compact-value-bfm-runtime.v1"},
    "family": {
        "papersoccer.compact-value-bfm-selection.v1",
        "papersoccer.compact-value-bfm.family-exhausted.v1",
        "papersoccer.compact-value-bfm.iteration-event.v1",
    },
    "development": {
        "papersoccer.compact-value-bfm.development-input.v1",
        "papersoccer.compact-value-bfm.immutable-selection.v1",
    },
    "protected": {
        "papersoccer.compact-value-bfm.final-aggregate.v1",
        "papersoccer.compact-value-bfm.rank4-qualified-inputs.v1",
        "papersoccer.compact-value-bfm-protected-report.v1",
    },
    "preflight": {"papersoccer.compact-value-bfm.preflight-receipt.v1"},
    "upload": {"papersoccer.compact-value-bfm.upload-event.v1"},
    "live": {"papersoccer.compact-value-bfm.live-window.v1"},
}
REQUIRED_COMPLETE = (
    "runtime", "source", "family", "development", "protected",
    "preflight", "ci", "upload", "live",
)
RAW_SUFFIXES = {".npz", ".npy", ".tsv", ".csv", ".parquet", ".arrow"}
RAW_PATH_MARKERS = {
    "dataset", "datasets", "labels", "raw", "transcript", "transcripts",
    "openingbank", "replay", "replays", "positions", "shards",
}
FORBIDDEN_PUBLIC_KEYS = {
    "transcript", "transcripts", "move", "moves", "action", "actions",
    "frame", "frames", "opening", "openings", "row", "rows", "label",
    "labels", "weight", "weights", "payloadbase64", "gameids", "battle",
    "battles", "replay", "replays", "record", "records", "shard", "shards",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")
TRANSCRIPT_RE = re.compile(r"[0-7]+(?:/[0-7]+)+")


class PublicationError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def body_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_body_hash(value: Mapping[str, Any], label: str) -> None:
    claimed = value.get("body_sha256")
    if claimed is None:
        return
    if not isinstance(claimed, str) or SHA_RE.fullmatch(claimed) is None:
        raise PublicationError(f"{label} body SHA-256 is malformed")
    body = dict(value)
    del body["body_sha256"]
    if sha256_bytes(canonical_json_bytes(body)) != claimed:
        raise PublicationError(f"{label} body SHA-256 mismatch")


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def safe_input(path: pathlib.Path, role: str) -> tuple[bytes, os.stat_result]:
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise PublicationError(f"{role} evidence is unavailable: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PublicationError(f"{role} evidence must be a regular non-symlink file")
    if path.suffix.casefold() in RAW_SUFFIXES:
        raise PublicationError(f"{role} evidence is a forbidden raw artifact")
    markers = {canonical_key(part) for part in path.parts}
    if markers & RAW_PATH_MARKERS:
        raise PublicationError(f"{role} evidence path identifies forbidden raw content")
    payload = path.read_bytes()
    after = path.lstat()
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(payload) != info.st_size):
        raise PublicationError(f"{role} evidence changed while read")
    return payload, info


def normalized_path(path: pathlib.Path, sha256: str) -> str:
    path = pathlib.Path(path).resolve()
    try:
        return path.relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", path.name) or "artifact"
        return f"external/{sha256[:12]}/{safe_name}"


def parse_json(payload: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"{role} evidence is not JSON") from error
    if not isinstance(value, dict):
        raise PublicationError(f"{role} evidence must be an object")
    verify_body_hash(value, role)
    return value


def artifact_record(path: pathlib.Path, role: str, payload: bytes,
                    document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    result: dict[str, Any] = {
        "path": normalized_path(path, digest),
        "sha256": digest,
        "bytes": len(payload),
    }
    if document is not None:
        schema = document.get("schema")
        if isinstance(schema, str):
            result["schema"] = schema
        if isinstance(document.get("body_sha256"), str):
            result["body_sha256"] = document["body_sha256"]
        if isinstance(document.get("status"), str):
            result["status"] = document["status"]
    return result


def finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationError(f"{field} must be finite")
    return result


def integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublicationError(f"{field} must be an integer >= {minimum}")
    return value


def schema_for(role: str, document: Mapping[str, Any]) -> str | None:
    schema = document.get("schema")
    if role == "ci" and schema is None:
        return None
    if not isinstance(schema, str) or schema not in ROLE_SCHEMAS.get(role, set()):
        raise PublicationError(f"{role} evidence schema is not publication-eligible")
    if document.get("namespace", NAMESPACE) != NAMESPACE:
        raise PublicationError(f"{role} evidence namespace changed")
    return schema


def reject_forbidden_claims(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = canonical_key(str(key))
            if normalized in {
                "rank1claimed", "leaderboardclaim", "rank4replaced",
                "rank4replacementauthorized",
            } and child is not False:
                raise PublicationError(f"{path} contains a forbidden promotion claim: {key}")
            if normalized in {"uploadsauthorized", "submitclicks"} and \
                    isinstance(child, int) and not isinstance(child, bool) and child > 1:
                raise PublicationError(f"{path} permits more than one upload")
            reject_forbidden_claims(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_claims(child, f"{path}[{index}]")


def compact_offline_gate(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    if type(value.get("passed")) is bool:
        result["passed"] = value["passed"]
    if isinstance(value.get("status"), str):
        result["status"] = value["status"]
    for name in (
        "common_sign_accuracy", "common_weighted_huber",
        "canonical_sign_accuracy", "canonical_weighted_huber",
        "sign_accuracy", "weighted_huber",
    ):
        if name in value:
            result[name] = finite(value[name], f"offline {name}")
    return result or None


def summarize_family(document: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: document[key]
        for key in ("architecture", "arm", "seed", "status", "deployment_eligible")
        if key in document and isinstance(document[key], (str, int, bool))
    }
    gate = compact_offline_gate(document.get("offline_gate"))
    if gate is not None:
        result["offline_gate"] = gate
    return result


def summarize_development(document: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(document.get("eligible_architectures"), list):
        result["eligible_architectures"] = [
            value for value in document["eligible_architectures"] if isinstance(value, str)
        ]
    for key in ("retained_model_ids", "tuple_candidate_id", "profile", "status"):
        value = document.get(key)
        if isinstance(value, str) or (
            key == "retained_model_ids" and isinstance(value, list)
            and all(isinstance(item, str) for item in value)
        ):
            result[key] = value
    actual = document.get("actual_clock")
    if isinstance(actual, Mapping):
        colors = actual.get("color_wins")
        result["actual_clock"] = {
            "candidate_id": actual.get("candidate_id"),
            "pairs": integer(actual.get("pairs"), "development pairs"),
            "games": integer(actual.get("games"), "development games"),
            "wins": integer(actual.get("wins"), "development wins"),
            "color_wins": {
                "0": integer(colors.get("0"), "development color 0 wins"),
                "1": integer(colors.get("1"), "development color 1 wins"),
            } if isinstance(colors, Mapping) else None,
            "failures": integer(actual.get("failures"), "development failures"),
            "latency_ms": finite(actual.get("latency_ms"), "development latency"),
        }
    return result


def summarize_protected(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema") != "papersoccer.compact-value-bfm.final-aggregate.v1":
        return {"status": document.get("status"), "rank4_qualified": False}
    summary = document.get("summary")
    verdict = document.get("verdict")
    if not isinstance(summary, Mapping) or not isinstance(verdict, Mapping):
        raise PublicationError("protected aggregate omits summary/verdict")
    colors = summary.get("candidate_color_wins")
    failures = summary.get("failures")
    timing = summary.get("timing")
    uncontended = summary.get("uncontended_timing")
    if not all(isinstance(value, Mapping) for value in (colors, failures, timing, uncontended)):
        raise PublicationError("protected aggregate metrics are malformed")
    compact = {
        "status": document.get("status"),
        "games": integer(summary.get("games"), "protected games"),
        "candidate_wins": integer(summary.get("candidate_wins"), "protected wins"),
        "candidate_color_wins": {
            "0": integer(colors.get("0"), "protected color 0 wins"),
            "1": integer(colors.get("1"), "protected color 1 wins"),
        },
        "failure_count": sum(integer(value, "protected failure count")
                             for value in failures.values()),
        "maximum_turns": integer(summary.get("maximum_turns"), "protected turns"),
        "timing": {
            "first_max_ms": finite(timing.get("first_max_ms"), "protected first timing"),
            "later_max_ms": finite(timing.get("later_max_ms"), "protected later timing"),
            "uncontended_first_max_ms": finite(
                uncontended.get("first_max_ms"), "uncontended first timing"),
            "uncontended_later_max_ms": finite(
                uncontended.get("later_max_ms"), "uncontended later timing"),
        },
        "rank4_qualified": verdict.get("passed") is True,
    }
    qualified = (
        compact["rank4_qualified"] and compact["status"] == "rank4-qualified"
        and compact["games"] == 1000 and compact["candidate_wins"] >= 527
        and compact["candidate_color_wins"]["0"] >= 260
        and compact["candidate_color_wins"]["1"] >= 260
        and compact["failure_count"] == 0 and compact["maximum_turns"] <= 320
        and compact["timing"]["first_max_ms"] < 1000
        and compact["timing"]["later_max_ms"] < 200
        and compact["timing"]["uncontended_first_max_ms"] < 900
        and compact["timing"]["uncontended_later_max_ms"] < 180
    )
    if compact["rank4_qualified"] is not qualified:
        raise PublicationError("protected Rank-4 qualification claim contradicts metrics")
    return compact


def summarize_preflight(document: Mapping[str, Any]) -> dict[str, Any]:
    checks = document.get("checks")
    passed_checks = 0
    if isinstance(checks, Mapping):
        passed_checks = sum(value == "passed" for value in checks.values())
        if passed_checks != len(checks):
            raise PublicationError("preflight receipt contains a failed check")
    passed = document.get("status") == "passed"
    if not passed:
        raise PublicationError("preflight receipt is not passed")
    return {"status": "passed", "passed_checks": passed_checks}


def summarize_ci(document: Mapping[str, Any]) -> dict[str, Any]:
    jobs = document.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs or any(value != "success" for value in jobs.values()):
        raise PublicationError("CI receipt does not contain an all-green job roster")
    if document.get("conclusion") != "success" or document.get("head_branch") != "compact-value-bfm":
        raise PublicationError("CI receipt is not green on compact-value-bfm")
    return {
        "conclusion": "success",
        "run_id": integer(document.get("run_id"), "CI run id", 1),
        "head_sha": document.get("head_sha"),
        "successful_jobs": sorted(str(name) for name in jobs),
    }


def summarize_upload(document: Mapping[str, Any]) -> dict[str, Any]:
    if (document.get("status") != "submission-attested"
            or document.get("submit_clicks") != 1):
        raise PublicationError("upload receipt is not an exact-one attestation")
    return {
        "status": "submission-attested",
        "submit_clicks": 1,
        "submitted_at_utc": document.get("submitted_at_utc"),
        "candidate_commit": document.get("candidate_commit"),
        "source_sha256": document.get("source_sha256"),
        "source_bytes": integer(document.get("source_bytes"), "uploaded source bytes", 1),
        "agent_id": integer(document.get("agent_id"), "agent id", 1),
        "submission_id": integer(document.get("submission_id"), "submission id", 1),
    }


def summarize_live(document: Mapping[str, Any]) -> dict[str, Any]:
    identity = document.get("identity")
    if not isinstance(identity, Mapping):
        raise PublicationError("live receipt identity is missing")
    exact = integer(document.get("exact_games"), "live game count")
    if exact != 90:
        raise PublicationError("live receipt does not contain exactly 90 games")
    own = integer(document.get("focus_operational_failures"), "live own failures")
    opponent = integer(document.get("opponent_operational_failures"), "live opponent failures")
    status = document.get("status")
    accepted = status == "complete-accepted" and own == 0
    if status not in {"complete-accepted", "complete-rejected-own-failure"}:
        raise PublicationError("live receipt status is not complete")
    if document.get("training_eligible") is not False or \
            document.get("opponent_failures_count_as_strength_wins") is not False:
        raise PublicationError("live diagnostic-only policy changed")
    return {
        "status": status,
        "exact_games": exact,
        "accepted": accepted,
        "focus_operational_failures": own,
        "opponent_operational_failures": opponent,
        "agent_id": identity.get("agent_id"),
        "submission_id": identity.get("submission_id"),
        "source_sha256": identity.get("source_sha256"),
        "training_eligible": False,
        "opponent_failures_count_as_strength_wins": False,
    }


def reject_public_details(value: Any, path: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PublicationError(f"{path} contains a non-string key")
            if canonical_key(key) in FORBIDDEN_PUBLIC_KEYS:
                raise PublicationError(f"{path} exposes forbidden detail field {key}")
            if canonical_key(key) == "games" and not isinstance(child, int):
                raise PublicationError(f"{path} exposes per-game details")
            reject_public_details(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_public_details(child, f"{path}[{index}]")
    elif isinstance(value, str) and TRANSCRIPT_RE.fullmatch(value):
        raise PublicationError(f"{path} exposes a raw turn transcript")


def load_inputs(paths: Mapping[str, pathlib.Path | None]) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for role, path in paths.items():
        if path is None:
            continue
        payload, _ = safe_input(path, role)
        if role == "source":
            try:
                payload.decode("ascii")
            except UnicodeDecodeError as error:
                raise PublicationError("selected source is not ASCII") from error
            artifacts[role] = artifact_record(path, role, payload)
            continue
        document = parse_json(payload, role)
        schema_for(role, document)
        reject_forbidden_claims(document, role)
        artifacts[role] = artifact_record(path, role, payload, document)
        if role == "runtime":
            architecture = document.get("architecture")
            selection = document.get("selection")
            quantization = document.get("quantization")
            if not all(isinstance(value, Mapping)
                       for value in (architecture, selection, quantization)):
                raise PublicationError("selected runtime metadata is malformed")
            metrics[role] = {
                "architecture": architecture.get("name"),
                "dimensions": architecture.get("dimensions"),
                "arm": selection.get("arm"),
                "seed": selection.get("seed"),
                "payload_sha256": quantization.get("payload_sha256"),
            }
        elif role == "family":
            metrics[role] = summarize_family(document)
        elif role == "development":
            metrics[role] = summarize_development(document)
        elif role == "protected":
            metrics[role] = summarize_protected(document)
        elif role == "preflight":
            metrics[role] = summarize_preflight(document)
        elif role == "ci":
            metrics[role] = summarize_ci(document)
        elif role == "upload":
            metrics[role] = summarize_upload(document)
        elif role == "live":
            metrics[role] = summarize_live(document)
    return artifacts, metrics


def build_evidence(paths: Mapping[str, pathlib.Path | None]) -> dict[str, Any]:
    artifacts, metrics = load_inputs(paths)
    missing = [role for role in REQUIRED_COMPLETE if role not in artifacts]
    source_sha = artifacts.get("source", {}).get("sha256")
    upload = metrics.get("upload")
    live = metrics.get("live")
    protected = metrics.get("protected")
    ci = metrics.get("ci")
    if upload is not None and upload.get("source_sha256") != source_sha:
        raise PublicationError("upload source does not match selected source")
    if live is not None:
        if upload is None or any(live.get(key) != upload.get(key)
                                 for key in ("agent_id", "submission_id", "source_sha256")):
            raise PublicationError("live window does not match exact uploaded identity")
    if upload is not None and ci is not None and upload.get("candidate_commit") != ci.get("head_sha"):
        raise PublicationError("upload commit does not match green CI")
    complete = (
        not missing
        and protected is not None and protected.get("rank4_qualified") is True
        and metrics.get("preflight", {}).get("status") == "passed"
        and ci is not None and ci.get("conclusion") == "success"
        and upload is not None and upload.get("submit_clicks") == 1
        and live is not None and live.get("exact_games") == 90
        and live.get("accepted") is True
    )
    negative_results = [
        "No Rank-1 claim is made.",
        "The maintained Rank-4 source is not replaced.",
        "Live games are diagnostic-only and never training data.",
    ]
    if missing:
        negative_results.append("Publication remains incomplete: " + ", ".join(missing) + ".")
    if live is not None and not live.get("accepted"):
        negative_results.append("The complete live window is rejected because of own operational failure.")
    evidence = body_hashed({
        "schema": EVIDENCE_SCHEMA,
        "namespace": NAMESPACE,
        "status": "complete" if complete else "incomplete",
        "claims": {
            "rank1_claimed": False,
            "rank4_replaced": False,
            "rank4_qualified": bool(protected and protected.get("rank4_qualified")),
            "uploads": 1 if upload is not None else 0,
            "live_games": live.get("exact_games", 0) if live is not None else 0,
            "publication_complete": complete,
        },
        "policy": {
            "models_included": False,
            "datasets_included": False,
            "labels_included": False,
            "banks_included": False,
            "transcripts_included": False,
            "replay_details_included": False,
            "live_training_eligible": False,
            "exact_one_upload_required": True,
            "exact_live_games_required": 90,
            "rank4_control_sha256": RANK4_SHA256,
        },
        "artifacts": artifacts,
        "metrics": metrics,
        "missing_evidence": missing,
        "negative_results": negative_results,
    })
    verify_evidence(evidence)
    return evidence


def verify_evidence(evidence: Mapping[str, Any]) -> None:
    if evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("namespace") != NAMESPACE:
        raise PublicationError("public evidence schema/namespace is invalid")
    verify_body_hash(evidence, "public evidence")
    reject_public_details(evidence)
    claims = evidence.get("claims")
    policy = evidence.get("policy")
    if not isinstance(claims, Mapping) or not isinstance(policy, Mapping):
        raise PublicationError("public claims/policy are missing")
    if claims.get("rank1_claimed") is not False or claims.get("rank4_replaced") is not False:
        raise PublicationError("forbidden Rank-1 or Rank-4 replacement claim")
    if claims.get("uploads") not in (0, 1):
        raise PublicationError("publication does not enforce exact-one upload")
    if claims.get("live_games") not in (0, 90):
        raise PublicationError("publication live window is not absent or exactly 90")
    complete = claims.get("publication_complete") is True
    if evidence.get("status") != ("complete" if complete else "incomplete"):
        raise PublicationError("publication status contradicts completeness")
    if complete and (claims.get("rank4_qualified") is not True
                     or claims.get("uploads") != 1 or claims.get("live_games") != 90
                     or evidence.get("missing_evidence") != []):
        raise PublicationError("complete publication claim lacks exact evidence")
    for key in (
        "models_included", "datasets_included", "labels_included", "banks_included",
        "transcripts_included", "replay_details_included", "live_training_eligible",
    ):
        if policy.get(key) is not False:
            raise PublicationError(f"publication policy permits forbidden content: {key}")


def render_report(evidence: Mapping[str, Any] | None = None) -> str:
    status = "incomplete" if evidence is None else str(evidence["status"])
    claims = {} if evidence is None else evidence["claims"]
    missing = list(REQUIRED_COMPLETE) if evidence is None else evidence["missing_evidence"]
    body_sha = "not-yet-published" if evidence is None else evidence["body_sha256"]
    lines = [
        "# Compact Value-BFM evidence",
        "",
        f"Status: **{status}**.",
        "",
        "This report publishes only compact metrics, hashes, policy state, and negative",
        "results. It contains no model weights, datasets, labels, opening banks,",
        "transcripts, replay details, per-game records, or live game IDs.",
        "",
        "## Claims boundary",
        "",
        "- No Rank-1 claim is made.",
        "- The maintained Rank-4 source is not replaced.",
        "- Exactly one upload and exactly 90 matching complete live games are required.",
        "- Live games are diagnostic-only and never training data.",
        "",
        f"Evidence body SHA-256: `{body_sha}`.",
        "",
        "## Current state",
        "",
    ]
    if missing:
        lines.append("Missing terminal evidence: " + ", ".join(missing) + ".")
    else:
        lines.extend([
            f"- Strict Rank-4 qualification: `{claims.get('rank4_qualified')}`.",
            f"- Upload count: `{claims.get('uploads')}`.",
            f"- Matching live games: `{claims.get('live_games')}`.",
        ])
    lines.extend([
        "",
        "Run `python3 benchmarks/compact_value_bfm/publish.py verify` to verify this",
        "report and any compact evidence file present beside it.",
        "",
    ])
    return "\n".join(lines)


def verify_report(report: str, evidence: Mapping[str, Any] | None) -> None:
    required = (
        "No Rank-1 claim is made.",
        "maintained Rank-4 source is not replaced",
        "diagnostic-only and never training data",
    )
    if any(value not in report for value in required):
        raise PublicationError("REPORT.md omits a required claims boundary")
    expected_status = "incomplete" if evidence is None else evidence["status"]
    if f"Status: **{expected_status}**." not in report:
        raise PublicationError("REPORT.md status contradicts evidence")
    if evidence is not None and evidence["body_sha256"] not in report:
        raise PublicationError("REPORT.md does not bind the evidence body hash")


def publish(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = {role: getattr(arguments, role) for role in REQUIRED_COMPLETE}
    evidence = build_evidence(paths)
    atomic_write(arguments.output, canonical_json_bytes(evidence))
    atomic_write(arguments.report, render_report(evidence).encode("utf-8"))
    return evidence


def verify(arguments: argparse.Namespace) -> None:
    evidence = None
    if arguments.evidence.exists():
        payload, _ = safe_input(arguments.evidence, "published")
        evidence = parse_json(payload, "published")
        verify_evidence(evidence)
    report_payload, _ = safe_input(arguments.report, "report")
    try:
        report = report_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("REPORT.md is not UTF-8") from error
    verify_report(report, evidence)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish_parser = commands.add_parser("publish")
    for role in REQUIRED_COMPLETE:
        publish_parser.add_argument(f"--{role.replace('_', '-')}",
                                    dest=role, type=pathlib.Path)
    publish_parser.add_argument("--output", type=pathlib.Path,
                                default=HERE / "compact_evidence.json")
    publish_parser.add_argument("--report", type=pathlib.Path,
                                default=HERE / "REPORT.md")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--evidence", type=pathlib.Path,
                               default=HERE / "compact_evidence.json")
    verify_parser.add_argument("--report", type=pathlib.Path,
                               default=HERE / "REPORT.md")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "publish":
            evidence = publish(arguments)
            print(json.dumps({
                "status": evidence["status"],
                "body_sha256": evidence["body_sha256"],
                "output": str(arguments.output),
                "report": str(arguments.report),
            }, sort_keys=True))
        else:
            verify(arguments)
            print("compact Value-BFM publication verified")
        return 0
    except PublicationError as error:
        parser.exit(1, f"compact Value-BFM publication failure: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
