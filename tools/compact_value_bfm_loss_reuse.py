#!/usr/bin/env python3
"""Materialize leakage-isolated replay roots from a rejected training attempt.

The actual-clock Rank-4 runner deliberately does not retain game transcripts.
Full unprotected student-vs-Rank-4 trajectories are instead taken from the
sealed phase game manifest, while the attempt outcome and its selected gate
closure establish that the attempt was rejected.  Protected and live inputs
are consumed only through fingerprint-only exclusion artifacts.

This tool is intentionally non-mutating with respect to the campaign ledger.
Its content-addressed ``roots.tsv`` and replay-roots manifest can be supplied
as the ``--roots-tsv`` and ``--roots-manifest`` overrides when the next attempt
is opened.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from submissions.codingame.bots.compact_value_bfm import rank4_gate_support
from tools import compact_value_bfm_openings as openings
from tools import compact_value_bfm_qualification as qualification
from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import jacek_replay_corpus as corpus
from tools import jacek_replay_features as features
from tools import jacek_replay_pack as replay_pack


class LossReuseError(ValueError):
    """A rejected-attempt loss reuse input violates the isolation contract."""


REUSE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-loss-root-reuse.v1"
)
GATE_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-plan.v1"
)
GATE_EXECUTION_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-execution-claim.v1"
)
GATE_VARIANT_EXECUTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-variant-execution.v1"
)
GATE_EXECUTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-gate-execution.v1"
)
FULL_SEARCH_SELECTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-search-selection.v1"
)
FULL_QUALIFICATION_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-full-qualification-plan.v1"
)
SCREEN_REQUEST_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-screen-request.v1"
)
PIPELINE_SCHEMA = "papersoccer.compact-value-bfm-teacher-phase-pipeline.v1"
GAME_MANIFEST_SCHEMA = (
    "papersoccer.compact-value-bfm-teacher-phase-games.v1"
)
FINGERPRINT_SET_SCHEMA = (
    "papersoccer.compact-value-bfm-pilot-fingerprint-set.v1"
)
PROTECTED_FINGERPRINT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-"
    "protected-canonical-fingerprints.v1"
)
LIVE_FINGERPRINT_EVIDENCE_SCHEMA = challenger.LIVE_FINGERPRINT_EVIDENCE_SCHEMA
FEATURE_FINGERPRINT_DOMAIN = "canonical-sparse-active-u16le-v1"
STATE_FINGERPRINT_DOMAIN = "canonical-opening-state-serialization-v1"
FOUR_WAY_CANONICALIZATION = (
    "minimum-sha256-over-exact+rotate+reflect+rotate-reflect"
)
SOURCE = "rank4-teacher-development-loss"
STUDENT_RANK4_MODES = {
    "student-p1-vs-rank4": 0,
    "student-p2-vs-rank4": 1,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REFLECT = str.maketrans("01234567", "07654321")


def _canonical_json_bytes(value: object) -> bytes:
    return qualification.canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular(path: pathlib.Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LossReuseError(f"artifact is not a regular file: {path}")
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": qualification.sha256_file(path),
    }


def _verify_record(value: object, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or not {
        "path", "bytes", "sha256"
    }.issubset(value):
        raise LossReuseError(f"{label} record is malformed")
    try:
        path = pathlib.Path(str(value["path"]))
        expected_bytes = value["bytes"]
        expected_sha256 = value["sha256"]
    except (KeyError, TypeError, ValueError) as error:
        raise LossReuseError(f"{label} record is malformed") from error
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not isinstance(expected_sha256, str)
        or SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise LossReuseError(f"{label} record metadata is malformed")
    observed = _regular(path)
    if (
        observed["bytes"] != expected_bytes
        or observed["sha256"] != expected_sha256
    ):
        raise LossReuseError(f"{label} bytes changed")
    return path.resolve()


def _sealed_record(path: pathlib.Path, schema: str) -> dict[str, object]:
    regular = _regular(path)
    value = qualification.load_sealed(path.resolve(), schema)
    return {
        **regular,
        "schema": schema,
        "body_sha256": value["body_sha256"],
    }


def _verify_sealed_record(
    value: object, schema: str, label: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    path = _verify_record(value, label)
    try:
        document = qualification.load_sealed(path, schema)
    except Exception as error:
        raise LossReuseError(f"{label} is not a valid sealed artifact") from error
    if isinstance(value, Mapping) and (
        value.get("schema") not in {None, schema}
        or value.get("body_sha256") not in {None, document["body_sha256"]}
    ):
        raise LossReuseError(f"{label} sealed identity changed")
    return path, document


def _resolve_campaign_record(
    context: Mapping[str, Any], record: object, label: str,
) -> pathlib.Path:
    try:
        return challenger._resolve_campaign_artifact(
            record, plan=context["plan"], label=label
        )
    except Exception as error:
        raise LossReuseError(f"{label} campaign artifact changed") from error


def _fingerprints(values: object, label: str) -> list[str]:
    if (
        not isinstance(values, list)
        or not values
        or values != sorted(set(values))
        or any(
            not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
            for value in values
        )
    ):
        raise LossReuseError(f"{label} fingerprints are malformed")
    return list(values)


def _safe_fingerprint_source(
    path: pathlib.Path, *, role: str,
) -> tuple[str, set[str], dict[str, object]]:
    """Load one explicitly fingerprint-only source without following references."""

    try:
        _regular(path)
        value = qualification.load_sealed(path.resolve())
    except Exception as error:
        raise LossReuseError(f"{role} exclusion is not a sealed fingerprint file") from error
    schema = value.get("schema")
    classification = value.get("classification")
    if schema == FINGERPRINT_SET_SCHEMA:
        values = _fingerprints(value.get("fingerprints"), role)
        expected_classification = {
            "frozen-training:prior-train-fingerprints": "prior-train",
            "frozen-training:prior-validation-fingerprints": "prior-validation",
            "frozen-protected:mixed-development-fingerprints": (
                "mixed-development"
            ),
        }.get(role)
        if (
            value.get("canonicalization") != FOUR_WAY_CANONICALIZATION
            or value.get("fingerprint_domain") != FEATURE_FINGERPRINT_DOMAIN
            or value.get("fingerprint_count") != len(values)
            or value.get("source_paths_followed") is not False
            or value.get("contains_labels") is not False
            or value.get("contains_metrics") is not False
            or value.get("contains_transcripts") is not False
            or (
                expected_classification is not None
                and classification != expected_classification
            )
            or (
                role.startswith("frozen-live:")
                and (
                    not isinstance(classification, str)
                    or not classification.startswith("live")
                )
            )
        ):
            raise LossReuseError(f"{role} feature fingerprint contract changed")
        domain = FEATURE_FINGERPRINT_DOMAIN
    elif schema == PROTECTED_FINGERPRINT_SCHEMA:
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise LossReuseError(f"{role} protected fingerprint rows are absent")
        values = _fingerprints(sorted({
            str(row.get("canonical_sha256"))
            for row in rows if isinstance(row, Mapping)
        }), role)
        if (
            len(rows) != value.get("position_count")
            or len(values) != value.get("unique_canonical_count")
            or value.get("canonicalization")
            != "minimum-sha256-over-exact+rotate+reflect+rotate_reflect"
            or value.get("contains_labels") is not False
            or value.get("contains_metrics") is not False
            or value.get("contains_transcripts") is not False
            or not isinstance(classification, str)
            or not classification.startswith("protected")
        ):
            raise LossReuseError(f"{role} protected fingerprint contract changed")
        domain = STATE_FINGERPRINT_DOMAIN
    elif schema == challenger.DYNAMIC_EXCLUSION_SCHEMA:
        try:
            dynamic = challenger.validate_dynamic_exclusion(path.resolve())
        except Exception as error:
            raise LossReuseError(f"{role} dynamic exclusion changed") from error
        values = _fingerprints(dynamic.get("fingerprints"), role)
        domain = STATE_FINGERPRINT_DOMAIN
        classification = dynamic["classification"]
    elif schema == LIVE_FINGERPRINT_EVIDENCE_SCHEMA:
        try:
            _, value = challenger._load_live_fingerprint_evidence(
                _sealed_record(path, LIVE_FINGERPRINT_EVIDENCE_SCHEMA)
            )
        except Exception as error:
            raise LossReuseError(
                f"{role} trusted live fingerprint evidence changed"
            ) from error
        values = _fingerprints(value.get("fingerprints"), role)
        domain = STATE_FINGERPRINT_DOMAIN
        classification = "live-diagnostic-canonical-fingerprints"
    elif schema == challenger.DEVELOPMENT_EXCLUSION_SCHEMA:
        values = _fingerprints(value.get("fingerprints"), role)
        if (
            value.get("classification")
            != "unprotected-development-fingerprints"
            or value.get("fingerprint_count") not in {None, len(values)}
            or value.get("protected_or_live_data_included") is not False
        ):
            raise LossReuseError(f"{role} development exclusion changed")
        domain = STATE_FINGERPRINT_DOMAIN
    else:
        raise LossReuseError(f"{role} is not a recognized fingerprint-only schema")
    return domain, set(values), {
        **_regular(path),
        "schema": str(schema),
        "body_sha256": value["body_sha256"],
        "classification": str(classification),
        "fingerprint_domain": domain,
        "fingerprint_count": len(values),
        "source_paths_followed": False,
        "contains_positions": False,
        "contains_metrics": False,
        "contains_transcripts": False,
    }


def _collect_exclusions(
    context: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
    *, source_outcome: Mapping[str, Any], source_opened: Mapping[str, Any],
) -> dict[str, object]:
    """Load all sanitized exclusions before any transcript-bearing source."""

    records: dict[str, pathlib.Path] = {}
    inputs = context.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LossReuseError("campaign inputs are absent")

    training = inputs.get("training_inputs")
    protected = inputs.get("protected_exclusions")
    live = inputs.get("live_exclusions")
    if not all(isinstance(value, Mapping) for value in (training, protected, live)):
        raise LossReuseError("campaign exclusion sections are malformed")

    for name, record in sorted(training.items()):
        if "fingerprint" in str(name).casefold():
            records[f"frozen-training:{name}"] = _resolve_campaign_record(
                context, record, f"frozen training exclusion {name}"
            )
    for name, record in sorted(protected.items()):
        normalized = str(name).casefold()
        if "fingerprint" in normalized:
            records[f"frozen-protected:{name}"] = _resolve_campaign_record(
                context, record, f"frozen protected exclusion {name}"
            )
        elif normalized not in {"mixed-six", "fresh-exclusion-receipt"}:
            raise LossReuseError(
                f"protected exclusion {name!r} has no fingerprint-only projection"
            )
    for name, record in sorted(live.items()):
        if "fingerprint" not in str(name).casefold():
            raise LossReuseError(
                f"live exclusion {name!r} has no fingerprint-only projection"
            )
        records[f"frozen-live:{name}"] = _resolve_campaign_record(
            context, record, f"frozen live exclusion {name}"
        )

    if not any(role.startswith("frozen-training:") for role in records):
        raise LossReuseError("frozen train/validation fingerprint exclusions are absent")
    if not any(role.startswith("frozen-protected:") for role in records):
        raise LossReuseError("frozen protected fingerprint exclusions are absent")

    dynamic = source_opened.get("dynamic_exclusions")
    if not isinstance(dynamic, list):
        raise LossReuseError("attempt dynamic exclusion roster is malformed")
    for ordinal, record in enumerate(dynamic):
        try:
            path = challenger._verify_dynamic_exclusion_record(
                record, f"loss reuse dynamic exclusion {ordinal}"
            )
        except Exception as error:
            raise LossReuseError("attempt dynamic exclusion changed") from error
        records[f"dynamic:{ordinal:04d}"] = path

    # Earlier development gates remain excluded.  The source outcome's own
    # exclusion is deliberately not placed in the rejection union: it proves
    # these exact rows are unprotected development material and is the narrow
    # exception authorized by the campaign's failed-loss reuse policy.
    for entry in entries:
        if (
            entry.get("sequence") == source_outcome.get("sequence")
            and entry.get("body_sha256") == source_outcome.get("body_sha256")
        ):
            break
        if entry.get("event") != "attempt-outcome-recorded":
            continue
        record = entry.get("development_exclusion")
        path = _verify_record(record, "prior attempt development exclusion")
        records[
            f"prior-development:{entry.get('attempt')}:{entry.get('phase')}"
        ] = path

    union: dict[str, set[str]] = {
        FEATURE_FINGERPRINT_DOMAIN: set(),
        STATE_FINGERPRINT_DOMAIN: set(),
    }
    by_role: dict[str, set[str]] = {}
    source_records: dict[str, dict[str, object]] = {}
    cross_intersections = Counter()
    for role, path in sorted(records.items()):
        domain, values, source = _safe_fingerprint_source(path, role=role)
        cross_intersections[domain] += len(union[domain] & values)
        union[domain].update(values)
        by_role[role] = values
        source_records[role] = source
    required_training = {
        "frozen-training:prior-train-fingerprints",
        "frozen-training:prior-validation-fingerprints",
    }
    if not required_training.issubset(source_records):
        raise LossReuseError("both prior train and validation exclusions are required")
    if not any(
        role.startswith("frozen-protected:")
        and str(source.get("classification", "")).startswith("protected")
        for role, source in source_records.items()
    ):
        raise LossReuseError("fingerprint-only protected exclusion is absent")
    return {
        "union": union,
        "by_role": by_role,
        "sources": source_records,
        "cross_source_intersections": dict(cross_intersections),
    }


def _actions_from_record(record: Mapping[str, Any]) -> list[str]:
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise LossReuseError("accepted base root has no complete-turn trajectory")
    actions: list[str] = []
    for index, turn in enumerate(turns):
        if (
            not isinstance(turn, Mapping)
            or turn.get("player_id") != index % 2
            or not isinstance(turn.get("action"), str)
            or not turn["action"]
            or any(character not in "01234567" for character in turn["action"])
        ):
            raise LossReuseError("accepted base root has a malformed complete turn")
        actions.append(str(turn["action"]))
    return actions


def _replay(
    transcript: str, *, expected_winner: int,
) -> tuple[list[dict[str, str]], features.ReplayState]:
    actions = transcript.split("/") if transcript else []
    if (
        not actions
        or any(not action or any(char not in "01234567" for char in action)
               for action in actions)
        or expected_winner not in (0, 1)
    ):
        raise LossReuseError("loss trajectory is malformed")
    state = features.ReplayState()
    boundaries: list[dict[str, str]] = []
    for action in actions:
        if state.winner is not None:
            raise LossReuseError("loss trajectory continues after termination")
        boundaries.append({
            "state": openings.state_fingerprints(state)["canonical"],
            "feature": corpus.canonical_feature_fingerprint(
                features.encode_active(state)
            ).hex(),
        })
        try:
            features.apply_complete_turn(state, state.to_move, action)
        except ValueError as error:
            raise LossReuseError("loss trajectory contains an illegal turn") from error
    if state.winner != expected_winner:
        raise LossReuseError("loss trajectory winner disagrees with replay")
    return boundaries, state


def _reflect_transcript(transcript: str) -> str:
    return transcript.translate(REFLECT)


def _canonical_trajectory(
    transcript: str, *, winner: int, focus_player: int,
) -> dict[str, object]:
    variants = []
    # A horizontal reflection preserves the player-to-move sequence and can be
    # emitted as another legal transcript from the standard initial state.  A
    # 180-degree state rotation also swaps player identity, so it is used by
    # every boundary fingerprint below but is not emitted as a rewritten full
    # game (there is no player-one-to-move initial transcript encoding).
    for name, reflect in (
        ("exact", False),
        ("reflect", True),
    ):
        transformed = _reflect_transcript(transcript) if reflect else transcript
        transformed_winner = winner
        transformed_focus = focus_player
        boundaries, _state = _replay(
            transformed, expected_winner=transformed_winner
        )
        variants.append((
            transformed, name, transformed_winner, transformed_focus,
            boundaries,
        ))
    selected = min(variants, key=lambda item: (item[0], item[1]))
    sequence = [item["state"] for item in selected[4]]
    identity = _sha256_bytes(_canonical_json_bytes({
        "canonical_boundary_states": sequence,
        "complete_turns": len(sequence),
    }))
    # Every transformed replay must collapse to the same four-way state orbit.
    if any([item["state"] for item in variant[4]] != sequence for variant in variants):
        raise LossReuseError("trajectory symmetry canonicalization disagrees")
    return {
        "transcript": selected[0],
        "transform": selected[1],
        "winner": selected[2],
        "focus_player": selected[3],
        "boundaries": selected[4],
        "identity": identity,
    }


def _event_path(plan: Mapping[str, Any], entry: Mapping[str, Any]) -> pathlib.Path:
    ledger = pathlib.Path(str(plan.get("outputs", {}).get("ledger", "")))
    path = ledger / (
        f"{int(entry['sequence']):06d}-{entry['body_sha256']}.json"
    )
    path_record = {
        **_regular(path),
        "schema": challenger.LEDGER_SCHEMA,
        "body_sha256": entry["body_sha256"],
    }
    verified_path, document = _verify_sealed_record(
        path_record, challenger.LEDGER_SCHEMA, "attempt outcome ledger event"
    )
    if document != dict(entry):
        raise LossReuseError("attempt outcome ledger event differs from its chain")
    return verified_path


def _same_artifact(left: object, right: object) -> bool:
    return bool(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _utc_instant(value: object, label: str) -> dt.datetime:
    try:
        normalized = challenger.utc(value, label)
        return dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except Exception as error:
        raise LossReuseError(f"{label} is invalid") from error


def _gate_requests(
    gate: Mapping[str, Any], *, label: str,
) -> tuple[list[str], dict[str, tuple[pathlib.Path, dict[str, Any]]]]:
    records = gate.get("requests")
    if not isinstance(records, list) or not records:
        raise LossReuseError(f"{label} request roster is absent")
    result: dict[str, tuple[pathlib.Path, dict[str, Any]]] = {}
    for record_value in records:
        if not isinstance(record_value, Mapping):
            raise LossReuseError(f"{label} request roster is malformed")
        variant = record_value.get("variant")
        if not isinstance(variant, str) or not variant or variant in result:
            raise LossReuseError(f"{label} request roster is duplicated")
        path, document = _verify_sealed_record(
            record_value.get("request"), SCREEN_REQUEST_SCHEMA,
            f"{label} {variant} request",
        )
        if (
            document.get("search_variant") != variant
            or document.get("protected_tests_opened") is not False
            or record_value.get("request_body_sha256") not in {
                None, document["body_sha256"]
            }
        ):
            raise LossReuseError(f"{label} {variant} request binding changed")
        result[variant] = (path, document)
    roster = list(result)
    active_roster = gate.get("active_search_variant_roster")
    if active_roster is not None and active_roster != roster:
        raise LossReuseError(f"{label} active variant roster changed")
    return roster, result


def _gate_execution_closure(
    record_value: object, *, gate_path: pathlib.Path,
    gate: Mapping[str, Any], attempt: int, phase: str, label: str,
) -> dict[str, Any]:
    execution_path, execution = _verify_sealed_record(
        record_value, GATE_EXECUTION_SCHEMA, f"{label} execution"
    )
    roster, requests = _gate_requests(gate, label=label)
    receipts = execution.get("variant_receipts")
    if (
        execution.get("attempt") != attempt
        or execution.get("phase") != phase
        or not _same_artifact(execution.get("gate_plan"), _regular(gate_path))
        or execution.get("gate_plan_body_sha256") != gate.get("body_sha256")
        or execution.get("variant_order") != roster
        or not isinstance(receipts, Mapping)
        or set(receipts) != set(roster)
        or execution.get("status")
        != "complete-serial-one-worker-no-retry"
        or execution.get("retry_authorized") is not False
    ):
        raise LossReuseError(f"{label} execution binding changed")

    raw_results: dict[str, pathlib.Path] = {}
    activations: dict[str, dict[str, Any]] = {}
    previous_finished: dt.datetime | None = None
    plan_ready = (
        _utc_instant(gate.get("prepared_at_utc"), f"{label} preparation time")
        if gate.get("schema") == FULL_QUALIFICATION_PLAN_SCHEMA else None
    )
    for variant in roster:
        request_path, request = requests[variant]
        receipt_path, receipt = _verify_sealed_record(
            receipts[variant], GATE_VARIANT_EXECUTION_SCHEMA,
            f"{label} {variant} execution receipt",
        )
        del receipt_path
        claim_path, claim = _verify_sealed_record(
            receipt.get("claim"), GATE_EXECUTION_CLAIM_SCHEMA,
            f"{label} {variant} no-retry claim",
        )
        del claim_path
        raw_path = _verify_record(
            receipt.get("raw_result"), f"{label} {variant} raw result"
        )
        worker = claim.get("worker")
        prelaunch = claim.get("prelaunch_audit")
        run = receipt.get("execution")
        activation = receipt.get("profile_activation")
        metadata = request.get("search_variant_metadata")
        profile = (
            metadata.get("candidate_search_profile")
            if isinstance(metadata, Mapping) else None
        )
        if (
            receipt.get("attempt") != attempt
            or receipt.get("phase") != phase
            or receipt.get("gate_plan_body_sha256") != gate.get("body_sha256")
            or receipt.get("variant") != variant
            or not _same_artifact(receipt.get("request"), _regular(request_path))
            or receipt.get("status") != "complete-no-retry"
            or receipt.get("retry_authorized") is not False
            or claim.get("attempt") != attempt
            or claim.get("phase") != phase
            or claim.get("gate_plan_body_sha256") != gate.get("body_sha256")
            or claim.get("variant") != variant
            or not _same_artifact(claim.get("request"), _regular(request_path))
            or claim.get("request_body_sha256") != request.get("body_sha256")
            or claim.get("no_retry") is not True
            or not isinstance(worker, Mapping)
            or worker.get("workers") != 1
            or worker.get("threads_per_worker") != 1
            or worker.get("whole_bank_process") is not True
            or worker.get("process_nice") != 0
            or not isinstance(prelaunch, Mapping)
            or prelaunch.get("competing_rank4_gate_processes") != []
            or not isinstance(run, Mapping)
            or not isinstance(run.get("launched_at_utc"), str)
            or not isinstance(run.get("finished_at_utc"), str)
            or run.get("workers") != 1
            or run.get("threads_per_worker") != 1
            or run.get("whole_bank_process") is not True
            or run.get("variants_serial") is not True
            or run.get("process_nice") != 0
            or not isinstance(activation, Mapping)
            or activation.get("candidate_search_profile") != profile
            or not isinstance(activation.get("exercised"), bool)
        ):
            raise LossReuseError(f"{label} {variant} execution evidence changed")
        claimed = _utc_instant(
            claim.get("claimed_at_utc"), f"{label} {variant} claim time"
        )
        launched = _utc_instant(
            run["launched_at_utc"], f"{label} {variant} launch time"
        )
        finished = _utc_instant(
            run["finished_at_utc"], f"{label} {variant} finish time"
        )
        if (
            not claimed <= launched <= finished
            or (previous_finished is not None and claimed < previous_finished)
            or (plan_ready is not None and claimed < plan_ready)
        ):
            raise LossReuseError(f"{label} {variant} chronology changed")
        previous_finished = finished
        raw_results[variant] = raw_path
        activations[variant] = dict(activation)
    return {
        "path": execution_path,
        "document": execution,
        "requests": requests,
        "raw_results": raw_results,
        "activations": activations,
        "last_finished": previous_finished,
    }


def _validate_gate_result_chain(
    *, gate_path: pathlib.Path, gate: Mapping[str, Any],
    execution_record: object, result_records: object,
    selected: Mapping[str, Any], attempt: int, phase: str, label: str,
) -> dict[str, Any]:
    if (
        gate.get("attempt") != attempt
        or gate.get("phase") != phase
        or gate.get("protected_tests_opened") is not False
    ):
        raise LossReuseError(f"{label} plan identity changed")
    execution = _gate_execution_closure(
        execution_record, gate_path=gate_path, gate=gate,
        attempt=attempt, phase=phase, label=label,
    )
    if not isinstance(result_records, Mapping) or set(result_records) != set(
        execution["raw_results"]
    ):
        raise LossReuseError(f"{label} result roster changed")
    copied_results: dict[str, pathlib.Path] = {}
    for variant, raw_path in execution["raw_results"].items():
        copied = _verify_record(
            result_records[variant], f"{label} {variant} copied result"
        )
        if qualification.sha256_file(copied) != qualification.sha256_file(raw_path):
            raise LossReuseError(f"{label} {variant} result differs from execution")
        copied_results[variant] = copied

    variant = selected.get("search_variant")
    if not isinstance(variant, str) or variant not in execution["requests"]:
        raise LossReuseError(f"{label} selected variant is absent")
    request_path, request = execution["requests"][variant]
    bank_record = gate.get("bank")
    if (
        not isinstance(bank_record, Mapping)
        or bank_record.get("classification") != "fresh-unprotected"
        or request.get("bank") != bank_record
        or request.get("candidate_source") != selected.get("source")
    ):
        raise LossReuseError(f"{label} selected request changed")
    bank_manifest_path = _verify_record(
        bank_record.get("manifest"), f"{label} opening bank"
    )
    gate_tsv_path = _verify_record(
        bank_record.get("gate_tsv"), f"{label} gate TSV"
    )
    metadata = request.get("search_variant_metadata")
    expected_profile = (
        metadata.get("candidate_search_profile")
        if isinstance(metadata, Mapping) else None
    )
    try:
        bank_document = openings.validate_bank(bank_manifest_path)
        gate_result = rank4_gate_support.validate_result(
            copied_results[variant],
            expected_bank_sha256=qualification.sha256_file(gate_tsv_path),
            expected_candidate_sha256=request["candidate_source"]["sha256"],
            expected_candidate_search_profile=str(expected_profile),
        )
        activation = rank4_gate_support.require_search_profile_exercised(
            gate_result, expected_profile=str(expected_profile)
        )
    except Exception as error:
        raise LossReuseError(f"{label} selected result failed validation") from error
    if (
        expected_profile not in rank4_gate_support.SEARCH_PROFILES
        or bank_document.get("classification") != "unprotected-development"
        or gate_result.get("config", {}).get("mode") != "actual-clock"
        or activation != execution["activations"][variant]
    ):
        raise LossReuseError(f"{label} is not clean actual-clock evidence")
    return {
        **execution,
        "gate_path": gate_path,
        "gate": gate,
        "result_paths": copied_results,
        "selected_variant": variant,
        "selected_result_path": copied_results[variant],
        "selected_request_path": request_path,
        "selected_request": request,
        "bank_manifest_path": bank_manifest_path,
        "gate_tsv_path": gate_tsv_path,
        "activation": activation,
    }


def _source_closure(
    context: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
    *, attempt: int, phase: str,
) -> dict[str, Any]:
    opened = [
        entry for entry in entries
        if entry.get("event") == "attempt-opened"
        and entry.get("attempt") == attempt
    ]
    outcomes = [
        entry for entry in entries
        if entry.get("event") == "attempt-outcome-recorded"
        and entry.get("attempt") == attempt and entry.get("phase") == phase
    ]
    if len(opened) != 1 or len(outcomes) != 1:
        raise LossReuseError("attempt open/outcome ledger identity is not unique")
    source_opened, outcome = opened[0], outcomes[0]
    if (
        attempt <= 0
        or phase not in {"pilot", "full"}
        or outcome.get("admitted") is not False
        or not str(outcome.get("adaptation_route", "")).startswith(
            "open-next-attempt-"
        )
        or entries[-1].get("body_sha256") != outcome.get("body_sha256")
        or entries[-1].get("sequence") != outcome.get("sequence")
    ):
        raise LossReuseError(
            "loss reuse requires the current trained attempt's rejected terminal phase"
        )

    outcome_path, outcome_document = _verify_sealed_record(
        outcome.get("outcome_receipt"),
        challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
        "attempt outcome receipt",
    )
    if (
        outcome_document.get("attempt") != attempt
        or outcome_document.get("phase") != phase
        or outcome_document.get("status") != "complete"
        or outcome_document.get("protected_or_live_metrics_read") is not False
        or outcome_document.get("all_games_finished") is not True
    ):
        raise LossReuseError("attempt outcome receipt is not a clean terminal closure")
    closure = outcome_document.get("evidence_closure")
    if not isinstance(closure, Mapping) or closure.get("protected_tests_opened") is not False:
        raise LossReuseError("attempt outcome has no unprotected evidence closure")

    pipeline_path, pipeline = _verify_sealed_record(
        closure.get("pipeline_plan"), PIPELINE_SCHEMA, "phase pipeline plan"
    )
    outputs = pipeline.get("outputs")
    if (
        pipeline.get("attempt") != attempt
        or pipeline.get("phase") != phase
        or not isinstance(outputs, Mapping)
    ):
        raise LossReuseError("phase pipeline identity changed")
    raw_games_path = pathlib.Path(str(outputs.get("games", "")))
    raw_games_manifest_path = pathlib.Path(
        str(outputs.get("games_manifest", ""))
    )
    if raw_games_path.is_symlink() or raw_games_manifest_path.is_symlink():
        raise LossReuseError("phase game outputs are redirected")
    games_path = raw_games_path.resolve()
    games_manifest_path = raw_games_manifest_path.resolve()

    for field in (
        "gate_execution", "full_search_selection", "full_qualification_plan",
        "full_qualification_execution", "qualification_result",
    ):
        if outcome_document.get(field) != closure.get(field):
            raise LossReuseError(
                f"attempt outcome {field.replace('_', ' ')} closure changed"
            )

    search_ab_gate_path, search_ab_gate = _verify_sealed_record(
        closure.get("gate_plan"), GATE_PLAN_SCHEMA, "phase search A/B gate plan"
    )
    selected = closure.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise LossReuseError("rejected attempt has no selected candidate")
    search_ab = _validate_gate_result_chain(
        gate_path=search_ab_gate_path,
        gate=search_ab_gate,
        execution_record=closure.get("gate_execution"),
        result_records=closure.get("gate_results"),
        selected=selected,
        attempt=attempt,
        phase=phase,
        label="search A/B gate",
    )

    full_search_selection_path = None
    full_qualification_path = None
    full_qualification_execution_path = None
    if phase == "pilot":
        if any(closure.get(field) is not None for field in (
            "full_search_selection", "full_qualification_plan",
            "full_qualification_execution", "qualification_result",
        )):
            raise LossReuseError("pilot outcome unexpectedly contains full qualification")
        selected_gate = search_ab
        expected_development_record = search_ab_gate.get("development_exclusion")
    else:
        full_search_selection_path, full_search_selection = (
            _verify_sealed_record(
                closure.get("full_search_selection"),
                FULL_SEARCH_SELECTION_SCHEMA,
                "full search selection",
            )
        )
        selected_at = _utc_instant(
            full_search_selection.get("selected_at_utc"),
            "full search selection time",
        )
        selection_results = full_search_selection.get("search_ab_results")
        if (
            full_search_selection.get("attempt") != attempt
            or full_search_selection.get("phase") != "full"
            or full_search_selection.get("selected_candidate") != selected
            or full_search_selection.get(
                "selected_before_qualification_bank_read"
            ) is not True
            or full_search_selection.get("qualification_bank_read") is not False
            or not _same_artifact(
                full_search_selection.get("search_ab_gate_plan"),
                _regular(search_ab_gate_path),
            )
            or not _same_artifact(
                full_search_selection.get("search_ab_execution"),
                _regular(search_ab["path"]),
            )
            or not isinstance(selection_results, Mapping)
            or set(selection_results) != set(search_ab["result_paths"])
            or search_ab["last_finished"] is None
            or selected_at < search_ab["last_finished"]
        ):
            raise LossReuseError("full search selection lost its A/B ancestry")
        for variant, result_path in search_ab["result_paths"].items():
            selected_ab_result = _verify_record(
                selection_results[variant],
                f"full search selection {variant} A/B result",
            )
            if qualification.sha256_file(selected_ab_result) != qualification.sha256_file(
                result_path
            ):
                raise LossReuseError("full search selection A/B result changed")

        full_qualification_path, full_qualification = _verify_sealed_record(
            closure.get("full_qualification_plan"),
            FULL_QUALIFICATION_PLAN_SCHEMA,
            "full qualification plan",
        )
        qualification_prepared_at = _utc_instant(
            full_qualification.get("prepared_at_utc"),
            "full qualification preparation time",
        )
        if (
            full_qualification.get("attempt") != attempt
            or full_qualification.get("phase") != "full"
            or full_qualification.get("selected_candidate") != selected
            or full_qualification.get(
                "qualification_bank_opened_after_selection"
            ) is not True
            or not _same_artifact(
                full_qualification.get("full_search_selection"),
                _regular(full_search_selection_path),
            )
            or full_qualification.get("search_ab_bank")
            != search_ab_gate.get("bank")
            or full_qualification.get("bank_disjointness", {}).get("passed")
            is not True
            or full_qualification.get("protected_tests_opened") is not False
            or qualification_prepared_at < selected_at
        ):
            raise LossReuseError("full qualification plan binding changed")
        qualifier_results = {
            str(selected.get("search_variant")): closure.get(
                "qualification_result"
            )
        }
        selected_gate = _validate_gate_result_chain(
            gate_path=full_qualification_path,
            gate=full_qualification,
            execution_record=closure.get("full_qualification_execution"),
            result_records=qualifier_results,
            selected=selected,
            attempt=attempt,
            phase="full",
            label="post-selection full qualification",
        )
        full_qualification_execution_path = selected_gate["path"]
        qualifier_request = selected_gate["selected_request"]
        if (
            qualifier_request.get("gate_purpose") != "full-qualification"
            or not _same_artifact(
                qualifier_request.get("full_search_selection"),
                _regular(full_search_selection_path),
            )
        ):
            raise LossReuseError("full qualification request lost its frozen selection")
        expected_development_record = full_qualification.get(
            "development_exclusion"
        )

    development_path, development = _verify_sealed_record(
        outcome.get("development_exclusion"),
        challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
        "source attempt development exclusion",
    )
    if (
        development.get("attempt") != attempt
        or development.get("phase") != phase
        or development.get("protected_or_live_data_included") is not False
        or not _same_artifact(
            outcome_document.get("development_exclusion"),
            _regular(development_path),
        )
        or not _same_artifact(expected_development_record, _regular(development_path))
        or (
            phase == "full"
            and (
                development.get("includes_search_ab_bank") is not True
                or development.get(
                    "includes_post_selection_qualification_bank"
                ) is not True
            )
        )
    ):
        raise LossReuseError("source development exclusion lost its gate binding")

    attempt_inputs = source_opened.get("attempt_inputs")
    if not isinstance(attempt_inputs, Mapping):
        raise LossReuseError("source attempt inputs are absent")
    roots_tsv = _resolve_campaign_record(
        context, attempt_inputs.get("roots_tsv"), "source attempt roots TSV"
    )
    roots_manifest = _resolve_campaign_record(
        context, attempt_inputs.get("roots_manifest"),
        "source attempt roots manifest",
    )
    return {
        "opened": source_opened,
        "outcome": outcome,
        "outcome_path": outcome_path,
        "pipeline_path": pipeline_path,
        "pipeline": pipeline,
        "games_path": games_path,
        "games_manifest_path": games_manifest_path,
        "gate_path": selected_gate["gate_path"],
        "gate": selected_gate["gate"],
        "gate_result_path": selected_gate["selected_result_path"],
        "gate_request_path": selected_gate["selected_request_path"],
        "gate_bank_manifest_path": selected_gate["bank_manifest_path"],
        "gate_tsv_path": selected_gate["gate_tsv_path"],
        "search_ab_gate_path": search_ab_gate_path,
        "search_ab_execution_path": search_ab["path"],
        "search_ab_result_paths": search_ab["result_paths"],
        "full_search_selection_path": full_search_selection_path,
        "full_qualification_path": full_qualification_path,
        "full_qualification_execution_path": full_qualification_execution_path,
        "development_path": development_path,
        "development": development,
        "base_roots_tsv": roots_tsv,
        "base_roots_manifest": roots_manifest,
        "selected_variant": selected_gate["selected_variant"],
        "search_profile_activation": selected_gate["activation"],
    }


def _load_base_roots(
    tsv_path: pathlib.Path, manifest_path: pathlib.Path,
) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
    try:
        manifest = replay_pack.load_roots(manifest_path)
        rendered = replay_pack.teacher_tsv_bytes(manifest)
    except Exception as error:
        raise LossReuseError("source attempt roots failed replay validation") from error
    if rendered != tsv_path.read_bytes():
        raise LossReuseError("source attempt roots TSV differs from its manifest")
    records = copy.deepcopy(manifest["accepted"])
    trajectory_ids: set[str] = set()
    state_fingerprints: set[str] = set()
    feature_fingerprints: set[str] = set()
    root_splits: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise LossReuseError("source attempt root record is malformed")
        if record.get("protected") is True or str(
            record.get("classification", "")
        ).casefold().startswith("protected"):
            raise LossReuseError("protected root entered source attempt inputs")
        actions = _actions_from_record(record)
        winner = record.get("winner")
        if winner not in (0, 1):
            raise LossReuseError("source attempt root winner is malformed")
        boundaries, _state = _replay(
            "/".join(actions), expected_winner=int(winner)
        )
        trajectory_ids.add(_sha256_bytes(_canonical_json_bytes({
            "canonical_boundary_states": [item["state"] for item in boundaries],
            "complete_turns": len(boundaries),
        })))
        state_fingerprints.update(item["state"] for item in boundaries)
        feature_fingerprints.update(item["feature"] for item in boundaries)
        root_group = record.get("root_group_id", record.get("group_id"))
        split = record.get("split")
        if not isinstance(root_group, str) or not root_group or split not in {
            "train", "validation", "test"
        }:
            raise LossReuseError("source root group/split is malformed")
        prior = root_splits.setdefault(root_group, str(split))
        if prior != split:
            raise LossReuseError("source root group crosses frozen splits")
    return records, trajectory_ids, state_fingerprints, feature_fingerprints


def _load_phase_games(
    games_path: pathlib.Path, manifest_path: pathlib.Path, *,
    pipeline: Mapping[str, Any], attempt: int, phase: str,
) -> list[dict[str, Any]]:
    try:
        manifest = qualification.load_sealed(
            manifest_path, GAME_MANIFEST_SCHEMA
        )
    except Exception as error:
        raise LossReuseError("phase game manifest failed sealed validation") from error
    rows = manifest.get("rows")
    if (
        manifest.get("pipeline_body_sha256") != pipeline.get("body_sha256")
        or manifest.get("attempt") != attempt
        or manifest.get("phase") != phase
        or not isinstance(rows, list)
        or manifest.get("games") != len(rows)
        or [row.get("game_ordinal") for row in rows if isinstance(row, Mapping)]
        != list(range(len(rows)))
    ):
        raise LossReuseError("phase game manifest does not cover its sealed phase")
    lines = ["group_id\tsource\twinner\ttranscript"]
    game_ids: set[str] = set()
    for expected_ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise LossReuseError("phase game manifest contains a non-object row")
        game_id = row.get("game_id")
        prefix_turns = row.get("prefix_turns")
        transcript = row.get("transcript")
        if (
            row.get("game_ordinal") != expected_ordinal
            or not isinstance(game_id, str)
            or not game_id
            or not game_id.isascii()
            or game_id in game_ids
            or not isinstance(row.get("actor_mode"), str)
            or not row["actor_mode"]
            or not isinstance(row.get("root_group_id"), str)
            or not row["root_group_id"]
            or row.get("source") != challenger.CAMPAIGN_ID
            or row.get("winner") not in (0, 1)
            or not isinstance(transcript, str)
            or not transcript
            or isinstance(prefix_turns, bool)
            or not isinstance(prefix_turns, int)
            or not 0 <= prefix_turns < len(transcript.split("/"))
        ):
            raise LossReuseError("phase game manifest row lineage is malformed")
        game_ids.add(game_id)
        try:
            lines.append("\t".join((
                str(row["root_group_id"]), str(row["source"]),
                str(row["winner"]), str(row["transcript"]),
            )))
        except KeyError as error:
            raise LossReuseError("phase game manifest row is incomplete") from error
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if (
        games_path.is_symlink()
        or not games_path.is_file()
        or games_path.read_bytes() != payload
        or manifest.get("games_sha256") != _sha256_bytes(payload)
    ):
        raise LossReuseError("phase game TSV differs from its sealed manifest")
    return [dict(row) for row in rows]


def _split_records(
    records: list[dict[str, Any]], *, base_count: int,
) -> dict[str, str]:
    frozen: dict[str, str] = {}
    representatives: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        group = str(record.get("root_group_id", record.get("group_id", "")))
        if not group:
            raise LossReuseError("root group identity is absent")
        stratum = (
            record.get("source"), record.get("focus_player"),
            record.get("winner"), record.get("opponent_tier"),
        )
        if group in representatives:
            prior = representatives[group]
            prior_stratum = (
                prior.get("source"), prior.get("focus_player"),
                prior.get("winner"), prior.get("opponent_tier"),
            )
            if prior_stratum != stratum:
                raise LossReuseError("one root group spans incompatible strata")
        else:
            representatives[group] = {
                **record, "group_id": group,
            }
        if index < base_count:
            split = record.get("split")
            if split not in {"train", "validation", "test"}:
                raise LossReuseError("base root has no frozen split")
            previous = frozen.setdefault(group, str(split))
            if previous != split:
                raise LossReuseError("base root group crosses frozen splits")
    try:
        assignments = corpus._assignment_for_strata(
            list(representatives.values()), frozen
        )
    except Exception as error:
        raise LossReuseError("deterministic whole-root split failed") from error
    for record in records:
        group = str(record.get("root_group_id", record.get("group_id")))
        record["split"] = assignments[group]
    return assignments


def _write_content_addressed(
    directory: pathlib.Path, payload: bytes, suffix: str,
) -> pathlib.Path:
    if directory.is_symlink():
        raise LossReuseError("loss reuse output directory is redirected")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_sha256_bytes(payload)}{suffix}"
    try:
        qualification.atomic_write_once(path, payload)
    except Exception as error:
        raise LossReuseError("loss reuse output collided with different bytes") from error
    return path.resolve()


def materialize_loss_reuse(
    *, campaign_plan: pathlib.Path, attempt: int, phase: str,
    output_directory: pathlib.Path,
) -> dict[str, pathlib.Path]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise LossReuseError("loss reuse requires a positive trained attempt")
    if phase not in {"pilot", "full"}:
        raise LossReuseError("loss reuse phase must be pilot or full")
    try:
        context = challenger.validate_campaign(campaign_plan.resolve())
        entries = challenger.load_ledger(context["plan"])
    except Exception as error:
        raise LossReuseError("campaign or attempt ledger failed validation") from error
    source = _source_closure(
        context, entries, attempt=attempt, phase=phase
    )

    # This call is deliberately before either transcript-bearing root input or
    # phase game manifest is opened.
    exclusion_context = _collect_exclusions(
        context, entries, source_outcome=source["outcome"],
        source_opened=source["opened"],
    )
    source_development = set(_fingerprints(
        source["development"].get("fingerprints"),
        "source development",
    ))

    base_records, trajectory_ids, retained_states, retained_features = (
        _load_base_roots(
            source["base_roots_tsv"], source["base_roots_manifest"]
        )
    )
    rows = _load_phase_games(
        source["games_path"], source["games_manifest_path"],
        pipeline=source["pipeline"], attempt=attempt, phase=phase,
    )

    rejected: list[dict[str, object]] = []
    new_records: list[dict[str, Any]] = []
    game_counts = Counter()
    exclusion_intersections = Counter()
    for row in rows:
        mode = row.get("actor_mode")
        if mode not in STUDENT_RANK4_MODES:
            game_counts["non_student_rank4"] += 1
            continue
        student = STUDENT_RANK4_MODES[str(mode)]
        winner = row.get("winner")
        if winner not in (0, 1):
            raise LossReuseError("student-vs-Rank-4 game winner is malformed")
        if winner == student:
            game_counts["student_wins"] += 1
            continue
        game_counts["student_losses"] += 1
        transcript = row.get("transcript")
        prefix_turns = row.get("prefix_turns")
        if (
            not isinstance(transcript, str)
            or isinstance(prefix_turns, bool)
            or not isinstance(prefix_turns, int)
        ):
            raise LossReuseError("student loss trajectory lineage is malformed")
        canonical = _canonical_trajectory(
            transcript, winner=int(winner), focus_player=student
        )
        boundaries = canonical["boundaries"]
        if not isinstance(boundaries, list) or not 0 <= prefix_turns < len(boundaries):
            raise LossReuseError("student loss prefix boundary is malformed")
        if any(item["state"] not in source_development for item in boundaries):
            raise LossReuseError(
                "student loss trajectory is absent from its development exclusion"
            )
        identity = str(canonical["identity"])
        game_id = str(row.get("game_id", ""))
        evidence = {
            "game_id": game_id,
            "game_ordinal": row.get("game_ordinal"),
            "actor_mode": mode,
            "student_player": student,
            "winner": winner,
            "prefix_turns": prefix_turns,
            "transcript_sha256": _sha256_bytes(transcript.encode("ascii")),
            "canonical_trajectory_sha256": identity,
        }
        if identity in trajectory_ids:
            game_counts["symmetry_duplicate_trajectories"] += 1
            rejected.append({**evidence, "reason": "symmetry-duplicate-trajectory"})
            continue

        # The branch boundary itself is inherited from an existing root.  Only
        # post-branch, nonterminal boundaries are new loss positions.
        derived = boundaries[prefix_turns + 1:]
        if not derived:
            raise LossReuseError("student loss has no post-branch positions")
        matched_roles: set[str] = set()
        for boundary in derived:
            for role, values in exclusion_context["by_role"].items():
                domain = exclusion_context["sources"][role][
                    "fingerprint_domain"
                ]
                candidate = (
                    boundary["feature"]
                    if domain == FEATURE_FINGERPRINT_DOMAIN
                    else boundary["state"]
                )
                if candidate in values:
                    matched_roles.add(role)
            if (
                boundary["state"] in retained_states
                or boundary["feature"] in retained_features
            ):
                matched_roles.add("retained-root-or-loss")
        if matched_roles:
            for role in matched_roles:
                exclusion_intersections[role] += 1
            game_counts["excluded_intersecting_trajectories"] += 1
            rejected.append({
                **evidence,
                "reason": "post-branch-symmetry-intersection",
                "matched_roles": sorted(matched_roles),
            })
            continue

        canonical_transcript = str(canonical["transcript"])
        actions = canonical_transcript.split("/")
        canonical_prefix = prefix_turns
        group_id = f"rank4-loss:{identity}"
        source_record_sha256 = _sha256_bytes(_canonical_json_bytes(evidence))
        record = {
            "game_id": game_id,
            "group_id": group_id,
            "root_group_id": group_id,
            "source": SOURCE,
            "focus_player": canonical["focus_player"],
            "winner": canonical["winner"],
            "opponent_tier": "maintained-rank-4",
            "turns": [
                {"player_id": index % 2, "action": action}
                for index, action in enumerate(actions)
            ],
            "source_record_sha256": source_record_sha256,
            "classification": "unprotected-development-loss-reuse",
            "protected": False,
            "loss_reuse": {
                **evidence,
                "canonical_transform": canonical["transform"],
                "canonical_prefix_turns": canonical_prefix,
                "post_branch_position_count": len(derived),
                "protected_or_live_position_intersections": 0,
                "all_exclusion_intersections": 0,
            },
        }
        new_records.append(record)
        trajectory_ids.add(identity)
        retained_states.update(item["state"] for item in derived)
        retained_features.update(item["feature"] for item in derived)
        game_counts["accepted_loss_trajectories"] += 1

    if not new_records:
        raise LossReuseError("rejected attempt has no leakage-isolated loss trajectories")

    records = [*base_records, *new_records]
    assignments = _split_records(records, base_count=len(base_records))
    records.sort(key=lambda record: str(record["group_id"]))

    provisional = {"accepted": records}
    tsv_payload = replay_pack.teacher_tsv_bytes(provisional)
    tsv_path = _write_content_addressed(
        output_directory, tsv_payload, ".loss-reuse-roots.tsv"
    )
    split_counts = Counter(assignments.values())
    if not all(split_counts[split] > 0 for split in ("train", "validation", "test")):
        raise LossReuseError("loss reuse roots do not retain all three whole-root splits")

    outcome_event_path = _event_path(context["plan"], source["outcome"])
    manifest: dict[str, object] = {
        "schema": corpus.ROOT_SCHEMA,
        "reuse_schema": REUSE_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "tool_sha256": {
            "normalizer": qualification.sha256_file(pathlib.Path(__file__)),
            "features": qualification.sha256_file(pathlib.Path(features.__file__)),
        },
        "exclusion_boundary": {
            "read_before_candidate_sources": True,
            "scope": "new-post-branch-nonterminal-boundaries",
            "canonicalization": FOUR_WAY_CANONICALIZATION,
            "sources": exclusion_context["sources"],
            "cross_source_intersections": exclusion_context[
                "cross_source_intersections"
            ],
            "source_attempt_development_exclusion": _sealed_record(
                source["development_path"],
                challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            ),
            "source_attempt_exception": (
                "own unprotected development fingerprints prove lineage and are "
                "not a rejection source"
            ),
            "candidate_intersections": dict(sorted(exclusion_intersections.items())),
            "protected_or_live_fingerprints_used_only_as_negative_exclusions": True,
            "protected_or_live_positions_used": False,
            "protected_or_live_metrics_used": False,
            "protected_or_live_transcripts_used": False,
        },
        "sources": [
            {
                "kind": "base-replay-roots",
                "roots_tsv": _regular(source["base_roots_tsv"]),
                "roots_manifest": _sealed_record(
                    source["base_roots_manifest"], corpus.ROOT_SCHEMA
                ),
            },
            {
                "kind": "rejected-unprotected-attempt",
                "campaign_plan": _sealed_record(
                    campaign_plan.resolve(), challenger.PLAN_SCHEMA
                ),
                "attempt_outcome_event": _sealed_record(
                    outcome_event_path, challenger.LEDGER_SCHEMA
                ),
                "attempt_outcome": _sealed_record(
                    source["outcome_path"],
                    challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
                ),
                "pipeline_plan": _sealed_record(
                    source["pipeline_path"], PIPELINE_SCHEMA
                ),
                "games": _regular(source["games_path"]),
                "games_manifest": _sealed_record(
                    source["games_manifest_path"], GAME_MANIFEST_SCHEMA
                ),
                "search_ab_gate_plan": _sealed_record(
                    source["search_ab_gate_path"], GATE_PLAN_SCHEMA,
                ),
                "search_ab_gate_execution": _sealed_record(
                    source["search_ab_execution_path"], GATE_EXECUTION_SCHEMA,
                ),
                "search_ab_results": {
                    variant: _regular(path)
                    for variant, path in source["search_ab_result_paths"].items()
                },
                "full_search_selection": (
                    None
                    if source["full_search_selection_path"] is None
                    else _sealed_record(
                        source["full_search_selection_path"],
                        FULL_SEARCH_SELECTION_SCHEMA,
                    )
                ),
                "full_qualification_plan": (
                    None
                    if source["full_qualification_path"] is None
                    else _sealed_record(
                        source["full_qualification_path"],
                        FULL_QUALIFICATION_PLAN_SCHEMA,
                    )
                ),
                "full_qualification_execution": (
                    None
                    if source["full_qualification_execution_path"] is None
                    else _sealed_record(
                        source["full_qualification_execution_path"],
                        GATE_EXECUTION_SCHEMA,
                    )
                ),
                "qualification_result": (
                    _regular(source["gate_result_path"])
                    if phase == "full" else None
                ),
                "selected_gate_plan": _sealed_record(
                    source["gate_path"],
                    (
                        FULL_QUALIFICATION_PLAN_SCHEMA
                        if phase == "full" else GATE_PLAN_SCHEMA
                    ),
                ),
                "selected_gate_request": _sealed_record(
                    source["gate_request_path"],
                    SCREEN_REQUEST_SCHEMA,
                ),
                "selected_gate_bank": _regular(
                    source["gate_bank_manifest_path"]
                ),
                "selected_gate_tsv": _regular(source["gate_tsv_path"]),
                "selected_gate_result": _regular(source["gate_result_path"]),
                "selected_search_variant": source["selected_variant"],
                "selected_search_profile_activation": source[
                    "search_profile_activation"
                ],
                "attempt": attempt,
                "phase": phase,
                "attempt_rejected": True,
                "gate_classification": "fresh-unprotected",
                "loss_selection": (
                    "student-vs-maintained-rank4 and terminal winner != student"
                ),
                "candidate_metrics_used_for_loss_selection": False,
                "protected_or_live_data_used_for_hypothesis_choice": False,
            },
        ],
        "split_policy": (
            "preserve all base whole-root assignments; deterministically re-split "
            "new canonical loss root groups toward 80/10/10 stratified by source, "
            "focus color, outcome, and opponent tier"
        ),
        "split_parent": {
            **_regular(source["base_roots_manifest"]),
            "frozen_groups": len({
                str(record.get("root_group_id", record["group_id"]))
                for record in base_records
            }),
        },
        "accepted": records,
        "excluded": sorted(
            rejected,
            key=lambda item: (
                str(item.get("reason")), str(item.get("game_id")),
                int(item.get("game_ordinal", -1)),
            ),
        ),
        "structurally_rejected": [],
        "counts": {
            "base_roots": len(base_records),
            "new_loss_roots": len(new_records),
            "accepted": len(records),
            "excluded_loss_roots": len(rejected),
            "phase_games": len(rows),
            "game_classification": dict(sorted(game_counts.items())),
            "split_games": {
                split: split_counts[split]
                for split in ("train", "validation", "test")
            },
        },
        "source_roots": _regular(tsv_path),
        "output_sha256": qualification.sha256_file(tsv_path),
    }
    manifest["body_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_path = _write_content_addressed(
        output_directory, manifest_payload, ".loss-reuse-roots.json"
    )
    validate_loss_reuse_manifest(manifest_path)
    return {"roots_tsv": tsv_path, "roots_manifest": manifest_path}


def _validate_loss_reuse_manifest(
    path: pathlib.Path, *, archived_roots_tsv: pathlib.Path | None,
) -> dict[str, Any]:
    if path.is_symlink():
        raise LossReuseError("loss reuse roots manifest is redirected")
    try:
        manifest = replay_pack.load_roots(path.resolve())
    except Exception as error:
        raise LossReuseError("loss reuse roots manifest failed validation") from error
    if manifest.get("reuse_schema") != REUSE_SCHEMA:
        raise LossReuseError("loss reuse schema changed")
    if path.name.split(".", 1)[0] != qualification.sha256_file(path):
        raise LossReuseError("loss reuse manifest is not content addressed")
    boundary = manifest.get("exclusion_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("read_before_candidate_sources") is not True
        or boundary.get("scope") != "new-post-branch-nonterminal-boundaries"
        or boundary.get(
            "protected_or_live_fingerprints_used_only_as_negative_exclusions"
        ) is not True
        or boundary.get("protected_or_live_positions_used") is not False
        or boundary.get("protected_or_live_metrics_used") is not False
        or boundary.get("protected_or_live_transcripts_used") is not False
    ):
        raise LossReuseError("loss reuse exclusion boundary changed")
    source_roots_record = manifest.get("source_roots")
    if archived_roots_tsv is None:
        source_roots = _verify_record(
            source_roots_record, "loss reuse roots TSV"
        )
    else:
        archived_record = _regular(archived_roots_tsv)
        if (
            not isinstance(source_roots_record, Mapping)
            or set(source_roots_record) != {"path", "bytes", "sha256"}
            or not isinstance(source_roots_record.get("path"), str)
            or not source_roots_record["path"]
            or isinstance(source_roots_record.get("bytes"), bool)
            or not isinstance(source_roots_record.get("bytes"), int)
            or source_roots_record["bytes"] < 0
            or not isinstance(source_roots_record.get("sha256"), str)
            or SHA256_RE.fullmatch(source_roots_record["sha256"]) is None
            or archived_record["bytes"] != source_roots_record["bytes"]
            or archived_record["sha256"] != source_roots_record["sha256"]
        ):
            raise LossReuseError("archived loss reuse roots TSV binding changed")
        source_roots = archived_roots_tsv.resolve()
    if (
        manifest.get("output_sha256") != qualification.sha256_file(source_roots)
        or replay_pack.teacher_tsv_bytes(manifest) != source_roots.read_bytes()
        or source_roots.name.split(".", 1)[0]
        != qualification.sha256_file(source_roots)
    ):
        raise LossReuseError("loss reuse roots TSV binding changed")
    new = [
        record for record in manifest["accepted"]
        if isinstance(record, Mapping) and record.get("source") == SOURCE
    ]
    if not new:
        raise LossReuseError("loss reuse accepted-root policy changed")
    identities: set[str] = set()
    for record in new:
        lineage = record.get("loss_reuse")
        actions = _actions_from_record(record)
        if not isinstance(lineage, Mapping):
            raise LossReuseError("loss reuse accepted-root lineage is absent")
        mode = lineage.get("actor_mode")
        student = STUDENT_RANK4_MODES.get(str(mode))
        try:
            canonical = _canonical_trajectory(
                "/".join(actions), winner=int(record.get("winner", -1)),
                focus_player=int(record.get("focus_player", -1)),
            )
        except (TypeError, ValueError) as error:
            raise LossReuseError("loss reuse accepted root no longer replays") from error
        identity = str(canonical["identity"])
        prefix_turns = lineage.get("canonical_prefix_turns")
        if (
            record.get("classification")
            != "unprotected-development-loss-reuse"
            or record.get("protected") is not False
            or record.get("group_id") != f"rank4-loss:{identity}"
            or record.get("root_group_id") != record.get("group_id")
            or identity in identities
            or lineage.get("canonical_trajectory_sha256") != identity
            or lineage.get("student_player") != student
            or student not in (0, 1)
            or lineage.get("winner") == student
            or record.get("winner") != lineage.get("winner")
            or record.get("focus_player") != student
            or isinstance(prefix_turns, bool)
            or not isinstance(prefix_turns, int)
            or not 0 <= prefix_turns < len(actions) - 1
            or lineage.get("post_branch_position_count")
            != len(actions) - prefix_turns - 1
            or lineage.get("protected_or_live_position_intersections") != 0
            or lineage.get("all_exclusion_intersections") != 0
        ):
            raise LossReuseError("loss reuse accepted-root policy changed")
        identities.add(identity)
    assignments = replay_pack.frozen_assignments(manifest)
    if not all(
        split in set(assignments.values())
        for split in ("train", "validation", "test")
    ):
        raise LossReuseError("loss reuse roots lost a whole-root split")
    counts = manifest.get("counts")
    split_counts = Counter(assignments.values())
    if (
        not isinstance(counts, Mapping)
        or counts.get("new_loss_roots") != len(new)
        or counts.get("accepted") != len(manifest["accepted"])
        or counts.get("base_roots") != len(manifest["accepted"]) - len(new)
        or counts.get("split_games") != {
            split: split_counts[split]
            for split in ("train", "validation", "test")
        }
    ):
        raise LossReuseError("loss reuse manifest counts changed")
    rejected_source = next((
        source for source in manifest.get("sources", [])
        if isinstance(source, Mapping)
        and source.get("kind") == "rejected-unprotected-attempt"
    ), None)
    activation = (
        rejected_source.get("selected_search_profile_activation")
        if isinstance(rejected_source, Mapping) else None
    )
    source_phase = (
        rejected_source.get("phase")
        if isinstance(rejected_source, Mapping) else None
    )
    search_ab_results = (
        rejected_source.get("search_ab_results")
        if isinstance(rejected_source, Mapping) else None
    )
    selected_variant = (
        rejected_source.get("selected_search_variant")
        if isinstance(rejected_source, Mapping) else None
    )
    full_fields = (
        "full_search_selection", "full_qualification_plan",
        "full_qualification_execution", "qualification_result",
    )
    phase_closure_valid = bool(
        source_phase in {"pilot", "full"}
        and isinstance(search_ab_results, Mapping)
        and isinstance(selected_variant, str)
        and selected_variant in search_ab_results
        and isinstance(rejected_source.get("search_ab_gate_plan"), Mapping)
        and rejected_source["search_ab_gate_plan"].get("schema")
        == GATE_PLAN_SCHEMA
        and isinstance(rejected_source.get("search_ab_gate_execution"), Mapping)
        and rejected_source["search_ab_gate_execution"].get("schema")
        == GATE_EXECUTION_SCHEMA
    ) if isinstance(rejected_source, Mapping) else False
    if phase_closure_valid and source_phase == "pilot":
        phase_closure_valid = bool(
            all(rejected_source.get(field) is None for field in full_fields)
            and rejected_source.get("selected_gate_plan")
            == rejected_source.get("search_ab_gate_plan")
            and rejected_source.get("selected_gate_result")
            == search_ab_results[selected_variant]
        )
    elif phase_closure_valid:
        phase_closure_valid = bool(
            isinstance(rejected_source.get("full_search_selection"), Mapping)
            and rejected_source["full_search_selection"].get("schema")
            == FULL_SEARCH_SELECTION_SCHEMA
            and isinstance(rejected_source.get("full_qualification_plan"), Mapping)
            and rejected_source["full_qualification_plan"].get("schema")
            == FULL_QUALIFICATION_PLAN_SCHEMA
            and isinstance(
                rejected_source.get("full_qualification_execution"), Mapping
            )
            and rejected_source["full_qualification_execution"].get("schema")
            == GATE_EXECUTION_SCHEMA
            and rejected_source.get("selected_gate_plan")
            == rejected_source.get("full_qualification_plan")
            and rejected_source.get("selected_gate_result")
            == rejected_source.get("qualification_result")
        )
    if (
        not isinstance(rejected_source, Mapping)
        or rejected_source.get("attempt_rejected") is not True
        or rejected_source.get("gate_classification") != "fresh-unprotected"
        or rejected_source.get("candidate_metrics_used_for_loss_selection")
        is not False
        or rejected_source.get(
            "protected_or_live_data_used_for_hypothesis_choice"
        ) is not False
        or not isinstance(activation, Mapping)
        or activation.get("exercised") is not True
        or not phase_closure_valid
    ):
        raise LossReuseError("loss reuse rejected-attempt provenance changed")
    return manifest


def validate_loss_reuse_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Validate a materialized reuse manifest and its original TSV closure."""

    return _validate_loss_reuse_manifest(path, archived_roots_tsv=None)


def validate_archived_loss_reuse_manifest(
    path: pathlib.Path, *, roots_tsv: pathlib.Path,
) -> dict[str, Any]:
    """Validate a copied manifest against its campaign-local copied TSV.

    The sealed manifest intentionally retains the original materialization path
    as provenance.  Attempt-ledger replay must not dereference that disposable
    staging path after the manifest and TSV have been copied into the campaign.
    """

    return _validate_loss_reuse_manifest(
        path, archived_roots_tsv=roots_tsv,
    )


def validate_loss_reuse_for_campaign(
    path: pathlib.Path, *, campaign_plan: pathlib.Path,
    expected_source_attempt: int, expected_source_phase: str,
) -> dict[str, Any]:
    """Deeply rederive an existing reuse artifact in its original directory."""

    manifest = validate_loss_reuse_manifest(path.resolve())
    source = next((
        item for item in manifest.get("sources", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "rejected-unprotected-attempt"
    ), None)
    if (
        not isinstance(source, Mapping)
        or source.get("attempt") != expected_source_attempt
        or source.get("phase") != expected_source_phase
        or source.get("campaign_plan")
        != _sealed_record(campaign_plan.resolve(), challenger.PLAN_SCHEMA)
    ):
        raise LossReuseError("loss reuse belongs to another parent attempt")
    reproduced = materialize_loss_reuse(
        campaign_plan=campaign_plan.resolve(),
        attempt=expected_source_attempt,
        phase=expected_source_phase,
        output_directory=path.resolve().parent,
    )
    if (
        reproduced["roots_manifest"] != path.resolve()
        or pathlib.Path(str(manifest["source_roots"]["path"])).resolve()
        != reproduced["roots_tsv"]
    ):
        raise LossReuseError("loss reuse does not rederive byte-for-byte")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--campaign-plan", type=pathlib.Path, required=True)
    materialize.add_argument("--attempt", type=int, required=True)
    materialize.add_argument("--phase", choices=("pilot", "full"), required=True)
    materialize.add_argument("--output-directory", type=pathlib.Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--roots-manifest", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "materialize":
            outputs = materialize_loss_reuse(
                campaign_plan=arguments.campaign_plan,
                attempt=arguments.attempt,
                phase=arguments.phase,
                output_directory=arguments.output_directory,
            )
            result: object = {
                name: str(path) for name, path in outputs.items()
            }
        else:
            manifest = validate_loss_reuse_manifest(arguments.roots_manifest)
            result = {
                "roots_manifest": str(arguments.roots_manifest.resolve()),
                "roots_tsv": manifest["source_roots"]["path"],
                "new_loss_roots": manifest["counts"]["new_loss_roots"],
                "status": "valid",
            }
    except (LossReuseError, qualification.QualificationError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
