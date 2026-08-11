#!/usr/bin/env python3
"""Aggregate full-turn replay decision audits without reading any match bank."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Callable, Iterable, Sequence


AUDIT_SCHEMA_VERSION = "fullturn-decision-audit-v3"
ANALYSIS_SCHEMA_VERSION = "fullturn-decision-analysis-v2"
ARENA_BATCH_SCHEMA_VERSION = "papersoccer.codingame-arena-batch.v1"
ARENA_GAME_SCHEMA_VERSION = "papersoccer.codingame-arena-game.v1"
ARENA_BINDING_SCHEMA_VERSION = "papersoccer.codingame-arena-binding.v1"
ARENA_SOURCE_BINDING_STATUS = "asserted-not-api-verified"
MINIMUM_NAMED_OPPONENT_GAMES = 3
MAXIMUM_CANDIDATE_WORK = 3_000_000
MAXIMUM_CANDIDATE_TREE_NODES = 120_000
MAXIMUM_CANDIDATE_ACTIONS = 250
MAXIMUM_CANDIDATE_PARTIAL_PATHS = 50_000
MAXIMUM_REFERENCE_NODES = 3_000_000
MAXIMUM_REFERENCE_TURN_DEPTH = 32
MAXIMUM_INITIAL_HEURISTIC_SCORE = 100_000
INITIAL_PROOF_SCORE_MAGNITUDES = {999_998, 999_999}
MAXIMUM_INPUT_BYTES = 128 * 1024 * 1024
MAXIMUM_ROWS = 1_000_000
MAXIMUM_LINE_BYTES = 2 * 1024 * 1024

STRING_FIELDS = (
    "schema_version",
    "game_id",
    "state_id",
    "transcript_prefix",
    "color",
    "result",
    "actual_action",
    "candidate_action",
    "reference_action",
    "replay_correction_lookup_action",
    "replay_correction_action",
    "reconstructed_deployed_action",
    "actual_retained_tactical_class",
    "reference_retained_tactical_class",
    "initial_eval_best_action",
)

INTEGER_FIELDS = (
    "turn_index",
    "own_decision_index",
    "candidate_player",
    "winner",
    "diagnostic_root_actions",
    "diagnostic_root_partial_paths",
    "diagnostic_root_completed_actions",
    "diagnostic_root_duplicates",
    "diagnostic_root_truncations",
    "actual_action_retained_ordinal",
    "actual_boundary_retained_ordinal",
    "candidate_action_retained_ordinal",
    "candidate_boundary_retained_ordinal",
    "reference_action_retained_ordinal",
    "reference_boundary_retained_ordinal",
    "initial_eval_best_score",
    "initial_eval_best_retained_ordinal",
    "actual_initial_eval_rank",
    "candidate_initial_eval_rank",
    "reference_initial_eval_rank",
    "candidate_work_limit",
    "candidate_tree_node_limit",
    "candidate_max_actions",
    "candidate_nonroot_actions",
    "candidate_max_partial_paths",
    "candidate_first_time_limit_ms",
    "candidate_later_time_limit_ms",
    "candidate_time_limit_ms",
    "candidate_work",
    "candidate_tree_nodes",
    "candidate_expansions",
    "candidate_child_evaluations",
    "candidate_generator_partial_paths",
    "candidate_completed_actions",
    "candidate_generator_duplicates",
    "candidate_tactical_actions",
    "candidate_generator_truncations",
    "candidate_root_score",
    "reference_nodes_limit",
    "reference_depth_limit",
    "reference_first_time_limit_ms",
    "reference_later_time_limit_ms",
    "reference_time_limit_ms",
    "reference_nodes",
    "reference_completed_turn_depth",
    "reference_attempted_turn_depth",
    "reference_root_score",
    "value_blend_percent",
    "teacher_residual_percent",
)

OPTIONAL_INTEGER_FIELDS = (
    "actual_initial_eval_score",
    "candidate_initial_eval_score",
    "reference_initial_eval_score",
)

FLOAT_FIELDS = (
    "candidate_exploration",
    "candidate_fpu",
    "candidate_final_visit_weight",
    "candidate_elapsed_ms",
    "reference_elapsed_ms",
)

BOOLEAN_FIELDS = (
    "replay_correction_lookup_found",
    "replay_correction_valid",
    "candidate_matches_actual",
    "reference_matches_actual",
    "candidate_matches_reference",
    "reconstructed_deployed_matches_actual",
    "diagnostic_root_exhaustive",
    "candidate_budget_exhausted",
    "candidate_deadline_reached",
    "candidate_node_cap_reached",
    "reference_budget_exhausted",
    "codingame_clock_mode",
    "candidate_root_only",
    "candidate_bfm_change_assessable",
    "candidate_bfm_changed_from_initial_best",
)

JSON_FIELD_ORDER = (
    "schema_version",
    "input_provenance",
    "game_id",
    "state_id",
    "transcript_prefix",
    "turn_index",
    "own_decision_index",
    "candidate_player",
    "color",
    "winner",
    "result",
    "actual_action",
    "candidate_action",
    "reference_action",
    "replay_correction_lookup_found",
    "replay_correction_lookup_action",
    "replay_correction_valid",
    "replay_correction_action",
    "reconstructed_deployed_action",
    "candidate_matches_actual",
    "reference_matches_actual",
    "candidate_matches_reference",
    "reconstructed_deployed_matches_actual",
    "diagnostic_root_actions",
    "diagnostic_root_exhaustive",
    "diagnostic_root_partial_paths",
    "diagnostic_root_completed_actions",
    "diagnostic_root_duplicates",
    "diagnostic_root_truncations",
    "actual_action_retained_ordinal",
    "actual_boundary_retained_ordinal",
    "actual_retained_tactical_class",
    "candidate_action_retained_ordinal",
    "candidate_boundary_retained_ordinal",
    "reference_action_retained_ordinal",
    "reference_boundary_retained_ordinal",
    "reference_retained_tactical_class",
    "initial_eval_best_action",
    "initial_eval_best_score",
    "initial_eval_best_retained_ordinal",
    "actual_initial_eval_score",
    "actual_initial_eval_rank",
    "candidate_initial_eval_score",
    "candidate_initial_eval_rank",
    "reference_initial_eval_score",
    "reference_initial_eval_rank",
    "candidate_bfm_change_assessable",
    "candidate_bfm_changed_from_initial_best",
    "candidate_work_limit",
    "candidate_tree_node_limit",
    "candidate_max_actions",
    "candidate_nonroot_actions",
    "candidate_max_partial_paths",
    "candidate_exploration",
    "candidate_fpu",
    "candidate_final_visit_weight",
    "candidate_root_only",
    "candidate_first_time_limit_ms",
    "candidate_later_time_limit_ms",
    "candidate_time_limit_ms",
    "candidate_elapsed_ms",
    "candidate_work",
    "candidate_tree_nodes",
    "candidate_expansions",
    "candidate_child_evaluations",
    "candidate_generator_partial_paths",
    "candidate_completed_actions",
    "candidate_generator_duplicates",
    "candidate_tactical_actions",
    "candidate_generator_truncations",
    "candidate_budget_exhausted",
    "candidate_deadline_reached",
    "candidate_node_cap_reached",
    "candidate_root_score",
    "reference_nodes_limit",
    "reference_depth_limit",
    "reference_first_time_limit_ms",
    "reference_later_time_limit_ms",
    "reference_time_limit_ms",
    "reference_elapsed_ms",
    "reference_nodes",
    "reference_completed_turn_depth",
    "reference_attempted_turn_depth",
    "reference_budget_exhausted",
    "reference_root_score",
    "value_blend_percent",
    "teacher_residual_percent",
    "codingame_clock_mode",
)

TSV_FIELD_ORDER = tuple(
    "input_provenance_json" if field == "input_provenance" else field
    for field in JSON_FIELD_ORDER
)

RUN_CONFIGURATION_FIELDS = (
    "candidate_work_limit",
    "candidate_tree_node_limit",
    "candidate_max_actions",
    "candidate_nonroot_actions",
    "candidate_max_partial_paths",
    "candidate_exploration",
    "candidate_fpu",
    "candidate_final_visit_weight",
    "candidate_root_only",
    "candidate_first_time_limit_ms",
    "candidate_later_time_limit_ms",
    "reference_nodes_limit",
    "reference_depth_limit",
    "reference_first_time_limit_ms",
    "reference_later_time_limit_ms",
    "value_blend_percent",
    "teacher_residual_percent",
    "codingame_clock_mode",
)

ORDINAL_FIELDS = (
    "actual_action_retained_ordinal",
    "actual_boundary_retained_ordinal",
    "candidate_action_retained_ordinal",
    "candidate_boundary_retained_ordinal",
    "reference_action_retained_ordinal",
    "reference_boundary_retained_ordinal",
    "initial_eval_best_retained_ordinal",
)

INITIAL_EVAL_RANK_FIELDS = (
    "actual_initial_eval_rank",
    "candidate_initial_eval_rank",
    "reference_initial_eval_rank",
)

NONNEGATIVE_INTEGER_FIELDS = tuple(
    field
    for field in INTEGER_FIELDS
    if field not in ORDINAL_FIELDS
    and field not in INITIAL_EVAL_RANK_FIELDS
    and field not in (
        "candidate_root_score",
        "reference_root_score",
        "initial_eval_best_score",
    )
)

TACTICAL_CLASSES = {
    "not-retained",
    "immediate-win",
    "forced-cutoff",
    "safe-handoff",
    "opponent-immediate-win",
    "terminal-loss",
}

GAME_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
ACTION_PATTERN = re.compile(r"[0-7]{1,1024}\Z")
TRANSCRIPT_PATTERN = re.compile(r"(?:[0-7]+(?:/[0-7]+)*)?\Z")
PROVENANCE_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
POSITIVE_INTEGER_TEXT_PATTERN = re.compile(r"[1-9][0-9]*\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")

ARENA_PURPOSE = {
    "diagnostic_only": True,
    "training_eligible": False,
    "note": (
        "arena observations may influence bot development and must not be "
        "treated as untouched evaluation or direct expert training labels"
    ),
}
ARENA_OPERATIONAL_STATUSES = {
    "ok",
    "empty-output",
    "invalid-output",
    "illegal-action",
    "runtime-error",
    "timeout",
}

ARENA_PROVENANCE_FIELDS = (
    "agent_id",
    "arena_manifest_sha256",
    "asserted_source_sha256",
    "asserted_submission_id",
    "collector_sha256",
    "exclusion_registry_sha256",
    "repository_commit",
    "run_id",
    "source_binding_status",
)


class AuditAnalysisError(ValueError):
    """An audit cannot be safely or unambiguously aggregated."""


def _valid_initial_evaluation_score(score: int) -> bool:
    return (
        abs(score) <= MAXIMUM_INITIAL_HEURISTIC_SCORE
        or abs(score) in INITIAL_PROOF_SCORE_MAGNITUDES
    )


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditAnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AuditAnalysisError(f"non-finite JSON number: {value}")


def _strict_json_loads(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except AuditAnalysisError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise AuditAnalysisError(f"{context}: invalid JSON: {error}") from error


def _parse_tsv_integer(value: str, field: str, line_number: int) -> int:
    if not INTEGER_PATTERN.fullmatch(value):
        raise AuditAnalysisError(
            f"line {line_number}: {field} is not a canonical integer"
        )
    return int(value)


def _parse_tsv_float(value: str, field: str, line_number: int) -> float:
    if not value or value.strip() != value:
        raise AuditAnalysisError(f"line {line_number}: {field} is not a float")
    try:
        result = float(value)
    except ValueError as error:
        raise AuditAnalysisError(
            f"line {line_number}: {field} is not a float"
        ) from error
    if not math.isfinite(result):
        raise AuditAnalysisError(f"line {line_number}: {field} is not finite")
    return result


def _parse_tsv_boolean(value: str, field: str, line_number: int) -> bool:
    if value not in ("0", "1"):
        raise AuditAnalysisError(
            f"line {line_number}: {field} must be encoded as 0 or 1"
        )
    return value == "1"


def _parse_json_lines(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        raise AuditAnalysisError("audit is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise AuditAnalysisError(f"line {line_number}: blank JSONL row")
        if len(line.encode("utf-8")) > MAXIMUM_LINE_BYTES:
            raise AuditAnalysisError(f"line {line_number}: row is too large")
        row = _strict_json_loads(line, f"line {line_number}")
        if not isinstance(row, dict):
            raise AuditAnalysisError(f"line {line_number}: row must be an object")
        rows.append(row)
        if len(rows) > MAXIMUM_ROWS:
            raise AuditAnalysisError("audit exceeds the row limit")
    return rows


def _parse_tsv(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if len(lines) < 2:
        raise AuditAnalysisError("TSV audit must contain a header and data")
    header = tuple(lines[0].removesuffix("\r").split("\t"))
    if len(header) != len(set(header)):
        raise AuditAnalysisError("TSV header contains duplicate fields")
    if header != TSV_FIELD_ORDER:
        raise AuditAnalysisError("TSV header does not match audit schema v3")

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines[1:], 2):
        line = raw_line.removesuffix("\r")
        if not line:
            raise AuditAnalysisError(f"line {line_number}: blank TSV row")
        if len(line.encode("utf-8")) > MAXIMUM_LINE_BYTES:
            raise AuditAnalysisError(f"line {line_number}: row is too large")
        values = line.split("\t")
        if len(values) != len(header):
            raise AuditAnalysisError(
                f"line {line_number}: expected {len(header)} fields, got {len(values)}"
            )
        raw = dict(zip(header, values))
        row: dict[str, Any] = {}
        for field in JSON_FIELD_ORDER:
            if field == "input_provenance":
                row[field] = _strict_json_loads(
                    raw["input_provenance_json"],
                    f"line {line_number} input_provenance_json",
                )
            elif field in INTEGER_FIELDS:
                row[field] = _parse_tsv_integer(raw[field], field, line_number)
            elif field in OPTIONAL_INTEGER_FIELDS:
                row[field] = (
                    None
                    if raw[field] == ""
                    else _parse_tsv_integer(raw[field], field, line_number)
                )
            elif field in FLOAT_FIELDS:
                row[field] = _parse_tsv_float(raw[field], field, line_number)
            elif field in BOOLEAN_FIELDS:
                row[field] = _parse_tsv_boolean(raw[field], field, line_number)
            else:
                row[field] = raw[field]
        rows.append(row)
        if len(rows) > MAXIMUM_ROWS:
            raise AuditAnalysisError("audit exceeds the row limit")
    return rows


def _validate_provenance(value: Any, line_number: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise AuditAnalysisError(
            f"line {line_number}: input_provenance must be an object"
        )
    if len(value) > 64:
        raise AuditAnalysisError(f"line {line_number}: too many provenance fields")
    result: dict[str, str] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not PROVENANCE_KEY_PATTERN.fullmatch(key):
            raise AuditAnalysisError(
                f"line {line_number}: invalid provenance key {key!r}"
            )
        if (
            not isinstance(entry, str)
            or not entry
            or len(entry) > 512
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in entry)
        ):
            raise AuditAnalysisError(
                f"line {line_number}: invalid provenance value for {key}"
            )
        result[key] = entry
    return dict(sorted(result.items()))


def _validate_field_types(row: dict[str, Any], line_number: int) -> None:
    actual_fields = set(row)
    expected_fields = set(JSON_FIELD_ORDER)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise AuditAnalysisError(
            f"line {line_number}: schema fields differ; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for field in STRING_FIELDS:
        if not isinstance(row[field], str):
            raise AuditAnalysisError(f"line {line_number}: {field} must be a string")
    for field in INTEGER_FIELDS:
        if type(row[field]) is not int:
            raise AuditAnalysisError(f"line {line_number}: {field} must be an integer")
    for field in OPTIONAL_INTEGER_FIELDS:
        if row[field] is not None and type(row[field]) is not int:
            raise AuditAnalysisError(
                f"line {line_number}: {field} must be an integer or null"
            )
    for field in FLOAT_FIELDS:
        if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
            raise AuditAnalysisError(f"line {line_number}: {field} must be finite")
        row[field] = float(row[field])
    for field in BOOLEAN_FIELDS:
        if type(row[field]) is not bool:
            raise AuditAnalysisError(f"line {line_number}: {field} must be boolean")


def _validate_row(row: dict[str, Any], line_number: int) -> None:
    _validate_field_types(row, line_number)
    row["input_provenance"] = _validate_provenance(
        row["input_provenance"], line_number
    )
    if row["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise AuditAnalysisError(
            f"line {line_number}: unsupported schema_version {row['schema_version']!r}"
        )
    if not GAME_ID_PATTERN.fullmatch(row["game_id"]):
        raise AuditAnalysisError(f"line {line_number}: invalid game_id")
    if (
        not row["state_id"]
        or len(row["state_id"]) > 256
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in row["state_id"])
    ):
        raise AuditAnalysisError(f"line {line_number}: invalid state_id")
    if (
        len(row["transcript_prefix"]) > 65_536
        or not TRANSCRIPT_PATTERN.fullmatch(row["transcript_prefix"])
    ):
        raise AuditAnalysisError(f"line {line_number}: invalid transcript_prefix")
    for field in (
        "actual_action",
        "candidate_action",
        "reference_action",
        "reconstructed_deployed_action",
        "initial_eval_best_action",
    ):
        if not ACTION_PATTERN.fullmatch(row[field]):
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
    for field in ("replay_correction_lookup_action", "replay_correction_action"):
        if row[field] and not ACTION_PATTERN.fullmatch(row[field]):
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
    if row["initial_eval_best_retained_ordinal"] < 0:
        raise AuditAnalysisError(
            f"line {line_number}: initial best action is not retained"
        )
    for field in ("candidate_root_score", "reference_root_score"):
        if row[field] is not None and abs(row[field]) > 1_000_000:
            raise AuditAnalysisError(
                f"line {line_number}: {field} exceeds candidate score bounds"
            )
    for field in ("initial_eval_best_score", *OPTIONAL_INTEGER_FIELDS):
        if row[field] is not None and not _valid_initial_evaluation_score(row[field]):
            raise AuditAnalysisError(
                f"line {line_number}: {field} has an impossible initial-evaluation score"
            )
    for field in NONNEGATIVE_INTEGER_FIELDS:
        if row[field] < 0:
            raise AuditAnalysisError(f"line {line_number}: {field} is negative")
    for field in ORDINAL_FIELDS:
        if row[field] < -1:
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
        if row[field] >= row["diagnostic_root_actions"]:
            raise AuditAnalysisError(
                f"line {line_number}: {field} exceeds retained root actions"
            )
    for field in INITIAL_EVAL_RANK_FIELDS:
        if row[field] < -1 or row[field] > row["diagnostic_root_actions"]:
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
    if row["diagnostic_root_actions"] <= 0:
        raise AuditAnalysisError(f"line {line_number}: root action set is empty")
    if not 1 <= row["candidate_work_limit"] <= MAXIMUM_CANDIDATE_WORK:
        raise AuditAnalysisError(f"line {line_number}: invalid candidate work limit")
    if not 2 <= row["candidate_tree_node_limit"] <= MAXIMUM_CANDIDATE_TREE_NODES:
        raise AuditAnalysisError(f"line {line_number}: invalid candidate tree limit")
    if (
        not 1 <= row["candidate_max_actions"] <= MAXIMUM_CANDIDATE_ACTIONS
        or not 0 <= row["candidate_nonroot_actions"] <= MAXIMUM_CANDIDATE_ACTIONS
    ):
        raise AuditAnalysisError(f"line {line_number}: invalid candidate action cap")
    if not 1 <= row["candidate_max_partial_paths"] <= MAXIMUM_CANDIDATE_PARTIAL_PATHS:
        raise AuditAnalysisError(
            f"line {line_number}: invalid candidate partial-path limit"
        )
    if not 1 <= row["reference_nodes_limit"] <= MAXIMUM_REFERENCE_NODES:
        raise AuditAnalysisError(f"line {line_number}: invalid reference node limit")
    if not 1 <= row["reference_depth_limit"] <= MAXIMUM_REFERENCE_TURN_DEPTH:
        raise AuditAnalysisError(f"line {line_number}: invalid reference depth limit")
    if row["diagnostic_root_actions"] > row["candidate_max_actions"]:
        raise AuditAnalysisError(
            f"line {line_number}: root actions exceed candidate_max_actions"
        )
    if row["diagnostic_root_exhaustive"] != (
        row["diagnostic_root_truncations"] == 0
    ):
        raise AuditAnalysisError(
            f"line {line_number}: diagnostic root exhaustiveness contradicts truncation"
        )
    for exact_field, boundary_field in (
        ("actual_action_retained_ordinal", "actual_boundary_retained_ordinal"),
        ("candidate_action_retained_ordinal", "candidate_boundary_retained_ordinal"),
        ("reference_action_retained_ordinal", "reference_boundary_retained_ordinal"),
    ):
        if row[exact_field] >= 0 and row[exact_field] != row[boundary_field]:
            raise AuditAnalysisError(
                f"line {line_number}: exact retention contradicts boundary retention"
            )
    initial_matches = (
        (
            "actual_boundary_retained_ordinal",
            "actual_initial_eval_score",
            "actual_initial_eval_rank",
        ),
        (
            "candidate_boundary_retained_ordinal",
            "candidate_initial_eval_score",
            "candidate_initial_eval_rank",
        ),
        (
            "reference_boundary_retained_ordinal",
            "reference_initial_eval_score",
            "reference_initial_eval_rank",
        ),
    )
    for ordinal_field, score_field, rank_field in initial_matches:
        retained = row[ordinal_field] >= 0
        if retained:
            if (
                type(row[score_field]) is not int
                or not 1 <= row[rank_field] <= row["diagnostic_root_actions"]
            ):
                raise AuditAnalysisError(
                    f"line {line_number}: {score_field}/{rank_field} contradict retention"
                )
        elif row[score_field] is not None or row[rank_field] != -1:
            raise AuditAnalysisError(
                f"line {line_number}: {score_field}/{rank_field} contradict absent boundary"
            )
        is_best = row[ordinal_field] == row["initial_eval_best_retained_ordinal"]
        if retained and (row[rank_field] == 1) != is_best:
            raise AuditAnalysisError(
                f"line {line_number}: {rank_field} contradicts initial best ordinal"
            )
        if is_best and row[score_field] != row["initial_eval_best_score"]:
            raise AuditAnalysisError(
                f"line {line_number}: {score_field} contradicts initial best score"
            )
    for left_index, (left_ordinal, left_score, left_rank) in enumerate(
        initial_matches
    ):
        for right_ordinal, right_score, right_rank in initial_matches[
            left_index + 1 :
        ]:
            if (
                row[left_ordinal] >= 0
                and row[left_ordinal] == row[right_ordinal]
                and (
                    row[left_score] != row[right_score]
                    or row[left_rank] != row[right_rank]
                )
            ):
                raise AuditAnalysisError(
                    f"line {line_number}: identical retained boundaries have "
                    "different initial evaluations"
                )
            if (
                row[left_ordinal] >= 0
                and row[right_ordinal] >= 0
                and row[left_ordinal] != row[right_ordinal]
                and row[left_rank] == row[right_rank]
            ):
                raise AuditAnalysisError(
                    f"line {line_number}: different retained boundaries share one rank"
                )
    action_retention = (
        (
            "actual_action",
            "actual_action_retained_ordinal",
            "actual_boundary_retained_ordinal",
        ),
        (
            "candidate_action",
            "candidate_action_retained_ordinal",
            "candidate_boundary_retained_ordinal",
        ),
        (
            "reference_action",
            "reference_action_retained_ordinal",
            "reference_boundary_retained_ordinal",
        ),
    )
    for action_field, exact_field, _ in action_retention:
        action_is_initial_best = (
            row[action_field] == row["initial_eval_best_action"]
        )
        exact_is_initial_best = (
            row[exact_field] == row["initial_eval_best_retained_ordinal"]
        )
        if action_is_initial_best != exact_is_initial_best:
            raise AuditAnalysisError(
                f"line {line_number}: {action_field} contradicts initial-best retention"
            )
    for left_index, left in enumerate(action_retention):
        for right in action_retention[left_index + 1 :]:
            left_action, left_exact, left_boundary = left
            right_action, right_exact, right_boundary = right
            if row[left_action] == row[right_action] and (
                row[left_exact] != row[right_exact]
                or row[left_boundary] != row[right_boundary]
            ):
                raise AuditAnalysisError(
                    f"line {line_number}: identical actions have different retention"
                )
            if (
                row[left_exact] >= 0
                and row[left_exact] == row[right_exact]
                and row[left_action] != row[right_action]
            ):
                raise AuditAnalysisError(
                    f"line {line_number}: one exact ordinal identifies different actions"
                )
    expected_change_assessable = row["candidate_boundary_retained_ordinal"] >= 0
    expected_changed = (
        expected_change_assessable
        and row["candidate_boundary_retained_ordinal"]
        != row["initial_eval_best_retained_ordinal"]
    )
    if (
        row["candidate_bfm_change_assessable"] != expected_change_assessable
        or row["candidate_bfm_changed_from_initial_best"] != expected_changed
    ):
        raise AuditAnalysisError(
            f"line {line_number}: candidate BFM initial-best change is inconsistent"
        )
    for ordinal_field, tactical_field in (
        ("actual_boundary_retained_ordinal", "actual_retained_tactical_class"),
        ("reference_boundary_retained_ordinal", "reference_retained_tactical_class"),
    ):
        if row[tactical_field] not in TACTICAL_CLASSES:
            raise AuditAnalysisError(
                f"line {line_number}: invalid {tactical_field}"
            )
        retained = row[ordinal_field] >= 0
        if retained == (row[tactical_field] == "not-retained"):
            raise AuditAnalysisError(
                f"line {line_number}: {tactical_field} contradicts its ordinal"
            )
    if (
        row["actual_boundary_retained_ordinal"]
        == row["reference_boundary_retained_ordinal"]
        and row["actual_retained_tactical_class"]
        != row["reference_retained_tactical_class"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: identical boundaries have different tactical classes"
        )
    if row["candidate_player"] not in (0, 1) or row["winner"] not in (0, 1):
        raise AuditAnalysisError(f"line {line_number}: invalid player identity")
    mover_sign = 1 if row["candidate_player"] == 0 else -1
    tactical_scores = {
        "immediate-win": mover_sign * 999_999,
        "forced-cutoff": mover_sign * 999_998,
        "opponent-immediate-win": -mover_sign * 999_998,
        "terminal-loss": -mover_sign * 999_999,
    }
    for tactical_field, score_field in (
        ("actual_retained_tactical_class", "actual_initial_eval_score"),
        ("reference_retained_tactical_class", "reference_initial_eval_score"),
    ):
        tactical = row[tactical_field]
        score = row[score_field]
        if tactical in tactical_scores and score != tactical_scores[tactical]:
            raise AuditAnalysisError(
                f"line {line_number}: {score_field} contradicts tactical proof class"
            )
        if tactical == "safe-handoff" and (
            score is None or abs(score) > MAXIMUM_INITIAL_HEURISTIC_SCORE
        ):
            raise AuditAnalysisError(
                f"line {line_number}: {score_field} contradicts safe-handoff class"
            )
    expected_color = "player_one" if row["candidate_player"] == 0 else "player_two"
    expected_result = "win" if row["candidate_player"] == row["winner"] else "loss"
    if row["color"] != expected_color or row["result"] != expected_result:
        raise AuditAnalysisError(
            f"line {line_number}: color/result contradict player identities"
        )
    for field, expected in (
        ("candidate_matches_actual", row["candidate_action"] == row["actual_action"]),
        ("reference_matches_actual", row["reference_action"] == row["actual_action"]),
        ("candidate_matches_reference", row["candidate_action"] == row["reference_action"]),
        (
            "reconstructed_deployed_matches_actual",
            row["reconstructed_deployed_action"] == row["actual_action"],
        ),
    ):
        if row[field] != expected:
            raise AuditAnalysisError(f"line {line_number}: {field} is inconsistent")
    if row["replay_correction_valid"] and not row["replay_correction_lookup_found"]:
        raise AuditAnalysisError(
            f"line {line_number}: valid replay correction has no lookup"
        )
    expected_deployed = (
        row["replay_correction_action"]
        if row["replay_correction_valid"]
        else row["candidate_action"]
    )
    if row["reconstructed_deployed_action"] != expected_deployed:
        raise AuditAnalysisError(
            f"line {line_number}: reconstructed deployed action is inconsistent"
        )
    if row["replay_correction_valid"] != bool(row["replay_correction_action"]):
        raise AuditAnalysisError(
            f"line {line_number}: replay correction action is inconsistent"
        )
    if row["replay_correction_lookup_found"] != bool(
        row["replay_correction_lookup_action"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: replay lookup action is inconsistent"
        )
    expected_candidate_limit = (
        row["candidate_first_time_limit_ms"]
        if row["own_decision_index"] == 0
        else row["candidate_later_time_limit_ms"]
    )
    expected_reference_limit = (
        row["reference_first_time_limit_ms"]
        if row["own_decision_index"] == 0
        else row["reference_later_time_limit_ms"]
    )
    if (
        row["candidate_time_limit_ms"] != expected_candidate_limit
        or row["reference_time_limit_ms"] != expected_reference_limit
    ):
        raise AuditAnalysisError(f"line {line_number}: per-decision clock is inconsistent")
    if row["codingame_clock_mode"] and (
        row["candidate_first_time_limit_ms"] != 800
        or row["candidate_later_time_limit_ms"] != 165
        or row["reference_first_time_limit_ms"] != 800
        or row["reference_later_time_limit_ms"] != 165
        or row["candidate_work_limit"] != MAXIMUM_CANDIDATE_WORK
        or row["reference_nodes_limit"] != MAXIMUM_REFERENCE_NODES
    ):
        raise AuditAnalysisError(
            f"line {line_number}: CodinGame clock mode configuration is inconsistent"
        )
    if row["candidate_work"] > row["candidate_work_limit"]:
        raise AuditAnalysisError(f"line {line_number}: candidate work exceeds its limit")
    if (
        row["candidate_deadline_reached"] or row["candidate_node_cap_reached"]
    ) and not row["candidate_budget_exhausted"]:
        raise AuditAnalysisError(
            f"line {line_number}: candidate pressure flags contradict budget exhaustion"
        )
    if (
        row["candidate_work"]
        != row["candidate_generator_partial_paths"]
        + row["candidate_child_evaluations"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: candidate work counters are inconsistent"
        )
    if row["candidate_tree_nodes"] > row["candidate_tree_node_limit"]:
        raise AuditAnalysisError(
            f"line {line_number}: candidate tree nodes exceed their limit"
        )
    if row["candidate_expansions"] > row["candidate_tree_nodes"]:
        raise AuditAnalysisError(
            f"line {line_number}: candidate expansions exceed tree nodes"
        )
    if (
        row["candidate_generator_duplicates"] > row["candidate_completed_actions"]
        or row["candidate_tactical_actions"] > row["candidate_completed_actions"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: candidate generator counters are inconsistent"
        )
    diagnostic_partial_limit = min(
        row["candidate_work_limit"], row["candidate_max_partial_paths"]
    )
    if row["diagnostic_root_partial_paths"] > diagnostic_partial_limit:
        raise AuditAnalysisError(
            f"line {line_number}: diagnostic root partial paths exceed their limit"
        )
    if (
        row["diagnostic_root_duplicates"] > row["diagnostic_root_completed_actions"]
        or row["diagnostic_root_actions"] + row["diagnostic_root_duplicates"]
        > row["diagnostic_root_completed_actions"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: diagnostic root counters are inconsistent"
        )
    if row["reference_nodes"] > row["reference_nodes_limit"]:
        raise AuditAnalysisError(
            f"line {line_number}: reference nodes exceed their limit"
        )
    if (
        row["reference_completed_turn_depth"]
        > row["reference_attempted_turn_depth"]
        or row["reference_attempted_turn_depth"] > row["reference_depth_limit"]
    ):
        raise AuditAnalysisError(
            f"line {line_number}: reference depth counters are inconsistent"
        )
    if (
        row["candidate_exploration"] < 0.0
        or row["candidate_final_visit_weight"] < 0.0
        or row["candidate_elapsed_ms"] < 0.0
        or row["reference_elapsed_ms"] < 0.0
        or not 0 <= row["value_blend_percent"] <= 100
        or not 0 <= row["teacher_residual_percent"] <= 100
    ):
        raise AuditAnalysisError(f"line {line_number}: invalid bounded configuration")


def _validate_dataset_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    if not rows:
        raise AuditAnalysisError("audit contains no decision rows")
    provenance: dict[str, str] | None = None
    configuration: dict[str, Any] | None = None
    game_state: dict[str, dict[str, Any]] = {}
    closed_games: set[str] = set()
    current_game: str | None = None
    seen_decisions: set[tuple[str, int]] = set()

    for line_number, row in enumerate(rows, 1):
        _validate_row(row, line_number)
        if provenance is None:
            provenance = row["input_provenance"]
        elif row["input_provenance"] != provenance:
            raise AuditAnalysisError(
                f"line {line_number}: input provenance differs within the audit"
            )
        row_configuration = {field: row[field] for field in RUN_CONFIGURATION_FIELDS}
        if configuration is None:
            configuration = row_configuration
        elif row_configuration != configuration:
            raise AuditAnalysisError(
                f"line {line_number}: run configuration differs within the audit"
            )

        game_id = row["game_id"]
        if game_id != current_game:
            if current_game is not None:
                closed_games.add(current_game)
            if game_id in closed_games:
                raise AuditAnalysisError(
                    f"line {line_number}: game {game_id} is not contiguous"
                )
            current_game = game_id
        decision_key = (game_id, row["own_decision_index"])
        if decision_key in seen_decisions:
            raise AuditAnalysisError(f"line {line_number}: duplicate decision key")
        seen_decisions.add(decision_key)

        prefix_actions = (
            tuple(row["transcript_prefix"].split("/"))
            if row["transcript_prefix"]
            else ()
        )
        if len(prefix_actions) != row["turn_index"]:
            raise AuditAnalysisError(
                f"line {line_number}: transcript prefix length contradicts turn_index"
            )
        expected_turn = 2 * row["own_decision_index"] + row["candidate_player"]
        if row["turn_index"] != expected_turn:
            raise AuditAnalysisError(
                f"line {line_number}: turn_index contradicts own decision index/color"
            )

        state = game_state.get(game_id)
        if state is None:
            if row["own_decision_index"] != 0:
                raise AuditAnalysisError(
                    f"line {line_number}: first decision index for game is not zero"
                )
            state = {
                "candidate_player": row["candidate_player"],
                "winner": row["winner"],
                "color": row["color"],
                "result": row["result"],
                "last_index": -1,
                "last_prefix": (),
                "last_turn": None,
                "last_action": None,
                "state_ids": set(),
            }
            game_state[game_id] = state
        for field in ("candidate_player", "winner", "color", "result"):
            if row[field] != state[field]:
                raise AuditAnalysisError(
                    f"line {line_number}: {field} differs within game {game_id}"
                )
        if row["own_decision_index"] != state["last_index"] + 1:
            raise AuditAnalysisError(
                f"line {line_number}: decision indices are not contiguous"
            )
        if row["state_id"] in state["state_ids"]:
            raise AuditAnalysisError(f"line {line_number}: duplicate state_id within game")
        state["state_ids"].add(row["state_id"])
        if state["last_turn"] is not None:
            previous_prefix = state["last_prefix"]
            if prefix_actions[: len(previous_prefix)] != previous_prefix:
                raise AuditAnalysisError(
                    f"line {line_number}: transcript prefix does not extend prior prefix"
                )
            if prefix_actions[state["last_turn"]] != state["last_action"]:
                raise AuditAnalysisError(
                    f"line {line_number}: transcript prefix contradicts prior actual action"
                )
        state["last_index"] = row["own_decision_index"]
        state["last_prefix"] = prefix_actions
        state["last_turn"] = row["turn_index"]
        state["last_action"] = row["actual_action"]

    assert provenance is not None
    assert configuration is not None
    return provenance, configuration


def _detect_format(path: pathlib.Path, text: str, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".tab"):
        return "tsv"
    if suffix in (".jsonl", ".ndjson", ".json"):
        return "jsonl"
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return "jsonl" if first_line.lstrip().startswith("{") else "tsv"


def load_audit(path: pathlib.Path, input_format: str = "auto") -> dict[str, Any]:
    if input_format not in ("auto", "jsonl", "tsv"):
        raise AuditAnalysisError(f"unsupported input format: {input_format}")
    if not path.is_file():
        raise AuditAnalysisError(f"audit input is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_INPUT_BYTES:
        raise AuditAnalysisError(f"audit input size is outside the supported bounds: {size}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditAnalysisError("audit input is not strict UTF-8") from error
    if "\x00" in text:
        raise AuditAnalysisError("audit input contains a NUL byte")
    detected_format = _detect_format(path, text, input_format)
    rows = _parse_json_lines(text) if detected_format == "jsonl" else _parse_tsv(text)
    provenance, configuration = _validate_dataset_rows(rows)
    return {
        "rows": rows,
        "provenance": provenance,
        "configuration": configuration,
        "source_name": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "input_format": detected_format,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise AuditAnalysisError("value cannot be encoded as canonical JSON") from error


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise AuditAnalysisError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise AuditAnalysisError(f"{context} must be a nonnegative integer")
    return value


def _bounded_printable_string(
    value: Any, context: str, *, maximum: int = 512, allow_empty: bool = False
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value) > maximum
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise AuditAnalysisError(f"{context} is not a bounded printable string")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AuditAnalysisError(f"{context} is not a lowercase SHA-256")
    return value


def _validate_content_addressed_reference(
    value: Any, expected_sha256: str, context: str, *, suffix: str = ".json"
) -> str:
    reference = _bounded_printable_string(value, context, maximum=4096)
    name = pathlib.PurePath(reference).name
    if name != f"{expected_sha256}{suffix}":
        raise AuditAnalysisError(
            f"{context} does not name its content-addressed JSON record"
        )
    return reference


def _validate_clean_arena_record(
    record: dict[str, Any],
    *,
    agent_id: int,
    submission_id: int,
    source_sha256: str,
    leaderboard_frozen_at: str,
) -> dict[str, Any] | None:
    game_id = _positive_integer(record.get("game_id"), "arena game_id")
    if record.get("schema") != ARENA_GAME_SCHEMA_VERSION:
        raise AuditAnalysisError(f"arena game {game_id} has an unsupported schema")
    if record.get("purpose") != ARENA_PURPOSE:
        raise AuditAnalysisError(f"arena game {game_id} has an invalid purpose binding")
    if record.get("source_sha256") != source_sha256:
        raise AuditAnalysisError(f"arena game {game_id} has a source hash mismatch")
    status = _bounded_printable_string(
        record.get("status"), f"arena game {game_id} status", maximum=64
    )
    focus = record.get("focus")
    if not isinstance(focus, dict) or focus.get("agent_id") != agent_id:
        raise AuditAnalysisError(f"arena game {game_id} contradicts the focus agent")
    if status == "accepted" and focus.get("submission_id") != submission_id:
        raise AuditAnalysisError(
            f"arena game {game_id} contradicts the asserted submission"
        )
    if status != "accepted":
        return None

    candidate_player = focus.get("player_id")
    if isinstance(candidate_player, bool) or candidate_player not in (0, 1):
        raise AuditAnalysisError(f"arena game {game_id} has an invalid focus player")
    expected_color = f"player-{candidate_player}"
    if focus.get("color") != expected_color:
        raise AuditAnalysisError(f"arena game {game_id} has an inconsistent focus color")

    opponent = record.get("opponent")
    if not isinstance(opponent, dict):
        raise AuditAnalysisError(f"arena game {game_id} omits opponent metadata")
    opponent_id = _positive_integer(
        opponent.get("agent_id"), f"arena game {game_id} opponent agent_id"
    )
    opponent_player = opponent.get("player_id")
    if (
        opponent_id == agent_id
        or type(opponent_player) is not int
        or opponent_player != 1 - candidate_player
    ):
        raise AuditAnalysisError(
            f"arena game {game_id} has inconsistent opponent identity"
        )
    opponent_name = _bounded_printable_string(
        opponent.get("name"),
        f"arena game {game_id} opponent name",
        maximum=256,
        allow_empty=True,
    )
    frozen_rank = opponent.get("frozen_rank")
    if frozen_rank is not None:
        frozen_rank = _positive_integer(
            frozen_rank, f"arena game {game_id} frozen opponent rank"
        )

    if record.get("leaderboard_frozen_at_utc") != leaderboard_frozen_at:
        raise AuditAnalysisError(
            f"arena game {game_id} contradicts the leaderboard freeze"
        )
    operational = record.get("operational")
    if not isinstance(operational, dict):
        raise AuditAnalysisError(f"arena game {game_id} omits operational metadata")
    classification = operational.get("classification")
    if classification not in ("clean", "operationally-terminated"):
        raise AuditAnalysisError(
            f"arena game {game_id} has an invalid operational classification"
        )
    focus_status = operational.get("focus_status")
    opponent_status = operational.get("opponent_status")
    if (
        focus_status not in ARENA_OPERATIONAL_STATUSES
        or opponent_status not in ARENA_OPERATIONAL_STATUSES
        or (classification == "clean") != (
            focus_status == "ok" and opponent_status == "ok"
        )
    ):
        raise AuditAnalysisError(
            f"arena game {game_id} classification contradicts player status"
        )
    if classification != "clean":
        return None

    outcome = record.get("outcome")
    if (
        not isinstance(outcome, dict)
        or type(outcome.get("winner_player_id")) is not int
        or outcome.get("winner_player_id") not in (0, 1)
    ):
        raise AuditAnalysisError(f"arena game {game_id} has an invalid outcome")
    winner = outcome["winner_player_id"]
    expected_winner_agent = agent_id if winner == candidate_player else opponent_id
    if outcome.get("winner_agent_id") != expected_winner_agent:
        raise AuditAnalysisError(
            f"arena game {game_id} winner agent contradicts its player identity"
        )
    expected_result = "win" if winner == candidate_player else "loss"
    if focus.get("result") != expected_result:
        raise AuditAnalysisError(f"arena game {game_id} has an inconsistent result")

    replay = record.get("replay")
    validation = replay.get("rules_validation") if isinstance(replay, dict) else None
    if not isinstance(validation, dict) or validation.get("status") != "terminal-valid":
        raise AuditAnalysisError(
            f"arena game {game_id} is clean but not rule-terminal"
        )
    if validation.get("terminal_winner_player_id") != winner:
        raise AuditAnalysisError(
            f"arena game {game_id} replay winner contradicts its outcome"
        )
    transcript = replay.get("valid_transcript")
    if (
        not isinstance(transcript, str)
        or not transcript
        or len(transcript) > 4 * 1024 * 1024
    ):
        raise AuditAnalysisError(f"arena game {game_id} has an invalid transcript")
    turns = tuple(transcript.split("/"))
    if any(ACTION_PATTERN.fullmatch(action) is None for action in turns):
        raise AuditAnalysisError(f"arena game {game_id} has an invalid transcript")
    valid_turns = validation.get("valid_turns")
    if not isinstance(valid_turns, list) or len(valid_turns) != len(turns):
        raise AuditAnalysisError(
            f"arena game {game_id} terminal validation has inconsistent turns"
        )
    for turn_index, (action, turn) in enumerate(zip(turns, valid_turns)):
        if (
            not isinstance(turn, dict)
            or set(turn) != {"action", "player_id"}
            or turn.get("action") != action
            or type(turn.get("player_id")) is not int
            or turn.get("player_id") != turn_index % 2
        ):
            raise AuditAnalysisError(
                f"arena game {game_id} terminal validation turn {turn_index} is inconsistent"
            )
    if (
        type(validation.get("valid_turn_count")) is not int
        or validation.get("valid_turn_count") != len(turns)
    ):
        raise AuditAnalysisError(
            f"arena game {game_id} terminal validation count is inconsistent"
        )
    if (
        replay.get("valid_turns") != valid_turns
        or replay.get("observed_turns") != valid_turns
        or replay.get("observed_transcript") != transcript
    ):
        raise AuditAnalysisError(
            f"arena game {game_id} clean replay transcript representations differ"
        )
    agents = replay.get("agents")
    if not isinstance(agents, list) or len(agents) != 2:
        raise AuditAnalysisError(f"arena game {game_id} replay agents are invalid")
    expected_agents = {
        candidate_player: agent_id,
        1 - candidate_player: opponent_id,
    }
    for replay_agent in agents:
        if not isinstance(replay_agent, dict):
            raise AuditAnalysisError(f"arena game {game_id} replay agents are invalid")
        player_id = replay_agent.get("player_id")
        if (
            type(player_id) is not int
            or player_id not in (0, 1)
            or replay_agent.get("agent_id") != expected_agents[player_id]
        ):
            raise AuditAnalysisError(
                f"arena game {game_id} replay agent identity is inconsistent"
            )
    if {agent["player_id"] for agent in agents} != {0, 1}:
        raise AuditAnalysisError(f"arena game {game_id} repeats a replay player")

    return {
        "game_id": str(game_id),
        "candidate_player": candidate_player,
        "winner": winner,
        "result": expected_result,
        "color": "player_one" if candidate_player == 0 else "player_two",
        "opponent_agent_id": opponent_id,
        "opponent_name": opponent_name,
        "opponent_frozen_rank": frozen_rank,
        "turns": turns,
        "expected_decisions": len(range(candidate_player, len(turns), 2)),
    }


def load_arena_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Load one explicitly named, self-contained collector manifest.

    The embedded records are trusted only after their canonical hashes and
    provenance bindings validate. Cross-reference paths are checked as
    content-addressed names but are deliberately not opened.
    """

    if not path.is_file():
        raise AuditAnalysisError(f"arena manifest is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_INPUT_BYTES:
        raise AuditAnalysisError(
            f"arena manifest size is outside the supported bounds: {size}"
        )
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditAnalysisError("arena manifest is not strict UTF-8") from error
    if "\x00" in text:
        raise AuditAnalysisError("arena manifest contains a NUL byte")
    payload = _strict_json_loads(text, "arena manifest")
    if not isinstance(payload, dict):
        raise AuditAnalysisError("arena manifest must be a JSON object")
    if _canonical_json_bytes(payload) != raw:
        raise AuditAnalysisError("arena manifest is not canonical JSON")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    filename_is_content_addressed = SHA256_PATTERN.fullmatch(path.stem) is not None
    if filename_is_content_addressed and (
        path.suffix != ".json" or path.stem != manifest_sha256
    ):
        raise AuditAnalysisError("arena manifest filename hash does not match its content")
    if payload.get("schema") != ARENA_BATCH_SCHEMA_VERSION:
        raise AuditAnalysisError("arena manifest has an unsupported schema")
    if payload.get("purpose") != ARENA_PURPOSE:
        raise AuditAnalysisError("arena manifest has an invalid purpose binding")

    collector_sha256 = _sha256(
        payload.get("collector_sha256"), "arena manifest collector_sha256"
    )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AuditAnalysisError("arena manifest run_id is invalid")
    binding = payload.get("binding")
    if not isinstance(binding, dict) or binding.get("schema") != ARENA_BINDING_SCHEMA_VERSION:
        raise AuditAnalysisError("arena manifest has an invalid source binding")
    if (
        binding.get("purpose") != ARENA_PURPOSE
        or binding.get("collector_sha256") != collector_sha256
        or binding.get("run_id") != run_id
    ):
        raise AuditAnalysisError("arena manifest source binding contradicts the manifest")
    agent_id = _positive_integer(binding.get("agent_id"), "arena binding agent_id")
    submission_id = _positive_integer(
        binding.get("asserted_submission_id"),
        "arena binding asserted_submission_id",
    )
    repository_commit = binding.get("repository_commit")
    if (
        not isinstance(repository_commit, str)
        or GIT_OBJECT_PATTERN.fullmatch(repository_commit) is None
    ):
        raise AuditAnalysisError("arena binding repository_commit is invalid")
    source = binding.get("source")
    if not isinstance(source, dict):
        raise AuditAnalysisError("arena binding omits source metadata")
    source_sha256 = _sha256(source.get("sha256"), "arena binding source sha256")
    _validate_content_addressed_reference(
        source.get("archived_path"),
        source_sha256,
        "arena binding archived source path",
        suffix=".source",
    )
    _bounded_printable_string(
        source.get("input_path"), "arena binding input source path", maximum=4096
    )
    if (
        source.get("encoding") != "utf-8"
        or isinstance(source.get("bytes"), bool)
        or type(source.get("bytes")) is not int
        or source.get("bytes") < 0
        or isinstance(source.get("characters"), bool)
        or type(source.get("characters")) is not int
        or source.get("characters") < 0
        or source.get("characters") > source.get("bytes")
    ):
        raise AuditAnalysisError("arena binding source metadata is inconsistent")

    exclusion = payload.get("exclusion_registry")
    if not isinstance(exclusion, dict):
        raise AuditAnalysisError("arena manifest omits its exclusion registry binding")
    exclusion_sha256 = _sha256(
        exclusion.get("sha256"), "arena manifest exclusion registry sha256"
    )
    _validate_content_addressed_reference(
        exclusion.get("path"),
        exclusion_sha256,
        "arena manifest exclusion registry path",
    )
    leaderboard = payload.get("leaderboard_snapshot")
    if not isinstance(leaderboard, dict):
        raise AuditAnalysisError("arena manifest omits its leaderboard snapshot")
    leaderboard_frozen_at = _bounded_printable_string(
        leaderboard.get("frozen_at_utc"),
        "arena manifest leaderboard freeze",
        maximum=64,
    )
    _sha256(
        leaderboard.get("normalized_sha256"),
        "arena manifest leaderboard normalized_sha256",
    )
    _sha256(
        leaderboard.get("raw_sha256"),
        "arena manifest leaderboard raw_sha256",
    )
    window = payload.get("window_snapshot")
    if not isinstance(window, dict):
        raise AuditAnalysisError("arena manifest omits its battle-window snapshot")
    _sha256(
        window.get("normalized_sha256"),
        "arena manifest battle-window normalized_sha256",
    )
    _sha256(
        window.get("raw_sha256"),
        "arena manifest battle-window raw_sha256",
    )

    stored_games = payload.get("games")
    if not isinstance(stored_games, list):
        raise AuditAnalysisError("arena manifest games must be a list")
    seen_game_ids: set[int] = set()
    statuses: collections.Counter[str] = collections.Counter()
    accepted = 0
    focus_failures = 0
    opponent_failures = 0
    clean_games: dict[str, dict[str, Any]] = {}
    opponent_snapshots: dict[int, tuple[str, int | None]] = {}
    for index, stored in enumerate(stored_games):
        if not isinstance(stored, dict) or not isinstance(stored.get("record"), dict):
            raise AuditAnalysisError(
                f"arena manifest game binding {index} is not an object"
            )
        record = stored["record"]
        record_sha256 = _sha256(
            stored.get("record_sha256"),
            f"arena manifest game binding {index} record_sha256",
        )
        if hashlib.sha256(_canonical_json_bytes(record)).hexdigest() != record_sha256:
            raise AuditAnalysisError(
                f"arena manifest game binding {index} record hash mismatch"
            )
        _validate_content_addressed_reference(
            stored.get("record_path"),
            record_sha256,
            f"arena manifest game binding {index} record_path",
        )
        game_id = _positive_integer(record.get("game_id"), "arena game_id")
        if game_id in seen_game_ids:
            raise AuditAnalysisError(f"arena manifest repeats game {game_id}")
        seen_game_ids.add(game_id)
        status = record.get("status")
        if not isinstance(status, str):
            raise AuditAnalysisError(f"arena game {game_id} has an invalid status")
        statuses[status] += 1
        if status == "accepted":
            accepted += 1
            operational = record.get("operational")
            if isinstance(operational, dict):
                focus_failures += operational.get("focus_status") != "ok"
                opponent_failures += operational.get("opponent_status") != "ok"
        clean = _validate_clean_arena_record(
            record,
            agent_id=agent_id,
            submission_id=submission_id,
            source_sha256=source_sha256,
            leaderboard_frozen_at=leaderboard_frozen_at,
        )
        if clean is not None:
            game_key = clean["game_id"]
            clean_games[game_key] = clean
            opponent_identity = clean["opponent_agent_id"]
            snapshot = (clean["opponent_name"], clean["opponent_frozen_rank"])
            previous = opponent_snapshots.setdefault(opponent_identity, snapshot)
            if previous != snapshot:
                raise AuditAnalysisError(
                    f"arena opponent {opponent_identity} has inconsistent frozen metadata"
                )

    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise AuditAnalysisError("arena manifest omits coverage metadata")
    expected_games = _positive_integer(
        coverage.get("expected_games"), "arena coverage expected_games"
    )
    for field in (
        "accepted_games",
        "battle_window_games",
        "clean_rule_terminal_games",
        "focus_operational_failures",
        "opponent_operational_failures",
    ):
        _nonnegative_integer(coverage.get(field), f"arena coverage {field}")
    raw_status_counts = coverage.get("status_counts")
    if not isinstance(raw_status_counts, dict):
        raise AuditAnalysisError("arena coverage status_counts must be an object")
    for status, count in raw_status_counts.items():
        _bounded_printable_string(status, "arena coverage status", maximum=64)
        _nonnegative_integer(count, f"arena coverage status count for {status}")
    expected_status_counts = dict(sorted(statuses.items()))
    if (
        coverage.get("battle_window_games") != len(stored_games)
        or coverage.get("accepted_games") != accepted
        or coverage.get("clean_rule_terminal_games") != len(clean_games)
        or coverage.get("focus_operational_failures") != focus_failures
        or coverage.get("opponent_operational_failures") != opponent_failures
        or raw_status_counts != expected_status_counts
    ):
        raise AuditAnalysisError("arena manifest coverage counters are inconsistent")
    fully_accounted = len(stored_games) >= expected_games and all(
        status in {"accepted", "excluded-protected", "already-known-local"}
        for status in statuses
    )
    if type(coverage.get("full_window_accounted")) is not bool or (
        coverage["full_window_accounted"] != fully_accounted
    ):
        raise AuditAnalysisError("arena manifest full-window coverage is inconsistent")
    if not fully_accounted:
        raise AuditAnalysisError(
            "arena manifest battle window is not fully accounted"
        )

    expected_provenance = {
        "agent_id": str(agent_id),
        "arena_manifest_sha256": manifest_sha256,
        "asserted_source_sha256": source_sha256,
        "asserted_submission_id": str(submission_id),
        "collector_sha256": collector_sha256,
        "exclusion_registry_sha256": exclusion_sha256,
        "repository_commit": repository_commit,
        "run_id": run_id,
        "source_binding_status": ARENA_SOURCE_BINDING_STATUS,
    }
    expected_game_ids = {
        game_id
        for game_id, game in clean_games.items()
        if game["expected_decisions"] > 0
    }
    return {
        "clean_games": clean_games,
        "expected_audited_game_ids": expected_game_ids,
        "expected_provenance": expected_provenance,
        "filename_is_content_addressed": filename_is_content_addressed,
        "manifest_sha256": manifest_sha256,
        "source_name": path.name,
        "zero_decision_game_ids": sorted(
            set(clean_games) - expected_game_ids, key=int
        ),
    }


def join_arena_manifest(
    dataset: dict[str, Any], arena_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Validate and attach an exact clean-game manifest join to an audit."""

    rows = dataset["rows"]
    expected_provenance = arena_manifest["expected_provenance"]
    by_game: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for line_number, row in enumerate(rows, 1):
        for field in ARENA_PROVENANCE_FIELDS:
            if row["input_provenance"].get(field) != expected_provenance[field]:
                raise AuditAnalysisError(
                    f"line {line_number}: arena provenance field {field} does not match"
                )
        game_id = row["game_id"]
        if POSITIVE_INTEGER_TEXT_PATTERN.fullmatch(game_id) is None:
            raise AuditAnalysisError(
                f"line {line_number}: manifest-joined game_id must be canonical decimal"
            )
        by_game[game_id].append(row)

    actual_game_ids = set(by_game)
    expected_game_ids = arena_manifest["expected_audited_game_ids"]
    if actual_game_ids != expected_game_ids:
        missing = sorted(expected_game_ids - actual_game_ids, key=int)
        unexpected = sorted(actual_game_ids - expected_game_ids, key=int)
        raise AuditAnalysisError(
            "arena manifest audit-game coverage differs; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for game_id in sorted(expected_game_ids, key=int):
        game = arena_manifest["clean_games"][game_id]
        game_rows = by_game[game_id]
        if len(game_rows) != game["expected_decisions"]:
            raise AuditAnalysisError(
                f"arena game {game_id} decision coverage differs: "
                f"expected {game['expected_decisions']}, got {len(game_rows)}"
            )
        for row in game_rows:
            turn_index = row["turn_index"]
            expected_prefix = "/".join(game["turns"][:turn_index])
            if (
                row["candidate_player"] != game["candidate_player"]
                or row["winner"] != game["winner"]
                or row["result"] != game["result"]
                or row["color"] != game["color"]
                or row["transcript_prefix"] != expected_prefix
                or turn_index >= len(game["turns"])
                or row["actual_action"] != game["turns"][turn_index]
            ):
                raise AuditAnalysisError(
                    f"arena game {game_id} audit row contradicts manifest replay context"
                )

    joined = dict(dataset)
    joined["arena_manifest"] = arena_manifest
    return joined


def phase_name(row: dict[str, Any]) -> str:
    index = row["own_decision_index"]
    if index <= 3:
        return "opening_0_3"
    if index <= 11:
        return "midgame_4_11"
    return "late_12_plus"


def clock_phase_name(row: dict[str, Any]) -> str:
    return "first_decision" if row["own_decision_index"] == 0 else "later_decisions"


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _event(rows: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    count = sum(1 for row in rows if predicate(row))
    return {"count": count, "rate": _rate(count, len(rows))}


def _rounded_mean(values: Iterable[int | float]) -> float:
    materialized = list(values)
    return round(sum(materialized) / len(materialized), 3) if materialized else 0.0


def _ordinal_summary(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(row[field] for row in rows if row[field] >= 0)
    missing = len(rows) - len(values)
    if values:
        percentile_index = max(0, math.ceil(0.90 * len(values)) - 1)
        present_ordinals: dict[str, Any] = {
            "min": values[0],
            "median": values[(len(values) - 1) // 2],
            "p90_nearest_rank": values[percentile_index],
            "max": values[-1],
            "mean": _rounded_mean(values),
        }
    else:
        present_ordinals = {}
    return {
        "present": {"count": len(values), "rate": _rate(len(values), len(rows))},
        "missing": {"count": missing, "rate": _rate(missing, len(rows))},
        "present_ordinal": present_ordinals,
    }


def _failure_bucket(row: dict[str, Any]) -> str:
    candidate_matches = row["candidate_matches_actual"]
    reference_matches = row["reference_matches_actual"]
    if candidate_matches and reference_matches:
        return "all_match_actual"
    if candidate_matches:
        return "candidate_only_matches_actual"
    if reference_matches:
        return "rank4_only_matches_actual"
    if row["candidate_matches_reference"]:
        return "both_miss_same_action"
    return "both_miss_split_actions"


def _initial_evaluation_bucket(row: dict[str, Any]) -> str:
    if row["actual_initial_eval_rank"] < 0:
        return "actual_boundary_not_retained"
    if row["candidate_initial_eval_rank"] < 0:
        return "candidate_boundary_not_retained"
    candidate_matches_actual_boundary = (
        row["candidate_boundary_retained_ordinal"]
        == row["actual_boundary_retained_ordinal"]
    )
    if row["actual_initial_eval_rank"] == 1:
        return (
            "initial_and_bfm_match_actual_boundary"
            if candidate_matches_actual_boundary
            else "bfm_overrode_actual_initial_best"
        )
    if candidate_matches_actual_boundary:
        return "bfm_corrected_initial_misranking"
    if row["candidate_initial_eval_rank"] == 1:
        return "initial_misranking_preserved"
    return "bfm_overrode_initial_best_to_other_boundary"


def _compact_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets = collections.Counter(_failure_bucket(row) for row in rows)
    initial_buckets = collections.Counter(
        _initial_evaluation_bucket(row) for row in rows
    )
    return {
        "decisions": len(rows),
        "agreement": {
            "candidate_vs_actual": _event(rows, lambda row: row["candidate_matches_actual"]),
            "rank4_vs_actual": _event(rows, lambda row: row["reference_matches_actual"]),
            "candidate_vs_rank4": _event(rows, lambda row: row["candidate_matches_reference"]),
        },
        "root_missing": {
            "actual_exact": _event(rows, lambda row: row["actual_action_retained_ordinal"] < 0),
            "actual_boundary": _event(rows, lambda row: row["actual_boundary_retained_ordinal"] < 0),
            "candidate_exact": _event(rows, lambda row: row["candidate_action_retained_ordinal"] < 0),
            "candidate_boundary": _event(rows, lambda row: row["candidate_boundary_retained_ordinal"] < 0),
            "rank4_exact": _event(rows, lambda row: row["reference_action_retained_ordinal"] < 0),
            "rank4_boundary": _event(rows, lambda row: row["reference_boundary_retained_ordinal"] < 0),
        },
        "pressure": {
            "candidate_deadline_reached": _event(rows, lambda row: row["candidate_deadline_reached"]),
            "candidate_generator_truncated": _event(rows, lambda row: row["candidate_generator_truncations"] > 0),
            "diagnostic_root_truncated": _event(rows, lambda row: row["diagnostic_root_truncations"] > 0),
            "diagnostic_root_nonexhaustive": _event(rows, lambda row: not row["diagnostic_root_exhaustive"]),
        },
        "initial_evaluation": {
            "actual_is_initial_best": _event(
                rows, lambda row: row["actual_initial_eval_rank"] == 1
            ),
            "candidate_is_initial_best": _event(
                rows, lambda row: row["candidate_initial_eval_rank"] == 1
            ),
            "rank4_is_initial_best": _event(
                rows, lambda row: row["reference_initial_eval_rank"] == 1
            ),
            "candidate_bfm_change_assessable": _event(
                rows, lambda row: row["candidate_bfm_change_assessable"]
            ),
            "candidate_bfm_changed_from_initial_best": _event(
                rows,
                lambda row: row["candidate_bfm_changed_from_initial_best"],
            ),
            "bfm_changed_to_actual_boundary": _event(
                rows,
                lambda row: row["candidate_bfm_changed_from_initial_best"]
                and row["actual_boundary_retained_ordinal"] >= 0
                and row["candidate_boundary_retained_ordinal"]
                == row["actual_boundary_retained_ordinal"],
            ),
            "bfm_changed_away_from_actual_initial_best": _event(
                rows,
                lambda row: row["candidate_bfm_changed_from_initial_best"]
                and row["actual_initial_eval_rank"] == 1,
            ),
        },
        "initial_evaluation_buckets": dict(sorted(initial_buckets.items())),
        "failure_buckets": dict(sorted(buckets.items())),
    }


def _overall_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = _compact_summary(rows)
    summary["agreement"]["reconstructed_deployed_vs_actual"] = _event(
        rows, lambda row: row["reconstructed_deployed_matches_actual"]
    )
    summary["root_coverage"] = {
        "actual": {
            "exact": _ordinal_summary(rows, "actual_action_retained_ordinal"),
            "boundary": _ordinal_summary(rows, "actual_boundary_retained_ordinal"),
        },
        "candidate": {
            "exact": _ordinal_summary(rows, "candidate_action_retained_ordinal"),
            "boundary": _ordinal_summary(rows, "candidate_boundary_retained_ordinal"),
        },
        "rank4": {
            "exact": _ordinal_summary(rows, "reference_action_retained_ordinal"),
            "boundary": _ordinal_summary(rows, "reference_boundary_retained_ordinal"),
        },
        "retained_actions_mean": _rounded_mean(
            row["diagnostic_root_actions"] for row in rows
        ),
    }
    summary["initial_evaluation"]["ranks"] = {
        "actual": _ordinal_summary(rows, "actual_initial_eval_rank"),
        "candidate": _ordinal_summary(rows, "candidate_initial_eval_rank"),
        "rank4": _ordinal_summary(rows, "reference_initial_eval_rank"),
    }
    summary["pressure"].update(
        {
            "candidate_work_budget_exhausted": _event(
                rows, lambda row: row["candidate_budget_exhausted"]
            ),
            "candidate_node_cap_reached": _event(
                rows, lambda row: row["candidate_node_cap_reached"]
            ),
            "rank4_work_budget_exhausted": _event(
                rows, lambda row: row["reference_budget_exhausted"]
            ),
        }
    )
    return summary


def _breakdown(
    rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {name: _compact_summary(grouped[name]) for name in sorted(grouped)}


def _clock_phase_breakdown(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        name: _compact_summary(
            [row for row in rows if clock_phase_name(row) == name]
        )
        for name in ("first_decision", "later_decisions")
    }


def _arena_bucket_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = _compact_summary(rows)
    game_results = _game_result_summary(rows)
    result["games"] = game_results["total"]
    result["game_results"] = game_results
    return result


def _game_result_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    games: dict[str, str] = {}
    for row in rows:
        previous = games.setdefault(row["game_id"], row["result"])
        if previous != row["result"]:
            raise AuditAnalysisError("one game has inconsistent results in a breakdown")
    counts = collections.Counter(games.values())
    return {
        "total": len(games),
        "wins": counts["win"],
        "losses": counts["loss"],
    }


def _arena_breakdowns(
    rows: Sequence[dict[str, Any]], arena_manifest: dict[str, Any]
) -> dict[str, Any]:
    games = arena_manifest["clean_games"]

    def rank(row: dict[str, Any]) -> int | None:
        return games[row["game_id"]]["opponent_frozen_rank"]

    # The top-N cohorts are intentionally cumulative. This answers the useful
    # arena questions "against the top 5/10/20" without relabeling rank 6 as a
    # rank-1-to-10 opponent. Rank 21+ and unranked remain disjoint tails.
    cohort_predicates: tuple[
        tuple[str, Callable[[int | None], bool]], ...
    ] = (
        ("rank_1_5", lambda value: value is not None and 1 <= value <= 5),
        ("rank_1_10", lambda value: value is not None and 1 <= value <= 10),
        ("rank_1_20", lambda value: value is not None and 1 <= value <= 20),
        ("rank_21_plus", lambda value: value is not None and value >= 21),
        ("unranked", lambda value: value is None),
    )
    by_rank = {
        name: _arena_bucket_summary(
            [row for row in rows if predicate(rank(row))]
        )
        for name, predicate in cohort_predicates
    }

    opponent_rows: dict[tuple[int, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    opponent_game_ids: dict[tuple[int, str], set[str]] = collections.defaultdict(set)
    for row in rows:
        game = games[row["game_id"]]
        key = (game["opponent_agent_id"], game["opponent_name"])
        opponent_rows[key].append(row)
        opponent_game_ids[key].add(row["game_id"])
    by_opponent = []
    for (agent_id, name), selected in sorted(
        opponent_rows.items(), key=lambda entry: (entry[0][1].casefold(), entry[0][0])
    ):
        game_count = len(opponent_game_ids[(agent_id, name)])
        if not name or game_count < MINIMUM_NAMED_OPPONENT_GAMES:
            continue
        by_opponent.append(
            {
                "agent_id": agent_id,
                "name": name,
                "games": game_count,
                "game_results": _game_result_summary(selected),
                "summary": _compact_summary(selected),
            }
        )
    return {
        "frozen_opponent_rank_cohorts": by_rank,
        "named_opponents": {
            "minimum_games": MINIMUM_NAMED_OPPONENT_GAMES,
            "buckets": by_opponent,
        },
    }


def _divergence_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": row["game_id"],
        "result": row["result"],
        "color": row["color"],
        "phase": phase_name(row),
        "turn_index": row["turn_index"],
        "own_decision_index": row["own_decision_index"],
        "state_id": row["state_id"],
        "failure_bucket": _failure_bucket(row),
        "actual_action": row["actual_action"],
        "candidate_action": row["candidate_action"],
        "rank4_action": row["reference_action"],
        "actual_exact_ordinal": row["actual_action_retained_ordinal"],
        "actual_boundary_ordinal": row["actual_boundary_retained_ordinal"],
        "candidate_exact_ordinal": row["candidate_action_retained_ordinal"],
        "candidate_boundary_ordinal": row["candidate_boundary_retained_ordinal"],
        "rank4_exact_ordinal": row["reference_action_retained_ordinal"],
        "rank4_boundary_ordinal": row["reference_boundary_retained_ordinal"],
        "initial_evaluation_bucket": _initial_evaluation_bucket(row),
        "initial_eval_best_action": row["initial_eval_best_action"],
        "initial_eval_best_score": row["initial_eval_best_score"],
        "initial_eval_best_retained_ordinal": row[
            "initial_eval_best_retained_ordinal"
        ],
        "actual_initial_eval_score": row["actual_initial_eval_score"],
        "actual_initial_eval_rank": row["actual_initial_eval_rank"],
        "candidate_initial_eval_score": row["candidate_initial_eval_score"],
        "candidate_initial_eval_rank": row["candidate_initial_eval_rank"],
        "rank4_initial_eval_score": row["reference_initial_eval_score"],
        "rank4_initial_eval_rank": row["reference_initial_eval_rank"],
        "candidate_bfm_changed_from_initial_best": row[
            "candidate_bfm_changed_from_initial_best"
        ],
        "diagnostic_root_actions": row["diagnostic_root_actions"],
        "candidate_deadline_reached": row["candidate_deadline_reached"],
        "candidate_generator_truncations": row["candidate_generator_truncations"],
        "diagnostic_root_truncations": row["diagnostic_root_truncations"],
    }


def analyze_dataset(dataset: dict[str, Any], label: str = "audit") -> dict[str, Any]:
    rows = dataset["rows"]
    games: dict[str, dict[str, Any]] = {}
    first_divergences: dict[str, dict[str, Any]] = {}
    for row in rows:
        games.setdefault(
            row["game_id"], {"result": row["result"], "color": row["color"]}
        )
        if row["game_id"] not in first_divergences and _failure_bucket(row) != "all_match_actual":
            first_divergences[row["game_id"]] = _divergence_detail(row)
    game_results = collections.Counter(entry["result"] for entry in games.values())
    decision_results = collections.Counter(row["result"] for row in rows)
    report = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "label": label,
        "source": {
            "name": dataset["source_name"],
            "sha256": dataset["source_sha256"],
            "format": dataset["input_format"],
        },
        "input_provenance": dataset["provenance"],
        "run_configuration": dataset["configuration"],
        "phase_definition": {
            "opening_0_3": "own_decision_index 0 through 3",
            "midgame_4_11": "own_decision_index 4 through 11",
            "late_12_plus": "own_decision_index 12 or greater",
        },
        "clock_phase_definition": {
            "first_decision": (
                "own_decision_index 0; uses candidate/reference first-time limit"
            ),
            "later_decisions": (
                "own_decision_index greater than 0; uses later-time limit"
            ),
        },
        "clean_replay_audited_games_represented": {
            "total": len(games),
            "wins": game_results["win"],
            "losses": game_results["loss"],
        },
        "clean_replay_audited_decisions": {
            "total": len(rows),
            "wins": decision_results["win"],
            "losses": decision_results["loss"],
        },
        "overall": _overall_summary(rows),
        "breakdowns": {
            "by_result": _breakdown(rows, lambda row: row["result"]),
            "by_color": _breakdown(rows, lambda row: row["color"]),
            "by_phase": _breakdown(rows, phase_name),
            "by_clock_phase": _clock_phase_breakdown(rows),
            "by_result_color_phase": _breakdown(
                rows,
                lambda row: f"{row['result']}|{row['color']}|{phase_name(row)}",
            ),
        },
        "first_divergence": {
            "games_with_divergence": len(first_divergences),
            "games_without_divergence": len(games) - len(first_divergences),
            "by_game": [first_divergences[name] for name in games if name in first_divergences],
        },
    }
    arena_manifest = dataset.get("arena_manifest")
    if arena_manifest is not None:
        report["arena_manifest_join"] = {
            "source": {
                "name": arena_manifest["source_name"],
                "sha256": arena_manifest["manifest_sha256"],
                "content_addressed_filename": arena_manifest[
                    "filename_is_content_addressed"
                ],
            },
            "coverage_semantics": (
                "exactly all accepted, operationally clean, rule-terminal manifest "
                "games in which the candidate has at least one decision"
            ),
            "clean_manifest_games": len(arena_manifest["clean_games"]),
            "audited_manifest_games": len(
                arena_manifest["expected_audited_game_ids"]
            ),
            "clean_games_without_candidate_decision": arena_manifest[
                "zero_decision_game_ids"
            ],
            "opponent_rank_cohorts": {
                "rank_1_5": "frozen rank 1 through 5",
                "rank_1_10": "frozen rank 1 through 10 (cumulative)",
                "rank_1_20": "frozen rank 1 through 20 (cumulative)",
                "rank_21_plus": "frozen rank 21 or greater",
                "unranked": "opponent absent from the frozen leaderboard snapshot",
            },
        }
        report["breakdowns"].update(_arena_breakdowns(rows, arena_manifest))
    return report


COMPARISON_EVENT_SPECS: tuple[
    tuple[str, Callable[[dict[str, Any]], bool], bool | None], ...
] = (
    ("candidate_matches_actual", lambda row: row["candidate_matches_actual"], True),
    ("rank4_matches_actual", lambda row: row["reference_matches_actual"], True),
    ("candidate_matches_rank4", lambda row: row["candidate_matches_reference"], None),
    ("actual_exact_retained", lambda row: row["actual_action_retained_ordinal"] >= 0, True),
    ("actual_boundary_retained", lambda row: row["actual_boundary_retained_ordinal"] >= 0, True),
    ("candidate_deadline_reached", lambda row: row["candidate_deadline_reached"], False),
    ("candidate_generator_truncated", lambda row: row["candidate_generator_truncations"] > 0, False),
    ("diagnostic_root_truncated", lambda row: row["diagnostic_root_truncations"] > 0, False),
    ("actual_is_initial_best", lambda row: row["actual_initial_eval_rank"] == 1, True),
    (
        "candidate_bfm_changed_from_initial_best",
        lambda row: row["candidate_bfm_changed_from_initial_best"],
        None,
    ),
)


def _compare_event(
    baseline_rows: Sequence[dict[str, Any]],
    hypothesis_rows: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    higher_is_better: bool | None,
) -> dict[str, Any]:
    baseline_values = [predicate(row) for row in baseline_rows]
    hypothesis_values = [predicate(row) for row in hypothesis_rows]
    baseline_count = sum(baseline_values)
    hypothesis_count = sum(hypothesis_values)
    false_to_true = sum(
        1 for before, after in zip(baseline_values, hypothesis_values) if not before and after
    )
    true_to_false = sum(
        1 for before, after in zip(baseline_values, hypothesis_values) if before and not after
    )
    result: dict[str, Any] = {
        "baseline": {"count": baseline_count, "rate": _rate(baseline_count, len(baseline_rows))},
        "hypothesis": {"count": hypothesis_count, "rate": _rate(hypothesis_count, len(hypothesis_rows))},
        "delta_count": hypothesis_count - baseline_count,
        "delta_rate": round(
            _rate(hypothesis_count, len(hypothesis_rows))
            - _rate(baseline_count, len(baseline_rows)),
            6,
        ),
        "false_to_true": false_to_true,
        "true_to_false": true_to_false,
    }
    if higher_is_better is not None:
        result["improved"] = false_to_true if higher_is_better else true_to_false
        result["regressed"] = true_to_false if higher_is_better else false_to_true
    return result


def _comparison_metrics(
    baseline_rows: Sequence[dict[str, Any]], hypothesis_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        name: _compare_event(baseline_rows, hypothesis_rows, predicate, direction)
        for name, predicate, direction in COMPARISON_EVENT_SPECS
    }


def _validate_comparable(
    baseline: dict[str, Any], hypothesis: dict[str, Any]
) -> None:
    if baseline["provenance"] != hypothesis["provenance"]:
        raise AuditAnalysisError("comparison inputs have inconsistent provenance")
    baseline_arena = baseline.get("arena_manifest")
    hypothesis_arena = hypothesis.get("arena_manifest")
    if (baseline_arena is None) != (hypothesis_arena is None):
        raise AuditAnalysisError("comparison inputs have inconsistent arena joins")
    if baseline_arena is not None and (
        baseline_arena["manifest_sha256"] != hypothesis_arena["manifest_sha256"]
    ):
        raise AuditAnalysisError("comparison inputs use different arena manifests")
    baseline_rows = baseline["rows"]
    hypothesis_rows = hypothesis["rows"]
    if len(baseline_rows) != len(hypothesis_rows):
        raise AuditAnalysisError("comparison inputs have different decision counts")
    context_fields = (
        "game_id",
        "state_id",
        "transcript_prefix",
        "turn_index",
        "own_decision_index",
        "candidate_player",
        "color",
        "winner",
        "result",
        "actual_action",
    )
    for index, (before, after) in enumerate(zip(baseline_rows, hypothesis_rows), 1):
        differing = [field for field in context_fields if before[field] != after[field]]
        if differing:
            raise AuditAnalysisError(
                f"comparison row {index} does not align: {', '.join(differing)}"
            )


def _comparison_arena_breakdowns(
    baseline_rows: Sequence[dict[str, Any]],
    hypothesis_rows: Sequence[dict[str, Any]],
    arena_manifest: dict[str, Any],
) -> dict[str, Any]:
    games = arena_manifest["clean_games"]

    def frozen_rank(row: dict[str, Any]) -> int | None:
        return games[row["game_id"]]["opponent_frozen_rank"]

    cohorts: tuple[tuple[str, Callable[[int | None], bool]], ...] = (
        ("rank_1_5", lambda value: value is not None and 1 <= value <= 5),
        ("rank_1_10", lambda value: value is not None and 1 <= value <= 10),
        ("rank_1_20", lambda value: value is not None and 1 <= value <= 20),
        ("rank_21_plus", lambda value: value is not None and value >= 21),
        ("unranked", lambda value: value is None),
    )
    by_rank: dict[str, Any] = {}
    for name, predicate in cohorts:
        selected = [
            index
            for index, row in enumerate(baseline_rows)
            if predicate(frozen_rank(row))
        ]
        selected_baseline = [baseline_rows[index] for index in selected]
        by_rank[name] = {
            "game_results": _game_result_summary(selected_baseline),
            "metrics": _comparison_metrics(
                selected_baseline,
                [hypothesis_rows[index] for index in selected],
            ),
        }

    opponent_indices: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    opponent_games: dict[tuple[int, str], set[str]] = collections.defaultdict(set)
    for index, row in enumerate(baseline_rows):
        game = games[row["game_id"]]
        key = (game["opponent_agent_id"], game["opponent_name"])
        opponent_indices[key].append(index)
        opponent_games[key].add(row["game_id"])
    by_opponent = []
    for (agent_id, name), selected in sorted(
        opponent_indices.items(),
        key=lambda entry: (entry[0][1].casefold(), entry[0][0]),
    ):
        game_count = len(opponent_games[(agent_id, name)])
        if not name or game_count < MINIMUM_NAMED_OPPONENT_GAMES:
            continue
        by_opponent.append(
            {
                "agent_id": agent_id,
                "name": name,
                "games": game_count,
                "game_results": _game_result_summary(
                    [baseline_rows[index] for index in selected]
                ),
                "metrics": _comparison_metrics(
                    [baseline_rows[index] for index in selected],
                    [hypothesis_rows[index] for index in selected],
                ),
            }
        )
    return {
        "frozen_opponent_rank_cohorts": by_rank,
        "named_opponents": {
            "minimum_games": MINIMUM_NAMED_OPPONENT_GAMES,
            "buckets": by_opponent,
        },
    }


def compare_datasets(
    baseline: dict[str, Any],
    hypothesis: dict[str, Any],
    baseline_label: str,
    hypothesis_label: str,
) -> dict[str, Any]:
    _validate_comparable(baseline, hypothesis)
    baseline_rows = baseline["rows"]
    hypothesis_rows = hypothesis["rows"]
    configuration_changes = {
        field: {
            "baseline": baseline["configuration"][field],
            "hypothesis": hypothesis["configuration"][field],
        }
        for field in RUN_CONFIGURATION_FIELDS
        if baseline["configuration"][field] != hypothesis["configuration"][field]
    }
    breakdowns: dict[str, Any] = {}
    for dimension, key in (
        ("by_result", lambda row: row["result"]),
        ("by_color", lambda row: row["color"]),
        ("by_phase", phase_name),
        (
            "by_result_color_phase",
            lambda row: f"{row['result']}|{row['color']}|{phase_name(row)}",
        ),
    ):
        indices: dict[str, list[int]] = collections.defaultdict(list)
        for index, row in enumerate(baseline_rows):
            indices[key(row)].append(index)
        breakdowns[dimension] = {
            name: _comparison_metrics(
                [baseline_rows[index] for index in selected],
                [hypothesis_rows[index] for index in selected],
            )
            for name, selected in sorted(indices.items())
        }
    breakdowns["by_clock_phase"] = {}
    for name in ("first_decision", "later_decisions"):
        selected = [
            index
            for index, row in enumerate(baseline_rows)
            if clock_phase_name(row) == name
        ]
        breakdowns["by_clock_phase"][name] = _comparison_metrics(
            [baseline_rows[index] for index in selected],
            [hypothesis_rows[index] for index in selected],
        )

    first_choice_changes: list[dict[str, Any]] = []
    changed_games: set[str] = set()
    for before, after in zip(baseline_rows, hypothesis_rows):
        if before["game_id"] in changed_games or before["candidate_action"] == after["candidate_action"]:
            continue
        changed_games.add(before["game_id"])
        first_choice_changes.append(
            {
                "game_id": before["game_id"],
                "result": before["result"],
                "color": before["color"],
                "turn_index": before["turn_index"],
                "own_decision_index": before["own_decision_index"],
                "state_id": before["state_id"],
                "actual_action": before["actual_action"],
                "baseline_candidate_action": before["candidate_action"],
                "hypothesis_candidate_action": after["candidate_action"],
                "baseline_matches_actual": before["candidate_matches_actual"],
                "hypothesis_matches_actual": after["candidate_matches_actual"],
                "baseline_deadline_reached": before["candidate_deadline_reached"],
                "hypothesis_deadline_reached": after["candidate_deadline_reached"],
            }
        )
    arena_manifest = baseline.get("arena_manifest")
    if arena_manifest is not None:
        breakdowns.update(
            _comparison_arena_breakdowns(
                baseline_rows, hypothesis_rows, arena_manifest
            )
        )
    comparison_report = {
        "baseline_label": baseline_label,
        "hypothesis_label": hypothesis_label,
        "baseline_source": {
            "name": baseline["source_name"],
            "sha256": baseline["source_sha256"],
        },
        "hypothesis_source": {
            "name": hypothesis["source_name"],
            "sha256": hypothesis["source_sha256"],
        },
        "input_provenance": baseline["provenance"],
        "aligned_decisions": len(baseline_rows),
        "clock_phase_definition": {
            "first_decision": (
                "own_decision_index 0; uses candidate/reference first-time limit"
            ),
            "later_decisions": (
                "own_decision_index greater than 0; uses later-time limit"
            ),
        },
        "configuration_changes": configuration_changes,
        "overall": _comparison_metrics(baseline_rows, hypothesis_rows),
        "breakdowns": breakdowns,
        "candidate_action_changes": {
            "decisions": sum(
                before["candidate_action"] != after["candidate_action"]
                for before, after in zip(baseline_rows, hypothesis_rows)
            ),
            "games": len(changed_games),
            "first_by_game": first_choice_changes,
        },
    }
    if arena_manifest is not None:
        comparison_report["arena_manifest_join"] = {
            "name": arena_manifest["source_name"],
            "sha256": arena_manifest["manifest_sha256"],
            "coverage_semantics": (
                "exactly all accepted, operationally clean, rule-terminal manifest "
                "games in which the candidate has at least one decision"
            ),
        }
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "comparison": comparison_report,
    }


def _percent(metric: dict[str, Any]) -> str:
    return f"{metric['count']} ({100.0 * metric['rate']:.1f}%)"


def render_text(report: dict[str, Any]) -> str:
    if "comparison" in report:
        comparison = report["comparison"]
        lines = [
            f"Decision-audit comparison: {comparison['baseline_label']} -> {comparison['hypothesis_label']}",
            f"Aligned clean decisions: {comparison['aligned_decisions']}",
            f"Configuration changes: {len(comparison['configuration_changes'])}",
        ]
        for name, metric in comparison["overall"].items():
            direction = ""
            if "improved" in metric:
                direction = f", improved={metric['improved']}, regressed={metric['regressed']}"
            lines.append(
                f"  {name}: {metric['baseline']['count']} -> {metric['hypothesis']['count']} "
                f"(delta {metric['delta_count']:+d}{direction})"
            )
        changes = comparison["candidate_action_changes"]
        lines.append(
            f"Candidate action changes: {changes['decisions']} decisions in {changes['games']} games"
        )
        if "arena_manifest_join" in comparison:
            arena = comparison["arena_manifest_join"]
            lines.append(
                f"Arena manifest: {arena['name']} sha256={arena['sha256']} (exact clean-game join)"
            )
            lines.append("Frozen opponent rank cohorts (candidate matches actual):")
            for name, metrics in comparison["breakdowns"][
                "frozen_opponent_rank_cohorts"
            ].items():
                metric = metrics["metrics"]["candidate_matches_actual"]
                games = metrics["game_results"]
                lines.append(
                    f"  {name}: games={games['total']} "
                    f"(wins={games['wins']}, losses={games['losses']}), "
                    f"candidate_matches={metric['baseline']['count']} -> "
                    f"{metric['hypothesis']['count']} (delta {metric['delta_count']:+d})"
                )
            named = comparison["breakdowns"]["named_opponents"]
            lines.append(
                f"Named opponents with at least {named['minimum_games']} games:"
            )
            for bucket in named["buckets"]:
                metric = bucket["metrics"]["candidate_matches_actual"]
                games = bucket["game_results"]
                lines.append(
                    f"  {bucket['name']} (agent {bucket['agent_id']}): "
                    f"games={games['total']} "
                    f"(wins={games['wins']}, losses={games['losses']}), "
                    f"candidate_matches={metric['baseline']['count']} -> "
                    f"{metric['hypothesis']['count']} "
                    f"(delta {metric['delta_count']:+d})"
                )
        return "\n".join(lines) + "\n"

    decisions = report["clean_replay_audited_decisions"]
    games = report["clean_replay_audited_games_represented"]
    overall = report["overall"]
    lines = [
        f"Decision audit: {report['label']}",
        f"Source: {report['source']['name']} sha256={report['source']['sha256']}",
        f"Clean games represented: {games['total']} (wins={games['wins']}, losses={games['losses']})",
        f"Clean decisions: {decisions['total']} (wins={decisions['wins']}, losses={decisions['losses']})",
        "Agreement:",
    ]
    if "arena_manifest_join" in report:
        arena = report["arena_manifest_join"]
        lines.insert(
            4,
            "Arena manifest: "
            f"{arena['source']['name']} sha256={arena['source']['sha256']} "
            f"(audited={arena['audited_manifest_games']}, "
            f"zero-decision={len(arena['clean_games_without_candidate_decision'])})",
        )
    for name, metric in overall["agreement"].items():
        lines.append(f"  {name}: {_percent(metric)}")
    lines.append("Root ordinal missing rates:")
    for actor, coverage in overall["root_coverage"].items():
        if actor == "retained_actions_mean":
            continue
        lines.append(
            f"  {actor}: exact={_percent(coverage['exact']['missing'])}, "
            f"boundary={_percent(coverage['boundary']['missing'])}"
        )
    lines.append("Initial evaluator / BFM separation:")
    for name in (
        "actual_is_initial_best",
        "candidate_is_initial_best",
        "rank4_is_initial_best",
        "candidate_bfm_change_assessable",
        "candidate_bfm_changed_from_initial_best",
        "bfm_changed_to_actual_boundary",
        "bfm_changed_away_from_actual_initial_best",
    ):
        lines.append(
            f"  {name}: {_percent(overall['initial_evaluation'][name])}"
        )
    lines.append("Initial-evaluation mechanism buckets:")
    for name, count in overall["initial_evaluation_buckets"].items():
        lines.append(f"  {name}: {count}")
    lines.append("Deadline/truncation pressure:")
    for name in (
        "candidate_deadline_reached",
        "candidate_generator_truncated",
        "diagnostic_root_truncated",
        "diagnostic_root_nonexhaustive",
    ):
        lines.append(f"  {name}: {_percent(overall['pressure'][name])}")
    lines.append("Failure buckets:")
    for name, count in overall["failure_buckets"].items():
        lines.append(f"  {name}: {count}")
    lines.append("Breakdowns (candidate match / actual-boundary missing / deadline / root truncation):")
    for dimension in ("by_result", "by_color", "by_phase", "by_clock_phase"):
        lines.append(f"  {dimension}:")
        for name, bucket in report["breakdowns"][dimension].items():
            lines.append(
                f"    {name}: decisions={bucket['decisions']}, "
                f"match={_percent(bucket['agreement']['candidate_vs_actual'])}, "
                f"boundary_missing={_percent(bucket['root_missing']['actual_boundary'])}, "
                f"deadline={_percent(bucket['pressure']['candidate_deadline_reached'])}, "
                f"root_truncated={_percent(bucket['pressure']['diagnostic_root_truncated'])}"
            )
    if "frozen_opponent_rank_cohorts" in report["breakdowns"]:
        lines.append("Frozen opponent rank cohorts (top-N cohorts are cumulative):")
        for name, bucket in report["breakdowns"][
            "frozen_opponent_rank_cohorts"
        ].items():
            games = bucket["game_results"]
            lines.append(
                f"  {name}: games={games['total']} "
                f"(wins={games['wins']}, losses={games['losses']}), "
                f"decisions={bucket['decisions']}, "
                f"match={_percent(bucket['agreement']['candidate_vs_actual'])}"
            )
        named = report["breakdowns"]["named_opponents"]
        lines.append(
            f"Named opponents with at least {named['minimum_games']} games:"
        )
        for bucket in named["buckets"]:
            summary = bucket["summary"]
            games = bucket["game_results"]
            lines.append(
                f"  {bucket['name']} (agent {bucket['agent_id']}): "
                f"games={games['total']} "
                f"(wins={games['wins']}, losses={games['losses']}), "
                f"decisions={summary['decisions']}, "
                f"match={_percent(summary['agreement']['candidate_vs_actual'])}"
            )
    divergence = report["first_divergence"]
    lines.append(
        f"First divergences: {divergence['games_with_divergence']} games; "
        f"no divergence in {divergence['games_without_divergence']} games"
    )
    for entry in divergence["by_game"]:
        lines.append(
            f"  {entry['game_id']} own={entry['own_decision_index']} "
            f"bucket={entry['failure_bucket']} actual={entry['actual_action']} "
            f"candidate={entry['candidate_action']} rank4={entry['rank4_action']} "
            f"initial={entry['initial_eval_best_action']} "
            f"mechanism={entry['initial_evaluation_bucket']}"
        )
    return "\n".join(lines) + "\n"


def _safe_label(value: str, option: str) -> str:
    if not value or len(value) > 128 or any(ord(character) < 0x20 for character in value):
        raise AuditAnalysisError(f"{option} is invalid")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate candidate-owned replay decision audit JSONL/TSV."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--input-format", choices=("auto", "jsonl", "tsv"), default="auto")
    parser.add_argument("--label", default="audit")
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--compare-format", choices=("auto", "jsonl", "tsv"), default="auto")
    parser.add_argument("--compare-label", default="hypothesis")
    parser.add_argument(
        "--arena-manifest",
        type=pathlib.Path,
        help=(
            "explicit collector arena manifest to validate and exactly join; "
            "no directories or network are consulted"
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        label = _safe_label(arguments.label, "--label")
        baseline = load_audit(arguments.input, arguments.input_format)
        arena_manifest = (
            None
            if arguments.arena_manifest is None
            else load_arena_manifest(arguments.arena_manifest)
        )
        if arena_manifest is not None:
            baseline = join_arena_manifest(baseline, arena_manifest)
        if arguments.compare is None:
            report = analyze_dataset(baseline, label)
        else:
            comparison_label = _safe_label(arguments.compare_label, "--compare-label")
            hypothesis = load_audit(arguments.compare, arguments.compare_format)
            if arena_manifest is not None:
                hypothesis = join_arena_manifest(hypothesis, arena_manifest)
            report = compare_datasets(baseline, hypothesis, label, comparison_label)
        if arguments.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            sys.stdout.write(render_text(report))
        return 0
    except AuditAnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
