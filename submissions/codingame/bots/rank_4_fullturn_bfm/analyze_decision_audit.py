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


AUDIT_SCHEMA_VERSION = "fullturn-decision-audit-v2"
ANALYSIS_SCHEMA_VERSION = "fullturn-decision-analysis-v1"
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
)

NONNEGATIVE_INTEGER_FIELDS = tuple(
    field
    for field in INTEGER_FIELDS
    if field not in ORDINAL_FIELDS
    and field not in ("candidate_root_score", "reference_root_score")
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


class AuditAnalysisError(ValueError):
    """An audit cannot be safely or unambiguously aggregated."""


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
        raise AuditAnalysisError("TSV header does not match audit schema v2")

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
    ):
        if not ACTION_PATTERN.fullmatch(row[field]):
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
    for field in ("replay_correction_lookup_action", "replay_correction_action"):
        if row[field] and not ACTION_PATTERN.fullmatch(row[field]):
            raise AuditAnalysisError(f"line {line_number}: invalid {field}")
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
    if row["diagnostic_root_actions"] <= 0:
        raise AuditAnalysisError(f"line {line_number}: root action set is empty")
    if (
        not 1 <= row["candidate_max_actions"] <= 250
        or not 0 <= row["candidate_nonroot_actions"] <= 250
    ):
        raise AuditAnalysisError(f"line {line_number}: invalid candidate action cap")
    if row["diagnostic_root_actions"] > row["candidate_max_actions"]:
        raise AuditAnalysisError(
            f"line {line_number}: root actions exceed candidate_max_actions"
        )
    for exact_field, boundary_field in (
        ("actual_action_retained_ordinal", "actual_boundary_retained_ordinal"),
        ("candidate_action_retained_ordinal", "candidate_boundary_retained_ordinal"),
        ("reference_action_retained_ordinal", "reference_boundary_retained_ordinal"),
    ):
        if row[exact_field] >= 0 and row[boundary_field] < 0:
            raise AuditAnalysisError(
                f"line {line_number}: exact retention lacks boundary retention"
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
    if row["candidate_player"] not in (0, 1) or row["winner"] not in (0, 1):
        raise AuditAnalysisError(f"line {line_number}: invalid player identity")
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


def phase_name(row: dict[str, Any]) -> str:
    index = row["own_decision_index"]
    if index <= 3:
        return "opening_0_3"
    if index <= 11:
        return "midgame_4_11"
    return "late_12_plus"


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


def _compact_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets = collections.Counter(_failure_bucket(row) for row in rows)
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
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "comparison": {
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
        },
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
    for dimension in ("by_result", "by_color", "by_phase"):
        lines.append(f"  {dimension}:")
        for name, bucket in report["breakdowns"][dimension].items():
            lines.append(
                f"    {name}: decisions={bucket['decisions']}, "
                f"match={_percent(bucket['agreement']['candidate_vs_actual'])}, "
                f"boundary_missing={_percent(bucket['root_missing']['actual_boundary'])}, "
                f"deadline={_percent(bucket['pressure']['candidate_deadline_reached'])}, "
                f"root_truncated={_percent(bucket['pressure']['diagnostic_root_truncated'])}"
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
            f"candidate={entry['candidate_action']} rank4={entry['rank4_action']}"
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
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        label = _safe_label(arguments.label, "--label")
        baseline = load_audit(arguments.input, arguments.input_format)
        if arguments.compare is None:
            report = analyze_dataset(baseline, label)
        else:
            comparison_label = _safe_label(arguments.compare_label, "--compare-label")
            hypothesis = load_audit(arguments.compare, arguments.compare_format)
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
