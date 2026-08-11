#!/usr/bin/env python3
"""Strict offline aggregation for Jacek-native replay decision audits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any


AUDIT_SCHEMA_VERSION = "jacek-native-decision-audit-v1"
ANALYSIS_SCHEMA_VERSION = "jacek-native-decision-analysis-v1"
ARENA_BATCH_SCHEMA_VERSION = "papersoccer.codingame-arena-batch.v1"
ARENA_GAME_SCHEMA_VERSION = "papersoccer.codingame-arena-game.v1"
ARENA_BINDING_SCHEMA_VERSION = "papersoccer.codingame-arena-binding.v1"
ARENA_SOURCE_BINDING_STATUS = "asserted-not-api-verified"
ARENA_PURPOSE = {
    "diagnostic_only": True,
    "training_eligible": False,
    "note": (
        "arena observations may influence bot development and must not be "
        "treated as untouched evaluation or direct expert training labels"
    ),
}

MAXIMUM_INPUT_BYTES = 128 * 1024 * 1024
MAXIMUM_LINE_BYTES = 2 * 1024 * 1024
MAXIMUM_ROWS = 1_000_000
MAXIMUM_TREE_NODES = 120_000
MAXIMUM_ACTIONS = 250
MAXIMUM_PARTIAL_PATHS = 50_000
MAXIMUM_EXPANSIONS = 2_000_000

GAME_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
STATE_ID_PATTERN = re.compile(r"fnv1a64:[0-9a-f]{16}\Z")
ACTION_PATTERN = re.compile(r"[0-7]{1,1024}\Z")
TRANSCRIPT_PATTERN = re.compile(r"(?:[0-7]+(?:/[0-7]+)*)?\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
PROVENANCE_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")

CLASSIFICATIONS = {
    "match",
    "boundary-equivalent",
    "generator-omission",
    "operational-failure",
    "tactical-miss",
    "bfm-override",
    "initial-evaluator-ordering",
}
TACTICAL_CLASSES = {
    "immediate-goal",
    "forced-cutoff",
    "safe-handoff",
    "opponent-immediate-goal",
    "own-goal",
}
ARENA_OPERATIONAL_STATUSES = {
    "ok",
    "empty-output",
    "invalid-output",
    "illegal-action",
    "runtime-error",
    "timeout",
}

STRING_FIELDS = {
    "schema_version",
    "game_id",
    "state_id",
    "transcript_prefix",
    "color",
    "result",
    "classification",
    "classification_reason",
    "audit_mode",
    "model_sha256",
    "packed_weights_sha256",
    "actual_action",
    "actual_retained_action",
    "actual_tactical_class",
    "actual_final_tactical_class",
    "chosen_action",
    "chosen_retained_action",
    "chosen_tactical_class",
    "chosen_final_tactical_class",
    "initial_best_action",
}
INTEGER_FIELDS = {
    "turn_index",
    "own_decision_index",
    "candidate_player",
    "winner",
    "fixed_work_limit",
    "max_actions",
    "max_partial_paths",
    "max_expansions",
    "first_time_limit_ms",
    "later_time_limit_ms",
    "time_limit_ms",
    "actual_exact_retained_ordinal",
    "actual_boundary_retained_ordinal",
    "actual_initial_rank",
    "chosen_exact_retained_ordinal",
    "chosen_boundary_retained_ordinal",
    "chosen_initial_rank",
    "search_root_actions",
    "search_tree_nodes",
    "search_expansions",
    "search_generated_children",
    "search_child_evaluations",
    "search_tactical_child_values",
    "search_generator_partial_paths",
    "search_tactical_proof_paths",
    "search_completed_actions",
    "search_duplicate_boundaries",
    "search_fifo_extractions",
    "search_lifo_extractions",
    "search_tactical_actions",
    "search_tactical_classes_found",
    "search_tactical_proof_truncations",
    "search_generator_truncations",
    "diagnostic_root_actions",
    "diagnostic_root_partial_paths",
    "diagnostic_root_tactical_proof_paths",
    "diagnostic_root_completed_actions",
    "diagnostic_root_duplicate_boundaries",
    "diagnostic_root_fifo_extractions",
    "diagnostic_root_lifo_extractions",
    "diagnostic_root_tactical_actions",
    "diagnostic_root_tactical_classes_found",
    "diagnostic_root_truncations",
    "diagnostic_root_maximum_deque_size",
}
OPTIONAL_INTEGER_FIELDS = {
    "actual_final_visits",
    "actual_final_selection_visits",
    "chosen_final_visits",
    "chosen_final_selection_visits",
    "search_solved_winner",
}
FLOAT_FIELDS = {
    "exploration",
    "first_play_urgency",
    "chosen_value",
    "search_elapsed_ms",
    "initial_best_value",
}
OPTIONAL_FLOAT_FIELDS = {
    "actual_initial_neural_value",
    "actual_initial_action_value",
    "actual_final_backed_value",
    "chosen_initial_neural_value",
    "chosen_initial_action_value",
    "chosen_final_backed_value",
}
BOOLEAN_FIELDS = {
    "search_solved",
    "search_deadline_reached",
    "search_tree_cap_reached",
    "search_expansion_cap_reached",
    "diagnostic_root_tactical_proof_truncated",
    "diagnostic_root_deadline_reached",
    "diagnostic_root_exhaustive",
}
EXPECTED_FIELDS = (
    STRING_FIELDS
    | INTEGER_FIELDS
    | OPTIONAL_INTEGER_FIELDS
    | FLOAT_FIELDS
    | OPTIONAL_FLOAT_FIELDS
    | BOOLEAN_FIELDS
    | {"input_provenance"}
)
ORDINAL_FIELDS = {
    "actual_exact_retained_ordinal",
    "actual_boundary_retained_ordinal",
    "chosen_exact_retained_ordinal",
    "chosen_boundary_retained_ordinal",
}
RANK_FIELDS = {"actual_initial_rank", "chosen_initial_rank"}
CONFIGURATION_FIELDS = (
    "audit_mode",
    "model_sha256",
    "packed_weights_sha256",
    "fixed_work_limit",
    "max_actions",
    "max_partial_paths",
    "max_expansions",
    "exploration",
    "first_play_urgency",
    "first_time_limit_ms",
    "later_time_limit_ms",
)
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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def _read_bounded(path: pathlib.Path, label: str) -> bytes:
    if not path.is_file():
        raise AuditAnalysisError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_INPUT_BYTES:
        raise AuditAnalysisError(f"{label} size is outside supported bounds: {size}")
    return path.read_bytes()


def _printable(value: Any, label: str, maximum: int, *, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not empty and not value)
        or len(value) > maximum
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise AuditAnalysisError(f"{label} is not a bounded printable string")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise AuditAnalysisError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AuditAnalysisError(f"{label} must be a nonnegative integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AuditAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_provenance(value: Any, line_number: int) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 64:
        raise AuditAnalysisError(
            f"line {line_number}: input_provenance must be an object of at most 64 fields"
        )
    result: dict[str, str] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or PROVENANCE_KEY_PATTERN.fullmatch(key) is None:
            raise AuditAnalysisError(f"line {line_number}: invalid provenance key {key!r}")
        result[key] = _printable(
            entry, f"line {line_number}: provenance value for {key}", 512
        )
    return dict(sorted(result.items()))


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AuditAnalysisError(f"{label} must be finite")
    return float(value)


def _validate_action_diagnostic(row: dict[str, Any], prefix: str, line: int) -> None:
    action = row[f"{prefix}_action"]
    if ACTION_PATTERN.fullmatch(action) is None:
        raise AuditAnalysisError(f"line {line}: invalid {prefix}_action")
    exact = row[f"{prefix}_exact_retained_ordinal"]
    boundary = row[f"{prefix}_boundary_retained_ordinal"]
    root_actions = row["diagnostic_root_actions"]
    for field, value in (("exact ordinal", exact), ("boundary ordinal", boundary)):
        if value < -1 or value >= root_actions:
            raise AuditAnalysisError(f"line {line}: {prefix} {field} is outside the root")
    retained = row[f"{prefix}_retained_action"]
    tactical = row[f"{prefix}_tactical_class"]
    initial_neural = row[f"{prefix}_initial_neural_value"]
    initial_action = row[f"{prefix}_initial_action_value"]
    rank = row[f"{prefix}_initial_rank"]
    final_value = row[f"{prefix}_final_backed_value"]
    final_visits = row[f"{prefix}_final_visits"]
    final_selection = row[f"{prefix}_final_selection_visits"]
    final_tactical = row[f"{prefix}_final_tactical_class"]
    if boundary < 0:
        if (
            exact != -1
            or retained
            or tactical != "not-retained"
            or initial_neural is not None
            or initial_action is not None
            or rank != -1
            or final_value is not None
            or final_visits is not None
            or final_selection is not None
            or final_tactical != "not-searched"
        ):
            raise AuditAnalysisError(
                f"line {line}: absent {prefix} boundary carries retained evidence"
            )
        return
    if (
        ACTION_PATTERN.fullmatch(retained) is None
        or tactical not in TACTICAL_CLASSES
        or initial_action is None
        or rank < 1
        or rank > root_actions
    ):
        raise AuditAnalysisError(f"line {line}: retained {prefix} diagnostics disagree")
    if exact >= 0 and (exact != boundary or retained != action):
        raise AuditAnalysisError(
            f"line {line}: exact {prefix} retention disagrees with its boundary"
        )
    final_fields = (final_value, final_visits, final_selection)
    if any(value is None for value in final_fields) != all(
        value is None for value in final_fields
    ):
        raise AuditAnalysisError(f"line {line}: partial {prefix} final search evidence")
    if final_value is None:
        if final_tactical != "not-searched":
            raise AuditAnalysisError(f"line {line}: unsearched {prefix} has tactics")
    elif final_tactical not in TACTICAL_CLASSES:
        raise AuditAnalysisError(f"line {line}: invalid final {prefix} tactical class")


def _expected_classification(row: dict[str, Any]) -> str:
    if row["actual_action"] == row["chosen_action"]:
        return "match"
    actual_boundary = row["actual_boundary_retained_ordinal"]
    chosen_boundary = row["chosen_boundary_retained_ordinal"]
    if actual_boundary >= 0 and actual_boundary == chosen_boundary:
        return "boundary-equivalent"
    if actual_boundary < 0:
        return "generator-omission"
    if (
        chosen_boundary < 0
        or row["chosen_final_backed_value"] is None
        or (
            row["search_deadline_reached"]
            and row["actual_final_backed_value"] is None
        )
    ):
        return "operational-failure"
    tactical_order = {
        "immediate-goal": 0,
        "forced-cutoff": 1,
        "safe-handoff": 2,
        "opponent-immediate-goal": 3,
        "own-goal": 4,
    }
    actual_tactical = tactical_order[row["actual_tactical_class"]]
    chosen_tactical = tactical_order[row["chosen_tactical_class"]]
    if (
        (actual_tactical <= 1 and actual_tactical < chosen_tactical)
        or (chosen_tactical >= 3 and actual_tactical < chosen_tactical)
    ):
        return "tactical-miss"
    if row["chosen_retained_action"] != row["initial_best_action"]:
        return "bfm-override"
    return "initial-evaluator-ordering"


def _validate_row(row: dict[str, Any], line: int) -> None:
    actual = set(row)
    if actual != EXPECTED_FIELDS:
        raise AuditAnalysisError(
            f"line {line}: schema fields differ; "
            f"missing={sorted(EXPECTED_FIELDS - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_FIELDS)}"
        )
    for field in STRING_FIELDS:
        if not isinstance(row[field], str):
            raise AuditAnalysisError(f"line {line}: {field} must be a string")
    for field in INTEGER_FIELDS:
        if type(row[field]) is not int:
            raise AuditAnalysisError(f"line {line}: {field} must be an integer")
    for field in OPTIONAL_INTEGER_FIELDS:
        if row[field] is not None and type(row[field]) is not int:
            raise AuditAnalysisError(f"line {line}: {field} must be an integer or null")
    for field in FLOAT_FIELDS:
        row[field] = _finite(row[field], f"line {line}: {field}")
    for field in OPTIONAL_FLOAT_FIELDS:
        if row[field] is not None:
            row[field] = _finite(row[field], f"line {line}: {field}")
    for field in BOOLEAN_FIELDS:
        if type(row[field]) is not bool:
            raise AuditAnalysisError(f"line {line}: {field} must be boolean")
    row["input_provenance"] = _validate_provenance(row["input_provenance"], line)

    if row["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise AuditAnalysisError(f"line {line}: unsupported audit schema")
    if GAME_ID_PATTERN.fullmatch(row["game_id"]) is None:
        raise AuditAnalysisError(f"line {line}: invalid game_id")
    if STATE_ID_PATTERN.fullmatch(row["state_id"]) is None:
        raise AuditAnalysisError(f"line {line}: invalid state_id")
    if (
        len(row["transcript_prefix"]) > 65_536
        or TRANSCRIPT_PATTERN.fullmatch(row["transcript_prefix"]) is None
    ):
        raise AuditAnalysisError(f"line {line}: invalid transcript_prefix")
    if row["candidate_player"] not in (0, 1) or row["winner"] not in (0, 1):
        raise AuditAnalysisError(f"line {line}: invalid player identity")
    expected_color = "player_one" if row["candidate_player"] == 0 else "player_two"
    expected_result = "win" if row["candidate_player"] == row["winner"] else "loss"
    if row["color"] != expected_color or row["result"] != expected_result:
        raise AuditAnalysisError(f"line {line}: color/result contradicts player identity")
    if row["turn_index"] != 2 * row["own_decision_index"] + row["candidate_player"]:
        raise AuditAnalysisError(f"line {line}: own decision index contradicts turn index")
    prefix_turns = 0 if not row["transcript_prefix"] else row["transcript_prefix"].count("/") + 1
    if prefix_turns != row["turn_index"]:
        raise AuditAnalysisError(f"line {line}: transcript prefix length contradicts turn index")
    if row["classification"] not in CLASSIFICATIONS:
        raise AuditAnalysisError(f"line {line}: invalid classification")
    _printable(row["classification_reason"], f"line {line}: classification_reason", 1024)
    if row["audit_mode"] not in {"fixed-work", "clock"}:
        raise AuditAnalysisError(f"line {line}: invalid audit mode")
    _sha256(row["model_sha256"], f"line {line}: model_sha256")
    _sha256(row["packed_weights_sha256"], f"line {line}: packed_weights_sha256")

    if not 2 <= row["fixed_work_limit"] <= MAXIMUM_TREE_NODES:
        raise AuditAnalysisError(f"line {line}: fixed work is outside native bounds")
    if not 1 <= row["max_actions"] <= MAXIMUM_ACTIONS:
        raise AuditAnalysisError(f"line {line}: max_actions is outside native bounds")
    if not 1 <= row["max_partial_paths"] <= MAXIMUM_PARTIAL_PATHS:
        raise AuditAnalysisError(f"line {line}: max_partial_paths is outside native bounds")
    if not 1 <= row["max_expansions"] <= MAXIMUM_EXPANSIONS:
        raise AuditAnalysisError(f"line {line}: max_expansions is outside native bounds")
    if row["exploration"] < 0:
        raise AuditAnalysisError(f"line {line}: exploration is negative")
    if row["audit_mode"] == "fixed-work":
        if any(row[field] != 0 for field in ("first_time_limit_ms", "later_time_limit_ms", "time_limit_ms")):
            raise AuditAnalysisError(f"line {line}: fixed-work audit carries clock limits")
    else:
        expected_time = (
            row["first_time_limit_ms"]
            if row["own_decision_index"] == 0
            else row["later_time_limit_ms"]
        )
        if min(row["first_time_limit_ms"], row["later_time_limit_ms"]) <= 0 or row["time_limit_ms"] != expected_time:
            raise AuditAnalysisError(f"line {line}: clock audit has inconsistent limits")

    for field in INTEGER_FIELDS - ORDINAL_FIELDS - RANK_FIELDS:
        if row[field] < 0:
            raise AuditAnalysisError(f"line {line}: {field} is negative")
    for field in OPTIONAL_INTEGER_FIELDS:
        if row[field] is not None and row[field] < 0:
            raise AuditAnalysisError(f"line {line}: {field} is negative")
    if row["search_solved_winner"] is not None and row["search_solved_winner"] not in (0, 1):
        raise AuditAnalysisError(f"line {line}: invalid solved winner")
    if row["search_solved"] != (row["search_solved_winner"] is not None):
        raise AuditAnalysisError(f"line {line}: solved flag and winner disagree")
    if row["search_elapsed_ms"] < 0:
        raise AuditAnalysisError(f"line {line}: negative elapsed time")
    if row["search_tree_nodes"] > row["fixed_work_limit"]:
        raise AuditAnalysisError(f"line {line}: search exceeded its tree-node limit")
    if row["search_expansions"] > row["max_expansions"]:
        raise AuditAnalysisError(f"line {line}: search exceeded its expansion limit")
    if max(row["search_root_actions"], row["diagnostic_root_actions"]) > row["max_actions"]:
        raise AuditAnalysisError(f"line {line}: root actions exceeded max_actions")
    if row["diagnostic_root_partial_paths"] > row["max_partial_paths"]:
        raise AuditAnalysisError(f"line {line}: diagnostic root exceeded its path limit")
    if row["diagnostic_root_fifo_extractions"] + row["diagnostic_root_lifo_extractions"] != row["diagnostic_root_partial_paths"]:
        raise AuditAnalysisError(f"line {line}: diagnostic deque counters disagree")
    # The bounded tactical-witness BFS is independent of the complete-turn
    # deque. Its 64-path proof cap may be reached even when the main generator
    # exhausts every complete turn. Only main-generator deadline/truncation
    # determines GenerationResult::exhaustive.
    expected_exhaustive = (
        not row["diagnostic_root_deadline_reached"]
        and row["diagnostic_root_truncations"] == 0
    )
    if row["diagnostic_root_exhaustive"] != expected_exhaustive:
        raise AuditAnalysisError(
            f"line {line}: diagnostic root exhaustiveness contradicts "
            "main-generator pressure"
        )
    if ACTION_PATTERN.fullmatch(row["initial_best_action"]) is None:
        raise AuditAnalysisError(f"line {line}: invalid initial_best_action")

    _validate_action_diagnostic(row, "actual", line)
    _validate_action_diagnostic(row, "chosen", line)
    # Search root generation xors its node index into the shuffle seed, while
    # the independent diagnostic root uses the generator's base seed. Under a
    # root cap, either legal boundary (including the chosen one) may therefore
    # be absent from the diagnostic retained set. Exhaustive generation cannot
    # omit a legal observed or chosen boundary.
    if row["diagnostic_root_exhaustive"] and (
        row["actual_boundary_retained_ordinal"] < 0
        or row["chosen_boundary_retained_ordinal"] < 0
    ):
        raise AuditAnalysisError(
            f"line {line}: exhaustive diagnostic root omits a legal boundary"
        )
    if row["classification"] != _expected_classification(row):
        raise AuditAnalysisError(f"line {line}: classification contradicts decision evidence")


def load_audit(path: pathlib.Path) -> dict[str, Any]:
    raw = _read_bounded(path, "audit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditAnalysisError("audit is not strict UTF-8") from error
    if "\x00" in text:
        raise AuditAnalysisError("audit contains a NUL byte")
    lines = text.splitlines()
    if not lines:
        raise AuditAnalysisError("audit is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise AuditAnalysisError(f"line {line_number}: blank JSONL row")
        if len(line.encode("utf-8")) > MAXIMUM_LINE_BYTES:
            raise AuditAnalysisError(f"line {line_number}: JSONL row is too large")
        row = _strict_json_loads(line, f"line {line_number}")
        if not isinstance(row, dict):
            raise AuditAnalysisError(f"line {line_number}: row must be an object")
        _validate_row(row, line_number)
        rows.append(row)
        if len(rows) > MAXIMUM_ROWS:
            raise AuditAnalysisError("audit exceeds the row limit")

    provenance = rows[0]["input_provenance"]
    configuration = {field: rows[0][field] for field in CONFIGURATION_FIELDS}
    seen_keys: set[tuple[str, str]] = set()
    closed_games: set[str] = set()
    current_game: str | None = None
    by_game: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for line_number, row in enumerate(rows, 1):
        if row["input_provenance"] != provenance:
            raise AuditAnalysisError(f"line {line_number}: provenance differs within audit")
        candidate_configuration = {field: row[field] for field in CONFIGURATION_FIELDS}
        if candidate_configuration != configuration:
            raise AuditAnalysisError(f"line {line_number}: run configuration differs within audit")
        key = (row["game_id"], row["state_id"])
        if key in seen_keys:
            raise AuditAnalysisError(f"line {line_number}: duplicate game/state identity")
        seen_keys.add(key)
        if row["game_id"] != current_game:
            if row["game_id"] in closed_games:
                raise AuditAnalysisError(f"line {line_number}: game rows are not contiguous")
            if current_game is not None:
                closed_games.add(current_game)
            current_game = row["game_id"]
        by_game[row["game_id"]].append(row)

    for game_id, game_rows in by_game.items():
        first = game_rows[0]
        for expected_index, row in enumerate(game_rows):
            if row["own_decision_index"] != expected_index:
                raise AuditAnalysisError(f"game {game_id}: own decision indices are not complete")
            for field in ("candidate_player", "winner", "color", "result"):
                if row[field] != first[field]:
                    raise AuditAnalysisError(f"game {game_id}: {field} differs within game")

    return {
        "rows": rows,
        "provenance": provenance,
        "configuration": configuration,
        "source_name": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _event(rows: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    count = sum(predicate(row) for row in rows)
    return {"count": count, "rate": _rate(count, len(rows))}


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90_nearest_rank": p90,
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
    }


def _compact_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    classifications = collections.Counter(row["classification"] for row in rows)
    actual_ranks = [row["actual_initial_rank"] for row in rows if row["actual_initial_rank"] > 0]
    return {
        "decisions": len(rows),
        "classifications": {
            name: {"count": classifications[name], "rate": _rate(classifications[name], len(rows))}
            for name in sorted(CLASSIFICATIONS)
        },
        "actual": {
            "exact_retained": _event(rows, lambda row: row["actual_exact_retained_ordinal"] >= 0),
            "boundary_retained": _event(rows, lambda row: row["actual_boundary_retained_ordinal"] >= 0),
            "boundary_omitted": _event(rows, lambda row: row["actual_boundary_retained_ordinal"] < 0),
            "initial_rank": {
                "distribution_when_retained": _distribution(actual_ranks),
                "rank_1": _event(rows, lambda row: row["actual_initial_rank"] == 1),
                "rank_top_3": _event(rows, lambda row: 1 <= row["actual_initial_rank"] <= 3),
                "rank_top_5": _event(rows, lambda row: 1 <= row["actual_initial_rank"] <= 5),
                "unranked": _event(rows, lambda row: row["actual_initial_rank"] < 0),
            },
        },
        "pressure": {
            "search_deadline": _event(rows, lambda row: row["search_deadline_reached"]),
            "search_tree_cap": _event(rows, lambda row: row["search_tree_cap_reached"]),
            "search_expansion_cap": _event(rows, lambda row: row["search_expansion_cap_reached"]),
            "search_generator_truncated": _event(rows, lambda row: row["search_generator_truncations"] > 0),
            "search_tactical_proof_truncated": _event(rows, lambda row: row["search_tactical_proof_truncations"] > 0),
            "diagnostic_root_deadline": _event(rows, lambda row: row["diagnostic_root_deadline_reached"]),
            "diagnostic_root_truncated": _event(rows, lambda row: row["diagnostic_root_truncations"] > 0),
            "diagnostic_root_proof_truncated": _event(rows, lambda row: row["diagnostic_root_tactical_proof_truncated"]),
            "diagnostic_root_nonexhaustive": _event(rows, lambda row: not row["diagnostic_root_exhaustive"]),
        },
    }


def _breakdown(rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {name: _compact_summary(grouped[name]) for name in sorted(grouped)}


def _first_divergences(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    games: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        games[row["game_id"]].append(row)
    divergent = []
    for game_id, game_rows in games.items():
        first = next(
            (
                row
                for row in game_rows
                if row["classification"] not in {"match", "boundary-equivalent"}
            ),
            None,
        )
        if first is not None:
            divergent.append(
                {
                    "game_id": game_id,
                    "result": first["result"],
                    "candidate_player": first["candidate_player"],
                    "turn_index": first["turn_index"],
                    "own_decision_index": first["own_decision_index"],
                    "state_id": first["state_id"],
                    "classification": first["classification"],
                    "actual_action": first["actual_action"],
                    "chosen_action": first["chosen_action"],
                    "actual_boundary_retained_ordinal": first["actual_boundary_retained_ordinal"],
                    "actual_initial_rank": first["actual_initial_rank"],
                }
            )
    return {
        "games_with_divergence": len(divergent),
        "games_without_divergence": len(games) - len(divergent),
        "by_game": divergent,
    }


def _game_results(games: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(games)
    wins = sum(game["result"] == "win" for game in materialized)
    losses = sum(game["result"] == "loss" for game in materialized)
    return {"games": len(materialized), "wins": wins, "losses": losses, "win_rate": _rate(wins, len(materialized))}


def _validate_reference_path(value: Any, digest: str, label: str) -> None:
    path = _printable(value, label, 4096)
    if pathlib.PurePosixPath(path).name.split(".", 1)[0] != digest:
        raise AuditAnalysisError(f"{label} does not name its content hash")


def _validate_clean_record(
    record: dict[str, Any],
    agent_id: int,
    submission_id: int,
    source_sha256: str,
    leaderboard_frozen_at: str,
) -> dict[str, Any] | None:
    game_id = _positive_integer(record.get("game_id"), "arena game_id")
    if record.get("schema") != ARENA_GAME_SCHEMA_VERSION or record.get("purpose") != ARENA_PURPOSE:
        raise AuditAnalysisError(f"arena game {game_id} has invalid schema/purpose")
    if record.get("source_sha256") != source_sha256:
        raise AuditAnalysisError(f"arena game {game_id} contradicts source binding")
    focus = record.get("focus")
    if not isinstance(focus, dict) or focus.get("agent_id") != agent_id:
        raise AuditAnalysisError(f"arena game {game_id} contradicts agent binding")
    status = record.get("status")
    if status == "accepted" and focus.get("submission_id") != submission_id:
        raise AuditAnalysisError(f"arena game {game_id} contradicts submission binding")
    if status != "accepted":
        return None
    player = focus.get("player_id")
    if type(player) is not int or player not in (0, 1) or focus.get("color") != f"player-{player}":
        raise AuditAnalysisError(f"arena game {game_id} has invalid focus player")
    opponent = record.get("opponent")
    if not isinstance(opponent, dict):
        raise AuditAnalysisError(f"arena game {game_id} omits opponent")
    opponent_id = _positive_integer(opponent.get("agent_id"), f"arena game {game_id} opponent")
    if opponent_id == agent_id or opponent.get("player_id") != 1 - player:
        raise AuditAnalysisError(f"arena game {game_id} has invalid opponent identity")
    opponent_name = _printable(opponent.get("name"), f"arena game {game_id} opponent name", 256, empty=True)
    frozen_rank = opponent.get("frozen_rank")
    if frozen_rank is not None:
        frozen_rank = _positive_integer(frozen_rank, f"arena game {game_id} frozen rank")
    if record.get("leaderboard_frozen_at_utc") != leaderboard_frozen_at:
        raise AuditAnalysisError(f"arena game {game_id} contradicts leaderboard freeze")
    operational = record.get("operational")
    if not isinstance(operational, dict):
        raise AuditAnalysisError(f"arena game {game_id} omits operational status")
    classification = operational.get("classification")
    focus_status = operational.get("focus_status")
    opponent_status = operational.get("opponent_status")
    if (
        classification not in {"clean", "operationally-terminated"}
        or focus_status not in ARENA_OPERATIONAL_STATUSES
        or opponent_status not in ARENA_OPERATIONAL_STATUSES
    ):
        raise AuditAnalysisError(f"arena game {game_id} has invalid operational status")
    clean = focus_status == "ok" and opponent_status == "ok"
    if (classification == "clean") != clean:
        raise AuditAnalysisError(f"arena game {game_id} has contradictory operational status")
    if not clean:
        return None
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("winner_player_id") not in (0, 1):
        raise AuditAnalysisError(f"arena game {game_id} has invalid outcome")
    winner = outcome["winner_player_id"]
    result = "win" if winner == player else "loss"
    if focus.get("result") != result:
        raise AuditAnalysisError(f"arena game {game_id} result contradicts winner")
    expected_winner_agent = agent_id if winner == player else opponent_id
    if outcome.get("winner_agent_id") != expected_winner_agent:
        raise AuditAnalysisError(f"arena game {game_id} winner agent contradicts player")
    replay = record.get("replay")
    validation = replay.get("rules_validation") if isinstance(replay, dict) else None
    if not isinstance(validation, dict) or validation.get("status") != "terminal-valid" or validation.get("terminal_winner_player_id") != winner:
        raise AuditAnalysisError(f"arena game {game_id} is not clean rule-terminal")
    transcript = replay.get("valid_transcript")
    if not isinstance(transcript, str) or not transcript or len(transcript) > 4 * 1024 * 1024:
        raise AuditAnalysisError(f"arena game {game_id} has invalid transcript")
    turns = transcript.split("/")
    if any(ACTION_PATTERN.fullmatch(action) is None for action in turns):
        raise AuditAnalysisError(f"arena game {game_id} has invalid transcript action")
    valid_turns = validation.get("valid_turns")
    if not isinstance(valid_turns, list) or validation.get("valid_turn_count") != len(turns) or len(valid_turns) != len(turns):
        raise AuditAnalysisError(f"arena game {game_id} has inconsistent validated turns")
    for index, (action, turn) in enumerate(zip(turns, valid_turns)):
        if not isinstance(turn, dict) or turn.get("action") != action or turn.get("player_id") != index % 2:
            raise AuditAnalysisError(f"arena game {game_id} validated turn {index} differs")
    if replay.get("valid_turns") != valid_turns or replay.get("observed_turns") != valid_turns or replay.get("observed_transcript") != transcript:
        raise AuditAnalysisError(f"arena game {game_id} transcript representations differ")
    agents = replay.get("agents")
    if not isinstance(agents, list) or len(agents) != 2:
        raise AuditAnalysisError(f"arena game {game_id} replay agents are invalid")
    expected_agents = {player: agent_id, 1 - player: opponent_id}
    seen_players: set[int] = set()
    for replay_agent in agents:
        if not isinstance(replay_agent, dict):
            raise AuditAnalysisError(f"arena game {game_id} replay agent is invalid")
        replay_player = replay_agent.get("player_id")
        if (
            type(replay_player) is not int
            or replay_player not in (0, 1)
            or replay_player in seen_players
            or replay_agent.get("agent_id") != expected_agents[replay_player]
        ):
            raise AuditAnalysisError(f"arena game {game_id} replay identity differs")
        seen_players.add(replay_player)
    return {
        "game_id": str(game_id),
        "candidate_player": player,
        "winner": winner,
        "result": result,
        "color": "player_one" if player == 0 else "player_two",
        "turns": turns,
        "expected_decisions": len(range(player, len(turns), 2)),
        "opponent_agent_id": opponent_id,
        "opponent_name": opponent_name,
        "opponent_frozen_rank": frozen_rank,
    }


def load_arena_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Read and validate only the explicitly named, self-contained manifest."""

    raw = _read_bounded(path, "arena manifest")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditAnalysisError("arena manifest is not strict UTF-8") from error
    payload = _strict_json_loads(text, "arena manifest")
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise AuditAnalysisError("arena manifest must be one canonical JSON object")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if SHA256_PATTERN.fullmatch(path.stem) and (path.suffix != ".json" or path.stem != manifest_sha256):
        raise AuditAnalysisError("arena manifest filename hash differs from content")
    if payload.get("schema") != ARENA_BATCH_SCHEMA_VERSION or payload.get("purpose") != ARENA_PURPOSE:
        raise AuditAnalysisError("arena manifest has invalid schema/purpose")
    collector_sha256 = _sha256(payload.get("collector_sha256"), "arena collector SHA-256")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AuditAnalysisError("arena run_id is invalid")
    binding = payload.get("binding")
    if not isinstance(binding, dict) or binding.get("schema") != ARENA_BINDING_SCHEMA_VERSION or binding.get("purpose") != ARENA_PURPOSE or binding.get("collector_sha256") != collector_sha256 or binding.get("run_id") != run_id:
        raise AuditAnalysisError("arena source binding contradicts manifest")
    agent_id = _positive_integer(binding.get("agent_id"), "arena agent_id")
    submission_id = _positive_integer(binding.get("asserted_submission_id"), "arena submission_id")
    repository_commit = binding.get("repository_commit")
    if not isinstance(repository_commit, str) or GIT_OBJECT_PATTERN.fullmatch(repository_commit) is None:
        raise AuditAnalysisError("arena repository commit is invalid")
    source = binding.get("source")
    if not isinstance(source, dict):
        raise AuditAnalysisError("arena source binding omits source metadata")
    source_sha256 = _sha256(source.get("sha256"), "arena source SHA-256")
    _validate_reference_path(source.get("archived_path"), source_sha256, "archived source path")
    _printable(source.get("input_path"), "arena input source path", 4096)
    if (
        source.get("encoding") != "utf-8"
        or type(source.get("bytes")) is not int
        or source.get("bytes") < 0
        or type(source.get("characters")) is not int
        or source.get("characters") < 0
        or source.get("characters") > source.get("bytes")
    ):
        raise AuditAnalysisError("arena source payload metadata is inconsistent")
    exclusion = payload.get("exclusion_registry")
    if not isinstance(exclusion, dict):
        raise AuditAnalysisError("arena manifest omits exclusion registry")
    exclusion_sha256 = _sha256(exclusion.get("sha256"), "arena exclusion SHA-256")
    _validate_reference_path(exclusion.get("path"), exclusion_sha256, "exclusion registry path")
    leaderboard = payload.get("leaderboard_snapshot")
    if not isinstance(leaderboard, dict):
        raise AuditAnalysisError("arena manifest omits leaderboard snapshot")
    leaderboard_frozen_at = _printable(
        leaderboard.get("frozen_at_utc"), "arena leaderboard freeze", 64
    )
    _sha256(leaderboard.get("normalized_sha256"), "arena leaderboard normalized SHA-256")
    _sha256(leaderboard.get("raw_sha256"), "arena leaderboard raw SHA-256")
    window = payload.get("window_snapshot")
    if not isinstance(window, dict):
        raise AuditAnalysisError("arena manifest omits battle-window snapshot")
    _sha256(window.get("normalized_sha256"), "arena window normalized SHA-256")
    _sha256(window.get("raw_sha256"), "arena window raw SHA-256")

    stored_games = payload.get("games")
    if not isinstance(stored_games, list):
        raise AuditAnalysisError("arena manifest games must be a list")
    statuses: collections.Counter[str] = collections.Counter()
    accepted = 0
    focus_failures = 0
    opponent_failures = 0
    seen: set[int] = set()
    clean_games: dict[str, dict[str, Any]] = {}
    opponent_snapshots: dict[int, tuple[str, int | None]] = {}
    for index, stored in enumerate(stored_games):
        if not isinstance(stored, dict) or not isinstance(stored.get("record"), dict):
            raise AuditAnalysisError(f"arena game binding {index} is invalid")
        record = stored["record"]
        record_hash = _sha256(stored.get("record_sha256"), f"arena game binding {index} hash")
        if hashlib.sha256(_canonical_json_bytes(record)).hexdigest() != record_hash:
            raise AuditAnalysisError(f"arena game binding {index} hash differs")
        _validate_reference_path(stored.get("record_path"), record_hash, f"arena game binding {index} path")
        game_id = _positive_integer(record.get("game_id"), "arena game_id")
        if game_id in seen:
            raise AuditAnalysisError(f"arena manifest repeats game {game_id}")
        seen.add(game_id)
        status = _printable(record.get("status"), f"arena game {game_id} status", 64)
        statuses[status] += 1
        if status == "accepted":
            accepted += 1
            operational = record.get("operational")
            if isinstance(operational, dict):
                focus_failures += operational.get("focus_status") != "ok"
                opponent_failures += operational.get("opponent_status") != "ok"
        clean = _validate_clean_record(
            record,
            agent_id,
            submission_id,
            source_sha256,
            leaderboard_frozen_at,
        )
        if clean is not None:
            clean_games[clean["game_id"]] = clean
            opponent_id = clean["opponent_agent_id"]
            snapshot = (clean["opponent_name"], clean["opponent_frozen_rank"])
            previous = opponent_snapshots.setdefault(opponent_id, snapshot)
            if previous != snapshot:
                raise AuditAnalysisError(
                    f"arena opponent {opponent_id} has inconsistent frozen metadata"
                )

    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise AuditAnalysisError("arena manifest omits coverage")
    expected_games = _positive_integer(coverage.get("expected_games"), "expected arena games")
    expected_counts = {
        "accepted_games": accepted,
        "battle_window_games": len(stored_games),
        "clean_rule_terminal_games": len(clean_games),
        "focus_operational_failures": focus_failures,
        "opponent_operational_failures": opponent_failures,
    }
    if any(coverage.get(field) != count for field, count in expected_counts.items()) or coverage.get("status_counts") != dict(sorted(statuses.items())):
        raise AuditAnalysisError("arena coverage counters contradict embedded games")
    fully_accounted = len(stored_games) >= expected_games and all(
        status in {"accepted", "excluded-protected", "already-known-local"}
        for status in statuses
    )
    if coverage.get("full_window_accounted") is not fully_accounted or not fully_accounted:
        raise AuditAnalysisError("arena battle window is not fully accounted")
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
    return {
        "source_name": path.name,
        "manifest_sha256": manifest_sha256,
        "expected_provenance": expected_provenance,
        "clean_games": clean_games,
        "expected_audited_game_ids": {
            game_id for game_id, game in clean_games.items() if game["expected_decisions"]
        },
        "zero_decision_game_ids": sorted(
            game_id for game_id, game in clean_games.items() if not game["expected_decisions"]
        ),
    }


def join_arena_manifest(dataset: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if dataset["provenance"] != manifest["expected_provenance"]:
        raise AuditAnalysisError("audit provenance does not exactly match arena manifest")
    by_game: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in dataset["rows"]:
        if not row["game_id"].isdigit() or str(int(row["game_id"])) != row["game_id"]:
            raise AuditAnalysisError("manifest-joined game_id is not canonical decimal")
        by_game[row["game_id"]].append(row)
    if set(by_game) != manifest["expected_audited_game_ids"]:
        raise AuditAnalysisError("audit game coverage does not exactly match arena manifest")
    for game_id, game_rows in by_game.items():
        game = manifest["clean_games"][game_id]
        if len(game_rows) != game["expected_decisions"]:
            raise AuditAnalysisError(f"arena game {game_id} decision coverage differs")
        for row in game_rows:
            turn = row["turn_index"]
            if (
                row["candidate_player"] != game["candidate_player"]
                or row["winner"] != game["winner"]
                or row["color"] != game["color"]
                or row["result"] != game["result"]
                or turn >= len(game["turns"])
                or row["transcript_prefix"] != "/".join(game["turns"][:turn])
                or row["actual_action"] != game["turns"][turn]
            ):
                raise AuditAnalysisError(f"arena game {game_id} audit context differs")
    result = dict(dataset)
    result["arena_manifest"] = manifest
    return result


def _arena_results(manifest: dict[str, Any]) -> dict[str, Any]:
    games = list(manifest["clean_games"].values())
    cohorts = {
        "top_5": lambda rank: rank is not None and rank <= 5,
        "top_10": lambda rank: rank is not None and rank <= 10,
        "top_20": lambda rank: rank is not None and rank <= 20,
    }
    rank_results = {
        name: _game_results(
            game for game in games if predicate(game["opponent_frozen_rank"])
        )
        for name, predicate in cohorts.items()
    }
    named: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for game in games:
        if game["opponent_name"]:
            named[(game["opponent_agent_id"], game["opponent_name"])].append(game)
    named_results = []
    for (agent_id, name), opponent_games in sorted(
        named.items(), key=lambda item: (item[0][1].casefold(), item[0][0])
    ):
        named_results.append(
            {
                "agent_id": agent_id,
                "name": name,
                **_game_results(opponent_games),
            }
        )
    return {
        "all_clean": _game_results(games),
        "frozen_rank_cohorts": rank_results,
        "named_opponents": named_results,
        "zero_candidate_decision_game_ids": manifest["zero_decision_game_ids"],
        "counting_unit": "one clean manifest game",
    }


def analyze_dataset(dataset: dict[str, Any], label: str) -> dict[str, Any]:
    rows = dataset["rows"]
    games = {row["game_id"]: row["result"] for row in rows}
    game_counts = collections.Counter(games.values())
    report: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "label": label,
        "source": {"name": dataset["source_name"], "sha256": dataset["source_sha256"]},
        "input_provenance": dataset["provenance"],
        "run_configuration": dataset["configuration"],
        "represented_games": {
            "games": len(games),
            "wins": game_counts["win"],
            "losses": game_counts["loss"],
        },
        "overall": _compact_summary(rows),
        "breakdowns": {
            "by_result": _breakdown(rows, lambda row: row["result"]),
            "by_player": _breakdown(rows, lambda row: f"player_{row['candidate_player']}"),
            "by_own_decision_phase": {
                "first": _compact_summary([row for row in rows if row["own_decision_index"] == 0]),
                "later": _compact_summary([row for row in rows if row["own_decision_index"] > 0]),
            },
        },
        "first_divergence": _first_divergences(rows),
    }
    if "arena_manifest" in dataset:
        manifest = dataset["arena_manifest"]
        report["arena_manifest_join"] = {
            "name": manifest["source_name"],
            "sha256": manifest["manifest_sha256"],
            "coverage": "exact clean rule-terminal candidate decisions",
        }
        report["arena_game_results"] = _arena_results(manifest)
    return report


def _metric_comparison(
    baseline: Sequence[dict[str, Any]],
    hypothesis: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    before = sum(predicate(row) for row in baseline)
    after = sum(predicate(row) for row in hypothesis)
    return {
        "baseline": {"count": before, "rate": _rate(before, len(baseline))},
        "hypothesis": {"count": after, "rate": _rate(after, len(hypothesis))},
        "delta_count": after - before,
        "delta_rate": round(_rate(after, len(hypothesis)) - _rate(before, len(baseline)), 6),
    }


def compare_datasets(
    baseline: dict[str, Any],
    hypothesis: dict[str, Any],
    baseline_label: str,
    hypothesis_label: str,
) -> dict[str, Any]:
    if baseline["provenance"] != hypothesis["provenance"]:
        raise AuditAnalysisError("comparison audits have different provenance")
    if ("arena_manifest" in baseline) != ("arena_manifest" in hypothesis):
        raise AuditAnalysisError("comparison audits have inconsistent arena joins")
    if "arena_manifest" in baseline and baseline["arena_manifest"]["manifest_sha256"] != hypothesis["arena_manifest"]["manifest_sha256"]:
        raise AuditAnalysisError("comparison audits use different arena manifests")
    before_by_key = {(row["game_id"], row["state_id"]): row for row in baseline["rows"]}
    after_by_key = {(row["game_id"], row["state_id"]): row for row in hypothesis["rows"]}
    if set(before_by_key) != set(after_by_key):
        raise AuditAnalysisError("comparison audits have different game/state coverage")
    keys = sorted(before_by_key, key=lambda key: (key[0], before_by_key[key]["turn_index"], key[1]))
    context = (
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
    for key in keys:
        before = before_by_key[key]
        after = after_by_key[key]
        differing = [field for field in context if before[field] != after[field]]
        if differing:
            raise AuditAnalysisError(f"comparison state {key} context differs: {differing}")
    baseline_rows = [before_by_key[key] for key in keys]
    hypothesis_rows = [after_by_key[key] for key in keys]
    transitions = collections.Counter(
        f"{before['classification']} -> {after['classification']}"
        for before, after in zip(baseline_rows, hypothesis_rows)
    )
    rank_changes = collections.Counter()
    for before, after in zip(baseline_rows, hypothesis_rows):
        left, right = before["actual_initial_rank"], after["actual_initial_rank"]
        if left < 0 <= right:
            rank_changes["gained_retention"] += 1
        elif right < 0 <= left:
            rank_changes["lost_retention"] += 1
        elif left == right:
            rank_changes["same"] += 1
        elif right < left:
            rank_changes["improved"] += 1
        else:
            rank_changes["worsened"] += 1
    configuration_changes = {
        field: {"baseline": baseline["configuration"][field], "hypothesis": hypothesis["configuration"][field]}
        for field in CONFIGURATION_FIELDS
        if baseline["configuration"][field] != hypothesis["configuration"][field]
    }
    metrics = {
        "actual_boundary_retained": lambda row: row["actual_boundary_retained_ordinal"] >= 0,
        "search_deadline": lambda row: row["search_deadline_reached"],
        "search_tree_cap": lambda row: row["search_tree_cap_reached"],
        "search_expansion_cap": lambda row: row["search_expansion_cap_reached"],
        "search_generator_truncated": lambda row: row["search_generator_truncations"] > 0,
        "diagnostic_root_truncated": lambda row: row["diagnostic_root_truncations"] > 0,
    }
    report: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "comparison": {"baseline": baseline_label, "hypothesis": hypothesis_label},
        "aligned_decisions": len(keys),
        "configuration_changes": configuration_changes,
        "classification_transitions": dict(sorted(transitions.items())),
        "chosen_action_changes": sum(
            before["chosen_action"] != after["chosen_action"]
            for before, after in zip(baseline_rows, hypothesis_rows)
        ),
        "actual_initial_rank_changes": dict(sorted(rank_changes.items())),
        "metrics": {
            name: _metric_comparison(baseline_rows, hypothesis_rows, predicate)
            for name, predicate in metrics.items()
        },
        "baseline": analyze_dataset(baseline, baseline_label),
        "hypothesis": analyze_dataset(hypothesis, hypothesis_label),
    }
    return report


def _text_report(report: dict[str, Any]) -> str:
    if "comparison" in report:
        return (
            f"{report['comparison']['baseline']} -> {report['comparison']['hypothesis']}: "
            f"{report['aligned_decisions']} aligned decisions, "
            f"{report['chosen_action_changes']} chosen-action changes\n"
        )
    overall = report["overall"]
    games = report["represented_games"]
    lines = [
        f"{report['label']}: {games['games']} games, {overall['decisions']} decisions",
        f"represented W/L: {games['wins']}/{games['losses']}",
    ]
    for name, value in overall["classifications"].items():
        if value["count"]:
            lines.append(f"{name}: {value['count']} ({value['rate']:.3f})")
    if "arena_game_results" in report:
        arena = report["arena_game_results"]["all_clean"]
        lines.append(f"manifest clean W/L: {arena['wins']}/{arena['losses']}")
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--label", default="audit")
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--compare-label", default="hypothesis")
    parser.add_argument("--arena-manifest", type=pathlib.Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        baseline = load_audit(arguments.input)
        manifest = None
        if arguments.arena_manifest is not None:
            manifest = load_arena_manifest(arguments.arena_manifest)
            baseline = join_arena_manifest(baseline, manifest)
        if arguments.compare is None:
            report = analyze_dataset(baseline, arguments.label)
        else:
            hypothesis = load_audit(arguments.compare)
            if manifest is not None:
                hypothesis = join_arena_manifest(hypothesis, manifest)
            report = compare_datasets(
                baseline, hypothesis, arguments.label, arguments.compare_label
            )
        if arguments.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(_text_report(report), end="")
        return 0
    except (AuditAnalysisError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
