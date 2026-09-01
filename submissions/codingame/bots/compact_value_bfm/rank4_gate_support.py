#!/usr/bin/env python3
"""Validate Compact Value-BFM opening banks and Rank-4 gate JSON receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any


BANK_SCHEMA = "papersoccer.compact-value-bfm-opening-bank.v1"
RESULT_SCHEMA = "papersoccer.compact-value-bfm-rank4-gate.v1"
RANK4_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
FAILURES = {
    "candidate_exception",
    "rank4_exception",
    "candidate_malformed",
    "rank4_malformed",
    "candidate_illegal",
    "rank4_illegal",
    "candidate_timeout",
    "rank4_timeout",
    "lockstep_mismatch",
    "unfinished",
}
ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bank(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError("opening bank must use LF line endings")
    text = raw.decode("ascii")
    header = False
    ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        if not header:
            if line != "opening_id\ttranscript":
                raise ValueError("opening bank header mismatch")
            header = True
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"line {line_number}: expected two fields")
        opening_id, transcript = fields
        if not ID.fullmatch(opening_id) or opening_id in ids:
            raise ValueError(f"line {line_number}: invalid or duplicate opening id")
        actions = transcript.split("/")
        if any(not action or not set(action) <= set("01234567") for action in actions):
            raise ValueError(f"line {line_number}: malformed complete-turn transcript")
        physical_plies = sum(map(len, actions))
        if physical_plies < 12:
            raise ValueError(f"line {line_number}: opening has fewer than 12 physical plies")
        ids.add(opening_id)
        rows.append({
            "opening_id": opening_id,
            "transcript": transcript,
            "complete_turns": len(actions),
            "physical_plies": physical_plies,
        })
    if not header or not rows:
        raise ValueError("opening bank is empty")
    return {
        "schema": BANK_SCHEMA,
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "openings": rows,
    }


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


def _engine(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not an object")
    required = {
        "decisions", "deadline_stops", "soft_overruns", "headroom_failures",
        "hard_timeouts", "work", "generated_children", "evaluated_children",
        "maximum_first_ms", "maximum_later_ms", "times_ms",
    }
    if set(value) != required:
        raise ValueError(f"{field} fields mismatch")
    integers = required - {"maximum_first_ms", "maximum_later_ms", "times_ms"}
    if any(isinstance(value[name], bool) or not isinstance(value[name], int)
           or value[name] < 0 for name in integers):
        raise ValueError(f"{field} has an invalid counter")
    times = value["times_ms"]
    if (not isinstance(times, list) or len(times) != value["decisions"]
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or item < 0 for item in times)):
        raise ValueError(f"{field} timing samples mismatch")
    return value


def _merge_engines(values: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "decisions": 0, "deadline_stops": 0, "soft_overruns": 0,
        "headroom_failures": 0, "hard_timeouts": 0, "work": 0,
        "generated_children": 0, "evaluated_children": 0,
        "maximum_first_ms": 0.0, "maximum_later_ms": 0.0, "times_ms": [],
    }
    for value in values:
        for field in (
            "decisions", "deadline_stops", "soft_overruns", "headroom_failures",
            "hard_timeouts", "work", "generated_children", "evaluated_children",
        ):
            result[field] += value[field]
        result["maximum_first_ms"] = max(
            result["maximum_first_ms"], value["maximum_first_ms"])
        result["maximum_later_ms"] = max(
            result["maximum_later_ms"], value["maximum_later_ms"])
        result["times_ms"].extend(value["times_ms"])
    return result


def validate_result(path: pathlib.Path, *, expected_bank_sha256: str | None = None,
                    expected_candidate_sha256: str | None = None) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected Rank-4 gate result schema")
    if set(document) != {"schema", "bindings", "config", "games", "result"}:
        raise ValueError("Rank-4 gate top-level fields mismatch")
    bindings = document["bindings"]
    if not isinstance(bindings, dict):
        raise ValueError("gate bindings are missing")
    candidate_sha = _sha(bindings.get("candidate_source_sha256"), "candidate source")
    rank4_sha = _sha(bindings.get("rank4_source_sha256"), "Rank-4 source")
    opponent_sha = _sha(bindings.get("opponent_sha256"), "opponent")
    bank_sha = _sha(bindings.get("bank_sha256"), "bank")
    _sha(bindings.get("candidate_runtime_body_sha256"), "runtime body")
    _sha(bindings.get("candidate_payload_sha256"), "payload")
    if rank4_sha != RANK4_SHA256 or opponent_sha != RANK4_SHA256:
        raise ValueError("gate does not bind the exact maintained Rank-4 source")
    if expected_bank_sha256 and bank_sha != expected_bank_sha256:
        raise ValueError("gate bank SHA-256 mismatch")
    if expected_candidate_sha256 and candidate_sha != expected_candidate_sha256:
        raise ValueError("gate candidate SHA-256 mismatch")
    config = document["config"]
    if not isinstance(config, dict) or config.get("mode") not in {
        "fixed-work", "actual-clock"
    }:
        raise ValueError("gate config mode is invalid")
    if config.get("candidate_clocks_ms") != [800, 155] or \
            config.get("rank4_clocks_ms") != [800, 165] or \
            not 1 <= config.get("max_turns", 0) <= 320:
        raise ValueError("gate clock/turn contract mismatch")
    pair_offset = config.get("pair_offset")
    pair_count = config.get("pair_count")
    minimum_wins = config.get("minimum_candidate_wins")
    minimum_per_color = config.get("minimum_wins_per_color")
    if (isinstance(pair_offset, bool) or not isinstance(pair_offset, int)
            or pair_offset < 0 or isinstance(pair_count, bool)
            or not isinstance(pair_count, int) or pair_count <= 0
            or isinstance(minimum_wins, bool) or not isinstance(minimum_wins, int)
            or minimum_wins < -1 or isinstance(minimum_per_color, bool)
            or not isinstance(minimum_per_color, int) or minimum_per_color < -1):
        raise ValueError("gate pair range is invalid")
    games = document["games"]
    if not isinstance(games, list) or len(games) != pair_count * 2:
        raise ValueError("gate must contain exactly two games per pair")
    identities: set[tuple[int, int]] = set()
    failures: dict[str, int] = {}
    candidate_wins = 0
    wins_by_color = [0, 0]
    rank4_wins = 0
    unfinished = 0
    candidate_engines = []
    rank4_engines = []
    for game in games:
        if not isinstance(game, dict):
            raise ValueError("gate game is not an object")
        pair = game.get("pair_index")
        color = game.get("candidate_player")
        if (isinstance(pair, bool) or not isinstance(pair, int)
                or not pair_offset <= pair < pair_offset + pair_count
                or color not in (0, 1) or (pair, color) in identities):
            raise ValueError("gate game pair/color identity mismatch")
        identities.add((pair, color))
        if not 0 <= game.get("turns", -1) <= config["max_turns"]:
            raise ValueError("gate game turn count is invalid")
        failure = game.get("failure")
        if failure is not None:
            if failure not in FAILURES:
                raise ValueError("gate game failure category is unknown")
            failures[failure] = failures.get(failure, 0) + 1
            unfinished += failure == "unfinished"
        elif game.get("winner") == color:
            candidate_wins += 1
            wins_by_color[color] += 1
        else:
            rank4_wins += 1
        candidate_engines.append(_engine(game.get("candidate"), "game candidate"))
        rank4_engines.append(_engine(game.get("rank4"), "game Rank-4"))
    result = document["result"]
    if not isinstance(result, dict):
        raise ValueError("gate result is missing")
    if (result.get("games") != len(games)
            or result.get("candidate_wins") != candidate_wins
            or result.get("candidate_wins_player0") != wins_by_color[0]
            or result.get("candidate_wins_player1") != wins_by_color[1]
            or result.get("rank4_wins") != rank4_wins
            or result.get("failures") != sum(failures.values())
            or result.get("unfinished") != unfinished
            or result.get("failure_categories") != failures):
        raise ValueError("gate aggregate does not reproduce its games")
    if (_engine(result.get("candidate"), "result candidate") !=
            _merge_engines(candidate_engines) or
            _engine(result.get("rank4"), "result Rank-4") !=
            _merge_engines(rank4_engines)):
        raise ValueError("gate engine aggregates do not reproduce game timings")
    expected_passed = not failures
    if minimum_wins >= 0:
        expected_passed = expected_passed and candidate_wins >= minimum_wins
    if minimum_per_color >= 0:
        expected_passed = expected_passed and min(wins_by_color) >= minimum_per_color
    if result.get("passed") is not expected_passed:
        raise ValueError("gate pass/fail decision does not match configured thresholds")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bank_parser = subparsers.add_parser("validate-bank")
    bank_parser.add_argument("--bank", type=pathlib.Path, required=True)
    result_parser = subparsers.add_parser("validate-result")
    result_parser.add_argument("--result", type=pathlib.Path, required=True)
    result_parser.add_argument("--expected-bank-sha256")
    result_parser.add_argument("--expected-candidate-sha256")
    arguments = parser.parse_args()
    if arguments.command == "validate-bank":
        value = validate_bank(arguments.bank)
    else:
        value = validate_result(
            arguments.result,
            expected_bank_sha256=arguments.expected_bank_sha256,
            expected_candidate_sha256=arguments.expected_candidate_sha256,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
