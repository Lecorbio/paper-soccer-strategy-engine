#!/usr/bin/env python3
"""Validate Compact Value-BFM opening banks and Rank-4 gate JSON receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
from collections.abc import Iterable
from typing import Any


BANK_SCHEMA = "papersoccer.compact-value-bfm-opening-bank.v1"
LEGACY_RESULT_SCHEMA = "papersoccer.compact-value-bfm-rank4-gate.v1"
RESULT_SCHEMA = "papersoccer.compact-value-bfm-rank4-gate.v2"
TRAJECTORY_SCHEMA = "papersoccer.compact-value-bfm-rank4-trajectories.v1"
SEARCH_PROFILE_ACTIVATION_SCHEMA = (
    "papersoccer.compact-value-bfm-search-profile-activation.v1"
)
SEARCH_PROFILE_ACTIVATION_AGGREGATE_SCHEMA = (
    "papersoccer.compact-value-bfm-search-profile-activation-aggregate.v1"
)
RANK4_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
SEARCH_PROFILES = {
    "standard-v1",
    "state-evaluation-cache-v1",
    "progressive-widening-v1",
    "subtree-reuse-v1",
}
SEARCH_INTERVENTION_COUNTERS = (
    "cache_probes",
    "cache_hits",
    "cache_misses",
    "widening_probes",
    "widening_restrictions",
    "widening_eligible",
    "widening_deferred",
    "reuse_probes",
    "reuse_hits",
    "reuse_misses",
    "reuse_rejections",
    "reused_children",
)
CACHE_COUNTERS = SEARCH_INTERVENTION_COUNTERS[:3]
WIDENING_COUNTERS = SEARCH_INTERVENTION_COUNTERS[3:7]
REUSE_COUNTERS = SEARCH_INTERVENTION_COUNTERS[7:]
ACTIVE_COUNTERS_BY_PROFILE = {
    "standard-v1": (),
    "state-evaluation-cache-v1": CACHE_COUNTERS,
    "progressive-widening-v1": WIDENING_COUNTERS,
    "subtree-reuse-v1": REUSE_COUNTERS,
}
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


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("ascii")


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


def _search_intervention(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(
            SEARCH_INTERVENTION_COUNTERS):
        raise ValueError(f"{field} fields mismatch")
    if any(isinstance(value[name], bool) or not isinstance(value[name], int)
           or value[name] < 0 for name in SEARCH_INTERVENTION_COUNTERS):
        raise ValueError(f"{field} has an invalid counter")
    if value["cache_hits"] + value["cache_misses"] != value["cache_probes"]:
        raise ValueError(f"{field} cache accounting mismatch")
    if (value["widening_restrictions"] > value["widening_probes"]
            or value["widening_deferred"] < value["widening_restrictions"]
            or (value["widening_probes"] == 0 and any(
                value[name] != 0 for name in WIDENING_COUNTERS[1:]
            ))):
        raise ValueError(f"{field} widening accounting mismatch")
    if (value["reuse_hits"] + value["reuse_misses"]
            + value["reuse_rejections"] != value["reuse_probes"]
            or value["reused_children"] < value["reuse_hits"]
            or (value["reuse_hits"] == 0) !=
            (value["reused_children"] == 0)):
        raise ValueError(f"{field} subtree-reuse accounting mismatch")
    return value


def _engine(value: object, field: str, *, candidate: bool,
            legacy: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not an object")
    required = {
        "decisions", "deadline_stops", "soft_overruns", "headroom_failures",
        "hard_timeouts", "work", "generated_children", "evaluated_children",
        "maximum_first_ms", "maximum_later_ms", "times_ms",
    }
    if candidate and not legacy:
        required.add("search_intervention")
    if set(value) != required:
        raise ValueError(f"{field} fields mismatch")
    integers = required - {
        "maximum_first_ms", "maximum_later_ms", "times_ms",
        "search_intervention",
    }
    if any(isinstance(value[name], bool) or not isinstance(value[name], int)
           or value[name] < 0 for name in integers):
        raise ValueError(f"{field} has an invalid counter")
    for name in ("maximum_first_ms", "maximum_later_ms"):
        sample = value[name]
        if (isinstance(sample, bool) or not isinstance(sample, (int, float))
                or not math.isfinite(sample) or sample < 0):
            raise ValueError(f"{field} has an invalid timing maximum")
    times = value["times_ms"]
    if (not isinstance(times, list) or len(times) != value["decisions"]
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) or item < 0 for item in times)):
        raise ValueError(f"{field} timing samples mismatch")
    if candidate and not legacy:
        _search_intervention(
            value["search_intervention"], f"{field} search intervention")
    return value


def _merge_engines(values: list[dict[str, Any]], *, candidate: bool,
                   legacy: bool = False) -> dict[str, Any]:
    result = {
        "decisions": 0, "deadline_stops": 0, "soft_overruns": 0,
        "headroom_failures": 0, "hard_timeouts": 0, "work": 0,
        "generated_children": 0, "evaluated_children": 0,
        "maximum_first_ms": 0.0, "maximum_later_ms": 0.0, "times_ms": [],
    }
    if candidate and not legacy:
        result["search_intervention"] = {
            name: 0 for name in SEARCH_INTERVENTION_COUNTERS
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
        if candidate and not legacy:
            for name in SEARCH_INTERVENTION_COUNTERS:
                result["search_intervention"][name] += (
                    value["search_intervention"][name]
                )
    return result


def _validate_profile_counter_scope(counters: dict[str, int], profile: str,
                                    field: str) -> None:
    active = ACTIVE_COUNTERS_BY_PROFILE[profile]
    inactive = set(SEARCH_INTERVENTION_COUNTERS) - set(active)
    if any(counters[name] != 0 for name in inactive):
        raise ValueError(f"{field} has counters from an inactive search profile")


def _candidate_profile_summary(document: object, *, expected_profile: str) -> (
        dict[str, Any]):
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise ValueError("search-profile activation requires a v2 gate result")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("gate config is missing")
    profile = config.get("candidate_search_profile")
    if profile not in SEARCH_PROFILES:
        raise ValueError("candidate search profile is invalid")
    if profile != expected_profile:
        raise ValueError("candidate search profile does not match expectation")
    games = document.get("games")
    result = document.get("result")
    if not isinstance(games, list) or not isinstance(result, dict):
        raise ValueError("gate result is missing games or summary")
    game_candidates = [
        _engine(game.get("candidate") if isinstance(game, dict) else None,
                "game candidate", candidate=True)
        for game in games
    ]
    for index, engine in enumerate(game_candidates):
        _validate_profile_counter_scope(
            engine["search_intervention"], profile,
            f"game {index} candidate search intervention",
        )
    candidate = _engine(
        result.get("candidate"), "result candidate", candidate=True)
    merged = _merge_engines(game_candidates, candidate=True)
    if candidate != merged:
        raise ValueError(
            "gate candidate search-profile aggregate does not reproduce games"
        )
    counters = candidate["search_intervention"]
    _validate_profile_counter_scope(
        counters, profile, "result candidate search intervention")
    return candidate


def _profile_exercise_requirements(
    profile: str, *, decisions: int, counters: dict[str, int],
) -> dict[str, bool]:
    requirements = {
        "candidate_decisions_positive": decisions > 0,
        "inactive_profile_counters_zero": all(
            counters[name] == 0
            for name in set(SEARCH_INTERVENTION_COUNTERS)
            - set(ACTIVE_COUNTERS_BY_PROFILE[profile])
        ),
    }
    if profile == "standard-v1":
        requirements["all_intervention_counters_zero"] = all(
            counters[name] == 0 for name in SEARCH_INTERVENTION_COUNTERS
        )
    elif profile == "state-evaluation-cache-v1":
        requirements.update({
            "cache_probes_positive": counters["cache_probes"] > 0,
            "cache_hits_positive": counters["cache_hits"] > 0,
        })
    elif profile == "progressive-widening-v1":
        requirements.update({
            "widening_probes_positive": counters["widening_probes"] > 0,
            "widening_restrictions_positive": (
                counters["widening_restrictions"] > 0
            ),
            "widening_eligible_positive": counters["widening_eligible"] > 0,
            "widening_deferred_positive": counters["widening_deferred"] > 0,
        })
    else:
        requirements.update({
            "reuse_probes_positive": counters["reuse_probes"] > 0,
            "reuse_hits_positive": counters["reuse_hits"] > 0,
            "reused_children_positive": counters["reused_children"] > 0,
        })
    return requirements


def _require_exercise(requirements: dict[str, bool]) -> None:
    if not all(requirements.values()):
        failed = sorted(name for name, passed in requirements.items() if not passed)
        raise ValueError(
            "candidate search profile was not exercised: " + ", ".join(failed)
        )


def require_search_profile_exercised(
    document: object, *, expected_profile: str | None = None,
) -> dict[str, Any]:
    """Return deterministic evidence that a validated v2 gate exercised its profile.

    This helper deliberately does not accept legacy attempt-zero results: those
    receipts predate the counters and cannot prove intervention activation.
    Call :func:`validate_result` first for the complete gate contract.
    """

    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise ValueError("search-profile activation requires a v2 gate result")
    config = document.get("config")
    profile = config.get("candidate_search_profile") \
        if isinstance(config, dict) else None
    if profile not in SEARCH_PROFILES:
        raise ValueError("candidate search profile is invalid")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("candidate search profile does not match expectation")
    candidate = _candidate_profile_summary(
        document, expected_profile=profile)
    counters = candidate["search_intervention"]
    requirements = _profile_exercise_requirements(
        profile, decisions=candidate["decisions"], counters=counters)
    _require_exercise(requirements)
    return {
        "schema": SEARCH_PROFILE_ACTIVATION_SCHEMA,
        "candidate_search_profile": profile,
        "candidate_decisions": candidate["decisions"],
        "search_intervention": dict(counters),
        "requirements": requirements,
        "exercised": True,
    }


def aggregate_search_profile_activation(
    documents: Iterable[object], expected_profile: str, *,
    require_exercised: bool = True,
) -> dict[str, Any]:
    """Seal aggregate activation evidence across already validated v2 shards.

    Individual shards need only bind the same compile-time profile and contain
    internally consistent counters. Profile-specific effects are required from
    the aggregate, so one shard may legitimately contain no hit or restriction.
    """

    if not isinstance(require_exercised, bool):
        raise ValueError("search-profile exercise policy must be boolean")
    if expected_profile not in SEARCH_PROFILES:
        raise ValueError("candidate search profile is invalid")
    if isinstance(documents, (str, bytes, bytearray, dict)):
        raise ValueError("search-profile activation documents must be nonempty")
    try:
        shards = tuple(documents)
    except TypeError as error:
        raise ValueError(
            "search-profile activation documents must be nonempty"
        ) from error
    if not shards:
        raise ValueError("search-profile activation documents must be nonempty")
    counters = {name: 0 for name in SEARCH_INTERVENTION_COUNTERS}
    decisions = 0
    for document in shards:
        candidate = _candidate_profile_summary(
            document, expected_profile=expected_profile)
        decisions += candidate["decisions"]
        for name in SEARCH_INTERVENTION_COUNTERS:
            counters[name] += candidate["search_intervention"][name]
    _validate_profile_counter_scope(
        counters, expected_profile,
        "aggregate candidate search intervention",
    )
    requirements = _profile_exercise_requirements(
        expected_profile, decisions=decisions, counters=counters)
    exercised = all(requirements.values())
    if require_exercised:
        _require_exercise(requirements)
    body = {
        "schema": SEARCH_PROFILE_ACTIVATION_AGGREGATE_SCHEMA,
        "candidate_search_profile": expected_profile,
        "document_count": len(shards),
        "candidate_decisions": decisions,
        "search_intervention": counters,
        "requirements": requirements,
        "exercised": exercised,
    }
    sealed = dict(body)
    sealed["body_sha256"] = hashlib.sha256(
        _canonical_json_bytes(body)).hexdigest()
    return sealed


def legacy_standard_configuration(document: object) -> dict[str, Any]:
    """Project a validated v1/v2 standard gate onto the historical config."""

    if not isinstance(document, dict) or document.get("schema") not in {
        LEGACY_RESULT_SCHEMA, RESULT_SCHEMA
    }:
        raise ValueError("legacy configuration requires a Rank-4 gate result")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("legacy configuration gate config is missing")
    result = dict(config)
    if document["schema"] == RESULT_SCHEMA:
        if result.pop("candidate_search_profile", None) != "standard-v1":
            raise ValueError(
                "legacy runner cannot consume an intervention search profile"
            )
    elif "candidate_search_profile" in result:
        raise ValueError("legacy gate unexpectedly contains a search profile")
    return result


def _validate_trajectories(document: dict[str, Any], bank_path: pathlib.Path) -> None:
    """Replay only accepted turns, retaining the pre-response state on failure."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
    from tools import jacek_replay_features as features

    if (document['schema'] != RESULT_SCHEMA
            or document['config'].get('trajectory_schema') != TRAJECTORY_SCHEMA):
        raise ValueError('gate did not request source-bound trajectories')
    bank = validate_bank(bank_path)
    if (bank['sha256'] != document['bindings']['bank_sha256']
            or bank['bytes'] != document['bindings']['bank_bytes']):
        raise ValueError('trajectory bank differs from the executed bank')
    for game in document['games']:
        index = game['pair_index']
        if not 0 <= index < len(bank['openings']):
            raise ValueError('trajectory opening index exceeds the bank')
        root = bank['openings'][index]
        transcript = game.get('transcript')
        if (game.get('opening_id') != root['opening_id']
                or game.get('root_transcript') != root['transcript']
                or not isinstance(transcript, str)
                or not re.fullmatch(r'[0-7]+(?:/[0-7]+)*', transcript)):
            raise ValueError('trajectory lost its exact frozen root or encoding')
        if hashlib.sha256(transcript.encode('ascii')).hexdigest() != game.get('transcript_sha256'):
            raise ValueError('trajectory transcript SHA-256 differs')
        actions = transcript.split('/')
        prefix = root['transcript'].split('/')
        if (actions[:len(prefix)] != prefix or isinstance(game['turns'], bool)
                or not isinstance(game['turns'], int) or len(actions) != game['turns']):
            raise ValueError('trajectory prefix or complete-turn count differs')
        state = features.ReplayState()
        decisions = {'candidate': 0, 'rank4': 0}
        for turn, action in enumerate(actions):
            if turn == len(prefix) and state.winner is not None:
                raise ValueError('trajectory root is terminal')
            if turn >= len(prefix):
                actor = 'candidate' if state.to_move == game['candidate_player'] else 'rank4'
                decisions[actor] += 1
            features.apply_complete_turn(state, state.to_move, action)
        if len(actions) == len(prefix) and state.winner is not None:
            raise ValueError('trajectory root is terminal')
        winner = state.winner if state.winner is not None else -1
        if game.get('winner') != winner:
            raise ValueError('trajectory winner differs from legal replay')
        failure = game['failure']
        if failure is None:
            if winner not in (0, 1):
                raise ValueError('successful trajectory is not terminal')
        else:
            if winner != -1:
                raise ValueError('failed trajectory contains an unaccepted terminal action')
            if failure != 'unfinished':
                actor = 'candidate' if state.to_move == game['candidate_player'] else 'rank4'
                if failure != 'lockstep_mismatch' and not failure.startswith(actor + '_'):
                    raise ValueError('trajectory failure belongs to a different actor')
                decisions[actor] += 1
        if any(game[actor]['decisions'] != count for actor, count in decisions.items()):
            raise ValueError('trajectory actor decisions do not reproduce the result')


def validate_result(path: pathlib.Path, *, expected_bank_sha256: str | None = None,
                    expected_candidate_sha256: str | None = None,
                    expected_candidate_search_profile: str | None = None,
                    allow_legacy_attempt_zero: bool = False,
                    require_trajectories: bool = False,
                    trajectory_bank: pathlib.Path | None = None) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema") not in {
        LEGACY_RESULT_SCHEMA, RESULT_SCHEMA
    }:
        raise ValueError("unexpected Rank-4 gate result schema")
    legacy = document["schema"] == LEGACY_RESULT_SCHEMA
    if legacy and not allow_legacy_attempt_zero:
        raise ValueError(
            "legacy Rank-4 gate result requires explicit attempt-zero compatibility"
        )
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
    profile = config.get("candidate_search_profile")
    if legacy:
        if profile is not None:
            raise ValueError("legacy attempt-zero gate unexpectedly binds a profile")
        if expected_candidate_search_profile is not None:
            raise ValueError(
                "legacy attempt-zero gate cannot prove a candidate search profile"
            )
        profile = "standard-v1"
    elif profile not in SEARCH_PROFILES:
        raise ValueError("candidate search profile is invalid")
    if (expected_candidate_search_profile is not None
            and profile != expected_candidate_search_profile):
        raise ValueError("candidate search profile does not match expectation")
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
        candidate_engine = _engine(
            game.get("candidate"), "game candidate",
            candidate=True, legacy=legacy,
        )
        rank4_engine = _engine(
            game.get("rank4"), "game Rank-4",
            candidate=False, legacy=legacy,
        )
        if not legacy:
            _validate_profile_counter_scope(
                candidate_engine["search_intervention"], profile,
                "game candidate search intervention",
            )
        candidate_engines.append(candidate_engine)
        rank4_engines.append(rank4_engine)
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
    candidate_summary = _engine(
        result.get("candidate"), "result candidate",
        candidate=True, legacy=legacy,
    )
    rank4_summary = _engine(
        result.get("rank4"), "result Rank-4",
        candidate=False, legacy=legacy,
    )
    if not legacy:
        _validate_profile_counter_scope(
            candidate_summary["search_intervention"], profile,
            "result candidate search intervention",
        )
    if (candidate_summary != _merge_engines(
            candidate_engines, candidate=True, legacy=legacy) or
            rank4_summary != _merge_engines(
                rank4_engines, candidate=False, legacy=legacy)):
        raise ValueError("gate engine aggregates do not reproduce games")
    expected_passed = not failures
    if minimum_wins >= 0:
        expected_passed = expected_passed and candidate_wins >= minimum_wins
    if minimum_per_color >= 0:
        expected_passed = expected_passed and min(wins_by_color) >= minimum_per_color
    if result.get("passed") is not expected_passed:
        raise ValueError("gate pass/fail decision does not match configured thresholds")
    if require_trajectories:
        if trajectory_bank is None:
            raise ValueError('trajectory validation requires the executed bank')
        _validate_trajectories(document, trajectory_bank)
    elif trajectory_bank is not None:
        raise ValueError('trajectory bank requires explicit trajectory validation')
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
    result_parser.add_argument("--expected-candidate-search-profile")
    result_parser.add_argument("--require-trajectories", action="store_true")
    result_parser.add_argument("--trajectory-bank", type=pathlib.Path)
    result_parser.add_argument(
        "--allow-legacy-attempt-zero", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "validate-bank":
        value = validate_bank(arguments.bank)
    else:
        value = validate_result(
            arguments.result,
            expected_bank_sha256=arguments.expected_bank_sha256,
            expected_candidate_sha256=arguments.expected_candidate_sha256,
            expected_candidate_search_profile=(
                arguments.expected_candidate_search_profile
            ),
            allow_legacy_attempt_zero=arguments.allow_legacy_attempt_zero,
            require_trajectories=arguments.require_trajectories,
            trajectory_bank=arguments.trajectory_bank,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
