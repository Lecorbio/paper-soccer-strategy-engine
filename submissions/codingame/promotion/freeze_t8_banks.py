#!/usr/bin/env python3

"""Freeze candidate-independent T8 validation and sealed-final banks."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib

from acquire_t7_evidence import SOURCE_AGENTS as T7_SOURCE_AGENTS
from acquire_t8_evidence import (
    PARSEABLE_RECORDS,
    RAW_RECORDS,
    SOURCE_AGENTS,
    T7_FINAL,
    T7_FINAL_SHA256,
    T7_VALIDATION,
    T7_VALIDATION_SHA256,
    t8_exclusions,
)
from build_goal_shell_banks import (
    ROOT,
    balanced_sample,
    extract_states,
    sha256_bytes,
    stable_json,
    tsv_bytes,
)
from freeze_t7_banks import prior_records_and_keys


HERE = pathlib.Path(__file__).resolve().parent
VALIDATION_BANK = HERE / "reference" / "t8_prospective_validation.tsv"
FINAL_BANK = HERE / "reference" / "t8_sealed_final.tsv"
EVIDENCE_MANIFEST = HERE / "reference" / "t8_evidence_manifest.json"
RAW_SHA256 = "c2ee19e042dad13bb67aa1a47c4c768fa8233ad8d762d17783405938b0a824c5"
PARSEABLE_SHA256 = (
    "780ddfc963fff710cacbbe0083f6ed87cd1f58888a9e85cdfe691645872e1f50"
)
EXPECTED_VALIDATION_SHA256 = (
    "e670fc39902308b66debd8deb3dc82fb9e0ce0f61562b3078328e99350c54f3b"
)
EXPECTED_FINAL_SHA256 = (
    "e6c8efaa094576ad4ac3dc22a69ea595f224aaa64d7c3ecdc39b7e98c7dfb204"
)

PUBLIC_ELITE_SCORE_CUTOFF = 35.0
GAME_CAP = 2
GAME_DISTANCE_CAP = 1
FOCUS_AGENT_CAP = 18
MIN_ELITE_ROWS = 12
MIN_FIELD_ROWS = 40
MIN_DISTINCT_FOCUS_AGENTS = 12
DISTANCE_MINIMUMS = {"d0": 4, "d1": 24, "d2": 24}

EXPECTED_RAW_RECORDS = 919
EXPECTED_PARSEABLE_RECORDS = 791
EXPECTED_STRUCTURALLY_REJECTED = 128
EXPECTED_PRIOR_CANONICAL_KEYS = 3_339
EXPECTED_NO_SHELL_GAMES = 582
EXPECTED_PRIOR_KEY_OVERLAP_GAMES = 79
EXPECTED_UNIQUE_STATES = 358
EXPECTED_UNIQUE_GAMES = 114
EXPECTED_PARTITION_STATES = {"validation": 177, "final": 181}
EXPECTED_PARTITION_GAMES = {"validation": 56, "final": 58}
EXPECTED_COMPOSITION = {
    "validation": {
        "source_games": 50,
        "focus_agents": 19,
        "maximum_focus_rows": 14,
        "winner_tiers": {"elite": 20, "field": 52},
        "distance": {"d0": 5, "d1": 34, "d2": 33},
    },
    "final": {
        "source_games": 49,
        "focus_agents": 19,
        "maximum_focus_rows": 13,
        "winner_tiers": {"elite": 19, "field": 53},
        "distance": {"d0": 7, "d1": 30, "d2": 35},
    },
}

TIME_PROFILE_MAX_NODES = 3_000_000
SHARD_COUNT = 4
INCUMBENT_SUBMISSION_SHA256 = (
    "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
)
PROSPECTIVE_STRENGTH_PROTOCOL = {
    "status": "predeclared_before_candidate_binding_and_outcomes",
    "stage_bank_policy": (
        "validation and final each use one common bank for every profile; the "
        "two banks are disjoint by source game and canonical state"
    ),
    "game_format": {
        "games_per_opening": 3,
        "candidate_games": "candidate versus rank5 once as each physical player",
        "control_game": "rank5 versus rank5 on the same opening and budget",
    },
    "execution_contract": {
        "candidate_runner_binding": (
            "the future active candidate manifest must hash-pin its comparison "
            "runner before either T8 stage runs"
        ),
        "incumbent_submission_sha256": INCUMBENT_SUBMISSION_SHA256,
        "common_shard_count": SHARD_COUNT,
        "common_jobs": SHARD_COUNT,
        "sharding": (
            "every profile in validation and final uses exactly four raw-evidence "
            "shards and a four-worker pool"
        ),
        "within_opening_order": (
            "the two candidate-color games and same-opening control run "
            "sequentially in one comparison-runner process"
        ),
        "time_budget_clock": "steady-clock wall time",
        "time_budget_scope": (
            "construction-inclusive: candidate bot construction and search both "
            "consume the 130ms wall-clock allowance"
        ),
        "time_profile_node_cap": TIME_PROFILE_MAX_NODES,
        "reason_for_four_shards": (
            "limits concurrent table initialization and scheduler/memory noise "
            "relative to the deployment timing profile"
        ),
    },
    "runner_projection": {
        "nodes": {"node_budget": "profile.value", "time_budget_ms": 0},
        "time_ms": {
            "node_budget": "profile.max_nodes",
            "time_budget_ms": "profile.value",
        },
    },
    "statistics": {
        "confidence": 0.95,
        "method": "source_game_cluster_percentile_bootstrap",
        "resamples": 10_000,
        "seed": 8_028_218_941_123_341_919,
        "unit": "source_game_cluster_of_color_swapped_opening_pairs",
    },
    "diagnostic_only": [
        "historical-role absolute score and uplift partitions",
        "time-profile candidate_to_incumbent_throughput",
    ],
    "stages": {
        "initial": {
            "evidence_status": "exposed_adaptive_not_prospective",
            "bank_policy": (
                "reuse the already exposed frozen initial bank at future "
                "candidate binding"
            ),
            "configuration": {
                "maximum_turns": 320,
                "minimum_mean": 0.5,
                "node_budget": 5_000,
            },
        },
        "development": {
            "evidence_status": "exposed_adaptive_not_prospective",
            "bank_policy": (
                "reuse the exposed frozen T3 development bank at future "
                "candidate binding"
            ),
            "bank": "reference/t3_development.tsv",
            "configuration": {
                "maximum_turns": 320,
                "minimum_mean": 0.52,
                "minimum_throughput_ratio": 0.9,
                "node_budgets": [5_000, 30_000],
                "node_budget_overrides": {
                    "5000": {
                        "minimum_mean": 0.5,
                        "require_at_least_as_many_wins_as_incumbent": True,
                        "require_more_wins_than_incumbent": False,
                    }
                },
                "require_more_wins_than_incumbent": True,
            },
        },
        "validation": {
            "bank": str(VALIDATION_BANK.relative_to(HERE)),
            "shard_count": SHARD_COUNT,
            "jobs": SHARD_COUNT,
            "configuration": {
                "required_jobs": SHARD_COUNT,
                "maximum_turns": 320,
                "minimum_mean": 0.51,
                "minimum_ci_lower": 0.47,
                "minimum_physical_color_uplift": -0.05,
                "minimum_control_winner_retention": 0.8,
                "minimum_stratum_score": 0.48,
                "minimum_winner_tier_score": 0.48,
                "minimum_elite_tier_score": 0.48,
                "minimum_throughput_ratio": 0.9,
                "strength_profiles": [
                    {"id": "30k-nodes", "mode": "nodes", "value": 30_000},
                    {
                        "id": "130ms",
                        "mode": "time_ms",
                        "value": 130,
                        "max_nodes": TIME_PROFILE_MAX_NODES,
                        "thresholds": {"minimum_throughput_ratio": None},
                    },
                ],
            },
        },
        "test": {
            "bank": str(FINAL_BANK.relative_to(HERE)),
            "shard_count": SHARD_COUNT,
            "jobs": SHARD_COUNT,
            "configuration": {
                "required_jobs": SHARD_COUNT,
                "maximum_turns": 320,
                "minimum_mean": 0.5,
                "minimum_ci_lower": 0.45,
                "minimum_physical_color_uplift": -0.05,
                "minimum_control_winner_retention": 0.75,
                "minimum_stratum_score": 0.45,
                "minimum_winner_tier_score": 0.4,
                "minimum_elite_tier_score": 0.45,
                "minimum_throughput_ratio": 0.9,
                "strength_profiles": [
                    {"id": "100k-nodes", "mode": "nodes", "value": 100_000},
                    {
                        "id": "130ms",
                        "mode": "time_ms",
                        "value": 130,
                        "max_nodes": TIME_PROFILE_MAX_NODES,
                        "thresholds": {"minimum_throughput_ratio": None},
                    },
                ],
            },
        },
    },
}
EXPECTED_STRENGTH_PROTOCOL_SHA256 = (
    "8d73a1c92d43d73a8ebe48a63084f5f5c578ead9516058920c0700165ec3851c"
)


def assert_strength_protocol() -> None:
    actual_hash = sha256_bytes(stable_json(PROSPECTIVE_STRENGTH_PROTOCOL))
    if EXPECTED_STRENGTH_PROTOCOL_SHA256 and (
        actual_hash != EXPECTED_STRENGTH_PROTOCOL_SHA256
    ):
        raise RuntimeError("T8 prospective strength protocol hash changed")
    initial = PROSPECTIVE_STRENGTH_PROTOCOL["stages"]["initial"]
    development = PROSPECTIVE_STRENGTH_PROTOCOL["stages"]["development"]
    if initial["evidence_status"] != "exposed_adaptive_not_prospective":
        raise RuntimeError("T8 initial evidence status changed")
    if initial["configuration"] != {
        "maximum_turns": 320,
        "minimum_mean": 0.5,
        "node_budget": 5_000,
    }:
        raise RuntimeError("T8 initial configuration changed")
    if development["evidence_status"] != "exposed_adaptive_not_prospective":
        raise RuntimeError("T8 development evidence status changed")
    expected_development = {
        "maximum_turns": 320,
        "minimum_mean": 0.52,
        "minimum_throughput_ratio": 0.9,
        "node_budgets": [5_000, 30_000],
        "node_budget_overrides": {
            "5000": {
                "minimum_mean": 0.5,
                "require_at_least_as_many_wins_as_incumbent": True,
                "require_more_wins_than_incumbent": False,
            }
        },
        "require_more_wins_than_incumbent": True,
    }
    if development["configuration"] != expected_development:
        raise RuntimeError("T8 development configuration changed")
    expected_profiles = {
        "validation": [("30k-nodes", "nodes", 30_000), ("130ms", "time_ms", 130)],
        "test": [("100k-nodes", "nodes", 100_000), ("130ms", "time_ms", 130)],
    }
    for stage, identities in expected_profiles.items():
        declaration = PROSPECTIVE_STRENGTH_PROTOCOL["stages"][stage]
        if (
            declaration["shard_count"] != SHARD_COUNT
            or declaration["jobs"] != SHARD_COUNT
            or declaration["configuration"].get("required_jobs") != SHARD_COUNT
        ):
            raise RuntimeError(f"T8 {stage} must use exactly four workers/shards")
        profiles = declaration["configuration"]["strength_profiles"]
        actual = [(item["id"], item["mode"], item["value"]) for item in profiles]
        if actual != identities:
            raise RuntimeError(f"T8 {stage} profile identities changed")
        time_profile = profiles[1]
        if time_profile.get("max_nodes") != TIME_PROFILE_MAX_NODES:
            raise RuntimeError(f"T8 {stage} time-profile node cap changed")
        if time_profile.get("thresholds", {}).get(
            "minimum_throughput_ratio", "missing"
        ) is not None:
            raise RuntimeError("T8 time-profile throughput must remain diagnostic")


def winner(record: dict) -> int:
    return (
        int(record["player_id"])
        if record.get("won")
        else 1 - int(record["player_id"])
    )


def t7_tier_metadata():
    path = HERE / "reference" / "t7_evidence_manifest.json"
    payload = json.loads(path.read_text())
    entries = payload["selection"]["tier_taxonomy"]["frozen_focus_scores"]
    scores = {int(item["agent_id"]): float(item["score"]) for item in entries}
    public_elite_names = {
        name
        for agent_id, name in T7_SOURCE_AGENTS
        if scores.get(agent_id, float("-inf")) >= PUBLIC_ELITE_SCORE_CUTOFF
    }
    expected = {"About", "Aketchan", "TetraktysPhi", "field3", "mokaspark"}
    if public_elite_names != expected:
        raise RuntimeError("T8 public-elite name taxonomy changed")
    return public_elite_names, scores, path


def prior_keys_and_games():
    keys = prior_records_and_keys()
    games = set()
    for path, expected_sha256 in (
        (T7_VALIDATION, T7_VALIDATION_SHA256),
        (T7_FINAL, T7_FINAL_SHA256),
    ):
        data = path.read_bytes()
        if sha256_bytes(data) != expected_sha256:
            raise RuntimeError(f"T7 bank hash mismatch: {path}")
        with path.open(newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                games.add(int(row["source_game_id"]))
                keys.add(row["canonical_key"])
    if len(keys) != EXPECTED_PRIOR_CANONICAL_KEYS:
        raise RuntimeError(
            f"expected {EXPECTED_PRIOR_CANONICAL_KEYS} prior keys, found {len(keys)}"
        )
    return keys, games


def relabel_public_elite(state: dict, winner_name: str,
                         public_elite_names: set[str]) -> dict:
    state = dict(state)
    if state["winner_tier"] == "field" and winner_name in public_elite_names:
        state["winner_tier"] = "elite"
        parts = state["_selection_stratum"].split("-", 4)
        if len(parts) != 5 or parts[1] != "field":
            raise RuntimeError("unexpected T8 selection-stratum encoding")
        parts[1] = "elite"
        state["_selection_stratum"] = "-".join(parts)
    return state


def eligible_states(prior_keys: set[str], excluded_game_ids: set[int],
                    public_elite_names: set[str]):
    raw_data = RAW_RECORDS.read_bytes()
    parseable_data = PARSEABLE_RECORDS.read_bytes()
    if sha256_bytes(raw_data) != RAW_SHA256:
        raise RuntimeError("T8 raw snapshot hash mismatch")
    if sha256_bytes(parseable_data) != PARSEABLE_SHA256:
        raise RuntimeError("T8 parseable snapshot hash mismatch")
    raw_payload = json.loads(raw_data)
    payload = json.loads(parseable_data)
    expected_agents = [agent_id for agent_id, _ in SOURCE_AGENTS]
    if (
        raw_payload.get("schema") != "papersoccer.frozen-t8-evidence-ladder.v1"
        or len(raw_payload.get("records", [])) != EXPECTED_RAW_RECORDS
        or raw_payload.get("agent_ids") != expected_agents
    ):
        raise RuntimeError("T8 raw snapshot provenance mismatch")
    if (
        payload.get("schema") != "papersoccer.frozen-t8-evidence-ladder.v2"
        or len(payload.get("records", [])) != EXPECTED_PARSEABLE_RECORDS
        or payload.get("structurally_rejected_games")
        != EXPECTED_STRUCTURALLY_REJECTED
        or payload.get("raw_sha256") != RAW_SHA256
        or payload.get("agent_ids") != expected_agents
    ):
        raise RuntimeError("T8 parseable snapshot provenance mismatch")

    candidates = []
    no_shell = 0
    prior_overlap = 0
    for record in payload["records"]:
        game_id = int(record["game_id"])
        if game_id in excluded_game_ids:
            raise RuntimeError(f"T8 acquisition game {game_id} overlaps prior data")
        states = extract_states(
            "validation", record, winner(record), elite_balance=True
        )
        if not states:
            no_shell += 1
            continue
        if any(state["canonical_key"] in prior_keys for state in states):
            prior_overlap += 1
            continue
        winner_name = (
            str(record["focus_name"])
            if record.get("won")
            else str(record["opponent_name"])
        )
        candidates.extend(
            relabel_public_elite(state, winner_name, public_elite_names)
            for state in states
        )
    if no_shell != EXPECTED_NO_SHELL_GAMES:
        raise RuntimeError(f"expected {EXPECTED_NO_SHELL_GAMES} no-shell games")
    if prior_overlap != EXPECTED_PRIOR_KEY_OVERLAP_GAMES:
        raise RuntimeError(
            f"expected {EXPECTED_PRIOR_KEY_OVERLAP_GAMES} key-overlap games"
        )

    seen = set(prior_keys)
    unique = []
    for state in sorted(candidates, key=lambda item: item["_selection_hash"]):
        if state["canonical_key"] in seen:
            continue
        seen.add(state["canonical_key"])
        unique.append(state)
    if len(unique) != EXPECTED_UNIQUE_STATES:
        raise RuntimeError(
            f"expected {EXPECTED_UNIQUE_STATES} unique states, found {len(unique)}"
        )
    if len({int(state["source_game_id"]) for state in unique}) != (
        EXPECTED_UNIQUE_GAMES
    ):
        raise RuntimeError("T8 unique-game capacity changed")
    return unique, raw_payload["exclusion_sources"]


def partition(states: list[dict]):
    by_game = collections.defaultdict(list)
    for state in states:
        by_game[int(state["source_game_id"])].append(state)
    groups = collections.defaultdict(list)
    for game_id, game_states in by_game.items():
        first = game_states[0]
        identity = (
            int(first["_focus_agent_id"]),
            first["winner_tier"],
            any(state["stratum"] == "d0" for state in game_states),
        )
        if any(
            (int(state["_focus_agent_id"]), state["winner_tier"])
            != identity[:2]
            for state in game_states
        ):
            raise RuntimeError("one source game spans focus/tier partition groups")
        groups[identity].append(game_id)

    side_by_game = {}
    for group, game_ids in sorted(groups.items()):
        game_ids.sort(
            key=lambda game_id: hashlib.sha256(
                f"t8-partition|{game_id}".encode()
            ).hexdigest()
        )
        start = int(
            hashlib.sha256(f"t8-group|{group}".encode()).hexdigest(), 16
        ) & 1
        for index, game_id in enumerate(game_ids):
            side_by_game[game_id] = (start + index) % 2

    pools = {
        "validation": [
            state
            for state in states
            if side_by_game[int(state["source_game_id"])] == 0
        ],
        "final": [
            state
            for state in states
            if side_by_game[int(state["source_game_id"])] == 1
        ],
    }
    for name, pool in pools.items():
        if len(pool) != EXPECTED_PARTITION_STATES[name]:
            raise RuntimeError(f"{name} partition state count changed")
        if len({state["source_game_id"] for state in pool}) != (
            EXPECTED_PARTITION_GAMES[name]
        ):
            raise RuntimeError(f"{name} partition game count changed")
    return pools


def composition(rows: list[dict]) -> dict:
    per_focus = collections.Counter(int(row["_focus_agent_id"]) for row in rows)
    return {
        "records": len(rows),
        "source_games": len({int(row["source_game_id"]) for row in rows}),
        "focus_agents": len(per_focus),
        "maximum_focus_rows": max(per_focus.values()),
        "focus_rows": [
            {"agent_id": agent_id, "rows": count}
            for agent_id, count in sorted(per_focus.items())
        ],
        "winner_tiers": dict(
            sorted(collections.Counter(row["winner_tier"] for row in rows).items())
        ),
        "distance": dict(
            sorted(collections.Counter(row["stratum"] for row in rows).items())
        ),
    }


def assert_bank(name: str, rows: list[dict], prior_keys: set[str],
                excluded_game_ids: set[int]):
    if len(rows) != 72:
        raise RuntimeError(f"{name} does not contain 72 rows")
    game_ids = [int(row["source_game_id"]) for row in rows]
    keys = [row["canonical_key"] for row in rows]
    per_game = collections.Counter(game_ids)
    per_game_distance = collections.Counter(
        (int(row["source_game_id"]), int(row["goal_distance_band"]))
        for row in rows
    )
    per_focus = collections.Counter(int(row["_focus_agent_id"]) for row in rows)
    tiers = collections.Counter(row["winner_tier"] for row in rows)
    distance = collections.Counter(row["stratum"] for row in rows)
    if len(keys) != len(set(keys)) or set(keys) & prior_keys:
        raise RuntimeError(f"{name} repeats or overlaps a canonical key")
    if set(game_ids) & excluded_game_ids:
        raise RuntimeError(f"{name} overlaps a prior source game")
    if max(per_game.values()) > GAME_CAP:
        raise RuntimeError(f"{name} exceeds its source-game cap")
    if max(per_game_distance.values()) > GAME_DISTANCE_CAP:
        raise RuntimeError(f"{name} exceeds its game-distance cap")
    if max(per_focus.values()) > FOCUS_AGENT_CAP:
        raise RuntimeError(f"{name} exceeds its focus-agent cap")
    if len(per_focus) < MIN_DISTINCT_FOCUS_AGENTS:
        raise RuntimeError(f"{name} lacks focus-agent diversity")
    if tiers["elite"] < MIN_ELITE_ROWS or tiers["field"] < MIN_FIELD_ROWS:
        raise RuntimeError(f"{name} lacks winner-tier coverage")
    if any(distance[key] < value for key, value in DISTANCE_MINIMUMS.items()):
        raise RuntimeError(f"{name} lacks goal-distance coverage")
    actual = composition(rows)
    for key, expected in EXPECTED_COMPOSITION[name].items():
        if actual[key] != expected:
            raise RuntimeError(
                f"{name} {key} changed: expected {expected}, found {actual[key]}"
            )


def build():
    assert_strength_protocol()
    excluded_game_ids, exclusion_sources = t8_exclusions()
    prior_keys, t7_bank_games = prior_keys_and_games()
    if not t7_bank_games <= excluded_game_ids:
        raise RuntimeError("T7 exposed bank games are absent from T8 exclusions")
    public_elite_names, public_scores, t7_manifest = t7_tier_metadata()
    states, frozen_exclusion_sources = eligible_states(
        prior_keys, excluded_game_ids, public_elite_names
    )
    if frozen_exclusion_sources != exclusion_sources:
        raise RuntimeError("T8 acquisition exclusion provenance changed")
    pools = partition(states)
    selected = {
        name: balanced_sample(
            pool,
            72,
            game_cap=GAME_CAP,
            game_distance_cap=GAME_DISTANCE_CAP,
            focus_agent_cap=FOCUS_AGENT_CAP,
            minimum_elite_rows=MIN_ELITE_ROWS,
        )
        for name, pool in pools.items()
    }
    for name, rows in selected.items():
        assert_bank(name, rows, prior_keys, excluded_game_ids)
    validation_games = {
        int(row["source_game_id"]) for row in selected["validation"]
    }
    final_games = {int(row["source_game_id"]) for row in selected["final"]}
    validation_keys = {row["canonical_key"] for row in selected["validation"]}
    final_keys = {row["canonical_key"] for row in selected["final"]}
    if validation_games & final_games or validation_keys & final_keys:
        raise RuntimeError("T8 banks are not mutually disjoint")

    validation_bytes = tsv_bytes(
        [dict(row, split="validation") for row in selected["validation"]]
    )
    final_bytes = tsv_bytes(
        [dict(row, split="test") for row in selected["final"]]
    )
    validation_hash = sha256_bytes(validation_bytes)
    final_hash = sha256_bytes(final_bytes)
    if EXPECTED_VALIDATION_SHA256 and validation_hash != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError("T8 validation hash changed")
    if EXPECTED_FINAL_SHA256 and final_hash != EXPECTED_FINAL_SHA256:
        raise RuntimeError("T8 final hash changed")

    sources = dict(exclusion_sources)
    for path in (
        RAW_RECORDS,
        PARSEABLE_RECORDS,
        HERE / "acquire_t8_evidence.py",
        HERE / "build_goal_shell_banks.py",
        HERE / "freeze_t8_banks.py",
        t7_manifest,
        ROOT / "submissions/codingame/bots/rank_5/submission.cpp",
    ):
        sources[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
    manifest = {
        "schema": "papersoccer.candidate-independent-t8-evidence.v1",
        "status": "frozen_before_candidate_binding",
        "candidate": None,
        "candidate_submission_sha256": None,
        "prospective_strength_protocol": PROSPECTIVE_STRENGTH_PROTOCOL,
        "prospective_strength_protocol_sha256": sha256_bytes(
            stable_json(PROSPECTIVE_STRENGTH_PROTOCOL)
        ),
        "immutability": {
            "before_candidate_binding": (
                "an explicit --update-manifest transition may only refresh "
                "candidate-independent source hashes"
            ),
            "at_candidate_binding": (
                "the future active manifest must pin this evidence-manifest hash, "
                "protocol hash, candidate submission, and runner hashes"
            ),
            "after_candidate_binding": (
                "this evidence manifest and both banks are immutable; any change "
                "requires a new versioned evidence ladder"
            ),
        },
        "prior_decisions": {
            "t7_validation": {
                "status": "exposed",
                "sha256": T7_VALIDATION_SHA256,
            },
            "t7_final": {"status": "exposed", "sha256": T7_FINAL_SHA256},
        },
        "unused_t7_capacity_audit": {
            "policy": "exclude both exposed T7 banks by source game and key",
            "eligible_unique_states": 62,
            "eligible_source_games": 16,
            "cap_two_row_capacity": 28,
            "focus_agents": 3,
            "distance": {"d0": 0, "d1": 31, "d2": 31},
            "conclusion": "insufficient for two fresh diverse 72-row banks",
        },
        "selection": {
            "candidate_independent": True,
            "input": "only the append-only T8 v2 public-game snapshot",
            "normalization": (
                "historical winner rotated to player zero; horizontal reflection dedup"
            ),
            "prior_exclusion": {
                "source_games": len(excluded_game_ids),
                "canonical_keys": len(prior_keys),
                "no_shell_games": EXPECTED_NO_SHELL_GAMES,
                "canonical_overlap_games": EXPECTED_PRIOR_KEY_OVERLAP_GAMES,
            },
            "tier_taxonomy": {
                "known_names": "existing rank1/elite name membership",
                "public_elite_score_cutoff": PUBLIC_ELITE_SCORE_CUTOFF,
                "public_elite_names": sorted(public_elite_names),
                "frozen_t7_focus_scores": [
                    {"agent_id": agent_id, "score": public_scores[agent_id]}
                    for agent_id in sorted(public_scores)
                ],
                "rule": (
                    "a pseudonym demonstrated at least the cutoff in frozen T7 "
                    "metadata; this label applies across that participant's "
                    "public agent versions"
                ),
            },
            "partition": (
                "within each frozen (focus_agent_id, winner_tier, has_d0) group, "
                "source games are ordered by sha256('t8-partition|' + game_id); "
                "alternating assignment starts at the low bit of "
                "sha256('t8-group|' + Python tuple representation)"
            ),
            "caps": {
                "rows_per_game": GAME_CAP,
                "rows_per_game_distance": GAME_DISTANCE_CAP,
                "rows_per_focus_agent": FOCUS_AGENT_CAP,
            },
            "floors": {
                "elite_rows": MIN_ELITE_ROWS,
                "field_rows": MIN_FIELD_ROWS,
                "distinct_focus_agents": MIN_DISTINCT_FOCUS_AGENTS,
                "distance_rows": DISTANCE_MINIMUMS,
            },
        },
        "sources": dict(sorted(sources.items())),
        "banks": {
            str(VALIDATION_BANK.relative_to(HERE)): {
                "role": "prospective_validation",
                "sealed": False,
                "sha256": validation_hash,
                **composition(selected["validation"]),
            },
            str(FINAL_BANK.relative_to(HERE)): {
                "role": "sealed_final",
                "sealed": True,
                "sha256": final_hash,
                **composition(selected["final"]),
            },
        },
    }
    return validation_bytes, final_bytes, stable_json(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update-manifest", action="store_true")
    arguments = parser.parse_args()
    if arguments.check and arguments.update_manifest:
        raise SystemExit("--check and --update-manifest are mutually exclusive")
    validation, final, manifest = build()
    artifacts = {
        VALIDATION_BANK: validation,
        FINAL_BANK: final,
        EVIDENCE_MANIFEST: manifest,
    }
    stale = []
    for path, content in artifacts.items():
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
            continue
        if path.exists():
            if path.read_bytes() != content:
                if path == EVIDENCE_MANIFEST and arguments.update_manifest:
                    path.write_bytes(content)
                    print(
                        f"updated {path.relative_to(ROOT)} "
                        f"sha256={sha256_bytes(content)}"
                    )
                    continue
                raise FileExistsError(f"refusing to replace frozen {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"froze {path.relative_to(ROOT)} sha256={sha256_bytes(content)}")
    if stale:
        raise SystemExit("stale T8 evidence artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
