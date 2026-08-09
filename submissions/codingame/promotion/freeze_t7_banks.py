#!/usr/bin/env python3

"""Freeze candidate-independent T7 validation and sealed-final banks."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib

from acquire_t7_evidence import (
    EVIDENCE_BANKS,
    PARSEABLE_RECORDS,
    RAW_RECORDS,
    SOURCE_AGENTS,
    frozen_exclusions,
)
from build_goal_shell_banks import (
    ELITE_FINAL_RAW_RECORDS,
    ELITE_FINAL_RECORDS,
    FRESH_RECORDS,
    ROOT,
    T6_VALIDATION_EXTENSION_RAW_RECORDS,
    T6_VALIDATION_EXTENSION_RECORDS,
    VALIDATION_EXTENSION_RAW_RECORDS,
    VALIDATION_EXTENSION_RECORDS,
    balanced_sample,
    elite_final_records,
    extract_states,
    fresh_records,
    prior_raw_sources,
    sha256_bytes,
    stable_json,
    t6_validation_extension_records,
    tsv_bytes,
    validation_extension_records,
)


HERE = pathlib.Path(__file__).resolve().parent
VALIDATION_BANK = HERE / "reference" / "t7_prospective_validation.tsv"
FINAL_BANK = HERE / "reference" / "t7_sealed_final.tsv"
EVIDENCE_MANIFEST = HERE / "reference" / "t7_evidence_manifest.json"
RAW_SHA256 = "fc409257a9e19cb8664385e9cdf32f07ae4ce196a6a619d67d17acf27fdf230e"
PARSEABLE_SHA256 = (
    "207384779c5bccd5bddec97eb6fbc40bf7132698310381d5c840d9e5e74755fc"
)
EXPECTED_VALIDATION_SHA256 = (
    "878ca510b63e50339eaeccc57b50445c6b5915b568ccb26a588b386341c5b002"
)
EXPECTED_FINAL_SHA256 = (
    "de7e592610c2ab2842874b5b647fca951cf7d51373b4c0c49d8b255b9543ed59"
)

PUBLIC_ELITE_SCORE_CUTOFF = 35.0
PUBLIC_ELITE_AGENT_IDS = {
    2_597_500,
    4_413_390,
    4_792_144,
    5_476_643,
    6_589_744,
}
GAME_CAP = 2
GAME_DISTANCE_CAP = 1
FOCUS_AGENT_CAP = 16
MIN_ELITE_ROWS = 12
MIN_FIELD_ROWS = 40
MIN_DISTINCT_FOCUS_AGENTS = 12
DISTANCE_MINIMUMS = {"d0": 8, "d1": 24, "d2": 24}

EXPECTED_PRIOR_CANONICAL_KEYS = 3_195
EXPECTED_NO_SHELL_GAMES = 192
EXPECTED_PRIOR_KEY_OVERLAP_GAMES = 28
EXPECTED_UNIQUE_STATES = 561
EXPECTED_UNIQUE_GAMES = 114
EXPECTED_PARTITION_STATES = {"validation": 286, "final": 275}
EXPECTED_PARTITION_GAMES = {"validation": 57, "final": 57}
EXPECTED_COMPOSITION = {
    "validation": {
        "source_games": 48,
        "focus_agents": 14,
        "maximum_focus_rows": 13,
        "winner_tiers": {"elite": 14, "field": 58},
        "distance": {"d0": 8, "d1": 36, "d2": 28},
    },
    "final": {
        "source_games": 47,
        "focus_agents": 14,
        "maximum_focus_rows": 13,
        "winner_tiers": {"elite": 19, "field": 53},
        "distance": {"d0": 9, "d1": 37, "d2": 26},
    },
}

TIME_PROFILE_MAX_NODES = 3_000_000
INCUMBENT_SUBMISSION_SHA256 = (
    "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
)
PROSPECTIVE_STRENGTH_PROTOCOL = {
    "status": "predeclared_before_candidate_binding_and_outcomes",
    "stage_bank_policy": (
        "each stage uses one bank for every strength profile; validation uses "
        "the prospective T7 validation bank and test uses the disjoint sealed "
        "T7 final bank"
    ),
    "game_format": {
        "games_per_opening": 3,
        "candidate_games": (
            "candidate versus rank5 once as each physical player"
        ),
        "control_game": (
            "rank5 versus rank5 on the same opening and execution budget"
        ),
    },
    "runner_projection": {
        "nodes": {
            "node_budget": "profile.value",
            "time_budget_ms": 0,
        },
        "time_ms": {
            "node_budget": "profile.max_nodes",
            "time_budget_ms": "profile.value",
        },
    },
    "statistics": {
        "confidence": 0.95,
        "method": "source_game_cluster_percentile_bootstrap",
        "resamples": 10000,
        "seed": 4_774_557_432_748_095_049,
        "unit": "source_game_cluster_of_color_swapped_opening_pairs",
    },
    "execution_contract": {
        "candidate_runner_binding": (
            "the active candidate manifest must hash-pin the candidate-specific "
            "comparison runner before any T7 stage runs"
        ),
        "incumbent_submission_sha256": INCUMBENT_SUBMISSION_SHA256,
        "profile_sharding": (
            "all profiles in one stage use the same positive shard count; one "
            "worker pool has max_workers equal to that count"
        ),
        "within_opening_order": (
            "the two candidate-color games and rank5-vs-rank5 control are run "
            "sequentially by the same runner process"
        ),
        "time_budget_clock": "steady-clock wall time",
        "time_profile_node_cap": TIME_PROFILE_MAX_NODES,
        "locked_final_identity": (
            "the v2 consumption marker binds ordered profile identities and "
            "the common raw-evidence shard count"
        ),
    },
    "diagnostic_only": [
        "control_normalization.candidate_historical_scores",
        "control_normalization.historical_baseline_scores",
        "control_normalization.historical_uplifts",
        "control_normalization.minimum_historical_uplift",
        "control_normalization.minimum_historical_role_score",
        "time-profile candidate_to_incumbent_throughput",
    ],
    "stages": {
        "initial": {
            "bank_policy": "reuse the frozen initial bank at candidate binding",
            "configuration": {
                "maximum_turns": 320,
                "minimum_mean": 0.5,
                "node_budget": 5000,
            },
        },
        "development": {
            "bank_policy": (
                "reuse the frozen development bank at candidate binding"
            ),
            "configuration": {
                "maximum_turns": 320,
                "minimum_mean": 0.52,
                "minimum_throughput_ratio": 0.9,
                "node_budgets": [5000, 30000],
                "node_budget_overrides": {
                    "5000": {
                        "minimum_mean": 0.5,
                        "require_at_least_as_many_wins_as_incumbent": True,
                        "require_more_wins_than_incumbent": False,
                    },
                },
                "require_more_wins_than_incumbent": True,
            },
        },
        "validation": {
            "bank": str(VALIDATION_BANK.relative_to(HERE)),
            "configuration": {
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
                    {
                        "id": "30k-nodes",
                        "mode": "nodes",
                        "value": 30000,
                    },
                    {
                        "id": "130ms",
                        "mode": "time_ms",
                        "value": 130,
                        "max_nodes": TIME_PROFILE_MAX_NODES,
                        "thresholds": {
                            "minimum_throughput_ratio": None,
                        },
                    },
                ],
            },
        },
        "test": {
            "bank": str(FINAL_BANK.relative_to(HERE)),
            "configuration": {
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
                    {
                        "id": "100k-nodes",
                        "mode": "nodes",
                        "value": 100000,
                    },
                    {
                        "id": "130ms",
                        "mode": "time_ms",
                        "value": 130,
                        "max_nodes": TIME_PROFILE_MAX_NODES,
                        "thresholds": {
                            "minimum_throughput_ratio": None,
                        },
                    },
                ],
            },
        },
    },
}
EXPECTED_STRENGTH_PROTOCOL_SHA256 = (
    "83433c7b74ceb38a166d3a52324d7c137fe8db04038c48edc57d839effe82dba"
)


def assert_strength_protocol() -> None:
    if sha256_bytes(stable_json(PROSPECTIVE_STRENGTH_PROTOCOL)) != (
            EXPECTED_STRENGTH_PROTOCOL_SHA256):
        raise RuntimeError("T7 prospective strength protocol hash changed")
    stages = PROSPECTIVE_STRENGTH_PROTOCOL["stages"]
    if stages["development"]["configuration"]["node_budgets"] != [5000, 30000]:
        raise RuntimeError("T7 development profiles changed")
    expected = {
        "validation": [
            ("30k-nodes", "nodes", 30000, 30000),
            ("130ms", "time_ms", 130, TIME_PROFILE_MAX_NODES),
        ],
        "test": [
            ("100k-nodes", "nodes", 100000, 100000),
            ("130ms", "time_ms", 130, TIME_PROFILE_MAX_NODES),
        ],
    }
    for stage, expected_profiles in expected.items():
        config = stages[stage]["configuration"]
        if "minimum_historical_role_score" in config:
            raise RuntimeError("historical-role score must remain diagnostic in T7")
        actual_profiles = []
        for profile in config["strength_profiles"]:
            thresholds = profile.get("thresholds", {})
            if ({"minimum_historical_role_score",
                 "minimum_control_adjusted_uplift"} & set(thresholds)):
                raise RuntimeError(
                    "historical role partitions must remain diagnostic in T7"
                )
            maximum_nodes = (
                profile["value"] if profile["mode"] == "nodes"
                else profile["max_nodes"]
            )
            actual_profiles.append((
                profile["id"], profile["mode"], profile["value"], maximum_nodes,
            ))
            if (profile["mode"] == "time_ms" and
                    profile.get("thresholds", {}).get(
                        "minimum_throughput_ratio", "missing"
                    ) is not None):
                raise RuntimeError("T7 time-profile throughput must be diagnostic")
        if actual_profiles != expected_profiles:
            raise RuntimeError(f"T7 {stage} strength profiles changed")
    if stages["validation"]["bank"] == stages["test"]["bank"]:
        raise RuntimeError("T7 validation and final banks must be distinct")


def winner(record: dict) -> int:
    return (int(record["player_id"]) if record.get("won")
            else 1 - int(record["player_id"]))


def prior_records_and_keys():
    prior, prior_game_ids, _ = prior_raw_sources()
    fresh, _ = fresh_records(prior_game_ids)
    fresh_game_ids = {int(record["game_id"]) for _, record, _ in fresh}
    elite = elite_final_records(prior_game_ids | fresh_game_ids)[0]
    elite_raw = json.loads(ELITE_FINAL_RAW_RECORDS.read_text())
    elite_raw_ids = {int(record["game_id"]) for record in elite_raw["records"]}
    elite_ids = {int(record["game_id"]) for _, record, _ in elite}
    extension = validation_extension_records(
        prior_game_ids | fresh_game_ids | elite_raw_ids | elite_ids
    )[0]
    extension_raw = json.loads(VALIDATION_EXTENSION_RAW_RECORDS.read_text())
    extension_raw_ids = {
        int(record["game_id"]) for record in extension_raw["records"]
    }
    extension_ids = {int(record["game_id"]) for _, record, _ in extension}
    t6_extension = t6_validation_extension_records(
        prior_game_ids | fresh_game_ids | elite_raw_ids | elite_ids |
        extension_raw_ids | extension_ids
    )[0]

    keys = set()
    for record, record_winner in prior:
        keys.update(
            state["canonical_key"]
            for state in extract_states("legacy", record, record_winner)
        )
    for _, record, record_winner in fresh:
        keys.update(
            state["canonical_key"]
            for state in extract_states("legacy", record, record_winner)
        )
    for source in (elite, extension, t6_extension):
        for _, record, record_winner in source:
            keys.update(
                state["canonical_key"]
                for state in extract_states(
                    "legacy", record, record_winner, elite_balance=True
                )
            )
    for path, _ in EVIDENCE_BANKS:
        with path.open(newline="") as source:
            keys.update(
                row["canonical_key"]
                for row in csv.DictReader(source, delimiter="\t")
            )
    if len(keys) != EXPECTED_PRIOR_CANONICAL_KEYS:
        raise RuntimeError(
            f"expected {EXPECTED_PRIOR_CANONICAL_KEYS} prior canonical keys, "
            f"found {len(keys)}"
        )
    return keys


def relabel_public_elite(state: dict) -> dict:
    state = dict(state)
    if (state["winner_tier"] == "field" and
            int(state["source_agent_id"]) in PUBLIC_ELITE_AGENT_IDS):
        state["winner_tier"] = "elite"
        parts = state["_selection_stratum"].split("-", 4)
        if len(parts) != 5 or parts[1] != "field":
            raise RuntimeError("unexpected T7 selection-stratum encoding")
        parts[1] = "elite"
        state["_selection_stratum"] = "-".join(parts)
    return state


def eligible_states(prior_keys: set[str], excluded_game_ids: set[int]):
    raw_data = RAW_RECORDS.read_bytes()
    parseable_data = PARSEABLE_RECORDS.read_bytes()
    if sha256_bytes(raw_data) != RAW_SHA256:
        raise RuntimeError("T7 raw snapshot hash mismatch")
    if sha256_bytes(parseable_data) != PARSEABLE_SHA256:
        raise RuntimeError("T7 parseable snapshot hash mismatch")
    raw_payload = json.loads(raw_data)
    payload = json.loads(parseable_data)
    if (raw_payload.get("schema") != "papersoccer.frozen-t7-evidence-ladder.v1"
            or len(raw_payload.get("records", [])) != 358
            or raw_payload.get("excluded_game_count") != 853):
        raise RuntimeError("T7 raw snapshot provenance mismatch")
    if (payload.get("schema") != "papersoccer.frozen-t7-evidence-ladder.v2"
            or payload.get("raw_sha256") != RAW_SHA256
            or len(payload.get("records", [])) != 352
            or payload.get("structurally_rejected_games") != 6):
        raise RuntimeError("T7 parseable snapshot provenance mismatch")
    expected_agents = [agent_id for agent_id, _ in SOURCE_AGENTS]
    if (raw_payload.get("agent_ids") != expected_agents or
            payload.get("agent_ids") != expected_agents):
        raise RuntimeError("T7 source-agent declaration mismatch")

    focus_scores = {}
    for record in payload["records"]:
        game_id = int(record["game_id"])
        if game_id in excluded_game_ids:
            raise RuntimeError(f"T7 game {game_id} overlaps prior evidence")
        score = record.get("focus_score")
        if score is not None:
            agent_id = int(record["focus_agent_id"])
            score = float(score)
            previous = focus_scores.setdefault(agent_id, score)
            if previous != score:
                raise RuntimeError("a frozen T7 focus score is not constant")
    derived_elite_ids = {
        agent_id for agent_id, score in focus_scores.items()
        if score >= PUBLIC_ELITE_SCORE_CUTOFF
    }
    if derived_elite_ids != PUBLIC_ELITE_AGENT_IDS:
        raise RuntimeError(
            f"public-elite taxonomy changed: {sorted(derived_elite_ids)}"
        )

    no_shell = 0
    prior_overlap = 0
    candidates = []
    for record in payload["records"]:
        states = extract_states(
            "validation", record, winner(record), elite_balance=True
        )
        if not states:
            no_shell += 1
            continue
        if any(state["canonical_key"] in prior_keys for state in states):
            prior_overlap += 1
            continue
        candidates.extend(relabel_public_elite(state) for state in states)
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
            f"expected {EXPECTED_UNIQUE_STATES} unique T7 states, found {len(unique)}"
        )
    games = {int(state["source_game_id"]) for state in unique}
    if len(games) != EXPECTED_UNIQUE_GAMES:
        raise RuntimeError(
            f"expected {EXPECTED_UNIQUE_GAMES} unique T7 games, found {len(games)}"
        )
    return unique, focus_scores, raw_payload["exclusion_sources"]


def partition(states: list[dict]):
    by_game = collections.defaultdict(list)
    for state in states:
        by_game[int(state["source_game_id"])].append(state)
    groups = collections.defaultdict(list)
    for game_id, game_states in by_game.items():
        first = game_states[0]
        identity = (int(first["_focus_agent_id"]), first["winner_tier"])
        if any(
            (int(state["_focus_agent_id"]), state["winner_tier"]) != identity
            for state in game_states
        ):
            raise RuntimeError("one source game spans focus/tier partition groups")
        groups[identity].append(game_id)

    side_by_game = {}
    for group, game_ids in sorted(groups.items()):
        game_ids.sort(key=lambda game_id: hashlib.sha256(
            f"t7-partition|{game_id}".encode()
        ).hexdigest())
        start = int(hashlib.sha256(
            f"t7-group|{group}".encode()
        ).hexdigest(), 16) & 1
        for index, game_id in enumerate(game_ids):
            side_by_game[game_id] = (start + index) % 2

    pools = {
        "validation": [
            state for state in states
            if side_by_game[int(state["source_game_id"])] == 0
        ],
        "final": [
            state for state in states
            if side_by_game[int(state["source_game_id"])] == 1
        ],
    }
    for name, pool in pools.items():
        if len(pool) != EXPECTED_PARTITION_STATES[name]:
            raise RuntimeError(f"{name} partition state count changed")
        if len({state["source_game_id"] for state in pool}) != (
                EXPECTED_PARTITION_GAMES[name]):
            raise RuntimeError(f"{name} partition game count changed")
    return pools


def composition(rows: list[dict]) -> dict:
    per_focus = collections.Counter(
        int(row["_focus_agent_id"]) for row in rows
    )
    return {
        "records": len(rows),
        "source_games": len({int(row["source_game_id"]) for row in rows}),
        "focus_agents": len(per_focus),
        "maximum_focus_rows": max(per_focus.values()),
        "focus_rows": [
            {"agent_id": agent_id, "rows": count}
            for agent_id, count in sorted(per_focus.items())
        ],
        "winner_tiers": dict(sorted(collections.Counter(
            row["winner_tier"] for row in rows
        ).items())),
        "distance": dict(sorted(collections.Counter(
            row["stratum"] for row in rows
        ).items())),
    }


def assert_bank(name: str, rows: list[dict], prior_keys: set[str],
                excluded_game_ids: set[int]):
    if len(rows) != 72:
        raise RuntimeError(f"{name} does not contain 72 rows")
    game_ids = [int(row["source_game_id"]) for row in rows]
    canonical_keys = [row["canonical_key"] for row in rows]
    per_game = collections.Counter(game_ids)
    per_game_distance = collections.Counter(
        (int(row["source_game_id"]), int(row["goal_distance_band"]))
        for row in rows
    )
    per_focus = collections.Counter(int(row["_focus_agent_id"]) for row in rows)
    tiers = collections.Counter(row["winner_tier"] for row in rows)
    distance = collections.Counter(row["stratum"] for row in rows)
    if len(canonical_keys) != len(set(canonical_keys)):
        raise RuntimeError(f"{name} repeats a canonical key")
    if set(canonical_keys) & prior_keys:
        raise RuntimeError(f"{name} overlaps a prior canonical key")
    if set(game_ids) & excluded_game_ids:
        raise RuntimeError(f"{name} overlaps a prior source game")
    if max(per_game.values()) > GAME_CAP:
        raise RuntimeError(f"{name} exceeds its game cap")
    if max(per_game_distance.values()) > GAME_DISTANCE_CAP:
        raise RuntimeError(f"{name} exceeds its game-distance cap")
    if max(per_focus.values()) > FOCUS_AGENT_CAP:
        raise RuntimeError(f"{name} exceeds its focus-agent cap")
    if len(per_focus) < MIN_DISTINCT_FOCUS_AGENTS:
        raise RuntimeError(f"{name} lacks focus-agent diversity")
    if tiers["elite"] < MIN_ELITE_ROWS or tiers["field"] < MIN_FIELD_ROWS:
        raise RuntimeError(f"{name} lacks tier coverage")
    if any(distance[key] < value for key, value in DISTANCE_MINIMUMS.items()):
        raise RuntimeError(f"{name} lacks distance coverage")
    actual = composition(rows)
    for key, expected in EXPECTED_COMPOSITION[name].items():
        if actual[key] != expected:
            raise RuntimeError(
                f"{name} {key} changed: expected {expected}, found {actual[key]}"
            )


def build():
    assert_strength_protocol()
    excluded_game_ids, exclusion_sources = frozen_exclusions()
    prior_keys = prior_records_and_keys()
    states, focus_scores, frozen_exclusion_sources = eligible_states(
        prior_keys, excluded_game_ids
    )
    if frozen_exclusion_sources != exclusion_sources:
        raise RuntimeError("T7 acquisition exclusion-source provenance changed")
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
    validation_keys = {
        row["canonical_key"] for row in selected["validation"]
    }
    final_keys = {row["canonical_key"] for row in selected["final"]}
    if validation_games & final_games:
        raise RuntimeError("T7 banks share a source game")
    if validation_keys & final_keys:
        raise RuntimeError("T7 banks share a canonical key")

    validation_rows = [dict(row, split="validation")
                       for row in selected["validation"]]
    final_rows = [dict(row, split="test") for row in selected["final"]]
    validation_bytes = tsv_bytes(validation_rows)
    final_bytes = tsv_bytes(final_rows)
    validation_hash = sha256_bytes(validation_bytes)
    final_hash = sha256_bytes(final_bytes)
    if (EXPECTED_VALIDATION_SHA256 and
            validation_hash != EXPECTED_VALIDATION_SHA256):
        raise RuntimeError("T7 validation hash changed")
    if EXPECTED_FINAL_SHA256 and final_hash != EXPECTED_FINAL_SHA256:
        raise RuntimeError("T7 final hash changed")

    sources = dict(exclusion_sources)
    for path in (
        RAW_RECORDS,
        PARSEABLE_RECORDS,
        HERE / "acquire_t7_evidence.py",
        HERE / "build_goal_shell_banks.py",
        HERE / "freeze_t7_banks.py",
        ROOT / "submissions/codingame/bots/rank_5/submission.cpp",
        ROOT / "submissions/codingame/tools/promotion_gate.py",
    ):
        sources[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
    manifest = {
        "schema": "papersoccer.candidate-independent-t7-evidence.v1",
        "status": "frozen_before_candidate_binding",
        "candidate": None,
        "candidate_submission_sha256": None,
        "prospective_strength_protocol": PROSPECTIVE_STRENGTH_PROTOCOL,
        "prospective_strength_protocol_sha256": sha256_bytes(
            stable_json(PROSPECTIVE_STRENGTH_PROTOCOL)
        ),
        "immutability": {
            "before_candidate_binding": (
                "an explicit --update-manifest transition is required to replace "
                "this candidate-independent manifest"
            ),
            "at_candidate_binding": (
                "the active T7 manifest must pin this evidence-manifest hash, "
                "the protocol hash, candidate submission, and runner hashes"
            ),
            "after_candidate_binding": (
                "this evidence manifest and both banks are immutable; any protocol "
                "change requires a new versioned evidence ladder"
            ),
        },
        "prior_decisions": {
            "t6_validation": {
                "status": "exposed_rejected",
                "sha256": (
                    "69c0f3e78c878ed6e51e599f7207d445dbeb8a8564fecac63ff4105953bf600d"
                ),
            },
            "t6_final": {
                "status": "exposed",
                "sha256": (
                    "ef48f9e190aa14cf4b791641ca73276f0794179b3a2f1c3fa26eda32d465cdd4"
                ),
            },
        },
        "selection": {
            "candidate_independent": True,
            "input": "only the append-only T7 v2 public-game snapshot",
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
                "public_elite_agent_ids": sorted(PUBLIC_ELITE_AGENT_IDS),
                "frozen_focus_scores": [
                    {"agent_id": agent_id, "score": focus_scores[agent_id]}
                    for agent_id in sorted(focus_scores)
                ],
            },
            "partition": (
                "within each frozen (focus_agent_id, winner_tier) group, source "
                "games are ordered by sha256('t7-partition|' + game_id); alternating "
                "assignment starts at the low bit of sha256('t7-group|' + Python "
                "tuple representation); side 0 is validation and side 1 is final"
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
                hint = (
                    "; use --update-manifest only before candidate binding"
                    if path == EVIDENCE_MANIFEST else ""
                )
                raise FileExistsError(
                    f"refusing to replace frozen {path}{hint}"
                )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"froze {path.relative_to(ROOT)} sha256={sha256_bytes(content)}")
    if stale:
        raise SystemExit("stale T7 evidence artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
