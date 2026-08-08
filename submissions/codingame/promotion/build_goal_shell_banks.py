#!/usr/bin/env python3

"""Build frozen, complete-turn CodinGame promotion banks from Arena records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Iterable


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = HERE.parent / "tools"
LOCKED_AGENT_ID = 6_273_433
LOCKED_AFTER_GAME = 896_637_709
LOCKED_RECORDS = HERE / "rank1_locked_games.json"
ELITE_NAMES = {"jacek", "Deltaspace", "Marchete", "Snekkers", "Laars"}
DEVELOPMENT_AGENTS = {6_567_975, 6_567_983, 6_567_993, 6_568_126, 6_568_130}
VALIDATION_AGENTS = {6_568_141, 6_568_150, 6_568_158}
DIRECTIONS = ((0, -1), (1, -1), (1, 0), (1, 1),
              (0, 1), (-1, 1), (-1, 0), (-1, -1))
FIELD_WIDTH = 8
ROTATION_HEIGHT = 12
BANK_LIMITS = {"development": 96, "validation": 96, "test": 72}
HEADER = (
    "opening_id\tsplit\tstratum\tsource_agent_id\tsource_game_id\t"
    "opponent_agent_id\twinner_player_id\tturn_index\tphysical_edges\t"
    "state_key\tcanonical_key\tball_x\tball_y\tmover\tgoal_distance_band\t"
    "used_edge_band\tshell_edge_band\topening_family\tobserved_winner_action\t"
    "transcript\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def rotate(point: tuple[int, int]) -> tuple[int, int]:
    return FIELD_WIDTH - point[0], ROTATION_HEIGHT - point[1]


def reflect(point: tuple[int, int]) -> tuple[int, int]:
    return FIELD_WIDTH - point[0], point[1]


def normalized_segment(a: tuple[int, int], b: tuple[int, int]):
    return tuple(sorted((a, b)))


def state_text(ball: tuple[int, int], mover: int, edges) -> str:
    edge_text = ";".join(
        f"{a[0]},{a[1]}-{b[0]},{b[1]}" for a, b in sorted(edges)
    )
    return f"ball={ball[0]},{ball[1]}|mover={mover}|edges={edge_text}"


def state_identity(ball, mover, edges, winner):
    if winner == 1:
        ball = rotate(ball)
        edges = {normalized_segment(rotate(a), rotate(b)) for a, b in edges}
        mover = 1 - mover
    raw = state_text(ball, mover, edges)
    reflected_ball = reflect(ball)
    reflected_edges = {
        normalized_segment(reflect(a), reflect(b)) for a, b in edges
    }
    mirrored = state_text(reflected_ball, mover, reflected_edges)
    return (
        sha256_bytes(raw.encode()),
        sha256_bytes(min(raw, mirrored).encode()),
        ball,
        mover,
        edges,
    )


def apply_action(ball, edges, action: str):
    for character in action:
        if character < "0" or character > "7":
            raise ValueError(f"invalid direction {character!r}")
        dx, dy = DIRECTIONS[ord(character) - ord("0")]
        destination = ball[0] + dx, ball[1] + dy
        edge = normalized_segment(ball, destination)
        if edge in edges:
            raise ValueError(f"reused edge in recorded action {action!r}")
        edges.add(edge)
        ball = destination
    return ball


def tier(name: str) -> str:
    if name == "jacek":
        return "rank1"
    if name in ELITE_NAMES:
        return "elite"
    return "field"


def edge_band(count: int) -> str:
    if count < 48:
        return "sparse"
    if count < 112:
        return "building"
    return "closed"


def shell_band(edges) -> str:
    count = sum(1 for a, b in edges if max(a[1], b[1]) >= 9)
    if count < 12:
        return "open"
    if count < 28:
        return "layering"
    return "dense"


def opening_family(actions: list[str]) -> str:
    return "/".join(actions[: min(4, len(actions))]) or "initial"


def records_from_checked_batches():
    paths = sorted((ROOT / "submissions/codingame/bots").glob("*/arena_batch_*.json"))
    records = []
    sources = {}
    for path in paths:
        payload = json.loads(path.read_text())
        agent_id = int(payload["agent_id"])
        if agent_id not in DEVELOPMENT_AGENTS | VALIDATION_AGENTS:
            continue
        split = "development" if agent_id in DEVELOPMENT_AGENTS else "validation"
        sources[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
        for loss in payload["loss_records"]:
            if loss.get("won") is not False:
                raise ValueError(f"expected a loss record in {path}")
            records.append((split, loss, 1 - int(loss["player_id"])))
    return records, sources


def locked_records():
    if not LOCKED_RECORDS.exists():
        raise FileNotFoundError(
            f"missing {LOCKED_RECORDS}; run with --fetch-locked once"
        )
    payload = json.loads(LOCKED_RECORDS.read_text())
    if payload.get("agent_id") != LOCKED_AGENT_ID:
        raise ValueError("locked corpus has the wrong agent id")
    result = []
    for record in payload["records"]:
        if not record.get("won") or int(record["game_id"]) <= LOCKED_AFTER_GAME:
            raise ValueError("locked corpus contains an ineligible game")
        result.append(("test", record, int(record["player_id"])))
    return result, {
        str(LOCKED_RECORDS.relative_to(ROOT)): sha256_bytes(LOCKED_RECORDS.read_bytes())
    }


def extract_states(split: str, record: dict, winner: int):
    turns = record["turns"]
    ball = (4, 6)
    edges = set()
    actions: list[str] = []
    states = []
    for turn_index, turn in enumerate(turns):
        mover = int(turn["player_id"])
        action = str(turn["action"])
        if mover == winner:
            state_key, canonical_key, normalized_ball, normalized_mover, normalized_edges = (
                state_identity(ball, mover, edges, winner)
            )
            if normalized_mover != 0:
                raise ValueError("winner normalization did not produce player zero")
            distance = 11 - normalized_ball[1]
            if 0 <= distance <= 2 and 1 <= normalized_ball[0] <= 7:
                source_agent = int(record["focus_agent_id"] if record.get("won")
                                   else record["opponent_agent_id"])
                opponent_agent = int(record["opponent_agent_id"] if record.get("won")
                                     else record["focus_agent_id"])
                opponent_name = str(record["focus_name"] if not record.get("won")
                                    else record["opponent_name"])
                winner_name = str(record["focus_name"] if record.get("won")
                                  else record["opponent_name"])
                phase = "early" if turn_index < 18 else (
                    "middle" if turn_index < 36 else "late"
                )
                cohort = tier(winner_name)
                stratum = f"d{distance}"
                transcript = "/".join(actions)
                identity_seed = (
                    f"{split}|{record['game_id']}|{turn_index}|{canonical_key}"
                )
                states.append({
                    "opening_id": "shell-" + sha256_bytes(identity_seed.encode())[:16],
                    "split": split,
                    "stratum": stratum,
                    "source_agent_id": source_agent,
                    "source_game_id": int(record["game_id"]),
                    "opponent_agent_id": opponent_agent,
                    "winner_player_id": winner,
                    "turn_index": turn_index,
                    "physical_edges": len(edges),
                    "state_key": state_key,
                    "canonical_key": canonical_key,
                    "ball_x": ball[0],
                    "ball_y": ball[1],
                    "mover": mover,
                    "goal_distance_band": distance,
                    "used_edge_band": edge_band(len(edges)),
                    "shell_edge_band": shell_band(normalized_edges),
                    "opening_family": opening_family(actions),
                    "observed_winner_action": action,
                    "transcript": transcript,
                    "_selection_hash": sha256_bytes(identity_seed.encode()),
                    "_selection_stratum": f"{cohort}-{stratum}-{phase}",
                    "_opponent_name": opponent_name,
                })
        ball = apply_action(ball, edges, action)
        actions.append(action)
    return states


def balanced_sample(states: Iterable[dict], limit: int):
    queues = collections.defaultdict(list)
    for state in states:
        queues[state["_selection_stratum"]].append(state)
    for values in queues.values():
        values.sort(key=lambda item: item["_selection_hash"])
    result = []
    per_game = collections.Counter()
    per_game_distance = collections.Counter()
    while len(result) < limit:
        progressed = False
        for key in sorted(queues):
            while queues[key]:
                candidate = queues[key].pop(0)
                game = candidate["source_game_id"]
                game_distance = (game, candidate["goal_distance_band"])
                if per_game[game] >= 4 or per_game_distance[game_distance] >= 2:
                    continue
                result.append(candidate)
                per_game[game] += 1
                per_game_distance[game_distance] += 1
                progressed = True
                break
            if len(result) == limit:
                break
        if not progressed:
            break
    return sorted(result, key=lambda item: item["opening_id"])


def tsv_bytes(states: list[dict]) -> bytes:
    columns = HEADER.rstrip("\n").split("\t")
    rows = [HEADER]
    for state in states:
        values = []
        for column in columns:
            value = state[column]
            text = str(value)
            if "\t" in text or "\n" in text:
                raise ValueError(f"invalid TSV value in {column}")
            values.append(text)
        rows.append("\t".join(values) + "\n")
    return "".join(rows).encode()


def build_artifacts():
    checked, checked_sources = records_from_checked_batches()
    locked, locked_sources = locked_records()
    all_records = checked + locked
    candidates = collections.defaultdict(list)
    for split, record, winner in all_records:
        candidates[split].extend(extract_states(split, record, winner))

    seen = set()
    banks = {}
    selected = {}
    for split in ("development", "validation", "test"):
        unique = []
        for state in sorted(candidates[split], key=lambda item: item["_selection_hash"]):
            if state["canonical_key"] in seen:
                continue
            seen.add(state["canonical_key"])
            unique.append(state)
        chosen = balanced_sample(unique, BANK_LIMITS[split])
        selected[split] = chosen
        banks[f"openings/{split}.tsv"] = tsv_bytes(chosen)

    initial_state_key, initial_canonical_key, _, _, _ = state_identity(
        (4, 6), 0, set(), 0
    )
    initial = {
        "opening_id": "initial",
        "split": "initial",
        "stratum": "initial",
        "source_agent_id": 0,
        "source_game_id": 0,
        "opponent_agent_id": 0,
        "winner_player_id": 0,
        "turn_index": 0,
        "physical_edges": 0,
        "state_key": initial_state_key,
        "canonical_key": initial_canonical_key,
        "ball_x": 4,
        "ball_y": 6,
        "mover": 0,
        "goal_distance_band": -1,
        "used_edge_band": "empty",
        "shell_edge_band": "empty",
        "opening_family": "initial",
        "observed_winner_action": "-",
        "transcript": "-",
    }
    banks["openings/initial.tsv"] = tsv_bytes([initial])

    source_hashes = dict(sorted((checked_sources | locked_sources).items()))
    harness_sources = (
        HERE / "build_goal_shell_banks.py",
        HERE.parent / "tools" / "promotion_gate.py",
        HERE.parent / "bots" / "topology" / "comparison_gate.cpp",
        HERE.parent / "bots" / "topology" / "submission_test.cpp",
        HERE.parent / "bots" / "topology" / "timing_probe.cpp",
    )
    for path in harness_sources:
        source_hashes[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
    source_hashes = dict(sorted(source_hashes.items()))
    manifest = {
        "schema": "papersoccer.codingame-promotion-manifest.v1",
        "candidate": "topology",
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "own_goals_allowed",
            "blocked_rule": "mover_loses",
            "positions_are_complete_turn_boundaries": True,
        },
        "incumbent": {
            "name": "rank_5",
            "submission_sha256": (
                "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
            ),
        },
        "hypothesis": "capped rebound-goal connectivity used only for move ordering",
        "selection": {
            "normalization": "winner rotated to player zero; horizontal reflection dedup",
            "goal_shell": "winner to move at normalized y 9, 10, or 11",
            "game_cap": "at most four states per game and two per goal-distance band",
            "cross_split_order": ["development", "validation", "test"],
            "development_agents": sorted(DEVELOPMENT_AGENTS),
            "validation_agents": sorted(VALIDATION_AGENTS),
            "locked_test_agent": LOCKED_AGENT_ID,
            "locked_after_game": LOCKED_AFTER_GAME,
        },
        "sources": source_hashes,
        "banks": {
            path: {
                "sha256": sha256_bytes(content),
                "records": content.count(b"\n") - 1,
            }
            for path, content in sorted(banks.items())
        },
        "stages": {
            "initial": {
                "bank": "openings/initial.tsv",
                "node_budget": 5000,
                "maximum_turns": 320,
                "minimum_mean": 0.50,
            },
            "development": {
                "bank": "openings/development.tsv",
                "node_budget": 5000,
                "maximum_turns": 320,
                "minimum_mean": 0.52,
                "minimum_throughput_ratio": 0.90,
                "require_more_wins_than_incumbent": True,
            },
            "validation": {
                "bank": "openings/validation.tsv",
                "node_budget": 5000,
                "maximum_turns": 320,
                "minimum_mean": 0.53,
                "minimum_ci_lower": 0.50,
                "minimum_color_score": 0.48,
                "minimum_stratum_score": 0.48,
                "minimum_throughput_ratio": 0.90,
            },
            "test": {
                "bank": "openings/test.tsv",
                "node_budgets": [30000, 100000],
                "maximum_turns": 320,
                "minimum_mean": 0.50,
                "minimum_ci_lower": 0.45,
                "minimum_color_score": 0.45,
                "minimum_stratum_score": 0.45,
                "minimum_throughput_ratio": 0.90,
            },
        },
        "statistics": {
            "unit": "source_game_cluster_of_color_swapped_opening_pairs",
            "method": "source_game_cluster_percentile_bootstrap",
            "confidence": 0.95,
            "resamples": 10000,
            "seed": 4_774_557_432_748_095_049,
        },
        "timing": {
            "fresh_process_samples": 20,
            "shell_cases": ["elite-d2", "rank1-d1", "elite-d0-dense"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        },
        "source_limit": 100000,
    }
    return banks, stable_json(manifest)


def fetch_locked():
    if LOCKED_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {LOCKED_RECORDS}")
    sys.path.insert(0, str(TOOLS))
    from analyze_arena import fetch_games, record  # noqa: PLC0415

    records = []
    for game in fetch_games(LOCKED_AGENT_ID, wins_only=True):
        item = record(game, LOCKED_AGENT_ID)
        if item is not None and item["won"] and int(item["game_id"]) > LOCKED_AFTER_GAME:
            records.append(item)
    records.sort(key=lambda item: int(item["game_id"]))
    if len(records) < 20:
        raise RuntimeError(f"only {len(records)} eligible locked rank-one wins")
    payload = {
        "schema": "papersoccer.frozen-rank1-winning-games.v1",
        "agent_id": LOCKED_AGENT_ID,
        "selection": f"completed wins with game_id > {LOCKED_AFTER_GAME}",
        "records": records,
    }
    LOCKED_RECORDS.write_bytes(stable_json(payload))
    print(f"froze {len(records)} rank-one wins in {LOCKED_RECORDS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-locked", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.fetch_locked:
        fetch_locked()
    banks, manifest = build_artifacts()
    artifacts = {HERE / path: content for path, content in banks.items()}
    artifacts[HERE / "manifest.json"] = manifest
    stale = []
    for path, content in artifacts.items():
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit("stale promotion artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
