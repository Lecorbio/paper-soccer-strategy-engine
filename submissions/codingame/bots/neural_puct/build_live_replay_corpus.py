#!/usr/bin/env python3

"""Freeze a leakage-safe whole-game live replay training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
DEFAULT_DATA_ROOT = HERE / "live_replay"
SCHEMA = "papersoccer.live-replay-training-snapshot.v1"
RELABEL_INPUT_SCHEMA = "papersoccer.live-replay-relabel-input.v1"


def load_collector():
    path = HERE / "collect_live_replays.py"
    spec = importlib.util.spec_from_file_location("live_replay_collector_api", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load_collector()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode() + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: pathlib.Path, repository: pathlib.Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()


def latest_poll(data_root: pathlib.Path, run_id: str):
    candidates = []
    for path in (data_root / "polls" / run_id).glob("poll-*.json"):
        payload = json.loads(path.read_text())
        candidates.append((int(payload["poll_index"]), path, payload))
    if not candidates:
        raise ValueError(f"run {run_id} has no poll records")
    return max(candidates, key=lambda item: item[0])


def selected_games(data_root: pathlib.Path, run_id: str):
    result = []
    seen = set()
    for discovery_path in sorted((data_root / "discoveries" / run_id).glob("*.json")):
        discovery = json.loads(discovery_path.read_text())
        game_id = int(discovery["game_id"])
        if game_id in seen:
            raise ValueError(f"run {run_id} repeats discovery {game_id}")
        seen.add(game_id)
        records = sorted((data_root / "games" / str(game_id)).glob("*.json"))
        if len(records) != 1:
            raise ValueError(
                f"game {game_id} has {len(records)} accepted record versions"
            )
        record_path = records[0]
        record = json.loads(record_path.read_text())
        if record.get("schema") != COLLECTOR.REPLAY_SCHEMA:
            raise ValueError(f"game {game_id} has an unexpected record schema")
        if canonical_bytes(record) != record_path.read_bytes():
            raise ValueError(f"game {game_id} record is not canonical")
        if sha256_file(record_path) != record_path.stem:
            raise ValueError(f"game {game_id} record hash disagrees with filename")
        if int(record["replay"]["game_id"]) != game_id:
            raise ValueError(f"game {game_id} record embeds another game id")
        result.append((game_id, discovery_path, record_path, record))
    return result


def build_snapshot(
    *,
    repository: pathlib.Path,
    data_root: pathlib.Path,
    run_id: str,
    minimum_games: int,
    own_agent_ids: set[int],
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run id contains unsafe characters")
    games = selected_games(data_root, run_id)
    if len(games) < minimum_games:
        return {
            "decision": "waiting-for-data",
            "minimum_games": minimum_games,
            "observed_games": len(games),
            "run_id": run_id,
        }

    exclusions = COLLECTOR.build_exclusion_registry(repository, data_root)
    exclusion_bytes = canonical_bytes(exclusions.payload)
    exclusion_hash = sha256_bytes(exclusion_bytes)
    exclusion_path = data_root / "exclusions" / f"{exclusion_hash}.json"
    COLLECTOR.write_once(exclusion_path, exclusion_bytes)

    poll_index, poll_path, poll = latest_poll(data_root, run_id)
    records = []
    relabel_rows: list[str] = []
    direct_primitives = 0
    self_primitives = 0
    tier_games: dict[str, int] = {}
    for game_id, discovery_path, record_path, record in games:
        if exclusions.is_protected(game_id):
            raise ValueError(f"accepted live game {game_id} crossed a protected boundary")
        replay = record["replay"]
        agents = replay["agents"]
        direct = [agent for agent in agents if agent["label_role"] == "direct-public-expert"]
        own = [
            agent
            for agent in agents
            if int(agent["agent_id"]) in own_agent_ids
            and agent["label_role"] == "self-relabel-only"
        ]
        if not direct:
            raise ValueError(f"game {game_id} has no direct strong-player label source")
        if len(own) > 1:
            raise ValueError(f"game {game_id} contains multiple owned agents")
        for agent in direct:
            tier = agent.get("strength_tier") or {}
            name = str(tier.get("name") or "")
            mass = tier.get("policy_mass")
            if name not in {"elite-1-5", "strong-6-10", "upper-11-20"} or mass not in {
                1.0,
                0.75,
                0.5,
            }:
                raise ValueError(f"game {game_id} has an invalid frozen strength tier")
            tier_games[name] = tier_games.get(name, 0) + 1
        turns = replay["turns"]
        direct_players = {int(agent["player_id"]) for agent in direct}
        direct_primitives += sum(
            len(turn["action"])
            for turn in turns
            if int(turn["player_id"]) in direct_players
        )
        own_player = int(own[0]["player_id"]) if own else None
        if own_player is not None:
            self_primitives += sum(
                len(turn["action"])
                for turn in turns
                if int(turn["player_id"]) == own_player
            )
            encoded_turns = "/".join(
                f"{int(turn['player_id'])}:{turn['action']}" for turn in turns
            )
            relabel_rows.append(
                "\t".join(
                    (
                        str(game_id),
                        record_path.stem,
                        str(int(replay["winner_player_id"])),
                        str(int(own[0]["agent_id"])),
                        str(own_player),
                        encoded_turns,
                    )
                )
            )
        records.append(
            {
                "direct_experts": [
                    {
                        "agent_id": int(agent["agent_id"]),
                        "player_id": int(agent["player_id"]),
                        "rank": int(agent["rank"]),
                        "score": float(agent["score"]),
                        "strength_tier": agent["strength_tier"],
                    }
                    for agent in sorted(direct, key=lambda item: int(item["player_id"]))
                ],
                "discovery_path": relative(discovery_path, repository),
                "discovery_sha256": sha256_file(discovery_path),
                "game_id": game_id,
                "own_agent_id": int(own[0]["agent_id"]) if own else None,
                "own_player_id": own_player,
                "record_path": relative(record_path, repository),
                "record_sha256": record_path.stem,
            }
        )

    relabel_header = "\t".join(
        (RELABEL_INPUT_SCHEMA, run_id, str(poll_index), str(len(relabel_rows)))
    )
    relabel_bytes = ("\n".join((relabel_header, *relabel_rows)) + "\n").encode()
    relabel_hash = sha256_bytes(relabel_bytes)
    corpus_root = data_root / "corpora"
    relabel_path = corpus_root / f"{relabel_hash}.relabel.tsv"
    COLLECTOR.write_once(relabel_path, relabel_bytes)
    snapshot = {
        "schema": SCHEMA,
        "run_id": run_id,
        "frozen_at_utc": poll["completed_at_utc"],
        "minimum_independent_games": minimum_games,
        "independent_games": len(records),
        "own_agent_ids": sorted(own_agent_ids),
        "exclusion_registry_path": relative(exclusion_path, repository),
        "exclusion_registry_sha256": exclusion_hash,
        "poll_path": relative(poll_path, repository),
        "poll_sha256": sha256_file(poll_path),
        "collector_sha256": poll["collector_sha256"],
        "direct_expert_primitives": direct_primitives,
        "self_primitives_for_relabel": self_primitives,
        "strength_tier_games": dict(sorted(tier_games.items())),
        "relabel_input": {
            "path": relative(relabel_path, repository),
            "sha256": relabel_hash,
            "games": len(relabel_rows),
        },
        "split_policy": (
            "whole games before primitive expansion; reflections inherit their game; "
            "trainer purges canonical train/validation/test overlap in that order"
        ),
        "value_policy": (
            "final outcomes diagnostic only; live value samples require deeper-search "
            "or temporal-difference targets"
        ),
        "records": records,
    }
    snapshot_bytes = canonical_bytes(snapshot)
    snapshot_hash = sha256_bytes(snapshot_bytes)
    snapshot_path = corpus_root / f"{snapshot_hash}.json"
    COLLECTOR.write_once(snapshot_path, snapshot_bytes)
    return {
        "decision": "snapshot-created",
        "manifest_path": relative(snapshot_path, repository),
        "manifest_sha256": snapshot_hash,
        "relabel_input_path": relative(relabel_path, repository),
        "relabel_input_sha256": relabel_hash,
        "independent_games": len(records),
        "direct_expert_primitives": direct_primitives,
        "self_primitives_for_relabel": self_primitives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minimum-games", type=int, default=50)
    parser.add_argument("--own-agent-id", action="append", type=int, required=True)
    parser.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    arguments = parser.parse_args()
    if arguments.minimum_games <= 0:
        parser.error("--minimum-games must be positive")
    result = build_snapshot(
        repository=arguments.repository.resolve(),
        data_root=arguments.data_root.resolve(),
        run_id=arguments.run_id,
        minimum_games=arguments.minimum_games,
        own_agent_ids=set(arguments.own_agent_id),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
