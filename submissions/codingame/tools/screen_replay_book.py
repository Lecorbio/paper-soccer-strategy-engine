#!/usr/bin/env python3

"""Measure which completed arena games a replay book would alter first."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from analyze_arena import fetch_games, record


def lookup(replays, player_id: int, prefix: str):
    completed_turns = 0 if not prefix else prefix.count("/") + 1
    for replay in replays:
        if (
            replay["player_id"] != player_id
            or completed_turns < replay["first_turn"]
            or completed_turns % 2 != player_id
        ):
            continue
        transcript = replay["transcript"]
        if prefix:
            if not transcript.startswith(prefix + "/"):
                continue
            begin = len(prefix) + 1
        else:
            if replay["first_turn"] != 0:
                continue
            begin = 0
        end = transcript.find("/", begin)
        if end < 0:
            end = len(transcript)
        return transcript[begin:end], replay
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_id", type=int)
    parser.add_argument("book")
    arguments = parser.parse_args()

    with open(arguments.book, encoding="utf8") as source:
        replays = json.load(source)["replays"]
    records = [
        item
        for game in fetch_games(arguments.agent_id)
        if (item := record(game, arguments.agent_id)) is not None
    ]
    changed = []
    matched_decisions = 0
    matched_games = set()
    matched_paths = Counter()
    for item in records:
        prefix = ""
        first_change = None
        for turn, played in enumerate(item["turns"]):
            if played["player_id"] == item["player_id"]:
                proposed, replay = lookup(replays, item["player_id"], prefix)
                if proposed is not None:
                    matched_decisions += 1
                    matched_games.add(item["game_id"])
                    matched_paths[replay["label"]] += 1
                    if proposed != played["action"]:
                        first_change = {
                            "turn": turn,
                            "played": played["action"],
                            "proposed": proposed,
                            "path": replay["label"],
                            "source_game_id": replay.get("source_game_id"),
                        }
                        break
            prefix = f"{prefix}/{played['action']}" if prefix else played["action"]
        if first_change is not None:
            changed.append(
                {
                    "game_id": item["game_id"],
                    "result": "win" if item["won"] else "loss",
                    "opponent": item["opponent_name"],
                    "player_id": item["player_id"],
                    **first_change,
                }
            )

    result = {
        "agent_id": arguments.agent_id,
        "games": len(records),
        "matched_decisions_before_divergence": matched_decisions,
        "matched_games": len(matched_games),
        "matched_paths": [
            {"path": path, "decisions": decisions}
            for path, decisions in sorted(
                matched_paths.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "changed_games": len(changed),
        "changed_wins": sum(item["result"] == "win" for item in changed),
        "changed_losses": sum(item["result"] == "loss" for item in changed),
        "changes": changed,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
