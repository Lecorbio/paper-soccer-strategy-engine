#!/usr/bin/env python3

import argparse
import collections
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent
TOOLS = ROOT.parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_arena import fetch_games, record  # noqa: E402


def selected_cases(loss):
    own_turns = [
        index
        for index, turn in enumerate(loss["turns"])
        if turn["player_id"] == loss["player_id"]
    ]
    if not own_turns:
        return []
    selected = {
        own_turns[0],
        own_turns[len(own_turns) // 3],
        own_turns[(2 * len(own_turns)) // 3],
    }
    return [
        {
            "game_id": loss["game_id"],
            "source_agent_id": loss["focus_agent_id"],
            "opponent_agent_id": loss["opponent_agent_id"],
            "player_id": loss["player_id"],
            "prefix": "/".join(
                turn["action"] for turn in loss["turns"][:index]
            ),
            "observed_action": loss["turns"][index]["action"],
            "turn_index": index,
        }
        for index in sorted(selected)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_id", type=int)
    parser.add_argument("expected_games", type=int)
    parser.add_argument("--history-version", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()

    games = fetch_games(arguments.agent_id)
    records = [
        item
        for game in games
        if (item := record(game, arguments.agent_id)) is not None
    ]
    if len(records) != arguments.expected_games:
        raise RuntimeError(
            f"agent {arguments.agent_id} returned {len(records)} complete "
            f"games; expected {arguments.expected_games}"
        )
    losses = [item for item in records if not item["won"]]
    opponents = collections.defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0})
    for item in records:
        summary = opponents[item["opponent_name"]]
        summary["games"] += 1
        summary["wins" if item["won"] else "losses"] += 1

    unique = {}
    for loss in losses:
        for case in selected_cases(loss):
            unique.setdefault((case["player_id"], case["prefix"]), case)
    cases = sorted(
        unique.values(),
        key=lambda item: (item["game_id"], item["turn_index"]),
    )
    score = next(
        (item["focus_score"] for item in records if item["focus_score"] is not None),
        None,
    )
    report = {
        "schema": "papersoccer.completed-arena-loss-regressions.v1",
        "agent_id": arguments.agent_id,
        "history_version": arguments.history_version,
        "score": score,
        "games": len(records),
        "wins": len(records) - len(losses),
        "losses": len(losses),
        "opponents": [
            {"opponent": name, **values}
            for name, values in sorted(opponents.items())
        ],
        "selection": (
            "first, one-third, and two-thirds own-turn states per loss; "
            "deduplicated by player and complete prefix"
        ),
        "loss_records": losses,
        "regression_cases": cases,
    }
    output = arguments.output or ROOT / f"arena_batch_{arguments.agent_id}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {output}: {len(records)} games, {len(losses)} losses, "
        f"{len(cases)} sampled loss states"
    )


if __name__ == "__main__":
    main()
