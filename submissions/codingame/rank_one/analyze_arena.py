#!/usr/bin/env python3

"""Analyze a CodinGame Paper Soccer arena batch and find elite corrections."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict


SERVICE_ROOT = "https://www.codingame.com/services"


def post(service: str, payload):
    for attempt in range(7):
        request = urllib.request.Request(
            f"{SERVICE_ROOT}/{service}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 6:
                raise
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError("unreachable replay download retry state")


def fetch_games(agent_id: int, wins_only: bool = False):
    battles = post(
        "gamesPlayersRankingRemoteService/findLastBattlesByAgentId",
        [agent_id, None],
    )
    battles = [battle for battle in battles if battle.get("done") is True]
    if wins_only:
        battles = [
            battle
            for battle in battles
            if any(
                player["playerAgentId"] == agent_id and player["position"] == 0
                for player in battle["players"]
            )
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        games = list(
            executor.map(
                lambda battle: post(
                    "gameResultRemoteService/findByGameId",
                    [battle["gameId"], None],
                ),
                battles,
            )
        )
    return games


def fetch_wins_against(agent_id: int):
    battles = post(
        "gamesPlayersRankingRemoteService/findLastBattlesByAgentId",
        [agent_id, None],
    )
    battles = [battle for battle in battles if battle.get("done") is True]
    battles = [
        battle
        for battle in battles
        if any(
            player["playerAgentId"] == agent_id and player["position"] == 1
            for player in battle["players"]
        )
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        games = list(
            executor.map(
                lambda battle: post(
                    "gameResultRemoteService/findByGameId",
                    [battle["gameId"], None],
                ),
                battles,
            )
        )
    result = []
    for game in games:
        winner = game["ranks"].index(0)
        winner_agent_id = next(
            agent["agentId"] for agent in game["agents"] if agent["index"] == winner
        )
        result.append(record(game, winner_agent_id))
    return result


def record(game, focus_agent_id: int):
    focus = next(
        (agent for agent in game["agents"] if agent["agentId"] == focus_agent_id),
        None,
    )
    if focus is None:
        return None
    opponent = next(
        agent for agent in game["agents"] if agent["agentId"] != focus_agent_id
    )
    player_id = focus["index"]
    winner = game["ranks"].index(0)
    turns = [
        {
            "player_id": frame["agentId"],
            "action": (frame.get("stdout") or "").strip(),
        }
        for frame in game["frames"]
        if frame.get("agentId", -1) >= 0
    ]
    return {
        "game_id": game["gameId"],
        "focus_agent_id": focus_agent_id,
        "focus_name": focus["codingamer"]["pseudo"],
        "focus_score": focus.get("score"),
        "player_id": player_id,
        "won": winner == player_id,
        "opponent_agent_id": opponent["agentId"],
        "opponent_name": opponent["codingamer"]["pseudo"],
        "turns": turns,
        "transcript": "/".join(turn["action"] for turn in turns),
    }


def compare(loss, elite):
    common = 0
    for left, right in zip(loss["turns"], elite["turns"]):
        if left != right:
            break
        common += 1
    if common >= len(elite["turns"]):
        return None
    elite_turn = elite["turns"][common]
    if elite_turn["player_id"] != elite["player_id"]:
        return None
    # Player 0 cannot identify an opponent before its opening. Preserve the
    # baseline opening and begin only after at least one complete round.
    minimum_turn = 1 if elite["player_id"] == 1 else 2
    if common < minimum_turn:
        return None
    loss_turn = loss["turns"][common] if common < len(loss["turns"]) else None
    return {
        "first_turn": common,
        "loss_action": None if loss_turn is None else loss_turn["action"],
        "elite_action": elite_turn["action"],
    }


def summarize(candidate_records, elite_records):
    losses = [item for item in candidate_records if not item["won"]]
    wins = len(candidate_records) - len(losses)
    by_opponent = defaultdict(lambda: Counter(games=0, wins=0, losses=0))
    for item in candidate_records:
        counter = by_opponent[item["opponent_name"]]
        counter["games"] += 1
        counter["wins" if item["won"] else "losses"] += 1

    comparisons = []
    book = {}
    for loss in losses:
        candidates = []
        for elite in elite_records:
            if (
                elite["won"]
                and elite["opponent_agent_id"] == loss["opponent_agent_id"]
                and elite["player_id"] == loss["player_id"]
            ):
                divergence = compare(loss, elite)
                if divergence is not None:
                    candidates.append((divergence["first_turn"], elite, divergence))
        candidates.sort(
            key=lambda value: (
                -value[0],
                -(value[1]["focus_score"] or 0.0),
                value[1]["game_id"],
            )
        )
        selected = None
        if candidates:
            _, elite, divergence = candidates[0]
            selected = {
                "source_agent_id": elite["focus_agent_id"],
                "source_name": elite["focus_name"],
                "source_game_id": elite["game_id"],
                "source_score": elite["focus_score"],
                **divergence,
                "transcript": elite["transcript"],
            }
            key = (elite["player_id"], elite["transcript"])
            previous = book.get(key)
            if previous is None or divergence["first_turn"] < previous["first_turn"]:
                book[key] = {
                    "player_id": elite["player_id"],
                    "first_turn": divergence["first_turn"],
                    "transcript": elite["transcript"],
                    "source_game_id": elite["game_id"],
                    "source_agent_id": elite["focus_agent_id"],
                    "source_name": elite["focus_name"],
                    "source_score": elite["focus_score"],
                    "opponent_agent_id": loss["opponent_agent_id"],
                    "opponent_name": loss["opponent_name"],
                }
        comparisons.append(
            {
                "loss_game_id": loss["game_id"],
                "opponent_agent_id": loss["opponent_agent_id"],
                "opponent_name": loss["opponent_name"],
                "player_id": loss["player_id"],
                "turn_count": len(loss["turns"]),
                "opening": loss["turns"][:6],
                "ending": loss["turns"][-6:],
                "transcript": loss["transcript"],
                "selected": selected,
            }
        )

    score = next(
        (item["focus_score"] for item in candidate_records if item["focus_score"]),
        None,
    )
    return {
        "agent_id": candidate_records[0]["focus_agent_id"],
        "score": score,
        "games": len(candidate_records),
        "wins": wins,
        "losses": len(losses),
        "by_opponent": [
            {"opponent": name, **dict(values)}
            for name, values in sorted(
                by_opponent.items(), key=lambda item: (-item[1]["games"], item[0])
            )
        ],
        "comparisons": comparisons,
        "book_candidates": sorted(
            book.values(),
            key=lambda item: (
                item["opponent_name"],
                item["player_id"],
                item["source_game_id"],
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_id", type=int)
    parser.add_argument(
        "--elite",
        action="append",
        default=[],
        type=int,
        help="agent id whose wins may supply same-color corrections",
    )
    parser.add_argument(
        "--opponent-pool",
        action="append",
        default=[],
        type=int,
        help="opponent agent id whose public losses may supply winning paths",
    )
    parser.add_argument("--pretty", action="store_true")
    arguments = parser.parse_args()

    downloaded = [fetch_games(arguments.agent_id)]
    downloaded.extend(fetch_games(agent_id, wins_only=True) for agent_id in arguments.elite)
    candidate_records = [
        item
        for game in downloaded[0]
        if (item := record(game, arguments.agent_id)) is not None
    ]
    elite_records = []
    for agent_id, games in zip(arguments.elite, downloaded[1:]):
        elite_records.extend(
            item for game in games if (item := record(game, agent_id)) is not None
        )
    for opponent_id in arguments.opponent_pool:
        elite_records.extend(fetch_wins_against(opponent_id))
    if not candidate_records:
        raise RuntimeError(f"agent {arguments.agent_id} has no public games")
    json.dump(
        summarize(candidate_records, elite_records),
        sys.stdout,
        indent=2 if arguments.pretty else None,
        sort_keys=arguments.pretty,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
