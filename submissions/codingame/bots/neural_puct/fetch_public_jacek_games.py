#!/usr/bin/env python3

"""Freeze public Jacek games that are not reserved by the rank-one lock.

The fetcher intentionally has no way to read promotion banks.  Its only
exclusion input is the public game-id lock, and the frozen artifact records
that lock's hash.  This keeps training acquisition independent from sealed
position outcomes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request


HERE = pathlib.Path(__file__).resolve().parent
PROMOTION = HERE.parents[1] / "promotion"
DEFAULT_LOCK = PROMOTION / "rank1_locked_games.json"
DEFAULT_OUTPUT = HERE / "public_jacek_unlocked_v1.json"
SERVICE_ROOT = "https://www.codingame.com/services"
JACEK_AGENT_ID = 6_273_433


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    raise RuntimeError("unreachable retry state")


def load_locked(path: pathlib.Path) -> set[int]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "papersoccer.frozen-rank1-winning-games.v1"
        or payload.get("agent_id") != JACEK_AGENT_ID
        or not isinstance(payload.get("records"), list)
    ):
        raise ValueError("unexpected rank-one lock schema")
    game_ids = [int(record["game_id"]) for record in payload["records"]]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("rank-one lock contains duplicate game ids")
    return set(game_ids)


def game_record(game: dict) -> dict:
    if len(game.get("agents", [])) != 2:
        raise ValueError(f"game {game.get('gameId')} is not a two-player game")
    focus = next(
        (agent for agent in game["agents"] if agent["agentId"] == JACEK_AGENT_ID),
        None,
    )
    if focus is None:
        raise ValueError(f"game {game.get('gameId')} does not contain Jacek")
    opponent = next(
        agent for agent in game["agents"] if agent["agentId"] != JACEK_AGENT_ID
    )
    winner = game["ranks"].index(0)
    turns = [
        {
            "player_id": int(frame["agentId"]),
            "action": (frame.get("stdout") or "").strip(),
        }
        for frame in game["frames"]
        if frame.get("agentId", -1) >= 0
    ]
    if not turns:
        raise ValueError(f"game {game['gameId']} has no turns")
    for turn in turns:
        if turn["player_id"] not in (0, 1):
            raise ValueError(f"game {game['gameId']} has an invalid player id")
        if not turn["action"] or any(ch not in "01234567" for ch in turn["action"]):
            raise ValueError(f"game {game['gameId']} has an invalid action")
    return {
        "focus_agent_id": JACEK_AGENT_ID,
        "focus_name": focus["codingamer"]["pseudo"],
        "focus_score": focus.get("score"),
        "game_id": int(game["gameId"]),
        "opponent_agent_id": int(opponent["agentId"]),
        "opponent_name": opponent["codingamer"]["pseudo"],
        "player_id": int(focus["index"]),
        "transcript": "/".join(turn["action"] for turn in turns),
        "turns": turns,
        "won": winner == focus["index"],
    }


def validate(payload: dict, lock_path: pathlib.Path) -> None:
    if payload.get("schema") != "papersoccer.public-jacek-training-games.v1":
        raise ValueError("unexpected public Jacek corpus schema")
    if payload.get("agent_id") != JACEK_AGENT_ID:
        raise ValueError("public Jacek corpus has the wrong agent")
    if payload.get("locked_games_sha256") != sha256(lock_path):
        raise ValueError("public Jacek corpus is bound to a stale lock")
    locked = load_locked(lock_path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("public Jacek corpus is empty")
    game_ids = [int(record["game_id"]) for record in records]
    if game_ids != sorted(set(game_ids)):
        raise ValueError("public Jacek games are not uniquely sorted")
    overlap = locked.intersection(game_ids)
    if overlap:
        raise ValueError(f"public Jacek corpus contains {len(overlap)} locked games")
    rejected = payload.get("structurally_rejected")
    if not isinstance(rejected, list):
        raise ValueError("public Jacek corpus omits structural rejections")
    rejected_ids = [int(record["game_id"]) for record in rejected]
    if rejected_ids != sorted(set(rejected_ids)):
        raise ValueError("public Jacek structural rejections are not uniquely sorted")
    if set(rejected_ids).intersection(locked):
        raise ValueError("public Jacek structural rejections contain a locked game")
    if set(rejected_ids).intersection(game_ids):
        raise ValueError("public Jacek game is both accepted and rejected")
    if any(
        not isinstance(record.get("reason"), str) or not record["reason"]
        for record in rejected
    ):
        raise ValueError("public Jacek structural rejection omits its reason")
    for record in records:
        if record.get("focus_agent_id") != JACEK_AGENT_ID:
            raise ValueError("public Jacek corpus contains another focus agent")
        if record.get("player_id") not in (0, 1):
            raise ValueError("public Jacek corpus contains an invalid player id")
        turns = record.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("public Jacek corpus contains an empty game")
        transcript = "/".join(turn["action"] for turn in turns)
        if transcript != record.get("transcript"):
            raise ValueError("public Jacek transcript does not match its turns")


def fetch(lock_path: pathlib.Path) -> dict:
    locked = load_locked(lock_path)
    battles = post(
        "gamesPlayersRankingRemoteService/findLastBattlesByAgentId",
        [JACEK_AGENT_ID, None],
    )
    eligible = []
    for battle in battles:
        if battle.get("done") is not True or int(battle["gameId"]) in locked:
            continue
        if not any(
            int(player["playerAgentId"]) == JACEK_AGENT_ID
            for player in battle.get("players", [])
        ):
            continue
        eligible.append(battle)
    eligible.sort(key=lambda battle: int(battle["gameId"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        games = list(
            executor.map(
                lambda battle: post(
                    "gameResultRemoteService/findByGameId",
                    [int(battle["gameId"]), None],
                ),
                eligible,
            )
        )
    records = []
    structurally_rejected = []
    for game in games:
        try:
            records.append(game_record(game))
        except (KeyError, TypeError, ValueError) as error:
            structurally_rejected.append(
                {"game_id": int(game.get("gameId", -1)), "reason": str(error)}
            )
    records.sort(key=lambda item: item["game_id"])
    structurally_rejected.sort(key=lambda item: item["game_id"])
    return {
        "schema": "papersoccer.public-jacek-training-games.v1",
        "agent_id": JACEK_AGENT_ID,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "selection": (
            "all completed games currently returned for agent 6273433, "
            "excluding every game id in the frozen rank-one lock"
        ),
        "locked_games_sha256": sha256(lock_path),
        "returned_battles": len(battles),
        "excluded_locked_games": len(
            locked.intersection(int(battle["gameId"]) for battle in battles)
        ),
        "structurally_rejected": structurally_rejected,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=pathlib.Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        payload = json.loads(arguments.output.read_text())
        validate(payload, arguments.lock)
        print(
            f"Public Jacek corpus is valid ({len(payload['records'])} unlocked games)."
        )
        return
    payload = fetch(arguments.lock)
    validate(payload, arguments.lock)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {arguments.output} ({len(payload['records'])} unlocked games).")


if __name__ == "__main__":
    main()
