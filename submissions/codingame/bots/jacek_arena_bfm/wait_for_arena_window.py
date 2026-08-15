#!/usr/bin/env python3
"""Wait for one exact CodinGame arena window without reading game details.

Only ``findLastBattlesByAgentId`` is queried.  A battle is complete for the
gate only when it is marked done and has two well-formed players.  Transitional
one-player and/or ``done=false`` records remain pending.  The generic collector
can be launched only for an exact, source-bound window of 90 games.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SERVICE_ROOT = "https://www.codingame.com/services"
BATTLE_LIST_SERVICE = (
    "gamesPlayersRankingRemoteService/findLastBattlesByAgentId"
)
BATTLE_LIST_URL = f"{SERVICE_ROOT}/{BATTLE_LIST_SERVICE}"
EXPECTED_GAME_COUNT = 90
REPORT_SCHEMA = "papersoccer.jacek-arena-bfm.arena-window-wait.v1"

# These keys belong to replay/game-detail payloads, never to the metadata
# needed by this gate.  Reject them before inspecting their values.
FORBIDDEN_DETAIL_KEYS = frozenset(
    {
        "agents",
        "frames",
        "gameinformation",
        "inputs",
        "observedtranscript",
        "outputs",
        "replay",
        "stderr",
        "stdin",
        "stdout",
        "transcript",
        "turns",
    }
)


class MetadataError(RuntimeError):
    """The battle-list response cannot safely prove an exact window."""


class CollectorRefused(RuntimeError):
    """The collector command is unsafe or the window is not exactly ready."""


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_detail_content(value: Any, path: str = "$") -> None:
    """Validate JSON structure while refusing any game-detail field.

    This function deliberately checks a forbidden mapping key before visiting
    its value, so accidentally embedded replay content is not consumed.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise MetadataError(f"non-string JSON key at {path}")
            if _canonical_key(key) in FORBIDDEN_DETAIL_KEYS:
                raise MetadataError(
                    f"game-detail field {key!r} is forbidden in battle metadata"
                )
            _reject_detail_content(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_detail_content(child, f"{path}[{index}]")
        return
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise MetadataError(f"non-JSON metadata value at {path}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _game_id(battle: Mapping[str, Any], index: int) -> str | int:
    value = battle.get("gameId")
    if _is_integer(value):
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise MetadataError(f"battle {index} has no valid gameId")


def _players(battle: Mapping[str, Any], index: int) -> list[Mapping[str, Any]]:
    value = battle.get("players")
    if not isinstance(value, list):
        raise MetadataError(f"battle {index} players is not a list")
    players: list[Mapping[str, Any]] = []
    for player_index, player in enumerate(value):
        if not isinstance(player, Mapping):
            raise MetadataError(
                f"battle {index} player {player_index} is not an object"
            )
        players.append(player)
    return players


def _focus_players(
    players: Sequence[Mapping[str, Any]], agent_id: int
) -> list[Mapping[str, Any]]:
    return [player for player in players if player.get("playerAgentId") == agent_id]


def _validate_complete_players(
    players: Sequence[Mapping[str, Any]], battle_index: int
) -> None:
    if len(players) != 2:
        raise MetadataError(
            f"battle {battle_index} was classified complete without two players"
        )
    positions = [player.get("position") for player in players]
    if any(not _is_integer(position) for position in positions):
        raise MetadataError(f"battle {battle_index} has an invalid player position")
    if set(positions) != {0, 1}:
        raise MetadataError(
            f"battle {battle_index} complete-player positions are not 0 and 1"
        )
    agent_ids = [player.get("playerAgentId") for player in players]
    if any(not _is_integer(player_agent_id) for player_agent_id in agent_ids):
        raise MetadataError(f"battle {battle_index} has an invalid playerAgentId")
    if len(set(agent_ids)) != 2:
        raise MetadataError(f"battle {battle_index} repeats a playerAgentId")


def classify_battle_metadata(
    battles: Any,
    *,
    agent_id: int,
    submission_id: int,
) -> dict[str, Any]:
    """Return a fail-closed exact-window report from battle-list metadata."""

    if not _is_integer(agent_id) or agent_id <= 0:
        raise MetadataError("agent_id must be a positive integer")
    if not _is_integer(submission_id) or submission_id <= 0:
        raise MetadataError("submission_id must be a positive integer")
    if not isinstance(battles, list):
        raise MetadataError("battle-list response is not a JSON list")
    _reject_detail_content(battles)

    complete_ids: list[str | int] = []
    pending: list[dict[str, Any]] = []
    ignored_other_submission = 0
    unrelated = 0
    seen_game_ids: set[str] = set()

    for index, raw_battle in enumerate(battles):
        if not isinstance(raw_battle, Mapping):
            raise MetadataError(f"battle {index} is not an object")
        game_id = _game_id(raw_battle, index)
        game_key = str(game_id)
        if game_key in seen_game_ids:
            raise MetadataError(f"duplicate gameId in battle metadata: {game_id}")
        seen_game_ids.add(game_key)
        players = _players(raw_battle, index)
        focus = _focus_players(players, agent_id)
        if len(focus) > 1:
            raise MetadataError(f"battle {index} repeats the focus agent")
        if not focus:
            unrelated += 1
            continue

        focus_submission = focus[0].get("submissionId")
        if focus_submission != submission_id:
            # A missing submission ID on an incomplete focus-agent record is
            # ambiguous and must block collection rather than being mistaken
            # for an older submission.
            if focus_submission is None and (
                raw_battle.get("done") is not True or len(players) != 2
            ):
                pending.append(
                    {
                        "game_id": game_id,
                        "reasons": ["focus_submission_missing"],
                    }
                )
            else:
                ignored_other_submission += 1
            continue

        done = raw_battle.get("done")
        if done is not None and not isinstance(done, bool):
            raise MetadataError(f"battle {index} done is not boolean")
        reasons: list[str] = []
        if done is not True:
            reasons.append("not_done")
        if len(players) != 2:
            reasons.append(f"player_count_{len(players)}")
        if reasons:
            pending.append({"game_id": game_id, "reasons": reasons})
            continue

        _validate_complete_players(players, index)
        complete_ids.append(game_id)

    complete_ids.sort(key=str)
    pending.sort(key=lambda item: str(item["game_id"]))
    matching_count = len(complete_ids) + len(pending)
    exact_complete = (
        len(complete_ids) == EXPECTED_GAME_COUNT
        and not pending
        and matching_count == EXPECTED_GAME_COUNT
    )
    return {
        "schema": REPORT_SCHEMA,
        "agent_id": agent_id,
        "submission_id": submission_id,
        "expected_games": EXPECTED_GAME_COUNT,
        "snapshot_battle_count": len(battles),
        "matching_battle_count": matching_count,
        "complete_two_player_count": len(complete_ids),
        "pending_count": len(pending),
        "ignored_other_submission_count": ignored_other_submission,
        "unrelated_battle_count": unrelated,
        "complete_game_ids": complete_ids,
        "pending": pending,
        "overfull": matching_count > EXPECTED_GAME_COUNT,
        "collector_permitted": exact_complete,
        "detail_requests": 0,
        "metadata_service": BATTLE_LIST_SERVICE,
    }


def fetch_battle_metadata(
    agent_id: int,
    *,
    timeout_seconds: float = 30.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """Fetch only the agent battle-list endpoint, retrying HTTP 429."""

    payload = json.dumps([agent_id, None], separators=(",", ":")).encode("ascii")
    request = urllib.request.Request(
        BATTLE_LIST_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(7):
        try:
            with opener(request, timeout=timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 6:
                raise
            sleeper(0.5 * (2**attempt))
    raise RuntimeError("unreachable battle-list retry state")


def wait_for_exact_window(
    fetcher: Callable[[], Any],
    *,
    agent_id: int,
    submission_id: int,
    poll_seconds: float,
    timeout_seconds: float,
    progress: Callable[[dict[str, Any]], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll metadata until exactly 90 complete games or the timeout expires."""

    if poll_seconds < 0 or timeout_seconds < 0:
        raise ValueError("poll and timeout seconds must be non-negative")
    started = monotonic()
    while True:
        report = classify_battle_metadata(
            fetcher(), agent_id=agent_id, submission_id=submission_id
        )
        if progress is not None:
            progress(report)
        if report["collector_permitted"]:
            return report
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            report["timed_out"] = True
            report["waited_seconds"] = elapsed
            return report
        sleeper(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _flag_value(command: Sequence[str], flag: str) -> str:
    values: list[str] = []
    for index, token in enumerate(command):
        if token == flag:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise CollectorRefused(f"collector command has no value for {flag}")
            values.append(command[index + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1:
        raise CollectorRefused(
            f"collector command must contain exactly one {flag} binding"
        )
    return values[0]


def validate_collector_command(
    command: Sequence[str], *, agent_id: int, submission_id: int
) -> list[str]:
    """Require the generic collector to be bound to this exact 90-game window."""

    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise CollectorRefused("collector command is empty")
    script_indexes = [
        index for index, token in enumerate(normalized)
        if pathlib.PurePath(token).name == "collect_arena_batch.py"
    ]
    if len(script_indexes) != 1:
        raise CollectorRefused(
            "collector command must invoke collect_arena_batch.py exactly once"
        )
    script_index = script_indexes[0]
    direct = script_index == 0
    through_python = (
        script_index == 1
        and re.fullmatch(
            r"python(?:3(?:\.\d+)?)?",
            pathlib.PurePath(normalized[0]).name,
        )
        is not None
    )
    through_env_python = (
        script_index == 2
        and pathlib.PurePath(normalized[0]).name == "env"
        and re.fullmatch(
            r"python(?:3(?:\.\d+)?)?",
            pathlib.PurePath(normalized[1]).name,
        )
        is not None
    )
    if not (direct or through_python or through_env_python):
        raise CollectorRefused(
            "collect_arena_batch.py must be the executed script, not an argument"
        )
    try:
        bound_agent = int(_flag_value(normalized, "--agent-id"))
        bound_submission = int(_flag_value(normalized, "--submission-id"))
        expected_games = int(_flag_value(normalized, "--expected-games"))
    except ValueError as error:
        raise CollectorRefused("collector bindings must be integers") from error
    if bound_agent != agent_id:
        raise CollectorRefused("collector --agent-id does not match the wait gate")
    if bound_submission != submission_id:
        raise CollectorRefused(
            "collector --submission-id does not match the wait gate"
        )
    if expected_games != EXPECTED_GAME_COUNT:
        raise CollectorRefused("collector --expected-games must be exactly 90")
    return normalized


def run_collector_if_ready(
    report: Mapping[str, Any],
    command: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    """Invoke the validated collector only after the exact-window proof."""

    complete_game_ids = report.get("complete_game_ids")
    exact_ids = (
        isinstance(complete_game_ids, list)
        and len(complete_game_ids) == EXPECTED_GAME_COUNT
        and len({str(value) for value in complete_game_ids})
        == EXPECTED_GAME_COUNT
    )
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("expected_games") != EXPECTED_GAME_COUNT
        or report.get("complete_two_player_count") != EXPECTED_GAME_COUNT
        or report.get("pending_count") != 0
        or report.get("pending") != []
        or report.get("matching_battle_count") != EXPECTED_GAME_COUNT
        or report.get("collector_permitted") is not True
        or report.get("detail_requests") != 0
        or report.get("metadata_service") != BATTLE_LIST_SERVICE
        or not exact_ids
    ):
        raise CollectorRefused(
            "collector refused: the metadata window is not exactly 90 complete games"
        )
    agent_id = report.get("agent_id")
    submission_id = report.get("submission_id")
    if not _is_integer(agent_id) or not _is_integer(submission_id):
        raise CollectorRefused("collector refused: invalid report identity")
    validated = validate_collector_command(
        command, agent_id=agent_id, submission_id=submission_id
    )
    completed = runner(validated, check=False)
    return int(completed.returncode)


def _read_metadata(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _progress(report: Mapping[str, Any]) -> None:
    print(
        "arena-window "
        f"complete={report['complete_two_player_count']}/90 "
        f"pending={report['pending_count']} "
        f"matching={report['matching_battle_count']} "
        f"permitted={str(report['collector_permitted']).lower()}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument(
        "--metadata",
        type=pathlib.Path,
        help="offline battle-list JSON; classify once without any HTTP request",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--collector-command",
        nargs=argparse.REMAINDER,
        help="optional exact collect_arena_batch.py command (must be last)",
    )
    arguments = parser.parse_args()

    try:
        if arguments.collector_command:
            # Validate binding before waiting, but do not execute it yet.
            validate_collector_command(
                arguments.collector_command,
                agent_id=arguments.agent_id,
                submission_id=arguments.submission_id,
            )
        if arguments.metadata is not None:
            report = classify_battle_metadata(
                _read_metadata(arguments.metadata),
                agent_id=arguments.agent_id,
                submission_id=arguments.submission_id,
            )
            _progress(report)
        else:
            report = wait_for_exact_window(
                lambda: fetch_battle_metadata(
                    arguments.agent_id,
                    timeout_seconds=arguments.request_timeout_seconds,
                ),
                agent_id=arguments.agent_id,
                submission_id=arguments.submission_id,
                poll_seconds=arguments.poll_seconds,
                timeout_seconds=arguments.timeout_seconds,
                progress=_progress,
            )
        collector_returncode: int | None = None
        if arguments.collector_command:
            collector_returncode = run_collector_if_ready(
                report, arguments.collector_command
            )
            report["collector_invoked"] = True
            report["collector_returncode"] = collector_returncode
        else:
            report["collector_invoked"] = False
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        if collector_returncode is not None:
            return collector_returncode
        return 0 if report["collector_permitted"] else 2
    except (CollectorRefused, MetadataError, OSError, ValueError) as error:
        parser.exit(1, f"arena-window gate failure: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
