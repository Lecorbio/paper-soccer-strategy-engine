#!/usr/bin/env python3

"""Collect append-only public CodinGame Paper Soccer replays safely.

The collector deliberately separates metadata discovery from replay retrieval:
it freezes a game-id exclusion registry first, snapshots the leaderboard and
battle windows second, and only then requests details for genuinely unseen,
training-eligible games.  Every response and decision is content addressed;
existing files are verified and never replaced.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
DEFAULT_DATA_ROOT = HERE / "live_replay"
SERVICE_ROOT = "https://www.codingame.com/services"
PUZZLE_PUBLIC_ID = "paper-soccer"
COLLECTOR_SCHEMA = "papersoccer.live-replay-collector.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"
REPLAY_SCHEMA = "papersoccer.codingame-live-replay.v1"
POLL_SCHEMA = "papersoccer.live-replay-poll.v1"
RUN_SCHEMA = "papersoccer.live-replay-run.v1"
REQUEST_SCHEMAS = {
    "leaderboard-v1": {
        "service": "Leaderboards/getFilteredPuzzleLeaderboard",
        "body": [
            "puzzle_public_id:string",
            "codingamer_public_handle:null|string",
            "tab:'global'",
            "filter:{active:boolean,column:string,filter:string}",
        ],
    },
    "agent-battles-v1": {
        "service": (
            "gamesPlayersRankingRemoteService/findLastBattlesByAgentId"
        ),
        "body": ["agent_id:integer", "remote_cursor:null"],
    },
    "game-detail-v1": {
        "service": "gameResultRemoteService/findByGameId",
        "body": ["game_id:integer", "viewer_id:null"],
    },
}
TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: pathlib.Path) -> str:
    return digest_bytes(path.read_bytes())


def write_once(path: pathlib.Path, content: bytes) -> bool:
    """Atomically create *path* without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"append-only path has conflicting content: {path}")
        return False
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(
                    f"append-only path raced with conflicting content: {path}"
                )
            created = False
    finally:
        temporary.unlink(missing_ok=True)
    return created


def write_content_addressed(directory: pathlib.Path, value: Any) -> tuple[str, pathlib.Path]:
    content = canonical_json_bytes(value)
    digest = digest_bytes(content)
    path = directory / f"{digest}.json"
    write_once(path, content)
    return digest, path


def relative_to_repository(path: pathlib.Path, repository: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tracked_evidence_paths(repository: pathlib.Path) -> list[pathlib.Path]:
    command = subprocess.run(
        ["git", "ls-files", "-z", "submissions/codingame"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    result = []
    for raw in command.stdout.split(b"\0"):
        if not raw:
            continue
        path = repository / os.fsdecode(raw)
        if path.suffix in {".json", ".jsonl", ".tsv"} and path.is_file():
            result.append(path)
    return sorted(result)


def extract_game_ids(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"game_id", "gameId"} and not isinstance(child, bool):
                try:
                    game_id = int(child)
                except (TypeError, ValueError):
                    pass
                else:
                    if game_id > 0:
                        result.add(game_id)
            else:
                result.update(extract_game_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(extract_game_ids(child))
    return result


def ids_from_path(path: pathlib.Path) -> set[int]:
    if path.suffix == ".tsv":
        # Opening banks are boundary artifacts, not replay-id sources.  They
        # are hashed but never parsed by the collector.
        return set()
    if path.suffix == ".jsonl":
        result: set[int] = set()
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                result.update(extract_game_ids(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        return result
    return extract_game_ids(json.loads(path.read_text()))


def evidence_category(relative: str) -> str:
    name = pathlib.PurePosixPath(relative).name
    if name.startswith("elite_final_holdout_v"):
        return "training_anchor"
    if name == "public_jacek_unlocked_v1.json":
        return "training_anchor"
    if name == "neural_puct_model.json":
        return "known_training_report"
    if name == "rank1_locked_games.json":
        return "protected_rank1_lock"
    if "arena_batch_" in name or "arena_loss" in name:
        return "protected_arena_diagnostic"
    if "/promotion/" in f"/{relative}":
        if "development" in relative:
            return "protected_development"
        return "protected_evaluation"
    return "known_local_evidence"


@dataclasses.dataclass(frozen=True)
class ExclusionRegistry:
    payload: dict[str, Any]

    @property
    def records(self) -> dict[int, dict[str, Any]]:
        return {
            int(record["game_id"]): record for record in self.payload["records"]
        }

    @property
    def known_ids(self) -> set[int]:
        return set(self.records)

    def is_protected(self, game_id: int) -> bool:
        record = self.records.get(game_id)
        return bool(
            record
            and any(
                str(category).startswith("protected_")
                for category in record["categories"]
            )
        )


def build_exclusion_registry(
    repository: pathlib.Path,
    data_root: pathlib.Path,
    *,
    evidence_paths: Iterable[pathlib.Path] | None = None,
) -> ExclusionRegistry:
    """Build the deterministic id-only boundary before any network request."""

    paths = (
        list(evidence_paths)
        if evidence_paths is not None
        else tracked_evidence_paths(repository)
    )
    sources = []
    by_game: dict[int, dict[str, set[str]]] = {}
    for path in sorted({candidate.resolve() for candidate in paths}):
        relative = relative_to_repository(path, repository)
        if "/live_replay/" in f"/{relative}":
            continue
        category = evidence_category(relative)
        game_ids = ids_from_path(path)
        sources.append(
            {
                "category": category,
                "game_id_count": len(game_ids),
                "path": relative,
                "sha256": digest_file(path),
            }
        )
        for game_id in game_ids:
            entry = by_game.setdefault(
                game_id, {"categories": set(), "sources": set()}
            )
            entry["categories"].add(category)
            entry["sources"].add(relative)

    # Previously accepted live replays are known even before their files are
    # tracked.  Only their id and content hash enter this boundary registry.
    replay_root = data_root / "replay_payloads"
    if replay_root.exists():
        for game_directory in sorted(replay_root.iterdir()):
            if not game_directory.is_dir() or not game_directory.name.isdigit():
                continue
            game_id = int(game_directory.name)
            payloads = sorted(game_directory.glob("*.json"))
            if not payloads:
                continue
            relative = relative_to_repository(game_directory, repository)
            sources.append(
                {
                    "category": "live_replay_archive",
                    "game_id_count": 1,
                    "path": relative,
                    "sha256": digest_bytes(
                        canonical_json_bytes(
                            [
                                {"name": path.name, "sha256": digest_file(path)}
                                for path in payloads
                            ]
                        )
                    ),
                }
            )
            entry = by_game.setdefault(
                game_id, {"categories": set(), "sources": set()}
            )
            entry["categories"].add("live_replay_archive")
            entry["sources"].add(relative)

    # Permanent structural rejections and payload conflicts are also known
    # outcomes.  Bind only their audit hashes and reasons so later runs do not
    # re-request or repeatedly revalidate the same unusable response.  Network
    # failures remain retryable and therefore never enter this registry.
    permanent_outcomes: dict[int, dict[str, Any]] = {}
    for path in sorted((data_root / "events").glob("*/*/rejection/*.json")):
        outcome = json.loads(path.read_text())
        if outcome.get("status") != "structural-rejection":
            continue
        game_id = int(outcome["game_id"])
        entry = permanent_outcomes.setdefault(
            game_id,
            {"category": "live_replay_structural_rejection", "files": []},
        )
        entry["files"].append({"path": path, "sha256": digest_file(path)})
    for path in sorted((data_root / "conflicts").glob("*/*.json")):
        outcome = json.loads(path.read_text())
        game_id = int(outcome["game_id"])
        entry = permanent_outcomes.setdefault(
            game_id, {"category": "live_replay_conflict", "files": []}
        )
        entry["category"] = "live_replay_conflict"
        entry["files"].append({"path": path, "sha256": digest_file(path)})
    for game_id, outcome in sorted(permanent_outcomes.items()):
        logical_path = f"{relative_to_repository(data_root, repository)}/outcomes/{game_id}"
        sources.append(
            {
                "category": outcome["category"],
                "game_id_count": 1,
                "path": logical_path,
                "sha256": digest_bytes(
                    canonical_json_bytes(
                        [
                            {
                                "path": relative_to_repository(item["path"], repository),
                                "sha256": item["sha256"],
                            }
                            for item in outcome["files"]
                        ]
                    )
                ),
            }
        )
        entry = by_game.setdefault(
            game_id, {"categories": set(), "sources": set()}
        )
        entry["categories"].add(outcome["category"])
        entry["sources"].add(logical_path)

    payload = {
        "schema": EXCLUSION_SCHEMA,
        "selection": (
            "game-id fields only from tracked CodinGame JSON/JSONL evidence; "
            "opening TSV banks are hash-bound without parsing; root matches.json "
            "is neither read nor modified"
        ),
        "sources": sorted(sources, key=lambda item: (item["path"], item["category"])),
        "records": [
            {
                "categories": sorted(value["categories"]),
                "game_id": game_id,
                "sources": sorted(value["sources"]),
            }
            for game_id, value in sorted(by_game.items())
        ],
    }
    return ExclusionRegistry(payload)


@dataclasses.dataclass(frozen=True)
class ApiResponse:
    body: bytes
    status: int = 200
    headers: Mapping[str, str] = dataclasses.field(default_factory=dict)
    attempts: int = 1


class PublicApi:
    """Minimal public POST client with bounded retries and 429 backoff."""

    def __init__(
        self,
        *,
        service_root: str = SERVICE_ROOT,
        timeout_seconds: float = 30.0,
        maximum_attempts: int = 7,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.service_root = service_root.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.maximum_attempts = maximum_attempts
        self.sleep = sleep

    def post(self, service: str, payload: Any) -> ApiResponse:
        encoded = canonical_json_bytes(payload).rstrip(b"\n")
        for attempt in range(1, self.maximum_attempts + 1):
            request = urllib.request.Request(
                f"{self.service_root}/{service}",
                data=encoded,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "paper-soccer-live-replay-collector/1",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return ApiResponse(
                        body=response.read(),
                        status=int(response.status),
                        headers={
                            key.lower(): value
                            for key, value in response.headers.items()
                            if key.lower() in {"content-type", "date", "retry-after"}
                        },
                        attempts=attempt,
                    )
            except urllib.error.HTTPError as error:
                if (
                    error.code not in TRANSIENT_HTTP_STATUS
                    or attempt == self.maximum_attempts
                ):
                    raise
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                delay = max(delay, 0.5 * (2 ** (attempt - 1)))
                self.sleep(min(delay, 30.0))
            except urllib.error.URLError:
                if attempt == self.maximum_attempts:
                    raise
                self.sleep(min(0.5 * (2 ** (attempt - 1)), 30.0))
        raise RuntimeError("unreachable public API retry state")


@dataclasses.dataclass(frozen=True)
class StoredResponse:
    request_key: str
    request_schema: str
    service: str
    payload: Any
    fetched_at_utc: str
    raw_sha256: str
    normalized_sha256: str
    normalized: Any
    status: int
    attempts: int
    cached: bool


def request_record(request_schema: str, service: str, payload: Any) -> dict[str, Any]:
    return {
        "body": payload,
        "body_schema": REQUEST_SCHEMAS[request_schema]["body"],
        "method": "POST",
        "request_schema": request_schema,
        "service": service,
    }


def request_key(request_schema: str, service: str, payload: Any) -> str:
    return digest_bytes(canonical_json_bytes(request_record(request_schema, service, payload)))


def strength_tier(rank: int) -> dict[str, Any] | None:
    if 1 <= rank <= 5:
        return {"name": "elite-1-5", "policy_mass": 1.0}
    if 6 <= rank <= 10:
        return {"name": "strong-6-10", "policy_mass": 0.75}
    if 11 <= rank <= 20:
        return {"name": "upper-11-20", "policy_mass": 0.5}
    return None


def leaderboard_entries(payload: Any, maximum_rank: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("users"), list):
        raise ValueError("leaderboard response omits users")
    result = []
    seen_agents: set[int] = set()
    for raw in payload["users"]:
        if not isinstance(raw, dict):
            raise ValueError("leaderboard contains a non-object user")
        try:
            rank = int(raw["rank"])
            agent_id = int(raw["agentId"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("leaderboard user omits rank or agent id") from error
        if rank < 1 or rank > maximum_rank:
            continue
        if agent_id in seen_agents:
            raise ValueError(f"leaderboard repeats agent {agent_id}")
        seen_agents.add(agent_id)
        codingamer = raw.get("codingamer") or {}
        result.append(
            {
                "agent_id": agent_id,
                "in_progress": bool(raw.get("inProgress", False)),
                "name": str(raw.get("pseudo") or codingamer.get("pseudo") or ""),
                "programming_language": raw.get("programmingLanguage"),
                "public_handle": codingamer.get("publicHandle"),
                "rank": rank,
                "score": float(raw["score"]),
                "session_id": raw.get("testSessionHandle"),
                "strength_tier": strength_tier(rank),
                "user_id": codingamer.get("userId"),
            }
        )
    result.sort(key=lambda item: (item["rank"], item["agent_id"]))
    if not result or result[0]["rank"] != 1:
        raise ValueError("leaderboard response does not contain rank one")
    return result


def normalize_battle(raw: Any, observed_agent_id: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("battle metadata entry is not an object")
    try:
        game_id = int(raw["gameId"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("battle metadata omits game id") from error
    players = raw.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError(f"game {game_id} is not a two-player battle")
    normalized_players = []
    for player in players:
        try:
            agent_id = int(player["playerAgentId"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"game {game_id} player omits agent id") from error
        normalized_players.append(
            {
                "agent_id": agent_id,
                "name": str(player.get("nickname") or ""),
                "public_handle": player.get("publicHandle"),
                "result_position": player.get("position"),
                "session_id": player.get("testSessionHandle"),
                "submission_id": player.get("submissionId"),
                "user_id": player.get("userId"),
            }
        )
    if len({player["agent_id"] for player in normalized_players}) != 2:
        raise ValueError(f"game {game_id} repeats an agent id")
    if observed_agent_id not in {
        player["agent_id"] for player in normalized_players
    }:
        raise ValueError(
            f"game {game_id} does not contain observed agent {observed_agent_id}"
        )
    return {
        "done": raw.get("done") is True,
        "game_id": game_id,
        "players": sorted(normalized_players, key=lambda item: item["agent_id"]),
    }


def battle_identity(battle: dict[str, Any]) -> bytes:
    return canonical_json_bytes(
        {"done": battle["done"], "game_id": battle["game_id"], "players": battle["players"]}
    )


_TRAINER = None
_TRAINER_LOCK = threading.Lock()


def trainer_module():
    global _TRAINER
    with _TRAINER_LOCK:
        if _TRAINER is None:
            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            import train_neural_puct  # pylint: disable=import-outside-toplevel

            _TRAINER = train_neural_puct
    return _TRAINER


def validate_replay_detail(
    payload: Any,
    *,
    game_id: int,
    metadata: dict[str, Any],
    leaderboard_by_agent: dict[int, dict[str, Any]],
    own_agent_ids: set[int],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or int(payload.get("gameId", -1)) != game_id:
        raise ValueError(f"game detail id does not match requested game {game_id}")
    raw_agents = payload.get("agents")
    ranks = payload.get("ranks")
    frames = payload.get("frames")
    if not isinstance(raw_agents, list) or len(raw_agents) != 2:
        raise ValueError(f"game {game_id} detail is not two-player")
    if not isinstance(ranks, list) or len(ranks) != 2 or ranks.count(0) != 1:
        raise ValueError(f"game {game_id} has invalid ranks")
    if not isinstance(frames, list):
        raise ValueError(f"game {game_id} omits frames")

    agents = []
    indexes: set[int] = set()
    agent_ids: set[int] = set()
    metadata_by_agent = {
        int(player["agent_id"]): player for player in metadata["players"]
    }
    for raw in raw_agents:
        try:
            agent_id = int(raw["agentId"])
            player_id = int(raw["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"game {game_id} agent omits identity") from error
        if player_id not in (0, 1):
            raise ValueError(f"game {game_id} has invalid player index")
        if agent_id in agent_ids or player_id in indexes:
            raise ValueError(f"game {game_id} repeats an agent identity")
        if agent_id not in metadata_by_agent:
            raise ValueError(f"game {game_id} detail disagrees with battle agents")
        result_position = metadata_by_agent[agent_id].get("result_position")
        if result_position is not None:
            try:
                result_position = int(result_position)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"game {game_id} battle result position is invalid"
                ) from error
            if result_position != int(ranks[player_id]):
                raise ValueError(
                    f"game {game_id} battle result disagrees with replay ranks"
                )
        agent_ids.add(agent_id)
        indexes.add(player_id)
        codingamer = raw.get("codingamer") or {}
        frozen = leaderboard_by_agent.get(agent_id)
        if agent_id in own_agent_ids:
            label_role = "self-relabel-only"
        elif frozen is not None and frozen.get("strength_tier") is not None:
            label_role = "direct-public-expert"
        else:
            label_role = "unlabelled-opponent"
        agents.append(
            {
                "agent_id": agent_id,
                "label_role": label_role,
                "name": str(codingamer.get("pseudo") or raw.get("name") or ""),
                "player_id": player_id,
                "rank": None if frozen is None else frozen["rank"],
                "score": None if frozen is None else frozen["score"],
                "session_id": metadata_by_agent[agent_id].get("session_id"),
                "strength_tier": (
                    None if frozen is None else frozen.get("strength_tier")
                ),
                "submission_id": metadata_by_agent[agent_id].get("submission_id"),
                "user_id": metadata_by_agent[agent_id].get("user_id"),
            }
        )
    if indexes != {0, 1}:
        raise ValueError(f"game {game_id} does not contain both player indexes")

    turns = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError(f"game {game_id} contains a non-object frame")
        frame_agent = frame.get("agentId", -1)
        if isinstance(frame_agent, bool):
            continue
        try:
            player_id = int(frame_agent)
        except (TypeError, ValueError):
            continue
        if player_id < 0:
            continue
        action = str(frame.get("stdout") or "").strip()
        if player_id not in (0, 1) or not action or any(
            character not in "01234567" for character in action
        ):
            raise ValueError(f"game {game_id} contains an invalid action frame")
        turns.append({"action": action, "player_id": player_id})
    if not turns:
        raise ValueError(f"game {game_id} has no played turns")

    winner = ranks.index(0)
    trainer = trainer_module()
    game = trainer.Game(
        key=f"codingame-live:{game_id}",
        game_id=game_id,
        source="codingame-live",
        focus_agent_id=-1,
        focus_player=None,
        winner=winner,
        turns=tuple((turn["player_id"], turn["action"]) for turn in turns),
        policy_start_turn=len(turns),
    )
    trainer.replay_game(game)
    if not any(agent["label_role"] == "direct-public-expert" for agent in agents):
        raise ValueError(f"game {game_id} has no eligible strong-player labels")
    return {
        "agents": sorted(agents, key=lambda item: item["player_id"]),
        "game_id": game_id,
        "transcript": "/".join(turn["action"] for turn in turns),
        "turns": turns,
        "winner_player_id": winner,
    }


class LiveReplayCollector:
    def __init__(
        self,
        *,
        repository: pathlib.Path = REPOSITORY,
        data_root: pathlib.Path = DEFAULT_DATA_ROOT,
        api: PublicApi | Any | None = None,
        clock: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        evidence_paths: Iterable[pathlib.Path] | None = None,
        exclusion_registry: ExclusionRegistry | None = None,
        own_agent_ids: Iterable[int] = (),
        maximum_workers: int = 2,
    ) -> None:
        if not 1 <= maximum_workers <= 4:
            raise ValueError("detail concurrency must remain between one and four")
        self.repository = repository.resolve()
        self.data_root = data_root.resolve()
        self.api = api or PublicApi(sleep=sleep)
        self.clock = clock
        self.sleep = sleep
        self.evidence_paths = evidence_paths
        self.maximum_workers = maximum_workers
        self.own_agent_ids = {int(value) for value in own_agent_ids}
        self.collector_sha256 = digest_file(pathlib.Path(__file__).resolve())
        self.registry = exclusion_registry or build_exclusion_registry(
            self.repository,
            self.data_root,
            evidence_paths=self.evidence_paths,
        )
        self.registry_sha256, self.registry_path = write_content_addressed(
            self.data_root / "exclusions", self.registry.payload
        )

    def _validate_run_id(self, run_id: str) -> None:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "run id must contain only letters, digits, dot, underscore, or dash"
            )

    def _store_response(
        self,
        *,
        run_id: str,
        poll_index: int,
        request_schema: str,
        service: str,
        payload: Any,
        response: ApiResponse,
        fetched_at: str,
    ) -> StoredResponse:
        key = request_key(request_schema, service, payload)
        raw_hash = digest_bytes(response.body)
        try:
            decoded = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{request_schema} response is not valid JSON") from error
        normalized_content = canonical_json_bytes(decoded)
        normalized_hash = digest_bytes(normalized_content)
        raw_path = self.data_root / "raw" / request_schema / key / f"{raw_hash}.json"
        normalized_path = (
            self.data_root
            / "normalized"
            / request_schema
            / key
            / f"{normalized_hash}.json"
        )
        write_once(raw_path, response.body)
        write_once(normalized_path, normalized_content)
        receipt = {
            "attempts": response.attempts,
            "collector_sha256": self.collector_sha256,
            "fetched_at_utc": fetched_at,
            "headers": dict(sorted(response.headers.items())),
            "normalized_path": relative_to_repository(normalized_path, self.repository),
            "normalized_sha256": normalized_hash,
            "poll_index": poll_index,
            "raw_path": relative_to_repository(raw_path, self.repository),
            "raw_sha256": raw_hash,
            "request": request_record(request_schema, service, payload),
            "request_key": key,
            "run_id": run_id,
            "status": response.status,
        }
        write_content_addressed(
            self.data_root / "receipts" / request_schema / key, receipt
        )
        return StoredResponse(
            request_key=key,
            request_schema=request_schema,
            service=service,
            payload=payload,
            fetched_at_utc=fetched_at,
            raw_sha256=raw_hash,
            normalized_sha256=normalized_hash,
            normalized=decoded,
            status=response.status,
            attempts=response.attempts,
            cached=False,
        )

    def _fetch(
        self,
        *,
        run_id: str,
        poll_index: int,
        request_schema: str,
        payload: Any,
    ) -> StoredResponse:
        service = REQUEST_SCHEMAS[request_schema]["service"]
        fetched_at = self.clock()
        response = self.api.post(service, payload)
        return self._store_response(
            run_id=run_id,
            poll_index=poll_index,
            request_schema=request_schema,
            service=service,
            payload=payload,
            response=response,
            fetched_at=fetched_at,
        )

    def _cached_detail_versions(self, game_id: int) -> list[StoredResponse]:
        schema = "game-detail-v1"
        service = REQUEST_SCHEMAS[schema]["service"]
        payload = [game_id, None]
        key = request_key(schema, service, payload)
        receipts = sorted((self.data_root / "receipts" / schema / key).glob("*.json"))
        by_normalized: dict[str, StoredResponse] = {}
        for receipt_path in receipts:
            receipt = json.loads(receipt_path.read_text())
            normalized_hash = str(receipt["normalized_sha256"])
            if normalized_hash in by_normalized:
                continue
            normalized_path = (
                self.data_root
                / "normalized"
                / schema
                / key
                / f"{normalized_hash}.json"
            )
            decoded = json.loads(normalized_path.read_text())
            by_normalized[normalized_hash] = StoredResponse(
                request_key=key,
                request_schema=schema,
                service=service,
                payload=payload,
                fetched_at_utc=str(receipt["fetched_at_utc"]),
                raw_sha256=str(receipt["raw_sha256"]),
                normalized_sha256=normalized_hash,
                normalized=decoded,
                status=int(receipt["status"]),
                attempts=int(receipt["attempts"]),
                cached=True,
            )
        return [by_normalized[key] for key in sorted(by_normalized)]

    def _write_event(
        self,
        run_id: str,
        poll_index: int,
        kind: str,
        logical_id: str,
        payload: dict[str, Any],
    ) -> pathlib.Path:
        content = canonical_json_bytes(payload)
        digest = digest_bytes(content)
        safe_logical = re.sub(r"[^A-Za-z0-9._-]", "_", logical_id)
        path = (
            self.data_root
            / "events"
            / run_id
            / f"poll-{poll_index:04d}"
            / kind
            / f"{safe_logical}-{digest}.json"
        )
        write_once(path, content)
        return path

    def _existing_replay_hashes(self, game_id: int) -> list[str]:
        return sorted(
            path.stem
            for path in (self.data_root / "replay_payloads" / str(game_id)).glob(
                "*.json"
            )
        )

    def _conflict_hashes(self, game_id: int) -> list[str]:
        versions = self._cached_detail_versions(game_id)
        return sorted({version.normalized_sha256 for version in versions})

    def _record_conflict(
        self,
        run_id: str,
        poll_index: int,
        game_id: int,
        hashes: list[str],
        network_requested: bool,
    ) -> dict[str, Any]:
        conflict = {
            "collector_sha256": self.collector_sha256,
            "detected_at_utc": self.clock(),
            "game_id": game_id,
            "normalized_payload_sha256": hashes,
            "reason": "conflicting normalized replay payloads; all versions quarantined",
            "schema": "papersoccer.live-replay-conflict.v1",
        }
        write_content_addressed(
            self.data_root / "conflicts" / str(game_id), conflict
        )
        self._write_event(
            run_id, poll_index, "conflict", str(game_id), conflict
        )
        return {
            "game_id": game_id,
            "network_requested": network_requested,
            "normalized_payload_sha256": hashes,
            "reason": conflict["reason"],
            "status": "conflict",
        }

    def _process_detail(
        self,
        *,
        run_id: str,
        poll_index: int,
        game_id: int,
        metadata: dict[str, Any],
        leaderboard_by_agent: dict[int, dict[str, Any]],
        refresh_details: bool,
    ) -> dict[str, Any]:
        versions = self._cached_detail_versions(game_id)
        response: StoredResponse | None = None
        fetched_new_response = False
        if refresh_details or not versions:
            try:
                response = self._fetch(
                    run_id=run_id,
                    poll_index=poll_index,
                    request_schema="game-detail-v1",
                    payload=[game_id, None],
                )
                fetched_new_response = True
            except Exception as error:  # network errors become auditable rejections
                rejection = {
                    "collector_sha256": self.collector_sha256,
                    "game_id": game_id,
                    "network_requested": True,
                    "reason": f"detail request failed: {type(error).__name__}: {error}",
                    "rejected_at_utc": self.clock(),
                    "status": "request-error",
                }
                self._write_event(
                    run_id, poll_index, "rejection", str(game_id), rejection
                )
                return rejection
            versions = self._cached_detail_versions(game_id)
        elif versions:
            response = versions[0]
        hashes = sorted({version.normalized_sha256 for version in versions})
        if len(hashes) > 1:
            return self._record_conflict(
                run_id,
                poll_index,
                game_id,
                hashes,
                network_requested=fetched_new_response,
            )
        if response is None:
            response = versions[0]
        try:
            replay = validate_replay_detail(
                response.normalized,
                game_id=game_id,
                metadata=metadata,
                leaderboard_by_agent=leaderboard_by_agent,
                own_agent_ids=self.own_agent_ids,
            )
        except (KeyError, TypeError, ValueError) as error:
            rejection = {
                "cached": response.cached,
                "collector_sha256": self.collector_sha256,
                "game_id": game_id,
                "network_requested": fetched_new_response,
                "normalized_payload_sha256": response.normalized_sha256,
                "raw_sha256": response.raw_sha256,
                "reason": str(error),
                "rejected_at_utc": self.clock(),
                "status": "structural-rejection",
            }
            self._write_event(
                run_id, poll_index, "rejection", str(game_id), rejection
            )
            return rejection

        existing = self._existing_replay_hashes(game_id)
        is_globally_new = not existing
        replay_content = canonical_json_bytes(response.normalized)
        replay_path = (
            self.data_root
            / "replay_payloads"
            / str(game_id)
            / f"{response.normalized_sha256}.json"
        )
        if is_globally_new and fetched_new_response:
            discovery = {
                "collector_sha256": self.collector_sha256,
                "discovered_at_utc": self.clock(),
                "game_id": game_id,
                "normalized_payload_sha256": response.normalized_sha256,
                "run_id": run_id,
                "schema": "papersoccer.live-replay-discovery.v1",
            }
            # Discovery is intentionally written before the payload.  A crash
            # between these writes remains restartable from the response cache.
            discovery_path = self.data_root / "discoveries" / run_id / f"{game_id}.json"
            write_once(discovery_path, canonical_json_bytes(discovery))
        write_once(replay_path, replay_content)
        record = {
            "acquisition": {
                "collector_sha256": self.collector_sha256,
                "fetched_at_utc": response.fetched_at_utc,
                "normalized_payload_sha256": response.normalized_sha256,
                "raw_sha256": response.raw_sha256,
                "request": request_record(
                    "game-detail-v1",
                    REQUEST_SCHEMAS["game-detail-v1"]["service"],
                    [game_id, None],
                ),
            },
            "battle_metadata": metadata,
            "replay": replay,
            "schema": REPLAY_SCHEMA,
        }
        record_hash, record_path = write_content_addressed(
            self.data_root / "games" / str(game_id), record
        )
        accepted = {
            "cached": response.cached,
            "game_id": game_id,
            "network_requested": fetched_new_response,
            "normalized_payload_sha256": response.normalized_sha256,
            "raw_sha256": response.raw_sha256,
            "record_path": relative_to_repository(record_path, self.repository),
            "record_sha256": record_hash,
            "status": "accepted",
        }
        self._write_event(run_id, poll_index, "acceptance", str(game_id), accepted)
        return accepted

    def _cursor_before(self, run_id: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for path in sorted((self.data_root / "polls" / run_id).glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("schema") != POLL_SCHEMA:
                continue
            for agent, game_id in payload.get("cursor_after", {}).items():
                result[agent] = max(result.get(agent, -1), int(game_id))
        return dict(sorted(result.items(), key=lambda item: int(item[0])))

    def _next_poll_index(self, run_id: str) -> int:
        indexes = []
        for path in (self.data_root / "polls" / run_id).glob("poll-*.json"):
            match = re.match(r"poll-(\d+)-", path.name)
            if match:
                indexes.append(int(match.group(1)))
        return 0 if not indexes else max(indexes) + 1

    def _metadata_candidate_count(
        self,
        observations: dict[int, list[dict[str, Any]]],
        leaderboard_by_agent: dict[int, dict[str, Any]],
    ) -> int:
        result = 0
        for game_id, values in observations.items():
            if game_id in self.registry.known_ids:
                continue
            identities = {battle_identity(value) for value in values}
            if len(identities) != 1 or not values[0]["done"]:
                continue
            if self._existing_replay_hashes(game_id):
                continue
            players = {player["agent_id"] for player in values[0]["players"]}
            eligible = any(
                agent in leaderboard_by_agent
                and agent not in self.own_agent_ids
                for agent in players
            )
            if eligible:
                result += 1
        return result

    def collect_poll(
        self,
        *,
        run_id: str,
        poll_index: int,
        minimum_new_games: int,
        initial_top: int = 5,
        expanded_top: int = 20,
        focus_agent_ids: Iterable[int] = (),
        refresh_details: bool = False,
    ) -> dict[str, Any]:
        self._validate_run_id(run_id)
        if not 1 <= initial_top <= expanded_top <= 100:
            raise ValueError("top-player bounds are invalid")
        started_at = self.clock()
        cursor_before = self._cursor_before(run_id)
        leaderboard_response = self._fetch(
            run_id=run_id,
            poll_index=poll_index,
            request_schema="leaderboard-v1",
            payload=[
                PUZZLE_PUBLIC_ID,
                None,
                "global",
                {"active": False, "column": "", "filter": ""},
            ],
        )
        entries = leaderboard_entries(
            leaderboard_response.normalized, expanded_top
        )
        leaderboard_by_agent = {entry["agent_id"]: entry for entry in entries}
        observations: dict[int, list[dict[str, Any]]] = {}
        metadata_snapshots = []
        cursor_after = dict(cursor_before)

        def fetch_agent(entry: dict[str, Any] | None, agent_id: int) -> None:
            response = self._fetch(
                run_id=run_id,
                poll_index=poll_index,
                request_schema="agent-battles-v1",
                payload=[agent_id, None],
            )
            if not isinstance(response.normalized, list):
                raise ValueError(f"battle window for agent {agent_id} is not a list")
            accepted_metadata = 0
            metadata_errors = []
            maximum_game = cursor_after.get(str(agent_id), -1)
            for raw in response.normalized:
                try:
                    battle = normalize_battle(raw, agent_id)
                except ValueError as error:
                    metadata_errors.append(str(error))
                    continue
                observations.setdefault(battle["game_id"], []).append(battle)
                accepted_metadata += 1
                maximum_game = max(maximum_game, battle["game_id"])
            if maximum_game >= 0:
                cursor_after[str(agent_id)] = maximum_game
            metadata_snapshots.append(
                {
                    "agent_id": agent_id,
                    "battle_count": len(response.normalized),
                    "metadata_errors": sorted(metadata_errors),
                    "normalized_sha256": response.normalized_sha256,
                    "rank": None if entry is None else entry["rank"],
                    "raw_sha256": response.raw_sha256,
                    "score": None if entry is None else entry["score"],
                    "session_id": None if entry is None else entry["session_id"],
                    "valid_battle_count": accepted_metadata,
                }
            )

        fetched_agents: set[int] = set()
        for entry in entries[:initial_top]:
            fetch_agent(entry, entry["agent_id"])
            fetched_agents.add(entry["agent_id"])
        for focus_agent_id in sorted({int(value) for value in focus_agent_ids}):
            if focus_agent_id not in fetched_agents:
                fetch_agent(leaderboard_by_agent.get(focus_agent_id), focus_agent_id)
                fetched_agents.add(focus_agent_id)
        prospective_top_five = self._metadata_candidate_count(
            observations, leaderboard_by_agent
        )
        expanded = prospective_top_five < minimum_new_games
        if expanded:
            for entry in entries[initial_top:expanded_top]:
                if entry["agent_id"] in fetched_agents:
                    continue
                fetch_agent(entry, entry["agent_id"])
                fetched_agents.add(entry["agent_id"])

        decisions = []
        candidates = []
        for game_id in sorted(observations):
            values = observations[game_id]
            identities = {battle_identity(value) for value in values}
            if len(identities) != 1:
                decisions.append(
                    {
                        "game_id": game_id,
                        "reason": "conflicting battle metadata across agent windows",
                        "status": "metadata-conflict",
                    }
                )
                continue
            battle = values[0]
            observer_ids = sorted(
                {
                    player["agent_id"]
                    for value in values
                    for player in value["players"]
                    if player["agent_id"] in fetched_agents
                }
            )
            metadata = {
                "observed_agent_ids": observer_ids,
                "observation_count": len(values),
                "players": battle["players"],
            }
            if not battle["done"]:
                decisions.append(
                    {
                        "game_id": game_id,
                        "reason": "battle is not complete",
                        "status": "ineligible",
                    }
                )
                continue
            registry_record = self.registry.records.get(game_id)
            if registry_record is not None:
                decisions.append(
                    {
                        "categories": registry_record["categories"],
                        "game_id": game_id,
                        "reason": (
                            "protected data boundary"
                            if self.registry.is_protected(game_id)
                            else "game already exists in local evidence"
                        ),
                        "status": (
                            "excluded-protected"
                            if self.registry.is_protected(game_id)
                            else "already-archived"
                        ),
                    }
                )
                continue
            players = {player["agent_id"] for player in battle["players"]}
            eligible_agents = sorted(
                agent
                for agent in players
                if agent in leaderboard_by_agent
                and agent not in self.own_agent_ids
            )
            if not eligible_agents:
                decisions.append(
                    {
                        "game_id": game_id,
                        "reason": "no current top-player action is eligible for direct labels",
                        "status": "ineligible",
                    }
                )
                continue
            existing_hashes = self._existing_replay_hashes(game_id)
            if existing_hashes and not refresh_details:
                decisions.append(
                    {
                        "game_id": game_id,
                        "normalized_payload_sha256": existing_hashes,
                        "reason": "game already exists in live replay archive",
                        "status": "already-collected",
                    }
                )
                continue
            candidates.append((game_id, metadata))

        detail_results = []
        if candidates:
            trainer_module()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.maximum_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._process_detail,
                        run_id=run_id,
                        poll_index=poll_index,
                        game_id=game_id,
                        metadata=metadata,
                        leaderboard_by_agent=leaderboard_by_agent,
                        refresh_details=refresh_details,
                    ): game_id
                    for game_id, metadata in candidates
                }
                for future in concurrent.futures.as_completed(futures):
                    detail_results.append(future.result())
        detail_results.sort(key=lambda item: item["game_id"])
        decisions.extend(detail_results)
        decisions.sort(key=lambda item: (item["game_id"], item["status"]))
        new_ids = self._valid_discoveries(run_id)
        poll = {
            "collector_sha256": self.collector_sha256,
            "completed_at_utc": self.clock(),
            "cursor_after": dict(
                sorted(cursor_after.items(), key=lambda item: int(item[0]))
            ),
            "cursor_before": cursor_before,
            "decisions": decisions,
            "detail_request_count": sum(
                item.get("network_requested", False) for item in detail_results
            ),
            "detail_validation_count": len(detail_results),
            "expanded_to_top": expanded_top if expanded else initial_top,
            "exclusion_registry_path": relative_to_repository(
                self.registry_path, self.repository
            ),
            "exclusion_registry_sha256": self.registry_sha256,
            "leaderboard": {
                "entries": entries,
                "normalized_sha256": leaderboard_response.normalized_sha256,
                "raw_sha256": leaderboard_response.raw_sha256,
            },
            "metadata_snapshots": sorted(
                metadata_snapshots, key=lambda item: item["agent_id"]
            ),
            "new_valid_game_ids_for_run": new_ids,
            "new_valid_games_for_run": len(new_ids),
            "poll_index": poll_index,
            "prospective_new_from_initial_top": prospective_top_five,
            "request_schemas": REQUEST_SCHEMAS,
            "run_id": run_id,
            "schema": POLL_SCHEMA,
            "started_at_utc": started_at,
        }
        content = canonical_json_bytes(poll)
        poll_hash = digest_bytes(content)
        poll_path = (
            self.data_root
            / "polls"
            / run_id
            / f"poll-{poll_index:04d}-{poll_hash}.json"
        )
        write_once(poll_path, content)
        return poll

    def _conflicted_game_ids(self) -> set[int]:
        result = set()
        root = self.data_root / "conflicts"
        if root.exists():
            for path in root.iterdir():
                if path.is_dir() and path.name.isdigit() and any(path.glob("*.json")):
                    result.add(int(path.name))
        return result

    def _valid_discoveries(self, run_id: str) -> list[int]:
        conflicts = self._conflicted_game_ids()
        result = []
        for path in sorted((self.data_root / "discoveries" / run_id).glob("*.json")):
            payload = json.loads(path.read_text())
            game_id = int(payload["game_id"])
            replay_path = (
                self.data_root
                / "replay_payloads"
                / str(game_id)
                / f"{payload['normalized_payload_sha256']}.json"
            )
            if game_id not in conflicts and replay_path.is_file():
                result.append(game_id)
        return sorted(set(result))

    def run(
        self,
        *,
        run_id: str,
        polls: int,
        poll_interval_seconds: float,
        minimum_new_games: int,
        initial_top: int = 5,
        expanded_top: int = 20,
        focus_agent_ids: Iterable[int] = (),
        refresh_details: bool = False,
    ) -> dict[str, Any]:
        self._validate_run_id(run_id)
        if polls < 1:
            raise ValueError("at least one poll is required")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval cannot be negative")
        started_at = self.clock()
        poll_summaries = []
        index = self._next_poll_index(run_id)
        for offset in range(polls):
            poll = self.collect_poll(
                run_id=run_id,
                poll_index=index + offset,
                minimum_new_games=minimum_new_games,
                initial_top=initial_top,
                expanded_top=expanded_top,
                focus_agent_ids=focus_agent_ids,
                refresh_details=refresh_details,
            )
            poll_summaries.append(
                {
                    "expanded_to_top": poll["expanded_to_top"],
                    "new_valid_games_for_run": poll["new_valid_games_for_run"],
                    "poll_index": poll["poll_index"],
                }
            )
            if offset + 1 < polls:
                self.sleep(poll_interval_seconds)
        new_ids = self._valid_discoveries(run_id)
        result = {
            "collector_sha256": self.collector_sha256,
            "completed_at_utc": self.clock(),
            "decision": (
                "enough-data" if len(new_ids) >= minimum_new_games else "waiting-for-data"
            ),
            "exclusion_registry_path": relative_to_repository(
                self.registry_path, self.repository
            ),
            "exclusion_registry_sha256": self.registry_sha256,
            "minimum_independent_new_games": minimum_new_games,
            "new_valid_game_ids": new_ids,
            "new_valid_games": len(new_ids),
            "own_agent_ids": sorted(self.own_agent_ids),
            "polls": poll_summaries,
            "request_schemas": REQUEST_SCHEMAS,
            "run_id": run_id,
            "schema": RUN_SCHEMA,
            "started_at_utc": started_at,
        }
        write_content_addressed(self.data_root / "runs" / run_id, result)
        return result


def check_store(data_root: pathlib.Path) -> dict[str, int]:
    counts = {
        "accepted_game_records": 0,
        "conflicts": 0,
        "discoveries": 0,
        "games": 0,
        "normalized_responses": 0,
        "raw_responses": 0,
        "receipts": 0,
    }
    for path in sorted((data_root / "exclusions").glob("*.json")):
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != EXCLUSION_SCHEMA
            or canonical_json_bytes(payload) != path.read_bytes()
            or digest_file(path) != path.stem
        ):
            raise ValueError(f"exclusion registry is not canonical: {path}")
    for path in sorted((data_root / "raw").rglob("*.json")):
        if digest_file(path) != path.stem:
            raise ValueError(f"raw response hash mismatch: {path}")
        json.loads(path.read_bytes())
        counts["raw_responses"] += 1
    for path in sorted((data_root / "normalized").rglob("*.json")):
        payload = json.loads(path.read_text())
        if canonical_json_bytes(payload) != path.read_bytes() or digest_file(path) != path.stem:
            raise ValueError(f"normalized response is not canonical: {path}")
        counts["normalized_responses"] += 1
    for path in sorted((data_root / "receipts").glob("*/*/*.json")):
        receipt = json.loads(path.read_text())
        if canonical_json_bytes(receipt) != path.read_bytes() or digest_file(path) != path.stem:
            raise ValueError(f"response receipt is not canonical: {path}")
        request = receipt.get("request") or {}
        schema = request.get("request_schema")
        if schema not in REQUEST_SCHEMAS:
            raise ValueError(f"response receipt has an unknown schema: {path}")
        expected_key = request_key(schema, request.get("service"), request.get("body"))
        if receipt.get("request_key") != expected_key or path.parent.name != expected_key:
            raise ValueError(f"response receipt request key mismatch: {path}")
        raw_path = (
            data_root
            / "raw"
            / schema
            / expected_key
            / f"{receipt['raw_sha256']}.json"
        )
        normalized_path = (
            data_root
            / "normalized"
            / schema
            / expected_key
            / f"{receipt['normalized_sha256']}.json"
        )
        if not raw_path.is_file() or digest_file(raw_path) != receipt["raw_sha256"]:
            raise ValueError(f"response receipt raw payload is missing: {path}")
        if (
            not normalized_path.is_file()
            or digest_file(normalized_path) != receipt["normalized_sha256"]
        ):
            raise ValueError(f"response receipt normalized payload is missing: {path}")
        counts["receipts"] += 1
    for path in sorted((data_root / "replay_payloads").glob("*/*.json")):
        payload = json.loads(path.read_text())
        if canonical_json_bytes(payload) != path.read_bytes() or digest_file(path) != path.stem:
            raise ValueError(f"replay payload is not canonical: {path}")
        if int(payload.get("gameId", -1)) != int(path.parent.name):
            raise ValueError(f"replay payload game id disagrees with path: {path}")
        counts["games"] += 1
    accepted_ids = set()
    trainer = None
    for path in sorted((data_root / "games").glob("*/*.json")):
        record = json.loads(path.read_text())
        if (
            record.get("schema") != REPLAY_SCHEMA
            or canonical_json_bytes(record) != path.read_bytes()
            or digest_file(path) != path.stem
        ):
            raise ValueError(f"accepted game record is not canonical: {path}")
        replay = record.get("replay") or {}
        game_id = int(replay.get("game_id", -1))
        if game_id != int(path.parent.name):
            raise ValueError(f"accepted game record id disagrees with path: {path}")
        payload_hash = record["acquisition"]["normalized_payload_sha256"]
        payload_path = data_root / "replay_payloads" / str(game_id) / f"{payload_hash}.json"
        if not payload_path.is_file():
            raise ValueError(f"accepted game omits its replay payload: {path}")
        roles = {agent.get("label_role") for agent in replay.get("agents", [])}
        if "direct-public-expert" not in roles or not roles.issubset(
            {"direct-public-expert", "self-relabel-only", "unlabelled-opponent"}
        ):
            raise ValueError(f"accepted game has invalid label roles: {path}")
        if trainer is None:
            trainer = trainer_module()
        turns = tuple(
            (int(turn["player_id"]), str(turn["action"]))
            for turn in replay["turns"]
        )
        trainer.replay_game(
            trainer.Game(
                key=f"store-check:{game_id}",
                game_id=game_id,
                source="codingame-live",
                focus_agent_id=-1,
                focus_player=None,
                winner=int(replay["winner_player_id"]),
                turns=turns,
                policy_start_turn=len(turns),
            )
        )
        accepted_ids.add(game_id)
        counts["accepted_game_records"] += 1
    for path in sorted((data_root / "discoveries").glob("*/*.json")):
        payload = json.loads(path.read_text())
        game_id = int(payload["game_id"])
        replay_path = (
            data_root
            / "replay_payloads"
            / str(game_id)
            / f"{payload['normalized_payload_sha256']}.json"
        )
        if (
            canonical_json_bytes(payload) != path.read_bytes()
            or int(path.stem) != game_id
            or game_id not in accepted_ids
            or not replay_path.is_file()
        ):
            raise ValueError(f"discovery filename disagrees with payload: {path}")
        counts["discoveries"] += 1
    for directory in sorted((data_root / "conflicts").glob("*")):
        if directory.is_dir() and any(directory.glob("*.json")):
            counts["conflicts"] += 1
    return counts


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--polls", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-new-games", type=int, default=50)
    parser.add_argument("--initial-top", type=int, default=5)
    parser.add_argument("--expanded-top", type=int, default=20)
    parser.add_argument("--focus-agent-id", type=int, action="append", default=[])
    parser.add_argument("--own-agent-id", type=int, action="append", default=[])
    parser.add_argument("--maximum-workers", type=int, default=2)
    parser.add_argument("--refresh-details", action="store_true")
    parser.add_argument("--build-exclusions-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.minimum_new_games < 1:
        parser.error("--minimum-new-games must be positive")
    if arguments.check:
        report = check_store(arguments.data_root.resolve())
        print(json.dumps(report, sort_keys=True))
        return
    collector = LiveReplayCollector(
        data_root=arguments.data_root,
        own_agent_ids=arguments.own_agent_id,
        maximum_workers=arguments.maximum_workers,
    )
    if arguments.build_exclusions_only:
        print(
            json.dumps(
                {
                    "known_game_ids": len(collector.registry.known_ids),
                    "path": relative_to_repository(
                        collector.registry_path, collector.repository
                    ),
                    "protected_game_ids": sum(
                        collector.registry.is_protected(game_id)
                        for game_id in collector.registry.known_ids
                    ),
                    "sha256": collector.registry_sha256,
                },
                sort_keys=True,
            )
        )
        return
    result = collector.run(
        run_id=arguments.run_id,
        polls=arguments.polls,
        poll_interval_seconds=arguments.poll_interval_seconds,
        minimum_new_games=arguments.minimum_new_games,
        initial_top=arguments.initial_top,
        expanded_top=arguments.expanded_top,
        focus_agent_ids=arguments.focus_agent_id,
        refresh_details=arguments.refresh_details,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
