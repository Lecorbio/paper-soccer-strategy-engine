#!/usr/bin/env python3

"""Collect a provenance-bound CodinGame Paper Soccer arena batch.

This is a diagnostic archive, not a training-corpus builder.  It snapshots the
entire battle window for one submitted agent, preserves public API responses in
an append-only content-addressed store, validates every available transcript,
and classifies operational endings from frames and program output.  The
``agents[].valid`` field is deliberately ignored because it is not a reliable
operational signal.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_DATA_ROOT = REPOSITORY / "results" / "codingame_arena_diagnostics"
LIVE_COLLECTOR_PATH = (
    REPOSITORY
    / "submissions"
    / "codingame"
    / "bots"
    / "neural_puct"
    / "collect_live_replays.py"
)

SPEC = importlib.util.spec_from_file_location(
    "papersoccer_live_replay_shared", LIVE_COLLECTOR_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load shared replay collector: {LIVE_COLLECTOR_PATH}")
shared = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, shared)
SPEC.loader.exec_module(shared)


PUZZLE_PUBLIC_ID = "paper-soccer"
ARENA_BATCH_SCHEMA = "papersoccer.codingame-arena-batch.v1"
ARENA_GAME_SCHEMA = "papersoccer.codingame-arena-game.v1"
ARENA_BINDING_SCHEMA = "papersoccer.codingame-arena-binding.v1"
ARENA_RECEIPT_SCHEMA = "papersoccer.codingame-arena-response-receipt.v1"
AUDITOR_TSV_HEADER = "game_id\tcandidate_player\twinner\tturns"
PURPOSE = {
    "diagnostic_only": True,
    "training_eligible": False,
    "note": (
        "arena observations may influence bot development and must not be "
        "treated as untouched evaluation or direct expert training labels"
    ),
}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

FAILURE_PATTERNS = (
    (
        "timeout",
        re.compile(
            r"\btime[ -]?out\b|\btimed out\b|did not respond in time|"
            r"exceeded (?:the )?time limit|too long to respond",
            re.IGNORECASE,
        ),
    ),
    (
        "empty-output",
        re.compile(
            r"provided no output|produced no output|empty output|no output was "
            r"provided|did not provide (?:an? )?output",
            re.IGNORECASE,
        ),
    ),
    (
        "invalid-output",
        re.compile(
            r"invalid output|malformed output|unrecognized output|could not "
            r"parse (?:the )?output",
            re.IGNORECASE,
        ),
    ),
    (
        "illegal-action",
        re.compile(
            r"illegal (?:move|action|edge)|invalid (?:move|action|edge)|"
            r"move is not legal|action is not legal",
            re.IGNORECASE,
        ),
    ),
    (
        "runtime-error",
        re.compile(
            r"runtime error|segmentation fault|uncaught exception|"
            r"terminated by signal|failed to execute|process (?:was )?killed|"
            r"out of memory",
            re.IGNORECASE,
        ),
    ),
)
FAILURE_PRECEDENCE = {
    "ok": 0,
    "empty-output": 1,
    "invalid-output": 2,
    "illegal-action": 3,
    "runtime-error": 4,
    "timeout": 5,
}
TEXT_FRAME_FIELDS = (
    "stderr",
    "gameInformation",
    "summary",
    "tooltip",
    "tooltips",
)


@dataclasses.dataclass(frozen=True)
class SourceBinding:
    agent_id: int
    submission_id: int
    source_path: pathlib.Path
    source_sha256: str
    source_bytes: int
    source_characters: int
    source_encoding: str
    archived_path: pathlib.Path
    repository_commit: str


class ArenaBatchCollector:
    """Append-only collector for one source-identifiable arena batch."""

    def __init__(
        self,
        *,
        repository: pathlib.Path = REPOSITORY,
        data_root: pathlib.Path = DEFAULT_DATA_ROOT,
        api: Any | None = None,
        clock: Callable[[], str] = shared.utc_now,
        exclusion_registry: Any | None = None,
        exclusion_registry_path: pathlib.Path | None = None,
        exclusion_registry_sha256: str | None = None,
        maximum_workers: int = 2,
    ) -> None:
        if not 1 <= maximum_workers <= 4:
            raise ValueError("detail concurrency must remain between one and four")
        self.repository = repository.resolve()
        self.data_root = data_root.resolve()
        self.api = api or shared.PublicApi()
        self.clock = clock
        self.maximum_workers = maximum_workers
        self.collector_sha256 = shared.digest_file(pathlib.Path(__file__).resolve())
        if exclusion_registry is not None and (
            exclusion_registry_path is not None
            or exclusion_registry_sha256 is not None
        ):
            raise ValueError(
                "provide either an exclusion registry object or path, not both"
            )
        if exclusion_registry is None and exclusion_registry_path is None:
            raise ValueError(
                "a pre-built frozen ID-only exclusion registry is required; "
                "the arena collector never scans protected evidence"
            )
        self.registry = exclusion_registry or load_exclusion_registry(
            exclusion_registry_path, exclusion_registry_sha256
        )
        self.registry_sha256, self.registry_path = shared.write_content_addressed(
            self.data_root / "exclusions", self.registry.payload
        )
        self._fetch_lock = threading.Lock()

    def bind_source(
        self,
        *,
        agent_id: int,
        submission_id: int,
        source_path: pathlib.Path,
        expected_source_sha256: str | None = None,
        repository_commit: str | None = None,
    ) -> SourceBinding:
        source_path = source_path.resolve()
        content = source_path.read_bytes()
        source_hash = shared.digest_bytes(content)
        if expected_source_sha256 is not None:
            expected = expected_source_sha256.lower()
            if SHA256_PATTERN.fullmatch(expected) is None:
                raise ValueError("expected source SHA-256 must be 64 lowercase hex digits")
            if expected != source_hash:
                raise ValueError(
                    f"source SHA-256 mismatch: expected {expected}, got {source_hash}"
                )
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("submitted source must be valid UTF-8") from error
        commit = repository_commit or repository_head(self.repository)
        archive_path = self.data_root / "source_payloads" / f"{source_hash}.source"
        shared.write_once(archive_path, content)
        return SourceBinding(
            agent_id=int(agent_id),
            submission_id=int(submission_id),
            source_path=source_path,
            source_sha256=source_hash,
            source_bytes=len(content),
            source_characters=len(decoded),
            source_encoding="utf-8",
            archived_path=archive_path,
            repository_commit=commit,
        )

    def _store_response(
        self,
        *,
        run_id: str,
        request_schema: str,
        payload: Any,
        response: Any,
        fetched_at: str,
    ) -> Any:
        service = shared.REQUEST_SCHEMAS[request_schema]["service"]
        key = shared.request_key(request_schema, service, payload)
        raw_hash = shared.digest_bytes(response.body)
        try:
            decoded = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{request_schema} response is not valid JSON") from error
        normalized_content = shared.canonical_json_bytes(decoded)
        normalized_hash = shared.digest_bytes(normalized_content)
        raw_path = self.data_root / "raw" / request_schema / key / f"{raw_hash}.json"
        normalized_path = (
            self.data_root
            / "normalized"
            / request_schema
            / key
            / f"{normalized_hash}.json"
        )
        shared.write_once(raw_path, response.body)
        shared.write_once(normalized_path, normalized_content)
        receipt = {
            "attempts": int(response.attempts),
            "collector_sha256": self.collector_sha256,
            "fetched_at_utc": fetched_at,
            "headers": dict(sorted(response.headers.items())),
            "normalized_path": shared.relative_to_repository(
                normalized_path, self.repository
            ),
            "normalized_sha256": normalized_hash,
            "purpose": PURPOSE,
            "raw_path": shared.relative_to_repository(raw_path, self.repository),
            "raw_sha256": raw_hash,
            "request": shared.request_record(request_schema, service, payload),
            "request_key": key,
            "run_id": run_id,
            "schema": ARENA_RECEIPT_SCHEMA,
            "status": int(response.status),
        }
        shared.write_content_addressed(
            self.data_root / "receipts" / request_schema / key, receipt
        )
        return shared.StoredResponse(
            request_key=key,
            request_schema=request_schema,
            service=service,
            payload=payload,
            fetched_at_utc=fetched_at,
            raw_sha256=raw_hash,
            normalized_sha256=normalized_hash,
            normalized=decoded,
            status=int(response.status),
            attempts=int(response.attempts),
            cached=False,
        )

    def _fetch(self, *, run_id: str, request_schema: str, payload: Any) -> Any:
        service = shared.REQUEST_SCHEMAS[request_schema]["service"]
        fetched_at = self.clock()
        response = self.api.post(service, payload)
        # Multiple detail workers may create directories at the same time.
        # write_once is atomic, while this lock keeps receipt timestamps and
        # fake-clock tests deterministic.
        with self._fetch_lock:
            return self._store_response(
                run_id=run_id,
                request_schema=request_schema,
                payload=payload,
                response=response,
                fetched_at=fetched_at,
            )

    def _cached_versions(self, request_schema: str, payload: Any) -> list[Any]:
        service = shared.REQUEST_SCHEMAS[request_schema]["service"]
        key = shared.request_key(request_schema, service, payload)
        receipts = sorted(
            (self.data_root / "receipts" / request_schema / key).glob("*.json")
        )
        by_hash: dict[str, Any] = {}
        for receipt_path in receipts:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            if (
                shared.digest_bytes(receipt_bytes) != receipt_path.stem
                or shared.canonical_json_bytes(receipt) != receipt_bytes
                or receipt.get("schema") != ARENA_RECEIPT_SCHEMA
                or receipt.get("request_key") != key
                or receipt.get("request", {}).get("request_schema")
                != request_schema
            ):
                raise ValueError(f"cached response receipt is invalid: {receipt_path}")
            normalized_hash = str(receipt["normalized_sha256"])
            if normalized_hash in by_hash:
                continue
            normalized_path = (
                self.data_root
                / "normalized"
                / request_schema
                / key
                / f"{normalized_hash}.json"
            )
            normalized_bytes = normalized_path.read_bytes()
            normalized = json.loads(normalized_bytes)
            if (
                shared.digest_bytes(normalized_bytes) != normalized_hash
                or shared.canonical_json_bytes(normalized) != normalized_bytes
            ):
                raise ValueError(f"cached normalized response is invalid: {normalized_path}")
            raw_path = (
                self.data_root
                / "raw"
                / request_schema
                / key
                / f"{receipt['raw_sha256']}.json"
            )
            if shared.digest_file(raw_path) != str(receipt["raw_sha256"]):
                raise ValueError(f"cached raw response is invalid: {raw_path}")
            by_hash[normalized_hash] = shared.StoredResponse(
                request_key=key,
                request_schema=request_schema,
                service=service,
                payload=payload,
                fetched_at_utc=str(receipt["fetched_at_utc"]),
                raw_sha256=str(receipt["raw_sha256"]),
                normalized_sha256=normalized_hash,
                normalized=normalized,
                status=int(receipt["status"]),
                attempts=int(receipt["attempts"]),
                cached=True,
            )
        return [by_hash[key] for key in sorted(by_hash)]

    def _fetch_or_cached_detail(
        self, *, run_id: str, game_id: int, refresh_details: bool
    ) -> tuple[Any | None, list[str], str | None]:
        payload = [game_id, None]
        versions = self._cached_versions("game-detail-v1", payload)
        if refresh_details or not versions:
            try:
                self._fetch(
                    run_id=run_id,
                    request_schema="game-detail-v1",
                    payload=payload,
                )
            except Exception as error:  # preserve a per-game auditable outcome
                return None, [], f"{type(error).__name__}: {error}"
            versions = self._cached_versions("game-detail-v1", payload)
        hashes = sorted({item.normalized_sha256 for item in versions})
        if len(hashes) != 1:
            return None, hashes, None
        return versions[0], hashes, None

    def collect(
        self,
        *,
        run_id: str,
        binding: SourceBinding,
        expected_games: int = 90,
        refresh_details: bool = False,
    ) -> dict[str, Any]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "run id must contain only letters, digits, dot, underscore, or dash"
            )
        if expected_games < 1:
            raise ValueError("expected game count must be positive")
        started_at = self.clock()
        stable_binding = source_binding_record(binding, self.repository)
        stable_binding.update(
            {
                "collector_sha256": self.collector_sha256,
                "purpose": PURPOSE,
                "run_id": run_id,
                "schema": ARENA_BINDING_SCHEMA,
            }
        )
        binding_path = self.data_root / "runs" / run_id / "binding.json"
        if binding_path.exists():
            binding_record = json.loads(binding_path.read_text())
            if any(
                binding_record.get(key) != value
                for key, value in stable_binding.items()
            ):
                raise RuntimeError(
                    f"run id {run_id!r} is already bound to different provenance"
                )
            started_at = str(binding_record["created_at_utc"])
        else:
            binding_record = {**stable_binding, "created_at_utc": started_at}
            shared.write_once(
                binding_path, shared.canonical_json_bytes(binding_record)
            )

        leaderboard_response = self._fetch(
            run_id=run_id,
            request_schema="leaderboard-v1",
            payload=[
                PUZZLE_PUBLIC_ID,
                None,
                "global",
                {"active": False, "column": "", "filter": ""},
            ],
        )
        leaderboard_frozen_at = leaderboard_response.fetched_at_utc
        leaderboard = freeze_leaderboard(leaderboard_response.normalized)

        battles_response = self._fetch(
            run_id=run_id,
            request_schema="agent-battles-v1",
            payload=[binding.agent_id, None],
        )
        if not isinstance(battles_response.normalized, list):
            raise ValueError("agent battle response is not a list")

        battles = []
        seen_game_ids: set[int] = set()
        for raw in battles_response.normalized:
            battle = shared.normalize_battle(raw, binding.agent_id)
            if battle["game_id"] in seen_game_ids:
                raise ValueError(f"battle window repeats game {battle['game_id']}")
            seen_game_ids.add(battle["game_id"])
            battles.append(battle)
        battles.sort(key=lambda item: item["game_id"])

        immediate_records: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for battle in battles:
            prepared = prepare_battle_record(
                battle=battle,
                binding=binding,
                leaderboard=leaderboard,
                leaderboard_frozen_at=leaderboard_frozen_at,
                registry=self.registry,
            )
            if prepared["status"] == "ready-for-detail":
                candidates.append(prepared)
            else:
                immediate_records.append(prepared)

        detail_records: list[dict[str, Any]] = []
        if candidates:
            shared.trainer_module()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.maximum_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._collect_game,
                        run_id=run_id,
                        prepared=prepared,
                        binding=binding,
                        refresh_details=refresh_details,
                    ): prepared["game_id"]
                    for prepared in candidates
                }
                for future in concurrent.futures.as_completed(futures):
                    detail_records.append(future.result())

        records = sorted(
            immediate_records + detail_records,
            key=lambda item: (item["game_id"], item["status"]),
        )
        stored_records = []
        for record in records:
            record.update(
                {
                    "purpose": PURPOSE,
                    "schema": ARENA_GAME_SCHEMA,
                    "source_sha256": binding.source_sha256,
                }
            )
            record_hash, record_path = shared.write_content_addressed(
                self.data_root / "game_records" / str(record["game_id"]), record
            )
            stored_records.append(
                {
                    "record": record,
                    "record_path": shared.relative_to_repository(
                        record_path, self.repository
                    ),
                    "record_sha256": record_hash,
                }
            )

        status_counts: dict[str, int] = {}
        for record in records:
            status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        accepted = [item for item in records if item["status"] == "accepted"]
        clean = [
            item
            for item in accepted
            if item["operational"]["classification"] == "clean"
        ]
        own_failures = [
            item
            for item in accepted
            if item["operational"]["focus_status"] != "ok"
        ]
        opponent_failures = [
            item
            for item in accepted
            if item["operational"]["opponent_status"] != "ok"
        ]
        fully_accounted = all(
            item["status"]
            in {
                "accepted",
                "excluded-protected",
                "already-known-local",
            }
            for item in records
        )
        manifest = {
            "binding": binding_record,
            "collector_sha256": self.collector_sha256,
            "completed_at_utc": self.clock(),
            "coverage": {
                "accepted_games": len(accepted),
                "battle_window_games": len(records),
                "clean_rule_terminal_games": len(clean),
                "expected_games": expected_games,
                "focus_operational_failures": len(own_failures),
                "full_window_accounted": (
                    len(records) >= expected_games and fully_accounted
                ),
                "opponent_operational_failures": len(opponent_failures),
                "status_counts": dict(sorted(status_counts.items())),
            },
            "exclusion_registry": {
                "path": shared.relative_to_repository(
                    self.registry_path, self.repository
                ),
                "sha256": self.registry_sha256,
            },
            "games": stored_records,
            "leaderboard_snapshot": {
                "focus": leaderboard.get(binding.agent_id),
                "frozen_at_utc": leaderboard_frozen_at,
                "normalized_sha256": leaderboard_response.normalized_sha256,
                "raw_sha256": leaderboard_response.raw_sha256,
            },
            "purpose": PURPOSE,
            "request_schemas": shared.REQUEST_SCHEMAS,
            "run_id": run_id,
            "schema": ARENA_BATCH_SCHEMA,
            "started_at_utc": started_at,
            "window_snapshot": {
                "normalized_sha256": battles_response.normalized_sha256,
                "raw_sha256": battles_response.raw_sha256,
            },
        }
        manifest_hash, manifest_path = shared.write_content_addressed(
            self.data_root
            / "manifests"
            / str(binding.agent_id)
            / str(binding.submission_id)
            / binding.source_sha256,
            manifest,
        )
        return {
            "coverage": manifest["coverage"],
            "manifest_path": shared.relative_to_repository(
                manifest_path, self.repository
            ),
            "manifest_sha256": manifest_hash,
            "run_id": run_id,
            "schema": ARENA_BATCH_SCHEMA,
        }

    def _collect_game(
        self,
        *,
        run_id: str,
        prepared: dict[str, Any],
        binding: SourceBinding,
        refresh_details: bool,
    ) -> dict[str, Any]:
        game_id = int(prepared["game_id"])
        response, hashes, request_error = self._fetch_or_cached_detail(
            run_id=run_id,
            game_id=game_id,
            refresh_details=refresh_details,
        )
        base = {key: value for key, value in prepared.items() if key != "status"}
        if request_error is not None:
            return {
                **base,
                "reason": f"detail request failed: {request_error}",
                "status": "request-error",
            }
        if response is None:
            return {
                **base,
                "normalized_payload_sha256": hashes,
                "reason": "conflicting normalized replay payloads are quarantined",
                "status": "payload-conflict",
            }
        try:
            replay = validate_arena_detail(
                response.normalized,
                game_id=game_id,
                battle=prepared["battle"],
                focus_agent_id=binding.agent_id,
                leaderboard_frozen_at=prepared["leaderboard_frozen_at_utc"],
            )
        except (KeyError, TypeError, ValueError) as error:
            return {
                **base,
                "acquisition": response_record(response, self.repository, self.data_root),
                "reason": str(error),
                "status": "structural-rejection",
            }
        replay_payload_path = (
            self.data_root
            / "replay_payloads"
            / str(game_id)
            / f"{response.normalized_sha256}.json"
        )
        shared.write_once(
            replay_payload_path, shared.canonical_json_bytes(response.normalized)
        )
        replay_focus = replay.pop("focus")
        replay_opponent = replay.pop("opponent")
        return {
            **base,
            "acquisition": {
                **response_record(response, self.repository, self.data_root),
                "replay_payload_path": shared.relative_to_repository(
                    replay_payload_path, self.repository
                ),
            },
            "focus": {**base["focus"], **replay_focus},
            "opponent": {**base["opponent"], **replay_opponent},
            **replay,
            "status": "accepted",
        }


def repository_head(repository: pathlib.Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise ValueError("repository HEAD is not a Git object id")
    return commit


def load_exclusion_registry(
    path: pathlib.Path | None, expected_sha256: str | None
) -> Any:
    if path is None or expected_sha256 is None:
        raise ValueError("exclusion registry path and approved SHA-256 are required")
    path = path.resolve()
    expected_sha256 = expected_sha256.lower()
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("exclusion registry SHA-256 must be 64 lowercase hex digits")
    if shared.digest_file(path) != expected_sha256:
        raise ValueError(f"exclusion registry SHA-256 mismatch: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema") != shared.EXCLUSION_SCHEMA:
        raise ValueError(f"unexpected exclusion registry schema: {path}")
    if shared.canonical_json_bytes(payload) != path.read_bytes():
        raise ValueError(f"exclusion registry is not canonical JSON: {path}")
    if SHA256_PATTERN.fullmatch(path.stem) and shared.digest_file(path) != path.stem:
        raise ValueError(f"exclusion registry filename hash mismatch: {path}")
    return shared.ExclusionRegistry(payload)


def source_binding_record(
    binding: SourceBinding, repository: pathlib.Path
) -> dict[str, Any]:
    return {
        "agent_id": binding.agent_id,
        "asserted_submission_id": binding.submission_id,
        "repository_commit": binding.repository_commit,
        "source": {
            "archived_path": shared.relative_to_repository(
                binding.archived_path, repository
            ),
            "bytes": binding.source_bytes,
            "characters": binding.source_characters,
            "encoding": binding.source_encoding,
            "input_path": shared.relative_to_repository(
                binding.source_path, repository
            ),
            "sha256": binding.source_sha256,
        },
    }


def freeze_leaderboard(payload: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("users"), list):
        raise ValueError("leaderboard response omits users")
    result = {}
    for raw in payload["users"]:
        if not isinstance(raw, dict):
            raise ValueError("leaderboard contains a non-object user")
        try:
            agent_id = int(raw["agentId"])
            rank = int(raw["rank"])
            score = float(raw["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("leaderboard user omits agent id, rank, or score") from error
        if agent_id in result:
            raise ValueError(f"leaderboard repeats agent {agent_id}")
        codingamer = raw.get("codingamer") or {}
        result[agent_id] = {
            "agent_id": agent_id,
            "in_progress": bool(raw.get("inProgress", False)),
            "name": str(raw.get("pseudo") or codingamer.get("pseudo") or ""),
            "programming_language": raw.get("programmingLanguage"),
            "public_handle": codingamer.get("publicHandle"),
            "rank": rank,
            "score": score,
            "session_id": raw.get("testSessionHandle"),
            "user_id": codingamer.get("userId"),
        }
    return result


def prepare_battle_record(
    *,
    battle: dict[str, Any],
    binding: SourceBinding,
    leaderboard: Mapping[int, dict[str, Any]],
    leaderboard_frozen_at: str,
    registry: Any,
) -> dict[str, Any]:
    players = battle["players"]
    focus = next(
        player for player in players if int(player["agent_id"]) == binding.agent_id
    )
    opponent = next(
        player for player in players if int(player["agent_id"]) != binding.agent_id
    )
    focus_submission = focus.get("submission_id")
    try:
        normalized_submission = int(focus_submission)
    except (TypeError, ValueError):
        normalized_submission = None
    focus_position = focus.get("result_position")
    opponent_position = opponent.get("result_position")
    result = None
    if focus_position is not None and opponent_position is not None:
        result = "win" if int(focus_position) < int(opponent_position) else "loss"
    frozen_opponent = leaderboard.get(int(opponent["agent_id"]))
    base = {
        "battle": {
            "done": bool(battle["done"]),
            "players": players,
        },
        "focus": {
            "agent_id": binding.agent_id,
            "color": "player-unknown",
            "result": result,
            "session_id": focus.get("session_id"),
            "submission_id": normalized_submission,
        },
        "game_id": int(battle["game_id"]),
        "leaderboard_frozen_at_utc": leaderboard_frozen_at,
        "opponent": {
            "agent_id": int(opponent["agent_id"]),
            "frozen_rank": None if frozen_opponent is None else frozen_opponent["rank"],
            "frozen_score": None if frozen_opponent is None else frozen_opponent["score"],
            "name": opponent.get("name"),
            "public_handle": opponent.get("public_handle"),
            "session_id": opponent.get("session_id"),
            "submission_id": opponent.get("submission_id"),
            "user_id": opponent.get("user_id"),
        },
    }
    if not battle["done"]:
        return {**base, "reason": "battle is not complete", "status": "pending"}
    registry_record = registry.records.get(int(battle["game_id"]))
    if registry_record is not None:
        protected = registry.is_protected(int(battle["game_id"]))
        return {
            **base,
            "exclusion_categories": registry_record["categories"],
            "reason": (
                "protected data boundary"
                if protected
                else "game already exists in known local evidence"
            ),
            "status": "excluded-protected" if protected else "already-known-local",
        }
    if normalized_submission != binding.submission_id:
        return {
            **base,
            "reason": (
                "battle focus submission does not match the source binding: "
                f"expected {binding.submission_id}, observed {normalized_submission}"
            ),
            "status": "identity-mismatch",
        }
    return {**base, "status": "ready-for-detail"}


def response_record(
    response: Any, repository: pathlib.Path, data_root: pathlib.Path
) -> dict[str, Any]:
    key = response.request_key
    schema = response.request_schema
    return {
        "cached": bool(response.cached),
        "fetched_at_utc": response.fetched_at_utc,
        "normalized_path": shared.relative_to_repository(
            data_root
            / "normalized"
            / schema
            / key
            / f"{response.normalized_sha256}.json",
            repository,
        ),
        "normalized_sha256": response.normalized_sha256,
        "raw_path": shared.relative_to_repository(
            data_root / "raw" / schema / key / f"{response.raw_sha256}.json",
            repository,
        ),
        "raw_sha256": response.raw_sha256,
        "request_key": key,
    }


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_text_values(child))
        return result
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(_text_values(child))
        return result
    return []


def _scoped_player(frame: Mapping[str, Any], texts: Iterable[str]) -> int | None:
    raw_agent = frame.get("agentId", -1)
    if not isinstance(raw_agent, bool):
        try:
            agent = int(raw_agent)
        except (TypeError, ValueError):
            agent = -1
        if agent in (0, 1):
            return agent
    explicit = set()
    for text in texts:
        explicit.update(int(value) for value in re.findall(r"\$([01])\b", text))
    return next(iter(explicit)) if len(explicit) == 1 else None


def parse_frames(frames: Any, game_id: int) -> dict[str, Any]:
    if not isinstance(frames, list):
        raise ValueError(f"game {game_id} omits frames")
    turns = []
    signals = []
    statuses = {0: "ok", 1: "ok"}
    unscoped_signals = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"game {game_id} contains a non-object frame")
        field_texts = {
            field: _text_values(frame[field])
            for field in TEXT_FRAME_FIELDS
            if field in frame
        }
        all_texts = [text for values in field_texts.values() for text in values]
        player_id = _scoped_player(frame, all_texts)
        for field, texts in field_texts.items():
            for text in texts:
                for category, pattern in FAILURE_PATTERNS:
                    if pattern.search(text) is None:
                        continue
                    signal = {
                        "category": category,
                        "evidence": text[:1000],
                        "field": field,
                        "frame_index": frame_index,
                        "player_id": player_id,
                    }
                    if player_id is None:
                        unscoped_signals.append(signal)
                    else:
                        signals.append(signal)
                        statuses[player_id] = stronger_failure(
                            statuses[player_id], category
                        )
                    break

        if player_id not in (0, 1) or "stdout" not in frame:
            continue
        raw_stdout = "" if frame.get("stdout") is None else str(frame["stdout"])
        action = raw_stdout.strip()
        if not action:
            category = "empty-output"
            if statuses[player_id] == "ok":
                statuses[player_id] = category
                signals.append(
                    {
                        "category": category,
                        "evidence": "agent action frame contains empty stdout",
                        "field": "stdout",
                        "frame_index": frame_index,
                        "player_id": player_id,
                    }
                )
            continue
        turns.append(
            {
                "action": action,
                "frame_index": frame_index,
                "player_id": player_id,
                "stdout": raw_stdout,
            }
        )
        if any(character not in "01234567" for character in action):
            statuses[player_id] = stronger_failure(
                statuses[player_id], "invalid-output"
            )
            signals.append(
                {
                    "category": "invalid-output",
                    "evidence": action[:1000],
                    "field": "stdout",
                    "frame_index": frame_index,
                    "player_id": player_id,
                }
            )
    return {
        "player_statuses": statuses,
        "signals": signals,
        "turns": turns,
        "unscoped_signals": unscoped_signals,
    }


def stronger_failure(current: str, candidate: str) -> str:
    return (
        candidate
        if FAILURE_PRECEDENCE[candidate] > FAILURE_PRECEDENCE[current]
        else current
    )


def validate_turns(turns: list[dict[str, Any]], winner: int) -> dict[str, Any]:
    trainer = shared.trainer_module()
    ball = (trainer.WIDTH // 2, trainer.HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    to_move = 0
    terminal_winner = None
    valid_turns = []
    for turn_index, turn in enumerate(turns):
        player = int(turn["player_id"])
        action = str(turn["action"])
        if terminal_winner is not None:
            return validation_failure(
                "turn-after-terminal", turn_index, player, valid_turns
            )
        if player != to_move:
            return validation_failure(
                "wrong-player-to-move", turn_index, player, valid_turns
            )
        if not action or any(character not in "01234567" for character in action):
            return validation_failure(
                "invalid-output", turn_index, player, valid_turns
            )
        for edge_offset, encoded in enumerate(action):
            direction = int(encoded)
            dx, dy = trainer.DIRECTIONS[direction]
            destination = (ball[0] + dx, ball[1] + dy)
            edge = trainer.segment(ball, destination)
            if edge not in trainer.EDGE_INDEX or edge in used_segments:
                return validation_failure(
                    "illegal-edge", turn_index, player, valid_turns
                )
            was_visited = destination in visited
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
            used_edge_ids = {trainer.EDGE_INDEX[value] for value in used_segments}
            free = [
                edge_id
                for _, edge_id in trainer.ADJACENCY[trainer.POINT_INDEX[ball]]
                if edge_id not in used_edge_ids
            ]
            if trainer.is_goal(ball):
                terminal_winner = 0 if ball[1] == trainer.GOAL_BOTTOM else 1
            elif not free:
                terminal_winner = 1 - player
            elif not (was_visited or trainer.is_boundary(ball)):
                to_move = 1 - player
            if edge_offset + 1 < len(action):
                if terminal_winner is not None or to_move != player:
                    return validation_failure(
                        "overlong-rebound-action", turn_index, player, valid_turns
                    )
            elif terminal_winner is None and to_move == player:
                return validation_failure(
                    "mandatory-rebound-omitted", turn_index, player, valid_turns
                )
        valid_turns.append(
            {"action": action, "player_id": player}
        )
    if terminal_winner is None:
        return {
            "failing_player_id": None,
            "reason": "transcript is incomplete under game rules",
            "status": "incomplete",
            "terminal_winner_player_id": None,
            "valid_turn_count": len(valid_turns),
            "valid_turns": valid_turns,
        }
    if terminal_winner != winner:
        return {
            "failing_player_id": None,
            "reason": (
                f"recorded winner {winner} disagrees with replayed winner "
                f"{terminal_winner}"
            ),
            "status": "winner-mismatch",
            "terminal_winner_player_id": terminal_winner,
            "valid_turn_count": len(valid_turns),
            "valid_turns": valid_turns,
        }
    return {
        "failing_player_id": None,
        "reason": None,
        "status": "terminal-valid",
        "terminal_winner_player_id": terminal_winner,
        "valid_turn_count": len(valid_turns),
        "valid_turns": valid_turns,
    }


def validation_failure(
    reason: str,
    turn_index: int,
    player_id: int,
    valid_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "failing_player_id": player_id,
        "reason": reason,
        "status": "invalid",
        "terminal_winner_player_id": None,
        "turn_index": turn_index,
        "valid_turn_count": len(valid_turns),
        "valid_turns": valid_turns,
    }


def validate_arena_detail(
    payload: Any,
    *,
    game_id: int,
    battle: dict[str, Any],
    focus_agent_id: int,
    leaderboard_frozen_at: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or int(payload.get("gameId", -1)) != game_id:
        raise ValueError(f"game detail id does not match requested game {game_id}")
    agents = payload.get("agents")
    ranks = payload.get("ranks")
    if not isinstance(agents, list) or len(agents) != 2:
        raise ValueError(f"game {game_id} detail is not two-player")
    if (
        not isinstance(ranks, list)
        or len(ranks) != 2
        or any(isinstance(rank, bool) or type(rank) is not int for rank in ranks)
        or sorted(ranks) != [0, 1]
    ):
        raise ValueError(f"game {game_id} has invalid ranks")
    metadata_by_agent = {
        int(player["agent_id"]): player for player in battle["players"]
    }
    by_player = {}
    seen_agents = set()
    for raw in agents:
        try:
            agent_id = int(raw["agentId"])
            player_id = int(raw["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"game {game_id} agent omits identity") from error
        if player_id not in (0, 1) or player_id in by_player or agent_id in seen_agents:
            raise ValueError(f"game {game_id} repeats or invalidates an agent identity")
        if agent_id not in metadata_by_agent:
            raise ValueError(f"game {game_id} detail disagrees with battle agents")
        result_position = metadata_by_agent[agent_id].get("result_position")
        if result_position is not None and int(result_position) != int(ranks[player_id]):
            raise ValueError(f"game {game_id} battle result disagrees with replay ranks")
        by_player[player_id] = {
            "agent_id": agent_id,
            "name": str((raw.get("codingamer") or {}).get("pseudo") or raw.get("name") or ""),
            "player_id": player_id,
            "rank_result": int(ranks[player_id]),
            "session_id": metadata_by_agent[agent_id].get("session_id"),
            "submission_id": metadata_by_agent[agent_id].get("submission_id"),
            "user_id": metadata_by_agent[agent_id].get("user_id"),
        }
        seen_agents.add(agent_id)
    if set(by_player) != {0, 1} or focus_agent_id not in seen_agents:
        raise ValueError(f"game {game_id} omits the focus agent or a player index")

    focus_player = next(
        player for player, value in by_player.items() if value["agent_id"] == focus_agent_id
    )
    opponent_player = 1 - focus_player
    winner = ranks.index(0)
    parsed = parse_frames(payload.get("frames"), game_id)
    validation = validate_turns(parsed["turns"], winner)
    statuses = parsed["player_statuses"]
    failing_player = validation.get("failing_player_id")
    if validation["status"] == "invalid" and failing_player in (0, 1):
        category = (
            "invalid-output"
            if validation["reason"] == "invalid-output"
            else "illegal-action"
        )
        statuses[failing_player] = stronger_failure(statuses[failing_player], category)
        parsed["signals"].append(
            {
                "category": category,
                "evidence": validation["reason"],
                "field": "rules-validation",
                "frame_index": parsed["turns"][validation["turn_index"]]["frame_index"],
                "player_id": failing_player,
            }
        )

    any_failure = any(status != "ok" for status in statuses.values())
    if validation["status"] == "terminal-valid" and any_failure:
        raise ValueError(
            f"game {game_id} has a rule-terminal transcript and operational failure signals"
        )
    if validation["status"] in {"incomplete", "invalid"} and not any_failure:
        raise ValueError(
            f"game {game_id} transcript is {validation['status']} without an operational ending"
        )
    if validation["status"] == "winner-mismatch":
        raise ValueError(f"game {game_id} {validation['reason']}")

    classification = "clean" if not any_failure else "operationally-terminated"
    focus_status = statuses[focus_player]
    opponent_status = statuses[opponent_player]
    observed_turns = [
        {"action": turn["action"], "player_id": turn["player_id"]}
        for turn in parsed["turns"]
    ]
    return {
        "focus": {
            "agent_id": focus_agent_id,
            "color": f"player-{focus_player}",
            "player_id": focus_player,
            "result": "win" if winner == focus_player else "loss",
        },
        "leaderboard_frozen_at_utc": leaderboard_frozen_at,
        "operational": {
            "classification": classification,
            "focus_status": focus_status,
            "opponent_status": opponent_status,
            "player_statuses": {
                str(player): statuses[player] for player in (0, 1)
            },
            "signals": parsed["signals"],
            "unscoped_signals": parsed["unscoped_signals"],
        },
        "opponent": {
            "agent_id": by_player[opponent_player]["agent_id"],
            "player_id": opponent_player,
        },
        "outcome": {
            "winner_agent_id": by_player[winner]["agent_id"],
            "winner_player_id": winner,
        },
        "replay": {
            "agents": [by_player[player] for player in (0, 1)],
            "observed_transcript": "/".join(
                turn["action"] for turn in observed_turns
            ),
            "observed_turns": observed_turns,
            "rules_validation": validation,
            "valid_transcript": "/".join(
                turn["action"] for turn in validation["valid_turns"]
            ),
            "valid_turns": validation["valid_turns"],
        },
    }


def check_store(data_root: pathlib.Path) -> dict[str, int]:
    counts = {
        "exclusion_registries": 0,
        "game_records": 0,
        "manifests": 0,
        "normalized_responses": 0,
        "raw_responses": 0,
        "receipts": 0,
        "replay_payloads": 0,
        "run_bindings": 0,
        "source_payloads": 0,
    }
    for path in sorted((data_root / "exclusions").glob("*.json")):
        payload = json.loads(path.read_bytes())
        if (
            shared.digest_file(path) != path.stem
            or shared.canonical_json_bytes(payload) != path.read_bytes()
            or payload.get("schema") != shared.EXCLUSION_SCHEMA
        ):
            raise ValueError(f"exclusion registry is invalid: {path}")
        counts["exclusion_registries"] += 1
    for path in sorted((data_root / "source_payloads").glob("*.source")):
        if shared.digest_file(path) != path.stem:
            raise ValueError(f"source payload hash mismatch: {path}")
        counts["source_payloads"] += 1
    for key, relative in (
        ("raw_responses", "raw"),
        ("normalized_responses", "normalized"),
        ("replay_payloads", "replay_payloads"),
    ):
        for path in sorted((data_root / relative).rglob("*.json")):
            if shared.digest_file(path) != path.stem:
                raise ValueError(f"content hash mismatch: {path}")
            json.loads(path.read_bytes())
            if relative != "raw":
                payload = json.loads(path.read_text())
                if shared.canonical_json_bytes(payload) != path.read_bytes():
                    raise ValueError(f"payload is not canonical: {path}")
            counts[key] += 1
    for path in sorted((data_root / "receipts").glob("*/*/*.json")):
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != ARENA_RECEIPT_SCHEMA
            or shared.canonical_json_bytes(payload) != path.read_bytes()
            or shared.digest_file(path) != path.stem
        ):
            raise ValueError(f"response receipt is invalid: {path}")
        counts["receipts"] += 1
    for path in sorted((data_root / "runs").glob("*/binding.json")):
        payload = json.loads(path.read_bytes())
        source = payload.get("source") or {}
        archived = pathlib.Path(str(source.get("archived_path", "")))
        if not archived.is_absolute():
            archived = REPOSITORY / archived
        if (
            payload.get("schema") != ARENA_BINDING_SCHEMA
            or shared.canonical_json_bytes(payload) != path.read_bytes()
            or not archived.is_file()
            or shared.digest_file(archived) != source.get("sha256")
        ):
            raise ValueError(f"run binding is invalid: {path}")
        counts["run_bindings"] += 1
    for key, relative, schema in (
        ("game_records", "game_records", ARENA_GAME_SCHEMA),
        ("manifests", "manifests", ARENA_BATCH_SCHEMA),
    ):
        for path in sorted((data_root / relative).rglob("*.json")):
            payload = json.loads(path.read_text())
            if (
                payload.get("schema") != schema
                or shared.canonical_json_bytes(payload) != path.read_bytes()
                or shared.digest_file(path) != path.stem
            ):
                raise ValueError(f"content-addressed record is invalid: {path}")
            if schema == ARENA_BATCH_SCHEMA:
                for stored in payload.get("games", []):
                    record_path = pathlib.Path(str(stored.get("record_path", "")))
                    if not record_path.is_absolute():
                        record_path = REPOSITORY / record_path
                    record_sha256 = str(stored.get("record_sha256", ""))
                    if (
                        not record_path.is_file()
                        or shared.digest_file(record_path) != record_sha256
                        or shared.digest_bytes(
                            shared.canonical_json_bytes(stored.get("record"))
                        )
                        != record_sha256
                    ):
                        raise ValueError(
                            f"manifest references an invalid game record: {path}"
                        )
            counts[key] += 1
    return counts


def _resolve_archived_path(raw_path: Any, repository: pathlib.Path) -> pathlib.Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("archive cross-reference path is missing or invalid")
    path = pathlib.Path(raw_path)
    return path if path.is_absolute() else repository / path


def validate_export_manifest(
    manifest_path: pathlib.Path,
    expected_exclusion_registry_sha256: str,
    *,
    repository: pathlib.Path = REPOSITORY,
) -> dict[str, Any]:
    """Validate every archive object trusted by an offline auditor export."""

    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != ARENA_BATCH_SCHEMA:
        raise ValueError(f"unexpected arena manifest schema: {manifest_path}")
    if (
        shared.canonical_json_bytes(manifest) != manifest_bytes
        or shared.digest_file(manifest_path) != manifest_path.stem
    ):
        raise ValueError(f"arena manifest is not content addressed: {manifest_path}")

    if not isinstance(expected_exclusion_registry_sha256, str):
        raise ValueError("approved exclusion registry SHA-256 is invalid")
    expected_registry_hash = expected_exclusion_registry_sha256.lower()
    if SHA256_PATTERN.fullmatch(expected_registry_hash) is None:
        raise ValueError("approved exclusion registry SHA-256 is invalid")
    exclusion = manifest.get("exclusion_registry")
    if not isinstance(exclusion, dict):
        raise ValueError("arena manifest omits its exclusion registry binding")
    if exclusion.get("sha256") != expected_registry_hash:
        raise ValueError("arena manifest exclusion registry SHA-256 is not approved")
    registry_path = _resolve_archived_path(exclusion.get("path"), repository)
    registry = load_exclusion_registry(registry_path, expected_registry_hash)

    binding = manifest.get("binding")
    if not isinstance(binding, dict) or binding.get("schema") != ARENA_BINDING_SCHEMA:
        raise ValueError("arena manifest has an invalid source binding")
    agent_id = binding.get("agent_id")
    submission_id = binding.get("asserted_submission_id")
    if (
        isinstance(agent_id, bool)
        or not isinstance(agent_id, int)
        or agent_id <= 0
        or isinstance(submission_id, bool)
        or not isinstance(submission_id, int)
        or submission_id <= 0
        or binding.get("run_id") != manifest.get("run_id")
        or binding.get("collector_sha256") != manifest.get("collector_sha256")
        or binding.get("purpose") != PURPOSE
        or manifest.get("purpose") != PURPOSE
    ):
        raise ValueError("arena manifest source binding contradicts the manifest")
    source = binding.get("source")
    if not isinstance(source, dict):
        raise ValueError("arena manifest omits its source payload binding")
    source_hash = str(source.get("sha256", ""))
    if SHA256_PATTERN.fullmatch(source_hash) is None:
        raise ValueError("arena manifest source SHA-256 is invalid")
    source_path = _resolve_archived_path(source.get("archived_path"), repository)
    source_bytes = source_path.read_bytes()
    if shared.digest_bytes(source_bytes) != source_hash:
        raise ValueError(f"source payload hash mismatch: {source_path}")
    try:
        decoded_source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"source payload is not valid UTF-8: {source_path}") from error
    if (
        source.get("encoding") != "utf-8"
        or source.get("bytes") != len(source_bytes)
        or source.get("characters") != len(decoded_source)
    ):
        raise ValueError("arena manifest source payload metadata is inconsistent")

    games = manifest.get("games")
    if not isinstance(games, list):
        raise ValueError("arena manifest games must be a list")
    seen_game_ids: set[int] = set()
    for stored in games:
        if not isinstance(stored, dict) or not isinstance(stored.get("record"), dict):
            raise ValueError("arena manifest contains an invalid game record binding")
        record = stored["record"]
        record_hash = str(stored.get("record_sha256", ""))
        if (
            SHA256_PATTERN.fullmatch(record_hash) is None
            or shared.digest_bytes(shared.canonical_json_bytes(record)) != record_hash
            or record.get("schema") != ARENA_GAME_SCHEMA
            or record.get("purpose") != PURPOSE
            or record.get("source_sha256") != source_hash
        ):
            raise ValueError("arena manifest embeds an invalid game record")
        record_path = _resolve_archived_path(stored.get("record_path"), repository)
        record_bytes = record_path.read_bytes()
        if (
            shared.digest_bytes(record_bytes) != record_hash
            or record_bytes != shared.canonical_json_bytes(record)
        ):
            raise ValueError(f"game record cross-reference mismatch: {record_path}")

        game_id = record.get("game_id")
        if (
            isinstance(game_id, bool)
            or not isinstance(game_id, int)
            or game_id <= 0
            or game_id in seen_game_ids
        ):
            raise ValueError("arena manifest contains an invalid or repeated game id")
        seen_game_ids.add(game_id)
        focus = record.get("focus")
        if (
            not isinstance(focus, dict)
            or focus.get("agent_id") != agent_id
            or (
                record.get("status") == "accepted"
                and focus.get("submission_id") != submission_id
            )
        ):
            raise ValueError(f"game {game_id} contradicts the source binding")
        if game_id in registry.known_ids:
            expected_status = (
                "excluded-protected"
                if registry.is_protected(game_id)
                else "already-known-local"
            )
            if record.get("status") != expected_status:
                raise ValueError(
                    f"game {game_id} contradicts the approved exclusion registry"
                )
        elif record.get("status") in {
            "excluded-protected",
            "already-known-local",
        }:
            raise ValueError(
                f"game {game_id} claims an exclusion absent from the approved registry"
            )

    return manifest


def export_auditor_tsv(
    manifest_path: pathlib.Path,
    output_path: pathlib.Path,
    expected_exclusion_registry_sha256: str,
    *,
    repository: pathlib.Path = REPOSITORY,
) -> dict[str, Any]:
    """Export clean terminal games to the native decision auditor's TSV."""

    manifest_path = manifest_path.resolve()
    manifest = validate_export_manifest(
        manifest_path,
        expected_exclusion_registry_sha256,
        repository=repository,
    )
    rows = []
    skipped = {}
    for stored in manifest.get("games", []):
        record = stored.get("record") or {}
        reason = None
        if record.get("status") != "accepted":
            reason = str(record.get("status") or "not-accepted")
        elif (record.get("operational") or {}).get("classification") != "clean":
            reason = "operationally-terminated"
        elif (
            ((record.get("replay") or {}).get("rules_validation") or {}).get(
                "status"
            )
            != "terminal-valid"
        ):
            reason = "not-rule-terminal"
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        game_id = int(record["game_id"])
        candidate_player = int(record["focus"]["player_id"])
        winner = int(record["outcome"]["winner_player_id"])
        transcript = str(record["replay"]["valid_transcript"])
        if candidate_player not in (0, 1) or winner not in (0, 1):
            raise ValueError(f"game {game_id} has an invalid player identity")
        if not transcript or any(
            not action or any(character not in "01234567" for character in action)
            for action in transcript.split("/")
        ):
            raise ValueError(f"game {game_id} has an invalid terminal transcript")
        rows.append((game_id, candidate_player, winner, transcript))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError("arena manifest has no clean rule-terminal games to export")
    binding = manifest.get("binding") or {}
    source = binding.get("source") or {}
    exclusion = manifest.get("exclusion_registry") or {}
    metadata = {
        "agent_id": str(binding["agent_id"]),
        "arena_manifest_sha256": manifest_path.stem,
        "asserted_source_sha256": str(source["sha256"]),
        "asserted_submission_id": str(binding["asserted_submission_id"]),
        "collector_sha256": str(manifest["collector_sha256"]),
        "exclusion_registry_sha256": str(exclusion["sha256"]),
        "repository_commit": str(binding["repository_commit"]),
        "run_id": str(manifest["run_id"]),
        "source_binding_status": "asserted-not-api-verified",
    }
    content = "".join(
        f"# {key}={value}\n" for key, value in sorted(metadata.items())
    )
    content += AUDITOR_TSV_HEADER + "\n" + "".join(
        f"{game_id}\t{candidate_player}\t{winner}\t{transcript}\n"
        for game_id, candidate_player, winner, transcript in rows
    )
    output_path = output_path.resolve()
    shared.write_once(output_path, content.encode("ascii"))
    return {
        "exported_games": len(rows),
        "output_path": output_path.as_posix(),
        "sha256": shared.digest_bytes(content.encode("ascii")),
        "skipped": dict(sorted(skipped.items())),
    }


def default_run_id() -> str:
    return shared.default_run_id()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", type=int)
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--repository-commit")
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--expected-games", type=int, default=90)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exclusion-registry", type=pathlib.Path)
    parser.add_argument("--exclusion-registry-sha256")
    parser.add_argument("--maximum-workers", type=int, default=2)
    parser.add_argument("--refresh-details", action="store_true")
    parser.add_argument("--export-auditor-tsv", type=pathlib.Path)
    parser.add_argument("--export-existing-manifest", type=pathlib.Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        print(json.dumps(check_store(arguments.data_root.resolve()), sort_keys=True))
        return
    if arguments.export_existing_manifest is not None:
        if arguments.export_auditor_tsv is None:
            parser.error("--export-existing-manifest requires --export-auditor-tsv")
        if arguments.exclusion_registry_sha256 is None:
            parser.error(
                "--export-existing-manifest requires "
                "--exclusion-registry-sha256"
            )
        print(
            json.dumps(
                export_auditor_tsv(
                    arguments.export_existing_manifest,
                    arguments.export_auditor_tsv,
                    arguments.exclusion_registry_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    missing = [
        name
        for name, value in (
            ("--agent-id", arguments.agent_id),
            ("--submission-id", arguments.submission_id),
            ("--source", arguments.source),
            ("--exclusion-registry", arguments.exclusion_registry),
            ("--exclusion-registry-sha256", arguments.exclusion_registry_sha256),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required unless --check: {', '.join(missing)}")
    collector = ArenaBatchCollector(
        data_root=arguments.data_root,
        exclusion_registry_path=arguments.exclusion_registry,
        exclusion_registry_sha256=arguments.exclusion_registry_sha256,
        maximum_workers=arguments.maximum_workers,
    )
    binding = collector.bind_source(
        agent_id=arguments.agent_id,
        submission_id=arguments.submission_id,
        source_path=arguments.source,
        expected_source_sha256=arguments.source_sha256,
        repository_commit=arguments.repository_commit,
    )
    result = collector.collect(
        run_id=arguments.run_id,
        binding=binding,
        expected_games=arguments.expected_games,
        refresh_details=arguments.refresh_details,
    )
    if arguments.export_auditor_tsv is not None:
        manifest_path = pathlib.Path(result["manifest_path"])
        if not manifest_path.is_absolute():
            manifest_path = collector.repository / manifest_path
        result["auditor_tsv"] = export_auditor_tsv(
            manifest_path,
            arguments.export_auditor_tsv,
            arguments.exclusion_registry_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
