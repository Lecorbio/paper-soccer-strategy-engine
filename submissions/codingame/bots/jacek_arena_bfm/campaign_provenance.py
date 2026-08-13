#!/usr/bin/env python3

"""Clean-room provenance contracts for the ``jacek_arena_bfm`` campaign.

This module intentionally has no dependency on a historical bot, model, replay
parser, or training pipeline.  The exclusion builder accepts only an already
sealed ID-only registry, newline-delimited numeric directory inventories, and
a strictly checked CodinGame battle-*metadata* snapshot.  Inputs containing
frames, outputs, transcripts, or replay payload fields are rejected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


NAMESPACE = "jacek_arena_bfm"
DEFAULT_T0_UTC = "2026-08-13T10:12:52Z"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"
WINDOW_PLAN_SCHEMA = "papersoccer.jacek-arena-bfm.window-plan.v1"
EDITOR_ATTESTATION_SCHEMA = "papersoccer.jacek-arena-bfm.editor-attestation.v1"
ARENA_DERIVATION_SCHEMA = "papersoccer.jacek-arena-bfm.arena-derivation.v1"
PROTECTED_SNAPSHOT_SCHEMA = "papersoccer.jacek-arena-bfm.protected-snapshot.v1"
ARENA_BATCH_SCHEMA = "papersoccer.codingame-arena-batch.v1"
ARENA_GAME_SCHEMA = "papersoccer.codingame-arena-game.v1"
ARENA_BINDING_SCHEMA = "papersoccer.codingame-arena-binding.v1"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
WINDOW_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")
ROLE_BY_ORDINAL = ("training", "training", "training", "arena-validation")
ASCII_SOURCE_LIMIT = 99_999
EXACT_WINDOW_GAMES = 90
SAFE_ROLLBACK_SOURCE_SHA256 = "d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71"
SAFE_ROLLBACK_SOURCE_BYTES = 99_810

_BASE_REGISTRY_KEYS = {"records", "schema", "selection", "sources"}
_BASE_SOURCE_KEYS = {"category", "game_id_count", "path", "sha256"}
_BASE_RECORD_KEYS = {"categories", "game_id", "sources"}
_BATTLE_KEYS = {"done", "gameId", "players"}
_PLAYER_KEYS = {
    "avatar",
    "nickname",
    "playerAgentId",
    "position",
    "publicHandle",
    "submissionId",
    "testSessionHandle",
    "userId",
}
_FORBIDDEN_METADATA_KEYS = {
    "agents",
    "frames",
    "gameInformation",
    "inputs",
    "observed_transcript",
    "outputs",
    "replay",
    "stderr",
    "stdin",
    "stdout",
    "transcript",
    "turns",
}


class ProvenanceError(ValueError):
    """Raised when a campaign artifact violates its immutable contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: Any, field: str = "timestamp") -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProvenanceError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        instant = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ProvenanceError(f"{field} is not a valid ISO-8601 timestamp") from error
    if instant.tzinfo != dt.timezone.utc:
        raise ProvenanceError(f"{field} must be UTC")
    return instant


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProvenanceError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _require_exact_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ProvenanceError(f"{field} contains forbidden/unexpected keys: {unexpected}")


def _read_canonical_json(path: pathlib.Path, *, expected_sha256: str | None = None) -> Any:
    content = path.read_bytes()
    actual_sha256 = sha256_bytes(content)
    if expected_sha256 is not None and actual_sha256 != _sha256(expected_sha256, "expected SHA-256"):
        raise ProvenanceError(f"SHA-256 mismatch for {path}: {actual_sha256}")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid JSON: {path}") from error
    if canonical_json_bytes(value) != content:
        raise ProvenanceError(f"JSON is not canonical: {path}")
    if SHA256_RE.fullmatch(path.stem) is not None and path.stem != actual_sha256:
        raise ProvenanceError(f"content-addressed filename does not match content: {path}")
    return value


def _write_once(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProvenanceError(f"immutable artifact collision: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ProvenanceError(f"immutable artifact collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_content_addressed(directory: pathlib.Path, value: Any) -> tuple[str, pathlib.Path]:
    content = canonical_json_bytes(value)
    digest = sha256_bytes(content)
    path = directory / f"{digest}.json"
    _write_once(path, content)
    return digest, path


def load_content_addressed(path: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    value = _read_canonical_json(path)
    if not isinstance(value, dict):
        raise ProvenanceError(f"content-addressed artifact must be an object: {path}")
    if schema is not None and value.get("schema") != schema:
        raise ProvenanceError(f"unexpected schema in {path}: {value.get('schema')!r}")
    return value


def load_id_only_registry(path: pathlib.Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load and strictly validate a canonical registry without following sources."""

    value = _read_canonical_json(path, expected_sha256=expected_sha256)
    if not isinstance(value, dict):
        raise ProvenanceError("base exclusion registry must be an object")
    _require_exact_keys(value, _BASE_REGISTRY_KEYS, "base registry")
    if value.get("schema") != EXCLUSION_SCHEMA or not isinstance(value.get("selection"), str):
        raise ProvenanceError("base exclusion registry has an invalid schema or selection")
    sources = value.get("sources")
    records = value.get("records")
    if not isinstance(sources, list) or not isinstance(records, list):
        raise ProvenanceError("base exclusion registry sources and records must be lists")
    source_paths: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ProvenanceError(f"base registry source {index} must be an object")
        _require_exact_keys(source, _BASE_SOURCE_KEYS, f"base registry source {index}")
        path_value = source.get("path")
        if not isinstance(path_value, str) or not path_value or path_value in source_paths:
            raise ProvenanceError(f"base registry source {index} has an invalid/repeated path")
        source_paths.add(path_value)
        if not isinstance(source.get("category"), str) or not source["category"]:
            raise ProvenanceError(f"base registry source {index} has an invalid category")
        if isinstance(source.get("game_id_count"), bool) or not isinstance(source.get("game_id_count"), int) or source["game_id_count"] < 0:
            raise ProvenanceError(f"base registry source {index} has an invalid game count")
        _sha256(source.get("sha256"), f"base registry source {index} SHA-256")
    seen: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ProvenanceError(f"base registry record {index} must be an object")
        _require_exact_keys(record, _BASE_RECORD_KEYS, f"base registry record {index}")
        game_id = _positive_int(record.get("game_id"), f"base record {index} game_id")
        if game_id in seen:
            raise ProvenanceError(f"base registry repeats game {game_id}")
        seen.add(game_id)
        for key in ("categories", "sources"):
            strings = record.get(key)
            if not isinstance(strings, list) or not strings or strings != sorted(set(strings)) or not all(isinstance(item, str) and item for item in strings):
                raise ProvenanceError(f"base registry record {index} has invalid {key}")
        if not set(record["sources"]).issubset(source_paths):
            raise ProvenanceError(f"base registry record {index} references an unknown source")
    if sources != sorted(sources, key=lambda item: (item["path"], item["category"])):
        raise ProvenanceError("base registry sources are not canonically ordered")
    if records != sorted(records, key=lambda item: item["game_id"]):
        raise ProvenanceError("base registry records are not canonically ordered")
    return value


def load_numeric_inventory(path: pathlib.Path) -> set[int]:
    """Read only positive numeric directory names, one per line."""

    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ProvenanceError(f"numeric inventory is not ASCII: {path}") from error
    ids: list[int] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw or re.fullmatch(r"[1-9][0-9]*", raw) is None:
            raise ProvenanceError(f"invalid numeric inventory entry at {path}:{line_number}")
        ids.append(int(raw))
    if not ids or ids != sorted(set(ids)):
        raise ProvenanceError(f"numeric inventory must be non-empty, unique, and sorted: {path}")
    return set(ids)


def load_battle_metadata_snapshot(path: pathlib.Path) -> set[int]:
    """Accept a battle-list response only when it is provably metadata-only."""

    content = path.read_bytes()
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid battle metadata JSON: {path}") from error
    if not isinstance(value, list) or not value:
        raise ProvenanceError("battle metadata snapshot must be a non-empty list")
    game_ids: list[int] = []
    for index, battle in enumerate(value):
        if not isinstance(battle, dict):
            raise ProvenanceError(f"battle metadata item {index} must be an object")
        forbidden = sorted(set(battle) & _FORBIDDEN_METADATA_KEYS)
        if forbidden:
            raise ProvenanceError(f"battle metadata item {index} contains replay fields: {forbidden}")
        _require_exact_keys(battle, _BATTLE_KEYS, f"battle metadata item {index}")
        if battle.get("done") is not True:
            raise ProvenanceError(f"battle metadata item {index} is not complete")
        game_ids.append(_positive_int(battle.get("gameId"), f"battle metadata item {index} gameId"))
        players = battle.get("players")
        if not isinstance(players, list) or len(players) != 2:
            raise ProvenanceError(f"battle metadata item {index} must have two players")
        positions: set[int] = set()
        for player_index, player in enumerate(players):
            if not isinstance(player, dict):
                raise ProvenanceError(f"battle {index} player {player_index} must be an object")
            forbidden = sorted(set(player) & _FORBIDDEN_METADATA_KEYS)
            if forbidden:
                raise ProvenanceError(f"battle {index} player {player_index} contains replay fields: {forbidden}")
            _require_exact_keys(player, _PLAYER_KEYS, f"battle {index} player {player_index}")
            for key in ("playerAgentId", "submissionId", "userId"):
                _positive_int(player.get(key), f"battle {index} player {player_index} {key}")
            if "avatar" in player:
                avatar = player["avatar"]
                if isinstance(avatar, bool) or not isinstance(avatar, int) or avatar < 0:
                    raise ProvenanceError(f"battle {index} player {player_index} has invalid avatar")
            position = player.get("position")
            if isinstance(position, bool) or position not in (0, 1):
                raise ProvenanceError(f"battle {index} player {player_index} has invalid position")
            positions.add(position)
            for key in ("nickname", "publicHandle", "testSessionHandle"):
                if not isinstance(player.get(key), str):
                    raise ProvenanceError(f"battle {index} player {player_index} has invalid {key}")
        if positions != {0, 1}:
            raise ProvenanceError(f"battle metadata item {index} has repeated player positions")
    if len(game_ids) != len(set(game_ids)):
        raise ProvenanceError("battle metadata snapshot repeats a game id")
    return set(game_ids)


def _logical_path(path: pathlib.Path, repository: pathlib.Path | None) -> str:
    resolved = path.resolve()
    if repository is not None:
        try:
            return resolved.relative_to(repository.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def build_exclusion_registry(
    *,
    base_registry_path: pathlib.Path,
    protected_inventory_paths: Sequence[pathlib.Path],
    battle_snapshot_paths: Sequence[pathlib.Path],
    t0_utc: str,
    output_directory: pathlib.Path,
    base_registry_sha256: str | None = None,
    repository: pathlib.Path | None = None,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    """Merge strictly ID-only pre-campaign evidence into one sealed registry."""

    parse_utc(t0_utc, "campaign T0")
    if not protected_inventory_paths:
        raise ProvenanceError("at least one protected numeric inventory is required")
    if not battle_snapshot_paths:
        raise ProvenanceError("at least one T0 battle metadata snapshot is required")
    base = load_id_only_registry(base_registry_path, base_registry_sha256)
    sources = [dict(source) for source in base["sources"]]
    by_game: dict[int, dict[str, set[str]]] = {
        int(record["game_id"]): {
            "categories": set(record["categories"]),
            "sources": set(record["sources"]),
        }
        for record in base["records"]
    }

    def merge_input(path: pathlib.Path, ids: Iterable[int], category: str) -> None:
        game_ids = set(ids)
        logical = _logical_path(path, repository)
        if any(source["path"] == logical for source in sources):
            raise ProvenanceError(f"input source path collides with base registry: {logical}")
        sources.append(
            {
                "category": category,
                "game_id_count": len(game_ids),
                "path": logical,
                "sha256": sha256_file(path),
            }
        )
        for game_id in game_ids:
            entry = by_game.setdefault(game_id, {"categories": set(), "sources": set()})
            entry["categories"].add(category)
            entry["sources"].add(logical)

    for path in sorted({item.resolve() for item in protected_inventory_paths}):
        merge_input(path, load_numeric_inventory(path), "protected_numeric_inventory")
    for path in sorted({item.resolve() for item in battle_snapshot_paths}):
        merge_input(path, load_battle_metadata_snapshot(path), "protected_pre_t0_battle_metadata")

    payload = {
        "schema": EXCLUSION_SCHEMA,
        "selection": (
            f"clean-room {NAMESPACE} boundary frozen at {t0_utc}; canonical prior "
            "ID-only registry plus numeric protected-directory inventories and strict "
            "battle metadata only; no replay content was read"
        ),
        "sources": sorted(sources, key=lambda item: (item["path"], item["category"])),
        "records": [
            {
                "categories": sorted(entry["categories"]),
                "game_id": game_id,
                "sources": sorted(entry["sources"]),
            }
            for game_id, entry in sorted(by_game.items())
        ],
    }
    digest, path = write_content_addressed(output_directory, payload)
    # Re-load through the strict compatibility validator before returning it.
    load_id_only_registry(path, digest)
    return digest, path, payload


def utc_text(instant: dt.datetime) -> str:
    if instant.tzinfo != dt.timezone.utc:
        raise ProvenanceError("timestamp formatting requires UTC")
    return instant.isoformat(timespec="seconds").replace("+00:00", "Z")


def campaign_deadlines(t0_utc: str) -> dict[str, str]:
    t0 = parse_utc(t0_utc, "campaign T0")
    return {
        "arena_freeze_cutoff_utc": utc_text(t0 + dt.timedelta(hours=2, minutes=35)),
        "experimental_upload_stop_utc": utc_text(t0 + dt.timedelta(hours=4, minutes=20)),
        "finalist_upload_deadline_utc": utc_text(t0 + dt.timedelta(hours=3, minutes=45)),
        "finalist_window_deadline_utc": utc_text(t0 + dt.timedelta(hours=4, minutes=15)),
        "goal_end_utc": utc_text(t0 + dt.timedelta(hours=5)),
        "rollback_window_start_utc": utc_text(t0 + dt.timedelta(hours=4, minutes=20)),
        "t0_utc": t0_utc,
    }


def create_window_plan(
    *,
    t0_utc: str,
    planned_at_utc: str,
    collection_windows: int,
    output_directory: pathlib.Path,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    """Pre-assign immutable train/validation roles before arena results exist."""

    t0 = parse_utc(t0_utc, "campaign T0")
    planned_at = parse_utc(planned_at_utc, "plan creation time")
    if planned_at < t0:
        raise ProvenanceError("window plan cannot predate campaign T0")
    if isinstance(collection_windows, bool) or not isinstance(collection_windows, int) or collection_windows < 4:
        raise ProvenanceError("at least four collection windows must be planned")
    deadlines = campaign_deadlines(t0_utc)
    if planned_at > parse_utc(deadlines["arena_freeze_cutoff_utc"]):
        raise ProvenanceError("collection roles must be planned before the arena freeze")
    windows = []
    for ordinal in range(1, collection_windows + 1):
        windows.append(
            {
                "expected_games": EXACT_WINDOW_GAMES,
                "ordinal": ordinal,
                "planned_before_results": True,
                "role": ROLE_BY_ORDINAL[(ordinal - 1) % 4],
                "window_id": f"collection-{ordinal:03d}",
            }
        )
    windows.extend(
        [
            {
                "expected_games": EXACT_WINDOW_GAMES,
                "ordinal": collection_windows + 1,
                "planned_before_results": True,
                "role": "final-holdout",
                "window_id": "finalist-holdout",
            },
            {
                "expected_games": EXACT_WINDOW_GAMES,
                "ordinal": collection_windows + 2,
                "planned_before_results": True,
                "role": "rollback-accounting",
                "window_id": "safe-rollback-accounting",
            },
        ]
    )
    payload = {
        "campaign": {
            **deadlines,
            "expected_games_per_window": EXACT_WINDOW_GAMES,
            "namespace": NAMESPACE,
        },
        "planned_at_utc": planned_at_utc,
        "results_observed_before_assignment": False,
        "schema": WINDOW_PLAN_SCHEMA,
        "windows": windows,
    }
    validate_window_plan(payload)
    digest, path = write_content_addressed(output_directory, payload)
    return digest, path, payload


def validate_window_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ProvenanceError("window plan must be an object")
    _require_exact_keys(
        plan,
        {"campaign", "planned_at_utc", "results_observed_before_assignment", "schema", "windows"},
        "window plan",
    )
    if plan.get("schema") != WINDOW_PLAN_SCHEMA:
        raise ProvenanceError("unexpected window plan schema")
    if plan.get("results_observed_before_assignment") is not False:
        raise ProvenanceError("window roles were not assigned blind")
    campaign = plan.get("campaign")
    expected_campaign_keys = {
        "arena_freeze_cutoff_utc",
        "expected_games_per_window",
        "experimental_upload_stop_utc",
        "finalist_upload_deadline_utc",
        "finalist_window_deadline_utc",
        "goal_end_utc",
        "namespace",
        "rollback_window_start_utc",
        "t0_utc",
    }
    if not isinstance(campaign, dict):
        raise ProvenanceError("window plan campaign must be an object")
    _require_exact_keys(campaign, expected_campaign_keys, "window plan campaign")
    if campaign.get("namespace") != NAMESPACE or campaign.get("expected_games_per_window") != EXACT_WINDOW_GAMES:
        raise ProvenanceError("window plan has the wrong namespace or window size")
    expected_deadlines = campaign_deadlines(campaign.get("t0_utc"))
    if expected_deadlines != {key: campaign[key] for key in expected_deadlines}:
        raise ProvenanceError("window plan deadlines do not match the fixed five-hour schedule")
    planned_at = parse_utc(plan.get("planned_at_utc"), "plan creation time")
    if not parse_utc(campaign["t0_utc"]) <= planned_at <= parse_utc(campaign["arena_freeze_cutoff_utc"]):
        raise ProvenanceError("window plan creation time is outside its allowed interval")
    windows = plan.get("windows")
    if not isinstance(windows, list) or len(windows) < 6:
        raise ProvenanceError("window plan must contain four collection, finalist, and rollback windows")
    seen_ids: set[str] = set()
    collection_count = len(windows) - 2
    for index, window in enumerate(windows, 1):
        if not isinstance(window, dict):
            raise ProvenanceError(f"window {index} must be an object")
        _require_exact_keys(
            window,
            {"expected_games", "ordinal", "planned_before_results", "role", "window_id"},
            f"window {index}",
        )
        window_id = window.get("window_id")
        if not isinstance(window_id, str) or WINDOW_ID_RE.fullmatch(window_id) is None or window_id in seen_ids:
            raise ProvenanceError(f"window {index} has an invalid/repeated id")
        seen_ids.add(window_id)
        if window.get("ordinal") != index or window.get("expected_games") != EXACT_WINDOW_GAMES or window.get("planned_before_results") is not True:
            raise ProvenanceError(f"window {window_id} violates ordinal/size/blind assignment")
        expected_role = (
            ROLE_BY_ORDINAL[(index - 1) % 4]
            if index <= collection_count
            else ("final-holdout" if index == collection_count + 1 else "rollback-accounting")
        )
        expected_id = (
            f"collection-{index:03d}"
            if index <= collection_count
            else ("finalist-holdout" if index == collection_count + 1 else "safe-rollback-accounting")
        )
        if window.get("role") != expected_role or window_id != expected_id:
            raise ProvenanceError(f"window {window_id} violates immutable role assignment")
    return dict(plan)


def window_from_plan(plan: Mapping[str, Any], window_id: str) -> dict[str, Any]:
    validate_window_plan(plan)
    matches = [window for window in plan["windows"] if window["window_id"] == window_id]
    if len(matches) != 1:
        raise ProvenanceError(f"window is absent from the plan: {window_id}")
    return dict(matches[0])


_PREFLIGHT_KEYS = {
    "compilation",
    "legal_action",
    "protocol",
    "purity",
    "source_size",
    "timing_both_colors",
}


def create_editor_attestation(
    *,
    plan_path: pathlib.Path,
    window_id: str,
    source_path: pathlib.Path,
    copied_back_path: pathlib.Path,
    repository_commit: str,
    agent_id: int,
    submission_id: int,
    uploaded_at_utc: str,
    checked_at_utc: str,
    preflight: Mapping[str, Any],
    play_stdout_legal: bool,
    play_telemetry_ok: bool,
    output_directory: pathlib.Path,
    repository: pathlib.Path | None = None,
    expected_plan_sha256: str | None = None,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    """Bind an editor paste/copy-back and Play check to one planned window."""

    plan = load_content_addressed(plan_path, WINDOW_PLAN_SCHEMA)
    plan_hash = sha256_file(plan_path)
    if expected_plan_sha256 is not None and plan_hash != _sha256(expected_plan_sha256, "window plan SHA-256"):
        raise ProvenanceError("window plan SHA-256 is not approved")
    validate_window_plan(plan)
    window = window_from_plan(plan, window_id)
    source = source_path.read_bytes()
    copied = copied_back_path.read_bytes()
    try:
        source_text = source.decode("ascii")
        copied_text = copied.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProvenanceError("generated and copied-back sources must be ASCII") from error
    if source != copied:
        raise ProvenanceError("CodinGame editor copy-back is not byte-identical to generated source")
    if len(source) > ASCII_SOURCE_LIMIT:
        raise ProvenanceError(f"source exceeds the {ASCII_SOURCE_LIMIT:,}-character contract")
    source_hash = sha256_bytes(source)
    if window["role"] == "rollback-accounting" and (
        source_hash != SAFE_ROLLBACK_SOURCE_SHA256 or len(source) != SAFE_ROLLBACK_SOURCE_BYTES
    ):
        raise ProvenanceError("rollback accounting must use the exact frozen safe H62 source")
    if window["role"] != "rollback-accounting" and source_hash == SAFE_ROLLBACK_SOURCE_SHA256:
        raise ProvenanceError("the frozen H62 source cannot occupy an experimental/finalist window")
    if not isinstance(repository_commit, str) or COMMIT_RE.fullmatch(repository_commit) is None:
        raise ProvenanceError("repository commit must be a lowercase Git object id")
    _positive_int(agent_id, "agent_id")
    _positive_int(submission_id, "submission_id")
    uploaded = parse_utc(uploaded_at_utc, "upload time")
    checked = parse_utc(checked_at_utc, "Play check time")
    campaign = plan["campaign"]
    if uploaded < parse_utc(campaign["t0_utc"]):
        raise ProvenanceError("upload predates campaign T0")
    if checked > uploaded:
        raise ProvenanceError("Play check must precede or equal the upload time")
    role = window["role"]
    role_deadline = (
        campaign["arena_freeze_cutoff_utc"]
        if role in {"training", "arena-validation"}
        else campaign["finalist_upload_deadline_utc"]
        if role == "final-holdout"
        else campaign["goal_end_utc"]
    )
    role_start = (
        campaign["arena_freeze_cutoff_utc"]
        if role == "final-holdout"
        else campaign["t0_utc"]
    )
    if uploaded < parse_utc(role_start) or uploaded > parse_utc(role_deadline):
        raise ProvenanceError(f"{role} upload missed its campaign deadline")
    if set(preflight) != _PREFLIGHT_KEYS or any(preflight[key] is not True for key in _PREFLIGHT_KEYS):
        raise ProvenanceError("all required pre-submission gates must be explicitly true")
    if play_stdout_legal is not True or play_telemetry_ok is not True:
        raise ProvenanceError("Play My Code did not attest legal stdout and expected telemetry")
    archived_path = output_directory / "source_payloads" / f"{source_hash}.source"
    _write_once(archived_path, source)
    payload = {
        "created_at_utc": uploaded_at_utc,
        "editor_copyback": {
            "api_readable": False,
            "byte_equal_to_generated": True,
            "bytes": len(copied),
            "characters": len(copied_text),
            "path": _logical_path(copied_back_path, repository),
            "sha256": source_hash,
            "status": "editor-attested-not-api-readable",
        },
        "identity": {
            "agent_id": agent_id,
            "repository_commit": repository_commit,
            "submission_id": submission_id,
        },
        "play_my_code": {
            "checked_at_utc": checked_at_utc,
            "expected_telemetry": True,
            "legal_stdout": True,
        },
        "preflight": dict(sorted(preflight.items())),
        "schema": EDITOR_ATTESTATION_SCHEMA,
        "source": {
            "archived_path": _logical_path(archived_path, repository),
            "ascii": True,
            "bytes": len(source),
            "characters": len(source_text),
            "generated_path": _logical_path(source_path, repository),
            "sha256": source_hash,
        },
        "uploaded_at_utc": uploaded_at_utc,
        "window": window,
        "window_plan": {"path": _logical_path(plan_path, repository), "sha256": plan_hash},
    }
    validate_editor_attestation(payload, plan)
    digest, path = write_content_addressed(output_directory / "attestations" / window_id, payload)
    return digest, path, payload


def validate_editor_attestation(attestation: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_window_plan(plan)
    if not isinstance(attestation, dict) or attestation.get("schema") != EDITOR_ATTESTATION_SCHEMA:
        raise ProvenanceError("unexpected editor attestation schema")
    required = {
        "created_at_utc", "editor_copyback", "identity", "play_my_code", "preflight",
        "schema", "source", "uploaded_at_utc", "window", "window_plan",
    }
    _require_exact_keys(attestation, required, "editor attestation")
    window = attestation.get("window")
    if not isinstance(window, dict) or window_from_plan(plan, window.get("window_id")) != window:
        raise ProvenanceError("editor attestation window contradicts its plan")
    identity = attestation.get("identity")
    if not isinstance(identity, dict) or set(identity) != {"agent_id", "repository_commit", "submission_id"}:
        raise ProvenanceError("editor attestation identity is invalid")
    _positive_int(identity.get("agent_id"), "agent_id")
    _positive_int(identity.get("submission_id"), "submission_id")
    if not isinstance(identity.get("repository_commit"), str) or COMMIT_RE.fullmatch(identity["repository_commit"]) is None:
        raise ProvenanceError("editor attestation commit is invalid")
    source = attestation.get("source")
    copyback = attestation.get("editor_copyback")
    if not isinstance(source, dict) or not isinstance(copyback, dict):
        raise ProvenanceError("editor attestation source bindings are invalid")
    _require_exact_keys(
        source,
        {"archived_path", "ascii", "bytes", "characters", "generated_path", "sha256"},
        "editor attestation source",
    )
    _require_exact_keys(
        copyback,
        {"api_readable", "byte_equal_to_generated", "bytes", "characters", "path", "sha256", "status"},
        "editor copy-back",
    )
    source_hash = _sha256(source.get("sha256"), "source SHA-256")
    if (
        source.get("ascii") is not True
        or source.get("bytes") != source.get("characters")
        or isinstance(source.get("bytes"), bool)
        or not isinstance(source.get("bytes"), int)
        or not 0 < source["bytes"] <= ASCII_SOURCE_LIMIT
        or copyback.get("sha256") != source_hash
        or copyback.get("bytes") != source["bytes"]
        or copyback.get("characters") != source["characters"]
        or copyback.get("byte_equal_to_generated") is not True
        or copyback.get("api_readable") is not False
        or copyback.get("status") != "editor-attested-not-api-readable"
    ):
        raise ProvenanceError("editor copy-back does not exactly attest the generated source")
    if window["role"] == "rollback-accounting" and (
        source_hash != SAFE_ROLLBACK_SOURCE_SHA256 or source["bytes"] != SAFE_ROLLBACK_SOURCE_BYTES
    ):
        raise ProvenanceError("rollback attestation is not the exact frozen safe H62 source")
    if window["role"] != "rollback-accounting" and source_hash == SAFE_ROLLBACK_SOURCE_SHA256:
        raise ProvenanceError("frozen H62 is restricted to rollback accounting")
    if not isinstance(attestation.get("preflight"), dict) or set(attestation["preflight"]) != _PREFLIGHT_KEYS or any(value is not True for value in attestation["preflight"].values()):
        raise ProvenanceError("editor attestation preflight is incomplete")
    play = attestation.get("play_my_code")
    if not isinstance(play, dict) or play.get("legal_stdout") is not True or play.get("expected_telemetry") is not True:
        raise ProvenanceError("editor attestation lacks a successful Play check")
    _require_exact_keys(play, {"checked_at_utc", "expected_telemetry", "legal_stdout"}, "Play check")
    plan_ref = attestation.get("window_plan")
    if not isinstance(plan_ref, dict):
        raise ProvenanceError("editor attestation window-plan binding is invalid")
    _require_exact_keys(plan_ref, {"path", "sha256"}, "editor window-plan binding")
    _sha256(plan_ref.get("sha256"), "editor window-plan SHA-256")
    uploaded = parse_utc(attestation.get("uploaded_at_utc"), "upload time")
    checked = parse_utc(play.get("checked_at_utc"), "Play check time")
    campaign = plan["campaign"]
    role = window["role"]
    role_deadline = (
        campaign["arena_freeze_cutoff_utc"]
        if role in {"training", "arena-validation"}
        else campaign["finalist_upload_deadline_utc"]
        if role == "final-holdout"
        else campaign["goal_end_utc"]
    )
    role_start = (
        campaign["arena_freeze_cutoff_utc"]
        if role == "final-holdout"
        else campaign["t0_utc"]
    )
    if not parse_utc(campaign["t0_utc"]) <= checked <= uploaded <= parse_utc(role_deadline) or uploaded < parse_utc(role_start):
        raise ProvenanceError("editor attestation violates T0/order/upload cutoff")
    if attestation.get("created_at_utc") != attestation.get("uploaded_at_utc"):
        raise ProvenanceError("editor attestation timestamps are contradictory")
    return dict(attestation)


def _resolve_path(raw: Any, repository: pathlib.Path) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise ProvenanceError("artifact path is missing")
    path = pathlib.Path(raw)
    return path if path.is_absolute() else repository / path


def _verify_file_ref(path: pathlib.Path, expected_sha256: Any, field: str) -> bytes:
    expected = _sha256(expected_sha256, f"{field} SHA-256")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ProvenanceError(f"cannot read {field}: {path}") from error
    if sha256_bytes(content) != expected:
        raise ProvenanceError(f"{field} SHA-256 mismatch: {path}")
    return content


def _window_deadline(plan: Mapping[str, Any], role: str) -> dt.datetime:
    campaign = plan["campaign"]
    if role in {"training", "arena-validation"}:
        return parse_utc(campaign["arena_freeze_cutoff_utc"])
    if role == "final-holdout":
        return parse_utc(campaign["finalist_window_deadline_utc"])
    return parse_utc(campaign["goal_end_utc"])


def derive_arena_window(
    *,
    plan_path: pathlib.Path,
    attestation_path: pathlib.Path,
    arena_manifest_path: pathlib.Path,
    exclusion_registry_path: pathlib.Path,
    output_directory: pathlib.Path,
    repository: pathlib.Path,
    expected_plan_sha256: str | None = None,
    expected_attestation_sha256: str | None = None,
    expected_exclusion_sha256: str | None = None,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    """Create an immutable, source-bound eligibility view of one 90-game window."""

    repository = repository.resolve()
    plan = load_content_addressed(plan_path, WINDOW_PLAN_SCHEMA)
    validate_window_plan(plan)
    plan_hash = sha256_file(plan_path)
    if expected_plan_sha256 is not None and plan_hash != _sha256(expected_plan_sha256, "plan SHA-256"):
        raise ProvenanceError("arena derivation plan is not approved")
    attestation = load_content_addressed(attestation_path, EDITOR_ATTESTATION_SCHEMA)
    attestation_hash = sha256_file(attestation_path)
    if expected_attestation_sha256 is not None and attestation_hash != _sha256(expected_attestation_sha256, "attestation SHA-256"):
        raise ProvenanceError("arena derivation attestation is not approved")
    validate_editor_attestation(attestation, plan)
    if attestation["window_plan"]["sha256"] != plan_hash:
        raise ProvenanceError("attestation points to a different window plan")
    registry = load_id_only_registry(exclusion_registry_path, expected_exclusion_sha256)
    registry_hash = sha256_file(exclusion_registry_path)
    forbidden_ids = {int(record["game_id"]) for record in registry["records"]}
    manifest = load_content_addressed(arena_manifest_path, ARENA_BATCH_SCHEMA)
    manifest_hash = sha256_file(arena_manifest_path)
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise ProvenanceError("arena manifest omits its source binding")
    identity = attestation["identity"]
    source = attestation["source"]
    manifest_source = binding.get("source")
    if (
        binding.get("schema") != ARENA_BINDING_SCHEMA
        or
        binding.get("agent_id") != identity["agent_id"]
        or binding.get("asserted_submission_id") != identity["submission_id"]
        or binding.get("repository_commit") != identity["repository_commit"]
        or not isinstance(manifest_source, dict)
        or manifest_source.get("sha256") != source["sha256"]
        or manifest_source.get("bytes") != source["bytes"]
        or manifest_source.get("characters") != source["characters"]
    ):
        raise ProvenanceError("arena manifest contradicts editor-attested source identity")
    collector_hash = _sha256(manifest.get("collector_sha256"), "arena collector SHA-256")
    if (
        binding.get("collector_sha256") != collector_hash
        or not isinstance(manifest.get("run_id"), str)
        or binding.get("run_id") != manifest.get("run_id")
    ):
        raise ProvenanceError("arena manifest binding contradicts its collector/run identity")
    archived_source_path = _resolve_path(manifest_source.get("archived_path"), repository)
    archived_source = _verify_file_ref(archived_source_path, source["sha256"], "archived source")
    try:
        archived_source.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProvenanceError("arena manifest source is not ASCII") from error
    manifest_exclusion = manifest.get("exclusion_registry")
    if not isinstance(manifest_exclusion, dict) or manifest_exclusion.get("sha256") != registry_hash:
        raise ProvenanceError("arena manifest was not collected against the approved exclusions")
    coverage = manifest.get("coverage")
    games = manifest.get("games")
    window = attestation["window"]
    window_snapshot = manifest.get("window_snapshot")
    if not isinstance(window_snapshot, dict) or set(window_snapshot) != {"normalized_sha256", "raw_sha256"}:
        raise ProvenanceError("arena manifest omits its immutable battle-window snapshot")
    window_snapshot = {
        "normalized_sha256": _sha256(window_snapshot.get("normalized_sha256"), "battle-window normalized payload"),
        "raw_sha256": _sha256(window_snapshot.get("raw_sha256"), "battle-window raw payload"),
    }
    if (
        not isinstance(coverage, dict)
        or coverage.get("expected_games") != EXACT_WINDOW_GAMES
        or coverage.get("battle_window_games") != EXACT_WINDOW_GAMES
        or coverage.get("full_window_accounted") is not True
        or not isinstance(games, list)
        or len(games) != EXACT_WINDOW_GAMES
        or window.get("expected_games") != EXACT_WINDOW_GAMES
    ):
        raise ProvenanceError("arena manifest is not an exact, fully accounted 90-game window")
    uploaded = parse_utc(attestation["uploaded_at_utc"], "upload time")
    started = parse_utc(manifest.get("started_at_utc"), "collection start")
    completed = parse_utc(manifest.get("completed_at_utc"), "collection completion")
    deadline = _window_deadline(plan, window["role"])
    if not uploaded <= started <= completed <= deadline:
        raise ProvenanceError("arena window timestamps violate upload/order/freeze cutoffs")

    derived_games = []
    seen_ids: set[int] = set()
    actual_status_counts: dict[str, int] = {}
    for stored_index, stored in enumerate(games):
        if not isinstance(stored, dict) or not isinstance(stored.get("record"), dict):
            raise ProvenanceError(f"arena game binding {stored_index} is invalid")
        record = stored["record"]
        record_hash = _sha256(stored.get("record_sha256"), f"game {stored_index} record")
        if sha256_bytes(canonical_json_bytes(record)) != record_hash:
            raise ProvenanceError(f"arena game {stored_index} embedded record hash mismatch")
        record_path = _resolve_path(stored.get("record_path"), repository)
        if _verify_file_ref(record_path, record_hash, f"game {stored_index} record") != canonical_json_bytes(record):
            raise ProvenanceError(f"arena game {stored_index} record is not canonical")
        game_id = _positive_int(record.get("game_id"), f"game {stored_index} id")
        if game_id in seen_ids:
            raise ProvenanceError(f"arena window repeats game {game_id}")
        seen_ids.add(game_id)
        if game_id in forbidden_ids:
            raise ProvenanceError(f"arena window contains pre-T0/protected game {game_id}")
        focus = record.get("focus")
        if (
            record.get("schema") != ARENA_GAME_SCHEMA
            or record.get("source_sha256") != source["sha256"]
            or not isinstance(focus, dict)
            or focus.get("agent_id") != identity["agent_id"]
            or focus.get("submission_id") != identity["submission_id"]
        ):
            raise ProvenanceError(f"game {game_id} contradicts exact source/submission binding")
        status = record.get("status")
        if not isinstance(status, str):
            raise ProvenanceError(f"game {game_id} has an invalid status")
        actual_status_counts[status] = actual_status_counts.get(status, 0) + 1
        if status not in {"accepted", "structural-rejection"}:
            raise ProvenanceError(f"game {game_id} prevents exact completion: {status}")
        acquisition = record.get("acquisition")
        if not isinstance(acquisition, dict):
            raise ProvenanceError(f"game {game_id} omits immutable raw/normalized acquisition")
        raw_hash = _sha256(acquisition.get("raw_sha256"), f"game {game_id} raw payload")
        normalized_hash = _sha256(acquisition.get("normalized_sha256"), f"game {game_id} normalized payload")
        raw_path = _resolve_path(acquisition.get("raw_path"), repository)
        normalized_path = _resolve_path(acquisition.get("normalized_path"), repository)
        _verify_file_ref(raw_path, raw_hash, f"game {game_id} raw payload")
        normalized_content = _verify_file_ref(normalized_path, normalized_hash, f"game {game_id} normalized payload")
        try:
            normalized_value = json.loads(normalized_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProvenanceError(f"game {game_id} normalized payload is not JSON") from error
        if canonical_json_bytes(normalized_value) != normalized_content:
            raise ProvenanceError(f"game {game_id} normalized payload is not canonical")
        fetched = parse_utc(acquisition.get("fetched_at_utc"), f"game {game_id} fetch time")
        if not uploaded <= fetched <= completed or fetched > deadline:
            raise ProvenanceError(f"game {game_id} fetch time violates campaign cutoff")
        replay_payload = acquisition.get("replay_payload_path")
        if status == "accepted":
            replay_path = _resolve_path(replay_payload, repository)
            if _verify_file_ref(replay_path, normalized_hash, f"game {game_id} replay payload") != normalized_content:
                raise ProvenanceError(f"game {game_id} replay payload is not the normalized response")

        rejection_reasons: list[str] = []
        operational = record.get("operational") or {}
        replay = record.get("replay") or {}
        rules = replay.get("rules_validation") or {}
        outcome = record.get("outcome") or {}
        if status != "accepted":
            rejection_reasons.append("malformed-or-structurally-rejected-transcript")
        if operational.get("classification") != "clean":
            rejection_reasons.append("operationally-terminated")
        if operational.get("focus_status") != "ok":
            rejection_reasons.append("focus-operational-failure")
        if operational.get("opponent_status") != "ok":
            rejection_reasons.append("opponent-operational-failure")
        if operational.get("signals") or operational.get("unscoped_signals"):
            rejection_reasons.append("operational-failure-signal")
        if rules.get("status") != "terminal-valid":
            rejection_reasons.append("not-unambiguous-rule-terminal")
        if replay.get("observed_transcript") != replay.get("valid_transcript"):
            rejection_reasons.append("partial-or-invalid-transcript")
        if outcome.get("winner_player_id") not in (0, 1):
            rejection_reasons.append("ambiguous-terminal-outcome")
        rejection_reasons = sorted(set(rejection_reasons))
        opponent = record.get("opponent") or {}
        opponent_rank = opponent.get("frozen_rank")
        uses: list[str] = []
        ranking_weight = 0.0
        if not rejection_reasons:
            if window["role"] == "training":
                uses.append("raw-terminal-value-candidate")
                if isinstance(opponent_rank, int) and not isinstance(opponent_rank, bool) and 1 <= opponent_rank <= 50:
                    uses.append("opponent-action-ranking-reanalysis-candidate")
                    ranking_weight = 1.0 if opponent_rank <= 20 else 0.5
            elif window["role"] == "arena-validation":
                uses.append("whole-game-arena-validation-only")
            elif window["role"] == "final-holdout":
                uses.append("untouched-live-final-holdout-only")
            else:
                uses.append("rollback-accounting-only")
        derived_games.append(
            {
                "disposition": "eligible" if not rejection_reasons else "rejected-entire-game",
                "game_id": game_id,
                "normalized_sha256": normalized_hash,
                "opponent_frozen_rank": opponent_rank,
                "ranking_candidate_weight": ranking_weight,
                "raw_sha256": raw_hash,
                "record_sha256": record_hash,
                "rejection_reasons": rejection_reasons,
                "uses": uses,
            }
        )
    derived_games.sort(key=lambda item: item["game_id"])
    if coverage.get("accepted_games") != actual_status_counts.get("accepted", 0) or coverage.get("status_counts") != dict(sorted(actual_status_counts.items())):
        raise ProvenanceError("arena manifest coverage counts contradict its game records")
    eligible = [item for item in derived_games if item["disposition"] == "eligible"]
    payload = {
        "arena_manifest": {"path": _logical_path(arena_manifest_path, repository), "sha256": manifest_hash},
        "battle_window_snapshot": window_snapshot,
        "campaign": {"namespace": NAMESPACE, "t0_utc": plan["campaign"]["t0_utc"]},
        "cutoff": {"deadline_utc": utc_text(deadline), "status": "passed"},
        "editor_attestation": {"path": _logical_path(attestation_path, repository), "sha256": attestation_hash},
        "exclusion_registry": {"path": _logical_path(exclusion_registry_path, repository), "sha256": registry_hash},
        "games": derived_games,
        "schema": ARENA_DERIVATION_SCHEMA,
        "source": {
            "agent_id": identity["agent_id"],
            "repository_commit": identity["repository_commit"],
            "sha256": source["sha256"],
            "submission_id": identity["submission_id"],
        },
        "summary": {
            "eligible_games": len(eligible),
            "exact_window_games": len(derived_games),
            "focus_operational_failures": sum("focus-operational-failure" in item["rejection_reasons"] for item in derived_games),
            "rejected_games": len(derived_games) - len(eligible),
            "role": window["role"],
            "source_binding": "editor-attested-not-api-readable",
            "window_id": window["window_id"],
        },
        "timing": {
            "collection_completed_at_utc": manifest["completed_at_utc"],
            "collection_started_at_utc": manifest["started_at_utc"],
            "uploaded_at_utc": attestation["uploaded_at_utc"],
        },
        "window": window,
        "window_plan": {"path": _logical_path(plan_path, repository), "sha256": plan_hash},
    }
    digest, path = write_content_addressed(output_directory / window["window_id"], payload)
    validate_arena_derivation(path, repository=repository, expected_sha256=digest)
    return digest, path, payload


def validate_arena_derivation(
    derivation_path: pathlib.Path,
    *,
    repository: pathlib.Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable accounting structure of a frozen derivation.

    Structural validity is deliberately distinct from permission to train on
    the window.  Corpus consumers must additionally apply
    :func:`arena_derivation_usage`; in particular, one focus operational
    failure rejects every otherwise eligible candidate in that submission
    window.
    """

    derivation = load_content_addressed(derivation_path, ARENA_DERIVATION_SCHEMA)
    derivation_hash = sha256_file(derivation_path)
    if expected_sha256 is not None and derivation_hash != _sha256(expected_sha256, "derivation SHA-256"):
        raise ProvenanceError("arena derivation SHA-256 is not approved")
    _require_exact_keys(
        derivation,
        {
            "arena_manifest", "campaign", "cutoff", "editor_attestation",
            "battle_window_snapshot", "exclusion_registry", "games", "schema", "source", "summary",
            "timing", "window", "window_plan",
        },
        "arena derivation",
    )
    for name in ("arena_manifest", "editor_attestation", "exclusion_registry", "window_plan"):
        reference = derivation.get(name)
        if not isinstance(reference, dict):
            raise ProvenanceError(f"arena derivation {name} reference is invalid")
        _require_exact_keys(reference, {"path", "sha256"}, f"arena derivation {name}")
        _verify_file_ref(
            _resolve_path(reference.get("path"), repository),
            reference.get("sha256"),
            f"arena derivation {name}",
        )
    plan_path = _resolve_path(derivation["window_plan"]["path"], repository)
    plan = load_content_addressed(plan_path, WINDOW_PLAN_SCHEMA)
    validate_window_plan(plan)
    attestation_path = _resolve_path(derivation["editor_attestation"]["path"], repository)
    attestation = load_content_addressed(attestation_path, EDITOR_ATTESTATION_SCHEMA)
    validate_editor_attestation(attestation, plan)
    registry_path = _resolve_path(derivation["exclusion_registry"]["path"], repository)
    registry = load_id_only_registry(registry_path, derivation["exclusion_registry"]["sha256"])
    forbidden = {record["game_id"] for record in registry["records"]}
    window = derivation.get("window")
    if not isinstance(window, dict) or window_from_plan(plan, window.get("window_id")) != window or attestation["window"] != window:
        raise ProvenanceError("arena derivation window contradicts its plan/attestation")
    campaign = derivation.get("campaign")
    if campaign != {"namespace": NAMESPACE, "t0_utc": plan["campaign"]["t0_utc"]}:
        raise ProvenanceError("arena derivation campaign identity is invalid")
    source = derivation.get("source")
    identity = attestation["identity"]
    if source != {
        "agent_id": identity["agent_id"],
        "repository_commit": identity["repository_commit"],
        "sha256": attestation["source"]["sha256"],
        "submission_id": identity["submission_id"],
    }:
        raise ProvenanceError("arena derivation source contradicts its attestation")
    cutoff = derivation.get("cutoff")
    expected_deadline = utc_text(_window_deadline(plan, window["role"]))
    if cutoff != {"deadline_utc": expected_deadline, "status": "passed"}:
        raise ProvenanceError("arena derivation cutoff is invalid")
    manifest_path = _resolve_path(derivation["arena_manifest"]["path"], repository)
    manifest = load_content_addressed(manifest_path, ARENA_BATCH_SCHEMA)
    if derivation.get("battle_window_snapshot") != manifest.get("window_snapshot"):
        raise ProvenanceError("arena derivation battle-window snapshot contradicts its manifest")
    timing = derivation.get("timing")
    expected_timing = {
        "collection_completed_at_utc": manifest.get("completed_at_utc"),
        "collection_started_at_utc": manifest.get("started_at_utc"),
        "uploaded_at_utc": attestation.get("uploaded_at_utc"),
    }
    if timing != expected_timing:
        raise ProvenanceError("arena derivation timing contradicts its manifest/attestation")
    uploaded = parse_utc(timing["uploaded_at_utc"], "derived upload time")
    started = parse_utc(timing["collection_started_at_utc"], "derived collection start")
    completed = parse_utc(timing["collection_completed_at_utc"], "derived collection completion")
    if not uploaded <= started <= completed <= _window_deadline(plan, window["role"]):
        raise ProvenanceError("arena derivation timing violates campaign order/cutoff")
    games = derivation.get("games")
    if not isinstance(games, list) or len(games) != EXACT_WINDOW_GAMES:
        raise ProvenanceError("arena derivation does not contain exactly 90 games")
    seen: set[int] = set()
    eligible = 0
    focus_failures = 0
    previous_id = 0
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ProvenanceError(f"derived game {index} must be an object")
        _require_exact_keys(
            game,
            {
                "disposition", "game_id", "normalized_sha256", "opponent_frozen_rank",
                "ranking_candidate_weight", "raw_sha256", "record_sha256",
                "rejection_reasons", "uses",
            },
            f"derived game {index}",
        )
        game_id = _positive_int(game.get("game_id"), f"derived game {index} id")
        if game_id in seen or game_id <= previous_id or game_id in forbidden:
            raise ProvenanceError(f"derived game {game_id} is repeated, unsorted, or excluded")
        seen.add(game_id)
        previous_id = game_id
        for key in ("normalized_sha256", "raw_sha256", "record_sha256"):
            _sha256(game.get(key), f"derived game {game_id} {key}")
        reasons = game.get("rejection_reasons")
        uses = game.get("uses")
        if (
            not isinstance(reasons, list)
            or reasons != sorted(set(reasons))
            or not all(isinstance(reason, str) and reason for reason in reasons)
            or not isinstance(uses, list)
            or len(uses) != len(set(uses))
            or not all(isinstance(use, str) and use for use in uses)
        ):
            raise ProvenanceError(f"derived game {game_id} has invalid reasons/uses")
        rank = game.get("opponent_frozen_rank")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
            raise ProvenanceError(f"derived game {game_id} has invalid opponent rank")
        expected_uses: list[str] = []
        expected_weight = 0.0
        if not reasons:
            eligible += 1
            if window["role"] == "training":
                expected_uses.append("raw-terminal-value-candidate")
                if rank is not None and rank <= 50:
                    expected_uses.append("opponent-action-ranking-reanalysis-candidate")
                    expected_weight = 1.0 if rank <= 20 else 0.5
            elif window["role"] == "arena-validation":
                expected_uses.append("whole-game-arena-validation-only")
            elif window["role"] == "final-holdout":
                expected_uses.append("untouched-live-final-holdout-only")
            else:
                expected_uses.append("rollback-accounting-only")
        if (
            game.get("disposition") != ("eligible" if not reasons else "rejected-entire-game")
            or uses != expected_uses
            or game.get("ranking_candidate_weight") != expected_weight
        ):
            raise ProvenanceError(f"derived game {game_id} has role-inconsistent disposition/uses")
        focus_failures += "focus-operational-failure" in reasons
    summary = derivation.get("summary")
    expected_summary = {
        "eligible_games": eligible,
        "exact_window_games": EXACT_WINDOW_GAMES,
        "focus_operational_failures": focus_failures,
        "rejected_games": EXACT_WINDOW_GAMES - eligible,
        "role": window["role"],
        "source_binding": "editor-attested-not-api-readable",
        "window_id": window["window_id"],
    }
    if summary != expected_summary:
        raise ProvenanceError("arena derivation summary contradicts its games")
    return derivation


def arena_derivation_usage(derivation: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit whole-window usage decision for a valid derivation.

    This helper does not replace ``validate_arena_derivation``.  Callers first
    validate the content-addressed artifact, then use this result to avoid
    confusing per-game ``eligible`` accounting candidates with authorization
    to include them in training.
    """

    window = derivation.get("window")
    summary = derivation.get("summary")
    if not isinstance(window, Mapping) or not isinstance(summary, Mapping):
        raise ProvenanceError("arena derivation must be structurally validated before usage classification")
    role = window.get("role")
    focus_failures = summary.get("focus_operational_failures")
    eligible_games = summary.get("eligible_games")
    if (
        role not in {"training", "arena-validation", "final-holdout", "rollback-accounting"}
        or isinstance(focus_failures, bool)
        or not isinstance(focus_failures, int)
        or focus_failures < 0
        or isinstance(eligible_games, bool)
        or not isinstance(eligible_games, int)
        or eligible_games < 0
    ):
        raise ProvenanceError("arena derivation has no valid usage classification")
    if focus_failures:
        return {
            "reason": "focus-operational-failure-rejects-entire-window",
            "training_eligible": False,
            "window_disposition": "rejected-entire-window",
        }
    if role == "training" and eligible_games:
        return {
            "reason": "clean-preassigned-training-window",
            "training_eligible": True,
            "window_disposition": "training-candidate",
        }
    disposition = {
        "training": "no-eligible-training-games",
        "arena-validation": "arena-validation-only",
        "final-holdout": "untouched-live-final-holdout-only",
        "rollback-accounting": "rollback-accounting-only",
    }[role]
    return {
        "reason": "window-role-or-empty-window-forbids-training",
        "training_eligible": False,
        "window_disposition": disposition,
    }


def validate_campaign_sequence(
    derivation_paths: Sequence[pathlib.Path],
    *,
    repository: pathlib.Path,
    plan_path: pathlib.Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Prove no upload superseded an incomplete window and roles stayed blind."""

    if not derivation_paths:
        raise ProvenanceError("campaign sequence requires at least one derivation")
    plan = load_content_addressed(plan_path, WINDOW_PLAN_SCHEMA)
    validate_window_plan(plan)
    plan_hash = sha256_file(plan_path)
    if plan_hash != _sha256(expected_plan_sha256, "campaign plan SHA-256"):
        raise ProvenanceError("campaign sequence uses an unapproved window plan")
    derivations = [
        validate_arena_derivation(path, repository=repository)
        for path in derivation_paths
    ]
    if any(item["window_plan"]["sha256"] != plan_hash for item in derivations):
        raise ProvenanceError("campaign derivations do not share the approved plan")
    by_upload = sorted(
        derivations,
        key=lambda item: parse_utc(item["timing"]["uploaded_at_utc"]),
    )
    window_ids = [item["window"]["window_id"] for item in by_upload]
    if len(window_ids) != len(set(window_ids)):
        raise ProvenanceError("campaign sequence repeats a planned window")
    source_hashes = [item["source"]["sha256"] for item in by_upload]
    if len(source_hashes) != len(set(source_hashes)):
        raise ProvenanceError("campaign submissions are not source-distinct")
    collections = [
        item["window"]["ordinal"]
        for item in by_upload
        if item["window"]["window_id"].startswith("collection-")
    ]
    if collections != list(range(1, len(collections) + 1)):
        raise ProvenanceError("collection windows were skipped or used out of order")
    phase = "collection"
    previous_completed: dt.datetime | None = None
    for derivation in by_upload:
        role = derivation["window"]["role"]
        uploaded = parse_utc(derivation["timing"]["uploaded_at_utc"])
        completed = parse_utc(derivation["timing"]["collection_completed_at_utc"])
        if previous_completed is not None and uploaded < previous_completed:
            raise ProvenanceError(
                f"window {derivation['window']['window_id']} was uploaded before its predecessor completed"
            )
        if role in {"training", "arena-validation"}:
            if phase != "collection":
                raise ProvenanceError("experimental collection continued after finalist/rollback")
        elif role == "final-holdout":
            if phase != "collection":
                raise ProvenanceError("campaign sequence repeats or misorders the finalist")
            phase = "finalist"
        elif role == "rollback-accounting":
            if phase == "finalist":
                if uploaded < parse_utc(plan["campaign"]["rollback_window_start_utc"]):
                    raise ProvenanceError("scheduled final rollback predates its decision cutoff")
            elif phase != "collection":
                raise ProvenanceError("campaign sequence repeats or misorders rollback accounting")
            # A rollback reached directly from collection is the one permitted
            # emergency path.  Its exact frozen H62 bytes were already checked
            # by the attestation validator, and no experiment may follow it.
            phase = "rollback"
        previous_completed = completed
    return {
        "collection_windows": len(collections),
        "finalist_windows": sum(item["window"]["role"] == "final-holdout" for item in by_upload),
        "last_completed_at_utc": by_upload[-1]["timing"]["collection_completed_at_utc"],
        "rollback_windows": sum(item["window"]["role"] == "rollback-accounting" for item in by_upload),
        "status": "sequential-and-complete",
        "windows": window_ids,
    }


def _safe_protected_roots(
    paths: Sequence[pathlib.Path],
    output_directory: pathlib.Path | None,
    repository: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    if not paths:
        raise ProvenanceError("at least one explicit protected path is required")
    resolved = sorted({path.resolve() for path in paths}, key=lambda path: path.as_posix())
    home = pathlib.Path.home().resolve()
    broad_roots = {pathlib.Path("/").resolve(), home}
    if repository is not None:
        broad_roots.add(repository.resolve())
    for path in resolved:
        if path in broad_roots:
            raise ProvenanceError(f"refusing dangerously broad protected root: {path}")
        if not path.exists() and not path.is_symlink():
            raise ProvenanceError(f"protected path does not exist: {path}")
    for index, path in enumerate(resolved):
        for other in resolved[index + 1 :]:
            if path in other.parents or other in path.parents:
                raise ProvenanceError(f"protected roots overlap: {path} and {other}")
    if output_directory is not None:
        output = output_directory.resolve()
        if any(path == output or path in output.parents or output in path.parents for path in resolved):
            raise ProvenanceError("protected snapshot output must not overlap a protected root")
    return resolved


def _scan_protected_root(root: pathlib.Path) -> list[dict[str, Any]]:
    candidates = [root]
    if root.is_dir() and not root.is_symlink():
        candidates.extend(sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()))
    entries = []
    for path in candidates:
        relative = "." if path == root else path.relative_to(root).as_posix()
        stat = path.lstat()
        common = {"mode": stat.st_mode & 0o7777, "path": relative}
        if path.is_symlink():
            target = os.readlink(path)
            entries.append({**common, "kind": "symlink", "target": target})
        elif path.is_dir():
            entries.append({**common, "kind": "directory"})
        elif path.is_file():
            entries.append({**common, "kind": "file", "sha256": sha256_file(path), "size": stat.st_size})
        else:
            raise ProvenanceError(f"unsupported protected filesystem object: {path}")
    return entries


def create_protected_snapshot(
    *,
    protected_paths: Sequence[pathlib.Path],
    label: str,
    created_at_utc: str,
    output_directory: pathlib.Path,
    repository: pathlib.Path | None = None,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    """Hash protected bytes/topology without parsing or modifying any content."""

    parse_utc(created_at_utc, "protected snapshot creation time")
    if not isinstance(label, str) or not label.strip():
        raise ProvenanceError("protected snapshot label must be non-empty")
    roots = _safe_protected_roots(protected_paths, output_directory, repository)
    payload = {
        "created_at_utc": created_at_utc,
        "label": label,
        "roots": [
            {"entries": _scan_protected_root(root), "path": _logical_path(root, repository)}
            for root in roots
        ],
        "schema": PROTECTED_SNAPSHOT_SCHEMA,
    }
    digest, path = write_content_addressed(output_directory, payload)
    return digest, path, payload


def verify_protected_snapshot(snapshot_path: pathlib.Path, *, repository: pathlib.Path) -> dict[str, Any]:
    snapshot = load_content_addressed(snapshot_path, PROTECTED_SNAPSHOT_SCHEMA)
    roots = snapshot.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ProvenanceError("protected snapshot has no roots")
    root_paths = [_resolve_path(stored.get("path"), repository) for stored in roots if isinstance(stored, dict)]
    if len(root_paths) != len(roots):
        raise ProvenanceError("protected snapshot root binding is invalid")
    _safe_protected_roots(root_paths, None, repository)
    mismatches = []
    for stored in roots:
        if not isinstance(stored, dict) or set(stored) != {"entries", "path"}:
            raise ProvenanceError("protected snapshot root binding is invalid")
        root = _resolve_path(stored["path"], repository)
        try:
            actual = _scan_protected_root(root)
        except (OSError, ProvenanceError) as error:
            mismatches.append({"path": stored["path"], "reason": str(error)})
            continue
        if actual != stored.get("entries"):
            mismatches.append(
                {
                    "actual_sha256": sha256_bytes(canonical_json_bytes(actual)),
                    "expected_sha256": sha256_bytes(canonical_json_bytes(stored.get("entries"))),
                    "path": stored["path"],
                    "reason": "content/topology/mode changed",
                }
            )
    if mismatches:
        raise ProvenanceError(f"protected paths changed: {json.dumps(mismatches, sort_keys=True)}")
    return {"roots_verified": len(roots), "sha256": sha256_file(snapshot_path), "status": "unchanged"}


def _main_build_exclusions(arguments: argparse.Namespace) -> dict[str, Any]:
    digest, path, payload = build_exclusion_registry(
        base_registry_path=arguments.base_registry,
        base_registry_sha256=arguments.base_registry_sha256,
        protected_inventory_paths=arguments.protected_inventory,
        battle_snapshot_paths=arguments.battle_snapshot,
        t0_utc=arguments.t0_utc,
        output_directory=arguments.output_directory,
        repository=arguments.repository,
    )
    return {
        "game_ids": len(payload["records"]),
        "path": path.as_posix(),
        "schema": EXCLUSION_SCHEMA,
        "sha256": digest,
        "sources": len(payload["sources"]),
    }


def _main_create_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    digest, path, payload = create_window_plan(
        t0_utc=arguments.t0_utc,
        planned_at_utc=arguments.planned_at_utc,
        collection_windows=arguments.collection_windows,
        output_directory=arguments.output_directory,
    )
    return {"path": path.as_posix(), "sha256": digest, "windows": len(payload["windows"])}


def _main_attest_editor(arguments: argparse.Namespace) -> dict[str, Any]:
    preflight = {
        "compilation": arguments.compilation,
        "legal_action": arguments.legal_action,
        "protocol": arguments.protocol,
        "purity": arguments.purity,
        "source_size": arguments.source_size,
        "timing_both_colors": arguments.timing_both_colors,
    }
    digest, path, payload = create_editor_attestation(
        plan_path=arguments.plan,
        expected_plan_sha256=arguments.plan_sha256,
        window_id=arguments.window_id,
        source_path=arguments.source,
        copied_back_path=arguments.copied_back_source,
        repository_commit=arguments.repository_commit,
        agent_id=arguments.agent_id,
        submission_id=arguments.submission_id,
        uploaded_at_utc=arguments.uploaded_at_utc,
        checked_at_utc=arguments.play_checked_at_utc,
        preflight=preflight,
        play_stdout_legal=arguments.play_stdout_legal,
        play_telemetry_ok=arguments.play_telemetry_ok,
        output_directory=arguments.output_directory,
        repository=arguments.repository,
    )
    return {
        "path": path.as_posix(),
        "sha256": digest,
        "source_sha256": payload["source"]["sha256"],
        "window_id": payload["window"]["window_id"],
    }


def _main_derive_window(arguments: argparse.Namespace) -> dict[str, Any]:
    digest, path, payload = derive_arena_window(
        plan_path=arguments.plan,
        expected_plan_sha256=arguments.plan_sha256,
        attestation_path=arguments.attestation,
        expected_attestation_sha256=arguments.attestation_sha256,
        arena_manifest_path=arguments.arena_manifest,
        exclusion_registry_path=arguments.exclusion_registry,
        expected_exclusion_sha256=arguments.exclusion_registry_sha256,
        output_directory=arguments.output_directory,
        repository=arguments.repository,
    )
    return {"path": path.as_posix(), "sha256": digest, "summary": payload["summary"]}


def _main_validate_derivation(arguments: argparse.Namespace) -> dict[str, Any]:
    payload = validate_arena_derivation(
        arguments.derivation,
        repository=arguments.repository,
        expected_sha256=arguments.derivation_sha256,
    )
    return {
        "sha256": sha256_file(arguments.derivation),
        "status": "structurally-valid",
        "summary": payload["summary"],
        "usage": arena_derivation_usage(payload),
    }


def _main_validate_sequence(arguments: argparse.Namespace) -> dict[str, Any]:
    return validate_campaign_sequence(
        arguments.derivation,
        repository=arguments.repository,
        plan_path=arguments.plan,
        expected_plan_sha256=arguments.plan_sha256,
    )


def _main_snapshot(arguments: argparse.Namespace) -> dict[str, Any]:
    digest, path, payload = create_protected_snapshot(
        protected_paths=arguments.protected_path,
        label=arguments.label,
        created_at_utc=arguments.created_at_utc,
        output_directory=arguments.output_directory,
        repository=arguments.repository,
    )
    return {"path": path.as_posix(), "roots": len(payload["roots"]), "sha256": digest}


def _main_verify_snapshot(arguments: argparse.Namespace) -> dict[str, Any]:
    return verify_protected_snapshot(arguments.snapshot, repository=arguments.repository)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-exclusions", help="build sealed ID-only exclusions")
    build.add_argument("--base-registry", type=pathlib.Path, required=True)
    build.add_argument("--base-registry-sha256")
    build.add_argument("--protected-inventory", type=pathlib.Path, action="append", required=True)
    build.add_argument("--battle-snapshot", type=pathlib.Path, action="append", required=True)
    build.add_argument("--t0-utc", default=DEFAULT_T0_UTC)
    build.add_argument("--output-directory", type=pathlib.Path, required=True)
    build.add_argument("--repository", type=pathlib.Path)
    plan = commands.add_parser("create-window-plan", help="freeze blind window roles")
    plan.add_argument("--t0-utc", default=DEFAULT_T0_UTC)
    plan.add_argument("--planned-at-utc", required=True)
    plan.add_argument("--collection-windows", type=int, default=8)
    plan.add_argument("--output-directory", type=pathlib.Path, required=True)
    attest = commands.add_parser("attest-editor", help="attest exact editor paste and Play check")
    attest.add_argument("--plan", type=pathlib.Path, required=True)
    attest.add_argument("--plan-sha256", required=True)
    attest.add_argument("--window-id", required=True)
    attest.add_argument("--source", type=pathlib.Path, required=True)
    attest.add_argument("--copied-back-source", type=pathlib.Path, required=True)
    attest.add_argument("--repository-commit", required=True)
    attest.add_argument("--agent-id", type=int, required=True)
    attest.add_argument("--submission-id", type=int, required=True)
    attest.add_argument("--uploaded-at-utc", required=True)
    attest.add_argument("--play-checked-at-utc", required=True)
    attest.add_argument("--repository", type=pathlib.Path, required=True)
    attest.add_argument("--output-directory", type=pathlib.Path, required=True)
    for flag in sorted(_PREFLIGHT_KEYS):
        attest.add_argument(f"--{flag.replace('_', '-')}", action="store_true")
    attest.add_argument("--play-stdout-legal", action="store_true")
    attest.add_argument("--play-telemetry-ok", action="store_true")
    derive = commands.add_parser("derive-arena-window", help="validate and bind one fresh 90-game window")
    derive.add_argument("--plan", type=pathlib.Path, required=True)
    derive.add_argument("--plan-sha256", required=True)
    derive.add_argument("--attestation", type=pathlib.Path, required=True)
    derive.add_argument("--attestation-sha256", required=True)
    derive.add_argument("--arena-manifest", type=pathlib.Path, required=True)
    derive.add_argument("--exclusion-registry", type=pathlib.Path, required=True)
    derive.add_argument("--exclusion-registry-sha256", required=True)
    derive.add_argument("--repository", type=pathlib.Path, required=True)
    derive.add_argument("--output-directory", type=pathlib.Path, required=True)
    validate_derived = commands.add_parser("validate-derivation", help="validate a frozen arena derivation")
    validate_derived.add_argument("--derivation", type=pathlib.Path, required=True)
    validate_derived.add_argument("--derivation-sha256", required=True)
    validate_derived.add_argument("--repository", type=pathlib.Path, required=True)
    sequence = commands.add_parser("validate-sequence", help="verify sequential complete submission windows")
    sequence.add_argument("--derivation", type=pathlib.Path, action="append", required=True)
    sequence.add_argument("--plan", type=pathlib.Path, required=True)
    sequence.add_argument("--plan-sha256", required=True)
    sequence.add_argument("--repository", type=pathlib.Path, required=True)
    snapshot = commands.add_parser("snapshot-protected", help="seal protected path bytes and topology")
    snapshot.add_argument("--protected-path", type=pathlib.Path, action="append", required=True)
    snapshot.add_argument("--label", required=True)
    snapshot.add_argument("--created-at-utc", required=True)
    snapshot.add_argument("--repository", type=pathlib.Path)
    snapshot.add_argument("--output-directory", type=pathlib.Path, required=True)
    verify = commands.add_parser("verify-protected", help="verify a protected snapshot is unchanged")
    verify.add_argument("--snapshot", type=pathlib.Path, required=True)
    verify.add_argument("--repository", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build-exclusions":
        result = _main_build_exclusions(arguments)
    elif arguments.command == "create-window-plan":
        result = _main_create_plan(arguments)
    elif arguments.command == "attest-editor":
        result = _main_attest_editor(arguments)
    elif arguments.command == "derive-arena-window":
        result = _main_derive_window(arguments)
    elif arguments.command == "validate-derivation":
        result = _main_validate_derivation(arguments)
    elif arguments.command == "validate-sequence":
        result = _main_validate_sequence(arguments)
    elif arguments.command == "snapshot-protected":
        result = _main_snapshot(arguments)
    elif arguments.command == "verify-protected":
        result = _main_verify_snapshot(arguments)
    else:  # pragma: no cover - argparse owns command validation
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
