#!/usr/bin/env python3
"""Fail-closed arena-window provenance for the Rank-4/Jacek hybrid.

The wrapper has one deliberately narrow arena purpose: collect a single,
predeclared 90-game validation window without making any of it trainable.  It
monitors newly completed game details for an attributable focus-agent failure
before the window completes, then delegates the immutable exact window to the
repository's generic collector.  Until all 90 games exist, console reports
contain only counts or a focus-safety failure category -- never outcomes,
opponents, moves, frames, or transcript text.

CodinGame does not expose uploaded editor source bytes through its public API.
The ``attest`` command therefore requires an exact editor copy-back and records
that limitation explicitly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
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


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
GENERIC_COLLECTOR = (
    REPOSITORY / "submissions" / "codingame" / "tools" / "collect_arena_batch.py"
)
DEFAULT_CAMPAIGN = REPOSITORY / "results" / "rank_4_jacek_hybrid" / "campaign.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY / "results" / "rank_4_jacek_hybrid" / "arena"

NAMESPACE = "rank_4_jacek_hybrid"
PLAN_SCHEMA = "papersoccer.rank4-jacek-hybrid.arena-window-plan.v1"
ATTESTATION_SCHEMA = "papersoccer.rank4-jacek-hybrid.editor-attestation.v1"
MONITOR_SNAPSHOT_SCHEMA = "papersoccer.rank4-jacek-hybrid.arena-monitor-snapshot.v1"
COLLECTION_SCHEMA = "papersoccer.rank4-jacek-hybrid.arena-collection.v1"
DERIVATION_SCHEMA = "papersoccer.rank4-jacek-hybrid.arena-validation-derivation.v1"
GENERIC_BATCH_SCHEMA = "papersoccer.codingame-arena-batch.v1"
GENERIC_GAME_SCHEMA = "papersoccer.codingame-arena-game.v1"
GENERIC_BINDING_SCHEMA = "papersoccer.codingame-arena-binding.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"

VALIDATION_WINDOW_ID = "hybrid-validation-001"
ROLLBACK_WINDOW_ID = "safe-h62-rollback-accounting"
EXACT_GAMES = 90
SOURCE_LIMIT = 99_999
FAILURE_EXIT = 42
SAFE_H62_SHA256 = "d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71"
SAFE_H62_BYTES = 99_810
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")

SERVICE_ROOT = "https://www.codingame.com/services"
BATTLE_SERVICE = "gamesPlayersRankingRemoteService/findLastBattlesByAgentId"
DETAIL_SERVICE = "gameResultRemoteService/findByGameId"

DISCLOSURE = (
    "CodinGame upload bytes are editor-attested by exact paste/copy-back "
    "byte/count/SHA equality; the public API does not expose editor source bytes."
)

PREFLIGHT_KEYS = frozenset(
    {
        "compilation",
        "legal_action",
        "protocol",
        "purity",
        "source_size",
        "timing_both_colors",
    }
)
FORBIDDEN_METADATA_KEYS = frozenset(
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
FAILURE_PATTERNS = (
    ("timeout", re.compile(r"\btime[ -]?out\b|timed out|exceeded (?:the )?time limit|too long to respond", re.I)),
    ("illegal-action", re.compile(r"illegal (?:move|action|edge)|invalid (?:move|action|edge)|move is not legal|action is not legal", re.I)),
    ("crash", re.compile(r"runtime error|segmentation fault|uncaught exception|terminated by signal|process (?:was )?killed|out of memory", re.I)),
    ("malformed-transcript", re.compile(r"invalid output|malformed output|unrecognized output|could not parse|provided no output|empty output", re.I)),
)
TEXT_FRAME_FIELDS = ("stderr", "gameInformation", "summary", "tooltip", "tooltips")


class ArenaWindowError(ValueError):
    """An artifact or live window violates the fail-closed contract."""


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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise ArenaWindowError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ArenaWindowError(f"{field} is not a valid UTC timestamp") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise ArenaWindowError(f"{field} must be UTC")
    return parsed


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArenaWindowError(f"{field} must be a positive integer")
    return value


def checked_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ArenaWindowError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def resolve_path(raw: Any, repository: pathlib.Path) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise ArenaWindowError("artifact path is missing")
    path = pathlib.Path(raw)
    return path if path.is_absolute() else repository / path


def require_within(path: pathlib.Path, root: pathlib.Path, field: str) -> pathlib.Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ArenaWindowError(f"{field} escaped its campaign-local archive root") from error
    return resolved


def logical_path(path: pathlib.Path, repository: pathlib.Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_once(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ArenaWindowError(f"immutable artifact collision: {path}")


def write_content_addressed(
    directory: pathlib.Path, payload: Mapping[str, Any]
) -> tuple[str, pathlib.Path]:
    content = canonical_json_bytes(dict(payload))
    digest = sha256_bytes(content)
    path = directory / f"{digest}.json"
    write_once(path, content)
    return digest, path


def load_json(path: pathlib.Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    content = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(content) != checked_sha(
        expected_sha256, "expected SHA-256"
    ):
        raise ArenaWindowError(f"artifact SHA-256 mismatch: {path}")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArenaWindowError(f"artifact is not JSON: {path}") from error
    if not isinstance(value, dict):
        raise ArenaWindowError(f"artifact must be a JSON object: {path}")
    return value


def load_content_addressed(
    path: pathlib.Path, schema: str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    value = load_json(path, expected_sha256=expected_sha256)
    content = path.read_bytes()
    digest = sha256_bytes(content)
    if path.stem != digest or content != canonical_json_bytes(value):
        raise ArenaWindowError(f"artifact is not canonical/content-addressed: {path}")
    if value.get("schema") != schema:
        raise ArenaWindowError(f"unexpected artifact schema: {value.get('schema')!r}")
    return value


def _window(window_id: str, ordinal: int, role: str, *, optional: bool) -> dict[str, Any]:
    return {
        "expected_games": EXACT_GAMES,
        "optional": optional,
        "ordinal": ordinal,
        "planned_before_results": True,
        "role": role,
        "training_eligible": False,
        "training_forbidden": True,
        "window_id": window_id,
    }


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ArenaWindowError("unexpected arena-window plan schema")
    if plan.get("namespace") != NAMESPACE:
        raise ArenaWindowError("arena-window plan has the wrong namespace")
    if plan.get("results_observed_before_assignment") is not False:
        raise ArenaWindowError("arena roles were not assigned before results")
    campaign = plan.get("campaign")
    exclusions = plan.get("exclusion_registry")
    if not isinstance(campaign, dict) or not isinstance(exclusions, dict):
        raise ArenaWindowError("plan omits campaign or exclusion registry binding")
    if (
        not isinstance(campaign.get("campaign_id"), str)
        or not campaign["campaign_id"]
        or not isinstance(campaign.get("path"), str)
        or not campaign["path"]
        or not isinstance(exclusions.get("path"), str)
        or not exclusions["path"]
    ):
        raise ArenaWindowError("plan campaign/exclusion paths or identity are invalid")
    checked_sha(campaign.get("sha256"), "campaign SHA-256")
    checked_sha(exclusions.get("sha256"), "exclusion-registry SHA-256")
    if exclusions.get("schema") != EXCLUSION_SCHEMA:
        raise ArenaWindowError("plan binds an unexpected exclusion-registry schema")
    t0 = parse_utc(campaign.get("t0_utc"), "campaign T0")
    deadline = parse_utc(campaign.get("deadline_utc"), "campaign deadline")
    planned = parse_utc(plan.get("planned_at_utc"), "plan creation time")
    if not t0 <= planned <= deadline:
        raise ArenaWindowError("plan creation is outside the campaign interval")
    expected = [
        _window(VALIDATION_WINDOW_ID, 1, "arena-validation", optional=False),
        _window(ROLLBACK_WINDOW_ID, 2, "rollback-accounting", optional=True),
    ]
    if plan.get("windows") != expected:
        raise ArenaWindowError("plan does not contain the exact immutable window roles")
    return dict(plan)


def create_plan(
    *,
    campaign_path: pathlib.Path,
    planned_at_utc: str,
    output_root: pathlib.Path,
    repository: pathlib.Path = REPOSITORY,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    campaign = load_json(campaign_path)
    if campaign.get("schema") != "papersoccer.rank4-jacek-hybrid-campaign.v1":
        raise ArenaWindowError("unexpected hybrid campaign schema")
    boundary = campaign.get("time_boundary")
    exclusions = campaign.get("arena_exclusions")
    if not isinstance(boundary, dict) or not isinstance(exclusions, dict):
        raise ArenaWindowError("campaign omits time boundary or arena exclusions")
    registry_path = resolve_path(exclusions.get("path"), repository)
    registry_hash = checked_sha(exclusions.get("sha256"), "campaign exclusion SHA-256")
    registry = load_json(registry_path, expected_sha256=registry_hash)
    if (
        registry.get("schema") != EXCLUSION_SCHEMA
        or registry_path.read_bytes() != canonical_json_bytes(registry)
        or (SHA256_RE.fullmatch(registry_path.stem) is not None and registry_path.stem != registry_hash)
    ):
        raise ArenaWindowError("campaign exclusion registry schema is invalid")
    payload = {
        "campaign": {
            "campaign_id": campaign.get("campaign_id"),
            "deadline_utc": boundary.get("deadline_utc"),
            "path": logical_path(campaign_path, repository),
            "sha256": sha256_file(campaign_path),
            "t0_utc": boundary.get("goal_created_at_utc"),
        },
        "exclusion_registry": {
            "path": logical_path(registry_path, repository),
            "schema": EXCLUSION_SCHEMA,
            "sha256": registry_hash,
        },
        "namespace": NAMESPACE,
        "planned_at_utc": planned_at_utc,
        "results_observed_before_assignment": False,
        "schema": PLAN_SCHEMA,
        "windows": [
            _window(VALIDATION_WINDOW_ID, 1, "arena-validation", optional=False),
            _window(ROLLBACK_WINDOW_ID, 2, "rollback-accounting", optional=True),
        ],
    }
    validate_plan(payload)
    digest, path = write_content_addressed(output_root / "plans", payload)
    return digest, path, payload


def validate_plan_references(
    plan: Mapping[str, Any], *, repository: pathlib.Path
) -> dict[str, Any]:
    validate_plan(plan)
    campaign_ref = plan["campaign"]
    exclusion_ref = plan["exclusion_registry"]
    campaign_path = resolve_path(campaign_ref["path"], repository)
    exclusion_path = resolve_path(exclusion_ref["path"], repository)
    campaign = load_json(campaign_path, expected_sha256=campaign_ref["sha256"])
    registry = load_json(exclusion_path, expected_sha256=exclusion_ref["sha256"])
    boundary = campaign.get("time_boundary")
    campaign_exclusion = campaign.get("arena_exclusions")
    if (
        campaign.get("schema") != "papersoccer.rank4-jacek-hybrid-campaign.v1"
        or campaign.get("campaign_id") != campaign_ref["campaign_id"]
        or not isinstance(boundary, dict)
        or boundary.get("goal_created_at_utc") != campaign_ref["t0_utc"]
        or boundary.get("deadline_utc") != campaign_ref["deadline_utc"]
        or not isinstance(campaign_exclusion, dict)
        or campaign_exclusion.get("sha256") != exclusion_ref["sha256"]
        or registry.get("schema") != EXCLUSION_SCHEMA
        or exclusion_path.read_bytes() != canonical_json_bytes(registry)
    ):
        raise ArenaWindowError("plan campaign/exclusion references are inconsistent")
    return dict(plan)


def window_from_plan(plan: Mapping[str, Any], window_id: str) -> dict[str, Any]:
    validate_plan(plan)
    found = [item for item in plan["windows"] if item["window_id"] == window_id]
    if len(found) != 1:
        raise ArenaWindowError(f"window is absent from the plan: {window_id}")
    return dict(found[0])


def verify_git_source(
    repository: pathlib.Path,
    source_path: pathlib.Path,
    repository_commit: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Require a tracked, committed source and a clean tracked worktree."""

    if not isinstance(repository_commit, str) or COMMIT_RE.fullmatch(repository_commit) is None:
        raise ArenaWindowError("repository commit must be a full lowercase Git SHA-1")
    repository = repository.resolve()
    source_path = source_path.resolve()
    try:
        relative = source_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise ArenaWindowError("generated source must be inside the repository") from error

    def git(*arguments: str, text: bool = False) -> Any:
        completed = runner(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=text,
        )
        if completed.returncode != 0:
            raise ArenaWindowError(
                f"Git source attestation failed for {' '.join(arguments[:2])}"
            )
        return completed.stdout

    head = str(git("rev-parse", "--verify", "HEAD", text=True)).strip()
    if head != repository_commit:
        raise ArenaWindowError("attested repository commit is not the current HEAD")
    tracked_status = str(
        git("status", "--porcelain=v1", "--untracked-files=no", text=True)
    )
    if tracked_status.strip():
        raise ArenaWindowError("tracked worktree is not clean at attestation time")
    git("ls-files", "--error-unmatch", "--", relative)
    committed = bytes(git("show", f"{repository_commit}:{relative}"))
    source = source_path.read_bytes()
    if committed != source:
        raise ArenaWindowError("generated source bytes differ from the committed tracked source")
    return {
        "commit": repository_commit,
        "source_path": relative,
        "status": "tracked-committed-clean",
    }


def validate_attestation(
    attestation: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    repository: pathlib.Path = REPOSITORY,
    verify_files: bool = True,
) -> dict[str, Any]:
    validate_plan(plan)
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ArenaWindowError("unexpected editor-attestation schema")
    window = window_from_plan(plan, str((attestation.get("window") or {}).get("window_id", "")))
    if attestation.get("window") != window:
        raise ArenaWindowError("attestation window contradicts the plan")
    identity = attestation.get("identity")
    source = attestation.get("source")
    copyback = attestation.get("editor_copyback")
    git = attestation.get("git")
    play = attestation.get("play_my_code")
    preflight = attestation.get("preflight")
    plan_ref = attestation.get("window_plan")
    if not all(isinstance(item, dict) for item in (identity, source, copyback, git, play, preflight, plan_ref)):
        raise ArenaWindowError("attestation omits a structured binding")
    positive_int(identity.get("agent_id"), "agent ID")
    positive_int(identity.get("submission_id"), "submission ID")
    commit = identity.get("repository_commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ArenaWindowError("attestation repository commit is invalid")
    source_hash = checked_sha(source.get("sha256"), "source SHA-256")
    size = source.get("bytes")
    if (
        source.get("ascii") is not True
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= SOURCE_LIMIT
        or source.get("characters") != size
        or copyback.get("sha256") != source_hash
        or copyback.get("bytes") != size
        or copyback.get("characters") != size
        or copyback.get("byte_equal_to_generated") is not True
        or copyback.get("api_readable") is not False
        or copyback.get("status") != "editor-attested-not-api-readable"
    ):
        raise ArenaWindowError("editor copy-back is not an exact ASCII source attestation")
    if window["role"] == "rollback-accounting":
        if source_hash != SAFE_H62_SHA256 or size != SAFE_H62_BYTES:
            raise ArenaWindowError("rollback accounting must bind the exact safe H62 source")
    elif source_hash == SAFE_H62_SHA256:
        raise ArenaWindowError("safe H62 cannot occupy the hybrid validation window")
    if set(preflight) != PREFLIGHT_KEYS or any(value is not True for value in preflight.values()):
        raise ArenaWindowError("all preflight gates must be explicitly true")
    if play.get("legal_stdout") is not True or play.get("expected_telemetry") is not True:
        raise ArenaWindowError("Play My Code legality and telemetry must both pass")
    if git != {
        "commit": commit,
        "source_path": source.get("generated_path"),
        "status": "tracked-committed-clean",
    }:
        raise ArenaWindowError("attestation Git binding is inconsistent")
    if attestation.get("upload_bytes_disclosure") != DISCLOSURE:
        raise ArenaWindowError("attestation omits the editor/API disclosure")
    plan_hash = sha256_bytes(canonical_json_bytes(plan))
    if plan_ref.get("sha256") != plan_hash:
        raise ArenaWindowError("attestation window-plan digest is inconsistent")
    checked = parse_utc(play.get("checked_at_utc"), "Play check time")
    uploaded = parse_utc(attestation.get("uploaded_at_utc"), "upload time")
    created = parse_utc(attestation.get("created_at_utc"), "attestation time")
    t0 = parse_utc(plan["campaign"]["t0_utc"], "campaign T0")
    deadline = parse_utc(plan["campaign"]["deadline_utc"], "campaign deadline")
    if not t0 <= checked <= uploaded <= created <= deadline:
        raise ArenaWindowError("Play/upload/attestation timestamps violate campaign order")
    if verify_files:
        plan_path = resolve_path(plan_ref.get("path"), repository)
        archive_path = resolve_path(source.get("archived_path"), repository)
        archive = archive_path.read_bytes()
        copy_path = resolve_path(copyback.get("path"), repository)
        generated_path = resolve_path(source.get("generated_path"), repository)
        if copy_path.resolve() == generated_path.resolve():
            raise ArenaWindowError(
                "editor copy-back must be a distinct retained file, not the generated source"
            )
        if (
            sha256_file(plan_path) != plan_hash
            or sha256_bytes(archive) != source_hash
            or len(archive) != size
            or generated_path.read_bytes() != archive
            or copy_path.read_bytes() != archive
        ):
            raise ArenaWindowError("attested source/archive/copy-back files no longer agree")
    return dict(attestation)


def create_attestation(
    *,
    plan_path: pathlib.Path,
    plan_sha256: str,
    window_id: str,
    generated_source: pathlib.Path,
    copied_back_source: pathlib.Path,
    repository: pathlib.Path,
    repository_commit: str,
    agent_id: int,
    submission_id: int,
    play_checked_at_utc: str,
    uploaded_at_utc: str,
    created_at_utc: str,
    preflight: Mapping[str, bool],
    play_stdout_legal: bool,
    play_telemetry_ok: bool,
    output_root: pathlib.Path,
    git_verifier: Callable[[pathlib.Path, pathlib.Path, str], Mapping[str, Any]] = verify_git_source,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    plan = load_content_addressed(plan_path, PLAN_SCHEMA, expected_sha256=plan_sha256)
    validate_plan_references(plan, repository=repository)
    window = window_from_plan(plan, window_id)
    if generated_source.resolve() == copied_back_source.resolve():
        raise ArenaWindowError(
            "editor copy-back must be a distinct retained file, not the generated source"
        )
    generated = generated_source.read_bytes()
    copied = copied_back_source.read_bytes()
    try:
        generated_text = generated.decode("ascii")
        copied_text = copied.decode("ascii")
    except UnicodeDecodeError as error:
        raise ArenaWindowError("generated and copied-back sources must be ASCII") from error
    if not generated or generated != copied:
        raise ArenaWindowError("editor copy-back is not byte-identical to generated source")
    if len(generated) > SOURCE_LIMIT:
        raise ArenaWindowError(f"generated source exceeds {SOURCE_LIMIT} ASCII bytes")
    git_binding = dict(git_verifier(repository, generated_source, repository_commit))
    source_hash = sha256_bytes(generated)
    archive_path = output_root / "source_payloads" / f"{source_hash}.source"
    write_once(archive_path, generated)
    payload = {
        "created_at_utc": created_at_utc,
        "editor_copyback": {
            "api_readable": False,
            "byte_equal_to_generated": True,
            "bytes": len(copied),
            "characters": len(copied_text),
            "path": logical_path(copied_back_source, repository),
            "sha256": source_hash,
            "status": "editor-attested-not-api-readable",
        },
        "git": git_binding,
        "identity": {
            "agent_id": positive_int(agent_id, "agent ID"),
            "repository_commit": repository_commit,
            "submission_id": positive_int(submission_id, "submission ID"),
        },
        "play_my_code": {
            "checked_at_utc": play_checked_at_utc,
            "expected_telemetry": play_telemetry_ok is True,
            "legal_stdout": play_stdout_legal is True,
        },
        "preflight": dict(sorted(preflight.items())),
        "schema": ATTESTATION_SCHEMA,
        "source": {
            "archived_path": logical_path(archive_path, repository),
            "ascii": True,
            "bytes": len(generated),
            "characters": len(generated_text),
            "generated_path": logical_path(generated_source, repository),
            "sha256": source_hash,
        },
        "upload_bytes_disclosure": DISCLOSURE,
        "uploaded_at_utc": uploaded_at_utc,
        "window": window,
        "window_plan": {
            "path": logical_path(plan_path, repository),
            "sha256": plan_sha256,
        },
    }
    validate_attestation(payload, plan, repository=repository)
    digest, path = write_content_addressed(
        output_root / "attestations" / window_id, payload
    )
    return digest, path, payload


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_detail_in_metadata(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArenaWindowError(f"non-string metadata key at {path}")
            if _canonical_key(key) in FORBIDDEN_METADATA_KEYS:
                raise ArenaWindowError(f"game-detail field {key!r} embedded in metadata")
            _reject_detail_in_metadata(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_detail_in_metadata(child, f"{path}[{index}]")


def classify_metadata(
    battles: Any, *, agent_id: int, submission_id: int
) -> dict[str, Any]:
    """Classify only the exact focus-agent/submission window."""

    positive_int(agent_id, "agent ID")
    positive_int(submission_id, "submission ID")
    if not isinstance(battles, list):
        raise ArenaWindowError("battle metadata response is not a list")
    _reject_detail_in_metadata(battles)
    matching: list[int] = []
    complete: list[int] = []
    pending: list[int] = []
    seen: set[int] = set()
    for index, battle in enumerate(battles):
        if not isinstance(battle, Mapping):
            raise ArenaWindowError(f"battle {index} is not an object")
        game_id = battle.get("gameId")
        if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id <= 0:
            raise ArenaWindowError(f"battle {index} has an invalid gameId")
        if game_id in seen:
            raise ArenaWindowError(f"battle metadata repeats game {game_id}")
        seen.add(game_id)
        players = battle.get("players")
        if not isinstance(players, list):
            raise ArenaWindowError(f"battle {index} players is not a list")
        focus = []
        for player_index, player in enumerate(players):
            if not isinstance(player, Mapping):
                raise ArenaWindowError(f"battle {index} player {player_index} is not an object")
            if player.get("playerAgentId") == agent_id:
                focus.append(player)
        if len(focus) > 1:
            raise ArenaWindowError(f"battle {index} repeats the focus agent")
        if not focus:
            continue
        focus_submission = focus[0].get("submissionId")
        if focus_submission is None:
            raise ArenaWindowError(
                f"battle {index} has ambiguous focus-agent submission metadata"
            )
        if focus_submission != submission_id:
            continue
        matching.append(game_id)
        done = battle.get("done")
        if done is not None and not isinstance(done, bool):
            raise ArenaWindowError(f"battle {index} done is not boolean")
        if done is not True or len(players) != 2:
            pending.append(game_id)
            continue
        positions = {player.get("position") for player in players}
        agent_ids = [player.get("playerAgentId") for player in players]
        if (
            positions != {0, 1}
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in agent_ids)
            or len(set(agent_ids)) != 2
        ):
            raise ArenaWindowError(f"battle {index} has malformed completed players")
        complete.append(game_id)
    matching.sort()
    complete.sort()
    pending.sort()
    return {
        "complete_game_ids": complete,
        "complete_games": len(complete),
        "expected_games": EXACT_GAMES,
        "matching_games": len(matching),
        "overfull": len(matching) > EXACT_GAMES,
        "pending_game_ids": pending,
        "pending_games": len(pending),
        "ready": len(matching) == EXACT_GAMES and len(complete) == EXACT_GAMES,
    }


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for child in value:
            result.extend(_texts(child))
        return result
    if isinstance(value, Mapping):
        result = []
        for child in value.values():
            result.extend(_texts(child))
        return result
    return []


def _frame_player(frame: Mapping[str, Any], texts: Sequence[str]) -> int | None:
    raw = frame.get("agentId")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw in (0, 1):
        return raw
    scoped: set[int] = set()
    for text in texts:
        scoped.update(int(match) for match in re.findall(r"\$([01])\b", text))
    return next(iter(scoped)) if len(scoped) == 1 else None


def _generic_rules_validation(turns: list[dict[str, Any]], winner: int) -> dict[str, Any]:
    module_name = "rank4_jacek_hybrid_arena_generic_collector"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, GENERIC_COLLECTOR)
        if spec is None or spec.loader is None:
            raise ArenaWindowError("cannot load generic arena rule validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.validate_turns(turns, winner)


def inspect_focus_detail(
    detail: Any,
    *,
    game_id: int,
    focus_agent_id: int,
    rules_validator: Callable[[list[dict[str, Any]], int], Mapping[str, Any]] = _generic_rules_validation,
) -> dict[str, Any]:
    """Return a sanitized focus-safety result; never return replay text."""

    safe = {"category": None, "focus_failure": False, "game_id": game_id}
    if not isinstance(detail, Mapping) or detail.get("gameId") != game_id:
        return {**safe, "category": "malformed-transcript", "focus_failure": True}
    agents = detail.get("agents")
    ranks = detail.get("ranks")
    frames = detail.get("frames")
    if not isinstance(agents, list) or not isinstance(ranks, list) or not isinstance(frames, list):
        return {**safe, "category": "malformed-transcript", "focus_failure": True}
    focus_indexes = [
        item.get("index")
        for item in agents
        if isinstance(item, Mapping) and item.get("agentId") == focus_agent_id
    ]
    if len(focus_indexes) != 1 or focus_indexes[0] not in (0, 1):
        return {**safe, "category": "malformed-transcript", "focus_failure": True}
    focus_player = int(focus_indexes[0])
    statuses: dict[int, str | None] = {0: None, 1: None}
    turns: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            return {**safe, "category": "malformed-transcript", "focus_failure": True}
        text_values = [
            text
            for field in TEXT_FRAME_FIELDS
            if field in frame
            for text in _texts(frame[field])
        ]
        player = _frame_player(frame, text_values)
        for text in text_values:
            for category, pattern in FAILURE_PATTERNS:
                if pattern.search(text) is not None:
                    if player in (0, 1):
                        statuses[int(player)] = category
                    break
        if player not in (0, 1) or "stdout" not in frame:
            continue
        action = "" if frame.get("stdout") is None else str(frame["stdout"]).strip()
        if not action or any(character not in "01234567" for character in action):
            statuses[int(player)] = "malformed-transcript"
        else:
            turns.append({"action": action, "player_id": int(player)})
    if statuses[focus_player] is not None:
        return {**safe, "category": statuses[focus_player], "focus_failure": True}
    if (
        len(ranks) != 2
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
        or sorted(ranks) != [0, 1]
    ):
        return {**safe, "category": "malformed-transcript", "focus_failure": True}
    try:
        validation = dict(rules_validator(turns, ranks.index(0)))
    except Exception:
        return {**safe, "category": "malformed-transcript", "focus_failure": True}
    status = validation.get("status")
    failing = validation.get("failing_player_id")
    if status == "terminal-valid":
        return safe
    if status == "invalid":
        if failing == 1 - focus_player:
            return safe
        if failing == focus_player:
            reason = str(validation.get("reason") or "")
            category = (
                "malformed-transcript"
                if reason == "invalid-output"
                else "illegal-action"
            )
            return {**safe, "category": category, "focus_failure": True}
    if status == "incomplete" and statuses[1 - focus_player] is not None:
        return safe
    return {**safe, "category": "malformed-transcript", "focus_failure": True}


class PublicJsonApi:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def post(self, service: str, payload: Any) -> Any:
        request = urllib.request.Request(
            f"{SERVICE_ROOT}/{service}",
            data=json.dumps(payload, separators=(",", ":")).encode("ascii"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(7):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 6:
                    raise
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError("unreachable API retry state")


def _archive_monitor_payload(output_root: pathlib.Path, kind: str, payload: Any) -> str:
    content = canonical_json_bytes(payload)
    digest = sha256_bytes(content)
    write_once(output_root / "monitor" / kind / f"{digest}.json", content)
    return digest


def build_collector_command(
    *,
    repository: pathlib.Path,
    attestation: Mapping[str, Any],
    exclusion_registry: pathlib.Path,
    exclusion_sha256: str,
    data_root: pathlib.Path,
    maximum_workers: int,
) -> list[str]:
    identity = attestation["identity"]
    source = attestation["source"]
    return [
        sys.executable,
        str(GENERIC_COLLECTOR),
        "--agent-id",
        str(identity["agent_id"]),
        "--submission-id",
        str(identity["submission_id"]),
        "--source",
        str(resolve_path(source["archived_path"], repository)),
        "--source-sha256",
        source["sha256"],
        "--repository-commit",
        identity["repository_commit"],
        "--run-id",
        attestation["window"]["window_id"],
        "--expected-games",
        str(EXACT_GAMES),
        "--data-root",
        str(data_root.resolve()),
        "--exclusion-registry",
        str(exclusion_registry.resolve()),
        "--exclusion-registry-sha256",
        exclusion_sha256,
        "--maximum-workers",
        str(maximum_workers),
    ]


def _run_collector(
    command: Sequence[str],
    *,
    repository: pathlib.Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    completed = runner(
        list(command),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ArenaWindowError(
            f"generic collector failed with return code {completed.returncode}"
        )
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ArenaWindowError("generic collector did not return one JSON result") from error
    if not isinstance(result, dict):
        raise ArenaWindowError("generic collector result is not an object")
    return result


def verify_collector_result(
    result: Mapping[str, Any],
    *,
    repository: pathlib.Path,
    data_root: pathlib.Path,
    attestation: Mapping[str, Any],
    exclusion_sha256: str,
    expected_game_ids: Sequence[int],
) -> dict[str, Any]:
    """Verify the collector's exact manifest, source, registry, and records."""

    if result.get("schema") != GENERIC_BATCH_SCHEMA:
        raise ArenaWindowError("collector returned an unexpected schema")
    manifest_hash = checked_sha(result.get("manifest_sha256"), "manifest SHA-256")
    manifest_path = resolve_path(result.get("manifest_path"), repository).resolve()
    require_within(manifest_path, data_root, "collector manifest")
    manifest = load_content_addressed(
        manifest_path, GENERIC_BATCH_SCHEMA, expected_sha256=manifest_hash
    )
    if result.get("run_id") != attestation["window"]["window_id"] or manifest.get("run_id") != result.get("run_id"):
        raise ArenaWindowError("collector run ID contradicts the planned window")
    coverage = manifest.get("coverage")
    binding = manifest.get("binding")
    exclusion = manifest.get("exclusion_registry")
    if not all(isinstance(item, dict) for item in (coverage, binding, exclusion)):
        raise ArenaWindowError("collector manifest omits coverage/provenance")
    collector_hash = sha256_file(GENERIC_COLLECTOR)
    if (
        coverage.get("expected_games") != EXACT_GAMES
        or coverage.get("battle_window_games") != EXACT_GAMES
        or coverage.get("accepted_games") != EXACT_GAMES
        or coverage.get("focus_operational_failures") != 0
        or coverage.get("full_window_accounted") is not True
        or result.get("coverage") not in (None, coverage)
    ):
        raise ArenaWindowError("collector archive is not one cleanly bound exact 90-game window")
    identity = attestation["identity"]
    source = attestation["source"]
    bound_source = binding.get("source")
    if (
        binding.get("schema") != GENERIC_BINDING_SCHEMA
        or manifest.get("collector_sha256") != collector_hash
        or binding.get("collector_sha256") != collector_hash
        or binding.get("agent_id") != identity["agent_id"]
        or binding.get("asserted_submission_id") != identity["submission_id"]
        or binding.get("repository_commit") != identity["repository_commit"]
        or not isinstance(bound_source, dict)
        or bound_source.get("sha256") != source["sha256"]
        or bound_source.get("bytes") != source["bytes"]
        or bound_source.get("characters") != source["characters"]
        or exclusion.get("sha256") != exclusion_sha256
    ):
        raise ArenaWindowError("collector archive contradicts source/exclusion attestation")
    bound_archive = require_within(
        resolve_path(bound_source.get("archived_path"), repository),
        data_root,
        "collector source archive",
    )
    if sha256_file(bound_archive) != source["sha256"]:
        raise ArenaWindowError("collector source archive is corrupt")
    registry_archive = require_within(
        resolve_path(exclusion.get("path"), repository),
        data_root,
        "collector exclusion archive",
    )
    registry_payload = load_json(registry_archive)
    if (
        sha256_file(registry_archive) != exclusion_sha256
        or registry_payload.get("schema") != EXCLUSION_SCHEMA
        or registry_archive.read_bytes() != canonical_json_bytes(registry_payload)
    ):
        raise ArenaWindowError("collector exclusion archive is corrupt")
    games = manifest.get("games")
    expected_ids = set(expected_game_ids)
    if not isinstance(games, list) or len(games) != EXACT_GAMES or len(expected_ids) != EXACT_GAMES:
        raise ArenaWindowError("collector manifest has the wrong exact game cardinality")
    observed_ids: set[int] = set()
    for stored in games:
        if not isinstance(stored, dict) or not isinstance(stored.get("record"), dict):
            raise ArenaWindowError("collector manifest contains an invalid record binding")
        record = stored["record"]
        record_hash = checked_sha(stored.get("record_sha256"), "record SHA-256")
        record_path = require_within(
            resolve_path(stored.get("record_path"), repository),
            data_root,
            "collector game-record archive",
        )
        content = record_path.read_bytes()
        if (
            sha256_bytes(content) != record_hash
            or content != canonical_json_bytes(record)
            or record.get("schema") != GENERIC_GAME_SCHEMA
            or record.get("status") != "accepted"
            or record.get("source_sha256") != source["sha256"]
            or (record.get("focus") or {}).get("agent_id") != identity["agent_id"]
            or (record.get("focus") or {}).get("submission_id") != identity["submission_id"]
            or (record.get("operational") or {}).get("focus_status") != "ok"
        ):
            raise ArenaWindowError("collector game-record archive is corrupt or misbound")
        acquisition = record.get("acquisition")
        if not isinstance(acquisition, dict):
            raise ArenaWindowError("accepted collector record omits acquisition archives")
        for path_key, hash_key in (
            ("raw_path", "raw_sha256"),
            ("normalized_path", "normalized_sha256"),
            ("replay_payload_path", "normalized_sha256"),
        ):
            payload_hash = checked_sha(acquisition.get(hash_key), f"acquisition {hash_key}")
            payload_path = require_within(
                resolve_path(acquisition.get(path_key), repository),
                data_root,
                f"acquisition {path_key}",
            )
            if sha256_file(payload_path) != payload_hash:
                raise ArenaWindowError("collector acquisition archive is corrupt")
        game_id = record.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id in observed_ids:
            raise ArenaWindowError("collector repeats or invalidates a game ID")
        observed_ids.add(game_id)
    if observed_ids != expected_ids:
        raise ArenaWindowError("collector archive game IDs differ from the monitored exact window")
    return {"manifest": manifest, "manifest_path": manifest_path, "manifest_sha256": manifest_hash}


def watch_collect(
    *,
    plan_path: pathlib.Path,
    plan_sha256: str,
    attestation_path: pathlib.Path,
    attestation_sha256: str,
    exclusion_registry: pathlib.Path,
    exclusion_sha256: str,
    data_root: pathlib.Path,
    repository: pathlib.Path = REPOSITORY,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 3600.0,
    request_timeout_seconds: float = 30.0,
    maximum_workers: int = 2,
    fetch_battles: Callable[[], Any] | None = None,
    fetch_detail: Callable[[int], Any] | None = None,
    detail_inspector: Callable[..., Mapping[str, Any]] = inspect_focus_detail,
    runner: Callable[..., Any] = subprocess.run,
    collector_verifier: Callable[..., Mapping[str, Any]] = verify_collector_result,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    clock: Callable[[], str] = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    if poll_seconds < 0 or timeout_seconds < 0 or request_timeout_seconds <= 0:
        raise ArenaWindowError("poll and timeout seconds must be non-negative")
    if maximum_workers not in (1, 2, 3, 4):
        raise ArenaWindowError("collector worker count must be between one and four")
    plan = load_content_addressed(plan_path, PLAN_SCHEMA, expected_sha256=plan_sha256)
    validate_plan_references(plan, repository=repository)
    attestation = load_content_addressed(
        attestation_path, ATTESTATION_SCHEMA, expected_sha256=attestation_sha256
    )
    validate_attestation(attestation, plan, repository=repository)
    if attestation["window"]["expected_games"] != EXACT_GAMES:
        raise ArenaWindowError("attested window is not exactly 90 games")
    expected_exclusion = plan["exclusion_registry"]
    exclusion_sha256 = checked_sha(exclusion_sha256, "exclusion-registry SHA-256")
    if (
        exclusion_sha256 != expected_exclusion["sha256"]
        or exclusion_registry.resolve()
        != resolve_path(expected_exclusion["path"], repository).resolve()
        or sha256_file(exclusion_registry) != exclusion_sha256
        or load_json(exclusion_registry).get("schema") != EXCLUSION_SCHEMA
    ):
        raise ArenaWindowError("watch collector exclusion registry is not the planned registry")
    api = PublicJsonApi(request_timeout_seconds)
    if fetch_battles is None:
        fetch_battles = lambda: api.post(
            BATTLE_SERVICE, [attestation["identity"]["agent_id"], None]
        )
    if fetch_detail is None:
        fetch_detail = lambda game_id: api.post(DETAIL_SERVICE, [game_id, None])
    inspected: set[int] = set()
    metadata_hashes: list[str] = []
    detail_hashes: dict[int, str] = {}
    started = monotonic()
    while True:
        battles = fetch_battles()
        metadata_hashes.append(_archive_monitor_payload(data_root, "metadata", battles))
        report = classify_metadata(
            battles,
            agent_id=attestation["identity"]["agent_id"],
            submission_id=attestation["identity"]["submission_id"],
        )
        if report["overfull"]:
            raise ArenaWindowError("matching submission window exceeds exactly 90 games")
        for game_id in report["complete_game_ids"]:
            if game_id in inspected:
                continue
            detail = fetch_detail(game_id)
            detail_hashes[game_id] = _archive_monitor_payload(data_root, "details", detail)
            safety = dict(
                detail_inspector(
                    detail,
                    game_id=game_id,
                    focus_agent_id=attestation["identity"]["agent_id"],
                )
            )
            inspected.add(game_id)
            if safety.get("focus_failure") is True:
                failure_receipt = {
                    "detected_at_utc": clock(),
                    "detail_payload_sha256": detail_hashes[game_id],
                    "editor_attestation": {
                        "path": logical_path(attestation_path, repository),
                        "sha256": attestation_sha256,
                    },
                    "category": safety.get("category") or "malformed-transcript",
                    "game_id": game_id,
                    "rollback_required": True,
                    "schema": MONITOR_SNAPSHOT_SCHEMA,
                    "status": "focus-operational-failure",
                    "window_id": attestation["window"]["window_id"],
                }
                failure_hash, failure_path = write_content_addressed(
                    data_root / "monitor" / "failures", failure_receipt
                )
                result = {
                    **failure_receipt,
                    "failure_receipt_path": logical_path(failure_path, repository),
                    "failure_receipt_sha256": failure_hash,
                    "exit_code": FAILURE_EXIT,
                }
                return FAILURE_EXIT, result
        safe_progress = {
            "complete_games": report["complete_games"],
            "expected_games": EXACT_GAMES,
            "focus_failure": False,
            "pending_games": report["pending_games"],
            "status": "ready" if report["ready"] else "waiting",
        }
        if progress is not None:
            progress(safe_progress)
        if report["ready"]:
            if len(inspected) != EXACT_GAMES:
                raise ArenaWindowError("exact metadata window was not fully safety-inspected")
            command = build_collector_command(
                repository=repository,
                attestation=attestation,
                exclusion_registry=exclusion_registry,
                exclusion_sha256=exclusion_sha256,
                data_root=data_root,
                maximum_workers=maximum_workers,
            )
            collector_result = _run_collector(command, repository=repository, runner=runner)
            verified = dict(
                collector_verifier(
                    collector_result,
                    repository=repository,
                    data_root=data_root,
                    attestation=attestation,
                    exclusion_sha256=exclusion_sha256,
                    expected_game_ids=report["complete_game_ids"],
                )
            )
            receipt = {
                "collected_at_utc": clock(),
                "collector_manifest": {
                    "path": logical_path(pathlib.Path(verified["manifest_path"]), repository),
                    "sha256": verified["manifest_sha256"],
                },
                "editor_attestation": {
                    "path": logical_path(attestation_path, repository),
                    "sha256": attestation_sha256,
                },
                "exact_games": EXACT_GAMES,
                "exclusion_registry": {
                    "path": logical_path(exclusion_registry, repository),
                    "sha256": exclusion_sha256,
                },
                "focus_operational_failures": 0,
                "monitor": {
                    "detail_payload_sha256": [detail_hashes[key] for key in sorted(detail_hashes)],
                    "metadata_payload_sha256": metadata_hashes,
                    "revealed_before_completion": "progress-counts-and-focus-failure-only",
                },
                "schema": COLLECTION_SCHEMA,
                "training_eligible": False,
                "training_forbidden": True,
                "window": attestation["window"],
                "window_plan": {
                    "path": logical_path(plan_path, repository),
                    "sha256": plan_sha256,
                },
            }
            digest, receipt_path = write_content_addressed(
                data_root / "collection_receipts" / attestation["window"]["window_id"],
                receipt,
            )
            return 0, {
                "collection_receipt_path": logical_path(receipt_path, repository),
                "collection_receipt_sha256": digest,
                "exact_games": EXACT_GAMES,
                "manifest_path": receipt["collector_manifest"]["path"],
                "manifest_sha256": receipt["collector_manifest"]["sha256"],
                "schema": COLLECTION_SCHEMA,
                "status": "exact-window-collected",
                "training_eligible": False,
                "window_id": attestation["window"]["window_id"],
            }
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            return 2, {
                "complete_games": report["complete_games"],
                "expected_games": EXACT_GAMES,
                "schema": MONITOR_SNAPSHOT_SCHEMA,
                "status": "timed-out-waiting",
                "window_id": attestation["window"]["window_id"],
            }
        sleeper(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def validate_collection_receipt(
    receipt: Mapping[str, Any],
    *,
    repository: pathlib.Path,
    plan: Mapping[str, Any],
    attestation: Mapping[str, Any],
    data_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    if receipt.get("schema") != COLLECTION_SCHEMA:
        raise ArenaWindowError("unexpected collection receipt schema")
    if (
        receipt.get("window") != attestation["window"]
        or receipt.get("exact_games") != EXACT_GAMES
        or receipt.get("focus_operational_failures") != 0
        or receipt.get("training_eligible") is not False
        or receipt.get("training_forbidden") is not True
    ):
        raise ArenaWindowError("collection receipt violates validation-only exact-window policy")
    plan_ref = receipt.get("window_plan")
    attest_ref = receipt.get("editor_attestation")
    manifest_ref = receipt.get("collector_manifest")
    if not all(isinstance(item, dict) for item in (plan_ref, attest_ref, manifest_ref)):
        raise ArenaWindowError("collection receipt omits provenance references")
    plan_path = resolve_path(plan_ref.get("path"), repository)
    attestation_path = resolve_path(attest_ref.get("path"), repository)
    manifest_path = resolve_path(manifest_ref.get("path"), repository)
    if (
        plan_ref.get("sha256") != sha256_file(plan_path)
        or plan_ref.get("sha256") != sha256_bytes(canonical_json_bytes(plan))
    ):
        raise ArenaWindowError("collection receipt plan reference is corrupt")
    if (
        attest_ref.get("sha256") != sha256_file(attestation_path)
        or attest_ref.get("sha256") != sha256_bytes(canonical_json_bytes(attestation))
    ):
        raise ArenaWindowError("collection receipt attestation reference is corrupt")
    if manifest_ref.get("sha256") != sha256_file(manifest_path):
        raise ArenaWindowError("collection receipt manifest reference is corrupt")
    validate_plan_references(plan, repository=repository)
    validate_attestation(attestation, plan, repository=repository)
    exclusion = receipt.get("exclusion_registry")
    if (
        not isinstance(exclusion, dict)
        or exclusion.get("sha256") != plan["exclusion_registry"]["sha256"]
        or resolve_path(exclusion.get("path"), repository).resolve()
        != resolve_path(plan["exclusion_registry"]["path"], repository).resolve()
    ):
        raise ArenaWindowError("collection receipt exclusion binding contradicts its plan")
    monitor = receipt.get("monitor")
    if not isinstance(monitor, dict) or monitor.get("revealed_before_completion") != "progress-counts-and-focus-failure-only":
        raise ArenaWindowError("collection receipt monitor disclosure is invalid")
    detail_hashes = monitor.get("detail_payload_sha256")
    metadata_hashes = monitor.get("metadata_payload_sha256")
    if (
        not isinstance(detail_hashes, list)
        or len(detail_hashes) != EXACT_GAMES
        or len(set(detail_hashes)) != EXACT_GAMES
        or not isinstance(metadata_hashes, list)
        or not metadata_hashes
        or any(SHA256_RE.fullmatch(str(value)) is None for value in detail_hashes + metadata_hashes)
    ):
        raise ArenaWindowError("collection receipt monitor payload hashes are incomplete")
    if data_root is not None:
        for kind, hashes in (("details", detail_hashes), ("metadata", metadata_hashes)):
            for digest in hashes:
                payload_path = data_root / "monitor" / kind / f"{digest}.json"
                if not payload_path.is_file() or sha256_file(payload_path) != digest:
                    raise ArenaWindowError("collection receipt monitor archive is corrupt")
        manifest = load_content_addressed(
            manifest_path,
            GENERIC_BATCH_SCHEMA,
            expected_sha256=manifest_ref["sha256"],
        )
        expected_ids = [
            (stored.get("record") or {}).get("game_id")
            for stored in manifest.get("games", [])
        ]
        verify_collector_result(
            {
                "coverage": manifest.get("coverage"),
                "manifest_path": logical_path(manifest_path, repository),
                "manifest_sha256": manifest_ref["sha256"],
                "run_id": manifest.get("run_id"),
                "schema": GENERIC_BATCH_SCHEMA,
            },
            repository=repository,
            data_root=data_root,
            attestation=attestation,
            exclusion_sha256=exclusion["sha256"],
            expected_game_ids=expected_ids,
        )
    return dict(receipt)


def derive_validation(
    *,
    plan_path: pathlib.Path,
    plan_sha256: str,
    attestation_path: pathlib.Path,
    attestation_sha256: str,
    collection_receipt_path: pathlib.Path,
    collection_receipt_sha256: str,
    output_root: pathlib.Path,
    repository: pathlib.Path = REPOSITORY,
) -> tuple[str, pathlib.Path, dict[str, Any]]:
    plan = load_content_addressed(plan_path, PLAN_SCHEMA, expected_sha256=plan_sha256)
    validate_plan_references(plan, repository=repository)
    attestation = load_content_addressed(
        attestation_path, ATTESTATION_SCHEMA, expected_sha256=attestation_sha256
    )
    receipt = load_content_addressed(
        collection_receipt_path,
        COLLECTION_SCHEMA,
        expected_sha256=collection_receipt_sha256,
    )
    validate_collection_receipt(
        receipt,
        repository=repository,
        plan=plan,
        attestation=attestation,
        data_root=collection_receipt_path.resolve().parents[2],
    )
    if attestation["window"]["role"] != "arena-validation":
        raise ArenaWindowError("only the planned hybrid validation window can be derived")
    manifest_ref = receipt["collector_manifest"]
    manifest_path = resolve_path(manifest_ref["path"], repository)
    manifest = load_content_addressed(
        manifest_path,
        GENERIC_BATCH_SCHEMA,
        expected_sha256=manifest_ref["sha256"],
    )
    validation_records: list[dict[str, Any]] = []
    rejected = 0
    for stored in manifest.get("games", []):
        record = stored.get("record") or {}
        clean = (
            record.get("status") == "accepted"
            and (record.get("operational") or {}).get("classification") == "clean"
            and ((record.get("replay") or {}).get("rules_validation") or {}).get("status")
            == "terminal-valid"
        )
        if not clean:
            rejected += 1
        validation_records.append(
            {
                "game_id": record.get("game_id"),
                "record_sha256": stored.get("record_sha256"),
                "validation_eligible": clean,
            }
        )
    if len(validation_records) != EXACT_GAMES:
        raise ArenaWindowError("validation derivation does not bind exactly 90 records")
    payload = {
        "arena_manifest": dict(manifest_ref),
        "collection_receipt": {
            "path": logical_path(collection_receipt_path, repository),
            "sha256": collection_receipt_sha256,
        },
        "created_at_utc": utc_now(),
        "fresh_arena_usage": {
            "action_ranking_games": 0,
            "policy_rows": 0,
            "training_games": 0,
            "value_rows": 0,
            "validation_games": EXACT_GAMES - rejected,
        },
        "records": validation_records,
        "rejected_validation_games": rejected,
        "schema": DERIVATION_SCHEMA,
        "training_eligible": False,
        "training_forbidden": True,
        "training_forbidden_reason": "window was preregistered as arena-validation before results",
        "window": attestation["window"],
    }
    digest, path = write_content_addressed(
        output_root / "derivations" / attestation["window"]["window_id"], payload
    )
    return digest, path, payload


def check_artifact(path: pathlib.Path, *, repository: pathlib.Path) -> dict[str, Any]:
    value = load_json(path)
    if path.stem != sha256_file(path) or path.read_bytes() != canonical_json_bytes(value):
        raise ArenaWindowError("checked artifact is not canonical/content-addressed")
    schema = value.get("schema")
    schema_set = {
        PLAN_SCHEMA,
        ATTESTATION_SCHEMA,
        MONITOR_SNAPSHOT_SCHEMA,
        COLLECTION_SCHEMA,
        DERIVATION_SCHEMA,
    }
    if schema not in schema_set:
        raise ArenaWindowError(f"unsupported artifact schema: {schema!r}")
    if schema == PLAN_SCHEMA:
        validate_plan_references(value, repository=repository)
    elif schema == ATTESTATION_SCHEMA:
        plan_ref = value.get("window_plan") or {}
        plan_path = resolve_path(plan_ref.get("path"), repository)
        plan = load_content_addressed(
            plan_path, PLAN_SCHEMA, expected_sha256=plan_ref.get("sha256")
        )
        validate_plan_references(plan, repository=repository)
        validate_attestation(value, plan, repository=repository)
    elif schema == MONITOR_SNAPSHOT_SCHEMA:
        attest_ref = value.get("editor_attestation")
        detail_hash = checked_sha(
            value.get("detail_payload_sha256"), "monitor detail SHA-256"
        )
        if (
            value.get("status") != "focus-operational-failure"
            or value.get("category")
            not in {"timeout", "illegal-action", "crash", "malformed-transcript"}
            or value.get("rollback_required") is not True
            or not isinstance(value.get("game_id"), int)
            or not isinstance(attest_ref, dict)
        ):
            raise ArenaWindowError("focus-failure monitor receipt is invalid")
        data_root = path.resolve().parents[2]
        detail_path = data_root / "monitor" / "details" / f"{detail_hash}.json"
        if not detail_path.is_file() or sha256_file(detail_path) != detail_hash:
            raise ArenaWindowError("focus-failure detail archive is corrupt")
        attestation_path = resolve_path(attest_ref.get("path"), repository)
        attestation = load_content_addressed(
            attestation_path,
            ATTESTATION_SCHEMA,
            expected_sha256=attest_ref.get("sha256"),
        )
        if value.get("window_id") != attestation["window"]["window_id"]:
            raise ArenaWindowError("focus-failure receipt contradicts its attestation")
    elif schema == COLLECTION_SCHEMA:
        plan_ref = value.get("window_plan") or {}
        attest_ref = value.get("editor_attestation") or {}
        plan = load_content_addressed(
            resolve_path(plan_ref.get("path"), repository),
            PLAN_SCHEMA,
            expected_sha256=plan_ref.get("sha256"),
        )
        validate_plan_references(plan, repository=repository)
        attestation = load_content_addressed(
            resolve_path(attest_ref.get("path"), repository),
            ATTESTATION_SCHEMA,
            expected_sha256=attest_ref.get("sha256"),
        )
        validate_collection_receipt(
            value,
            repository=repository,
            plan=plan,
            attestation=attestation,
            data_root=path.resolve().parents[2],
        )
    else:
        collection_ref = value.get("collection_receipt")
        manifest_ref = value.get("arena_manifest")
        records = value.get("records")
        if (
            value.get("training_eligible") is not False
            or value.get("training_forbidden") is not True
            or (value.get("fresh_arena_usage") or {}).get("training_games") != 0
            or (value.get("fresh_arena_usage") or {}).get("value_rows") != 0
            or (value.get("fresh_arena_usage") or {}).get("action_ranking_games") != 0
            or (value.get("fresh_arena_usage") or {}).get("policy_rows") != 0
            or not isinstance(collection_ref, dict)
            or not isinstance(manifest_ref, dict)
            or not isinstance(records, list)
            or len(records) != EXACT_GAMES
        ):
            raise ArenaWindowError("derivation permits forbidden training use")
        seen_ids: set[int] = set()
        for record in records:
            if (
                not isinstance(record, dict)
                or set(record) != {"game_id", "record_sha256", "validation_eligible"}
                or isinstance(record.get("game_id"), bool)
                or not isinstance(record.get("game_id"), int)
                or record["game_id"] in seen_ids
                or SHA256_RE.fullmatch(str(record.get("record_sha256"))) is None
                or not isinstance(record.get("validation_eligible"), bool)
            ):
                raise ArenaWindowError("validation derivation records are invalid")
            seen_ids.add(record["game_id"])
        collection_path = resolve_path(collection_ref.get("path"), repository)
        collection = load_content_addressed(
            collection_path,
            COLLECTION_SCHEMA,
            expected_sha256=collection_ref.get("sha256"),
        )
        if collection.get("collector_manifest") != manifest_ref:
            raise ArenaWindowError("derivation manifest contradicts its collection receipt")
        plan_ref = collection.get("window_plan") or {}
        attest_ref = collection.get("editor_attestation") or {}
        plan = load_content_addressed(
            resolve_path(plan_ref.get("path"), repository),
            PLAN_SCHEMA,
            expected_sha256=plan_ref.get("sha256"),
        )
        validate_plan_references(plan, repository=repository)
        attestation = load_content_addressed(
            resolve_path(attest_ref.get("path"), repository),
            ATTESTATION_SCHEMA,
            expected_sha256=attest_ref.get("sha256"),
        )
        validate_collection_receipt(
            collection,
            repository=repository,
            plan=plan,
            attestation=attestation,
            data_root=collection_path.resolve().parents[2],
        )
        manifest = load_content_addressed(
            resolve_path(manifest_ref.get("path"), repository),
            GENERIC_BATCH_SCHEMA,
            expected_sha256=manifest_ref.get("sha256"),
        )
        expected_records = []
        for stored in manifest.get("games", []):
            raw = stored.get("record") or {}
            clean = (
                raw.get("status") == "accepted"
                and (raw.get("operational") or {}).get("classification") == "clean"
                and ((raw.get("replay") or {}).get("rules_validation") or {}).get("status")
                == "terminal-valid"
            )
            expected_records.append(
                {
                    "game_id": raw.get("game_id"),
                    "record_sha256": stored.get("record_sha256"),
                    "validation_eligible": clean,
                }
            )
        usage = value.get("fresh_arena_usage") or {}
        eligible = sum(item["validation_eligible"] for item in expected_records)
        if (
            records != expected_records
            or value.get("window") != attestation["window"]
            or value.get("rejected_validation_games") != EXACT_GAMES - eligible
            or usage.get("validation_games") != eligible
        ):
            raise ArenaWindowError("validation derivation contradicts its exact arena archive")
    return {"path": logical_path(path, repository), "schema": schema, "sha256": sha256_file(path), "status": "ok"}


def _safe_progress(report: Mapping[str, Any]) -> None:
    print(
        f"arena-window complete={report['complete_games']}/{report['expected_games']} "
        f"pending={report['pending_games']} focus_failure=false",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--campaign", type=pathlib.Path, default=DEFAULT_CAMPAIGN)
    plan.add_argument("--planned-at-utc", default=utc_now())
    plan.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)

    attest = commands.add_parser("attest")
    attest.add_argument("--plan", type=pathlib.Path, required=True)
    attest.add_argument("--plan-sha256", required=True)
    attest.add_argument("--window-id", choices=(VALIDATION_WINDOW_ID, ROLLBACK_WINDOW_ID), required=True)
    attest.add_argument("--generated-source", type=pathlib.Path, required=True)
    attest.add_argument("--copied-back-source", type=pathlib.Path, required=True)
    attest.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    attest.add_argument("--repository-commit", required=True)
    attest.add_argument("--agent-id", type=int, required=True)
    attest.add_argument("--submission-id", type=int, required=True)
    attest.add_argument("--play-checked-at-utc", required=True)
    attest.add_argument("--uploaded-at-utc", required=True)
    attest.add_argument("--created-at-utc", default=utc_now())
    attest.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    for flag in sorted(PREFLIGHT_KEYS):
        attest.add_argument(f"--{flag.replace('_', '-')}-ok", action="store_true")
    attest.add_argument("--play-stdout-legal", action="store_true")
    attest.add_argument("--play-telemetry-ok", action="store_true")

    watch = commands.add_parser("watch-collect")
    watch.add_argument("--plan", type=pathlib.Path, required=True)
    watch.add_argument("--plan-sha256", required=True)
    watch.add_argument("--attestation", type=pathlib.Path, required=True)
    watch.add_argument("--attestation-sha256", required=True)
    watch.add_argument("--exclusion-registry", type=pathlib.Path, required=True)
    watch.add_argument("--exclusion-registry-sha256", required=True)
    watch.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    watch.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    watch.add_argument("--poll-seconds", type=float, default=10.0)
    watch.add_argument("--timeout-seconds", type=float, default=3600.0)
    watch.add_argument("--request-timeout-seconds", type=float, default=30.0)
    watch.add_argument("--maximum-workers", type=int, default=2)

    derive = commands.add_parser("derive")
    derive.add_argument("--plan", type=pathlib.Path, required=True)
    derive.add_argument("--plan-sha256", required=True)
    derive.add_argument("--attestation", type=pathlib.Path, required=True)
    derive.add_argument("--attestation-sha256", required=True)
    derive.add_argument("--collection-receipt", type=pathlib.Path, required=True)
    derive.add_argument("--collection-receipt-sha256", required=True)
    derive.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    derive.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)

    check = commands.add_parser("check")
    check.add_argument("--artifact", type=pathlib.Path, action="append", required=True)
    check.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            digest, path, payload = create_plan(
                campaign_path=arguments.campaign,
                planned_at_utc=arguments.planned_at_utc,
                output_root=arguments.output_root,
                repository=arguments.repository,
            )
            result = {"path": logical_path(path, arguments.repository), "sha256": digest, "schema": payload["schema"]}
            code = 0
        elif arguments.command == "attest":
            preflight = {
                key: bool(getattr(arguments, f"{key}_ok")) for key in PREFLIGHT_KEYS
            }
            digest, path, payload = create_attestation(
                plan_path=arguments.plan,
                plan_sha256=arguments.plan_sha256,
                window_id=arguments.window_id,
                generated_source=arguments.generated_source,
                copied_back_source=arguments.copied_back_source,
                repository=arguments.repository,
                repository_commit=arguments.repository_commit,
                agent_id=arguments.agent_id,
                submission_id=arguments.submission_id,
                play_checked_at_utc=arguments.play_checked_at_utc,
                uploaded_at_utc=arguments.uploaded_at_utc,
                created_at_utc=arguments.created_at_utc,
                preflight=preflight,
                play_stdout_legal=arguments.play_stdout_legal,
                play_telemetry_ok=arguments.play_telemetry_ok,
                output_root=arguments.output_root,
            )
            result = {"path": logical_path(path, arguments.repository), "sha256": digest, "schema": payload["schema"], "source": payload["source"], "upload_bytes_disclosure": DISCLOSURE}
            code = 0
        elif arguments.command == "watch-collect":
            code, result = watch_collect(
                plan_path=arguments.plan,
                plan_sha256=arguments.plan_sha256,
                attestation_path=arguments.attestation,
                attestation_sha256=arguments.attestation_sha256,
                exclusion_registry=arguments.exclusion_registry,
                exclusion_sha256=arguments.exclusion_registry_sha256,
                data_root=arguments.data_root,
                repository=arguments.repository,
                poll_seconds=arguments.poll_seconds,
                timeout_seconds=arguments.timeout_seconds,
                request_timeout_seconds=arguments.request_timeout_seconds,
                maximum_workers=arguments.maximum_workers,
                progress=_safe_progress,
            )
        elif arguments.command == "derive":
            digest, path, payload = derive_validation(
                plan_path=arguments.plan,
                plan_sha256=arguments.plan_sha256,
                attestation_path=arguments.attestation,
                attestation_sha256=arguments.attestation_sha256,
                collection_receipt_path=arguments.collection_receipt,
                collection_receipt_sha256=arguments.collection_receipt_sha256,
                output_root=arguments.output_root,
                repository=arguments.repository,
            )
            result = {"path": logical_path(path, arguments.repository), "sha256": digest, "schema": payload["schema"], "fresh_arena_usage": payload["fresh_arena_usage"], "training_eligible": False}
            code = 0
        else:
            checks = [check_artifact(path, repository=arguments.repository) for path in arguments.artifact]
            result = {"checked": checks, "status": "ok"}
            code = 0
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return code
    except (ArenaWindowError, OSError, subprocess.SubprocessError, urllib.error.URLError) as error:
        print(f"arena-window failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
