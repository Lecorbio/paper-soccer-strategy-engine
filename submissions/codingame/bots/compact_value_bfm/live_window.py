#!/usr/bin/env python3
"""Collect the exact diagnostic 90-game compact-value-BFM live window.

This wrapper never uploads, rolls back, trains, or promotes anything.  It binds
the sealed one-upload submission attestation, a separately frozen pre-upload
ID-only exclusion registry, and the generic append-only arena collector.  A
focus operational failure rejects the diagnostic window but deliberately does
not stop collection before the complete matching 90-game window is archived.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
QUALIFICATION_PATH = REPOSITORY / "tools/compact_value_bfm_qualification.py"
GENERIC_COLLECTOR_PATH = (
    REPOSITORY / "submissions/codingame/tools/collect_arena_batch.py"
)
REPLAY_FEATURES_PATH = REPOSITORY / "tools/jacek_replay_features.py"
OPENINGS_PATH = REPOSITORY / "tools/compact_value_bfm_openings.py"

NAMESPACE = "compact_value_bfm"
EXCLUSION_BINDING_SCHEMA = (
    "papersoccer.compact-value-bfm.live-exclusion-binding.v1"
)
MONITOR_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.live-monitor-receipt.v1"
MONITOR_GAME_SCHEMA = "papersoccer.compact-value-bfm.live-monitor-game.v1"
WAIT_SNAPSHOT_SCHEMA = "papersoccer.compact-value-bfm.live-wait-snapshot.v1"
WINDOW_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.live-window-diagnostic.v1"
WINDOW_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm.live-window-reference.v1"
LIVE_FINGERPRINT_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.verified-live-canonical-fingerprints.v1"
)
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"
GENERIC_BATCH_SCHEMA = "papersoccer.codingame-arena-batch.v1"
GENERIC_GAME_SCHEMA = "papersoccer.codingame-arena-game.v1"
GENERIC_BINDING_SCHEMA = "papersoccer.codingame-arena-binding.v1"
EXACT_GAMES = 90
SOURCE_LIMIT = 95_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
UTC_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
)
FORBIDDEN_REGISTRY_KEYS = frozenset({
    "agents", "frames", "gameinformation", "inputs", "outputs", "replay",
    "stderr", "stdin", "stdout", "transcript", "turns",
})
FAILURE_PATTERNS = (
    ("timeout", re.compile(
        r"\btime[ -]?out\b|timed out|exceeded (?:the )?time limit|"
        r"too long to respond", re.I
    )),
    ("illegal-action", re.compile(
        r"illegal (?:move|action|edge)|invalid (?:move|action|edge)|"
        r"move is not legal|action is not legal", re.I
    )),
    ("crash", re.compile(
        r"runtime error|segmentation fault|uncaught exception|"
        r"terminated by signal|process (?:was )?killed|out of memory", re.I
    )),
    ("malformed-output", re.compile(
        r"invalid output|malformed output|unrecognized output|could not parse|"
        r"provided no output|empty output", re.I
    )),
)
TEXT_FIELDS = ("stderr", "gameInformation", "summary", "tooltip", "tooltips")


class LiveWindowError(ValueError):
    """A live-window input or append-only artifact violates the contract."""


def _load_module(name: str, path: pathlib.Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise LiveWindowError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def qualification_module() -> Any:
    return _load_module(
        "compact_value_bfm_live_qualification", QUALIFICATION_PATH
    )


def generic_collector_module() -> Any:
    return _load_module(
        "compact_value_bfm_live_generic_collector", GENERIC_COLLECTOR_PATH
    )


def replay_features_module() -> Any:
    return _load_module(
        "compact_value_bfm_live_replay_features", REPLAY_FEATURES_PATH
    )


def opening_fingerprints_module() -> Any:
    return _load_module(
        "compact_value_bfm_live_opening_fingerprints", OPENINGS_PATH
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=True, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def parse_utc(value: object, field: str) -> dt.datetime:
    if not isinstance(value, str) or UTC_PATTERN.fullmatch(value) is None:
        raise LiveWindowError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        result = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise LiveWindowError(f"{field} is invalid") from error
    if result.tzinfo != dt.timezone.utc:
        raise LiveWindowError(f"{field} must be UTC")
    return result


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    if "body_sha256" in body:
        raise LiveWindowError("body_sha256 is reserved")
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def validate_seal(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or not valid_sha256(claimed)
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise LiveWindowError(f"{label} body SHA-256 is invalid")
    return dict(value)


def write_once(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise LiveWindowError(f"append-only artifact collision: {path}")
        return
    temporary: pathlib.Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = pathlib.Path(raw)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise LiveWindowError(f"append-only artifact raced: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_sealed(path: pathlib.Path, body: Mapping[str, Any]) -> dict[str, Any]:
    document = seal(body)
    write_once(path, canonical_json_bytes(document))
    return document


def write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str = ".json"
) -> tuple[pathlib.Path, dict[str, Any]]:
    document = seal(body)
    payload = canonical_json_bytes(document)
    path = directory / f"{sha256_bytes(payload)}{suffix}"
    write_once(path, payload)
    return path, document


def load_sealed(path: pathlib.Path, schema: str, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LiveWindowError(f"could not load {label}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise LiveWindowError(f"{label} is not canonical JSON")
    return validate_seal(value, schema, label)


def logical_path(path: pathlib.Path, repository: pathlib.Path = REPOSITORY) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_path(raw: object, repository: pathlib.Path = REPOSITORY) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise LiveWindowError("artifact path is missing")
    path = pathlib.Path(raw)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def artifact_reference(path: pathlib.Path, schema: str) -> dict[str, Any]:
    value = load_sealed(path, schema, path.name)
    return {
        "path": logical_path(path),
        "sha256": sha256_file(path),
        "body_sha256": value["body_sha256"],
    }


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_detail_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise LiveWindowError("ID-only registry contains a non-string key")
            if _canonical_key(key) in FORBIDDEN_REGISTRY_KEYS:
                raise LiveWindowError(
                    f"ID-only registry contains forbidden replay field {key}"
                )
            _reject_detail_payload(child)
    elif isinstance(value, list):
        for child in value:
            _reject_detail_payload(child)


def validate_id_only_registry(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        registry = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LiveWindowError("could not load exclusion registry") from error
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != EXCLUSION_SCHEMA
        or raw != canonical_json_bytes(registry)
        or not isinstance(registry.get("records"), list)
    ):
        raise LiveWindowError("exclusion registry is not canonical ID-only JSON")
    _reject_detail_payload(registry)
    seen: set[int] = set()
    for record in registry["records"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"categories", "game_id", "sources"}
            or isinstance(record.get("game_id"), bool)
            or not isinstance(record.get("game_id"), int)
            or record["game_id"] <= 0
            or record["game_id"] in seen
            or not isinstance(record.get("categories"), list)
            or not all(isinstance(item, str) for item in record["categories"])
            or not isinstance(record.get("sources"), list)
            or not all(isinstance(item, str) for item in record["sources"])
        ):
            raise LiveWindowError("exclusion registry record is not ID-only")
        seen.add(record["game_id"])
    return registry


def freeze_exclusion_binding(
    output: pathlib.Path,
    *,
    registry_path: pathlib.Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    registry = validate_id_only_registry(registry_path)
    parse_utc(frozen_at_utc, "exclusion freeze time")
    return write_sealed(output, {
        "schema": EXCLUSION_BINDING_SCHEMA,
        "namespace": NAMESPACE,
        "frozen_at_utc": frozen_at_utc,
        "registry": {
            "path": logical_path(registry_path),
            "sha256": sha256_file(registry_path),
            "bytes": registry_path.stat().st_size,
            "schema": EXCLUSION_SCHEMA,
            "game_ids": len(registry["records"]),
        },
        "content_scope": "game-ids-categories-and-source-identifiers-only",
        "replay_payloads_read": False,
        "created_before_upload_required": True,
    })


def validate_exclusion_binding(
    path: pathlib.Path,
    *,
    submitted_at_utc: str | None = None,
) -> tuple[dict[str, Any], pathlib.Path]:
    binding = load_sealed(path, EXCLUSION_BINDING_SCHEMA, "exclusion binding")
    registry = binding.get("registry")
    if (
        set(binding) != {
            "schema", "namespace", "frozen_at_utc", "registry",
            "content_scope", "replay_payloads_read",
            "created_before_upload_required", "body_sha256",
        }
        or binding.get("namespace") != NAMESPACE
        or binding.get("content_scope")
        != "game-ids-categories-and-source-identifiers-only"
        or binding.get("replay_payloads_read") is not False
        or binding.get("created_before_upload_required") is not True
        or not isinstance(registry, dict)
        or set(registry) != {
            "path", "sha256", "bytes", "schema", "game_ids"
        }
        or registry.get("schema") != EXCLUSION_SCHEMA
        or not valid_sha256(registry.get("sha256"))
    ):
        raise LiveWindowError("exclusion binding policy changed")
    registry_path = resolve_path(registry["path"])
    payload = validate_id_only_registry(registry_path)
    if (
        sha256_file(registry_path) != registry["sha256"]
        or registry_path.stat().st_size != registry["bytes"]
        or len(payload["records"]) != registry["game_ids"]
    ):
        raise LiveWindowError("bound exclusion registry changed")
    frozen = parse_utc(binding.get("frozen_at_utc"), "exclusion freeze time")
    if submitted_at_utc is not None and frozen >= parse_utc(
        submitted_at_utc, "submission time"
    ):
        raise LiveWindowError("exclusion registry was not frozen before upload")
    return binding, registry_path


@dataclasses.dataclass(frozen=True)
class LiveIdentity:
    agent_id: int
    submission_id: int
    repository_commit: str
    source_sha256: str
    source_bytes: int
    source_path: pathlib.Path
    submitted_at_utc: str
    attestation_path: pathlib.Path
    authorization_path: pathlib.Path


def identity_record(identity: LiveIdentity) -> dict[str, Any]:
    return dataclasses.asdict(identity) | {
        "source_path": logical_path(identity.source_path),
        "attestation_path": logical_path(identity.attestation_path),
        "authorization_path": logical_path(identity.authorization_path),
    }


def load_live_identity(
    attestation_path: pathlib.Path,
    exclusion_binding_path: pathlib.Path,
) -> tuple[LiveIdentity, dict[str, Any], pathlib.Path]:
    qualification = qualification_module()
    try:
        attestation = qualification.load_sealed(
            attestation_path, qualification.UPLOAD_EVENT_SCHEMA
        )
    except Exception as error:
        raise LiveWindowError("submission attestation is invalid") from error
    authorization_ref = attestation.get("authorization")
    if (
        attestation.get("namespace") != NAMESPACE
        or attestation.get("status") != "submission-attested"
        or attestation.get("submit_clicks") != 1
        or not isinstance(attestation.get("agent_id"), int)
        or isinstance(attestation.get("agent_id"), bool)
        or attestation["agent_id"] <= 0
        or not isinstance(attestation.get("submission_id"), int)
        or isinstance(attestation.get("submission_id"), bool)
        or attestation["submission_id"] <= 0
        or not COMMIT_PATTERN.fullmatch(str(attestation.get("candidate_commit", "")))
        or not valid_sha256(attestation.get("source_sha256"))
        or not isinstance(attestation.get("source_bytes"), int)
        or not 0 < attestation["source_bytes"] <= SOURCE_LIMIT
        or not isinstance(authorization_ref, dict)
        or set(authorization_ref) != {"path", "sha256"}
    ):
        raise LiveWindowError("submission attestation identity changed")
    submitted_at = str(attestation.get("submitted_at_utc", ""))
    parse_utc(submitted_at, "submission time")
    authorization_path = resolve_path(authorization_ref["path"])
    try:
        authorization = qualification.load_sealed(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        )
    except Exception as error:
        raise LiveWindowError("upload authorization is invalid") from error
    candidate = authorization.get("candidate")
    if (
        sha256_file(authorization_path) != authorization_ref["sha256"]
        or authorization.get("namespace") != NAMESPACE
        or authorization.get("uploads_authorized") != 1
        or authorization.get("rank4_replacement_authorized") is not False
        or authorization.get("candidate_commit") != attestation["candidate_commit"]
        or not isinstance(candidate, dict)
        or candidate.get("sha256") != attestation["source_sha256"]
        or candidate.get("bytes") != attestation["source_bytes"]
        or not isinstance(candidate.get("path"), str)
    ):
        raise LiveWindowError("submission attestation contradicts authorization")
    source_path = pathlib.Path(candidate["path"]).resolve()
    source = source_path.read_bytes()
    try:
        source.decode("ascii")
    except UnicodeDecodeError as error:
        raise LiveWindowError("attested source is not ASCII") from error
    if (
        len(source) != attestation["source_bytes"]
        or sha256_bytes(source) != attestation["source_sha256"]
    ):
        raise LiveWindowError("attested source bytes changed")
    exclusion, registry_path = validate_exclusion_binding(
        exclusion_binding_path, submitted_at_utc=submitted_at
    )
    return LiveIdentity(
        agent_id=attestation["agent_id"],
        submission_id=attestation["submission_id"],
        repository_commit=attestation["candidate_commit"],
        source_sha256=attestation["source_sha256"],
        source_bytes=attestation["source_bytes"],
        source_path=source_path,
        submitted_at_utc=submitted_at,
        attestation_path=attestation_path.resolve(),
        authorization_path=authorization_path,
    ), exclusion, registry_path


def classify_matching_window(
    battles: Any, *, agent_id: int, submission_id: int
) -> dict[str, Any]:
    try:
        report = qualification_module().classify_matching_window(
            battles, agent_id=agent_id, submission_id=submission_id
        )
    except Exception as error:
        raise LiveWindowError(str(error)) from error
    return {
        **report,
        "overfull": False,
        "ready": report["collector_permitted"],
        "complete_game_ids": report["game_ids"],
    }


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for child in value for text in _text_values(child)]
    if isinstance(value, Mapping):
        return [text for child in value.values() for text in _text_values(child)]
    return []


def _frame_player(frame: Mapping[str, Any], texts: Sequence[str]) -> int | None:
    raw = frame.get("agentId")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw in (0, 1):
        return raw
    scoped = {
        int(match) for text in texts for match in re.findall(r"\$([01])\b", text)
    }
    return next(iter(scoped)) if len(scoped) == 1 else None


def classify_operational_detail(
    detail: Any, *, game_id: int, focus_agent_id: int
) -> dict[str, Any]:
    malformed = {
        "game_id": game_id,
        "focus_status": "malformed-detail",
        "opponent_status": "unknown",
        "focus_failure": True,
        "opponent_failure": False,
    }
    if not isinstance(detail, Mapping) or detail.get("gameId") != game_id:
        return malformed
    agents = detail.get("agents")
    frames = detail.get("frames")
    if not isinstance(agents, list) or len(agents) != 2 or not isinstance(frames, list):
        return malformed
    focus_indexes = [
        row.get("index") for row in agents
        if isinstance(row, Mapping) and row.get("agentId") == focus_agent_id
    ]
    if len(focus_indexes) != 1 or focus_indexes[0] not in (0, 1):
        return malformed
    focus_player = int(focus_indexes[0])
    statuses = {0: "ok", 1: "ok"}
    for frame in frames:
        if not isinstance(frame, Mapping):
            return malformed
        texts = [
            text for field in TEXT_FIELDS if field in frame
            for text in _text_values(frame[field])
        ]
        player = _frame_player(frame, texts)
        for text in texts:
            for category, pattern in FAILURE_PATTERNS:
                if pattern.search(text):
                    if player is None:
                        return malformed
                    statuses[player] = category
                    break
        if player in (0, 1) and "stdout" in frame:
            output = "" if frame.get("stdout") is None else str(frame["stdout"]).strip()
            if not output or any(character not in "01234567" for character in output):
                statuses[player] = "malformed-output"
    focus = statuses[focus_player]
    opponent = statuses[1 - focus_player]
    return {
        "game_id": game_id,
        "focus_status": focus,
        "opponent_status": opponent,
        "focus_failure": focus != "ok",
        "opponent_failure": opponent != "ok",
    }


def _archive_monitor_payload(
    data_root: pathlib.Path,
    kind: str,
    payload: Any,
    *,
    fetched_at_utc: str,
    request_identity: Mapping[str, Any],
) -> tuple[pathlib.Path, str]:
    raw = canonical_json_bytes(payload)
    digest = sha256_bytes(raw)
    raw_path = data_root / "raw/monitor" / kind / f"{digest}.json"
    write_once(raw_path, raw)
    write_content_addressed(data_root / "receipts/monitor" / kind, {
        "schema": MONITOR_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "kind": kind,
        "fetched_at_utc": fetched_at_utc,
        "request": dict(request_identity),
        "raw": {
            "path": logical_path(raw_path),
            "sha256": digest,
            "bytes": len(raw),
        },
        "diagnostic_only": True,
        "training_eligible": False,
    })
    return raw_path, digest


def _monitor_game(
    data_root: pathlib.Path,
    *,
    game_id: int,
    focus_agent_id: int,
    fetch_detail: Callable[[int], Any],
    detail_classifier: Callable[..., Mapping[str, Any]],
    clock: Callable[[], str],
) -> dict[str, Any]:
    stable_path = data_root / "monitor/games" / f"{game_id}.json"
    if stable_path.exists():
        return load_sealed(stable_path, MONITOR_GAME_SCHEMA, "monitor game")
    detail = fetch_detail(game_id)
    fetched = clock()
    _raw_path, detail_sha = _archive_monitor_payload(
        data_root,
        "detail",
        detail,
        fetched_at_utc=fetched,
        request_identity={"game_id": game_id},
    )
    classification = dict(detail_classifier(
        detail, game_id=game_id, focus_agent_id=focus_agent_id
    ))
    expected = {
        "game_id", "focus_status", "opponent_status",
        "focus_failure", "opponent_failure",
    }
    if set(classification) != expected or classification["game_id"] != game_id:
        raise LiveWindowError("detail classifier returned an invalid result")
    return write_sealed(stable_path, {
        "schema": MONITOR_GAME_SCHEMA,
        "namespace": NAMESPACE,
        "game_id": game_id,
        "detail_payload_sha256": detail_sha,
        "classification": classification,
        "inspected_at_utc": fetched,
        "detail_text_disclosed_in_progress": False,
    })


class _FilteredArenaApi:
    def __init__(self, base: Any, shared: Any, expected_game_ids: Sequence[int]) -> None:
        self.base = base
        self.shared = shared
        self.expected = set(expected_game_ids)
        self.battle_service = shared.REQUEST_SCHEMAS["agent-battles-v1"]["service"]

    def post(self, service: str, payload: Any) -> Any:
        response = self.base.post(service, payload)
        if service != self.battle_service:
            return response
        try:
            rows = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LiveWindowError("collector battle response is invalid") from error
        if not isinstance(rows, list):
            raise LiveWindowError("collector battle response is not a list")
        filtered = [
            row for row in rows if isinstance(row, Mapping)
            and row.get("gameId") in self.expected
        ]
        observed = {row["gameId"] for row in filtered}
        if observed != self.expected or len(filtered) != EXACT_GAMES:
            raise LiveWindowError("collector could not isolate the exact monitored IDs")
        return self.shared.ApiResponse(
            body=self.shared.canonical_json_bytes(filtered),
            status=response.status,
            headers=response.headers,
            attempts=response.attempts,
        )


def collect_generic_window(
    *,
    identity: LiveIdentity,
    registry_path: pathlib.Path,
    registry_sha256: str,
    data_root: pathlib.Path,
    expected_game_ids: Sequence[int],
    maximum_workers: int,
) -> dict[str, Any]:
    generic = generic_collector_module()
    base_api = generic.shared.PublicApi()
    collector = generic.ArenaBatchCollector(
        repository=REPOSITORY,
        data_root=data_root,
        api=_FilteredArenaApi(base_api, generic.shared, expected_game_ids),
        exclusion_registry_path=registry_path,
        exclusion_registry_sha256=registry_sha256,
        maximum_workers=maximum_workers,
    )
    binding = collector.bind_source(
        agent_id=identity.agent_id,
        submission_id=identity.submission_id,
        source_path=identity.source_path,
        expected_source_sha256=identity.source_sha256,
        repository_commit=identity.repository_commit,
    )
    return collector.collect(
        run_id=f"compact-value-bfm-live-{identity.submission_id}",
        binding=binding,
        expected_games=EXACT_GAMES,
    )


def verify_generic_result(
    collector_result: Mapping[str, Any],
    *,
    identity: LiveIdentity,
    registry_sha256: str,
    expected_game_ids: Sequence[int],
) -> dict[str, Any]:
    generic = generic_collector_module()
    manifest: dict[str, Any]
    manifest_path: pathlib.Path | None = None
    if isinstance(collector_result.get("manifest"), Mapping):
        manifest = dict(collector_result["manifest"])
        claimed = collector_result.get("manifest_sha256")
        if claimed is not None and claimed != sha256_bytes(canonical_json_bytes(manifest)):
            raise LiveWindowError("injected collector manifest hash changed")
    else:
        raw_path = collector_result.get("manifest_path")
        if not isinstance(raw_path, str):
            raise LiveWindowError("generic collector returned no manifest")
        manifest_path = resolve_path(raw_path)
        try:
            manifest = generic.validate_export_manifest(
                manifest_path, registry_sha256, repository=REPOSITORY
            )
        except Exception as error:
            raise LiveWindowError("generic collector archive failed validation") from error
        if collector_result.get("manifest_sha256") != sha256_file(manifest_path):
            raise LiveWindowError("generic collector manifest SHA-256 changed")
    coverage = manifest.get("coverage")
    binding = manifest.get("binding")
    exclusion = manifest.get("exclusion_registry")
    games = manifest.get("games")
    if (
        manifest.get("schema") != GENERIC_BATCH_SCHEMA
        or not isinstance(coverage, Mapping)
        or coverage.get("expected_games") != EXACT_GAMES
        or coverage.get("battle_window_games") != EXACT_GAMES
        or coverage.get("accepted_games") != EXACT_GAMES
        or coverage.get("full_window_accounted") is not True
        or not isinstance(binding, Mapping)
        or binding.get("schema") != GENERIC_BINDING_SCHEMA
        or binding.get("agent_id") != identity.agent_id
        or binding.get("asserted_submission_id") != identity.submission_id
        or binding.get("repository_commit") != identity.repository_commit
        or (binding.get("source") or {}).get("sha256") != identity.source_sha256
        or not isinstance(exclusion, Mapping)
        or exclusion.get("sha256") != registry_sha256
        or not isinstance(games, list)
        or len(games) != EXACT_GAMES
    ):
        raise LiveWindowError("generic collector manifest contradicts live identity")
    expected = set(expected_game_ids)
    observed: set[int] = set()
    records = []
    for stored in games:
        record = stored.get("record") if isinstance(stored, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or record.get("schema") != GENERIC_GAME_SCHEMA
            or record.get("status") != "accepted"
            or record.get("source_sha256") != identity.source_sha256
            or (record.get("focus") or {}).get("agent_id") != identity.agent_id
            or (record.get("focus") or {}).get("submission_id")
            != identity.submission_id
            or not isinstance(record.get("operational"), Mapping)
        ):
            raise LiveWindowError("generic collector game record is misbound")
        game_id = record.get("game_id")
        if (
            isinstance(game_id, bool) or not isinstance(game_id, int)
            or game_id in observed
        ):
            raise LiveWindowError("generic collector repeats a game ID")
        observed.add(game_id)
        records.append(dict(record))
    if observed != expected:
        raise LiveWindowError("generic collector IDs differ from monitored window")
    return {
        "manifest": manifest,
        "manifest_path": None if manifest_path is None else manifest_path,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "records": sorted(records, key=lambda item: item["game_id"]),
    }


def summarize_window(
    records: Sequence[Mapping[str, Any]],
    monitor: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    focus: dict[int, set[str]] = {}
    opponents: dict[int, set[str]] = {}
    clean_strength_wins = 0
    clean_strength_games = 0
    for record in records:
        game_id = int(record["game_id"])
        operational = record["operational"]
        focus_status = str(operational.get("focus_status", "unknown"))
        opponent_status = str(operational.get("opponent_status", "unknown"))
        if focus_status != "ok":
            focus.setdefault(game_id, set()).add(focus_status)
        if opponent_status != "ok":
            opponents.setdefault(game_id, set()).add(opponent_status)
        monitored = monitor[game_id]["classification"]
        if monitored["focus_failure"]:
            focus.setdefault(game_id, set()).add(str(monitored["focus_status"]))
        if monitored["opponent_failure"]:
            opponents.setdefault(game_id, set()).add(
                str(monitored["opponent_status"])
            )
        if focus_status == "ok" and opponent_status == "ok" \
                and not monitored["focus_failure"] \
                and not monitored["opponent_failure"]:
            clean_strength_games += 1
            if (record.get("focus") or {}).get("result") == "win":
                clean_strength_wins += 1
    focus_rows = [
        {"game_id": game_id, "categories": sorted(categories)}
        for game_id, categories in sorted(focus.items())
    ]
    opponent_rows = [
        {"game_id": game_id, "categories": sorted(categories)}
        for game_id, categories in sorted(opponents.items())
    ]
    return {
        "status": (
            "complete-rejected-focus-operational-failure"
            if focus_rows else "complete-accepted-diagnostic"
        ),
        "focus_operational_failures": focus_rows,
        "focus_operational_failure_games": len(focus_rows),
        "opponent_operational_failures": opponent_rows,
        "opponent_operational_failure_games": len(opponent_rows),
        "clean_strength_games": clean_strength_games,
        "clean_strength_wins": clean_strength_wins,
        "opponent_failure_games_counted_as_strength_wins": 0,
    }


def _window_reference_path(data_root: pathlib.Path) -> pathlib.Path:
    return data_root / "live-window.reference.json"


def watch_window(
    *,
    submission_attestation_path: pathlib.Path,
    exclusion_binding_path: pathlib.Path,
    data_root: pathlib.Path,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 3_600.0,
    maximum_workers: int = 2,
    fetch_battles: Callable[[], Any] | None = None,
    fetch_detail: Callable[[int], Any] | None = None,
    detail_classifier: Callable[..., Mapping[str, Any]] = classify_operational_detail,
    collector: Callable[..., Mapping[str, Any]] = collect_generic_window,
    collector_verifier: Callable[..., Mapping[str, Any]] = verify_generic_result,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    clock: Callable[[], str] = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if poll_seconds < 0 or timeout_seconds < 0 or maximum_workers not in (1, 2, 3, 4):
        raise LiveWindowError("poll/timeout/workers configuration is invalid")
    identity, exclusion_binding, registry_path = load_live_identity(
        submission_attestation_path, exclusion_binding_path
    )
    registry_sha = exclusion_binding["registry"]["sha256"]
    reference_path = _window_reference_path(data_root)
    if reference_path.exists():
        return verify_window_reference(reference_path, data_root=data_root)
    if fetch_battles is None or fetch_detail is None:
        generic = generic_collector_module()
        api = generic.shared.PublicApi()

        def fetch_json(service: str, payload: Any) -> Any:
            response = api.post(service, payload)
            return json.loads(response.body)

        fetch_battles = fetch_battles or (lambda: fetch_json(
            generic.shared.REQUEST_SCHEMAS["agent-battles-v1"]["service"],
            [identity.agent_id, None],
        ))
        fetch_detail = fetch_detail or (lambda game_id: fetch_json(
            generic.shared.REQUEST_SCHEMAS["game-detail-v1"]["service"],
            [game_id, None],
        ))
    monitor: dict[int, dict[str, Any]] = {}
    started = monotonic()
    while True:
        battles = fetch_battles()
        fetched = clock()
        _archive_monitor_payload(
            data_root,
            "metadata",
            battles,
            fetched_at_utc=fetched,
            request_identity={
                "agent_id": identity.agent_id,
                "submission_id": identity.submission_id,
            },
        )
        report = classify_matching_window(
            battles,
            agent_id=identity.agent_id,
            submission_id=identity.submission_id,
        )
        for game_id in report["complete_game_ids"]:
            if game_id not in monitor:
                monitor[game_id] = _monitor_game(
                    data_root,
                    game_id=game_id,
                    focus_agent_id=identity.agent_id,
                    fetch_detail=fetch_detail,
                    detail_classifier=detail_classifier,
                    clock=clock,
                )
        safe_progress = {
            "status": "ready" if report["ready"] else "waiting",
            "complete_games": report["complete_games"],
            "pending_games": report["pending_games"],
            "expected_games": EXACT_GAMES,
            "focus_operational_failure_games": sum(
                row["classification"]["focus_failure"] for row in monitor.values()
            ),
        }
        if progress is not None:
            progress(safe_progress)
        if report["ready"]:
            expected_ids = report["complete_game_ids"]
            if len(monitor) != EXACT_GAMES or set(monitor) != set(expected_ids):
                raise LiveWindowError("exact window was not fully monitored")
            collected = collector(
                identity=identity,
                registry_path=registry_path,
                registry_sha256=registry_sha,
                data_root=data_root,
                expected_game_ids=expected_ids,
                maximum_workers=maximum_workers,
            )
            verified = dict(collector_verifier(
                collected,
                identity=identity,
                registry_sha256=registry_sha,
                expected_game_ids=expected_ids,
            ))
            records = verified.get("records")
            if not isinstance(records, list) or len(records) != EXACT_GAMES:
                raise LiveWindowError("collector verifier returned wrong cardinality")
            summary = summarize_window(records, monitor)
            manifest_path = verified.get("manifest_path")
            manifest_reference = {
                "path": None if manifest_path is None else logical_path(manifest_path),
                "sha256": verified["manifest_sha256"],
            }
            receipt_path, receipt = write_content_addressed(
                data_root / "window-receipts",
                {
                    "schema": WINDOW_RECEIPT_SCHEMA,
                    "namespace": NAMESPACE,
                    "identity": identity_record(identity),
                    "submission_attestation": artifact_reference(
                        submission_attestation_path,
                        qualification_module().UPLOAD_EVENT_SCHEMA,
                    ),
                    "exclusion_binding": artifact_reference(
                        exclusion_binding_path, EXCLUSION_BINDING_SCHEMA
                    ),
                    "collector_manifest": manifest_reference,
                    "exact_games": EXACT_GAMES,
                    "game_ids": expected_ids,
                    "summary": summary,
                    "diagnostic_only": True,
                    "training_eligible": False,
                    "training_forbidden": True,
                    "rollback_authorized": False,
                    "second_upload_authorized": False,
                    "rank1_claim": False,
                    "collection_continued_after_focus_failure": True,
                },
            )
            reference = write_sealed(reference_path, {
                "schema": WINDOW_REFERENCE_SCHEMA,
                "namespace": NAMESPACE,
                "receipt": {
                    "path": logical_path(receipt_path),
                    "sha256": sha256_file(receipt_path),
                    "body_sha256": receipt["body_sha256"],
                },
                "status": summary["status"],
                "exact_games": EXACT_GAMES,
                "training_eligible": False,
                "rollback_authorized": False,
                "second_upload_authorized": False,
            })
            return reference
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            snapshot_path, snapshot = write_content_addressed(
                data_root / "wait-snapshots",
                {
                    "schema": WAIT_SNAPSHOT_SCHEMA,
                    "namespace": NAMESPACE,
                    **safe_progress,
                    "timed_out": True,
                    "collector_invoked": False,
                    "training_eligible": False,
                },
            )
            return {
                **snapshot,
                "snapshot_path": logical_path(snapshot_path),
                "snapshot_sha256": sha256_file(snapshot_path),
            }
        sleeper(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def verify_window_reference(
    reference_path: pathlib.Path, *, data_root: pathlib.Path
) -> dict[str, Any]:
    reference = load_sealed(
        reference_path, WINDOW_REFERENCE_SCHEMA, "live-window reference"
    )
    receipt_ref = reference.get("receipt")
    if (
        reference.get("namespace") != NAMESPACE
        or reference.get("exact_games") != EXACT_GAMES
        or reference.get("training_eligible") is not False
        or reference.get("rollback_authorized") is not False
        or reference.get("second_upload_authorized") is not False
        or not isinstance(receipt_ref, Mapping)
    ):
        raise LiveWindowError("live-window reference policy changed")
    receipt_path = resolve_path(receipt_ref.get("path"))
    try:
        receipt_path.relative_to(data_root.resolve())
    except ValueError as error:
        raise LiveWindowError("live-window receipt escaped its data root") from error
    if sha256_file(receipt_path) != receipt_ref.get("sha256"):
        raise LiveWindowError("live-window receipt hash changed")
    receipt = load_sealed(
        receipt_path, WINDOW_RECEIPT_SCHEMA, "live-window receipt"
    )
    summary = receipt.get("summary")
    attestation_ref = receipt.get("submission_attestation")
    exclusion_ref = receipt.get("exclusion_binding")
    collector_ref = receipt.get("collector_manifest")
    if (
        receipt.get("namespace") != NAMESPACE
        or receipt.get("exact_games") != EXACT_GAMES
        or not isinstance(receipt.get("game_ids"), list)
        or len(receipt["game_ids"]) != EXACT_GAMES
        or len(set(receipt["game_ids"])) != EXACT_GAMES
        or receipt.get("diagnostic_only") is not True
        or receipt.get("training_eligible") is not False
        or receipt.get("training_forbidden") is not True
        or receipt.get("rollback_authorized") is not False
        or receipt.get("second_upload_authorized") is not False
        or receipt.get("rank1_claim") is not False
        or not isinstance(summary, Mapping)
        or not isinstance(attestation_ref, Mapping)
        or not isinstance(exclusion_ref, Mapping)
        or not isinstance(collector_ref, Mapping)
        or summary.get("opponent_failure_games_counted_as_strength_wins") != 0
        or receipt_ref.get("body_sha256") != receipt.get("body_sha256")
        or reference.get("status") != summary.get("status")
    ):
        raise LiveWindowError("live-window receipt policy changed")
    attestation_path = resolve_path(attestation_ref.get("path"))
    exclusion_path = resolve_path(exclusion_ref.get("path"))
    if (
        sha256_file(attestation_path) != attestation_ref.get("sha256")
        or sha256_file(exclusion_path) != exclusion_ref.get("sha256")
    ):
        raise LiveWindowError("live-window sealed input hash changed")
    identity, exclusion, _registry_path = load_live_identity(
        attestation_path, exclusion_path
    )
    if (
        receipt.get("identity") != identity_record(identity)
        or attestation_ref.get("body_sha256")
        != qualification_module().load_sealed(
            attestation_path, qualification_module().UPLOAD_EVENT_SCHEMA
        ).get("body_sha256")
        or exclusion_ref.get("body_sha256") != exclusion.get("body_sha256")
    ):
        raise LiveWindowError("live-window source or submission identity changed")
    monitor = {}
    for game_id in receipt["game_ids"]:
        monitor[game_id] = load_sealed(
            data_root / "monitor/games" / f"{game_id}.json",
            MONITOR_GAME_SCHEMA,
            "monitor game",
        )
    if collector_ref.get("path") is not None:
        manifest_path = resolve_path(collector_ref["path"])
        if sha256_file(manifest_path) != collector_ref.get("sha256"):
            raise LiveWindowError("live-window collector manifest changed")
        verified = verify_generic_result(
            {
                "manifest_path": logical_path(manifest_path),
                "manifest_sha256": collector_ref["sha256"],
            },
            identity=identity,
            registry_sha256=exclusion["registry"]["sha256"],
            expected_game_ids=receipt["game_ids"],
        )
        recomputed = summarize_window(verified["records"], monitor)
        if recomputed != summary:
            raise LiveWindowError("live-window operational summary changed")
    return reference


def _canonical_live_boundaries(
    record: Mapping[str, Any], *, identity: LiveIdentity,
) -> list[str]:
    """Replay one accepted record and return every nonterminal turn boundary."""

    game_id = record.get("game_id")
    focus = record.get("focus")
    replay = record.get("replay")
    operational = record.get("operational")
    if (
        isinstance(game_id, bool) or not isinstance(game_id, int) or game_id <= 0
        or record.get("schema") != GENERIC_GAME_SCHEMA
        or record.get("status") != "accepted"
        or record.get("source_sha256") != identity.source_sha256
        or not isinstance(focus, Mapping)
        or focus.get("agent_id") != identity.agent_id
        or focus.get("submission_id") != identity.submission_id
        or not isinstance(replay, Mapping)
        or not isinstance(operational, Mapping)
    ):
        raise LiveWindowError("accepted live replay source identity changed")
    transcript = replay.get("valid_transcript")
    turns = replay.get("valid_turns")
    rules = replay.get("rules_validation")
    if (
        not isinstance(transcript, str) or not transcript
        or not isinstance(turns, list) or not turns
        or not isinstance(rules, Mapping)
        or rules.get("valid_turns") != turns
        or rules.get("valid_turn_count") != len(turns)
        or rules.get("status") not in {"terminal-valid", "incomplete", "invalid"}
    ):
        raise LiveWindowError(f"game {game_id} has no complete validated transcript")
    actions = []
    players = []
    for ordinal, turn in enumerate(turns):
        if (
            not isinstance(turn, Mapping)
            or set(turn) != {"action", "player_id"}
            or not isinstance(turn.get("action"), str)
            or not turn["action"]
            or any(character not in "01234567" for character in turn["action"])
            or turn.get("player_id") not in (0, 1)
        ):
            raise LiveWindowError(
                f"game {game_id} validated turn {ordinal} is malformed"
            )
        actions.append(str(turn["action"]))
        players.append(int(turn["player_id"]))
    if transcript != "/".join(actions):
        raise LiveWindowError(f"game {game_id} valid transcript/turns disagree")

    features = replay_features_module()
    opening_tools = opening_fingerprints_module()
    state = features.ReplayState()
    canonical = [opening_tools.state_fingerprints(state)["canonical"]]
    for ordinal, (player, action) in enumerate(zip(players, actions, strict=True)):
        if state.winner is not None or player != state.to_move:
            raise LiveWindowError(
                f"game {game_id} validated turn {ordinal} has wrong player/terminal order"
            )
        try:
            features.apply_complete_turn(state, player, action)
        except (KeyError, TypeError, ValueError) as error:
            raise LiveWindowError(
                f"game {game_id} valid transcript contains an incomplete or illegal complete turn"
            ) from error
        if state.winner is None:
            canonical.append(opening_tools.state_fingerprints(state)["canonical"])
    status = rules["status"]
    if status == "terminal-valid":
        if (
            state.winner is None
            or rules.get("terminal_winner_player_id") != state.winner
        ):
            raise LiveWindowError(f"game {game_id} terminal replay result changed")
    elif (
        state.winner is not None
        or operational.get("classification") != "operationally-terminated"
    ):
        raise LiveWindowError(
            f"game {game_id} incomplete replay lacks a bound operational ending"
        )
    return canonical


def extract_verified_live_fingerprints(
    reference_path: pathlib.Path, *, data_root: pathlib.Path,
) -> dict[str, Any]:
    """Return sealed, transcript-free fingerprints from a verified live window.

    This is the only live-position extraction entrypoint intended for campaign
    governance.  It first performs the complete existing live-window audit,
    then revalidates the exact collector manifest and all 90 accepted records
    before replaying their complete-turn boundaries independently.
    """

    validated = verify_window_reference(reference_path, data_root=data_root)
    reference_path = reference_path.resolve()
    data_root = data_root.resolve()
    if reference_path != _window_reference_path(data_root).resolve():
        raise LiveWindowError("trusted live fingerprint reference path changed")
    reference = load_sealed(
        reference_path, WINDOW_REFERENCE_SCHEMA, "live-window reference"
    )
    if validated != reference:
        raise LiveWindowError("full live-window validation returned another reference")
    receipt_ref = reference.get("receipt")
    if not isinstance(receipt_ref, Mapping):
        raise LiveWindowError("live-window receipt reference is absent")
    receipt_path = resolve_path(receipt_ref.get("path"))
    try:
        receipt_path.relative_to(data_root)
    except ValueError as error:
        raise LiveWindowError("live fingerprint receipt escaped its data root") from error
    if (
        receipt_path.is_symlink() or not receipt_path.is_file()
        or sha256_file(receipt_path) != receipt_ref.get("sha256")
        or receipt_path.name != f"{receipt_ref.get('sha256')}.json"
    ):
        raise LiveWindowError("live fingerprint receipt identity changed")
    receipt = load_sealed(
        receipt_path, WINDOW_RECEIPT_SCHEMA, "live-window receipt"
    )
    game_ids = receipt.get("game_ids")
    collector_ref = receipt.get("collector_manifest")
    if (
        not isinstance(game_ids, list)
        or game_ids != sorted(set(game_ids))
        or len(game_ids) != EXACT_GAMES
        or any(
            isinstance(game_id, bool) or not isinstance(game_id, int)
            or game_id <= 0 for game_id in game_ids
        )
        or not isinstance(collector_ref, Mapping)
        or not isinstance(collector_ref.get("path"), str)
        or not collector_ref["path"]
        or not valid_sha256(collector_ref.get("sha256"))
    ):
        raise LiveWindowError("trusted live collector/game-ID binding is incomplete")
    attestation_ref = receipt.get("submission_attestation")
    exclusion_ref = receipt.get("exclusion_binding")
    if not isinstance(attestation_ref, Mapping) or not isinstance(exclusion_ref, Mapping):
        raise LiveWindowError("trusted live receipt lost its source inputs")
    attestation_path = resolve_path(attestation_ref.get("path"))
    exclusion_path = resolve_path(exclusion_ref.get("path"))
    if (
        dict(attestation_ref) != artifact_reference(
            attestation_path, qualification_module().UPLOAD_EVENT_SCHEMA
        )
        or dict(exclusion_ref) != artifact_reference(
            exclusion_path, EXCLUSION_BINDING_SCHEMA
        )
    ):
        raise LiveWindowError("trusted live sealed source input changed")
    identity, exclusion, _registry_path = load_live_identity(
        attestation_path, exclusion_path,
    )
    if receipt.get("identity") != identity_record(identity):
        raise LiveWindowError("trusted live receipt source identity changed")
    manifest_path = resolve_path(collector_ref["path"])
    try:
        manifest_path.relative_to(data_root)
    except ValueError as error:
        raise LiveWindowError("trusted collector manifest escaped its data root") from error
    if (
        manifest_path.is_symlink() or not manifest_path.is_file()
        or sha256_file(manifest_path) != collector_ref["sha256"]
    ):
        raise LiveWindowError("trusted collector manifest changed")
    verified = verify_generic_result(
        {
            "manifest_path": logical_path(manifest_path),
            "manifest_sha256": collector_ref["sha256"],
        },
        identity=identity,
        registry_sha256=exclusion["registry"]["sha256"],
        expected_game_ids=game_ids,
    )
    manifest = verified.get("manifest")
    records = verified.get("records")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(records, list)
        or len(records) != EXACT_GAMES
        or any(
            not isinstance(record, Mapping)
            or isinstance(record.get("game_id"), bool)
            or not isinstance(record.get("game_id"), int)
            for record in records
        )
        or verified.get("manifest_path") != manifest_path
        or verified.get("manifest_sha256") != collector_ref["sha256"]
    ):
        raise LiveWindowError("trusted collector verification is incomplete")
    binding = manifest.get("binding")
    source = binding.get("source") if isinstance(binding, Mapping) else None
    if (
        not isinstance(binding, Mapping) or not isinstance(source, Mapping)
        or binding.get("schema") != GENERIC_BINDING_SCHEMA
        or binding.get("agent_id") != identity.agent_id
        or binding.get("asserted_submission_id") != identity.submission_id
        or binding.get("repository_commit") != identity.repository_commit
        or source.get("sha256") != identity.source_sha256
        or source.get("bytes") != identity.source_bytes
        or not valid_sha256(manifest.get("collector_sha256"))
    ):
        raise LiveWindowError("trusted collector source binding changed")
    manifest_games = manifest.get("games")
    embedded_records = [
        stored.get("record") for stored in manifest_games
        if isinstance(stored, Mapping)
    ] if isinstance(manifest_games, list) else []
    records = sorted((dict(record) for record in records), key=lambda row: row["game_id"])
    if (
        [record["game_id"] for record in records] != game_ids
        or len(embedded_records) != EXACT_GAMES
        or sorted(
            (dict(record) for record in embedded_records if isinstance(record, Mapping)),
            key=lambda row: row.get("game_id", -1),
        ) != records
    ):
        raise LiveWindowError("trusted collector accepted-record roster changed")
    canonical: set[str] = set()
    boundary_count = 0
    for record in records:
        values = _canonical_live_boundaries(record, identity=identity)
        boundary_count += len(values)
        canonical.update(values)
    fingerprints = sorted(canonical)
    if not fingerprints or any(not valid_sha256(value) for value in fingerprints):
        raise LiveWindowError("trusted live canonical fingerprint set is invalid")
    game_ids_sha256 = sha256_bytes(canonical_json_bytes(game_ids))
    fingerprints_sha256 = sha256_bytes(canonical_json_bytes(fingerprints))
    return seal({
        "schema": LIVE_FINGERPRINT_EVIDENCE_SCHEMA,
        "namespace": NAMESPACE,
        "status": "verified-live-canonical-fingerprints",
        "live_window_reference": artifact_reference(
            reference_path, WINDOW_REFERENCE_SCHEMA
        ),
        "live_window_receipt": artifact_reference(
            receipt_path, WINDOW_RECEIPT_SCHEMA
        ),
        "collector_manifest": {
            "path": logical_path(manifest_path),
            "sha256": collector_ref["sha256"],
            "schema": GENERIC_BATCH_SCHEMA,
            "collector_sha256": manifest.get("collector_sha256"),
            "accepted_records_sha256": sha256_bytes(
                canonical_json_bytes(records)
            ),
        },
        "source_identity": {
            "agent_id": identity.agent_id,
            "submission_id": identity.submission_id,
            "repository_commit": identity.repository_commit,
            "source_sha256": identity.source_sha256,
            "source_bytes": identity.source_bytes,
        },
        "exact_games": EXACT_GAMES,
        "game_ids": game_ids,
        "game_ids_sha256": game_ids_sha256,
        "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
        "boundary_count": boundary_count,
        "fingerprints": fingerprints,
        "fingerprint_count": len(fingerprints),
        "fingerprints_sha256": fingerprints_sha256,
        "contains_transcripts": False,
        "contains_metrics": False,
        "contains_labels": False,
        "training_eligible": False,
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind-exclusions")
    bind.add_argument("--registry", type=pathlib.Path, required=True)
    bind.add_argument("--frozen-at-utc", required=True)
    bind.add_argument("--output", type=pathlib.Path, required=True)
    watch = commands.add_parser("watch")
    watch.add_argument("--submission-attestation", type=pathlib.Path, required=True)
    watch.add_argument("--exclusion-binding", type=pathlib.Path, required=True)
    watch.add_argument("--data-root", type=pathlib.Path, required=True)
    watch.add_argument("--poll-seconds", type=float, default=10.0)
    watch.add_argument("--timeout-seconds", type=float, default=3_600.0)
    watch.add_argument("--maximum-workers", type=int, choices=(1, 2, 3, 4), default=2)
    verify = commands.add_parser("verify")
    verify.add_argument("--reference", type=pathlib.Path, required=True)
    verify.add_argument("--data-root", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "bind-exclusions":
            result = freeze_exclusion_binding(
                arguments.output,
                registry_path=arguments.registry,
                frozen_at_utc=arguments.frozen_at_utc,
            )
        elif arguments.command == "watch":
            result = watch_window(
                submission_attestation_path=arguments.submission_attestation,
                exclusion_binding_path=arguments.exclusion_binding,
                data_root=arguments.data_root,
                poll_seconds=arguments.poll_seconds,
                timeout_seconds=arguments.timeout_seconds,
                maximum_workers=arguments.maximum_workers,
            )
        else:
            result = verify_window_reference(
                arguments.reference, data_root=arguments.data_root
            )
    except (OSError, LiveWindowError) as error:
        parser.exit(1, f"compact live-window failure: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
