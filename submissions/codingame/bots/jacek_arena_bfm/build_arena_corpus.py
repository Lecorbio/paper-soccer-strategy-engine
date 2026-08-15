#!/usr/bin/env python3
"""Build fresh arena value/ranking rows from one frozen training window.

The CLI accepts only a campaign-local auditor TSV and an immutable arena
derivation that passes :func:`campaign_provenance.validate_arena_derivation`.
It cross-checks every TSV transcript against the derivation's content-addressed
fresh arena manifest before invoking the clean-room C++ rules/reanalysis tool.
No historical replay, action, label, model, restart, or arena-state path is
discovered or opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .campaign_provenance import validate_arena_derivation
    from .fresh_corpus import (
        APPROVED_EXCLUSION_SHA256,
        APPROVED_WINDOW_PLAN_SHA256,
        CORPUS_SCHEMA,
        FEATURE_COUNT,
        NAMESPACE,
        ArenaGameBinding,
        CampaignContract,
        CorpusValidationError,
        FreshCorpusValidator,
        load_arena_game_bindings,
        load_contract,
        load_excluded_game_ids,
        parse_utc,
        require_within_campaign_root,
    )
    from .immutable_artifacts import (
        canonical_json_bytes,
        file_inventory,
        sha256_bytes,
        sha256_file,
        write_content_addressed_bytes,
        write_content_addressed_json,
    )
except ImportError:  # pragma: no cover - standalone CLI execution
    from campaign_provenance import validate_arena_derivation
    from fresh_corpus import (
        APPROVED_EXCLUSION_SHA256,
        APPROVED_WINDOW_PLAN_SHA256,
        CORPUS_SCHEMA,
        FEATURE_COUNT,
        NAMESPACE,
        ArenaGameBinding,
        CampaignContract,
        CorpusValidationError,
        FreshCorpusValidator,
        load_arena_game_bindings,
        load_contract,
        load_excluded_game_ids,
        parse_utc,
        require_within_campaign_root,
    )
    from immutable_artifacts import (
        canonical_json_bytes,
        file_inventory,
        sha256_bytes,
        sha256_file,
        write_content_addressed_bytes,
        write_content_addressed_json,
    )


AUDITOR_HEADER = "game_id\tcandidate_player\twinner\tturns"
AUDITOR_METADATA_KEYS = frozenset({
    "agent_id",
    "arena_manifest_sha256",
    "asserted_source_sha256",
    "asserted_submission_id",
    "collector_sha256",
    "exclusion_registry_sha256",
    "repository_commit",
    "run_id",
    "source_binding_status",
})
REANALYSIS_SCHEMA = "papersoccer.jacek-arena-bfm.reanalysis.v1"
BUILD_MANIFEST_SCHEMA = "papersoccer.jacek-arena-bfm.arena-corpus-build.v1"
REPRESENTATION = "mover_relative_316_edges_plus_105x8_distance_v1"
ACTION_PATTERN = re.compile(r"[0-7]{1,316}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
POSITION_PATTERN = re.compile(r"fnv1a64:[0-9a-f]{16}\Z")
MAXIMUM_AUDITOR_BYTES = 128 * 1024 * 1024
MAXIMUM_REANALYSIS_BYTES = 512 * 1024 * 1024


class ArenaCorpusError(ValueError):
    """Raised before any derived corpus artifact is published."""


@dataclasses.dataclass(frozen=True)
class AuditorGame:
    game_id: int
    candidate_player: int
    winner: int
    turns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AuditorInput:
    path: Path
    sha256: str
    byte_count: int
    metadata: Mapping[str, str]
    games: Mapping[int, AuditorGame]


@dataclasses.dataclass(frozen=True)
class ReanalysisOutput:
    meta: Mapping[str, Any]
    value_rows: tuple[Mapping[str, Any], ...]
    pairwise_rows: tuple[Mapping[str, Any], ...]
    sha256: str
    byte_count: int


def _resolve_reference(raw: Any, repository: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ArenaCorpusError("artifact reference path is missing")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArenaCorpusError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArenaCorpusError(f"{field} must be an integer >= {minimum}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], field: str) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        raise ArenaCorpusError(
            f"{field} keys differ: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def parse_auditor_tsv(path: Path) -> AuditorInput:
    payload = path.read_bytes()
    if not payload or len(payload) > MAXIMUM_AUDITOR_BYTES:
        raise ArenaCorpusError("auditor TSV is empty or exceeds the byte limit")
    if not payload.endswith(b"\n") or b"\r" in payload or b"\x00" in payload:
        raise ArenaCorpusError("auditor TSV must use canonical LF-terminated ASCII lines")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ArenaCorpusError("auditor TSV is not ASCII") from error
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    cursor = 0
    while cursor < len(lines) and lines[cursor].startswith("# "):
        raw = lines[cursor][2:]
        key, separator, value = raw.partition("=")
        if separator != "=" or not key or not value or key in metadata:
            raise ArenaCorpusError(f"invalid/duplicate auditor metadata at line {cursor + 1}")
        metadata[key] = value
        cursor += 1
    if set(metadata) != AUDITOR_METADATA_KEYS:
        raise ArenaCorpusError("auditor TSV metadata is incomplete or contains unknown keys")
    if cursor >= len(lines) or lines[cursor] != AUDITOR_HEADER:
        raise ArenaCorpusError("auditor TSV has a non-canonical header")
    cursor += 1
    games: dict[int, AuditorGame] = {}
    previous_id = 0
    for line_number, line in enumerate(lines[cursor:], cursor + 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ArenaCorpusError(f"auditor row {line_number} must have exactly four fields")
        try:
            game_id, candidate, winner = map(int, fields[:3])
        except ValueError as error:
            raise ArenaCorpusError(f"auditor row {line_number} has non-integer identity fields") from error
        if game_id <= previous_id:
            raise ArenaCorpusError("auditor game IDs must be positive, unique, and sorted")
        previous_id = game_id
        if candidate not in (0, 1) or winner not in (0, 1):
            raise ArenaCorpusError("auditor candidate/winner fields must be zero or one")
        turns = tuple(fields[3].split("/"))
        if not turns or any(ACTION_PATTERN.fullmatch(turn) is None for turn in turns):
            raise ArenaCorpusError(f"auditor game {game_id} has an invalid complete-turn transcript")
        games[game_id] = AuditorGame(game_id, candidate, winner, turns)
    if not games:
        raise ArenaCorpusError("auditor TSV contains no games")
    return AuditorInput(
        path=path.resolve(),
        sha256=sha256_bytes(payload),
        byte_count=len(payload),
        metadata=dict(sorted(metadata.items())),
        games=games,
    )


def crosscheck_auditor(
    auditor: AuditorInput,
    derivation: Mapping[str, Any],
    *,
    repository: Path,
) -> dict[int, Mapping[str, Any]]:
    manifest_reference = derivation["arena_manifest"]
    manifest_path = _resolve_reference(manifest_reference["path"], repository)
    manifest_payload = manifest_path.read_bytes()
    if sha256_bytes(manifest_payload) != manifest_reference["sha256"]:
        raise ArenaCorpusError("arena manifest content hash changed after derivation validation")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArenaCorpusError("arena manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ArenaCorpusError("arena manifest must be an object")

    binding = manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise ArenaCorpusError("arena manifest source binding is missing")
    source = derivation["source"]
    expected_metadata = {
        "agent_id": str(source["agent_id"]),
        "arena_manifest_sha256": str(manifest_reference["sha256"]),
        "asserted_source_sha256": str(source["sha256"]),
        "asserted_submission_id": str(source["submission_id"]),
        "collector_sha256": str(manifest.get("collector_sha256")),
        "exclusion_registry_sha256": str(derivation["exclusion_registry"]["sha256"]),
        "repository_commit": str(source["repository_commit"]),
        "run_id": str(manifest.get("run_id")),
        "source_binding_status": "asserted-not-api-verified",
    }
    if dict(auditor.metadata) != dict(sorted(expected_metadata.items())):
        raise ArenaCorpusError("auditor metadata contradicts its validated arena derivation")

    derived_by_id = {int(row["game_id"]): row for row in derivation["games"]}
    eligible_ids = {
        game_id for game_id, row in derived_by_id.items()
        if row["disposition"] == "eligible"
    }
    if set(auditor.games) != eligible_ids:
        raise ArenaCorpusError(
            "auditor game IDs do not exactly equal the derivation's operationally clean games"
        )

    records: dict[int, Mapping[str, Any]] = {}
    record_hashes: dict[int, str] = {}
    for stored in manifest.get("games", []):
        if not isinstance(stored, Mapping) or not isinstance(stored.get("record"), Mapping):
            raise ArenaCorpusError("arena manifest contains an invalid stored record")
        record = stored["record"]
        game_id = int(record.get("game_id", 0))
        if game_id in records:
            raise ArenaCorpusError(f"arena manifest repeats game {game_id}")
        records[game_id] = record
        record_hashes[game_id] = _require_sha256(
            stored.get("record_sha256"), f"manifest game {game_id} record hash"
        )
    for game_id, auditor_game in auditor.games.items():
        record = records.get(game_id)
        if record is None:
            raise ArenaCorpusError(f"auditor game {game_id} is absent from its arena manifest")
        focus = record.get("focus") or {}
        outcome = record.get("outcome") or {}
        replay = record.get("replay") or {}
        observed = (
            int(focus.get("player_id", -1)),
            int(outcome.get("winner_player_id", -1)),
            str(replay.get("valid_transcript", "")),
        )
        expected = (
            auditor_game.candidate_player,
            auditor_game.winner,
            "/".join(auditor_game.turns),
        )
        if observed != expected:
            raise ArenaCorpusError(f"auditor game {game_id} contradicts its immutable manifest record")
        stored_hash = _require_sha256(
            derived_by_id[game_id]["record_sha256"],
            f"derived game {game_id} record hash",
        )
        if stored_hash != record_hashes[game_id]:
            raise ArenaCorpusError(f"arena game {game_id} record hash binding changed")
    return {game_id: derived_by_id[game_id] for game_id in sorted(eligible_ids)}


def _active_to_dense(active: Any, field: str) -> list[int]:
    if not isinstance(active, list) or not 105 <= len(active) <= 421:
        raise ArenaCorpusError(f"{field} must contain 105..421 sparse feature indices")
    dense = [0] * FEATURE_COUNT
    previous = -1
    for item in active:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < FEATURE_COUNT:
            raise ArenaCorpusError(f"{field} has an invalid feature index")
        if item <= previous:
            raise ArenaCorpusError(f"{field} indices must be strictly increasing")
        previous = item
        dense[item] = 1
    return dense


def parse_reanalysis(payload: bytes, auditor: AuditorInput) -> ReanalysisOutput:
    if not payload or len(payload) > MAXIMUM_REANALYSIS_BYTES or not payload.endswith(b"\n"):
        raise ArenaCorpusError("reanalyzer output is empty, truncated, or exceeds its byte limit")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ArenaCorpusError("reanalyzer output is not ASCII") from error
    decoded: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArenaCorpusError(f"reanalyzer line {line_number} is invalid JSON") from error
        if not isinstance(value, Mapping) or value.get("schema") != REANALYSIS_SCHEMA:
            raise ArenaCorpusError(f"reanalyzer line {line_number} has an invalid schema/object")
        decoded.append(value)
    if not decoded or decoded[0].get("kind") != "meta":
        raise ArenaCorpusError("reanalyzer output must begin with one meta record")
    if any(row.get("kind") == "meta" for row in decoded[1:]):
        raise ArenaCorpusError("reanalyzer output repeats its meta record")
    meta = decoded[0]
    value_rows = tuple(row for row in decoded[1:] if row.get("kind") == "value")
    pair_rows = tuple(row for row in decoded[1:] if row.get("kind") == "pairwise")
    if len(value_rows) + len(pair_rows) != len(decoded) - 1:
        raise ArenaCorpusError("reanalyzer emitted an unknown record kind")
    if (
        meta.get("work_checkpoints") != [30000, 100000]
        or meta.get("games") != len(auditor.games)
        or meta.get("value_rows") != len(value_rows)
        or meta.get("pairwise_rows") != len(pair_rows)
    ):
        raise ArenaCorpusError("reanalyzer meta counts/checkpoints contradict its records")
    maximum_pairs_decision = _require_int(
        meta.get("maximum_pairs_per_decision"), "maximum_pairs_per_decision")
    maximum_pairs_game = _require_int(
        meta.get("maximum_pairs_per_game"), "maximum_pairs_per_game")
    if maximum_pairs_decision > 4 or maximum_pairs_game > 32:
        raise ArenaCorpusError("reanalyzer pair caps violate the corpus safety contract")

    values_by_game: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in value_rows:
        game_id = _require_int(row.get("game_id"), "value game_id", minimum=1)
        turn = _require_int(row.get("turn_index"), "value turn_index")
        game = auditor.games.get(game_id)
        if game is None or turn >= len(game.turns) or turn in values_by_game[game_id]:
            raise ArenaCorpusError("reanalyzer value row has an unknown/duplicate game turn")
        if (
            row.get("candidate_player") != game.candidate_player
            or row.get("winner") != game.winner
            or row.get("actor") != turn % 2
            or row.get("target") != (1 if turn % 2 == game.winner else -1)
            or not isinstance(row.get("position_id"), str)
            or POSITION_PATTERN.fullmatch(row["position_id"]) is None
        ):
            raise ArenaCorpusError("reanalyzer value row contradicts the auditor game")
        _active_to_dense(row.get("active_features"), "value active_features")
        values_by_game[game_id][turn] = row
    for game_id, game in auditor.games.items():
        if set(values_by_game.get(game_id, {})) != set(range(len(game.turns))):
            raise ArenaCorpusError(f"reanalyzer omitted value positions for game {game_id}")

    counts_by_decision: Counter[tuple[int, str]] = Counter()
    counts_by_game: Counter[int] = Counter()
    indices: set[tuple[int, str, int]] = set()
    for row in pair_rows:
        game_id = _require_int(row.get("game_id"), "pair game_id", minimum=1)
        turn = _require_int(row.get("turn_index"), "pair turn_index")
        game = auditor.games.get(game_id)
        if game is None or turn >= len(game.turns):
            raise ArenaCorpusError("reanalyzer pair row references an unknown game turn")
        decision_id = row.get("decision_id")
        pair_index = _require_int(row.get("pair_index"), "pair_index")
        observed = row.get("observed_action")
        inferior = row.get("inferior_action")
        if (
            row.get("candidate_player") != game.candidate_player
            or row.get("winner") != game.winner
            or row.get("actor") != 1 - game.candidate_player
            or turn % 2 != 1 - game.candidate_player
            or decision_id != values_by_game[game_id][turn]["position_id"]
            or observed != game.turns[turn]
            or not isinstance(inferior, str)
            or ACTION_PATTERN.fullmatch(inferior) is None
            or observed == inferior
        ):
            raise ArenaCorpusError("ranking target is not the legal observed opponent action")
        _active_to_dense(row.get("preferred_active_features"), "preferred_active_features")
        _active_to_dense(row.get("inferior_active_features"), "inferior_active_features")
        if row.get("preferred_active_features") == row.get("inferior_active_features"):
            raise ArenaCorpusError("pairwise successors are not distinct")
        key = (game_id, str(decision_id))
        index_key = (*key, pair_index)
        counts_by_decision[key] += 1
        counts_by_game[game_id] += 1
        if (
            index_key in indices
            or pair_index >= maximum_pairs_decision
            or counts_by_decision[key] > maximum_pairs_decision
            or counts_by_game[game_id] > maximum_pairs_game
        ):
            raise ArenaCorpusError("reanalyzer pair indices/counts exceed their declared caps")
        indices.add(index_key)

        exact = row.get("exact")
        if not isinstance(exact, bool):
            raise ArenaCorpusError("pair exact flag must be boolean")
        values: dict[tuple[str, int], float] = {}
        for side in ("preferred", "inferior"):
            terminal = row.get(f"{side}_terminal")
            if not isinstance(terminal, bool):
                raise ArenaCorpusError("pair terminal flags must be boolean")
            for work in (30000, 100000):
                raw_value = row.get(f"{side}_value_{work}")
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise ArenaCorpusError("pair search values must be numeric")
                value = float(raw_value)
                if not -1.0 <= value <= 1.0:
                    raise ArenaCorpusError("pair search values must be in [-1, 1]")
                values[(side, work)] = value
                used = _require_int(row.get(f"{side}_work_{work}"), "pair work")
                if used != (0 if terminal else work):
                    raise ArenaCorpusError("each nonterminal pair successor must receive exact 30k/100k work")
                depth = _require_int(row.get(f"{side}_depth_{work}"), "pair completed depth")
                if terminal and depth != 0:
                    raise ArenaCorpusError("terminal pair successor cannot claim a search depth")
        if exact:
            if (
                row.get("preferred_terminal") is not True
                or row.get("inferior_terminal") is not True
                or values[("preferred", 30000)] != 1.0
                or values[("inferior", 30000)] != -1.0
            ):
                raise ArenaCorpusError("exact pair must be an immediate proven win over loss")
        elif any(
            values[("preferred", work)] - values[("inferior", work)] < 0.10 - 1e-9
            for work in (30000, 100000)
        ):
            raise ArenaCorpusError("pair ordering is not stable by >=0.10 at both work gates")
        if (
            values[("preferred", 100000)] <= -1.0 + 1e-9
            and values[("inferior", 100000)] >= 1.0 - 1e-9
        ):
            raise ArenaCorpusError("observed action is proved losing while an alternative wins")

    return ReanalysisOutput(
        meta=meta,
        value_rows=value_rows,
        pairwise_rows=pair_rows,
        sha256=sha256_bytes(payload),
        byte_count=len(payload),
    )


def _common_arena_fields(
    *,
    derivation: Mapping[str, Any],
    derivation_sha256: str,
    game: Mapping[str, Any],
    generated_at_utc: str,
    evidence_at_utc: str,
) -> dict[str, Any]:
    source = derivation["source"]
    return {
        "agent_id": str(source["agent_id"]),
        "arena_derivation_sha256": derivation_sha256,
        "arena_game_id": int(game["game_id"]),
        "arena_record_sha256": str(game["record_sha256"]),
        "campaign_id": f"{NAMESPACE}@{derivation['campaign']['t0_utc']}",
        "complete_transcript": True,
        "evidence_at_utc": evidence_at_utc,
        "evidence_sha256": str(game["record_sha256"]),
        "generated_at_utc": generated_at_utc,
        "illegal_action": False,
        "malformed_transcript": False,
        "namespace": NAMESPACE,
        "normalized_sha256": str(game["normalized_sha256"]),
        "operational_clean": True,
        "producer_source_sha256": str(source["sha256"]),
        "raw_sha256": str(game["raw_sha256"]),
        "representation": REPRESENTATION,
        "schema": CORPUS_SCHEMA,
        "submission_id": str(source["submission_id"]),
        "submitted_source_sha256": str(source["sha256"]),
        "terminal_unambiguous": True,
        "timeout": False,
        "window_id": str(derivation["window"]["window_id"]),
        "window_role": str(derivation["window"]["role"]),
    }


def build_rows(
    reanalysis: ReanalysisOutput,
    auditor: AuditorInput,
    derivation: Mapping[str, Any],
    *,
    derivation_sha256: str,
    generated_at_utc: str,
    terminal_value_weight: float = 0.25,
) -> list[dict[str, Any]]:
    if not 0.0 < terminal_value_weight <= 0.5:
        raise ArenaCorpusError("terminal_value_weight must be in (0, 0.5]")
    evidence_at = str(derivation["timing"]["collection_completed_at_utc"])
    generated = parse_utc(generated_at_utc, "generated_at_utc")
    if generated < parse_utc(evidence_at, "collection_completed_at_utc"):
        raise ArenaCorpusError("corpus generation timestamp precedes collection completion")
    by_game = {int(row["game_id"]): row for row in derivation["games"]}
    rows: list[dict[str, Any]] = []
    window_id = str(derivation["window"]["window_id"])
    for value in reanalysis.value_rows:
        game_id = int(value["game_id"])
        game = by_game[game_id]
        common = _common_arena_fields(
            derivation=derivation,
            derivation_sha256=derivation_sha256,
            game=game,
            generated_at_utc=generated_at_utc,
            evidence_at_utc=evidence_at,
        )
        rows.append({
            **common,
            "features": _active_to_dense(value["active_features"], "value active_features"),
            "game_id": game_id,
            "kind": "value",
            "label_method": "terminal_outcome",
            "position_id": str(value["position_id"]),
            "sample_id": f"arena:{window_id}:{game_id}:value:{value['turn_index']}",
            "source_kind": "arena_terminal",
            "target": int(value["target"]),
            "theoretical_value_claim": False,
            "weight": terminal_value_weight,
        })

    for pair in reanalysis.pairwise_rows:
        game_id = int(pair["game_id"])
        game = by_game[game_id]
        if "opponent-action-ranking-reanalysis-candidate" not in game["uses"]:
            raise ArenaCorpusError(f"reanalyzer emitted a pair for unauthorized game {game_id}")
        rank = game["opponent_frozen_rank"]
        weight = float(game["ranking_candidate_weight"])
        if rank is None or not 1 <= int(rank) <= 50 or weight not in (0.5, 1.0):
            raise ArenaCorpusError(f"game {game_id} lacks a frozen eligible opponent rank")
        common = _common_arena_fields(
            derivation=derivation,
            derivation_sha256=derivation_sha256,
            game=game,
            generated_at_utc=generated_at_utc,
            evidence_at_utc=evidence_at,
        )
        row = {
            **common,
            "actor_origin": "opponent",
            "complete_action_legal": True,
            "counterfactual_replay_verified": True,
            "counterfactual_verdict": "observed-not-proved-losing-vs-winning-alternative",
            "decision_id": str(pair["decision_id"]),
            "exact": bool(pair["exact"]),
            "game_id": game_id,
            "inferior_complete_action": str(pair["inferior_action"]),
            "inferior_features": _active_to_dense(
                pair["inferior_active_features"], "inferior_active_features"
            ),
            "kind": "pairwise",
            "observed_complete_action": str(pair["observed_action"]),
            "opponent_snapshot_rank": int(rank),
            "pair_index": int(pair["pair_index"]),
            "position_id": str(pair["decision_id"]),
            "preferred_features": _active_to_dense(
                pair["preferred_active_features"], "preferred_active_features"
            ),
            "sample_id": (
                f"arena:{window_id}:{game_id}:pair:{pair['turn_index']}:"
                f"{pair['pair_index']}"
            ),
            "source_kind": "arena_opponent_ranking",
            "weight": weight,
        }
        for work in (30000, 100000):
            row[f"preferred_value_{work}"] = float(pair[f"preferred_value_{work}"])
            row[f"inferior_value_{work}"] = float(pair[f"inferior_value_{work}"])
        if pair["exact"]:
            row["exact_ordering"] = "preferred"
        rows.append(row)
    return rows


def _compile_reanalyzer(
    *, repository: Path,
    output: Path,
    cxx: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    bot = repository / "submissions/codingame/bots/jacek_arena_bfm"
    source_paths = [
        bot / "arena_corpus_reanalyzer.cpp",
        bot / "engine.cpp",
        bot / "engine.hpp",
        bot / "model.hpp",
    ]
    command = [
        cxx,
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        str(source_paths[0]),
        str(source_paths[1]),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, cwd=repository, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ArenaCorpusError(f"reanalyzer compilation failed:\n{completed.stderr}")
    reproducible_command = [
        cxx,
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        str(source_paths[0].relative_to(repository)),
        str(source_paths[1].relative_to(repository)),
        "-o",
        "$REANALYZER",
    ]
    return reproducible_command, file_inventory(source_paths, root=repository)


def _run_reanalyzer(
    *,
    executable: Path,
    auditor: AuditorInput,
    ranking_game_ids: Sequence[int],
    maximum_analyzed_decisions: int,
    maximum_alternatives: int,
    maximum_pairs_per_game: int,
    search_width: int,
    generator: str,
    timeout_seconds: int,
    temporary_directory: Path,
) -> tuple[bytes, list[str]]:
    ranking_path = temporary_directory / "ranking-game-ids.txt"
    ranking_path.write_text(
        "".join(f"{game_id}\n" for game_id in ranking_game_ids), encoding="ascii"
    )
    command = [
        str(executable),
        "--input", str(auditor.path),
        "--ranking-game-ids", str(ranking_path),
        "--max-analyzed-decisions", str(maximum_analyzed_decisions),
        "--max-alternatives", str(maximum_alternatives),
        "--max-pairs-per-decision", "4",
        "--max-pairs-per-game", str(maximum_pairs_per_game),
        "--search-width", str(search_width),
        "--generator", generator,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=temporary_directory,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ArenaCorpusError("reanalyzer exceeded its explicit timeout") from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise ArenaCorpusError(f"reanalyzer rejected the arena input: {stderr.strip()}")
    reproducible_arguments = [
        "$REANALYZER" if item == str(executable)
        else "$AUDITOR_TSV" if item == str(auditor.path)
        else "$RANKING_GAME_IDS" if item == str(ranking_path)
        else item
        for item in command
    ]
    return completed.stdout, reproducible_arguments


def build_arena_corpus(
    *,
    repository: Path,
    campaign_root: Path,
    derivation_path: Path,
    expected_derivation_sha256: str,
    auditor_path: Path,
    output_directory: Path,
    generated_at_utc: str,
    terminal_value_weight: float = 0.25,
    maximum_analyzed_decisions: int = 2,
    maximum_alternatives: int = 6,
    maximum_pairs_per_game: int = 16,
    search_width: int = 8,
    generator: str = "tactical-progressive",
    timeout_seconds: int = 3600,
    cxx: str = "c++",
) -> dict[str, Any]:
    repository = repository.resolve()
    campaign_root = campaign_root.resolve()
    derivation_path = require_within_campaign_root(derivation_path, campaign_root)
    auditor_path = require_within_campaign_root(auditor_path, campaign_root)
    output_directory = output_directory.resolve()
    try:
        output_directory.relative_to(campaign_root)
    except ValueError as error:
        raise ArenaCorpusError("output directory must stay inside the fresh campaign root") from error
    derivation_sha = _require_sha256(
        expected_derivation_sha256, "expected_derivation_sha256"
    )
    if sha256_file(derivation_path) != derivation_sha:
        raise ArenaCorpusError("arena derivation hash differs from the explicitly approved hash")
    try:
        derivation = validate_arena_derivation(
            derivation_path,
            repository=repository,
            expected_sha256=derivation_sha,
        )
    except ValueError as error:
        raise ArenaCorpusError(str(error)) from error
    if derivation["window"]["role"] != "training":
        raise ArenaCorpusError("only pre-assigned training windows may produce corpus rows")
    if derivation["window_plan"]["sha256"] != APPROVED_WINDOW_PLAN_SHA256:
        raise ArenaCorpusError("derivation does not use the frozen campaign window plan")
    if derivation["exclusion_registry"]["sha256"] != APPROVED_EXCLUSION_SHA256:
        raise ArenaCorpusError("derivation does not use the frozen pre-T0 exclusions")
    auditor = parse_auditor_tsv(auditor_path)
    eligible = crosscheck_auditor(auditor, derivation, repository=repository)
    ranking_ids = sorted(
        game_id for game_id, game in eligible.items()
        if "opponent-action-ranking-reanalysis-candidate" in game["uses"]
    )

    with tempfile.TemporaryDirectory(prefix="jacek-arena-reanalysis-") as temporary:
        temporary_path = Path(temporary)
        executable = temporary_path / "arena-corpus-reanalyzer"
        compile_command, source_inventory = _compile_reanalyzer(
            repository=repository, output=executable, cxx=cxx
        )
        binary_sha = sha256_file(executable)
        raw_reanalysis, run_command = _run_reanalyzer(
            executable=executable,
            auditor=auditor,
            ranking_game_ids=ranking_ids,
            maximum_analyzed_decisions=maximum_analyzed_decisions,
            maximum_alternatives=maximum_alternatives,
            maximum_pairs_per_game=maximum_pairs_per_game,
            search_width=search_width,
            generator=generator,
            timeout_seconds=timeout_seconds,
            temporary_directory=temporary_path,
        )
    reanalysis = parse_reanalysis(raw_reanalysis, auditor)
    rows = build_rows(
        reanalysis,
        auditor,
        derivation,
        derivation_sha256=derivation_sha,
        generated_at_utc=generated_at_utc,
        terminal_value_weight=terminal_value_weight,
    )

    window_plan_path = _resolve_reference(derivation["window_plan"]["path"], repository)
    exclusion_path = _resolve_reference(derivation["exclusion_registry"]["path"], repository)
    contract: CampaignContract = load_contract(window_plan_path)
    bindings = load_arena_game_bindings(
        [derivation_path], campaign_root=campaign_root, repository=repository
    )
    validator = FreshCorpusValidator(
        contract,
        excluded_game_ids=load_excluded_game_ids(exclusion_path),
        approved_producer_source_sha256={str(derivation["source"]["sha256"])},
        arena_game_bindings=bindings,
        training_only=True,
        max_pairs_per_decision=4,
        max_pairs_per_game=maximum_pairs_per_game,
    )
    validated, summary = validator.validate_rows(rows)
    if len(validated) != len(rows):  # defensive; validator is all-or-error
        raise ArenaCorpusError("fresh corpus validator did not account for every derived row")

    corpus_payload = b"".join(canonical_json_bytes(row) for row in rows)
    corpus_path = write_content_addressed_bytes(
        output_directory / "rows", corpus_payload, ".jsonl"
    )
    reanalysis_path = write_content_addressed_bytes(
        output_directory / "reanalysis", raw_reanalysis, ".jsonl"
    )
    analysis_source_identity = sha256_bytes(canonical_json_bytes(source_inventory))
    manifest = {
        "campaign": dict(derivation["campaign"]),
        "config": {
            "generator": generator,
            "maximum_alternatives": maximum_alternatives,
            "maximum_analyzed_decisions_per_game": maximum_analyzed_decisions,
            "maximum_pairs_per_decision": 4,
            "maximum_pairs_per_game": maximum_pairs_per_game,
            "search_width": search_width,
            "terminal_value_weight": terminal_value_weight,
            "work_checkpoints": [30000, 100000],
        },
        "generated_at_utc": generated_at_utc,
        "inputs": {
            "arena_derivation": {
                "path": str(derivation_path.relative_to(repository)),
                "sha256": derivation_sha,
            },
            "auditor_tsv": {
                "path": str(auditor.path.relative_to(repository)),
                "sha256": auditor.sha256,
                "bytes": auditor.byte_count,
            },
        },
        "output": {
            "corpus": {
                "path": str(corpus_path.relative_to(repository)),
                "sha256": sha256_file(corpus_path),
                "bytes": corpus_path.stat().st_size,
            },
            "reanalyzer_transcript": {
                "path": str(reanalysis_path.relative_to(repository)),
                "sha256": reanalysis.sha256,
                "bytes": reanalysis.byte_count,
            },
        },
        "reanalyzer": {
            "binary_sha256": binary_sha,
            "compile_command": compile_command,
            "run_arguments": run_command[1:],
            "source_identity_sha256": analysis_source_identity,
            "source_inventory": source_inventory,
            "summary": dict(reanalysis.meta),
        },
        "schema": BUILD_MANIFEST_SCHEMA,
        "summary": summary.to_json(),
        "window": dict(derivation["window"]),
    }
    manifest_path = write_content_addressed_json(
        output_directory / "manifests", manifest
    )
    return {
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reanalysis_path": str(reanalysis_path),
        "reanalysis_sha256": reanalysis.sha256,
        "summary": summary.to_json(),
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Build clean-room jacek_arena_bfm rows from one fresh training window"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--derivation", type=Path, required=True)
    parser.add_argument("--expected-derivation-sha256", required=True)
    parser.add_argument("--auditor-tsv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--generated-at-utc", default=_utc_now())
    parser.add_argument("--terminal-value-weight", type=float, default=0.25)
    parser.add_argument("--maximum-analyzed-decisions", type=int, default=2)
    parser.add_argument("--maximum-alternatives", type=int, default=6)
    parser.add_argument("--maximum-pairs-per-game", type=int, default=16)
    parser.add_argument("--search-width", type=int, default=8)
    parser.add_argument(
        "--generator",
        choices=("tactical-progressive", "priority-beam"),
        default="tactical-progressive",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    for name, value, maximum in (
        ("maximum_analyzed_decisions", args.maximum_analyzed_decisions, 64),
        ("maximum_alternatives", args.maximum_alternatives, 32),
        ("maximum_pairs_per_game", args.maximum_pairs_per_game, 32),
        ("search_width", args.search_width, 32),
    ):
        if value < 0 or value > maximum:
            parser.error(f"--{name.replace('_', '-')} must be in [0, {maximum}]")
    if args.maximum_analyzed_decisions and (
        args.maximum_alternatives == 0 or args.search_width < 2
    ):
        parser.error("enabled reanalysis requires alternatives > 0 and search width >= 2")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = build_arena_corpus(
        repository=args.repository,
        campaign_root=args.campaign_root,
        derivation_path=args.derivation,
        expected_derivation_sha256=args.expected_derivation_sha256,
        auditor_path=args.auditor_tsv,
        output_directory=args.output_directory,
        generated_at_utc=args.generated_at_utc,
        terminal_value_weight=args.terminal_value_weight,
        maximum_analyzed_decisions=args.maximum_analyzed_decisions,
        maximum_alternatives=args.maximum_alternatives,
        maximum_pairs_per_game=args.maximum_pairs_per_game,
        search_width=args.search_width,
        generator=args.generator,
        timeout_seconds=args.timeout_seconds,
        cxx=args.cxx,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
