#!/usr/bin/env python3
"""Deterministic control plane for the gated Jacek self-search campaign."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Callable, Iterable, Mapping, Sequence

import jacek_replay_corpus as corpus
import jacek_replay_features as features
from jacek_replay_workflow import (
    StageManager,
    artifact_snapshot,
    canonical_json_bytes,
    environment_identity,
    sha256,
    validate_canonical_workflow_chain,
)


AUTO_CAMPAIGN_ID = "selfsearch-auto-20260825-v4"
PILOT_CAMPAIGN_ID = "selfsearch-pilot-20260825-v4"
FULL_CAMPAIGN_ID = "selfsearch-full-20260825-v4"
GAME_PLAN_SCHEMA = "papersoccer.jacek-selfsearch-game-plan.v1"
GAME_MANIFEST_SCHEMA = "papersoccer.jacek-selfsearch-games.v1"
POSITION_SCHEMA = "papersoccer.jacek-replay-position.v1"
POSITION_MANIFEST_SCHEMA = "papersoccer.jacek-replay-position-manifest.v1"
HARD_SELECTION_SCHEMA = "papersoccer.jacek-selfsearch-hard-selection.v1"
INCUMBENT_SELECTION_SCHEMA = "papersoccer.jacek-selfsearch-incumbent.v1"
PILOT_DECISION_SCHEMA = "papersoccer.jacek-selfsearch-pilot-decision.v1"
FINAL_DECISION_SCHEMA = "papersoccer.jacek-selfsearch-final-decision.v1"
CAMPAIGN_STATUS_SCHEMA = "papersoccer.jacek-selfsearch-campaign-status.v1"
CAMPAIGN_RECEIPT_SCHEMA = "papersoccer.jacek-selfsearch-chunk-receipt.v1"
CAMPAIGN_SUMMARY_SCHEMA = "papersoccer.jacek-selfsearch-campaign-summary.v1"
BUILD_MANIFEST_SCHEMA = "papersoccer.jacek-selfsearch-release-build.v1"
SEARCH_TEACHER_SCHEMA = corpus.SEARCH_TEACHER_SCHEMA
RANK4_TEACHER_SCHEMA = corpus.RANK4_TEACHER_SCHEMA

PILOT_OPENING_SEED = 2026082505
FULL_OPENING_SEED = 2026082507
PILOT_GAME_SEED = 2026082501
FULL_GAME_SEED = 2026082503
MINIMUM_FREE_BYTES = 12 * 1024**3
POSITION_CHUNK_GAMES = 25
SEARCH_MAX_ACTIONS = 250
SEARCH_MAX_PARTIAL_PATHS = 50_000
SEARCH_SAFETY_MS = 60_000

_CAMPAIGN_LOCK_FD: int | None = None

RANK4_ACTOR_SOURCE_PATHS = (
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/bots/mcts_internal.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "submissions/codingame/bots/rank_4/replay_book.hpp",
    "submissions/codingame/bots/rank_4/replay_value_model.hpp",
    "submissions/codingame/bots/rank_4/teacher_residual_model.hpp",
    "submissions/codingame/bots/rank_4/bot.cpp",
)
JACEK_NN_ACTOR_SOURCE_PATHS = (
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/bots/mcts_internal.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "submissions/codingame/bots/jacek_nn/replay_book.hpp",
    "submissions/codingame/bots/jacek_nn/replay_value_model.hpp",
    "submissions/codingame/bots/jacek_nn/teacher_residual_model.hpp",
    "submissions/codingame/bots/jacek_nn/bot.cpp",
)
BFM_RUNTIME_SOURCE_PATHS = (
    "include/papersoccer/bot.hpp",
    "src/bots/bot.cpp",
    "src/bots/jacek_replay_bfm/features.cpp",
    "src/bots/jacek_replay_bfm/model.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp",
)
CONTINUATION_SOURCE_PATHS = (
    *BFM_RUNTIME_SOURCE_PATHS,
    "tools/jacek_replay_continuations.cpp",
    "tools/jacek_replay_continuations_internal.hpp",
    *RANK4_ACTOR_SOURCE_PATHS,
    *JACEK_NN_ACTOR_SOURCE_PATHS,
)
SEARCH_TEACHER_SOURCE_PATHS = (
    "include/papersoccer/bot.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/bots/bot.cpp",
    "src/bots/mcts_internal.hpp",
    "src/bots/jacek_replay_bfm/features.cpp",
    "src/bots/jacek_replay_bfm/model.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "tools/jacek_replay_bfm_search_teacher.cpp",
    "tools/jacek_replay_bfm_search_teacher_internal.hpp",
)
JACEK_NN_COMPARISON_SOURCE_PATHS = (
    *JACEK_NN_ACTOR_SOURCE_PATHS,
    "tools/jacek_replay_bfm_jacek_nn_control.cpp",
)
SHARED_CORE_SOURCE_PATHS = (
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "src/bots/mcts_internal.hpp",
)
COMPARISON_SOURCE_PATHS = (
    "tools/jacek_replay_bfm_comparison.cpp",
    "tools/jacek_replay_bfm_gate_internal.hpp",
)

PILOT_QUOTAS = {
    "incumbent-selfplay": 1_000,
    "incumbent-p1-vs-rank4": 200,
    "incumbent-p2-vs-rank4": 200,
    "incumbent-p1-vs-jacek-nn": 200,
    "incumbent-p2-vs-jacek-nn": 200,
    "incumbent-p1-vs-runner-up": 100,
    "incumbent-p2-vs-runner-up": 100,
}
FULL_QUOTAS = {
    "student-selfplay": 5_000,
    "student-p1-vs-rank4": 1_000,
    "student-p2-vs-rank4": 1_000,
    "student-p1-vs-jacek-nn": 1_000,
    "student-p2-vs-jacek-nn": 1_000,
    "student-p1-vs-prior-incumbent": 500,
    "student-p2-vs-prior-incumbent": 500,
}

PILOT_CONFIGURATION = {
    "campaign_id": PILOT_CAMPAIGN_ID,
    "games": 2_000,
    "game_chunk_size": 25,
    "game_workers": 10,
    "positions_per_game": 24,
    "bfm_actor_tree_nodes": 8_000,
    "rank4_actor_nodes": 16_000,
    "jacek_nn_actor_nodes": 64_000,
    "exploration": 0.5,
    "fpu": 0.5,
    "early_exploration_percent": 15,
    "early_exploration_turns": 8,
    "bfm_shallow_tree_nodes": 64_000,
    "bfm_deep_tree_nodes": 500_000,
    "rank4_shallow_nodes": 32_000,
    "rank4_deep_nodes": 400_000,
    "hard_fraction_numerator": 1,
    "hard_fraction_denominator": 4,
    "adjudicator_positions": 2_000,
    "adjudicator_tree_nodes": 1_000_000,
    "new_rows_per_batch": 128,
    "anchor_rows_per_batch": 128,
    "training_seeds": [20260901, 20260902, 20260903],
}
FULL_CONFIGURATION = {
    **{key: value for key, value in PILOT_CONFIGURATION.items() if key not in {
        "campaign_id", "games", "positions_per_game", "adjudicator_positions",
        "new_rows_per_batch", "anchor_rows_per_batch", "training_seeds"
    }},
    "campaign_id": FULL_CAMPAIGN_ID,
    "games": 10_000,
    "positions_per_game": 20,
    "adjudicator_positions": 4_000,
    "new_rows_per_batch": 192,
    "anchor_rows_per_batch": 64,
    "training_seeds": [20260904, 20260905, 20260906],
}


def _load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as destination:
            temporary = pathlib.Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: pathlib.Path, value: object) -> None:
    _atomic_bytes(path, canonical_json_bytes(value, pretty=True))


def _append_field(material: bytearray, value: object) -> None:
    encoded = str(value).encode("utf-8")
    material.extend(len(encoded).to_bytes(4, "little"))
    material.extend(encoded)


def stable_identifier(kind: str, *fields: object) -> str:
    material = bytearray()
    _append_field(material, kind)
    for field in fields:
        _append_field(material, field)
    return f"{kind}:{hashlib.sha256(material).hexdigest()}"


def _round_validation_metrics(round_directory: pathlib.Path) -> dict:
    manifest = _load_json(
        round_directory / "model/jacek_replay_bfm.runtime.json",
        f"round {round_directory.name} model",
    )
    training = manifest.get("training", {})
    chosen = training.get("chosen_seed")
    report = next(
        (
            item
            for item in training.get("seed_reports", [])
            if isinstance(item, dict) and item.get("seed") == chosen
        ),
        None,
    )
    if not isinstance(report, dict) or not isinstance(report.get("validation"), dict):
        raise ValueError(f"round model has no selected validation report: {round_directory}")
    return report["validation"]


def select_incumbent(
    evaluation_summary: pathlib.Path,
    canonical_campaign: pathlib.Path,
) -> dict:
    """Rank R0/R1/R2 from the complete direct league using frozen tie-breaks."""

    summary = _load_json(evaluation_summary, "completed game evaluation")
    if (
        summary.get("schema") != "papersoccer.jacek-replay-postcampaign-summary.v1"
        or summary.get("games") != 5_000
        or not isinstance(summary.get("step2"), dict)
        or not isinstance(summary.get("sequential_latency_audit"), dict)
    ):
        raise ValueError("game evaluation is not complete and validated")
    step2 = summary["step2"]
    expected = {"r0-vs-r1", "r1-vs-r2", "r0-vs-r2"}
    if set(step2) != expected:
        raise ValueError("round league does not contain all direct matchups")
    scores = {
        round_: {
            "round": round_,
            "wins": 0,
            "matchup_wins": [],
            "sweep_differential": 0,
            "color_wins": [0, 0],
            "illegal": 0,
            "unfinished": 0,
        }
        for round_ in range(3)
    }
    for matchup in sorted(expected):
        lower, higher = (int(part[1:]) for part in matchup.split("-vs-"))
        lower_summary = step2[matchup].get("lower_round_as_candidate")
        if not isinstance(lower_summary, dict) or lower_summary.get("games") != 1_000:
            raise ValueError(f"round league matchup is malformed: {matchup}")
        lower_wins = lower_summary.get("wins")
        if not isinstance(lower_wins, int):
            raise ValueError(f"round league wins are missing: {matchup}")
        higher_wins = 1_000 - lower_wins - int(lower_summary.get("unfinished", 0))
        scores[lower]["wins"] += lower_wins
        scores[higher]["wins"] += higher_wins
        scores[lower]["matchup_wins"].append(lower_wins)
        scores[higher]["matchup_wins"].append(higher_wins)
        sweeps = int(lower_summary.get("opening_sweeps", 0))
        losses = int(lower_summary.get("opening_losses", 0))
        scores[lower]["sweep_differential"] += sweeps - losses
        scores[higher]["sweep_differential"] += losses - sweeps
        colors = lower_summary.get("colors", {})
        lower_colors = [int(colors[str(color)]["wins"]) for color in (0, 1)]
        higher_colors = [500 - value for value in reversed(lower_colors)]
        for color in (0, 1):
            scores[lower]["color_wins"][color] += lower_colors[color]
            scores[higher]["color_wins"][color] += higher_colors[color]
        for field in ("illegal", "unfinished"):
            count = int(lower_summary.get(field, 0))
            scores[lower][field] += count
            scores[higher][field] += count
    eligible = []
    for round_, record in scores.items():
        record["worst_color_wins"] = min(record["color_wins"])
        metrics = _round_validation_metrics(canonical_campaign / f"round-{round_}")
        record["validation_huber"] = metrics["weighted_huber"]
        record["runtime"] = str(
            (canonical_campaign / f"round-{round_}/model/jacek_replay_bfm.runtime").resolve()
        )
        record["runtime_sha256"] = sha256(pathlib.Path(record["runtime"]))
        if record["illegal"] == 0 and record["unfinished"] == 0:
            eligible.append(record)
    if len(eligible) < 2:
        raise ValueError("fewer than two league runtimes are operationally eligible")
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["wins"],
            -min(item["matchup_wins"]),
            -item["sweep_differential"],
            -item["worst_color_wins"],
            item["validation_huber"],
            -item["round"],
        ),
    )
    return {
        "schema": INCUMBENT_SELECTION_SCHEMA,
        "evaluation": artifact_snapshot(evaluation_summary),
        "ranking_policy": (
            "total wins, worst matchup, sweep differential, worst color, "
            "validation Huber, later round"
        ),
        "ranked": ranked,
        "incumbent": ranked[0],
        "runner_up": ranked[1],
    }


def make_game_plan(
    *, campaign_id: str, seed: int, quotas: Mapping[str, int]
) -> dict:
    if not campaign_id or not quotas or any(
        not isinstance(value, int) or value <= 0 for value in quotas.values()
    ):
        raise ValueError("game-plan configuration is invalid")
    schedule = [mode for mode, count in quotas.items() for _ in range(count)]
    rng = random.Random(seed ^ 0x243F6A8885A308D3)
    rng.shuffle(schedule)
    rows = []
    for ordinal, mode in enumerate(schedule):
        rows.append(
            {
                "game_ordinal": ordinal,
                "actor_mode": mode,
                "base_seed": (
                    seed + ordinal * 0x9E3779B97F4A7C15
                ) & ((1 << 64) - 1),
            }
        )
    return {
        "schema": GAME_PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "seed": seed,
        "quotas": dict(quotas),
        "games": len(rows),
        "rows": rows,
    }


def render_game_plan_tsv(plan: Mapping[str, object]) -> bytes:
    if plan.get("schema") != GAME_PLAN_SCHEMA or not isinstance(plan.get("rows"), list):
        raise ValueError("game plan cannot be rendered")
    lines = ["game_ordinal\tactor_mode\tbase_seed"]
    for expected_ordinal, row in enumerate(plan["rows"]):
        if (
            not isinstance(row, dict)
            or row.get("game_ordinal") != expected_ordinal
            or not isinstance(row.get("actor_mode"), str)
            or not isinstance(row.get("base_seed"), int)
        ):
            raise ValueError("game plan row is malformed")
        lines.append(
            f"{expected_ordinal}\t{row['actor_mode']}\t{row['base_seed']}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def merge_game_chunks(
    *,
    campaign_id: str,
    plan_path: pathlib.Path,
    chunk_tsvs: Sequence[pathlib.Path],
    chunk_manifests: Sequence[pathlib.Path],
) -> tuple[bytes, dict]:
    if len(chunk_tsvs) != len(chunk_manifests) or not chunk_tsvs:
        raise ValueError("game chunk sets are missing or unequal")
    plan = _load_json(plan_path, "game plan")
    plan_rows = plan.get("rows")
    if (
        plan.get("schema") != GAME_PLAN_SCHEMA
        or plan.get("campaign_id") != campaign_id
        or not isinstance(plan_rows, list)
        or len(plan_rows) != plan.get("games")
    ):
        raise ValueError("game plan identity is stale")
    collected = []
    chunk_bindings = []
    common_configuration = None
    common_models = None
    for chunk_ordinal, (tsv_path, manifest_path) in enumerate(
        zip(chunk_tsvs, chunk_manifests, strict=True)
    ):
        chunk = _load_json(manifest_path, f"game chunk {chunk_ordinal}")
        if (
            chunk.get("schema") != GAME_MANIFEST_SCHEMA
            or chunk.get("campaign_id") != campaign_id
            or chunk.get("successful_games") != chunk.get("requested_games")
            or chunk.get("bindings", {}).get("output_sha256") != sha256(tsv_path)
        ):
            raise ValueError(f"game chunk {chunk_ordinal} is invalid")
        lines = tsv_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "group_id\tsource\twinner\ttranscript":
            raise ValueError(f"game chunk {chunk_ordinal} TSV is invalid")
        rows = chunk.get("rows")
        if not isinstance(rows, list) or len(rows) != len(lines) - 1:
            raise ValueError(f"game chunk {chunk_ordinal} rows disagree")
        configuration = chunk.get("configuration")
        bindings = chunk.get("bindings")
        source_fields = (
            "producer_source_sha256",
            "rank4_actor_source_sha256",
            "jacek_nn_actor_source_sha256",
        )
        if not isinstance(configuration, dict) or not isinstance(bindings, dict):
            raise ValueError(f"game chunk {chunk_ordinal} provenance is missing")
        for field in source_fields:
            value = configuration.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                or bindings.get(field) != value
            ):
                raise ValueError(
                    f"game chunk {chunk_ordinal} {field} is invalid"
                )
        models = {
            key: bindings.get(key)
            for key in (
                "incumbent_model_sha256",
                "runner_up_model_sha256",
                "roots_sha256",
                *source_fields,
            )
        }
        if common_configuration is None:
            common_configuration = configuration
            common_models = models
        elif configuration != common_configuration or models != common_models:
            raise ValueError("game chunks use different producers or configurations")
        for chunk_row_ordinal, (line, row) in enumerate(
            zip(lines[1:], rows, strict=True)
        ):
            fields = line.split("\t")
            if (
                len(fields) != 4
                or not isinstance(row, dict)
                or row.get("row_ordinal") != chunk_row_ordinal
                or type(row.get("game_ordinal")) is not int
                or type(row.get("base_seed")) is not int
                or row.get("root_group_id") != fields[0]
                or fields[1] != campaign_id
                or str(row.get("winner")) != fields[2]
                or not isinstance(row.get("actor_mode"), str)
                or not isinstance(row.get("game_id"), str)
                or not row["game_id"]
            ):
                raise ValueError("game chunk row is malformed")
            if row.get("transcript_sha256") != hashlib.sha256(
                fields[3].encode("utf-8")
            ).hexdigest():
                raise ValueError("game chunk transcript binding is stale")
            collected.append((row["game_ordinal"], fields, dict(row)))
        chunk_bindings.append(
            {
                "chunk_ordinal": chunk_ordinal,
                "tsv": artifact_snapshot(tsv_path),
                "manifest": artifact_snapshot(manifest_path),
            }
        )
    collected.sort(key=lambda item: item[0])
    expected_ordinals = list(range(int(plan.get("games", -1))))
    if [item[0] for item in collected] != expected_ordinals:
        raise ValueError("game chunks do not exactly cover the frozen plan")
    for (game_ordinal, _fields, row), planned in zip(
        collected, plan_rows, strict=True
    ):
        if (
            not isinstance(planned, dict)
            or planned.get("game_ordinal") != game_ordinal
            or row.get("actor_mode") != planned.get("actor_mode")
            or row.get("base_seed") != planned.get("base_seed")
        ):
            raise ValueError("game chunk row differs from the frozen plan")
    output_lines = ["group_id\tsource\twinner\ttranscript"]
    rows = []
    quotas: collections.Counter[str] = collections.Counter()
    for row_ordinal, (game_ordinal, fields, row) in enumerate(collected):
        output_lines.append("\t".join(fields))
        row["row_ordinal"] = row_ordinal
        rows.append(row)
        quotas[str(row["actor_mode"])] += 1
    if dict(quotas) != plan.get("quotas"):
        raise ValueError("merged game chunks do not satisfy exact quotas")
    payload = ("\n".join(output_lines) + "\n").encode("utf-8")
    manifest = {
        "schema": GAME_MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "requested_games": len(rows),
        "successful_games": len(rows),
        "configuration": common_configuration,
        "quotas": dict(quotas),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "bindings": {
            **(common_models or {}),
            "plan": artifact_snapshot(plan_path),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "chunks": chunk_bindings,
        },
        "rows": rows,
    }
    return payload, manifest


def _root_splits(roots_manifest: pathlib.Path) -> dict[str, str]:
    roots = _load_json(roots_manifest, "replay roots")
    assignments = {}
    for row in roots.get("accepted", []):
        if not isinstance(row, dict):
            raise ValueError("replay roots contain a malformed row")
        group_id, split = row.get("group_id"), row.get("split")
        if not isinstance(group_id, str) or split not in {"train", "validation", "test"}:
            raise ValueError("replay root split assignment is invalid")
        assignments[group_id] = split
    if not assignments:
        raise ValueError("replay roots contain no accepted split assignments")
    return assignments


def _sample_suffix_boundaries(
    action_count: int, prefix_turns: int, maximum: int
) -> list[int]:
    if prefix_turns < 0 or prefix_turns >= action_count or maximum <= 0:
        raise ValueError("generated-game suffix cannot be sampled")
    count = action_count - prefix_turns
    retained = min(count, maximum)
    retained -= retained % 4
    if retained == 0:
        return []
    return [prefix_turns + index * count // retained for index in range(retained)]


def freeze_positions(
    *,
    campaign_id: str,
    games_tsv: pathlib.Path,
    games_manifest: pathlib.Path,
    roots_manifest: pathlib.Path,
    maximum_per_game: int,
) -> tuple[bytes, dict]:
    manifest = _load_json(games_manifest, "generated-game manifest")
    if (
        manifest.get("schema") != GAME_MANIFEST_SCHEMA
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("successful_games") != manifest.get("requested_games")
        or manifest.get("bindings", {}).get("output_sha256") != sha256(games_tsv)
    ):
        raise ValueError("generated-game manifest is stale or incomplete")
    lines = games_tsv.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "group_id\tsource\twinner\ttranscript":
        raise ValueError("generated-game TSV header is invalid")
    game_rows = [line.split("\t") for line in lines[1:]]
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(game_rows) != len(manifest_rows):
        raise ValueError("generated-game TSV/manifest counts disagree")
    splits = _root_splits(roots_manifest)
    output = [
        "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
    ]
    rows = []
    split_counts: collections.Counter[str] = collections.Counter()
    for ordinal, (fields, game) in enumerate(zip(game_rows, manifest_rows, strict=True)):
        if len(fields) != 4 or not isinstance(game, dict):
            raise ValueError("generated-game row is malformed")
        root_group_id, source, winner_raw, transcript = fields
        if game.get("row_ordinal") != ordinal or game.get("root_group_id") != root_group_id:
            raise ValueError("generated-game row lineage is stale")
        if game.get("transcript_sha256") != hashlib.sha256(transcript.encode()).hexdigest():
            raise ValueError("generated-game transcript hash is stale")
        split = splits.get(root_group_id)
        if split is None:
            raise ValueError("generated game has no frozen root split")
        winner = int(winner_raw)
        actions = transcript.split("/")
        selected = set(
            _sample_suffix_boundaries(
                len(actions), int(game.get("prefix_turns", -1)), maximum_per_game
            )
        )
        state = features.ReplayState()
        prefix: list[str] = []
        for turn, action in enumerate(actions):
            if turn in selected:
                prefix_text = "/".join(prefix)
                position_id = stable_identifier(
                    "position",
                    campaign_id,
                    game.get("game_id"),
                    root_group_id,
                    turn,
                    hashlib.sha256(prefix_text.encode()).hexdigest(),
                )
                mover = state.to_move
                group_id = str(game.get("game_id"))
                output.append(
                    "\t".join(
                        (
                            position_id,
                            root_group_id,
                            group_id,
                            source,
                            split,
                            str(winner),
                            str(mover),
                            prefix_text,
                        )
                    )
                )
                rows.append(
                    {
                        "position_id": position_id,
                        "row_ordinal": len(rows),
                        "game_id": game.get("game_id"),
                        "game_row_ordinal": ordinal,
                        "turn": turn,
                        "root_group_id": root_group_id,
                        "split": split,
                        "mover": mover,
                        "winner": winner,
                    }
                )
                split_counts[split] += 1
            mover = state.to_move
            features.apply_complete_turn(state, mover, action)
            prefix.append(action)
        if state.winner != winner:
            raise ValueError("generated-game transcript winner is invalid")
    payload = ("\n".join(output) + "\n").encode("utf-8")
    position_manifest = {
        "schema": POSITION_MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "games": artifact_snapshot(games_tsv),
        "game_manifest": artifact_snapshot(games_manifest),
        "roots": artifact_snapshot(roots_manifest),
        "maximum_positions_per_game": maximum_per_game,
        "positions": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": rows,
    }
    return payload, position_manifest


def _teacher_value(row: Mapping[str, object]) -> float:
    schema = row.get("schema")
    mover = row.get("mover")
    proven = row.get("proven_winner")
    if mover not in (0, 1) or (proven is not None and proven not in (0, 1)):
        raise ValueError("teacher mover/proof is invalid")
    if schema == SEARCH_TEACHER_SCHEMA:
        value = row.get("teacher_value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("search teacher value is invalid")
        value = float(value)
        if value < -1.0 or value > 1.0:
            raise ValueError("search teacher value is outside [-1,1]")
        if proven is not None and value != (1.0 if proven == mover else -1.0):
            raise ValueError("search proof/value disagree")
        return value
    if schema in {corpus.TEACHER_SCHEMA, RANK4_TEACHER_SCHEMA}:
        if proven is not None:
            return 1.0 if proven == mover else -1.0
        score = row.get("root_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("Rank-4 teacher score is invalid")
        return (1.0 if mover == 0 else -1.0) * math.tanh(float(score) / 12_000.0)
    raise ValueError("unknown teacher schema")


def load_labels(path: pathlib.Path, expected_schema: str) -> dict[str, dict]:
    labels = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"label line {line_number} is invalid JSON") from error
        if not isinstance(row, dict) or row.get("schema") != expected_schema:
            raise ValueError(f"label line {line_number} has the wrong schema")
        position_id = row.get("position_id")
        if not isinstance(position_id, str) or position_id in labels:
            raise ValueError("label position IDs are missing or duplicate")
        if expected_schema in {SEARCH_TEACHER_SCHEMA, RANK4_TEACHER_SCHEMA}:
            try:
                corpus.sample_from_teacher_row(row)
            except ValueError as error:
                label_kind = (
                    "search"
                    if expected_schema == SEARCH_TEACHER_SCHEMA
                    else "Rank-4"
                )
                raise ValueError(
                    f"{label_kind} label line {line_number} violates its teacher contract"
                ) from error
        _teacher_value(row)
        labels[position_id] = row
    if not labels:
        raise ValueError("teacher label file is empty")
    return labels


def select_hard_positions(
    *,
    positions_tsv: pathlib.Path,
    search_labels: pathlib.Path,
    rank4_labels: pathlib.Path,
    numerator: int = 1,
    denominator: int = 4,
) -> tuple[bytes, dict]:
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("hard-position fraction is invalid")
    lines = positions_tsv.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("position_id\t"):
        raise ValueError("position manifest TSV is invalid")
    fields = lines[0].split("\t")
    rows = [dict(zip(fields, line.split("\t"), strict=True)) for line in lines[1:]]
    search = load_labels(search_labels, SEARCH_TEACHER_SCHEMA)
    rank4 = load_labels(rank4_labels, RANK4_TEACHER_SCHEMA)
    position_ids = {row["position_id"] for row in rows}
    if set(search) != position_ids or set(rank4) != position_ids:
        raise ValueError("shallow teachers do not cover identical positions")
    by_game: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        position_id = row["position_id"]
        left, right = search[position_id], rank4[position_id]
        left_value, right_value = _teacher_value(left), _teacher_value(right)
        outcome = 1.0 if int(row["winner"]) == int(row["mover"]) else -1.0
        row = dict(row)
        row["hard_key"] = (
            -int((left_value >= 0.0) != (right_value >= 0.0)),
            -abs(left_value - right_value),
            -int((left_value >= 0.0) != (outcome >= 0.0) or
                 (right_value >= 0.0) != (outcome >= 0.0)),
            min(abs(left_value), abs(right_value)),
            position_id,
        )
        by_game[row["group_id"]].append(row)
    selected = set()
    for game_rows in by_game.values():
        selected_numerator = len(game_rows) * numerator
        if selected_numerator % denominator != 0:
            raise ValueError("per-game position count cannot satisfy exact hard fraction")
        count = selected_numerator // denominator
        selected.update(
            row["position_id"]
            for row in sorted(game_rows, key=lambda item: item["hard_key"])[:count]
        )
    output = [lines[0]]
    output.extend(
        line
        for line, row in zip(lines[1:], rows, strict=True)
        if row["position_id"] in selected
    )
    payload = ("\n".join(output) + "\n").encode("utf-8")
    manifest = {
        "schema": HARD_SELECTION_SCHEMA,
        "positions": artifact_snapshot(positions_tsv),
        "search_labels": artifact_snapshot(search_labels),
        "rank4_labels": artifact_snapshot(rank4_labels),
        "fraction": [numerator, denominator],
        "selected": len(selected),
        "games": len(by_game),
        "position_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selected)).encode("utf-8")
        ).hexdigest(),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, manifest


def merge_deep_labels(
    *, shallow: pathlib.Path, deep: pathlib.Path, expected_schema: str
) -> bytes:
    shallow_rows = load_labels(shallow, expected_schema)
    deep_rows = load_labels(deep, expected_schema)
    if not set(deep_rows) < set(shallow_rows):
        raise ValueError("deep labels must be a strict subset of shallow labels")
    merged = {**shallow_rows, **deep_rows}
    return b"".join(
        canonical_json_bytes(merged[position_id])
        for position_id in shallow_rows
    )


def common_adjudicator_positions(
    positions_tsv: pathlib.Path, maximum: int,
    prior_manifests: Sequence[pathlib.Path] = (),
) -> bytes:
    lines = positions_tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    fields = header.split("\t")
    rows = [dict(zip(fields, line.split("\t"), strict=True)) for line in lines[1:]]

    def fingerprint(row: Mapping[str, str]) -> bytes:
        state = features.ReplayState()
        prefix = row["prefix"]
        if prefix:
            for action in prefix.split("/"):
                features.apply_complete_turn(state, state.to_move, action)
        return corpus.canonical_feature_fingerprint(features.encode_active(state))

    excluded: set[bytes] = set()
    if prior_manifests:
        import jacek_replay_train as training

        for manifest in prior_manifests:
            shard = training.load_csr_shard(manifest)
            for index in range(len(shard)):
                excluded.add(
                    corpus.canonical_feature_fingerprint(
                        shard.active(index).tolist()
                    )
                )
    excluded.update(fingerprint(row) for row in rows if row["split"] == "train")
    validation = []
    selected_fingerprints: set[bytes] = set()
    for row in sorted(
        (row for row in rows if row["split"] == "validation"),
        key=lambda item: hashlib.sha256(item["position_id"].encode()).digest(),
    ):
        canonical = fingerprint(row)
        if canonical in excluded or canonical in selected_fingerprints:
            continue
        validation.append(row)
        selected_fingerprints.add(canonical)
        if len(validation) == maximum:
            break
    if len(validation) != maximum:
        raise ValueError(
            "not enough split-isolated validation positions for common adjudicator"
        )
    by_id = {row["position_id"]: line for row, line in zip(rows, lines[1:], strict=True)}
    selected_ids = {row["position_id"] for row in validation}
    ordered = [
        by_id[row["position_id"]] for row in rows
        if row["position_id"] in selected_ids
    ]
    return (header + "\n" + "\n".join(ordered) + "\n").encode()


def validate_gate_report(
    path: pathlib.Path, *, pairs: int, opponent: str, time_ms: int = 20,
    bank_classification: str | None = None,
) -> dict:
    report = _load_json(path, "game gate report")
    configuration = report.get("configuration", {})
    results = report.get("results")
    if (
        report.get("schema") != "papersoccer.jacek-replay-bfm-comparison.v1"
        or configuration.get("pairs") != pairs
        or configuration.get("time_ms") != time_ms
        or configuration.get("exploration") != 0.5
        or configuration.get("fpu") != 0.5
        or configuration.get("opponent") != opponent
        or (
            bank_classification is not None
            and configuration.get("opening_bank_classification") != bank_classification
        )
        or not isinstance(results, list)
        or len(results) != 2 * pairs
    ):
        raise ValueError(f"game gate report configuration is invalid: {path}")
    return report


def _gate_counts(report: dict) -> dict:
    results = report["results"]
    wins = sum(game.get("winner") == game.get("candidate_player") for game in results)
    return {
        "games": len(results),
        "wins": wins,
        "colors": [
            sum(
                game.get("candidate_player") == color and game.get("winner") == color
                for game in results
            )
            for color in (0, 1)
        ],
        "illegal": sum(bool(game.get("illegal")) for game in results),
        "unfinished": sum(game.get("winner") not in (0, 1) for game in results),
        "candidate_samples": [
            sample for game in results for sample in game.get("candidate_ms", [])
        ],
    }


def pilot_decision(
    *,
    matched_report: pathlib.Path,
    incumbent_report: pathlib.Path,
    rank4_report: pathlib.Path,
    jacek_nn_report: pathlib.Path,
    anchor_candidate: Mapping[str, float],
    anchor_incumbent: Mapping[str, float],
    uncontended_max_ms: float,
) -> dict:
    reports = {
        "matched": validate_gate_report(matched_report, pairs=300, opponent="jacek-replay"),
        "incumbent": validate_gate_report(incumbent_report, pairs=300, opponent="jacek-replay"),
        "rank4": validate_gate_report(rank4_report, pairs=300, opponent="rank4"),
        "jacek-nn": validate_gate_report(jacek_nn_report, pairs=300, opponent="jacek-nn"),
    }
    counts = {name: _gate_counts(report) for name, report in reports.items()}
    errors = []
    for name in ("matched", "incumbent"):
        if counts[name]["wins"] < 325 or min(counts[name]["colors"]) < 156:
            errors.append(f"{name} primary strength gate failed")
    for name in ("rank4", "jacek-nn"):
        if counts[name]["wins"] < 306 or min(counts[name]["colors"]) < 143:
            errors.append(f"{name} external strength gate failed")
    if any(value["illegal"] or value["unfinished"] for value in counts.values()):
        errors.append("game gate has illegal or unfinished games")
    samples = [sample for value in counts.values() for sample in value["candidate_samples"]]
    if not samples or any(
        isinstance(sample, bool)
        or not isinstance(sample, (int, float))
        or not math.isfinite(float(sample))
        or float(sample) < 0.0
        for sample in samples
    ):
        raise ValueError("game gate candidate latency samples are invalid")
    samples.sort()
    p99 = samples[min(len(samples) - 1, math.ceil(0.99 * len(samples)) - 1)]
    if p99 > 25.0 or uncontended_max_ms >= 1_000.0:
        errors.append("latency gate failed")
    candidate_sign = float(anchor_candidate["sign_accuracy"])
    incumbent_sign = float(anchor_incumbent["sign_accuracy"])
    candidate_huber = float(anchor_candidate["weighted_huber"])
    incumbent_huber = float(anchor_incumbent["weighted_huber"])
    if candidate_sign < incumbent_sign - 0.005:
        errors.append("canonical anchor sign noninferiority failed")
    if candidate_huber > incumbent_huber * 1.02:
        errors.append("canonical anchor Huber noninferiority failed")
    return {
        "schema": PILOT_DECISION_SCHEMA,
        "eligible_for_full": not errors,
        "errors": errors,
        "counts": {
            name: {key: value for key, value in record.items() if key != "candidate_samples"}
            for name, record in counts.items()
        },
        "candidate_p99_ms": p99,
        "uncontended_max_ms": uncontended_max_ms,
        "anchor_candidate": dict(anchor_candidate),
        "anchor_incumbent": dict(anchor_incumbent),
        "reports": {name: artifact_snapshot(path) for name, path in {
            "matched": matched_report,
            "incumbent": incumbent_report,
            "rank4": rank4_report,
            "jacek-nn": jacek_nn_report,
        }.items()},
    }


def final_decision(
    *,
    pilot_report: pathlib.Path,
    matched_report: pathlib.Path,
    rank4_report: pathlib.Path,
    jacek_nn_report: pathlib.Path,
    uncontended_max_ms: float,
) -> dict:
    """Apply the frozen 980 ms promotion thresholds to four protected panels."""

    reports = {
        "pilot-teacher": validate_gate_report(
            pilot_report, pairs=500, opponent="jacek-replay", time_ms=980
        ),
        "matched": validate_gate_report(
            matched_report, pairs=500, opponent="jacek-replay", time_ms=980
        ),
        "rank4": validate_gate_report(
            rank4_report, pairs=500, opponent="rank4", time_ms=980
        ),
        "jacek-nn": validate_gate_report(
            jacek_nn_report, pairs=500, opponent="jacek-nn", time_ms=980
        ),
    }
    counts = {name: _gate_counts(report) for name, report in reports.items()}
    errors: list[str] = []
    for name in ("pilot-teacher", "matched"):
        if counts[name]["wins"] < 527 or min(counts[name]["colors"]) < 260:
            errors.append(f"{name} primary strength gate failed")
    for name in ("rank4", "jacek-nn"):
        if counts[name]["wins"] < 501 or min(counts[name]["colors"]) < 238:
            errors.append(f"{name} external strength gate failed")
    if any(value["illegal"] or value["unfinished"] for value in counts.values()):
        errors.append("game gate has illegal or unfinished games")
    if (
        isinstance(uncontended_max_ms, bool)
        or not isinstance(uncontended_max_ms, (int, float))
        or not math.isfinite(float(uncontended_max_ms))
        or float(uncontended_max_ms) >= 1_000.0
    ):
        errors.append("uncontended latency gate failed")
    return {
        "schema": FINAL_DECISION_SCHEMA,
        "eligible_for_local_publication": not errors,
        "canonical_promotion_eligible": False,
        "errors": errors,
        "counts": {
            name: {key: value for key, value in record.items() if key != "candidate_samples"}
            for name, record in counts.items()
        },
        "uncontended_max_ms": uncontended_max_ms,
        "reports": {
            name: artifact_snapshot(path)
            for name, path in {
                "pilot-teacher": pilot_report,
                "matched": matched_report,
                "rank4": rank4_report,
                "jacek-nn": jacek_nn_report,
            }.items()
        },
    }


@dataclasses.dataclass(frozen=True)
class CampaignExecutables:
    continuation_generator: pathlib.Path
    search_teacher: pathlib.Path
    rank4_teacher: pathlib.Path
    comparison: pathlib.Path
    pack_tool: pathlib.Path
    trainer: pathlib.Path

    def resolved(self) -> "CampaignExecutables":
        return CampaignExecutables(
            **{
                field.name: pathlib.Path(getattr(self, field.name)).resolve()
                for field in dataclasses.fields(self)
            }
        )

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            path = pathlib.Path(getattr(self, field.name))
            if not path.is_file():
                raise ValueError(f"campaign producer is missing: {path}")
            if field.name not in {"pack_tool", "trainer"} and not os.access(path, os.X_OK):
                raise ValueError(f"campaign producer is not executable: {path}")

    def snapshots(self) -> dict[str, dict]:
        return {
            field.name: artifact_snapshot(pathlib.Path(getattr(self, field.name)))
            for field in dataclasses.fields(self)
        }


class GuardedStageManager(StageManager):
    """Recheck the frozen producer set before and after every stage."""

    def __init__(self, *, producer_guard: Callable[[], None], **arguments: object) -> None:
        super().__init__(**arguments)
        self._producer_guard = producer_guard
        self._stage_artifacts: dict[str, dict] = {}

    def _validate_stage_artifacts(self) -> None:
        for record in self._stage_artifacts.values():
            if artifact_snapshot(pathlib.Path(record["path"])) != record:
                raise ValueError("completed campaign stage ancestry changed")

    def _capture_stage_artifacts(self, value: object) -> None:
        if isinstance(value, dict):
            raw_path = value.get("path")
            if isinstance(raw_path, str) and pathlib.Path(raw_path).is_file():
                snapshot = artifact_snapshot(pathlib.Path(raw_path))
                self._stage_artifacts[snapshot["path"]] = snapshot
            for child in value.values():
                self._capture_stage_artifacts(child)
        elif isinstance(value, list):
            for child in value:
                self._capture_stage_artifacts(child)

    def execute(self, **arguments: object) -> dict:
        self._producer_guard()
        self._validate_stage_artifacts()
        if self.resume and "resumable_outputs" not in arguments:
            outputs = arguments.get("outputs")
            if isinstance(outputs, dict):
                # Outer-stage products are deterministic aggregates.  Inner
                # chunks/seeds remain protected by their own receipts; after a
                # crash we may revalidate and republish only these aggregates.
                arguments["resumable_outputs"] = set(outputs)
        action = arguments.get("action")
        if not callable(action):
            raise TypeError("guarded stage requires a callable action")

        def guarded_action() -> dict | None:
            self._producer_guard()
            self._validate_stage_artifacts()
            result = action()
            self._producer_guard()
            self._validate_stage_artifacts()
            return result

        arguments["action"] = guarded_action
        result = super().execute(**arguments)
        self._producer_guard()
        receipt = self.receipt_path(
            int(arguments["ordinal"]), str(arguments["name"])
        )
        self._capture_stage_artifacts(_load_json(receipt, "completed stage receipt"))
        self._stage_artifacts[str(receipt.resolve())] = artifact_snapshot(receipt)
        self._validate_stage_artifacts()
        return result


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    name: str
    campaign_id: str
    configuration: Mapping[str, object]
    quotas: Mapping[str, int]
    game_seed: int
    opening_seed: int
    pairs: int
    gate_time_ms: int
    gate_workers: int
    bank_classification: str


PILOT_SPEC = PhaseSpec(
    name="pilot",
    campaign_id=PILOT_CAMPAIGN_ID,
    configuration=PILOT_CONFIGURATION,
    quotas=PILOT_QUOTAS,
    game_seed=PILOT_GAME_SEED,
    opening_seed=PILOT_OPENING_SEED,
    pairs=300,
    gate_time_ms=20,
    gate_workers=4,
    bank_classification="development",
)
FULL_SPEC = PhaseSpec(
    name="full",
    campaign_id=FULL_CAMPAIGN_ID,
    configuration=FULL_CONFIGURATION,
    quotas=FULL_QUOTAS,
    game_seed=FULL_GAME_SEED,
    opening_seed=FULL_OPENING_SEED,
    pairs=500,
    gate_time_ms=980,
    gate_workers=4,
    bank_classification="final",
)


def _run(command: Sequence[str], *, cwd: pathlib.Path | None = None) -> str:
    pass_fds = (() if _CAMPAIGN_LOCK_FD is None else (_CAMPAIGN_LOCK_FD,))
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        pass_fds=pass_fds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed.stdout.decode("utf-8", "strict")


def _run_stdout_to_path(command: Sequence[str], output: pathlib.Path) -> None:
    payload = _run(command).encode("utf-8")
    if not payload:
        raise RuntimeError(f"producer emitted an empty artifact: {' '.join(command)}")
    _atomic_bytes(output, payload)


def _replace_path_prefix(value: object, old: pathlib.Path, new: pathlib.Path) -> object:
    if isinstance(value, str):
        old_text = str(old)
        return str(new) + value[len(old_text):] if value.startswith(old_text) else value
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, old, new)
            for key, item in value.items()
        }
    return value


def _status(path: pathlib.Path, phase: str, **fields: object) -> None:
    _atomic_json(
        path,
        {
            "schema": CAMPAIGN_STATUS_SCHEMA,
            "phase": phase,
            "updated_at_unix": time.time(),
            **fields,
        },
    )


def _verify_frozen_hashes(records: Mapping[str, object]) -> None:
    if not records:
        raise ValueError("frozen input registry is empty")
    for raw_path, expected in records.items():
        path = pathlib.Path(raw_path)
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"frozen evaluation input changed: {path}")


def validate_evaluation_trigger(evaluation_directory: pathlib.Path) -> dict:
    """Validate the completed 5,000-game run and its sequential audit."""

    evaluation_directory = evaluation_directory.resolve()
    manifest_path = evaluation_directory / "run-manifest.json"
    status_path = evaluation_directory / "supervisor-status.json"
    summary_path = evaluation_directory / "final-summary.json"
    manifest = _load_json(manifest_path, "evaluation run manifest")
    status = _load_json(status_path, "evaluation supervisor status")
    if manifest.get("schema") != "papersoccer.jacek-replay-postcampaign-run.v1":
        raise ValueError("evaluation run manifest schema is invalid")
    if manifest.get("configuration") != {
        "workers": 4,
        "pair_count_per_panel": 500,
        "pairs_per_shard": 5,
        "time_ms": 980,
        "exploration": 0.5,
        "fpu": 0.5,
        "step1_games": 2_000,
        "step2_games": 3_000,
    }:
        raise ValueError("evaluation run configuration is not the frozen run")
    if status.get("phase") != "complete":
        if status.get("phase") == "failed":
            raise ValueError(f"prerequisite evaluation failed: {status.get('error')}")
        raise ValueError("prerequisite evaluation is not complete")
    if not summary_path.is_file() or status.get("summary_sha256") != sha256(summary_path):
        raise ValueError("evaluation final summary is missing or stale")
    summary = _load_json(summary_path, "evaluation final summary")
    if (
        summary.get("schema")
        != "papersoccer.jacek-replay-postcampaign-summary.v1"
        or summary.get("games") != 5_000
        or summary.get("producer_commit") != manifest.get("producer_commit")
        or set(summary.get("step2", {}))
        != {"r0-vs-r1", "r1-vs-r2", "r0-vs-r2"}
        or not isinstance(summary.get("sequential_latency_audit"), dict)
    ):
        raise ValueError("evaluation final summary is incomplete")
    _verify_frozen_hashes(manifest.get("inputs", {}))
    expected_jobs = {
        *(f"step1-r2-controls-{offset:03d}" for offset in range(0, 500, 5)),
        *(
            f"step2-{matchup}-{offset:03d}"
            for matchup in ("r0-r1", "r1-r2", "r0-r2")
            for offset in range(0, 500, 5)
        ),
    }
    report_bindings = summary.get("reports")
    if (
        not isinstance(report_bindings, list)
        or len(report_bindings) != 400
        or {binding.get("job_id") for binding in report_bindings
            if isinstance(binding, dict)} != expected_jobs
    ):
        raise ValueError("evaluation final summary does not bind all 400 jobs")
    total_report_games = 0
    for report in report_bindings:
        if not isinstance(report, dict):
            raise ValueError("evaluation report binding is malformed")
        job_id = report.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("evaluation report binding has no job ID")
        phase = "step1" if job_id.startswith("step1-") else "step2"
        report_path = evaluation_directory / "shards" / phase / f"{job_id}.json"
        receipt_path = evaluation_directory / "receipts" / f"{job_id}.json"
        if (
            not report_path.is_file()
            or sha256(report_path) != report.get("report_sha256")
            or not receipt_path.is_file()
            or sha256(receipt_path) != report.get("receipt_sha256")
        ):
            raise ValueError("evaluation report/receipt binding is stale")
        receipt = _load_json(receipt_path, "evaluation shard receipt")
        payload = _load_json(report_path, "evaluation shard report")
        job = receipt.get("job", {})
        expected_games = 20 if phase == "step1" else 10
        offset = int(job_id.rsplit("-", 1)[1])
        configuration = payload.get("configuration", {})
        if (
            receipt.get("schema") != "papersoccer.jacek-replay-shard-receipt.v1"
            or job.get("job_id") != job_id
            or receipt.get("report_sha256") != sha256(report_path)
            or job.get("phase") != phase
            or job.get("pairs") != 5
            or job.get("offset") != offset
            or job.get("expected_games") != expected_games
            or job.get("time_ms") != 980
            or payload.get("schema")
            != "papersoccer.jacek-replay-bfm-comparison.v1"
            or payload.get("model_sha256") != job.get("model_sha256")
            or configuration.get("pairs") != 5
            or configuration.get("pair_offset") != offset
            or configuration.get("time_ms") != 980
            or configuration.get("exploration") != 0.5
            or configuration.get("fpu") != 0.5
            or configuration.get("opening_plies") != 12
            or configuration.get("max_turns") != 320
            or configuration.get("single_thread") is not True
            or configuration.get("opponent") != job.get("opponent")
            or configuration.get("opening_bank_sha256") != job.get("bank_sha256")
            or configuration.get("comparison_executable_sha256")
            != manifest.get("producers", {}).get("comparison_executable_sha256")
            or payload.get("summary", {}).get("games") != expected_games
            or not isinstance(payload.get("results"), list)
            or len(payload["results"]) != expected_games
            or receipt.get("wins") != payload.get("summary", {}).get("wins")
            or receipt.get("losses") != payload.get("summary", {}).get("losses")
            or receipt.get("unfinished") != payload.get("summary", {}).get("unfinished")
            or receipt.get("illegal") != payload.get("summary", {}).get("illegal")
        ):
            raise ValueError("evaluation shard receipt is malformed")
        total_report_games += expected_games
    if total_report_games != 5_000:
        raise ValueError("evaluation shard reports do not total 5,000 games")
    if (
        set(summary.get("step1", {})) != {"rank4", "jacek-nn"}
        or any(
            panel.get("games") != 1_000
            or panel.get("illegal") != 0
            or panel.get("unfinished") != 0
            for panel in summary["step1"].values()
        )
        or any(
            matchup.get("lower_round_as_candidate", {}).get("games") != 1_000
            or matchup.get("lower_round_as_candidate", {}).get("illegal") != 0
            or matchup.get("lower_round_as_candidate", {}).get("unfinished") != 0
            for matchup in summary["step2"].values()
        )
    ):
        raise ValueError("evaluation panel summaries are incomplete or invalid")
    latency_path = evaluation_directory / "latency-audit.json"
    if not latency_path.is_file():
        raise ValueError("evaluation sequential latency report is missing")
    latency = _load_json(latency_path, "evaluation sequential latency report")
    latency_configuration = latency.get("configuration", {})
    if (
        latency.get("schema") != "papersoccer.jacek-replay-bfm-comparison.v1"
        or latency.get("summary") != summary["sequential_latency_audit"]
        or latency.get("summary", {}).get("games") != 20
        or latency.get("summary", {}).get("illegal") != 0
        or latency.get("summary", {}).get("unfinished") != 0
        or not isinstance(latency.get("results"), list)
        or len(latency["results"]) != 20
        or latency_configuration.get("pairs") != 5
        or latency_configuration.get("pair_offset") != 0
        or latency_configuration.get("time_ms") != 980
        or latency_configuration.get("exploration") != 0.5
        or latency_configuration.get("fpu") != 0.5
        or latency_configuration.get("opening_plies") != 12
        or latency_configuration.get("max_turns") != 320
        or latency_configuration.get("single_thread") is not True
        or latency_configuration.get("opponent") != "rank4-jacek-nn"
        or latency_configuration.get("comparison_executable_sha256")
        != manifest.get("producers", {}).get("comparison_executable_sha256")
    ):
        raise ValueError("evaluation sequential latency summary is stale")
    return {
        "schema": "papersoccer.jacek-selfsearch-evaluation-trigger.v1",
        "evaluation_manifest": artifact_snapshot(manifest_path),
        "evaluation_status": artifact_snapshot(status_path),
        "evaluation_summary": artifact_snapshot(summary_path),
        "latency_audit": artifact_snapshot(latency_path),
        "producer_commit": manifest["producer_commit"],
        "sequential_latency_audit": summary["sequential_latency_audit"],
    }


def _evaluation_processes(evaluation_directory: pathlib.Path) -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("could not inspect prerequisite evaluation processes")
    needle = str(evaluation_directory.resolve())
    evaluation_binary = f"{needle}/bin/comparison"
    processes = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        fields = stripped.split(maxsplit=1)
        if not fields or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        command = fields[1] if len(fields) == 2 else ""
        if needle in command and (
            "evaluation_supervisor.py" in command or evaluation_binary in command
        ):
            processes.append(stripped)
    return processes


def validate_host_health(output: pathlib.Path, *, skip_power: bool = False) -> dict:
    free = shutil.disk_usage(output.parent.resolve()).free
    if free < MINIMUM_FREE_BYTES:
        raise ValueError("self-search campaign requires at least 12 GiB free disk")
    power = "unchecked"
    if not skip_power and shutil.which("pmset"):
        completed = subprocess.run(
            ["pmset", "-g", "batt"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode != 0 or "AC Power" not in completed.stdout:
            raise ValueError("self-search campaign requires AC power")
        power = "AC Power"
    return {"free_bytes": free, "power": power}


def _plan_chunks(plan: Mapping[str, object], directory: pathlib.Path, size: int) -> list[pathlib.Path]:
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows or size <= 0:
        raise ValueError("cannot split malformed game plan")
    paths = []
    for ordinal, begin in enumerate(range(0, len(rows), size)):
        chunk = {**plan, "games": len(rows[begin : begin + size]), "rows": rows[begin : begin + size]}
        quotas: collections.Counter[str] = collections.Counter(
            str(row["actor_mode"]) for row in chunk["rows"]
        )
        chunk["quotas"] = dict(quotas)
        path = directory / f"chunk-{ordinal:06d}.tsv"
        payload = (
            "game_ordinal\tactor_mode\tbase_seed\n"
            + "".join(
                f"{row['game_ordinal']}\t{row['actor_mode']}\t{row['base_seed']}\n"
                for row in chunk["rows"]
            )
        ).encode("utf-8")
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"game-plan chunk {ordinal} is stale")
        if not path.exists():
            _atomic_bytes(path, payload)
        paths.append(path)
    return paths


def _validate_game_chunk(
    *,
    campaign_id: str,
    plan_chunk: pathlib.Path,
    games: pathlib.Path,
    manifest_path: pathlib.Path,
    roots_tsv: pathlib.Path,
    actor: pathlib.Path,
    diversity: pathlib.Path,
    configuration: Mapping[str, object],
) -> dict:
    manifest = _load_json(manifest_path, "self-search game chunk")
    plan_lines = plan_chunk.read_text(encoding="utf-8").splitlines()
    plan_rows = [line.split("\t") for line in plan_lines[1:]]
    expected_ordinals = [int(row[0]) for row in plan_rows]
    expected_quotas = collections.Counter(row[1] for row in plan_rows)
    bindings = manifest.get("bindings", {})
    actual_configuration = manifest.get("configuration")
    rows = manifest.get("rows")
    if (
        manifest.get("schema") != GAME_MANIFEST_SCHEMA
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("requested_games") != len(plan_rows)
        or manifest.get("successful_games") != len(plan_rows)
        or not isinstance(actual_configuration, dict)
        or any(actual_configuration.get(key) != value
               for key, value in configuration.items())
        or bindings.get("roots_sha256") != sha256(roots_tsv)
        or bindings.get("plan_sha256") != sha256(plan_chunk)
        or bindings.get("output_sha256") != sha256(games)
        or bindings.get("incumbent_model_sha256") != sha256(actor)
        or bindings.get("runner_up_model_sha256") != sha256(diversity)
        or not isinstance(rows, list)
        or [row.get("game_ordinal") for row in rows if isinstance(row, dict)]
        != expected_ordinals
        or collections.Counter(
            row.get("actor_mode") for row in rows if isinstance(row, dict)
        ) != expected_quotas
    ):
        raise ValueError("self-search game chunk is stale or malformed")
    for field in (
        "producer_source_sha256", "rank4_actor_source_sha256",
        "jacek_nn_actor_source_sha256",
    ):
        value = actual_configuration.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or bindings.get(field) != value
        ):
            raise ValueError("self-search game chunk source identity is stale")
    lines = games.read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["group_id\tsource\twinner\ttranscript"] or len(lines) != len(rows) + 1:
        raise ValueError("self-search game chunk TSV is malformed")
    for ordinal, (line, row) in enumerate(zip(lines[1:], rows, strict=True)):
        fields = line.split("\t")
        planned = plan_rows[ordinal]
        if (
            len(fields) != 4
            or row.get("row_ordinal") != ordinal
            or row.get("root_group_id") != fields[0]
            or row.get("winner") != int(fields[2])
            or row.get("transcript_sha256")
            != hashlib.sha256(fields[3].encode()).hexdigest()
            or row.get("actor_mode") != planned[1]
            or row.get("base_seed") != int(planned[2])
        ):
            raise ValueError("self-search game chunk row is stale")
    return manifest


def run_game_chunks(
    *,
    manager: StageManager,
    stage_ordinal: int,
    spec: PhaseSpec,
    plan_path: pathlib.Path,
    roots_tsv: pathlib.Path,
    actor: pathlib.Path,
    diversity: pathlib.Path,
    generator: pathlib.Path,
    workers: int,
    source_identities: Mapping[str, str],
) -> dict:
    plan = _load_json(plan_path, "self-search game plan")
    chunk_root = manager.output / "game-chunks"
    plan_root = chunk_root / "plans"
    output_root = chunk_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_root = manager.receipts / f"{stage_ordinal:02d}-games-chunks"
    chunk_plans = _plan_chunks(
        plan, plan_root, int(spec.configuration["game_chunk_size"])
    )
    configuration = {
        "bfm_tree_nodes": int(spec.configuration["bfm_actor_tree_nodes"]),
        "rank4_nodes": int(spec.configuration["rank4_actor_nodes"]),
        "jacek_nn_nodes": int(spec.configuration["jacek_nn_actor_nodes"]),
        "exploration": float(spec.configuration["exploration"]),
        "fpu": float(spec.configuration["fpu"]),
        "early_exploration_percent": int(spec.configuration["early_exploration_percent"]),
        "early_exploration_turns": int(spec.configuration["early_exploration_turns"]),
        "maximum_turns": 320,
        "producer_source_sha256": source_identities["continuation_source_sha256"],
        "rank4_actor_source_sha256": source_identities["rank4_actor_source_sha256"],
        "jacek_nn_actor_source_sha256": source_identities["jacek_nn_actor_source_sha256"],
    }
    producer = artifact_snapshot(generator)
    specs = []
    for ordinal, chunk_plan in enumerate(chunk_plans):
        games = output_root / f"chunk-{ordinal:06d}.tsv"
        manifest = output_root / f"chunk-{ordinal:06d}.manifest.json"
        receipt = receipt_root / f"chunk-{ordinal:06d}.json"
        expected = {
            "schema": CAMPAIGN_RECEIPT_SCHEMA,
            "campaign_id": spec.campaign_id,
            "stage": "games",
            "chunk_ordinal": ordinal,
            "configuration": configuration,
            "producer": producer,
            "inputs": {
                "plan": artifact_snapshot(chunk_plan),
                "roots": artifact_snapshot(roots_tsv),
                "actor": artifact_snapshot(actor),
                "diversity": artifact_snapshot(diversity),
            },
        }
        specs.append((ordinal, chunk_plan, games, manifest, receipt, expected))

    def validate(item: tuple) -> dict:
        _, chunk_plan, games, manifest, receipt, expected = item
        saved = _load_json(receipt, "game chunk receipt")
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError("game chunk receipt is stale")
        if saved.get("outputs") != {
            "games": artifact_snapshot(games),
            "manifest": artifact_snapshot(manifest),
        }:
            raise ValueError("game chunk receipt output is stale")
        _validate_game_chunk(
            campaign_id=spec.campaign_id,
            plan_chunk=chunk_plan,
            games=games,
            manifest_path=manifest,
            roots_tsv=roots_tsv,
            actor=actor,
            diversity=diversity,
            configuration=configuration,
        )
        return saved

    def execute(item: tuple) -> dict:
        ordinal, chunk_plan, games, manifest, receipt, expected = item
        if receipt.exists():
            if not manager.resume:
                raise ValueError("game chunk already completed; use --resume")
            return validate(item)
        if games.exists() or manifest.exists():
            raise ValueError("game chunk has unreceipted output")
        attempt = pathlib.Path(tempfile.mkdtemp(dir=chunk_root, prefix=f".chunk-{ordinal:06d}."))
        try:
            staged_games = attempt / "games.tsv"
            staged_manifest = attempt / "manifest.json"
            command = [
                str(generator), "--input", str(roots_tsv), "--output", str(staged_games),
                "--manifest", str(staged_manifest), "--model", str(actor),
                "--runner-up-model", str(diversity), "--selfsearch-plan", str(chunk_plan),
                "--campaign-id", spec.campaign_id, "--games", str(len(chunk_plan.read_text().splitlines()) - 1),
                "--actor-nodes", str(spec.configuration["rank4_actor_nodes"]),
                "--candidate-tree-nodes", str(spec.configuration["bfm_actor_tree_nodes"]),
                "--jacek-nn-nodes", str(spec.configuration["jacek_nn_actor_nodes"]),
                "--candidate-exploration", str(spec.configuration["exploration"]),
                "--candidate-fpu", str(spec.configuration["fpu"]),
            ]
            _run(command)
            _validate_game_chunk(
                campaign_id=spec.campaign_id, plan_chunk=chunk_plan,
                games=staged_games, manifest_path=staged_manifest,
                roots_tsv=roots_tsv, actor=actor, diversity=diversity,
                configuration=configuration,
            )
            os.replace(staged_games, games)
            os.replace(staged_manifest, manifest)
            saved = {**expected, "outputs": {
                "games": artifact_snapshot(games), "manifest": artifact_snapshot(manifest)}}
            _atomic_json(receipt, saved)
            return saved
        finally:
            shutil.rmtree(attempt, ignore_errors=True)

    missing = [item for item in specs if not item[4].exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(execute, item) for item in missing]
            ):
                future.result()
    chunk_tsvs, chunk_manifests, receipts, artifacts = [], [], [], []
    for item in specs:
        validate(item)
        chunk_tsvs.append(item[2])
        chunk_manifests.append(item[3])
        receipts.append(artifact_snapshot(item[4]))
        artifacts.extend(
            artifact_snapshot(path) for path in (item[1], item[2], item[3])
        )
    games_path = manager.output / "games.tsv"
    manifest_path = manager.output / "games.manifest.json"
    payload, manifest = merge_game_chunks(
        campaign_id=spec.campaign_id, plan_path=plan_path,
        chunk_tsvs=chunk_tsvs, chunk_manifests=chunk_manifests,
    )
    write_pair(payload, manifest, games_path, manifest_path)
    return {
        "games": int(manifest["successful_games"]),
        "chunks": receipts,
        "chunk_artifacts": artifacts,
    }


def _position_rows(path: pathlib.Path) -> tuple[str, list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != (
        "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
    ):
        raise ValueError("position TSV has the wrong header")
    ids = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 8 or not fields[0] or fields[0] in ids:
            raise ValueError("position TSV contains a malformed or duplicate row")
        ids.append(fields[0])
    if not ids:
        raise ValueError("position TSV is empty")
    return lines[0], lines[1:], ids


def _validate_label_output(
    *, output: pathlib.Path, positions: pathlib.Path, schema: str,
    campaign_id: str, nodes: int, model_sha256: str | None,
    source_sha256: str,
) -> int:
    _, position_rows, ids = _position_rows(positions)
    labels = load_labels(output, schema)
    if list(labels) != ids:
        raise ValueError("teacher labels are not in exact position order")
    position_by_id = {row.split("\t")[0]: row.split("\t") for row in position_rows}
    for position_id, row in labels.items():
        corpus.sample_from_teacher_row(row)
        fields = position_by_id[position_id]
        actions = fields[7].split("/") if fields[7] else []
        expected_prefix = [
            {"player_id": turn % 2, "action": action}
            for turn, action in enumerate(actions)
        ]
        if (
            row.get("campaign_id") != campaign_id
            or row.get("root_group_id") != fields[1]
            or row.get("group_id") != fields[2]
            or row.get("source") != fields[3]
            or row.get("split") != fields[4]
            or row.get("winner") != int(fields[5])
            or row.get("mover") != int(fields[6])
            or row.get("prefix") != expected_prefix
        ):
            raise ValueError("teacher label lineage is stale")
        if schema == SEARCH_TEACHER_SCHEMA:
            configuration = row.get("search_config", {})
            stats = row.get("search_stats", {})
            expected_seed = int.from_bytes(
                hashlib.sha256(
                    f"{campaign_id}\0{position_id}\0{nodes}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            if (
                row.get("teacher", {}).get("model_sha256") != model_sha256
                or row.get("teacher", {}).get("source_sha256") != source_sha256
                or row.get("teacher", {}).get("feature_schema")
                != features.FEATURE_SCHEMA
                or row.get("teacher", {}).get("feature_schema_sha256")
                != hashlib.sha256(features.FEATURE_SCHEMA.encode("utf-8")).hexdigest()
                or configuration != {
                    "seed": expected_seed,
                    "max_time_ms": SEARCH_SAFETY_MS,
                    "max_tree_nodes": nodes,
                    "max_actions": SEARCH_MAX_ACTIONS,
                    "max_partial_paths": SEARCH_MAX_PARTIAL_PATHS,
                    "exploration": 0.5,
                    "fpu": 0.5,
                }
                or stats.get("deadline_reached") is not False
                or stats.get("generation_deadline_stops") != 0
                or stats.get("materialization_deadline_stops") != 0
                or stats.get("closed_unsolved_nodes") != 0
                or stats.get("closed_unsolved_nonexhaustive_nodes") != 0
                or not isinstance(stats.get("max_open_children"), int)
                or stats.get("max_open_children", SEARCH_MAX_ACTIONS + 1)
                > SEARCH_MAX_ACTIONS
                or not isinstance(stats.get("max_complete_turn_depth"), int)
                or stats.get("max_complete_turn_depth", 0) <= 0
                or not isinstance(stats.get("completed_actions"), int)
                or stats.get("completed_actions", 0) <= 0
                or row.get("root_solved")
                != (row.get("proven_winner") is not None)
                or stats.get("termination_reason")
                != (
                    "root-solved"
                    if row.get("root_solved") is True
                    else "fixed-work-cap"
                )
                or (
                    row.get("root_solved") is False
                    and (
                        stats.get("tree_cap_reached") is not True
                        or stats.get("tree_nodes") != nodes
                        or stats.get("visits", 0) <= 0
                    )
                )
            ):
                raise ValueError("search-teacher fixed-work binding is stale")
        else:
            observed_nodes = row.get("nodes", row.get("search_stats", {}).get("nodes"))
            stats = row.get("search_stats", {})
            configuration = row.get("search_config", {})
            if (
                row.get("teacher") != {
                    "kind": "rank4-fixed-work", "source_sha256": source_sha256
                }
                or configuration != {
                    "max_nodes": nodes,
                    "max_time_ms": SEARCH_SAFETY_MS,
                    "max_turn_depth": 32,
                    "replay_value_blend_percent": 15,
                    "teacher_residual_weight_percent": 100,
                }
                or set(stats) != {
                    "attempted_depth", "completed_depth", "nodes",
                    "leaf_evaluations", "terminal_nodes", "completed_actions",
                    "budget_exhausted", "node_cap_reached", "depth_cap_reached",
                    "deadline_reached", "termination_reason",
                }
                or not isinstance(observed_nodes, int)
                or not 0 < observed_nodes <= nodes
                or stats.get("nodes", observed_nodes) != observed_nodes
                or stats.get("deadline_reached") is not False
                or stats.get("budget_exhausted") is not stats.get("node_cap_reached")
                or row.get("completed_depth") != stats.get("completed_depth")
                or not isinstance(stats.get("completed_depth"), int)
                or stats.get("completed_depth", -1) < 0
                or not isinstance(stats.get("attempted_depth"), int)
                or stats.get("attempted_depth", 0)
                < max(1, stats.get("completed_depth", 0))
                or not isinstance(stats.get("completed_actions"), int)
                or stats.get("completed_actions", 0) <= 0
                or row.get("root_solved") != (row.get("proven_winner") is not None)
                or stats.get("termination_reason")
                != (
                    "root-solved"
                    if row.get("root_solved") is True
                    else "fixed-work-cap"
                )
                or stats.get("depth_cap_reached")
                is not (
                    stats.get("completed_depth")
                    == row.get("search_config", {}).get("max_turn_depth")
                )
                or (
                    stats.get("node_cap_reached") is True
                    and stats.get("nodes") != nodes
                )
                or (
                    stats.get("completed_depth") == 0
                    and stats.get("node_cap_reached") is not True
                )
                or not (
                    row.get("root_solved") is True
                    or stats.get("node_cap_reached") is True
                    or stats.get("depth_cap_reached") is True
                )
            ):
                raise ValueError("Rank-4 teacher work binding is stale")
    return len(labels)


def run_label_chunks(
    *, manager: StageManager, stage_ordinal: int, stage_name: str,
    positions: pathlib.Path, output: pathlib.Path, teacher: pathlib.Path,
    schema: str, campaign_id: str, nodes: int, workers: int,
    source_sha256: str,
    model: pathlib.Path | None = None, chunk_games: int = POSITION_CHUNK_GAMES,
) -> dict:
    header, rows, position_ids = _position_rows(positions)
    chunk_root = manager.output / "label-chunks" / stage_name
    receipt_root = manager.receipts / f"{stage_ordinal:02d}-{stage_name}-chunks"
    producer = artifact_snapshot(teacher)
    model_hash = sha256(model) if model is not None else None
    if chunk_games <= 0:
        raise ValueError("position teacher chunk game count must be positive")
    grouped_by_game: dict[str, list[str]] = {}
    for row in rows:
        group_id = row.split("\t")[2]
        grouped_by_game.setdefault(group_id, []).append(row)
    grouped = list(grouped_by_game.values())
    row_chunks = [
        [row for game_rows in grouped[begin : begin + chunk_games] for row in game_rows]
        for begin in range(0, len(grouped), chunk_games)
    ]
    specs = []
    row_begin = 0
    for ordinal, chunk_data in enumerate(row_chunks):
        chunk_input = chunk_root / f"chunk-{ordinal:06d}.tsv"
        chunk_output = chunk_root / f"chunk-{ordinal:06d}.jsonl"
        receipt = receipt_root / f"chunk-{ordinal:06d}.json"
        payload = (header + "\n" + "\n".join(chunk_data) + "\n").encode()
        if chunk_input.exists() and chunk_input.read_bytes() != payload:
            raise ValueError("teacher chunk input is stale")
        if not chunk_input.exists():
            _atomic_bytes(chunk_input, payload)
        expected = {
            "schema": CAMPAIGN_RECEIPT_SCHEMA, "campaign_id": campaign_id,
            "stage": stage_name, "chunk_ordinal": ordinal, "row_begin": row_begin,
            "rows": len(chunk_data), "games": min(chunk_games, len(grouped) - ordinal * chunk_games),
            "nodes": nodes,
            "producer": producer, "model": artifact_snapshot(model) if model else None,
            "source_sha256": source_sha256,
            "input": artifact_snapshot(chunk_input),
        }
        specs.append((ordinal, chunk_input, chunk_output, receipt, expected))
        row_begin += len(chunk_data)

    def validate(item: tuple) -> dict:
        _, chunk_input, chunk_output, receipt, expected = item
        saved = _load_json(receipt, "label chunk receipt")
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError("label chunk receipt is stale")
        if saved.get("output") != artifact_snapshot(chunk_output):
            raise ValueError("label chunk output is stale")
        count = _validate_label_output(
            output=chunk_output, positions=chunk_input, schema=schema,
            campaign_id=campaign_id, nodes=nodes, model_sha256=model_hash,
            source_sha256=source_sha256,
        )
        if saved.get("teacher_rows") != count:
            raise ValueError("label chunk count is stale")
        return saved

    def execute(item: tuple) -> dict:
        _, chunk_input, chunk_output, receipt, expected = item
        if receipt.exists():
            if not manager.resume:
                raise ValueError("label chunk already completed; use --resume")
            return validate(item)
        if chunk_output.exists():
            raise ValueError("label chunk has unreceipted output")
        if schema == SEARCH_TEACHER_SCHEMA:
            command = [
                str(teacher), "--model", str(model), "--model-sha256", str(model_hash),
                "--campaign-id", campaign_id, "--tree-nodes", str(nodes),
                "--time-ms", str(SEARCH_SAFETY_MS), "--max-actions", str(SEARCH_MAX_ACTIONS),
                "--max-partial-paths", str(SEARCH_MAX_PARTIAL_PATHS),
                "--exploration", "0.5", "--fpu", "0.5",
            ]
        else:
            command = [
                str(teacher), "--campaign-id", campaign_id, "--nodes", str(nodes),
                "--time-ms", str(SEARCH_SAFETY_MS),
            ]
        with chunk_input.open("rb") as source, tempfile.NamedTemporaryFile(
            dir=chunk_output.parent, prefix=f".{chunk_output.name}.", delete=False
        ) as destination:
            temporary = pathlib.Path(destination.name)
            pass_fds = (
                () if _CAMPAIGN_LOCK_FD is None else (_CAMPAIGN_LOCK_FD,)
            )
            completed = subprocess.run(command, stdin=source, stdout=destination,
                                       stderr=subprocess.PIPE, check=False,
                                       pass_fds=pass_fds)
        try:
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
            os.replace(temporary, chunk_output)
        finally:
            temporary.unlink(missing_ok=True)
        count = _validate_label_output(
            output=chunk_output, positions=chunk_input, schema=schema,
            campaign_id=campaign_id, nodes=nodes, model_sha256=model_hash,
            source_sha256=source_sha256,
        )
        saved = {**expected, "output": artifact_snapshot(chunk_output), "teacher_rows": count}
        _atomic_json(receipt, saved)
        return saved

    missing = [item for item in specs if not item[3].exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(execute, item) for item in missing]
            ):
                future.result()
    bindings = []
    artifacts = []
    merged_labels: dict[str, dict] = {}
    for item in specs:
        validate(item)
        chunk_labels = load_labels(item[2], schema)
        overlap = set(merged_labels) & set(chunk_labels)
        if overlap:
            raise ValueError("label chunks repeat position IDs")
        merged_labels.update(chunk_labels)
        bindings.append(artifact_snapshot(item[3]))
        artifacts.extend(artifact_snapshot(path) for path in (item[1], item[2]))
    if set(merged_labels) != set(position_ids):
        raise ValueError("label chunks do not exactly cover frozen positions")
    payload = b"".join(canonical_json_bytes(merged_labels[position_id])
                       for position_id in position_ids)
    _atomic_bytes(output, payload)
    count = _validate_label_output(
        output=output, positions=positions, schema=schema, campaign_id=campaign_id,
        nodes=nodes, model_sha256=model_hash, source_sha256=source_sha256,
    )
    return {
        "teacher_rows": count,
        "chunks": bindings,
        "chunk_artifacts": artifacts,
    }


def _validate_pack_report(
    report_path: pathlib.Path, *, roots: pathlib.Path, labels: pathlib.Path,
    prior_manifests: Sequence[pathlib.Path] = (),
) -> dict:
    report = _load_json(report_path, "self-search pack report")
    expected_prior = []
    for path in prior_manifests:
        manifest = _load_json(path, "prior self-search shard")
        expected_prior.append({
            "manifest_sha256": sha256(path),
            "npz_sha256": manifest.get("npz_sha256"),
            "split": manifest.get("split"),
        })
    if (
        report.get("schema") != "papersoccer.jacek-replay-pack-report.v1"
        or report.get("roots_manifest_sha256") != sha256(roots)
        or report.get("teacher_jsonl_sha256")
        != [{"name": labels.name, "sha256": sha256(labels)}]
        or not isinstance(report.get("target_policies"), list)
        or not report["target_policies"]
        or report.get("prior_shards", []) != expected_prior
    ):
        raise ValueError("self-search pack report binding is stale")
    for split in ("train", "validation", "test"):
        record = report.get("shards", {}).get(split)
        if not isinstance(record, dict):
            raise ValueError("self-search pack report omits a split")
        manifest_path = pathlib.Path(str(record.get("manifest", "")))
        npz_path = pathlib.Path(str(record.get("npz", "")))
        if (
            not manifest_path.is_file()
            or sha256(manifest_path) != record.get("manifest_sha256")
            or not npz_path.is_file()
            or sha256(npz_path) != record.get("sha256")
        ):
            raise ValueError("self-search shard binding is stale")
        manifest = _load_json(manifest_path, "self-search shard manifest")
        if (
            manifest.get("split") != split
            or manifest.get("samples") != record.get("samples")
            or manifest.get("npz_sha256") != record.get("sha256")
            or manifest.get("provenance", {}).get("target_policies")
            != report["target_policies"]
        ):
            raise ValueError("self-search shard manifest is stale")
    return report


def run_pack(
    *, python: pathlib.Path, pack_tool: pathlib.Path, roots: pathlib.Path,
    labels: pathlib.Path, output_directory: pathlib.Path,
    prior_manifests: Sequence[pathlib.Path] = (),
) -> dict:
    if output_directory.exists():
        report = output_directory / "pack-report.json"
        if not report.is_file():
            raise ValueError(f"pack output is partial: {output_directory}")
        return _validate_pack_report(
            report, roots=roots, labels=labels,
            prior_manifests=prior_manifests,
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(
        dir=output_directory.parent, prefix=f".{output_directory.name}.inprogress."
    ))
    published = False
    try:
        command = [
            str(python), str(pack_tool), "pack", "--roots", str(roots),
            "--teacher", str(labels), "--output-directory", str(scratch), "--streaming",
        ]
        for path in prior_manifests:
            command.extend(("--prior-shard-manifest", str(path)))
        _run(command)
        staged_report = scratch / "pack-report.json"
        report = _load_json(staged_report, "staged self-search pack report")
        report = _replace_path_prefix(report, scratch, output_directory)
        _atomic_json(staged_report, report)
        os.replace(scratch, output_directory)
        published = True
    finally:
        if not published:
            shutil.rmtree(scratch, ignore_errors=True)
    return _validate_pack_report(
        output_directory / "pack-report.json", roots=roots, labels=labels,
        prior_manifests=prior_manifests,
    )


def _validate_model_output(
    output_directory: pathlib.Path, *, seeds: Sequence[int],
    new_manifests: Sequence[pathlib.Path],
    anchor_manifests: Sequence[pathlib.Path],
    adjudicator_manifest: pathlib.Path,
    new_rows: int,
    anchor_rows: int,
) -> dict:
    runtime = output_directory / "jacek_replay_bfm.runtime"
    manifest_path = output_directory / "jacek_replay_bfm.runtime.json"
    manifest = _load_json(manifest_path, "self-search model manifest")
    training_report = manifest.get("training", {})
    reports = training_report.get("seed_reports")
    expected_sources = [
        _load_json(path, "self-search source shard")
        for path in (*new_manifests, *anchor_manifests, adjudicator_manifest)
    ]
    expected_batching = {
        "kind": "deterministic-two-stream-cycling-v1",
        "new_rows_per_batch": new_rows,
        "anchor_rows_per_batch": anchor_rows,
        "epoch_length": "new-stream-covered-once-anchor-sampled",
        "row_order": "new-then-anchor",
    }
    if (
        manifest.get("schema") != "papersoccer.jacek-replay-bfm-model.v1"
        or not runtime.is_file()
        or manifest.get("runtime", {}).get("artifact_sha256") != sha256(runtime)
        or manifest.get("architecture", {}).get("dimensions") != [6301, 192, 32, 1]
        or manifest.get("architecture", {}).get("biases") is not False
        or manifest.get("source_shards") != expected_sources
        or not isinstance(reports, list)
        or [report.get("seed") for report in reports if isinstance(report, dict)]
        != list(seeds)
        or training_report.get("selection_validation", {}).get("kind")
        != "explicit-common-adjudicator"
        or training_report.get("optimizer") != {
            "name": "adamw", "epochs": 50, "patience": 8,
            "batch_size": 256, "learning_rate": 0.001,
            "weight_decay": 1e-5, "gradient_norm_clip": 5.0,
        }
        or training_report.get("loss") != {"name": "weighted-huber", "delta": 0.25}
        or training_report.get("batching") != expected_batching
    ):
        raise ValueError("self-search selected model is stale or incomplete")
    publications = training_report.get("seed_checkpoints")
    checkpoint_directory = output_directory / "training-seeds"
    if (
        not isinstance(publications, list)
        or [item.get("seed") for item in publications if isinstance(item, dict)]
        != list(seeds)
        or not checkpoint_directory.is_dir()
    ):
        raise ValueError("self-search seed checkpoint set is incomplete")
    expected_files: set[str] = set()
    for seed, publication in zip(seeds, publications, strict=True):
        if not isinstance(publication, dict):
            raise ValueError("self-search seed checkpoint binding is malformed")
        checkpoint_name = f"seed-{seed}.runtime"
        receipt_name = f"seed-{seed}.json"
        checkpoint = checkpoint_directory / checkpoint_name
        receipt_path = checkpoint_directory / receipt_name
        expected_files.update((checkpoint_name, receipt_name))
        if (
            publication.get("checkpoint") != checkpoint_name
            or publication.get("receipt") != receipt_name
            or not checkpoint.is_file()
            or not receipt_path.is_file()
            or publication.get("checkpoint_sha256") != sha256(checkpoint)
            or publication.get("receipt_sha256") != sha256(receipt_path)
        ):
            raise ValueError("self-search seed checkpoint artifact is stale")
        receipt = _load_json(receipt_path, "self-search seed checkpoint receipt")
        if (
            receipt.get("schema")
            != "papersoccer.jacek-replay-bfm-seed-checkpoint.v1"
            or receipt.get("seed") != seed
            or receipt.get("checkpoint", {}).get("file") != checkpoint_name
            or receipt.get("checkpoint", {}).get("artifact_sha256")
            != sha256(checkpoint)
        ):
            raise ValueError("self-search seed checkpoint receipt is stale")
    if {path.name for path in checkpoint_directory.iterdir()} != expected_files:
        raise ValueError("self-search seed checkpoint directory has unexpected files")
    # This also validates the exact 6301->192->32->1 runtime layout.
    import jacek_replay_train as training
    training.load_runtime(runtime)
    return manifest


def run_training_arm(
    *, python: pathlib.Path, trainer: pathlib.Path,
    new_manifests: Sequence[pathlib.Path], anchor_manifests: Sequence[pathlib.Path],
    adjudicator_manifest: pathlib.Path, output_directory: pathlib.Path,
    seeds: Sequence[int], new_rows: int, anchor_rows: int,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    selected_runtime = output_directory / "jacek_replay_bfm.runtime"
    selected_manifest = output_directory / "jacek_replay_bfm.runtime.json"
    if selected_runtime.exists() or selected_manifest.exists():
        if not selected_runtime.is_file() or not selected_manifest.is_file():
            raise ValueError("self-search selected model publication is partial")
        return _validate_model_output(
            output_directory, seeds=seeds, new_manifests=new_manifests,
            anchor_manifests=anchor_manifests,
            adjudicator_manifest=adjudicator_manifest,
            new_rows=new_rows, anchor_rows=anchor_rows,
        )
    command = [str(python), str(trainer)]
    for path in new_manifests:
        command.extend(("--new-shard-manifest", str(path)))
    for path in anchor_manifests:
        command.extend(("--anchor-shard-manifest", str(path)))
    command.extend(("--selection-validation-manifest", str(adjudicator_manifest)))
    command.extend(
        (
            "--new-rows-per-batch", str(new_rows),
            "--anchor-rows-per-batch", str(anchor_rows),
            "--batch-size", "256", "--seeds", ",".join(map(str, seeds)),
            "--epochs", "50", "--patience", "8", "--learning-rate", "0.001",
            "--weight-decay", "1e-5", "--seed-workers", "2",
            "--seed-checkpoint-directory", str(output_directory / "training-seeds"),
            "--resume-seeds", "--output-directory", str(output_directory),
        )
    )
    _run(command)
    return _validate_model_output(
        output_directory, seeds=seeds, new_manifests=new_manifests,
        anchor_manifests=anchor_manifests,
        adjudicator_manifest=adjudicator_manifest,
        new_rows=new_rows, anchor_rows=anchor_rows,
    )


def anchor_metrics(
    *, candidate_runtime: pathlib.Path, incumbent_runtime: pathlib.Path,
    anchor_validation_manifests: Sequence[pathlib.Path],
) -> dict:
    import jacek_replay_train as training

    shards = [training.load_csr_shard(path) for path in anchor_validation_manifests]
    if not shards or any(shard.split != "validation" or len(shard) == 0 for shard in shards):
        raise ValueError("canonical anchor validation shards are invalid")
    dataset = training.combine_shards(shards)
    candidate, _ = training.load_runtime(candidate_runtime)
    incumbent, _ = training.load_runtime(incumbent_runtime)
    return {
        "schema": "papersoccer.jacek-selfsearch-anchor-metrics.v1",
        "anchor_validation": [
            artifact_snapshot(path) for path in anchor_validation_manifests
        ],
        "candidate": artifact_snapshot(candidate_runtime),
        "incumbent": artifact_snapshot(incumbent_runtime),
        "candidate_metrics": training.metrics(candidate, dataset),
        "incumbent_metrics": training.metrics(incumbent, dataset),
    }


def _comparison_bank_states(
    path: pathlib.Path, classification: str | None = None
) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        len(lines) < 7
        or lines[0] != "# papersoccer.jacek-replay-bfm-opening-bank.v1"
        or (
            classification is not None
            and lines[2] != f"# classification={classification}"
        )
        or lines[5] != "opening_id\ttranscript\tstate_identity"
    ):
        raise ValueError(f"comparison opening bank is malformed: {path}")
    states = set()
    for line in lines[6:]:
        fields = line.split("\t")
        if len(fields) != 3 or fields[2] in states:
            raise ValueError(f"comparison opening bank repeats a state: {path}")
        states.add(fields[2])
    if not states:
        raise ValueError(f"comparison opening bank is empty: {path}")
    return states


def generate_comparison_bank(
    *, comparison: pathlib.Path, output: pathlib.Path, pairs: int, seed: int,
    exclusions: Sequence[pathlib.Path], classification: str,
) -> dict:
    if classification not in {"development", "final"}:
        raise ValueError("comparison bank classification is invalid")
    excluded: set[str] = set()
    for path in exclusions:
        excluded.update(_comparison_bank_states(path))
    if output.exists():
        lines = output.read_text(encoding="utf-8").splitlines()
        states = _comparison_bank_states(output, classification)
        if (
            len(states) != pairs
            or len(lines) < 6
            or lines[3] != f"# seed={seed}"
            or lines[4] != "# minimum-physical-plies=12"
            or states & excluded
        ):
            raise ValueError("existing comparison opening bank is stale")
        return {
            "pairs": pairs, "seed": seed, "opening_plies": 12,
            "classification": classification,
            "states_sha256": hashlib.sha256(
                "\n".join(sorted(states)).encode()
            ).hexdigest(),
            "exclusions": [artifact_snapshot(path) for path in exclusions],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
    temporary.unlink()
    try:
        command = [
            str(comparison), "--generate-bank", str(temporary),
            "--bank-classification", classification, "--pairs", str(pairs),
            "--opening-plies", "12", "--seed", str(seed),
        ]
        _run(command)
        states = _comparison_bank_states(temporary, classification)
        if len(states) != pairs:
            raise ValueError("comparison opening bank pair count is stale")
        if states & excluded:
            raise ValueError("new comparison opening bank overlaps protected evidence")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "pairs": pairs, "seed": seed, "opening_plies": 12,
        "classification": classification,
        "states_sha256": hashlib.sha256("\n".join(sorted(states)).encode()).hexdigest(),
        "exclusions": [artifact_snapshot(path) for path in exclusions],
    }


@dataclasses.dataclass(frozen=True)
class Panel:
    name: str
    opponent: str
    control_model: pathlib.Path | None = None


def _comparison_command(
    *, comparison: pathlib.Path, model: pathlib.Path, bank: pathlib.Path,
    output: pathlib.Path, panel: Panel, pairs: int, time_ms: int,
    classification: str,
    pair_offset: int = 0,
) -> list[str]:
    command = [
        str(comparison), "--model", str(model), "--output", str(output),
        "--bank", str(bank), "--bank-classification", classification,
        "--opponent", panel.opponent, "--pairs", str(pairs),
        "--pair-offset", str(pair_offset),
        "--opening-plies", "12", "--max-turns", "320",
        "--seed", "20919592877381169", "--time-ms", str(time_ms),
        "--control-work", "3000000", "--tree-nodes", "1000000",
        "--control-tree-nodes", "1000000", "--max-actions", "250",
        "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
    ]
    if panel.control_model is not None:
        command.extend(("--control-model", str(panel.control_model)))
    return command


def _validate_panel_report(
    *, path: pathlib.Path, comparison: pathlib.Path, model: pathlib.Path,
    bank: pathlib.Path,
    panel: Panel, pairs: int, time_ms: int, classification: str,
    source_identities: Mapping[str, str],
    pair_offset: int = 0,
) -> dict:
    report = validate_gate_report(
        path, pairs=pairs, opponent=panel.opponent, time_ms=time_ms,
        bank_classification=classification,
    )
    configuration = report["configuration"]
    expected_control = sha256(panel.control_model) if panel.control_model else None
    if (
        report.get("model_sha256") != sha256(model)
        or configuration.get("opening_bank_sha256") != sha256(bank)
        or configuration.get("opening_bank_classification") != classification
        or configuration.get("pair_offset") != pair_offset
        or configuration.get("opening_plies") != 12
        or configuration.get("control_model_sha256") != expected_control
        or configuration.get("candidate_tree_nodes") != 1_000_000
        or configuration.get("control_tree_nodes") != 1_000_000
        or configuration.get("control_work") != 3_000_000
        or configuration.get("max_actions") != 250
        or configuration.get("max_partial_paths") != 50_000
        or configuration.get("max_turns") != 320
        or configuration.get("seed") != 20_919_592_877_381_169
        or configuration.get("single_thread") is not True
        or configuration.get("comparison_executable_sha256") != sha256(comparison)
        or any(
            configuration.get(field) != source_identities[field]
            for field in (
                "rank4_control_sha256", "rank4_engine_sha256",
                "neural_puct_control_sha256", "neural_puct_engine_sha256",
                "jacek_nn_control_sha256", "jacek_nn_engine_sha256",
                "rank4_adapter_sha256", "neural_puct_adapter_sha256",
                "jacek_nn_adapter_sha256", "jacek_nn_source_sha256",
                "shared_core_sha256", "candidate_source_sha256",
                "comparison_source_sha256",
            )
        )
        or any(not isinstance(game.get("candidate_ms"), list)
               for game in report["results"])
    ):
        raise ValueError(f"comparison panel is stale or incomplete: {path}")
    return report


def _merge_comparison_shards(
    *, shards: Sequence[pathlib.Path], output: pathlib.Path, pairs: int
) -> None:
    if not shards:
        raise ValueError("comparison panel has no shards")
    payloads = [_load_json(path, "comparison gate shard") for path in shards]
    results = [game for payload in payloads for game in payload["results"]]
    configurations = [payload["configuration"] for payload in payloads]
    state_ids = [
        state
        for configuration in configurations
        for state in configuration.get("opening_state_identities", [])
    ]
    if len(results) != 2 * pairs or len(state_ids) != pairs:
        raise ValueError("comparison gate shards do not exactly cover the panel")
    configuration = dict(configurations[0])
    configuration["pairs"] = pairs
    configuration["pair_offset"] = 0
    configuration["opening_state_identities"] = state_ids
    candidate_samples = [sample for game in results for sample in game["candidate_ms"]]
    control_samples = [sample for game in results for sample in game["control_ms"]]

    def latency(samples: list[float]) -> dict:
        ordered = sorted(samples)
        return {
            "decisions": len(ordered),
            "total_ms": sum(ordered),
            "p99_ms": ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)],
            "max_ms": max(ordered),
        }

    wins = sum(game.get("winner") == game.get("candidate_player") for game in results)
    unfinished = sum(game.get("winner") not in (0, 1) for game in results)
    report = {
        **{key: value for key, value in payloads[0].items()
           if key not in {"configuration", "summary", "results"}},
        "configuration": configuration,
        "summary": {
            "games": len(results), "wins": wins,
            "losses": len(results) - wins - unfinished,
            "unfinished": unfinished,
            "illegal": sum(bool(game.get("illegal")) for game in results),
            "colors": [
                {
                    "games": sum(game.get("candidate_player") == color for game in results),
                    "wins": sum(
                        game.get("candidate_player") == color
                        and game.get("winner") == color for game in results
                    ),
                }
                for color in (0, 1)
            ],
            "candidate": latency(candidate_samples),
            "control": latency(control_samples),
        },
        "results": results,
        "shards": [artifact_snapshot(path) for path in shards],
    }
    _atomic_json(output, report)


def run_comparison_panels(
    *, manager: StageManager, stage_ordinal: int, comparison: pathlib.Path,
    model: pathlib.Path, bank: pathlib.Path, panels: Sequence[Panel],
    pairs: int, time_ms: int, workers: int, classification: str,
    source_identities: Mapping[str, str], shard_pairs: int = 5,
) -> dict:
    if pairs <= 0 or shard_pairs <= 0 or pairs % shard_pairs != 0:
        raise ValueError("comparison panel pair count is not evenly shardable")
    receipt_root = manager.receipts / f"{stage_ordinal:02d}-game-gates-panels"
    shard_root = manager.output / "game-gates/shards"
    report_root = manager.output / "game-gates"
    specs = []
    common_inputs = {
        "comparison": artifact_snapshot(comparison), "model": artifact_snapshot(model),
        "bank": artifact_snapshot(bank),
    }
    for panel_ordinal, panel in enumerate(panels):
        for pair_offset in range(0, pairs, shard_pairs):
            report = shard_root / panel.name / f"offset-{pair_offset:04d}.json"
            receipt = receipt_root / panel.name / f"offset-{pair_offset:04d}.json"
            expected = {
                "schema": CAMPAIGN_RECEIPT_SCHEMA,
                "campaign_id": manager.campaign_id,
                "stage": "game-gates",
                "panel": {
                    "ordinal": panel_ordinal, "name": panel.name,
                    "opponent": panel.opponent,
                    "control_model": str(panel.control_model)
                    if panel.control_model else None,
                },
                "pair_offset": pair_offset, "pairs": shard_pairs,
                "time_ms": time_ms, "bank_classification": classification,
                "inputs": {
                    **common_inputs,
                    "control_model": artifact_snapshot(panel.control_model)
                    if panel.control_model else None,
                },
            }
            specs.append((panel, pair_offset, report, receipt, expected))

    def validate(item: tuple) -> dict:
        panel, pair_offset, report, receipt, expected = item
        saved = _load_json(receipt, "comparison panel shard receipt")
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError("comparison panel shard receipt is stale")
        if saved.get("report") != artifact_snapshot(report):
            raise ValueError("comparison panel shard report binding is stale")
        _validate_panel_report(
            path=report, comparison=comparison, model=model, bank=bank, panel=panel,
            pairs=shard_pairs, pair_offset=pair_offset, time_ms=time_ms,
            classification=classification, source_identities=source_identities,
        )
        return saved

    def execute(item: tuple) -> dict:
        panel, pair_offset, report, receipt, expected = item
        if receipt.exists():
            if not manager.resume:
                raise ValueError("comparison shard already completed; use --resume")
            return validate(item)
        if report.exists():
            raise ValueError("comparison shard has unreceipted output")
        report.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=report.parent, prefix=f".{report.name}.", delete=False
        ) as handle:
            temporary = pathlib.Path(handle.name)
        temporary.unlink()
        try:
            _run(_comparison_command(
                comparison=comparison, model=model, bank=bank, output=temporary,
                panel=panel, pairs=shard_pairs, pair_offset=pair_offset,
                time_ms=time_ms, classification=classification,
            ))
            _validate_panel_report(
                path=temporary, comparison=comparison, model=model, bank=bank,
                panel=panel, pairs=shard_pairs, pair_offset=pair_offset,
                time_ms=time_ms, classification=classification,
                source_identities=source_identities,
            )
            os.replace(temporary, report)
        finally:
            temporary.unlink(missing_ok=True)
        saved = {**expected, "report": artifact_snapshot(report)}
        _atomic_json(receipt, saved)
        return saved

    missing = [item for item in specs if not item[3].exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(execute, item) for item in missing]
            ):
                future.result()
    for item in specs:
        validate(item)
    reports = {}
    for panel in panels:
        final_report = report_root / f"{panel.name}.json"
        panel_shards = [item[2] for item in specs if item[0] == panel]
        _merge_comparison_shards(shards=panel_shards, output=final_report, pairs=pairs)
        _validate_panel_report(
            path=final_report, comparison=comparison, model=model, bank=bank,
            panel=panel, pairs=pairs, time_ms=time_ms,
            classification=classification, source_identities=source_identities,
        )
        reports[panel.name] = str(final_report)
    return {
        "reports": reports,
        "panel_receipts": [artifact_snapshot(item[3]) for item in specs],
        "panel_shards": [artifact_snapshot(item[2]) for item in specs],
    }


def run_latency_audit(
    *, comparison: pathlib.Path, model: pathlib.Path, bank: pathlib.Path,
    output: pathlib.Path, classification: str,
    source_identities: Mapping[str, str],
) -> dict:
    panel = Panel("uncontended", "rank4")
    if output.exists():
        report = _validate_panel_report(
            path=output, comparison=comparison, model=model, bank=bank,
            panel=panel, pairs=10, time_ms=980,
            classification=classification, source_identities=source_identities,
        )
        samples = [
            sample for row in report["results"] for sample in row["candidate_ms"]
        ]
        if not samples:
            raise ValueError("latency audit contains no candidate samples")
        return {"candidate_samples": len(samples), "candidate_max_ms": max(samples)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
    temporary.unlink()
    try:
        _run(_comparison_command(
            comparison=comparison, model=model, bank=bank, output=temporary,
            panel=panel, pairs=10, time_ms=980, classification=classification,
        ))
        report = _validate_panel_report(
            path=temporary, comparison=comparison, model=model, bank=bank,
            panel=panel, pairs=10,
            time_ms=980, classification=classification,
            source_identities=source_identities,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    samples = [sample for row in report["results"] for sample in row["candidate_ms"]]
    if not samples:
        raise ValueError("latency audit contains no candidate samples")
    return {"candidate_samples": len(samples), "candidate_max_ms": max(samples)}


def _validate_snapshot_bindings(value: object, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} has no receipt bindings")
    for record in value:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or artifact_snapshot(pathlib.Path(record["path"])) != record
        ):
            raise ValueError(f"{label} receipt binding is stale")


def _validate_chunk_stage_result(result: Mapping[str, object], label: str) -> None:
    _validate_snapshot_bindings(result.get("chunks"), f"{label} receipts")
    _validate_snapshot_bindings(
        result.get("chunk_artifacts"), f"{label} chunk artifacts"
    )


def _validate_gate_stage_result(result: Mapping[str, object]) -> None:
    reports = result.get("reports")
    if not isinstance(reports, dict) or not reports:
        raise ValueError("game gate result has no reports")
    _validate_snapshot_bindings(
        [artifact_snapshot(pathlib.Path(path)) for path in reports.values()],
        "game gate reports",
    )
    _validate_snapshot_bindings(
        result.get("panel_receipts"), "game gate panel receipts"
    )
    _validate_snapshot_bindings(
        result.get("panel_shards"), "game gate panel shards"
    )


def _pack_manifest(report_path: pathlib.Path, split: str) -> pathlib.Path:
    report = _load_json(report_path, "pack report")
    path = pathlib.Path(str(report.get("shards", {}).get(split, {}).get("manifest", "")))
    if not path.is_file():
        raise ValueError(f"pack report has no {split} shard manifest")
    return path


def run_phase(
    *, spec: PhaseSpec, output: pathlib.Path, resume: bool,
    roots_tsv: pathlib.Path, roots_manifest: pathlib.Path,
    actor: pathlib.Path, diversity: pathlib.Path,
    executables: CampaignExecutables,
    anchor_train_manifests: Sequence[pathlib.Path],
    anchor_validation_manifests: Sequence[pathlib.Path],
    canonical_prior_manifests: Sequence[pathlib.Path],
    opening_exclusions: Sequence[pathlib.Path],
    prior_search_manifests: Sequence[pathlib.Path] = (),
    prior_rank4_manifests: Sequence[pathlib.Path] = (),
    producer_guard: Callable[[], None] = lambda: None,
    build_manifest: pathlib.Path | None = None,
    source_identities: Mapping[str, str] | None = None,
) -> dict:
    """Run one generic pilot/full phase through its decision receipt."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if build_manifest is None or source_identities is None:
        raise ValueError("campaign phase has no frozen Release build identity")
    phase_environment = environment_identity()
    phase_environment["release_build"] = artifact_snapshot(build_manifest)
    manager = GuardedStageManager(
        output=output, campaign_id=spec.campaign_id,
        round_index=0 if spec.name == "pilot" else 1,
        resume=resume, environment=phase_environment,
        producer_guard=producer_guard,
    )
    workflow_source = pathlib.Path(__file__).resolve()
    # Keep a virtual-environment launcher path intact.  Resolving its symlink
    # selects the base interpreter and silently drops the research dependencies
    # (notably NumPy) installed only in the campaign venv.
    python = pathlib.Path(os.path.abspath(sys.executable))
    plan_json = output / "game-plan.json"
    plan_tsv = output / "game-plan.tsv"
    games_tsv = output / "games.tsv"
    games_manifest = output / "games.manifest.json"
    positions_tsv = output / "positions.tsv"
    positions_manifest = output / "positions.manifest.json"
    shallow_search = output / "labels/search-shallow.jsonl"
    shallow_rank4 = output / "labels/rank4-shallow.jsonl"
    hard_positions = output / "hard-positions.tsv"
    hard_manifest = output / "hard-positions.manifest.json"
    deep_search = output / "labels/search-deep.jsonl"
    deep_rank4 = output / "labels/rank4-deep.jsonl"
    merged_search = output / "labels/search-merged.jsonl"
    merged_rank4 = output / "labels/rank4-merged.jsonl"
    adjudicator_positions = output / "adjudicator-positions.tsv"
    adjudicator_labels = output / "labels/adjudicator.jsonl"
    pack_search_directory = output / "shards/search"
    pack_rank4_directory = output / "shards/rank4"
    pack_adjudicator_directory = output / "shards/adjudicator"
    pack_search_report = pack_search_directory / "pack-report.json"
    pack_rank4_report = pack_rank4_directory / "pack-report.json"
    pack_adjudicator_report = pack_adjudicator_directory / "pack-report.json"
    search_model_directory = output / "models/search"
    rank4_model_directory = output / "models/rank4"
    search_runtime = search_model_directory / "jacek_replay_bfm.runtime"
    search_model_manifest = search_model_directory / "jacek_replay_bfm.runtime.json"
    rank4_runtime = rank4_model_directory / "jacek_replay_bfm.runtime"
    rank4_model_manifest = rank4_model_directory / "jacek_replay_bfm.runtime.json"
    anchor_metrics_path = output / "anchor-metrics.json"
    bank = output / "gate-openings.tsv"
    latency_report = output / "latency-audit.json"
    decision_path = output / "decision.json"

    def make_plan() -> dict:
        plan = make_game_plan(
            campaign_id=spec.campaign_id, seed=spec.game_seed, quotas=spec.quotas
        )
        _atomic_json(plan_json, plan)
        _atomic_bytes(plan_tsv, render_game_plan_tsv(plan))
        return {"games": plan["games"], "quotas": plan["quotas"]}

    manager.execute(
        ordinal=0, name="game-plan",
        configuration={"seed": spec.game_seed, "quotas": dict(spec.quotas)},
        producers={"workflow": workflow_source}, inputs={},
        outputs={"plan": plan_json, "plan_tsv": plan_tsv}, action=make_plan,
    )

    game_result = manager.execute(
        ordinal=1, name="games",
        configuration={
            "workers": int(spec.configuration["game_workers"]),
            "chunk_games": int(spec.configuration["game_chunk_size"]),
            "actors": {
                key: spec.configuration[key]
                for key in (
                    "bfm_actor_tree_nodes", "rank4_actor_nodes", "jacek_nn_actor_nodes",
                    "exploration", "fpu", "early_exploration_percent",
                    "early_exploration_turns",
                )
            },
        },
        producers={"workflow": workflow_source,
                   "continuation_generator": executables.continuation_generator},
        inputs={"plan": plan_json, "plan_tsv": plan_tsv, "roots_tsv": roots_tsv,
                "actor": actor, "diversity": diversity},
        outputs={"games": games_tsv, "manifest": games_manifest},
        action=lambda: run_game_chunks(
            manager=manager, stage_ordinal=1, spec=spec, plan_path=plan_json,
            roots_tsv=roots_tsv, actor=actor, diversity=diversity,
            generator=executables.continuation_generator,
            workers=int(spec.configuration["game_workers"]),
            source_identities=source_identities,
        ),
        validator=lambda result: _validate_chunk_stage_result(result, "game stage"),
    )
    if game_result.get("games") != int(spec.configuration["games"]):
        raise ValueError("game stage did not satisfy the exact profile quota")

    def make_positions() -> dict:
        payload, manifest = freeze_positions(
            campaign_id=spec.campaign_id, games_tsv=games_tsv,
            games_manifest=games_manifest, roots_manifest=roots_manifest,
            maximum_per_game=int(spec.configuration["positions_per_game"]),
        )
        write_pair(payload, manifest, positions_tsv, positions_manifest)
        return {"positions": manifest["positions"], "split_counts": manifest["split_counts"]}

    manager.execute(
        ordinal=2, name="positions",
        configuration={"maximum_per_game": spec.configuration["positions_per_game"]},
        producers={"workflow": workflow_source},
        inputs={"games": games_tsv, "game_manifest": games_manifest,
                "roots": roots_manifest},
        outputs={"positions": positions_tsv, "manifest": positions_manifest},
        action=make_positions,
    )

    def label_stage(
        *, ordinal: int, name: str, source: pathlib.Path, target: pathlib.Path,
        teacher: pathlib.Path, schema: str, nodes: int,
        model: pathlib.Path | None = None,
    ) -> dict:
        source_sha256 = source_identities[
            "search_teacher_source_sha256"
            if schema == SEARCH_TEACHER_SCHEMA
            else "rank4_teacher_source_sha256"
        ]
        return manager.execute(
            ordinal=ordinal, name=name,
            configuration={
                "nodes": nodes, "workers": 10, "chunk_games": 25,
                "safety_ms": SEARCH_SAFETY_MS,
                "max_actions": SEARCH_MAX_ACTIONS if model else None,
                "max_partial_paths": SEARCH_MAX_PARTIAL_PATHS if model else None,
                "exploration": 0.5 if model else None,
                "fpu": 0.5 if model else None,
                "source_sha256": source_sha256,
            },
            producers={"workflow": workflow_source, "teacher": teacher},
            inputs={"positions": source, **({"model": model} if model else {})},
            outputs={"labels": target},
            action=lambda: run_label_chunks(
                manager=manager, stage_ordinal=ordinal, stage_name=name,
                positions=source, output=target, teacher=teacher, schema=schema,
                campaign_id=spec.campaign_id, nodes=nodes, workers=10, model=model,
                source_sha256=source_sha256,
            ),
            validator=lambda result: _validate_chunk_stage_result(
                result, f"{name} stage"
            ),
        )

    label_stage(
        ordinal=3, name="search-shallow", source=positions_tsv,
        target=shallow_search, teacher=executables.search_teacher,
        schema=SEARCH_TEACHER_SCHEMA,
        nodes=int(spec.configuration["bfm_shallow_tree_nodes"]), model=actor,
    )
    label_stage(
        ordinal=4, name="rank4-shallow", source=positions_tsv,
        target=shallow_rank4, teacher=executables.rank4_teacher,
        schema=RANK4_TEACHER_SCHEMA,
        nodes=int(spec.configuration["rank4_shallow_nodes"]),
    )

    def make_hard() -> dict:
        payload, manifest = select_hard_positions(
            positions_tsv=positions_tsv, search_labels=shallow_search,
            rank4_labels=shallow_rank4,
            numerator=int(spec.configuration["hard_fraction_numerator"]),
            denominator=int(spec.configuration["hard_fraction_denominator"]),
        )
        write_pair(payload, manifest, hard_positions, hard_manifest)
        return {"selected": manifest["selected"], "games": manifest["games"]}

    manager.execute(
        ordinal=5, name="hard-selection",
        configuration={"fraction": [spec.configuration["hard_fraction_numerator"],
                                      spec.configuration["hard_fraction_denominator"]]},
        producers={"workflow": workflow_source},
        inputs={"positions": positions_tsv, "search": shallow_search,
                "rank4": shallow_rank4},
        outputs={"positions": hard_positions, "manifest": hard_manifest}, action=make_hard,
    )
    label_stage(
        ordinal=6, name="search-deep", source=hard_positions,
        target=deep_search, teacher=executables.search_teacher,
        schema=SEARCH_TEACHER_SCHEMA,
        nodes=int(spec.configuration["bfm_deep_tree_nodes"]), model=actor,
    )
    label_stage(
        ordinal=7, name="rank4-deep", source=hard_positions,
        target=deep_rank4, teacher=executables.rank4_teacher,
        schema=RANK4_TEACHER_SCHEMA,
        nodes=int(spec.configuration["rank4_deep_nodes"]),
    )

    def merge_labels(shallow: pathlib.Path, deep: pathlib.Path,
                     target: pathlib.Path, schema: str) -> dict:
        payload = merge_deep_labels(shallow=shallow, deep=deep, expected_schema=schema)
        _atomic_bytes(target, payload)
        return {"rows": len(payload.splitlines())}

    manager.execute(
        ordinal=8, name="search-targets", configuration={"deep_override": True},
        producers={"workflow": workflow_source, "corpus": pathlib.Path(corpus.__file__)},
        inputs={"shallow": shallow_search, "deep": deep_search},
        outputs={"labels": merged_search},
        action=lambda: merge_labels(shallow_search, deep_search, merged_search,
                                    SEARCH_TEACHER_SCHEMA),
    )
    manager.execute(
        ordinal=9, name="rank4-targets", configuration={"deep_override": True},
        producers={"workflow": workflow_source, "corpus": pathlib.Path(corpus.__file__)},
        inputs={"shallow": shallow_rank4, "deep": deep_rank4},
        outputs={"labels": merged_rank4},
        action=lambda: merge_labels(shallow_rank4, deep_rank4, merged_rank4,
                                    RANK4_TEACHER_SCHEMA),
    )

    adjudicator_priors = [
        *canonical_prior_manifests, *prior_search_manifests
    ]
    manager.execute(
        ordinal=10, name="adjudicator-positions",
        configuration={
            "positions": spec.configuration["adjudicator_positions"],
            "split_isolation": "exclude-current-train-and-all-prior-canonical-fingerprints",
        },
        producers={"workflow": workflow_source, "corpus": pathlib.Path(corpus.__file__)},
        inputs={
            "positions": positions_tsv,
            **{
                f"prior_{index}": path
                for index, path in enumerate(adjudicator_priors)
            },
        },
        outputs={"positions": adjudicator_positions},
        action=lambda: (
            _atomic_bytes(adjudicator_positions, common_adjudicator_positions(
                positions_tsv, int(spec.configuration["adjudicator_positions"]),
                adjudicator_priors,
            ))
            or {"positions": int(spec.configuration["adjudicator_positions"])}
        ),
    )
    label_stage(
        ordinal=11, name="adjudicator-labels", source=adjudicator_positions,
        target=adjudicator_labels, teacher=executables.search_teacher,
        schema=SEARCH_TEACHER_SCHEMA,
        nodes=int(spec.configuration["adjudicator_tree_nodes"]), model=actor,
    )

    def pack_stage(ordinal: int, name: str, labels: pathlib.Path,
                   directory: pathlib.Path, report: pathlib.Path,
                   prior_manifests: Sequence[pathlib.Path]) -> dict:
        return manager.execute(
            ordinal=ordinal, name=name,
            configuration={"streaming": True, "prior_shards": len(prior_manifests)},
            producers={"workflow": workflow_source, "pack": executables.pack_tool,
                       "corpus": pathlib.Path(corpus.__file__)},
            inputs={
                "roots": roots_manifest, "labels": labels,
                **{
                    f"prior_{index}": path
                    for index, path in enumerate(prior_manifests)
                },
            },
            outputs={"report": report},
            action=lambda: run_pack(
                python=python, pack_tool=executables.pack_tool, roots=roots_manifest,
                labels=labels, output_directory=directory,
                prior_manifests=prior_manifests,
            ),
            validator=lambda _result: _validate_pack_report(
                report, roots=roots_manifest, labels=labels,
                prior_manifests=prior_manifests,
            ),
        )

    search_priors = [*canonical_prior_manifests, *prior_search_manifests]
    rank4_priors = [*canonical_prior_manifests, *prior_rank4_manifests]
    pack_stage(
        12, "pack-search", merged_search, pack_search_directory,
        pack_search_report, search_priors,
    )
    pack_stage(
        13, "pack-rank4", merged_rank4, pack_rank4_directory,
        pack_rank4_report, rank4_priors,
    )
    search_new_manifests = tuple(
        _pack_manifest(pack_search_report, split)
        for split in ("train", "validation", "test")
    )
    rank4_new_manifests = tuple(
        _pack_manifest(pack_rank4_report, split)
        for split in ("train", "validation", "test")
    )
    pack_stage(
        14, "pack-adjudicator", adjudicator_labels,
        pack_adjudicator_directory, pack_adjudicator_report,
        [*search_priors, *search_new_manifests],
    )
    search_new = search_new_manifests[0]
    rank4_new = rank4_new_manifests[0]
    adjudicator_validation = _pack_manifest(pack_adjudicator_report, "validation")

    def training_stage(
        ordinal: int, name: str, new_manifests: Sequence[pathlib.Path],
        directory: pathlib.Path, runtime: pathlib.Path, manifest: pathlib.Path,
    ) -> dict:
        inputs = {f"new_{index}": path for index, path in enumerate(new_manifests)}
        inputs.update({
            **{
                f"anchor_{index}": path
                for index, path in enumerate(anchor_train_manifests)
            },
            "adjudicator": adjudicator_validation,
        })
        return manager.execute(
            ordinal=ordinal, name=name,
            configuration={
                "seeds": list(spec.configuration["training_seeds"]),
                "new_rows_per_batch": spec.configuration["new_rows_per_batch"],
                "anchor_rows_per_batch": spec.configuration["anchor_rows_per_batch"],
                "batch_size": 256, "epochs": 50, "patience": 8,
                "learning_rate": 0.001, "weight_decay": 1e-5, "seed_workers": 2,
            },
            producers={"workflow": workflow_source, "trainer": executables.trainer},
            inputs=inputs, outputs={"runtime": runtime, "manifest": manifest},
            action=lambda: run_training_arm(
                python=python, trainer=executables.trainer,
                new_manifests=new_manifests,
                anchor_manifests=anchor_train_manifests,
                adjudicator_manifest=adjudicator_validation, output_directory=directory,
                seeds=list(spec.configuration["training_seeds"]),
                new_rows=int(spec.configuration["new_rows_per_batch"]),
                anchor_rows=int(spec.configuration["anchor_rows_per_batch"]),
            ),
            validator=lambda _result: _validate_model_output(
                directory,
                seeds=list(spec.configuration["training_seeds"]),
                new_manifests=new_manifests,
                anchor_manifests=anchor_train_manifests,
                adjudicator_manifest=adjudicator_validation,
                new_rows=int(spec.configuration["new_rows_per_batch"]),
                anchor_rows=int(spec.configuration["anchor_rows_per_batch"]),
            ),
        )

    prior_search_train = [
        path for path in prior_search_manifests
        if _load_json(path, "prior search shard").get("split") == "train"
    ]
    prior_rank4_train = [
        path for path in prior_rank4_manifests
        if _load_json(path, "prior Rank-4 shard").get("split") == "train"
    ]
    training_stage(15, "train-search", [*prior_search_train, search_new],
                   search_model_directory, search_runtime, search_model_manifest)
    training_stage(16, "train-rank4", [*prior_rank4_train, rank4_new],
                   rank4_model_directory, rank4_runtime, rank4_model_manifest)

    def make_anchor_metrics() -> dict:
        report = anchor_metrics(
            candidate_runtime=search_runtime, incumbent_runtime=actor,
            anchor_validation_manifests=anchor_validation_manifests,
        )
        _atomic_json(anchor_metrics_path, report)
        return report

    manager.execute(
        ordinal=17, name="anchor-metrics", configuration={"canonical_anchor": True},
        producers={"workflow": workflow_source, "trainer": executables.trainer},
        inputs={
            "candidate": search_runtime,
            "incumbent": actor,
            **{
                f"validation_{index}": path
                for index, path in enumerate(anchor_validation_manifests)
            },
        },
        outputs={"metrics": anchor_metrics_path}, action=make_anchor_metrics,
    )

    manager.execute(
        ordinal=18, name="opening-bank",
        configuration={"pairs": spec.pairs, "plies": 12, "seed": spec.opening_seed,
                       "classification": spec.bank_classification},
        producers={"workflow": workflow_source, "comparison": executables.comparison},
        inputs={f"excluded_{index}": path
                for index, path in enumerate(opening_exclusions)},
        outputs={"bank": bank},
        action=lambda: generate_comparison_bank(
            comparison=executables.comparison, output=bank, pairs=spec.pairs,
            seed=spec.opening_seed, exclusions=opening_exclusions,
            classification=spec.bank_classification,
        ),
    )

    panels = [
        Panel("matched", "jacek-replay", rank4_runtime),
        Panel("incumbent" if spec.name == "pilot" else "pilot-teacher",
              "jacek-replay", actor),
        Panel("rank4", "rank4"), Panel("jacek-nn", "jacek-nn"),
    ]
    gate_reports = {panel.name: output / "game-gates" / f"{panel.name}.json"
                    for panel in panels}
    gate_result = manager.execute(
        ordinal=19, name="game-gates",
        configuration={"pairs": spec.pairs, "time_ms": spec.gate_time_ms,
                       "workers": spec.gate_workers, "shard_pairs": 5,
                       "panels": [p.name for p in panels],
                       "bank_classification": spec.bank_classification},
        producers={"workflow": workflow_source, "comparison": executables.comparison},
        inputs={"model": search_runtime, "bank": bank, "matched": rank4_runtime,
                "actor": actor}, outputs=gate_reports,
        action=lambda: run_comparison_panels(
            manager=manager, stage_ordinal=19, comparison=executables.comparison,
            model=search_runtime, bank=bank, panels=panels, pairs=spec.pairs,
            time_ms=spec.gate_time_ms, workers=spec.gate_workers,
            classification=spec.bank_classification,
            source_identities=source_identities,
        ),
        validator=_validate_gate_stage_result,
    )
    if set(gate_result.get("reports", {})) != set(gate_reports):
        raise ValueError("game gate report set is incomplete")

    manager.execute(
        ordinal=20, name="latency-audit", configuration={"pairs": 10, "time_ms": 980,
                                                          "workers": 1,
                                                          "bank_classification": spec.bank_classification},
        producers={"workflow": workflow_source, "comparison": executables.comparison},
        inputs={"model": search_runtime, "bank": bank},
        outputs={"report": latency_report},
        action=lambda: run_latency_audit(
            comparison=executables.comparison, model=search_runtime,
            bank=bank, output=latency_report,
            classification=spec.bank_classification,
            source_identities=source_identities,
        ),
        validator=lambda result: (
            None if result.get("candidate_max_ms") == max(
                sample for row in _load_json(latency_report, "latency report")["results"]
                for sample in row["candidate_ms"]
            ) else (_ for _ in ()).throw(ValueError("latency result is stale"))
        ),
    )
    latency = _load_json(manager.receipt_path(20, "latency-audit"), "latency receipt")["result"]

    def decide() -> dict:
        if spec.name == "pilot":
            metrics = _load_json(anchor_metrics_path, "anchor metrics")
            decision = pilot_decision(
                matched_report=gate_reports["matched"],
                incumbent_report=gate_reports["incumbent"],
                rank4_report=gate_reports["rank4"],
                jacek_nn_report=gate_reports["jacek-nn"],
                anchor_candidate=metrics["candidate_metrics"],
                anchor_incumbent=metrics["incumbent_metrics"],
                uncontended_max_ms=float(latency["candidate_max_ms"]),
            )
        else:
            decision = final_decision(
                pilot_report=gate_reports["pilot-teacher"],
                matched_report=gate_reports["matched"],
                rank4_report=gate_reports["rank4"],
                jacek_nn_report=gate_reports["jacek-nn"],
                uncontended_max_ms=float(latency["candidate_max_ms"]),
            )
        _atomic_json(decision_path, decision)
        return decision

    decision = manager.execute(
        ordinal=21, name="decision", configuration={"profile": spec.name},
        producers={"workflow": workflow_source},
        inputs={**gate_reports, "latency": latency_report,
                "anchor_metrics": anchor_metrics_path},
        outputs={"decision": decision_path}, action=decide,
    )
    return {
        "profile": spec.name, "campaign_id": spec.campaign_id,
        "decision": decision, "decision_path": str(decision_path),
        "search_runtime": str(search_runtime), "search_manifest": str(search_model_manifest),
        "rank4_runtime": str(rank4_runtime), "rank4_manifest": str(rank4_model_manifest),
        "search_new_train_manifest": str(search_new),
        "rank4_new_train_manifest": str(rank4_new),
        "search_new_manifests": [str(path) for path in search_new_manifests],
        "rank4_new_manifests": [str(path) for path in rank4_new_manifests],
        "opening_bank": str(bank),
    }


def _repository_record(repository: pathlib.Path, expected_commit: str) -> dict:
    repository = repository.resolve()
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=repository, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode != 0:
            raise ValueError(f"git inspection failed: {completed.stderr.strip()}")
        return completed.stdout.strip()
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    if head != expected_commit:
        raise ValueError(f"campaign producer commit changed: expected {expected_commit}, got {head}")
    if status:
        raise ValueError("campaign producer tree has tracked modifications")
    return {
        "path": str(repository), "head": head,
        "branch": git("branch", "--show-current") or None,
        "tree": git("rev-parse", "HEAD^{tree}"), "clean": True,
    }


def _source_closure(repository: pathlib.Path, paths: Sequence[str]) -> str:
    material = "".join(
        f"{path}:{sha256(repository / path)}\n" for path in paths
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _source_identities(repository: pathlib.Path) -> dict:
    files = {
        "rank4_control_sha256": "submissions/codingame/bots/rank_4/submission.cpp",
        "rank4_engine_sha256": "submissions/codingame/bots/rank_4/bot.cpp",
        "neural_puct_control_sha256":
            "submissions/codingame/bots/neural_puct/submission.cpp",
        "neural_puct_engine_sha256":
            "submissions/codingame/bots/neural_puct/bot.cpp",
        "jacek_nn_control_sha256":
            "submissions/codingame/bots/jacek_nn/submission.cpp",
        "jacek_nn_engine_sha256": "submissions/codingame/bots/jacek_nn/bot.cpp",
        "rank4_adapter_sha256": "tools/jacek_replay_bfm_rank4_control.cpp",
        "neural_puct_adapter_sha256":
            "tools/jacek_replay_bfm_neural_puct_control.cpp",
        "jacek_nn_adapter_sha256": "tools/jacek_replay_bfm_jacek_nn_control.cpp",
    }
    return {
        "continuation_source_sha256": _source_closure(
            repository, CONTINUATION_SOURCE_PATHS
        ),
        "rank4_actor_source_sha256": _source_closure(
            repository, RANK4_ACTOR_SOURCE_PATHS
        ),
        "jacek_nn_actor_source_sha256": _source_closure(
            repository, JACEK_NN_ACTOR_SOURCE_PATHS
        ),
        "rank4_teacher_source_sha256": _source_closure(
            repository,
            (*RANK4_ACTOR_SOURCE_PATHS,
             "tools/jacek_replay_rank4_position_teacher.cpp"),
        ),
        "search_teacher_source_sha256": _source_closure(
            repository, SEARCH_TEACHER_SOURCE_PATHS
        ),
        "jacek_nn_source_sha256": _source_closure(
            repository, JACEK_NN_COMPARISON_SOURCE_PATHS
        ),
        "shared_core_sha256": _source_closure(
            repository, SHARED_CORE_SOURCE_PATHS
        ),
        "candidate_source_sha256": _source_closure(
            repository, BFM_RUNTIME_SOURCE_PATHS
        ),
        "comparison_source_sha256": _source_closure(
            repository, COMPARISON_SOURCE_PATHS
        ),
        **{name: sha256(repository / path) for name, path in files.items()},
    }


def _cmake_cache_values(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    return values


def _build_manifest_body(
    *, repository: pathlib.Path, expected_commit: str,
    executables: CampaignExecutables,
) -> dict:
    repository_record = _repository_record(repository, expected_commit)
    binary_paths = (
        executables.continuation_generator,
        executables.search_teacher,
        executables.rank4_teacher,
        executables.comparison,
    )
    build_directories = {path.resolve().parent for path in binary_paths}
    if len(build_directories) != 1:
        raise ValueError("campaign binaries do not come from one build directory")
    build_directory = next(iter(build_directories))
    cache_path = build_directory / "CMakeCache.txt"
    if not cache_path.is_file():
        raise ValueError("campaign Release build has no CMakeCache.txt")
    cache = _cmake_cache_values(cache_path)
    if (
        cache.get("CMAKE_BUILD_TYPE") != "Release"
        or pathlib.Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve()
        != repository.resolve()
        or cache.get("PAPERSOCCER_ENABLE_SANITIZERS") != "OFF"
    ):
        raise ValueError("campaign binaries are not one unsanitized Release build")
    tool_paths = {
        "workflow": pathlib.Path(__file__).resolve(),
        "canonical_workflow": repository / "tools/jacek_replay_workflow.py",
        "corpus": repository / "tools/jacek_replay_corpus.py",
        "features": repository / "tools/jacek_replay_features.py",
        "pack": executables.pack_tool,
        "trainer": executables.trainer,
        "cmake": repository / "CMakeLists.txt",
    }
    return {
        "schema": BUILD_MANIFEST_SCHEMA,
        "repository": repository_record,
        "build": {
            "directory": str(build_directory),
            "type": "Release",
            "sanitizers": False,
            "cmake_cache": artifact_snapshot(cache_path),
        },
        "executables": executables.snapshots(),
        "source_identities": _source_identities(repository),
        "tool_sources": {
            name: artifact_snapshot(path.resolve()) for name, path in tool_paths.items()
        },
    }


def write_build_manifest(
    *, repository: pathlib.Path, expected_commit: str,
    executables: CampaignExecutables, output: pathlib.Path,
) -> dict:
    body = _build_manifest_body(
        repository=repository.resolve(), expected_commit=expected_commit,
        executables=executables.resolved(),
    )
    manifest = {
        **body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    _atomic_json(output, manifest)
    return manifest


def validate_build_manifest(
    path: pathlib.Path, *, repository: pathlib.Path, expected_commit: str,
    executables: CampaignExecutables,
) -> dict:
    manifest = _load_json(path, "self-search Release build manifest")
    body = dict(manifest)
    body_sha256 = body.pop("body_sha256", None)
    expected = _build_manifest_body(
        repository=repository.resolve(), expected_commit=expected_commit,
        executables=executables.resolved(),
    )
    if (
        body != expected
        or body_sha256 != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError("self-search Release build manifest is stale")
    return manifest


def _evaluation_opening_banks(evaluation_directory: pathlib.Path) -> list[pathlib.Path]:
    manifest = _load_json(evaluation_directory / "run-manifest.json", "evaluation manifest")
    candidates: list[pathlib.Path] = []
    for panel in manifest.get("panels", {}).values():
        if isinstance(panel, dict) and isinstance(panel.get("opening_bank"), str):
            candidates.append(pathlib.Path(panel["opening_bank"]))
    for path in manifest.get("bank_selection", {}).get("excluded_banks", []):
        if isinstance(path, str):
            candidates.append(pathlib.Path(path))
    result = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            _comparison_bank_states(resolved)
            result.append(resolved)
            seen.add(resolved)
    if not result:
        raise ValueError("evaluation has no opening banks to exclude")
    return result


def _canonical_split_manifests(
    canonical_campaign: pathlib.Path,
) -> dict[str, tuple[pathlib.Path, ...]]:
    """Resolve every exact cumulative R0/R1/R2 shard manifest."""

    manifests_by_split: dict[str, list[pathlib.Path]] = {
        split: [] for split in ("train", "validation", "test")
    }
    source_shards: list[dict] = []
    for round_index in range(3):
        round_directory = canonical_campaign / f"round-{round_index}"
        workflow = _load_json(
            round_directory / "workflow.json", f"round-{round_index} workflow"
        )
        report_path = pathlib.Path(
            str(workflow.get("artifacts", {}).get("pack_report", {}).get("report", ""))
        )
        if not report_path.is_file():
            raise ValueError(
                f"canonical Round-{round_index} pack report is unavailable"
            )
        report = _load_json(
            report_path, f"canonical Round-{round_index} pack report"
        )
        for split in ("train", "validation", "test"):
            record = report.get("shards", {}).get(split)
            if not isinstance(record, dict):
                raise ValueError(
                    f"canonical Round-{round_index} omits its {split} shard"
                )
            path = pathlib.Path(str(record.get("manifest", ""))).resolve()
            manifest = _load_json(
                path, f"canonical Round-{round_index} {split} shard"
            )
            npz = path.parent / str(manifest.get("npz", ""))
            if (
                manifest.get("split") != split
                or sha256(path) != record.get("manifest_sha256")
                or not npz.is_file()
                or sha256(npz) != manifest.get("npz_sha256")
                or manifest.get("npz_sha256") != record.get("sha256")
            ):
                raise ValueError(
                    f"canonical Round-{round_index} {split} shard is stale"
                )
            source_shards.append(manifest)
            manifests_by_split[split].append(path)

    round_two_model = _load_json(
        canonical_campaign / "round-2/model/jacek_replay_bfm.runtime.json",
        "canonical Round-2 selected model",
    )
    if round_two_model.get("source_shards") != source_shards:
        raise ValueError(
            "canonical anchor manifests do not match the selected Round-2 ancestry"
        )
    if any(len(paths) != 3 for paths in manifests_by_split.values()):
        raise ValueError("canonical R0/R1/R2 anchor set is incomplete")
    return {split: tuple(paths) for split, paths in manifests_by_split.items()}


def _canonical_anchor_manifests(
    canonical_campaign: pathlib.Path,
) -> tuple[tuple[pathlib.Path, ...], tuple[pathlib.Path, ...]]:
    manifests = _canonical_split_manifests(canonical_campaign)
    return manifests["train"], manifests["validation"]


def _validate_canonical_campaign_inputs(
    canonical_campaign: pathlib.Path,
    *,
    roots_tsv: pathlib.Path,
    roots_manifest: pathlib.Path,
    anchor_train: Sequence[pathlib.Path],
    anchor_validation: Sequence[pathlib.Path],
) -> dict:
    workflow_path = (canonical_campaign / "round-2/workflow.json").resolve()
    chain = validate_canonical_workflow_chain(
        workflow_path, expected_round=2, offline=True
    )
    if [entry.get("round") for entry in chain.get("entries", [])] != [0, 1, 2]:
        raise ValueError("canonical workflow chain does not contain R0/R1/R2")
    receipt = chain["receipt"]
    artifacts = receipt.get("artifacts", {})
    roots = artifacts.get("roots", {})
    teacher_tsv = artifacts.get("teacher_tsv", {})

    def bound_path(record: Mapping[str, object], label: str) -> pathlib.Path:
        raw = record.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"canonical {label} path is missing")
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = workflow_path.parent / path
        return path.resolve()

    if (
        bound_path(roots, "roots") != roots_manifest.resolve()
        or roots.get("sha256") != sha256(roots_manifest)
        or bound_path(teacher_tsv, "teacher TSV") != roots_tsv.resolve()
        or teacher_tsv.get("sha256") != sha256(roots_tsv)
    ):
        raise ValueError("canonical roots or teacher TSV differ from validated ancestry")
    current_train, current_validation = _canonical_anchor_manifests(
        canonical_campaign
    )
    if tuple(anchor_train) != current_train or tuple(anchor_validation) != current_validation:
        raise ValueError("canonical anchor set changed after ancestry validation")
    return {
        "workflow": artifact_snapshot(workflow_path),
        "entries": chain["entries"],
        "roots": artifact_snapshot(roots_manifest),
        "teacher_tsv": artifact_snapshot(roots_tsv),
        "anchor_train": [artifact_snapshot(path) for path in anchor_train],
        "anchor_validation": [
            artifact_snapshot(path) for path in anchor_validation
        ],
    }


def _copy_atomic(source: pathlib.Path, target: pathlib.Path) -> None:
    payload = source.read_bytes()
    _atomic_bytes(target, payload)
    if sha256(target) != sha256(source):
        raise RuntimeError("atomic campaign snapshot changed bytes")


def run_campaign(
    *, repository: pathlib.Path, expected_commit: str,
    evaluation_directory: pathlib.Path, canonical_campaign: pathlib.Path,
    output: pathlib.Path, executables: CampaignExecutables,
    build_manifest: pathlib.Path,
    resume: bool, wait_for_evaluation: bool, poll_seconds: float,
    skip_power_check: bool,
) -> dict:
    """Wait for the prerequisite audit, then run pilot and conditional full phase."""

    global _CAMPAIGN_LOCK_FD

    repository = repository.resolve()
    evaluation_directory = evaluation_directory.resolve()
    canonical_campaign = canonical_campaign.resolve()
    output = output.resolve()
    if output.name != AUTO_CAMPAIGN_ID:
        raise ValueError(
            f"campaign output directory must be named {AUTO_CAMPAIGN_ID}"
        )
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "supervisor-status.json"
    lock_path = output / "supervisor.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("self-search campaign supervisor is already running") from error
        _CAMPAIGN_LOCK_FD = lock.fileno()
        try:
            executables = executables.resolved()
            executables.validate()
            repository_record = _repository_record(repository, expected_commit)
            executable_records = executables.snapshots()
            build_manifest = build_manifest.resolve()
            build_record = validate_build_manifest(
                build_manifest, repository=repository,
                expected_commit=expected_commit, executables=executables,
            )
            frozen_input_records: dict[str, dict] = {}

            def producer_guard() -> None:
                if _repository_record(repository, expected_commit) != repository_record:
                    raise ValueError("campaign repository identity changed after freeze")
                if executables.snapshots() != executable_records:
                    raise ValueError("campaign executable identity changed after freeze")
                if validate_build_manifest(
                    build_manifest, repository=repository,
                    expected_commit=expected_commit, executables=executables,
                ) != build_record:
                    raise ValueError("campaign Release build identity changed after freeze")
                for record in frozen_input_records.values():
                    if artifact_snapshot(pathlib.Path(record["path"])) != record:
                        raise ValueError("campaign prerequisite input changed after trigger")

            while True:
                try:
                    trigger = validate_evaluation_trigger(evaluation_directory)
                    break
                except ValueError as error:
                    prerequisite = _load_json(
                        evaluation_directory / "supervisor-status.json",
                        "evaluation supervisor status",
                    )
                    if prerequisite.get("phase") in {"failed", "complete"} or not wait_for_evaluation:
                        raise
                    _status(status_path, "waiting-for-evaluation", detail=str(error))
                    time.sleep(poll_seconds)
            deadline = time.monotonic() + 300.0
            while _evaluation_processes(evaluation_directory):
                if time.monotonic() >= deadline:
                    raise ValueError("completed evaluation still has live producer processes")
                time.sleep(min(poll_seconds, 10.0))
            producer_guard()
            health = validate_host_health(output, skip_power=skip_power_check)
            roots_tsv = (canonical_campaign / "round-2/teacher-input.tsv").resolve()
            roots_manifest = (canonical_campaign / "round-2/replay-roots.json").resolve()
            if not roots_tsv.is_file() or not roots_manifest.is_file():
                raise ValueError("canonical Round-2 replay roots are missing")
            canonical_splits = _canonical_split_manifests(canonical_campaign)
            anchor_train = canonical_splits["train"]
            anchor_validation = canonical_splits["validation"]
            canonical_prior_manifests = tuple(
                canonical_splits[split][round_index]
                for round_index in range(3)
                for split in ("train", "validation", "test")
            )
            canonical_record = _validate_canonical_campaign_inputs(
                canonical_campaign,
                roots_tsv=roots_tsv,
                roots_manifest=roots_manifest,
                anchor_train=anchor_train,
                anchor_validation=anchor_validation,
            )
            opening_exclusions = _evaluation_opening_banks(evaluation_directory)
            manager = GuardedStageManager(
                output=output, campaign_id=AUTO_CAMPAIGN_ID,
                round_index=-1, resume=resume, environment=environment_identity(),
                producer_guard=producer_guard,
            )
            workflow_source = pathlib.Path(__file__).resolve()
            trigger_path = output / "evaluation-trigger.json"

            def record_trigger() -> dict:
                current = validate_evaluation_trigger(evaluation_directory)
                record = {
                    **current, "repository": repository_record,
                    "executables": executable_records, "health": health,
                    "release_build": artifact_snapshot(build_manifest),
                    "canonical_ancestry": canonical_record,
                    "canonical_roots": artifact_snapshot(roots_manifest),
                    "canonical_teacher_tsv": artifact_snapshot(roots_tsv),
                    "anchor_train": [artifact_snapshot(path) for path in anchor_train],
                    "anchor_validation": [
                        artifact_snapshot(path) for path in anchor_validation
                    ],
                    "canonical_prior_manifests": [
                        artifact_snapshot(path) for path in canonical_prior_manifests
                    ],
                }
                _atomic_json(trigger_path, record)
                return record

            def validate_trigger_result(result: dict) -> None:
                producer_guard()
                if result.get("evaluation_summary") != trigger["evaluation_summary"]:
                    raise ValueError("evaluation trigger result is stale")
                current_canonical = _validate_canonical_campaign_inputs(
                    canonical_campaign,
                    roots_tsv=roots_tsv,
                    roots_manifest=roots_manifest,
                    anchor_train=anchor_train,
                    anchor_validation=anchor_validation,
                )
                if result.get("canonical_ancestry") != current_canonical:
                    raise ValueError("canonical trigger ancestry is stale")

            trigger_inputs = {
                "run_manifest": evaluation_directory / "run-manifest.json",
                "status": evaluation_directory / "supervisor-status.json",
                "summary": evaluation_directory / "final-summary.json",
                "latency": evaluation_directory / "latency-audit.json",
                "release_build": build_manifest,
                "roots": roots_manifest, "roots_tsv": roots_tsv,
                **{
                    f"canonical_workflow_{index}":
                        canonical_campaign / f"round-{index}/workflow.json"
                    for index in range(3)
                },
                **{
                    f"anchor_train_{index}": path
                    for index, path in enumerate(anchor_train)
                },
                **{
                    f"anchor_validation_{index}": path
                    for index, path in enumerate(anchor_validation)
                },
                **{
                    f"canonical_prior_{index}": path
                    for index, path in enumerate(canonical_prior_manifests)
                },
            }
            evaluation_manifest = _load_json(
                evaluation_directory / "run-manifest.json", "evaluation manifest"
            )
            for index, raw_path in enumerate(sorted(evaluation_manifest.get("inputs", {}))):
                trigger_inputs[f"evaluation_input_{index:03d}"] = pathlib.Path(raw_path)
            evaluation_summary = _load_json(
                evaluation_directory / "final-summary.json", "evaluation summary"
            )
            for index, binding in enumerate(evaluation_summary.get("reports", [])):
                job_id = binding["job_id"]
                phase = "step1" if job_id.startswith("step1-") else "step2"
                trigger_inputs[f"evaluation_report_{index:03d}"] = (
                    evaluation_directory / "shards" / phase / f"{job_id}.json"
                )
                trigger_inputs[f"evaluation_receipt_{index:03d}"] = (
                    evaluation_directory / "receipts" / f"{job_id}.json"
                )
            for index, path in enumerate(opening_exclusions):
                trigger_inputs[f"opening_exclusion_{index}"] = path
            for index, manifest_path in enumerate(canonical_prior_manifests):
                manifest = _load_json(manifest_path, "canonical frozen shard")
                trigger_inputs[f"canonical_npz_{index}"] = (
                    manifest_path.parent / str(manifest.get("npz", ""))
                )
            frozen_paths = {path.resolve() for path in trigger_inputs.values()}
            frozen_paths.update(path.resolve() for path in opening_exclusions)
            frozen_input_records.update({
                str(path): artifact_snapshot(path) for path in sorted(frozen_paths)
            })
            manager.execute(
                ordinal=0, name="evaluation-trigger",
                configuration={"expected_commit": expected_commit, "minimum_free_bytes": MINIMUM_FREE_BYTES},
                producers={"workflow": workflow_source,
                           **{name: pathlib.Path(record["path"])
                              for name, record in executables.snapshots().items()}},
                inputs=trigger_inputs, outputs={"trigger": trigger_path},
                action=record_trigger,
                validator=validate_trigger_result,
            )
            selection_path = output / "incumbent-selection.json"
            incumbent_snapshot = output / "inputs/incumbent.runtime"
            runner_snapshot = output / "inputs/runner-up.runtime"
            planned = select_incumbent(
                evaluation_directory / "final-summary.json", canonical_campaign
            )
            original_incumbent = pathlib.Path(planned["incumbent"]["runtime"])
            original_runner = pathlib.Path(planned["runner_up"]["runtime"])

            def freeze_incumbent() -> dict:
                selected = select_incumbent(
                    evaluation_directory / "final-summary.json", canonical_campaign
                )
                if selected != planned:
                    raise ValueError("incumbent ranking changed during selection")
                _copy_atomic(original_incumbent, incumbent_snapshot)
                _copy_atomic(original_runner, runner_snapshot)
                selected["incumbent_snapshot"] = artifact_snapshot(incumbent_snapshot)
                selected["runner_up_snapshot"] = artifact_snapshot(runner_snapshot)
                _atomic_json(selection_path, selected)
                return selected

            manager.execute(
                ordinal=1, name="incumbent-selection",
                configuration={"policy": planned["ranking_policy"]},
                producers={"workflow": workflow_source},
                inputs={
                    "evaluation": evaluation_directory / "final-summary.json",
                    "incumbent": original_incumbent, "runner_up": original_runner,
                    **{f"round_{index}": canonical_campaign / f"round-{index}/workflow.json"
                       for index in range(3)},
                },
                outputs={"selection": selection_path, "incumbent": incumbent_snapshot,
                         "runner_up": runner_snapshot}, action=freeze_incumbent,
            )
            for path in (
                selection_path, incumbent_snapshot, runner_snapshot,
                manager.receipt_path(1, "incumbent-selection"),
            ):
                frozen_input_records[str(path.resolve())] = artifact_snapshot(path)
            producer_guard()
            _status(status_path, "running-pilot", incumbent_round=planned["incumbent"]["round"])
            pilot = run_phase(
                spec=PILOT_SPEC, output=output / "pilot", resume=resume,
                roots_tsv=roots_tsv, roots_manifest=roots_manifest,
                actor=incumbent_snapshot, diversity=runner_snapshot,
                executables=executables, anchor_train_manifests=anchor_train,
                anchor_validation_manifests=anchor_validation,
                canonical_prior_manifests=canonical_prior_manifests,
                opening_exclusions=opening_exclusions,
                producer_guard=producer_guard,
                build_manifest=build_manifest,
                source_identities=build_record["source_identities"],
            )
            if not pilot["decision"].get("eligible_for_full"):
                summary = {
                    "schema": CAMPAIGN_SUMMARY_SCHEMA,
                    "terminal": "pilot-rejected", "canonical_promotion_eligible": False,
                    "incumbent_selection": artifact_snapshot(selection_path), "pilot": pilot,
                    "full": None, "publication": None,
                }
                summary_path = output / "final-summary.json"
                _atomic_json(summary_path, summary)
                _status(status_path, "pilot-rejected", summary=str(summary_path),
                        summary_sha256=sha256(summary_path))
                return summary
            pilot_runtime = pathlib.Path(pilot["search_runtime"])
            pilot_search_manifests = [
                pathlib.Path(path) for path in pilot["search_new_manifests"]
            ]
            pilot_rank4_manifests = [
                pathlib.Path(path) for path in pilot["rank4_new_manifests"]
            ]
            pilot_guard_paths = {
                pilot_runtime,
                pathlib.Path(pilot["search_manifest"]),
                pathlib.Path(pilot["decision_path"]),
                pathlib.Path(pilot["opening_bank"]),
                *pilot_search_manifests,
                *pilot_rank4_manifests,
            }
            for manifest_path in (*pilot_search_manifests, *pilot_rank4_manifests):
                manifest = _load_json(manifest_path, "pilot frozen shard")
                pilot_guard_paths.add(
                    manifest_path.parent / str(manifest.get("npz", ""))
                )
            frozen_input_records.update({
                str(path.resolve()): artifact_snapshot(path.resolve())
                for path in pilot_guard_paths
            })
            producer_guard()
            full_health = validate_host_health(
                output, skip_power=skip_power_check
            )
            _status(status_path, "running-full", pilot_runtime_sha256=sha256(pilot_runtime))
            full = run_phase(
                spec=FULL_SPEC, output=output / "full", resume=resume,
                roots_tsv=roots_tsv, roots_manifest=roots_manifest,
                actor=pilot_runtime, diversity=incumbent_snapshot,
                executables=executables, anchor_train_manifests=anchor_train,
                anchor_validation_manifests=anchor_validation,
                canonical_prior_manifests=canonical_prior_manifests,
                opening_exclusions=[*opening_exclusions, pathlib.Path(pilot["opening_bank"])],
                prior_search_manifests=pilot_search_manifests,
                prior_rank4_manifests=pilot_rank4_manifests,
                producer_guard=producer_guard,
                build_manifest=build_manifest,
                source_identities=build_record["source_identities"],
            )
            full["launch_health"] = full_health
            if not full["decision"].get("eligible_for_local_publication"):
                summary = {
                    "schema": CAMPAIGN_SUMMARY_SCHEMA,
                    "terminal": "full-rejected", "canonical_promotion_eligible": False,
                    "incumbent_selection": artifact_snapshot(selection_path), "pilot": pilot,
                    "full": full, "retained_research_incumbent": artifact_snapshot(pilot_runtime),
                    "publication": None,
                }
                summary_path = output / "final-summary.json"
                _atomic_json(summary_path, summary)
                _status(status_path, "full-rejected", summary=str(summary_path),
                        summary_sha256=sha256(summary_path))
                return summary
            publication_directory = output / "promoted" / FULL_CAMPAIGN_ID
            published_runtime = publication_directory / "jacek_replay_bfm.runtime"
            published_manifest = publication_directory / "jacek_replay_bfm.runtime.json"
            publication_receipt = publication_directory / "publication.json"
            evidence_directory = publication_directory / "gate-evidence"
            full_runtime = pathlib.Path(full["search_runtime"])
            full_manifest = pathlib.Path(full["search_manifest"])
            evidence_sources = {
                "pilot-decision.json": output / "pilot/receipts/21-decision.json",
                "full-game-gates.json": output / "full/receipts/19-game-gates.json",
                "full-latency-audit.json": output / "full/receipts/20-latency-audit.json",
                "full-decision.json": output / "full/receipts/21-decision.json",
            }
            panel_receipts = sorted(
                (output / "full/receipts/19-game-gates-panels").rglob("*.json")
            )
            expected_panel_receipts = 4 * (FULL_SPEC.pairs // 5)
            if len(panel_receipts) != expected_panel_receipts:
                raise ValueError(
                    "full gate publication requires every sharded panel receipt"
                )
            for index, path in enumerate(panel_receipts):
                evidence_sources[f"full-panel-{index}.json"] = path
            publication_guard_paths = {
                full_runtime,
                full_manifest,
                pathlib.Path(full["decision_path"]),
                *evidence_sources.values(),
            }
            full_decision = _load_json(
                pathlib.Path(full["decision_path"]), "full publication decision"
            )
            for record in full_decision.get("reports", {}).values():
                if isinstance(record, dict) and isinstance(record.get("path"), str):
                    publication_guard_paths.add(pathlib.Path(record["path"]))
            frozen_input_records.update({
                str(path.resolve()): artifact_snapshot(path.resolve())
                for path in publication_guard_paths
            })
            producer_guard()
            published_evidence = {
                name: evidence_directory / name for name in evidence_sources
            }

            def publish() -> dict:
                publication_directory.mkdir(parents=True, exist_ok=True)
                _copy_atomic(full_runtime, published_runtime)
                _copy_atomic(full_manifest, published_manifest)
                for name, source in evidence_sources.items():
                    _copy_atomic(source, published_evidence[name])
                record = {
                    "schema": "papersoccer.jacek-selfsearch-local-publication.v1",
                    "classification": "local-noncanonical-research-model",
                    "canonical_promotion_eligible": False,
                    "runtime": artifact_snapshot(published_runtime),
                    "manifest": artifact_snapshot(published_manifest),
                    "source_runtime": artifact_snapshot(full_runtime),
                    "source_manifest": artifact_snapshot(full_manifest),
                    "pilot_decision": artifact_snapshot(pathlib.Path(pilot["decision_path"])),
                    "full_decision": artifact_snapshot(pathlib.Path(full["decision_path"])),
                    "gate_evidence": {
                        name: artifact_snapshot(path) for name, path in published_evidence.items()
                    },
                    "external_upload": False, "rank4_replaced": False,
                }
                _atomic_json(publication_receipt, record)
                return record

            publication = manager.execute(
                ordinal=2, name="local-publication",
                configuration={"external_upload": False, "replace_rank4": False},
                producers={"workflow": workflow_source},
                inputs={"runtime": full_runtime, "manifest": full_manifest,
                        "pilot_decision": pathlib.Path(pilot["decision_path"]),
                        "full_decision": pathlib.Path(full["decision_path"]),
                        **{f"evidence_{index}": path
                           for index, path in enumerate(evidence_sources.values())}},
                outputs={"runtime": published_runtime, "manifest": published_manifest,
                         "receipt": publication_receipt,
                         **{f"evidence_{index}": path
                            for index, path in enumerate(published_evidence.values())}},
                action=publish,
            )
            summary = {
                "schema": CAMPAIGN_SUMMARY_SCHEMA,
                "terminal": "published", "canonical_promotion_eligible": False,
                "incumbent_selection": artifact_snapshot(selection_path), "pilot": pilot,
                "full": full, "publication": publication,
            }
            summary_path = output / "final-summary.json"
            _atomic_json(summary_path, summary)
            _status(status_path, "published", summary=str(summary_path),
                    summary_sha256=sha256(summary_path))
            return summary
        except Exception as error:
            _status(status_path, "failed", error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            _CAMPAIGN_LOCK_FD = None


def write_pair(payload: bytes, manifest: dict, output: pathlib.Path, sidecar: pathlib.Path) -> None:
    if hashlib.sha256(payload).hexdigest() != manifest.get("output_sha256"):
        raise ValueError("output payload does not match its manifest")
    _atomic_bytes(output, payload)
    _atomic_json(sidecar, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    campaign = subparsers.add_parser(
        "run", help="wait for the frozen evaluation and run pilot/full campaign"
    )
    campaign.add_argument("--repository", type=pathlib.Path, required=True)
    campaign.add_argument("--expected-commit", required=True)
    campaign.add_argument("--evaluation-directory", type=pathlib.Path, required=True)
    campaign.add_argument("--canonical-campaign", type=pathlib.Path, required=True)
    campaign.add_argument("--output-directory", type=pathlib.Path, required=True)
    campaign.add_argument("--continuation-generator", type=pathlib.Path, required=True)
    campaign.add_argument("--search-teacher", type=pathlib.Path, required=True)
    campaign.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    campaign.add_argument("--comparison", type=pathlib.Path, required=True)
    campaign.add_argument("--pack-tool", type=pathlib.Path, required=True)
    campaign.add_argument("--trainer", type=pathlib.Path, required=True)
    campaign.add_argument("--build-manifest", type=pathlib.Path, required=True)
    campaign.add_argument("--resume", action="store_true")
    campaign.add_argument("--wait-for-evaluation", action="store_true")
    campaign.add_argument("--poll-seconds", type=float, default=30.0)
    campaign.add_argument(
        "--skip-power-check", action="store_true",
        help="development/smoke only; canonical local launch must not set this",
    )
    build = subparsers.add_parser(
        "write-build-manifest", help="freeze one clean Release producer build"
    )
    build.add_argument("--repository", type=pathlib.Path, required=True)
    build.add_argument("--expected-commit", required=True)
    build.add_argument("--continuation-generator", type=pathlib.Path, required=True)
    build.add_argument("--search-teacher", type=pathlib.Path, required=True)
    build.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    build.add_argument("--comparison", type=pathlib.Path, required=True)
    build.add_argument("--pack-tool", type=pathlib.Path, required=True)
    build.add_argument("--trainer", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)
    incumbent = subparsers.add_parser("select-incumbent")
    incumbent.add_argument("--evaluation", type=pathlib.Path, required=True)
    incumbent.add_argument("--canonical-campaign", type=pathlib.Path, required=True)
    incumbent.add_argument("--output", type=pathlib.Path, required=True)
    game_plan = subparsers.add_parser("game-plan")
    game_plan.add_argument("--profile", choices=("pilot", "full"), required=True)
    game_plan.add_argument("--seed", type=int, required=True)
    game_plan.add_argument("--output", type=pathlib.Path, required=True)
    game_plan.add_argument("--tsv-output", type=pathlib.Path, required=True)
    positions = subparsers.add_parser("freeze-positions")
    positions.add_argument("--campaign-id", required=True)
    positions.add_argument("--games", type=pathlib.Path, required=True)
    positions.add_argument("--game-manifest", type=pathlib.Path, required=True)
    positions.add_argument("--roots", type=pathlib.Path, required=True)
    positions.add_argument("--maximum", type=int, required=True)
    positions.add_argument("--output", type=pathlib.Path, required=True)
    positions.add_argument("--manifest", type=pathlib.Path, required=True)
    hard = subparsers.add_parser("select-hard")
    hard.add_argument("--positions", type=pathlib.Path, required=True)
    hard.add_argument("--search-labels", type=pathlib.Path, required=True)
    hard.add_argument("--rank4-labels", type=pathlib.Path, required=True)
    hard.add_argument("--output", type=pathlib.Path, required=True)
    hard.add_argument("--manifest", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        if (
            not math.isfinite(arguments.poll_seconds)
            or arguments.poll_seconds <= 0.0
            or len(arguments.expected_commit) != 40
            or any(character not in "0123456789abcdef" for character in arguments.expected_commit)
        ):
            parser.error("run requires a positive poll interval and a lowercase commit hash")
        run_campaign(
            repository=arguments.repository,
            expected_commit=arguments.expected_commit,
            evaluation_directory=arguments.evaluation_directory,
            canonical_campaign=arguments.canonical_campaign,
            output=arguments.output_directory,
            executables=CampaignExecutables(
                continuation_generator=arguments.continuation_generator,
                search_teacher=arguments.search_teacher,
                rank4_teacher=arguments.rank4_teacher,
                comparison=arguments.comparison,
                pack_tool=arguments.pack_tool,
                trainer=arguments.trainer,
            ),
            build_manifest=arguments.build_manifest,
            resume=arguments.resume,
            wait_for_evaluation=arguments.wait_for_evaluation,
            poll_seconds=arguments.poll_seconds,
            skip_power_check=arguments.skip_power_check,
        )
    elif arguments.command == "write-build-manifest":
        if (
            len(arguments.expected_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.expected_commit)
        ):
            parser.error("write-build-manifest requires a lowercase commit hash")
        write_build_manifest(
            repository=arguments.repository,
            expected_commit=arguments.expected_commit,
            executables=CampaignExecutables(
                continuation_generator=arguments.continuation_generator,
                search_teacher=arguments.search_teacher,
                rank4_teacher=arguments.rank4_teacher,
                comparison=arguments.comparison,
                pack_tool=arguments.pack_tool,
                trainer=arguments.trainer,
            ),
            output=arguments.output,
        )
    elif arguments.command == "select-incumbent":
        _atomic_json(
            arguments.output,
            select_incumbent(arguments.evaluation, arguments.canonical_campaign),
        )
    elif arguments.command == "game-plan":
        campaign_id = PILOT_CAMPAIGN_ID if arguments.profile == "pilot" else FULL_CAMPAIGN_ID
        quotas = PILOT_QUOTAS if arguments.profile == "pilot" else FULL_QUOTAS
        plan = make_game_plan(campaign_id=campaign_id, seed=arguments.seed, quotas=quotas)
        _atomic_json(arguments.output, plan)
        _atomic_bytes(arguments.tsv_output, render_game_plan_tsv(plan))
    elif arguments.command == "freeze-positions":
        payload, manifest = freeze_positions(
            campaign_id=arguments.campaign_id,
            games_tsv=arguments.games,
            games_manifest=arguments.game_manifest,
            roots_manifest=arguments.roots,
            maximum_per_game=arguments.maximum,
        )
        write_pair(payload, manifest, arguments.output, arguments.manifest)
    elif arguments.command == "select-hard":
        payload, manifest = select_hard_positions(
            positions_tsv=arguments.positions,
            search_labels=arguments.search_labels,
            rank4_labels=arguments.rank4_labels,
        )
        write_pair(payload, manifest, arguments.output, arguments.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
