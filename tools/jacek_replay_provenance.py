#!/usr/bin/env python3
"""Shared source and executable provenance for Jacek replay game gates."""

from __future__ import annotations

import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]

CANDIDATE_SOURCE_PATHS = (
    "include/papersoccer/bot.hpp",
    "src/bots/bot.cpp",
    "src/bots/jacek_replay_bfm/features.cpp",
    "src/bots/jacek_replay_bfm/model.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp",
    "src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp",
)

COMPARISON_SOURCE_PATHS = (
    "tools/jacek_replay_bfm_comparison.cpp",
    "tools/jacek_replay_bfm_gate_internal.hpp",
)

CONTROL_SOURCE_PATHS = {
    "rank4_control_sha256": (
        "submissions/codingame/bots/rank_4/submission.cpp"
    ),
    "rank4_engine_sha256": "submissions/codingame/bots/rank_4/bot.cpp",
    "neural_puct_control_sha256": (
        "submissions/codingame/bots/neural_puct/submission.cpp"
    ),
    "neural_puct_engine_sha256": (
        "submissions/codingame/bots/neural_puct/bot.cpp"
    ),
    "rank4_adapter_sha256": "tools/jacek_replay_bfm_rank4_control.cpp",
    "neural_puct_adapter_sha256": (
        "tools/jacek_replay_bfm_neural_puct_control.cpp"
    ),
}

SHARED_CORE_PATHS = (
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
    "src/bots/mcts_internal.hpp",
)


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure_sha256(paths: tuple[str, ...]) -> str:
    material = "".join(
        f"{path}:{file_sha256(ROOT / path)}\n" for path in paths
    )
    return hashlib.sha256(material.encode()).hexdigest()


def candidate_source_sha256() -> str:
    return source_closure_sha256(CANDIDATE_SOURCE_PATHS)


def comparison_source_sha256() -> str:
    return source_closure_sha256(COMPARISON_SOURCE_PATHS)


def shared_core_sha256() -> str:
    return source_closure_sha256(SHARED_CORE_PATHS)


def control_source_sha256() -> dict[str, str]:
    return {
        field: file_sha256(ROOT / path)
        for field, path in CONTROL_SOURCE_PATHS.items()
    }
