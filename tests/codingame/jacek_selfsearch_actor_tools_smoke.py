#!/usr/bin/env python3
"""Lightweight black-box contracts for the self-search actor producers."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile


HEADER = (
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\t"
    "winner\tmover\tprefix\n"
)
SHORT_WIN = "0/0/3/0/61/0/07"
ACTOR_MODES = (
    "incumbent-selfplay",
    "incumbent-p1-vs-rank4",
    "incumbent-p2-vs-rank4",
    "incumbent-p1-vs-jacek-nn",
    "incumbent-p2-vs-jacek-nn",
    "incumbent-p1-vs-runner-up",
    "incumbent-p2-vs-runner-up",
    "student-selfplay",
    "student-p1-vs-rank4",
    "student-p2-vs-rank4",
    "student-p1-vs-jacek-nn",
    "student-p2-vs-jacek-nn",
    "student-p1-vs-prior-incumbent",
    "student-p2-vs-prior-incumbent",
)
MASK64 = (1 << 64) - 1


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def splitmix_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (value ^ (value >> 31)) & MASK64


def near_goal_seeds(count: int) -> list[int]:
    """Choose attempt-zero seeds with prefix 6 and no random first turn."""
    result = []
    seed = 0
    while len(result) < count:
        state, prefix_draw = splitmix_next(seed)
        _, exploration_draw = splitmix_next(state)
        if prefix_draw % 7 == 6 and exploration_draw % 100 >= 15:
            result.append(seed)
        seed += 1
    return result


def run_continuations(
    executable: pathlib.Path,
    model: pathlib.Path,
    directory: pathlib.Path,
    suffix: str,
) -> tuple[bytes, dict]:
    roots = directory / "roots.tsv"
    roots.write_text(
        "group_id\tsource\twinner\ttranscript\n"
        f"root:near-goal\tfixture\t0\t{SHORT_WIN}\n",
        encoding="utf-8",
    )
    plan = directory / "plan.tsv"
    seeds = near_goal_seeds(len(ACTOR_MODES))
    plan.write_text(
        "game_ordinal\tactor_mode\tbase_seed\n"
        + "".join(
            f"{ordinal}\t{mode}\t{seed}\n"
            for ordinal, (mode, seed) in enumerate(zip(ACTOR_MODES, seeds, strict=True))
        ),
        encoding="utf-8",
    )
    output = directory / f"games-{suffix}.tsv"
    manifest = directory / f"games-{suffix}.json"
    command = (
        str(executable),
        "--input", str(roots),
        "--output", str(output),
        "--manifest", str(manifest),
        "--model", str(model),
        "--runner-up-model", str(model),
        "--selfsearch-plan", str(plan),
        "--campaign-id", "selfsearch-actor-smoke",
        "--games", str(len(ACTOR_MODES)),
        "--candidate-tree-nodes", "16",
        "--actor-nodes", "16",
        "--jacek-nn-nodes", "16",
        "--candidate-exploration", "0.5",
        "--candidate-fpu", "0.5",
        "--max-turns", "64",
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"continuation smoke failed: {completed.stderr}")
    payload = output.read_bytes()
    report = json.loads(manifest.read_bytes())
    if (
        report.get("schema") != "papersoccer.jacek-selfsearch-games.v1"
        or report.get("requested_games") != len(ACTOR_MODES)
        or report.get("successful_games") != len(ACTOR_MODES)
        or report.get("bindings", {}).get("roots_sha256") != sha256(roots)
        or report.get("bindings", {}).get("plan_sha256") != sha256(plan)
        or report.get("bindings", {}).get("output_sha256")
        != hashlib.sha256(payload).hexdigest()
        or report.get("bindings", {}).get("incumbent_model_sha256") != sha256(model)
        or report.get("bindings", {}).get("runner_up_model_sha256") != sha256(model)
    ):
        raise RuntimeError("continuation smoke manifest bindings are invalid")
    rows = report.get("rows")
    if (
        not isinstance(rows, list)
        or [row.get("row_ordinal") for row in rows] != list(range(len(ACTOR_MODES)))
        or [row.get("game_ordinal") for row in rows] != list(range(len(ACTOR_MODES)))
        or [row.get("actor_mode") for row in rows] != list(ACTOR_MODES)
        or [row.get("base_seed") for row in rows] != seeds
        or any(row.get("prefix_turns") != 6 for row in rows)
        or any(row.get("attempt_ordinal") != 0 for row in rows)
    ):
        raise RuntimeError("continuation smoke lineage is invalid")
    for field in (
        "producer_source_sha256",
        "rank4_actor_source_sha256",
        "jacek_nn_actor_source_sha256",
    ):
        value = report.get("configuration", {}).get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"continuation smoke lacks {field}")
    lines = payload.decode("utf-8").splitlines()
    if lines[0] != "group_id\tsource\twinner\ttranscript" or len(lines) != 15:
        raise RuntimeError("continuation smoke TSV is invalid")
    return payload, report


def run_rank4_teacher(executable: pathlib.Path) -> None:
    row = "p0\troot:near-goal\tgame:0\tpilot\tvalidation\t0\t0\t0/0/3/0/61/0\n"
    command = (
        str(executable), "--campaign-id", "selfsearch-actor-smoke",
        "--nodes", "64", "--time-ms", "60000",
    )
    first = subprocess.run(
        command, input=HEADER + row, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        command, input=HEADER + row, text=True, capture_output=True, check=False
    )
    if first.returncode != 0 or first.stdout != second.stdout:
        raise RuntimeError(f"Rank-4 teacher is not deterministic: {first.stderr}")
    label = json.loads(first.stdout)
    source_hash = label.get("teacher", {}).get("source_sha256")
    if (
        label.get("schema") != "papersoccer.jacek-replay-teacher.v1"
        or label.get("position_id") != "p0"
        or label.get("mover") != 0
        or not label.get("root_solved")
        or label.get("proven_winner") != 0
        or label.get("search_config", {}).get("max_nodes") != 64
        or label.get("search_stats", {}).get("deadline_reached") is not False
        or not isinstance(source_hash, str)
        or len(source_hash) != 64
    ):
        raise RuntimeError("Rank-4 teacher label contract is invalid")

    duplicate = subprocess.run(
        command, input=HEADER + row + row, text=True, capture_output=True, check=False
    )
    if duplicate.returncode == 0 or duplicate.stdout:
        raise RuntimeError("Rank-4 teacher did not fail closed on duplicate IDs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuations", type=pathlib.Path, required=True)
    parser.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = pathlib.Path(raw_directory)
        first_payload, first_report = run_continuations(
            arguments.continuations, arguments.model, directory, "one"
        )
        second_payload, second_report = run_continuations(
            arguments.continuations, arguments.model, directory, "two"
        )
        if first_payload != second_payload or first_report != second_report:
            raise RuntimeError("self-search continuation output is not deterministic")
        run_rank4_teacher(arguments.rank4_teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
