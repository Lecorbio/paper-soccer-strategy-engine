#!/usr/bin/env python3
"""Prove Rank-4 fixed-work labels are identical at 1, 2, and 10 workers."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_selfsearch_workflow as workflow  # noqa: E402


CAMPAIGN_ID = "selfsearch-rank4-teacher-parallel-smoke"
SHORT_WIN_PREFIX = "0/0/3/0/61/0"
REGRESSION_POSITION_ID = (
    "position:c1eb882b3646bfb79053eb6379dacd5100567eed0459b06dc181d417c0d92748"
)
REGRESSION_PREFIX = (
    "4/4/7/23/1/2/4/163635/27/457/41/24765/67/2/5050/21/47/23/21/"
    "145075/46167/14/17/2505/47/23224/630750547/0/721/3606/1/3606/71/"
    "4/72714/43/0/5443522301/77/0/632/3/1/3/5741/2255/"
    "461274606160505711/432/3607"
)


def rank4_source_sha256() -> str:
    return workflow._source_closure(
        ROOT,
        (
            *workflow.RANK4_ACTOR_SOURCE_PATHS,
            "tools/jacek_replay_rank4_position_teacher.cpp",
        ),
    )


def run_once(
    *, teacher: pathlib.Path, positions: pathlib.Path,
    root: pathlib.Path, workers: int, source_sha256: str,
) -> bytes:
    output = root / f"workers-{workers}"
    output.mkdir()
    manager = workflow.StageManager(
        output=output,
        campaign_id=CAMPAIGN_ID,
        round_index=0,
        resume=False,
        environment={"fixture": "rank4-parallel-fixed-work-v1"},
    )
    labels = output / "labels.jsonl"
    result = workflow.run_label_chunks(
        manager=manager,
        stage_ordinal=4,
        stage_name="rank4-parallel-fixed-work",
        positions=positions,
        output=labels,
        teacher=teacher,
        schema=workflow.RANK4_TEACHER_SCHEMA,
        campaign_id=CAMPAIGN_ID,
        nodes=32_000,
        workers=workers,
        source_sha256=source_sha256,
        chunk_games=1,
    )
    if result.get("teacher_rows") != 11 or len(result.get("chunks", [])) != 11:
        raise RuntimeError("Rank-4 parallel regression did not exercise eleven chunks")
    return labels.read_bytes()


def validate_regression(payload: bytes, source_sha256: str) -> None:
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) != 11:
        raise RuntimeError("Rank-4 parallel regression emitted the wrong row count")
    regression = next(
        (row for row in rows if row.get("position_id") == REGRESSION_POSITION_ID),
        None,
    )
    if not isinstance(regression, dict):
        raise RuntimeError("Rank-4 depth-one regression label is missing")
    stats = regression.get("search_stats", {})
    if (
        regression.get("schema") != workflow.RANK4_TEACHER_SCHEMA
        or regression.get("campaign_id") != CAMPAIGN_ID
        or regression.get("completed_depth") != 0
        or regression.get("nodes") != 32_000
        or regression.get("root_solved") is not False
        or regression.get("proven_winner") is not None
        or regression.get("teacher")
        != {"kind": "rank4-fixed-work", "source_sha256": source_sha256}
        or regression.get("search_config", {}).get("max_nodes") != 32_000
        or stats.get("attempted_depth") != 1
        or stats.get("completed_depth") != 0
        or stats.get("nodes") != 32_000
        or not isinstance(stats.get("completed_actions"), int)
        or stats.get("completed_actions", 0) <= 0
        or stats.get("budget_exhausted") is not True
        or stats.get("node_cap_reached") is not True
        or stats.get("depth_cap_reached") is not False
        or stats.get("deadline_reached") is not False
        or stats.get("termination_reason") != "fixed-work-cap"
    ):
        raise RuntimeError("Rank-4 depth-one fixed-work label contract is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    source_sha256 = rank4_source_sha256()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        positions = root / "positions.tsv"
        positions.write_text(
            "position_id\troot_group_id\tgroup_id\tsource\tsplit\t"
            "winner\tmover\tprefix\n"
            f"{REGRESSION_POSITION_ID}\town-live:898437522\t"
            "selfsearch-game:cef40925f005f13ffa5e0c40e69f910e"
            "e025a6a6f7a1ecbcf5d44c096a7e0015\t"
            f"selfsearch-pilot-20260825-v3\ttrain\t0\t1\t{REGRESSION_PREFIX}\n"
            + "".join(
                f"position:rank4-parity-{index:03d}\troot:{index:03d}\t"
                f"game:{index:03d}\tfixture\ttrain\t0\t0\t{SHORT_WIN_PREFIX}\n"
                for index in range(10)
            ),
            encoding="utf-8",
        )
        outputs = [
            run_once(
                teacher=arguments.rank4_teacher,
                positions=positions,
                root=root,
                workers=workers,
                source_sha256=source_sha256,
            )
            for workers in (1, 2, 10)
        ]
        if outputs[0] != outputs[1] or outputs[0] != outputs[2]:
            raise RuntimeError("Rank-4 labels changed with worker count")
        validate_regression(outputs[0], source_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
