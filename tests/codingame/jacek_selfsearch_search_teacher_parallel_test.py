#!/usr/bin/env python3
"""Prove fixed-work search labels are byte-identical at 1, 2, and 10 workers."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_selfsearch_workflow as workflow  # noqa: E402


PREFIX = "0/0/3/0/61/0"
WIDENING_POSITION_ID = (
    "position:fc5350b1be0e9887a10b97bd5569a1251a9824cb4a4e9c6d7404471eeb4616e3"
)
WIDENING_PREFIX = (
    "1/2/7/5/207/6/1/45/00/75/03/35/22/445/7/2/177/44/47/2345/7/53/"
    "0/23/0/3/1/4/1/3/17/54/1/36350/00017/25/017/27/035075/5433/00/"
    "35235663357/025766/752530/010/245021/065052507117/723064534/17/"
    "7245201235/05223/63/00117"
)


def run_once(
    *,
    teacher: pathlib.Path,
    model: pathlib.Path,
    positions: pathlib.Path,
    root: pathlib.Path,
    workers: int,
    fixture: str,
    campaign_id: str,
    nodes: int,
) -> bytes:
    output = root / f"{fixture}-workers-{workers}"
    output.mkdir()
    manager = workflow.StageManager(
        output=output,
        campaign_id=campaign_id,
        round_index=0,
        resume=False,
        environment={"fixture": fixture},
    )
    labels = output / "labels.jsonl"
    workflow.run_label_chunks(
        manager=manager,
        stage_ordinal=3,
        stage_name=f"search-{fixture}",
        positions=positions,
        output=labels,
        teacher=teacher,
        schema=workflow.SEARCH_TEACHER_SCHEMA,
        campaign_id=campaign_id,
        nodes=nodes,
        workers=workers,
        source_sha256=workflow._source_closure(
            ROOT, workflow.SEARCH_TEACHER_SOURCE_PATHS
        ),
        model=model,
    )
    return labels.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        positions = root / "positions.tsv"
        positions.write_text(
            "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
            + "".join(
                f"position-{index}\troot-{index}\tgame-{index}\tfixture\ttrain\t0\t0\t{PREFIX}\n"
                for index in range(251)
            ),
            encoding="utf-8",
        )
        outputs = [
            run_once(
                teacher=arguments.search_teacher,
                model=arguments.model,
                positions=positions,
                root=root,
                workers=workers,
                fixture="parallel-fixed-work-v1",
                campaign_id="selfsearch-search-teacher-parallel-smoke",
                nodes=16,
            )
            for workers in (1, 2, 10)
        ]
        if outputs[0] != outputs[1] or outputs[0] != outputs[2]:
            raise RuntimeError("search-teacher output changed with worker count")

        widening_positions = root / "widening-positions.tsv"
        widening_positions.write_text(
            "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
            f"{WIDENING_POSITION_ID}\troot-regression\tgame-regression\t"
            f"selfsearch-pilot-20260825-v1\ttrain\t0\t1\t{WIDENING_PREFIX}\n",
            encoding="utf-8",
        )
        widening_outputs = [
            run_once(
                teacher=arguments.search_teacher,
                model=arguments.model,
                positions=widening_positions,
                root=root,
                workers=workers,
                fixture="parallel-progressive-widening-v1",
                campaign_id="selfsearch-pilot-20260825-v1",
                nodes=64_000,
            )
            for workers in (1, 2, 10)
        ]
        if not all(
            output == widening_outputs[0] for output in widening_outputs[1:]
        ):
            raise RuntimeError("progressive widening changed with worker count")
        widening = json.loads(widening_outputs[0])
        if (
            widening["position_id"] != WIDENING_POSITION_ID
            or widening["search_stats"]["progressive_widenings"] <= 0
            or widening["search_stats"]["termination_reason"]
            not in {"root-solved", "fixed-work-cap"}
        ):
            raise RuntimeError("progressive-widening regression was not exercised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
