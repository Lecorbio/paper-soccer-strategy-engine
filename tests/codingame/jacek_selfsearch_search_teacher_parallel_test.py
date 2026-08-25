#!/usr/bin/env python3
"""Prove fixed-work search labels are byte-identical at 1, 2, and 10 workers."""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_selfsearch_workflow as workflow  # noqa: E402


PREFIX = "0/0/3/0/61/0"


def run_once(
    *, teacher: pathlib.Path, model: pathlib.Path,
    positions: pathlib.Path, root: pathlib.Path, workers: int,
) -> bytes:
    output = root / f"workers-{workers}"
    output.mkdir()
    manager = workflow.StageManager(
        output=output,
        campaign_id="selfsearch-search-teacher-parallel-smoke",
        round_index=0,
        resume=False,
        environment={"fixture": "parallel-fixed-work-v1"},
    )
    labels = output / "labels.jsonl"
    workflow.run_label_chunks(
        manager=manager,
        stage_ordinal=3,
        stage_name="search-parallel-smoke",
        positions=positions,
        output=labels,
        teacher=teacher,
        schema=workflow.SEARCH_TEACHER_SCHEMA,
        campaign_id="selfsearch-search-teacher-parallel-smoke",
        nodes=16,
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
            )
            for workers in (1, 2, 10)
        ]
        if outputs[0] != outputs[1] or outputs[0] != outputs[2]:
            raise RuntimeError("search-teacher output changed with worker count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
