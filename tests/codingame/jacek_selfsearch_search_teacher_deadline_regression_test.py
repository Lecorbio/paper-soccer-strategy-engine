#!/usr/bin/env python3
"""Regress the four production v4 positions that hit wall-clock deadlines."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_selfsearch_workflow as workflow  # noqa: E402


HEADER = (
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
)
SCHEMA = "papersoccer.jacek-replay-search-teacher.v4"
CAMPAIGN_ID = "selfsearch-pilot-20260825-v4"
TREE_NODES = 500_000
WORKERS = 10
STAGE_ORDINAL = 6
STAGE_NAME = "search-deadline-regression-v4"

# These are the exact rows that prevented four v4 deep-label chunks from
# publishing receipts.  Keep the complete production lineage and prefix here:
# shortening a prefix would exercise a different game state and seed binding.
REGRESSIONS = (
    {
        "position_id": (
            "position:1a4bdf2367a54b1d160aa8830a7d0c9d2db9982675f818089dd815b42f0cf1ef"
        ),
        "seed": 6683591067618027234,
        "row": (
            "position:1a4bdf2367a54b1d160aa8830a7d0c9d2db9982675f818089dd815b42f0cf1ef\t"
            "own-live:898437522\t"
            "selfsearch-game:0d0a6af028f88eb03500c5493bff24c7d6b02d286be8e2fd9219ed4832ac7641\t"
            "selfsearch-pilot-20260825-v4\ttrain\t0\t1\t"
            "4/4/7/23/1/2/4/163635/27/457/41/24765/67/2/5050/21/47/23/21/"
            "145075/46167/14/17/2505/47/23224/630750547/0/721/3606/1/3606/"
            "71/4/72714/43/0/5443522301/77/0/632/3/1/3/5741/2255/"
            "461274606160505711/432/22/0/17/035/7/03435250500/16/025357/17/"
            "035/6/1424500366/77"
        ),
    },
    {
        "position_id": (
            "position:3ddfcf4d32d39a6a644148948b4f5b45fe4ae9b7ac117c36120f055bfc08fa79"
        ),
        "seed": 10438668778165952465,
        "row": (
            "position:3ddfcf4d32d39a6a644148948b4f5b45fe4ae9b7ac117c36120f055bfc08fa79\t"
            "own-live:898428147\t"
            "selfsearch-game:0176fd84b06ba291bc4bc97e56fb660a5f23fc457120da0b6e5725c5e6a17cc8\t"
            "selfsearch-pilot-20260825-v4\ttrain\t1\t0\t"
            "4/4/1/4/71/465/7/4/7/53/16/02543/30/3/21/47/22/2577/01/2/16/"
            "36054/547/02525/5214761421/063675/52506/02/457161721/47/471/33/"
            "121/2/7/430256/01/255335/6316501/05247714676/1/2/36017/5/0/22/"
            "3600/74/14657/05/0366/034/3/235070/01/344/11/613636/566720/"
            "1642327576525"
        ),
    },
    {
        "position_id": (
            "position:f065cc8c8fe7e6aa946f10abb4c6016ab31af456fc8d66511c0ad5ba7ce94b02"
        ),
        "seed": 4166782000499300387,
        "row": (
            "position:f065cc8c8fe7e6aa946f10abb4c6016ab31af456fc8d66511c0ad5ba7ce94b02\t"
            "own-live:898428147\t"
            "selfsearch-game:a4ed22753c42b431a9421ed6d84fa0420b90115518e11324be36c3044e0beb1f\t"
            "selfsearch-pilot-20260825-v4\ttrain\t1\t1\t"
            "4/4/1/4/71/465/7/4/7/53/16/02543/30/3/21/47/22/2577/01/2/16/"
            "36054/547/02525/5214761421/063675/52506/02/457161721/47/471/33/"
            "121/2/7/430256/01/255335/6316501/05247714676/1/2/501/21/6/"
            "302565476/4611/211/75/03/64/113574656/1/13474360744746/46110/6/1"
        ),
    },
    {
        "position_id": (
            "position:fe638e49e4c2c09050fe7a74fb2277f22c4cbc68f3e144a8f840c01f98f33bfa"
        ),
        "seed": 2980796312927321205,
        "row": (
            "position:fe638e49e4c2c09050fe7a74fb2277f22c4cbc68f3e144a8f840c01f98f33bfa\t"
            "own-live:898428001\t"
            "selfsearch-game:689c030d32762033359c4d5281053453069d5f8903ac0e96e3c8bd91d7963d91\t"
            "selfsearch-pilot-20260825-v4\ttrain\t1\t1\t"
            "4/4/7/23/1/2/25/03635/27/457/41/24765/67/2/5050/21/47/23/21/"
            "145075/46167/14/17/2505/47/23224/63075056/0/7246171/4/41/"
            "36063522301/77/460/0/3570/27/0/13/5/61/14643/4612/3/0/3/61/61/"
            "2/542/706/47/71/23/47/14550123174/4350/55/0524277"
        ),
    },
)


def expected_prefix(row: str) -> list[dict[str, object]]:
    actions = row.split("\t")[7].split("/")
    return [
        {"player_id": turn % 2, "action": action}
        for turn, action in enumerate(actions)
    ]


def validate_labels(payload: bytes) -> None:
    lines = payload.splitlines()
    if len(lines) != len(REGRESSIONS):
        raise RuntimeError("deadline regression did not produce exactly four labels")
    labels = [json.loads(line) for line in lines]
    for label, regression in zip(labels, REGRESSIONS, strict=True):
        row = regression["row"]
        position_id = regression["position_id"]
        fields = row.split("\t")
        if (
            len(fields) != 8
            or fields[0] != position_id
            or label.get("schema") != SCHEMA
            or label.get("campaign_id") != CAMPAIGN_ID
            or label.get("position_id") != position_id
            or label.get("prefix") != expected_prefix(row)
        ):
            raise RuntimeError("deadline regression label lost its frozen v4 binding")

        configuration = label.get("search_config", {})
        stats = label.get("search_stats", {})
        if (
            configuration.get("seed") != regression["seed"]
            or configuration.get("max_time_ms") != 0
            or configuration.get("max_tree_nodes") != TREE_NODES
            or stats.get("deadline_reached") is not False
            or stats.get("generation_deadline_stops") != 0
            or stats.get("materialization_deadline_stops") != 0
            or stats.get("generation_queue_drops") != 0
            or stats.get("closed_unsolved_nodes") != 0
            or stats.get("closed_unsolved_nonexhaustive_nodes") != 0
        ):
            raise RuntimeError("deadline regression label has premature termination")

        if label.get("root_solved") is True:
            complete = (
                stats.get("termination_reason") == "root-solved"
                and label.get("proven_winner") in {0, 1}
            )
        elif label.get("root_solved") is False:
            complete = (
                stats.get("termination_reason") == "fixed-work-cap"
                and stats.get("tree_cap_reached") is True
                and stats.get("tree_nodes") == TREE_NODES
                and label.get("proven_winner") is None
            )
        else:
            complete = False
        if not complete:
            raise RuntimeError(
                "deadline regression label lacks a proof or exact fixed-work cap"
            )


def run_once(
    *, teacher: pathlib.Path, model: pathlib.Path, positions: pathlib.Path,
    root: pathlib.Path, run: int,
) -> bytes:
    output = root / f"run-{run}"
    output.mkdir()
    manager = workflow.StageManager(
        output=output,
        campaign_id=CAMPAIGN_ID,
        round_index=0,
        resume=False,
        environment={"fixture": STAGE_NAME, "run": run},
    )
    labels = output / "labels.jsonl"
    result = workflow.run_label_chunks(
        manager=manager,
        stage_ordinal=STAGE_ORDINAL,
        stage_name=STAGE_NAME,
        positions=positions,
        output=labels,
        teacher=teacher,
        schema=SCHEMA,
        campaign_id=CAMPAIGN_ID,
        nodes=TREE_NODES,
        workers=WORKERS,
        source_sha256=workflow._source_closure(
            ROOT, workflow.SEARCH_TEACHER_SOURCE_PATHS
        ),
        model=model,
        chunk_games=1,
    )
    receipts = sorted(
        (manager.receipts / f"{STAGE_ORDINAL:02d}-{STAGE_NAME}-chunks").glob(
            "chunk-*.json"
        )
    )
    if result.get("teacher_rows") != len(REGRESSIONS) or len(receipts) != len(
        REGRESSIONS
    ):
        raise RuntimeError("deadline regression did not publish four chunk receipts")
    for receipt in receipts:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            record.get("rows") != 1
            or record.get("games") != 1
            or record.get("teacher_rows") != 1
            or record.get("nodes") != TREE_NODES
            or record.get("teacher_configuration", {}).get("max_time_ms") != 0
        ):
            raise RuntimeError("deadline regression did not use one-row fixed-work chunks")
    payload = labels.read_bytes()
    validate_labels(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if workflow.SEARCH_TEACHER_SCHEMA != SCHEMA:
        raise RuntimeError("deadline regression requires the v4 search-label schema")
    if workflow.FIXED_WORK_TIME_MS != 0:
        raise RuntimeError("deadline regression requires internal search time zero")
    rows = [regression["row"] for regression in REGRESSIONS]
    if len({row.split("\t")[2] for row in rows}) != len(rows):
        raise RuntimeError("deadline regressions no longer form one-row game chunks")

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        positions = root / "positions.tsv"
        positions.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        first = run_once(
            teacher=arguments.search_teacher,
            model=arguments.model,
            positions=positions,
            root=root,
            run=1,
        )
        second = run_once(
            teacher=arguments.search_teacher,
            model=arguments.model,
            positions=positions,
            root=root,
            run=2,
        )
        if first != second:
            raise RuntimeError("deadline regression labels changed between identical runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
