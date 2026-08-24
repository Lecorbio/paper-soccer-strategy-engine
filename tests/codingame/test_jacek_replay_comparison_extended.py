#!/usr/bin/env python3
"""Contract tests for post-campaign comparison extensions."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import tempfile


def run(command: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, text=True, capture_output=True)
    if expect_success and completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    if not expect_success and completed.returncode == 0:
        raise RuntimeError("command unexpectedly succeeded: " + " ".join(command))
    return completed


def normalized_game(game: dict) -> dict:
    result = copy.deepcopy(game)
    result.pop("candidate_ms", None)
    result.pop("control_ms", None)
    for key in ("candidate_work", "control_work"):
        work = result.get(key, {})
        for timing in ("total_ms", "p99_ms", "max_ms"):
            work.pop(timing, None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    executable = str(arguments.comparison.resolve())
    model = str(arguments.model.resolve())
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        bank = root / "bank.tsv"
        run(
            [
                executable,
                "--generate-bank",
                str(bank),
                "--bank-classification",
                "development",
                "--pairs",
                "4",
                "--opening-plies",
                "12",
                "--seed",
                "424242",
            ]
        )

        common = [
            executable,
            "--model",
            model,
            "--bank",
            str(bank),
            "--bank-classification",
            "development",
            "--time-ms",
            "1000",
            "--tree-nodes",
            "32",
            "--max-partial-paths",
            "32",
            "--max-turns",
            "80",
        ]
        direct = root / "direct.json"
        run(
            common
            + [
                "--control-model",
                model,
                "--opponent",
                "jacek-replay",
                "--pairs",
                "4",
                "--output",
                str(direct),
            ]
        )
        direct_report = json.loads(direct.read_text(encoding="utf-8"))
        by_opening: dict[str, list[dict]] = {}
        for game in direct_report["results"]:
            by_opening.setdefault(game["opening"], []).append(game)
        if len(by_opening) != 4:
            raise RuntimeError("direct-model report did not preserve four pairs")
        for games in by_opening.values():
            if len(games) != 2 or games[0]["transcript"] != games[1]["transcript"]:
                raise RuntimeError("identical models did not replay an identical game")
            if games[0]["winner"] != games[1]["winner"]:
                raise RuntimeError("identical models disagreed on the physical winner")

        full = root / "full.json"
        first = root / "first.json"
        second = root / "second.json"
        base = common + [
            "--opponent",
            "jacek-nn",
            "--control-work",
            "16",
        ]
        run(base + ["--pairs", "4", "--output", str(full)])
        run(base + ["--pairs", "2", "--pair-offset", "0", "--output", str(first)])
        run(base + ["--pairs", "2", "--pair-offset", "2", "--output", str(second)])
        full_games = json.loads(full.read_text(encoding="utf-8"))["results"]
        shard_games = (
            json.loads(first.read_text(encoding="utf-8"))["results"]
            + json.loads(second.read_text(encoding="utf-8"))["results"]
        )
        if [normalized_game(game) for game in full_games] != [
            normalized_game(game) for game in shard_games
        ]:
            raise RuntimeError("pair-offset shards differ from the full bank")

        run(
            [
                executable,
                "--generate-bank",
                str(root / "bad.tsv"),
                "--pairs",
                "1",
                "--pair-offset",
                "1",
            ],
            expect_success=False,
        )
        run(common + ["--opponent", "jacek-nn", "--control-model", model],
            expect_success=False)
        run(common + ["--opponent", "jacek-replay"], expect_success=False)
        run(common + ["--opponent", "rank4", "--pairs", "1", "--pair-offset", "4"],
            expect_success=False)


if __name__ == "__main__":
    main()
