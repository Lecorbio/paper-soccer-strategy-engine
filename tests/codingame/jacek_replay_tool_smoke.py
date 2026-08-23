#!/usr/bin/env python3
"""Exercise deep relabel selection and both continuation actor rounds."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile


def source_rows(repository: pathlib.Path, count: int) -> str:
    path = repository / (
        "submissions/codingame/bots/neural_puct/live_replay/corpora/"
        "a6ba0d2e76b44d22432070e85bf215e6ab395e361d978b43e1915e05965ac3da.relabel.tsv"
    )
    rows = ["group_id\tsource\twinner\ttranscript"]
    for raw in path.read_text().splitlines()[1 : count + 1]:
        fields = raw.split("\t")
        actions = "/".join(item.split(":", 1)[1] for item in fields[5].split("/"))
        rows.append(f"own-live:{fields[0]}\town-live\t{fields[4]}\t{actions}")
    return "\n".join(rows) + "\n"


def run_checked(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n{completed.stderr}"
        )
    if completed.stderr:
        raise RuntimeError(f"command wrote stderr: {completed.stderr}")
    return completed.stdout


def run_failed(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0:
        raise RuntimeError(f"command unexpectedly passed: {' '.join(command)}")
    if completed.stdout:
        raise RuntimeError(f"failed command wrote stdout: {completed.stdout}")
    return completed.stderr


ACTOR_MODES = (
    "rank4-vs-rank4",
    "candidate-selfplay",
    "candidate-p1-vs-rank4",
    "candidate-p2-vs-rank4",
)


def expected_quotas(round_number: int, games: int) -> dict[str, int]:
    if round_number == 0:
        return dict(zip(ACTOR_MODES, (games, 0, 0, 0), strict=True))
    quotient, remainder = divmod(games, 4)
    counts = [2 * quotient + (2 * remainder) // 4, quotient, quotient]
    residuals = [(2 * remainder) % 4, remainder, remainder]
    for index in sorted(range(3), key=lambda item: (-residuals[item], item)):
        if sum(counts) == games:
            break
        counts[index] += 1
    return dict(zip(ACTOR_MODES, (0, *counts), strict=True))


def validate_continuations(
    *,
    roots: pathlib.Path,
    output: pathlib.Path,
    manifest_path: pathlib.Path,
    model: pathlib.Path,
    round_number: int,
    games: int,
) -> dict:
    output_bytes = output.read_bytes()
    text = output_bytes.decode()
    manifest = json.loads(manifest_path.read_bytes())
    expected_policy = (
        "rank4-vs-rank4"
        if round_number == 0
        else "50%-candidate-selfplay+50%-candidate-rank4-balanced"
    )
    if f"# actor-policy={expected_policy}" not in text:
        raise RuntimeError("continuation actor policy is not recorded")

    data_lines = [line for line in text.splitlines() if not line.startswith("#")]
    if data_lines[0] != "group_id\tsource\twinner\ttranscript":
        raise RuntimeError("continuation TSV is no longer teacher-compatible")
    tsv_rows = [line.split("\t") for line in data_lines[1:]]
    if len(tsv_rows) != games or any(len(row) != 4 for row in tsv_rows):
        raise RuntimeError("continuation output has the wrong row shape")

    quotas = expected_quotas(round_number, games)
    if (
        manifest.get("schema")
        != "papersoccer.jacek-replay-continuations-manifest.v1"
        or manifest.get("tsv_schema")
        != "papersoccer.jacek-replay-continuations.v1"
        or manifest.get("round") != round_number
        or manifest.get("requested_games") != games
        or manifest.get("successful_games") != games
        or manifest.get("planned_quotas") != quotas
        or manifest.get("successful_quotas") != quotas
    ):
        raise RuntimeError("continuation manifest quota contract is invalid")
    bindings = manifest.get("bindings", {})
    if (
        bindings.get("input_sha256")
        != hashlib.sha256(roots.read_bytes()).hexdigest()
        or bindings.get("output_sha256")
        != hashlib.sha256(output_bytes).hexdigest()
        or bindings.get("model_sha256")
        != (
            None
            if round_number == 0
            else hashlib.sha256(model.read_bytes()).hexdigest()
        )
    ):
        raise RuntimeError("continuation manifest artifact binding is invalid")

    rows = manifest.get("rows", [])
    if len(rows) != games:
        raise RuntimeError("continuation manifest row count is invalid")
    root_rows = {}
    for ordinal, raw in enumerate(roots.read_text().splitlines()[1:]):
        group_id, _source, _winner, transcript = raw.split("\t")
        root_rows[(ordinal, group_id)] = hashlib.sha256(
            transcript.encode()
        ).hexdigest()
    ids: set[str] = set()
    colors = {
        "rank4-vs-rank4": "none",
        "candidate-selfplay": "both",
        "candidate-p1-vs-rank4": "player-one",
        "candidate-p2-vs-rank4": "player-two",
    }
    for ordinal, (row, tsv_row) in enumerate(zip(rows, tsv_rows, strict=True)):
        identifier = row.get("continuation_id", "")
        lineage = row.get("root_lineage", {})
        root_key = (lineage.get("root_row_ordinal"), lineage.get("group_id"))
        if (
            not re.fullmatch(r"continuation:[0-9a-f]{64}", identifier)
            or identifier in ids
            or row.get("row_ordinal") != ordinal
            or row.get("candidate_color") != colors.get(row.get("actor_mode"))
            or root_key not in root_rows
            or lineage.get("root_transcript_sha256") != root_rows[root_key]
            or tsv_row[0] != lineage.get("group_id")
            or tsv_row[1] != f"continuation-round-{round_number}"
            or row.get("transcript_sha256")
            != hashlib.sha256(tsv_row[3].encode()).hexdigest()
        ):
            raise RuntimeError("continuation manifest row binding is invalid")
        ids.add(identifier)
    if collections.Counter(row["actor_mode"] for row in rows) != {
        mode: count for mode, count in quotas.items() if count
    }:
        raise RuntimeError("successful actor rows do not satisfy exact quotas")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--continuations", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    one_game = source_rows(arguments.repository, 1)
    labels = [
        json.loads(line)
        for line in run_checked(
            [
                str(arguments.teacher),
                "--nodes",
                "500",
                "--deep-nodes",
                "5000",
                "--deep-percent",
                "20",
                "--max-samples",
                "10",
            ],
            input_text=one_game,
        ).splitlines()
    ]
    if len(labels) < 2 or not any(row["nodes"] > 500 for row in labels):
        raise RuntimeError("deep uncertainty relabel did not replace any bulk label")
    if not any(row["nodes"] == 500 for row in labels):
        raise RuntimeError("deep uncertainty relabel unexpectedly replaced every label")

    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        roots = directory / "roots.tsv"
        roots.write_text(source_rows(arguments.repository, 8))
        outputs: dict[int, tuple[pathlib.Path, pathlib.Path]] = {}
        for round_number in (0, 1, 2):
            output = directory / f"round-{round_number}.tsv"
            manifest_path = directory / f"round-{round_number}.manifest.json"
            command = [
                str(arguments.continuations),
                "--input",
                str(roots),
                "--output",
                str(output),
                "--manifest",
                str(manifest_path),
                "--games",
                "4",
                "--round",
                str(round_number),
                "--seed",
                "17",
                "--actor-nodes",
                "500",
                "--candidate-tree-nodes",
                "32",
                "--max-turns",
                "160",
            ]
            if round_number:
                command.extend(["--model", str(arguments.model)])
            run_checked(command)
            validate_continuations(
                roots=roots,
                output=output,
                manifest_path=manifest_path,
                model=arguments.model,
                round_number=round_number,
                games=4,
            )
            run_checked(
                [str(arguments.teacher), "--nodes", "500", "--max-samples", "1"],
                input_text=output.read_text(),
            )
            outputs[round_number] = (output, manifest_path)

        duplicate_output = directory / "round-1-duplicate.tsv"
        duplicate_manifest = directory / "round-1-duplicate.manifest.json"
        run_checked(
            [
                str(arguments.continuations),
                "--input",
                str(roots),
                "--output",
                str(duplicate_output),
                "--manifest",
                str(duplicate_manifest),
                "--games",
                "4",
                "--round",
                "1",
                "--seed",
                "17",
                "--actor-nodes",
                "500",
                "--candidate-tree-nodes",
                "32",
                "--max-turns",
                "160",
                "--model",
                str(arguments.model),
            ]
        )
        if (
            duplicate_output.read_bytes() != outputs[1][0].read_bytes()
            or duplicate_manifest.read_bytes() != outputs[1][1].read_bytes()
        ):
            raise RuntimeError("continuation generation is not byte-deterministic")

        unfinished_root = directory / "unfinished-root.tsv"
        unfinished_root.write_text(
            "group_id\tsource\twinner\ttranscript\n"
            "unfinished\tsmoke\t0\t0\n"
        )
        preserved_output = directory / "preserved.tsv"
        preserved_manifest = directory / "preserved.manifest.json"
        preserved_output.write_bytes(b"existing-output\n")
        preserved_manifest.write_bytes(b"existing-manifest\n")
        failure = run_failed(
            [
                str(arguments.continuations),
                "--input",
                str(unfinished_root),
                "--output",
                str(preserved_output),
                "--manifest",
                str(preserved_manifest),
                "--games",
                "1",
                "--round",
                "0",
                "--seed",
                "17",
                "--actor-nodes",
                "16",
                "--candidate-tree-nodes",
                "32",
                "--max-turns",
                "1",
            ]
        )
        if (
            "before attempt cap" not in failure
            or preserved_output.read_bytes() != b"existing-output\n"
            or preserved_manifest.read_bytes() != b"existing-manifest\n"
        ):
            raise RuntimeError("failed continuation generation was not atomic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
