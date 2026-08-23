#!/usr/bin/env python3
"""Run the frozen noncanonical R0 -> R1 -> R2 workflow and resume it."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--continuations", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    try:
        import numpy  # noqa: F401
    except ImportError:
        research_python = repository / ".venv/bin/python"
        if (
            research_python.is_file()
            and pathlib.Path(sys.prefix).resolve()
            != (repository / ".venv").resolve()
        ):
            completed = subprocess.run(
                [
                    str(research_python),
                    str(pathlib.Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
                check=False,
            )
            return completed.returncode
        print("jacek replay workflow smoke: skipped (NumPy unavailable)")
        return 0
    sys.path.insert(0, str(repository / "tools"))
    import jacek_replay_workflow as workflow

    snapshot = repository / (
        "submissions/codingame/bots/neural_puct/live_replay/corpora/"
        "687468e84c475107eee840f4d731fbc51182e8cfc20d2bf2cd7039d344f48f97.json"
    )
    snapshot_payload = json.loads(snapshot.read_bytes())
    exclusions = repository / snapshot_payload["exclusion_registry_path"]
    public = repository / (
        "submissions/codingame/bots/neural_puct/public_jacek_unlocked_v1.json"
    )
    tool = repository / "tools/jacek_replay_workflow.py"
    with tempfile.TemporaryDirectory(prefix="jacek-replay-workflow-smoke.") as raw:
        campaign = pathlib.Path(raw)
        rounds = []
        for round_index in range(3):
            output = campaign / f"round-{round_index}"
            command = [
                sys.executable,
                str(tool),
                "--repository",
                str(repository),
                "--exclusions",
                str(exclusions),
                "--public-jacek",
                str(public),
                "--live-snapshot",
                str(snapshot),
                "--teacher",
                str(arguments.teacher.resolve()),
                "--continuation-generator",
                str(arguments.continuations.resolve()),
                "--round",
                str(round_index),
                "--output-directory",
                str(output),
                "--smoke-profile",
            ]
            if round_index:
                previous = rounds[-1]
                command.extend(
                    [
                        "--previous-workflow",
                        str(previous / "workflow.json"),
                        "--previous-roots",
                        str(previous / "replay-roots.json"),
                        "--continuation-model",
                        str(previous / "model/jacek_replay_bfm.runtime"),
                    ]
                )
                for prior in rounds:
                    command.extend(
                        ["--prior-pack-report", str(prior / "shards/pack-report.json")]
                    )
            report = run(command)
            if pathlib.Path(report["workflow"]).resolve() != (
                output / "workflow.json"
            ).resolve():
                raise RuntimeError("workflow reported the wrong receipt path")
            rounds.append(output)

        validation = workflow.validate_smoke_workflow_chain(
            rounds[2] / "workflow.json", 2
        )
        if [entry["round"] for entry in validation["entries"]] != [0, 1, 2]:
            raise RuntimeError("smoke workflow ancestry is incomplete")
        receipts = sorted((rounds[2] / "receipts").glob("*.json"))
        before = {path.name: digest(path) for path in receipts}
        resume_command = command + ["--resume"]
        run(resume_command)
        after = {path.name: digest(path) for path in receipts}
        if before != after:
            raise RuntimeError("resume rewrote a completed stage receipt")
        for round_index, output in enumerate(rounds):
            receipt = workflow.validate_smoke_workflow_chain(
                output / "workflow.json", round_index
            )["receipt"]
            if (
                receipt["configuration"]["campaign_eligible"] is not False
                or receipt["configuration"]["final_test_revealed"]
                != (round_index == 2)
            ):
                raise RuntimeError("smoke eligibility/test-reveal contract changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
