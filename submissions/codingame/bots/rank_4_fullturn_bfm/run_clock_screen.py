#!/usr/bin/env python3

"""Run and record the construction-inclusive CodinGame clock safety screen."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = "papersoccer.fullturn-bfm-clock-screen.v1"
DEFAULT_OPENING_TURNS = (0, 1, 2, 3, 6, 7, 12)


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_opening_turns(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("opening turns must be integers") from error
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "opening turns must be a non-empty list of unique nonnegative integers"
        )
    return values


def parse_summary(stdout: str) -> dict[str, str]:
    summaries = [
        line for line in stdout.splitlines() if line.startswith("summary batch=all ")
    ]
    if len(summaries) != 1:
        raise ValueError("comparison output must contain one all-games summary")
    fields: dict[str, str] = {}
    for token in summaries[0].split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def integer(fields: dict[str, str], name: str) -> int:
    try:
        return int(fields[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"summary field {name} is missing or invalid") from error


def decimal(fields: dict[str, str], name: str) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"summary field {name} is missing or invalid") from error
    if not math.isfinite(value):
        raise ValueError(f"summary field {name} is missing or invalid")
    return value


def timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def evaluate_summary(
    fields: dict[str, str],
    *,
    expected_games: int,
    first_headroom_ms: float,
    later_headroom_ms: float,
) -> list[str]:
    failures: list[str] = []
    if integer(fields, "games") != expected_games:
        failures.append("unexpected game count")
    for name in (
        "unfinished",
        "candidate_operational_timeouts",
        "reference_operational_timeouts",
    ):
        if integer(fields, name) != 0:
            failures.append(f"{name} is nonzero")
    for engine in ("candidate", "reference"):
        if decimal(fields, f"{engine}_max_first_ms") >= first_headroom_ms:
            failures.append(f"{engine} exceeded first-response headroom")
        if decimal(fields, f"{engine}_max_later_ms") >= later_headroom_ms:
            failures.append(f"{engine} exceeded later-response headroom")
    return failures


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("ascii")
        + b"\n"
    )


def write_once(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(content)
        return
    except FileExistsError:
        pass
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace existing report: {path}")
        return
    raise FileNotFoundError(f"report path disappeared while writing: {path}")


def build_command(arguments: argparse.Namespace) -> list[str]:
    opening_turns = ",".join(str(value) for value in arguments.opening_turns)
    return [
        str(arguments.gate),
        "--pairs-per-depth",
        str(arguments.pairs_per_depth),
        "--candidate-work",
        str(arguments.candidate_work),
        "--reference-nodes",
        str(arguments.reference_nodes),
        "--max-turns",
        str(arguments.max_turns),
        "--batch-start",
        str(arguments.batch_start),
        "--batch-count",
        str(arguments.batch_count),
        "--opening-turns",
        opening_turns,
        "--candidate-first-ms",
        str(arguments.first_ms),
        "--candidate-later-ms",
        str(arguments.later_ms),
        "--reference-first-ms",
        str(arguments.first_ms),
        "--reference-later-ms",
        str(arguments.later_ms),
        "--candidate-replay",
        "1",
        "--reference-replay",
        "1",
        "--candidate-max-actions",
        str(arguments.candidate_max_actions),
        "--candidate-nonroot-actions",
        str(arguments.candidate_nonroot_actions),
        "--candidate-exploration",
        str(arguments.candidate_exploration),
        "--candidate-fpu",
        str(arguments.candidate_fpu),
        "--candidate-final-visit-weight",
        str(arguments.candidate_final_visit_weight),
        "--candidate-replay-blend",
        str(arguments.candidate_replay_blend),
        "--candidate-residual-weight",
        str(arguments.candidate_residual_weight),
        "--candidate-root-only",
        "1" if arguments.candidate_root_only else "0",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--submission", type=pathlib.Path, default=HERE / "submission.cpp"
    )
    parser.add_argument("--pairs-per-depth", type=int, default=1)
    parser.add_argument("--candidate-work", type=int, default=3_000_000)
    parser.add_argument("--reference-nodes", type=int, default=3_000_000)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument(
        "--opening-turns",
        type=parse_opening_turns,
        default=DEFAULT_OPENING_TURNS,
    )
    parser.add_argument("--first-ms", type=int, default=800)
    parser.add_argument("--later-ms", type=int, default=165)
    parser.add_argument("--candidate-exploration", type=float, default=1.5)
    parser.add_argument("--candidate-max-actions", type=int, default=250)
    parser.add_argument("--candidate-nonroot-actions", type=int, default=0)
    parser.add_argument("--candidate-fpu", type=float, default=0.5)
    parser.add_argument("--candidate-final-visit-weight", type=float, default=0.0)
    parser.add_argument("--candidate-replay-blend", type=int, default=15)
    parser.add_argument("--candidate-residual-weight", type=int, default=100)
    parser.add_argument("--candidate-root-only", action="store_true")
    parser.add_argument("--first-headroom-ms", type=float, default=900.0)
    parser.add_argument("--later-headroom-ms", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    arguments = parser.parse_args()

    positive = (
        arguments.pairs_per_depth,
        arguments.candidate_work,
        arguments.reference_nodes,
        arguments.max_turns,
        arguments.batch_count,
        arguments.first_ms,
        arguments.later_ms,
    )
    if any(value <= 0 for value in positive) or arguments.batch_start < 0:
        parser.error("budgets, counts, turn limit, and clocks must be positive")
    if arguments.first_headroom_ms <= arguments.first_ms:
        parser.error("first-response headroom must exceed the search clock")
    if arguments.later_headroom_ms <= arguments.later_ms:
        parser.error("later-response headroom must exceed the search clock")
    if (
        not math.isfinite(arguments.candidate_exploration)
        or arguments.candidate_exploration < 0
    ):
        parser.error("candidate exploration must be nonnegative")
    if not 1 <= arguments.candidate_max_actions <= 250:
        parser.error("candidate max actions must be between 1 and 250")
    if not 0 <= arguments.candidate_nonroot_actions <= 250:
        parser.error("candidate non-root actions must be between 0 and 250")
    if not math.isfinite(arguments.candidate_fpu):
        parser.error("candidate FPU must be finite")
    if (
        not math.isfinite(arguments.candidate_final_visit_weight)
        or arguments.candidate_final_visit_weight < 0
    ):
        parser.error("candidate final visit weight must be nonnegative")
    if not 0 <= arguments.candidate_replay_blend <= 100:
        parser.error("candidate replay blend must be between 0 and 100")
    if not 0 <= arguments.candidate_residual_weight <= 100:
        parser.error("candidate residual weight must be between 0 and 100")
    gate = arguments.gate.resolve()
    submission = arguments.submission.resolve()
    if not gate.is_file() or not submission.is_file():
        parser.error("gate and submission must be files")
    arguments.gate = gate

    command = build_command(arguments)
    started = utc_now()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=arguments.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        timed_out = True
        completed = subprocess.CompletedProcess(
            command,
            -1,
            timeout_text(error.stdout),
            timeout_text(error.stderr),
        )
    finished = utc_now()
    parse_error: str | None = None
    summary: dict[str, str] = {}
    failures: list[str] = []
    try:
        summary = parse_summary(completed.stdout)
        expected_games = 10 + (
            2
            * arguments.pairs_per_depth
            * arguments.batch_count
            * len(arguments.opening_turns)
        )
        failures.extend(
            evaluate_summary(
                summary,
                expected_games=expected_games,
                first_headroom_ms=arguments.first_headroom_ms,
                later_headroom_ms=arguments.later_headroom_ms,
            )
        )
    except ValueError as error:
        parse_error = str(error)
        failures.append("comparison summary could not be validated")
    if completed.returncode != 0:
        failures.append(f"comparison gate exited {completed.returncode}")
    if timed_out:
        failures.append("comparison gate timed out")

    report = {
        "schema": SCHEMA,
        "started_at_utc": started,
        "completed_at_utc": finished,
        "passed": not failures,
        "failures": failures,
        "parse_error": parse_error,
        "command": command,
        "gate_sha256": sha256_file(gate),
        "submission_path": submission.as_posix(),
        "submission_sha256": sha256_file(submission),
        "configuration": {
            "pairs_per_depth": arguments.pairs_per_depth,
            "batch_start": arguments.batch_start,
            "batch_count": arguments.batch_count,
            "opening_turns": list(arguments.opening_turns),
            "candidate_work_ceiling": arguments.candidate_work,
            "reference_node_ceiling": arguments.reference_nodes,
            "maximum_turns": arguments.max_turns,
            "first_clock_ms": arguments.first_ms,
            "later_clock_ms": arguments.later_ms,
            "first_headroom_ms": arguments.first_headroom_ms,
            "later_headroom_ms": arguments.later_headroom_ms,
            "candidate_exploration": arguments.candidate_exploration,
            "candidate_max_actions": arguments.candidate_max_actions,
            "candidate_nonroot_actions": arguments.candidate_nonroot_actions,
            "candidate_fpu": arguments.candidate_fpu,
            "candidate_final_visit_weight": arguments.candidate_final_visit_weight,
            "candidate_replay_blend": arguments.candidate_replay_blend,
            "candidate_residual_weight": arguments.candidate_residual_weight,
            "candidate_root_only": arguments.candidate_root_only,
        },
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_once(arguments.output.resolve(), canonical_bytes(report))
    print(
        json.dumps(
            {
                "output": arguments.output.resolve().as_posix(),
                "passed": not failures,
                "failures": failures,
                "submission_sha256": report["submission_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
