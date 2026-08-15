#!/usr/bin/env python3
"""Frozen no-run paired benchmark for frontier weight 10.

This runner is archived as a preregistered DEVELOPMENT instrument. Archiving
it does not authorize execution. The campaign owner must release the shared
prototype lock before the first run.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


CONTROL_SHA256 = (
    "2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7"
)
CANDIDATE_SHA256 = (
    "08d0c0859ef8a197f8bfdd89afb048bec41c3a888228433b85991cd937882550"
)
HARNESS_NAME = "leaf_proof_microbenchmark.549b6c29.cpp"
HARNESS_SHA256 = (
    "549b6c293f05d05b8e1724073decee034fc2959c2981902215925a53c91de059"
)
LOCK_PATH = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
WARMUP_PAIRS = 7
MEASURED_PAIRS = 31
MEDIAN_RATIO_MAX = 1.010
P99_RATIO_MAX = 1.020
COMPILE_FLAGS = ("-std=c++20", "-O3", "-DNDEBUG")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def build(compiler: str, harness: Path, implementation: Path,
          output: Path) -> list[str]:
    command = [
        compiler,
        *COMPILE_FLAGS,
        f'-DFRONTIER_IMPL="{implementation}"',
        str(harness),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return command


def run(binary: Path) -> dict[str, object]:
    fields = subprocess.check_output([binary], text=True).split()
    if len(fields) != 9:
        raise RuntimeError(f"unexpected benchmark output: {fields!r}")
    values = [int(field) for field in fields]
    return {
        "elapsed_ns": values[0],
        "fixture_digest": values[1],
        "result_digest": values[2],
        "calls": values[3],
        "leaf_probes": values[4],
        "leaf_wins": values[5],
        "leaf_losses": values[6],
        "evaluation_probes": values[7],
        "evaluation_hits": values[8],
    }


def common_signature(result: dict[str, object]) -> tuple[object, ...]:
    return (
        result["fixture_digest"],
        result["calls"],
        result["leaf_probes"],
        result["leaf_wins"],
        result["leaf_losses"],
        result["evaluation_probes"],
        result["evaluation_hits"],
    )


def paired_run(control: Path, candidate: Path, index: int) -> dict[str, object]:
    if index % 2 == 0:
        order = "control-candidate"
        control_result = run(control)
        candidate_result = run(candidate)
    else:
        order = "candidate-control"
        candidate_result = run(candidate)
        control_result = run(control)
    if common_signature(control_result) != common_signature(candidate_result):
        raise RuntimeError(
            "control/candidate fixture or proof-work signature mismatch")
    return {
        "index": index,
        "order": order,
        "control": control_result,
        "candidate": candidate_result,
        "paired_ratio": (
            candidate_result["elapsed_ns"] / control_result["elapsed_ns"]
        ),
    }


def write_exclusive(path: Path, report: dict[str, object]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="/usr/bin/c++")
    args = parser.parse_args()

    control_source = args.control_source.resolve(strict=True)
    candidate_source = args.candidate_source.resolve(strict=True)
    output = args.output.resolve()
    harness = Path(__file__).resolve().with_name(HARNESS_NAME)
    if sha256(control_source) != CONTROL_SHA256:
        raise RuntimeError("control source does not match frozen rollback")
    if sha256(candidate_source) != CANDIDATE_SHA256:
        raise RuntimeError("candidate source does not match frozen weight 10")
    if sha256(harness) != HARNESS_SHA256:
        raise RuntimeError("benchmark harness hash mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"shared prototype benchmark lock is busy: {LOCK_PATH}"
            ) from error

        with tempfile.TemporaryDirectory(
                prefix="rank4-frontier-leaf-benchmark.") as temporary:
            temporary_path = Path(temporary)
            control_binary = temporary_path / "control"
            candidate_binary = temporary_path / "candidate"
            control_build = build(
                args.compiler, harness, control_source, control_binary)
            candidate_build = build(
                args.compiler, harness, candidate_source, candidate_binary)
            compiler_version = subprocess.check_output(
                [args.compiler, "--version"], text=True).splitlines()[0]

            for index in range(WARMUP_PAIRS):
                paired_run(control_binary, candidate_binary, index)

            samples = [
                paired_run(control_binary, candidate_binary, index)
                for index in range(MEASURED_PAIRS)
            ]
            ratios = [float(sample["paired_ratio"]) for sample in samples]
            control_times = [
                int(sample["control"]["elapsed_ns"]) for sample in samples
            ]
            candidate_times = [
                int(sample["candidate"]["elapsed_ns"]) for sample in samples
            ]
            control_digests = {
                int(sample["control"]["result_digest"]) for sample in samples
            }
            candidate_digests = {
                int(sample["candidate"]["result_digest"])
                for sample in samples
            }
            if len(control_digests) != 1 or len(candidate_digests) != 1:
                raise RuntimeError("a result digest changed between repetitions")
            if control_digests == candidate_digests:
                raise RuntimeError("frontier feature did not change the leaf panel")

            median_ratio = statistics.median(ratios)
            p99_ratio = percentile(ratios, 0.99)
            passed = (
                median_ratio <= MEDIAN_RATIO_MAX and
                p99_ratio <= P99_RATIO_MAX
            )
            report = {
                "schema": "rank4-frontier-weight10-leaf-proof-benchmark-v1",
                "classification": "development-timing-not-strength-evidence",
                "lock": {
                    "path": str(LOCK_PATH),
                    "mode": "exclusive-nonblocking",
                    "held_for_build_warmup_measurement_and_write": True,
                },
                "source": {
                    "control_path": str(control_source),
                    "control_sha256": CONTROL_SHA256,
                    "candidate_path": str(candidate_source),
                    "candidate_sha256": CANDIDATE_SHA256,
                    "harness_path": str(harness),
                    "harness_sha256": HARNESS_SHA256,
                },
                "build": {
                    "compiler": args.compiler,
                    "compiler_version": compiler_version,
                    "flags": list(COMPILE_FLAGS),
                    "control_command": control_build,
                    "candidate_command": candidate_build,
                    "control_binary_sha256": sha256(control_binary),
                    "candidate_binary_sha256": sha256(candidate_binary),
                },
                "fixture_contract": {
                    "fixture_count": 512,
                    "public_tactical_transcripts": 8,
                    "remaining_fixtures": "deterministic public-rules random walk",
                    "timed_region": (
                        "one first-use exact leaf-boundary proof/evaluation call "
                        "per preconstructed search; corpus and topology setup excluded"
                    ),
                    "cache_contract": "512 evaluation probes and zero hits",
                },
                "sampling": {
                    "warmup_pairs": WARMUP_PAIRS,
                    "measured_pairs": MEASURED_PAIRS,
                    "order": "alternating AB/BA",
                    "raw_samples": samples,
                },
                "metrics": {
                    "control_median_ns": statistics.median(control_times),
                    "candidate_median_ns": statistics.median(candidate_times),
                    "control_p99_ns": percentile(control_times, 0.99),
                    "candidate_p99_ns": percentile(candidate_times, 0.99),
                    "paired_ratio_median": median_ratio,
                    "paired_ratio_p99": p99_ratio,
                },
                "thresholds": {
                    "paired_ratio_median_max": MEDIAN_RATIO_MAX,
                    "paired_ratio_p99_max": P99_RATIO_MAX,
                    "conjunctive": True,
                },
                "attestation": {
                    "whole_games_run": False,
                    "validation_bank_read": False,
                    "final_bank_read": False,
                    "strength_evidence": False,
                },
                "passed": passed,
            }
            write_exclusive(output, report)
            return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
