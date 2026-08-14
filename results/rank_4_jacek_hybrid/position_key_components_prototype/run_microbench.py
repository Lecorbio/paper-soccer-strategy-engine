#!/usr/bin/env python3
import fcntl
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "microbench_control"
CANDIDATE = ROOT / "microbench_candidate_safe"
LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
WARMUP = 30
PAIRS = 300


def run(path: Path, panel: str):
    fields = subprocess.check_output([path, panel], text=True).split()
    if len(fields) != 7:
        raise RuntimeError(fields)
    return int(fields[0]), tuple(map(int, fields[1:]))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def measure(panel: str):
    def pair(index):
        if index % 2 == 0:
            control = run(CONTROL, panel)
            candidate = run(CANDIDATE, panel)
        else:
            candidate = run(CANDIDATE, panel)
            control = run(CONTROL, panel)
        if control[1] != candidate[1]:
            raise RuntimeError(
                f"signature mismatch: {control[1]} != {candidate[1]}")
        return control, candidate

    for index in range(WARMUP):
        pair(index)
    control_times = []
    candidate_times = []
    signatures = set()
    for index in range(PAIRS):
        control, candidate = pair(index)
        control_times.append(control[0])
        candidate_times.append(candidate[0])
        signatures.add(control[1])
    control_median = statistics.median(control_times)
    candidate_median = statistics.median(candidate_times)
    control_p99 = percentile(control_times, 0.99)
    candidate_p99 = percentile(candidate_times, 0.99)
    return {
        "panel": panel,
        "warmup_pairs": WARMUP,
        "measured_pairs": PAIRS,
        "signatures": [list(value) for value in sorted(signatures)],
        "control_ns": {
            "median": control_median,
            "p99": control_p99,
            "min": min(control_times),
            "max": max(control_times),
        },
        "candidate_ns": {
            "median": candidate_median,
            "p99": candidate_p99,
            "min": min(candidate_times),
            "max": max(candidate_times),
        },
        "ratios": {
            "median": candidate_median / control_median,
            "p99": candidate_p99 / control_p99,
        },
    }


started = time.monotonic()
with LOCK.open("a+") as lock:
    lock_wait_started = time.monotonic()
    fcntl.flock(lock, fcntl.LOCK_EX)
    lock_wait = time.monotonic() - lock_wait_started
    panels = [measure("forced-prod"), measure("mixed-prod")]

report = {
    "schema": "rank4-position-key-component-safe-private-microbench-v1",
    "lock": str(LOCK),
    "lock_wait_seconds": lock_wait,
    "elapsed_seconds": time.monotonic() - started,
    "panels": panels,
    "thresholds": {"median_ratio_max": 0.99, "p99_ratio_max": 1.005},
}
encoded = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
(ROOT / "position_key_microbench_safe.json").write_text(encoded)
print(json.dumps(report, indent=2, sort_keys=True))
if not all(panel["ratios"]["median"] <= 0.99 and
           panel["ratios"]["p99"] <= 1.005 for panel in panels):
    raise SystemExit(1)
