#!/usr/bin/env python3
import math
import statistics
import subprocess
import sys

ROOT = "/tmp/jacek-single-edge-rollback.BgsqVV/"
CONTROL = ROOT + "microbench_control"
CANDIDATE = ROOT + "microbench_candidate"
PANEL = sys.argv[1]
WARMUP = 30
PAIRS = 300


def run(path):
    fields = subprocess.check_output([path, PANEL], text=True).split()
    if len(fields) != 7:
        raise RuntimeError(fields)
    return int(fields[0]), tuple(map(int, fields[1:]))


def pair(index):
    if index % 2 == 0:
        left = run(CONTROL)
        right = run(CANDIDATE)
    else:
        right = run(CANDIDATE)
        left = run(CONTROL)
    if left[1] != right[1]:
        raise RuntimeError(f"signature mismatch: {left[1]} != {right[1]}")
    return left, right


for index in range(WARMUP):
    pair(index)

control = []
candidate = []
signatures = set()
for index in range(PAIRS):
    left, right = pair(index)
    control.append(left[0])
    candidate.append(right[0])
    signatures.add(left[1])


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


cm = statistics.median(control)
tm = statistics.median(candidate)
cp = percentile(control, 0.99)
tp = percentile(candidate, 0.99)
ratios = [right / left for left, right in zip(control, candidate)]
print(f"panel={PANEL} warmup_pairs={WARMUP} measured_pairs={PAIRS}")
print(f"signatures={sorted(signatures)}")
print(f"control_ns median={cm:.1f} p99={cp} min={min(control)} max={max(control)}")
print(f"candidate_ns median={tm:.1f} p99={tp} min={min(candidate)} max={max(candidate)}")
print(f"distribution_delta_percent median={(tm / cm - 1) * 100:.6f} p99={(tp / cp - 1) * 100:.6f}")
print(f"paired_ratio_delta_percent median={(statistics.median(ratios) - 1) * 100:.6f} p99={(percentile(ratios, 0.99) - 1) * 100:.6f}")
