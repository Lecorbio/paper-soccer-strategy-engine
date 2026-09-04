#!/usr/bin/env python3
"""Prove state-evaluation-cache-v1 preserves the complete search trace."""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from search_variant_parity import ParityError, _fields, frozen_corpus, run_probe


CACHE_STAT_INDICES = (4, 5, 6)


def normalized(payload: str, *, cached: bool) -> tuple[str, tuple[int, int, int]]:
    lines = payload.splitlines()
    totals = [0, 0, 0]
    output = [lines[0]]
    for line in lines[1:]:
        fields = _fields(line)
        stats = [int(value) for value in fields["stats"].split(",")]
        if len(stats) != 26:
            raise ParityError("cache parity received the wrong stat registry")
        cache = [stats[index] for index in CACHE_STAT_INDICES]
        if cache[0] != cache[1] + cache[2]:
            raise ParityError("cache probes do not equal hits plus misses")
        if not cached and cache != [0, 0, 0]:
            raise ParityError("baseline unexpectedly enabled the evaluation cache")
        for index, value in enumerate(cache):
            totals[index] += value
        for index in CACHE_STAT_INDICES:
            stats[index] = 0
        pieces = line.split("\t")
        pieces[next(
            index for index, piece in enumerate(pieces) if piece.startswith("stats=")
        )] = "stats=" + ",".join(str(value) for value in stats)
        output.append("\t".join(pieces))
    suffix = "\n" if payload.endswith("\n") else ""
    return "\n".join(output) + suffix, tuple(totals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--cached", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    corpus = frozen_corpus()
    baseline = run_probe(arguments.baseline, corpus)
    candidate = run_probe(arguments.cached, corpus)
    baseline_normalized, baseline_cache = normalized(baseline, cached=False)
    candidate_normalized, candidate_cache = normalized(candidate, cached=True)
    probes, hits, misses = candidate_cache
    if probes <= 0 or misses <= 0:
        raise ParityError("cache mode produced no lookup/miss evidence")
    if hits <= 0:
        raise ParityError("frozen corpus did not exercise a cache hit")
    if baseline_cache != (0, 0, 0):
        raise ParityError("baseline cache evidence changed")
    if candidate_normalized != baseline_normalized:
        left = baseline_normalized.splitlines()
        right = candidate_normalized.splitlines()
        mismatch = next(
            (
                index
                for index, (baseline_line, candidate_line) in enumerate(
                    zip(left, right, strict=False)
                )
                if baseline_line != candidate_line
            ),
            min(len(left), len(right)),
        )
        raise ParityError(
            "state-evaluation-cache-v1 changed semantic trace line "
            f"{mismatch + 1}"
        )
    trace_sha = hashlib.sha256(baseline_normalized.encode("ascii")).hexdigest()
    print(
        "compact state-evaluation-cache-v1 parity passed "
        f"probes={probes} hits={hits} misses={misses} trace_sha256={trace_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
