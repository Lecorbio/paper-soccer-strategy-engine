#!/usr/bin/env python3
"""Prove subtree-reuse-v1 skips work without changing fixed-work semantics."""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from search_variant_parity import ParityError, _fields, frozen_corpus, run_probe


CACHE_STATS = (4, 5, 6)
WIDENING_STATS = (7, 8, 9, 10)
REUSE_STATS = (11, 12, 13, 14, 15)
GENERATOR_WORK_STATS = (16, 17, 18, 19, 20)


def parse_stats(fields: dict[str, str]) -> list[int]:
    result = [int(value) for value in fields["stats"].split(",")]
    if len(result) != 26:
        raise ParityError("subtree reuse received the wrong stat registry")
    return result


def normalized_stats(stats: list[int]) -> tuple[int, ...]:
    result = list(stats)
    for index in (*REUSE_STATS, *GENERATOR_WORK_STATS):
        result[index] = 0
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--reused", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    corpus = frozen_corpus()
    baseline = run_probe(arguments.baseline, corpus)
    reused = run_probe(arguments.reused, corpus)
    repeated = run_probe(arguments.reused, corpus)
    if reused != repeated:
        raise ParityError("subtree reuse is not fixed-work deterministic")

    baseline_rows = [_fields(line) for line in baseline.splitlines()[1:]]
    reused_rows = [_fields(line) for line in reused.splitlines()[1:]]
    if len(baseline_rows) != len(reused_rows):
        raise ParityError("subtree reuse changed the case registry")

    totals = [0, 0, 0, 0, 0]
    baseline_generator_work = 0
    reused_generator_work = 0
    for baseline_fields, reused_fields in zip(
        baseline_rows, reused_rows, strict=True
    ):
        baseline_stats = parse_stats(baseline_fields)
        reused_stats = parse_stats(reused_fields)
        if any(
            baseline_stats[index]
            for index in (*CACHE_STATS, *WIDENING_STATS, *REUSE_STATS)
        ):
            raise ParityError("baseline unexpectedly enabled an intervention")
        if any(reused_stats[index] for index in (*CACHE_STATS, *WIDENING_STATS)):
            raise ParityError("subtree reuse implicitly enabled another profile")

        baseline_semantics = dict(baseline_fields)
        reused_semantics = dict(reused_fields)
        baseline_semantics.pop("stats")
        reused_semantics.pop("stats")
        if baseline_semantics != reused_semantics:
            raise ParityError(
                "subtree reuse changed state identity, legality, scores, ties, "
                "inference, visits, or ordered root transcripts"
            )
        if normalized_stats(baseline_stats) != normalized_stats(reused_stats):
            raise ParityError("subtree reuse changed logical search accounting")

        evidence = [reused_stats[index] for index in REUSE_STATS]
        probes, hits, misses, rejections, children = evidence
        if probes != hits + misses + rejections or (hits == 0 and children != 0):
            raise ParityError("subtree reuse evidence counters are inconsistent")
        for index, value in enumerate(evidence):
            totals[index] += value
        baseline_generator_work += sum(
            baseline_stats[index] for index in GENERATOR_WORK_STATS
        )
        reused_generator_work += sum(
            reused_stats[index] for index in GENERATOR_WORK_STATS
        )

    probes, hits, misses, rejections, children = totals
    if min(probes, hits, misses, rejections, children) <= 0:
        raise ParityError("frozen corpus did not exercise every reuse outcome")
    if reused_generator_work >= baseline_generator_work:
        raise ParityError("subtree reuse did not avoid deterministic generator work")
    trace_sha = hashlib.sha256(reused.encode("ascii")).hexdigest()
    print(
        "compact subtree-reuse-v1 invariance passed "
        f"probes={probes} hits={hits} misses={misses} "
        f"rejections={rejections} children={children} "
        f"generator_work_saved={baseline_generator_work - reused_generator_work} "
        f"trace_sha256={trace_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
