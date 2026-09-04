#!/usr/bin/env python3
"""Verify deterministic progressive widening preserves search invariants."""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from search_variant_parity import ParityError, _fields, frozen_corpus, run_probe


CACHE_STAT_INDICES = (4, 5, 6)
WIDENING_STAT_INDICES = (7, 8, 9, 10)


def _stats(fields: dict[str, str]) -> list[int]:
    values = [int(value) for value in fields["stats"].split(",")]
    if len(values) != 26:
        raise ParityError("progressive widening received the wrong stat registry")
    return values


def _roots(fields: dict[str, str]) -> list[list[str]]:
    roots = [entry.split(",") for entry in fields["roots"].split(";")]
    if len(roots) != int(fields["root_count"]) or any(
        len(root) != 10 for root in roots
    ):
        raise ParityError("progressive widening root transcript is malformed")
    return roots


def invariant_signature(fields: dict[str, str]) -> tuple[object, ...]:
    roots = _roots(fields)
    return (
        fields["index"],
        fields["profile"],
        fields["state"],
        fields["canonical"],
        fields["feature"],
        fields["feature_count"],
        fields["legal"],
        fields["root_count"],
        tuple((root[0], root[1], root[3], root[7], root[9]) for root in roots),
    )


def allocation_signature(fields: dict[str, str]) -> tuple[object, ...]:
    roots = _roots(fields)
    stats = _stats(fields)
    for index in (*CACHE_STAT_INDICES, *WIDENING_STAT_INDICES):
        stats[index] = 0
    return (
        fields["action"],
        fields["successor"],
        fields["value"],
        fields["solved"],
        fields["selected_root"],
        fields["top_ties"],
        tuple((root[2], root[4], root[5], root[6], root[8]) for root in roots),
        tuple(stats),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--widened", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    corpus = frozen_corpus()
    baseline = run_probe(arguments.baseline, corpus)
    widened = run_probe(arguments.widened, corpus)
    repeated = run_probe(arguments.widened, corpus)
    if widened != repeated:
        raise ParityError("progressive widening is not fixed-work deterministic")

    baseline_rows = [_fields(line) for line in baseline.splitlines()[1:]]
    widened_rows = [_fields(line) for line in widened.splitlines()[1:]]
    if len(baseline_rows) != len(widened_rows):
        raise ParityError("progressive widening changed the case registry")

    totals = [0, 0, 0, 0]
    allocation_changed = False
    for baseline_fields, widened_fields in zip(
        baseline_rows, widened_rows, strict=True
    ):
        if invariant_signature(baseline_fields) != invariant_signature(widened_fields):
            raise ParityError(
                "progressive widening changed legality, generation order, or inference"
            )
        baseline_stats = _stats(baseline_fields)
        widened_stats = _stats(widened_fields)
        if any(baseline_stats[index] for index in CACHE_STAT_INDICES):
            raise ParityError("baseline unexpectedly enabled the value cache")
        if any(baseline_stats[index] for index in WIDENING_STAT_INDICES):
            raise ParityError("baseline unexpectedly enabled progressive widening")
        if any(widened_stats[index] for index in CACHE_STAT_INDICES):
            raise ParityError("progressive widening implicitly enabled the value cache")
        evidence = [widened_stats[index] for index in WIDENING_STAT_INDICES]
        probes, restrictions, eligible, deferred = evidence
        if restrictions > probes or (probes == 0 and (eligible != 0 or deferred != 0)):
            raise ParityError("progressive widening counters are inconsistent")
        for index, value in enumerate(evidence):
            totals[index] += value
        baseline_allocation = allocation_signature(baseline_fields)
        widened_allocation = allocation_signature(widened_fields)
        if (
            baseline_fields["profile"] == "tie-root"
            and baseline_allocation != widened_allocation
        ):
            raise ParityError("progressive widening changed the root-only tie profile")
        allocation_changed = allocation_changed or (
            baseline_allocation != widened_allocation
        )

    probes, restrictions, eligible, deferred = totals
    if probes <= 0 or restrictions <= 0 or eligible <= 0 or deferred <= 0:
        raise ParityError("frozen corpus did not exercise progressive widening")
    if not allocation_changed:
        raise ParityError("progressive widening did not change search allocation")
    trace_sha = hashlib.sha256(widened.encode("ascii")).hexdigest()
    print(
        "compact progressive-widening-v1 invariance passed "
        f"probes={probes} restrictions={restrictions} eligible={eligible} "
        f"deferred={deferred} trace_sha256={trace_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
