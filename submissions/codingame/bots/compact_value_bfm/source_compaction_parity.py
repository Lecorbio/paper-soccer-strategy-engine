#!/usr/bin/env python3
"""Bind modular and minified traces to the pre-intervention semantic golden."""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from search_variant_parity import ParityError, _fields, frozen_corpus, run_probe


GOLDEN_COMMIT = "43d9ec91df82eeea1a23b9ad7d48c40cf6ceb672"
# This is the 24-state/two-profile trace emitted by that commit's standalone
# submission, with the current row schema and its original 14 SearchStats
# fields. Current-only intervention counters are removed before hashing.
GOLDEN_SEMANTIC_SHA256 = (
    "db5e97b614e9d3f61007e09e3bf6e0d43e5b42297d0e2e26683cc6952356f56e"
)
INTERVENTION_STATS = tuple(range(4, 16))
PRE_INTERVENTION_STATS = (*range(4), *range(16, 26))


def semantic_trace(payload: str) -> str:
    lines = payload.splitlines()
    output = [lines[0]]
    for line in lines[1:]:
        fields = _fields(line)
        stats = [int(value) for value in fields["stats"].split(",")]
        if len(stats) != 26:
            raise ParityError("source parity received the wrong stat registry")
        if any(stats[index] for index in INTERVENTION_STATS):
            raise ParityError("standard source unexpectedly enabled an intervention")
        pieces = line.split("\t")
        position = next(
            index for index, piece in enumerate(pieces) if piece.startswith("stats=")
        )
        pieces[position] = "stats=" + ",".join(
            str(stats[index]) for index in PRE_INTERVENTION_STATS
        )
        output.append("\t".join(pieces))
    return "\n".join(output) + ("\n" if payload.endswith("\n") else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modular", type=pathlib.Path, required=True)
    parser.add_argument(
        "--generated", type=pathlib.Path, action="append", required=True
    )
    arguments = parser.parse_args()

    corpus = frozen_corpus()
    modular = run_probe(arguments.modular, corpus)
    traces = [run_probe(path, corpus) for path in arguments.generated]
    for trace in traces:
        if trace != modular:
            raise ParityError("unminified modular and generated traces differ")
    normalized = semantic_trace(modular)
    digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()
    if digest != GOLDEN_SEMANTIC_SHA256:
        raise ParityError(
            f"standard semantics differ from pre-intervention commit {GOLDEN_COMMIT}"
        )
    print(
        "compact source compaction parity passed "
        f"generated_variants={len(traces)} golden_commit={GOLDEN_COMMIT} "
        f"semantic_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
