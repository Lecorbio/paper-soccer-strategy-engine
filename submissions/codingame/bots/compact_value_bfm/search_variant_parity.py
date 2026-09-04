#!/usr/bin/env python3
"""Require byte-exact fixed-work traces from all four search variants."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess

from feature_parity import fixtures


SCHEMA = "papersoccer.compact-value-bfm-search-trace.v1"
STATE_COUNT = 24
PROFILE_ORDER = ("tie-root", "patterned-deep")
CORPUS_SHA256 = "c5756229a0a239880cc155a0deaa2767265ae91a53aca1e16f59e7ef5c9a7469"
VARIANT_ARGUMENTS = (
    ("baseline", "baseline"),
    ("no-feature-sort-only", "no_feature_sort_only"),
    ("single-pass-selection-only", "single_pass_selection_only"),
    ("combined", "combined"),
)
HEX32 = re.compile(r"[0-9a-f]{8}\Z")
HEX64 = re.compile(r"[0-9a-f]{16}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ParityError(RuntimeError):
    """The search variants did not emit complete identical evidence."""


def frozen_corpus() -> str:
    transcripts, _ = fixtures(STATE_COUNT)
    payload = "\n".join(transcripts) + "\n"
    actual = hashlib.sha256(payload.encode("ascii")).hexdigest()
    if actual != CORPUS_SHA256:
        raise ParityError("frozen search-state corpus identity changed")
    return payload


def _fields(line: str) -> dict[str, str]:
    pieces = line.split("\t")
    if not pieces or pieces[0] != "case":
        raise ParityError("trace row has the wrong record kind")
    fields: dict[str, str] = {}
    for piece in pieces[1:]:
        key, separator, value = piece.partition("=")
        if not separator or not key or key in fields:
            raise ParityError("trace row has malformed or duplicate fields")
        fields[key] = value
    expected = {
        "index",
        "profile",
        "state",
        "canonical",
        "feature",
        "feature_count",
        "action",
        "successor",
        "value",
        "solved",
        "selected_root",
        "top_ties",
        "legal",
        "root_count",
        "stats",
        "roots",
    }
    if set(fields) != expected:
        raise ParityError("trace row field registry changed")
    return fields


def validate_trace(payload: str) -> None:
    lines = payload.splitlines()
    header = f"{SCHEMA}\tstates={STATE_COUNT}\tprofiles=2\tarchitecture=6301x12x8x1"
    if not lines or lines[0] != header:
        raise ParityError("trace header changed")
    rows = lines[1:]
    if len(rows) != STATE_COUNT * len(PROFILE_ORDER):
        raise ParityError("trace has the wrong case count")

    state_identity: dict[int, tuple[str, str, str, str]] = {}
    exercised_tie = False
    for ordinal, line in enumerate(rows):
        fields = _fields(line)
        index = int(fields["index"])
        profile = fields["profile"]
        if index != ordinal // len(PROFILE_ORDER):
            raise ParityError("trace state order changed")
        if profile != PROFILE_ORDER[ordinal % len(PROFILE_ORDER)]:
            raise ParityError("trace profile order changed")
        if not HEX64.fullmatch(fields["state"]) or not HEX64.fullmatch(
            fields["canonical"]
        ):
            raise ParityError("trace state identity is malformed")
        if not SHA256.fullmatch(fields["feature"]):
            raise ParityError("trace feature identity is malformed")
        if int(fields["feature_count"]) <= 0:
            raise ParityError("trace omitted active features")
        if not fields["action"] or any(ch not in "01234567" for ch in fields["action"]):
            raise ParityError("trace selected action is malformed")
        if fields["legal"] != "1" or not HEX64.fullmatch(fields["successor"]):
            raise ParityError("trace lacks legal successor evidence")
        if not HEX32.fullmatch(fields["value"]) or fields["solved"] not in {"0", "1"}:
            raise ParityError("trace value/proof evidence is malformed")

        root_count = int(fields["root_count"])
        selected_root = int(fields["selected_root"])
        top_ties = int(fields["top_ties"])
        roots = fields["roots"].split(";") if fields["roots"] else []
        if root_count <= 0 or len(roots) != root_count:
            raise ParityError("trace root transcript count changed")
        if not 0 <= selected_root < root_count or not 1 <= top_ties <= root_count:
            raise ParityError("trace selection evidence is out of range")
        parsed_roots = [root.split(",") for root in roots]
        if any(len(root) != 10 for root in parsed_roots):
            raise ParityError("trace root transcript shape changed")
        for root in parsed_roots:
            (
                action,
                tactical,
                value,
                initial,
                visits,
                selections,
                solved,
                order,
                score,
                successor,
            ) = root
            if (
                not action
                or any(ch not in "01234567" for ch in action)
                or int(tactical) not in range(5)
                or not HEX32.fullmatch(value)
                or not HEX32.fullmatch(initial)
                or int(visits) <= 0
                or int(selections) < 0
                or solved not in {"0", "1"}
                or int(order) < 0
                or not HEX64.fullmatch(score)
                or not HEX64.fullmatch(successor)
            ):
                raise ParityError("trace root transcript evidence is malformed")
        if parsed_roots[selected_root][0] != fields["action"]:
            raise ParityError("selected root does not match the emitted action")

        stats = [int(value) for value in fields["stats"].split(",")]
        if len(stats) != 26 or any(value < 0 for value in stats):
            raise ParityError("trace search-stat registry changed")
        if stats[23] != 0:
            raise ParityError("fixed-work trace unexpectedly reached a deadline")

        identity = (
            fields["state"],
            fields["canonical"],
            fields["feature"],
            fields["feature_count"],
        )
        if index in state_identity and state_identity[index] != identity:
            raise ParityError("profiles did not search the identical state")
        state_identity[index] = identity
        exercised_tie = exercised_tie or (profile == "tie-root" and top_ties > 1)
    if len(state_identity) != STATE_COUNT or not exercised_tie:
        raise ParityError("trace did not cover every root and an exact top-score tie")


def run_probe(executable: pathlib.Path, corpus: str) -> str:
    completed = subprocess.run(
        [str(executable.resolve())],
        input=corpus,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ParityError(
            f"probe {executable} failed with {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    if completed.stderr:
        raise ParityError(f"probe {executable} emitted unexpected stderr")
    validate_trace(completed.stdout)
    return completed.stdout


def compare(arguments: argparse.Namespace) -> tuple[str, str]:
    corpus = frozen_corpus()
    traces: dict[str, str] = {}
    for variant, attribute in VARIANT_ARGUMENTS:
        traces[variant] = run_probe(getattr(arguments, attribute), corpus)
    baseline = traces["baseline"]
    for variant, trace in traces.items():
        if trace != baseline:
            baseline_lines = baseline.splitlines()
            variant_lines = trace.splitlines()
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(baseline_lines, variant_lines, strict=False)
                    )
                    if left != right
                ),
                min(len(baseline_lines), len(variant_lines)),
            )
            raise ParityError(
                f"{variant} differs from baseline at trace line {mismatch + 1}"
            )
    return CORPUS_SHA256, hashlib.sha256(baseline.encode("ascii")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--no-feature-sort-only", type=pathlib.Path, required=True)
    parser.add_argument("--single-pass-selection-only", type=pathlib.Path, required=True)
    parser.add_argument("--combined", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    corpus_sha, trace_sha = compare(arguments)
    print(
        "compact search-variant parity passed "
        f"variants={len(VARIANT_ARGUMENTS)} states={STATE_COUNT} profiles=2 "
        f"corpus_sha256={corpus_sha} trace_sha256={trace_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
