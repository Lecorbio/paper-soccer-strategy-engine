#!/usr/bin/env python3
"""Compile-check every allowed profile and every forbidden profile pair."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


CACHE = "COMPACT_VALUE_BFM_STATE_EVALUATION_CACHE_V1"
WIDENING = "COMPACT_VALUE_BFM_PROGRESSIVE_WIDENING_V1"
REUSE = "COMPACT_VALUE_BFM_SUBTREE_REUSE_V1"
PROFILES = (CACHE, WIDENING, REUSE)
FORBIDDEN = (
    (
        CACHE,
        WIDENING,
        "state-evaluation-cache-v1 and progressive-widening-v1 are independent profiles",
    ),
    (CACHE, REUSE, "subtree-reuse-v1 is an independent search profile"),
    (WIDENING, REUSE, "subtree-reuse-v1 is an independent search profile"),
)


def compile_header(
    compiler: pathlib.Path, include_dir: pathlib.Path, macros: tuple[str, ...]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(compiler.resolve()),
            "-std=c++20",
            "-fsyntax-only",
            "-x",
            "c++",
            f"-I{include_dir.resolve()}",
            *(f"-D{macro}=1" for macro in macros),
            "-",
        ],
        input='#include "engine.hpp"\n',
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=pathlib.Path, required=True)
    parser.add_argument("--include-dir", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    for profile in PROFILES:
        result = compile_header(arguments.compiler, arguments.include_dir, (profile,))
        if result.returncode != 0:
            raise RuntimeError(f"standalone profile {profile} did not compile")
    for left, right, expected in FORBIDDEN:
        result = compile_header(
            arguments.compiler, arguments.include_dir, (left, right)
        )
        if result.returncode == 0 or expected not in result.stderr:
            raise RuntimeError(f"forbidden profile pair {left}+{right} was accepted")
    print(
        "compact search profile exclusion passed "
        f"standalone={len(PROFILES)} forbidden_pairs={len(FORBIDDEN)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
