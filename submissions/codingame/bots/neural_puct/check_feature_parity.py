#!/usr/bin/env python3

"""Require exact float32 parity between Python and C++ feature encoders."""

from __future__ import annotations

import argparse
import csv
import io
import pathlib
import subprocess

import numpy as np

import train_neural_puct as trainer


ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_PROBE = ROOT / "build" / "papersoccer_codingame_neural_puct_feature_probe"


def state_after(actions: tuple[str, ...]):
    ball = (trainer.WIDTH // 2, trainer.HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    player = 0
    for action in actions:
        for index, encoded in enumerate(action):
            direction = int(encoded)
            dx, dy = trainer.DIRECTIONS[direction]
            destination = ball[0] + dx, ball[1] + dy
            edge = trainer.segment(ball, destination)
            if edge not in trainer.EDGE_INDEX or edge in used_segments:
                raise RuntimeError("parity fixture contains an illegal edge")
            bounced = destination in visited or trainer.is_boundary(destination)
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
            if not bounced:
                player = 1 - player
            if index + 1 < len(action) and not bounced:
                raise RuntimeError("parity fixture crosses a possession boundary")
        if bounced:
            raise RuntimeError("parity fixture omits a mandatory rebound")
    return ball, used_segments, visited, player


def python_fixtures():
    initial = (4, 6), set(), {(4, 6)}, 0
    transcript = state_after(("0", "5", "21"))
    return {
        "initial_player_one": trainer.feature_vector(
            *initial[:3], initial[3], False
        ),
        "initial_player_two_rotated": trainer.feature_vector(
            trainer.rotate(initial[0]),
            set(),
            {trainer.rotate(point) for point in initial[2]},
            1,
            False,
        ),
        "transcript_0_5_21": trainer.feature_vector(
            *transcript[:3], transcript[3], False
        ),
        "transcript_0_3_67_reflected": trainer.feature_vector(
            *transcript[:3], transcript[3], True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=pathlib.Path, default=DEFAULT_PROBE)
    arguments = parser.parse_args()
    output = subprocess.run(
        [str(arguments.probe)], check=True, text=True, capture_output=True
    ).stdout
    actual = {
        row[0]: np.asarray([float.fromhex(value) for value in row[1:]], dtype=np.float32)
        for row in csv.reader(io.StringIO(output))
    }
    expected = python_fixtures()
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"C++ feature fixtures differ: {sorted(actual)} != {sorted(expected)}"
        )
    compared = 0
    for name, values in expected.items():
        observed = actual[name]
        if observed.shape != values.shape:
            raise RuntimeError(f"{name} feature count differs")
        mismatches = np.flatnonzero(
            observed.view(np.uint32) != values.view(np.uint32)
        )
        if len(mismatches):
            first = int(mismatches[0])
            raise RuntimeError(
                f"{name} has {len(mismatches)} bit mismatches; first at {first}: "
                f"Python={float(values[first]).hex()}, C++={float(observed[first]).hex()}"
            )
        compared += len(values)
    print(f"Feature parity is exact ({compared}/{compared} float32 values).")


if __name__ == "__main__":
    main()
