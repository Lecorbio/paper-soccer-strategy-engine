#!/usr/bin/env python3
"""Compare 4,096 deterministic legal states with the independent encoder."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_features as reference  # noqa: E402


STATE_COUNT = 4096
TRANSCRIPT_SHA256 = "0797e9b0b2739c951faa0467b04d82c1ea297f2a96d5b464c34295855a5a88d7"
FEATURES_SHA256 = "842f9213aa11d0346fa8e36e8f3772b3c2689d3c9ae912ffab85e388deadc224"


def choose_turn(state: reference.ReplayState, game: int, turn: int) -> str:
    mover = state.to_move
    action = []
    primitive = 0
    while state.winner is None and state.to_move == mover:
        legal = []
        for direction, (dx, dy) in enumerate(reference.DIRECTION_DELTAS):
            destination = state.ball[0] + dx, state.ball[1] + dy
            if reference._legal_destination(state, destination):
                legal.append(direction)
        if not legal:
            raise RuntimeError("reference state has no legal primitive")
        key = f"compact-value-bfm:{game}:{turn}:{primitive}:{state.ball}:{len(state.used_segments)}"
        selected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        direction = legal[selected % len(legal)]
        action.append(str(direction))
        reference.apply_primitive(state, direction)
        primitive += 1
    return "".join(action)


def fixtures(count: int) -> tuple[list[str], list[tuple[int, ...]]]:
    transcripts = []
    features = []
    game = 0
    while len(transcripts) < count:
        state = reference.ReplayState()
        turns: list[str] = []
        for turn in range(96):
            if state.winner is not None or len(transcripts) >= count:
                break
            transcripts.append("/".join(turns))
            features.append(reference.encode_active(state))
            turns.append(choose_turn(state, game, turn))
        game += 1
    return transcripts, features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--states", type=int, default=STATE_COUNT)
    arguments = parser.parse_args()
    if arguments.states < STATE_COUNT:
        parser.error(f"parity requires at least {STATE_COUNT} states")
    transcripts, expected = fixtures(arguments.states)
    completed = subprocess.run(
        [str(arguments.probe.resolve())],
        input="\n".join(transcripts) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != len(expected):
        raise RuntimeError("feature probe returned the wrong row count")
    aggregate = hashlib.sha256()
    for index, (line, wanted) in enumerate(zip(lines, expected, strict=True)):
        actual = tuple(int(value) for value in line.split(","))
        if actual != wanted:
            raise RuntimeError(f"feature mismatch at frozen state {index}")
        aggregate.update(len(actual).to_bytes(2, "little"))
        for value in actual:
            aggregate.update(value.to_bytes(2, "little"))
    transcript_sha = hashlib.sha256(
        ("\n".join(transcripts) + "\n").encode("ascii")).hexdigest()
    features_sha = aggregate.hexdigest()
    if arguments.states == STATE_COUNT and (
        transcript_sha != TRANSCRIPT_SHA256 or features_sha != FEATURES_SHA256
    ):
        raise RuntimeError("frozen 4,096-state parity corpus identity changed")
    print(
        f"compact feature parity passed states={len(expected)} "
        f"transcript_sha256={transcript_sha} "
        f"features_sha256={features_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
