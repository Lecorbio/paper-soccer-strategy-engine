"""Declared long-turn protocol coverage using unchanged production binaries."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import select
import subprocess
import time

from tools import jacek_replay_features as rules
from tools import rank_two_focused_campaign_v1 as campaign


def centre_turn(state, direction_offset):
    """Avoid goals while filling the centre; retain every considered primitive."""
    mover = state.to_move
    action, proposals = "", []
    while state.winner is None and state.to_move == mover:
        choices = []
        for direction in range(8):
            child = copy.deepcopy(state)
            try:
                rules.apply_primitive(child, direction)
            except ValueError:
                continue
            distance = (child.ball[0] - 4)**2 + (child.ball[1] - 6)**2
            free = sum(rules._segment(child.ball, rules.POINTS[destination])
                       not in child.used_segments
                       for destination, _ in rules.ADJACENCY[rules.POINT_INDEX[child.ball]])
            key = (child.winner is not None, distance, -free,
                   (direction - direction_offset) % 8)
            choices.append((key, direction, child))
        if not choices:
            raise ValueError("nonterminal reference position has no legal move")
        choices.sort(key=lambda row: row[0])
        proposals.append(dict(partial=action, alternatives=[
            dict(direction=d, priority=list(key)) for key, d, _ in choices]))
        _, direction, state = choices[0]
        action += str(direction)
    return state, action, proposals


def game(build, color, direction_offset, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    campaign.immutable(output / "CLAIM.json", dict(
        build=build, color=color, direction_offset=direction_offset,
        kind="declared-centre-seeking-mechanical-coverage", recipe=campaign.record(__file__)))
    binary = campaign.verify(build["binary"])
    campaign.verify(build["source"])
    state, transcript, responses, proposals = rules.ReplayState(), [], [], []
    first, last_opponent, failure = True, "-", None
    # Construct the first opponent input before launch, so the cold-response
    # timer charges construction/loading and excludes the harness's own work.
    if color == 1:
        state, last_opponent, choices = centre_turn(state, direction_offset)
        proposals.append(dict(prefix="", choices=choices))
        transcript.append(last_opponent)
    started = time.monotonic()
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    try:
        process.stdin.write(str(color) + "\n")
        process.stdin.flush()
        for _ in range(rules.EDGE_COUNT + 1):
            if state.winner is not None:
                break
            if state.to_move != color:
                prefix = "/".join(transcript)
                state, last_opponent, choices = centre_turn(state, direction_offset)
                proposals.append(dict(prefix=prefix, choices=choices))
                transcript.append(last_opponent)
                continue
            prefix = "/".join(transcript)
            response_started = started if first else time.monotonic()
            process.stdin.write(str(0 if last_opponent == "-" else len(last_opponent))
                               + "\n" + last_opponent + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], 1.1 if first else .25)
            if not ready:
                raise TimeoutError("production response missing")
            action = process.stdout.readline().rstrip("\r\n")
            elapsed = 1000 * (time.monotonic() - response_started)
            limit = 550 if first else 140
            responses.append(dict(prefix=prefix, action=action, elapsed_ms=elapsed,
                                  limit_ms=limit, first=first))
            if elapsed > limit:
                raise TimeoutError("whole-response envelope exceeded")
            if not action:
                raise ValueError("empty or EOF response")
            rules.apply_complete_turn(state, color, action)
            transcript.append(action)
            first = False
            campaign.atomic(output / "PROGRESS.json", dict(
                transcript=transcript, responses=responses, proposals=proposals, complete=False))
        if state.winner is None:
            raise ValueError("finite edge bound exceeded")
        process.stdin.close()
        process.wait(timeout=3)
        if process.returncode:
            raise ValueError("production process returned nonzero")
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
        if process.poll() is None:
            process.kill()
        process.wait()
    finally:
        campaign.immutable(output / "stderr.txt", process.stderr.read().encode())
    result = dict(source=build["source"], binary=build["binary"], color=color,
                  transcript="/".join(transcript), responses=responses, proposals=proposals,
                  complete=state.winner is not None and failure is None, failure=failure,
                  first_max_ms=max((r["elapsed_ms"] for r in responses if r["first"]), default=0),
                  later_max_ms=max((r["elapsed_ms"] for r in responses if not r["first"]), default=0),
                  longest_response_edges=max((len(r["action"]) for r in responses), default=0),
                  training_eligible=False, strength_eligible=False)
    campaign.immutable(output / "RESULT.json", result)
    return result


def certify(plan_path, output):
    plan = campaign.read(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if plan["recipe"] != campaign.record(__file__):
        raise ValueError("coverage recipe changed")
    build = campaign.read(campaign.verify(plan["build"]))
    prior = campaign.read(campaign.verify(plan["prior_safety"]))
    if prior["own_failures"] != 0 or prior["mechanical_games"] != 6:
        raise ValueError("required six complete mechanical games differ")
    receipts, failed = [], False
    for slot in plan["slots"]:
        if slot["profile"] != "dd8" or slot["color"] not in (0, 1):
            raise ValueError("unexpected declared coverage slot")
        if failed:
            campaign.immutable(output / slot["id"] / "CANCELLED_UNCLAIMED.json",
                               dict(reason="prior operational failure", slot=slot, claimed=False))
            continue
        result = game(build["builds"][slot["profile"]], slot["color"],
                      slot["direction_offset"], output / slot["id"])
        failed = result["failure"] is not None
        receipts.append(dict(profile=slot["profile"], color=slot["color"],
                             receipt=campaign.record(output / slot["id"] / "RESULT.json"),
                             complete=result["complete"], failure=result["failure"],
                             longest_response_edges=result["longest_response_edges"]))
    coverage = any(row["longest_response_edges"] >= 8 for row in receipts)
    result = dict(schema=campaign.SCHEMA + ".long-turn-main-coverage", passed=(
        not failed and len(receipts) == len(plan["slots"]) and coverage),
        isolated=True, source_modified=False, build=plan["build"], plan=campaign.record(plan_path),
        prior_safety=plan["prior_safety"], games=receipts, own_failures=int(failed),
        dd8_long_turn_observed=coverage, training_eligible=False, strength_eligible=False)
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=campaign.SCHEMA + ".mechanical-game-exposure", report=campaign.record(output / "RESULT.json"),
        game_receipts=[row["receipt"] for row in receipts], recipe=campaign.record(__file__),
        training_eligible=False, before_next_fresh_bank=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(certify(args.plan, args.output)))


if __name__ == "__main__":
    main()
