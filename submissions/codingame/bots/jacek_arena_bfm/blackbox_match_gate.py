#!/usr/bin/env python3
"""Run source-isolated local matches against the exact frozen H62 bot.

The H62 source is an opaque compiler input.  This program never decodes,
parses, copies, embeds, or prints it.  A separate digest utility attests its
identity, the compiler receives its path directly, and the referee interacts
with the resulting process only through the public CodinGame protocol.

No action transcript is written to the report.  The report contains only
source identities, aggregate/per-game outcomes, operational categories, and
wall-clock timing evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any


BOT_DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = BOT_DIRECTORY.parents[3]
DEFAULT_CANDIDATE_SOURCE = BOT_DIRECTORY / "submission.cpp"
DEFAULT_H62_SOURCE = (
    REPOSITORY
    / "submissions"
    / "codingame"
    / "bots"
    / "jacek_native_bfm"
    / "submission.cpp"
)

H62_SOURCE_SHA256 = (
    "d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71"
)
H62_SOURCE_BYTES = 99_810
EXPECTED_INTERNAL_FIRST_MS = 800
EXPECTED_INTERNAL_LATER_MS = 155
MAXIMUM_ACTION_LENGTH = 316
REPORT_SCHEMA = "papersoccer.jacek-arena-bfm.blackbox-match-gate.v1"

DIRECTIONS = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


class GateError(RuntimeError):
    pass


class RuleViolation(RuntimeError):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


class BotFailure(RuntimeError):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


def _regular(x: int, y: int) -> bool:
    return 0 <= x <= 8 and 1 <= y <= 11


def _goal(x: int, y: int) -> bool:
    return 3 <= x <= 5 and y in (0, 12)


def _boundary(x: int, y: int) -> bool:
    return _regular(x, y) and (
        x in (0, 8) or (y in (1, 11) and x != 4)
    )


def _permitted_edge(first: tuple[int, int], second: tuple[int, int]) -> bool:
    ax, ay = first
    bx, by = second
    if max(abs(ax - bx), abs(ay - by)) != 1:
        return False
    if _regular(ax, ay) and _regular(bx, by):
        if _boundary(ax, ay) and _boundary(bx, by):
            if ay == by and ay in (1, 11):
                return False
            if ax == bx and ax in (0, 8):
                return False
        return True
    if _goal(ax, ay) == _goal(bx, by):
        return False
    field_x, field_y = (ax, ay) if _regular(ax, ay) else (bx, by)
    goal_x, goal_y = (ax, ay) if _goal(ax, ay) else (bx, by)
    if not (
        3 <= field_x <= 5
        and ((field_y == 1 and goal_y == 0) or (field_y == 11 and goal_y == 12))
    ):
        return False
    return not (field_x == goal_x and field_x in (3, 5))


class Topology:
    """The campaign's public 105-vertex, 316-edge Paper Soccer board."""

    def __init__(self) -> None:
        coordinates = [(x, 0) for x in range(3, 6)]
        coordinates.extend((x, y) for y in range(1, 12) for x in range(9))
        coordinates.extend((x, 12) for x in range(3, 6))
        if len(coordinates) != 105:
            raise AssertionError("Paper Soccer topology must have 105 vertices")
        self.coordinates = tuple(coordinates)
        self.vertex_by_coordinate = {
            coordinate: vertex for vertex, coordinate in enumerate(self.coordinates)
        }
        edges: list[tuple[int, int]] = []
        for first in range(len(self.coordinates)):
            for second in range(first + 1, len(self.coordinates)):
                if _permitted_edge(self.coordinates[first], self.coordinates[second]):
                    edges.append((first, second))
        if len(edges) != 316:
            raise AssertionError("Paper Soccer topology must have 316 edges")
        self.edges = tuple(edges)
        self.edge_by_vertices = {
            (first, second): edge for edge, (first, second) in enumerate(self.edges)
        }
        self.arcs: list[dict[int, tuple[int, int]]] = [dict() for _ in coordinates]
        self.incident_edges: list[list[int]] = [[] for _ in coordinates]
        for edge, (first, second) in enumerate(self.edges):
            ax, ay = self.coordinates[first]
            bx, by = self.coordinates[second]
            forward = DIRECTIONS.index((bx - ax, by - ay))
            backward = DIRECTIONS.index((ax - bx, ay - by))
            self.arcs[first][forward] = (edge, second)
            self.arcs[second][backward] = (edge, first)
            self.incident_edges[first].append(edge)
            self.incident_edges[second].append(edge)
        self.boundaries = tuple(_boundary(*coordinate) for coordinate in coordinates)
        self.initial_ball = self.vertex_by_coordinate[(4, 6)]

    def vertex(self, x: int, y: int) -> int:
        return self.vertex_by_coordinate[(x, y)]

    def edge(self, first: int, second: int) -> int:
        return self.edge_by_vertices[tuple(sorted((first, second)))]


TOPOLOGY = Topology()


@dataclasses.dataclass
class State:
    used: set[int] = dataclasses.field(default_factory=set)
    ball: int = TOPOLOGY.initial_ball
    to_move: int = 0
    winner: int | None = None
    ply: int = 0

    def clone(self) -> "State":
        return State(set(self.used), self.ball, self.to_move, self.winner, self.ply)


def initial_state() -> State:
    return State()


def _vertex_visited(state: State, vertex: int) -> bool:
    return TOPOLOGY.boundaries[vertex] or any(
        edge in state.used for edge in TOPOLOGY.incident_edges[vertex]
    )


def _legal_arc_count(state: State) -> int:
    return sum(
        edge not in state.used for edge, _ in TOPOLOGY.arcs[state.ball].values()
    )


def _apply_edge(state: State, direction: int) -> None:
    if state.winner is not None or direction not in TOPOLOGY.arcs[state.ball]:
        raise RuleViolation("illegal-edge")
    edge, destination = TOPOLOGY.arcs[state.ball][direction]
    if edge in state.used:
        raise RuleViolation("reused-edge")
    mover = state.to_move
    rebound = _vertex_visited(state, destination)
    state.used.add(edge)
    state.ball = destination
    state.ply += 1
    _, y = TOPOLOGY.coordinates[destination]
    if y in (0, 12):
        attacking = y == 0 if mover == 0 else y == 12
        state.winner = mover if attacking else 1 - mover
        return
    if not rebound:
        state.to_move = 1 - mover
    if _legal_arc_count(state) == 0:
        state.winner = 1 - mover


def apply_complete_action(state: State, action: str) -> State:
    """Validate and atomically apply one mandatory-complete rebound turn."""

    if state.winner is not None:
        raise RuleViolation("action-after-terminal")
    if not isinstance(action, str) or not (1 <= len(action) <= MAXIMUM_ACTION_LENGTH):
        raise RuleViolation("invalid-output")
    if re.fullmatch(r"[0-7]+", action) is None:
        raise RuleViolation("invalid-output")
    result = state.clone()
    mover = result.to_move
    for index, character in enumerate(action):
        if result.winner is not None or result.to_move != mover:
            raise RuleViolation("overlong-complete-turn")
        _apply_edge(result, ord(character) - ord("0"))
        if index + 1 < len(action) and (
            result.winner is not None or result.to_move != mover
        ):
            raise RuleViolation("overlong-complete-turn")
    if result.winner is None and result.to_move == mover:
        raise RuleViolation("mandatory-rebound-omitted")
    state.used = result.used
    state.ball = result.ball
    state.to_move = result.to_move
    state.winner = result.winner
    state.ply = result.ply
    return state


def _opaque_sha256(path: pathlib.Path) -> tuple[str, str]:
    """Hash with a separate utility; source bytes never enter this process."""

    shasum = shutil.which("shasum")
    sha256sum = shutil.which("sha256sum")
    if shasum is not None:
        command = [shasum, "-a", "256", "--", os.fspath(path)]
        utility = "shasum"
    elif sha256sum is not None:
        command = [sha256sum, "--", os.fspath(path)]
        utility = "sha256sum"
    else:
        raise GateError("no external SHA-256 attestation utility is available")
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        diagnostic_hash = hashlib.sha256(completed.stderr).hexdigest()
        raise GateError(
            "opaque SHA-256 attestation failed "
            f"(diagnostics_sha256={diagnostic_hash})"
        )
    match = re.match(rb"^([0-9a-f]{64})(?:[ \t]|$)", completed.stdout)
    if match is None:
        raise GateError("opaque SHA-256 utility returned a malformed identity")
    return match.group(1).decode("ascii"), utility


def attest_source(
    path: pathlib.Path,
    *,
    role: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise GateError(f"{role} source must be a regular, non-symlink file")
    size = resolved.stat().st_size
    digest, utility = _opaque_sha256(resolved)
    if expected_bytes is not None and size != expected_bytes:
        raise GateError(f"{role} source byte count does not match the frozen identity")
    if expected_sha256 is not None and digest != expected_sha256:
        raise GateError(f"{role} source SHA-256 does not match the frozen identity")
    return {
        "role": role,
        "bytes": size,
        "sha256": digest,
        "digest_utility": utility,
        "source_opened_by_referee": False,
        "source_parsed_by_referee": False,
        "source_content_emitted": False,
    }


def compile_opaque_source(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    compiler: str,
    role: str,
) -> dict[str, Any]:
    flags = ["-std=c++20", "-O3", "-DNDEBUG", "-pipe"]
    command = [compiler, *flags, os.fspath(source.resolve()), "-o", os.fspath(output)]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    diagnostics = completed.stdout + completed.stderr
    diagnostics_sha256 = hashlib.sha256(diagnostics).hexdigest()
    if completed.returncode != 0:
        # Never echo compiler diagnostics: an opaque compiler may quote source.
        raise GateError(
            f"{role} opaque compilation failed with return code "
            f"{completed.returncode} (diagnostics_sha256={diagnostics_sha256})"
        )
    if not output.is_file():
        raise GateError(f"{role} compiler did not create an executable")
    output.chmod(0o700)
    return {
        "role": role,
        "flags": flags,
        "returncode": completed.returncode,
        "diagnostics_bytes": len(diagnostics),
        "diagnostics_sha256": diagnostics_sha256,
        "source_passed_directly_to_compiler": True,
        "linked_with_other_bot": False,
    }


class ProtocolBot:
    """One opaque CodinGame protocol subprocess."""

    def __init__(self, executable: pathlib.Path, label: str) -> None:
        self.label = label
        try:
            self.process = subprocess.Popen(
                [os.fspath(executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as error:
            raise BotFailure("spawn-failure") from error
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise BotFailure("pipe-failure")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        os.set_blocking(self.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()

    def initialize(self, player_id: int) -> None:
        self._write(f"{player_id}\n".encode("ascii"))

    def _write(self, payload: bytes) -> None:
        try:
            self.stdin.write(payload)
            self.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            raise BotFailure("broken-input-pipe") from error

    def _read_line(self, timeout_ms: float) -> bytes:
        deadline = time.perf_counter() + max(timeout_ms, 0.0) / 1000.0
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                return line
            if len(self.buffer) > 4096:
                raise BotFailure("oversized-output")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BotFailure("timeout")
            events = self.selector.select(remaining)
            if not events:
                raise BotFailure("timeout")
            try:
                chunk = os.read(self.stdout.fileno(), 4096)
            except BlockingIOError:
                continue
            except OSError as error:
                raise BotFailure("output-pipe-failure") from error
            if not chunk:
                raise BotFailure(
                    "partial-output-eof" if self.buffer else "empty-output-eof"
                )
            self.buffer.extend(chunk)

    def request(self, opponent_action: str | None, timeout_ms: float) -> str:
        if opponent_action is None:
            request = b"0\n-\n"
        else:
            request = f"{len(opponent_action)}\n{opponent_action}\n".encode("ascii")
        self._write(request)
        raw = self._read_line(timeout_ms)
        try:
            action = raw.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise BotFailure("non-ascii-output") from error
        if re.fullmatch(r"[0-7]{1,316}", action) is None:
            raise BotFailure("malformed-output")
        return action

    def close(self) -> None:
        selector = getattr(self, "selector", None)
        if selector is not None:
            try:
                selector.close()
            except OSError:
                pass
        process = getattr(self, "process", None)
        if process is None:
            return
        stdin = getattr(self, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=0.05)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.terminate()
            process.wait(timeout=0.20)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=0.20)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _timing_digest(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_ms": None, "p50_ms": None,
                "p95_ms": None, "p99_ms": None, "max_ms": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


def _empty_timings() -> dict[str, dict[str, list[float]]]:
    return {
        "candidate": {"first": [], "later": []},
        "h62": {"first": [], "later": []},
    }


def play_game(
    game_index: int,
    *,
    candidate_executable: pathlib.Path,
    h62_executable: pathlib.Path,
    candidate_player: int,
    first_timeout_ms: float,
    later_timeout_ms: float,
    process_factory: Callable[[pathlib.Path, str], Any] = ProtocolBot,
) -> dict[str, Any]:
    if candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    state = initial_state()
    labels = ["h62", "h62"]
    labels[candidate_player] = "candidate"
    executables = [h62_executable, h62_executable]
    executables[candidate_player] = candidate_executable
    bots: list[Any | None] = [None, None]
    previous_action: str | None = None
    decisions = [0, 0]
    timings = _empty_timings()
    failure: dict[str, Any] | None = None
    try:
        for _ in range(MAXIMUM_ACTION_LENGTH):
            player = state.to_move
            label = labels[player]
            first = bots[player] is None
            hard_timeout = first_timeout_ms if first else later_timeout_ms
            started = time.perf_counter()
            try:
                if first:
                    bots[player] = process_factory(executables[player], label)
                    bots[player].initialize(player)
                construction_ms = (time.perf_counter() - started) * 1000.0
                remaining_ms = hard_timeout - construction_ms
                if remaining_ms <= 0:
                    raise BotFailure("timeout")
                action = bots[player].request(previous_action, remaining_ms)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if elapsed_ms > hard_timeout:
                    raise BotFailure("timeout")
                apply_complete_action(state, action)
            except BotFailure as error:
                failure = {
                    "side": label,
                    "player": player,
                    "category": error.category,
                }
                state.winner = 1 - player
                break
            except RuleViolation as error:
                failure = {
                    "side": label,
                    "player": player,
                    "category": error.category,
                }
                state.winner = 1 - player
                break
            timings[label]["first" if first else "later"].append(elapsed_ms)
            decisions[player] += 1
            previous_action = action
            if state.winner is not None:
                break
        else:
            failure = {
                "side": "referee",
                "player": None,
                "category": "decision-limit",
            }
    finally:
        for bot in bots:
            if bot is not None:
                bot.close()

    if state.winner is None:
        candidate_result = "referee-failure"
    elif failure is not None:
        candidate_result = (
            "operational-loss"
            if failure["side"] == "candidate"
            else "operational-win"
            if failure["side"] == "h62"
            else "referee-failure"
        )
    else:
        candidate_result = "win" if state.winner == candidate_player else "loss"
    game_timings = {
        label: {
            phase: _timing_digest(values)
            for phase, values in phases.items()
        }
        for label, phases in timings.items()
    }
    return {
        "game_index": game_index,
        "candidate_player": candidate_player,
        "winner_player": state.winner,
        "candidate_result": candidate_result,
        "clean": failure is None and state.winner is not None,
        "complete_actions": sum(decisions),
        "physical_edges": state.ply,
        "decisions_by_player": decisions,
        "failure": failure,
        "timing": game_timings,
        "_raw_timings": timings,
    }


def run_games(
    *,
    games: int,
    workers: int,
    seed: int,
    candidate_executable: pathlib.Path,
    h62_executable: pathlib.Path,
    first_timeout_ms: float,
    later_timeout_ms: float,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if games <= 0 or games % 2 != 0:
        raise GateError("games must be a positive even number for balanced colors")
    if workers <= 0:
        raise GateError("workers must be positive")
    if first_timeout_ms <= 0 or later_timeout_ms <= 0:
        raise GateError("decision timeouts must be positive")

    def one(index: int) -> dict[str, Any]:
        candidate_player = (index + (seed & 1)) & 1
        return play_game(
            index,
            candidate_executable=candidate_executable,
            h62_executable=h62_executable,
            candidate_player=candidate_player,
            first_timeout_ms=first_timeout_ms,
            later_timeout_ms=later_timeout_ms,
        )

    results: list[dict[str, Any]] = []
    if workers == 1:
        for index in range(games):
            results.append(one(index))
            if progress is not None:
                progress(len(results), games)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(one, index) for index in range(games)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                if progress is not None:
                    progress(len(results), games)
    results.sort(key=lambda game: game["game_index"])
    return results


def summarize_matches(
    games: Sequence[Mapping[str, Any]],
    *,
    requested_games: int,
    profile: str,
    profile_exact: bool,
    workers: int,
    seed: int,
    first_timeout_ms: float,
    later_timeout_ms: float,
    candidate_source: Mapping[str, Any],
    h62_source: Mapping[str, Any],
    builds: Sequence[Mapping[str, Any]],
    compiler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    clean = [game for game in games if game["clean"]]
    candidate_wins = sum(game["candidate_result"] == "win" for game in clean)
    candidate_losses = sum(game["candidate_result"] == "loss" for game in clean)
    color_counts = Counter(int(game["candidate_player"]) for game in games)
    failure_counts = Counter(
        f"{game['failure']['side']}:{game['failure']['category']}"
        for game in games
        if game["failure"] is not None
    )
    raw_timing = _empty_timings()
    for game in games:
        for label in ("candidate", "h62"):
            for phase in ("first", "later"):
                raw_timing[label][phase].extend(game["_raw_timings"][label][phase])
    timing = {
        label: {
            phase: _timing_digest(values)
            for phase, values in phases.items()
        }
        for label, phases in raw_timing.items()
    }
    report_games = [
        {key: value for key, value in game.items() if key != "_raw_timings"}
        for game in games
    ]
    exact_game_count = len(games) == requested_games
    balanced = color_counts[0] == color_counts[1] == requested_games // 2
    zero_operational_failures = not failure_counts
    beats_h62 = candidate_wins > candidate_losses
    timing_within_hard_limits = all(
        summary[phase]["max_ms"] is None
        or summary[phase]["max_ms"]
        <= (first_timeout_ms if phase == "first" else later_timeout_ms)
        for summary in timing.values()
        for phase in ("first", "later")
    )
    gate_pass = (
        exact_game_count
        and profile_exact
        and balanced
        and zero_operational_failures
        and len(clean) == requested_games
        and beats_h62
        and timing_within_hard_limits
    )
    return {
        "schema": REPORT_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "profile": profile,
        "profile_exact": profile_exact,
        "configuration": {
            "requested_games": requested_games,
            "workers": workers,
            "seed": seed,
            "balanced_colors": True,
            "construction_inclusive_first_decision": True,
            "expected_bot_work_budget_ms": {
                "first": EXPECTED_INTERNAL_FIRST_MS,
                "later": EXPECTED_INTERNAL_LATER_MS,
            },
            "referee_hard_timeout_ms": {
                "first": first_timeout_ms,
                "later": later_timeout_ms,
            },
            "action_transcripts_retained": False,
        },
        "isolation": {
            "h62_source_decoded": False,
            "h62_source_parsed": False,
            "h62_source_content_emitted": False,
            "h62_stdout_retained": False,
            "h62_stderr_retained": False,
            "action_transcripts_retained": False,
            "sources_compiled_separately": True,
            "binaries_linked_together": False,
            "candidate_imported_h62_content": False,
        },
        "sources": {"candidate": dict(candidate_source), "h62": dict(h62_source)},
        "compiler": dict(compiler_identity),
        "builds": [dict(build) for build in builds],
        "results": {
            "completed_games": len(games),
            "clean_games": len(clean),
            "candidate_wins": candidate_wins,
            "candidate_losses": candidate_losses,
            "candidate_score": (
                candidate_wins / len(clean) if clean else None
            ),
            "candidate_color_counts": {
                "player_0": color_counts[0],
                "player_1": color_counts[1],
            },
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "timing": timing,
        "gate": {
            "exact_game_count": exact_game_count,
            "balanced_colors": balanced,
            "zero_operational_failures": zero_operational_failures,
            "beats_h62": beats_h62,
            "timing_within_hard_limits": timing_within_hard_limits,
            "pass": gate_pass,
        },
        "games": report_games,
    }


PROFILES = {
    "fast-1000": {
        "games": 1000,
        "workers": min(8, max(1, os.cpu_count() or 1)),
        # Parallel games retain the bots' frozen 800/155 work clocks.  The
        # wider referee ceiling absorbs scheduler contention; this is a fast
        # strength screen, not the same-runtime timing qualification.
        "first_timeout_ms": 1500.0,
        "later_timeout_ms": 300.0,
    },
    "actual-212": {
        "games": 212,
        "workers": 1,
        "first_timeout_ms": 1000.0,
        "later_timeout_ms": 200.0,
    },
    "custom": {
        "games": 2,
        "workers": 1,
        "first_timeout_ms": 1000.0,
        "later_timeout_ms": 200.0,
    },
}


def _compiler_identity(compiler: str) -> dict[str, Any]:
    completed = subprocess.run(
        [compiler, "--version"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False
    )
    payload = completed.stdout + completed.stderr
    first_line = payload.splitlines()[0].decode("ascii", errors="replace") \
        if payload.splitlines() else "unknown"
    return {
        "executable": pathlib.PurePath(compiler).name,
        "version_first_line": first_line,
        "version_output_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _progress(completed: int, total: int) -> None:
    if completed == total or completed == 1 or completed % max(1, total // 20) == 0:
        print(f"blackbox-match completed={completed}/{total}", file=sys.stderr, flush=True)


def _canonical_json(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="custom")
    parser.add_argument("--candidate-source", type=pathlib.Path,
                        default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--h62-source", type=pathlib.Path,
                        default=DEFAULT_H62_SOURCE)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--games", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--first-timeout-ms", type=float)
    parser.add_argument("--later-timeout-ms", type=float)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()

    defaults = PROFILES[arguments.profile]
    games = arguments.games if arguments.games is not None else defaults["games"]
    workers = (
        arguments.workers if arguments.workers is not None else defaults["workers"]
    )
    first_timeout = (
        arguments.first_timeout_ms
        if arguments.first_timeout_ms is not None
        else defaults["first_timeout_ms"]
    )
    later_timeout = (
        arguments.later_timeout_ms
        if arguments.later_timeout_ms is not None
        else defaults["later_timeout_ms"]
    )
    profile_exact = (
        games == defaults["games"]
        and workers == defaults["workers"]
        and first_timeout == defaults["first_timeout_ms"]
        and later_timeout == defaults["later_timeout_ms"]
    )
    try:
        candidate_path = arguments.candidate_source.resolve(strict=True)
        h62_path = arguments.h62_source.resolve(strict=True)
        if candidate_path == h62_path:
            raise GateError("candidate and H62 source paths must be distinct")
        candidate_source = attest_source(
            candidate_path,
            role="candidate",
            expected_sha256=arguments.candidate_sha256,
        )
        h62_source = attest_source(
            h62_path,
            role="frozen-h62",
            expected_sha256=H62_SOURCE_SHA256,
            expected_bytes=H62_SOURCE_BYTES,
        )
        with tempfile.TemporaryDirectory(prefix="jacek_arena_blackbox_") as directory:
            build_directory = pathlib.Path(directory)
            candidate_executable = build_directory / "candidate"
            h62_executable = build_directory / "frozen_h62"
            candidate_build = compile_opaque_source(
                candidate_path, candidate_executable,
                compiler=arguments.cxx, role="candidate"
            )
            h62_build = compile_opaque_source(
                h62_path, h62_executable,
                compiler=arguments.cxx, role="frozen-h62"
            )
            games_result = run_games(
                games=games,
                workers=workers,
                seed=arguments.seed,
                candidate_executable=candidate_executable,
                h62_executable=h62_executable,
                first_timeout_ms=first_timeout,
                later_timeout_ms=later_timeout,
                progress=_progress,
            )
            report = summarize_matches(
                games_result,
                requested_games=games,
                profile=arguments.profile,
                profile_exact=profile_exact,
                workers=workers,
                seed=arguments.seed,
                first_timeout_ms=first_timeout,
                later_timeout_ms=later_timeout,
                candidate_source=candidate_source,
                h62_source=h62_source,
                builds=[candidate_build, h62_build],
                compiler_identity=_compiler_identity(arguments.cxx),
            )
        encoded = _canonical_json(report)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_bytes(encoded)
        sys.stdout.buffer.write(encoded)
        return 0 if report["gate"]["pass"] else 2
    except (GateError, OSError, ValueError) as error:
        parser.exit(1, f"blackbox match gate failure: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
