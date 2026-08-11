#!/usr/bin/env python3
"""Validate and summarize ground-up Jacek-native self-play corpora.

The JSONL contract deliberately contains complete games rather than detached
positions.  That makes whole-game train/validation/test splits enforceable and
keeps rotations, reflections, and reanalysis of the same trajectory together.
No expert action is a label: the only mandatory target is the final outcome
from the player-to-move's perspective.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
from collections import Counter, defaultdict, deque
from typing import Iterable, Iterator, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
GAME_SCHEMA = "papersoccer.jacek-native-game/v1"
FEATURE_SCHEMA = "canonical-edges316-onehot-true-turn-distance105x8-v1"
GENERATOR_SCHEMA = "jacek-native-complete-turn-bfm/v1"
BUILD_PROVENANCE_SCHEMA = "papersoccer.jacek-native-build-provenance/v1"
BUILD_PROVENANCE_NAME = "build-provenance.json"
ARCHIVED_BINARY_NAME = "selfplay-binary"
EDGE_COUNT = 316
VERTEX_COUNT = 105
DISTANCE_BUCKETS = 8
INPUT_COUNT = EDGE_COUNT + VERTEX_COUNT * DISTANCE_BUCKETS
RULES = {
    "width": 8,
    "height": 10,
    "goal_rule": "own-goals-allowed",
    "blocked_rule": "mover-loses",
}
DEQUE_SCHEDULE = "nine-lifo-one-fifo"
FORBIDDEN_PROVENANCE = (
    "rank_4",
    "rank-4",
    "rank4",
    "replay-book",
    "replay_book",
    "alpha-beta-teacher",
    "alpha_beta_teacher",
)
BUILD_SOURCE_PATHS = (
    "tools/jacek_native_selfplay.cpp",
    "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
    "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "src/bots/mcts_internal.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
)
CANONICAL_BUILD_ARGV = (
    "$CXX",
    "-std=c++20",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Iinclude",
    "-Isrc/bots",
    "tools/jacek_native_selfplay.cpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "-o",
    "$OUTPUT",
)

Point = tuple[int, int]
Segment = tuple[Point, Point]
DIRECTION_DELTAS: tuple[Point, ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


@dataclasses.dataclass
class _ReplayState:
    """Minimal rules state kept independent of the C++ self-play producer."""

    ball: Point
    to_move: int
    winner: int | None
    used_segments: set[Segment]
    visit_count: dict[Point, int]


def _initial_replay_state() -> _ReplayState:
    ball = (RULES["width"] // 2, RULES["height"] // 2 + 1)
    return _ReplayState(
        ball=ball,
        to_move=0,
        winner=None,
        used_segments=set(),
        visit_count={ball: 1},
    )


def _normalized_segment(first: Point, second: Point) -> Segment:
    first_key = (first[1], first[0])
    second_key = (second[1], second[0])
    return (first, second) if first_key <= second_key else (second, first)


def _field_bottom_y() -> int:
    return RULES["height"] + 1


def _south_goal_y() -> int:
    return RULES["height"] + 2


def _mouth_bounds() -> tuple[int, int]:
    center = RULES["width"] // 2
    return center - 1, center + 1


def _is_regular_point(point: Point) -> bool:
    x, y = point
    return 0 <= x <= RULES["width"] and 1 <= y <= _field_bottom_y()


def _is_goal_point(point: Point) -> bool:
    x, y = point
    mouth_left, mouth_right = _mouth_bounds()
    return mouth_left <= x <= mouth_right and y in (0, _south_goal_y())


def _goal_winner(point: Point) -> int:
    if not _is_goal_point(point):
        raise ValueError("goal winner requested for a non-goal point")
    # Player One attacks north and Player Two attacks south.  With own goals
    # enabled, the geometric goal owner wins regardless of who drew the edge.
    return 0 if point[1] == 0 else 1


def _is_boundary_point(point: Point) -> bool:
    if not _is_regular_point(point):
        return False
    x, y = point
    if x in (0, RULES["width"]):
        return True
    mouth_left, mouth_right = _mouth_bounds()
    on_goal_line = y in (1, _field_bottom_y())
    inside_goal_opening = mouth_left < x < mouth_right
    return on_goal_line and not inside_goal_opening


def _is_forbidden_boundary_segment(first: Point, second: Point) -> bool:
    mouth_left, mouth_right = _mouth_bounds()
    north_post = (
        (first[1], second[1]) in ((0, 1), (1, 0)) and
        first[0] == second[0] and first[0] in (mouth_left, mouth_right)
    )
    south_post = (
        (first[1], second[1]) in (
            (_field_bottom_y(), _south_goal_y()),
            (_south_goal_y(), _field_bottom_y()),
        ) and
        first[0] == second[0] and first[0] in (mouth_left, mouth_right)
    )
    if north_post or south_post:
        return True
    if not (_is_regular_point(first) and _is_regular_point(second)):
        return False
    if not (_is_boundary_point(first) and _is_boundary_point(second)):
        return False
    dx = abs(first[0] - second[0])
    dy = abs(first[1] - second[1])
    horizontal_wall = (
        first[1] == second[1] and first[1] in (1, _field_bottom_y()) and
        dx == 1 and dy == 0
    )
    vertical_wall = (
        first[0] == second[0] and first[0] in (0, RULES["width"]) and
        dx == 0 and dy == 1
    )
    return horizontal_wall or vertical_wall


def _neighbor_points(point: Point) -> Iterator[Point]:
    x, y = point
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if _is_regular_point(candidate):
                yield candidate

    mouth_left, mouth_right = _mouth_bounds()
    if mouth_left <= x <= mouth_right and y in (1, _field_bottom_y()):
        goal_y = 0 if y == 1 else _south_goal_y()
        for goal_x in range(mouth_left, mouth_right + 1):
            candidate = (goal_x, goal_y)
            if abs(x - goal_x) <= 1:
                # RULES fixes OwnGoalsAllowed, so both goal mouths are legal
                # for either mover.
                yield candidate


def _make_feature_topology() -> tuple[
    tuple[Point, ...],
    Mapping[Point, int],
    Mapping[Segment, int],
    tuple[tuple[tuple[int, int], ...], ...],
]:
    """Recreate the producer's canonical vertex/edge insertion ordering."""
    points = [
        (x, y)
        for y in range(1, RULES["height"] + 2)
        for x in range(RULES["width"] + 1)
    ]
    mouth_left, mouth_right = _mouth_bounds()
    for x in range(mouth_left, mouth_right + 1):
        points.extend(((x, 0), (x, _south_goal_y())))
    point_index = {point: index for index, point in enumerate(points)}
    edge_index: dict[Segment, int] = {}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in points]
    for source_index, source in enumerate(points):
        if not _is_regular_point(source):
            continue
        for destination in _neighbor_points(source):
            if _is_forbidden_boundary_segment(source, destination):
                continue
            edge = _normalized_segment(source, destination)
            if edge not in edge_index:
                edge_index[edge] = len(edge_index)
            adjacency[source_index].append(
                (point_index[destination], edge_index[edge])
            )
    if len(points) != VERTEX_COUNT or len(edge_index) != EDGE_COUNT:
        raise RuntimeError(
            f"unexpected native topology: {len(points)} vertices and "
            f"{len(edge_index)} edges"
        )
    return (
        tuple(points),
        point_index,
        edge_index,
        tuple(tuple(arcs) for arcs in adjacency),
    )


(
    _FEATURE_POINTS,
    _FEATURE_POINT_INDEX,
    _FEATURE_EDGE_INDEX,
    _FEATURE_ADJACENCY,
) = _make_feature_topology()


def _transform_feature_point(
    point: Point, player: int, reflected: bool
) -> Point:
    x, y = point
    if player == 1:
        x, y = RULES["width"] - x, RULES["height"] + 2 - y
    if reflected:
        x = RULES["width"] - x
    return x, y


def _turn_distances(
    ball: Point, used_edges: set[int], visited: set[Point]
) -> tuple[int, ...]:
    unreachable = 1_000_000
    distances = [unreachable] * VERTEX_COUNT
    start = _FEATURE_POINT_INDEX[ball]
    distances[start] = 0
    pending = deque((start,))
    while pending:
        vertex = pending.popleft()
        for destination, edge in _FEATURE_ADJACENCY[vertex]:
            if edge in used_edges:
                continue
            destination_point = _FEATURE_POINTS[destination]
            rebounds = (
                destination_point in visited or
                _is_boundary_point(destination_point) or
                _is_goal_point(destination_point)
            )
            candidate = distances[vertex] + (0 if rebounds else 1)
            if candidate >= distances[destination]:
                continue
            distances[destination] = candidate
            if rebounds:
                pending.appendleft(destination)
            else:
                pending.append(destination)
    return tuple(min(distance, DISTANCE_BUCKETS - 1) for distance in distances)


def _encode_replay_features(
    state: _ReplayState, reflected: bool = False
) -> tuple[int, ...]:
    transform = lambda point: _transform_feature_point(
        point, state.to_move, reflected
    )
    ball = transform(state.ball)
    transformed_segments = {
        _normalized_segment(transform(first), transform(second))
        for first, second in state.used_segments
    }
    transformed_visited = {
        transform(point)
        for point, count in state.visit_count.items()
        if count > 0
    }
    try:
        used_edges = {
            _FEATURE_EDGE_INDEX[segment]
            for segment in transformed_segments
        }
    except KeyError as error:
        raise ValueError("replayed state contains a non-canonical edge") from error
    distances = _turn_distances(ball, used_edges, transformed_visited)
    active = sorted(used_edges)
    active.extend(
        EDGE_COUNT + vertex * DISTANCE_BUCKETS + distance
        for vertex, distance in enumerate(distances)
    )
    return tuple(active)


def _is_legal_destination(state: _ReplayState, destination: Point) -> bool:
    if state.winner is not None:
        return False
    if destination not in _neighbor_points(state.ball):
        return False
    segment = _normalized_segment(state.ball, destination)
    return (
        segment not in state.used_segments and
        not _is_forbidden_boundary_segment(state.ball, destination)
    )


def _has_legal_move(state: _ReplayState) -> bool:
    return any(
        _is_legal_destination(state, destination)
        for destination in _neighbor_points(state.ball)
    )


def _apply_primitive(state: _ReplayState, direction: str) -> None:
    dx, dy = DIRECTION_DELTAS[ord(direction) - ord("0")]
    destination = (state.ball[0] + dx, state.ball[1] + dy)
    if not _is_legal_destination(state, destination):
        raise ValueError("illegal primitive direction")

    mover = state.to_move
    extra_turn = (
        _is_boundary_point(destination) or
        state.visit_count.get(destination, 0) > 0
    )
    state.used_segments.add(_normalized_segment(state.ball, destination))
    state.ball = destination
    state.visit_count[destination] = state.visit_count.get(destination, 0) + 1

    if _is_goal_point(destination):
        state.winner = _goal_winner(destination)
        return

    state.to_move = mover if extra_turn else 1 - mover
    if not _has_legal_move(state):
        # The CodinGame contract is MoverLoses, including when the landing
        # would otherwise have handed the turn to the opponent.
        state.winner = 1 - mover


def _apply_complete_turn(
    state: _ReplayState,
    action: str,
    turn: int,
    line_number: int,
    opening: bool,
) -> None:
    context = "opening" if opening else "game"
    if state.winner is not None:
        raise ValueError(
            f"{context} transcript continues after terminal result before "
            f"turn {turn} on line {line_number}"
        )
    mover = state.to_move
    for primitive, direction in enumerate(action):
        if state.winner is not None:
            raise ValueError(
                f"{context} complete turn {turn} continues after a terminal "
                f"result at primitive {primitive} on line {line_number}"
            )
        if state.to_move != mover:
            raise ValueError(
                f"{context} complete turn {turn} continues after handoff at "
                f"primitive {primitive} on line {line_number}"
            )
        try:
            _apply_primitive(state, direction)
        except ValueError as error:
            raise ValueError(
                f"illegal primitive {primitive} in {context} complete turn "
                f"{turn} on line {line_number}"
            ) from error

    if state.winner is None and state.to_move == mover:
        raise ValueError(
            f"{context} complete turn {turn} ends before rebound chain "
            f"completion on line {line_number}"
        )


def _replay_recorded_game(
    actions: Sequence[str],
    opening_depth: int,
    recorded_winner: int,
    line_number: int,
) -> _ReplayState:
    """Replay the opening and game suffix using an independent rules model."""
    state, _ = _replay_recorded_game_with_features(
        actions, opening_depth, recorded_winner, line_number, ()
    )
    return state


def _replay_recorded_game_with_features(
    actions: Sequence[str],
    opening_depth: int,
    recorded_winner: int,
    line_number: int,
    feature_turns: Iterable[int],
) -> tuple[_ReplayState, dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Replay once and independently encode requested turn boundaries."""
    requested_turns = set(feature_turns)
    captured: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    state = _initial_replay_state()
    for turn, action in enumerate(actions):
        if turn in requested_turns:
            captured[turn] = (
                _encode_replay_features(state),
                _encode_replay_features(state, reflected=True),
            )
        _apply_complete_turn(
            state,
            action,
            turn,
            line_number,
            opening=turn < opening_depth,
        )
        if turn + 1 == opening_depth and state.winner is not None:
            raise ValueError(
                f"recorded procedural opening is terminal on line {line_number}"
            )
    if state.winner is None:
        raise ValueError(f"game transcript is nonterminal on line {line_number}")
    if state.winner != recorded_winner:
        raise ValueError(
            f"game transcript winner {state.winner} does not match recorded "
            f"winner {recorded_winner} on line {line_number}"
        )
    missing = requested_turns - set(captured)
    if missing:
        raise ValueError(
            f"sample turns are absent from replay on line {line_number}: "
            + ",".join(str(turn) for turn in sorted(missing))
        )
    return state, captured


@dataclasses.dataclass(frozen=True)
class NativeSample:
    game_key: str
    split_group: str
    turn: int
    player: int
    active: tuple[int, ...]
    outcome: float
    auxiliary_value: float | None
    exact: bool
    symmetry: str

    @property
    def fingerprint(self) -> bytes:
        payload = bytearray()
        for index in self.active:
            payload.extend(index.to_bytes(2, "little"))
        return hashlib.sha256(payload).digest()


@dataclasses.dataclass(frozen=True)
class NativeModelArtifact:
    artifact_sha256: str
    model_sha256: str
    packed_sha256: str


@dataclasses.dataclass(frozen=True)
class NativeGame:
    key: str
    split_group: str
    seed: int
    game: int
    shard_index: int
    winner: int
    samples: tuple[NativeSample, ...]
    producer_sha256: str
    build_provenance_sha256: str
    model_artifacts: tuple[NativeModelArtifact, ...]
    search_stats: Mapping[str, int]
    opening_depth: int
    temperature_turns: int
    transcript_sha256: str
    build_contract: Mapping[str, object] | None = None

    @property
    def model_artifact_sha256(self) -> tuple[str, ...]:
        return tuple(artifact.artifact_sha256 for artifact in self.model_artifacts)


def _game_sort_key(game: NativeGame) -> tuple[int, int, str, str]:
    return game.seed, game.game, game.key, game.transcript_sha256


def _sample_sort_key(
    sample: NativeSample,
) -> tuple[str, int, int, tuple[int, ...]]:
    symmetry = 0 if sample.symmetry == "identity" else 1
    return sample.game_key, sample.turn, symmetry, sample.active


def _require_integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _check_purity(record: Mapping[str, object], line_number: int) -> None:
    rendered = json.dumps(record, sort_keys=True, separators=(",", ":")).lower()
    for forbidden in FORBIDDEN_PROVENANCE:
        if forbidden in rendered:
            raise ValueError(
                f"forbidden non-native provenance {forbidden!r} on line "
                f"{line_number}"
            )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _validate_build_contract(
    raw: bytes,
    directory: pathlib.Path,
    verify_local_build: bool,
) -> tuple[str, dict]:
    contract = json.loads(raw)
    if raw != _canonical_json_bytes(contract):
        raise ValueError(
            f"{directory / BUILD_PROVENANCE_NAME} is not canonical JSON"
        )
    if not isinstance(contract, dict) or set(contract) != {
        "schema",
        "binary",
        "compiler",
        "build_argv",
        "producer_sha256",
        "sources",
    }:
        raise ValueError("build provenance fields are not frozen")
    if contract.get("schema") != BUILD_PROVENANCE_SCHEMA:
        raise ValueError("build provenance schema is not frozen")
    if contract.get("build_argv") != list(CANONICAL_BUILD_ARGV):
        raise ValueError("build provenance argv is not frozen")

    binary = contract.get("binary")
    if (
        not isinstance(binary, dict)
        or set(binary) != {"path", "sha256"}
        or binary.get("path") != ARCHIVED_BINARY_NAME
        or not _valid_sha256(binary.get("sha256"))
    ):
        raise ValueError("build provenance binary identity is invalid")
    compiler = contract.get("compiler")
    if (
        not isinstance(compiler, dict)
        or set(compiler)
        != {"executable", "sha256", "version", "version_sha256"}
        or not isinstance(compiler.get("executable"), str)
        or not compiler["executable"]
        or pathlib.PurePath(compiler["executable"]).name
        != compiler["executable"]
        or not _valid_sha256(compiler.get("sha256"))
        or not isinstance(compiler.get("version"), str)
        or not compiler["version"]
        or len(compiler["version"]) > 16_384
        or not _valid_sha256(compiler.get("version_sha256"))
        or hashlib.sha256(compiler["version"].encode()).hexdigest()
        != compiler["version_sha256"]
    ):
        raise ValueError("build provenance compiler identity is invalid")

    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) != len(BUILD_SOURCE_PATHS):
        raise ValueError("build provenance source list is incomplete")
    source_pairs: list[list[str]] = []
    for expected_path, entry in zip(BUILD_SOURCE_PATHS, sources):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry.get("path") != expected_path
            or not _valid_sha256(entry.get("sha256"))
        ):
            raise ValueError("build provenance source identity is invalid")
        source_pairs.append([entry["path"], entry["sha256"]])
    producer = hashlib.sha256(json.dumps(
        source_pairs, separators=(",", ":")
    ).encode()).hexdigest()
    if contract.get("producer_sha256") != producer:
        raise ValueError("build provenance producer SHA-256 is inconsistent")

    rendered = raw.decode("utf-8").lower()
    forbidden = set(FORBIDDEN_PROVENANCE) | {
        "matches.json", "protected-bank", "protected_bank",
        "sealed-bank", "sealed_bank", "/users/", "/home/", "\\users\\",
    }
    home = str(pathlib.Path.home()).lower()
    if any(token in rendered for token in forbidden) or (
        home and home in rendered
    ):
        raise ValueError("build provenance contains a forbidden dependency or path")

    if verify_local_build:
        for entry in sources:
            path = ROOT / entry["path"]
            if not path.is_file() or hashlib.sha256(
                path.read_bytes()
            ).hexdigest() != entry["sha256"]:
                raise ValueError(
                    f"build provenance source is stale: {entry['path']}"
                )
        archived_binary = directory / binary["path"]
        if not archived_binary.is_file() or hashlib.sha256(
            archived_binary.read_bytes()
        ).hexdigest() != binary["sha256"]:
            raise ValueError("build provenance archived binary is stale")
        resolved_compiler = shutil.which(compiler["executable"])
        if resolved_compiler is None:
            raise ValueError("build provenance compiler is unavailable")
        resolved_path = pathlib.Path(resolved_compiler).resolve()
        version = subprocess.run(
            [str(resolved_path), "--version"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if (
            hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            != compiler["sha256"]
            or version != compiler["version"]
        ):
            raise ValueError("build provenance compiler identity is stale")

    return hashlib.sha256(raw).hexdigest(), contract


def validate_active(active_value: object, label: str) -> tuple[int, ...]:
    if not isinstance(active_value, list):
        raise ValueError(f"{label}.active must be an array")
    active = tuple(_require_integer(value, f"{label}.active[]")
                   for value in active_value)
    if not active or active != tuple(sorted(set(active))):
        raise ValueError(f"{label}.active must be sorted and unique")
    if active[-1] >= INPUT_COUNT:
        raise ValueError(f"{label}.active index exceeds {INPUT_COUNT - 1}")
    active_set = set(active)
    for vertex in range(VERTEX_COUNT):
        first = EDGE_COUNT + vertex * DISTANCE_BUCKETS
        if sum(first + bucket in active_set
               for bucket in range(DISTANCE_BUCKETS)) != 1:
            raise ValueError(
                f"{label}.active must select one distance for vertex {vertex}"
            )
    return active


def _validate_generator(value: object, line_number: int) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"generator must be an object on line {line_number}")
    expected = {
        "schema": GENERATOR_SCHEMA,
        "action": "complete-turn",
        "max_actions": 250,
        "deque_schedule": DEQUE_SCHEDULE,
        "work_unit": "maximum-tree-nodes",
        "value_target": "mover-relative-final-outcome",
        "checkpoint_color_schedule":
            "swap-player-checkpoints-on-odd-games",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                f"unexpected generator.{field} on line {line_number}"
            )
    work = _require_integer(value.get("search_work"), "generator.search_work", 1)
    if work <= 0:
        raise ValueError(f"invalid generator work on line {line_number}")
    temperature = value.get("sampling_temperature")
    if (not isinstance(temperature, (int, float)) or
            not math.isfinite(float(temperature)) or float(temperature) < 0.0):
        raise ValueError(
            f"invalid generator.sampling_temperature on line {line_number}"
        )
    temperature_turns = _require_integer(
        value.get("temperature_turns"), "generator.temperature_turns", 0
    )
    if value.get("temperature_schedule") != (
            "absolute-complete-turn-index-before-cutoff/v1"):
        raise ValueError(
            f"invalid generator temperature schedule on line {line_number}"
        )
    if value.get("opening_schema") != (
            "deterministic-procedural-complete-turn-prefix/v1"):
        raise ValueError(f"invalid generator opening schema on line {line_number}")
    opening_depth = _require_integer(
        value.get("opening_depth"), "generator.opening_depth", 0
    )
    opening_retry = _require_integer(
        value.get("opening_retry"), "generator.opening_retry", 0
    )
    if opening_retry >= 4_096:
        raise ValueError(f"generator opening retry is out of range on line {line_number}")
    opening_seed = value.get("opening_seed")
    if (not isinstance(opening_seed, str) or not opening_seed.isdigit() or
            int(opening_seed) >= 1 << 64):
        raise ValueError(f"invalid generator opening seed on line {line_number}")
    transcript = value.get("opening_transcript")
    if not isinstance(transcript, str):
        raise ValueError(f"invalid generator opening transcript on line {line_number}")
    actions = [] if transcript == "" else transcript.split("/")
    if (len(actions) != opening_depth or
            any(not action or any(direction not in "01234567" for direction in action)
                for action in actions)):
        raise ValueError(
            f"generator opening transcript/depth mismatch on line {line_number}"
        )
    if opening_depth == 0 and opening_retry != 0:
        raise ValueError(f"zero-depth opening must use retry zero on line {line_number}")
    producer_sha = value.get("producer_sha256")
    if not _valid_sha256(producer_sha):
        raise ValueError(f"invalid generator producer SHA-256 on line {line_number}")
    if not _valid_sha256(value.get("build_provenance_sha256")):
        raise ValueError(
            f"invalid generator build-provenance SHA-256 on line {line_number}"
        )
    models = value.get("models")
    if not isinstance(models, dict) or set(models) != {"player_one", "player_two"}:
        raise ValueError(f"generator must identify both player models on line {line_number}")
    for player, metadata in models.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid {player} model provenance on line {line_number}")
        for field in ("model_sha256", "packed_sha256", "artifact_sha256"):
            digest = metadata.get(field)
            if not _valid_sha256(digest):
                raise ValueError(
                    f"invalid generator.models.{player}.{field} on line {line_number}"
                )
    stats = value.get("search_stats")
    required_stats = {
        "searches", "expansions", "child_evaluations", "completed_actions",
        "partial_paths", "generator_truncations", "tree_cap_searches",
        "expansion_cap_searches", "tactical_proof_paths",
        "tactical_classes_found", "tactical_proof_truncations",
    }
    if not isinstance(stats, dict) or set(stats) != required_stats:
        raise ValueError(f"invalid generator search stats on line {line_number}")
    for field in required_stats:
        _require_integer(stats[field], f"generator.search_stats.{field}", 0)
    return opening_depth, temperature_turns


def canonical_state_id(active: Sequence[int]) -> str:
    value = 1469598103934665603
    for index in active:
        value ^= index & 0xFF
        value = (value * 1099511628211) & ((1 << 64) - 1)
        value ^= index >> 8
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}"


def validate_record(record: object, line_number: int = 1) -> NativeGame:
    if not isinstance(record, dict):
        raise ValueError(f"record on line {line_number} must be an object")
    _check_purity(record, line_number)
    if record.get("schema") != GAME_SCHEMA:
        raise ValueError(f"unexpected schema on line {line_number}")
    if record.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"unexpected feature schema on line {line_number}")
    if record.get("rules") != RULES:
        raise ValueError(f"unexpected rules contract on line {line_number}")
    opening_depth, temperature_turns = _validate_generator(
        record.get("generator"), line_number
    )

    seed_text = record.get("seed")
    if not isinstance(seed_text, str) or not seed_text.isdigit():
        raise ValueError(f"seed must be an unsigned decimal string on line {line_number}")
    seed = int(seed_text)
    if seed >= 1 << 64:
        raise ValueError(f"seed exceeds uint64 on line {line_number}")
    game_number = _require_integer(record.get("game"), "game", 0)
    shard_index = _require_integer(record.get("shard_index"), "shard_index", 0)
    shard_count = _require_integer(record.get("shard_count"), "shard_count", 1)
    if shard_index >= shard_count or game_number % shard_count != shard_index:
        raise ValueError(f"game/shard assignment mismatch on line {line_number}")
    winner = record.get("winner")
    if winner not in (0, 1):
        raise ValueError(f"winner must be 0 or 1 on line {line_number}")
    complete_turns = _require_integer(
        record.get("complete_turns"), "complete_turns", 1
    )
    if opening_depth >= complete_turns:
        raise ValueError(f"opening consumes the complete game on line {line_number}")
    if record["generator"]["search_stats"]["searches"] != (
            complete_turns - opening_depth):
        raise ValueError(f"search count/opening depth mismatch on line {line_number}")
    if record.get("transcript_schema") != "complete-turn-directions-slash/v1":
        raise ValueError(f"invalid transcript schema on line {line_number}")
    transcript = record.get("transcript")
    if not isinstance(transcript, str):
        raise ValueError(f"game transcript must be a string on line {line_number}")
    transcript_actions = [] if transcript == "" else transcript.split("/")
    if (len(transcript_actions) != complete_turns or
            any(not action or any(direction not in "01234567" for direction in action)
                for action in transcript_actions)):
        raise ValueError(f"game transcript/turn count mismatch on line {line_number}")
    opening_transcript = record["generator"]["opening_transcript"]
    opening_actions = [] if opening_transcript == "" else opening_transcript.split("/")
    if transcript_actions[:opening_depth] != opening_actions:
        raise ValueError(f"game transcript does not begin with its opening on line {line_number}")

    samples_value = record.get("samples")
    if not isinstance(samples_value, list) or not samples_value:
        raise ValueError(f"samples must be non-empty on line {line_number}")
    if len(samples_value) > 100:
        raise ValueError(f"record exceeds 100 samples on line {line_number}")
    sample_turns: list[int] = []
    previous_turn = -1
    for sample_index, sample in enumerate(samples_value):
        label = f"line {line_number} sample {sample_index}"
        if not isinstance(sample, dict):
            raise ValueError(f"{label} must be an object")
        turn = _require_integer(sample.get("turn"), f"{label}.turn", 0)
        if (turn <= previous_turn or turn < opening_depth or
                turn >= complete_turns):
            raise ValueError(f"{label}.turn is not strictly increasing/in range")
        previous_turn = turn
        sample_turns.append(turn)
    _, replayed_features = _replay_recorded_game_with_features(
        transcript_actions, opening_depth, winner, line_number, sample_turns
    )

    game_key = f"native:{seed}:{game_number}"
    split_group = record.get("split_group", game_key)
    if not isinstance(split_group, str) or not split_group:
        raise ValueError(f"invalid split_group on line {line_number}")
    samples: list[NativeSample] = []
    for sample_index, (sample, turn) in enumerate(zip(samples_value, sample_turns)):
        label = f"line {line_number} sample {sample_index}"
        forbidden_labels = set(sample) & {
            "policy",
            "policy_target",
            "teacher_move",
            "expert_move",
            "rank4_value",
        }
        if forbidden_labels:
            raise ValueError(f"{label} contains action/teacher labels")
        player = sample.get("player")
        if player not in (0, 1):
            raise ValueError(f"{label}.player must be 0 or 1")
        if player != turn % 2:
            raise ValueError(f"{label}.player does not match complete-turn parity")
        active = validate_active(sample.get("active"), label)
        if sample.get("canonical_state_id") != canonical_state_id(active):
            raise ValueError(f"{label}.canonical_state_id does not match features")
        expected_active, expected_reflected = replayed_features[turn]
        if active != expected_active:
            raise ValueError(
                f"{label}.active does not match replayed boundary features"
            )
        reflected = validate_active(
            sample.get("reflected_active"), f"{label}.reflected"
        )
        if sample.get("reflected_state_id") != canonical_state_id(reflected):
            raise ValueError(f"{label}.reflected_state_id does not match features")
        if reflected != expected_reflected:
            raise ValueError(
                f"{label}.reflected_active does not match replayed boundary "
                "features"
            )

        auxiliary_value = None
        exact = False
        reanalysis = sample.get("reanalysis")
        if reanalysis is not None:
            if not isinstance(reanalysis, dict):
                raise ValueError(f"{label}.reanalysis must be an object")
            value = reanalysis.get("value")
            if (not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or
                    not -1.0 <= float(value) <= 1.0):
                raise ValueError(f"{label}.reanalysis.value must be in [-1,1]")
            work = _require_integer(
                reanalysis.get("work"), f"{label}.reanalysis.work", 0
            )
            verification_work = _require_integer(
                reanalysis.get("verification_work"),
                f"{label}.reanalysis.verification_work", 0,
            )
            if verification_work < work or verification_work > 100_000:
                raise ValueError(f"{label}.reanalysis budgets are out of order")
            truncated = reanalysis.get("truncated")
            stable = reanalysis.get("stable")
            exact = reanalysis.get("exact", False)
            action_stable = reanalysis.get("action_stable")
            if not all(isinstance(flag, bool) for flag in (
                    truncated, stable, exact, action_stable)):
                raise ValueError(f"{label}.reanalysis flags must be booleans")
            value_delta = reanalysis.get("value_delta")
            if (not isinstance(value_delta, (int, float)) or
                    not math.isfinite(float(value_delta)) or
                    not 0.0 <= float(value_delta) <= 2.0):
                raise ValueError(f"{label}.reanalysis value delta is invalid")
            if stable and not exact and (
                    not action_stable or float(value_delta) > 0.05):
                raise ValueError(f"{label}.reanalysis stability is unsupported")
            if exact and abs(float(value)) != 1.0:
                raise ValueError(
                    f"{label}.reanalysis exact result must be an outcome"
                )
            if stable and not exact and (truncated or work < 30_000):
                raise ValueError(
                    f"{label}.reanalysis is not eligible for stable auxiliary use"
                )
            if stable or exact:
                auxiliary_value = float(value)

        outcome = 1.0 if player == winner else -1.0
        variants = (("identity", active),)
        if reflected != active:
            variants += (("reflection", reflected),)
        for symmetry, variant in variants:
            samples.append(NativeSample(
                game_key=game_key,
                split_group=split_group,
                turn=turn,
                player=player,
                active=variant,
                outcome=outcome,
                auxiliary_value=auxiliary_value,
                exact=exact,
                symmetry=symmetry,
            ))

    model_artifacts = tuple(sorted({
        NativeModelArtifact(
            artifact_sha256=metadata["artifact_sha256"],
            model_sha256=metadata["model_sha256"],
            packed_sha256=metadata["packed_sha256"],
        )
        for metadata in record["generator"]["models"].values()
    }, key=lambda artifact: (
        artifact.artifact_sha256,
        artifact.model_sha256,
        artifact.packed_sha256,
    )))
    return NativeGame(
        key=game_key,
        split_group=split_group,
        seed=seed,
        game=game_number,
        shard_index=shard_index,
        winner=winner,
        samples=tuple(samples),
        producer_sha256=record["generator"]["producer_sha256"],
        build_provenance_sha256=(
            record["generator"]["build_provenance_sha256"]
        ),
        model_artifacts=model_artifacts,
        search_stats=dict(record["generator"]["search_stats"]),
        opening_depth=opening_depth,
        temperature_turns=temperature_turns,
        transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
    )


def load_games(
    paths: Sequence[pathlib.Path], verify_local_build: bool = False
) -> tuple[list[NativeGame], dict]:
    games: list[NativeGame] = []
    source_hashes: dict[str, str] = {}
    seen_keys: set[str] = set()
    directory_contracts: dict[pathlib.Path, tuple[str, dict]] = {}
    for path in paths:
        directory = path.resolve().parent
        contract_identity = directory_contracts.get(directory)
        if contract_identity is None:
            provenance_path = directory / BUILD_PROVENANCE_NAME
            if not provenance_path.is_file():
                raise ValueError(
                    f"missing sibling build provenance: {provenance_path}"
                )
            contract_identity = _validate_build_contract(
                provenance_path.read_bytes(), directory, verify_local_build
            )
            directory_contracts[directory] = contract_identity
        contract_sha256, build_contract = contract_identity
        raw = path.read_bytes()
        source_digest = hashlib.sha256(raw).hexdigest()
        # Content addressing keeps provenance stable when league rounds reuse
        # conventional shard basenames in different run directories.
        source_id = f"sha256:{source_digest}"
        if source_id in source_hashes:
            raise ValueError(f"duplicate corpus shard content {source_id}")
        source_hashes[source_id] = source_digest
        for line_number, raw_line in enumerate(raw.splitlines(), 1):
            if not raw_line.strip():
                continue
            game = validate_record(json.loads(raw_line), line_number)
            if game.build_provenance_sha256 != contract_sha256:
                raise ValueError(
                    f"game build provenance does not match its shard directory "
                    f"on line {line_number}"
                )
            if game.producer_sha256 != build_contract["producer_sha256"]:
                raise ValueError(
                    f"game producer does not match its build provenance on "
                    f"line {line_number}"
                )
            game = dataclasses.replace(game, build_contract=build_contract)
            if game.key in seen_keys:
                raise ValueError(f"duplicate game key {game.key}")
            seen_keys.add(game.key)
            games.append(game)
    if not games:
        raise ValueError("corpus contains no games")
    games.sort(key=_game_sort_key)
    source_hashes = {
        source_id: source_hashes[source_id]
        for source_id in sorted(source_hashes)
    }
    return games, source_hashes


def assign_splits(games: Sequence[NativeGame]) -> dict[str, str]:
    """Assign deterministic, winner-stratified whole-game 80/10/10 splits."""
    grouped: dict[str, list[NativeGame]] = defaultdict(list)
    for game in games:
        grouped[game.split_group].append(game)
    conflicts = {
        group for group, members in grouped.items()
        if len({member.winner for member in members}) > 1
    }
    if conflicts:
        raise ValueError(
            "split_group has inconsistent winners: " + ", ".join(sorted(conflicts))
        )
    by_winner: dict[int, list[str]] = defaultdict(list)
    for group, members in grouped.items():
        by_winner[members[0].winner].append(group)
    assignment: dict[str, str] = {}
    for winner, groups in sorted(by_winner.items()):
        ordered = sorted(
            groups,
            key=lambda value: (hashlib.sha256(value.encode()).digest(), value),
        )
        count = len(ordered)
        train_end = (8 * count) // 10
        validation_end = (9 * count) // 10
        if count >= 3:
            train_end = max(1, min(train_end, count - 2))
            validation_end = max(train_end + 1, min(validation_end, count - 1))
        for index, group in enumerate(ordered):
            assignment[group] = (
                "train" if index < train_end
                else "validation" if index < validation_end
                else "test"
            )
    return assignment


def purge_cross_split_overlaps(
    samples: Mapping[str, Sequence[NativeSample]],
) -> tuple[dict[str, list[NativeSample]], dict[str, int]]:
    """Keep train > validation > test when canonical states cross splits."""
    result: dict[str, list[NativeSample]] = {}
    removed: dict[str, int] = {}
    seen: set[bytes] = set()
    for split in ("train", "validation", "test"):
        retained = []
        removed[split] = 0
        for sample in sorted(samples.get(split, ()), key=_sample_sort_key):
            fingerprint = sample.fingerprint
            if fingerprint in seen:
                removed[split] += 1
            else:
                retained.append(sample)
        seen.update(sample.fingerprint for sample in retained)
        result[split] = retained
    return result, removed


def prepare_splits(games: Sequence[NativeGame]):
    assignment = assign_splits(games)
    samples: dict[str, list[NativeSample]] = {
        "train": [], "validation": [], "test": []
    }
    for game in games:
        samples[assignment[game.split_group]].extend(game.samples)
    return (*purge_cross_split_overlaps(samples), assignment)


def build_contracts(games: Sequence[NativeGame]) -> list[dict]:
    contracts: dict[str, Mapping[str, object]] = {}
    for game in games:
        if game.build_contract is None:
            raise ValueError("game is not bound to a file-backed build contract")
        existing = contracts.get(game.build_provenance_sha256)
        if existing is not None and existing != game.build_contract:
            raise ValueError("build-provenance SHA-256 collision")
        contracts[game.build_provenance_sha256] = game.build_contract
    return [
        {"sha256": digest, "contract": contracts[digest]}
        for digest in sorted(contracts)
    ]


def summarize(
    paths: Sequence[pathlib.Path], verify_local_build: bool = False
) -> dict:
    games, source_hashes = load_games(paths, verify_local_build)
    splits, overlaps_removed, assignment = prepare_splits(games)
    split_games = Counter(assignment[game.split_group] for game in games)
    split_winners = {
        split: dict(Counter(
            game.winner for game in games
            if assignment[game.split_group] == split
        ))
        for split in ("train", "validation", "test")
    }
    search_stats = {
        field: sum(game.search_stats[field] for game in games)
        for field in games[0].search_stats
    }
    model_artifacts = sorted({
        artifact
        for game in games
        for artifact in game.model_artifacts
    }, key=lambda artifact: (
        artifact.artifact_sha256,
        artifact.model_sha256,
        artifact.packed_sha256,
    ))
    builds = build_contracts(games)
    return {
        "schema": "papersoccer.jacek-native-corpus-report/v1",
        "game_schema": GAME_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "rules": RULES,
        "sources": source_hashes,
        "games": len(games),
        "samples": sum(len(game.samples) for game in games),
        "split_games": dict(split_games),
        "split_samples": {name: len(value) for name, value in splits.items()},
        "split_winners": split_winners,
        "cross_split_overlaps_removed": overlaps_removed,
        "stable_reanalysis_samples": sum(
            sample.auxiliary_value is not None
            for game in games for sample in game.samples
        ),
        "generation": {
            "producer_sha256": sorted({game.producer_sha256 for game in games}),
            "build_provenance_sha256": [
                build["sha256"] for build in builds
            ],
            "build_contracts": builds,
            "model_artifact_sha256": sorted({
                artifact.artifact_sha256 for artifact in model_artifacts
            }),
            "model_artifacts": [dataclasses.asdict(artifact)
                                for artifact in model_artifacts],
            "search_stats": search_stats,
            "opening_depths": dict(Counter(game.opening_depth for game in games)),
            "temperature_turns": sorted({game.temperature_turns for game in games}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Jacek-native JSONL self-play shards."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--verify-local-build", action="store_true")
    arguments = parser.parse_args()
    report = summarize(arguments.corpus, arguments.verify_local_build)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
