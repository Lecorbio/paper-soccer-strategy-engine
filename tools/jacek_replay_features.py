#!/usr/bin/env python3
"""Reference feature encoder for the Jacek replay BFM value network.

The implementation is intentionally independent from the native bot.  It is
used to validate generated training shards and to produce parity fixtures for
the C++ runtime.  Inputs are mover-relative and contain 316 used-edge flags
followed by one of 57 categories for each of the 105 canonical vertices.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterable, Iterator, Sequence


WIDTH = 8
HEIGHT = 10
EDGE_COUNT = 316
VERTEX_COUNT = 105
VERTEX_CATEGORIES = 57
INPUT_COUNT = EDGE_COUNT + VERTEX_COUNT * VERTEX_CATEGORIES
FEATURE_SCHEMA = (
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree"
)

DIRECTION_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)

Point = tuple[int, int]
Segment = tuple[Point, Point]


def _regular(point: Point) -> bool:
    x, y = point
    return 0 <= x <= WIDTH and 1 <= y <= HEIGHT + 1


def _goal(point: Point) -> bool:
    x, y = point
    return 3 <= x <= 5 and y in (0, HEIGHT + 2)


def _boundary(point: Point) -> bool:
    if not _regular(point):
        return False
    x, y = point
    if x in (0, WIDTH):
        return True
    return y in (1, HEIGHT + 1) and x != WIDTH // 2


def _segment(first: Point, second: Point) -> Segment:
    return tuple(sorted((first, second), key=lambda point: (point[1], point[0])))  # type: ignore[return-value]


def _forbidden_boundary_segment(first: Point, second: Point) -> bool:
    a, b = _segment(first, second)
    if _goal(a) or _goal(b):
        return (
            a[0] == b[0]
            and a[0] in (3, 5)
            and {a[1], b[1]} in ({0, 1}, {HEIGHT + 1, HEIGHT + 2})
        )
    if not (_regular(a) and _regular(b) and _boundary(a) and _boundary(b)):
        return False
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (
        a[1] == b[1]
        and a[1] in (1, HEIGHT + 1)
        and dx == 1
        and dy == 0
    ) or (
        a[0] == b[0]
        and a[0] in (0, WIDTH)
        and dx == 0
        and dy == 1
    )


def _neighbors(point: Point) -> Iterator[Point]:
    x, y = point
    if not _regular(point):
        return
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = x + dx, y + dy
            if _regular(candidate):
                yield candidate
    if y in (1, HEIGHT + 1) and 3 <= x <= 5:
        goal_y = 0 if y == 1 else HEIGHT + 2
        for goal_x in range(3, 6):
            if abs(x - goal_x) <= 1:
                yield goal_x, goal_y


def _make_topology():
    points = [(x, y) for y in range(1, HEIGHT + 2) for x in range(WIDTH + 1)]
    for x in range(3, 6):
        points.extend(((x, 0), (x, HEIGHT + 2)))
    point_index = {point: index for index, point in enumerate(points)}
    edge_index: dict[Segment, int] = {}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in points]
    for source_index, source in enumerate(points):
        if not _regular(source):
            continue
        for destination in _neighbors(source):
            if _forbidden_boundary_segment(source, destination):
                continue
            edge = _segment(source, destination)
            if edge not in edge_index:
                edge_index[edge] = len(edge_index)
            adjacency[source_index].append(
                (point_index[destination], edge_index[edge])
            )
    if len(points) != VERTEX_COUNT or len(edge_index) != EDGE_COUNT:
        raise RuntimeError(
            f"unexpected topology: {len(points)} vertices, {len(edge_index)} edges"
        )
    edges: list[Segment | None] = [None] * len(edge_index)
    for edge, index in edge_index.items():
        edges[index] = edge
    return (
        tuple(points),
        point_index,
        tuple(edges),
        edge_index,
        tuple(tuple(arcs) for arcs in adjacency),
    )


POINTS, POINT_INDEX, EDGES, EDGE_INDEX, ADJACENCY = _make_topology()


def rotate_point(point: Point) -> Point:
    return WIDTH - point[0], HEIGHT + 2 - point[1]


def reflect_point(point: Point) -> Point:
    return WIDTH - point[0], point[1]


def _build_transform_maps(transform):
    vertices = tuple(POINT_INDEX[transform(point)] for point in POINTS)
    edges = tuple(
        EDGE_INDEX[_segment(transform(edge[0]), transform(edge[1]))]
        for edge in EDGES
        if edge is not None
    )
    return vertices, edges


ROTATED_VERTICES, ROTATED_EDGES = _build_transform_maps(rotate_point)
REFLECTED_VERTICES, REFLECTED_EDGES = _build_transform_maps(reflect_point)


@dataclasses.dataclass
class ReplayState:
    ball: Point = (WIDTH // 2, HEIGHT // 2 + 1)
    to_move: int = 0
    winner: int | None = None
    used_segments: set[Segment] = dataclasses.field(default_factory=set)
    visit_count: dict[Point, int] = dataclasses.field(
        default_factory=lambda: {(WIDTH // 2, HEIGHT // 2 + 1): 1}
    )


def _free_degrees(used_edges: set[int]) -> tuple[int, ...]:
    return tuple(
        sum(edge not in used_edges for _, edge in arcs)
        for arcs in ADJACENCY
    )


def _turn_distances(
    ball: Point, used_edges: set[int], visited: set[Point]
) -> tuple[int, ...]:
    """Return minimum complete-turn distance using a deterministic 0-1 BFS."""

    unreachable = 1_000_000
    distances = [unreachable] * VERTEX_COUNT
    start = POINT_INDEX[ball]
    distances[start] = 0
    pending = deque((start,))
    while pending:
        vertex = pending.popleft()
        for destination, edge in ADJACENCY[vertex]:
            if edge in used_edges:
                continue
            point = POINTS[destination]
            rebounds = point in visited or _boundary(point) or _goal(point)
            candidate = distances[vertex] + (0 if rebounds else 1)
            if candidate >= distances[destination]:
                continue
            distances[destination] = candidate
            if rebounds:
                pending.appendleft(destination)
            else:
                pending.append(destination)
    return tuple(distances)


def vertex_category(distance: int, free_degree: int) -> int:
    """Combine distance and degree exactly as the 57-way schema specifies."""

    if distance >= 7:
        return 56
    return 8 * distance + min(max(free_degree - 1, 0), 7)


def encode_active(state: ReplayState, *, reflected: bool = False) -> tuple[int, ...]:
    """Encode a nonterminal boundary state from the current mover's view."""

    if state.to_move not in (0, 1):
        raise ValueError("to_move must be zero or one")
    transform = rotate_point if state.to_move == 1 else (lambda point: point)
    if reflected:
        base_transform = transform
        transform = lambda point: reflect_point(base_transform(point))
    ball = transform(state.ball)
    try:
        used_edges = {
            EDGE_INDEX[_segment(transform(first), transform(second))]
            for first, second in state.used_segments
        }
    except KeyError as error:
        raise ValueError("state contains a non-canonical edge") from error
    visited = {
        transform(point)
        for point, count in state.visit_count.items()
        if count > 0
    }
    distances = _turn_distances(ball, used_edges, visited)
    degrees = _free_degrees(used_edges)
    active = sorted(used_edges)
    active.extend(
        EDGE_COUNT + vertex * VERTEX_CATEGORIES
        + vertex_category(distances[vertex], degrees[vertex])
        for vertex in range(VERTEX_COUNT)
    )
    return tuple(active)


def validate_active(active: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in active)
    if not normalized or normalized != tuple(sorted(set(normalized))):
        raise ValueError("active features must be sorted and unique")
    if normalized[0] < 0 or normalized[-1] >= INPUT_COUNT:
        raise ValueError("active feature is outside the 6301-input schema")
    selected = set(normalized)
    for vertex in range(VERTEX_COUNT):
        start = EDGE_COUNT + vertex * VERTEX_CATEGORIES
        if sum(start + category in selected for category in range(57)) != 1:
            raise ValueError(f"vertex {vertex} must select exactly one category")
    return normalized


def transform_active(
    active: Sequence[int], vertex_map: Sequence[int], edge_map: Sequence[int]
) -> tuple[int, ...]:
    active = validate_active(active)
    transformed: list[int] = []
    for feature in active:
        if feature < EDGE_COUNT:
            transformed.append(edge_map[feature])
            continue
        relative = feature - EDGE_COUNT
        vertex, category = divmod(relative, VERTEX_CATEGORIES)
        transformed.append(
            EDGE_COUNT + vertex_map[vertex] * VERTEX_CATEGORIES + category
        )
    return tuple(sorted(transformed))


def reflect_active(active: Sequence[int]) -> tuple[int, ...]:
    return transform_active(active, REFLECTED_VERTICES, REFLECTED_EDGES)


def rotate_active(active: Sequence[int]) -> tuple[int, ...]:
    return transform_active(active, ROTATED_VERTICES, ROTATED_EDGES)


def _legal_destination(state: ReplayState, destination: Point) -> bool:
    return (
        state.winner is None
        and destination in set(_neighbors(state.ball))
        and _segment(state.ball, destination) not in state.used_segments
        and not _forbidden_boundary_segment(state.ball, destination)
    )


def _has_legal_move(state: ReplayState) -> bool:
    return any(_legal_destination(state, point) for point in _neighbors(state.ball))


def apply_primitive(state: ReplayState, direction: int | str) -> None:
    try:
        direction_index = int(direction)
    except (TypeError, ValueError) as error:
        raise ValueError("direction must be in 0..7") from error
    if direction_index < 0 or direction_index >= len(DIRECTION_DELTAS):
        raise ValueError("direction must be in 0..7")
    dx, dy = DIRECTION_DELTAS[direction_index]
    destination = state.ball[0] + dx, state.ball[1] + dy
    if not _legal_destination(state, destination):
        raise ValueError("illegal primitive direction")
    mover = state.to_move
    extra = _boundary(destination) or state.visit_count.get(destination, 0) > 0
    state.used_segments.add(_segment(state.ball, destination))
    state.ball = destination
    state.visit_count[destination] = state.visit_count.get(destination, 0) + 1
    if _goal(destination):
        state.winner = 0 if destination[1] == 0 else 1
        return
    state.to_move = mover if extra else 1 - mover
    if not _has_legal_move(state):
        state.winner = 1 - mover


def apply_complete_turn(state: ReplayState, player: int, action: str) -> None:
    if state.winner is not None:
        raise ValueError("transcript continues after terminal position")
    if player != state.to_move:
        raise ValueError("recorded player does not match player to move")
    if not action or any(character not in "01234567" for character in action):
        raise ValueError("complete turn action must contain directions 0..7")
    mover = state.to_move
    for primitive, direction in enumerate(action):
        if state.winner is not None:
            raise ValueError("complete turn continues after terminal position")
        if state.to_move != mover:
            raise ValueError(f"complete turn continues after handoff at {primitive}")
        apply_primitive(state, direction)
    if state.winner is None and state.to_move == mover:
        raise ValueError("complete turn ends before rebound chain completion")


def replay_boundaries(
    turns: Iterable[dict[str, object]], *, include_reflections: bool = False
) -> tuple[tuple[int, ...], ...]:
    state = ReplayState()
    result: list[tuple[int, ...]] = []
    for turn in turns:
        result.append(encode_active(state))
        if include_reflections:
            result.append(encode_active(state, reflected=True))
        apply_complete_turn(state, int(turn["player_id"]), str(turn["action"]))
    return tuple(result)
