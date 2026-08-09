#!/usr/bin/env python3

"""Train the compact policy/value network used by ``neural_puct``.

The maintained fit is expert-first: exposed elite games plus public Jacek
games that have already been filtered through the rank-one game-id lock.
Self-play is supported only as an explicit optional input with recorded teacher
and weighting provenance; it is never loaded by default.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import re
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:
    # Replay-only validation deliberately remains available to the live
    # collector in minimal Python environments.  Actual sample construction
    # and training still require the pinned research dependencies.
    np = None


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
PROMOTION = HERE.parents[1] / "promotion"
DEFAULT_ELITE = (
    PROMOTION / "elite_final_holdout_v2.json",
    PROMOTION / "elite_final_holdout_v1.json",
)
DEFAULT_JACEK = HERE / "public_jacek_unlocked_v1.json"
DEFAULT_OUTPUT = HERE / "neural_puct_model.json"
JACEK_AGENT_ID = 6_273_433

WIDTH = 8
HEIGHT = 10
GOAL_BOTTOM = 0
GOAL_TOP = HEIGHT + 2
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
MIRRORED_DIRECTION = (0, 7, 6, 5, 4, 3, 2, 1)

EDGE_COUNT = 316
VERTEX_COUNT = 105
DISTANCE_BUCKETS = 8
GLOBAL_COUNT = 25
INPUT_COUNT = EDGE_COUNT + VERTEX_COUNT * DISTANCE_BUCKETS + VERTEX_COUNT + GLOBAL_COUNT
HIDDEN_ONE = 32
HIDDEN_TWO = 32
POLICY_COUNT = 8
ACTION_FEATURE_NAMES = (
    "direction_x",
    "direction_attack",
    "rebound",
    "handoff",
    "immediate_win",
    "immediate_loss",
    "attack_goal_progress",
    "own_goal_safety",
    "remaining_degree",
    "safe_frontiers",
    "dead_frontiers",
    "continuation_size",
    "layer_fill_after",
    "layer_closure",
    "escape_routes",
    "opponent_mobility",
)
ACTION_FEATURE_COUNT = len(ACTION_FEATURE_NAMES)
ACTION_POLICY_HIDDEN = 8
INT4_LIMIT = 7
POLICY_TARGET_SCHEMA = "canonical-primitive-root-visits-v1"
LIVE_SNAPSHOT_SCHEMA = "papersoccer.live-replay-training-snapshot.v1"
LIVE_REPLAY_SCHEMA = "papersoccer.codingame-live-replay.v1"
LIVE_RELABEL_SCHEMA = "papersoccer.live-replay-relabel.v1"


@dataclasses.dataclass(frozen=True)
class PolicyTarget:
    probabilities: tuple[float, ...]
    total_visits: int
    fallback: bool


@dataclasses.dataclass(frozen=True)
class Game:
    key: str
    game_id: int
    source: str
    focus_agent_id: int
    focus_player: int | None
    winner: int
    turns: tuple[tuple[int, str], ...]
    policy_start_turn: int = 0
    policy_targets: tuple[tuple[PolicyTarget, ...] | None, ...] | None = None
    value_targets: tuple[tuple[float, ...] | None, ...] | None = None
    priority_weights: tuple[tuple[float, ...] | None, ...] | None = None
    split_group: str | None = None
    duplicate_count: int = 1
    split_scope: str = "agent"
    source_group: str = "anchor"
    policy_mass: float = 1.0
    value_mass: float = 1.0
    allow_policy_disagreement: bool = False


@dataclasses.dataclass
class Sample:
    features: np.ndarray
    action_features: np.ndarray
    legal: np.ndarray
    policy: int
    policy_target: np.ndarray
    value: float
    has_policy: bool
    game_key: str
    focus_agent_id: int
    source: str
    state_key: bytes
    has_value: bool = True
    source_group: str = "anchor"
    policy_mass: float = 1.0
    value_mass: float = 1.0
    value_weight: float = 0.0
    policy_weight: float = 0.0


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_regular(point: tuple[int, int]) -> bool:
    x, y = point
    return 0 <= x <= WIDTH and 1 <= y <= HEIGHT + 1


def is_goal(point: tuple[int, int]) -> bool:
    x, y = point
    return 3 <= x <= 5 and y in (GOAL_BOTTOM, GOAL_TOP)


def is_boundary(point: tuple[int, int]) -> bool:
    x, y = point
    if not is_regular(point):
        return False
    if x in (0, WIDTH):
        return True
    return y in (1, HEIGHT + 1) and x != WIDTH // 2


def segment(first: tuple[int, int], second: tuple[int, int]):
    return tuple(sorted((first, second), key=lambda point: (point[1], point[0])))


def forbidden_boundary_segment(first: tuple[int, int], second: tuple[int, int]) -> bool:
    a, b = segment(first, second)
    if is_goal(a) or is_goal(b):
        return (
            a[0] == b[0]
            and a[0] in (3, 5)
            and {a[1], b[1]} in ({0, 1}, {HEIGHT + 1, HEIGHT + 2})
        )
    if not (is_regular(a) and is_regular(b) and is_boundary(a) and is_boundary(b)):
        return False
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (
        (a[1] == b[1] and a[1] in (1, HEIGHT + 1) and dx == 1 and dy == 0)
        or (a[0] == b[0] and a[0] in (0, WIDTH) and dx == 0 and dy == 1)
    )


def geometric_neighbors(point: tuple[int, int]):
    x, y = point
    result = []
    if not is_regular(point):
        return result
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if is_regular(candidate):
                result.append(candidate)
    if y in (1, HEIGHT + 1) and 3 <= x <= 5:
        goal_y = GOAL_BOTTOM if y == 1 else GOAL_TOP
        for goal_x in range(3, 6):
            candidate = (goal_x, goal_y)
            if max(abs(x - goal_x), abs(y - goal_y)) == 1:
                result.append(candidate)
    return result


def make_topology():
    points = [(x, y) for y in range(1, HEIGHT + 2) for x in range(WIDTH + 1)]
    for x in range(3, 6):
        points.extend(((x, GOAL_BOTTOM), (x, GOAL_TOP)))
    point_index = {point: index for index, point in enumerate(points)}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    edge_index = {}
    for source in points:
        for destination in geometric_neighbors(source):
            if forbidden_boundary_segment(source, destination):
                continue
            edge = segment(source, destination)
            if edge not in edge_index:
                edge_index[edge] = len(edges)
                edges.append(edge)
    if len(points) != VERTEX_COUNT or len(edges) != EDGE_COUNT:
        raise RuntimeError(f"unexpected topology: {len(points)} vertices, {len(edges)} edges")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in points]
    for edge_id, (first, second) in enumerate(edges):
        a, b = point_index[first], point_index[second]
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))
    # The production SearchTopology exposes goal edges from the mouth vertex,
    # but terminal goal vertices themselves have no outgoing adjacency.
    for vertex, point in enumerate(points):
        if is_goal(point):
            adjacency[vertex].clear()
    cut_edges: list[list[int]] = [[] for _ in range(HEIGHT + 2)]
    for edge_id, (first, second) in enumerate(edges):
        if first[1] != second[1]:
            cut_edges[min(first[1], second[1])].append(edge_id)
    if any(not values for values in cut_edges):
        raise RuntimeError("topology has an empty horizontal cut")
    return points, point_index, edges, edge_index, adjacency, cut_edges


POINTS, POINT_INDEX, EDGES, EDGE_INDEX, ADJACENCY, CUT_EDGES = make_topology()
EDGE_CUT = {
    edge: cut
    for cut, edges in enumerate(CUT_EDGES)
    for edge in edges
}


def rotate(point: tuple[int, int]) -> tuple[int, int]:
    return WIDTH - point[0], HEIGHT + 2 - point[1]


def mirror(point: tuple[int, int]) -> tuple[int, int]:
    return WIDTH - point[0], point[1]


def canonical_state(ball, used_segments, visited, player: int, reflected: bool):
    def transform(point):
        result = rotate(point) if player == 1 else point
        return mirror(result) if reflected else result

    return (
        transform(ball),
        {segment(transform(a), transform(b)) for a, b in used_segments},
        {transform(point) for point in visited},
    )


def true_turn_distances(ball, used_edges: set[int], visited: set[tuple[int, int]]):
    infinity = 1_000_000
    distances = [infinity] * VERTEX_COUNT
    start = POINT_INDEX[ball]
    distances[start] = 0
    queue = collections.deque([start])
    while queue:
        vertex = queue.popleft()
        if is_goal(POINTS[vertex]):
            continue
        for destination, edge in ADJACENCY[vertex]:
            if edge in used_edges:
                continue
            point = POINTS[destination]
            cost = 0 if point in visited or is_boundary(point) or is_goal(point) else 1
            candidate = distances[vertex] + cost
            if candidate >= distances[destination]:
                continue
            distances[destination] = candidate
            if cost == 0:
                queue.appendleft(destination)
            else:
                queue.append(destination)
    return [min(distance, 7) for distance in distances]


def free_degrees(used_edges: set[int]):
    return [sum(edge not in used_edges for _, edge in neighbors) for neighbors in ADJACENCY]


def rebound_component(ball, used_edges, visited, degrees):
    start = POINT_INDEX[ball]
    component = {start}
    queue = collections.deque([start])
    while queue:
        vertex = queue.popleft()
        if is_goal(POINTS[vertex]):
            continue
        for destination, edge in ADJACENCY[vertex]:
            if edge in used_edges or destination in component:
                continue
            point = POINTS[destination]
            if point in visited or is_boundary(point) or is_goal(point):
                component.add(destination)
                queue.append(destination)
    frontiers = set()
    for vertex in component:
        if is_goal(POINTS[vertex]):
            continue
        for destination, edge in ADJACENCY[vertex]:
            if edge in used_edges or destination in component:
                continue
            point = POINTS[destination]
            if point not in visited and not is_boundary(point) and not is_goal(point):
                frontiers.add(destination)
    safe = sum(degrees[vertex] > 1 for vertex in frontiers)
    dead = len(frontiers) - safe
    internal = sum(
        edge not in used_edges
        and POINT_INDEX[first] in component
        and POINT_INDEX[second] in component
        for edge, (first, second) in enumerate(EDGES)
    )
    touch_attack = any(is_goal(POINTS[vertex]) and POINTS[vertex][1] == 0 for vertex in component)
    touch_own = any(
        is_goal(POINTS[vertex]) and POINTS[vertex][1] == HEIGHT + 2
        for vertex in component
    )
    return len(component), safe, dead, internal, touch_attack, touch_own


def feature_vector(ball, used_segments, visited, player: int, reflected: bool):
    ball, used_segments, visited = canonical_state(
        ball, used_segments, visited, player, reflected
    )
    used_edges = {EDGE_INDEX[edge] for edge in used_segments}
    distances = true_turn_distances(ball, used_edges, visited)
    degrees = free_degrees(used_edges)
    component, safe, dead, internal, touch_attack, touch_own = rebound_component(
        ball, used_edges, visited, degrees
    )
    result = np.zeros(INPUT_COUNT, dtype=np.float32)
    for edge in used_edges:
        result[edge] = 1.0
    distance_offset = EDGE_COUNT
    for vertex, distance in enumerate(distances):
        result[distance_offset + vertex * DISTANCE_BUCKETS + distance] = 1.0
    degree_offset = distance_offset + VERTEX_COUNT * DISTANCE_BUCKETS
    result[degree_offset : degree_offset + VERTEX_COUNT] = (
        np.asarray(degrees, dtype=np.float32) / 8.0
    )
    global_offset = degree_offset + VERTEX_COUNT
    attack_distance = min(
        distances[POINT_INDEX[(x, GOAL_BOTTOM)]] for x in range(3, 6)
    )
    own_distance = min(distances[POINT_INDEX[(x, GOAL_TOP)]] for x in range(3, 6))
    globals_ = [
        ball[0] / WIDTH,
        ball[1] / (HEIGHT + 2),
        len(used_edges) / EDGE_COUNT,
        len(visited) / VERTEX_COUNT,
        degrees[POINT_INDEX[ball]] / 8.0,
        attack_distance / 7.0,
        own_distance / 7.0,
        component / VERTEX_COUNT,
        min(safe, 64) / 64.0,
        min(dead, 64) / 64.0,
        internal / EDGE_COUNT,
        float(touch_attack),
        float(touch_own),
    ]
    globals_.extend(
        sum(edge in used_edges for edge in cut) / len(cut) for cut in CUT_EDGES
    )
    if len(globals_) != GLOBAL_COUNT:
        raise RuntimeError("neural feature global count drifted")
    result[global_offset:] = globals_
    return result


def legal_policy_mask(ball, used_segments, player: int, reflected: bool):
    mask = np.zeros(POLICY_COUNT, dtype=bool)
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        destination = (ball[0] + dx, ball[1] + dy)
        if segment(ball, destination) not in EDGE_INDEX:
            continue
        if segment(ball, destination) in used_segments:
            continue
        canonical = (direction + 4) % 8 if player == 1 else direction
        if reflected:
            canonical = MIRRORED_DIRECTION[canonical]
        mask[canonical] = True
    if not np.any(mask):
        raise ValueError("non-terminal state has no legal action")
    return mask


def action_feature_matrix(ball, used_segments, visited, player: int, reflected: bool):
    """Encode each legal primitive consequence in the current mover's frame."""

    ball, used_segments, visited = canonical_state(
        ball, used_segments, visited, player, reflected
    )
    used_edges = {EDGE_INDEX[edge] for edge in used_segments}
    base_distances = true_turn_distances(ball, used_edges, visited)
    base_attack = min(
        base_distances[POINT_INDEX[(x, GOAL_BOTTOM)]] for x in range(3, 6)
    )
    base_own = min(
        base_distances[POINT_INDEX[(x, GOAL_TOP)]] for x in range(3, 6)
    )
    result = np.zeros((POLICY_COUNT, ACTION_FEATURE_COUNT), dtype=np.float32)

    for direction, (dx, dy) in enumerate(DIRECTIONS):
        destination = ball[0] + dx, ball[1] + dy
        edge = segment(ball, destination)
        edge_id = EDGE_INDEX.get(edge)
        if edge_id is None or edge_id in used_edges:
            continue

        child_used = set(used_edges)
        child_used.add(edge_id)
        child_visited = set(visited)
        was_rebound = destination in visited or is_boundary(destination)
        child_visited.add(destination)
        degrees = free_degrees(child_used)
        remaining_degree = degrees[POINT_INDEX[destination]]
        immediate_win = is_goal(destination) and destination[1] == GOAL_BOTTOM
        immediate_loss = (
            (is_goal(destination) and destination[1] == GOAL_TOP)
            or (not is_goal(destination) and remaining_degree == 0)
        )
        terminal = immediate_win or immediate_loss
        rebound = was_rebound and not terminal
        handoff = not was_rebound and not terminal

        child_distances = true_turn_distances(
            destination, child_used, child_visited
        )
        child_attack = min(
            child_distances[POINT_INDEX[(x, GOAL_BOTTOM)]] for x in range(3, 6)
        )
        child_own = min(
            child_distances[POINT_INDEX[(x, GOAL_TOP)]] for x in range(3, 6)
        )
        component, safe, dead, _, _, _ = rebound_component(
            destination, child_used, child_visited, degrees
        )

        cut = EDGE_CUT.get(edge_id)
        layer_fill = 0.0
        layer_closure = False
        if cut is not None:
            cut_edges = CUT_EDGES[cut]
            used_in_cut = sum(value in child_used for value in cut_edges)
            layer_fill = used_in_cut / len(cut_edges)
            layer_closure = used_in_cut == len(cut_edges)

        child_player = 0 if rebound else 1
        escape_routes = 0
        if not terminal:
            for next_vertex, next_edge in ADJACENCY[POINT_INDEX[destination]]:
                if next_edge in child_used:
                    continue
                next_point = POINTS[next_vertex]
                if is_goal(next_point):
                    attacking_y = GOAL_BOTTOM if child_player == 0 else GOAL_TOP
                    escape_routes += next_point[1] == attacking_y
                    continue
                next_used = set(child_used)
                next_used.add(next_edge)
                if free_degrees(next_used)[next_vertex] > 0:
                    escape_routes += 1

        result[direction] = np.asarray(
            [
                float(dx),
                float(-dy),
                float(rebound),
                float(handoff),
                float(immediate_win),
                float(immediate_loss),
                (base_attack - child_attack) / 7.0,
                (child_own - base_own) / 7.0,
                remaining_degree / 8.0,
                min(safe, 64) / 64.0,
                min(dead, 64) / 64.0,
                component / VERTEX_COUNT,
                layer_fill,
                float(layer_closure),
                escape_routes / 8.0,
                escape_routes / 8.0 if handoff else 0.0,
            ],
            dtype=np.float32,
        )
    return result


def canonical_direction(direction: int, player: int, reflected: bool):
    result = (direction + 4) % 8 if player == 1 else direction
    return MIRRORED_DIRECTION[result] if reflected else result


def reflect_policy_target(probabilities: np.ndarray):
    result = np.zeros(POLICY_COUNT, dtype=np.float32)
    for direction, probability in enumerate(probabilities):
        result[MIRRORED_DIRECTION[direction]] = probability
    return result


def winner_from_record(record: dict) -> int:
    focus = int(record["player_id"])
    return focus if bool(record["won"]) else 1 - focus


def load_public_games(paths: Iterable[pathlib.Path]):
    games = {}
    source_hashes = {}
    for path in paths:
        payload = json.loads(path.read_text())
        schema = payload.get("schema")
        if schema not in {
            "papersoccer.frozen-elite-final-holdout.v1",
            "papersoccer.frozen-elite-final-holdout.v2",
            "papersoccer.public-jacek-training-games.v1",
        }:
            raise ValueError(f"unsupported public corpus schema in {path}: {schema}")
        if schema == "papersoccer.public-jacek-training-games.v1":
            lock_path = PROMOTION / "rank1_locked_games.json"
            if payload.get("locked_games_sha256") != sha256(lock_path):
                raise ValueError("public Jacek corpus is bound to a stale game-id lock")
            locked = {
                int(record["game_id"])
                for record in json.loads(lock_path.read_text())["records"]
            }
            if locked.intersection(int(record["game_id"]) for record in payload["records"]):
                raise ValueError("public Jacek corpus contains a locked game")
        source_hashes[str(path.relative_to(HERE.parents[3]))] = sha256(path)
        for record in payload["records"]:
            game_id = int(record["game_id"])
            turns = tuple(
                (int(turn["player_id"]), str(turn["action"]))
                for turn in record["turns"]
            )
            game = Game(
                key=f"public:{game_id}",
                game_id=game_id,
                source=("jacek" if int(record["focus_agent_id"]) == JACEK_AGENT_ID else "elite"),
                focus_agent_id=int(record["focus_agent_id"]),
                focus_player=int(record["player_id"]),
                winner=winner_from_record(record),
                turns=turns,
            )
            previous = games.get(game_id)
            if previous is not None and previous.turns != game.turns:
                raise ValueError(f"public game {game_id} has conflicting transcripts")
            if previous is None or game.source == "jacek":
                games[game_id] = game
    return list(games.values()), source_hashes


def parse_policy_targets(record: dict, actions: tuple[str, ...], location: str):
    schema = record.get("policy_target_schema")
    raw_targets = record.get("policy_targets")
    if schema is None and raw_targets is None:
        return None, None
    if schema != POLICY_TARGET_SCHEMA or not isinstance(raw_targets, list):
        raise ValueError(f"unsupported self-play policy targets at {location}")
    if len(raw_targets) != len(actions):
        raise ValueError(f"self-play policy targets do not align with turns at {location}")
    policy_start = int(record["teacher_start_turn"])
    parsed = []
    for turn_index, (action, turn_targets) in enumerate(zip(actions, raw_targets)):
        if turn_index < policy_start:
            if turn_targets is not None:
                raise ValueError(
                    f"random-opening policy target is not null at {location}"
                )
            parsed.append(None)
            continue
        if not isinstance(turn_targets, list) or len(turn_targets) != len(action):
            raise ValueError(
                f"primitive policy targets do not align at {location}, turn {turn_index}"
            )
        primitive_targets = []
        for primitive_index, target in enumerate(turn_targets):
            if not isinstance(target, dict):
                raise ValueError(
                    f"invalid primitive policy target at {location}, "
                    f"turn {turn_index}, primitive {primitive_index}"
                )
            probabilities = target.get("probabilities")
            if not isinstance(probabilities, list) or len(probabilities) != POLICY_COUNT:
                raise ValueError(
                    f"policy probabilities have the wrong size at {location}"
                )
            probabilities = tuple(float(value) for value in probabilities)
            if (
                any(not math.isfinite(value) or value < 0.0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1.0e-5
            ):
                raise ValueError(
                    f"policy probabilities are not normalized at {location}"
                )
            total_visits = target.get("total_visits")
            fallback = target.get("fallback")
            if (
                isinstance(total_visits, bool)
                or not isinstance(total_visits, int)
                or total_visits < 0
                or not isinstance(fallback, bool)
                or fallback != (total_visits == 0)
            ):
                raise ValueError(
                    f"policy visit/fallback provenance is invalid at {location}"
                )
            if fallback and (
                max(probabilities) < 1.0 - 1.0e-6
                or sum(value > 1.0e-6 for value in probabilities) != 1
            ):
                raise ValueError(
                    f"fallback policy target is not one-hot at {location}"
                )
            primitive_targets.append(
                PolicyTarget(probabilities, total_visits, fallback)
            )
        parsed.append(tuple(primitive_targets))
    return tuple(parsed), schema


def selfplay_trajectory_key(
    actions: tuple[str, ...], winner: int, policy_start_turn: int
):
    payload = json.dumps(
        [winner, policy_start_turn, actions], separators=(",", ":")
    ).encode()
    return f"selfplay-trajectory:{hashlib.sha256(payload).hexdigest()}"


def load_selfplay(paths: Iterable[pathlib.Path]):
    games = []
    teachers = set()
    teacher_models = {}
    policy_target_schemas = set()
    seen = set()
    payload_representatives = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            location = f"{path}:{line_number}"
            if record.get("schema") != "papersoccer.selfplay.v1":
                raise ValueError(f"unexpected self-play schema at {location}")
            teacher = record.get("teacher")
            if not isinstance(teacher, str) or not teacher:
                raise ValueError(f"self-play teacher is missing at {location}")
            if teacher == "neural_puct":
                teacher_model = record.get("teacher_model_sha256")
                if not isinstance(teacher_model, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", teacher_model
                ):
                    raise ValueError(
                        f"neural self-play teacher model hash is missing at {location}"
                    )
                previous_model = teacher_models.setdefault(teacher, teacher_model)
                if previous_model != teacher_model:
                    raise ValueError(
                        "neural self-play shards disagree on teacher model: "
                        f"{previous_model} != {teacher_model} at {location}"
                    )
            identity = int(record["seed"]), int(record["game"])
            if identity in seen:
                raise ValueError(
                    f"duplicate self-play seed/game identity {identity} at {location}"
                )
            seen.add(identity)
            teachers.add(teacher)
            actions = tuple(str(action) for action in record["turns"])
            policy_start_turn = int(record["teacher_start_turn"])
            if not 0 <= policy_start_turn <= len(actions):
                raise ValueError(f"self-play policy start is invalid at {location}")
            policy_targets, policy_target_schema = parse_policy_targets(
                record, actions, location
            )
            if policy_target_schema is not None:
                policy_target_schemas.add(policy_target_schema)
            turns = tuple((index % 2, action) for index, action in enumerate(actions))
            source = (
                "neural-selfplay"
                if teacher == "neural_puct"
                else f"selfplay:{teacher}"
            )
            split_group = selfplay_trajectory_key(
                actions, int(record["winner"]), policy_start_turn
            )
            game = Game(
                key=f"selfplay:{identity[0]}:{identity[1]}",
                game_id=-1 - identity[1],
                source=source,
                focus_agent_id=-1,
                focus_player=None,
                winner=int(record["winner"]),
                turns=turns,
                policy_start_turn=policy_start_turn,
                policy_targets=policy_targets,
                split_group=split_group,
            )
            payload_key = (split_group, source, policy_targets)
            representative = payload_representatives.get(payload_key)
            if representative is None:
                payload_representatives[payload_key] = len(games)
                games.append(game)
            else:
                previous = games[representative]
                games[representative] = dataclasses.replace(
                    previous, duplicate_count=previous.duplicate_count + 1
                )
    return games, sorted(teachers), teacher_models, sorted(policy_target_schemas)


def repository_path(relative: str, location: str):
    if not isinstance(relative, str) or pathlib.PurePosixPath(relative).is_absolute():
        raise ValueError(f"invalid repository-relative path at {location}")
    path = (REPOSITORY / relative).resolve()
    try:
        path.relative_to(REPOSITORY.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes the repository at {location}") from error
    return path


def verified_snapshot_file(relative: str, expected_sha256: str, location: str):
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError(f"invalid SHA-256 at {location}")
    path = repository_path(relative, location)
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"snapshot-bound file hash mismatch at {location}")
    return path


def parse_live_policy_target(target: object, location: str):
    if not isinstance(target, dict):
        raise ValueError(f"invalid live policy target at {location}")
    probabilities = target.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != POLICY_COUNT:
        raise ValueError(f"live policy probabilities have the wrong size at {location}")
    probabilities = tuple(float(value) for value in probabilities)
    if (
        any(not math.isfinite(value) or value < 0.0 for value in probabilities)
        or abs(sum(probabilities) - 1.0) > 1.0e-5
    ):
        raise ValueError(f"live policy probabilities are not normalized at {location}")
    total_visits = target.get("total_visits")
    fallback = target.get("fallback")
    if (
        isinstance(total_visits, bool)
        or not isinstance(total_visits, int)
        or total_visits < 0
        or not isinstance(fallback, bool)
        or fallback != (total_visits == 0)
    ):
        raise ValueError(f"live visit provenance is invalid at {location}")
    return PolicyTarget(probabilities, total_visits, fallback)


def load_live_replay(snapshot_path: pathlib.Path, relabel_path: pathlib.Path):
    snapshot_path = snapshot_path.resolve()
    relabel_path = relabel_path.resolve()
    try:
        snapshot_path.relative_to(REPOSITORY.resolve())
        relabel_path.relative_to(REPOSITORY.resolve())
    except ValueError as error:
        raise ValueError("live replay inputs must remain inside the repository") from error
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema") != LIVE_SNAPSHOT_SCHEMA:
        raise ValueError("unexpected live replay snapshot schema")
    if snapshot_path.stem != sha256(snapshot_path):
        raise ValueError("live replay snapshot is not content-addressed")
    if int(snapshot.get("independent_games", -1)) < int(
        snapshot.get("minimum_independent_games", 50)
    ):
        raise ValueError("live replay snapshot is below its frozen game floor")

    exclusion_path = verified_snapshot_file(
        snapshot["exclusion_registry_path"],
        snapshot["exclusion_registry_sha256"],
        "snapshot exclusion registry",
    )
    exclusions = json.loads(exclusion_path.read_text())
    if exclusions.get("schema") != "papersoccer.live-replay-exclusions.v1":
        raise ValueError("unexpected exclusion registry schema")
    protected_ids = {
        int(record["game_id"])
        for record in exclusions["records"]
        if any(
            str(category).startswith("protected_")
            for category in record.get("categories", [])
        )
    }
    verified_snapshot_file(
        snapshot["poll_path"], snapshot["poll_sha256"], "snapshot poll"
    )
    relabel_binding = snapshot.get("relabel_input")
    if not isinstance(relabel_binding, dict):
        raise ValueError("snapshot has no relabel input binding")
    verified_snapshot_file(
        relabel_binding["path"], relabel_binding["sha256"], "relabel input"
    )
    if relabel_path.name != f"{sha256(relabel_path)}.relabel.jsonl":
        raise ValueError("live relabel output is not content-addressed")

    games = []
    snapshot_records = {}
    direct_primitives = 0
    for snapshot_record in snapshot.get("records", []):
        game_id = int(snapshot_record["game_id"])
        if game_id in snapshot_records:
            raise ValueError(f"live snapshot repeats game {game_id}")
        if game_id in protected_ids:
            raise ValueError(f"live snapshot contains protected game {game_id}")
        record_path = verified_snapshot_file(
            snapshot_record["record_path"],
            snapshot_record["record_sha256"],
            f"live game {game_id}",
        )
        record = json.loads(record_path.read_text())
        if record.get("schema") != LIVE_REPLAY_SCHEMA:
            raise ValueError(f"unexpected replay schema for live game {game_id}")
        replay = record.get("replay", {})
        if int(replay.get("game_id", -1)) != game_id:
            raise ValueError(f"live replay id mismatch for game {game_id}")
        turns = tuple(
            (int(turn["player_id"]), str(turn["action"]))
            for turn in replay.get("turns", [])
        )
        winner = int(replay["winner_player_id"])
        replay_agents = {int(agent["agent_id"]): agent for agent in replay["agents"]}
        direct = snapshot_record.get("direct_experts", [])
        if not direct:
            raise ValueError(f"live game {game_id} has no frozen direct expert")
        for expert in direct:
            agent_id = int(expert["agent_id"])
            player_id = int(expert["player_id"])
            agent = replay_agents.get(agent_id)
            tier = expert.get("strength_tier")
            if (
                agent is None
                or agent.get("label_role") != "direct-public-expert"
                or int(agent["player_id"]) != player_id
                or agent.get("strength_tier") != tier
            ):
                raise ValueError(f"live expert binding mismatch in game {game_id}")
            tier_name = str(tier.get("name"))
            policy_mass = float(tier.get("policy_mass"))
            if tier_name not in {"elite-1-5", "strong-6-10", "upper-11-20"} or policy_mass not in {
                1.0,
                0.75,
                0.5,
            }:
                raise ValueError(f"invalid frozen strength tier in game {game_id}")
            direct_primitives += sum(
                len(action) for player, action in turns if player == player_id
            )
            games.append(
                Game(
                    key=f"codingame-live-direct:{game_id}:{agent_id}",
                    game_id=game_id,
                    source=f"codingame-live-expert:{tier_name}",
                    focus_agent_id=agent_id,
                    focus_player=player_id,
                    winner=winner,
                    turns=turns,
                    split_group=f"codingame-live:{game_id}",
                    split_scope="global",
                    source_group="live",
                    policy_mass=policy_mass,
                    value_mass=0.0,
                )
            )
        snapshot_records[game_id] = (snapshot_record, record, turns, winner)
    if len(snapshot_records) != int(snapshot["independent_games"]):
        raise ValueError("live snapshot record count mismatch")
    if direct_primitives != int(snapshot["direct_expert_primitives"]):
        raise ValueError("live direct primitive count mismatch")

    relabelled = set()
    relabel_searches = 0
    teacher_models = set()
    for line_number, line in enumerate(relabel_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        location = f"{relabel_path}:{line_number}"
        if record.get("schema") != LIVE_RELABEL_SCHEMA:
            raise ValueError(f"unexpected live relabel schema at {location}")
        if record.get("teacher") != "neural_puct" or int(
            record.get("requested_simulations", 0)
        ) <= 2_000:
            raise ValueError(f"live relabel teacher is not deeper neural PUCT at {location}")
        if int(record.get("max_nodes", 0)) > 100_000:
            raise ValueError(f"live relabel exceeds the production node cap at {location}")
        teacher_model = record.get("teacher_model_sha256")
        if not isinstance(teacher_model, str) or not re.fullmatch(
            r"[0-9a-f]{64}", teacher_model
        ):
            raise ValueError(f"live relabel teacher hash is invalid at {location}")
        teacher_models.add(teacher_model)
        game_id = int(record["game_id"])
        if game_id in relabelled or game_id not in snapshot_records:
            raise ValueError(f"unexpected or duplicate relabel game {game_id}")
        snapshot_record, _, expected_turns, expected_winner = snapshot_records[game_id]
        if snapshot_record.get("own_agent_id") is None:
            raise ValueError(f"game {game_id} has relabels but no owned agent")
        own_agent = int(record["own_agent_id"])
        own_player = int(record["own_player_id"])
        turns = tuple(zip(
            (int(value) for value in record["turn_players"]),
            (str(value) for value in record["turns"]),
        ))
        if (
            turns != expected_turns
            or int(record["winner"]) != expected_winner
            or own_agent != int(snapshot_record["own_agent_id"])
            or own_player != int(snapshot_record["own_player_id"])
            or record.get("source_record_sha256") != snapshot_record["record_sha256"]
        ):
            raise ValueError(f"live relabel binding mismatch for game {game_id}")
        raw_policy = record.get("policy_targets")
        raw_values = record.get("value_targets")
        raw_priorities = record.get("priorities")
        if not all(
            isinstance(value, list) and len(value) == len(turns)
            for value in (raw_policy, raw_values, raw_priorities)
        ):
            raise ValueError(f"live relabel turn alignment mismatch for game {game_id}")
        policy_targets = []
        value_targets = []
        priority_weights = []
        for turn_index, ((player, action), policy, values, priorities) in enumerate(
            zip(turns, raw_policy, raw_values, raw_priorities)
        ):
            location = f"game {game_id}, turn {turn_index}"
            if player != own_player:
                if policy is not None or values is not None or priorities is not None:
                    raise ValueError(f"opponent action copied into relabels at {location}")
                policy_targets.append(None)
                value_targets.append(None)
                priority_weights.append(None)
                continue
            if not all(
                isinstance(value, list) and len(value) == len(action)
                for value in (policy, values, priorities)
            ):
                raise ValueError(f"owned primitive targets do not align at {location}")
            parsed_policy = tuple(
                parse_live_policy_target(target, f"{location}, primitive {index}")
                for index, target in enumerate(policy)
            )
            parsed_values = tuple(float(value) for value in values)
            parsed_priorities = tuple(float(value["weight"]) for value in priorities)
            if any(
                not math.isfinite(value) or value < -1.0 or value > 1.0
                for value in parsed_values
            ) or any(
                not math.isfinite(value) or value <= 0.0
                for value in parsed_priorities
            ):
                raise ValueError(f"invalid live value or priority target at {location}")
            policy_targets.append(parsed_policy)
            value_targets.append(parsed_values)
            priority_weights.append(parsed_priorities)
            relabel_searches += len(action)
        games.append(
            Game(
                key=f"codingame-live-relabel:{game_id}:{own_agent}",
                game_id=game_id,
                source="codingame-live-neural-relabel",
                focus_agent_id=own_agent,
                focus_player=own_player,
                winner=expected_winner,
                turns=turns,
                policy_targets=tuple(policy_targets),
                value_targets=tuple(value_targets),
                priority_weights=tuple(priority_weights),
                split_group=f"codingame-live:{game_id}",
                split_scope="global",
                source_group="live",
                allow_policy_disagreement=True,
            )
        )
        relabelled.add(game_id)
    expected_relabels = {
        game_id
        for game_id, (record, _, _, _) in snapshot_records.items()
        if record.get("own_agent_id") is not None
    }
    if relabelled != expected_relabels or relabel_searches != int(
        snapshot["self_primitives_for_relabel"]
    ):
        raise ValueError("live relabel coverage does not match the frozen snapshot")
    return games, {
        "snapshot_sha256": sha256(snapshot_path),
        "relabel_sha256": sha256(relabel_path),
        "independent_games": len(snapshot_records),
        "direct_expert_primitives": direct_primitives,
        "relabelled_primitives": relabel_searches,
        "teacher_model_sha256": sorted(teacher_models),
        "source_policy_mass": {"anchor": 0.75, "live": 0.25},
    }


def replay_game(game: Game):
    ball = (WIDTH // 2, HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    to_move = 0
    terminal_winner = None
    samples = []
    for turn_index, (player, action) in enumerate(game.turns):
        if terminal_winner is not None:
            raise ValueError("game contains turns after a terminal state")
        if player != to_move or not action or any(ch not in "01234567" for ch in action):
            raise ValueError("game contains an invalid turn")
        for edge_index, encoded in enumerate(action):
            direction = int(encoded)
            dx, dy = DIRECTIONS[direction]
            destination = (ball[0] + dx, ball[1] + dy)
            edge = segment(ball, destination)
            if edge not in EDGE_INDEX or edge in used_segments:
                raise ValueError("game contains an illegal edge")
            include_sample = turn_index >= game.policy_start_turn
            if include_sample and np is None:
                raise RuntimeError(
                    "NumPy is required to construct neural training samples"
                )
            has_policy = include_sample and (
                game.focus_player is None or player == game.focus_player
            )
            soft_target = None
            if game.policy_targets is not None:
                turn_targets = game.policy_targets[turn_index]
                if has_policy:
                    if turn_targets is None:
                        raise ValueError("labelled turn omits its soft policy targets")
                    soft_target = np.asarray(
                        turn_targets[edge_index].probabilities, dtype=np.float32
                    )
                elif turn_targets is not None:
                    raise ValueError("unlabelled turn contains soft policy targets")
            value_target = 1.0 if game.winner == player else -1.0
            has_value = include_sample and game.value_mass > 0.0
            if game.value_targets is not None:
                turn_values = game.value_targets[turn_index]
                has_value = include_sample and turn_values is not None
                if has_value:
                    value_target = float(turn_values[edge_index])
            priority = 1.0
            if game.priority_weights is not None:
                turn_priorities = game.priority_weights[turn_index]
                if has_policy or has_value:
                    if turn_priorities is None:
                        raise ValueError("labelled turn omits its priority weights")
                    priority = float(turn_priorities[edge_index])
                elif turn_priorities is not None:
                    raise ValueError("unlabelled turn contains priority weights")
            if include_sample and (has_policy or has_value):
                pair = []
                for reflected in (False, True):
                    features = feature_vector(
                        ball, used_segments, visited, player, reflected
                    )
                    action_features = action_feature_matrix(
                        ball, used_segments, visited, player, reflected
                    )
                    legal = legal_policy_mask(
                        ball, used_segments, player, reflected
                    )
                    played_policy = canonical_direction(direction, player, reflected)
                    if not legal[played_policy]:
                        raise ValueError("encoded expert action is not legal")
                    policy_target = np.zeros(POLICY_COUNT, dtype=np.float32)
                    if soft_target is None:
                        policy_target[played_policy] = 1.0
                    else:
                        policy_target = (
                            reflect_policy_target(soft_target)
                            if reflected
                            else soft_target.copy()
                        )
                        if np.any(policy_target[~legal] > 1.0e-6):
                            raise ValueError(
                                "soft policy target assigns mass to an illegal edge"
                            )
                        if (
                            not game.allow_policy_disagreement
                            and policy_target[played_policy] + 1.0e-6
                            < np.max(policy_target)
                        ):
                            raise ValueError(
                                "played edge is not visit-max in soft policy target"
                            )
                    policy = int(np.argmax(policy_target))
                    pair.append(
                        (features, action_features, legal, policy, policy_target)
                    )
                state_key = min(pair[0][0].tobytes(), pair[1][0].tobytes())
                for features, action_features, legal, policy, policy_target in pair:
                    samples.append(
                        Sample(
                            features=features,
                            action_features=action_features,
                            legal=legal,
                            policy=policy,
                            policy_target=policy_target,
                            value=value_target,
                            has_policy=has_policy,
                            game_key=game.key,
                            focus_agent_id=game.focus_agent_id,
                            source=game.source,
                            state_key=state_key,
                            has_value=has_value,
                            source_group=game.source_group,
                            policy_mass=game.policy_mass * priority,
                            value_mass=game.value_mass * priority,
                        )
                    )
            was_visited = destination in visited
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
            free = [
                edge_id
                for _, edge_id in ADJACENCY[POINT_INDEX[ball]]
                if edge_id not in {EDGE_INDEX[value] for value in used_segments}
            ]
            if is_goal(ball):
                terminal_winner = 0 if ball[1] == GOAL_BOTTOM else 1
            elif not free:
                terminal_winner = 1 - player
            elif not (was_visited or is_boundary(ball)):
                to_move = 1 - player
            if edge_index + 1 < len(action):
                if terminal_winner is not None or to_move != player:
                    raise ValueError("game contains an overlong rebound action")
            elif terminal_winner is None and to_move == player:
                raise ValueError("game ends a turn before a mandatory rebound")
    if terminal_winner is None:
        raise ValueError("game is incomplete")
    if terminal_winner != game.winner:
        raise ValueError("recorded winner disagrees with replayed rules")
    return samples


def split_name(game: Game):
    key = game.split_group or game.key
    bucket = hashlib.sha256(key.encode()).digest()[0] % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def stratified_splits(games: list[Game]):
    result = {}
    global_groups = collections.defaultdict(list)
    by_agent = collections.defaultdict(lambda: collections.defaultdict(list))
    for game in games:
        group = game.split_group or game.key
        if game.split_scope == "global":
            global_groups[group].append(game)
        elif game.split_scope == "agent":
            by_agent[game.focus_agent_id][group].append(game)
        else:
            raise ValueError(f"unsupported split scope {game.split_scope!r}")

    def assign(groups):
        ordered = sorted(
            groups.items(),
            key=lambda item: (hashlib.sha256(item[0].encode()).digest(), item[0]),
        )
        count = len(ordered)
        if count < 3:
            for _, members in ordered:
                split = split_name(members[0])
                for game in members:
                    result[game.key] = split
            return
        validation_count = max(1, int(round(count * 0.1)))
        test_count = max(1, int(round(count * 0.1)))
        if validation_count + test_count >= count:
            validation_count = test_count = 1
        train_count = count - validation_count - test_count
        for index, (_, members) in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            for game in members:
                result[game.key] = split

    if global_groups:
        assign(global_groups)
    for groups in by_agent.values():
        assign(groups)
    return result


def assign_weights(
    samples: list[Sample], selfplay_multiplier: float, live_mass: float = 0.25
):
    if not 0.2 <= live_mass <= 0.3:
        raise ValueError("live policy mass must remain between 0.2 and 0.3")
    value_mass_by_game = collections.defaultdict(float)
    value_games_by_agent = collections.defaultdict(set)
    for sample in samples:
        if sample.has_value:
            value_mass_by_game[sample.game_key] += sample.value_mass
            value_games_by_agent[sample.focus_agent_id].add(sample.game_key)
    for sample in samples:
        if not sample.has_value:
            sample.value_weight = 0.0
            continue
        multiplier = selfplay_multiplier if sample.focus_agent_id == -1 else 1.0
        sample.value_weight = multiplier * sample.value_mass / (
            len(value_games_by_agent[sample.focus_agent_id])
            * value_mass_by_game[sample.game_key]
        )
    policy_mass_by_game = collections.defaultdict(float)
    games_by_agent = collections.defaultdict(set)
    for sample in samples:
        if sample.has_policy:
            policy_mass_by_game[sample.game_key] += sample.policy_mass
            games_by_agent[sample.focus_agent_id].add(sample.game_key)
    for sample in samples:
        if not sample.has_policy:
            continue
        multiplier = (
            4.0
            if sample.focus_agent_id == JACEK_AGENT_ID
            else selfplay_multiplier
            if sample.focus_agent_id == -1
            else 1.0
        )
        sample.policy_weight = multiplier * sample.policy_mass / (
            len(games_by_agent[sample.focus_agent_id])
            * policy_mass_by_game[sample.game_key]
        )

    def rebalance(attribute: str):
        totals = collections.Counter()
        for sample in samples:
            weight = getattr(sample, attribute)
            if weight > 0.0:
                totals[sample.source_group] += weight
        if totals["anchor"] > 0.0 and totals["live"] > 0.0:
            combined = totals["anchor"] + totals["live"]
            scales = {
                "anchor": combined * (1.0 - live_mass) / totals["anchor"],
                "live": combined * live_mass / totals["live"],
            }
            for sample in samples:
                if sample.source_group in scales:
                    setattr(
                        sample,
                        attribute,
                        getattr(sample, attribute) * scales[sample.source_group],
                    )

    rebalance("value_weight")
    rebalance("policy_weight")
    value_values = [sample.value_weight for sample in samples if sample.has_value]
    policy_values = [sample.policy_weight for sample in samples if sample.has_policy]
    value_mean = np.mean(value_values) if value_values else 1.0
    policy_mean = np.mean(policy_values) if policy_values else 1.0
    for sample in samples:
        if sample.has_value:
            sample.value_weight /= value_mean
        if sample.has_policy:
            sample.policy_weight /= policy_mean


def purge_held_out_overlaps(buckets):
    train_keys = {sample.state_key for sample in buckets["train"]}
    validation_keys = {sample.state_key for sample in buckets["validation"]}
    validation = [
        sample for sample in buckets["validation"] if sample.state_key not in train_keys
    ]
    test = [
        sample
        for sample in buckets["test"]
        if sample.state_key not in train_keys and sample.state_key not in validation_keys
    ]
    removed = {
        "validation": len(buckets["validation"]) - len(validation),
        "test": len(buckets["test"]) - len(test),
    }
    buckets["validation"] = validation
    buckets["test"] = test
    return removed


def preteacher_primitive_count(game: Game):
    return sum(len(action) for _, action in game.turns[: game.policy_start_turn])


def selfplay_preprocessing_report(games: list[Game]):
    selfplay = [game for game in games if game.focus_agent_id == -1]
    groups = collections.Counter(game.split_group or game.key for game in selfplay)
    retained_prefixes = sum(preteacher_primitive_count(game) for game in selfplay)
    raw_prefixes = sum(
        preteacher_primitive_count(game) * game.duplicate_count for game in selfplay
    )
    raw_records = sum(game.duplicate_count for game in selfplay)
    return {
        "selfplay_raw_records": raw_records,
        "selfplay_retained_payload_records": len(selfplay),
        "selfplay_trajectory_groups": len(groups),
        "selfplay_exact_duplicate_records_collapsed": raw_records - len(selfplay),
        "selfplay_trajectory_payload_conflict_groups": sum(
            count > 1 for count in groups.values()
        ),
        "selfplay_raw_preteacher_primitives_excluded": raw_prefixes,
        "selfplay_retained_preteacher_primitives_excluded": retained_prefixes,
        "selfplay_raw_preteacher_reflected_samples_excluded": raw_prefixes * 2,
        "selfplay_retained_preteacher_reflected_samples_excluded": retained_prefixes * 2,
    }


def dataset_from_games(
    games: list[Game], selfplay_multiplier: float = 1.0, live_mass: float = 0.25
):
    buckets = {name: [] for name in ("train", "validation", "test")}
    rejected = []
    game_counts = collections.Counter()
    agent_splits = collections.defaultdict(lambda: collections.Counter())
    agent_group_splits = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    preprocessing = selfplay_preprocessing_report(games)
    replayed = []
    for game in sorted(games, key=lambda item: item.key):
        try:
            samples = replay_game(game)
        except ValueError as error:
            rejected.append(
                {
                    "game_id": game.game_id,
                    "collapsed_records": game.duplicate_count,
                    "reason": str(error),
                }
            )
            continue
        if not samples:
            rejected.append(
                {
                    "game_id": game.game_id,
                    "collapsed_records": game.duplicate_count,
                    "reason": "no samples",
                }
            )
            continue
        replayed.append((game, samples))
    split_for_game = stratified_splits([game for game, _ in replayed])
    for game, samples in replayed:
        split = split_for_game[game.key]
        buckets[split].extend(samples)
        game_counts[split] += 1
        agent_splits[game.focus_agent_id][split] += 1
        agent_group_splits[game.focus_agent_id][split].add(
            game.split_group or game.key
        )
    if any(not buckets[name] for name in buckets):
        raise RuntimeError("expert corpus did not populate every game split")
    removed = purge_held_out_overlaps(buckets)
    if not buckets["validation"] or not buckets["test"]:
        raise RuntimeError("unseen held-out expert samples are empty")
    for split in buckets:
        assign_weights(buckets[split], selfplay_multiplier, live_mass)
    source_mass = {}
    for split, samples in buckets.items():
        policy = collections.Counter()
        value = collections.Counter()
        for sample in samples:
            policy[sample.source_group] += sample.policy_weight
            value[sample.source_group] += sample.value_weight
        policy_total = sum(policy.values()) or 1.0
        value_total = sum(value.values()) or 1.0
        source_mass[split] = {
            "policy": {
                key: amount / policy_total for key, amount in sorted(policy.items())
            },
            "value": {
                key: amount / value_total for key, amount in sorted(value.items())
            },
        }
    return buckets, {
        "games": dict(game_counts),
        "agent_game_splits": {
            str(agent): dict(counts) for agent, counts in sorted(agent_splits.items())
        },
        "agent_trajectory_group_splits": {
            str(agent): {
                split: len(groups) for split, groups in sorted(splits.items())
            }
            for agent, splits in sorted(agent_group_splits.items())
        },
        "rejected_games": rejected,
        "held_out_feature_overlaps_removed": removed,
        "source_mass_after_weighting": source_mass,
        **preprocessing,
    }


def arrays(samples: list[Sample]):
    return {
        "x": np.stack([sample.features for sample in samples]),
        "action_x": np.stack([sample.action_features for sample in samples]),
        "legal": np.stack([sample.legal for sample in samples]),
        "policy": np.asarray([sample.policy for sample in samples], dtype=np.int64),
        "policy_target": np.stack([sample.policy_target for sample in samples]),
        "value": np.asarray([sample.value for sample in samples], dtype=np.float32),
        "value_weight": np.asarray(
            [sample.value_weight for sample in samples], dtype=np.float32
        ),
        "policy_weight": np.asarray(
            [sample.policy_weight for sample in samples], dtype=np.float32
        ),
        "source": np.asarray([sample.source for sample in samples]),
    }


def initialize(seed: int, policy_head: str = "legacy-directional"):
    rng = np.random.default_rng(seed)
    parameters = {
        "w1": rng.normal(0, math.sqrt(2 / INPUT_COUNT), (INPUT_COUNT, HIDDEN_ONE)).astype(np.float32),
        "b1": np.zeros(HIDDEN_ONE, dtype=np.float32),
        "w2": rng.normal(0, math.sqrt(2 / HIDDEN_ONE), (HIDDEN_ONE, HIDDEN_TWO)).astype(np.float32),
        "b2": np.zeros(HIDDEN_TWO, dtype=np.float32),
        "wv": rng.normal(0, math.sqrt(1 / HIDDEN_TWO), (HIDDEN_TWO, 1)).astype(np.float32),
        "bv": np.zeros(1, dtype=np.float32),
    }
    if policy_head == "legacy-directional":
        parameters.update(
            {
                "wp": rng.normal(
                    0,
                    math.sqrt(1 / HIDDEN_TWO),
                    (HIDDEN_TWO, POLICY_COUNT),
                ).astype(np.float32),
                "bp": np.zeros(POLICY_COUNT, dtype=np.float32),
            }
        )
    elif policy_head == "shared-action-conditioned-v1":
        parameters.update(
            {
                "wps": rng.normal(
                    0,
                    math.sqrt(1 / HIDDEN_TWO),
                    (HIDDEN_TWO, ACTION_POLICY_HIDDEN),
                ).astype(np.float32),
                "wpa": rng.normal(
                    0,
                    math.sqrt(1 / ACTION_FEATURE_COUNT),
                    (ACTION_FEATURE_COUNT, ACTION_POLICY_HIDDEN),
                ).astype(np.float32),
                "bpa": np.zeros(ACTION_POLICY_HIDDEN, dtype=np.float32),
                "wpo": rng.normal(
                    0,
                    math.sqrt(1 / ACTION_POLICY_HIDDEN),
                    (ACTION_POLICY_HIDDEN, 1),
                ).astype(np.float32),
            }
        )
    else:
        raise ValueError(f"unsupported policy head: {policy_head}")
    return rng, parameters


def policy_forward(parameters, hidden_two, action_features):
    if "wp" in parameters:
        return hidden_two @ parameters["wp"] + parameters["bp"], None
    if action_features is None:
        raise ValueError("action-conditioned policy requires action features")
    state_projection = hidden_two @ parameters["wps"]
    action_projection = np.einsum(
        "nda,ah->ndh", action_features, parameters["wpa"]
    )
    preactivation = (
        state_projection[:, None, :] + action_projection + parameters["bpa"]
    )
    hidden = np.maximum(preactivation, 0)
    logits = (hidden @ parameters["wpo"]).reshape(
        len(hidden_two), POLICY_COUNT
    )
    return logits, (preactivation, hidden)


def forward(parameters, x, action_features=None):
    z1 = x @ parameters["w1"] + parameters["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ parameters["w2"] + parameters["b2"]
    h2 = np.maximum(z2, 0)
    value = (h2 @ parameters["wv"] + parameters["bv"]).reshape(-1)
    policy, _ = policy_forward(parameters, h2, action_features)
    return z1, h1, z2, h2, value, policy


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def masked_softmax(logits, legal):
    masked = np.where(legal, logits, -1.0e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    values = np.exp(shifted) * legal
    return values / np.sum(values, axis=1, keepdims=True)


def metrics(parameters, data):
    _, _, _, _, value_logits, policy_logits = forward(
        parameters, data["x"], data["action_x"]
    )
    value_targets = (data["value"] + 1.0) * 0.5
    value_losses = (
        np.maximum(value_logits, 0)
        - value_logits * value_targets
        + np.log1p(np.exp(-np.abs(value_logits)))
    )
    probabilities = masked_softmax(policy_logits, data["legal"])
    policy_losses = -np.sum(
        data["policy_target"] * np.log(np.maximum(probabilities, 1e-12)), axis=1
    )
    policy_mask = data["policy_weight"] > 0
    top3 = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    value_denominator = max(float(np.sum(data["value_weight"])), 1e-9)
    policy_denominator = max(float(np.sum(data["policy_weight"])), 1e-9)
    return {
        "samples": int(len(value_logits)),
        "policy_samples": int(np.sum(policy_mask)),
        "value_loss": float(np.sum(value_losses * data["value_weight"]) / value_denominator),
        "value_accuracy": float(np.mean((value_logits >= 0) == (data["value"] >= 0))),
        "policy_loss": float(np.sum(policy_losses * data["policy_weight"]) / policy_denominator),
        "policy_top1": float(np.mean(np.argmax(probabilities[policy_mask], axis=1) == data["policy"][policy_mask])),
        "policy_top3": float(np.mean([target in row for target, row in zip(data["policy"][policy_mask], top3[policy_mask])])),
        "soft_policy_samples": int(
            np.sum(
                policy_mask
                & (np.max(data["policy_target"], axis=1) < 1.0 - 1.0e-6)
            )
        ),
        "policy_target_entropy": float(
            np.sum(
                -np.sum(
                    data["policy_target"]
                    * np.log(np.maximum(data["policy_target"], 1.0e-12)),
                    axis=1,
                )
                * data["policy_weight"]
            )
            / policy_denominator
        ),
    }


def metrics_by_source(parameters, data):
    result = {}
    for source in sorted(set(data["source"])):
        selected = data["source"] == source
        subset = {
            name: values[selected]
            for name, values in data.items()
            if name != "source"
        }
        result[str(source)] = metrics(parameters, subset)
    return result


def train(
    dataset,
    seed: int,
    maximum_epochs: int,
    policy_head: str = "legacy-directional",
):
    rng, parameters = initialize(seed, policy_head)
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    train_data = dataset["train"]
    batch_size = 512
    best = None
    best_loss = float("inf")
    best_epoch = 0
    step = 0
    for epoch in range(1, maximum_epochs + 1):
        order = rng.permutation(len(train_data["x"]))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_data["x"][indices]
            action_x = train_data["action_x"][indices]
            legal = train_data["legal"][indices]
            policy_target_probabilities = train_data["policy_target"][indices]
            value_target = (train_data["value"][indices] + 1.0) * 0.5
            value_weight = train_data["value_weight"][indices]
            policy_weight = train_data["policy_weight"][indices]
            z1, h1, z2, h2, value_logits, _ = forward(
                parameters, x, action_x
            )
            policy_logits, policy_cache = policy_forward(
                parameters, h2, action_x
            )
            delta_value = (sigmoid(value_logits) - value_target) * value_weight
            delta_value /= max(float(np.sum(value_weight)), 1e-9)
            probabilities = masked_softmax(policy_logits, legal)
            delta_policy = probabilities - policy_target_probabilities
            delta_policy *= policy_weight[:, None]
            delta_policy /= max(float(np.sum(policy_weight)), 1e-9)
            gradients = {
                "wv": h2.T @ delta_value[:, None],
                "bv": np.asarray([np.sum(delta_value)], dtype=np.float32),
            }
            if "wp" in parameters:
                gradients["wp"] = h2.T @ delta_policy
                gradients["bp"] = np.sum(delta_policy, axis=0)
                policy_delta2 = delta_policy @ parameters["wp"].T
            else:
                policy_preactivation, policy_hidden = policy_cache
                gradients["wpo"] = np.einsum(
                    "ndh,nd->h", policy_hidden, delta_policy
                )[:, None]
                delta_policy_hidden = (
                    delta_policy[:, :, None] * parameters["wpo"].reshape(-1)
                )
                delta_policy_preactivation = delta_policy_hidden * (
                    policy_preactivation > 0
                )
                shared_delta = np.sum(delta_policy_preactivation, axis=1)
                gradients["wps"] = h2.T @ shared_delta
                gradients["wpa"] = np.einsum(
                    "nda,ndh->ah", action_x, delta_policy_preactivation
                )
                gradients["bpa"] = np.sum(
                    delta_policy_preactivation, axis=(0, 1)
                )
                policy_delta2 = shared_delta @ parameters["wps"].T
            delta2 = (
                delta_value[:, None] @ parameters["wv"].T + policy_delta2
            ) * (z2 > 0)
            gradients["w2"] = h1.T @ delta2
            gradients["b2"] = np.sum(delta2, axis=0)
            delta1 = (delta2 @ parameters["w2"].T) * (z1 > 0)
            gradients["w1"] = x.T @ delta1
            gradients["b1"] = np.sum(delta1, axis=0)
            step += 1
            for name, parameter in parameters.items():
                gradient = gradients[name]
                if name.startswith("w"):
                    gradient = gradient + 1e-5 * parameter
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                corrected_first = first[name] / (1 - 0.9**step)
                corrected_second = second[name] / (1 - 0.999**step)
                parameter -= 0.0015 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        validation = metrics(parameters, dataset["validation"])
        score = validation["value_loss"] + validation["policy_loss"]
        print(
            f"epoch {epoch}: held-out value {validation['value_loss']:.6f}, "
            f"policy {validation['policy_loss']:.6f}, top1 {validation['policy_top1']:.4f}"
        )
        if score < best_loss - 1e-6:
            best_loss = score
            best_epoch = epoch
            best = {name: value.copy() for name, value in parameters.items()}
        elif epoch - best_epoch >= 7:
            break
    if best is None:
        raise RuntimeError("training did not produce a model")
    return best, best_epoch


def quantize_matrix(matrix):
    maximum = np.max(np.abs(matrix), axis=0)
    scales = np.where(maximum > 0, maximum / INT4_LIMIT, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(matrix / scales), -INT4_LIMIT, INT4_LIMIT).astype(np.int8)
    flattened = quantized.reshape(-1)
    if len(flattened) % 2:
        flattened = np.append(flattened, np.int8(0))
    low = (flattened[0::2].astype(np.int16) & 0x0F).astype(np.uint8)
    high = ((flattened[1::2].astype(np.int16) & 0x0F) << 4).astype(np.uint8)
    packed = low | high
    restored = quantized.astype(np.float32) * scales
    return {
        "shape": list(matrix.shape),
        "scales": [float(value) for value in scales],
        "data": base64.b64encode(packed.tobytes()).decode(),
    }, restored


def quantized_model(parameters):
    tensors = {}
    restored = {}
    matrix_names = ["w1", "w2", "wv"]
    matrix_names.extend(
        ["wp"] if "wp" in parameters else ["wps", "wpa", "wpo"]
    )
    for name in matrix_names:
        tensors[name], restored[name] = quantize_matrix(parameters[name])
    bias_names = ["b1", "b2", "bv"]
    bias_names.extend(["bp"] if "bp" in parameters else ["bpa"])
    for name in bias_names:
        restored[name] = parameters[name].copy()
    return tensors, restored


def unpack_quantized_tensor(tensor):
    count = math.prod(tensor["shape"])
    packed = np.frombuffer(base64.b64decode(tensor["data"]), dtype=np.uint8)
    nibbles = np.empty(len(packed) * 2, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    signed = nibbles[:count].astype(np.int8)
    signed[signed >= 8] -= 16
    if np.any(signed == -8):
        raise RuntimeError("quantized tensor contains forbidden -8 weight")
    return signed.reshape(tensor["shape"])


def quantized_golden(tensors, parameters):
    """Evaluate the generated-model golden input in the C++ loop order."""

    weights = {name: unpack_quantized_tensor(tensor) for name, tensor in tensors.items()}
    scales = {
        name: np.asarray(tensor["scales"], dtype=np.float32)
        for name, tensor in tensors.items()
    }
    features = np.zeros(INPUT_COUNT, dtype=np.float32)
    for index in range(0, INPUT_COUNT, 17):
        features[index] = np.float32(((index % 11) + 1) / 11)

    hidden_one = parameters["b1"].copy()
    for input_index, feature in enumerate(features):
        if feature == 0.0:
            continue
        for output in range(HIDDEN_ONE):
            contribution = np.float32(
                np.float32(feature * np.float32(weights["w1"][input_index, output]))
                * scales["w1"][output]
            )
            hidden_one[output] = np.float32(hidden_one[output] + contribution)
    hidden_one = np.maximum(hidden_one, np.float32(0.0))

    hidden_two = parameters["b2"].copy()
    for input_index in range(HIDDEN_ONE):
        for output in range(HIDDEN_TWO):
            contribution = np.float32(
                np.float32(
                    hidden_one[input_index]
                    * np.float32(weights["w2"][input_index, output])
                )
                * scales["w2"][output]
            )
            hidden_two[output] = np.float32(hidden_two[output] + contribution)
    hidden_two = np.maximum(hidden_two, np.float32(0.0))

    value_logit = np.float32(parameters["bv"][0])
    for hidden in range(HIDDEN_TWO):
        value_contribution = np.float32(
            np.float32(hidden_two[hidden] * np.float32(weights["wv"][hidden, 0]))
            * scales["wv"][0]
        )
        value_logit = np.float32(value_logit + value_contribution)
    if "wp" in parameters:
        policy_logits = parameters["bp"].copy()
        for hidden in range(HIDDEN_TWO):
            for policy in range(POLICY_COUNT):
                contribution = np.float32(
                    np.float32(
                        hidden_two[hidden]
                        * np.float32(weights["wp"][hidden, policy])
                    )
                    * scales["wp"][policy]
                )
                policy_logits[policy] = np.float32(
                    policy_logits[policy] + contribution
                )
        golden_input = "stride17-mod11-v1"
    else:
        action_features = np.zeros(
            (POLICY_COUNT, ACTION_FEATURE_COUNT), dtype=np.float32
        )
        for direction in range(POLICY_COUNT):
            for feature in range(ACTION_FEATURE_COUNT):
                action_features[direction, feature] = np.float32(
                    ((direction * ACTION_FEATURE_COUNT + feature) % 13 + 1) / 13
                )
        policy_logits = np.zeros(POLICY_COUNT, dtype=np.float32)
        for policy in range(POLICY_COUNT):
            policy_hidden = parameters["bpa"].copy()
            for hidden in range(HIDDEN_TWO):
                for output in range(ACTION_POLICY_HIDDEN):
                    contribution = np.float32(
                        np.float32(
                            hidden_two[hidden]
                            * np.float32(weights["wps"][hidden, output])
                        )
                        * scales["wps"][output]
                    )
                    policy_hidden[output] = np.float32(
                        policy_hidden[output] + contribution
                    )
            for feature in range(ACTION_FEATURE_COUNT):
                for output in range(ACTION_POLICY_HIDDEN):
                    contribution = np.float32(
                        np.float32(
                            action_features[policy, feature]
                            * np.float32(weights["wpa"][feature, output])
                        )
                        * scales["wpa"][output]
                    )
                    policy_hidden[output] = np.float32(
                        policy_hidden[output] + contribution
                    )
            policy_hidden = np.maximum(policy_hidden, np.float32(0.0))
            for hidden in range(ACTION_POLICY_HIDDEN):
                contribution = np.float32(
                    np.float32(
                        policy_hidden[hidden]
                        * np.float32(weights["wpo"][hidden, 0])
                    )
                    * scales["wpo"][0]
                )
                policy_logits[policy] = np.float32(
                    policy_logits[policy] + contribution
                )
        golden_input = "stride17-mod11-action-mod13-v1"
    return {
        "input": golden_input,
        # BCE learns a win log-odds logit. Convert sigmoid(logit) back to the
        # mover-relative [-1, 1] expectation used by PUCT.
        "value": float(np.tanh(np.float32(0.5) * value_logit)),
        "policy": [float(value) for value in policy_logits],
    }


def feature_contract():
    return {
        "schema": "mover-canonical-graph-v1",
        "input_count": INPUT_COUNT,
        "layout": [
            {"name": "used_edges", "offset": 0, "count": EDGE_COUNT},
            {"name": "true_turn_distance_onehot", "offset": EDGE_COUNT, "vertices": VERTEX_COUNT, "buckets": DISTANCE_BUCKETS},
            {"name": "free_degree_div8", "offset": EDGE_COUNT + VERTEX_COUNT * DISTANCE_BUCKETS, "count": VERTEX_COUNT},
            {
                "name": "globals",
                "offset": INPUT_COUNT - GLOBAL_COUNT,
                "values": [
                    "ball_x_div8", "ball_y_div12", "used_edges_div316",
                    "visited_vertices_div105", "ball_degree_div8",
                    "attack_goal_turn_distance_div7", "own_goal_turn_distance_div7",
                    "zero_cost_component_vertices_div105", "safe_fresh_handoffs_div64",
                    "dead_fresh_frontiers_div64", "unused_internal_component_edges_div316",
                    "touch_attack_goal", "touch_own_goal",
                    *[f"horizontal_cut_{cut}_used_fraction" for cut in range(12)],
                ],
            },
        ],
        "perspective": "player0 unchanged attacking y0; player1 rotated 180 degrees",
        "augmentation": "horizontal reflection with policy remap 0,7,6,5,4,3,2,1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite-corpus", type=pathlib.Path, action="append")
    parser.add_argument("--jacek-corpus", type=pathlib.Path, default=DEFAULT_JACEK)
    parser.add_argument("--selfplay-corpus", type=pathlib.Path, action="append")
    parser.add_argument("--live-replay-manifest", type=pathlib.Path)
    parser.add_argument("--live-relabel-corpus", type=pathlib.Path)
    parser.add_argument(
        "--live-mass",
        type=float,
        default=0.25,
        help="policy/value mass reserved for frozen live direct and relabelled data",
    )
    parser.add_argument(
        "--selfplay-multiplier",
        type=float,
        default=1.0,
        help="relative value and policy mass for focus_agent_id=-1 self-play",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--policy-head",
        choices=("legacy-directional", "shared-action-conditioned-v1"),
        default="legacy-directional",
    )
    arguments = parser.parse_args()
    if np is None:
        parser.error("NumPy is required; install requirements-research.txt")
    if not math.isfinite(arguments.selfplay_multiplier) or arguments.selfplay_multiplier <= 0:
        parser.error("--selfplay-multiplier must be finite and greater than zero")
    if arguments.selfplay_corpus is None and arguments.selfplay_multiplier != 1.0:
        parser.error("--selfplay-multiplier is only active with --selfplay-corpus")
    if not 0.2 <= arguments.live_mass <= 0.3:
        parser.error("--live-mass must remain between 0.2 and 0.3")
    if (arguments.live_replay_manifest is None) != (
        arguments.live_relabel_corpus is None
    ):
        parser.error("live replay manifest and relabel corpus must be supplied together")
    elite_paths = tuple(arguments.elite_corpus or DEFAULT_ELITE)
    public_paths = (*elite_paths, arguments.jacek_corpus)
    games, source_hashes = load_public_games(public_paths)
    selfplay_teachers = []
    selfplay_teacher_models = {}
    selfplay_policy_target_schemas = []
    live_report = None
    if arguments.selfplay_corpus is not None:
        resolved_corpora = [path.resolve() for path in arguments.selfplay_corpus]
        if len(resolved_corpora) != len(set(resolved_corpora)):
            parser.error("--selfplay-corpus paths must be unique")
        (
            selfplay_games,
            selfplay_teachers,
            selfplay_teacher_models,
            selfplay_policy_target_schemas,
        ) = load_selfplay(arguments.selfplay_corpus)
        games.extend(selfplay_games)
        for path in arguments.selfplay_corpus:
            source_hashes[str(path)] = sha256(path)
    if arguments.live_replay_manifest is not None:
        live_games, live_report = load_live_replay(
            arguments.live_replay_manifest, arguments.live_relabel_corpus
        )
        games.extend(live_games)
        source_hashes[str(arguments.live_replay_manifest)] = sha256(
            arguments.live_replay_manifest
        )
        source_hashes[str(arguments.live_relabel_corpus)] = sha256(
            arguments.live_relabel_corpus
        )
    buckets, dataset_report = dataset_from_games(
        games, arguments.selfplay_multiplier, arguments.live_mass
    )
    datasets = {name: arrays(samples) for name, samples in buckets.items()}
    parameters, best_epoch = train(
        datasets,
        arguments.seed,
        arguments.epochs,
        arguments.policy_head,
    )
    packed, restored = quantized_model(parameters)
    golden = quantized_golden(packed, parameters)
    float_metrics = {name: metrics(parameters, data) for name, data in datasets.items()}
    int4_metrics = {name: metrics(restored, data) for name, data in datasets.items()}
    float_source_metrics = {
        name: metrics_by_source(parameters, data) for name, data in datasets.items()
    }
    int4_source_metrics = {
        name: metrics_by_source(restored, data) for name, data in datasets.items()
    }
    trainer_hash = sha256(pathlib.Path(__file__))
    report = {
        "schema": "papersoccer.neural-puct-model.v1",
        "feature_schema": "neural-puct-features-v1",
        "model_kind": (
            "shared-action-conditioned-policy-value-v1"
            if arguments.policy_head == "shared-action-conditioned-v1"
            else "expert-dagger-policy-value-v1"
            if arguments.selfplay_corpus is not None
            else "expert-imitation-policy-value-v1"
        ),
        "policy_head": arguments.policy_head,
        "action_feature_count": ACTION_FEATURE_COUNT,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "action_policy_hidden": (
            ACTION_POLICY_HIDDEN
            if arguments.policy_head == "shared-action-conditioned-v1"
            else 0
        ),
        "input_count": INPUT_COUNT,
        "edge_count": EDGE_COUNT,
        "vertex_count": VERTEX_COUNT,
        "distance_buckets": DISTANCE_BUCKETS,
        "global_count": GLOBAL_COUNT,
        "hidden_one": HIDDEN_ONE,
        "hidden_two": HIDDEN_TWO,
        "policy_count": POLICY_COUNT,
        "quantization": "signed-int4-symmetric-per-output-channel",
        "golden": golden,
        "feature_contract": feature_contract(),
        "network": {
            "hidden_one": HIDDEN_ONE,
            "hidden_two": HIDDEN_TWO,
            "policy_count": POLICY_COUNT,
            "activation": "relu",
            "value": "mover-relative-logit",
            "policy": (
                "shared-action-conditioned-primitive-logits"
                if arguments.policy_head == "shared-action-conditioned-v1"
                else "canonical-primitive-direction-logits"
            ),
        },
        "training": {
            "seed": arguments.seed,
            "policy_head": arguments.policy_head,
            "best_epoch": best_epoch,
            "trainer_sha256": trainer_hash,
            "source_sha256": source_hashes,
            "selfplay_corpus_used": arguments.selfplay_corpus is not None,
            "selfplay_corpora": [
                str(path) for path in (arguments.selfplay_corpus or [])
            ],
            "selfplay_teachers": selfplay_teachers,
            "selfplay_teacher_model_sha256": selfplay_teacher_models,
            "selfplay_policy_target_schemas": selfplay_policy_target_schemas,
            "selfplay_multiplier": arguments.selfplay_multiplier,
            "live_replay_used": live_report is not None,
            "live_replay": live_report,
            "live_mass": arguments.live_mass,
            "rank5_selfplay_used": "rank_5" in selfplay_teachers,
            "policy_weighting": (
                "normalized per player and game; Jacek anchor multiplier 4; "
                "frozen live strength tiers and relabel priorities applied; "
                "anchor/live source mass fixed at 0.75/0.25 when both exist"
            ),
            "policy_target_training": (
                "weighted cross entropy; public and legacy self-play actions are "
                "one-hot; canonical-primitive-root-visits-v1 is trained as a "
                "normalized soft distribution"
            ),
            "value_weighting": (
                "normalized per focus player and game; live direct final outcomes "
                "have zero weight; live self positions use deeper neural PUCT root "
                "values; anchor/live mass fixed at 0.75/0.25 when both exist"
            ),
            **dataset_report,
        },
        "samples": {name: len(data["x"]) for name, data in datasets.items()},
        "metrics": {
            "float": float_metrics,
            "int4": int4_metrics,
            "by_source": {
                "float": float_source_metrics,
                "int4": int4_source_metrics,
            },
        },
        "quantization_metadata": {
            "kind": "signed-symmetric-int4-per-output-channel",
            "range": [-INT4_LIMIT, INT4_LIMIT],
            "packing": "row-major; first value low nibble; two's-complement nibble",
        },
        "model": {
            **packed,
            **{
                name: [float(value) for value in parameters[name].reshape(-1)]
                for name in ("b1", "b2", "bv", "bp", "bpa")
                if name in parameters
            },
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "model"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {arguments.output} ({arguments.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
