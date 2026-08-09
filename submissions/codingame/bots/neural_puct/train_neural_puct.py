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

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
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
INT4_LIMIT = 7
POLICY_TARGET_SCHEMA = "canonical-primitive-root-visits-v1"


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
    split_group: str | None = None
    duplicate_count: int = 1


@dataclasses.dataclass
class Sample:
    features: np.ndarray
    legal: np.ndarray
    policy: int
    policy_target: np.ndarray
    value: float
    has_policy: bool
    game_key: str
    focus_agent_id: int
    source: str
    state_key: bytes
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
            if include_sample:
                pair = []
                for reflected in (False, True):
                    features = feature_vector(
                        ball, used_segments, visited, player, reflected
                    )
                    legal = legal_policy_mask(
                        ball, used_segments, player, reflected
                    )
                    policy = canonical_direction(direction, player, reflected)
                    if not legal[policy]:
                        raise ValueError("encoded expert action is not legal")
                    policy_target = np.zeros(POLICY_COUNT, dtype=np.float32)
                    if soft_target is None:
                        policy_target[policy] = 1.0
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
                        if policy_target[policy] + 1.0e-6 < np.max(policy_target):
                            raise ValueError(
                                "played edge is not visit-max in soft policy target"
                            )
                    pair.append((features, legal, policy, policy_target))
                state_key = min(pair[0][0].tobytes(), pair[1][0].tobytes())
                for features, legal, policy, policy_target in pair:
                    samples.append(
                        Sample(
                            features=features,
                            legal=legal,
                            policy=policy,
                            policy_target=policy_target,
                            value=1.0 if game.winner == player else -1.0,
                            has_policy=has_policy,
                            game_key=game.key,
                            focus_agent_id=game.focus_agent_id,
                            source=game.source,
                            state_key=state_key,
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
    by_agent = collections.defaultdict(lambda: collections.defaultdict(list))
    for game in games:
        by_agent[game.focus_agent_id][game.split_group or game.key].append(game)
    result = {}
    for groups in by_agent.values():
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
            continue
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
    return result


def assign_weights(samples: list[Sample], selfplay_multiplier: float):
    by_game = collections.Counter(sample.game_key for sample in samples)
    value_games_by_agent = collections.defaultdict(set)
    for sample in samples:
        value_games_by_agent[sample.focus_agent_id].add(sample.game_key)
    for sample in samples:
        multiplier = selfplay_multiplier if sample.focus_agent_id == -1 else 1.0
        sample.value_weight = multiplier / (
            len(value_games_by_agent[sample.focus_agent_id])
            * by_game[sample.game_key]
        )
    policy_by_game = collections.Counter(
        sample.game_key for sample in samples if sample.has_policy
    )
    games_by_agent = collections.defaultdict(set)
    for sample in samples:
        if sample.has_policy:
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
        sample.policy_weight = multiplier / (
            len(games_by_agent[sample.focus_agent_id])
            * policy_by_game[sample.game_key]
        )
    value_mean = np.mean([sample.value_weight for sample in samples])
    policy_values = [sample.policy_weight for sample in samples if sample.has_policy]
    policy_mean = np.mean(policy_values)
    for sample in samples:
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


def dataset_from_games(games: list[Game], selfplay_multiplier: float = 1.0):
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
        assign_weights(buckets[split], selfplay_multiplier)
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
        **preprocessing,
    }


def arrays(samples: list[Sample]):
    return {
        "x": np.stack([sample.features for sample in samples]),
        "legal": np.stack([sample.legal for sample in samples]),
        "policy": np.asarray([sample.policy for sample in samples], dtype=np.int64),
        "policy_target": np.stack([sample.policy_target for sample in samples]),
        "value": np.asarray([sample.value for sample in samples], dtype=np.float32),
        "value_weight": np.asarray(
            [sample.value_weight or 1.0 for sample in samples], dtype=np.float32
        ),
        "policy_weight": np.asarray(
            [sample.policy_weight for sample in samples], dtype=np.float32
        ),
        "source": np.asarray([sample.source for sample in samples]),
    }


def initialize(seed: int):
    rng = np.random.default_rng(seed)
    return rng, {
        "w1": rng.normal(0, math.sqrt(2 / INPUT_COUNT), (INPUT_COUNT, HIDDEN_ONE)).astype(np.float32),
        "b1": np.zeros(HIDDEN_ONE, dtype=np.float32),
        "w2": rng.normal(0, math.sqrt(2 / HIDDEN_ONE), (HIDDEN_ONE, HIDDEN_TWO)).astype(np.float32),
        "b2": np.zeros(HIDDEN_TWO, dtype=np.float32),
        "wv": rng.normal(0, math.sqrt(1 / HIDDEN_TWO), (HIDDEN_TWO, 1)).astype(np.float32),
        "bv": np.zeros(1, dtype=np.float32),
        "wp": rng.normal(0, math.sqrt(1 / HIDDEN_TWO), (HIDDEN_TWO, POLICY_COUNT)).astype(np.float32),
        "bp": np.zeros(POLICY_COUNT, dtype=np.float32),
    }


def forward(parameters, x):
    z1 = x @ parameters["w1"] + parameters["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ parameters["w2"] + parameters["b2"]
    h2 = np.maximum(z2, 0)
    value = (h2 @ parameters["wv"] + parameters["bv"]).reshape(-1)
    policy = h2 @ parameters["wp"] + parameters["bp"]
    return z1, h1, z2, h2, value, policy


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def masked_softmax(logits, legal):
    masked = np.where(legal, logits, -1.0e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    values = np.exp(shifted) * legal
    return values / np.sum(values, axis=1, keepdims=True)


def metrics(parameters, data):
    _, _, _, _, value_logits, policy_logits = forward(parameters, data["x"])
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


def train(dataset, seed: int, maximum_epochs: int):
    rng, parameters = initialize(seed)
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
            legal = train_data["legal"][indices]
            policy_target_probabilities = train_data["policy_target"][indices]
            value_target = (train_data["value"][indices] + 1.0) * 0.5
            value_weight = train_data["value_weight"][indices]
            policy_weight = train_data["policy_weight"][indices]
            z1, h1, z2, h2, value_logits, policy_logits = forward(parameters, x)
            delta_value = (sigmoid(value_logits) - value_target) * value_weight
            delta_value /= max(float(np.sum(value_weight)), 1e-9)
            probabilities = masked_softmax(policy_logits, legal)
            delta_policy = probabilities - policy_target_probabilities
            delta_policy *= policy_weight[:, None]
            delta_policy /= max(float(np.sum(policy_weight)), 1e-9)
            gradients = {
                "wv": h2.T @ delta_value[:, None],
                "bv": np.asarray([np.sum(delta_value)], dtype=np.float32),
                "wp": h2.T @ delta_policy,
                "bp": np.sum(delta_policy, axis=0),
            }
            delta2 = (
                delta_value[:, None] @ parameters["wv"].T
                + delta_policy @ parameters["wp"].T
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
    for name in ("w1", "w2", "wv", "wp"):
        tensors[name], restored[name] = quantize_matrix(parameters[name])
    for name in ("b1", "b2", "bv", "bp"):
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
    policy_logits = parameters["bp"].copy()
    for hidden in range(HIDDEN_TWO):
        value_contribution = np.float32(
            np.float32(hidden_two[hidden] * np.float32(weights["wv"][hidden, 0]))
            * scales["wv"][0]
        )
        value_logit = np.float32(value_logit + value_contribution)
        for policy in range(POLICY_COUNT):
            contribution = np.float32(
                np.float32(
                    hidden_two[hidden] * np.float32(weights["wp"][hidden, policy])
                )
                * scales["wp"][policy]
            )
            policy_logits[policy] = np.float32(policy_logits[policy] + contribution)
    return {
        "input": "stride17-mod11-v1",
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
    parser.add_argument(
        "--selfplay-multiplier",
        type=float,
        default=1.0,
        help="relative value and policy mass for focus_agent_id=-1 self-play",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=40)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.selfplay_multiplier) or arguments.selfplay_multiplier <= 0:
        parser.error("--selfplay-multiplier must be finite and greater than zero")
    if arguments.selfplay_corpus is None and arguments.selfplay_multiplier != 1.0:
        parser.error("--selfplay-multiplier is only active with --selfplay-corpus")
    elite_paths = tuple(arguments.elite_corpus or DEFAULT_ELITE)
    public_paths = (*elite_paths, arguments.jacek_corpus)
    games, source_hashes = load_public_games(public_paths)
    selfplay_teachers = []
    selfplay_teacher_models = {}
    selfplay_policy_target_schemas = []
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
    buckets, dataset_report = dataset_from_games(games, arguments.selfplay_multiplier)
    datasets = {name: arrays(samples) for name, samples in buckets.items()}
    parameters, best_epoch = train(datasets, arguments.seed, arguments.epochs)
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
            "expert-dagger-policy-value-v1"
            if arguments.selfplay_corpus is not None
            else "expert-imitation-policy-value-v1"
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
            "policy": "canonical-primitive-direction-logits",
        },
        "training": {
            "seed": arguments.seed,
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
            "rank5_selfplay_used": "rank_5" in selfplay_teachers,
            "policy_weighting": (
                "equal mass per expert; Jacek mass multiplier 4; equal mass "
                "per retained trajectory payload within expert"
            ),
            "policy_target_training": (
                "weighted cross entropy; public and legacy self-play actions are "
                "one-hot; canonical-primitive-root-visits-v1 is trained as a "
                "normalized soft distribution"
            ),
            "value_weighting": (
                "equal mass per focus expert; equal mass per retained trajectory "
                "payload within expert; primitives before teacher_start_turn excluded; "
                "no source multiplier"
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
                for name in ("b1", "b2", "bv", "bp")
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
