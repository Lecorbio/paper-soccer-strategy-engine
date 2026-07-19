#!/usr/bin/env python3

import base64
import concurrent.futures
import json
import math
import pathlib
import random
import urllib.request

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
AGENT_ID = 6_273_433
WIDTH = 8
HEIGHT = 10
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


def post(service, payload):
    request = urllib.request.Request(
        f"https://www.codingame.com/services/{service}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def is_regular(point):
    x, y = point
    return 0 <= x <= WIDTH and 1 <= y <= HEIGHT + 1


def is_goal(point):
    x, y = point
    return 3 <= x <= 5 and y in (0, HEIGHT + 2)


def is_boundary(point):
    x, y = point
    if not is_regular(point):
        return False
    if x in (0, WIDTH):
        return True
    return y in (1, HEIGHT + 1) and x != WIDTH // 2


def segment(first, second):
    return tuple(sorted((first, second), key=lambda point: (point[1], point[0])))


def forbidden_boundary_segment(first, second):
    a, b = segment(first, second)
    touches_goal = is_goal(a) or is_goal(b)
    if touches_goal:
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


def neighbors(point):
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
        goal_y = 0 if y == 1 else HEIGHT + 2
        for goal_x in range(3, 6):
            candidate = (goal_x, goal_y)
            if max(abs(x - goal_x), abs(y - goal_y)) == 1:
                result.append(candidate)
    return result


def make_topology():
    points = [(x, y) for y in range(1, HEIGHT + 2) for x in range(WIDTH + 1)]
    for x in range(3, 6):
        points.extend(((x, 0), (x, HEIGHT + 2)))
    point_index = {point: index for index, point in enumerate(points)}
    edge_index = {}
    adjacency = [[] for _ in points]
    for source_index, source in enumerate(points):
        if not is_regular(source):
            continue
        for destination in neighbors(source):
            if forbidden_boundary_segment(source, destination):
                continue
            edge = segment(source, destination)
            index = edge_index.setdefault(edge, len(edge_index))
            adjacency[source_index].append((point_index[destination], index))
    if len(points) != 105 or len(edge_index) != 316:
        raise RuntimeError(
            f"unexpected topology: {len(points)} vertices, {len(edge_index)} edges"
        )
    return points, point_index, edge_index, adjacency


POINTS, POINT_INDEX, EDGE_INDEX, ADJACENCY = make_topology()
INPUT_COUNT = len(EDGE_INDEX) + len(POINTS) * 8


def rotate(point):
    return WIDTH - point[0], HEIGHT + 2 - point[1]


def turn_distances(ball, used_edges, visited):
    unreachable = 10**6
    distances = [unreachable] * len(POINTS)
    start = POINT_INDEX[ball]
    distances[start] = 0
    queue = [start]
    head = 0
    while head < len(queue):
        vertex = queue[head]
        head += 1
        for destination, edge in ADJACENCY[vertex]:
            if edge in used_edges:
                continue
            destination_point = POINTS[destination]
            cost = 0 if (
                destination_point in visited
                or is_boundary(destination_point)
                or is_goal(destination_point)
            ) else 1
            candidate = distances[vertex] + cost
            if candidate >= distances[destination]:
                continue
            distances[destination] = candidate
            if cost == 0:
                queue.insert(head, destination)
            else:
                queue.append(destination)
    return [min(distance, 7) for distance in distances]


def feature_vector(ball, used_segments, visited, player):
    if player == 1:
        ball = rotate(ball)
        used_segments = {segment(rotate(a), rotate(b)) for a, b in used_segments}
        visited = {rotate(point) for point in visited}
    used_edges = {EDGE_INDEX[edge] for edge in used_segments}
    distances = turn_distances(ball, used_edges, visited)
    vector = np.zeros(INPUT_COUNT, dtype=np.float32)
    for edge in used_edges:
        vector[edge] = 1.0
    offset = len(EDGE_INDEX)
    for vertex, distance in enumerate(distances):
        vector[offset + vertex * 8 + distance] = 1.0
    return vector


def replay_samples(game):
    jacek = next(agent["index"] for agent in game["agents"] if agent["agentId"] == AGENT_ID)
    winner = game["ranks"].index(0)
    turns = [
        (frame["agentId"], (frame.get("stdout") or "").strip())
        for frame in game["frames"]
        if frame.get("agentId", -1) >= 0
    ]
    ball = (WIDTH // 2, HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    samples = []
    for turn_index, (player, action) in enumerate(turns):
        label = 1.0 if player == winner else -1.0
        samples.append((feature_vector(ball, used_segments, visited, player), label))
        mover = player
        for direction in action:
            dx, dy = DIRECTIONS[int(direction)]
            destination = (ball[0] + dx, ball[1] + dy)
            edge = segment(ball, destination)
            if edge in used_segments or edge not in EDGE_INDEX:
                raise RuntimeError(f"illegal replay edge in game {game['gameId']}: {edge}")
            was_visited = destination in visited
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
            if is_goal(ball):
                break
            continues = was_visited or is_boundary(ball)
            if not continues:
                mover = 1 - mover
        if not is_goal(ball) and mover == player and turn_index + 1 < len(turns):
            raise RuntimeError(f"incomplete replay action in game {game['gameId']}")
    return samples, jacek, winner


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def train_network(train_x, train_y, validation_x, validation_y, seed=20260719):
    rng = np.random.default_rng(seed)
    hidden_one = 8
    hidden_two = 8
    parameters = {
        "w1": rng.normal(0.0, math.sqrt(2.0 / INPUT_COUNT), (INPUT_COUNT, hidden_one)).astype(np.float32),
        "b1": np.zeros(hidden_one, dtype=np.float32),
        "w2": rng.normal(0.0, math.sqrt(2.0 / hidden_one), (hidden_one, hidden_two)).astype(np.float32),
        "b2": np.zeros(hidden_two, dtype=np.float32),
        "w3": rng.normal(0.0, math.sqrt(1.0 / hidden_two), (hidden_two, 1)).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    best = None
    best_loss = float("inf")
    step = 0
    batch_size = 256
    learning_rate = 0.002
    for epoch in range(220):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_x[indices]
            target = (train_y[indices] + 1.0) * 0.5
            z1 = x @ parameters["w1"] + parameters["b1"]
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ parameters["w2"] + parameters["b2"]
            h2 = np.maximum(z2, 0.0)
            logits = (h2 @ parameters["w3"] + parameters["b3"]).reshape(-1)
            delta3 = (sigmoid(logits) - target)[:, None] / len(indices)
            gradients = {
                "w3": h2.T @ delta3,
                "b3": delta3.sum(axis=0),
            }
            delta2 = (delta3 @ parameters["w3"].T) * (z2 > 0.0)
            gradients["w2"] = h1.T @ delta2
            gradients["b2"] = delta2.sum(axis=0)
            delta1 = (delta2 @ parameters["w2"].T) * (z1 > 0.0)
            gradients["w1"] = x.T @ delta1
            gradients["b1"] = delta1.sum(axis=0)
            step += 1
            for name, parameter in parameters.items():
                gradient = gradients[name] + (1e-5 * parameter if name.startswith("w") else 0.0)
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                corrected_first = first[name] / (1.0 - 0.9**step)
                corrected_second = second[name] / (1.0 - 0.999**step)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        validation_logits = predict_logits(parameters, validation_x)
        validation_target = (validation_y + 1.0) * 0.5
        validation_loss = float(
            np.mean(
                np.maximum(validation_logits, 0.0)
                - validation_logits * validation_target
                + np.log1p(np.exp(-np.abs(validation_logits)))
            )
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best = {name: value.copy() for name, value in parameters.items()}
    return best


def predict_logits(parameters, x):
    h1 = np.maximum(x @ parameters["w1"] + parameters["b1"], 0.0)
    h2 = np.maximum(h1 @ parameters["w2"] + parameters["b2"], 0.0)
    return (h2 @ parameters["w3"] + parameters["b3"]).reshape(-1)


def quantize(array):
    maximum = float(np.max(np.abs(array)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    values = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return {
        "shape": list(array.shape),
        "scale": scale,
        "data": base64.b64encode(values.tobytes()).decode(),
    }


def dequantize(value):
    data = np.frombuffer(base64.b64decode(value["data"]), dtype=np.int8)
    return data.reshape(value["shape"]).astype(np.float32) * value["scale"]


def metrics(parameters, x, y):
    logits = predict_logits(parameters, x)
    predictions = np.where(logits >= 0.0, 1.0, -1.0)
    target = (y + 1.0) * 0.5
    loss = float(
        np.mean(
            np.maximum(logits, 0.0)
            - logits * target
            + np.log1p(np.exp(-np.abs(logits)))
        )
    )
    return {
        "samples": len(y),
        "accuracy": float(np.mean(predictions == y)),
        "loss": loss,
        "mean_logit": float(np.mean(logits)),
    }


def main():
    battles = post(
        "gamesPlayersRankingRemoteService/findLastBattlesByAgentId",
        [AGENT_ID, None],
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        games = list(
            executor.map(
                lambda battle: post(
                    "gameResultRemoteService/findByGameId",
                    [battle["gameId"], None],
                ),
                battles,
            )
        )

    random.Random(20260719).shuffle(games)
    validation_game_ids = {game["gameId"] for game in games[:18]}
    train_samples = []
    validation_samples = []
    results = {"jacek_wins": 0, "jacek_losses": 0}
    for game in games:
        samples, jacek, winner = replay_samples(game)
        results["jacek_wins" if jacek == winner else "jacek_losses"] += 1
        target = validation_samples if game["gameId"] in validation_game_ids else train_samples
        target.extend(samples)

    train_x = np.stack([sample[0] for sample in train_samples])
    train_y = np.asarray([sample[1] for sample in train_samples], dtype=np.float32)
    validation_x = np.stack([sample[0] for sample in validation_samples])
    validation_y = np.asarray([sample[1] for sample in validation_samples], dtype=np.float32)
    train_hashes = {row.tobytes() for row in train_x}
    unseen_mask = np.asarray(
        [row.tobytes() not in train_hashes for row in validation_x], dtype=bool
    )
    parameters = train_network(train_x, train_y, validation_x, validation_y)

    quantized_parameters = {
        "w1": quantize(parameters["w1"]),
        "b1": [float(value) for value in parameters["b1"]],
        "w2": quantize(parameters["w2"]),
        "b2": [float(value) for value in parameters["b2"]],
        "w3": quantize(parameters["w3"]),
        "b3": [float(value) for value in parameters["b3"]],
    }
    restored_parameters = {
        "w1": dequantize(quantized_parameters["w1"]),
        "b1": np.asarray(quantized_parameters["b1"], dtype=np.float32),
        "w2": dequantize(quantized_parameters["w2"]),
        "b2": np.asarray(quantized_parameters["b2"], dtype=np.float32),
        "w3": dequantize(quantized_parameters["w3"]),
        "b3": np.asarray(quantized_parameters["b3"], dtype=np.float32),
    }
    report = {
        "schema": "papersoccer.replay-value-model.v1",
        "agent_id": AGENT_ID,
        "games": len(games),
        **results,
        "input_count": INPUT_COUNT,
        "edge_count": len(EDGE_INDEX),
        "vertex_count": len(POINTS),
        "validation_game_ids": sorted(validation_game_ids),
        "train_positive_fraction": float(np.mean(train_y > 0.0)),
        "validation_positive_fraction": float(np.mean(validation_y > 0.0)),
        "train": metrics(parameters, train_x, train_y),
        "validation": metrics(parameters, validation_x, validation_y),
        "validation_unseen": metrics(
            parameters, validation_x[unseen_mask], validation_y[unseen_mask]
        ),
        "quantized_validation": metrics(
            restored_parameters, validation_x, validation_y
        ),
        "model": quantized_parameters,
    }
    output = ROOT / "replay_value_model.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "model"}, indent=2))
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
