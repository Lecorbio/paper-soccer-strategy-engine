#!/usr/bin/env python3

import base64
import hashlib
import json
import math
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT.parents[3]
    / "results"
    / "codingame"
    / "selfplay_nn"
    / "neural_model.json"
)
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
MIRRORED_DIRECTION = (0, 7, 6, 5, 4, 3, 2, 1)
HIDDEN_ONE = 16
HIDDEN_TWO = 16
POLICY_COUNT = 8
MAX_SAMPLES_PER_GAME = 160


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
EDGE_COUNT = len(EDGE_INDEX)
VERTEX_COUNT = len(POINTS)
INPUT_COUNT = EDGE_COUNT + VERTEX_COUNT + 2


def rotate(point):
    return WIDTH - point[0], HEIGHT + 2 - point[1]


def mirror(point):
    return WIDTH - point[0], point[1]


def canonicalize(ball, used_segments, visited, player, reflected):
    transform = lambda point: point
    if player == 1:
        transform = rotate
    ball = transform(ball)
    used_segments = {segment(transform(a), transform(b)) for a, b in used_segments}
    visited = {transform(point) for point in visited}
    if reflected:
        ball = mirror(ball)
        used_segments = {segment(mirror(a), mirror(b)) for a, b in used_segments}
        visited = {mirror(point) for point in visited}
    return ball, used_segments, visited


def turn_distances(ball, used_edges, visited):
    unreachable = 10**6
    distances = [unreachable] * VERTEX_COUNT
    start = POINT_INDEX[ball]
    distances[start] = 0
    queue = [start]
    while queue:
        vertex = queue.pop(0)
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
                queue.insert(0, destination)
            else:
                queue.append(destination)
    return [min(distance, 7) for distance in distances]


def feature_vector(ball, used_segments, visited, player, reflected=False):
    ball, used_segments, visited = canonicalize(
        ball, used_segments, visited, player, reflected
    )
    used_edges = {EDGE_INDEX[edge] for edge in used_segments}
    distances = turn_distances(ball, used_edges, visited)
    vector = np.zeros(INPUT_COUNT, dtype=np.float32)
    for edge in used_edges:
        vector[edge] = 1.0
    vector[EDGE_COUNT : EDGE_COUNT + VERTEX_COUNT] = (
        np.asarray(distances, dtype=np.float32) / 7.0
    )
    vector[-2] = ball[0] / WIDTH
    vector[-1] = ball[1] / (HEIGHT + 2)
    return vector


def normalized_direction(direction, player, reflected):
    result = (direction + 4) % 8 if player == 1 else direction
    return MIRRORED_DIRECTION[result] if reflected else result


def legal_policy_mask(ball, used_segments, player, reflected):
    mask = np.zeros(POLICY_COUNT, dtype=bool)
    for direction, (dx, dy) in enumerate(DIRECTIONS):
        destination = (ball[0] + dx, ball[1] + dy)
        edge = segment(ball, destination)
        if edge in EDGE_INDEX and edge not in used_segments:
            mask[normalized_direction(direction, player, reflected)] = True
    if not np.any(mask):
        raise RuntimeError("non-terminal sample has no legal direction")
    return mask


def game_samples(game):
    winner = int(game["winner"])
    teacher_start = int(game["teacher_start_turn"])
    ball = (WIDTH // 2, HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    samples = []
    for turn_index, action in enumerate(game["turns"]):
        if not action:
            raise RuntimeError(f"game {game['game']} contains an empty action")
        player = turn_index % 2
        for edge_index, encoded_direction in enumerate(action):
            direction = int(encoded_direction)
            if turn_index >= teacher_start:
                value = 1.0 if winner == player else -1.0
                samples.append(
                    (
                        feature_vector(ball, used_segments, visited, player),
                        normalized_direction(direction, player, False),
                        value,
                        legal_policy_mask(
                            ball, used_segments, player, reflected=False
                        ),
                        1.0 if edge_index == 0 else 0.0,
                    )
                )
                samples.append(
                    (
                        feature_vector(
                            ball, used_segments, visited, player, reflected=True
                        ),
                        normalized_direction(direction, player, True),
                        value,
                        legal_policy_mask(
                            ball, used_segments, player, reflected=True
                        ),
                        1.0 if edge_index == 0 else 0.0,
                    )
                )

            dx, dy = DIRECTIONS[direction]
            destination = (ball[0] + dx, ball[1] + dy)
            edge = segment(ball, destination)
            if edge not in EDGE_INDEX or edge in used_segments:
                raise RuntimeError(
                    f"illegal edge in game {game['game']} turn {turn_index}: {edge}"
                )
            was_visited = destination in visited
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
            if is_goal(ball):
                blocked = False
            else:
                blocked = not any(
                    candidate_edge not in used_segments
                    for candidate_edge in (
                        segment(ball, candidate) for candidate in neighbors(ball)
                    )
                    if candidate_edge in EDGE_INDEX
                    and not forbidden_boundary_segment(*candidate_edge)
                )
            terminal = is_goal(ball) or blocked
            continues = was_visited or is_boundary(ball)
            if edge_index + 1 < len(action) and (terminal or not continues):
                raise RuntimeError(
                    f"overlong action in game {game['game']} turn {turn_index}"
                )
            if edge_index + 1 == len(action) and not terminal and continues:
                raise RuntimeError(
                    f"incomplete rebound in game {game['game']} turn {turn_index}"
                )

    if len(samples) > MAX_SAMPLES_PER_GAME * 2:
        indices = np.linspace(
            0, len(samples) // 2 - 1, MAX_SAMPLES_PER_GAME, dtype=np.int64
        )
        selected = []
        for index in indices:
            selected.extend(samples[2 * int(index) : 2 * int(index) + 2])
        samples = selected
    return samples


def load_dataset(path):
    buckets = {"train": [], "validation": [], "test": []}
    games = {name: 0 for name in buckets}
    raw = path.read_bytes()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        game = json.loads(line)
        if game.get("schema") != "papersoccer.selfplay.v1":
            raise RuntimeError(f"invalid schema on line {line_number}")
        split_value = hashlib.sha256(
            str(int(game["seed"])).encode("ascii")
        ).digest()[0] % 10
        split = "train" if split_value < 8 else (
            "validation" if split_value == 8 else "test"
        )
        buckets[split].extend(game_samples(game))
        games[split] += 1
    overlap_removed = {"validation": 0, "test": 0}
    seen_features = {sample[0].tobytes() for sample in buckets["train"]}
    for split in ("validation", "test"):
        filtered = []
        for sample in buckets[split]:
            fingerprint = sample[0].tobytes()
            if fingerprint in seen_features:
                overlap_removed[split] += 1
                continue
            filtered.append(sample)
            seen_features.add(fingerprint)
        buckets[split] = filtered
    for split in buckets:
        if not buckets[split]:
            raise RuntimeError(f"self-play corpus has no {split} samples")
    arrays = {}
    for split, samples in buckets.items():
        arrays[split] = (
            np.stack([sample[0] for sample in samples]),
            np.asarray([sample[1] for sample in samples], dtype=np.int64),
            np.asarray([sample[2] for sample in samples], dtype=np.float32),
            np.stack([sample[3] for sample in samples]),
            np.asarray([sample[4] for sample in samples], dtype=np.float32),
        )
    return arrays, games, overlap_removed, hashlib.sha256(raw).hexdigest()


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def softmax(logits, legal_mask=None):
    masked = np.where(legal_mask, logits, -1e9) if legal_mask is not None else logits
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -30.0, 30.0))
    if legal_mask is not None:
        values = np.where(legal_mask, values, 0.0)
    return values / np.sum(values, axis=1, keepdims=True)


def predict(parameters, x):
    hidden_one = np.maximum(x @ parameters["w1"] + parameters["b1"], 0.0)
    hidden_two = np.maximum(
        hidden_one @ parameters["w2"] + parameters["b2"], 0.0
    )
    value = (hidden_two @ parameters["wv"] + parameters["bv"]).reshape(-1)
    policy = hidden_two @ parameters["wp"] + parameters["bp"]
    return value, policy


def loss_and_metrics(
    parameters, x, policy_target, value_target, legal_mask, value_mask
):
    value_logits, policy_logits = predict(parameters, x)
    binary_target = (value_target + 1.0) * 0.5
    boundary = value_mask > 0.0
    value_loss = float(
        np.mean(
            np.maximum(value_logits[boundary], 0.0)
            - value_logits[boundary] * binary_target[boundary]
            + np.log1p(np.exp(-np.abs(value_logits[boundary])))
        )
    )
    probabilities = softmax(policy_logits, legal_mask)
    policy_loss = float(
        -np.mean(np.log(np.maximum(probabilities[np.arange(len(x)), policy_target], 1e-9)))
    )
    order = np.argsort(np.where(legal_mask, policy_logits, -1e9), axis=1)
    return {
        "samples": len(x),
        "value_loss": value_loss,
        "value_samples": int(np.sum(boundary)),
        "value_accuracy": float(
            np.mean(
                (value_logits[boundary] >= 0.0)
                == (value_target[boundary] > 0.0)
            )
        ),
        "policy_loss": policy_loss,
        "policy_top1": float(np.mean(order[:, -1] == policy_target)),
        "policy_top3": float(
            np.mean(np.any(order[:, -3:] == policy_target[:, None], axis=1))
        ),
        "combined_loss": value_loss + 0.35 * policy_loss,
    }


def train_network(train, validation, seed=20260722):
    train_x, train_policy, train_value, train_legal, train_value_mask = train
    (
        validation_x,
        validation_policy,
        validation_value,
        validation_legal,
        validation_value_mask,
    ) = validation
    rng = np.random.default_rng(seed)
    parameters = {
        "w1": rng.normal(
            0.0, math.sqrt(2.0 / INPUT_COUNT), (INPUT_COUNT, HIDDEN_ONE)
        ).astype(np.float32),
        "b1": np.zeros(HIDDEN_ONE, dtype=np.float32),
        "w2": rng.normal(
            0.0, math.sqrt(2.0 / HIDDEN_ONE), (HIDDEN_ONE, HIDDEN_TWO)
        ).astype(np.float32),
        "b2": np.zeros(HIDDEN_TWO, dtype=np.float32),
        "wv": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_TWO), (HIDDEN_TWO, 1)
        ).astype(np.float32),
        "bv": np.zeros(1, dtype=np.float32),
        "wp": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_TWO), (HIDDEN_TWO, POLICY_COUNT)
        ).astype(np.float32),
        "bp": np.zeros(POLICY_COUNT, dtype=np.float32),
    }
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    boundary_values = train_value[train_value_mask > 0.0]
    positive_fraction = float(np.mean(boundary_values > 0.0))
    positive_weight = 0.5 / max(positive_fraction, 1e-3)
    negative_weight = 0.5 / max(1.0 - positive_fraction, 1e-3)
    best = None
    best_loss = float("inf")
    best_epoch = 0
    step = 0
    batch_size = 256
    learning_rate = 0.0015
    policy_weight = 0.35
    patience = 18
    for epoch in range(1, 121):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_x[indices]
            policy_target = train_policy[indices]
            value_target = train_value[indices]
            legal_mask = train_legal[indices]
            value_mask = train_value_mask[indices]
            z1 = x @ parameters["w1"] + parameters["b1"]
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ parameters["w2"] + parameters["b2"]
            h2 = np.maximum(z2, 0.0)
            value_logits = (h2 @ parameters["wv"] + parameters["bv"]).reshape(-1)
            policy_logits = h2 @ parameters["wp"] + parameters["bp"]

            binary_target = (value_target + 1.0) * 0.5
            value_weights = np.where(
                value_target > 0.0, positive_weight, negative_weight
            )
            value_denominator = max(float(np.sum(value_mask)), 1.0)
            delta_value = (
                (sigmoid(value_logits) - binary_target)
                * value_weights
                * value_mask
                / value_denominator
            )[:, None]
            probabilities = softmax(policy_logits, legal_mask)
            probabilities[np.arange(len(indices)), policy_target] -= 1.0
            delta_policy = probabilities * (policy_weight / len(indices))

            gradients = {
                "wv": h2.T @ delta_value,
                "bv": delta_value.sum(axis=0),
                "wp": h2.T @ delta_policy,
                "bp": delta_policy.sum(axis=0),
            }
            delta2 = (
                delta_value @ parameters["wv"].T
                + delta_policy @ parameters["wp"].T
            ) * (z2 > 0.0)
            gradients["w2"] = h1.T @ delta2
            gradients["b2"] = delta2.sum(axis=0)
            delta1 = (delta2 @ parameters["w2"].T) * (z1 > 0.0)
            gradients["w1"] = x.T @ delta1
            gradients["b1"] = delta1.sum(axis=0)

            step += 1
            for name, parameter in parameters.items():
                gradient = gradients[name]
                if name.startswith("w"):
                    gradient = gradient + 1e-5 * parameter
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                corrected_first = first[name] / (1.0 - 0.9**step)
                corrected_second = second[name] / (1.0 - 0.999**step)
                parameter -= (
                    learning_rate
                    * corrected_first
                    / (np.sqrt(corrected_second) + 1e-8)
                )

        validation_metrics = loss_and_metrics(
            parameters, validation_x, validation_policy, validation_value,
            validation_legal, validation_value_mask
        )
        if validation_metrics["combined_loss"] < best_loss - 1e-5:
            best_loss = validation_metrics["combined_loss"]
            best = {name: value.copy() for name, value in parameters.items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break
    return best, best_epoch


def quantize(array):
    maximum = float(np.max(np.abs(array)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    values = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return {
        "shape": list(array.shape),
        "scale": scale,
        "data": base64.b64encode(values.tobytes()).decode(),
    }


def dequantize(tensor):
    values = np.frombuffer(base64.b64decode(tensor["data"]), dtype=np.int8)
    return values.reshape(tensor["shape"]).astype(np.float32) * tensor["scale"]


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: train_selfplay_model.py SELFPLAY.jsonl [neural_model.json]"
        )
    input_path = pathlib.Path(sys.argv[1])
    output_path = (
        pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT
    )
    dataset, game_counts, overlap_removed, corpus_hash = load_dataset(input_path)
    parameters, best_epoch = train_network(
        dataset["train"], dataset["validation"]
    )
    quantized = {
        name: quantize(parameters[name]) for name in ("w1", "w2", "wv", "wp")
    }
    restored = {
        name: dequantize(quantized[name]) for name in quantized
    }
    for name in ("b1", "b2", "bv", "bp"):
        restored[name] = parameters[name]

    report = {
        "schema": "papersoccer.policy-value-model.v1",
        "input_count": INPUT_COUNT,
        "edge_count": EDGE_COUNT,
        "vertex_count": VERTEX_COUNT,
        "hidden_one": HIDDEN_ONE,
        "hidden_two": HIDDEN_TWO,
        "policy_count": POLICY_COUNT,
        "feature_schema": "canonical-edges316-turn-distance105-ballxy-v1",
        "corpus_sha256": corpus_hash,
        "games": game_counts,
        "held_out_feature_overlaps_removed": overlap_removed,
        "best_epoch": best_epoch,
        "metrics": {
            "float_train": loss_and_metrics(parameters, *dataset["train"]),
            "float_validation": loss_and_metrics(parameters, *dataset["validation"]),
            "float_test": loss_and_metrics(parameters, *dataset["test"]),
            "quantized_validation": loss_and_metrics(restored, *dataset["validation"]),
            "quantized_test": loss_and_metrics(restored, *dataset["test"]),
        },
        "model": {
            **quantized,
            "b1": [float(value) for value in parameters["b1"]],
            "b2": [float(value) for value in parameters["b2"]],
            "bv": [float(value) for value in parameters["bv"]],
            "bp": [float(value) for value in parameters["bp"]],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "model"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
