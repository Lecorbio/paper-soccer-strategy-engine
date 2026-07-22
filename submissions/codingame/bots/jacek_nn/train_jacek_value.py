#!/usr/bin/env python3

import base64
import collections
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
WIDTH = 8
HEIGHT = 10
DIRECTIONS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)
HIDDEN_ONE = 32
HIDDEN_TWO = 32
QUANTIZATION_BITS = 3
QUANTIZATION_LIMIT = 3
TARGET_TEMPERATURE = 12_000.0
MATE_THRESHOLD = 900_000
MAX_SAMPLES_PER_GAME = 24


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
    return x in (0, WIDTH) or (y in (1, HEIGHT + 1) and x != WIDTH // 2)


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
        (a[1] == b[1] and a[1] in (1, HEIGHT + 1) and dx == 1)
        or (a[0] == b[0] and a[0] in (0, WIDTH) and dy == 1)
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
INPUT_COUNT = EDGE_COUNT + VERTEX_COUNT * 8


def rotate(point):
    return WIDTH - point[0], HEIGHT + 2 - point[1]


def mirror(point):
    return WIDTH - point[0], point[1]


def transformed(point, player, reflected):
    result = rotate(point) if player == 1 else point
    return mirror(result) if reflected else result


def turn_distances(ball, used_edges, visited):
    unreachable = 1_000_000
    distances = [unreachable] * VERTEX_COUNT
    start = POINT_INDEX[ball]
    distances[start] = 0
    queue = collections.deque((start,))
    while queue:
        vertex = queue.popleft()
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
                queue.appendleft(destination)
            else:
                queue.append(destination)
    return [min(distance, 7) for distance in distances]


def feature_vector(ball, used_segments, visited, player, reflected=False):
    ball = transformed(ball, player, reflected)
    normalized_segments = {
        segment(transformed(a, player, reflected), transformed(b, player, reflected))
        for a, b in used_segments
    }
    normalized_visited = {
        transformed(point, player, reflected) for point in visited
    }
    used_edges = {EDGE_INDEX[edge] for edge in normalized_segments}
    distances = turn_distances(ball, used_edges, normalized_visited)
    vector = np.zeros(INPUT_COUNT, dtype=np.uint8)
    if used_edges:
        vector[np.fromiter(used_edges, dtype=np.int64)] = 1
    for vertex, distance in enumerate(distances):
        vector[EDGE_COUNT + vertex * 8 + distance] = 1
    return vector


def reconstruct(transcript):
    ball = (WIDTH // 2, HEIGHT // 2 + 1)
    used_segments = set()
    visited = {ball}
    turns = [] if transcript == "" else transcript.split("/")
    for turn_index, action in enumerate(turns):
        if not action:
            raise RuntimeError("transcript contains an empty turn")
        for encoded in action:
            direction = int(encoded)
            dx, dy = DIRECTIONS[direction]
            destination = (ball[0] + dx, ball[1] + dy)
            edge = segment(ball, destination)
            if edge not in EDGE_INDEX or edge in used_segments:
                raise RuntimeError(f"illegal transcript edge {edge}")
            used_segments.add(edge)
            visited.add(destination)
            ball = destination
    return ball, used_segments, visited, len(turns) % 2


def split_bucket(seed):
    digest = hashlib.sha256(str(int(seed)).encode("ascii")).digest()[0] % 10
    return "train" if digest < 8 else "validation" if digest == 8 else "test"


def arena_holdouts():
    raw = (ROOT / "known_arena_loss_regressions.json").read_bytes()
    report = json.loads(raw)
    if report.get("schema") != "papersoccer.known-arena-loss-regressions.v1":
        raise RuntimeError("invalid arena-loss holdout schema")
    result = {(int(case["player_id"]), case["prefix"]) for case in report["cases"]}
    observed = (ROOT / "observed_arena_loss_regressions.hpp").read_text()
    declared = re.search(r"std::array<Case,\s*(\d+)>\s+kCases", observed)
    cases = re.findall(
        r'\{\s*"(?:[^"\\]|\\.)*"\s*,\s*(\d+)\s*,\s*"([0-7/]*)"\s*\}',
        observed,
    )
    if declared is None or len(cases) != int(declared.group(1)):
        raise RuntimeError("could not parse every observed arena-loss holdout")
    for player, prefix in cases:
        result.add((int(player), prefix))
    digest = hashlib.sha256(raw + observed.encode()).hexdigest()
    return result, digest


def load_dataset(path, include_mates):
    raw = path.read_bytes()
    holdouts, holdout_hash = arena_holdouts()
    holdout_features = set()
    for player, transcript in holdouts:
        ball, used_segments, visited, mover = reconstruct(transcript)
        if mover != player:
            raise RuntimeError("arena-loss holdout player does not match transcript")
        for reflected in (False, True):
            holdout_features.add(
                feature_vector(
                    ball, used_segments, visited, player, reflected
                ).tobytes()
            )
    buckets = {name: [] for name in ("train", "validation", "test")}
    game_counts = {name: 0 for name in buckets}
    rejected = collections.Counter()
    state_cache = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        game = json.loads(line)
        if game.get("schema") != "papersoccer.teacher-residual-samples.v1":
            raise RuntimeError(f"invalid teacher schema on line {line_number}")
        split = split_bucket(game["seed"])
        game_counts[split] += 1
        eligible = []
        for sample in game["samples"]:
            player = int(sample["player_id"])
            transcript = sample["transcript"]
            if int(sample["completed_depth"]) < 2:
                rejected["shallow"] += 1
                continue
            teacher = int(sample["teacher_score"])
            if abs(teacher) >= MATE_THRESHOLD and not include_mates:
                rejected["mate"] += 1
                continue
            if (player, transcript) in holdouts:
                rejected["arena_holdout"] += 1
                continue
            eligible.append(sample)
        if len(eligible) > MAX_SAMPLES_PER_GAME:
            indices = np.linspace(
                0, len(eligible) - 1, MAX_SAMPLES_PER_GAME, dtype=np.int64
            )
            eligible = [eligible[int(index)] for index in indices]
        for sample in eligible:
            player = int(sample["player_id"])
            transcript = sample["transcript"]
            cache_key = (player, transcript)
            if cache_key not in state_cache:
                state = reconstruct(transcript)
                if state[3] != player:
                    raise RuntimeError("teacher sample player does not match transcript")
                state_cache[cache_key] = state[:3]
            ball, used_segments, visited = state_cache[cache_key]
            sign = 1.0 if player == 0 else -1.0
            mover_score = sign * float(sample["teacher_score"])
            target = 1.0 / (
                1.0 + math.exp(-np.clip(mover_score / TARGET_TEMPERATURE, -8, 8))
            )
            depth = int(sample["completed_depth"])
            budget = int(sample["node_budget"])
            weight = min(depth, 6) / 4.0 * math.sqrt(budget / 16_000.0)
            for reflected in (False, True):
                vector = feature_vector(
                    ball, used_segments, visited, player, reflected
                )
                if vector.tobytes() in holdout_features:
                    rejected["arena_holdout_feature"] += 1
                    continue
                buckets[split].append((vector, target, weight, mover_score))

    seen = set()
    overlaps_removed = collections.Counter()
    arrays = {}
    for split in ("train", "validation", "test"):
        unique = []
        for sample in buckets[split]:
            fingerprint = sample[0].tobytes()
            if fingerprint in seen:
                overlaps_removed[split] += 1
                continue
            seen.add(fingerprint)
            unique.append(sample)
        if not unique:
            raise RuntimeError(f"teacher corpus has no {split} samples")
        arrays[split] = (
            np.stack([sample[0] for sample in unique]),
            np.asarray([sample[1] for sample in unique], dtype=np.float32),
            np.asarray([sample[2] for sample in unique], dtype=np.float32),
            np.asarray([sample[3] for sample in unique], dtype=np.float32),
        )
    return (
        arrays,
        game_counts,
        dict(rejected),
        dict(overlaps_removed),
        hashlib.sha256(raw).hexdigest(),
        holdout_hash,
    )


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def quantize_values(array):
    maximum = np.max(np.abs(array), axis=0)
    scales = np.where(maximum > 0, maximum / QUANTIZATION_LIMIT, 1.0).astype(
        np.float32
    )
    values = np.clip(
        np.rint(array / scales), -QUANTIZATION_LIMIT, QUANTIZATION_LIMIT
    ).astype(np.int8)
    return values, scales


def fake_quantize(array):
    values, scales = quantize_values(array)
    return values.astype(np.float32) * scales


def effective_parameters(parameters, quantized):
    result = {name: value for name, value in parameters.items()}
    if quantized:
        for name in ("w1", "w2", "w3"):
            result[name] = fake_quantize(parameters[name])
    return result


def predict(parameters, x):
    hidden_one = np.maximum(x @ parameters["w1"] + parameters["b1"], 0.0)
    hidden_two = np.maximum(
        hidden_one @ parameters["w2"] + parameters["b2"], 0.0
    )
    return (hidden_two @ parameters["w3"] + parameters["b3"]).reshape(-1)


def metrics(parameters, data):
    x, target, weight, teacher_score = data
    logits = predict(parameters, x)
    losses = np.maximum(logits, 0.0) - logits * target + np.log1p(
        np.exp(-np.abs(logits))
    )
    denominator = max(float(np.sum(weight)), 1e-9)
    centered_teacher = teacher_score - np.mean(teacher_score)
    predicted_score = logits * TARGET_TEMPERATURE
    centered_prediction = predicted_score - np.mean(predicted_score)
    correlation_denominator = math.sqrt(
        float(np.sum(centered_teacher**2) * np.sum(centered_prediction**2))
    )
    return {
        "samples": len(x),
        "weighted_soft_bce": float(np.sum(losses * weight) / denominator),
        "teacher_score_mae": float(np.mean(np.abs(predicted_score - teacher_score))),
        "teacher_score_rmse": float(
            np.sqrt(np.mean((predicted_score - teacher_score) ** 2))
        ),
        "teacher_score_correlation": (
            float(np.sum(centered_teacher * centered_prediction) / correlation_denominator)
            if correlation_denominator > 0
            else 0.0
        ),
        "teacher_sign_accuracy": float(
            np.mean((logits >= 0.0) == (teacher_score >= 0.0))
        ),
    }


def train_phase(parameters, train, validation, rng, epochs, learning_rate,
                quantized, patience):
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    step = 0
    best = None
    best_loss = float("inf")
    best_epoch = 0
    train_x, train_target, train_weight, _ = train
    batch_size = 256
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_x[indices]
            target = train_target[indices]
            weight = train_weight[indices]
            effective = effective_parameters(parameters, quantized)
            z1 = x @ effective["w1"] + parameters["b1"]
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ effective["w2"] + parameters["b2"]
            h2 = np.maximum(z2, 0.0)
            logits = (h2 @ effective["w3"] + parameters["b3"]).reshape(-1)
            delta3 = (
                (sigmoid(logits) - target) * weight / max(float(np.sum(weight)), 1e-9)
            )[:, None]
            gradients = {
                "w3": h2.T @ delta3,
                "b3": delta3.sum(axis=0),
            }
            delta2 = (delta3 @ effective["w3"].T) * (z2 > 0.0)
            gradients["w2"] = h1.T @ delta2
            gradients["b2"] = delta2.sum(axis=0)
            delta1 = (delta2 @ effective["w2"].T) * (z1 > 0.0)
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
        evaluated = effective_parameters(parameters, quantized)
        validation_loss = metrics(evaluated, validation)["weighted_soft_bce"]
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best = {name: value.copy() for name, value in parameters.items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break
    if best is None:
        raise RuntimeError("training phase did not produce a model")
    return best, best_epoch


def train_network(train, validation, seed=20260722):
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
        "w3": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_TWO), (HIDDEN_TWO, 1)
        ).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }
    float_best, float_epoch = train_phase(
        parameters, train, validation, rng, 100, 0.0015, False, 16
    )
    quantized_best, quantized_epoch = train_phase(
        float_best, train, validation, rng, 35, 0.00035, True, 10
    )
    return quantized_best, float_epoch, quantized_epoch


def pack_three_bit(values):
    output = bytearray()
    buffer = 0
    bits = 0
    for value in values.reshape(-1):
        buffer |= (int(value) & 7) << bits
        bits += 3
        while bits >= 8:
            output.append(buffer & 0xFF)
            buffer >>= 8
            bits -= 8
    if bits:
        output.append(buffer & 0xFF)
    return bytes(output)


def quantized_tensor(array):
    values, scales = quantize_values(array)
    return {
        "shape": list(array.shape),
        "bits": QUANTIZATION_BITS,
        "scales": [float(value) for value in scales.reshape(-1)],
        "data": base64.b64encode(pack_three_bit(values)).decode("ascii"),
    }


def main():
    arguments = sys.argv[1:]
    include_mates = True
    if "--exclude-mates" in arguments:
        arguments.remove("--exclude-mates")
        include_mates = False
    if len(arguments) not in (1, 2):
        raise SystemExit(
            "usage: train_jacek_value.py TEACHER_SAMPLES.jsonl [MODEL.json] "
            "[--exclude-mates]"
        )
    input_path = pathlib.Path(arguments[0])
    output_path = pathlib.Path(arguments[1]) if len(arguments) == 2 else ROOT / "jacek_value_model.json"
    (
        dataset,
        game_counts,
        rejected,
        overlaps_removed,
        corpus_hash,
        holdout_hash,
    ) = load_dataset(input_path, include_mates)
    parameters, float_epoch, quantized_epoch = train_network(
        dataset["train"], dataset["validation"]
    )
    effective = effective_parameters(parameters, True)
    report = {
        "schema": "papersoccer.jacek-value-model.v1",
        "input_count": INPUT_COUNT,
        "edge_count": EDGE_COUNT,
        "vertex_count": VERTEX_COUNT,
        "hidden_one": HIDDEN_ONE,
        "hidden_two": HIDDEN_TWO,
        "quantization_bits": QUANTIZATION_BITS,
        "feature_schema": "canonical-edges316-onehot-true-turn-distance105x8-v1",
        "target": "mover-relative-soft-teacher-root-score",
        "target_temperature": TARGET_TEMPERATURE,
        "include_mates": include_mates,
        "corpus_sha256": corpus_hash,
        "trainer_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "arena_holdout_sha256": holdout_hash,
        "games": game_counts,
        "samples": {name: len(values[0]) for name, values in dataset.items()},
        "rejected": rejected,
        "held_out_feature_overlaps_removed": overlaps_removed,
        "best_float_epoch": float_epoch,
        "best_quantized_epoch": quantized_epoch,
        "metrics": {
            name: metrics(effective, values) for name, values in dataset.items()
        },
        "model": {
            "w1": quantized_tensor(parameters["w1"]),
            "b1": [float(value) for value in parameters["b1"]],
            "w2": quantized_tensor(parameters["w2"]),
            "b2": [float(value) for value in parameters["b2"]],
            "w3": quantized_tensor(parameters["w3"]),
            "b3": [float(value) for value in parameters["b3"]],
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "model"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
