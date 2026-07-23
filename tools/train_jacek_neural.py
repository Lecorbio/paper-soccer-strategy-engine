#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "jacek_article_value_model.json"
INPUT_COUNT = 1156
EDGE_COUNT = 316
VERTEX_COUNT = 105
DISTANCE_BUCKETS = 8
HIDDEN_ONE = 32
HIDDEN_TWO = 32
TARGET_TEMPERATURE = 12_000.0


def split_name(seed):
    bucket = hashlib.sha256(str(seed).encode("ascii")).digest()[0] % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def load_samples(path):
    buckets = {name: [] for name in ("train", "validation", "test")}
    raw = path.read_bytes()
    corpus_contract = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != "papersoccer.jacek-training-sample.v2":
            raise ValueError(f"invalid sample schema on line {line_number}")
        contract = {
            "feature_schema": record.get("feature_schema"),
            "rules": record.get("rules"),
            "teacher": record.get("teacher"),
        }
        if contract["feature_schema"] != (
                "canonical-edges316-onehot-true-turn-distance105x8-v1"):
            raise ValueError(
                f"invalid feature schema on line {line_number}")
        if contract["rules"] != {
                "width": 8,
                "height": 10,
                "goal_rule": "opponent-goal-only",
                "blocked_rule": "player-to-move-loses",
        }:
            raise ValueError(f"invalid rules contract on line {line_number}")
        teacher = contract["teacher"]
        if (not isinstance(teacher, dict) or
                teacher.get("kind") != "alpha-beta" or
                teacher.get("max_turn_depth") != 5 or
                not isinstance(teacher.get("max_nodes"), int) or
                teacher["max_nodes"] <= 0 or
                teacher.get("transposition_table_entries") != 16_384 or
                teacher.get("max_search_plies") != 12):
            raise ValueError(
                f"invalid teacher contract on line {line_number}")
        if corpus_contract is None:
            corpus_contract = contract
        elif contract != corpus_contract:
            raise ValueError(
                f"inconsistent corpus contract on line {line_number}")
        completed_depth = int(record["completed_depth"])
        if completed_depth <= 0:
            raise ValueError(
                f"unfinished teacher search on line {line_number}")
        active = [int(value) for value in record["active"]]
        if (len(active) < VERTEX_COUNT or active != sorted(set(active)) or
                active[0] < 0 or active[-1] >= INPUT_COUNT):
            raise ValueError(f"invalid active inputs on line {line_number}")
        active_set = set(active)
        for vertex in range(VERTEX_COUNT):
            first = EDGE_COUNT + vertex * DISTANCE_BUCKETS
            if sum((first + bucket) in active_set
                   for bucket in range(DISTANCE_BUCKETS)) != 1:
                raise ValueError(
                    f"invalid distance one-hot group on line {line_number}")
        player = int(record["player"])
        if player not in (0, 1):
            raise ValueError(f"invalid player on line {line_number}")
        teacher_score = float(record["teacher_score"])
        if not math.isfinite(teacher_score):
            raise ValueError(
                f"non-finite teacher score on line {line_number}")
        player_sign = 1.0 if player == 0 else -1.0
        mover_score = player_sign * teacher_score
        logit = np.clip(mover_score / TARGET_TEMPERATURE, -8.0, 8.0)
        target = 1.0 / (1.0 + math.exp(-float(logit)))
        depth = completed_depth
        nodes = max(1, int(record["nodes"]))
        weight = min(depth, 5) / 4.0 * math.sqrt(nodes / 4_000.0)
        buckets[split_name(record["seed"])].append(
            (active, target, weight, mover_score))

    arrays = {}
    seen = set()
    removed = {}
    for name in ("train", "validation", "test"):
        unique = []
        removed[name] = 0
        for active, target, weight, mover_score in buckets[name]:
            fingerprint = tuple(active)
            if fingerprint in seen:
                removed[name] += 1
                continue
            seen.add(fingerprint)
            unique.append((active, target, weight, mover_score))
        if not unique:
            raise ValueError(f"no unique {name} samples")
        features = np.zeros((len(unique), INPUT_COUNT), dtype=np.float32)
        for row, sample in enumerate(unique):
            features[row, sample[0]] = 1.0
        arrays[name] = (
            features,
            np.asarray([sample[1] for sample in unique], dtype=np.float32),
            np.asarray([sample[2] for sample in unique], dtype=np.float32),
            np.asarray([sample[3] for sample in unique], dtype=np.float32),
        )
    return arrays, removed, hashlib.sha256(raw).hexdigest(), corpus_contract


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def predict(parameters, features):
    hidden_one = np.maximum(
        features @ parameters["w1"] + parameters["b1"], 0.0)
    hidden_two = np.maximum(
        hidden_one @ parameters["w2"] + parameters["b2"], 0.0)
    return (hidden_two @ parameters["w3"] + parameters["b3"]).reshape(-1)


def calculate_metrics(parameters, data):
    features, target, weight, teacher_score = data
    logits = predict(parameters, features)
    losses = (np.maximum(logits, 0.0) - logits * target +
              np.log1p(np.exp(-np.abs(logits))))
    predicted_score = logits * TARGET_TEMPERATURE
    correlation = np.corrcoef(predicted_score, teacher_score)[0, 1]
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "samples": int(len(features)),
        "weighted_soft_bce":
            float(np.sum(losses * weight) / max(float(np.sum(weight)), 1e-9)),
        "teacher_score_mae":
            float(np.mean(np.abs(predicted_score - teacher_score))),
        "teacher_score_correlation": float(correlation),
        "teacher_sign_accuracy":
            float(np.mean((logits >= 0.0) == (teacher_score >= 0.0))),
    }


def train(train_data, validation_data, seed, maximum_epochs):
    rng = np.random.default_rng(seed)
    parameters = {
        "w1": rng.normal(
            0.0, math.sqrt(2.0 / INPUT_COUNT),
            (INPUT_COUNT, HIDDEN_ONE)).astype(np.float32),
        "b1": np.zeros(HIDDEN_ONE, dtype=np.float32),
        "w2": rng.normal(
            0.0, math.sqrt(2.0 / HIDDEN_ONE),
            (HIDDEN_ONE, HIDDEN_TWO)).astype(np.float32),
        "b2": np.zeros(HIDDEN_TWO, dtype=np.float32),
        "w3": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_TWO),
            (HIDDEN_TWO, 1)).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    train_x, train_target, train_weight, _ = train_data
    batch_size = 256
    step = 0
    best = None
    best_epoch = 0
    best_loss = float("inf")
    patience = 8
    for epoch in range(1, maximum_epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            indices = order[start:start + batch_size]
            x = train_x[indices]
            target = train_target[indices]
            weight = train_weight[indices]

            z1 = x @ parameters["w1"] + parameters["b1"]
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ parameters["w2"] + parameters["b2"]
            h2 = np.maximum(z2, 0.0)
            logits = (h2 @ parameters["w3"] + parameters["b3"]).reshape(-1)
            delta3 = ((sigmoid(logits) - target) * weight /
                      max(float(np.sum(weight)), 1e-9))[:, None]
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
                gradient = gradients[name]
                if name.startswith("w"):
                    gradient = gradient + 1e-5 * parameter
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                corrected_first = first[name] / (1.0 - 0.9 ** step)
                corrected_second = second[name] / (1.0 - 0.999 ** step)
                parameter -= (0.0015 * corrected_first /
                              (np.sqrt(corrected_second) + 1e-8))

        validation = calculate_metrics(parameters, validation_data)
        print(
            f"epoch {epoch}: validation BCE "
            f"{validation['weighted_soft_bce']:.6f}, correlation "
            f"{validation['teacher_score_correlation']:.4f}")
        if validation["weighted_soft_bce"] < best_loss - 1e-6:
            best_loss = validation["weighted_soft_bce"]
            best_epoch = epoch
            best = {name: value.copy() for name, value in parameters.items()}
        elif epoch - best_epoch >= patience:
            break
    if best is None:
        raise RuntimeError("training did not produce a model")
    return best, best_epoch


def tensor(value):
    return {
        "shape": list(value.shape),
        "values": [float(item) for item in value.reshape(-1)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the demo's Jacek-inspired value network.")
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--epochs", type=int, default=50)
    arguments = parser.parse_args()

    dataset, duplicates_removed, corpus_sha, corpus_contract = load_samples(
        arguments.corpus)
    parameters, best_epoch = train(
        dataset["train"], dataset["validation"],
        arguments.seed, arguments.epochs)
    metrics = {
        name: calculate_metrics(parameters, values)
        for name, values in dataset.items()
    }
    trainer_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema": "papersoccer.jacek-inspired-model.v1",
        "feature_schema":
            "canonical-edges316-onehot-true-turn-distance105x8-v1",
        "input_count": INPUT_COUNT,
        "edge_count": EDGE_COUNT,
        "vertex_count": VERTEX_COUNT,
        "distance_buckets": DISTANCE_BUCKETS,
        "hidden_one": HIDDEN_ONE,
        "hidden_two": HIDDEN_TWO,
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "opponent-goal-only",
            "blocked_rule": "player-to-move-loses",
        },
        "target": {
            "kind": "mover-relative-soft-alpha-beta-root-score",
            "temperature": TARGET_TEMPERATURE,
        },
        "training": {
            "corpus_sha256": corpus_sha,
            "trainer_sha256": trainer_sha,
            "seed": arguments.seed,
            "best_epoch": best_epoch,
            "duplicates_removed": duplicates_removed,
            "corpus_contract": corpus_contract,
        },
        "metrics": metrics,
        "model": {
            "w1": tensor(parameters["w1"]),
            "b1": tensor(parameters["b1"]),
            "w2": tensor(parameters["w2"]),
            "b2": tensor(parameters["b2"]),
            "w3": tensor(parameters["w3"]),
            "b3": tensor(parameters["b3"]),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "best_epoch": best_epoch,
        "metrics": metrics,
        "output": str(arguments.output),
    }, indent=2))


if __name__ == "__main__":
    main()
