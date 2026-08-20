import hashlib
import json
import math
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
INPUT_COUNT = 24
TARGET_SCALE = 20_000.0
MATE_THRESHOLD = 900_000
HARD_CAP = 6_000.0


def split_bucket(seed):
    value = int(seed) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value % 10


def load_samples(path):
    raw = path.read_bytes()
    corpus_hash = hashlib.sha256(raw).hexdigest()
    buckets = {"train": [], "validation": [], "test": []}
    games = {name: set() for name in buckets}
    rejected = {
        "shallow": 0,
        "mate": 0,
        "direct_goal": 0,
        "invalid": 0,
        "overlap": 0,
        "arena_holdout": 0,
        "opening_gate": 0,
    }
    holdout_path = ROOT / "known_arena_loss_regressions.json"
    holdout_raw = holdout_path.read_bytes()
    holdout = json.loads(holdout_raw)
    if holdout.get("schema") != "papersoccer.known-arena-loss-regressions.v1":
        raise RuntimeError("invalid arena-loss holdout schema")
    held_out_states = {
        (int(case["player_id"]), case["prefix"])
        for case in holdout["cases"]
    }
    for line_number, line in enumerate(raw.decode().splitlines(), 1):
        game = json.loads(line)
        if game.get("schema") != "papersoccer.teacher-residual-samples.v1":
            raise RuntimeError(f"invalid schema on line {line_number}")
        bucket = split_bucket(game["seed"])
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        games[split].add(int(game["game"]))
        for sample in game["samples"]:
            features = np.asarray(sample["features"], dtype=np.float64)
            if features.shape != (INPUT_COUNT,) or not np.all(np.isfinite(features)):
                rejected["invalid"] += 1
                continue
            if int(sample["completed_depth"]) < 2:
                rejected["shallow"] += 1
                continue
            teacher = int(sample["teacher_score"])
            anchor = int(sample["anchor_score"])
            if abs(teacher) >= MATE_THRESHOLD:
                rejected["mate"] += 1
                continue
            if features[20] != 0.0:
                rejected["direct_goal"] += 1
                continue
            if (int(sample["player_id"]), sample["transcript"]) in held_out_states:
                rejected["arena_holdout"] += 1
                continue
            if features[23] * 64.0 < 12.0:
                rejected["opening_gate"] += 1
                continue
            sign = 1 if int(sample["player_id"]) == 0 else -1
            residual = float(sign * (teacher - anchor))
            target = float(np.clip(residual / TARGET_SCALE, -1.0, 1.0))
            depth = int(sample["completed_depth"])
            node_budget = int(sample["node_budget"])
            weight = min(depth, 6) / 4.0 * math.sqrt(node_budget / 16_000.0)
            buckets[split].append(
                (features, target, weight, residual, anchor, depth)
            )

    seen = set()
    for split in ("train", "validation", "test"):
        unique = []
        for sample in buckets[split]:
            fingerprint = sample[0].astype(np.float32).tobytes()
            if fingerprint in seen:
                rejected["overlap"] += 1
                continue
            seen.add(fingerprint)
            unique.append(sample)
        buckets[split] = unique
        if not unique:
            raise RuntimeError(f"teacher corpus has no {split} samples")
    return (
        buckets,
        {key: len(value) for key, value in games.items()},
        rejected,
        corpus_hash,
        hashlib.sha256(holdout_raw).hexdigest(),
    )


def arrays(samples):
    return (
        np.stack([sample[0] for sample in samples]),
        np.asarray([sample[1] for sample in samples]),
        np.asarray([sample[2] for sample in samples]),
        np.asarray([sample[3] for sample in samples]),
        np.asarray([sample[4] for sample in samples]),
        np.asarray([sample[5] for sample in samples]),
    )


def fit_huber(x, y, sample_weight, ridge, delta):
    design = np.column_stack((x, np.ones(len(x))))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = ridge * 0.01
    for _ in range(40):
        residual = y - design @ coefficients
        robust = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-9))
        weights = sample_weight * robust
        normal = design.T @ (design * weights[:, None]) + penalty
        right = design.T @ (y * weights)
        updated = np.linalg.solve(normal, right)
        if np.max(np.abs(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    return coefficients[:-1], float(coefficients[-1])


def predict(weights, bias, x):
    return np.clip(x @ weights + bias, -1.0, 1.0)


def inference_correction(prediction, anchor, strength):
    raw = prediction * TARGET_SCALE * strength / 100.0
    cap = np.maximum(0.0, HARD_CAP - np.abs(anchor) / 10.0)
    return np.clip(raw, -cap, cap)


def metrics(weights, bias, data, strength=100):
    x, target, _, residual, anchor, depth = data
    prediction = predict(weights, bias, x)
    correction = inference_correction(prediction, anchor, strength)
    errors = residual - correction
    centered_target = target - np.mean(target)
    centered_prediction = prediction - np.mean(prediction)
    denominator = math.sqrt(
        float(np.sum(centered_target**2) * np.sum(centered_prediction**2))
    )
    correlation = (
        float(np.sum(centered_target * centered_prediction) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "samples": len(x),
        "target_mae": float(np.mean(np.abs(target - prediction))),
        "target_rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "target_correlation": correlation,
        "residual_sign_accuracy": float(
            np.mean((prediction >= 0.0) == (target >= 0.0))
        ),
        "anchor_teacher_mae": float(np.mean(np.abs(residual))),
        "corrected_teacher_mae": float(np.mean(np.abs(errors))),
        "corrected_teacher_rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_completed_depth": float(np.mean(depth)),
        "strength_percent": strength,
    }


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: train_teacher_residual.py SAMPLES.jsonl [MODEL.json]"
        )
    input_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else ROOT / "teacher_residual_model.json"
    samples, game_counts, rejected, corpus_hash, holdout_hash = load_samples(input_path)
    dataset = {name: arrays(values) for name, values in samples.items()}

    best = None
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        for delta in (0.10, 0.20, 0.35):
            weights, bias = fit_huber(
                dataset["train"][0], dataset["train"][1],
                dataset["train"][2], ridge, delta
            )
            validation = metrics(weights, bias, dataset["validation"])
            score = validation["target_mae"]
            if best is None or score < best[0]:
                best = (score, ridge, delta, weights, bias)

    _, ridge, delta, weights, bias = best
    strength_sweep = {}
    for strength in (10, 20, 30, 40, 50, 75, 100):
        strength_sweep[str(strength)] = metrics(
            weights, bias, dataset["validation"], strength
        )["corrected_teacher_mae"]
    recommended_strength = int(min(strength_sweep, key=strength_sweep.get))

    report = {
        "schema": "papersoccer.teacher-residual-model.v1",
        "input_count": INPUT_COUNT,
        "feature_schema": "rank5-hidden2-hand-scalars-used-edge-phase-v2",
        "target": "clipped-mover-relative-teacher-minus-rank5-anchor",
        "target_scale": int(TARGET_SCALE),
        "hard_cap": int(HARD_CAP),
        "trainer_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "corpus_sha256": corpus_hash,
        "arena_loss_holdout_sha256": holdout_hash,
        "games": game_counts,
        "samples": {name: len(values) for name, values in samples.items()},
        "rejected": rejected,
        "fit": {
            "loss": "huber-irls",
            "ridge": ridge,
            "huber_delta": delta,
            "recommended_strength_percent": recommended_strength,
            "validation_strength_mae": strength_sweep,
        },
        "metrics": {
            name: metrics(weights, bias, data)
            for name, data in dataset.items()
        },
        "model": {
            "weights": [float(value) for value in weights],
            "bias": bias,
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "model"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
