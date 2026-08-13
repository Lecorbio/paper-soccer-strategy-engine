#!/usr/bin/env python3
"""Fit a compact ordering-only successor ranker on fresh procedural labels.

The corpus generator uses the frozen Rank-4 search as a proof-off teacher at
two work budgets.  This trainer retains only pair orderings that agree at both
budgets, assigns train/validation/test by whole procedural root, and never
reads replay or arena payloads.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


FEATURE_COUNT = 24
CORPUS_SCHEMA = "papersoccer.rank4-jacek-successor.v2"
META_SCHEMA = "papersoccer.rank4-jacek-corpus-meta.v2"
MODEL_SCHEMA = "papersoccer.rank4-jacek-successor-ranker.v2"
SPLITS = ("train", "validation", "test")
VARIANTS = ("base", "mirror", "rotate", "rotate_mirror")
MIRROR_DIRECTIONS = str.maketrans("01234567", "07654321")
ROTATE_DIRECTIONS = str.maketrans("01234567", "45670123")


@dataclass(frozen=True)
class Candidate:
    root_id: str
    split: str
    variant: str
    root_depth: int
    root_mover_sign: int
    successor_mover_sign: int
    action: str
    features: tuple[float, ...]
    low_score: int
    high_score: int
    low_depth: int
    high_depth: int


@dataclass(frozen=True)
class Pair:
    root_id: str
    split: str
    variant: str
    root_depth: int
    root_mover_sign: int
    successor_mover_sign: int
    delta: tuple[float, ...]  # worse features minus better features
    margin_low: int
    margin_high: int
    weight: float
    better_action: str
    worse_action: str


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def load_corpus(path: pathlib.Path) -> tuple[dict[str, object], list[Candidate]]:
    meta: dict[str, object] | None = None
    candidates: list[Candidate] = []
    root_splits: dict[str, str] = {}
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}") from error
            schema = row.get("schema")
            if line_number == 1:
                if schema != META_SCHEMA:
                    raise ValueError("the first row is not corpus metadata")
                if row.get("teacher") != "frozen-rank-4-proof-off":
                    raise ValueError("unexpected teacher identity")
                if row.get("rules") != "8x10;own-goals-allowed;mover-loses":
                    raise ValueError("unexpected rules identity")
                if not isinstance(row.get("seed"), int):
                    raise ValueError("missing procedural seed")
                if row.get("mirror") is not True or row.get("color_swap") is not True:
                    raise ValueError("v2 requires mirror and color-swap variants")
                if tuple(row.get("variants", ())) != VARIANTS:
                    raise ValueError("unexpected v2 variant registry")
                meta = row
                continue
            if schema != CORPUS_SCHEMA:
                raise ValueError(f"unexpected schema at line {line_number}")
            root_id = row.get("root_id")
            split = row.get("split")
            variant = row.get("variant")
            action = row.get("action")
            if not isinstance(root_id, str) or not root_id.startswith("fresh-r"):
                raise ValueError(f"invalid root_id at line {line_number}")
            if split not in SPLITS:
                raise ValueError(f"invalid split at line {line_number}")
            if variant not in VARIANTS:
                raise ValueError(f"invalid variant at line {line_number}")
            if not isinstance(action, str) or not action or any(
                direction not in "01234567" for direction in action
            ):
                raise ValueError(f"invalid action at line {line_number}")
            prior_split = root_splits.setdefault(root_id, split)
            if prior_split != split:
                raise ValueError(f"whole-root split leakage for {root_id}")
            raw_features = row.get("features")
            if not isinstance(raw_features, list) or len(raw_features) != FEATURE_COUNT:
                raise ValueError(f"invalid features at line {line_number}")
            features = tuple(
                finite_number(value, f"feature {index} at line {line_number}")
                for index, value in enumerate(raw_features)
            )
            root_depth = int(row["root_depth"])
            root_mover_sign = int(row["root_mover_sign"])
            successor_mover_sign = int(row["successor_mover_sign"])
            if root_depth not in (4, 8, 12, 20):
                raise ValueError(f"invalid root depth at line {line_number}")
            if root_mover_sign not in (-1, 1) or successor_mover_sign not in (-1, 1):
                raise ValueError(f"invalid mover sign at line {line_number}")
            if successor_mover_sign != -root_mover_sign:
                raise ValueError(f"successor is not a turn boundary at line {line_number}")
            candidates.append(
                Candidate(
                    root_id=root_id,
                    split=split,
                    variant=variant,
                    root_depth=root_depth,
                    root_mover_sign=root_mover_sign,
                    successor_mover_sign=successor_mover_sign,
                    action=action,
                    features=features,
                    low_score=int(row["low_score"]),
                    high_score=int(row["high_score"]),
                    low_depth=int(row["low_depth"]),
                    high_depth=int(row["high_depth"]),
                )
            )
    if meta is None or not candidates:
        raise ValueError("empty corpus")
    verify_corpus_balance(meta, candidates)
    return meta, candidates


def verify_corpus_balance(
    meta: dict[str, object], candidates: Sequence[Candidate]
) -> None:
    families: dict[str, dict[str, list[Candidate]]] = {}
    for candidate in candidates:
        families.setdefault(candidate.root_id, {}).setdefault(
            candidate.variant, []
        ).append(candidate)
    if len(families) != int(meta["roots"]):
        raise ValueError("metadata root count does not match corpus")
    for root_id, variants in families.items():
        if tuple(sorted(variants)) != tuple(sorted(VARIANTS)):
            raise ValueError(f"root family lacks four variants: {root_id}")
        counts = {variant: len(rows) for variant, rows in variants.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"root family candidate count is color-imbalanced: {root_id}")
        identities = {
            variant: {
                (row.split, row.root_depth, row.root_mover_sign)
                for row in rows
            }
            for variant, rows in variants.items()
        }
        if any(len(values) != 1 for values in identities.values()):
            raise ValueError(f"root family variant mixes metadata: {root_id}")
        base = next(iter(identities["base"]))
        mirror = next(iter(identities["mirror"]))
        rotate = next(iter(identities["rotate"]))
        rotate_mirror = next(iter(identities["rotate_mirror"]))
        if base != mirror or rotate != rotate_mirror:
            raise ValueError(f"root family reflection metadata mismatch: {root_id}")
        if base[:2] != rotate[:2] or base[2] != -rotate[2]:
            raise ValueError(f"root family color rotation metadata mismatch: {root_id}")
        for variant, rows in variants.items():
            actions = {row.action for row in rows}
            if len(actions) != len(rows):
                raise ValueError(f"duplicate action in {root_id}/{variant}")

    # Enforce exact row and root-variant balance inside every split/depth cell.
    for split in SPLITS:
        for depth in (4, 8, 12, 20):
            cell = [
                row
                for row in candidates
                if row.split == split and row.root_depth == depth
            ]
            if not cell:
                raise ValueError(f"empty split/depth cell: {split}/{depth}")
            signs = collections.Counter(row.root_mover_sign for row in cell)
            variants = collections.Counter(row.variant for row in cell)
            if signs[-1] != signs[1] or len(set(variants.values())) != 1:
                raise ValueError(f"non-exact mover/variant balance: {split}/{depth}")


def group_candidates(
    candidates: Iterable[Candidate],
) -> dict[tuple[str, str], list[Candidate]]:
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.root_id, candidate.variant), []).append(candidate)
    return groups


def make_pairs(
    candidates: Sequence[Candidate], minimum_margin: int, minimum_depth: int
) -> list[Pair]:
    if candidates and (
        len({candidate.root_depth for candidate in candidates}) != 1
        or len({candidate.root_mover_sign for candidate in candidates}) != 1
        or len({candidate.successor_mover_sign for candidate in candidates}) != 1
        or len({candidate.split for candidate in candidates}) != 1
    ):
        raise ValueError("candidate group mixes root metadata")
    provisional: list[tuple[Candidate, Candidate, int, int]] = []
    for left_index, left in enumerate(candidates):
        if left.low_depth < minimum_depth or left.high_depth < minimum_depth:
            continue
        for right in candidates[left_index + 1 :]:
            if right.low_depth < minimum_depth or right.high_depth < minimum_depth:
                continue
            low_delta = left.low_score - right.low_score
            high_delta = left.high_score - right.high_score
            if (
                low_delta == 0
                or high_delta == 0
                or (low_delta < 0) != (high_delta < 0)
                or abs(low_delta) < minimum_margin
                or abs(high_delta) < minimum_margin
            ):
                continue
            better, worse = (left, right) if high_delta < 0 else (right, left)
            provisional.append((better, worse, abs(low_delta), abs(high_delta)))
    if not provisional:
        return []
    # A complex rebound root can expose many more endpoints than an ordinary
    # root.  Equal total weight per root/variant prevents it from dominating.
    weight = 1.0 / len(provisional)
    return [
        Pair(
            root_id=better.root_id,
            split=better.split,
            variant=better.variant,
            root_depth=better.root_depth,
            root_mover_sign=better.root_mover_sign,
            successor_mover_sign=better.successor_mover_sign,
            delta=tuple(
                worse_value - better_value
                for better_value, worse_value in zip(
                    better.features, worse.features, strict=True
                )
            ),
            margin_low=low_margin,
            margin_high=high_margin,
            weight=weight,
            better_action=better.action,
            worse_action=worse.action,
        )
        for better, worse, low_margin, high_margin in provisional
    ]


def all_pairs(
    groups: dict[tuple[str, str], list[Candidate]],
    minimum_margin: int,
    minimum_depth: int,
) -> list[Pair]:
    result: list[Pair] = []
    for candidates in groups.values():
        result.extend(make_pairs(candidates, minimum_margin, minimum_depth))
    return result


def dot(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


def pair_accuracy(pairs: Sequence[Pair], weights: Sequence[float]) -> dict[str, float]:
    correct = 0.0
    total = 0.0
    raw_correct = 0
    for pair in pairs:
        prediction = dot(weights, pair.delta)
        correct += pair.weight * (1.0 if prediction > 0 else 0.5 if prediction == 0 else 0.0)
        total += pair.weight
        raw_correct += prediction > 0
    return {
        "pairs": len(pairs),
        "root_balanced_accuracy": correct / total if total else 0.0,
        "raw_accuracy": raw_correct / len(pairs) if pairs else 0.0,
    }


def baseline_weights(feature: int) -> list[float]:
    weights = [0.0] * FEATURE_COUNT
    weights[feature] = 1.0
    return weights


def residual_score(candidate: Candidate, residual: dict[str, object]) -> float:
    model = residual["model"]
    if not isinstance(model, dict):
        raise ValueError("invalid residual model")
    weights = model["weights"]
    if not isinstance(weights, list) or len(weights) != FEATURE_COUNT:
        raise ValueError("invalid residual weights")
    prediction = float(model["bias"]) + dot(
        [float(value) for value in weights], candidate.features
    )
    prediction = max(-1.0, min(1.0, prediction))
    anchor = candidate.features[10] * 100_000.0
    confidence_cap = max(0.0, 6000.0 - abs(anchor) / 10.0)
    correction = max(-confidence_cap, min(confidence_cap, prediction * 20_000.0))
    if candidate.features[20] != 0.0 or candidate.features[23] * 64.0 < 12.0:
        correction = 0.0
    return (anchor + correction) / 100_000.0


def score_pair_function(
    pairs: Sequence[Pair],
    groups: dict[tuple[str, str], list[Candidate]],
    scorer,
) -> dict[str, float]:
    # The incumbent residual confidence cap is nonlinear and therefore cannot
    # be represented by a single weight vector.
    candidate_scores: dict[tuple[str, str, str], float] = {}
    for (root_id, variant), candidates in groups.items():
        for candidate in candidates:
            candidate_scores[(root_id, variant, candidate.action)] = scorer(candidate)
    correct = 0.0
    total = 0.0
    raw_correct = 0
    for pair in pairs:
        better = candidate_scores[(pair.root_id, pair.variant, pair.better_action)]
        worse = candidate_scores[(pair.root_id, pair.variant, pair.worse_action)]
        prediction = worse - better
        outcome = 1.0 if prediction > 0 else 0.5 if prediction == 0 else 0.0
        correct += pair.weight * outcome
        total += pair.weight
        raw_correct += prediction > 0
    return {
        "pairs": len(pairs),
        "root_balanced_accuracy": correct / total if total else 0.0,
        "raw_accuracy": raw_correct / len(pairs) if pairs else 0.0,
    }


def feature_scales(train_pairs: Sequence[Pair]) -> list[float]:
    totals = [0.0] * FEATURE_COUNT
    weight_sum = 0.0
    for pair in train_pairs:
        for index, value in enumerate(pair.delta):
            totals[index] += pair.weight * value * value
        weight_sum += pair.weight
    if weight_sum == 0.0:
        raise ValueError("training split has no stable pairs")
    return [
        max(1.0e-4, math.sqrt(total / weight_sum)) for total in totals
    ]


def train_logistic(
    train_pairs: Sequence[Pair],
    *,
    seed: int,
    ridge: float,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    scales = feature_scales(train_pairs)
    random_source = random.Random(seed)
    weights = [random_source.gauss(0.0, 0.01) for _ in range(FEATURE_COUNT)]
    first_moment = [0.0] * FEATURE_COUNT
    second_moment = [0.0] * FEATURE_COUNT
    order = list(range(len(train_pairs)))
    step = 0
    batch_size = min(128, len(order))
    for epoch in range(epochs):
        random_source.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            gradients = [ridge * value for value in weights]
            total_weight = sum(train_pairs[index].weight for index in batch)
            for pair_index in batch:
                pair = train_pairs[pair_index]
                normalized = [
                    value / scale
                    for value, scale in zip(pair.delta, scales, strict=True)
                ]
                logit = max(-40.0, min(40.0, dot(weights, normalized)))
                factor = -pair.weight / ((1.0 + math.exp(logit)) * total_weight)
                for feature, value in enumerate(normalized):
                    gradients[feature] += factor * value
            step += 1
            rate = learning_rate * (0.25 + 0.75 * (1.0 - epoch / epochs))
            for feature, gradient in enumerate(gradients):
                first_moment[feature] = 0.9 * first_moment[feature] + 0.1 * gradient
                second_moment[feature] = (
                    0.999 * second_moment[feature] + 0.001 * gradient * gradient
                )
                corrected_first = first_moment[feature] / (1.0 - 0.9**step)
                corrected_second = second_moment[feature] / (1.0 - 0.999**step)
                weights[feature] -= rate * corrected_first / (
                    math.sqrt(corrected_second) + 1.0e-8
                )
    return [
        weight / scale for weight, scale in zip(weights, scales, strict=True)
    ]


def quantize(weights: Sequence[float]) -> tuple[list[int], float, list[float]]:
    maximum = max(abs(value) for value in weights)
    if maximum == 0.0:
        raise ValueError("cannot quantize an all-zero model")
    scale = maximum / 127.0
    integers = [max(-127, min(127, round(value / scale))) for value in weights]
    restored = [value * scale for value in integers]
    return integers, scale, restored


def top_one_metrics(
    groups: dict[tuple[str, str], list[Candidate]],
    split: str,
    weights: Sequence[float],
) -> dict[str, float]:
    count = 0
    exact = 0
    normalized_regret = 0.0
    for candidates in groups.values():
        if not candidates or candidates[0].split != split:
            continue
        predicted = min(candidates, key=lambda candidate: dot(weights, candidate.features))
        best_score = min(candidate.high_score for candidate in candidates)
        exact += predicted.high_score == best_score
        normalized_regret += (predicted.high_score - best_score) / 100_000.0
        count += 1
    return {
        "groups": count,
        "exact_best_rate": exact / count if count else 0.0,
        "mean_normalized_regret": normalized_regret / count if count else 0.0,
    }


def symmetry_metrics(
    groups: dict[tuple[str, str], list[Candidate]], weights: Sequence[float]
) -> dict[str, object]:
    transforms = {
        "horizontal_base": ("base", "mirror", MIRROR_DIRECTIONS),
        "horizontal_rotated": ("rotate", "rotate_mirror", MIRROR_DIRECTIONS),
        "color_rotation": ("base", "rotate", ROTATE_DIRECTIONS),
        "color_rotation_mirrored": (
            "mirror",
            "rotate_mirror",
            ROTATE_DIRECTIONS,
        ),
    }
    return {
        name: one_symmetry_metric(groups, weights, source, target, directions)
        for name, (source, target, directions) in transforms.items()
    }


def one_symmetry_metric(
    groups: dict[tuple[str, str], list[Candidate]],
    weights: Sequence[float],
    source_variant: str,
    target_variant: str,
    directions: dict[int, int],
) -> dict[str, float]:
    compared = 0
    selection_matches = 0
    teacher_selection_matches = 0
    score_differences: list[float] = []
    teacher_score_differences: list[float] = []
    root_ids = {root_id for root_id, _ in groups}
    for root_id in root_ids:
        base = groups.get((root_id, source_variant))
        mirror = groups.get((root_id, target_variant))
        if not base or not mirror:
            continue
        mirror_by_action = {candidate.action: candidate for candidate in mirror}
        base_best = min(base, key=lambda candidate: dot(weights, candidate.features))
        expected_action = base_best.action.translate(directions)
        mirror_best = min(mirror, key=lambda candidate: dot(weights, candidate.features))
        selection_matches += mirror_best.action == expected_action
        teacher_base_best = min(base, key=lambda candidate: candidate.high_score)
        teacher_mirror_best = min(mirror, key=lambda candidate: candidate.high_score)
        teacher_selection_matches += teacher_mirror_best.action == (
            teacher_base_best.action.translate(directions)
        )
        compared += 1
        for candidate in base:
            mirrored = mirror_by_action.get(candidate.action.translate(directions))
            if mirrored is not None:
                score_differences.append(
                    abs(dot(weights, candidate.features) - dot(weights, mirrored.features))
                )
                teacher_score_differences.append(
                    abs(candidate.high_score - mirrored.high_score) / 100_000.0
                )
    return {
        "roots": compared,
        "selection_match_rate": selection_matches / compared if compared else 0.0,
        "teacher_selection_match_rate": (
            teacher_selection_matches / compared if compared else 0.0
        ),
        "mean_paired_score_abs_difference": (
            sum(score_differences) / len(score_differences)
            if score_differences
            else 0.0
        ),
        "teacher_mean_paired_value_abs_difference": (
            sum(teacher_score_differences) / len(teacher_score_differences)
            if teacher_score_differences
            else 0.0
        ),
    }


def candidate_counts(candidates: Sequence[Candidate]) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLITS:
        selected = [candidate for candidate in candidates if candidate.split == split]
        result[split] = {
            "candidates": len(selected),
            "roots": len({candidate.root_id for candidate in selected}),
            "groups": len({(candidate.root_id, candidate.variant) for candidate in selected}),
        }
    return result


def pair_counts(pairs: Sequence[Pair]) -> dict[str, int]:
    return {split: sum(pair.split == split for pair in pairs) for split in SPLITS}


def sliced_pair_metrics(
    pairs: Sequence[Pair], weights: Sequence[float]
) -> dict[str, object]:
    result: dict[str, object] = {}
    dimensions = {
        "root_depth": (4, 8, 12, 20),
        "root_mover_sign": (-1, 1),
        "successor_mover_sign": (-1, 1),
        "variant": VARIANTS,
    }
    for name, values in dimensions.items():
        result[name] = {
            str(value): pair_accuracy(
                [pair for pair in pairs if getattr(pair, name) == value], weights
            )
            for value in values
        }
    return result


def render_header(quantized: Sequence[int]) -> str:
    values = ",".join(str(value) for value in quantized)
    return (
        "#pragma once\n"
        "#include<array>\n#include<cstdint>\n"
        "namespace papersoccer::turn_action_v2::successor_rank_model{\n"
        f"inline constexpr std::array<std::int8_t,24>W{{{{{values}}}}};\n"
        "}\n"
    )


def fit(args: argparse.Namespace) -> dict[str, object]:
    corpus_path = args.corpus.resolve()
    meta, candidates = load_corpus(corpus_path)
    groups = group_candidates(candidates)
    pairs = all_pairs(groups, args.minimum_margin, args.minimum_depth)
    split_pairs = {
        split: [pair for pair in pairs if pair.split == split] for split in SPLITS
    }
    if any(not split_pairs[split] for split in SPLITS):
        raise ValueError("every split must contain at least one stable pair")

    trials: list[dict[str, object]] = []
    selected_key: tuple[float, float, int, float] | None = None
    selected_weights: list[float] | None = None
    selected_trial: dict[str, object] | None = None
    for ridge in args.ridge:
        for seed in args.fit_seed:
            weights = train_logistic(
                split_pairs["train"],
                seed=seed,
                ridge=ridge,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            train_metric = pair_accuracy(split_pairs["train"], weights)
            validation_metric = pair_accuracy(split_pairs["validation"], weights)
            trial = {
                "seed": seed,
                "ridge": ridge,
                "train": train_metric,
                "validation": validation_metric,
            }
            trials.append(trial)
            key = (
                float(validation_metric["root_balanced_accuracy"]),
                float(train_metric["root_balanced_accuracy"]),
                -seed,
                -ridge,
            )
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_weights = weights
                selected_trial = trial
    assert selected_weights is not None and selected_trial is not None
    weights = selected_weights
    quantized, quantization_scale, restored = quantize(weights)

    residual_path = args.residual_model.resolve()
    residual = json.loads(residual_path.read_text(encoding="utf-8"))
    baseline: dict[str, object] = {"anchor_feature_10": {}, "incumbent_residual": {}}
    metrics: dict[str, object] = {
        "float": {},
        "int8": {},
        "top_one": {},
        "int8_slices": {},
    }
    for split in SPLITS:
        baseline["anchor_feature_10"][split] = pair_accuracy(
            split_pairs[split], baseline_weights(10)
        )
        baseline["incumbent_residual"][split] = score_pair_function(
            split_pairs[split], groups, lambda candidate: residual_score(candidate, residual)
        )
        metrics["float"][split] = pair_accuracy(split_pairs[split], weights)
        metrics["int8"][split] = pair_accuracy(split_pairs[split], restored)
        metrics["top_one"][split] = top_one_metrics(groups, split, restored)
        metrics["int8_slices"][split] = sliced_pair_metrics(
            split_pairs[split], restored
        )

    header_text = render_header(quantized)
    hybrid_source = args.hybrid_source.resolve()
    hybrid_bytes = hybrid_source.read_bytes()
    if not hybrid_bytes.isascii():
        raise ValueError("hybrid generated source is not ASCII")
    validation_gain = (
        metrics["int8"]["validation"]["root_balanced_accuracy"]
        - baseline["incumbent_residual"]["validation"]["root_balanced_accuracy"]
    )
    test_gain = (
        metrics["int8"]["test"]["root_balanced_accuracy"]
        - baseline["incumbent_residual"]["test"]["root_balanced_accuracy"]
    )
    quantization_loss = (
        metrics["float"]["validation"]["root_balanced_accuracy"]
        - metrics["int8"]["validation"]["root_balanced_accuracy"]
    )
    symmetry = symmetry_metrics(groups, restored)
    projected_minimum = len(hybrid_bytes) + len(header_text.encode("ascii")) + 450
    projected_maximum = len(hybrid_bytes) + len(header_text.encode("ascii")) + 650
    qualification_checks = {
        "validation_residual_gain_at_least_0_01": validation_gain >= 0.01,
        "test_residual_gain_at_least_0_01": test_gain >= 0.01,
        "quantization_loss_at_most_0_005": quantization_loss <= 0.005,
        "symmetry_selection_not_materially_below_teacher": all(
            metric["selection_match_rate"] + 0.02
            >= metric["teacher_selection_match_rate"]
            for metric in symmetry.values()
        ),
        "projected_maximum_source_within_99999": projected_maximum <= 99_999,
    }

    output = {
        "schema": MODEL_SCHEMA,
        "provenance": {
            "corpus": str(corpus_path),
            "corpus_sha256": sha256(corpus_path),
            "trainer_sha256": sha256(pathlib.Path(__file__).resolve()),
            "generator_sha256": sha256(args.generator.resolve()),
            "teacher_source_sha256": sha256(args.teacher_source.resolve()),
            "residual_model_sha256": sha256(residual_path),
            "hybrid_source_sha256": hashlib.sha256(hybrid_bytes).hexdigest(),
            "corpus_meta": meta,
            "forbidden_inputs": [
                "arena payloads",
                "replay payloads",
                "validation opening banks",
                "final opening banks",
            ],
        },
        "selection": {
            "method": "validation-root-balanced-pair-accuracy",
            "minimum_margin": args.minimum_margin,
            "minimum_depth": args.minimum_depth,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "trials": trials,
            "selected": {
                "seed": selected_trial["seed"],
                "ridge": selected_trial["ridge"],
            },
        },
        "data": {
            "candidates": candidate_counts(candidates),
            "stable_pairs": pair_counts(pairs),
        },
        "baseline": baseline,
        "metrics": metrics,
        "symmetry": symmetry,
        "qualification": {
            "recommended_for_timed_ablation": all(qualification_checks.values()),
            "checks": qualification_checks,
            "validation_gain_over_incumbent_residual": validation_gain,
            "test_gain_over_incumbent_residual": test_gain,
            "validation_quantization_loss": quantization_loss,
            "source_projection": {
                "current_hybrid_ascii_bytes": len(hybrid_bytes),
                "model_header_ascii_bytes": len(header_text.encode("ascii")),
                "integration_hook_estimate_bytes": [450, 650],
                "projected_flattened_source_bytes": [
                    projected_minimum,
                    projected_maximum,
                ],
            },
        },
        "model": {
            "feature_schema": "rank4-hidden8-hand-scalars-used-edge-phase-v2",
            "integration": "ordering-only; lower successor-mover score is better",
            "float_weights": weights,
            "int8_weights": quantized,
            "quantization_scale": quantization_scale,
        },
    }
    return output


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--header", type=pathlib.Path, required=True)
    parser.add_argument(
        "--generator",
        type=pathlib.Path,
        default=pathlib.Path("tools/rank4_jacek_hybrid_ranker_samples.cpp"),
    )
    parser.add_argument(
        "--hybrid-source",
        type=pathlib.Path,
        default=pathlib.Path(
            "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp"
        ),
    )
    parser.add_argument(
        "--teacher-source",
        type=pathlib.Path,
        default=pathlib.Path("submissions/codingame/bots/rank_4/submission.cpp"),
    )
    parser.add_argument(
        "--residual-model",
        type=pathlib.Path,
        default=pathlib.Path(
            "submissions/codingame/bots/rank_4/teacher_residual_model.json"
        ),
    )
    # Rank-4's non-mate value scale is 100,000, so 10,000 is a 0.10 margin.
    parser.add_argument("--minimum-margin", type=int, default=10_000)
    parser.add_argument("--minimum-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--fit-seed", type=int, action="append", default=[])
    parser.add_argument("--ridge", type=float, action="append", default=[])
    args = parser.parse_args(argv)
    if not args.fit_seed:
        args.fit_seed = [1701, 2909]
    if not args.ridge:
        args.ridge = [0.0001, 0.001, 0.01]
    if args.minimum_margin <= 0 or args.minimum_depth < 1 or args.epochs <= 0:
        parser.error("margin, depth, and epochs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        model = fit(args)
        output_text = json.dumps(model, indent=2, sort_keys=True) + "\n"
        header_text = render_header(model["model"]["int8_weights"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="ascii")
        args.header.write_text(header_text, encoding="ascii")
        print(
            f"wrote {args.output} and {args.header}; "
            f"header={len(header_text.encode('ascii'))} bytes"
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"ranker training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
