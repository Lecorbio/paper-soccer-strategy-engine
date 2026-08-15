#!/usr/bin/env python3
"""Train the fresh bias-free scalar evaluator and emit its packed C++ header.

The trainer accepts only rows validated by ``fresh_corpus.py``.  It always
starts from a named random seed, has no checkpoint/resume input, splits by
whole game, and supports terminal-value plus successor-ranking supervision.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable

import numpy as np

try:
    from .fresh_corpus import (
        FEATURE_COUNT,
        FreshCorpusValidator,
        iter_jsonl,
        load_arena_game_bindings,
        load_contract,
        load_excluded_game_ids,
        load_producer_source_hashes,
    )
    from .immutable_artifacts import canonical_json_bytes, sha256_file, write_immutable
except ImportError:
    from fresh_corpus import (
        FEATURE_COUNT,
        FreshCorpusValidator,
        iter_jsonl,
        load_arena_game_bindings,
        load_contract,
        load_excluded_game_ids,
        load_producer_source_hashes,
    )
    from immutable_artifacts import canonical_json_bytes, sha256_file, write_immutable


TRAINER_SCHEMA = "papersoccer.jacek-arena-bfm.model.v1"


def game_bucket(game_id: str) -> int:
    return hashlib.sha256(game_id.encode("utf-8")).digest()[0] % 10


@dataclasses.dataclass
class Data:
    value_x: np.ndarray
    value_y: np.ndarray
    value_w: np.ndarray
    value_split: np.ndarray
    value_arena: np.ndarray
    pair_preferred: np.ndarray
    pair_inferior: np.ndarray
    pair_w: np.ndarray
    pair_split: np.ndarray
    pair_arena: np.ndarray
    games: int
    source_counts: dict[str, int]
    corpus_inputs: list[dict[str, Any]]
    # Defaults preserve the small in-memory construction interface used by
    # metric/runtime tests.  Real training always fills these fields through
    # load_data(), where the strict campaign gates are mandatory.
    scratch_games: int = 0
    scratch_games_by_opening_depth: dict[int, int] = dataclasses.field(default_factory=dict)
    window_plan_sha256: str = ""
    exclusion_registry_sha256: str = ""
    producer_source_sha256: list[str] = dataclasses.field(default_factory=list)
    arena_derivation_sha256: list[str] = dataclasses.field(default_factory=list)


def load_data(
    paths: Iterable[pathlib.Path],
    *,
    window_plan: pathlib.Path,
    exclusions: pathlib.Path,
    producer_sources: Iterable[pathlib.Path],
    arena_derivations: Iterable[pathlib.Path],
    repository: pathlib.Path,
    campaign_root: pathlib.Path,
    minimum_scratch_games: int = 2_000,
) -> Data:
    root = campaign_root.resolve()
    producer_sources = list(producer_sources)
    arena_derivations = list(arena_derivations)
    producer_hashes = load_producer_source_hashes(
        producer_sources,
        campaign_root=root,
    )
    arena_bindings = load_arena_game_bindings(
        arena_derivations,
        campaign_root=root,
        repository=repository,
    )
    validator = FreshCorpusValidator(
        load_contract(window_plan),
        excluded_game_ids=load_excluded_game_ids(exclusions),
        approved_producer_source_sha256=producer_hashes,
        arena_game_bindings=arena_bindings,
        training_only=True,
    )
    values: list[tuple[Any, ...]] = []
    pairs: list[tuple[Any, ...]] = []
    games: set[str] = set()
    scratch_games: set[str] = set()
    scratch_depth_by_game: dict[str, int] = {}
    source_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    inputs: list[dict[str, Any]] = []
    pair_decisions: Counter[tuple[str, str]] = Counter()
    pair_games: Counter[str] = Counter()
    pair_indices: set[tuple[str, str, int]] = set()
    pair_signatures: set[tuple[str, str, str, str]] = set()
    game_provenance: dict[str, tuple[Any, ...]] = {}
    counterfactual_pairs: dict[str, dict[int, str]] = {}
    arena_position_labels: dict[tuple[str, str], set[str]] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"corpus is outside campaign root: {path}") from error
        if path.is_symlink():
            raise ValueError(f"corpus may not be a symlink: {path}")
        input_digest = hashlib.sha256()
        for raw in iter_jsonl(path, digest=input_digest):
            item = validator.validate_row(raw)
            if item.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {item.sample_id}")
            sample_ids.add(item.sample_id)
            games.add(item.game_id)
            source_counts[item.source_kind] += 1
            split = game_bucket(item.split_game_id)
            if item.is_arena:
                provenance = (
                    item.raw.get("arena_game_id"),
                    item.raw.get("arena_derivation_sha256"),
                    item.raw.get("arena_record_sha256"),
                    item.raw.get("raw_sha256"),
                    item.raw.get("normalized_sha256"),
                    item.raw.get("submitted_source_sha256"),
                    item.raw.get("agent_id"),
                    item.raw.get("submission_id"),
                    item.raw.get("window_id"),
                    item.raw.get("window_role"),
                )
            else:
                provenance = (
                    item.raw.get("evidence_sha256"),
                    item.raw.get("producer_source_sha256"),
                    item.raw.get("opening_depth"),
                    item.raw.get("initialization"),
                    tuple(item.raw.get("checkpoint_inputs") or ()),
                )
                scratch_games.add(item.game_id)
                depth = int(item.raw["opening_depth"])
                previous_depth = scratch_depth_by_game.setdefault(item.game_id, depth)
                if previous_depth != depth:
                    raise ValueError(f"scratch game {item.game_id!r} has contradictory opening depths")
            previous_provenance = game_provenance.setdefault(item.game_id, provenance)
            if previous_provenance != provenance:
                raise ValueError(f"game {item.game_id!r} has contradictory provenance")
            if item.kind == "value":
                values.append(
                    (
                        np.asarray(item.features, dtype=np.uint8),
                        float(item.target),
                        float(item.weight),
                        split,
                        bool(item.is_arena),
                    )
                )
            else:
                decision = (item.game_id, str(item.raw["decision_id"]))
                pair_decisions[decision] += 1
                pair_games[item.game_id] += 1
                if pair_decisions[decision] > validator.max_pairs_per_decision:
                    raise ValueError(f"too many pairs for decision {decision}")
                if pair_games[item.game_id] > validator.max_pairs_per_game:
                    raise ValueError(f"too many pairs for game {item.game_id}")
                index_key = (*decision, int(item.raw["pair_index"]))
                if index_key in pair_indices:
                    raise ValueError(f"duplicate pair_index: {index_key}")
                pair_indices.add(index_key)
                signature = (
                    item.game_id,
                    decision[1],
                    str(item.raw["observed_complete_action"]),
                    str(item.raw["inferior_complete_action"]),
                )
                if signature in pair_signatures:
                    raise ValueError(f"duplicate pairwise alternative: {signature}")
                pair_signatures.add(signature)
                pairs.append(
                    (
                        np.asarray(item.preferred_features, dtype=np.uint8),
                        np.asarray(item.inferior_features, dtype=np.uint8),
                        float(item.weight),
                        split,
                        bool(item.is_arena),
                    )
                )
            if item.kind == "value" and item.is_arena:
                position = (str(item.raw["arena_game_id"]), str(item.raw["position_id"]))
                arena_position_labels.setdefault(position, set()).add(str(item.raw["label_method"]))
                if item.source_kind == "arena_counterfactual":
                    pair_id = str(item.raw["counterfactual_pair_id"])
                    variant = int(item.raw["color_swap_variant"])
                    variants = counterfactual_pairs.setdefault(pair_id, {})
                    previous_game = variants.setdefault(variant, item.game_id)
                    if previous_game != item.game_id:
                        raise ValueError(
                            f"counterfactual pair {pair_id!r} repeats color variant {variant}"
                        )
        inputs.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": input_digest.hexdigest(),
            }
        )
    if minimum_scratch_games < 0:
        raise ValueError("minimum_scratch_games cannot be negative")
    if len(scratch_games) < minimum_scratch_games:
        raise ValueError(
            f"training corpus contains {len(scratch_games)} scratch games; "
            f"at least {minimum_scratch_games} are required"
        )
    observed_depths = set(scratch_depth_by_game.values())
    if scratch_games and observed_depths != {0, 4, 8, 12}:
        raise ValueError("scratch corpus must cover procedural opening depths 0, 4, 8, and 12")
    for pair_id, variants in counterfactual_pairs.items():
        if set(variants) != {0, 1}:
            raise ValueError(
                f"counterfactual pair {pair_id!r} must contain both color-swapped continuations"
            )
    for position, methods in arena_position_labels.items():
        if "exact" in methods and len(methods) > 1:
            raise ValueError(
                f"exact solved value for arena position {position!r} must replace lower-confidence labels"
            )
    if not values:
        raise ValueError("training corpus contains no value rows")

    value_x = np.stack([row[0] for row in values])
    value_y = np.asarray([row[1] for row in values], dtype=np.float32)
    value_w = np.asarray([row[2] for row in values], dtype=np.float32)
    value_split = np.asarray([row[3] for row in values], dtype=np.uint8)
    value_arena = np.asarray([row[4] for row in values], dtype=bool)
    if pairs:
        pair_preferred = np.stack([row[0] for row in pairs])
        pair_inferior = np.stack([row[1] for row in pairs])
        pair_w = np.asarray([row[2] for row in pairs], dtype=np.float32)
        pair_split = np.asarray([row[3] for row in pairs], dtype=np.uint8)
        pair_arena = np.asarray([row[4] for row in pairs], dtype=bool)
    else:
        pair_preferred = np.empty((0, FEATURE_COUNT), dtype=np.uint8)
        pair_inferior = np.empty((0, FEATURE_COUNT), dtype=np.uint8)
        pair_w = np.empty(0, dtype=np.float32)
        pair_split = np.empty(0, dtype=np.uint8)
        pair_arena = np.empty(0, dtype=bool)
    return Data(
        value_x=value_x,
        value_y=value_y,
        value_w=value_w,
        value_split=value_split,
        value_arena=value_arena,
        pair_preferred=pair_preferred,
        pair_inferior=pair_inferior,
        pair_w=pair_w,
        pair_split=pair_split,
        pair_arena=pair_arena,
        games=len(games),
        scratch_games=len(scratch_games),
        scratch_games_by_opening_depth=dict(sorted(Counter(scratch_depth_by_game.values()).items())),
        source_counts=dict(sorted(source_counts.items())),
        corpus_inputs=inputs,
        window_plan_sha256=sha256_file(window_plan),
        exclusion_registry_sha256=sha256_file(exclusions),
        producer_source_sha256=sorted(producer_hashes),
        arena_derivation_sha256=sorted(sha256_file(path) for path in arena_derivations),
    )


def scale_arena_weights(weights: np.ndarray, arena: np.ndarray, exposure: float) -> np.ndarray:
    result = weights.astype(np.float32, copy=True)
    arena_total = float(result[arena].sum())
    scratch_total = float(result[~arena].sum())
    if arena_total == 0.0:
        if exposure != 0.0:
            raise ValueError("nonzero arena exposure requested without arena rows")
        return result
    if scratch_total == 0.0:
        if exposure != 1.0:
            raise ValueError("arena-only data requires arena exposure 1")
        return result
    if exposure <= 0.0:
        result[arena] = 0.0
    elif exposure >= 1.0:
        result[~arena] = 0.0
    else:
        result[arena] *= (exposure / (1.0 - exposure)) * (scratch_total / arena_total)
    return result


def effective_training_weights(
    data: Data,
    exposure: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale value and pair rows together to one effective arena exposure."""

    value_result = data.value_w.astype(np.float32, copy=True)
    pair_result = data.pair_w.astype(np.float32, copy=True)
    value_train = np.flatnonzero(data.value_split >= 2)
    pair_train = np.flatnonzero(data.pair_split >= 2)
    weights = np.concatenate((value_result[value_train], pair_result[pair_train]))
    arena = np.concatenate((data.value_arena[value_train], data.pair_arena[pair_train]))
    if not len(weights):
        raise ValueError("training split contains no rows")
    scaled = scale_arena_weights(weights, arena, exposure)
    value_result[value_train] = scaled[: len(value_train)]
    pair_result[pair_train] = scaled[len(value_train) :]
    positive = scaled > 0
    if not np.any(positive):
        raise ValueError("arena exposure removed every training row")
    total = float(scaled[positive].sum())
    realized = float(scaled[arena].sum() / total) if total > 0.0 else 0.0
    return value_result, pair_result, realized


@dataclasses.dataclass
class Network:
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray

    @classmethod
    def random(cls, hidden1: int, seed: int) -> "Network":
        rng = np.random.default_rng(seed)
        def xavier(out_size: int, in_size: int) -> np.ndarray:
            bound = math.sqrt(6.0 / (in_size + out_size))
            return rng.uniform(-bound, bound, (out_size, in_size)).astype(np.float32)
        return cls(xavier(hidden1, FEATURE_COUNT), xavier(32, hidden1), xavier(1, 32)[0])

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = np.tanh(x @ self.w1.T)
        h2 = np.tanh(h1 @ self.w2.T)
        out = np.tanh(h2 @ self.w3)
        return h1, h2, out

    def backward(
        self, x: np.ndarray, h1: np.ndarray, h2: np.ndarray, out: np.ndarray,
        d_value: np.ndarray,
    ) -> list[np.ndarray]:
        dz3 = d_value * (1.0 - out * out)
        g3 = dz3 @ h2
        dz2 = (dz3[:, None] * self.w3[None, :]) * (1.0 - h2 * h2)
        g2 = dz2.T @ h1
        dz1 = (dz2 @ self.w2) * (1.0 - h1 * h1)
        g1 = dz1.T @ x
        return [g1, g2, g3]


class Adam:
    def __init__(self, network: Network, learning_rate: float) -> None:
        self.parameters = [network.w1, network.w2, network.w3]
        self.m = [np.zeros_like(value) for value in self.parameters]
        self.v = [np.zeros_like(value) for value in self.parameters]
        self.learning_rate = learning_rate
        self.step_number = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        self.step_number += 1
        for parameter, moment, variance, gradient in zip(
            self.parameters, self.m, self.v, gradients, strict=True
        ):
            np.multiply(moment, 0.9, out=moment)
            moment += 0.1 * gradient
            np.multiply(variance, 0.999, out=variance)
            variance += 0.001 * gradient * gradient
            corrected_m = moment / (1.0 - 0.9 ** self.step_number)
            corrected_v = variance / (1.0 - 0.999 ** self.step_number)
            parameter -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)


def value_batch(
    network: Network,
    x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    normalization: float | None = None,
) -> tuple[float, list[np.ndarray]]:
    dense = x.astype(np.float32, copy=False)
    h1, h2, out = network.forward(dense)
    total = max(float(weight.sum()) if normalization is None else normalization, 1e-12)
    error = out - target
    loss = float((weight * error * error).sum() / total)
    d_value = 2.0 * weight * error / total
    return loss, network.backward(dense, h1, h2, out, d_value)


def pair_batch(
    network: Network,
    preferred: np.ndarray,
    inferior: np.ndarray,
    weight: np.ndarray,
    normalization: float | None = None,
) -> tuple[float, list[np.ndarray]]:
    p = preferred.astype(np.float32, copy=False)
    i = inferior.astype(np.float32, copy=False)
    p1, p2, pv = network.forward(p)
    i1, i2, iv = network.forward(i)
    delta = np.clip(pv - iv, -30.0, 30.0)
    total = max(float(weight.sum()) if normalization is None else normalization, 1e-12)
    # Each feature vector is mover-relative *after* the complete action, so it
    # is evaluated for the opponent.  A preferred actor action therefore has
    # the lower successor value.  Minimize softplus(preferred - inferior).
    loss = float((weight * np.logaddexp(0.0, delta)).sum() / total)
    derivative = 1.0 / (1.0 + np.exp(-delta))
    dp = weight * derivative / total
    di = -dp
    gp = network.backward(p, p1, p2, pv, dp)
    gi = network.backward(i, i1, i2, iv, di)
    return loss, [left + right for left, right in zip(gp, gi, strict=True)]


def metrics(network: Network, data: Data, split: int) -> dict[str, float | int | None]:
    value_indices = np.flatnonzero(data.value_split == split)
    predictions: list[np.ndarray] = []
    for start in range(0, len(value_indices), 2048):
        x = data.value_x[value_indices[start:start + 2048]].astype(np.float32)
        predictions.append(network.forward(x)[2])
    pred = np.concatenate(predictions) if predictions else np.empty(0)
    target = data.value_y[value_indices]
    pair_indices = np.flatnonzero(data.pair_split == split)
    pair_correct = 0
    for start in range(0, len(pair_indices), 2048):
        idx = pair_indices[start:start + 2048]
        pv = network.forward(data.pair_preferred[idx].astype(np.float32))[2]
        iv = network.forward(data.pair_inferior[idx].astype(np.float32))[2]
        pair_correct += int(np.count_nonzero(pv < iv))
    return {
        "value_rows": int(len(value_indices)),
        "value_mse": float(np.mean((pred - target) ** 2)) if len(target) else None,
        "value_sign_accuracy": float(np.mean((pred >= 0) == (target >= 0))) if len(target) else None,
        "pair_rows": int(len(pair_indices)),
        "pair_accuracy": float(pair_correct / len(pair_indices)) if len(pair_indices) else None,
    }


def train(
    data: Data, hidden1: int, seed: int, epochs: int, batch_size: int,
    learning_rate: float, arena_exposure: float,
) -> tuple[Network, list[dict[str, Any]]]:
    network = Network.random(hidden1, seed)
    optimizer = Adam(network, learning_rate)
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    value_weights, pair_weights, realized_exposure = effective_training_weights(
        data, arena_exposure
    )
    value_train = np.flatnonzero((data.value_split >= 2) & (value_weights > 0))
    pair_train = np.flatnonzero((data.pair_split >= 2) & (pair_weights > 0))
    positive_weights = np.concatenate((value_weights[value_train], pair_weights[pair_train]))
    batch_normalization = max(
        float(positive_weights.sum()) / len(positive_weights) * batch_size,
        1e-12,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        rng.shuffle(value_train)
        rng.shuffle(pair_train)
        losses: list[float] = []
        batches: list[tuple[str, np.ndarray]] = []
        for start in range(0, len(value_train), batch_size):
            batches.append(("value", value_train[start:start + batch_size]))
        for start in range(0, len(pair_train), batch_size):
            batches.append(("pairwise", pair_train[start:start + batch_size]))
        rng.shuffle(batches)
        for kind, index in batches:
            if kind == "value":
                loss, gradients = value_batch(
                    network,
                    data.value_x[index],
                    data.value_y[index],
                    value_weights[index],
                    batch_normalization,
                )
            else:
                loss, gradients = pair_batch(
                    network,
                    data.pair_preferred[index],
                    data.pair_inferior[index],
                    pair_weights[index],
                    batch_normalization,
                )
            optimizer.step(gradients)
            losses.append(loss)
        history.append(
            {
                "effective_arena_exposure": realized_exposure,
                "epoch": epoch + 1,
                "mean_batch_loss": float(np.mean(losses)),
                "validation": metrics(network, data, 0),
            }
        )
        print(json.dumps(history[-1], sort_keys=True), file=sys.stderr, flush=True)
    return network, history


def quantize(value: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.max(np.abs(value)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    packed = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
    return packed, scale


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()


def c_string(name: str, encoded: str) -> str:
    chunks = [encoded[index:index + 96] for index in range(0, len(encoded), 96)]
    return f"inline constexpr char {name}[] =\n" + "\n".join(f'    "{chunk}"' for chunk in chunks) + ";\n"


def emit_header(path: pathlib.Path, network: Network, seed: int, identity: str) -> dict[str, Any]:
    q1, s1 = quantize(network.w1)
    q2, s2 = quantize(network.w2)
    q3, s3 = quantize(network.w3)
    b1 = base64.b64encode(q1.tobytes()).decode("ascii")
    b2 = base64.b64encode(q2.tobytes()).decode("ascii")
    b3 = base64.b64encode(q3.tobytes()).decode("ascii")
    content = (
        "#pragma once\n\n#include <cstddef>\n\n"
        "namespace jacek_arena_bfm::model {\n"
        "inline constexpr int kInputSize = 1156;\n"
        f"inline constexpr int kHidden1Size = {network.w1.shape[0]};\n"
        "inline constexpr int kHidden2Size = 32;\n"
        f"inline constexpr std::size_t kW1Count = {q1.size};\n"
        f"inline constexpr std::size_t kW2Count = {q2.size};\n"
        f"inline constexpr std::size_t kW3Count = {q3.size};\n"
        f"inline constexpr float kW1Scale = {s1:.10g}F;\n"
        f"inline constexpr float kW2Scale = {s2:.10g}F;\n"
        f"inline constexpr float kW3Scale = {s3:.10g}F;\n"
        + c_string("kW1Packed", b1)
        + c_string("kW2Packed", b2)
        + c_string("kW3Packed", b3)
        + f"inline constexpr unsigned long long kBootstrapSeed = {seed}ULL;\n"
        + f'inline constexpr char kIdentity[] = "{identity}";\n'
        + "}  // namespace jacek_arena_bfm::model\n"
    ).encode("ascii")
    write_immutable(path, content)
    return {
        "header_sha256": hashlib.sha256(content).hexdigest(),
        "quantized_sha256": {
            "w1": hashlib.sha256(q1.tobytes()).hexdigest(),
            "w2": hashlib.sha256(q2.tobytes()).hexdigest(),
            "w3": hashlib.sha256(q3.tobytes()).hexdigest(),
        },
        "scales": [s1, s2, s3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--window-plan", type=pathlib.Path, required=True)
    parser.add_argument("--exclusions", type=pathlib.Path, required=True)
    parser.add_argument("--producer-source", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--arena-derivation", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--campaign-root", type=pathlib.Path, required=True)
    parser.add_argument("--hidden1", type=int, choices=(32, 48, 64), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--arena-exposure", type=float, choices=(0.0, 0.25, 0.40, 0.55), default=0.0)
    parser.add_argument("--minimum-scratch-games", type=int, default=2_000)
    parser.add_argument("--output-header", type=pathlib.Path, required=True)
    parser.add_argument("--output-model", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("epochs, batch size, and learning rate must be positive")
    if not 0 <= args.seed <= (1 << 64) - 1:
        parser.error("seed must fit an unsigned 64-bit integer")
    if args.minimum_scratch_games < 2_000:
        parser.error("campaign training requires at least 2,000 scratch games")
    data = load_data(
        args.corpus,
        window_plan=args.window_plan,
        exclusions=args.exclusions,
        producer_sources=args.producer_source,
        arena_derivations=args.arena_derivation,
        repository=args.repository,
        campaign_root=args.campaign_root,
        minimum_scratch_games=args.minimum_scratch_games,
    )
    network, history = train(
        data, args.hidden1, args.seed, args.epochs, args.batch_size,
        args.learning_rate, args.arena_exposure,
    )
    weight_sha = {
        "w1": digest_array(network.w1),
        "w2": digest_array(network.w2),
        "w3": digest_array(network.w3),
    }
    model_identity_hash = hashlib.sha256(
        canonical_json_bytes({"shape": [FEATURE_COUNT, args.hidden1, 32, 1], "seed": args.seed, "weights": weight_sha})
    ).hexdigest()
    identity = f"fresh-{args.hidden1}x32-s{args.seed}-{model_identity_hash[:12]}"
    header = emit_header(args.output_header, network, args.seed, identity)
    model_stream = io.BytesIO()
    np.savez_compressed(model_stream, w1=network.w1, w2=network.w2, w3=network.w3)
    write_immutable(args.output_model, model_stream.getvalue())
    manifest = {
        "schema": TRAINER_SCHEMA,
        "namespace": "jacek_arena_bfm",
        "identity": identity,
        "initialization": "random",
        "checkpoint_inputs": [],
        "seed": args.seed,
        "shape": [FEATURE_COUNT, args.hidden1, 32, 1],
        "bias_tensors": [],
        "activation": "tanh",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "arena_exposure": args.arena_exposure,
        "effective_arena_exposure": history[-1]["effective_arena_exposure"],
        "games": data.games,
        "scratch_games": data.scratch_games,
        "scratch_games_by_opening_depth": data.scratch_games_by_opening_depth,
        "value_rows": int(len(data.value_y)),
        "pair_rows": int(len(data.pair_w)),
        "source_counts": data.source_counts,
        "corpus_inputs": data.corpus_inputs,
        "window_plan_sha256": data.window_plan_sha256,
        "exclusion_registry_sha256": data.exclusion_registry_sha256,
        "producer_source_sha256": data.producer_source_sha256,
        "arena_derivation_sha256": data.arena_derivation_sha256,
        "trainer_source_sha256": sha256_file(pathlib.Path(__file__)),
        "whole_game_split": (
            "sha256(provenance_parent_game_id)[0] % 10: "
            "0 validation, 1 scratch-test, 2-9 train"
        ),
        "history": history,
        "validation": metrics(network, data, 0),
        "scratch_test": metrics(network, data, 1),
        "float_weight_sha256": weight_sha,
        "npz_sha256": sha256_file(args.output_model),
        **header,
    }
    payload = canonical_json_bytes(manifest)
    write_immutable(args.output_manifest, payload)
    print(json.dumps({"manifest": str(args.output_manifest), "identity": identity, "validation": manifest["validation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
