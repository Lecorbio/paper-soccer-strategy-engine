#!/usr/bin/env python3
"""Train the value-only Jacek-native 1156 -> 32 -> 32 -> 1 network.

This trainer intentionally has no policy loss and accepts no incumbent-bot
labels.  The primary target is the final self-play result relative to the
player to move.  Stable native BFM reanalysis may contribute a separately
reported auxiliary loss with a default weight of 25 percent.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus as corpus_contract  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "jacek_native_bootstrap_model.json"
UNTRAINED_SEED_RUNTIME = ROOT / "models" / "jacek_native_untrained_seed.runtime"
MODEL_SCHEMA = "jacek_native_model/v1"
INPUT_COUNT = corpus_contract.INPUT_COUNT
HIDDEN_ONE = 32
HIDDEN_TWO = 32
QUANTIZATION_BITS = 3
QUANTIZATION_MAX = 3
LEAK = np.float32(0.01)


@dataclasses.dataclass(frozen=True)
class Dataset:
    active: tuple[np.ndarray, ...]
    outcome: np.ndarray
    auxiliary: np.ndarray
    auxiliary_mask: np.ndarray
    exact_mask: np.ndarray
    game_keys: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.active)


def dataset_from_samples(samples: Sequence[corpus_contract.NativeSample]) -> Dataset:
    return Dataset(
        active=tuple(np.asarray(sample.active, dtype=np.int32)
                     for sample in samples),
        outcome=np.asarray([sample.outcome for sample in samples], dtype=np.float32),
        auxiliary=np.asarray([
            sample.auxiliary_value if sample.auxiliary_value is not None else 0.0
            for sample in samples
        ], dtype=np.float32),
        auxiliary_mask=np.asarray([
            sample.auxiliary_value is not None for sample in samples
        ], dtype=bool),
        exact_mask=np.asarray([sample.exact for sample in samples], dtype=bool),
        game_keys=tuple(sample.game_key for sample in samples),
    )


def checkpoint_provenance(
    games: Sequence[corpus_contract.NativeGame],
) -> dict:
    artifacts = sorted({
        artifact
        for game in games
        for artifact in game.model_artifacts
    }, key=lambda artifact: (
        artifact.artifact_sha256,
        artifact.model_sha256,
        artifact.packed_sha256,
    ))
    seed_lines = UNTRAINED_SEED_RUNTIME.read_text(encoding="utf-8").splitlines()
    if len(seed_lines) != 7:
        raise ValueError("untrained seed runtime provenance is malformed")
    seed_artifact = corpus_contract.NativeModelArtifact(
        artifact_sha256=hashlib.sha256(
            UNTRAINED_SEED_RUNTIME.read_bytes()
        ).hexdigest(),
        model_sha256=seed_lines[3],
        packed_sha256=seed_lines[4],
    )
    mode = (
        "untrained-seed-bootstrap/v1"
        if artifacts == [seed_artifact]
        else "native-runtime-models/v1"
    )
    return {
        "mode": mode,
        "artifacts": [dataclasses.asdict(artifact) for artifact in artifacts],
    }


def load_datasets(paths: Sequence[pathlib.Path]):
    games, source_hashes = corpus_contract.load_games(
        paths, verify_local_build=True
    )
    split_samples, overlaps_removed, assignments = corpus_contract.prepare_splits(games)
    datasets = {
        split: dataset_from_samples(samples)
        for split, samples in split_samples.items()
    }
    empty = [split for split, dataset in datasets.items() if not dataset]
    if empty:
        raise ValueError(
            "whole-game split has no retained samples: " + ", ".join(empty)
        )
    provenance_payload = json.dumps(
        sorted(source_hashes.items()), separators=(",", ":")
    ).encode()
    model_provenance = checkpoint_provenance(games)
    build_provenance = corpus_contract.build_contracts(games)
    report = {
        "source_sha256": source_hashes,
        "corpus_sha256": hashlib.sha256(provenance_payload).hexdigest(),
        "corpus_validator_sha256": hashlib.sha256(
            pathlib.Path(corpus_contract.__file__).read_bytes()
        ).hexdigest(),
        "games": len(games),
        "split_games": {
            split: sum(
                assignments[game.split_group] == split for game in games
            )
            for split in ("train", "validation", "test")
        },
        "split_samples": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "cross_split_overlaps_removed": overlaps_removed,
        "augmentation": {
            "reflection": True,
            "rotation": "player-two-canonicalization-in-feature-encoder",
            "grouping": "whole-game-before-augmentation",
        },
        "generation": {
            "producer_sha256": sorted({game.producer_sha256 for game in games}),
            "build_provenance_sha256": [
                build["sha256"] for build in build_provenance
            ],
            "build_contracts": build_provenance,
            "model_artifact_sha256": sorted({
                artifact["artifact_sha256"]
                for artifact in model_provenance["artifacts"]
            }),
            "checkpoint_provenance": model_provenance,
            "search_stats": {
                field: sum(game.search_stats[field] for game in games)
                for field in games[0].search_stats
            },
            "opening_depths": dict(sorted(Counter(
                game.opening_depth for game in games
            ).items())),
            "temperature_turns": sorted({
                game.temperature_turns for game in games
            }),
        },
    }
    return datasets, report


def initialize(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.02, (INPUT_COUNT, HIDDEN_ONE)).astype(np.float32),
        "w2": rng.normal(0.0, math.sqrt(1.0 / HIDDEN_ONE),
                         (HIDDEN_ONE, HIDDEN_TWO)).astype(np.float32),
        "w3": rng.normal(0.0, math.sqrt(1.0 / HIDDEN_TWO),
                         (HIDDEN_TWO,)).astype(np.float32),
    }


def quantize_array(value: np.ndarray) -> tuple[np.ndarray, np.float32]:
    maximum = float(np.max(np.abs(value))) if value.size else 0.0
    scale = np.float32(maximum / QUANTIZATION_MAX if maximum > 0.0 else 1.0)
    quantized = np.clip(
        np.rint(value / scale), -QUANTIZATION_MAX, QUANTIZATION_MAX
    ).astype(np.int8)
    return quantized, scale


def quantize(parameters: Mapping[str, np.ndarray]):
    integer: dict[str, np.ndarray] = {}
    scales: dict[str, np.float32] = {}
    dequantized: dict[str, np.ndarray] = {}
    for name in ("w1", "w2", "w3"):
        integer[name], scales[name] = quantize_array(parameters[name])
        dequantized[name] = integer[name].astype(np.float32) * scales[name]
    return integer, scales, dequantized


def _hidden_one_activation(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value * value, LEAK * value)


def _hidden_one_derivative(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 2.0 * value, LEAK)


def _leaky_relu(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value, LEAK * value)


def _leaky_relu_derivative(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 1.0, LEAK).astype(np.float32)


def _fast_tanh(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -4.95, 4.95).astype(np.float32)
    square = clipped * clipped
    numerator = clipped * (
        135135.0 + square * (17325.0 + square * (378.0 + square))
    )
    denominator = (
        135135.0 + square * (62370.0 + square * (3150.0 + 28.0 * square))
    )
    result = numerator / denominator
    return np.where(value < -4.95, -1.0, np.where(value > 4.95, 1.0, result))


def _fast_tanh_derivative(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -4.95, 4.95).astype(np.float32)
    square = clipped * clipped
    numerator = clipped * (
        135135.0 + square * (17325.0 + square * (378.0 + square))
    )
    denominator = (
        135135.0 + square * (62370.0 + square * (3150.0 + 28.0 * square))
    )
    numerator_derivative = (
        135135.0 + square * (51975.0 + square * (1890.0 + 7.0 * square))
    )
    denominator_derivative = 2.0 * clipped * (
        62370.0 + square * (6300.0 + 84.0 * square)
    )
    derivative = (
        numerator_derivative * denominator - numerator * denominator_derivative
    ) / (denominator * denominator)
    return np.where(np.abs(value) > 4.95, 0.0, derivative).astype(np.float32)


def forward(
    parameters: Mapping[str, np.ndarray], active: Sequence[np.ndarray]
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    first_pre = np.empty((len(active), HIDDEN_ONE), dtype=np.float32)
    for row, indices in enumerate(active):
        first_pre[row] = parameters["w1"][indices].sum(axis=0)
    first = _hidden_one_activation(first_pre)
    second_pre = first @ parameters["w2"]
    second = _leaky_relu(second_pre)
    output_pre = second @ parameters["w3"]
    output = _fast_tanh(output_pre)
    return output, (first_pre, first, second_pre, second, output_pre)


def metrics(
    parameters: Mapping[str, np.ndarray], dataset: Dataset, batch_size: int = 1024
) -> dict:
    predictions = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        predictions[start:stop], _ = forward(parameters, dataset.active[start:stop])
    primary = (predictions - dataset.outcome) ** 2
    mask = dataset.auxiliary_mask
    auxiliary_mse = (
        float(np.mean((predictions[mask] - dataset.auxiliary[mask]) ** 2))
        if np.any(mask) else None
    )
    exact_mask = dataset.exact_mask
    outcome_weight = np.where(exact_mask, 0.0, np.where(mask, 0.75, 1.0))
    auxiliary_weight = np.where(exact_mask, 1.0, np.where(mask, 0.25, 0.0))
    combined = (
        outcome_weight * primary +
        auxiliary_weight * (predictions - dataset.auxiliary) ** 2
    )
    return {
        "samples": len(dataset),
        "outcome_mse": float(np.mean(primary)),
        "outcome_mae": float(np.mean(np.abs(predictions - dataset.outcome))),
        "outcome_sign_accuracy": float(np.mean(
            (predictions >= 0.0) == (dataset.outcome >= 0.0)
        )),
        "combined_target_mse": float(np.mean(combined)),
        "stable_reanalysis_samples": int(np.count_nonzero(mask)),
        "exact_reanalysis_samples": int(np.count_nonzero(exact_mask)),
        "stable_reanalysis_mse": auxiliary_mse,
        "prediction_mean": float(np.mean(predictions)),
    }


class AdamW:
    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.learning_rate = np.float32(learning_rate)
        self.weight_decay = np.float32(weight_decay)
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step = 0

    def update(
        self, parameters: Mapping[str, np.ndarray], gradients: Mapping[str, np.ndarray]
    ) -> None:
        self.step += 1
        correction_one = 1.0 - 0.9 ** self.step
        correction_two = 1.0 - 0.999 ** self.step
        for name, parameter in parameters.items():
            gradient = gradients[name]
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = (
                0.999 * self.second[name] + 0.001 * gradient * gradient
            )
            corrected_first = self.first[name] / correction_one
            corrected_second = self.second[name] / correction_two
            parameter *= 1.0 - self.learning_rate * self.weight_decay
            parameter -= (
                self.learning_rate * corrected_first /
                (np.sqrt(corrected_second) + 1e-8)
            )


def train_batch(
    master: Mapping[str, np.ndarray],
    optimizer: AdamW,
    dataset: Dataset,
    indices: np.ndarray,
    auxiliary_weight: float,
    quantization_aware: bool,
) -> float:
    active = tuple(dataset.active[int(index)] for index in indices)
    if quantization_aware:
        _, _, effective = quantize(master)
    else:
        effective = master
    prediction, cache = forward(effective, active)
    first_pre, first, second_pre, second, output_pre = cache
    outcome = dataset.outcome[indices]
    difference = prediction - outcome
    exact_mask = dataset.exact_mask[indices]
    auxiliary_mask = dataset.auxiliary_mask[indices]
    outcome_weights = np.where(
        exact_mask, 0.0, np.where(auxiliary_mask, 1.0 - auxiliary_weight, 1.0)
    ).astype(np.float32)
    auxiliary_weights = np.where(
        exact_mask, 1.0, np.where(auxiliary_mask, auxiliary_weight, 0.0)
    ).astype(np.float32)
    loss = float(np.mean(outcome_weights * difference * difference))
    output_gradient = (
        2.0 * outcome_weights * difference / max(len(indices), 1)
    )

    auxiliary_count = int(np.count_nonzero(auxiliary_mask))
    if auxiliary_count:
        auxiliary_difference = prediction - dataset.auxiliary[indices]
        loss += float(np.mean(
            auxiliary_weights * auxiliary_difference * auxiliary_difference
        ))
        output_gradient += (
            2.0 * auxiliary_weights * auxiliary_difference / max(len(indices), 1)
        )

    output_pre_gradient = output_gradient * _fast_tanh_derivative(output_pre)
    gradients: dict[str, np.ndarray] = {
        "w3": second.T @ output_pre_gradient,
    }
    second_gradient = output_pre_gradient[:, None] * effective["w3"][None, :]
    second_pre_gradient = second_gradient * _leaky_relu_derivative(second_pre)
    gradients["w2"] = first.T @ second_pre_gradient
    first_gradient = second_pre_gradient @ effective["w2"].T
    first_pre_gradient = first_gradient * _hidden_one_derivative(first_pre)
    gradients["w1"] = np.zeros_like(master["w1"])
    for row, active_indices in enumerate(active):
        np.add.at(gradients["w1"], active_indices, first_pre_gradient[row])

    squared_norm = sum(float(np.sum(value * value)) for value in gradients.values())
    norm = math.sqrt(squared_norm)
    if norm > 5.0:
        scale = np.float32(5.0 / norm)
        for gradient in gradients.values():
            gradient *= scale
    optimizer.update(master, gradients)
    return loss


def train_seed(
    datasets: Mapping[str, Dataset],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    auxiliary_weight: float,
    qat_epochs: int,
) -> tuple[dict[str, np.ndarray], dict]:
    master = initialize(seed)
    optimizer = AdamW(master, learning_rate, weight_decay)
    rng = np.random.default_rng(seed ^ 0xD1B54A32D192ED03)
    best = None
    best_epoch = 0
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(datasets["train"]))
        losses = []
        for start in range(0, len(order), batch_size):
            losses.append(train_batch(
                master,
                optimizer,
                datasets["train"],
                order[start:start + batch_size],
                auxiliary_weight,
                False,
            ))
        validation = metrics(master, datasets["validation"])
        history.append({
            "epoch": epoch,
            "train_combined_mse": float(np.mean(losses)),
            "validation_outcome_mse": validation["outcome_mse"],
        })
        print(
            f"seed {seed} epoch {epoch}: validation outcome MSE "
            f"{validation['outcome_mse']:.6f}",
            flush=True,
        )
        if validation["outcome_mse"] < best_loss - 1e-8:
            best_loss = validation["outcome_mse"]
            best_epoch = epoch
            best = {name: value.copy() for name, value in master.items()}
        elif epoch - best_epoch >= patience:
            break
    if best is None:
        raise RuntimeError("training did not produce a checkpoint")

    master = best
    optimizer = AdamW(master, learning_rate * 0.25, weight_decay)
    best_qat = {name: value.copy() for name, value in master.items()}
    _, _, quantized_effective = quantize(best_qat)
    best_quantized_loss = metrics(
        quantized_effective, datasets["validation"]
    )["outcome_mse"]
    for qat_epoch in range(1, qat_epochs + 1):
        order = rng.permutation(len(datasets["train"]))
        for start in range(0, len(order), batch_size):
            train_batch(
                master,
                optimizer,
                datasets["train"],
                order[start:start + batch_size],
                auxiliary_weight,
                True,
            )
        _, _, effective = quantize(master)
        validation_loss = metrics(
            effective, datasets["validation"]
        )["outcome_mse"]
        history.append({
            "qat_epoch": qat_epoch,
            "validation_quantized_outcome_mse": validation_loss,
        })
        if validation_loss < best_quantized_loss:
            best_quantized_loss = validation_loss
            best_qat = {name: value.copy() for name, value in master.items()}

    integer, scales, effective = quantize(best_qat)
    report = {
        "seed": seed,
        "best_float_epoch": best_epoch,
        "history": history,
        "float_metrics": {
            split: metrics(best_qat, dataset)
            for split, dataset in datasets.items()
        },
        "quantized_metrics": {
            split: metrics(effective, dataset)
            for split, dataset in datasets.items()
        },
        "scales": {name: float(value) for name, value in scales.items()},
    }
    return best_qat, report


def tensor(value: np.ndarray) -> dict:
    return {
        "shape": list(value.shape),
        "values": [float(item) for item in value.reshape(-1)],
    }


def integer_tensor(value: np.ndarray) -> dict:
    return {
        "shape": list(value.shape),
        "values": [int(item) for item in value.reshape(-1)],
    }


def parse_seeds(text: str) -> list[int]:
    result = []
    for item in text.split(","):
        try:
            seed = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
        if seed < 0 or seed >= 1 << 64:
            raise argparse.ArgumentTypeError("seeds must fit uint64")
        result.append(seed)
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return result


def quantization_report(parameters: Mapping[str, np.ndarray]) -> dict:
    integer, scales, _ = quantize(parameters)
    return {
        "bits": QUANTIZATION_BITS,
        "minimum": -QUANTIZATION_MAX,
        "maximum": QUANTIZATION_MAX,
        "scheme": "symmetric-per-layer-round-to-nearest",
        "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
        "scales": {name: float(scales[name]) for name in ("w1", "w2", "w3")},
        "weights": {
            name: integer_tensor(integer[name]) for name in ("w1", "w2", "w3")
        },
    }


def build_report(
    candidates: Sequence[Mapping[str, np.ndarray]],
    seed_reports: Sequence[dict],
    corpus_report: dict,
    arguments: argparse.Namespace,
) -> dict:
    trainer_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    chosen_report = min(
        seed_reports,
        key=lambda report: (
            report["quantized_metrics"]["validation"]["outcome_mse"],
            report["seed"],
        ),
    )
    chosen_index = next(
        index for index, report in enumerate(seed_reports)
        if report["seed"] == chosen_report["seed"]
    )
    chosen = candidates[chosen_index]
    return {
        "schema": MODEL_SCHEMA,
        "feature_schema": corpus_contract.FEATURE_SCHEMA,
        "architecture": {
            "inputs": INPUT_COUNT,
            "hidden_one": HIDDEN_ONE,
            "hidden_two": HIDDEN_TWO,
            "outputs": 1,
            "biases": False,
            "hidden_one_activation": "square-nonnegative-leaky-0.01-negative",
            "hidden_two_activation": "leaky-relu-0.01",
            "output_activation": "tanh",
        },
        "rules": corpus_contract.RULES,
        "target": {
            "primary": "mover-relative-final-outcome",
            "auxiliary": "stable-native-bfm-reanalysis",
            "auxiliary_weight": arguments.auxiliary_weight,
            "policy_target": None,
        },
        "provenance": {
            **corpus_report,
            "trainer_sha256": trainer_sha,
            "incumbent_labels": False,
            "protected_data": False,
        },
        "training": {
            "optimizer": "adamw",
            "batch_size": arguments.batch_size,
            "maximum_epochs": arguments.epochs,
            "patience": arguments.patience,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "qat_epochs": arguments.qat_epochs,
            "seeds": [report["seed"] for report in seed_reports],
            "chosen_seed": chosen_report["seed"],
            "selection": "minimum-quantized-validation-outcome-mse-then-seed",
        },
        "seed_reports": list(seed_reports),
        "model": {name: tensor(chosen[name]) for name in ("w1", "w2", "w3")},
        "quantization": quantization_report(chosen),
        "checkpoints": [
            {
                "seed": report["seed"],
                "model": {
                    name: tensor(candidate[name]) for name in ("w1", "w2", "w3")
                },
                "quantization": quantization_report(candidate),
            }
            for candidate, report in zip(candidates, seed_reports)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a ground-up Jacek-native complete-turn value model."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds", type=parse_seeds,
        default=parse_seeds("20260811,20260812,20260813"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--qat-epochs", type=int, default=4)
    arguments = parser.parse_args()
    if (arguments.epochs <= 0 or arguments.patience <= 0 or
            arguments.batch_size <= 0 or arguments.qat_epochs < 0):
        parser.error("epochs, patience and batch size must be positive")
    finite_values = (
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.auxiliary_weight,
    )
    if not all(math.isfinite(value) for value in finite_values):
        parser.error("floating-point hyperparameters must be finite")
    if arguments.learning_rate <= 0.0 or arguments.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    if arguments.auxiliary_weight != 0.25:
        parser.error("the native stable-reanalysis mixture is fixed at 0.25")

    datasets, corpus_report = load_datasets(arguments.corpus)
    candidates = []
    reports = []
    training_started = time.perf_counter()
    for seed in arguments.seeds:
        candidate, report = train_seed(
            datasets,
            seed,
            arguments.epochs,
            arguments.patience,
            arguments.batch_size,
            arguments.learning_rate,
            arguments.weight_decay,
            arguments.auxiliary_weight,
            arguments.qat_epochs,
        )
        candidates.append(candidate)
        reports.append(report)
    training_elapsed = time.perf_counter() - training_started
    training_examples = sum(
        len(datasets["train"]) * len(report["history"])
        for report in reports
    )
    chosen_index = min(
        range(len(reports)),
        key=lambda index: (
            reports[index]["quantized_metrics"]["validation"]["outcome_mse"],
            reports[index]["seed"],
        ),
    )
    model = build_report(candidates, reports, corpus_report, arguments)
    model["training"]["examples_processed"] = training_examples
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(
        model, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) + "\n")
    print(json.dumps({
        "output": str(arguments.output),
        "chosen_seed": reports[chosen_index]["seed"],
        "measured_examples_per_second": training_examples / training_elapsed,
        "quantized_metrics": reports[chosen_index]["quantized_metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
