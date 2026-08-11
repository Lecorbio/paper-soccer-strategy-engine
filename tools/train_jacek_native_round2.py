#!/usr/bin/env python3
"""Train cumulative Jacek-native round-two value checkpoints.

The trainer emits every deterministic seed as an identified checkpoint.  Its
held-out outcome MSE choice is explicitly provisional; final seed selection is
performed by the native actual-clock match gate outside this trainer.
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
from typing import Mapping, Sequence

import numpy as np


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus_round2 as corpus_contract  # noqa: E402
import jacek_native_restart_corpus_round2 as restart_contract  # noqa: E402
import train_jacek_native as round1_trainer  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "jacek_native_round2_candidate.json"
MODEL_SCHEMA = round1_trainer.MODEL_SCHEMA
INPUT_COUNT = round1_trainer.INPUT_COUNT
HIDDEN_ONE = round1_trainer.HIDDEN_ONE
HIDDEN_TWO = round1_trainer.HIDDEN_TWO
QUANTIZATION_BITS = round1_trainer.QUANTIZATION_BITS
QUANTIZATION_MAX = round1_trainer.QUANTIZATION_MAX

initialize = round1_trainer.initialize
quantize = round1_trainer.quantize
forward = round1_trainer.forward
AdamW = round1_trainer.AdamW
train_batch = round1_trainer.train_batch
parse_seeds = round1_trainer.parse_seeds
tensor = round1_trainer.tensor
integer_tensor = round1_trainer.integer_tensor


@dataclasses.dataclass(frozen=True)
class Dataset:
    active: tuple[np.ndarray, ...]
    outcome: np.ndarray
    auxiliary: np.ndarray
    auxiliary_mask: np.ndarray
    exact_mask: np.ndarray
    game_keys: tuple[str, ...]
    turn: np.ndarray

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
        turn=np.asarray([sample.turn for sample in samples], dtype=np.int32),
    )


def checkpoint_provenance(
    games: Sequence[corpus_contract.NativeGame],
) -> dict:
    report = round1_trainer.checkpoint_provenance(games)
    seed_runtime = round1_trainer.UNTRAINED_SEED_RUNTIME
    seed_lines = seed_runtime.read_text(encoding="utf-8").splitlines()
    if len(seed_lines) != 7:
        raise ValueError("untrained seed runtime provenance is malformed")
    seed_identity = {
        "artifact_sha256": hashlib.sha256(seed_runtime.read_bytes()).hexdigest(),
        "model_sha256": seed_lines[3],
        "packed_sha256": seed_lines[4],
    }
    if (
        report["mode"] == "native-runtime-models/v1"
        and seed_identity in report["artifacts"]
    ):
        report["mode"] = "cumulative-native-runtime-models/v2"
    return report


def load_datasets(
    current_paths: Sequence[pathlib.Path],
    archived_round1_paths: Sequence[pathlib.Path] = (),
    restart_round2_paths: Sequence[pathlib.Path] = (),
):
    games, source_hashes, lineage = corpus_contract.load_games(
        current_paths, archived_round1_paths
    )
    lineage = {**lineage, "live_restart_round2": []}
    if restart_round2_paths:
        grouped: dict[pathlib.Path, list[pathlib.Path]] = {}
        for path in restart_round2_paths:
            resolved = path.resolve()
            grouped.setdefault(resolved.parent, []).append(resolved)
        game_keys = {game.key for game in games}
        for directory, paths in sorted(grouped.items()):
            restart_games, restart_sources, restart_lineage = (
                restart_contract.load_games(paths, verify_local_build=True)
            )
            duplicate_games = game_keys & {game.key for game in restart_games}
            if duplicate_games:
                raise ValueError(
                    "duplicate current/restart game key: "
                    + ", ".join(sorted(duplicate_games))
                )
            duplicate_sources = set(source_hashes) & set(restart_sources)
            if duplicate_sources:
                raise ValueError(
                    "duplicate current/restart corpus content: "
                    + ", ".join(sorted(duplicate_sources))
                )
            game_keys.update(game.key for game in restart_games)
            games.extend(restart_games)
            source_hashes.update(restart_sources)
            lineage["live_restart_round2"].append(restart_lineage)
        lineage["live_restart_round2"].sort(key=lambda item: (
            item["collector_tsv_sha256"], item["manifest_sha256"]
        ))
        games.sort(key=corpus_contract.game_sort_key)
        source_hashes = dict(sorted(source_hashes.items()))
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
    builds = corpus_contract.build_contracts(games)
    report = {
        "source_sha256": source_hashes,
        "corpus_sha256": hashlib.sha256(provenance_payload).hexdigest(),
        "corpus_validator_sha256": hashlib.sha256(
            pathlib.Path(corpus_contract.__file__).read_bytes()
        ).hexdigest(),
        "restart_corpus_validator_sha256": hashlib.sha256(
            pathlib.Path(restart_contract.__file__).read_bytes()
        ).hexdigest(),
        "lineage": lineage,
        "observed_move_policy_labels": 0,
        "games": len(games),
        "split_games": {
            split: sum(assignments[game.split_group] == split for game in games)
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
            "build_provenance_sha256": [item["sha256"] for item in builds],
            "build_contracts": builds,
            "model_artifact_sha256": sorted({
                artifact.artifact_sha256
                for game in games for artifact in game.model_artifacts
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


TURN_BINS = (
    ("turns_0_11", 0, 12),
    ("turns_12_23", 12, 24),
    ("turns_24_39", 24, 40),
    ("turns_40_plus", 40, None),
)


def _turn_calibration(
    predictions: np.ndarray, dataset: Dataset
) -> dict[str, dict]:
    result = {}
    for name, lower, upper in TURN_BINS:
        mask = dataset.turn >= lower
        if upper is not None:
            mask &= dataset.turn < upper
        count = int(np.count_nonzero(mask))
        if count == 0:
            result[name] = {
                "samples": 0,
                "outcome_mse": None,
                "outcome_sign_accuracy": None,
                "prediction_mean": None,
                "outcome_mean": None,
                "calibration_bias": None,
            }
            continue
        predicted = predictions[mask]
        outcome = dataset.outcome[mask]
        result[name] = {
            "samples": count,
            "outcome_mse": float(np.mean((predicted - outcome) ** 2)),
            "outcome_sign_accuracy": float(np.mean(
                (predicted >= 0.0) == (outcome >= 0.0)
            )),
            "prediction_mean": float(np.mean(predicted)),
            "outcome_mean": float(np.mean(outcome)),
            "calibration_bias": float(np.mean(predicted - outcome)),
        }
    return result


def metrics(
    parameters: Mapping[str, np.ndarray],
    dataset: Dataset,
    batch_size: int = 1024,
) -> dict:
    predictions = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        predictions[start:stop], _ = forward(
            parameters, dataset.active[start:stop]
        )
    primary = (predictions - dataset.outcome) ** 2
    auxiliary_mask = dataset.auxiliary_mask
    auxiliary_mse = (
        float(np.mean((predictions[auxiliary_mask]
                       - dataset.auxiliary[auxiliary_mask]) ** 2))
        if np.any(auxiliary_mask) else None
    )
    outcome_weight = np.where(
        dataset.exact_mask, 0.0, np.where(auxiliary_mask, 0.75, 1.0)
    )
    auxiliary_weight = np.where(
        dataset.exact_mask, 1.0, np.where(auxiliary_mask, 0.25, 0.0)
    )
    combined = (
        outcome_weight * primary
        + auxiliary_weight * (predictions - dataset.auxiliary) ** 2
    )
    return {
        "samples": len(dataset),
        "outcome_mse": float(np.mean(primary)),
        "outcome_mae": float(np.mean(np.abs(
            predictions - dataset.outcome
        ))),
        "outcome_sign_accuracy": float(np.mean(
            (predictions >= 0.0) == (dataset.outcome >= 0.0)
        )),
        "combined_target_mse": float(np.mean(combined)),
        "stable_reanalysis_samples": int(np.count_nonzero(auxiliary_mask)),
        "exact_reanalysis_samples": int(np.count_nonzero(dataset.exact_mask)),
        "stable_reanalysis_mse": auxiliary_mse,
        "prediction_mean": float(np.mean(predictions)),
        "turn_calibration": _turn_calibration(predictions, dataset),
    }


def quantized_w1_coverage(parameters: Mapping[str, np.ndarray]) -> dict:
    integer, _, _ = quantize(parameters)
    w1 = integer["w1"]

    def rows(begin: int, end: int) -> dict[str, int | float]:
        selected = w1[begin:end]
        nonzero_rows = int(np.count_nonzero(np.any(selected != 0, axis=1)))
        total = end - begin
        return {
            "rows": total,
            "nonzero_rows": nonzero_rows,
            "zero_rows": total - nonzero_rows,
            "coverage": nonzero_rows / total,
        }

    return {
        "all": rows(0, INPUT_COUNT),
        "used_edges": rows(0, corpus_contract.EDGE_COUNT),
        "turn_distances": rows(corpus_contract.EDGE_COUNT, INPUT_COUNT),
        "nonzero_weight_fraction": float(np.mean(w1 != 0)),
        "levels": {
            str(level): int(np.count_nonzero(w1 == level))
            for level in range(-QUANTIZATION_MAX, QUANTIZATION_MAX + 1)
        },
    }


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
    best_float = None
    best_epoch = 0
    best_float_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(datasets["train"]))
        losses = []
        for start in range(0, len(order), batch_size):
            losses.append(train_batch(
                master, optimizer, datasets["train"],
                order[start:start + batch_size], auxiliary_weight, False,
            ))
        validation = metrics(master, datasets["validation"])
        history.append({
            "epoch": epoch,
            "train_combined_mse": float(np.mean(losses)),
            "validation_outcome_mse": validation["outcome_mse"],
        })
        print(
            f"round2 seed {seed} epoch {epoch}: validation outcome MSE "
            f"{validation['outcome_mse']:.6f}", flush=True,
        )
        if validation["outcome_mse"] < best_float_loss - 1e-8:
            best_float_loss = validation["outcome_mse"]
            best_epoch = epoch
            best_float = {name: value.copy() for name, value in master.items()}
        elif epoch - best_epoch >= patience:
            break
    if best_float is None:
        raise RuntimeError("training did not produce a float checkpoint")

    float_best_metrics = {
        split: metrics(best_float, dataset)
        for split, dataset in datasets.items()
    }
    selected = {name: value.copy() for name, value in best_float.items()}
    _, _, selected_effective = quantize(selected)
    selected_loss = metrics(
        selected_effective, datasets["validation"]
    )["outcome_mse"]
    selected_qat_epoch = 0

    master = {name: value.copy() for name, value in best_float.items()}
    optimizer = AdamW(master, learning_rate * 0.25, weight_decay)
    for qat_epoch in range(1, qat_epochs + 1):
        order = rng.permutation(len(datasets["train"]))
        for start in range(0, len(order), batch_size):
            train_batch(
                master, optimizer, datasets["train"],
                order[start:start + batch_size], auxiliary_weight, True,
            )
        _, _, effective = quantize(master)
        validation_loss = metrics(
            effective, datasets["validation"]
        )["outcome_mse"]
        history.append({
            "qat_epoch": qat_epoch,
            "validation_quantized_outcome_mse": validation_loss,
        })
        # A tie deliberately retains the pre-QAT best.
        if validation_loss < selected_loss - 1e-8:
            selected_loss = validation_loss
            selected_qat_epoch = qat_epoch
            selected = {name: value.copy() for name, value in master.items()}

    _, _, effective = quantize(selected)
    report = {
        "seed": seed,
        "best_float_epoch": best_epoch,
        "qat_selection": {
            "selected": (
                "pre-qat-best" if selected_qat_epoch == 0
                else f"qat-epoch-{selected_qat_epoch}"
            ),
            "selected_qat_epoch": selected_qat_epoch,
            "pre_qat_retained": selected_qat_epoch == 0,
            "tie_break": "prefer-pre-qat-best",
        },
        "history": history,
        "float_best_metrics": float_best_metrics,
        "selected_float_metrics": {
            split: metrics(selected, dataset)
            for split, dataset in datasets.items()
        },
        "quantized_metrics": {
            split: metrics(effective, dataset)
            for split, dataset in datasets.items()
        },
        "quantized_w1_coverage": quantized_w1_coverage(selected),
    }
    return selected, report


def quantization_report(parameters: Mapping[str, np.ndarray]) -> dict:
    return round1_trainer.quantization_report(parameters)


def _checkpoint_payload(
    candidate: Mapping[str, np.ndarray], report: Mapping[str, object]
) -> dict:
    payload = {
        "seed": report["seed"],
        "model": {
            name: tensor(candidate[name]) for name in ("w1", "w2", "w3")
        },
        "quantization": quantization_report(candidate),
    }
    payload["checkpoint_sha256"] = hashlib.sha256(
        corpus_contract._canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def build_report(
    candidates: Sequence[Mapping[str, np.ndarray]],
    seed_reports: Sequence[dict],
    corpus_report: dict,
    arguments: argparse.Namespace,
) -> dict:
    ordering = sorted(
        range(len(seed_reports)),
        key=lambda index: (
            seed_reports[index]["quantized_metrics"]["validation"]["outcome_mse"],
            seed_reports[index]["seed"],
        ),
    )
    provisional_index = ordering[0]
    provisional = candidates[provisional_index]
    provisional_report = seed_reports[provisional_index]
    checkpoints = [
        _checkpoint_payload(candidate, report)
        for candidate, report in zip(candidates, seed_reports)
    ]
    trainer_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
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
            # The training artifact is immutable evidence, not a promotion
            # pointer.  Actual-clock selection is recorded in a separate,
            # content-addressed sidecar after every retained seed has passed
            # through the frozen screen and decisive gates.
            "chosen_seed": None,
            "provisional_seed": provisional_report["seed"],
            "selection": (
                "provisional-minimum-quantized-validation-outcome-mse-then-seed;"
                "final-native-actual-clock-strength-external"
            ),
            "external_actual_clock_selection": {
                "required": True,
                "status": "pending",
                "criterion": "native-actual-clock-match-strength",
                "eligible_seed_order": [seed_reports[index]["seed"]
                                        for index in ordering],
            },
        },
        "seed_reports": list(seed_reports),
        "model": {
            name: tensor(provisional[name]) for name in ("w1", "w2", "w3")
        },
        "quantization": quantization_report(provisional),
        "checkpoints": checkpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train cumulative ground-up Jacek-native round-two values."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path,
                        help="strict-current round-two shards")
    parser.add_argument("--archived-round1", nargs="*", type=pathlib.Path,
                        default=[])
    parser.add_argument(
        "--restart-round2", nargs="*", type=pathlib.Path, default=[],
        help=(
            "explicit provenance-safe live-restart shards; observed moves "
            "construct states only and never become labels"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds", type=parse_seeds,
        default=parse_seeds("20260821,20260822,20260823"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--qat-epochs", type=int, default=4)
    arguments = parser.parse_args()
    if (
        arguments.epochs <= 0 or arguments.patience <= 0
        or arguments.batch_size <= 0 or arguments.qat_epochs < 0
    ):
        parser.error("epochs, patience and batch size must be positive")
    finite = (
        arguments.learning_rate, arguments.weight_decay,
        arguments.auxiliary_weight,
    )
    if not all(math.isfinite(value) for value in finite):
        parser.error("floating-point hyperparameters must be finite")
    if arguments.learning_rate <= 0.0 or arguments.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    if arguments.auxiliary_weight != 0.25:
        parser.error("the native stable-reanalysis mixture is fixed at 0.25")

    datasets, corpus_report = load_datasets(
        arguments.corpus, arguments.archived_round1, arguments.restart_round2
    )
    candidates = []
    reports = []
    started = time.perf_counter()
    for seed in arguments.seeds:
        candidate, report = train_seed(
            datasets, seed, arguments.epochs, arguments.patience,
            arguments.batch_size, arguments.learning_rate,
            arguments.weight_decay, arguments.auxiliary_weight,
            arguments.qat_epochs,
        )
        candidates.append(candidate)
        reports.append(report)
    elapsed = time.perf_counter() - started
    examples = sum(
        len(datasets["train"]) * len(report["history"])
        for report in reports
    )
    model = build_report(candidates, reports, corpus_report, arguments)
    model["training"]["examples_processed"] = examples
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(
        model, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) + "\n")
    print(json.dumps({
        "output": str(arguments.output),
        "provisional_seed": model["training"]["provisional_seed"],
        "external_actual_clock_selection": "pending",
        "measured_examples_per_second": examples / elapsed,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
