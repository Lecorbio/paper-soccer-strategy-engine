#!/usr/bin/env python3
"""Train cumulative Jacek-native round-two value checkpoints.

The trainer emits every deterministic seed as an identified checkpoint.  Its
held-out combined-target MSE choice is explicitly provisional; final seed
selection is performed by the native actual-clock match gate outside this
trainer. Robust validation-selected layer scales stay fixed through QAT;
emitted tensors are the exact exporter-idempotent dequantized 3-bit values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
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
parse_seeds = round1_trainer.parse_seeds
tensor = round1_trainer.tensor
integer_tensor = round1_trainer.integer_tensor

PHASE_WEIGHTS = (
    ("turns_0_11", 0, 12, np.float32(3.0)),
    ("turns_12_23", 12, 24, np.float32(1.5)),
    ("turns_24_plus", 24, None, np.float32(1.0)),
)
PHASE_WEIGHT_APPLICATION = "after-exact-override-combined-target-loss/v1"
AUXILIARY_WEIGHT = 0.25
VALIDATION_SELECTION_METRIC = (
    "validation-phase-weighted-combined-target-mse"
)


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


def dataset_from_samples(
    samples: Sequence[corpus_contract.NativeSample], *, release: bool = False
) -> Dataset:
    """Materialize sparse samples without retaining two full representations.

    Feature indices fit in uint16.  During a real corpus load ``samples`` is a
    private list and ``release`` drops each Python-heavy NativeSample as soon
    as its packed NumPy representation exists.  Tests and other callers keep
    the non-mutating default.
    """
    if INPUT_COUNT > np.iinfo(np.uint16).max:
        raise RuntimeError("native feature indices no longer fit uint16")
    active: list[np.ndarray] = []
    outcome: list[float] = []
    auxiliary: list[float] = []
    auxiliary_mask: list[bool] = []
    exact_mask: list[bool] = []
    game_keys: list[str] = []
    turn: list[int] = []
    mutable = samples if release and isinstance(samples, list) else None
    for index, sample in enumerate(samples):
        active.append(np.asarray(sample.active, dtype=np.uint16))
        outcome.append(sample.outcome)
        auxiliary.append(
            sample.auxiliary_value
            if sample.auxiliary_value is not None else 0.0
        )
        auxiliary_mask.append(sample.auxiliary_value is not None)
        exact_mask.append(sample.exact)
        game_keys.append(sample.game_key)
        turn.append(sample.turn)
        if mutable is not None:
            mutable[index] = None
    return Dataset(
        active=tuple(active),
        outcome=np.asarray(outcome, dtype=np.float32),
        auxiliary=np.asarray(auxiliary, dtype=np.float32),
        auxiliary_mask=np.asarray(auxiliary_mask, dtype=bool),
        exact_mask=np.asarray(exact_mask, dtype=bool),
        game_keys=tuple(game_keys),
        turn=np.asarray(turn, dtype=np.int32),
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
    archived_round2_paths: Sequence[pathlib.Path] = (),
    archived_restart_round2_paths: Sequence[pathlib.Path] = (),
):
    games, source_hashes, lineage = corpus_contract.load_games(
        current_paths, archived_round1_paths, archived_round2_paths
    )
    lineage = {
        **lineage,
        "live_restart_round2": [],
        "archived_restart_round2": [],
    }
    game_keys = {game.key for game in games}

    def merge_restart_runs(
        paths_to_merge: Sequence[pathlib.Path], *, lineage_key: str,
        verify_local_build: bool,
    ) -> None:
        if not paths_to_merge:
            return
        grouped: dict[pathlib.Path, list[pathlib.Path]] = {}
        for path in paths_to_merge:
            resolved = path.resolve()
            grouped.setdefault(resolved.parent, []).append(resolved)
        for directory, paths in sorted(grouped.items()):
            restart_games, restart_sources, restart_lineage = (
                restart_contract.load_games(
                    paths, verify_local_build=verify_local_build
                )
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
            lineage[lineage_key].append(restart_lineage)
        lineage[lineage_key].sort(key=lambda item: (
            item["collector_tsv_sha256"], item["manifest_sha256"]
        ))

    merge_restart_runs(
        restart_round2_paths,
        lineage_key="live_restart_round2",
        verify_local_build=True,
    )
    merge_restart_runs(
        archived_restart_round2_paths,
        lineage_key="archived_restart_round2",
        verify_local_build=False,
    )
    games.sort(key=corpus_contract.game_sort_key)
    source_hashes = dict(sorted(source_hashes.items()))
    split_samples, overlaps_removed, assignments = corpus_contract.prepare_splits(games)
    split_sample_counts = {
        split: len(samples) for split, samples in split_samples.items()
    }
    empty = [split for split, count in split_sample_counts.items() if count == 0]
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
            split: split_sample_counts[split]
            for split in ("train", "validation", "test")
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
    # NativeGame/NativeSample objects retain Python integer tuples that are
    # much larger than the uint16 training representation.  All provenance
    # has now been reduced into ``report``; release the game graph, then drop
    # each sample while materializing its split to keep a ~2M-sample load from
    # holding both complete representations at once.
    del games
    datasets = {
        split: dataset_from_samples(split_samples[split], release=True)
        for split in ("train", "validation", "test")
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
                "prediction_std": None,
                "prediction_min": None,
                "prediction_max": None,
                "prediction_quantiles": None,
                "outcome_mean": None,
                "calibration_bias": None,
                "stable_reanalysis_samples": 0,
                "exact_reanalysis_samples": 0,
            }
            continue
        predicted = predictions[mask]
        outcome = dataset.outcome[mask]
        quantiles = np.quantile(
            predicted, (0.05, 0.25, 0.5, 0.75, 0.95)
        )
        result[name] = {
            "samples": count,
            "outcome_mse": float(np.mean((predicted - outcome) ** 2)),
            "outcome_sign_accuracy": float(np.mean(
                (predicted >= 0.0) == (outcome >= 0.0)
            )),
            "prediction_mean": float(np.mean(predicted)),
            "prediction_std": float(np.std(predicted)),
            "prediction_min": float(np.min(predicted)),
            "prediction_max": float(np.max(predicted)),
            "prediction_quantiles": {
                name: float(value) for name, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"), quantiles
                )
            },
            "outcome_mean": float(np.mean(outcome)),
            "calibration_bias": float(np.mean(predicted - outcome)),
            "stable_reanalysis_samples": int(np.count_nonzero(
                dataset.auxiliary_mask[mask] & ~dataset.exact_mask[mask]
            )),
            "exact_reanalysis_samples": int(np.count_nonzero(
                dataset.exact_mask[mask]
            )),
        }
    return result


def _phase_weights(turns: np.ndarray) -> np.ndarray:
    weights = np.full(turns.shape, np.float32(1.0), dtype=np.float32)
    for _, lower, upper, weight in PHASE_WEIGHTS:
        mask = turns >= lower
        if upper is not None:
            mask &= turns < upper
        weights[mask] = weight
    return weights


def _combined_target_loss(
    predictions: np.ndarray,
    outcome: np.ndarray,
    auxiliary: np.ndarray,
    auxiliary_mask: np.ndarray,
    exact_mask: np.ndarray,
    turns: np.ndarray,
    auxiliary_weight: float = AUXILIARY_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact-overridden mixture before and after phase weighting."""
    outcome_weights = np.where(
        exact_mask,
        0.0,
        np.where(auxiliary_mask, 1.0 - auxiliary_weight, 1.0),
    ).astype(np.float32)
    auxiliary_weights = np.where(
        exact_mask,
        1.0,
        np.where(auxiliary_mask, auxiliary_weight, 0.0),
    ).astype(np.float32)
    outcome_difference = predictions - outcome
    auxiliary_difference = predictions - auxiliary
    unweighted = (
        outcome_weights * outcome_difference * outcome_difference
        + auxiliary_weights * auxiliary_difference * auxiliary_difference
    )
    phase_weights = _phase_weights(turns)
    weighted = phase_weights * unweighted
    return (
        unweighted,
        weighted,
        outcome_weights,
        auxiliary_weights,
        phase_weights,
    )


def _phase_metrics(
    combined: np.ndarray, weighted: np.ndarray, dataset: Dataset
) -> dict[str, dict]:
    result = {}
    for name, lower, upper, weight in PHASE_WEIGHTS:
        mask = dataset.turn >= lower
        if upper is not None:
            mask &= dataset.turn < upper
        count = int(np.count_nonzero(mask))
        result[name] = {
            "samples": count,
            "weight": float(weight),
            "unweighted_combined_target_mse": (
                float(np.mean(combined[mask])) if count else None
            ),
            "weighted_combined_target_mse": (
                float(np.mean(weighted[mask])) if count else None
            ),
        }
    return result


def _weighted_validation_loss(validation: Mapping[str, object]) -> float:
    """Return the sole checkpoint-improvement objective explicitly."""
    return float(validation["weighted_combined_target_mse"])


def _validation_order_key(validation: Mapping[str, object]) -> tuple[float, ...]:
    """Order independent candidates after the approved weighted objective."""
    return (
        _weighted_validation_loss(validation),
        float(validation["unweighted_combined_target_mse"]),
        float(validation["outcome_mse"]),
    )


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
    combined, weighted, _, _, _ = _combined_target_loss(
        predictions,
        dataset.outcome,
        dataset.auxiliary,
        auxiliary_mask,
        dataset.exact_mask,
        dataset.turn,
    )
    weighted_mse = float(np.mean(weighted))
    return {
        "samples": len(dataset),
        "outcome_mse": float(np.mean(primary)),
        "outcome_mae": float(np.mean(np.abs(
            predictions - dataset.outcome
        ))),
        "outcome_sign_accuracy": float(np.mean(
            (predictions >= 0.0) == (dataset.outcome >= 0.0)
        )),
        # Preserve the established selection key while making its new
        # phase-weighted meaning explicit in the adjacent metrics.
        "combined_target_mse": weighted_mse,
        "weighted_combined_target_mse": weighted_mse,
        "unweighted_combined_target_mse": float(np.mean(combined)),
        "stable_reanalysis_samples": int(np.count_nonzero(auxiliary_mask)),
        "exact_reanalysis_samples": int(np.count_nonzero(dataset.exact_mask)),
        "stable_reanalysis_mse": auxiliary_mse,
        "prediction_mean": float(np.mean(predictions)),
        "phase_metrics": _phase_metrics(combined, weighted, dataset),
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


ROBUST_SCALE_QUANTILES = (
    ("p800", 800, 1_000),
    ("p900", 900, 1_000),
    ("p950", 950, 1_000),
    ("p975", 975, 1_000),
    ("p990", 990, 1_000),
    ("p995", 995, 1_000),
)
SCALE_SEARCH_PASSES = 2


def _scale_payload(scales: Mapping[str, object]) -> dict[str, float]:
    return {
        name: float(np.float32(scales[name]))
        for name in ("w1", "w2", "w3")
    }


def _fixed_quantize(
    parameters: Mapping[str, np.ndarray], scales: Mapping[str, object]
):
    integer: dict[str, np.ndarray] = {}
    normalized: dict[str, np.float32] = {}
    effective: dict[str, np.ndarray] = {}
    for name in ("w1", "w2", "w3"):
        scale = np.float32(scales[name])
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise ValueError(f"fixed {name} scale must be finite and positive")
        normalized[name] = scale
        integer[name] = np.clip(
            np.rint(parameters[name] / scale),
            -QUANTIZATION_MAX,
            QUANTIZATION_MAX,
        ).astype(np.int8)
        effective[name] = integer[name].astype(np.float32) * scale
    return integer, normalized, effective


def _canonical_fixed_quantization(
    parameters: Mapping[str, np.ndarray], scales: Mapping[str, object]
):
    """Return fixed-scale weights in the exporter's exact normal form.

    The runtime format derives each scale from the largest dequantized weight.
    Normalizing once through that frozen quantizer makes every returned tensor
    an exact ``q * scale`` checkpoint and guarantees that the existing exporter
    reproduces the same integers, scales, and float32 tensors byte-for-byte.
    """
    _, _, fixed_effective = _fixed_quantize(parameters, scales)
    integer, canonical_scales, effective = quantize(fixed_effective)
    again_integer, again_scales, again_effective = quantize(effective)
    for name in ("w1", "w2", "w3"):
        if (
            not np.array_equal(integer[name], again_integer[name])
            or canonical_scales[name] != again_scales[name]
            or not np.array_equal(effective[name], again_effective[name])
        ):
            raise RuntimeError(
                f"fixed-scale {name} checkpoint is not exporter-idempotent"
            )
    return integer, canonical_scales, effective


def _robust_scale_candidates(value: np.ndarray) -> tuple[np.float32, ...]:
    """Build deterministic clipping scales without consulting max-abs.

    The upper candidate is the lower-rank 99.5th percentile.  Consequently a
    rare maximum outlier can be clipped but can never determine the layer's
    only 3-bit step, which is the failure mode of the round-one checkpoint.
    """
    ordered = np.sort(np.abs(value).astype(np.float32, copy=False).reshape(-1))
    if not ordered.size:
        return (np.float32(1.0),)
    candidates: list[np.float32] = []
    for _, numerator, denominator in ROBUST_SCALE_QUANTILES:
        index = ((ordered.size - 1) * numerator) // denominator
        threshold = float(ordered[index])
        if threshold <= 0.0 or not math.isfinite(threshold):
            continue
        scale = np.float32(threshold / QUANTIZATION_MAX)
        if scale > 0.0 and all(scale != prior for prior in candidates):
            candidates.append(scale)
    if not candidates:
        positive = ordered[ordered > 0.0]
        if not positive.size:
            return (np.float32(1.0),)
        candidates.append(np.float32(float(positive[0]) / QUANTIZATION_MAX))
    return tuple(candidates)


def select_fixed_scales(
    parameters: Mapping[str, np.ndarray], validation: Dataset
) -> tuple[dict[str, np.float32], dict[str, np.ndarray], dict]:
    """Choose robust per-layer scales by deterministic coordinate search."""
    names = ("w1", "w2", "w3")
    candidates = {
        name: _robust_scale_candidates(parameters[name]) for name in names
    }
    requested = {name: candidates[name][-1] for name in names}
    trials = []
    for search_pass in range(SCALE_SEARCH_PASSES):
        for name in names:
            best = None
            for candidate in candidates[name]:
                trial_requested = dict(requested)
                trial_requested[name] = candidate
                _, trial_scales, effective = _canonical_fixed_quantization(
                    parameters, trial_requested
                )
                validation_metrics = metrics(effective, validation)
                coverage = quantized_w1_coverage(effective)
                loss = _weighted_validation_loss(validation_metrics)
                trial = {
                    "pass": search_pass + 1,
                    "layer": name,
                    "requested_scale": float(candidate),
                    "canonical_scales": _scale_payload(trial_scales),
                    "validation_combined_target_mse": loss,
                    "validation_weighted_combined_target_mse": (
                        validation_metrics["weighted_combined_target_mse"]
                    ),
                    "validation_unweighted_combined_target_mse": (
                        validation_metrics["unweighted_combined_target_mse"]
                    ),
                    "validation_phase_metrics": validation_metrics[
                        "phase_metrics"
                    ],
                    "validation_outcome_mse": validation_metrics["outcome_mse"],
                    "w1_coverage": coverage,
                }
                trials.append(trial)
                key = (*_validation_order_key(validation_metrics),
                       float(candidate))
                if best is None or key < best[0]:
                    best = (key, candidate)
            requested[name] = best[1]
    _, selected_scales, selected = _canonical_fixed_quantization(
        parameters, requested
    )
    selected_metrics = metrics(selected, validation)
    report = {
        "scheme": (
            "fixed-symmetric-3bit-validation-coordinate-search-"
            "lower-rank-robust-quantiles/v1"
        ),
        "selection_metric": VALIDATION_SELECTION_METRIC,
        "selection_tie_break": (
            "unweighted-combined-target-mse-then-outcome-mse-then-scale"
        ),
        "passes": SCALE_SEARCH_PASSES,
        "maximum_candidate_quantile": "p995-lower-rank",
        "max_abs_is_not_a_scale_candidate": True,
        "candidates": {
            name: [float(value) for value in candidates[name]]
            for name in names
        },
        "trials": trials,
        "selected_requested_scales": _scale_payload(requested),
        "selected_canonical_scales": _scale_payload(selected_scales),
        "selected_validation_combined_target_mse": selected_metrics[
            "combined_target_mse"
        ],
        "selected_validation_weighted_combined_target_mse": selected_metrics[
            "weighted_combined_target_mse"
        ],
        "selected_validation_unweighted_combined_target_mse": (
            selected_metrics["unweighted_combined_target_mse"]
        ),
        "selected_validation_phase_metrics": selected_metrics[
            "phase_metrics"
        ],
        "selected_validation_turn_calibration": selected_metrics[
            "turn_calibration"
        ],
        "selected_w1_coverage": quantized_w1_coverage(selected),
        "exporter_round_trip_verified": True,
    }
    return dict(selected_scales), selected, report


def train_batch(
    master: Mapping[str, np.ndarray],
    optimizer: AdamW,
    dataset: Dataset,
    indices: np.ndarray,
    auxiliary_weight: float,
    quantization_aware: bool,
    fixed_scales: Mapping[str, object] | None = None,
) -> float:
    active = tuple(dataset.active[int(index)] for index in indices)
    if not quantization_aware:
        effective = master
    elif fixed_scales is None:
        _, _, effective = quantize(master)
    else:
        _, _, effective = _fixed_quantize(master, fixed_scales)
    prediction, cache = forward(effective, active)
    first_pre, first, second_pre, second, output_pre = cache
    outcome = dataset.outcome[indices]
    difference = prediction - outcome
    exact_mask = dataset.exact_mask[indices]
    auxiliary_mask = dataset.auxiliary_mask[indices]
    _, weighted_loss, outcome_weights, auxiliary_weights, phase_weights = (
        _combined_target_loss(
            prediction,
            outcome,
            dataset.auxiliary[indices],
            auxiliary_mask,
            exact_mask,
            dataset.turn[indices],
            auxiliary_weight,
        )
    )
    weighted_outcome = phase_weights * outcome_weights
    weighted_auxiliary = phase_weights * auxiliary_weights
    loss = float(np.mean(weighted_loss))
    output_gradient = (
        2.0 * weighted_outcome * difference / max(len(indices), 1)
    )
    if np.any(auxiliary_weights):
        auxiliary_difference = prediction - dataset.auxiliary[indices]
        output_gradient += (
            2.0 * weighted_auxiliary * auxiliary_difference
            / max(len(indices), 1)
        )

    output_pre_gradient = (
        output_gradient * round1_trainer._fast_tanh_derivative(output_pre)
    )
    gradients: dict[str, np.ndarray] = {
        "w3": second.T @ output_pre_gradient,
    }
    second_gradient = output_pre_gradient[:, None] * effective["w3"][None, :]
    second_pre_gradient = (
        second_gradient * round1_trainer._leaky_relu_derivative(second_pre)
    )
    gradients["w2"] = first.T @ second_pre_gradient
    first_gradient = second_pre_gradient @ effective["w2"].T
    first_pre_gradient = (
        first_gradient * round1_trainer._hidden_one_derivative(first_pre)
    )
    gradients["w1"] = np.zeros_like(master["w1"])
    for row, active_indices in enumerate(active):
        np.add.at(gradients["w1"], active_indices, first_pre_gradient[row])

    squared_norm = sum(
        float(np.sum(value * value)) for value in gradients.values()
    )
    norm = math.sqrt(squared_norm)
    if norm > 5.0:
        gradient_scale = np.float32(5.0 / norm)
        for gradient in gradients.values():
            gradient *= gradient_scale
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
        _, dynamic_scales, dynamic_effective = quantize(master)
        dynamic_validation = metrics(
            dynamic_effective, datasets["validation"]
        )
        history.append({
            "epoch": epoch,
            "train_combined_mse": float(np.mean(losses)),
            "validation_outcome_mse": validation["outcome_mse"],
            "validation_combined_target_mse": validation[
                "combined_target_mse"
            ],
            "validation_weighted_combined_target_mse": validation[
                "weighted_combined_target_mse"
            ],
            "validation_unweighted_combined_target_mse": validation[
                "unweighted_combined_target_mse"
            ],
            "validation_phase_metrics": validation["phase_metrics"],
            "validation_turn_calibration": validation["turn_calibration"],
            "dynamic_max_quantization": {
                "scales": _scale_payload(dynamic_scales),
                "validation_combined_target_mse": dynamic_validation[
                    "combined_target_mse"
                ],
                "validation_weighted_combined_target_mse": (
                    dynamic_validation["weighted_combined_target_mse"]
                ),
                "validation_unweighted_combined_target_mse": (
                    dynamic_validation["unweighted_combined_target_mse"]
                ),
                "validation_phase_metrics": dynamic_validation[
                    "phase_metrics"
                ],
                "validation_turn_calibration": dynamic_validation[
                    "turn_calibration"
                ],
                "w1_coverage": quantized_w1_coverage(master),
            },
        })
        print(
            f"round2 seed {seed} epoch {epoch}: validation combined MSE "
            f"{validation['combined_target_mse']:.6f}", flush=True,
        )
        validation_loss = _weighted_validation_loss(validation)
        if validation_loss < best_float_loss - 1e-8:
            best_float_loss = validation_loss
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
    _, dynamic_scales, dynamic_effective = quantize(best_float)
    dynamic_max_baseline = {
        "scales": _scale_payload(dynamic_scales),
        "metrics": {
            split: metrics(dynamic_effective, dataset)
            for split, dataset in datasets.items()
        },
        "w1_coverage": quantized_w1_coverage(best_float),
    }
    fixed_scales, pre_qat_effective, scale_search = select_fixed_scales(
        best_float, datasets["validation"]
    )
    selected = {
        name: value.copy() for name, value in pre_qat_effective.items()
    }
    pre_qat_quantized_metrics = {
        split: metrics(pre_qat_effective, dataset)
        for split, dataset in datasets.items()
    }
    selected_loss = _weighted_validation_loss(
        pre_qat_quantized_metrics["validation"]
    )
    selected_qat_epoch = 0

    master = {name: value.copy() for name, value in best_float.items()}
    optimizer = AdamW(master, learning_rate * 0.25, weight_decay)
    for qat_epoch in range(1, qat_epochs + 1):
        order = rng.permutation(len(datasets["train"]))
        for start in range(0, len(order), batch_size):
            train_batch(
                master, optimizer, datasets["train"],
                order[start:start + batch_size], auxiliary_weight, True,
                fixed_scales=fixed_scales,
            )
        _, canonical_scales, effective = _canonical_fixed_quantization(
            master, fixed_scales
        )
        validation = metrics(effective, datasets["validation"])
        validation_loss = _weighted_validation_loss(validation)
        history.append({
            "qat_epoch": qat_epoch,
            "fixed_scales": _scale_payload(fixed_scales),
            "canonical_scales": _scale_payload(canonical_scales),
            "validation_quantized_outcome_mse": validation["outcome_mse"],
            "validation_quantized_combined_target_mse": validation_loss,
            "validation_quantized_weighted_combined_target_mse": validation[
                "weighted_combined_target_mse"
            ],
            "validation_quantized_unweighted_combined_target_mse": (
                validation["unweighted_combined_target_mse"]
            ),
            "validation_phase_metrics": validation["phase_metrics"],
            "validation_turn_calibration": validation["turn_calibration"],
            "quantized_w1_coverage": quantized_w1_coverage(effective),
        })
        # A tie deliberately retains the pre-QAT best.
        if validation_loss < selected_loss - 1e-8:
            selected_loss = validation_loss
            selected_qat_epoch = qat_epoch
            selected = {name: value.copy() for name, value in effective.items()}

    _, selected_scales, effective = quantize(selected)
    if any(
        not np.array_equal(selected[name], effective[name])
        for name in ("w1", "w2", "w3")
    ):
        raise RuntimeError("selected round-two checkpoint is not exporter-idempotent")
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
            "selection_metric": VALIDATION_SELECTION_METRIC,
            "fixed_scale_qat": True,
            "fixed_scales": _scale_payload(fixed_scales),
            "selected_export_scales": _scale_payload(selected_scales),
        },
        "history": history,
        "float_best_metrics": float_best_metrics,
        "dynamic_max_baseline": dynamic_max_baseline,
        "scale_search": scale_search,
        "pre_qat_quantized_metrics": pre_qat_quantized_metrics,
        "selected_dequantized_metrics": {
            split: metrics(selected, dataset)
            for split, dataset in datasets.items()
        },
        "quantized_metrics": {
            split: metrics(effective, dataset)
            for split, dataset in datasets.items()
        },
        "quantized_w1_coverage": quantized_w1_coverage(selected),
        "exporter_round_trip_verified": True,
    }
    return effective, report


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
            *_validation_order_key(
                seed_reports[index]["quantized_metrics"]["validation"]
            ),
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
            "phase_weights": {
                name: float(weight) for name, _, _, weight in PHASE_WEIGHTS
            },
            "phase_weight_application": PHASE_WEIGHT_APPLICATION,
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
                "provisional-minimum-quantized-validation-phase-weighted-"
                "combined-target-mse-then-unweighted-combined-target-mse-"
                "then-outcome-mse-then-seed;"
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


def write_output_exclusive(path: pathlib.Path, raw: bytes) -> None:
    """Atomically install a fresh training artifact without replacing history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = pathlib.Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable training artifact: {path}"
            ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train cumulative ground-up Jacek-native round-two values."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path,
                        help="strict-current round-two shards")
    parser.add_argument("--archived-round1", nargs="*", type=pathlib.Path,
                        default=[])
    parser.add_argument(
        "--archived-round2", nargs="*", type=pathlib.Path, default=[],
        help=(
            "explicit canonical round-two archives; validate their archived "
            "identities without requiring current source/compiler hashes"
        ),
    )
    parser.add_argument(
        "--restart-round2", nargs="*", type=pathlib.Path, default=[],
        help=(
            "explicit provenance-safe live-restart shards; observed moves "
            "construct states only and never become labels"
        ),
    )
    parser.add_argument(
        "--archived-restart-round2", nargs="*", type=pathlib.Path,
        default=[],
        help=(
            "explicit canonical historical live-restart archives; validate "
            "archived binary/checkpoint/collector identities without requiring "
            "current source/compiler hashes"
        ),
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True,
        help="fresh immutable output path (existing files are rejected)",
    )
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
        arguments.corpus,
        archived_round1_paths=arguments.archived_round1,
        restart_round2_paths=arguments.restart_round2,
        archived_round2_paths=arguments.archived_round2,
        archived_restart_round2_paths=arguments.archived_restart_round2,
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
    model_raw = (json.dumps(
        model, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) + "\n").encode()
    write_output_exclusive(arguments.output, model_raw)
    print(json.dumps({
        "output": str(arguments.output),
        "provisional_seed": model["training"]["provisional_seed"],
        "external_actual_clock_selection": "pending",
        "measured_examples_per_second": examples / elapsed,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
