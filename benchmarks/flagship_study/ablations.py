"""Preregistered paired development/validation ablations.

The contrasts consume compact per-opening pair scores, align configurations on
the same frozen opening IDs, and resample whole pair-score differences within
opening-depth strata.  Test data is deliberately not an input.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from benchmarks.flagship_study import studylib


SCHEMA = "papersoccer.flagship-study-ablations.v1"


class AblationError(ValueError):
    """Raised when curated inputs cannot support a frozen ablation."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AblationError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AblationError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AblationError(f"{where} must be an integer >= {minimum}")
    return value


def _score(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationError(f"{where} must be a pair score")
    result = float(value)
    if result not in (0.0, 0.5, 1.0):
        raise AblationError(f"{where} must be 0, 0.5, or 1")
    return result


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AblationError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _paired_scores(curated: Mapping[str, Any], phase: str,
                   expected_ids: set[str], *, expected_depths: tuple[int, ...],
                   pairs_per_depth: int
                   ) -> dict[str, dict[tuple[int, str], float]]:
    if curated.get("phase") != phase:
        raise AblationError(f"expected curated {phase} data")
    payloads = _mapping(curated.get("paired_scores"), f"{phase}.paired_scores")
    if set(payloads) != expected_ids:
        raise AblationError(f"{phase} paired scores do not match the candidate grid")
    result: dict[str, dict[tuple[int, str], float]] = {}
    for config_id in sorted(expected_ids):
        payload = _mapping(payloads[config_id], f"{phase}.paired_scores.{config_id}")
        if set(payload) != {
            "phase", "bot_id", "opponent_config_id", "opening_ids",
            "opening_depths", "scores",
        }:
            raise AblationError(f"{phase} paired score fields changed for {config_id}")
        if (payload["phase"] != phase or payload["bot_id"] != config_id
                or payload["opponent_config_id"] != "rank5-fixed-50k"):
            raise AblationError(f"{phase} paired score identity changed for {config_id}")
        opening_ids = _sequence(payload["opening_ids"], "paired opening IDs")
        depths = _sequence(payload["opening_depths"], "paired opening depths")
        scores = _sequence(payload["scores"], "paired scores")
        if not opening_ids or len(opening_ids) != len(depths) or len(depths) != len(scores):
            raise AblationError(f"{phase} paired score columns are empty or misaligned")
        rows: dict[tuple[int, str], float] = {}
        for index, (opening_id, depth, score) in enumerate(
            zip(opening_ids, depths, scores, strict=True)
        ):
            key = (
                _integer(depth, f"paired opening depth {index}", minimum=1),
                _string(opening_id, f"paired opening ID {index}"),
            )
            if key in rows:
                raise AblationError(f"duplicate aligned opening pair for {config_id}: {key}")
            rows[key] = _score(score, f"paired score {index}")
        depth_counts = {
            depth: sum(key[0] == depth for key in rows)
            for depth in expected_depths
        }
        if (set(key[0] for key in rows) != set(expected_depths)
                or any(count != pairs_per_depth for count in depth_counts.values())
                or len(rows) != len(expected_depths) * pairs_per_depth):
            raise AblationError(
                f"{phase} paired scores for {config_id} do not match the frozen "
                "per-depth sample counts"
            )
        if "binary_games" in curated and "matchups" in curated:
            matchups = _mapping(curated["matchups"], f"{phase}.matchups")
            matching = [
                matchup_id
                for matchup_id, raw_matchup in matchups.items()
                if _mapping(raw_matchup, f"{phase}.matchups.{matchup_id}").get(
                    "left_config_id"
                ) == config_id
            ]
            if len(matching) != 1:
                raise AblationError(
                    f"{phase} cannot bind {config_id} to exactly one tuning matchup"
                )
            matchup_id = matching[0]
            prefix = f"{phase}:{matchup_id}:"
            expected_keys: set[tuple[int, str]] = set()
            for raw_game in _sequence(
                    curated["binary_games"], f"{phase}.binary_games"):
                game = _mapping(raw_game, f"{phase} binary game")
                if game.get("matchup_id") != matchup_id:
                    continue
                pair_id = _string(game.get("pair_id"), "binary pair ID")
                if not pair_id.startswith(prefix):
                    raise AblationError("binary pair ID is not bound to its matchup")
                expected_keys.add((
                    _integer(game.get("opening_depth"), "binary opening depth", minimum=1),
                    pair_id[len(prefix):],
                ))
            if set(rows) != expected_keys:
                raise AblationError(
                    f"{phase} paired-score openings differ from curated binary games "
                    f"for {config_id}"
                )
        result[config_id] = rows
    key_sets = {frozenset(rows) for rows in result.values()}
    if len(key_sets) != 1:
        raise AblationError(f"{phase} candidate scores are not aligned on identical openings")
    return result


def _paired_difference_interval(
    lower: Mapping[tuple[int, str], float],
    higher: Mapping[tuple[int, str], float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if set(lower) != set(higher) or not lower:
        raise AblationError("paired difference requires identical non-empty openings")
    strata: dict[int, list[float]] = defaultdict(list)
    for key in sorted(lower):
        strata[key[0]].append(higher[key] - lower[key])
    generator = random.Random(seed)
    total_pairs = len(lower)
    samples: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for values in (strata[depth] for depth in sorted(strata)):
            total += sum(values[generator.randrange(len(values))] for _ in values)
        samples.append(total / total_pairs)
    return {
        "method": "paired_difference_percentile_stratified",
        "seed": str(seed),
        "resamples": resamples,
        "confidence": 0.95,
        "lower": _nearest_rank(samples, 0.025),
        "upper": _nearest_rank(samples, 0.975),
    }


def _budget(config: Mapping[str, Any]) -> int:
    settings = _mapping(config.get("settings"), "configuration settings")
    value = settings.get("iterations", settings.get("max_nodes"))
    return _integer(value, "configuration budget", minimum=1)


def _scaling_classification(lower: float, upper: float,
                            threshold: float) -> str:
    if upper < 0.0:
        return "supported_regression"
    if lower > threshold:
        return "supported_practical_gain"
    if upper < threshold:
        return "supported_no_practical_gain"
    return "unresolved_at_1pp"


def _evaluator_classification(lower: float, upper: float,
                              threshold: float) -> str:
    if lower > threshold:
        return "neural_materially_stronger"
    if upper < -threshold:
        return "hand_materially_stronger"
    if lower >= -threshold and upper <= threshold:
        return "practical_equivalence_supported"
    return "unresolved_at_1pp"


def compute(
    manifest: Mapping[str, Any],
    development: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the exact manifest-declared dev/validation contrast set."""

    contract = _mapping(
        _mapping(manifest.get("statistics"), "manifest.statistics").get("ablations"),
        "statistics.ablations",
    )
    comparisons = _mapping(contract.get("comparisons"), "ablation comparisons")
    expected_families = {
        "mcts", "alpha_beta", "jacek_inspired", "equal_budget_evaluator"
    }
    if set(comparisons) != expected_families:
        raise AblationError("ablation comparison families changed")
    comparison_pairs: dict[str, list[tuple[str, str]]] = {}
    expected_ids: set[str] = set()
    for family in sorted(expected_families):
        values = _sequence(comparisons[family], f"{family} comparisons")
        pairs: list[tuple[str, str]] = []
        for index, raw_pair in enumerate(values):
            pair = _sequence(raw_pair, f"{family} comparison {index}")
            if len(pair) != 2:
                raise AblationError("each ablation comparison must contain two IDs")
            lower_id = _string(pair[0], "ablation lower ID")
            higher_id = _string(pair[1], "ablation higher ID")
            pairs.append((lower_id, higher_id))
            expected_ids.update((lower_id, higher_id))
        comparison_pairs[family] = pairs

    expected_depths = tuple(
        _integer(value, "opening depth", minimum=1)
        for value in _sequence(
            _mapping(manifest.get("openings"), "manifest.openings").get("depths"),
            "manifest.openings.depths",
        )
    )
    samples = _mapping(manifest.get("samples"), "manifest.samples")
    scores = {
        phase: _paired_scores(
            curated, phase, expected_ids,
            expected_depths=expected_depths,
            pairs_per_depth=_integer(
                _mapping(samples.get(phase), f"samples.{phase}").get(
                    "color_swapped_pairs_per_depth_matchup"
                ),
                f"samples.{phase}.color_swapped_pairs_per_depth_matchup",
                minimum=1,
            ),
        )
        for phase, curated in (
            ("development", development), ("validation", validation)
        )
    }
    configs = {
        _string(config.get("id"), "configuration ID"): _mapping(config, "configuration")
        for config in _sequence(manifest.get("configurations"), "configurations")
    }
    if not expected_ids <= set(configs):
        raise AblationError("ablation comparison names an unknown configuration")
    resamples = _integer(contract.get("bootstrap_resamples"), "ablation resamples", minimum=1)
    threshold_raw = contract.get("practical_gain_threshold")
    if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, (int, float)):
        raise AblationError("ablation threshold must be numeric")
    threshold = float(threshold_raw)

    def comparison(family: str, lower_id: str, higher_id: str,
                   *, evaluator: bool) -> dict[str, Any]:
        phase_results: dict[str, Any] = {}
        for phase in ("development", "validation"):
            lower = scores[phase][lower_id]
            higher = scores[phase][higher_id]
            seed = studylib._derived_seed(
                manifest["seeds"]["analysis"][phase], "ablation", family,
                lower_id, higher_id,
            )
            interval = _paired_difference_interval(
                lower, higher, seed=seed, resamples=resamples
            )
            lower_mean = sum(lower.values()) / len(lower)
            higher_mean = sum(higher.values()) / len(higher)
            phase_results[phase] = {
                "pairs": len(lower),
                "lower_score": lower_mean,
                "higher_score": higher_mean,
                "delta": higher_mean - lower_mean,
                "pair_difference_bootstrap_95": interval,
            }
        validation_interval = phase_results["validation"][
            "pair_difference_bootstrap_95"
        ]
        classification = (
            _evaluator_classification(
                validation_interval["lower"], validation_interval["upper"],
                threshold,
            )
            if evaluator else _scaling_classification(
                validation_interval["lower"], validation_interval["upper"],
                threshold,
            )
        )
        return {
            "id": f"{family}:{lower_id}-to-{higher_id}",
            "contrast": "neural_minus_hand" if evaluator
            else "higher_budget_minus_lower_budget",
            "lower_config_id": lower_id,
            "higher_config_id": higher_id,
            "lower_budget": _budget(configs[lower_id]),
            "higher_budget": _budget(configs[higher_id]),
            "phases": phase_results,
            "validation_classification": classification,
        }

    scaling = {
        family: [
            comparison(family, lower_id, higher_id, evaluator=False)
            for lower_id, higher_id in comparison_pairs[family]
        ]
        for family in ("mcts", "alpha_beta", "jacek_inspired")
    }
    evaluator = [
        comparison("equal_budget_evaluator", lower_id, higher_id, evaluator=True)
        for lower_id, higher_id in comparison_pairs["equal_budget_evaluator"]
    ]
    return {
        "schema": SCHEMA,
        "source_phases": ["development", "validation"],
        "practical_gain_threshold": threshold,
        "bootstrap": {
            "method": contract["bootstrap_method"],
            "resamples": resamples,
            "confidence": 0.95,
            "unit": contract["comparison_unit"],
            "stratify_by": contract["stratify_by"],
        },
        "scaling": scaling,
        "equal_budget_evaluator": evaluator,
    }


__all__ = ["AblationError", "SCHEMA", "compute"]
