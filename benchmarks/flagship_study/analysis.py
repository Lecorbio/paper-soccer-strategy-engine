"""Deterministic statistical analysis for the flagship bot study.

The module intentionally depends only on the Python standard library and the
study's local ``studylib`` helpers.  All fitting entry points validate their
inputs before doing any numerical work; in particular, truncations never enter
strength estimates, cached Rank5 continuations never enter calibration, and a
calibration fit can consume validation observations only.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from benchmarks.flagship_study import studylib


BOOTSTRAP_RESAMPLES = 10_000
RELIABILITY_BINS = 10


class AnalysisError(ValueError):
    """Raised when an analysis input or numerical result is invalid."""


class DisconnectedComparisonError(AnalysisError):
    """Raised when a Bradley--Terry comparison graph is disconnected."""


class SeparationError(AnalysisError):
    """Raised when finite maximum-likelihood estimates do not exist."""


class ConvergenceError(AnalysisError):
    """Raised when an iterative fit cannot reach its convergence criterion."""


def _binary(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise AnalysisError(f"{where} must be the integer 0 or 1")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{where} must be a finite number")
    return result


def _positive_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AnalysisError(f"{where} must be a positive integer")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError(f"{where} must be a boolean")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{where} must be a non-empty string")
    return value


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _bootstrap_parameters(seed: Any, resamples: Any) -> tuple[int, int]:
    if (isinstance(seed, bool) or not isinstance(seed, int) or
            seed < 0 or seed > (1 << 64) - 1):
        raise AnalysisError("bootstrap seed must be a uint64 integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise AnalysisError("bootstrap resamples must be a positive integer")
    return seed, resamples


def _convergence_parameters(tolerance: Any, max_iterations: Any) -> tuple[float, int]:
    checked_tolerance = _finite(tolerance, "convergence tolerance")
    if checked_tolerance <= 0.0:
        raise AnalysisError("convergence tolerance must be positive")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or
            max_iterations <= 0):
        raise AnalysisError("max_iterations must be a positive integer")
    return checked_tolerance, max_iterations


@dataclasses.dataclass(frozen=True)
class PairedComparison:
    """One color-swapped pair, oriented so outcomes are for ``bot_a``."""

    pair_id: str
    opening_depth: int
    bot_a: str
    bot_b: str
    outcomes_for_a: tuple[int, int]
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["outcomes_for_a"] = list(self.outcomes_for_a)
        return value


def _paired_comparison(value: PairedComparison | Mapping[str, Any],
                       index: int) -> PairedComparison:
    if isinstance(value, PairedComparison):
        pair = value
    elif isinstance(value, Mapping):
        outcomes = value.get("outcomes_for_a")
        if not isinstance(outcomes, (list, tuple)):
            raise AnalysisError(
                f"pairs[{index}].outcomes_for_a must contain exactly two games"
            )
        pair = PairedComparison(
            pair_id=value.get("pair_id"),
            opening_depth=value.get("opening_depth"),
            bot_a=value.get("bot_a"),
            bot_b=value.get("bot_b"),
            outcomes_for_a=tuple(outcomes),
            truncated=value.get("truncated", False),
        )
    else:
        raise AnalysisError(f"pairs[{index}] must be a paired comparison")

    pair_id = _nonempty_string(pair.pair_id, f"pairs[{index}].pair_id")
    opening_depth = _positive_integer(
        pair.opening_depth, f"pairs[{index}].opening_depth"
    )
    bot_a = _nonempty_string(pair.bot_a, f"pairs[{index}].bot_a")
    bot_b = _nonempty_string(pair.bot_b, f"pairs[{index}].bot_b")
    if bot_a == bot_b:
        raise AnalysisError(f"pairs[{index}] compares a bot with itself")
    if pair.truncated is not False:
        raise AnalysisError(f"truncation in pair {pair_id}; strength analysis refused")
    if len(pair.outcomes_for_a) != 2:
        raise AnalysisError(f"pair {pair_id} must contain exactly two binary games")
    outcomes = tuple(
        _binary(outcome, f"pairs[{index}].outcomes_for_a[{game_index}]")
        for game_index, outcome in enumerate(pair.outcomes_for_a)
    )
    return PairedComparison(
        pair_id=pair_id,
        opening_depth=opening_depth,
        bot_a=bot_a,
        bot_b=bot_b,
        outcomes_for_a=outcomes,
        truncated=False,
    )


def _validated_pairs(
    pairs: Sequence[PairedComparison | Mapping[str, Any]],
    *,
    require_unique_ids: bool = True,
) -> list[PairedComparison]:
    if not pairs:
        raise AnalysisError("at least one color-swapped pair is required")
    result = [_paired_comparison(value, index) for index, value in enumerate(pairs)]
    if require_unique_ids:
        identifiers = [pair.pair_id for pair in result]
        if len(set(identifiers)) != len(identifiers):
            raise AnalysisError("pair IDs must be unique")
    return result


def pair_score(outcomes_for_candidate: Sequence[int]) -> float:
    """Return 1, 0.5, or 0 for exactly two non-truncated binary games."""

    if len(outcomes_for_candidate) != 2:
        raise AnalysisError("a color-swapped pair must contain exactly two games")
    wins = sum(
        _binary(outcome, f"outcomes_for_candidate[{index}]")
        for index, outcome in enumerate(outcomes_for_candidate)
    )
    return wins / 2.0


def summarize_pair_outcomes(
    pairs: Sequence[PairedComparison | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize a single ordered matchup across opening-depth strata."""

    checked = _validated_pairs(pairs)
    matchup = {(pair.bot_a, pair.bot_b) for pair in checked}
    if len(matchup) != 1:
        raise AnalysisError("pair summary requires one consistently oriented matchup")
    scores = [pair_score(pair.outcomes_for_a) for pair in checked]
    game_wins = sum(sum(pair.outcomes_for_a) for pair in checked)
    won = sum(score == 1.0 for score in scores)
    split = sum(score == 0.5 for score in scores)
    lost = sum(score == 0.0 for score in scores)
    bot_a, bot_b = next(iter(matchup))
    return {
        "bot": bot_a,
        "opponent": bot_b,
        "pairs": len(checked),
        "games": len(checked) * 2,
        "game_wins": game_wins,
        "game_losses": len(checked) * 2 - game_wins,
        "truncations": 0,
        "pairs_won_2_0": won,
        "pairs_split_1_1": split,
        "pairs_lost_0_2": lost,
        "mean_pair_score": sum(scores) / len(scores),
    }


def depth_stratified_pair_bootstrap(
    pairs: Sequence[PairedComparison | Mapping[str, Any]],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Percentile-bootstrap whole pairs while preserving depth sample sizes."""

    seed, resamples = _bootstrap_parameters(seed, resamples)
    checked = _validated_pairs(pairs)
    matchup = {(pair.bot_a, pair.bot_b) for pair in checked}
    if len(matchup) != 1:
        raise AnalysisError("pairwise bootstrap requires one ordered matchup")
    strata: dict[int, list[float]] = defaultdict(list)
    for pair in checked:
        strata[pair.opening_depth].append(pair_score(pair.outcomes_for_a))
    result = studylib.stratified_pair_bootstrap(strata, seed, resamples)
    result["pairs"] = len(checked)
    result["opening_depths"] = sorted(strata)
    return result


@dataclasses.dataclass(frozen=True)
class BradleyTerryFit:
    bot_ids: tuple[str, ...]
    abilities: dict[str, float]
    games: int
    iterations: int
    log_likelihood: float
    converged: bool = True
    identifiability: str = "sum_to_zero"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_ids": list(self.bot_ids),
            "abilities": dict(self.abilities),
            "games": self.games,
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "identifiability": self.identifiability,
        }


def _reachable(start: int, adjacency: Sequence[set[int]]) -> set[int]:
    reached = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for neighbour in adjacency[node]:
            if neighbour not in reached:
                reached.add(neighbour)
                pending.append(neighbour)
    return reached


def _solve_linear(matrix: Sequence[Sequence[float]],
                  right_hand_side: Sequence[float]) -> list[float]:
    size = len(right_hand_side)
    augmented = [list(matrix[row]) + [right_hand_side[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ConvergenceError("singular information matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0.0:
                continue
            augmented[row] = [
                augmented[row][entry] - multiplier * augmented[column][entry]
                for entry in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _bt_counts(pairs: Sequence[PairedComparison], bot_ids: tuple[str, ...]) \
        -> tuple[list[list[int]], list[list[int]]]:
    index = {bot_id: position for position, bot_id in enumerate(bot_ids)}
    wins = [[0 for _ in bot_ids] for _ in bot_ids]
    games = [[0 for _ in bot_ids] for _ in bot_ids]
    for pair in pairs:
        if pair.bot_a not in index or pair.bot_b not in index:
            raise AnalysisError(f"pair {pair.pair_id} names a bot outside bot_ids")
        left = index[pair.bot_a]
        right = index[pair.bot_b]
        for outcome in pair.outcomes_for_a:
            games[left][right] += 1
            games[right][left] += 1
            if outcome == 1:
                wins[left][right] += 1
            else:
                wins[right][left] += 1
    return wins, games


def _check_bt_graph(wins: Sequence[Sequence[int]],
                    games: Sequence[Sequence[int]]) -> None:
    size = len(wins)
    undirected = [set() for _ in range(size)]
    directed = [set() for _ in range(size)]
    for left in range(size):
        for right in range(size):
            if games[left][right] > 0:
                undirected[left].add(right)
            if wins[left][right] > 0:
                directed[left].add(right)
    if len(_reachable(0, undirected)) != size:
        raise DisconnectedComparisonError("Bradley-Terry comparison graph is disconnected")
    if any(len(_reachable(bot, directed)) != size for bot in range(size)):
        raise SeparationError(
            "Bradley-Terry outcomes are separated; finite abilities do not exist"
        )


def _bt_log_likelihood(theta: Sequence[float], wins: Sequence[Sequence[int]]) -> float:
    result = 0.0
    for left in range(len(theta)):
        for right in range(left + 1, len(theta)):
            difference = theta[left] - theta[right]
            result += wins[left][right] * _log_sigmoid(difference)
            result += wins[right][left] * _log_sigmoid(-difference)
    return result


def _fit_bt_checked(
    pairs: Sequence[PairedComparison],
    bot_ids: tuple[str, ...],
    *,
    tolerance: float,
    max_iterations: int,
) -> BradleyTerryFit:
    wins, games = _bt_counts(pairs, bot_ids)
    _check_bt_graph(wins, games)
    size = len(bot_ids)
    theta = [0.0] * size
    total_games = sum(sum(row) for row in wins)

    for iteration in range(max_iterations + 1):
        gradient = [0.0] * size
        information = [[0.0] * size for _ in range(size)]
        for left in range(size):
            for right in range(left + 1, size):
                count = games[left][right]
                if count == 0:
                    continue
                probability = _sigmoid(theta[left] - theta[right])
                residual = wins[left][right] - count * probability
                weight = count * probability * (1.0 - probability)
                gradient[left] += residual
                gradient[right] -= residual
                information[left][left] += weight
                information[right][right] += weight
                information[left][right] -= weight
                information[right][left] -= weight

        if max(abs(value) for value in gradient) <= tolerance:
            centered = [value - sum(theta) / size for value in theta]
            return BradleyTerryFit(
                bot_ids=bot_ids,
                abilities=dict(zip(bot_ids, centered, strict=True)),
                games=total_games,
                iterations=iteration,
                log_likelihood=_bt_log_likelihood(centered, wins),
            )
        if iteration == max_iterations:
            break

        augmented = [row + [1.0] for row in information]
        augmented.append([1.0] * size + [0.0])
        step = _solve_linear(augmented, gradient + [0.0])[:size]
        current_likelihood = _bt_log_likelihood(theta, wins)
        multiplier = 1.0
        accepted = False
        for _ in range(60):
            candidate = [theta[index] + multiplier * step[index]
                         for index in range(size)]
            center = sum(candidate) / size
            candidate = [value - center for value in candidate]
            candidate_likelihood = _bt_log_likelihood(candidate, wins)
            if candidate_likelihood >= current_likelihood - 1e-12:
                theta = candidate
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or max(abs(value) for value in theta) > 50.0:
            raise ConvergenceError("Bradley-Terry Newton iteration diverged")

    raise ConvergenceError(
        f"Bradley-Terry fit did not converge within {max_iterations} iterations"
    )


def fit_bradley_terry(
    pairs: Sequence[PairedComparison | Mapping[str, Any]],
    *,
    bot_ids: Sequence[str] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> BradleyTerryFit:
    """Fit four-bot Bradley--Terry abilities under a sum-to-zero constraint."""

    tolerance, max_iterations = _convergence_parameters(
        tolerance, max_iterations
    )
    checked = _validated_pairs(pairs)
    observed = {pair.bot_a for pair in checked} | {pair.bot_b for pair in checked}
    resolved = tuple(
        sorted(observed) if bot_ids is None else
        (_nonempty_string(value, f"bot_ids[{index}]")
         for index, value in enumerate(bot_ids))
    )
    if len(resolved) != 4 or len(set(resolved)) != 4:
        raise AnalysisError("Bradley-Terry flagship fit requires exactly four bots")
    if observed - set(resolved):
        raise AnalysisError("observed comparison names a bot outside bot_ids")
    return _fit_bt_checked(
        checked, resolved, tolerance=tolerance, max_iterations=max_iterations
    )


def bootstrap_bradley_terry(
    pairs: Sequence[PairedComparison | Mapping[str, Any]],
    *,
    seed: int,
    bot_ids: Sequence[str] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    minimum_success_fraction: float = 1.0,
) -> dict[str, Any]:
    """Matchup/depth-stratified whole-pair bootstrap for BT abilities.

    The strict default refuses intervals when any bootstrap refit fails.  A
    different threshold must therefore be an explicit preregistered caller
    choice rather than an implicit, post-outcome relaxation.
    """

    seed, resamples = _bootstrap_parameters(seed, resamples)
    tolerance, max_iterations = _convergence_parameters(
        tolerance, max_iterations
    )
    checked = _validated_pairs(pairs)
    if not 0.0 <= minimum_success_fraction <= 1.0:
        raise AnalysisError("minimum_success_fraction must be in [0,1]")
    base = fit_bradley_terry(
        checked,
        bot_ids=bot_ids,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    strata: dict[tuple[int, str, str], list[PairedComparison]] = defaultdict(list)
    for pair in checked:
        matchup = tuple(sorted((pair.bot_a, pair.bot_b)))
        strata[(pair.opening_depth, *matchup)].append(pair)
    generator = random.Random(seed)
    samples = {bot_id: [] for bot_id in base.bot_ids}
    failures = {"disconnected": 0, "separation": 0, "convergence": 0}
    for _ in range(resamples):
        replicate: list[PairedComparison] = []
        for stratum in sorted(strata):
            values = strata[stratum]
            replicate.extend(values[generator.randrange(len(values))] for _ in values)
        try:
            fitted = _fit_bt_checked(
                replicate,
                base.bot_ids,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        except DisconnectedComparisonError:
            failures["disconnected"] += 1
            continue
        except SeparationError:
            failures["separation"] += 1
            continue
        except ConvergenceError:
            failures["convergence"] += 1
            continue
        for bot_id in base.bot_ids:
            samples[bot_id].append(fitted.abilities[bot_id])

    successful = len(next(iter(samples.values())))
    required = math.ceil(resamples * minimum_success_fraction)
    if successful < max(1, required):
        raise ConvergenceError(
            "too few finite Bradley-Terry bootstrap fits: "
            f"{successful}/{resamples}; failures={failures}"
        )
    intervals = {
        bot_id: {
            "estimate": base.abilities[bot_id],
            "lower": _nearest_rank(samples[bot_id], 0.025),
            "upper": _nearest_rank(samples[bot_id], 0.975),
        }
        for bot_id in base.bot_ids
    }
    return {
        "method": "matchup_and_depth_stratified_color_swapped_pair_percentile",
        "identifiability": "sum_to_zero",
        "seed": str(seed),
        "resamples": resamples,
        "successful_resamples": successful,
        "failed_resamples": failures,
        "opening_depths": sorted({stratum[0] for stratum in strata}),
        "strata": [
            {
                "opening_depth": depth,
                "bot_a": left,
                "bot_b": right,
                "pairs": len(strata[(depth, left, right)]),
            }
            for depth, left, right in sorted(strata)
        ],
        "intervals": intervals,
    }


def _player_number(value: Any, where: str = "player_to_move") -> int:
    aliases = {
        1: 1,
        2: 2,
        "one": 1,
        "two": 2,
        "player_one": 1,
        "player_two": 2,
    }
    try:
        return aliases[value]
    except (KeyError, TypeError) as error:
        raise AnalysisError(f"{where} must identify Player One or Player Two") from error


def orient_player_one_score(score: float, player_to_move: int | str) -> float:
    """Convert a Player-One-oriented signed score to player-to-move orientation."""

    checked = _finite(score, "score")
    return checked if _player_number(player_to_move) == 1 else -checked


def orient_player_one_probability(probability: float,
                                  player_to_move: int | str) -> float:
    """Convert a Player-One win probability to player-to-move orientation."""

    checked = _finite(probability, "probability")
    if not 0.0 <= checked <= 1.0:
        raise AnalysisError("probability must be in [0,1]")
    return checked if _player_number(player_to_move) == 1 else 1.0 - checked


@dataclasses.dataclass(frozen=True)
class CalibrationMapping:
    bot_id: str
    score_kind: str
    score_mean: float
    score_scale: float
    intercept: float
    slope: float
    sample_count: int
    iterations: int
    excluded_cached_continuations: int = 0
    excluded_truncations: int = 0
    excluded_invalid_depths: int = 0
    fit_phase: str = "validation"
    converged: bool = True
    schema: str = "papersoccer.flagship-calibration.v1"

    def __post_init__(self) -> None:
        _nonempty_string(self.bot_id, "calibration bot_id")
        if self.score_kind not in ("signed", "probability"):
            raise AnalysisError("calibration score_kind must be signed or probability")
        _finite(self.score_mean, "calibration score_mean")
        if _finite(self.score_scale, "calibration score_scale") <= 0.0:
            raise AnalysisError("calibration score_scale must be positive")
        _finite(self.intercept, "calibration intercept")
        _finite(self.slope, "calibration slope")
        _positive_integer(self.sample_count, "calibration sample_count")
        if (isinstance(self.iterations, bool) or
                not isinstance(self.iterations, int) or self.iterations < 0):
            raise AnalysisError("calibration iterations must be a nonnegative integer")
        for name in (
            "excluded_cached_continuations",
            "excluded_truncations",
            "excluded_invalid_depths",
        ):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise AnalysisError(f"calibration {name} must be a nonnegative integer")
        if self.fit_phase != "validation":
            raise AnalysisError("calibration mapping fit_phase must be validation")
        _boolean(self.converged, "calibration converged")
        if self.schema != "papersoccer.flagship-calibration.v1":
            raise AnalysisError("unsupported calibration mapping schema")

    def predict(self, oriented_score: float) -> float:
        checked = _finite(oriented_score, "oriented_score")
        standardized = (checked - self.score_mean) / self.score_scale
        return _sigmoid(self.intercept + self.slope * standardized)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationMapping":
        if not isinstance(value, Mapping):
            raise AnalysisError("calibration mapping must be an object")
        expected = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - expected
        if unknown:
            raise AnalysisError(f"unknown calibration mapping fields: {sorted(unknown)}")
        try:
            return cls(**value)
        except TypeError as error:
            raise AnalysisError(f"invalid calibration mapping: {error}") from error


def _calibration_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_phase: str | None,
) -> tuple[list[tuple[str, str, float, int]], dict[str, int]]:
    rows: list[tuple[str, str, float, int]] = []
    excluded = {"cached": 0, "truncated": 0, "invalid_depth": 0}
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise AnalysisError(f"observations[{index}] must be an object")
        phase = _nonempty_string(observation.get("phase"),
                                 f"observations[{index}].phase")
        if required_phase is not None and phase != required_phase:
            raise AnalysisError(
                f"observations[{index}] belongs to {phase}, expected {required_phase}"
            )
        cached = _boolean(
            observation.get("cached_continuation", False),
            f"observations[{index}].cached_continuation",
        )
        truncated = _boolean(
            observation.get("truncated", False),
            f"observations[{index}].truncated",
        )
        if cached:
            excluded["cached"] += 1
            continue
        if truncated:
            excluded["truncated"] += 1
            continue
        completed_depth = observation.get("completed_depth")
        if completed_depth is not None:
            if isinstance(completed_depth, bool) or not isinstance(completed_depth, int):
                raise AnalysisError(
                    f"observations[{index}].completed_depth must be an integer or null"
                )
            if completed_depth <= 0:
                excluded["invalid_depth"] += 1
                continue

        bot_id = _nonempty_string(observation.get("bot_id"),
                                  f"observations[{index}].bot_id")
        score_kind = observation.get("score_kind", "signed")
        if score_kind not in ("signed", "probability"):
            raise AnalysisError(
                f"observations[{index}].score_kind must be signed or probability"
            )
        perspective = observation.get("score_perspective", "player_one")
        if perspective not in ("player_one", "player_to_move"):
            raise AnalysisError(
                f"observations[{index}].score_perspective is unsupported"
            )
        raw_score = _finite(observation.get("raw_score"),
                            f"observations[{index}].raw_score")
        if score_kind == "probability" and not 0.0 <= raw_score <= 1.0:
            raise AnalysisError(f"observations[{index}].raw_score is not a probability")
        if perspective == "player_one":
            player = observation.get("player_to_move")
            if score_kind == "probability":
                raw_score = orient_player_one_probability(raw_score, player)
            else:
                raw_score = orient_player_one_score(raw_score, player)
        outcome = _binary(observation.get("outcome"),
                          f"observations[{index}].outcome")
        rows.append((bot_id, score_kind, raw_score, outcome))
    return rows, excluded


def _logistic_log_likelihood(scores: Sequence[float], outcomes: Sequence[int],
                             intercept: float, slope: float) -> float:
    return sum(
        outcome * _log_sigmoid(intercept + slope * score)
        + (1 - outcome) * _log_sigmoid(-(intercept + slope * score))
        for score, outcome in zip(scores, outcomes, strict=True)
    )


def fit_logistic_calibration(
    observations: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> CalibrationMapping:
    """Fit one bot's logistic mapping, accepting validation rows only."""

    if phase != "validation":
        raise AnalysisError("calibration fitting is permitted only on validation data")
    rows, excluded = _calibration_rows(observations, required_phase="validation")
    if len(rows) < 3:
        raise AnalysisError("at least three valid calibration observations are required")
    bot_ids = {row[0] for row in rows}
    score_kinds = {row[1] for row in rows}
    if len(bot_ids) != 1:
        raise AnalysisError("fit separate calibration mappings for each bot")
    if len(score_kinds) != 1:
        raise AnalysisError("one calibration mapping cannot mix score kinds")
    scores = [row[2] for row in rows]
    outcomes = [row[3] for row in rows]
    if len(set(outcomes)) != 2:
        raise SeparationError("calibration outcomes contain only one class")
    mean = sum(scores) / len(scores)
    scale = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
    if scale <= 1e-15:
        raise AnalysisError("calibration scores have zero variance")
    standardized = [(score - mean) / scale for score in scores]
    zeros = [score for score, outcome in zip(standardized, outcomes, strict=True)
             if outcome == 0]
    ones = [score for score, outcome in zip(standardized, outcomes, strict=True)
            if outcome == 1]
    if max(zeros) < min(ones) or max(ones) < min(zeros):
        raise SeparationError("calibration scores completely separate the outcomes")
    tolerance, max_iterations = _convergence_parameters(
        tolerance, max_iterations
    )

    prevalence = sum(outcomes) / len(outcomes)
    intercept = math.log(prevalence / (1.0 - prevalence))
    slope = 0.0
    for iteration in range(max_iterations + 1):
        gradient_intercept = 0.0
        gradient_slope = 0.0
        information_00 = 0.0
        information_01 = 0.0
        information_11 = 0.0
        for score, outcome in zip(standardized, outcomes, strict=True):
            probability = _sigmoid(intercept + slope * score)
            residual = outcome - probability
            weight = probability * (1.0 - probability)
            gradient_intercept += residual
            gradient_slope += residual * score
            information_00 += weight
            information_01 += weight * score
            information_11 += weight * score * score
        if max(abs(gradient_intercept), abs(gradient_slope)) <= tolerance:
            return CalibrationMapping(
                bot_id=next(iter(bot_ids)),
                score_kind=next(iter(score_kinds)),
                score_mean=mean,
                score_scale=scale,
                intercept=intercept,
                slope=slope,
                sample_count=len(rows),
                iterations=iteration,
                excluded_cached_continuations=excluded["cached"],
                excluded_truncations=excluded["truncated"],
                excluded_invalid_depths=excluded["invalid_depth"],
            )
        if iteration == max_iterations:
            break
        determinant = information_00 * information_11 - information_01 ** 2
        if determinant <= 1e-15:
            raise SeparationError("calibration information matrix is singular")
        step_intercept = (
            information_11 * gradient_intercept
            - information_01 * gradient_slope
        ) / determinant
        step_slope = (
            -information_01 * gradient_intercept
            + information_00 * gradient_slope
        ) / determinant
        current = _logistic_log_likelihood(
            standardized, outcomes, intercept, slope
        )
        multiplier = 1.0
        accepted = False
        for _ in range(60):
            next_intercept = intercept + multiplier * step_intercept
            next_slope = slope + multiplier * step_slope
            candidate = _logistic_log_likelihood(
                standardized, outcomes, next_intercept, next_slope
            )
            if candidate >= current - 1e-12:
                intercept = next_intercept
                slope = next_slope
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or max(abs(intercept), abs(slope)) > 50.0:
            raise SeparationError("calibration coefficients diverged")
    raise ConvergenceError(
        f"logistic calibration did not converge within {max_iterations} iterations"
    )


def apply_calibration(
    mapping: CalibrationMapping | Mapping[str, Any],
    oriented_scores: Sequence[float],
) -> list[float]:
    """Apply a previously frozen mapping without fitting or changing it."""

    resolved = (mapping if isinstance(mapping, CalibrationMapping)
                else CalibrationMapping.from_dict(mapping))
    if resolved.fit_phase != "validation" or not resolved.converged:
        raise AnalysisError("calibration mapping is not a frozen validation fit")
    return [resolved.predict(score) for score in oriented_scores]


def calibration_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = RELIABILITY_BINS,
    log_loss_epsilon: float = 1e-15,
) -> dict[str, Any]:
    """Return Brier score, log loss, and equal-width reliability bins."""

    if len(probabilities) != len(outcomes) or not probabilities:
        raise AnalysisError("calibration metrics require equal non-empty samples")
    if bins != RELIABILITY_BINS:
        raise AnalysisError("the flagship calibration contract requires ten bins")
    if not 0.0 < log_loss_epsilon < 0.5:
        raise AnalysisError("invalid log-loss clipping epsilon")
    checked_probabilities = []
    checked_outcomes = []
    for index, (probability, outcome) in enumerate(
        zip(probabilities, outcomes, strict=True)
    ):
        checked = _finite(probability, f"probabilities[{index}]")
        if not 0.0 <= checked <= 1.0:
            raise AnalysisError(f"probabilities[{index}] must be in [0,1]")
        checked_probabilities.append(checked)
        checked_outcomes.append(_binary(outcome, f"outcomes[{index}]"))
    count = len(checked_outcomes)
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(
            checked_probabilities, checked_outcomes, strict=True
        )
    ) / count
    log_loss = 0.0
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, outcome in zip(
        checked_probabilities, checked_outcomes, strict=True
    ):
        clipped = min(1.0 - log_loss_epsilon, max(log_loss_epsilon, probability))
        log_loss -= outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)
        grouped[min(bins - 1, int(probability * bins))].append((probability, outcome))
    reliability = []
    for index, values in enumerate(grouped):
        observed_successes = sum(value[1] for value in values)
        reliability.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "upper_inclusive": index == bins - 1,
            "count": len(values),
            "mean_prediction": (
                sum(value[0] for value in values) / len(values) if values else None
            ),
            "observed_frequency": (
                observed_successes / len(values) if values else None
            ),
        })
    return {
        "samples": count,
        "brier_score": brier,
        "log_loss": log_loss / count,
        "reliability_bins": reliability,
    }


def pair_clustered_calibration_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    pair_cluster_ids: Sequence[str],
    stratum_ids: Sequence[str],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    bins: int = RELIABILITY_BINS,
    minimum_bin_successful_resamples: int = 1_000,
    log_loss_epsilon: float = 1e-15,
) -> dict[str, Any]:
    """Score calibration with whole-pair, within-stratum bootstrap intervals.

    All decisions from both games in a color-swapped pair share one cluster.
    Resampling those clusters preserves arbitrary within-game and within-pair
    dependence instead of pretending decision-level Bernoulli observations are
    independent.
    """

    point = calibration_metrics(
        probabilities, outcomes, bins=bins, log_loss_epsilon=log_loss_epsilon
    )
    count = len(probabilities)
    if len(pair_cluster_ids) != count or len(stratum_ids) != count:
        raise AnalysisError("calibration cluster columns must align with predictions")
    checked_seed, checked_resamples = _bootstrap_parameters(seed, resamples)
    if (isinstance(minimum_bin_successful_resamples, bool)
            or not isinstance(minimum_bin_successful_resamples, int)
            or minimum_bin_successful_resamples <= 0
            or minimum_bin_successful_resamples > checked_resamples):
        raise AnalysisError("invalid minimum successful calibration-bin resamples")

    checked_probabilities = [
        _finite(value, f"probabilities[{index}]")
        for index, value in enumerate(probabilities)
    ]
    checked_outcomes = [
        _binary(value, f"outcomes[{index}]")
        for index, value in enumerate(outcomes)
    ]
    clusters: dict[str, dict[str, Any]] = {}
    strata: dict[str, set[str]] = defaultdict(set)
    for index, (cluster_raw, stratum_raw) in enumerate(
        zip(pair_cluster_ids, stratum_ids, strict=True)
    ):
        cluster = _nonempty_string(cluster_raw, f"pair_cluster_ids[{index}]")
        stratum = _nonempty_string(stratum_raw, f"stratum_ids[{index}]")
        stats = clusters.setdefault(cluster, {
            "stratum": stratum,
            "count": 0,
            "brier_sum": 0.0,
            "log_loss_sum": 0.0,
            "bins": {},
        })
        if stats["stratum"] != stratum:
            raise AnalysisError("one calibration pair cluster crosses strata")
        probability = checked_probabilities[index]
        if not 0.0 <= probability <= 1.0:
            raise AnalysisError(f"probabilities[{index}] must be in [0,1]")
        outcome = checked_outcomes[index]
        clipped = min(1.0 - log_loss_epsilon, max(log_loss_epsilon, probability))
        stats["count"] += 1
        stats["brier_sum"] += (probability - outcome) ** 2
        stats["log_loss_sum"] -= (
            outcome * math.log(clipped) + (1 - outcome) * math.log(1.0 - clipped)
        )
        bin_index = min(bins - 1, int(probability * bins))
        bin_stats = stats["bins"].setdefault(bin_index, [0, 0])
        bin_stats[0] += 1
        bin_stats[1] += outcome
        strata[stratum].add(cluster)
    if not clusters or not strata:
        raise AnalysisError("calibration bootstrap requires non-empty pair clusters")

    for bin_value in point["reliability_bins"]:
        bin_index = bin_value["bin"]
        bin_value["pair_clusters"] = sum(
            bin_index in stats["bins"] for stats in clusters.values()
        )

    ordered_strata = [tuple(sorted(strata[key])) for key in sorted(strata)]
    generator = random.Random(checked_seed)
    brier_samples: list[float] = []
    log_loss_samples: list[float] = []
    observed_samples: list[list[float]] = [[] for _ in range(bins)]
    for _ in range(checked_resamples):
        replicate_count = 0
        replicate_brier = 0.0
        replicate_log_loss = 0.0
        bin_counts = [0] * bins
        bin_successes = [0] * bins
        for cluster_ids in ordered_strata:
            for _ in cluster_ids:
                stats = clusters[cluster_ids[generator.randrange(len(cluster_ids))]]
                replicate_count += stats["count"]
                replicate_brier += stats["brier_sum"]
                replicate_log_loss += stats["log_loss_sum"]
                for bin_index, (bin_count, bin_success) in stats["bins"].items():
                    bin_counts[bin_index] += bin_count
                    bin_successes[bin_index] += bin_success
        if replicate_count <= 0:
            raise AnalysisError("calibration bootstrap produced an empty replicate")
        brier_samples.append(replicate_brier / replicate_count)
        log_loss_samples.append(replicate_log_loss / replicate_count)
        for bin_index in range(bins):
            if bin_counts[bin_index] > 0:
                observed_samples[bin_index].append(
                    bin_successes[bin_index] / bin_counts[bin_index]
                )

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {
            "lower": _nearest_rank(values, 0.025),
            "upper": _nearest_rank(values, 0.975),
        }

    point["pair_clusters"] = len(clusters)
    point["pair_cluster_bootstrap_95"] = {
        "method": "pair_cluster_percentile_stratified",
        "seed": str(checked_seed),
        "resamples": checked_resamples,
        "successful_resamples": checked_resamples,
        "confidence": 0.95,
        "stratify_by": "matchup_and_opening_depth",
        "brier_score": interval(brier_samples),
        "log_loss": interval(log_loss_samples),
    }
    for bin_value, values in zip(
        point["reliability_bins"], observed_samples, strict=True
    ):
        bin_value["observed_frequency_pair_bootstrap_95"] = (
            {
                "method": "pair_cluster_percentile_stratified",
                "confidence": 0.95,
                "resamples": checked_resamples,
                "successful_resamples": len(values),
                **interval(values),
            }
            if len(values) >= minimum_bin_successful_resamples else None
        )
        bin_value["bootstrap_successful_resamples"] = len(values)
    return point


def evaluate_calibration(
    mapping: CalibrationMapping | Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    """Apply a frozen mapping and score valid, fresh observations in a phase."""

    resolved = (mapping if isinstance(mapping, CalibrationMapping)
                else CalibrationMapping.from_dict(mapping))
    rows, excluded = _calibration_rows(observations, required_phase=phase)
    if any(row[0] != resolved.bot_id for row in rows):
        raise AnalysisError("calibration observations do not match mapping bot_id")
    if any(row[1] != resolved.score_kind for row in rows):
        raise AnalysisError("calibration observations do not match mapping score_kind")
    probabilities = [resolved.predict(row[2]) for row in rows]
    outcomes = [row[3] for row in rows]
    metrics = calibration_metrics(probabilities, outcomes)
    metrics["phase"] = phase
    metrics["excluded"] = {
        "cached_continuations": excluded["cached"],
        "truncations": excluded["truncated"],
        "invalid_depths": excluded["invalid_depth"],
    }
    return metrics


def classify_pareto(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Classify constrained and unconstrained validation Pareto frontiers."""

    if not points:
        raise AnalysisError("at least one Pareto point is required")
    identifiers: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        identifier = _nonempty_string(point.get("id"), f"points[{index}].id")
        if identifier in identifiers:
            raise AnalysisError("Pareto point IDs must be unique")
        identifiers.add(identifier)
        phases = point.get("strength_phases")
        if not isinstance(phases, (list, tuple)) or not phases:
            raise AnalysisError(f"points[{index}].strength_phases must be non-empty")
        if not set(phases) <= {"development", "validation"}:
            raise AnalysisError("test strength cannot enter the validation Pareto frontier")
        if point.get("latency_phase") != "validation":
            raise AnalysisError("Pareto latency must come from validation")
        strength = _finite(point.get("strength"), f"points[{index}].strength")
        latency = _finite(point.get("p95_ms"), f"points[{index}].p95_ms")
        if not 0.0 <= strength <= 1.0:
            raise AnalysisError("Pareto strength must be a mean pair score in [0,1]")
        if latency < 0.0:
            raise AnalysisError("Pareto p95 latency cannot be negative")
        if not isinstance(point.get("gate_eligible"), bool):
            raise AnalysisError(f"points[{index}].gate_eligible must be boolean")
        normalized_point = dict(point)
        normalized_point["strength"] = strength
        normalized_point["p95_ms"] = latency
        normalized.append(normalized_point)
    unconstrained = {
        point["id"]: point for point in studylib.pareto_frontier(normalized)
    }
    eligible = [point for point in normalized if point["gate_eligible"]]
    constrained = {
        point["id"]: point for point in studylib.pareto_frontier(eligible)
    }
    result: list[dict[str, Any]] = []
    for point in normalized:
        identifier = point["id"]
        constrained_point = constrained.get(identifier)
        annotated = dict(point)
        annotated["unconstrained_pareto_optimal"] = \
            unconstrained[identifier]["pareto_optimal"]
        annotated["unconstrained_dominated_by"] = \
            unconstrained[identifier]["dominated_by"]
        annotated["constrained_pareto_optimal"] = bool(
            constrained_point and constrained_point["pareto_optimal"]
        )
        annotated["constrained_dominated_by"] = (
            constrained_point["dominated_by"] if constrained_point else []
        )
        # The unsuffixed fields deliberately mean the preregistered <=50 ms
        # frontier so downstream charts cannot accidentally promote an
        # ineligible configuration into a constrained claim.
        annotated["pareto_optimal"] = annotated["constrained_pareto_optimal"]
        annotated["dominated_by"] = annotated["constrained_dominated_by"]
        result.append(annotated)
    return result


__all__ = [
    "AnalysisError",
    "BOOTSTRAP_RESAMPLES",
    "BradleyTerryFit",
    "CalibrationMapping",
    "ConvergenceError",
    "DisconnectedComparisonError",
    "PairedComparison",
    "SeparationError",
    "apply_calibration",
    "bootstrap_bradley_terry",
    "calibration_metrics",
    "classify_pareto",
    "depth_stratified_pair_bootstrap",
    "evaluate_calibration",
    "fit_bradley_terry",
    "fit_logistic_calibration",
    "orient_player_one_probability",
    "orient_player_one_score",
    "pair_clustered_calibration_metrics",
    "pair_score",
    "summarize_pair_outcomes",
]
