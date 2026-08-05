"""Publish a small, deterministic website snapshot of the flagship study.

The checked-in benchmark artifacts deliberately contain much more information
than the public website needs.  This module validates their publication
contract and exports only bot-performance results.  In particular, the output
does not contain games, execution environments, timestamps, or hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import sys
from collections.abc import Mapping
from typing import Any


SUMMARY_SCHEMA = "papersoccer.benchmark-summary.v1"
MANIFEST_SCHEMA = "papersoccer.flagship-study-manifest.v2"
SELECTION_SCHEMA = "papersoccer.flagship-study-selection.v1"
CURATED_SCHEMA = "papersoccer.flagship-study-curated.v1"

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/flagship_study/manifest.json"
DEFAULT_SELECTION = REPOSITORY_ROOT / "benchmarks/flagship_study/selection_lock.json"
DEFAULT_TEST_DATA = REPOSITORY_ROOT / "benchmarks/flagship_study/data/test.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "web/benchmarks/benchmark-results.js"

REPOSITORY_WEB = "https://github.com/Lecorbio/paper-soccer-strategy-engine"
SHORT_LABELS = {
    "mcts": "Tactical MCTS",
    "alpha_beta": "Hand alpha-beta",
    "jacek_inspired": "Neural alpha-beta",
    "rank5_derived": "Rank5Derived",
}
TUNABLE_FAMILIES = ("mcts", "alpha_beta", "jacek_inspired")
RANK5_REFERENCE_DEFINITION = "defined common-opponent reference level"


class SummaryError(ValueError):
    """Raised when source artifacts are not safe to publish."""


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise SummaryError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError(f"{where} must be a non-empty string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise SummaryError(f"{where} must be a boolean")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SummaryError(f"{where} must be an integer at least {minimum}")
    return value


def _number(value: Any, where: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{where} must be a number")
    if not math.isfinite(value):
        raise SummaryError(f"{where} must be finite")
    return value


def _interval(
    value: Any,
    where: str,
    estimate: float | int,
    *,
    probability: bool = False,
) -> tuple[float | int, float | int]:
    interval = _object(value, where)
    lower = _number(interval.get("lower"), f"{where}.lower")
    upper = _number(interval.get("upper"), f"{where}.upper")
    if lower > estimate or estimate > upper:
        raise SummaryError(f"{where} must contain its estimate")
    if probability and (lower < 0 or upper > 1):
        raise SummaryError(f"{where} must stay within [0, 1]")
    return lower, upper


def _read_json(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"could not read JSON {path}: {error}") from error
    return _object(value, str(path))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SummaryError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _configurations(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    configurations: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(_array(manifest.get("configurations"), "configurations")):
        config = _object(value, f"configurations[{index}]")
        config_id = _string(config.get("id"), f"configurations[{index}].id")
        if config_id in configurations:
            raise SummaryError(f"duplicate configuration id: {config_id}")
        configurations[config_id] = config
    return configurations


def _selected_ids(selection: Mapping[str, Any]) -> dict[str, str]:
    selected = _object(selection.get("selected_configurations"), "selected_configurations")
    if set(selected) != set(TUNABLE_FAMILIES):
        raise SummaryError("selected_configurations must select every tunable family exactly once")
    result = {
        family: _string(selected.get(family), f"selected_configurations.{family}")
        for family in TUNABLE_FAMILIES
    }
    result["rank5_derived"] = _string(
        selection.get("fixed_rank5_configuration"), "fixed_rank5_configuration"
    )
    if len(set(result.values())) != len(result):
        raise SummaryError("selected entrant ids must be unique")
    return result


def _budget(config: Mapping[str, Any]) -> int:
    settings = _object(config.get("settings"), f"configuration {config.get('id')} settings")
    value = settings.get("iterations", settings.get("max_nodes"))
    return _integer(value, f"configuration {config.get('id')} budget", minimum=1)


def _validate_provenance(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    manifest_hash: str,
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SummaryError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise SummaryError(f"unsupported selection schema: {selection.get('schema_version')!r}")
    if test.get("schema_version") != CURATED_SCHEMA:
        raise SummaryError(f"unsupported test-data schema: {test.get('schema_version')!r}")
    for name, artifact in (("selection lock", selection), ("test data", test)):
        if artifact.get("manifest_sha256") != manifest_hash:
            raise SummaryError(f"{name} does not match the manifest provenance")
    if selection.get("source_phase") != "validation" or not _boolean(
        selection.get("test_authorized"), "test_authorized"
    ):
        raise SummaryError("selection lock must authorize test from validation")
    if test.get("phase") != "test":
        raise SummaryError("curated test data must have phase 'test'")
    study = _object(manifest.get("study"), "study")
    if not _boolean(study.get("frozen"), "study.frozen"):
        raise SummaryError("only a frozen study may be published")


def _validate_completeness(test: Mapping[str, Any]) -> tuple[int, int, list[int]]:
    if not _boolean(test.get("analysis_complete"), "analysis_complete"):
        raise SummaryError("test analysis is not complete")
    completeness = _object(test.get("completeness"), "completeness")
    expected_games = _integer(completeness.get("expected_games"), "expected_games", minimum=1)
    completed_games = _integer(completeness.get("completed_games"), "completed_games", minimum=1)
    unique_games = _integer(completeness.get("unique_game_ids"), "unique_game_ids", minimum=1)
    if expected_games != completed_games or completed_games != unique_games:
        raise SummaryError("test game counts are incomplete")
    if not _boolean(completeness.get("operationally_valid"), "operationally_valid"):
        raise SummaryError("test results are not operationally valid")
    if _integer(completeness.get("truncations"), "test truncations") != 0:
        raise SummaryError("test results contain truncations")
    binary_games = _array(test.get("binary_games"), "binary_games")
    if len(binary_games) != completed_games:
        raise SummaryError("binary game count does not match completeness")

    sizes = _object(test.get("sample_sizes"), "sample_sizes")
    games = _integer(sizes.get("games"), "sample_sizes.games", minimum=1)
    pairs = _integer(sizes.get("pairs"), "sample_sizes.pairs", minimum=1)
    depths = [_integer(value, "opening depth", minimum=1) for value in _array(
        sizes.get("opening_depths"), "sample_sizes.opening_depths"
    )]
    if games != completed_games or games != 2 * pairs or not depths:
        raise SummaryError("test sample sizes are inconsistent")
    return games, pairs, depths


def _validate_selected(
    configurations: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, str],
    test: Mapping[str, Any],
) -> None:
    test_configs = _object(test.get("configurations"), "test configurations")
    for family, config_id in selected.items():
        config = configurations.get(config_id)
        if config is None:
            raise SummaryError(f"selected configuration is absent from manifest: {config_id}")
        if config.get("family") != family:
            raise SummaryError(f"selected configuration {config_id} is not from family {family}")
        if config_id not in test_configs:
            raise SummaryError(f"selected configuration is absent from test analysis: {config_id}")
    if set(test_configs) != set(selected.values()):
        raise SummaryError("test analysis must contain exactly the four selected entrants")


def _bradley_terry(test: Mapping[str, Any], entrant_ids: set[str]) -> dict[str, dict[str, Any]]:
    bt = _object(test.get("bradley_terry"), "bradley_terry")
    point_fit = _object(bt.get("point_fit"), "bradley_terry.point_fit")
    if not _boolean(point_fit.get("converged"), "bradley_terry.point_fit.converged"):
        raise SummaryError("Bradley-Terry point fit did not converge")
    if set(_array(point_fit.get("bot_ids"), "bradley_terry bot ids")) != entrant_ids:
        raise SummaryError("Bradley-Terry fit does not cover every selected entrant")
    abilities = _object(point_fit.get("abilities"), "bradley_terry.point_fit.abilities")
    if set(abilities) != entrant_ids:
        raise SummaryError("Bradley-Terry abilities do not cover every selected entrant")
    resamples = _integer(bt.get("resamples"), "bradley_terry.resamples", minimum=1)
    successful = _integer(
        bt.get("successful_resamples"), "bradley_terry.successful_resamples", minimum=1
    )
    if successful != resamples:
        raise SummaryError("Bradley-Terry bootstrap is incomplete")

    intervals = _object(bt.get("intervals"), "bradley_terry.intervals")
    if set(intervals) != entrant_ids:
        raise SummaryError("Bradley-Terry intervals do not cover every selected entrant")
    result: dict[str, dict[str, Any]] = {}
    for config_id in sorted(entrant_ids):
        interval = _object(intervals[config_id], f"Bradley-Terry interval {config_id}")
        estimate = _number(interval.get("estimate"), f"Bradley-Terry estimate {config_id}")
        ability = _number(abilities[config_id], f"Bradley-Terry ability {config_id}")
        if estimate != ability:
            raise SummaryError(
                f"Bradley-Terry interval estimate does not match point-fit ability: {config_id}"
            )
        lower, upper = _interval(interval, f"Bradley-Terry interval {config_id}", estimate)
        result[config_id] = {"estimate": estimate, "lower": lower, "upper": upper}
    return result


def _validation_rows(
    manifest: Mapping[str, Any],
    configurations: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    selected_ids: set[str],
    fixed_rank5_id: str,
) -> list[dict[str, Any]]:
    pareto = _array(selection.get("validation_pareto"), "validation_pareto")
    expected_candidates = {
        _string(config_id, "candidate grid id")
        for values in _object(manifest.get("candidate_grids"), "candidate_grids").values()
        for config_id in _array(values, "candidate grid")
    }
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(pareto):
        row = _object(value, f"validation_pareto[{index}]")
        config_id = _string(row.get("id"), f"validation_pareto[{index}].id")
        if config_id in rows_by_id:
            raise SummaryError(f"duplicate validation candidate: {config_id}")
        rows_by_id[config_id] = row
    if set(rows_by_id) != expected_candidates or not selected_ids <= set(rows_by_id):
        raise SummaryError("validation Pareto rows do not match the manifest candidate grid")

    candidates = []
    selected_marked = set()
    gate_ms = _number(
        _object(manifest.get("latency_protocol"), "latency_protocol").get("gate_ms"),
        "latency gate",
    )
    for config_id in sorted(rows_by_id):
        row = rows_by_id[config_id]
        config = configurations.get(config_id)
        if config is None:
            raise SummaryError(f"validation candidate is absent from manifest: {config_id}")
        family = _string(row.get("family"), f"validation candidate {config_id} family")
        if config.get("family") != family:
            raise SummaryError(f"validation candidate family mismatch: {config_id}")
        strength = _number(row.get("validation_strength"), f"{config_id} validation strength")
        definition = row.get("strength_definition")
        reference = config_id == fixed_rank5_id
        if reference:
            if definition != RANK5_REFERENCE_DEFINITION:
                raise SummaryError(
                    "the fixed Rank5 candidate must remain the defined validation reference"
                )
        elif definition is not None:
            raise SummaryError(
                f"only the fixed Rank5 candidate may define a validation reference: {config_id}"
            )
        interval_value = row.get("validation_strength_pair_bootstrap_95")
        pairs_value = row.get("validation_strength_pairs")
        if reference:
            if strength != 0.5 or interval_value is not None or pairs_value is not None:
                raise SummaryError("Rank5 validation strength must remain a defined 50% reference")
            lower = upper = pairs = None
        else:
            lower, upper = _interval(
                interval_value, f"{config_id} validation interval", strength, probability=True
            )
            pairs = _integer(pairs_value, f"{config_id} validation pairs", minimum=1)
        latency = _number(row.get("validation_p95_ms"), f"{config_id} p95 latency")
        decisions = _integer(
            row.get("validation_latency_decisions"), f"{config_id} latency decisions", minimum=1
        )
        eligible = _boolean(row.get("gate_eligible"), f"{config_id} gate eligibility")
        if eligible != (latency <= gate_ms):
            raise SummaryError(f"{config_id} gate eligibility disagrees with its p95 latency")
        selected = _boolean(row.get("selected"), f"{config_id} selected")
        if selected:
            selected_marked.add(config_id)
        fixed = _boolean(row.get("fixed"), f"{config_id} fixed")
        if fixed != reference:
            raise SummaryError(
                f"only the fixed Rank5 candidate may have fixed validation status: {config_id}"
            )
        candidates.append({
            "id": config_id,
            "family": family,
            "label": _string(config.get("public_label"), f"{config_id} public label"),
            "budget": _budget(config),
            "strength": strength,
            "strengthIsReference": reference,
            "strengthLower": lower,
            "strengthUpper": upper,
            "pairs": pairs,
            "p95LatencyMs": latency,
            "latencyDecisions": decisions,
            "gateEligible": eligible,
            "selected": selected,
            "fixed": fixed,
            "paretoOptimal": _boolean(
                row.get("constrained_pareto_optimal"), f"{config_id} Pareto status"
            ),
        })
    if selected_marked != selected_ids:
        raise SummaryError("validation Pareto selection does not match the selection lock")
    return candidates


def _matchups(
    test: Mapping[str, Any],
    entrant_ids: set[str],
    *,
    expected_games: int,
    expected_pairs: int,
) -> list[dict[str, Any]]:
    source = _object(test.get("matchups"), "matchups")
    expected_pairings = {frozenset(pair) for pair in itertools.combinations(entrant_ids, 2)}
    seen_pairs: set[frozenset[str]] = set()
    result = []
    for matchup_id in sorted(source):
        value = _object(source[matchup_id], f"matchup {matchup_id}")
        left_id = _string(value.get("left_config_id"), f"{matchup_id}.left_config_id")
        right_id = _string(value.get("right_config_id"), f"{matchup_id}.right_config_id")
        pair = frozenset((left_id, right_id))
        if len(pair) != 2 or not pair <= entrant_ids or pair in seen_pairs:
            raise SummaryError(f"invalid or duplicate entrant pairing in {matchup_id}")
        seen_pairs.add(pair)
        score = _number(value.get("mean_pair_score"), f"{matchup_id} score")
        lower, upper = _interval(
            value.get("pair_bootstrap_95"), f"{matchup_id} interval", score, probability=True
        )
        pairs = _integer(value.get("pairs"), f"{matchup_id} pairs", minimum=1)
        games = _integer(value.get("games"), f"{matchup_id} games", minimum=1)
        if games != 2 * pairs:
            raise SummaryError(f"{matchup_id} must contain two games per pair")
        if _integer(value.get("truncations"), f"{matchup_id} truncations") != 0:
            raise SummaryError(f"{matchup_id} contains truncations")
        left_wins = _integer(value.get("left_wins"), f"{matchup_id} left wins")
        right_wins = _integer(value.get("right_wins"), f"{matchup_id} right wins")
        if left_wins + right_wins != games:
            raise SummaryError(f"{matchup_id} win counts are incomplete")
        if not math.isclose(score, left_wins / games, rel_tol=0.0, abs_tol=1e-12):
            raise SummaryError(f"{matchup_id} score does not match its decisive-game wins")
        conclusion = _object(value.get("conclusion"), f"{matchup_id} conclusion")
        classification = _string(
            conclusion.get("classification"), f"{matchup_id} classification"
        )
        stronger_id = conclusion.get("stronger_config_id")
        if classification == "statistically_unresolved":
            if stronger_id is not None or not (lower <= 0.5 <= upper):
                raise SummaryError(f"{matchup_id} unresolved conclusion is inconsistent")
        elif classification == "stronger":
            stronger_id = _string(stronger_id, f"{matchup_id} stronger id")
            inferred = left_id if lower > 0.5 else right_id if upper < 0.5 else None
            if inferred != stronger_id:
                raise SummaryError(f"{matchup_id} stronger conclusion is inconsistent")
        else:
            raise SummaryError(f"unsupported matchup classification: {classification}")
        result.append({
            "id": matchup_id,
            "leftId": left_id,
            "rightId": right_id,
            "leftScore": score,
            "leftScoreLower": lower,
            "leftScoreUpper": upper,
            "pairs": pairs,
            "games": games,
            "classification": classification,
            "strongerId": stronger_id,
        })
    if seen_pairs != expected_pairings:
        raise SummaryError("test analysis must contain all six selected-entrant matchups")
    if (
        sum(row["games"] for row in result) != expected_games
        or sum(row["pairs"] for row in result) != expected_pairs
    ):
        raise SummaryError("matchup game and pair counts do not sum to the study totals")
    return result


def _calibration(
    test: Mapping[str, Any],
    configurations: Mapping[str, Mapping[str, Any]],
    entrant_ids: set[str],
) -> list[dict[str, Any]]:
    source = _object(test.get("calibration"), "calibration")
    if set(source) != entrant_ids:
        raise SummaryError("calibration analysis must cover every selected entrant")
    result = []
    for config_id in sorted(source):
        value = _object(source[config_id], f"calibration {config_id}")
        excluded = _object(value.get("excluded"), f"calibration {config_id} exclusions")
        if _integer(excluded.get("truncations"), f"calibration {config_id} truncations") != 0:
            raise SummaryError(f"calibration {config_id} contains truncations")
        brier_score = _number(value.get("brier_score"), f"{config_id} Brier score")
        if brier_score < 0 or brier_score > 1:
            raise SummaryError(f"{config_id} Brier score must stay within [0, 1]")
        log_loss = _number(value.get("log_loss"), f"{config_id} log loss")
        if log_loss < 0:
            raise SummaryError(f"{config_id} log loss must be nonnegative")
        samples = _integer(value.get("samples"), f"{config_id} calibration samples", minimum=1)
        decision_count = _integer(
            value.get("decision_count"), f"{config_id} calibration decisions", minimum=1
        )
        if samples > decision_count:
            raise SummaryError(f"{config_id} calibration samples exceed its decision count")
        result.append({
            "id": config_id,
            "label": _string(configurations[config_id].get("public_label"), f"{config_id} label"),
            "brierScore": brier_score,
            "logLoss": log_loss,
            "samples": samples,
            "decisionCount": decision_count,
        })
    return result


def _headline(
    entrants: list[dict[str, Any]],
    matchups: list[dict[str, Any]],
    neural_id: str,
    rank5_id: str,
) -> str:
    top = entrants[0]
    neural_rank5 = next(
        matchup
        for matchup in matchups
        if {matchup["leftId"], matchup["rightId"]} == {neural_id, rank5_id}
    )
    neural_label = SHORT_LABELS["jacek_inspired"]
    rank5_label = SHORT_LABELS["rank5_derived"]

    if neural_rank5["classification"] == "statistically_unresolved":
        if top["id"] == neural_id:
            return (
                f"{neural_label} has the highest strength estimate, while its matchup "
                f"with {rank5_label} remains statistically unresolved."
            )
        if top["id"] == rank5_id:
            return (
                f"{rank5_label} has the highest strength estimate, while its matchup "
                f"with {neural_label} remains statistically unresolved."
            )
        return (
            f"{top['shortLabel']} has the highest strength estimate; the matchup between "
            f"{neural_label} and {rank5_label} remains statistically unresolved."
        )

    stronger_id = neural_rank5["strongerId"]
    stronger_label = neural_label if stronger_id == neural_id else rank5_label
    weaker_label = rank5_label if stronger_id == neural_id else neural_label
    return (
        f"{top['shortLabel']} has the highest strength estimate; {stronger_label} is "
        f"stronger than {weaker_label} in their direct matchup."
    )


def build_summary(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    *,
    manifest_hash: str,
) -> dict[str, Any]:
    """Validate source artifacts and return the website-safe summary."""

    _validate_provenance(manifest, selection, test, manifest_hash)
    configurations = _configurations(manifest)
    selected = _selected_ids(selection)
    _validate_selected(configurations, selected, test)
    entrant_ids = set(selected.values())
    games, pairs, opening_depths = _validate_completeness(test)
    bt = _bradley_terry(test, entrant_ids)
    candidates = _validation_rows(
        manifest,
        configurations,
        selection,
        entrant_ids,
        selected["rank5_derived"],
    )
    candidates_by_id = {row["id"]: row for row in candidates}
    matchup_rows = _matchups(
        test,
        entrant_ids,
        expected_games=games,
        expected_pairs=pairs,
    )
    calibration_rows = _calibration(test, configurations, entrant_ids)

    entrants = []
    for config_id in sorted(entrant_ids, key=lambda item: (-bt[item]["estimate"], item)):
        config = configurations[config_id]
        family = _string(config.get("family"), f"{config_id} family")
        validation = candidates_by_id[config_id]
        entrants.append({
            "id": config_id,
            "family": family,
            "label": _string(config.get("public_label"), f"{config_id} label"),
            "shortLabel": SHORT_LABELS[family],
            "bradleyTerry": bt[config_id],
            "validation": {
                key: validation[key]
                for key in (
                    "strength", "strengthIsReference", "strengthLower", "strengthUpper",
                    "pairs", "p95LatencyMs", "latencyDecisions", "gateEligible",
                    "selected", "fixed", "paretoOptimal",
                )
            },
        })

    study = _object(manifest.get("study"), "study")
    latency = _object(manifest.get("latency_protocol"), "latency_protocol")
    return {
        "schema": SUMMARY_SCHEMA,
        "study": {
            "id": _string(study.get("id"), "study.id"),
            "title": _string(study.get("title"), "study.title"),
            "headline": _headline(
                entrants,
                matchup_rows,
                selected["jacek_inspired"],
                selected["rank5_derived"],
            ),
            "entrantCount": len(entrants),
            "games": games,
            "pairs": pairs,
            "openingDepths": opening_depths,
            "latencyGateMs": _number(latency.get("gate_ms"), "latency gate"),
        },
        "entrants": entrants,
        "matchups": matchup_rows,
        "validationCandidates": candidates,
        "calibration": calibration_rows,
        "caveats": {
            "rank5": _string(study.get("rank5_disclaimer"), "study.rank5_disclaimer"),
            "validationReference": (
                "Rank5Derived's 50% validation strength is a defined common-opponent "
                "reference level, not an independently observed score."
            ),
            "relativeStrength": (
                "Bradley-Terry strengths are relative within these four entrants; "
                "their zero point is not an absolute playing-strength scale."
            ),
            "latency": (
                "Latency is native, single-threaded validation p95 on the study gate "
                "machine and will vary across hardware."
            ),
        },
        "links": {
            "report": f"{REPOSITORY_WEB}/blob/main/benchmarks/flagship_study/REPORT.md",
        },
    }


def generate_summary(
    manifest_path: pathlib.Path = DEFAULT_MANIFEST,
    selection_path: pathlib.Path = DEFAULT_SELECTION,
    test_path: pathlib.Path = DEFAULT_TEST_DATA,
) -> dict[str, Any]:
    return build_summary(
        _read_json(manifest_path),
        _read_json(selection_path),
        _read_json(test_path),
        manifest_hash=_sha256_file(manifest_path),
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return f"globalThis.PaperSoccerBenchmarkResults = {payload};\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="replace the checked-in snapshot")
    action.add_argument("--check", action="store_true", help="fail if the snapshot is stale")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-lock", type=pathlib.Path, default=DEFAULT_SELECTION)
    parser.add_argument("--test-data", type=pathlib.Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected = render_summary(
            generate_summary(args.manifest, args.selection_lock, args.test_data)
        )
        if args.write:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(expected, encoding="utf-8")
            return 0
        try:
            actual = args.output.read_text(encoding="utf-8")
        except OSError as error:
            raise SummaryError(f"could not read snapshot {args.output}: {error}") from error
        if actual != expected:
            raise SummaryError(
                f"benchmark web snapshot is stale: run {pathlib.Path(__file__).as_posix()} --write"
            )
        return 0
    except SummaryError as error:
        print(f"web summary error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
