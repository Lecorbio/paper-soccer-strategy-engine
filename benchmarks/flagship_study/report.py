"""Deterministic Markdown renderer for the frozen flagship study."""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.flagship_study import studylib


class ReportError(ValueError):
    """Raised when curated evidence cannot support a publishable report."""


_FAMILIES = ("mcts", "alpha_beta", "jacek_inspired", "rank5_derived")
_FAMILY_NAMES = {
    "mcts": "Tactical MCTS",
    "alpha_beta": "Hand alpha-beta",
    "jacek_inspired": "Neural alpha-beta",
    "rank5_derived": studylib.PUBLIC_RANK5_LABEL,
}
_PROHIBITED_BENCHMARK = re.compile(r"random[\s_-]*bot", re.IGNORECASE)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportError(f"{where} must be a non-empty string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReportError(f"{where} must be a finite number")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportError(f"{where} must be an integer >= {minimum}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ReportError(f"{where} must be a boolean")
    return value


def _percent(value: Any, where: str, digits: int = 1) -> str:
    checked = _number(value, where)
    if not 0.0 <= checked <= 1.0:
        raise ReportError(f"{where} must be in [0,1]")
    return f"{checked:.{digits}%}"


def _signed_percentage_points(value: Any, where: str) -> str:
    checked = _number(value, where)
    if not -1.0 <= checked <= 1.0:
        raise ReportError(f"{where} must be in [-1,1]")
    return f"{checked * 100.0:+.1f} pp"


def _signed_interval(lower: Any, upper: Any, where: str) -> str:
    low = _number(lower, f"{where} lower")
    high = _number(upper, f"{where} upper")
    if not -1.0 <= low <= high <= 1.0:
        raise ReportError(f"{where} must be an ordered interval in [-1,1]")
    return f"[{low * 100.0:+.1f}, {high * 100.0:+.1f}] pp"


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _setting(config: Mapping[str, Any]) -> str:
    kind = config.get("kind")
    settings = _mapping(config.get("settings"), f"{config.get('id')} settings")
    if kind == "mcts":
        return (
            f"{_integer(settings.get('iterations'), 'MCTS iterations', minimum=1):,} "
            "iterations; tactical rollout; rollout-only leaf; tree reuse"
        )
    if kind in ("alpha-beta", "jacek-inspired"):
        depth = _integer(settings.get("max_turn_depth"), "alpha-beta depth", minimum=1)
        nodes = _integer(settings.get("max_nodes"), "alpha-beta nodes", minimum=1)
        suffix = "; frozen neural model" if kind == "jacek-inspired" else "; hand evaluator"
        return f"depth {depth}; {nodes:,} nodes; wall clock disabled{suffix}"
    if kind == "rank5-derived":
        return (
            f"depth {_integer(settings.get('max_turn_depth'), 'Rank5 depth', minimum=1)}; "
            f"{_integer(settings.get('max_nodes'), 'Rank5 nodes', minimum=1):,} nodes; "
            f"{_integer(settings.get('transposition_table_entries'), 'Rank5 TT entries', minimum=1):,} TT; "
            f"{_integer(settings.get('evaluation_cache_entries'), 'Rank5 cache entries', minimum=1):,} eval cache; "
            f"wall clock {'disabled' if _integer(settings.get('wall_clock_limit_ms'), 'Rank5 wall clock') == 0 else 'enabled'}; "
            f"replay corrections {'disabled' if not _bool(settings.get('replay_corrections'), 'Rank5 replay corrections') else 'enabled'}; "
            f"{_integer(settings.get('learned_value_blend_percent'), 'Rank5 learned blend')}% learned blend; "
            f"seed {'ignored' if _bool(settings.get('seed_ignored'), 'Rank5 seed policy') else 'used'}; "
            f"{_string(settings.get('rules_profile'), 'Rank5 rules profile')} rules"
        )
    raise ReportError(f"unsupported entrant kind {kind!r}")


def _configurations(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = _sequence(manifest.get("configurations"), "manifest.configurations")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        config = _mapping(value, f"manifest.configurations[{index}]")
        identifier = _string(config.get("id"), f"configurations[{index}].id")
        if identifier in result:
            raise ReportError(f"duplicate configuration ID {identifier}")
        label = _string(config.get("public_label"), f"{identifier}.public_label")
        if _PROHIBITED_BENCHMARK.search(label) or _PROHIBITED_BENCHMARK.search(identifier):
            raise ReportError("prohibited benchmark entrant or label")
        result[identifier] = config
    return result


def _validate_curated_phase(data: Mapping[str, Any], phase: str,
                            manifest_hash: str) -> None:
    if data.get("phase") != phase:
        raise ReportError(f"expected {phase} curated data")
    if data.get("manifest_sha256") != manifest_hash:
        raise ReportError(f"{phase} curated data belongs to another manifest")
    completeness = _mapping(data.get("completeness"), f"{phase}.completeness")
    expected_units = _integer(
        completeness.get("expected_units"), f"{phase}.expected_units"
    )
    completed_units = _integer(
        completeness.get("completed_units"), f"{phase}.completed_units"
    )
    expected_games = _integer(
        completeness.get("expected_games"), f"{phase}.expected_games"
    )
    completed_games = _integer(
        completeness.get("completed_games"), f"{phase}.completed_games"
    )
    unique_games = _integer(
        completeness.get("unique_game_ids"), f"{phase}.unique_game_ids"
    )
    truncations = _integer(
        completeness.get("truncations"), f"{phase}.truncations"
    )
    operational = _bool(
        completeness.get("operationally_valid"), f"{phase}.operationally_valid"
    )
    if truncations != 0:
        raise ReportError(f"{phase} contains truncations; publication refused")
    if (not operational or expected_units != completed_units or
            expected_games != completed_games or completed_games != unique_games):
        raise ReportError(f"{phase} curated data is incomplete")


def _resolve_slot(slot: str, selection: Mapping[str, Any]) -> str:
    if slot == "fixed:rank5_derived":
        return _string(selection.get("fixed_rank5_configuration"),
                       "fixed Rank5 selection")
    if not slot.startswith("selected:"):
        raise ReportError(f"unsupported test slot {slot}")
    family = slot.removeprefix("selected:")
    selected = _mapping(selection.get("selected_configurations"),
                        "selection.selected_configurations")
    return _string(selected.get(family), f"selected {family}")


def _selected_ids(selection: Mapping[str, Any]) -> list[str]:
    selected = _mapping(selection.get("selected_configurations"),
                        "selection.selected_configurations")
    values = [
        _string(selected.get(family), f"selected {family}")
        for family in _FAMILIES[:3]
    ]
    values.append(_string(selection.get("fixed_rank5_configuration"),
                          "fixed Rank5 configuration"))
    if len(set(values)) != 4:
        raise ReportError("selection lock does not identify four distinct entrants")
    return values


def _validate_selection(selection: Mapping[str, Any], manifest_hash: str,
                        configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if selection.get("manifest_sha256") != manifest_hash:
        raise ReportError("selection lock belongs to another manifest")
    if selection.get("source_phase") != "validation":
        raise ReportError("selection lock is not validation-only")
    if selection.get("test_authorized") is not True:
        raise ReportError("selection lock does not authorize frozen test evaluation")
    identifiers = _selected_ids(selection)
    if any(identifier not in configs for identifier in identifiers):
        raise ReportError("selection lock names an unknown configuration")
    rank5 = identifiers[-1]
    if configs[rank5].get("kind") != "rank5-derived" or rank5 != "rank5-fixed-50k":
        raise ReportError("test results cannot substitute an authentic ranked artifact")
    if configs[rank5].get("public_label") != studylib.PUBLIC_RANK5_LABEL:
        raise ReportError("fixed Rank5Derived public label changed")
    settings = _mapping(configs[rank5].get("settings"), "fixed Rank5Derived settings")
    fixed_integers = {
        "max_turn_depth": 32,
        "max_nodes": 50_000,
        "transposition_table_entries": 65_536,
        "evaluation_cache_entries": 32_768,
        "wall_clock_limit_ms": 0,
        "learned_value_blend_percent": 0,
    }
    for field, expected in fixed_integers.items():
        if _integer(settings.get(field), f"fixed Rank5Derived {field}") != expected:
            raise ReportError(f"fixed Rank5Derived setting changed: {field}")
    if _bool(settings.get("replay_corrections"), "Rank5 replay corrections"):
        raise ReportError("fixed Rank5Derived replay corrections must be disabled")
    if not _bool(settings.get("seed_ignored"), "Rank5 seed policy"):
        raise ReportError("fixed Rank5Derived seed must be ignored")
    if _string(settings.get("rules_profile"), "Rank5 rules profile") != "standard-8x10-demo":
        raise ReportError("fixed Rank5Derived rules profile changed")
    if _hash(settings.get("original_artifact_sha256"), "Rank5 source SHA-256") != studylib.RANK5_SOURCE_SHA256:
        raise ReportError("fixed Rank5Derived protected source hash changed")
    return identifiers


def _relative_link(manifest: Mapping[str, Any], target: str) -> str:
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    report_path = _string(outputs.get("report"), "outputs.report")
    start = posixpath.dirname(report_path) or "."
    return posixpath.relpath(target, start=start)


def _pairwise_rows(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[list[str]], list[str]]:
    schedule = _sequence(
        _mapping(manifest.get("schedule"), "manifest.schedule").get("test"),
        "manifest.schedule.test",
    )
    matchups = _mapping(test.get("matchups"), "test.matchups")
    schedule_ids = [_string(item.get("id"), "test matchup id")
                    for item in map(lambda value: _mapping(value, "test matchup"), schedule)]
    if set(schedule_ids) != set(matchups):
        raise ReportError("frozen test matchup set is incomplete or contains unknown entries")
    rows: list[list[str]] = []
    unresolved: list[str] = []
    for raw_schedule in schedule:
        scheduled = _mapping(raw_schedule, "test matchup")
        matchup_id = _string(scheduled.get("id"), "test matchup id")
        summary = _mapping(matchups[matchup_id], f"test.matchups.{matchup_id}")
        expected_left = _resolve_slot(_string(scheduled.get("left_slot"), "left slot"), selection)
        expected_right = _resolve_slot(
            _string(scheduled.get("right_slot"), "right slot"), selection
        )
        left_id = _string(summary.get("left_config_id"), f"{matchup_id}.left_config_id")
        right_id = _string(summary.get("right_config_id"), f"{matchup_id}.right_config_id")
        if (left_id, right_id) != (expected_left, expected_right):
            raise ReportError(f"{matchup_id} participants differ from the selection lock")
        left_label = _string(configs[left_id].get("public_label"), f"{left_id} label")
        right_label = _string(configs[right_id].get("public_label"), f"{right_id} label")
        games = _integer(summary.get("games"), f"{matchup_id}.games", minimum=1)
        wins = _integer(summary.get("left_wins"), f"{matchup_id}.left_wins")
        losses = _integer(summary.get("left_losses"), f"{matchup_id}.left_losses")
        truncations = _integer(summary.get("truncations"), f"{matchup_id}.truncations")
        pairs = _integer(summary.get("pairs"), f"{matchup_id}.pairs", minimum=1)
        won = _integer(summary.get("pairs_won_2_0"), f"{matchup_id}.pairs_won_2_0")
        split = _integer(summary.get("pairs_split_1_1"), f"{matchup_id}.pairs_split_1_1")
        lost = _integer(summary.get("pairs_lost_0_2"), f"{matchup_id}.pairs_lost_0_2")
        if truncations != 0:
            raise ReportError(f"truncation in frozen test matchup {matchup_id}")
        if (wins + losses != games or won + split + lost != pairs or
                games != pairs * 2 or wins != 2 * won + split or
                losses != 2 * lost + split):
            raise ReportError(f"incomplete or inconsistent pair accounting in {matchup_id}")
        mean = _number(summary.get("mean_pair_score"), f"{matchup_id}.mean_pair_score")
        expected_mean = (won + 0.5 * split) / pairs
        if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ReportError(f"paired mean disagrees with outcomes in {matchup_id}")
        interval = _mapping(summary.get("pair_bootstrap_95"),
                            f"{matchup_id}.pair_bootstrap_95")
        lower = _number(interval.get("lower"), f"{matchup_id}.CI lower")
        upper = _number(interval.get("upper"), f"{matchup_id}.CI upper")
        if not (0.0 <= lower <= upper <= 1.0):
            raise ReportError(f"invalid paired interval in {matchup_id}")
        if lower > 0.5:
            conclusion = f"**{left_label} is stronger than {right_label}.**"
        elif 1.0 - upper > 0.5:
            conclusion = f"**{right_label} is stronger than {left_label}.**"
        else:
            conclusion = "Statistically unresolved."
            unresolved.append(f"{left_label} versus {right_label} was statistically unresolved.")
        rows.append([
            left_label,
            right_label,
            str(wins),
            str(losses),
            str(won),
            str(split),
            str(lost),
            _percent(mean, f"{matchup_id} mean"),
            f"[{_percent(lower, 'CI lower')}, {_percent(upper, 'CI upper')}]",
            conclusion,
        ])
    return rows, unresolved


def _analysis(test: Mapping[str, Any]) -> Mapping[str, Any]:
    value = test.get("analysis")
    return _mapping(value, "test.analysis") if value is not None else test


def _hash(value: Any, where: str) -> str:
    text = _string(value, where)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReportError(f"{where} must be a lowercase SHA-256")
    return text


def _artifact_rows(manifest: Mapping[str, Any],
                   artifact_hashes: Mapping[str, Any]) -> list[tuple[str, str]]:
    result: dict[str, str] = {}
    for path, digest in artifact_hashes.items():
        result[_string(path, "artifact path")] = _hash(digest, f"artifact hash {path}")
    source = _mapping(manifest.get("source"), "manifest.source")
    protected = _mapping(source.get("protected_artifacts"), "source.protected_artifacts")
    fixed = {
        _string(protected.get("rank5_submission_path"), "rank5 path"):
            _hash(protected.get("rank5_submission_sha256"), "rank5 SHA-256"),
        _string(protected.get("jacek_model_path"), "model path"):
            _hash(protected.get("jacek_model_sha256"), "model SHA-256"),
        _string(source.get("analysis_contract_path"), "analysis contract path"):
            _hash(source.get("analysis_contract_sha256"), "analysis contract SHA-256"),
    }
    supersession_value = manifest.get("supersession")
    if supersession_value is not None:
        supersession = _mapping(supersession_value, "manifest.supersession")
        fixed.update({
            _string(
                supersession.get("failure_record_path"),
                "supersession failure record path",
            ): _hash(
                supersession.get("failure_record_sha256"),
                "supersession failure record SHA-256",
            ),
            _string(
                supersession.get("predecessor_manifest_path"),
                "predecessor manifest path",
            ): _hash(
                supersession.get("predecessor_manifest_sha256"),
                "predecessor manifest SHA-256",
            ),
        })
    for path, digest in fixed.items():
        if path in result and result[path] != digest:
            raise ReportError(f"artifact hash disagrees with manifest: {path}")
        result[path] = digest
    openings = _mapping(manifest.get("openings"), "manifest.openings")
    for index, value in enumerate(_sequence(openings.get("banks"), "openings.banks")):
        bank = _mapping(value, f"openings.banks[{index}]")
        path = _string(bank.get("path"), f"opening bank {index} path")
        digest = _hash(bank.get("sha256"), f"opening bank {index} SHA-256")
        if path in result and result[path] != digest:
            raise ReportError(f"artifact hash disagrees with opening bank: {path}")
        result[path] = digest
    return sorted(result.items())


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown(value) for value in row) + " |"
        for row in rows
    )
    return lines


def render_markdown_report(
    manifest: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    development: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
) -> str:
    """Render a byte-stable report from already validated frozen inputs."""

    manifest = _mapping(manifest, "manifest")
    selection = _mapping(selection_lock, "selection lock")
    development = _mapping(development, "development curated data")
    validation = _mapping(validation, "validation curated data")
    test = _mapping(test, "test curated data")
    artifact_hashes = _mapping(artifact_hashes, "artifact hashes")
    study = _mapping(manifest.get("study"), "manifest.study")
    disclaimer = _string(study.get("rank5_disclaimer"), "study.rank5_disclaimer")
    if disclaimer != studylib.RANK5_DISCLAIMER:
        raise ReportError("mandated Rank5Derived disclaimer changed")
    configs = _configurations(manifest)
    manifest_hash = _hash(selection.get("manifest_sha256"), "manifest SHA-256")
    selected_ids = _validate_selection(selection, manifest_hash, configs)
    for phase, data in (
        ("development", development),
        ("validation", validation),
        ("test", test),
    ):
        _validate_curated_phase(data, phase, manifest_hash)
    if "analysis_complete" in test and test.get("analysis_complete") is not True:
        raise ReportError("test analysis is incomplete")

    pairwise_rows, unresolved_findings = _pairwise_rows(
        manifest, selection, test, configs
    )
    analysis = _analysis(test)
    bt = _mapping(analysis.get("bradley_terry"), "test Bradley-Terry analysis")
    bt_intervals = _mapping(bt.get("intervals"), "Bradley-Terry intervals")
    if set(bt_intervals) != set(selected_ids):
        raise ReportError("Bradley-Terry analysis must contain exactly four selected bots")
    calibration = _mapping(analysis.get("calibration"), "test calibration analysis")
    if set(calibration) != set(selected_ids):
        raise ReportError("calibration analysis must contain exactly four selected bots")

    labels = _mapping(study.get("public_labels"), "study.public_labels")
    if labels.get("rank5_derived") != studylib.PUBLIC_RANK5_LABEL:
        raise ReportError("Rank5Derived public identity changed")
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    charts = _mapping(outputs.get("charts"), "outputs.charts")
    data_paths = _mapping(outputs.get("curated_data"), "outputs.curated_data")
    selection_metrics = _mapping(selection.get("validation_metrics"),
                                 "selection.validation_metrics")
    development_configs = _mapping(development.get("configurations"),
                                   "development.configurations")
    statistics = _mapping(manifest.get("statistics"), "manifest.statistics")
    bootstrap = _mapping(statistics.get("bootstrap"), "statistics.bootstrap")
    bootstrap_resamples = _integer(
        bootstrap.get("resamples"), "bootstrap resamples", minimum=1
    )
    environment = _mapping(manifest.get("environment"), "manifest.environment")
    python_version = _string(
        environment.get("python_version"), "manifest Python version"
    )
    source = _mapping(manifest.get("source"), "manifest.source")
    source_commit = _string(source.get("git_commit"), "source commit")
    arena_sha256 = _hash(source.get("arena_sha256"), "arena SHA-256")
    opening_tool_sha256 = _hash(
        source.get("opening_tool_sha256"), "opening tool SHA-256"
    )
    preregistered_at = _string(
        study.get("preregistered_at_utc"), "study.preregistered_at_utc"
    )
    supersession_value = manifest.get("supersession")
    supersession = (
        _mapping(supersession_value, "manifest.supersession")
        if supersession_value is not None else None
    )
    if supersession is not None:
        if (
            supersession.get("predecessor_status")
            != "stopped_before_test_calibration_implementation_defect"
            or supersession.get("predecessor_test_outcomes_accessed") is not False
            or supersession.get(
                "predecessor_validation_results_used_for_v4_selection_or_calibration"
            ) is not False
            or supersession.get("fresh_opening_phases") != ["validation"]
            or supersession.get("fresh_validation_exclusion_scope")
            != "all_predecessor_opening_banks"
            or supersession.get("reused_opening_phases")
            != ["development", "test"]
        ):
            raise ReportError("unsupported prospective recovery lineage")
    ablations = _mapping(
        selection.get("development_validation_ablations"),
        "selection.development_validation_ablations",
    )
    if ablations.get("schema") != "papersoccer.flagship-study-ablations.v1":
        raise ReportError("unsupported development/validation ablation schema")
    practical_threshold = _number(
        ablations.get("practical_gain_threshold"), "ablation practical threshold"
    )
    if practical_threshold != 0.01:
        raise ReportError("ablation practical threshold changed")
    scaling_ablations = _mapping(ablations.get("scaling"), "scaling ablations")
    if set(scaling_ablations) != {"mcts", "alpha_beta", "jacek_inspired"}:
        raise ReportError("ablation scaling families changed")
    evaluator_ablations = _sequence(
        ablations.get("equal_budget_evaluator"), "equal-budget evaluator ablations"
    )

    ablation_rows: list[list[str]] = []
    ablation_findings: list[str] = []

    def append_ablation(raw: Any, family_label: str,
                        *, evaluator: bool) -> None:
        comparison = _mapping(raw, f"{family_label} ablation")
        lower_id = _string(comparison.get("lower_config_id"), "ablation lower ID")
        higher_id = _string(comparison.get("higher_config_id"), "ablation higher ID")
        if lower_id not in configs or higher_id not in configs:
            raise ReportError("ablation names an unknown configuration")
        phases = _mapping(comparison.get("phases"), "ablation phases")
        if set(phases) != {"development", "validation"}:
            raise ReportError("ablation must contain development and validation")
        formatted: dict[str, tuple[str, str]] = {}
        for phase in ("development", "validation"):
            metrics = _mapping(phases[phase], f"ablation {phase}")
            pairs = _integer(metrics.get("pairs"), f"ablation {phase} pairs", minimum=1)
            lower_score = _number(
                metrics.get("lower_score"), f"ablation {phase} lower score"
            )
            higher_score = _number(
                metrics.get("higher_score"), f"ablation {phase} higher score"
            )
            delta = _number(metrics.get("delta"), f"ablation {phase} delta")
            if (not 0.0 <= lower_score <= 1.0
                    or not 0.0 <= higher_score <= 1.0
                    or not math.isclose(
                        delta, higher_score - lower_score,
                        rel_tol=0.0, abs_tol=1e-12,
                    )):
                raise ReportError("ablation scores and delta are inconsistent")
            interval = _mapping(
                metrics.get("pair_difference_bootstrap_95"),
                f"ablation {phase} interval",
            )
            if _integer(
                interval.get("resamples"), f"ablation {phase} resamples", minimum=1
            ) != bootstrap_resamples:
                raise ReportError("ablation bootstrap resample count changed")
            interval_text = _signed_interval(
                interval.get("lower"), interval.get("upper"),
                f"ablation {phase} interval",
            )
            formatted[phase] = (
                f"{_percent(lower_score, 'ablation lower')} → "
                f"{_percent(higher_score, 'ablation higher')} (n={pairs})",
                f"{_signed_percentage_points(delta, 'ablation delta')} "
                f"{interval_text}",
            )
        classification = _string(
            comparison.get("validation_classification"),
            "ablation validation classification",
        )
        allowed = (
            {
                "neural_materially_stronger", "hand_materially_stronger",
                "practical_equivalence_supported", "unresolved_at_1pp",
            }
            if evaluator else {
                "supported_practical_gain", "supported_regression",
                "supported_no_practical_gain", "unresolved_at_1pp",
            }
        )
        if classification not in allowed:
            raise ReportError("unknown ablation classification")
        contrast = (
            f"{lower_id} → {higher_id} (neural minus hand)"
            if evaluator else f"{lower_id} → {higher_id}"
        )
        ablation_rows.append([
            family_label, contrast,
            formatted["development"][0], formatted["development"][1],
            formatted["validation"][0], formatted["validation"][1],
            classification.replace("_", " "),
        ])
        if classification not in {
            "supported_practical_gain", "neural_materially_stronger"
        }:
            ablation_findings.append(
                f"{contrast}: validation classification was "
                f"{classification.replace('_', ' ')}; the paired interval and point "
                "delta are reported in the preregistered ablation table."
            )

    for family in ("mcts", "alpha_beta", "jacek_inspired"):
        values = _sequence(scaling_ablations[family], f"{family} scaling ablations")
        if len(values) != 3:
            raise ReportError(f"{family} must contain three scaling contrasts")
        for value in values:
            append_ablation(value, _FAMILY_NAMES[family], evaluator=False)
    if len(evaluator_ablations) != 3:
        raise ReportError("equal-budget evaluator ablations must contain three contrasts")
    for value in evaluator_ablations:
        append_ablation(value, "Neural minus hand", evaluator=True)

    lines = [
        f"# {_string(study.get('title'), 'study.title')}",
        "",
        "## Research question and hypotheses",
        "",
        "**Question.** Under standard 8×10 demo rules, which of the four competitive "
        "entrants has the strongest frozen test performance, the best calibrated "
        "predictions, and the most favorable validation strength/latency tradeoff?",
        "",
        "The preregistered hypotheses were:",
        "",
        "- Additional fixed computation may improve paired strength, with a measurable latency cost.",
        "- Hand-crafted and neural alpha-beta evaluation may differ in both strength and calibration at equal node budgets.",
        f"- The fixed {studylib.PUBLIC_RANK5_LABEL} may fall on or off the constrained validation frontier without any profile tuning.",
        "",
        "## Entrants",
        "",
        f"- **{_markdown(labels['mcts'])}.** Tactical rollouts, rollout-only leaves, deterministic tree reuse, and the frozen iteration sweep.",
        f"- **{_markdown(labels['alpha_beta'])}.** Depth-six possession search with the hand evaluator and fixed node budgets.",
        f"- **{_markdown(labels['jacek_inspired'])}.** The separate depth-six neural entrant using the frozen model hash listed below.",
        f"- **{studylib.PUBLIC_RANK5_LABEL}.** The exact hard-locked 32-turn-depth, 50,000-node demo comparator with replay corrections and learned-value blending disabled.",
        "",
        disclaimer,
        "",
        "## Controls and frozen openings",
        "",
    ]
    rules = _mapping(manifest.get("rules"), "manifest.rules")
    openings = _mapping(manifest.get("openings"), "manifest.openings")
    generator = _mapping(openings.get("generator"), "openings.generator")
    depths = [_integer(value, "opening depth", minimum=1)
              for value in _sequence(openings.get("depths"), "openings.depths")]
    lines.extend([
        f"Games used {_integer(rules.get('width'), 'rules.width')}×"
        f"{_integer(rules.get('height'), 'rules.height')} demo rules, "
        f"{_integer(rules.get('playable_edges'), 'rules.playable_edges')} playable edges, "
        f"and a {_integer(rules.get('max_game_plies'), 'rules.max_game_plies')}-ply safety limit. "
        "The engine has no draw status.",
        "",
        f"Openings at {', '.join(map(str, depths))} physical plies were produced by "
        f"**{_markdown(generator.get('description'))}**. It was solely a data-generation "
        "mechanism, never an entrant or strength baseline. Transcripts were replay-validated, "
        "phase-disjoint, duplicate-screened, color-swapped, and frozen by the hashes below.",
        "",
        f"Opening ply: {_markdown(rules.get('opening_ply_definition'))}",
        "",
    ])
    if supersession is not None:
        lines.extend([
            "## Prospective recovery lineage",
            "",
            "Version 4 prospectively superseded version 3 after the predecessor "
            "stopped before test because of a validation calibration implementation "
            "defect. No version-3 test outcomes were accessed, and no version-3 "
            "validation results were used for version-4 selection or calibration. "
            "Version 4 used fresh validation banks excluded from every predecessor "
            "opening bank while reusing the development and test banks byte-for-byte. "
            "The predecessor manifest and failure record are bound by SHA-256 in the "
            "artifact table.",
            "",
        ])
    lines.extend([
        "## Candidate grids",
        "",
    ])
    grid_rows: list[list[str]] = []
    grids = _mapping(manifest.get("candidate_grids"), "manifest.candidate_grids")
    for family in _FAMILIES:
        identifiers = [_string(value, f"{family} grid ID")
                       for value in _sequence(grids.get(family), f"{family} grid")]
        for identifier in identifiers:
            if identifier not in configs:
                raise ReportError(f"candidate grid names unknown config {identifier}")
            grid_rows.append([
                _FAMILY_NAMES[family], identifier, _setting(configs[identifier])
            ])
    lines.extend(_table(("Family", "Configuration", "Frozen work/profile"), grid_rows))

    latency = _mapping(manifest.get("latency_protocol"), "manifest.latency_protocol")
    lines.extend([
        "",
        "## Latency protocol",
        "",
        f"Validation used a native {_markdown(latency.get('build_type'))}, one foreground "
        f"thread, and a {_number(latency.get('gate_ms'), 'latency gate'):g} ms p95 gate. "
        f"Timer boundary: {_markdown(latency.get('timer_boundary'))}. "
        f"Warm-up: {_markdown(latency.get('warmup'))}.",
        "",
        f"State/setup policy: {_markdown(latency.get('state_copying'))}. "
        f"Power conditions: {_markdown(latency.get('power_conditions'))}.",
        "",
        f"{studylib.PUBLIC_RANK5_LABEL} eligibility uses fresh-root p95; all returned edges, including cached "
        "continuations, are reported separately.",
        "",
        f"Gate machine: {_markdown(environment.get('machine_id'))}; "
        f"{_markdown(environment.get('os'))}; {_markdown(environment.get('cpu'))}; "
        f"{_markdown(environment.get('compiler_version'))}. Build flags: "
        f"{_markdown(environment.get('build_flags'))}.",
        "",
        "## Selection rule",
        "",
    ])
    selection_rule = _mapping(manifest.get("selection_rule"), "manifest.selection_rule")
    lines.extend([
        f"Selection used validation {_markdown(selection_rule.get('strength_metric'))}. "
        f"Candidates within {_number(selection_rule.get('practical_tie_percentage_points'), 'tie threshold'):g} "
        "percentage point of the strongest eligible result were tied; ties were resolved by "
        "lower p95, smaller work budget, then stable configuration ID. A family with no "
        "eligible candidate would have stopped the study before test.",
        "",
        "## Development and validation findings",
        "",
    ])
    tuning_rows: list[list[str]] = []
    negative_findings: list[str] = []
    for identifier in sorted(selection_metrics):
        metric = _mapping(selection_metrics[identifier], f"validation metric {identifier}")
        development_summary = _mapping(
            development_configs.get(identifier), f"development config {identifier}"
        )
        development_strength = _mapping(
            development_summary.get("strength"), f"development strength {identifier}"
        )
        dev_value = _number(
            development_strength.get("mean_pair_score"), f"{identifier} development strength"
        )
        validation_strength = _number(
            metric.get("validation_strength"), f"{identifier} validation strength"
        )
        p95 = _number(metric.get("validation_p95_ms"), f"{identifier} validation p95")
        eligible = _bool(metric.get("eligible"), f"{identifier} eligibility")
        selected = _bool(metric.get("selected"), f"{identifier} selected")
        tuning_rows.append([
            identifier,
            _percent(dev_value, f"{identifier} development strength"),
            _percent(validation_strength, f"{identifier} validation strength"),
            f"{p95:.3f}",
            "eligible" if eligible else "rejected",
            "selected" if selected else "—",
        ])
        if not eligible:
            negative_findings.append(
                f"{identifier} missed the 50 ms validation p95 gate ({p95:.3f} ms)."
            )
    lines.extend(_table(
        ("Configuration", "Development score", "Validation score", "Validation p95 (ms)",
         "Gate", "Lock"),
        tuning_rows,
    ))
    rank_latency = _mapping(selection.get("rank5_latency"), "selection.rank5_latency")
    lines.extend([
        "",
        f"{studylib.PUBLIC_RANK5_LABEL} fresh-root p95 was "
        f"{_number(rank_latency.get('fresh_root_p95_ms'), 'Rank5 fresh p95'):.3f} ms; "
        f"all-edge p95 was {_number(rank_latency.get('all_edge_p95_ms'), 'Rank5 all p95'):.3f} ms. "
        f"Its constrained status was "
        f"{'eligible' if _bool(rank_latency.get('eligible_under_50_ms'), 'Rank5 eligibility') else 'ineligible'}; "
        "the profile remained fixed either way.",
        "",
        "### Preregistered development/validation ablations",
        "",
        f"All contrasts align the same opening pairs and use {bootstrap_resamples:,} "
        "whole-pair difference bootstraps stratified by opening depth. The frozen "
        f"practical threshold is {practical_threshold:.1%}. Test outcomes do not enter "
        "these classifications.",
        "",
    ])
    lines.extend(_table(
        ("Family", "Contrast", "Development scores", "Development delta [95% CI]",
         "Validation scores", "Validation delta [95% CI]", "Validation class"),
        ablation_rows,
    ))
    lines.extend([
        "",
        "## Locked configurations",
        "",
    ])
    locked_rows = []
    for family, identifier in zip(_FAMILIES, selected_ids, strict=True):
        locked_rows.append([
            _FAMILY_NAMES[family],
            identifier,
            _setting(configs[identifier]),
            "fixed comparator" if family == "rank5_derived" else "validation selection",
        ])
    lines.extend(_table(("Entrant", "Locked ID", "Exact profile", "Basis"), locked_rows))

    lines.extend([
        "",
        "## Frozen test results",
        "",
        f"The frozen tournament completed "
        f"{_integer(_mapping(test.get('completeness'), 'test completeness').get('completed_games'), 'test completed games'):,} "
        "decisive games with zero truncations. A 1–1 pair split is two decisive games, not a draw.",
        "",
        "### Pairwise results",
        "",
        f"Intervals use {bootstrap_resamples:,} deterministic whole-pair bootstrap "
        "resamples while preserving opening-depth strata.",
        "",
    ])
    lines.extend(_table(
        ("Left entrant", "Right entrant", "Wins", "Losses", "2–0", "1–1", "0–2",
         "Mean paired score", "Paired bootstrap 95% CI", "Conclusion"),
        pairwise_rows,
    ))
    lines.extend([
        "",
        "## Bradley–Terry relative strength",
        "",
        "Abilities use a sum-to-zero identifiability constraint; zero is a relative reference, "
        "not absolute skill. Uncertainty resamples complete color-swapped pairs within matchup "
        "and opening-depth strata.",
        "",
        f"![Test Bradley–Terry relative strength]({_relative_link(manifest, _string(charts.get('bradley_terry'), 'BT chart path'))})",
        "",
    ])
    bt_rows = []
    for identifier in selected_ids:
        interval = _mapping(bt_intervals[identifier], f"BT interval {identifier}")
        estimate = _number(interval.get("estimate"), f"BT estimate {identifier}")
        lower = _number(interval.get("lower"), f"BT lower {identifier}")
        upper = _number(interval.get("upper"), f"BT upper {identifier}")
        if lower > upper:
            raise ReportError(f"invalid Bradley-Terry interval for {identifier}")
        bt_rows.append([
            _string(configs[identifier].get("public_label"), f"{identifier} label"),
            f"{estimate:.3f}",
            f"[{lower:.3f}, {upper:.3f}]",
        ])
    lines.extend(_table(("Entrant", "Relative ability", "Bootstrap 95% CI"), bt_rows))

    lines.extend([
        "",
        "## Calibration",
        "",
        "Logistic mappings were fitted and frozen on validation scores after Player-One-to-"
        "player-to-move orientation and population-standardization. Test metrics apply those "
        f"mappings without refitting. {studylib.PUBLIC_RANK5_LABEL} uses fresh-root "
        "predictions only. Uncertainty resamples whole color-swapped pairs within matchup × "
        "opening-depth strata, preserving dependent decisions within both games.",
        "",
        f"![Test reliability and calibration]({_relative_link(manifest, _string(charts.get('calibration'), 'calibration chart path'))})",
        "",
    ])
    calibration_rows = []
    reliability_rows = []
    for identifier in selected_ids:
        metrics = _mapping(calibration[identifier], f"calibration {identifier}")
        samples = _integer(metrics.get("samples"), f"{identifier} calibration samples", minimum=1)
        decisions = _integer(
            metrics.get("decision_count"),
            f"{identifier} calibration decision opportunities",
            minimum=samples,
        )
        excluded = _mapping(
            metrics.get("excluded"), f"{identifier} calibration exclusions"
        )
        if set(excluded) != {
                "cached_continuations", "truncations", "invalid_depths"}:
            raise ReportError(f"{identifier} calibration exclusion schema changed")
        excluded_cached = _integer(
            excluded.get("cached_continuations"),
            f"{identifier} cached-continuation exclusions",
        )
        excluded_truncations = _integer(
            excluded.get("truncations"), f"{identifier} truncation exclusions"
        )
        excluded_invalid = _integer(
            excluded.get("invalid_depths"),
            f"{identifier} invalid-depth exclusions",
        )
        if samples + excluded_cached + excluded_truncations + excluded_invalid != decisions:
            raise ReportError(
                f"{identifier} retained and excluded calibration counts do not total decisions"
            )
        pair_clusters = _integer(
            metrics.get("pair_clusters"), f"{identifier} calibration pair clusters",
            minimum=1,
        )
        brier = _number(metrics.get("brier_score"), f"{identifier} Brier score")
        log_loss = _number(metrics.get("log_loss"), f"{identifier} log loss")
        score_bootstrap = _mapping(
            metrics.get("pair_cluster_bootstrap_95"),
            f"{identifier} calibration bootstrap",
        )
        if (score_bootstrap.get("method") != "pair_cluster_percentile_stratified"
                or _integer(
                    score_bootstrap.get("resamples"),
                    f"{identifier} calibration resamples", minimum=1,
                ) != bootstrap_resamples):
            raise ReportError(f"{identifier} calibration bootstrap contract changed")
        brier_interval = _mapping(
            score_bootstrap.get("brier_score"), f"{identifier} Brier interval"
        )
        log_interval = _mapping(
            score_bootstrap.get("log_loss"), f"{identifier} log-loss interval"
        )
        brier_lower = _number(brier_interval.get("lower"), "Brier interval lower")
        brier_upper = _number(brier_interval.get("upper"), "Brier interval upper")
        log_lower = _number(log_interval.get("lower"), "log-loss interval lower")
        log_upper = _number(log_interval.get("upper"), "log-loss interval upper")
        if not 0.0 <= brier_lower <= brier_upper <= 1.0 or \
           not 0.0 <= log_lower <= log_upper:
            raise ReportError(f"{identifier} calibration score interval is invalid")
        bins = _sequence(metrics.get("reliability_bins"), f"{identifier} reliability bins")
        if len(bins) != 10:
            raise ReportError(f"{identifier} calibration must contain ten bins")
        label = _string(configs[identifier].get("public_label"), f"{identifier} label")
        calibration_rows.append([
            label, f"{samples}/{decisions}", str(excluded_cached),
            str(excluded_invalid), str(excluded_truncations), str(pair_clusters),
            f"{brier:.4f} [{brier_lower:.4f}, {brier_upper:.4f}]",
            f"{log_loss:.4f} [{log_lower:.4f}, {log_upper:.4f}]",
        ])
        for expected_bin, raw_bin in enumerate(bins):
            bin_value = _mapping(raw_bin, f"{identifier} reliability bin")
            if _integer(bin_value.get("bin"), "reliability bin index") != expected_bin:
                raise ReportError(f"{identifier} reliability bins are out of order")
            lower_edge = _number(bin_value.get("lower"), "reliability lower edge")
            upper_edge = _number(bin_value.get("upper"), "reliability upper edge")
            count = _integer(bin_value.get("count"), "reliability prediction count")
            clusters = _integer(
                bin_value.get("pair_clusters"), "reliability pair-cluster count"
            )
            if not (math.isclose(lower_edge, expected_bin / 10.0)
                    and math.isclose(upper_edge, (expected_bin + 1) / 10.0)):
                raise ReportError(f"{identifier} reliability bin edges changed")
            if count == 0:
                mean_text = observed_text = "—"
            else:
                mean_prediction = _number(
                    bin_value.get("mean_prediction"), "reliability mean prediction"
                )
                observed = _number(
                    bin_value.get("observed_frequency"), "reliability observed frequency"
                )
                if not 0.0 <= mean_prediction <= 1.0 or not 0.0 <= observed <= 1.0:
                    raise ReportError("reliability values must be probabilities")
                mean_text = f"{mean_prediction:.3f}"
                observed_text = f"{observed:.3f}"
            interval_value = bin_value.get(
                "observed_frequency_pair_bootstrap_95"
            )
            successful = _integer(
                bin_value.get("bootstrap_successful_resamples"),
                "reliability successful resamples",
            )
            if interval_value is None:
                interval_text = f"— ({successful:,} populated replicates)"
            else:
                interval = _mapping(interval_value, "reliability pair interval")
                interval_lower = _number(interval.get("lower"), "reliability CI lower")
                interval_upper = _number(interval.get("upper"), "reliability CI upper")
                if (interval.get("method") != "pair_cluster_percentile_stratified"
                        or not 0.0 <= interval_lower <= interval_upper <= 1.0
                        or _integer(
                            interval.get("successful_resamples"),
                            "reliability interval successes",
                        ) != successful):
                    raise ReportError("reliability pair-cluster interval is invalid")
                interval_text = (
                    f"[{interval_lower:.3f}, {interval_upper:.3f}] "
                    f"({successful:,} populated replicates)"
                )
            edge_text = (
                f"[{lower_edge:.1f}, {upper_edge:.1f}]"
                if expected_bin == 9 else f"[{lower_edge:.1f}, {upper_edge:.1f})"
            )
            reliability_rows.append([
                label, str(expected_bin), edge_text, str(count), str(clusters),
                mean_text, observed_text, interval_text,
            ])
    lines.extend(_table(
        ("Entrant", "Retained/decision opportunities", "Excluded cached",
         "Excluded invalid depth", "Excluded truncation", "Pair clusters",
         "Brier [pair 95% CI]", "Log loss [pair 95% CI]"),
        calibration_rows,
    ))
    lines.extend([
        "",
        "### Ten-bin reliability summaries",
        "",
    ])
    lines.extend(_table(
        ("Entrant", "Bin", "Probability range", "Prediction n", "Pair n",
         "Mean prediction", "Observed frequency", "Pair-bootstrap 95% CI"),
        reliability_rows,
    ))

    lines.extend([
        "",
        "## Validation Pareto frontier",
        "",
        "The constrained frontier includes only configurations at or below the 50 ms gate, "
        "maximizes common-opponent validation paired score, and minimizes validation p95 latency. "
        "Unconstrained status is shown separately; test results never revise either classification.",
        "",
        f"![Validation strength versus p95 latency]({_relative_link(manifest, _string(charts.get('pareto'), 'Pareto chart path'))})",
        "",
    ])
    pareto_rows = []
    pareto = _sequence(selection.get("validation_pareto"), "selection.validation_pareto")
    for value in sorted(pareto, key=lambda item: str(_mapping(item, "Pareto point").get("id"))):
        point = _mapping(value, "Pareto point")
        identifier = _string(point.get("id"), "Pareto point ID")
        strength = _number(
            point.get("validation_strength"), f"{identifier} validation strength"
        )
        development_strength = _number(
            point.get("development_strength"), f"{identifier} development strength"
        )
        p95 = _number(
            point.get("validation_p95_ms"), f"{identifier} validation p95"
        )
        latency_decisions = _integer(
            point.get("validation_latency_decisions"),
            f"{identifier} validation latency decisions", minimum=1,
        )
        optimal = _bool(
            point.get("constrained_pareto_optimal"),
            f"{identifier} constrained Pareto status",
        )
        unconstrained = _bool(
            point.get("unconstrained_pareto_optimal"),
            f"{identifier} unconstrained Pareto status",
        )
        gate = _bool(point.get("gate_eligible"), f"{identifier} gate status")
        selected = _bool(point.get("selected"), f"{identifier} selection status")
        fixed = _bool(point.get("fixed"), f"{identifier} fixed status")
        display_identifier = (
            _string(configs[identifier].get("public_label"), f"{identifier} label")
            if fixed else identifier
        )
        validation_interval_value = point.get(
            "validation_strength_pair_bootstrap_95"
        )
        development_interval_value = point.get(
            "development_strength_pair_bootstrap_95"
        )
        if fixed:
            if (point.get("validation_strength_pairs") is not None
                    or point.get("development_strength_pairs") is not None
                    or validation_interval_value is not None
                    or development_interval_value is not None):
                raise ReportError("defined fixed Pareto reference cannot have strength samples")
            development_text = _percent(
                development_strength, f"{identifier} development reference"
            ) + " (defined; n=N/A)"
            validation_text = _percent(
                strength, f"{identifier} validation reference"
            ) + " (defined; n=N/A)"
        else:
            validation_pairs = _integer(
                point.get("validation_strength_pairs"),
                f"{identifier} validation strength pairs", minimum=1,
            )
            development_pairs = _integer(
                point.get("development_strength_pairs"),
                f"{identifier} development strength pairs", minimum=1,
            )
            validation_interval = _mapping(
                validation_interval_value, f"{identifier} validation strength interval"
            )
            development_interval = _mapping(
                development_interval_value, f"{identifier} development strength interval"
            )
            validation_lower = _number(
                validation_interval.get("lower"), "validation strength CI lower"
            )
            validation_upper = _number(
                validation_interval.get("upper"), "validation strength CI upper"
            )
            development_lower = _number(
                development_interval.get("lower"), "development strength CI lower"
            )
            development_upper = _number(
                development_interval.get("upper"), "development strength CI upper"
            )
            if (not 0.0 <= validation_lower <= validation_upper <= 1.0
                    or not 0.0 <= development_lower <= development_upper <= 1.0):
                raise ReportError("Pareto strength interval is invalid")
            development_text = (
                f"{_percent(development_strength, 'development strength')} "
                f"[{_percent(development_lower, 'development CI lower')}, "
                f"{_percent(development_upper, 'development CI upper')}] "
                f"(n={development_pairs})"
            )
            validation_text = (
                f"{_percent(strength, 'validation strength')} "
                f"[{_percent(validation_lower, 'validation CI lower')}, "
                f"{_percent(validation_upper, 'validation CI upper')}] "
                f"(n={validation_pairs})"
            )
        status = []
        if gate:
            status.append("constrained-Pareto" if optimal else "constrained-dominated")
        else:
            status.append("outside constrained frontier")
            status.append("gate-rejected")
        status.append(
            "unconstrained-Pareto" if unconstrained else "unconstrained-dominated"
        )
        if selected:
            status.append("selected")
        if fixed:
            status.append("fixed")
        if gate and not optimal:
            negative_findings.append(
                f"{display_identifier} was dominated on the frozen validation frontier."
            )
        if not gate:
            negative_findings.append(
                f"{display_identifier} missed the 50 ms gate and was excluded from "
                "the constrained frontier."
            )
        pareto_rows.append([
            display_identifier,
            development_text,
            validation_text,
            f"{p95:.3f} (n={latency_decisions})",
            ", ".join(status),
        ])
    lines.extend(_table(
        ("Configuration", "Development score [95% CI]", "Validation score [95% CI]",
         "Validation p95 ms (decision n)", "Status"),
        pareto_rows,
    ))

    supplied_findings = analysis.get("negative_findings", [])
    for value in _sequence(supplied_findings, "analysis.negative_findings"):
        negative_findings.append(_string(value, "negative finding"))
    negative_findings.extend(ablation_findings)
    negative_findings.extend(unresolved_findings)
    unique_findings = list(dict.fromkeys(negative_findings))
    lines.extend([
        "",
        "## Negative and statistically unresolved findings",
        "",
    ])
    if unique_findings:
        lines.extend(f"- {_markdown(finding)}" for finding in unique_findings)
    else:
        lines.append("- No additional preregistered negative finding was recorded.")

    lines.extend([
        "",
        "## Limitations and threats to validity",
        "",
        "- Latency is machine-, compiler-, power-, and thermal-state-specific; fixed work improves reproducibility but not cross-machine timing equivalence.",
        "- Frozen opening banks control color and opening variation but do not enumerate every reachable position.",
        "- Bradley–Terry abilities are relative to this four-entrant comparison graph and ruleset.",
        "- Calibration decisions within a game are dependent; prediction counts are not independent-game sample sizes.",
        "- The hand-versus-neural comparison changes the evaluator within a shared search family and does not isolate every implementation interaction.",
        f"- {studylib.PUBLIC_RANK5_LABEL} is measured only under demo rules and its fixed-work profile, as stated in the provenance disclaimer.",
        "- The frozen validation Pareto SVG has dense annotations at its native viewport; the immediately preceding table is the canonical readable listing of every point identity, sample size, interval, and status. The plot was not revised after test access.",
        "",
        "## Exact reproduction commands",
        "",
        "From a clean clone, the following launches a new from-source rerun with the "
        "same preregistered inputs. It does not reuse or overwrite the completed frozen "
        "test identity reported above:",
        "",
        "```bash",
        f"git switch -c reproduce-flagship-study {source_commit}",
        f"test \"$(python3 -c 'import platform; print(platform.python_version())')\" = \"{python_version}\"",
        "cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \\",
        "  -DPAPERSOCCER_ENABLE_SANITIZERS=OFF",
        "cmake --build build/release --parallel",
        f"test \"$(shasum -a 256 build/release/papersoccer_arena | awk '{{print $1}}')\" = \"{arena_sha256}\"",
        f"test \"$(shasum -a 256 build/release/papersoccer_opening_bank | awk '{{print $1}}')\" = \"{opening_tool_sha256}\"",
        "python3 benchmarks/flagship_study/prepare_manifest.py \\",
        "  --opening-tool build/release/papersoccer_opening_bank \\",
        f"  --source-commit {source_commit} \\",
        f"  --preregistered-at-utc {preregistered_at} \\",
        "  --fresh-validation-keep-frozen-test",
        "python3 benchmarks/flagship_study/run_study.py validate",
        "git add benchmarks/flagship_study/manifest.json benchmarks/flagship_study/openings",
        "git commit -m 'Freeze flagship manifest and opening banks'",
        "for index in 0 3 4 7 8 11 12 15 16 19 20 23 24 27 28 31 32 35; do",
        "  python3 benchmarks/flagship_study/run_study.py run --phase development \\",
        "    --arena build/release/papersoccer_arena --shard-count 36 --shard-index \"$index\"",
        "done",
        "python3 benchmarks/flagship_study/run_study.py project-runtime --write",
        "for index in $(seq 0 35); do",
        "  python3 benchmarks/flagship_study/run_study.py run --phase development \\",
        "    --arena build/release/papersoccer_arena --shard-count 36 --shard-index \"$index\"",
        "done",
        "python3 benchmarks/flagship_study/run_study.py aggregate --phase development",
        "for index in $(seq 0 35); do",
        "  python3 benchmarks/flagship_study/run_study.py run --phase validation \\",
        "    --arena build/release/papersoccer_arena --shard-count 36 --shard-index \"$index\"",
        "done",
        "python3 benchmarks/flagship_study/run_study.py aggregate --phase validation",
        "python3 benchmarks/flagship_study/run_study.py lock-selection",
        "git add benchmarks/flagship_study/data/development.json \\",
        "  benchmarks/flagship_study/data/validation.json \\",
        "  benchmarks/flagship_study/runtime_projection.json \\",
        "  benchmarks/flagship_study/selection_lock.json",
        "git commit -m 'Lock flagship validation selection'",
        "for index in $(seq 0 23); do",
        "  python3 benchmarks/flagship_study/run_study.py run --phase test \\",
        "    --arena build/release/papersoccer_arena --shard-count 24 --shard-index \"$index\"",
        "done",
        "python3 benchmarks/flagship_study/run_study.py aggregate --phase test",
        "python3 benchmarks/flagship_study/run_study.py analyze-test",
        "git add benchmarks/flagship_study/data/test.json \\",
        "  benchmarks/flagship_study/charts benchmarks/flagship_study/REPORT.md",
        "git commit -m 'Publish frozen flagship test analysis'",
        "```",
        "",
        "Each indexed test command resumes the same frozen run identity and refuses a "
        "second completed evaluation. No destructive override is used.",
        "",
        "## Artifact hashes",
        "",
        f"Source commit: `{source_commit}`",
        "",
    ])
    lines.extend(_table(
        ("Build executable", "SHA-256"),
        (
            ("Native arena", f"`{arena_sha256}`"),
            ("Opening-bank generator", f"`{opening_tool_sha256}`"),
        ),
    ))
    lines.extend([
        "",
        "### Observed validation execution environment",
        "",
    ])
    validation_environments = _sequence(
        selection.get("validation_execution_environments"),
        "selection.validation_execution_environments",
    )
    if not validation_environments:
        raise ReportError("selection lock lacks validation execution provenance")
    environment_rows = []
    for index, raw_environment in enumerate(validation_environments):
        observed = _mapping(raw_environment, f"validation environment {index}")
        provenance = _mapping(
            observed.get("build_provenance"),
            f"validation environment {index} build provenance",
        )
        if (_hash(observed.get("arena_sha256"), "observed arena SHA-256") !=
                arena_sha256):
            raise ReportError("validation arena hash differs from the manifest")
        if provenance.get("sanitizers_enabled") is not False:
            raise ReportError("validation provenance is sanitized or incomplete")
        environment_rows.append([
            str(index + 1),
            _string(observed.get("observed_at_utc"), "validation observation time"),
            _string(
                _mapping(
                    observed.get("gate_conditions_after"),
                    "validation ending gate conditions",
                ).get("observed_at_utc"),
                "validation ending observation time",
            ),
            _string(observed.get("processor"), "validation processor"),
            _string(observed.get("platform"), "validation platform"),
            _string(observed.get("python_version"), "validation Python version"),
            f"{_string(provenance.get('compiler_id'), 'validation compiler')} "
            f"{_string(provenance.get('compiler_version'), 'validation compiler version')}",
            _string(provenance.get("configured_flags"), "validation build flags"),
            (
                f"start: {_string(observed.get('power_source'), 'validation power source')}; "
                f"{_string(observed.get('power_settings'), 'validation power settings')}; "
                f"end: {_string(_mapping(observed.get('gate_conditions_after'), 'ending gate').get('power_source'), 'ending power source')}; "
                f"{_string(_mapping(observed.get('gate_conditions_after'), 'ending gate').get('power_settings'), 'ending power settings')}"
            ),
            (
                f"start: {_string(observed.get('thermal_status'), 'validation thermal status')}; "
                f"end: {_string(_mapping(observed.get('gate_conditions_after'), 'ending gate').get('thermal_status'), 'ending thermal status')}"
            ),
        ])
    lines.extend(_table(
        ("Run environment", "Start UTC", "End UTC", "CPU", "OS/kernel",
         "Python", "Compiler", "Build flags", "Power/settings start → end",
         "Thermal start → end"),
        environment_rows,
    ))
    lines.append("")
    hash_rows = []
    for path, digest in _artifact_rows(manifest, artifact_hashes):
        hash_rows.append([
            f"[{path}]({_relative_link(manifest, path)})",
            f"`{digest}`",
        ])
    lines.extend(_table(("Artifact", "SHA-256"), hash_rows))
    lines.extend([
        "",
        "Curated inputs:",
        "",
    ])
    for phase in ("development", "validation", "test"):
        target = _string(data_paths.get(phase), f"{phase} curated path")
        lines.append(f"- [{phase.title()} data]({_relative_link(manifest, target)})")
    lines.extend([
        f"- [Selection lock]({_relative_link(manifest, _string(outputs.get('selection_lock'), 'selection lock path'))})",
        "",
        "## Integrity",
        "",
        "Development, validation, and frozen test aggregation each report zero truncations and complete unique game sets. No truncated game entered paired strength, Bradley–Terry, or calibration calculations.",
        "",
    ])
    result = "\n".join(lines)
    if _PROHIBITED_BENCHMARK.search(result):
        raise ReportError("rendered report contains a prohibited benchmark label")
    return result


render_report = render_markdown_report


__all__ = ["ReportError", "render_markdown_report", "render_report"]
