#!/usr/bin/env python3
"""Generate compact, deterministic release summaries from the frozen study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "papersoccer.flagship-study-release-summary.v1"
MANIFEST_SCHEMA = "papersoccer.flagship-study-manifest.v2"
SELECTION_SCHEMA = "papersoccer.flagship-study-selection.v1"
CURATED_SCHEMA = "papersoccer.flagship-study-curated.v1"

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDY_ROOT = REPOSITORY_ROOT / "benchmarks/flagship_study"
DEFAULT_MANIFEST = STUDY_ROOT / "manifest.json"
DEFAULT_SELECTION = STUDY_ROOT / "selection_lock.json"
DEFAULT_TEST_DATA = STUDY_ROOT / "data/test.json"
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "summary"

OUTPUT_FILES = ("summary.json", "pairwise.csv", "configurations.csv")
FAMILIES = ("mcts", "alpha_beta", "jacek_inspired", "rank5_derived")
TUNABLE_FAMILIES = FAMILIES[:3]

PAIRWISE_FIELDS = (
    "matchup_id",
    "left_config_id",
    "left_label",
    "right_config_id",
    "right_label",
    "games",
    "pairs",
    "left_wins",
    "left_losses",
    "pairs_won_2_0",
    "pairs_split_1_1",
    "pairs_lost_0_2",
    "left_mean_pair_score",
    "ci_lower",
    "ci_upper",
    "classification",
    "stronger_config_id",
    "stronger_label",
    "conclusion",
)

CONFIGURATION_FIELDS = (
    "config_id",
    "family",
    "public_label",
    "kind",
    "role",
    "locked",
    "fixed",
    "work_kind",
    "work_budget",
    "validation_pair_score",
    "validation_strength_definition",
    "validation_ci_lower",
    "validation_ci_upper",
    "validation_pairs",
    "validation_p95_ms",
    "all_edge_p95_ms",
    "validation_latency_decisions",
    "gate_eligible",
    "constrained_pareto_optimal",
    "unconstrained_pareto_optimal",
    "settings_json",
)


class ReleaseSummaryError(ValueError):
    """Raised when frozen inputs cannot support the release summary."""


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseSummaryError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReleaseSummaryError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseSummaryError(f"{where} must be a non-empty string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ReleaseSummaryError(f"{where} must be a boolean")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseSummaryError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseSummaryError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReleaseSummaryError(f"{where} must be a finite number")
    return result


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ReleaseSummaryError(f"{where} must be a lowercase SHA-256")
    return result


def _read_json(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseSummaryError(f"could not read JSON {path}: {error}") from error
    return _object(value, str(path))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReleaseSummaryError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _configurations(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_array(manifest.get("configurations"), "configurations")):
        config = _object(raw, f"configurations[{index}]")
        config_id = _string(config.get("id"), f"configurations[{index}].id")
        if config_id in result:
            raise ReleaseSummaryError(f"duplicate configuration ID: {config_id}")
        family = _string(config.get("family"), f"{config_id}.family")
        if family not in FAMILIES:
            raise ReleaseSummaryError(f"unsupported configuration family: {family}")
        _string(config.get("public_label"), f"{config_id}.public_label")
        _string(config.get("kind"), f"{config_id}.kind")
        _string(config.get("role"), f"{config_id}.role")
        _object(config.get("settings"), f"{config_id}.settings")
        result[config_id] = config
    return result


def _selected_ids(
    selection: Mapping[str, Any], configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    raw_selected = _object(
        selection.get("selected_configurations"), "selected_configurations"
    )
    if set(raw_selected) != set(TUNABLE_FAMILIES):
        raise ReleaseSummaryError(
            "selected_configurations must contain the three tunable families"
        )
    selected = {
        family: _string(raw_selected.get(family), f"selected_configurations.{family}")
        for family in TUNABLE_FAMILIES
    }
    selected["rank5_derived"] = _string(
        selection.get("fixed_rank5_configuration"), "fixed_rank5_configuration"
    )
    if len(set(selected.values())) != len(FAMILIES):
        raise ReleaseSummaryError("locked configuration IDs must be unique")
    for family, config_id in selected.items():
        if config_id not in configs or configs[config_id].get("family") != family:
            raise ReleaseSummaryError(f"locked configuration does not match {family}: {config_id}")
    if configs[selected["rank5_derived"]].get("role") != "fixed_comparator":
        raise ReleaseSummaryError("Rank5Derived lock must identify the fixed comparator")
    return selected


def _validate_provenance(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ReleaseSummaryError("unsupported manifest schema")
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ReleaseSummaryError("unsupported selection-lock schema")
    if test.get("schema_version") != CURATED_SCHEMA:
        raise ReleaseSummaryError("unsupported curated-data schema")
    manifest_hash = _sha256(source_hashes.get("manifest"), "manifest file hash")
    _sha256(source_hashes.get("selection_lock"), "selection-lock file hash")
    _sha256(source_hashes.get("test_data"), "test-data file hash")
    if selection.get("manifest_sha256") != manifest_hash:
        raise ReleaseSummaryError("selection lock does not match the manifest")
    if test.get("manifest_sha256") != manifest_hash:
        raise ReleaseSummaryError("test data does not match the manifest")
    if selection.get("source_phase") != "validation":
        raise ReleaseSummaryError("configuration lock must come from validation")
    if not _boolean(selection.get("test_authorized"), "test_authorized"):
        raise ReleaseSummaryError("selection lock does not authorize the frozen test")
    if test.get("phase") != "test" or not _boolean(
        test.get("analysis_complete"), "analysis_complete"
    ):
        raise ReleaseSummaryError("test analysis is not complete")
    study = _object(manifest.get("study"), "study")
    if not _boolean(study.get("frozen"), "study.frozen"):
        raise ReleaseSummaryError("release summaries require a frozen study")


def _test_counts(test: Mapping[str, Any]) -> dict[str, Any]:
    completeness = _object(test.get("completeness"), "test.completeness")
    expected = _integer(completeness.get("expected_games"), "expected games", minimum=1)
    completed = _integer(completeness.get("completed_games"), "completed games", minimum=1)
    unique = _integer(completeness.get("unique_game_ids"), "unique games", minimum=1)
    truncations = _integer(completeness.get("truncations"), "truncations")
    if expected != completed or completed != unique:
        raise ReleaseSummaryError("test game counts are incomplete")
    if not _boolean(completeness.get("operationally_valid"), "operationally_valid"):
        raise ReleaseSummaryError("test results are not operationally valid")
    if truncations != 0:
        raise ReleaseSummaryError("test results contain truncations")

    sample_sizes = _object(test.get("sample_sizes"), "sample_sizes")
    games = _integer(sample_sizes.get("games"), "sample_sizes.games", minimum=1)
    pairs = _integer(sample_sizes.get("pairs"), "sample_sizes.pairs", minimum=1)
    depths = [
        _integer(value, "opening depth", minimum=1)
        for value in _array(sample_sizes.get("opening_depths"), "opening_depths")
    ]
    resamples = _integer(
        sample_sizes.get("bootstrap_resamples"), "bootstrap resamples", minimum=1
    )
    if games != completed or games != 2 * pairs or not depths:
        raise ReleaseSummaryError("test sample sizes are inconsistent")
    if len(_array(test.get("binary_games"), "binary_games")) != games:
        raise ReleaseSummaryError("binary-game count does not match test completeness")
    return {
        "games": games,
        "pairs": pairs,
        "truncations": truncations,
        "opening_depths": depths,
        "bootstrap_resamples": resamples,
    }


def _work(config: Mapping[str, Any]) -> tuple[str, int]:
    settings = _object(config.get("settings"), f"{config.get('id')}.settings")
    if config.get("kind") == "mcts":
        return "iterations", _integer(settings.get("iterations"), "MCTS iterations", minimum=1)
    return "nodes", _integer(settings.get("max_nodes"), "node budget", minimum=1)


def _configuration_rows(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, str],
) -> list[dict[str, Any]]:
    pareto_values = _array(selection.get("validation_pareto"), "validation_pareto")
    pareto: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(pareto_values):
        row = _object(raw, f"validation_pareto[{index}]")
        config_id = _string(row.get("id"), f"validation_pareto[{index}].id")
        if config_id in pareto:
            raise ReleaseSummaryError(f"duplicate validation configuration: {config_id}")
        pareto[config_id] = row
    if set(pareto) != set(configs):
        raise ReleaseSummaryError("validation configuration summary is incomplete")

    metrics = _object(selection.get("validation_metrics"), "validation_metrics")
    rank5_latency = _object(selection.get("rank5_latency"), "rank5_latency")
    gate_ms = _number(
        _object(manifest.get("latency_protocol"), "latency_protocol").get("gate_ms"),
        "latency gate",
    )
    locked_ids = set(selected.values())
    rows: list[dict[str, Any]] = []
    for config_id, config in configs.items():
        result = pareto[config_id]
        family = _string(result.get("family"), f"{config_id}.validation family")
        if family != config.get("family"):
            raise ReleaseSummaryError(f"validation family changed for {config_id}")
        p95 = _number(result.get("validation_p95_ms"), f"{config_id}.validation p95")
        gate_eligible = _boolean(result.get("gate_eligible"), f"{config_id}.gate_eligible")
        if gate_eligible != (p95 <= gate_ms):
            raise ReleaseSummaryError(f"latency gate classification is inconsistent for {config_id}")
        is_fixed = family == "rank5_derived"
        is_locked = config_id in locked_ids
        if _boolean(result.get("selected"), f"{config_id}.selected") != is_locked:
            raise ReleaseSummaryError(f"selection status is inconsistent for {config_id}")
        if is_locked and not is_fixed and not gate_eligible:
            raise ReleaseSummaryError(f"locked tunable configuration missed the gate: {config_id}")
        if is_fixed:
            if config_id != selected["rank5_derived"] or not _boolean(
                result.get("fixed"), f"{config_id}.fixed"
            ):
                raise ReleaseSummaryError("fixed Rank5Derived validation row changed")
            if not math.isclose(
                p95,
                _number(rank5_latency.get("fresh_root_p95_ms"), "Rank5 fresh-root p95"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ReleaseSummaryError("Rank5 fresh-root latency disagrees with Pareto data")
            if _boolean(
                rank5_latency.get("eligible_under_50_ms"), "Rank5 eligibility"
            ) != gate_eligible:
                raise ReleaseSummaryError("Rank5 eligibility disagrees with Pareto data")
            all_edge_p95: float | None = _number(
                rank5_latency.get("all_edge_p95_ms"), "Rank5 all-edge p95"
            )
        else:
            metric = _object(metrics.get(config_id), f"validation_metrics.{config_id}")
            if (
                metric.get("family") != family
                or _boolean(metric.get("eligible"), f"{config_id}.eligible") != gate_eligible
                or not math.isclose(
                    _number(metric.get("validation_p95_ms"), f"{config_id}.metric p95"),
                    p95,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ReleaseSummaryError(f"validation lock disagrees for {config_id}")
            all_edge_p95 = None

        score = _number(result.get("validation_strength"), f"{config_id}.validation score")
        if not 0.0 <= score <= 1.0:
            raise ReleaseSummaryError(f"validation score is outside [0,1] for {config_id}")
        strength_definition = result.get("strength_definition")
        if strength_definition is not None:
            strength_definition = _string(
                strength_definition, f"{config_id}.strength_definition"
            )
        if is_fixed and strength_definition != "defined common-opponent reference level":
            raise ReleaseSummaryError("Rank5 validation reference definition changed")
        if not is_fixed and strength_definition is not None:
            raise ReleaseSummaryError(
                f"observed candidate strength has a reference definition: {config_id}"
            )
        raw_interval = result.get("validation_strength_pair_bootstrap_95")
        raw_pairs = result.get("validation_strength_pairs")
        if raw_interval is None:
            lower = upper = None
            if not is_fixed or raw_pairs is not None:
                raise ReleaseSummaryError(
                    f"validation uncertainty is missing for observed candidate {config_id}"
                )
            validation_pairs: int | None = None
        else:
            interval = _object(raw_interval, f"{config_id}.validation interval")
            lower = _number(interval.get("lower"), f"{config_id}.validation CI lower")
            upper = _number(interval.get("upper"), f"{config_id}.validation CI upper")
            if not 0.0 <= lower <= score <= upper <= 1.0:
                raise ReleaseSummaryError(f"invalid validation interval for {config_id}")
            validation_pairs = _integer(
                raw_pairs, f"{config_id}.validation pairs", minimum=1
            )
        work_kind, work_budget = _work(config)
        rows.append({
            "config_id": config_id,
            "family": family,
            "public_label": _string(config.get("public_label"), f"{config_id}.public_label"),
            "kind": _string(config.get("kind"), f"{config_id}.kind"),
            "role": _string(config.get("role"), f"{config_id}.role"),
            "locked": is_locked,
            "fixed": is_fixed,
            "work_kind": work_kind,
            "work_budget": work_budget,
            "validation_pair_score": score,
            "validation_strength_definition": strength_definition,
            "validation_ci_lower": lower,
            "validation_ci_upper": upper,
            "validation_pairs": validation_pairs,
            "validation_p95_ms": p95,
            "all_edge_p95_ms": all_edge_p95,
            "validation_latency_decisions": _integer(
                result.get("validation_latency_decisions"),
                f"{config_id}.validation latency decisions",
                minimum=1,
            ),
            "gate_eligible": gate_eligible,
            "constrained_pareto_optimal": _boolean(
                result.get("constrained_pareto_optimal"),
                f"{config_id}.constrained Pareto status",
            ),
            "unconstrained_pareto_optimal": _boolean(
                result.get("unconstrained_pareto_optimal"),
                f"{config_id}.unconstrained Pareto status",
            ),
            "settings": dict(_object(config.get("settings"), f"{config_id}.settings")),
        })
    family_index = {family: index for index, family in enumerate(FAMILIES)}
    rows.sort(key=lambda row: (family_index[row["family"]], row["work_budget"], row["config_id"]))
    return rows


def _resolve_slot(slot: str, selected: Mapping[str, str]) -> str:
    if slot == "fixed:rank5_derived":
        return selected["rank5_derived"]
    if slot.startswith("selected:"):
        family = slot.removeprefix("selected:")
        if family in TUNABLE_FAMILIES:
            return selected[family]
    raise ReleaseSummaryError(f"unsupported test schedule slot: {slot}")


def _pairwise_rows(
    manifest: Mapping[str, Any],
    test: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, str],
) -> list[dict[str, Any]]:
    schedule = _array(
        _object(manifest.get("schedule"), "schedule").get("test"), "schedule.test"
    )
    matchups = _object(test.get("matchups"), "test.matchups")
    schedule_ids = [
        _string(_object(raw, "test schedule row").get("id"), "test schedule ID")
        for raw in schedule
    ]
    if len(set(schedule_ids)) != len(schedule_ids) or set(schedule_ids) != set(matchups):
        raise ReleaseSummaryError("test matchup schedule is incomplete or duplicated")

    rows: list[dict[str, Any]] = []
    for raw_schedule in schedule:
        scheduled = _object(raw_schedule, "test schedule row")
        matchup_id = _string(scheduled.get("id"), "test matchup ID")
        result = _object(matchups[matchup_id], f"matchups.{matchup_id}")
        left_id = _string(result.get("left_config_id"), f"{matchup_id}.left_config_id")
        right_id = _string(result.get("right_config_id"), f"{matchup_id}.right_config_id")
        expected = (
            _resolve_slot(_string(scheduled.get("left_slot"), "left slot"), selected),
            _resolve_slot(_string(scheduled.get("right_slot"), "right slot"), selected),
        )
        if (left_id, right_id) != expected:
            raise ReleaseSummaryError(f"{matchup_id} participants differ from the lock")
        games = _integer(result.get("games"), f"{matchup_id}.games", minimum=1)
        pairs = _integer(result.get("pairs"), f"{matchup_id}.pairs", minimum=1)
        left_wins = _integer(result.get("left_wins"), f"{matchup_id}.left_wins")
        left_losses = _integer(result.get("left_losses"), f"{matchup_id}.left_losses")
        won = _integer(result.get("pairs_won_2_0"), f"{matchup_id}.pairs_won_2_0")
        split = _integer(result.get("pairs_split_1_1"), f"{matchup_id}.pairs_split_1_1")
        lost = _integer(result.get("pairs_lost_0_2"), f"{matchup_id}.pairs_lost_0_2")
        truncations = _integer(result.get("truncations"), f"{matchup_id}.truncations")
        if truncations != 0:
            raise ReleaseSummaryError(f"{matchup_id} contains truncations")
        if (
            games != 2 * pairs
            or left_wins + left_losses != games
            or won + split + lost != pairs
            or left_wins != 2 * won + split
            or left_losses != 2 * lost + split
        ):
            raise ReleaseSummaryError(f"pair accounting is inconsistent in {matchup_id}")
        score = _number(result.get("mean_pair_score"), f"{matchup_id}.mean_pair_score")
        if not math.isclose(score, (won + 0.5 * split) / pairs, rel_tol=0.0, abs_tol=1e-12):
            raise ReleaseSummaryError(f"pair score is inconsistent in {matchup_id}")
        interval = _object(result.get("pair_bootstrap_95"), f"{matchup_id}.interval")
        lower = _number(interval.get("lower"), f"{matchup_id}.CI lower")
        upper = _number(interval.get("upper"), f"{matchup_id}.CI upper")
        if not 0.0 <= lower <= score <= upper <= 1.0:
            raise ReleaseSummaryError(f"pair interval is invalid in {matchup_id}")

        if lower > 0.5:
            classification = "left_stronger"
            stronger_id: str | None = left_id
        elif upper < 0.5:
            classification = "right_stronger"
            stronger_id = right_id
        else:
            classification = "statistically_unresolved"
            stronger_id = None
        source_conclusion = _object(result.get("conclusion"), f"{matchup_id}.conclusion")
        if source_conclusion.get("stronger_config_id") != stronger_id:
            raise ReleaseSummaryError(f"published conclusion is inconsistent in {matchup_id}")
        expected_source_class = (
            "statistically_unresolved" if stronger_id is None else "stronger"
        )
        if source_conclusion.get("classification") != expected_source_class:
            raise ReleaseSummaryError(
                f"published conclusion classification is inconsistent in {matchup_id}"
            )

        left_label = _string(configs[left_id].get("public_label"), f"{left_id}.label")
        right_label = _string(configs[right_id].get("public_label"), f"{right_id}.label")
        stronger_label = (
            _string(configs[stronger_id].get("public_label"), f"{stronger_id}.label")
            if stronger_id is not None
            else None
        )
        conclusion = (
            f"{stronger_label} is stronger than "
            f"{right_label if stronger_id == left_id else left_label}."
            if stronger_id is not None
            else f"{left_label} versus {right_label} is statistically unresolved."
        )
        rows.append({
            "matchup_id": matchup_id,
            "left_config_id": left_id,
            "left_label": left_label,
            "right_config_id": right_id,
            "right_label": right_label,
            "games": games,
            "pairs": pairs,
            "left_wins": left_wins,
            "left_losses": left_losses,
            "pairs_won_2_0": won,
            "pairs_split_1_1": split,
            "pairs_lost_0_2": lost,
            "left_mean_pair_score": score,
            "ci_lower": lower,
            "ci_upper": upper,
            "classification": classification,
            "stronger_config_id": stronger_id,
            "stronger_label": stronger_label,
            "conclusion": conclusion,
        })
    return rows


def build_release_summary(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate frozen inputs and return the compact release contract."""

    _validate_provenance(manifest, selection, test, source_hashes)
    configs = _configurations(manifest)
    selected = _selected_ids(selection, configs)
    counts = _test_counts(test)
    configuration_rows = _configuration_rows(manifest, selection, configs, selected)
    pairwise_rows = _pairwise_rows(manifest, test, configs, selected)
    if sum(row["games"] for row in pairwise_rows) != counts["games"]:
        raise ReleaseSummaryError("pairwise game counts do not match the test total")
    if sum(row["pairs"] for row in pairwise_rows) != counts["pairs"]:
        raise ReleaseSummaryError("pairwise pair counts do not match the test total")

    study = _object(manifest.get("study"), "study")
    source = _object(manifest.get("source"), "source")
    protected = _object(source.get("protected_artifacts"), "protected_artifacts")
    outputs = _object(manifest.get("outputs"), "outputs")
    data_paths = _object(outputs.get("curated_data"), "curated_data paths")
    rank5_hash = _sha256(
        protected.get("rank5_submission_sha256"), "rank5 submission SHA-256"
    )
    model_hash = _sha256(protected.get("jacek_model_sha256"), "neural model SHA-256")
    rank5_settings = _object(
        configs[selected["rank5_derived"]].get("settings"), "Rank5 settings"
    )
    neural_settings = _object(
        configs[selected["jacek_inspired"]].get("settings"), "neural settings"
    )
    if rank5_settings.get("original_artifact_sha256") != rank5_hash:
        raise ReleaseSummaryError("Rank5Derived profile does not match protected source identity")
    if neural_settings.get("model_sha256") != model_hash:
        raise ReleaseSummaryError("selected neural profile does not match protected model identity")

    locked = []
    for family in FAMILIES:
        config_id = selected[family]
        row = next(value for value in configuration_rows if value["config_id"] == config_id)
        locked.append({
            "family": family,
            "config_id": config_id,
            "public_label": row["public_label"],
            "fixed": row["fixed"],
            "work_kind": row["work_kind"],
            "work_budget": row["work_budget"],
            "validation_p95_ms": row["validation_p95_ms"],
            "all_edge_p95_ms": row["all_edge_p95_ms"],
            "gate_eligible": row["gate_eligible"],
            "settings": row["settings"],
        })

    canonical_manifest_path = "benchmarks/flagship_study/manifest.json"
    canonical_selection_path = _string(outputs.get("selection_lock"), "selection path")
    canonical_test_path = _string(data_paths.get("test"), "test-data path")
    git_commit = _string(source.get("git_commit"), "source git commit")
    if len(git_commit) != 40 or any(character not in "0123456789abcdef" for character in git_commit):
        raise ReleaseSummaryError("source git commit must be a lowercase 40-character hash")
    return {
        "schema_version": SCHEMA,
        "study": {
            "id": _string(study.get("id"), "study.id"),
            "version": _string(study.get("version"), "study.version"),
            "title": _string(study.get("title"), "study.title"),
            "frozen": True,
            "rank5_disclaimer": _string(
                study.get("rank5_disclaimer"), "study.rank5_disclaimer"
            ),
            "latency_gate_ms": _number(
                _object(manifest.get("latency_protocol"), "latency_protocol").get("gate_ms"),
                "latency gate",
            ),
        },
        "provenance": {
            "source_git_commit": git_commit,
            "source_artifacts": {
                "manifest": {
                    "path": canonical_manifest_path,
                    "sha256": _sha256(source_hashes.get("manifest"), "manifest hash"),
                },
                "selection_lock": {
                    "path": canonical_selection_path,
                    "sha256": _sha256(
                        source_hashes.get("selection_lock"), "selection-lock hash"
                    ),
                },
                "test_data": {
                    "path": canonical_test_path,
                    "sha256": _sha256(source_hashes.get("test_data"), "test-data hash"),
                },
            },
            "analysis_contract_sha256": _sha256(
                source.get("analysis_contract_sha256"), "analysis-contract hash"
            ),
            "rank5_submission": {
                "path": _string(
                    protected.get("rank5_submission_path"), "rank5 submission path"
                ),
                "sha256": rank5_hash,
            },
            "neural_model": {
                "path": _string(protected.get("jacek_model_path"), "neural model path"),
                "sha256": model_hash,
            },
        },
        "test": counts,
        "locked_configurations": locked,
        "pairwise_results": pairwise_rows,
    }


def _render_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(
        summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return stream.getvalue()


def render_release_files(
    summary: Mapping[str, Any],
    configuration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    csv_configurations = []
    for raw in configuration_rows:
        row = dict(raw)
        row["settings_json"] = json.dumps(
            row.pop("settings"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        csv_configurations.append(row)
    pairwise_rows = _array(summary.get("pairwise_results"), "pairwise_results")
    return {
        "summary.json": _render_json(summary),
        "pairwise.csv": _render_csv(PAIRWISE_FIELDS, pairwise_rows),
        "configurations.csv": _render_csv(
            CONFIGURATION_FIELDS, csv_configurations
        ),
    }


def generate_release_files(
    manifest_path: pathlib.Path = DEFAULT_MANIFEST,
    selection_path: pathlib.Path = DEFAULT_SELECTION,
    test_path: pathlib.Path = DEFAULT_TEST_DATA,
) -> dict[str, str]:
    manifest = _read_json(manifest_path)
    selection = _read_json(selection_path)
    test = _read_json(test_path)
    hashes = {
        "manifest": _sha256_file(manifest_path),
        "selection_lock": _sha256_file(selection_path),
        "test_data": _sha256_file(test_path),
    }
    summary = build_release_summary(
        manifest, selection, test, source_hashes=hashes
    )
    configs = _configurations(manifest)
    configuration_rows = _configuration_rows(
        manifest,
        selection,
        configs,
        _selected_ids(selection, configs),
    )
    return render_release_files(summary, configuration_rows)


def _atomic_write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write compact release files")
    action.add_argument("--check", action="store_true", help="fail if release files are stale")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-lock", type=pathlib.Path, default=DEFAULT_SELECTION)
    parser.add_argument("--test-data", type=pathlib.Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected = generate_release_files(
            args.manifest, args.selection_lock, args.test_data
        )
        if args.write:
            for name in OUTPUT_FILES:
                _atomic_write(args.output_dir / name, expected[name])
            return 0
        stale = []
        for name in OUTPUT_FILES:
            path = args.output_dir / name
            try:
                actual = path.read_text(encoding="utf-8")
            except OSError:
                stale.append(name)
                continue
            if actual != expected[name]:
                stale.append(name)
        if stale:
            raise ReleaseSummaryError(
                "stale release summary files: "
                + ", ".join(stale)
                + "; run release_summary.py --write"
            )
        return 0
    except ReleaseSummaryError as error:
        print(f"release summary error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
