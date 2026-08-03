"""Manifest, execution, and analysis support for the flagship bot study.

The full study deliberately uses only the Python standard library.  That keeps
the statistical implementation reviewable and makes the checked-in dependency
contract exact: Python itself is the analysis runtime, while the bots execute
in the optimized native arena.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "papersoccer.flagship-study-manifest.v1"
SELECTION_SCHEMA_VERSION = "papersoccer.flagship-study-selection.v1"
CURATED_SCHEMA_VERSION = "papersoccer.flagship-study-curated.v1"
OPENING_BANK_SCHEMA_VERSION = "papersoccer.opening-bank.v1"
RUNTIME_PROJECTION_SCHEMA_VERSION = (
    "papersoccer.flagship-study-runtime-projection.v2"
)
PUBLIC_RANK5_LABEL = "Rank5DerivedBot — fixed 50k demo profile"
PUBLIC_LABELS = {
    "mcts": "Tactical MctsBot",
    "alpha_beta": "Hand-evaluated AlphaBetaBot",
    "jacek_inspired": "Neural alpha-beta (JacekInspiredBot)",
    "rank5_derived": PUBLIC_RANK5_LABEL,
}
RANK5_DISCLAIMER = (
    "Rank5DerivedBot adapts search code from the rank 5/206 CodinGame "
    "submission to different demo rules and a fixed-work profile. These "
    "measurements do not evaluate the authentic ranked submission."
)
RANK5_SOURCE_SHA256 = (
    "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
)
FULL_PHASES = ("development", "validation", "test")
TUNABLE_FAMILIES = ("mcts", "alpha_beta", "jacek_inspired")
EXPECTED_OPENING_DEPTHS = (4, 8, 12, 20)
EXPECTED_PAIR_COUNTS = {"development": 25, "validation": 50, "test": 100}
RANK5_FRESH_COUNTER_FIELDS = (
    "leaf_evaluations",
    "terminal_nodes",
    "completed_actions",
    "cutoffs",
    "transposition_probes",
    "transposition_hits",
    "transposition_cutoffs",
    "transposition_stores",
    "continuation_transposition_hits",
    "evaluation_cache_probes",
    "evaluation_cache_hits",
    "terminal_bound_cutoffs",
    "forced_edges",
    "root_seed_actions",
    "root_transposition_reuses",
    "max_action_edges",
)
CALIBRATION_OBSERVATION_KEYS = {
    "schema",
    "phase",
    "bot_id",
    "score_kind",
    "score_perspective",
    "decision_count",
    "scores",
    "outcomes",
    "pair_cluster_ids",
    "stratum_ids",
    "excluded",
}
CALIBRATION_EXCLUSION_KEYS = {
    "cached_continuations",
    "truncations",
    "invalid_depths",
}
ARENA_BUILD_PROVENANCE_KEYS = {
    "schema",
    "runtime",
    "build_type",
    "ndebug",
    "sanitizers_enabled",
    "compiler_id",
    "compiler_version",
    "configured_flags",
    "cxx_standard",
    "source_commit",
    "source_dirty",
}


class StudyError(RuntimeError):
    """Raised for an invalid or unsafe study operation."""


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"could not read JSON {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise StudyError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def manifest_sha256(path: pathlib.Path) -> str:
    return sha256_file(path)


def write_json_atomic(path: pathlib.Path, value: Any, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise StudyError(f"refusing to overwrite existing file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudyError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise StudyError(f"{where} must be an array")
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    obj = _object(value, where)
    actual = set(obj)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise StudyError(f"{where} has " + " and ".join(details))
    return obj


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise StudyError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StudyError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise StudyError(f"{where} must be at least {minimum}")
    return value


def validate_arena_build_provenance(value: Any) -> dict[str, Any]:
    provenance = _exact_keys(
        value, ARENA_BUILD_PROVENANCE_KEYS, "arena build provenance"
    )
    for key in (
        "schema", "runtime", "build_type", "compiler_id",
        "compiler_version", "configured_flags", "source_commit",
    ):
        _string(provenance.get(key), f"arena build provenance {key}")
    for key in ("ndebug", "sanitizers_enabled", "source_dirty"):
        if not isinstance(provenance.get(key), bool):
            raise StudyError(f"arena build provenance {key} must be boolean")
    _integer(
        provenance.get("cxx_standard"),
        "arena build provenance cxx_standard",
        1,
    )
    return provenance


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise StudyError(f"{where} must be finite")
    return result


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise StudyError(f"{where} must be a boolean")
    return value


def _seed(value: Any, where: str) -> int:
    text = _string(value, where)
    if not text.isdecimal():
        raise StudyError(f"{where} must be an unsigned decimal string")
    parsed = int(text)
    if parsed < 0 or parsed > (1 << 64) - 1:
        raise StudyError(f"{where} is outside uint64 range")
    return parsed


def _sha256(value: Any, where: str) -> str:
    text = _string(value, where)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StudyError(f"{where} must be a lowercase SHA-256 hex digest")
    return text


def _repository_relative_path(repository: pathlib.Path, value: Any,
                              where: str) -> pathlib.Path:
    text = _string(value, where)
    relative = pathlib.Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise StudyError(f"{where} must be a repository-relative path without '..'")
    resolved = (repository / relative).resolve()
    repository_resolved = repository.resolve()
    if resolved == repository_resolved or repository_resolved not in resolved.parents:
        raise StudyError(f"{where} resolves outside the repository")
    return resolved


def _unique_ids(values: Sequence[Mapping[str, Any]], where: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(values):
        identifier = _string(value.get("id"), f"{where}[{index}].id")
        if identifier in seen:
            raise StudyError(f"duplicate ID {identifier!r} in {where}")
        seen.add(identifier)


def _validate_configuration(config: Any, index: int, study_class: str) -> None:
    where = f"configurations[{index}]"
    obj = _exact_keys(
        config, {"id", "family", "kind", "public_label", "role", "settings"}, where
    )
    identifier = _string(obj["id"], f"{where}.id")
    family = _string(obj["family"], f"{where}.family")
    kind = _string(obj["kind"], f"{where}.kind")
    label = _string(obj["public_label"], f"{where}.public_label")
    role = _string(obj["role"], f"{where}.role")
    if "random" in identifier.lower() or "random" in kind.lower() or "randombot" in label.lower():
        raise StudyError("RandomBot cannot be a study configuration")
    if family not in (*TUNABLE_FAMILIES, "rank5_derived"):
        raise StudyError(f"{where}.family is unsupported")
    if kind not in ("mcts", "alpha-beta", "jacek-inspired", "rank5-derived"):
        raise StudyError(f"{where}.kind is unsupported")
    if role not in ("candidate", "fixed_comparator"):
        raise StudyError(f"{where}.role is unsupported")
    expected_kind = {
        "mcts": "mcts",
        "alpha_beta": "alpha-beta",
        "jacek_inspired": "jacek-inspired",
        "rank5_derived": "rank5-derived",
    }[family]
    expected_role = "fixed_comparator" if family == "rank5_derived" else "candidate"
    if kind != expected_kind or role != expected_role:
        raise StudyError(f"{where} has an incompatible family/kind/role combination")
    if study_class == "flagship" and label != PUBLIC_LABELS[family]:
        raise StudyError(f"{where}.public_label differs from the frozen family label")

    settings = _object(obj["settings"], f"{where}.settings")
    if kind == "mcts":
        expected = {
            "iterations", "exploration", "rollout_policy", "leaf_policy",
            "reuse_tree", "node_capacity", "quiescence_enabled",
            "quiescence_max_depth", "quiescence_max_nodes", "wall_clock_limit_ms",
            "seed_derivation",
        }
        settings = _exact_keys(settings, expected, f"{where}.settings")
        _integer(settings["iterations"], f"{where}.settings.iterations", 1)
        _number(settings["exploration"], f"{where}.settings.exploration")
        if settings["rollout_policy"] != "tactical":
            raise StudyError("flagship MCTS must use tactical rollouts")
        if settings["leaf_policy"] != "rollout_only":
            raise StudyError("flagship MCTS must use the maintained rollout-only leaf policy")
        if settings["reuse_tree"] is not True or settings["quiescence_enabled"] is not False:
            raise StudyError("flagship MCTS tree reuse/quiescence settings are incompatible")
        _integer(settings["node_capacity"], f"{where}.settings.node_capacity", 2)
        _integer(settings["quiescence_max_depth"], f"{where}.settings.quiescence_max_depth", 1)
        _integer(settings["quiescence_max_nodes"], f"{where}.settings.quiescence_max_nodes", 1)
        if settings["wall_clock_limit_ms"] != 0:
            raise StudyError("MCTS wall-clock limit must be disabled")
        if study_class == "flagship" and settings != {
            "iterations": settings["iterations"],
            "exploration": 1.4142135623730951,
            "rollout_policy": "tactical",
            "leaf_policy": "rollout_only",
            "reuse_tree": True,
            "node_capacity": 65_536,
            "quiescence_enabled": False,
            "quiescence_max_depth": 8,
            "quiescence_max_nodes": 256,
            "wall_clock_limit_ms": 0,
            "seed_derivation": "sha256-domain-separated-uint64/v1",
        }:
            raise StudyError(f"{where} changes a frozen MCTS setting")
    elif kind in ("alpha-beta", "jacek-inspired"):
        expected = {
            "max_turn_depth", "max_nodes", "transposition_table_entries",
            "max_search_plies", "wall_clock_limit_ms", "seed_ignored",
        }
        if kind == "jacek-inspired":
            expected |= {"model_path", "model_sha256"}
        settings = _exact_keys(settings, expected, f"{where}.settings")
        _integer(settings["max_turn_depth"], f"{where}.settings.max_turn_depth", 1)
        if study_class == "flagship" and settings["max_turn_depth"] != 6:
            raise StudyError(f"{where} must use alpha-beta depth 6")
        _integer(settings["max_nodes"], f"{where}.settings.max_nodes", 1)
        _integer(settings["transposition_table_entries"],
                 f"{where}.settings.transposition_table_entries", 1)
        _integer(settings["max_search_plies"], f"{where}.settings.max_search_plies", 1)
        if settings["wall_clock_limit_ms"] != 0 or settings["seed_ignored"] is not True:
            raise StudyError(f"{where} must disable wall time and ignore seed")
        if study_class == "flagship" and (
                settings["transposition_table_entries"] != 65_536 or
                settings["max_search_plies"] != 12):
            raise StudyError(f"{where} changes a frozen alpha-beta setting")
        if kind == "jacek-inspired":
            _string(settings["model_path"], f"{where}.settings.model_path")
            _sha256(settings["model_sha256"], f"{where}.settings.model_sha256")
    else:
        expected = {
            "max_turn_depth", "max_nodes", "transposition_table_entries",
            "evaluation_cache_entries", "wall_clock_limit_ms",
            "replay_corrections", "learned_value_blend_percent", "seed_ignored",
            "rules_profile", "original_artifact_sha256",
        }
        settings = _exact_keys(settings, expected, f"{where}.settings")
        locked = {
            "max_turn_depth": 32,
            "max_nodes": 50_000,
            "transposition_table_entries": 65_536,
            "evaluation_cache_entries": 32_768,
            "wall_clock_limit_ms": 0,
            "replay_corrections": False,
            "learned_value_blend_percent": 0,
            "seed_ignored": True,
            "rules_profile": "standard-8x10-demo",
            "original_artifact_sha256": RANK5_SOURCE_SHA256,
        }
        if settings != locked:
            raise StudyError("Rank5Derived settings differ from the fixed 50k demo profile")
        if label != PUBLIC_RANK5_LABEL or role != "fixed_comparator":
            raise StudyError("Rank5Derived public identity is not hard-locked")


def validate_manifest(manifest: Any, repository: pathlib.Path,
                      *, verify_files: bool = True) -> dict[str, Any]:
    """Strictly validate a full or CI-smoke study manifest.

    Unknown fields are rejected at every structural level used by the study.
    Semantic checks intentionally go beyond the accompanying JSON Schema.
    """

    top = _exact_keys(
        manifest,
        {
            "schema_version", "study", "source", "rules", "configurations",
            "candidate_grids", "openings", "seeds", "samples", "schedule",
            "latency_protocol", "selection_rule", "statistics", "outputs",
            "environment",
        },
        "manifest",
    )
    if top["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise StudyError("unsupported manifest schema_version")

    study = _exact_keys(
        top["study"],
        {"id", "version", "title", "study_class", "preregistered_at_utc", "frozen",
         "public_labels", "rank5_disclaimer"},
        "study",
    )
    _string(study["id"], "study.id")
    _string(study["version"], "study.version")
    _string(study["title"], "study.title")
    timestamp = _string(study["preregistered_at_utc"], "study.preregistered_at_utc")
    try:
        parsed_timestamp = dt.datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise StudyError("study.preregistered_at_utc must be an ISO-8601 timestamp") from error
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != dt.timedelta(0):
        raise StudyError("study.preregistered_at_utc must include an explicit UTC offset")
    study_class = _string(study["study_class"], "study.study_class")
    if study_class not in ("flagship", "ci_smoke"):
        raise StudyError("study.study_class must be flagship or ci_smoke")
    if study["frozen"] is not True:
        raise StudyError("study manifest must be frozen")
    labels = _exact_keys(
        study["public_labels"],
        {"mcts", "alpha_beta", "jacek_inspired", "rank5_derived"},
        "study.public_labels",
    )
    if labels != PUBLIC_LABELS:
        raise StudyError("study.public_labels do not match the four frozen entrant labels")
    if study["rank5_disclaimer"] != RANK5_DISCLAIMER:
        raise StudyError("Rank5Derived disclaimer is not exact")

    source = _exact_keys(
        top["source"],
        {"git_commit", "dirty_worktree", "analysis_contract_path",
         "analysis_contract_sha256", "arena_sha256", "opening_tool_sha256",
         "protected_artifacts"},
        "source",
    )
    commit = _string(source["git_commit"], "source.git_commit")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise StudyError("source.git_commit must be a full lowercase commit hash")
    if _bool(source["dirty_worktree"], "source.dirty_worktree"):
        raise StudyError("a frozen study cannot declare a dirty source worktree")
    _sha256(source["analysis_contract_sha256"], "source.analysis_contract_sha256")
    _sha256(source["arena_sha256"], "source.arena_sha256")
    _sha256(source["opening_tool_sha256"], "source.opening_tool_sha256")
    artifacts = _exact_keys(
        source["protected_artifacts"],
        {"rank5_submission_path", "rank5_submission_sha256", "jacek_model_path",
         "jacek_model_sha256"},
        "source.protected_artifacts",
    )
    if artifacts["rank5_submission_sha256"] != RANK5_SOURCE_SHA256:
        raise StudyError("authentic rank-5 artifact hash is not the verified hash")
    _sha256(artifacts["jacek_model_sha256"],
            "source.protected_artifacts.jacek_model_sha256")
    source_paths = {
        "analysis_contract": _repository_relative_path(
            repository, source["analysis_contract_path"],
            "source.analysis_contract_path",
        ),
        "rank5_submission": _repository_relative_path(
            repository, artifacts["rank5_submission_path"],
            "source.protected_artifacts.rank5_submission_path",
        ),
        "jacek_model": _repository_relative_path(
            repository, artifacts["jacek_model_path"],
            "source.protected_artifacts.jacek_model_path",
        ),
    }

    rules = _exact_keys(
        top["rules"],
        {"width", "height", "goal_rule", "blocked_rule", "playable_edges",
         "max_game_plies", "maximum_game_length_policy", "opening_ply_definition",
         "natural_draws"},
        "rules",
    )
    locked_rules = {
        "width": 8,
        "height": 10,
        "goal_rule": "opponent_goal_only",
        "blocked_rule": "player_to_move_loses",
        "playable_edges": 316,
        "max_game_plies": 512,
        "natural_draws": False,
    }
    for key, expected in locked_rules.items():
        if rules[key] != expected:
            raise StudyError(f"unsupported rules.{key}: expected {expected!r}")
    _string(rules["maximum_game_length_policy"], "rules.maximum_game_length_policy")
    _string(rules["opening_ply_definition"], "rules.opening_ply_definition")

    configurations = _array(top["configurations"], "configurations")
    if not configurations:
        raise StudyError("configurations must not be empty")
    configuration_objects = [_object(value, f"configurations[{index}]")
                             for index, value in enumerate(configurations)]
    _unique_ids(configuration_objects, "configurations")
    for index, config in enumerate(configurations):
        _validate_configuration(config, index, study_class)
    configs_by_id = {config["id"]: config for config in configuration_objects}
    for config in configuration_objects:
        if config["kind"] == "jacek-inspired" and (
                config["settings"]["model_path"] != artifacts["jacek_model_path"] or
                config["settings"]["model_sha256"] != artifacts["jacek_model_sha256"]):
            raise StudyError("JacekInspired configuration does not match the protected model")
    rank5_ids = [identifier for identifier, config in configs_by_id.items()
                 if config["kind"] == "rank5-derived"]
    if study_class == "flagship" and rank5_ids != ["rank5-fixed-50k"]:
        raise StudyError("flagship manifest requires exactly rank5-fixed-50k")

    grids = _exact_keys(
        top["candidate_grids"],
        {"mcts", "alpha_beta", "jacek_inspired", "rank5_derived"},
        "candidate_grids",
    )
    grid_ids: set[str] = set()
    for family, raw_ids in grids.items():
        ids = _array(raw_ids, f"candidate_grids.{family}")
        if len(ids) != len(set(ids)):
            raise StudyError(f"candidate_grids.{family} contains duplicate IDs")
        for identifier in ids:
            identifier = _string(identifier, f"candidate_grids.{family}[]")
            if identifier not in configs_by_id:
                raise StudyError(f"candidate grid references unknown config {identifier}")
            if configs_by_id[identifier]["family"] != family:
                raise StudyError(f"candidate grid family mismatch for {identifier}")
            grid_ids.add(identifier)
    if grid_ids != set(configs_by_id):
        raise StudyError("every configuration must occur exactly in its family grid")
    if study_class == "flagship":
        expected_iterations = [1000, 2000, 4000]
        actual_iterations = [configs_by_id[i]["settings"]["iterations"] for i in grids["mcts"]]
        if actual_iterations != expected_iterations:
            raise StudyError("MCTS grid must be exactly 1000/2000/4000 iterations")
        for family in ("alpha_beta", "jacek_inspired"):
            actual_nodes = [configs_by_id[i]["settings"]["max_nodes"] for i in grids[family]]
            if actual_nodes != [20_000, 50_000, 100_000]:
                raise StudyError(f"{family} grid must be exactly 20k/50k/100k nodes")
        if grids["rank5_derived"] != ["rank5-fixed-50k"]:
            raise StudyError("Rank5Derived must be fixed and must not be swept")

    openings = _exact_keys(
        top["openings"], {"generator", "depths", "banks"}, "openings"
    )
    generator = _exact_keys(
        openings["generator"],
        {"id", "description", "selection", "terminal_rejection",
         "duplicate_policy", "canonical_equivalence", "state_hash_algorithm"},
        "openings.generator",
    )
    if generator["id"] != "uniform-legal-move-generator/v1":
        raise StudyError("unsupported opening generator")
    if "bot" in generator["description"].lower():
        raise StudyError("opening generator must be described as data generation, not a bot")
    depths = tuple(_array(openings["depths"], "openings.depths"))
    if study_class == "flagship" and depths != EXPECTED_OPENING_DEPTHS:
        raise StudyError("flagship opening depths must be exactly 4/8/12/20")
    if any(not isinstance(depth, int) or depth <= 0 for depth in depths):
        raise StudyError("opening depths must be positive integers")
    banks = [_object(value, f"openings.banks[{index}]")
             for index, value in enumerate(_array(openings["banks"], "openings.banks"))]
    _unique_ids(banks, "openings.banks")
    bank_keys: set[tuple[str, int]] = set()
    bank_hashes: set[str] = set()
    for index, bank in enumerate(banks):
        where = f"openings.banks[{index}]"
        bank = _exact_keys(bank, {"id", "phase", "depth", "pairs", "path", "sha256", "seed"}, where)
        phase = _string(bank["phase"], f"{where}.phase")
        if phase not in FULL_PHASES:
            raise StudyError(f"{where}.phase is unsupported")
        depth = _integer(bank["depth"], f"{where}.depth", 1)
        pairs = _integer(bank["pairs"], f"{where}.pairs", 1)
        if depth not in depths:
            raise StudyError(f"{where}.depth is not declared")
        if study_class == "flagship" and pairs != EXPECTED_PAIR_COUNTS[phase]:
            raise StudyError(f"{where}.pairs does not match the frozen phase size")
        bank_path = _repository_relative_path(
            repository, bank["path"], f"{where}.path"
        )
        key = (phase, depth)
        if key in bank_keys:
            raise StudyError(f"duplicate opening bank for phase/depth {key}")
        bank_keys.add(key)
        digest = _sha256(bank["sha256"], f"{where}.sha256")
        if digest in bank_hashes:
            raise StudyError("opening banks must have distinct SHA-256 hashes")
        bank_hashes.add(digest)
        _seed(bank["seed"], f"{where}.seed")
        if verify_files:
            path = bank_path
            if not path.is_file() or sha256_file(path) != digest:
                raise StudyError(f"opening bank hash mismatch: {path}")
            records = parse_opening_bank(path)
            metadata = opening_bank_metadata(path)
            if len(records) != pairs:
                raise StudyError(f"opening bank pair count mismatch: {path}")
            if any(record.depth != depth or record.phase != phase for record in records):
                raise StudyError(f"opening bank phase/depth mismatch: {path}")
            if metadata.get("generator_seed") != bank["seed"]:
                raise StudyError(f"opening bank seed mismatch: {path}")
    expected_bank_keys = {(phase, depth) for phase in FULL_PHASES for depth in depths}
    if bank_keys != expected_bank_keys:
        raise StudyError("manifest must contain exactly one bank per phase and depth")

    seeds = _exact_keys(
        top["seeds"], {"opening", "bot", "bootstrap", "calibration", "analysis"}, "seeds"
    )
    seen_seeds: dict[int, str] = {}

    def register_seed(raw: Any, where: str) -> None:
        parsed = _seed(raw, where)
        if parsed in seen_seeds:
            raise StudyError(f"overlapping phase seeds: {where} and {seen_seeds[parsed]}")
        seen_seeds[parsed] = where

    opening_seed_map = _exact_keys(seeds["opening"], set(FULL_PHASES), "seeds.opening")
    for phase in FULL_PHASES:
        depth_map = _exact_keys(
            opening_seed_map[phase], {str(depth) for depth in depths}, f"seeds.opening.{phase}"
        )
        for depth in depths:
            register_seed(depth_map[str(depth)], f"seeds.opening.{phase}.{depth}")
            matching_bank = next(
                bank for bank in banks
                if bank["phase"] == phase and bank["depth"] == depth
            )
            if matching_bank["seed"] != depth_map[str(depth)]:
                raise StudyError(
                    f"opening bank seed differs from seeds.opening.{phase}.{depth}"
                )
    for category in ("bot", "bootstrap", "analysis"):
        phase_map = _exact_keys(seeds[category], set(FULL_PHASES), f"seeds.{category}")
        for phase in FULL_PHASES:
            register_seed(phase_map[phase], f"seeds.{category}.{phase}")
    calibration_seeds = _exact_keys(seeds["calibration"], {"validation"}, "seeds.calibration")
    register_seed(calibration_seeds["validation"], "seeds.calibration.validation")

    samples = _exact_keys(top["samples"], set(FULL_PHASES), "samples")
    for phase in FULL_PHASES:
        sample = _exact_keys(samples[phase], {"color_swapped_pairs_per_depth_matchup",
                                              "games_per_pair"}, f"samples.{phase}")
        pairs = _integer(sample["color_swapped_pairs_per_depth_matchup"],
                         f"samples.{phase}.color_swapped_pairs_per_depth_matchup", 1)
        if study_class == "flagship" and pairs != EXPECTED_PAIR_COUNTS[phase]:
            raise StudyError(f"samples.{phase} is not the frozen sample size")
        if sample["games_per_pair"] != 2:
            raise StudyError("a color-swapped pair must contain exactly two games")

    schedule = _exact_keys(top["schedule"], {"tuning", "test"}, "schedule")
    tuning = [_object(value, f"schedule.tuning[{i}]") for i, value in
              enumerate(_array(schedule["tuning"], "schedule.tuning"))]
    test_schedule = [_object(value, f"schedule.test[{i}]") for i, value in
                     enumerate(_array(schedule["test"], "schedule.test"))]
    _unique_ids(tuning, "schedule.tuning")
    _unique_ids(test_schedule, "schedule.test")
    for index, matchup in enumerate(tuning):
        matchup = _exact_keys(matchup, {"id", "candidate", "opponent", "phases"},
                              f"schedule.tuning[{index}]")
        if matchup["candidate"] not in configs_by_id or matchup["opponent"] not in configs_by_id:
            raise StudyError("tuning matchup references an unknown configuration")
        phases = matchup["phases"]
        if phases != ["development", "validation"]:
            raise StudyError("tuning matchups must be development+validation only")
        if study_class == "flagship" and matchup["opponent"] != "rank5-fixed-50k":
            raise StudyError("all flagship tuning candidates must use the common Rank5 panel")
    if study_class == "flagship":
        tuning_candidates = [matchup["candidate"] for matchup in tuning]
        expected_tuning = list(grids["mcts"] + grids["alpha_beta"] + grids["jacek_inspired"])
        if tuning_candidates != expected_tuning:
            raise StudyError("tuning schedule must cover every tunable candidate once in stable grid order")
    for index, matchup in enumerate(test_schedule):
        matchup = _exact_keys(matchup, {"id", "left_slot", "right_slot"},
                              f"schedule.test[{index}]")
        for side in ("left_slot", "right_slot"):
            if matchup[side] not in (
                "selected:mcts", "selected:alpha_beta", "selected:jacek_inspired",
                "fixed:rank5_derived",
            ):
                raise StudyError(f"unsupported test slot {matchup[side]}")
    if study_class == "flagship" and len(test_schedule) != 6:
        raise StudyError("flagship test schedule must contain six round-robin matchups")
    if study_class == "flagship":
        slots = (
            "selected:mcts", "selected:alpha_beta", "selected:jacek_inspired",
            "fixed:rank5_derived",
        )
        expected_pairs = {
            frozenset((slots[left], slots[right]))
            for left in range(len(slots)) for right in range(left + 1, len(slots))
        }
        actual_pairs = {
            frozenset((matchup["left_slot"], matchup["right_slot"]))
            for matchup in test_schedule
        }
        if len(actual_pairs) != 6 or actual_pairs != expected_pairs or any(
                matchup["left_slot"] == matchup["right_slot"]
                for matchup in test_schedule):
            raise StudyError("flagship test schedule must be the complete four-bot round robin")

    latency = _exact_keys(
        top["latency_protocol"],
        {"runtime", "build_type", "single_threaded", "warmup", "timer_boundary",
         "state_copying", "quantiles", "gate_ms", "rank5_gate_distribution",
         "measurement_phase", "power_conditions"},
        "latency_protocol",
    )
    if latency["runtime"] != "native" or latency["build_type"] != "Release":
        raise StudyError("latency gate must use the optimized native Release build")
    if latency["single_threaded"] is not True or latency["measurement_phase"] != "validation":
        raise StudyError("latency gate must be single-threaded validation data")
    if latency["quantiles"] != ["median", "p90", "p95", "p99", "maximum"]:
        raise StudyError("latency quantiles are not frozen")
    if latency["gate_ms"] != 50 or latency["rank5_gate_distribution"] != "fresh_root_only":
        raise StudyError("50 ms gate or Rank5 fresh-root rule changed")

    selection = _exact_keys(
        top["selection_rule"],
        {"phase", "strength_metric", "practical_tie_percentage_points",
         "tie_break_order", "no_eligible_family_policy", "rank5_policy"},
        "selection_rule",
    )
    if selection["phase"] != "validation" or selection["practical_tie_percentage_points"] != 1.0:
        raise StudyError("selection phase/tie threshold changed")
    if selection["tie_break_order"] != ["lower_p95_latency", "smaller_budget", "stable_config_id"]:
        raise StudyError("selection tie-break order changed")
    if selection["no_eligible_family_policy"] != "stop_before_test":
        raise StudyError("ineligible family policy must stop before test")

    statistics_fields = {
        "bootstrap", "pair_score", "truncations", "bradley_terry",
        "calibration", "pareto", "claim_rule",
    }
    if study_class == "flagship":
        statistics_fields.add("ablations")
    statistics_contract = _exact_keys(
        top["statistics"], statistics_fields, "statistics"
    )
    bootstrap = _exact_keys(statistics_contract["bootstrap"],
                            {"resamples", "confidence", "unit", "stratify_by"},
                            "statistics.bootstrap")
    if bootstrap != {"resamples": 10_000, "confidence": 0.95,
                     "unit": "color_swapped_pair", "stratify_by": "opening_depth"}:
        raise StudyError("bootstrap contract changed")
    if statistics_contract["pair_score"] != {"two_wins": 1.0, "split": 0.5, "two_losses": 0.0}:
        raise StudyError("pair-score definition changed")
    if statistics_contract["truncations"] != "reject_strength_and_calibration":
        raise StudyError("truncations cannot be treated as draws")
    bt = _exact_keys(statistics_contract["bradley_terry"],
                     {"identifiability", "outcomes", "bootstrap_unit",
                      "separation_policy", "convergence_policy",
                      "minimum_bootstrap_success_fraction"},
                     "statistics.bradley_terry")
    if bt["identifiability"] != "sum_to_zero" or bt["outcomes"] != "binary_games_only":
        raise StudyError("Bradley-Terry contract changed")
    if bt["minimum_bootstrap_success_fraction"] != 1.0:
        raise StudyError("Bradley-Terry bootstrap success threshold changed")
    calibration_fields = {
        "fit_phase", "link", "bins", "outcome_perspective",
        "rank5_predictions", "test_metrics",
    }
    if study_class == "flagship":
        calibration_fields |= {
            "uncertainty_method", "bootstrap_resamples", "bootstrap_unit",
            "bootstrap_stratify_by", "bootstrap_seed_source",
            "minimum_bin_successful_resamples",
        }
    calibration = _exact_keys(
        statistics_contract["calibration"], calibration_fields,
        "statistics.calibration",
    )
    if calibration["fit_phase"] != "validation" or calibration["bins"] != 10:
        raise StudyError("calibration must be fit on validation with ten bins")
    if calibration["rank5_predictions"] != "fresh_root_only":
        raise StudyError("cached Rank5 edges cannot be calibration samples")
    if study_class == "flagship" and (
            calibration["test_metrics"] != [
                "brier_score", "log_loss", "ten_bin_reliability",
                "pair_clustered_95_intervals",
            ]
            or calibration["uncertainty_method"] !=
            "pair_cluster_percentile_bootstrap"
            or calibration["bootstrap_resamples"] != 10_000
            or calibration["bootstrap_unit"] != "color_swapped_pair"
            or calibration["bootstrap_stratify_by"] !=
            ["matchup", "opening_depth"]
            or calibration["bootstrap_seed_source"] !=
            "derived_from_seeds.analysis.test_per_bot"
            or calibration["minimum_bin_successful_resamples"] != 1_000):
        raise StudyError("pair-clustered calibration uncertainty contract changed")
    pareto = _exact_keys(statistics_contract["pareto"],
                         {"strength_source", "latency_source", "maximize", "minimize"},
                         "statistics.pareto")
    if pareto["strength_source"] != ["development", "validation"] or \
       pareto["latency_source"] != "validation":
        raise StudyError("Pareto sources changed")
    if study_class == "flagship":
        expected_ablations = {
            "phases": ["development", "validation"],
            "practical_gain_threshold": 0.01,
            "comparison_unit": "aligned_color_swapped_opening_pair",
            "bootstrap_method": "paired_difference_percentile",
            "bootstrap_resamples": 10_000,
            "stratify_by": "opening_depth",
            "comparisons": {
                "mcts": [
                    ["mcts-1000", "mcts-2000"],
                    ["mcts-2000", "mcts-4000"],
                    ["mcts-1000", "mcts-4000"],
                ],
                "alpha_beta": [
                    ["alpha-beta-20k", "alpha-beta-50k"],
                    ["alpha-beta-50k", "alpha-beta-100k"],
                    ["alpha-beta-20k", "alpha-beta-100k"],
                ],
                "jacek_inspired": [
                    ["jacek-20k", "jacek-50k"],
                    ["jacek-50k", "jacek-100k"],
                    ["jacek-20k", "jacek-100k"],
                ],
                "equal_budget_evaluator": [
                    ["alpha-beta-20k", "jacek-20k"],
                    ["alpha-beta-50k", "jacek-50k"],
                    ["alpha-beta-100k", "jacek-100k"],
                ],
            },
            "scaling_classification": {
                "supported_practical_gain": "interval_lower_gt_plus_0.01",
                "supported_regression": "interval_upper_lt_0",
                "supported_no_practical_gain": "interval_upper_lt_plus_0.01",
                "unresolved_at_1pp": "otherwise",
            },
            "evaluator_classification": {
                "neural_materially_stronger": "interval_lower_gt_plus_0.01",
                "hand_materially_stronger": "interval_upper_lt_minus_0.01",
                "practical_equivalence_supported":
                    "interval_within_minus_0.01_plus_0.01",
                "unresolved_at_1pp": "otherwise",
            },
        }
        if statistics_contract["ablations"] != expected_ablations:
            raise StudyError("development/validation ablation contract changed")

    outputs = _exact_keys(
        top["outputs"],
        {"raw_results_root", "curated_root", "selection_lock", "curated_data",
         "charts", "report", "runtime_projection"},
        "outputs",
    )
    raw_results_root = _string(outputs["raw_results_root"], "outputs.raw_results_root")
    curated_root_text = _string(outputs["curated_root"], "outputs.curated_root")

    def output_path(raw: Any, where: str) -> pathlib.Path:
        return _repository_relative_path(repository, raw, where)

    repository_results = (repository / "results").resolve()
    raw_resolved = output_path(raw_results_root, "outputs.raw_results_root")
    if raw_resolved == repository_results or repository_results not in raw_resolved.parents:
        raise StudyError("raw outputs must remain below ignored results/")
    if study_class == "flagship":
        allowed_curated = (repository / "benchmarks" / "flagship_study").resolve()
    else:
        allowed_curated = (repository / "results" / "flagship_study_smoke").resolve()
    curated_resolved = output_path(curated_root_text, "outputs.curated_root")
    if curated_resolved != allowed_curated and allowed_curated not in curated_resolved.parents:
        qualifier = "benchmarks/flagship_study" if study_class == "flagship" else \
            "ignored results/flagship_study_smoke"
        raise StudyError(f"curated outputs must remain under {qualifier}/")
    curated_data = _exact_keys(
        outputs["curated_data"], set(FULL_PHASES), "outputs.curated_data"
    )
    chart_paths = _exact_keys(outputs["charts"],
                              {"bradley_terry", "pareto", "calibration"},
                              "outputs.charts")
    if any(not str(path).endswith(".svg") for path in chart_paths.values()):
        raise StudyError("all committed charts must be deterministic SVG files")
    curated_artifacts = {
        "outputs.selection_lock": outputs["selection_lock"],
        "outputs.report": outputs["report"],
        "outputs.runtime_projection": outputs["runtime_projection"],
        **{f"outputs.curated_data.{key}": value
           for key, value in curated_data.items()},
        **{f"outputs.charts.{key}": value for key, value in chart_paths.items()},
    }
    for where, raw in curated_artifacts.items():
        resolved = output_path(raw, where)
        if resolved == curated_resolved or curated_resolved not in resolved.parents:
            raise StudyError(f"{where} must remain below outputs.curated_root")

    environment = _exact_keys(
        top["environment"],
        {"os", "kernel", "architecture", "cpu", "physical_cores", "logical_cores",
         "memory_bytes", "compiler", "compiler_version", "cmake_version",
         "python_version", "build_flags", "machine_id", "power_measurement"},
        "environment",
    )
    for key in ("os", "kernel", "architecture", "cpu", "compiler", "compiler_version",
                "cmake_version", "python_version", "build_flags", "machine_id",
                "power_measurement"):
        _string(environment[key], f"environment.{key}")
    _integer(environment["physical_cores"], "environment.physical_cores", 1)
    _integer(environment["logical_cores"], "environment.logical_cores", 1)
    _integer(environment["memory_bytes"], "environment.memory_bytes", 1)

    if verify_files:
        artifact_checks = (
            (source_paths["analysis_contract"], source["analysis_contract_sha256"]),
            (source_paths["rank5_submission"], artifacts["rank5_submission_sha256"]),
            (source_paths["jacek_model"], artifacts["jacek_model_sha256"]),
        )
        for path, expected in artifact_checks:
            if not path.is_file() or sha256_file(path) != expected:
                raise StudyError(f"protected artifact hash mismatch: {path}")

    return top


@dataclasses.dataclass(frozen=True)
class OpeningRecord:
    opening_id: str
    phase: str
    depth: int
    generation_seed: str
    state_hash: str
    canonical_key: str
    to_move: str
    moves: tuple[tuple[int, int], ...]


def parse_opening_bank(path: pathlib.Path) -> list[OpeningRecord]:
    """Parse the strict, line-oriented frozen opening-bank format."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise StudyError(f"could not read opening bank {path}: {error}") from error
    if not lines or lines[0] != f"schema\t{OPENING_BANK_SCHEMA_VERSION}":
        raise StudyError(f"opening bank has an unsupported schema: {path}")
    metadata: dict[str, str] = {}
    records: list[OpeningRecord] = []
    header_seen = False
    expected_header = (
        "opening_id\tphase\tdepth\tgeneration_seed\tstate_hash\tcanonical_key\t"
        "to_move\tmoves"
    )
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise StudyError(f"blank line in opening bank {path}:{line_number}")
        if not header_seen:
            if line == expected_header:
                header_seen = True
                continue
            pieces = line.split("\t")
            if len(pieces) != 2 or pieces[0] in metadata:
                raise StudyError(f"invalid opening-bank metadata at {path}:{line_number}")
            metadata[pieces[0]] = pieces[1]
            continue
        pieces = line.split("\t")
        if len(pieces) != 8:
            raise StudyError(f"invalid opening record at {path}:{line_number}")
        opening_id, phase, depth_text, generation_seed, state_hash, canonical_key, to_move, moves_text = pieces
        try:
            depth = int(depth_text)
            _seed(generation_seed, f"{path}:{line_number}.generation_seed")
        except (ValueError, StudyError) as error:
            raise StudyError(f"invalid opening numeric field at {path}:{line_number}") from error
        if to_move not in ("one", "two"):
            raise StudyError(f"invalid side to move at {path}:{line_number}")
        moves: list[tuple[int, int]] = []
        if moves_text:
            for point_text in moves_text.split(";"):
                coordinates = point_text.split(",")
                if len(coordinates) != 2:
                    raise StudyError(f"invalid move transcript at {path}:{line_number}")
                try:
                    moves.append((int(coordinates[0]), int(coordinates[1])))
                except ValueError as error:
                    raise StudyError(f"invalid move coordinate at {path}:{line_number}") from error
        if len(moves) != depth:
            raise StudyError(f"opening depth/transcript mismatch at {path}:{line_number}")
        records.append(OpeningRecord(opening_id, phase, depth, generation_seed,
                                     state_hash, canonical_key, to_move, tuple(moves)))
    required_metadata = {
        "phase", "depth", "pairs", "rules", "generator", "generator_seed",
        "selection", "state_hash_algorithm", "canonicalization",
        "opening_ply_definition",
    }
    if set(metadata) != required_metadata or not header_seen:
        raise StudyError(f"opening bank metadata/header is incomplete: {path}")
    if metadata["generator"] != "uniform-legal-move-generator/v1":
        raise StudyError(f"opening bank generator mismatch: {path}")
    if metadata["rules"] != "8x10;opponent_goal_only;player_to_move_loses":
        raise StudyError(f"opening bank rules mismatch: {path}")
    if metadata["selection"] != "splitmix64-unbiased-rejection-sampling/v1":
        raise StudyError(f"opening bank selection algorithm mismatch: {path}")
    if metadata["state_hash_algorithm"] != "sha256-canonical-game-state/v1":
        raise StudyError(f"opening bank state-hash algorithm mismatch: {path}")
    if metadata["canonicalization"] != \
            "horizontal-reflection-min-serialization-sha256/v1":
        raise StudyError(f"opening bank canonicalization mismatch: {path}")
    if metadata["opening_ply_definition"] != \
            "one physical selected edge, including rebound edges":
        raise StudyError(f"opening bank ply definition mismatch: {path}")
    _seed(metadata["generator_seed"], f"{path}.generator_seed")
    if int(metadata["pairs"]) != len(records):
        raise StudyError(f"opening bank declared pair count mismatch: {path}")
    if any(record.phase != metadata["phase"] or record.depth != int(metadata["depth"])
           for record in records):
        raise StudyError(f"opening record metadata mismatch: {path}")
    identifiers = [record.opening_id for record in records]
    state_hashes = [record.state_hash for record in records]
    canonical = [record.canonical_key for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise StudyError(f"duplicate opening IDs in {path}")
    if len(set(state_hashes)) != len(state_hashes):
        raise StudyError(f"duplicate states in {path}")
    if len(set(canonical)) != len(canonical):
        raise StudyError(f"canonically equivalent states in {path}")
    return records


def opening_bank_metadata(path: pathlib.Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.startswith("opening_id\t"):
            break
        pieces = line.split("\t")
        if len(pieces) == 2:
            metadata[pieces[0]] = pieces[1]
    return metadata


def verify_opening_phase_disjointness(manifest: Mapping[str, Any],
                                       repository: pathlib.Path) -> None:
    seen_state: dict[str, str] = {}
    seen_canonical: dict[str, str] = {}
    for bank in manifest["openings"]["banks"]:
        path = repository / bank["path"]
        for record in parse_opening_bank(path):
            for value, seen, label in (
                (record.state_hash, seen_state, "state"),
                (record.canonical_key, seen_canonical, "canonical state"),
            ):
                previous = seen.get(value)
                if previous is not None:
                    raise StudyError(
                        f"opening {label} overlap between {previous} and {record.opening_id}"
                    )
                seen[value] = record.opening_id


def repository_root_from_manifest(manifest_path: pathlib.Path) -> pathlib.Path:
    candidate = manifest_path.resolve()
    for parent in (candidate.parent, *candidate.parents):
        if (parent / ".git").exists() and (parent / "CMakeLists.txt").is_file():
            return parent
    raise StudyError(f"could not locate repository root above {manifest_path}")


def verify_flagship_source_checkout(manifest: Mapping[str, Any],
                                    repository: pathlib.Path) -> None:
    """Bind execution and analysis code to the preregistered framework commit."""

    if manifest["study"]["study_class"] != "flagship":
        return
    source_commit = manifest["source"]["git_commit"]
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise StudyError(f"preregistered source commit is unavailable: {source_commit}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise StudyError("current HEAD does not descend from the preregistered source commit")
    frozen_framework_paths = [
        "CMakeLists.txt", "include", "src", "models/jacek_article_value_model.json",
        "submissions/codingame/bots/rank_5",
        "benchmarks/flagship_study/__init__.py",
        "benchmarks/flagship_study/ablations.py",
        "benchmarks/flagship_study/analysis.py",
        "benchmarks/flagship_study/analysis_contract.md",
        "benchmarks/flagship_study/charts.py",
        "benchmarks/flagship_study/manifest.schema.json",
        "benchmarks/flagship_study/prepare_manifest.py",
        "benchmarks/flagship_study/report.py",
        "benchmarks/flagship_study/run_study.py",
        "benchmarks/flagship_study/studylib.py",
    ]
    differs = subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--", *frozen_framework_paths],
        cwd=repository, check=False,
    )
    if differs.returncode != 0:
        raise StudyError(
            "execution/analysis framework differs from source.git_commit; "
            "refusing an unfrozen study run"
        )


def configurations_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {config["id"]: config for config in manifest["configurations"]}


def _derived_seed(base_seed: str, *parts: str) -> int:
    material = "\0".join((base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _bot_cli(prefix: str, config: Mapping[str, Any]) -> list[str]:
    kind = config["kind"]
    settings = config["settings"]
    result = [f"--{prefix}-kind", kind]
    if kind == "mcts":
        result += [
            f"--{prefix}-iterations", str(settings["iterations"]),
            f"--{prefix}-policy", settings["rollout_policy"],
            f"--{prefix}-leaf-policy", "rollout-only",
            f"--{prefix}-quiescence-max-depth", str(settings["quiescence_max_depth"]),
            f"--{prefix}-quiescence-max-nodes", str(settings["quiescence_max_nodes"]),
            f"--{prefix}-reuse", str(settings["reuse_tree"]).lower(),
            f"--{prefix}-max-nodes", str(settings["node_capacity"]),
            f"--{prefix}-exploration", format(settings["exploration"], ".17g"),
        ]
    elif kind in ("alpha-beta", "jacek-inspired"):
        result += [
            f"--{prefix}-alpha-beta-depth", str(settings["max_turn_depth"]),
            f"--{prefix}-alpha-beta-max-nodes", str(settings["max_nodes"]),
            f"--{prefix}-alpha-beta-table-entries",
            str(settings["transposition_table_entries"]),
            f"--{prefix}-alpha-beta-max-search-plies", str(settings["max_search_plies"]),
        ]
    return result


@dataclasses.dataclass(frozen=True)
class StudyUnit:
    phase: str
    matchup_id: str
    left_config_id: str
    right_config_id: str
    bank_id: str
    bank_path: str
    opening_depth: int
    pairs: int

    @property
    def unit_id(self) -> str:
        return f"{self.phase}--{self.matchup_id}--d{self.opening_depth:02d}"


def _selection_slot(slot: str, selection: Mapping[str, Any]) -> str:
    if slot == "fixed:rank5_derived":
        return "rank5-fixed-50k"
    family = slot.removeprefix("selected:")
    try:
        return selection["selected_configurations"][family]
    except (KeyError, TypeError) as error:
        raise StudyError(f"selection lock does not resolve slot {slot}") from error


def units_for_phase(manifest: Mapping[str, Any], phase: str,
                    selection: Mapping[str, Any] | None = None) -> list[StudyUnit]:
    if phase not in FULL_PHASES:
        raise StudyError(f"unsupported phase {phase}")
    banks = sorted(
        (bank for bank in manifest["openings"]["banks"] if bank["phase"] == phase),
        key=lambda bank: (bank["depth"], bank["id"]),
    )
    matchups: list[tuple[str, str, str]] = []
    if phase in ("development", "validation"):
        for matchup in manifest["schedule"]["tuning"]:
            if phase in matchup["phases"]:
                matchups.append((matchup["id"], matchup["candidate"], matchup["opponent"]))
    else:
        if selection is None:
            raise StudyError("test units require a selection lock")
        for matchup in manifest["schedule"]["test"]:
            matchups.append((matchup["id"],
                             _selection_slot(matchup["left_slot"], selection),
                             _selection_slot(matchup["right_slot"], selection)))
    return [
        StudyUnit(phase, matchup_id, left, right, bank["id"], bank["path"],
                  bank["depth"], bank["pairs"])
        for matchup_id, left, right in matchups
        for bank in banks
    ]


def deterministic_shard(units: Sequence[StudyUnit], shard_count: int,
                        shard_index: int) -> list[StudyUnit]:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise StudyError("invalid deterministic shard index/count")
    return [unit for index, unit in enumerate(units)
            if index % shard_count == shard_index]


def _raw_root(manifest: Mapping[str, Any], repository: pathlib.Path,
              manifest_hash: str) -> pathlib.Path:
    return repository / manifest["outputs"]["raw_results_root"] / manifest_hash


def _selection_path(manifest: Mapping[str, Any], repository: pathlib.Path) -> pathlib.Path:
    return repository / manifest["outputs"]["selection_lock"]


def _acquire_validation_gate_lock(manifest: Mapping[str, Any],
                                  repository: pathlib.Path,
                                  manifest_hash: str) -> Any | None:
    if manifest["study"]["study_class"] != "flagship":
        return None
    try:
        import fcntl
    except ImportError as error:
        raise StudyError("validation gate serialization requires POSIX file locking") from error
    path = _raw_root(manifest, repository, manifest_hash) / "validation/gate-process.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise StudyError(
            "another validation arena process is active; the latency gate is serialized"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def load_selection_lock(manifest: Mapping[str, Any], repository: pathlib.Path,
                        manifest_hash: str, *,
                        verify_raw_derivation: bool = True) -> dict[str, Any]:
    """Load and recompute a lock, always checking its curated inputs' raw hashes.

    The first flagship authorization and direct callers rebuild every curated
    field from the shards. Resumed test shard processes may use the hash-only
    mode after that authorization to avoid repeatedly parsing all decisions.
    """

    from benchmarks.flagship_study import analysis

    is_flagship = manifest.get("study", {}).get("study_class") == "flagship"

    path = _selection_path(manifest, repository)
    selection = _exact_keys(
        load_json(path),
        {
            "schema_version", "manifest_sha256", "manifest_path", "source_phase",
            "created_at_utc", "opening_bank_sha256", "curated_input_sha256",
            "runtime_projection_sha256",
            "validation_execution_environments", "selection_rule",
            "selected_configurations", "fixed_rank5_configuration",
            "validation_metrics", "rank5_latency", "calibration_seed",
            "calibration_mappings", "validation_pareto",
            "development_validation_ablations", "test_authorized",
        },
        "selection lock",
    )
    if selection["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise StudyError("unsupported selection-lock schema")
    if selection["manifest_sha256"] != manifest_hash:
        raise StudyError("selection lock belongs to another manifest")
    locked_manifest_path = _repository_relative_path(
        repository, selection["manifest_path"], "selection.manifest_path"
    )
    if not locked_manifest_path.is_file() or sha256_file(locked_manifest_path) != manifest_hash:
        raise StudyError("selection lock manifest path/hash is invalid")
    if selection["source_phase"] != "validation" or \
       selection["selection_rule"] != manifest["selection_rule"] or \
       selection["test_authorized"] is not True:
        raise StudyError("selection lock does not authorize the frozen validation selection")
    created = _string(selection["created_at_utc"], "selection.created_at_utc")
    try:
        created_at = dt.datetime.fromisoformat(created)
    except ValueError as error:
        raise StudyError("selection.created_at_utc is invalid") from error
    if created_at.tzinfo is None:
        raise StudyError("selection.created_at_utc must include a time zone")

    expected_bank_hashes = {
        bank["id"]: bank["sha256"] for bank in manifest["openings"]["banks"]
    }
    if selection["opening_bank_sha256"] != expected_bank_hashes:
        raise StudyError("selection lock opening hashes differ from the manifest")

    if is_flagship:
        _, runtime_projection_hash = _validate_runtime_projection_artifact(
            manifest, repository, manifest_hash,
            verify_raw_derivation=verify_raw_derivation,
        )
        if selection["runtime_projection_sha256"] != runtime_projection_hash:
            raise StudyError("selection lock runtime projection has changed")
    elif selection["runtime_projection_sha256"] is not None:
        raise StudyError("non-flagship selection lock has a runtime projection")

    development_path = repository / manifest["outputs"]["curated_data"]["development"]
    validation_path = repository / manifest["outputs"]["curated_data"]["validation"]
    input_hashes = _exact_keys(
        selection["curated_input_sha256"], {"development", "validation"},
        "selection.curated_input_sha256",
    )
    expected_input_hashes = {
        "development": sha256_file(development_path),
        "validation": sha256_file(validation_path),
    }
    if input_hashes != expected_input_hashes:
        raise StudyError("selection lock curated inputs have changed")
    development = _object(load_json(development_path), "development curated data")
    validation = _object(load_json(validation_path), "validation curated data")
    if is_flagship:
        if verify_raw_derivation:
            development = _assert_curated_matches_raw(
                manifest, repository, manifest_hash, "development"
            )
            validation = _assert_curated_matches_raw(
                manifest, repository, manifest_hash, "validation"
            )
        else:
            _validate_curated_raw_source(
                manifest, repository, manifest_hash, "development", development
            )
            _validate_curated_raw_source(
                manifest, repository, manifest_hash, "validation", validation
            )
    for phase, curated in (("development", development), ("validation", validation)):
        if curated.get("schema_version") != CURATED_SCHEMA_VERSION or \
           curated.get("phase") != phase or curated.get("manifest_sha256") != manifest_hash or \
           curated.get("completeness", {}).get("operationally_valid") is not True or \
           curated.get("completeness", {}).get("truncations") != 0:
            raise StudyError(f"selection lock uses invalid {phase} curated data")
        if is_flagship:
            _validate_curated_phase_contract(
                manifest, curated, phase, manifest_hash
            )
    expected_environments = validation.get("source", {}).get("execution_environments", [])
    if selection["validation_execution_environments"] != expected_environments:
        raise StudyError("selection lock validation environments have changed")

    selected = _exact_keys(
        selection["selected_configurations"], set(TUNABLE_FAMILIES),
        "selection.selected_configurations",
    )
    expected_metrics: dict[str, Any] = {}
    for family in TUNABLE_FAMILIES:
        expected_id, rows = _select_family(manifest, validation, family)
        if selected[family] != expected_id or selected[family] not in \
                manifest["candidate_grids"][family]:
            raise StudyError(f"selection lock has an invalid {family} choice")
        expected_metrics.update({row["id"]: row for row in rows})
    if len(set(selected.values())) != len(TUNABLE_FAMILIES):
        raise StudyError("selection lock reuses one configuration across families")
    if selection["validation_metrics"] != expected_metrics:
        raise StudyError("selection lock validation metrics are not reproducible")
    if selection["fixed_rank5_configuration"] != "rank5-fixed-50k":
        raise StudyError("selection lock changes the fixed Rank5Derived identity")

    rank5 = _object(
        validation["configurations"].get("rank5-fixed-50k"),
        "validation Rank5 configuration",
    )
    fresh_p95 = _number(
        _object(rank5.get("fresh_root_latency"), "Rank5 fresh latency").get("p95_ms"),
        "Rank5 fresh p95",
    )
    all_p95 = _number(
        _object(rank5.get("all_edge_latency"), "Rank5 all-edge latency").get("p95_ms"),
        "Rank5 all-edge p95",
    )
    rank5_eligible = fresh_p95 <= manifest["latency_protocol"]["gate_ms"]

    calibration_ids = set(selected.values()) | {"rank5-fixed-50k"}
    if selection["calibration_seed"] != manifest["seeds"]["calibration"]["validation"]:
        raise StudyError("selection lock calibration seed changed")
    observations = _curated_calibration_observations(
        validation, "validation", calibration_ids
    )
    expected_mappings: dict[str, Any] = {}
    for identifier in sorted(calibration_ids):
        try:
            expected_mappings[identifier] = _fit_curated_calibration(
                analysis, observations[identifier]
            ).to_dict()
        except analysis.AnalysisError as error:
            raise StudyError(f"could not reproduce calibration for {identifier}: {error}") from error
    if selection["calibration_mappings"] != expected_mappings:
        raise StudyError("selection lock calibration mappings are not reproducible")

    if is_flagship:
        from benchmarks.flagship_study import ablations
        try:
            expected_ablations = ablations.compute(
                manifest, development, validation
            )
        except ablations.AblationError as error:
            raise StudyError(f"could not reproduce preregistered ablations: {error}") from error
        if selection["development_validation_ablations"] != expected_ablations:
            raise StudyError("selection lock ablations are not reproducible")

    if is_flagship:
        expected_pareto = _build_validation_pareto(
            analysis, manifest, development, validation, expected_metrics
        )
    else:
        legacy_points = [
            {
                "id": identifier,
                "family": row["family"],
                "strength": row["validation_strength"],
                "development_strength": development["configurations"].get(
                    identifier, {}
                ).get("strength", {}).get("mean_pair_score"),
                "p95_ms": row["validation_p95_ms"],
                "strength_phases": ["development", "validation"],
                "latency_phase": "validation",
                "gate_eligible": row["eligible"],
                "selected": row["selected"],
                "fixed": False,
            }
            for identifier, row in sorted(expected_metrics.items())
        ]
        legacy_points.append({
            "id": "rank5-fixed-50k", "family": "rank5_derived",
            "strength": 0.5, "development_strength": 0.5,
            "p95_ms": fresh_p95,
            "strength_phases": ["development", "validation"],
            "latency_phase": "validation", "gate_eligible": rank5_eligible,
            "selected": True, "fixed": True,
            "strength_definition": "fixed common-opponent reference level",
        })
        expected_pareto = analysis.classify_pareto(legacy_points)
    if selection["validation_pareto"] != expected_pareto:
        raise StudyError("selection lock Pareto classification is not reproducible")
    rank5_point = next(
        point for point in expected_pareto if point["id"] == "rank5-fixed-50k"
    )
    if not rank5_eligible:
        frontier_status = "outside_gate"
    elif rank5_point["constrained_pareto_optimal"]:
        frontier_status = "inside_constrained_frontier"
    else:
        frontier_status = "eligible_but_dominated"
    expected_rank5_latency = {
        "fresh_root_p95_ms": fresh_p95,
        "all_edge_p95_ms": all_p95,
        "eligible_under_50_ms": rank5_eligible,
        "test_inclusion": (
            "fixed_constrained_and_unconstrained" if rank5_eligible
            else "fixed_unconstrained"
        ),
        "constrained_frontier_status": frontier_status,
        "unconstrained_pareto_optimal": rank5_point[
            "unconstrained_pareto_optimal"
        ],
    }
    if selection["rank5_latency"] != expected_rank5_latency:
        raise StudyError("selection lock Rank5 latency/frontier status changed")
    return selection


def _require_committed_paths(repository: pathlib.Path,
                             paths: Sequence[pathlib.Path]) -> None:
    for path in paths:
        try:
            relative = path.resolve().relative_to(repository.resolve())
        except ValueError as error:
            raise StudyError(f"path is outside repository: {path}") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode != 0:
            raise StudyError(f"test prerequisite is not tracked: {relative}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative)],
            cwd=repository, capture_output=True, text=True, check=True,
        )
        if dirty.stdout:
            raise StudyError(f"test prerequisite is not committed: {relative}")


def _prepare_test_once(manifest: Mapping[str, Any], repository: pathlib.Path,
                       manifest_path: pathlib.Path, manifest_hash: str,
                       selection_path: pathlib.Path, selection_hash: str,
                       arena_sha256: str,
                       *, destructive_override: bool) -> str:
    raw_root = _raw_root(manifest, repository, manifest_hash) / "test"
    marker = raw_root / "test-once.json"
    run_id = sha256_bytes(
        f"{manifest_hash}\0{selection_hash}\0{arena_sha256}".encode("ascii")
    )[:24]
    if destructive_override and raw_root.exists():
        resolved = raw_root.resolve()
        allowed = (repository / "results" / "flagship_study").resolve()
        if allowed not in resolved.parents:
            raise StudyError("refusing destructive override outside flagship results")
        shutil.rmtree(raw_root)
    if marker.exists():
        current = _object(load_json(marker), "test-once marker")
        if current.get("run_id") != run_id:
            raise StudyError("test results already exist for an independent run")
        if current.get("arena_sha256") != arena_sha256:
            raise StudyError("test results are bound to a different arena executable")
        if current.get("completed") is True:
            raise StudyError("frozen test tournament already completed; refusing a second evaluation")
        return run_id
    if raw_root.exists() and any(raw_root.rglob("*.json")):
        raise StudyError("test result files exist without a compatible test-once marker")
    curated_test = repository / manifest["outputs"]["curated_data"]["test"]
    if curated_test.exists():
        raise StudyError(
            "curated test data already exists; refusing to authorize another evaluation"
        )
    _require_committed_paths(
        repository,
        [manifest_path, selection_path] +
        [repository / bank["path"] for bank in manifest["openings"]["banks"]] +
        [repository / manifest["outputs"]["curated_data"][phase]
         for phase in ("development", "validation")] +
        ([repository / manifest["outputs"]["runtime_projection"]]
         if manifest.get("study", {}).get("study_class") == "flagship" else []),
    )
    marker_value = {
        "schema": "papersoccer.flagship-study-test-once.v1",
        "run_id": run_id,
        "manifest_sha256": manifest_hash,
        "selection_sha256": selection_hash,
        "arena_sha256": arena_sha256,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "completed": False,
    }
    write_json_atomic(marker, marker_value, replace=False)
    return run_id


def _annotate_report(report: dict[str, Any], unit: StudyUnit,
                     manifest_hash: str, run_id: str,
                     bank_records: Sequence[OpeningRecord],
                     execution_environment: Mapping[str, Any] | None = None) -> None:
    expected_games = unit.pairs * 2
    games = report.get("games")
    if not isinstance(games, list) or len(games) != expected_games:
        raise StudyError(f"arena returned the wrong game count for {unit.unit_id}")
    report["study"] = {
        "manifest_sha256": manifest_hash,
        "run_id": run_id,
        "phase": unit.phase,
        "unit_id": unit.unit_id,
        "matchup_id": unit.matchup_id,
        "left_config_id": unit.left_config_id,
        "right_config_id": unit.right_config_id,
        "opening_bank_id": unit.bank_id,
        "opening_depth": unit.opening_depth,
        "execution_environment": dict(execution_environment or {}),
    }
    seen: set[str] = set()
    for game in games:
        pair_index = game.get("pair_index")
        game_in_pair = game.get("game_in_pair")
        if not isinstance(pair_index, int) or pair_index < 0 or pair_index >= len(bank_records):
            raise StudyError(f"invalid pair index in arena report for {unit.unit_id}")
        if game_in_pair not in (0, 1):
            raise StudyError(f"invalid game-in-pair in arena report for {unit.unit_id}")
        opening = bank_records[pair_index]
        pair_id = f"{unit.phase}:{unit.matchup_id}:{opening.opening_id}"
        game_id = f"{pair_id}:g{game_in_pair}"
        if game_id in seen:
            raise StudyError(f"duplicate game ID from arena: {game_id}")
        seen.add(game_id)
        game["study_ids"] = {
            "configuration_left": unit.left_config_id,
            "configuration_right": unit.right_config_id,
            "matchup": unit.matchup_id,
            "opening": opening.opening_id,
            "pair": pair_id,
            "game": game_id,
        }


def _expected_arena_bot_config(config: Mapping[str, Any]) -> dict[str, Any]:
    kind = config["kind"]
    settings = config["settings"]
    if kind == "mcts":
        return {
            "kind": kind,
            "iterations": settings["iterations"],
            "exploration": settings["exploration"],
            "rollout_policy": settings["rollout_policy"],
            "leaf_policy": settings["leaf_policy"],
            "quiescence_max_depth": settings["quiescence_max_depth"],
            "quiescence_max_nodes": settings["quiescence_max_nodes"],
            "reuse_tree": settings["reuse_tree"],
            "max_nodes": settings["node_capacity"],
        }
    if kind in ("alpha-beta", "jacek-inspired"):
        expected = {
            "kind": kind,
            "max_turn_depth": settings["max_turn_depth"],
            "max_nodes": settings["max_nodes"],
            "transposition_table_entries": settings["transposition_table_entries"],
            "max_search_plies": settings["max_search_plies"],
        }
        if kind == "jacek-inspired":
            expected["model_sha256"] = settings["model_sha256"]
        return expected
    return {
        "kind": "rank5-derived",
        "profile": "50k-demo",
        "max_turn_depth": settings["max_turn_depth"],
        "max_nodes": settings["max_nodes"],
        "transposition_table_entries": settings["transposition_table_entries"],
        "evaluation_cache_entries": settings["evaluation_cache_entries"],
        "max_time_ms": settings["wall_clock_limit_ms"],
        "model_blend_percent": settings["learned_value_blend_percent"],
        "replay_corrections": settings["replay_corrections"],
        "replay_book_enabled": False,
        "original_sha256": settings["original_artifact_sha256"],
    }


def _verify_arena_report_contract(report: Mapping[str, Any], unit: StudyUnit,
                                  manifest: Mapping[str, Any],
                                  configs: Mapping[str, Mapping[str, Any]]) -> None:
    if report.get("schema") != "papersoccer.arena.v1" or \
       report.get("mode") != "matches" or report.get("runtime") != "native":
        raise StudyError(f"arena report contract mismatch in {unit.unit_id}")
    configuration = _object(report.get("configuration"), "arena configuration")
    expected_seed = _derived_seed(
        manifest["seeds"]["bot"][unit.phase], unit.matchup_id,
        str(unit.opening_depth),
    )
    expected_common = {
        "rules": {"width": manifest["rules"]["width"],
                  "height": manifest["rules"]["height"]},
        "base_seed": str(expected_seed),
        "seed_pairs": unit.pairs,
        "games": unit.pairs * 2,
        "opening_plies": unit.opening_depth,
        "max_plies": manifest["rules"]["max_game_plies"],
        "bootstrap_samples": 1,
        "opening_generator": "frozen_uniform_legal_move_data_generation_bank",
        "opening_seed_derivation": "committed_bank_accepted_generation_seeds",
    }
    for key, expected in expected_common.items():
        if configuration.get(key) != expected:
            raise StudyError(
                f"arena configuration {key} mismatch in {unit.unit_id}: "
                f"expected {expected!r}, got {configuration.get(key)!r}"
            )
    warmup = _object(configuration.get("warmup"), "arena warmup configuration")
    if warmup.get("decisions_per_entrant") != 8 or warmup.get("timed") is not False or \
       warmup.get("bot_instances") != "separate_from_measured_games":
        raise StudyError(f"arena warmup contract mismatch in {unit.unit_id}")
    for entrant, config_id in (("candidate", unit.left_config_id),
                                ("reference", unit.right_config_id)):
        actual = _object(configuration.get(entrant), f"arena {entrant} configuration")
        expected = _expected_arena_bot_config(configs[config_id])
        if actual != expected:
            raise StudyError(
                f"arena {entrant} configuration differs from manifest for {config_id}"
            )


def capture_execution_environment(arena_path: pathlib.Path,
                                  manifest: Mapping[str, Any]) -> dict[str, Any]:
    def command_output(command: list[str]) -> str:
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return "unavailable"
        if process.returncode != 0:
            return "unavailable"
        return " ".join(process.stdout.split()) or "unavailable"

    power = command_output(["pmset", "-g", "batt"])
    if "AC Power" in power:
        power_source = "ac"
    elif "Battery Power" in power:
        power_source = "battery"
    else:
        power_source = "unknown"
    try:
        provenance_process = subprocess.run(
            [str(arena_path), "provenance"], capture_output=True, text=True,
            check=False,
        )
        provenance = json.loads(provenance_process.stdout)
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"could not read arena build provenance: {error}") from error
    provenance = validate_arena_build_provenance(provenance)
    if provenance_process.returncode != 0 or \
       provenance["schema"] != "papersoccer.arena-build.v1":
        raise StudyError("arena build provenance is invalid")
    cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu == "unavailable":
        cpu = platform.processor() or manifest["environment"]["cpu"]
    physical = command_output(["sysctl", "-n", "hw.physicalcpu"])
    logical = command_output(["sysctl", "-n", "hw.logicalcpu"])
    memory = command_output(["sysctl", "-n", "hw.memsize"])
    return {
        "observed_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "physical_cores": int(physical) if physical.isdecimal() else os.cpu_count() or 1,
        "logical_cores": int(logical) if logical.isdecimal() else os.cpu_count() or 1,
        "memory_bytes": int(memory) if memory.isdecimal() else manifest["environment"]["memory_bytes"],
        "power_source": power_source,
        "power_status": power,
        "power_settings": command_output(["pmset", "-g"]),
        "thermal_status": command_output(["pmset", "-g", "therm"]),
        "arena_sha256": sha256_file(arena_path),
        "build_provenance": provenance,
        "compiler": provenance["compiler_id"],
        "compiler_version": provenance["compiler_version"],
        "build_flags": provenance["configured_flags"],
        "single_threaded": True,
    }


def _verify_flagship_execution_environment(
        manifest: Mapping[str, Any], environment: Mapping[str, Any]) -> None:
    if manifest["study"]["study_class"] != "flagship":
        return
    expected = manifest["environment"]
    provenance = validate_arena_build_provenance(
        environment.get("build_provenance")
    )
    if provenance.get("runtime") != "native" or \
       provenance.get("build_type") != "Release" or \
       provenance.get("ndebug") is not True or \
       provenance.get("sanitizers_enabled") is not False or \
       provenance.get("cxx_standard", 0) < 202002 or \
       provenance.get("source_commit") != manifest["source"]["git_commit"] or \
       provenance.get("source_dirty") is not False:
        raise StudyError("full study requires the optimized native Release C++20 arena")
    comparisons = {
        "arena binary SHA-256": (
            environment.get("arena_sha256"), manifest["source"]["arena_sha256"]
        ),
        "compiler": (provenance.get("compiler_id"), expected["compiler"]),
        "compiler version": (
            provenance.get("compiler_version"), expected["compiler_version"]
        ),
        "build flags": (provenance.get("configured_flags"), expected["build_flags"]),
        "architecture": (environment.get("machine"), expected["architecture"]),
        "CPU": (environment.get("processor"), expected["cpu"]),
        "physical cores": (
            environment.get("physical_cores"), expected["physical_cores"]
        ),
        "logical cores": (
            environment.get("logical_cores"), expected["logical_cores"]
        ),
        "memory": (environment.get("memory_bytes"), expected["memory_bytes"]),
        "kernel/platform": (environment.get("platform"), expected["kernel"]),
    }
    mismatches = [
        f"{label}: observed {observed!r}, expected {recorded!r}"
        for label, (observed, recorded) in comparisons.items()
        if observed != recorded
    ]
    if mismatches:
        raise StudyError(
            "study executable/gate machine differs from the frozen manifest: " +
            "; ".join(mismatches)
        )


def run_phase(manifest_path: pathlib.Path, arena_path: pathlib.Path, phase: str,
              *, shard_count: int = 1, shard_index: int = 0,
              destructive_override: bool = False) -> dict[str, Any]:
    repository = repository_root_from_manifest(manifest_path)
    manifest = validate_manifest(load_json(manifest_path), repository, verify_files=True)
    verify_flagship_source_checkout(manifest, repository)
    verify_opening_phase_disjointness(manifest, repository)
    manifest_hash = manifest_sha256(manifest_path)
    if manifest["study"]["study_class"] == "flagship":
        _require_committed_paths(
            repository,
            [manifest_path] +
            [repository / bank["path"] for bank in manifest["openings"]["banks"]] +
            [repository / manifest["source"]["analysis_contract_path"]],
        )
    if not arena_path.is_file():
        raise StudyError(f"arena executable does not exist: {arena_path}")
    execution_environment = capture_execution_environment(arena_path, manifest)
    _verify_flagship_execution_environment(manifest, execution_environment)
    arena_hash = execution_environment["arena_sha256"]
    selection: dict[str, Any] | None = None
    selection_hash = "none"
    run_id = sha256_bytes(
        f"{manifest_hash}\0{phase}\0{arena_hash}".encode("ascii")
    )[:24]
    if phase == "test":
        selection_path = _selection_path(manifest, repository)
        marker_path = (
            _raw_root(manifest, repository, manifest_hash) /
            "test/test-once.json"
        )
        selection = load_selection_lock(
            manifest, repository, manifest_hash,
            verify_raw_derivation=not marker_path.is_file(),
        )
        selection_hash = sha256_file(selection_path)
        run_id = _prepare_test_once(
            manifest, repository, manifest_path, manifest_hash,
            selection_path, selection_hash, arena_hash,
            destructive_override=destructive_override,
        )
    elif destructive_override:
        raise StudyError("destructive override is only valid for the protected test phase")
    units = units_for_phase(manifest, phase, selection)
    configs = configurations_by_id(manifest)
    selected_units = deterministic_shard(units, shard_count, shard_index)
    if (manifest["study"]["study_class"] == "flagship" and phase == "validation" and
            execution_environment["power_source"] != "ac"):
        raise StudyError(
            "validation latency gate requires the preregistered AC-power condition; "
            f"observed {execution_environment['power_source']}"
        )
    if (manifest["study"]["study_class"] == "flagship" and phase == "validation" and
            "lowpowermode 1" in execution_environment["power_settings"].lower()):
        raise StudyError("validation latency gate requires Low Power Mode to be disabled")
    gate_lock = (
        _acquire_validation_gate_lock(manifest, repository, manifest_hash)
        if phase == "validation" else None
    )
    completed = 0
    resumed = 0
    for unit in selected_units:
        output = _raw_root(manifest, repository, manifest_hash) / phase / "shards" / f"{unit.unit_id}.json"
        if output.exists():
            previous = _object(load_json(output), str(output))
            metadata = previous.get("study", {})
            if metadata.get("manifest_sha256") != manifest_hash or \
               metadata.get("unit_id") != unit.unit_id or \
               metadata.get("run_id") != run_id:
                raise StudyError(f"refusing to overwrite incompatible completed shard: {output}")
            resumed += 1
            continue
        bank_path = repository / unit.bank_path
        bank_records = parse_opening_bank(bank_path)
        base_seed = _derived_seed(
            manifest["seeds"]["bot"][phase], unit.matchup_id, str(unit.opening_depth)
        )
        command = [
            str(arena_path), "matches",
            "--seed", str(base_seed),
            "--pairs", str(unit.pairs),
            "--opening-bank", str(bank_path),
            "--max-plies", str(manifest["rules"]["max_game_plies"]),
            "--bootstrap-samples", "1",
            "--warmup-decisions", "8",
        ]
        command += _bot_cli("candidate", configs[unit.left_config_id])
        command += _bot_cli("reference", configs[unit.right_config_id])
        wall_start = time.perf_counter()
        process = subprocess.run(command, cwd=repository, capture_output=True, text=True,
                                 check=False)
        wall_seconds = time.perf_counter() - wall_start
        if process.returncode != 0:
            raise StudyError(
                f"arena failed for {unit.unit_id} with exit {process.returncode}:\n"
                f"{process.stderr.strip()}"
            )
        try:
            report = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise StudyError(f"arena returned invalid JSON for {unit.unit_id}: {error}") from error
        _verify_arena_report_contract(report, unit, manifest, configs)
        if report.get("summary", {}).get("truncations") != 0:
            raise StudyError(f"operational defect: truncation in {unit.unit_id}")
        _annotate_report(
            report, unit, manifest_hash, run_id, bank_records,
            execution_environment,
        )
        report["study"]["wall_seconds"] = wall_seconds
        report["study"]["arena_command"] = command
        write_json_atomic(output, report, replace=False)
        completed += 1
    result = {
        "phase": phase,
        "manifest_sha256": manifest_hash,
        "run_id": run_id,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "units_assigned": len(selected_units),
        "units_completed": completed,
        "units_resumed": resumed,
        "total_phase_units": len(units),
    }
    if gate_lock is not None:
        gate_lock.close()
    return result


def nearest_rank_quantile(values: Sequence[int | float], probability: float) -> float:
    if not values:
        raise StudyError("cannot calculate a quantile of an empty sample")
    if probability < 0.0 or probability > 1.0:
        raise StudyError("quantile probability must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def latency_summary(elapsed_ns: Sequence[int]) -> dict[str, Any]:
    if not elapsed_ns:
        return {"decisions": 0, "median_ms": None, "p90_ms": None,
                "p95_ms": None, "p99_ms": None, "maximum_ms": None}
    return {
        "decisions": len(elapsed_ns),
        "median_ms": nearest_rank_quantile(elapsed_ns, 0.5) / 1_000_000.0,
        "p90_ms": nearest_rank_quantile(elapsed_ns, 0.9) / 1_000_000.0,
        "p95_ms": nearest_rank_quantile(elapsed_ns, 0.95) / 1_000_000.0,
        "p99_ms": nearest_rank_quantile(elapsed_ns, 0.99) / 1_000_000.0,
        "maximum_ms": max(elapsed_ns) / 1_000_000.0,
    }


def stratified_pair_bootstrap(strata: Mapping[int, Sequence[float]], seed: int,
                              resamples: int = 10_000) -> dict[str, Any]:
    if resamples <= 0 or not strata or any(not values for values in strata.values()):
        raise StudyError("stratified bootstrap requires nonempty strata and resamples")
    generator = random.Random(seed)
    ordered = [(depth, tuple(values)) for depth, values in sorted(strata.items())]
    total_pairs = sum(len(values) for _, values in ordered)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _, values in ordered:
            total += sum(values[generator.randrange(len(values))] for _ in values)
        means.append(total / total_pairs)
    means.sort()
    return {
        "method": "depth_stratified_pair_percentile",
        "seed": str(seed),
        "resamples": resamples,
        "confidence": 0.95,
        "lower": nearest_rank_quantile(means, 0.025),
        "upper": nearest_rank_quantile(means, 0.975),
    }


def pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Classify configurations while minimizing latency and maximizing strength."""

    result: list[dict[str, Any]] = []
    for point in points:
        latency = _number(point["p95_ms"], "Pareto point p95_ms")
        strength = _number(point["strength"], "Pareto point strength")
        dominated_by: list[str] = []
        for other in points:
            if other is point:
                continue
            other_latency = _number(other["p95_ms"], "Pareto point p95_ms")
            other_strength = _number(other["strength"], "Pareto point strength")
            if (other_latency <= latency and other_strength >= strength and
                    (other_latency < latency or other_strength > strength)):
                dominated_by.append(str(other["id"]))
        annotated = dict(point)
        annotated["pareto_optimal"] = not dominated_by
        annotated["dominated_by"] = sorted(dominated_by)
        result.append(annotated)
    return result


def _read_unit_reports(manifest: Mapping[str, Any], repository: pathlib.Path,
                       manifest_hash: str, phase: str,
                       selection: Mapping[str, Any] | None) -> list[tuple[StudyUnit, dict[str, Any]]]:
    expected_units = units_for_phase(manifest, phase, selection)
    shard_directory = _raw_root(manifest, repository, manifest_hash) / phase / "shards"
    expected_paths = {f"{unit.unit_id}.json": unit for unit in expected_units}
    present_paths = {path.name: path for path in shard_directory.glob("*.json")} \
        if shard_directory.is_dir() else {}
    missing = sorted(set(expected_paths) - set(present_paths))
    unknown = sorted(set(present_paths) - set(expected_paths))
    if missing or unknown:
        raise StudyError(
            f"phase {phase} shard set is incomplete: missing={missing}, unknown={unknown}"
        )
    reports: list[tuple[StudyUnit, dict[str, Any]]] = []
    run_ids: set[str] = set()
    arena_hashes: set[str] = set()
    for name, unit in expected_paths.items():
        report = _object(load_json(present_paths[name]), str(present_paths[name]))
        study = _object(report.get("study"), f"{name}.study")
        if study.get("manifest_sha256") != manifest_hash or \
           study.get("unit_id") != unit.unit_id or study.get("phase") != phase:
            raise StudyError(f"shard metadata mismatch: {present_paths[name]}")
        run_ids.add(_string(study.get("run_id"), f"{name}.study.run_id"))
        environment = _object(
            study.get("execution_environment"),
            f"{name}.study.execution_environment",
        )
        arena_hashes.add(_sha256(environment.get("arena_sha256"),
                                 f"{name}.arena_sha256"))
        reports.append((unit, report))
    if len(run_ids) != 1 or len(arena_hashes) != 1:
        raise StudyError(
            f"phase {phase} shards mix run identities or arena executables"
        )
    only_run_id = next(iter(run_ids))
    only_arena_hash = next(iter(arena_hashes))
    if phase == "test":
        marker_path = _raw_root(manifest, repository, manifest_hash) / "test/test-once.json"
        marker = _object(load_json(marker_path), "test-once marker")
        if marker.get("run_id") != only_run_id or \
           marker.get("arena_sha256") != only_arena_hash:
            raise StudyError("test shards do not match the protected test-once identity")
    else:
        expected_run_id = sha256_bytes(
            f"{manifest_hash}\0{phase}\0{only_arena_hash}".encode("ascii")
        )[:24]
        if only_run_id != expected_run_id:
            raise StudyError(f"phase {phase} run identity is not bound to its arena binary")
    return reports


def _verify_report_openings(report: Mapping[str, Any], unit: StudyUnit,
                            records: Sequence[OpeningRecord]) -> None:
    openings = report.get("openings")
    if not isinstance(openings, list) or len(openings) != len(records):
        raise StudyError(f"opening count mismatch in {unit.unit_id}")
    for index, (actual, expected) in enumerate(zip(openings, records, strict=True)):
        if actual.get("pair_index") != index:
            raise StudyError(f"opening pair order mismatch in {unit.unit_id}")
        moves = actual.get("moves")
        expected_moves = [{"x": x, "y": y} for x, y in expected.moves]
        if moves != expected_moves:
            raise StudyError(
                f"arena did not use frozen transcript {expected.opening_id} in {unit.unit_id}"
            )
        if actual.get("actual_plies") != expected.depth or \
           actual.get("requested_plies") != expected.depth:
            raise StudyError(f"opening ply mismatch in {unit.unit_id}")
        state = actual.get("state", {})
        if state.get("to_move") != expected.to_move:
            raise StudyError(f"opening side-to-move mismatch in {unit.unit_id}")
        if "opening_id" in actual and actual["opening_id"] != expected.opening_id:
            raise StudyError(f"opening ID mismatch in {unit.unit_id}")
        if "state_hash" in actual and actual["state_hash"] != expected.state_hash:
            raise StudyError(f"opening state hash mismatch in {unit.unit_id}")


def _winner_config(game: Mapping[str, Any], unit: StudyUnit) -> str:
    outcome = _object(game.get("outcome"), "game.outcome")
    if outcome.get("truncated") is True:
        raise StudyError(f"operational defect: truncated game {game.get('study_ids', {}).get('game')}")
    winner = outcome.get("winner")
    if winner == "candidate":
        return unit.left_config_id
    if winner == "reference":
        return unit.right_config_id
    raise StudyError("completed arena game has no candidate/reference winner")


def _decision_config(decision: Mapping[str, Any], unit: StudyUnit) -> str:
    entrant = decision.get("bot")
    if entrant == "candidate":
        return unit.left_config_id
    if entrant == "reference":
        return unit.right_config_id
    raise StudyError("decision has an unknown arena entrant")


def _validate_rank5_sequences(game: Mapping[str, Any], unit: StudyUnit,
                              configs: Mapping[str, Mapping[str, Any]]) -> None:
    decisions = game.get("decisions")
    if not isinstance(decisions, list):
        raise StudyError("game decisions must be an array")
    index = 0
    while index < len(decisions):
        decision = _object(decisions[index], "game decision")
        if decision.get("legal") is not True:
            raise StudyError("arena report contains an unvalidated edge")
        config_id = _decision_config(decision, unit)
        rank5 = decision.get("rank5_derived")
        if configs[config_id]["kind"] != "rank5-derived":
            if rank5 is not None:
                raise StudyError("non-Rank5 decision contains Rank5 diagnostics")
            index += 1
            continue
        stats = _object(rank5, "Rank5 decision diagnostics")
        if stats.get("cached_continuation") is True:
            raise StudyError("Rank5 cached continuation appears without its fresh root")
        planned = stats.get("planned_action_length")
        if not isinstance(planned, int) or planned <= 0 or stats.get("current_edge_index") != 0:
            raise StudyError("Rank5 fresh root has invalid complete-action diagnostics")
        for edge_index in range(planned):
            cursor = index + edge_index
            if cursor >= len(decisions):
                raise StudyError("Rank5 planned action is incomplete at game end")
            continuation = _object(decisions[cursor], "Rank5 action decision")
            if continuation.get("legal") is not True:
                raise StudyError("Rank5 cached action contains an unvalidated edge")
            if _decision_config(continuation, unit) != config_id:
                raise StudyError("Rank5 planned action crosses entrant possession")
            continuation_stats = _object(
                continuation.get("rank5_derived"), "Rank5 action diagnostics"
            )
            if continuation_stats.get("planned_action_length") != planned or \
               continuation_stats.get("current_edge_index") != edge_index or \
               continuation_stats.get("cached_continuation") is not (edge_index != 0):
                raise StudyError("Rank5 complete-action diagnostic sequence is not contiguous")
        index += planned


def _pair_summaries(pair_winners: Mapping[str, Sequence[str]], left_config: str,
                    right_config: str) -> tuple[dict[str, Any], dict[int, list[float]]]:
    won = split = lost = 0
    left_game_wins = right_game_wins = 0
    strata: dict[int, list[float]] = defaultdict(list)
    for compound, winners in sorted(pair_winners.items()):
        if len(winners) != 2:
            raise StudyError(f"pair {compound} does not contain exactly two games")
        depth_text = compound.split("\0", maxsplit=1)[0]
        depth = int(depth_text)
        wins = sum(winner == left_config for winner in winners)
        if any(winner not in (left_config, right_config) for winner in winners):
            raise StudyError(f"pair {compound} has a winner outside its matchup")
        left_game_wins += wins
        right_game_wins += 2 - wins
        score = wins / 2.0
        strata[depth].append(score)
        if wins == 2:
            won += 1
        elif wins == 1:
            split += 1
        else:
            lost += 1
    pairs = won + split + lost
    return ({
        "left_config_id": left_config,
        "right_config_id": right_config,
        "games": pairs * 2,
        "left_wins": left_game_wins,
        "left_losses": right_game_wins,
        "right_wins": right_game_wins,
        "right_losses": left_game_wins,
        "truncations": 0,
        "pairs": pairs,
        "pairs_won_2_0": won,
        "pairs_split_1_1": split,
        "pairs_lost_0_2": lost,
        "mean_pair_score": (won + 0.5 * split) / pairs if pairs else None,
    }, strata)


def _empty_diagnostics(kind: str) -> dict[str, Any]:
    if kind == "mcts":
        return {"searches": 0, "iterations": 0, "nodes": 0,
                "simulated_plies": 0, "expansion_saturated_searches": 0}
    if kind in ("alpha-beta", "jacek-inspired"):
        return {"searches": 0, "nodes": 0, "budget_exhausted_searches": 0,
                "completed_depth_zero_searches": 0, "max_completed_turn_depth": 0,
                "max_attempted_turn_depth": 0}
    summary = {"decisions": 0, "fresh_root_searches": 0,
               "cached_continuation_edges": 0, "requested_nodes": 0,
               "visited_nodes": 0, "budget_exhausted_fresh_searches": 0,
               "max_completed_turn_depth": 0, "max_attempted_turn_depth": 0,
               "completed_turn_depth_histogram": {},
               "attempted_turn_depth_histogram": {},
               "planned_action_length_histogram": {},
               "maximum_current_edge_index": 0,
               "minimum_root_score": None, "maximum_root_score": None}
    for field in RANK5_FRESH_COUNTER_FIELDS:
        summary[f"{field}_sum"] = 0
        summary[f"{field}_max"] = 0
    return summary


def _add_diagnostics(summary: dict[str, Any], decision: Mapping[str, Any],
                     kind: str) -> None:
    if kind == "mcts":
        stats = _object(decision.get("mcts"), "MCTS diagnostics")
        summary["searches"] += 1
        summary["iterations"] += int(stats["iterations"])
        summary["nodes"] += int(stats["nodes"])
        summary["simulated_plies"] += int(stats["simulated_plies"])
        summary["expansion_saturated_searches"] += int(bool(stats["expansion_saturated"]))
    elif kind in ("alpha-beta", "jacek-inspired"):
        stats = _object(decision.get("alpha_beta"), "alpha-beta diagnostics")
        summary["searches"] += 1
        summary["nodes"] += int(stats["nodes"])
        summary["budget_exhausted_searches"] += int(bool(stats["budget_exhausted"]))
        summary["completed_depth_zero_searches"] += int(stats["completed_turn_depth"] == 0)
        summary["max_completed_turn_depth"] = max(
            summary["max_completed_turn_depth"], int(stats["completed_turn_depth"])
        )
        summary["max_attempted_turn_depth"] = max(
            summary["max_attempted_turn_depth"], int(stats["attempted_turn_depth"])
        )
    else:
        stats = _object(decision.get("rank5_derived"), "Rank5 diagnostics")
        summary["decisions"] += 1
        requested_nodes = _integer(
            stats.get("requested_nodes"), "Rank5 requested nodes", 0
        )
        visited_nodes = _integer(
            stats.get("visited_nodes"), "Rank5 visited nodes", 0
        )
        summary["maximum_current_edge_index"] = max(
            summary["maximum_current_edge_index"],
            _integer(stats.get("current_edge_index"), "Rank5 current edge index", 0),
        )
        if stats["cached_continuation"]:
            if requested_nodes != 0 or visited_nodes != 0:
                raise StudyError(
                    "Rank5 cached continuation must not report fresh-search nodes"
                )
            summary["cached_continuation_edges"] += 1
        else:
            if requested_nodes <= 0 or visited_nodes > requested_nodes:
                raise StudyError("Rank5 fresh-root node diagnostics are inconsistent")
            summary["fresh_root_searches"] += 1
            summary["requested_nodes"] += requested_nodes
            summary["visited_nodes"] += visited_nodes
            summary["budget_exhausted_fresh_searches"] += int(bool(stats["budget_exhausted"]))
            summary["max_completed_turn_depth"] = max(
                summary["max_completed_turn_depth"], int(stats["completed_turn_depth"])
            )
            summary["max_attempted_turn_depth"] = max(
                summary["max_attempted_turn_depth"], int(stats["attempted_turn_depth"])
            )
            for key, value in (
                ("completed_turn_depth_histogram", stats["completed_turn_depth"]),
                ("attempted_turn_depth_histogram", stats["attempted_turn_depth"]),
                ("planned_action_length_histogram", stats["planned_action_length"]),
            ):
                text_value = str(int(value))
                summary[key][text_value] = summary[key].get(text_value, 0) + 1
            root_score = int(stats["root_score"])
            summary["minimum_root_score"] = (
                root_score if summary["minimum_root_score"] is None
                else min(summary["minimum_root_score"], root_score)
            )
            summary["maximum_root_score"] = (
                root_score if summary["maximum_root_score"] is None
                else max(summary["maximum_root_score"], root_score)
            )
            for field in RANK5_FRESH_COUNTER_FIELDS:
                value = _integer(
                    stats.get(field), f"Rank5 fresh-root {field}", 0
                )
                summary[f"{field}_sum"] += value
                summary[f"{field}_max"] = max(
                    summary[f"{field}_max"], value
                )


def _validate_rank5_diagnostics_summary(summary: Mapping[str, Any]) -> None:
    fresh = int(summary["fresh_root_searches"])
    cached = int(summary["cached_continuation_edges"])
    if summary["decisions"] != fresh + cached:
        raise StudyError("Rank5 diagnostic decision classification is inconsistent")
    if summary["requested_nodes"] != fresh * 50_000 or \
       summary["visited_nodes"] > summary["requested_nodes"] or \
       summary["budget_exhausted_fresh_searches"] > fresh:
        raise StudyError("Rank5 fresh-root work totals are inconsistent")
    for histogram_name in (
        "completed_turn_depth_histogram",
        "attempted_turn_depth_histogram",
        "planned_action_length_histogram",
    ):
        if sum(summary[histogram_name].values()) != fresh:
            raise StudyError("Rank5 fresh-root histogram total is inconsistent")
    if (summary["minimum_root_score"] is None) != (fresh == 0) or \
       (summary["maximum_root_score"] is None) != (fresh == 0):
        raise StudyError("Rank5 root-score extrema are inconsistent")
    for field in RANK5_FRESH_COUNTER_FIELDS:
        total = int(summary[f"{field}_sum"])
        maximum = int(summary[f"{field}_max"])
        if maximum > total or (fresh == 0 and (total != 0 or maximum != 0)):
            raise StudyError(f"Rank5 {field} aggregate is inconsistent")


def _add_calibration_observation(summary: dict[str, Any],
                                 decision: Mapping[str, Any],
                                 config_id: str,
                                 config: Mapping[str, Any],
                                 winner_config_id: str,
                                 *, pair_cluster_id: str,
                                 stratum_id: str,
                                 truncated: bool) -> None:
    pair_cluster = _string(pair_cluster_id, "calibration pair-cluster ID")
    stratum = _string(stratum_id, "calibration stratum ID")
    summary["decision_count"] += 1
    kind = config["kind"]
    completed_depth: int | None = None
    cached = False
    if kind == "mcts":
        raw_score = _number(
            _object(decision.get("mcts"), "MCTS diagnostics").get("root_value"),
            "MCTS root value",
        )
    elif kind in ("alpha-beta", "jacek-inspired"):
        stats = _object(decision.get("alpha_beta"), "alpha-beta diagnostics")
        raw_score = _number(stats.get("root_score"), "alpha-beta root score")
        completed_depth = _integer(
            stats.get("completed_turn_depth"), "completed alpha-beta depth", 0
        )
    else:
        stats = _object(decision.get("rank5_derived"), "Rank5 diagnostics")
        raw_score = _number(stats.get("root_score"), "Rank5 root score")
        completed_depth = _integer(
            stats.get("completed_turn_depth"), "completed Rank5 depth", 0
        )
        cached = _bool(stats.get("cached_continuation"),
                       "Rank5 cached classification")
    if truncated:
        summary["excluded"]["truncations"] += 1
        return
    if cached:
        summary["excluded"]["cached_continuations"] += 1
        return
    if completed_depth is not None and completed_depth <= 0:
        summary["excluded"]["invalid_depths"] += 1
        return
    player = decision.get("player")
    if player == "one":
        oriented_score = raw_score
    elif player == "two":
        oriented_score = -raw_score
    else:
        raise StudyError("calibration decision has an invalid player-to-move value")
    summary["scores"].append(oriented_score)
    summary["outcomes"].append(int(winner_config_id == config_id))
    summary["pair_cluster_ids"].append(pair_cluster)
    summary["stratum_ids"].append(stratum)


def _idempotent_curated_write(path: pathlib.Path, value: Any) -> bool:
    encoded = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() == encoded:
            return True
        raise StudyError(f"refusing to replace different curated artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return False


def _build_curated_phase(
        manifest: Mapping[str, Any], repository: pathlib.Path,
        manifest_hash: str, phase: str,
        selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Recompute one canonical curated payload entirely from raw arena shards."""

    reports = _read_unit_reports(manifest, repository, manifest_hash, phase, selection)
    configs = configurations_by_id(manifest)
    pair_winners_by_matchup: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    matchups: dict[str, tuple[str, str]] = {}
    elapsed: dict[str, list[int]] = defaultdict(list)
    fresh_elapsed: dict[str, list[int]] = defaultdict(list)
    diagnostics = {identifier: _empty_diagnostics(config["kind"])
                   for identifier, config in configs.items()}
    calibration_observations: dict[str, dict[str, Any]] = {
        identifier: {
            "schema": "papersoccer.flagship-calibration-observations.v1",
            "phase": phase,
            "bot_id": identifier,
            "score_kind": "signed",
            "score_perspective": "player_to_move",
            "decision_count": 0,
            "scores": [],
            "outcomes": [],
            "pair_cluster_ids": [],
            "stratum_ids": [],
            "excluded": {
                "cached_continuations": 0,
                "truncations": 0,
                "invalid_depths": 0,
            },
        }
        for identifier in configs
    }
    binary_games: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    execution_environments: dict[str, dict[str, Any]] = {}
    total_decisions = 0
    for unit, report in reports:
        _verify_arena_report_contract(report, unit, manifest, configs)
        environment = _object(
            _object(report.get("study"), "report.study").get("execution_environment"),
            "report execution environment",
        )
        if environment:
            execution_environments[sha256_bytes(canonical_json_bytes(environment))] = dict(environment)
        bank = parse_opening_bank(repository / unit.bank_path)
        _verify_report_openings(report, unit, bank)
        existing_matchup = matchups.setdefault(
            unit.matchup_id, (unit.left_config_id, unit.right_config_id)
        )
        if existing_matchup != (unit.left_config_id, unit.right_config_id):
            raise StudyError(f"matchup {unit.matchup_id} changes participants across depths")
        games = _array(report.get("games"), f"{unit.unit_id}.games")
        if len(games) != unit.pairs * 2:
            raise StudyError(f"wrong game count in {unit.unit_id}")
        for game in games:
            game = _object(game, "game")
            identifiers = _object(game.get("study_ids"), "game.study_ids")
            game_id = _string(identifiers.get("game"), "game ID")
            if game_id in seen_game_ids:
                raise StudyError(f"duplicate game ID during aggregation: {game_id}")
            seen_game_ids.add(game_id)
            winner = _winner_config(game, unit)
            loser = unit.right_config_id if winner == unit.left_config_id else unit.left_config_id
            pair_id = _string(identifiers.get("pair"), "pair ID")
            pair_key = f"{unit.opening_depth}\0{pair_id}"
            pair_winners_by_matchup[unit.matchup_id][pair_key].append(winner)
            binary_games.append({
                "game_id": game_id,
                "pair_id": pair_id,
                "matchup_id": unit.matchup_id,
                "opening_depth": unit.opening_depth,
                "winner_config_id": winner,
                "loser_config_id": loser,
            })
            _validate_rank5_sequences(game, unit, configs)
            for decision in _array(game.get("decisions"), "game.decisions"):
                decision = _object(decision, "decision")
                config_id = _decision_config(decision, unit)
                elapsed_ns = _integer(decision.get("elapsed_ns"), "decision.elapsed_ns", 0)
                elapsed[config_id].append(elapsed_ns)
                total_decisions += 1
                if configs[config_id]["kind"] == "rank5-derived":
                    rank5 = _object(decision.get("rank5_derived"), "Rank5 diagnostics")
                    if not rank5["cached_continuation"]:
                        fresh_elapsed[config_id].append(elapsed_ns)
                _add_diagnostics(diagnostics[config_id], decision, configs[config_id]["kind"])
                _add_calibration_observation(
                    calibration_observations[config_id], decision, config_id,
                    configs[config_id], winner,
                    pair_cluster_id=pair_id,
                    stratum_id=(
                        f"{unit.matchup_id}:opening-depth-"
                        f"{unit.opening_depth}"
                    ),
                    truncated=False,
                )

    matchup_summaries: dict[str, Any] = {}
    strength_by_config: dict[str, Any] = {}
    paired_scores_by_config: dict[str, Any] = {}
    for matchup_id, (left, right) in sorted(matchups.items()):
        summary, strata = _pair_summaries(
            pair_winners_by_matchup[matchup_id], left, right
        )
        bootstrap_seed = _derived_seed(
            manifest["seeds"]["bootstrap"][phase], phase, matchup_id
        )
        summary["pair_bootstrap_95"] = stratified_pair_bootstrap(
            strata, bootstrap_seed, manifest["statistics"]["bootstrap"]["resamples"]
        )
        summary["by_opening_depth"] = {
            str(depth): {
                "pairs": len(scores),
                "mean_pair_score": sum(scores) / len(scores),
                "pairs_won_2_0": sum(score == 1.0 for score in scores),
                "pairs_split_1_1": sum(score == 0.5 for score in scores),
                "pairs_lost_0_2": sum(score == 0.0 for score in scores),
            }
            for depth, scores in sorted(strata.items())
        }
        matchup_summaries[matchup_id] = summary
        if phase in ("development", "validation"):
            if left in strength_by_config:
                raise StudyError(f"tuning config {left} occurs in multiple matchups")
            strength_by_config[left] = {
                "opponent_config_id": right,
                "mean_pair_score": summary["mean_pair_score"],
                "pair_bootstrap_95": summary["pair_bootstrap_95"],
                "pairs": summary["pairs"],
            }
            opening_ids: list[str] = []
            opening_depths: list[int] = []
            pair_scores: list[float] = []
            prefix = f"{phase}:{matchup_id}:"
            for compound, winners in sorted(
                    pair_winners_by_matchup[matchup_id].items()):
                depth_text, pair_id = compound.split("\0", maxsplit=1)
                if not pair_id.startswith(prefix):
                    raise StudyError(
                        f"pair {pair_id} is not bound to matchup {matchup_id}"
                    )
                opening_id = pair_id[len(prefix):]
                if not opening_id:
                    raise StudyError("paired score lacks a frozen opening ID")
                opening_ids.append(opening_id)
                opening_depths.append(int(depth_text))
                pair_scores.append(
                    sum(winner == left for winner in winners) / 2.0
                )
            paired_scores_by_config[left] = {
                "phase": phase,
                "bot_id": left,
                "opponent_config_id": right,
                "opening_ids": opening_ids,
                "opening_depths": opening_depths,
                "scores": pair_scores,
            }

    if (phase in ("development", "validation")
            and manifest["study"]["study_class"] == "flagship"):
        expected_tuning_ids = {
            identifier
            for family in TUNABLE_FAMILIES
            for identifier in manifest["candidate_grids"][family]
        }
        if set(paired_scores_by_config) != expected_tuning_ids:
            raise StudyError("paired-score candidate set differs from the tuning grid")
        aligned_keys = {
            tuple(zip(
                payload["opening_depths"], payload["opening_ids"], strict=True
            ))
            for payload in paired_scores_by_config.values()
        }
        if len(aligned_keys) != 1:
            raise StudyError("tuning configurations are not aligned on frozen opening pairs")

    for config_id, config in configs.items():
        if config["kind"] == "rank5-derived":
            _validate_rank5_diagnostics_summary(diagnostics[config_id])

    configuration_summaries: dict[str, Any] = {}
    participating = sorted({identifier for pair in matchups.values() for identifier in pair})
    for config_id in participating:
        _validate_compact_calibration_observations(
            calibration_observations[config_id], phase, config_id
        )
    for config_id in participating:
        all_timing = latency_summary(elapsed[config_id])
        fresh_timing = (
            latency_summary(fresh_elapsed[config_id])
            if configs[config_id]["kind"] == "rank5-derived" else None
        )
        gate_timing = fresh_timing if fresh_timing is not None else all_timing
        configuration_summaries[config_id] = {
            "family": configs[config_id]["family"],
            "public_label": configs[config_id]["public_label"],
            "all_edge_latency": all_timing,
            "fresh_root_latency": fresh_timing,
            "latency_gate_p95_ms": gate_timing["p95_ms"],
            "latency_gate_eligible": (
                gate_timing["p95_ms"] is not None and
                gate_timing["p95_ms"] <= manifest["latency_protocol"]["gate_ms"]
            ),
            "diagnostics": diagnostics[config_id],
            "strength": strength_by_config.get(config_id),
        }

    expected_games = sum(unit.pairs * 2 for unit, _ in reports)
    if manifest["study"]["study_class"] == "flagship" and phase == "validation":
        if not execution_environments or any(
            environment.get("power_source") != "ac"
            for environment in execution_environments.values()
        ):
            raise StudyError("validation shards violate the preregistered AC-power gate condition")
    curated = {
        "schema_version": CURATED_SCHEMA_VERSION,
        "phase": phase,
        "manifest_sha256": manifest_hash,
        "source": {
            "raw_root": str(_raw_root(manifest, repository, manifest_hash).relative_to(repository)),
            "units": len(reports),
            "raw_shard_sha256": {
                unit.unit_id: sha256_file(
                    _raw_root(manifest, repository, manifest_hash) / phase /
                    "shards" / f"{unit.unit_id}.json"
                )
                for unit, _ in reports
            },
            "execution_environments": [
                execution_environments[key] for key in sorted(execution_environments)
            ],
        },
        "completeness": {
            "expected_units": len(reports),
            "completed_units": len(reports),
            "expected_games": expected_games,
            "completed_games": len(binary_games),
            "unique_game_ids": len(seen_game_ids),
            "decisions": total_decisions,
            "truncations": 0,
            "operationally_valid": len(binary_games) == expected_games,
        },
        "matchups": matchup_summaries,
        "configurations": configuration_summaries,
        "binary_games": binary_games,
        "paired_scores": paired_scores_by_config,
        "calibration_observations": {
            identifier: calibration_observations[identifier]
            for identifier in participating
        },
    }
    if len(binary_games) != expected_games:
        raise StudyError("missing games detected during aggregation")
    return curated


def aggregate_phase(manifest_path: pathlib.Path, phase: str) -> dict[str, Any]:
    repository = repository_root_from_manifest(manifest_path)
    manifest = validate_manifest(load_json(manifest_path), repository, verify_files=True)
    verify_flagship_source_checkout(manifest, repository)
    verify_opening_phase_disjointness(manifest, repository)
    manifest_hash = manifest_sha256(manifest_path)
    selection = None
    if phase == "test":
        selection = load_selection_lock(
            manifest, repository, manifest_hash, verify_raw_derivation=False
        )
    curated = _build_curated_phase(
        manifest, repository, manifest_hash, phase, selection
    )
    curated_path = repository / manifest["outputs"]["curated_data"][phase]
    resumed = _idempotent_curated_write(curated_path, curated)
    if phase == "test":
        marker = _raw_root(manifest, repository, manifest_hash) / "test" / "test-once.json"
        marker_value = _object(load_json(marker), "test-once marker")
        if marker_value.get("completed") is not True:
            marker_value["completed"] = True
            marker_value["completed_games"] = curated["completeness"]["completed_games"]
            marker_value["completed_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
            write_json_atomic(marker, marker_value, replace=True)
    return {
        "phase": phase,
        "manifest_sha256": manifest_hash,
        "curated_path": str(curated_path.relative_to(repository)),
        "curated_sha256": sha256_file(curated_path),
        "units": curated["completeness"]["completed_units"],
        "games": curated["completeness"]["completed_games"],
        "decisions": curated["completeness"]["decisions"],
        "truncations": 0,
        "resumed": resumed,
    }


_CURATED_BASE_KEYS = {
    "schema_version", "phase", "manifest_sha256", "source", "completeness",
    "matchups", "configurations", "binary_games", "paired_scores",
    "calibration_observations",
}
_CURATED_TEST_ANALYSIS_KEYS = {
    "bradley_terry", "calibration", "validation_pareto", "sample_sizes",
    "analysis_complete",
}


def _validate_curated_raw_source(
        manifest: Mapping[str, Any], repository: pathlib.Path,
        manifest_hash: str, phase: str, curated: Mapping[str, Any],
        selection: Mapping[str, Any] | None = None) -> None:
    """Bind recorded shard provenance to the current complete raw shard set."""

    source = _exact_keys(
        curated.get("source"),
        {"raw_root", "units", "raw_shard_sha256", "execution_environments"},
        f"{phase}.source",
    )
    expected_raw_root = str(
        _raw_root(manifest, repository, manifest_hash).relative_to(repository)
    )
    if source["raw_root"] != expected_raw_root:
        raise StudyError(f"{phase} curated raw root differs from the manifest")
    units = units_for_phase(manifest, phase, selection)
    if source["units"] != len(units):
        raise StudyError(f"{phase} curated raw unit count differs from the manifest")
    _array(source["execution_environments"], f"{phase}.source.execution_environments")
    recorded_hashes = _exact_keys(
        source["raw_shard_sha256"], {unit.unit_id for unit in units},
        f"{phase}.source.raw_shard_sha256",
    )
    shard_directory = (
        _raw_root(manifest, repository, manifest_hash) / phase / "shards"
    )
    expected_names = {f"{unit.unit_id}.json" for unit in units}
    actual_names = (
        {path.name for path in shard_directory.glob("*.json")}
        if shard_directory.is_dir() else set()
    )
    if actual_names != expected_names:
        raise StudyError(
            f"{phase} raw shard set differs from the manifest: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unknown={sorted(actual_names - expected_names)}"
        )
    for unit in units:
        expected_hash = _sha256(
            recorded_hashes[unit.unit_id],
            f"{phase}.source.raw_shard_sha256.{unit.unit_id}",
        )
        path = shard_directory / f"{unit.unit_id}.json"
        if sha256_file(path) != expected_hash:
            raise StudyError(
                f"{phase} raw shard hash differs from curated provenance: {path}"
            )


def _curated_base_for_raw_comparison(
        curated: Mapping[str, Any], expected: Mapping[str, Any], phase: str,
        *, allow_analyzed_test: bool) -> dict[str, Any]:
    """Project an analyzed test artifact back to its exact raw-derived payload."""

    if not (allow_analyzed_test and phase == "test"
            and curated.get("analysis_complete") is True):
        return _exact_keys(curated, set(_CURATED_BASE_KEYS), f"{phase} curated data")

    analyzed = _exact_keys(
        curated,
        set(_CURATED_BASE_KEYS) | set(_CURATED_TEST_ANALYSIS_KEYS),
        "analyzed test curated data",
    )
    base = {key: analyzed[key] for key in _CURATED_BASE_KEYS}
    actual_matchups = _object(analyzed["matchups"], "analyzed test matchups")
    expected_matchups = _object(expected["matchups"], "raw-derived test matchups")
    if set(actual_matchups) != set(expected_matchups):
        raise StudyError("analyzed test matchup set differs from raw shards")
    base["matchups"] = {}
    for matchup_id, expected_summary_value in expected_matchups.items():
        expected_summary = _object(
            expected_summary_value, f"raw-derived test matchup {matchup_id}"
        )
        actual_summary = _exact_keys(
            actual_matchups[matchup_id], set(expected_summary) | {"conclusion"},
            f"analyzed test matchup {matchup_id}",
        )
        base["matchups"][matchup_id] = {
            key: actual_summary[key] for key in expected_summary
        }
    return base


def _assert_curated_matches_raw(
        manifest: Mapping[str, Any], repository: pathlib.Path,
        manifest_hash: str, phase: str,
        selection: Mapping[str, Any] | None = None,
        *, allow_analyzed_test: bool = False) -> dict[str, Any]:
    """Reject any curated field that cannot be reproduced from frozen raw shards."""

    path = repository / manifest["outputs"]["curated_data"][phase]
    curated = _object(load_json(path), f"{phase} curated data")
    _validate_curated_raw_source(
        manifest, repository, manifest_hash, phase, curated, selection
    )
    expected = _build_curated_phase(
        manifest, repository, manifest_hash, phase, selection
    )
    comparable = _curated_base_for_raw_comparison(
        curated, expected, phase, allow_analyzed_test=allow_analyzed_test
    )
    if canonical_json_bytes(comparable) != canonical_json_bytes(expected):
        raise StudyError(
            f"{phase} curated data is not reproducible from frozen raw shards"
        )
    return expected


def _validate_compact_calibration_observations(
        value: Any, phase: str, identifier: str) -> dict[str, Any]:
    where = f"{phase}.calibration_observations.{identifier}"
    payload = _exact_keys(value, CALIBRATION_OBSERVATION_KEYS, where)
    if payload["schema"] != "papersoccer.flagship-calibration-observations.v1" or \
       payload["phase"] != phase or payload["bot_id"] != identifier or \
       payload["score_kind"] != "signed" or \
       payload["score_perspective"] != "player_to_move":
        raise StudyError(f"invalid curated calibration identity for {identifier}")

    decision_count = _integer(
        payload["decision_count"], f"{where}.decision_count", 1
    )
    columns = {
        name: _array(payload[name], f"{where}.{name}")
        for name in (
            "scores", "outcomes", "pair_cluster_ids", "stratum_ids"
        )
    }
    retained = len(columns["scores"])
    if retained == 0 or any(len(column) != retained
                            for column in columns.values()):
        raise StudyError(
            f"curated calibration columns are empty/misaligned for {identifier}"
        )
    for index, score in enumerate(columns["scores"]):
        _number(score, f"{where}.scores[{index}]")
    if any(outcome not in (0, 1) or isinstance(outcome, bool)
           for outcome in columns["outcomes"]):
        raise StudyError(
            f"curated calibration outcomes are not binary for {identifier}"
        )

    cluster_strata: dict[str, str] = {}
    for index, (cluster_value, stratum_value) in enumerate(zip(
            columns["pair_cluster_ids"], columns["stratum_ids"], strict=True)):
        cluster = _string(cluster_value, f"{where}.pair_cluster_ids[{index}]")
        stratum = _string(stratum_value, f"{where}.stratum_ids[{index}]")
        previous = cluster_strata.setdefault(cluster, stratum)
        if previous != stratum:
            raise StudyError(
                f"calibration pair cluster crosses strata for {identifier}"
            )

    excluded = _exact_keys(
        payload["excluded"], CALIBRATION_EXCLUSION_KEYS,
        f"{where}.excluded",
    )
    excluded_count = sum(
        _integer(count, f"{where}.excluded.{key}", 0)
        for key, count in excluded.items()
    )
    if retained + excluded_count != decision_count:
        raise StudyError(
            f"calibration retained/excluded decisions do not total "
            f"decision_count for {identifier}"
        )
    return payload


def _curated_calibration_observations(
        curated: Mapping[str, Any], phase: str,
        config_ids: set[str]) -> dict[str, dict[str, Any]]:
    if curated.get("phase") != phase:
        raise StudyError(f"calibration input is not curated {phase} data")
    by_config = _object(
        curated.get("calibration_observations"),
        f"{phase}.calibration_observations",
    )
    return {
        identifier: _validate_compact_calibration_observations(
            by_config.get(identifier), phase, identifier
        )
        for identifier in sorted(config_ids)
    }


def _expanded_calibration_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": payload["phase"],
            "bot_id": payload["bot_id"],
            "score_kind": payload["score_kind"],
            "score_perspective": "player_to_move",
            "raw_score": score,
            "outcome": outcome,
            "cached_continuation": False,
            "truncated": False,
            "completed_depth": None,
        }
        for score, outcome in zip(payload["scores"], payload["outcomes"], strict=True)
    ]


def _fit_curated_calibration(analysis_module: Any,
                             payload: Mapping[str, Any]) -> Any:
    mapping = analysis_module.fit_logistic_calibration(
        _expanded_calibration_rows(payload), phase="validation"
    )
    excluded = payload["excluded"]
    return dataclasses.replace(
        mapping,
        excluded_cached_continuations=excluded["cached_continuations"],
        excluded_truncations=excluded["truncations"],
        excluded_invalid_depths=excluded["invalid_depths"],
    )


def _calibration_evaluation_options(
        manifest: Mapping[str, Any], identifier: str) -> dict[str, int]:
    contract = manifest["statistics"]["calibration"]
    return {
        "seed": _derived_seed(
            manifest["seeds"]["analysis"]["test"],
            "calibration-pair-cluster",
            identifier,
        ),
        "resamples": int(contract["bootstrap_resamples"]),
        "bins": int(contract["bins"]),
        "minimum_bin_successful_resamples": int(
            contract["minimum_bin_successful_resamples"]
        ),
    }


def _evaluate_curated_calibration(analysis_module: Any,
                                  mapping: Mapping[str, Any],
                                  payload: Mapping[str, Any],
                                  *, seed: int, resamples: int, bins: int,
                                  minimum_bin_successful_resamples: int
                                  ) -> dict[str, Any]:
    if mapping.get("bot_id") != payload["bot_id"] or \
       mapping.get("score_kind") != payload["score_kind"]:
        raise StudyError("calibration mapping and curated payload identities differ")
    probabilities = analysis_module.apply_calibration(
        mapping, payload["scores"]
    )
    metrics = analysis_module.pair_clustered_calibration_metrics(
        probabilities,
        payload["outcomes"],
        payload["pair_cluster_ids"],
        payload["stratum_ids"],
        seed=seed,
        resamples=resamples,
        bins=bins,
        minimum_bin_successful_resamples=minimum_bin_successful_resamples,
    )
    metrics["phase"] = "test"
    metrics["decision_count"] = payload["decision_count"]
    metrics["excluded"] = dict(payload["excluded"])
    return metrics


def _configuration_budget(config: Mapping[str, Any]) -> int:
    if config["kind"] == "mcts":
        return int(config["settings"]["iterations"])
    return int(config["settings"]["max_nodes"])


def _validate_curated_phase_contract(
        manifest: Mapping[str, Any], curated: Mapping[str, Any], phase: str,
        manifest_hash: str,
        selection: Mapping[str, Any] | None = None) -> None:
    """Recompute full manifest samples and outcomes before publication."""

    if curated.get("schema_version") != CURATED_SCHEMA_VERSION or \
       curated.get("phase") != phase or \
       curated.get("manifest_sha256") != manifest_hash:
        raise StudyError(f"{phase} curated data does not match the frozen manifest")
    units = units_for_phase(manifest, phase, selection)
    expected_games = sum(unit.pairs * 2 for unit in units)
    expected_by_matchup: dict[str, dict[str, Any]] = {}
    for unit in units:
        value = expected_by_matchup.setdefault(unit.matchup_id, {
            "left": unit.left_config_id,
            "right": unit.right_config_id,
            "pairs": 0,
            "games": 0,
            "depth_pairs": defaultdict(int),
        })
        if (value["left"], value["right"]) != (
                unit.left_config_id, unit.right_config_id):
            raise StudyError(f"manifest matchup {unit.matchup_id} changes participants")
        value["pairs"] += unit.pairs
        value["games"] += unit.pairs * 2
        value["depth_pairs"][unit.opening_depth] += unit.pairs

    completeness = _object(curated.get("completeness"), f"{phase}.completeness")
    expected_completeness = {
        "expected_units": len(units),
        "completed_units": len(units),
        "expected_games": expected_games,
        "completed_games": expected_games,
        "unique_game_ids": expected_games,
        "truncations": 0,
        "operationally_valid": True,
    }
    for key, expected in expected_completeness.items():
        if completeness.get(key) != expected:
            raise StudyError(
                f"{phase} curated sample contract changed: {key}="
                f"{completeness.get(key)!r}, expected {expected!r}"
            )

    binary_games = _array(curated.get("binary_games"), f"{phase}.binary_games")
    if len(binary_games) != expected_games:
        raise StudyError(f"{phase} binary-game count differs from the manifest")
    game_ids: set[str] = set()
    grouped_pairs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, raw_game in enumerate(binary_games):
        game = _exact_keys(
            raw_game,
            {
                "game_id", "pair_id", "matchup_id", "opening_depth",
                "winner_config_id", "loser_config_id",
            },
            f"{phase}.binary_games[{index}]",
        )
        game_id = _string(game.get("game_id"), "curated game ID")
        pair_id = _string(game.get("pair_id"), "curated pair ID")
        matchup_id = _string(game.get("matchup_id"), "curated matchup ID")
        depth = _integer(game.get("opening_depth"), "curated opening depth", 1)
        if game_id in game_ids or matchup_id not in expected_by_matchup:
            raise StudyError(f"{phase} has a duplicate or unknown curated game")
        expected = expected_by_matchup[matchup_id]
        if depth not in expected["depth_pairs"]:
            raise StudyError(
                f"{phase} matchup {matchup_id} has an unknown opening depth"
            )
        prefix = f"{phase}:{matchup_id}:"
        if not pair_id.startswith(prefix) or not pair_id.removeprefix(prefix):
            raise StudyError(
                f"{phase} pair {pair_id} is not bound to matchup {matchup_id}"
            )
        if game_id not in (f"{pair_id}:g0", f"{pair_id}:g1"):
            raise StudyError(f"{phase} game {game_id} is not bound to pair {pair_id}")
        winner = _string(game.get("winner_config_id"), "curated winner config ID")
        loser = _string(game.get("loser_config_id"), "curated loser config ID")
        if {winner, loser} != {expected["left"], expected["right"]}:
            raise StudyError(
                f"{phase} game {game_id} has participants outside its matchup"
            )
        game_ids.add(game_id)
        grouped_pairs[(matchup_id, pair_id)].append(game)
    if len(game_ids) != expected_games or any(
            len(games) != 2 for games in grouped_pairs.values()):
        raise StudyError(f"{phase} curated pairs are incomplete or duplicated")

    pair_winners_by_matchup: dict[str, dict[str, list[str]]] = defaultdict(dict)
    pair_scores_by_matchup: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    for (matchup_id, pair_id), games in sorted(grouped_pairs.items()):
        expected_game_ids = {f"{pair_id}:g0", f"{pair_id}:g1"}
        if {game["game_id"] for game in games} != expected_game_ids:
            raise StudyError(f"{phase} pair {pair_id} does not contain g0 and g1")
        depths = {game["opening_depth"] for game in games}
        if len(depths) != 1:
            raise StudyError(f"{phase} pair {pair_id} crosses opening depths")
        depth = next(iter(depths))
        winners = [game["winner_config_id"] for game in games]
        left = expected_by_matchup[matchup_id]["left"]
        prefix = f"{phase}:{matchup_id}:"
        opening_id = pair_id[len(prefix):]
        pair_key = (depth, opening_id)
        if pair_key in pair_scores_by_matchup[matchup_id]:
            raise StudyError(
                f"{phase} matchup {matchup_id} repeats opening {opening_id}"
            )
        pair_scores_by_matchup[matchup_id][pair_key] = (
            sum(winner == left for winner in winners) / 2.0
        )
        pair_winners_by_matchup[matchup_id][f"{depth}\0{pair_id}"] = winners

    expected_pair_total = sum(
        expected["pairs"] for expected in expected_by_matchup.values()
    )
    if len(grouped_pairs) != expected_pair_total:
        raise StudyError(f"{phase} curated pair count differs from the manifest")
    opening_key_sets = {
        frozenset(scores) for scores in pair_scores_by_matchup.values()
    }
    if len(opening_key_sets) != 1:
        raise StudyError(
            f"{phase} matchups are not aligned on the frozen opening pairs"
        )

    matchups = _object(curated.get("matchups"), f"{phase}.matchups")
    if set(matchups) != set(expected_by_matchup):
        raise StudyError(f"{phase} curated matchup set differs from the manifest")
    configurations = _object(
        curated.get("configurations"), f"{phase}.configurations"
    )
    participating = {
        identifier
        for expected in expected_by_matchup.values()
        for identifier in (expected["left"], expected["right"])
    }
    if set(configurations) != participating:
        raise StudyError(
            f"{phase} curated configuration set differs from the manifest"
        )
    expected_strengths: dict[str, dict[str, Any]] = {}
    expected_paired_scores: dict[str, dict[tuple[int, str], float]] = {}
    for matchup_id, expected in expected_by_matchup.items():
        recomputed, strata = _pair_summaries(
            pair_winners_by_matchup[matchup_id],
            expected["left"],
            expected["right"],
        )
        for depth, pairs in expected["depth_pairs"].items():
            if len(strata.get(depth, ())) != pairs:
                raise StudyError(
                    f"{phase} matchup {matchup_id} has the wrong depth-{depth} sample"
                )
        if set(strata) != set(expected["depth_pairs"]):
            raise StudyError(
                f"{phase} matchup {matchup_id} has unknown depth strata"
            )
        recomputed["pair_bootstrap_95"] = stratified_pair_bootstrap(
            strata,
            _derived_seed(
                manifest["seeds"]["bootstrap"][phase], phase, matchup_id
            ),
            manifest["statistics"]["bootstrap"]["resamples"],
        )
        recomputed["by_opening_depth"] = {
            str(depth): {
                "pairs": len(scores),
                "mean_pair_score": sum(scores) / len(scores),
                "pairs_won_2_0": sum(score == 1.0 for score in scores),
                "pairs_split_1_1": sum(score == 0.5 for score in scores),
                "pairs_lost_0_2": sum(score == 0.0 for score in scores),
            }
            for depth, scores in sorted(strata.items())
        }
        summary = _object(matchups[matchup_id], f"{phase}.matchups.{matchup_id}")
        if summary != recomputed:
            raise StudyError(
                f"{phase} matchup {matchup_id} summary differs from binary games"
            )
        if phase in ("development", "validation"):
            if expected["left"] in expected_strengths:
                raise StudyError(
                    f"{phase} tuning candidate occurs in multiple matchups"
                )
            expected_strengths[expected["left"]] = {
                "opponent_config_id": expected["right"],
                "mean_pair_score": recomputed["mean_pair_score"],
                "pair_bootstrap_95": recomputed["pair_bootstrap_95"],
                "pairs": recomputed["pairs"],
            }
            expected_paired_scores[expected["left"]] = dict(
                pair_scores_by_matchup[matchup_id]
            )

    for config_id in sorted(participating):
        config = _object(
            configurations[config_id], f"{phase}.configurations.{config_id}"
        )
        if "strength" not in config:
            raise StudyError(f"{phase} configuration {config_id} lacks strength")
        if config_id in expected_strengths:
            strength = _exact_keys(
                config["strength"],
                {
                    "opponent_config_id", "mean_pair_score",
                    "pair_bootstrap_95", "pairs",
                },
                f"{phase} strength {config_id}",
            )
            if strength != expected_strengths[config_id]:
                raise StudyError(
                    f"{phase} strength for {config_id} differs from binary games"
                )
        elif config["strength"] is not None:
            raise StudyError(
                f"{phase} non-tuning configuration {config_id} has a strength claim"
            )

    paired_scores = _object(curated.get("paired_scores"), f"{phase}.paired_scores")
    if set(paired_scores) != set(expected_paired_scores):
        raise StudyError(f"{phase} paired-score set differs from tuning matchups")
    for config_id, expected_rows in sorted(expected_paired_scores.items()):
        payload = _exact_keys(
            paired_scores[config_id],
            {
                "phase", "bot_id", "opponent_config_id", "opening_ids",
                "opening_depths", "scores",
            },
            f"{phase}.paired_scores.{config_id}",
        )
        expected_opponent = expected_strengths[config_id]["opponent_config_id"]
        if (payload["phase"], payload["bot_id"], payload["opponent_config_id"]) != (
                phase, config_id, expected_opponent):
            raise StudyError(f"{phase} paired-score identity changed for {config_id}")
        opening_ids = _array(
            payload["opening_ids"], f"{phase} paired opening IDs {config_id}"
        )
        opening_depths = _array(
            payload["opening_depths"], f"{phase} paired opening depths {config_id}"
        )
        scores = _array(payload["scores"], f"{phase} paired scores {config_id}")
        if not (len(opening_ids) == len(opening_depths) == len(scores)):
            raise StudyError(f"{phase} paired-score columns are misaligned for {config_id}")
        actual_rows: dict[tuple[int, str], float] = {}
        for index, (opening_id, depth, score) in enumerate(
                zip(opening_ids, opening_depths, scores, strict=True)):
            key = (
                _integer(depth, f"{phase} paired opening depth {index}", 1),
                _string(opening_id, f"{phase} paired opening ID {index}"),
            )
            value = _number(score, f"{phase} paired score {index}")
            if value not in (0.0, 0.5, 1.0) or key in actual_rows:
                raise StudyError(
                    f"{phase} paired-score row is invalid or duplicated for {config_id}"
                )
            actual_rows[key] = value
        if actual_rows != expected_rows:
            raise StudyError(
                f"{phase} paired scores for {config_id} differ from binary games"
            )


def _build_validation_pareto(
        analysis_module: Any, manifest: Mapping[str, Any],
        development: Mapping[str, Any], validation: Mapping[str, Any],
        validation_metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the validation frontier and copy all plotted sample provenance."""

    inputs: list[dict[str, Any]] = []
    for identifier, row in sorted(validation_metrics.items()):
        validation_summary = _object(
            validation["configurations"].get(identifier),
            f"validation configuration {identifier}",
        )
        development_summary = _object(
            development["configurations"].get(identifier),
            f"development configuration {identifier}",
        )
        validation_strength = _object(
            validation_summary.get("strength"),
            f"validation strength {identifier}",
        )
        development_strength = _object(
            development_summary.get("strength"),
            f"development strength {identifier}",
        )
        latency = _object(
            validation_summary.get("all_edge_latency"),
            f"validation latency {identifier}",
        )
        inputs.append({
            "id": identifier,
            "family": row["family"],
            "strength": row["validation_strength"],
            "p95_ms": row["validation_p95_ms"],
            "validation_strength": row["validation_strength"],
            "validation_strength_pairs": _integer(
                validation_strength.get("pairs"),
                f"validation strength pairs {identifier}", 1,
            ),
            "validation_strength_pair_bootstrap_95": dict(_object(
                validation_strength.get("pair_bootstrap_95"),
                f"validation strength interval {identifier}",
            )),
            "development_strength": _number(
                development_strength.get("mean_pair_score"),
                f"development strength {identifier}",
            ),
            "development_strength_pairs": _integer(
                development_strength.get("pairs"),
                f"development strength pairs {identifier}", 1,
            ),
            "development_strength_pair_bootstrap_95": dict(_object(
                development_strength.get("pair_bootstrap_95"),
                f"development strength interval {identifier}",
            )),
            "validation_p95_ms": row["validation_p95_ms"],
            "validation_latency_decisions": _integer(
                latency.get("decisions"),
                f"validation latency decisions {identifier}", 1,
            ),
            "strength_phases": ["development", "validation"],
            "latency_phase": "validation",
            "gate_eligible": row["eligible"],
            "selected": row["selected"],
            "fixed": False,
        })

    rank5 = _object(
        validation["configurations"].get("rank5-fixed-50k"),
        "validation Rank5 configuration",
    )
    fresh_latency = _object(
        rank5.get("fresh_root_latency"), "Rank5 fresh-root latency"
    )
    fresh_p95 = _number(fresh_latency.get("p95_ms"), "Rank5 fresh-root p95")
    rank5_eligible = fresh_p95 <= manifest["latency_protocol"]["gate_ms"]
    inputs.append({
        "id": "rank5-fixed-50k",
        "family": "rank5_derived",
        "strength": 0.5,
        "p95_ms": fresh_p95,
        "validation_strength": 0.5,
        "validation_strength_pairs": None,
        "validation_strength_pair_bootstrap_95": None,
        "development_strength": 0.5,
        "development_strength_pairs": None,
        "development_strength_pair_bootstrap_95": None,
        "validation_p95_ms": fresh_p95,
        "validation_latency_decisions": _integer(
            fresh_latency.get("decisions"), "Rank5 fresh-root decisions", 1
        ),
        "validation_comparator_pairs_total": sum(
            _integer(
                _object(
                    validation["configurations"][identifier].get("strength"),
                    f"validation strength {identifier}",
                ).get("pairs"),
                f"validation pairs {identifier}", 1,
            )
            for identifier in validation_metrics
        ),
        "strength_phases": ["development", "validation"],
        "latency_phase": "validation",
        "gate_eligible": rank5_eligible,
        "selected": True,
        "fixed": True,
        "strength_definition": "defined common-opponent reference level",
    })
    try:
        classified = analysis_module.classify_pareto(inputs)
    except analysis_module.AnalysisError as error:
        raise StudyError(f"Pareto classification failed: {error}") from error
    for point in classified:
        point.pop("strength", None)
        point.pop("p95_ms", None)
    return classified


def _select_family(manifest: Mapping[str, Any], validation: Mapping[str, Any],
                   family: str) -> tuple[str, list[dict[str, Any]]]:
    configs = configurations_by_id(manifest)
    rows: list[dict[str, Any]] = []
    for identifier in manifest["candidate_grids"][family]:
        summary = validation["configurations"].get(identifier)
        if not isinstance(summary, dict) or not isinstance(summary.get("strength"), dict):
            raise StudyError(f"validation summary is missing configuration {identifier}")
        strength = _number(summary["strength"]["mean_pair_score"],
                           f"{identifier} validation strength")
        latency = _number(summary["latency_gate_p95_ms"], f"{identifier} validation p95")
        rows.append({
            "id": identifier,
            "family": family,
            "validation_strength": strength,
            "validation_p95_ms": latency,
            "eligible": latency <= manifest["latency_protocol"]["gate_ms"],
            "budget": _configuration_budget(configs[identifier]),
        })
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise StudyError(
            f"no {family} configuration meets the 50 ms validation p95 gate; stop before test"
        )
    strongest = max(row["validation_strength"] for row in eligible)
    tied = [row for row in eligible
            if row["validation_strength"] >= strongest - 0.01]
    selected = min(
        tied,
        key=lambda row: (row["validation_p95_ms"], row["budget"], row["id"]),
    )
    for row in rows:
        row["within_practical_tie"] = (
            row["eligible"] and row["validation_strength"] >= strongest - 0.01
        )
        row["selected"] = row["id"] == selected["id"]
    return selected["id"], rows


def create_selection_lock(manifest_path: pathlib.Path, *, replace: bool = False) -> dict[str, Any]:
    from benchmarks.flagship_study import analysis

    repository = repository_root_from_manifest(manifest_path)
    manifest = validate_manifest(load_json(manifest_path), repository, verify_files=True)
    is_flagship = manifest.get("study", {}).get("study_class") == "flagship"
    verify_flagship_source_checkout(manifest, repository)
    manifest_hash = manifest_sha256(manifest_path)
    validation_path = repository / manifest["outputs"]["curated_data"]["validation"]
    development_path = repository / manifest["outputs"]["curated_data"]["development"]
    validation = _object(load_json(validation_path), "validation curated data")
    development = _object(load_json(development_path), "development curated data")
    if is_flagship:
        development = _assert_curated_matches_raw(
            manifest, repository, manifest_hash, "development"
        )
        validation = _assert_curated_matches_raw(
            manifest, repository, manifest_hash, "validation"
        )
        _, runtime_projection_hash = _validate_runtime_projection_artifact(
            manifest, repository, manifest_hash, verify_raw_derivation=True
        )
    else:
        runtime_projection_hash = None
    for phase, data in (("validation", validation), ("development", development)):
        if data.get("schema_version") != CURATED_SCHEMA_VERSION or \
           data.get("phase") != phase or data.get("manifest_sha256") != manifest_hash:
            raise StudyError(f"{phase} curated data does not match the frozen manifest")
        if data.get("completeness", {}).get("truncations") != 0 or \
           data.get("completeness", {}).get("operationally_valid") is not True:
            raise StudyError(f"{phase} data is not operationally valid")
        if is_flagship:
            _validate_curated_phase_contract(
                manifest, data, phase, manifest_hash
            )

    selected: dict[str, str] = {}
    validation_metrics: dict[str, Any] = {}
    for family in TUNABLE_FAMILIES:
        selected_id, rows = _select_family(manifest, validation, family)
        selected[family] = selected_id
        for row in rows:
            validation_metrics[row["id"]] = row

    rank5 = validation["configurations"].get("rank5-fixed-50k")
    if not isinstance(rank5, dict) or not isinstance(rank5.get("fresh_root_latency"), dict):
        raise StudyError("validation data lacks Rank5 fresh-root latency")
    rank5_fresh_p95 = _number(
        rank5["fresh_root_latency"].get("p95_ms"), "Rank5 fresh-root validation p95"
    )
    rank5_all_p95 = _number(
        rank5["all_edge_latency"].get("p95_ms"), "Rank5 all-edge validation p95"
    )
    rank5_eligible = rank5_fresh_p95 <= manifest["latency_protocol"]["gate_ms"]

    calibration_ids = set(selected.values()) | {"rank5-fixed-50k"}
    observations = _curated_calibration_observations(
        validation, "validation", calibration_ids
    )
    calibration_mappings: dict[str, Any] = {}
    for identifier in sorted(calibration_ids):
        try:
            mapping = _fit_curated_calibration(
                analysis, observations[identifier]
            )
        except analysis.AnalysisError as error:
            raise StudyError(
                f"validation calibration failed for {identifier}; stop before test: {error}"
            ) from error
        calibration_mappings[identifier] = mapping.to_dict()

    if is_flagship:
        classified_pareto = _build_validation_pareto(
            analysis, manifest, development, validation, validation_metrics
        )
        from benchmarks.flagship_study import ablations
        try:
            frozen_ablations: Any = ablations.compute(
                manifest, development, validation
            )
        except ablations.AblationError as error:
            raise StudyError(
                f"preregistered development/validation ablations failed: {error}"
            ) from error
    else:
        legacy_points: list[dict[str, Any]] = []
        for identifier, row in sorted(validation_metrics.items()):
            development_summary = development["configurations"].get(identifier, {})
            legacy_points.append({
                "id": identifier, "family": row["family"],
                "strength": row["validation_strength"],
                "development_strength": development_summary.get(
                    "strength", {}
                ).get("mean_pair_score"),
                "p95_ms": row["validation_p95_ms"],
                "strength_phases": ["development", "validation"],
                "latency_phase": "validation", "gate_eligible": row["eligible"],
                "selected": row["selected"], "fixed": False,
            })
        legacy_points.append({
            "id": "rank5-fixed-50k", "family": "rank5_derived",
            "strength": 0.5, "development_strength": 0.5,
            "p95_ms": rank5_fresh_p95,
            "strength_phases": ["development", "validation"],
            "latency_phase": "validation", "gate_eligible": rank5_eligible,
            "selected": True, "fixed": True,
            "strength_definition": "fixed common-opponent reference level",
        })
        classified_pareto = analysis.classify_pareto(legacy_points)
        frozen_ablations = None
    rank5_pareto = next(
        point for point in classified_pareto if point["id"] == "rank5-fixed-50k"
    )
    if not rank5_eligible:
        rank5_frontier_status = "outside_gate"
    elif rank5_pareto["constrained_pareto_optimal"]:
        rank5_frontier_status = "inside_constrained_frontier"
    else:
        rank5_frontier_status = "eligible_but_dominated"

    lock = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "manifest_path": str(manifest_path.resolve().relative_to(repository.resolve())),
        "source_phase": "validation",
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "opening_bank_sha256": {
            bank["id"]: bank["sha256"] for bank in manifest["openings"]["banks"]
        },
        "curated_input_sha256": {
            "development": sha256_file(development_path),
            "validation": sha256_file(validation_path),
        },
        "runtime_projection_sha256": runtime_projection_hash,
        "validation_execution_environments": validation["source"].get(
            "execution_environments", []
        ),
        "selection_rule": manifest["selection_rule"],
        "selected_configurations": selected,
        "fixed_rank5_configuration": "rank5-fixed-50k",
        "validation_metrics": validation_metrics,
        "rank5_latency": {
            "fresh_root_p95_ms": rank5_fresh_p95,
            "all_edge_p95_ms": rank5_all_p95,
            "eligible_under_50_ms": rank5_eligible,
            "test_inclusion": "fixed_unconstrained" if not rank5_eligible else "fixed_constrained_and_unconstrained",
            "constrained_frontier_status": rank5_frontier_status,
            "unconstrained_pareto_optimal": rank5_pareto[
                "unconstrained_pareto_optimal"
            ],
        },
        "calibration_seed": manifest["seeds"]["calibration"]["validation"],
        "calibration_mappings": calibration_mappings,
        "validation_pareto": classified_pareto,
        "development_validation_ablations": frozen_ablations,
        "test_authorized": True,
    }
    lock_path = _selection_path(manifest, repository)
    if lock_path.exists() and replace:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(lock_path.relative_to(repository))],
            cwd=repository, capture_output=True, text=True, check=True,
        ).stdout
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(lock_path.relative_to(repository))],
            cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if tracked and not status:
            raise StudyError("refusing to replace a committed selection lock")
    write_json_atomic(lock_path, lock, replace=replace)
    return {
        "selection_lock": str(lock_path.relative_to(repository)),
        "selection_lock_sha256": sha256_file(lock_path),
        "selected_configurations": selected,
        "rank5_fresh_root_p95_ms": rank5_fresh_p95,
        "rank5_all_edge_p95_ms": rank5_all_p95,
        "rank5_gate_eligible": rank5_eligible,
    }


def _test_pairs(test: Mapping[str, Any]) -> list[Any]:
    from benchmarks.flagship_study import analysis

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for game in _array(test.get("binary_games"), "test.binary_games"):
        game = _object(game, "binary game")
        grouped[_string(game.get("pair_id"), "binary game pair_id")].append(game)
    pairs: list[Any] = []
    for pair_id, games in sorted(grouped.items()):
        if len(games) != 2:
            raise StudyError(f"test pair {pair_id} does not contain exactly two games")
        matchup_id = games[0]["matchup_id"]
        if any(game["matchup_id"] != matchup_id for game in games):
            raise StudyError(f"test pair {pair_id} crosses matchups")
        matchup = _object(test["matchups"].get(matchup_id), f"test matchup {matchup_id}")
        left = matchup["left_config_id"]
        right = matchup["right_config_id"]
        outcomes = tuple(
            int(game["winner_config_id"] == left)
            for game in sorted(games, key=lambda value: value["game_id"])
        )
        if any(game["winner_config_id"] not in (left, right) for game in games):
            raise StudyError(f"test pair {pair_id} contains an impossible winner")
        pairs.append(analysis.PairedComparison(
            pair_id=pair_id,
            opening_depth=int(games[0]["opening_depth"]),
            bot_a=left,
            bot_b=right,
            outcomes_for_a=outcomes,
        ))
    return pairs


def _pairwise_conclusion(summary: Mapping[str, Any]) -> dict[str, Any]:
    interval = _object(summary.get("pair_bootstrap_95"), "pairwise interval")
    lower = _number(interval.get("lower"), "pairwise lower interval")
    upper = _number(interval.get("upper"), "pairwise upper interval")
    left = _string(summary.get("left_config_id"), "left config")
    right = _string(summary.get("right_config_id"), "right config")
    if lower > 0.5:
        return {"classification": "stronger", "stronger_config_id": left,
                "wording": f"{left} is stronger than {right}."}
    if 1.0 - upper > 0.5:
        return {"classification": "stronger", "stronger_config_id": right,
                "wording": f"{right} is stronger than {left}."}
    return {"classification": "statistically_unresolved", "stronger_config_id": None,
            "wording": f"{left} versus {right} is statistically unresolved."}


def _require_uncommitted_replacement(repository: pathlib.Path,
                                     path: pathlib.Path) -> None:
    try:
        relative = path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise StudyError(f"analysis artifact is outside repository: {path}") from error
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not tracked:
        return
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(relative)],
        cwd=repository, capture_output=True, text=True, check=True,
    ).stdout
    if not status:
        raise StudyError(f"refusing to replace committed analysis artifact: {relative}")


def _write_text_atomic(path: pathlib.Path, text_value: str, *, replace: bool,
                       repository: pathlib.Path | None = None) -> bool:
    data = text_value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return True
        if not replace:
            raise StudyError(f"refusing to replace different artifact: {path}")
        if repository is not None:
            _require_uncommitted_replacement(repository, path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return False


def analyze_test(manifest_path: pathlib.Path, *, replace: bool = False) -> dict[str, Any]:
    from benchmarks.flagship_study import analysis, charts, report as report_module

    repository = repository_root_from_manifest(manifest_path)
    manifest = validate_manifest(load_json(manifest_path), repository, verify_files=True)
    verify_flagship_source_checkout(manifest, repository)
    manifest_hash = manifest_sha256(manifest_path)
    selection_path = _selection_path(manifest, repository)
    selection = load_selection_lock(
        manifest, repository, manifest_hash, verify_raw_derivation=False
    )
    test_path = repository / manifest["outputs"]["curated_data"]["test"]
    development_path = repository / manifest["outputs"]["curated_data"]["development"]
    validation_path = repository / manifest["outputs"]["curated_data"]["validation"]
    test = _object(load_json(test_path), "test curated data")
    loaded_test_was_analyzed = bool(test.get("analysis_complete"))
    development = _object(load_json(development_path), "development curated data")
    validation = _object(load_json(validation_path), "validation curated data")
    if manifest.get("study", {}).get("study_class") == "flagship":
        test = _assert_curated_matches_raw(
            manifest, repository, manifest_hash, "test", selection,
            allow_analyzed_test=True,
        )
    for phase, value in (("development", development), ("validation", validation), ("test", test)):
        if value.get("schema_version") != CURATED_SCHEMA_VERSION or \
           value.get("phase") != phase or value.get("manifest_sha256") != manifest_hash:
            raise StudyError(f"{phase} curated data does not match the frozen manifest")
        if value.get("completeness", {}).get("truncations") != 0 or \
           value.get("completeness", {}).get("operationally_valid") is not True:
            raise StudyError(f"{phase} data is incomplete or contains truncations")
        if manifest.get("study", {}).get("study_class") == "flagship":
            _validate_curated_phase_contract(
                manifest, value, phase, manifest_hash,
                selection if phase == "test" else None,
            )

    pairs = _test_pairs(test)
    selected_ids = [selection["selected_configurations"][family]
                    for family in TUNABLE_FAMILIES]
    bot_ids = selected_ids + [selection["fixed_rank5_configuration"]]
    try:
        fitted = analysis.fit_bradley_terry(pairs, bot_ids=bot_ids)
        bt = analysis.bootstrap_bradley_terry(
            pairs,
            seed=_derived_seed(manifest["seeds"]["analysis"]["test"], "bradley-terry"),
            bot_ids=bot_ids,
            resamples=manifest["statistics"]["bootstrap"]["resamples"],
            minimum_success_fraction=manifest["statistics"]["bradley_terry"]
                ["minimum_bootstrap_success_fraction"],
        )
    except analysis.AnalysisError as error:
        raise StudyError(f"Bradley-Terry test analysis failed: {error}") from error
    bt["point_fit"] = fitted.to_dict()

    observations = _curated_calibration_observations(
        test, "test", set(bot_ids)
    )
    calibration: dict[str, Any] = {}
    for identifier in bot_ids:
        mapping = selection["calibration_mappings"].get(identifier)
        if not isinstance(mapping, dict):
            raise StudyError(f"selection lock lacks calibration mapping for {identifier}")
        try:
            calibration[identifier] = _evaluate_curated_calibration(
                analysis, mapping, observations[identifier],
                **_calibration_evaluation_options(manifest, identifier),
            )
        except analysis.AnalysisError as error:
            raise StudyError(f"test calibration failed for {identifier}: {error}") from error

    enriched = dict(test)
    enriched["matchups"] = {
        matchup_id: {**summary, "conclusion": _pairwise_conclusion(summary)}
        for matchup_id, summary in sorted(test["matchups"].items())
    }
    enriched["bradley_terry"] = bt
    enriched["calibration"] = calibration
    enriched["validation_pareto"] = selection["validation_pareto"]
    enriched["sample_sizes"] = {
        "games": len(test["binary_games"]),
        "pairs": len(pairs),
        "opening_depths": list(EXPECTED_OPENING_DEPTHS),
        "bootstrap_resamples": manifest["statistics"]["bootstrap"]["resamples"],
    }
    enriched["analysis_complete"] = True
    already_analyzed = loaded_test_was_analyzed
    enriched_bytes = canonical_json_bytes(enriched)
    test_data_resumed = test_path.read_bytes() == enriched_bytes
    if not test_data_resumed:
        if already_analyzed and not replace:
            raise StudyError(
                "refusing to replace different completed test analysis"
            )
        _require_uncommitted_replacement(repository, test_path)
        write_json_atomic(test_path, enriched, replace=True)

    configs = configurations_by_id(manifest)
    labels = {identifier: configs[identifier]["public_label"] for identifier in configs}
    # Candidate-budget suffixes keep chart labels distinct without changing the
    # centrally controlled entrant names.
    for identifier, config in configs.items():
        if config["kind"] == "mcts":
            labels[identifier] = f"Tactical MctsBot ({config['settings']['iterations']} iter)"
        elif config["kind"] in ("alpha-beta", "jacek-inspired"):
            labels[identifier] = (
                f"{config['public_label']} ({config['settings']['max_nodes'] // 1000}k nodes)"
            )

    chart_values = {
        "bradley_terry": charts.bradley_terry_svg(enriched, labels),
        "pareto": charts.pareto_svg(selection, labels),
        "calibration": charts.calibration_svg(enriched, labels),
    }
    chart_hashes: dict[str, str] = {}
    charts_resumed = True
    for key, svg in chart_values.items():
        chart_path = repository / manifest["outputs"]["charts"][key]
        charts_resumed = _write_text_atomic(
            chart_path, svg, replace=replace, repository=repository
        ) and charts_resumed
        chart_hashes[key] = sha256_file(chart_path)

    artifact_hashes = {
        str(manifest_path.resolve().relative_to(repository.resolve())): manifest_hash,
        manifest["outputs"]["selection_lock"]: sha256_file(selection_path),
        manifest["outputs"]["curated_data"]["development"]: sha256_file(development_path),
        manifest["outputs"]["curated_data"]["validation"]: sha256_file(validation_path),
        manifest["outputs"]["curated_data"]["test"]: sha256_file(test_path),
        manifest["outputs"]["charts"]["bradley_terry"]: chart_hashes["bradley_terry"],
        manifest["outputs"]["charts"]["pareto"]: chart_hashes["pareto"],
        manifest["outputs"]["charts"]["calibration"]: chart_hashes["calibration"],
    }
    if manifest.get("study", {}).get("study_class") == "flagship":
        artifact_hashes[manifest["outputs"]["runtime_projection"]] = (
            selection["runtime_projection_sha256"]
        )
    try:
        report_text = report_module.render_report(
            manifest, selection, development, validation, enriched, artifact_hashes
        )
    except ValueError as error:
        raise StudyError(f"report generation failed: {error}") from error
    report_path = repository / manifest["outputs"]["report"]
    report_resumed = _write_text_atomic(
        report_path, report_text, replace=replace, repository=repository
    )
    return {
        "test_data": str(test_path.relative_to(repository)),
        "test_data_sha256": sha256_file(test_path),
        "bradley_terry": bt,
        "calibration": calibration,
        "chart_sha256": chart_hashes,
        "report": str(report_path.relative_to(repository)),
        "report_sha256": sha256_file(report_path),
        "truncations": 0,
        "resumed": test_data_resumed and charts_resumed and report_resumed,
    }


def _project_runtime_from_samples(
        manifest: Mapping[str, Any], manifest_hash: str,
        samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the preregistered conservative projection from exact-budget samples."""

    if manifest.get("study", {}).get("study_class") != "flagship":
        raise StudyError("representative runtime projection is only defined for flagship")
    candidate_ids = tuple(
        identifier
        for family in TUNABLE_FAMILIES
        for identifier in manifest["candidate_grids"][family]
    )
    if len(candidate_ids) != 9 or len(set(candidate_ids)) != 9:
        raise StudyError("runtime projection requires all nine candidate configurations")

    development_units = units_for_phase(manifest, "development")
    validation_units = units_for_phase(manifest, "validation")
    required_depths = (4, 20)
    required_units: dict[tuple[str, int], StudyUnit] = {}
    for unit in development_units:
        if unit.opening_depth not in required_depths:
            continue
        key = (unit.left_config_id, unit.opening_depth)
        if key in required_units:
            raise StudyError(f"duplicate representative runtime unit for {key}")
        required_units[key] = unit
    expected_keys = {
        (identifier, depth)
        for identifier in candidate_ids
        for depth in required_depths
    }
    if set(required_units) != expected_keys or len(development_units) != 36:
        raise StudyError(
            "flagship runtime design must cover 18 of 36 development units"
        )
    if len(validation_units) != 36:
        raise StudyError("flagship runtime design must contain 36 validation units")

    samples_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    expected_games_per_unit = (
        manifest["samples"]["development"]
        ["color_swapped_pairs_per_depth_matchup"] * 2
    )
    for index, raw_sample in enumerate(samples):
        sample = _exact_keys(
            raw_sample,
            {"unit_id", "games", "wall_seconds", "opening_depth",
             "left_config_id", "right_config_id"},
            f"runtime samples[{index}]",
        )
        identifier = _string(
            sample["left_config_id"], f"runtime samples[{index}].left_config_id"
        )
        depth = _integer(
            sample["opening_depth"], f"runtime samples[{index}].opening_depth", 1
        )
        key = (identifier, depth)
        if key not in required_units:
            raise StudyError(f"runtime sample is outside preregistered coverage: {key}")
        if key in samples_by_key:
            raise StudyError(f"duplicate runtime sample for {key}")
        unit = required_units[key]
        games = _integer(sample["games"], f"runtime samples[{index}].games", 1)
        wall_seconds = _number(
            sample["wall_seconds"], f"runtime samples[{index}].wall_seconds"
        )
        if wall_seconds <= 0.0:
            raise StudyError("runtime sample wall time must be positive")
        if games != expected_games_per_unit or games != unit.pairs * 2:
            raise StudyError(
                f"runtime sample does not use the exact pair budget for {unit.unit_id}"
            )
        if sample["unit_id"] != unit.unit_id or \
           sample["right_config_id"] != unit.right_config_id:
            raise StudyError(f"runtime sample identity mismatch for {unit.unit_id}")
        samples_by_key[key] = {
            "unit_id": unit.unit_id,
            "games": games,
            "wall_seconds": wall_seconds,
            "seconds_per_game": wall_seconds / games,
            "opening_depth": depth,
            "left_config_id": identifier,
            "right_config_id": unit.right_config_id,
        }
    missing = sorted(expected_keys - set(samples_by_key))
    if missing:
        missing_text = ", ".join(f"{identifier}@d{depth}" for identifier, depth in missing)
        raise StudyError(
            "runtime projection requires exact-budget development coverage for "
            f"all candidates at depths 4 and 20; missing {missing_text}"
        )
    if len(samples_by_key) != 18:
        raise StudyError("runtime projection requires exactly 18 representative samples")

    observed_rates: dict[str, dict[str, Any]] = {}
    rate_ranges: dict[str, dict[str, float]] = {}
    for identifier in candidate_ids:
        by_depth: dict[str, Any] = {}
        rates = []
        for depth in required_depths:
            sample = samples_by_key[(identifier, depth)]
            rates.append(sample["seconds_per_game"])
            by_depth[str(depth)] = dict(sample)
        observed_rates[identifier] = by_depth
        rate_ranges[identifier] = {
            "minimum_observed_seconds_per_game": min(rates),
            "maximum_observed_seconds_per_game": max(rates),
            "unmeasured_depth_lower_seconds_per_game": min(rates),
            "unmeasured_depth_conservative_seconds_per_game": max(rates),
        }

    remaining_development = [
        unit for unit in development_units
        if (unit.left_config_id, unit.opening_depth) not in expected_keys
    ]

    def candidate_rate(unit: StudyUnit, conservative: bool) -> float:
        if unit.opening_depth in required_depths:
            return samples_by_key[
                (unit.left_config_id, unit.opening_depth)
            ]["seconds_per_game"]
        key = (
            "maximum_observed_seconds_per_game" if conservative
            else "minimum_observed_seconds_per_game"
        )
        return rate_ranges[unit.left_config_id][key]

    development_lower = sum(
        unit.pairs * 2 * candidate_rate(unit, False)
        for unit in remaining_development
    )
    development_conservative = sum(
        unit.pairs * 2 * candidate_rate(unit, True)
        for unit in remaining_development
    )
    validation_lower = sum(
        unit.pairs * 2 * candidate_rate(unit, False)
        for unit in validation_units
    )
    validation_conservative = sum(
        unit.pairs * 2 * candidate_rate(unit, True)
        for unit in validation_units
    )

    test_schedule_count = len(manifest["schedule"]["test"])
    test_banks = sorted(
        (bank for bank in manifest["openings"]["banks"]
         if bank["phase"] == "test"),
        key=lambda bank: bank["depth"],
    )
    if test_schedule_count != 6 or len(test_banks) != 4:
        raise StudyError(
            "flagship runtime design must contain six test matchups at four depths"
        )
    test_rate_ranges: dict[str, dict[str, float]] = {}
    test_games = 0
    test_lower = 0.0
    test_conservative = 0.0
    for bank in test_banks:
        depth = bank["depth"]
        if depth in required_depths:
            depth_rates = [
                samples_by_key[(identifier, depth)]["seconds_per_game"]
                for identifier in candidate_ids
            ]
            lower_rate = max(depth_rates)
            upper_rate = lower_rate
        else:
            lower_rate = max(
                value["minimum_observed_seconds_per_game"]
                for value in rate_ranges.values()
            )
            upper_rate = max(
                value["maximum_observed_seconds_per_game"]
                for value in rate_ranges.values()
            )
        games_at_depth = bank["pairs"] * 2 * test_schedule_count
        test_games += games_at_depth
        test_lower += games_at_depth * lower_rate
        test_conservative += games_at_depth * 2.0 * upper_rate
        test_rate_ranges[str(depth)] = {
            "single_observed_matchup_proxy_seconds_per_game": lower_rate,
            "two_expensive_entrants_conservative_seconds_per_game": 2.0 * upper_rate,
        }

    def workload(units: int, games: int, lower: float,
                 conservative: float) -> dict[str, Any]:
        return {
            "units": units,
            "games": games,
            "range_seconds": {
                "lower_proxy": lower,
                "conservative": conservative,
            },
            "range_hours": {
                "lower_proxy": lower / 3600.0,
                "conservative": conservative / 3600.0,
            },
        }

    development_games = sum(unit.pairs * 2 for unit in remaining_development)
    validation_games = sum(unit.pairs * 2 for unit in validation_units)
    workloads = {
        "remaining_development": workload(
            len(remaining_development), development_games,
            development_lower, development_conservative,
        ),
        "full_validation": workload(
            len(validation_units), validation_games,
            validation_lower, validation_conservative,
        ),
        "full_test": workload(
            test_schedule_count * len(test_banks), test_games,
            test_lower, test_conservative,
        ),
    }
    total_units = sum(value["units"] for value in workloads.values())
    total_games = sum(value["games"] for value in workloads.values())
    total_lower = sum(
        value["range_seconds"]["lower_proxy"] for value in workloads.values()
    )
    total_conservative = sum(
        value["range_seconds"]["conservative"] for value in workloads.values()
    )
    workloads["total_remaining"] = workload(
        total_units, total_games, total_lower, total_conservative
    )

    ordered_samples = [
        samples_by_key[(identifier, depth)]
        for identifier in candidate_ids
        for depth in required_depths
    ]
    return {
        "schema": RUNTIME_PROJECTION_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "coverage": {
            "phase": "development",
            "required_opening_depths": list(required_depths),
            "candidate_configurations": list(candidate_ids),
            "required_units": 18,
            "completed_required_units": len(ordered_samples),
            "total_development_units": len(development_units),
            "coverage_fraction": len(ordered_samples) / len(development_units),
            "observed_games": sum(sample["games"] for sample in ordered_samples),
            "observed_wall_seconds": sum(
                sample["wall_seconds"] for sample in ordered_samples
            ),
            "required_unit_ids": [sample["unit_id"] for sample in ordered_samples],
        },
        "observed_rates_by_configuration_and_depth": observed_rates,
        "observed_rate_ranges_by_configuration": rate_ranges,
        "test_rate_ranges_by_depth": test_rate_ranges,
        "projected_workloads": workloads,
        "assumptions": {
            "coverage_gate": (
                "All nine exact-budget candidate-vs-fixed development units at opening "
                "depths 4 and 20 must complete before projection."
            ),
            "intermediate_depths": (
                "Depths 8 and 12 use each candidate's observed depth-4/depth-20 range; "
                "the conservative estimate uses the slower endpoint."
            ),
            "validation": (
                "Full validation preserves candidate, depth, and exact search budget and "
                "scales only the preregistered pair count."
            ),
            "test": (
                "Before selection, every test game uses the slowest candidate rate at its "
                "depth; the conservative bound doubles that rate for two expensive entrants."
            ),
            "remaining_scope": (
                "The 18 observed representative development units are sunk elapsed work; "
                "totals include only remaining development plus full validation and test."
            ),
        },
    }


def _runtime_projection_from_raw(
        manifest: Mapping[str, Any], repository: pathlib.Path,
        manifest_hash: str) -> dict[str, Any]:
    """Rebuild the projection from the exact preregistered raw shards."""

    if manifest["study"]["study_class"] != "flagship":
        raise StudyError("representative runtime projection is only defined for flagship")
    configs = configurations_by_id(manifest)
    representative_units = [
        unit for unit in units_for_phase(manifest, "development")
        if unit.opening_depth in (4, 20)
    ]
    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    for unit in representative_units:
        path = (_raw_root(manifest, repository, manifest_hash) /
                "development" / "shards" / f"{unit.unit_id}.json")
        if not path.is_file():
            missing.append(unit.unit_id)
            continue
        report = _object(load_json(path), str(path))
        study = _object(report.get("study"), f"{path}.study")
        if study.get("manifest_sha256") != manifest_hash or \
           study.get("phase") != "development" or \
           study.get("unit_id") != unit.unit_id or \
           study.get("matchup_id") != unit.matchup_id or \
           study.get("left_config_id") != unit.left_config_id or \
           study.get("right_config_id") != unit.right_config_id or \
           study.get("opening_depth") != unit.opening_depth:
            raise StudyError(f"runtime sample does not match manifest unit: {path}")
        _verify_arena_report_contract(report, unit, manifest, configs)
        wall_seconds = _number(study.get("wall_seconds"), f"{path}.wall_seconds")
        games = _array(report.get("games"), f"{path}.games")
        summary = _object(report.get("summary"), f"{path}.summary")
        truncated = any(
            _object(
                _object(game, "runtime game").get("outcome"),
                "runtime game outcome",
            ).get("truncated") is True
            for game in games
        )
        if wall_seconds <= 0.0 or len(games) != unit.pairs * 2 or \
           summary.get("truncations") != 0 or truncated:
            raise StudyError(f"runtime sample is incomplete or truncated: {path}")
        samples.append({
            "unit_id": unit.unit_id,
            "games": len(games),
            "wall_seconds": wall_seconds,
            "opening_depth": unit.opening_depth,
            "left_config_id": unit.left_config_id,
            "right_config_id": unit.right_config_id,
        })
    if missing:
        raise StudyError(
            "runtime projection requires all 18 exact-budget development shards at "
            "depths 4 and 20; missing " + ", ".join(sorted(missing))
        )
    return _project_runtime_from_samples(manifest, manifest_hash, samples)


def _validate_runtime_projection_artifact(
        manifest: Mapping[str, Any], repository: pathlib.Path,
        manifest_hash: str, *, verify_raw_derivation: bool) -> tuple[pathlib.Path, str]:
    """Validate the projection identity and optionally reproduce every byte."""

    path = repository / manifest["outputs"]["runtime_projection"]
    projection = _exact_keys(
        load_json(path),
        {
            "schema", "manifest_sha256", "coverage",
            "observed_rates_by_configuration_and_depth",
            "observed_rate_ranges_by_configuration",
            "test_rate_ranges_by_depth", "projected_workloads", "assumptions",
        },
        "runtime projection",
    )
    if projection["schema"] != RUNTIME_PROJECTION_SCHEMA_VERSION:
        raise StudyError("unsupported runtime-projection schema")
    if projection["manifest_sha256"] != manifest_hash:
        raise StudyError("runtime projection belongs to another manifest")
    if verify_raw_derivation:
        expected = canonical_json_bytes(
            _runtime_projection_from_raw(manifest, repository, manifest_hash)
        )
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise StudyError(f"could not read runtime projection {path}: {error}") from error
        if actual != expected:
            raise StudyError("runtime projection is not reproducible from raw shards")
    return path, sha256_file(path)


def project_runtime(manifest_path: pathlib.Path, *, write: bool = False,
                    replace: bool = False) -> dict[str, Any]:
    """Project remaining workloads after representative development coverage."""

    repository = repository_root_from_manifest(manifest_path)
    manifest = validate_manifest(load_json(manifest_path), repository, verify_files=True)
    verify_flagship_source_checkout(manifest, repository)
    manifest_hash = manifest_sha256(manifest_path)
    projection = _runtime_projection_from_raw(manifest, repository, manifest_hash)
    if write:
        path = repository / manifest["outputs"]["runtime_projection"]
        write_json_atomic(path, projection, replace=replace)
        projection = dict(projection)
        projection["written_path"] = str(path.relative_to(repository))
        projection["written_sha256"] = sha256_file(path)
    return projection
