#!/usr/bin/env python3
"""Fail closed if the native Jacek track acquires incumbent dependencies."""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import sys


BOT_DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = BOT_DIRECTORY.parents[3]
EXPECTED_SOURCE_LIMIT = 94_999
ROUND1_PURITY_DEPENDENCIES = (
    "models/jacek_native_bootstrap_model.json",
    "models/jacek_native_untrained_seed.json",
    "models/jacek_native_untrained_seed.runtime",
    "submissions/codingame/tools/generate_jacek_native_model.py",
    "tools/generate_jacek_native_seed.py",
    "tools/train_jacek_native.py",
    "tools/jacek_native_selfplay.cpp",
    "tools/jacek_native_workflow.py",
)
ROUND1_SEMANTIC_DEPENDENCIES = ("tools/jacek_native_corpus.py",)
ROUND2_PURITY_DEPENDENCIES = (
    "models/jacek_native_round2_candidate.json",
    "models/jacek_native_untrained_seed.json",
    "models/jacek_native_untrained_seed.runtime",
    "submissions/codingame/tools/generate_jacek_native_model_round2.py",
    "submissions/codingame/tools/jacek_native_round2_activation.py",
    "submissions/codingame/tools/generate_jacek_native_model.py",
    "tools/jacek_native_round2_selection.py",
    "tools/generate_jacek_native_seed.py",
    "tools/train_jacek_native_round2.py",
    "tools/train_jacek_native.py",
    "tools/jacek_native_selfplay_round2.cpp",
    "tools/jacek_native_restart_round2.cpp",
    "tools/jacek_native_selfplay.cpp",
    "tools/jacek_native_workflow_round2.py",
    "tools/jacek_native_restart_round2.py",
    "tools/jacek_native_workflow.py",
)
ROUND2_DEPLOYMENT_PATH = "models/jacek_native_round2_deployment.json"
ROUND2_SELECTION_PATH = "models/jacek_native_round2_selection.json"
ROUND2_RUNTIME_PATH = "models/jacek_native_round2_selected.runtime"
ROUND2_SEMANTIC_DEPENDENCIES = (
    "tools/jacek_native_corpus_round2.py",
    "tools/jacek_native_restart_corpus_round2.py",
    "tools/jacek_native_corpus.py",
)
# Backward-compatible names are the exact frozen round-one contract.
EXPECTED_PURITY_DEPENDENCIES = ROUND1_PURITY_DEPENDENCIES
EXPECTED_SEMANTIC_DEPENDENCIES = ROUND1_SEMANTIC_DEPENDENCIES
RUNTIME_SCHEMA = "papersoccer.jacek-native-runtime-model/v1"
MODEL_SCHEMA = "jacek_native_model/v1"
FEATURE_SCHEMA = "canonical-edges316-onehot-true-turn-distance105x8-v1"
MODEL_SHAPES = {"w1": (1156, 32), "w2": (32, 32), "w3": (32,)}
HISTORICAL_ROUND2_BASELINES = {
    "b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14": {
        "trainer_sha256": (
            "a84b038e675765ed7e849fea89583dc6188fdd0635b58ef79afb655796a5d711"
        ),
        "corpus_validator_sha256": (
            "bd908ec96f3e64b010cfe07404e005ad3587d8b354b1f327659e7f9575827f74"
        ),
        "restart_corpus_validator_sha256": (
            "990a0f99745fe0420ea446d9553eea25de725e8366946fd6b7f311e2b0b5981f"
        ),
        "runtime_sha256": (
            "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3"
        ),
        "retained_runtime_sha256": {
            "1f8d5696742c94aec74df1cf107cf478cdf885d1a9866b46a93e4ec32359b9c8",
            "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3",
            "d0d5478385f52dd80b8073ce24ae587afe7f8b55d3123f0939d55716b5451ed0",
        },
    },
}
BUILD_PROVENANCE_SCHEMA = "papersoccer.jacek-native-build-provenance/v1"
ROUND2_BUILD_PROVENANCE_SCHEMA = (
    "papersoccer.jacek-native-build-provenance/v2"
)
RESTART_BUILD_PROVENANCE_SCHEMA = (
    "papersoccer.jacek-native-restart-build-provenance/v1"
)
BUILD_SOURCE_PATHS = (
    "tools/jacek_native_selfplay.cpp",
    "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
    "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "src/bots/mcts_internal.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
)
CANONICAL_BUILD_ARGV = (
    "$CXX",
    "-std=c++20",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Iinclude",
    "-Isrc/bots",
    "tools/jacek_native_selfplay.cpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "-o",
    "$OUTPUT",
)
ROUND2_BUILD_SOURCE_PATHS = (
    "tools/jacek_native_selfplay_round2.cpp",
    *BUILD_SOURCE_PATHS,
)
ROUND2_CANONICAL_BUILD_ARGV = (
    "$CXX",
    "-std=c++20",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Iinclude",
    "-Isrc/bots",
    "tools/jacek_native_selfplay_round2.cpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "-o",
    "$OUTPUT",
)
RESTART_BUILD_SOURCE_PATHS = (
    "tools/jacek_native_restart_round2.cpp",
    *ROUND2_BUILD_SOURCE_PATHS,
)
RESTART_CANONICAL_BUILD_ARGV = (
    "$CXX", "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra",
    "-Wpedantic", "-Iinclude", "-Isrc/bots",
    "tools/jacek_native_restart_round2.cpp", "src/core/rules.cpp",
    "src/core/geometry.cpp", "-o", "$OUTPUT",
)

BANNED_PATTERNS = (
    (re.compile(r"rank[_-]?4", re.IGNORECASE), "rank-4 dependency"),
    (re.compile(r"replay[_-]?book", re.IGNORECASE), "replay-book dependency"),
    (
        re.compile(r"replay[_-]?value[_-]?model", re.IGNORECASE),
        "replay-value dependency",
    ),
    (
        re.compile(r"teacher[_-]?residual", re.IGNORECASE),
        "teacher-residual dependency",
    ),
    (re.compile(r"alpha[_-]?beta", re.IGNORECASE), "alpha-beta dependency"),
    (re.compile(r"\bteacher\b", re.IGNORECASE), "teacher-label dependency"),
)
ROUND2_BANNED_PATTERNS = tuple(
    entry for entry in BANNED_PATTERNS
    if entry[1] != "teacher-label dependency"
) + ((
    re.compile(
        r"teacher[_-]?(?:move|action|policy|label)", re.IGNORECASE
    ),
    "teacher action-label dependency",
),)


def _contained(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    if not relative or pathlib.PurePath(relative).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed directory: {relative}") from error
    return resolved


def _manifest_sources(path: pathlib.Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _purity_track(config: dict) -> str:
    dependencies = config.get("purity_dependencies")
    semantic = config.get("purity_semantic_dependencies")
    if (
        dependencies == list(ROUND1_PURITY_DEPENDENCIES)
        and semantic == list(ROUND1_SEMANTIC_DEPENDENCIES)
    ):
        return "round1"
    if (
        dependencies == list(ROUND2_PURITY_DEPENDENCIES)
        and semantic == list(ROUND2_SEMANTIC_DEPENDENCIES)
    ):
        return "round2"
    raise ValueError(
        "purity dependencies do not match a frozen native round contract"
    )


def production_files(
    bot_directory: pathlib.Path, repository_root: pathlib.Path
) -> tuple[dict, list[pathlib.Path]]:
    config_path = bot_directory / "submission.json"
    if not config_path.is_file():
        raise ValueError("missing submission.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "papersoccer.codingame-submission.v1":
        raise ValueError("unexpected submission schema")
    if config.get("source_limit") != EXPECTED_SOURCE_LIMIT:
        raise ValueError(
            f"source_limit must be exactly {EXPECTED_SOURCE_LIMIT}, got "
            f"{config.get('source_limit')!r}"
        )

    manifest_path = _contained(
        bot_directory, config.get("sources", "sources.txt"), "sources manifest"
    )
    if not manifest_path.is_file():
        raise ValueError("missing sources manifest")
    source_names = _manifest_sources(manifest_path)
    if not source_names:
        raise ValueError("sources manifest is empty")

    files = [config_path, manifest_path]
    files.extend(
        _contained(repository_root, relative, "production source")
        for relative in source_names
    )
    files.extend(
        _contained(bot_directory, relative, "generator")
        for relative in config.get("generators", [])
    )
    _purity_track(config)
    dependencies = config["purity_dependencies"]
    semantic_dependencies = config["purity_semantic_dependencies"]
    files.extend(
        _contained(repository_root, relative, "purity dependency")
        for relative in dependencies
    )
    native_checkpoints = config.get("native_checkpoint_provenance", [])
    if not isinstance(native_checkpoints, list):
        raise ValueError("native_checkpoint_provenance must be an array")
    seen_checkpoint_pairs: set[tuple[pathlib.Path, pathlib.Path]] = set()
    seen_runtime_paths: set[pathlib.Path] = set()
    for index, entry in enumerate(native_checkpoints):
        if not isinstance(entry, dict) or set(entry) != {"model", "runtime"}:
            raise ValueError(
                f"native checkpoint {index} must name exactly model and runtime"
            )
        model_path = _contained(
            repository_root, entry["model"], "native checkpoint model"
        )
        runtime_path = _contained(
            repository_root, entry["runtime"], "native checkpoint runtime"
        )
        if model_path == (
            repository_root / dependencies[0]
        ).resolve():
            raise ValueError(
                "native checkpoint provenance must not self-reference the "
                "active model"
            )
        pair = (model_path, runtime_path)
        if pair in seen_checkpoint_pairs or runtime_path in seen_runtime_paths:
            raise ValueError("native checkpoint declarations must be unique")
        seen_checkpoint_pairs.add(pair)
        seen_runtime_paths.add(runtime_path)
        if model_path not in files:
            files.append(model_path)
        files.append(runtime_path)
    output = _contained(
        bot_directory, config.get("output", "submission.cpp"), "output"
    )
    if output.exists():
        files.append(output)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ValueError(
            "missing production file(s): "
            + ", ".join(str(path) for path in missing)
        )
    return config, files


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_build_provenance(
    model: dict, label: str, track: str = "round1"
) -> None:
    generation = model.get("provenance", {}).get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label} generation provenance is missing")
    declared_hashes = generation.get("build_provenance_sha256")
    contracts = generation.get("build_contracts")
    if (
        not isinstance(declared_hashes, list)
        or not declared_hashes
        or declared_hashes != sorted(set(declared_hashes))
        or not all(_valid_sha256(value) for value in declared_hashes)
        or not isinstance(contracts, list)
        or len(contracts) != len(declared_hashes)
    ):
        raise ValueError(f"{label} build-provenance index is invalid")
    observed_hashes: list[str] = []
    producers: set[str] = set()
    for item in contracts:
        if not isinstance(item, dict) or set(item) != {"sha256", "contract"}:
            raise ValueError(f"{label} build contract entry is invalid")
        digest = item.get("sha256")
        contract = item.get("contract")
        if not _valid_sha256(digest) or not isinstance(contract, dict):
            raise ValueError(f"{label} build contract identity is invalid")
        canonical = (
            json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != digest:
            raise ValueError(f"{label} build contract SHA-256 is stale")
        rendered = canonical.decode().lower()
        sensitive_tokens = (
            "matches.json", "protected-bank", "protected_bank",
            "sealed-bank", "sealed_bank", "/users/", "/home/", "\\users\\",
        )
        if any(token in rendered for token in sensitive_tokens):
            raise ValueError(f"{label} build contract exposes a sensitive path")
        if set(contract) != {
            "schema",
            "binary",
            "compiler",
            "build_argv",
            "producer_sha256",
            "sources",
        }:
            raise ValueError(f"{label} build contract schema is not frozen")
        schema = contract.get("schema")
        if schema == BUILD_PROVENANCE_SCHEMA:
            expected_argv = CANONICAL_BUILD_ARGV
            expected_sources = BUILD_SOURCE_PATHS
            expected_binary = "selfplay-binary"
        elif track == "round2" and schema == ROUND2_BUILD_PROVENANCE_SCHEMA:
            expected_argv = ROUND2_CANONICAL_BUILD_ARGV
            expected_sources = ROUND2_BUILD_SOURCE_PATHS
            expected_binary = "selfplay-round2-binary"
        elif track == "round2" and schema == RESTART_BUILD_PROVENANCE_SCHEMA:
            expected_argv = RESTART_CANONICAL_BUILD_ARGV
            expected_sources = RESTART_BUILD_SOURCE_PATHS
            expected_binary = "selfplay-restart-round2-binary"
        else:
            raise ValueError(f"{label} build contract schema is not frozen")
        if contract.get("build_argv") != list(expected_argv):
            raise ValueError(f"{label} build argv is not frozen")
        binary = contract.get("binary")
        if (
            not isinstance(binary, dict)
            or set(binary) != {"path", "sha256"}
            or binary.get("path") != expected_binary
            or not _valid_sha256(binary.get("sha256"))
        ):
            raise ValueError(f"{label} build binary identity is invalid")
        compiler = contract.get("compiler")
        if (
            not isinstance(compiler, dict)
            or set(compiler)
            != {"executable", "sha256", "version", "version_sha256"}
            or not isinstance(compiler.get("executable"), str)
            or not compiler["executable"]
            or pathlib.PurePath(compiler["executable"]).name
            != compiler["executable"]
            or not _valid_sha256(compiler.get("sha256"))
            or not isinstance(compiler.get("version"), str)
            or not compiler["version"]
            or not _valid_sha256(compiler.get("version_sha256"))
            or hashlib.sha256(compiler["version"].encode()).hexdigest()
            != compiler["version_sha256"]
        ):
            raise ValueError(f"{label} build compiler identity is invalid")
        sources = contract.get("sources")
        if not isinstance(sources, list) or len(sources) != len(
            expected_sources
        ):
            raise ValueError(f"{label} build source list is incomplete")
        source_pairs: list[list[str]] = []
        for expected_path, source in zip(expected_sources, sources):
            if (
                not isinstance(source, dict)
                or set(source) != {"path", "sha256"}
                or source.get("path") != expected_path
                or not _valid_sha256(source.get("sha256"))
            ):
                raise ValueError(f"{label} build source identity is invalid")
            source_pairs.append([source["path"], source["sha256"]])
        producer = hashlib.sha256(json.dumps(
            source_pairs, separators=(",", ":")
        ).encode()).hexdigest()
        if contract.get("producer_sha256") != producer:
            raise ValueError(f"{label} build producer SHA-256 is inconsistent")
        producers.add(producer)
        observed_hashes.append(digest)
    if observed_hashes != declared_hashes:
        raise ValueError(f"{label} build contracts are not canonical")
    if generation.get("producer_sha256") != sorted(producers):
        raise ValueError(f"{label} producer/build provenance disagrees")


def _validate_native_model_provenance(
    model_path: pathlib.Path,
    trainer_path: pathlib.Path,
    corpus_path: pathlib.Path,
    track: str = "round1",
) -> dict:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("schema") != MODEL_SCHEMA or model.get(
        "feature_schema"
    ) != FEATURE_SCHEMA:
        raise ValueError("native model schema is not frozen")
    target = model.get("target")
    if (
        not isinstance(target, dict)
        or target.get("primary") != "mover-relative-final-outcome"
        or target.get("policy_target") is not None
    ):
        raise ValueError("native model target permits non-outcome labels")
    provenance = model.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("native model provenance is missing")
    if provenance.get("incumbent_labels") is not False:
        raise ValueError("native model provenance permits incumbent labels")
    if provenance.get("protected_data") is not False:
        raise ValueError("native model provenance permits protected data")
    trainer_sha = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
    if provenance.get("trainer_sha256") != trainer_sha:
        raise ValueError("native model trainer SHA-256 is stale")
    corpus_validator_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if provenance.get("corpus_validator_sha256") != corpus_validator_sha:
        raise ValueError("native model corpus-validator SHA-256 is stale")
    if not _valid_sha256(provenance.get("corpus_sha256")):
        raise ValueError("native corpus SHA-256 is missing")
    sources = provenance.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or not all(_valid_sha256(value) for value in sources.values())
    ):
        raise ValueError("native corpus source hashes are incomplete")
    _validate_build_provenance(model, "native model", track)
    if track == "round2":
        _validate_round2_model_metadata(model, corpus_path.parents[1])
    return model


def _validate_historical_round2_model(model_path: pathlib.Path) -> dict:
    raw = model_path.read_bytes()
    model_sha256 = hashlib.sha256(raw).hexdigest()
    historical = HISTORICAL_ROUND2_BASELINES.get(model_sha256)
    if historical is None:
        raise ValueError("historical active model bytes are not allowlisted")
    model = json.loads(raw)
    provenance = model.get("provenance")
    if (
        not isinstance(provenance, dict)
        or model.get("schema") != MODEL_SCHEMA
        or model.get("feature_schema") != FEATURE_SCHEMA
        or provenance.get("trainer_sha256") != historical["trainer_sha256"]
        or provenance.get("corpus_validator_sha256")
        != historical["corpus_validator_sha256"]
        or provenance.get("restart_corpus_validator_sha256")
        != historical["restart_corpus_validator_sha256"]
        or provenance.get("incumbent_labels") is not False
        or provenance.get("protected_data") is not False
        or model.get("target") != {
            "primary": "mover-relative-final-outcome",
            "auxiliary": "stable-native-bfm-reanalysis",
            "auxiliary_weight": 0.25,
            "policy_target": None,
        }
    ):
        raise ValueError("historical active model identity is not frozen")
    _validate_build_provenance(model, "historical active model", "round2")
    return model


def _validate_round2_model_metadata(
    model: dict, repository_root: pathlib.Path
) -> None:
    target = model.get("target")
    phase_weights = target.get("phase_weights") if isinstance(target, dict) else None
    if (
        not isinstance(target, dict)
        or set(target) != {
            "primary", "auxiliary", "auxiliary_weight", "phase_weights",
            "phase_weight_application", "policy_target",
        }
        or target.get("primary") != "mover-relative-final-outcome"
        or target.get("auxiliary") != "stable-native-bfm-reanalysis"
        or isinstance(target.get("auxiliary_weight"), bool)
        or target.get("auxiliary_weight") != 0.25
        or target.get("policy_target") is not None
        or target.get("phase_weight_application")
        != "after-exact-override-combined-target-loss/v1"
        or not isinstance(phase_weights, dict)
        or set(phase_weights) != {
            "turns_0_11", "turns_12_23", "turns_24_plus"
        }
        or any(
            isinstance(phase_weights.get(name), bool)
            or not isinstance(phase_weights.get(name), (int, float))
            or phase_weights[name] != expected
            for name, expected in {
                "turns_0_11": 3.0,
                "turns_12_23": 1.5,
                "turns_24_plus": 1.0,
            }.items()
        )
    ):
        raise ValueError("round-two phase-weighted target contract is stale")
    provenance = model["provenance"]
    generation = provenance["generation"]
    restart_corpus = repository_root / ROUND2_SEMANTIC_DEPENDENCIES[1]
    if provenance.get("restart_corpus_validator_sha256") != hashlib.sha256(
            restart_corpus.read_bytes()).hexdigest():
        raise ValueError("round-two restart corpus-validator SHA-256 is stale")
    sources = provenance["source_sha256"]
    if not all(key == f"sha256:{value}" for key, value in sources.items()):
        raise ValueError("round-two corpus source identities are not content-addressed")
    expected_corpus = hashlib.sha256(json.dumps(
        sorted(sources.items()), separators=(",", ":")
    ).encode()).hexdigest()
    if provenance.get("corpus_sha256") != expected_corpus:
        raise ValueError("round-two cumulative corpus SHA-256 is stale")
    lineage = provenance.get("lineage")
    legacy_lineage_fields = {
        "strict_current", "archived_round1", "live_restart_round2",
    }
    extended_lineage_fields = {
        "strict_current", "archived_round1", "archived_round2",
        "live_restart_round2", "archived_restart_round2",
    }
    if (
        not isinstance(lineage, dict)
        or set(lineage) not in {
            frozenset(legacy_lineage_fields),
            frozenset(extended_lineage_fields),
        }
        or not isinstance(lineage["strict_current"], list)
        or not lineage["strict_current"]
    ):
        raise ValueError("round-two cumulative corpus lineage is incomplete")
    normal_categories = ["strict_current", "archived_round1"]
    if "archived_round2" in lineage:
        normal_categories.append("archived_round2")
    for category in normal_categories:
        entries = lineage[category]
        if not isinstance(entries, list):
            raise ValueError("round-two corpus lineage category is invalid")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "manifest_sha256", "build_provenance_sha256",
                "binary_sha256", "shard_sha256", "games", "seed",
            }:
                raise ValueError("round-two corpus lineage entry is not frozen")
            if not all(_valid_sha256(entry.get(field)) for field in (
                "manifest_sha256", "build_provenance_sha256", "binary_sha256"
            )):
                raise ValueError("round-two corpus lineage identity is invalid")
            shards = entry.get("shard_sha256")
            games = entry.get("games")
            seed = entry.get("seed")
            if (
                not isinstance(shards, list) or not shards
                or shards != sorted(shards)
                or not all(_valid_sha256(value) for value in shards)
                or isinstance(games, bool) or not isinstance(games, int)
                or games <= 0 or isinstance(seed, bool)
                or not isinstance(seed, int) or not 0 <= seed < 1 << 64
            ):
                raise ValueError("round-two corpus lineage values are invalid")
        if entries != sorted(entries, key=lambda entry: (
            entry["seed"], entry["manifest_sha256"]
        )):
            raise ValueError("round-two corpus lineage entries are not canonical")
    restart_categories = ["live_restart_round2"]
    if "archived_restart_round2" in lineage:
        restart_categories.append("archived_restart_round2")
    for category in restart_categories:
        restarts = lineage[category]
        if not isinstance(restarts, list):
            raise ValueError("round-two restart lineage is invalid")
        for entry in restarts:
            if not isinstance(entry, dict) or set(entry) != {
                "manifest_sha256", "build_provenance_sha256", "binary_sha256",
                "collector_tsv_sha256", "arena_manifest_sha256",
                "asserted_source_sha256", "exclusion_registry_sha256",
                "source_binding_status", "games", "selected_prefixes",
            }:
                raise ValueError("round-two restart lineage entry is not frozen")
            if not all(_valid_sha256(entry.get(field)) for field in (
                "manifest_sha256", "build_provenance_sha256", "binary_sha256",
                "collector_tsv_sha256", "arena_manifest_sha256",
                "asserted_source_sha256", "exclusion_registry_sha256",
            )) or entry.get("source_binding_status") not in {
                "asserted-not-api-verified", "api-verified"
            } or any(
                isinstance(entry.get(field), bool)
                or not isinstance(entry.get(field), int)
                or entry[field] <= 0
                for field in ("games", "selected_prefixes")
            ):
                raise ValueError("round-two restart lineage values are invalid")
        if restarts != sorted(restarts, key=lambda entry: (
            entry["collector_tsv_sha256"], entry["manifest_sha256"]
        )):
            raise ValueError("round-two restart lineage is not canonical")
    lineage_entries = [
        entry for category in normal_categories + restart_categories
        for entry in lineage[category]
    ]
    declared_builds = generation.get("build_provenance_sha256")
    lineage_builds = sorted({
        entry["build_provenance_sha256"] for entry in lineage_entries
    })
    if declared_builds != lineage_builds:
        raise ValueError("round-two lineage/build provenance disagrees")
    binary_by_build = {
        item["sha256"]: item["contract"]["binary"]["sha256"]
        for item in generation["build_contracts"]
    }
    if any(binary_by_build.get(entry["build_provenance_sha256"]) != entry[
            "binary_sha256"] for entry in lineage_entries):
        raise ValueError("round-two lineage/build binary identity disagrees")
    training = model.get("training")
    checkpoints = model.get("checkpoints")
    if not isinstance(training, dict) or not isinstance(checkpoints, list) or (
            not checkpoints):
        raise ValueError("round-two retained checkpoints are missing")
    seeds = training.get("seeds")
    if (
        not isinstance(seeds, list) or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int)
               or not 0 <= seed < 1 << 64 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("round-two training seed set is invalid")
    observed = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "seed", "model", "quantization", "checkpoint_sha256"
        }:
            raise ValueError("round-two checkpoint fields are not frozen")
        payload = {
            "seed": checkpoint["seed"],
            "model": checkpoint["model"],
            "quantization": checkpoint["quantization"],
        }
        canonical = json.dumps(
            payload, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        canonical = (canonical + "\n").encode()
        if hashlib.sha256(canonical).hexdigest() != checkpoint.get(
                "checkpoint_sha256"):
            raise ValueError("round-two checkpoint SHA-256 is stale")
        # This also freezes seed, architecture, target, quantization shapes,
        # integer ranges, and positive finite scales.
        _checkpoint_runtime_bytes(
            model, "0" * 64, checkpoint
        )
        observed.append(checkpoint["seed"])
    if observed != seeds:
        raise ValueError("round-two checkpoint order disagrees with training seeds")
    external = training.get("external_actual_clock_selection")
    if (
        training.get("chosen_seed") is not None
        or training.get("provisional_seed") not in observed
        or not isinstance(external, dict)
        or external.get("required") is not True
        or external.get("criterion") != "native-actual-clock-match-strength"
        or external.get("status") != "pending"
        or not isinstance(external.get("eligible_seed_order"), list)
        or sorted(external["eligible_seed_order"]) != sorted(observed)
    ):
        raise ValueError("round-two actual-clock selection contract is invalid")


def _validate_model_provenance(
    repository_root: pathlib.Path, track: str = "round1"
) -> dict:
    if track == "round1":
        dependencies = ROUND1_PURITY_DEPENDENCIES
        trainer = "tools/train_jacek_native.py"
        corpus = ROUND1_SEMANTIC_DEPENDENCIES[0]
    elif track == "round2":
        dependencies = ROUND2_PURITY_DEPENDENCIES
        trainer = "tools/train_jacek_native_round2.py"
        corpus = ROUND2_SEMANTIC_DEPENDENCIES[0]
    else:
        raise ValueError("unknown native purity track")
    return _validate_native_model_provenance(
        repository_root / dependencies[0],
        repository_root / trainer,
        repository_root / corpus,
        track,
    )


def _pack_signed_three_bit(values: list[int]) -> bytes:
    output = bytearray()
    accumulator = 0
    available = 0
    for value in values:
        accumulator |= (value & 0b111) << available
        available += 3
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _checkpoint_runtime_bytes(
    model: dict, model_sha256: str, checkpoint: dict
) -> tuple[bytes, dict[str, str]]:
    architecture = {
        "inputs": 1156,
        "hidden_one": 32,
        "hidden_two": 32,
        "outputs": 1,
        "biases": False,
        "hidden_one_activation": "square-nonnegative-leaky-0.01-negative",
        "hidden_two_activation": "leaky-relu-0.01",
        "output_activation": "tanh",
    }
    rules = {
        "width": 8,
        "height": 10,
        "goal_rule": "own-goals-allowed",
        "blocked_rule": "mover-loses",
    }
    target = model.get("target")
    try:
        auxiliary_weight = float(target.get("auxiliary_weight", -1.0))
    except (AttributeError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "native checkpoint model target is not frozen"
        ) from error
    if model.get("architecture") != architecture or model.get("rules") != rules:
        raise ValueError("native checkpoint model architecture is not frozen")
    if (
        not isinstance(target, dict)
        or target.get("primary") != "mover-relative-final-outcome"
        or target.get("auxiliary") != "stable-native-bfm-reanalysis"
        or auxiliary_weight != 0.25
        or target.get("policy_target") is not None
    ):
        raise ValueError("native checkpoint model target is not frozen")
    if not isinstance(checkpoint, dict):
        raise ValueError("native checkpoint entry is malformed")
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError("native checkpoint seed is invalid")
    quantization = checkpoint.get("quantization")
    if not isinstance(quantization, dict) or {
        "bits": quantization.get("bits"),
        "minimum": quantization.get("minimum"),
        "maximum": quantization.get("maximum"),
        "scheme": quantization.get("scheme"),
        "packing": quantization.get("packing"),
    } != {
        "bits": 3,
        "minimum": -3,
        "maximum": 3,
        "scheme": "symmetric-per-layer-round-to-nearest",
        "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
    }:
        raise ValueError("native checkpoint quantization is not frozen")
    weights = quantization.get("weights")
    scales = quantization.get("scales")
    if not isinstance(weights, dict) or not isinstance(scales, dict):
        raise ValueError("native checkpoint weights or scales are missing")
    flattened: list[int] = []
    rendered_scales: list[str] = []
    for name, shape in MODEL_SHAPES.items():
        tensor = weights.get(name)
        if not isinstance(tensor, dict) or tensor.get("shape") != list(shape):
            raise ValueError(f"native checkpoint {name} shape is invalid")
        values = tensor.get("values")
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise ValueError(f"native checkpoint {name} values are incomplete")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -3
            or value > 3
            for value in values
        ):
            raise ValueError(f"native checkpoint {name} values are invalid")
        flattened.extend(values)
        try:
            scale = float(scales.get(name, 0.0))
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"native checkpoint {name} scale is invalid"
            ) from error
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"native checkpoint {name} scale is invalid")
        rendered_scales.append(f"{scale:.9g}")
    payload = _pack_signed_three_bit(flattened)
    packed_sha256 = hashlib.sha256(payload).hexdigest()
    runtime = (
        f"{RUNTIME_SCHEMA}\n"
        f"{MODEL_SCHEMA}\n"
        f"{FEATURE_SCHEMA}\n"
        f"{model_sha256}\n"
        f"{packed_sha256}\n"
        f"{' '.join(rendered_scales)}\n"
        f"{base64.b64encode(payload).decode('ascii')}\n"
    ).encode()
    return runtime, {
        "artifact_sha256": hashlib.sha256(runtime).hexdigest(),
        "model_sha256": model_sha256,
        "packed_sha256": packed_sha256,
    }


def _validate_native_checkpoint_files(
    repository_root: pathlib.Path, config: dict
) -> dict[str, tuple[dict[str, str], dict, pathlib.Path]]:
    entries = config.get("native_checkpoint_provenance", [])
    if not entries:
        raise ValueError(
            "native checkpoint league requires file-backed model/runtime provenance"
        )
    declarations: dict[
        str, tuple[dict[str, str], dict, pathlib.Path]
    ] = {}
    for index, entry in enumerate(entries):
        model_path = _contained(
            repository_root, entry["model"], f"native checkpoint {index} model"
        )
        runtime_path = _contained(
            repository_root, entry["runtime"], f"native checkpoint {index} runtime"
        )
        raw_model = model_path.read_bytes()
        raw_candidate = json.loads(raw_model)
        model_sha256 = hashlib.sha256(raw_model).hexdigest()
        historical = HISTORICAL_ROUND2_BASELINES.get(model_sha256)
        trainer_sha = raw_candidate.get("provenance", {}).get("trainer_sha256")
        round1_trainer = repository_root / "tools/train_jacek_native.py"
        round2_trainer = repository_root / "tools/train_jacek_native_round2.py"
        if historical is not None:
            provenance = raw_candidate.get("provenance")
            if (
                not isinstance(provenance, dict)
                or raw_candidate.get("schema") != MODEL_SCHEMA
                or raw_candidate.get("feature_schema") != FEATURE_SCHEMA
                or provenance.get("trainer_sha256")
                != historical["trainer_sha256"]
                or provenance.get("corpus_validator_sha256")
                != historical["corpus_validator_sha256"]
                or provenance.get("restart_corpus_validator_sha256")
                != historical["restart_corpus_validator_sha256"]
                or provenance.get("incumbent_labels") is not False
                or provenance.get("protected_data") is not False
                or raw_candidate.get("target") != {
                    "primary": "mover-relative-final-outcome",
                    "auxiliary": "stable-native-bfm-reanalysis",
                    "auxiliary_weight": 0.25,
                    "policy_target": None,
                }
            ):
                raise ValueError(
                    "historical native checkpoint identity is not frozen"
                )
            _validate_build_provenance(
                raw_candidate, "historical native checkpoint", "round2"
            )
            model = raw_candidate
        elif trainer_sha == hashlib.sha256(round1_trainer.read_bytes()).hexdigest():
            model_track = "round1"
            trainer_path = round1_trainer
            corpus_path = repository_root / ROUND1_SEMANTIC_DEPENDENCIES[0]
        elif (
            round2_trainer.is_file()
            and trainer_sha == hashlib.sha256(round2_trainer.read_bytes()).hexdigest()
        ):
            model_track = "round2"
            trainer_path = round2_trainer
            corpus_path = repository_root / ROUND2_SEMANTIC_DEPENDENCIES[0]
        else:
            raise ValueError("native checkpoint trainer lineage is unrecognized")
        if historical is None:
            model = _validate_native_model_provenance(
                model_path, trainer_path, corpus_path, model_track
            )
        checkpoints = model.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("native checkpoint model retains no seed checkpoints")
        expected = {}
        for checkpoint in checkpoints:
            runtime, identity = _checkpoint_runtime_bytes(
                model, model_sha256, checkpoint
            )
            expected[runtime] = identity
        runtime = runtime_path.read_bytes()
        if historical is not None and hashlib.sha256(runtime).hexdigest() not in (
            historical["retained_runtime_sha256"]
        ):
            raise ValueError(
                "historical native checkpoint runtime is not an exact retained export"
            )
        if runtime not in expected:
            raise ValueError(
                f"native checkpoint runtime {runtime_path} is not generated by "
                f"its declared model"
            )
        identity = expected[runtime]
        artifact_sha256 = identity["artifact_sha256"]
        if artifact_sha256 in declarations:
            raise ValueError("native checkpoint runtime artifacts must be unique")
        declarations[artifact_sha256] = (identity, model, model_path)
    return declarations


def _validate_seed_provenance(repository_root: pathlib.Path) -> dict[str, str]:
    descriptor_path = repository_root / "models/jacek_native_untrained_seed.json"
    runtime_path = repository_root / "models/jacek_native_untrained_seed.runtime"
    generator_path = repository_root / "tools/generate_jacek_native_seed.py"
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    if (
        descriptor.get("schema")
        != "papersoccer.jacek-native-untrained-seed/v1"
        or descriptor.get("model_schema") != "jacek_native_model/v1"
        or descriptor.get("feature_schema")
        != "canonical-edges316-onehot-true-turn-distance105x8-v1"
        or descriptor.get("training") is not None
    ):
        raise ValueError("native untrained-seed schema is not frozen")
    if descriptor.get("incumbent_dependencies") is not False:
        raise ValueError("native untrained seed permits incumbent dependencies")
    if descriptor.get("protected_data") is not False:
        raise ValueError("native untrained seed permits protected data")
    generator_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    if descriptor.get("generator_sha256") != generator_sha:
        raise ValueError("native untrained-seed generator SHA-256 is stale")

    lines = runtime_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 7 or lines[:3] != [
        "papersoccer.jacek-native-runtime-model/v1",
        "jacek_native_model/v1",
        "canonical-edges316-onehot-true-turn-distance105x8-v1",
    ]:
        raise ValueError("native untrained runtime schema is not frozen")
    descriptor_sha = hashlib.sha256(descriptor_bytes).hexdigest()
    if lines[3] != descriptor_sha:
        raise ValueError("native untrained runtime descriptor SHA-256 is stale")
    try:
        payload = base64.b64decode(lines[6], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("native untrained runtime payload is invalid") from error
    packed_sha = hashlib.sha256(payload).hexdigest()
    weights = descriptor.get("weights")
    if (
        not isinstance(weights, dict)
        or weights.get("packed_sha256") != packed_sha
        or lines[4] != packed_sha
    ):
        raise ValueError("native untrained runtime packed SHA-256 is stale")
    counts = weights.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"w1", "w2", "w3"}
        or any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in counts.values())
        or len(payload) != (sum(counts.values()) * 3 + 7) // 8
    ):
        raise ValueError("native untrained runtime payload size is invalid")
    scales = weights.get("scales")
    try:
        runtime_scales = [float(value) for value in lines[5].split()]
        descriptor_scales = [
            float(scales[name]) for name in ("w1", "w2", "w3")
        ]
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise ValueError("native untrained runtime scales are invalid") from error
    if runtime_scales != descriptor_scales:
        raise ValueError("native untrained runtime scales are stale")

    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    return {
        "artifact_sha256": runtime_sha,
        "model_sha256": descriptor_sha,
        "packed_sha256": packed_sha,
    }


def _checkpoint_provenance(
    model: dict, label: str
) -> tuple[str, list[dict[str, str]]]:
    generation = model.get("provenance", {}).get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label} generation provenance is missing")
    checkpoint = generation.get("checkpoint_provenance")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"mode", "artifacts"}
    ):
        raise ValueError(f"{label} checkpoint provenance mode is missing")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{label} checkpoint provenance artifacts are missing")
    normalized: list[dict[str, str]] = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact)
            != {"artifact_sha256", "model_sha256", "packed_sha256"}
            or not all(_valid_sha256(artifact.get(field)) for field in artifact)
        ):
            raise ValueError(f"{label} checkpoint provenance artifact is invalid")
        normalized.append(dict(artifact))
    normalized.sort(key=lambda value: (
        value["artifact_sha256"],
        value["model_sha256"],
        value["packed_sha256"],
    ))
    if artifacts != normalized:
        raise ValueError(
            f"{label} checkpoint provenance artifacts are not canonical"
        )
    if len({value["artifact_sha256"] for value in normalized}) != len(normalized):
        raise ValueError(
            f"{label} checkpoint provenance artifacts are duplicated"
        )
    legacy = generation.get("model_artifact_sha256")
    if legacy != [value["artifact_sha256"] for value in normalized]:
        raise ValueError(
            f"{label} checkpoint provenance disagrees with artifact hashes"
        )
    return checkpoint.get("mode"), normalized


def _validate_checkpoint_provenance(
    repository_root: pathlib.Path,
    config: dict,
    model: dict,
    seed_identity: dict[str, str],
    track: str = "round1",
) -> None:
    mode, normalized = _checkpoint_provenance(model, "native model")

    configured = config.get("native_checkpoint_provenance", [])
    if mode == "untrained-seed-bootstrap/v1":
        if normalized != [seed_identity]:
            raise ValueError(
                "bootstrap corpus is not bound to the exact untrained seed runtime"
            )
        if configured:
            raise ValueError(
                "bootstrap purity must not declare later native checkpoints"
            )
        return
    cumulative = mode == "cumulative-native-runtime-models/v2"
    if cumulative and track != "round2":
        raise ValueError("cumulative native checkpoint mode requires round two")
    if mode != "native-runtime-models/v1" and not cumulative:
        raise ValueError("unsupported native checkpoint provenance mode")
    if seed_identity in normalized and not cumulative:
        raise ValueError(
            "native checkpoint mode may not mix the untrained seed runtime"
        )
    if cumulative and (seed_identity not in normalized or len(normalized) < 2):
        raise ValueError(
            "cumulative native checkpoint mode requires its seed root and a "
            "native checkpoint"
        )
    declarations = _validate_native_checkpoint_files(repository_root, config)
    reachable: set[str] = set()
    visiting: set[str] = set()

    def validate_lineage(identity: dict[str, str]) -> None:
        if identity == seed_identity:
            return
        artifact_sha256 = identity["artifact_sha256"]
        declaration = declarations.get(artifact_sha256)
        if declaration is None or declaration[0] != identity:
            raise ValueError(
                "native checkpoint league artifacts do not match file-backed "
                "provenance"
            )
        if artifact_sha256 in visiting:
            raise ValueError("native checkpoint provenance contains a cycle")
        if artifact_sha256 in reachable:
            return
        visiting.add(artifact_sha256)
        _, parent_model, parent_path = declaration
        parent_mode, parents = _checkpoint_provenance(
            parent_model,
            f"native checkpoint model {parent_path.relative_to(repository_root)}",
        )
        if parent_mode == "untrained-seed-bootstrap/v1":
            if parents != [seed_identity]:
                raise ValueError(
                    "native checkpoint bootstrap ancestry is not bound to the "
                    "exact untrained seed runtime"
                )
        elif parent_mode in {
            "native-runtime-models/v1", "cumulative-native-runtime-models/v2"
        }:
            parent_cumulative = (
                parent_mode == "cumulative-native-runtime-models/v2"
            )
            if seed_identity in parents and not parent_cumulative:
                raise ValueError(
                    "native checkpoint ancestry may not mix the untrained seed "
                    "runtime"
                )
            if parent_cumulative and seed_identity not in parents:
                raise ValueError(
                    "cumulative native checkpoint ancestry is missing its seed root"
                )
            for parent in parents:
                if parent == seed_identity:
                    continue
                validate_lineage(parent)
        else:
            raise ValueError("unsupported native checkpoint ancestry mode")
        visiting.remove(artifact_sha256)
        reachable.add(artifact_sha256)

    for identity in normalized:
        if identity == seed_identity:
            continue
        validate_lineage(identity)
    if reachable != set(declarations):
        raise ValueError(
            "native checkpoint provenance contains unused file declarations"
        )


def _validate_corpus_semantics(repository_root: pathlib.Path) -> None:
    path = repository_root / EXPECTED_SEMANTIC_DEPENDENCIES[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = None
    validate_record = None
    literal_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name)
                and target.id == "FORBIDDEN_PROVENANCE"
                for target in targets
            ):
                forbidden = set(ast.literal_eval(node.value))
        if isinstance(node, ast.FunctionDef) and node.name == "validate_record":
            validate_record = node
        if isinstance(node, ast.Set):
            try:
                values = set(ast.literal_eval(node))
            except (ValueError, TypeError):
                continue
            if all(isinstance(value, str) for value in values):
                literal_sets.append(values)
    required_provenance = {
        "rank_4",
        "rank-4",
        "rank4",
        "replay-book",
        "replay_book",
        "alpha-beta-teacher",
        "alpha_beta_teacher",
    }
    if forbidden is None or not required_provenance.issubset(forbidden):
        raise ValueError("native corpus provenance guard is incomplete")
    if validate_record is None or not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_check_purity"
        for node in ast.walk(validate_record)
    ):
        raise ValueError("native corpus validation does not call its purity guard")
    required_labels = {
        "policy",
        "policy_target",
        "teacher_move",
        "expert_move",
        "rank4_value",
    }
    if not any(required_labels.issubset(values) for values in literal_sets):
        raise ValueError("native corpus action-label guard is incomplete")


def _validate_round2_corpus_semantics(repository_root: pathlib.Path) -> None:
    # Round two deliberately layers stricter lineage/reanalysis checks over the
    # frozen round-one replay and action-label guard.
    _validate_corpus_semantics(repository_root)
    path = repository_root / ROUND2_SEMANTIC_DEPENDENCIES[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports_round1 = any(
        isinstance(node, ast.Import)
        and any(alias.name == "jacek_native_corpus" and alias.asname == "round1"
                for alias in node.names)
        for node in ast.walk(tree)
    )
    validate_record = next((
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "validate_record"
    ), None)
    calls = set()
    if validate_record is not None:
        for node in ast.walk(validate_record):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "round1"
            ):
                calls.add(node.func.attr)
    if not imports_round1 or not {"_check_purity", "validate_record"}.issubset(calls):
        raise ValueError(
            "round-two corpus does not inherit the frozen purity/replay guards"
        )
    restart_path = repository_root / ROUND2_SEMANTIC_DEPENDENCIES[1]
    restart_tree = ast.parse(
        restart_path.read_text(encoding="utf-8"), filename=str(restart_path)
    )
    restart_validate = next((
        node for node in ast.walk(restart_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "validate_record"
    ), None)
    restart_calls: set[tuple[str, str]] = set()
    if restart_validate is not None:
        for node in ast.walk(restart_validate):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            ):
                restart_calls.add((node.func.value.id, node.func.attr))
    observed_usage = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "OBSERVED_USAGE"
                for target in (node.targets if isinstance(node, ast.Assign)
                               else [node.target]))
        and isinstance(node.value, ast.Constant)
        and node.value.value == "state-construction-only"
        for node in ast.walk(restart_tree)
    )
    if (
        ("round1", "_check_purity") not in restart_calls
        or ("round2", "validate_record") not in restart_calls
        or not observed_usage
    ):
        raise ValueError(
            "round-two live-restart corpus does not preserve no-label guards"
        )


def _load_round2_activation(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "papersoccer_jacek_native_round2_activation_purity", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load round-two activation contract")
    activation = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, activation)
    spec.loader.exec_module(activation)
    return activation


def _validate_round1_header(
    repository_root: pathlib.Path,
    bot_directory: pathlib.Path,
    production_files: list[pathlib.Path],
) -> None:
    header_path = (bot_directory / "jacek_native_model.hpp").resolve()
    if header_path not in production_files:
        return
    exporter_path = (
        repository_root
        / "submissions/codingame/tools/generate_jacek_native_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "papersoccer_jacek_native_round1_exporter_purity", exporter_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen round-one model exporter")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    model_path = repository_root / ROUND1_PURITY_DEPENDENCIES[0]
    model_raw = model_path.read_bytes()
    model = json.loads(model_raw)
    expected, _ = exporter.render(
        model, hashlib.sha256(model_raw).hexdigest()
    )
    if header_path.read_text(encoding="utf-8") != expected:
        raise ValueError("active round-one model header is stale")


def purity_violations(
    bot_directory: pathlib.Path = BOT_DIRECTORY,
    repository_root: pathlib.Path = REPOSITORY_ROOT,
) -> list[str]:
    repository_root = repository_root.resolve()
    config, files = production_files(bot_directory.resolve(), repository_root)
    deployment_path = repository_root / ROUND2_DEPLOYMENT_PATH
    activated = None
    if deployment_path.is_file():
        activation_path = (
            repository_root
            / "submissions/codingame/tools/jacek_native_round2_activation.py"
        )
        activation = _load_round2_activation(activation_path)
        activated = activation.load_deployment(
            deployment_path, repository_root
        )
        track = "round2"
        if activated.get("historical_active") is True:
            model = _validate_historical_round2_model(activated["model_path"])
        else:
            model = _validate_native_model_provenance(
                activated["model_path"],
                repository_root / "tools/train_jacek_native_round2.py",
                repository_root / ROUND2_SEMANTIC_DEPENDENCIES[0],
                track,
            )
        expected_header, _ = activation.render_deployment(activated)
        header_path = bot_directory / "jacek_native_model.hpp"
        if header_path.read_text(encoding="utf-8") != expected_header:
            raise ValueError("active round-two model header is stale")
        # The descriptor is the sole mutable activation pointer.  The selected
        # model/selection/runtime remain immutable generation-specific files
        # under models/ and were already containment-checked by activation.
        for relative in ROUND2_PURITY_DEPENDENCIES[1:]:
            path = repository_root / relative
            if not path.is_file():
                raise ValueError(f"missing round-two purity dependency: {relative}")
            if path not in files:
                files.append(path)
        for path in (
            deployment_path,
            activated["model_path"],
            activated["selection_path"],
            activated["runtime_path"],
            activated["baseline_model_path"],
            activated["baseline_runtime_path"],
            activated.get("baseline_selection_path"),
            activated.get("baseline_deployment_path"),
            activated.get("historical_deployment_path"),
            *activated["evidence_paths"],
        ):
            if path is None:
                continue
            if path not in files:
                files.append(path)
        config = dict(config)
        config["native_checkpoint_provenance"] = []
        for checkpoint in activated["checkpoint_paths"]:
            model_path = checkpoint["model"]
            runtime_path = checkpoint["runtime"]
            if model_path == activated["model_path"]:
                raise ValueError(
                    "native checkpoint provenance must not self-reference the "
                    "active round-two model"
                )
            config["native_checkpoint_provenance"].append({
                "model": model_path.relative_to(repository_root).as_posix(),
                "runtime": runtime_path.relative_to(repository_root).as_posix(),
            })
            for path in (model_path, runtime_path):
                if path not in files:
                    files.append(path)
    else:
        track = _purity_track(config)
        model = _validate_model_provenance(repository_root, track)
        if track == "round1":
            _validate_round1_header(repository_root, bot_directory, files)
    seed_identity = _validate_seed_provenance(repository_root)
    _validate_checkpoint_provenance(
        repository_root, config, model, seed_identity, track
    )
    if track == "round1":
        _validate_corpus_semantics(repository_root)
        banned_patterns = BANNED_PATTERNS
    else:
        _validate_round2_corpus_semantics(repository_root)
        banned_patterns = ROUND2_BANNED_PATTERNS
    violations: list[str] = []
    for path in files:
        relative = path.resolve().relative_to(repository_root.resolve())
        path_text = relative.as_posix()
        contents = path.read_text(encoding="utf-8")
        for pattern, label in banned_patterns:
            if pattern.search(path_text) or pattern.search(contents):
                violations.append(f"{relative}: {label}")
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify jacek_native_bfm production dependency purity."
    )
    parser.add_argument("--bot-directory", type=pathlib.Path, default=BOT_DIRECTORY)
    parser.add_argument("--repository-root", type=pathlib.Path, default=REPOSITORY_ROOT)
    options = parser.parse_args(argv)
    try:
        violations = purity_violations(
            options.bot_directory.resolve(), options.repository_root.resolve()
        )
    except (
        OSError,
        ValueError,
        SyntaxError,
        json.JSONDecodeError,
        UnicodeError,
    ) as error:
        print(f"jacek_native_bfm purity check failed: {error}", file=sys.stderr)
        return 1
    if violations:
        print("jacek_native_bfm production dependency impurity:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print(
        "jacek_native_bfm production sources are independent of incumbent, "
        "replay, alpha-beta, and teacher dependencies."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
