#!/usr/bin/env python3
"""Strict round-two corpus validation with cumulative round-one lineage.

Round-one validation stays frozen because the active model records its exact
validator hash.  This module accepts archived, manifest-complete round-one runs
and a strict-current round-two run, then returns one whole-game dataset without
introducing policy or incumbent labels.

For round-two auxiliary eligibility, "non-truncated" means the configured,
fixed-work capped BFM completed without a deadline or other operational
interruption.  The architecture's 250-action retention, partial-path bound,
bounded tactical-proof sampling, and planned tree/work exhaustion remain
explicitly recorded and do not masquerade as operational failures.
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Mapping, Sequence


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus as round1  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
GAME_SCHEMA = "papersoccer.jacek-native-game/v2"
GENERATOR_SCHEMA = "jacek-native-complete-turn-bfm/v2"
RUN_SCHEMA = "papersoccer.jacek-native-selfplay-run/v2"
BUILD_PROVENANCE_SCHEMA = "papersoccer.jacek-native-build-provenance/v2"
BUILD_PROVENANCE_NAME = round1.BUILD_PROVENANCE_NAME
ARCHIVED_BINARY_NAME = "selfplay-round2-binary"
MANIFEST_NAME = "manifest.json"
COLOR_SCHEDULE = "paired-opening-depth-then-swap-checkpoints/v2"
REANALYSIS_SELECTION = "alternating-hard-error-and-low-confidence/v1"
TEACHER_WORK = 30_000
VERIFICATION_WORK = 100_000

FEATURE_SCHEMA = round1.FEATURE_SCHEMA
EDGE_COUNT = round1.EDGE_COUNT
VERTEX_COUNT = round1.VERTEX_COUNT
DISTANCE_BUCKETS = round1.DISTANCE_BUCKETS
INPUT_COUNT = round1.INPUT_COUNT
RULES = round1.RULES
DEQUE_SCHEDULE = round1.DEQUE_SCHEDULE
FORBIDDEN_PROVENANCE = round1.FORBIDDEN_PROVENANCE
NativeSample = round1.NativeSample
NativeModelArtifact = round1.NativeModelArtifact
NativeGame = round1.NativeGame
purge_cross_split_overlaps = round1.purge_cross_split_overlaps
canonical_state_id = round1.canonical_state_id
game_sort_key = round1._game_sort_key


def assign_splits(games: Sequence[NativeGame]) -> dict[str, str]:
    """Assign atomic groups, including mixed-outcome live restart groups."""
    grouped: dict[str, list[NativeGame]] = defaultdict(list)
    for game in games:
        grouped[game.split_group].append(game)
    by_outcomes: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for group, members in grouped.items():
        outcomes = tuple(sorted({member.winner for member in members}))
        if not outcomes or any(winner not in (0, 1) for winner in outcomes):
            raise ValueError(f"split_group has invalid outcomes: {group}")
        by_outcomes[outcomes].append(group)
    assignment: dict[str, str] = {}
    for _, groups in sorted(by_outcomes.items()):
        ordered = sorted(
            groups,
            key=lambda value: (hashlib.sha256(value.encode()).digest(), value),
        )
        count = len(ordered)
        train_end = (8 * count) // 10
        validation_end = (9 * count) // 10
        if count >= 3:
            train_end = max(1, min(train_end, count - 2))
            validation_end = max(train_end + 1, min(validation_end, count - 1))
        for index, group in enumerate(ordered):
            assignment[group] = (
                "train" if index < train_end
                else "validation" if index < validation_end
                else "test"
            )
    return assignment


def prepare_splits(games: Sequence[NativeGame]):
    assignment = assign_splits(games)
    samples: dict[str, list[NativeSample]] = {
        "train": [], "validation": [], "test": [],
    }
    for game in games:
        samples[assignment[game.split_group]].extend(game.samples)
    return (*purge_cross_split_overlaps(samples), assignment)

BUILD_SOURCE_PATHS = (
    "tools/jacek_native_selfplay_round2.cpp",
    *round1.BUILD_SOURCE_PATHS,
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
    "tools/jacek_native_selfplay_round2.cpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "-o",
    "$OUTPUT",
)


def _valid_sha256(value: object) -> bool:
    return round1._valid_sha256(value)


def _canonical_json_bytes(value: object) -> bytes:
    return round1._canonical_json_bytes(value)


def _model_identity(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
            "model_sha256", "packed_sha256", "artifact_sha256"}:
        raise ValueError(f"{label} model identity is invalid")
    for field, digest in value.items():
        if not _valid_sha256(digest):
            raise ValueError(f"{label}.{field} is not a SHA-256")
    return dict(value)


def _validate_round2_build_contract(
    raw: bytes,
    directory: pathlib.Path,
    verify_local_build: bool,
) -> tuple[str, dict]:
    contract = json.loads(raw)
    if raw != _canonical_json_bytes(contract):
        raise ValueError(
            f"{directory / BUILD_PROVENANCE_NAME} is not canonical JSON"
        )
    if not isinstance(contract, dict) or set(contract) != {
            "schema", "binary", "compiler", "build_argv",
            "producer_sha256", "sources"}:
        raise ValueError("round-two build provenance fields are not frozen")
    if contract.get("schema") != BUILD_PROVENANCE_SCHEMA:
        raise ValueError("round-two build provenance schema is not frozen")
    if contract.get("build_argv") != list(CANONICAL_BUILD_ARGV):
        raise ValueError("round-two build provenance argv is not frozen")

    binary = contract.get("binary")
    if (
        not isinstance(binary, dict)
        or set(binary) != {"path", "sha256"}
        or binary.get("path") != ARCHIVED_BINARY_NAME
        or not _valid_sha256(binary.get("sha256"))
    ):
        raise ValueError("round-two build binary identity is invalid")
    compiler = contract.get("compiler")
    if (
        not isinstance(compiler, dict)
        or set(compiler) != {
            "executable", "sha256", "version", "version_sha256"
        }
        or not isinstance(compiler.get("executable"), str)
        or not compiler["executable"]
        or pathlib.PurePath(compiler["executable"]).name
        != compiler["executable"]
        or not _valid_sha256(compiler.get("sha256"))
        or not isinstance(compiler.get("version"), str)
        or not compiler["version"]
        or len(compiler["version"]) > 16_384
        or not _valid_sha256(compiler.get("version_sha256"))
        or hashlib.sha256(compiler["version"].encode()).hexdigest()
        != compiler["version_sha256"]
    ):
        raise ValueError("round-two build compiler identity is invalid")

    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) != len(BUILD_SOURCE_PATHS):
        raise ValueError("round-two build source list is incomplete")
    source_pairs: list[list[str]] = []
    for expected_path, entry in zip(BUILD_SOURCE_PATHS, sources):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry.get("path") != expected_path
            or not _valid_sha256(entry.get("sha256"))
        ):
            raise ValueError("round-two build source identity is invalid")
        source_pairs.append([entry["path"], entry["sha256"]])
    producer = hashlib.sha256(json.dumps(
        source_pairs, separators=(",", ":")
    ).encode()).hexdigest()
    if contract.get("producer_sha256") != producer:
        raise ValueError("round-two build producer SHA-256 is inconsistent")

    rendered = raw.decode("utf-8").lower()
    forbidden = set(FORBIDDEN_PROVENANCE) | {
        "matches.json", "protected-bank", "protected_bank",
        "sealed-bank", "sealed_bank", "/users/", "/home/", "\\users\\",
    }
    home = str(pathlib.Path.home()).lower()
    if any(token in rendered for token in forbidden) or (
            home and home in rendered):
        raise ValueError("round-two build provenance contains a forbidden path")

    archived_binary = directory / binary["path"]
    if not archived_binary.is_file() or hashlib.sha256(
            archived_binary.read_bytes()).hexdigest() != binary["sha256"]:
        raise ValueError("round-two archived binary is stale")
    if verify_local_build:
        for entry in sources:
            path = ROOT / entry["path"]
            if not path.is_file() or hashlib.sha256(
                    path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(
                    f"round-two build source is stale: {entry['path']}"
                )
        resolved_compiler = shutil.which(compiler["executable"])
        if resolved_compiler is None:
            raise ValueError("round-two build compiler is unavailable")
        resolved_path = pathlib.Path(resolved_compiler).resolve()
        version = subprocess.run(
            [str(resolved_path), "--version"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if (
            hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            != compiler["sha256"]
            or version != compiler["version"]
        ):
            raise ValueError("round-two build compiler identity is stale")
    return hashlib.sha256(raw).hexdigest(), contract


def _validate_archived_round1_contract(
    raw: bytes, directory: pathlib.Path
) -> tuple[str, dict]:
    digest, contract = round1._validate_build_contract(raw, directory, False)
    binary = contract["binary"]
    archived_binary = directory / binary["path"]
    if not archived_binary.is_file() or hashlib.sha256(
            archived_binary.read_bytes()).hexdigest() != binary["sha256"]:
        raise ValueError("archived round-one binary is stale")
    return digest, contract


def _validate_manifest(
    directory: pathlib.Path,
    paths: Sequence[pathlib.Path],
    expected_schema: str,
    build_digest: str,
    build_contract: Mapping[str, object],
    require_canonical: bool,
) -> tuple[str, dict]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing sibling run manifest: {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if require_canonical and raw != _canonical_json_bytes(manifest):
        raise ValueError("strict-current round-two manifest is not canonical JSON")
    if not isinstance(manifest, dict) or manifest.get("schema") != expected_schema:
        raise ValueError("unexpected self-play run manifest schema")
    if expected_schema == RUN_SCHEMA and set(manifest) != {
        "schema", "run_id", "producer_sha256", "build_provenance",
        "binary", "checkpoints", "config", "shard_outputs",
    }:
        raise ValueError("strict-current round-two manifest fields are not frozen")
    provenance = manifest.get("build_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("path") != BUILD_PROVENANCE_NAME
        or provenance.get("sha256") != build_digest
    ):
        raise ValueError("manifest/build-provenance identity mismatch")
    binary = manifest.get("binary")
    contract_binary = build_contract["binary"]
    if (
        not isinstance(binary, dict)
        or binary.get("path") != contract_binary["path"]
        or binary.get("sha256") != contract_binary["sha256"]
    ):
        raise ValueError("manifest/binary identity mismatch")
    if manifest.get("producer_sha256") != build_contract["producer_sha256"]:
        raise ValueError("manifest/producer identity mismatch")

    checkpoints = manifest.get("checkpoints")
    if expected_schema == RUN_SCHEMA:
        if not isinstance(checkpoints, dict):
            raise ValueError("round-two manifest checkpoints are absent")
        for role, metadata in checkpoints.items():
            if role not in {"player_one", "player_two", "teacher"}:
                raise ValueError("round-two manifest checkpoint role is invalid")
            if not isinstance(metadata, dict) or set(metadata) != {
                    "name", "runtime", "artifact_sha256",
                    "model_sha256", "packed_sha256"}:
                raise ValueError("round-two checkpoint identity is not frozen")
            _model_identity({
                field: metadata[field]
                for field in ("artifact_sha256", "model_sha256", "packed_sha256")
            }, f"manifest.checkpoints.{role}")
            if (
                not isinstance(metadata["name"], str)
                or not metadata["name"]
                or len(metadata["name"]) > 64
                or not all(character.islower() or character.isdigit()
                           or character in "_-"
                           for character in metadata["name"])
                or metadata["runtime"] != f"checkpoints/{role}.runtime"
            ):
                raise ValueError("round-two checkpoint archive path is invalid")
            runtime_path = directory / metadata["runtime"]
            if not runtime_path.is_file() or hashlib.sha256(
                    runtime_path.read_bytes()).hexdigest() != metadata[
                        "artifact_sha256"]:
                raise ValueError(f"round-two archived {role} checkpoint is stale")
            lines = runtime_path.read_text(encoding="utf-8").splitlines()
            if (
                len(lines) != 7
                or lines[0] != "papersoccer.jacek-native-runtime-model/v1"
                or lines[1] != "jacek_native_model/v1"
                or lines[2] != FEATURE_SCHEMA
            ):
                raise ValueError(
                    f"round-two archived {role} runtime is malformed"
                )
            try:
                packed = base64.b64decode(lines[6], validate=True)
            except ValueError as error:
                raise ValueError(
                    f"round-two archived {role} payload is malformed"
                ) from error
            if (
                lines[3] != metadata["model_sha256"]
                or lines[4] != metadata["packed_sha256"]
                or hashlib.sha256(packed).hexdigest() != lines[4]
            ):
                raise ValueError(
                    f"round-two archived {role} runtime identity is stale"
                )
            try:
                scales = [float(value) for value in lines[5].split()]
            except ValueError as error:
                raise ValueError(
                    f"round-two archived {role} scales are malformed"
                ) from error
            expected_bytes = (38_048 * 3 + 7) // 8
            if (
                len(scales) != 3
                or any(not math.isfinite(value) or value <= 0.0
                       for value in scales)
                or len(packed) != expected_bytes
            ):
                raise ValueError(
                    f"round-two archived {role} tensor contract is malformed"
                )

    outputs = manifest.get("shard_outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("manifest has no shard outputs")
    expected_paths = {path.resolve() for path in paths}
    observed_paths: set[pathlib.Path] = set()
    observed_shards: set[int] = set()
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("manifest shard output is not an object")
        if expected_schema == RUN_SCHEMA and set(output) != {
            "shard", "path", "sha256", "bytes", "stderr_sha256",
        }:
            raise ValueError(
                "strict-current round-two shard metadata is not deterministic"
            )
        shard = output.get("shard")
        if isinstance(shard, bool) or not isinstance(shard, int) or shard < 0:
            raise ValueError("manifest shard index is invalid")
        if shard in observed_shards:
            raise ValueError("manifest repeats a shard index")
        observed_shards.add(shard)
        name = pathlib.PurePath(str(output.get("path", ""))).name
        if not name or name != f"shard-{shard:02d}-of-{len(outputs):02d}.jsonl":
            raise ValueError("manifest shard filename is not canonical")
        path = (directory / name).resolve()
        if path in observed_paths:
            raise ValueError("manifest repeats a shard path")
        observed_paths.add(path)
        if (
            not path.is_file()
            or not _valid_sha256(output.get("sha256"))
            or hashlib.sha256(path.read_bytes()).hexdigest() != output["sha256"]
            or output.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"manifest shard identity is stale: {name}")
    if observed_shards != set(range(len(outputs))):
        raise ValueError("manifest shard index set is incomplete")
    if observed_paths != expected_paths:
        raise ValueError("corpus inputs must include every manifest shard exactly once")
    return hashlib.sha256(raw).hexdigest(), manifest


def _validate_round2_generator(record: dict, line_number: int) -> None:
    generator = record.get("generator")
    if not isinstance(generator, dict):
        raise ValueError(f"generator must be an object on line {line_number}")
    if generator.get("schema") != GENERATOR_SCHEMA:
        raise ValueError(f"unexpected round-two generator schema on line {line_number}")
    if generator.get("checkpoint_color_schedule") != COLOR_SCHEDULE:
        raise ValueError(f"unexpected round-two color schedule on line {line_number}")
    game = record.get("game")
    if isinstance(game, bool) or not isinstance(game, int) or game < 0:
        raise ValueError(f"invalid game index on line {line_number}")
    if generator.get("opening_pair_index") != game // 2:
        raise ValueError(f"opening pair index mismatch on line {line_number}")

    reanalysis = generator.get("reanalysis")
    if not isinstance(reanalysis, dict) or set(reanalysis) != {
            "selection", "samples_per_game", "work",
            "verification_work", "teacher"}:
        raise ValueError(f"round-two reanalysis contract is invalid on line {line_number}")
    if reanalysis.get("selection") != REANALYSIS_SELECTION:
        raise ValueError(f"reanalysis selection is not frozen on line {line_number}")
    samples_per_game = round1._require_integer(
        reanalysis.get("samples_per_game"), "reanalysis.samples_per_game", 0
    )
    work = round1._require_integer(reanalysis.get("work"), "reanalysis.work", 0)
    verification_work = round1._require_integer(
        reanalysis.get("verification_work"), "reanalysis.verification_work", 0
    )
    if work == 0:
        if samples_per_game != 0 or verification_work != 0 or (
                reanalysis.get("teacher") is not None):
            raise ValueError(f"disabled reanalysis is inconsistent on line {line_number}")
    else:
        if (
            work != TEACHER_WORK
            or verification_work != VERIFICATION_WORK
            or samples_per_game <= 0
        ):
            raise ValueError(f"round-two reanalysis budgets are not 30k/100k")
        _model_identity(reanalysis.get("teacher"), "reanalysis.teacher")

    selected = 0
    for sample_index, sample in enumerate(record.get("samples", ())):
        if not isinstance(sample, dict):
            continue
        value = sample.get("reanalysis")
        if value is None:
            continue
        selected += 1
        label = f"line {line_number} sample {sample_index}.reanalysis"
        if not isinstance(value, dict) or set(value) != {
            "selection_reason", "value", "work", "verification_work",
            "operational_interruption", "primary_planned_work_exhaustion",
            "verification_planned_work_exhaustion",
            "primary_generator_sampling_truncations",
            "verification_generator_sampling_truncations",
            "primary_proof_sampling_truncations",
            "verification_proof_sampling_truncations",
            "action_stable", "value_delta", "stable", "exact",
        }:
            raise ValueError(f"{label} fields are not frozen")
        if value.get("selection_reason") not in {"hard", "uncertain"}:
            raise ValueError(f"{label} selection reason is invalid")
        if value.get("work") != work or value.get("verification_work") != verification_work:
            raise ValueError(f"{label} budgets do not match the generator")
        flags = (
            value.get("operational_interruption"),
            value.get("primary_planned_work_exhaustion"),
            value.get("verification_planned_work_exhaustion"),
            value.get("action_stable"), value.get("stable"), value.get("exact"),
        )
        if not all(isinstance(flag, bool) for flag in flags):
            raise ValueError(f"{label} flags must be booleans")
        for field in (
            "primary_generator_sampling_truncations",
            "verification_generator_sampling_truncations",
            "primary_proof_sampling_truncations",
            "verification_proof_sampling_truncations",
        ):
            round1._require_integer(value.get(field), f"{label}.{field}", 0)
        numeric_value = value.get("value")
        delta = value.get("value_delta")
        if (
            not isinstance(numeric_value, (int, float))
            or isinstance(numeric_value, bool)
            or not -1.0 <= float(numeric_value) <= 1.0
            or not isinstance(delta, (int, float))
            or isinstance(delta, bool)
            or not 0.0 <= float(delta) <= 2.0
        ):
            raise ValueError(f"{label} numeric values are invalid")
        expected_stable = value["exact"] or (
            not value["operational_interruption"]
            and value["action_stable"]
            and float(delta) <= 0.05
        )
        if value["stable"] != expected_stable:
            raise ValueError(f"{label} stability classification is inconsistent")
        if value["exact"] and abs(float(numeric_value)) != 1.0:
            raise ValueError(f"{label} exact result is not an outcome override")
    if selected != samples_per_game:
        raise ValueError(
            f"round-two selected reanalysis count mismatch on line {line_number}"
        )


def validate_record(record: object, line_number: int = 1) -> NativeGame:
    if not isinstance(record, dict) or record.get("schema") != GAME_SCHEMA:
        raise ValueError(f"unexpected round-two game schema on line {line_number}")
    round1._check_purity(record, line_number)
    _validate_round2_generator(record, line_number)

    compatible = copy.deepcopy(record)
    compatible["schema"] = round1.GAME_SCHEMA
    generator = compatible["generator"]
    generator["schema"] = round1.GENERATOR_SCHEMA
    generator["checkpoint_color_schedule"] = (
        "swap-player-checkpoints-on-odd-games"
    )
    generator.pop("opening_pair_index")
    generator.pop("reanalysis")
    for sample in compatible["samples"]:
        reanalysis = sample.get("reanalysis")
        if reanalysis is None:
            continue
        sample["reanalysis"] = {
            "value": reanalysis["value"],
            "work": reanalysis["work"],
            "verification_work": reanalysis["verification_work"],
            "truncated": reanalysis["operational_interruption"],
            "action_stable": reanalysis["action_stable"],
            "value_delta": reanalysis["value_delta"],
            "stable": reanalysis["stable"],
            "exact": reanalysis["exact"],
        }
    game = round1.validate_record(compatible, line_number)
    teacher = record["generator"]["reanalysis"]["teacher"]
    if teacher is None:
        return game
    teacher_artifact = NativeModelArtifact(
        artifact_sha256=teacher["artifact_sha256"],
        model_sha256=teacher["model_sha256"],
        packed_sha256=teacher["packed_sha256"],
    )
    artifacts = tuple(sorted(
        {*game.model_artifacts, teacher_artifact},
        key=lambda artifact: (
            artifact.artifact_sha256,
            artifact.model_sha256,
            artifact.packed_sha256,
        ),
    ))
    return dataclasses.replace(game, model_artifacts=artifacts)


def _pair_signature(record: Mapping[str, object]) -> tuple:
    generator = record["generator"]
    return (
        generator["opening_depth"], generator["opening_seed"],
        generator["opening_retry"], generator["opening_transcript"],
        generator["reanalysis"]["teacher"],
    )


def _validate_run_schedule(records: Sequence[dict], manifest: Mapping[str, object]) -> None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("round-two manifest config is absent")
    games = config.get("games")
    depths = config.get("opening_depths")
    if (
        isinstance(games, bool) or not isinstance(games, int) or games <= 0
        or not isinstance(depths, list) or not depths
        or any(isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
               for depth in depths)
        or games % (2 * len(depths)) != 0
    ):
        raise ValueError("round-two manifest depth/color schedule is incomplete")
    by_game = {record["game"]: record for record in records}
    if set(by_game) != set(range(games)):
        raise ValueError("round-two run does not contain every scheduled game")
    checkpoints = manifest.get("checkpoints")
    reanalysis_enabled = records[0]["generator"]["reanalysis"]["work"] != 0
    expected_roles = {"player_one", "player_two"} | (
        {"teacher"} if reanalysis_enabled else set()
    )
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_roles:
        raise ValueError("round-two manifest checkpoint roles are incomplete")
    checkpoint_identities = {
        role: {
            field: metadata[field]
            for field in ("model_sha256", "packed_sha256", "artifact_sha256")
        }
        for role, metadata in checkpoints.items()
    }
    for game in range(0, games, 2):
        even = by_game[game]
        odd = by_game[game + 1]
        if _pair_signature(even) != _pair_signature(odd):
            raise ValueError("paired colors do not share an identical opening/teacher")
        expected_depth = depths[(game // 2) % len(depths)]
        if even["generator"]["opening_depth"] != expected_depth:
            raise ValueError("opening depth is coupled to the wrong color slot")
        even_models = even["generator"]["models"]
        odd_models = odd["generator"]["models"]
        if not (
            even_models["player_one"] == odd_models["player_two"]
            and even_models["player_two"] == odd_models["player_one"]
        ):
            raise ValueError("paired checkpoints were not swapped across colors")
        if (
            even_models["player_one"] != checkpoint_identities["player_one"]
            or even_models["player_two"] != checkpoint_identities["player_two"]
        ):
            raise ValueError("game checkpoint identities do not match the archive")
        teacher = even["generator"]["reanalysis"]["teacher"]
        if reanalysis_enabled:
            if teacher != checkpoint_identities["teacher"]:
                raise ValueError("reanalysis teacher does not match the archive")
        elif teacher is not None:
            raise ValueError("disabled reanalysis unexpectedly names a teacher")


def _directory_paths(paths: Sequence[pathlib.Path]) -> dict[pathlib.Path, list[pathlib.Path]]:
    grouped: dict[pathlib.Path, list[pathlib.Path]] = {}
    for path in paths:
        resolved = path.resolve()
        grouped.setdefault(resolved.parent, []).append(resolved)
    return grouped


def load_games(
    current_paths: Sequence[pathlib.Path],
    archived_round1_paths: Sequence[pathlib.Path] = (),
) -> tuple[list[NativeGame], dict[str, str], dict]:
    if not current_paths:
        raise ValueError("strict-current round-two corpus is required")
    games: list[NativeGame] = []
    sources: dict[str, str] = {}
    seen_keys: set[str] = set()
    lineage = {"strict_current": [], "archived_round1": []}
    current_seeds: set[int] = set()

    def load_group(
        directory: pathlib.Path,
        paths: Sequence[pathlib.Path],
        current: bool,
    ) -> None:
        provenance_path = directory / BUILD_PROVENANCE_NAME
        if not provenance_path.is_file():
            raise ValueError(f"missing sibling build provenance: {provenance_path}")
        raw_contract = provenance_path.read_bytes()
        if current:
            contract_digest, contract = _validate_round2_build_contract(
                raw_contract, directory, True
            )
            manifest_schema = RUN_SCHEMA
        else:
            contract_digest, contract = _validate_archived_round1_contract(
                raw_contract, directory
            )
            manifest_schema = "papersoccer.jacek-native-selfplay-run/v1"
        manifest_digest, manifest = _validate_manifest(
            directory, paths, manifest_schema, contract_digest, contract,
            require_canonical=current,
        )
        if current:
            seed = manifest.get("config", {}).get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("round-two manifest seed is invalid")
            if seed in current_seeds:
                raise ValueError("round-two league run seeds must be unique")
            current_seeds.add(seed)

        records: list[dict] = []
        for path in sorted(paths):
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            source_id = f"sha256:{digest}"
            if source_id in sources:
                raise ValueError(f"duplicate corpus shard content {source_id}")
            sources[source_id] = digest
            for line_number, raw_line in enumerate(raw.splitlines(), 1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                game = (
                    validate_record(record, line_number)
                    if current else round1.validate_record(record, line_number)
                )
                if game.build_provenance_sha256 != contract_digest:
                    raise ValueError("game/build-provenance identity mismatch")
                if game.producer_sha256 != contract["producer_sha256"]:
                    raise ValueError("game/producer identity mismatch")
                game = dataclasses.replace(game, build_contract=contract)
                if game.key in seen_keys:
                    raise ValueError(f"duplicate game key {game.key}")
                seen_keys.add(game.key)
                games.append(game)
                if current:
                    records.append(record)
        if current:
            _validate_run_schedule(records, manifest)
        lineage["strict_current" if current else "archived_round1"].append({
            "manifest_sha256": manifest_digest,
            "build_provenance_sha256": contract_digest,
            "binary_sha256": contract["binary"]["sha256"],
            "shard_sha256": sorted(output["sha256"]
                                     for output in manifest["shard_outputs"]),
            "games": manifest["config"]["games"],
            "seed": manifest["config"]["seed"],
        })

    for directory, paths in sorted(_directory_paths(current_paths).items()):
        load_group(directory, paths, True)
    for directory, paths in sorted(_directory_paths(archived_round1_paths).items()):
        load_group(directory, paths, False)
    if not games:
        raise ValueError("cumulative corpus contains no games")
    games.sort(key=round1._game_sort_key)
    for category in lineage.values():
        category.sort(key=lambda item: (item["seed"], item["manifest_sha256"]))
    return games, {key: sources[key] for key in sorted(sources)}, lineage


def build_contracts(games: Sequence[NativeGame]) -> list[dict]:
    return round1.build_contracts(games)


def summarize(
    current_paths: Sequence[pathlib.Path],
    archived_round1_paths: Sequence[pathlib.Path] = (),
) -> dict:
    games, source_hashes, lineage = load_games(
        current_paths, archived_round1_paths
    )
    splits, overlaps_removed, assignment = prepare_splits(games)
    builds = build_contracts(games)
    return {
        "schema": "papersoccer.jacek-native-corpus-report/v2",
        "game_schema": GAME_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "rules": RULES,
        "sources": source_hashes,
        "lineage": lineage,
        "games": len(games),
        "samples": sum(len(game.samples) for game in games),
        "split_games": dict(Counter(
            assignment[game.split_group] for game in games
        )),
        "split_samples": {name: len(value) for name, value in splits.items()},
        "cross_split_overlaps_removed": overlaps_removed,
        "stable_reanalysis_samples": sum(
            sample.auxiliary_value is not None
            for game in games for sample in game.samples
        ),
        "generation": {
            "producer_sha256": sorted({game.producer_sha256 for game in games}),
            "build_provenance_sha256": [item["sha256"] for item in builds],
            "build_contracts": builds,
            "model_artifacts": [dataclasses.asdict(artifact) for artifact in sorted({
                artifact for game in games for artifact in game.model_artifacts
            }, key=lambda item: (
                item.artifact_sha256, item.model_sha256, item.packed_sha256
            ))],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate strict-current round-two and archived round-one corpora."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--archived-round1", nargs="*", type=pathlib.Path, default=[]
    )
    parser.add_argument("--report", type=pathlib.Path)
    arguments = parser.parse_args()
    report = summarize(arguments.corpus, arguments.archived_round1)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
