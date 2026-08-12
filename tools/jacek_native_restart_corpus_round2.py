#!/usr/bin/env python3
"""Strict validation for native self-play restarted from live loss prefixes.

Only an explicitly supplied collector clean-auditor TSV and the artifacts
named by its sibling run manifest are opened.  Observed actions are replayed to
construct a boundary; they are never emitted as policy or value targets.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
from collections import Counter
from typing import Mapping, Sequence


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus as round1  # noqa: E402
import jacek_native_corpus_round2 as round2  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
GAME_SCHEMA = "papersoccer.jacek-native-restart-game/v1"
GENERATOR_SCHEMA = "jacek-native-live-restart-bfm/v1"
RUN_SCHEMA = "papersoccer.jacek-native-restart-run/v1"
BUILD_PROVENANCE_SCHEMA = (
    "papersoccer.jacek-native-restart-build-provenance/v1"
)
BUILD_PROVENANCE_NAME = "build-provenance.json"
ARCHIVED_BINARY_NAME = "selfplay-restart-round2-binary"
ARCHIVED_INPUT_NAME = "collector-clean.tsv"
MANIFEST_NAME = "manifest.json"
OPENING_SCHEMA = "collector-clean-candidate-loss-prefix/v1"
COLOR_SCHEDULE = "swap-player-checkpoints-on-odd-continuations/v1"
TEMPERATURE_SCHEDULE = (
    "restart-relative-complete-turn-index-before-cutoff/v1"
)
OBSERVED_USAGE = "state-construction-only"
COLLECTOR_HEADER = b"game_id\tcandidate_player\twinner\tturns"
REQUIRED_METADATA = (
    "agent_id",
    "arena_manifest_sha256",
    "asserted_source_sha256",
    "asserted_submission_id",
    "collector_sha256",
    "exclusion_registry_sha256",
    "repository_commit",
    "run_id",
    "source_binding_status",
)
SHA_FIELDS = (
    "arena_manifest_sha256",
    "asserted_source_sha256",
    "collector_sha256",
    "exclusion_registry_sha256",
)
BUILD_SOURCE_PATHS = (
    "tools/jacek_native_restart_round2.cpp",
    *round2.BUILD_SOURCE_PATHS,
)
CANONICAL_BUILD_ARGV = (
    "$CXX", "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra",
    "-Wpedantic", "-Iinclude", "-Isrc/bots",
    "tools/jacek_native_restart_round2.cpp", "src/core/rules.cpp",
    "src/core/geometry.cpp", "-o", "$OUTPUT",
)
FORBIDDEN_PATH_TOKENS = (
    "matches.json", "protected", "sealed", "prospective", "final-bank",
    "final_bank",
)
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
LOWER_SHA = re.compile(r"[0-9a-f]{64}")


NativeGame = round1.NativeGame
NativeSample = round1.NativeSample
NativeModelArtifact = round1.NativeModelArtifact
prepare_splits = round2.prepare_splits
build_contracts = round1.build_contracts


@dataclasses.dataclass(frozen=True)
class CollectorGame:
    game_id: str
    candidate_player: int
    winner: int
    actions: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SelectedPrefix:
    game_id: str
    candidate_player: int
    observed_winner: int
    observed_turn_count: int
    prefix_turn: int
    transcript: str
    state_id: str


@dataclasses.dataclass(frozen=True)
class CollectorInput:
    sha256: str
    metadata: Mapping[str, str]
    games: tuple[CollectorGame, ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) + "\n").encode()


def _safe_explicit_path(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is not an explicit file: {path}") from error
    for candidate in (path, resolved):
        rendered = str(candidate).lower()
        components = tuple(component.lower() for component in candidate.parts)
        names_final_evidence = any(
            component == "final" or component.startswith("final.")
            for component in components
        )
        if (
            any(token in rendered for token in FORBIDDEN_PATH_TOKENS)
            or names_final_evidence
        ):
            raise ValueError(
                f"{label} path contains a forbidden evidence token"
            )
    if not resolved.is_file():
        raise ValueError(f"{label} is not an explicit file: {path}")
    return resolved


def parse_collector_bytes(raw: bytes) -> CollectorInput:
    if (
        not raw or len(raw) > 32 * 1024 * 1024 or b"\x00" in raw
        or b"\r" in raw
    ):
        raise ValueError("collector TSV must be bounded canonical LF text")
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if not lines or any(not line for line in lines):
        raise ValueError("collector TSV contains a blank line")

    metadata: dict[str, str] = {}
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith(b"# "):
            if header_index is not None or line.count(b"=") != 1:
                raise ValueError("collector metadata syntax/order is invalid")
            key_raw, value_raw = line[2:].split(b"=", 1)
            try:
                key = key_raw.decode("ascii")
                value = value_raw.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("collector metadata is not ASCII") from error
            if (
                not IDENTIFIER.fullmatch(key) or not value
                or len(value) > 512 or key in metadata
                or any(ord(character) < 0x20 or ord(character) > 0x7E
                       for character in value)
            ):
                raise ValueError("collector metadata is unsafe or duplicated")
            metadata[key] = value
            continue
        if line != COLLECTOR_HEADER:
            raise ValueError("collector TSV header is not exact")
        header_index = index
        break
    if header_index is None:
        raise ValueError("collector TSV header is missing")
    if set(metadata) != set(REQUIRED_METADATA):
        raise ValueError("collector TSV metadata fields are not frozen")
    if any(LOWER_SHA.fullmatch(metadata[field]) is None for field in SHA_FIELDS):
        raise ValueError("collector TSV SHA-256 metadata is malformed")
    for field in ("agent_id", "asserted_submission_id"):
        if not metadata[field].isdigit() or int(metadata[field]) >= 1 << 64:
            raise ValueError(f"collector {field} is malformed")
    if re.fullmatch(r"[0-9a-f]{40}", metadata["repository_commit"]) is None:
        raise ValueError("collector repository_commit is not a full commit")
    if IDENTIFIER.fullmatch(metadata["run_id"]) is None:
        raise ValueError("collector run_id is unsafe")
    if metadata["source_binding_status"] not in {
        "asserted-not-api-verified", "api-verified",
    }:
        raise ValueError("collector source binding status is unsupported")

    games: list[CollectorGame] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines[header_index + 1:],
                                           header_index + 2):
        fields = raw_line.split(b"\t")
        if len(fields) != 4:
            raise ValueError(
                f"collector line {line_number} does not have four fields"
            )
        try:
            game_id, candidate, winner, transcript = (
                field.decode("ascii") for field in fields
            )
        except UnicodeDecodeError as error:
            raise ValueError(
                f"collector line {line_number} is not ASCII"
            ) from error
        if (
            not game_id.isdigit() or len(game_id) > 32
            or int(game_id) >= 1 << 64 or game_id in seen_ids
        ):
            raise ValueError(
                f"collector line {line_number} has invalid/duplicate game_id"
            )
        if candidate not in {"0", "1"} or winner not in {"0", "1"}:
            raise ValueError(
                f"collector line {line_number} has invalid players"
            )
        actions = transcript.split("/")
        if (
            not transcript or len(actions) > 1024
            or any(not action or len(action) > 65_536
                   or any(direction not in "01234567" for direction in action)
                   for action in actions)
        ):
            raise ValueError(
                f"collector line {line_number} has invalid complete turns"
            )
        seen_ids.add(game_id)
        games.append(CollectorGame(
            game_id, int(candidate), int(winner), tuple(actions)
        ))
    if not games or len(games) > 512:
        raise ValueError("collector TSV has no games or exceeds its limit")
    return CollectorInput(sha256_bytes(raw), dict(metadata), tuple(games))


def read_collector(path: pathlib.Path) -> CollectorInput:
    return parse_collector_bytes(
        _safe_explicit_path(path, "collector TSV").read_bytes()
    )


def select_prefixes(
    collector: CollectorInput, prefixes_per_loss: int,
    maximum_selected_prefixes: int = 0,
) -> tuple[SelectedPrefix, ...]:
    if isinstance(prefixes_per_loss, bool) or not 1 <= prefixes_per_loss <= 32:
        raise ValueError("prefixes_per_loss must be in [1,32]")
    selected: list[SelectedPrefix] = []
    seen_states: set[str] = set()
    losses = 0
    for game in collector.games:
        state = round1._initial_replay_state()
        prefix_actions: list[str] = []
        eligible: list[SelectedPrefix] = []
        is_loss = game.winner != game.candidate_player
        losses += int(is_loss)
        for turn, action in enumerate(game.actions):
            if state.winner is not None:
                raise ValueError(
                    f"collector game {game.game_id} continues after terminal"
                )
            if is_loss and turn != 0 and state.to_move == game.candidate_player:
                active = round1._encode_replay_features(state)
                eligible.append(SelectedPrefix(
                    game.game_id, game.candidate_player, game.winner,
                    len(game.actions), turn, "/".join(prefix_actions),
                    round1.canonical_state_id(active),
                ))
            round1._apply_complete_turn(
                state, action, turn, 1, opening=False
            )
            prefix_actions.append(action)
        if state.winner is None or state.winner != game.winner:
            raise ValueError(
                f"collector game {game.game_id} is nonterminal or mismatched"
            )
        count = min(prefixes_per_loss, len(eligible))
        for index in range(count):
            source = 0 if count == 1 else index * (len(eligible) - 1) // (count - 1)
            prefix = eligible[source]
            if prefix.state_id not in seen_states:
                seen_states.add(prefix.state_id)
                selected.append(prefix)
    if not losses:
        raise ValueError("collector TSV contains no candidate losses")
    if not selected:
        raise ValueError("candidate losses contain no eligible restart prefixes")
    if (
        isinstance(maximum_selected_prefixes, bool)
        or not 0 <= maximum_selected_prefixes <= 4_096
    ):
        raise ValueError("maximum_selected_prefixes must be in [0,4096]")
    if maximum_selected_prefixes and len(selected) > maximum_selected_prefixes:
        if maximum_selected_prefixes == 1:
            selected = [selected[len(selected) // 2]]
        else:
            selected = [
                selected[index * (len(selected) - 1)
                         // (maximum_selected_prefixes - 1)]
                for index in range(maximum_selected_prefixes)
            ]
    return tuple(selected)


def _model_identity(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "model_sha256", "packed_sha256", "artifact_sha256",
    }:
        raise ValueError(f"{label} identity fields are invalid")
    if any(LOWER_SHA.fullmatch(value[field]) is None for field in value):
        raise ValueError(f"{label} identity is malformed")
    return dict(value)


def _runtime_identity(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"restart checkpoint archive is missing: {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        len(lines) != 7
        or lines[0] != "papersoccer.jacek-native-runtime-model/v1"
        or lines[1] != "jacek_native_model/v1"
        or lines[2] != round2.FEATURE_SCHEMA
        or LOWER_SHA.fullmatch(lines[3]) is None
        or LOWER_SHA.fullmatch(lines[4]) is None
    ):
        raise ValueError("restart checkpoint runtime contract is malformed")
    try:
        scales = [float(value) for value in lines[5].split()]
        packed = base64.b64decode(lines[6], validate=True)
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise ValueError("restart checkpoint payload is malformed") from error
    expected_bytes = (38_048 * 3 + 7) // 8
    if (
        len(scales) != 3
        or any(not math.isfinite(value) or value <= 0 for value in scales)
        or len(packed) != expected_bytes
        or hashlib.sha256(packed).hexdigest() != lines[4]
    ):
        raise ValueError("restart checkpoint tensor identity is malformed")
    return {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256": lines[3],
        "packed_sha256": lines[4],
    }


def _validate_build_contract(
    raw: bytes, directory: pathlib.Path, verify_local_build: bool,
) -> tuple[str, dict]:
    contract = json.loads(raw)
    if raw != canonical_json_bytes(contract):
        raise ValueError("restart build provenance is not canonical JSON")
    if not isinstance(contract, dict) or set(contract) != {
        "schema", "binary", "compiler", "build_argv", "producer_sha256",
        "sources",
    }:
        raise ValueError("restart build provenance fields are not frozen")
    if (
        contract["schema"] != BUILD_PROVENANCE_SCHEMA
        or contract["build_argv"] != list(CANONICAL_BUILD_ARGV)
    ):
        raise ValueError("restart build contract is not frozen")
    binary = contract["binary"]
    if (
        not isinstance(binary, dict) or set(binary) != {"path", "sha256"}
        or binary["path"] != ARCHIVED_BINARY_NAME
        or LOWER_SHA.fullmatch(str(binary["sha256"])) is None
    ):
        raise ValueError("restart binary identity is malformed")
    compiler = contract["compiler"]
    if (
        not isinstance(compiler, dict) or set(compiler) != {
            "executable", "sha256", "version", "version_sha256",
        }
        or not isinstance(compiler["executable"], str)
        or pathlib.PurePath(compiler["executable"]).name != compiler["executable"]
        or LOWER_SHA.fullmatch(str(compiler["sha256"])) is None
        or not isinstance(compiler["version"], str) or not compiler["version"]
        or len(compiler["version"]) > 16_384
        or hashlib.sha256(compiler["version"].encode()).hexdigest()
        != compiler["version_sha256"]
    ):
        raise ValueError("restart compiler identity is malformed")
    sources = contract["sources"]
    if not isinstance(sources, list) or len(sources) != len(BUILD_SOURCE_PATHS):
        raise ValueError("restart source identity list is incomplete")
    pairs = []
    for expected, entry in zip(BUILD_SOURCE_PATHS, sources):
        if (
            not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
            or entry["path"] != expected
            or LOWER_SHA.fullmatch(str(entry["sha256"])) is None
        ):
            raise ValueError("restart source identity is malformed")
        pairs.append([entry["path"], entry["sha256"]])
    producer = hashlib.sha256(json.dumps(
        pairs, separators=(",", ":")
    ).encode()).hexdigest()
    if contract["producer_sha256"] != producer:
        raise ValueError("restart producer identity is inconsistent")
    rendered = raw.decode("utf-8").lower()
    home = str(pathlib.Path.home()).lower()
    if any(token in rendered for token in (
        *round1.FORBIDDEN_PROVENANCE, *FORBIDDEN_PATH_TOKENS,
        "/users/", "/home/", "\\users\\",
    )) or (home and home in rendered):
        raise ValueError("restart build provenance contains a forbidden path")
    binary_path = directory / ARCHIVED_BINARY_NAME
    if (
        not binary_path.is_file()
        or hashlib.sha256(binary_path.read_bytes()).hexdigest() != binary["sha256"]
    ):
        raise ValueError("restart archived binary is stale")
    if verify_local_build:
        for entry in sources:
            source = ROOT / entry["path"]
            if (
                not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise ValueError(f"restart source is stale: {entry['path']}")
        resolved = shutil.which(compiler["executable"])
        if resolved is None:
            raise ValueError("restart compiler is unavailable")
        executable = pathlib.Path(resolved).resolve()
        version = subprocess.run(
            [str(executable), "--version"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if (
            hashlib.sha256(executable.read_bytes()).hexdigest()
            != compiler["sha256"] or version != compiler["version"]
        ):
            raise ValueError("restart compiler identity is stale")
    return hashlib.sha256(raw).hexdigest(), contract


def _validate_manifest(
    directory: pathlib.Path, paths: Sequence[pathlib.Path],
    build_sha256: str, build_contract: Mapping[str, object],
) -> tuple[dict, CollectorInput, tuple[SelectedPrefix, ...]]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ValueError("restart manifest is missing")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("restart manifest is not canonical JSON")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "input", "build_provenance", "binary", "checkpoints",
        "config", "selected_prefixes", "shard_outputs",
    } or manifest["schema"] != RUN_SCHEMA:
        raise ValueError("restart manifest fields/schema are not frozen")
    if manifest["build_provenance"] != {
        "path": BUILD_PROVENANCE_NAME, "sha256": build_sha256,
    } or manifest["binary"] != build_contract["binary"]:
        raise ValueError("restart manifest build identity is inconsistent")

    input_meta = manifest["input"]
    if not isinstance(input_meta, dict) or set(input_meta) != {
        "path", "sha256", "metadata",
    } or input_meta["path"] != ARCHIVED_INPUT_NAME:
        raise ValueError("restart manifest input identity is malformed")
    input_path = directory / ARCHIVED_INPUT_NAME
    collector = read_collector(input_path)
    if (
        collector.sha256 != input_meta["sha256"]
        or dict(collector.metadata) != input_meta["metadata"]
    ):
        raise ValueError("restart archived collector TSV is stale")

    config = manifest["config"]
    if not isinstance(config, dict) or set(config) != {
        "seed", "work", "samples_per_game", "reanalysis_samples_per_game",
        "prefixes_per_loss", "max_selected_prefixes",
        "continuations_per_prefix", "shards",
        "temperature", "temperature_turns", "max_generated_complete_turns",
        "reanalysis_work", "verification_work", "records",
    }:
        raise ValueError("restart manifest config is not frozen")
    integer_fields = {
        "seed": 0, "work": 2, "samples_per_game": 1,
        "reanalysis_samples_per_game": 0, "prefixes_per_loss": 1,
        "max_selected_prefixes": 0,
        "continuations_per_prefix": 2, "shards": 1,
        "temperature_turns": 0, "max_generated_complete_turns": 1,
        "reanalysis_work": 0, "verification_work": 0, "records": 1,
    }
    for field, minimum in integer_fields.items():
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"restart config.{field} is invalid")
    if (
        config["seed"] >= 1 << 64
        or config["samples_per_game"] > 100
        or config["reanalysis_samples_per_game"] > config["samples_per_game"]
        or config["prefixes_per_loss"] > 32
        or config["max_selected_prefixes"] > 4_096
        or config["continuations_per_prefix"] > 32
        or config["continuations_per_prefix"] % 2
        or not isinstance(config["temperature"], (int, float))
        or isinstance(config["temperature"], bool)
        or not math.isfinite(float(config["temperature"]))
        or not 0 <= float(config["temperature"])
    ):
        raise ValueError("restart config limits are invalid")
    if config["reanalysis_work"] == 0:
        if config["verification_work"] != 0:
            raise ValueError("disabled restart reanalysis is inconsistent")
    elif (
        config["reanalysis_work"] != round2.TEACHER_WORK
        or config["verification_work"] != round2.VERIFICATION_WORK
        or config["reanalysis_samples_per_game"] == 0
    ):
        raise ValueError("restart reanalysis budgets are not 30k/100k")

    selected = select_prefixes(
        collector, config["prefixes_per_loss"],
        config["max_selected_prefixes"],
    )
    expected_selected = [dataclasses.asdict(prefix) for prefix in selected]
    if manifest["selected_prefixes"] != expected_selected:
        raise ValueError("restart selected-prefix plan is stale or cherry-picked")
    expected_records = len(selected) * config["continuations_per_prefix"]
    if config["records"] != expected_records:
        raise ValueError("restart record count is inconsistent")

    checkpoints = manifest["checkpoints"]
    roles = {"player_one", "player_two"} | (
        {"teacher"} if config["reanalysis_work"] else set()
    )
    if not isinstance(checkpoints, dict) or set(checkpoints) != roles:
        raise ValueError("restart checkpoint roles are incomplete")
    for role, metadata in checkpoints.items():
        if not isinstance(metadata, dict) or set(metadata) != {
            "name", "runtime", "artifact_sha256", "model_sha256",
            "packed_sha256",
        }:
            raise ValueError(f"restart {role} checkpoint metadata is malformed")
        _model_identity({
            field: metadata[field]
            for field in ("model_sha256", "packed_sha256", "artifact_sha256")
        }, role)
        runtime = directory / metadata["runtime"]
        if (
            pathlib.PurePath(metadata["runtime"]).name
            != metadata["runtime"]
            or _runtime_identity(runtime) != {
                field: metadata[field] for field in (
                    "artifact_sha256", "model_sha256", "packed_sha256"
                )
            }
        ):
            raise ValueError(f"restart {role} checkpoint archive is stale")

    outputs = manifest["shard_outputs"]
    if not isinstance(outputs, list) or len(outputs) != config["shards"]:
        raise ValueError("restart shard output list is incomplete")
    expected_paths = {path.resolve() for path in paths}
    observed_paths: set[pathlib.Path] = set()
    observed_shards: set[int] = set()
    for output in outputs:
        if not isinstance(output, dict) or set(output) != {
            "shard", "path", "sha256", "bytes", "records",
        }:
            raise ValueError("restart shard metadata fields are invalid")
        shard = output["shard"]
        name = f"shard-{shard:02d}-of-{config['shards']:02d}.jsonl"
        if (
            isinstance(shard, bool) or not isinstance(shard, int)
            or shard in observed_shards or output["path"] != name
            or LOWER_SHA.fullmatch(str(output["sha256"])) is None
            or isinstance(output["bytes"], bool)
            or not isinstance(output["bytes"], int) or output["bytes"] <= 0
            or isinstance(output["records"], bool)
            or not isinstance(output["records"], int) or output["records"] <= 0
        ):
            raise ValueError("restart shard index/name is invalid")
        observed_shards.add(shard)
        shard_path = (directory / name).resolve()
        observed_paths.add(shard_path)
        if (
            not shard_path.is_file()
            or hashlib.sha256(shard_path.read_bytes()).hexdigest()
            != output["sha256"] or shard_path.stat().st_size != output["bytes"]
        ):
            raise ValueError("restart shard identity is stale")
    if observed_shards != set(range(config["shards"])):
        raise ValueError("restart shard index set is incomplete")
    if observed_paths != expected_paths:
        raise ValueError("all and only manifest restart shards must be supplied")
    return manifest, collector, selected


def validate_record(
    record: object, manifest: Mapping[str, object], collector: CollectorInput,
    selected: Sequence[SelectedPrefix], line_number: int = 1,
) -> NativeGame:
    if not isinstance(record, dict) or set(record) != {
        "schema", "feature_schema", "rules", "generator", "seed", "game",
        "continuation", "shard_index", "shard_count", "split_group",
        "winner", "complete_turns", "transcript_schema", "transcript",
        "samples",
    } or record["schema"] != GAME_SCHEMA:
        raise ValueError(f"restart record fields/schema invalid on line {line_number}")
    round1._check_purity(record, line_number)
    generator = record["generator"]
    if not isinstance(generator, dict) or set(generator) != {
        "schema", "action", "max_actions", "deque_schedule", "search_work",
        "work_unit", "sampling_temperature", "temperature_turns",
        "temperature_schedule", "opening_schema", "opening_depth",
        "opening_transcript", "value_target", "checkpoint_color_schedule",
        "producer_sha256", "build_provenance_sha256", "models",
        "reanalysis", "source", "search_stats",
    }:
        raise ValueError(f"restart generator fields invalid on line {line_number}")
    config = manifest["config"]
    fixed = {
        "schema": GENERATOR_SCHEMA, "action": "complete-turn",
        "max_actions": 250, "deque_schedule": round1.DEQUE_SCHEDULE,
        "work_unit": "maximum-tree-nodes",
        "value_target": "mover-relative-final-outcome",
        "checkpoint_color_schedule": COLOR_SCHEDULE,
        "temperature_schedule": TEMPERATURE_SCHEDULE,
        "opening_schema": OPENING_SCHEMA,
        "search_work": config["work"],
        "sampling_temperature": config["temperature"],
        "temperature_turns": config["temperature_turns"],
    }
    if any(generator.get(field) != value for field, value in fixed.items()):
        raise ValueError(f"restart generator contract invalid on line {line_number}")
    game = record["game"]
    continuation = record["continuation"]
    if (
        isinstance(game, bool) or not isinstance(game, int)
        or not 0 <= game < config["records"]
        or isinstance(continuation, bool) or not isinstance(continuation, int)
        or not 0 <= continuation < config["continuations_per_prefix"]
    ):
        raise ValueError(f"restart game/continuation invalid on line {line_number}")
    prefix_index = game // config["continuations_per_prefix"]
    if continuation != game % config["continuations_per_prefix"]:
        raise ValueError(f"restart continuation schedule invalid on line {line_number}")
    prefix = selected[prefix_index]
    if (
        generator["opening_depth"] != prefix.prefix_turn
        or generator["opening_transcript"] != prefix.transcript
    ):
        raise ValueError(f"restart opening prefix mismatch on line {line_number}")
    source = generator["source"]
    if not isinstance(source, dict) or set(source) != {
        "input_sha256", "game_id", "candidate_player", "observed_winner",
        "observed_turn_count", "prefix_turn", "prefix_state_id",
        "observed_moves_usage", "policy_target", "input_provenance",
    } or source != {
        "input_sha256": collector.sha256,
        "game_id": prefix.game_id,
        "candidate_player": prefix.candidate_player,
        "observed_winner": prefix.observed_winner,
        "observed_turn_count": prefix.observed_turn_count,
        "prefix_turn": prefix.prefix_turn,
        "prefix_state_id": prefix.state_id,
        "observed_moves_usage": OBSERVED_USAGE,
        "policy_target": None,
        "input_provenance": dict(collector.metadata),
    }:
        raise ValueError(f"restart source provenance mismatch on line {line_number}")
    expected_seed = (
        config["seed"] + game * 0x9E3779B97F4A7C15
    ) % (1 << 64)
    if record["seed"] != str(expected_seed):
        raise ValueError(f"restart seed schedule mismatch on line {line_number}")
    if record["split_group"] != (
        f"native-live-restart:{collector.sha256}:{prefix.game_id}"
    ):
        raise ValueError(f"restart split group does not bind source game")

    checkpoint_roles = manifest["checkpoints"]
    expected_models = {
        role: {
            field: checkpoint_roles[source_role][field]
            for field in ("model_sha256", "packed_sha256", "artifact_sha256")
        }
        for role, source_role in (
            ("player_one", "player_two" if continuation % 2 else "player_one"),
            ("player_two", "player_one" if continuation % 2 else "player_two"),
        )
    }
    if generator["models"] != expected_models:
        raise ValueError(f"restart checkpoint swap mismatch on line {line_number}")
    reanalysis = generator["reanalysis"]
    teacher = None
    if config["reanalysis_work"]:
        teacher = {
            field: checkpoint_roles["teacher"][field]
            for field in ("model_sha256", "packed_sha256", "artifact_sha256")
        }
    if (
        not isinstance(reanalysis, dict) or set(reanalysis) != {
            "selection", "samples_per_game", "work", "verification_work",
            "teacher",
        }
        or reanalysis["selection"] != round2.REANALYSIS_SELECTION
        or reanalysis["work"] != config["reanalysis_work"]
        or reanalysis["verification_work"] != config["verification_work"]
        or reanalysis["teacher"] != teacher
    ):
        raise ValueError(f"restart reanalysis contract mismatch on line {line_number}")

    compatible = copy.deepcopy(record)
    compatible.pop("continuation")
    compatible["schema"] = round2.GAME_SCHEMA
    copied = compatible["generator"]
    copied["schema"] = round2.GENERATOR_SCHEMA
    copied["checkpoint_color_schedule"] = round2.COLOR_SCHEDULE
    copied["temperature_schedule"] = (
        "absolute-complete-turn-index-before-cutoff/v1"
    )
    copied["opening_schema"] = (
        "deterministic-procedural-complete-turn-prefix/v1"
    )
    copied["opening_seed"] = "0"
    copied["opening_retry"] = 0
    copied["opening_pair_index"] = game // 2
    copied.pop("source")
    native_game = round2.validate_record(compatible, line_number)
    if native_game.search_stats["searches"] > config["max_generated_complete_turns"]:
        raise ValueError(f"restart generated-turn cap exceeded on line {line_number}")
    restart_key = f"native-live-restart:{collector.sha256}:{expected_seed}:{game}"
    samples = tuple(dataclasses.replace(sample, game_key=restart_key)
                    for sample in native_game.samples)
    return dataclasses.replace(native_game, key=restart_key, samples=samples)


def load_games(
    paths: Sequence[pathlib.Path], verify_local_build: bool = True,
) -> tuple[list[NativeGame], dict[str, str], dict]:
    if not paths:
        raise ValueError("explicit restart shard paths are required")
    resolved = [path.resolve() for path in paths]
    directories = {path.parent for path in resolved}
    if len(directories) != 1:
        raise ValueError("one restart run directory must be validated at a time")
    directory = next(iter(directories))
    provenance_path = directory / BUILD_PROVENANCE_NAME
    if not provenance_path.is_file():
        raise ValueError("restart build provenance is missing")
    build_sha, build_contract = _validate_build_contract(
        provenance_path.read_bytes(), directory, verify_local_build
    )
    manifest, collector, selected = _validate_manifest(
        directory, resolved, build_sha, build_contract
    )
    games: list[NativeGame] = []
    seen_keys: set[str] = set()
    source_hashes: dict[str, str] = {}
    record_count = 0
    report_by_path = {
        (directory / output["path"]).resolve(): output
        for output in manifest["shard_outputs"]
    }
    for path in sorted(resolved):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        source_id = f"sha256:{digest}"
        if source_id in source_hashes:
            raise ValueError(f"duplicate restart shard content {source_id}")
        source_hashes[source_id] = digest
        shard_records = 0
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line:
                raise ValueError("restart shard contains a blank JSONL line")
            game = validate_record(
                json.loads(line), manifest, collector, selected, line_number
            )
            if (
                game.build_provenance_sha256 != build_sha
                or game.producer_sha256 != build_contract["producer_sha256"]
            ):
                raise ValueError("restart game/build provenance mismatch")
            game = dataclasses.replace(game, build_contract=build_contract)
            if game.key in seen_keys:
                raise ValueError(f"duplicate restart game key {game.key}")
            seen_keys.add(game.key)
            games.append(game)
            record_count += 1
            shard_records += 1
        if shard_records != report_by_path[path]["records"]:
            raise ValueError("restart shard record count is stale")
    if record_count != manifest["config"]["records"]:
        raise ValueError("restart JSONL record count is incomplete")
    games.sort(key=round1._game_sort_key)
    lineage = {
        "manifest_sha256": hashlib.sha256(
            (directory / MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "build_provenance_sha256": build_sha,
        "binary_sha256": build_contract["binary"]["sha256"],
        "collector_tsv_sha256": collector.sha256,
        "arena_manifest_sha256": collector.metadata["arena_manifest_sha256"],
        "asserted_source_sha256": collector.metadata["asserted_source_sha256"],
        "exclusion_registry_sha256": (
            collector.metadata["exclusion_registry_sha256"]
        ),
        "source_binding_status": collector.metadata["source_binding_status"],
        "games": len(games),
        "selected_prefixes": len(selected),
    }
    return games, dict(sorted(source_hashes.items())), lineage


def summarize(
    paths: Sequence[pathlib.Path], verify_local_build: bool = True
) -> dict:
    games, sources, lineage = load_games(paths, verify_local_build)
    splits, removed, assignments = prepare_splits(games)
    return {
        "schema": "papersoccer.jacek-native-restart-corpus-report/v1",
        "game_schema": GAME_SCHEMA,
        "feature_schema": round2.FEATURE_SCHEMA,
        "rules": round2.RULES,
        "sources": sources,
        "lineage": lineage,
        "games": len(games),
        "samples": sum(len(game.samples) for game in games),
        "split_games": dict(Counter(assignments[game.split_group]
                                     for game in games)),
        "split_samples": {name: len(samples) for name, samples in splits.items()},
        "cross_split_overlaps_removed": removed,
        "observed_move_policy_labels": 0,
        "build_contracts": build_contracts(games),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate explicit provenance-safe live-restart shards."
    )
    parser.add_argument("corpus", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--archived", action="store_true",
        help=(
            "validate explicit historical identities without requiring the "
            "archived compiler/source bytes to equal the current workspace"
        ),
    )
    parser.add_argument("--report", type=pathlib.Path)
    arguments = parser.parse_args()
    report = summarize(
        arguments.corpus, verify_local_build=not arguments.archived
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
