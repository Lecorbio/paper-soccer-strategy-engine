"""Deterministic infrastructure for the separate Game Review strength gate.

The flagship study is a published, immutable record.  This module deliberately
does not import or mutate its runner.  It uses the same public arena and opening
bank formats while giving DeepTurnSearch its own identities, selection rule,
calibration mapping, raw-output root, and protected one-shot test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


MANIFEST_SCHEMA = "papersoccer.game-review-gate-manifest.v1"
OPENING_SCHEMA = "papersoccer.opening-bank.v1"
OPENING_IDENTITIES_SCHEMA = "papersoccer.game-review-opening-identities.v1"
PHASE_RESULT_SCHEMA = "papersoccer.game-review-gate-phase.v1"
LATENCY_SCHEMA = "papersoccer.game-review-gate-wasm-latency.v1"
CALIBRATION_SCHEMA = "papersoccer.game-review-calibration.v1"
SELECTION_SCHEMA = "papersoccer.game-review-gate-selection.v1"
COMPACT_RESULT_SCHEMA = "papersoccer.game-review-gate-result.v1"
WEB_GATE_SCHEMA = "papersoccer.game-review-gate-web.v1"
TEST_MARKER_SCHEMA = "papersoccer.game-review-gate-test-once.v1"
COMPETITION_SOURCE_SCHEMA = "papersoccer.game-review-competition-source.v1"

PHASES = ("development", "validation", "test")
DEPTHS = (4, 8, 12, 20)
CANDIDATE_BUDGETS = (100_000, 200_000, 400_000)
REFERENCE_IDS = ("rank5-derived-fixed-50k", "jacek-inspired-20k")
BOOTSTRAP_RESAMPLES = 10_000
RANK5_SOURCE_SHA256 = (
    "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
)
JACEK_MODEL_SHA256 = (
    "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084"
)
FAST_ANALYSIS_PROFILE = {
    "id": "complete-turn-analysis-fast-50k",
    "kind": "complete-turn-analysis",
    "review_mode": "Fast",
    "max_turn_depth": 32,
    "max_nodes": 50_000,
    "transposition_entries": 65_536,
    "evaluation_entries": 32_768,
    "wall_clock_limit_ms": 0,
    "replay_corrections": False,
    "learned_value_blend_percent": 0,
    "ranked_evaluator_sha256": RANK5_SOURCE_SHA256,
}

_COMPETITION_SOURCE_PATHS = (
    "CMakeLists.txt",
    "include/papersoccer",
    "src/analysis",
    "src/arena",
    "src/bots",
    "src/core",
    "src/opening_bank",
    "src/web",
    "submissions/codingame/bots/rank_5/bot.cpp",
    "submissions/codingame/bots/rank_5/replay_book.hpp",
    "submissions/codingame/bots/rank_5/replay_value_model.hpp",
    "benchmarks/game_review_gate/gate.py",
    "benchmarks/game_review_gate/run_gate.py",
    "benchmarks/game_review_gate/measure_wasm_latency.mjs",
    "tools/opening_bank.cpp",
)


class GateError(RuntimeError):
    """Raised when frozen gate data or workflow state is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise GateError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def competition_source_identity(repository: pathlib.Path) -> dict[str, Any]:
    """Hash the tracked competition/search implementation, excluding evidence."""

    process = subprocess.run(
        ["git", "ls-files", "-z", "--", *_COMPETITION_SOURCE_PATHS],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    relative_paths = sorted(
        encoded.decode("utf-8")
        for encoded in process.stdout.split(b"\0")
        if encoded
    )
    if not relative_paths:
        raise GateError("competition source identity has no tracked files")
    entries = [
        {"path": relative, "sha256": sha256_file(repository / relative)}
        for relative in relative_paths
    ]
    return {
        "schema": COMPETITION_SOURCE_SCHEMA,
        "algorithm": "sha256-canonical-tracked-path-file-digests-v1",
        "tracked_files": len(entries),
        "sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def _checked_competition_source_identity(
    context: ManifestContext, value: Any
) -> dict[str, Any]:
    identity = _exact_keys(
        value,
        {"schema", "algorithm", "tracked_files", "sha256"},
        "competition source identity",
    )
    if identity != competition_source_identity(context.repository):
        raise GateError(
            "competition source identity differs from development/validation evidence"
        )
    return identity


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"could not read JSON {path}: {error}") from error


def _atomic_write(path: pathlib.Path, data: bytes, *, replace: bool = False) -> bool:
    """Write one artifact atomically; return True when identical data existed."""

    if path.exists():
        try:
            current = path.read_bytes()
        except OSError as error:
            raise GateError(f"could not read existing artifact {path}: {error}") from error
        if current == data:
            return True
        if not replace:
            raise GateError(f"refusing to replace different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_json(path: pathlib.Path, value: Any, *, replace: bool = False) -> bool:
    return _atomic_write(path, canonical_json_bytes(value), replace=replace)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise GateError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{where} must be a nonempty string")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{where} must be a finite number")
    return result


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise GateError(f"{where} must be boolean")
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    result = _object(value, where)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GateError(f"{where} fields differ; missing={missing}, extra={extra}")
    return result


def _sha256(value: Any, where: str) -> str:
    text = _string(value, where)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise GateError(f"{where} must be lowercase SHA-256")
    return text


def _repo_path(repository: pathlib.Path, value: Any, where: str) -> pathlib.Path:
    text = _string(value, where)
    raw = pathlib.Path(text)
    if raw.is_absolute() or ".." in raw.parts:
        raise GateError(f"{where} must be a repository-relative path")
    resolved = (repository / raw).resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise GateError(f"{where} escapes the repository") from error
    return resolved


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


@dataclasses.dataclass(frozen=True)
class OpeningBank:
    path: str
    phase: str
    depth: int
    pairs: int
    generator_seed: str
    sha256: str
    records: tuple[OpeningRecord, ...]


_OPENING_METADATA = (
    "phase",
    "depth",
    "pairs",
    "rules",
    "generator",
    "generator_seed",
    "selection",
    "state_hash_algorithm",
    "canonicalization",
    "opening_ply_definition",
)


def parse_opening_bank(path: pathlib.Path, repository: pathlib.Path) -> OpeningBank:
    try:
        text = path.read_bytes()
    except OSError as error:
        raise GateError(f"could not read opening bank {path}: {error}") from error
    if b"\r" in text:
        raise GateError(f"opening bank must use LF line endings: {path}")
    try:
        lines = text.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise GateError(f"opening bank is not UTF-8: {path}") from error
    if not lines or lines[0] != f"schema\t{OPENING_SCHEMA}":
        raise GateError(f"unsupported opening-bank schema: {path}")
    if len(lines) < 12 or any(not line for line in lines):
        raise GateError(f"opening bank has missing or blank lines: {path}")
    metadata: dict[str, str] = {}
    for index, key in enumerate(_OPENING_METADATA, start=1):
        fields = lines[index].split("\t")
        if len(fields) != 2 or fields[0] != key:
            raise GateError(f"opening-bank metadata mismatch at {path}:{index + 1}")
        metadata[key] = fields[1]
    header_index = 1 + len(_OPENING_METADATA)
    expected_header = (
        "opening_id\tphase\tdepth\tgeneration_seed\tstate_hash\tcanonical_key\t"
        "to_move\tmoves"
    )
    if lines[header_index] != expected_header:
        raise GateError(f"opening-bank header mismatch: {path}")
    expected_metadata = {
        "rules": "8x10;opponent_goal_only;player_to_move_loses",
        "generator": "uniform-legal-move-generator/v1",
        "selection": "splitmix64-unbiased-rejection-sampling/v1",
        "state_hash_algorithm": "sha256-canonical-game-state/v1",
        "canonicalization": "horizontal-reflection-min-serialization-sha256/v1",
        "opening_ply_definition": (
            "one physical selected edge, including rebound edges"
        ),
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise GateError(f"opening-bank {key} mismatch: {path}")
    try:
        depth = int(metadata["depth"])
        pairs = int(metadata["pairs"])
    except ValueError as error:
        raise GateError(f"opening-bank numeric metadata is invalid: {path}") from error
    records: list[OpeningRecord] = []
    for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        fields = line.split("\t")
        if len(fields) != 8:
            raise GateError(f"opening record must have eight fields at {path}:{line_number}")
        try:
            record_depth = int(fields[2])
        except ValueError as error:
            raise GateError(f"opening record depth is invalid at {path}:{line_number}") from error
        moves: list[tuple[int, int]] = []
        for endpoint in fields[7].split(";") if fields[7] else []:
            coordinates = endpoint.split(",")
            if len(coordinates) != 2:
                raise GateError(f"opening move is invalid at {path}:{line_number}")
            try:
                moves.append((int(coordinates[0]), int(coordinates[1])))
            except ValueError as error:
                raise GateError(f"opening move is invalid at {path}:{line_number}") from error
        record = OpeningRecord(
            opening_id=_string(fields[0], f"{path}:{line_number} opening_id"),
            phase=fields[1],
            depth=record_depth,
            generation_seed=fields[3],
            state_hash=_sha256(fields[4], f"{path}:{line_number} state_hash"),
            canonical_key=_sha256(fields[5], f"{path}:{line_number} canonical_key"),
            to_move=fields[6],
            moves=tuple(moves),
        )
        if record.phase != metadata["phase"] or record.depth != depth:
            raise GateError(f"opening record metadata mismatch at {path}:{line_number}")
        if record.to_move not in ("one", "two") or len(record.moves) != depth:
            raise GateError(f"opening record content mismatch at {path}:{line_number}")
        expected_prefix = f"{record.phase}-d{record.depth}-{record.state_hash}"
        if record.opening_id != expected_prefix:
            raise GateError(f"opening ID is not stable at {path}:{line_number}")
        records.append(record)
    if depth <= 0 or pairs <= 0 or len(records) != pairs:
        raise GateError(f"opening-bank pair count is inconsistent: {path}")
    try:
        relative = str(path.resolve().relative_to(repository.resolve()))
    except ValueError as error:
        raise GateError(f"opening bank lies outside repository: {path}") from error
    return OpeningBank(
        path=relative,
        phase=metadata["phase"],
        depth=depth,
        pairs=pairs,
        generator_seed=metadata["generator_seed"],
        sha256=sha256_bytes(text),
        records=tuple(records),
    )


def _validate_disjoint(banks: Sequence[OpeningBank]) -> None:
    identifiers: dict[str, str] = {}
    state_hashes: dict[str, str] = {}
    canonical_keys: dict[str, str] = {}
    for bank in banks:
        for record in bank.records:
            for value, seen, label in (
                (record.opening_id, identifiers, "opening ID"),
                (record.state_hash, state_hashes, "opening state"),
                (record.canonical_key, canonical_keys, "canonical opening"),
            ):
                previous = seen.get(value)
                if previous is not None:
                    raise GateError(
                        f"{label} overlap between {previous} and {bank.path}: {value}"
                    )
                seen[value] = bank.path


def profile_sha256(profile: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(profile)))


def _validate_candidate_profile(value: Any, index: int) -> dict[str, Any]:
    profile = _exact_keys(
        value,
        {
            "id",
            "kind",
            "public_label",
            "max_turn_depth",
            "max_nodes",
            "transposition_entries",
            "evaluation_entries",
            "wall_clock_limit_ms",
            "replay_corrections",
            "learned_value_blend_percent",
        },
        f"profiles.candidates[{index}]",
    )
    budget = CANDIDATE_BUDGETS[index]
    expected_id = f"deep-turn-search-{budget // 1000}k"
    if profile != {
        "id": expected_id,
        "kind": "deep-turn-search",
        "public_label": f"DeepTurnSearch — {budget // 1000}k analysis profile",
        "max_turn_depth": 32,
        "max_nodes": budget,
        "transposition_entries": 65_536,
        "evaluation_entries": 32_768,
        "wall_clock_limit_ms": 0,
        "replay_corrections": False,
        "learned_value_blend_percent": 0,
    }:
        raise GateError(f"candidate profile {expected_id} changed")
    return profile


def _validate_reference(value: Any, index: int) -> dict[str, Any]:
    reference = _exact_keys(
        value,
        {"id", "kind", "public_label", "settings"},
        f"profiles.references[{index}]",
    )
    if reference["id"] != REFERENCE_IDS[index]:
        raise GateError("reference order/identity changed")
    settings = _object(reference["settings"], f"reference {reference['id']} settings")
    if index == 0:
        expected = {
            "max_turn_depth": 32,
            "max_nodes": 50_000,
            "transposition_entries": 65_536,
            "evaluation_entries": 32_768,
            "wall_clock_limit_ms": 0,
            "replay_corrections": False,
            "learned_value_blend_percent": 0,
            "ranked_source_sha256": RANK5_SOURCE_SHA256,
        }
        if reference["kind"] != "rank5-derived" or settings != expected:
            raise GateError("fixed Rank5Derived reference changed")
    else:
        expected = {
            "max_turn_depth": 6,
            "max_nodes": 20_000,
            "transposition_entries": 65_536,
            "max_search_plies": 12,
            "wall_clock_limit_ms": 0,
            "model_sha256": JACEK_MODEL_SHA256,
        }
        if reference["kind"] != "jacek-inspired" or settings != expected:
            raise GateError("selected JacekInspired reference changed")
    return reference


@dataclasses.dataclass(frozen=True)
class ManifestContext:
    path: pathlib.Path
    repository: pathlib.Path
    manifest: dict[str, Any]
    manifest_sha256: str
    candidates: tuple[dict[str, Any], ...]
    references: tuple[dict[str, Any], ...]
    gate_banks: tuple[OpeningBank, ...]
    excluded_banks: tuple[OpeningBank, ...]


def validate_manifest(
    manifest_path: pathlib.Path,
    *,
    repository: pathlib.Path | None = None,
    verify_files: bool = True,
    verify_identities: bool = True,
) -> ManifestContext:
    manifest_path = manifest_path.resolve()
    repository = (
        repository.resolve()
        if repository is not None
        else manifest_path.parents[2].resolve()
    )
    top = _exact_keys(
        load_json(manifest_path),
        {
            "schema_version",
            "study",
            "rules",
            "profiles",
            "samples",
            "openings",
            "seeds",
            "latency_protocol",
            "selection_rule",
            "statistics",
            "source",
            "outputs",
        },
        "manifest",
    )
    if top["schema_version"] != MANIFEST_SCHEMA:
        raise GateError("unsupported Game Review gate manifest schema")
    study = _exact_keys(
        top["study"], {"id", "title", "frozen", "status"}, "study"
    )
    if study["frozen"] is not True or study["status"] != "preregistered":
        raise GateError("gate manifest must be frozen and preregistered")
    rules = _exact_keys(
        top["rules"],
        {
            "width",
            "height",
            "goal_rule",
            "blocked_rule",
            "natural_draws",
            "max_game_plies",
            "opening_ply_definition",
        },
        "rules",
    )
    if rules != {
        "width": 8,
        "height": 10,
        "goal_rule": "opponent_goal_only",
        "blocked_rule": "player_to_move_loses",
        "natural_draws": False,
        "max_game_plies": 512,
        "opening_ply_definition": "one physical legal edge including rebounds",
    }:
        raise GateError("gate rules changed")

    profiles = _exact_keys(top["profiles"], {"candidates", "references"}, "profiles")
    raw_candidates = _array(profiles["candidates"], "profiles.candidates")
    raw_references = _array(profiles["references"], "profiles.references")
    if len(raw_candidates) != 3 or len(raw_references) != 2:
        raise GateError("gate requires exactly three candidates and two references")
    candidates = tuple(
        _validate_candidate_profile(value, index)
        for index, value in enumerate(raw_candidates)
    )
    references = tuple(
        _validate_reference(value, index) for index, value in enumerate(raw_references)
    )

    samples = _exact_keys(top["samples"], set(PHASES), "samples")
    expected_pairs = {"development": 25, "validation": 50, "test": 100}
    for phase in PHASES:
        sample = _exact_keys(
            samples[phase],
            {"color_swapped_pairs_per_depth_matchup", "games_per_pair"},
            f"samples.{phase}",
        )
        if sample != {
            "color_swapped_pairs_per_depth_matchup": expected_pairs[phase],
            "games_per_pair": 2,
        }:
            raise GateError(f"{phase} sample size changed")

    openings = _exact_keys(
        top["openings"],
        {"depths", "generator", "banks", "excluded_flagship_banks", "identities_path"},
        "openings",
    )
    if openings["depths"] != list(DEPTHS):
        raise GateError("opening depths must be exactly 4/8/12/20")
    generator = _exact_keys(
        openings["generator"],
        {"id", "selection", "canonical_equivalence", "duplicate_policy"},
        "openings.generator",
    )
    if generator != {
        "id": "uniform-legal-move-generator/v1",
        "selection": "splitmix64-unbiased-rejection-sampling/v1",
        "canonical_equivalence": "horizontal reflection about x=width/2",
        "duplicate_policy": "reject across gate and every flagship bank",
    }:
        raise GateError("opening generator contract changed")

    def declared_banks(raw: Any, where: str, gate: bool) -> tuple[OpeningBank, ...]:
        declarations = _array(raw, where)
        banks: list[OpeningBank] = []
        for index, declaration_value in enumerate(declarations):
            declaration = _exact_keys(
                declaration_value,
                {"id", "phase", "depth", "pairs", "seed", "path", "sha256"},
                f"{where}[{index}]",
            )
            path = _repo_path(repository, declaration["path"], f"{where}[{index}].path")
            if not verify_files:
                continue
            bank = parse_opening_bank(path, repository)
            if bank.sha256 != _sha256(declaration["sha256"], f"{where}[{index}].sha256"):
                raise GateError(f"opening bank hash mismatch: {path}")
            if bank.phase != declaration["phase"] or bank.depth != declaration["depth"]:
                raise GateError(f"opening bank phase/depth mismatch: {path}")
            if bank.pairs != declaration["pairs"] or bank.generator_seed != declaration["seed"]:
                raise GateError(f"opening bank pairs/seed mismatch: {path}")
            if declaration["id"] != (
                f"game-review-{bank.phase}-d{bank.depth:02d}"
                if gate
                else f"flagship-exclusion-{index:02d}"
            ):
                raise GateError(f"opening bank declaration ID changed: {path}")
            banks.append(bank)
        return tuple(banks)

    gate_banks = declared_banks(openings["banks"], "openings.banks", True)
    excluded_banks = declared_banks(
        openings["excluded_flagship_banks"],
        "openings.excluded_flagship_banks",
        False,
    )
    if len(_array(openings["banks"], "openings.banks")) != 12:
        raise GateError("manifest must contain twelve gate opening banks")
    if len(_array(openings["excluded_flagship_banks"], "excluded banks")) != 16:
        raise GateError("manifest must freeze all sixteen flagship-bank exclusions")
    if verify_files:
        expected_keys = {(phase, depth) for phase in PHASES for depth in DEPTHS}
        if {(bank.phase, bank.depth) for bank in gate_banks} != expected_keys:
            raise GateError("gate opening bank phase/depth grid is incomplete")
        for bank in gate_banks:
            expected_count = expected_pairs[bank.phase]
            if bank.pairs != expected_count:
                raise GateError(f"gate opening sample count changed: {bank.path}")
        _validate_disjoint((*excluded_banks, *gate_banks))

    seeds = _exact_keys(
        top["seeds"], {"opening", "bot", "bootstrap", "calibration"}, "seeds"
    )
    seen_seeds: set[str] = set()
    opening_seeds = _exact_keys(seeds["opening"], set(PHASES), "seeds.opening")
    for phase in PHASES:
        phase_seeds = _exact_keys(
            opening_seeds[phase], {str(depth) for depth in DEPTHS}, f"seeds.opening.{phase}"
        )
        for depth in DEPTHS:
            seed = _string(phase_seeds[str(depth)], f"opening seed {phase}/{depth}")
            if seed in seen_seeds:
                raise GateError("gate seeds must be domain-distinct")
            seen_seeds.add(seed)
            if verify_files:
                bank = next(
                    item for item in gate_banks if item.phase == phase and item.depth == depth
                )
                if bank.generator_seed != seed:
                    raise GateError("opening seed differs from bank metadata")
    for category in ("bot", "bootstrap"):
        phase_seeds = _exact_keys(seeds[category], set(PHASES), f"seeds.{category}")
        for phase in PHASES:
            seed = _string(phase_seeds[phase], f"seeds.{category}.{phase}")
            if seed in seen_seeds:
                raise GateError("gate seeds must be domain-distinct")
            seen_seeds.add(seed)
    calibration = _exact_keys(seeds["calibration"], {"validation"}, "seeds.calibration")
    if _string(calibration["validation"], "calibration seed") in seen_seeds:
        raise GateError("gate seeds must be domain-distinct")

    latency = _exact_keys(
        top["latency_protocol"],
        {
            "runtime",
            "phase",
            "fresh_possession_boundaries_only",
            "minimum_samples_per_candidate",
            "samples_per_opening_depth_per_candidate",
            "warmup_searches_per_candidate",
            "native_parity_reference",
            "quantiles",
            "p95_limit_ms",
            "maximum_limit_ms",
            "initial_memory_bytes",
            "memory_growth",
            "emscripten_version",
        },
        "latency_protocol",
    )
    if latency != {
        "runtime": "wasm",
        "phase": "validation",
        "fresh_possession_boundaries_only": True,
        "minimum_samples_per_candidate": 80,
        "samples_per_opening_depth_per_candidate": 20,
        "warmup_searches_per_candidate": 8,
        "native_parity_reference": "rank5-derived-fixed-50k",
        "quantiles": ["median", "p95", "maximum"],
        "p95_limit_ms": 400,
        "maximum_limit_ms": 750,
        "initial_memory_bytes": 67_108_864,
        "memory_growth": False,
        "emscripten_version": "6.0.2",
    }:
        raise GateError("Wasm latency protocol changed")

    selection = _exact_keys(
        top["selection_rule"],
        {
            "phase",
            "strength_metric",
            "practical_tie_percentage_points",
            "tie_break_order",
            "no_eligible_candidate_policy",
        },
        "selection_rule",
    )
    if selection["phase"] != "validation" or selection[
        "practical_tie_percentage_points"
    ] != 1.0:
        raise GateError("selection phase/tie changed")
    if selection["tie_break_order"] != [
        "lower_wasm_p95",
        "lower_work",
        "stable_profile_id",
    ]:
        raise GateError("selection tie-break order changed")
    if selection["no_eligible_candidate_policy"] != "stop_before_test":
        raise GateError("selection must stop before test without an eligible candidate")

    statistics = _exact_keys(
        top["statistics"], {"bootstrap", "calibration", "expert_gate"}, "statistics"
    )
    bootstrap = _exact_keys(
        statistics["bootstrap"],
        {"resamples", "confidence", "unit", "stratify_by"},
        "statistics.bootstrap",
    )
    if bootstrap != {
        "resamples": 10_000,
        "confidence": 0.95,
        "unit": "whole_color_swapped_pair",
        "stratify_by": "opening_depth",
    }:
        raise GateError("bootstrap contract changed")
    calibration_contract = _exact_keys(
        statistics["calibration"],
        {
            "fit_phase",
            "link",
            "score_perspective",
            "outcome",
            "mapping_profiles",
            "observation_sources",
        },
        "statistics.calibration",
    )
    expected_calibration = {
        "fit_phase": "validation",
        "link": "profile_specific_standardized_logistic",
        "score_perspective": "player_who_made_possession",
        "outcome": "eventual_binary_win",
        "mapping_profiles": {
            "fast": FAST_ANALYSIS_PROFILE,
            "deep": "selected_validation_candidate",
        },
        "observation_sources": {
            "fast": "fresh_fixed_rank5_reference_decisions",
            "deep": "fresh_deep_turn_search_candidate_decisions",
        },
    }
    if calibration_contract != expected_calibration:
        raise GateError("calibration contract changed")
    expert = _exact_keys(
        statistics["expert_gate"],
        {"test_pairs_per_depth_opponent", "bootstrap_resamples", "claim_rule"},
        "statistics.expert_gate",
    )
    if expert["test_pairs_per_depth_opponent"] != 100 or expert[
        "bootstrap_resamples"
    ] != 10_000:
        raise GateError("Expert gate sample/bootstrap changed")

    source = _exact_keys(
        top["source"], {"ranked_source", "jacek_model"}, "source"
    )
    ranked = _exact_keys(source["ranked_source"], {"path", "sha256"}, "ranked source")
    jacek = _exact_keys(source["jacek_model"], {"path", "sha256"}, "Jacek model")
    if ranked["sha256"] != RANK5_SOURCE_SHA256 or jacek["sha256"] != JACEK_MODEL_SHA256:
        raise GateError("protected source identity changed")
    if verify_files:
        for declaration, label in ((ranked, "ranked source"), (jacek, "Jacek model")):
            path = _repo_path(repository, declaration["path"], f"{label}.path")
            if sha256_file(path) != declaration["sha256"]:
                raise GateError(f"{label} hash mismatch: {path}")

    outputs = _exact_keys(
        top["outputs"],
        {
            "raw_results_root",
            "opening_identities",
            "phase_results",
            "wasm_latency",
            "selection_lock",
            "cpp_lock_header",
            "compact_results",
            "report",
            "web_gate_status",
        },
        "outputs",
    )
    raw_root = _repo_path(repository, outputs["raw_results_root"], "raw_results_root")
    ignored_root = (repository / "results" / "game_review_gate").resolve()
    if raw_root != ignored_root and ignored_root not in raw_root.parents:
        raise GateError("raw gate output must stay below results/game_review_gate")
    phase_results = _exact_keys(outputs["phase_results"], set(PHASES), "phase_results")
    curated_root = (repository / "benchmarks" / "game_review_gate").resolve()
    for where, raw in (
        ("opening identities", outputs["opening_identities"]),
        ("Wasm latency", outputs["wasm_latency"]),
        ("selection lock", outputs["selection_lock"]),
        ("C++ calibration lock", outputs["cpp_lock_header"]),
        ("compact results", outputs["compact_results"]),
        ("report", outputs["report"]),
        *((f"{phase} result", value) for phase, value in phase_results.items()),
    ):
        path = _repo_path(repository, raw, where)
        if path == curated_root or curated_root not in path.parents:
            raise GateError(f"{where} must stay below benchmarks/game_review_gate")
    web_gate_path = _repo_path(
        repository, outputs["web_gate_status"], "Web Expert gate status"
    )
    expected_web_gate_path = (repository / "web" / "game-review-gate.js").resolve()
    if web_gate_path != expected_web_gate_path:
        raise GateError("Web Expert gate status must be web/game-review-gate.js")

    context = ManifestContext(
        path=manifest_path,
        repository=repository,
        manifest=top,
        manifest_sha256=sha256_file(manifest_path),
        candidates=candidates,
        references=references,
        gate_banks=gate_banks,
        excluded_banks=excluded_banks,
    )
    if verify_files and verify_identities:
        identities_path = _repo_path(
            repository, openings["identities_path"], "openings.identities_path"
        )
        expected = build_opening_identities(context)
        if load_json(identities_path) != expected:
            raise GateError("opening identity artifact is stale")
    return context


def build_opening_identities(context: ManifestContext) -> dict[str, Any]:
    def bank_value(bank: OpeningBank) -> dict[str, Any]:
        return {
            "path": bank.path,
            "sha256": bank.sha256,
            "phase": bank.phase,
            "depth": bank.depth,
            "pairs": bank.pairs,
            "generator_seed": bank.generator_seed,
            "opening_ids": [record.opening_id for record in bank.records],
            "state_hashes": [record.state_hash for record in bank.records],
            "canonical_keys": [record.canonical_key for record in bank.records],
        }

    return {
        "schema": OPENING_IDENTITIES_SCHEMA,
        "study_id": context.manifest["study"]["id"],
        "generator": context.manifest["openings"]["generator"],
        "gate_banks": [bank_value(bank) for bank in context.gate_banks],
        "excluded_flagship_banks": [bank_value(bank) for bank in context.excluded_banks],
        "validation": {
            "all_banks": len(context.gate_banks) + len(context.excluded_banks),
            "gate_openings": sum(bank.pairs for bank in context.gate_banks),
            "excluded_flagship_openings": sum(
                bank.pairs for bank in context.excluded_banks
            ),
            "exact_state_overlaps": 0,
            "horizontal_reflection_overlaps": 0,
        },
    }


def run_opening_tool_validation(
    context: ManifestContext, opening_tool: pathlib.Path
) -> None:
    command = [str(opening_tool), "validate"]
    for bank in context.gate_banks:
        command += ["--bank", str(context.repository / bank.path)]
    for bank in context.excluded_banks:
        command += ["--exclude-bank", str(context.repository / bank.path)]
    process = subprocess.run(
        command,
        cwd=context.repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise GateError(
            "opening-bank tool rejected the frozen banks: " + process.stderr.strip()
        )


def verify_opening_regeneration(
    context: ManifestContext, opening_tool: pathlib.Path
) -> None:
    """Regenerate each gate bank in frozen order and require byte identity."""

    exclusions = [context.repository / bank.path for bank in context.excluded_banks]
    for bank in context.gate_banks:
        command = [
            str(opening_tool),
            "generate",
            "--phase",
            bank.phase,
            "--depth",
            str(bank.depth),
            "--pairs",
            str(bank.pairs),
            "--seed",
            bank.generator_seed,
        ]
        for exclusion in exclusions:
            command += ["--exclude-bank", str(exclusion)]
        process = subprocess.run(
            command,
            cwd=context.repository,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            message = process.stderr.decode("utf-8", errors="replace").strip()
            raise GateError(f"opening regeneration failed for {bank.path}: {message}")
        committed_path = context.repository / bank.path
        if process.stdout != committed_path.read_bytes():
            raise GateError(f"opening regeneration is not byte-identical: {bank.path}")
        exclusions.append(committed_path)


def _derived_seed(base: str, *parts: str) -> int:
    material = "\0".join((base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclasses.dataclass(frozen=True)
class GateUnit:
    phase: str
    matchup_id: str
    candidate_id: str
    reference_id: str
    bank_path: str
    opening_depth: int
    pairs: int

    @property
    def unit_id(self) -> str:
        return f"{self.phase}--{self.matchup_id}--d{self.opening_depth:02d}"


def _selection_path(context: ManifestContext) -> pathlib.Path:
    return _repo_path(
        context.repository,
        context.manifest["outputs"]["selection_lock"],
        "selection lock",
    )


def _phase_result_path(context: ManifestContext, phase: str) -> pathlib.Path:
    return _repo_path(
        context.repository,
        context.manifest["outputs"]["phase_results"][phase],
        f"{phase} result",
    )


def units_for_phase(
    context: ManifestContext,
    phase: str,
    selection: Mapping[str, Any] | None = None,
) -> list[GateUnit]:
    if phase not in PHASES:
        raise GateError(f"unsupported gate phase: {phase}")
    if phase == "test":
        if selection is None:
            raise GateError("test units require the frozen selection lock")
        candidate_ids = [_string(selection.get("selected_profile_id"), "selected profile")]
    else:
        candidate_ids = [profile["id"] for profile in context.candidates]
    banks = sorted(
        (bank for bank in context.gate_banks if bank.phase == phase),
        key=lambda bank: bank.depth,
    )
    return [
        GateUnit(
            phase=phase,
            matchup_id=f"{candidate_id}-vs-{reference['id']}",
            candidate_id=candidate_id,
            reference_id=reference["id"],
            bank_path=bank.path,
            opening_depth=bank.depth,
            pairs=bank.pairs,
        )
        for candidate_id in candidate_ids
        for reference in context.references
        for bank in banks
    ]


def arena_command(
    context: ManifestContext, unit: GateUnit, arena_path: pathlib.Path
) -> list[str]:
    candidate = next(
        profile for profile in context.candidates if profile["id"] == unit.candidate_id
    )
    reference = next(
        profile for profile in context.references if profile["id"] == unit.reference_id
    )
    base_seed = _derived_seed(
        context.manifest["seeds"]["bot"][unit.phase],
        unit.matchup_id,
        str(unit.opening_depth),
    )
    command = [
        str(arena_path),
        "matches",
        "--seed",
        str(base_seed),
        "--pairs",
        str(unit.pairs),
        "--opening-bank",
        str(context.repository / unit.bank_path),
        "--max-plies",
        "512",
        "--bootstrap-samples",
        "1",
        "--warmup-decisions",
        "8",
        "--candidate-kind",
        "deep-turn-search",
        "--candidate-complete-turn-max-nodes",
        str(candidate["max_nodes"]),
        "--reference-kind",
        reference["kind"],
    ]
    if reference["kind"] == "jacek-inspired":
        settings = reference["settings"]
        command += [
            "--reference-alpha-beta-depth",
            str(settings["max_turn_depth"]),
            "--reference-alpha-beta-max-nodes",
            str(settings["max_nodes"]),
            "--reference-alpha-beta-table-entries",
            str(settings["transposition_entries"]),
            "--reference-alpha-beta-max-search-plies",
            str(settings["max_search_plies"]),
        ]
    return command


def _arena_provenance(arena_path: pathlib.Path, repository: pathlib.Path) -> dict[str, Any]:
    process = subprocess.run(
        [str(arena_path), "provenance"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise GateError(f"could not read arena provenance: {process.stderr.strip()}")
    try:
        provenance = _object(json.loads(process.stdout), "arena provenance")
    except json.JSONDecodeError as error:
        raise GateError("arena provenance is not JSON") from error
    if (
        provenance.get("schema") != "papersoccer.arena-build.v1"
        or provenance.get("runtime") != "native"
        or provenance.get("build_type") != "Release"
        or provenance.get("ndebug") is not True
        or provenance.get("sanitizers_enabled") is not False
        or provenance.get("source_dirty") is not False
    ):
        raise GateError(
            "frozen gate requires a clean optimized native Release arena"
        )
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if provenance.get("source_commit") != current_commit:
        raise GateError(
            "arena provenance does not match the current frozen gate commit; rebuild"
        )
    return provenance


def _require_live_gate_source_clean(context: ManifestContext) -> None:
    """Reject source drift after the evidence arena was configured.

    CMake records cleanliness at configure time.  The frozen gate also needs to
    reject later edits while still allowing its own not-yet-committed curated
    outputs between development and validation.
    """

    for command in (
        ["git", "diff", "--quiet", "--"],
        ["git", "diff", "--cached", "--quiet", "--"],
    ):
        process = subprocess.run(command, cwd=context.repository, check=False)
        if process.returncode == 1:
            raise GateError(
                "frozen gate requires a clean tracked source tree; commit changes first"
            )
        if process.returncode != 0:
            raise GateError("could not verify the frozen gate source tree")

    outputs = context.manifest["outputs"]
    allowed_untracked = {
        _repo_path(context.repository, path, f"allowed gate output {name}")
        for name, path in {
            **outputs["phase_results"],
            "wasm_latency": outputs["wasm_latency"],
            "selection_lock": outputs["selection_lock"],
            "cpp_lock_header": outputs["cpp_lock_header"],
            "compact_results": outputs["compact_results"],
            "report": outputs["report"],
            "web_gate_status": outputs["web_gate_status"],
        }.items()
    }
    allowed_untracked.update(
        {
            (context.repository / "AGENTS.md").resolve(),
            (context.repository / "matches.json").resolve(),
        }
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=context.repository,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    unexpected = []
    for encoded in untracked:
        if not encoded:
            continue
        try:
            relative = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GateError("untracked source path is not UTF-8") from error
        path = (context.repository / relative).resolve()
        if path not in allowed_untracked:
            unexpected.append(relative)
    if unexpected:
        raise GateError(
            "frozen gate requires a clean source tree; unexpected untracked paths: "
            + ", ".join(sorted(unexpected))
        )


def _raw_root(context: ManifestContext) -> pathlib.Path:
    root = _repo_path(
        context.repository,
        context.manifest["outputs"]["raw_results_root"],
        "raw results root",
    )
    return root / context.manifest_sha256


def _checked_raw_annotation(
    context: ManifestContext, unit: GateUnit, value: Any
) -> dict[str, Any]:
    annotation = _exact_keys(
        value,
        {
            "manifest_sha256",
            "run_id",
            "unit_id",
            "phase",
            "matchup_id",
            "candidate_id",
            "reference_id",
            "opening_depth",
            "arena_sha256",
            "arena_provenance",
            "arena_command",
            "selection_sha256",
            "competition_source",
        },
        f"raw gate annotation {unit.unit_id}",
    )
    if (
        annotation["manifest_sha256"] != context.manifest_sha256
        or annotation["unit_id"] != unit.unit_id
        or annotation["phase"] != unit.phase
        or annotation["matchup_id"] != unit.matchup_id
        or annotation["candidate_id"] != unit.candidate_id
        or annotation["reference_id"] != unit.reference_id
        or annotation["opening_depth"] != unit.opening_depth
    ):
        raise GateError(f"raw gate annotation identity mismatch: {unit.unit_id}")

    arena_hash = _sha256(
        annotation["arena_sha256"], f"{unit.unit_id} arena SHA-256"
    )
    source_identity = _checked_competition_source_identity(
        context, annotation["competition_source"]
    )
    source_hash = source_identity["sha256"]
    expected_selection_hash = (
        sha256_file(_selection_path(context)) if unit.phase == "test" else "none"
    )
    if annotation["selection_sha256"] != expected_selection_hash:
        raise GateError(f"raw gate selection identity mismatch: {unit.unit_id}")
    expected_run_id = sha256_bytes(
        (
            f"{context.manifest_sha256}\0{expected_selection_hash}\0{source_hash}\0{arena_hash}"
            if unit.phase == "test"
            else f"{context.manifest_sha256}\0{unit.phase}\0{source_hash}\0{arena_hash}"
        ).encode("ascii")
    )[:24]
    if annotation["run_id"] != expected_run_id:
        raise GateError(f"raw gate run identity mismatch: {unit.unit_id}")

    provenance = _exact_keys(
        annotation["arena_provenance"],
        {
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
        },
        f"raw arena provenance {unit.unit_id}",
    )
    if (
        provenance["schema"] != "papersoccer.arena-build.v1"
        or provenance["runtime"] != "native"
        or provenance["build_type"] != "Release"
        or provenance["ndebug"] is not True
        or provenance["sanitizers_enabled"] is not False
        or provenance["source_dirty"] is not False
        or provenance["cxx_standard"] != 202002
    ):
        raise GateError(f"raw arena provenance mismatch: {unit.unit_id}")
    for key in ("compiler_id", "compiler_version", "configured_flags"):
        _string(provenance[key], f"{unit.unit_id} provenance {key}")
    commit = _string(provenance["source_commit"], f"{unit.unit_id} source commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise GateError(f"raw arena source commit is invalid: {unit.unit_id}")

    command = _array(annotation["arena_command"], f"{unit.unit_id} arena command")
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise GateError(f"raw arena command is invalid: {unit.unit_id}")
    executable = pathlib.Path(command[0])
    if not executable.is_absolute() or command != arena_command(
        context, unit, executable
    ):
        raise GateError(f"raw arena command differs from the frozen schedule: {unit.unit_id}")
    return annotation


def _require_git_tracked_clean(repository: pathlib.Path, paths: Sequence[pathlib.Path]) -> None:
    for path in paths:
        try:
            relative = path.resolve().relative_to(repository.resolve())
        except ValueError as error:
            raise GateError(f"test prerequisite is outside repository: {path}") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode != 0:
            raise GateError(f"test prerequisite is not committed: {relative}")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative)],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout:
            raise GateError(f"test prerequisite has uncommitted changes: {relative}")


def _prepare_test_once(
    raw_test_root: pathlib.Path,
    *,
    manifest_sha256: str,
    selection_sha256: str,
    competition_source_sha256: str,
    arena_sha256: str,
    committed_test_result: pathlib.Path | None = None,
) -> tuple[str, pathlib.Path]:
    run_id = sha256_bytes(
        (
            f"{manifest_sha256}\0{selection_sha256}\0"
            f"{competition_source_sha256}\0{arena_sha256}"
        ).encode("ascii")
    )[:24]
    marker = raw_test_root / "test-once.json"
    if committed_test_result is not None and committed_test_result.exists():
        raise GateError("frozen test result already exists; refusing a second test")
    if marker.exists():
        value = _object(load_json(marker), "test-once marker")
        if (
            value.get("schema") != TEST_MARKER_SCHEMA
            or value.get("run_id") != run_id
            or value.get("manifest_sha256") != manifest_sha256
            or value.get("selection_sha256") != selection_sha256
            or value.get("competition_source_sha256")
            != competition_source_sha256
            or value.get("arena_sha256") != arena_sha256
        ):
            raise GateError("incompatible test-once marker already exists")
        if value.get("completed") is True:
            raise GateError("frozen test already completed; refusing a second evaluation")
        return run_id, marker
    raw_test_root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": TEST_MARKER_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "competition_source_sha256": competition_source_sha256,
        "arena_sha256": arena_sha256,
        "completed": False,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return _prepare_test_once(
            raw_test_root,
            manifest_sha256=manifest_sha256,
            selection_sha256=selection_sha256,
            competition_source_sha256=competition_source_sha256,
            arena_sha256=arena_sha256,
            committed_test_result=committed_test_result,
        )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    return run_id, marker


def _load_and_validate_selection(context: ManifestContext) -> dict[str, Any]:
    path = _selection_path(context)
    actual = _object(load_json(path), "selection lock")
    expected = create_selection_value(context)
    if actual != expected:
        raise GateError("selection/calibration lock is stale or invalid")
    return actual


def run_phase(
    context: ManifestContext,
    arena_path: pathlib.Path,
    phase: str,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise GateError("invalid shard index/count")
    arena_path = arena_path.resolve()
    if not arena_path.is_file():
        raise GateError(f"arena executable does not exist: {arena_path}")
    _require_live_gate_source_clean(context)
    source_identity = competition_source_identity(context.repository)
    source_hash = source_identity["sha256"]
    provenance = _arena_provenance(arena_path, context.repository)
    arena_hash = sha256_file(arena_path)
    selection = None
    selection_hash = "none"
    marker: pathlib.Path | None = None
    if phase == "test":
        selection = _load_and_validate_selection(context)
        selection_path = _selection_path(context)
        latency_path = _repo_path(
            context.repository,
            context.manifest["outputs"]["wasm_latency"],
            "Wasm latency",
        )
        latency = validate_latency_value(context, load_json(latency_path))
        wasm_module_path = _repo_path(
            context.repository,
            latency["module"]["path"],
            "latency module path",
        )
        prerequisites = [
            context.path,
            context.path.parent / "gate.py",
            context.path.parent / "run_gate.py",
            context.path.parent / "measure_wasm_latency.mjs",
            _repo_path(
                context.repository,
                context.manifest["outputs"]["opening_identities"],
                "opening identities",
            ),
            selection_path,
            _repo_path(
                context.repository,
                context.manifest["outputs"]["cpp_lock_header"],
                "C++ calibration lock",
            ),
            _phase_result_path(context, "development"),
            _phase_result_path(context, "validation"),
            latency_path,
            wasm_module_path,
            _repo_path(
                context.repository,
                context.manifest["source"]["ranked_source"]["path"],
                "ranked source",
            ),
            _repo_path(
                context.repository,
                context.manifest["source"]["jacek_model"]["path"],
                "Jacek model",
            ),
            *[context.repository / bank.path for bank in context.gate_banks],
        ]
        _require_git_tracked_clean(context.repository, prerequisites)
        selection_hash = sha256_file(selection_path)
        run_id, marker = _prepare_test_once(
            _raw_root(context) / "test",
            manifest_sha256=context.manifest_sha256,
            selection_sha256=selection_hash,
            competition_source_sha256=source_hash,
            arena_sha256=arena_hash,
            committed_test_result=_phase_result_path(context, "test"),
        )
    else:
        run_id = sha256_bytes(
            (
                f"{context.manifest_sha256}\0{phase}\0"
                f"{source_hash}\0{arena_hash}"
            ).encode("ascii")
        )[:24]
    units = units_for_phase(context, phase, selection)
    selected_units = [
        unit for index, unit in enumerate(units) if index % shard_count == shard_index
    ]
    completed = resumed = 0
    for unit in selected_units:
        output = _raw_root(context) / phase / "shards" / f"{unit.unit_id}.json"
        if output.exists():
            prior = _object(load_json(output), str(output))
            _validate_arena_report(context, unit, prior)
            annotation = _checked_raw_annotation(
                context, unit, prior.get("game_review_gate")
            )
            if (
                annotation["run_id"] != run_id
                or annotation["arena_sha256"] != arena_hash
                or annotation["arena_provenance"] != provenance
                or annotation["competition_source"] != source_identity
            ):
                raise GateError(f"incompatible raw shard already exists: {output}")
            resumed += 1
            continue
        command = arena_command(context, unit, arena_path)
        process = subprocess.run(
            command,
            cwd=context.repository,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise GateError(
                f"arena failed for {unit.unit_id} with exit {process.returncode}: "
                f"{process.stderr.strip()}"
            )
        try:
            report = _object(json.loads(process.stdout), f"arena report {unit.unit_id}")
        except json.JSONDecodeError as error:
            raise GateError(f"arena returned invalid JSON for {unit.unit_id}") from error
        _validate_arena_report(context, unit, report)
        report["game_review_gate"] = {
            "manifest_sha256": context.manifest_sha256,
            "run_id": run_id,
            "unit_id": unit.unit_id,
            "phase": phase,
            "matchup_id": unit.matchup_id,
            "candidate_id": unit.candidate_id,
            "reference_id": unit.reference_id,
            "opening_depth": unit.opening_depth,
            "arena_sha256": arena_hash,
            "arena_provenance": provenance,
            "arena_command": command,
            "selection_sha256": selection_hash,
            "competition_source": source_identity,
        }
        _checked_raw_annotation(context, unit, report["game_review_gate"])
        write_json(output, report)
        completed += 1
    return {
        "phase": phase,
        "manifest_sha256": context.manifest_sha256,
        "run_id": run_id,
        "units_assigned": len(selected_units),
        "units_completed": completed,
        "units_resumed": resumed,
        "total_phase_units": len(units),
        "test_marker": str(marker) if marker is not None else None,
    }


def _expected_reference_config(reference: Mapping[str, Any]) -> dict[str, Any]:
    settings = reference["settings"]
    if reference["kind"] == "rank5-derived":
        return {
            "kind": "rank5-derived",
            "profile": "50k-demo",
            "max_turn_depth": 32,
            "max_nodes": 50_000,
            "transposition_table_entries": 65_536,
            "evaluation_cache_entries": 32_768,
            "max_time_ms": 0,
            "model_blend_percent": 0,
            "replay_corrections": False,
            "replay_book_enabled": False,
            "original_sha256": RANK5_SOURCE_SHA256,
        }
    return {
        "kind": "jacek-inspired",
        "max_turn_depth": settings["max_turn_depth"],
        "max_nodes": settings["max_nodes"],
        "transposition_table_entries": settings["transposition_entries"],
        "max_search_plies": settings["max_search_plies"],
        "model_sha256": settings["model_sha256"],
    }


def _validate_arena_report(
    context: ManifestContext, unit: GateUnit, report: Mapping[str, Any]
) -> None:
    if (
        report.get("schema") != "papersoccer.arena.v1"
        or report.get("mode") != "matches"
        or report.get("runtime") != "native"
    ):
        raise GateError(f"arena report contract mismatch in {unit.unit_id}")
    configuration = _object(report.get("configuration"), "arena configuration")
    expected_seed = str(
        _derived_seed(
            context.manifest["seeds"]["bot"][unit.phase],
            unit.matchup_id,
            str(unit.opening_depth),
        )
    )
    expected_common = {
        "rules": {"width": 8, "height": 10},
        "base_seed": expected_seed,
        "seed_pairs": unit.pairs,
        "games": unit.pairs * 2,
        "opening_plies": unit.opening_depth,
        "max_plies": 512,
        "bootstrap_samples": 1,
        "opening_generator": "frozen_uniform_legal_move_data_generation_bank",
        "opening_seed_derivation": "committed_bank_accepted_generation_seeds",
        "warmup": {
            "decisions_per_entrant": 8,
            "timed": False,
            "generation_plies": 24,
            "position_generator": "uniform_legal_move_generator",
            "seed_derivation": "domain_separated_splitmix64",
            "bot_instances": "separate_from_measured_games",
        },
    }
    for key, expected in expected_common.items():
        if configuration.get(key) != expected:
            raise GateError(f"arena configuration {key} mismatch in {unit.unit_id}")
    candidate = next(
        profile for profile in context.candidates if profile["id"] == unit.candidate_id
    )
    candidate_config = _object(configuration.get("candidate"), "candidate config")
    expected_candidate = {
        "kind": "deep-turn-search",
        "profile": f"deep-{candidate['max_nodes'] // 1000}k",
        "max_turn_depth": candidate["max_turn_depth"],
        "max_nodes": candidate["max_nodes"],
        "transposition_table_entries": candidate["transposition_entries"],
        "evaluation_cache_entries": candidate["evaluation_entries"],
        "max_time_ms": 0,
        "model_blend_percent": 0,
        "replay_corrections": False,
        "ranked_source_sha256": RANK5_SOURCE_SHA256,
    }
    for key, expected in expected_candidate.items():
        if candidate_config.get(key) != expected:
            raise GateError(f"DeepTurnSearch config {key} mismatch in {unit.unit_id}")
    reference = next(
        profile for profile in context.references if profile["id"] == unit.reference_id
    )
    actual_reference = _object(configuration.get("reference"), "reference config")
    expected_reference = _expected_reference_config(reference)
    for key, expected in expected_reference.items():
        if actual_reference.get(key) != expected:
            raise GateError(f"reference config {key} mismatch in {unit.unit_id}")
    games = _array(report.get("games"), "arena games")
    if len(games) != unit.pairs * 2:
        raise GateError(f"arena game count mismatch in {unit.unit_id}")
    pair_games: dict[int, set[int]] = defaultdict(set)
    for game in games:
        game = _object(game, "arena game")
        pair = _integer(game.get("pair_index"), "pair_index")
        game_in_pair = _integer(game.get("game_in_pair"), "game_in_pair")
        if pair >= unit.pairs or game_in_pair not in (0, 1):
            raise GateError(f"invalid pair/game index in {unit.unit_id}")
        if game_in_pair in pair_games[pair]:
            raise GateError(f"duplicate pair game in {unit.unit_id}")
        pair_games[pair].add(game_in_pair)
        outcome = _object(game.get("outcome"), "arena outcome")
        if outcome.get("truncated") is not False or outcome.get("winner") not in (
            "candidate",
            "reference",
        ):
            raise GateError(f"truncated or indecisive game in {unit.unit_id}")
        decisions = _array(game.get("decisions"), "arena decisions")
        if not decisions or any(
            _object(decision, "arena decision").get("legal") is not True
            for decision in decisions
        ):
            raise GateError(f"illegal or missing arena decision in {unit.unit_id}")
        _validate_complete_turn_sequences(
            game,
            unit,
            bot="candidate",
            diagnostics_field="deep_turn_search",
            expected_node_budget=next(
                profile["max_nodes"]
                for profile in context.candidates
                if profile["id"] == unit.candidate_id
            ),
        )
        if unit.reference_id == "rank5-derived-fixed-50k":
            _validate_complete_turn_sequences(
                game,
                unit,
                bot="reference",
                diagnostics_field="rank5_derived",
                expected_node_budget=50_000,
            )
    if len(pair_games) != unit.pairs or any(value != {0, 1} for value in pair_games.values()):
        raise GateError(f"incomplete color-swapped pair in {unit.unit_id}")
    summary = _object(report.get("summary"), "arena summary")
    if summary.get("illegal_moves") != 0 or summary.get("truncations") != 0:
        raise GateError(f"arena operational failure in {unit.unit_id}")


def _validate_complete_turn_sequences(
    game: Mapping[str, Any],
    unit: GateUnit,
    *,
    bot: str,
    diagnostics_field: str,
    expected_node_budget: int,
) -> None:
    decisions = _array(game.get("decisions"), "game decisions")
    index = 0
    while index < len(decisions):
        decision = _object(decisions[index], "game decision")
        if decision.get("bot") != bot:
            if decision.get(diagnostics_field) is not None:
                raise GateError(
                    f"unrelated decision contains {diagnostics_field} diagnostics"
                )
            index += 1
            continue
        stats = _object(
            decision.get(diagnostics_field), f"{diagnostics_field} diagnostics"
        )
        if stats.get("cached_continuation") is not False:
            raise GateError(f"orphan {diagnostics_field} cached continuation")
        if (
            stats.get("profile_node_budget") != expected_node_budget
            or stats.get("requested_nodes") != expected_node_budget
            or not 0 < _integer(stats.get("visited_nodes"), "visited nodes")
            <= expected_node_budget
        ):
            raise GateError(f"{diagnostics_field} fresh search work mismatch")
        planned = _integer(stats.get("planned_action_length"), "planned action length", 1)
        if stats.get("current_edge_index") != 0:
            raise GateError(f"fresh {diagnostics_field} action does not begin at edge zero")
        search_ordinal = _integer(
            stats.get("search_ordinal_in_game"), "search ordinal", 1
        )
        for edge_index in range(planned):
            cursor = index + edge_index
            if cursor >= len(decisions):
                raise GateError(f"incomplete {diagnostics_field} action in {unit.unit_id}")
            edge = _object(decisions[cursor], f"{diagnostics_field} action edge")
            edge_stats = _object(
                edge.get(diagnostics_field), f"{diagnostics_field} edge diagnostics"
            )
            if (
                edge.get("bot") != bot
                or edge.get("legal") is not True
                or edge_stats.get("profile_node_budget") != expected_node_budget
                or edge_stats.get("planned_action_length") != planned
                or edge_stats.get("current_edge_index") != edge_index
                or edge_stats.get("cached_continuation") is not (edge_index != 0)
                or edge_stats.get("search_ordinal_in_game") != search_ordinal
                or edge_stats.get("cached_moves_remaining") != planned - edge_index - 1
            ):
                raise GateError(
                    f"non-contiguous {diagnostics_field} action in {unit.unit_id}"
                )
            if edge_index != 0 and (
                edge_stats.get("requested_nodes") != 0
                or edge_stats.get("visited_nodes") != 0
            ):
                raise GateError(f"cached {diagnostics_field} edge reports new search work")
        index += planned


def nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise GateError("nearest-rank quantile requires values and probability in [0,1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def stratified_pair_bootstrap(
    strata: Mapping[int, Sequence[float]],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if resamples != BOOTSTRAP_RESAMPLES:
        raise GateError("frozen gate requires exactly 10,000 bootstrap resamples")
    if set(strata) != set(DEPTHS) or any(not values for values in strata.values()):
        raise GateError("bootstrap requires nonempty 4/8/12/20 strata")
    generator = random.Random(seed)
    ordered = [(depth, tuple(strata[depth])) for depth in DEPTHS]
    count = sum(len(values) for _, values in ordered)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _, values in ordered:
            total += sum(values[generator.randrange(len(values))] for _ in values)
        means.append(total / count)
    return {
        "method": "opening_depth_stratified_whole_pair_percentile",
        "seed": str(seed),
        "resamples": resamples,
        "confidence": 0.95,
        "pairs": count,
        "lower": nearest_rank(means, 0.025),
        "upper": nearest_rank(means, 0.975),
    }


def _pair_summary(
    scores_by_depth: Mapping[int, Sequence[float]], *, seed: int
) -> dict[str, Any]:
    scores = [score for depth in DEPTHS for score in scores_by_depth[depth]]
    if not scores:
        raise GateError("pair summary requires at least one whole-pair score")
    return {
        "pairs": len(scores),
        "games": len(scores) * 2,
        "pairs_won_2_0": sum(score == 1.0 for score in scores),
        "pairs_split_1_1": sum(score == 0.5 for score in scores),
        "pairs_lost_0_2": sum(score == 0.0 for score in scores),
        "mean_pair_score": sum(scores) / len(scores),
        "by_opening_depth": {
            str(depth): {
                "pairs": len(scores_by_depth[depth]),
                "mean_pair_score": sum(scores_by_depth[depth])
                / len(scores_by_depth[depth]),
            }
            for depth in DEPTHS
        },
        "pair_scores_by_opening_depth": {
            str(depth): list(scores_by_depth[depth]) for depth in DEPTHS
        },
        "pair_bootstrap_95": stratified_pair_bootstrap(
            scores_by_depth, seed=seed
        ),
    }


def _calibration_observation_payload(
    profile: Mapping[str, Any], observation_source: str
) -> dict[str, Any]:
    return {
        "profile_id": profile["id"],
        "profile_sha256": profile_sha256(profile),
        "fit_phase": "validation",
        "score_perspective": "player_who_made_possession",
        "observation_source": observation_source,
        "scores": [],
        "outcomes": [],
        "pair_ids": [],
        "opening_depths": [],
        "excluded_completed_depth_zero": 0,
    }


def _append_calibration_observation(
    payload: dict[str, Any],
    decision: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    eventual_win: bool,
    pair_id: str,
    opening_depth: int,
) -> None:
    if stats.get("cached_continuation") is True:
        return
    completed_depth = _integer(stats.get("completed_turn_depth"), "completed depth")
    if completed_depth == 0:
        payload["excluded_completed_depth_zero"] += 1
        return
    raw_score = _number(stats.get("root_score"), "root score")
    player = decision.get("player")
    if player == "two":
        raw_score = -raw_score
    elif player != "one":
        raise GateError("complete-turn decision has invalid player")
    payload["scores"].append(raw_score)
    payload["outcomes"].append(1 if eventual_win else 0)
    payload["pair_ids"].append(pair_id)
    payload["opening_depths"].append(opening_depth)


def aggregate_phase(context: ManifestContext, phase: str) -> dict[str, Any]:
    selection = _load_and_validate_selection(context) if phase == "test" else None
    units = units_for_phase(context, phase, selection)
    matchup_scores: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    calibration: dict[str, dict[str, Any]] = {
        profile["id"]: _calibration_observation_payload(
            profile, "fresh_deep_turn_search_candidate_decisions"
        )
        for profile in context.candidates
    }
    calibration[FAST_ANALYSIS_PROFILE["id"]] = _calibration_observation_payload(
        FAST_ANALYSIS_PROFILE, "fresh_fixed_rank5_reference_decisions"
    )
    raw_hashes: dict[str, str] = {}
    arena_hashes: set[str] = set()
    run_ids: set[str] = set()
    competition_sources: dict[str, dict[str, Any]] = {}
    total_games = total_decisions = incomplete_actions = 0
    for unit in units:
        path = _raw_root(context) / phase / "shards" / f"{unit.unit_id}.json"
        if not path.is_file():
            raise GateError(f"missing raw gate shard: {path}")
        report = _object(load_json(path), str(path))
        _validate_arena_report(context, unit, report)
        annotation = _checked_raw_annotation(
            context, unit, report.get("game_review_gate")
        )
        raw_hashes[unit.unit_id] = sha256_file(path)
        arena_hashes.add(_sha256(annotation.get("arena_sha256"), "arena hash"))
        run_ids.add(_string(annotation.get("run_id"), "run ID"))
        source_identity = _checked_competition_source_identity(
            context, annotation.get("competition_source")
        )
        competition_sources[source_identity["sha256"]] = source_identity
        games = _array(report.get("games"), "arena games")
        total_games += len(games)
        for game in games:
            game = _object(game, "arena game")
            pair_index = _integer(game.get("pair_index"), "pair index")
            outcome = _object(game.get("outcome"), "arena outcome")
            candidate_won = outcome.get("winner") == "candidate"
            pair_id = f"{unit.matchup_id}:{unit.opening_depth}:{pair_index}"
            scores = matchup_scores[unit.matchup_id][unit.opening_depth]
            while len(scores) <= pair_index:
                scores.append(0.0)
            scores[pair_index] += 0.5 if candidate_won else 0.0
            decisions = _array(game.get("decisions"), "arena decisions")
            total_decisions += len(decisions)
            if phase == "validation":
                for decision in decisions:
                    decision = _object(decision, "arena decision")
                    if decision.get("bot") == "candidate":
                        stats = _object(
                            decision.get("deep_turn_search"),
                            "DeepTurnSearch diagnostics",
                        )
                        _append_calibration_observation(
                            calibration[unit.candidate_id],
                            decision,
                            stats,
                            eventual_win=candidate_won,
                            pair_id=pair_id,
                            opening_depth=unit.opening_depth,
                        )
                    elif unit.reference_id == "rank5-derived-fixed-50k":
                        stats = _object(
                            decision.get("rank5_derived"),
                            "fixed Rank5Derived diagnostics",
                        )
                        _append_calibration_observation(
                            calibration[FAST_ANALYSIS_PROFILE["id"]],
                            decision,
                            stats,
                            eventual_win=not candidate_won,
                            pair_id=pair_id,
                            opening_depth=unit.opening_depth,
                        )
    if (len(arena_hashes) != 1 or len(run_ids) != 1 or
            len(competition_sources) != 1):
        raise GateError(
            f"{phase} shards mix arena, run, or competition-source identities"
        )
    matchups: dict[str, Any] = {}
    for unit in units:
        if unit.matchup_id in matchups:
            continue
        strata = matchup_scores[unit.matchup_id]
        expected_per_depth = unit.pairs
        if any(len(strata[depth]) != expected_per_depth for depth in DEPTHS):
            raise GateError(f"pair accounting mismatch for {unit.matchup_id}")
        seed = _derived_seed(
            context.manifest["seeds"]["bootstrap"][phase], unit.matchup_id
        )
        matchups[unit.matchup_id] = {
            "candidate_id": unit.candidate_id,
            "reference_id": unit.reference_id,
            **_pair_summary(strata, seed=seed),
        }
    candidate_strength: dict[str, Any] = {}
    candidate_ids = sorted({unit.candidate_id for unit in units})
    for candidate_id in candidate_ids:
        # Two binary games per opening pair/ref become pair scores when grouped
        # in matchup summaries. Aggregate candidate selection over those actual
        # color-swapped pair scores, not over individual game rows.
        pair_strata: dict[int, list[float]] = {depth: [] for depth in DEPTHS}
        for reference in context.references:
            matchup_id = f"{candidate_id}-vs-{reference['id']}"
            for depth in DEPTHS:
                pair_strata[depth].extend(matchup_scores[matchup_id][depth])
        seed = _derived_seed(
            context.manifest["seeds"]["bootstrap"][phase],
            candidate_id,
            "both-references",
        )
        candidate_strength[candidate_id] = _pair_summary(pair_strata, seed=seed)
    expected_games = sum(unit.pairs * 2 for unit in units)
    if total_games != expected_games:
        raise GateError(f"{phase} game count differs from frozen schedule")
    result = {
        "schema": PHASE_RESULT_SCHEMA,
        "phase": phase,
        "manifest_sha256": context.manifest_sha256,
        "source": {
            "raw_shard_sha256": dict(sorted(raw_hashes.items())),
            "arena_sha256": next(iter(arena_hashes)),
            "run_id": next(iter(run_ids)),
            "competition_source": next(iter(competition_sources.values())),
        },
        "completeness": {
            "expected_units": len(units),
            "completed_units": len(units),
            "expected_games": expected_games,
            "completed_games": total_games,
            "color_swapped_pairs": expected_games // 2,
            "decisions": total_decisions,
            "illegal_moves": 0,
            "incomplete_actions": incomplete_actions,
            "unexplained_truncations": 0,
            "parity_failures": 0,
            "operationally_valid": True,
        },
        "matchups": dict(sorted(matchups.items())),
        "candidate_strength": dict(sorted(candidate_strength.items())),
        "calibration_observations": (
            {identifier: calibration[identifier] for identifier in sorted(calibration)}
            if phase == "validation"
            else {}
        ),
    }
    path = _phase_result_path(context, phase)
    resumed = write_json(path, result)
    if phase == "test":
        marker = _raw_root(context) / "test" / "test-once.json"
        value = _object(load_json(marker), "test-once marker")
        if value.get("completed") is not True:
            value["completed"] = True
            value["completed_games"] = total_games
            write_json(marker, value, replace=True)
    return {
        "phase": phase,
        "result_path": str(path.relative_to(context.repository)),
        "result_sha256": sha256_file(path),
        "units": len(units),
        "games": total_games,
        "resumed": resumed,
    }


def _validate_phase_result(
    context: ManifestContext, phase: str, value: Any
) -> dict[str, Any]:
    result = _exact_keys(
        value,
        {
            "schema",
            "phase",
            "manifest_sha256",
            "source",
            "completeness",
            "matchups",
            "candidate_strength",
            "calibration_observations",
        },
        f"{phase} result",
    )
    if (
        result.get("schema") != PHASE_RESULT_SCHEMA
        or result.get("phase") != phase
        or result.get("manifest_sha256") != context.manifest_sha256
    ):
        raise GateError(f"{phase} result identity mismatch")

    selection = _load_and_validate_selection(context) if phase == "test" else None
    units = units_for_phase(context, phase, selection)
    expected_units = len(units)
    expected_games = sum(unit.pairs * 2 for unit in units)
    completeness = _exact_keys(
        result.get("completeness"),
        {
            "expected_units",
            "completed_units",
            "expected_games",
            "completed_games",
            "color_swapped_pairs",
            "decisions",
            "illegal_moves",
            "incomplete_actions",
            "unexplained_truncations",
            "parity_failures",
            "operationally_valid",
        },
        f"{phase} completeness",
    )
    required_zero = (
        "illegal_moves",
        "incomplete_actions",
        "unexplained_truncations",
        "parity_failures",
    )
    if (
        completeness.get("expected_units") != expected_units
        or completeness.get("completed_units") != expected_units
        or completeness.get("expected_games") != expected_games
        or completeness.get("completed_games") != expected_games
        or completeness.get("color_swapped_pairs") != expected_games // 2
        or completeness.get("operationally_valid") is not True
        or any(completeness.get(key) != 0 for key in required_zero)
    ):
        raise GateError(f"{phase} result is incomplete or operationally invalid")
    _integer(completeness.get("decisions"), f"{phase} decision count")

    source = _exact_keys(
        result.get("source"),
        {
            "raw_shard_sha256",
            "arena_sha256",
            "run_id",
            "competition_source",
        },
        f"{phase} source",
    )
    _checked_competition_source_identity(context, source["competition_source"])
    raw_hashes = _object(source["raw_shard_sha256"], f"{phase} raw shard hashes")
    expected_unit_ids = {unit.unit_id for unit in units}
    if set(raw_hashes) != expected_unit_ids:
        raise GateError(f"{phase} raw shard identity set changed")
    for unit_id, digest in raw_hashes.items():
        _sha256(digest, f"{phase} raw shard hash {unit_id}")
    _sha256(source["arena_sha256"], f"{phase} arena hash")
    run_id = _string(source["run_id"], f"{phase} run ID")
    if len(run_id) != 24 or any(character not in "0123456789abcdef" for character in run_id):
        raise GateError(f"{phase} run ID is invalid")

    summary_keys = {
        "pairs",
        "games",
        "pairs_won_2_0",
        "pairs_split_1_1",
        "pairs_lost_0_2",
        "mean_pair_score",
        "by_opening_depth",
        "pair_scores_by_opening_depth",
        "pair_bootstrap_95",
    }

    def checked_pair_summary(
        raw: Any,
        *,
        where: str,
        expected_pairs_per_depth: int,
        bootstrap_seed: int,
    ) -> dict[int, list[float]]:
        summary = _exact_keys(raw, summary_keys, where)
        raw_strata = _exact_keys(
            summary["pair_scores_by_opening_depth"],
            {str(depth) for depth in DEPTHS},
            f"{where}.pair_scores_by_opening_depth",
        )
        strata: dict[int, list[float]] = {}
        for depth in DEPTHS:
            values = [
                _number(score, f"{where} pair score")
                for score in _array(raw_strata[str(depth)], f"{where} d{depth} scores")
            ]
            if len(values) != expected_pairs_per_depth or any(
                score not in (0.0, 0.5, 1.0) for score in values
            ):
                raise GateError(f"{where} has invalid whole-pair scores")
            strata[depth] = values
        if summary != _pair_summary(strata, seed=bootstrap_seed):
            raise GateError(f"{where} summary/bootstrap is stale")
        return strata

    expected_matchups: dict[str, GateUnit] = {}
    for unit in units:
        expected_matchups.setdefault(unit.matchup_id, unit)
    matchups = _object(result["matchups"], f"{phase} matchups")
    if set(matchups) != set(expected_matchups):
        raise GateError(f"{phase} matchup identity set changed")
    matchup_strata: dict[str, dict[int, list[float]]] = {}
    for matchup_id, unit in expected_matchups.items():
        matchup = _exact_keys(
            matchups[matchup_id],
            {"candidate_id", "reference_id", *summary_keys},
            f"{phase} matchup {matchup_id}",
        )
        if (
            matchup["candidate_id"] != unit.candidate_id
            or matchup["reference_id"] != unit.reference_id
        ):
            raise GateError(f"{phase} matchup participants changed: {matchup_id}")
        seed = _derived_seed(
            context.manifest["seeds"]["bootstrap"][phase], matchup_id
        )
        matchup_strata[matchup_id] = checked_pair_summary(
            {key: matchup[key] for key in summary_keys},
            where=f"{phase} matchup {matchup_id}",
            expected_pairs_per_depth=unit.pairs,
            bootstrap_seed=seed,
        )

    expected_candidate_ids = sorted({unit.candidate_id for unit in units})
    candidate_strength = _object(
        result["candidate_strength"], f"{phase} candidate strength"
    )
    if set(candidate_strength) != set(expected_candidate_ids):
        raise GateError(f"{phase} candidate-strength identity set changed")
    for candidate_id in expected_candidate_ids:
        combined: dict[int, list[float]] = {depth: [] for depth in DEPTHS}
        for reference in context.references:
            matchup_id = f"{candidate_id}-vs-{reference['id']}"
            for depth in DEPTHS:
                combined[depth].extend(matchup_strata[matchup_id][depth])
        seed = _derived_seed(
            context.manifest["seeds"]["bootstrap"][phase],
            candidate_id,
            "both-references",
        )
        checked = checked_pair_summary(
            candidate_strength[candidate_id],
            where=f"{phase} candidate strength {candidate_id}",
            expected_pairs_per_depth=(
                next(unit.pairs for unit in units if unit.candidate_id == candidate_id)
                * len(context.references)
            ),
            bootstrap_seed=seed,
        )
        if checked != combined:
            raise GateError(f"{phase} candidate-strength pair ordering changed")

    calibration = _object(
        result["calibration_observations"], f"{phase} calibration observations"
    )
    if phase != "validation":
        if calibration:
            raise GateError(f"{phase} must not contain calibration observations")
        return result

    profiles_by_id: dict[str, Mapping[str, Any]] = {
        profile["id"]: profile for profile in context.candidates
    }
    profiles_by_id[FAST_ANALYSIS_PROFILE["id"]] = FAST_ANALYSIS_PROFILE
    if set(calibration) != set(profiles_by_id):
        raise GateError("validation calibration profile set changed")
    for profile_id, profile in profiles_by_id.items():
        observations = _exact_keys(
            calibration[profile_id],
            {
                "profile_id",
                "profile_sha256",
                "fit_phase",
                "score_perspective",
                "observation_source",
                "scores",
                "outcomes",
                "pair_ids",
                "opening_depths",
                "excluded_completed_depth_zero",
            },
            f"validation calibration {profile_id}",
        )
        expected_source = (
            "fresh_fixed_rank5_reference_decisions"
            if profile_id == FAST_ANALYSIS_PROFILE["id"]
            else "fresh_deep_turn_search_candidate_decisions"
        )
        if (
            observations["profile_id"] != profile_id
            or observations["profile_sha256"] != profile_sha256(profile)
            or observations["fit_phase"] != "validation"
            or observations["score_perspective"] != "player_who_made_possession"
            or observations["observation_source"] != expected_source
        ):
            raise GateError(f"validation calibration identity changed: {profile_id}")
        scores = [
            _number(score, f"{profile_id} calibration score")
            for score in _array(observations["scores"], f"{profile_id} scores")
        ]
        outcomes = _array(observations["outcomes"], f"{profile_id} outcomes")
        pair_ids = _array(observations["pair_ids"], f"{profile_id} pair IDs")
        opening_depths = _array(
            observations["opening_depths"], f"{profile_id} opening depths"
        )
        if not scores or not (
            len(scores) == len(outcomes) == len(pair_ids) == len(opening_depths)
        ):
            raise GateError(f"{profile_id} calibration observations are unaligned")
        _integer(
            observations["excluded_completed_depth_zero"],
            f"{profile_id} excluded zero-depth searches",
        )
        valid_matchups = {
            unit.matchup_id
            for unit in units
            if (
                unit.reference_id == "rank5-derived-fixed-50k"
                if profile_id == FAST_ANALYSIS_PROFILE["id"]
                else unit.candidate_id == profile_id
            )
        }
        pairs_by_matchup_depth = {
            (unit.matchup_id, unit.opening_depth): unit.pairs for unit in units
        }
        for index, (outcome, pair_id, depth) in enumerate(
            zip(outcomes, pair_ids, opening_depths, strict=True)
        ):
            if isinstance(outcome, bool) or outcome not in (0, 1):
                raise GateError(f"{profile_id} calibration outcome {index} is invalid")
            if isinstance(depth, bool) or depth not in DEPTHS:
                raise GateError(f"{profile_id} calibration depth {index} is invalid")
            pair_text = _string(pair_id, f"{profile_id} calibration pair ID")
            try:
                matchup_id, encoded_depth, encoded_index = pair_text.rsplit(":", 2)
                pair_index = int(encoded_index)
                pair_depth = int(encoded_depth)
            except (ValueError, TypeError) as error:
                raise GateError(f"{profile_id} calibration pair ID is invalid") from error
            pair_limit = pairs_by_matchup_depth.get((matchup_id, depth))
            if (
                matchup_id not in valid_matchups
                or pair_depth != depth
                or pair_limit is None
                or pair_index < 0
                or pair_index >= pair_limit
            ):
                raise GateError(f"{profile_id} calibration pair ID is out of schedule")
    return result


_PARITY_DIAGNOSTIC_FIELDS = {
    "completed_turn_depth": "completed_turn_depth",
    "attempted_turn_depth": "attempted_turn_depth",
    "nodes": "visited_nodes",
    "leaf_evaluations": "leaf_evaluations",
    "terminal_nodes": "terminal_nodes",
    "completed_actions": "completed_actions",
    "cutoffs": "cutoffs",
    "transposition_probes": "transposition_probes",
    "transposition_hits": "transposition_hits",
    "transposition_cutoffs": "transposition_cutoffs",
    "transposition_stores": "transposition_stores",
    "continuation_transposition_hits": "continuation_transposition_hits",
    "evaluation_cache_probes": "evaluation_cache_probes",
    "evaluation_cache_hits": "evaluation_cache_hits",
    "terminal_bound_cutoffs": "terminal_bound_cutoffs",
    "forced_edges": "forced_edges",
    "root_seed_actions": "root_seed_actions",
    "root_transposition_reuses": "root_transposition_reuses",
    "max_action_edges": "max_action_edges",
    "root_score": "root_score",
    "budget_exhausted": "budget_exhausted",
}


def _checked_parity_transcript(value: Any, where: str) -> dict[str, Any]:
    transcript = _exact_keys(
        value, {"action", "root_score", "diagnostics"}, where
    )
    action: list[dict[str, int]] = []
    for index, point_value in enumerate(_array(transcript["action"], f"{where}.action")):
        point = _exact_keys(point_value, {"x", "y"}, f"{where}.action[{index}]")
        if any(
            isinstance(point[key], bool) or not isinstance(point[key], int)
            for key in ("x", "y")
        ):
            raise GateError(f"{where} action coordinates are invalid")
        action.append({"x": point["x"], "y": point["y"]})
    if not action:
        raise GateError(f"{where} action is incomplete")
    root_score = transcript["root_score"]
    if isinstance(root_score, bool) or not isinstance(root_score, int):
        raise GateError(f"{where} root score is invalid")
    diagnostics = _exact_keys(
        transcript["diagnostics"], set(_PARITY_DIAGNOSTIC_FIELDS), f"{where}.diagnostics"
    )
    checked_diagnostics: dict[str, Any] = {}
    for key, raw in diagnostics.items():
        if key == "budget_exhausted":
            checked_diagnostics[key] = _boolean(raw, f"{where}.{key}")
        elif key == "root_score":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise GateError(f"{where}.{key} is invalid")
            checked_diagnostics[key] = raw
        else:
            checked_diagnostics[key] = _integer(raw, f"{where}.{key}")
    if checked_diagnostics["root_score"] != root_score:
        raise GateError(f"{where} root score disagrees with diagnostics")
    return {
        "action": action,
        "root_score": root_score,
        "diagnostics": checked_diagnostics,
    }


def _native_parity_transcript(
    report: Mapping[str, Any],
    opening: OpeningRecord,
    pair_index: int,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    reported_opening = next(
        (
            _object(item, "native parity opening")
            for item in _array(report.get("openings"), "native parity openings")
            if _object(item, "native parity opening").get("pair_index") == pair_index
        ),
        None,
    )
    if (
        reported_opening is None
        or reported_opening.get("opening_id") != opening.opening_id
        or reported_opening.get("state_hash") != opening.state_hash
        or reported_opening.get("actual_plies") != opening.depth
    ):
        raise GateError(f"native parity opening mismatch: {opening.opening_id}")
    state = _object(reported_opening.get("state"), "native parity opening state")
    if state.get("to_move") != opening.to_move:
        raise GateError(f"native parity player mismatch: {opening.opening_id}")
    player_field = "player_one" if state.get("to_move") == "one" else "player_two"
    games = _array(report.get("games"), "native parity games")
    game = next(
        (
            _object(item, "native parity game")
            for item in games
            if _object(item, "native parity game").get("pair_index") == pair_index
            and _object(
                _object(item, "native parity game").get(player_field),
                f"native parity game {player_field}",
            ).get("bot")
            == "candidate"
        ),
        None,
    )
    if game is None:
        raise GateError(f"native candidate-to-move game is missing: {opening.opening_id}")
    decisions = _array(game.get("decisions"), "native parity decisions")
    if not decisions:
        raise GateError(f"native parity decision is missing: {opening.opening_id}")
    first = _object(decisions[0], "native parity first decision")
    stats = _object(first.get("deep_turn_search"), "native parity diagnostics")
    if (
        first.get("bot") != "candidate"
        or first.get("ply") != opening.depth + 1
        or stats.get("cached_continuation") is not False
        or stats.get("current_edge_index") != 0
        or stats.get("profile_node_budget") != profile["max_nodes"]
    ):
        raise GateError(f"native parity search is not fresh: {opening.opening_id}")
    action_length = _integer(
        stats.get("planned_action_length"), "native planned action length", 1
    )
    action_decisions = decisions[:action_length]
    if len(action_decisions) != action_length:
        raise GateError(f"native parity action is incomplete: {opening.opening_id}")
    action: list[dict[str, int]] = []
    for edge_index, edge_value in enumerate(action_decisions):
        edge = _object(edge_value, "native parity action edge")
        edge_stats = _object(edge.get("deep_turn_search"), "native parity edge stats")
        if (
            edge.get("bot") != "candidate"
            or edge.get("legal") is not True
            or edge_stats.get("current_edge_index") != edge_index
            or edge_stats.get("planned_action_length") != action_length
        ):
            raise GateError(f"native parity action is non-contiguous: {opening.opening_id}")
        point = _exact_keys(edge.get("to"), {"x", "y"}, "native action endpoint")
        action.append({"x": point["x"], "y": point["y"]})
    normalized = {
        "action": action,
        "root_score": stats.get("root_score"),
        "diagnostics": {
            normalized_key: stats.get(native_key)
            for normalized_key, native_key in _PARITY_DIAGNOSTIC_FIELDS.items()
        },
    }
    return _checked_parity_transcript(normalized, "native parity transcript")


def validate_latency_value(
    context: ManifestContext,
    value: Any,
    *,
    verify_module: bool = True,
    verify_native_sources: bool = True,
) -> dict[str, Any]:
    latency = _exact_keys(
        value,
        {
            "schema",
            "manifest_sha256",
            "competition_source",
            "module",
            "environment",
            "sample_source",
            "profiles",
        },
        "Wasm latency",
    )
    if latency["schema"] != LATENCY_SCHEMA or latency[
        "manifest_sha256"
    ] != context.manifest_sha256:
        raise GateError("Wasm latency belongs to another manifest")
    latency_source_identity = _checked_competition_source_identity(
        context, latency["competition_source"]
    )
    module = _exact_keys(
        latency["module"],
        {
            "path",
            "sha256",
            "emscripten_version",
            "initial_memory_bytes",
            "memory_growth",
        },
        "latency.module",
    )
    _sha256(module["sha256"], "latency.module.sha256")
    if (
        module["path"] != "web/papersoccer-analysis-wasm.js"
        or
        module["emscripten_version"] != "6.0.2"
        or module["initial_memory_bytes"] != 67_108_864
        or module["memory_growth"] is not False
    ):
        raise GateError("latency module does not use the frozen analysis Wasm contract")
    if verify_module:
        module_path = _repo_path(context.repository, module["path"], "latency module path")
        if sha256_file(module_path) != module["sha256"]:
            raise GateError("latency module SHA-256 mismatch")
    environment = _exact_keys(
        latency["environment"],
        {
            "runtime",
            "node_version",
            "v8_version",
            "platform",
            "architecture",
            "cpu_model",
            "logical_cpus",
            "total_memory_bytes",
            "timer",
            "warmup_searches_per_candidate",
        },
        "latency.environment",
    )
    if (
        environment["runtime"] != "node-webassembly"
        or environment["timer"] != "performance.now"
        or environment["warmup_searches_per_candidate"] != 8
    ):
        raise GateError("Wasm latency environment protocol changed")
    for key in (
        "node_version",
        "v8_version",
        "platform",
        "architecture",
        "cpu_model",
    ):
        _string(environment[key], f"latency.environment.{key}")
    _integer(environment["logical_cpus"], "latency logical CPUs", 1)
    _integer(environment["total_memory_bytes"], "latency total memory", 1)
    source = _exact_keys(
        latency["sample_source"],
        {
            "phase",
            "opening_depths",
            "fresh_possession_boundaries_only",
            "samples_per_opening_depth_per_candidate",
            "native_parity_reference",
            "native_raw_root",
        },
        "latency.sample_source",
    )
    expected_native_root = str(
        (_raw_root(context) / "validation" / "shards").relative_to(
            context.repository
        )
    )
    if source != {
        "phase": "validation",
        "opening_depths": list(DEPTHS),
        "fresh_possession_boundaries_only": True,
        "samples_per_opening_depth_per_candidate": 20,
        "native_parity_reference": "rank5-derived-fixed-50k",
        "native_raw_root": expected_native_root,
    }:
        raise GateError("Wasm latency does not use the frozen validation sample source")
    profiles = _object(latency["profiles"], "latency.profiles")
    expected_ids = {profile["id"] for profile in context.candidates}
    if set(profiles) != expected_ids:
        raise GateError("Wasm latency must contain every candidate exactly once")
    minimum_samples = context.manifest["latency_protocol"][
        "minimum_samples_per_candidate"
    ]
    validation_openings = {
        record.opening_id: (bank.depth, index, record)
        for bank in context.gate_banks
        if bank.phase == "validation"
        for index, record in enumerate(bank.records)
    }
    validation_raw_hashes: Mapping[str, Any] | None = None
    if verify_native_sources:
        validation_path = _phase_result_path(context, "validation")
        if not validation_path.is_file():
            raise GateError("validation result is required to verify Wasm/native parity")
        validation_result = _validate_phase_result(
            context, "validation", load_json(validation_path)
        )
        validation_raw_hashes = _object(
            validation_result["source"]["raw_shard_sha256"],
            "validation raw shard hashes",
        )
    shared_sample_ids: list[str] | None = None
    for profile in context.candidates:
        result = _exact_keys(
            profiles[profile["id"]],
            {
                "profile_sha256",
                "node_budget",
                "sample_ids",
                "opening_depths",
                "samples_ms",
                "timing_ms",
                "native_shard_sha256",
                "parity",
                "operational_counts",
            },
            f"latency profile {profile['id']}",
        )
        if result["profile_sha256"] != profile_sha256(profile) or result[
            "node_budget"
        ] != profile["max_nodes"]:
            raise GateError(f"Wasm latency profile identity mismatch: {profile['id']}")
        samples = [
            _number(sample, f"{profile['id']} latency sample")
            for sample in _array(result["samples_ms"], "latency samples")
        ]
        sample_ids = [
            _string(sample_id, f"{profile['id']} latency sample ID")
            for sample_id in _array(result["sample_ids"], "latency sample IDs")
        ]
        opening_depths = _array(
            result["opening_depths"], "latency sample opening depths"
        )
        if (
            len(samples) < minimum_samples
            or len(samples) != len(sample_ids)
            or len(samples) != len(opening_depths)
            or len(set(sample_ids)) != len(sample_ids)
            or any(sample < 0.0 for sample in samples)
        ):
            raise GateError(f"too few or negative latency samples: {profile['id']}")
        depth_counts = {depth: 0 for depth in DEPTHS}
        for sample_id, depth in zip(sample_ids, opening_depths, strict=True):
            if isinstance(depth, bool) or depth not in DEPTHS:
                raise GateError(f"invalid latency opening depth: {profile['id']}")
            location = validation_openings.get(sample_id)
            if location is None or location[0] != depth:
                raise GateError(f"latency sample is not a frozen validation opening")
            depth_counts[depth] += 1
        if depth_counts != {depth: 20 for depth in DEPTHS}:
            raise GateError("latency samples are not equally stratified by opening depth")
        if shared_sample_ids is None:
            shared_sample_ids = sample_ids
        elif sample_ids != shared_sample_ids:
            raise GateError("Wasm candidates were not timed on identical validation openings")
        native_hashes = _exact_keys(
            result["native_shard_sha256"],
            {str(depth) for depth in DEPTHS},
            f"{profile['id']} native shard hashes",
        )
        native_reports: dict[int, Mapping[str, Any]] = {}
        for depth in DEPTHS:
            digest = _sha256(
                native_hashes[str(depth)], f"{profile['id']} d{depth} native hash"
            )
            if validation_raw_hashes is not None:
                unit_id = (
                    f"validation--{profile['id']}-vs-"
                    f"rank5-derived-fixed-50k--d{depth:02d}"
                )
                if validation_raw_hashes.get(unit_id) != digest:
                    raise GateError(f"Wasm parity source is stale: {unit_id}")
                raw_path = _raw_root(context) / "validation" / "shards" / f"{unit_id}.json"
                if sha256_file(raw_path) != digest:
                    raise GateError(f"native parity shard hash mismatch: {unit_id}")
                report = _object(load_json(raw_path), f"native parity shard {unit_id}")
                unit = next(
                    item
                    for item in units_for_phase(context, "validation")
                    if item.unit_id == unit_id
                )
                _validate_arena_report(context, unit, report)
                native_reports[depth] = report
        parity_entries = _array(result["parity"], f"{profile['id']} parity entries")
        if len(parity_entries) != len(sample_ids):
            raise GateError(f"{profile['id']} parity entry count changed")
        for index, (entry_value, sample_id) in enumerate(
            zip(parity_entries, sample_ids, strict=True)
        ):
            entry = _exact_keys(
                entry_value,
                {
                    "sample_id",
                    "action_sha256",
                    "transcript_sha256",
                    "wasm_transcript",
                },
                f"{profile['id']} parity[{index}]",
            )
            if entry["sample_id"] != sample_id:
                raise GateError(f"{profile['id']} parity sample order changed")
            transcript = _checked_parity_transcript(
                entry["wasm_transcript"], f"{profile['id']} parity[{index}]"
            )
            action_hash = sha256_bytes(canonical_json_bytes(transcript["action"]))
            transcript_hash = sha256_bytes(canonical_json_bytes(transcript))
            if (
                entry["action_sha256"] != action_hash
                or entry["transcript_sha256"] != transcript_hash
            ):
                raise GateError(f"{profile['id']} parity identity is stale")
            if verify_native_sources:
                depth, pair_index, opening = validation_openings[sample_id]
                native = _native_parity_transcript(
                    native_reports[depth], opening, pair_index, profile
                )
                if native != transcript:
                    raise GateError(
                        f"native/Wasm parity transcript differs: {profile['id']}/{sample_id}"
                    )
        timing = _exact_keys(
            result["timing_ms"], {"samples", "median", "p95", "maximum"}, "timing"
        )
        expected_timing = {
            "samples": len(samples),
            "median": nearest_rank(samples, 0.5),
            "p95": nearest_rank(samples, 0.95),
            "maximum": max(samples),
        }
        for key, expected in expected_timing.items():
            actual = timing.get(key)
            if actual != expected:
                raise GateError(f"Wasm latency {key} is stale for {profile['id']}")
        operational = _exact_keys(
            result["operational_counts"],
            {
                "illegal_moves",
                "incomplete_actions",
                "unexplained_truncations",
                "parity_failures",
            },
            "latency operational counts",
        )
        if any(operational[key] != 0 for key in operational):
            raise GateError(f"Wasm operational failure for {profile['id']}")
    return latency


def record_latency(
    context: ManifestContext, input_path: pathlib.Path
) -> dict[str, Any]:
    value = validate_latency_value(context, load_json(input_path))
    output = _repo_path(
        context.repository, context.manifest["outputs"]["wasm_latency"], "Wasm latency"
    )
    resumed = write_json(output, value)
    return {
        "latency_path": str(output.relative_to(context.repository)),
        "latency_sha256": sha256_file(output),
        "resumed": resumed,
    }


def fit_logistic_calibration(
    *,
    profile: Mapping[str, Any],
    scores: Sequence[float],
    outcomes: Sequence[int],
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> dict[str, Any]:
    if len(scores) != len(outcomes) or len(scores) < 3:
        raise GateError("calibration needs at least three aligned observations")
    checked_scores = [_number(score, "calibration score") for score in scores]
    if any(outcome not in (0, 1) for outcome in outcomes) or len(set(outcomes)) != 2:
        raise GateError("calibration outcomes must contain both binary classes")
    mean = sum(checked_scores) / len(checked_scores)
    scale = math.sqrt(
        sum((score - mean) ** 2 for score in checked_scores) / len(checked_scores)
    )
    if scale <= 1e-15:
        raise GateError("calibration scores have zero variance")
    standardized = [(score - mean) / scale for score in checked_scores]
    zeros = [score for score, outcome in zip(standardized, outcomes, strict=True) if not outcome]
    ones = [score for score, outcome in zip(standardized, outcomes, strict=True) if outcome]
    if max(zeros) <= min(ones) or max(ones) <= min(zeros):
        raise GateError("calibration scores completely separate outcomes")
    prevalence = sum(outcomes) / len(outcomes)
    intercept = math.log(prevalence / (1.0 - prevalence))
    slope = 0.0

    def log_sigmoid(value: float) -> float:
        return -math.log1p(math.exp(-value)) if value >= 0 else value - math.log1p(math.exp(value))

    def likelihood(a: float, b: float) -> float:
        return sum(
            log_sigmoid(a + b * score) if outcome else log_sigmoid(-(a + b * score))
            for score, outcome in zip(standardized, outcomes, strict=True)
        )

    for iteration in range(max_iterations + 1):
        gradient_a = gradient_b = info_aa = info_ab = info_bb = 0.0
        for score, outcome in zip(standardized, outcomes, strict=True):
            linear = intercept + slope * score
            probability = (
                1.0 / (1.0 + math.exp(-linear))
                if linear >= 0
                else math.exp(linear) / (1.0 + math.exp(linear))
            )
            residual = outcome - probability
            weight = probability * (1.0 - probability)
            gradient_a += residual
            gradient_b += residual * score
            info_aa += weight
            info_ab += weight * score
            info_bb += weight * score * score
        if max(abs(gradient_a), abs(gradient_b)) <= tolerance:
            if slope <= 0.0:
                raise GateError(
                    "calibration slope is not positive after score orientation"
                )
            return {
                "schema": CALIBRATION_SCHEMA,
                "calibration_id": f"{profile['id']}-validation-logistic-v1",
                "profile_id": profile["id"],
                "profile_sha256": profile_sha256(profile),
                "fit_phase": "validation",
                "score_perspective": "player_who_made_possession",
                "link": "standardized_logistic",
                "score_mean": mean,
                "score_scale": scale,
                "intercept": intercept,
                "slope": slope,
                "sample_count": len(scores),
                "iterations": iteration,
            }
        if iteration == max_iterations:
            break
        determinant = info_aa * info_bb - info_ab * info_ab
        if determinant <= 1e-15:
            raise GateError("calibration information matrix is singular")
        step_a = (info_bb * gradient_a - info_ab * gradient_b) / determinant
        step_b = (-info_ab * gradient_a + info_aa * gradient_b) / determinant
        current = likelihood(intercept, slope)
        multiplier = 1.0
        for _ in range(60):
            candidate_a = intercept + multiplier * step_a
            candidate_b = slope + multiplier * step_b
            if likelihood(candidate_a, candidate_b) >= current - 1e-12:
                intercept, slope = candidate_a, candidate_b
                break
            multiplier *= 0.5
        else:
            raise GateError("calibration Newton step did not improve likelihood")
    raise GateError("calibration did not converge")


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    validation_strength: Mapping[str, float],
    latency_profiles: Mapping[str, Mapping[str, Any]],
    *,
    p95_limit_ms: float = 400.0,
    maximum_limit_ms: float = 750.0,
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for profile in candidates:
        identifier = profile["id"]
        timing = _object(latency_profiles[identifier]["timing_ms"], "candidate timing")
        strength = _number(validation_strength[identifier], "validation strength")
        p95 = _number(timing["p95"], "Wasm p95")
        maximum = _number(timing["maximum"], "Wasm maximum")
        rows.append(
            {
                "profile_id": identifier,
                "profile_sha256": profile_sha256(profile),
                "max_nodes": profile["max_nodes"],
                "validation_mean_pair_score": strength,
                "wasm_p95_ms": p95,
                "wasm_maximum_ms": maximum,
                "latency_eligible": p95 <= p95_limit_ms and maximum <= maximum_limit_ms,
            }
        )
    leader = max(row["validation_mean_pair_score"] for row in rows)
    strength_band = [
        row
        for row in rows
        if row["validation_mean_pair_score"] >= leader - 0.01
    ]
    eligible = [row for row in strength_band if row["latency_eligible"]]
    if not eligible:
        raise GateError(
            "no DeepTurnSearch candidate in the validation leader band meets "
            "both Wasm latency limits"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["wasm_p95_ms"],
            row["max_nodes"],
            row["profile_id"],
        ),
    )
    for row in rows:
        row["within_one_percentage_point_of_leader"] = row in strength_band
        row["selected"] = row is selected
    return selected["profile_id"], rows


def create_selection_value(
    context: ManifestContext,
    *,
    latency_value: Any | None = None,
    latency_sha256: str | None = None,
) -> dict[str, Any]:
    development_path = _phase_result_path(context, "development")
    validation_path = _phase_result_path(context, "validation")
    development = _validate_phase_result(
        context, "development", load_json(development_path)
    )
    validation = _validate_phase_result(context, "validation", load_json(validation_path))
    if latency_value is None:
        if latency_sha256 is not None:
            raise GateError("latency SHA-256 was provided without a latency value")
        latency_path = _repo_path(
            context.repository,
            context.manifest["outputs"]["wasm_latency"],
            "Wasm latency",
        )
        latency = validate_latency_value(context, load_json(latency_path))
        latency_input_sha256 = sha256_file(latency_path)
    else:
        latency = validate_latency_value(context, latency_value)
        latency_input_sha256 = _sha256(
            latency_sha256, "draft latency input SHA-256"
        )
    source_identities = {
        canonical_json_bytes(development["source"]["competition_source"]),
        canonical_json_bytes(validation["source"]["competition_source"]),
        canonical_json_bytes(latency["competition_source"]),
    }
    if len(source_identities) != 1:
        raise GateError(
            "development, validation, and Wasm latency use different competition sources"
        )
    selection_source_identity = _checked_competition_source_identity(
        context, latency["competition_source"]
    )
    strengths = {
        profile["id"]: _number(
            _object(
                validation["candidate_strength"].get(profile["id"]),
                f"validation strength {profile['id']}",
            ).get("mean_pair_score"),
            f"validation strength {profile['id']}",
        )
        for profile in context.candidates
    }
    selected_id, rows = select_candidate(
        context.candidates,
        strengths,
        latency["profiles"],
        p95_limit_ms=context.manifest["latency_protocol"]["p95_limit_ms"],
        maximum_limit_ms=context.manifest["latency_protocol"]["maximum_limit_ms"],
    )
    selected = next(profile for profile in context.candidates if profile["id"] == selected_id)
    deep_observations = _object(
        validation["calibration_observations"].get(selected_id),
        "selected validation calibration observations",
    )
    deep_calibration = fit_logistic_calibration(
        profile=selected,
        scores=_array(deep_observations.get("scores"), "Deep calibration scores"),
        outcomes=_array(
            deep_observations.get("outcomes"), "Deep calibration outcomes"
        ),
    )
    deep_calibration["observation_source"] = deep_observations[
        "observation_source"
    ]
    deep_calibration["excluded_completed_depth_zero"] = _integer(
        deep_observations.get("excluded_completed_depth_zero"),
        "excluded zero-depth Deep calibration searches",
    )
    fast_id = FAST_ANALYSIS_PROFILE["id"]
    fast_observations = _object(
        validation["calibration_observations"].get(fast_id),
        "Fast validation calibration observations",
    )
    fast_calibration = fit_logistic_calibration(
        profile=FAST_ANALYSIS_PROFILE,
        scores=_array(fast_observations.get("scores"), "Fast calibration scores"),
        outcomes=_array(fast_observations.get("outcomes"), "Fast calibration outcomes"),
    )
    fast_calibration["observation_source"] = fast_observations[
        "observation_source"
    ]
    fast_calibration["excluded_completed_depth_zero"] = _integer(
        fast_observations.get("excluded_completed_depth_zero"),
        "excluded zero-depth Fast calibration searches",
    )
    return {
        "schema": SELECTION_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "competition_source": selection_source_identity,
        "source_phase": "validation",
        "input_sha256": {
            "development": sha256_file(development_path),
            "validation": sha256_file(validation_path),
            "wasm_latency": latency_input_sha256,
        },
        "selection_rule": context.manifest["selection_rule"],
        "candidate_metrics": rows,
        "selected_profile_id": selected_id,
        "selected_profile_sha256": profile_sha256(selected),
        "review_mode_calibration_profiles": {
            "Fast": fast_id,
            "Deep": selected_id,
        },
        "calibration_mappings": {
            fast_id: fast_calibration,
            selected_id: deep_calibration,
        },
        "test_authorized": True,
    }


def lock_selection(context: ManifestContext) -> dict[str, Any]:
    value = create_selection_value(context)
    path = _selection_path(context)
    resumed = write_json(path, value)
    return {
        "selection_lock": str(path.relative_to(context.repository)),
        "selection_sha256": sha256_file(path),
        "selected_profile_id": value["selected_profile_id"],
        "resumed": resumed,
    }


def _cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _raw_logistic_coefficients(mapping: Mapping[str, Any]) -> tuple[float, float]:
    mean = _number(mapping.get("score_mean"), "calibration score mean")
    scale = _number(mapping.get("score_scale"), "calibration score scale")
    intercept = _number(mapping.get("intercept"), "calibration intercept")
    slope = _number(mapping.get("slope"), "calibration slope")
    if scale <= 0.0 or slope <= 0.0:
        raise GateError("calibration cannot be converted to a positive raw-score mapping")
    coefficient = slope / scale
    return intercept - coefficient * mean, coefficient


def render_cpp_lock_bytes(
    context: ManifestContext, selection: Mapping[str, Any] | None = None
) -> bytes:
    if selection is None:
        selection = _load_and_validate_selection(context)
    roles = _exact_keys(
        selection.get("review_mode_calibration_profiles"),
        {"Fast", "Deep"},
        "review-mode calibration profiles",
    )
    mappings = _object(
        selection.get("calibration_mappings"), "calibration mappings"
    )
    fast_id = _string(roles["Fast"], "Fast calibration profile")
    deep_id = _string(roles["Deep"], "Deep calibration profile")
    if fast_id == deep_id or fast_id != FAST_ANALYSIS_PROFILE["id"]:
        raise GateError("Fast and Deep calibration identities are not distinct")
    fast_mapping = _object(mappings.get(fast_id), "Fast calibration mapping")
    deep_mapping = _object(mappings.get(deep_id), "Deep calibration mapping")
    selected = next(
        (profile for profile in context.candidates if profile["id"] == deep_id), None
    )
    if selected is None:
        raise GateError("Deep calibration does not name the selected candidate")
    fast_intercept, fast_coefficient = _raw_logistic_coefficients(fast_mapping)
    deep_intercept, deep_coefficient = _raw_logistic_coefficients(deep_mapping)

    def mapping_lines(
        name: str,
        evidence_profile_id: str,
        search_profile_name: str,
        mapping: Mapping[str, Any],
        raw_intercept: float,
        raw_coefficient: float,
    ) -> list[str]:
        calibration_id = _string(mapping.get("calibration_id"), "calibration ID")
        profile_hash = _sha256(mapping.get("profile_sha256"), "profile hash")
        mapping_hash = sha256_bytes(canonical_json_bytes(mapping))
        return [
            f"inline constexpr Calibration {name}{{",
            f"    {_cpp_string(calibration_id)},",
            f"    {_cpp_string(evidence_profile_id)},",
            f"    {_cpp_string(search_profile_name)},",
            f"    {_cpp_string(profile_hash)},",
            f"    {_cpp_string(mapping_hash)},",
            f"    {format(raw_intercept, '.17g')},",
            f"    {format(raw_coefficient, '.17g')},",
            "};",
        ]

    deep_search_profile = f"deep-{selected['max_nodes'] // 1000}k"
    lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace papersoccer::game_review_lock {",
        "",
        "struct Calibration {",
        "  const char *identity;",
        "  const char *evidence_profile_id;",
        "  const char *search_profile_name;",
        "  const char *profile_sha256;",
        "  const char *mapping_sha256;",
        "  double raw_intercept;",
        "  double raw_score_coefficient;",
        "};",
        "",
        f"inline constexpr std::uint64_t selected_deep_nodes = {selected['max_nodes']}ULL;",
        "",
        *mapping_lines(
            "fast_calibration",
            fast_id,
            "fast-50k",
            fast_mapping,
            fast_intercept,
            fast_coefficient,
        ),
        "",
        *mapping_lines(
            "deep_calibration",
            deep_id,
            deep_search_profile,
            deep_mapping,
            deep_intercept,
            deep_coefficient,
        ),
        "",
        "}  // namespace papersoccer::game_review_lock",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _selection_from_latency_input(
    context: ManifestContext, latency_input: pathlib.Path
) -> dict[str, Any]:
    return create_selection_value(
        context,
        latency_value=load_json(latency_input),
        latency_sha256=sha256_file(latency_input),
    )


def render_cpp_lock(
    context: ManifestContext, latency_input: pathlib.Path | None = None
) -> dict[str, Any]:
    path = _repo_path(
        context.repository,
        context.manifest["outputs"]["cpp_lock_header"],
        "C++ calibration lock",
    )
    if latency_input is not None and _selection_path(context).exists():
        raise GateError(
            "draft lock rendering is disabled after the official selection lock exists"
        )
    selection = (
        _selection_from_latency_input(context, latency_input)
        if latency_input is not None
        else None
    )
    resumed = _atomic_write(
        path,
        render_cpp_lock_bytes(context, selection),
        replace=latency_input is not None,
    )
    return {
        "cpp_lock_header": str(path.relative_to(context.repository)),
        "sha256": sha256_file(path),
        "resumed": resumed,
    }


def check_cpp_lock(
    context: ManifestContext, latency_input: pathlib.Path | None = None
) -> dict[str, Any]:
    path = _repo_path(
        context.repository,
        context.manifest["outputs"]["cpp_lock_header"],
        "C++ calibration lock",
    )
    selection = (
        _selection_from_latency_input(context, latency_input)
        if latency_input is not None
        else None
    )
    if not path.is_file() or path.read_bytes() != render_cpp_lock_bytes(
        context, selection
    ):
        raise GateError("C++ Game Review calibration lock is missing or stale")
    return {
        "cpp_lock_header": str(path.relative_to(context.repository)),
        "sha256": sha256_file(path),
        "current": True,
    }


def build_compact_result(context: ManifestContext) -> tuple[dict[str, Any], str]:
    selection = _load_and_validate_selection(context)
    test_path = _phase_result_path(context, "test")
    test = _validate_phase_result(context, "test", load_json(test_path))
    selected_id = selection["selected_profile_id"]
    opponent_results: list[dict[str, Any]] = []
    for reference in context.references:
        matchup_id = f"{selected_id}-vs-{reference['id']}"
        matchup = _object(test["matchups"].get(matchup_id), f"test matchup {matchup_id}")
        interval = _object(matchup.get("pair_bootstrap_95"), "test bootstrap")
        opponent_results.append(
            {
                "reference_id": reference["id"],
                "pairs": matchup["pairs"],
                "games": matchup["games"],
                "mean_pair_score": matchup["mean_pair_score"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "lower_bound_strictly_above_half": interval["lower"] > 0.5,
            }
        )
    completeness = test["completeness"]
    operationally_clean = completeness["operationally_valid"] is True and all(
        completeness[key] == 0
        for key in (
            "illegal_moves",
            "incomplete_actions",
            "unexplained_truncations",
            "parity_failures",
        )
    )
    expert_passed = operationally_clean and all(
        result["lower_bound_strictly_above_half"] for result in opponent_results
    )
    calibration_mappings = _object(
        selection.get("calibration_mappings"), "selection calibration mappings"
    )
    value = {
        "schema": COMPACT_RESULT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "competition_source": selection["competition_source"],
        "selection_sha256": sha256_file(_selection_path(context)),
        "test_sha256": sha256_file(test_path),
        "selected_profile_id": selected_id,
        "selected_profile_sha256": selection["selected_profile_sha256"],
        "review_mode_calibration_profiles": selection[
            "review_mode_calibration_profiles"
        ],
        "calibration_mapping_sha256": {
            profile_id: sha256_bytes(canonical_json_bytes(mapping))
            for profile_id, mapping in sorted(calibration_mappings.items())
        },
        "test_games": completeness["completed_games"],
        "bootstrap_resamples": 10_000,
        "operationally_clean": operationally_clean,
        "opponents": opponent_results,
        "expert_gate": {
            "passed": expert_passed,
            "selector_label": "Expert — DeepTurnSearch" if expert_passed else None,
            "strength_status": "validated" if expert_passed else "strength unresolved",
            "claim_rule": context.manifest["statistics"]["expert_gate"]["claim_rule"],
        },
    }
    report_lines = [
        "# DeepTurnSearch strength and calibration gate",
        "",
        "This report is generated only from the frozen Game Review gate artifacts. ",
        "It is separate from the unchanged flagship study and does not evaluate the ",
        "authentic ranked `rank_5` submission.",
        "",
        "## Locked profile",
        "",
        f"Selected review profile: `{selected_id}`.",
        "",
        "The profile was selected on validation strength within one percentage point ",
        "of the eligible leader, then by lower WebAssembly p95 and lower fixed work. ",
        "The selected Deep mapping was fit only on fresh validation decisions from ",
        "that Deep profile. The separate Fast mapping was fit only on fresh validation ",
        "decisions from the fixed Rank5Derived reference, whose immutable 50k search ",
        "settings match Fast analysis. Their profile IDs and hashes remain distinct.",
        "",
        "## Frozen test",
        "",
        "| Reference | Pairs | Games | Score | Paired 95% interval |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in opponent_results:
        report_lines.append(
            f"| `{result['reference_id']}` | {result['pairs']} | {result['games']} | "
            f"{100 * result['mean_pair_score']:.2f}% | "
            f"{100 * result['ci_lower']:.2f}%–{100 * result['ci_upper']:.2f}% |"
        )
    report_lines += [
        "",
        "All intervals use 10,000 opening-depth-stratified whole-pair bootstrap ",
        "resamples. The test contains exactly 1,600 decisive games.",
        "",
        "## Expert decision",
        "",
        (
            "The gate passed against both references. The playable selector may show "
            "**Expert — DeepTurnSearch**."
            if expert_passed
            else "Strength remains unresolved. Deep Game Review ships without an Expert "
            "opponent claim or selector entry."
        ),
        "",
        "The decision requires zero illegal moves, incomplete actions, unexplained ",
        "truncations, and parity failures, plus a paired 95% lower bound strictly above ",
        "50% against each reference. No overall accuracy number is inferred.",
        "",
    ]
    return value, "\n".join(report_lines)


def render_web_gate_status_bytes(value: Mapping[str, Any]) -> bytes:
    expert = _object(value.get("expert_gate"), "compact result expert_gate")
    passed = expert.get("passed") is True
    payload = {
        "schema": WEB_GATE_SCHEMA,
        "expertOpponentEnabled": passed,
        "strengthStatus": expert.get("strength_status"),
        "selectorLabel": expert.get("selector_label") if passed else None,
        "selectedProfileId": value.get("selected_profile_id"),
        "testGames": value.get("test_games"),
        "compactResultSha256": sha256_bytes(canonical_json_bytes(value)),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return (
        "(function (root) {\n"
        "  \"use strict\";\n\n"
        "  root.PaperSoccerGameReviewGate = Object.freeze(" + encoded + ");\n"
        "}(globalThis));\n"
    ).encode("utf-8")


def write_report(context: ManifestContext) -> dict[str, Any]:
    value, report = build_compact_result(context)
    compact_path = _repo_path(
        context.repository,
        context.manifest["outputs"]["compact_results"],
        "compact results",
    )
    report_path = _repo_path(
        context.repository, context.manifest["outputs"]["report"], "report"
    )
    web_gate_path = _repo_path(
        context.repository,
        context.manifest["outputs"]["web_gate_status"],
        "Web Expert gate status",
    )
    compact_resumed = write_json(compact_path, value)
    report_resumed = _atomic_write(report_path, report.encode("utf-8"))
    web_gate_resumed = _atomic_write(
        web_gate_path, render_web_gate_status_bytes(value), replace=True
    )
    return {
        "compact_results": str(compact_path.relative_to(context.repository)),
        "compact_sha256": sha256_file(compact_path),
        "report": str(report_path.relative_to(context.repository)),
        "report_sha256": sha256_file(report_path),
        "web_gate_status": str(web_gate_path.relative_to(context.repository)),
        "web_gate_status_sha256": sha256_file(web_gate_path),
        "expert_gate_passed": value["expert_gate"]["passed"],
        "resumed": compact_resumed and report_resumed and web_gate_resumed,
    }


def validate_available_artifacts(
    context: ManifestContext, *, require_complete: bool = False
) -> dict[str, Any]:
    outputs = context.manifest["outputs"]
    validated: list[str] = ["manifest", "opening_identities"]
    for phase in PHASES:
        path = _phase_result_path(context, phase)
        if path.exists():
            _validate_phase_result(context, phase, load_json(path))
            validated.append(f"phase:{phase}")
        elif require_complete:
            raise GateError(f"required {phase} result is missing")
    latency_path = _repo_path(
        context.repository, outputs["wasm_latency"], "Wasm latency"
    )
    if latency_path.exists():
        validate_latency_value(context, load_json(latency_path))
        validated.append("wasm_latency")
    elif require_complete:
        raise GateError("required Wasm latency artifact is missing")
    selection_path = _selection_path(context)
    if selection_path.exists():
        _load_and_validate_selection(context)
        validated.append("selection_lock")
        check_cpp_lock(context)
        validated.append("cpp_lock_header")
    elif require_complete:
        raise GateError("required selection lock is missing")
    compact_path = _repo_path(
        context.repository, outputs["compact_results"], "compact results"
    )
    report_path = _repo_path(context.repository, outputs["report"], "report")
    web_gate_path = _repo_path(
        context.repository, outputs["web_gate_status"], "Web Expert gate status"
    )
    if compact_path.exists() or report_path.exists():
        expected, report = build_compact_result(context)
        if not compact_path.exists() or load_json(compact_path) != expected:
            raise GateError("compact result is stale")
        if not report_path.exists() or report_path.read_text(encoding="utf-8") != report:
            raise GateError("gate report is stale")
        if (not web_gate_path.exists() or
                web_gate_path.read_bytes() != render_web_gate_status_bytes(expected)):
            raise GateError("Web Expert gate status is missing or stale")
        validated += ["compact_results", "report", "web_gate_status"]
    elif require_complete:
        raise GateError("required compact result/report is missing")
    return {
        "valid": True,
        "manifest_sha256": context.manifest_sha256,
        "gate_opening_banks": len(context.gate_banks),
        "gate_openings": sum(bank.pairs for bank in context.gate_banks),
        "excluded_flagship_banks": len(context.excluded_banks),
        "artifacts_validated": validated,
        "complete": require_complete,
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "GateError",
    "ManifestContext",
    "arena_command",
    "aggregate_phase",
    "build_compact_result",
    "build_opening_identities",
    "check_cpp_lock",
    "create_selection_value",
    "fit_logistic_calibration",
    "lock_selection",
    "nearest_rank",
    "parse_opening_bank",
    "record_latency",
    "render_cpp_lock",
    "render_cpp_lock_bytes",
    "render_web_gate_status_bytes",
    "run_opening_tool_validation",
    "run_phase",
    "select_candidate",
    "stratified_pair_bootstrap",
    "units_for_phase",
    "validate_available_artifacts",
    "validate_latency_value",
    "validate_manifest",
    "verify_opening_regeneration",
    "write_report",
]
