#!/usr/bin/env python3
"""Generate disjoint complete-turn opening banks for compact_value_bfm."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import secrets
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _load_module(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opening helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module(
    HERE / "compact_value_bfm_qualification.py",
    "compact_value_bfm_openings_qualification",
)
reference = _load_module(
    HERE / "jacek_replay_features.py",
    "compact_value_bfm_openings_reference",
)
OpeningError = base.QualificationError

NAMESPACE = "compact_value_bfm"
BANK_SCHEMA = "papersoccer.compact-value-bfm.opening-bank.v1"
SEED_SCHEMA = "papersoccer.compact-value-bfm.protected-seed-receipt.v1"
PREFLIGHT_SCHEMA = "papersoccer.compact-value-bfm.preflight-receipt.v1"
RULES = "8x10;own-goals-allowed;mover-loses"
MINIMUM_PHYSICAL_PLIES = 12
FINAL_OPENINGS = 500
SHA256_RE = re.compile(r"[0-9a-f]{64}")

DEVELOPMENT_COUNTS = {
    "model_screen": 100,
    "tuple_screen": 100,
    "tuple_confirmation": 250,
    "profile_screen": 100,
    "profile_confirmation": 250,
    "actual_clock": 200,
}
DEVELOPMENT_ORDER = tuple(DEVELOPMENT_COUNTS)
DEVELOPMENT_MASTER_SEED = hashlib.sha256(
    b"compact-value-bfm-development-openings-v1"
).digest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise OpeningError(f"{field} must be a lowercase SHA-256")
    return value


def clone_state(state):
    return reference.ReplayState(
        ball=state.ball,
        to_move=state.to_move,
        winner=state.winner,
        used_segments=set(state.used_segments),
        visit_count=dict(state.visit_count),
    )


def replay_transcript(transcript: str):
    if not isinstance(transcript, str) or not transcript or "/" not in transcript:
        raise OpeningError("opening transcript must contain slash-separated turns")
    actions = transcript.split("/")
    if any(not action or re.fullmatch(r"[0-7]+", action) is None for action in actions):
        raise OpeningError("opening transcript contains an invalid complete turn")
    state = reference.ReplayState()
    for action in actions:
        try:
            reference.apply_complete_turn(state, state.to_move, action)
        except ValueError as error:
            raise OpeningError(f"opening transcript is not complete/legal: {error}") from error
    primitive_plies = sum(len(action) for action in actions)
    if primitive_plies < MINIMUM_PHYSICAL_PLIES:
        raise OpeningError("opening transcript is shallower than 12 physical plies")
    if state.winner is not None:
        raise OpeningError("opening transcript is terminal")
    return state, primitive_plies


def _transform_point(point, *, rotate: bool, reflect: bool):
    result = reference.rotate_point(point) if rotate else point
    return reference.reflect_point(result) if reflect else result


def transform_state(state, *, rotate: bool, reflect: bool):
    transform = lambda point: _transform_point(
        point, rotate=rotate, reflect=reflect
    )
    return reference.ReplayState(
        ball=transform(state.ball),
        to_move=1 - state.to_move if rotate else state.to_move,
        winner=None if state.winner is None else (
            1 - state.winner if rotate else state.winner
        ),
        used_segments={
            reference._segment(transform(first), transform(second))
            for first, second in state.used_segments
        },
        visit_count={transform(point): count
                     for point, count in state.visit_count.items()},
    )


def state_serialization(state) -> bytes:
    if state.winner is not None:
        raise OpeningError("terminal states cannot be opening fingerprints")
    try:
        used = sorted(reference.EDGE_INDEX[segment] for segment in state.used_segments)
    except KeyError as error:
        raise OpeningError("state contains a noncanonical segment") from error
    visits = sorted(
        (reference.POINT_INDEX[point], count)
        for point, count in state.visit_count.items() if count > 0
    )
    payload = {
        "ball": reference.POINT_INDEX[state.ball],
        "to_move": state.to_move,
        "used_edges": used,
        "visit_counts": visits,
    }
    return base.canonical_json_bytes(payload)


def state_fingerprints(state) -> dict[str, str]:
    variants = {
        "exact": (False, False),
        "rotate": (True, False),
        "reflect": (False, True),
        "rotate_reflect": (True, True),
    }
    result = {
        name: base.sha256_bytes(state_serialization(transform_state(
            state, rotate=rotate, reflect=reflect
        )))
        for name, (rotate, reflect) in variants.items()
    }
    result["canonical"] = min(result.values())
    return result


def _source_snapshot(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OpeningError(f"opening exclusion is not a regular file: {path}")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }


def load_exclusion_bank(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise OpeningError(f"opening exclusion is unreadable/non-ASCII: {path}") from error
    if b"\r" in raw:
        raise OpeningError("opening exclusion must use LF line endings")
    lines = text.splitlines()
    if (len(lines) < 7
            or lines[0] != "# papersoccer.jacek-replay-bfm-opening-bank.v1"
            or lines[1] != f"# rules={RULES}"
            or not lines[2].startswith("# classification=")
            or not lines[3].startswith("# seed=")
            or lines[4] != "# minimum-physical-plies=12"
            or lines[5] != "opening_id\ttranscript\tstate_identity"):
        raise OpeningError("opening exclusion metadata contract is invalid")
    openings = []
    ids = set()
    all_fingerprints: set[str] = set()
    for line_number, line in enumerate(lines[6:], 7):
        fields = line.split("\t")
        if len(fields) != 3 or not all(fields):
            raise OpeningError(f"malformed opening exclusion row {line_number}")
        opening_id, transcript, source_identity = fields
        if opening_id in ids:
            raise OpeningError("opening exclusion repeats an opening id")
        ids.add(opening_id)
        state, primitive_plies = replay_transcript(transcript)
        fingerprints = state_fingerprints(state)
        all_fingerprints.update(
            value for key, value in fingerprints.items() if key != "canonical"
        )
        openings.append({
            "opening_id": opening_id,
            "transcript": transcript,
            "primitive_plies": primitive_plies,
            "completed_turn_overshoot": primitive_plies - MINIMUM_PHYSICAL_PLIES,
            "source_state_identity": source_identity,
            "fingerprints": fingerprints,
        })
    if not openings:
        raise OpeningError("opening exclusion is empty")
    return {
        "source": _source_snapshot(path),
        "classification": lines[2].split("=", 1)[1],
        "seed": lines[3].split("=", 1)[1],
        "openings": openings,
        "fingerprints": sorted(all_fingerprints),
    }


def load_all_exclusions(paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    if len(paths) != 7 or len({path.resolve() for path in paths}) != 7:
        raise OpeningError("exactly seven distinct copied opening-exclusion paths are required")
    banks = [load_exclusion_bank(path) for path in paths]
    fingerprints = sorted({
        fingerprint for bank in banks for fingerprint in bank["fingerprints"]
    })
    sources = [bank["source"] for bank in banks]
    material = base.canonical_json_bytes({
        "sources": sources,
        "fingerprints": fingerprints,
    })
    return {
        "sources": sources,
        "banks": banks,
        "fingerprints": fingerprints,
        "body_sha256": base.sha256_bytes(material),
    }


def load_protected_exclusions(
    copied_paths: Sequence[pathlib.Path],
    development_bank_paths: Sequence[pathlib.Path],
) -> dict[str, Any]:
    copied = load_all_exclusions(copied_paths)
    if len(development_bank_paths) != len(DEVELOPMENT_ORDER):
        raise OpeningError("protected generation requires all six development banks")
    by_stage = {}
    for path in development_bank_paths:
        bank = validate_bank(path)
        stage = bank.get("stage")
        if (stage not in DEVELOPMENT_COUNTS or stage in by_stage
                or bank.get("classification") != "unprotected-development"
                or bank.get("opening_count") != DEVELOPMENT_COUNTS[stage]):
            raise OpeningError("protected development-bank roster/count is invalid")
        by_stage[stage] = (path, bank)
    if tuple(stage for stage in DEVELOPMENT_ORDER if stage in by_stage) != DEVELOPMENT_ORDER:
        raise OpeningError("protected development-bank stage roster is incomplete")
    fingerprints = set(copied["fingerprints"])
    development_sources = []
    for stage in DEVELOPMENT_ORDER:
        path, bank = by_stage[stage]
        for opening in bank["openings"]:
            variants = {
                value for name, value in opening["fingerprints"].items()
                if name != "canonical"
            }
            if variants & fingerprints:
                raise OpeningError("development opening overlaps prior exclusions")
            fingerprints.update(variants)
        development_sources.append({
            **_source_snapshot(path),
            "stage": stage,
            "opening_count": bank["opening_count"],
        })
    sources = [*copied["sources"], *development_sources]
    material = base.canonical_json_bytes({
        "sources": sources,
        "fingerprints": sorted(fingerprints),
    })
    return {
        "sources": sources,
        "copied_sources": copied["sources"],
        "development_sources": development_sources,
        "fingerprints": sorted(fingerprints),
        "body_sha256": base.sha256_bytes(material),
    }


class HashRandom:
    def __init__(self, seed: bytes):
        self.seed = seed
        self.counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(
            self.seed + self.counter.to_bytes(16, "big")
        ).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def index(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise OpeningError("random choice has no candidates")
        ceiling = 1 << 64
        limit = ceiling - ceiling % upper_bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper_bound


def legal_directions(state) -> list[int]:
    result = []
    for direction, (dx, dy) in enumerate(reference.DIRECTION_DELTAS):
        destination = state.ball[0] + dx, state.ball[1] + dy
        if reference._legal_destination(state, destination):
            result.append(direction)
    return result


def generate_candidate(seed: bytes):
    random = HashRandom(seed)
    state = reference.ReplayState()
    turns = []
    primitive_plies = 0
    while primitive_plies < MINIMUM_PHYSICAL_PLIES:
        mover = state.to_move
        action = []
        while state.winner is None and state.to_move == mover:
            legal = legal_directions(state)
            if not legal:
                raise OpeningError("generated nonterminal state has no legal primitive")
            direction = legal[random.index(len(legal))]
            reference.apply_primitive(state, direction)
            action.append(str(direction))
            primitive_plies += 1
        if state.winner is not None:
            return None
        if not action:
            raise OpeningError("generated complete turn is empty")
        turns.append("".join(action))
    transcript = "/".join(turns)
    replayed, checked_plies = replay_transcript(transcript)
    if state_serialization(replayed) != state_serialization(state):
        raise OpeningError("generated transcript does not reproduce its state")
    return state, transcript, checked_plies


def generate_openings(
    *, stage: str, count: int, seed: bytes,
    excluded_fingerprints: set[str],
) -> list[dict[str, Any]]:
    if count <= 0:
        raise OpeningError("opening count must be positive")
    openings = []
    seen = set(excluded_fingerprints)
    attempt = 0
    while len(openings) < count:
        candidate_seed = hashlib.sha256(
            seed + stage.encode("ascii") + attempt.to_bytes(16, "big")
        ).digest()
        attempt += 1
        generated = generate_candidate(candidate_seed)
        if generated is None:
            continue
        state, transcript, primitive_plies = generated
        fingerprints = state_fingerprints(state)
        variants = {
            value for key, value in fingerprints.items() if key != "canonical"
        }
        if variants & seen:
            continue
        seen.update(variants)
        openings.append({
            "opening_id": f"{stage}-{len(openings):03d}",
            "transcript": transcript,
            "primitive_plies": primitive_plies,
            "completed_turn_overshoot": primitive_plies - MINIMUM_PHYSICAL_PLIES,
            "ball": list(state.ball),
            "to_move": state.to_move,
            "fingerprints": fingerprints,
        })
        if attempt > count * 10_000:
            raise OpeningError("opening generator exhausted its attempt limit")
    return openings


def _atomic_publish(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise OpeningError(f"immutable opening artifact collision: {path}")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_bank(directory: pathlib.Path, payload: Mapping[str, Any]) -> pathlib.Path:
    artifact = base.seal(payload)
    raw = base.canonical_json_bytes(artifact)
    digest = base.sha256_bytes(raw)
    path = directory / f"{digest}.opening-bank.json"
    _atomic_publish(path, raw)
    return path


def bank_payload(
    *, stage: str, classification: str, seed: bytes,
    exclusions: Mapping[str, Any], openings: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any] | None = None,
    seed_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    campaign_binding = {
        "bank_id": stage,
        "pairs": len(openings),
        "fingerprints": sorted(opening["fingerprints"]["canonical"] for opening in openings),
        "transcripts": [opening["transcript"] for opening in openings],
        "primitive_ply_counts": [opening["primitive_plies"] for opening in openings],
    }
    payload = {
        "schema": BANK_SCHEMA,
        "namespace": NAMESPACE,
        "stage": stage,
        "classification": classification,
        "rules": RULES,
        "minimum_physical_plies": MINIMUM_PHYSICAL_PLIES,
        "seed_hex": seed.hex(),
        "exclusions_body_sha256": exclusions["body_sha256"],
        "exclusion_sources": exclusions["sources"],
        "opening_count": len(openings),
        "openings": list(openings),
        "campaign_binding": campaign_binding,
    }
    if source_binding is not None:
        payload["source_binding"] = dict(source_binding)
    if seed_receipt is not None:
        payload["seed_receipt"] = dict(seed_receipt)
        payload["bank_consumed_at_launch_policy"] = {
            "exclusive_marker_required_before_first_game": True,
            "marker_filename": "bank-consumed.json",
            "marker_must_bind_bank_sha256": True,
            "started_shard_without_receipt_is_terminal": True,
        }
    return payload


def validate_bank(path: pathlib.Path) -> dict[str, Any]:
    artifact = base.load_sealed(path, BANK_SCHEMA)
    expected_prefix = path.name.split(".", 1)[0]
    if expected_prefix != base.sha256_file(path):
        raise OpeningError("opening bank filename is not content-addressed")
    try:
        seed = bytes.fromhex(artifact.get("seed_hex", ""))
    except (TypeError, ValueError) as error:
        raise OpeningError("opening bank seed is malformed") from error
    if (len(seed) != 32 or artifact.get("rules") != RULES
            or artifact.get("minimum_physical_plies") != MINIMUM_PHYSICAL_PLIES
            or artifact.get("namespace") != NAMESPACE):
        raise OpeningError("opening bank header contract changed")
    openings = artifact.get("openings")
    if not isinstance(openings, list) or len(openings) != artifact.get("opening_count"):
        raise OpeningError("opening bank count is inconsistent")
    canonical = []
    transcripts = []
    counts = []
    seen = set()
    for index, opening in enumerate(openings):
        if not isinstance(opening, dict):
            raise OpeningError("opening bank row is malformed")
        state, primitive_plies = replay_transcript(opening.get("transcript"))
        fingerprints = state_fingerprints(state)
        if (opening.get("opening_id") !=
                f"{artifact.get('stage')}-{index:03d}"
                or opening.get("primitive_plies") != primitive_plies
                or opening.get("completed_turn_overshoot") !=
                primitive_plies - MINIMUM_PHYSICAL_PLIES
                or opening.get("fingerprints") != fingerprints
                or opening.get("ball") != list(state.ball)
                or opening.get("to_move") != state.to_move):
            raise OpeningError(f"opening bank row {index} is stale")
        variants = {value for key, value in fingerprints.items() if key != "canonical"}
        if variants & seen:
            raise OpeningError("opening bank repeats a symmetry-equivalent state")
        seen.update(variants)
        canonical.append(fingerprints["canonical"])
        transcripts.append(opening["transcript"])
        counts.append(primitive_plies)
    if artifact.get("campaign_binding") != {
        "bank_id": artifact.get("stage"),
        "pairs": len(openings),
        "fingerprints": sorted(canonical),
        "transcripts": transcripts,
        "primitive_ply_counts": counts,
    }:
        raise OpeningError("opening bank campaign binding is stale")
    if artifact.get("classification") == "protected-final":
        if (len(openings) != FINAL_OPENINGS
                or artifact.get("bank_consumed_at_launch_policy") != {
                    "exclusive_marker_required_before_first_game": True,
                    "marker_filename": "bank-consumed.json",
                    "marker_must_bind_bank_sha256": True,
                    "started_shard_without_receipt_is_terminal": True,
                }):
            raise OpeningError("protected bank size/consumption policy changed")
    return artifact


def _generate_development_banks(
    output_root: pathlib.Path, *, exclusion_paths: Sequence[pathlib.Path],
    counts: Mapping[str, int],
) -> dict[str, pathlib.Path]:
    if tuple(counts) != DEVELOPMENT_ORDER:
        raise OpeningError("development stage order/roster is invalid")
    exclusions = load_all_exclusions(exclusion_paths)
    seen = set(exclusions["fingerprints"])
    result = {}
    for stage in DEVELOPMENT_ORDER:
        seed = hashlib.sha256(DEVELOPMENT_MASTER_SEED + stage.encode("ascii")).digest()
        openings = generate_openings(
            stage=stage, count=counts[stage], seed=seed,
            excluded_fingerprints=seen,
        )
        for opening in openings:
            seen.update(
                value for key, value in opening["fingerprints"].items()
                if key != "canonical"
            )
        path = write_bank(
            output_root / stage,
            bank_payload(
                stage=stage, classification="unprotected-development",
                seed=seed, exclusions=exclusions, openings=openings,
            ),
        )
        validate_bank(path)
        result[stage] = path
    return result


def generate_development_banks(
    output_root: pathlib.Path, *, exclusion_paths: Sequence[pathlib.Path],
) -> dict[str, pathlib.Path]:
    """Generate only the exact frozen 100/100/250/100/250/200 roster."""

    return _generate_development_banks(
        output_root, exclusion_paths=exclusion_paths,
        counts=DEVELOPMENT_COUNTS,
    )


def _load_clean_binding(
    path: pathlib.Path, *, source_binding_path: pathlib.Path,
) -> dict[str, Any]:
    value = base.load_sealed(path)
    source = base.load_sealed(source_binding_path, base.SOURCE_BINDING_SCHEMA)
    base.validate_source_binding(source)
    expected_ref = base.artifact_reference(
        source_binding_path, base.SOURCE_BINDING_SCHEMA
    )
    if value.get("schema") == PREFLIGHT_SCHEMA:
        before = value.get("inputs_before", {})
        checks = value.get("checks")
        if (value.get("status") != "passed" or value.get("inputs_after") != before
                or before.get("candidate_commit") != source["candidate_commit"]
                or before.get("candidate", {}).get("sha256") != source["candidate"]["sha256"]
                or value.get("git_writes") != 0
                or value.get("uploads") != 0
                or value.get("protected_banks_accessed") != []
                or not isinstance(checks, dict) or not checks
                or any(status != "passed" for status in checks.values())
                or not path.name.endswith(".json")
                or path.name[:-5] != base.sha256_file(path)):
            raise OpeningError("preflight receipt is not a clean source binding")
    else:
        raise OpeningError("unsupported clean source-binding schema")
    return {"clean": value, "source": source, "source_reference": expected_ref}


def create_protected_seed_receipt(
    output: pathlib.Path, *, source_binding_path: pathlib.Path,
    clean_binding_path: pathlib.Path,
    exclusion_paths: Sequence[pathlib.Path],
    development_bank_paths: Sequence[pathlib.Path],
    created_at_utc: str,
    entropy: Any = secrets.token_bytes,
) -> dict[str, Any]:
    binding = _load_clean_binding(
        clean_binding_path, source_binding_path=source_binding_path
    )
    exclusions = load_protected_exclusions(
        exclusion_paths, development_bank_paths
    )
    seed = entropy(32)
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise OpeningError("OS entropy source did not return exactly 256 bits")
    payload = {
        "schema": SEED_SCHEMA,
        "namespace": NAMESPACE,
        "status": "protected-seed-frozen-before-bank-generation",
        "created_at_utc": created_at_utc,
        "seed_256_hex": seed.hex(),
        "source_binding": binding["source_reference"],
        "clean_binding": base.artifact_reference(clean_binding_path),
        "candidate_commit": binding["source"]["candidate_commit"],
        "candidate_sha256": binding["source"]["candidate"]["sha256"],
        "exclusions_body_sha256": exclusions["body_sha256"],
        "exclusion_sources": exclusions["sources"],
        "exclusion_fingerprint_count": len(exclusions["fingerprints"]),
        "entropy_bits": 256,
        "bank_generated": False,
    }
    artifact = base.seal(payload)
    _atomic_publish(output, base.canonical_json_bytes(artifact))
    return artifact


def generate_protected_bank(
    output_root: pathlib.Path, *, seed_receipt_path: pathlib.Path,
    exclusion_paths: Sequence[pathlib.Path],
    development_bank_paths: Sequence[pathlib.Path],
) -> pathlib.Path:
    receipt = base.load_sealed(seed_receipt_path, SEED_SCHEMA)
    exclusions = load_protected_exclusions(
        exclusion_paths, development_bank_paths
    )
    if (receipt.get("status") != "protected-seed-frozen-before-bank-generation"
            or receipt.get("entropy_bits") != 256
            or receipt.get("exclusions_body_sha256") != exclusions["body_sha256"]
            or receipt.get("exclusion_sources") != exclusions["sources"]):
        raise OpeningError("protected seed receipt/exclusion binding changed")
    try:
        seed = bytes.fromhex(receipt["seed_256_hex"])
    except (KeyError, TypeError, ValueError) as error:
        raise OpeningError("protected seed receipt is malformed") from error
    if len(seed) != 32:
        raise OpeningError("protected seed is not 256 bits")
    openings = generate_openings(
        stage="protected_final", count=FINAL_OPENINGS, seed=seed,
        excluded_fingerprints=set(exclusions["fingerprints"]),
    )
    path = write_bank(
        output_root / "protected_final",
        bank_payload(
            stage="protected_final", classification="protected-final",
            seed=seed, exclusions=exclusions, openings=openings,
            source_binding=receipt["source_binding"],
            seed_receipt=base.artifact_reference(seed_receipt_path, SEED_SCHEMA),
        ),
    )
    validate_bank(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    development = commands.add_parser("generate-development")
    development.add_argument("--output-root", type=pathlib.Path, required=True)
    development.add_argument("--exclusion-bank", type=pathlib.Path,
                             action="append", required=True)
    seed = commands.add_parser("protected-seed")
    seed.add_argument("--output", type=pathlib.Path, required=True)
    seed.add_argument("--source-binding", type=pathlib.Path, required=True)
    seed.add_argument("--clean-binding", type=pathlib.Path, required=True)
    seed.add_argument("--exclusion-bank", type=pathlib.Path,
                      action="append", required=True)
    seed.add_argument("--development-bank", type=pathlib.Path,
                      action="append", required=True)
    seed.add_argument("--created-at-utc", required=True)
    protected = commands.add_parser("generate-protected")
    protected.add_argument("--output-root", type=pathlib.Path, required=True)
    protected.add_argument("--seed-receipt", type=pathlib.Path, required=True)
    protected.add_argument("--exclusion-bank", type=pathlib.Path,
                           action="append", required=True)
    protected.add_argument("--development-bank", type=pathlib.Path,
                           action="append", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--bank", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-development":
            result = {
                name: str(path) for name, path in generate_development_banks(
                    args.output_root, exclusion_paths=args.exclusion_bank
                ).items()
            }
        elif args.command == "protected-seed":
            receipt = create_protected_seed_receipt(
                args.output, source_binding_path=args.source_binding,
                clean_binding_path=args.clean_binding,
                exclusion_paths=args.exclusion_bank,
                development_bank_paths=args.development_bank,
                created_at_utc=args.created_at_utc,
            )
            result = {"path": str(args.output), "body_sha256": receipt["body_sha256"]}
        elif args.command == "generate-protected":
            path = generate_protected_bank(
                args.output_root, seed_receipt_path=args.seed_receipt,
                exclusion_paths=args.exclusion_bank,
                development_bank_paths=args.development_bank,
            )
            result = {"path": str(path), "sha256": base.sha256_file(path)}
        else:
            bank = validate_bank(args.bank)
            result = {"path": str(args.bank), "stage": bank["stage"],
                      "openings": bank["opening_count"]}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OpeningError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact opening failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
