#!/usr/bin/env python3
"""Fail-closed qualification and deployment ledgers for compact_value_bfm.

The module deliberately does not run a bot or fetch CodinGame data.  It binds
the immutable inputs and records produced by those operations, with exclusive
write-once transitions.  A caller may safely resume completed final shards;
once a shard start claim exists without a valid receipt, that shard is spent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


NAMESPACE = "compact_value_bfm"
SOURCE_BINDING_SCHEMA = "papersoccer.compact-value-bfm.source-binding.v1"
FINAL_BANK_SCHEMA = "papersoccer.compact-value-bfm.protected-bank.v1"
GATE_BINDING_SCHEMA = "papersoccer.compact-value-bfm.gate-binding.v1"
SHARD_CLAIM_SCHEMA = "papersoccer.compact-value-bfm.final-shard-claim.v1"
SHARD_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.final-shard-receipt.v1"
FINAL_AGGREGATE_SCHEMA = "papersoccer.compact-value-bfm.final-aggregate.v1"
UPLOAD_AUTH_SCHEMA = "papersoccer.compact-value-bfm.one-upload-authorization.v1"
UPLOAD_EVENT_SCHEMA = "papersoccer.compact-value-bfm.upload-event.v1"
LIVE_WINDOW_SCHEMA = "papersoccer.compact-value-bfm.live-window.v1"

RANK4_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
RANK4_BYTES = 98_624
SOURCE_LIMIT = 95_000
FINAL_OPENINGS = 500
FINAL_SHARDS = 100
PAIRS_PER_SHARD = 5
FINAL_GAMES = 1_000
MINIMUM_WINS = 527
MINIMUM_COLOR_WINS = 260
MAXIMUM_TURNS = 320
EXACT_LIVE_GAMES = 90
FAILURE_CATEGORIES = (
    "illegal", "unfinished", "timeout", "crash", "malformed", "overlong"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_BATTLE_KEYS = frozenset({
    "agents", "frames", "gameinformation", "inputs", "outputs", "replay",
    "stderr", "stdin", "stdout", "transcript", "turns",
})


class QualificationError(ValueError):
    """A requested transition or artifact violates the frozen contract."""


class SpentShardError(QualificationError):
    """A final shard was started but has no valid immutable receipt."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise QualificationError(f"{field} must be a lowercase SHA-256")
    return value


def _commit(value: Any, field: str = "commit") -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise QualificationError(f"{field} must be a full lowercase Git SHA-1")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationError(f"{field} must be a positive integer")
    return value


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise QualificationError(f"{field} must be finite and nonnegative")
    return result


def _utc(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise QualificationError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise QualificationError(f"{field} is not a valid UTC timestamp") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise QualificationError(f"{field} must be UTC")
    return parsed


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    if "body_sha256" in body:
        raise QualificationError("body_sha256 is reserved")
    body["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return body


def validate_seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise QualificationError("sealed artifact must be an object")
    body = dict(payload)
    claimed = _sha(body.pop("body_sha256", None), "body SHA-256")
    if sha256_bytes(canonical_json_bytes(body)) != claimed:
        raise QualificationError("sealed artifact body SHA-256 mismatch")
    return dict(payload)


def atomic_write_once(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = pathlib.Path(stream.name)
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise QualificationError(f"immutable artifact collision: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_sealed(path: pathlib.Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = seal(payload)
    atomic_write_once(path, canonical_json_bytes(artifact))
    return artifact


def load_sealed(path: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot load artifact: {path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise QualificationError(f"artifact is not canonical JSON: {path}")
    validate_seal(value)
    if schema is not None and value.get("schema") != schema:
        raise QualificationError(f"unexpected artifact schema: {path}")
    return value


def artifact_reference(path: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    load_sealed(path, schema)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _file_record(path: pathlib.Path, *, ascii_required: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"source must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    record = {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256_bytes(raw)}
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise QualificationError(f"source is not ASCII: {path}") from error
        record["ascii"] = True
    return record


def create_source_binding(
    output: pathlib.Path,
    *,
    candidate_source: pathlib.Path,
    candidate_commit: str,
    rank4_source: pathlib.Path,
    opponent_source: pathlib.Path,
) -> dict[str, Any]:
    candidate = _file_record(candidate_source, ascii_required=True)
    rank4 = _file_record(rank4_source, ascii_required=True)
    opponent = _file_record(opponent_source, ascii_required=True)
    if not 0 < candidate["bytes"] <= SOURCE_LIMIT:
        raise QualificationError("candidate source exceeds the 95,000-byte contract")
    for label, record in (("Rank-4", rank4), ("opponent", opponent)):
        if record["sha256"] != RANK4_SHA256 or record["bytes"] != RANK4_BYTES:
            raise QualificationError(f"{label} is not the exact maintained Rank-4 source")
    if candidate["sha256"] == RANK4_SHA256:
        raise QualificationError("candidate must remain distinct from maintained Rank-4")
    return write_sealed(output, {
        "schema": SOURCE_BINDING_SCHEMA,
        "namespace": NAMESPACE,
        "candidate_commit": _commit(candidate_commit, "candidate commit"),
        "candidate": candidate,
        "rank4": rank4,
        "opponent": opponent,
    })


def validate_source_binding(binding: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    validate_seal(binding)
    if binding.get("schema") != SOURCE_BINDING_SCHEMA or binding.get("namespace") != NAMESPACE:
        raise QualificationError("invalid compact source binding")
    _commit(binding.get("candidate_commit"), "candidate commit")
    for name in ("candidate", "rank4", "opponent"):
        record = binding.get(name)
        if not isinstance(record, dict):
            raise QualificationError(f"source binding omits {name}")
        _sha(record.get("sha256"), f"{name} SHA-256")
        _positive_int(record.get("bytes"), f"{name} bytes")
        if record.get("ascii") is not True:
            raise QualificationError(f"{name} ASCII binding is absent")
        if verify_files and _file_record(pathlib.Path(record["path"]), ascii_required=True) != record:
            raise QualificationError(f"bound {name} source changed")
    if binding["candidate"]["bytes"] > SOURCE_LIMIT:
        raise QualificationError("bound candidate is oversized")
    for name in ("rank4", "opponent"):
        if binding[name]["sha256"] != RANK4_SHA256 or binding[name]["bytes"] != RANK4_BYTES:
            raise QualificationError(f"bound {name} is not exact Rank-4")
    return dict(binding)


def _opening(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationError(f"opening {index} is not an object")
    expected = {
        "opening_id", "transcript", "primitive_plies", "fingerprint",
        "symmetry_fingerprints",
    }
    if set(value) != expected:
        raise QualificationError(f"opening {index} field set is invalid")
    if value["opening_id"] != f"final-{index:03d}":
        raise QualificationError(f"opening {index} id is not canonical")
    transcript = value["transcript"]
    if not isinstance(transcript, str) or "/" not in transcript:
        raise QualificationError(
            f"opening {index} is not a slash-separated complete-turn transcript"
        )
    turns = transcript.split("/")
    if any(not turn or re.fullmatch(r"[0-7]+", turn) is None for turn in turns):
        raise QualificationError(f"opening {index} has a malformed complete turn")
    primitive_plies = sum(len(turn) for turn in turns)
    if (
        primitive_plies < 12
        or isinstance(value["primitive_plies"], bool)
        or value["primitive_plies"] != primitive_plies
    ):
        raise QualificationError(
            f"opening {index} is shallower than the 12-ply completed-turn bank"
        )
    fingerprint = _sha(value["fingerprint"], f"opening {index} fingerprint")
    symmetries = value["symmetry_fingerprints"]
    if (not isinstance(symmetries, list) or not symmetries or
            symmetries != sorted(set(symmetries)) or
            any(SHA256_RE.fullmatch(str(item)) is None for item in symmetries) or
            fingerprint not in symmetries):
        raise QualificationError(f"opening {index} symmetry fingerprints are invalid")
    return dict(value)


def create_final_bank(
    output: pathlib.Path,
    *,
    source_binding_path: pathlib.Path,
    openings: Sequence[Mapping[str, Any]],
    excluded_fingerprints: Sequence[str] = (),
    seed_factory=secrets.token_bytes,
) -> dict[str, Any]:
    binding = load_sealed(source_binding_path, SOURCE_BINDING_SCHEMA)
    validate_source_binding(binding)
    if len(openings) != FINAL_OPENINGS:
        raise QualificationError("protected bank must contain exactly 500 openings")
    excluded = {_sha(item, "excluded fingerprint") for item in excluded_fingerprints}
    normalized = [_opening(value, index) for index, value in enumerate(openings)]
    seen: set[str] = set()
    for opening in normalized:
        symmetries = set(opening["symmetry_fingerprints"])
        if symmetries & excluded:
            raise QualificationError("protected opening overlaps an excluded symmetry")
        if symmetries & seen:
            raise QualificationError("protected openings overlap by symmetry")
        seen.update(symmetries)
    # Deliberately obtain entropy only after the source binding and exclusions
    # have been fully validated.
    seed = seed_factory(32)
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise QualificationError("OS entropy source did not return 256 bits")
    return write_sealed(output, {
        "schema": FINAL_BANK_SCHEMA,
        "namespace": NAMESPACE,
        "classification": "fresh-protected-final",
        "source_binding": artifact_reference(source_binding_path, SOURCE_BINDING_SCHEMA),
        "candidate_commit": binding["candidate_commit"],
        "candidate_sha256": binding["candidate"]["sha256"],
        "rank4_sha256": RANK4_SHA256,
        "seed_256_hex": seed.hex(),
        "opening_plies": 12,
        "opening_count": FINAL_OPENINGS,
        "excluded_fingerprint_count": len(excluded),
        "openings": normalized,
    })


def create_gate_binding(
    output: pathlib.Path,
    *,
    source_binding_path: pathlib.Path,
    bank_path: pathlib.Path,
    harness_path: pathlib.Path,
) -> dict[str, Any]:
    source = load_sealed(source_binding_path, SOURCE_BINDING_SCHEMA)
    validate_source_binding(source)
    bank = load_sealed(bank_path, FINAL_BANK_SCHEMA)
    if (bank.get("source_binding") != artifact_reference(source_binding_path, SOURCE_BINDING_SCHEMA)
            or bank.get("candidate_commit") != source["candidate_commit"]
            or bank.get("candidate_sha256") != source["candidate"]["sha256"]
            or bank.get("rank4_sha256") != RANK4_SHA256
            or bank.get("opening_count") != FINAL_OPENINGS):
        raise QualificationError("protected bank contradicts the source binding")
    return write_sealed(output, {
        "schema": GATE_BINDING_SCHEMA,
        "namespace": NAMESPACE,
        "candidate_commit": source["candidate_commit"],
        "candidate": source["candidate"],
        "rank4": source["rank4"],
        "opponent": source["opponent"],
        "source_binding": artifact_reference(source_binding_path, SOURCE_BINDING_SCHEMA),
        "bank": artifact_reference(bank_path, FINAL_BANK_SCHEMA),
        "harness": _file_record(harness_path),
        "shards": FINAL_SHARDS,
        "pairs_per_shard": PAIRS_PER_SHARD,
    })


def strict_gate_verdict(summary: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if summary.get("games") != FINAL_GAMES:
        errors.append("games")
    wins = summary.get("candidate_wins")
    if isinstance(wins, bool) or not isinstance(wins, int) or wins < MINIMUM_WINS:
        errors.append("candidate_wins")
    colors = summary.get("candidate_color_wins")
    if not isinstance(colors, dict) or set(colors) != {"0", "1"}:
        errors.append("candidate_color_wins")
    else:
        for color in ("0", "1"):
            value = colors[color]
            if isinstance(value, bool) or not isinstance(value, int) or value < MINIMUM_COLOR_WINS:
                errors.append(f"candidate_color_{color}")
    failures = summary.get("failures")
    if not isinstance(failures, dict) or set(failures) != set(FAILURE_CATEGORIES):
        errors.append("failures")
    elif any(isinstance(value, bool) or not isinstance(value, int) or value != 0
             for value in failures.values()):
        errors.append("failures_nonzero")
    maximum_turns = summary.get("maximum_turns")
    if isinstance(maximum_turns, bool) or not isinstance(maximum_turns, int) or maximum_turns > MAXIMUM_TURNS:
        errors.append("maximum_turns")
    for section, first_limit, later_limit in (
        ("timing", 1000.0, 200.0),
        ("uncontended_timing", 900.0, 180.0),
    ):
        timing = summary.get(section)
        if not isinstance(timing, dict) or set(timing) != {"first_max_ms", "later_max_ms"}:
            errors.append(section)
            continue
        try:
            first = _nonnegative_finite(timing["first_max_ms"], f"{section} first")
            later = _nonnegative_finite(timing["later_max_ms"], f"{section} later")
        except QualificationError:
            errors.append(section)
            continue
        if first >= first_limit:
            errors.append(f"{section}_first")
        if later >= later_limit:
            errors.append(f"{section}_later")
    return {
        "passed": not errors,
        "errors": errors,
        "thresholds": {
            "games": FINAL_GAMES,
            "candidate_wins_min": MINIMUM_WINS,
            "candidate_color_wins_min": MINIMUM_COLOR_WINS,
            "maximum_turns": MAXIMUM_TURNS,
            "first_ms_exclusive": 1000,
            "later_ms_exclusive": 200,
            "uncontended_first_ms_exclusive": 900,
            "uncontended_later_ms_exclusive": 180,
        },
    }


def _binding_digest(binding_path: pathlib.Path) -> str:
    load_sealed(binding_path, GATE_BINDING_SCHEMA)
    return sha256_file(binding_path)


def _claim_path(root: pathlib.Path, index: int) -> pathlib.Path:
    return root / "claims" / f"shard-{index:03d}.json"


def _receipt_path(root: pathlib.Path, index: int) -> pathlib.Path:
    return root / "receipts" / f"shard-{index:03d}.json"


def _validate_shard_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < FINAL_SHARDS:
        raise QualificationError("final shard index must be in [0, 100)")


def start_final_shard(
    root: pathlib.Path, *, binding_path: pathlib.Path, index: int, started_at_utc: str
) -> dict[str, Any]:
    _validate_shard_index(index)
    binding = load_sealed(binding_path, GATE_BINDING_SCHEMA)
    binding_sha = _binding_digest(binding_path)
    consumption_path = root / "bank-consumed.json"
    if consumption_path.exists():
        consumption = load_sealed(consumption_path)
        if (consumption.get("binding_sha256") != binding_sha
                or consumption.get("bank") != binding["bank"]):
            raise QualificationError("protected bank was consumed by another identity")
    else:
        write_sealed(consumption_path, {
            "schema": "papersoccer.compact-value-bfm.bank-consumption.v1",
            "namespace": NAMESPACE,
            "binding_sha256": binding_sha,
            "bank": binding["bank"],
            "consumed_at_utc": started_at_utc,
        })
    claim_path = _claim_path(root, index)
    receipt_path = _receipt_path(root, index)
    if claim_path.exists():
        claim = load_sealed(claim_path, SHARD_CLAIM_SCHEMA)
        if claim.get("binding_sha256") != binding_sha or claim.get("shard_index") != index:
            raise QualificationError("existing shard claim has another identity")
        if not receipt_path.exists():
            raise SpentShardError(f"shard {index} was started without a valid receipt")
        validate_shard_receipt(receipt_path, binding_path=binding_path, index=index)
        return {"status": "complete-reused", "claim": claim, "receipt": str(receipt_path)}
    if receipt_path.exists():
        raise QualificationError("shard receipt exists without its start claim")
    claim = write_sealed(claim_path, {
        "schema": SHARD_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "one_shot": True,
        "binding_sha256": binding_sha,
        "bank_sha256": binding["bank"]["sha256"],
        "candidate_sha256": binding["candidate"]["sha256"],
        "opponent_sha256": binding["opponent"]["sha256"],
        "harness_sha256": binding["harness"]["sha256"],
        "shard_index": index,
        "pair_begin": index * PAIRS_PER_SHARD,
        "pair_count": PAIRS_PER_SHARD,
        "started_at_utc": started_at_utc,
    })
    return {"status": "started", "claim": claim}


def _normalize_shard_games(games: Any, index: int) -> list[dict[str, Any]]:
    if not isinstance(games, list) or len(games) != PAIRS_PER_SHARD * 2:
        raise QualificationError("final shard receipt must contain exactly ten games")
    pair_begin = index * PAIRS_PER_SHARD
    expected = {(pair, color) for pair in range(pair_begin, pair_begin + PAIRS_PER_SHARD)
                for color in (0, 1)}
    observed: set[tuple[int, int]] = set()
    result = []
    for ordinal, raw in enumerate(games):
        if not isinstance(raw, dict):
            raise QualificationError(f"shard game {ordinal} is not an object")
        required = {"pair_index", "candidate_color", "candidate_win", "turns",
                    "failure", "first_ms", "later_max_ms"}
        if set(raw) != required:
            raise QualificationError(f"shard game {ordinal} field set is invalid")
        key = (raw["pair_index"], raw["candidate_color"])
        if key not in expected or key in observed:
            raise QualificationError("shard game pair/color coverage is invalid")
        observed.add(key)
        if not isinstance(raw["candidate_win"], bool):
            raise QualificationError("candidate_win must be boolean")
        turns = _positive_int(raw["turns"], "game turns")
        failure = raw["failure"]
        if failure is not None and failure not in FAILURE_CATEGORIES:
            raise QualificationError("game failure category is invalid")
        first = _nonnegative_finite(raw["first_ms"], "game first_ms")
        later = _nonnegative_finite(raw["later_max_ms"], "game later_max_ms")
        result.append({**raw, "turns": turns, "first_ms": first, "later_max_ms": later})
    if observed != expected:
        raise QualificationError("shard game coverage is incomplete")
    return sorted(result, key=lambda item: (item["pair_index"], item["candidate_color"]))


def record_shard_receipt(
    root: pathlib.Path, *, binding_path: pathlib.Path, index: int,
    games: Sequence[Mapping[str, Any]], completed_at_utc: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_shard_index(index)
    claim_path = _claim_path(root, index)
    receipt_path = _receipt_path(root, index)
    if not claim_path.exists():
        raise QualificationError("cannot finish a shard before its start claim")
    if receipt_path.exists():
        return validate_shard_receipt(receipt_path, binding_path=binding_path, index=index)
    claim = load_sealed(claim_path, SHARD_CLAIM_SCHEMA)
    binding_sha = _binding_digest(binding_path)
    if claim.get("binding_sha256") != binding_sha:
        raise QualificationError("shard claim binding changed")
    normalized = _normalize_shard_games(list(games), index)
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != {"path", "sha256"}
        or not pathlib.Path(str(evidence.get("path"))).is_file()
        or sha256_file(pathlib.Path(str(evidence["path"])))
        != evidence.get("sha256")
    ):
        raise QualificationError("shard receipt lacks immutable raw gate evidence")
    return write_sealed(receipt_path, {
        "schema": SHARD_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "binding_sha256": binding_sha,
        "claim": artifact_reference(claim_path, SHARD_CLAIM_SCHEMA),
        "shard_index": index,
        "pair_begin": index * PAIRS_PER_SHARD,
        "pair_count": PAIRS_PER_SHARD,
        "games": normalized,
        "evidence": dict(evidence),
        "completed_at_utc": completed_at_utc,
    })


def validate_shard_receipt(
    receipt_path: pathlib.Path, *, binding_path: pathlib.Path, index: int
) -> dict[str, Any]:
    receipt = load_sealed(receipt_path, SHARD_RECEIPT_SCHEMA)
    claim_path = _claim_path(receipt_path.parents[1], index)
    if (receipt.get("binding_sha256") != _binding_digest(binding_path)
            or receipt.get("shard_index") != index
            or receipt.get("pair_begin") != index * PAIRS_PER_SHARD
            or receipt.get("pair_count") != PAIRS_PER_SHARD
            or receipt.get("claim") != artifact_reference(claim_path, SHARD_CLAIM_SCHEMA)):
        raise QualificationError("shard receipt identity mismatch")
    evidence = receipt.get("evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"path", "sha256"}
        or not pathlib.Path(str(evidence.get("path"))).is_file()
        or sha256_file(pathlib.Path(str(evidence["path"])))
        != evidence.get("sha256")
    ):
        raise QualificationError("shard raw gate evidence changed")
    _normalize_shard_games(receipt.get("games"), index)
    return receipt


def aggregate_final(
    root: pathlib.Path, *, binding_path: pathlib.Path,
    uncontended_timing: Mapping[str, Any], completed_at_utc: str,
) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    for index in range(FINAL_SHARDS):
        claim = _claim_path(root, index)
        receipt = _receipt_path(root, index)
        if not claim.exists():
            raise QualificationError(f"final shard {index} was never started")
        if not receipt.exists():
            raise SpentShardError(f"final shard {index} is spent without a receipt")
        games.extend(validate_shard_receipt(
            receipt, binding_path=binding_path, index=index
        )["games"])
    failures = Counter(game["failure"] for game in games if game["failure"] is not None)
    summary = {
        "games": len(games),
        "candidate_wins": sum(game["candidate_win"] for game in games),
        "candidate_color_wins": {
            str(color): sum(game["candidate_win"] and game["candidate_color"] == color
                            for game in games)
            for color in (0, 1)
        },
        "failures": {name: failures[name] for name in FAILURE_CATEGORIES},
        "maximum_turns": max(game["turns"] for game in games),
        "timing": {
            "first_max_ms": max(game["first_ms"] for game in games),
            "later_max_ms": max(game["later_max_ms"] for game in games),
        },
        "uncontended_timing": dict(uncontended_timing),
    }
    verdict = strict_gate_verdict(summary)
    return write_sealed(root / "aggregate.json", {
        "schema": FINAL_AGGREGATE_SCHEMA,
        "namespace": NAMESPACE,
        "binding": artifact_reference(binding_path, GATE_BINDING_SCHEMA),
        "completed_at_utc": completed_at_utc,
        "summary": summary,
        "verdict": verdict,
        "status": "rank4-qualified" if verdict["passed"] else "final-gate-failed",
    })


def create_upload_authorization(
    output: pathlib.Path, *, binding_path: pathlib.Path,
    aggregate_path: pathlib.Path, ci_record: Mapping[str, Any],
) -> dict[str, Any]:
    binding = load_sealed(binding_path, GATE_BINDING_SCHEMA)
    aggregate = load_sealed(aggregate_path, FINAL_AGGREGATE_SCHEMA)
    if aggregate.get("binding") != artifact_reference(binding_path, GATE_BINDING_SCHEMA):
        raise QualificationError("final aggregate uses another binding")
    if strict_gate_verdict(aggregate.get("summary", {}))["passed"] is not True:
        raise QualificationError("one-upload authorization requires a passing final gate")
    required_jobs = {
        "replay-training-contract", "leaderboard-contract", "test-gcc",
        "test-clang", "test-sanitizers",
    }
    jobs = ci_record.get("jobs") if isinstance(ci_record, Mapping) else None
    if (ci_record.get("conclusion") != "success"
            or ci_record.get("head_sha") != binding["candidate_commit"]
            or type(ci_record.get("run_id")) is not int
            or ci_record.get("workflow") != "CI and Pages"
            or ci_record.get("workflow_file") != "pages.yml"
            or ci_record.get("event") != "workflow_dispatch"
            or ci_record.get("head_branch") != "compact-value-bfm"
            or ci_record.get("head_ref") != "refs/heads/compact-value-bfm"
            or not isinstance(ci_record.get("url"), str)
            or not ci_record["url"].startswith("https://github.com/")
            or not isinstance(jobs, dict) or set(jobs) != required_jobs
            or any(value != "success" for value in jobs.values())):
        raise QualificationError("green CI does not bind the qualified commit")
    return write_sealed(output, {
        "schema": UPLOAD_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "uploads_authorized": 1,
        "rank4_replacement_authorized": False,
        "candidate_commit": binding["candidate_commit"],
        "candidate": binding["candidate"],
        "binding": artifact_reference(binding_path, GATE_BINDING_SCHEMA),
        "aggregate": artifact_reference(aggregate_path, FINAL_AGGREGATE_SCHEMA),
        "ci": dict(ci_record),
        "upload_ledger_root": str(output.parent.resolve()),
    })


def _event_path(root: pathlib.Path, name: str) -> pathlib.Path:
    return root / "upload" / name


def _load_authorization(path: pathlib.Path) -> dict[str, Any]:
    authorization = load_sealed(path, UPLOAD_AUTH_SCHEMA)
    if authorization.get("upload_ledger_root") != str(path.parent.resolve()):
        raise QualificationError("upload authorization was copied to another ledger root")
    return authorization


def _require_upload_root(root: pathlib.Path, authorization_path: pathlib.Path) -> None:
    if root.resolve() != authorization_path.parent.resolve():
        raise QualificationError("upload ledger root differs from its authorization")


def _require_event(root: pathlib.Path, name: str, status: str | None = None) -> dict[str, Any]:
    event = load_sealed(_event_path(root, name), UPLOAD_EVENT_SCHEMA)
    if status is not None and event.get("status") != status:
        raise QualificationError(f"upload event {name} has unexpected status")
    return event


def prepare_upload(
    root: pathlib.Path, *, authorization_path: pathlib.Path,
    created_at_utc: str, fresh_editor: bool,
) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    authorization = _load_authorization(authorization_path)
    _utc(created_at_utc, "upload preparation time")
    if fresh_editor is not True:
        raise QualificationError("upload preparation requires a fresh editor")
    return write_sealed(_event_path(root, "00-prepared.json"), {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "prepared", "created_at_utc": created_at_utc,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
        "candidate": authorization["candidate"],
        "fresh_editor": True,
    })


def attest_editor_copyback(
    root: pathlib.Path, *, authorization_path: pathlib.Path,
    generated_source: pathlib.Path, copied_back_source: pathlib.Path,
    created_at_utc: str,
) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    prepared = _require_event(root, "00-prepared.json", "prepared")
    authorization = _load_authorization(authorization_path)
    if _utc(created_at_utc, "copy-back time") < _utc(
        prepared.get("created_at_utc"), "preparation time"
    ):
        raise QualificationError("editor copy-back predates upload preparation")
    if prepared.get("authorization") != artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA):
        raise QualificationError("prepared upload uses another authorization")
    if generated_source.resolve() == copied_back_source.resolve():
        raise QualificationError("editor copy-back must be a distinct file")
    generated = generated_source.read_bytes()
    copied = copied_back_source.read_bytes()
    try:
        generated.decode("ascii")
        copied.decode("ascii")
    except UnicodeDecodeError as error:
        raise QualificationError("editor sources must be ASCII") from error
    if generated != copied:
        raise QualificationError("editor copy-back differs from qualified source")
    if (sha256_bytes(generated) != authorization["candidate"]["sha256"]
            or len(generated) != authorization["candidate"]["bytes"]):
        raise QualificationError("editor source differs from upload authorization")
    return write_sealed(_event_path(root, "01-editor-copyback.json"), {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "editor-copyback-verified", "created_at_utc": created_at_utc,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
        "generated_path": str(generated_source.resolve()),
        "copyback_path": str(copied_back_source.resolve()),
        "source_sha256": sha256_bytes(generated), "source_bytes": len(generated),
        "api_source_readable": False,
    })


def record_play(
    root: pathlib.Path, *, authorization_path: pathlib.Path,
    legal_stdout: bool, expected_telemetry: bool, created_at_utc: str,
) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    copyback = _require_event(
        root, "01-editor-copyback.json", "editor-copyback-verified"
    )
    _load_authorization(authorization_path)
    if _utc(created_at_utc, "Play time") < _utc(
        copyback.get("created_at_utc"), "copy-back time"
    ):
        raise QualificationError("Play predates editor copy-back")
    passed = legal_stdout is True and expected_telemetry is True
    return write_sealed(_event_path(root, "02-play.json"), {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "play-passed" if passed else "play-failed",
        "created_at_utc": created_at_utc,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
        "legal_stdout": legal_stdout is True,
        "expected_telemetry": expected_telemetry is True,
    })


def start_submit(root: pathlib.Path, *, authorization_path: pathlib.Path, started_at_utc: str) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    _load_authorization(authorization_path)
    play = _require_event(root, "02-play.json", "play-passed")
    if _utc(started_at_utc, "Submit start time") < _utc(
        play.get("created_at_utc"), "Play time"
    ):
        raise QualificationError("Submit predates Play")
    path = _event_path(root, "03-submit-started.json")
    if path.exists():
        raise QualificationError("Submit has already been started; never click again")
    return write_sealed(path, {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "submit-started", "started_at_utc": started_at_utc,
        "one_shot": True,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
    })


def record_submit_ambiguous(
    root: pathlib.Path, *, authorization_path: pathlib.Path,
    observed_at_utc: str, evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    _load_authorization(authorization_path)
    started = _require_event(root, "03-submit-started.json", "submit-started")
    if _utc(observed_at_utc, "ambiguous Submit time") < _utc(
        started.get("started_at_utc"), "Submit start time"
    ):
        raise QualificationError("ambiguous Submit observation predates Submit")
    if _event_path(root, "05-submission-attested.json").exists():
        raise QualificationError("submission is already attested")
    return write_sealed(_event_path(root, "04-submit-ambiguous.json"), {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "submit-ambiguous", "observed_at_utc": observed_at_utc,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
        "evidence": dict(evidence), "submit_must_not_be_clicked_again": True,
    })


def attest_submission(
    root: pathlib.Path, *, authorization_path: pathlib.Path,
    agent_id: int, submission_id: int, submitted_at_utc: str,
    ambiguity_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_upload_root(root, authorization_path)
    authorization = _load_authorization(authorization_path)
    started = _require_event(root, "03-submit-started.json", "submit-started")
    lower_bound = _utc(started.get("started_at_utc"), "Submit start time")
    final_path = _event_path(root, "05-submission-attested.json")
    if final_path.exists():
        raise QualificationError("submission is already attested")
    ambiguous_path = _event_path(root, "04-submit-ambiguous.json")
    if ambiguous_path.exists():
        ambiguous = _require_event(
            root, "04-submit-ambiguous.json", "submit-ambiguous"
        )
        lower_bound = _utc(
            ambiguous.get("observed_at_utc"), "ambiguous Submit time"
        )
        if not isinstance(ambiguity_resolution, Mapping):
            raise QualificationError("ambiguous Submit requires history/API resolution evidence")
        if (ambiguity_resolution.get("matching_submissions") != 1
                or ambiguity_resolution.get("agent_id") != agent_id
                or ambiguity_resolution.get("submission_id") != submission_id):
            raise QualificationError("ambiguous Submit resolution is not unique and identity-bound")
    elif ambiguity_resolution is not None:
        raise QualificationError("ambiguity resolution supplied without an ambiguous Submit")
    if _utc(submitted_at_utc, "submission attestation time") < lower_bound:
        raise QualificationError("submission attestation predates its upload evidence")
    return write_sealed(final_path, {
        "schema": UPLOAD_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "submission-attested", "submitted_at_utc": submitted_at_utc,
        "authorization": artifact_reference(authorization_path, UPLOAD_AUTH_SCHEMA),
        "candidate_commit": authorization["candidate_commit"],
        "source_sha256": authorization["candidate"]["sha256"],
        "source_bytes": authorization["candidate"]["bytes"],
        "agent_id": _positive_int(agent_id, "agent ID"),
        "submission_id": _positive_int(submission_id, "submission ID"),
        "ambiguity_resolution": None if ambiguity_resolution is None else dict(ambiguity_resolution),
        "submit_clicks": 1,
    })


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_detail_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise QualificationError("battle metadata contains a non-string key")
            if _canonical_key(key) in FORBIDDEN_BATTLE_KEYS:
                raise QualificationError(f"battle metadata contains forbidden detail field: {key}")
            _reject_detail_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_detail_fields(child)


def classify_matching_window(
    battles: Any, *, agent_id: int, submission_id: int
) -> dict[str, Any]:
    _positive_int(agent_id, "agent ID")
    _positive_int(submission_id, "submission ID")
    if not isinstance(battles, list):
        raise QualificationError("battle metadata must be a list")
    _reject_detail_fields(battles)
    complete: list[int] = []
    pending: list[int] = []
    seen: set[int] = set()
    for index, battle in enumerate(battles):
        if not isinstance(battle, Mapping):
            raise QualificationError(f"battle {index} is not an object")
        game_id = battle.get("gameId")
        if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id <= 0 or game_id in seen:
            raise QualificationError("battle metadata has an invalid/repeated game ID")
        seen.add(game_id)
        players = battle.get("players")
        if not isinstance(players, list):
            raise QualificationError("battle players must be a list")
        focus = [player for player in players if isinstance(player, Mapping)
                 and player.get("playerAgentId") == agent_id]
        if len(focus) > 1:
            raise QualificationError("battle repeats the focus agent")
        if not focus or focus[0].get("submissionId") != submission_id:
            continue
        if battle.get("done") is True and len(players) == 2:
            complete.append(game_id)
        else:
            pending.append(game_id)
    matching = len(complete) + len(pending)
    if matching > EXACT_LIVE_GAMES:
        raise QualificationError("matching submission window exceeds exactly 90 games")
    ready = len(complete) == EXACT_LIVE_GAMES and not pending
    return {
        "status": "ready" if ready else "waiting",
        "expected_games": EXACT_LIVE_GAMES,
        "matching_games": matching,
        "complete_games": len(complete),
        "pending_games": len(pending),
        "game_ids": sorted(complete),
        "agent_id": agent_id,
        "submission_id": submission_id,
        "collector_permitted": ready,
    }


def finalize_live_window(
    output: pathlib.Path, *, battles: Any, collector_manifest: Mapping[str, Any],
    submission_attestation_path: pathlib.Path,
) -> dict[str, Any]:
    attestation = load_sealed(submission_attestation_path, UPLOAD_EVENT_SCHEMA)
    if attestation.get("status") != "submission-attested":
        raise QualificationError("live window requires a submitted identity")
    report = classify_matching_window(
        battles, agent_id=attestation["agent_id"],
        submission_id=attestation["submission_id"],
    )
    if not report["collector_permitted"]:
        raise QualificationError(
            f"exact live window is incomplete: {report['complete_games']}/90"
        )
    expected = {
        "agent_id": attestation["agent_id"],
        "submission_id": attestation["submission_id"],
        "source_sha256": attestation["source_sha256"],
        "repository_commit": attestation["candidate_commit"],
        "game_ids": report["game_ids"],
    }
    if any(collector_manifest.get(key) != value for key, value in expected.items()):
        raise QualificationError("collector manifest contradicts the exact live identity")
    own_failures = collector_manifest.get("focus_operational_failures")
    opponent_failures = collector_manifest.get("opponent_operational_failures")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (own_failures, opponent_failures)):
        raise QualificationError("collector operational counts are invalid")
    return write_sealed(output, {
        "schema": LIVE_WINDOW_SCHEMA, "namespace": NAMESPACE,
        "status": "complete-accepted" if own_failures == 0 else "complete-rejected-own-failure",
        "exact_games": EXACT_LIVE_GAMES,
        "identity": expected,
        "submission_attestation": artifact_reference(
            submission_attestation_path, UPLOAD_EVENT_SCHEMA
        ),
        "focus_operational_failures": own_failures,
        "opponent_operational_failures": opponent_failures,
        "opponent_failures_count_as_strength_wins": False,
        "training_eligible": False,
    })


def _json_file(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate-check")
    gate.add_argument("--summary", type=pathlib.Path, required=True)
    status = commands.add_parser("window-status")
    status.add_argument("--battles", type=pathlib.Path, required=True)
    status.add_argument("--agent-id", type=int, required=True)
    status.add_argument("--submission-id", type=int, required=True)
    start = commands.add_parser("start-shard")
    start.add_argument("--root", type=pathlib.Path, required=True)
    start.add_argument("--binding", type=pathlib.Path, required=True)
    start.add_argument("--index", type=int, required=True)
    start.add_argument("--started-at-utc", required=True)
    finish = commands.add_parser("finish-shard")
    finish.add_argument("--root", type=pathlib.Path, required=True)
    finish.add_argument("--binding", type=pathlib.Path, required=True)
    finish.add_argument("--index", type=int, required=True)
    finish.add_argument("--games", type=pathlib.Path, required=True)
    finish.add_argument("--evidence", type=pathlib.Path, required=True)
    finish.add_argument("--completed-at-utc", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "gate-check":
            result = strict_gate_verdict(_json_file(args.summary))
            code = 0 if result["passed"] else 10
        elif args.command == "window-status":
            result = classify_matching_window(
                _json_file(args.battles), agent_id=args.agent_id,
                submission_id=args.submission_id,
            )
            code = 0 if result["collector_permitted"] else 2
        elif args.command == "start-shard":
            result = start_final_shard(
                args.root, binding_path=args.binding, index=args.index,
                started_at_utc=args.started_at_utc,
            )
            code = 0
        else:
            result = record_shard_receipt(
                args.root, binding_path=args.binding, index=args.index,
                games=_json_file(args.games), completed_at_utc=args.completed_at_utc,
                evidence=artifact_reference(args.evidence),
            )
            code = 0
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return code
    except (QualificationError, OSError, json.JSONDecodeError) as error:
        print(f"compact qualification failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
