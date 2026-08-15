#!/usr/bin/env python3
"""Fail closed if the jacek_arena_bfm runtime can consume prior lineage.

The checker examines a fixed allowlist inside this bot directory.  It does not
walk results, protected evidence, another bot directory, or any corpus.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import pathlib
import re
from typing import Any


BOT_DIRECTORY = pathlib.Path(__file__).resolve().parent
RUNTIME_FILES = (
    "model.hpp",
    "engine.hpp",
    "engine.cpp",
    "bot.cpp",
    "sources.txt",
    "submission.json",
    "submission.cpp",
)
EXPECTED_SOURCES = (
    "submissions/codingame/bots/jacek_arena_bfm/model.hpp",
    "submissions/codingame/bots/jacek_arena_bfm/engine.hpp",
    "submissions/codingame/bots/jacek_arena_bfm/engine.cpp",
    "submissions/codingame/bots/jacek_arena_bfm/bot.cpp",
)
ALLOWED_LOCAL_INCLUDES = frozenset({"engine.hpp", "model.hpp"})
FORBIDDEN_RUNTIME_PATTERNS = (
    r"submissions/codingame/bots/(?!jacek_arena_bfm(?:/|\b))",
    r"(?:^|[\"'/])(?:results|models|replays|corpus|evidence)/",
    r"matches\.json",
    r"arena_batch",
    r"jacek_native_bfm",
    r"rank_[45](?:\b|_)",
    r"selfplay_nn",
    r"jacek_nn",
    r"neural_puct",
)


class PurityError(RuntimeError):
    pass


def _read_ascii(path: pathlib.Path) -> str:
    if path.is_symlink():
        raise PurityError(f"runtime file may not be a symlink: {path.name}")
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise PurityError(f"missing runtime file: {path.name}") from error
    if any(byte > 0x7F for byte in raw):
        raise PurityError(f"runtime file is not ASCII: {path.name}")
    return raw.decode("ascii")


def _without_packed_payloads(text: str) -> str:
    # Packed base64 is opaque model data.  Remove only those three declared
    # arrays; ordinary runtime string literals remain visible to the audit.
    return re.sub(
        r"\bkW[123]Packed\[\]\s*=\s*(?:\s*\"[A-Za-z0-9+/=]*\")+\s*;",
        "kWPacked[] = \"\";",
        text,
    )


def _check_forbidden_references(name: str, text: str) -> None:
    inspectable = _without_packed_payloads(text)
    for pattern in FORBIDDEN_RUNTIME_PATTERNS:
        if re.search(pattern, inspectable, flags=re.IGNORECASE | re.MULTILINE):
            raise PurityError(f"{name} contains forbidden runtime reference matching {pattern!r}")


def _integer_constant(text: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*(\d+)(?:ULL|U|L)?\s*;", text)
    if match is None:
        raise PurityError(f"model is missing integer constant {name}")
    return int(match.group(1))


def _packed_value(text: str, name: str) -> bytes:
    match = re.search(
        rf"\b{name}\[\]\s*=\s*((?:\s*\"[A-Za-z0-9+/=]*\")+)\s*;",
        text,
    )
    if match is None:
        raise PurityError(f"model is missing packed array {name}")
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]*)"', match.group(1)))
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise PurityError(f"model packed array {name} is not canonical base64") from error


def _check_model(text: str) -> dict[str, Any]:
    inputs = _integer_constant(text, "kInputSize")
    hidden1 = _integer_constant(text, "kHidden1Size")
    hidden2 = _integer_constant(text, "kHidden2Size")
    if inputs != 1156 or hidden1 not in {32, 48, 64} or hidden2 != 32:
        raise PurityError("model shape is outside the clean campaign contract")
    expected = {
        "kW1Packed": _integer_constant(text, "kW1Count"),
        "kW2Packed": _integer_constant(text, "kW2Count"),
        "kW3Packed": _integer_constant(text, "kW3Count"),
    }
    required_counts = {
        "kW1Packed": hidden1 * inputs,
        "kW2Packed": hidden2 * hidden1,
        "kW3Packed": hidden2,
    }
    decoded_hashes: dict[str, str] = {}
    for name, declared in expected.items():
        if declared != required_counts[name]:
            raise PurityError(f"{name} declared count does not match the bias-free shape")
        decoded = _packed_value(text, name)
        if len(decoded) != declared:
            raise PurityError(f"{name} decoded byte count does not match its declaration")
        decoded_hashes[name] = hashlib.sha256(decoded).hexdigest()
    identity = re.search(r'\bkIdentity\[\]\s*=\s*"([^"]+)"\s*;', text)
    if identity is None or not re.fullmatch(
        r"fresh-(?:32|48|64)x32-s\d+-[0-9a-f]{12}", identity.group(1)
    ):
        raise PurityError("model identity is not a fresh random-initialized identity")
    for forbidden in ("bias", "checkpoint", "resume", "warm_start"):
        if re.search(rf"\bk\w*{forbidden}\w*\b", text, flags=re.IGNORECASE):
            raise PurityError(f"model exposes forbidden {forbidden} lineage")
    return {
        "identity": identity.group(1),
        "shape": [inputs, hidden1, hidden2, 1],
        "packed_sha256": decoded_hashes,
    }


def validate_runtime(bot_directory: pathlib.Path = BOT_DIRECTORY) -> dict[str, Any]:
    directory = bot_directory.resolve()
    texts = {name: _read_ascii(directory / name) for name in RUNTIME_FILES}
    for name, text in texts.items():
        _check_forbidden_references(name, text)

    source_lines = tuple(
        line.strip() for line in texts["sources.txt"].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if source_lines != EXPECTED_SOURCES:
        raise PurityError("sources.txt must contain only the four clean-room runtime files")

    try:
        config = json.loads(texts["submission.json"])
    except json.JSONDecodeError as error:
        raise PurityError("submission.json is invalid") from error
    if config.get("schema") != "papersoccer.codingame-submission.v1":
        raise PurityError("submission schema is not approved")
    if set(config.get("allowed_local_includes", ())) != ALLOWED_LOCAL_INCLUDES:
        raise PurityError("submission allows an external local include")
    if config.get("generators", []) != []:
        raise PurityError("runtime generation may not invoke an external data generator")
    if config.get("source_limit") != 99999:
        raise PurityError("submission source limit must be exactly 99,999")

    for name in ("engine.hpp", "engine.cpp", "bot.cpp"):
        includes = set(re.findall(r'^\s*#include\s*"([^"]+)"', texts[name], re.MULTILINE))
        if not includes.issubset(ALLOWED_LOCAL_INCLUDES):
            raise PurityError(f"{name} imports a non-clean local header")

    submission = texts["submission.cpp"]
    if len(submission) > 99999:
        raise PurityError("generated submission exceeds 99,999 characters")
    if re.search(r'^\s*#include\s*"', submission, flags=re.MULTILINE):
        raise PurityError("generated submission retains a local include")
    model = _check_model(texts["model.hpp"])
    if model["identity"] not in submission:
        raise PurityError("generated submission does not embed the selected fresh model")
    if "jacek_arena_bfm" not in submission:
        raise PurityError("generated submission does not retain the clean namespace")
    return {
        "valid": True,
        "namespace": "jacek_arena_bfm",
        "model": model,
        "submission": {
            "bytes": len(submission.encode("ascii")),
            "sha256": hashlib.sha256(submission.encode("ascii")).hexdigest(),
        },
        "scope": list(RUNTIME_FILES),
        "protected_content_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-directory", type=pathlib.Path, default=BOT_DIRECTORY)
    arguments = parser.parse_args()
    try:
        report = validate_runtime(arguments.bot_directory)
    except PurityError as error:
        parser.exit(1, f"jacek_arena_bfm purity failure: {error}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
