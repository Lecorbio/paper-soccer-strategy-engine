#!/usr/bin/env python3
"""Small content-addressing helpers for campaign-owned artifacts.

The helpers deliberately never replace an existing path.  A matching blob is a
successful idempotent write; a hash collision or a manually changed file is an
error.  This keeps raw, normalized, corpus, model, and manifest artifacts
immutable without depending on the repository's older evidence stores.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 RFC-8259 JSON with a deterministic byte representation."""

    return (json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path | str, payload: bytes) -> Path:
    """Create *path* exactly once, accepting an identical existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = destination.read_bytes()
        if existing != payload:
            raise ValueError(f"immutable path already contains different bytes: {destination}")
        return destination

    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Only the just-created incomplete file can be removed here.  Existing
        # evidence is never opened for writing and is never replaced.
        destination.unlink(missing_ok=True)
        raise
    return destination


def write_content_addressed_bytes(
    directory: Path | str,
    payload: bytes,
    suffix: str,
) -> Path:
    if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
        raise ValueError("suffix must be a simple extension beginning with '.'")
    digest = sha256_bytes(payload)
    return write_immutable(Path(directory) / f"{digest}{suffix}", payload)


def write_content_addressed_json(directory: Path | str, value: Any) -> Path:
    return write_content_addressed_bytes(directory, canonical_json_bytes(value), ".json")


def verify_content_addressed_path(path: Path | str) -> str:
    artifact = Path(path)
    expected = artifact.name.split(".", 1)[0]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"not a lowercase SHA-256 content-addressed name: {artifact.name}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise ValueError(f"content hash mismatch for {artifact}: expected {expected}, got {actual}")
    return actual


def file_inventory(paths: Iterable[Path | str], *, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Build a stable byte/count/hash inventory without reading semantic content."""

    base = Path(root).resolve() if root is not None else None
    inventory: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if base is not None:
            try:
                display = str(path.relative_to(base))
            except ValueError as error:
                raise ValueError(f"artifact is outside inventory root: {path}") from error
        else:
            display = str(path)
        payload = path.read_bytes()
        inventory.append({
            "path": display,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        })
    return sorted(inventory, key=lambda row: row["path"])

