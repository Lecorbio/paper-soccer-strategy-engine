#!/usr/bin/env python3
"""Atomically build or verify the self-contained compact submission source."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIG = HERE / "submission.json"


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def contained(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    if not relative or pathlib.PurePath(relative).is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    return resolved


def render(*, model_header: bytes | None = None) -> tuple[pathlib.Path, bytes]:
    config = json.loads(CONFIG.read_text())
    if config.get("schema") != "papersoccer.codingame-submission.v1":
        raise ValueError("invalid submission schema")
    manifest = contained(HERE, config.get("sources", "sources.txt"), "sources")
    output_path = contained(HERE, config.get("output", "submission.cpp"), "output")
    allowed = set(config.get("allowed_local_includes", []))
    sources = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not sources:
        raise ValueError("empty source manifest")
    system_headers: set[str] = set()
    bodies: list[str] = []
    for relative in sources:
        source = contained(ROOT, relative, "source")
        source_text = (
            model_header.decode("ascii")
            if model_header is not None and relative ==
            "submissions/codingame/bots/compact_value_bfm/model.hpp"
            else source.read_text()
        )
        kept: list[str] = []
        for line in source_text.replace("\r\n", "\n").split("\n"):
            if line.strip() == "#pragma once":
                continue
            system = re.fullmatch(r"\s*#include\s*<([^>]+)>\s*", line)
            if system:
                system_headers.add(system.group(1))
                continue
            local = re.fullmatch(r'\s*#include\s*"([^"]+)"\s*', line)
            if local:
                if local.group(1) not in allowed:
                    raise ValueError(f"unexpected local include {local.group(1)}")
                continue
            if line.lstrip().startswith("//"):
                continue
            kept.append(line.lstrip() if config.get("strip_leading_whitespace") else line)
        while kept and not kept[0].strip():
            kept.pop(0)
        while kept and not kept[-1].strip():
            kept.pop()
        bodies.append("\n".join(kept))
    banner = "\n#if defined(__GNUG__) && !defined(__clang__)\n#pragma GCC optimize(\"O3\")\n#endif\n"
    includes = "\n".join(f"#include <{header}>" for header in sorted(system_headers))
    readable = f"{banner}{includes}\n\n" + "\n\n".join(bodies) + "\n"
    compact: list[str] = []
    for line in readable.splitlines():
        if not line.strip() and compact and not compact[-1].strip():
            continue
        compact.append(line)
    payload = ("\n".join(compact) + "\n").encode("ascii")
    limit = int(config.get("source_limit", 95_000))
    if len(payload) > limit:
        raise ValueError(f"generated source has {len(payload)} characters; limit is {limit}")
    if re.search(rb'^\s*#include\s*"', payload, re.MULTILINE):
        raise ValueError("generated source retains a local include")
    return output_path, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-header", type=pathlib.Path)
    source.add_argument("--runtime", type=pathlib.Path)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--render-output", type=pathlib.Path)
    arguments = parser.parse_args()
    model_header: bytes | None = None
    runtime_metadata: dict[str, Any] | None = None
    if arguments.model_header:
        model_header = arguments.model_header.read_bytes()
        model_header.decode("ascii")
    elif arguments.runtime:
        import export_model
        model_header, runtime_metadata = export_model.render_header(arguments.runtime)
    output, payload = render(model_header=model_header)
    if arguments.measure:
        report: dict[str, Any] = {
            "ascii_characters": len(payload),
            "eligible": len(payload) <= 95_000,
            "limit": 95_000,
        }
        if runtime_metadata is not None:
            report["architecture"] = runtime_metadata["architecture"]
            report["runtime_body_sha256"] = runtime_metadata["body_sha256"]
        print(json.dumps(report, sort_keys=True))
        return 0 if report["eligible"] else 1
    if arguments.render_output:
        atomic_write(arguments.render_output, payload)
        print(f"wrote measured compact source ({len(payload)} ASCII characters)")
        return 0
    if arguments.check:
        if model_header is not None:
            raise SystemExit("--check cannot be combined with a model override")
        if not output.exists() or output.read_bytes() != payload:
            raise SystemExit(f"{output} is stale")
    else:
        atomic_write(output, payload)
    print(f"compact submission current ({len(payload)} ASCII characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
