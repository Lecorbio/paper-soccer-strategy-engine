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
OPERATOR_CHARACTERS = frozenset("+-*/%<>=!&|^~?.:#")
PRIVATE_IDENTIFIERS = (
    "config_", "nodes_", "stats_", "output_", "root_", "deadline_",
    "model_", "emergency_", "hidden_one_", "hidden_two_", "scale_one_",
    "scale_two_", "scale_three_", "payload_sha256_", "weights_", "xs_",
    "ys_", "arcs_", "degrees_", "rotated_vertices_", "rotated_edges_",
    "boundaries_", "next_order_", "cap_dropped_", "BfmSearch",
    "TurnGenerator", "SplitMix64", "Partial", "Node", "RootTranscript",
    "CachedValue", "append", "set_edge", "valid_sha256", "rotate_right",
    "base64_value", "decode_base64", "canonical_direction",
    "canonical_action_less", "sort_canonical_arcs", "ordered_arcs",
    "analyze_turn", "exact_goal_path", "classify", "boundary_coordinate",
    "permitted_edge", "direction_for", "retain_goal", "retain_witnesses",
    "remember_expansion", "reuse_expansion", "reuse_slot",
    "selectable_child_count", "evaluate_child", "budget_available",
    "select_descendant", "select_path", "first_child", "child_index",
    "child_perspective", "selected_score", "selected_order",
    "cache_", "reuse_", "traversal_closed",
)
PRIVATE_ALIASES = {
    identifier: f"{prefix}{suffix}"
    for identifier, (prefix, suffix) in zip(
        PRIVATE_IDENTIFIERS,
        ((prefix, suffix) for prefix in "zyx" for suffix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        strict=False,
    )
}


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


def _needs_token_space(previous: str, current: str) -> bool:
    """Return whether deleting whitespace would merge two C++ tokens."""
    if ((previous.isalnum() or previous == "_") and
            (current.isalnum() or current == "_")):
        return True
    if previous in OPERATOR_CHARACTERS and current in OPERATOR_CHARACTERS:
        return True
    return ((previous == "." and current.isdigit()) or
            (previous.isdigit() and current == "."))


def compact_cpp_code(source: str) -> str:
    """Deterministically remove non-semantic C++ comments and whitespace.

    Preprocessor directives remain on dedicated lines. String and character
    literals are copied byte-for-byte, while whitespace between ordinary code
    tokens is retained only where deleting it could change tokenization.
    """
    parts: list[str] = []
    code: list[str] = []

    def flush_code() -> None:
        if not code:
            return
        text = "\n".join(code)
        output: list[str] = []
        pending_space = False
        index = 0
        while index < len(text):
            current = text[index]
            following = text[index + 1] if index + 1 < len(text) else ""
            if current.isspace():
                pending_space = True
                index += 1
                continue
            if current == "/" and following == "/":
                newline = text.find("\n", index + 2)
                index = len(text) if newline < 0 else newline + 1
                pending_space = True
                continue
            if current == "/" and following == "*":
                closing = text.find("*/", index + 2)
                if closing < 0:
                    raise ValueError("unterminated C++ block comment")
                index = closing + 2
                pending_space = True
                continue
            if pending_space and output and _needs_token_space(output[-1][-1], current):
                output.append(" ")
            pending_space = False
            numeric_separator = (
                current == "'" and index > 0 and text[index - 1].isdigit() and
                following.isdigit()
            )
            if current in {'"', "'"} and not numeric_separator:
                quote = current
                literal = [current]
                index += 1
                escaped = False
                while index < len(text):
                    character = text[index]
                    literal.append(character)
                    index += 1
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == quote:
                        break
                else:
                    raise ValueError("unterminated C++ literal")
                output.append("".join(literal))
                continue
            output.append(current)
            index += 1
        compacted = "".join(output)
        if compacted:
            parts.append(compacted)
        code.clear()

    for line in source.replace("\r\n", "\n").split("\n"):
        if line.lstrip().startswith("#"):
            flush_code()
            parts.append(line.strip())
        else:
            code.append(line)
    flush_code()
    return "\n".join(part for part in parts if part) + "\n"


def minify_private_identifiers(source: str) -> str:
    """Shorten only deployment-private C++ names without touching literals."""
    output: list[str] = []
    present: set[str] = set()
    index = 0
    while index < len(source):
        current = source[index]
        numeric_separator = (
            current == "'" and index > 0 and source[index - 1].isdigit()
            and index + 1 < len(source) and source[index + 1].isdigit()
        )
        if current in {'"', "'"} and not numeric_separator:
            quote = current
            begin = index
            index += 1
            escaped = False
            while index < len(source):
                character = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    break
            else:
                raise ValueError("unterminated C++ literal during identifier minification")
            output.append(source[begin:index])
            continue
        if current.isalpha() or current == "_":
            end = index + 1
            while end < len(source) and (
                source[end].isalnum() or source[end] == "_"
            ):
                end += 1
            identifier = source[index:end]
            present.add(identifier)
            output.append(PRIVATE_ALIASES.get(identifier, identifier))
            index = end
            continue
        output.append(current)
        index += 1
    collisions = set(PRIVATE_ALIASES.values()) & (present - set(PRIVATE_ALIASES))
    if collisions:
        raise ValueError(f"private identifier alias collision: {sorted(collisions)}")
    return "".join(output)


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
    payload = minify_private_identifiers(compact_cpp_code(readable)).encode("ascii")
    limit = int(config.get("source_limit", 95_000))
    if len(payload) >= limit:
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
        reserve = 95_000 - len(payload)
        reserve_target = int(json.loads(CONFIG.read_text()).get(
            "source_reserve_target", 0))
        report: dict[str, Any] = {
            "ascii_characters": len(payload),
            "eligible": len(payload) < 95_000,
            "limit": 95_000,
            "reserve": reserve,
            "reserve_target": reserve_target,
            "reserve_target_met": reserve >= reserve_target,
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
