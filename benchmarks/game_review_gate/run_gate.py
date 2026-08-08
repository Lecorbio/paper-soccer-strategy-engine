#!/usr/bin/env python3
"""Validate, run, aggregate, lock, and report the frozen Game Review gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.game_review_gate import gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/game_review_gate/manifest.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser(
        "validate", help="validate frozen identities and every available artifact"
    )
    validate_parser.add_argument("--opening-tool", type=pathlib.Path)
    validate_parser.add_argument(
        "--verify-regeneration",
        action="store_true",
        help="regenerate all twelve banks in frozen order and require byte identity",
    )
    validate_parser.add_argument("--require-complete", action="store_true")

    identities_parser = commands.add_parser(
        "opening-identities", help="render the deterministic opening identity artifact"
    )
    identities_parser.add_argument("--write", action="store_true")

    run_parser = commands.add_parser(
        "run", help="run or resume deterministic arena units for one phase"
    )
    run_parser.add_argument("--phase", choices=gate.PHASES, required=True)
    run_parser.add_argument(
        "--arena",
        type=pathlib.Path,
        default=pathlib.Path("build/release/papersoccer_arena"),
    )
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)

    aggregate_parser = commands.add_parser(
        "aggregate", help="validate all phase shards and write the compact phase result"
    )
    aggregate_parser.add_argument("--phase", choices=gate.PHASES, required=True)

    latency_parser = commands.add_parser(
        "record-latency", help="validate and promote an independently measured Wasm result"
    )
    latency_parser.add_argument("--input", type=pathlib.Path, required=True)

    commands.add_parser(
        "lock-selection",
        help="select the validation profile and freeze its validation-only calibration",
    )
    render_parser = commands.add_parser(
        "render-lock",
        help="render the C++ profile/calibration lock from frozen selection evidence",
    )
    render_parser.add_argument(
        "--latency-input",
        type=pathlib.Path,
        help="use a draft latency artifact while stabilizing the linked Wasm module",
    )
    check_parser = commands.add_parser(
        "check-lock",
        help="require the C++ profile/calibration lock to match selection evidence",
    )
    check_parser.add_argument(
        "--latency-input",
        type=pathlib.Path,
        help="check against a draft latency artifact before the official lock exists",
    )
    commands.add_parser(
        "report", help="write the compact frozen result and concise evidence report"
    )

    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    context = gate.validate_manifest(
        manifest_path,
        verify_identities=args.command != "opening-identities",
    )

    if args.command == "validate":
        if args.verify_regeneration and args.opening_tool is None:
            raise gate.GateError("--verify-regeneration requires --opening-tool")
        if args.opening_tool is not None:
            opening_tool = args.opening_tool
            if not opening_tool.is_absolute():
                opening_tool = context.repository / opening_tool
            gate.run_opening_tool_validation(context, opening_tool.resolve())
            if args.verify_regeneration:
                gate.verify_opening_regeneration(context, opening_tool.resolve())
        result = gate.validate_available_artifacts(
            context, require_complete=args.require_complete
        )
    elif args.command == "opening-identities":
        value = gate.build_opening_identities(context)
        if args.write:
            path = context.repository / context.manifest["outputs"][
                "opening_identities"
            ]
            resumed = gate.write_json(path, value)
            result = {
                "opening_identities": str(path.relative_to(context.repository)),
                "sha256": gate.sha256_file(path),
                "resumed": resumed,
            }
        else:
            result = value
    elif args.command == "run":
        arena = args.arena
        if not arena.is_absolute():
            arena = context.repository / arena
        result = gate.run_phase(
            context,
            arena,
            args.phase,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    elif args.command == "aggregate":
        result = gate.aggregate_phase(context, args.phase)
    elif args.command == "record-latency":
        input_path = args.input
        if not input_path.is_absolute():
            input_path = context.repository / input_path
        result = gate.record_latency(context, input_path.resolve())
    elif args.command == "lock-selection":
        result = gate.lock_selection(context)
    elif args.command == "render-lock":
        latency_input = args.latency_input
        if latency_input is not None and not latency_input.is_absolute():
            latency_input = context.repository / latency_input
        result = gate.render_cpp_lock(
            context,
            latency_input.resolve() if latency_input is not None else None,
        )
    elif args.command == "check-lock":
        latency_input = args.latency_input
        if latency_input is not None and not latency_input.is_absolute():
            latency_input = context.repository / latency_input
        result = gate.check_cpp_lock(
            context,
            latency_input.resolve() if latency_input is not None else None,
        )
    else:
        result = gate.write_report(context)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except gate.GateError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
