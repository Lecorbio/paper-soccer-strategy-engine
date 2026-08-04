#!/usr/bin/env python3
"""Validate, execute, resume, aggregate, and analyze the flagship study."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.flagship_study import studylib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=pathlib.Path,
        default=pathlib.Path("benchmarks/flagship_study/manifest.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate", help="strictly validate manifest, artifacts, and banks")

    run_parser = commands.add_parser("run", help="run/resume deterministic phase shards")
    run_parser.add_argument("--phase", choices=studylib.FULL_PHASES, required=True)
    run_parser.add_argument("--arena", type=pathlib.Path,
                            default=pathlib.Path("build/release/papersoccer_arena"))
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--destructive-test-override", action="store_true")

    aggregate_parser = commands.add_parser(
        "aggregate", help="validate complete phase output and write compact curated data"
    )
    aggregate_parser.add_argument("--phase", choices=studylib.FULL_PHASES, required=True)

    lock_parser = commands.add_parser(
        "lock-selection", help="select eligible validation configurations and freeze calibration"
    )
    lock_parser.add_argument("--replace", action="store_true",
                             help="replace only an uncommitted lock for repair before test")

    analyze_parser = commands.add_parser(
        "analyze-test", help="analyze frozen test data and generate charts/report"
    )
    analyze_parser.add_argument("--replace", action="store_true")

    projection_parser = commands.add_parser(
        "project-runtime", help="project full workloads from completed development units"
    )
    projection_parser.add_argument("--write", action="store_true")
    projection_parser.add_argument("--replace", action="store_true")

    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    repository = studylib.repository_root_from_manifest(manifest_path)
    manifest = studylib.validate_manifest(
        studylib.load_json(manifest_path), repository, verify_files=True
    )
    studylib.verify_opening_phase_disjointness(manifest, repository)

    if args.command == "validate":
        result = {
            "valid": True,
            "manifest_sha256": studylib.manifest_sha256(manifest_path),
            "configurations": len(manifest["configurations"]),
            "opening_banks": len(manifest["openings"]["banks"]),
            "opening_records": sum(bank["pairs"] for bank in manifest["openings"]["banks"]),
        }
    else:
        studylib.verify_flagship_source_checkout(manifest, repository)
        if args.command == "run":
            arena = args.arena
            if not arena.is_absolute():
                arena = repository / arena
            result = studylib.run_phase(
                manifest_path, arena, args.phase,
                shard_count=args.shard_count, shard_index=args.shard_index,
                destructive_override=args.destructive_test_override,
            )
        elif args.command == "aggregate":
            result = studylib.aggregate_phase(manifest_path, args.phase)
        elif args.command == "lock-selection":
            result = studylib.create_selection_lock(manifest_path, replace=args.replace)
        elif args.command == "analyze-test":
            result = studylib.analyze_test(manifest_path, replace=args.replace)
        else:
            result = studylib.project_runtime(
                manifest_path, write=args.write, replace=args.replace
            )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except studylib.StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
