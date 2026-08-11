#!/usr/bin/env python3
"""Build and run deterministic sharded Jacek-native self-play."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_PROVENANCE_SCHEMA = "papersoccer.jacek-native-build-provenance/v1"
BUILD_PROVENANCE_NAME = "build-provenance.json"
ARCHIVED_BINARY_NAME = "selfplay-binary"
SELFPLAY_SOURCE = ROOT / "tools" / "jacek_native_selfplay.cpp"
ENGINE_SOURCE = (
    ROOT / "submissions" / "codingame" / "bots" / "jacek_native_bfm" /
    "bot.cpp"
)
MODEL_HEADER = ENGINE_SOURCE.with_name("jacek_native_model.hpp")
PROVENANCE_SOURCES = (
    SELFPLAY_SOURCE,
    ENGINE_SOURCE,
    MODEL_HEADER,
    ROOT / "src" / "core" / "rules.cpp",
    ROOT / "src" / "core" / "geometry.cpp",
    ROOT / "src" / "bots" / "mcts_internal.hpp",
    ROOT / "include" / "papersoccer" / "types.hpp",
    ROOT / "include" / "papersoccer" / "geometry.hpp",
    ROOT / "include" / "papersoccer" / "rules.hpp",
)
CANONICAL_BUILD_ARGV = (
    "$CXX",
    "-std=c++20",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Iinclude",
    "-Isrc/bots",
    "tools/jacek_native_selfplay.cpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "-o",
    "$OUTPUT",
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def source_contract() -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }
        for path in PROVENANCE_SOURCES
    ]


def producer_sha256(
    sources: list[dict[str, str]] | None = None,
) -> str:
    if sources is None:
        sources = source_contract()
    payload = json.dumps(
        [[entry["path"], entry["sha256"]] for entry in sources],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def compiler_metadata(resolved_compiler: pathlib.Path) -> dict[str, str]:
    version = subprocess.run(
        [str(resolved_compiler), "--version"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if not version:
        raise RuntimeError("compiler version output is empty")
    return {
        "executable": resolved_compiler.name,
        "sha256": sha256(resolved_compiler),
        "version": version,
        "version_sha256": hashlib.sha256(version.encode()).hexdigest(),
    }


def build_provenance(
    binary: pathlib.Path,
    resolved_compiler: pathlib.Path,
    sources: list[dict[str, str]] | None = None,
    compiler: dict[str, str] | None = None,
) -> dict:
    if sources is None:
        sources = source_contract()
    if compiler is None:
        compiler = compiler_metadata(resolved_compiler)
    report = {
        "schema": BUILD_PROVENANCE_SCHEMA,
        "binary": {
            "path": ARCHIVED_BINARY_NAME,
            "sha256": sha256(binary),
        },
        "compiler": compiler,
        "build_argv": list(CANONICAL_BUILD_ARGV),
        "producer_sha256": producer_sha256(sources),
        "sources": sources,
    }
    rendered = canonical_json_bytes(report).decode()
    home = str(pathlib.Path.home())
    if home and home in rendered:
        raise RuntimeError("canonical build provenance leaks a home-directory path")
    return report


def materialize_build_command(
    binary: pathlib.Path, resolved_compiler: pathlib.Path
) -> list[str]:
    result = []
    source_paths = {
        "tools/jacek_native_selfplay.cpp",
        "src/core/rules.cpp",
        "src/core/geometry.cpp",
    }
    for argument in CANONICAL_BUILD_ARGV:
        if argument == "$CXX":
            result.append(str(resolved_compiler))
        elif argument == "$OUTPUT":
            result.append(str(binary))
        elif argument.startswith("-I"):
            result.append(f"-I{ROOT / argument[2:]}")
        elif argument in source_paths:
            result.append(str(ROOT / argument))
        else:
            result.append(argument)
    return result


def build(binary: pathlib.Path, compiler: str) -> dict:
    resolved_compiler = shutil.which(compiler)
    if resolved_compiler is None:
        raise RuntimeError(f"compiler {compiler!r} is unavailable")
    resolved_compiler_path = pathlib.Path(resolved_compiler).resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    sources = source_contract()
    compiler_report = compiler_metadata(resolved_compiler_path)
    with tempfile.TemporaryDirectory(prefix="jacek-native-build-") as temporary:
        built_binary = pathlib.Path(temporary) / ARCHIVED_BINARY_NAME
        command = materialize_build_command(
            built_binary, resolved_compiler_path
        )
        subprocess.run(command, cwd=ROOT, check=True)
        shutil.copyfile(built_binary, binary)
        binary.chmod(0o755)
    if binary.stat().st_mode & 0o111 != 0o111:
        raise RuntimeError("self-play binary is not executable")
    if source_contract() != sources:
        raise RuntimeError("self-play sources changed during compilation")
    if compiler_metadata(resolved_compiler_path) != compiler_report:
        raise RuntimeError("compiler identity changed during compilation")
    report = build_provenance(
        binary, resolved_compiler_path, sources, compiler_report
    )
    binary.with_suffix(binary.suffix + ".provenance.json").write_bytes(
        canonical_json_bytes(report)
    )
    return report


def run_shard(
    binary: pathlib.Path,
    output: pathlib.Path,
    shard: int,
    arguments: argparse.Namespace,
    producer: str,
    build_provenance_sha256: str,
    checkpoint_hashes: dict[str, str],
) -> dict:
    command = [
        str(binary),
        "--output", str(output),
        "--games", str(arguments.games),
        "--seed", str(arguments.seed),
        "--work", str(arguments.work),
        "--samples-per-game", str(arguments.samples_per_game),
        "--shard-index", str(shard),
        "--shard-count", str(arguments.shards),
        "--temperature", str(arguments.temperature),
        "--temperature-turns", str(arguments.temperature_turns),
        "--opening-depths", arguments.opening_depths,
        "--max-complete-turns", str(arguments.max_complete_turns),
        "--reanalysis-work", str(arguments.reanalysis_work),
        "--producer-sha256", producer,
        "--build-provenance-sha256", build_provenance_sha256,
    ]
    if arguments.player_one_checkpoint:
        command.extend([
            "--player-one-checkpoint", str(arguments.player_one_checkpoint),
            "--player-one-artifact-sha256", checkpoint_hashes["player_one"],
        ])
    if arguments.player_two_checkpoint:
        command.extend([
            "--player-two-checkpoint", str(arguments.player_two_checkpoint),
            "--player-two-artifact-sha256", checkpoint_hashes["player_two"],
        ])
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"shard {shard} failed ({completed.returncode}):\n{completed.stderr}"
        )
    return {
        "shard": shard,
        "path": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "stderr": completed.stderr.strip(),
    }


def generate(arguments: argparse.Namespace) -> dict:
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: "
            f"{arguments.output_dir}"
        )
    build_report = build(arguments.binary, arguments.compiler)
    producer = build_report["producer_sha256"]
    build_provenance_bytes = canonical_json_bytes(build_report)
    build_provenance_sha256 = hashlib.sha256(
        build_provenance_bytes
    ).hexdigest()
    checkpoint_hashes = {}
    for name, path in (
        ("player_one", arguments.player_one_checkpoint),
        ("player_two", arguments.player_two_checkpoint),
    ):
        if path:
            if not path.is_file():
                raise ValueError(f"{name} checkpoint does not exist: {path}")
            checkpoint_hashes[name] = sha256(path)
    if bool(arguments.player_one_checkpoint) != bool(arguments.player_two_checkpoint):
        raise ValueError("provide both player checkpoints or neither")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / BUILD_PROVENANCE_NAME).write_bytes(
        build_provenance_bytes
    )
    archived_binary = arguments.output_dir / ARCHIVED_BINARY_NAME
    shutil.copyfile(arguments.binary, archived_binary)
    archived_binary.chmod(0o755)
    if archived_binary.stat().st_mode & 0o111 != 0o111:
        raise RuntimeError("archived self-play binary is not executable")
    if sha256(archived_binary) != build_report["binary"]["sha256"]:
        raise RuntimeError("archived self-play binary hash mismatch")
    tasks = [
        (
            arguments.output_dir /
            f"shard-{shard:02d}-of-{arguments.shards:02d}.jsonl",
            shard,
        )
        for shard in range(arguments.shards)
    ]
    started = time.monotonic()
    reports = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(arguments.parallel, arguments.shards)
    ) as executor:
        futures = [
            executor.submit(
                run_shard, arguments.binary, path, shard, arguments,
                producer, build_provenance_sha256, checkpoint_hashes,
            )
            for path, shard in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"shard {report['shard']} finished in "
                f"{report['elapsed_seconds']:.2f}s",
                file=sys.stderr,
            )
    elapsed = time.monotonic() - started
    manifest = {
        "schema": "papersoccer.jacek-native-selfplay-run/v1",
        "producer_sha256": producer,
        "build_provenance": {
            "path": BUILD_PROVENANCE_NAME,
            "sha256": build_provenance_sha256,
        },
        "binary": {
            "path": ARCHIVED_BINARY_NAME,
            "sha256": sha256(archived_binary),
        },
        "model_artifact_sha256": checkpoint_hashes,
        "config": {
            "games": arguments.games,
            "seed": arguments.seed,
            "work": arguments.work,
            "samples_per_game": arguments.samples_per_game,
            "shards": arguments.shards,
            "parallel": arguments.parallel,
            "temperature": arguments.temperature,
            "temperature_turns": arguments.temperature_turns,
            "temperature_schedule":
                "absolute-complete-turn-index-before-cutoff/v1",
            "opening_depths": [int(value) for value in arguments.opening_depths.split(",")],
            "opening_schema":
                "deterministic-procedural-complete-turn-prefix/v1",
            "reanalysis_work": arguments.reanalysis_work,
            "max_complete_turns": arguments.max_complete_turns,
            "checkpoint_color_schedule": "swap-player-checkpoints-on-odd-games",
        },
        "elapsed_seconds": elapsed,
        "games_per_second": arguments.games / elapsed,
        "shard_outputs": sorted(reports, key=lambda report: report["shard"]),
    }
    manifest_path = arguments.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or run the exact Jacek-native self-play engine."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--binary", type=pathlib.Path,
                              default=ROOT / "build" / "jacek_native_selfplay")
    build_parser.add_argument("--compiler", default="clang++")

    run_parser = subparsers.add_parser("generate")
    run_parser.add_argument("--binary", type=pathlib.Path,
                            default=ROOT / "build" / "jacek_native_selfplay")
    run_parser.add_argument("--compiler", default="clang++")
    run_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    run_parser.add_argument("--games", type=int, default=10_000)
    run_parser.add_argument("--seed", type=int, default=73194721)
    run_parser.add_argument("--work", type=int, default=2_048)
    run_parser.add_argument("--samples-per-game", type=int, default=100)
    run_parser.add_argument("--shards", type=int, default=14)
    run_parser.add_argument("--parallel", type=int, default=14)
    run_parser.add_argument("--temperature", type=float, default=3.0)
    run_parser.add_argument("--temperature-turns", type=int, default=12)
    run_parser.add_argument("--opening-depths", default="0")
    run_parser.add_argument("--reanalysis-work", type=int, default=0)
    run_parser.add_argument("--max-complete-turns", type=int, default=384)
    run_parser.add_argument("--player-one-checkpoint", type=pathlib.Path)
    run_parser.add_argument("--player-two-checkpoint", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.command == "build":
        print(json.dumps(build(arguments.binary, arguments.compiler),
                         indent=2, sort_keys=True))
        return 0
    numeric_positive = (
        arguments.games, arguments.work, arguments.samples_per_game,
        arguments.shards, arguments.parallel, arguments.max_complete_turns,
    )
    if any(value <= 0 for value in numeric_positive):
        parser.error("generation counts must be positive")
    if not 1 <= arguments.samples_per_game <= 100:
        parser.error("samples per game must be in [1,100]")
    if arguments.temperature_turns < 0:
        parser.error("temperature turns must be nonnegative")
    if arguments.reanalysis_work not in (0, *range(30_000, 100_001)):
        parser.error("reanalysis work must be zero or in [30000,100000]")
    try:
        depths = [int(value) for value in arguments.opening_depths.split(",")]
    except ValueError:
        parser.error("opening depths must be comma-separated integers")
    if not depths or any(value < 0 for value in depths):
        parser.error("opening depths must be nonnegative")
    generate(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
