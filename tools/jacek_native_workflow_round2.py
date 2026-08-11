#!/usr/bin/env python3
"""Build, archive, and run deterministic Jacek-native round-two leagues."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus_round2 as contract  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELFPLAY_SOURCE = ROOT / "tools" / "jacek_native_selfplay_round2.cpp"
PROVENANCE_SOURCES = tuple(ROOT / path for path in contract.BUILD_SOURCE_PATHS)
BUILD_PROVENANCE_SCHEMA = contract.BUILD_PROVENANCE_SCHEMA
BUILD_PROVENANCE_NAME = contract.BUILD_PROVENANCE_NAME
ARCHIVED_BINARY_NAME = contract.ARCHIVED_BINARY_NAME
CANONICAL_BUILD_ARGV = contract.CANONICAL_BUILD_ARGV
RUN_SCHEMA = contract.RUN_SCHEMA
RUNTIME_SCHEMA = "papersoccer.jacek-native-runtime-model/v1"
CHECKPOINT_DIRECTORY = "checkpoints"
TIMING_REPORT_NAME = "timing-report.json"
IDENTITY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return contract._canonical_json_bytes(value)


def source_contract() -> list[dict[str, str]]:
    return [{
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    } for path in PROVENANCE_SOURCES]


def producer_sha256(sources: list[dict[str, str]] | None = None) -> str:
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
        raise RuntimeError("build provenance leaks a home-directory path")
    return report


def materialize_build_command(
    binary: pathlib.Path, resolved_compiler: pathlib.Path
) -> list[str]:
    source_paths = {
        "tools/jacek_native_selfplay_round2.cpp",
        "src/core/rules.cpp",
        "src/core/geometry.cpp",
    }
    command = []
    for argument in CANONICAL_BUILD_ARGV:
        if argument == "$CXX":
            command.append(str(resolved_compiler))
        elif argument == "$OUTPUT":
            command.append(str(binary))
        elif argument.startswith("-I"):
            command.append(f"-I{ROOT / argument[2:]}")
        elif argument in source_paths:
            command.append(str(ROOT / argument))
        else:
            command.append(argument)
    return command


def build(binary: pathlib.Path, compiler: str) -> dict:
    resolved = shutil.which(compiler)
    if resolved is None:
        raise RuntimeError(f"compiler {compiler!r} is unavailable")
    resolved_path = pathlib.Path(resolved).resolve()
    sources = source_contract()
    compiler_report = compiler_metadata(resolved_path)
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jacek-native-round2-build-") as temp:
        built = pathlib.Path(temp) / ARCHIVED_BINARY_NAME
        subprocess.run(
            materialize_build_command(built, resolved_path), cwd=ROOT, check=True
        )
        shutil.copyfile(built, binary)
        binary.chmod(0o755)
    if binary.stat().st_mode & 0o111 != 0o111:
        raise RuntimeError("round-two self-play binary is not executable")
    if source_contract() != sources:
        raise RuntimeError("round-two self-play sources changed during compilation")
    if compiler_metadata(resolved_path) != compiler_report:
        raise RuntimeError("compiler identity changed during compilation")
    report = build_provenance(binary, resolved_path, sources, compiler_report)
    binary.with_suffix(binary.suffix + ".provenance.json").write_bytes(
        canonical_json_bytes(report)
    )
    return report


def runtime_identity(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        len(lines) != 7
        or lines[0] != RUNTIME_SCHEMA
        or lines[1] != "jacek_native_model/v1"
        or lines[2] != contract.FEATURE_SCHEMA
    ):
        raise ValueError(f"checkpoint runtime contract is malformed: {path}")
    for digest in lines[3:5]:
        if not contract._valid_sha256(digest):
            raise ValueError(f"checkpoint runtime SHA-256 is malformed: {path}")
    try:
        packed = base64.b64decode(lines[6], validate=True)
    except ValueError as error:
        raise ValueError(f"checkpoint runtime payload is malformed: {path}") from error
    if hashlib.sha256(packed).hexdigest() != lines[4]:
        raise ValueError(f"checkpoint runtime packed SHA-256 is stale: {path}")
    try:
        scales = [float(value) for value in lines[5].split()]
    except ValueError as error:
        raise ValueError(f"checkpoint runtime scales are malformed: {path}") from error
    expected_bytes = (38_048 * 3 + 7) // 8
    if (
        len(scales) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in scales)
        or len(packed) != expected_bytes
    ):
        raise ValueError(f"checkpoint runtime tensor contract is malformed: {path}")
    return {
        "artifact_sha256": sha256(path),
        "model_sha256": lines[3],
        "packed_sha256": lines[4],
    }


def _identity_name(value: str, label: str) -> str:
    if IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must match {IDENTITY_PATTERN.pattern!r}"
        )
    return value


def archive_checkpoint(
    output_dir: pathlib.Path,
    role: str,
    name: str,
    source: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    identity = runtime_identity(source)
    relative = pathlib.Path(CHECKPOINT_DIRECTORY) / f"{role}.runtime"
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(destination) != identity["artifact_sha256"]:
        raise RuntimeError(f"archived {role} checkpoint hash mismatch")
    return destination, {
        "name": _identity_name(name, f"{role} name"),
        "runtime": relative.as_posix(),
        **identity,
    }


def run_shard(
    binary: pathlib.Path,
    output: pathlib.Path,
    shard: int,
    arguments: argparse.Namespace,
    producer: str,
    build_provenance_sha256: str,
    checkpoints: dict[str, tuple[pathlib.Path, dict]],
) -> dict:
    command = [
        str(binary),
        "--output", str(output),
        "--games", str(arguments.games),
        "--seed", str(arguments.seed),
        "--work", str(arguments.work),
        "--samples-per-game", str(arguments.samples_per_game),
        "--reanalysis-samples-per-game",
        str(arguments.reanalysis_samples_per_game),
        "--shard-index", str(shard),
        "--shard-count", str(arguments.shards),
        "--temperature", str(arguments.temperature),
        "--temperature-turns", str(arguments.temperature_turns),
        "--opening-depths", arguments.opening_depths,
        "--max-complete-turns", str(arguments.max_complete_turns),
        "--reanalysis-work", str(arguments.reanalysis_work),
        "--verification-work", str(arguments.verification_work),
        "--producer-sha256", producer,
        "--build-provenance-sha256", build_provenance_sha256,
    ]
    for role, option in (
        ("player_one", "player-one"),
        ("player_two", "player-two"),
        ("teacher", "reanalysis"),
    ):
        if role not in checkpoints:
            continue
        path, identity = checkpoints[role]
        command.extend([
            f"--{option}-checkpoint", str(path),
            f"--{option}-artifact-sha256", identity["artifact_sha256"],
        ])
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"round-two shard {shard} failed ({completed.returncode}):\n"
            f"{completed.stderr}"
        )
    return {
        "shard": shard,
        "path": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }


def stable_shard_reports(reports: list[dict]) -> list[dict]:
    """Exclude volatile wall-clock measurements from corpus identity."""
    return sorted(({
        "shard": report["shard"],
        "path": report["path"],
        "sha256": report["sha256"],
        "bytes": report["bytes"],
        "stderr_sha256": report["stderr_sha256"],
    } for report in reports), key=lambda report: report["shard"])


def validate_generate_arguments(arguments: argparse.Namespace) -> list[int]:
    positive = (
        arguments.games, arguments.work, arguments.samples_per_game,
        arguments.shards, arguments.parallel, arguments.max_complete_turns,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("generation counts must be positive")
    if not 0 <= arguments.seed < 1 << 64:
        raise ValueError("generation seed must fit uint64")
    if not 1 <= arguments.samples_per_game <= 100:
        raise ValueError("samples per game must be in [1,100]")
    if not 0 <= arguments.reanalysis_samples_per_game <= arguments.samples_per_game:
        raise ValueError("reanalysis samples must fit samples per game")
    if arguments.temperature_turns < 0:
        raise ValueError("temperature turns must be nonnegative")
    try:
        depths = [int(value) for value in arguments.opening_depths.split(",")]
    except ValueError as error:
        raise ValueError("opening depths must be comma-separated integers") from error
    if not depths or any(value < 0 for value in depths):
        raise ValueError("opening depths must be nonnegative")
    if arguments.games % (2 * len(depths)) != 0:
        raise ValueError("games must contain complete depth/color schedule cycles")
    if not arguments.player_one_checkpoint or not arguments.player_two_checkpoint:
        raise ValueError("round-two generation requires both player checkpoints")
    if arguments.reanalysis_work == 0:
        if arguments.teacher_checkpoint or arguments.reanalysis_samples_per_game:
            raise ValueError("disabled reanalysis cannot name a teacher or samples")
    elif (
        arguments.reanalysis_work != contract.TEACHER_WORK
        or arguments.verification_work != contract.VERIFICATION_WORK
        or not arguments.teacher_checkpoint
        or arguments.reanalysis_samples_per_game == 0
    ):
        raise ValueError("round-two reanalysis requires a fixed 30k/100k teacher")
    _identity_name(arguments.run_id, "run id")
    return depths


def generate(arguments: argparse.Namespace) -> dict:
    depths = validate_generate_arguments(arguments)
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: "
            f"{arguments.output_dir}"
        )
    build_report = build(arguments.binary, arguments.compiler)
    build_bytes = canonical_json_bytes(build_report)
    build_digest = hashlib.sha256(build_bytes).hexdigest()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / BUILD_PROVENANCE_NAME).write_bytes(build_bytes)
    archived_binary = arguments.output_dir / ARCHIVED_BINARY_NAME
    shutil.copyfile(arguments.binary, archived_binary)
    archived_binary.chmod(0o755)
    if sha256(archived_binary) != build_report["binary"]["sha256"]:
        raise RuntimeError("archived round-two binary hash mismatch")

    checkpoints: dict[str, tuple[pathlib.Path, dict]] = {}
    for role, name, source in (
        ("player_one", arguments.player_one_name,
         arguments.player_one_checkpoint),
        ("player_two", arguments.player_two_name,
         arguments.player_two_checkpoint),
        ("teacher", arguments.teacher_name, arguments.teacher_checkpoint),
    ):
        if source:
            checkpoints[role] = archive_checkpoint(
                arguments.output_dir, role, name, source
            )

    tasks = [(
        arguments.output_dir /
        f"shard-{shard:02d}-of-{arguments.shards:02d}.jsonl",
        shard,
    ) for shard in range(arguments.shards)]
    started = time.monotonic()
    reports = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(arguments.parallel, arguments.shards)
    ) as executor:
        futures = [executor.submit(
            run_shard, arguments.binary, path, shard, arguments,
            build_report["producer_sha256"], build_digest, checkpoints,
        ) for path, shard in tasks]
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"round-two shard {report['shard']} finished in "
                f"{report['elapsed_seconds']:.2f}s",
                file=sys.stderr,
            )
    elapsed = time.monotonic() - started
    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": arguments.run_id,
        "producer_sha256": build_report["producer_sha256"],
        "build_provenance": {
            "path": BUILD_PROVENANCE_NAME,
            "sha256": build_digest,
        },
        "binary": {
            "path": ARCHIVED_BINARY_NAME,
            "sha256": sha256(archived_binary),
        },
        "checkpoints": {
            role: metadata for role, (_, metadata) in sorted(checkpoints.items())
        },
        "config": {
            "games": arguments.games,
            "seed": arguments.seed,
            "work": arguments.work,
            "samples_per_game": arguments.samples_per_game,
            "reanalysis_samples_per_game": arguments.reanalysis_samples_per_game,
            "shards": arguments.shards,
            "parallel": arguments.parallel,
            "temperature": arguments.temperature,
            "temperature_turns": arguments.temperature_turns,
            "temperature_schedule":
                "absolute-complete-turn-index-before-cutoff/v1",
            "opening_depths": depths,
            "opening_schema":
                "deterministic-procedural-complete-turn-prefix/v1",
            "checkpoint_color_schedule": contract.COLOR_SCHEDULE,
            "reanalysis_selection": contract.REANALYSIS_SELECTION,
            "reanalysis_work": arguments.reanalysis_work,
            "verification_work": (
                arguments.verification_work if arguments.reanalysis_work else 0
            ),
            "max_complete_turns": arguments.max_complete_turns,
        },
        "shard_outputs": stable_shard_reports(reports),
    }
    manifest_path = arguments.output_dir / contract.MANIFEST_NAME
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    timing_report = {
        "schema": "papersoccer.jacek-native-run-timing/v1",
        "run_id": arguments.run_id,
        "elapsed_seconds": elapsed,
        "games_per_second": arguments.games / elapsed,
        "shards": [{
            "shard": report["shard"],
            "elapsed_seconds": report["elapsed_seconds"],
        } for report in sorted(reports, key=lambda report: report["shard"])],
    }
    (arguments.output_dir / TIMING_REPORT_NAME).write_text(
        json.dumps(timing_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_member(value: str) -> tuple[str, pathlib.Path]:
    name, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("league members must be NAME=RUNTIME")
    try:
        _identity_name(name, "league member")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return name, pathlib.Path(path)


def league_pairings(
    members: dict[str, pathlib.Path], anchor: str, base_seed: int
) -> list[tuple[str, pathlib.Path, int]]:
    if anchor not in members:
        raise ValueError("league anchor is not a member")
    if not 0 <= base_seed < 1 << 64:
        raise ValueError("league base seed must fit uint64")
    ordered = [anchor] + sorted(name for name in members if name != anchor)
    result = []
    used: set[int] = set()
    for index, name in enumerate(ordered):
        seed = (base_seed + index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        if seed in used:
            raise ValueError("league seed derivation collided")
        used.add(seed)
        result.append((name, members[name], seed))
    return result


def run_league(arguments: argparse.Namespace) -> dict:
    members = dict(arguments.member)
    if len(members) != len(arguments.member):
        raise ValueError("league member names must be unique")
    pairings = league_pairings(members, arguments.anchor, arguments.base_seed)
    if arguments.output_root.exists() and any(arguments.output_root.iterdir()):
        raise ValueError("league output root must be empty")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, (opponent_name, opponent_path, seed) in enumerate(pairings):
        run_id = f"{arguments.anchor}-vs-{opponent_name}"
        run_arguments = argparse.Namespace(**vars(arguments))
        run_arguments.output_dir = (
            arguments.output_root / f"{index:02d}-{run_id}"
        )
        run_arguments.run_id = run_id
        run_arguments.seed = seed
        run_arguments.player_one_checkpoint = members[arguments.anchor]
        run_arguments.player_two_checkpoint = opponent_path
        run_arguments.player_one_name = arguments.anchor
        run_arguments.player_two_name = opponent_name
        manifest = generate(run_arguments)
        manifest_path = run_arguments.output_dir / contract.MANIFEST_NAME
        runs.append({
            "run_id": run_id,
            "seed": seed,
            "manifest_sha256": sha256(manifest_path),
            "games": manifest["config"]["games"],
        })
    report = {
        "schema": "papersoccer.jacek-native-league/v1",
        "anchor": arguments.anchor,
        "members": sorted(members),
        "runs": runs,
    }
    (arguments.output_root / "league-manifest.json").write_bytes(
        canonical_json_bytes(report)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", type=pathlib.Path,
                        default=ROOT / "build" / "jacek_native_selfplay_round2")
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026081102)
    parser.add_argument("--work", type=int, default=2_048)
    parser.add_argument("--samples-per-game", type=int, default=100)
    parser.add_argument("--reanalysis-samples-per-game", type=int, default=12)
    parser.add_argument("--shards", type=int, default=14)
    parser.add_argument("--parallel", type=int, default=14)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--temperature-turns", type=int, default=12)
    parser.add_argument("--opening-depths", default="0,4,8,12")
    parser.add_argument("--reanalysis-work", type=int, default=30_000)
    parser.add_argument("--verification-work", type=int, default=100_000)
    parser.add_argument("--max-complete-turns", type=int, default=384)
    parser.add_argument("--teacher-checkpoint", type=pathlib.Path)
    parser.add_argument("--teacher-name", default="current")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or run strict Jacek-native round-two self-play."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--binary", type=pathlib.Path,
                              default=ROOT / "build" /
                              "jacek_native_selfplay_round2")
    build_parser.add_argument("--compiler", default="clang++")

    generate_parser = subparsers.add_parser("generate")
    add_generation_arguments(generate_parser)
    generate_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    generate_parser.add_argument("--run-id", required=True)
    generate_parser.add_argument("--player-one-checkpoint", type=pathlib.Path)
    generate_parser.add_argument("--player-two-checkpoint", type=pathlib.Path)
    generate_parser.add_argument("--player-one-name", default="current")
    generate_parser.add_argument("--player-two-name", default="current")

    league_parser = subparsers.add_parser("league")
    add_generation_arguments(league_parser)
    league_parser.add_argument("--output-root", type=pathlib.Path, required=True)
    league_parser.add_argument("--member", action="append", type=parse_member,
                               required=True)
    league_parser.add_argument("--anchor", default="current")
    league_parser.add_argument("--base-seed", type=int, default=2026081102)
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            print(json.dumps(build(arguments.binary, arguments.compiler),
                             indent=2, sort_keys=True))
        elif arguments.command == "generate":
            generate(arguments)
        else:
            run_league(arguments)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
