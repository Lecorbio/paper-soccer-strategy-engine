#!/usr/bin/env python3
"""Archive and run provenance-safe native continuations from live losses."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_restart_corpus_round2 as contract  # noqa: E402
import jacek_native_workflow_round2 as round2_workflow  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTITY = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_parallel_workers(arguments: argparse.Namespace) -> int:
    """Resolve execution-only concurrency without changing run identity."""
    shards = arguments.shards
    if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1:
        raise ValueError("shard count must be a positive integer")
    requested = getattr(arguments, "parallel", None)
    if requested is None:
        available = os.cpu_count() or 1
        return min(shards, max(1, available))
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or not 1 <= requested <= shards
    ):
        raise ValueError("parallel workers must be in [1, shard count]")
    return requested


def source_contract() -> list[dict[str, str]]:
    return [{
        "path": path,
        "sha256": sha256(ROOT / path),
    } for path in contract.BUILD_SOURCE_PATHS]


def producer_sha256(sources: list[dict[str, str]]) -> str:
    payload = json.dumps(
        [[entry["path"], entry["sha256"]] for entry in sources],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def compiler_metadata(executable: pathlib.Path) -> dict[str, str]:
    version = subprocess.run(
        [str(executable), "--version"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if not version:
        raise RuntimeError("compiler version output is empty")
    return {
        "executable": executable.name,
        "sha256": sha256(executable),
        "version": version,
        "version_sha256": hashlib.sha256(version.encode()).hexdigest(),
    }


def materialize_build_command(
    compiler: pathlib.Path, output: pathlib.Path,
) -> list[str]:
    source_arguments = {
        "tools/jacek_native_restart_round2.cpp",
        "src/core/rules.cpp",
        "src/core/geometry.cpp",
    }
    result = []
    for argument in contract.CANONICAL_BUILD_ARGV:
        if argument == "$CXX":
            result.append(str(compiler))
        elif argument == "$OUTPUT":
            result.append(str(output))
        elif argument.startswith("-I"):
            result.append(f"-I{ROOT / argument[2:]}")
        elif argument in source_arguments:
            result.append(str(ROOT / argument))
        else:
            result.append(argument)
    return result


def build_binary(directory: pathlib.Path, compiler_name: str) -> tuple[dict, str]:
    resolved = shutil.which(compiler_name)
    if resolved is None:
        raise RuntimeError(f"compiler {compiler_name!r} is unavailable")
    compiler = pathlib.Path(resolved).resolve()
    sources = source_contract()
    compiler_report = compiler_metadata(compiler)
    binary = directory / contract.ARCHIVED_BINARY_NAME
    subprocess.run(
        materialize_build_command(compiler, binary), cwd=ROOT, check=True
    )
    binary.chmod(0o755)
    if source_contract() != sources or compiler_metadata(compiler) != compiler_report:
        raise RuntimeError("restart build inputs changed during compilation")
    report = {
        "schema": contract.BUILD_PROVENANCE_SCHEMA,
        "binary": {
            "path": contract.ARCHIVED_BINARY_NAME,
            "sha256": sha256(binary),
        },
        "compiler": compiler_report,
        "build_argv": list(contract.CANONICAL_BUILD_ARGV),
        "producer_sha256": producer_sha256(sources),
        "sources": sources,
    }
    raw = contract.canonical_json_bytes(report)
    (directory / contract.BUILD_PROVENANCE_NAME).write_bytes(raw)
    return report, hashlib.sha256(raw).hexdigest()


def archive_checkpoint(
    directory: pathlib.Path, role: str, name: str, source: pathlib.Path,
) -> tuple[pathlib.Path, dict]:
    if IDENTITY.fullmatch(name) is None:
        raise ValueError(f"{role} checkpoint name is unsafe")
    identity = round2_workflow.runtime_identity(source)
    destination = directory / f"{role}.runtime"
    shutil.copyfile(source, destination)
    if sha256(destination) != identity["artifact_sha256"]:
        raise RuntimeError(f"archived {role} checkpoint hash mismatch")
    return destination, {
        "name": name,
        "runtime": destination.name,
        **identity,
    }


def assert_expected_input(
    collector: contract.CollectorInput, arguments: argparse.Namespace,
) -> None:
    expected = {
        "agent_id": arguments.expected_agent_id,
        "asserted_submission_id": arguments.expected_submission_id,
        "asserted_source_sha256": arguments.expected_source_sha256,
        "arena_manifest_sha256": arguments.expected_manifest_sha256,
        "exclusion_registry_sha256": (
            arguments.expected_exclusion_registry_sha256
        ),
    }
    for field, value in expected.items():
        if collector.metadata[field] != value:
            raise ValueError(
                f"explicit expected {field} does not match collector TSV"
            )


def run_shard(
    binary: pathlib.Path, input_path: pathlib.Path, output: pathlib.Path,
    shard: int, arguments: argparse.Namespace, collector: contract.CollectorInput,
    producer: str, build_sha256: str,
    checkpoints: dict[str, tuple[pathlib.Path, dict]],
) -> dict:
    command = [
        str(binary),
        "--input", str(input_path),
        "--output", str(output),
        "--input-sha256", collector.sha256,
        "--expected-source-sha256", arguments.expected_source_sha256,
        "--expected-manifest-sha256", arguments.expected_manifest_sha256,
        "--expected-exclusion-registry-sha256",
        arguments.expected_exclusion_registry_sha256,
        "--expected-submission-id", arguments.expected_submission_id,
        "--expected-agent-id", arguments.expected_agent_id,
        "--player-one-checkpoint", str(checkpoints["player_one"][0]),
        "--player-two-checkpoint", str(checkpoints["player_two"][0]),
        "--player-one-artifact-sha256",
        checkpoints["player_one"][1]["artifact_sha256"],
        "--player-two-artifact-sha256",
        checkpoints["player_two"][1]["artifact_sha256"],
        "--producer-sha256", producer,
        "--build-provenance-sha256", build_sha256,
        "--seed", str(arguments.seed),
        "--work", str(arguments.work),
        "--samples-per-game", str(arguments.samples_per_game),
        "--reanalysis-samples-per-game",
        str(arguments.reanalysis_samples_per_game),
        "--prefixes-per-loss", str(arguments.prefixes_per_loss),
        "--max-selected-prefixes", str(arguments.max_selected_prefixes),
        "--continuations-per-prefix", str(arguments.continuations_per_prefix),
        "--shard-index", str(shard),
        "--shard-count", str(arguments.shards),
        "--temperature", str(arguments.temperature),
        "--temperature-turns", str(arguments.temperature_turns),
        "--max-generated-complete-turns",
        str(arguments.max_generated_complete_turns),
        "--reanalysis-work", str(arguments.reanalysis_work),
        "--verification-work", str(arguments.verification_work),
    ]
    if arguments.selected_prefixes is not None:
        command.extend([
            "--selected-prefixes",
            str(input_path.parent / contract.ARCHIVED_SELECTED_PREFIXES_NAME),
            "--selected-prefixes-sha256", arguments.selected_prefixes_sha256,
        ])
    if arguments.reanalysis_work:
        command.extend([
            "--reanalysis-checkpoint", str(checkpoints["teacher"][0]),
            "--reanalysis-artifact-sha256",
            checkpoints["teacher"][1]["artifact_sha256"],
        ])
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"live-restart shard {shard} failed ({completed.returncode}):\n"
            f"{completed.stderr}"
        )
    raw = output.read_bytes()
    records = len(raw.splitlines())
    return {
        "shard": shard,
        "path": output.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "records": records,
    }


def run(arguments: argparse.Namespace) -> pathlib.Path:
    parallel_workers = resolve_parallel_workers(arguments)
    arguments.selected_prefixes = getattr(arguments, "selected_prefixes", None)
    input_path = contract._safe_explicit_path(
        arguments.input, "collector TSV"
    )
    collector = contract.parse_collector_bytes(input_path.read_bytes())
    assert_expected_input(collector, arguments)
    selected_input = None
    if arguments.selected_prefixes is None:
        selected = contract.select_prefixes(
            collector, arguments.prefixes_per_loss,
            arguments.max_selected_prefixes,
        )
        arguments.selected_prefixes_sha256 = None
    else:
        selected_path = contract._safe_explicit_path(
            arguments.selected_prefixes, "selected-prefix manifest"
        )
        selected_input = selected_path.read_bytes()
        selected = contract.select_manifest_prefixes(collector, selected_input)
        arguments.selected_prefixes_sha256 = hashlib.sha256(
            selected_input
        ).hexdigest()
    record_count = len(selected) * arguments.continuations_per_prefix
    if record_count > 65_536:
        raise ValueError("live-restart record plan exceeds 65536 games")
    if arguments.shards > record_count:
        raise ValueError("shard count cannot exceed restart record count")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an existing restart run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as temporary:
        directory = pathlib.Path(temporary)
        archived_input = directory / contract.ARCHIVED_INPUT_NAME
        shutil.copyfile(input_path, archived_input)
        if sha256(archived_input) != collector.sha256:
            raise RuntimeError("archived collector TSV hash mismatch")
        archived_selected = None
        if selected_input is not None:
            archived_selected = directory / contract.ARCHIVED_SELECTED_PREFIXES_NAME
            archived_selected.write_bytes(selected_input)
            if sha256(archived_selected) != arguments.selected_prefixes_sha256:
                raise RuntimeError("archived selected-prefix manifest hash mismatch")
        build_report, build_sha = build_binary(directory, arguments.compiler)
        binary = directory / contract.ARCHIVED_BINARY_NAME
        checkpoints = {
            "player_one": archive_checkpoint(
                directory, "player_one", arguments.player_one_name,
                arguments.player_one_checkpoint,
            ),
            "player_two": archive_checkpoint(
                directory, "player_two", arguments.player_two_name,
                arguments.player_two_checkpoint,
            ),
        }
        if arguments.reanalysis_work:
            checkpoints["teacher"] = archive_checkpoint(
                directory, "teacher", arguments.teacher_name,
                arguments.teacher_checkpoint,
            )
        shard_reports = []
        shard_paths = []
        shard_tasks = []
        for shard in range(arguments.shards):
            shard_path = directory / (
                f"shard-{shard:02d}-of-{arguments.shards:02d}.jsonl"
            )
            shard_paths.append(shard_path)
            shard_tasks.append((shard, shard_path))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_workers
        ) as executor:
            futures = [executor.submit(
                run_shard, binary, archived_input, shard_path, shard,
                arguments, collector, build_report["producer_sha256"],
                build_sha, checkpoints,
            ) for shard, shard_path in shard_tasks]
            for future in concurrent.futures.as_completed(futures):
                shard_reports.append(future.result())
        shard_reports.sort(key=lambda report: report["shard"])
        if sum(report["records"] for report in shard_reports) != record_count:
            raise RuntimeError("restart producer emitted an incomplete record set")
        input_manifest = {
            "path": contract.ARCHIVED_INPUT_NAME,
            "sha256": collector.sha256,
            "metadata": dict(collector.metadata),
        }
        if archived_selected is not None:
            input_manifest.update({
                "selected_prefixes_path": contract.ARCHIVED_SELECTED_PREFIXES_NAME,
                "selected_prefixes_sha256": arguments.selected_prefixes_sha256,
            })
        manifest = {
            "schema": contract.RUN_SCHEMA,
            "input": input_manifest,
            "build_provenance": {
                "path": contract.BUILD_PROVENANCE_NAME,
                "sha256": build_sha,
            },
            "binary": build_report["binary"],
            "checkpoints": {
                role: metadata for role, (_, metadata) in checkpoints.items()
            },
            "config": {
                "seed": arguments.seed,
                "work": arguments.work,
                "samples_per_game": arguments.samples_per_game,
                "reanalysis_samples_per_game": (
                    arguments.reanalysis_samples_per_game
                    if arguments.reanalysis_work else 0
                ),
                "prefixes_per_loss": arguments.prefixes_per_loss,
                "max_selected_prefixes": arguments.max_selected_prefixes,
                "continuations_per_prefix": arguments.continuations_per_prefix,
                "shards": arguments.shards,
                "temperature": arguments.temperature,
                "temperature_turns": arguments.temperature_turns,
                "max_generated_complete_turns": (
                    arguments.max_generated_complete_turns
                ),
                "reanalysis_work": arguments.reanalysis_work,
                "verification_work": (
                    arguments.verification_work
                    if arguments.reanalysis_work else 0
                ),
                "records": record_count,
            },
            "selected_prefixes": [
                dataclasses.asdict(prefix) for prefix in selected
            ],
            "shard_outputs": shard_reports,
        }
        # Import locally to keep the public contract module lightweight.
        (directory / contract.MANIFEST_NAME).write_bytes(
            contract.canonical_json_bytes(manifest)
        )
        contract.load_games(shard_paths, verify_local_build=True)
        directory.rename(output)
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive one explicit clean collector TSV and generate fresh native "
            "self-play/reanalysis continuations from deterministic loss prefixes."
        )
    )
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument(
        "--selected-prefixes", type=pathlib.Path,
        help=(
            "explicit provenance-bound prefix manifest; every full collector "
            "transcript is replay-validated and observed moves remain state-only"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--expected-agent-id", required=True)
    parser.add_argument("--expected-submission-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-exclusion-registry-sha256", required=True
    )
    parser.add_argument("--player-one-checkpoint", required=True,
                        type=pathlib.Path)
    parser.add_argument("--player-two-checkpoint", required=True,
                        type=pathlib.Path)
    parser.add_argument("--player-one-name", default="current")
    parser.add_argument("--player-two-name", default="previous")
    parser.add_argument("--teacher-checkpoint", type=pathlib.Path)
    parser.add_argument("--teacher-name", default="teacher")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--work", type=int, default=4096)
    parser.add_argument("--samples-per-game", type=int, default=100)
    parser.add_argument("--reanalysis-samples-per-game", type=int, default=12)
    parser.add_argument("--prefixes-per-loss", type=int, default=4)
    parser.add_argument(
        "--max-selected-prefixes", type=int, default=0,
        help="deterministic evenly-spaced pilot cap; zero retains every prefix",
    )
    parser.add_argument("--continuations-per-prefix", type=int, default=2)
    parser.add_argument("--shards", type=int, default=14)
    parser.add_argument(
        "--parallel", type=int,
        help=(
            "concurrent shard workers; defaults to the smaller of available "
            "CPU cores and --shards"
        ),
    )
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--temperature-turns", type=int, default=12)
    parser.add_argument("--max-generated-complete-turns", type=int, default=384)
    parser.add_argument("--reanalysis-work", type=int, default=30_000)
    parser.add_argument("--verification-work", type=int, default=100_000)
    parser.add_argument("--compiler", default="c++")
    arguments = parser.parse_args()
    integers = (
        arguments.seed, arguments.work, arguments.samples_per_game,
        arguments.reanalysis_samples_per_game, arguments.prefixes_per_loss,
        arguments.max_selected_prefixes,
        arguments.continuations_per_prefix, arguments.shards,
        arguments.temperature_turns, arguments.max_generated_complete_turns,
        arguments.reanalysis_work, arguments.verification_work,
    )
    if any(isinstance(value, bool) or value < 0 for value in integers):
        parser.error("integer limits must be nonnegative")
    if (
        arguments.seed >= 1 << 64 or arguments.work < 2
        or not 1 <= arguments.samples_per_game <= 100
        or not 1 <= arguments.prefixes_per_loss <= 32
        or not 0 <= arguments.max_selected_prefixes <= 4_096
        or not 2 <= arguments.continuations_per_prefix <= 32
        or arguments.continuations_per_prefix % 2
        or arguments.shards < 1 or arguments.max_generated_complete_turns < 1
        or not math.isfinite(arguments.temperature)
        or arguments.temperature < 0
    ):
        parser.error("restart generation limits are invalid")
    try:
        arguments.parallel = resolve_parallel_workers(arguments)
    except ValueError as error:
        parser.error(str(error))
    if arguments.reanalysis_work:
        if (
            arguments.reanalysis_work != 30_000
            or arguments.verification_work != 100_000
            or arguments.reanalysis_samples_per_game < 1
            or arguments.reanalysis_samples_per_game > arguments.samples_per_game
            or arguments.teacher_checkpoint is None
        ):
            parser.error("enabled reanalysis requires teacher and 30k/100k")
    elif arguments.teacher_checkpoint is not None:
        parser.error("disabled reanalysis cannot name a teacher checkpoint")
    for label, digest in (
        ("source", arguments.expected_source_sha256),
        ("manifest", arguments.expected_manifest_sha256),
        ("exclusion registry", arguments.expected_exclusion_registry_sha256),
    ):
        if contract.LOWER_SHA.fullmatch(digest) is None:
            parser.error(f"expected {label} identity must be a SHA-256")
    if (
        not arguments.expected_agent_id.isdigit()
        or not arguments.expected_submission_id.isdigit()
    ):
        parser.error("expected agent/submission IDs must be decimal")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    output = run(arguments)
    print(json.dumps({
        "output_dir": str(output),
        "manifest_sha256": sha256(output / contract.MANIFEST_NAME),
        "status": "validated",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
