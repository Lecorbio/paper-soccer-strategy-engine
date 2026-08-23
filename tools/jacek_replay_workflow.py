#!/usr/bin/env python3
"""Run the reproducible replay-root, Rank-4 relabel, pack, and train workflow."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable


HERE = pathlib.Path(__file__).resolve().parent
WORKFLOW_SCHEMA = "papersoccer.jacek-replay-bfm-workflow.v1"
MODEL_SCHEMA = "papersoccer.jacek-replay-bfm-model.v1"
STAGE_RECEIPT_SCHEMA = "papersoccer.jacek-replay-bfm-stage-receipt.v1"
CHUNK_RECEIPT_SCHEMA = "papersoccer.jacek-replay-bfm-teacher-chunk-receipt.v1"
CONTINUATION_MANIFEST_SCHEMA = (
    "papersoccer.jacek-replay-continuations-manifest.v1"
)
CANONICAL_CAMPAIGN_ID = "canonical-20260823-v1"
SMOKE_PROFILE = "development-three-round-smoke-v1"
CANONICAL_CONFIGURATION = {
    "continuation_games": 10_000,
    "nodes": 32_000,
    "root_nodes": 400_000,
    "deep_percent": 10,
    "max_samples_per_game": 100,
    "actor_nodes": 16_000,
    "candidate_tree_nodes": 2_000,
    "epochs": 50,
    "patience": 8,
    "batch_size": 256,
    "seeds": "20260823,20260824,20260825",
    "campaign_id": CANONICAL_CAMPAIGN_ID,
    "teacher_workers": 10,
    "teacher_chunk_games": 25,
    "seed_workers": 2,
}
SMOKE_CONFIGURATION = {
    "continuation_games": 40,
    "nodes": 100,
    "root_nodes": 100,
    "deep_percent": 10,
    "max_samples": 5,
    "actor_nodes": 100,
    "candidate_tree_nodes": 16,
    "epochs": 1,
    "patience": 1,
    "batch_size": 16,
    "seeds": "20260823,20260824,20260825",
    "campaign_id": SMOKE_PROFILE,
    "teacher_workers": 2,
    "teacher_chunk_games": 5,
    "seed_workers": 2,
}

CANDIDATE_SOURCE_CLOSURE_PATHS = (
    "include/papersoccer/bot.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
    "src/bots/bot.cpp",
    "src/bots/mcts_internal.hpp",
    "src/core/geometry.cpp",
    "src/core/rules.cpp",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    """Serialize receipt material without platform-dependent formatting."""

    if pretty:
        rendered = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
    else:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (rendered + "\n").encode("utf-8")


def _line_count(path: pathlib.Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + int(path.stat().st_size > 0 and last != b"\n")


def artifact_snapshot(path: pathlib.Path) -> dict:
    """Return a recursively verifiable, path-bound file snapshot."""

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": sha256(resolved),
            "bytes": resolved.stat().st_size,
            "lines": _line_count(resolved),
        }
    if resolved.is_dir():
        files = []
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": child.relative_to(resolved).as_posix(),
                    "sha256": sha256(child),
                    "bytes": child.stat().st_size,
                    "lines": _line_count(child),
                }
            )
        if not files:
            raise ValueError(f"artifact directory is empty: {resolved}")
        return {"kind": "directory", "path": str(resolved), "files": files}
    raise ValueError(f"artifact does not exist: {resolved}")


def _snapshots(paths: dict[str, pathlib.Path]) -> dict[str, dict]:
    return {label: artifact_snapshot(path) for label, path in sorted(paths.items())}


def environment_identity() -> dict:
    try:
        import numpy as np

        numpy_version = np.__version__
    except ImportError:
        numpy_version = None
    executable = pathlib.Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_executable_sha256": sha256(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": numpy_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _identity_command(repository: pathlib.Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "could not identify campaign repository: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def repository_identity(repository: pathlib.Path) -> dict:
    """Bind a campaign to one current, inspectable Git checkout."""

    repository = repository.resolve()
    head = _identity_command(repository, "rev-parse", "HEAD").decode().strip()
    tree = _identity_command(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    branch = _identity_command(repository, "branch", "--show-current").decode().strip()
    status = _identity_command(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", head) or not re.fullmatch(
        r"[0-9a-f]{40,64}", tree
    ):
        raise ValueError("campaign repository has an invalid commit or tree identity")
    return {
        "path": str(repository),
        "head": head,
        "tree": tree,
        "branch": branch or None,
        "clean": status == b"",
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _cmake_cache_value(cache: pathlib.Path, name: str) -> str | None:
    prefix = f"{name}:"
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    return None


def release_build_identity(
    teacher: pathlib.Path,
    continuation_generator: pathlib.Path | None,
) -> dict:
    """Record the exact CMake cache and native producers used by a round."""

    binaries = {"teacher": artifact_snapshot(teacher)}
    if continuation_generator is not None:
        binaries["continuation_generator"] = artifact_snapshot(
            continuation_generator
        )
    parents = {pathlib.Path(item["path"]).parent for item in binaries.values()}
    build_directory = next(iter(parents)) if len(parents) == 1 else None
    cache = build_directory / "CMakeCache.txt" if build_directory else None
    cache_record = artifact_snapshot(cache) if cache and cache.is_file() else None
    return {
        "build_directory": str(build_directory) if build_directory else None,
        "cmake_cache": cache_record,
        "cmake_build_type": (
            _cmake_cache_value(cache, "CMAKE_BUILD_TYPE")
            if cache_record is not None
            else None
        ),
        "cxx_compiler": (
            _cmake_cache_value(cache, "CMAKE_CXX_COMPILER")
            if cache_record is not None
            else None
        ),
        "binaries": binaries,
    }


def _current_release_build_identity(record: object) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("binaries"), dict):
        raise ValueError("workflow Release build identity is missing")
    binaries = record["binaries"]
    teacher = binaries.get("teacher")
    continuation = binaries.get("continuation_generator")
    if not isinstance(teacher, dict) or not isinstance(teacher.get("path"), str):
        raise ValueError("workflow teacher build identity is missing")
    if continuation is not None and (
        not isinstance(continuation, dict)
        or not isinstance(continuation.get("path"), str)
    ):
        raise ValueError("workflow continuation build identity is invalid")
    return release_build_identity(
        pathlib.Path(teacher["path"]),
        pathlib.Path(continuation["path"]) if continuation is not None else None,
    )


def source_closure_sha256(repository: pathlib.Path) -> str:
    """Bind the candidate evaluator/search sources used by campaign actors."""

    repository = repository.resolve()
    candidates = [repository / path for path in CANDIDATE_SOURCE_CLOSURE_PATHS]
    candidates.extend(
        sorted((repository / "src/bots/jacek_replay_bfm").glob("**/*"))
    )
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise ValueError("candidate search source closure is empty")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        payload_hash = bytes.fromhex(sha256(path))
        digest.update(payload_hash)
    return digest.hexdigest()


class StageManager:
    """Fail-closed stage execution with atomic, recursively verified receipts."""

    def __init__(
        self,
        *,
        output: pathlib.Path,
        campaign_id: str,
        round_index: int,
        resume: bool,
        environment: dict,
    ) -> None:
        self.output = output.resolve()
        self.receipts = self.output / "receipts"
        self.campaign_id = campaign_id
        self.round_index = round_index
        self.resume = resume
        self.environment = environment

    def receipt_path(self, ordinal: int, name: str) -> pathlib.Path:
        return self.receipts / f"{ordinal:02d}-{name}.json"

    def execute(
        self,
        *,
        ordinal: int,
        name: str,
        configuration: dict,
        producers: dict[str, pathlib.Path],
        inputs: dict[str, pathlib.Path],
        outputs: dict[str, pathlib.Path],
        action: Callable[[], dict | None],
        validator: Callable[[dict], None] | None = None,
        resumable_outputs: set[str] | None = None,
    ) -> dict:
        receipt_path = self.receipt_path(ordinal, name)
        expected = {
            "schema": STAGE_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "round": self.round_index,
            "ordinal": ordinal,
            "stage": name,
            "configuration": configuration,
            "environment": self.environment,
            "producers": _snapshots(producers),
            "inputs": _snapshots(inputs),
        }
        if receipt_path.exists():
            if not self.resume:
                raise ValueError(
                    f"stage {name} already has a receipt; use --resume or a fresh "
                    "attempt directory"
                )
            receipt = _load_json(receipt_path, f"{name} stage receipt")
            for field, value in expected.items():
                if receipt.get(field) != value:
                    raise ValueError(f"stage {name} receipt {field} is stale")
            expected_outputs = _snapshots(outputs)
            if receipt.get("outputs") != expected_outputs:
                raise ValueError(f"stage {name} receipt outputs are stale or corrupt")
            if not isinstance(receipt.get("result"), dict):
                raise ValueError(f"stage {name} receipt result is invalid")
            if validator is not None:
                validator(receipt["result"])
            return receipt["result"]

        allowed = resumable_outputs or set()
        collisions = [
            str(path)
            for label, path in outputs.items()
            if path.exists() and not (self.resume and label in allowed)
        ]
        if collisions:
            raise ValueError(
                f"stage {name} has unreceipted output ({', '.join(collisions)}); "
                "use a fresh attempt directory"
            )
        result = action() or {}
        if not isinstance(result, dict):
            raise ValueError(f"stage {name} returned a non-object result")
        if validator is not None:
            validator(result)
        receipt = {
            **expected,
            "outputs": _snapshots(outputs),
            "result": result,
        }
        atomic_write(receipt_path, canonical_json_bytes(receipt, pretty=True))
        return result


def stage_receipt_bindings(
    manager: StageManager, *, include_root_labels: bool
) -> list[dict]:
    stages = [
        (0, "roots"),
        (1, "teacher-tsv"),
        *(([(2, "root-labels")]) if include_root_labels else []),
        (3, "continuations"),
        (4, "continuation-labels"),
        (5, "concatenation"),
        (6, "packing"),
        (7, "training"),
        (8, "selected-runtime"),
    ]
    bindings = []
    for ordinal, stage in stages:
        path = manager.receipt_path(ordinal, stage)
        if not path.is_file():
            raise ValueError(f"completed workflow is missing the {stage} receipt")
        bindings.append(
            {
                "ordinal": ordinal,
                "stage": stage,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    return bindings


def _validate_snapshot_records(records: object, label: str) -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError(f"canonical workflow {label} snapshots are incomplete")
    for record in records.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"canonical workflow {label} snapshot is invalid")
        if artifact_snapshot(pathlib.Path(record["path"])) != record:
            raise ValueError(f"canonical workflow {label} snapshot is stale")


def validate_final_workflow_stage_receipt(
    workflow_path: pathlib.Path, receipt: dict, bindings: list[dict]
) -> None:
    """Verify the derived final-stage sidecar without a self-hash cycle."""

    path = workflow_path.resolve().parent / "receipts" / "09-workflow.json"
    final = _load_json(path, "final workflow stage receipt")
    configuration = receipt.get("configuration", {})
    execution = receipt.get("execution", {})
    expected_configuration = {
        "campaign_id": configuration.get("campaign_id"),
        "round": configuration.get("round"),
        "campaign_eligible": configuration.get("campaign_eligible"),
    }
    if (
        final.get("schema") != STAGE_RECEIPT_SCHEMA
        or final.get("campaign_id") != configuration.get("campaign_id")
        or final.get("round") != configuration.get("round")
        or final.get("ordinal") != 9
        or final.get("stage") != "workflow"
        or final.get("configuration") != expected_configuration
        or final.get("environment") != execution.get("environment")
        or final.get("outputs") != {"workflow": artifact_snapshot(workflow_path)}
        or final.get("result") != {"workflow_sha256": sha256(workflow_path)}
    ):
        raise ValueError("final workflow stage receipt is stale or corrupt")
    if set(final.get("producers", {})) != {
        "workflow",
        "corpus",
        "pack",
        "trainer",
        "features",
    }:
        raise ValueError("final workflow stage producer set is incomplete")
    _validate_snapshot_records(final.get("producers"), "final-stage producer")

    expected_inputs = {
        "roots": pathlib.Path(receipt["artifacts"]["roots"]["path"]),
        "teacher_tsv": pathlib.Path(receipt["artifacts"]["teacher_tsv"]["path"]),
        "teacher_jsonl": pathlib.Path(
            receipt["artifacts"]["teacher_jsonl"]["path"]
        ),
        "pack_report": pathlib.Path(receipt["artifacts"]["pack_report"]["report"]),
        "model_manifest": pathlib.Path(
            receipt["artifacts"]["model"]["manifest_path"]
        ),
        "runtime": pathlib.Path(receipt["artifacts"]["model"]["runtime_path"]),
    }
    expected_inputs.update(
        {
            f"stage_receipt_{binding['ordinal']}": pathlib.Path(binding["path"])
            for binding in bindings
        }
    )
    if final.get("inputs") != _snapshots(expected_inputs):
        raise ValueError("final workflow stage inputs are stale or incomplete")


def _valid_metric_report(value: object, expected_samples: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "samples",
        "weighted_huber",
        "sign_accuracy",
        "correlation",
        "mae",
        "prediction_mean",
    }:
        return False
    if value.get("samples") != expected_samples:
        return False
    numbers = [
        value.get("weighted_huber"),
        value.get("sign_accuracy"),
        value.get("correlation"),
        value.get("mae"),
        value.get("prediction_mean"),
    ]
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        for number in numbers
    ):
        return False
    return (
        value["weighted_huber"] >= 0.0
        and 0.0 <= value["sign_accuracy"] <= 1.0
        and -1.000_000_1 <= value["correlation"] <= 1.000_000_1
        and value["mae"] >= 0.0
    )


def validate_test_reveal_contract(model_manifest: dict, round_index: int) -> None:
    training = model_manifest.get("training", {})
    reports = training.get("seed_reports")
    if not isinstance(reports, list):
        raise ValueError("model seed reports are missing")
    revealed = [report for report in reports if isinstance(report, dict) and "test" in report]
    if round_index < 2:
        if revealed:
            raise ValueError("intermediate campaign round exposed test metrics")
        return
    chosen_seed = training.get("chosen_seed")
    test_shards = [
        shard
        for shard in model_manifest.get("source_shards", [])
        if isinstance(shard, dict) and shard.get("split") == "test"
    ]
    if not test_shards or any(
        isinstance(shard.get("samples"), bool)
        or not isinstance(shard.get("samples"), int)
        or shard["samples"] < 0
        for shard in test_shards
    ):
        raise ValueError("final model test-shard sample counts are invalid")
    expected_samples = sum(shard["samples"] for shard in test_shards)
    if (
        expected_samples <= 0
        or len(revealed) != 1
        or revealed[0].get("seed") != chosen_seed
        or not _valid_metric_report(revealed[0].get("test"), expected_samples)
    ):
        raise ValueError("final test metrics are not bound to the selected seed")


def validate_embedded_stage_receipts(
    workflow_path: pathlib.Path, receipt: dict
) -> None:
    execution = receipt.get("execution")
    configuration = receipt.get("configuration", {})
    if not isinstance(execution, dict):
        raise ValueError("canonical workflow execution provenance is missing")
    environment = execution.get("environment")
    feature_encoder = execution.get("feature_encoder")
    repository_path = execution.get("repository_path")
    closure_hash = execution.get("candidate_search_source_closure_sha256")
    repository = execution.get("repository")
    release_build = execution.get("release_build")
    if (
        execution.get("campaign_id") != configuration.get("campaign_id")
        or execution.get("resumable_stage_receipt_schema")
        != STAGE_RECEIPT_SCHEMA
        or not isinstance(environment, dict)
        or environment != environment_identity()
        or not isinstance(feature_encoder, dict)
        or not isinstance(feature_encoder.get("path"), str)
        or not pathlib.Path(feature_encoder["path"]).is_file()
        or not isinstance(repository_path, str)
        or not pathlib.Path(repository_path).is_dir()
        or not isinstance(repository, dict)
        or repository.get("path") != str(pathlib.Path(repository_path).resolve())
        or repository_identity(pathlib.Path(repository_path)) != repository
        or _current_release_build_identity(release_build) != release_build
        or artifact_snapshot(pathlib.Path(feature_encoder.get("path", "")))
        != feature_encoder
        or not isinstance(closure_hash, str)
        or source_closure_sha256(pathlib.Path(repository_path)) != closure_hash
    ):
        raise ValueError("canonical workflow execution provenance is stale")
    if configuration.get("campaign_eligible") is True and (
        repository.get("clean") is not True
        or release_build.get("cmake_build_type") != "Release"
        or "continuation_generator" not in release_build.get("binaries", {})
    ):
        raise ValueError("canonical workflow was not frozen from a clean Release build")
    bindings = execution.get("stage_receipts")
    expected_stages = [
        (0, "roots"),
        (1, "teacher-tsv"),
        *(([(2, "root-labels")]) if configuration.get("round") == 0 else []),
        (3, "continuations"),
        (4, "continuation-labels"),
        (5, "concatenation"),
        (6, "packing"),
        (7, "training"),
        (8, "selected-runtime"),
    ]
    if not isinstance(bindings, list) or [
        (item.get("ordinal"), item.get("stage"))
        for item in bindings
        if isinstance(item, dict)
    ] != expected_stages:
        raise ValueError("canonical workflow stage receipt list is incomplete")
    for binding in bindings:
        path = _receipt_path(workflow_path, binding.get("path"), "stage receipt")
        if sha256(path) != binding.get("sha256"):
            raise ValueError("canonical workflow stage receipt hash is stale")
        stage = _load_json(path, "stage receipt")
        if (
            stage.get("schema") != STAGE_RECEIPT_SCHEMA
            or stage.get("campaign_id") != configuration.get("campaign_id")
            or stage.get("round") != configuration.get("round")
            or stage.get("ordinal") != binding["ordinal"]
            or stage.get("stage") != binding["stage"]
            or stage.get("environment") != environment
        ):
            raise ValueError("canonical workflow stage receipt identity is invalid")
        for category in ("producers", "inputs", "outputs"):
            records = stage.get(category)
            if not isinstance(records, dict) or not records:
                raise ValueError(
                    f"canonical workflow stage receipt {category} is incomplete"
                )
            for record in records.values():
                if not isinstance(record, dict) or not isinstance(
                    record.get("path"), str
                ):
                    raise ValueError("canonical workflow stage artifact is invalid")
                if artifact_snapshot(pathlib.Path(record["path"])) != record:
                    raise ValueError("canonical workflow stage artifact is stale")
        if binding["stage"] in {"root-labels", "continuation-labels"}:
            validate_teacher_chunks_result(stage.get("result", {}))
    validate_final_workflow_stage_receipt(workflow_path, receipt, bindings)


def validate_model_seed_checkpoints(receipt: dict, model_manifest: dict) -> None:
    bindings = receipt.get("execution", {}).get("stage_receipts", [])
    training_binding = next(
        (item for item in bindings if item.get("stage") == "training"), None
    )
    if not isinstance(training_binding, dict):
        raise ValueError("workflow training stage receipt is missing")
    training_receipt = _load_json(
        pathlib.Path(training_binding["path"]), "training stage receipt"
    )
    seed_output = training_receipt.get("outputs", {}).get("seed_checkpoints")
    publications = model_manifest.get("training", {}).get("seed_checkpoints")
    seeds = model_manifest.get("training", {}).get("seeds")
    if (
        not isinstance(seed_output, dict)
        or seed_output.get("kind") != "directory"
        or not isinstance(publications, list)
        or not isinstance(seeds, list)
        or [item.get("seed") for item in publications if isinstance(item, dict)]
        != seeds
    ):
        raise ValueError("model seed checkpoint publication list is invalid")
    directory = pathlib.Path(seed_output["path"])
    expected_files = set()
    for publication in publications:
        seed = publication["seed"]
        checkpoint_name = publication.get("checkpoint")
        receipt_name = publication.get("receipt")
        if (
            not isinstance(checkpoint_name, str)
            or pathlib.Path(checkpoint_name).name != checkpoint_name
            or not isinstance(receipt_name, str)
            or pathlib.Path(receipt_name).name != receipt_name
        ):
            raise ValueError("model seed checkpoint filename is invalid")
        checkpoint_path = directory / checkpoint_name
        seed_receipt_path = directory / receipt_name
        expected_files.update((checkpoint_name, receipt_name))
        if (
            sha256(checkpoint_path) != publication.get("checkpoint_sha256")
            or sha256(seed_receipt_path) != publication.get("receipt_sha256")
        ):
            raise ValueError("model seed checkpoint hash is stale")
        seed_receipt_payload = seed_receipt_path.read_bytes()
        seed_receipt = _load_json(seed_receipt_path, "seed checkpoint receipt")
        body = dict(seed_receipt)
        body_hash = body.pop("body_sha256", None)
        checkpoint = seed_receipt.get("checkpoint", {})
        if (
            seed_receipt_payload != canonical_json_bytes(seed_receipt)
            or seed_receipt.get("schema")
            != "papersoccer.jacek-replay-bfm-seed-checkpoint.v1"
            or seed_receipt.get("seed") != seed
            or body_hash != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
            or checkpoint.get("file") != checkpoint_name
            or checkpoint.get("artifact_sha256")
            != publication.get("checkpoint_sha256")
        ):
            raise ValueError("model seed checkpoint receipt is stale or corrupt")
    if {path.name for path in directory.iterdir() if path.is_file()} != expected_files:
        raise ValueError("model seed checkpoint directory has unknown files")
    chosen_seed = model_manifest.get("training", {}).get("chosen_seed")
    selected = next(
        (item for item in publications if item.get("seed") == chosen_seed), None
    )
    if (
        not isinstance(selected, dict)
        or selected.get("checkpoint_sha256")
        != model_manifest.get("runtime", {}).get("artifact_sha256")
    ):
        raise ValueError("selected runtime is not the chosen seed checkpoint")


def _receipt_path(receipt_path: pathlib.Path, raw: object, label: str) -> pathlib.Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"canonical workflow {label} path is missing")
    path = pathlib.Path(raw)
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def _load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_entry(receipt_path: pathlib.Path, receipt: dict) -> dict:
    artifacts = receipt["artifacts"]
    model = artifacts["model"]
    pack_path = _receipt_path(
        receipt_path, artifacts["pack_report"].get("report"), "pack report"
    )
    return {
        "round": receipt["configuration"]["round"],
        "workflow_path": str(receipt_path.resolve()),
        "workflow_sha256": sha256(receipt_path),
        "roots_path": str(
            _receipt_path(receipt_path, artifacts["roots"]["path"], "roots")
        ),
        "roots_sha256": artifacts["roots"]["sha256"],
        "pack_report_path": str(pack_path),
        "pack_report_sha256": sha256(pack_path),
        "model_manifest_path": str(
            _receipt_path(receipt_path, model["manifest_path"], "model manifest")
        ),
        "model_manifest_sha256": model["manifest_sha256"],
        "runtime_path": str(
            _receipt_path(receipt_path, model["runtime_path"], "runtime")
        ),
        "runtime_sha256": model["runtime_sha256"],
    }


def validate_canonical_workflow_chain(
    receipt_path: pathlib.Path, expected_round: int | None = None
) -> dict:
    """Validate a canonical receipt and every file-backed ancestor round."""

    receipt_path = receipt_path.resolve()
    receipt = _load_json(receipt_path, "workflow receipt")
    if receipt.get("schema") != WORKFLOW_SCHEMA:
        raise ValueError("canonical workflow receipt has the wrong schema")
    configuration = receipt.get("configuration")
    inputs = receipt.get("inputs")
    artifacts = receipt.get("artifacts")
    lineage = receipt.get("lineage")
    producer = receipt.get("producer")
    if not all(isinstance(value, dict) for value in (
        configuration, inputs, artifacts, lineage, producer
    )):
        raise ValueError("canonical workflow receipt is incomplete")
    required_producers = {"workflow", "corpus", "pack", "trainer"}
    if set(producer) != required_producers or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in producer.values()
    ):
        raise ValueError("canonical workflow producer hashes are invalid")
    round_index = configuration.get("round")
    if round_index not in (0, 1, 2) or (
        expected_round is not None and round_index != expected_round
    ):
        raise ValueError("canonical workflow round is not the expected predecessor")
    expected_configuration = {
        **CANONICAL_CONFIGURATION,
        "profile": None,
        "campaign_eligible": True,
        "round": round_index,
        "final_test_revealed": round_index == 2,
    }
    for field, expected in expected_configuration.items():
        if configuration.get(field) != expected:
            raise ValueError(f"canonical workflow configuration.{field} is not frozen")
    if len(configuration.get("prior_pack_report_sha256", ())) != round_index:
        raise ValueError("canonical workflow prior-pack count is inconsistent")

    roots = artifacts.get("roots")
    teacher_tsv_artifact = artifacts.get("teacher_tsv")
    teacher_jsonl_artifact = artifacts.get("teacher_jsonl")
    continuation_artifact = artifacts.get("continuations")
    model = artifacts.get("model")
    pack = artifacts.get("pack_report")
    training = artifacts.get("training")
    if not all(
        isinstance(value, dict)
        for value in (
            roots,
            teacher_tsv_artifact,
            teacher_jsonl_artifact,
            continuation_artifact,
            model,
            pack,
            training,
        )
    ):
        raise ValueError("canonical workflow artifacts are incomplete")
    roots_path = _receipt_path(receipt_path, roots.get("path"), "roots")
    if sha256(roots_path) != roots.get("sha256"):
        raise ValueError("canonical workflow roots hash is stale")
    teacher_tsv_path = _receipt_path(
        receipt_path, teacher_tsv_artifact.get("path"), "teacher TSV"
    )
    if sha256(teacher_tsv_path) != teacher_tsv_artifact.get("sha256"):
        raise ValueError("canonical workflow teacher TSV hash is stale")
    _teacher_tsv_parts(teacher_tsv_path)
    teacher_jsonl_path = _receipt_path(
        receipt_path, teacher_jsonl_artifact.get("path"), "teacher JSONL"
    )
    teacher_rows = validate_teacher_output(teacher_jsonl_path)
    if (
        sha256(teacher_jsonl_path) != teacher_jsonl_artifact.get("sha256")
        or teacher_rows != teacher_jsonl_artifact.get("rows")
    ):
        raise ValueError("canonical workflow teacher JSONL binding is stale")
    continuations_path = _receipt_path(
        receipt_path, continuation_artifact.get("path"), "continuations TSV"
    )
    continuation_manifest_path = _receipt_path(
        receipt_path,
        continuation_artifact.get("manifest_path"),
        "continuation manifest",
    )
    if (
        sha256(continuations_path) != continuation_artifact.get("sha256")
        or sha256(continuation_manifest_path)
        != continuation_artifact.get("manifest_sha256")
    ):
        raise ValueError("canonical workflow continuation artifacts are stale")
    pack_path = _receipt_path(receipt_path, pack.get("report"), "pack report")
    pack_payload = _load_json(pack_path, "pack report")
    embedded_pack = dict(pack)
    embedded_pack.pop("report", None)
    if (
        pack_payload != embedded_pack
        or pack_payload.get("schema")
        != "papersoccer.jacek-replay-pack-report.v1"
        or pack_payload.get("packing") != "sqlite-streaming-bounded-memory-v1"
        or pack_payload.get("roots_manifest_sha256") != roots.get("sha256")
    ):
        raise ValueError("canonical workflow embedded pack report is stale")
    model_manifest_path = _receipt_path(
        receipt_path, model.get("manifest_path"), "model manifest"
    )
    if sha256(model_manifest_path) != model.get("manifest_sha256"):
        raise ValueError("canonical workflow model manifest hash is stale")
    model_manifest = _load_json(model_manifest_path, "model manifest")
    campaign = model_manifest.get("campaign_contract")
    model_training = model_manifest.get("training")
    if (
        model_manifest.get("schema") != MODEL_SCHEMA
        or model_manifest.get("status")
        != "canonical-campaign-candidate-not-game-gated"
        or not isinstance(campaign, dict)
        or campaign.get("eligible") is not True
        or campaign.get("profile") is not None
        or campaign.get("round") != round_index
        or campaign.get("prior_rounds") != round_index
        or campaign.get("test_revealed") != (round_index == 2)
    ):
        raise ValueError("canonical workflow model campaign contract is invalid")
    if not isinstance(model_training, dict):
        raise ValueError("canonical workflow model training selection is missing")
    validate_model_seed_checkpoints(receipt, model_manifest)
    selected_seed = model_training.get("chosen_seed")
    seed_reports = model_training.get("seed_reports")
    if (
        model_training.get("seeds") != [20260823, 20260824, 20260825]
        or not isinstance(seed_reports, list)
        or len(seed_reports) != 3
        or len({report.get("seed") for report in seed_reports if isinstance(report, dict)})
        != 3
        or sum(
            isinstance(report, dict) and report.get("seed") == selected_seed
            for report in seed_reports
        )
        != 1
        or model_training.get("test_revealed_after_selection")
        != (round_index == 2)
    ):
        raise ValueError("canonical workflow selected-seed contract is invalid")
    validate_test_reveal_contract(model_manifest, round_index)
    campaign_fields = {
        "continuation_games": "continuation_games",
        "bulk_nodes": "nodes",
        "root_and_deep_nodes": "root_nodes",
        "deep_percent": "deep_percent",
        "max_samples_per_game": "max_samples_per_game",
        "actor_nodes": "actor_nodes",
        "candidate_tree_nodes": "candidate_tree_nodes",
        "campaign_id": "campaign_id",
        "teacher_workers": "teacher_workers",
        "teacher_chunk_games": "teacher_chunk_games",
        "seed_workers": "seed_workers",
    }
    if any(
        campaign.get(campaign_field) != CANONICAL_CONFIGURATION[config_field]
        for campaign_field, config_field in campaign_fields.items()
    ):
        raise ValueError("canonical model campaign parameters are not frozen")
    runtime_path = _receipt_path(receipt_path, model.get("runtime_path"), "runtime")
    runtime_hash = sha256(runtime_path)
    if (
        runtime_hash != model.get("runtime_sha256")
        or runtime_hash != training.get("artifact_sha256")
        or runtime_hash != model_manifest.get("runtime", {}).get("artifact_sha256")
    ):
        raise ValueError("canonical workflow selected runtime binding is stale")

    previous = lineage.get("previous_workflow")
    declared_ancestors = lineage.get("ancestors")
    if not isinstance(declared_ancestors, list):
        raise ValueError("canonical workflow ancestor list is missing")
    if round_index == 0:
        if (
            previous is not None
            or declared_ancestors
            or inputs.get("previous_roots_sha256") is not None
            or inputs.get("continuation_model_sha256") is not None
            or configuration.get("prior_pack_report_sha256") != []
        ):
            raise ValueError("canonical round 0 must not have prior-round lineage")
        ancestors = []
    else:
        if not isinstance(previous, dict):
            raise ValueError("canonical workflow predecessor is missing")
        previous_path = _receipt_path(
            receipt_path, previous.get("path"), "previous workflow"
        )
        if sha256(previous_path) != previous.get("sha256"):
            raise ValueError("canonical previous-workflow hash is stale")
        previous_validation = validate_canonical_workflow_chain(
            previous_path, round_index - 1
        )
        ancestors = previous_validation["entries"]
        if declared_ancestors != ancestors:
            raise ValueError("canonical workflow ancestor chain was edited")
        immediate = ancestors[-1]
        if (
            inputs.get("previous_roots_sha256") != immediate["roots_sha256"]
            or inputs.get("continuation_model_sha256")
            != immediate["runtime_sha256"]
            or configuration.get("prior_pack_report_sha256")
            != [entry["pack_report_sha256"] for entry in ancestors]
        ):
            raise ValueError("canonical workflow predecessor artifacts do not match")
        previous_receipt = previous_validation["receipt"]
        if producer != previous_receipt.get("producer"):
            raise ValueError("canonical workflow producer changed across rounds")
        if configuration.get("campaign_id") != previous_receipt[
            "configuration"
        ].get("campaign_id"):
            raise ValueError("canonical workflow campaign_id changed across rounds")
        current_execution = receipt.get("execution", {})
        previous_execution = previous_receipt.get("execution", {})
        for field in (
            "environment",
            "repository_path",
            "repository",
            "release_build",
            "feature_encoder",
            "candidate_search_source_closure_sha256",
        ):
            if current_execution.get(field) != previous_execution.get(field):
                raise ValueError(
                    f"canonical workflow execution {field} changed across rounds"
                )
        for field in (
            "exclusions_sha256",
            "public_jacek_sha256",
            "live_snapshot_sha256",
            "teacher_executable_sha256",
            "continuation_generator_sha256",
        ):
            if inputs.get(field) != previous_receipt["inputs"].get(field):
                raise ValueError(f"canonical workflow input {field} changed across rounds")

    continuation_payload = validate_continuation_manifest(
        continuation_manifest_path,
        continuations_path,
        round_index=round_index,
        games=CANONICAL_CONFIGURATION["continuation_games"],
        input_path=teacher_tsv_path,
        model_path=(
            pathlib.Path(ancestors[-1]["runtime_path"]) if ancestors else None
        ),
    )
    if (
        continuation_artifact.get("successful_games")
        != continuation_payload.get("successful_games")
        or continuation_artifact.get("successful_quotas")
        != continuation_payload.get("successful_quotas")
    ):
        raise ValueError("canonical workflow continuation counts are stale")

    expected_prior_shards = []
    for ancestor in ancestors:
        ancestor_pack = _load_json(
            pathlib.Path(ancestor["pack_report_path"]), "ancestor pack report"
        )
        for split in ("train", "validation", "test"):
            shard = ancestor_pack.get("shards", {}).get(split, {})
            expected_prior_shards.append(
                {
                    "manifest_sha256": shard.get("manifest_sha256"),
                    "npz_sha256": shard.get("sha256"),
                    "split": split,
                }
            )
    if pack_payload.get("prior_shards", []) != expected_prior_shards:
        raise ValueError("canonical pack report does not bind prior round shards")

    entry = _canonical_entry(receipt_path, receipt)
    if (
        campaign.get("canonical_ancestry", []) != ancestors
        or campaign.get("previous_workflow_sha256")
        != (ancestors[-1]["workflow_sha256"] if ancestors else None)
    ):
        raise ValueError("canonical model ancestry differs from workflow ancestry")
    validate_embedded_stage_receipts(receipt_path, receipt)
    return {
        "receipt": receipt,
        "entry": entry,
        "entries": [*ancestors, entry],
    }


def validate_canonical_predecessor_inputs(
    *,
    round_index: int,
    previous_workflow: pathlib.Path,
    previous_roots: pathlib.Path,
    continuation_model: pathlib.Path,
    prior_pack_reports: list[pathlib.Path],
) -> dict:
    """Bind round N inputs to the exact canonical round N-1 receipt."""

    prior_chain = validate_canonical_workflow_chain(
        previous_workflow, round_index - 1
    )
    entries = prior_chain["entries"]
    immediate = entries[-1]
    if (
        previous_roots.resolve() != pathlib.Path(immediate["roots_path"])
        or sha256(previous_roots) != immediate["roots_sha256"]
    ):
        raise ValueError("--previous-roots is not the predecessor roots artifact")
    if (
        continuation_model.resolve() != pathlib.Path(immediate["runtime_path"])
        or sha256(continuation_model) != immediate["runtime_sha256"]
    ):
        raise ValueError(
            "--continuation-model is not the predecessor selected runtime"
        )
    expected_packs = [pathlib.Path(entry["pack_report_path"]) for entry in entries]
    supplied_packs = [path.resolve() for path in prior_pack_reports]
    if supplied_packs != expected_packs or [
        sha256(path) for path in supplied_packs
    ] != [entry["pack_report_sha256"] for entry in entries]:
        raise ValueError(
            "--prior-pack-report does not match canonical workflow ancestry"
        )
    return prior_chain


def validate_smoke_workflow_chain(
    receipt_path: pathlib.Path, expected_round: int | None = None
) -> dict:
    """Validate the frozen, explicitly non-promotable three-round smoke chain."""

    receipt_path = receipt_path.resolve()
    receipt = _load_json(receipt_path, "smoke workflow receipt")
    configuration = receipt.get("configuration")
    artifacts = receipt.get("artifacts")
    lineage = receipt.get("lineage")
    if (
        receipt.get("schema") != WORKFLOW_SCHEMA
        or not isinstance(configuration, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(lineage, dict)
        or configuration.get("profile") != SMOKE_PROFILE
        or configuration.get("campaign_eligible") is not False
    ):
        raise ValueError("smoke workflow identity is invalid")
    round_index = configuration.get("round")
    if round_index not in (0, 1, 2) or (
        expected_round is not None and round_index != expected_round
    ):
        raise ValueError("smoke workflow round is invalid")
    for field, expected in SMOKE_CONFIGURATION.items():
        config_field = "max_samples_per_game" if field == "max_samples" else field
        if configuration.get(config_field) != expected:
            raise ValueError(f"smoke workflow configuration.{config_field} is stale")
    if configuration.get("final_test_revealed") != (round_index == 2):
        raise ValueError("smoke workflow test-reveal boundary is invalid")
    if len(configuration.get("prior_pack_report_sha256", [])) != round_index:
        raise ValueError("smoke workflow prior-pack count is invalid")

    validate_embedded_stage_receipts(receipt_path, receipt)
    roots = artifacts.get("roots", {})
    pack = artifacts.get("pack_report", {})
    model = artifacts.get("model", {})
    roots_path = _receipt_path(receipt_path, roots.get("path"), "smoke roots")
    pack_path = _receipt_path(receipt_path, pack.get("report"), "smoke pack")
    runtime_path = _receipt_path(
        receipt_path, model.get("runtime_path"), "smoke runtime"
    )
    manifest_path = _receipt_path(
        receipt_path, model.get("manifest_path"), "smoke model manifest"
    )
    pack_payload = _load_json(pack_path, "smoke pack report")
    if (
        sha256(roots_path) != roots.get("sha256")
        or {key: value for key, value in pack.items() if key != "report"}
        != pack_payload
        or sha256(runtime_path) != model.get("runtime_sha256")
        or sha256(manifest_path) != model.get("manifest_sha256")
    ):
        raise ValueError("smoke workflow core artifact is stale")
    model_manifest = _load_json(manifest_path, "smoke model manifest")
    contract = model_manifest.get("campaign_contract", {})
    training = model_manifest.get("training", {})
    if (
        model_manifest.get("status") != "development-candidate-not-promotable"
        or contract.get("eligible") is not False
        or contract.get("profile") != SMOKE_PROFILE
        or contract.get("round") != round_index
        or contract.get("test_revealed") != (round_index == 2)
        or training.get("test_revealed_after_selection") != (round_index == 2)
    ):
        raise ValueError("smoke model is not explicitly development-only")
    validate_test_reveal_contract(model_manifest, round_index)
    validate_model_seed_checkpoints(receipt, model_manifest)

    previous = lineage.get("previous_workflow")
    declared_ancestors = lineage.get("ancestors")
    if not isinstance(declared_ancestors, list):
        raise ValueError("smoke workflow ancestor list is missing")
    if round_index == 0:
        if previous is not None or declared_ancestors:
            raise ValueError("smoke round zero has prior lineage")
        ancestors = []
    else:
        if not isinstance(previous, dict):
            raise ValueError("smoke workflow predecessor is missing")
        previous_path = _receipt_path(
            receipt_path, previous.get("path"), "smoke predecessor"
        )
        if sha256(previous_path) != previous.get("sha256"):
            raise ValueError("smoke predecessor hash is stale")
        previous_chain = validate_smoke_workflow_chain(
            previous_path, round_index - 1
        )
        ancestors = previous_chain["entries"]
        if declared_ancestors != ancestors:
            raise ValueError("smoke workflow ancestry was edited")
        immediate = ancestors[-1]
        inputs = receipt.get("inputs", {})
        if (
            inputs.get("previous_roots_sha256") != immediate["roots_sha256"]
            or inputs.get("continuation_model_sha256")
            != immediate["runtime_sha256"]
            or configuration.get("prior_pack_report_sha256")
            != [entry["pack_report_sha256"] for entry in ancestors]
        ):
            raise ValueError("smoke predecessor artifact binding is stale")
    teacher_tsv = artifacts.get("teacher_tsv", {})
    continuations = artifacts.get("continuations", {})
    teacher_tsv_path = _receipt_path(
        receipt_path, teacher_tsv.get("path"), "smoke teacher TSV"
    )
    continuation_path = _receipt_path(
        receipt_path, continuations.get("path"), "smoke continuations TSV"
    )
    continuation_manifest = _receipt_path(
        receipt_path,
        continuations.get("manifest_path"),
        "smoke continuation manifest",
    )
    if (
        sha256(teacher_tsv_path) != teacher_tsv.get("sha256")
        or sha256(continuation_path) != continuations.get("sha256")
        or sha256(continuation_manifest)
        != continuations.get("manifest_sha256")
    ):
        raise ValueError("smoke continuation inputs are stale")
    continuation_payload = validate_continuation_manifest(
        continuation_manifest,
        continuation_path,
        round_index=round_index,
        games=SMOKE_CONFIGURATION["continuation_games"],
        input_path=teacher_tsv_path,
        model_path=(
            pathlib.Path(ancestors[-1]["runtime_path"]) if ancestors else None
        ),
    )
    if (
        continuations.get("successful_games")
        != continuation_payload.get("successful_games")
        or continuations.get("successful_quotas")
        != continuation_payload.get("successful_quotas")
    ):
        raise ValueError("smoke continuation quotas are stale")
    entry = _canonical_entry(receipt_path, receipt)
    return {
        "receipt": receipt,
        "entry": entry,
        "entries": [*ancestors, entry],
    }


def validate_smoke_predecessor_inputs(
    *,
    round_index: int,
    previous_workflow: pathlib.Path,
    previous_roots: pathlib.Path,
    continuation_model: pathlib.Path,
    prior_pack_reports: list[pathlib.Path],
) -> dict:
    chain = validate_smoke_workflow_chain(previous_workflow, round_index - 1)
    immediate = chain["entries"][-1]
    if (
        previous_roots.resolve() != pathlib.Path(immediate["roots_path"])
        or sha256(previous_roots) != immediate["roots_sha256"]
        or continuation_model.resolve() != pathlib.Path(immediate["runtime_path"])
        or sha256(continuation_model) != immediate["runtime_sha256"]
    ):
        raise ValueError("smoke predecessor roots/runtime do not match")
    expected_packs = [pathlib.Path(entry["pack_report_path"]) for entry in chain["entries"]]
    supplied = [path.resolve() for path in prior_pack_reports]
    if supplied != expected_packs or [sha256(path) for path in supplied] != [
        entry["pack_report_sha256"] for entry in chain["entries"]
    ]:
        raise ValueError("smoke predecessor pack reports do not match")
    return chain


def run(command: list[str], *, stdin: pathlib.Path | None = None) -> str:
    input_handle = stdin.open("rb") if stdin else None
    try:
        completed = subprocess.run(
            command,
            stdin=input_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        if input_handle:
            input_handle.close()
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr.decode("utf-8", "replace")
        )
    if completed.stderr:
        raise RuntimeError(
            f"command wrote unexpected stderr: {' '.join(command)}\n"
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed.stdout.decode("utf-8")


def run_to_path(
    command: list[str], *, stdin: pathlib.Path, output: pathlib.Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with stdin.open("rb") as input_handle, tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as output_handle:
            temporary = pathlib.Path(output_handle.name)
            completed = subprocess.run(
                command,
                stdin=input_handle,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                check=False,
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                + completed.stderr.decode("utf-8", "replace")
            )
        if completed.stderr:
            raise RuntimeError(
                f"command wrote unexpected stderr: {' '.join(command)}\n"
                + completed.stderr.decode("utf-8", "replace")
            )
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def run_with_atomic_directory(
    command: list[str], *, output_flag: str, output: pathlib.Path
) -> str:
    """Run a producer in scratch space and atomically publish its directory."""

    if output.exists():
        raise ValueError(f"atomic directory output already exists: {output}")
    matches = [index for index, value in enumerate(command) if value == output_flag]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"command does not bind exactly one {output_flag}")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.inprogress.")
    )
    published = False
    try:
        staged_command = list(command)
        staged_command[matches[0] + 1] = str(scratch)
        stdout = run(staged_command)
        if not scratch.is_dir() or not any(scratch.iterdir()):
            raise RuntimeError("directory producer published no artifacts")
        os.replace(scratch, output)
        published = True
        return stdout.replace(str(scratch), str(output))
    finally:
        if not published:
            shutil.rmtree(scratch, ignore_errors=True)


def run_with_atomic_output(
    command: list[str], *, output_flag: str, output: pathlib.Path
) -> str:
    """Run a command whose named file output must become visible atomically."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
    temporary.unlink()
    replaced = list(command)
    try:
        index = replaced.index(output_flag)
    except ValueError as error:
        raise ValueError(f"command does not contain {output_flag}") from error
    replaced[index + 1] = str(temporary)
    try:
        stdout = run(replaced)
        if not temporary.is_file():
            raise RuntimeError(f"command did not create {output_flag} file")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
        return stdout
    finally:
        temporary.unlink(missing_ok=True)


def _teacher_tsv_parts(path: pathlib.Path) -> tuple[bytes, list[bytes]]:
    lines = path.read_bytes().splitlines(keepends=True)
    header = b"group_id\tsource\twinner\ttranscript"
    header_index = next(
        (index for index, line in enumerate(lines) if line.rstrip(b"\r\n") == header),
        None,
    )
    if header_index is None:
        raise ValueError(f"teacher TSV has no canonical header: {path}")
    prefix = b"".join(lines[: header_index + 1])
    if prefix and not prefix.endswith(b"\n"):
        prefix += b"\n"
    rows = [line if line.endswith(b"\n") else line + b"\n" for line in lines[header_index + 1 :]]
    if not rows or any(not row.strip() or row.startswith(b"#") for row in rows):
        raise ValueError(f"teacher TSV has no canonical contiguous data rows: {path}")
    if any(len(row.rstrip(b"\r\n").split(b"\t")) != 4 for row in rows):
        raise ValueError(f"teacher TSV contains a malformed data row: {path}")
    return prefix, rows


def _validate_chunk_receipt(
    path: pathlib.Path, expected: dict, output: pathlib.Path
) -> dict:
    receipt = _load_json(path, "teacher chunk receipt")
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"teacher chunk receipt {field} is stale")
    if receipt.get("output") != artifact_snapshot(output):
        raise ValueError("teacher chunk receipt output is stale or corrupt")
    rows = validate_teacher_output(output)
    if receipt.get("teacher_rows") != rows:
        raise ValueError("teacher chunk receipt row count is stale")
    return receipt


def run_teacher_chunks(
    *,
    manager: StageManager,
    stage_ordinal: int,
    stage_name: str,
    teacher: pathlib.Path,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    teacher_arguments: list[str],
    workers: int,
    chunk_games: int,
) -> dict:
    """Label deterministic game chunks and merge strictly by chunk ordinal."""

    prefix, rows = _teacher_tsv_parts(input_path)
    chunk_root = manager.output / "teacher-chunks" / stage_name
    receipt_root = manager.receipts / f"{stage_ordinal:02d}-{stage_name}-chunks"
    chunk_specs = []
    teacher_record = artifact_snapshot(teacher)
    for ordinal, begin in enumerate(range(0, len(rows), chunk_games)):
        chunk_input = chunk_root / f"chunk-{ordinal:06d}.tsv"
        chunk_output = chunk_root / f"chunk-{ordinal:06d}.jsonl"
        chunk_receipt = receipt_root / f"chunk-{ordinal:06d}.json"
        payload = prefix + b"".join(rows[begin : begin + chunk_games])
        if chunk_input.exists():
            if chunk_input.read_bytes() != payload:
                raise ValueError(
                    f"teacher chunk {ordinal} input exists with stale content"
                )
        else:
            atomic_write(chunk_input, payload)
        expected = {
            "schema": CHUNK_RECEIPT_SCHEMA,
            "campaign_id": manager.campaign_id,
            "round": manager.round_index,
            "stage": stage_name,
            "chunk_ordinal": ordinal,
            "game_row_begin": begin,
            "game_rows": min(chunk_games, len(rows) - begin),
            "configuration": {
                "teacher_arguments": teacher_arguments,
                "teacher_chunk_games": chunk_games,
            },
            "environment": manager.environment,
            "producer": teacher_record,
            "input": artifact_snapshot(chunk_input),
        }
        if chunk_receipt.exists():
            if not manager.resume:
                raise ValueError(
                    f"teacher chunk {ordinal} already has a receipt; use --resume"
                )
            receipt = _validate_chunk_receipt(
                chunk_receipt, expected, chunk_output
            )
            chunk_specs.append((ordinal, chunk_output, chunk_receipt, receipt))
            continue
        if chunk_output.exists():
            raise ValueError(
                f"teacher chunk {ordinal} has unreceipted output; use a fresh "
                "attempt directory"
            )
        chunk_specs.append((ordinal, chunk_output, chunk_receipt, expected))

    def execute_chunk(spec: tuple[int, pathlib.Path, pathlib.Path, dict]) -> dict:
        ordinal, chunk_output, chunk_receipt, expected = spec
        if chunk_receipt.exists():
            return _validate_chunk_receipt(chunk_receipt, expected, chunk_output)
        chunk_input = pathlib.Path(expected["input"]["path"])
        run_to_path(
            [str(teacher), *teacher_arguments],
            stdin=chunk_input,
            output=chunk_output,
        )
        teacher_rows = validate_teacher_output(chunk_output)
        receipt = {
            **expected,
            "output": artifact_snapshot(chunk_output),
            "teacher_rows": teacher_rows,
        }
        atomic_write(chunk_receipt, canonical_json_bytes(receipt, pretty=True))
        return receipt

    missing = [spec for spec in chunk_specs if not spec[2].exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute_chunk, spec): spec[0] for spec in missing}
            for future in concurrent.futures.as_completed(futures):
                future.result()

    ordered_outputs = []
    chunk_bindings = []
    for ordinal, chunk_output, chunk_receipt, expected in chunk_specs:
        receipt = _validate_chunk_receipt(chunk_receipt, expected, chunk_output)
        ordered_outputs.append(chunk_output)
        chunk_bindings.append(
            {
                "chunk_ordinal": ordinal,
                "game_row_begin": receipt["game_row_begin"],
                "game_rows": receipt["game_rows"],
                "teacher_rows": receipt["teacher_rows"],
                "input_sha256": receipt["input"]["sha256"],
                "output_sha256": receipt["output"]["sha256"],
                "input_path": receipt["input"]["path"],
                "output_path": receipt["output"]["path"],
                "receipt_path": str(chunk_receipt.resolve()),
                "receipt_sha256": sha256(chunk_receipt),
            }
        )
    concatenate(ordered_outputs, output_path)
    total_teacher_rows = validate_teacher_output(output_path)
    if total_teacher_rows != sum(item["teacher_rows"] for item in chunk_bindings):
        raise ValueError("merged teacher row count differs from chunk receipts")
    return {
        "input_games": len(rows),
        "teacher_rows": total_teacher_rows,
        "chunks": chunk_bindings,
    }


def validate_teacher_chunks_result(result: dict) -> None:
    chunks = result.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("teacher stage has no chunk bindings")
    if [item.get("chunk_ordinal") for item in chunks if isinstance(item, dict)] != list(
        range(len(chunks))
    ):
        raise ValueError("teacher stage chunk ordinals are invalid")
    total = 0
    for item in chunks:
        if not isinstance(item, dict):
            raise ValueError("teacher stage chunk binding is invalid")
        input_path = pathlib.Path(item.get("input_path", ""))
        output_path = pathlib.Path(item.get("output_path", ""))
        receipt_path = pathlib.Path(item.get("receipt_path", ""))
        if (
            not input_path.is_file()
            or sha256(input_path) != item.get("input_sha256")
            or not output_path.is_file()
            or sha256(output_path) != item.get("output_sha256")
            or not receipt_path.is_file()
            or sha256(receipt_path) != item.get("receipt_sha256")
        ):
            raise ValueError("teacher stage chunk binding is stale")
        chunk_receipt = _load_json(receipt_path, "teacher chunk receipt")
        producer = chunk_receipt.get("producer")
        if (
            chunk_receipt.get("schema") != CHUNK_RECEIPT_SCHEMA
            or chunk_receipt.get("chunk_ordinal") != item["chunk_ordinal"]
            or chunk_receipt.get("game_row_begin") != item["game_row_begin"]
            or chunk_receipt.get("game_rows") != item["game_rows"]
            or chunk_receipt.get("teacher_rows") != item["teacher_rows"]
            or chunk_receipt.get("input") != artifact_snapshot(input_path)
            or chunk_receipt.get("output") != artifact_snapshot(output_path)
            or not isinstance(producer, dict)
            or not isinstance(producer.get("path"), str)
            or artifact_snapshot(pathlib.Path(producer["path"])) != producer
        ):
            raise ValueError("teacher chunk receipt is stale or corrupt")
        total += validate_teacher_output(output_path)
    if total != result.get("teacher_rows"):
        raise ValueError("teacher stage chunk row total is stale")


def validate_continuation_manifest(
    manifest_path: pathlib.Path,
    tsv_path: pathlib.Path,
    *,
    round_index: int,
    games: int,
    input_path: pathlib.Path | None = None,
    model_path: pathlib.Path | None = None,
) -> dict:
    manifest = _load_json(manifest_path, "continuation manifest")
    if (
        manifest.get("schema") != CONTINUATION_MANIFEST_SCHEMA
        or manifest.get("tsv_schema")
        != "papersoccer.jacek-replay-continuations.v1"
    ):
        raise ValueError("continuation manifest has the wrong schema")
    if (
        manifest.get("round") != round_index
        or manifest.get("requested_games") != games
        or manifest.get("successful_games") != games
        or manifest.get("attempt_cap_per_requested_game") != 20
    ):
        raise ValueError("continuation manifest campaign counts are invalid")
    bindings = manifest.get("bindings")
    rows = manifest.get("rows")
    if (
        not isinstance(bindings, dict)
        or bindings.get("output_sha256") != sha256(tsv_path)
        or not isinstance(rows, list)
        or len(rows) != games
        or [row.get("row_ordinal") for row in rows if isinstance(row, dict)]
        != list(range(games))
    ):
        raise ValueError("continuation manifest row/TSV binding is stale")
    if input_path is not None and bindings.get("input_sha256") != sha256(input_path):
        raise ValueError("continuation manifest input binding is stale")
    expected_model_hash = sha256(model_path) if model_path is not None else None
    if bindings.get("model_sha256") != expected_model_hash:
        raise ValueError("continuation manifest model binding is stale")
    expected_quotas = (
        {
            "rank4-vs-rank4": games,
            "candidate-selfplay": 0,
            "candidate-p1-vs-rank4": 0,
            "candidate-p2-vs-rank4": 0,
        }
        if round_index == 0
        else _planned_candidate_quotas(games)
    )
    if (
        manifest.get("planned_quotas") != expected_quotas
        or manifest.get("successful_quotas") != expected_quotas
    ):
        raise ValueError("continuation manifest actor quotas are invalid")
    if (
        not isinstance(manifest.get("attempts"), int)
        or not isinstance(manifest.get("failed_attempts"), int)
        or manifest["attempts"] != games + manifest["failed_attempts"]
        or manifest["attempts"] > games * 20
    ):
        raise ValueError("continuation manifest attempt accounting is invalid")

    _, tsv_rows_raw = _teacher_tsv_parts(tsv_path)
    tsv_rows = [row.rstrip(b"\r\n").decode("utf-8").split("\t") for row in tsv_rows_raw]
    if len(tsv_rows) != games:
        raise ValueError("continuation TSV game-row count is invalid")
    root_rows: dict[tuple[int, str], tuple[str, list[str]]] = {}
    if input_path is not None:
        _, input_rows_raw = _teacher_tsv_parts(input_path)
        for ordinal, raw in enumerate(input_rows_raw):
            group_id, _source, _winner, transcript = raw.rstrip(b"\r\n").decode(
                "utf-8"
            ).split("\t")
            root_rows[(ordinal, group_id)] = (
                hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                transcript.split("/"),
            )
    colors = {
        "rank4-vs-rank4": "none",
        "candidate-selfplay": "both",
        "candidate-p1-vs-rank4": "player-one",
        "candidate-p2-vs-rank4": "player-two",
    }
    identifiers: set[str] = set()
    attempt_ordinals = []
    actor_counts: collections.Counter[str] = collections.Counter()
    for ordinal, (row, tsv_row) in enumerate(zip(rows, tsv_rows, strict=True)):
        if not isinstance(row, dict):
            raise ValueError("continuation manifest row is invalid")
        identifier = row.get("continuation_id")
        actor_mode = row.get("actor_mode")
        lineage = row.get("root_lineage")
        attempt = row.get("attempt_ordinal")
        transcript = tsv_row[3]
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"continuation:[0-9a-f]{64}", identifier) is None
            or identifier in identifiers
            or row.get("row_ordinal") != ordinal
            or not isinstance(attempt, int)
            or attempt < 0
            or attempt >= manifest["attempts"]
            or actor_mode not in colors
            or row.get("candidate_color") != colors[actor_mode]
            or tsv_row[1] != f"continuation-round-{round_index}"
            or row.get("transcript_sha256")
            != hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            or not isinstance(lineage, dict)
            or lineage.get("group_id") != tsv_row[0]
        ):
            raise ValueError("continuation manifest row binding is invalid")
        if input_path is not None:
            root_key = (lineage.get("root_row_ordinal"), lineage.get("group_id"))
            root = root_rows.get(root_key)
            prefix_turns = lineage.get("prefix_turns")
            continuation_turns = transcript.split("/")
            if (
                root is None
                or lineage.get("root_transcript_sha256") != root[0]
                or not isinstance(prefix_turns, int)
                or prefix_turns < 0
                or prefix_turns > len(root[1])
                or continuation_turns[:prefix_turns] != root[1][:prefix_turns]
            ):
                raise ValueError("continuation root lineage is invalid")
        identifiers.add(identifier)
        attempt_ordinals.append(attempt)
        actor_counts[actor_mode] += 1
    if attempt_ordinals != sorted(set(attempt_ordinals)):
        raise ValueError("continuation attempt ordinals are not strictly increasing")
    if {
        name: actor_counts.get(name, 0) for name in expected_quotas
    } != expected_quotas:
        raise ValueError("continuation actor rows do not satisfy exact quotas")
    return manifest


def _planned_candidate_quotas(games: int) -> dict[str, int]:
    quotient, remainder = divmod(games, 4)
    counts = [2 * quotient + (2 * remainder) // 4, quotient, quotient]
    residuals = [(2 * remainder) % 4, remainder, remainder]
    for index in sorted(range(3), key=lambda item: (-residuals[item], item)):
        if sum(counts) == games:
            break
        counts[index] += 1
    return {
        "rank4-vs-rank4": 0,
        "candidate-selfplay": counts[0],
        "candidate-p1-vs-rank4": counts[1],
        "candidate-p2-vs-rank4": counts[2],
    }


def validate_teacher_output(path: pathlib.Path) -> int:
    count = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"teacher output line {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, dict) or row.get("schema") != (
                "papersoccer.jacek-replay-teacher.v1"
            ):
                raise ValueError(
                    f"teacher output line {line_number} has the wrong schema"
                )
            count += 1
    if count == 0:
        raise ValueError("teacher produced no rows")
    return count


def concatenate(paths: list[pathlib.Path], output: pathlib.Path) -> None:
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as destination:
            temporary = pathlib.Path(destination.name)
            for path in paths:
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--exclusions", type=pathlib.Path, required=True)
    parser.add_argument("--public-jacek", type=pathlib.Path, required=True)
    parser.add_argument("--live-snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--previous-workflow", type=pathlib.Path)
    parser.add_argument("--previous-roots", type=pathlib.Path)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--continuation-generator", type=pathlib.Path)
    parser.add_argument("--continuation-model", type=pathlib.Path)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--campaign-id", default=CANONICAL_CAMPAIGN_ID)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-workers", type=int, default=10)
    parser.add_argument("--teacher-chunk-games", type=int, default=25)
    parser.add_argument("--seed-workers", type=int, default=2)
    parser.add_argument("--continuation-games", type=int, default=0)
    parser.add_argument("--actor-nodes", type=int, default=16_000)
    parser.add_argument("--candidate-tree-nodes", type=int, default=2_000)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--nodes", type=int, default=32_000)
    parser.add_argument("--root-nodes", type=int, default=400_000)
    parser.add_argument("--deep-percent", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="20260823,20260824,20260825")
    parser.add_argument("--final-test-reveal", action="store_true")
    parser.add_argument("--development-mode", action="store_true")
    parser.add_argument(
        "--smoke-profile",
        action="store_true",
        help="use the frozen, non-promotable three-round development profile",
    )
    parser.add_argument("--prior-pack-report", type=pathlib.Path, action="append", default=[])
    arguments = parser.parse_args()
    if arguments.smoke_profile:
        for field, value in SMOKE_CONFIGURATION.items():
            setattr(arguments, field, value)
        arguments.development_mode = True
        arguments.final_test_reveal = arguments.round == 2
    if (
        arguments.nodes <= 0
        or arguments.root_nodes <= 0
        or not 1 <= arguments.deep_percent <= 100
        or arguments.max_samples <= 0
        or arguments.epochs <= 0
        or arguments.patience <= 0
        or arguments.batch_size <= 0
        or arguments.round < 0
        or arguments.round > 2
        or arguments.continuation_games < 0
        or arguments.actor_nodes <= 0
        or arguments.candidate_tree_nodes < 2
        or arguments.teacher_workers <= 0
        or arguments.teacher_chunk_games <= 0
        or arguments.seed_workers <= 0
    ):
        parser.error("numeric workflow limits must be positive")
    if (
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", arguments.campaign_id)
        or arguments.campaign_id in {".", ".."}
    ):
        parser.error("--campaign-id must be a safe non-empty identifier")
    teacher = arguments.teacher.resolve()
    if not teacher.is_file() or not os.access(teacher, os.X_OK):
        parser.error("--teacher must identify an executable file")
    continuation_generator = (
        arguments.continuation_generator.resolve()
        if arguments.continuation_generator
        else None
    )
    if arguments.continuation_games > 0:
        if (
            continuation_generator is None
            or not continuation_generator.is_file()
            or not os.access(continuation_generator, os.X_OK)
        ):
            parser.error("continuation games require an executable generator")
        if arguments.round > 0 and arguments.continuation_model is None:
            parser.error("rounds one and two require --continuation-model")
    if arguments.round > 0 and (
        arguments.continuation_games == 0
        or not arguments.prior_pack_report
        or arguments.previous_roots is None
    ):
        parser.error(
            "rounds one and two require continuations, previous roots, and prior packs"
        )
    if arguments.smoke_profile and arguments.round > 0 and (
        arguments.previous_workflow is None or arguments.continuation_model is None
    ):
        parser.error(
            "smoke rounds one and two require --previous-workflow and "
            "--continuation-model"
        )
    if len(arguments.prior_pack_report) != arguments.round:
        parser.error("round N requires exactly N prior pack reports")
    if arguments.final_test_reveal and arguments.round != 2:
        parser.error("--final-test-reveal is valid only for round 2")
    if arguments.round == 0 and (
        arguments.previous_workflow is not None
        or arguments.previous_roots is not None
        or arguments.continuation_model is not None
        or arguments.prior_pack_report
    ):
        parser.error("round 0 rejects every prior-round input")
    campaign_eligible = not arguments.development_mode
    if campaign_eligible:
        frozen = {
            "continuation_games": 10_000,
            "nodes": 32_000,
            "root_nodes": 400_000,
            "deep_percent": 10,
            "max_samples": 100,
            "actor_nodes": 16_000,
            "candidate_tree_nodes": 2_000,
            "epochs": 50,
            "patience": 8,
            "batch_size": 256,
            "seeds": "20260823,20260824,20260825",
            "campaign_id": CANONICAL_CAMPAIGN_ID,
            "teacher_workers": 10,
            "teacher_chunk_games": 25,
            "seed_workers": 2,
        }
        actual = {
            "continuation_games": arguments.continuation_games,
            "nodes": arguments.nodes,
            "root_nodes": arguments.root_nodes,
            "deep_percent": arguments.deep_percent,
            "max_samples": arguments.max_samples,
            "actor_nodes": arguments.actor_nodes,
            "candidate_tree_nodes": arguments.candidate_tree_nodes,
            "epochs": arguments.epochs,
            "patience": arguments.patience,
            "batch_size": arguments.batch_size,
            "seeds": arguments.seeds,
            "campaign_id": arguments.campaign_id,
            "teacher_workers": arguments.teacher_workers,
            "teacher_chunk_games": arguments.teacher_chunk_games,
            "seed_workers": arguments.seed_workers,
        }
        if actual != frozen or continuation_generator is None:
            parser.error(
                "canonical campaign parameters are frozen; use --development-mode "
                "for smaller diagnostics"
            )
        if arguments.round < 2 and arguments.final_test_reveal:
            parser.error("intermediate canonical rounds cannot reveal test metrics")
        if arguments.round == 2 and not arguments.final_test_reveal:
            parser.error("canonical round 2 requires --final-test-reveal")
        if arguments.round > 0 and arguments.previous_workflow is None:
            parser.error("canonical rounds one and two require --previous-workflow")

    try:
        repository_record = repository_identity(arguments.repository)
        release_build_record = release_build_identity(
            teacher, continuation_generator
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    if campaign_eligible and (
        repository_record["clean"] is not True
        or repository_record["branch"] is None
        or release_build_record["cmake_build_type"] != "Release"
        or release_build_record["cmake_cache"] is None
        or "continuation_generator" not in release_build_record["binaries"]
    ):
        parser.error(
            "canonical campaign requires a clean named Git branch and both "
            "producer binaries from one CMake Release build"
        )

    prior_chain = None
    if campaign_eligible and arguments.round > 0:
        try:
            prior_chain = validate_canonical_predecessor_inputs(
                round_index=arguments.round,
                previous_workflow=arguments.previous_workflow,
                previous_roots=arguments.previous_roots,
                continuation_model=arguments.continuation_model,
                prior_pack_reports=arguments.prior_pack_report,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif arguments.smoke_profile and arguments.round > 0:
        try:
            prior_chain = validate_smoke_predecessor_inputs(
                round_index=arguments.round,
                previous_workflow=arguments.previous_workflow,
                previous_roots=arguments.previous_roots,
                continuation_model=arguments.continuation_model,
                prior_pack_reports=arguments.prior_pack_report,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))

    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = output / "replay-roots.json"
    teacher_tsv = output / "teacher-input.tsv"
    teacher_jsonl = output / "teacher-labels.jsonl"
    root_labels = output / "teacher-root-labels.jsonl"
    continuations_tsv = output / f"continuations-round-{arguments.round}.tsv"
    continuations_manifest = output / (
        f"continuations-round-{arguments.round}.manifest.json"
    )
    continuation_labels = output / "teacher-continuation-labels.jsonl"
    shards = output / "shards"
    seed_checkpoints = output / "training-seeds"
    unbound_model = output / "model-unbound"
    model = output / "model"
    python = sys.executable
    environment = environment_identity()
    manager = StageManager(
        output=output,
        campaign_id=arguments.campaign_id,
        round_index=arguments.round,
        resume=arguments.resume,
        environment=environment,
    )
    workflow_source = pathlib.Path(__file__).resolve()
    corpus_source = HERE / "jacek_replay_corpus.py"
    pack_source = HERE / "jacek_replay_pack.py"
    trainer_source = HERE / "jacek_replay_train.py"
    feature_source = HERE / "jacek_replay_features.py"

    corpus_command = [
        python,
        str(corpus_source),
        "--repository",
        str(arguments.repository.resolve()),
        "--exclusions",
        str(arguments.exclusions.resolve()),
        "--public-jacek",
        str(arguments.public_jacek.resolve()),
        "--live-snapshot",
        str(arguments.live_snapshot.resolve()),
        "--output",
        str(roots),
    ]
    corpus_inputs = {
        "exclusions": arguments.exclusions.resolve(),
        "public_jacek": arguments.public_jacek.resolve(),
        "live_snapshot": arguments.live_snapshot.resolve(),
    }
    if arguments.previous_roots is not None:
        corpus_command.extend(
            ["--previous-roots", str(arguments.previous_roots.resolve())]
        )
        corpus_inputs["previous_roots"] = arguments.previous_roots.resolve()
    def run_corpus() -> dict:
        report = json.loads(
            run_with_atomic_output(corpus_command, output_flag="--output", output=roots)
        )
        report["output"] = str(roots)
        return {"tool_report": report}

    manager.execute(
        ordinal=0,
        name="roots",
        configuration={"command": corpus_command[2:]},
        producers={
            "workflow": workflow_source,
            "corpus": corpus_source,
            "features": feature_source,
        },
        inputs=corpus_inputs,
        outputs={"roots": roots},
        action=run_corpus,
    )

    teacher_tsv_command = [
        python,
        str(pack_source),
        "teacher-tsv",
        "--roots",
        str(roots),
        "--output",
        str(teacher_tsv),
    ]
    def run_teacher_tsv() -> dict:
        report = json.loads(
            run_with_atomic_output(
                teacher_tsv_command, output_flag="--output", output=teacher_tsv
            )
        )
        report["output"] = str(teacher_tsv)
        return {"tool_report": report}

    manager.execute(
        ordinal=1,
        name="teacher-tsv",
        configuration={"format": "group_id-source-winner-transcript-v1"},
        producers={
            "workflow": workflow_source,
            "pack": pack_source,
            "corpus": corpus_source,
            "features": feature_source,
        },
        inputs={"roots": roots},
        outputs={"teacher_tsv": teacher_tsv},
        action=run_teacher_tsv,
    )

    teacher_outputs: list[pathlib.Path] = []
    if arguments.round == 0:
        root_teacher_arguments = [
            "--nodes",
            str(arguments.root_nodes),
            "--max-samples",
            str(arguments.max_samples),
        ]
        manager.execute(
            ordinal=2,
            name="root-labels",
            configuration={
                "teacher_arguments": root_teacher_arguments,
                "teacher_workers": arguments.teacher_workers,
                "teacher_chunk_games": arguments.teacher_chunk_games,
            },
            producers={"workflow": workflow_source, "teacher": teacher},
            inputs={"teacher_tsv": teacher_tsv},
            outputs={"root_labels": root_labels},
            action=lambda: run_teacher_chunks(
                manager=manager,
                stage_ordinal=2,
                stage_name="root-labels",
                teacher=teacher,
                input_path=teacher_tsv,
                output_path=root_labels,
                teacher_arguments=root_teacher_arguments,
                workers=arguments.teacher_workers,
                chunk_games=arguments.teacher_chunk_games,
            ),
            validator=validate_teacher_chunks_result,
        )
        teacher_outputs.append(root_labels)

    continuation_manifest_payload = None
    if arguments.continuation_games > 0:
        continuation_seed = 0x4A5242464D5631 + arguments.round * 1_000_000_000
        continuation_command = [
            str(continuation_generator),
            "--input",
            str(teacher_tsv),
            "--output",
            str(continuations_tsv),
            "--manifest",
            str(continuations_manifest),
            "--games",
            str(arguments.continuation_games),
            "--round",
            str(arguments.round),
            "--seed",
            str(continuation_seed),
            "--actor-nodes",
            str(arguments.actor_nodes),
            "--candidate-tree-nodes",
            str(arguments.candidate_tree_nodes),
        ]
        continuation_inputs = {"teacher_tsv": teacher_tsv}
        if arguments.continuation_model:
            continuation_command.extend(
                ["--model", str(arguments.continuation_model.resolve())]
            )
            continuation_inputs["model"] = arguments.continuation_model.resolve()

        def run_continuations() -> dict:
            # The generator itself publishes both outputs atomically. Its
            # sidecar binds the TSV, successful quotas, and every row lineage.
            run(continuation_command)
            payload = validate_continuation_manifest(
                continuations_manifest,
                continuations_tsv,
                round_index=arguments.round,
                games=arguments.continuation_games,
                input_path=teacher_tsv,
                model_path=(
                    arguments.continuation_model.resolve()
                    if arguments.continuation_model is not None
                    else None
                ),
            )
            return {
                "successful_games": payload["successful_games"],
                "successful_quotas": payload["successful_quotas"],
                "manifest_sha256": sha256(continuations_manifest),
            }

        def validate_continuations(result: dict) -> None:
            payload = validate_continuation_manifest(
                continuations_manifest,
                continuations_tsv,
                round_index=arguments.round,
                games=arguments.continuation_games,
                input_path=teacher_tsv,
                model_path=(
                    arguments.continuation_model.resolve()
                    if arguments.continuation_model is not None
                    else None
                ),
            )
            if (
                result.get("successful_games") != payload["successful_games"]
                or result.get("successful_quotas") != payload["successful_quotas"]
                or result.get("manifest_sha256") != sha256(continuations_manifest)
            ):
                raise ValueError("continuation stage result is stale")

        manager.execute(
            ordinal=3,
            name="continuations",
            configuration={
                "games": arguments.continuation_games,
                "round": arguments.round,
                "seed": continuation_seed,
                "actor_nodes": arguments.actor_nodes,
                "candidate_tree_nodes": arguments.candidate_tree_nodes,
            },
            producers={
                "workflow": workflow_source,
                "continuation_generator": continuation_generator,
            },
            inputs=continuation_inputs,
            outputs={
                "continuations_tsv": continuations_tsv,
                "continuations_manifest": continuations_manifest,
            },
            action=run_continuations,
            validator=validate_continuations,
        )
        continuation_manifest_payload = validate_continuation_manifest(
            continuations_manifest,
            continuations_tsv,
            round_index=arguments.round,
            games=arguments.continuation_games,
            input_path=teacher_tsv,
            model_path=(
                arguments.continuation_model.resolve()
                if arguments.continuation_model is not None
                else None
            ),
        )
        continuation_teacher_arguments = [
            "--nodes",
            str(arguments.nodes),
            "--max-samples",
            str(arguments.max_samples),
            "--deep-nodes",
            str(arguments.root_nodes),
            "--deep-percent",
            str(arguments.deep_percent),
        ]
        manager.execute(
            ordinal=4,
            name="continuation-labels",
            configuration={
                "teacher_arguments": continuation_teacher_arguments,
                "teacher_workers": arguments.teacher_workers,
                "teacher_chunk_games": arguments.teacher_chunk_games,
            },
            producers={"workflow": workflow_source, "teacher": teacher},
            inputs={
                "continuations_tsv": continuations_tsv,
                "continuations_manifest": continuations_manifest,
            },
            outputs={"continuation_labels": continuation_labels},
            action=lambda: run_teacher_chunks(
                manager=manager,
                stage_ordinal=4,
                stage_name="continuation-labels",
                teacher=teacher,
                input_path=continuations_tsv,
                output_path=continuation_labels,
                teacher_arguments=continuation_teacher_arguments,
                workers=arguments.teacher_workers,
                chunk_games=arguments.teacher_chunk_games,
            ),
            validator=validate_teacher_chunks_result,
        )
        teacher_outputs.append(continuation_labels)

    if not teacher_outputs:
        raise ValueError("workflow has no teacher label source")

    def concatenate_labels() -> dict:
        concatenate(teacher_outputs, teacher_jsonl)
        return {"teacher_rows": validate_teacher_output(teacher_jsonl)}

    concatenation_result = manager.execute(
        ordinal=5,
        name="concatenation",
        configuration={"order": [path.name for path in teacher_outputs]},
        producers={"workflow": workflow_source},
        inputs={path.stem: path for path in teacher_outputs},
        outputs={"teacher_jsonl": teacher_jsonl},
        action=concatenate_labels,
        validator=lambda result: (
            None
            if result.get("teacher_rows") == validate_teacher_output(teacher_jsonl)
            else (_ for _ in ()).throw(ValueError("teacher row count is stale"))
        ),
    )
    teacher_rows = concatenation_result["teacher_rows"]

    prior_pack_hashes = []
    prior_pack_payloads = []
    for prior_path in arguments.prior_pack_report:
        prior = _load_json(prior_path, "prior pack report")
        if prior.get("schema") != "papersoccer.jacek-replay-pack-report.v1":
            raise ValueError(f"invalid prior pack report: {prior_path}")
        prior_pack_hashes.append(sha256(prior_path))
        prior_pack_payloads.append(prior)
    if len(set(prior_pack_hashes)) != len(prior_pack_hashes):
        raise ValueError("the same prior pack report was supplied more than once")

    pack_command = [
        python,
        str(pack_source),
        "pack",
        "--roots",
        str(roots),
        "--teacher",
        str(teacher_jsonl),
        "--output-directory",
        str(shards),
        "--streaming",
    ]
    for prior in prior_pack_payloads:
        for split in ("train", "validation", "test"):
            pack_command.extend(
                ["--prior-shard-manifest", prior["shards"][split]["manifest"]]
            )

    def run_pack() -> dict:
        report = json.loads(run(pack_command))
        report_path = shards / "pack-report.json"
        persisted = _load_json(report_path, "pack report")
        if {key: value for key, value in report.items() if key != "report"} != persisted:
            raise ValueError("pack stdout differs from pack-report.json")
        return {"pack_report": report}

    pack_inputs = {"roots": roots, "teacher_jsonl": teacher_jsonl}
    pack_inputs.update(
        {f"prior_pack_{index}": path.resolve() for index, path in enumerate(arguments.prior_pack_report)}
    )
    for prior_index, prior in enumerate(prior_pack_payloads):
        for split in ("train", "validation", "test"):
            manifest_path = pathlib.Path(prior["shards"][split]["manifest"])
            manifest_payload = _load_json(manifest_path, "prior shard manifest")
            pack_inputs[f"prior_{prior_index}_{split}_manifest"] = manifest_path
            pack_inputs[f"prior_{prior_index}_{split}_npz"] = (
                manifest_path.parent / manifest_payload["npz"]
            )
    pack_result = manager.execute(
        ordinal=6,
        name="packing",
        configuration={"streaming": True, "prior_pack_sha256": prior_pack_hashes},
        producers={
            "workflow": workflow_source,
            "pack": pack_source,
            "corpus": corpus_source,
            "trainer": trainer_source,
            "features": feature_source,
        },
        inputs=pack_inputs,
        outputs={"shards": shards},
        action=run_pack,
        validator=lambda result: (
            None
            if {
                key: value
                for key, value in result.get("pack_report", {}).items()
                if key != "report"
            }
            == _load_json(shards / "pack-report.json", "pack report")
            else (_ for _ in ()).throw(ValueError("packing stage report is stale"))
        ),
    )
    pack_report = pack_result["pack_report"]
    # Cumulative training is chronological.  The matched small-model control
    # receives pack reports as R0, R1, R2, so the large candidate must consume
    # the identical ordered shard stream rather than prepending the newest
    # round and silently changing fixed-seed SGD trajectories.
    shard_manifests = []
    for prior in prior_pack_payloads:
        shard_manifests.extend(
            prior["shards"][split]["manifest"]
            for split in ("train", "validation", "test")
        )
    shard_manifests.extend(
        pack_report["shards"][split]["manifest"]
        for split in ("train", "validation", "test")
    )

    train_command = [
        python,
        str(trainer_source),
        *shard_manifests,
        "--output-directory",
        str(unbound_model),
        "--seeds",
        arguments.seeds,
        "--seed-workers",
        str(arguments.seed_workers),
        "--seed-checkpoint-directory",
        str(seed_checkpoints),
        "--epochs",
        str(arguments.epochs),
        "--patience",
        str(arguments.patience),
        "--batch-size",
        str(arguments.batch_size),
    ]
    if arguments.final_test_reveal:
        train_command.append("--reveal-test")
    if arguments.resume:
        train_command.append("--resume-seeds")

    def run_training() -> dict:
        return {
            "train_report": json.loads(
                run_with_atomic_directory(
                    train_command,
                    output_flag="--output-directory",
                    output=unbound_model,
                )
            )
        }

    training_inputs = {}
    for index, raw_path in enumerate(shard_manifests):
        manifest_path = pathlib.Path(raw_path)
        manifest_payload = _load_json(manifest_path, "training shard manifest")
        training_inputs[f"shard_manifest_{index}"] = manifest_path
        training_inputs[f"shard_npz_{index}"] = (
            manifest_path.parent / manifest_payload["npz"]
        )
    training_result = manager.execute(
        ordinal=7,
        name="training",
        configuration={
            "seeds": arguments.seeds,
            "seed_workers": arguments.seed_workers,
            "epochs": arguments.epochs,
            "patience": arguments.patience,
            "batch_size": arguments.batch_size,
            "reveal_test": arguments.final_test_reveal,
        },
        producers={
            "workflow": workflow_source,
            "trainer": trainer_source,
            "corpus": corpus_source,
            "features": feature_source,
        },
        inputs=training_inputs,
        outputs={
            "unbound_model": unbound_model,
            "seed_checkpoints": seed_checkpoints,
        },
        action=run_training,
        resumable_outputs={"seed_checkpoints"},
    )
    raw_train_report = training_result["train_report"]
    canonical_ancestors = prior_chain["entries"] if prior_chain is not None else []
    workflow_ancestors = (
        prior_chain["entries"] if prior_chain is not None else []
    )

    def bind_selected_model() -> dict:
        shutil.copytree(unbound_model, model)
        raw_manifest = pathlib.Path(raw_train_report["manifest"])
        model_manifest_path = model / raw_manifest.name
        model_manifest = _load_json(model_manifest_path, "model manifest")
        model_manifest["status"] = (
            "canonical-campaign-candidate-not-game-gated"
            if campaign_eligible
            else "development-candidate-not-promotable"
        )
        model_manifest["campaign_contract"] = {
            "eligible": campaign_eligible,
            "profile": SMOKE_PROFILE if arguments.smoke_profile else None,
            "campaign_id": arguments.campaign_id,
            "round": arguments.round,
            "continuation_games": arguments.continuation_games,
            "bulk_nodes": arguments.nodes,
            "root_and_deep_nodes": arguments.root_nodes,
            "deep_percent": arguments.deep_percent,
            "max_samples_per_game": arguments.max_samples,
            "actor_nodes": arguments.actor_nodes,
            "candidate_tree_nodes": arguments.candidate_tree_nodes,
            "teacher_workers": arguments.teacher_workers,
            "teacher_chunk_games": arguments.teacher_chunk_games,
            "seed_workers": arguments.seed_workers,
            "prior_rounds": len(arguments.prior_pack_report),
            "test_revealed": arguments.final_test_reveal,
            "canonical_ancestry": canonical_ancestors if campaign_eligible else [],
            "previous_workflow_sha256": (
                prior_chain["entry"]["workflow_sha256"]
                if prior_chain is not None
                else None
            ),
        }
        atomic_write(model_manifest_path, canonical_json_bytes(model_manifest))
        runtime_path = (model / model_manifest["runtime"]["path"]).resolve()
        train_report = dict(raw_train_report)
        train_report["manifest"] = str(model_manifest_path.resolve())
        return {
            "training": train_report,
            "model": {
                "manifest_path": str(model_manifest_path.resolve()),
                "manifest_sha256": sha256(model_manifest_path),
                "runtime_path": str(runtime_path),
                "runtime_sha256": sha256(runtime_path),
            },
        }

    selected_result = manager.execute(
        ordinal=8,
        name="selected-runtime",
        configuration={
            "campaign_eligible": campaign_eligible,
            "campaign_id": arguments.campaign_id,
            "round": arguments.round,
            "ancestry": canonical_ancestors if campaign_eligible else [],
        },
        producers={"workflow": workflow_source},
        inputs={"unbound_model": unbound_model},
        outputs={"model": model},
        action=bind_selected_model,
    )
    train_report = selected_result["training"]
    model_artifact = selected_result["model"]
    completed_stage_receipts = stage_receipt_bindings(
        manager, include_root_labels=arguments.round == 0
    )
    workflow = {
        "schema": WORKFLOW_SCHEMA,
        "producer": {
            "workflow": sha256(pathlib.Path(__file__)),
            "corpus": sha256(HERE / "jacek_replay_corpus.py"),
            "pack": sha256(HERE / "jacek_replay_pack.py"),
            "trainer": sha256(HERE / "jacek_replay_train.py"),
        },
        "configuration": {
            "nodes": arguments.nodes,
            "root_nodes": arguments.root_nodes,
            "deep_percent": arguments.deep_percent,
            "max_samples_per_game": arguments.max_samples,
            "epochs": arguments.epochs,
            "patience": arguments.patience,
            "batch_size": arguments.batch_size,
            "seeds": arguments.seeds,
            "profile": SMOKE_PROFILE if arguments.smoke_profile else None,
            "campaign_id": arguments.campaign_id,
            "teacher_workers": arguments.teacher_workers,
            "teacher_chunk_games": arguments.teacher_chunk_games,
            "seed_workers": arguments.seed_workers,
            "prior_pack_report_sha256": prior_pack_hashes,
            "final_test_revealed": arguments.final_test_reveal,
            "campaign_eligible": campaign_eligible,
            "round": arguments.round,
            "continuation_games": arguments.continuation_games,
            "actor_nodes": arguments.actor_nodes,
            "candidate_tree_nodes": arguments.candidate_tree_nodes,
        },
        "inputs": {
            "exclusions_sha256": sha256(arguments.exclusions),
            "public_jacek_sha256": sha256(arguments.public_jacek),
            "live_snapshot_sha256": sha256(arguments.live_snapshot),
            "previous_roots_sha256": (
                sha256(arguments.previous_roots)
                if arguments.previous_roots is not None
                else None
            ),
            "teacher_executable_sha256": sha256(teacher),
            "continuation_generator_sha256": (
                sha256(continuation_generator)
                if continuation_generator is not None
                else None
            ),
            "continuation_model_sha256": (
                sha256(arguments.continuation_model)
                if arguments.continuation_model is not None
                else None
            ),
        },
        "artifacts": {
            "roots": {"path": str(roots), "sha256": sha256(roots)},
            "teacher_tsv": {
                "path": str(teacher_tsv),
                "sha256": sha256(teacher_tsv),
            },
            "teacher_jsonl": {
                "path": str(teacher_jsonl),
                "sha256": sha256(teacher_jsonl),
                "rows": teacher_rows,
            },
            "continuations": (
                {
                    "path": str(continuations_tsv),
                    "sha256": sha256(continuations_tsv),
                    "manifest_path": str(continuations_manifest),
                    "manifest_sha256": sha256(continuations_manifest),
                    "successful_games": continuation_manifest_payload[
                        "successful_games"
                    ],
                    "successful_quotas": continuation_manifest_payload[
                        "successful_quotas"
                    ],
                }
                if arguments.continuation_games > 0
                else None
            ),
            "pack_report": pack_report,
            "training": train_report,
            "model": model_artifact,
        },
        "lineage": {
            "previous_workflow": (
                {
                    "path": str(arguments.previous_workflow.resolve()),
                    "sha256": prior_chain["entry"]["workflow_sha256"],
                }
                if prior_chain is not None
                else None
            ),
            "ancestors": workflow_ancestors,
        },
        "execution": {
            "campaign_id": arguments.campaign_id,
            "resumable_stage_receipt_schema": STAGE_RECEIPT_SCHEMA,
            "environment": environment,
            "repository_path": str(arguments.repository.resolve()),
            "repository": repository_record,
            "release_build": release_build_record,
            "feature_encoder": artifact_snapshot(feature_source),
            "candidate_search_source_closure_sha256": source_closure_sha256(
                arguments.repository
            ),
            "stage_receipts": completed_stage_receipts,
        },
    }
    workflow_path = output / "workflow.json"
    final_workflow_inputs = {
        "roots": roots,
        "teacher_tsv": teacher_tsv,
        "teacher_jsonl": teacher_jsonl,
        "pack_report": shards / "pack-report.json",
        "model_manifest": pathlib.Path(model_artifact["manifest_path"]),
        "runtime": pathlib.Path(model_artifact["runtime_path"]),
    }
    final_workflow_inputs.update(
        {
            f"stage_receipt_{binding['ordinal']}": pathlib.Path(binding["path"])
            for binding in completed_stage_receipts
        }
    )
    manager.execute(
        ordinal=9,
        name="workflow",
        configuration={
            "campaign_id": arguments.campaign_id,
            "round": arguments.round,
            "campaign_eligible": campaign_eligible,
        },
        producers={
            "workflow": workflow_source,
            "corpus": corpus_source,
            "pack": pack_source,
            "trainer": trainer_source,
            "features": feature_source,
        },
        inputs=final_workflow_inputs,
        outputs={"workflow": workflow_path},
        action=lambda: (
            atomic_write(workflow_path, canonical_json_bytes(workflow, pretty=True))
            or {"workflow_sha256": sha256(workflow_path)}
        ),
        validator=lambda result: (
            None
            if result.get("workflow_sha256") == sha256(workflow_path)
            else (_ for _ in ()).throw(ValueError("workflow stage hash is stale"))
        ),
    )
    if campaign_eligible:
        validate_canonical_workflow_chain(workflow_path, arguments.round)
    elif arguments.smoke_profile:
        validate_smoke_workflow_chain(workflow_path, arguments.round)
    print(json.dumps({"workflow": str(workflow_path), **train_report}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"jacek replay workflow: {error}", file=sys.stderr)
        raise SystemExit(2)
