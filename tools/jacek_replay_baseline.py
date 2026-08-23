#!/usr/bin/env python3
"""Train the matched 1156->32->32->1 value-control on replay BFM shards."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import jacek_replay_features as replay_features  # noqa: E402
import jacek_replay_train as training  # noqa: E402
import jacek_replay_workflow as workflow_contract  # noqa: E402


BASELINE_INPUTS = 316 + 105 * 8
BASELINE_ARCHITECTURE = [BASELINE_INPUTS, 32, 32, 1]
BASELINE_WEIGHT_COUNT = BASELINE_INPUTS * 32 + 32 * 32 + 32
BASELINE_FEATURE_SCHEMA = (
    "papersoccer.jacek-replay-bfm-baseline.features.v1:"
    "edge316+vertex105x8:mover-relative:true-turn-distance-only"
)
BASELINE_FEATURE_SCHEMA_HASH = hashlib.sha256(
    BASELINE_FEATURE_SCHEMA.encode()
).digest()
FIXED_EPOCHS = 50
FIXED_PATIENCE = 8
FIXED_BATCH_SIZE = 256
FIXED_LEARNING_RATE = 0.001
FIXED_WEIGHT_DECAY = 1e-5
METRIC_FIELDS = (
    "weighted_huber",
    "sign_accuracy",
    "correlation",
    "mae",
    "prediction_mean",
)


def manifests_from_pack_report(path: pathlib.Path) -> list[pathlib.Path]:
    payload = json.loads(path.read_bytes())
    if payload.get("schema") != "papersoccer.jacek-replay-pack-report.v1":
        raise ValueError("unexpected pack-report schema")
    return [
        pathlib.Path(payload["shards"][split]["manifest"])
        for split in ("train", "validation", "test")
    ]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def producer_identity() -> dict[str, str]:
    return {
        "baseline_sha256": sha256(pathlib.Path(__file__)),
        "trainer_sha256": sha256(pathlib.Path(training.__file__)),
        "features_sha256": sha256(pathlib.Path(replay_features.__file__)),
        "workflow_sha256": sha256(pathlib.Path(workflow_contract.__file__)),
    }


@contextlib.contextmanager
def baseline_runtime_contract():
    """Temporarily configure the shape-parametric trainer for the small model."""

    previous = (
        replay_features.INPUT_COUNT,
        training.HIDDEN_ONE,
        training.HIDDEN_TWO,
        training.WEIGHT_COUNT,
        training.FEATURE_SCHEMA_HASH,
    )
    replay_features.INPUT_COUNT = BASELINE_INPUTS
    training.HIDDEN_ONE = 32
    training.HIDDEN_TWO = 32
    training.WEIGHT_COUNT = BASELINE_WEIGHT_COUNT
    training.FEATURE_SCHEMA_HASH = BASELINE_FEATURE_SCHEMA_HASH
    try:
        yield
    finally:
        (
            replay_features.INPUT_COUNT,
            training.HIDDEN_ONE,
            training.HIDDEN_TWO,
            training.WEIGHT_COUNT,
            training.FEATURE_SCHEMA_HASH,
        ) = previous


def fixed_selection_contract() -> dict:
    return {
        "seeds": list(training.FIXED_SEEDS),
        "optimizer": {
            "name": "adamw",
            "epochs": FIXED_EPOCHS,
            "patience": FIXED_PATIENCE,
            "batch_size": FIXED_BATCH_SIZE,
            "learning_rate": FIXED_LEARNING_RATE,
            "weight_decay": FIXED_WEIGHT_DECAY,
            "gradient_norm_clip": 5.0,
        },
        "loss": {"name": "weighted-huber", "delta": 0.25},
        "augmentation": "reflection rows inherit root game split",
        "test_revealed_after_selection": False,
    }


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            output.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def metrics_match(actual: dict, declared: dict) -> bool:
    if actual.get("samples") != declared.get("samples"):
        return False
    return all(
        np.isclose(
            actual.get(field, np.nan),
            declared.get(field, np.nan),
            rtol=0.0,
            atol=1e-6,
        )
        for field in METRIC_FIELDS
    )


def validate_selection(selection: object, baseline_validation: dict) -> None:
    if not isinstance(selection, dict):
        raise ValueError("matched-baseline training selection is missing")
    contract = fixed_selection_contract()
    for field, expected in contract.items():
        if selection.get(field) != expected:
            raise ValueError(f"matched-baseline selection.{field} is not frozen")
    reports = selection.get("seed_reports")
    chosen = selection.get("chosen_seed")
    if (
        not isinstance(reports, list)
        or len(reports) != 3
        or [item.get("seed") for item in reports if isinstance(item, dict)]
        != list(training.FIXED_SEEDS)
        or chosen not in training.FIXED_SEEDS
        or any("test" in item for item in reports if isinstance(item, dict))
    ):
        raise ValueError("matched-baseline seed selection is invalid")
    selected = next(item for item in reports if item.get("seed") == chosen)
    if not isinstance(selected.get("validation"), dict) or not metrics_match(
        selected["validation"], baseline_validation
    ):
        raise ValueError("matched-baseline selected validation is stale")


def load_bound_datasets(bindings: dict) -> dict[str, training.Dataset]:
    entries = bindings["canonical_workflow_entries"]
    loaded = []
    for entry in entries:
        report_path = pathlib.Path(entry["pack_report_path"])
        report = json.loads(report_path.read_bytes())
        if (
            report.get("schema") != "papersoccer.jacek-replay-pack-report.v1"
            or sha256(report_path) != entry["pack_report_sha256"]
        ):
            raise ValueError("matched-baseline pack report is stale")
        validation_manifest = pathlib.Path(
            report["shards"]["validation"]["manifest"]
        )
        loaded.append(training.load_csr_shard(validation_manifest))
    training.validate_shard_collection(loaded)
    return {
        "validation": training.combine_shards(loaded)
    }


def validate_receipt(
    receipt: object,
    model_sha256: object,
    *,
    verify_files: bool = False,
    require_advance: bool = True,
) -> None:
    """Validate and, for real gates, recompute both validation panels."""

    if (
        not isinstance(receipt, dict)
        or receipt.get("schema")
        != "papersoccer.jacek-replay-bfm-baseline-gate.v1"
        or receipt.get("candidate_architecture") != [6301, 192, 32, 1]
        or receipt.get("baseline_architecture") != BASELINE_ARCHITECTURE
    ):
        raise ValueError("matched-baseline receipt schema or architecture is invalid")
    candidate = receipt.get("candidate_validation")
    baseline = receipt.get("baseline_validation")
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        raise ValueError("matched-baseline validation metrics are missing")
    for metrics in (candidate, baseline):
        if (
            isinstance(metrics.get("samples"), bool)
            or type(metrics.get("samples")) is not int
            or metrics["samples"] <= 0
            or any(
                isinstance(metrics.get(field), bool)
                or not isinstance(metrics.get(field), (int, float))
                or not np.isfinite(float(metrics[field]))
                for field in METRIC_FIELDS
            )
            or float(metrics["weighted_huber"]) < 0.0
            or not 0.0 <= float(metrics["sign_accuracy"]) <= 1.0
        ):
            raise ValueError("matched-baseline validation metrics are invalid")
    computed_advance = (
        float(candidate["weighted_huber"]) < float(baseline["weighted_huber"])
        and float(candidate["sign_accuracy"])
        >= float(baseline["sign_accuracy"])
    )
    if candidate["samples"] != baseline["samples"]:
        raise ValueError("matched-baseline validation panels differ")
    if receipt.get("advance_to_game_gates") is not computed_advance:
        if receipt.get("advance_to_game_gates") is True and not computed_advance:
            raise ValueError("matched-baseline metric gate did not pass")
        raise ValueError("matched-baseline pass flag differs from its metrics")
    if require_advance and not computed_advance:
        raise ValueError("matched-baseline metric gate did not pass")
    validate_selection(receipt.get("selection"), baseline)

    bindings = receipt.get("bindings")
    artifact = receipt.get("baseline_artifact")
    producer = receipt.get("producer")
    if not isinstance(bindings, dict) or not valid_sha256(model_sha256):
        raise ValueError("matched-baseline bindings are invalid")
    entries = bindings.get("canonical_workflow_entries")
    pack_hashes = bindings.get("pack_report_sha256")
    source_shards = bindings.get("source_shards")
    hash_fields = (
        "workflow_sha256",
        "roots_sha256",
        "pack_report_sha256",
        "model_manifest_sha256",
        "runtime_sha256",
    )
    path_fields = (
        "workflow_path",
        "roots_path",
        "pack_report_path",
        "model_manifest_path",
        "runtime_path",
    )
    if (
        bindings.get("candidate_runtime_sha256") != model_sha256
        or not valid_sha256(bindings.get("candidate_manifest_sha256"))
        or not valid_sha256(bindings.get("workflow_receipt_sha256"))
        or not isinstance(entries, list)
        or len(entries) != 3
        or [entry.get("round") for entry in entries if isinstance(entry, dict)]
        != [0, 1, 2]
        or any(
            not all(valid_sha256(entry.get(field)) for field in hash_fields)
            or not all(
                isinstance(entry.get(field), str)
                and pathlib.Path(entry[field]).is_absolute()
                for field in path_fields
            )
            for entry in entries
        )
        or bindings.get("workflow_receipt_sha256")
        != entries[-1]["workflow_sha256"]
        or bindings.get("candidate_manifest_sha256")
        != entries[-1]["model_manifest_sha256"]
        or bindings.get("candidate_runtime_sha256")
        != entries[-1]["runtime_sha256"]
        or not isinstance(pack_hashes, list)
        or pack_hashes != [entry["pack_report_sha256"] for entry in entries]
        or len(set(pack_hashes)) != 3
        or not isinstance(source_shards, list)
        or not source_shards
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or item[0] not in ("train", "validation", "test")
            or not valid_sha256(item[1])
            for item in source_shards
        )
    ):
        raise ValueError("matched-baseline canonical bindings are incomplete")
    if (
        not isinstance(artifact, dict)
        or artifact.get("architecture") != BASELINE_ARCHITECTURE
        or artifact.get("feature_schema") != BASELINE_FEATURE_SCHEMA
        or artifact.get("feature_schema_sha256")
        != BASELINE_FEATURE_SCHEMA_HASH.hex()
        or artifact.get("diagnostic_only") is not True
        or not isinstance(artifact.get("path"), str)
        or not pathlib.Path(artifact["path"]).is_absolute()
        or not valid_sha256(artifact.get("artifact_sha256"))
        or artifact.get("weight_count") != BASELINE_WEIGHT_COUNT
    ):
        raise ValueError("matched-baseline artifact binding is invalid")
    if (
        not isinstance(producer, dict)
        or set(producer) != set(producer_identity())
        or not all(valid_sha256(value) for value in producer.values())
    ):
        raise ValueError("matched-baseline producer binding is invalid")
    if not verify_files:
        return

    if producer != producer_identity():
        raise ValueError("matched-baseline producer source changed")
    for entry in entries:
        for path_field, hash_field in zip(path_fields, hash_fields):
            if sha256(pathlib.Path(entry[path_field])) != entry[hash_field]:
                raise ValueError(f"matched-baseline {path_field} artifact is stale")
    datasets = load_bound_datasets(bindings)
    expected_shards = []
    for entry in entries:
        pack = json.loads(pathlib.Path(entry["pack_report_path"]).read_bytes())
        expected_shards.extend(ordered_shard_ids(pack))
    if [list(item) for item in expected_shards] != source_shards:
        raise ValueError("matched-baseline source shard binding is stale")
    candidate_parameters, _ = training.load_runtime(
        pathlib.Path(entries[-1]["runtime_path"])
    )
    recomputed_candidate = training.metrics(
        candidate_parameters, datasets["validation"]
    )
    if not metrics_match(recomputed_candidate, candidate):
        raise ValueError("matched-baseline candidate metrics do not recompute")
    artifact_path = pathlib.Path(artifact["path"])
    if sha256(artifact_path) != artifact["artifact_sha256"]:
        raise ValueError("matched-baseline parameter artifact is stale")
    with baseline_runtime_contract():
        baseline_parameters, runtime_report = training.load_runtime(artifact_path)
        recomputed_baseline = training.metrics(
            baseline_parameters,
            baseline_dataset(datasets["validation"]),
        )
    for field in (
        "artifact_sha256",
        "payload_sha256",
        "feature_schema_sha256",
        "bytes",
        "weight_count",
    ):
        if artifact.get(field) != runtime_report.get(field):
            raise ValueError("matched-baseline parameter artifact binding is stale")
    chosen_report = next(
        item
        for item in receipt["selection"]["seed_reports"]
        if item.get("seed") == receipt["selection"]["chosen_seed"]
    )
    if chosen_report.get("checkpoint") != runtime_report:
        raise ValueError("matched-baseline artifact is not the selected seed checkpoint")
    if not metrics_match(recomputed_baseline, baseline):
        raise ValueError("matched-baseline metrics do not recompute")


def expected_shard_ids(pack_report: dict) -> set[tuple[str, str]]:
    return {
        (split, str(pack_report["shards"][split]["sha256"]))
        for split in ("train", "validation", "test")
    }


def ordered_shard_ids(pack_report: dict) -> list[tuple[str, str]]:
    return [
        (split, str(pack_report["shards"][split]["sha256"]))
        for split in ("train", "validation", "test")
    ]


def validate_candidate_binding(
    manifest_path: pathlib.Path,
    pack_report_paths: list[pathlib.Path],
    pack_reports: list[dict],
    workflow_receipt_path: pathlib.Path,
    workflow_validation: dict,
) -> tuple[dict, dict]:
    candidate = json.loads(manifest_path.read_bytes())
    if (
        candidate.get("schema") != "papersoccer.jacek-replay-bfm-model.v1"
        or candidate.get("architecture", {}).get("dimensions")
        != [6301, 192, 32, 1]
    ):
        raise ValueError("candidate manifest has the wrong model contract")
    campaign = candidate.get("campaign_contract")
    if (
        candidate.get("status")
        != "canonical-campaign-candidate-not-game-gated"
        or not isinstance(campaign, dict)
        or campaign.get("eligible") is not True
        or campaign.get("round") != 2
        or campaign.get("continuation_games") != 10_000
        or campaign.get("prior_rounds") != 2
        or campaign.get("test_revealed") is not True
    ):
        raise ValueError("candidate did not complete the canonical three-round campaign")
    actual = [
        (str(shard.get("split")), str(shard.get("npz_sha256")))
        for shard in candidate.get("source_shards", [])
        if isinstance(shard, dict)
    ]
    expected = [
        identity
        for report in pack_reports
        for identity in ordered_shard_ids(report)
    ]
    if actual != expected:
        raise ValueError(
            "candidate and matched baseline do not use the same shards in the same order"
        )
    entries = workflow_validation.get("entries")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("final workflow does not bind three canonical rounds")
    final_entry = entries[-1]
    if (
        pathlib.Path(final_entry["workflow_path"]) != workflow_receipt_path.resolve()
        or final_entry["workflow_sha256"] != sha256(workflow_receipt_path)
        or pathlib.Path(final_entry["model_manifest_path"]) != manifest_path.resolve()
        or final_entry["model_manifest_sha256"] != sha256(manifest_path)
        or campaign.get("canonical_ancestry") != entries[:-1]
        or campaign.get("previous_workflow_sha256")
        != entries[-2]["workflow_sha256"]
    ):
        raise ValueError("candidate does not match the canonical workflow lineage")
    supplied_pack_hashes = [sha256(path) for path in pack_report_paths]
    if (
        [path.resolve() for path in pack_report_paths]
        != [pathlib.Path(entry["pack_report_path"]) for entry in entries]
        or supplied_pack_hashes
        != [entry["pack_report_sha256"] for entry in entries]
    ):
        raise ValueError("baseline pack reports do not match canonical round ancestry")
    runtime = candidate.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("path"), str):
        raise ValueError("candidate runtime binding is missing")
    runtime_path = manifest_path.parent / runtime["path"]
    if sha256(runtime_path) != runtime.get("artifact_sha256"):
        raise ValueError("candidate runtime SHA-256 does not match its manifest")
    return candidate, {
        "candidate_manifest_sha256": sha256(manifest_path),
        "candidate_runtime_sha256": sha256(runtime_path),
        "pack_report_sha256": [sha256(path) for path in pack_report_paths],
        "workflow_receipt_sha256": sha256(workflow_receipt_path),
        "canonical_workflow_entries": entries,
        "source_shards": [list(item) for item in expected],
    }


def collapse(active: np.ndarray) -> np.ndarray:
    result: list[int] = []
    for raw in active:
        feature = int(raw)
        if feature < replay_features.EDGE_COUNT:
            result.append(feature)
            continue
        vertex, category = divmod(
            feature - replay_features.EDGE_COUNT,
            replay_features.VERTEX_CATEGORIES,
        )
        distance = 7 if category == 56 else category // 8
        result.append(replay_features.EDGE_COUNT + vertex * 8 + distance)
    normalized = np.asarray(sorted(set(result)), dtype=np.int32)
    if len(normalized) < 105 or normalized[-1] >= BASELINE_INPUTS:
        raise ValueError("could not collapse a replay feature row")
    return normalized


def baseline_dataset(dataset: training.Dataset) -> training.Dataset:
    indices = dataset.indices.copy()
    joint = indices >= replay_features.EDGE_COUNT
    relative = indices[joint] - replay_features.EDGE_COUNT
    vertices, categories = np.divmod(relative, replay_features.VERTEX_CATEGORIES)
    distances = np.where(categories == 56, 7, categories // 8)
    indices[joint] = replay_features.EDGE_COUNT + vertices * 8 + distances
    return training.Dataset(
        dataset.indptr.copy(),
        indices,
        dataset.targets.copy(),
        dataset.weights.copy(),
        dataset.group_ids.copy(),
    )


def selected_validation(manifest: dict) -> dict:
    report = manifest.get("training")
    if not isinstance(report, dict):
        raise ValueError("candidate manifest has no training report")
    chosen = report.get("chosen_seed")
    matches = [
        item
        for item in report.get("seed_reports", [])
        if isinstance(item, dict) and item.get("seed") == chosen
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("validation"), dict):
        raise ValueError("candidate manifest does not bind selected validation")
    return matches[0]["validation"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-report", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument("--candidate-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--workflow-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--patience", type=int, default=FIXED_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    arguments = parser.parse_args()
    if (
        arguments.epochs,
        arguments.patience,
        arguments.batch_size,
    ) != (FIXED_EPOCHS, FIXED_PATIENCE, FIXED_BATCH_SIZE):
        parser.error(
            "matched-baseline training is frozen at 50 epochs, patience 8, "
            "and batch size 256"
        )
    artifact_path = (
        arguments.artifact
        if arguments.artifact is not None
        else arguments.output.with_suffix(".baseline.runtime")
    ).resolve()
    if artifact_path == arguments.output.resolve():
        parser.error("--artifact and --output must identify different files")
    try:
        pack_reports = [
            json.loads(path.read_bytes()) for path in arguments.pack_report
        ]
        if any(
            report.get("schema") != "papersoccer.jacek-replay-pack-report.v1"
            for report in pack_reports
        ):
            raise ValueError("unexpected pack-report schema")
        pack_hashes = [sha256(path) for path in arguments.pack_report]
        if len(set(pack_hashes)) != len(pack_hashes):
            raise ValueError("the same pack report was supplied more than once")
        workflow_validation = workflow_contract.validate_canonical_workflow_chain(
            arguments.workflow_receipt, 2
        )
        loaded = [
            training.load_csr_shard(path)
            for report_path in arguments.pack_report
            for path in manifests_from_pack_report(report_path)
        ]
        training.validate_shard_collection(loaded)
        full_datasets = {
            split: training.combine_shards(
                [shard for shard in loaded if shard.split == split]
            )
            for split in ("train", "validation", "test")
        }
        del loaded
        candidate, bindings = validate_candidate_binding(
            arguments.candidate_manifest,
            arguments.pack_report,
            pack_reports,
            arguments.workflow_receipt,
            workflow_validation,
        )
        runtime_path = (
            arguments.candidate_manifest.parent / candidate["runtime"]["path"]
        )
        candidate_parameters, _ = training.load_runtime(runtime_path)
        candidate_validation = training.metrics(
            candidate_parameters, full_datasets["validation"]
        )
        declared_validation = selected_validation(candidate)
        for field in (
            "weighted_huber",
            "sign_accuracy",
            "correlation",
            "mae",
            "prediction_mean",
        ):
            if not np.isclose(
                candidate_validation[field],
                declared_validation[field],
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"candidate validation metric {field} is stale or edited"
                )
        if candidate_validation["samples"] != declared_validation["samples"]:
            raise ValueError("candidate validation sample count is stale or edited")
        del candidate_parameters
        datasets = {
            split: baseline_dataset(dataset)
            for split, dataset in full_datasets.items()
        }
        del full_datasets

        # The imported trainer is shape-parametric. Shards and the large
        # candidate are validated before this isolated context switches to the
        # distance-only diagnostic architecture.
        with baseline_runtime_contract():
            selected, report = training.train_three_seeds(
                datasets,
                seeds=training.FIXED_SEEDS,
                epochs=FIXED_EPOCHS,
                patience=FIXED_PATIENCE,
                batch_size=FIXED_BATCH_SIZE,
                learning_rate=FIXED_LEARNING_RATE,
                weight_decay=FIXED_WEIGHT_DECAY,
                reveal_test=False,
            )
            runtime_report = training.export_runtime(artifact_path, selected)
        chosen_seed = report["chosen_seed"]
        baseline_validation = next(
            item["validation"]
            for item in report["seed_reports"]
            if item["seed"] == chosen_seed
        )
        advance = (
            candidate_validation["weighted_huber"]
            < baseline_validation["weighted_huber"]
            and candidate_validation["sign_accuracy"]
            >= baseline_validation["sign_accuracy"]
        )
        output = {
            "schema": "papersoccer.jacek-replay-bfm-baseline-gate.v1",
            "baseline_architecture": [1156, 32, 32, 1],
            "candidate_architecture": [6301, 192, 32, 1],
            "selection": report,
            "candidate_validation": candidate_validation,
            "baseline_validation": baseline_validation,
            "advance_to_game_gates": advance,
            "bindings": bindings,
            "producer": producer_identity(),
            "baseline_artifact": {
                "path": str(artifact_path),
                "architecture": BASELINE_ARCHITECTURE,
                "feature_schema": BASELINE_FEATURE_SCHEMA,
                "diagnostic_only": True,
                **runtime_report,
            },
        }
        del datasets
        validate_receipt(
            output,
            bindings["candidate_runtime_sha256"],
            verify_files=True,
            require_advance=False,
        )
        atomic_json(arguments.output.resolve(), output)
        print(json.dumps({"output": str(arguments.output), "advance": advance}))
        return 0 if advance else 1
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"jacek replay baseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
