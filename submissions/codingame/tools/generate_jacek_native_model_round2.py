#!/usr/bin/env python3
"""Export an identified Jacek-native round-two seed checkpoint.

Round one remains frozen and source-bound to its original trainer and corpus
validator.  This additive exporter applies the same compact runtime/header
format after verifying the round-two trainer, corpus contract, cumulative
lineage, and every retained checkpoint identity.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import pathlib
import sys
from typing import Mapping


TOOL_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOL_DIRECTORY))
import generate_jacek_native_model as round1_exporter  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUND2_TRAINER = ROOT / "tools" / "train_jacek_native_round2.py"
ROUND2_CORPUS_VALIDATOR = ROOT / "tools" / "jacek_native_corpus_round2.py"
RESTART_CORPUS_VALIDATOR = (
    ROOT / "tools" / "jacek_native_restart_corpus_round2.py"
)
DEFAULT_MODEL = ROOT / "models" / "jacek_native_round2_candidate.json"
DEFAULT_HEADER = (
    ROOT / "submissions" / "codingame" / "bots" / "jacek_native_bfm" /
    "jacek_native_model.hpp"
)

MODEL_SCHEMA = round1_exporter.MODEL_SCHEMA
FEATURE_SCHEMA = round1_exporter.FEATURE_SCHEMA
RULES = round1_exporter.RULES


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ) + "\n").encode()


def _validate_lineage(provenance: Mapping[str, object]) -> None:
    lineage = provenance.get("lineage")
    if not isinstance(lineage, dict) or set(lineage) != {
        "strict_current", "archived_round1", "live_restart_round2"
    }:
        raise ValueError("round-two cumulative corpus lineage is malformed")
    if not isinstance(lineage["strict_current"], list) or not lineage[
            "strict_current"]:
        raise ValueError("round-two lineage has no strict-current run")
    for category in ("strict_current", "archived_round1"):
        entries = lineage[category]
        if not isinstance(entries, list):
            raise ValueError("round-two lineage category is not a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "manifest_sha256", "build_provenance_sha256",
                "binary_sha256", "shard_sha256", "games", "seed",
            }:
                raise ValueError("round-two lineage entry is not frozen")
            if not all(_valid_sha256(entry.get(field)) for field in (
                "manifest_sha256", "build_provenance_sha256", "binary_sha256"
            )):
                raise ValueError("round-two lineage identity is invalid")
            shards = entry.get("shard_sha256")
            if (
                not isinstance(shards, list)
                or not shards
                or shards != sorted(shards)
                or not all(_valid_sha256(value) for value in shards)
            ):
                raise ValueError("round-two lineage shard identities are invalid")
            games = entry.get("games")
            seed = entry.get("seed")
            if (
                isinstance(games, bool) or not isinstance(games, int)
                or games <= 0 or isinstance(seed, bool)
                or not isinstance(seed, int) or not 0 <= seed < 1 << 64
            ):
                raise ValueError("round-two lineage counts are invalid")
    restarts = lineage["live_restart_round2"]
    if not isinstance(restarts, list):
        raise ValueError("round-two restart lineage is not a list")
    for entry in restarts:
        if not isinstance(entry, dict) or set(entry) != {
            "manifest_sha256", "build_provenance_sha256", "binary_sha256",
            "collector_tsv_sha256", "arena_manifest_sha256",
            "asserted_source_sha256", "exclusion_registry_sha256",
            "source_binding_status", "games", "selected_prefixes",
        }:
            raise ValueError("round-two restart lineage entry is not frozen")
        if not all(_valid_sha256(entry.get(field)) for field in (
            "manifest_sha256", "build_provenance_sha256", "binary_sha256",
            "collector_tsv_sha256", "arena_manifest_sha256",
            "asserted_source_sha256", "exclusion_registry_sha256",
        )):
            raise ValueError("round-two restart lineage identity is invalid")
        if entry.get("source_binding_status") not in {
            "asserted-not-api-verified", "api-verified"
        } or any(
            isinstance(entry.get(field), bool)
            or not isinstance(entry.get(field), int)
            or entry[field] <= 0
            for field in ("games", "selected_prefixes")
        ):
            raise ValueError("round-two restart lineage values are invalid")


def _validate_checkpoint_provenance(provenance: Mapping[str, object]) -> None:
    generation = provenance.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("round-two generation provenance is missing")
    checkpoint = generation.get("checkpoint_provenance")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "mode", "artifacts"
    }:
        raise ValueError("round-two checkpoint provenance is malformed")
    if checkpoint.get("mode") not in {
        "untrained-seed-bootstrap/v1", "native-runtime-models/v1",
        "cumulative-native-runtime-models/v2",
    }:
        raise ValueError("round-two checkpoint provenance mode is invalid")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("round-two checkpoint provenance is empty")
    normalized = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "artifact_sha256", "model_sha256", "packed_sha256"
        } or not all(_valid_sha256(artifact.get(field)) for field in artifact):
            raise ValueError("round-two checkpoint identity is invalid")
        normalized.append(dict(artifact))
    key = lambda value: (
        value["artifact_sha256"], value["model_sha256"],
        value["packed_sha256"],
    )
    if normalized != sorted(normalized, key=key) or len({
            tuple(sorted(value.items())) for value in normalized}) != len(normalized):
        raise ValueError("round-two checkpoint identities are not canonical")
    model_artifacts = generation.get("model_artifact_sha256")
    expected_artifacts = sorted(value["artifact_sha256"] for value in normalized)
    if model_artifacts != expected_artifacts:
        raise ValueError("round-two model-artifact summary is incomplete")


def _validate_retained_checkpoints(model: Mapping[str, object]) -> None:
    training = model.get("training")
    checkpoints = model.get("checkpoints")
    if not isinstance(training, dict) or not isinstance(checkpoints, list) or (
            not checkpoints):
        raise ValueError("round-two retained checkpoints are missing")
    seeds = training.get("seeds")
    if (
        not isinstance(seeds, list) or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int)
               or not 0 <= seed < 1 << 64 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("round-two training seeds are invalid")
    observed = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "seed", "model", "quantization", "checkpoint_sha256"
        }:
            raise ValueError("round-two retained checkpoint fields are not frozen")
        seed = checkpoint["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise ValueError("round-two checkpoint seed is invalid")
        payload = {
            "seed": seed,
            "model": checkpoint["model"],
            "quantization": checkpoint["quantization"],
        }
        if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != checkpoint[
                "checkpoint_sha256"]:
            raise ValueError("round-two checkpoint SHA-256 is stale")
        # Exercise the frozen tensor/range checks for every retained seed, not
        # only for the seed requested by this invocation.
        structural = dict(model)
        structural["model"] = checkpoint["model"]
        structural["quantization"] = checkpoint["quantization"]
        compatible = copy.deepcopy(structural)
        compatible["provenance"]["trainer_sha256"] = hashlib.sha256(
            round1_exporter.TRAINER.read_bytes()).hexdigest()
        compatible["provenance"]["corpus_validator_sha256"] = hashlib.sha256(
            round1_exporter.CORPUS_VALIDATOR.read_bytes()).hexdigest()
        round1_exporter.validate_model(compatible)
        observed.append(seed)
    if observed != seeds:
        raise ValueError("round-two checkpoint order does not match training seeds")
    chosen = training.get("chosen_seed")
    provisional = training.get("provisional_seed")
    if chosen is not None or provisional not in observed:
        raise ValueError(
            "round-two training artifact must remain pending and unselected"
        )
    external = training.get("external_actual_clock_selection")
    if (
        not isinstance(external, dict)
        or external.get("required") is not True
        or external.get("criterion") != "native-actual-clock-match-strength"
        or external.get("status") != "pending"
        or sorted(external.get("eligible_seed_order", ())) != sorted(observed)
    ):
        raise ValueError("round-two actual-clock selection contract is invalid")


def validate_model(model: Mapping[str, object]) -> tuple[list[int], dict[str, float]]:
    provenance = model.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("round-two model has no training provenance")
    trainer_sha = hashlib.sha256(ROUND2_TRAINER.read_bytes()).hexdigest()
    corpus_sha = hashlib.sha256(ROUND2_CORPUS_VALIDATOR.read_bytes()).hexdigest()
    if provenance.get("trainer_sha256") != trainer_sha:
        raise ValueError("round-two model trainer SHA-256 is stale")
    if provenance.get("corpus_validator_sha256") != corpus_sha:
        raise ValueError("round-two corpus-validator SHA-256 is stale")
    restart_corpus_sha = hashlib.sha256(
        RESTART_CORPUS_VALIDATOR.read_bytes()
    ).hexdigest()
    if provenance.get("restart_corpus_validator_sha256") != restart_corpus_sha:
        raise ValueError("round-two restart corpus-validator SHA-256 is stale")
    sources = provenance.get("source_sha256")
    if (
        not isinstance(sources, dict) or not sources
        or not all(isinstance(key, str) and _valid_sha256(value)
                   and key == f"sha256:{value}"
                   for key, value in sources.items())
    ):
        raise ValueError("round-two corpus source identities are incomplete")
    expected_corpus_sha = hashlib.sha256(json.dumps(
        sorted(sources.items()), separators=(",", ":")
    ).encode()).hexdigest()
    if provenance.get("corpus_sha256") != expected_corpus_sha:
        raise ValueError("round-two cumulative corpus SHA-256 is stale")
    _validate_lineage(provenance)
    _validate_checkpoint_provenance(provenance)
    _validate_retained_checkpoints(model)

    # The architecture, target, symmetry, quantization, and tensor contracts
    # are intentionally identical to round one.  Substitute only the two
    # already-verified semantic dependency hashes in a private copy so the
    # frozen structural validator can be reused without changing it.
    compatible = copy.deepcopy(model)
    compatible["provenance"]["trainer_sha256"] = hashlib.sha256(
        round1_exporter.TRAINER.read_bytes()).hexdigest()
    compatible["provenance"]["corpus_validator_sha256"] = hashlib.sha256(
        round1_exporter.CORPUS_VALIDATOR.read_bytes()).hexdigest()
    return round1_exporter.validate_model(compatible)


def render(
    model: Mapping[str, object], model_sha256: str, seed: int | None = None
) -> tuple[str, dict]:
    if seed is None:
        raise ValueError(
            "round-two export requires an explicit seed or a verified "
            "selection sidecar"
        )
    selected, selected_seed = round1_exporter.select_checkpoint(model, seed)
    weights, scales = validate_model(selected)
    payload = round1_exporter.pack_signed_three_bit(weights)
    if round1_exporter.unpack_signed_three_bit(payload, len(weights)) != weights:
        raise RuntimeError("internal signed three-bit packing mismatch")
    encoded = base64.b64encode(payload).decode("ascii")
    schema_payload = json.dumps({
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "architecture": selected["architecture"],
        "rules": RULES,
        "packing": selected["quantization"]["packing"],
    }, sort_keys=True, separators=(",", ":")).encode()
    schema_sha256 = hashlib.sha256(schema_payload).hexdigest()
    packed_sha256 = hashlib.sha256(payload).hexdigest()
    header = f"""#pragma once

#include <cstddef>
#include <string_view>

namespace papersoccer::jacek_native_model {{

inline constexpr std::size_t kInputs = 1156;
inline constexpr std::size_t kHiddenOne = 32;
inline constexpr std::size_t kHiddenTwo = 32;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = {len(weights)};
inline constexpr std::size_t kPackedByteCount = {len(payload)};
inline constexpr int kQuantizationBits = 3;
inline constexpr unsigned long long kTrainingSeed = {selected_seed}ULL;
inline constexpr bool kBootstrapSeed = false;
inline constexpr float kScale1 = {round1_exporter._float_literal(scales['w1'])};
inline constexpr float kScale2 = {round1_exporter._float_literal(scales['w2'])};
inline constexpr float kScale3 = {round1_exporter._float_literal(scales['w3'])};
inline constexpr std::string_view kModelSchema = "{MODEL_SCHEMA}";
inline constexpr std::string_view kFeatureSchema = "{FEATURE_SCHEMA}";
inline constexpr std::string_view kModelSha256 = "{model_sha256}";
inline constexpr std::string_view kSchemaSha256 = "{schema_sha256}";
inline constexpr std::string_view kPackedSha256 = "{packed_sha256}";
inline constexpr std::string_view kPackedWeights =
{round1_exporter._quoted_chunks(encoded)};

}}  // namespace papersoccer::jacek_native_model
"""
    return header, {
        "model_sha256": model_sha256,
        "schema_sha256": schema_sha256,
        "packed_sha256": packed_sha256,
        "weight_count": len(weights),
        "packed_bytes": len(payload),
        "base64_characters": len(encoded),
        "header_characters": len(header),
        "training_seed": selected_seed,
    }


def render_runtime(
    model: Mapping[str, object], model_sha256: str, seed: int | None = None
) -> str:
    if seed is None:
        raise ValueError(
            "round-two runtime export requires an explicit seed or a "
            "verified selection sidecar"
        )
    selected, _ = round1_exporter.select_checkpoint(model, seed)
    weights, scales = validate_model(selected)
    payload = round1_exporter.pack_signed_three_bit(weights)
    packed_sha256 = hashlib.sha256(payload).hexdigest()
    return (
        "papersoccer.jacek-native-runtime-model/v1\n"
        f"{MODEL_SCHEMA}\n"
        f"{FEATURE_SCHEMA}\n"
        f"{model_sha256}\n"
        f"{packed_sha256}\n"
        f"{scales['w1']:.9g} {scales['w2']:.9g} {scales['w3']:.9g}\n"
        f"{base64.b64encode(payload).decode('ascii')}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export an identified Jacek-native round-two seed."
    )
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_HEADER)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runtime-output", type=pathlib.Path)
    arguments = parser.parse_args()
    raw = arguments.model.read_bytes()
    model = json.loads(raw)
    model_sha256 = hashlib.sha256(raw).hexdigest()
    header, metadata = render(model, model_sha256, arguments.seed)
    runtime = render_runtime(
        model, model_sha256, arguments.seed
    ) if arguments.runtime_output else None
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != header:
            print(f"{arguments.output} is stale", file=sys.stderr)
            return 1
        if arguments.runtime_output and (
            not arguments.runtime_output.exists()
            or arguments.runtime_output.read_text() != runtime
        ):
            print(f"{arguments.runtime_output} is stale", file=sys.stderr)
            return 1
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(header)
        print(f"wrote {arguments.output}")
        if arguments.runtime_output:
            arguments.runtime_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.runtime_output.write_text(runtime)
            print(f"wrote {arguments.runtime_output}")
    if arguments.metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
