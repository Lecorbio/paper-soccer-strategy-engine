#!/usr/bin/env python3
"""Generate the compact checked-in Jacek-native model header."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import pathlib
import sys
from typing import Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "models" / "jacek_native_bootstrap_model.json"
DEFAULT_HEADER = (
    ROOT / "submissions" / "codingame" / "bots" / "jacek_native_bfm" /
    "jacek_native_model.hpp"
)
TRAINER = ROOT / "tools" / "train_jacek_native.py"
CORPUS_VALIDATOR = ROOT / "tools" / "jacek_native_corpus.py"
MODEL_SCHEMA = "jacek_native_model/v1"
FEATURE_SCHEMA = "canonical-edges316-onehot-true-turn-distance105x8-v1"
RULES = {
    "width": 8,
    "height": 10,
    "goal_rule": "own-goals-allowed",
    "blocked_rule": "mover-loses",
}
SHAPES = {"w1": (1156, 32), "w2": (32, 32), "w3": (32,)}
BITS = 3
MINIMUM = -3
MAXIMUM = 3


def tensor_values(model: Mapping, name: str) -> list[int]:
    tensor = model.get("quantization", {}).get("weights", {}).get(name)
    if not isinstance(tensor, dict) or tensor.get("shape") != list(SHAPES[name]):
        raise ValueError(f"quantized {name} has the wrong shape")
    values = tensor.get("values")
    count = math.prod(SHAPES[name])
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"quantized {name} is incomplete")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"quantized {name} must contain integers")
        if not MINIMUM <= value <= MAXIMUM:
            raise ValueError(f"quantized {name} exceeds signed 3-bit range")
        result.append(value)
    return result


def pack_signed_three_bit(values: Sequence[int]) -> bytes:
    """Pack signed three-bit two's-complement values, least-significant bit first."""
    output = bytearray()
    accumulator = 0
    available = 0
    for value in values:
        if not MINIMUM <= int(value) <= MAXIMUM:
            raise ValueError("weight exceeds supported signed 3-bit range")
        accumulator |= (int(value) & 0b111) << available
        available += BITS
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_signed_three_bit(payload: bytes, count: int) -> list[int]:
    expected_bytes = (count * BITS + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"packed payload has {len(payload)} bytes; expected {expected_bytes}"
        )
    result = []
    accumulator = 0
    available = 0
    source = iter(payload)
    for _ in range(count):
        while available < BITS:
            try:
                value = next(source)
            except StopIteration as error:
                raise ValueError("packed payload is truncated") from error
            accumulator |= value << available
            available += 8
        encoded = accumulator & 0b111
        accumulator >>= BITS
        available -= BITS
        result.append(encoded - 8 if encoded & 0b100 else encoded)
    if accumulator != 0 or next(source, None) is not None:
        raise ValueError("packed payload has nonzero padding or trailing bytes")
    return result


def _float_literal(value: object) -> str:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("quantization scales must be finite and positive")
    rendered = f"{number:.9g}"
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "F"


def select_checkpoint(model: Mapping, seed: int | None) -> tuple[Mapping, int]:
    training = model.get("training")
    if not isinstance(training, dict):
        raise ValueError("model has no training selection metadata")
    selected_seed = training.get("chosen_seed") if seed is None else seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
        raise ValueError("selected checkpoint seed is invalid")
    checkpoints = model.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("model does not retain seed checkpoints")
    matching = [checkpoint for checkpoint in checkpoints
                if isinstance(checkpoint, dict) and
                checkpoint.get("seed") == selected_seed]
    if len(matching) != 1:
        raise ValueError(f"checkpoint seed {selected_seed} is unavailable")
    selected = dict(model)
    selected["model"] = matching[0].get("model")
    selected["quantization"] = matching[0].get("quantization")
    return selected, selected_seed


def validate_model(model: Mapping) -> tuple[list[int], dict[str, float]]:
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unexpected Jacek-native model schema")
    if model.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("unexpected Jacek-native feature schema")
    if model.get("rules") != RULES:
        raise ValueError("model does not use the current CodinGame rules")
    expected_architecture = {
        "inputs": 1156,
        "hidden_one": 32,
        "hidden_two": 32,
        "outputs": 1,
        "biases": False,
        "hidden_one_activation": "square-nonnegative-leaky-0.01-negative",
        "hidden_two_activation": "leaky-relu-0.01",
        "output_activation": "tanh",
    }
    if model.get("architecture") != expected_architecture:
        raise ValueError("unexpected Jacek-native architecture")
    target = model.get("target")
    if not isinstance(target, dict) or target.get("primary") != (
        "mover-relative-final-outcome"
    ) or target.get("policy_target") is not None:
        raise ValueError("model is not outcome-only and mover-relative")
    if target.get("auxiliary") != "stable-native-bfm-reanalysis":
        raise ValueError("unexpected auxiliary target")
    if float(target.get("auxiliary_weight", -1.0)) != 0.25:
        raise ValueError("native reanalysis auxiliary must have 25% weight")
    provenance = model.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("model has no training provenance")
    if provenance.get("incumbent_labels") is not False:
        raise ValueError("model provenance permits incumbent labels")
    if provenance.get("protected_data") is not False:
        raise ValueError("model provenance permits protected data")
    if provenance.get("augmentation") != {
        "reflection": True,
        "rotation": "player-two-canonicalization-in-feature-encoder",
        "grouping": "whole-game-before-augmentation",
    }:
        raise ValueError("model lacks the native symmetry augmentation contract")
    trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    if provenance.get("trainer_sha256") != trainer_sha:
        raise ValueError("model was not produced by the current native trainer")
    corpus_validator_sha = hashlib.sha256(CORPUS_VALIDATOR.read_bytes()).hexdigest()
    if provenance.get("corpus_validator_sha256") != corpus_validator_sha:
        raise ValueError("model was not validated by the current corpus contract")
    quantization = model.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("model has no quantization artifact")
    expected_quantization = {
        "bits": BITS,
        "minimum": MINIMUM,
        "maximum": MAXIMUM,
        "scheme": "symmetric-per-layer-round-to-nearest",
        "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
    }
    for name, expected in expected_quantization.items():
        if quantization.get(name) != expected:
            raise ValueError(f"unexpected quantization field {name}")
    scales_value = quantization.get("scales")
    if not isinstance(scales_value, dict):
        raise ValueError("quantization scales are missing")
    scales = {}
    for name in ("w1", "w2", "w3"):
        scales[name] = float(scales_value.get(name))
        _float_literal(scales[name])
    weights = []
    for name in ("w1", "w2", "w3"):
        weights.extend(tensor_values(model, name))
    return weights, scales


def _quoted_chunks(value: str, width: int = 96) -> str:
    return "\n".join(
        f'    "{value[start:start + width]}"'
        for start in range(0, len(value), width)
    )


def render(
    model: Mapping, model_sha256: str, seed: int | None = None
) -> tuple[str, dict]:
    model, selected_seed = select_checkpoint(model, seed)
    weights, scales = validate_model(model)
    payload = pack_signed_three_bit(weights)
    if unpack_signed_three_bit(payload, len(weights)) != weights:
        raise RuntimeError("internal signed three-bit packing mismatch")
    encoded = base64.b64encode(payload).decode("ascii")
    schema_payload = json.dumps({
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "architecture": model["architecture"],
        "rules": RULES,
        "packing": model["quantization"]["packing"],
    }, sort_keys=True, separators=(",", ":")).encode()
    schema_sha256 = hashlib.sha256(schema_payload).hexdigest()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
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
inline constexpr float kScale1 = {_float_literal(scales['w1'])};
inline constexpr float kScale2 = {_float_literal(scales['w2'])};
inline constexpr float kScale3 = {_float_literal(scales['w3'])};
inline constexpr std::string_view kModelSchema = "{MODEL_SCHEMA}";
inline constexpr std::string_view kFeatureSchema = "{FEATURE_SCHEMA}";
inline constexpr std::string_view kModelSha256 = "{model_sha256}";
inline constexpr std::string_view kSchemaSha256 = "{schema_sha256}";
inline constexpr std::string_view kPackedSha256 = "{payload_sha256}";
inline constexpr std::string_view kPackedWeights =
{_quoted_chunks(encoded)};

}}  // namespace papersoccer::jacek_native_model
"""
    metadata = {
        "model_sha256": model_sha256,
        "schema_sha256": schema_sha256,
        "packed_sha256": payload_sha256,
        "weight_count": len(weights),
        "packed_bytes": len(payload),
        "base64_characters": len(encoded),
        "header_characters": len(header),
        "training_seed": selected_seed,
    }
    return header, metadata


def render_runtime(model: Mapping, model_sha256: str, seed: int | None = None) -> str:
    selected, _ = select_checkpoint(model, seed)
    weights, scales = validate_model(selected)
    payload = pack_signed_three_bit(weights)
    encoded = base64.b64encode(payload).decode("ascii")
    packed_sha256 = hashlib.sha256(payload).hexdigest()
    return (
        "papersoccer.jacek-native-runtime-model/v1\n"
        f"{MODEL_SCHEMA}\n"
        f"{FEATURE_SCHEMA}\n"
        f"{model_sha256}\n"
        f"{packed_sha256}\n"
        f"{scales['w1']:.9g} {scales['w2']:.9g} {scales['w3']:.9g}\n"
        f"{encoded}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the compact Jacek-native Codingame model header."
    )
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_HEADER)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runtime-output", type=pathlib.Path)
    arguments = parser.parse_args()
    raw = arguments.model.read_bytes()
    header, metadata = render(
        json.loads(raw), hashlib.sha256(raw).hexdigest(), arguments.seed
    )
    runtime = render_runtime(
        json.loads(raw), hashlib.sha256(raw).hexdigest(), arguments.seed
    ) if arguments.runtime_output else None
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != header:
            print(f"{arguments.output} is stale", file=sys.stderr)
            return 1
        if (arguments.runtime_output and
                (not arguments.runtime_output.exists() or
                 arguments.runtime_output.read_text() != runtime)):
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
