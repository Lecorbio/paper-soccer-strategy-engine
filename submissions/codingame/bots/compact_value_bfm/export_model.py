#!/usr/bin/env python3
"""Validate a selected compact runtime and atomically emit ``model.hpp``."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import pathlib
import struct
import tempfile
from typing import Any


RUNTIME_SCHEMA = "papersoccer.compact-value-bfm-runtime.v1"
CHANNEL_RUNTIME_SCHEMA = "papersoccer.compact-value-bfm-runtime.v2"
CHANNEL_QAT_PROFILE = "channel-prediction-qat-v1"
CHANNEL_SCALE_COUNTS = {"w1": 12, "w2": 8, "w3": 1}
FEATURE_SCHEMA = (
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree"
)
ACTIVATIONS = [
    "square-leaky-0.01",
    "leaky-relu-0.01",
    "fast-tanh-rational-v1",
]
LAYOUT = "w1-input-major,w2-input-major,w3"
QUANTIZATION = {
    "bits": 3,
    "minimum": -3,
    "maximum": 3,
    "scheme": "symmetric-signed-three-bit-per-layer-fixed-scale",
    "packing": "signed-three-bit-twos-complement-lsb-first",
}
CHANNEL_QUANTIZATION = {
    **QUANTIZATION,
    "scheme": "symmetric-signed-three-bit-per-output-channel-fixed-scale",
    "granularity": "per-output-channel",
    "scale_axis": "output",
}
ELIGIBLE = {
    (8, 8): "compact-8x8",
    (8, 16): "source-neutral-8x16",
    (12, 8): "capacity-12x8",
}
HERE = pathlib.Path(__file__).resolve().parent


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("ascii")


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def quoted_chunks(value: str, width: int = 96) -> str:
    return "\n".join(
        f'    "{value[index:index + width]}"'
        for index in range(0, len(value), width)
    )


def float_literal(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric float32")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    try:
        float32 = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{field} is outside float32") from error
    if float32 != number:
        raise ValueError(f"{field} is not an exact finite float32 value")
    rendered = f"{number:.9g}"
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "F"


def validate_runtime(path: pathlib.Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    expected_name = f"{file_sha}.runtime.json"
    if path.name != expected_name:
        raise ValueError(f"runtime filename must be content-addressed as {expected_name}")
    runtime = json.loads(raw)
    if not isinstance(runtime, dict) or runtime.get("schema") not in {
        RUNTIME_SCHEMA, CHANNEL_RUNTIME_SCHEMA
    }:
        raise ValueError("unexpected compact runtime schema")
    channel = runtime["schema"] == CHANNEL_RUNTIME_SCHEMA
    if set(runtime) != {"schema", "feature_schema", "architecture", "quantization", "selection", "body_sha256"}:
        raise ValueError("runtime fields changed")
    if canonical_json_bytes(runtime) != raw:
        raise ValueError("runtime JSON is not canonical")
    if runtime.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("unexpected compact feature schema")
    body_sha = runtime.get("body_sha256")
    if not valid_sha256(body_sha):
        raise ValueError("runtime body_sha256 is invalid")
    body = dict(runtime)
    del body["body_sha256"]
    actual_body_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if body_sha != actual_body_sha:
        raise ValueError("runtime body SHA-256 mismatch")

    architecture = runtime.get("architecture")
    if not isinstance(architecture, dict) or set(architecture) != {
        "name", "dimensions", "biases", "activations", "payload_layout"
    }:
        raise ValueError("runtime architecture is missing")
    dimensions = architecture.get("dimensions")
    if (
        dimensions not in ([6301, 8, 8, 1], [6301, 8, 16, 1], [6301, 12, 8, 1])
        or any(type(value) is not int for value in dimensions)
        or channel and dimensions != [6301, 12, 8, 1]
        or architecture.get("biases") is not False
        or architecture.get("activations") != ACTIVATIONS
        or architecture.get("payload_layout") != LAYOUT
    ):
        raise ValueError("runtime architecture contract mismatch")
    hidden_one, hidden_two = dimensions[1], dimensions[2]
    if architecture.get("name") != ELIGIBLE[(hidden_one, hidden_two)]:
        raise ValueError("runtime architecture name does not match its dimensions")

    quantization = runtime.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("runtime quantization is missing")
    constants = CHANNEL_QUANTIZATION if channel else QUANTIZATION
    expected_fields = set(constants) | {"scales", "weight_counts", "packed_byte_count", "payload_sha256", "payload_base64"}
    if channel:
        expected_fields.add("scale_counts")
    if set(quantization) != expected_fields:
        raise ValueError("runtime mixes quantization schema fields")
    for field, expected in constants.items():
        if quantization.get(field) != expected:
            raise ValueError(f"unexpected quantization field {field}")
        if type(expected) is int and type(quantization[field]) is not int:
            raise ValueError(f"quantization field {field} must be integer")
    scales = quantization.get("scales")
    if not isinstance(scales, dict) or set(scales) != {"w1", "w2", "w3"}:
        raise ValueError("runtime scales are incomplete")
    if channel:
        scale_counts = quantization.get("scale_counts")
        if (scale_counts != CHANNEL_SCALE_COUNTS or not isinstance(scale_counts, dict)
                or any(type(value) is not int for value in scale_counts.values())):
            raise ValueError("runtime channel scale counts changed")
        if any(not isinstance(scales[name], list) or len(scales[name]) != count
               for name, count in CHANNEL_SCALE_COUNTS.items()):
            raise ValueError("runtime channel scale lengths changed")
        scale_literals = {name: [float_literal(value, f"scale {name}[{index}]")
                                for index, value in enumerate(scales[name])]
                          for name in CHANNEL_SCALE_COUNTS}
    else:
        scale_literals = {name: float_literal(scales[name], f"scale {name}")
                          for name in ("w1", "w2", "w3")}
    counts = {
        "w1": 6301 * hidden_one,
        "w2": hidden_one * hidden_two,
        "w3": hidden_two,
    }
    counts["total"] = counts["w1"] + counts["w2"] + counts["w3"]
    if (quantization.get("weight_counts") != counts
            or any(type(value) is not int for value in quantization["weight_counts"].values())):
        raise ValueError("runtime weight counts mismatch")
    expected_bytes = (counts["total"] * 3 + 7) // 8
    if type(quantization.get("packed_byte_count")) is not int or quantization["packed_byte_count"] != expected_bytes:
        raise ValueError("runtime packed byte count mismatch")
    encoded = quantization.get("payload_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("runtime payload_base64 is missing")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError("runtime payload is not canonical base64") from error
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise ValueError("runtime payload base64 is not canonical")
    if len(payload) != expected_bytes:
        raise ValueError("runtime payload length mismatch")
    payload_sha = quantization.get("payload_sha256")
    if not valid_sha256(payload_sha) or hashlib.sha256(payload).hexdigest() != payload_sha:
        raise ValueError("runtime payload SHA-256 mismatch")
    for index in range(counts["total"]):
        bit = index * 3
        window = payload[bit // 8]
        if bit % 8 > 5 and bit // 8 + 1 < len(payload):
            window |= payload[bit // 8 + 1] << 8
        code = (window >> (bit % 8)) & 7
        if code == 4:
            raise ValueError("runtime payload contains forbidden code 100")
        if channel:
            if index < counts["w1"]:
                scale = scales["w1"][index % hidden_one]
            elif index < counts["w1"] + counts["w2"]:
                scale = scales["w2"][(index - counts["w1"]) % hidden_two]
            else:
                scale = scales["w3"][0]
            signed_code = code - 8 if code & 4 else code
            try:
                effective = struct.unpack("<f", struct.pack("<f", signed_code * scale))[0]
            except (OverflowError, struct.error) as error:
                raise ValueError("runtime channel effective weight is nonfinite") from error
            if not math.isfinite(effective):
                raise ValueError("runtime channel effective weight is nonfinite")
    tail = counts["total"] * 3 % 8
    if tail and payload[-1] >> tail:
        raise ValueError("runtime payload has nonzero trailing padding")

    selection = runtime.get("selection")
    selection_fields = {"arm", "seed", "float_epoch", "qat_epoch", "source_bundle_body_sha256"}
    if channel:
        selection_fields.update(("qat_profile", "qat_evidence_sha256"))
    if not isinstance(selection, dict) or set(selection) != selection_fields:
        raise ValueError("runtime selection binding is incomplete")
    if (
        selection["arm"] not in {"search-target", "teacher-assisted"}
        or isinstance(selection["seed"], bool)
        or not isinstance(selection["seed"], int)
        or selection["seed"] not in {20260907, 20260908, 20260909}
        or isinstance(selection["float_epoch"], bool)
        or not isinstance(selection["float_epoch"], int)
        or not 1 <= selection["float_epoch"] <= 50
        or isinstance(selection["qat_epoch"], bool)
        or not isinstance(selection["qat_epoch"], int)
        or not 0 <= selection["qat_epoch"] <= 4
        or not valid_sha256(selection["source_bundle_body_sha256"])
    ):
        raise ValueError("runtime selection values are invalid")
    if channel and (selection["qat_profile"] != CHANNEL_QAT_PROFILE
                    or not valid_sha256(selection["qat_evidence_sha256"])):
        raise ValueError("runtime channel QAT evidence binding changed")
    metadata = {
        "file_sha256": file_sha,
        "body_sha256": body_sha,
        "payload_sha256": payload_sha,
        "encoded": encoded,
        "hidden_one": hidden_one,
        "hidden_two": hidden_two,
        "counts": counts,
        "packed_bytes": expected_bytes,
        "scales": scale_literals,
        "identity": f"{architecture['name']}-s{selection['seed']}-{body_sha[:12]}",
    }
    return runtime, payload, metadata


def render_header(path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    runtime, _payload, metadata = validate_runtime(path)
    architecture = runtime["architecture"]
    if runtime["schema"] == CHANNEL_RUNTIME_SCHEMA:
        declarations = "\n".join(
            f'inline constexpr std::array<float, {CHANNEL_SCALE_COUNTS[name]}> {constant}{{'
            + ", ".join(metadata["scales"][name]) + "};"
            for name, constant in (("w1", "kScaleOne"), ("w2", "kScaleTwo"), ("w3", "kScaleThree"))
        )
        content = f'''#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#define COMPACT_VALUE_BFM_CHANNEL_MODEL_V2 1

namespace compact_value_bfm::model {{
inline constexpr std::size_t kInputs = 6301;
inline constexpr std::size_t kHiddenOne = 12;
inline constexpr std::size_t kHiddenTwo = 8;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = {metadata["counts"]["total"]};
inline constexpr std::size_t kPackedByteCount = {metadata["packed_bytes"]};
{declarations}
inline constexpr std::string_view kRuntimeSchema = "{CHANNEL_RUNTIME_SCHEMA}";
inline constexpr std::string_view kFeatureSchema = "{FEATURE_SCHEMA}";
inline constexpr std::string_view kPayloadSha256 = "{metadata["payload_sha256"]}";
inline constexpr std::string_view kRuntimeBodySha256 = "{metadata["body_sha256"]}";
inline constexpr std::string_view kQatProfile = "{CHANNEL_QAT_PROFILE}";
inline constexpr std::string_view kQatEvidenceSha256 = "{runtime["selection"]["qat_evidence_sha256"]}";
inline constexpr std::string_view kIdentity = "{metadata["identity"]}";
inline constexpr std::string_view kPackedWeights =
{quoted_chunks(metadata["encoded"])};
}}  // namespace compact_value_bfm::model
'''.encode("ascii")
        metadata.update({"architecture": architecture, "header_sha256": hashlib.sha256(content).hexdigest(),
                         "header_characters": len(content), "runtime_schema": CHANNEL_RUNTIME_SCHEMA})
        return content, metadata
    content = f'''#pragma once

#include <cstddef>
#include <string_view>

namespace compact_value_bfm::model {{
inline constexpr std::size_t kInputs = 6301;
inline constexpr std::size_t kHiddenOne = {metadata["hidden_one"]};
inline constexpr std::size_t kHiddenTwo = {metadata["hidden_two"]};
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = {metadata["counts"]["total"]};
inline constexpr std::size_t kPackedByteCount = {metadata["packed_bytes"]};
inline constexpr float kScaleOne = {metadata["scales"]["w1"]};
inline constexpr float kScaleTwo = {metadata["scales"]["w2"]};
inline constexpr float kScaleThree = {metadata["scales"]["w3"]};
inline constexpr bool kBootstrapZero = false;
inline constexpr std::string_view kRuntimeSchema = "{RUNTIME_SCHEMA}";
inline constexpr std::string_view kFeatureSchema = "{FEATURE_SCHEMA}";
inline constexpr std::string_view kPayloadSha256 = "{metadata["payload_sha256"]}";
inline constexpr std::string_view kRuntimeBodySha256 = "{metadata["body_sha256"]}";
inline constexpr std::string_view kIdentity = "{metadata["identity"]}";
inline constexpr std::string_view kPackedWeights =
{quoted_chunks(metadata["encoded"])};
}}  // namespace compact_value_bfm::model
'''.encode("ascii")
    metadata.update({
        "architecture": architecture,
        "header_sha256": hashlib.sha256(content).hexdigest(),
        "header_characters": len(content),
    })
    return content, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "model.hpp")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    arguments = parser.parse_args()
    content, metadata = render_header(arguments.runtime)
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_bytes() != content:
            raise SystemExit(f"{arguments.output} is stale")
    else:
        atomic_write(arguments.output, content)
    if arguments.metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        print(f"compact model header current ({len(content)} ASCII characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
