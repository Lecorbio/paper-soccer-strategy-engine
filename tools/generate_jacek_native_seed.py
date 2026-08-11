#!/usr/bin/env python3
"""Generate the deterministic untrained checkpoint for iteration-zero self-play."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR = ROOT / "models" / "jacek_native_untrained_seed.json"
DEFAULT_RUNTIME = ROOT / "models" / "jacek_native_untrained_seed.runtime"
MODEL_SCHEMA = "jacek_native_model/v1"
FEATURE_SCHEMA = "canonical-edges316-onehot-true-turn-distance105x8-v1"
RUNTIME_SCHEMA = "papersoccer.jacek-native-runtime-model/v1"
SEED = 0x4A4143454B202601
COUNTS = {"w1": 1156 * 32, "w2": 32 * 32, "w3": 32}
SCALES = {"w1": 0.01, "w2": 0.05, "w3": 0.05}
DISTRIBUTIONS = {
    "w1": (-1, 0, 0, 1),
    "w2": (-2, -1, 0, 0, 1, 2),
    "w3": (-2, -1, 0, 0, 1, 2),
}


class SplitMix64:
    def __init__(self, state: int) -> None:
        self.state = state

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)


def weights() -> list[int]:
    random = SplitMix64(SEED)
    result = []
    for layer in ("w1", "w2", "w3"):
        distribution = DISTRIBUTIONS[layer]
        result.extend(
            distribution[random.next() % len(distribution)]
            for _ in range(COUNTS[layer])
        )
    return result


def pack(values: list[int]) -> bytes:
    output = bytearray()
    accumulator = 0
    bits = 0
    for value in values:
        accumulator |= (value & 0b111) << bits
        bits += 3
        while bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bits -= 8
    if bits:
        output.append(accumulator & 0xFF)
    return bytes(output)


def render() -> tuple[str, str, dict]:
    values = weights()
    payload = pack(values)
    payload_sha = hashlib.sha256(payload).hexdigest()
    generator_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    descriptor = {
        "schema": "papersoccer.jacek-native-untrained-seed/v1",
        "model_schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "training": None,
        "purpose": "deterministic-near-zero-iteration-zero-selfplay",
        "prng": "splitmix64-v1",
        "seed": str(SEED),
        "weights": {
            "counts": COUNTS,
            "distributions": {
                name: list(value) for name, value in DISTRIBUTIONS.items()
            },
            "scales": SCALES,
            "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
            "packed_sha256": payload_sha,
        },
        "generator_sha256": generator_sha,
        "incumbent_dependencies": False,
        "protected_data": False,
    }
    descriptor_text = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ) + "\n"
    model_sha = hashlib.sha256(descriptor_text.encode()).hexdigest()
    runtime_text = (
        f"{RUNTIME_SCHEMA}\n{MODEL_SCHEMA}\n{FEATURE_SCHEMA}\n{model_sha}\n"
        f"{payload_sha}\n{SCALES['w1']} {SCALES['w2']} {SCALES['w3']}\n"
        f"{base64.b64encode(payload).decode('ascii')}\n"
    )
    metadata = {
        "descriptor_sha256": model_sha,
        "runtime_sha256": hashlib.sha256(runtime_text.encode()).hexdigest(),
        "packed_sha256": payload_sha,
        "weights": len(values),
        "packed_bytes": len(payload),
    }
    return descriptor_text, runtime_text, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic untrained Jacek-native seed."
    )
    parser.add_argument("--descriptor", type=pathlib.Path,
                        default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--runtime", type=pathlib.Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    descriptor, runtime, metadata = render()
    expected = ((arguments.descriptor, descriptor), (arguments.runtime, runtime))
    if arguments.check:
        stale = [str(path) for path, value in expected
                 if not path.exists() or path.read_text() != value]
        if stale:
            print("stale Jacek-native seed artifacts: " + ", ".join(stale),
                  file=sys.stderr)
            return 1
    else:
        for path, value in expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
