#!/usr/bin/env python3
"""Losslessly encode the exact historical compact deployment; never edit it.

This research exporter changes model storage only. The source hash, packed
weights, dimensions, scales, evaluator, generator, and search remain bound to
the immutable discrete-v3 release. Future model formats require a distinct
exporter contract rather than silently weakening this identity check.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp"
DIRECTORY = ROOT / "submissions/codingame/bots/compact_huffman"
SOURCE_SHA256 = "add71c369052f232209d69c3b40b6bb459a2d7326ef15c5980377b1526fb8ea9"
PAYLOAD_SHA256 = "76739bbf764741a8722d873cef3b1efbac22b6f759106b6db00b2248fd4e4a00"
WORDS = {0: "0", 7: "10", 1: "110", 6: "1110", 2: "11110", 3: "111110", 5: "111111"}
PAYLOAD_PATTERN = re.compile(r'kPackedWeights\s*=\s*((?:\s*"[^"\n]*"\s*)+);')


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def unpack_three_bit(payload: bytes, count: int) -> list[int]:
    if count < 0 or len(payload) != (count * 3 + 7) // 8:
        raise ValueError("three-bit payload length")
    if count * 3 % 8 and payload[-1] >> (count * 3 % 8):
        raise ValueError("three-bit payload padding")
    codes = []
    for index in range(count):
        bit = index * 3
        word = int.from_bytes(payload[bit // 8:bit // 8 + 2], "little")
        code = (word >> (bit % 8)) & 7
        if code == 4:
            raise ValueError("forbidden signed -4 code")
        codes.append(code)
    return codes


def pack_three_bit(codes: list[int]) -> bytes:
    payload = bytearray((len(codes) * 3 + 7) // 8)
    for index, code in enumerate(codes):
        if code not in WORDS:
            raise ValueError("invalid signed three-bit code")
        bit = index * 3
        payload[bit // 8] |= (code << (bit % 8)) & 255
        if bit % 8 > 5:
            payload[bit // 8 + 1] |= code >> (8 - bit % 8)
    return bytes(payload)


def encode(codes: list[int]) -> tuple[bytes, int]:
    output = bytearray()
    cursor = 0
    for code in codes:
        if code not in WORDS:
            raise ValueError("invalid signed three-bit code")
        for value in WORDS[code]:
            if cursor % 8 == 0:
                output.append(0)
            output[-1] |= int(value) << (cursor % 8)
            cursor += 1
    return bytes(output), cursor


def decode(payload: bytes, count: int) -> list[int]:
    if count < 0 or count > len(payload) * 8:
        raise ValueError("huffman weight count")
    order = (0, 7, 1, 6, 2, 3, 5)
    codes = []
    cursor = 0
    for _ in range(count):
        ones = 0
        while ones < 6:
            if cursor // 8 >= len(payload):
                raise ValueError("huffman truncated code")
            bit = (payload[cursor // 8] >> (cursor % 8)) & 1
            cursor += 1
            if not bit:
                break
            ones += 1
        codes.append(order[ones])
    if ((cursor + 7) // 8 != len(payload)
            or (cursor % 8 and payload[-1] >> (cursor % 8))):
        raise ValueError("huffman trailing data")
    return codes


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"source anchor must occur exactly once: {old[:80]}")
    return text.replace(old, new, 1)


def generate(base: Path = BASE) -> tuple[bytes, dict]:
    original = base.read_bytes()
    if sha(original) != SOURCE_SHA256:
        raise ValueError("immutable discrete-v3 source identity changed")
    source = original.decode("ascii")
    match = PAYLOAD_PATTERN.search(source)
    if match is None:
        raise ValueError("model payload absent")
    packed = base64.b64decode("".join(re.findall(r'"([^"\n]*)"', match[1])), validate=True)
    if sha(packed) != PAYLOAD_SHA256:
        raise ValueError("model payload identity changed")
    count = int(re.search(r"kWeightCount\s*=\s*(\d+)", source)[1])
    codes = unpack_three_bit(packed, count)
    compressed, bits = encode(codes)
    if pack_three_bit(decode(compressed, count)) != packed:
        raise ValueError("model compression does not roundtrip")
    encoded = base64.b64encode(compressed).decode("ascii")
    literal = "\n".join('"' + encoded[i:i + 96] + '"' for i in range(0, len(encoded), 96))
    source = source[:match.start(1)] + literal + source[match.end(1):]
    source = replace_once(source, "bool allow_empty_bootstrap{};", "bool allow_empty_bootstrap{};\nbool huffman{};")
    source = replace_once(source,
        "const std::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);",
        "std::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);\n"
        "if (descriptor.huffman) bytes = compact_huffman::decode(bytes, count);")
    source = replace_once(source,
        "model::kPackedWeights, model::kPayloadSha256, model::kBootstrapZero});",
        "model::kPackedWeights, model::kPayloadSha256, model::kBootstrapZero, true});")
    codec_path = DIRECTORY / "runtime_codec.hpp"
    codec = codec_path.read_text(encoding="ascii")
    # Insert the independent codec before the original namespace. Keep its
    # comments for review; only redundant header directives are dropped.
    body = "\n".join(line for line in codec.splitlines() if not line.startswith(("#pragma", "#include")))
    source = replace_once(source, "namespace compact_value_bfm::model {", body + "\nnamespace compact_value_bfm::model {")
    result = source.encode("ascii")
    if len(result) >= 95_000:
        raise ValueError("research source exceeds conservative 95,000-character cap")
    counts = collections.Counter(codes)
    report = {
        "schema": "papersoccer.compact-huffman.export.v1",
        "classification": "lossless-storage-research-candidate-not-game-qualified",
        "base_source": {"path": BASE.relative_to(ROOT).as_posix(), "sha256": sha(original), "bytes": len(original)},
        "codec": {"path": codec_path.relative_to(ROOT).as_posix(), "sha256": sha(codec.encode("ascii")), "name": "signed3-prefix-huffman-lsb-v1"},
        "original_payload_sha256": sha(packed),
        "compressed_payload_sha256": sha(compressed),
        "weight_count": count,
        "signed_weight_counts": {str(code if code < 4 else code - 8): counts[code] for code in sorted(counts)},
        "compressed_bits": bits,
        "original_packed_bytes": len(packed),
        "compressed_bytes": len(compressed),
        "original_base64_characters": len(base64.b64encode(packed)),
        "compressed_base64_characters": len(encoded),
        "source": {"path": "submissions/codingame/bots/compact_huffman/submission.cpp", "sha256": sha(result), "bytes": len(result)},
        "net_source_characters_saved": len(original) - len(result),
        "reserve_under_95000": 95_000 - len(result),
        "weights_roundtrip_exact": True,
        "architecture": [6301, 12, 8, 1],
        "weight_bits": 3,
        "evaluation_and_search_changed": False,
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source, report = generate()
    outputs = {DIRECTORY / "submission.cpp": source,
               DIRECTORY / "export.json": (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("ascii")}
    for path, data in outputs.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f"stale generated artifact: {path}")
        else:
            path.write_bytes(data)
    print(json.dumps({"checked": args.check, "source_bytes": len(source),
                      "saved": report["net_source_characters_saved"],
                      "sha256": report["source"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
