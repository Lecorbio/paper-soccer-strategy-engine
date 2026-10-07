#!/usr/bin/env python3
"""Independent signed-3/4-bit compact evaluator format and source exporter."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import struct

import compact_huffman as incumbent

ROOT = incumbent.ROOT
DIRECTORY = ROOT / "submissions/codingame/bots/compact_representation"
SCHEMA = "papersoccer.compact-representation.runtime.v1"
FEATURE_SCHEMA = "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:mover-relative-rotate180:true-turn-distance+free-degree"
PROFILES = {"h12-b3": (12, 3), "h12-b4": (12, 4), "h16-b3": (16, 3)}


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack(values: list[int], width: int) -> bytes:
    if width not in (3, 4): raise ValueError("unsupported weight width")
    limit = (1 << (width - 1)) - 1
    output = bytearray((len(values) * width + 7) // 8)
    for index, value in enumerate(values):
        if type(value) is not int or not -limit <= value <= limit: raise ValueError("signed weight out of range")
        code = value & ((1 << width) - 1)
        bit = index * width
        output[bit // 8] |= (code << (bit % 8)) & 255
        if bit % 8 + width > 8: output[bit // 8 + 1] |= code >> (8 - bit % 8)
    return bytes(output)


def unpack(payload: bytes, count: int, width: int) -> list[int]:
    if width not in (3, 4) or count < 0 or len(payload) != (count * width + 7) // 8:
        raise ValueError("packed weight dimensions")
    if count * width % 8 and payload[-1] >> (count * width % 8): raise ValueError("packed weight padding")
    values = []
    for index in range(count):
        bit = index * width
        code = (int.from_bytes(payload[bit // 8:bit // 8 + 2], "little") >> (bit % 8)) & ((1 << width) - 1)
        if code == 1 << (width - 1): raise ValueError("reserved negative weight")
        values.append(code - (1 << width) if code >= 1 << (width - 1) else code)
    return values


def codebook(lengths: list[int], width: int) -> dict[int, str]:
    if width not in (3, 4) or len(lengths) != 16: raise ValueError("codebook dimensions")
    entries = []
    for symbol, length in enumerate(lengths):
        if type(length) is not int or not 0 <= length <= 15: raise ValueError("invalid code length")
        if length:
            if symbol >= 1 << width or symbol == 1 << (width - 1): raise ValueError("reserved codebook symbol")
            entries.append((length, symbol))
    if not entries: raise ValueError("empty alphabet")
    code, previous = 0, 0
    result = {}
    for length, symbol in sorted(entries):
        code <<= length - previous
        if code >= 1 << length: raise ValueError("oversubscribed codebook")
        result[symbol] = format(code, f"0{length}b")
        code += 1
        previous = length
    return result


def compress(values: list[int], width: int) -> tuple[bytes, list[int], int]:
    pack(values, width)  # Validate the full signed domain before coding.
    if not values: raise ValueError("cannot compress empty model")
    symbols = [value & ((1 << width) - 1) for value in values]
    frequencies = Counter(symbols)
    heap = [(frequency, [symbol]) for symbol, frequency in frequencies.items()]
    heapq.heapify(heap)
    lengths = [0] * 16
    if len(heap) == 1: lengths[heap[0][1][0]] = 1
    while len(heap) > 1:
        first, left = heapq.heappop(heap)
        second, right = heapq.heappop(heap)
        for symbol in left + right: lengths[symbol] += 1
        heapq.heappush(heap, (first + second, left + right))
    words = codebook(lengths, width)
    output, cursor = bytearray(), 0
    for symbol in symbols:
        for value in words[symbol]:
            if cursor % 8 == 0: output.append(0)
            output[-1] |= int(value) << (cursor % 8)
            cursor += 1
    return bytes(output), lengths, cursor


def decompress(payload: bytes, count: int, width: int, lengths: list[int]) -> list[int]:
    if count < 0 or count > len(payload) * 8: raise ValueError("huffman count")
    words = {word: symbol for symbol, word in codebook(lengths, width).items()}
    result, cursor = [], 0
    for _ in range(count):
        word = ""
        for _ in range(15):
            if cursor // 8 >= len(payload): raise ValueError("truncated Huffman stream")
            word += str((payload[cursor // 8] >> (cursor % 8)) & 1)
            cursor += 1
            if word in words: break
        else: raise ValueError("invalid Huffman prefix")
        symbol = words[word]
        result.append(symbol - (1 << width) if symbol >= 1 << (width - 1) else symbol)
    if (cursor + 7) // 8 != len(payload) or (cursor % 8 and payload[-1] >> (cursor % 8)):
        raise ValueError("trailing Huffman data")
    return result


def document(profile: str, values: list[int], scales: list[float], lineage: dict) -> dict:
    if profile not in PROFILES: raise ValueError("unknown representation profile")
    hidden, width = PROFILES[profile]
    if len(values) != 6301 * hidden + hidden * 8 + 8: raise ValueError("weight count")
    scales = [f32(scale) for scale in scales]
    if len(scales) != 3 or any(not math.isfinite(scale) or scale <= 0 for scale in scales): raise ValueError("invalid scales")
    payload = pack(values, width)
    body = {"schema": SCHEMA, "profile": profile, "architecture": [6301, hidden, 8, 1],
            "feature_schema": FEATURE_SCHEMA, "weight_bits": width, "biases": False,
            "scales": scales, "payload_base64": base64.b64encode(payload).decode("ascii"),
            "payload_sha256": incumbent.sha(payload), "lineage": lineage}
    return {**body, "body_sha256": incumbent.sha(canonical(body))}


def validate(runtime: dict) -> tuple[list[int], int, int]:
    if not isinstance(runtime, dict): raise ValueError("runtime must be an object")
    body = {key: value for key, value in runtime.items() if key != "body_sha256"}
    if runtime.get("body_sha256") != incumbent.sha(canonical(body)): raise ValueError("runtime body digest")
    profile = runtime.get("profile")
    if profile not in PROFILES: raise ValueError("unknown runtime profile")
    hidden, width = PROFILES[profile]
    if (runtime.get("schema") != SCHEMA or runtime.get("architecture") != [6301, hidden, 8, 1]
            or runtime.get("feature_schema") != FEATURE_SCHEMA or runtime.get("weight_bits") != width
            or runtime.get("biases") is not False): raise ValueError("runtime contract")
    payload = base64.b64decode(runtime["payload_base64"], validate=True)
    if incumbent.sha(payload) != runtime.get("payload_sha256"): raise ValueError("runtime payload digest")
    values = unpack(payload, 6301 * hidden + hidden * 8 + 8, width)
    expected = document(profile, values, runtime["scales"], runtime["lineage"])
    if expected != runtime: raise ValueError("noncanonical runtime")
    return values, hidden, width


def initialize(profile: str) -> dict:
    source = incumbent.BASE.read_bytes()
    if incumbent.sha(source) != incumbent.SOURCE_SHA256: raise ValueError("immutable incumbent changed")
    text = source.decode("ascii")
    match = incumbent.PAYLOAD_PATTERN.search(text)
    payload = base64.b64decode("".join(re.findall(r'"([^"\n]*)"', match[1])), validate=True)
    if incumbent.sha(payload) != incumbent.PAYLOAD_SHA256: raise ValueError("incumbent payload changed")
    values = unpack(payload, 75716, 3)
    scales = [float(re.search(rf"kScale{name}\s*=\s*([\d.eE+-]+)F", text)[1]) for name in ("One", "Two", "Three")]
    hidden, _ = PROFILES[profile]
    if hidden == 16:
        # Function-preserving zero-padding; train initialization may subsequently
        # activate extra hidden units under a separately reported operation.
        w1 = [value for row in range(6301) for value in values[row * 12:row * 12 + 12] + [0] * 4]
        w2 = values[6301 * 12:6301 * 12 + 96] + [0] * 32
        values = w1 + w2 + values[-8:]
    return document(profile, values, scales, {"kind": "function-preserving-incumbent-conversion",
                    "base_source_sha256": incumbent.SOURCE_SHA256, "trained": False})


def source_size_policy(size: int, source_limit: int = 99999) -> dict:
    if type(source_limit) is not int or not 1 <= source_limit <= 99999:
        raise ValueError("source limit must be at most 99,999 ASCII characters")
    return {"hard_source_limit": source_limit, "preferred_source_target": 95000,
            "deployable_size": size <= source_limit, "preferred_source_size": size <= 95000,
            "reserve_under_limit": source_limit - size, "reserve_under_95000": 95000 - size}


def export(runtime: dict, source_limit: int = 99999) -> tuple[bytes, dict]:
    source_size_policy(0, source_limit)
    values, hidden, width = validate(runtime)
    compressed, lengths, bits = compress(values, width)
    if decompress(compressed, len(values), width, lengths) != values: raise ValueError("codec roundtrip")
    original = incumbent.BASE.read_bytes()
    if incumbent.sha(original) != incumbent.SOURCE_SHA256: raise ValueError("immutable base source changed")
    text = original.decode("ascii")
    # The independent Python scalar contract rounds every operation to float32.
    # Enforce that same contract even when the caller omits compiler flags.
    text = text.replace('#pragma GCC optimize("O3")', '#pragma GCC optimize("O3", "fp-contract=off")', 1)
    text = '#if defined(__clang__)\n#pragma clang fp contract(off)\n#endif\n' + text
    match = incumbent.PAYLOAD_PATTERN.search(text)
    encoded = base64.b64encode(compressed).decode("ascii")
    literal = "\n".join('"' + encoded[i:i + 96] + '"' for i in range(0, len(encoded), 96))
    text = text[:match.start(1)] + literal + text[match.end(1):]
    replace = incumbent.replace_once
    for key, value in {"kHiddenOne": hidden, "kWeightCount": len(values), "kPackedByteCount": len(pack(values, width))}.items():
        text = re.sub(rf"({key}\s*=\s*)\d+", rf"\g<1>{value}", text, count=1)
    for name, scale in zip(("One", "Two", "Three"), runtime["scales"]):
        rendered = format(scale, ".9g")
        if "." not in rendered and "e" not in rendered: rendered += ".0"
        text = re.sub(rf"(kScale{name}\s*=\s*)[\d.eE+-]+F", rf"\g<1>{rendered}F", text, count=1)
    for key, value in {"kPayloadSha256": runtime["payload_sha256"], "kRuntimeBodySha256": runtime["body_sha256"],
                       "kRuntimeSchema": SCHEMA, "kIdentity": "repr-" + runtime["profile"] + "-" + runtime["body_sha256"][:12]}.items():
        text = re.sub(rf'({key}\s*=\s*)"[^"]*"', rf'\g<1>"{value}"', text, count=1)
    metadata = f"inline constexpr unsigned kWeightBits = {width};\ninline constexpr std::array<std::uint8_t, 16> kHuffmanLengths{{{','.join(map(str, lengths))}}};\n"
    text = replace(text, "inline constexpr bool kBootstrapZero", metadata + "inline constexpr bool kBootstrapZero")
    text = replace(text, "bool allow_empty_bootstrap{};", "bool allow_empty_bootstrap{};\nunsigned weight_bits{3};\nbool huffman{};\nstd::array<std::uint8_t, 16> huffman_lengths{};")
    text = text.replace("std::array<std::int32_t, 12>", "std::array<std::int32_t, 16>")
    text = text.replace("std::array<float, 12>", "std::array<float, 16>")
    text = replace(text, "(hidden_one_ == 12 && hidden_two_ == 8));", "((hidden_one_ == 12 || hidden_one_ == 16) && hidden_two_ == 8));")
    text = replace(text, "if (!architecture || !std::isfinite(scale_one_)", "if ((descriptor.weight_bits != 3 && descriptor.weight_bits != 4) || !architecture || !std::isfinite(scale_one_)")
    text = replace(text, "const std::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);",
                   "const unsigned width = descriptor.weight_bits;\nstd::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);\n"
                   "if (descriptor.huffman) bytes = compact_representation::decode(bytes, count, width, descriptor.huffman_lengths);")
    text = replace(text, "bytes.size() != (count * 3U + 7U) / 8U", "bytes.size() != (count * width + 7U) / 8U")
    text = replace(text, "const std::size_t tail = count * 3U % 8U;", "const std::size_t tail = count * width % 8U;")
    text = replace(text, "const std::size_t bit = index * 3U;", "const std::size_t bit = index * width;")
    text = replace(text, "if (bit % 8U > 5U && bit / 8U + 1U < bytes.size())", "if (bit % 8U + width > 8U && bit / 8U + 1U < bytes.size())")
    text = replace(text, "const int code = static_cast<int>((window >> (bit % 8U)) & 7U);\nconst int signed_value = (code & 4) != 0 ? code - 8 : code;\nif (signed_value == -4) throw std::invalid_argument(\"model code 100\");",
                   "const int code = static_cast<int>((window >> (bit % 8U)) & ((1U << width) - 1U));\n"
                   "const int signed_value = (code & (1U << (width - 1U))) != 0 ? code - static_cast<int>(1U << width) : code;\n"
                   "if (signed_value == -(1 << (width - 1U))) throw std::invalid_argument(\"model reserved code\");")
    text = replace(text, "model::kPackedWeights, model::kPayloadSha256, model::kBootstrapZero});",
                   "model::kPackedWeights, model::kPayloadSha256, model::kBootstrapZero, model::kWeightBits, true, model::kHuffmanLengths});")
    codec = (DIRECTORY / "runtime_codec.hpp").read_text(encoding="ascii")
    body = "\n".join(line for line in codec.splitlines() if not line.startswith(("#pragma", "#include")))
    text = replace(text, "namespace compact_value_bfm::model {", body + "\nnamespace compact_value_bfm::model {")
    output = text.encode("ascii")
    report = {"schema": "papersoccer.compact-representation.export.v1", "profile": runtime["profile"],
              "runtime_body_sha256": runtime["body_sha256"], "base_source_sha256": incumbent.SOURCE_SHA256,
              "codec_sha256": incumbent.sha(codec.encode("ascii")), "payload_sha256": runtime["payload_sha256"],
              "source_sha256": incumbent.sha(output), "source_bytes": len(output),
              "compressed_bytes": len(compressed), "compressed_bits": bits, "code_lengths": lengths,
              **source_size_policy(len(output), source_limit),
              "game_qualified": False}
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init")
    init.add_argument("--profile", choices=PROFILES, required=True)
    init.add_argument("--output", type=Path, required=True)
    emit = subs.add_parser("export")
    emit.add_argument("--runtime", type=Path, required=True)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--source-limit", type=int, default=99999)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "init":
        runtime = initialize(args.profile)
        args.output.write_bytes(canonical(runtime))
        print(json.dumps({"profile": args.profile, "runtime_body_sha256": runtime["body_sha256"]}))
    else:
        source, report = export(json.loads(args.runtime.read_bytes()), args.source_limit)
        if not report["deployable_size"]: raise SystemExit("source exceeds the declared ASCII character limit")
        args.output.write_bytes(source)
        args.output.with_suffix(".export.json").write_bytes(canonical(report))
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
