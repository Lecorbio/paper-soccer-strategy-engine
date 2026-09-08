import base64
import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_model  # noqa: E402
import export_submission  # noqa: E402


class CompactExporterTests(unittest.TestCase):
    def runtime(self, root: pathlib.Path, hidden_one: int, hidden_two: int,
                *, name: str | None = None, scale: float = 0.125,
                mutate_payload=None, channel: bool = False, mutate_body=None) -> pathlib.Path:
        names = {(8, 8): "compact-8x8", (8, 16): "source-neutral-8x16",
                 (12, 8): "capacity-12x8"}
        counts = {
            "w1": 6301 * hidden_one,
            "w2": hidden_one * hidden_two,
            "w3": hidden_two,
        }
        counts["total"] = sum(counts.values())
        payload = bytearray((counts["total"] * 3 + 7) // 8)
        if mutate_payload:
            mutate_payload(payload)
        body = {
            "schema": export_model.RUNTIME_SCHEMA,
            "feature_schema": export_model.FEATURE_SCHEMA,
            "architecture": {
                "name": name or names[(hidden_one, hidden_two)],
                "dimensions": [6301, hidden_one, hidden_two, 1],
                "biases": False,
                "activations": export_model.ACTIVATIONS,
                "payload_layout": export_model.LAYOUT,
            },
            "quantization": {
                **export_model.QUANTIZATION,
                "scales": {"w1": scale, "w2": scale, "w3": scale},
                "weight_counts": counts,
                "packed_byte_count": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            },
            "selection": {
                "arm": "search-target",
                "seed": 20260907,
                "float_epoch": 1,
                "qat_epoch": 0,
                "source_bundle_body_sha256": "1" * 64,
            },
        }
        if channel:
            body["schema"] = export_model.CHANNEL_RUNTIME_SCHEMA
            body["quantization"].update(export_model.CHANNEL_QUANTIZATION)
            body["quantization"]["scale_counts"] = dict(export_model.CHANNEL_SCALE_COUNTS)
            body["quantization"]["scales"] = {
                key: [scale] * count for key, count in export_model.CHANNEL_SCALE_COUNTS.items()
            }
            body["selection"].update(qat_profile=export_model.CHANNEL_QAT_PROFILE,
                                     qat_evidence_sha256="2" * 64)
        if mutate_body:
            mutate_body(body)
        runtime = dict(body)
        runtime["body_sha256"] = hashlib.sha256(
            export_model.canonical_json_bytes(body)).hexdigest()
        raw = export_model.canonical_json_bytes(runtime)
        path = root / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
        path.write_bytes(raw)
        return path

    def test_all_architectures_render_under_source_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sizes = []
            for shape in ((8, 8), (8, 16), (12, 8)):
                runtime = self.runtime(root, *shape)
                header, metadata = export_model.render_header(runtime)
                self.assertEqual(metadata["architecture"]["dimensions"],
                                 [6301, *shape, 1])
                _, source = export_submission.render(model_header=header)
                self.assertLess(len(source), 95_000)
                if shape == (12, 8):
                    self.assertLessEqual(len(source), 93_000)
                sizes.append(len(source))
            self.assertGreater(sizes[-1], sizes[0])

    def test_cpp_compaction_preserves_literals_and_token_boundaries(self):
        source = '''
#if defined(TEST)
int value = left + +right; // remove this comment
const char *message = "spaces ; // stay";
char newline = '\\n';
value = value - -right;
#endif
'''
        compacted = export_submission.compact_cpp_code(source)
        self.assertEqual(compacted, (
            "#if defined(TEST)\n"
            'int value=left+ +right;const char*message="spaces ; // stay";'
            "char newline='\\n';value=value- -right;\n"
            "#endif\n"
        ))

    def test_cpp_compaction_rejects_unterminated_constructs(self):
        with self.assertRaisesRegex(ValueError, "unterminated C\\+\\+ literal"):
            export_submission.compact_cpp_code('const char *value = "broken;')
        with self.assertRaisesRegex(ValueError, "unterminated C\\+\\+ block comment"):
            export_submission.compact_cpp_code("int value; /* broken")

    def test_private_identifier_minification_preserves_literals(self):
        source = 'int config_; const char *text = "config_ nodes_"; int nodes_;\n'
        self.assertEqual(
            export_submission.minify_private_identifiers(source),
            'int zA; const char *text = "config_ nodes_"; int zB;\n',
        )

    def test_truncated_search_dead_ends_close_before_single_pass_selection(self):
        source = (HERE / "engine.cpp").read_text()
        self.assertIn("return solved || all_children_closed;", source)
        self.assertIn(
            "node.closed = traversal_closed(node.solved, all_closed);", source
        )
        self.assertIn("BFM open tree has no selectable descendant", source)
        optimized = source[
            source.index("#else\n    const double parent_log"):
            source.index("#endif\n  }\n\n  std::optional", source.index(
                "#else\n    const double parent_log"
            ))
        ]
        self.assertNotIn("while (true)", optimized)

    def test_exporter_rejects_name_scale_code_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            wrong_name = self.runtime(root, 8, 8, name="capacity-12x8")
            with self.assertRaisesRegex(ValueError, "architecture name"):
                export_model.render_header(wrong_name)
            noncanonical_scale = self.runtime(root, 8, 8, scale=0.1)
            with self.assertRaisesRegex(ValueError, "float32"):
                export_model.render_header(noncanonical_scale)
            forbidden = self.runtime(
                root, 8, 8, mutate_payload=lambda payload: payload.__setitem__(0, 4))
            with self.assertRaisesRegex(ValueError, "code 100"):
                export_model.render_header(forbidden)
            padding = self.runtime(
                root, 12, 8,
                mutate_payload=lambda payload: payload.__setitem__(-1, 0x10))
            with self.assertRaisesRegex(ValueError, "padding"):
                export_model.render_header(padding)

    def test_content_address_and_body_hash_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            runtime = self.runtime(root, 8, 8)
            alias = root / "runtime.json"
            alias.write_bytes(runtime.read_bytes())
            with self.assertRaisesRegex(ValueError, "content-addressed"):
                export_model.render_header(alias)
            payload = bytearray(runtime.read_bytes())
            payload[-2] = ord("0") if payload[-2] != ord("0") else ord("1")
            raw = bytes(payload)
            tampered = root / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
            tampered.write_bytes(raw)
            with self.assertRaises((ValueError, UnicodeDecodeError)):
                export_model.render_header(tampered)

    def test_channel_source_reserve_including_extreme_canonical_float32_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            for raw_bits in (1, 0x00800001, 0x3dcccccd, 0x7f7fffff):
                scale = struct.unpack("<f", struct.pack("<I", raw_bits))[0]
                path = self.runtime(pathlib.Path(directory), 12, 8, scale=scale, channel=True)
                header, metadata = export_model.render_header(path)
                _, source = export_submission.render(model_header=header)
                with self.subTest(scale=scale):
                    self.assertEqual(metadata["runtime_schema"], export_model.CHANNEL_RUNTIME_SCHEMA)
                    self.assertIn(b"std::array<float, 12> kScaleOne", header)
                    self.assertLessEqual(len(source), 93_000)
                    self.assertTrue(source.isascii())
                    self.assertEqual(export_submission.compact_cpp_code(source.decode()).encode(), source)
                    self.assertNotIn(b"model::kBootstrapZero", source)

    def test_source_region_selection_is_nested_and_rejects_unbalanced_markers(self):
        source = ("common\n// COMPACT_RUNTIME_V1_BEGIN\nlegacy\n// COMPACT_RUNTIME_V1_END\n"
                  "// COMPACT_RUNTIME_V2_BEGIN\nchannel\n// COMPACT_RUNTIME_NATIVE_BEGIN\n"
                  "dispatch\n// COMPACT_RUNTIME_NATIVE_END\n// COMPACT_RUNTIME_V2_END\n")
        self.assertEqual(export_submission.runtime_source_region(source, export_model.RUNTIME_SCHEMA),
                         "common\nlegacy\n")
        self.assertEqual(export_submission.runtime_source_region(source, export_model.CHANNEL_RUNTIME_SCHEMA),
                         "common\nchannel\n")
        for invalid_source in ("// COMPACT_RUNTIME_V1_END\n", "// COMPACT_RUNTIME_V2_BEGIN\n",
                               "// COMPACT_RUNTIME_V1_BEGIN\n// COMPACT_RUNTIME_V2_END\n"):
            with self.subTest(source=invalid_source), self.assertRaises(ValueError):
                export_submission.runtime_source_region(invalid_source, export_model.RUNTIME_SCHEMA)

    def test_channel_effective_float32_weight_overflow_is_rejected_per_output_axis(self):
        maximum = struct.unpack("<f", struct.pack("<I", 0x7f7fffff))[0]
        with tempfile.TemporaryDirectory() as directory:
            for name, index, output in (("w1", 6301 * 12 - 1, 11),
                                         ("w2", 6301 * 12 + 12 * 8 - 1, 7),
                                         ("w3", 6301 * 12 + 12 * 8 + 7, 0)):
                for code in (-3, -2, -1, 0, 1, 2, 3):
                    def packed(payload):
                        for bit in range(3):
                            position = index * 3 + bit
                            payload[position // 8] |= ((code >> bit) & 1) << (position % 8)
                    def scaled(body):
                        body["quantization"]["scales"][name][output] = maximum
                    path = self.runtime(pathlib.Path(directory), 12, 8, channel=True,
                                        mutate_payload=packed, mutate_body=scaled)
                    with self.subTest(layer=name, code=code):
                        if abs(code) >= 2:
                            with self.assertRaisesRegex(ValueError, "effective weight"):
                                export_model.validate_runtime(path)
                        else:
                            export_model.validate_runtime(path)
            # The v2 consistency rule must not alter historical v1 acceptance.
            path = self.runtime(pathlib.Path(directory), 12, 8, scale=maximum,
                                mutate_payload=lambda payload: payload.__setitem__(0, 3))
            export_model.validate_runtime(path)

    def test_model_header_schema_and_native_macro_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            for channel in (False, True):
                runtime = self.runtime(pathlib.Path(directory), 12, 8, channel=channel)
                header, _ = export_model.render_header(runtime)
                malformed = (header.replace(b"#define COMPACT_VALUE_BFM_CHANNEL_MODEL_V2 1", b"")
                             if channel else b"#define COMPACT_VALUE_BFM_CHANNEL_MODEL_V2 1\n" + header)
                with self.subTest(channel=channel), self.assertRaisesRegex(ValueError, "mixes"):
                    export_submission.render(model_header=malformed)

    def test_content_addressed_duplicate_keys_are_still_noncanonical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for channel in (False, True):
                runtime = self.runtime(root, 12, 8, channel=channel)
                document = json.loads(runtime.read_bytes())
                # Parsing would discard this duplicate without changing the body.
                raw = runtime.read_bytes().replace(b'{', b'{"schema":' +
                    json.dumps(document["schema"]).encode() + b',', 1)
                path = root / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
                path.write_bytes(raw)
                with self.subTest(channel=channel), self.assertRaisesRegex(ValueError, "canonical"):
                    export_model.render_header(path)


if __name__ == "__main__":
    unittest.main()
