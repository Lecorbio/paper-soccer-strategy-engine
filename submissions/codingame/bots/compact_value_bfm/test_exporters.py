import base64
import hashlib
import pathlib
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
                mutate_payload=None) -> pathlib.Path:
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
                self.assertLessEqual(len(source), 95_000)
                sizes.append(len(source))
            self.assertGreater(sizes[-1], sizes[0])

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


if __name__ == "__main__":
    unittest.main()
