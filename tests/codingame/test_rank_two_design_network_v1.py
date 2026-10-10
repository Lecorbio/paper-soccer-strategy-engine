import base64
import copy
import hashlib
from pathlib import Path
import re
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import rank_two_design_network_v1 as net
import rank_two_network32_v2 as historical


def sealed(runtime):
    body = {k: v for k, v in runtime.items() if k != 'body_sha256'}
    return {**body, 'body_sha256': hashlib.sha256(net.canonical(body)).hexdigest()}


def output_pre(values, scales, dimensions, active):
    """Independent scalar float32 arithmetic; no native process or corpus input."""
    inputs, h1, h2, _ = dimensions
    f32 = net.codec.f32
    first = []
    for j in range(h1):
        x = f32(sum(values[i * h1 + j] for i in active) * scales[0])
        first.append(f32(x * x) if x >= 0 else f32(f32(.01) * x))
    second = []
    for j in range(h2):
        x = f32(0)
        for i in range(h1):
            term = f32(f32(first[i] * scales[1]) * values[inputs * h1 + i * h2 + j])
            x = f32(x + term)
        second.append(x if x >= 0 else f32(f32(.01) * x))
    result = f32(0)
    for j in range(h2):
        result = f32(result + f32(f32(second[j] * scales[2]) * values[inputs * h1 + h1 * h2 + j]))
    return struct.pack('<f', result)


class DesignNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The public fixture has the exact historical A180 bytes. Keep the
        # producer's private campaign binding unchanged outside this test.
        template = ROOT / 'submissions/codingame/research/focused-network/template.cpp'
        for attribute, value in (
            ('TEMPLATE_PATH', template),
            ('TEMPLATE_REF', {'path': str(template), 'sha256': net.TEMPLATE_SHA256}),
        ):
            binding = patch.object(net, attribute, value)
            binding.start()
            cls.addClassCleanup(binding.stop)
        cls.base_source, cls.old, cls.scales = historical.baseline()
        cls.runtimes = {name: net.initialize(name) for name in net.PROFILES}
        cls.template = net.TEMPLATE_PATH.read_text()

    def test_dd8_exact_subnetwork_mapping_without_preservation_claim(self):
        runtime = self.runtimes['dd8']
        _, values, scales = net.validate(runtime)
        self.assertEqual(runtime['architecture'], [6301, 8, 8, 1])
        self.assertEqual(len(values), 50480)
        for i in range(6301):
            self.assertEqual(values[i * 8:(i + 1) * 8], self.old[i * 12:i * 12 + 8])
        self.assertEqual(values[6301 * 8:6301 * 8 + 64], self.old[6301 * 12:6301 * 12 + 64])
        self.assertEqual(values[-8:], self.old[-8:])
        self.assertEqual(scales, self.scales)
        self.assertFalse(runtime['lineage']['function_preserving_initialization'])

    def test_dd12wide_preserves_paths_and_scalar_output_bits(self):
        runtime = self.runtimes['dd12wide']
        _, values, scales = net.validate(runtime)
        self.assertEqual(runtime['architecture'], [6301, 12, 16, 1])
        self.assertEqual(len(values), 75820)
        self.assertEqual(values[:6301 * 12], self.old[:6301 * 12])
        extras = []
        for i in range(12):
            start = 6301 * 12 + i * 16
            self.assertEqual(values[start:start + 8], self.old[6301 * 12 + i * 8:6301 * 12 + (i + 1) * 8])
            extras.extend(values[start + 8:start + 16])
        self.assertEqual(set(extras), {-1, 0, 1})
        self.assertEqual(values[-16:-8], self.old[-8:])
        self.assertEqual(values[-8:], [0] * 8)
        self.assertTrue(runtime['lineage']['function_preserving_initialization'])
        for category, edges in ((0, []), (56, list(range(316))), (17, [0, 14, 57])):
            active = edges + [316 + v * 57 + category for v in range(105)]
            self.assertEqual(output_pre(values, scales, runtime['architecture'], active),
                             output_pre(self.old, self.scales, [6301, 12, 8, 1], active))

    def test_determinism_four_bit_roundtrip_and_independent_metadata(self):
        for profile, runtime in self.runtimes.items():
            self.assertEqual(runtime, net.initialize(profile))
            _, values, _ = net.validate(runtime)
            self.assertEqual(net.codec.unpack(base64.b64decode(runtime['payload_base64']), len(values), 4), values)
            self.assertFalse(runtime['trained'])
        lineage = {'nested': {'value': 1}}
        runtime = net.document('dd8', net.validate(self.runtimes['dd8'])[1], self.scales, lineage=lineage)
        lineage['nested']['value'] = 2
        self.assertEqual(runtime['lineage']['nested']['value'], 1)
        with self.assertRaises(ValueError):
            net.initialize('dd12wide', seed=net.SEED + 1)

    def test_historical_d32_reuse_retains_every_code_and_scale(self):
        old = historical.initialize('d32')
        _, old_values, old_scales = historical.validate(old)
        self.assertEqual(net.validate(self.runtimes['d32'])[1:], (old_values, old_scales))
        trained = historical.document('d32', old_values, old_scales, trained=True, lineage={'epoch': 4})
        reused = net.reuse_historical_d32(trained)
        self.assertEqual(net.validate(reused)[1:], (old_values, old_scales))
        self.assertTrue(reused['trained'])
        self.assertEqual(reused['lineage']['historical_runtime_body_sha256'], trained['body_sha256'])
        self.assertFalse(reused['lineage']['new_training_performed'])
        with self.assertRaises(ValueError):
            net.reuse_historical_d32(historical.initialize('dd12'))

    def test_wrong_contracts_and_rehashed_tampering_are_rejected(self):
        valid = self.runtimes['dd8']
        _, values, _ = net.validate(valid)
        with self.assertRaises(ValueError):
            net.document('dd8', values[:-1], self.scales)
        with self.assertRaises(ValueError):
            net.document('dd8', [-8] + values[1:], self.scales)
        changes = [('schema', historical.SCHEMA), ('architecture', [6301, 12, 8, 1]),
                   ('feature_schema', 'wrong'), ('weight_bits', 3), ('biases', True),
                   ('trained', 1), ('source_refs', {}), ('scales', [x * 2 for x in self.scales]),
                   ('payload_sha256', '0' * 64)]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                net.validate(sealed(dict(valid, **{key: value})))
        with self.assertRaises(ValueError):
            net.validate(dict(valid, body_sha256='0' * 64))
        with self.assertRaises(ValueError):
            net.validate(dict(valid, extra=True))
        payload = bytes([8]) + base64.b64decode(valid['payload_base64'])[1:]
        with self.assertRaises(ValueError):
            net.validate(sealed(dict(valid, payload_base64=base64.b64encode(payload).decode(),
                                     payload_sha256=hashlib.sha256(payload).hexdigest())))

    def test_degree_preserving_exports_leave_all_search_and_book_source_unchanged(self):
        profiles_before, base_before = copy.deepcopy(historical.PROFILES), historical.BASE
        marker = 'namespace papersoccer::turn_action_v2 {'
        for profile in ('dd8', 'dd12wide'):
            raw, report = net.export(self.runtimes[profile], net.TEMPLATE_REF, minify=False)
            source = raw.decode('ascii')
            self.assertEqual(source[source.index(marker):], self.template[self.template.index(marker):])
            marker_model = 'namespace papersoccer::turn_action_v2::learned_model {'
            self.assertEqual(source[:source.index(marker_model)], self.template[:self.template.index(marker_model)])
            self.assertIn('volatile float scaled=activated[i]*learned_model::kScaleTwo;', source)
            self.assertIn('kFirstSearchTimeMs = 990;', source)
            self.assertIn('kLaterSearchTimeMs = 180;', source)
            self.assertEqual(report['clocks_ms'], [990, 180])
            self.assertEqual(report['parent'], net.TEMPLATE_REF)
        self.assertEqual(historical.PROFILES, profiles_before)
        self.assertEqual(historical.BASE, base_before)

    def test_d32_projection_changes_only_feature_body_within_search(self):
        raw, _ = net.export(self.runtimes['d32'], net.TEMPLATE_REF, minify=False)
        source = raw.decode()
        marker = 'namespace papersoccer::turn_action_v2 {'
        after = 'EvaluationSnapshot make_evaluation_snapshot()'
        feature = 'learned_eval::Features learned_features()'
        self.assertEqual(source[source.index(marker):source.index(feature)],
                         self.template[self.template.index(marker):self.template.index(feature)])
        self.assertEqual(source[source.index(after):], self.template[self.template.index(after):])
        self.assertIn('316+canonical*8+category', source)
        self.assertIn('calculate_goal_turn_distances(mover);', source)
        self.assertIn('const std::clock_t startup_cpu = std::clock();', source)
        self.assertIn('SearchClock::now() >= *deadline_', source)

    def test_historical_clocks_are_explicit_measurement_only(self):
        runtime = self.runtimes['d32']
        with self.assertRaises(ValueError):
            net.export(runtime, net.TEMPLATE_REF, clocks=(550, 140))
        with self.assertRaises(ValueError):
            net.export(self.runtimes['dd8'], net.TEMPLATE_REF, clocks=(550, 140), measurement_only=True)
        with self.assertRaises(ValueError):
            net.export(runtime, net.TEMPLATE_REF, clocks=(990, 185), measurement_only=True)
        normal, _ = net.export(runtime, net.TEMPLATE_REF, minify=False)
        measured, report = net.export(runtime, net.TEMPLATE_REF, clocks=(550, 140), measurement_only=True, minify=False)
        expected = normal.replace(b'kFirstSearchTimeMs = 990;', b'kFirstSearchTimeMs = 550;').replace(
            b'kLaterSearchTimeMs = 180;', b'kLaterSearchTimeMs = 140;')
        self.assertEqual(measured, expected)
        self.assertEqual(report['clocks_ms'], [550, 140])
        self.assertTrue(report['measurement_only'])
        self.assertFalse(report['live_admitted'])

    def test_template_hash_and_reference_fail_closed(self):
        with self.assertRaises(ValueError):
            net.export(self.runtimes['dd8'], dict(net.TEMPLATE_REF, sha256='0' * 64))
        with self.assertRaises(ValueError):
            net.export(self.runtimes['dd8'], dict(net.TEMPLATE_REF, path=str(historical.BASE)))
        read_bytes = Path.read_bytes
        def corrupt(path):
            payload = read_bytes(path)
            return payload + b' ' if path == net.TEMPLATE_PATH else payload
        with patch.object(Path, 'read_bytes', corrupt), self.assertRaises(ValueError):
            net.export(self.runtimes['dd8'], net.TEMPLATE_REF)

    def test_minified_export_metadata_and_compressed_payload_match_source(self):
        for profile, runtime in self.runtimes.items():
            raw, report = net.export(runtime, net.TEMPLATE_REF)
            self.assertEqual(report['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(report['characters'], len(raw))
            self.assertTrue(report['deployable_size'])
            self.assertLess(len(raw), 100000)
            self.assertEqual(report['architecture'], runtime['architecture'])
            self.assertEqual(report['payload_sha256'], runtime['payload_sha256'])
            self.assertEqual(report['source_refs'], runtime['source_refs'])
            self.assertFalse(report['measurement_only'])
            self.assertFalse(report['native_parity_verified'])
            self.assertFalse(report['live_admitted'])
            source = raw.decode()
            packed = historical.PAYLOAD.search(source)[1]
            payload = base64.b64decode(''.join(re.findall(r'"([^"\n]*)"', packed)))
            lengths = list(map(int, re.search(r'kHuffmanLengths\{([^}]+)\}', source)[1].split(',')))
            values = net.validate(runtime)[1]
            self.assertEqual(net.codec.decompress(payload, len(values), 4, lengths), values)
            self.assertIn(runtime['body_sha256'], source)
            self.assertIn(net.SCHEMA, source)

    def test_oversize_export_reports_failure_without_resizing(self):
        runtime = self.runtimes['dd12wide']
        values = [i % 15 - 7 for i in range(75820)]
        dense = net.document('dd12wide', values, self.scales, trained=True)
        raw, report = net.export(dense, net.TEMPLATE_REF)
        self.assertGreaterEqual(len(raw), 100000)
        self.assertFalse(report['deployable_size'])
        self.assertEqual(report['failure'], 'source-size-limit')
        self.assertEqual(report['architecture'], runtime['architecture'])
        self.assertEqual(report['payload_sha256'], dense['payload_sha256'])
        self.assertFalse(report['live_admitted'])


if __name__ == '__main__':
    unittest.main()
