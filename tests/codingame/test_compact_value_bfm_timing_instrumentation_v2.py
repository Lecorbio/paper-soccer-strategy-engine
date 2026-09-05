import copy
import hashlib
from pathlib import Path
import unittest

from tools import compact_value_bfm_timing_instrumentation_v2 as timing

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'submissions/codingame/bots/compact_value_bfm/submission.cpp'


def original_probe(payload):
    return {'schema': 'papersoccer.compact-engine-version-probe.v2', 'mode': 'fixed',
            'payload_sha256': payload, 'all_actions_legal': True, 'all_root_actions_legal': True,
            'actual_model_full_delta_bit_exact': True, 'all_root_actions_full_delta_bit_exact': True,
            'rows': [{'id': 'fixture', 'action': '02', 'milliseconds': .000005,
                      'fixed_trace': '1:0;02:2:123:123:1:0:0:0', 'nodes': 20,
                      'expansions': 2, 'generated_successors': 19, 'evaluated_successors': 14}]}


class SourceTransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_bytes()
        cls.digest = hashlib.sha256(cls.source).hexdigest()
        cls.derivative, cls.manifest = timing.instrument_source(cls.source, cls.digest)

    def test_actual_source_is_recoverable_and_manifest_is_deterministic(self):
        self.assertEqual((self.derivative, self.manifest),
                         timing.instrument_source(self.source, self.digest))
        recovered = self.derivative[self.manifest['runtime_bytes']:]
        for item in self.manifest['anchors']:
            recovered = recovered.replace(item['addition'].encode('ascii'), b'', 1)
        self.assertEqual(recovered, self.source)
        self.assertEqual(SOURCE.read_bytes(), self.source)
        self.assertEqual(self.manifest['instrumented_source_sha256'], timing.sha(self.derivative))
        self.assertTrue(self.manifest['attribution_only'])
        self.assertFalse(self.manifest['eligible_for_speed_gate'])
        self.assertEqual(len(self.manifest['anchors']), 12)
        self.assertEqual({row['name'] for row in self.manifest['categories']}, set(timing.CATEGORIES))
        for item in self.manifest['anchors']:
            self.assertEqual(item['count'], 1)
            self.assertEqual(len(item['original_function_sha256']), 64)

    def test_wrong_identity_duplicate_anchor_and_unsupported_body_fail_closed(self):
        mutations = [self.source + b'\n',
                     self.source + b'\n// ' + timing.ANCHORS[0][1].encode(),
                     self.source.replace(timing.ANCHORS[0][1].encode(),
                                         b'SparseFeatures renamed(const State&state,std::uint8_t perspective){'),
                     self.source.replace(b'kHiddenOne=12;', b'kHiddenOne=8;')]
        with self.assertRaisesRegex(ValueError, 'exact source SHA256'):
            timing.instrument_source(mutations[0], self.digest)
        for mutated in mutations[1:]:
            with self.subTest(digest=timing.sha(mutated)):
                with self.assertRaises(ValueError):
                    timing.instrument_source(mutated, timing.sha(mutated))
        with self.assertRaisesRegex(ValueError, 'already contains'):
            timing.instrument_source(self.derivative, timing.sha(self.derivative))

    def test_signature_in_a_comment_cannot_stand_in_for_missing_code(self):
        anchor = timing.ANCHORS[0][1].encode()
        changed = self.source.replace(anchor, b'/*' + anchor + b'*/' + anchor.replace(b'active_features', b'wrong'))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            timing.instrument_source(changed, timing.sha(changed))

    def test_payload_identity_is_checked_independently_of_expected_source_digest(self):
        payload = self.manifest['payload_sha256'].encode()
        changed = self.source.replace(b'kPayloadSha256="' + payload,
                                      b'kPayloadSha256="' + b'0' * 64)
        with self.assertRaisesRegex(ValueError, 'payload identity mismatch'):
            timing.instrument_source(changed, timing.sha(changed))

    def test_macro_variants_preserve_the_same_timing_scopes_and_payload(self):
        flags = (b'#define COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT\n'
                 b'#define COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT\n')
        _, variant = timing.instrument_source(flags + self.source, timing.sha(flags + self.source))
        self.assertEqual(variant['payload_sha256'], self.manifest['payload_sha256'])
        self.assertEqual([(row['anchor'], row['original_function_sha256']) for row in variant['anchors']],
                         [(row['anchor'], row['original_function_sha256']) for row in self.manifest['anchors']])

    def test_literal_mask_handles_digit_separators_and_ignores_quoted_braces(self):
        text = 'int count=4\'000; char brace=\'}\'; /* { */ const char* s="{\\\"}"; {}'
        mask = timing.code_mask(text)
        self.assertEqual(len(text), len(mask))
        self.assertIn("4'000", mask)
        self.assertEqual(mask.count('{'), 1)
        self.assertEqual(mask.count('}'), 1)
        for bad in ('/* never closed', '"never closed', 'R"(raw)"'):
            with self.assertRaises(ValueError):
                timing.code_mask(bad)


class TimingParityTests(unittest.TestCase):
    def setUp(self):
        source = SOURCE.read_bytes()
        _, self.manifest = timing.instrument_source(source, timing.sha(source))
        self.original = original_probe(self.manifest['payload_sha256'])
        self.timed = copy.deepcopy(self.original)
        self.timed.update(schema=timing.PROBE_SCHEMA, source_sha256=self.manifest['source_sha256'],
                          instrumentation_version=timing.VERSION, attribution_only=True,
                          eligible_for_speed_gate=False)
        self.row = self.timed['rows'][0]
        self.row.update(category_exclusive_ns={name: 1 for name in timing.CATEGORIES},
                        category_calls={name: 1 for name in timing.CATEGORIES}, total_search_ns=7,
                        category_sum_ns=7, reconciled=True, milliseconds=.000007)

    def test_fixed_trace_and_exclusive_totals_pass_despite_measured_overhead(self):
        result = timing.validate_probe_parity(self.original, self.timed, self.manifest)
        self.assertTrue(result['fixed_trace_bit_exact'])
        self.assertTrue(result['all_category_totals_reconciled'])
        self.assertFalse(result['eligible_for_speed_gate'])

    def test_changed_search_behavior_and_model_are_rejected(self):
        for key, value in [('action', '4'), ('fixed_trace', 'different'), ('nodes', 21),
                           ('id', 'different'), ('generated_successors', 20)]:
            changed = copy.deepcopy(self.timed)
            changed['rows'][0][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'changed fixed-work trace'):
                timing.validate_probe_parity(self.original, changed, self.manifest)
        for key, value in [('payload_sha256', '0' * 64), ('mode', 'clock'),
                           ('all_root_actions_legal', False), ('eligible_for_speed_gate', True),
                           ('source_sha256', '0' * 64), ('instrumentation_version', 'unknown')]:
            changed = copy.deepcopy(self.timed)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                timing.validate_probe_parity(self.original, changed, self.manifest)

    def test_missing_overlapping_and_malformed_timers_are_rejected(self):
        mutations = [lambda row: row['category_exclusive_ns'].pop('first_layer'),
                     lambda row: row['category_exclusive_ns'].update(first_layer=2),
                     lambda row: row['category_calls'].update(first_layer=0),
                     lambda row: row['category_calls'].update(first_layer=True),
                     lambda row: row.update(category_sum_ns=8),
                     lambda row: row.update(total_search_ns=True),
                     lambda row: row.update(milliseconds=.000008),
                     lambda row: row.update(milliseconds=float('nan')),
                     lambda row: row.update(reconciled=False)]
        for number, mutation in enumerate(mutations):
            changed = copy.deepcopy(self.timed)
            mutation(changed['rows'][0])
            with self.subTest(number=number), self.assertRaises(ValueError):
                timing.validate_probe_parity(self.original, changed, self.manifest)

    def test_missing_or_repeated_roots_and_zero_counters_do_not_fake_measurement(self):
        changed = copy.deepcopy(self.timed)
        changed['rows'] = []
        with self.assertRaises(ValueError):
            timing.validate_probe_parity(self.original, changed, self.manifest)
        original, timed = copy.deepcopy(self.original), copy.deepcopy(self.timed)
        original['rows'] *= 2
        timed['rows'] *= 2
        with self.assertRaisesRegex(ValueError, 'identity/action/trace'):
            timing.validate_probe_parity(original, timed, self.manifest)
        for raw in (original, timed):
            raw['rows'] = raw['rows'][:1]
            raw['rows'][0]['nodes'] = False
        with self.assertRaisesRegex(ValueError, 'counters'):
            timing.validate_probe_parity(original, timed, self.manifest)


if __name__ == '__main__':
    unittest.main()
