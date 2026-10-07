import unittest
import re
from unittest import mock

from tools import rank_two_design_bfm_v1 as bfm


class BfmExportTests(unittest.TestCase):
    def test_pair_changes_only_exploration(self):
        zero, zr = bfm.export(0)
        uct, ur = bfm.export(.95)
        self.assertEqual(zero.replace(b'kExploration = 0.0', b'kExploration = 0.95'), uct)
        self.assertEqual(zr['incumbent_payload_codes_sha256'], ur['incumbent_payload_codes_sha256'])
        for report in (zr, ur):
            self.assertTrue(report['deployable_size'])
            self.assertEqual(report['clocks_ms'], [990, 180])
            self.assertEqual(report['max_tree_nodes'], 32768)
            self.assertFalse(report['native_parity_verified'])
            self.assertFalse(report['live_admitted'])

    def test_payload_format_and_full_response_deadline_are_explicit(self):
        source, _ = bfm.export(0)
        text = source.decode()
        self.assertTrue(re.search(r'codec::decode\(compressed,\s*count,\s*4', text), 'four-bit decode absent')
        self.assertTrue('signed_value == -8' in text, 'reserved code guard absent')
        self.assertTrue('std::clock()' in text, 'startup accounting absent')
        self.assertLess(text.index('const auto started ='), text.index('int length ='))
        self.assertTrue('partial_paths % 10U == 0U' in text, 'hybrid traversal absent')
        self.assertTrue('winning == previous_winning' in text, 'proved-win priority absent')

    def test_unapproved_variant_or_changed_predecessor_rejects(self):
        for value in (True, .1, 1., float('nan')):
            with self.assertRaises(ValueError):
                bfm.export(value)
        with mock.patch.dict(bfm.BOUND, {'engine.hpp': '0' * 64}):
            with self.assertRaises(ValueError):
                bfm.export(0)


if __name__ == '__main__':
    unittest.main()
