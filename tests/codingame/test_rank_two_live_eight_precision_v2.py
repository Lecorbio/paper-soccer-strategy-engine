"""Matched precision must share numerical ancestry and all evaluation inputs."""
import unittest
from tools import rank_two_live_eight_precision_v2 as precision


class PrecisionPlans(unittest.TestCase):
    def test_only_precision_and_output_change_between_arms(self):
        original = dict(seed=2026101017, phase_mode='float', weight_bits=4,
                        inputs={}, resource_budget=dict(workers=1, numerical_threads=1),
                        corpus=dict(path='corpus', sha256='corpus'),
                        fresh_validation=dict(path='validation', sha256='validation'),
                        live_roots=['new-root'], qat_maximum=32)
        parent = dict(path='parent', sha256='parent')
        selection = dict(path='selected', sha256='selected')
        plans = precision.paired_plans(original, parent, selection, '/tmp/precision-unit')
        for bits in (4, 6):
            self.assertEqual(plans[bits]['float_parent'], parent)
            self.assertEqual(plans[bits]['inputs']['common_float_selection'], selection)
            self.assertEqual(plans[bits]['phase_mode'], 'qat')
            self.assertFalse(plans[bits]['duplicate_float_work'])
            self.assertEqual(plans[bits]['resource_budget']['numerical_threads'], 1)
        left = {k:v for k,v in plans[4].items() if k not in ('weight_bits', 'output')}
        right = {k:v for k,v in plans[6].items() if k not in ('weight_bits', 'output')}
        self.assertEqual(left, right)
        self.assertEqual(original['phase_mode'], 'float')
        self.assertEqual(original['resource_budget']['workers'], 1)
        plans[4]['live_roots'].append('other-root')
        self.assertEqual(plans[6]['live_roots'], ['new-root'])


if __name__ == '__main__':
    unittest.main()
