import copy
import unittest

from tools import rank_two_live_eight_precision_selection_v1 as selection


class PrecisionSelectionTests(unittest.TestCase):
    def arms(self):
        shared = dict(common_float_parent='full-optimizer-parent', seed=17, corpus='corpus',
            validation='fresh-bank', live_roots=['old-live', 'new-live'],
            live_training_provenance='closed-provenance', selection=selection.RULE,
            calibration_split='train', epochs=8, early_roots=128, huber=.02, epoch=2)
        return [dict(shared, bits=4, regret=.080, characters=74678),
                dict(shared, bits=6, regret=.078, characters=90456)]

    def test_native_regret_precedes_smaller_source(self):
        arms = self.arms()
        self.assertEqual(selection.select(arms)['bits'], 6)
        arms[1]['regret'] = arms[0]['regret']
        arms[0]['huber'] = .019
        self.assertEqual(selection.select(arms)['bits'], 4)

    def test_rejects_mismatched_parent_bank_or_provenance(self):
        for field in ('common_float_parent', 'validation', 'corpus', 'seed',
                      'live_training_provenance', 'live_roots'):
            with self.subTest(field=field):
                arms = copy.deepcopy(self.arms())
                arms[1][field] = 'different'
                with self.assertRaises(ValueError):
                    selection.select(arms)

    def test_rejects_incomplete_calibration_coverage_or_source(self):
        for field, value in [('epochs', 7), ('early_roots', 99), ('characters', 99001),
                             ('calibration_split', 'validation'), ('regret', float('nan'))]:
            with self.subTest(field=field):
                arms = self.arms()
                arms[1][field] = value
                with self.assertRaises(ValueError):
                    selection.select(arms)


if __name__ == '__main__':
    unittest.main()
