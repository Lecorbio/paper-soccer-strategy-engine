import copy
import unittest

from tools.rank_two_live_eight_validation_sampling_v1 import early_parents


class EarlyValidationSamplingTest(unittest.TestCase):
    def rows(self):
        return [dict(root_group_id='root-' + str(root), position_id='p-' + str(root) + '-' + str(index),
                     canonical_features='features-' + str(index), edges=8 + index,
                     split='validation', winner=index % 2)
                for root in range(128) for index in range(8)]

    def test_frozen_quota_and_early_cap(self):
        selected = early_parents(self.rows())
        self.assertEqual(256, len(selected))
        self.assertEqual(128, len({row['root_group_id'] for row in selected}))
        self.assertTrue(all(row['edges'] <= 12 for row in selected))

    def test_results_and_input_order_do_not_affect_selection(self):
        rows = self.rows()
        expected = early_parents(rows)
        changed = copy.deepcopy(list(reversed(rows)))
        for row in changed:
            row['winner'] = 1 - row['winner']
        self.assertEqual([row['position_id'] for row in expected],
                         [row['position_id'] for row in early_parents(changed)])

    def test_canonical_alias_is_not_extra_evidence(self):
        rows = self.rows()
        self.assertEqual(early_parents(rows), early_parents(rows + [dict(rows[0], position_id='zz-alias')]))

    def test_missing_early_root_is_not_replaced(self):
        rows = self.rows()
        for row in rows:
            if row['root_group_id'] == 'root-0':
                row['edges'] = 20
        with self.assertRaises(ValueError):
            early_parents(rows)

    def test_train_and_cap_expansion_are_rejected(self):
        rows = self.rows()
        rows[0]['split'] = 'train'
        with self.assertRaises(PermissionError):
            early_parents(rows)
        with self.assertRaises(ValueError):
            early_parents(self.rows(), 3)


if __name__ == '__main__':
    unittest.main()
