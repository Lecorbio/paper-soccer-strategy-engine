import copy
import unittest

from tools import rank_two_live_eight_data_v4 as data


class BoundedLiveDataTest(unittest.TestCase):
    def rows(self):
        return [dict(root_group_id='root-' + str(root), position_id='p-' + str(root) + '-' + str(index),
                     canonical_state='state-' + str(root) + '-' + str(index),
                     canonical_features='features-' + str(root) + '-' + str(index),
                     split='train', outcome_as_target=False, winner=index % 2)
                for root in range(4) for index in range(8)]

    def test_order_and_outcomes_do_not_change_sample(self):
        rows = self.rows()
        expected = data.bounded_parents(rows, 123, 2, 3)
        changed = copy.deepcopy(list(reversed(rows)))
        for row in changed:
            row['winner'] = 1 - row['winner']
        actual = data.bounded_parents(changed, 123, 2, 3)
        self.assertEqual([r['position_id'] for r in expected], [r['position_id'] for r in actual])
        self.assertEqual(6, len(actual))
        self.assertEqual(2, len({r['root_group_id'] for r in actual}))

    def test_canonical_aliases_do_not_add_mass(self):
        rows = self.rows()
        alias = dict(rows[0], position_id='zz-alias')
        self.assertEqual(data.bounded_parents(rows, 123), data.bounded_parents(rows + [alias], 123))

    def test_held_out_rows_are_rejected(self):
        rows = self.rows()
        rows[0]['split'] = 'validation'
        with self.assertRaises(PermissionError):
            data.bounded_parents(rows, 123)

    def test_previous_live_roots_are_retained(self):
        self.assertEqual(['old-live', 'shared', 'new-live'], sorted(
            data.merge_live_roots(['old-live', 'shared'], {'shared', 'new-live'},
                                 {'old-live', 'shared', 'new-live', 'historical'}),
            key=lambda x: {'old-live': 0, 'shared': 1, 'new-live': 2}[x]))
        with self.assertRaises(ValueError):
            data.merge_live_roots(['missing'], {'new'}, {'new'})

    def test_combined_provenance_keeps_all_eligible_live_roots(self):
        old = dict(passed=True, roots=['old', 'shared'], only_predesignated_new_exploration=True,
                   protected_games_read=False)
        new = dict(passed=True, roots=['shared', 'new', 'unlabeled'],
                   only_predesignated_new_exploration=True, protected_games_read=False)
        result = data.combined_live_provenance(['old', 'shared'], {'shared', 'new'}, old, new)
        self.assertEqual(['new', 'old', 'shared'], result['roots'])
        self.assertNotIn('unlabeled', result['roots'])
        with self.assertRaises(ValueError):
            data.combined_live_provenance(['unknown'], {'new'}, old, new)
        with self.assertRaises(PermissionError):
            data.combined_live_provenance(['old'], {'new'}, old, dict(new, protected_games_read=True))



if __name__ == '__main__':
    unittest.main()
