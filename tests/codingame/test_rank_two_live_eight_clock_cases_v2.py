import copy
import unittest

from tools import rank_two_live_eight_clock_cases_v2 as cases


class ContinuationClockCasesTests(unittest.TestCase):
    def models(self):
        return [dict(model=dict(clocks_ms=clock, runtime='one-tensor', profile='dd8'))
                for clock in cases.c.PLAN['clock_ladder']]

    def test_complete_clock_color_offset_grid(self):
        grid = cases.case_grid(self.models())
        self.assertEqual(len(grid), 16)
        self.assertEqual({(tuple(row['model']['clocks_ms']), color, offset)
                          for row, color, offset in grid},
                         {(tuple(clock), color, offset) for clock in cases.c.PLAN['clock_ladder']
                          for color in (0, 1) for offset in (0, 3)})

    def test_rejects_duplicate_clock_or_changed_tensor(self):
        models = self.models()
        models[-1]['model']['clocks_ms'] = models[0]['model']['clocks_ms']
        with self.assertRaises(ValueError):
            cases.case_grid(models)
        models = copy.deepcopy(self.models())
        models[-1]['model']['runtime'] = 'changed-tensor'
        with self.assertRaises(ValueError):
            cases.case_grid(models)

    def test_next_case_uses_consecutive_completion_count(self):
        frozen = ['first', 'second']
        self.assertEqual(cases.next_case(frozen, []), 'first')
        self.assertEqual(cases.next_case(frozen, ['receipt']), 'second')
        self.assertIsNone(cases.next_case(frozen, ['receipt1', 'receipt2']))
        with self.assertRaises(ValueError):
            cases.next_case(frozen, ['r1', 'r2', 'r3'])


if __name__ == '__main__':
    unittest.main()
