import unittest

from tools import rank_two_live_eight_admission_v2 as admission


class ContinuationAdmissionTests(unittest.TestCase):
    def responses(self):
        return dict(passed=True, own_failures=0, all_spent_cases_archived=True,
            isolated_production_games=16, highest_safe_common_clocks_ms=[990, 180],
            source_bound_models=[dict(model=dict(profile='dd8', clocks_ms=clock))
                                 for clock in admission.c.PLAN['clock_ladder']])

    def test_selects_exact_certified_clock(self):
        row = admission.selected_model(self.responses(), [990, 180])
        self.assertEqual(row['model']['clocks_ms'], [990, 180])

    def test_rejects_failure_missing_case_or_stale_ladder(self):
        for field, value in [('own_failures', 1), ('isolated_production_games', 15),
                             ('all_spent_cases_archived', False), ('highest_safe_common_clocks_ms', [900, 170])]:
            with self.subTest(field=field):
                proof = self.responses()
                proof[field] = value
                with self.assertRaises(PermissionError):
                    admission.selected_model(proof, [990, 180])
        proof = self.responses()
        proof['source_bound_models'] = proof['source_bound_models'][:-1]
        with self.assertRaises(PermissionError):
            admission.selected_model(proof, [990, 180])


if __name__ == '__main__':
    unittest.main()
