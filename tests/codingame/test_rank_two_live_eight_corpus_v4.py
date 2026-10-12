import copy
import unittest

from tools.rank_two_live_eight_corpus_v4 import ready_inputs, strict_manifest


class ContinuationAdmissionTests(unittest.TestCase):
    def fixture(self):
        keys = ('retained_train_corpus', 'live_teacher_plan', 'live_teacher_execution',
                'validation_teacher_plan', 'validation_teacher_execution')
        plan = {key: dict(path=key, sha256=key) for key in keys}
        plan['validation_admission_plan'] = dict(path='admitted-plan', sha256='frozen')
        admitted = {key: copy.deepcopy(plan[key]) for key in keys}
        admission = dict(passed=True, plan=plan['validation_admission_plan'])
        train = dict(passed=True, outcomes_as_targets=False,
            all_historical_validation_excluded=True, train_roots=3, live_roots=2,
            eligible_root_ids=['historical', 'old-live', 'new-live'],
            live_root_ids=['old-live', 'new-live'])
        fresh = dict(passed=True, roots=128, early_roots=100, training_eligible=False,
            outcomes_as_targets=False, protected_gameplay_reads=False,
            plan=plan['validation_teacher_plan'], execution=plan['validation_teacher_execution'])
        fresh['group_artifacts'] = [dict(root_group_id=f'r{i}', split='validation') for i in range(128)]
        return plan, {}, admitted, admission, train, fresh

    def test_complete_bound_cumulative_admission(self):
        self.assertIsNone(ready_inputs(*self.fixture()))

    def test_prior_quarantine_is_retained_without_becoming_current_gate(self):
        args = self.fixture()
        prior_gate = dict(path='old-approval', sha256='old')
        original = dict(validation_gate_approval=prior_gate,
                        validation_root_accounting=dict(frozen=128, evaluated=127, excluded=1))
        updated = strict_manifest(original, {'historical', 'old-live', 'new-live'}, args[5])
        self.assertEqual(updated['prior_validation_gate_retained'], prior_gate)
        self.assertNotIn('validation_gate_approval', updated)
        self.assertEqual(updated['validation_root_accounting']['evaluated'], 128)
        self.assertEqual(updated['validation_root_accounting']['excluded'], 0)
        self.assertEqual(len(updated['eligible_validation_root_group_ids']), 128)
        self.assertEqual(original['validation_root_accounting']['evaluated'], 127)

    def test_missing_root_or_new_quarantine_cannot_enter_strict_bank(self):
        for changed in ('missing', 'quarantine'):
            with self.subTest(changed=changed):
                args = self.fixture()
                if changed == 'missing':
                    args[5]['group_artifacts'].pop()
                else:
                    args[5]['gate_approval'] = dict(path='old', sha256='old')
                with self.assertRaises(PermissionError):
                    ready_inputs(*args)

    def test_unclosed_spent_exposures_prevent_training(self):
        args = self.fixture()
        args[1]['pending_exposures'] = ['spent-validation-game']
        with self.assertRaises(PermissionError):
            ready_inputs(*args)

    def test_replacement_bank_or_teacher_is_rejected(self):
        for key in ('retained_train_corpus', 'validation_teacher_execution'):
            args = self.fixture()
            args[2][key]['sha256'] = 'replacement'
            with self.assertRaises(ValueError):
                ready_inputs(*args)

    def test_early_coverage_and_retained_live_namespace_remain_required(self):
        args = self.fixture()
        args[5]['early_roots'] = 99
        with self.assertRaises(PermissionError):
            ready_inputs(*args)
        args = self.fixture()
        args[4]['eligible_root_ids'].remove('old-live')
        with self.assertRaises(ValueError):
            ready_inputs(*args)


if __name__ == '__main__':
    unittest.main()
