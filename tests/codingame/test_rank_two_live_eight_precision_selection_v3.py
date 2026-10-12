import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_precision_selection_v3 as selection
from tests.codingame import test_rank_two_live_eight_precision_selection_v1 as previous


class CompleteSelectionTests(previous.PrecisionSelectionTests):
    def setUp(self):
        self.original = previous.selection
        previous.selection = selection

    def tearDown(self):
        previous.selection = self.original

    def arms(self):
        arms = super().arms()
        for arm in arms:
            arm['validation_root_accounting'] = dict(frozen=128, evaluated=128, excluded=0,
                                                     comparable_early=128)
        return arms

    def test_specific_prior_quarantine_cannot_carry_forward(self):
        arms = self.arms()
        for arm in arms:
            arm['early_roots'] = 127
            arm['validation_root_accounting'] = dict(frozen=128, evaluated=127, excluded=1,
                                                     comparable_early=127)
        with self.assertRaises(ValueError):
            selection.select(arms)

    def test_complete_bank_cannot_use_a_favorable_subset(self):
        arms = self.arms()
        arms[1]['validation_root_accounting']['comparable_early'] = 127
        with self.assertRaises(ValueError):
            selection.select(arms)

    def test_primary_regret_precedes_lower_huber(self):
        arms = self.arms()
        arms[0].update(regret=.0692, huber=.0219, epoch=1)
        arms[1].update(regret=.0726, huber=.0191, epoch=6)
        self.assertEqual(selection.select(arms)['bits'], 4)


class RuntimeCompletionTests(unittest.TestCase):
    def fixture(self):
        receipt = dict(plan='numeric-plan', epoch=1, checkpoint=dict(optimizer_step=15992))
        completion = dict(passed=True, plan='runtime-plan', numeric_result='numeric-result',
                          selected='selected', selected_epoch=1, selected_optimizer_step=15992,
                          weight_bits=4, epochs=8, updates=5424)
        proof = dict(passed=True, numeric_plan='numeric-plan', runtime_plan='runtime-plan',
                     numeric_result='numeric-result', selected='selected', selected_epoch=1,
                     initial_optimizer_step=14636, final_optimizer_step=20060,
                     epochs=8, updates=5424, verified_epochs=list(range(8)),
                     calibration_trials=list(range(18)), calibration_root_contributions=12204,
                     all_root_journals_moments_native128_and_runtime_tensors_verified=True,
                     atomic_head_exact_last_epoch=True, calibration_TRAIN_only=True,
                     first_epoch_native_full_streamed_exact=True)
        runtime = dict(producer=selection.c.record(selection.qat_runtime.__file__),
                       numeric_plan='numeric-plan')
        numeric = dict(passed=True, phase='qat', weight_bits=4, selected='selected')
        return completion, proof, runtime, numeric, receipt

    def test_complete_runtime_accounting(self):
        self.assertEqual(selection.validate_completion(*self.fixture(), 4, 14636), 678)

    def test_rejects_incomplete_proofs(self):
        for key in ('atomic_head_exact_last_epoch', 'calibration_TRAIN_only',
                    'first_epoch_native_full_streamed_exact',
                    'all_root_journals_moments_native128_and_runtime_tensors_verified'):
            values = self.fixture()
            values[1][key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                selection.validate_completion(*values, 4, 14636)

    def test_rejects_mismatched_parent_plan_selection_or_counts(self):
        for index, key, value in [(1, 'initial_optimizer_step', 100),
                                  (1, 'numeric_plan', 'other'), (1, 'selected', 'other'),
                                  (1, 'calibration_root_contributions', 12203),
                                  (1, 'verified_epochs', list(range(7))),
                                  (0, 'selected_optimizer_step', 15993),
                                  (3, 'weight_bits', 6)]:
            values = copy.deepcopy(self.fixture())
            values[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                selection.validate_completion(*values, 4, 14636)


class RetentionTests(unittest.TestCase):
    def fixture(self):
        epoch = dict(passed=True, full_optimizer_moments_verified=True,
                     tensor_and_export_verified=True, updates=2, journal_receipts=['j1', 'j2'],
                     validation_roots=128, receipt='selected', native='native', runtime='runtime',
                     source='source', root_order='order', checkpoint='checkpoint')
        trial = dict(passed=True, TRAIN_only=True, weighted_loss_exact_sum=True,
                     root_contributions=['r1', 'r2'], roots=2, recipe='recipe', result='result')
        return dict(verified_epochs=['epoch'], calibration_trials=['trial']), epoch, trial

    def test_hashes_journals_and_calibration_contributions(self):
        proof, epoch, trial = self.fixture()
        with patch.object(selection.c, 'verify', side_effect=lambda value: value) as verify:
            with patch.object(selection.c, 'read', side_effect=lambda value: epoch if value == 'epoch' else trial):
                selection.verify_retention(proof, 'selected', 2)
        for reference in ('checkpoint', 'j1', 'j2', 'r1', 'r2'):
            verify.assert_any_call(reference)

    def test_incomplete_retention_is_rejected(self):
        proof, epoch, trial = self.fixture()
        epoch['journal_receipts'].pop()
        with patch.object(selection.c, 'verify', side_effect=lambda value: value):
            with patch.object(selection.c, 'read', return_value=epoch):
                with self.assertRaises(ValueError):
                    selection.verify_retention(proof, 'selected', 2)


if __name__ == '__main__':
    unittest.main()
