import copy
import unittest

from tools import rank_two_live_eight_admission_v3 as admission
from tests.codingame.test_rank_two_live_eight_admission_v2 import ContinuationAdmissionTests


class CertifiedClockChoiceTests(unittest.TestCase):
    def fixture(self):
        responses = ContinuationAdmissionTests().responses()
        for row in responses['source_bound_models']:
            row['model'].update(source=dict(path='source', sha256=str(row['model']['clocks_ms'])),
                                runtime=dict(path='runtime', sha256='same-tensor'))
        model = responses['source_bound_models'][0]['model']
        choice = dict(approved_scope='human-approved-live-eight-clock-ladder',
            question='Does the exact8/8 tensor improve at550/140?',
            highest_safe_clocks_ms=[990, 180], chosen_clocks_ms=[550, 140],
            source=model['source'], runtime=model['runtime'], exact_tensor_unchanged=True,
            all_float_parameters_and_optimizer_moments_retained=True, training_updates=0)
        return responses, model, choice, dict(runtime=model['runtime'])

    def test_lower_certified_clock_preserves_highest_safe_proof(self):
        responses, model, _, _ = self.fixture()
        original = copy.deepcopy(responses)
        self.assertEqual(admission.selected_model(responses, [550, 140])['model'], model)
        self.assertEqual(responses, original)
        self.assertEqual(responses['highest_safe_common_clocks_ms'], [990, 180])

    def test_unproved_clock_failed_or_missing_coverage_is_rejected(self):
        responses, _, _, _ = self.fixture()
        with self.assertRaises(PermissionError):
            admission.selected_model(responses, [600, 150])
        for key, value in [('own_failures', 1), ('isolated_production_games', 15),
                           ('all_spent_cases_archived', False)]:
            changed = copy.deepcopy(responses); changed[key] = value
            with self.assertRaises(PermissionError):
                admission.selected_model(changed, [550, 140])

    def test_frozen_choice_requires_same_source_tensor_and_no_updates(self):
        responses, model, choice, native = self.fixture()
        admission.validate_choice(choice, responses, model, native)
        for key, value in [('training_updates', 1), ('exact_tensor_unchanged', False),
                           ('all_float_parameters_and_optimizer_moments_retained', False),
                           ('question', ''), ('runtime', dict(path='runtime', sha256='replacement')),
                           ('source', dict(path='other', sha256='new')),
                           ('highest_safe_clocks_ms', [550, 140])]:
            changed = copy.deepcopy(choice); changed[key] = value
            with self.assertRaises(PermissionError):
                admission.validate_choice(changed, responses, model, native)


if __name__ == '__main__':
    unittest.main()
