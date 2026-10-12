import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_validation_v6 as validation


class FrozenValidationRecipeTests(unittest.TestCase):
    def fixture(self):
        return dict(architecture=[6301, 8, 8, 1], outcomes_as_targets=False,
            fresh_validation_recipe=dict(colors=[0, 1], maximum_early_parents_per_root=2,
                minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
                outcomes_as_targets=False, root_depths=[8, 10], roots=128, seed=2026101237))

    def test_declared_recipe_passes(self):
        validation.verify_recipe(self.fixture())

    def test_previous_continuation_seed_is_rejected(self):
        declaration = self.fixture()
        declaration['fresh_validation_recipe']['seed'] = 2026101135
        with self.assertRaises(PermissionError):
            validation.verify_recipe(declaration)

    def test_current_declaration_cannot_be_substituted(self):
        ref = dict(path='current-declaration.json', sha256='bound')
        state = dict(live_eight_continuation_05_declaration=ref,
                     live_eight_known_windows={})
        with patch.object(validation.work, 'checked', return_value=state):
            validation.check_current(dict(continuation_declaration=ref), 'generate')
            with self.assertRaises(PermissionError):
                validation.check_current(dict(continuation_declaration={}), 'generate')

    def test_prior_continuation_seed_and_declaration_are_rejected(self):
        declaration = self.fixture()
        declaration['fresh_validation_recipe']['seed'] = 2026101235
        with self.assertRaises(PermissionError):
            validation.verify_recipe(declaration)
        ref = dict(path='prior-declaration.json', sha256='bound')
        state = dict(live_eight_continuation_04_declaration=ref, live_eight_known_windows={})
        with patch.object(validation.work, 'checked', return_value=state):
            with self.assertRaises(PermissionError):
                validation.check_current(dict(continuation_declaration=ref), 'generate')

    def test_seed_replacement_or_relaxed_coverage_is_rejected(self):
        for key, value in [('seed', 2026101127), ('roots', 127),
                           ('minimum_comparable_early_roots', 99),
                           ('maximum_early_parents_per_root', 3),
                           ('no_seed_or_root_replacement', False), ('colors', [0])]:
            with self.subTest(key=key):
                declaration = copy.deepcopy(self.fixture())
                declaration['fresh_validation_recipe'][key] = value
                with self.assertRaises(PermissionError):
                    validation.verify_recipe(declaration)

    def test_outcomes_and_other_architectures_are_rejected(self):
        for key, value in [('outcomes_as_targets', True), ('architecture', [6301, 12, 8, 1])]:
            with self.subTest(key=key):
                declaration = self.fixture()
                declaration[key] = value
                with self.assertRaises(PermissionError):
                    validation.verify_recipe(declaration)


if __name__ == '__main__':
    unittest.main()
