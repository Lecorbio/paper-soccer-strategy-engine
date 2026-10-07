import copy
import unittest
from tools import rank_two_design_training_v2 as training
import numpy as np


class ObservedTargetTests(unittest.TestCase):
    def setUp(self):
        self.roster = {'train_roots': ['root'], 'validation_roots': ['validation']}
        self.position = {'split': 'train', 'root_group_id': 'root', 'game_id': 'game',
                         'canonical_state': 'state', 'mover': 1, 'winner': 1}
        self.group = {**self.position, 'group_id': 'group', 'teacher_value': .4,
                      'parent_active': [0], 'exhaustive': True,
                      'successors': [{'successor_id': 'a', 'active': [1], 'teacher_value': .3, 'value_mover': 0},
                                     {'successor_id': 'b', 'active': [2], 'teacher_value': -.1, 'value_mover': 1}]}
        self.parents = training.ObservedParents([self.position], self.roster)

    def test_mixed_target_uses_parent_perspective_only(self):
        self.assertAlmostEqual(self.parents.target(self.group, .2), .52)
        lost = {**self.position, 'winner': 0}
        parents = training.ObservedParents([lost], self.roster)
        self.assertAlmostEqual(parents.target(self.group, .2), .12)
        active, targets, signs, spans, parents_at = training.rows([self.group], parents, .2, True)
        np.testing.assert_array_equal(targets[:2], np.asarray([.3, -.1], dtype=np.float32))
        self.assertEqual(parents_at, [2])
        self.assertAlmostEqual(float(targets[2]), .12)
        self.assertEqual([list(x) for x in active], [[1], [2], [0]])

    def test_proved_or_saturated_parent_targets_are_preserved(self):
        for value in (-1., 1.):
            self.assertEqual(self.parents.target({**self.group, 'teacher_value': value}, .2), value)
        proved = {**self.group, 'proof': {'solved': True, 'proven_winner': 1}}
        with self.assertRaises(ValueError):
            self.parents.target(proved, .2)
        self.assertEqual(self.parents.target({**proved, 'teacher_value': 1.}, .2), 1.)

    def test_unknown_or_protected_positions_fail_closed(self):
        with self.assertRaises(ValueError):
            self.parents.target({**self.group, 'canonical_state': 'unplayed'}, .2)
        with self.assertRaises(ValueError):
            training.ObservedParents([{**self.position, 'split': 'confirmation'}], self.roster)
        with self.assertRaises(ValueError):
            training.ObservedParents([self.position, {**self.position, 'winner': 0}], self.roster)
        with self.assertRaises(ValueError):
            self.parents.target(self.group, .3)

    def test_half_parent_half_successor_gradient(self):
        _, targets, signs, spans, parent_at = training.rows([self.group], self.parents, 0., True)
        pred = np.asarray([.15, -.05, .2], dtype=np.float32)
        loss, _, gradient, _ = training.objective([self.group], pred, targets, signs, spans, parent_at, 0., None)
        eps = 1e-3
        for i in range(3):
            plus, minus = pred.copy(), pred.copy()
            plus[i] += eps
            minus[i] -= eps
            lp = training.objective([self.group], plus, targets, signs, spans, parent_at, 0., None)[0]
            lm = training.objective([self.group], minus, targets, signs, spans, parent_at, 0., None)[0]
            self.assertAlmostEqual(float(gradient[i]), (lp-lm)/(2*eps), places=5)
        duplicated = [copy.deepcopy(self.group), copy.deepcopy(self.group)]
        _, t2, s2, spans2, parent2 = training.rows(duplicated, self.parents, 0., True)
        pred2 = np.asarray([.15, -.05, .15, -.05, .2, .2], dtype=np.float32)
        self.assertAlmostEqual(training.objective(duplicated, pred2, t2, s2, spans2, parent2, 0., None)[0], loss)

    @unittest.skipUnless((training.HISTORICAL / "producer.py").is_file(),
                         "requires retained local numerical-recipe artifacts")
    def test_frozen_numerical_support_loads_without_training(self):
        driver, helper = training.support()
        self.assertTrue(callable(driver.native_order_forward))
        self.assertTrue(callable(helper.native_metrics))

    def test_changed_data_or_template_rejects_before_training(self):
        approved = {key: {'path': key, 'sha256': 'hash'} for key in
                    ('corpus', 'roster', 'played_positions', 'deployment_template')}
        plan = {**approved, 'network_support': training.live.record(training.net.__file__)}
        training.validate_wave_inputs(plan, approved)
        for key in approved:
            with self.subTest(key=key):
                altered = {**plan, key: {'path': key, 'sha256': 'other'}}
                with self.assertRaises(ValueError):
                    training.validate_wave_inputs(altered, approved)


if __name__ == '__main__':
    unittest.main()
