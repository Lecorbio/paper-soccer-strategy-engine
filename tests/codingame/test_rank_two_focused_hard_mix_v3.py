"""Check difficult-position weighting without reading campaign gameplay."""
import copy
import unittest
from unittest import mock

import numpy as np

from tools import rank_two_focused_hard_mix_v3 as mixture
from tools import rank_two_focused_training_v12 as training
from tests.codingame.test_rank_two_focused_training_v1 import groups


class HardMixtureTests(unittest.TestCase):
    def test_hard_selection_rejects_partial_and_saturated_groups(self):
        rows = []
        for name, exhaustive, values in (
                ("hard", True, (.3, -.2, -.1)),
                ("partial", False, (.3, -.2, -.1)),
                ("saturated", True, (-.98, -.99, -1.))):
            group = groups()[0]
            group.update(group_id=name, exhaustive=exhaustive)
            for child, value in zip(group["successors"], values):
                child.update(value_mover=0, teacher_value=value)
            rows.append(group)
        recipe = dict(hard_mass=.5, teacher_regret_threshold=.05,
                      minimum_best_teacher_value=-.95, comparable_tolerance=1e-4)
        with mock.patch.object(mixture.retained, "forward",
                               return_value=(np.tile([0., 1., 0.], 3), None)):
            weights = mixture.weights_for_root(rows, {}, None, recipe)
        self.assertEqual([w["hard"] for w in weights], [True, False, False])
        self.assertAlmostEqual(sum(w["weight"] for w in weights), 1.)
        self.assertAlmostEqual(weights[0]["weight"], 2/3)
        rows[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            mixture.weights_for_root(rows, {}, None, recipe)

    def test_latest_weighted_gradient_matches_finite_difference(self):
        rows = groups() + copy.deepcopy(groups(False))
        rows[0]["training_group_weight"] = .8
        rows[1]["training_group_weight"] = .2
        active, targets, signs, spans = training.root_rows(rows)
        prediction = np.linspace(-.1, .2, len(active), dtype=np.float32)
        _, _, gradient = training.objective(rows, prediction, targets, signs, spans)
        for i in range(len(prediction)):
            losses = []
            for direction in (-1, 1):
                shifted = prediction.copy()
                shifted[i] += np.float32(direction * .001)
                huber, ranking, _ = training.objective(rows, shifted, targets, signs, spans)
                losses.append(huber + .25 * ranking)
            self.assertAlmostEqual(float(gradient[i]), (losses[1]-losses[0])/.002,
                                   delta=1e-4)
        rows[1]["split"] = "validation"
        with self.assertRaises(ValueError):
            training.objective(rows, prediction, targets, signs, spans)


if __name__ == "__main__":
    unittest.main()
