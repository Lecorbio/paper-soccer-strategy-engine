import tempfile
from pathlib import Path
import unittest
import numpy as np

from tools import rank_two_live_eight_calibration_v4 as calibration
from tests.codingame.test_rank_two_focused_training_v1 import groups


class Corpus:
    identity = dict(path='unit-corpus', sha256='fixed')
    def __init__(self):
        self.calls = 0
    def root(self, split, root):
        self.calls += 1
        assert split == 'train'
        return groups()


class BoundedCalibrationTests(unittest.TestCase):
    def test_batched_predictions_match_full_float_and_quantized_bits(self):
        training = calibration.training
        architecture, parameters = training.retained.initialize('dd8', 2026101134)
        active = [[i % 316, 316 + i % 105, 6300] for i in range(2053)]
        scales = {k: np.float32(max(float(np.max(np.abs(v))) / 31, 1e-8)) for k, v in parameters.items()}
        for q in [None, training.model.quantize(parameters, scales, 6)]:
            expected = training.retained.forward(parameters, architecture, active, q)[0]
            actual = calibration.bounded_predict(parameters, architecture, active, q, maximum=7)
            np.testing.assert_array_equal(actual, expected)

    def test_exact_recipe_and_completed_roots_survive_interruption(self):
        training = calibration.training
        architecture, parameters = training.retained.initialize('dd8', 2026101134)
        roots, masses = ['a', 'b'], dict(a=1., b=1.)
        expected_scales, expected_report = training.calibration(parameters, architecture, Corpus(), roots, 4, masses)
        corpus = Corpus()
        boundaries = 0
        def stop():
            nonlocal boundaries
            boundaries += 1
            if boundaries == 4:
                raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as directory:
            binding = dict(path='numeric-plan', sha256='unchanged')
            with self.assertRaises(RuntimeError):
                calibration.calibration(parameters, architecture, corpus, roots, 4, masses,
                    Path(directory), binding, stop)
            self.assertEqual(corpus.calls, 3)
            scales, report = calibration.calibration(parameters, architecture, corpus, roots, 4, masses,
                Path(directory), binding)
            self.assertEqual(corpus.calls, 36)
            self.assertEqual(report, expected_report)
            for key in scales:
                np.testing.assert_array_equal(scales[key], expected_scales[key])
            with self.assertRaises(ValueError):
                calibration.calibration(parameters, architecture, corpus, roots, 6, masses,
                    Path(directory), binding)


if __name__ == '__main__':
    unittest.main()
