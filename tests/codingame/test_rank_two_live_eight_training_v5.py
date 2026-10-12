"""Verify exact root accounting and unchanged numerical training safeguards."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_training_v5 as training
from tests.codingame import test_rank_two_live_eight_training_v4 as previous


class NumericalSafeguards(previous.Training):
    def setUp(self):
        self.original = previous.training
        previous.training = training

    def tearDown(self):
        previous.training = self.original


class QuarantinedEvaluationTests(unittest.TestCase):
    def fixture(self):
        roots = ['root-%03d' % i for i in range(127)]
        approval = dict(path='approval', sha256='bound')
        corpus = SimpleNamespace(manifest=dict(validation_gate_approval=approval,
            fresh_validation=dict(path='bank', sha256='bank')),
            roots=dict(validation=dict.fromkeys(roots)))
        evidence = dict(gate_approval=approval, evaluated_root_ids=roots)
        return corpus, evidence, roots

    def check(self, corpus, evidence, roots, early):
        with patch.object(training.campaign, 'verify', return_value='bank'), \
             patch.object(training.campaign, 'read', return_value=evidence), \
             patch.object(training.validation_gate, 'validate',
                          return_value=dict(frozen=128, evaluated=127, excluded=1, comparable_early=127)):
            return training.validation_accounting(corpus, roots, early)

    def test_every_approved_evaluated_root_is_reported(self):
        corpus, evidence, roots = self.fixture()
        self.assertEqual(self.check(corpus, evidence, roots, 127),
            dict(frozen=128, evaluated=127, excluded=1, comparable_early=127))

    def test_missing_or_reintroduced_root_is_rejected(self):
        corpus, evidence, roots = self.fixture()
        for changed in [roots[:-1], [*roots, 'quarantined']]:
            with self.assertRaises(ValueError):
                self.check(corpus, evidence, changed, 127)

    def test_early_count_and_approval_must_match_bank(self):
        corpus, evidence, roots = self.fixture()
        with self.assertRaises(ValueError):
            self.check(corpus, evidence, roots, 126)
        evidence['gate_approval'] = dict(path='another', sha256='changed')
        with self.assertRaises(PermissionError):
            self.check(corpus, evidence, roots, 127)

    def test_unapproved_bank_keeps_previous_gate(self):
        corpus = SimpleNamespace(manifest={})
        with self.assertRaises(ValueError):
            training.validation_accounting(corpus, range(127), 127)
        self.assertEqual(training.validation_accounting(corpus, range(128), 128)['excluded'], 0)


if __name__ == '__main__':
    unittest.main()
