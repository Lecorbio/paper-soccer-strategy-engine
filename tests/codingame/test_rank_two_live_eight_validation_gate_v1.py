import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_validation_gate_v1 as gate


class ApprovedBankTests(unittest.TestCase):
    def roots(self):
        frozen = ['root-%03d' % i for i in range(128)]
        return frozen, frozen[:-1], frozen[-1:], frozen[:127]

    def test_exact_partition_reports_all_counts(self):
        self.assertEqual(gate.root_accounting(*self.roots()),
            dict(frozen=128, evaluated=127, excluded=1, comparable_early=127))

    def test_missing_or_replacement_root_is_rejected(self):
        frozen, evaluated, excluded, early = self.roots()
        for replacement in [evaluated[:-1], [*evaluated[:-1], 'replacement']]:
            with self.assertRaises(PermissionError):
                gate.root_accounting(frozen, replacement, excluded, early[:100])

    def test_overlap_duplicate_and_early_shortfall_are_rejected(self):
        frozen, evaluated, excluded, early = self.roots()
        with self.assertRaises(PermissionError):
            gate.root_accounting(frozen, evaluated, [evaluated[0]], early)
        with self.assertRaises(ValueError):
            gate.root_accounting(frozen, evaluated, excluded, [*early[:100], early[0]])
        with self.assertRaises(PermissionError):
            gate.root_accounting(frozen, evaluated, excluded, early[:99])

    def fixture(self):
        frozen, evaluated, excluded, early = self.roots()
        ref = lambda name: dict(path=name, sha256=name)
        approval = dict(approved_by='human', frozen_roots=128, evaluated_roots=127, excluded_roots=1,
            minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
            no_qualification_rule_change=True, generation_plan=ref('generated'),
            validation_teacher_plan=ref('teacher'), validation_teacher_execution=ref('execution'),
            excluded_root_ids=excluded, seed=1)
        evidence = dict(gate_approval=ref('approval'), generation_plan=ref('generated'),
            plan=ref('teacher'), execution=ref('execution'), roots=128, early_roots=127,
            frozen_root_ids=frozen, evaluated_root_ids=evaluated, excluded_root_ids=excluded,
            comparable_early_root_ids=early,
            root_accounting=gate.root_accounting(frozen, evaluated, excluded, early))
        docs = dict(approval=approval, generated=dict(seed=1, rows=[dict(root_group_id=r) for r in frozen]))
        return evidence, evaluated, docs

    def check(self, evidence, evaluated, docs):
        with patch.object(gate.c, 'verify', side_effect=lambda ref: ref['path']), \
             patch.object(gate.c, 'read', side_effect=lambda path: docs[path]):
            return gate.validate(evidence, evaluated)

    def test_bound_approval_passes(self):
        self.assertEqual(self.check(*self.fixture())['evaluated'], 127)

    def test_changed_seed_authority_or_exclusion_is_rejected(self):
        for key, value in [('seed', 2), ('approved_by', 'automatic'),
                           ('excluded_root_ids', ['another']), ('no_qualification_rule_change', False)]:
            with self.subTest(key=key):
                evidence, evaluated, docs = self.fixture()
                docs['approval'][key] = value
                with self.assertRaises(PermissionError):
                    self.check(evidence, evaluated, docs)

    def test_quarantined_root_cannot_reenter_evaluation(self):
        evidence, evaluated, docs = self.fixture()
        with self.assertRaises(PermissionError):
            self.check(evidence, [*evaluated, evidence['excluded_root_ids'][0]], docs)

    def test_counts_cannot_claim128_evaluated(self):
        evidence, evaluated, docs = self.fixture()
        evidence['root_accounting'] = copy.deepcopy(evidence['root_accounting'])
        evidence['root_accounting']['evaluated'] = 128
        with self.assertRaises(ValueError):
            self.check(evidence, evaluated, docs)


if __name__ == '__main__':
    unittest.main()
