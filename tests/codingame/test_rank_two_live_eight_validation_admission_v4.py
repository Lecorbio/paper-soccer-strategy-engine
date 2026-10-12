import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_validation_admission_v4 as admission


class AdmissionBindingTests(unittest.TestCase):
    def fixture(self):
        def ref(name):
            return dict(path=name, sha256=name)
        plan = dict(declaration=ref('declaration'), live_teacher_plan=ref('train-plan'),
                    validation_teacher_plan=ref('validation-plan'), ancestry=ref('union'),
                    retained_train_corpus=ref('retained-train'))
        current = dict(pending_exposures=[], active_claims={},
            live_eight_continuation_05_declaration=plan['declaration'],
            live_eight_continuation_05_teacher_plan=plan['live_teacher_plan'],
            live_eight_continuation_05_validation_teacher_plan=plan['validation_teacher_plan'],
            live_eight_ancestry_validated=plan['ancestry'], live_eight_census_layers=['all40'])
        declaration = dict(architecture=[6301, 8, 8, 1], outcomes_as_targets=False,
            retained_train_corpus=plan['retained_train_corpus'],
            fresh_validation_recipe=dict(colors=[0, 1], maximum_early_parents_per_root=2,
                minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
                outcomes_as_targets=False, root_depths=[8, 10], roots=128, seed=2026101237))
        docs = dict(declaration=declaration, union=dict(passed=True, layers=['all40']))
        return plan, current, docs

    def check(self, plan, current, docs):
        with patch.object(admission.c, 'verify', side_effect=lambda ref: ref['path']), \
             patch.object(admission.c, 'read', side_effect=lambda path: docs[str(path)]):
            admission.ready_inputs(plan, current)

    def test_current_closed_bank_passes(self):
        self.check(*self.fixture())

    def test_unclosed_exposures_or_claims_block_admission(self):
        for field in ('pending_exposures', 'active_claims'):
            with self.subTest(field=field):
                plan, current, docs = self.fixture()
                current[field] = ['unclosed']
                with self.assertRaises(PermissionError):
                    self.check(plan, current, docs)

    def test_replaced_plan_or_corpus_is_rejected(self):
        for field in ('declaration', 'live_teacher_plan', 'validation_teacher_plan',
                      'ancestry', 'retained_train_corpus'):
            with self.subTest(field=field):
                plan, current, docs = self.fixture()
                plan[field] = {}
                with self.assertRaises(PermissionError):
                    self.check(plan, current, docs)

    def test_stale_ancestry_or_prior_seed_is_rejected(self):
        for changed in ('layers', 'seed'):
            with self.subTest(changed=changed):
                plan, current, docs = self.fixture()
                if changed == 'layers':
                    docs['union']['layers'] = ['only39']
                else:
                    docs['declaration']['fresh_validation_recipe']['seed'] = 2026101135
                with self.assertRaises(PermissionError):
                    self.check(plan, current, docs)

    def test_held_out_label_cannot_enter_train(self):
        retained = dict(group_artifacts=[], live_root_ids=[])
        teacher = dict(positions=[dict(position_id='p', root_group_id='r')])
        execution = dict(passed=True, parent_receipts=[dict(path='receipt', sha256='bound')])
        receipt = dict(position_id='p', success=True, output={})
        held_out = dict(split='validation', root_group_id='r')
        with patch.object(admission.c, 'verify', return_value='receipt'), \
             patch.object(admission.c, 'read', return_value=receipt), \
             patch.object(admission.live.prior, 'legacy_reference', return_value={}), \
             patch.object(admission.data, 'bound') as bound, \
             patch.object(admission.live.prior, 'normalized_label', return_value=held_out):
            bound.return_value.read_text.return_value = '{}'
            with self.assertRaises(PermissionError):
                admission.train_feature_keys(retained, teacher, execution)


if __name__ == '__main__':
    unittest.main()
