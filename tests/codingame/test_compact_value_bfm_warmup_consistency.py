"""Bounded new-profile grammar and real tiny 1+4 execution; no campaign data."""
import copy
import dataclasses
import os
from pathlib import Path
import unittest
from unittest import mock

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
import numpy as np
from tools import compact_value_bfm_train as t
from tests.codingame.test_compact_value_bfm_student_rivals import (
    ARCH, ARM, SEED, POLICY, tiny_inputs, parameters, warmup)

PROFILE = t.WARMUP_CONSISTENCY_QAT_PROFILE
BASE = t.STUDENT_RIVALS_RETENTION_PROFILE


def reseal(value):
    return t.body_hashed({k: v for k, v in value.items() if k != 'body_sha256'})


class WarmupConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = tiny_inputs()
        cls.initial = parameters()
        cls.runs = {}
        with t.native_thread_execution_scope():
            for weight in (0., .1):
                f = warmup(cls.inputs, cls.initial, weight)
                base = t.run_fixed_scale_qat(f, cls.inputs, ARCH, ARM, SEED,
                    ranking_weight=weight, qat_profile=BASE)
                new = t.run_fixed_scale_qat(f, cls.inputs, ARCH, ARM, SEED,
                    ranking_weight=weight, qat_profile=PROFILE)
                cls.runs[weight] = (f, base, new)

    def test_registered_profile_preserves_all_five_historical_contracts(self):
        expected = {
            'standard-v1': '9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
            'refined-adaptive-scales-v1': '7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd',
            'retention-first-low-rate-v1': '3cdbd16892860b6db547ca510e97604ba7742ae07392df64fd817b315b975de3',
            'channel-prediction-qat-v1': '4ae78b345fb3f586d0b590a04127993aa9880c6e1cec265191b7db7398cafaa0',
            'student-rivals-retention-v1': 'a41b6afc383ada5d977e2673fee9344e296b1f156b0627457cacc534006ea653',
        }
        for name, digest in expected.items():
            self.assertEqual(t.qat_profile_contract(name)['body_sha256'], digest)
        new = t.qat_profile_contract(PROFILE)
        base = t.qat_profile_contract(BASE)
        self.assertEqual({k: v for k, v in new.items() if k not in ('body_sha256', 'qat_profile', 'warmup_consistency')},
                         {k: v for k, v in base.items() if k not in ('body_sha256', 'qat_profile')})
        self.assertEqual(new['warmup_consistency']['coefficient'], 1.)
        self.assertEqual(new['warmup_consistency']['huber_delta'], .25)
        self.assertEqual(t.validate_qat_profile_contract(new), new)
        with self.assertRaises(t.TrainingError):
            t.resolve_qat_profile(dataclasses.replace(t.resolve_qat_profile(PROFILE), qat_learning_rate=.00025))

    def test_genuine_one_plus_four_both_arms_have_active_frozen_teacher(self):
        for weight, (f, base, new) in self.runs.items():
            summary = t.summarize_warmup_consistency_execution(f.report, new.report)
            self.assertEqual(summary['validated_qat_epochs'], 4)
            self.assertEqual(summary['teacher_forward_calls'], 8)
            self.assertEqual(summary['consistency_gradient_additions'], 8)
            self.assertGreater(summary['nonzero_derivative_batches'], 0)
            self.assertFalse(summary['control_checked_before_optimizer'])
            self.assertFalse(summary['per_term_parameter_attribution_measured'])
            self.assertEqual(new.report['optimizer_steps'], 8)
            self.assertEqual(new.report['learning_rate'], .0000625)
            self.assertEqual(f.report['optimizer']['learning_rate'], .00006)
            self.assertEqual(new.report['pre_qat_validation'], base.report['pre_qat_validation'])
            self.assertEqual(new.report['scale_search']['selected_scales'], base.report['scale_search']['selected_scales'])
            for epoch in new.report['history']:
                ev = epoch['warmup_consistency']
                self.assertEqual(ev['reference_before'], ev['reference_after'])
                self.assertEqual(ev['training_inputs'], summary['reference']['training_inputs'])
                self.assertEqual(epoch['training_total_objective'], ev['mean_loss_components']['total_objective'])
                self.assertEqual(epoch['training_total_objective_definition'], t.CONSISTENCY_TRAINING_OBJECTIVE)
                if weight == 0:
                    self.assertEqual(ev['mean_loss_components']['weighted_ranking_loss'], 0.)
                for layer in ('w1', 'w2', 'w3'):
                    self.assertTrue(epoch['fake_quantization']['master_parameter_updates'][layer]['changed'])
            self.assertNotEqual(new.report['final_master_per_layer_update_evidence'], base.report['final_master_per_layer_update_evidence'])
            self.assertEqual(t.validate_successor_schedule_execution(f.report, new.report, seed=SEED)
                ['warmup_consistency_execution'], summary)

    def test_reference_is_bytes_backed_and_detects_forced_private_corruption(self):
        initial = parameters()
        teacher = t._FrozenWarmupTeacher(initial, ARCH)
        original = teacher.identity
        initial['w1'][:] = 0
        self.assertEqual(teacher.identity, original)
        for values in teacher.parameters.values():
            with self.assertRaises(ValueError): values.setflags(write=True)
        altered = list(teacher._payloads)
        altered[0] = bytes(len(altered[0]))
        object.__setattr__(teacher, '_payloads', tuple(altered))
        with self.assertRaisesRegex(t.TrainingError, 'teacher changed'): teacher.verify()
        with self.assertRaisesRegex(t.TrainingError, 'teacher changed'):
            teacher.predict((np.array([0], dtype='<u2'),), ARCH)

    def test_teacher_hook_rejects_float_phase_before_forward(self):
        teacher = t._FrozenWarmupTeacher(parameters(), ARCH)
        evidence = t._WarmupConsistencyEpochEvidence(teacher, seed=SEED, qat_epoch=1,
            input_identity={}, expectation_sha256=None)
        with mock.patch.object(t, 'forward', side_effect=AssertionError('unexpected forward')):
            with self.assertRaises(t.TrainingError):
                t._train_mixed_batch(parameters(), ARCH, ARM, None, self.inputs,
                    np.arange(64), np.arange(192), training_pair_policy=POLICY,
                    warmup_teacher=teacher, consistency_evidence=evidence)
        for name in ('new', 'anchor'):
            bad = dataclasses.replace(self.inputs, **{name: dataclasses.replace(getattr(self.inputs, name), split='validation')})
            with mock.patch.object(t, 'forward', side_effect=AssertionError('heldout reference')):
                with self.assertRaises(t.TrainingError):
                    t._train_mixed_batch(parameters(), ARCH, ARM, None, bad,
                        np.arange(64), np.arange(192), fixed_scales={'w1': .1, 'w2': .1, 'w3': .1},
                        training_pair_policy=POLICY, warmup_teacher=teacher, consistency_evidence=evidence)

    def test_resealed_epoch_phase_counts_components_reference_tampers_reject(self):
        f, _, new = self.runs[.1]
        mutations = [
            lambda e: e.update(qat_epoch=2),
            lambda e: e.update(consistency_gradient_additions=1),
            lambda e: e.update(teacher_forward_calls=True),
            lambda e: e.update(reference_optimizer_steps=1),
            lambda e: e.update(heldout_reference_predictions=True),
            lambda e: e.update(nonzero_derivative_batches=0),
            lambda e: e['mean_loss_components'].update(consistency_loss=-1.),
            lambda e: e['mean_loss_components'].update(label_loss=e['mean_loss_components']['label_loss'] + .01),
            lambda e: e['reference_after']['layers']['w3'].update(sha256='a' * 64),
            lambda e: e['training_inputs']['anchor'].update(sha256='b' * 64),
        ]
        for change in mutations:
            value = copy.deepcopy(new.report)
            change(value['history'][0]['warmup_consistency'])
            value['history'][0]['warmup_consistency'] = reseal(value['history'][0]['warmup_consistency'])
            with self.assertRaises(t.TrainingError):
                t.summarize_warmup_consistency_execution(f.report, value)

    def test_total_component_reporting_keeps_actual_sequential_addition_order(self):
        teacher = t._FrozenWarmupTeacher(parameters(), ARCH)
        evidence = t._WarmupConsistencyEpochEvidence(teacher, seed=SEED, qat_epoch=1,
            input_identity={}, expectation_sha256=None)
        # Python sum() may use compensated summation. The optimizer's objective
        # uses these two explicit additions, and its reported mean must match.
        evidence.losses = [(1., 1e-16, 1e-16)]
        result = evidence.finish()
        self.assertEqual(result['mean_loss_components']['total_objective'], (1. + 1e-16) + 1e-16)

    def test_subnormal_disagreement_may_have_zero_weighted_derivative(self):
        predictions = np.full(256, np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32)
        targets = np.zeros(256, dtype=np.float32)
        weights = np.full(256, np.float32(1. / 256), dtype=np.float32)
        loss, derivative = t._weighted_huber_loss_gradient(predictions, targets, weights)
        self.assertTrue(np.all(predictions != targets))
        self.assertTrue(np.all(derivative == 0))
        self.assertEqual(loss, 0.)
        f, _, new = self.runs[.1]
        value = copy.deepcopy(new.report)
        item = value['history'][0]
        evidence = item['warmup_consistency']
        evidence['nonzero_derivative_batches'] = 0
        evidence['consistency_output_gradient_l2_sum'] = 0.
        evidence['nonzero_teacher_disagreement_rows'] = 256
        components = evidence['mean_loss_components']
        components['consistency_loss'] = 0.
        components['total_objective'] = components['label_loss'] + components['weighted_ranking_loss']
        item['training_total_objective'] = components['total_objective']
        item['warmup_consistency'] = reseal(evidence)
        t.summarize_warmup_consistency_execution(f.report, value)

    def test_reference_policy_and_warmup_transplants_reject(self):
        f, base, new = self.runs[.1]
        foreign = copy.deepcopy(f.report)
        foreign['validation']['common_adjudicator']['weighted_huber'] += .001
        with self.assertRaises(t.TrainingError):
            t.summarize_warmup_consistency_execution(foreign, new.report)
        changed = copy.deepcopy(new.report)
        changed['warmup_consistency']['policy']['coefficient'] = .5
        changed['warmup_consistency']['policy'] = reseal(changed['warmup_consistency']['policy'])
        with self.assertRaises(t.TrainingError):
            t.validate_qat_execution_evidence(changed, expected_profile=PROFILE, float_validation_reference=f.metrics)
        boolean_policy = copy.deepcopy(new.report)
        boolean_policy['warmup_consistency']['policy']['coefficient'] = True
        with self.assertRaises(t.TrainingError):
            t.validate_qat_execution_evidence(boolean_policy, expected_profile=PROFILE, float_validation_reference=f.metrics)
        old = copy.deepcopy(base.report)
        old['warmup_consistency'] = copy.deepcopy(new.report['warmup_consistency'])
        with self.assertRaises(t.TrainingError):
            t.validate_qat_execution_evidence(old, expected_profile=BASE, float_validation_reference=f.metrics)

    def test_legacy_expectation_rejects_before_any_training(self):
        with mock.patch.object(t, 'train_float_seed', side_effect=AssertionError('warmup')):
            with self.assertRaises(t.TrainingError):
                t.train_seed_candidate(None, self.inputs, ARCH, ARM, SEED, Path('unused'),
                    qat_profile=BASE, warmup_consistency_expectation={})
        f, _, _ = self.runs[.1]
        with mock.patch.object(t, 'select_fixed_scales', side_effect=AssertionError('search')):
            with self.assertRaises(t.TrainingError):
                t.run_fixed_scale_qat(f, self.inputs, ARCH, ARM, SEED, ranking_weight=.1,
                    qat_profile=BASE, warmup_consistency_expectation={})

    def test_source_job_forwards_only_explicit_control(self):
        from tools import compact_value_bfm_seed_process_v2 as process
        fake_trainer = mock.Mock()
        fake_trainer.ARCHITECTURES = {'capacity-12x8': ARCH}
        fake_trainer.ARMS = {'search-target': ARM}
        receipt = {'native_thread_execution': {'test': True}}
        fake_trainer.train_seed_candidate.return_value = receipt
        fake_trainer._load_seed_receipt_from_reference.return_value = receipt
        fake_campaign = mock.Mock()
        job = {'seed': SEED, 'weight': .1, 'directory': '/unused', 'binding': {'body_sha256': 'a' * 64},
               'warmup_consistency_expectation': {'body_sha256': 'b' * 64}}
        with mock.patch.object(process, '_modules', return_value=(fake_campaign, fake_trainer)), \
             mock.patch.object(process, '_WORKER', ({'qat_profile': PROFILE}, None, None, None)), \
             mock.patch.object(process, 'prediction_options', return_value={}):
            process._train(job, {})
        self.assertEqual(fake_trainer.train_seed_candidate.call_args.kwargs['warmup_consistency_expectation'], job['warmup_consistency_expectation'])


if __name__ == '__main__':
    unittest.main()
