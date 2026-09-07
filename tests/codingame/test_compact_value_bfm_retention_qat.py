"""Frozen-reference retention ordering and quarter-rate QAT, on tiny fixtures."""
import copy
import dataclasses
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

for _name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

import numpy as np
from tools import compact_value_bfm_train as trainer
from tests.codingame import test_compact_value_bfm_training as fixtures
from tests.codingame import test_compact_value_bfm_deterministic_best as small

PROFILE = trainer.RETENTION_FIRST_LOW_RATE_QAT_PROFILE


def metrics(huber=.05, sign=.9, flip=.1):
    return {
        'common_adjudicator': {'weighted_huber': huber, 'objective_weighted_huber': huber, 'sign_accuracy': sign},
        'canonical_validation': {'weighted_huber': huber, 'objective_weighted_huber': huber, 'sign_accuracy': sign},
        'successor_ranking': {'loss_weight': 0., 'float_vs_quantized_action_flip_rate': flip,
            'mean_teacher_regret': flip, 'top1_agreement': 1. - flip},
    }


def old_key(report, profile, **_kwargs):
    common, canonical = report['common_adjudicator'], report['canonical_validation']
    base = (float(common['objective_weighted_huber']), float(canonical['objective_weighted_huber']),
            -float(common['sign_accuracy']), -float(canonical['sign_accuracy']))
    ranking = report.get('successor_ranking')
    if ranking is not None and float(ranking.get('loss_weight', 0.)) != 0.:
        base += (float(ranking['mean_teacher_regret']), -float(ranking['top1_agreement']),
                 float(ranking['float_vs_quantized_action_flip_rate']), float(ranking['pairwise_loss']))
    if profile.name == trainer.STANDARD_QAT_PROFILE: return base
    return (float(ranking['float_vs_quantized_action_flip_rate']), float(ranking['mean_teacher_regret']),
            -float(ranking['top1_agreement']), *base)


class RetentionQATTests(unittest.TestCase):
    def setUp(self):
        self.profile = trainer.resolve_qat_profile(PROFILE)
        self.architecture = trainer.ARCHITECTURES['capacity-12x8']
        self.arm = trainer.ARMS['search-target']; self.seed = trainer.FIXED_SEEDS[0]

    def key(self, value, reference=None):
        return trainer._qat_validation_key(value, self.profile,
            float_validation_reference=metrics() if reference is None else reference)

    def test_old_profile_contract_hashes_unchanged_and_new_closed_recipe(self):
        expected = {'standard-v1': '9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
            'refined-adaptive-scales-v1': '7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd'}
        for name, digest in expected.items():
            contract = trainer.qat_profile_contract(name)
            self.assertEqual(contract['body_sha256'], digest)
            self.assertNotIn('retention_policy', contract['scale_selection'])
        contract = trainer.qat_profile_contract(PROFILE)
        self.assertEqual(contract['schedule']['qat_learning_rate'], .0000625)
        self.assertEqual(contract['schedule']['qat_learning_rate'], trainer.QAT_LEARNING_RATE / 4)
        self.assertEqual(contract['schedule']['float_warmup_epochs'], 1)
        self.assertEqual(contract['schedule']['qat_epochs'], 4)
        self.assertTrue(contract['schedule']['all_layers_trainable_each_qat_epoch'])
        old = trainer.qat_profile_contract(trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE)['scale_selection']
        self.assertEqual({key: value for key, value in contract['scale_selection'].items() if key not in ('validation_objective', 'retention_policy')},
                         {key: value for key, value in old.items() if key != 'validation_objective'})
        self.assertEqual(trainer.validate_qat_profile_contract(contract, expected_name=PROFILE), contract)
        with self.assertRaises(trainer.TrainingError):
            trainer.resolve_qat_profile(dataclasses.replace(self.profile, qat_learning_rate=.00025))

    def test_feasibility_precedes_violations_and_refined_ranking(self):
        passing = metrics(.0505, .899, .9); failing = metrics(.06, .89, 0.)
        self.assertEqual(self.key(passing)[:2], (0., 0.))
        self.assertEqual(self.key(failing)[0], 1.)
        self.assertLess(self.key(passing), self.key(failing))
        mild = metrics(.052, .9, .9); severe = metrics(.06, .9, 0.)
        self.assertLess(self.key(mild), self.key(severe))
        better_rank = metrics(.0505, .899, .2)
        self.assertLess(self.key(better_rank), self.key(passing))
        self.assertEqual(self.key(passing)[2:], old_key(passing, self.profile))

    def test_exact_eight_component_normalization(self):
        candidate = metrics(); candidate['common_adjudicator'].update(sign_accuracy=.89, weighted_huber=.058, objective_weighted_huber=.058)
        expected = math.fsum((max(0., (.8475 - .89) / .8475), max(0., (.058 - .056) / .056),
            max(0., (.9 - .89 - .005) / .005), max(0., (.058 - .05 * 1.02) / max(.05 * 1.02, .056))))
        self.assertEqual(self.key(candidate)[:2], (1., expected))
        self.assertEqual(trainer.qat_profile_contract(PROFILE)['scale_selection']['retention_policy']['sum'], 'math.fsum')

    def test_strict_sign_boundary_uses_gate_even_when_excess_is_zero(self):
        baseline = metrics(); candidate = metrics()
        baseline['common_adjudicator']['sign_accuracy'] = .005
        candidate['common_adjudicator']['sign_accuracy'] = 0.
        self.assertEqual(.005 - 0. - trainer.MAXIMUM_SIGN_LOSS, 0.)
        gate = trainer.offline_advancement_gate(baseline, candidate)
        self.assertIn('common_adjudicator quantized sign loss is not below .005', gate['errors'])
        self.assertEqual(self.key(candidate, baseline)[:2], (1., 1.))  # Only absolute-sign deficit contributes.
        # Real near-boundary reports above the absolute floors use exact binary
        # subtraction, not isclose/rounding/epsilon relaxation.
        for sign in (.895, float(np.nextafter(.895, 1.))):
            value = metrics(sign=sign)
            self.assertEqual(self.key(value)[0] == 0., trainer.offline_advancement_gate(metrics(), value)['passed'])
        with mock.patch.object(trainer, 'offline_advancement_gate', return_value={'passed': False}):
            self.assertEqual(self.key(metrics())[:2], (1., 0.))

    def test_absolute_and_relative_huber_boundaries_are_inclusive(self):
        baseline = metrics(.1); candidate = metrics()
        candidate['common_adjudicator']['weighted_huber'] = .056
        candidate['canonical_validation']['weighted_huber'] = .0551
        self.assertEqual(self.key(candidate, baseline)[0], 0.)
        candidate = metrics(.05 * 1.02)
        self.assertEqual(self.key(candidate)[0], 0.)
        candidate['common_adjudicator']['weighted_huber'] = float(np.nextafter(.05 * 1.02, math.inf))
        self.assertEqual(self.key(candidate)[0], 1.)

    def test_zero_huber_reference_and_nonfinite_scores(self):
        baseline = metrics(0.)
        self.assertEqual(self.key(metrics(0.), baseline)[:2], (0., 0.))
        for value in (.01, 1e-300):
            key = self.key(metrics(value), baseline)
            self.assertEqual(key[0], 1.); self.assertGreater(key[1], 0.)
            self.assertTrue(all(math.isfinite(item) for item in key))
        for field, value in (('weighted_huber', math.nan), ('weighted_huber', math.inf), ('sign_accuracy', -1e308)):
            candidate = metrics(); candidate['common_adjudicator'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(trainer.TrainingError): self.key(candidate)
        with self.assertRaises(trainer.TrainingError): self.key(metrics(), metrics(math.inf))

    def test_reference_is_required_copied_and_immutable(self):
        baseline = metrics(); frozen = trainer._retention_reference(self.profile, baseline)
        key = self.key(metrics(.052), frozen); payload = frozen.payload
        baseline['common_adjudicator']['weighted_huber'] = 1.
        document = frozen.document(); document['float_validation']['canonical_validation']['weighted_huber'] = 2.
        self.assertEqual(frozen.payload, payload); self.assertEqual(self.key(metrics(.052), frozen), key)
        with self.assertRaises(dataclasses.FrozenInstanceError): frozen.payload = b'{}'
        parameters = trainer.initialize_parameters(self.architecture, self.seed)
        with mock.patch.object(trainer, 'evaluate_validation_pair', side_effect=AssertionError('reference must precede forwards')):
            with self.assertRaisesRegex(trainer.TrainingError, 'requires frozen'):
                trainer.select_fixed_scales(parameters, self.architecture, object(), self.arm, qat_profile=PROFILE)

    def floating(self):
        data = small.inputs(small.ranking_groups())
        parameters = trainer.initialize_parameters(self.architecture, self.seed)
        value = trainer.train_float_seed(data, self.architecture, self.arm, self.seed,
            maximum_epochs=1, patience=1, learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
            ranking_weight=.1, initial_parameters=parameters)
        return data, value

    def test_initial_and_adaptive_search_choose_passing_scale_over_lower_flip(self):
        parameters = trainer.initialize_parameters(self.architecture, self.seed)
        candidates = (np.float32(1.), np.float32(2.))
        def evaluated(*_args, quantized, **_kwargs):
            return metrics(.05, .9, .9) if quantized.scales['w1'] == 2. else metrics(.06, .88, 0.)
        with mock.patch.object(trainer, 'robust_scale_candidates', return_value=candidates), \
                mock.patch.object(trainer, '_refined_scale_candidates', return_value=candidates), \
                mock.patch.object(trainer, 'evaluate_validation_pair', side_effect=evaluated):
            selected, report = trainer.select_fixed_scales(parameters, self.architecture, object(), self.arm,
                qat_profile=PROFILE, float_validation_reference=metrics())
            adapted, adaptive = trainer._adapt_fixed_scales(parameters, self.architecture, object(), self.arm,
                {'w1': 1., 'w2': 2., 'w3': 2.}, self.profile, qat_epoch=1, ranking_weight=0.,
                float_validation_reference=metrics())
        self.assertEqual(report['selected_scales'], {'w1': 2., 'w2': 1., 'w3': 1.})
        self.assertEqual(adaptive['selected_scales'], report['selected_scales'])
        self.assertTrue(trainer.offline_advancement_gate(metrics(), report['selected_validation'])['passed'])

    def test_epoch_selection_prefers_feasibility_then_ranking_and_earlier_ties(self):
        data = small.inputs(small.ranking_groups()); train_batch = trainer._train_mixed_batch
        with trainer.native_thread_execution_scope():
            with mock.patch.object(trainer, 'evaluate_validation_pair', return_value=metrics()):
                floating = trainer.train_float_seed(data, self.architecture, self.arm, self.seed,
                    maximum_epochs=1, patience=1, learning_rate=trainer.RANKING_FLOAT_LEARNING_RATE,
                    ranking_weight=0., initial_parameters=trainer.initialize_parameters(self.architecture, self.seed))
            for third_flip, expected_epoch in ((.8, 3), (.9, 2)):
                epoch = [0]
                def batch(*args, **kwargs):
                    result = train_batch(*args, **kwargs); epoch[0] += 1; return result
                def validation(*_args, **_kwargs):
                    if epoch[0] in (2, 3): return metrics(.0505, .899, .9 if epoch[0] == 2 else third_flip)
                    return metrics(.06, .88, 0.)
                with self.subTest(third_flip=third_flip), mock.patch.object(trainer, '_train_mixed_batch', side_effect=batch), \
                        mock.patch.object(trainer, 'evaluate_validation_pair', side_effect=validation):
                    result = trainer.run_fixed_scale_qat(floating, data, self.architecture, self.arm, self.seed, qat_profile=PROFILE)
                self.assertEqual(result.qat_epoch, expected_epoch)
                self.assertTrue(trainer.offline_advancement_gate(floating.metrics, result.metrics)['passed'])
                self.assertEqual(result.report['executed_qat_epochs'], [1, 2, 3, 4])

    def test_same_frozen_reference_across_initial_adaptive_and_epoch_selection(self):
        with trainer.native_thread_execution_scope():
            data, floating = self.floating(); references = []; masters = []
            initial, adaptive = trainer.select_fixed_scales, trainer._adapt_fixed_scales
            def initial_spy(*args, **kwargs):
                references.append(kwargs['float_validation_reference']); return initial(*args, **kwargs)
            def adaptive_spy(*args, **kwargs):
                references.append(kwargs['float_validation_reference'])
                masters.append(trainer._parameter_identity(args[0], args[1]))
                return adaptive(*args, **kwargs)
            with mock.patch.object(trainer, 'select_fixed_scales', side_effect=initial_spy), mock.patch.object(trainer, '_adapt_fixed_scales', side_effect=adaptive_spy):
                result = trainer.run_fixed_scale_qat(floating, data, self.architecture, self.arm, self.seed, ranking_weight=.1, qat_profile=PROFILE)
        self.assertEqual(len(references), 5); self.assertEqual(len({id(value) for value in references}), 1)
        self.assertEqual(references[0].metrics(), floating.metrics)
        self.assertGreater(len({repr(value) for value in masters}), 1)
        document = result.report['retention_reference']
        self.assertEqual(document, result.report['scale_search']['retention_reference'])
        self.assertTrue(all(epoch['adaptive_scale_search']['retention_reference'] == document for epoch in result.report['history']))
        trainer.validate_qat_execution_evidence(result.report, expected_profile=PROFILE, float_validation_reference=floating.metrics)
        changed = copy.deepcopy(result.report); changed['history'][0]['adaptive_scale_search']['retention_reference'] = trainer._retention_reference(self.profile, metrics()).document()
        with self.assertRaisesRegex(trainer.TrainingError, 'retention reference'):
            trainer.validate_qat_execution_evidence(changed, expected_profile=PROFILE, float_validation_reference=floating.metrics)
        with self.assertRaisesRegex(trainer.TrainingError, 'differs from frozen float'):
            trainer.validate_qat_execution_evidence(result.report, expected_profile=PROFILE, float_validation_reference=metrics())

    def test_actual_optimizer_and_completed_seed_receipt_use_quarter_rate(self):
        data = small.inputs(small.ranking_groups())
        # Satisfy the maintained 4,096-row scalar-parity contract with a tiny
        # synthetic sparse pool; no corpus or geometric qualification is claimed.
        data = dataclasses.replace(data, common_adjudicator=fixtures.dataset(
            [np.asarray([index % 4], dtype='<u2') for index in range(4096)], split='validation'))
        rates = []; adam = trainer.AdamW
        def observed(parameters, **kwargs):
            optimizer = adam(parameters, **kwargs); rates.append((kwargs['learning_rate'], float(optimizer.learning_rate)))
            return optimizer
        with tempfile.TemporaryDirectory() as temporary, trainer.native_thread_execution_scope():
            root = Path(temporary); bundle = fixtures.bundle_fixture(root)
            checkpoint = trainer.write_float_checkpoint(root/'initial', trainer.initialize_parameters(self.architecture, self.seed), self.architecture)
            with mock.patch.object(trainer, 'AdamW', side_effect=observed):
                receipt = trainer.train_seed_candidate(bundle, data, self.architecture, self.arm, self.seed,
                    root/'seed', ranking_weight=.1, initial_checkpoint=checkpoint, qat_profile=PROFILE)
            self.assertIn((.0000625, float(np.float32(.0000625))), rates)
            self.assertNotIn((.00025, float(np.float32(.00025))), rates)
            self.assertEqual(receipt['binding']['settings']['qat_learning_rate'], .0000625)
            qat = receipt['quantized_training']; self.assertEqual(qat['learning_rate'], .0000625)
            self.assertEqual(qat['executed_qat_epochs'], [1,2,3,4]); self.assertEqual(qat['optimizer_steps'], 4)
            self.assertEqual(qat['retention_reference']['float_validation'], receipt['float_validation'])
            for epoch in qat['history']:
                self.assertTrue(epoch['fake_quantization']['all_layers_trainable'])
                for change in epoch['fake_quantization']['master_parameter_updates'].values():
                    self.assertTrue(change['changed']); self.assertTrue(math.isfinite(change['l2_delta'])); self.assertGreater(change['l2_delta'], 0.)
            reference = trainer._seed_reference_path(root/'seed', self.architecture, self.arm, self.seed)
            self.assertEqual(trainer._load_seed_receipt_from_reference(root/'seed', reference, receipt['binding']), receipt)

    def test_resealed_metrics_cannot_diverge_from_winning_trials_or_epoch_selection(self):
        with trainer.native_thread_execution_scope():
            data, floating = self.floating()
            result = trainer.run_fixed_scale_qat(floating, data, self.architecture, self.arm, self.seed,
                ranking_weight=.1, qat_profile=PROFILE)
        trainer.validate_qat_execution_evidence(result.report, expected_profile=PROFILE,
            float_validation_reference=floating.metrics)
        for stage, error in (
            ('initial-selected-and-pre', 'winning scale trial'),
            ('pre-only', 'pre-QAT metrics'),
            ('adaptive-selected-and-epoch', 'winning scale trial'),
            ('epoch-only', 'epoch metrics'),
        ):
            changed = copy.deepcopy(result.report)
            epoch = changed['history'][changed['selected_qat_epoch'] - 1]
            # A forged feasible report would beat the failed actual epochs
            # while preserving all scale values and the selected epoch number.
            forged = copy.deepcopy(epoch['validation'])
            for pool in ('common_adjudicator', 'canonical_validation'):
                forged[pool].update(sign_accuracy=1., weighted_huber=0., objective_weighted_huber=0.)
            forged['successor_ranking'].update(float_vs_quantized_action_flip_rate=0.,
                mean_teacher_regret=0., top1_agreement=1.)
            self.assertTrue(trainer.offline_advancement_gate(floating.metrics, forged)['passed'])
            if stage == 'initial-selected-and-pre':
                changed['scale_search']['selected_validation'] = copy.deepcopy(forged)
                changed['pre_qat_validation'] = copy.deepcopy(forged)
            elif stage == 'pre-only':
                changed['pre_qat_validation'] = copy.deepcopy(forged)
            else:
                epoch['validation'] = copy.deepcopy(forged)
                changed['selected_validation'] = copy.deepcopy(forged)
                if stage == 'adaptive-selected-and-epoch':
                    epoch['adaptive_scale_search']['selected_validation'] = copy.deepcopy(forged)
            # Recompute the outer integrity hash so rejection must come from
            # semantic continuity, not from a stale serialized body hash.
            sealed = trainer.body_hashed({'schema': 'retention-metric-fixture.v1', 'quantized_training': changed})
            trainer.verify_body_hash(sealed, schema='retention-metric-fixture.v1', label='resealed metric fixture')
            with self.subTest(stage=stage), self.assertRaisesRegex(trainer.TrainingError, error):
                trainer.validate_qat_execution_evidence(sealed['quantized_training'], expected_profile=PROFILE,
                    float_validation_reference=floating.metrics)

    def test_existing_profiles_keep_complete_metrics_report_and_codes(self):
        with trainer.native_thread_execution_scope():
            data, floating = self.floating()
            for profile in (trainer.STANDARD_QAT_PROFILE, trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE):
                with self.subTest(profile=profile):
                    with mock.patch.object(trainer, '_qat_validation_key', old_key):
                        old = trainer.run_fixed_scale_qat(floating, data, self.architecture, self.arm, self.seed, ranking_weight=.1, qat_profile=profile)
                    actual = trainer.run_fixed_scale_qat(floating, data, self.architecture, self.arm, self.seed, ranking_weight=.1, qat_profile=profile)
                    self.assertEqual(trainer.canonical_json_bytes(actual.report), trainer.canonical_json_bytes(old.report))
                    self.assertEqual(trainer.canonical_json_bytes(actual.metrics), trainer.canonical_json_bytes(old.metrics))
                    self.assertNotIn('retention_reference', actual.report)
                    self.assertEqual(actual.report['learning_rate'], .00025)
                    for name in ('w1','w2','w3'):
                        self.assertEqual(actual.quantized.integer[name].tobytes(), old.quantized.integer[name].tobytes())
                        self.assertEqual(actual.quantized.scales[name].tobytes(), old.quantized.scales[name].tobytes())


if __name__ == '__main__': unittest.main()
