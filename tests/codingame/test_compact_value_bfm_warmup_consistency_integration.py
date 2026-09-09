"""Independent frozen-warmup objective integration; synthetic inputs only."""
import copy
import dataclasses
import hashlib
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

for _name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

import numpy as np
from tools import compact_value_bfm_train as t
from tests.codingame import test_compact_value_bfm_student_rivals_integration as fixtures

ARCH = t.ARCHITECTURES['capacity-12x8']
ARM = t.ARMS['search-target']
SEED = 20260907
LAYERS = ('w1', 'w2', 'w3')
BASE = t.STUDENT_RIVALS_RETENTION_PROFILE


def inputs_for(parameters=None, *, validation_rows=7):
    default, scalars, group = fixtures.controlled_inputs()
    labels = t.SuccessorRankingLabels(
        train=(group,), validation=(group,), teacher={'artifact_sha256': '1' * 64},
        source_bundle_body_sha256='a' * 64, artifact_sha256='b' * 64,
        body_sha256='c' * 64)
    inputs = t.TrainingInputs(new=scalars.new, anchor=scalars.anchor,
        common_adjudicator=dataclasses.replace(fixtures.dataset(validation_rows), split='validation'),
        canonical_validation=dataclasses.replace(fixtures.dataset(11), split='validation'),
        source_routes={}, successor_rankings=labels)
    return default if parameters is None else parameters, inputs, group


def independent_huber(predictions, targets, weights):
    """The declared float32 Huber arithmetic, not the production loss helper."""
    difference = np.asarray(predictions, np.float32) - np.asarray(targets, np.float32)
    delta = np.float32(.25)
    losses = np.empty(len(difference), np.float32)
    slope = np.empty(len(difference), np.float32)
    for i, value in enumerate(difference):
        if abs(value) <= delta:
            losses[i] = np.float32(.5) * value * value
            slope[i] = value
        else:
            losses[i] = delta * (abs(value) - np.float32(.5) * delta)
            slope[i] = delta if value > 0 else -delta
    denominator = float(np.sum(weights, dtype=np.float64))
    loss = float(np.sum(weights * losses, dtype=np.float64) / denominator)
    gradient = (weights * slope / np.float32(denominator)).astype(np.float32)
    return loss, gradient


def independent_weights(inputs, new_rows, anchor_rows):
    new, anchor = inputs.new.weights[new_rows], inputs.anchor.weights[anchor_rows]
    return np.concatenate((new * np.float32(.25 / float(np.sum(new, dtype=np.float64))),
                           anchor * np.float32(.75 / float(np.sum(anchor, dtype=np.float64)))))


def independent_batch(parameters, teacher_parameters, inputs, group, weight, scales):
    new_rows, anchor_rows = np.arange(64), np.arange(192)
    active = (*inputs.new.active_rows(new_rows), *inputs.anchor.active_rows(anchor_rows))
    quantized = t.quantize_fixed(parameters, ARCH, scales)
    prediction, cache = t.forward(parameters, ARCH, active, quantized=quantized)
    teacher_prediction = t.forward(teacher_parameters, ARCH, active)[0]
    targets = np.concatenate((inputs.new.targets[new_rows], inputs.anchor.targets[anchor_rows]))
    weights = independent_weights(inputs, new_rows, anchor_rows)
    label_loss, label_gradient = independent_huber(prediction, targets, weights)
    auxiliary_loss, auxiliary_gradient = independent_huber(prediction, teacher_prediction, weights)
    scalar_gradient = label_gradient + auxiliary_gradient
    gradients = t._network_gradients(parameters, ARCH, active, cache, scalar_gradient, quantized.effective())
    ranking_loss = 0.
    if weight:
        actions = tuple(successor.active for successor in group.successors)
        ranking, ranking_cache = t.forward(parameters, ARCH, actions, quantized=quantized)
        ranking_loss, derivative, _ = fixtures.independent_pair_objective(group, ranking, student=True)
        ranking_gradients = t._network_gradients(parameters, ARCH, actions, ranking_cache,
            derivative * np.float32(weight), quantized.effective())
        for name in LAYERS:
            gradients[name] += ranking_gradients[name]
    norm = math.sqrt(sum(float(np.sum(value * value, dtype=np.float64)) for value in gradients.values()))
    if norm > 5:
        for value in gradients.values():
            value *= np.float32(5. / norm)
    return label_loss + auxiliary_loss + weight * ranking_loss, gradients, {
        'weights': weights, 'label_gradient': label_gradient, 'auxiliary_gradient': auxiliary_gradient,
        'teacher_prediction': teacher_prediction, 'norm_before_clip': norm}


class ConsistencyMathIntegrationTests(unittest.TestCase):
    def assert_bits(self, left, right):
        self.assertEqual((left.shape, left.dtype), (right.shape, right.dtype))
        self.assertEqual(left.tobytes(), right.tobytes())

    def evidence(self, teacher, inputs):
        return t._WarmupConsistencyEpochEvidence(teacher, seed=SEED, qat_epoch=1,
            input_identity={name: t.dataset_identity(getattr(inputs, name)) for name in ('new', 'anchor')},
            expectation_sha256=None)

    def test_full_gradient_and_adam_use_one_weighted_consistency_term_before_joint_clip(self):
        ordinary, inputs, group = inputs_for()
        teacher = {name: value.copy() for name, value in ordinary.items()}
        teacher['w1'] *= 4
        large = {name: np.zeros(shape, np.float32) for name, shape in ARCH.shapes.items()}
        large['w1'][:24, 0] = 100
        large['w2'][0, :2] = .5
        large['w3'][:2] = (.5, -.5)
        large_teacher = {name: value.copy() for name, value in large.items()}
        large_teacher['w3'][0] = np.float32(.50001)
        examples = ((ordinary, teacher, .1, {'w1': .03, 'w2': .25, 'w3': .25}, False),
                    (large, large_teacher, 0., {'w1': 50., 'w2': .25, 'w3': .25}, True))
        class NoHeldout:
            def __getattr__(self, name):
                raise AssertionError('mixed training batch opened heldout input')
        training_only = dataclasses.replace(inputs, common_adjudicator=NoHeldout(), canonical_validation=NoHeldout())
        with t.native_thread_execution_scope():
            for parameters, teacher_parameters, weight, scales, clipped in examples:
                with self.subTest(weight=weight, clipped=clipped):
                    expected_loss, expected_gradients, trace = independent_batch(
                        parameters, teacher_parameters, inputs, group, weight, scales)
                    self.assertEqual(trace['norm_before_clip'] > 5, clipped)
                    self.assertTrue(np.any(trace['auxiliary_gradient'] != 0))
                    self.assertAlmostEqual(float(np.sum(trace['weights'][:64], dtype=np.float64)), .25, places=7)
                    self.assertAlmostEqual(float(np.sum(trace['weights'][64:], dtype=np.float64)), .75, places=7)
                    frozen = t._FrozenWarmupTeacher(teacher_parameters, ARCH)
                    evidence = self.evidence(frozen, inputs)
                    capture = fixtures.Capture()
                    derivative_calls, forward_calls = [], []
                    backprop, forward = t._network_gradients, t.forward
                    def observe_backprop(*args, **kwargs):
                        # Stop-gradient: every backward pass belongs to current
                        # QAT masters, never the copied floating reference.
                        self.assertIs(args[0], parameters)
                        derivative_calls.append(args[4].copy())
                        return backprop(*args, **kwargs)
                    def observe_forward(*args, **kwargs):
                        mode = kwargs.get('quantized') is not None
                        forward_calls.append((len(args[2]), mode))
                        if not mode:
                            self.assertEqual(len(args[2]), 256)
                        return forward(*args, **kwargs)
                    with mock.patch.object(t, '_network_gradients', side_effect=observe_backprop), \
                         mock.patch.object(t, 'forward', side_effect=observe_forward):
                        actual_loss = t._train_mixed_batch(parameters, ARCH, ARM, capture,
                            training_only, np.arange(64), np.arange(192), fixed_scales=scales,
                            ranking_groups=(group,) if weight else None, ranking_weight=weight,
                            training_pair_policy=t.STUDENT_RIVAL_PAIR_POLICY,
                            warmup_teacher=frozen, consistency_evidence=evidence)
                    self.assertEqual(actual_loss, expected_loss)
                    self.assertEqual(len(derivative_calls), 2 if weight else 1)
                    self.assert_bits(derivative_calls[0], trace['label_gradient'] + trace['auxiliary_gradient'])
                    self.assertEqual(forward_calls.count((256, False)), 1)
                    self.assertEqual(forward_calls.count((256, True)), 1)
                    self.assertEqual(len(forward_calls), 3 if weight else 2)
                    for name in LAYERS:
                        self.assert_bits(capture.gradients[name], expected_gradients[name])
                    report = evidence.finish()
                    self.assertEqual(report['consistency_gradient_additions'], 1)
                    self.assertEqual(report['reference_optimizer_steps'], 0)
                    self.assertEqual(report['reference_before'], report['reference_after'])
                    expected_master = {name: value.copy() for name, value in parameters.items()}
                    actual_master = {name: value.copy() for name, value in parameters.items()}
                    t.AdamW(expected_master, learning_rate=.0000625, weight_decay=1e-5).update(expected_master, expected_gradients)
                    t._train_mixed_batch(actual_master, ARCH, ARM,
                        t.AdamW(actual_master, learning_rate=.0000625, weight_decay=1e-5),
                        training_only, np.arange(64), np.arange(192), fixed_scales=scales,
                        ranking_groups=(group,) if weight else None, ranking_weight=weight,
                        training_pair_policy=t.STUDENT_RIVAL_PAIR_POLICY,
                        warmup_teacher=frozen, consistency_evidence=self.evidence(frozen, inputs))
                    for name in LAYERS:
                        self.assert_bits(actual_master[name], expected_master[name])

    def test_frozen_teacher_owns_immutable_parameters_and_checks_forced_corruption(self):
        parameters, inputs, _ = inputs_for()
        original = {name: value.copy() for name, value in parameters.items()}
        teacher = t._FrozenWarmupTeacher(parameters, ARCH)
        identity = copy.deepcopy(teacher.identity)
        active = inputs.new.active_rows(range(64))
        with t.native_thread_execution_scope():
            expected = t.forward(original, ARCH, active)[0]
            parameters['w1'][:] = 17
            parameters['w3'][:] = -12
            self.assert_bits(teacher.predict(active, ARCH), expected)
            self.assertEqual(teacher.identity, identity)
            for value in teacher.parameters.values():
                with self.assertRaises(ValueError):
                    value.setflags(write=True)
            teacher.identity['layers']['w3']['sha256'] = '0' * 64
            self.assertEqual(teacher.identity, identity)
            changed = original['w1'].copy()
            changed[0, 0] += np.float32(.1)
            object.__setattr__(teacher, '_payloads', (changed.tobytes(), *teacher._payloads[1:]))
            with mock.patch.object(t, 'forward', side_effect=AssertionError('corrupted reference was forwarded')):
                with self.assertRaisesRegex(t.TrainingError, 'teacher changed'):
                    teacher.predict(active, ARCH)

    def test_consistency_hook_cannot_run_during_float_warmup(self):
        parameters, inputs, _ = inputs_for()
        teacher = t._FrozenWarmupTeacher(parameters, ARCH)
        with mock.patch.object(t, 'forward', side_effect=AssertionError('warmup consistency forwarded')):
            with self.assertRaisesRegex(t.TrainingError, 'QAT-only'):
                t._train_mixed_batch(parameters, ARCH, ARM, fixtures.Capture(), inputs,
                    np.arange(64), np.arange(192), ranking_weight=0.,
                    training_pair_policy=t.STUDENT_RIVAL_PAIR_POLICY,
                    warmup_teacher=teacher, consistency_evidence=self.evidence(teacher, inputs))

    def test_reference_batch_rejects_non_training_roles_before_forward(self):
        parameters, inputs, _ = inputs_for()
        teacher = t._FrozenWarmupTeacher(parameters, ARCH)
        for source in ('new', 'anchor'):
            altered = dataclasses.replace(inputs, **{source: dataclasses.replace(getattr(inputs, source), split='validation')})
            with self.subTest(source=source), \
                 mock.patch.object(t, 'forward', side_effect=AssertionError('heldout reference was forwarded')):
                with self.assertRaisesRegex(t.TrainingError, 'TRAIN scalar'):
                    t._train_mixed_batch(parameters, ARCH, ARM, fixtures.Capture(), altered,
                        np.arange(64), np.arange(192), fixed_scales={'w1': .03, 'w2': .25, 'w3': .25},
                        training_pair_policy=t.STUDENT_RIVAL_PAIR_POLICY, warmup_teacher=teacher,
                        consistency_evidence=self.evidence(teacher, inputs))


class ConsistencyReceiptIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        initial, cls.inputs, _ = inputs_for(validation_rows=4096)
        cls.bundle = SimpleNamespace(body_sha256='a' * 64)
        cls.initial = t.write_float_checkpoint(cls.root / 'initial', initial, ARCH)
        cls.base, cls.fresh, cls.expectations = {}, {}, {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for weight in (0., .1):
                base_dir = cls.root / f'base-{weight}'
                base = t.train_seed_candidate(cls.bundle, cls.inputs, ARCH, ARM, SEED,
                    base_dir, ranking_weight=weight, initial_checkpoint=cls.initial,
                    qat_profile=BASE if weight else t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE)
                reference = t._load_canonical_json(t._seed_reference_path(base_dir, ARCH, ARM, SEED), 'test reference')[1]
                control_path = base_dir / reference['receipt']
                with mock.patch.object(t, 'forward', side_effect=AssertionError('control extraction repeated inference')):
                    expected = t.build_warmup_consistency_expectation(control_path)
                fresh = t.train_seed_candidate(cls.bundle, cls.inputs, ARCH, ARM, SEED,
                    cls.root / f'fresh-{weight}', ranking_weight=weight,
                    initial_checkpoint=cls.initial, qat_profile=t.WARMUP_CONSISTENCY_QAT_PROFILE,
                    warmup_consistency_expectation=expected)
                cls.base[weight], cls.fresh[weight], cls.expectations[weight] = base, fresh, expected

    def floating(self, weight):
        receipt = self.base[weight]
        parameters = t.load_float_checkpoint(self.root / f'base-{weight}' / receipt['float_checkpoint']['path'], ARCH)
        # The older scalar control has legacy reporting. The fresh scalar
        # warmup has identical numbers plus the inactive student-policy proof.
        report = receipt if weight else self.fresh[weight]
        return t.FloatTrainingResult(parameters, 1, copy.deepcopy(report['float_validation']),
                                     copy.deepcopy(report['float_training']))

    def test_real_two_arm_producers_preserve_warmup_and_bind_frozen_reference_all_epochs(self):
        for weight in (0., .1):
            with self.subTest(weight=weight):
                base, fresh = self.base[weight], self.fresh[weight]
                expected = self.expectations[weight]
                if weight:
                    self.assertEqual(base['float_training'], fresh['float_training'])
                else:
                    self.assertEqual(expected['control_profile'], t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE)
                    self.assertNotIn('training_pair_policy', base['float_training'])
                    self.assertIn('training_pair_policy', fresh['float_training'])
                    for key in ('validation', 'optimizer', 'batching', 'initialization', 'per_layer_update_evidence'):
                        self.assertEqual(base['float_training'][key], fresh['float_training'][key], key)
                self.assertEqual(base['float_checkpoint']['sha256'], fresh['float_checkpoint']['sha256'])
                self.assertNotIn('warmup_consistency', base['binding']['settings'])
                self.assertNotIn('warmup_consistency', base['quantized_training'])
                qat = fresh['quantized_training']
                self.assertEqual(qat['executed_qat_epochs'], [1, 2, 3, 4])
                self.assertEqual(qat['optimizer_steps'], 4)
                self.assertEqual(qat['scale_search']['selected_scales'], base['quantized_training']['scale_search']['selected_scales'])
                self.assertEqual(qat['pre_qat_validation'], base['quantized_training']['pre_qat_validation'])
                proof = qat['warmup_consistency']['pre_update_control_check']
                self.assertEqual(proof['initial_quantization'], expected['initial_quantization'])
                self.assertIs(proof['checked_before_optimizer_construction'], True)
                self.assertEqual(proof['optimizer_steps'], 0)
                summary = t.summarize_warmup_consistency_execution(fresh['float_training'], qat,
                    expected_binding=fresh['binding'])
                self.assertEqual(summary['coefficient'], 1.)
                self.assertEqual(summary['teacher_forward_calls'], 4)
                self.assertEqual(summary['consistency_gradient_additions'], 4)
                self.assertEqual(summary['reference_optimizer_steps'], 0)
                self.assertGreater(summary['nonzero_derivative_batches'], 0)
                reference = qat['warmup_consistency']['reference']
                for epoch in qat['history']:
                    evidence = epoch['warmup_consistency']
                    self.assertEqual(evidence['reference_before'], reference['parameters'])
                    self.assertEqual(evidence['reference_after'], reference['parameters'])
                    self.assertEqual(evidence['training_inputs'], reference['training_inputs'])
                    self.assertIs(evidence['heldout_reference_predictions'], False)
                output = self.root / f'fresh-{weight}'
                loaded = t._load_seed_receipt_from_reference(output,
                    t._seed_reference_path(output, ARCH, ARM, SEED), fresh['binding'])
                self.assertEqual(loaded, fresh)
                self.assertEqual(t.load_runtime(output / fresh['quantized_runtime']['path'])[3]['schema'], t.RUNTIME_SCHEMA)
                with mock.patch.object(t, 'train_float_seed', side_effect=AssertionError('completed receipt retrained')):
                    resumed = t.train_seed_candidate(self.bundle, self.inputs, ARCH, ARM, SEED,
                        output, ranking_weight=weight, initial_checkpoint=self.initial,
                        qat_profile=t.WARMUP_CONSISTENCY_QAT_PROFILE,
                        warmup_consistency_expectation=expected, resume=True)
                self.assertEqual(resumed, fresh)
                if weight:
                    # Selection assembly reuses this actual receipt; no extra
                    # seed or prediction work is performed by the adapter.
                    with mock.patch.object(t, 'load_training_inputs', return_value=self.inputs), \
                         mock.patch.object(t, '_train_seed_roster', return_value=[fresh]):
                        selected_path = t.train_arm_campaign(self.bundle, ARCH, ARM, output,
                            successor_labels=self.root / 'synthetic-labels.json', ranking_weight=weight,
                            initial_checkpoint=self.initial, seed_workers=2,
                            qat_profile=t.WARMUP_CONSISTENCY_QAT_PROFILE,
                            generated_source_ascii_bytes=92000)
                    selected = t.validate_selection(selected_path, output, self.bundle)
                    self.assertEqual(selected['selected_seed_receipt_body_sha256'], fresh['body_sha256'])

    def test_control_warmup_and_initial_grid_mismatches_precede_optimizer_construction(self):
        weight = .1
        expected = self.expectations[weight]
        floating = self.floating(weight)
        arguments = dict(qat_profile=t.WARMUP_CONSISTENCY_QAT_PROFILE,
                         ranking_weight=weight, warmup_consistency_expectation=expected)
        changed = self.floating(weight)
        changed.parameters['w3'][0] += np.float32(.001)
        with mock.patch.object(t, 'AdamW', side_effect=AssertionError('optimizer created before control check')):
            with self.assertRaises(t.TrainingError):
                t.run_fixed_scale_qat(changed, self.inputs, ARCH, ARM, SEED, **arguments)
        original = t.select_fixed_scales
        for change in ('codes', 'scales'):
            def wrong_grid(*args, **kwargs):
                quantized, report = original(*args, **kwargs)
                quantized = t.QuantizedWeights({name: value.copy() for name, value in quantized.integer.items()}, dict(quantized.scales))
                if change == 'codes':
                    old = int(quantized.integer['w3'][0])
                    quantized.integer['w3'][0] = old + 1 if old < 3 else old - 1
                else:
                    quantized.scales['w1'] = np.nextafter(quantized.scales['w1'], np.float32(np.inf))
                return quantized, report
            with self.subTest(change=change), t.native_thread_execution_scope(), \
                 np.errstate(divide='ignore', invalid='ignore'), \
                 mock.patch.object(t, 'select_fixed_scales', side_effect=wrong_grid), \
                 mock.patch.object(t, 'AdamW', side_effect=AssertionError('optimizer created before grid check')):
                with self.assertRaisesRegex(t.TrainingError, 'initial QAT'):
                    t.run_fixed_scale_qat(floating, self.inputs, ARCH, ARM, SEED, **arguments)

    def test_optional_control_does_not_change_algorithm_and_legacy_rejects_new_expectation(self):
        floating = self.floating(0.)
        with t.native_thread_execution_scope(), np.errstate(divide='ignore', invalid='ignore'):
            qat = t.run_fixed_scale_qat(floating, self.inputs, ARCH, ARM, SEED,
                qat_profile=t.WARMUP_CONSISTENCY_QAT_PROFILE, ranking_weight=0.)
        self.assertIsNone(qat.report['warmup_consistency']['expectation'])
        self.assertIsNone(qat.report['warmup_consistency']['pre_update_control_check'])
        expected = self.fresh[0.]['quantized_training']
        self.assertEqual(qat.metrics, expected['selected_validation'])
        self.assertEqual(qat.report['selected_scales'], expected['selected_scales'])
        self.assertEqual(qat.report['final_master_per_layer_update_evidence'], expected['final_master_per_layer_update_evidence'])
        with mock.patch.object(t, 'train_float_seed', side_effect=AssertionError('legacy control reached warmup')):
            with self.assertRaises(t.TrainingError):
                t.train_seed_candidate(None, None, ARCH, ARM, SEED, self.root / 'forbidden',
                    qat_profile=BASE, warmup_consistency_expectation=self.expectations[0.])
        self.assertFalse((self.root / 'forbidden').exists())

    def test_resealed_reference_input_and_coefficient_transplants_are_rejected(self):
        fresh = self.fresh[.1]
        for change in ('reference', 'inputs', 'coefficient', 'expectation'):
            altered = copy.deepcopy(fresh)
            qat = altered['quantized_training']
            evidence = qat['warmup_consistency']
            if change in ('reference', 'inputs'):
                reference = evidence['reference']
                if change == 'reference':
                    reference['parameters']['layers']['w3']['sha256'] = '0' * 64
                else:
                    reference['training_inputs']['new']['sha256'] = '0' * 64
                evidence['reference'] = t.body_hashed({key: value for key, value in reference.items() if key != 'body_sha256'})
            elif change == 'coefficient':
                policy = evidence['policy']
                policy['coefficient'] = .5
                evidence['policy'] = t.body_hashed({key: value for key, value in policy.items() if key != 'body_sha256'})
            else:
                evidence['expectation'] = self.expectations[0.]
            with self.subTest(change=change), self.assertRaises(t.TrainingError):
                t.summarize_warmup_consistency_execution(altered['float_training'], qat,
                    expected_binding=altered['binding'])


if __name__ == '__main__':
    unittest.main()
