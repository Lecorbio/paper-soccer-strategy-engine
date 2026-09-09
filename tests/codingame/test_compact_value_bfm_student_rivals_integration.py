"""Independent prospective pair-policy integration; synthetic inputs only."""
import ast
import copy
import dataclasses
import hashlib
import math
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

for _key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

import numpy as np
from tools import compact_value_bfm_train as t
from tests.codingame import test_compact_value_bfm_deterministic_best as small

BASE = t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE
PROFILE = t.STUDENT_RIVALS_RETENTION_PROFILE
POLICY = t.STUDENT_RIVAL_PAIR_POLICY
ARCH = t.ARCHITECTURES['capacity-12x8']
ARM = t.ARMS['search-target']
SEED = 20260907
LAYERS = ('w1', 'w2', 'w3')


def dataset(count, offset=0):
    return t.Dataset(
        np.arange(count + 1, dtype=np.int64),
        (np.arange(count, dtype=np.uint16) % 12 + offset).astype('<u2'),
        np.linspace(-.8, .9, count, dtype=np.float32),
        np.linspace(.5, 1.5, count, dtype=np.float32),
        np.asarray([hashlib.sha256(str(i + offset).encode()).digest()
                    for i in range(count)], dtype='V32'),
        'train', 'a' * 64, 'b' * 64)


def controlled_inputs(same_subset=False):
    parameters = {name: np.zeros(shape, dtype=np.float32)
                  for name, shape in ARCH.shapes.items()}
    for index in range(12):
        parameters['w1'][index, 0] = np.float32(
            .29 - index * .002 if same_subset and 1 <= index <= 8
            else .01 if same_subset else (index + 1) * .024)
    parameters['w2'][0, 0] = .5
    parameters['w3'][0] = .5
    group = t.CompleteTurnGroup('integration-group', 1, tuple(
        t.CompleteTurnSuccessor(
            f'{100 - index:064x}', np.asarray([index], dtype='<u2'),
            float(np.float32(1. if index == 0 else .07 * index)) * (-1. if index == 10 else 1.),
            0 if index == 10 else 1, {})
        for index in range(12)))
    inputs = type('ScalarInputs', (), {'new': dataset(64), 'anchor': dataset(192)})()
    return parameters, inputs, group


def independent_pair_objective(group, predictions, *, student):
    """Explicit mathematical oracle, without calling either production selector."""
    signs = np.asarray([1. if s.value_mover == group.parent_mover else -1.
                        for s in group.successors], dtype=np.float32)
    teacher = np.asarray([s.teacher_value for s in group.successors], np.float32) * signs
    scores = np.asarray(predictions, np.float32) * signs
    ids = [s.successor_id for s in group.successors]
    best = min(range(len(ids)), key=lambda i: (-float(teacher[i]), ids[i], i))
    eligible = [i for i in range(len(ids)) if float(np.float32(teacher[best] - teacher[i])) > 0]
    if student:
        selected = sorted(eligible, key=lambda i: (-float(scores[i]), ids[i], i))[:8]
    else:
        selected = sorted(eligible, key=lambda i: (-float(np.float32(teacher[best] - teacher[i])), ids[i], i))[:8]
    selected.sort(key=lambda i: (-float(np.float32(teacher[best] - teacher[i])), ids[i], i))
    gaps = np.asarray([teacher[best] - teacher[i] for i in selected], np.float32)
    weights = gaps / np.float32(np.sum(gaps, dtype=np.float64))
    derivative = np.zeros(len(ids), np.float32)
    loss = 0.
    for weight, index in zip(weights, selected):
        margin = float(np.float32(scores[best] - scores[index]))
        loss += float(weight) * float(np.logaddexp(0., -margin))
        value = np.float32(float(weight) * (-1. / (1. + math.exp(margin))))
        derivative[best] += value
        derivative[index] -= value
    return loss, derivative * signs, selected


class Capture:
    def update(self, parameters, gradients):
        self.gradients = {name: value.copy() for name, value in gradients.items()}


def independent_mixed_gradient(parameters, inputs, group, weight, scales):
    new, anchor = np.arange(64), np.arange(192)
    quantized = t.quantize_fixed(parameters, ARCH, scales) if scales is not None else None
    effective = quantized.effective() if quantized is not None else parameters
    active = (*inputs.new.active_rows(new), *inputs.anchor.active_rows(anchor))
    scalar, cache = t.forward(parameters, ARCH, active, quantized=quantized)
    targets = np.concatenate((inputs.new.targets[new], inputs.anchor.targets[anchor]))
    weights = t.independently_normalized_mixed_weights(inputs.new.weights[new], inputs.anchor.weights[anchor])
    scalar_loss, derivative, _ = t.arm_loss_gradient(ARM, scalar, targets, weights)
    gradients = t._network_gradients(parameters, ARCH, active, cache, derivative, effective)
    actions = tuple(s.active for s in group.successors)
    predictions, ranking_cache = t.forward(parameters, ARCH, actions, quantized=quantized)
    ranking_loss, output_gradient, selected = independent_pair_objective(group, predictions, student=True)
    ranking_gradient = t._network_gradients(parameters, ARCH, actions, ranking_cache,
                                          output_gradient * np.float32(weight), effective)
    for name in gradients:
        gradients[name] += ranking_gradient[name]
    length = math.sqrt(sum(float(np.sum(g * g, dtype=np.float64)) for g in gradients.values()))
    if length > 5:
        for name in gradients:
            gradients[name] *= np.float32(5. / length)
    return scalar_loss + weight * ranking_loss, gradients, selected


class StudentRivalIntegrationTests(unittest.TestCase):
    def assert_array_bits(self, left, right):
        self.assertEqual((left.dtype, left.shape), (right.dtype, right.shape))
        self.assertEqual(left.tobytes(), right.tobytes())

    def test_full_mixed_gradient_and_adam_follow_independent_rival_objective(self):
        parameters, inputs, group = controlled_inputs()
        for scales in (None, {'w1': .1, 'w2': .25, 'w3': .25}):
            with self.subTest(scales=scales):
                expected_loss, expected, selected = independent_mixed_gradient(parameters, inputs, group, .1, scales)
                old = t._ranking_pairs(group)[1]
                self.assertNotEqual(set(selected), set(old))
                capture = Capture()
                actual_loss = t._train_mixed_batch(parameters, ARCH, ARM, capture, inputs,
                    np.arange(64), np.arange(192), fixed_scales=scales,
                    ranking_groups=(group,), ranking_weight=.1, training_pair_policy=POLICY)
                self.assertEqual(actual_loss, expected_loss)
                for name in LAYERS:
                    self.assert_array_bits(capture.gradients[name], expected[name])
                left = {n: v.copy() for n, v in parameters.items()}
                right = {n: v.copy() for n, v in parameters.items()}
                t.AdamW(left, learning_rate=.0000625, weight_decay=1e-5).update(left, expected)
                t._train_mixed_batch(right, ARCH, ARM,
                    t.AdamW(right, learning_rate=.0000625, weight_decay=1e-5), inputs,
                    np.arange(64), np.arange(192), fixed_scales=scales,
                    ranking_groups=(group,), ranking_weight=.1, training_pair_policy=POLICY)
                for name in LAYERS:
                    self.assert_array_bits(left[name], right[name])

    def test_same_subset_keeps_exact_full_gradient_reduction_order(self):
        parameters, inputs, group = controlled_inputs(same_subset=True)
        for scales in (None, {'w1': .1, 'w2': .25, 'w3': .25}):
            quantized = t.quantize_fixed(parameters, ARCH, scales) if scales else None
            predictions = t.forward(parameters, ARCH, tuple(s.active for s in group.successors), quantized=quantized)[0]
            selected = independent_pair_objective(group, predictions, student=True)[2]
            self.assertEqual(selected, list(t._ranking_pairs(group)[1]))
            captures = []
            for policy in (None, POLICY):
                capture = Capture()
                loss = t._train_mixed_batch(parameters, ARCH, ARM, capture, inputs,
                    np.arange(64), np.arange(192), fixed_scales=scales,
                    ranking_groups=(group,), ranking_weight=.25, training_pair_policy=policy)
                captures.append((loss, capture.gradients))
            self.assertEqual(captures[0][0], captures[1][0])
            for name in LAYERS:
                self.assert_array_bits(captures[0][1][name], captures[1][1][name])

    def test_validation_remains_static_even_when_student_training_would_differ(self):
        parameters, _, group = controlled_inputs()
        prediction = t.forward(parameters, ARCH, tuple(s.active for s in group.successors))[0]
        expected_loss = independent_pair_objective(group, prediction, student=False)[0]
        dynamic_loss = independent_pair_objective(group, prediction, student=True)[0]
        self.assertNotEqual(expected_loss, dynamic_loss)
        with mock.patch.object(t, '_student_ranking_pairs', side_effect=AssertionError('TRAIN selector reached validation')):
            report = t.successor_ranking_metrics(parameters, ARCH, (group,))
        self.assertEqual(report['pairwise_loss'], expected_loss)
        self.assertEqual(report['pairs'], 8)

    def test_historical_numerical_and_validation_kernels_match_8856_asts(self):
        # Captured from immutable 8856daf; no ignored snapshot needed by CI.
        expected = {
            '_network_gradients': '8aee72986955b3c86e3f77a7aab6a5449b32bea90b3ac95647764f2d8d709251',
            'AdamW': 'daaaf670549f198b3aff8cba451af9b4919e4376e88dd0f731646c4a8f19aad9',
            'pairwise_successor_ranking_loss_gradient': '5356f2625cfbbbb1819cac1dbff5b25f50092d451f94980ac89f652534633766',
            'successor_ranking_metrics': 'bafbe63c69faa18d1581692779dc8b9120a62c8af0c2a2c097a8aa685ff8e14c',
            'mixed_epoch_batches': '328ef6fe1ac9cbd4954cad85ddc3aca7b35ff6269b40d0547dee5d0a6b941f68',
            'successor_ranking_epoch_schedule': '21137e94cd40c464f69c9a10f346a3ced5626e60965ec3e8d94ece1fe32ad3b8',
        }
        nodes = {n.name: n for n in ast.parse(Path(t.__file__).read_text()).body if hasattr(n, 'name')}
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest(), digest)

    def test_real_seed_producer_receipt_runtime_and_immutable_selection_roundtrip(self):
        parameters, scalar_inputs, group = controlled_inputs()
        labels = t.SuccessorRankingLabels(
            train=(group,), validation=(group,), teacher={'artifact_sha256': '1' * 64},
            source_bundle_body_sha256='a' * 64, artifact_sha256='b' * 64, body_sha256='c' * 64)
        inputs = t.TrainingInputs(
            new=scalar_inputs.new, anchor=scalar_inputs.anchor,
            common_adjudicator=dataclasses.replace(dataset(4096), split='validation'),
            canonical_validation=dataclasses.replace(dataset(9), split='validation'),
            source_routes={}, successor_rankings=labels)
        bundle = SimpleNamespace(body_sha256='a' * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = t.write_float_checkpoint(root / 'initial', parameters, ARCH)
            output = root / 'seed'
            # All optimizer, scale selection, validation and4096 parity math is real;
            # only selection's roster/input assembly below reuses this single receipt.
            # Constant trial predictions have undefined correlation; the maintained
            # metric function maps it to0. Suppress that expected NumPy warning.
            with np.errstate(divide='ignore', invalid='ignore'):
                receipt = t.train_seed_candidate(bundle, inputs, ARCH, ARM, SEED, output,
                    ranking_weight=.1, initial_checkpoint=initial, qat_profile=PROFILE)
            self.assertEqual(receipt['quantized_training']['executed_qat_epochs'], [1, 2, 3, 4])
            self.assertEqual(receipt['qat_profile'], PROFILE)
            self.assertEqual(receipt['quantized_training']['optimizer_steps'], 4)
            reference = t._seed_reference_path(output, ARCH, ARM, SEED)
            with mock.patch.object(t, '_validate_student_receipt_artifacts', wraps=t._validate_student_receipt_artifacts) as artifact_checks:
                loaded = t._load_seed_receipt_from_reference(output, reference, receipt['binding'])
                self.assertEqual(loaded, receipt)
                with mock.patch.object(t, 'load_training_inputs', return_value=inputs), \
                        mock.patch.object(t, '_train_seed_roster', return_value=[receipt]):
                    selection_path = t.train_arm_campaign(bundle, ARCH, ARM, output,
                        successor_labels=root / 'synthetic-label-binding.json', ranking_weight=.1,
                        initial_checkpoint=initial, seed_workers=2, qat_profile=PROFILE,
                        generated_source_ascii_bytes=92000)
                selected = t.validate_selection(selection_path, output, bundle)
                self.assertGreaterEqual(artifact_checks.call_count, 3)
            self.assertEqual(selected['selected_seed_receipt_body_sha256'], receipt['body_sha256'])
            self.assertEqual(selected['qat_profile'], PROFILE)
            runtime = output / receipt['quantized_runtime']['path']
            self.assertEqual(t.load_runtime(runtime)[3]['schema'], t.RUNTIME_SCHEMA)


class StudentRivalInactiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = small.inputs(small.ranking_groups())
        initial = t.initialize_parameters(ARCH, SEED)
        cls.results = {}
        cls.master_traces = {}
        original_update = t.AdamW.update
        with t.native_thread_execution_scope(), mock.patch.object(t, '_student_ranking_pairs', side_effect=AssertionError('lambda0 selected rivals')):
            for profile, policy in ((BASE, None), (PROFILE, POLICY)):
                trace = []
                def observe_update(optimizer, parameters, gradients):
                    original_update(optimizer, parameters, gradients)
                    trace.append((float(optimizer.learning_rate), optimizer.step,
                                  tuple(parameters[n].tobytes() for n in LAYERS)))
                with mock.patch.object(t.AdamW, 'update', observe_update):
                    floating = t.train_float_seed(cls.inputs, ARCH, ARM, SEED,
                        maximum_epochs=1, patience=1, learning_rate=.00006,
                        initial_parameters={n: v.copy() for n, v in initial.items()}, ranking_weight=0.,
                        training_pair_policy=policy)
                    qat = t.run_fixed_scale_qat(floating, cls.inputs, ARCH, ARM, SEED,
                        ranking_weight=0., qat_profile=profile)
                cls.results[profile] = floating, qat
                cls.master_traces[profile] = trace

    def test_lambda_zero_full_one_plus_four_is_numerically_identical(self):
        old_float, old_qat = self.results[BASE]
        new_float, new_qat = self.results[PROFILE]
        self.assertEqual(old_float.metrics, new_float.metrics)
        self.assertEqual(old_qat.metrics, new_qat.metrics)
        self.assertEqual((old_float.epoch, old_qat.qat_epoch), (new_float.epoch, new_qat.qat_epoch))
        self.assertEqual(new_qat.report['executed_qat_epochs'], [1, 2, 3, 4])
        self.assertEqual(old_qat.report['applied_scale_trajectory'], new_qat.report['applied_scale_trajectory'])
        self.assertEqual(old_qat.quantized.scales, new_qat.quantized.scales)
        self.assertEqual(len(self.master_traces[BASE]), 5)  # One real warmup update and four real QAT updates.
        self.assertEqual(self.master_traces[BASE], self.master_traces[PROFILE])
        for name in LAYERS:
            self.assertEqual(old_float.parameters[name].tobytes(), new_float.parameters[name].tobytes())
            self.assertEqual(old_qat.quantized.integer[name].tobytes(), new_qat.quantized.integer[name].tobytes())
        self.assertEqual(old_qat.report['retention_reference'], new_qat.report['retention_reference'])
        self.assertEqual(old_qat.report['pre_qat_validation'], new_qat.report['pre_qat_validation'])
        self.assertEqual([r['validation'] for r in old_qat.report['history']], [r['validation'] for r in new_qat.report['history']])

    def test_lambda_zero_runtime_v1_checkpoint_bytes_and_evidence_are_separate(self):
        artifacts = []
        with tempfile.TemporaryDirectory() as temporary:
            for profile in (BASE, PROFILE):
                floating, qat = self.results[profile]
                root = Path(temporary) / profile
                checkpoint = t.write_float_checkpoint(root / 'float', floating.parameters, ARCH)
                runtime = t.write_runtime(root / 'runtime', ARCH, qat.quantized, arm=ARM, seed=SEED,
                    float_epoch=floating.epoch, qat_epoch=qat.qat_epoch, source_bundle_body_sha256='a' * 64)
                arch, quantized, selection, document = t.load_runtime(runtime)
                self.assertEqual(document['schema'], t.RUNTIME_SCHEMA)
                self.assertNotIsInstance(quantized, t.ChannelQuantizedWeights)
                self.assertEqual((selection['seed'], selection['arm']), (SEED, 'search-target'))
                artifacts.append((checkpoint.read_bytes(), runtime.read_bytes()))
            self.assertEqual(artifacts[0], artifacts[1])
        floating, qat = self.results[PROFILE]
        summary = t.validate_student_rival_execution(floating.report, qat.report, seed=SEED)
        self.assertEqual(summary['validated_epoch_reports'], 5)
        self.assertEqual(summary['group_evaluations'], 0)
        self.assertEqual(summary['selected_pairs'], 0)
        self.assertFalse(summary['independent_numerical_replay_claimed'])
        self.assertIsNone(t.training_pair_policy_contract(BASE))
        self.assertNotEqual(t.qat_profile_contract(BASE), t.qat_profile_contract(PROFILE))

    def test_outer_policy_seed_initialization_and_warmup_transplants_reject(self):
        floating, qat = self.results[PROFILE]
        policy = t.training_pair_policy_contract(PROFILE)
        binding = {'seed': SEED, 'settings': {'qat_profile': PROFILE,
            'qat_profile_contract': t.qat_profile_contract(PROFILE), 'training_pair_policy': policy},
            'successor_ranking': {'training_pair_policy': policy, 'loss_weight': 0.,
                'initial_checkpoint': {'parameters': floating.report['initialization']['parameters']},
                'float_warmup': {'epochs': 1, 'learning_rate': .00006}}}
        t.validate_student_rival_execution(floating.report, qat.report, seed=SEED, expected_binding=binding)
        for field in ('seed', 'policy', 'initialization'):
            changed = copy.deepcopy(binding)
            if field == 'seed': changed['seed'] += 1
            if field == 'policy': changed['settings']['training_pair_policy'] = None
            if field == 'initialization': changed['successor_ranking']['initial_checkpoint']['parameters']['layers']['w3']['sha256'] = '0' * 64
            with self.subTest(field=field), self.assertRaises(t.TrainingError):
                t.validate_student_rival_execution(floating.report, qat.report, seed=SEED, expected_binding=changed)
        forged = copy.deepcopy(qat.report)
        forged['float_warmup_report_sha256'] = '0' * 64
        with self.assertRaises(t.TrainingError):
            t.validate_student_rival_execution(floating.report, forged, seed=SEED, expected_binding=binding)

    def test_phase_and_epoch_are_bound_even_when_stream_digest_is_unchanged(self):
        floating, qat = self.results[PROFILE]
        original = qat.report['history'][0]['training_pair_selection']
        for field, changed in (('phase', 'float-warmup'), ('schedule_epoch', 4)):
            for reseal in (False, True):
                forged = copy.deepcopy(qat.report)
                epoch = forged['history'][0]['training_pair_selection']
                epoch[field] = changed
                if reseal:
                    forged['history'][0]['training_pair_selection'] = t.body_hashed(
                        {key: value for key, value in epoch.items() if key != 'body_sha256'})
                self.assertEqual(epoch['ordered_pair_prediction_sha256'], original['ordered_pair_prediction_sha256'])
                with self.subTest(field=field, reseal=reseal), self.assertRaises(t.TrainingError):
                    t.validate_student_rival_execution(floating.report, forged, seed=SEED)


if __name__ == '__main__':
    unittest.main()
