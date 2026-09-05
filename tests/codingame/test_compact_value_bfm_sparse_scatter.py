"""Bitwise comparisons against the original complete first-layer scatter."""
import os
import struct
import unittest
from unittest import mock

for _name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

import numpy as np
from tools import compact_value_bfm_train as trainer
from tests.codingame.test_compact_value_bfm_training import active_row, dataset


def original_network_gradients(parameters, architecture, active, cache, output_gradient, effective):
    """Frozen pre-optimization numerical order, including zero-row scatters."""
    output_gradient = np.asarray(output_gradient, dtype=np.float32)
    first_pre, first, second_pre, second, output_pre = cache
    if output_gradient.shape != output_pre.shape:
        raise trainer.TrainingError('compact output gradient shape changed')
    output_pre_gradient = output_gradient * trainer.fast_tanh_derivative(output_pre)
    gradients = {'w3': np.asarray(second.T @ output_pre_gradient, dtype=np.float32)}
    second_gradient = (output_pre_gradient[:, None] * effective['w3'][None, :]).astype(np.float32)
    second_pre_gradient = second_gradient * trainer.second_activation_derivative(second_pre)
    gradients['w2'] = np.asarray(first.T @ second_pre_gradient, dtype=np.float32)
    first_gradient = np.asarray(second_pre_gradient @ effective['w2'].T, dtype=np.float32)
    first_pre_gradient = first_gradient * trainer.first_activation_derivative(first_pre)
    gradients['w1'] = np.zeros_like(parameters['w1'], dtype=np.float32)
    for row, indices in enumerate(active):
        np.add.at(gradients['w1'], indices, first_pre_gradient[row])
    return gradients


class AccessRows:
    def __init__(self, rows):
        self.rows = rows
        self.accesses = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        self.accesses.append(int(index))
        return row


def ranking_group(count, mover):
    successors = []
    for index in range(count):
        active = active_row(index % 7, index % 23)
        if index % 9 == 0:
            # Forward accepts repeated feature indices. Their multiplicity
            # must remain intact even though corpus encoders normally dedup.
            active = np.concatenate((active[:1], active)).astype('<u2')
        successors.append(trainer.CompleteTurnSuccessor(
            f'{mover * 1000 + index:064x}', active, -.85 + 1.7 * index / (count - 1),
            index % 2, {}))
    return trainer.CompleteTurnGroup(f'gradient-group-{mover}', mover, tuple(successors))


class SparseScatterTests(unittest.TestCase):
    def assert_bits_equal(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.tobytes(order='C'), right.tobytes(order='C'))

    def assert_gradients_equal(self, left, right):
        self.assertEqual(set(left), {'w1', 'w2', 'w3'})
        for name in left:
            with self.subTest(layer=name):
                self.assert_bits_equal(left[name], right[name])

    def assert_optimizer_equal(self, left, right):
        self.assertEqual(left.step, right.step)
        for name in left.first:
            self.assert_bits_equal(left.first[name], right.first[name])
            self.assert_bits_equal(left.second[name], right.second[name])

    def controlled_forward(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = {name: np.zeros(shape, dtype=np.float32) for name, shape in architecture.shapes.items()}
        parameters['w1'][:8, 0] = .5
        parameters['w1'][3:5, 0] = .25
        parameters['w2'][0, 0] = 1
        parameters['w3'][0] = 1
        active = tuple(np.asarray(row, dtype='<u2') for row in
            ([0], [1], [2], [3, 3], [3], [4, 4], [4, 4], [5], [5], [6], [7]))
        _prediction, cache = trainer.forward(parameters, architecture, active)
        return architecture, parameters, active, cache

    def test_signed_zero_subnormal_cancellation_duplicates_and_row_order(self):
        architecture, parameters, active, cache = self.controlled_forward()
        subnormal = np.nextafter(np.float32(0), np.float32(1))
        output = np.asarray([0., subnormal, -0., 1., -.5, 2.**20, -(2.**20),
                             2.**-20, -(2.**-20), -0., -subnormal], dtype=np.float32)
        with trainer.native_thread_execution_scope():
            reference = original_network_gradients(parameters, architecture, active, cache, output, parameters)
            tracked = AccessRows(active)
            optimized = trainer._network_gradients(parameters, architecture, tracked, cache, output, parameters)
        self.assert_gradients_equal(reference, optimized)
        self.assertEqual(tracked.accesses, sorted(tracked.accesses))
        self.assertTrue(set((0, 2, 9)).isdisjoint(tracked.accesses))
        self.assertIn(3, tracked.accesses)
        self.assertIn(4, tracked.accesses)
        magnitude = np.abs(reference['w1'])
        self.assertTrue(np.any((magnitude > 0) & (magnitude < np.finfo(np.float32).tiny)))
        self.assertTrue(np.all(reference['w1'][reference['w1'] == 0].view(np.uint32) == 0))

    def test_all_zero_rows_skip_every_scatter_and_all_live_rows_keep_order(self):
        architecture, parameters, active, cache = self.controlled_forward()
        for output in (np.asarray([0., -0.] * 5 + [0.], dtype=np.float32),
                       np.ones(len(active), dtype=np.float32)):
            with self.subTest(all_zero=bool(np.all(output == 0))):
                tracked = AccessRows(active)
                reference = original_network_gradients(parameters, architecture, active, cache, output, parameters)
                optimized = trainer._network_gradients(parameters, architecture, tracked, cache, output, parameters)
                self.assert_gradients_equal(reference, optimized)
                self.assertEqual(tracked.accesses, [] if np.all(output == 0) else list(range(len(active))))

    def test_nonfinite_computed_rows_are_not_hidden_by_zero_output_derivatives(self):
        architecture, parameters, active, cache = self.controlled_forward()
        for nonfinite in (np.nan, np.inf, -np.inf):
            changed = tuple(value.copy() for value in cache)
            changed[0][2, 0] = nonfinite
            changed[4][2] = nonfinite
            output = np.zeros(len(active), dtype=np.float32)
            tracked = AccessRows(active)
            with self.subTest(value=nonfinite), np.errstate(invalid='ignore', over='ignore'):
                reference = original_network_gradients(parameters, architecture, active, changed, output, parameters)
                optimized = trainer._network_gradients(parameters, architecture, tracked, changed, output, parameters)
            self.assert_gradients_equal(reference, optimized)
            if nonfinite != -np.inf:
                # Positive infinity makes the first derivative infinite; zero
                # times infinity is NaN and must still reach the scatter.
                self.assertIn(2, tracked.accesses)
                self.assertTrue(np.any(~np.isfinite(optimized['w1'])))

    def test_nonfinite_mixed_batch_still_rejects_before_optimizer_update(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        data = dataset([active_row(1, 2)], [.2])
        inputs = trainer.TrainingInputs(data, data, data, data, {})
        for nonfinite in (np.nan, np.inf, -np.inf):
            derivative = np.zeros(256, dtype=np.float32)
            derivative[0] = nonfinite
            for reference in (False, True):
                with self.subTest(value=nonfinite, reference=reference), np.errstate(invalid='ignore', over='ignore'):
                    parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
                    before = {name: value.copy() for name, value in parameters.items()}
                    optimizer = trainer.AdamW(parameters, learning_rate=trainer.QAT_LEARNING_RATE,
                        weight_decay=trainer.WEIGHT_DECAY)
                    original = trainer._network_gradients
                    with (mock.patch.object(trainer, 'arm_loss_gradient', return_value=(0., derivative, {})),
                          mock.patch.object(trainer, '_network_gradients',
                            side_effect=original_network_gradients if reference else original)):
                        with self.assertRaisesRegex(trainer.TrainingError, 'gradient norm is nonfinite'):
                            trainer._train_mixed_batch(parameters, architecture, trainer.ARMS['search-target'],
                                optimizer, inputs, np.zeros(64, dtype=np.int64), np.zeros(192, dtype=np.int64))
                    self.assertEqual(optimizer.step, 0)
                    self.assert_gradients_equal(before, parameters)

    def test_full_dense_gradients_remain_bitwise_equal_for_capped_ranking_pairs(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
        scales = {name: float(np.max(np.abs(value))) / 3 for name, value in parameters.items()}
        for mover in (0, 1):
            group = ranking_group(67, mover)
            active = tuple(successor.active for successor in group.successors)
            for quantized in (None, trainer.quantize_fixed(parameters, architecture, scales)):
                with self.subTest(mover=mover, qat=quantized is not None), trainer.native_thread_execution_scope():
                    predictions, cache = trainer.forward(parameters, architecture, active, quantized=quantized)
                    _loss, derivative, report = trainer.ranking_microbatch_loss_gradient((group,), predictions)
                    self.assertEqual(report['pairs'], 8)
                    effective = parameters if quantized is None else quantized.effective()
                    reference = original_network_gradients(parameters, architecture, active, cache, derivative, effective)
                    tracked = AccessRows(active)
                    optimized = trainer._network_gradients(parameters, architecture, tracked, cache, derivative, effective)
                self.assert_gradients_equal(reference, optimized)
                self.assertLessEqual(len(tracked.accesses), 9)
                self.assertEqual(tracked.accesses, sorted(tracked.accesses))

    def test_mixed_warmup_and_qat_batches_preserve_loss_parameters_moments_and_payload(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        arm = trainer.ARMS['search-target']
        seed = trainer.FIXED_SEEDS[0]
        inputs = trainer.TrainingInputs(
            new=dataset([active_row(1, 2), active_row(2, 3), active_row(3, 4)], [.2, -.3, .7]),
            anchor=dataset([active_row(4, 5), active_row(5, 6)], [-.8, .3]),
            common_adjudicator=dataset([active_row(6, 7)], [.5], split='validation'),
            canonical_validation=dataset([active_row(7, 8)], [-.5], split='validation'),
            source_routes={})
        groups = (ranking_group(37, 0), ranking_group(23, 1))
        for weight in (0., .1, .25):
            with self.subTest(weight=weight), trainer.native_thread_execution_scope():
                reference = trainer.initialize_parameters(architecture, seed)
                optimized = {name: value.copy() for name, value in reference.items()}
                for epoch in (1, 2):
                    lr = trainer.RANKING_FLOAT_LEARNING_RATE if epoch == 1 else trainer.QAT_LEARNING_RATE
                    reference_optimizer = trainer.AdamW(reference, learning_rate=lr, weight_decay=trainer.WEIGHT_DECAY)
                    optimized_optimizer = trainer.AdamW(optimized, learning_rate=lr, weight_decay=trainer.WEIGHT_DECAY)
                    scales = None if epoch == 1 else {name: float(np.max(np.abs(value))) / 3 for name, value in reference.items()}
                    new_rows, anchor_rows = next(trainer.mixed_epoch_batches(len(inputs.new), len(inputs.anchor),
                        seed=seed, epoch=epoch))
                    self.assertEqual(new_rows.shape, (64,))
                    self.assertEqual(anchor_rows.shape, (192,))
                    arguments = {'fixed_scales': scales, 'ranking_groups': groups if weight else None, 'ranking_weight': weight}
                    with mock.patch.object(trainer, '_network_gradients', side_effect=original_network_gradients):
                        reference_loss = trainer._train_mixed_batch(reference, architecture, arm, reference_optimizer,
                            inputs, new_rows, anchor_rows, **arguments)
                    optimized_loss = trainer._train_mixed_batch(optimized, architecture, arm, optimized_optimizer,
                        inputs, new_rows, anchor_rows, **arguments)
                    self.assertEqual(struct.pack('!d', reference_loss), struct.pack('!d', optimized_loss))
                    self.assert_gradients_equal(reference, optimized)
                    self.assert_optimizer_equal(reference_optimizer, optimized_optimizer)
                    if scales is not None:
                        left = trainer.quantize_fixed(reference, architecture, scales)
                        right = trainer.quantize_fixed(optimized, architecture, scales)
                        self.assertEqual(trainer.pack_signed_three_bit(trainer._flatten_quantized(left, architecture)),
                            trainer.pack_signed_three_bit(trainer._flatten_quantized(right, architecture)))


if __name__ == '__main__':
    unittest.main()
