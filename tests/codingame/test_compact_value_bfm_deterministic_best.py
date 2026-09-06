"""Exact selector parity against the original Python key over every successor."""
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
from tools import compact_value_bfm_train as trainer
from tests.codingame import test_compact_value_bfm_training as fixtures
from tests.codingame import test_jacek_replay_corpus as corpus_fixtures


def original_best(group, values):
    """Unchanged selector from the frozen 7d trainer, including validation."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(group.successors),) or not np.all(np.isfinite(values)):
        raise trainer.TrainingError('successor ranking best-action values are invalid')
    return min(range(len(group.successors)), key=lambda index: (
        -float(values[index]), group.successors[index].successor_id))


class IDs:
    def __init__(self, values): self.values, self.accesses = values, []
    def __len__(self): return len(self.values)
    def __getitem__(self, index):
        if type(index) is not int: raise AssertionError('selector index must remain a Python int')
        self.accesses.append(index)
        return SimpleNamespace(successor_id=self.values[index])


def id_group(ids): return SimpleNamespace(successors=IDs(ids))


def ranking_groups():
    groups = []
    for parent in (0, 1):
        successors = tuple(trainer.CompleteTurnSuccessor(
            f'{(13 - index) % 9:064x}', fixtures.active_row(index % 8, index + 1),
            (.8, .8, .2, -.5, -1., 1.)[index % 6], index % 2,
            {'proof': {'solved': index % 6 in (4, 5)}}) for index in range(17))
        groups.append(trainer.CompleteTurnGroup(f'parent-{parent}', parent, successors))
    return tuple(groups)


def inputs(groups):
    labels = trainer.SuccessorRankingLabels(train=groups, validation=groups,
        teacher={'artifact_sha256': '1' * 64}, source_bundle_body_sha256='a' * 64,
        artifact_sha256='b' * 64, body_sha256='c' * 64)
    return trainer.TrainingInputs(
        new=fixtures.dataset([fixtures.active_row(i, i) for i in range(4)], [.2, -.3, .8, -.4]),
        anchor=fixtures.dataset([fixtures.active_row(i + 2, i + 4) for i in range(3)], [-.7, .4, .1]),
        common_adjudicator=fixtures.dataset([fixtures.active_row(5, 6), fixtures.active_row(6, 7)], [.7, -.8], split='validation'),
        canonical_validation=fixtures.dataset([fixtures.active_row(7, 8), fixtures.active_row(8, 9)], [-.4, .9], split='validation'),
        source_routes={}, successor_rankings=labels)


class DeterministicBestTests(unittest.TestCase):
    def assert_bits_equal(self, left, right):
        self.assertEqual((left.dtype, left.shape), (right.dtype, right.shape))
        self.assertEqual(left.tobytes(), right.tobytes())

    def assert_json_equal(self, left, right):
        self.assertEqual(trainer.canonical_json_bytes(left), trainer.canonical_json_bytes(right))

    def test_randomized_selector_parity_and_return_type(self):
        rng = np.random.default_rng(641923)
        for case in range(400):
            count = int(rng.integers(1, 258))
            ids = [f'{int(value):064x}' for value in rng.integers(0, max(2, count // 2), count)]
            if case % 3 == 0:
                values = rng.integers(-4, 5, count).astype(np.float32)
            elif case % 3 == 1:
                values = rng.normal(size=count)  # Normalization must precede selection.
            else:
                values = rng.integers(0, 1 << 32, count, dtype=np.uint32).view(np.float32)
                values[~np.isfinite(values)] = 0
            group = id_group(ids)
            with self.subTest(case=case):
                expected = original_best(group, values)
                actual = trainer._deterministic_best(group, values)
                self.assertEqual(actual, expected); self.assertIs(type(actual), int)

    def test_unsorted_and_duplicate_ids_keep_exact_tie_rules(self):
        for values, ids, expected in (([2., 2., 1.], ['z', 'a', '0'], 1),
                ([2., 2., 2.], ['b', 'a', 'a'], 1),
                ([-2., -2., -3.], ['z', 'a', '0'], 1),
                ([.5], ['only'], 0)):
            with self.subTest(values=values, ids=ids):
                self.assertEqual(trainer._deterministic_best(id_group(ids), values), expected)
                self.assertEqual(trainer._deterministic_best(id_group(ids), values), original_best(id_group(ids), values))

    def test_signed_zeros_subnormals_and_finite_extremes(self):
        cases = [np.asarray([0., -0., 0.], dtype=np.float32),
            np.asarray([1, 2, 0, 0x80000001, 2], dtype=np.uint32).view(np.float32),
            np.asarray([0x80000002, 0x80000001, 0x80000001], dtype=np.uint32).view(np.float32),
            np.asarray([np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32)]
        for values in cases:
            ids = [f'{len(values) - index:064x}' for index in range(len(values))]
            with self.subTest(bits=values.view(np.uint32).tolist()):
                self.assertEqual(trainer._deterministic_best(id_group(ids), values), original_best(id_group(ids), values))
        self.assertEqual(trainer._deterministic_best(id_group(['z', 'a']), [+0., -0.]), 1)
        self.assertEqual(trainer._deterministic_best(id_group(['a', 'z']),
            np.asarray([1, 2], dtype=np.uint32).view(np.float32)), 1)

    def test_float32_normalization_creates_ties_before_selection(self):
        values = np.asarray([1. + 2.**-25, 1.], dtype=np.float64)
        self.assertGreater(values[0], values[1])
        self.assertEqual(trainer._deterministic_best(id_group(['z', 'a']), values), 1)
        self.assertEqual(trainer._deterministic_best(id_group(['z', 'a']), values), original_best(id_group(['z', 'a']), values))

    def test_noncontiguous_readonly_inputs_are_unchanged(self):
        original = np.asarray([3., 1., 7., 9., 7., 2.], dtype=np.float32)
        for values in (original[::2], original[::-1]):
            values.flags.writeable = False; before = values.tobytes()
            group = id_group([f'{i:064x}' for i in reversed(range(len(values)))])
            self.assertEqual(trainer._deterministic_best(group, values), original_best(group, values))
            self.assertEqual(values.tobytes(), before)

    def test_invalid_values_preserve_errors_and_never_materialize_ids(self):
        for values in ([np.nan, 1.], [np.inf, 1.], [-np.inf, 1.], [1.], [[1., 2.]],
                       ['invalid', 'value'], [object(), 1.], [1e100, 1.], None):
            group = id_group(['a', 'b'])
            with self.subTest(values=str(values)), np.errstate(over='ignore', invalid='ignore'):
                try: original_best(group, values)
                except Exception as expected:
                    with mock.patch.object(trainer.np, 'max', side_effect=AssertionError('validation must precede max')):
                        with self.assertRaises(type(expected)) as actual: trainer._deterministic_best(group, values)
                    self.assertEqual(str(actual.exception), str(expected))
                else: self.fail('invalid reference input was accepted')
                self.assertEqual(group.successors.accesses, [])

    def test_empty_input_preserves_builtin_min_exception(self):
        group = id_group([])
        for values in ([], np.asarray([], dtype=np.float64)):
            with self.subTest(dtype=np.asarray(values).dtype):
                with self.assertRaises(ValueError) as expected: original_best(group, values)
                with mock.patch.object(trainer.np, 'max', side_effect=AssertionError('empty input must not call max')):
                    with self.assertRaises(ValueError) as actual: trainer._deterministic_best(group, values)
                self.assertEqual(str(actual.exception), str(expected.exception))

    def test_only_ascending_maxima_materialize_ids(self):
        group = id_group(['z', 'b', '0', 'a', 'a', '0'])
        values = np.asarray([0, 4, -1, 4, 4, 2], dtype=np.float32)
        self.assertEqual(trainer._deterministic_best(group, values), 3)
        self.assertEqual(group.successors.accesses, [1, 3, 4])

    def test_actual_mapped_successor_materializations_are_reduced(self):
        from tools import compact_value_bfm_ranking_store as store
        row = corpus_fixtures.JacekReplayCorpusTests.complete_turn_action_group_row(root_action='0')
        row['source_bundle_body_sha256'] = 'a' * 64
        prototype = row['group']['successors'][0]; successors = []
        for ordinal, physical in enumerate(('0', '1', '2', '3', '5', '6', '7')):
            state = store.corpus._prefix_state(row['group']['source_binding']['prefix'])
            store.corpus.features.apply_complete_turn(state, 1, physical)
            successors.append({**prototype, 'successor_id': store.corpus._mover_canonical_position_identity(state),
                'active': list(store.corpus.features.encode_active(state)), 'value_mover': state.to_move,
                'transcript': str((int(physical) + 4) % 8), 'teacher_value': -.1 * (ordinal + 1)})
        row['group']['successors'] = sorted(successors, key=lambda value: value['successor_id'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'labels.jsonl'
            source.write_bytes(store.corpus.canonical_json_bytes(row)); bundle = fixtures.bundle_fixture(root)
            group = store.RankingStore(store.build_store([source], root / 'store', bundle), bundle).labels().train[0]
            getter = store.MappedSuccessors.__getitem__; accesses = []
            def observed(self, index): accesses.append(int(index)); return getter(self, index)
            for values, expected_accesses in (([0., 1., 0., -1., 2., 0., 0.], [4]),
                    ([0., 2., 0., -1., 2., 0., 0.], [1, 4]), ([0.] * 7, list(range(7)))):
                with self.subTest(values=values), mock.patch.object(store.MappedSuccessors, '__getitem__', observed):
                    accesses.clear(); expected = original_best(group, values)
                    self.assertEqual(accesses, list(range(7)))
                    accesses.clear(); actual = trainer._deterministic_best(group, values)
                    self.assertEqual(actual, expected); self.assertEqual(accesses, expected_accesses)

    def test_full_metrics_and_pairwise_microbatch_gradients_are_bitwise_equal(self):
        groups = ranking_groups(); data = inputs(groups)
        architecture = trainer.ARCHITECTURES['capacity-12x8']; arm = trainer.ARMS['search-target']
        parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
        scales = {name: float(np.max(np.abs(value))) / 3 for name, value in parameters.items()}
        predictions = np.asarray([0., -0., .4, -.4, .4, .2] * 6, dtype=np.float32)[:34]
        with mock.patch.object(trainer, '_deterministic_best', original_best):
            old_loss, old_gradient, old_report = trainer.ranking_microbatch_loss_gradient(groups, predictions)
        loss, gradient, report = trainer.ranking_microbatch_loss_gradient(groups, predictions)
        self.assertEqual(loss.hex(), old_loss.hex()); self.assert_bits_equal(gradient, old_gradient)
        self.assert_json_equal(report, old_report)
        for weight in (0., .1, .25):
            for factor in (.8, 1., 1.2):
                quantized = trainer.quantize_fixed(parameters, architecture, {name: value * factor for name, value in scales.items()})
                with self.subTest(weight=weight, factor=factor), trainer.native_thread_execution_scope():
                    with mock.patch.object(trainer, '_deterministic_best', original_best):
                        old = trainer.evaluate_validation_pair(parameters, architecture, data, arm, quantized=quantized, ranking_weight=weight)
                    actual = trainer.evaluate_validation_pair(parameters, architecture, data, arm, quantized=quantized, ranking_weight=weight)
                    self.assert_json_equal(actual, old)

    def test_complete_scale_search_reports_scales_and_codes_match(self):
        groups = ranking_groups(); data = inputs(groups)
        architecture = trainer.ARCHITECTURES['capacity-12x8']; arm = trainer.ARMS['search-target']
        parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
        for profile in (trainer.STANDARD_QAT_PROFILE, trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE):
            for weight in (0., .1, .25):
                with self.subTest(profile=profile, weight=weight), trainer.native_thread_execution_scope():
                    with mock.patch.object(trainer, '_deterministic_best', original_best):
                        old, old_report = trainer.select_fixed_scales(parameters, architecture, data, arm, ranking_weight=weight, qat_profile=profile)
                    actual, report = trainer.select_fixed_scales(parameters, architecture, data, arm, ranking_weight=weight, qat_profile=profile)
                self.assert_json_equal(report, old_report)
                for name in ('w1', 'w2', 'w3'):
                    self.assert_bits_equal(actual.integer[name], old.integer[name])
                    self.assertEqual(actual.scales[name].tobytes(), old.scales[name].tobytes())

    def test_warmup_and_qat_mixed_batch_gradients_and_updates_match(self):
        data = inputs(ranking_groups()); architecture = trainer.ARCHITECTURES['capacity-12x8']; arm = trainer.ARMS['search-target']
        current_selector = trainer._deterministic_best
        for weight in (0., .1, .25):
            outcomes = []
            for selector in (original_best, current_selector):
                parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0]); captured = []
                optimizer = trainer.AdamW(parameters, learning_rate=trainer.QAT_LEARNING_RATE, weight_decay=trainer.WEIGHT_DECAY)
                update = optimizer.update
                def capture(values, gradients):
                    captured.append({name: value.copy() for name, value in gradients.items()}); return update(values, gradients)
                with mock.patch.object(trainer, '_deterministic_best', selector), mock.patch.object(optimizer, 'update', side_effect=capture), trainer.native_thread_execution_scope():
                    for qat in (False, True):
                        scales = {name: float(np.max(np.abs(value))) / 3 for name, value in parameters.items()} if qat else None
                        trainer._train_mixed_batch(parameters, architecture, arm, optimizer, data,
                            np.resize(np.arange(len(data.new)), 64), np.resize(np.arange(len(data.anchor)), 192),
                            fixed_scales=scales, ranking_groups=data.successor_rankings.train if weight else None, ranking_weight=weight)
                outcomes.append((parameters, captured, optimizer))
            with self.subTest(weight=weight):
                for name in ('w1', 'w2', 'w3'):
                    self.assert_bits_equal(outcomes[0][0][name], outcomes[1][0][name])
                    for old, actual in zip(outcomes[0][1], outcomes[1][1], strict=True): self.assert_bits_equal(old[name], actual[name])
                    self.assert_bits_equal(outcomes[0][2].first[name], outcomes[1][2].first[name])
                    self.assert_bits_equal(outcomes[0][2].second[name], outcomes[1][2].second[name])


if __name__ == '__main__': unittest.main()
