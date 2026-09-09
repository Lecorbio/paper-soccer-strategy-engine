"""Independent scalar-validation acceleration checks; synthetic inputs only."""
import ast
import copy
import dataclasses
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

for _key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

import numpy as np
from tools import compact_value_bfm_train as t
from tools import compact_value_bfm_prediction_pool as pools
from tests.codingame import test_compact_value_bfm_student_rivals_integration as fixtures

ARCH = t.ARCHITECTURES['capacity-12x8']
ARM = t.ARMS['search-target']
SEED = 20260907
LAYERS = ('w1', 'w2', 'w3')


def tiny_inputs():
    parameters, scalars, group = fixtures.controlled_inputs()
    labels = t.SuccessorRankingLabels(
        train=(group,), validation=(group,), teacher={'artifact_sha256': '1' * 64},
        source_bundle_body_sha256='a' * 64, artifact_sha256='b' * 64,
        body_sha256='c' * 64)
    inputs = t.TrainingInputs(
        new=scalars.new, anchor=scalars.anchor,
        common_adjudicator=dataclasses.replace(fixtures.dataset(7), split='validation'),
        canonical_validation=dataclasses.replace(fixtures.dataset(11), split='validation'),
        source_routes={}, successor_rankings=labels)
    return parameters, inputs


class HistoricalScalarValidationTests(unittest.TestCase):
    def test_opt_in_does_not_change_serial_prediction_metric_math_or_learning_contracts(self):
        # These are exact UTF-8 definition segments from published e2f1bfb.
        # AST is only a locator, avoiding Python-version-specific ast.dump text.
        expected = {
            'predict_dataset': '6fddfef0c44c1aec792ce0d017b25fc01d4ddae3f6542149de7129c9c5199a48',
            'metrics_from_predictions': '4d9c79bf43246fb5f6bbad92d9e2580b5d5e1272a8073464c1f5023ed73b57ba',
        }
        source = Path(t.__file__).read_text(encoding='utf-8')
        nodes = {node.name: node for node in ast.parse(source).body if hasattr(node, 'name')}
        for name, digest in expected.items():
            with self.subTest(definition=name):
                self.assertFalse(nodes[name].decorator_list)
                segment = ast.get_source_segment(source, nodes[name])
                self.assertIsNotNone(segment)
                self.assertEqual(hashlib.sha256(segment.encode()).hexdigest(), digest)
        contracts = {
            'standard-v1': '9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
            'refined-adaptive-scales-v1': '7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd',
            'retention-first-low-rate-v1': '3cdbd16892860b6db547ca510e97604ba7742ae07392df64fd817b315b975de3',
            'channel-prediction-qat-v1': '4ae78b345fb3f586d0b590a04127993aa9880c6e1cec265191b7db7398cafaa0',
            'student-rivals-retention-v1': 'a41b6afc383ada5d977e2673fee9344e296b1f156b0627457cacc534006ea653',
        }
        for profile, digest in contracts.items():
            with self.subTest(profile=profile):
                self.assertEqual(t.qat_profile_contract(profile)['body_sha256'], digest)


class PredictionRefreshIntegrationTests(unittest.TestCase):
    def test_persistent_helpers_refresh_float_codes_scales_and_clear_quantization(self):
        # Deliberately cross shard boundaries inside original global batches.
        shards = [dataclasses.replace(fixtures.dataset(count, offset), split='validation')
                  for count, offset in ((4093, 0), (8200, 12), (4094, 24))]
        canonical = t.concatenate_datasets(shards, split='validation')
        common = dataclasses.replace(fixtures.dataset(7), split='validation')
        self.assertEqual(len(canonical), 16387)
        datasets = {'common_adjudicator': common, 'canonical_validation': canonical}
        with t.native_thread_execution_scope(), tempfile.TemporaryDirectory() as temporary:
            for workers in (2, 4):
                parameters, _ = tiny_inputs()
                quantized = t.quantize_fixed(parameters, ARCH,
                    {'w1': np.float32(.03), 'w2': np.float32(.25), 'w3': np.float32(.25)})
                results = []
                pool = pools.ValidationPredictionPool(datasets, Path(temporary) / str(workers), workers, trainer=t)
                with pool:
                    for mode in ('quantized', 'float', 'changed-quantized', 'changed-float-master'):
                        if mode == 'float':
                            parameters['w3'][0] = np.float32(-.75)
                        elif mode == 'changed-quantized':
                            quantized.integer['w3'][0] = -3
                            quantized.scales['w3'] = np.float32(.125)
                        elif mode == 'changed-float-master':
                            # QAT forwards use codes, but the provenance packet
                            # must still reflect the current mutable master.
                            parameters['w1'][0, 0] += np.float32(.007)
                        q = None if mode == 'float' else quantized
                        shapes = []
                        original = t.forward
                        def observe(*args, **kwargs):
                            shapes.append(len(args[2]))
                            return original(*args, **kwargs)
                        with mock.patch.object(t, 'forward', side_effect=observe):
                            expected = t.predict_dataset(parameters, ARCH, canonical, quantized=q)
                        self.assertEqual(shapes, [4096, 4096, 4096, 4096, 3])
                        actual = pool.predict(parameters, ARCH, canonical, quantized=q)
                        self.assertEqual(actual.dtype, np.dtype('float32'))
                        self.assertEqual(actual.tobytes(), expected.tobytes(), (workers, mode))
                        self.assertEqual(t.metrics_from_predictions(actual, canonical, ARM),
                                         t.metrics_from_predictions(expected, canonical, ARM))
                        results.append(actual.tobytes())
                    # Fewer batches than workers still require every helper's
                    # refresh/done acknowledgement before parent work resumes.
                    small = pool.predict(parameters, ARCH, common)
                    self.assertEqual(small.tobytes(), t.predict_dataset(parameters, ARCH, common).tobytes())
                self.assertNotEqual(results[0], results[1])
                self.assertNotEqual(results[1], results[2])
                self.assertEqual(results[2], results[3])
                evidence = pool.evidence()
                calls = evidence['calls']
                self.assertEqual([row['model_mode'] for row in calls],
                    ['runtime.v1', 'float', 'runtime.v1', 'runtime.v1', 'float'])
                self.assertEqual(len({row['model_sha256'] for row in calls[:4]}), 4)
                self.assertEqual(calls[-1]['active_slots'], [0])
                self.assertTrue(all(row['all_helpers_refreshed_and_done'] for row in calls))
                self.assertEqual(evidence['all_helper_exitcodes'], [0] * workers)
                self.assertEqual(len({row['pid'] for row in evidence['helpers']}), workers)
                self.assertNotIn(os.getpid(), {row['pid'] for row in evidence['helpers']})
                settings = t.validation_prediction_settings(workers, 1)
                settings['features'] = {name: pools._features(data, np)
                                        for name, data in datasets.items()}
                identities = {name: t.dataset_identity(data) for name, data in datasets.items()}
                pools.validate_execution(evidence, settings, artifact_root=pool.directory,
                                         dataset_identities=identities, trainer=t)
                # A new internally consistent CSR snapshot of the same shape
                # cannot replace the inputs bound before training. This check
                # must fail before even opening the forged evidence files.
                forged = copy.deepcopy(evidence)
                changed = canonical.indices.copy()
                changed[0] = 4
                changed_path = pool.directory / 'different-same-shape-indices.npy'
                with changed_path.open('xb') as handle:
                    np.save(handle, changed, allow_pickle=False)
                forged['datasets']['canonical_validation']['features']['indices_sha256'] = pools._array_hash(changed)
                forged['datasets']['canonical_validation']['indices'] = pools._record(changed_path)
                forged = t.body_hashed({key: value for key, value in forged.items() if key != 'body_sha256'})
                with mock.patch.object(pools, '_verify', side_effect=AssertionError('opened substituted input')):
                    with self.assertRaisesRegex(pools.PredictionError, 'outer binding'):
                        pools.validate_execution(forged, settings, artifact_root=pool.directory,
                                                 dataset_identities=identities, trainer=t)

    def test_mutated_feature_content_rejects_before_dispatch(self):
        parameters, inputs = tiny_inputs()
        datasets = {name: getattr(inputs, name)
                    for name in ('common_adjudicator', 'canonical_validation')}
        with t.native_thread_execution_scope(), tempfile.TemporaryDirectory() as temporary:
            with pools.ValidationPredictionPool(datasets, Path(temporary) / 'maps', 2, trainer=t) as pool:
                canonical = inputs.canonical_validation
                pool.predict(parameters, ARCH, canonical)
                before = pool.generation
                canonical.indices[0] = 4
                with self.assertRaisesRegex(pools.PredictionError, 'features changed'):
                    pool.predict(parameters, ARCH, canonical)
                self.assertEqual(pool.generation, before)


class TrainingPredictionIntegrationTests(unittest.TestCase):
    def test_actual_seed_receipt_resume_and_selection_validate_helper_execution(self):
        initial_parameters, inputs = tiny_inputs()
        inputs = dataclasses.replace(inputs, common_adjudicator=
            dataclasses.replace(fixtures.dataset(4096), split='validation'))
        bundle = SimpleNamespace(body_sha256='a' * 64)
        receipts = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = t.write_float_checkpoint(root / 'initial', initial_parameters, ARCH)
            with np.errstate(divide='ignore', invalid='ignore'):
                for workers in (1, 2):
                    receipts[workers] = t.train_seed_candidate(bundle, inputs, ARCH, ARM,
                        SEED, root / str(workers), ranking_weight=.1,
                        initial_checkpoint=initial, validation_workers=workers, _seed_concurrency=2)
            for key in ('float_training', 'quantized_training', 'float_validation',
                        'quantized_validation', 'offline_gate'):
                self.assertEqual(receipts[1][key], receipts[2][key], key)
            self.assertNotIn('validation_prediction', receipts[1]['binding']['settings'])
            self.assertNotIn('validation_prediction_execution', receipts[1])
            self.assertEqual(receipts[1]['native_thread_execution']['schema'], t.NATIVE_THREAD_EXECUTION_SCHEMA)
            native = receipts[2]['native_thread_execution']
            self.assertEqual(native['schema'], t.PARALLEL_NATIVE_EXECUTION_SCHEMA)
            self.assertNotIn('native_threads_per_seed_maximum', native)
            self.assertEqual(native['maximum_active_numerical_streams'], 4)
            self.assertEqual(native['coordinator_kernel']['native_threads_per_kernel_maximum'], 1)
            for name in ('float_checkpoint', 'quantized_runtime'):
                left = root / '1' / receipts[1][name]['path']
                right = root / '2' / receipts[2][name]['path']
                self.assertEqual(left.read_bytes(), right.read_bytes())
            output = root / '2'
            receipt = receipts[2]
            reference = t._seed_reference_path(output, ARCH, ARM, SEED)
            with mock.patch.object(t, 'validate_validation_prediction_receipt',
                    wraps=t.validate_validation_prediction_receipt) as validation:
                self.assertEqual(t._load_seed_receipt_from_reference(output, reference,
                    receipt['binding']), receipt)
                with mock.patch.object(pools, 'ValidationPredictionPool',
                        side_effect=AssertionError('completed resume spawned new helpers')):
                    resumed = t.train_seed_candidate(bundle, inputs, ARCH, ARM, SEED, output,
                        ranking_weight=.1, initial_checkpoint=initial, validation_workers=2,
                        _seed_concurrency=2, resume=True)
                    self.assertEqual(resumed, receipt)
                    with self.assertRaisesRegex(t.TrainingError, 'binding changed'):
                        t.train_seed_candidate(bundle, inputs, ARCH, ARM, SEED, output,
                            ranking_weight=.1, initial_checkpoint=initial, validation_workers=4,
                            _seed_concurrency=1, resume=True)
                # Reuse one actual seed for selection assembly only. No further
                # optimizer, prediction or roster work occurs in this adapter.
                with mock.patch.object(t, 'load_training_inputs', return_value=inputs), \
                     mock.patch.object(t, '_train_seed_roster', return_value=[receipt]):
                    selected_path = t.train_arm_campaign(bundle, ARCH, ARM, output,
                        successor_labels=root / 'synthetic-labels.json', ranking_weight=.1,
                        initial_checkpoint=initial, seed_workers=2, validation_workers=2,
                        generated_source_ascii_bytes=92000)
                selected = t.validate_selection(selected_path, output, bundle)
                self.assertEqual(selected['selected_seed_receipt_body_sha256'], receipt['body_sha256'])
                self.assertEqual(selected['validation_prediction'], receipt['binding']['settings']['validation_prediction'])
                self.assertGreaterEqual(validation.call_count, 3)

    def test_real_warmup_and_four_qat_epochs_have_exact_serial_outputs(self):
        initial, inputs = tiny_inputs()
        profile = t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE
        results, updates, evidence = {}, {}, None
        original_update = t.AdamW.update
        with t.native_thread_execution_scope(), tempfile.TemporaryDirectory() as temporary:
            for workers in (1, 2):
                traces = []
                def observe_update(optimizer, parameters, gradients):
                    original_update(optimizer, parameters, gradients)
                    traces.append((optimizer.step, float(optimizer.learning_rate),
                        tuple(parameters[name].tobytes() for name in LAYERS)))
                def train(predictor=None):
                    kwargs = {} if predictor is None else {'_prediction_executor': predictor}
                    with mock.patch.object(t.AdamW, 'update', observe_update), np.errstate(divide='ignore', invalid='ignore'):
                        floating = t.train_float_seed(inputs, ARCH, ARM, SEED,
                            maximum_epochs=1, patience=1, learning_rate=.00006,
                            initial_parameters={name: value.copy() for name, value in initial.items()},
                            ranking_weight=.1, **kwargs)
                        qat = t.run_fixed_scale_qat(floating, inputs, ARCH, ARM, SEED,
                            qat_profile=profile, ranking_weight=.1, **kwargs)
                    return floating, qat
                if workers == 1:
                    results[workers] = train()
                else:
                    with pools.ValidationPredictionPool(
                        {name: getattr(inputs, name) for name in
                         ('common_adjudicator', 'canonical_validation')},
                        Path(temporary) / 'maps', workers, trainer=t) as pool:
                        results[workers] = train(pool)
                    evidence = pool.evidence()
                updates[workers] = traces
            self.assertEqual(len(updates[1]), 5)
            self.assertEqual(updates[1], updates[2])
            serial_float, serial_qat = results[1]
            parallel_float, parallel_qat = results[2]
            # Complete reports include each scale trial, adaptive scale choice,
            # ranking metric, selected epoch and per-layer update evidence.
            self.assertEqual(t.canonical_json_bytes(serial_float.report),
                             t.canonical_json_bytes(parallel_float.report))
            self.assertEqual(t.canonical_json_bytes(serial_qat.report),
                             t.canonical_json_bytes(parallel_qat.report))
            self.assertEqual(serial_qat.report['executed_qat_epochs'], [1, 2, 3, 4])
            artifacts = []
            for workers, (floating, qat) in results.items():
                root = Path(temporary) / f'output-{workers}'
                checkpoint = t.write_float_checkpoint(root / 'float', floating.parameters, ARCH)
                runtime = t.write_runtime(root / 'runtime', ARCH, qat.quantized, arm=ARM,
                    seed=SEED, float_epoch=floating.epoch, qat_epoch=qat.qat_epoch,
                    source_bundle_body_sha256='a' * 64)
                artifacts.append((checkpoint.read_bytes(), runtime.read_bytes()))
            self.assertEqual(artifacts[0], artifacts[1])
            self.assertGreater(len(evidence['calls']), 10)
            self.assertEqual({row['dataset'] for row in evidence['calls']},
                             {'common_adjudicator', 'canonical_validation'})
            self.assertEqual(evidence['all_helper_exitcodes'], [0, 0])

    def test_channel_and_oversubscribed_roster_reject_before_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'never-created'
            with mock.patch.object(pools, 'ValidationPredictionPool', side_effect=AssertionError('spawn before admission')), \
                 mock.patch.object(t, 'train_float_seed', side_effect=AssertionError('warmup before admission')), \
                 mock.patch.object(t, 'load_training_inputs', side_effect=AssertionError('data before admission')):
                with self.assertRaisesRegex(t.TrainingError, 'channel'):
                    t.train_seed_candidate(None, None, ARCH, ARM, SEED, output,
                        qat_profile=t.CHANNEL_PREDICTION_QAT_PROFILE, validation_workers=2)
                for seeds, workers in ((4, 2), (2, 4)):
                    with self.subTest(seeds=seeds, workers=workers):
                        with self.assertRaisesRegex(t.TrainingError, 'concurrency'):
                            t.train_arm_campaign(None, ARCH, ARM, output,
                                seed_workers=seeds, validation_workers=workers)
                self.assertFalse(output.exists())
        for seeds, workers in ((1, 4), (2, 2), (4, 1)):
            setting = t.validation_prediction_settings(workers, seeds)
            if workers == 1:
                self.assertIsNone(setting)
            else:
                self.assertEqual(setting['policy']['maximum_active_numerical_streams'], 4)
        for invalid in (True, 1., 3, 8):
            with self.subTest(invalid=invalid), self.assertRaises(t.TrainingError):
                t.validation_prediction_settings(invalid, 1)


if __name__ == '__main__':
    unittest.main()
