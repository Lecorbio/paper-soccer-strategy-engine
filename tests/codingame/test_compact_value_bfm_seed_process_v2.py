"""Bounded synthetic spawn equivalence; no campaign seed or protected data."""
import concurrent.futures
import contextlib
import dataclasses
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from tools import compact_value_bfm_seed_process_v2 as process
os.environ.update(process.ENVIRONMENT)
os.environ[process.MARKER] = '1'

import numpy as np
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_train as trainer
from tools import compact_value_bfm_teacher_training as adapter
from tools import compact_value_bfm_ranking_store as store
from tests.codingame.test_compact_value_bfm_training import bundle_fixture, dataset, active_row
from tests.codingame import test_compact_value_bfm_ranking_store as store_tests


def core_datasets():
    empty = np.asarray(campaign.features.encode_active(campaign.features.ReplayState()), dtype='<u2')
    return (dataset([active_row(2, 4), empty, active_row(3, 5)], [0.2, 0.3, -0.4]),
        dataset([active_row(1, 9)], [0.6], split='validation'),
        dataset([active_row(4, 3)], [-0.7], split='validation'),
        {'anchor': ('canonical/train',), 'common_adjudicator': ('common',),
         'canonical_validation': ('canonical/validation',)})


@contextlib.contextmanager
def reconstruction_fixture(root):
    bundle = bundle_fixture(root)
    new = dataset([active_row(0, 1), active_row(1, 2)], [0.1, -0.2])
    with (mock.patch.object(trainer.FrozenBundle, 'load', return_value=bundle),
          mock.patch.object(adapter, '_load_core_inputs', side_effect=lambda _bundle: core_datasets()),
          mock.patch.object(trainer, 'load_shard', return_value=new)):
        yield


def write_fixture(root):
    phase = root / 'pilot'
    phase.mkdir()
    _rows, _bundle, index = store_tests.RankingStoreTests().fixture(root)
    (root / 'bundle-manifest.json').write_bytes(b'synthetic bundle')
    architecture = trainer.ARCHITECTURES['capacity-12x8']
    initial = trainer.write_float_checkpoint(root / 'initial',
        trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0]), architecture)
    campaign.seal(root / 'campaign.json', {'bundle': campaign.record(root / 'bundle-manifest.json'),
        'training_executor': process.MODE, 'inputs': {
            'attempt_one_initial_checkpoint': campaign.record(initial)}})
    campaign.seal(root / 'anchor-exclusions.json', {'synthetic': True})
    campaign.seal(phase / 'positions.json', {'rows': [
        {'split': 'train', 'drawn_edges': 0, 'prefix': ''}]})
    campaign.seal(phase / 'labels.json', {'merged': campaign.record(root / 'labels.jsonl'),
        'positions': campaign.record(phase / 'positions.json')})
    for name in ('train.json', 'train.npz', 'validation.json', 'validation.npz'):
        (phase / name).write_bytes(b'synthetic scalar shard')
    audit = {'schema': campaign.ID + '.training-input-audit.v2',
        'bundle': campaign.record(root / 'bundle-manifest.json'),
        'exclusion_index': campaign.record(root / 'anchor-exclusions.json'),
        'labels': campaign.record(phase / 'labels.json'),
        'position_closure': campaign.record(phase / 'positions.json'),
        'ranking_store': campaign.record(index), 'anchor_duplicates_removed': 1,
        'shards': {split: {kind: campaign.record(phase / f'{split}.{extension}')
            for kind, extension in (('manifest', 'json'), ('npz', 'npz'))}
            for split in ('train', 'validation')}, 'protected_tests_opened': False}
    campaign.seal(phase / 'training-input-audit.json', audit)


def _probe_inner(root, output, execution):
    with reconstruction_fixture(Path(root)):
        _bundle, inputs, identity = process.reconstruct_inputs(Path(root), 'pilot')
    architecture = trainer.ARCHITECTURES['capacity-12x8']
    params = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
    active = inputs.new.active_rows(range(len(inputs.new)))
    predictions, cache = trainer.forward(params, architecture, active)
    loss, gradient, _report = trainer.arm_loss_gradient(
        'search-target', predictions, inputs.new.targets, inputs.new.weights)
    gradients = trainer._network_gradients(params, architecture, active, cache, gradient, params)
    group = trainer.CompleteTurnGroup('synthetic-gradient', 0, (
        trainer.CompleteTurnSuccessor('a' * 64, active_row(0, 2), .7, 0, {}),
        trainer.CompleteTurnSuccessor('b' * 64, active_row(1, 3), -.2, 0, {})))
    ranking_predictions = np.linspace(-.4, .6, len(group.successors), dtype=np.float32)
    ranking_loss, ranking_gradient, report = trainer.pairwise_successor_ranking_loss_gradient(group, ranking_predictions)
    # One actual 64/192 mixed QAT batch tests deterministic optimizer updates,
    # while avoiding extra campaign training or a full production seed.
    scales = {name: float(np.max(np.abs(value))) / 3 for name, value in params.items()}
    optimizer = trainer.AdamW(params, learning_rate=trainer.QAT_LEARNING_RATE,
        weight_decay=trainer.WEIGHT_DECAY)
    new_rows, anchor_rows = next(trainer.mixed_epoch_batches(len(inputs.new), len(inputs.anchor),
        seed=trainer.FIXED_SEEDS[0], epoch=1))
    batch_loss = trainer._train_mixed_batch(params, architecture, trainer.ARMS['search-target'],
        optimizer, inputs, new_rows, anchor_rows, fixed_scales=scales,
        ranking_groups=(group,), ranking_weight=.1)
    checkpoint = trainer.write_float_checkpoint(Path(output), params, architecture)
    quantized = trainer.quantize_fixed(params, architecture, scales)
    return {'identity': identity, 'loss': loss, 'gradient': trainer._array_identity(gradient),
        'gradients': {key: trainer._array_identity(value) for key, value in gradients.items()},
        'ranking_loss': ranking_loss, 'ranking_gradient': trainer._array_identity(ranking_gradient),
        'ranking_report': report, 'batch_loss': batch_loss, 'checkpoint': campaign.sha(checkpoint),
        'quantized': {key: trainer._array_identity(value) for key, value in quantized.integer.items()},
        'metrics': trainer.evaluate_validation_pair(params, architecture, inputs=inputs,
            arm=trainer.ARMS['search-target'], quantized=quantized),
        'execution': execution, 'pid': os.getpid()}


def probe(root, output):
    return process.seed_thread(_probe_inner, root, output)


class SeedProcessV2Tests(unittest.TestCase):
    def test_frozen_opt_in_only(self):
        self.assertEqual(process.executor_mode({}), 'threads')
        self.assertEqual(process.executor_mode({'training_executor': process.MODE}), 'spawn-v2')
        for setting in ('spawn', {'mode': 'spawn-v2', 'maximum_workers': 3}, False):
            with self.assertRaisesRegex(ValueError, 'frozen training executor'):
                process.executor_mode({'training_executor': setting})

    def test_four_workers_require_explicit_source_bound_authority(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        self.assertEqual(process.MODE, process.MODE2)
        self.assertEqual(process.normalize_executor(process.MODE4)['maximum_workers'], 4)
        with self.assertRaisesRegex(ValueError, 'frozen parent campaign'):
            process.executor_mode({'training_executor': process.MODE4})
        plan = {'training_executor': process.MODE4, 'training_resource_authorization': {'frozen': True}}
        with mock.patch.object(resources, 'expected_workers', return_value=4) as authority:
            self.assertEqual(process.executor_mode(plan), 'spawn-v2')
            authority.assert_called_once_with(plan)
        with mock.patch.object(resources, 'expected_workers', side_effect=ValueError('changed authority')):
            with self.assertRaisesRegex(ValueError, 'changed authority'):
                process.executor_mode(plan)
        for workers in (True, 4.0, 3, 5):
            with self.assertRaisesRegex(ValueError, 'frozen training executor'):
                process.normalize_executor({'mode': 'spawn-v2', 'maximum_workers': workers})

    def test_four_worker_roster_dispatches_across_lambdas_and_keeps_receipt_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'spec.json'
            jobs = [{'weight': weight, 'seed': seed} for weight in (0., .1, .25) for seed in trainer.FIXED_SEEDS]
            campaign.seal(path, {'executor': process.MODE4, 'ranking_weights': [0., .1, .25], 'jobs': jobs})
            executor = process.SpawnSeedExecutor(path)
            barrier = threading.Barrier(4)
            started = []
            lock = threading.Lock()
            def bounded_worker(job):
                with lock:
                    started.append((job['weight'], job['seed']))
                    ordinal = len(started)
                if ordinal <= 4:
                    barrier.wait(timeout=3)
                return (job['weight'], job['seed'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                executor.pool = pool
                with (mock.patch.object(process, '_run', side_effect=bounded_worker),
                      mock.patch.object(executor, '_collect', side_effect=lambda _jobs, returned: returned)):
                    result = executor.run_roster()
                self.assertEqual(result, [(row['weight'], row['seed']) for row in jobs])
                self.assertEqual(set(started[:4]), {(row['weight'], row['seed']) for row in jobs[:4]})
                self.assertIn(.1, {weight for weight, _seed in started[:4]})
                with self.assertRaisesRegex(ValueError, 'exactly once'):
                    executor.run_roster()
                with self.assertRaisesRegex(ValueError, 'flattened run_roster'):
                    executor.run_weight(0.)

    def test_four_worker_pool_is_created_only_after_phase_authority_validation(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / 'authorization.json'
            campaign.seal(authorization, {'frozen': True})
            auth = campaign.record(authorization)
            phase_path = root / 'campaign.json'
            campaign.seal(phase_path, {'training_executor': process.MODE4,
                'training_resource_authorization': auth})
            spec = root / 'spec.json'
            campaign.seal(spec, {'executor': process.MODE4, 'phase_contract': campaign.record(phase_path),
                'training_resource_authorization': auth})
            with (mock.patch.object(resources, 'expected_workers', return_value=4),
                  mock.patch.object(process.concurrent.futures, 'ProcessPoolExecutor') as pool):
                with process.SpawnSeedExecutor(spec):
                    self.assertEqual(pool.call_args.kwargs['max_workers'], 4)
                    self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')
                pool.return_value.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
            with (mock.patch.object(resources, 'expected_workers', side_effect=ValueError('changed authority')),
                  mock.patch.object(process.concurrent.futures, 'ProcessPoolExecutor') as pool):
                with self.assertRaisesRegex(ValueError, 'changed authority'):
                    process.SpawnSeedExecutor(spec).__enter__()
                pool.assert_not_called()

    def test_four_worker_spec_and_initializer_rebind_the_phase_authorization(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            phase = campaign.read(root / 'campaign.json')
            phase.pop('body_sha256')
            authorization = root / 'authorization.json'
            campaign.seal(authorization, {'synthetic': True})
            phase.update({'training_executor': process.MODE4,
                'training_resource_authorization': campaign.record(authorization)})
            (root / 'campaign.json').unlink()
            campaign.seal(root / 'campaign.json', phase)
            with (reconstruction_fixture(root),
                  mock.patch.object(resources, 'expected_workers', return_value=4),
                  mock.patch.object(process, 'source_closure', return_value=[])):
                bundle, inputs, identity = process.reconstruct_inputs(root, 'pilot')
                path = process.freeze_spec(root, 'pilot', bundle, inputs, identity['anchor_filter'],
                    (0., .1, .25), trainer.FIXED_SEEDS)
                spec = campaign.read(path)
                self.assertEqual(spec['executor'], process.MODE4)
                self.assertEqual(spec['training_resource_authorization'], phase['training_resource_authorization'])
                self.assertEqual(len(spec['jobs']), 9)
                process._initialize(str(path))
                changed = {key: value for key, value in spec.items() if key != 'body_sha256'}
                changed['training_resource_authorization'] = {'changed': True}
                invalid = root / 'changed-spec.json'
                campaign.seal(invalid, changed)
                with self.assertRaisesRegex(ValueError, 'resource authorization changed'):
                    process._initialize(str(invalid))
            process._WORKER = None

    def test_real_collector_preserves_multi_weight_receipts_and_process_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            jobs, returned, receipts = [], [], []
            with trainer.native_thread_execution_scope() as execution:
                for weight in (0., .1, .25):
                    directory = root / f'lambda-{weight:.2f}'
                    for seed in trainer.FIXED_SEEDS:
                        binding = trainer.body_hashed({'weight': weight, 'seed': seed})
                        job = {'weight': weight, 'seed': seed, 'directory': str(directory), 'binding': binding}
                        reference = trainer._seed_reference_path(directory,
                            trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target'], seed)
                        campaign.once(reference, campaign.raw({'synthetic': True, 'weight': weight, 'seed': seed}))
                        jobs.append(job)
                        receipts.append({'seed': seed, 'binding': binding, 'native_thread_execution': execution})
                        returned.append({'reference': campaign.record(reference),
                            'binding_sha256': binding['body_sha256'], 'native_thread_execution': execution,
                            'process': {'pid': 101 + len(jobs) % 4}})
            by_identity = {(job['weight'], job['seed']): (job, result, receipt)
                for job, result, receipt in zip(jobs, returned, receipts, strict=True)}
            def load_reference(directory, reference, binding):
                job, result, receipt = by_identity[binding['weight'], binding['seed']]
                self.assertEqual(directory, Path(job['directory']))
                self.assertEqual(reference, Path(result['reference']['path']))
                self.assertEqual(binding, job['binding'])
                return receipt
            for mode in (process.MODE2, process.MODE4):
                with self.subTest(mode=mode):
                    path = root / f'spec-{mode["maximum_workers"]}.json'
                    campaign.seal(path, {'executor': mode, 'ranking_weights': [0., .1, .25], 'jobs': jobs})
                    executor = process.SpawnSeedExecutor(path)
                    executor.pool = mock.Mock()
                    executor.pool.map.side_effect = lambda _worker, roster: [
                        by_identity[job['weight'], job['seed']][1] for job in roster]
                    with mock.patch.object(trainer, '_load_seed_receipt_from_reference', side_effect=load_reference) as loader:
                        actual = executor.run_roster() if mode == process.MODE4 else [
                            receipt for weight in (0., .1, .25) for receipt in executor.run_weight(weight)]
                    self.assertEqual(actual, receipts)
                    self.assertEqual(loader.call_count, 9)
                    self.assertEqual(executor.pool.map.call_count, 1 if mode == process.MODE4 else 3)
                    self.assertEqual(executor.evidence, [
                        {'weight': job['weight'], 'seed': job['seed'], **result}
                        for job, result in zip(jobs, returned, strict=True)])
                    invalid = [{**returned[0], 'binding_sha256': 'wrong'}]
                    with self.assertRaisesRegex(ValueError, 'seed reference changed'):
                        executor._collect(jobs[:1], invalid)
                    invalid = [{**returned[0], 'native_thread_execution': {'wrong': True}}]
                    with mock.patch.object(trainer, '_load_seed_receipt_from_reference', side_effect=load_reference):
                        with self.assertRaisesRegex(ValueError, 'numerical limiter evidence changed'):
                            executor._collect(jobs[:1], invalid)

    def test_phase_profile_is_validated_before_training_inputs_are_opened(self):
        for plan, override in (({'attempt': 1, 'qat_profile': 'refined-adaptive-scales-v1'}, None),
                ({'attempt': 1}, 'refined-adaptive-scales-v1')):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                campaign.seal(root / 'campaign.json', plan)
                with self.assertRaisesRegex(ValueError, 'QAT profile'):
                    campaign.train_models(root, 'pilot', qat_profile=override)

    def test_filter_matches_original_four_transform_order(self):
        state, prefix = campaign.fresh_root(6, campaign.random.Random(91))
        matches = [np.asarray(campaign.features.encode_active(campaign.openings.transform_state(
            state, rotate=rotate, reflect=reflect)), dtype='<u2')
            for rotate in (False, True) for reflect in (False, True)]
        anchor = dataset([active_row(2, 3), *matches, active_row(4, 5)])
        rows = [{'split': 'train', 'drawn_edges': 6, 'prefix': prefix}]
        filtered, evidence = process.filter_early_anchor(anchor, rows)
        self.assertEqual(evidence['removed_rows'], 4)
        self.assertEqual(evidence['removed_indices_sha256'], trainer._array_identity(np.arange(1, 5, dtype='<i8')))
        self.assertEqual(evidence['kept_indices_sha256'], trainer._array_identity(np.asarray([0, 5], dtype='<i8')))
        self.assertEqual(filtered.source_route, anchor.source_route)
        np.testing.assert_array_equal(filtered.targets, anchor.targets[[0, 5]])
        for index, old in enumerate((0, 5)):
            np.testing.assert_array_equal(filtered.active_row(index), anchor.active_row(old))
        for replacement in ({'split': 'validation'}, {'drawn_edges': 7}):
            unchanged, evidence = process.filter_early_anchor(anchor, [{**rows[0], **replacement}])
            self.assertIs(unchanged, anchor)
            self.assertEqual(evidence['removed_rows'], 0)

    def test_reconstruction_uses_core_canonical_not_fresh_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            # If reconstruction touches fresh validation, its bound bytes fail.
            (root / 'pilot' / 'validation.json').unlink()
            (root / 'pilot' / 'validation.npz').unlink()
            with reconstruction_fixture(root):
                _bundle, inputs, identity = process.reconstruct_inputs(root, 'pilot')
            self.assertEqual(trainer.dataset_identity(inputs.canonical_validation), trainer.dataset_identity(core_datasets()[2]))
            self.assertEqual(identity['anchor_filter']['removed_rows'], 1)
            self.assertFalse(inputs.new.indices.flags.writeable)
            self.assertFalse(inputs.anchor.targets.flags.writeable)
            changed = dataclasses.replace(inputs, successor_rankings=dataclasses.replace(
                inputs.successor_rankings, train=inputs.successor_rankings.validation))
            self.assertNotEqual(process.input_identity(changed, identity['anchor_filter']), identity)

    def test_spawn_matches_serial_inputs_losses_gradients_checkpoint_and_qat_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            serial = probe(str(root), str(root / 'serial'))
            with concurrent.futures.ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                spawned = pool.submit(probe, str(root), str(root / 'spawned')).result(timeout=45)
            self.assertNotEqual(serial.pop('pid'), spawned.pop('pid'))
            self.assertEqual(serial, spawned)
            self.assertEqual(serial['execution']['native_threads_per_seed_maximum'], 1)

    def test_spec_binds_reconstructed_inputs_original_checkpoint_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            source = root / 'source.py'
            source.write_text('# frozen synthetic producer\n')
            with reconstruction_fixture(root), mock.patch.object(process, 'source_closure',
                    side_effect=lambda: [campaign.record(source)]):
                bundle, inputs, identity = process.reconstruct_inputs(root, 'pilot')
                spec_path = process.freeze_spec(root, 'pilot', bundle, inputs,
                    identity['anchor_filter'], (0.0, .1, .25), trainer.FIXED_SEEDS)
                spec = campaign.read(spec_path)
                self.assertEqual([row['weight'] for row in spec['jobs']], [0.0] * 3 + [.1] * 3 + [.25] * 3)
                self.assertEqual([row['seed'] for row in spec['jobs']], list(trainer.FIXED_SEEDS) * 3)
                self.assertEqual(spec['reconstruction'], identity)
                for job in spec['jobs']:
                    self.assertEqual(job['binding']['settings']['qat_profile'], 'standard-v1')
                    self.assertEqual(job['binding']['successor_ranking']['initial_checkpoint']['sha256'],
                        spec['initial_checkpoint']['sha256'])
                    self.assertEqual(job['binding'], trainer.training_binding(bundle, inputs,
                        trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target'],
                        job['seed'], None, job['weight'], Path(spec['initial_checkpoint']['path']),
                        spec['qat_profile']))
                process._initialize(str(spec_path))
                self.assertEqual(process._WORKER[0], spec)
                source.write_text('# changed producer\n')
                with self.assertRaisesRegex(ValueError, 'source closure changed'):
                    process._initialize(str(spec_path))
                source.write_text('# frozen synthetic producer\n')
                with mock.patch.object(process, 'input_identity', return_value={'changed': True}):
                    with self.assertRaisesRegex(ValueError, 'input reconstruction differs'):
                        process._initialize(str(spec_path))
                with self.assertRaisesRegex(ValueError, 'QAT profile differs'):
                    process.freeze_spec(root, 'pilot', bundle, inputs, identity['anchor_filter'],
                        (0.0,), trainer.FIXED_SEEDS[:1], qat_profile='refined-adaptive-scales-v1')
            process._WORKER = None

    def test_integer_zero_full_roster_matches_maintained_float_normalized_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with reconstruction_fixture(root), mock.patch.object(process, 'source_closure', return_value=[]):
                bundle, inputs, identity = process.reconstruct_inputs(root, 'pilot')
                initial = Path(campaign.read(root / 'campaign.json')['inputs']['attempt_one_initial_checkpoint']['path'])
                architecture, arm = trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target']
                # Full phases use [0, admitted_weight], whose scalar entry is
                # an integer even though maintained bindings normalize floats.
                spec_path = process.freeze_spec(root, 'pilot', bundle, inputs,
                    identity['anchor_filter'], (0, .1), trainer.FIXED_SEEDS)
                spec = campaign.read(spec_path)
                self.assertEqual(len(spec['jobs']), 6)
                for job in spec['jobs']:
                    expected = trainer.training_binding(bundle, inputs, architecture, arm, job['seed'],
                        None, job['weight'], initial, trainer.STANDARD_QAT_PROFILE)
                    self.assertEqual(job['binding'], expected)
                    self.assertEqual(job['binding']['body_sha256'], expected['body_sha256'])
                    self.assertIs(type(job['binding']['successor_ranking']['loss_weight']), float)
                base = spec['jobs'][0]['binding']
                self.assertEqual(process.roster_binding(base, trainer.FIXED_SEEDS[0], 0),
                    process.roster_binding(base, trainer.FIXED_SEEDS[0], 0.0))
                with self.assertRaisesRegex(trainer.TrainingError, 'weight is not numeric'):
                    process.roster_binding(base, trainer.FIXED_SEEDS[0], False)

    def test_shutdown_waits_and_does_not_replace_failed_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'spec.json'
            campaign.seal(path, {'ranking_weights': [0, .1], 'jobs': []})
            executor = process.SpawnSeedExecutor(path)
            executor.pool = mock.Mock()
            pool = executor.pool
            with self.assertRaisesRegex(ValueError, 'lambda execution order'):
                executor.run_weight(.1)
            executor.__exit__(RuntimeError, RuntimeError('failed'), None)
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
            self.assertIsNone(executor.pool)


if __name__ == '__main__':
    unittest.main()
