"""Bounded metadata, observer and comparison tests; no real seed execution."""
import concurrent.futures
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_validation_cache_check_v2 as check


def hashed(body):
    return {**body, 'body_sha256': hashlib.sha256(check.raw(body)).hexdigest()}


def base_binding():
    return hashed({'schema': 'papersoccer.compact-value-bfm-training-binding.v1',
        'seed': check.SEEDS[0], 'datasets': {'new': {'digest': 'real-smoke'}},
        'source_routes': {'new': ['/same/original/smoke/shard']},
        'settings': {'qat_profile': 'standard-v1', 'qat_profile_contract': {'qat_profile': 'standard-v1'},
                     'batch_size': 256, 'new_rows': 64, 'anchor_rows': 192},
        'successor_ranking': {'loss_weight': 0., 'teacher': 'accepted', 'train_groups': 829, 'validation_groups': 165}})


def observation(candidate=False):
    return {'observer': check.record(check.__file__), 'policy': check.POLICY['observer'],
        'profile_hook_restored': True, 'counts': {'ranking_metrics_calls': 200,
            'float_ranking_forward_calls': 30 if candidate else 150,
            'quantized_ranking_forward_calls': 149, 'other_float_forward_calls': 99,
            'other_quantized_forward_calls': 99, 'cache_factory_calls': 5 if candidate else 0,
            'immutable_mapped_factory_returns': 5 if candidate else 0}}


def comparison_fixture():
    profile = {'qat_profile': check.PROFILE, 'schedule': {'float_warmup_epochs': 1, 'qat_epochs': 4}}
    base = check.refined_binding(base_binding(), profile)
    rows = []
    for weight in check.WEIGHTS:
        for seed in check.SEEDS:
            binding = copy.deepcopy(base); binding['seed'] = seed; binding['successor_ranking']['loss_weight'] = weight
            receipt = {'binding': binding, 'native_thread_execution': {'fixture_one_thread': True},
                'seed': seed, 'qat_profile': check.PROFILE,
                'float_training': {'gradient_updates': [0.1, 0.2, 0.3]},
                'quantized_training': {'trials': [{'scales': {'w1': 0.1}, 'metric': 0.8}], 'epochs': 4},
                'quantized_validation': {'accuracy': 0.85, 'passed': False}}
            rows.append({'weight': weight, 'seed': seed, 'receipt': hashed(receipt),
                **{key: {'sha256': 'a' * 64, 'bytes': 123} for key in ('float_checkpoint', 'quantized_runtime', 'source')}})
    reference = {'results': rows, 'reconstruction': {'exact': 'input-arrays-and-order'},
        'mapped_inputs': {'train': {'groups': 829}, 'validation': {'groups': 165}},
        'process_evidence': [{'cache_observation': observation()} for _ in rows]}
    candidate = copy.deepcopy(reference)
    candidate['process_evidence'] = [{'cache_observation': observation(True)} for _ in rows]
    return {'engines': {'reference': 'FA', 'candidate': 'cache'}}, reference, candidate


class CacheComparisonTests(unittest.TestCase):
    def test_refined_binding_changes_only_the_two_named_profile_settings(self):
        base = base_binding(); original = copy.deepcopy(base)
        profile = {'qat_profile': check.PROFILE, 'schedule': {'qat_epochs': 4}}
        result = check.refined_binding(base, profile)
        self.assertEqual(base, original)
        self.assertEqual({key: value for key, value in result.items() if key not in ('settings', 'body_sha256')},
                         {key: value for key, value in base.items() if key not in ('settings', 'body_sha256')})
        self.assertEqual({key for key in result['settings'] if result['settings'][key] != base['settings'][key]},
                         {'qat_profile', 'qat_profile_contract'})
        with self.assertRaisesRegex(ValueError, 'approved refined recipe'):
            check.refined_binding(base, {'qat_profile': 'standard-v1'})

    def test_all_nine_exact_results_and_actual_reduced_cache_work_are_required(self):
        plan, reference, candidate = comparison_fixture()
        result = check.compare_results(plan, reference, candidate)
        self.assertEqual(len(result['checks']), 9)
        self.assertTrue(result['exact_equivalence_passed']); self.assertTrue(result['mapped_cache_exercised'])
        self.assertFalse(result['production_activation_allowed']); self.assertFalse(result['qualification_eligible'])
        self.assertNotIn('wall_time_ratio', result)
        for changed in ('missing', 'duplicate'):
            broken = copy.deepcopy(candidate)
            if changed == 'missing': broken['results'].pop()
            else: broken['results'][-1] = broken['results'][-2]
            with self.assertRaisesRegex(ValueError, 'exact nine-seed roster'):
                check.compare_results(plan, reference, broken)

    def test_binding_trial_metric_and_artifact_drift_cannot_be_stripped(self):
        plan, reference, candidate = comparison_fixture()
        for change in ('binding', 'trial', 'typed_metric', 'artifact', 'native', 'reconstruction'):
            broken = copy.deepcopy(candidate)
            if change == 'binding': broken['results'][0]['receipt']['binding']['datasets']['new']['digest'] = 'other'
            elif change == 'trial': broken['results'][0]['receipt']['quantized_training']['trials'][0]['metric'] += .01
            elif change == 'typed_metric': broken['results'][0]['receipt']['quantized_validation']['passed'] = 0
            elif change == 'artifact': broken['results'][0]['float_checkpoint']['sha256'] = 'b' * 64
            elif change == 'native': broken['results'][0]['receipt']['native_thread_execution']['fixture_one_thread'] = 1
            else: broken['reconstruction']['exact'] = 'different-order'
            with self.subTest(change=change), self.assertRaises(ValueError):
                check.compare_results(plan, reference, broken)

    def test_noop_eager_fallback_or_changed_validation_work_cannot_pass(self):
        plan, reference, candidate = comparison_fixture()
        for key, value in [('cache_factory_calls', 0), ('immutable_mapped_factory_returns', 0),
                           ('float_ranking_forward_calls', 150), ('quantized_ranking_forward_calls', 148),
                           ('other_float_forward_calls', 98), ('ranking_metrics_calls', 199)]:
            broken = copy.deepcopy(candidate); broken['process_evidence'][0]['cache_observation']['counts'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'demonstrate exact mapped cache use'):
                check.compare_results(plan, reference, broken)

    def test_real_mapped_container_guard_checks_all_backing_arrays(self):
        import numpy as np
        class Store: pass
        class Successors:
            def __len__(self): return 1
        engine = types.SimpleNamespace(store=types.SimpleNamespace(MappedSuccessors=Successors, RankingStore=Store),
                                       trainer=types.SimpleNamespace(np=np))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'array.bin'; source.write_bytes(b'\0' * 32)
            store = Store()
            for name in ('metadata', 'indices', 'transcripts'): setattr(store, name, np.memmap(source, dtype='u1', mode='r'))
            successors = Successors(); successors.store = store
            group = types.SimpleNamespace(successors=successors)
            inputs = types.SimpleNamespace(successor_rankings=types.SimpleNamespace(train=(group,), validation=(group,)))
            result = check.mapped_inputs(engine, inputs)
            self.assertTrue(result['validation']['all_backings_readonly_memmap'])
            store.indices = np.zeros(4, dtype='u1')
            with self.assertRaisesRegex(ValueError, 'read-only'):
                check.mapped_inputs(engine, inputs)
            group.successors = []
            with self.assertRaisesRegex(ValueError, 'eager/mutable'):
                check.mapped_inputs(engine, inputs)


class IndependentCohortTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(); self.output = self.root / 'output'
        self.directory = self.output / 'reference'; self.plan_path = self.output / 'plan.json'
        self.metadata = check.seal(self.output / 'metadata/reference.json', {'smoke': {'fixture': 'native64'}})
        self.plan = {'output': str(self.output), 'metadata': {'reference': self.metadata},
            'expected_binding': check.refined_binding(base_binding(), {'qat_profile': check.PROFILE})}
        check.seal(self.plan_path, self.plan)
        def roster(base, seed, weight):
            value = copy.deepcopy(base); value.pop('body_sha256'); value['seed'] = seed
            value['successor_ranking']['loss_weight'] = weight; return hashed(value)
        def reference(directory, _architecture, _arm, seed):
            return directory / 'seed-references' / f'seed-{seed}.json'
        def output_artifact(directory, relative, *, expected_sha256, label):
            path = (directory / relative).resolve()
            self.assertTrue(path.is_relative_to(directory))
            if check.record(path)['sha256'] != expected_sha256: raise ValueError('changed ' + label)
            return path
        def load(directory, path, expected):
            value = check.read(path); receipt = check.read(check.verify(value['receipt']))
            if receipt['binding'] != expected: raise ValueError('source-specific binding changed')
            return receipt
        self.loads = []
        def loaded(directory, path, expected):
            self.loads.append((directory, path, expected)); return load(directory, path, expected)
        self.native = []
        def native(value):
            if value != {'threads': 1}: raise ValueError('not single threaded')
            self.native.append(value)
        trainer = types.SimpleNamespace(ARCHITECTURES={'capacity-12x8': object()}, ARMS={'search-target': object()},
            _seed_reference_path=reference, _output_artifact=output_artifact,
            _load_seed_receipt_from_reference=loaded, validate_native_thread_execution=native)
        self.engine = types.SimpleNamespace(trainer=trainer, process=types.SimpleNamespace(roster_binding=roster),
            check=types.SimpleNamespace(validate_reconstruction=lambda *_args: None,
                adapter=types.SimpleNamespace(_runtime_source=lambda runtime: b'export:' + runtime.read_bytes())))
        mapped = {split: {'container': 'tuple', 'groups': count, 'successors': count * 2,
                         'mapped_successors': True, 'all_backings_readonly_memmap': True}
                  for split, count in (('train', 829), ('validation', 165))}
        identity = {'fixture': 'complete-array-and-group-order-identity'}
        spec = check.worker_spec(self.plan_path, 'reference', identity, mapped, self.engine)
        spec_binding = check.seal(self.directory / 'worker-spec.json', spec)
        rows, process_evidence = [], []
        for ordinal, job in enumerate(spec['jobs']):
            directory = Path(job['directory']); directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / f'{job["seed"]}.float'; checkpoint.write_bytes(b'float')
            runtime = directory / f'{job["seed"]}.runtime'; runtime.write_bytes(b'runtime')
            receipt = {'binding': job['binding'], 'native_thread_execution': {'threads': 1},
                'float_checkpoint': {'path': checkpoint.name, 'sha256': check.record(checkpoint)['sha256']},
                'quantized_runtime': {'path': runtime.name, 'sha256': check.record(runtime)['sha256']}}
            receipt_binding = check.seal(directory / f'{job["seed"]}.receipt.json', receipt)
            path = reference(directory, None, None, job['seed'])
            reference_binding = check.seal(path, {'receipt': receipt_binding})
            source = directory / f'seed-{job["seed"]}.cpp'; source.write_bytes(b'export:runtime')
            rows.append({'weight': job['weight'], 'seed': job['seed'], 'reference': reference_binding,
                'receipt': check.read(Path(receipt_binding['path'])), 'source': check.record(source),
                'float_checkpoint': check.record(checkpoint), 'quantized_runtime': check.record(runtime)})
            process_evidence.append({'weight': job['weight'], 'seed': job['seed'], 'reference': reference_binding,
                'binding_sha256': job['binding']['body_sha256'], 'native_thread_execution': {'threads': 1},
                'process': {'pid': 100 + ordinal % 4}, 'cache_observation': observation()})
        self.result = {'schema': check.SCHEMA + '.cohort', 'plan': check.record(self.plan_path), 'engine_name': 'reference',
            'metadata': self.metadata, 'worker_spec': spec_binding, 'reconstruction': identity,
            'mapped_inputs': mapped, 'results': rows, 'process_evidence': process_evidence,
            'policy': check.POLICY, 'qualification_eligible': False}
        check.seal(self.directory / 'result.json', self.result)

    def validate(self):
        with patch.object(check, 'engine_for_plan', return_value=(self.engine, self.plan)):
            return check.validate_cohort(self.plan_path, 'reference')

    def rewrite(self, result):
        path = self.directory / 'result.json'; path.unlink(); check.seal(path, result)

    def test_all_nine_references_artifacts_and_native_contracts_reopen_in_their_own_engine(self):
        self.validate()
        self.assertEqual(len(self.loads), 9); self.assertEqual(len(self.native), 9)
        self.assertEqual([(row[2]['successor_ranking']['loss_weight'], row[2]['seed']) for row in self.loads],
                         [(weight, seed) for weight in check.WEIGHTS for seed in check.SEEDS])

    def test_historical_reference_redirect_or_substantive_binding_change_is_rejected(self):
        other = self.root / 'historical-reference.json'
        other.write_bytes(Path(self.result['results'][0]['reference']['path']).read_bytes())
        changed = copy.deepcopy(self.result); changed['results'][0]['reference'] = check.record(other); self.rewrite(changed)
        with self.assertRaisesRegex(ValueError, 'path or hash changed'): self.validate()
        self.assertEqual(self.loads, [])
        changed = copy.deepcopy(self.result)
        changed['results'][0]['receipt']['binding']['datasets']['new']['digest'] = 'wrong'
        self.rewrite(changed)
        with self.assertRaisesRegex(ValueError, 'whole substantive binding or receipt'): self.validate()

    def test_changed_runtime_bytes_or_eager_summary_cannot_pass_validation(self):
        runtime = Path(self.result['results'][0]['quantized_runtime']['path']); runtime.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed quantized_runtime'): self.validate()
        runtime.write_bytes(b'runtime')
        changed = copy.deepcopy(self.result); changed['mapped_inputs']['validation']['mapped_successors'] = False
        spec_path = self.directory / 'worker-spec.json'; spec = check.read(spec_path)
        spec['mapped_inputs'] = changed['mapped_inputs']; spec.pop('body_sha256'); spec_path.unlink()
        changed['worker_spec'] = check.seal(spec_path, spec); self.rewrite(changed)
        with self.assertRaisesRegex(ValueError, 'mapped-input evidence'): self.validate()


class ObserverTests(unittest.TestCase):
    def fake_engine(self, cached, mapped=True):
        def forward(parameters=None, architecture=None, active=(1,), *, quantized=None):
            return 7
        def metrics(cache=None):
            if cache is None or not cache.used:
                forward(quantized=None)
                if cache is not None: cache.used = True
            forward(quantized='fixed')
            return 7
        def factory(*_args):
            return types.SimpleNamespace(immutable_mapped_groups=mapped, used=False)
        trainer = types.SimpleNamespace(forward=forward, successor_ranking_metrics=metrics)
        if cached: trainer._new_float_ranking_decision_cache = factory
        def inner():
            cache = factory() if cached else None
            for _ in range(3): metrics(cache)
            return forward(quantized=None)
        def run(_job):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return {'unchanged_math_result': pool.submit(inner).result()}
        job = {'weight': .1, 'seed': check.SEEDS[0]}
        return types.SimpleNamespace(trainer=trainer, process=types.SimpleNamespace(_WORKER=({'jobs': [job]},), _run=run)), job

    def test_same_readonly_observer_counts_actual_calls_and_restores_thread_hook(self):
        values = []
        for cached in (False, True):
            engine, job = self.fake_engine(cached)
            with patch.object(check, '_ENGINE', engine): values.append(check.observed_run(job))
            self.assertIsNone(threading.getprofile())
        self.assertEqual(values[0]['unchanged_math_result'], values[1]['unchanged_math_result'])
        left, right = [value['cache_observation']['counts'] for value in values]
        self.assertEqual(left['float_ranking_forward_calls'], 3)
        self.assertEqual(right['float_ranking_forward_calls'], 1)
        self.assertEqual(left['quantized_ranking_forward_calls'], right['quantized_ranking_forward_calls'])
        self.assertEqual(right['immutable_mapped_factory_returns'], 1)
        self.assertEqual(right['other_float_forward_calls'], 1)

    def test_eager_factory_fallback_raises_without_leaving_a_profile_hook(self):
        engine, job = self.fake_engine(True, mapped=False)
        with patch.object(check, '_ENGINE', engine), self.assertRaisesRegex(ValueError, 'immutable mapped path'):
            check.observed_run(job)
        self.assertIsNone(threading.getprofile())

    def test_pool_adapter_only_observes_the_maintained_dispatch(self):
        function = object(); calls = []
        pool = types.SimpleNamespace(map=lambda target, values: calls.append((target, values)) or ['done'],
                                     shutdown=lambda **kwargs: calls.append(kwargs))
        wrapped = check.ObservedPool(pool, function)
        self.assertEqual(wrapped.map(function, ['job']), ['done'])
        self.assertIs(calls[0][0], check.observed_run)
        with self.assertRaisesRegex(ValueError, 'unexpected maintained'):
            wrapped.map(object(), ['job'])
        wrapped.shutdown(wait=True, cancel_futures=True)
        self.assertEqual(calls[-1], {'wait': True, 'cancel_futures': True})


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(); self.output = self.root / 'diagnostics/validation-cache-equivalence/check'
        self.output.mkdir(parents=True); self.plan_path = self.output / 'plan.json'
        self.plan = {'root': str(self.root), 'output': str(self.output)}
        check.seal(self.plan_path, self.plan)

    def test_active_driver_blocks_before_claim_or_heavy_reconstruction(self):
        with patch.object(check, 'validate_plan', return_value=self.plan), \
                patch.object(check, 'require_driver_finished', side_effect=ValueError('active driver')), \
                patch.object(check, 'subprocess_cohort') as cohort, self.assertRaisesRegex(ValueError, 'active driver'):
            check.run(self.plan_path)
        cohort.assert_not_called(); self.assertFalse((self.output / 'claim.json').exists())
        with patch.object(check, 'engine_for_plan', return_value=(object(), self.plan)), \
                patch.object(check, 'require_driver_finished', side_effect=ValueError('active driver')), \
                patch.object(check, 'reconstruct') as reconstruct, self.assertRaisesRegex(ValueError, 'active driver'):
            check.run_cohort(self.plan_path, 'reference', 3)
        reconstruct.assert_not_called()

    def test_busy_global_lock_and_spent_claim_never_execute(self):
        with (self.root / '.heavy-stage.lock').open('a') as held:
            check.fcntl.flock(held, check.fcntl.LOCK_EX | check.fcntl.LOCK_NB)
            with patch.object(check, 'validate_plan', return_value=self.plan), patch.object(check, 'require_driver_finished', return_value={'fixture': 'completed-driver'}), \
                    patch.object(check, 'subprocess_cohort') as cohort, self.assertRaises(BlockingIOError):
                check.run(self.plan_path)
            cohort.assert_not_called()
        check.seal(self.output / 'claim.json', {'spent': True})
        with patch.object(check, 'validate_plan', return_value=self.plan), patch.object(check, 'require_driver_finished', return_value={'fixture': 'completed-driver'}), \
                patch.object(check, 'subprocess_cohort') as cohort, self.assertRaisesRegex(ValueError, 'already claimed'):
            check.run(self.plan_path)
        cohort.assert_not_called()

    def test_reference_and_candidate_cohorts_are_sequential_and_no_activation_occurs(self):
        events = []
        def cohort(_plan, name, fd):
            self.assertGreaterEqual(fd, 0); events.append(name)
        comparison = check.seal(self.output / 'comparison.json', {'passed': True})
        with patch.object(check, 'validate_plan', return_value=self.plan), patch.object(check, 'require_driver_finished', return_value={'fixture': 'completed-driver'}), \
                patch.object(check, 'subprocess_cohort', side_effect=cohort), patch.object(check, 'validate_result', return_value=comparison):
            self.assertEqual(check.run(self.plan_path), comparison)
        self.assertEqual(events, ['reference', 'candidate'])
        result = check.read(self.output / 'result.json')
        self.assertTrue(result['completed']); self.assertFalse(result['production_activation_allowed'])

    def test_incomplete_first_cohort_stops_before_candidate_and_preserves_failure(self):
        with patch.object(check, 'validate_plan', return_value=self.plan), patch.object(check, 'require_driver_finished', return_value={'fixture': 'completed-driver'}), \
                patch.object(check, 'subprocess_cohort', side_effect=ValueError('failed real seed')) as cohort, \
                self.assertRaisesRegex(ValueError, 'failed real seed'):
            check.run(self.plan_path)
        self.assertEqual(cohort.call_count, 1)
        result = check.read(self.output / 'result.json'); self.assertFalse(result['completed'])
        self.assertEqual(result['failure']['message'], 'failed real seed')

    def test_abnormal_leader_exit_still_cleans_only_owned_group_and_records_logs(self):
        class Child:
            pid = 12345
            returncode = -9
            def wait(self, **_kwargs): return self.returncode
            def poll(self): return self.returncode
        with (self.root / '.heavy-stage.lock').open('a') as lock, \
                patch.object(check.subprocess, 'Popen', return_value=Child()) as launch, \
                patch.object(check.os, 'killpg') as kill, self.assertRaisesRegex(ValueError, 'owned equivalence cohort failed'):
            check.subprocess_cohort(self.plan_path, 'reference', lock.fileno())
        kill.assert_called_once_with(12345, check.signal.SIGKILL)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual(len(launch.call_args.kwargs['pass_fds']), 2)
        result = check.read(self.output / 'reference/execution.json')
        self.assertEqual(result['returncode'], -9); self.assertIsNotNone(result['failure'])
        check.verify(result['stdout']); check.verify(result['stderr'])

    def test_parent_death_watch_stops_its_dedicated_group(self):
        read_fd, write_fd = os.pipe(); os.close(write_fd)
        try:
            with patch.object(check.os, 'getpgrp', return_value=12345), patch.object(check.os, 'getpid', return_value=12345), \
                    patch.object(check.os, 'killpg') as kill:
                check.parent_watch(read_fd)
            kill.assert_called_once_with(12345, check.signal.SIGKILL)
        finally:
            os.close(read_fd)

    def test_prepare_runs_only_metadata_helpers_not_core_reconstruction_or_seeds(self):
        self.plan_path.unlink(); self.output.rmdir()
        descriptors = {}
        for name, commit in [('reference', check.REFERENCE_COMMIT), ('candidate', 'b' * 40)]:
            directory = self.root / 'source-snapshots' / commit; directory.mkdir(parents=True)
            archive = directory / 'source.tar'
            with tarfile.open(archive, 'w'): pass
            binding = check.seal(directory / 'snapshot.json', {'schema': check.ID + '.source-snapshot.v2',
                'commit': commit, 'repository': str(directory / 'repository'), 'archive': check.record(archive)})
            descriptors[name] = Path(binding['path'])
        driver = self.root / 'drivers/attempt-003-data-through-workspace/launch.json'
        check.seal(driver, {'fixture': 'active driver'})
        authorization = self.root / 'training-resources/more-cores-authorization.json'
        check.seal(authorization, {'fixture': 'authorization'})
        profile = {'qat_profile': check.PROFILE}
        def metadata(root, descriptor, output):
            name = 'reference' if descriptor['commit'] == check.REFERENCE_COMMIT else 'candidate'
            return check.seal(output, {'engine': descriptor, 'sources': [], 'smoke': {'expected_base_binding': base_binding()},
                'profile_contract': profile, 'refined_base_binding': check.refined_binding(base_binding(), profile),
                'runtime': {'python': 'same'}, 'has_cache_factory': name == 'candidate',
                'metadata_reconstructed_core_inputs': False, 'metadata_trained_seeds': False})
        with patch.object(check, '_metadata_process', side_effect=metadata) as meta, \
                patch.object(check, 'reconstruct') as reconstruct, patch.object(check, 'run_cohort') as train:
            plan = check.prepare(self.root, descriptors['candidate'], self.output, after_driver_launch=driver)
        self.assertEqual(meta.call_count, 2); reconstruct.assert_not_called(); train.assert_not_called()
        loaded = check.read(check.verify(plan)); self.assertEqual(loaded['policy']['fresh_seeds_per_engine'], 9)
        self.assertFalse(loaded['heavy_comparison_started'])


if __name__ == '__main__':
    unittest.main()
