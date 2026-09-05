"""Executor check planning, historical smoke routing and comparison contracts."""
import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from tools import compact_value_bfm_seed_process_v2 as process
os.environ.update(process.ENVIRONMENT)
os.environ[process.MARKER] = '1'

from tools import compact_value_bfm_seed_process_check_v2 as check
from tests.codingame import test_compact_value_bfm_seed_process_v2 as fixtures

campaign, trainer = check.campaign, check.trainer


class SeedProcessCheckTests(unittest.TestCase):
    def test_prepare_and_inspect_never_reconstruct_or_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'diagnostics/seed-process-equivalence/source-check'
            metadata = {'historical_audit': 'frozen'}
            with (mock.patch.object(check, 'smoke_inputs_metadata', return_value=metadata),
                  mock.patch.object(check, 'sources', return_value=[]),
                  mock.patch.object(check, 'reconstruct_smoke') as reconstruction,
                  mock.patch.object(trainer, 'train_seed_candidate') as training):
                record = check.prepare(root, output)
                plan = check.validate_plan(record['path'])
                self.assertEqual(plan['stage_order'], ['threads', 'spawn'])
                self.assertEqual(plan['seeds'], list(trainer.FIXED_SEEDS[:2]))
                self.assertEqual(plan['maximum_active_real_seeds'], 2)
                self.assertFalse(plan['real_training_started'])
                self.assertFalse(plan['qualification_eligible'])
                self.assertFalse(plan['automatic_production_opt_in'])
                self.assertTrue(plan['historical_seed_equivalence_required'])
                reconstruction.assert_not_called()
                training.assert_not_called()
                with mock.patch.object(check, 'sources', return_value=[{'changed': True}]):
                    with self.assertRaisesRegex(ValueError, 'source/runtime changed'):
                        check.validate_plan(record['path'])

    def test_four_worker_plan_has_nine_fresh_jobs_and_requires_authority(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / 'diagnostics/seed-process-equivalence/four-workers'
            auth = {'path': str(root / 'authorization.json'), 'sha256': 'a' * 64, 'bytes': 1}
            with (mock.patch.object(check, 'smoke_inputs_metadata', return_value={'historical': True}),
                  mock.patch.object(check, 'sources', return_value=[]),
                  mock.patch.object(resources, 'validate_authorization', return_value={}) as validation):
                with self.assertRaisesRegex(ValueError, 'source-bound resource authorization'):
                    check.prepare(root, output, executor=process.MODE4)
                validation.assert_not_called()
                record = check.prepare(root, output, executor=process.MODE4,
                    training_resource_authorization=auth)
                plan = check.validate_plan(record['path'])
                self.assertEqual(check.plan_seeds(plan), trainer.FIXED_SEEDS)
                self.assertEqual(plan['seeds'], list(trainer.FIXED_SEEDS))
                self.assertEqual(plan['maximum_active_real_seeds'], 4)
                self.assertEqual(plan['training_resource_authorization'], auth)
                self.assertTrue(plan['global_heavy_stage_lease_required'])
                self.assertTrue(plan['historical_seed_equivalence_required'])
                self.assertEqual(validation.call_args.args, (auth, root))

    def test_global_lock_cannot_be_removed_without_authorized_concurrent_context(self):
        for executor in (process.MODE2, process.MODE4):
            with self.subTest(executor=executor):
                plan = {'root': '/fixture', 'executor': executor,
                    'global_heavy_stage_lease_required': False}
                if executor == process.MODE4:
                    plan['training_resource_authorization'] = {'frozen': True}
                from tools import compact_value_bfm_training_resources_v2 as resources
                with mock.patch.object(resources, 'validate_authorization', return_value={}):
                    with self.assertRaisesRegex(ValueError, 'global heavy-stage lease'):
                        check.validate_execution_authority(plan)

    def test_explicit_concurrent_context_delegates_locking_and_disables_speedup_claim(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / 'diagnostics/seed-process-equivalence/concurrent-four'
            auth = {'path': str(root / 'auth.json')}
            context_record = {'path': str(root / 'context.json')}
            context = {'root': str(root), 'output': str(output),
                'training_resource_authorization': auth, 'validation_lock': str(root / '.training-validation.lock')}
            module = types.ModuleType('tools.compact_value_bfm_training_acceleration_v2')
            module.validate_context = mock.Mock(return_value=context)
            module.run = mock.Mock(return_value={'controller': 'owns validation lock'})
            with (mock.patch.object(check, '_concurrency_controller', return_value=module),
                  mock.patch.object(resources, 'validate_authorization', return_value={}),
                  mock.patch.object(check, 'smoke_inputs_metadata', return_value={'historical': True}),
                  mock.patch.object(check, 'sources', return_value=[])):
                record = check.prepare(root, output, executor=process.MODE4,
                    training_resource_authorization=auth, concurrent_context=context_record)
                plan = check.validate_plan(record['path'])
                self.assertFalse(plan['global_heavy_stage_lease_required'])
                self.assertEqual(plan['execution_timing'], 'contended-equivalence-only')
                self.assertFalse(plan['speedup_qualification_allowed'])
                with mock.patch.object(campaign, 'lease', side_effect=AssertionError('controller owns separate lock')):
                    self.assertEqual(check.run(Path(record['path'])), {'controller': 'owns validation lock'})
                module.run.assert_called_once_with(context_record, Path(record['path']))
                invalid = {**plan, 'speedup_qualification_allowed': True}
                with self.assertRaisesRegex(ValueError, 'speedup qualification'):
                    check.validate_execution_authority(invalid)
                module.validate_context.return_value = {**context, 'output': str(root / 'other')}
                with self.assertRaisesRegex(ValueError, 'output or resource authorization'):
                    check.validate_execution_authority(plan)

    def test_outputs_cannot_enter_any_campaign_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for output in (root / 'smoke-064', root / 'phases/attempt-001-pilot',
                           root / 'diagnostics/seed-process-equivalence', root.parent / 'outside'):
                with self.assertRaisesRegex(ValueError, 'fresh child'):
                    check.output_path(root, output)

    def test_busy_global_heavy_lease_prevents_claim_and_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (mock.patch.object(check, 'validate_plan', return_value={'root': str(root)}),
                  mock.patch.object(check, 'run_stage') as stage,
                  campaign.lease(root)):
                with self.assertRaises(BlockingIOError):
                    check.run(root / 'plan.json')
                stage.assert_not_called()
            self.assertFalse(any(root.rglob('claim.json')))

    def test_preexisting_seed_artifact_cannot_short_circuit_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'spawn'
            directory.mkdir()
            (directory / 'preexisting-seed.json').write_bytes(b'old seed')
            with (mock.patch.object(check, 'validate_plan', return_value={'output': str(root)}),
                  mock.patch.object(check, 'reconstruct_smoke') as reconstruction):
                with self.assertRaisesRegex(ValueError, 'fresh output directories'):
                    check.run_stage(root / 'plan.json', 'spawn')
                reconstruction.assert_not_called()
            self.assertFalse((directory / 'claim.json').exists())

    def test_historical_smoke_uses_eligible_positions_for_anchor_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures.write_fixture(root)
            phase = root / 'pilot'
            # Distinct documents deliberately have different early-state rows.
            full = phase / 'full-preflight.json'
            eligible = phase / 'eligible-positions.json'
            campaign.seal(full, {'rows': []})
            campaign.seal(eligible, {'rows': [{'split': 'train', 'drawn_edges': 0, 'prefix': ''}]})
            original = campaign.read(phase / 'training-input-audit.json')
            label = campaign.read(phase / 'labels.json')
            labels_path = phase / 'historical-labels.json'
            campaign.seal(labels_path, {**{key: value for key, value in label.items() if key != 'body_sha256'},
                'positions': campaign.record(eligible)})
            audit_path = phase / 'historical-audit.json'
            campaign.seal(audit_path, {**{key: value for key, value in original.items() if key != 'body_sha256'},
                'position_closure': campaign.record(full), 'labels': campaign.record(labels_path)})
            initial = campaign.read(root / 'campaign.json')['inputs']['attempt_one_initial_checkpoint']
            metadata = {'audit': campaign.record(audit_path), 'labels': campaign.record(labels_path),
                'audited_position_closure': campaign.record(full),
                'eligible_positions_used_for_anchor_filter': campaign.record(eligible),
                'expected_anchor_rows_removed': 1, 'initial_checkpoint': initial,
                'expected_base_binding': {'synthetic': True}}
            with fixtures.reconstruction_fixture(root), mock.patch.object(trainer, 'training_binding',
                    return_value={'synthetic': True}) as binding:
                _bundle, inputs, identity = check.reconstruct_smoke({'smoke': metadata})
            self.assertEqual(identity['anchor_filter']['removed_rows'], 1)
            self.assertEqual(len(inputs.anchor), 2)
            self.assertEqual(inputs.input_audit['position_closure'], campaign.record(full))
            self.assertEqual(inputs.input_audit['labels'], campaign.record(labels_path))
            self.assertEqual(trainer.dataset_identity(inputs.canonical_validation),
                trainer.dataset_identity(fixtures.core_datasets()[2]))
            self.assertIs(binding.call_args.args[1], inputs)
            self.assertFalse(inputs.anchor.targets.flags.writeable)

    def stage_pair(self, workers=2):
        executor = process.MODE4 if workers == 4 else process.MODE2
        seeds = trainer.FIXED_SEEDS if workers == 4 else check.SEEDS
        with trainer.native_thread_execution_scope() as execution:
            rows = [{'weight': weight, 'seed': seed,
                'receipt': {'seed': seed, 'native_thread_execution': execution,
                    'float_training': {'gradient_updates': 'abc'},
                    'quantized_training': {'scale_search': {'trials': [{'scales': [.1, .2, .3]}]}},
                    'quantized_validation': {'loss': .13}},
                **{key: {'sha256': key, 'bytes': 100} for key in
                   ('float_checkpoint', 'quantized_runtime', 'source')}}
                for weight in check.WEIGHTS for seed in seeds]
        thread = {'stage': 'threads', 'plan': {'frozen': True}, 'reconstruction': {'digest': 'same'},
            'results': rows, 'elapsed_seconds': 20, 'memory': {'samples': 2, 'errors': [],
                'process_tree_peak_rss': 200, 'per_pid_peak_rss': {'100': 80, '101': 50, '102': 60}},
            'process_evidence': []}
        spawn = {**copy.deepcopy(thread), 'stage': 'spawn', 'elapsed_seconds': 10,
            'process_evidence': [{'process': {'pid': 101}}, {'process': {'pid': 102}}]}
        if workers == 4:
            thread['executor'] = executor
            spawn['executor'] = executor
            spawn['process_evidence'] += [{'process': {'pid': 103}}, {'process': {'pid': 104}}]
            spawn['memory']['per_pid_peak_rss'].update({'103': 50, '104': 60})
        return thread, spawn

    def test_four_worker_comparison_requires_all_nine_jobs_and_four_observed_children(self):
        thread, spawn = self.stage_pair(4)
        result = check.compare_stage_results(thread, spawn)
        self.assertEqual(len(result['checks']), 9)
        self.assertTrue(result['four_spawn_children_observed'])
        self.assertTrue(result['requested_spawn_children_observed'])
        self.assertTrue(result['executor_ready_for_review'])
        historical = [row for row in thread['results'] if row['seed'] == check.SEEDS[0]]
        self.assertEqual(len(check.compare_historical_results(historical, thread)), 3)
        incomplete = copy.deepcopy(spawn)
        incomplete['results'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete seed roster'):
            check.compare_stage_results(thread, incomplete)
        spawn['process_evidence'].pop()
        result = check.compare_stage_results(thread, spawn)
        self.assertFalse(result['four_spawn_children_observed'])
        self.assertFalse(result['executor_ready_for_review'])

    def test_contended_equivalence_records_no_elapsed_speedup_ratio(self):
        thread, spawn = self.stage_pair(4)
        for stage in (thread, spawn):
            stage.update({'concurrent_equivalence_context': {'frozen': True},
                'execution_timing': 'contended-equivalence-only', 'speedup_qualification_allowed': False})
        result = check.compare_stage_results(thread, spawn)
        self.assertTrue(result['exact_equivalence_passed'])
        self.assertTrue(result['executor_ready_for_review'])
        self.assertIsNone(result['observed_thread_over_spawn_elapsed_ratio'])
        self.assertFalse(result['speedup_qualification_allowed'])

    def test_comparison_requires_full_receipts_scales_and_export_equality(self):
        thread, spawn = self.stage_pair()
        result = check.compare_stage_results(thread, spawn)
        self.assertTrue(result['exact_equivalence_passed'])
        self.assertTrue(result['two_spawn_children_observed'])
        self.assertEqual(result['observed_thread_over_spawn_elapsed_ratio'], 2)
        self.assertTrue(result['performance_is_single_ordered_experiment'])
        self.assertFalse(result['qualification_eligible'])
        for field in ('source', 'float_checkpoint', 'quantized_runtime'):
            changed = copy.deepcopy(spawn)
            changed['results'][0][field]['sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'full maintained seed differs'):
                check.compare_stage_results(thread, changed)
        changed = copy.deepcopy(spawn)
        changed['results'][0]['receipt']['quantized_training']['scale_search']['trials'][0]['scales'][0] = .11
        with self.assertRaisesRegex(ValueError, 'full maintained seed differs'):
            check.compare_stage_results(thread, changed)
        changed = copy.deepcopy(spawn)
        changed['results'][0]['receipt']['float_training']['gradient_updates'] = 'different'
        with self.assertRaisesRegex(ValueError, 'full maintained seed differs'):
            check.compare_stage_results(thread, changed)

    def test_no_memory_or_single_child_is_reported_as_incomplete_observation(self):
        thread, spawn = self.stage_pair()
        spawn['process_evidence'] = [{'process': {'pid': 101}}]
        spawn['memory']['errors'] = ['sampling failed']
        result = check.compare_stage_results(thread, spawn)
        self.assertFalse(result['two_spawn_children_observed'])
        self.assertFalse(result['memory_measurements_complete'])
        self.assertFalse(result['executor_ready_for_review'])
        self.assertFalse(result['automatic_production_opt_in'])

    def test_agreeing_new_executors_must_also_match_original_smoke_training(self):
        thread, spawn = self.stage_pair()
        historical = copy.deepcopy([row for row in thread['results'] if row['seed'] == check.SEEDS[0]])
        self.assertEqual(len(check.compare_historical_results(historical, thread)), 3)
        for stage in (thread, spawn):
            stage['results'][0]['receipt']['float_training']['gradient_updates'] = 'same-new-drift'
        self.assertTrue(check.compare_stage_results(thread, spawn)['exact_equivalence_passed'])
        with self.assertRaisesRegex(ValueError, 'full maintained seed differs'):
            check.compare_historical_results(historical, thread)
        with self.assertRaisesRegex(ValueError, 'three original smoke recipes'):
            check.compare_historical_results(historical[:-1], thread)

    def test_historical_rows_reopen_original_references_and_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            output = directory / 'training/lambda-0.10'
            receipt = {'binding': {'frozen': True}}
            for key in ('float_checkpoint', 'quantized_runtime'):
                path = output / (key + '.bin'); campaign.once(path, key.encode())
                receipt[key] = {'path': path.name, 'sha256': campaign.sha(path)}
            reference = trainer._seed_reference_path(output, trainer.ARCHITECTURES['capacity-12x8'],
                trainer.ARMS['search-target'], check.SEEDS[0])
            campaign.once(reference, b'fixture reference')
            source = output / f'seed-{check.SEEDS[0]}.cpp'; campaign.once(source, b'fixture source')
            row = {'weight': .1, 'seed': check.SEEDS[0], 'seed_receipt': receipt,
                'float_checkpoint': campaign.record(output / receipt['float_checkpoint']['path']),
                'runtime': campaign.record(output / receipt['quantized_runtime']['path']),
                'source': campaign.record(source)}
            with mock.patch.object(trainer, '_load_seed_receipt_from_reference', return_value=receipt) as load, \
                    mock.patch.object(check.adapter, '_runtime_source', return_value=b'fixture source'):
                checked = check.historical_seed_results(directory, {'results': [row]})
                load.assert_called_once_with(output, reference, receipt['binding'])
                self.assertEqual(checked[0]['reference'], campaign.record(reference))
                replacement = directory / 'other-source.cpp'; campaign.once(replacement, source.read_bytes())
                row['source'] = campaign.record(replacement)
                with self.assertRaisesRegex(ValueError, 'canonical output path'):
                    check.historical_seed_results(directory, {'results': [row]})
                row['source'] = campaign.record(source)
                external = directory / 'external-reference.json'; campaign.once(external, reference.read_bytes())
                reference.unlink(); reference.symlink_to(external)
                load.side_effect = AssertionError('redirected reference must reject before loading')
                with self.assertRaisesRegex(ValueError, 'canonical output path'):
                    check.historical_seed_results(directory, {'results': [row]})

    def test_elapsed_times_must_be_finite_positive_before_ratio(self):
        thread, spawn = self.stage_pair()
        for value in (0, -1, True, float('inf'), float('nan')):
            changed = copy.deepcopy(spawn)
            changed['elapsed_seconds'] = value
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                check.compare_stage_results(thread, changed)

    def test_canonical_binding_rejects_historical_reference_redirection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root / 'old/seed-reference.json'
            historical.parent.mkdir()
            historical.write_bytes(b'historical seed')
            with self.assertRaisesRegex(ValueError, 'canonical output path'):
                check.bound(campaign.record(historical), root / 'new/seed-reference.json')
            link = root / 'redirected'
            link.symlink_to(historical.parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'canonical output path'):
                check.bound({'path': str(link / historical.name)}, link / historical.name)

    def test_worker_spec_is_rederived_from_plan_not_trusted_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan_path = root / 'plan.json'
            campaign.seal(plan_path, {'output': str(root), 'smoke': {'expected_base_binding': {'synthetic': True}}})
            with mock.patch.object(process, 'roster_binding', side_effect=lambda base, seed, weight: {
                    'seed': seed, 'weight': weight, 'body_sha256': f'{seed}-{weight}'}):
                body = check.expected_worker_spec(plan_path, 'spawn', {'identity': 'fixture'})
                self.assertEqual(len(body['jobs']), 6)
                for job in body['jobs']:
                    self.assertEqual(Path(job['directory']), root / 'spawn' / f'lambda-{job["weight"]:.2f}')
                    self.assertEqual(job['binding']['seed'], job['seed'])
                    self.assertEqual(job['binding']['weight'], job['weight'])
                directory = root / 'spawn'
                directory.mkdir()
                campaign.seal(directory / 'claim.json', check.expected_claim(plan_path, 'spawn'))
                body['jobs'][0]['directory'] = str(root / 'old-output')
                campaign.seal(directory / 'worker-spec.json', body)
                campaign.seal(directory / 'result.json', {
                    'schema': check.SCHEMA + '.stage', 'stage': 'spawn', 'plan': campaign.record(plan_path),
                    'claim': campaign.record(directory / 'claim.json'),
                    'specification': campaign.record(directory / 'worker-spec.json'),
                    'reconstruction': {'identity': 'fixture'}, 'qualification_eligible': False})
                with (mock.patch.object(check, 'validate_plan', return_value={'output': str(root)}),
                      mock.patch.object(check, 'validate_reconstruction')):
                    with self.assertRaisesRegex(ValueError, 'exact planned jobs'):
                        check.validated_stage(plan_path, 'spawn')


if __name__ == '__main__':
    unittest.main()
