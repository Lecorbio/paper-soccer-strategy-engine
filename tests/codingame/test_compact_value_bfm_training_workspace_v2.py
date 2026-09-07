"""Small numerical/metadata fixtures; no actual workspace or corpus execution."""
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

PROBE_PATH = Path(__file__).resolve().parents[2] / 'tools/compact_value_bfm_training_workspace_v2.py'
spec = importlib.util.spec_from_file_location('workspace_probe_unit', PROBE_PATH)
probe = importlib.util.module_from_spec(spec); sys.modules[spec.name] = probe; spec.loader.exec_module(probe)
os.environ.update(probe.ENVIRONMENT); os.environ[probe.MARKER] = '1'
import numpy as np
from tools import compact_value_bfm_train as trainer


class Rows:
    def __init__(self, count): self.count = count
    def __len__(self): return self.count
    def active_rows(self, rows): return tuple(np.asarray([0, 1, 2], dtype='<u2') for _ in rows)


def group(name, count, profile='standard-v1'):
    return SimpleNamespace(group_id=name, evidence={'work_budget': {'teacher_ranking_profile': profile}},
        successors=[SimpleNamespace(active=np.asarray([0, 1, 2], dtype='<u2')) for _ in range(count)])


def inputs():
    return SimpleNamespace(new=Rows(128), anchor=Rows(192),
        successor_rankings=SimpleNamespace(train=[group('c', 7), group('a', 9), group('b', 7), group('d', 3)],
                                          validation=[group('v-small', 2), group('v-largest', 4)]))


class WorkspaceTests(unittest.TestCase):
    def test_standalone_help_imports_no_engine_and_works_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(PROBE_PATH), '--help'], cwd=tmp,
                                    capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare', result.stdout)

    def test_mixed_namespace_and_foreign_cached_exporter_alias_are_rejected(self):
        fake = SimpleNamespace(__path__=['/different/tools'])
        with mock.patch.dict(sys.modules, {'tools': fake}):
            with self.assertRaisesRegex(ValueError, 'fresh interpreter'):
                probe.isolate_engine_namespace(Path('/frozen/engine'))
        script = '''import importlib.util,sys,types
from pathlib import Path
spec=importlib.util.spec_from_file_location('workspace_probe',sys.argv[1]);p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
engine=Path(sys.argv[2]).resolve();(engine/'tools').mkdir(parents=True)
alias=types.ModuleType('teacher_training_model_exporter');alias.__file__='/foreign/export_model.py';sys.modules[alias.__name__]=alias
try:p.isolate_engine_namespace(engine)
except ValueError:print('rejected')
else:raise AssertionError('foreign alias survived')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', script, str(PROBE_PATH), str(Path(tmp)/'engine')],
                                    cwd=tmp, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'rejected')

    def test_fresh_namespace_is_pinned_and_probe_remains_top_level_importable(self):
        script = '''import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('workspace_probe',sys.argv[1]);p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
engine=Path(sys.argv[2]).resolve();other=engine.parent/'other';(engine/'tools').mkdir(parents=True);(other/'tools').mkdir(parents=True)
(engine/'tools'/'fixture.py').write_text("MARKER='frozen'")
(other/'tools'/'fixture.py').write_text("MARKER='wrong'")
sys.path.insert(0,str(other));p.isolate_engine_namespace(engine)
import tools.fixture
assert tools.fixture.MARKER=='frozen'
assert list(tools.__path__)==[str(engine/'tools')]
assert str(Path(sys.argv[1]).parent) in sys.path
assert str(Path(sys.argv[1]).parents[1]) not in sys.path
print('isolated')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', script, str(PROBE_PATH), str(Path(tmp)/'engine')],
                                    cwd=tmp, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'isolated')

    def test_bootstrap_rejects_unbound_engine_before_any_tools_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'capacity-plan.json'
            value = {'root': str(Path(tmp)), 'sources': []}
            value['body_sha256'] = probe.hashlib.sha256(probe.raw(value)).hexdigest()
            path.write_bytes(probe.raw(value))
            with mock.patch.object(probe, 'isolate_engine_namespace') as isolate:
                with self.assertRaisesRegex(ValueError, 'fa012e7'):
                    probe.bootstrap(path)
            isolate.assert_not_called()

    def seal_fixture(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {**body, 'body_sha256': probe.hashlib.sha256(probe.raw(body)).hexdigest()}
        path.write_bytes(probe.raw(value)); return probe.record(path)

    def route_fixture(self, root, commit):
        root = root.resolve()
        engine = root / 'source-snapshots' / commit / 'repository'
        tools = engine / 'tools'; tools.mkdir(parents=True)
        names = ('compact_value_bfm_training_capacity_v2.py', 'compact_value_bfm_seed_process_v2.py',
                 'compact_value_bfm_train.py', 'compact_value_bfm_campaign_v2.py', 'compact_value_bfm_ranking_store.py',
                 'compact_value_bfm_teacher_training.py', 'compact_value_bfm_intervention_v2.py')
        for name in names: (tools / name).write_text('# exact fixture ' + name)
        extra = engine / 'submissions/codingame/bots/compact_value_bfm'; extra.mkdir(parents=True)
        for name in ('export_model.py', 'export_submission.py', 'rank4_gate_support.py'):
            (extra / name).write_text('# frozen exporter ' + name)
        archive = engine.parent / 'source.tar'
        with tarfile.open(archive, 'w:') as tar:
            for path in sorted(engine.rglob('*.py')): tar.add(path, arcname=path.relative_to(engine).as_posix())
        self.seal_fixture(engine.parent / 'snapshot.json', {'schema': 'compact-value-bfm-trained-v2.source-snapshot.v2',
            'commit': commit, 'repository': str(engine), 'archive': probe.record(archive)})
        plan = {'root': str(root), 'sources': [probe.record(path) for path in sorted(tools.glob('*.py'))]}
        return plan, engine, probe.record(archive)['sha256']

    def test_829_requires_explicit_route_and_authenticated_complete_provider_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, engine, digest = self.route_fixture(Path(tmp), probe.RETENTION_ENGINE_COMMIT)
            with self.assertRaisesRegex(ValueError, 'fa012e7'): probe.engine_route(plan)
            with self.assertRaisesRegex(ValueError, 'explicitly supported'): probe.engine_route(plan, '0'*40)
            with mock.patch.object(probe, 'RETENTION_ENGINE_ARCHIVE_SHA256', digest):
                root, actual, sources, binding = probe.engine_route(plan, probe.RETENTION_ENGINE_COMMIT)
                self.assertEqual(actual, engine); self.assertEqual(len(sources), 7)
                self.assertEqual(binding['snapshot'], probe.record(engine.parent / 'snapshot.json'))
                self.assertTrue(binding['archive_provider_bytes_verified'])
                for mutation in ('missing', 'duplicate', 'order', 'foreign-path', 'source-bytes', 'exporter-bytes', 'missing-file', 'snapshot-link', 'snapshot', 'archive'):
                    changed = copy.deepcopy(plan)
                    snapshot_path = engine.parent / 'snapshot.json'; old_snapshot = snapshot_path.read_bytes()
                    touched = None; old_bytes = None
                    if mutation == 'missing': changed['sources'] = changed['sources'][:-1]
                    elif mutation == 'duplicate': changed['sources'].append(changed['sources'][0])
                    elif mutation == 'order': changed['sources'].reverse()
                    elif mutation == 'foreign-path':
                        foreign = Path(tmp) / 'foreign.py'; foreign.write_bytes(Path(changed['sources'][0]['path']).read_bytes())
                        changed['sources'][0] = probe.record(foreign)
                    elif mutation == 'missing-file':
                        touched = engine / 'tools/compact_value_bfm_intervention_v2.py'; old_bytes = touched.read_bytes(); touched.unlink()
                        changed['sources'] = [item for item in changed['sources'] if Path(item['path']) != touched]
                    elif mutation == 'snapshot-link':
                        external = engine.parent / 'redirected-snapshot.json'; external.write_bytes(old_snapshot)
                        snapshot_path.unlink(); snapshot_path.symlink_to(external)
                    elif mutation in ('source-bytes', 'exporter-bytes'):
                        touched = Path(changed['sources'][0]['path']) if mutation == 'source-bytes' else engine / 'submissions/codingame/bots/compact_value_bfm/export_model.py'
                        old_bytes = touched.read_bytes(); touched.write_bytes(old_bytes + b'\n# substituted')
                        if mutation == 'source-bytes': changed['sources'][0] = probe.record(touched)
                    else:
                        snapshot = probe.sealed(snapshot_path); snapshot.pop('body_sha256')
                        if mutation == 'snapshot': snapshot['commit'] = probe.ENGINE_COMMIT
                        else: snapshot['archive']['sha256'] = '0'*64
                        self.seal_fixture(snapshot_path, snapshot)
                    try:
                        with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                            probe.engine_route(changed, probe.RETENTION_ENGINE_COMMIT)
                    finally:
                        if snapshot_path.is_symlink(): snapshot_path.unlink()
                        snapshot_path.write_bytes(old_snapshot)
                        if touched is not None: touched.write_bytes(old_bytes)

    def test_legacy_engine_route_keeps_default_without_new_snapshot_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, engine, _digest = self.route_fixture(Path(tmp), probe.ENGINE_COMMIT)
            (engine.parent / 'snapshot.json').unlink(); (engine.parent / 'source.tar').unlink()
            _root, actual, _sources, route = probe.engine_route(plan)
            self.assertEqual(actual, engine); self.assertIsNone(route)
            self.assertIsNone(probe.phase_recipe_binding(SimpleNamespace(engine_commit=probe.ENGINE_COMMIT), {}))

    def test_829_recipe_checks_actual_intervention_profile_audit_and_worker_authority(self):
        root = Path('/fixture').resolve()
        profile = trainer.qat_profile_contract(probe.RETENTION_PROFILE)
        contract = {'attempt': 4, 'phase': 'pilot', 'qat_profile': probe.RETENTION_PROFILE, 'qat_profile_contract': profile,
            'training_executor': {'mode': 'spawn-v2', 'maximum_workers': 4}, 'training_resource_authorization': {'authority': True},
            'intervention': {'actual': 4}}
        base = {'phase': 'attempt-004-pilot', 'phase_contract': {'contract': True}, 'input_audit': {'audit': 'actual corpus'},
            'training_resource_authorization': contract['training_resource_authorization'],
            'policy': {'workers': 4, 'start_method': 'spawn', 'numerical_threads_per_worker': 1}}
        expected = mock.Mock(return_value=probe.RETENTION_PROFILE)
        intervention = SimpleNamespace(__file__=str(root / 'source-snapshots' / probe.RETENTION_ENGINE_COMMIT /
            'repository/tools/compact_value_bfm_intervention_v2.py'), expected_qat_profile=expected)
        engine = SimpleNamespace(root=root, engine_commit=probe.RETENTION_ENGINE_COMMIT,
            campaign=SimpleNamespace(read=lambda _p: contract), trainer=trainer,
            resources=SimpleNamespace(expected_workers=lambda _c: 4))
        with mock.patch.object(probe, 'verified', side_effect=lambda value: value), \
                mock.patch.object(probe.importlib, 'import_module', return_value=intervention):
            binding = probe.phase_recipe_binding(engine, base)
            expected.assert_called_once_with(contract)
            self.assertEqual(binding['input_audit'], base['input_audit']); self.assertEqual(binding['qat_profile_contract'], profile)
            for key, value in (('attempt', 3), ('phase', 'full'), ('qat_profile', 'refined-adaptive-scales-v1'),
                               ('qat_profile_contract', {}), ('training_executor', {'mode': 'spawn-v2', 'maximum_workers': 2}),
                               ('training_resource_authorization', {})):
                old = contract[key]; contract[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'profile/resource'): probe.phase_recipe_binding(engine, base)
                contract[key] = old
            for key, value in (('workers', 3), ('numerical_threads_per_worker', 2), ('start_method', 'fork')):
                old = base['policy'][key]; base['policy'][key] = value
                with self.assertRaisesRegex(ValueError, 'profile/resource'): probe.phase_recipe_binding(engine, base)
                base['policy'][key] = old
            intervention.__file__ = '/foreign/compact_value_bfm_intervention_v2.py'
            with self.assertRaisesRegex(ValueError, 'another source'): probe.phase_recipe_binding(engine, base)

    def test_legacy_plan_shape_and_new_recipe_source_binding_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); output = root / 'diagnostics/training-workspace/fixture'
            base_path = root / 'capacity/plan.json'; self.seal_fixture(base_path, {'fixture': True})
            self.seal_fixture(base_path.parent / 'result.json', {'fixture': True})
            contract_path = root / 'campaign.json'
            initial = root / 'original.npz'; initial.write_bytes(b'fixture original')
            runtime = root / 'deployed.json'; runtime.write_bytes(b'fixture runtime')
            contract = {'inputs': {'attempt_one_initial_checkpoint': probe.record(initial), 'attempt_zero_runtime': probe.record(runtime)}}
            self.seal_fixture(contract_path, contract)
            base = {'output': str(base_path.parent), 'root': str(root), 'context': str(root / 'context'), 'phase': 'fixture',
                    'phase_contract': probe.record(contract_path)}
            complete = {'memory_measurements_complete': True, 'coordinator': {'identity': {'audited': True}},
                        'workspace_dimensions_metadata_only': {'N': 20}}
            engine = SimpleNamespace(root=root, engine_commit=probe.ENGINE_COMMIT, engine_route=None, sources=['exact'], provider_sources={'exporter': 'exact'},
                campaign=SimpleNamespace(read=lambda path: json.loads(Path(path).read_bytes()), seal=lambda path, body: self.seal_fixture(Path(path), body)),
                capacity=SimpleNamespace(validate_plan=lambda _p: base, validate_result=lambda _p: complete, runtime=lambda: {}), trainer=trainer)
            with mock.patch.object(probe, 'bootstrap', return_value=engine) as bootstrap, \
                    mock.patch.object(probe, 'validation_envelope', return_value={'V': 3}), mock.patch.object(probe, 'phase_recipe_binding', return_value=None) as recipe:
                binding = probe.prepare(base_path, output)
                original = probe.sealed(binding['path']); probe.validate_plan(binding['path'])
                self.assertNotIn('engine_route', original); self.assertNotIn('phase_recipe', original)
                self.assertEqual(original['engine_commit'], probe.ENGINE_COMMIT)
                self.assertEqual(set(original), {'schema','capacity_plan','capacity_result','output','root','context','phase','inputs',
                    'engine_commit','engine_sources','adapter_dependency_sources','probe_source','runtime','numpy_version','policy',
                    'expected_input_identity','training_envelope','validation_envelope','hold_seconds','reconstruction_timeout_seconds',
                    'metadata_only_preparation','body_sha256'})
                engine.engine_commit = probe.RETENTION_ENGINE_COMMIT; engine.engine_route = {'archive': 'authenticated'}
                recipe.return_value = {'input_audit': 'actual', 'qat_profile': probe.RETENTION_PROFILE, 'maximum_workers': 4}
                newer = root / 'diagnostics/training-workspace/newer'
                new_binding = probe.prepare(base_path, newer, engine_commit=probe.RETENTION_ENGINE_COMMIT)
                new_plan = probe.sealed(new_binding['path']); probe.validate_plan(new_binding['path'])
                self.assertEqual(new_plan['engine_route'], engine.engine_route)
                self.assertEqual(new_plan['phase_recipe'], recipe.return_value)
                self.assertEqual(bootstrap.call_args.kwargs['engine_commit'], probe.RETENTION_ENGINE_COMMIT)
                for key in ('engine_route', 'phase_recipe'):
                    bad = copy.deepcopy(new_plan); bad.pop('body_sha256'); bad[key] = {'substituted': True}
                    self.seal_fixture(Path(new_binding['path']), bad)
                    with self.assertRaisesRegex(ValueError, 'snapshot/profile/input/resource'): probe.validate_plan(new_binding['path'])
                bad = copy.deepcopy(original); bad.pop('body_sha256'); bad['engine_route'] = {'forged': True}
                self.seal_fixture(Path(binding['path']), bad); engine.engine_commit = probe.ENGINE_COMMIT; recipe.return_value = None
                with self.assertRaisesRegex(ValueError, 'historical FA'): probe.validate_plan(binding['path'])

    def test_top_k_all_group_envelope_and_largest_validation_are_deterministic(self):
        selected, validation = probe.select_groups(trainer, inputs())
        self.assertEqual([item.group_id for item in selected], ['a', 'b'])
        summary = probe.selection_summary(selected, validation)
        self.assertEqual((summary['K'], summary['N'], summary['V']), (2, 16, 4))
        plan = {'training_envelope': {'unweighted_all_groups_upper_bound_used': True,
                                     'ranking_groups_per_batch_upper_bound': 2, 'successors_per_batch_conservative_upper_bound': 16},
                'validation_envelope': {'V': 4, 'validation_group_id': 'v-largest'}}
        probe.validate_selection(summary, plan)
        for key, value in (('K', 1), ('N', 15), ('V', 3)):
            bad = copy.deepcopy(summary); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): probe.validate_selection(bad, plan)
        changed = inputs(); changed.successor_rankings.train[0].evidence['work_budget']['teacher_ranking_profile'] = 'hardest-5pct-2m-v1'
        with self.assertRaisesRegex(ValueError, 'standard unweighted'):
            probe.select_groups(trainer, changed)

    def test_return_hook_retains_dense_arrays_without_changing_gradient_bits(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
        active = tuple(np.asarray([0, 1, 2], dtype='<u2') for _ in range(3))
        _, cache = trainer.forward(parameters, architecture, active)
        derivative = np.asarray([.01, 0, -.01], dtype=np.float32)
        args = parameters, architecture, active, cache, derivative, parameters
        expected = trainer._network_gradients(*args)
        previous = sys.getprofile(); events = []
        def prior(frame, event, result): events.append(event)
        try:
            sys.setprofile(prior)
            gradients, retained = probe.retain_backward(trainer._network_gradients, *args)
            self.assertIs(sys.getprofile(), prior)
        finally: sys.setprofile(previous)
        for name in expected: self.assertTrue(np.array_equal(expected[name], gradients[name]))
        self.assertEqual(retained['first_pre_gradient'].shape, (3, 12))
        self.assertEqual(retained['second_gradient'].shape, (3, 8))
        reference = weakref.ref(retained['first_pre_gradient'])
        gc.collect(); self.assertIsNotNone(reference())
        del retained; gc.collect(); self.assertIsNone(reference())
        self.assertTrue(events)

    def test_return_hook_restores_previous_profile_on_exception_and_ignores_other_returns(self):
        previous = sys.getprofile()
        def prior(frame, event, result): pass
        def unrelated():
            wrong_frame_local = True
            return {'unrelated': wrong_frame_local}
        def succeeds():
            unrelated(); return {'w1': 1, 'w2': 2, 'w3': 3}
        def fails(): raise ValueError('fixture failure')
        try:
            sys.setprofile(prior)
            value, retained = probe.retain_backward(succeeds)
            self.assertEqual(value, {'w1': 1, 'w2': 2, 'w3': 3})
            self.assertNotIn('wrong_frame_local', retained)
            with self.assertRaisesRegex(ValueError, 'fixture failure'): probe.retain_backward(fails)
            self.assertIs(sys.getprofile(), prior)
        finally: sys.setprofile(previous)

    def test_allocation_fixture_holds_real_arrays_and_never_uses_optimizer(self):
        supplied = inputs(); selected, validation = probe.select_groups(trainer, supplied)
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, trainer.FIXED_SEEDS[0])
        before = {key: value.copy() for key, value in parameters.items()}
        quantized = trainer.quantize_fixed(parameters, architecture, {key: .02 for key in parameters})
        with mock.patch.object(trainer, 'AdamW', side_effect=AssertionError('optimizer called')):
            retained, report = probe.allocation_fixture(trainer, supplied, parameters, quantized, selected, validation)
        summary = probe.selection_summary(selected, validation)
        probe.validate_allocations(report, summary)
        for name in parameters: self.assertTrue(np.array_equal(parameters[name], before[name]))
        self.assertEqual(len(retained['training_active']), 16)
        self.assertEqual(len(retained['scalar_active']), 256)
        self.assertEqual(len(retained['previous_validation_active']), 4)
        self.assertIsNot(retained['previous_validation_active'], retained['next_validation_active'])
        self.assertEqual(retained['backward_locals']['first_pre_gradient'].shape, (16, 12))
        self.assertEqual(np.count_nonzero(retained['artificial_derivative']), 16)
        self.assertTrue(report['zero_gradient_array_fully_written'])
        for alteration in ('rows', 'array'):
            changed = copy.deepcopy(report)
            if alteration == 'rows': changed['all_training_successors_forwarded'] = 15
            else: changed['backward_named_arrays']['first_pre_gradient']['bytes'] -= 4
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                probe.validate_allocations(changed, summary)

    def test_largest_validation_envelope_reads_only_the_bound_group_index(self):
        documents = {'audit': {'ranking_store': {'path': 'index'}}, 'index': {'groups': [
            {'split': 'train', 'begin': 0, 'end': 100, 'group': {'group_id': 'train'}},
            {'split': 'validation', 'begin': 100, 'end': 110, 'group': {'group_id': 'z'}},
            {'split': 'validation', 'begin': 110, 'end': 120, 'group': {'group_id': 'a'}}]}}
        engine = SimpleNamespace(campaign=SimpleNamespace(read=lambda path: documents[str(path)]))
        with mock.patch.object(probe, 'verified', side_effect=lambda binding: binding['path']):
            result = probe.validation_envelope(engine, {'input_audit': {'path': 'audit'}})
        self.assertEqual(result['validation_group_id'], 'a')
        self.assertEqual(result['V'], 10)

    def test_extent_accounting_cannot_claim_unobserved_pages_resident(self):
        arrays = {'indices': {'bytes': 100}, 'transcripts': {'bytes': 20}, 'successors': {'bytes': 40}}
        plan = {'training_envelope': {'mapped_array_artifacts': arrays}}
        ranges = {'mapped_artifacts': arrays, 'total_extent_bytes': 160,
                  'selected_index_transcript_range_bytes': {'indices': 30, 'transcripts': 10},
                  'outside_selected_index_transcript_ranges_bytes': 80,
                  'metadata_scanned_during_reconstruction': True, 'range_coverage_is_not_pinned_residency': True,
                  'same_selected_ranges_in_all_four_workers': True}
        probe.validate_ranges(ranges, plan)
        for key, value in (('outside_selected_index_transcript_ranges_bytes', 0), ('range_coverage_is_not_pinned_residency', False)):
            changed = copy.deepcopy(ranges); changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mapped-extent'):
                probe.validate_ranges(changed, plan)

    def test_result_requires_complete_sampling_of_all_five_distinct_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve(); path = output/'plan.json'; path.write_text('{}')
            claim = {'policy': probe.POLICY, 'plan': probe.record(path), 'retry_allowed': False, 'pid': 100,
                     'global_lock': {'device': 1, 'inode': 2}}
            (output/'claim.json').write_bytes(probe.raw(claim))
            selection = {'K': 1, 'N': 10, 'V': 2, 'validation_group_id': 'v', 'training_groups': [{'group_id': 't', 'successors': 10}]}
            workspace = {'selection': selection, 'mapped_ranges': {},
                         'allocations': {'parameters_unchanged': True, 'profile_hook_restored': True},
                         'usage_before_workspace': {'ru_maxrss': 10, 'ru_maxrss_units': 'bytes' if sys.platform=='darwin' else 'KiB',
                            'minor_page_faults': 1, 'major_page_faults': 0, 'ru_maxrss_is_process_lifetime_peak': True},
                         'usage_after_workspace': {'ru_maxrss': 20, 'ru_maxrss_units': 'bytes' if sys.platform=='darwin' else 'KiB',
                            'minor_page_faults': 2, 'major_page_faults': 0, 'ru_maxrss_is_process_lifetime_peak': True}}
            child_rows = [{'ordinal': i, 'pid': 101+i, 'identity': {}, 'lock': claim['global_lock'], 'runtime': {},
                           'native_thread_execution': {}, 'workspace': copy.deepcopy(workspace)} for i in range(4)]
            result = {'schema': probe.SCHEMA+'.result', 'plan': probe.record(path), 'claim': probe.record(output/'claim.json'),
                      'policy': probe.POLICY, 'optimizer_steps': 0, 'production_peak_headroom_proven': False,
                      'automatic_production_activation': False, 'training_artifacts_written': False, 'quality_test': False,
                      'coordinator': {'pid': 100, 'identity': {}, 'native_thread_execution': {}},
                      'cohort': {'children': child_rows, 'all_four_ready_and_held': True, 'child_exitcodes': [0]*4, 'minimum_hold_seconds': 2},
                      'selection': selection, 'mapped_ranges': {},
                      'memory': {'errors': [], 'samples': 2, 'per_pid_peak_rss': {str(pid): 1 for pid in range(100,105)}}}
            docs = {'claim.json': claim, 'result.json': result}
            engine = SimpleNamespace(campaign=SimpleNamespace(read=lambda p: docs[Path(p).name]),
                capacity=SimpleNamespace(runtime=lambda: {}),
                trainer=SimpleNamespace(validate_native_thread_execution=lambda value: value),
                sampling=SimpleNamespace(validate_memory=lambda value: value))
            plan = {'output': str(output), 'hold_seconds': 2, 'expected_input_identity': {}}
            with mock.patch.object(probe, 'validate_plan', return_value=(engine,plan)), \
                    mock.patch.object(probe, 'validate_selection'), mock.patch.object(probe, 'validate_ranges'), \
                    mock.patch.object(probe, 'validate_allocations'):
                probe.validate(path)
                for mode in ('missing', 'error', 'parent_pid'):
                    changed=copy.deepcopy(result)
                    if mode=='missing': changed['memory']['per_pid_peak_rss'].pop('104')
                    elif mode=='error': changed['memory']['errors']=['sampling failed']
                    else: changed['cohort']['children'][0]['pid']=100
                    docs['result.json']=changed
                    with self.subTest(mode=mode), self.assertRaises(ValueError): probe.validate(path)


if __name__ == '__main__': unittest.main()
