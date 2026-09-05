import fcntl
import copy
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

from tools import compact_value_bfm_training_capacity_v2 as capacity
os.environ.update(capacity.process.ENVIRONMENT)
os.environ[capacity.process.MARKER] = '1'
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_training_resources_v2 as resources


class Sampler:
    def __init__(self): self.samples = 0
    def sample(self): self.samples += 1


def synthetic_child(plan_path, ordinal, connection, ticket, expected_lock):
    fd = capacity.retain_shared_lock(ticket, expected_lock)
    capacity.parent_death_guard()
    try:
        plan = json.loads(Path(plan_path).read_text())
        if plan.get('fail') and ordinal == 0:
            connection.send({'kind': 'error', 'ordinal': ordinal, 'pid': os.getpid(), 'error': 'synthetic failure'})
            return
        _campaign, trainer, _resources, _sampling = capacity.modules()
        with trainer.native_thread_execution_scope() as native:
            retained = bytearray(4096)
            connection.send({'kind': 'ready', 'ordinal': ordinal, 'pid': os.getpid(),
                'identity': plan['identity'], 'runtime': capacity.runtime(), 'lock': capacity.lock_identity(fd),
                'native_thread_execution': native})
            try:
                capacity.retain_until_release(connection, 10)
                assert len(retained) == 4096
            except Exception:
                pass
    finally:
        os.close(fd)


def lock_child(ticket, expected, messages, release, guard):
    fd = capacity.retain_shared_lock(ticket, expected)
    if guard: capacity.parent_death_guard()
    retained = bytearray(4096)
    messages.put(os.getpid())
    release.poll(10)
    assert len(retained) == 4096
    os.close(fd)


def death_owner(path, messages, release):
    context = multiprocessing.get_context('spawn')
    with open(path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = context.Process(target=lock_child, args=(capacity.SpawnLockTicket(lock.fileno()), capacity.lock_identity(lock.fileno()), messages, release, True))
        worker.start()
        time.sleep(20)
        worker.join()


def fixture(root):
    root = root.resolve(); phase = 'attempt-002-pilot'; context = root / 'phases' / phase
    bundle = root / 'bundle.json'; campaign.once(bundle, b'fixture bundle')
    campaign.seal(root / 'campaign.json', {'policy': campaign.POLICY})
    campaign.seal(context / 'campaign.json', {'attempt': 2, 'phase': 'pilot', 'policy': campaign.POLICY,
        'parent_campaign': campaign.record(root / 'campaign.json'), 'bundle': campaign.record(bundle)})
    for name in ('labels.json', 'positions.json'):
        campaign.seal(context / phase / name, {'fixture': name})
    campaign.seal(context / phase / 'training-input-audit.json', {'schema': campaign.ID + '.training-input-audit.v2',
        'bundle': campaign.record(bundle), 'protected_tests_opened': False,
        'labels': campaign.record(context / phase / 'labels.json'), 'position_closure': campaign.record(context / phase / 'positions.json')})
    resources.authorize(root, authorized_at_utc='2026-09-06T00:00:00Z', user_request=resources.USER_REQUEST)
    return root, context, phase, root / 'diagnostics/training-capacity/test'


class CapacityTests(unittest.TestCase):
    def test_absolute_cli_help_works_from_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            finished = subprocess.run([sys.executable, str(Path(capacity.__file__).resolve()), '--help'], cwd=tmp,
                                      capture_output=True, text=True, timeout=10)
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertIn('prepare', finished.stdout)

    def test_prepare_freezes_actual_audit_without_reconstructing_or_starting_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, context, phase, output = fixture(Path(tmp))
            before = {p: p.read_bytes() for p in context.rglob('*') if p.is_file()}
            with mock.patch.object(capacity.process, 'reconstruct_inputs', side_effect=AssertionError('heavy reconstruction')):
                record = capacity.prepare(root, context, phase, output)
                plan = capacity.validate_plan(record['path'])
            self.assertEqual(plan['policy']['workers'], 4)
            self.assertFalse(plan['preparation_reconstructed_inputs'])
            self.assertFalse(plan['policy']['mapped_pages_pinned_or_fully_prefaulted'])
            self.assertEqual(before, {p: p.read_bytes() for p in context.rglob('*') if p.is_file()})

    def test_source_drift_and_other_output_roots_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, context, phase, output = fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'fresh child'):
                capacity.prepare(root, context, phase, root / 'phases/output')
            plan = capacity.prepare(root, context, phase, output)
            with mock.patch.object(capacity.process, 'source_closure', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'source/runtime'):
                    capacity.validate_plan(plan['path'])

    def test_busy_global_lease_and_spent_claim_reject_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, context, phase, output = fixture(Path(tmp))
            plan = capacity.prepare(root, context, phase, output)
            with (root / '.heavy-stage.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(capacity.process, 'reconstruct_inputs', side_effect=AssertionError('must not load')):
                    with self.assertRaises(BlockingIOError): capacity.run(plan['path'])
            self.assertFalse((output / 'claim.json').exists())
            campaign.seal(output / 'claim.json', {'spent': True})
            with mock.patch.object(capacity.process, 'reconstruct_inputs', side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError, 'spent'):
                    capacity.run(plan['path'])

    def test_four_distinct_children_hold_together_and_are_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'synthetic.json'
            identity = {'fixture': 'same inputs'}; path.write_text(json.dumps({'identity': identity}))
            before = {item.pid for item in multiprocessing.active_children()}
            with (root / 'lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sampler = Sampler()
                with mock.patch.object(capacity, 'observe_system', return_value={'synthetic': True}):
                    result = capacity.hold_four(path, {'hold_seconds': .05, 'reconstruction_timeout_seconds': 10},
                        identity, sampler, lock.fileno(), worker_target=synthetic_child)
            self.assertEqual(len({row['pid'] for row in result['children']}), 4)
            self.assertEqual(result['child_exitcodes'], [0, 0, 0, 0])
            self.assertEqual(sampler.samples, 2)
            self.assertEqual(before, {item.pid for item in multiprocessing.active_children()})

    def test_child_failure_aborts_and_joins_the_entire_owned_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'synthetic.json'
            path.write_text(json.dumps({'identity': {}, 'fail': True}))
            before = {item.pid for item in multiprocessing.active_children()}
            with (root / 'lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'child'):
                    capacity.hold_four(path, {'hold_seconds': .01, 'reconstruction_timeout_seconds': 5},
                        {}, Sampler(), lock.fileno(), worker_target=synthetic_child)
            self.assertEqual(before, {item.pid for item in multiprocessing.active_children()})

    def test_child_duplicate_keeps_the_global_lease_after_parent_fd_closes(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'; messages = context.Queue(); release, release_send = context.Pipe(duplex=False)
            lock = path.open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = context.Process(target=lock_child, args=(capacity.SpawnLockTicket(lock.fileno()), capacity.lock_identity(lock.fileno()), messages, release, False))
            worker.start()
            try:
                self.assertEqual(messages.get(timeout=5), worker.pid)
                lock.close()
                with path.open('a') as contender:
                    with self.assertRaises(BlockingIOError): fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                release_send.send('release'); worker.join(timeout=5)
                self.assertEqual(worker.exitcode, 0)
                with path.open('a') as contender: fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                release_send.close(); release.close()
                if worker.is_alive(): worker.kill()
                worker.join(); lock.close(); messages.close(); messages.join_thread()

    def test_parent_death_guard_exits_orphan_and_releases_its_retained_lease(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'; messages = context.Queue(); release, release_send = context.Pipe(duplex=False)
            owner = context.Process(target=death_owner, args=(str(path), messages, release))
            owner.start(); child_pid = None
            try:
                child_pid = messages.get(timeout=5)
                owner.terminate(); owner.join(timeout=5)
                deadline = time.monotonic() + 5
                released = False
                while time.monotonic() < deadline:
                    with path.open('a') as contender:
                        try:
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB); released = True; break
                        except BlockingIOError:
                            time.sleep(.05)
                self.assertTrue(released, 'orphan retained loaded inputs/lease after coordinator death')
                state = subprocess.run(['ps', '-p', str(child_pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'), state)
            finally:
                release_send.close(); release.close()
                if owner.is_alive(): owner.kill()
                owner.join()
                if child_pid is not None:
                    state = subprocess.run(['ps', '-p', str(child_pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
                    if state and not state.startswith('Z'):
                        os.kill(child_pid, 9)  # Exact PID emitted by this test's owned child.
                messages.close(); messages.join_thread()

    def test_failed_reconstruction_traceback_does_not_keep_private_arrays_alive(self):
        class Payload: pass
        seen = []
        def failed():
            payload = Payload(); seen.append(weakref.ref(payload)); raise ValueError('failed load')
        try:
            failed()
        except ValueError as error:
            self.assertIsNotNone(seen[0]())
            capacity.clear_failed_frames(error)
            self.assertIsNone(seen[0]())

    def test_workspace_report_is_only_metadata_and_does_not_claim_batch_residency(self):
        _campaign, trainer, _resources, _sampling = capacity.modules()
        groups = [SimpleNamespace(successors=range(count)) for count in (10, 50, 20)]
        inputs = SimpleNamespace(new=range(128), successor_rankings=SimpleNamespace(train=groups, validation=[]))
        with mock.patch.object(trainer, '_ranking_group_profile', return_value='standard-v1'):
            result = capacity.workspace_dimensions(inputs)
        self.assertEqual(result['ranking_groups_per_batch_upper_bound'], 2)
        self.assertEqual(result['successors_per_batch_conservative_upper_bound'], 70)
        self.assertEqual(result['forward_cache_bytes_without_predictions_per_worker'], 70 * 164)
        self.assertTrue(result['actual_batch_not_observed'])
        self.assertFalse(result['backward_temporary_view_and_allocator_bytes_included'])

    def test_identity_rejects_changed_body_audit_and_unrelated_dataset_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, context, phase, _output = fixture(Path(tmp))
            audit_path = context / phase / 'training-input-audit.json'
            audit = campaign.read(audit_path); audit.pop('body_sha256')
            opaque = root / 'train-shard'; opaque.write_bytes(b'only metadata is needed')
            audit.update({'shards': {'train': {'manifest': campaign.record(opaque), 'npz': campaign.record(opaque)}},
                          'ranking_store': campaign.record(opaque), 'anchor_duplicates_removed': 0})
            audit_path.unlink(); audit = campaign.seal(audit_path, audit)
            plan = {'input_audit': campaign.record(audit_path)}
            routes = {'anchor': ['anchor/train'], 'canonical_validation': ['canonical/validation'],
                      'common_adjudicator': ['common/validation'], 'new': [str(opaque)]}
            body = {'input_audit': audit, 'source_routes': routes,
                    'split_isolation': {'closure_audit': audit['body_sha256']},
                    'paired_row_validation': {'external_source_bound': True},
                    'datasets': {'new': {'source_manifest_sha256': audit['shards']['train']['manifest']['sha256'],
                                        'source_npz_sha256': audit['shards']['train']['npz']['sha256']}},
                    'ranking': {'artifact_sha256': audit['ranking_store']['sha256']},
                    'anchor_filter': {'removed_rows': 0}}
            identity = {**body, 'body_sha256': campaign.hashlib.sha256(campaign.raw(body)).hexdigest()}
            bundle = SimpleNamespace(canonical_routes=lambda split: routes['anchor' if split == 'train' else 'canonical_validation'],
                                     common_adjudicator_route=lambda: 'common/validation')
            _c, trainer, _r, _s = capacity.modules()
            with mock.patch.object(trainer.FrozenBundle, 'load', return_value=bundle):
                capacity.validate_identity(identity, plan)
                for kind in ('body', 'audit', 'routes'):
                    changed = copy.deepcopy(identity)
                    if kind == 'body': changed['body_sha256'] = '0' * 64
                    elif kind == 'audit': changed['input_audit']['protected_tests_opened'] = True
                    else: changed['source_routes']['new'] = ['/unrelated/training-shard']
                    if kind != 'body':
                        changed['body_sha256'] = campaign.hashlib.sha256(campaign.raw(
                            {key: value for key, value in changed.items() if key != 'body_sha256'})).hexdigest()
                    with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'actual audited inputs'):
                        capacity.validate_identity(changed, plan)

    def test_result_rejects_forged_memory_completeness_and_parent_as_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve(); plan_path = output / 'plan.json'
            campaign.seal(plan_path, {'fixture': True})
            plan = {'output': str(output), 'hold_seconds': 2}
            claim = campaign.seal(output / 'claim.json', {'schema': capacity.SCHEMA + '.claim',
                'plan': campaign.record(plan_path), 'retry_allowed': False, 'policy': capacity.POLICY,
                'pid': 100, 'global_lock': {'device': 1, 'inode': 2}})
            identity = {'fixture': True}
            children = [{'ordinal': i, 'pid': 101 + i, 'identity': identity, 'runtime': capacity.runtime(),
                         'lock': claim['global_lock'], 'native_thread_execution': {}} for i in range(4)]
            body = {'schema': capacity.SCHEMA + '.result', 'plan': campaign.record(plan_path),
                'claim': campaign.record(output / 'claim.json'), 'policy': capacity.POLICY, 'runtime': capacity.runtime(),
                'optimizer_steps': 0, 'training_artifacts_written': False, 'production_peak_headroom_proven': False,
                'automatic_production_activation': False, 'qualification_eligible': False,
                'coordinator': {'pid': 100, 'identity': identity, 'native_thread_execution': {}},
                'cohort': {'children': children, 'all_four_ready_and_held': True, 'child_exitcodes': [0]*4, 'minimum_hold_seconds': 2},
                'memory': {'units': 'KiB', 'interval_seconds': 2, 'includes_coordinator_and_descendants': True,
                    'samples': 2, 'errors': [], 'process_tree_peak_rss': 1000,
                    'per_pid_peak_rss': {str(pid): 100 for pid in range(100, 105)}},
                'memory_measurements_complete': True}
            _c, trainer, _r, _s = capacity.modules()
            with mock.patch.object(capacity, 'validate_plan', return_value=plan), \
                    mock.patch.object(capacity, 'validate_identity'), \
                    mock.patch.object(trainer, 'validate_native_thread_execution'):
                campaign.seal(output / 'result.json', body); capacity.validate_result(plan_path)
                for kind in ('memory', 'parent'):
                    changed = copy.deepcopy(body)
                    if kind == 'memory': changed['memory']['samples'] = 1
                    else: changed['cohort']['children'][0]['pid'] = 100
                    (output / 'result.json').unlink(); campaign.seal(output / 'result.json', changed)
                    with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'memory completeness|cohort'):
                        capacity.validate_result(plan_path)


if __name__ == '__main__': unittest.main()
