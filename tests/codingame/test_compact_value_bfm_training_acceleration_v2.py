"""Synthetic process/receipt tests; no training, process stops or campaign data."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_training_acceleration_v2 as acceleration
from tools import compact_value_bfm_seed_process_check_v2 as check

campaign = acceleration.campaign


def file(path, content=b'fixture'):
    campaign.once(path, content)
    return campaign.record(path)


def sealed(path, body):
    campaign.seal(path, body)
    return campaign.record(path)


def process(pid, command, ppid=1, started='2026-09-05T01:00:00Z'):
    return {'pid': pid, 'ppid': ppid, 'command': command, 'started_at_utc': started}


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(); self.phase = 'attempt-001-pilot'
        self.context = self.root / 'phases' / self.phase
        self.output = self.root / 'diagnostics/seed-process-equivalence/concurrent-test'
        parent = sealed(self.root / 'campaign.json', {'policy': campaign.POLICY})
        sealed(self.context / 'campaign.json', {'parent_campaign': parent, 'policy': campaign.POLICY})
        acceleration.resources.authorize(self.root, authorized_at_utc='2026-09-05T00:00:00Z',
                                        user_request=acceleration.resources.USER_REQUEST)
        self.authorization = campaign.record(acceleration.resources.authorization_path(self.root))
        snapshot = self.root / 'source-snapshots' / ('a' * 40)
        tools = snapshot / 'repository/tools'
        names = ['compact_value_bfm_pilot_selection_v2.py', 'compact_value_bfm_pilot_gate_v2.py',
                 'compact_value_bfm_full_v2.py', 'compact_value_bfm_campaign_v2.py', 'compact_value_bfm_train.py']
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w') as tar:
            for name in names:
                data = ('synthetic ' + name).encode(); file(tools / name, data)
                member = tarfile.TarInfo('tools/' + name); member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        archive_record = file(snapshot / 'source.tar', archive.getvalue())
        sealed(snapshot / 'snapshot.json', {'schema': campaign.ID + '.source-snapshot.v2',
            'commit': 'a' * 40, 'repository': str(snapshot / 'repository'), 'archive': archive_record})
        common = f'--root {self.root} --context {self.context} --phase {self.phase}'
        self.table = {
            101: process(101, f'{sys.executable} {tools / names[0]} {common} --wait-for-labels train'),
            102: process(102, f'{sys.executable} {tools / names[1]} {common} --wait-for-selection 101'),
            103: process(103, f'{sys.executable} {tools / names[2]} --root {self.root} --pilot-context {self.context} --pilot-phase {self.phase} --wait-for-pilot 102 run'),
            os.getpid(): process(os.getpid(), f'{sys.executable} {acceleration.__file__} run')}

    def prepare(self):
        def harness(root, output, **kwargs):
            self.assertEqual(kwargs['executor'], {'mode': 'spawn-v2', 'maximum_workers': 4})
            self.assertEqual(kwargs['training_resource_authorization'], self.authorization)
            return sealed(output / 'plan.json', {'root': str(root), 'output': str(output),
                'concurrent_equivalence_context': kwargs['concurrent_context'],
                'training_resource_authorization': self.authorization, 'global_heavy_stage_lease_required': False,
                'execution_timing': 'contended-equivalence-only', 'speedup_qualification_allowed': False})
        with patch.object(acceleration, 'process_table', return_value=self.table), \
                patch.object(check, 'prepare', side_effect=harness):
            result = acceleration.prepare(self.root, self.output, self.authorization, 101, 102, 103)
        self.record = result['context']; self.plan = Path(result['plan']['path'])
        self.document = campaign.read(campaign.verify(self.record))
        return result

    def retire(self):
        self.table.pop(102); self.table.pop(103)

    def test_prepare_binds_actual_processes_and_source_without_training_or_stops(self):
        with patch.object(acceleration.os, 'killpg') as kill, patch.object(check, 'run_stage') as train:
            result = self.prepare()
        self.assertFalse(result['real_training_started'])
        self.assertEqual(self.document['legacy_training']['process'], acceleration.identity(self.table[101]))
        self.assertEqual([row['process']['pid'] for row in self.document['retired_waiters']], [102, 103])
        self.assertEqual(self.document['policy']['temporary_total_seed_workers'], 6)
        self.assertFalse(self.document['policy']['global_heavy_stage_lease_owned'])
        train.assert_not_called(); kill.assert_not_called()
        self.assertEqual(acceleration.validate_context(self.record, root=self.root, plan_path=self.plan), self.document)

    def test_context_is_bound_to_canonical_plan_and_explicit_resource_authority(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'canonical output'):
            acceleration.validate_context(self.record, root=self.root, plan_path=self.root / 'elsewhere.json')
        with patch.object(acceleration.resources, 'validate_authorization', side_effect=ValueError('authorization changed')), \
                self.assertRaisesRegex(ValueError, 'authorization changed'):
            acceleration.validate_context(self.record, root=self.root)
        changed = dict(self.document); changed['sources'] = {}
        changed_record = sealed(self.output / 'other-context.json', {key: value for key, value in changed.items() if key != 'body_sha256'})
        with self.assertRaisesRegex(ValueError, 'canonical path'):
            acceleration.validate_context(changed_record, root=self.root)

    def test_waiter_or_gate_cannot_overlap_validation(self):
        self.prepare()
        with patch.object(acceleration, 'process_table', return_value=self.table), self.assertRaisesRegex(ValueError, 'not been retired'):
            acceleration.guard_runtime(self.document)
        self.retire()
        with patch.object(acceleration, 'process_table', return_value=self.table):
            self.assertEqual(acceleration.guard_runtime(self.document)['status'], 'original-training-running')
        file(self.context / self.phase / 'rank4-screen/claim.json')
        with self.assertRaisesRegex(ValueError, 'gate, pilot outcome or full phase'):
            acceleration.guard_runtime(self.document)

    def test_changed_or_prematurely_exited_legacy_pid_never_authorizes_overlap(self):
        self.prepare(); self.retire()
        self.table[101]['started_at_utc'] = '2026-09-05T02:00:00Z'
        with patch.object(acceleration, 'process_table', return_value=self.table), self.assertRaisesRegex(ValueError, 'reused or changed'):
            acceleration.guard_runtime(self.document)
        self.table.pop(101)
        with patch.object(acceleration, 'process_table', return_value=self.table), self.assertRaisesRegex(ValueError, 'exited without complete'):
            acceleration.guard_runtime(self.document)

    def test_competing_heavy_process_is_rejected_but_owned_descendants_are_permitted(self):
        self.prepare(); self.retire()
        self.table[222] = process(222, f'{sys.executable} {acceleration.__file__} _stage', os.getpid())
        self.table[223] = process(223, f'{sys.executable} -c multiprocessing.spawn', 222)
        with patch.object(acceleration, 'process_table', return_value=self.table):
            acceleration.guard_runtime(self.document, owned_stage_pid=222)
        self.table[333] = process(333, f'{sys.executable} /tmp/compact_value_bfm_protected_v2.py run --root {self.root}')
        with patch.object(acceleration, 'process_table', return_value=self.table), self.assertRaisesRegex(ValueError, 'qualification/live/training'):
            acceleration.guard_runtime(self.document, owned_stage_pid=222)

    def test_completed_legacy_training_may_finish_while_waiters_remain_held(self):
        self.prepare(); self.retire(); self.table.pop(101)
        directory = self.context / self.phase
        rows = []
        for weight in (0., .1, .25):
            job = directory / 'training' / f'lambda-{weight:.2f}'
            for seed in (20260907, 20260908, 20260909):
                runtime = file(job / f'runtime-{seed}.json'); checkpoint = file(job / f'float-{seed}.npz')
                rows.append({'weight': weight, 'seed': seed, 'source': file(job / f'seed-{seed}.cpp'),
                    'runtime': runtime, 'float_checkpoint': checkpoint, 'seed_receipt': {
                        'quantized_runtime': {'path': Path(runtime['path']).name, 'sha256': runtime['sha256']},
                        'float_checkpoint': {'path': Path(checkpoint['path']).name, 'sha256': checkpoint['sha256']}}})
        training = sealed(directory / 'training.json', {'smoke': False, 'mandatory_training_verified': True,
            'results': rows, 'producer': self.document['legacy_training']['source_files'][1],
            'input_audit': sealed(directory / 'training-input-audit.json', {'fixture': 'audit'})})
        sealed(directory / 'model-selection.json', {'training': training, 'pilot_admitted': False,
            'policy': sealed(directory / 'selection-policy.json', {'fixture': 'policy'})})
        with patch.object(acceleration, 'process_table', return_value=self.table):
            result = acceleration.guard_runtime(self.document)
        self.assertEqual(result['status'], 'original-training-completed-with-waiters-held')

    def test_separate_lock_can_coexist_with_heavy_lease_and_inherited_fd_retains_it(self):
        self.prepare()
        with campaign.lease(self.root):
            with acceleration.validation_lock(self.document) as fd:
                inherited = os.dup(fd)
                with self.assertRaises(BlockingIOError):
                    with acceleration.validation_lock(self.document): pass
            try:
                with self.assertRaises(BlockingIOError):
                    with acceleration.validation_lock(self.document): pass
            finally:
                os.close(inherited)
            with acceleration.validation_lock(self.document): pass

    def test_parent_death_watch_stops_only_the_dedicated_owned_group(self):
        read_fd, write_fd = os.pipe(); os.close(write_fd)
        try:
            with patch.object(acceleration.os, 'getpgrp', return_value=222), \
                    patch.object(acceleration.os, 'getpid', return_value=222), patch.object(acceleration.os, 'killpg') as kill:
                acceleration._parent_watch(read_fd)
            kill.assert_called_once_with(222, acceleration.signal.SIGKILL)
        finally:
            os.close(read_fd)

    def test_controller_failure_stops_owned_stage_and_preserves_process_claim(self):
        self.prepare(); self.retire()
        self.table[222] = process(222, f'{sys.executable} {acceleration.__file__} _stage', os.getpid())
        class Child:
            pid = 222
            returncode = None
            def poll(self): return self.returncode
            def wait(self): self.returncode = -9; return self.returncode
        child = Child()
        with acceleration.validation_lock(self.document) as fd, \
                patch.object(acceleration, 'process_table', return_value=self.table), \
                patch.object(acceleration, 'guard_runtime', side_effect=[{'status': 'original-training-running'}, ValueError('forbidden gate appeared')]), \
                patch.object(acceleration.subprocess, 'Popen', return_value=child) as launch, \
                patch.object(acceleration.os, 'getpgid', return_value=222), \
                patch.object(acceleration.os, 'killpg') as kill, \
                self.assertRaisesRegex(ValueError, 'forbidden gate'):
            acceleration._run_stage(self.record, self.plan, self.document, 'threads', fd)
        kill.assert_called_once_with(222, acceleration.signal.SIGKILL)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertIn(fd, launch.call_args.kwargs['pass_fds'])
        owned = campaign.read(self.output / 'controller/threads/process.json')
        self.assertEqual(owned['process']['pid'], 222)
        self.assertEqual(owned['process_group_id'], 222)
        completed = campaign.read(self.output / 'controller/threads/execution.json')
        self.assertEqual(completed['failure']['message'], 'forbidden gate appeared')
        self.assertFalse(completed['existing_training_stopped'])
        self.assertFalse(completed['global_heavy_stage_lease_owned'])

    def test_busy_or_spent_controller_never_starts_training(self):
        self.prepare(); self.retire()
        with patch.object(check, 'validate_plan'), patch.object(acceleration, 'process_table', return_value=self.table), \
                patch.object(acceleration, '_run_stage') as train, acceleration.validation_lock(self.document), \
                self.assertRaises(BlockingIOError):
            acceleration.run(self.record, self.plan)
        train.assert_not_called()
        sealed(self.output / 'controller-claim.json', {'spent': True})
        with patch.object(check, 'validate_plan'), patch.object(acceleration, 'process_table', return_value=self.table), \
                patch.object(acceleration, '_run_stage') as train, self.assertRaisesRegex(ValueError, 'already claimed'):
            acceleration.run(self.record, self.plan)
        train.assert_not_called()

    def test_partial_first_stage_prevents_second_stage_and_persists_failure(self):
        self.prepare(); self.retire()
        with patch.object(check, 'validate_plan'), patch.object(acceleration, 'process_table', return_value=self.table), \
                patch.object(acceleration, '_run_stage', side_effect=ValueError('incomplete thread roster')) as train, \
                self.assertRaisesRegex(ValueError, 'incomplete thread'):
            acceleration.run(self.record, self.plan)
        self.assertEqual(train.call_count, 1)
        result = campaign.read(self.output / 'controller-result.json')
        self.assertFalse(result['completed']); self.assertFalse(result['speedup_qualification_allowed'])
        self.assertFalse(result['existing_training_stopped'])


if __name__ == '__main__':
    unittest.main()
