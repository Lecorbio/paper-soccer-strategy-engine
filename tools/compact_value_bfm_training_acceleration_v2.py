#!/usr/bin/env python3
"""Supervise one explicitly authorized concurrent smoke equivalence experiment.

The existing pilot remains untouched. Its two seed slots plus four validation
slots are the temporary six-worker ceiling. The validation lock is distinct from
the campaign heavy-stage lock: all measurements are contended and cannot prove a
speedup. A controller failure stops only its own validation process group and
preserves its non-retryable claims. Preparation launches no training or stops.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time

THREADS = {key: '1' for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
MARKER = 'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'
if __name__ == '__main__':
    os.environ.update(THREADS); os.environ[MARKER] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_training_resources_v2 as resources

SCHEMA = campaign.ID + '.concurrent-training-equivalence-context.v2'
POLICY = {'existing_seed_workers': 2, 'validation_seed_workers': 4, 'temporary_total_seed_workers': 6,
    'native_threads_per_worker': 1, 'current_training_restart_allowed': False,
    'validation_stage_order': ['threads', 'spawn'], 'global_heavy_stage_lease_owned': False,
    'actual_clock_stages_allowed': False, 'qualification_allowed': False,
    'timing': 'contended-equivalence-only', 'speedup_qualification_allowed': False,
    'automatic_production_activation': False, 'retry_interrupted_stage': False,
    'guard_poll_seconds': 2, 'owned_stage_watchdog_seconds': 86400}
SCRIPT = 'compact_value_bfm_training_acceleration_v2.py'


def _bound(binding, expected):
    expected = Path(expected).absolute()
    if expected.resolve() != expected or Path(binding.get('path', '')).absolute() != expected:
        raise ValueError('concurrent validation artifact left its canonical path')
    return campaign.verify(binding)


def output_path(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent = root / 'diagnostics/seed-process-equivalence'
    if output == parent or not output.is_relative_to(parent):
        raise ValueError('concurrent output must be a fresh diagnostic child')
    return output


def process_table():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,lstart=,args='], capture_output=True,
        text=True, check=True, env={**os.environ, 'LC_ALL': 'C', 'TZ': 'UTC'})
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 7)
        if len(parts) != 8:
            raise ValueError('process inventory does not expose stable PID/start/command identities')
        pid, parent = int(parts[0]), int(parts[1])
        started = dt.datetime.strptime(' '.join(parts[2:7]), '%a %b %d %H:%M:%S %Y').replace(tzinfo=dt.timezone.utc)
        rows[pid] = {'pid': pid, 'ppid': parent, 'started_at_utc': started.isoformat().replace('+00:00', 'Z'),
                     'command': parts[7]}
    return rows


def identity(row):
    return {key: row[key] for key in ('pid', 'started_at_utc', 'command')}


def _live_pid(table, pid):
    if type(pid) is not int or pid <= 0 or pid not in table:
        raise ValueError('the expected training/waiter process is not running')
    return table[pid]


def _bind_process(root, row, script_name, arguments):
    argv = shlex.split(row['command'])
    if len(argv) < 2 or argv[2:] != list(map(str, arguments)) or Path(argv[1]).name != script_name:
        raise ValueError('training/waiter command differs from the explicitly scoped process')
    source = Path(argv[1]).absolute()
    if source.resolve() != source:
        raise ValueError('legacy process source is redirected')
    relative = source.relative_to(root / 'source-snapshots')
    if (len(relative.parts) != 4 or re.fullmatch('[0-9a-f]{40}', relative.parts[0]) is None
            or relative.parts[1:3] != ('repository', 'tools')):
        raise ValueError('legacy process must run an immutable campaign source snapshot')
    snapshot_directory = root / 'source-snapshots' / relative.parts[0]
    snapshot_path = snapshot_directory / 'snapshot.json'
    snapshot = campaign.read(snapshot_path)
    if (snapshot.get('schema') != campaign.ID + '.source-snapshot.v2'
            or snapshot.get('commit') != relative.parts[0]
            or snapshot.get('repository') != str(snapshot_directory / 'repository')):
        raise ValueError('legacy source snapshot identity changed')
    archive = _bound(snapshot['archive'], snapshot_directory / 'source.tar')
    paths = [source, source.parent / 'compact_value_bfm_campaign_v2.py', source.parent / 'compact_value_bfm_train.py']
    with tarfile.open(archive, 'r:') as tar:
        for path in paths:
            member = tar.getmember('tools/' + path.name)
            if not member.isfile() or tar.extractfile(member).read() != path.read_bytes():
                raise ValueError('running process source differs from its frozen archive')
    return {'process': identity(row), 'executable': campaign.record(Path(argv[0])),
            'script': campaign.record(source), 'snapshot': campaign.record(snapshot_path),
            'source_files': [campaign.record(path) for path in paths]}


def _verify_process_binding(binding, root, script_name, arguments):
    process = binding['process']
    if (set(process) != {'pid', 'started_at_utc', 'command'} or type(process['pid']) is not int
            or process['pid'] <= 0):
        raise ValueError('bound process identity is malformed')
    dt.datetime.fromisoformat(process['started_at_utc'].replace('Z', '+00:00'))
    argv = shlex.split(process['command'])
    if len(argv) < 2 or argv[2:] != list(map(str, arguments)) or Path(argv[1]).name != script_name:
        raise ValueError('bound process command changed its scoped role')
    snapshot_path = campaign.verify(binding['snapshot']); snapshot = campaign.read(snapshot_path)
    repository = root / 'source-snapshots' / snapshot['commit'] / 'repository'
    if (snapshot.get('schema') != campaign.ID + '.source-snapshot.v2'
            or re.fullmatch('[0-9a-f]{40}', snapshot['commit']) is None
            or snapshot_path != repository.parent / 'snapshot.json'
            or snapshot['repository'] != str(repository)):
        raise ValueError('bound process snapshot schema or canonical path changed')
    source = _bound(binding['script'], repository / 'tools' / script_name)
    if Path(argv[1]).absolute() != source or campaign.verify(binding['executable']) != Path(argv[0]).resolve():
        raise ValueError('bound process executable or script changed')
    expected_files = [source, source.parent / 'compact_value_bfm_campaign_v2.py',
                      source.parent / 'compact_value_bfm_train.py']
    if len(binding['source_files']) != 3:
        raise ValueError('bound process source closure changed')
    for record, path in zip(binding['source_files'], expected_files, strict=True):
        _bound(record, path)


def _sources():
    directory = Path(__file__).resolve().parent
    return {name: campaign.record(directory / name) for name in (
        SCRIPT, 'compact_value_bfm_training_resources_v2.py', 'compact_value_bfm_seed_process_check_v2.py',
        'compact_value_bfm_seed_process_v2.py', 'compact_value_bfm_campaign_v2.py')}


def _forbidden_artifacts(context):
    phase_directory = Path(context['legacy_context']) / context['legacy_phase']
    paths = [phase_directory / 'rank4-screen', phase_directory / 'pilot-outcome.json',
             Path(context['root']) / 'phases' / context['legacy_phase'].replace('-pilot', '-full')]
    if any(path.exists() for path in paths):
        raise ValueError('actual-clock gate, pilot outcome or full phase already materialized')
    live = Path(context['root']) / 'live'
    if live.exists() and any(live.iterdir()):
        raise ValueError('live work cannot overlap concurrent training validation')


def prepare(root, output, authorization, legacy_pid, pilot_waiter_pid, full_waiter_pid):
    from tools import compact_value_bfm_seed_process_check_v2 as check
    from tools import compact_value_bfm_seed_process_v2 as process
    root, output = Path(root).resolve(), output_path(root, output)
    resources.validate_authorization(authorization, root)
    if output.exists() and any(output.iterdir()):
        raise ValueError('prepare requires a fresh concurrent diagnostic output')
    table = process_table()
    if len({legacy_pid, pilot_waiter_pid, full_waiter_pid}) != 3:
        raise ValueError('training and both waiter PIDs must be distinct')
    legacy = _live_pid(table, legacy_pid)
    argv = shlex.split(legacy['command'])
    if '--phase' not in argv:
        raise ValueError('legacy training command has no phase')
    phase = argv[argv.index('--phase') + 1]
    if re.fullmatch('attempt-[0-9]{3}-pilot', phase) is None:
        raise ValueError('concurrent validation only accompanies an existing pilot')
    legacy_context = root / 'phases' / phase
    common = ['--root', root, '--context', legacy_context, '--phase', phase]
    training = _bind_process(root, legacy, 'compact_value_bfm_pilot_selection_v2.py',
                             [*common, '--wait-for-labels', 'train'])
    waiters = [
        _bind_process(root, _live_pid(table, pilot_waiter_pid), 'compact_value_bfm_pilot_gate_v2.py',
                      [*common, '--wait-for-selection', legacy_pid]),
        _bind_process(root, _live_pid(table, full_waiter_pid), 'compact_value_bfm_full_v2.py',
            ['--root', root, '--pilot-context', legacy_context, '--pilot-phase', phase, '--wait-for-pilot', pilot_waiter_pid, 'run'])]
    contract = campaign.read(legacy_context / 'campaign.json')
    if (contract['parent_campaign'] != campaign.record(root / 'campaign.json')
            or contract['policy'] != campaign.POLICY or contract.get('training_executor') not in
                (None, {'mode': 'spawn-v2', 'maximum_workers': 2})):
        raise ValueError('existing pilot is not the unchanged frozen two-worker phase')
    if (legacy_context / phase / 'model-selection.json').exists():
        raise ValueError('waiters may only be retired while the pilot selection is unspent')
    body = {'schema': SCHEMA, 'root': str(root), 'output': str(output),
        'training_resource_authorization': authorization, 'legacy_training': training,
        'retired_waiters': waiters, 'legacy_context': str(legacy_context), 'legacy_phase': phase,
        'legacy_contract': campaign.record(legacy_context / 'campaign.json'),
        'validation_lock': str(root / '.training-validation.lock'), 'sources': _sources(), 'policy': POLICY,
        'waiter_retirement_performed_by_this_command': False, 'waiters_must_be_retired_before_run': True,
        'real_training_started': False, 'campaign_success': False}
    _forbidden_artifacts(body)
    campaign.seal(output / 'concurrent-context.json', body)
    context_record = campaign.record(output / 'concurrent-context.json')
    plan = check.prepare(root, output, executor=process.MODE4,
        training_resource_authorization=authorization, concurrent_context=context_record)
    validate_context(context_record, root=root, plan_path=campaign.verify(plan))
    return {'context': context_record, 'plan': plan, 'real_training_started': False,
            'waiters_must_be_retired_before_run': True, 'campaign_success': False}


def validate_context(record, *, root, plan_path=None):
    root = Path(root).resolve()
    path = campaign.verify(record); context = campaign.read(path)
    output = output_path(root, context['output'])
    _bound(record, output / 'concurrent-context.json')
    if (context.get('schema') != SCHEMA or context['root'] != str(root)
            or context['validation_lock'] != str(root / '.training-validation.lock')
            or context['policy'] != POLICY or context['sources'] != _sources()
            or context.get('real_training_started') is not False or context.get('campaign_success') is not False
            or context.get('waiters_must_be_retired_before_run') is not True
            or context.get('waiter_retirement_performed_by_this_command') is not False):
        raise ValueError('concurrent context changed source, scope or contended timing policy')
    authority = resources.validate_authorization(context['training_resource_authorization'], root)
    if authority['benchmark'] != resources.BENCHMARK:
        raise ValueError('concurrent authority changed its two-plus-four worker ceiling')
    legacy_context = root / 'phases' / context['legacy_phase']
    if (Path(context['legacy_context']) != legacy_context
            or re.fullmatch('attempt-[0-9]{3}-pilot', context['legacy_phase']) is None):
        raise ValueError('legacy pilot context changed')
    contract = campaign.read(_bound(context['legacy_contract'], legacy_context / 'campaign.json'))
    if (contract['parent_campaign'] != authority['campaign'] or contract['policy'] != campaign.POLICY
            or contract.get('training_executor') not in (None, {'mode': 'spawn-v2', 'maximum_workers': 2})):
        raise ValueError('concurrent context lost its original two-worker pilot')
    bindings = [context['legacy_training'], *context['retired_waiters']]
    if len(bindings) != 3 or len({binding['process']['pid'] for binding in bindings}) != 3:
        raise ValueError('concurrent process roster changed')
    legacy_pid, waiter_pid = bindings[0]['process']['pid'], bindings[1]['process']['pid']
    common = ['--root', root, '--context', legacy_context, '--phase', context['legacy_phase']]
    _verify_process_binding(bindings[0], root, 'compact_value_bfm_pilot_selection_v2.py', [*common, '--wait-for-labels', 'train'])
    _verify_process_binding(bindings[1], root, 'compact_value_bfm_pilot_gate_v2.py', [*common, '--wait-for-selection', legacy_pid])
    _verify_process_binding(bindings[2], root, 'compact_value_bfm_full_v2.py', ['--root', root, '--pilot-context', legacy_context,
        '--pilot-phase', context['legacy_phase'], '--wait-for-pilot', waiter_pid, 'run'])
    if plan_path is not None:
        plan_path = Path(plan_path).resolve()
        if plan_path != output / 'plan.json':
            raise ValueError('concurrent harness plan escaped its canonical output')
        plan = campaign.read(plan_path)
        if (plan.get('concurrent_equivalence_context') != record
                or plan.get('training_resource_authorization') != context['training_resource_authorization']
                or plan.get('global_heavy_stage_lease_required') is not False
                or plan.get('execution_timing') != 'contended-equivalence-only'
                or plan.get('speedup_qualification_allowed') is not False):
            raise ValueError('harness plan did not bind explicit concurrent-only authority')
    return context


def _legacy_status(context, table):
    expected = context['legacy_training']['process']; current = table.get(expected['pid'])
    if current is not None:
        if identity(current) != expected:
            raise ValueError('legacy training PID was reused or changed its command/start identity')
        return {'status': 'original-training-running', 'process': expected}
    directory = Path(context['legacy_context']) / context['legacy_phase']
    training_path, selection_path = directory / 'training.json', directory / 'model-selection.json'
    if not training_path.is_file() or not selection_path.is_file():
        raise ValueError('legacy training exited without complete source-bound selection; validation stops')
    training, selected = campaign.read(training_path), campaign.read(selection_path)
    if (training.get('smoke') is not False or training.get('mandatory_training_verified') is not True
            or selected.get('training') != campaign.record(training_path)
            or len(training.get('results', [])) != 9 or selected.get('pilot_admitted') is not False):
        raise ValueError('legacy completion does not establish the original nine trained seeds')
    expected = {(weight, seed) for weight in (0., .1, .25) for seed in (20260907, 20260908, 20260909)}
    if {(row['weight'], row['seed']) for row in training['results']} != expected:
        raise ValueError('legacy completion seed roster changed')
    producer = campaign.verify(training['producer'])
    original = context['legacy_training']['source_files'][1]
    if campaign.sha(producer) != original['sha256']:
        raise ValueError('legacy completion came from a different training producer')
    _bound(training['input_audit'], directory / 'training-input-audit.json')
    _bound(selected['policy'], directory / 'selection-policy.json')
    for row in training['results']:
        job = directory / 'training' / f'lambda-{row["weight"]:.2f}'
        _bound(row['source'], job / f'seed-{row["seed"]}.cpp')
        for key, receipt_key in (('runtime', 'quantized_runtime'), ('float_checkpoint', 'float_checkpoint')):
            item = row['seed_receipt'][receipt_key]
            if row[key]['sha256'] != item['sha256']:
                raise ValueError('legacy completion model differs from its seed receipt')
            _bound(row[key], job / item['path'])
    return {'status': 'original-training-completed-with-waiters-held',
            'training': campaign.record(training_path), 'selection': campaign.record(selection_path)}


def descendants(table, root_pid):
    values = {root_pid}
    while True:
        extra = {pid for pid, row in table.items() if row['ppid'] in values} - values
        if not extra:
            return values
        values.update(extra)


def _heavy(row, root):
    try:
        argv = shlex.split(row['command'])
    except ValueError:
        return str(root) in row['command']
    if not argv:
        return False
    program = Path(argv[0]).name
    if program in ('ps', 'rg', 'grep', 'cat', 'head', 'tail', 'git', 'zsh', 'bash', 'sh', 'tee'):
        return False
    if any(Path(arg).name.startswith('compact_value_bfm_') and arg.endswith('.py') for arg in argv[1:]):
        return True
    executable = Path(argv[0])
    if executable.is_absolute() and executable.is_relative_to(root):
        return True
    return program.startswith('papersoccer_') and any(part in program for part in ('gate', 'referee', 'timing', 'state_adapter'))


def guard_runtime(context, *, owned_stage_pid=None, controller_pid=None):
    _forbidden_artifacts(context)
    table = process_table()
    for waiter in context['retired_waiters']:
        if waiter['process']['pid'] in table:
            raise ValueError('expected downstream waiter has not been retired or its PID was reused')
    legacy = _legacy_status(context, table)
    controller_pid = controller_pid or os.getpid()
    allowed = {controller_pid}
    if legacy['status'] == 'original-training-running':
        allowed.update(descendants(table, context['legacy_training']['process']['pid']))
    if owned_stage_pid is not None:
        allowed.update(descendants(table, owned_stage_pid))
    conflicts = [identity(row) for pid, row in table.items()
                 if pid not in allowed and _heavy(row, Path(context['root']))]
    if conflicts:
        raise ValueError('another qualification/live/training process would overlap validation: ' +
                         ', '.join(str(row['pid']) for row in conflicts))
    return legacy


@contextlib.contextmanager
def validation_lock(context):
    path = Path(context['validation_lock'])
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock.fileno()


def _stage_command(context_record, plan_path, stage, parent_fd, lock_fd):
    return [sys.executable, str(Path(__file__).resolve()), '_stage', '--context', context_record['path'],
            '--plan', str(Path(plan_path).resolve()), '--stage', stage, '--parent-watch-fd', str(parent_fd),
            '--validation-lock-fd', str(lock_fd)]


def _stop_owned_group(process):
    # Popen owns this child and its session; the existing pilot never shares it.
    if process.poll() is None:
        if os.getpgid(process.pid) != process.pid:
            raise ValueError('validation child did not retain its dedicated process group')
        os.killpg(process.pid, signal.SIGKILL)
    else:
        # An abnormal leader exit can leave its spawned children in that group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_stage(context_record, plan_path, context, stage, lock_fd):
    from tools import compact_value_bfm_seed_process_check_v2 as check
    output = Path(context['output']); directory = output / 'controller' / stage
    if directory.exists():
        raise ValueError('a concurrent stage already has controller evidence; never retry its timing')
    if (output / stage).exists():
        raise ValueError('concurrent validation requires fresh harness stage outputs')
    before = guard_runtime(context)
    read_fd, write_fd = os.pipe()
    command = _stage_command(context_record, plan_path, stage, read_fd, lock_fd)
    campaign.seal(directory / 'claim.json', {'schema': SCHEMA + '.stage-claim',
        'context': context_record, 'plan': campaign.record(plan_path), 'stage': stage, 'command': command,
        'controller': identity(_live_pid(process_table(), os.getpid())), 'policy': POLICY,
        'legacy_at_start': before, 'retry_allowed': False})
    started = time.monotonic(); process = None; failure = None; status = None
    try:
        with (directory / 'stdout.log').open('xb') as out, (directory / 'stderr.log').open('xb') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, pass_fds=(read_fd, lock_fd),
                start_new_session=True, env={**os.environ, **THREADS, MARKER: '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            os.close(read_fd); read_fd = -1
            owned_identity = identity(_live_pid(process_table(), process.pid))
            group = os.getpgid(process.pid)
            if group != process.pid:
                raise ValueError('owned stage did not create an isolated validation session')
            campaign.seal(directory / 'process.json', {'schema': SCHEMA + '.owned-process',
                'claim': campaign.record(directory / 'claim.json'), 'process': owned_identity,
                'process_group_id': group, 'inherited_validation_lock': context['validation_lock'],
                'controller': identity(_live_pid(process_table(), os.getpid()))})
            while process.poll() is None:
                status = guard_runtime(context, owned_stage_pid=process.pid)
                if time.monotonic() - started > POLICY['owned_stage_watchdog_seconds']:
                    raise ValueError('owned concurrent validation stage exceeded its watchdog')
                time.sleep(POLICY['guard_poll_seconds'])
            if process.returncode != 0:
                raise ValueError('owned concurrent validation stage failed; its claim is spent')
            status = guard_runtime(context)
            check.validated_stage(plan_path, stage)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        if process is not None:
            _stop_owned_group(process)
        raise
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
        result_path = output / stage / 'result.json'
        campaign.seal(directory / 'execution.json', {'schema': SCHEMA + '.stage-execution',
            'claim': campaign.record(directory / 'claim.json'), 'pid': process.pid if process is not None else None,
            'returncode': process.returncode if process is not None else None, 'failure': failure,
            'elapsed_seconds': time.monotonic() - started, 'legacy_at_end': status,
            'stage_result': campaign.record(result_path) if result_path.exists() else None,
            'stdout': campaign.record(directory / 'stdout.log') if (directory / 'stdout.log').exists() else None,
            'stderr': campaign.record(directory / 'stderr.log') if (directory / 'stderr.log').exists() else None,
            'measurements_contended': True, 'speedup_qualification_allowed': False,
            'global_heavy_stage_lease_owned': False, 'existing_training_stopped': False})


def _parent_watch(fd):
    # No other process inherits the controller's write end. EOF means its lock
    # authority disappeared; terminate this validation session and only this one.
    try:
        while os.read(fd, 1):
            pass
    finally:
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpid(), signal.SIGKILL)


def stage_entry(context_record, plan_path, stage, parent_fd, lock_fd):
    from tools import compact_value_bfm_seed_process_check_v2 as check
    context = validate_context(context_record, root=campaign.read(campaign.verify(context_record))['root'], plan_path=plan_path)
    if stage not in POLICY['validation_stage_order'] or os.getpgrp() != os.getpid():
        raise ValueError('owned stage must run in its dedicated validation session')
    claim_path = Path(context['output']) / 'controller' / stage / 'claim.json'
    claim = campaign.read(claim_path)
    if (claim['context'] != context_record or claim['plan'] != campaign.record(plan_path)
            or claim['command'] != _stage_command(context_record, plan_path, stage, parent_fd, lock_fd)
            or claim['stage'] != stage or claim['policy'] != POLICY or claim.get('retry_allowed') is not False):
        raise ValueError('owned validation stage changed its controller claim')
    parent = _live_pid(process_table(), os.getppid())
    if identity(parent) != claim['controller']:
        raise ValueError('owned validation stage lost its original controller process')
    inherited, expected_lock = os.fstat(lock_fd), Path(context['validation_lock']).stat()
    if (inherited.st_dev, inherited.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
        raise ValueError('owned validation stage inherited a different lock descriptor')
    # The inherited descriptor retains the separate lease even after parent death.
    with Path(context['validation_lock']).open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('owned validation stage has no live validation lease')
    threading.Thread(target=_parent_watch, args=(parent_fd,), daemon=True).start()
    guard_runtime(context, owned_stage_pid=os.getpid(), controller_pid=os.getppid())
    return check.run_stage(plan_path, stage)


def run(context_record, plan_path):
    from tools import compact_value_bfm_seed_process_check_v2 as check
    root = campaign.read(campaign.verify(context_record))['root']
    context = validate_context(context_record, root=root, plan_path=plan_path)
    check.validate_plan(plan_path)
    output = Path(context['output'])
    with validation_lock(context) as lock_fd:
        guard_runtime(context)
        claim_path = output / 'controller-claim.json'
        if claim_path.exists() or (output / 'controller').exists():
            raise ValueError('concurrent experiment already claimed; inspect preserved evidence, never retry')
        campaign.seal(claim_path, {'schema': SCHEMA + '.run-claim', 'context': context_record,
            'plan': campaign.record(plan_path), 'controller': identity(_live_pid(process_table(), os.getpid())),
            'validation_lock': context['validation_lock'], 'global_heavy_stage_lease_owned': False,
            'policy': POLICY, 'retry_allowed': False})
        started = time.monotonic(); result = None; failure = None
        try:
            for stage in POLICY['validation_stage_order']:
                _run_stage(context_record, plan_path, context, stage, lock_fd)
            guard_runtime(context)
            result = check.validate_result(plan_path)
            if result.get('speedup_qualification_allowed') is not False:
                raise ValueError('contended comparison must not authorize a speedup claim')
            return result
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            comparison = output / 'comparison.json'
            campaign.seal(output / 'controller-result.json', {'schema': SCHEMA + '.run-result',
                'claim': campaign.record(claim_path), 'completed': result is not None and failure is None,
                'failure': failure, 'elapsed_seconds': time.monotonic() - started,
                'comparison': campaign.record(comparison) if comparison.exists() else None,
                'measurements_contended': True, 'cold_start_and_memory_measured_by_harness': True,
                'speedup_qualification_allowed': False, 'automatic_production_activation': False,
                'global_heavy_stage_lease_owned': False, 'existing_training_stopped': False,
                'waiter_restoration_performed_by_this_command': False, 'campaign_success': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'inspect', 'run', '_stage'))
    for name in ('root', 'output', 'authorization', 'context', 'plan'):
        parser.add_argument('--' + name, type=Path)
    for name in ('legacy-pid', 'pilot-waiter-pid', 'full-waiter-pid', 'parent-watch-fd', 'validation-lock-fd'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--stage', choices=('threads', 'spawn'))
    args = parser.parse_args()
    if args.command == 'prepare':
        if not all((args.root, args.output, args.authorization, args.legacy_pid, args.pilot_waiter_pid, args.full_waiter_pid)):
            parser.error('prepare requires root/output/authorization and the three current process IDs')
        result = prepare(args.root, args.output, campaign.record(args.authorization), args.legacy_pid,
                         args.pilot_waiter_pid, args.full_waiter_pid)
    else:
        if args.context is None or args.plan is None:
            parser.error('inspection/execution requires context and plan')
        record = campaign.record(args.context)
        if args.command == 'run':
            result = run(record, args.plan)
        elif args.command == '_stage':
            if args.parent_watch_fd is None or args.validation_lock_fd is None or args.stage is None:
                parser.error('owned stage requires stage and parent-watch-fd')
            result = stage_entry(record, args.plan, args.stage, args.parent_watch_fd, args.validation_lock_fd)
        else:
            context = validate_context(record, root=campaign.read(args.context)['root'], plan_path=args.plan)
            result = {'context': record, 'legacy': guard_runtime(context), 'new_training_started': False,
                      'measurements_contended': True, 'speedup_qualification_allowed': False}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
