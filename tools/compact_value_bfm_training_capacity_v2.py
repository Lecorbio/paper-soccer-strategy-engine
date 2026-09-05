#!/usr/bin/env python3
"""Measure four simultaneous production-input reconstructions without training.

The coordinator and four distinct spawned children retain their datasets during
sampling. This does not pin mapped pages, measure optimizer/batch workspace, or
automatically establish production readiness. Every child retains the global
lease and exits on coordinator death, so an orphan cannot overlap timed work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import heapq
import json
import math
import multiprocessing
from multiprocessing.connection import wait
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_seed_process_v2 as process

if __name__ == '__main__':
    os.environ.update(process.ENVIRONMENT)
    os.environ[process.MARKER] = '1'

SCHEMA = 'compact-value-bfm-trained-v2.training-capacity.v2'
POLICY = {'workers': 4, 'start_method': 'spawn', 'numerical_threads_per_worker': 1,
          'coordinator_inputs_retained': True, 'all_children_ready_before_measurement': True,
          'child_retains_global_lock_description': True, 'child_exits_on_parent_sentinel': True,
          'mapped_pages_pinned_or_fully_prefaulted': False,
          'batch_workspace_measured': False, 'optimizer_steps': 0,
          'training_artifacts_written': False, 'automatic_production_activation': False}


def modules():
    from tools import compact_value_bfm_campaign_v2 as campaign
    from tools import compact_value_bfm_train as trainer
    from tools import compact_value_bfm_training_resources_v2 as resources
    from tools import compact_value_bfm_seed_process_check_v2 as sampling
    return campaign, trainer, resources, sampling


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def runtime():
    return {'python_executable': str(Path(sys.executable).resolve()), 'python_version': sys.version,
            'platform': platform.platform(), 'machine': platform.machine()}


def output_path(root, output):
    allowed = Path(root).resolve() / 'diagnostics/training-capacity'
    output = Path(output).resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('capacity output must be a fresh child of diagnostics/training-capacity')
    return output


def prepare(root, context, phase, output, *, authorization=None, hold_seconds=10, reconstruction_timeout_seconds=900):
    campaign, _trainer, resources, _sampling = modules()
    root, context = Path(root).resolve(), Path(context).resolve()
    output = output_path(root, output)
    if (type(hold_seconds) is not int or not 2 <= hold_seconds <= 60
            or type(reconstruction_timeout_seconds) is not int or not 1 <= reconstruction_timeout_seconds <= 3600):
        raise ValueError('capacity hold/timeout is outside its bounded diagnostic range')
    contract = campaign.read(context / 'campaign.json')
    if (context != root / 'phases' / phase or phase != context.name
            or contract.get('phase') not in ('pilot', 'full')
            or phase != f'attempt-{contract["attempt"]:03d}-{contract["phase"]}'
            or contract.get('parent_campaign') != campaign.record(root / 'campaign.json')
            or contract.get('policy') != campaign.POLICY):
        raise ValueError('capacity requires an actual source-bound campaign phase')
    auth = (campaign.record(resources.authorization_path(root)) if authorization is None else
            authorization if isinstance(authorization, dict) else campaign.record(authorization))
    resources.validate_authorization(auth, root)
    audit_path = context / phase / 'training-input-audit.json'
    audit = campaign.read(audit_path)
    if (audit.get('schema') != campaign.ID + '.training-input-audit.v2'
            or audit.get('bundle') != contract['bundle'] or audit.get('protected_tests_opened') is not False
            or audit.get('labels') != campaign.record(context / phase / 'labels.json')
            or audit.get('position_closure') != campaign.record(context / phase / 'positions.json')):
        raise ValueError('capacity requires the already prepared audited production inputs')
    body = {'schema': SCHEMA + '.plan', 'root': str(root), 'context': str(context), 'phase': phase,
            'output': str(output), 'policy': POLICY, 'runtime': runtime(),
            'campaign': campaign.record(root / 'campaign.json'), 'phase_contract': campaign.record(context / 'campaign.json'),
            'input_audit': campaign.record(audit_path), 'training_resource_authorization': auth,
            'sources': process.source_closure(), 'hold_seconds': hold_seconds,
            'reconstruction_timeout_seconds': reconstruction_timeout_seconds,
            'preparation_reconstructed_inputs': False, 'qualification_eligible': False}
    campaign.seal(output / 'plan.json', body)
    return campaign.record(output / 'plan.json')


def validate_plan(path):
    campaign, _trainer, resources, _sampling = modules()
    path = Path(path).resolve(); plan = campaign.read(path)
    root, context = Path(plan['root']), Path(plan['context'])
    if (plan.get('schema') != SCHEMA + '.plan' or plan.get('policy') != POLICY
            or output_path(root, plan['output']) / 'plan.json' != path
            or plan['runtime'] != runtime() or plan['sources'] != process.source_closure()
            or plan.get('preparation_reconstructed_inputs') is not False
            or plan.get('qualification_eligible') is not False
            or type(plan.get('hold_seconds')) is not int or not 2 <= plan['hold_seconds'] <= 60
            or type(plan.get('reconstruction_timeout_seconds')) is not int
            or not 1 <= plan['reconstruction_timeout_seconds'] <= 3600):
        raise ValueError('capacity plan/source/runtime changed')
    if (campaign.verify(plan['campaign']) != root / 'campaign.json'
            or campaign.verify(plan['phase_contract']) != context / 'campaign.json'
            or campaign.verify(plan['input_audit']) != context / plan['phase'] / 'training-input-audit.json'
            or context != root / 'phases' / plan['phase']):
        raise ValueError('capacity plan input paths changed')
    resources.validate_authorization(plan['training_resource_authorization'], root)
    contract = campaign.read(context / 'campaign.json'); audit = campaign.read(context / plan['phase'] / 'training-input-audit.json')
    if (contract['parent_campaign'] != plan['campaign'] or contract['policy'] != campaign.POLICY
            or audit['bundle'] != contract['bundle'] or audit['protected_tests_opened'] is not False):
        raise ValueError('capacity plan no longer binds the prepared phase')
    return plan


def lock_identity(fd):
    status = os.fstat(fd)
    return {'device': status.st_dev, 'inode': status.st_ino}


def retain_shared_lock(ticket, expected):
    fd = ticket.detach()
    try:
        if lock_identity(fd) != expected:
            raise ValueError('capacity child received another global lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def parent_death_guard():
    parent = multiprocessing.parent_process()
    if parent is None:
        raise ValueError('capacity child requires a managed spawning parent')
    def watch():
        wait([parent.sentinel])
        os._exit(72)
    threading.Thread(target=watch, name='capacity-parent-liveness', daemon=True).start()


def clear_failed_frames(error):
    """Exceptions from reconstruction must not retain arrays past lease close."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)


def restore_ticket(ticket):
    return ticket


class SpawnLockTicket:
    """Reduce during spawn, avoiding an unconsumed resource-sharer fd cache."""
    def __init__(self, fd):
        self.fd = fd

    def __reduce__(self):
        return restore_ticket, (DupFd(self.fd),)


def retain_until_release(connection, timeout):
    if not connection.poll(timeout) or connection.recv() != 'hold':
        raise TimeoutError('capacity measured barrier did not release')
    connection.send({'kind': 'holding', 'pid': os.getpid()})
    if not connection.poll(timeout + 180) or connection.recv() != 'release':
        raise TimeoutError('capacity parent did not release retained inputs')


def child(plan_path, ordinal, connection, lock_ticket, expected_lock):
    owned_lock = None
    bundle = inputs = None
    try:
        owned_lock = retain_shared_lock(lock_ticket, expected_lock)
        parent_death_guard()
        plan = validate_plan(plan_path)
        _campaign, trainer, _resources, _sampling = modules()
        with trainer.native_thread_execution_scope() as native:
            bundle, inputs, identity = process.reconstruct_inputs(Path(plan['context']), plan['phase'])
            connection.send({'kind': 'ready', 'ordinal': ordinal, 'pid': os.getpid(), 'identity': identity,
                             'native_thread_execution': native, 'runtime': runtime(), 'lock': lock_identity(owned_lock)})
            retain_until_release(connection, plan['reconstruction_timeout_seconds'] + plan['hold_seconds'])
            if bundle is None or inputs is None:
                raise ValueError('capacity child lost its retained inputs')
    except BaseException as error:
        clear_failed_frames(error)
        try:
            connection.send({'kind': 'error', 'ordinal': ordinal, 'pid': os.getpid(),
                             'error': type(error).__name__ + ': ' + str(error)})
        except (OSError, EOFError):
            pass
        raise
    finally:
        bundle = inputs = None
        gc.collect()
        connection.close()
        if owned_lock is not None:
            os.close(owned_lock)  # Never LOCK_UN: other children share this description.


def observe_system(pids):
    commands = {'processes': ['ps', '-p', ','.join(map(str, [os.getpid(), *pids])), '-o', 'pid=,ppid=,rss=,vsz=,comm=']}
    if sys.platform == 'darwin':
        commands.update({'hardware_swap': ['sysctl', 'hw.memsize', 'vm.swapusage'],
                         'vm_stat': ['vm_stat'], 'memory_pressure': ['memory_pressure', '-Q']})
        commands.update({f'vmmap-{pid}': ['vmmap', '-summary', str(pid)] for pid in [os.getpid(), *pids]})
    records = {}
    for name, command in commands.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            records[name] = {'command': command, 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr}
        except subprocess.TimeoutExpired:
            records[name] = {'command': command, 'timeout': True}
    return {'observed_at_utc': utc_now(), 'commands': records}


def send_release(connections):
    for connection in connections:
        try:
            connection.send('release')
        except (OSError, EOFError):
            pass


def join_owned(children, connections):
    send_release(connections)
    for item in children:
        item.join(timeout=5)
    for item in children:
        if item.is_alive():
            item.terminate()
    for item in children:
        item.join(timeout=5)
    for item in children:
        if item.is_alive():
            item.kill()
        item.join()


def hold_four(plan_path, plan, identity, sampler, lock_fd, *, worker_target=child):
    """Measured ready/holding barrier using death-safe per-child pipes."""
    context = multiprocessing.get_context('spawn')
    children, connections, ready = [], [], {}
    lock = lock_identity(lock_fd)
    started = time.monotonic()
    success = False
    try:
        os.environ.update(process.ENVIRONMENT); os.environ[process.MARKER] = '1'
        for ordinal in range(4):
            parent_connection, child_connection = context.Pipe(duplex=True)
            connections.append(parent_connection)
            item = context.Process(target=worker_target, args=(str(plan_path), ordinal, child_connection,
                                   SpawnLockTicket(lock_fd), lock), name=f'capacity-inputs-{ordinal}')
            try:
                item.start()
            finally:
                child_connection.close()
                if item.pid is not None:
                    children.append(item)
        deadline = time.monotonic() + plan['reconstruction_timeout_seconds']
        while len(ready) < 4:
            if time.monotonic() >= deadline:
                raise TimeoutError('four capacity children did not become ready')
            if any(item.exitcode is not None for item in children):
                raise ValueError('capacity child exited before the four-process barrier')
            pending = [connection for ordinal, connection in enumerate(connections) if ordinal not in ready]
            for connection in wait(pending, timeout=min(.1, max(.001, deadline - time.monotonic()))):
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise ValueError('capacity child closed its readiness channel') from error
                ordinal = message.get('ordinal')
                if message.get('kind') == 'error':
                    raise ValueError('capacity child error: ' + message['error'])
                if (message.get('kind') != 'ready' or type(ordinal) is not int or ordinal not in range(4)
                        or connection is not connections[ordinal] or ordinal in ready
                        or message['pid'] != children[ordinal].pid or message['identity'] != identity
                        or message['lock'] != lock or message['runtime'] != runtime()):
                    raise ValueError('capacity ready identity/ownership differs from the coordinator')
                _campaign, trainer, _resources, _sampling = modules()
                trainer.validate_native_thread_execution(message['native_thread_execution'])
                ready[ordinal] = message
        for connection in connections:
            connection.send('hold')
        for ordinal, connection in enumerate(connections):
            if not connection.poll(max(.001, deadline - time.monotonic())):
                raise TimeoutError('capacity child did not acknowledge simultaneous retention')
            if connection.recv() != {'kind': 'holding', 'pid': children[ordinal].pid}:
                raise ValueError('capacity holding acknowledgment changed ownership')
        sampler.sample()
        observation = observe_system([item.pid for item in children])
        held_at = utc_now(); hold_deadline = time.monotonic() + plan['hold_seconds']
        while time.monotonic() < hold_deadline:
            if any(item.exitcode is not None for item in children):
                raise ValueError('capacity child exited during simultaneous retention')
            time.sleep(min(.1, max(.001, hold_deadline - time.monotonic())))
        sampler.sample()
        send_release(connections)
        for item in children:
            item.join(timeout=10)
        if any(item.exitcode != 0 for item in children):
            raise ValueError('capacity child did not release and exit cleanly')
        success = True
        return {'children': [ready[index] for index in range(4)], 'all_four_ready_and_held': True,
                'hold_started_at_utc': held_at, 'minimum_hold_seconds': plan['hold_seconds'],
                'elapsed_seconds': time.monotonic() - started, 'system_observation': observation,
                'child_exitcodes': [item.exitcode for item in children]}
    finally:
        if not success:
            join_owned(children, connections)
        for connection in connections:
            connection.close()


def workspace_dimensions(inputs):
    """Metadata bounds only; no model forward, gradients, or tensor workspaces."""
    _campaign, trainer, _resources, _sampling = modules()
    groups = inputs.successor_rankings.train
    batches = math.ceil(len(inputs.new) / trainer.NEW_ROWS_PER_BATCH)
    standard = all(trainer._ranking_group_profile(group) == 'standard-v1' for group in groups)
    maximum_groups = math.ceil(len(groups) / batches) if standard and batches else None
    successors = sum(heapq.nlargest(maximum_groups, (len(group.successors) for group in groups))) if maximum_groups else None
    any_group = next(iter(groups or inputs.successor_rankings.validation), None)
    store = getattr(getattr(any_group, 'successors', None), 'store', None)
    arrays = store.document['arrays'] if store is not None else None
    return {'train_groups': len(groups), 'new_scalar_rows': len(inputs.new), 'scalar_batches_per_epoch': batches,
            'mapped_array_artifacts': arrays,
            'mapped_array_extent_bytes': sum(row['bytes'] for row in arrays.values()) if arrays is not None else None,
            'mapped_array_resident_bytes_not_inferred_from_extent': True,
            'maximum_single_train_group_successors': max(map(lambda group: len(group.successors), groups), default=0),
            'unweighted_all_groups_upper_bound_used': standard,
            'ranking_groups_per_batch_upper_bound': maximum_groups,
            'successors_per_batch_conservative_upper_bound': successors,
            'forward_cache_bytes_without_predictions_per_worker': 164 * successors if successors is not None else None,
            'bound_includes_noncomparable_groups': True, 'actual_batch_not_observed': True,
            'backward_temporary_view_and_allocator_bytes_included': False}


def run(plan_path):
    campaign, trainer, _resources, sampling = modules()
    plan_path = Path(plan_path).resolve(); plan = validate_plan(plan_path)
    output = Path(plan['output']); root = Path(plan['root'])
    # Same ordinary global lease as campaign.lease, retaining its fd for DupFd.
    with (root / '.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_plan(plan_path)
        if any((output / name).exists() for name in ('claim.json', 'result.json', 'failure.json')):
            raise ValueError('capacity execution is spent; never duplicate or retry a claimed load')
        campaign.seal(output / 'claim.json', {'schema': SCHEMA + '.claim', 'plan': campaign.record(plan_path),
                      'started_at_utc': utc_now(), 'pid': os.getpid(), 'global_lock': lock_identity(lock.fileno()),
                      'retry_allowed': False, 'policy': POLICY})
        bundle = inputs = None
        try:
            with sampling.MemorySampler() as sampler:
                with trainer.native_thread_execution_scope() as native:
                    bundle, inputs, identity = process.reconstruct_inputs(Path(plan['context']), plan['phase'])
                    dimensions = workspace_dimensions(inputs)
                    cohort = hold_four(plan_path, plan, identity, sampler, lock.fileno())
                    if bundle is None or inputs is None:
                        raise ValueError('coordinator lost retained inputs')
            memory = sampler.report()
            pids = [row['pid'] for row in cohort['children']]
            complete = (not memory['errors'] and memory['samples'] >= 2
                        and all(str(pid) in memory['per_pid_peak_rss'] for pid in [os.getpid(), *pids]))
            result = campaign.seal(output / 'result.json', {'schema': SCHEMA + '.result',
                'plan': campaign.record(plan_path), 'claim': campaign.record(output / 'claim.json'),
                'completed_at_utc': utc_now(), 'policy': POLICY, 'runtime': runtime(),
                'coordinator': {'pid': os.getpid(), 'identity': identity, 'native_thread_execution': native},
                'cohort': cohort, 'memory': memory, 'memory_measurements_complete': complete,
                'workspace_dimensions_metadata_only': dimensions,
                'preparation_peak_measured': False,
                'mapped_residency_interpretation': 'Objects and mappings were retained together; file pages were not pinned or fully prefaulted and may be paged by the OS.',
                'production_peak_headroom_proven': False, 'automatic_production_activation': False,
                'optimizer_steps': 0, 'training_artifacts_written': False, 'qualification_eligible': False})
            return result
        except BaseException as error:
            clear_failed_frames(error)
            campaign.seal(output / 'failure.json', {'schema': SCHEMA + '.failure',
                'plan': campaign.record(plan_path), 'claim': campaign.record(output / 'claim.json'),
                'failed_at_utc': utc_now(), 'error': type(error).__name__ + ': ' + str(error),
                'owned_children_joined_before_lease_release': True, 'retry_allowed': False})
            raise
        finally:
            bundle = inputs = None
            gc.collect()  # Release retained arrays before closing the final parent lease fd.


def validate_identity(identity, plan):
    campaign, trainer, _resources, _sampling = modules()
    body = {key: value for key, value in identity.items() if key != 'body_sha256'}
    audit = campaign.read(campaign.verify(plan['input_audit']))
    bundle = trainer.FrozenBundle.load(campaign.verify(audit['bundle']))
    expected_routes = {'anchor': list(bundle.canonical_routes('train')),
        'canonical_validation': list(bundle.canonical_routes('validation')),
        'common_adjudicator': [bundle.common_adjudicator_route()],
        'new': [audit['shards']['train']['manifest']['path']]}
    if (identity.get('body_sha256') != campaign.hashlib.sha256(campaign.raw(body)).hexdigest()
            or identity.get('input_audit') != audit or identity.get('source_routes') != expected_routes
            or identity.get('split_isolation') != {'closure_audit': audit['body_sha256']}
            or identity.get('paired_row_validation') != {'external_source_bound': True}
            or identity['datasets']['new']['source_manifest_sha256'] != audit['shards']['train']['manifest']['sha256']
            or identity['datasets']['new']['source_npz_sha256'] != audit['shards']['train']['npz']['sha256']
            or identity['ranking']['artifact_sha256'] != audit['ranking_store']['sha256']
            or identity['anchor_filter']['removed_rows'] != audit['anchor_duplicates_removed']):
        raise ValueError('capacity reconstruction identity lost its actual audited inputs and routes')


def validate_result(plan_path):
    campaign, trainer, _resources, sampling = modules()
    plan = validate_plan(plan_path); output = Path(plan['output'])
    result = campaign.read(output / 'result.json')
    if (result['plan'] != campaign.record(plan_path) or result['claim'] != campaign.record(output / 'claim.json')
            or result.get('schema') != SCHEMA + '.result' or result.get('policy') != POLICY
            or result.get('runtime') != runtime() or result.get('optimizer_steps') != 0
            or result.get('training_artifacts_written') is not False
            or result.get('production_peak_headroom_proven') is not False
            or result.get('automatic_production_activation') is not False
            or result.get('qualification_eligible') is not False):
        raise ValueError('capacity result changed its no-training measurement scope')
    claim = campaign.read(campaign.verify(result['claim']))
    if (claim.get('schema') != SCHEMA + '.claim' or claim['plan'] != result['plan']
            or claim['retry_allowed'] is not False or claim['policy'] != POLICY):
        raise ValueError('capacity claim changed')
    children = result['cohort']['children']; identity = result['coordinator']['identity']
    if (len(children) != 4 or [row['ordinal'] for row in children] != list(range(4))
            or len({row['pid'] for row in children}) != 4
            or any(type(row['pid']) is not int or row['pid'] <= 0 or row['pid'] == claim['pid'] for row in children)
            or result['coordinator']['pid'] != claim['pid']
            or result['cohort']['all_four_ready_and_held'] is not True
            or result['cohort']['child_exitcodes'] != [0, 0, 0, 0]
            or result['cohort']['minimum_hold_seconds'] != plan['hold_seconds']
            or any(row['identity'] != identity or row['lock'] != claim['global_lock'] or row['runtime'] != runtime() for row in children)):
        raise ValueError('capacity cohort did not preserve four source-bound input owners')
    validate_identity(identity, plan)
    for row in [result['coordinator'], *children]:
        trainer.validate_native_thread_execution(row['native_thread_execution'])
    sampling.validate_memory(result['memory'])
    memory = result['memory']
    pids = [claim['pid'], *(row['pid'] for row in children)]
    complete = (not memory['errors'] and memory['samples'] >= 2
                and all(str(pid) in memory['per_pid_peak_rss'] for pid in pids))
    if result['memory_measurements_complete'] is not complete:
        raise ValueError('capacity memory completeness differs from sampled process identities')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'context', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--phase', required=True); prep.add_argument('--authorization', type=Path)
    prep.add_argument('--hold-seconds', type=int, default=10)
    prep.add_argument('--reconstruction-timeout-seconds', type=int, default=900)
    for name in ('run', 'validate'):
        sub.add_parser(name).add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.context, args.phase, args.output, authorization=args.authorization,
                         hold_seconds=args.hold_seconds, reconstruction_timeout_seconds=args.reconstruction_timeout_seconds)
    else:
        result = run(args.plan) if args.command == 'run' else validate_result(args.plan)
    print(json.dumps(result if args.command == 'prepare' else {'memory_measurements_complete': result['memory_measurements_complete'],
                     'production_peak_headroom_proven': False, 'optimizer_steps': 0}), flush=True)


if __name__ == '__main__':
    main()
