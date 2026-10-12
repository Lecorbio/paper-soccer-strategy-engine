"""Freeze matched4/6 QAT after an audited float runtime under the same numeric plan."""
from __future__ import annotations

import argparse
import copy
import subprocess
import json
from pathlib import Path
import re
import shutil

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_precision_v2 as prior
from tools import rank_two_live_eight_training_v5 as training
from tools import rank_two_live_eight_float_runtime_v1 as float_runtime
from tools import top_three_queue_v5 as q

SCHEMA = 'papersoccer.live-eight.paired-precision.v6'


def validate_runtime_binding(runtime, original_ref, verification, completion, job_ref):
    if (runtime['numeric_plan'] != original_ref or runtime['resource_budget']['workers'] != 10
            or runtime['producer'] != c.record(float_runtime.__file__)
            or completion.get('job') != job_ref or not completion.get('passed')
            or verification.get('numeric_plan') != original_ref
            or not all(verification.get(key) is True for key in
                ('passed', 'all_epochs_native128parity_retained',
                 'all_journals_root_orders_full_moments_verified', 'atomic_head_exact_last_epoch'))):
        raise PermissionError('audited unchanged complete float runtime required')


def continuation_id(original):
    name = Path(original['output']).parent.name
    match = re.fullmatch(r'live-data-continuation-([0-9]{2})', name)
    if not match:
        raise ValueError('a bound continuation namespace is required')
    return match.group(1)


def paired_plans(original, parent, selection, output):
    plans = {}
    for bits in (4, 6):
        plan = copy.deepcopy(original)
        plan.update(producer=c.record(training.__file__), phase_mode='qat',
            float_parent=parent, weight_bits=bits, output=str(Path(output) / f'qat{bits}-v1'),
            paired_precision=True, duplicate_float_work=False)
        plan['inputs'].update(training_producer=plan['producer'],
            tests=c.record(c.ROOT / 'tests/codingame/test_rank_two_live_eight_training_v5.py'),
            native_probe=c.record(training.native_probe.__file__),
            common_float_selection=selection, precision_producer=c.record(__file__))
        plan['resource_budget']['workers'] = 10
        plans[bits] = plan
    return plans


def freeze(base, usage_path, output):
    base, output = Path(base), Path(output).resolve()
    state = c.read(base / 'CURRENT.json')
    c.guard(base, 'freeze', state['owner_thread_id'], usage_path)
    if state.get('active_jobs') or state.get('active_claims') or state.get('pending_exposures'):
        raise PermissionError('finish current work and close metadata before precision freeze')
    job_ref = state['live_eight_next_common_float_job']
    job = c.read(c.verify(job_ref))
    original_ref = state['live_eight_next_common_float_plan']
    original = c.read(c.verify(original_ref))
    identifier = continuation_id(original)
    expected = f'live-eight-continuation-{identifier}-common-float-v1'
    if job['id'] != expected:
        if job['id'] == f'live-eight-continuation-{identifier}-common-float-runtime-v1':
            runtime_ref = state['live_eight_next_common_float_runtime_plan']
            runtime = c.read(c.verify(runtime_ref))
            completion = c.read(c.verify(state['live_eight_next_common_float_complete']))
            if completion['plan'] != runtime_ref:
                raise PermissionError('float completion runtime plan differs')
            verification = c.read(c.verify(completion['verification']))
            validate_runtime_binding(runtime, original_ref, verification, completion, job_ref)
        elif job['id'] != f'live-eight-continuation-{identifier}-common-float-resume-v1':
            raise ValueError('completed float job belongs to another continuation')
        else:
            resumed = c.read(c.verify(state['live_eight_next_common_float_resume_plan']))
            if resumed['numeric_plan'] != original_ref or resumed['resource_budget']['workers'] != 10:
                raise PermissionError('audited successor must retain the original numeric plan')
    status = c.read(Path(state['queue_root']) / 'jobs' / job['id'] / 'state.json')
    if status['status'] != 'complete' or status['job']['sha256'] != job_ref['sha256']:
        raise PermissionError('current exact float queue completion required')
    completed = c.read(c.verify(status['result']))
    if completed['status'] != 'complete':
        raise PermissionError('float result differs')
    directory = Path(original['output'])
    result_path = directory / 'RESULT.json'
    result = c.read(result_path)
    stop = c.read(directory / 'float/STOP.json')
    if (not result['passed'] or result['phase'] != 'float' or not result['epochs_retained']
            or result['duplicate_float_training'] is not False or not 10 <= stop['epochs'] <= 80):
        raise ValueError('complete declared shared float selection required')
    parent = result['common_float_parent']
    checkpoint = c.read(c.verify(parent))
    c.verify(checkpoint['checkpoint'])
    if (checkpoint['cursor']['phase'] != 'float' or checkpoint['plan'] != original_ref
            or result['selected'] != stop['selected']
            or original['fresh_validation'] != state['live_eight_next_fresh_validation']
            or original['corpus'] != state['live_eight_next_corpus']):
        raise ValueError('float parent, selection, corpus or validation binding differs')
    if not output.is_relative_to(directory.parent):
        raise PermissionError('precision output must stay inside its continuation')
    estimated = 512 * 1024**2
    free = shutil.disk_usage(output.parent).free
    if free < 1024**3 + estimated:
        raise prior.training.storage.DiskBudgetExceeded('paired retention and reserve do not fit')
    selection = c.record(result_path)
    evidence = c.read(c.verify(original['fresh_validation']))
    accounting = evidence.get('root_accounting', dict(frozen=128, evaluated=128, excluded=0,
                                                     comparable_early=evidence['early_roots']))
    if (accounting['frozen'] != 128 or accounting['evaluated'] != 128 or accounting['excluded'] != 0
            or evidence.get('gate_approval') or evidence['early_roots'] < 100):
        raise PermissionError('strict complete128-root current validation required')
    declaration = c.immutable(output / 'DECLARATION.json', dict(schema=SCHEMA,
        common_float_parent=parent, common_float_selection=selection,
        selected_epoch=checkpoint['cursor']['epoch'], optimizer_step=checkpoint['optimizer_step'],
        bits=[4, 6], shared_parameters=True, shared_optimizer_moments=True,
        shared_root_seed=original['seed'], calibration_split='train',
        validation=original['fresh_validation'], corpus=original['corpus'],
        validation_root_accounting=accounting,
        qat_minimum=8, qat_maximum=32, float_retraining=False,
        playing_strength_qualified=False, prior_job=job['id'], prior_queue_result=status['result'],
        estimated_retention_bytes=estimated, free_bytes=free, producer=c.record(__file__)))
    plans = paired_plans(original, parent, selection, output)
    jobs = []
    for bits, plan in plans.items():
        branch = Path(plan['output'])
        plan['storage'] = c.immutable(branch / 'STORAGE.json', dict(passed=True,
            free_bytes=free, reserve_bytes=1024**3, maximum_retention_bytes=estimated // 2,
            paired_total_retention_bytes=estimated, compact_update_journal=True,
            full_root_order_once_per_epoch=True, all_epoch_checkpoints_retained=True,
            all_old_checkpoints_unchanged=True, atomic_resumable_state=True))
        plan['inputs']['paired_precision_declaration'] = declaration
        plan_ref = c.immutable(branch / 'PLAN.json', plan)
        branch_job = dict(job)
        branch_job['id'] = f'live-eight-continuation-{identifier}-qat{bits}-v1'
        branch_job['argv'] = [job['argv'][0], '-m', 'tools.rank_two_live_eight_training_v5',
                              '--plan', plan_ref['path']]
        branch_job['limits'] = dict(job['limits'], max_workers=10, max_processes=3, coordinator_threads=0)
        branch_job['inputs'] = [plan_ref, plan['storage'], parent, selection,
            *plan['inputs'].values(), original['corpus'], original['fresh_validation']]
        branch_job['acceptance'] = [dict(path=str(branch/'RESULT.json'), pointer='/passed', equals=True),
            dict(path=str(branch/'RESULT.json'), pointer='/phase', equals='qat'),
            dict(path=str(branch/'RESULT.json'), pointer='/weight_bits', equals=bits)]
        branch_job['compiler'] = dict(executable=c.record(branch_job['argv'][0]),
            version=subprocess.run([branch_job['argv'][0], '--version'], capture_output=True, text=True, check=True).stdout)
        q.queue.validate(branch_job)
        job_record = c.immutable(branch / 'JOB.json', branch_job)
        jobs.append(dict(id=branch_job['id'], job=job_record, plan=plan_ref, weight_bits=bits))
    receipt = c.immutable(output / 'PREPARED.json', dict(passed=True, declaration=declaration,
        jobs=jobs, not_enqueued=True, admission_requires_fresh_usage_storage_and_resource_guard=True))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        current['live_eight_next_precision_prepared'] = receipt
        current['live_eight_next_precision_declaration'] = declaration
        current['live_eight_next_precision_jobs'] = jobs
        c.atomic(base / 'CURRENT.json', current)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--usage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.base, args.usage, args.output)))
