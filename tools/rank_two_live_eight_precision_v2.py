"""Freeze matched precision plans with bounded native compilation and journals."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from tools import rank_two_live_eight_v1 as campaign
from tools import rank_two_live_eight_training_v4 as training
from tools import top_three_queue_v5 as queue

SCHEMA = 'papersoccer.live-eight.paired-precision.v2'


def paired_plans(original, parent, selection, output):
    plans = {}
    for bits in (4, 6):
        plan = copy.deepcopy(original)
        plan.update(producer=campaign.record(training.__file__), phase_mode='qat',
                    float_parent=parent, weight_bits=bits,
                    output=str(Path(output) / f'qat{bits}-v1'),
                    paired_precision=True, duplicate_float_work=False)
        plan['inputs'].update(training_producer=plan['producer'],
                              tests=campaign.record(Path(campaign.ROOT) / 'tests/codingame/test_rank_two_live_eight_training_v4.py'),
                              native_probe=campaign.record(training.native_probe.__file__),
                              common_float_selection=selection,
                              precision_producer=campaign.record(__file__))
        # One numerical trainer; serial compiler/probe subprocesses also count
        # toward the queue's resident-thread allowance.
        plan['resource_budget']['workers'] = 3
        plans[bits] = plan
    return plans


def prepare(base, usage_path, output):
    base, output = Path(base), Path(output).resolve()
    state = campaign.read(base / 'CURRENT.json')
    campaign.guard(base, 'freeze', state['owner_thread_id'], usage_path)
    current_job = campaign.read(campaign.verify(state['live_eight_common_float_job']))
    queue_root = Path(state['queue_root'])
    job_directory = queue_root / 'jobs' / current_job['id']
    job_state = campaign.read(job_directory / 'state.json')
    if job_state.get('status') != 'complete':
        raise PermissionError('common float training must finish before precision admission')
    completed = campaign.read(campaign.verify(job_state['result']))
    if completed.get('status') != 'complete':
        raise PermissionError('common float queue result differs')
    original = campaign.read(campaign.verify(state['live_eight_common_float_plan']))
    float_directory = Path(original['output'])
    result_path = float_directory / 'RESULT.json'
    result = campaign.read(result_path)
    if (result.get('passed') is not True or result.get('phase') != 'float'
            or result.get('epochs_retained') is not True
            or result.get('duplicate_float_training') is not False):
        raise ValueError('verified shared float selection required')
    parent = result['common_float_parent']
    checkpoint = campaign.read(campaign.verify(parent))
    if checkpoint['cursor']['phase'] != 'float':
        raise ValueError('precision arms require the same float checkpoint and moments')
    campaign.verify(checkpoint['checkpoint'])
    stop = campaign.read(float_directory / 'float/STOP.json')
    if not 10 <= stop['epochs'] <= 80:
        raise ValueError('common float minimum/cap differs')
    estimated = 512 * 1024**2
    available = shutil.disk_usage(output.parent).free
    if available < training.storage.DISK_RESERVE + estimated:
        raise training.storage.DiskBudgetExceeded('paired precision retention plus reserve does not fit')
    selection = campaign.record(result_path)
    plans = paired_plans(original, parent, selection, output)
    receipt = campaign.immutable(output / 'DECLARATION.json', dict(
        schema=SCHEMA, common_float_parent=parent, common_float_selection=selection,
        bits=[4, 6], shared_parameters=True, shared_optimizer_moments=True,
        shared_root_seed=original['seed'], calibration_split='train',
        selection=original['selection'], validation=original['fresh_validation'],
        qat_minimum=8, qat_maximum=32, float_retraining=False,
        independent_live_qualification_required=True, prior_job=current_job['id'],
        prior_queue_result=job_state['result'], estimated_retention_bytes=estimated,
        free_bytes=available, producer=campaign.record(__file__)))
    admitted = []
    for bits, plan in plans.items():
        directory = Path(plan['output'])
        plan['storage'] = campaign.immutable(directory / 'STORAGE.json', dict(
            compact_update_journal=True, full_root_order_once_per_epoch=True,
            all_epoch_checkpoints_retained=True, all_old_checkpoints_unchanged=True,
            maximum_retention_bytes=estimated // 2, atomic_resumable_state=True))
        plan['inputs']['paired_precision_declaration'] = receipt
        plan_ref = campaign.immutable(directory / 'PLAN.json', plan)
        job = copy.deepcopy(current_job)
        job['id'] = f'live-eight-precision-01-qat{bits}-v1'
        job['argv'] = [job['argv'][0], '-m', 'tools.rank_two_live_eight_training_v4',
                       '--plan', plan_ref['path']]
        job['limits'].update(max_workers=3, max_processes=3, coordinator_threads=0)
        job['inputs'] = [plan_ref, plan['storage'], parent, selection,
                         *plan['inputs'].values(), original['corpus'], original['fresh_validation']]
        job['acceptance'] = [dict(path=str(directory / 'RESULT.json'), pointer='/passed', equals=True),
                             dict(path=str(directory / 'RESULT.json'), pointer='/phase', equals='qat'),
                             dict(path=str(directory / 'RESULT.json'), pointer='/weight_bits', equals=bits)]
        job_ref = campaign.immutable(directory / 'JOB.json', job)
        campaign.guard(base, 'enqueue', state['owner_thread_id'], usage_path,
                       budget=plan['resource_budget'])
        queue.queue.enqueue(queue_root, job)
        admitted.append(dict(id=job['id'], job=job_ref, plan=plan_ref, weight_bits=bits))
    with campaign.locked(base):
        current = campaign.read(base / 'CURRENT.json')
        current['live_eight_common_float_complete'] = selection
        current['live_eight_precision_declaration'] = receipt
        current['live_eight_precision_jobs'] = admitted
        current['active_jobs'] = admitted
        current['phase'] = 'live-eight-paired-precision-training'
        current['next_action'] = (
            'Run admitted 4bit then6bit QAT jobs serially with fresh actual usage before each launch. '
            'Both use the same selected float parameters/moments,598TRAIN roots/half retained half live, '
            'and128fresh independent VALIDATION roots. Retain every epoch and compact update journal; '
            'no duplicated float work. Complete native selection before source proofs/clock ladder/live admission. '
            'No fresh bank until all validation/teacher/recipe/diagnostic exposures close incrementally. '
            'Original19Oct finalist cutoff/21Oct deadline,12versions/two late reserves and bans remain unchanged.')
        campaign.atomic(base / 'CURRENT.json', current)
    return dict(passed=True, declaration=receipt, jobs=admitted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--usage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.usage, args.output)))
