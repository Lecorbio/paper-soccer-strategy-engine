"""Freeze and advance isolated response cases for one continuation clock ladder."""
from __future__ import annotations

import copy
from pathlib import Path
import re

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_timing_v2 as timing
from tools import top_three_queue_v5 as queue

PREFIX = 'live_eight_next_clock_response_'


def case_grid(models):
    clocks = [tuple(row['model']['clocks_ms']) for row in models]
    if len(models) != 4 or set(clocks) != {tuple(x) for x in c.PLAN['clock_ladder']}:
        raise ValueError('four unique declared clock variants required')
    runtimes = [row['model']['runtime'] for row in models]
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError('one exact tensor runtime required across the ladder')
    if any(row['model']['profile'] != 'dd8' for row in models):
        raise ValueError('continuation response cases require the fixed eight/eight model')
    return [(row, color, offset) for row in sorted(models, key=lambda row: row['model']['clocks_ms'])
            for color in (0, 1) for offset in (0, 3)]


def next_case(cases, completed):
    if len(completed) > len(cases):
        raise ValueError('completed cases exceed the frozen declaration')
    return None if len(completed) == len(cases) else cases[len(completed)]


def prepare(base, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    c.guard(base, 'freeze', state['owner_thread_id'], usage)
    if state.get('active_jobs') or state.get('active_claims') or state.get(PREFIX + 'cases'):
        raise PermissionError('finish current work; do not replace an existing response declaration')
    complete = state['live_eight_next_clock_build_complete']
    saved = c.read(c.verify(complete))
    result = c.read(c.verify(saved['result']))
    if not saved['passed'] or not result['passed']:
        raise PermissionError('verified production builds required')
    root = Path(saved['result']['path']).parent
    match = re.fullmatch(r'live-data-continuation-([0-9]{2})', root.parent.parent.name)
    if not match:
        raise ValueError('response cases must belong to a continuation namespace')
    identifier = match[1]
    prototype = c.read(c.verify(state['live_eight_sixbit_550_certification_cases'][0]['job']))
    budget = dict(workers=2, numerical_threads=1, rss_bytes=4 * 1024**3,
                  wall_seconds=300, cpu_capacity_seconds=3000)
    cases = []
    for row, color, offset in case_grid(result['models']):
        model = row['model']
        build = c.read(c.verify(row['build']))
        if build['source'] != model['source'] or build['runtime'] != model['runtime']:
            raise ValueError('prebuilt exact source differs')
        c.verify(build['binary'])
        tag = row['tag'] + f'-p{color}-offset{offset}'
        output = root.parent / 'certification-v1' / tag
        job_id = f'live-eight-c{identifier}-clock-{tag}-v1'
        plan = dict(schema=c.SCHEMA + '.production-case.v2.plan', producer=c.record(timing.__file__),
            campaign_current=str(base / 'CURRENT.json'), owner_thread_id=state['owner_thread_id'],
            activation=state['focused_activation'], output=str(output), source=model['source'],
            runtime=model['runtime'], export=model['export'], build=row['build'], clocks_ms=model['clocks_ms'],
            case=dict(color=color, direction_offset=offset), job_id=job_id, resource_budget=budget,
            inputs=dict(build_complete=complete, mechanical=state['live_eight_next_clock_mechanical_complete'],
                producer=c.record(timing.__file__), case_helper=c.record(__file__),
                coverage=c.record(c.ROOT / 'tools/rank_two_focused_coverage_v1.py')),
            training_eligible=False, strength_eligible=False)
        reference = c.immutable(output / 'PLAN.json', plan)
        job = copy.deepcopy(prototype)
        job.update(id=job_id, argv=[job['argv'][0], '-m', 'tools.rank_two_live_eight_timing_v2',
            '--plan', reference['path'], '--mode', 'case'], dependencies=[],
            inputs=[reference, model['source'], model['runtime'], model['export'], row['build'],
                    build['binary'], *plan['inputs'].values()],
            acceptance=[dict(path=str(output / 'RESULT.json'), pointer='/passed', equals=True)])
        queue.queue.validate(job)
        jobref = c.immutable(output / 'JOB.json', job)
        cases.append(dict(id=job_id, job=jobref, plan=reference, model_tag=row['tag'], case=plan['case']))
    frozen = c.immutable(root.parent / 'certification-v1/DECLARATION.json', dict(
        schema=c.SCHEMA + '.continuation-clock-response-declaration', cases=cases,
        production_builds=complete, direction_offsets=[0, 3], both_colors=True,
        minimum_longest_response_edges=8, serial_isolated=True, training_eligible=False,
        playing_strength_qualified=False, producer=c.record(__file__)))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        current[PREFIX + 'declaration'] = frozen
        current[PREFIX + 'cases'] = cases
        current[PREFIX + 'completed'] = []
        current['live_eight_next_clock_case_helper'] = c.record(__file__)
        current['phase'] = 'live-eight-next-continuation-isolated-response'
        current['next_action'] = (
            'Advance the sixteen frozen continuation response cases serially. Fresh actual usage, '
            'source and resource guards precede each admission and launch. Close every complete case '
            'before advancing; never replay a spent case. No compiler, training, helper or browser '
            'claims during timing. After all pass aggregate the highest safe declared clock, retain '
            'all exposures and validate incremental ancestry before live admission. Own failures '
            'require exact/semantic source stop, retained-evidence classification and incumbent restoration.')
        c.atomic(base / 'CURRENT.json', current)
    return frozen


def close_current(base, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    c.guard(base, 'archive', state['owner_thread_id'], usage)
    if not state.get('active_jobs'):
        return None
    if len(state['active_jobs']) != 1:
        raise PermissionError('one isolated active case required')
    previous = state['active_jobs'][0]
    expected = next_case(state[PREFIX + 'cases'], state[PREFIX + 'completed'])
    if previous != expected:
        raise PermissionError('active case differs from the frozen consecutive order')
    status = c.read(Path(state['queue_root']) / 'jobs' / previous['id'] / 'state.json')
    if status['status'] != 'complete' or status['job']['sha256'] != previous['job']['sha256']:
        raise PermissionError('active or spent case requires observation or classification')
    queue_result = c.read(c.verify(status['result']))
    if queue_result['status'] != 'complete':
        raise PermissionError('case queue completion differs')
    plan = c.read(c.verify(previous['plan']))
    out = Path(plan['output'])
    proof = c.read(out / 'RESULT.json')
    if not proof['passed'] or proof['own_failures'] != 0:
        raise PermissionError('source failure requires stop and restoration')
    for key in ('source', 'runtime', 'clocks_ms'):
        if proof[key] != plan[key]:
            raise ValueError('complete response case identity differs')
    if proof.get('isolated') is not True:
        raise ValueError('authoritative response must be isolated')
    c.verify(proof['game'])
    closed = c.immutable(out / 'QUEUE_COMPLETION.json', dict(passed=True, job=previous['job'],
        queue_result=status['result'], proof=c.record(out / 'RESULT.json'), game=proof['game'],
        exposure=c.record(out / 'EXPOSURE_PENDING.json'), source=proof['source'],
        all_spent_case_artifacts_retained=True))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        if current['active_jobs'] != [previous]:
            raise PermissionError('active case changed before archive')
        current[PREFIX + 'completed'].append(closed)
        exposure = c.record(out / 'EXPOSURE_PENDING.json')
        if exposure not in current['pending_exposures']:
            current['pending_exposures'].append(exposure)
        current['active_jobs'] = []
        c.atomic(base / 'CURRENT.json', current)
    return closed


def admit_next(base, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    if state.get('active_jobs') or state.get('active_claims'):
        raise PermissionError('archive current case before admission')
    following = next_case(state[PREFIX + 'cases'], state[PREFIX + 'completed'])
    if following is None:
        return None
    plan = c.read(c.verify(following['plan']))
    queue_root = Path(state['queue_root'])
    if (queue_root / 'jobs' / following['id']).exists():
        raise PermissionError('a spent or admitted case must not be replayed')
    if any(row['status'] in ('pending', 'running') for row in queue.queue.status(queue_root)['jobs']):
        raise PermissionError('another queue workload is admitted')
    c.guard(base, 'enqueue', state['owner_thread_id'], usage,
            source=plan['source']['sha256'], budget=plan['resource_budget'])
    queue.queue.enqueue(queue_root, c.read(c.verify(following['job'])))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        current['active_jobs'] = [following]
        c.atomic(base / 'CURRENT.json', current)
    return following
