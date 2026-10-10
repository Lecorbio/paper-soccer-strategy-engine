"""Source-bound clock-ladder certification with the retained legal-game generator."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import subprocess

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_coverage_v1 as coverage
from tools import rank_two_focused_work_v1 as work


def isolated(state,plan):
    if any(not w.get('all90_archived') for w in state.get('live_eight_known_windows',{}).values()):
        raise PermissionError('authoritative live window is active')
    if any(j['id']!=plan['job_id'] for j in state.get('active_jobs',[])):
        raise PermissionError('another fixed workload is active')
    if state.get('active_helper') or state.get('active_claims'):raise PermissionError('helper or browser claim is active')


def run(plan_path):
    plan=c.read(plan_path);state=work.checked(plan,'experiment',launch=True);isolated(state,plan)
    if plan['producer']!=c.record(__file__) or plan['clocks_ms'] not in c.PLAN['clock_ladder']:
        raise ValueError('declared timing source/budget differs')
    for ref in plan['inputs'].values():c.verify(ref)
    out=Path(plan['output']);source=c.verify(plan['source']);binary=out/'production-main';out.mkdir(parents=True,exist_ok=True)
    command=['/usr/bin/clang++','-std=c++20','-O3','-ffp-contract=off',str(source),'-o',str(binary)]
    process=subprocess.run(command,text=True,capture_output=True,timeout=120)
    c.immutable(out/'BUILD.json',dict(source=plan['source'],command=command,returncode=process.returncode,stderr=process.stderr))
    if process.returncode:raise ValueError('unchanged source compile failed')
    original=inspect.getsource(coverage.game)
    old='limit = 550 if first else 140'
    if original.count(old)!=1:raise ValueError('retained envelope check anchor differs')
    adapted=original.replace(old,f"limit = {plan['clocks_ms'][0]} if first else {plan['clocks_ms'][1]}")
    namespace=dict(coverage.__dict__);exec(compile(adapted,'<declared-envelope-adapter>','exec'),namespace)
    c.immutable(out/'GAME_ADAPTER.py',adapted.encode())
    entry=dict(source=plan['source'],runtime=plan['runtime'],binary=c.record(binary));cases=[]
    for color in (0,1):
        for offset in (0,3):
            work.checked(plan,'experiment');isolated(c.read(plan['campaign_current']),plan)
            result=namespace['game'](entry,color,offset,out/f'p{color}-offset{offset}')
            if not result['complete'] or result.get('failure'):raise ValueError('whole-response production safety failed')
            cases.append(dict(color=color,offset=offset,result=c.record(out/f'p{color}-offset{offset}/RESULT.json'),
                first_max_ms=result['first_max_ms'],later_max_ms=result['later_max_ms'],longest_response_edges=result['longest_response_edges']))
    passed=max(r['longest_response_edges'] for r in cases)>=8
    if not passed:raise ValueError('long rebound coverage missing')
    receipt=c.immutable(out/'RESULT.json',dict(passed=True,source=plan['source'],runtime=plan['runtime'],
        clocks_ms=plan['clocks_ms'],own_failures=0,isolated=True,production_main_unchanged=True,
        retained_generator=c.record(coverage.__file__),adapter=c.record(out/'GAME_ADAPTER.py'),cases=cases))
    c.immutable(out/'EXPOSURE.json',dict(cases=[r['result'] for r in cases],training_eligible=False))
    return dict(passed=True,receipt=receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.plan)))
