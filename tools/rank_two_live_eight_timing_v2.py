"""Prebuild frozen sources, then certify one isolated production response game."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import unittest

from tools import rank_two_live_eight_v1 as campaign
from tools import rank_two_focused_native_v5 as native
from tools import rank_two_focused_coverage_v1 as coverage
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + '.production-safety.v2'


def checked(plan_path, launch=True):
    plan = campaign.read(plan_path)
    state = work.checked(plan, 'experiment', launch=launch)
    if plan['producer'] != campaign.record(__file__) or plan['clocks_ms'] not in campaign.PLAN['clock_ladder']:
        raise ValueError('declared production source/clock policy differs')
    for reference in plan['inputs'].values():campaign.verify(reference)
    campaign.verify(plan['source']);campaign.verify(plan['runtime'])
    export=campaign.read(campaign.verify(plan['export']))
    if export['source_sha256']!=plan['source']['sha256'] or export['clocks_ms']!=plan['clocks_ms']:
        raise ValueError('export source/clock binding differs')
    campaign.resource_guard(plan['resource_budget'])
    campaign.banned(state,campaign.read(campaign.verify(state['closures'])),plan['source']['sha256'])
    return plan,state


def build(plan_path):
    plan,_ = checked(plan_path)
    out = Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.codingame.test_rank_two_live_eight_timing_v2')
    checked_tests=unittest.TextTestRunner(verbosity=1).run(suite)
    campaign.immutable(out/'UNIT_TESTS.json',dict(passed=checked_tests.wasSuccessful(),tests=checked_tests.testsRun))
    if not checked_tests.wasSuccessful():raise ValueError('isolated timing guard tests failed')
    source = campaign.verify(plan['source']);binary = out/'production-main'
    command = ['/usr/bin/clang++',*native.compiler_flags(),str(source),'-o',str(binary)]
    process = subprocess.run(command,text=True,capture_output=True,timeout=120)
    compiled = campaign.immutable(out/'COMPILE.json',dict(command=command,
        returncode=process.returncode,stderr=process.stderr,source=plan['source']))
    if process.returncode:raise ValueError('frozen production source compile failed')
    result = dict(schema=SCHEMA+'.build',passed=True,source=plan['source'],runtime=plan['runtime'],
        clocks_ms=plan['clocks_ms'],binary=campaign.record(binary),compiler=campaign.record('/usr/bin/clang++'),
        compile=compiled,flags=native.compiler_flags(),source_modified=False,new_games=0)
    campaign.immutable(out/'BUILD.json',result)
    return result


def isolated(state,plan):
    if any(row['id']!=plan['job_id'] for row in state.get('active_jobs',[])):
        raise PermissionError('another admitted workload is active')
    if state.get('active_helper') or state.get('active_claims'):
        raise PermissionError('helper or browser claim is active')
    if any(not row.get('all90_archived') for row in state.get('live_eight_known_windows',{}).values()):
        raise PermissionError('archive live windows before authoritative certification')


def certify_case(plan_path):
    plan,state = checked(plan_path)
    isolated(state,plan)
    saved = campaign.read(campaign.verify(plan['build']))
    if (not saved.get('passed') or saved['source']!=plan['source']
            or saved['runtime']!=plan['runtime'] or saved['clocks_ms']!=plan['clocks_ms']
            or saved['source_modified'] is not False
            or saved['compiler']!=campaign.record('/usr/bin/clang++')):
        raise ValueError('exact prebuilt production source/compiler differs')
    campaign.verify(saved['binary'])
    color,offset = plan['case']['color'],plan['case']['direction_offset']
    if color not in (0,1) or offset not in (0,3):raise ValueError('predeclared case differs')
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    original=inspect.getsource(coverage.game);anchor='limit = 550 if first else 140'
    if original.count(anchor)!=1:raise ValueError('frozen response-envelope anchor differs')
    adapted=original.replace(anchor,f"limit = {plan['clocks_ms'][0]} if first else {plan['clocks_ms'][1]}")
    namespace=dict(coverage.__dict__);exec(compile(adapted,'<declared-response-envelope>','exec'),namespace)
    adapter=campaign.immutable(out/'GAME_ADAPTER.py',adapted.encode())
    # No compilation, training, or helper work runs in this timing phase.
    work.checked(plan,'experiment');isolated(campaign.read(plan['campaign_current']),plan)
    result=namespace['game'](saved,color,offset,out/'game')
    game=campaign.record(out/'game/RESULT.json')
    proof=dict(schema=SCHEMA+'.case',passed=result['complete'] and result['failure'] is None,
        source=plan['source'],runtime=plan['runtime'],build=plan['build'],clocks_ms=plan['clocks_ms'],
        color=color,direction_offset=offset,game=game,adapter=adapter,
        own_failures=int(result['failure'] is not None),failure=result['failure'],isolated=True,
        production_main_unchanged=True,first_max_ms=result['first_max_ms'],
        later_max_ms=result['later_max_ms'],longest_response_edges=result['longest_response_edges'],
        playing_strength_qualified=False)
    campaign.immutable(out/'RESULT.json',proof)
    campaign.immutable(out/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',
        game_receipts=[game],report=campaign.record(out/'RESULT.json'),training_eligible=False,
        before_next_fresh_bank=True))
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--mode',choices=('build','case'),required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.plan) if args.mode=='build' else certify_case(args.plan)))
