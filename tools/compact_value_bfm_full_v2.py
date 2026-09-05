#!/usr/bin/env python3
"""Scale an actually admitted pilot to fresh 10,000-game teacher training."""
from __future__ import annotations

import argparse
import fcntl
from collections import defaultdict
import os
from pathlib import Path
import sys

if __name__=='__main__':
    for key in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_stream_v2 as positions
from tools import compact_value_bfm_labels_v2 as labels
from tools import compact_value_bfm_pilot_gate_v2 as gate
from tools import compact_value_bfm_intervention_v2 as intervention
from submissions.codingame.bots.compact_value_bfm import rank4_gate_support


def admitted_pilot(context,phase):
    outcome_path=context/phase/'pilot-outcome.json';outcome=campaign.read(outcome_path)
    if (outcome.get('status')!='pilot-admitted' or outcome.get('admitted') is not True
            or outcome.get('games')!=200 or outcome.get('wins',0)<105 or outcome.get('failures')!=0):
        raise ValueError('full training requires an actually admitted pilot')
    selected=outcome['selected'];campaign.verify(selected['source']);campaign.verify(selected['runtime'])
    execution=campaign.read(campaign.verify(outcome['screen']));claim=campaign.read(campaign.verify(execution['claim']))
    bank=campaign.read(campaign.verify(claim['bank']));campaign.verify(bank['tsv'])
    raw=campaign.verify(execution['raw'])
    result=rank4_gate_support.validate_result(raw,expected_bank_sha256=bank['tsv']['sha256'],
        expected_candidate_sha256=selected['source']['sha256'],expected_candidate_search_profile='standard-v1',
        require_trajectories=True,trajectory_bank=campaign.verify(bank['tsv']))
    runtime=campaign.read(campaign.verify(selected['runtime']))
    if (result['bindings']['candidate_runtime_body_sha256']!=runtime['body_sha256']
            or result['bindings']['candidate_payload_sha256']!=runtime['quantization']['payload_sha256']):
        raise ValueError('pilot screen payload differs from admitted model')
    selection=campaign.read(campaign.verify(outcome['selection']))
    control=next(arm for arm in selection['arms'] if arm['lambda']==0)
    if selection['selected']!=selected or not gate.model_selection.compare_candidate(control,selected)['eligible_for_rank4_screen']:
        raise ValueError('pilot admission lost its offline eligibility')
    if (result['result']!=execution['result'] or result['result']['candidate_wins']!=outcome['wins']
            or result['result']['games']!=200 or result['result']['failures']!=0):
        raise ValueError('pilot admission does not reproduce its actual screen')
    expected_exclusions=gate.played_exclusions(context,phase,campaign.verify(outcome['screen']),result,bank)
    if outcome.get('played_trajectory_closure_preserved') is not True or outcome['development_exclusions']!=expected_exclusions:
        raise ValueError('pilot admission lost its played-state isolation closure')
    return outcome


def pilot_fingerprints(root,context,phase):
    index=context/phase/'full-carry-exclusions.json'
    if index.exists():
        result=campaign.read(index)
        for item in result['artifacts']:campaign.verify(item)
        return result['artifacts']
    manifest=campaign.read(context/phase/'positions.json');values=defaultdict(set)
    for item in manifest['census_files']:
        for row in positions.read_gzip(campaign.verify(item)):
            role='prior-train' if row['split']=='train' else 'prior-validation'
            for member in row['closure']:
                for domain,fingerprint in member.items():values[role,domain].add(fingerprint)
    games=campaign.read(context/phase/'games.json')
    for row in games['rows']:
        state=campaign.features.ReplayState();role='prior-train' if row['split']=='train' else 'prior-validation'
        for turn,action in enumerate(row['game']['transcript'].split('/')):
            if turn>=row['game']['prefix_turns']:
                for domain,value in campaign.fingerprints(state).items():values[role,domain].add(value)
            campaign.features.apply_complete_turn(state,state.to_move,action)
    artifacts=[]
    for ordinal,((role,domain),fingerprints) in enumerate(sorted(values.items())):
        path=context/phase/f'full-carry-{ordinal}.json'
        campaign.seal(path,{'schema':campaign.ID+'.pilot-carry-fingerprints.v2','role':role,'domain':domain,
            'fingerprints':sorted(fingerprints),'positions':campaign.record(context/phase/'positions.json'),
            'games':campaign.record(context/phase/'games.json'),'contains_transcripts':False,
            'contains_labels':False,'contains_metrics':False,'source_paths_followed_during_filtering':False})
        artifacts.append(campaign.record(path))
    campaign.seal(index,{'schema':campaign.ID+'.pilot-carry-index.v2','artifacts':artifacts,
        'new_teacher_labels_required':True,'early_training_exception_unchanged':True})
    return artifacts


def prepare(root,pilot_context,pilot_phase):
    pilot=admitted_pilot(pilot_context,pilot_phase)
    parent=campaign.read(pilot_context/'campaign.json');attempt=parent['attempt']
    profile=intervention.expected_qat_profile(parent)
    context=root/'phases'/f'attempt-{attempt:03d}-full';phase=f'attempt-{attempt:03d}-full'
    path=context/'campaign.json'
    if path.exists():
        frozen=campaign.read(path)
        if (intervention.expected_qat_profile(frozen)!=profile
                or any(frozen.get(key)!=parent.get(key) for key in
                    ('qat_profile','qat_profile_contract','intervention','training_executor'))):
            raise ValueError('full training changed its admitted pilot intervention')
        return context,phase,frozen
    selected=pilot['selected'];weight=selected['lambda']
    if weight not in (.1,.25):raise ValueError('pilot did not admit a ranking recipe')
    body={key:value for key,value in parent.items() if key!='body_sha256'}
    body.pop('pilot_games',None);body.pop('pilot_training_roster',None)
    body['inputs']=dict(parent['inputs']);body['inputs']['attempt_zero_runtime']=selected['runtime']
    body.update({'phase':'full','full_games':10000,'admitted_pilot':campaign.record(pilot_context/pilot_phase/'pilot-outcome.json'),
        'pilot_context':campaign.record(pilot_context/'campaign.json'),
        'exclusions':parent['exclusions']+pilot_fingerprints(root,pilot_context,pilot_phase)+pilot['development_exclusions'],
        'full_training_roster':{'lambdas':[0,weight],'seeds':[20260907,20260908,20260909]},
        'candidate_lineage':{'mandatory_training':True,'initial_float':parent['inputs']['attempt_one_initial_checkpoint'],
            'generation_student':selected['runtime'],'pilot_source':selected['source'],'smoke_weights_reused':False},
        'full_driver':campaign.copy_checked(Path(__file__),context/'full-driver.py')})
    for name in ('anchor-derived.json','prior-search-validation.json'):
        campaign.copy_checked(root/'exclusions'/name,context/'exclusions'/name)
    result=campaign.seal(path,body)
    campaign.event(root,'full-context-frozen',{'context':campaign.record(path),'admitted_pilot':body['admitted_pilot']})
    return context,phase,result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot-context',type=Path,required=True);parser.add_argument('--pilot-phase',required=True)
    parser.add_argument('--wait-for-pilot',type=int,metavar='UPSTREAM_PID')
    parser.add_argument('command',choices=('prepare','run'));args=parser.parse_args()
    root=args.root.resolve();pilot_context=args.pilot_context.resolve()
    if args.wait_for_pilot:
        gate.wait_for_selection(pilot_context/args.pilot_phase/'pilot-outcome.json',args.wait_for_pilot,args.pilot_phase,
            expected_script='compact_value_bfm_pilot_gate_v2.py')
    outcome=campaign.read(pilot_context/args.pilot_phase/'pilot-outcome.json')
    if not outcome.get('admitted'):
        campaign.event(root,'full-not-admitted',{'pilot_outcome':campaign.record(pilot_context/args.pilot_phase/'pilot-outcome.json')})
        print('full stage not admitted; a fresh isolated attempt is required',flush=True);return
    with (root/'.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if args.wait_for_pilot else fcntl.LOCK_EX|fcntl.LOCK_NB)
        context,phase,contract=prepare(root,pilot_context,args.pilot_phase)
        if args.command=='prepare':return
        campaign.run_games(context,phase,10000,8)
        campaign.event(root,'full-games-completed',{'games':campaign.record(context/phase/'games.json')})
        positions.run_positions(context,phase,8)
        campaign.event(root,'full-positions-completed',{'positions':campaign.record(context/phase/'positions.json')})
        labels.run_labels(root,context,phase,8)
        campaign.event(root,'full-labels-completed',{'labels':campaign.record(context/phase/'labels.json')})
        campaign.train_models(context,phase,smoke=False,ranking_weights=tuple(contract['full_training_roster']['lambdas']))
        campaign.event(root,'full-training-completed-awaiting-evaluation',{'training':campaign.record(context/phase/'training.json'),
            'qualification_passed':False,'campaign_success':False})
    print('full training complete; search, multi-opponent and final qualification remain',flush=True)

if __name__=='__main__':main()
