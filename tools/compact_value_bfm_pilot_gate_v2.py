#!/usr/bin/env python3
"""Fresh, source-bound 200-game historical Rank4 screen for a selected v2 model."""
from __future__ import annotations

import argparse
import fcntl
import select
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_stream_v2 as stream
from tools import compact_value_bfm_pilot_selection_v2 as model_selection
from submissions.codingame.bots.compact_value_bfm import rank4_gate_support


def prepare_bank(context,phase,selection):
    if selection.get('status')!='model-selected-before-rank4-screen' or not selection.get('selected'):
        raise ValueError('model selection must be frozen before any screen bank')
    if selection!=campaign.read(context/phase/'model-selection.json'):
        raise ValueError('screen model selection differs from its sealed file')
    campaign.verify(selection['selected']['source']);campaign.verify(selection['selected']['runtime'])
    directory=context/phase/'rank4-screen';directory.mkdir(parents=True,exist_ok=True)
    reference=directory/'bank.json'
    if reference.exists():
        bank=campaign.read(reference);campaign.verify(bank['tsv']);return bank
    positions_path=context/phase/'positions.json';games_path=context/phase/'games.json'
    contract=campaign.read(context/'campaign.json');positions=campaign.read(positions_path)
    inputs={'selection':campaign.record(context/phase/'model-selection.json'),'positions':campaign.record(positions_path),
        'games':campaign.record(games_path),'context':campaign.record(context/'campaign.json')}
    claim=campaign.seal(directory/'seed-claim.json',{'schema':campaign.ID+'.pilot-screen-seed-claim.v2',
        'inputs':inputs,'purpose':'post-model-selection-historical-rank4-200','pairs':100})
    seed=hashlib.sha256(campaign.raw({'claim':claim['body_sha256'],'purpose':'fresh-unprotected-root-pairs'})).digest()
    excluded=campaign.exclusion_sets(contract);candidates=[];seen=set();attempt=0
    # Oversampling is decided before reading campaign states, never from game results.
    while len(candidates)<1024:
        generated=campaign.openings.generate_candidate(hashlib.sha256(seed+attempt.to_bytes(16,'big')).digest());attempt+=1
        if attempt>100000:raise ValueError('screen root proposal limit reached')
        if generated is None:continue
        state,transcript,plies=generated;fps=campaign.fingerprints(state)
        if fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN] in seen or campaign.rejection(state,'validation',excluded):continue
        seen.add(fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN]);candidates.append({'transcript':transcript,'plies':plies,'fingerprints':fps})
    target={domain:{row['fingerprints'][domain] for row in candidates} for domain in
        (campaign.legacy.STATE_FINGERPRINT_DOMAIN,campaign.legacy.FEATURE_FINGERPRINT_DOMAIN)}
    collided={domain:set() for domain in target}
    for record in positions['census_files']:
        for row in stream.read_gzip(campaign.verify(record)):
            for member in row['closure']:
                for domain,value in member.items():
                    if value in target[domain]:collided[domain].add(value)
    games=campaign.read(games_path)
    for row in games['rows']:
        state=campaign.features.ReplayState()
        for turn,action in enumerate(row['game']['transcript'].split('/')):
            if turn>=row['game']['prefix_turns']:
                for domain,value in campaign.fingerprints(state).items():
                    if value in target[domain]:collided[domain].add(value)
            campaign.features.apply_complete_turn(state,state.to_move,action)
    retained=[row for row in candidates if all(value not in collided[domain] for domain,value in row['fingerprints'].items())][:100]
    if len(retained)!=100:raise ValueError('not enough isolated roots in the frozen proposal pool')
    for index,row in enumerate(retained):row['opening_id']=f'pilot-screen-{index:03d}'
    tsv=directory/'bank.tsv';campaign.once(tsv,('opening_id\ttranscript\n'+''.join(row['opening_id']+'\t'+row['transcript']+'\n' for row in retained)).encode())
    rank4_gate_support.validate_bank(tsv)
    exclusions=[]
    for index,domain in enumerate(target):
        path=directory/f'exclusion-{index}.json';campaign.seal(path,{'schema':campaign.ID+'.pilot-screen-exclusion.v2',
            'role':'mixed-development','domain':domain,'fingerprints':sorted(row['fingerprints'][domain] for row in retained),
            'contains_transcripts':False,'contains_labels':False,'contains_metrics':False,'bank_sha256':campaign.sha(tsv)})
        exclusions.append(campaign.record(path))
    return campaign.seal(reference,{'schema':campaign.ID+'.pilot-screen-bank.v2','claim':campaign.record(directory/'seed-claim.json'),
        'tsv':campaign.record(tsv),'rows':retained,'exclusions':exclusions,'pairs':100,
        'proposed_roots':1024,'filtered_from_frozen_training_states':True})


def played_exclusions(context,phase,execution,checked,bank):
    """Carry every actually played screen boundary, without labels or scores."""
    directory=context/phase/'rank4-screen';values=defaultdict(set)
    if checked['config'].get('trajectory_schema')!='papersoccer.compact-value-bfm-rank4-trajectories.v1':
        raise ValueError('pilot screen must retain source-bound played trajectories')
    for game in checked['games']:
        state=campaign.features.ReplayState();prefix_turns=len(game['root_transcript'].split('/'))
        for turn,action in enumerate(game['transcript'].split('/')):
            if turn>=prefix_turns:
                for domain,value in campaign.fingerprints(state).items():values[domain].add(value)
            campaign.features.apply_complete_turn(state,state.to_move,action)
        for domain,value in campaign.fingerprints(state).items():values[domain].add(value)
    result=list(bank['exclusions'])
    for ordinal,(domain,fingerprints) in enumerate(sorted(values.items())):
        path=directory/f'played-exclusion-{ordinal}.json'
        campaign.seal(path,{'schema':campaign.ID+'.pilot-screen-played-exclusions.v2',
            'role':'mixed-development','domain':domain,'fingerprints':sorted(fingerprints),
            'execution':campaign.record(execution),'bank_sha256':bank['tsv']['sha256'],
            'contains_transcripts':False,'contains_labels':False,'contains_metrics':False,
            'includes_all_played_postroot_boundaries':True,'includes_terminal_features':True})
        result.append(campaign.record(path))
    return result


def run_screen(root,context,phase):
    selection_path=context/phase/'model-selection.json';selection=campaign.read(selection_path)
    if selection.get('pilot_admitted') is not False:raise ValueError('unexpected pre-screen admission state')
    if selection.get('selected') is None:
        return campaign.seal(context/phase/'pilot-outcome.json',{'schema':campaign.ID+'.pilot-outcome.v2',
            'selection':campaign.record(selection_path),'status':'offline-rejected','admitted':False,'campaign_success':False})
    policy=campaign.read(campaign.verify(selection['policy']))
    if {k:v for k,v in policy.items() if k!='body_sha256'}!=model_selection.SELECTION_POLICY:
        raise ValueError('pilot selection policy changed')
    selected=selection['selected']
    control=next(arm for arm in selection['arms'] if arm['lambda']==0)
    comparison=model_selection.compare_candidate(control,selected)
    if not comparison['eligible_for_rank4_screen']:raise ValueError('selected model fails frozen offline criteria')
    training=campaign.read(campaign.verify(selection['training']))
    matching=[row for row in training['results'] if row['weight']==selected['lambda'] and row['seed']==selected['seed']]
    if len(matching)!=1 or matching[0]['source']!=selected['source'] or matching[0]['runtime']!=selected['runtime']:
        raise ValueError('selected model lost its trained source lineage')
    candidate=campaign.verify(selected['source']);runtime=campaign.verify(selected['runtime'])
    contract=campaign.read(context/'campaign.json');rank4=campaign.verify(contract['opponents']['rank_4']['submission.cpp'])
    if campaign.sha(rank4)!=rank4_gate_support.RANK4_SHA256:raise ValueError('historical Rank4 changed')
    bank=prepare_bank(context,phase,selection);directory=context/phase/'rank4-screen'
    execution=directory/'execution.json'
    if execution.exists():return campaign.read(context/phase/'pilot-outcome.json')
    claim=directory/'execution-claim.json';raw=directory/'result.json'
    if claim.exists() or raw.exists():raise ValueError('claimed screen is spent; seal its interruption and use a fresh isolated attempt')
    compiler=Path('/opt/homebrew/bin/g++-15').resolve()
    if campaign.sha(compiler)!=contract['compiler']['sha256']:raise ValueError('compiler differs from frozen identity')
    source=campaign.REPO/'submissions/codingame/bots/compact_value_bfm/rank4_gate_trajectories.cpp'
    compiled_rank4=source.parent.parent/'rank_4/submission.cpp'
    if campaign.sha(compiled_rank4)!=campaign.sha(rank4):raise ValueError('compiled Rank4 source differs from frozen opponent')
    binary=directory/'gate.bin';command=[str(compiler),'-std=c++20','-O3',f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{candidate}"',str(source),'-o',str(binary)]
    with (directory/'build.log').open('wb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    gate_command=[str(binary),'--bank',bank['tsv']['path'],'--candidate-source',str(candidate),'--rank4-source',str(rank4),
        '--output',str(raw),'--expected-bank-sha256',bank['tsv']['sha256'],'--expected-candidate-sha256',campaign.sha(candidate),
        '--pair-offset','0','--pair-count','100','--mode','actual-clock','--minimum-candidate-wins','105',
        '--include-trajectories']
    if os.getpriority(os.PRIO_PROCESS,0)!=0:raise ValueError('actual-clock screen requires nice zero')
    campaign.seal(claim,{'schema':campaign.ID+'.pilot-screen-execution-claim.v2','candidate':selected['source'],
        'runtime':selected['runtime'],'bank':campaign.record(directory/'bank.json'),'binary':campaign.record(binary),
        'gate_source':campaign.record(source),'compiler':contract['compiler'],'compile_command':command,'command':gate_command,
        'workers':1,'retry_allowed':False})
    started=time.monotonic()
    with (directory/'stdout.log').open('wb') as out,(directory/'stderr.log').open('wb') as err:
        finished=subprocess.run(gate_command,stdout=out,stderr=err,env={**os.environ,**campaign.THREADS})
    if finished.returncode not in (0,2) or not raw.exists():raise ValueError('screen process failed without a complete valid result')
    checked=rank4_gate_support.validate_result(raw,expected_bank_sha256=bank['tsv']['sha256'],
        expected_candidate_sha256=campaign.sha(candidate),expected_candidate_search_profile='standard-v1',
        require_trajectories=True,trajectory_bank=campaign.verify(bank['tsv']))
    runtime_document=campaign.read(runtime)
    if (checked['bindings']['candidate_runtime_body_sha256']!=runtime_document['body_sha256']
            or checked['bindings']['candidate_payload_sha256']!=runtime_document['quantization']['payload_sha256']):
        raise ValueError('screen compiled a different model payload')
    if [(game['opening_id'],game['candidate_player']) for game in checked['games']]!=[(row['opening_id'],color) for row in bank['rows'] for color in (0,1)]:
        raise ValueError('screen root/color schedule changed')
    result=checked['result'];admitted=result['games']==200 and result['candidate_wins']>=105 and result['failures']==0
    campaign.seal(execution,{'schema':campaign.ID+'.pilot-screen-execution.v2','claim':campaign.record(claim),
        'raw':campaign.record(raw),'elapsed_seconds':time.monotonic()-started,'returncode':finished.returncode,'result':result})
    exclusions=played_exclusions(context,phase,execution,checked,bank)
    outcome=campaign.seal(context/phase/'pilot-outcome.json',{'schema':campaign.ID+'.pilot-outcome.v2',
        'selection':campaign.record(selection_path),'screen':campaign.record(execution),'selected':selected,
        'status':'pilot-admitted' if admitted else 'rank4-screen-rejected','admitted':admitted,
        'wins':result['candidate_wins'],'games':result['games'],'failures':result['failures'],
        'development_exclusions':exclusions,'played_trajectory_closure_preserved':True,'campaign_success':False})
    return outcome



def wait_for_selection(path,upstream_pid,phase,expected_script="compact_value_bfm_pilot_selection_v2.py"):
    """One-shot dependency wait; a failed upstream process cannot hang forever."""
    while not path.exists():
        process=subprocess.run(['ps','-p',str(upstream_pid),'-o','args='],capture_output=True,text=True)
        if process.returncode or expected_script not in process.stdout or phase not in process.stdout:
            raise ValueError('upstream training ended without a completed model selection')
        if hasattr(select,'kqueue'):
            descriptor=os.open(path.parent,os.O_RDONLY)
            try:
                queue=select.kqueue()
                try:
                    event=select.kevent(descriptor,filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD|select.KQ_EV_ENABLE|select.KQ_EV_ONESHOT,
                        fflags=select.KQ_NOTE_WRITE|select.KQ_NOTE_RENAME)
                    if not path.exists():queue.control([event],1,60)
                finally:queue.close()
            finally:os.close(descriptor)
        else:
            time.sleep(10)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--wait-for-selection',type=int,metavar='UPSTREAM_PID')
    args=parser.parse_args();root=args.root.resolve();context=args.context.resolve()
    contract=campaign.read(context/'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent!=root:raise ValueError('phase parent changed')
    if args.wait_for_selection:
        wait_for_selection(context/args.phase/'model-selection.json',args.wait_for_selection,args.phase)
    with (root/'.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if args.wait_for_selection else fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run_screen(root,context,args.phase)
        campaign.event(root,'pilot-outcome-recorded',{'outcome':campaign.record(context/args.phase/'pilot-outcome.json'),'status':result['status']})
    print(json.dumps({'status':result['status'],'admitted':result['admitted']}),flush=True)

if __name__=='__main__':main()
