#!/usr/bin/env python3
"""Bounded-memory, resumable position production for the v2 campaign."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __name__=='__main__':
    for name in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
import numpy as np

_EXECUTION_BYTES=Path(__file__).read_bytes()
_EXCLUDED={}
_WORKER_PLAN=None


class FingerprintIndex:
    """Read-only binary search over full 256-bit hashes, shared through mmap."""
    def __init__(self,path):
        path=Path(path)
        if path.stat().st_size%32: raise ValueError('malformed fingerprint array')
        self.values=np.memmap(path,dtype='S32',mode='r') if path.stat().st_size else np.empty(0,dtype='S32')

    def __contains__(self,fingerprint):
        value=np.asarray(bytes.fromhex(fingerprint),dtype='S32')
        index=int(np.searchsorted(self.values,value))
        return index<len(self.values) and bool(self.values[index]==value)


def write_gzip(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as file:
        with gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0,compresslevel=1) as compressed:
            for row in rows: compressed.write(campaign.raw(row))
        file.flush();os.fsync(file.fileno())
    try:
        if path.exists():
            if campaign.sha(path)!=campaign.sha(temporary): raise ValueError('immutable compressed artifact changed')
        else: os.link(temporary,path)
    finally: temporary.unlink()


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)


def initialize_worker(plan_path,barrier_path):
    global _EXCLUDED,_WORKER_PLAN
    _WORKER_PLAN=campaign.read(Path(plan_path))
    contract=campaign.read(campaign.verify(_WORKER_PLAN['context']))
    if contract['policy']!=campaign.POLICY: raise ValueError('approved campaign policy changed')
    _EXCLUDED=campaign.exclusion_sets(contract)
    if barrier_path:
        barrier=campaign.read(Path(barrier_path))
        for domain,record in barrier['arrays'].items():
            _EXCLUDED['phase-validation',domain]=FingerprintIndex(campaign.verify(record))


def candidate_positions(item):
    state=campaign.features.ReplayState();prefix=[];candidates=[]
    game=item['game']
    for turn,action in enumerate(game['transcript'].split('/')):
        if turn>=game['prefix_turns'] and not campaign.rejection(state,item['split'],_EXCLUDED):
            candidates.append({'state':campaign.openings.clone_state(state),'turn':turn,
                'prefix':'/'.join(prefix),'edges':len(state.used_segments),
                'tactical':campaign.legacy._tactical_state_evidence(state)})
        campaign.features.apply_complete_turn(state,state.to_move,action);prefix.append(action)
    if state.winner!=game['winner']: raise ValueError('generated terminal winner changed')
    early=[p for p in candidates if p['edges']<=12]
    tactical=sorted(candidates,key=lambda p:(-p['tactical']['target_goal_edges'],
        -p['tactical']['rebound_edges'],-p['tactical']['constrained_edges'],p['turn']))
    late=[p for p in candidates if p['edges']>=80]
    decisive=list(reversed(candidates));chosen=[];used=set()
    for pool in (early,tactical,late,decisive):
        taken=0
        for position in pool:
            if position['turn'] in used: continue
            chosen.append(position);used.add(position['turn']);taken+=1
            if taken==5: break
    for position in candidates:
        if len(chosen)>=20: break
        if position['turn'] not in used: chosen.append(position);used.add(position['turn'])
    return chosen[:20]


def mine_game(job):
    item,directory,barrier_record=job;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    inputs={'plan_body_sha256':_WORKER_PLAN['body_sha256'],'game':item,'validation_barrier':barrier_record}
    receipt=directory/'receipt.json';output=directory/'positions.jsonl.gz'
    if receipt.exists():
        stored=campaign.read(receipt)
        if stored['inputs']!=inputs: raise ValueError('position chunk resume binding changed')
        campaign.verify(stored['output']);return campaign.record(receipt)
    campaign.verify(item['receipt']);started=time.monotonic();rows=[];rejected=Counter();seen=set()
    for position in candidate_positions(item):
        state=position['state'];fingerprints=campaign.fingerprints(state)
        parent=fingerprints[campaign.legacy.FEATURE_FINGERPRINT_DOMAIN]
        if parent in seen: rejected['same-game-duplicate-parent']+=1;continue
        checked=campaign.preflight_group(state,item['split'],_EXCLUDED)
        if not checked['eligible']: rejected[checked['reason']]+=1;continue
        seen.add(parent)
        rows.append({'position_id':f'{_WORKER_PLAN["phase"]}:{item["ordinal"]}:{position["turn"]}',
            'root_group_id':item['root_id'],'group_id':item['game']['game_id'],'source':campaign.ID,
            'split':item['split'],'winner':item['game']['winner'],'mover':state.to_move,
            'prefix':position['prefix'],'drawn_edges':position['edges'],
            'parent_identity':campaign.corpus._mover_canonical_position_identity(state),
            'parent_feature_fingerprint':parent,'canonical':fingerprints[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
            'legal_actions':checked['legal_actions'],'successor_identities':checked['successor_identities'],
            'closure':checked['closure'],'tactical':position['tactical']})
    write_gzip(output,rows)
    campaign.seal(receipt,{'schema':campaign.ID+'.position-chunk.v2','inputs':inputs,
        'output':campaign.record(output),'positions':len(rows),'rejections':dict(rejected),
        'elapsed_seconds':time.monotonic()-started,'all_retained_groups_preflighted':True})
    return campaign.record(receipt)


def build_barrier(root,phase,records):
    output=root/phase/'position-chunks/validation-barrier.json'
    if output.exists(): return campaign.read(output)
    values={domain:set() for domain in (campaign.legacy.FEATURE_FINGERPRINT_DOMAIN,campaign.legacy.STATE_FINGERPRINT_DOMAIN)}
    for record in records:
        receipt=campaign.read(campaign.verify(record))
        for row in read_gzip(campaign.verify(receipt['output'])):
            for member in row['closure']:
                for domain,value in member.items(): values[domain].add(value)
    arrays={}
    for ordinal,(domain,fingerprints) in enumerate(values.items()):
        payload=np.asarray(sorted(bytes.fromhex(value) for value in fingerprints),dtype='S32').tobytes()
        path=output.parent/f'validation-{ordinal}.bin';campaign.once(path,payload);arrays[domain]=campaign.record(path)
    return campaign.seal(output,{'schema':campaign.ID+'.validation-barrier.v2','arrays':arrays,
        'chunks':records,'contains_labels':False,'contains_metrics':False,'contains_transcripts':False,
        'all_validation_successors_included':True})


def run_positions(context,phase,workers=8):
    if not 1<=workers<=8: raise ValueError('position workers must be in 1..8')
    output=context/phase/'positions.json'
    if output.exists():
        result=campaign.read(output)
        for record in result['census_files']: campaign.verify(record)
        return result
    games_path=context/phase/'games.json';games=campaign.read(games_path)
    plan_path=context/phase/'positions-plan.json'
    source=context/phase/'position-chunks'/(hashlib.sha256(_EXECUTION_BYTES).hexdigest()+'.producer.py')
    campaign.once(source,_EXECUTION_BYTES)
    campaign.seal(plan_path,{'schema':campaign.ID+'.stream-positions-plan.v2','context':campaign.record(context/'campaign.json'),
        'games':campaign.record(games_path),'phase':phase,'workers':workers,'producer':campaign.record(source),
        'sample_limit':20,'opening_max_edges':12,'validation_first':True,'closure_limit':100000})
    records=[];barrier_path=None;barrier_record=None;started=time.monotonic()
    for split in ('validation','train'):
        items=sorted((row for row in games['rows'] if row['split']==split),key=lambda row:row['ordinal'])
        jobs=[(row,str(context/phase/'position-chunks'/split/f'{row["ordinal"]:05d}'),barrier_record) for row in items]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
                initargs=(str(plan_path),None if barrier_path is None else str(barrier_path))) as pool:
            for completed,record in enumerate(pool.map(mine_game,jobs),1):
                records.append(record)
                if completed%50==0 or completed==len(jobs):
                    print(json.dumps({'stage':'positions','split':split,'completed_games':completed,'total_games':len(jobs)}),flush=True)
        if split=='validation':
            build_barrier(context,phase,records)
            barrier_path=context/phase/'position-chunks/validation-barrier.json';barrier_record=campaign.record(barrier_path)
    rows=[];census_files=[];seen=set();duplicates=Counter();rejected=Counter()
    for record in records:
        receipt=campaign.read(campaign.verify(record));rejected.update(receipt['rejections'])
        census=receipt['output'];campaign.verify(census);ordinal=len(census_files);census_files.append(census)
        for row_number,row in enumerate(read_gzip(census['path'])):
            fingerprint=row['parent_feature_fingerprint']
            if fingerprint in seen: duplicates[row['split']]+=1;continue
            seen.add(fingerprint)
            rows.append({**{key:value for key,value in row.items() if key not in ('closure','legal_actions','successor_identities')},
                'census_file':ordinal,'census_row':row_number})
    result=campaign.seal(output,{'schema':campaign.ID+'.stream-positions.v2','rows':rows,'census_files':census_files,
        'chunk_receipts':records,'validation_barrier':barrier_record,'plan':campaign.record(plan_path),
        'counts':dict(Counter(row['split'] for row in rows)),'early_counts':dict(Counter(row['split'] for row in rows if row['drawn_edges']<=12)),
        'duplicate_parents_removed':dict(duplicates),'whole_group_rejections':dict(rejected),
        'elapsed_seconds':time.monotonic()-started,'all_retained_groups_preflighted':True})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('command',choices=('positions',))
    args=parser.parse_args();root=args.root.resolve();context=args.context.resolve()
    contract=campaign.read(context/'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent!=root: raise ValueError('phase parent changed')
    with campaign.lease(root):
        result=run_positions(context,args.phase,args.workers)
        campaign.event(root,'pilot-positions-completed',{'artifact':campaign.record(context/args.phase/'positions.json'),'counts':result['counts']})
    print(json.dumps({'command':args.command,'counts':result['counts'],'early_counts':result['early_counts']}),flush=True)

if __name__=='__main__': main()
