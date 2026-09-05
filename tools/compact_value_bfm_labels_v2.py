#!/usr/bin/env python3
"""Stream real native teacher labels and preserve their complete rich evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__=='__main__':
    for key in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_stream_v2 as stream
from tools import compact_value_bfm_train as trainer

_EXECUTION_BYTES=Path(__file__).read_bytes()
_PLAN=None
_POSITIONS=None
_BUNDLE=None
_TEACHER=None
RANK4_PATHS=(
    'include/papersoccer/types.hpp','include/papersoccer/geometry.hpp','include/papersoccer/rules.hpp',
    'src/bots/mcts_internal.hpp','src/core/geometry.cpp','src/core/rules.cpp',
    'submissions/codingame/bots/rank_4/replay_book.hpp','submissions/codingame/bots/rank_4/replay_value_model.hpp',
    'submissions/codingame/bots/rank_4/teacher_residual_model.hpp','submissions/codingame/bots/rank_4/bot.cpp',
    'tools/jacek_replay_rank4_position_teacher.cpp')
ACTION_PATHS=(
    'include/papersoccer/bot.hpp','include/papersoccer/types.hpp','include/papersoccer/geometry.hpp',
    'include/papersoccer/rules.hpp','src/bots/bot.cpp','src/bots/mcts_internal.hpp',
    'src/bots/jacek_replay_bfm/features.cpp','src/bots/jacek_replay_bfm/model.cpp',
    'src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp','src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp',
    'src/core/geometry.cpp','src/core/rules.cpp','tools/jacek_replay_bfm_search_teacher.cpp',
    'tools/jacek_replay_bfm_search_teacher_internal.hpp')


def native_source_closures(root,contract):
    sources=campaign.read(root/'native-source-closure.json')['sources']
    def record(relative):
        if relative.startswith('tools/'): return contract['sources'][relative]
        if relative.startswith('submissions/codingame/bots/rank_4/'):
            return contract['opponents']['rank_4'][Path(relative).name]
        return sources[relative]
    result={}
    for name,paths in (('rank4',RANK4_PATHS),('action',ACTION_PATHS)):
        records=[record(relative) for relative in paths]
        for item in records: campaign.verify(item)
        # CMake hashes its declared order, not a sorted set or the standalone bot.
        material=''.join(f'{relative}:{item["sha256"]}\n' for relative,item in zip(paths,records))
        result[name]={'sha256':hashlib.sha256(material.encode()).hexdigest(),
            'ordered_paths':list(paths),'records':records}
    return result


def initialize_worker(plan_path):
    global _PLAN,_POSITIONS,_BUNDLE,_TEACHER
    _PLAN=campaign.read(Path(plan_path));_POSITIONS=campaign.read(campaign.verify(_PLAN['positions']))
    _BUNDLE=trainer.FrozenBundle.load(campaign.verify(_PLAN['bundle']));_TEACHER=None
    for record in (_PLAN['action_binary'],_PLAN['rank4_binary'],_PLAN['teacher']): campaign.verify(record)


def lineage_matches(source,position,mover):
    prefix=[{'player_id':index%2,'action':action} for index,action in enumerate(position['prefix'].split('/') if position['prefix'] else [])]
    return (all(source.get(key)==position[key] for key in ('root_group_id','group_id','source','split','winner'))
        and source.get('prefix')==prefix and mover==position['mover'])


def validate_row(row,position,census,kind,nodes):
    global _TEACHER
    if kind=='action':
        normalized=campaign.corpus.validate_complete_turn_action_group(row)
        group=normalized['group'];source=group['source_binding'];teacher=normalized['teacher']
        if (source['position_id']!=position['position_id'] or source.get('campaign_id')!=campaign.ID
                or normalized['source_bundle_body_sha256']!=_BUNDLE.body_sha256
                or teacher['artifact_sha256']!=_PLAN['teacher']['sha256']
                or teacher['source_sha256']!=_PLAN['native_source_closures']['action']['sha256']
                or group['work_budget']['max_tree_nodes']!=nodes
                or group['work_budget'].get('teacher_ranking_profile','standard-v1')!='standard-v1'
                or group['parent_identity']!=position['parent_identity']
                or not lineage_matches(source,position,group['parent_mover'])):
            raise ValueError('action teacher/source/position binding changed')
        if _TEACHER is None:
            core={key:teacher[key] for key in ('artifact_sha256','payload_sha256','feature_schema_sha256')}
            if trainer._validate_teacher_identity(_BUNDLE,core)!=core: raise ValueError('accepted teacher changed')
            _TEACHER=dict(teacher)
        elif teacher!=_TEACHER: raise ValueError('mixed accepted teachers')
        actual=[item['successor_id'] for item in group['successors']]
        expected=census['successor_identities']
        if group['successors_exhaustive']:
            if actual!=expected: raise ValueError('native exhaustive successor census differs')
        elif nodes==500000 or not set(actual)<=set(expected):
            raise ValueError('deep group is incomplete or shallow group has a foreign successor')
        return normalized
    campaign.corpus.sample_from_teacher_row(row)
    if (row.get('position_id')!=position['position_id']
            or row.get('teacher',{}).get('source_sha256')!=_PLAN['native_source_closures']['rank4']['sha256']
            or row.get('search_config',{}).get('max_nodes')!=nodes
            or row.get('search_config',{}).get('max_time_ms')!=0
            or not lineage_matches(row,position,row.get('mover'))):
        raise ValueError('Rank4 teacher/source/position binding changed')
    return row


def label_job(job):
    kind,nodes,ordinal,positions,directory=job;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    inputs={'plan_body_sha256':_PLAN['body_sha256'],'kind':kind,'nodes':nodes,'ordinal':ordinal,'positions':positions}
    receipt=directory/'receipt.json'
    if receipt.exists():
        stored=campaign.read(receipt)
        if stored['inputs']!=inputs: raise ValueError('native label resume changed')
        campaign.verify(stored['output']);return campaign.record(receipt)
    census={};needed={row['position_id'] for row in positions}
    if kind=='action':
        for index in sorted({row['census_file'] for row in positions}):
            path=campaign.verify(_POSITIONS['census_files'][index])
            for row in stream.read_gzip(path):
                if row['position_id'] in needed: census[row['position_id']]=row
        if set(census)!=needed: raise ValueError('preflight census coverage changed')
    payload=(campaign.HEADER+''.join('\t'.join(str(row[key]) for key in
        ('position_id','root_group_id','group_id','source','split','winner','mover','prefix'))+'\n' for row in positions)).encode()
    stdin=directory/'positions.tsv';campaign.once(stdin,payload)
    if kind=='action':
        command=[_PLAN['action_binary']['path'],'--model',_PLAN['teacher']['path'],
            '--model-sha256',_PLAN['teacher']['sha256'],'--campaign-id',campaign.ID,
            '--tree-nodes',str(nodes),'--time-ms','0','--max-actions','250','--max-partial-paths','50000',
            '--exploration','0.5','--fpu','0.5','--emit-action-groups','--teacher-ranking-profile','standard-v1',
            '--source-bundle-body-sha256',_BUNDLE.body_sha256]
    else:
        command=[_PLAN['rank4_binary']['path'],'--campaign-id',campaign.ID,'--nodes',str(nodes),'--time-ms','0']
    raw=directory/'native-output.partial';stderr=directory/'stderr.log';started=time.monotonic()
    with raw.open('wb') as out,stderr.open('wb') as err:
        result=subprocess.run(command,input=payload,stdout=out,stderr=err,env={**os.environ,**campaign.THREADS})
    native_seconds=time.monotonic()-started
    if result.returncode: raise RuntimeError(f'native label command failed: {stderr}')
    native_sha=campaign.sha(raw);observed=[];scores={}
    def validated():
        with raw.open('rb') as source:
            for number,line in enumerate(source):
                if number>=len(positions): raise ValueError('native teacher emitted extra rows')
                position=positions[number]
                row=validate_row(json.loads(line),position,census.get(position['position_id']),kind,nodes)
                observed.append(position['position_id'])
                if kind=='rank4': scores[position['position_id']]=campaign.legacy._rank4_value(row)
                yield row
        if observed!=[row['position_id'] for row in positions]: raise ValueError('native teacher omitted positions')
    output=directory/'labels.jsonl.gz';stream.write_gzip(output,validated())
    campaign.seal(receipt,{'schema':campaign.ID+'.stream-label-chunk.v2','inputs':inputs,
        'stdin':campaign.record(stdin),'command':command,'output':campaign.record(output),
        'native_stdout_sha256':native_sha,'stderr':campaign.record(stderr),'rank4_scores':scores,
        'native_seconds':native_seconds,'total_seconds':time.monotonic()-started,
        'maintained_rich_validator_passed':True,'positions':len(positions),'returncode':0})
    raw.unlink()  # Canonical, lossless compressed evidence is now sealed.
    return campaign.record(receipt)


def chunks(plan_path,positions,kind,nodes,directory,workers):
    jobs=[(kind,nodes,i//32,positions[i:i+32],str(directory/f'{i//32:05d}')) for i in range(0,len(positions),32)]
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(str(plan_path),)) as pool:
        for count,record in enumerate(pool.map(label_job,jobs),1):
            results.append(record)
            if count%25==0 or count==len(jobs):
                print(json.dumps({'stage':'labels','kind':kind,'nodes':nodes,'completed_chunks':count,'chunks':len(jobs)}),flush=True)
    return results


def rows_from_receipts(records):
    for record in records:
        receipt=campaign.read(campaign.verify(record))
        yield from stream.read_gzip(campaign.verify(receipt['output']))


def prepare_plan(root,context,phase,workers):
    contract=campaign.read(context/'campaign.json')
    if contract['policy']!=campaign.POLICY: raise ValueError('approved policy changed')
    path=context/phase/'labels-plan.json';source=path.parent/'labels'/(hashlib.sha256(_EXECUTION_BYTES).hexdigest()+'.producer.py')
    campaign.once(source,_EXECUTION_BYTES)
    campaign.seal(path,{'schema':campaign.ID+'.stream-label-plan.v2','context':campaign.record(context/'campaign.json'),
        'positions':campaign.record(context/phase/'positions.json'),'phase':phase,'workers':workers,
        'bundle':contract['bundle'],'teacher':contract['inputs']['teacher_runtime'],
        'action_binary':contract['binaries']['search_teacher'],'rank4_binary':contract['binaries']['rank4_position_teacher'],
        'student':contract['inputs']['attempt_zero_runtime'],'native_source_closures':native_source_closures(root,contract),
        'producer':campaign.record(source),'shallow_nodes':64000,'deep_nodes':500000,'deep_fraction':.25})
    return path


def run_labels(root,context,phase,workers=8):
    if not 1<=workers<=8: raise ValueError('label workers must be in 1..8')
    output=context/phase/'labels.json'
    if output.exists():
        done=campaign.read(output);campaign.verify(done['merged']);return done
    plan_path=prepare_plan(root,context,phase,workers);plan=campaign.read(plan_path)
    positions=campaign.read(campaign.verify(plan['positions']))['rows'];directory=context/phase/'labels';started=time.monotonic()
    shallow=chunks(plan_path,positions,'action',64000,directory/'shallow',workers)
    rank4=chunks(plan_path,positions,'rank4',32000,directory/'rank4',workers)
    rank4_scores={}
    for record in rank4: rank4_scores.update(campaign.read(campaign.verify(record))['rank4_scores'])
    predictor=campaign.batched_scalar_predictor(campaign.verify(plan['student']));evidence=[];nonexhaustive=0
    for row in rows_from_receipts(shallow):
        group=row['group'];source=group['source_binding'];position_id=source['position_id']
        regret=campaign.legacy.action_regret(group,predictor(group));search=float(group['root_value']);reference=rank4_scores[position_id]
        outcome=1 if source['winner']==group['parent_mover'] else -1
        key=(int(group['successors_exhaustive']),-regret['regret'],-int(regret['action_disagreement']),
            -int((search>=0)!=(reference>=0)),-abs(search-reference),-int((search>=0)!=(outcome>=0)),min(abs(search),abs(reference)),position_id)
        evidence.append({'position_id':position_id,'key':list(key),'regret':regret,'search_value':search,'rank4_value':reference,'terminal_outcome':outcome})
        nonexhaustive+=not group['successors_exhaustive']
    count=math.ceil(len(positions)/4)
    if nonexhaustive>count: raise ValueError('nonexhaustive shallow groups exceed the approved deep quarter')
    ids={row['position_id'] for row in sorted(evidence,key=lambda row:row['key'])[:count]}
    selection=context/phase/'deep-selection.json';campaign.seal(selection,{'schema':campaign.ID+'.stream-deep-selection.v2',
        'position_ids':sorted(ids),'evidence':evidence,'fraction':.25,'rounding':'ceil','nonexhaustive_mandatory':nonexhaustive,
        'shallow_receipts':shallow,'rank4_receipts':rank4,'student':plan['student']})
    deep=chunks(plan_path,[row for row in positions if row['position_id'] in ids],'action',500000,directory/'deep',workers)
    def merged_rows():
        replacement=iter(rows_from_receipts(deep));next_deep=next(replacement,None);seen=set();used=0
        for row in rows_from_receipts(shallow):
            position_id=row['group']['source_binding']['position_id']
            if position_id in ids:
                if next_deep is None or next_deep['group']['source_binding']['position_id']!=position_id: raise ValueError('deep replacement order changed')
                for key in ('feature_schema','source_bundle_body_sha256','teacher','ranking','split'):
                    if row[key]!=next_deep[key]: raise ValueError('deep replacement identity changed')
                for key in ('group_id','parent_identity','identity_algorithm','parent_mover','source_binding'):
                    if row['group'][key]!=next_deep['group'][key]: raise ValueError('deep replacement parent changed')
                row=next_deep;next_deep=next(replacement,None);used+=1
            group=row['group']
            if not group['successors_exhaustive'] or group['group_id'] in seen: raise ValueError('final group incomplete or duplicated')
            seen.add(group['group_id']);yield row
        if next_deep is not None or used!=len(ids) or len(seen)!=len(positions): raise ValueError('final label coverage changed')
    merged=context/phase/'labels.merged.jsonl.gz';stream.write_gzip(merged,merged_rows())
    return campaign.seal(output,{'schema':campaign.ID+'.labels.v2','positions':plan['positions'],'groups':len(positions),
        'deep_groups':len(ids),'merged':campaign.record(merged),'compression':'gzip-canonical-jsonl-lossless',
        'teacher':plan['teacher'],'plan':campaign.record(plan_path),'shallow_receipts':shallow,'rank4_receipts':rank4,
        'deep_receipts':deep,'deep_selection':campaign.record(selection),'elapsed_seconds':time.monotonic()-started,
        'all_groups_exhaustive':True,'all_native_labels_validated':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();root=args.root.resolve();context=args.context.resolve()
    contract=campaign.read(context/'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent!=root: raise ValueError('phase parent changed')
    with campaign.lease(root):
        result=run_labels(root,context,args.phase,args.workers)
        campaign.event(root,'pilot-labels-completed',{'artifact':campaign.record(context/args.phase/'labels.json'),'groups':result['groups']})
    print(json.dumps({'groups':result['groups'],'deep_groups':result['deep_groups']}),flush=True)

if __name__=='__main__': main()
