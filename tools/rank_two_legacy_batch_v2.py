#!/usr/bin/env python3
"""Finite single-hand H13 pilot batches using the existing queue in process."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SCHEMA='papersoccer.rank-two.finite-hand-pilot-batch.v2'


def record(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return {'path':str(path),'sha256':digest.hexdigest()}


def verify(ref):
    if record(ref['path'])!=ref:raise ValueError('changed input: '+ref['path'])
    return Path(ref['path'])


def read(path):return json.loads(Path(path).read_text())


def emit(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('immutable artifact differs: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(raw)
        path.chmod(0o444)


def merge_refs(*collections):
    result={}
    for collection in collections:
        for ref in collection:
            if ref['path'] in result and result[ref['path']]!=ref:raise ValueError('conflicting input identity')
            result[ref['path']]=ref
    return sorted(result.values(),key=lambda ref:ref['path'])


def flag(argv,name):
    if argv.count(name)!=1:raise ValueError('ambiguous argv flag: '+name)
    return argv[argv.index(name)+1]


def replace_flag(argv,name,value):
    result=list(argv);flag(result,name);result[result.index(name)+1]=str(value);return result


def nested_refs(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:yield {'path':value['path'],'sha256':value['sha256']}
        else:
            for item in value.values():yield from nested_refs(item)
    elif isinstance(value,list):
        for item in value:yield from nested_refs(item)


def prepare(queue_path,previous_job,runner_plan,output,start_job=254,start_chunk=1,chunks=8):
    queue_path=Path(queue_path).resolve();output=Path(output).resolve();runner_plan=Path(runner_plan).resolve()
    job_path=queue_path/'jobs'/previous_job/'job.json';previous=read(job_path)
    state=read(job_path.with_name('state.json'))
    if state['status'] not in ('complete','running') or (start_chunk==1 and state['status']!='complete'):raise ValueError('predecessor must be running or complete; completion required before advance')
    runner=read(runner_plan);strength_path=Path(runner['strength_output'])/'plan.json';strength=read(strength_path)
    ready_path=Path(runner['indexed_plan']['path']).parent/'ready.json';ready=read(ready_path)
    bundle=read(verify(strength['bundle']))
    if set(bundle['candidates'])!={'hand'} or ready['strength']['plan']!=record(strength_path) or ready['bank']!=strength['bank']:
        raise ValueError('exact single-hand prepared bank required')
    if not 1<=chunks<=8 or start_chunk<1 or start_chunk+chunks-1>32:raise ValueError('bounded1..8chunk pilot range required')
    if start_chunk==1:
        if read(previous['acceptance'][0]['path'])!=ready:raise ValueError('first batch needs completed bank preparation')
    elif flag(previous['argv'],'--chunk-id')!=f'pilot-{start_chunk-1:03d}':raise ValueError('wrong predecessor chunk')
    root=Path(previous['cwd']);output.mkdir(parents=True,exist_ok=True);frozen=output/'producer.py'
    raw=Path(__file__).read_bytes()
    if frozen.exists() and frozen.read_bytes()!=raw:raise ValueError('frozen driver changed')
    if not frozen.exists():frozen.write_bytes(raw);frozen.chmod(0o444)
    base=copy.deepcopy(previous)
    base.update(argv=[previous['argv'][0],runner['producer']['path'],'chunk','--plan',str(runner_plan),'--stage','pilot','--chunk-id',f'pilot-{start_chunk:03d}','--max-new-roots','2'],
        workload='authoritative-timing',matches=1,resume_policy='never',acceleration='disabled',
        limits={'max_workers':2,'coordinator_threads':1,'max_processes':3,'rss_bytes':2147483648,'wall_seconds':600,'cpu_capacity_seconds':2400,'termination_grace_seconds':2})
    base['inputs']=merge_refs(previous['inputs'],list(nested_refs(runner)),list(nested_refs(strength)),list(nested_refs(bundle)),
        list(nested_refs(ready)),[record(runner_plan),record(strength_path),record(ready_path),state['job']]+([state['result']] if state.get('result') else []))
    emit(output/'base-spec.json',base)
    steps=[{'job_id':f'{start_job+n}-h13-pilot-{start_chunk+n:03d}','chunk_id':f'pilot-{start_chunk+n:03d}',
        'previous_job':previous_job if n==0 else f'{start_job+n-1}-h13-pilot-{start_chunk+n-1:03d}'} for n in range(chunks)]
    plan={'schema':SCHEMA,'queue':str(queue_path),'base_job':record(output/'base-spec.json'),'previous_job':previous_job,
        'previous_result':state.get('result'),'predecessor_job':record(job_path),'runner_plan':record(runner_plan),'strength_plan':record(strength_path),'ready':record(ready_path),
        'queue_v5':record(root/'tools/top_three_queue_v5.py'),'queue_v4':record(root/'tools/top_three_queue_v4.py'),
        'producer':record(frozen),'original_producer':record(__file__),'steps':steps,'maximum_new_chunks':chunks,'roots_per_chunk':2,
        'start_new_chunks_until_seconds':480,'maximum_new_games_per_chunk':8,'total_pilot_roots':64,'total_pilot_games':256,
        'outcome_policy':'Completion and identities only; no interim outcomes or tuning. Separate assessment after all64roots.',
        'scheduling':'Existing queue v5 run(max_jobs=1) in process; yield shared lock between chunks; stop on busy/failure/unknown.',
        'no_api_or_ui':True,'games_launched':0}
    emit(output/'plan.json',plan);return {'passed':True,'plan':record(output/'plan.json'),'games_launched':0}


def completed_chunk(plan,job_id):
    directory=Path(plan['queue'])/'jobs'/job_id;state=read(directory/'state.json')
    if state['status']!='complete':raise RuntimeError('previous job requires completion or classification: '+job_id)
    job=read(verify(state['job']));result=read(verify(state['result']))
    if result['status']!='complete' or result['exit_code']!=0 or result['job']!=state['job']:raise ValueError('queue binding differs')
    path=Path(job['acceptance'][0]['path']);chunk=read(path);binding=chunk['binding'];value=chunk['value']
    expected={'runner':plan['runner_plan'],'plan':plan['strength_plan'],'chunk_id':flag(job['argv'],'--chunk-id'),'max_new_roots':2,'stage':'pilot'}
    if any(binding.get(k)!=v for k,v in expected.items()) or value.get('passed') is not True or value.get('outcome_totals_reported') is not False:
        raise ValueError('chunk binding differs')
    stage=read(verify(binding['stage_plan']))
    if stage['candidates']!=['hand'] or stage['stage']!='pilot' or stage['plan']!=plan['strength_plan'] or stage['runner']!=plan['runner_plan'] or stage['indexed_ready']!=plan['ready'] or len(stage['root_indices'])!=64:
        raise ValueError('complete single-hand64root pilot plan required')
    if value['newly_executed_games']!=8 or len(value['game_results'])!=8:raise ValueError('exact two-root/eight-game chunk required')
    for ref in value['game_results']:verify(ref)
    return job,chunk,merge_refs([state['job'],state['result'],record(path),binding['stage_plan']],value['game_results'])


def derive_spec(plan,pref,step):
    base=read(verify(plan['base_job']));spec=copy.deepcopy(base);previous_inputs=[];refs=[]
    if step['chunk_id']=='pilot-001':
        state=read(Path(plan['queue'])/'jobs'/step['previous_job']/'state.json')
        if state['status']!='complete' or state['result']!=plan['previous_result']:raise ValueError('bank setup dependency differs')
        verify(plan['previous_result']);verify(plan['ready'])
    else:
        previous,chunk,refs=completed_chunk(plan,step['previous_job']);previous_inputs=previous['inputs']
        if chunk['value']['ready_for_assessment']:raise StopIteration('pilot complete; separate assessment required')
    spec['id']=step['job_id'];spec['dependencies']=[step['previous_job']]
    spec['argv']=replace_flag(spec['argv'],'--chunk-id',step['chunk_id'])
    strength_root=Path(plan['strength_plan']['path']).parent
    spec['acceptance']=[{'path':str(strength_root/'pilot/chunks'/step['chunk_id']/'result.json'),'pointer':'/value/passed','equals':True}]
    spec['inputs']=merge_refs(base['inputs'],previous_inputs,refs,[pref,plan['producer'],plan['original_producer'],plan['queue_v5'],plan['queue_v4'],plan['runner_plan'],plan['strength_plan'],plan['ready']])
    if spec['resume_policy']!='never' or spec['workload']!='authoritative-timing' or spec['matches']!=1:raise ValueError('queue contract changed')
    return spec


def execute_steps(steps,derive,lookup,enqueue,run_one,can_start,persist,maximum=8):
    """Pure scheduling logic; callbacks own immutable IO and the existing queue."""
    completed=[];launched=0
    for step in steps:
        state=lookup(step['job_id'])
        if state and (state['status'] not in ('pending','complete') or (state['status']=='pending' and state['attempts']!=0)):
            return {'status':'blocked','reason':'spent or unknown queue job','job_id':step['job_id'],'completed':completed,'queue_runs':launched}
        try:spec=derive(step)
        except StopIteration as exc:return {'status':'assessment-ready','reason':str(exc),'completed':completed,'queue_runs':launched}
        if state and state['status']=='complete':
            persist(step,spec);completed.append(step['job_id']);continue
        if launched>=maximum:return {'status':'bounded-stop','completed':completed,'queue_runs':launched}
        allowed,reason=can_start()
        if not allowed:return {'status':'stopped','reason':reason,'completed':completed,'queue_runs':launched}
        enqueue(spec)
        allowed,reason=can_start()
        if not allowed:return {'status':'stopped','reason':reason,'completed':completed,'queue_runs':launched}
        result=run_one(step['job_id']);launched+=1
        state=lookup(step['job_id'])
        if result.get('admission') or not state or state['status']!='complete':
            return {'status':'stopped','reason':result.get('reason','target job did not complete'),
                    'admission':result.get('admission'),'job_id':step['job_id'],'completed':completed,'queue_runs':launched}
        persist(step,spec);completed.append(step['job_id'])
    return {'status':'complete','completed':completed,'queue_runs':launched}


def run_target(q,root,job_id):
    """Keep queue.run recovery/locking/monitoring; admit only this pending job.

    The temporary in-memory status view is active solely inside queue._run.
    All complete/running states remain intact; other pending jobs are deferred,
    never mutated. queue.run's outer shared-lock/recovery checks see full status.
    """
    original_run=q._run;original_status=q.status
    def targeted_run(*args,**kwargs):
        def target_status(directory):
            value=original_status(directory)
            return {**value,'jobs':[{**row,'status':'deferred'} if row['id']!=job_id and row['status']=='pending' else row for row in value['jobs']]}
        q.status=target_status
        try:return original_run(*args,**kwargs)
        finally:q.status=original_status
    q._run=targeted_run
    try:return q.run(root,max_jobs=1)
    finally:q._run=original_run;q.status=original_status


def advance(path):
    plan=read(path);pref=record(path)
    if plan['schema']!=SCHEMA or not 1<=len(plan['steps'])<=8 or len(plan['steps'])!=plan['maximum_new_chunks']:raise ValueError('bounded frozen chunk range required')
    for name in ('base_job','predecessor_job','runner_plan','strength_plan','ready','queue_v5','queue_v4','producer','original_producer'):
        verify(plan[name])
    # Import exactly the existing queue implementation on this worktree.
    sys.path.insert(0,str(Path(plan['queue_v5']['path']).parent))
    spec=importlib.util.spec_from_file_location('_h9_batch_queue_v5',verify(plan['queue_v5']));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q=module.queue
    if record(q.__file__)!=plan['queue_v4']:raise ValueError('queue implementation drift')
    root=Path(path).resolve().parent;queue_root=Path(plan['queue'])
    with (root/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        number=1
        while (root/f'invocation-{number:03d}-claim.json').exists():number+=1
        started=time.monotonic()
        emit(root/f'invocation-{number:03d}-claim.json',{'plan':pref,'pid':os.getpid(),'started_unix':time.time()})
        def lookup(job_id):
            file=queue_root/'jobs'/job_id/'state.json'
            return read(file) if file.exists() else None
        def derive(step):
            value=derive_spec(plan,pref,step)
            emit(root/'specs'/(step['job_id']+'.json'),value)
            existing=queue_root/'jobs'/step['job_id']/'job.json'
            if existing.exists() and read(existing)!=value:raise ValueError('existing job differs from bound batch spec')
            return value
        def can_start():
            if time.monotonic()-started>=plan['start_new_chunks_until_seconds']:return False,'batch start deadline reached'
            # No API/UI calls: probe the same advisory shared-resource lock only.
            try:
                with q.lock(q.shared_root()):pass
            except RuntimeError as exc:
                if 'active supervisor' in str(exc):return False,'shared live/resource lock busy'
                raise
            for state_path in (queue_root/'jobs').glob('*/state.json'):
                if read(state_path)['status']=='running':return False,'running or unknown queue claim requires classification'
            return True,None
        def persist(step,spec):
            actual,chunk,refs=completed_chunk(plan,step['job_id'])
            if actual!=spec:raise ValueError('completed job differs from intended spec')
            emit(root/'steps'/(step['chunk_id']+'.json'),{'passed':True,'plan':pref,'job_id':step['job_id'],
                'chunk_id':step['chunk_id'],'bindings':refs,'completed_roots':chunk['value']['completed_roots'],
                'newly_executed_games':chunk['value']['newly_executed_games'],'outcome_totals_read':False})
        try:
            result=execute_steps(plan['steps'],derive,lookup,lambda value:q.enqueue(queue_root,value),
                lambda job_id:run_target(q,queue_root,job_id),can_start,persist,plan['maximum_new_chunks'])
        except BaseException as exc:
            emit(root/f'invocation-{number:03d}-result.json',{'plan':pref,'status':'failed','error':type(exc).__name__+': '+str(exc),'no_automatic_replay':True})
            raise
        value={'plan':pref,**result,'elapsed_seconds':time.monotonic()-started,'outcome_totals_read':False,'api_ui_actions':0}
        emit(root/f'invocation-{number:03d}-result.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('prepare');x.add_argument('--queue',type=Path,required=True);x.add_argument('--previous-job',required=True);x.add_argument('--runner-plan',type=Path,required=True);x.add_argument('--output',type=Path,required=True);x.add_argument('--start-job',type=int,default=254);x.add_argument('--start-chunk',type=int,default=1);x.add_argument('--chunks',type=int,default=8)
    x=s.add_parser('advance');x.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();result=prepare(a.queue,a.previous_job,a.runner_plan,a.output,a.start_job,a.start_chunk,a.chunks) if a.command=='prepare' else advance(a.plan)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
