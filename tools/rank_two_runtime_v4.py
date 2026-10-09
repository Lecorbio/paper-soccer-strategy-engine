#!/usr/bin/env python3
"""Source-bound runtime with durable actor and response failure evidence."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selectors
import subprocess
import time
try:
    from . import top_three_experiments as e, rank_two_campaign_v3 as c
except ImportError:
    import top_three_experiments as e, rank_two_campaign_v3 as c


def build(actor, directory, compiler):
    source=e.verify(actor['source']);signature=e.compiler_signature(compiler)
    key=e.campaign.digest(e.campaign.canonical(['rank-two-classic-v1',actor,signature,e.raw_record(e.WORKER)]))
    directory=Path(directory).resolve()/key;receipt=directory/'build.json'
    if receipt.exists():
        value=e.read(receipt);e.verify_build_receipt(value,actor,signature);return value
    archived=directory/'submission.cpp';wrapper=directory/'worker.cpp'
    e.campaign.immutable(archived,source.read_bytes());e.campaign.immutable(wrapper,e.WORKER.read_bytes())
    binary=directory/'worker'
    command=[signature['executable']['path'],'-fintegrated-cc1','-std=c++20','-O3','-DNDEBUG','-Wl,-ld_classic',
             '-DTOP_THREE_SOURCE="'+str(archived)+'"','-DTOP_THREE_FAMILY='+str(e.FAMILIES[actor['family']]),str(wrapper),'-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    e.emit(directory/'compile-process.json',{'command':command,'returncode':result.returncode,'stderr':result.stderr})
    if result.returncode:raise RuntimeError('worker compilation failed: '+result.stderr[:500])
    value={'descriptor':actor,'worker_source':e.raw_record(wrapper),'compiled_source':e.raw_record(archived),
           'binary':e.raw_record(binary),'compiler_binary':signature['executable'],'compiler':signature['version'],'command':command}
    e.emit(receipt,value);e.verify_build_receipt(value,actor,signature);return value


def prepare(manifest_path, candidates, output):
    output=Path(output).resolve();manifest=c.load(manifest_path)
    loaded={name:e.read(path) for name,path in candidates.items()}
    if not 1<=len(loaded)<=2 or 'control' in loaded:raise ValueError('one or two named candidates required')
    actors={'control':manifest['control'],**{'candidate:'+n:a for n,a in loaded.items()},
            **{'opponent:'+n:a for n,a in manifest['opponents'].items()}}
    signature=e.compiler_signature('/usr/bin/clang++')
    builds={name:build(actor,output/'workers',signature['executable']['path']) for name,actor in actors.items()}
    standalone={}
    for name in ['control',*['candidate:'+n for n in loaded]]:
        actor=actors[name];source=e.verify(actor['source']);binary=output/'standalone'/name.replace(':','-')
        binary.parent.mkdir(parents=True,exist_ok=True)
        command=[signature['executable']['path'],'-fintegrated-cc1','-std=c++20','-O3','-DNDEBUG','-Wl,-ld_classic',str(source),'-o',str(binary)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        e.emit(binary.with_suffix('.compile.json'),{'command':command,'source':actor['source'],'compiler':signature,'returncode':result.returncode,'stderr':result.stderr})
        if result.returncode:raise RuntimeError('standalone compile failure')
        standalone[name]=e.raw_record(binary)
    value={'schema':'papersoccer.rank-two.runtime.v1','passed':True,'manifest':e.raw_record(manifest_path),
           'candidates':loaded,'candidate_descriptors':{name:e.raw_record(path) for name,path in candidates.items()},
           'control':manifest['control'],'opponents':manifest['opponents'],'builds':builds,'standalone':standalone,
           'compiler':signature,'producer':e.raw_record(__file__),'worker':e.raw_record(e.WORKER)}
    e.emit(output/'bundle.json',value);return {'passed':True,'bundle':e.raw_record(output/'bundle.json')}


def exchange(child, request, timeout=3):
    child.stdin.write(request.encode('ascii'));child.stdin.flush();data=bytearray();started=time.monotonic()
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout,selectors.EVENT_READ)
        while b'\n' not in data:
            if time.monotonic()-started>timeout:raise RuntimeError('response watchdog')
            if selector.select(.02):
                part=os.read(child.stdout.fileno(),65537-len(data))
                if not part:raise RuntimeError('closed output before response')
                data.extend(part)
                if len(data)>65536:raise RuntimeError('oversized response')
    if data.count(b'\n')!=1 or not data.endswith(b'\n'):raise RuntimeError('extra response bytes')
    return data.decode('ascii').rstrip('\n')


class WorkerResponseFailure(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class Worker(e.StateWorker):
    def choose(self,prefix,budget,external,watchdog_seconds=3):
        # The shared queue owns aggregate process/RSS monitoring. Avoid spawning
        # another ps helper inside every engine response or perturbing its clock.
        details={'budget_ms':budget,'external_limit_ms':external,
                 'watchdog_seconds':watchdog_seconds,'response':None,
                 'action':None,'elapsed_text':None,'elapsed_ms':None}
        try:
            response=exchange(self.child,f'{budget}\t{prefix or "-"}\n',watchdog_seconds)
        except (RuntimeError,ValueError,OSError,BrokenPipeError) as error:
            details['kind']='response_watchdog' if str(error)=='response watchdog' else 'response_transport_or_format'
            raise WorkerResponseFailure(str(error),details) from error
        details['response']=response
        fields=response.split('\t')
        if len(fields)!=2:
            details['kind']='malformed_response'
            raise WorkerResponseFailure('malformed state-worker response',details)
        action=fields[0];details['action']=action;details['elapsed_text']=fields[1]
        try:
            elapsed=float(fields[1])
        except ValueError as error:
            details['kind']='invalid_elapsed_text'
            raise WorkerResponseFailure(str(error),details) from error
        if not math.isfinite(elapsed):
            details['kind']='nonfinite_elapsed'
        else:
            details['elapsed_ms']=elapsed
            if elapsed<0:details['kind']='negative_elapsed'
            elif elapsed>external:details['kind']='external_limit_exceeded'
            else:return action,elapsed
        raise WorkerResponseFailure('worker operational timeout',details)


def play(row,candidate_player,candidate,opponent,candidate_build,opponent_build):
    position=e.state(row['transcript']);prefix='' if row['transcript'] in ('','-') else row['transcript']
    actors=[None,None];builds=[None,None]
    actors[candidate_player]=candidate;actors[1-candidate_player]=opponent
    builds[candidate_player]=candidate_build;builds[1-candidate_player]=opponent_build
    workers=[];counts=e.prior_decision_counts(prefix);initial=list(counts);decisions=[];failure=None;failure_details=None;context={}
    def actor_context(player,stage):
        return {'stage':stage,'player':player,'role':'candidate' if player==candidate_player else 'opponent',
                'source':actors[player].get('source'),'worker_binary':builds[player].get('binary'),
                'prefix':prefix,'own_decision_counts_before_query':list(counts)}
    try:
        for player,item in enumerate(builds):
            context=actor_context(player,'worker_launch');workers.append(Worker(item))
        for _ in range(320):
            if position.winner is not None:break
            mover=position.to_move;first=counts[mover]==0;budget=actors[mover]['clocks_ms'][0 if first else 1]
            external=1000 if first else 200;context=actor_context(mover,'worker_query')
            context.update(first=first,budget_ms=budget,external_limit_ms=external)
            counts[mover]+=1;action,elapsed=workers[mover].choose(prefix,budget,external)
            context.update(stage='legal_complete_turn',action=action,elapsed_ms=elapsed)
            e.rules.apply_complete_turn(position,mover,action)
            decisions.append({'player':mover,'action':action,'elapsed_ms':elapsed,'budget_ms':budget,'first':first})
            prefix+=('/' if prefix else '')+action
        if position.winner is None:
            context={'stage':'game_completion','kind':'unfinished_game','prefix':prefix,'player':None,'role':None}
            raise RuntimeError('unfinished game')
    except (ValueError,RuntimeError,OSError,BrokenPipeError) as error:
        failure=str(error);failure_details={**context,**getattr(error,'details',{}),'exception':type(error).__name__}
    finally:
        for worker in workers:worker.close()
    return {'candidate_player':candidate_player,'winner':position.winner,
            'candidate_won':position.winner==candidate_player if failure is None else None,
            'failure':failure,'failure_details':failure_details,'decisions':decisions,'transcript':prefix,'root':row['state_sha256'],
            'cluster_id':row['cluster_id'],'opponent':row['opponent'],'clock_policy':e.DEPLOYMENT_PREFIX_CLOCK,
            'initial_own_decision_counts':initial}


def once(path,binding,action):
    result=path/'result.json';claim=path/'claim.json'
    if result.exists():
        saved=e.read(result)
        if saved['binding']!=binding or e.read(claim)!=binding:raise ValueError('retained safety binding changed')
        if not saved['value']['passed']:raise RuntimeError('retained safety failure')
        return saved['value']
    if claim.exists():raise RuntimeError('unknown spent safety claim')
    e.emit(claim,binding)
    try:value=action()
    except Exception as error:
        value={'passed':False,'error':str(error)}
        if hasattr(error,'details'):value['failure_details']=error.details
    e.emit(result,{'binding':binding,'value':value})
    if not value['passed']:raise RuntimeError('safety failed: '+str(value))
    return value


def protocol(bundle_path,name,output):
    bundle=e.read(bundle_path);e.verify(bundle['producer']);e.verify(bundle['worker'])
    actor=bundle['candidates'][name];e.verify(actor['source'])
    binary=e.verify(bundle['standalone']['candidate:'+name]);worker=bundle['builds']['candidate:'+name]
    e.verify_build_receipt(worker,actor,bundle['compiler'])
    fixture=e.ROOT/'results/top_three_20260916/rank4-pure-margin650-140-v1/PLAN.json'
    fixtures=e.read(fixture)['stress_fixtures']
    binding={'bundle':e.raw_record(bundle_path),'source':actor['source'],'fixture':e.raw_record(fixture),'producer':e.raw_record(__file__)}
    results=[]
    for color in (0,1):
        def native():
            child=None;traces=[]
            try:
                prefix='' if color==0 else '0';state=e.state(prefix)
                request=f'{color}\n1\n'+('-' if color==0 else '0')+'\n'
                started=time.monotonic();child=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
                os.set_blocking(child.stdout.fileno(),False)
                action=exchange(child,request);first_ms=1000*(time.monotonic()-started)
                e.rules.apply_complete_turn(state,color,action);prefix+=('/' if prefix else '')+action;traces.append(prefix)
                other=state.to_move;reply=''
                while state.winner is None and state.to_move==other:
                    digit=next(i for i,(dx,dy) in enumerate(e.rules.DIRECTION_DELTAS) if e.rules._legal_destination(state,(state.ball[0]+dx,state.ball[1]+dy)))
                    e.rules.apply_primitive(state,digit);reply+=str(digit)
                if state.winner is not None:raise RuntimeError('later fixture became terminal')
                started=time.monotonic();action=exchange(child,f'{len(reply)}\n{reply}\n');later_ms=1000*(time.monotonic()-started)
                e.rules.apply_complete_turn(state,color,action);traces.append(prefix+'/'+reply+'/'+action)
                child.stdin.close();child.wait(timeout=3)
                if child.returncode or child.stdout.read():raise RuntimeError('native exit/extra output')
                return {'passed':first_ms<=1000 and later_ms<=200,'color':color,'first_wall_ms':first_ms,'later_wall_ms':later_ms,'traces':traces}
            finally:
                if child is not None and child.poll() is None:child.kill();child.wait()
        results.append(once(output/f'native-p{color}',{**binding,'color':color},native))
    def stress():
        child=None;rows=[]
        try:
            started=time.monotonic();child=Worker(worker);action,ms=child.choose('',550,1000)
            wall=1000*(time.monotonic()-started);s=e.state('');e.rules.apply_complete_turn(s,0,action)
            rows.append({'prefix':'','action':action,'engine_ms':ms,'wall_ms':wall,'passed':wall<=1000})
            for row in fixtures:
                started=time.monotonic();action,ms=child.choose(row['prefix'],140,200);wall=1000*(time.monotonic()-started)
                s=e.state(row['prefix']);e.rules.apply_complete_turn(s,s.to_move,action)
                rows.append({**row,'action':action,'engine_ms':ms,'wall_ms':wall,'passed':wall<=200})
            return {'passed':len(rows)==13 and all(r['passed'] for r in rows),'decisions':rows}
        finally:
            if child:child.close()
    results.append(once(output/'stress',binding,stress))
    value={'passed':True,'source':actor['source'],'binding':binding,'cases':results,'strength_qualified':False}
    e.emit(output/'result.json',value);return {'passed':True,'result':e.raw_record(output/'result.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','protocol'])
    p.add_argument('--manifest',type=Path);p.add_argument('--candidate',nargs=2,action='append');p.add_argument('--bundle',type=Path)
    p.add_argument('--name');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.manifest,dict(a.candidate),a.output) if a.command=='prepare' else protocol(a.bundle,a.name,a.output),indent=2))
