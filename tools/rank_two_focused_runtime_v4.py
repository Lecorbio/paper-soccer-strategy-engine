"""Reviewed runtime_v4 adaptation for four matched sources and full response clocks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from tools import rank_two_runtime_v4 as retained
from tools import top_three_experiments as e
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_export_v2 as exporter
from tools import rank_two_focused_work_v1 as work

SCHEMA=campaign.SCHEMA+'.reviewed-runtime-v4'


def worker_source(aliases):
    source=e.WORKER.read_text()
    anchor='#include TOP_THREE_SOURCE\n'
    if source.count(anchor)!=1:raise ValueError('worker include anchor changed')
    source=source.replace(anchor,anchor+exporter.alias_cleanup(aliases))
    timer='      const auto measured_start = Clock::now();\n'
    if source.count(timer)!=2:raise ValueError('worker clock anchors changed')
    source=source.replace(timer,'')
    anchor="      const auto separator = line.find('\\t');\n"
    if source.count(anchor)!=1:raise ValueError('worker request anchor changed')
    source=source.replace(anchor,'      const auto measured_start = Clock::now();\n'+anchor)
    anchor='      if (budget < 1 || budget > 1000) throw std::invalid_argument("budget");\n'
    if source.count(anchor)!=1:raise ValueError('worker budget anchor changed')
    source=source.replace(anchor,anchor+"      const auto remaining_budget = [&]() { const auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-measured_start).count(); return std::max(1, budget-10-static_cast<int>(spent)); };\n")
    source=source.replace('std::chrono::milliseconds(budget);','std::chrono::milliseconds(remaining_budget());')
    source=source.replace('bot::choose_complete_turn(state, budget);','bot::choose_complete_turn(state, remaining_budget());')
    return source


def verify_build_receipt(build,actor,signature):
    if build['descriptor']!=actor or build['compiler_binary']!=signature['executable'] or build['compiler']!=signature['version']:
        raise ValueError('runtime source/compiler descriptor differs')
    for name in ('binary','compiled_source','worker_source','compiler_binary'):e.verify(build[name])
    if build['compiled_source']['sha256']!=actor['source']['sha256']:raise ValueError('compiled actor source differs')
    if build['adapter_producer']!=campaign.record(__file__):raise ValueError('runtime adapter producer differs')
    if e.read(Path(build['binary']['path']).parent/'build.json')!=build:raise ValueError('worker receipt differs')


def build(actor,aliases,output,signature):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    archived=directory/'submission.cpp';wrapper=directory/'worker.cpp';binary=directory/'worker'
    campaign.immutable(archived,e.verify(actor['source']).read_bytes())
    campaign.immutable(wrapper,worker_source(aliases).encode())
    command=[signature['executable']['path'],'-fintegrated-cc1','-std=c++20','-O3','-DNDEBUG','-ffp-contract=off','-Wl,-ld_classic',
             '-DTOP_THREE_SOURCE="'+str(archived)+'"','-DTOP_THREE_FAMILY='+str(e.FAMILIES[actor['family']]),str(wrapper),'-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    campaign.immutable(directory/'compile-process.json',dict(command=command,returncode=result.returncode,stderr=result.stderr))
    if result.returncode:raise ValueError('runtime worker compilation failed: '+result.stderr[-2000:])
    value=dict(descriptor=actor,worker_source=e.raw_record(wrapper),compiled_source=e.raw_record(archived),binary=e.raw_record(binary),
        compiler_binary=signature['executable'],compiler=signature['version'],command=command,adapter_producer=campaign.record(__file__),
        reviewed_parent_runtime=campaign.record(retained.__file__),parent_worker=e.raw_record(e.WORKER),whole_response_cleanup_reserve_ms=10)
    campaign.immutable(directory/'build.json',value);verify_build_receipt(value,actor,signature)
    return value


class Worker(retained.Worker):
    def choose(self,prefix,budget,external,watchdog_seconds=3):
        started=time.monotonic()
        action,engine_ms=super().choose(prefix,budget,external,watchdog_seconds)
        elapsed=1000*(time.monotonic()-started)
        if elapsed>budget:
            raise retained.WorkerResponseFailure('complete state-worker response envelope exceeded',
                dict(kind='whole_response_envelope_exceeded',action=action,elapsed_ms=elapsed,reported_ms=engine_ms,budget_ms=budget,external_limit_ms=external))
        return action,elapsed


def prepare(plan_path):
    plan=campaign.read(plan_path)
    if plan['producer']!=campaign.record(__file__):raise ValueError('reviewed runtime producer changed')
    work.checked(plan,'experiment',launch=True)
    frozen=campaign.read(campaign.verify(plan['sources']));manifest=campaign.read(campaign.verify(plan['manifest']))
    if len(frozen['models'])!=4:raise ValueError('four frozen matched candidates required')
    signature=e.compiler_signature('/usr/bin/clang++');output=Path(plan['output']);actors={};exports={}
    control=dict(manifest['control']);campaign.verify(control['source']);control['source']=e.raw_record(control['source']['path']);actors['control']=control
    for model in frozen['models']:
        campaign.verify(model['source'])
        tag=model['profile']+'-seed-'+str(model['seed'])
        actors['candidate:'+tag]=dict(source=e.raw_record(model['source']['path']),family='turn_action_v2',clocks_ms=[550,140])
        exports['candidate:'+tag]=campaign.read(campaign.verify(model['export']))['aliases']
    for name,actor in manifest['opponents'].items():
        actor=dict(actor);campaign.verify(actor['source']);actor['source']=e.raw_record(actor['source']['path']);actors['opponent:'+name]=actor
    builds={}
    for name,actor in actors.items():
        work.checked(plan,'experiment')
        builds[name]=build(actor,exports.get(name,{}),output/'workers'/name.replace(':','-'),signature)
    bundle=dict(schema=SCHEMA,passed=True,sources=plan['sources'],manifest=plan['manifest'],actors=actors,builds=builds,
        producer=campaign.record(__file__),reviewed_parent_runtime=campaign.record(retained.__file__),compiler=signature,
        plan=campaign.record(plan_path),clock_policy=e.DEPLOYMENT_PREFIX_CLOCK,whole_response_measured=True,new_games=0)
    campaign.immutable(output/'bundle.json',bundle)
    return bundle


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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    result=prepare(args.plan);print(json.dumps(dict(passed=result['passed'],actors=len(result['actors']))))


if __name__=='__main__':main()
