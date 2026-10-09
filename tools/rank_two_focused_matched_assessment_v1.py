"""Validate complete source-bound panels and assess only whole-root aggregates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v6 as runtime
from tools import rank_two_focused_work_v1 as work

SCHEMA=campaign.SCHEMA+'.matched-assessment.v1'


def percentile(values,p):
    ordered=sorted(values);index=(len(ordered)-1)*p
    lower=math.floor(index);upper=math.ceil(index)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)


def bootstrap(vectors,seed,repetitions):
    n=len(next(iter(vectors.values())))
    if not n or any(len(v)!=n for v in vectors.values()):raise ValueError('whole-root vector lengths differ')
    rng=random.Random(seed);draws={name:[] for name in vectors}
    for _ in range(repetitions):
        indexes=[rng.randrange(n) for _ in range(n)]
        for name,values in vectors.items():draws[name].append(sum(values[i] for i in indexes)/n)
    return {name:dict(mean=sum(values)/n,lower95=percentile(draws[name],.025),
        upper95=percentile(draws[name],.975),roots=n,bootstrap_unit='whole root',
        bootstrap_seed=seed,bootstrap_repetitions=repetitions) for name,values in vectors.items()}


def validate_game(document,claim,row,color,actor,opponent,plan_ref,runtime_ref):
    expected=dict(plan=plan_ref,row=row,actor=claim['actor'],source=actor['source'],
                  opponent=opponent,color=color,runtime=runtime_ref)
    if claim!=expected or document['claim']!=claim:raise ValueError('exact game claim/source/runtime differs')
    game=document['game']
    if (game['failure'] is not None or game['failure_details'] is not None
            or game['candidate_player']!=color or game['root']!=row['state_sha256']
            or game['cluster_id']!=row['cluster_id'] or game['opponent']!=row['opponent']
            or game['clock_policy']!=runtime.e.DEPLOYMENT_PREFIX_CLOCK):
        raise ValueError('game identity/completion differs')
    state=runtime.e.state(row['transcript']);counts=runtime.e.prior_decision_counts(row['transcript'])
    if game['initial_own_decision_counts']!=counts:raise ValueError('prefix clock counts differ')
    prefix=row['transcript'] if row['transcript']!='-' else ''
    for decision in game['decisions']:
        player=state.to_move;first=counts[player]==0;who=actor if player==color else opponent
        budget=who['clocks_ms'][0 if first else 1];external=1000 if first else 200
        limit=external if who['response_policy']=='frozen-opponent-engine-budget' else budget
        elapsed=decision['elapsed_ms']
        if (decision['player']!=player or decision['first']!=first or decision['budget_ms']!=budget
                or not math.isfinite(elapsed) or not 0<=elapsed<=limit):
            raise ValueError('declared whole-response policy differs')
        counts[player]+=1;runtime.e.rules.apply_complete_turn(state,player,decision['action'])
        prefix+=('/' if prefix else '')+decision['action']
    if (state.winner is None or state.winner!=game['winner'] or prefix!=game['transcript']
            or game['candidate_won']!=(state.winner==color)):
        raise ValueError('complete protocol replay/winner differs')
    return float(game['candidate_won'])


def run(plan_path):
    plan=campaign.read(plan_path)
    if plan['producer']!=campaign.record(__file__):raise ValueError('assessment producer differs')
    current=work.checked(plan,'experiment',launch=True)
    for ref in plan['inputs'].values():campaign.verify(ref)
    gameplay=campaign.read(campaign.verify(plan['gameplay_plan']))
    result=campaign.read(campaign.verify(plan['result']))
    bank=campaign.read(campaign.verify(gameplay['bank']))
    bundle=campaign.read(campaign.verify(gameplay['runtime']))
    recipe=gameplay['evaluation_recipe'];names=gameplay['candidates'];all_names=['control',*names]
    if (not result['passed'] or result['completed_games']!=2048 or result['own_failures']!=0
            or result['plan']!=plan['gameplay_plan'] or len(result['root_receipts'])!=256
            or result['bank']!=gameplay['bank'] or result['runtime']!=gameplay['runtime']
            or result['canceled_unclaimed_slots']!=512 or result['original_declared_slots']!=2560
            or not result['all_declared_games_archived'] or len(bank['rows'])!=256):
        raise ValueError('complete declared panel archive required')
    if (gameplay['roster_approval']!=current['focused_h62_roster_exception_approval']
            or gameplay['offline_roster']!=current['focused_active_offline_roster']
            or gameplay['runtime']!=current['focused_runtime_v6_bundle']):
        raise ValueError('approved roster/runtime binding differs')
    models=campaign.read(campaign.verify(plan['sources']))['models']
    model_by_name={'candidate:'+m['profile']+'-seed-'+str(m['seed']):m for m in models}
    roots=[];seen_games=set();cancellations=[];decisions=0
    for index,(row,ref) in enumerate(zip(bank['rows'],result['root_receipts'])):
        work.checked(plan,'experiment');root=campaign.read(campaign.verify(ref))
        order=all_names[index%4:]+all_names[:index%4]
        if (row['index']!=index or root['row']!=row or root['plan']!=plan['gameplay_plan']
                or root['execution_order']!=order or set(root['arms'])!=set(all_names)
                or not root['control_reused_once_per_root_color']):
            raise ValueError('shared control/root/order binding differs')
        canceled=root['canceled']
        if (canceled['source']!=bank['excluded_source'] or canceled['status']!='canceled-unclaimed'
                or canceled['observed_games']!=0 or canceled['substitutions']!=0
                or canceled['slots']!=[dict(root=row['cluster_id'],color=c) for c in (0,1)]):
            raise ValueError('banned original source cancellation differs')
        cancellations.append(canceled);wins={}
        for name in all_names:
            if len(root['arms'][name])!=2:raise ValueError('both colors required')
            pair=[];actor=bundle['actors'][name];opponent=bundle['actors']['opponent:'+row['opponent']]
            for color,game_ref in enumerate(root['arms'][name]):
                if game_ref['path'] in seen_games:raise ValueError('game reused across slots')
                seen_games.add(game_ref['path']);path=campaign.verify(game_ref)
                document=campaign.read(path);claim=campaign.read(path.with_name('CLAIM.json'))
                if claim['actor']!=name:raise ValueError('game assigned to a different arm')
                pair.append(validate_game(document,claim,row,color,actor,opponent,plan['gameplay_plan'],gameplay['runtime']))
                decisions+=len(document['game']['decisions'])
            wins[name]=sum(pair)/2
        roots.append(dict(index=index,cluster_id=row['cluster_id'],opponent=row['opponent'],wins=wins,
            deltas={name:wins[name]-wins['control'] for name in names}))
    if (len(seen_games)!=2048 or len({r['cluster_id'] for r in roots})!=256
            or len({r['state_sha256'] for r in bank['rows']})!=256):raise ValueError('independent whole roots/games differ')
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    archive=campaign.immutable(output/'ARCHIVE.json',dict(schema=SCHEMA+'.archive',passed=True,
        gameplay_plan=plan['gameplay_plan'],result=plan['result'],bank=gameplay['bank'],runtime=gameplay['runtime'],
        root_receipts=result['root_receipts'],validated_games=2048,validated_decisions=decisions,
        canceled_unclaimed_slots=512,cancellations=cancellations,original_declared_slots=2560,
        zero_operational_failures=True,protocol_replay_verified=True,control_reused_once_per_root_color=True,
        all_source_roster_runtime_bindings_verified=True))
    vectors={name:[r['deltas'][name] for r in roots] for name in names}
    pilot=bootstrap({n:v[:64] for n,v in vectors.items()},recipe['bootstrap_seed'],recipe['bootstrap_repetitions'])
    development=bootstrap(vectors,recipe['bootstrap_seed'],recipe['bootstrap_repetitions'])
    for name in names:
        by_opponent={op:sum(r['deltas'][name] for r in roots if r['opponent']==op)/
            sum(r['opponent']==op for r in roots) for op in sorted({r['opponent'] for r in roots})}
        pilot[name].update(paired_games=256,zero_operational_failures=True,
            passed=pilot[name]['upper95']>=recipe['pilot']['upper95_at_least'])
        development[name].update(paired_games=1024,per_opponent_mean=by_opponent,zero_operational_failures=True,
            passed=development[name]['mean']>=recipe['development']['mean_at_least'] and
                min(by_opponent.values())>=recipe['development']['each_opponent_floor'])
    feasible={profile:[n for n in names if model_by_name[n]['profile']==profile] for profile in ('dd8','dd12wide')}
    if len(feasible['dd8'])!=2 or len(feasible['dd12wide'])!=1:
        raise ValueError('declared feasibility exception no longer matches original seeds')
    priority=[n for n in feasible['dd8'] if pilot[n]['passed'] and development[n]['passed']]
    priority.sort(key=lambda n:(-development[n]['mean'],model_by_name[n]['validation']['regret'],
        model_by_name[n]['validation']['huber'],model_by_name[n]['seed']))
    dd8mean=[sum(vectors[n][i] for n in feasible['dd8'])/2 for i in range(256)]
    family_vs_control=bootstrap({'dd8':dd8mean},recipe['bootstrap_seed'],recipe['bootstrap_repetitions'])['dd8']
    paired_seed_two={m['profile']:n for n,m in model_by_name.items() if n in names and m['seed']==2026100702}
    diagnostic=sum(roots[i]['wins'][paired_seed_two['dd12wide']]-roots[i]['wins'][paired_seed_two['dd8']] for i in range(256))/256
    value=dict(schema=SCHEMA,passed=True,plan=campaign.record(plan_path),archive=archive,
        pilot=pilot,development=development,models={n:dict(profile=model_by_name[n]['profile'],
            seed=model_by_name[n]['seed'],source=model_by_name[n]['source']) for n in names},
        family_choice='dd8',family_choice_basis='only two-seed mechanically feasible family',
        architecture_superiority_claim=False,complete_two_seed_width_comparison=False,
        missing_width_seed=2026100701,missing_width_source=bank['excluded_source'],
        surviving_seed_two_width_difference_descriptive_only=diagnostic,
        dd8_family_vs_control=family_vs_control,eligible_candidate_priority=priority,
        live_promotion_complete=False,training_eligible=False,new_games=0,
        quarantined_panels_included=False,individual_game_strategy_mining=False)
    campaign.immutable(output/'RESULT.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();r=run(args.plan)
    print(json.dumps(dict(passed=r['passed'],family=r['family_choice'],eligible=r['eligible_candidate_priority'])))


if __name__=='__main__':main()
