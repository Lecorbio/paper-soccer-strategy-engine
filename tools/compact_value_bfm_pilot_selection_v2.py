#!/usr/bin/env python3
"""Run the nine pilot trainings and freeze model selection before any game gate."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

if __name__=='__main__':
    for key in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_train as trainer
from tools import compact_value_bfm_teacher_training as selection
from tools import compact_value_bfm_ranking_store as storage

SELECTION_POLICY={
    'schema':campaign.ID+'.pilot-selection-policy.v2',
    'lambdas':[0,.1,.25],'seeds':[20260907,20260908,20260909],
    'seed_selection':'maintained-canonical-validation-key-among-passing-seeds-else-diagnostic',
    'minimum_comparable_groups':100,'minimum_comparable_fraction':.8,
    'early_max_drawn_edges':12,'early_uses_same_coverage_floors':True,
    'minimum_regret_reduction':.10,'maximum_flip_increase':.005,
    'source_reserve_target':2000,'rank4_screen_required':True,
    'rank4_wins':105,'rank4_games':200,'zero_failures':True,
    'pilot_search':'maintained-export-default;no-search-selection-in-pilot',
    'offline_selection_is_not_admission':True,
}


def prepare_policy(context,phase):
    path=context/phase/'selection-policy.json';campaign.seal(path,SELECTION_POLICY);return path


def coverage(metrics):
    return (metrics['comparable_groups']>=100 and metrics['groups']>0
        and metrics['comparable_groups']/metrics['groups']>=.8)


def compare_candidate(control,candidate):
    comparisons={}
    for stratum in ('overall','early'):
        baseline=control[stratum];evaluated=candidate[stratum]
        reduction=selection._regret_reduction(baseline['mean_teacher_regret'],evaluated['mean_teacher_regret'])
        comparisons[stratum]={'regret_reduction':reduction,
            'control_coverage_passed':coverage(baseline),'candidate_coverage_passed':coverage(evaluated),
            'control_comparable':baseline['comparable_groups'],'candidate_comparable':evaluated['comparable_groups'],
            'groups':evaluated['groups'],'control_flip_rate':baseline['float_vs_quantized_action_flip_rate'],
            'candidate_flip_rate':evaluated['float_vs_quantized_action_flip_rate'],
            'passed':coverage(baseline) and coverage(evaluated) and reduction>=.10
                and evaluated['float_vs_quantized_action_flip_rate']<=baseline['float_vs_quantized_action_flip_rate']+.005}
    passed=candidate['canonical_retention_passed'] and candidate['source_reserve']>=2000 and all(x['passed'] for x in comparisons.values())
    return {'lambda':candidate['lambda'],'seed':candidate['seed'],'strata':comparisons,
        'canonical_retention_passed':candidate['canonical_retention_passed'],
        'source_reserve':candidate['source_reserve'],'eligible_for_rank4_screen':bool(passed)}


def assess(context,phase):
    policy_path=prepare_policy(context,phase);path=context/phase/'model-selection.json'
    if path.exists():return campaign.read(path)
    training=campaign.read(context/phase/'training.json')
    if training['smoke'] or not training['mandatory_training_verified']:raise ValueError('pilot requires real nonsmoke training')
    roster={(row['weight'],row['seed']) for row in training['results']}
    if roster!={(weight,seed) for weight in (0,.1,.25) for seed in trainer.FIXED_SEEDS}:raise ValueError('pilot must finish all nine trainings')
    contract=campaign.read(context/'campaign.json');bundle=trainer.FrozenBundle.load(campaign.verify(contract['bundle']))
    ranking=storage.RankingStore(context/phase/'ranking-store/index.json',bundle).labels()
    # Native prefixes are arrays of complete-turn records, not strings.
    early=tuple(group for group in ranking.validation
        if sum(len(turn['action']) for turn in group.evidence['source_binding']['prefix'])<=12)
    if not early: raise ValueError('no eligible early validation groups; admission is unavailable')
    arms=[]
    for weight in (0,.1,.25):
        records=[row for row in training['results'] if row['weight']==weight]
        receipt=selection._selected_seed([row['seed_receipt'] for row in records])
        chosen=next(row for row in records if row['seed']==receipt['seed'])
        architecture,quantized,_,_=trainer.load_runtime(campaign.verify(chosen['runtime']))
        parameters=trainer.load_float_checkpoint(campaign.verify(chosen['float_checkpoint']),architecture)
        campaign.verify(chosen['source'])
        arms.append({'lambda':weight,'seed':chosen['seed'],'runtime':chosen['runtime'],'source':chosen['source'],
            'float_checkpoint':chosen['float_checkpoint'],'canonical_retention_passed':receipt['offline_gate']['passed'],
            'source_reserve':chosen['source_reserve'],'overall':receipt['quantized_validation']['successor_ranking'],
            'early':trainer.successor_ranking_metrics(parameters,architecture,early,quantized=quantized)})
    comparisons=[compare_candidate(arms[0],candidate) for candidate in arms[1:]]
    eligible=[arm for arm,comparison in zip(arms[1:],comparisons) if comparison['eligible_for_rank4_screen']]
    winner=min(eligible,key=lambda arm:(arm['overall']['mean_teacher_regret'],-arm['overall']['top1_agreement'],
        arm['overall']['float_vs_quantized_action_flip_rate'],arm['lambda'])) if eligible else None
    return campaign.seal(path,{'schema':campaign.ID+'.pilot-model-selection.v2','policy':campaign.record(policy_path),
        'training':campaign.record(context/phase/'training.json'),'arms':arms,'comparisons':comparisons,
        'selected':winner,'status':'model-selected-before-rank4-screen' if winner else 'offline-rejected-before-rank4-screen',
        'rank4_bank_opened':False,'pilot_admitted':False,'campaign_success':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--wait-for-labels',action='store_true');parser.add_argument('command',choices=('prepare','train','assess'))
    args=parser.parse_args();root=args.root.resolve();context=args.context.resolve()
    contract=campaign.read(context/'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent!=root:raise ValueError('pilot parent changed')
    prepare_policy(context,args.phase)
    if args.command=='prepare':return
    # A one-shot sequential pipeline wait, not a recurring automation.
    with (root/'.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if args.wait_for_labels else fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':
            labels=campaign.read(context/args.phase/'labels.json')
            if not labels.get('all_groups_exhaustive') or not labels.get('all_native_labels_validated'):
                raise ValueError('pilot labels have not completed native validation')
            campaign.event(root,'pilot-training-started',{'labels':campaign.record(context/args.phase/'labels.json'),
                'selection_policy':campaign.record(context/args.phase/'selection-policy.json')})
            campaign.train_models(context,args.phase,smoke=False)
            campaign.event(root,'pilot-training-completed',{'training':campaign.record(context/args.phase/'training.json')})
        result=assess(context,args.phase)
        campaign.event(root,'pilot-model-selection-completed',{'selection':campaign.record(context/args.phase/'model-selection.json'),'status':result['status']})
    print(json.dumps({'status':result['status'],'pilot_admitted':False}),flush=True)

if __name__=='__main__':main()
