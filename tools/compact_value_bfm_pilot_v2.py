#!/usr/bin/env python3
"""Prepare and execute the mandatory-training v2 pilot after the native smoke."""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import sys

if __name__=='__main__':
    for variable in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[variable]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign


def context_root(root,attempt):
    if attempt<1: raise ValueError('attempt zero is forbidden')
    if attempt!=1: raise ValueError('later attempts require frozen outcome and accumulated-exclusion integration')
    return root/'phases'/f'attempt-{attempt:03d}-pilot'


def validate_smoke(root):
    path=root/'smoke-064/smoke-completion.json';smoke=campaign.read(path)
    if (smoke.get('status')!='native-smoke-complete-not-qualified'
            or smoke.get('new_training_verified') is not True
            or smoke.get('qualification_eligible') is not False
            or smoke.get('comparable_validation_groups',0)<100
            or smoke['comparable_validation_groups']<.8*smoke['validation_groups']):
        raise ValueError('native smoke evidence is incomplete')
    training=campaign.read(campaign.verify(smoke['training_receipt']))
    if not training['smoke'] or not training['mandatory_training_verified']: raise ValueError('real smoke training is absent')
    if {r['weight'] for r in training['results']}!={0,.1,.25}: raise ValueError('smoke recipe roster changed')
    for result in training['results']:
        for key in ('runtime','source','float_checkpoint'): campaign.verify(result[key])
        if set(result['master_updates'])!={'w1','w2','w3'} or any(
                not v['changed'] or not math.isfinite(v['l2_delta']) or v['l2_delta']<=0
                or v['before_sha256']==v['after_sha256'] for v in result['master_updates'].values()):
            raise ValueError('smoke has no bound all-layer updates')
        if not any(v['changed_codes'] for v in result['quantized_changes_vs_initialization'].values()):
            raise ValueError('smoke retained the initialization payload')
    native=campaign.read(campaign.verify(smoke['native_export_checks']))
    if not native['all_passed'] or len(native['checks'])!=3: raise ValueError('native export checks are absent')
    observed=set()
    for record in native['checks']:
        checked=campaign.read(campaign.verify(record))
        if not checked['passed'] or checked['inference_parity']['states']!=4096: raise ValueError('native export parity failed')
        campaign.verify(checked['source']);campaign.verify(checked['runtime'])
        observed.add((checked['source']['sha256'],checked['runtime']['sha256']))
    if observed!={(r['source']['sha256'],r['runtime']['sha256']) for r in training['results']}:
        raise ValueError('native checks do not bind the three trained smoke sources')
    return campaign.record(path)


def smoke_exclusions(root):
    index=root/'exclusions/smoke-data-index.json'
    if index.exists():
        document=campaign.read(index)
        for record in document['artifacts']: campaign.verify(record)
        return document['artifacts']
    positions=campaign.read(root/'smoke-064/eligible-positions.json')
    games=campaign.read(root/'smoke-064/games.json')
    groups=defaultdict(set)
    for row in positions['rows']:
        role='prior-train' if row['split']=='train' else 'prior-validation'
        for member in row['closure']:
            for domain,value in member.items(): groups[role,domain].add(value)
    # Prefix ancestors of a nonempty root are not validation descendants.
    # Only root and post-root boundaries enter that root's frozen split.
    for row in games['rows']:
        role='prior-train' if row['split']=='train' else 'prior-validation'
        state=campaign.features.ReplayState()
        for turn,action in enumerate(row['game']['transcript'].split('/')):
            if turn>=row['game']['prefix_turns']:
                for domain,value in campaign.fingerprints(state).items(): groups[role,domain].add(value)
            campaign.features.apply_complete_turn(state,state.to_move,action)
    artifacts=[]
    for ordinal,((role,domain),values) in enumerate(sorted(groups.items())):
        path=root/'exclusions'/f'smoke-data-{ordinal}.json'
        campaign.seal(path,{'schema':campaign.ID+'.smoke-data-exclusions.v2','role':role,'domain':domain,
            'fingerprints':sorted(values),'source_games':campaign.record(root/'smoke-064/games.json'),
            'source_positions':campaign.record(root/'smoke-064/eligible-positions.json'),
            'labels_reused':False,'contains_labels':False,'contains_metrics':False,'contains_transcripts':False})
        artifacts.append(campaign.record(path))
    campaign.seal(index,{'schema':campaign.ID+'.smoke-data-exclusion-index.v2','artifacts':artifacts,
        'narrow_train_opening_exception_unchanged':True,'validation_never_exempt':True})
    campaign.event(root,'smoke-data-exclusions-frozen',{'index':campaign.record(index)})
    return artifacts


def prepare(root,attempt):
    parent=campaign.read(root/'campaign.json');smoke=validate_smoke(root)
    baseline=campaign.read(root/'baseline-engine-comparison.json')
    if not baseline['same_weights'] or not baseline['all_checks_passed']: raise ValueError('engine-version baseline is incomplete')
    context=context_root(root,attempt);context.mkdir(parents=True,exist_ok=True)
    path=context/'campaign.json'
    if path.exists(): return campaign.read(path)
    exclusions=list(parent['exclusions'])+smoke_exclusions(root)+baseline['exclusions']
    anchors=campaign.read(root/'exclusions/anchor-derived.json')
    for role,values in anchors['fingerprints'].items():
        out=root/'exclusions'/f'pilot-anchor-{role}.json'
        campaign.seal(out,{'schema':campaign.ID+'.pilot-anchor-exclusion.v2','role':role,
            'domain':anchors['domain'],'fingerprints':values,'source':campaign.record(root/'exclusions/anchor-derived.json')})
        exclusions.append(campaign.record(out))
    exclusions.append(campaign.record(root/'exclusions/prior-search-validation.json'))
    # Reuse immutable feature indexes; their artifact paths stay within this campaign.
    for name in ('anchor-derived.json','prior-search-validation.json'):
        campaign.copy_checked(root/'exclusions'/name,context/'exclusions'/name)
    body={k:v for k,v in parent.items() if k!='body_sha256'}
    body.update({'parent_campaign':campaign.record(root/'campaign.json'),'smoke_completion':smoke,
        'baseline_engine_comparison':campaign.record(root/'baseline-engine-comparison.json'),
        'execution_sources':{name:campaign.copy_checked(Path(module.__file__),context/'source-closure'/f'{name}.py') for name,module in (('campaign',campaign),('features',campaign.features),('corpus',campaign.corpus),('openings',campaign.openings))},
        'attempt':attempt,'phase':'pilot','exclusions':exclusions,'heavy_stage_root':str(root),
        'candidate_lineage':{'mandatory_training':True,'initial_float':parent['inputs']['attempt_one_initial_checkpoint'],
            'generation_student':parent['inputs']['attempt_zero_runtime'],'smoke_weights_reused':False},
        'pilot_games':2000,'pilot_training_roster':{'lambdas':[0,.1,.25],'seeds':[20260907,20260908,20260909]},
        'execution_source':campaign.copy_checked(Path(__file__),context/'pilot-driver.py')})
    result=campaign.seal(path,body)
    campaign.event(root,'pilot-context-frozen',{'context':campaign.record(path),'attempt':attempt})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt',type=int,default=1);parser.add_argument('command',choices=('prepare','games'))
    args=parser.parse_args();root=args.root.resolve();os.environ.update(campaign.THREADS)
    with campaign.lease(root):
        prepare(root,args.attempt)
        if args.command=='games':
            context=context_root(root,args.attempt)
            result=campaign.run_games(context,f'attempt-{args.attempt:03d}-pilot',2000,8)
            campaign.event(root,'pilot-games-completed',{'attempt':args.attempt,
                'games':campaign.record(context/f'attempt-{args.attempt:03d}-pilot/games.json'),'count':result['games']})
    print('pilot '+args.command+' complete',flush=True)

if __name__=='__main__': main()
