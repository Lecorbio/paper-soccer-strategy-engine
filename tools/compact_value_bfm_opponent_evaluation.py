#!/usr/bin/env python3
"""Paired, root-clustered evaluation of six frozen local opponent styles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign


def load_games(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    games={}
    for row in rows:
        if row.get('schema')!='papersoccer.compact-state-evaluation.v2': raise ValueError('wrong state-adapter schema')
        key=(row['root_id'],row['candidate_player'])
        if key in games: raise ValueError('duplicate root/color outcome')
        if row['candidate_player'] not in (0,1): raise ValueError('invalid candidate color')
        games[key]=row
    if not games: raise ValueError('empty state-adapter result')
    return games


def assess(candidate,control,*,replicates=10000,seed=20260909):
    """Resample canonical root pairs within each opponent, retaining paired arms."""
    import numpy as np
    if set(candidate)!=set(campaign.OPPONENTS) or set(control)!=set(candidate):
        raise ValueError('six-opponent roster must be complete')
    differences=[]; opponents={}; failures=[]; all_roots=set()
    for name in campaign.OPPONENTS:
        left,right=candidate[name],control[name]
        if set(left)!=set(right): raise ValueError('candidate/control schedules differ')
        roots=sorted({key[0] for key in left})
        if len(left)!=2*len(roots): raise ValueError('incomplete color pairs')
        if all_roots.intersection(roots): raise ValueError('root reused across opponents; requires joint-cluster bootstrap')
        all_roots.update(roots)
        if len(roots) not in (32,100): raise ValueError('only admitted screen/confirmation sizes are accepted')
        if any((root,color) not in left for root in roots for color in (0,1)): raise ValueError('missing color')
        scores=[[],[]]
        for index,arm in enumerate((left,right)):
            for root in roots:
                pair=[]
                for color in (0,1):
                    game=arm[root,color]
                    if game.get('failure') or game.get('winner') not in (0,1):
                        failures.append({'opponent':name,'arm':index,'root':root,'color':color,'failure':game.get('failure') or 'invalid-winner'})
                        pair.append(0.0)
                    else: pair.append(float(game['winner']==color))
                    if game.get('first_budget_ms')!=800 or game.get('later_budget_ms')!=155:
                        raise ValueError('qualification requires actual 800/155ms internal clocks')
                scores[index].append(sum(pair)/2)
        left_scores=np.asarray(scores[0]);right_scores=np.asarray(scores[1]);delta=left_scores-right_scores
        if any(left[root,color]['root_edges']!=right[root,color]['root_edges']
               or left[root,color].get('root_transcript')!=right[root,color].get('root_transcript')
               for root in roots for color in (0,1)):
            raise ValueError('root progress changed across arms')
        early=sum(8<=left[root,0]['root_edges']<=12 for root in roots)
        if early!=len(roots)//2 or any(left[root,0]['root_edges']<8 for root in roots):
            raise ValueError('early/later root marginals changed')
        opponents[name]={'root_pairs':len(roots),'candidate_win_rate':float(left_scores.mean()),
            'control_win_rate':float(right_scores.mean()),'improvement':float(delta.mean())}
        differences.append(delta)
    rng=np.random.default_rng(seed)
    boot=np.zeros(replicates,dtype=np.float64)
    for delta in differences:
        boot+=delta[rng.integers(0,len(delta),size=(replicates,len(delta)))].mean(axis=1)/len(differences)
    lower,upper=np.quantile(boot,[.025,.975]);improvement=sum(d.mean() for d in differences)/len(differences)
    passed=not failures and improvement>=.03 and lower>0 and all(d.mean()>=-.05 for d in differences)
    return {'schema':'papersoccer.compact-multi-opponent-assessment.v2','passed':bool(passed),
        'opponents':opponents,'equal_weight_improvement':float(improvement),
        'paired_95_interval':[float(lower),float(upper)],'failures':failures,
        'bootstrap':{'replicates':replicates,'seed':seed,'unit':'canonical-root-pair',
            'paired_arms_preserved':True,'stratified_by_opponent':True},
        'proxy_opponents':True,'live_success':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=campaign.read(args.manifest)
    games={}
    for arm in ('candidate','control'):
        games[arm]={name:load_games(campaign.verify(manifest[arm][name])) for name in campaign.OPPONENTS}
    result=assess(games['candidate'],games['control'])
    campaign.seal(args.output,{**result,'manifest':campaign.record(args.manifest)})
    print(json.dumps({'passed':result['passed'],'improvement':result['equal_weight_improvement']}))

if __name__=='__main__': main()
