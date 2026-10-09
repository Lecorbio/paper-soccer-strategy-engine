"""Parameterized, retained training-only coverage expansion within a locked family."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work
from tools import rank_two_runtime_v4 as retained_runtime
from tools import top_three_experiments as games
from tools import top_three_development_banks as proposals

SCHEMA=campaign.SCHEMA+'.expansion-data.v1'


def bound_plan(path,action,launch=False):
    plan=campaign.read(path)
    if plan['producer']!=campaign.record(__file__):raise ValueError('expansion producer differs')
    current=work.checked(plan,action,launch=launch)
    if (current.get('focused_family')!='dd8' or current.get('focused_family_lock')!=plan['family_lock']
            or current.get('focused_round_01_declaration')!=plan['declaration']
            or current.get('focused_active_offline_roster')!=plan['offline_roster']):
        raise PermissionError('locked family/round/approved roster binding changed')
    declaration=campaign.read(campaign.verify(plan['declaration']))
    if (declaration['fresh_train_roots']!=384 or declaration['fresh_validation_roots']!=128
            or declaration['root_generation_seed']!=plan['seed'] or declaration['root_depth']!=10
            or declaration['training_roots_after']!=512 or declaration['principal_intervention']!='training-coverage-expansion'):
        raise ValueError('declared single data intervention differs')
    for reference in plan['inputs'].values():campaign.verify(reference)
    return plan,current


def draw(plan_path):
    plan,current=bound_plan(plan_path,'prepare-bank',launch=True)
    if current['pending_exposures'] or current['focused_ancestry_ready']!=plan['ancestry']:
        raise PermissionError('exact completed ancestry required before new roots')
    output=Path(plan['output'])/'bank';output.mkdir(parents=True,exist_ok=True)
    if (output/'CLAIM.json').exists():raise RuntimeError('spent expansion bank claim retained')
    campaign.immutable(output/'CLAIM.json',dict(plan=campaign.record(plan_path),ancestry=plan['ancestry']))
    ready=campaign.read(campaign.verify(plan['ancestry']));prior=campaign.read(campaign.verify(ready['prior']))
    database=campaign.verify(prior['inputs']['database'])
    manifest=campaign.read(campaign.verify(plan['offline_roster']));names=sorted(manifest['opponents'])
    if len(names)!=8 or manifest['control']['source']!=current['incumbent']:
        raise ValueError('exact frozen incumbent/eight-opponent distribution required')
    rows=[];seen_states=set();seen_features=set();drawn=0
    try:
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            db.execute('PRAGMA cache_size=-16384')
            for index in range(512):
                bound_plan(plan_path,'generate')
                split='train' if index<384 else 'validation'
                slot=dict(opponent=names[index%8],drawn_edges=10,physical_mover=(index//8)%2,
                          depth_slot=index,split=split)
                for attempt in range(plan['maximum_per_slot']):
                    if drawn>=plan['maximum_proposals']:raise RuntimeError('declared expansion proposal cap reached')
                    seed=proposals.proposal_seed(plan['seed'],plan['proposal_stage'],slot,attempt)
                    directory=output/'proposals'/f'{drawn:06}'
                    campaign.immutable(directory/'CLAIM.json',dict(slot=index,attempt=attempt,proposal_seed=seed,drawn_edges=10))
                    position,prefix,error=proposals.propose(seed,10);drawn+=1
                    record=dict(slot=index,attempt=attempt,proposal_seed=seed,drawn_edges=10,
                                status=error or 'complete',training_eligible=False)
                    accepted=False
                    if error is None:
                        key=games.fingerprint(position);feature=proposals.feature_key(position)
                        overlap=key in seen_states or feature in seen_features or any(
                            db.execute('SELECT 1 FROM keys WHERE category=? AND value=?',(category,value)).fetchone()
                            for category,value in [('states',key),('features',feature)])
                        accepted=position.to_move==slot['physical_mover'] and not overlap
                        record.update(prefix=prefix,state_sha256=key,feature_sha256=feature,accepted=accepted,
                                      rejection='ancestry-overlap' if overlap else ('mover-mismatch' if not accepted else None))
                    campaign.immutable(directory/'RESULT.json',record)
                    if accepted:
                        seen_states.add(key);seen_features.add(feature)
                        group=f'focused-round01-{split}-{index:04}'
                        rows.append(dict(index=index,transcript=prefix,state_sha256=key,feature_sha256=feature,
                            root_group_id=group,cluster_id=group,opponent=slot['opponent'],split=split,
                            physical_mover=slot['physical_mover'],colors=[0,1],drawn_edges=10,
                            proposal=campaign.record(directory/'RESULT.json')))
                        break
                else:raise RuntimeError('expansion draw exhausted; no seed/depth replacement')
                campaign.atomic(output/'PROGRESS.json',dict(roots=len(rows),proposals=drawn,total_roots=512))
    finally:
        campaign.immutable(output/'PROPOSAL_EXPOSURE.json',dict(schema=SCHEMA+'.proposal-exposure',
            proposal_receipts=[campaign.record(p) for p in sorted((output/'proposals').glob('*/RESULT.json'))],
            retained_claims=[campaign.record(p) for p in sorted((output/'proposals').glob('*/CLAIM.json'))],
            recipe=campaign.record(plan_path),training_eligible=False,before_next_fresh_bank=True))
    if Counter(r['split'] for r in rows)!={'train':384,'validation':128}:
        raise ValueError('new384TRAIN/128independentVALIDATION allocation differs')
    bank=dict(schema=SCHEMA+'.bank',passed=True,plan=campaign.record(plan_path),rows=rows,seed=plan['seed'],
        ancestry=plan['ancestry'],offline_roster=plan['offline_roster'],new_games=0,
        old_training_roots_retained=128,training_roots_after=512,new_validation_roots=128,
        whole_root_reflected_feature_separation=True,proposals=drawn)
    campaign.immutable(output/'BANK.json',bank)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.reserved-training-bank',
        bank=campaign.record(output/'BANK.json'),proposal_receipts=campaign.record(output/'PROPOSAL_EXPOSURE.json'),
        training_eligible=False,before_next_fresh_bank=True))
    campaign.immutable(output/'RESULT.json',dict(passed=True,bank=campaign.record(output/'BANK.json'),
        roots=512,proposals=drawn,new_games=0,all_proposals_and_partials_retained=True))
    return bank


def build(plan_path):
    plan,current=bound_plan(plan_path,'generate',launch=True)
    manifest=campaign.read(campaign.verify(plan['offline_roster']));actors={}
    for name,actor in manifest['opponents'].items():
        actors[name]=games.descriptor(campaign.verify(actor['source']),actor['family'],[50,10])
    actors['rank_3']=games.descriptor(campaign.verify(current['incumbent']),'turn_action_v2',[50,10])
    output=Path(plan['output']);builds={}
    # Preserve the original training-only 50/10 engine-clock producer. This is
    # separate from strict whole-response production admission at550/140.
    for name,actor in actors.items():
        bound_plan(plan_path,'generate')
        builds[name]=retained_runtime.build(actor,output/'builds','/usr/bin/clang++')
    value=dict(schema=SCHEMA+'.build',passed=True,actors=actors,builds=builds,plan=campaign.record(plan_path),
        training_clocks_ms=[50,10],strength_eligible=False,outcomes_as_targets=False,new_games=0)
    campaign.immutable(output/'BUILDS.json',value)
    return value


def generate(plan_path):
    plan,current=bound_plan(plan_path,'generate',launch=True)
    bank=campaign.read(campaign.verify(plan['bank']));builds=campaign.read(campaign.verify(plan['builds']))
    if bank['seed']!=plan['seed'] or bank['offline_roster']!=plan['offline_roster']:
        raise ValueError('exact expansion bank/roster differs')
    output=Path(plan['output']);envelopes=[];references=[];failure=None
    for row in bank['rows']:
        for color in (0,1):
            bound_plan(plan_path,'generate');identifier=row['root_group_id']+'-p'+str(color)
            directory=output/'games'/identifier;path=directory/'RESULT.json'
            binding=dict(plan=campaign.record(plan_path),row=row,color=color,
                         source=builds['actors']['rank_3']['source'],opponent=builds['actors'][row['opponent']])
            if path.exists():
                envelope=campaign.read(path)
                if envelope['plan']!=binding['plan'] or campaign.read(directory/'CLAIM.json')!=binding:
                    raise ValueError('retained trajectory binding differs')
            else:
                if (directory/'CLAIM.json').exists():raise RuntimeError('unknown spent trajectory claim retained')
                campaign.immutable(directory/'CLAIM.json',binding);started=time.monotonic()
                def guard():
                    if time.monotonic()-started>120:raise RuntimeError('declared training-game wall cap reached')
                played=games.play(row,color,builds['actors']['rank_3'],builds['actors'][row['opponent']],
                    builds['builds']['rank_3'],builds['builds'][row['opponent']],decision_guard=guard)
                envelope=dict(plan=binding['plan'],game_id=identifier,seed=plan['seed'],split=row['split'],
                    opponent_name=row['opponent'],candidate=builds['actors']['rank_3'],
                    opponent=builds['actors'][row['opponent']],root=dict(source_kind='fresh-generated',
                    root_group_id=row['root_group_id'],transcript=row['transcript']),game=played,
                    generation_wall_seconds=time.monotonic()-started)
                campaign.immutable(path,envelope)
            references.append(campaign.record(path))
            if envelope['game']['failure'] is not None:
                failure=dict(game=campaign.record(path),details=envelope['game'].get('failure_details'),
                             message=envelope['game']['failure']);break
            games.training_trajectory(envelope);envelopes.append(envelope)
            campaign.atomic(output/'PROGRESS.json',dict(games_completed=len(envelopes),total_games=1024))
        if failure:break
    passed=failure is None and len(envelopes)==1024
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.trajectory-exposure',
        bank=plan['bank'],game_receipts=references,recipe=campaign.record(plan_path),
        training_eligible=False,before_next_fresh_bank=True,includes_partial=failure is not None))
    if passed:
        compact=b''.join((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode() for row in envelopes)
        campaign.immutable(output/'envelopes.jsonl',compact)
        exported=games.export_training(output/'envelopes.jsonl',output/'games.jsonl')
        if exported['split_root_groups']!={'train':384,'validation':128}:
            raise ValueError('expanded trajectory export allocation differs')
    value=dict(schema=SCHEMA+'.generation',passed=passed,games=len(envelopes),failure=failure,
        plan=campaign.record(plan_path),bank=plan['bank'],new_games=len(references),
        trajectories=campaign.record(output/'games.jsonl') if passed else None,
        strength_eligible=False,outcomes_as_targets=False)
    campaign.immutable(output/'RESULT.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['draw','build','generate']);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();result={'draw':draw,'build':build,'generate':generate}[args.stage](args.plan)
    print(json.dumps(dict(passed=result['passed'])))


if __name__=='__main__':main()
