"""Extract only designated new live TRAIN parents for accepted teacher labels."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import gzip
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_data_v3 as retained
from tools import rank_two_focused_work_v1 as work
from tools import rank_two_live_eight_ancestry_v1 as ancestry
from tools import rank_two_focused_teacher_budget_v1 as teacher_support

rules,games,data=retained.rules,retained.games,retained.data
SCHEMA=c.SCHEMA+'.live-label-data'
CAMPAIGN_ID=retained.CAMPAIGN_ID


def check_current(plan,action,launch=False):
    state=work.checked(plan,action,launch=launch)
    if any(not x.get('all90_archived') for x in state.get('live_eight_known_windows',{}).values()):
        raise PermissionError('live authoritative game window is active')
    return state


def normalized_label(native,position,plan):
    return teacher_support.normalized_label(native,position,plan)


def verify_teacher_plan(plan):
    check_current(plan,'label')
    body={key:value for key,value in plan.items() if key not in ('bundle_sha256','command')}
    if hashlib.sha256(c.old.retained.canonical(body)).hexdigest()!=plan['bundle_sha256']:
        raise ValueError('teacher source bundle differs')
    if plan['nodes'] not in (256000,1000000) or plan['outcomes_as_targets'] is not False:
        raise ValueError('teacher labels must preserve unknown outcomes')
    for key in ('teacher','native_teacher','positions_file','games'):
        data.bound(plan[key])
    provenance=c.read(c.verify(plan['live_training_provenance']))
    if not provenance['only_predesignated_new_exploration'] or provenance['protected_games_read']:
        raise PermissionError('teacher live-data provenance differs')


def parents(window,reference,seed):
    if (window.get('extension')!=c.SCHEMA or window.get('purpose')!='exploration'
            or window.get('training_designated') is not True or window.get('complete') is not True):
        raise PermissionError('only new predesignated complete exploration can supply TRAIN')
    result=[]
    for game in window['games']:
        if not c.training_allowed(window,game):continue
        state=rules.ReplayState();prefix=[];candidates=[];opening=None
        winner=game['color'] if game['won'] else 1-game['color']
        for turn,row in enumerate(game['turns']):
            action=row['action'] if isinstance(row,dict) else row
            mover=row.get('player_id',state.to_move) if isinstance(row,dict) else state.to_move
            if mover!=state.to_move:raise ValueError('live mover binding differs')
            if turn==4:opening=games.fingerprint(state)
            if state.winner is None and turn>=4:
                active=rules.encode_active(state);canonical=games.fingerprint(state)
                identifier='live-'+hashlib.sha256((str(game['game_id'])+':'+str(turn)).encode()).hexdigest()[:24]
                candidates.append(dict(position_id=identifier,game_id=str(game['game_id']),root_group_id='live-opening:'+opening,
                    group_id=identifier,source='live-eight-exploration',split='train',winner=winner,
                    mover=state.to_move,prefix='/'.join(prefix),edges=len(state.used_segments),
                    parent_active=list(active),canonical_state=canonical,canonical_features=data.feature_key(active),
                    provenance=reference,outcome_as_target=False,
                    position_selection_seed=int(hashlib.sha256((str(seed)+':'+canonical).encode()).hexdigest()[:8],16)))
            rules.apply_complete_turn(state,mover,action);prefix.append(action)
        if state.winner!=winner:raise ValueError('clean live trajectory/terminal outcome differs')
        early=[p for p in candidates if p['edges']<=16][:4]
        later=[p for p in candidates if p['edges']>16]
        if len(later)>8:later=[later[i*(len(later)-1)//7] for i in range(8)]
        result.extend(early+later)
    # A canonical/reflected parent is represented once, not as extra evidence.
    unique={}
    for row in result:unique.setdefault((row['canonical_state'],row['canonical_features']),row)
    return list(unique.values())


def prepare(plan_path):
    plan=c.read(plan_path);state=work.checked(plan,'label',launch=True)
    if plan['producer']!=c.record(__file__) or plan['teacher_nodes'] not in (256000,1000000):
        raise ValueError('teacher preparation binding differs')
    if plan['teacher_nodes']==1000000:
        declaration=c.read(c.verify(plan['deeper_teacher_declaration']))
        if declaration.get('intervention')!='matched-deeper-TRAIN-teacher' or declaration.get('teacher_nodes')!=1000000:
            raise PermissionError('deeper TRAIN labels require separate declaration')
    out=Path(plan['output']);all_rows=[]
    for ref in plan['windows']:
        window=c.read(c.verify(ref));all_rows.extend(parents(window,ref,plan['seed']))
    if not all_rows:raise ValueError('no eligible live training parents')
    root_ids=sorted({r['root_group_id'] for r in all_rows})
    if len(root_ids)>1536:raise ValueError('live roots exceed remaining cumulative training capacity')
    positions=c.immutable(out/'POSITIONS.jsonl',b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in all_rows))
    proof=c.immutable(out/'PROVENANCE.json',dict(passed=True,roots=root_ids,windows=plan['windows'],
        positions=positions,only_predesignated_new_exploration=True,protected_games_read=False,
        both_outcomes_eligible=True,outcomes_as_targets=False,teacher_nodes=plan['teacher_nodes'],
        native_teacher_seed_policy='accepted position-and-work binding; budget-dependent seeds are retained, no pure-budget causal claim',grouping='canonical opening after four complete turns',
        canonical_reflections_not_independent=True))
    result=dict(passed=True,positions=positions,provenance=proof,parent_count=len(all_rows))
    c.immutable(out/'RESULT.json',result)
    return result


def teacher_plan(plan_path,prepared,output):
    """Freeze a compatible source-bound plan for the versioned fixed worker pool."""
    plan=c.read(plan_path);check_current(plan,'label',launch=True)
    result=c.read(c.verify(prepared));provenance=c.read(c.verify(result['provenance']))
    positions=[json.loads(line) for line in c.verify(result['positions']).read_text().splitlines()]
    body=dict(plan['teacher_execution'],schema=SCHEMA+'.teacher-plan',positions=positions,
        positions_file=result['positions'],games=prepared,live_training_provenance=result['provenance'],
        producer=c.record(__file__),validator=c.record(retained.labels.__file__),lifecycle=c.record(work.__file__),
        nodes=plan['teacher_nodes'],outcomes_as_targets=False,
        campaign_current=plan['campaign_current'],activation=plan['activation'],owner_thread_id=plan['owner_thread_id'])
    if len(positions)>6144:raise ValueError('bounded teacher parent count exceeded')
    digest=hashlib.sha256(c.old.retained.canonical(body)).hexdigest()
    command=teacher_support.teacher_command(body,digest)
    return c.immutable(output,dict(body,bundle_sha256=digest,command=command))


def import_labels(teacher_plan_path,execution_path,retained_corpus,fresh_validation,output):
    """Keep old TRAIN only; append new live TRAIN and independent fresh VALIDATION."""
    plan=c.read(teacher_plan_path);check_current(plan,'label',launch=True);verify_teacher_plan(plan)
    execution=c.read(execution_path)
    if not execution.get('passed') or execution['plan']!=c.record(teacher_plan_path):raise ValueError('teacher execution incomplete')
    old=c.read(c.verify(retained_corpus));fresh=c.read(c.verify(fresh_validation))
    if (fresh.get('passed') is not True or fresh.get('roots')!=128 or fresh.get('early_roots',0)<100
            or fresh.get('protected_gameplay_reads') is not False):
        raise PermissionError('fresh independent validation receipt required')
    directory=Path(output).parent/'live-groups';directory.mkdir(parents=True,exist_ok=True)
    artifacts=[item for item in old['group_artifacts'] if item['split']=='train'];live_roots=set()
    for index,ref in enumerate(execution['parent_receipts']):
        receipt=c.read(c.verify(ref))
        if not receipt['success']:
            if not receipt['unsupported']:raise ValueError('unexpected teacher failure retained')
            continue
        native=json.loads(c.verify(receipt['output']).read_text());position=plan['positions'][index]
        group=normalized_label(native,position,plan);live_roots.add(group['root_group_id'])
        payload=(json.dumps(group,sort_keys=True,separators=(',',':'))+'\n').encode()
        path=directory/(position['position_id']+'.json.gz');c.immutable(path,gzip.compress(payload,mtime=0))
        artifacts.append(dict(group_id=group['group_id'],root_group_id=group['root_group_id'],split='train',
            edges=group['edges'],artifact=c.record(path)))
    artifacts.extend(fresh['group_artifacts'])
    train_roots={g['root_group_id'] for g in artifacts if g['split']=='train'}
    if not live_roots or len(train_roots)>2048:raise ValueError('live training root bounds differ')
    value=dict(old,group_artifacts=artifacts,live_root_ids=sorted(live_roots),
           independent_comparable_early_validation_root_groups=fresh['early_roots'],
           new_live_provenance=plan['live_training_provenance'],fresh_validation=fresh_validation,
           outcomes_as_targets=False,production_training_admitted=True)
    # Actual reflected-feature overlap is checked before admission, including
    # successor features, rather than assuming different root IDs suffice.
    train_features=set()
    for item in artifacts:
        if item['split']!='train':continue
        path=data.bound(item['artifact'])
        with gzip.open(path,'rt') as stream:group=json.load(stream)
        train_features.add(data.feature_key(group['parent_active']))
        train_features.update(data.feature_key(child['active']) for child in group['successors'])
    for item in fresh['group_artifacts']:
        path=data.bound(item['artifact'])
        with gzip.open(path,'rt') as stream:group=json.load(stream)
        features=[group['parent_active'],*[child['active'] for child in group['successors']]]
        if any(data.feature_key(active) in train_features for active in features):raise ValueError('TRAIN/reflected VALIDATION feature overlap')
    return c.immutable(output,value)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.plan)))
