#!/usr/bin/env python3
"""Mandatory-training campaign v2: immutable inputs and native smoke stages."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from collections import Counter

THREADS = {k: '1' for k in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
if __name__ == '__main__':
    os.environ.update(THREADS)
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
EXECUTING_SOURCE_BYTES = Path(__file__).read_bytes()
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import jacek_replay_features as features
from tools import jacek_replay_corpus as corpus
from tools import compact_value_bfm_openings as openings
from tools import compact_value_bfm_pilot_pipeline as legacy

START = '231b5ce0af9171670d68fe5fe022a67f86122cfd'
ID = 'compact-value-bfm-trained-v2'
OPPONENTS = ('rank_4', 'rank_4_jacek_hybrid', 'rank_4_fullturn_bfm',
             'challenger', 'jacek_native_bfm', 'neural_puct')
MODES = ('student-p1-vs-rank4', 'student-p2-vs-rank4', 'student-selfplay',
         'student-p1-vs-prior-incumbent', 'student-p2-vs-prior-incumbent',
         'incumbent-p1-vs-rank4', 'incumbent-p2-vs-rank4')
RATIOS = (4, 4, 4, 1, 1, 1, 1)
HEADER = 'position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n'
POLICY = {
    'score_target_strictly_above': 44.29750553418035,
    'mandatory_large_teacher_training': True, 'attempt_zero_allowed': False,
    'architecture': [6301, 12, 8, 1], 'biases': False, 'weight_bits': 3,
    'opening_policy': {'empty_train_fraction': .25, 'nonempty_edges': [8,12,20],
        'nonempty_split': [80,20], 'sample_limit': 20, 'retain_short_games': True,
        'overlap_exception': {'max_drawn_edges': 6, 'split': 'train',
            'allowed_roles': ['prior-train','live'], 'apply_to_each_successor': True},
        'validation_first_dedup': True, 'closure_rejection': 'whole-group'},
    'teacher_nodes': 64000, 'deep_nodes': 500000, 'deep_fraction': .25,
    'training': {'lambdas':[0,.10,.25], 'seeds':[20260907,20260908,20260909],
        'new_anchor_batch':[64,192], 'warmup_epochs':1, 'qat_epochs':4,
        'search_terminal_target':[.75,.25]},
    'pilot': {'games':2000, 'regret_reduction':.10, 'early_max_edges':12,
        'flip_margin':.005, 'comparable_groups':100, 'coverage':.80,
        'rank4_wins':105, 'rank4_games':200, 'failures':0},
    'full_games':10000,
    'multi_opponent': {'opponents':list(OPPONENTS), 'proxy_sources':True,
        'screen_pairs_each':32, 'confirmation_pairs_each':100,
        'improvement':.03, 'paired_ci_lower_above':0, 'max_point_regression':.05},
    'development': {'games':1000, 'wins':550, 'wins_per_color':265,
        'paired_ci_lower_above':.5, 'failures':0},
    'protected': {'independent_gates':2,'games_each':1000,'wins_each':527,
        'wins_per_color':260,'failures':0,'workers':4},
    'diagnostic_uploads_authorized':True, 'identical_source_reuploads':False,
    'success_requires': ['new_training','clean_90_live_games','calibration_complete',
                         'score_strictly_above_target'],
    'rank_reported_separately':True, 'smoke_qualification_eligible':False,
    'workers': {'generation_labels_max':8,'training_seeds_max':2,'timing':1},
    'source_limit_exclusive':95000,'source_reserve_target':2000,
    'recurring_automation':False,
}


def raw(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def record(path):
    path=Path(path).resolve()
    if path.is_symlink() or not path.is_file(): raise ValueError(f'not a regular file: {path}')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def once(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f'immutable artifact differs: {path}')
        return
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    try: os.link(temporary,path)
    except FileExistsError:
        if path.read_bytes()!=data: raise ValueError(f'concurrent artifact differs: {path}')
    finally: temporary.unlink()


def seal(path,body):
    body=json.loads(raw(body))
    value={**body,'body_sha256':hashlib.sha256(raw(body)).hexdigest()}
    once(path,raw(value)); return value


def read(path):
    value=json.loads(Path(path).read_bytes()); body=dict(value); digest=body.pop('body_sha256')
    if digest!=hashlib.sha256(raw(body)).hexdigest():
        # Initial v2 schedule encoder sorted integer histogram keys numerically.
        # Preserve its bytes and verify that exact known encoding on resume.
        if body.get('schema')==ID+'.schedule.v2':
            body['root_depth_counts']={int(k):v for k,v in body['root_depth_counts'].items()}
        if digest!=hashlib.sha256(raw(body)).hexdigest(): raise ValueError(f'changed receipt {path}')
    return value


def execution_source(root):
    path=root/'execution-sources'/(hashlib.sha256(EXECUTING_SOURCE_BYTES).hexdigest()+'.py')
    once(path,EXECUTING_SOURCE_BYTES)
    return record(path)


def verify(rec):
    if record(rec['path'])!=rec: raise ValueError(f'changed artifact {rec["path"]}')
    return Path(rec['path'])


def event(root,kind,details):
    directory=root/'ledger'; directory.mkdir(parents=True,exist_ok=True)
    with (root/'.ledger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=sorted(directory.glob('*.json')); previous=None
        for i,path in enumerate(rows):
            row=read(path)
            if row['ordinal']!=i or row['previous']!=previous: raise ValueError('ledger chain changed')
            previous=row['body_sha256']
        return seal(directory/f'{len(rows):06d}.json',{'schema':ID+'.event.v2',
            'ordinal':len(rows),'previous':previous,'kind':kind,'details':details,
            'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})


@contextlib.contextmanager
def lease(root):
    contract=root/'campaign.json'
    if contract.exists():
        document=read(contract)
        if 'heavy_stage_root' in document:
            parent=verify(document['parent_campaign']).parent
            if Path(document['heavy_stage_root']).resolve()!=parent:
                raise ValueError('phase heavy-stage authority changed')
            root=parent
    with (root/'.heavy-stage.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); yield


def copy_checked(source,destination,expected=None):
    source=Path(source); digest=sha(source)
    if expected is not None and digest!=expected: raise ValueError(f'input hash mismatch: {source}')
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists(): shutil.copy2(source,destination)
    if sha(destination)!=digest: raise ValueError(f'copy mismatch: {destination}')
    return record(destination)


def extract_live(previous,output):
    """Validate the corrected reference, then export only canonical boundaries."""
    old_repo=previous.parents[3]
    module_path=old_repo/'submissions/codingame/bots/compact_value_bfm/live_window.py'
    spec=importlib.util.spec_from_file_location('campaign_v2_prior_live',module_path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    archive=previous/'live/archive'; ref=archive/'live-window.corrected.reference.json'
    verified=module.verify_window_reference(ref,data_root=archive)
    if verified['body_sha256']!='5c4a668169e502aeb6918cc6d6bb23a944b1e42e0c71c4248903e8428aca9884':
        raise ValueError('unexpected prior live reference')
    receipt_path=module.resolve_path(verified['receipt']['path']); receipt=read(receipt_path)
    identity,exclusion,_=module.load_live_identity(
        module.resolve_path(receipt['submission_attestation']['path']),
        module.resolve_path(receipt['exclusion_binding']['path']))
    result=module.verify_generic_result({'manifest_path':receipt['collector_manifest']['path'],
        'manifest_sha256':receipt['collector_manifest']['sha256']},identity=identity,
        registry_sha256=exclusion['registry']['sha256'],expected_game_ids=receipt['game_ids'])
    values=set()
    empty_operational_games=0
    for row in result['records']:
        replay=row.get('replay',{}); rules=replay.get('rules_validation',{})
        if (replay.get('valid_transcript')=='' and replay.get('valid_turns')==[]
                and rules.get('valid_turns')==[] and rules.get('valid_turn_count')==0
                and rules.get('status') in ('incomplete','invalid')
                and row.get('operational',{}).get('classification')=='operationally-terminated'):
            values.add(openings.state_fingerprints(features.ReplayState())['canonical'])
            empty_operational_games+=1
        else:
            values.update(module._canonical_live_boundaries(row,identity=identity))
    return seal(output,{'schema':ID+'.live-fingerprints.v2','role':'live',
        'domain':legacy.STATE_FINGERPRINT_DOMAIN,'fingerprints':sorted(values),
        'reference':record(ref),'receipt':record(receipt_path),'extractor':record(module_path),
        'exact_games':len(result['records']),'empty_operational_games':empty_operational_games,'contains_transcripts':False,
        'contains_labels':False,'contains_metrics':False})


def freeze(root,previous,build):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'campaign.json').exists(): return read(root/'campaign.json')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=START:
        raise ValueError('initial freeze must start at verified main commit')
    source=previous/'inputs'; destination=root/'inputs'
    envelope=json.loads(next(source.glob('*.inputs.json')).read_bytes())
    # Copy opaque bundle bytes, including locked tests, without loading their labels.
    records=[]
    for path in sorted(source.rglob('*')):
        if path.is_file(): records.append(copy_checked(path,destination/path.relative_to(source)))
    bundle=destination/'training-bundle/bundle-manifest.json'
    if sha(bundle)!='58e4d8ca648e52d2df31d27f13faa805d45e7c4e0c4b87f43b146118b768c742':
        raise ValueError('anchor bundle changed')
    for item in envelope['training_bundle']['artifacts']:
        if sha(destination/item['route'])!=item['sha256']: raise ValueError('bundle member changed')
    inputs={name:record(destination/item['route']) for name,item in envelope['allowlisted_inputs'].items()
        if name in ('teacher_runtime','teacher_manifest','attempt_one_initial_checkpoint','attempt_zero_runtime','prior_runtime')}
    exclusions=[]
    def add(path,role):
        values=legacy._fingerprint_values(path,role); domain=legacy._fingerprint_domain(path,role)
        out=root/'exclusions'/f'{sha(path)}.json'
        seal(out,{'schema':ID+'.fingerprints.v2','role':role,'domain':domain,
            'fingerprints':sorted(values),'source':record(path)})
        exclusions.append(record(out))
    for name,item in envelope['protected_exclusions'].items():
        if 'fingerprint' in name: add(destination/item['route'],'mixed-development' if name.startswith('mixed') else 'protected')
    for name in ('prior-train','prior-validation'):
        add(destination/envelope['training_inputs'][name+'-fingerprints']['route'],name)
    for gate in ('a','b'):
        path=previous/f'dual-final/attempt-000/sanitized-exclusions/gate-{gate}.json'
        dest=root/'inputs/later-exclusions'/f'gate-{gate}.json'; copy_checked(path,dest); add(dest,'protected')
    live=root/'exclusions/live.json'; extract_live(previous,live); exclusions.append(record(live))
    opponents={}
    for name in OPPONENTS:
        src=REPO/f'submissions/codingame/bots/{name}'
        files={}
        for p in sorted(src.glob('*')):
            if p.is_file() and p.suffix in ('.cpp','.hpp','.txt','.json'):
                files[p.name]=copy_checked(p,root/'inputs/opponents'/name/p.name)
        if not files: raise ValueError(f'missing opponent {name}')
        opponents[name]=files
    sources={}
    for p in sorted(REPO.glob('tools/*')):
        if p.suffix in ('.py','.cpp','.hpp'): sources[str(p.relative_to(REPO))]=copy_checked(p,root/'sources'/p.relative_to(REPO))
    for name,expected in [('discrete_v3_deployment.cpp','add71c369052f232209d69c3b40b6bb459a2d7326ef15c5980377b1526fb8ea9'),
            ('submission.cpp','92313bb411f15f5ab0a40223e4ba0d64ce41d729976a28e941d597c0bed40f24')]:
        inputs[name]=copy_checked(REPO/'submissions/codingame/bots/compact_value_bfm'/name,root/'inputs/baselines'/name,expected)
    binaries={name:copy_checked(build/f'papersoccer_jacek_replay_{name}',root/'bin'/name)
        for name in ('continuations','search_teacher','rank4_position_teacher')}
    compiler=copy_checked(Path('/opt/homebrew/bin/g++-15').resolve(),root/'inputs/compiler/g++-15')
    plan=seal(root/'campaign.json',{'schema':ID+'.contract.v2','campaign_id':ID,
        'starting_commit':START,'policy':POLICY,'inputs':inputs,'copied_inputs':records,
        'bundle':record(bundle),'exclusions':exclusions,'opponents':opponents,
        'sources':sources,'binaries':binaries,'compiler':compiler,
        'compiler_version':subprocess.check_output(['/opt/homebrew/bin/g++-15','--version'],text=True),
        'prior_campaign':str(previous),'prior_ledger_mutated':False})
    event(root,'inputs-frozen',{'contract':record(root/'campaign.json')}); return plan


def fingerprints(state):
    ids={legacy.FEATURE_FINGERPRINT_DOMAIN:fast_feature_fingerprint(features.encode_active(state)).hex()}
    # Opening-state exclusions contain only playable states. Terminal successors
    # still undergo feature-domain isolation, just as in the maintained trainer.
    if state.winner is None: ids[legacy.STATE_FINGERPRINT_DOMAIN]=openings.state_fingerprints(state)['canonical']
    return ids


def exclusion_sets(plan):
    result={}
    for rec in plan['exclusions']:
        doc=read(verify(rec)); result.setdefault((doc['role'],doc['domain']),set()).update(doc['fingerprints'])
    return result


def rejection(state,split,excluded):
    ids=fingerprints(state)
    for (role,domain),values in excluded.items():
        if domain in ids and ids[domain] in values:
            if split=='train' and len(state.used_segments)<=6 and role in ('prior-train','live'): continue
            return role
    return None


def iter_successors(state,limit=100000):
    """Exhaustive rules-only DFS; any resource limit rejects the whole group."""
    mover=state.to_move; stack=[('',state)]; nodes=0; visited=set(); emitted=set()
    while stack:
        prefix,current=stack.pop()
        key=(current.ball,current.to_move,current.winner,frozenset(current.used_segments),frozenset(current.visit_count.items()))
        if key in visited: continue
        visited.add(key)
        for d in openings.legal_directions(current):
            nodes+=1
            if nodes>limit: raise ValueError('closure-resource-limit')
            child=openings.clone_state(current); features.apply_primitive(child,d); action=prefix+str(d)
            if child.winner is not None or child.to_move!=mover:
                key=(child.ball,child.to_move,child.winner,frozenset(child.used_segments),frozenset(child.visit_count.items()))
                if key not in emitted:
                    emitted.add(key); yield action,child
            else: stack.append((action,child))


def successors(state,limit=100000):
    return list(iter_successors(state,limit))


def preflight_group(state,split,excluded,validation_closure=None):
    reason=rejection(state,split,excluded)
    if reason: return {'eligible':False,'reason':'parent:'+reason}
    closure=[fingerprints(state)]; actions=[]; identities=set()
    try:
        for action,child in iter_successors(state):
            reason=rejection(child,split,excluded)
            if reason: return {'eligible':False,'reason':'successor:'+reason}
            closure.append(fingerprints(child)); actions.append(action)
            identities.add(corpus._mover_canonical_position_identity(child))
    except ValueError as error:
        if str(error)!='closure-resource-limit': raise
        return {'eligible':False,'reason':'closure-resource-limit'}
    if split=='train' and validation_closure is not None and any(
            value in validation_closure.get(domain,set()) for member in closure for domain,value in member.items()):
        return {'eligible':False,'reason':'validation-closure'}
    return {'eligible':True,'closure':closure,'legal_actions':sorted(actions),
        'successor_identities':sorted(identities)}


def fresh_root(edges,rng):
    for _ in range(10000):
        state=features.ReplayState(); turns=[]
        while len(state.used_segments)<edges and state.winner is None:
            mover=state.to_move; action=''
            while state.winner is None and state.to_move==mover:
                ds=openings.legal_directions(state)
                if not ds: break
                d=rng.choice(ds); features.apply_primitive(state,d); action+=str(d)
            turns.append(action)
        if state.winner is None and len(state.used_segments)==edges: return state,'/'.join(turns)
    raise ValueError('fresh root generation exhausted')


def schedule(root,phase,count):
    path=root/phase/'schedule.json'
    if path.exists(): return read(path)
    if count not in (64,128,2000,10000): raise ValueError('unapproved game count')
    plan=read(root/'campaign.json'); excluded=exclusion_sets(plan)
    rng=random.Random(f'{ID}:{phase}'); rows=[]; seen=set(); roots=[]
    actors=[mode for mode,ratio in zip(MODES,RATIOS) for _ in range(count*ratio//16)]
    rng.shuffle(actors)
    depths=[d for d in (0,8,12,20) for _ in range(count//4)]; rng.shuffle(depths)
    for i,(actor,depth) in enumerate(zip(actors,depths,strict=True)):
        if depth==0: state=features.ReplayState(); transcript=''; canonical=fingerprints(state)[legacy.STATE_FINGERPRINT_DOMAIN]
        else:
            while True:
                state,transcript=fresh_root(depth,rng); canonical=fingerprints(state)[legacy.STATE_FINGERPRINT_DOMAIN]
                if canonical not in seen and not rejection(state,'validation',excluded): break
            seen.add(canonical)
        roots.append((i,state,transcript,canonical,depth,actor))
    nonempty=sorted((r for r in roots if r[4]),key=lambda r:r[3]); nval=round(len(nonempty)*.2)
    validation={r[3] for r in nonempty[:nval]}
    for i,state,transcript,canonical,depth,actor in roots:
        rows.append({'ordinal':i,'actor_mode':actor,'base_seed':rng.getrandbits(63),
            'root_id':f'{phase}:root:{canonical}', 'canonical':canonical,'transcript':transcript,
            'drawn_edges':depth,'split':'validation' if canonical in validation else 'train'})
    result=seal(path,{'schema':ID+'.schedule.v2','phase':phase,'games':count,'rows':rows,
        'contract':record(root/'campaign.json'),'actor_counts':dict(Counter(actors)),
        'root_depth_counts':dict(Counter(depths)),'validation_roots':nval})
    event(root,'schedule-frozen',{'schedule':record(path)}); return result


def command_receipt(directory,command,inputs,stdin=None):
    directory.mkdir(parents=True,exist_ok=True); receipt=directory/'receipt.json'
    if receipt.exists():
        r=read(receipt)
        if r['command']!=command or r['inputs']!=inputs: raise ValueError('resume inputs changed')
        for output in r['outputs']: verify(output)
        return r
    once(directory/'launch.json',raw({'command':command,'inputs':inputs}))
    started=time.monotonic()
    with (directory/'stderr.log').open('wb') as err, (directory/'stdout.log').open('wb') as out:
        p=subprocess.run(command,input=stdin,stdout=out,stderr=err,env={**os.environ,**THREADS})
    if p.returncode: raise RuntimeError(f'native command failed ({p.returncode}): {directory}/stderr.log')
    outputs=[record(p) for p in directory.iterdir() if p.name not in ('receipt.json','launch.json') and p.is_file()]
    return seal(receipt,{'schema':ID+'.command.v2','command':command,'inputs':inputs,
        'outputs':outputs,'seconds':time.monotonic()-started,'returncode':p.returncode})


def run_games(root,phase,count,workers):
    plan=read(root/'campaign.json'); scheduled=schedule(root,phase,count)
    for rec in plan['binaries'].values(): verify(rec)
    teacher=verify(plan['inputs']['teacher_runtime']); student=verify(plan['inputs']['attempt_zero_runtime']); prior=verify(plan['inputs']['prior_runtime'])
    def run(row):
        directory=root/phase/'games'/f'{row["ordinal"]:05d}'; directory.mkdir(parents=True,exist_ok=True)
        roots=directory/'root.tsv'; gameplan=directory/'plan.tsv'
        once(roots,('group_id\tsource\twinner\ttranscript\n'+f'{row["root_id"]}\trules-generated\t0\t{row["transcript"] or "-"}\n').encode())
        once(gameplan,('game_ordinal\tactor_mode\tbase_seed\n'+f'{row["ordinal"]}\t{row["actor_mode"]}\t{row["base_seed"]}\n').encode())
        cmd=[plan['binaries']['continuations']['path'],'--input',str(roots),'--output',str(directory/'games.tsv'),
            '--manifest',str(directory/'games.json'),'--model',str(teacher),'--runner-up-model',str(teacher),
            '--selfsearch-plan',str(gameplan),'--campaign-id',ID,'--games','1',
            '--candidate-tree-nodes','2000','--actor-nodes','16000','--jacek-nn-nodes','64000',
            '--candidate-exploration','0.5','--candidate-fpu','0.5','--max-turns','320',
            '--root-policy','fresh-complete-v2']
        if row['actor_mode'].startswith('student'):
            cmd+=['--compact-student-runtime',str(student),'--compact-prior-runtime',str(prior)]
        receipt=command_receipt(directory,cmd,{'schedule':record(root/phase/'schedule.json'),
            'binary':plan['binaries']['continuations'],'teacher':record(teacher),'student':record(student),'prior':record(prior)})
        data=json.loads((directory/'games.json').read_bytes()); game=data['rows'][0]
        if data['successful_games']!=1 or game['prefix_turns']!=len(row['transcript'].split('/'))*(bool(row['transcript'])):
            raise ValueError('native root prefix changed')
        if game['root_group_id']!=row['root_id']: raise ValueError('native root assignment changed')
        fields=(directory/'games.tsv').read_text().splitlines()[1].split('\t')
        if len(fields)!=4 or fields[0]!=row['root_id'] or int(fields[2])!=game['winner']: raise ValueError('native TSV identity changed')
        game['transcript']=fields[3]
        if hashlib.sha256(fields[3].encode()).hexdigest()!=game['transcript_sha256']: raise ValueError('native transcript hash changed')
        actions=game['transcript'].split('/'); state=features.ReplayState()
        if '/'.join(actions[:game['prefix_turns']])!=row['transcript']: raise ValueError('native root transcript changed')
        for action in actions: features.apply_complete_turn(state,state.to_move,action)
        if state.winner!=game['winner']: raise ValueError('native terminal result changed')
        return {**row, 'game':game,'receipt':record(directory/'receipt.json'),'seconds':receipt['seconds']}
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(run,scheduled['rows']))
    path=root/phase/'games.json'
    if path.exists(): return read(path)
    result=seal(path,{'schema':ID+'.games.v2','rows':rows,'games':len(rows),
        'elapsed_seconds':time.monotonic()-started,'workers':workers,
        'unique_trajectories':len({r['game']['transcript'] for r in rows}),
        'short_completed_games':sum(len(r['game']['transcript'].split('/'))-r['game']['prefix_turns']<20 for r in rows)})
    event(root,'games-completed',{'games':record(path)}); return result


def fast_feature_fingerprint(active):
    """Same canonical bytes as corpus; input must already pass shard validation."""
    import numpy as np
    if not hasattr(fast_feature_fingerprint,'maps'):
        maps=[]
        for vertices,edges in ((features.REFLECTED_VERTICES,features.REFLECTED_EDGES),
                (features.ROTATED_VERTICES,features.ROTATED_EDGES)):
            lookup=list(edges)+[features.EDGE_COUNT+vertices[v]*features.VERTEX_CATEGORIES+c
                for v in range(features.VERTEX_COUNT) for c in range(features.VERTEX_CATEGORIES)]
            maps.append(np.asarray(lookup,dtype='<u2'))
        maps.append(maps[0][maps[1]])
        fast_feature_fingerprint.maps=maps
    a=np.asarray(active,dtype='<u2')
    variants=[a.tobytes()]+[np.sort(lookup[a]).astype('<u2',copy=False).tobytes()
                           for lookup in fast_feature_fingerprint.maps]
    return hashlib.sha256(min(variants)).digest()


def anchor_exclusions(root):
    output=root/'exclusions/anchor-derived.json'
    if output.exists(): return read(output)
    from tools import compact_value_bfm_train as trainer
    plan=read(root/'campaign.json'); bundle=trainer.FrozenBundle.load(verify(plan['bundle']))
    by_role={'prior-train':set(),'prior-validation':set()}; bindings=[]
    for split in ('train','validation'):
        routes=list(bundle.canonical_routes(split))
        if split=='train': routes+=list(bundle.arm_train_routes('search-target'))
        else: routes.append(bundle.common_adjudicator_route())
        for route in routes:
            dataset=trainer.load_shard(bundle,route); role='prior-'+split
            for row in range(len(dataset)):
                by_role[role].add(fast_feature_fingerprint(dataset.active_row(row)).hex())
            bindings.append({'route':route,'rows':len(dataset),'manifest_sha256':dataset.source_manifest_sha256,
                'npz_sha256':dataset.source_npz_sha256})
    result=seal(output,{'schema':ID+'.anchor-exclusions.v2','bundle':plan['bundle'],
        'bindings':bindings,'fingerprints':{k:sorted(v) for k,v in by_role.items()},
        'domain':legacy.FEATURE_FINGERPRINT_DOMAIN,'contains_labels':False,
        'contains_metrics':False,'contains_transcripts':False})
    event(root,'anchor-fingerprints-completed',{'artifact':record(output)}); return result


def mine_positions(root,phase):
    output=root/phase/'positions.json'
    if output.exists(): return read(output)
    plan=read(root/'campaign.json'); excluded=exclusion_sets(plan)
    anchors=anchor_exclusions(root)
    for role,values in anchors['fingerprints'].items(): excluded.setdefault((role,anchors['domain']),set()).update(values)
    games=read(root/phase/'games.json'); rows=[]; rejected=Counter(); seen_parents=set()
    validation_closure={d:set() for d in (legacy.STATE_FINGERPRINT_DOMAIN,legacy.FEATURE_FINGERPRINT_DOMAIN)}
    # All validation descendants are excluded, not only the sampled parents.
    ordered=sorted(games['rows'],key=lambda r:(r['split']!='validation',r['ordinal']))
    for item in ordered:
        print(json.dumps({'stage':'preflight','game':item['ordinal'],'split':item['split'],'retained_so_far':len(rows)}),flush=True)
        game=item['game']; state=features.ReplayState(); prefix=[]; candidates=[]
        actions=game['transcript'].split('/')
        for turn,action in enumerate(actions):
            if turn>=game['prefix_turns']:
                ids=fingerprints(state)
                reason=rejection(state,item['split'],excluded)
                if reason: rejected['parent:'+reason]+=1
                elif ids[legacy.STATE_FINGERPRINT_DOMAIN] not in seen_parents:
                    tactical=legacy._tactical_state_evidence(state)
                    candidates.append({'state':openings.clone_state(state),'turn':turn,'prefix':'/'.join(prefix),
                        'edges':len(state.used_segments),'ids':ids,'tactical':tactical})
            features.apply_complete_turn(state,state.to_move,action); prefix.append(action)
        # Absolute board progress defines early states. Short games retain available strata.
        early=[p for p in candidates if p['edges']<=12]
        tactical=sorted(candidates,key=lambda p:(-p['tactical']['target_goal_edges'],
            -p['tactical']['rebound_edges'],-p['tactical']['constrained_edges'],p['turn']))
        late=[p for p in candidates if p['edges']>=80]
        decisive=list(reversed(candidates))
        chosen=[]; used=set()
        for pool in (early,tactical,late,decisive):
            taken=0
            for p in pool:
                if p['turn'] in used: continue
                chosen.append(p); used.add(p['turn'])
                taken+=1
                if taken>=5: break
        for p in candidates:
            if len(chosen)>=20: break
            if p['turn'] not in used: chosen.append(p); used.add(p['turn'])
        for position in chosen[:20]:
            state=position['state']; ids=position['ids']; canonical=ids[legacy.STATE_FINGERPRINT_DOMAIN]
            if canonical in seen_parents: rejected['duplicate-parent']+=1; continue
            checked=preflight_group(state,item['split'],excluded,validation_closure)
            if not checked['eligible']: rejected[checked['reason']]+=1; continue
            closure=checked['closure']
            if item['split']=='validation':
                for values in closure:
                    for d,v in values.items(): validation_closure[d].add(v)
            seen_parents.add(canonical)
            rows.append({'position_id':f'{phase}:{item["ordinal"]}:{position["turn"]}',
                'root_group_id':item['root_id'],'group_id':game['game_id'],'source':ID,
                'split':item['split'],'winner':game['winner'],'mover':state.to_move,
                'prefix':position['prefix'],'drawn_edges':position['edges'],
                'canonical':canonical,'legal_actions':checked['legal_actions'],
                'successor_identities':checked['successor_identities'],
                'closure':closure,'tactical':position['tactical']})
    # The old producer requires exact quarter replacement; v2 rounds upward and records it.
    payload=HEADER+''.join('\t'.join(str(r[k]) for k in ('position_id','root_group_id','group_id','source','split','winner','mover','prefix'))+'\n' for r in rows)
    once(root/phase/'positions.tsv',payload.encode())
    result=seal(output,{'schema':ID+'.positions.v2','rows':rows,'rejections':dict(rejected),
        'counts':dict(Counter(r['split'] for r in rows)),
        'early_counts':dict(Counter(r['split'] for r in rows if r['drawn_edges']<=12)),
        'producer':execution_source(root),'games':record(root/phase/'games.json'),
        'positions':record(root/phase/'positions.tsv'),'full_closure_preflight':True,
        'validation_first':True,'sample_policy':'absolute-progress-tactical-short-retained'})
    event(root,'positions-preflighted',{'positions':record(output)}); return result


def eligible_positions(root,phase):
    """Extend preflight to inherited new-corpus validation, before any label work."""
    path=root/phase/'eligible-positions.json'
    if path.exists(): return read(path)
    from tools import compact_value_bfm_train as trainer
    plan=read(root/'campaign.json'); positions=mine_positions(root,phase)
    index=root/'exclusions/prior-search-validation.json'
    if not index.exists():
        bundle=trainer.FrozenBundle.load(verify(plan['bundle'])); values=set(); bindings=[]
        for stage in ('pilot','full'):
            route=bundle.routes[f'{stage}_search_manifests'][1]
            dataset=trainer.load_shard(bundle,route)
            if dataset.split!='validation': raise ValueError('inherited validation route changed')
            values.update(fast_feature_fingerprint(dataset.active_row(i)).hex() for i in range(len(dataset)))
            bindings.append({'route':route,'manifest_sha256':dataset.source_manifest_sha256,'npz_sha256':dataset.source_npz_sha256})
        seal(index,{'schema':ID+'.prior-search-validation.v2','fingerprints':sorted(values),
            'role':'prior-validation','domain':legacy.FEATURE_FINGERPRINT_DOMAIN,'bindings':bindings,
            'contains_transcripts':False,'contains_metrics':False,'contains_labels':False})
        event(root,'prior-validation-index-completed',{'artifact':record(index)})
    values=set(read(index)['fingerprints']); retained=[]; rejected=[]; duplicate=[]; seen_parents=set()
    for row in positions['rows']:
        if any(member.get(legacy.FEATURE_FINGERPRINT_DOMAIN) in values for member in row['closure']): rejected.append(row['position_id'])
        elif row['closure'][0][legacy.FEATURE_FINGERPRINT_DOMAIN] in seen_parents: duplicate.append(row['position_id'])
        else:
            seen_parents.add(row['closure'][0][legacy.FEATURE_FINGERPRINT_DOMAIN]); retained.append(row)
    result=seal(path,{'schema':ID+'.eligible-positions.v2','rows':retained,
        'preflight':record(root/phase/'positions.json'),'additional_validation':record(index),
        'rejected_whole_groups':rejected,'duplicate_parent_groups':duplicate,'counts':dict(Counter(r['split'] for r in retained)),
        'early_counts':dict(Counter(r['split'] for r in retained if r['drawn_edges']<=12))})
    event(root,'final-closure-preflight-completed',{'artifact':record(path)}); return result


def batched_scalar_predictor(runtime):
    """Vectorize rows while preserving deployment's scalar operation order."""
    import numpy as np
    from tools import compact_value_bfm_train as trainer
    arch,q,_,_=trainer.load_runtime(runtime)
    def predict(group):
        rows=group['successors']; first=np.empty((len(rows),arch.hidden_one),dtype=np.float32)
        for i,row in enumerate(rows):
            summed=np.sum(q.integer['w1'][np.asarray(row['active'],dtype=np.uint16)],axis=0,dtype=np.int32)
            first[i]=summed.astype(np.float32)*q.scales['w1']
        first=trainer.first_activation(first)
        second=np.zeros((len(rows),arch.hidden_two),dtype=np.float32)
        for h in range(arch.hidden_one):
            scaled=np.asarray(first[:,h]*q.scales['w2'],dtype=np.float32)
            second+=scaled[:,None]*q.integer['w2'][h].astype(np.float32)[None,:]
        second=trainer.second_activation(second); total=np.zeros(len(rows),dtype=np.float32)
        for h in range(arch.hidden_two):
            total+=np.asarray(second[:,h]*q.scales['w3'],dtype=np.float32)*np.float32(q.integer['w3'][h])
        return trainer.fast_tanh(total).astype(float).tolist()
    return predict


def label_positions(root,phase,workers):
    plan=read(root/'campaign.json'); positions=eligible_positions(root,phase); teacher=verify(plan['inputs']['teacher_runtime'])
    bundle=json.loads(verify(plan['bundle']).read_bytes())
    def batch(rows,directory,nodes):
        input_payload=(HEADER+''.join('\t'.join(str(r[k]) for k in ('position_id','root_group_id','group_id','source','split','winner','mover','prefix'))+'\n' for r in rows)).encode()
        directory.mkdir(parents=True,exist_ok=True); once(directory/'positions.tsv',input_payload)
        cmd=[str(verify(plan['binaries']['search_teacher'])),'--model',str(teacher),'--model-sha256',sha(teacher),
            '--campaign-id',ID,'--tree-nodes',str(nodes),'--time-ms','0','--max-actions','250',
            '--max-partial-paths','50000','--exploration','0.5','--fpu','0.5','--emit-action-groups',
            '--teacher-ranking-profile','standard-v1','--source-bundle-body-sha256',bundle['body_sha256']]
        command_receipt(directory,cmd,{'positions':record(directory/'positions.tsv'),
            'teacher':record(teacher),'binary':plan['binaries']['search_teacher']},stdin=input_payload)
        loaded=corpus.load_complete_turn_action_groups((directory/'stdout.log',))
        by_id={r['position_id']:r for r in rows}
        if len(loaded)!=len(rows): raise ValueError('teacher omitted parent rows')
        for value in loaded:
            g=value['group']; pos=by_id[g['source_binding']['position_id']]
            # Corpus validation independently replays every action; verify full native roster too.
            if not g['successors_exhaustive']: raise ValueError('non-exhaustive native label group')
            if sorted(s['successor_id'] for s in g['successors'])!=pos['successor_identities']:
                raise ValueError('native legal successor identities differ from independent full closure')
        return loaded
    def run_batches(rows,stage,nodes):
        chunks=[rows[i:i+8] for i in range(0,len(rows),8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            values=list(pool.map(lambda x:batch(x[1],root/phase/'labels'/stage/f'{x[0]:05d}',nodes),enumerate(chunks)))
        return [r for chunk in values for r in chunk]
    output=root/phase/'labels.json'
    if output.exists(): return read(output)
    started=time.monotonic(); shallow=run_batches(positions['rows'],'shallow',64000)
    def rank4_batch(item):
        ordinal,rows=item; directory=root/phase/'labels/rank4'/f'{ordinal:05d}'
        payload=(HEADER+''.join('\t'.join(str(r[k]) for k in ('position_id','root_group_id','group_id','source','split','winner','mover','prefix'))+'\n' for r in rows)).encode()
        directory.mkdir(parents=True,exist_ok=True); once(directory/'positions.tsv',payload)
        cmd=[str(verify(plan['binaries']['rank4_position_teacher'])),'--campaign-id',ID,'--nodes','32000','--time-ms','0']
        command_receipt(directory,cmd,{'positions':record(directory/'positions.tsv'),
            'binary':plan['binaries']['rank4_position_teacher']},stdin=payload)
        values=[json.loads(line) for line in (directory/'stdout.log').read_text().splitlines()]
        for value in values: corpus.sample_from_teacher_row(value)
        return values
    chunks=[positions['rows'][i:i+8] for i in range(0,len(positions['rows']),8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rank4_values=[v for values in pool.map(rank4_batch,enumerate(chunks)) for v in values]
    rank4={v['position_id']:v for v in rank4_values}
    score=batched_scalar_predictor(verify(plan['inputs']['attempt_zero_runtime']))
    evidence=[]
    for value in shallow:
        group=value['group']; position_id=group['source_binding']['position_id']
        regret=legacy.action_regret(group,score(group)); search=float(group['root_value'])
        reference=legacy._rank4_value(rank4[position_id]); outcome=1 if group['source_binding']['winner']==group['parent_mover'] else -1
        key=(-regret['regret'],-int(regret['action_disagreement']),-int((search>=0)!=(reference>=0)),
            -abs(search-reference),-int((search>=0)!=(outcome>=0)),min(abs(search),abs(reference)),position_id)
        evidence.append({'key':list(key),'position_id':position_id,'regret':regret,
            'search_value':search,'rank4_value':reference,'terminal_outcome':outcome})
    hardest=sorted(evidence,key=lambda e:e['key'])[:math.ceil(len(shallow)/4)]
    hard_ids={r['position_id'] for r in hardest}
    hard_rows=[r for r in positions['rows'] if r['position_id'] in hard_ids]
    seal(root/phase/'deep-selection.json',{'schema':ID+'.deep-selection.v2',
        'position_ids':sorted(hard_ids),'policy':'student-regret-action-rank4-terminal-disagreement',
        'evidence':evidence,'fraction':.25,'rounding':'ceil','shallow_rows':len(shallow)})
    deep=run_batches(hard_rows,'deep',500000)
    merged=corpus.merge_complete_turn_action_groups(shallow,deep)
    once(root/phase/'labels.jsonl',b''.join(raw(r) for r in merged))
    aggregate=corpus.build_complete_turn_successor_labels(merged)
    payload=raw(aggregate); aggregate_path=root/phase/(hashlib.sha256(payload).hexdigest()+'.successor-labels.json')
    once(aggregate_path,payload)
    result=seal(output,{'schema':ID+'.labels.v2','positions':record(root/phase/'eligible-positions.json'),
        'groups':len(merged),'deep_groups':len(deep),'successor_labels':record(aggregate_path),
        'merged':record(root/phase/'labels.jsonl'),'elapsed_seconds':time.monotonic()-started,
        'teacher':record(teacher),'producer':execution_source(root)})
    event(root,'labels-completed',{'labels':record(output)}); return result


def train_models(root,phase,*,smoke=False,ranking_weights=(0.0,.10,.25),qat_profile=None):
    if tuple(sorted(set(ranking_weights)))!=tuple(ranking_weights) or 0.0 not in ranking_weights or not set(ranking_weights)<={0.0,.10,.25}:
        raise ValueError('invalid ranking recipe roster')
    from tools import compact_value_bfm_train as trainer
    from tools import compact_value_bfm_teacher_training as adapter
    from tools import compact_value_bfm_seed_process_v2 as seed_process
    from tools import compact_value_bfm_intervention_v2 as intervention
    from tools import jacek_replay_train as packer
    plan=read(root/'campaign.json')
    mode=seed_process.executor_mode(plan)
    frozen_profile=intervention.expected_qat_profile(plan)
    profile=trainer.resolve_qat_profile(frozen_profile if qat_profile is None else qat_profile)
    if profile.name!=frozen_profile:
        raise ValueError('QAT profile differs from frozen phase')
    labels=read(root/phase/'labels.json')
    bundle=trainer.FrozenBundle.load(verify(plan['bundle']))
    from tools import compact_value_bfm_ranking_store as ranking_store
    store_index=ranking_store.build_store([verify(labels['merged'])],root/phase/'ranking-store',bundle)
    rankings=ranking_store.RankingStore(store_index,bundle).labels()
    positions=read(verify(labels['positions']))
    samples={'train':[],'validation':[]}; seen={'train':set(),'validation':set()}
    for split,groups in (('validation',rankings.validation),('train',rankings.train)):
        for group in groups:
            for sample in ranking_store.scalar_samples(group):
                active=tuple(sample.active)
                if active in seen[split]: continue
                seen[split].add(active); samples[split].append(sample)
    datasets={}; shard_records={}
    for split in ('train','validation'):
        npz,manifest,_=packer.write_csr_shard(root/phase/'shards',split,samples[split],
            provenance={'campaign':ID,'labels':labels['merged'],'positions':record(root/phase/'positions.json'),
                        'target_policy':corpus.target_policy_for_schema(corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA)})
        view=adapter._ExternalShardView(manifest,npz)
        datasets[split]=trainer.load_shard(view,manifest.name)
        shard_records[split]={'manifest':record(manifest),'npz':record(npz)}
    anchor,common,validation,routes=adapter._load_core_inputs(bundle)
    # Rows shared under the narrow early exception receive the new teacher label
    # only once: drop matching old anchor rows from the mixed-batch anchor pool.
    anchor,anchor_filter=seed_process.filter_early_anchor(anchor,positions['rows'])
    del positions
    audit={'schema':ID+'.training-input-audit.v2','bundle':plan['bundle'],
        'exclusion_index':record(root/'exclusions/anchor-derived.json'),
        'position_closure':record(root/phase/'positions.json'),'labels':record(root/phase/'labels.json'),'ranking_store':record(store_index),
        'shards':shard_records,'anchor_duplicates_removed':anchor_filter['removed_rows'],
        'smoke_qualification_eligible':False if smoke else None,'protected_tests_opened':False}
    audit=seal(root/phase/'training-input-audit.json',audit)
    inputs=trainer.TrainingInputs(new=datasets['train'],anchor=anchor,common_adjudicator=common,
        canonical_validation=validation,source_routes={**routes,'new':(shard_records['train']['manifest']['path'],)},
        paired_row_validation={'external_source_bound':True},split_isolation={'closure_audit':audit['body_sha256']},
        input_audit=audit,successor_rankings=rankings)
    architecture=trainer.ARCHITECTURES['capacity-12x8']; arm=trainer.ARMS['search-target']
    initial=verify(plan['inputs']['attempt_one_initial_checkpoint'])
    params=trainer.load_float_checkpoint(initial,architecture)
    baseline_path=root/phase/'initialization-measurement.json'
    if not baseline_path.exists():
        initial_metrics=trainer.evaluate_validation_pair(params,architecture,inputs=inputs,arm=arm)
        deployed_arch,deployed,_sel,_doc=trainer.load_runtime(verify(plan['inputs']['attempt_zero_runtime']))
        deployed_metrics=trainer.evaluate_validation_pair(params,deployed_arch,inputs=inputs,arm=arm,quantized=deployed)
        seal(baseline_path,{'schema':ID+'.initialization-measurement.v2','initial':record(initial),
            'deployed':plan['inputs']['attempt_zero_runtime'],'float_metrics':initial_metrics,
            'deployed_metrics':deployed_metrics,'equivalence_assumed':False})
    result_path=root/phase/'training.json'
    if result_path.exists(): return read(result_path)
    started=time.monotonic(); results=[]
    # Smoke exercises all loss recipes; one seed bounds its training cost.
    seeds=trainer.FIXED_SEEDS[:1] if smoke else trainer.FIXED_SEEDS
    process_spec=None;process_evidence=None
    with contextlib.ExitStack() as scopes:
        if mode=='spawn-v2':
            process_spec=seed_process.freeze_spec(root,phase,bundle,inputs,anchor_filter,
                ranking_weights,seeds,qat_profile=profile.name)
            process_executor=scopes.enter_context(seed_process.SpawnSeedExecutor(process_spec))
        else:
            execution=scopes.enter_context(trainer.native_thread_execution_scope())
        for weight in ranking_weights:
            directory=root/phase/'training'/f'lambda-{weight:.2f}'
            def run(seed):
                return trainer.train_seed_candidate(bundle,inputs,architecture,arm,seed,directory,
                    ranking_weight=weight,initial_checkpoint=initial,qat_profile=profile.name,resume=True,
                    _native_thread_execution=execution)
            if mode=='spawn-v2':
                receipts=process_executor.run_weight(weight)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    receipts=list(pool.map(run,seeds))
            for receipt in receipts:
                checkpoint=directory/receipt['float_checkpoint']['path']
                updated=trainer.load_float_checkpoint(checkpoint,architecture)
                update=trainer._parameter_update_evidence(params,updated)
                if any(not r['changed'] or r['l2_delta']<=0 or not math.isfinite(r['l2_delta']) for r in update.values()):
                    raise ValueError('all three master tensors must have finite nonzero trained updates')
                runtime=directory/receipt['quantized_runtime']['path']
                arch,quantized,_,_=trainer.load_runtime(runtime)
                initial_codes=trainer.quantize_fixed(params,arch,quantized.scales)
                changes=trainer._quantized_update_evidence(initial_codes,quantized)
                if not any(r['changed_codes'] for r in changes.values()): raise ValueError('export matches quantized initialization')
                output=directory/f'seed-{receipt["seed"]}.cpp'
                cmd=[sys.executable,str(REPO/'submissions/codingame/bots/compact_value_bfm/export_submission.py'),
                    '--runtime',str(runtime),'--render-output',str(output)]
                subprocess.run(cmd,check=True,env={**os.environ,**THREADS},stdout=subprocess.DEVNULL)
                source=output.read_bytes();source.decode('ascii')
                if len(source)>=95000: raise ValueError('export exceeds hard source limit')
                results.append({'weight':weight,'seed':receipt['seed'],'runtime':record(runtime),
                    'source':record(output),'float_checkpoint':record(checkpoint),
                    'master_updates':update,'quantized_changes_vs_initialization':changes,
                    'source_reserve':95000-len(source),'seed_receipt':receipt})
        if mode=='spawn-v2':
            evidence={'schema':ID+'.seed-process-execution.v2',
                'specification':record(process_spec),'start_method':'spawn','maximum_workers':2,
                'per_child_seed_threads':1,'numerical_threads_per_seed':1,
                'lambda_order':list(ranking_weights),'results':process_executor.evidence}
            process_evidence=root/phase/'seed-process-executions'/(hashlib.sha256(raw(evidence)).hexdigest()+'.json')
            seal(process_evidence,evidence)
    executor_evidence={} if process_spec is None else {'seed_process_spec':record(process_spec),
        'seed_process_execution':record(process_evidence)}
    result=seal(result_path,{'schema':ID+'.training.v2','results':results,
        'input_audit':record(root/phase/'training-input-audit.json'),'elapsed_seconds':time.monotonic()-started,
        'smoke':smoke,'mandatory_training_verified':True,'qualification_passed':False,
        'producer':execution_source(root),**executor_evidence})
    event(root,'training-completed',{'training':record(result_path)}); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    freeze_parser=sub.add_parser('freeze'); freeze_parser.add_argument('--previous',type=Path,required=True); freeze_parser.add_argument('--build',type=Path,required=True)
    g=sub.add_parser('games'); g.add_argument('--phase',default='smoke-064'); g.add_argument('--games',type=int,default=64); g.add_argument('--workers',type=int,default=8)
    for stage in ('positions','labels','anchor-exclusions','train-smoke'):
        g=sub.add_parser(stage); g.add_argument('--phase',default='smoke-064'); g.add_argument('--workers',type=int,default=8)
    a=parser.parse_args(); root=a.root.resolve()
    if a.command=='freeze': result=freeze(root,a.previous.resolve(),a.build.resolve())
    elif a.command=='games':
        if not 1<=a.workers<=8: raise ValueError('workers must be 1..8')
        with lease(root): result=run_games(root,a.phase,a.games,a.workers)
    elif a.command in ('positions','labels','anchor-exclusions','train-smoke'):
        if not 1<=a.workers<=8: raise ValueError('workers must be 1..8')
        with lease(root):
            if a.command=='anchor-exclusions': result=anchor_exclusions(root)
            elif a.command=='train-smoke': result=train_models(root,a.phase,smoke=True)
            else: result=mine_positions(root,a.phase) if a.command=='positions' else label_positions(root,a.phase,a.workers)
    print(json.dumps({'command':a.command,'body_sha256':result['body_sha256']},sort_keys=True))

if __name__=='__main__': main()
