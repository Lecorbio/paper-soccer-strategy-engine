#!/usr/bin/env python3
"""Versioned experiment-driven lifecycle, using the existing identity/archive protocol."""
from __future__ import annotations
import argparse, contextlib, datetime, importlib.util, json, re, sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rank_two_campaign_v3 as policy, rank_two_live_v2 as old, top_three_queue_v5 as q5

e = old.e
record = q5.record
ROOT = old.ROOT
SCHEMA = policy.SCHEMA


def read(path): return json.loads(Path(path).read_text())
def emit(path, value): e.emit(Path(path), value); return record(path)
def verify(ref): q5.queue.verify(ref); return Path(ref['path'])
def refs(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}: verify(value)
        else:
            for v in value.values(): refs(v)
    elif isinstance(value, list):
        for v in value: refs(v)


def config(root):
    m = read(Path(root)/'campaign.json')
    if m.get('schema') != SCHEMA or read(verify(m['policy'])) != policy.POLICY: raise ValueError('campaign policy identity')
    predecessor = policy.prior.load(verify(m['predecessor_manifest']))
    for k in ('control', 'rollback', 'opponents'):
        if m[k] != predecessor[k]: raise ValueError('frozen roster or rollback changed')
    refs(m['migration']); refs(m['producer']); return m


def migrate(previous, root, inventory):
    previous, root = Path(previous).resolve(), Path(root).resolve()
    if previous == root: raise ValueError('separate successor required')
    with old.lock(root):
        if (root/'campaign.json').exists(): return config(root)
        m = policy.prior.load(previous/'campaign.json'); spent, cancelled, qualifications = [], [], []
        for pp in old.packets(previous):
            packet = read(pp)
            if packet.get('qualification_source'): qualifications.append({'source': packet['qualification_source'], 'plan': record(pp)})
            slots = []
            for slot in packet['slots']:
                out = previous/'attempts'/slot['attempt_id']
                if (out/'claim.json').exists() and not (out/'result.json').exists():
                    raise ValueError('archive or classify existing claimed attempt before migration: '+slot['attempt_id'])
                if (out/'result.json').exists():
                    result = read(out/'result.json')
                    if result.get('status') != 'complete' or not result.get('coverage', {}).get('full_window_accounted'):
                        raise ValueError('incomplete historical archive')
                    spent.append(record(out/'result.json'))
                elif not (pp.parent/'result.json').exists(): slots.append(slot['attempt_id'])
            if slots: cancelled.append({'packet': record(pp), 'unclaimed_slots': slots})
        active = read(previous/'active.json')
        if active['source_sha256'] != m['control']['source']['sha256']:
            raise ValueError('restore incumbent with old protocol before migration')
        verify(inventory)
        migration = {'predecessor': str(previous), 'spent_attempts': spent, 'cancelled_unclaimed': cancelled,
            'qualification_history': qualifications, 'exposure_inventory': inventory,
            'active': record(previous/'active.json'), 'created_at': old.now(), 'historical_files_modified': False}
        mr = emit(root/'migration.json', migration)
        emit(root/'policy.json', policy.POLICY)
        value = {'schema': SCHEMA, 'policy': record(root/'policy.json'), 'predecessor_manifest': record(previous/'campaign.json'),
            **{k: m[k] for k in ('control', 'rollback', 'opponents', 'baseline_release', 'confirmation_roots', 'confirmation_games')},
            'migration': mr, 'producer': record(__file__), 'created_at': old.now()}
        emit(root/'campaign.json', value)
        old.atomic(root/'active.json', active)
        old.atomic(root/'incumbent.json', {'source': m['control']['source'], 'admission': None, 'baseline': True})
        emit(root/'admissions'/policy.BASELINE_SHA/'baseline.json', {'schema': SCHEMA, 'kind': 'verified-baseline',
            'source': m['control']['source'], 'manifest': record(root/'campaign.json'), 'promotion_eligible': False,
            'evidence': {'baseline_release': m['baseline_release']}})
        # Historical producers remain unchanged; their hash-bound snapshots remain readable.
        emit(previous/'MIGRATED.json', {'successor': str(root), 'migration': mr, 'action': 'No further unclaimed v2 submissions authorized.'})
        return value


def declaration(root, path):
    m = config(root); d = read(path); refs(d)
    for k in ('id', 'hypothesis', 'mechanism', 'target_states', 'evidence', 'intervention', 'tests', 'primary_cells', 'stopping_rules'):
        if not d.get(k): raise ValueError('experiment missing '+k)
    old.safe_id(d['id'])
    d.setdefault('cycle_id', 'cycle-001')
    old.safe_id(d['cycle_id'])
    existing=[read(p) for p in (Path(root)/'experiments').glob('*/declaration.json')]
    if sum(x.get('cycle_id')==d['cycle_id'] and x['id']!=d['id'] for x in existing)>=2: raise ValueError('two hypotheses per cycle')
    if d.get('baseline_sha256') != policy.BASELINE_SHA: raise ValueError('baseline identity')
    if d.get('variants', 1) > 2 or d.get('cycle_hypotheses', 1) > 2: raise ValueError('research breadth limit')
    for cell in d['primary_cells']:
        if set(cell) != {'opponent_agent_id', 'opponent_submission_id', 'color', 'state_sha256'} or cell['color'] not in (0, 1):
            raise ValueError('exact version/color/state primary cell required')
    if d['stopping_rules'].get('minimum_clean_per_source') != 8 or d['stopping_rules'].get('maximum_windows') != 10:
        raise ValueError('comparison stopping rule')
    d = {**d, 'schema': SCHEMA+'.experiment', 'manifest': record(Path(root)/'campaign.json')}
    return emit(Path(root)/'experiments'/d['id']/'declaration.json', d)


def admission(root, source, experiment, correctness, safety, targeted, screen, *, pilot=None, development=None, panel=None):
    root = Path(root); m = config(root); src = old.source_ref(source)
    dr = record(experiment); d = read(experiment); refs(d)
    if d.get('id') is None or Path(experiment).resolve()!=(root/'experiments'/d['id']/'declaration.json').resolve() or d.get('schema')!=SCHEMA+'.experiment' or d['manifest'] != record(root/'campaign.json'): raise ValueError('registered experiment campaign identity')
    existing=[read(p) for p in (root/'admissions').glob('*/exploratory.json')]
    if len({x['source']['sha256'] for x in existing if x['experiment']==dr and x['source']['sha256']!=src['sha256']})>=2: raise ValueError('two variants per hypothesis')
    paths = dict(correctness=correctness, safety=safety, targeted=targeted, screen=screen)
    values = {k: read(p) for k,p in paths.items()}
    for value in values.values(): refs(value)
    result = policy.admission(src, **values, roster=m['opponents'])
    if not result['passed']: raise ValueError('exploratory admission: '+','.join(result['reasons']))
    if any((pilot, development, panel)):
        if not all((pilot, development, panel)): raise ValueError('all promotion evidence required')
        more = {k: read(p) for k,p in dict(pilot=pilot, development=development, panel=panel).items()}
        for value in more.values(): refs(value)
        result = policy.promotion(src, **more)
        if not result['passed']: raise ValueError('promotion admission: '+','.join(result['reasons']))
        paths.update(pilot=pilot, development=development, panel=panel)
    value = {**result, 'schema': SCHEMA+'.admission', 'source': src, 'manifest': record(root/'campaign.json'),
             'experiment': dr, 'evidence': {k: record(p) for k,p in paths.items()}}
    return emit(root/'admissions'/src['sha256']/(result['kind']+'.json'), value)


def coverage_request(root, path):
    config(root); value = read(path); refs(value); old.safe_id(value['id'])
    if not value.get('question') or not value.get('cells') or type(value.get('requested_samples')) is not int or value['requested_samples'] <= 0:
        raise ValueError('named gap, exact cells and sample count required')
    for c in value['cells']:
        if set(c) != {'opponent_agent_id','opponent_submission_id','color','state_sha256'}: raise ValueError('coverage cells')
    return emit(Path(root)/'coverage'/value['id']/'request.json', {**value, 'manifest':record(Path(root)/'campaign.json'), 'maximum_windows': 2})


def validate_admission(root, ref):
    if ref is None: raise ValueError('registered admission required')
    root=Path(root);m=config(root);a=read(verify(ref));refs(a)
    kind=a.get('kind');source=a.get('source',{});digest=source.get('sha256','')
    filename='baseline.json' if kind=='verified-baseline' else kind+'.json' if kind in ('exploratory','promotion') else ''
    if not filename or Path(ref['path']).resolve()!=(root/'admissions'/digest/filename).resolve(): raise ValueError('registered admission required')
    if a.get('manifest')!=record(root/'campaign.json'): raise ValueError('admission campaign identity')
    if kind=='verified-baseline':
        if source!=m['control']['source']: raise ValueError('baseline admission source')
        return a
    if a.get('passed') is not True: raise ValueError('passing admission required')
    d=read(verify(a['experiment']))
    if Path(a['experiment']['path']).resolve()!=(root/'experiments'/d['id']/'declaration.json').resolve() or d['manifest']!=a['manifest']: raise ValueError('registered declaration required')
    ev={k:read(verify(v)) for k,v in a['evidence'].items()}
    result=policy.admission(source,ev['correctness'],ev['safety'],ev['targeted'],ev['screen'],m['opponents'])
    if not result['passed']: raise ValueError('exploratory evidence no longer passes')
    if kind=='promotion':
        result=policy.promotion(source,ev['pilot'],ev['development'],ev['panel'])
        if not result['passed'] or a.get('promotion_eligible') is not True: raise ValueError('promotion evidence no longer passes')
    elif a.get('promotion_eligible') is not False: raise ValueError('exploratory cannot promote')
    return a


def abort_failed(root):
    root=Path(root);config(root)
    with old.lock(root):
        if not (root/'STOP.json').exists(): raise ValueError('operational stop required')
        pp=packet_paths(root)[-1];plan=read(pp);retained=[];cancelled=[]
        if (pp.parent/'result.json').exists(): return read(pp.parent/'result.json')
        for slot in plan['slots']:
            out=root/'attempts'/slot['attempt_id']
            if (out/'result.json').exists(): retained.append(record(out/'result.json'))
            elif (out/'claim.json').exists(): raise ValueError('unknown or unresolved claimed slot cannot be cancelled')
            else: cancelled.append(slot['attempt_id'])
        result={'schema':SCHEMA,'plan':record(pp),'status':'aborted-own-failure','windows':retained,'cancelled_unclaimed':cancelled,
                'decision':{'decision':'retire','reason':'own-operational-failure'},'goal_achieved':False,'closed_at':old.now()}
        emit(pp.parent/'result.json',result);return result


def resolve_incident(root, classification):
    root=Path(root);config(root);c=read(classification);refs(c)
    with old.lock(root):
        stop=read(root/'STOP.json');pp=packet_paths(root)[-1];plan=read(pp);result=read(pp.parent/'result.json')
        if plan['kind']!='restoration' or result['own_failures'] or c.get('incident')!=record(root/'STOP.json') or not c.get('reason'):
            raise ValueError('clean restoration and bound explicit classification required')
        if read(root/'active.json')['source_sha256']!=incumbent(root)['source']['sha256']: raise ValueError('incumbent not restored')
        receipt=emit(root/'incidents'/record(root/'STOP.json')['sha256']/'resolved.json',{'incident':stop,'classification':record(classification),'restoration':record(pp.parent/'result.json')})
        (root/'STOP.json').unlink();return receipt


def packet_paths(root): return sorted((Path(root)/'packets').glob('*/plan.json'))
def incumbent(root): return read(Path(root)/'incumbent.json')


def packet(root, kind, evidence=None):
    root = Path(root); m = config(root)
    with old.lock(root):
        for p in packet_paths(root):
            if not (p.parent/'result.json').exists(): raise ValueError('unresolved packet')
        inc = incumbent(root); b = m['control']['source']; baseline = record(root/'admissions'/policy.BASELINE_SHA/'baseline.json')
        if kind != 'restoration' and (root/'RESTORE_REQUIRED.json').exists(): raise ValueError('incumbent restoration required')
        if kind != 'restoration' and (root/'STOP.json').exists(): raise ValueError('operational incident requires restoration and classification')
        er = record(evidence) if evidence else None; a = read(evidence) if evidence else None
        if a: refs(a)
        if kind in ('discovery','replication','qualification'): validate_admission(root,er)
        prior = [read(p) for p in packet_paths(root)]
        extra = {}
        if kind in ('discovery', 'replication', 'qualification'):
            if not a or a.get('manifest') != record(root/'campaign.json'): raise ValueError('bound admission required')
            src = a['source']; digest = src['sha256']
            if (root/'failed-sources'/digest).exists(): raise ValueError('source has own operational failure')
            if src['sha256'] == b['sha256']: raise ValueError('distinct challenger required')
            if kind == 'qualification':
                if a.get('kind') != 'promotion' or a.get('promotion_eligible') is not True: raise ValueError('promotion admission required')
                inherited = read(verify(m['migration']))['qualification_history']
                if any(x.get('qualification_source') == digest for x in prior) or any(x['source'] == digest for x in inherited):
                    raise ValueError('qualification block already spent')
                order = 'CCC'; extra['qualification_source'] = digest
            else:
                if a.get('kind') != 'exploratory': raise ValueError('exploratory admission required')
                matches = [x for x in prior if x.get('candidate_sha256') == digest and x['kind'] in ('discovery','replication')]
                if kind == 'discovery' and matches: raise ValueError('discovery already spent')
                if kind == 'replication':
                    if len(matches) != 1 or matches[0]['kind'] != 'discovery': raise ValueError('one replication only')
                    result = read(root/'packets'/matches[0]['packet_id']/'result.json')
                    if result['decision']['decision'] != 'replicate': raise ValueError('replication not warranted')
                order = ''.join(policy.POLICY[kind+'_packet'])
                if sum(len(x['slots']) for x in matches)+len(order)>10: raise ValueError('ten-window budget exhausted')
            extra.update(candidate_sha256=digest, experiment=a['experiment'])
            admissions = [baseline if role=='B' else er for role in order]
            sources = [b if role=='B' else src for role in order]
        elif kind == 'coverage':
            if not a or not a.get('id') or Path(evidence).resolve()!=(root/'coverage'/a['id']/'request.json').resolve() or a.get('manifest') != record(root/'campaign.json') or not a.get('question') or a.get('maximum_windows') != 2:
                raise ValueError('coverage request required; automatic baseline prohibited')
            matches = [x for x in prior if x.get('coverage_request') == er]
            if len(matches) >= 2: raise ValueError('coverage budget exhausted')
            if matches:
                value = read(root/'packets'/matches[-1]['packet_id']/'result.json')['coverage']
                if value['new_eligible'] <= 0 or value['total_eligible'] >= a['requested_samples']: raise ValueError('coverage extension not justified')
            order='B'; sources=[b]; admissions=[baseline]; extra['coverage_request']=er
        elif kind == 'restoration':
            active=read(root/'active.json')
            if active['source_sha256'] == inc['source']['sha256'] and not (root/'STOP.json').exists(): raise ValueError('incumbent already active')
            order='B'; sources=[inc['source']]; admissions=[inc['admission'] or baseline]
        else: raise ValueError('unsupported packet kind; no automatic baseline')
        pid=f'packet-{len(prior)+1:04d}'
        value={'schema':SCHEMA, 'packet_id':pid, 'kind':kind, 'manifest':record(root/'campaign.json'), 'created_at':old.now(),
            'slots':[{'attempt_id':f'{pid}-slot-{i+1:02d}', 'role':role, 'source':s, 'admission':r} for i,(role,s,r) in enumerate(zip(order,sources,admissions))], **extra}
        emit(root/'packets'/pid/'plan.json', value); return value


def adapter(root):
    """Isolated module binding reuses v2 protocol without changing imported v2 state."""
    config(root)
    spec=importlib.util.spec_from_file_location('_live_loss_identity', old.__file__)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.config=config; module.SCHEMA=SCHEMA
    def guard(r, sha):
        if (Path(r)/'STOP.json').exists() and sha!=incumbent(r)['source']['sha256']: raise ValueError('operational stop')
        if (Path(r)/'failed-sources'/sha).exists() and sha!=incumbent(r)['source']['sha256']: raise ValueError('source has own operational failure')
        if (Path(r)/'RESTORE_REQUIRED.json').exists() and sha != incumbent(r)['source']['sha256']:
            raise ValueError('restoration required')
    module.guard_source=guard
    return module


def operate(root, operation, aid=None, registry=None, copyback=None):
    root=Path(root); m=adapter(root)
    if operation=='prepare':
        item=m.next_slot(root)
        if item is None: raise ValueError('no declared experiment or coverage packet')
        a=validate_admission(root,item[1]['admission'])
        return m.prepare(root,registry)
    if operation=='claim': return m.claim(root,aid,copyback)
    if operation=='lease': return m.lease(root,aid)
    if operation in ('attest','observe','close'):
        value=getattr(m,operation)(root,aid)
        if operation=='close':
            out=root/'attempts'/aid; ap=read(out/'plan.json'); pp=read(verify(ap['packet']))
            if value['coverage']['focus_operational_failures']:
                old.atomic(root/'RESTORE_REQUIRED.json', {'attempt':aid,'reason':'own-operational-failure'})
                emit(root/'failed-sources'/value['source_sha256'],{'attempt':aid,'result':record(out/'result.json')})
            if pp['kind']=='qualification':
                games=read(out/'research-games.json')['games']
                emit(out/'exposure-transcripts.json',{'source_sha256':value['source_sha256'],'attempt_id':aid,'games':[{'game_id':g['game_id'],'transcript':g['transcript']} for g in games]})
        return value
    raise ValueError('operation')


def match_cell(game, cell, target_states):
    op=game['opponent']; focus=game['focus']
    if (op['agent_id']!=cell['opponent_agent_id'] or op['submission_id']!=cell['opponent_submission_id'] or focus['player_id']!=cell['color']): return None
    from tools import top_three_development_banks as banks
    state=banks.features.ReplayState()
    for index, action in enumerate(game['transcript'].split('/')):
        if not action: continue
        key=e.fingerprint(state)
        if key==cell['state_sha256'] and state.to_move==cell['color']:
            return {'turn_index':index,'action':action}
        banks.features.apply_complete_turn(state,state.to_move,action)
    return None


def summarize(root, plan):
    root=Path(root); declaration=read(verify(plan['experiment'])) if 'experiment' in plan else None
    request=read(verify(plan['coverage_request'])) if 'coverage_request' in plan else None
    cells=(declaration or request or {}).get('primary_cells', (request or {}).get('cells', []))
    buckets=[{'cell':c,'B':{'games':0,'score':0.,'windows':[]},'C':{'games':0,'score':0.,'windows':[]}} for c in cells]
    all_cells={}; seen=set(); failures=0; activation=[]; results=[]
    for slot in plan['slots']:
        out=root/'attempts'/slot['attempt_id'];result=read(out/'result.json')
        if result['source_sha256']!=slot['source']['sha256'] or result['status']!='complete' or not result['coverage']['full_window_accounted']: raise ValueError('incomplete bound window')
        results.append(record(out/'result.json'));failures+=result['coverage']['focus_operational_failures']
        games=read(out/'research-games.json')['games']
        for game in games:
            if game['game_id'] in seen: raise ValueError('duplicate spent game')
            seen.add(game['game_id'])
            if game['source_sha256']!=slot['source']['sha256']: raise ValueError('game source identity')
            if game['operational']['classification']!='clean': continue
            focus=game['focus']; score={'win':1.,'loss':0.,'draw':.5}[focus['result']]
            op=game['opponent'];key=(op['agent_id'],op['submission_id'],focus['player_id'])
            broad=all_cells.setdefault(key,{'B':{'games':0,'score':0.},'C':{'games':0,'score':0.}})[slot['role']]
            broad['games']+=1;broad['score']+=score
            for cell in buckets:
                hit=match_cell(game,cell['cell'],(declaration or {}).get('target_states',[]))
                if hit:
                    b=cell[slot['role']];b['games']+=1;b['score']+=score
                    if slot['attempt_id'] not in b['windows']:b['windows'].append(slot['attempt_id'])
                    activation.append({'game_id':game['game_id'],'attempt_id':slot['attempt_id'],'role':slot['role'],**hit})
    return {'primary_cells':buckets,'broader_cells':[{'opponent_agent_id':k[0],'opponent_submission_id':k[1],'color':k[2],**v} for k,v in sorted(all_cells.items())],
            'activation':activation,'own_failures':failures,'windows':results,'games':len(seen)}


def finish(root):
    root=Path(root);config(root)
    with old.lock(root):
        ps=packet_paths(root)
        if not ps: raise ValueError('no packet')
        pp=ps[-1];plan=read(pp)
        if (pp.parent/'result.json').exists(): return read(pp.parent/'result.json')
        summary=summarize(root,plan);kind=plan['kind']
        value={'schema':SCHEMA, 'plan':record(pp), 'closed_at':old.now(), **summary, 'goal_achieved':False}
        if kind in ('discovery','replication'):
            windows=[s['attempt_id'] for s in plan['slots'] if s['role']=='C']
            comparison=summary['primary_cells']
            if kind=='replication':
                discovery=next(read(p.parent/'result.json') for p in ps[:-1] if read(p).get('candidate_sha256')==plan['candidate_sha256'] and read(p)['kind']=='discovery')
                comparison=json.loads(json.dumps(comparison))
                for cell, earlier in zip(comparison,discovery['primary_cells']):
                    if cell['cell']!=earlier['cell']: raise ValueError('primary cells changed between packets')
                    for role in ('B','C'):
                        cell[role]['games']+=earlier[role]['games'];cell[role]['score']+=earlier[role]['score']
                        cell[role]['windows']+=earlier[role]['windows']
                        if role=='C':windows+=earlier[role]['windows']
                value['combined_primary_cells']=comparison
            value['decision']=policy.decision(kind,comparison,windows,summary['own_failures'])
            if read(root/'active.json')['source_sha256'] != incumbent(root)['source']['sha256']:
                old.atomic(root/'RESTORE_REQUIRED.json',{'packet':record(pp),'reason':'research-complete'})
        elif kind=='qualification':
            windows=[read(verify(r)) for r in summary['windows']]
            value['assessment']=policy.prior.assess_live_windows(windows,plan['qualification_source'],[s['attempt_id'] for s in plan['slots']])
        elif kind=='coverage':
            request=read(verify(plan['coverage_request']));total=len({x['game_id'] for x in summary['activation']});previous=0
            for p in ps[:-1]:
                if read(p).get('coverage_request')==plan['coverage_request']: previous+=read(p.parent/'result.json')['coverage']['new_eligible']
            value['coverage']={'new_eligible':total,'total_eligible':total+previous,'requested':request['requested_samples']}
        elif kind=='restoration':
            if summary['own_failures']: raise ValueError('restoration failure retained; intervention required')
            if read(root/'active.json')['source_sha256']!=incumbent(root)['source']['sha256']: raise ValueError('wrong restored source')
            if (root/'RESTORE_REQUIRED.json').exists():
                emit(pp.parent/'restoration-cleared.json',read(root/'RESTORE_REQUIRED.json'));(root/'RESTORE_REQUIRED.json').unlink()
        emit(pp.parent/'result.json',value);return value


def status(root):
    root=Path(root);m=adapter(root);s=m.status(root)
    if s['next_stage']=='new-packet': s['next_stage']='restore' if (root/'RESTORE_REQUIRED.json').exists() else 'await-experiment-or-coverage'
    s['automatic_baseline_allowed']=False
    s['recovery_minutes']=policy.cadence(s['next_stage'] in ('attest','observe','close'))
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('migrate');a.add_argument('--previous',type=Path,required=True);a.add_argument('--inventory',type=Path,required=True)
    for name in ('declare','coverage-request'):
        a=sub.add_parser(name);a.add_argument('--input',type=Path,required=True)
    a=sub.add_parser('admit')
    for k in ('source','experiment','correctness','safety','targeted','screen'):a.add_argument('--'+k,type=Path,required=True)
    for k in ('pilot','development','panel'):a.add_argument('--'+k,type=Path)
    a=sub.add_parser('packet');a.add_argument('--kind',choices=['discovery','replication','coverage','restoration','qualification'],required=True);a.add_argument('--evidence',type=Path)
    sub.add_parser('status');sub.add_parser('finish-packet');sub.add_parser('abort-failed-packet')
    a=sub.add_parser('resolve-incident');a.add_argument('--classification',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--registry',type=Path,required=True)
    for name in ('claim','lease','attest','observe','close'):
        a=sub.add_parser(name);a.add_argument('--attempt',required=True)
        if name=='claim':a.add_argument('--copyback',type=Path,required=True)
    a=p.parse_args();args=vars(a);cmd=args.pop('cmd');root=args.pop('root')
    if cmd=='migrate':v=migrate(args['previous'],root,record(args['inventory']))
    elif cmd=='declare':v=declaration(root,args['input'])
    elif cmd=='coverage-request':v=coverage_request(root,args['input'])
    elif cmd=='admit':v=admission(root,**args)
    elif cmd=='packet':v=packet(root,**args)
    elif cmd=='finish-packet':v=finish(root)
    elif cmd=='abort-failed-packet':v=abort_failed(root)
    elif cmd=='resolve-incident':v=resolve_incident(root,args['classification'])
    elif cmd=='status':v=status(root)
    else:
        if 'attempt'in args:args['aid']=args.pop('attempt')
        v=operate(root,cmd,**args)
    print(json.dumps(v,indent=2))
if __name__=='__main__':main()
