"""Use the retained serial identity and archive protocol for live-eight windows."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_v2 as retained


def config(root):
    value=c.read(Path(root)/'campaign.json')
    state=c.read(value['campaign_current'])
    if value['activation']!=state['live_eight_activation'] or value['control']['source']['sha256']!=c.BASELINE:
        raise PermissionError('live campaign/activation identity differs')
    c.activation(state)
    return value


def validate_window(plan,submission,snapshot):
    user=snapshot['user']
    if user['agentId']!=submission['agent_id'] or user.get('testSessionHandle')!=submission['test_session_handle']:
        raise ValueError('active source/session changed')
    games=[]
    for game in snapshot['battles']:
        focus=[p for p in game['players'] if p.get('playerAgentId')==submission['agent_id']]
        if len(focus)!=1 or focus[0]['submissionId']!=submission['submission_id'] or focus[0].get('testSessionHandle')!=submission['test_session_handle']:
            raise ValueError('mixed source window')
        games.append(game)
    ids=[g['gameId'] for g in games]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate game')
    # The chronological first ninety are the calibration. Later games are
    # retained separately; neither rank nor outcome chooses this boundary.
    games.sort(key=lambda g:g['gameId'])
    initial=games[:plan['expected_games']]
    complete=user.get('percentage')==100 and len(initial)==plan['expected_games'] and all(g.get('done') is True for g in initial)
    return complete,[g['gameId'] for g in initial]


def adapter(root):
    config(root)
    spec=importlib.util.spec_from_file_location('_live_eight_identity_protocol',retained.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.config=config;module.SCHEMA=c.SCHEMA;module.validate_window=validate_window
    def source_guard(r,sha):
        state=c.read(config(r)['campaign_current'])
        c.banned(state,c.read(c.verify(state['closures'])),sha)
        if state.get('incident') and sha!=c.BASELINE:raise PermissionError('incumbent restoration required')
    module.guard_source=source_guard
    return module


def register_candidate(base,source,usage):
    base=Path(base);state=c.read(base/'CURRENT.json');source=c.record(source)
    c.guard(base,'admit',state['owner_thread_id'],usage,source=source['sha256'])
    with c.locked(base):
        state=c.read(base/'CURRENT.json')
        state['live_eight_versions'].setdefault(source['sha256'],dict(source=source,admitted_at=c.now().isoformat()))
        c.atomic(base/'CURRENT.json',state)
    return source


def declare(base,candidate,question,usage,kind='exploration',coverage=None):
    base=Path(base);state=c.read(base/'CURRENT.json');root=base/'live-eight-v1'
    c.guard(base,'freeze',state['owner_thread_id'],usage)
    if not question.strip():raise ValueError('frozen research question required')
    if any(not x.get('all90_archived') for x in state.get('live_eight_known_windows',{}).values()):
        raise PermissionError('archive prior windows before declaration')
    for block in state.get('live_eight_blocks',[]):
        prior=c.read(c.verify(block['plan']))
        if any(not (root/'attempts'/x['attempt_id']/'result.json').exists() for x in prior['slots']):
            raise PermissionError('finish the existing declared block before another')
    if kind not in ('exploration','coverage','qualification','formal','restoration'):
        raise ValueError('unsupported block purpose')
    if kind=='coverage':
        if not coverage or c.read(c.verify(coverage)).get('sufficient') is not False:
            raise PermissionError('coverage extension requires a bound inadequate-coverage assessment')
    order={'exploration':'BCCB','coverage':'BC','qualification':'BCCBBCCB','formal':'CCC','restoration':'B'}[kind]
    purpose='exploration' if kind=='coverage' else kind
    candidate=state['incumbent'] if kind=='restoration' else c.record(candidate)
    if candidate['sha256']!=c.BASELINE and candidate['sha256'] not in state['live_eight_versions']:
        raise PermissionError('register exact experimental source before declaring it')
    if kind=='coverage' and any(b.get('candidate_sha256')==candidate['sha256'] and b.get('kind')=='coverage' for b in state['live_eight_blocks']):
        raise PermissionError('one coverage extension per source')
    number=len(state['live_eight_blocks'])+1;identifier=f'live-eight-{number:03d}'
    admissions={}
    for ref in (state['incumbent'],candidate):
        sha=ref['sha256'];evidence={}
        if sha!=c.BASELINE:evidence['mechanical']=state['frozen_sources'][sha]['admission']
        admissions[sha]=c.immutable(root/'admissions'/sha/'live.json',dict(schema=c.SCHEMA+'.exploratory-admission',source=ref,evidence=evidence,offline_strength_veto=False,promotion_eligible=False))
    slots=[]
    for index,arm in enumerate(order,1):
        ref=state['incumbent'] if arm=='B' else candidate
        slots.append(dict(attempt_id=f'{identifier}-slot-{index:02d}',arm=arm,source=ref,admission=admissions[ref['sha256']],purpose=purpose))
    packet=c.immutable(root/'packets'/identifier/'plan.json',dict(schema=c.SCHEMA+'.block',packet_id=identifier,kind=kind,
        question=question,candidate_sha256=candidate['sha256'],slots=slots,primary_roster=state['live_eight_primary_roster'],
        activation=state['live_eight_activation'],training_designated=purpose=='exploration',created_at=c.now().isoformat(),
        calibration_boundary='chronological first90matching submission games; archive later games separately',coverage=coverage))
    with c.locked(base):
        state=c.read(base/'CURRENT.json')
        if kind=='formal':
            if candidate['sha256'] in state.get('live_eight_formal_sources',{}):raise PermissionError('formal block already declared')
            state.setdefault('live_eight_formal_sources',{})[candidate['sha256']]=packet
        for slot in slots:state['live_eight_declared_windows'][slot['attempt_id']]=dict(source_sha256=slot['source']['sha256'],purpose=purpose,plan=packet,block=packet)
        state['live_eight_blocks'].append(dict(id=identifier,kind=kind,candidate_sha256=candidate['sha256'],plan=packet))
        c.atomic(base/'CURRENT.json',state)
    return packet


def operate(base,operation,usage,aid=None,registry=None,copyback=None):
    base=Path(base);root=base/'live-eight-v1';state=c.read(base/'CURRENT.json');m=adapter(root)
    action={'prepare':'observe','lease':'observe','claim':'claim','intent':'submit','attest':'observe','observe':'observe','close':'archive'}[operation]
    if operation=='prepare':
        next_slot=m.next_slot(root)
        if next_slot is None:raise ValueError('no pending declared slot')
        aid=next_slot[1]['attempt_id']
    if aid:
        declared=state['live_eight_declared_windows'][aid];sha=declared['source_sha256']
    else:sha=None
    c.guard(base,action,state['owner_thread_id'],usage,source=sha,attempt=aid)
    if operation=='prepare':return m.prepare(root,registry or config(root)['registry']['path'])
    if operation=='lease':return m.lease(root,aid)
    if operation=='claim':
        value=m.claim(root,aid,copyback)
        with c.locked(base):
            state=c.read(base/'CURRENT.json');state['active_claims'][aid]=dict(source_sha256=sha,claim=c.record(root/'attempts'/aid/'claim.json'));c.atomic(base/'CURRENT.json',state)
        return value
    if operation=='intent':
        out,plan=m.attempt(root,aid)
        return c.immutable(out/'UI_SUBMIT_INTENT.json',dict(claim=c.record(out/'claim.json'),source_sha256=sha,exactly_one_action=True,created_at=c.now().isoformat()))
    if operation=='attest':
        value=m.attest(root,aid)
        with c.locked(base):
            state=c.read(base/'CURRENT.json');ref=c.record(root/'attempts'/aid/'submission.json')
            state['live_eight_declared_windows'][aid]['attested']=ref
            state['live_eight_known_windows'][aid]=dict(source_sha256=sha,submission=ref,all90_archived=False)
            state['active_claims'].pop(aid,None);state['live_eight_active_attempt']=aid;c.atomic(base/'CURRENT.json',state)
        return value
    if operation=='observe':return m.observe(root,aid)
    if operation=='close':return close(base,aid)


def assess_block(base,packet_path,output):
    base=Path(base);root=base/'live-eight-v1';packet=c.read(packet_path)
    primary=c.read(c.verify(packet['primary_roster']))['opponents']
    windows=[c.read(root/'attempts'/slot['attempt_id']/'result.json') for slot in packet['slots']]
    value=c.score_windows(windows,primary)
    value.update(packet=c.record(packet_path),source_sha256=packet['candidate_sha256'],
                 role='qualification' if packet['kind']=='qualification' else 'exploration')
    return c.immutable(output,value)


def close(base,aid):
    base=Path(base);root=base/'live-eight-v1';m=adapter(root);out,plan=m.attempt(root,aid)
    if (out/'result.json').exists():return c.read(out/'result.json')
    completed=c.read(out/'completed.json');submission=c.read(out/'submission.json')
    collector=m.c.arena.ArenaBatchCollector(repository=c.ROOT,data_root=out/'archive',maximum_workers=4,
        exclusion_registry_path=c.verify(plan['registry']),exclusion_registry_sha256=plan['registry']['sha256'],api=m.c.shared.PublicApi(maximum_attempts=4))
    binding=collector.bind_source(agent_id=submission['agent_id'],submission_id=submission['submission_id'],source_path=c.verify(plan['source']),expected_source_sha256=plan['source']['sha256'])
    result=collector.collect(run_id=aid,binding=binding,expected_games=90)
    path=Path(result['manifest_path']);path=path if path.is_absolute() else c.ROOT/path
    manifest=m.c.arena.validate_export_manifest(path,plan['registry']['sha256'])
    accepted={x['record']['game_id']:x['record'] for x in manifest['games'] if x['record']['status']=='accepted'}
    if not manifest['coverage']['full_window_accounted'] or not set(completed['game_ids'])<=accepted.keys():
        raise ValueError('all declared ninety games and supplemental exposures must be archived')
    block=c.read(c.verify(plan['packet']));slot=next(x for x in block['slots'] if x['attempt_id']==aid)
    games=[];traces=[]
    for gid,r in sorted(accepted.items()):
        op=r['operational'];focus=r['focus'];other=r['opponent']
        own=op['focus_status']!='ok';opfail=op['opponent_status']!='ok';clean=not own and not opfail
        games.append(dict(game_id=gid,won=focus['result']=='win',color=focus['player_id'],
            opponent_user_id=other['user_id'],opponent_submission_id=other['submission_id'],
            own_failure=own,opponent_failure=opfail,clean=clean,turns=r['replay']['valid_turns']))
        traces.append(dict(game_id=gid,turns=r['replay']['valid_turns']))
    value=dict(schema=c.SCHEMA+'.window',extension=c.SCHEMA,purpose=slot['purpose'],arm=slot['arm'],
        training_designated=block['training_designated'],complete=True,source_sha256=plan['source']['sha256'],
        source=plan['source'],submission=c.record(out/'submission.json'),manifest=c.record(path),
        games=[g for g in games if g['game_id'] in completed['game_ids']],
        supplemental_games=[g for g in games if g['game_id'] not in completed['game_ids']],
        completed_rank=completed['completed_rank'],score=completed['score'],
        coverage=manifest['coverage'],status='complete',own_failures=sum(g['own_failure'] for g in games))
    exposure=c.immutable(out/'EXPOSURE.json',dict(schema=c.SCHEMA+'.played-exposure',games=traces,training_eligible=False))
    c.immutable(out/'result.json',value)
    with c.locked(base):
        state=c.read(base/'CURRENT.json');state['live_eight_known_windows'][aid]['all90_archived']=True
        state['live_eight_completed_attempts'].append(aid);state.setdefault('pending_exposures',[]).append(exposure)
        if value['own_failures']:
            entry=state.get('frozen_sources',{}).get(plan['source']['sha256'],{})
            entry['operationally_banned']=True
            semantic=entry.get('semantic_identity')
            if semantic and semantic not in state['focused_banned_semantic_identities']:state['focused_banned_semantic_identities'].append(semantic)
            incident=c.immutable(out/'INCIDENT.json',dict(source=plan['source'],own_failures=value['own_failures'],result=c.record(out/'result.json'),underlying_cause='unknown',restore_required=True))
            state['incident']=dict(receipt=incident)
        c.atomic(base/'CURRENT.json',state)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--usage',type=Path,required=True)
    p.add_argument('operation',choices=['prepare','lease','claim','intent','attest','observe','close'])
    p.add_argument('--attempt');p.add_argument('--registry');p.add_argument('--copyback');a=p.parse_args()
    value=operate(a.base,a.operation,a.usage,a.attempt,a.registry,a.copyback)
    print(json.dumps(value))


if __name__=='__main__':main()
