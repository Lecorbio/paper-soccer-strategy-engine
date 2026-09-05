#!/usr/bin/env python3
"""Source-bound live lifecycle and strict trained-v2 success assessment.

No command clicks Submit or writes to CodinGame. An operator performs the one
authorized UI action between start-submit and attest. New collector compatibility
records truthfully refer to this v2 authorization; historical ledgers and their
one-upload receipts are never modified or reused as v2 qualification evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_qualification as legacy
from submissions.codingame.bots.compact_value_bfm import live_window as collector
from submissions.codingame.bots.neural_puct import collect_live_replays as shared

TARGET=Decimal('44.29750553418035')
# Public ranking DTOs round/truncate to hundredths. A whole hundredth is a
# conservative envelope for either direction; near-target values need a future
# authoritative full-precision adapter, never a later favorable re-snapshot.
PUBLIC_SCORE_ERROR=Decimal('0.01')
LEADERBOARD_REQUEST=['paper-soccer',None,'global',{'active':False,'column':'','filter':''}]
LEADERBOARD_SERVICE=shared.REQUEST_SCHEMAS['leaderboard-v1']['service']
BATTLES_SERVICE=shared.REQUEST_SCHEMAS['agent-battles-v1']['service']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')


def positive(value,name):
    if type(value) is not int or value<=0:raise ValueError(name+' must be a positive integer')
    return value


def directory(root,source_sha):
    if not isinstance(source_sha,str) or legacy.SHA256_RE.fullmatch(source_sha) is None:
        raise ValueError('live source identity is not an exact SHA256')
    return Path(root).resolve()/'live'/source_sha


def fetch(output,service,payload):
    """Bounded read-only public API request, with raw HTTP evidence retained."""
    response=shared.PublicApi(timeout_seconds=20,maximum_attempts=1).post(service,payload)
    if response.status!=200:raise ValueError('live public API did not return HTTP200')
    decoded=json.loads(response.body)
    raw=Path(output)/'raw'/(hashlib.sha256(response.body).hexdigest()+'.json')
    campaign.once(raw,response.body)
    receipt={'schema':campaign.ID+'.live-http-snapshot.v2','service':service,'request':payload,
        'status':response.status,'headers':dict(response.headers),'raw':campaign.record(raw),'fetched_at_utc':now()}
    path=Path(output)/'http'/(hashlib.sha256(campaign.raw(receipt)).hexdigest()+'.json')
    campaign.seal(path,receipt)
    return campaign.record(path),decoded


def response(record,service,payload):
    receipt=campaign.read(campaign.verify(record))
    if (receipt['schema']!=campaign.ID+'.live-http-snapshot.v2' or receipt['service']!=service
            or receipt['request']!=payload or receipt['status']!=200):
        raise ValueError('live snapshot request identity changed')
    collector.parse_utc(receipt['fetched_at_utc'],'live fetch time')
    return json.loads(campaign.verify(receipt['raw']).read_bytes())


def user_row(payload,user_id,required=True):
    if not isinstance(payload,dict) or not isinstance(payload.get('users'),list):
        raise ValueError('leaderboard response has no users roster')
    rows=[row for row in payload['users'] if isinstance(row,dict) and
        isinstance(row.get('codingamer'),dict) and row['codingamer'].get('userId')==user_id]
    if len(rows)>1 or (required and len(rows)!=1):raise ValueError('leaderboard owner identity is not singular')
    return rows[0] if rows else None


def authorize(root,context,phase,registry_path,user_id):
    """Normal release path; useful diagnostic authority remains a separate scope."""
    from tools import compact_value_bfm_protected_v2 as protected
    positive(user_id,'CodinGame user ID')
    qualified=protected.validate(Path(root).resolve(),Path(context).resolve(),phase)
    if qualified.get('passed') is not True:raise ValueError('live release requires both actual protected gates')
    selected=qualified['selected'];source=campaign.verify(selected['source'])
    freeze=campaign.read(campaign.verify(qualified['freeze']))
    selected_bytes=source.read_bytes();selected_bytes.decode('ascii')
    if not 0<len(selected_bytes)<=93000:raise ValueError('live source lost the required reserve')
    collector.validate_id_only_registry(Path(registry_path))
    output=directory(root,campaign.sha(source))
    copied=campaign.copy_checked(source,output/'source.cpp')
    registry=campaign.copy_checked(registry_path,output/'excluded-game-ids.json')
    if (output/'authorization.json').exists():
        auth=load_authorization(root,source_sha=copied['sha256'])
        if auth['qualification']!=campaign.record(Path(context)/phase/'protected/assessment.json') or auth['user_id']!=user_id:
            raise ValueError('this source already has another live authorization')
        collector_authorization(output,auth)
        return auth
    if (output/'exclusion-binding.json').exists():
        binding,bound_registry=collector.validate_exclusion_binding(output/'exclusion-binding.json')
        if bound_registry.resolve()!=Path(registry['path']):raise ValueError('partially frozen live registry changed')
        created=binding['frozen_at_utc']
    else:
        created=now()
        collector.freeze_exclusion_binding(output/'exclusion-binding.json',registry_path=Path(registry['path']),frozen_at_utc=created)
    auth=campaign.seal(output/'authorization.json',{'schema':campaign.ID+'.live-authorization.v2',
        'mode':'qualified-release','root':str(Path(root).resolve()),'context':str(Path(context).resolve()),'phase':phase,
        'source':copied,'qualified_source':selected['source'],'runtime':selected['runtime'],'payload_sha256':selected['payload_sha256'],
        'candidate_commit':freeze['candidate_commit'],'qualification':campaign.record(Path(context)/phase/'protected/assessment.json'),
        'release_freeze':qualified['freeze'],'registry':registry,'exclusion_binding':campaign.record(output/'exclusion-binding.json'),
        'user_id':user_id,'created_at_utc':created,'one_submit_for_this_source':True,
        'historical_authorization_modified':False,'diagnostic_upload_policy_unchanged':True,'campaign_success':False})
    collector_authorization(output,auth)
    campaign.event(Path(root),'qualified-live-source-authorized',{'authorization':campaign.record(output/'authorization.json'),'source_sha256':copied['sha256']})
    return auth


def collector_authorization(output,auth):
    return campaign.seal(output/'collector-authorization.json',{'schema':legacy.UPLOAD_AUTH_SCHEMA,'namespace':'compact_value_bfm',
        'uploads_authorized':1,'rank4_replacement_authorized':False,'candidate_commit':auth['candidate_commit'],
        'candidate':{**auth['source'],'ascii':True},'v2_authorization':campaign.record(output/'authorization.json'),
        'purpose':'new truthful per-source identity adapter for the maintained live collector',
        'historical_authorization_modified':False})


def collector_submission(output,auth,submitted):
    collector_authorization(output,auth)
    return campaign.seal(output/'collector-submission.json',{'schema':legacy.UPLOAD_EVENT_SCHEMA,'namespace':'compact_value_bfm',
        'status':'submission-attested','submit_clicks':1,'authorization':legacy.artifact_reference(output/'collector-authorization.json',legacy.UPLOAD_AUTH_SCHEMA),
        'candidate_commit':auth['candidate_commit'],'source_sha256':auth['source']['sha256'],'source_bytes':auth['source']['bytes'],
        'agent_id':submitted['agent_id'],'submission_id':submitted['submission_id'],'submitted_at_utc':submitted['submitted_at_utc'],
        'v2_submission':campaign.record(output/'submission.json'),'historical_authorization_modified':False})


def load_authorization(root,source_sha):
    output=directory(root,source_sha);auth=campaign.read(output/'authorization.json')
    if (auth.get('schema')!=campaign.ID+'.live-authorization.v2' or auth.get('mode')!='qualified-release'
            or auth['root']!=str(Path(root).resolve()) or auth['source']['sha256']!=source_sha
            or auth.get('one_submit_for_this_source') is not True or auth.get('historical_authorization_modified') is not False):
        raise ValueError('live authorization scope changed')
    for key in ('source','qualified_source','runtime','qualification','release_freeze','registry','exclusion_binding'):campaign.verify(auth[key])
    if Path(auth['source']['path'])!=output/'source.cpp':raise ValueError('live source escaped immutable copy')
    positive(auth['user_id'],'owner ID')
    return auth


def revalidate_authorization(root,source_sha):
    from tools import compact_value_bfm_protected_v2 as protected
    auth=load_authorization(root,source_sha)
    context=Path(auth['context']);qualified=protected.validate(Path(root).resolve(),context,auth['phase'])
    expected_path=context/auth['phase']/'protected/assessment.json'
    if (auth['qualification']!=campaign.record(expected_path) or qualified.get('passed') is not True
            or qualified['freeze']!=auth['release_freeze'] or qualified['selected']['source']!=auth['qualified_source']
            or qualified['selected']['source']['sha256']!=source_sha or qualified['selected']['source']['bytes']!=auth['source']['bytes']
            or qualified['selected']['runtime']!=auth['runtime'] or qualified['selected']['payload_sha256']!=auth['payload_sha256']):
        raise ValueError('live authorization does not reproduce actual protected qualification')
    freeze=campaign.read(campaign.verify(qualified['freeze']))
    if auth['candidate_commit']!=freeze['candidate_commit']:
        raise ValueError('live authorization changed the qualified commit')
    return auth


def copyback(root,source_sha,captured_source,editor_url,editor_session):
    auth=load_authorization(root,source_sha);output=directory(root,source_sha)
    if (output/'submit-claim.json').exists():raise ValueError('source submit was already claimed')
    url=urlparse(editor_url)
    if url.scheme!='https' or url.hostname not in ('codingame.com','www.codingame.com') or url.path!='/ide/puzzle/paper-soccer':
        raise ValueError('copyback does not identify the Paper Soccer editor')
    if not isinstance(editor_session,str) or not editor_session:raise ValueError('fresh editor session is required')
    original=Path(captured_source)
    if original.is_symlink() or original.resolve() in (Path(auth['source']['path']),Path(auth['qualified_source']['path'])):
        raise ValueError('copyback must come from a separate regular editor capture')
    observed=campaign.record(original)
    if observed['sha256']!=source_sha or observed['bytes']!=auth['source']['bytes']:
        raise ValueError('live editor bytes differ from the qualified source')
    captured=campaign.copy_checked(original,output/'editor-copyback.cpp')
    return campaign.seal(output/'copyback.json',{'schema':campaign.ID+'.live-copyback.v2',
        'authorization':campaign.record(output/'authorization.json'),'captured_source':captured,
        'editor_url':editor_url,'editor_session':editor_session,'observed_at_utc':now()})


def start_submit(root,source_sha):
    auth=load_authorization(root,source_sha);output=directory(root,source_sha)
    if (output/'submit-claim.json').exists():raise ValueError('one submit already claimed; resolve identity without clicking again')
    revalidate_authorization(root,source_sha)
    copied=campaign.read(output/'copyback.json')
    if copied['authorization']!=campaign.record(output/'authorization.json') or campaign.verify(copied['captured_source']).read_bytes()!=campaign.verify(auth['source']).read_bytes():
        raise ValueError('fresh source copyback changed')
    before,payload=fetch(output,LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    prior=user_row(payload,auth['user_id'],required=False)
    battles=None
    if prior is not None:
        positive(prior['agentId'],'prior agent ID')
        battles,_=fetch(output,BATTLES_SERVICE,[prior['agentId'],None])
    return campaign.seal(output/'submit-claim.json',{'schema':campaign.ID+'.live-submit-claim.v2',
        'authorization':campaign.record(output/'authorization.json'),'copyback':campaign.record(output/'copyback.json'),
        'before_leaderboard':before,'before_battles':battles,'started_at_utc':now(),
        'ui_action_permitted_once':True,'retry_click_allowed':False,'ui_action_performed_by_this_command':False})


def submission_identity(auth,claim,before,after,battles,prior_battles):
    old=user_row(before,auth['user_id'],required=False);new=user_row(after,auth['user_id'])
    agent=positive(new.get('agentId'),'new agent ID');session=new.get('testSessionHandle')
    if not isinstance(session,str) or not session or old and session==old.get('testSessionHandle'):
        raise ValueError('a new server test-session identity has not appeared')
    if not isinstance(battles,list) or not isinstance(prior_battles,list):raise ValueError('battle metadata is malformed')
    def ids(rows):
        result=set()
        for game in rows:
            for player in game.get('players',[]):
                if player.get('playerAgentId')==agent:
                    result.add(positive(player.get('submissionId'),'submission ID'))
        return result
    candidates=ids(battles)-ids(prior_battles)
    if len(candidates)!=1:raise ValueError('new submission identity is ambiguous or not visible; do not resubmit')
    submission=candidates.pop()
    matching=[player for game in battles for player in game.get('players',[])
        if player.get('playerAgentId')==agent and player.get('submissionId')==submission]
    if not matching or any(player.get('testSessionHandle')!=session or player.get('userId')!=auth['user_id'] for player in matching):
        raise ValueError('new battle source session/owner differs from the leaderboard; retry observation only')
    collector.classify_matching_window(battles,agent_id=agent,submission_id=submission)
    return {'agent_id':agent,'submission_id':submission,'test_session_handle':session,'user_id':auth['user_id']}


def attest(root,source_sha):
    auth=load_authorization(root,source_sha);output=directory(root,source_sha)
    if (output/'submission.json').exists():
        _,recorded=validate_submission(root,source_sha)
        collector_submission(output,auth,recorded)
        return recorded
    claim=campaign.read(output/'submit-claim.json')
    if claim['authorization']!=campaign.record(output/'authorization.json') or claim.get('retry_click_allowed') is not False:
        raise ValueError('submission claim changed')
    before=response(claim['before_leaderboard'],LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    after_record,after=fetch(output,LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    row=user_row(after,auth['user_id']);agent=positive(row.get('agentId'),'observed agent')
    battles_record,battles=fetch(output,BATTLES_SERVICE,[agent,None])
    prior=user_row(before,auth['user_id'],required=False)
    prior_battles=response(claim['before_battles'],BATTLES_SERVICE,[prior['agentId'],None]) if prior is not None else []
    identity=submission_identity(auth,claim,before,after,battles,prior_battles)
    result=campaign.seal(output/'submission.json',{'schema':campaign.ID+'.live-submission.v2',
        'authorization':campaign.record(output/'authorization.json'),'claim':campaign.record(output/'submit-claim.json'),
        'after_leaderboard':after_record,'after_battles':battles_record,**identity,'source_sha256':source_sha,
        'source_bytes':auth['source']['bytes'],'candidate_commit':auth['candidate_commit'],
        'submitted_at_utc':claim['started_at_utc'],'submit_click_claims':1})
    collector_submission(output,auth,result)
    freeze_roster(output,result,battles_record,battles)
    return result


def validate_submission(root,source_sha):
    auth=load_authorization(root,source_sha);output=directory(root,source_sha)
    recorded=campaign.read(output/'submission.json');claim=campaign.read(campaign.verify(recorded['claim']))
    if (recorded['authorization']!=campaign.record(output/'authorization.json') or recorded['source_sha256']!=source_sha
            or recorded.get('schema')!=campaign.ID+'.live-submission.v2' or recorded['source_bytes']!=auth['source']['bytes']
            or recorded['candidate_commit']!=auth['candidate_commit'] or recorded.get('submit_click_claims')!=1
            or Path(recorded['claim']['path'])!=output/'submit-claim.json'
            or claim['authorization']!=recorded['authorization'] or claim.get('retry_click_allowed') is not False
            or recorded['submitted_at_utc']!=claim['started_at_utc']):
        raise ValueError('live submission source changed')
    copied=campaign.read(campaign.verify(claim['copyback']))
    if (Path(claim['copyback']['path'])!=output/'copyback.json' or copied['authorization']!=recorded['authorization']
            or campaign.verify(copied['captured_source']).read_bytes()!=campaign.verify(auth['source']).read_bytes()):
        raise ValueError('submitted editor copyback changed')
    started=collector.parse_utc(claim['started_at_utc'],'submit claim time')
    if collector.parse_utc(copied['observed_at_utc'],'copyback time')>started:
        raise ValueError('copyback observation postdates the submit claim')
    for key,comparison in (('before_leaderboard','before'),('after_leaderboard','after'),('after_battles','after')):
        record=claim[key] if comparison=='before' else recorded[key]
        fetched=collector.parse_utc(campaign.read(campaign.verify(record))['fetched_at_utc'],'snapshot time')
        if comparison=='before' and fetched>started or comparison=='after' and fetched<started:
            raise ValueError('live API snapshot crosses its submit-claim time boundary')
    if claim.get('before_battles'):
        fetched=collector.parse_utc(campaign.read(campaign.verify(claim['before_battles']))['fetched_at_utc'],'prior battle time')
        if fetched>started:raise ValueError('prior battle observation postdates the submit claim')
    before=response(claim['before_leaderboard'],LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    after=response(recorded['after_leaderboard'],LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    prior=user_row(before,auth['user_id'],required=False)
    prior_battles=response(claim['before_battles'],BATTLES_SERVICE,[prior['agentId'],None]) if prior is not None else []
    battles=response(recorded['after_battles'],BATTLES_SERVICE,[recorded['agent_id'],None])
    identity=submission_identity(auth,claim,before,after,battles,prior_battles)
    if any(recorded.get(key)!=value for key,value in identity.items()):raise ValueError('live submission does not reproduce observed server identity')
    return auth,recorded


def freeze_roster(output,submission,snapshot,payload):
    path=output/'window-roster.json'
    if path.exists():
        roster=campaign.read(path)
        original=response(roster['snapshot'],BATTLES_SERVICE,[submission['agent_id'],None])
        collector.classify_matching_window(original,agent_id=submission['agent_id'],submission_id=submission['submission_id'])
        ids=sorted(game['gameId'] for game in original if any(p.get('playerAgentId')==submission['agent_id'] and p.get('submissionId')==submission['submission_id'] for p in game.get('players',[])))
        if len(ids)!=90 or ids!=roster['game_ids'] or roster['submission']!=campaign.record(output/'submission.json'):
            raise ValueError('first observed exact90 roster changed')
        return roster
    collector.classify_matching_window(payload,agent_id=submission['agent_id'],submission_id=submission['submission_id'])
    ids=sorted(game['gameId'] for game in payload if any(p.get('playerAgentId')==submission['agent_id'] and p.get('submissionId')==submission['submission_id'] for p in game.get('players',[])))
    if len(ids)!=90:return None
    return campaign.seal(path,{'schema':campaign.ID+'.live-window-roster.v2','submission':campaign.record(output/'submission.json'),
        'snapshot':snapshot,'game_ids':ids,'selection':'first observed exact90 matching IDs; no result-based selection'})


def watch(root,source_sha,timeout_seconds=3600):
    auth,submission=validate_submission(root,source_sha);output=directory(root,source_sha)
    collector_submission(output,auth,submission)
    freeze_roster(output,submission,submission['after_battles'],response(submission['after_battles'],BATTLES_SERVICE,[submission['agent_id'],None]))
    def battles():
        snapshot,payload=fetch(output,BATTLES_SERVICE,[submission['agent_id'],None])
        roster=freeze_roster(output,submission,snapshot,payload)
        if roster is None:return payload
        allowed=set(roster['game_ids']);filtered=[row for row in payload if row.get('gameId') in allowed]
        if len(filtered)!=90 or {row['gameId'] for row in filtered}!=allowed:
            raise ValueError('frozen live IDs are no longer visible; preserve evidence without choosing replacements')
        return filtered
    return collector.watch_window(submission_attestation_path=output/'collector-submission.json',
        exclusion_binding_path=Path(auth['exclusion_binding']['path']),data_root=output/'window',
        poll_seconds=10,timeout_seconds=timeout_seconds,maximum_workers=2,fetch_battles=battles)


def live_window(root,source_sha):
    auth,submission=validate_submission(root,source_sha);output=directory(root,source_sha)
    collector_submission(output,auth,submission)
    path=output/'window/live-window.reference.json'
    reference=collector.verify_window_reference(path,data_root=output/'window')
    receipt=collector.load_sealed(collector.resolve_path(reference['receipt']['path']),collector.WINDOW_RECEIPT_SCHEMA,'v2 live window')
    roster=freeze_roster(output,submission,submission['after_battles'],response(submission['after_battles'],BATTLES_SERVICE,[submission['agent_id'],None]))
    if roster is None or sorted(receipt['game_ids'])!=roster['game_ids']:
        raise ValueError('live collector did not preserve the first observed exact90 roster')
    if receipt['submission_attestation']['sha256']!=campaign.sha(output/'collector-submission.json'):
        raise ValueError('live window belongs to another submitted source')
    manifest=collector.resolve_path(receipt['collector_manifest']['path'])
    identity,exclusions,_=collector.load_live_identity(output/'collector-submission.json',Path(auth['exclusion_binding']['path']))
    verified=collector.verify_generic_result({'manifest_path':str(manifest),'manifest_sha256':receipt['collector_manifest']['sha256']},
        identity=identity,registry_sha256=exclusions['registry']['sha256'],expected_game_ids=receipt['game_ids'])
    sessions={game['focus']['session_id'] for game in verified['records']}
    if sessions!={submission['test_session_handle']}:
        raise ValueError('90-game collector session differs from server submission session')
    return auth,submission,path,receipt


def score_row(payload,submission):
    row=user_row(payload,submission['user_id'])
    if row.get('agentId')!=submission['agent_id'] or row.get('testSessionHandle')!=submission['test_session_handle']:
        raise ValueError('leaderboard score belongs to another source session')
    score=row.get('score')
    if isinstance(score,bool) or not isinstance(score,(int,float,Decimal)):
        raise ValueError('leaderboard score is not numeric')
    numeric=Decimal(str(score))
    if not numeric.is_finite() or not 0<=numeric<=100:raise ValueError('leaderboard score is outside its numeric domain')
    percent=row.get('percentage')
    if isinstance(percent,bool) or not isinstance(percent,(int,float,Decimal)) or not Decimal(str(percent)).is_finite() or not 0<=Decimal(str(percent))<=100:
        raise ValueError('leaderboard calibration percentage is invalid')
    if type(row.get('inProgress')) is not bool:raise ValueError('leaderboard calibration status is absent')
    complete=Decimal(str(percent))==100 and row['inProgress'] is False
    rank=positive(row.get('rank'),'rank') if complete else row.get('rank')
    lower=max(Decimal(0),numeric-PUBLIC_SCORE_ERROR);upper=min(Decimal(100),numeric+PUBLIC_SCORE_ERROR)
    return {'score':str(numeric),'rank':rank,
        'percentage':str(percent),'in_progress':row['inProgress'],
        'calibration_complete':complete,'score_representation':'public-rounded-score',
        'conservative_score_interval':[str(lower),str(upper)],
        'strict_target_proven':lower>TARGET,'below_target_proven':upper<=TARGET,
        'precise_score_required':not(lower>TARGET or upper<=TARGET)}


def capture_score(root,source_sha):
    _,submission,window_path,_=live_window(root,source_sha);output=directory(root,source_sha)
    if (output/'calibrated-score.json').exists():return campaign.read(output/'calibrated-score.json')
    snapshot,payload=fetch(output,LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    scored=score_row(payload,submission)
    body={'schema':campaign.ID+'.live-score-observation.v2','snapshot':snapshot,
        'submission':campaign.record(output/'submission.json'),'window':campaign.record(window_path),**scored}
    if scored['calibration_complete']:
        return campaign.seal(output/'calibrated-score.json',body)
    path=output/'calibration-wait'/(hashlib.sha256(campaign.raw(body)).hexdigest()+'.json')
    return campaign.seal(path,body)


def clean_summary(summary):
    # All90 must be operationally clean; opponent forfeits never become strength evidence.
    keys=('focus_operational_failure_games','opponent_operational_failure_games','clean_strength_games','opponent_failure_games_counted_as_strength_wins')
    return (all(type(summary.get(key)) is int for key in keys)
        and summary.get('focus_operational_failure_games')==0 and summary.get('opponent_operational_failure_games')==0
        and summary.get('clean_strength_games')==90 and summary.get('opponent_failure_games_counted_as_strength_wins')==0)


def assess(root,source_sha):
    from tools import compact_value_bfm_protected_v2 as protected
    auth,submission,window_path,receipt=live_window(root,source_sha);output=directory(root,source_sha)
    score_path=output/'calibrated-score.json';observation=campaign.read(score_path)
    if observation['submission']!=campaign.record(output/'submission.json') or observation['window']!=campaign.record(window_path):
        raise ValueError('calibrated score lost its exact90 source window')
    payload=response(observation['snapshot'],LEADERBOARD_SERVICE,LEADERBOARD_REQUEST)
    derived=score_row(payload,submission)
    if not derived['calibration_complete'] or any(observation.get(key)!=value for key,value in derived.items()):
        raise ValueError('source calibration is incomplete or score summary changed')
    qualified=protected.validate(Path(root).resolve(),Path(auth['context']),auth['phase'])
    freeze=campaign.read(campaign.verify(qualified['freeze']))
    if (qualified.get('passed') is not True or campaign.record(Path(auth['qualification']['path']))!=auth['qualification']
            or qualified['selected']['source']['sha256']!=source_sha or qualified['selected']['runtime']!=auth['runtime']
            or qualified['selected']['payload_sha256']!=auth['payload_sha256'] or freeze['candidate_commit']!=auth['candidate_commit']):
        raise ValueError('live completion lost its source-specific protected qualification')
    clean=clean_summary(receipt['summary'])
    passed=clean and derived['strict_target_proven']
    result_path=output/('assessment-precision-inconclusive.json' if clean and derived['precise_score_required'] else 'assessment.json')
    result=campaign.seal(result_path,{'schema':campaign.ID+'.live-assessment.v2',
        'authorization':campaign.record(output/'authorization.json'),'submission':campaign.record(output/'submission.json'),
        'window':campaign.record(window_path),'score_observation':campaign.record(score_path),
        'exact_games':90,'clean_window':clean,'calibration_complete':True,'score':derived['score'],'rank':derived['rank'],
        'score_representation':derived['score_representation'],'score_interval':derived['conservative_score_interval'],
        'precise_score_required':derived['precise_score_required'],
        'strict_score_target':str(TARGET),'rank_is_separate_from_success':True,'campaign_success':passed,
        'status':'campaign-success' if passed else 'score-precision-inconclusive' if clean and derived['precise_score_required'] else 'completed-live-attempt-below-objective',
        'training_eligible':False,'identical_source_reupload_allowed':False})
    if passed:campaign.seal(Path(root)/'success.json',{'schema':campaign.ID+'.campaign-success.v2','live_assessment':campaign.record(result_path),'score':derived['score'],'rank':derived['rank']})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-sha256');parser.add_argument('--context',type=Path);parser.add_argument('--phase')
    parser.add_argument('--registry',type=Path);parser.add_argument('--user-id',type=int)
    parser.add_argument('--captured-source',type=Path);parser.add_argument('--editor-url');parser.add_argument('--editor-session')
    parser.add_argument('command',choices=('authorize','copyback','start-submit','attest','watch','score','assess'))
    args=parser.parse_args()
    if args.command=='authorize':
        with campaign.lease(args.root.resolve()):result=authorize(args.root,args.context,args.phase,args.registry,args.user_id)
    elif args.command=='copyback':result=copyback(args.root,args.source_sha256,args.captured_source,args.editor_url,args.editor_session)
    elif args.command in ('start-submit','assess'):
        with campaign.lease(args.root.resolve()):
            result={'start-submit':start_submit,'assess':assess}[args.command](args.root,args.source_sha256)
    else:result={'attest':attest,'watch':watch,'score':capture_score}[args.command](args.root,args.source_sha256)
    print(json.dumps({'status':result.get('status'), 'campaign_success':result.get('campaign_success',False),
        'calibration_complete':result.get('calibration_complete',False)},sort_keys=True))


if __name__=='__main__':main()
