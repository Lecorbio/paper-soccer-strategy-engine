import copy
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_live_v2 as live
from tools import compact_value_bfm_protected_v2 as protected

campaign=live.campaign


def leader(agent=2,session='new-session',score=45.0,percentage=100,progress=False):
    return {'users':[{'agentId':agent,'testSessionHandle':session,'codingamer':{'userId':42},
        'score':score,'rank':7,'percentage':percentage,'inProgress':progress}]}


def battles(agent=2,submission=200,session='new-session',count=1):
    return [{'gameId':1000+i,'done':False,'players':[
        {'playerAgentId':agent,'submissionId':submission,'testSessionHandle':session,'userId':42},
        {'playerAgentId':99,'submissionId':999,'testSessionHandle':'other','userId':99}]} for i in range(count)]


class LiveTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.context=self.root/'phases/attempt-001-full';self.phase='attempt-001-full'
        source=self.root/'candidate.cpp';source.write_text('int main(){return 0;}\n')
        runtime=self.root/'runtime.json';campaign.seal(runtime,{'quantization':{'payload_sha256':'a'*64}})
        selected={'source':campaign.record(source),'runtime':campaign.record(runtime),'payload_sha256':'a'*64}
        freeze=self.context/self.phase/'release/freeze.json';campaign.seal(freeze,{'candidate_commit':'b'*40})
        self.qualified=campaign.seal(self.context/self.phase/'protected/assessment.json',{'passed':True,'selected':selected,'freeze':campaign.record(freeze)})
        self.registry=self.root/'registry.json'
        self.registry.write_bytes(live.collector.canonical_json_bytes({'schema':live.collector.EXCLUSION_SCHEMA,'records':[]}))
        patch=mock.patch.object(protected,'validate',return_value=self.qualified);patch.start();self.addCleanup(patch.stop)
        self.source_sha=selected['source']['sha256'];self.output=live.directory(self.root,self.source_sha)

    def authorize(self):
        with mock.patch.object(live,'now',return_value='2026-09-05T00:00:00Z'):
            return live.authorize(self.root,self.context,self.phase,self.registry,42)

    def snapshot(self,name,payload,service,request,when):
        path=self.root/(name+'.raw.json');path.write_text(json.dumps(payload))
        receipt=self.root/(name+'.json')
        campaign.seal(receipt,{'schema':campaign.ID+'.live-http-snapshot.v2','service':service,'request':request,
            'status':200,'raw':campaign.record(path),'fetched_at_utc':when})
        return campaign.record(receipt),payload

    def submit_fixture(self,count=1):
        auth=self.authorize()
        captured=self.root/'capture.cpp';captured.write_bytes(Path(auth['source']['path']).read_bytes())
        with mock.patch.object(live,'now',return_value='2026-09-05T00:00:20Z'):
            live.copyback(self.root,self.source_sha,captured,'https://www.codingame.com/ide/puzzle/paper-soccer','fresh-editor')
        snapshots=[
            self.snapshot('before',leader(1,'old-session'),live.LEADERBOARD_SERVICE,live.LEADERBOARD_REQUEST,'2026-09-05T00:00:30Z'),
            self.snapshot('before-battles',battles(1,100,'old-session'),live.BATTLES_SERVICE,[1,None],'2026-09-05T00:00:31Z'),
            self.snapshot('after',leader(),live.LEADERBOARD_SERVICE,live.LEADERBOARD_REQUEST,'2026-09-05T00:02:00Z'),
            self.snapshot('after-battles',battles(count=count),live.BATTLES_SERVICE,[2,None],'2026-09-05T00:02:01Z')]
        with mock.patch.object(live,'fetch',side_effect=snapshots),mock.patch.object(live,'now',return_value='2026-09-05T00:01:00Z'):
            live.start_submit(self.root,self.source_sha)
            submitted=live.attest(self.root,self.source_sha)
        return auth,submitted

    def test_truthful_new_compatibility_records_and_submit_once(self):
        historical=self.root/'old-authorization.json';historical.write_bytes(b'original historical receipt')
        auth,submitted=self.submit_fixture()
        identity,_,_=live.collector.load_live_identity(self.output/'collector-submission.json',self.output/'exclusion-binding.json')
        self.assertEqual(identity.source_sha256,self.source_sha)
        self.assertEqual((identity.agent_id,identity.submission_id),(2,200))
        self.assertEqual(live.validate_submission(self.root,self.source_sha)[1],submitted)
        self.assertEqual(historical.read_bytes(),b'original historical receipt')
        with self.assertRaisesRegex(ValueError,'already claimed'):
            live.start_submit(self.root,self.source_sha)
        (self.output/'collector-submission.json').unlink()
        with mock.patch.object(live,'fetch',side_effect=AssertionError('must not re-fetch submitted identity')):
            self.assertEqual(live.attest(self.root,self.source_sha),submitted)
        self.assertTrue((self.output/'collector-submission.json').exists())

    def test_authorization_recovery_preserves_exclusion_freeze(self):
        original=campaign.seal
        def fail(path,body):
            if Path(path).name=='authorization.json':raise RuntimeError('interrupted')
            return original(path,body)
        with mock.patch.object(campaign,'seal',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.authorize()
        binding=(self.output/'exclusion-binding.json').read_bytes()
        with mock.patch.object(live,'now',return_value='2026-09-05T00:00:20Z'):
            auth=live.authorize(self.root,self.context,self.phase,self.registry,42)
        self.assertEqual((self.output/'exclusion-binding.json').read_bytes(),binding)
        self.assertEqual(auth['created_at_utc'],'2026-09-05T00:00:00Z')
        (self.output/'collector-authorization.json').unlink()
        self.authorize()
        self.assertTrue((self.output/'collector-authorization.json').exists())

    def test_wrong_copyback_does_not_consume_source_capture(self):
        auth=self.authorize();captured=self.root/'capture.cpp';captured.write_text('wrong')
        with self.assertRaisesRegex(ValueError,'editor bytes differ'):
            live.copyback(self.root,self.source_sha,captured,'https://www.codingame.com/ide/puzzle/paper-soccer','editor')
        self.assertFalse((self.output/'editor-copyback.cpp').exists())
        captured.write_bytes(Path(auth['source']['path']).read_bytes())
        live.copyback(self.root,self.source_sha,captured,'https://www.codingame.com/ide/puzzle/paper-soccer','editor')

    def test_submission_requires_new_matching_session_and_owner(self):
        args=({'user_id':42},{},leader(1,'old-session'),leader(),battles(),battles(1,100,'old-session'))
        self.assertEqual(live.submission_identity(*args)['submission_id'],200)
        for key,value in (('testSessionHandle','stale'),('userId',99)):
            changed=copy.deepcopy(args);changed[4][0]['players'][0][key]=value
            with self.assertRaisesRegex(ValueError,'session/owner'):
                live.submission_identity(*changed)
        changed=list(args);changed[3]=leader(1,'old-session')
        with self.assertRaisesRegex(ValueError,'new server test-session'):
            live.submission_identity(*changed)

    def test_first_exact90_roster_is_immutable(self):
        _,submitted=self.submit_fixture(count=90)
        roster=campaign.read(self.output/'window-roster.json')
        self.assertEqual(len(roster['game_ids']),90)
        repeated=live.freeze_roster(self.output,submitted,submitted['after_battles'],battles(count=91))
        self.assertEqual(repeated,roster)
        self.assertEqual(repeated['game_ids'],list(range(1000,1090)))

    def test_rounded_score_cannot_falsely_clear_strict_target(self):
        submitted={'user_id':42,'agent_id':2,'test_session_handle':'new-session'}
        near=live.score_row(leader(score=44.30),submitted)
        self.assertFalse(near['strict_target_proven']);self.assertTrue(near['precise_score_required'])
        self.assertTrue(live.score_row(leader(score=44.31),submitted)['strict_target_proven'])
        self.assertTrue(live.score_row(leader(score=44.28),submitted)['below_target_proven'])
        self.assertFalse(live.score_row(leader(score=90,percentage=99),submitted)['calibration_complete'])
        self.assertFalse(live.score_row(leader(score=90,progress=True),submitted)['calibration_complete'])
        with self.assertRaisesRegex(ValueError,'another source session'):
            live.score_row(leader(session='another'),submitted)

    def test_clean_means_all90_without_forfeit_strength_credit(self):
        summary={'focus_operational_failure_games':0,'opponent_operational_failure_games':0,
            'clean_strength_games':90,'opponent_failure_games_counted_as_strength_wins':0}
        self.assertTrue(live.clean_summary(summary))
        for key in summary:
            changed=dict(summary);changed[key]=False
            self.assertFalse(live.clean_summary(changed))
        changed=dict(summary);changed['opponent_operational_failure_games']=1
        self.assertFalse(live.clean_summary(changed))

    def test_first_completed_calibration_snapshot_is_not_refetched(self):
        _,submitted=self.submit_fixture()
        window=self.root/'window.json';campaign.seal(window,{'exact_games':90})
        snapshot=self.snapshot('score',leader(),live.LEADERBOARD_SERVICE,live.LEADERBOARD_REQUEST,'2026-09-05T00:10:00Z')
        with mock.patch.object(live,'live_window',return_value=({},submitted,window,{})),mock.patch.object(live,'fetch',return_value=snapshot) as fetch:
            first=live.capture_score(self.root,self.source_sha)
            second=live.capture_score(self.root,self.source_sha)
        self.assertEqual(first,second);self.assertEqual(fetch.call_count,1)

    def test_watch_captures_calibration_immediately_and_assesses_under_the_lease(self):
        events=[]
        window={'schema':live.collector.WINDOW_REFERENCE_SCHEMA,'exact_games':90}
        @contextlib.contextmanager
        def lease(_root):
            events.append('lease-enter')
            yield
            events.append('lease-exit')
        def score(*_args):
            events.append('capture-score')
            return {'calibration_complete':True}
        def assess(*_args):
            events.append('assess')
            return {'status':'campaign-success','campaign_success':True}
        with mock.patch.object(live,'_collect_window',return_value=window), \
                mock.patch.object(live,'capture_score',side_effect=score), \
                mock.patch.object(live,'assess',side_effect=assess), \
                mock.patch.object(campaign,'lease',side_effect=lease), \
                mock.patch.object(live.time,'sleep',side_effect=AssertionError('complete calibration must not wait')):
            result=live.watch(self.root,self.source_sha)
        self.assertTrue(result['campaign_success'])
        self.assertEqual(events,['capture-score','lease-enter','assess','lease-exit'])

    def test_watch_waits_for_calibration_without_recollecting_or_selecting_a_new_score(self):
        window={'schema':live.collector.WINDOW_REFERENCE_SCHEMA,'exact_games':90}
        with mock.patch.object(live,'_collect_window',return_value=window) as collect, \
                mock.patch.object(live,'capture_score',side_effect=[{'calibration_complete':False},{'calibration_complete':True}]) as score, \
                mock.patch.object(live,'assess',return_value={'status':'score-precision-inconclusive','campaign_success':False}) as assess, \
                mock.patch.object(campaign,'lease',return_value=contextlib.nullcontext()), \
                mock.patch.object(live.time,'monotonic',side_effect=[0,1,11]), \
                mock.patch.object(live.time,'sleep') as sleep:
            result=live.watch(self.root,self.source_sha,timeout_seconds=20)
        self.assertEqual(result['status'],'score-precision-inconclusive')
        self.assertEqual(collect.call_count,1);self.assertEqual(score.call_count,2);self.assertEqual(assess.call_count,1)
        sleep.assert_called_once_with(10)

    def test_watch_timeout_preserves_incomplete_calibration_without_assessment(self):
        window={'schema':live.collector.WINDOW_REFERENCE_SCHEMA,'exact_games':90}
        with mock.patch.object(live,'_collect_window',return_value=window), \
                mock.patch.object(live,'capture_score',return_value={'calibration_complete':False}), \
                mock.patch.object(live,'assess',side_effect=AssertionError('incomplete calibration')), \
                mock.patch.object(live.time,'monotonic',side_effect=[0,20]), \
                mock.patch.object(live.time,'sleep',side_effect=AssertionError('deadline already elapsed')):
            result=live.watch(self.root,self.source_sha,timeout_seconds=20)
        self.assertEqual(result['status'],'awaiting-source-calibration')
        self.assertFalse(result['campaign_success']);self.assertTrue(result['timed_out'])

    def test_watch_does_not_start_another_score_request_at_the_deadline(self):
        window={'schema':live.collector.WINDOW_REFERENCE_SCHEMA,'exact_games':90}
        with mock.patch.object(live,'_collect_window',return_value=window), \
                mock.patch.object(live,'capture_score',return_value={'calibration_complete':False}) as score, \
                mock.patch.object(live,'assess',side_effect=AssertionError('incomplete calibration')), \
                mock.patch.object(live.time,'monotonic',side_effect=[0,19,20]), \
                mock.patch.object(live.time,'sleep') as sleep:
            result=live.watch(self.root,self.source_sha,timeout_seconds=20)
        self.assertEqual(result['status'],'awaiting-source-calibration')
        self.assertEqual(score.call_count,1);sleep.assert_called_once_with(1)

    def test_watch_incomplete_collection_never_fetches_score(self):
        waiting={'schema':live.collector.WAIT_SNAPSHOT_SCHEMA,'timed_out':True}
        with mock.patch.object(live,'_collect_window',return_value=waiting), \
                mock.patch.object(live,'capture_score',side_effect=AssertionError('collection incomplete')):
            self.assertEqual(live.watch(self.root,self.source_sha),waiting)
        for value in (True,-1,float('nan'),float('inf')):
            with self.subTest(value=value),mock.patch.object(live,'_collect_window') as collect:
                with self.assertRaises(ValueError):live.watch(self.root,self.source_sha,value)
                collect.assert_not_called()

    def test_submit_rechecks_actual_protected_source_before_network(self):
        auth=self.authorize()
        captured=self.root/'capture.cpp';captured.write_bytes(Path(auth['source']['path']).read_bytes())
        live.copyback(self.root,self.source_sha,captured,'https://www.codingame.com/ide/puzzle/paper-soccer','editor')
        changed=copy.deepcopy(self.qualified);changed['selected']['payload_sha256']='c'*64
        with mock.patch.object(protected,'validate',return_value=changed),mock.patch.object(live,'fetch',side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(ValueError,'actual protected qualification'):
                live.start_submit(self.root,self.source_sha)
        self.assertFalse((self.output/'submit-claim.json').exists())

    def completed_window(self,score):
        auth,submitted=self.submit_fixture(count=90)
        window=self.output/'window-reference.json';campaign.seal(window,{'exact_games':90})
        receipt={'summary':{'focus_operational_failure_games':0,'opponent_operational_failure_games':0,
            'clean_strength_games':90,'opponent_failure_games_counted_as_strength_wins':0}}
        value=(auth,submitted,window,receipt)
        observed=self.snapshot('final-score',leader(score=score),live.LEADERBOARD_SERVICE,live.LEADERBOARD_REQUEST,'2026-09-05T00:10:00Z')
        with mock.patch.object(live,'live_window',return_value=value),mock.patch.object(live,'fetch',return_value=observed):
            live.capture_score(self.root,self.source_sha)
        return value

    def test_goal_success_requires_clean_window_and_proven_score(self):
        value=self.completed_window(45)
        with mock.patch.object(live,'live_window',return_value=value):
            result=live.assess(self.root,self.source_sha)
        self.assertTrue(result['campaign_success'])
        self.assertTrue((self.root/'success.json').exists())
        self.assertEqual(result['rank'],7)
        self.assertTrue(result['rank_is_separate_from_success'])

    def test_watch_resume_reuses_frozen_calibration_without_network(self):
        value=self.completed_window(45)
        window={'schema':live.collector.WINDOW_REFERENCE_SCHEMA,'exact_games':90}
        with mock.patch.object(live,'_collect_window',return_value=window), \
                mock.patch.object(live,'live_window',return_value=value), \
                mock.patch.object(live,'fetch',side_effect=AssertionError('frozen score must not refresh')), \
                mock.patch.object(campaign,'lease',return_value=contextlib.nullcontext()):
            result=live.watch(self.root,self.source_sha)
        self.assertTrue(result['campaign_success'])
        self.assertTrue((self.root/'success.json').exists())

    def test_near_threshold_does_not_finalize_failure_or_success(self):
        value=self.completed_window(44.30)
        with mock.patch.object(live,'live_window',return_value=value):
            result=live.assess(self.root,self.source_sha)
        self.assertEqual(result['status'],'score-precision-inconclusive')
        self.assertFalse(result['campaign_success'])
        self.assertFalse((self.root/'success.json').exists())
        self.assertFalse((self.output/'assessment.json').exists())

    def test_high_score_cannot_hide_operational_failure(self):
        value=self.completed_window(70)
        value[3]['summary']['opponent_operational_failure_games']=1
        value[3]['summary']['clean_strength_games']=89
        with mock.patch.object(live,'live_window',return_value=value):
            result=live.assess(self.root,self.source_sha)
        self.assertFalse(result['campaign_success'])
        self.assertFalse((self.root/'success.json').exists())


if __name__=='__main__':unittest.main()
