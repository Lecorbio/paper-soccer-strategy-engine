"""Synthetic checks for the live extension's boundaries and statistical units."""
import copy
import datetime as dt
from pathlib import Path
import unittest
import tempfile
from unittest import mock
from tools import rank_two_live_eight_v1 as live
from tests.codingame.test_rank_two_focused_campaign_v2 import usage

BASE = live.BASELINE
AT = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
DOC = dict(deadline_utc="2026-10-21T21:28:13+00:00", research_cutoff_utc="2026-10-19T21:28:13+00:00")


def state():
    return dict(owner_thread_id="owner", ownership_status="active", worktree=str(live.ROOT),
                research_closed=False, frozen_sources={}, live_eight_versions={},
                active_claims={}, focused_banned_semantic_identities=[],
                live_eight_declared_windows={"slot":dict(source_sha256=BASE,
                    purpose="exploration",plan=dict(path="fixture",sha256="fixture"))})


class Boundaries(unittest.TestCase):
    def call(self, body, action="observe", **kwargs):
        with mock.patch.object(live,"activation",return_value=DOC), \
             mock.patch.object(live,"source_admission"), mock.patch.object(live,"verify",side_effect=lambda r:Path(r['path']) if Path(r['path']).is_file() else Path(__file__)):
            return live.validate(body,{},"owner",action,19,at=kwargs.pop("at",AT),**kwargs)

    def test_research_cutoff_preserves_qualification_and_terminal_work(self):
        body=state();at=dt.datetime(2026,10,20,tzinfo=dt.timezone.utc)
        with self.assertRaises(PermissionError):self.call(body,"train",at=at)
        self.assertTrue(self.call(body,"archive",at=at))
        body['live_eight_declared_windows']['slot']['purpose']='qualification'
        with self.assertRaises(KeyError):self.call(body,"claim",source=BASE,attempt='slot',at=at)
        with tempfile.TemporaryDirectory() as directory:
            body['live_eight_finalist']=live.immutable(Path(directory)/'finalist.json',dict(source=dict(sha256=BASE)))
            self.assertTrue(self.call(body,"claim",source=BASE,attempt='slot',at=at))

    def test_original_exact_and_semantic_bans_cannot_be_lifted_by_strength_override(self):
        body=state()
        with self.assertRaises(PermissionError):live.banned(body,dict(operationally_failed=[dict(source_sha256='bad')]),'bad')
        body.update(frozen_sources={'new-format':dict(semantic_identity='bad-body')},focused_banned_semantic_identities=['bad-body'])
        with self.assertRaises(PermissionError):live.banned(body,{},'new-format')

    def test_late_slots_duplicate_windows_and_unarchived_games(self):
        body=state();body['live_eight_versions']={str(i):{} for i in range(10)}
        with self.assertRaises(PermissionError):self.call(body,'admit',source='new')
        self.assertTrue(self.call(body,'admit',source='new',at=dt.datetime(2026,10,18,tzinfo=dt.timezone.utc)))
        body['live_eight_declared_windows']['slot']['attested']=True
        with self.assertRaises(PermissionError):self.call(body,'claim',source=BASE,attempt='slot')
        body['live_eight_declared_windows']['slot']['attested']=False
        body['live_eight_known_windows']={'old':dict(all90_archived=False)}
        with self.assertRaises(PermissionError):self.call(body,'claim',source=BASE,attempt='slot')

    def test_real_usage_and_exposure_barriers(self):
        with self.assertRaises(PermissionError):live.actual_usage(usage(70))
        with self.assertRaises(PermissionError):live.actual_usage(usage(19,AT))
        body=state();body['pending_exposures']=['trace'];body['live_eight_ancestry_validated']=True
        with self.assertRaises(PermissionError):self.call(body,'prepare-bank')

    def test_only_designated_new_clean_exploration_is_training(self):
        window=dict(extension=live.SCHEMA,purpose='exploration',training_designated=True,complete=True)
        game=dict(clean=True)
        self.assertTrue(live.training_allowed(window,game))
        for purpose in ('private','formal','qualification','confirmation'):
            self.assertFalse(live.training_allowed({**window,'purpose':purpose},game))
        self.assertFalse(live.training_allowed({**window,'extension':'old'},game))
        self.assertFalse(live.training_allowed(window,dict(clean=True,opponent_failure=True)))


PRIMARY=[dict(user_id=i,name=str(i),agent_id=i) for i in range(1,7)]


def windows():
    result=[]
    for number,arm in enumerate('BCCB'):
        games=[]
        for opponent in range(1,7):
            for color in (0,1):
                for repeat in range(3):
                    games.append(dict(game_id=number*1000+len(games),opponent_user_id=opponent,
                        opponent_submission_id=100+opponent,color=color,won=arm=='C',clean=True))
        while len(games)<90:
            games.append(dict(game_id=number*1000+len(games),opponent_user_id=999,
                opponent_submission_id=999,color=0,won=False,clean=True))
        result.append(dict(arm=arm,complete=True,games=games))
    return result


class Assessment(unittest.TestCase):
    def test_whole_window_units_and_live_gate_do_not_claim_deployment(self):
        r=live.score_windows(windows(),PRIMARY,repetitions=100)
        self.assertTrue(r['live_gate_passed']);self.assertFalse(r['qualified_for_deployment'])
        self.assertEqual(r['point']['mean'],1.)
        self.assertEqual(r['interval95'],[1.,1.])
        self.assertEqual(r['bootstrap_unit'],'paired complete calibration windows')

    def test_version_drift_missing_color_and_own_failure_are_not_successes(self):
        w=windows()
        for g in w[1]['games']:g['opponent_submission_id']+=1000
        for g in w[2]['games']:g['opponent_submission_id']+=1000
        self.assertFalse(live.score_windows(w,PRIMARY,repetitions=20)['sufficient'])
        w=windows();w[1]['games'][0]['own_failure']=True
        self.assertFalse(live.score_windows(w,PRIMARY,repetitions=20)['live_gate_passed'])
        w=windows();w[3]['games'][0]['game_id']=w[0]['games'][0]['game_id']
        with self.assertRaises(ValueError):live.score_windows(w,PRIMARY,repetitions=20)

    def test_primary_roster_uses_actual_ladder_identities(self):
        rows=[dict(rank=i,agentId=i,codingamer=dict(userId=i,pseudo=name))
              for i,name in enumerate(['jacek','owner','Marchete','a','b','c','d'],1)]
        chosen=live.select_primary(rows,2)
        self.assertEqual([r['name'] for r in chosen],['jacek','Marchete','a','b','c','d'])


if __name__=='__main__':unittest.main()
