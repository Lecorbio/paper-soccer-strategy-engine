"""V2 root policy, leakage closure, and immutable resume contracts."""
import copy
import json
import pathlib
import random
import tempfile
import unittest

from tools import compact_value_bfm_campaign_v2 as campaign


class CampaignV2Tests(unittest.TestCase):
    def test_exact_root_progress_and_complete_turns(self):
        rng=random.Random(19)
        for edges in (8,12,20):
            for _ in range(12):
                state,transcript=campaign.fresh_root(edges,rng)
                replay=campaign.features.ReplayState()
                for action in transcript.split('/'):
                    campaign.features.apply_complete_turn(replay,replay.to_move,action)
                self.assertEqual(len(replay.used_segments),edges)
                self.assertEqual(campaign.fingerprints(replay),campaign.fingerprints(state))
                self.assertIsNone(replay.winner)

    def test_exception_never_exempts_validation_or_protected(self):
        state=campaign.features.ReplayState()
        for domain,fp in campaign.fingerprints(state).items():
            for role in ('live','prior-train'):
                excluded={(role,domain):{fp}}
                self.assertIsNone(campaign.rejection(state,'train',excluded))
                self.assertEqual(campaign.rejection(state,'validation',excluded),role)
            for role in ('protected','prior-validation','mixed-development'):
                self.assertEqual(campaign.rejection(state,'train',{(role,domain):{fp}}),role)

    def test_six_edge_parent_does_not_exempt_seven_edge_successor(self):
        state,_=campaign.fresh_root(6,random.Random(51))
        action,child=campaign.successors(state)[0]
        self.assertGreater(len(child.used_segments),6)
        domain=campaign.legacy.STATE_FINGERPRINT_DOMAIN
        excluded={('live',domain):{campaign.fingerprints(state)[domain],campaign.fingerprints(child)[domain]}}
        self.assertIsNone(campaign.rejection(state,'train',excluded))
        self.assertEqual(campaign.rejection(child,'train',excluded),'live')

    def test_forbidden_successor_rejects_entire_group(self):
        state,_=campaign.fresh_root(6,random.Random(51))
        _,child=campaign.successors(state)[-1]
        domain=campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
        excluded={('live',domain):{campaign.fingerprints(child)[domain]}}
        report=campaign.preflight_group(state,'train',excluded)
        self.assertEqual(report,{'eligible':False,'reason':'successor:live'})
        clean=campaign.preflight_group(state,'train',{})
        self.assertTrue(clean['eligible'])
        self.assertEqual(set(clean['legal_actions']),{a for a,_ in campaign.successors(state)})

    def test_empty_root_has_eight_legal_complete_successors(self):
        state=campaign.features.ReplayState(); successors=campaign.successors(state)
        self.assertEqual({a for a,s in successors},set('01234567'))
        for action,child in successors:
            replay=copy.deepcopy(state)
            campaign.features.apply_complete_turn(replay,replay.to_move,action)
            self.assertEqual(campaign.fingerprints(replay),campaign.fingerprints(child))
        with self.assertRaisesRegex(ValueError,'closure-resource-limit'):
            campaign.successors(state,limit=1)

    def test_schedule_marginals_and_validation_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            campaign.seal(root/'campaign.json',{'exclusions':[]})
            plan=campaign.schedule(root,'smoke',64)
            self.assertEqual(plan['root_depth_counts'],{'0':16,'8':16,'12':16,'20':16})
            self.assertEqual(plan['actor_counts'],dict(zip(campaign.MODES,(16,16,16,4,4,4,4))))
            self.assertEqual(plan['validation_roots'],10)
            self.assertTrue(all(r['split']=='train' for r in plan['rows'] if r['drawn_edges']==0))
            self.assertEqual(len({r['canonical'] for r in plan['rows'] if r['drawn_edges']}),48)
            self.assertEqual(campaign.schedule(root,'smoke',64)['body_sha256'],plan['body_sha256'])

    def test_vectorized_fingerprint_matches_reference(self):
        rng=random.Random(981)
        for edges in (0,6,8,12,20,40,80):
            state=campaign.features.ReplayState() if edges==0 else campaign.fresh_root(edges,rng)[0]
            active=campaign.features.encode_active(state)
            self.assertEqual(campaign.fast_feature_fingerprint(active),campaign.corpus.canonical_feature_fingerprint(active))
            for child in (campaign.openings.transform_state(state,rotate=True,reflect=False),
                          campaign.openings.transform_state(state,rotate=False,reflect=True)):
                active=campaign.features.encode_active(child)
                self.assertEqual(campaign.fast_feature_fingerprint(active),campaign.corpus.canonical_feature_fingerprint(active))

    def test_append_only_ledger_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            first=campaign.event(root,'one',{})
            second=campaign.event(root,'two',{})
            self.assertEqual(second['previous'],first['body_sha256'])
            path=root/'ledger/000000.json'
            doc=json.loads(path.read_text());doc['kind']='tampered';path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'changed receipt'):
                campaign.event(root,'three',{})


if __name__=='__main__': unittest.main()
