"""V2 root policy, leakage closure, and immutable resume contracts."""
import copy
import contextlib
import io
import json
import pathlib
import random
import tempfile
import types
import weakref
import unittest
from unittest import mock

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


class TrainingInputPreparationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from tools import compact_value_bfm_train as trainer
        from tools import compact_value_bfm_teacher_training as adapter
        from tools import compact_value_bfm_seed_process_v2 as process
        from tools import compact_value_bfm_ranking_store as storage
        from tools import jacek_replay_train as packer
        from tests.codingame.test_compact_value_bfm_training import dataset, active_row
        self.trainer,self.adapter,self.process,self.storage,self.packer=trainer,adapter,process,storage,packer
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=pathlib.Path(self.temporary.name).resolve(); self.phase='attempt-002-pilot'
        directory=self.root/self.phase; directory.mkdir()
        bundle_path=self.root/'bundle.json'; bundle_path.write_bytes(b'synthetic bundle')
        self.plan=campaign.seal(self.root/'campaign.json',{'bundle':campaign.record(bundle_path),
            'inputs':{'attempt_one_initial_checkpoint':{'must_not_be_opened':True}}})
        campaign.seal(self.root/'exclusions/anchor-derived.json',{'fingerprints':{}})
        self.positions=[{'split':'train','drawn_edges':0,'prefix':''}]
        campaign.seal(directory/'positions.json',{'rows':self.positions})
        merged=directory/'labels.jsonl'; merged.write_bytes(b'synthetic retained labels')
        self.labels=campaign.seal(directory/'labels.json',{'merged':campaign.record(merged),
            'positions':campaign.record(directory/'positions.json')})
        self.index=directory/'ranking-store/index.json'
        campaign.seal(self.index,{'sources':[self.labels['merged']]})
        self.bundle=object()
        self.rankings=types.SimpleNamespace(train=('t1','t2'),validation=('v1','v2'))
        a,b,c=tuple(active_row(0,1)),tuple(active_row(1,2)),tuple(active_row(2,3))
        sample=lambda active,value:types.SimpleNamespace(active=active,value=value)
        self.samples={'v1':[sample(a,.9),sample(c,.8)],'v2':[sample(a,-.9)],
                      't1':[sample(b,.1),sample(a,.2)],'t2':[sample(b,-.1),sample(c,.3)]}
        empty=np.asarray(campaign.features.encode_active(campaign.features.ReplayState()),dtype='<u2')
        self.anchor=dataset([active_row(2,4),empty,active_row(3,5)],[.2,.3,-.4])
        self.common=dataset([active_row(1,9)],[.6],split='validation')
        self.canonical=dataset([active_row(4,3)],[-.7],split='validation')
        self.routes={'anchor':('canonical/train',),'common_adjudicator':('common',),
                     'canonical_validation':('canonical/validation',)}
        self.datasets={}; self.packing=[]; self.group_visits=[]
        def write(directory,split,samples,*,provenance):
            self.packing.append((split,list(samples),provenance))
            value=dataset([list(sample.active) for sample in samples],[sample.value for sample in samples],split=split)
            self.datasets[split]=value
            manifest=directory/(split+'.json'); npz=directory/(split+'.npz')
            campaign.once(manifest,campaign.raw({'split':split,'provenance':provenance}))
            campaign.once(npz,trainer.deterministic_npz({name:getattr(value,name)
                for name in ('indptr','indices','targets','weights','group_ids')}))
            return npz,manifest,{}
        def scalar(group):
            self.group_visits.append(group)
            return self.samples[group]
        self.stack=self.enterContext(contextlib.ExitStack())
        for patch in (
            mock.patch.object(trainer.FrozenBundle,'load',return_value=self.bundle),
            mock.patch.object(storage,'build_store',return_value=self.index),
            mock.patch.object(storage,'RankingStore',return_value=types.SimpleNamespace(labels=lambda:self.rankings)),
            mock.patch.object(storage,'scalar_samples',side_effect=scalar),
            mock.patch.object(packer,'write_csr_shard',side_effect=write),
            mock.patch.object(trainer,'load_shard',side_effect=lambda _view,name:self.datasets[name.split('.')[0]]),
            mock.patch.object(adapter,'_load_core_inputs',side_effect=lambda _bundle:(self.anchor,self.common,self.canonical,self.routes))):
            self.stack.enter_context(patch)

    def test_preparation_preserves_order_targets_anchor_filter_and_canonical_validation(self):
        import numpy as np
        with mock.patch.object(self.trainer,'load_float_checkpoint',side_effect=AssertionError('seed initialization')), \
                mock.patch.object(self.trainer,'train_seed_candidate',side_effect=AssertionError('training')):
            prepared=campaign.prepare_training_inputs(self.root,self.phase)
        inputs=prepared['inputs']
        self.assertEqual(self.group_visits,['v1','v2','t1','t2'])
        self.assertEqual([row[0] for row in self.packing],['train','validation'])
        self.assertEqual([sample.value for sample in self.packing[0][1]],[.1,.2,.3])
        self.assertEqual([sample.value for sample in self.packing[1][1]],[.9,.8])
        self.assertIs(inputs.new,self.datasets['train'])
        self.assertIs(inputs.common_adjudicator,self.common)
        self.assertIs(inputs.canonical_validation,self.canonical)
        self.assertIsNot(inputs.canonical_validation,self.datasets['validation'])
        self.assertIs(inputs.successor_rankings,self.rankings)
        self.assertEqual(prepared['anchor_filter']['removed_rows'],1)
        for name in ('targets','weights','group_ids'):
            np.testing.assert_array_equal(getattr(inputs.anchor,name),getattr(self.anchor,name)[[0,2]])
        for ordinal,original in enumerate((0,2)):
            np.testing.assert_array_equal(inputs.anchor.active_row(ordinal),self.anchor.active_row(original))
        self.assertFalse((self.root/self.phase/'initialization-measurement.json').exists())
        self.assertFalse((self.root/self.phase/'training.json').exists())

    def test_existing_audit_and_arrays_reproduce_exactly_without_reinterpreting_smoke(self):
        import numpy as np
        first=campaign.prepare_training_inputs(self.root,self.phase)
        audit=first['audit']; directory=self.root/self.phase
        expected={'schema':campaign.ID+'.training-input-audit.v2','bundle':self.plan['bundle'],
            'exclusion_index':campaign.record(self.root/'exclusions/anchor-derived.json'),
            'position_closure':campaign.record(directory/'positions.json'),'labels':campaign.record(directory/'labels.json'),
            'ranking_store':campaign.record(self.index),'shards':{split:{
                'manifest':campaign.record(directory/'shards'/(split+'.json')),
                'npz':campaign.record(directory/'shards'/(split+'.npz'))} for split in ('train','validation')},
            'anchor_duplicates_removed':1,'smoke_qualification_eligible':None,'protected_tests_opened':False}
        self.assertEqual({key:value for key,value in audit.items() if key!='body_sha256'},expected)
        before=(directory/'training-input-audit.json').read_bytes()
        second=campaign.prepare_training_inputs(self.root,self.phase)
        self.assertEqual(first['audit'],second['audit'])
        self.assertEqual(first['anchor_filter'],second['anchor_filter'])
        self.assertEqual(first['inputs'].source_routes,second['inputs'].source_routes)
        for name in ('new','anchor','common_adjudicator','canonical_validation'):
            for field in ('indptr','indices','targets','weights','group_ids'):
                np.testing.assert_array_equal(getattr(getattr(first['inputs'],name),field),getattr(getattr(second['inputs'],name),field))
        with self.assertRaisesRegex(ValueError,'immutable artifact differs'):
            campaign.prepare_training_inputs(self.root,self.phase,smoke=True)
        self.assertEqual(before,(directory/'training-input-audit.json').read_bytes())

    def test_smoke_keeps_eligible_anchor_filter_and_original_full_closure_binding(self):
        directory=self.root/self.phase
        eligible=directory/'eligible-positions.json'; campaign.seal(eligible,{'rows':self.positions})
        (directory/'positions.json').unlink(); campaign.seal(directory/'positions.json',{'rows':[]})
        (directory/'labels.json').unlink(); campaign.seal(directory/'labels.json',{
            'merged':self.labels['merged'],'positions':campaign.record(eligible)})
        prepared=campaign.prepare_training_inputs(self.root,self.phase,smoke=True)
        self.assertEqual(prepared['anchor_filter']['removed_rows'],1)
        self.assertIs(prepared['audit']['smoke_qualification_eligible'],False)
        self.assertEqual(prepared['audit']['position_closure'],campaign.record(directory/'positions.json'))

    def test_train_entry_checks_recipes_executor_and_profile_before_preparing(self):
        from tools import compact_value_bfm_training_resources_v2 as resources
        from tools import compact_value_bfm_intervention_v2 as intervention
        with mock.patch.object(campaign,'prepare_training_inputs') as prepare:
            with self.assertRaisesRegex(ValueError,'ranking recipe'):
                campaign.train_models(self.root,self.phase,ranking_weights=(.1,))
            with mock.patch.object(self.process,'executor_mode',side_effect=ValueError('invalid executor')):
                with self.assertRaisesRegex(ValueError,'invalid executor'):
                    campaign.train_models(self.root,self.phase)
            with mock.patch.object(self.process,'executor_mode',return_value='threads'), \
                    mock.patch.object(resources,'expected_workers',side_effect=ValueError('invalid workers')):
                with self.assertRaisesRegex(ValueError,'invalid workers'):
                    campaign.train_models(self.root,self.phase)
            with mock.patch.object(self.process,'executor_mode',return_value='threads'), \
                    mock.patch.object(resources,'expected_workers',return_value=2), \
                    mock.patch.object(intervention,'expected_qat_profile',return_value='standard-v1'), \
                    mock.patch.object(self.trainer,'resolve_qat_profile',return_value=types.SimpleNamespace(name='other')):
                with self.assertRaisesRegex(ValueError,'QAT profile differs'):
                    campaign.train_models(self.root,self.phase)
            prepare.assert_not_called()

    def test_cli_uses_context_parent_lease_and_releases_large_inputs_before_unlocking(self):
        parent=self.root/'parent'; parent.mkdir()
        campaign.seal(parent/'campaign.json',{'fixture':'parent'})
        context=parent/'phases/attempt-002-pilot'; context.mkdir(parents=True)
        campaign.seal(context/'campaign.json',{'parent_campaign':campaign.record(parent/'campaign.json'),
            'heavy_stage_root':str(parent)})
        audit=campaign.seal(context/self.phase/'training-input-audit.json',{'ranking_store':{},'shards':{}})
        events=[]
        class Inputs:
            new=(1,2); anchor=(3,); common_adjudicator=(); canonical_validation=(4,)
            successor_rankings=types.SimpleNamespace(train=(1,2),validation=(3,))
        def preparation(*args,**kwargs):
            inputs=Inputs()
            weakref.finalize(inputs,lambda:events.append('inputs-released'))
            return {'inputs':inputs,'audit':audit}
        original_lease=campaign.lease
        @contextlib.contextmanager
        def observed_lease(path):
            with original_lease(path):
                events.append('locked'); yield; events.append('unlocking')
        argv=['campaign','--root',str(context),'prepare-training-inputs','--phase',self.phase]
        with mock.patch.object(campaign.sys,'argv',argv), mock.patch.object(campaign,'prepare_training_inputs') as prepare, original_lease(parent):
            with self.assertRaises(BlockingIOError): campaign.main()
            prepare.assert_not_called()
        output=io.StringIO()
        with mock.patch.object(campaign.sys,'argv',argv), \
                mock.patch.object(campaign,'prepare_training_inputs',side_effect=preparation) as prepare, \
                mock.patch.object(campaign,'lease',side_effect=observed_lease), contextlib.redirect_stdout(output):
            campaign.main()
        prepare.assert_called_once_with(context,self.phase,smoke=False)
        self.assertEqual(events,['locked','inputs-released','unlocking'])
        summary=json.loads(output.getvalue())
        self.assertEqual(summary['samples']['new'],2)
        self.assertEqual(summary['ranking_groups'],{'train':2,'validation':1})
        self.assertFalse(summary['seed_initialization_started']); self.assertFalse(summary['real_training_started'])


if __name__=='__main__': unittest.main()
