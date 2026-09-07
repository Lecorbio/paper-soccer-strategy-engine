import pathlib
import tempfile
import unittest
from unittest import mock
from tools import compact_value_bfm_full_v2 as full
from tools import compact_value_bfm_training_resources_v2 as resources


class FullAdmissionTests(unittest.TestCase):
    def test_offline_success_cannot_authorize_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            full.campaign.seal(root/'pilot/pilot-outcome.json',{'status':'offline-qualified','admitted':True,
                'games':0,'wins':0,'failures':0})
            with self.assertRaisesRegex(ValueError,'actually admitted pilot'):
                full.admitted_pilot(root,'pilot')


class FourthFullContextTests(unittest.TestCase):
    def setUp(self):
        temporary=self.enterContext(tempfile.TemporaryDirectory())
        self.root=pathlib.Path(temporary).resolve();self.phase='attempt-004-pilot'
        self.context=self.root/'phases'/self.phase
        self.inputs={}
        for name in ('attempt_one_initial_checkpoint','teacher_runtime','attempt_zero_runtime','admitted-runtime','admitted-source'):
            path=self.root/name;full.campaign.once(path,name.encode())
            self.inputs[name]=full.campaign.record(path)
        full.campaign.seal(self.root/'campaign.json',{'inputs':self.inputs})
        self.parent=full.campaign.seal(self.context/'campaign.json',{
            'attempt':4,'phase':'pilot','parent_campaign':full.campaign.record(self.root/'campaign.json'),
            'inputs':{key:self.inputs[key] for key in ('attempt_one_initial_checkpoint','teacher_runtime','attempt_zero_runtime')},
            'qat_profile':'retention-first-low-rate-v1','qat_profile_contract':{'frozen':'retention'},
            'intervention':{'frozen':'after-three'},'completed_unsuccessful_trained_attempts':3,
            'previous_failed_attempts':[{'attempt':value,'carry':{'frozen':value}} for value in (1,2,3)],
            'pilot_games':2000,'pilot_training_roster':{'lambdas':[0,.1,.25]},'exclusions':[]})
        self.pilot=full.campaign.seal(self.context/self.phase/'pilot-outcome.json',{
            'selected':{'lambda':.25,'runtime':self.inputs['admitted-runtime'],'source':self.inputs['admitted-source']},
            'development_exclusions':[]})
        for name in ('anchor-derived.json','prior-search-validation.json'):
            full.campaign.seal(self.root/'exclusions'/name,{'fingerprints':[]})
        self.enterContext(mock.patch.object(full,'admitted_pilot',return_value=self.pilot))
        self.fingerprints=self.enterContext(mock.patch.object(full,'pilot_fingerprints',return_value=[]))
        self.profile=self.enterContext(mock.patch.object(full.intervention,'expected_qat_profile',
            return_value='retention-first-low-rate-v1'))
        self.enterContext(mock.patch.object(resources,'execution_fields',return_value={}))
        self.enterContext(mock.patch.object(resources,'expected_workers',return_value=4))
        self.enterContext(mock.patch.object(resources,'check_resume'))

    def prepare(self):
        return full.prepare(self.root,self.context,self.phase)

    def test_fourth_full_keeps_intervention_parents_and_original_float_with_six_seed_roster(self):
        before=(self.context/'campaign.json').read_bytes()
        context,phase,contract=self.prepare()
        self.assertEqual((context.name,phase),('attempt-004-full','attempt-004-full'))
        for key in ('parent_campaign','qat_profile','qat_profile_contract','intervention',
                    'completed_unsuccessful_trained_attempts','previous_failed_attempts'):
            self.assertEqual(contract[key],self.parent[key])
        self.assertEqual(contract['pilot_context'],full.campaign.record(self.context/'campaign.json'))
        self.assertEqual(contract['admitted_pilot'],full.campaign.record(self.context/self.phase/'pilot-outcome.json'))
        self.assertEqual(contract['inputs'],{**self.parent['inputs'],'attempt_zero_runtime':self.inputs['admitted-runtime']})
        self.assertEqual(contract['candidate_lineage']['initial_float'],self.inputs['attempt_one_initial_checkpoint'])
        self.assertEqual(contract['full_training_roster'],{'lambdas':[0,.25],'seeds':[20260907,20260908,20260909]})
        self.assertNotIn('pilot_training_roster',contract)
        self.assertEqual(self.prepare(),(context,phase,contract))
        self.assertEqual(before,(self.context/'campaign.json').read_bytes())
        self.assertTrue(any(call.args[0].get('phase')=='full' for call in self.profile.call_args_list))

    def test_fourth_full_rejects_foreign_parent_and_frozen_admission_graft_before_carry(self):
        self.parent['parent_campaign']={'other':'campaign'}
        path=self.context/'campaign.json';path.unlink()
        full.campaign.seal(path,{key:value for key,value in self.parent.items() if key!='body_sha256'})
        with self.assertRaisesRegex(ValueError,'canonical admitted pilot and parent'):self.prepare()
        self.fingerprints.assert_not_called()

    def test_fourth_full_resume_rejects_another_admitted_pilot(self):
        context,_,contract=self.prepare();path=context/'campaign.json'
        path.unlink();full.campaign.seal(path,{**{key:value for key,value in contract.items() if key!='body_sha256'},
            'admitted_pilot':{'foreign':'outcome'}})
        self.fingerprints.reset_mock()
        with self.assertRaisesRegex(ValueError,'resume changed its admitted pilot'):self.prepare()
        self.fingerprints.assert_not_called()

    def test_unbound_profile_never_materializes_full_context(self):
        self.profile.side_effect=ValueError('missing after-three intervention')
        with self.assertRaisesRegex(ValueError,'after-three intervention'):self.prepare()
        self.assertFalse((self.root/'phases/attempt-004-full').exists())
        self.fingerprints.assert_not_called()

class FourthFullInterventionIntegrationTests(unittest.TestCase):
    def test_actual_after_three_binding_allows_full_student_change_and_preserves_profile(self):
        from tests.codingame.test_compact_value_bfm_intervention_v2 import fourth_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root,_,_,_,pilot_contract=fourth_fixture(pathlib.Path(tmp))
            phase='attempt-004-pilot';context=root/'phases'/phase
            full.campaign.seal(context/'campaign.json',pilot_contract)
            runtime=root/'admitted-runtime';source=root/'admitted-source'
            full.campaign.once(runtime,b'fresh admitted runtime');full.campaign.once(source,b'fresh admitted source')
            selected={'lambda':.1,'runtime':full.campaign.record(runtime),'source':full.campaign.record(source)}
            admission=full.campaign.seal(context/phase/'pilot-outcome.json',{
                'selected':selected,'development_exclusions':[]})
            with mock.patch.object(full,'admitted_pilot',return_value=admission), \
                    mock.patch.object(full,'pilot_fingerprints',return_value=[]):
                full_context,full_phase,contract=full.prepare(root,context,phase)
                self.assertEqual(full.prepare(root,context,phase),(full_context,full_phase,contract))
            self.assertEqual(full.intervention.expected_qat_profile(contract),'retention-first-low-rate-v1')
            self.assertEqual(contract['intervention'],pilot_contract['intervention'])
            self.assertEqual(contract['inputs']['attempt_one_initial_checkpoint'],
                             pilot_contract['inputs']['attempt_one_initial_checkpoint'])
            self.assertEqual(contract['inputs']['attempt_zero_runtime'],selected['runtime'])
            self.assertEqual([row['attempt'] for row in contract['previous_failed_attempts']],[1,2,3])


if __name__=='__main__':unittest.main()
