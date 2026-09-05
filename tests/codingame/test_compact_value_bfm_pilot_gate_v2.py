import pathlib
import tempfile
import unittest
from unittest import mock
import subprocess
from tools import compact_value_bfm_pilot_gate_v2 as gate


class PilotGateTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(gate.select,'kqueue'),'kqueue host required')
    def test_one_shot_wait_closes_kqueue_without_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'selection.json';closed=[]
            class Queue:
                def control(self,*args):path.write_text('{}')
                def close(self):closed.append(True)
            process=subprocess.CompletedProcess([],0,'python compact_value_bfm_pilot_selection_v2.py --phase pilot','')
            with mock.patch.object(gate.select,'kqueue',return_value=Queue()),mock.patch.object(gate.subprocess,'run',return_value=process):
                gate.wait_for_selection(path,123,'pilot')
            self.assertEqual(closed,[True])

    def test_bank_cannot_open_before_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=pathlib.Path(tmp)
            with self.assertRaisesRegex(ValueError,'model selection must be frozen'):
                gate.prepare_bank(context,'pilot',{'selected':None,'status':'offline-rejected'})
            self.assertFalse((context/'pilot/rank4-screen').exists())

    def test_screen_carry_includes_played_boundaries_and_terminal_features(self):
        campaign=gate.campaign
        with tempfile.TemporaryDirectory() as tmp:
            context=pathlib.Path(tmp);directory=context/'pilot/rank4-screen';directory.mkdir(parents=True)
            execution=directory/'execution.json';campaign.seal(execution,{'fixture':True})
            checked={'config':{'trajectory_schema':'papersoccer.compact-value-bfm-rank4-trajectories.v1'},
                'games':[{'root_transcript':'0/0','transcript':'0/0/0/0/0/0'}]}
            records=gate.played_exclusions(context,'pilot',execution,checked,{'exclusions':[],'tsv':{'sha256':'a'*64}})
            documents=[campaign.read(campaign.verify(item)) for item in records]
            values={doc['domain']:set(doc['fingerprints']) for doc in documents}
            state=campaign.features.ReplayState()
            campaign.features.apply_complete_turn(state,state.to_move,'0')
            before=campaign.fingerprints(state)
            self.assertTrue(all(value not in values[domain] for domain,value in before.items()))
            for _ in range(5):
                campaign.features.apply_complete_turn(state,state.to_move,'0')
                self.assertTrue(all(value in values[domain] for domain,value in campaign.fingerprints(state).items()))
            self.assertIsNotNone(state.winner)
            self.assertTrue(all(doc['includes_terminal_features'] and doc['role']=='mixed-development' for doc in documents))
            self.assertEqual(records,gate.played_exclusions(context,'pilot',execution,checked,{'exclusions':[],'tsv':{'sha256':'a'*64}}))

if __name__=='__main__':unittest.main()
