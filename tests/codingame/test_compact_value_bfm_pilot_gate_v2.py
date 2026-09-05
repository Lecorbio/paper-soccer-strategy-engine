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

if __name__=='__main__':unittest.main()
