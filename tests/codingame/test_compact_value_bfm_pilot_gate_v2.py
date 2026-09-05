import pathlib
import tempfile
import unittest
from tools import compact_value_bfm_pilot_gate_v2 as gate


class PilotGateTests(unittest.TestCase):
    def test_bank_cannot_open_before_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=pathlib.Path(tmp)
            with self.assertRaisesRegex(ValueError,'model selection must be frozen'):
                gate.prepare_bank(context,'pilot',{'selected':None,'status':'offline-rejected'})
            self.assertFalse((context/'pilot/rank4-screen').exists())

if __name__=='__main__':unittest.main()
