import pathlib
import tempfile
import unittest
from tools import compact_value_bfm_full_v2 as full


class FullAdmissionTests(unittest.TestCase):
    def test_offline_success_cannot_authorize_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            full.campaign.seal(root/'pilot/pilot-outcome.json',{'status':'offline-qualified','admitted':True,
                'games':0,'wins':0,'failures':0})
            with self.assertRaisesRegex(ValueError,'actually admitted pilot'):
                full.admitted_pilot(root,'pilot')

if __name__=='__main__':unittest.main()
