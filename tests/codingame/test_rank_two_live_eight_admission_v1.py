"""A passed receipt alone cannot admit different tensors or clocks."""
import unittest

from tools import rank_two_live_eight_admission_v1 as admission


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(source={'sha256':'source'}, runtime={'sha256':'tensor'}, clocks_ms=[990,180])

    def test_mismatched_clock_or_tensor_is_rejected(self):
        for changed in ({'runtime':{'sha256':'different'}}, {'clocks_ms':[550,140]}, {'source':{'sha256':'different'}}, {'passed':False}):
            proof = dict(self.model, passed=True)
            proof.update(changed)
            with self.subTest(changed=changed), self.assertRaises(PermissionError):
                admission.require_identity(proof, self.model)

    def test_exact_source_clock_and_tensor_pass(self):
        admission.require_identity(dict(self.model, passed=True), self.model)


if __name__ == '__main__':
    unittest.main()
