"""Legacy size-bearing references must preserve exact bytes and path scope."""
import tempfile
from pathlib import Path
import unittest
from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_data_v2 as data


class LiveArtifactReferenceTests(unittest.TestCase):
    def setUp(self):
        parent=c.ROOT/"build/live-eight-reference-tests"
        parent.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/"artifact.json"
        self.path.write_text('{"fixture":true}\n')

    def test_two_field_reference_gains_verified_size(self):
        reference=c.record(self.path)
        converted=data.legacy_reference(reference)
        self.assertEqual(converted["bytes"],self.path.stat().st_size)
        self.assertEqual(converted["sha256"],reference["sha256"])
        self.assertEqual(data.data.bound(converted),self.path.resolve())

    def test_legacy_reference_remains_exact(self):
        reference=data.data.record(self.path)
        self.assertEqual(data.legacy_reference(reference),reference)

    def test_bad_size_or_hash_are_rejected(self):
        reference=data.data.record(self.path)
        for changed in ({**reference,"bytes":reference["bytes"]+1},
                        {**reference,"sha256":"0"*64}):
            with self.assertRaisesRegex(ValueError,"SHA or byte count changed"):
                data.legacy_reference(changed)

    def test_unexpected_fields_do_not_weaken_identity(self):
        with self.assertRaisesRegex(ValueError,"unexpected artifact reference fields"):
            data.legacy_reference({**c.record(self.path),"unchecked":True})


if __name__=="__main__":
    unittest.main()
