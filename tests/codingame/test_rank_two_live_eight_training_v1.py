"""Mixture, exact numerical resume, and native float checks without live games."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools import rank_two_live_eight_training_v1 as training
from tests.codingame.test_rank_two_focused_training_v1 import groups


class Training(unittest.TestCase):
    def test_half_mass_for_unequal_pools(self):
        masses=training.root_masses(['a','b','c','live'],['live'])
        self.assertAlmostEqual(sum(masses[r] for r in ['a','b','c'])/4,.5)
        self.assertAlmostEqual(masses['live']/4,.5)
        with self.assertRaises(ValueError):training.root_masses(['a'],[])

    def test_current_state_is_exact_and_epoch_independent(self):
        a,p=training.retained.initialize('dd8',2026101001)
        optimizer=training.retained.core.AdamW(p,learning_rate=.001,weight_decay=1e-5)
        binding=dict(path='unit',sha256='unit')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for step in range(3):
                training.update_root(p,a,optimizer,groups(),None,6)
                training.resumable(root,p,optimizer,dict(phase='float',epoch=0,next_root=step+1),binding)
            expected={k:v.copy() for k,v in p.items()}
            head=training.campaign.read(root/'HEAD.json');receipt=training.campaign.read(training.campaign.verify(head['receipt']))
            for v in p.values():v.fill(0)
            training.storage.restore(receipt,p,optimizer,binding)
            for k in p:np.testing.assert_array_equal(p[k],expected[k])
            self.assertEqual(len(list((root/'transient').iterdir())),1)
            self.assertEqual(len(list((root/'journal').iterdir())),3)

    def test_native_float_probe_matches_literal_reference(self):
        a,p=training.retained.initialize('dd8',2026101002)
        active=[[0,316,6300],[1,317,6299]]
        with tempfile.TemporaryDirectory() as directory:
            binary=training.float_probe(p,Path(directory))
            actual=training.native_probe.query(binary,['eval 3 '+' '.join(map(str,row)) for row in active])
        expected=training.retained.forward(p,a,active)[0]
        np.testing.assert_array_equal(np.asarray(actual,dtype=np.float32),expected)


if __name__=='__main__':unittest.main()
