"""Mixture, exact numerical resume, and native float checks without live games."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools import rank_two_live_eight_training_v4 as training
from tests.codingame.test_rank_two_focused_training_v1 import groups


class Training(unittest.TestCase):
    def test_native_compiler_policy_is_platform_scoped(self):
        darwin=training.native_probe.compiler_flags(platform='darwin')
        linux=training.native_probe.compiler_flags(platform='linux')
        self.assertIn('-fintegrated-cc1',darwin)
        self.assertIn('-Wl,-ld_classic',darwin)
        self.assertNotIn('-Wl,-ld_classic',linux)
        self.assertIn('-ffp-contract=off',linux)

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

    def test_journal_retains_root_order_once_and_restores_exact_state(self):
        a,p=training.retained.initialize('dd8',2026101017)
        optimizer=training.retained.core.AdamW(p,learning_rate=.001,weight_decay=1e-5)
        binding=dict(path='unit',sha256='unit')
        order=['root-'+str(i) for i in range(598)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index in range(2):
                training.update_root(p,a,optimizer,groups(),None,6)
                cursor=dict(phase='qat',epoch=0,next_root=index+1,root_order=order,rng_seed=2026101017)
                training.resumable(root,p,optimizer,cursor,binding)
            head=training.campaign.read(root/'HEAD.json')
            journal=training.campaign.read(training.campaign.verify(head['journal']))
            saved_order=training.campaign.read(training.campaign.verify(journal['root_order']))
            self.assertEqual(saved_order['root_order'],order)
            self.assertNotIn('root_order',journal['cursor'])
            self.assertLess(Path(head['journal']['path']).stat().st_size,1024)
            receipt=training.campaign.read(training.campaign.verify(head['receipt']))
            self.assertEqual(receipt['cursor']['root_order'],order)
            expected={k:v.copy() for k,v in p.items()}
            for value in p.values():value.fill(0)
            training.storage.restore(receipt,p,optimizer,binding)
            for key in p:np.testing.assert_array_equal(p[key],expected[key])

    def test_partial_epoch_validation_resume_reuses_only_identical_state(self):
        _,p=training.retained.initialize('dd8',2026101017)
        optimizer=training.retained.core.AdamW(p,learning_rate=.001,weight_decay=1e-5)
        cursor=dict(phase='float',epoch=0,next_root=598)
        binding=dict(path='unit',sha256='unit')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.psc'
            receipt=training.epoch_checkpoint(path,p,optimizer,cursor,binding)
            self.assertEqual(training.epoch_checkpoint(path,p,optimizer,cursor,binding),receipt)
            optimizer.second['w1'][0,0]+=1
            with self.assertRaisesRegex(ValueError,'exact numerical resume'):
                training.epoch_checkpoint(path,p,optimizer,cursor,binding)


if __name__=='__main__':unittest.main()
