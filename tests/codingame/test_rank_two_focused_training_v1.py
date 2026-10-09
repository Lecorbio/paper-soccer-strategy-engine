"""Numerical and recovery tests using explicitly synthetic, nonproduction rows."""
import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from tools import rank_two_focused_training_v1 as training


def groups(exhaustive=True):
    return [dict(mover=0,exhaustive=exhaustive,split="train",successors=[
        dict(active=[0,316,324],teacher_value=.3,value_mover=0),
        dict(active=[1,317,325],teacher_value=-.2,value_mover=1),
        dict(active=[2,318,326],teacher_value=-.1,value_mover=0)])]


class Numerics(unittest.TestCase):
    def test_fresh_reproducible_initialization_for_both_widths(self):
        for profile in ("dd8","dd12wide"):
            a,p=training.initialize(profile,2026100701);_,again=training.initialize(profile,2026100701)
            self.assertEqual(p["w2"].shape,(a.hidden_one,a.hidden_two))
            for key in p:np.testing.assert_array_equal(p[key],again[key])

    def test_gradient_finite_differences(self):
        a,p=training.initialize("dd8",2026100701);g=groups()
        active,target,signs,spans=training.root_rows(g)
        predictions,cache=training.forward(p,a,active)
        _,_,derivative=training.objective(g,predictions,target,signs,spans)
        gradient=training.core._network_gradients(p,a,active,cache,derivative,p)
        for key,index in (("w1",(0,0)),("w2",(0,0)),("w3",(0,))):
            original=p[key][index];epsilon=np.float32(.001)
            values=[]
            for offset in (-epsilon,epsilon):
                p[key][index]=original+offset
                q,_=training.forward(p,a,active);h,r,_=training.objective(g,q,target,signs,spans);values.append(h+.25*r)
            p[key][index]=original
            numerical=(values[1]-values[0])/(2*epsilon)
            self.assertAlmostEqual(float(gradient[key][index]),float(numerical),delta=2e-4)

    def test_nonexhaustive_has_no_ranking_loss(self):
        a,p=training.initialize("dd8",2026100701);g=groups(False)
        active,target,signs,spans=training.root_rows(g);prediction,_=training.forward(p,a,active)
        self.assertEqual(training.objective(g,prediction,target,signs,spans)[1],0)

    def test_mover_signs_and_exact_proof_labels(self):
        active,target,signs,spans=training.root_rows(groups())
        np.testing.assert_array_equal(signs,[1,-1,1])
        g=groups();g[0]["successors"][0].update(proof={"solved":True,"proven_winner":0},teacher_value=.5)
        with self.assertRaises(ValueError):training.root_rows(g)

    def test_train_only_scale_calibration_and_small_fixture_learning(self):
        a,p=training.initialize("dd8",2026100701);g=groups()
        optimizer=training.core.AdamW(p,learning_rate=.001,weight_decay=1e-5)
        first=training.update_root(p,a,optimizer,g)["objective"]
        for _ in range(200):last=training.update_root(p,a,optimizer,g)["objective"]
        self.assertLess(last,first/2)
        scales,report=training.calibrate(p,a,[g]);self.assertEqual(report["split"],"train")
        self.assertEqual(set(scales),{"w1","w2","w3"})
        bad=copy.deepcopy(g);bad[0]["split"]="validation"
        with self.assertRaises(ValueError):training.calibrate(p,a,[bad])

    def test_stopping_reason_and_exact_recovery(self):
        self.assertEqual(training.stopping([.5]*10,10,40),"plateau")
        self.assertEqual(training.stopping([.5]*40,10,40),"budget-limited")
        a,p=training.initialize("dd8",2026100701);optimizer=training.core.AdamW(p,learning_rate=.001,weight_decay=1e-5)
        training.update_root(p,a,optimizer,groups());binding={"path":"fixture-plan","sha256":"fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            receipt=training.checkpoint(Path(temporary)/"checkpoint.npz",p,optimizer,{"phase":"float","epoch":0,"root":1,"rng_seed":2026100701},binding)
            training.update_root(p,a,optimizer,groups());expected={key:value.copy() for key,value in p.items()}
            cursor=training.restore(receipt,p,optimizer,binding);self.assertEqual(cursor["root"],1)
            training.update_root(p,a,optimizer,groups())
            for key in p:np.testing.assert_array_equal(p[key],expected[key])
            with self.assertRaises(ValueError):training.restore(receipt,p,optimizer,{"path":"other","sha256":"other"})


if __name__=="__main__":unittest.main()
