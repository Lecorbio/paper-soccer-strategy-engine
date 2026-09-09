"""Bounded protocol/lifecycle tests for optional validation prediction helpers."""
import copy
import multiprocessing.process
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

for name in ('MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
import numpy as np
from tools import compact_value_bfm_train as t
from tools import compact_value_bfm_prediction_pool as p
from tools import compact_value_bfm_seed_process_v2 as process


class Features:
    def __init__(self,rows):
        self.indptr=np.arange(rows+1,dtype='<i8')
        self.indices=(np.arange(rows)%10).astype('<u2')
    def __len__(self):return len(self.indptr)-1


def datasets():return {'common_adjudicator':Features(17),'canonical_validation':Features(23)}


class PredictionPoolTests(unittest.TestCase):
    def test_active_concurrency_admission_uses_actual_executor(self):
        self.assertIsNone(p.policy(1,4))
        for workers,seeds in ((2,1),(2,2),(4,1)):
            self.assertEqual(p.policy(workers,seeds)['maximum_active_numerical_streams'],workers*seeds)
        for workers,seeds in ((4,2),(2,4),(True,1),(3,1),(2,0)):
            with self.assertRaises(p.PredictionError):p.policy(workers,seeds)
        for executor,workers in ((process.MODE4,2),(process.MODE2,4)):
            with self.assertRaises(ValueError):process.prediction_options({'executor':executor,'validation_workers':workers,'jobs':[{}]})
        self.assertEqual(process.prediction_options({'executor':process.MODE2,'validation_workers':2}),{'validation_workers':2,'_seed_concurrency':2})
        self.assertEqual(process.prediction_options({'jobs':[{}],'validation_workers':4}),{'validation_workers':4,'_seed_concurrency':1})
        with self.assertRaises(ValueError):process.prediction_options({'jobs':[{},{}],'validation_workers':4})

    def test_model_packet_is_immutable_primitive_snapshot_and_clears_quantized_mode(self):
        arch=t.ARCHITECTURES['capacity-12x8'];params=t.initialize_parameters(arch,t.FIXED_SEEDS[0])
        q=t.quantize_fixed(params,arch,{'w1':.125,'w2':.25,'w3':.125})
        packet,digest,mode=p.model_packet(params,arch,q,t)
        header,buffers=pickle.loads(packet)
        self.assertTrue(all(type(b) is bytes for b in buffers));self.assertEqual(mode,'runtime.v1')
        initial={k:v.copy() for k,v in params.items()};initial_codes={k:v.copy() for k,v in q.integer.items()}
        params['w3']+=np.float32(.25);q.integer['w2'][0,0]=np.int8(3);q.scales['w3']=np.float32(.5)
        old,_,old_q=p._decode_model(packet,digest,t)
        for key in initial:np.testing.assert_array_equal(old[key],initial[key]);np.testing.assert_array_equal(old_q.integer[key],initial_codes[key])
        fp,fd,fm=p.model_packet(params,arch,None,t);decoded,_,none=p._decode_model(fp,fd,t)
        self.assertEqual(fm,'float');self.assertIsNone(none);np.testing.assert_array_equal(decoded['w3'],params['w3'])
        self.assertNotEqual(fd,digest)
        with self.assertRaises(p.PredictionError):p._decode_model(packet,'0'*64,t)

    def test_channel_or_noncontiguous_models_reject_before_dispatch(self):
        arch=t.ARCHITECTURES['capacity-12x8'];params=t.initialize_parameters(arch,t.FIXED_SEEDS[0])
        q=t.quantize_channels(params,arch,{k:np.full(n,.125,np.float32) for k,n in t.CHANNEL_SCALE_COUNTS.items()})
        with self.assertRaisesRegex(p.PredictionError,'channel'):p.model_packet(params,arch,q,t)
        params['w2']=np.asfortranarray(params['w2'])
        with self.assertRaisesRegex(p.PredictionError,'C-contiguous'):p.model_packet(params,arch,None,t)

    def test_failed_start_after_real_spawn_registers_and_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp,t.native_thread_execution_scope():
            pool=p.ValidationPredictionPool(datasets(),Path(tmp)/'pool',2,trainer=t)
            original=multiprocessing.process.BaseProcess.start
            def start_then_raise(child):
                original(child)
                raise RuntimeError('injected after successful spawn')
            with mock.patch.object(multiprocessing.process.BaseProcess,'start',start_then_raise):
                with self.assertRaisesRegex(RuntimeError,'injected'):pool.__enter__()
            self.assertEqual(len(pool.processes),1)
            self.assertIsNotNone(pool.processes[0].pid)
            self.assertFalse(pool.processes[0].is_alive())
            self.assertIsNotNone(pool.processes[0].exitcode)
            self.assertTrue(all(w.closed for w in (*pool.watches,*pool.watch_reads)))
            self.assertFalse(pool.reserved);self.assertEqual(p._RESERVED,0)

    def test_unjoined_child_retains_reservation_and_failed_ownership(self):
        child=mock.Mock(pid=123,exitcode=None)
        child.is_alive.return_value=True
        pool=p.ValidationPredictionPool.__new__(p.ValidationPredictionPool)
        pool.closed=False;pool.ready=[];pool.processes=[child];pool.commands=[];pool.watches=[];pool.watch_reads=[]
        pool.replies=None;pool.start_attempted=[True];pool.workers=2;pool.reserved=True
        previous=p._RESERVED;p._RESERVED=2
        try:
            with self.assertRaisesRegex(p.PredictionError,'reservation retained'):pool.close(failed=True)
            self.assertEqual(p._RESERVED,2);self.assertTrue(pool.reserved);self.assertFalse(pool.closed)
            child.terminate.assert_called_once();child.kill.assert_called_once()
        finally:p._RESERVED=previous

    def test_generation_acknowledgements_are_exact_and_distinct(self):
        pool=p.ValidationPredictionPool.__new__(p.ValidationPredictionPool)
        pool.workers=2;pool.generation=3
        rows=[{'kind':'refreshed','slot':0,'generation':3,'model_sha256':'a'},
              {'kind':'refreshed','slot':1,'generation':3,'model_sha256':'a'}]
        with mock.patch.object(pool,'_receive',side_effect=rows):self.assertEqual(pool._barrier('refreshed',model_sha256='a'),rows)
        for bad in ([rows[0],rows[0]],[{**rows[0],'generation':2}],[{**rows[0],'model_sha256':'stale'}]):
            with mock.patch.object(pool,'_receive',side_effect=bad),self.assertRaises(p.PredictionError):pool._barrier('refreshed',model_sha256='a')

    def test_truthful_parallel_native_envelope_preserves_legacy_component(self):
        with t.native_thread_execution_scope() as original:
            self.assertEqual(t.prediction_native_execution(original,None),original)
            settings=t.validation_prediction_settings(4,1)
            envelope=t.prediction_native_execution(original,settings)
        self.assertEqual(t.validate_native_thread_execution(envelope),envelope)
        self.assertNotIn('native_threads_per_seed_maximum',envelope)
        self.assertNotIn('native_threads_per_seed_maximum',envelope['coordinator_kernel'])
        self.assertEqual(envelope['maximum_active_numerical_streams'],4)
        bad=copy.deepcopy(envelope);bad['maximum_active_numerical_streams']=1
        with self.assertRaises(t.TrainingError):t.validate_native_thread_execution(bad)

if __name__=='__main__':unittest.main()
