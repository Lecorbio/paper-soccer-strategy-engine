"""Bounded synthetic training and artifact tests; no campaign corpus execution."""
import copy
import contextlib
import io
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY']='1'
import numpy as np
from tools import compact_value_bfm_train as t


def dataset(n,split='train'):
    return t.Dataset(np.arange(n+1,dtype='<i8'),np.asarray([i%3 for i in range(n)],dtype='<u2'),
        np.asarray([(.6,-.5,.2)[i%3] for i in range(n)],dtype='<f4'),np.ones(n,dtype='<f4'),
        np.zeros(n,dtype='V32'),split,'1'*64,'2'*64)


def training_inputs():
    groups=tuple(t.CompleteTurnGroup(str(i),0,(
        t.CompleteTurnSuccessor('a',np.array([0],dtype='<u2'),.6,0,{}),
        t.CompleteTurnSuccessor('b',np.array([1],dtype='<u2'),-.5,0,{}))) for i in range(16))
    labels=t.SuccessorRankingLabels(groups,groups[:1],{},'a'*64,'b'*64,'c'*64)
    return t.TrainingInputs(new=dataset(1024),anchor=dataset(3072),common_adjudicator=dataset(8,'validation'),
        canonical_validation=dataset(9,'validation'),source_routes={},successor_rankings=labels)


class ChannelQATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.root=Path(cls.directory.name).resolve()
        cls.arch=t.ARCHITECTURES['capacity-12x8'];cls.arm=t.ARMS['search-target'];cls.inputs=training_inputs()
        cls.initial=t.initialize_parameters(cls.arch,t.FIXED_SEEDS[0]);cls.initial['w1'][:]=0
        cls.initial['w1'][:3]=np.linspace(-.02,.02,36,dtype=np.float32).reshape(3,12)
        cls.initial['w3'][:]=np.linspace(-.04,.04,8,dtype=np.float32)
        cls.warmup=t.train_float_seed(cls.inputs,cls.arch,cls.arm,t.FIXED_SEEDS[0],maximum_epochs=1,patience=1,
            learning_rate=.00006,initial_parameters=cls.initial,ranking_weight=0.)
        cls.result=t.run_fixed_scale_qat(cls.warmup,cls.inputs,cls.arch,cls.arm,t.FIXED_SEEDS[0],
            qat_profile=t.CHANNEL_PREDICTION_QAT_PROFILE,original_parameters=cls.initial,calibration_directory=cls.root,ranking_weight=0.)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_new_contract_closed_and_historical_contract_hashes_unchanged(self):
        expected={'standard-v1':'9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
            'refined-adaptive-scales-v1':'7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd',
            'retention-first-low-rate-v1':'3cdbd16892860b6db547ca510e97604ba7742ae07392df64fd817b315b975de3'}
        for name,digest in expected.items():self.assertEqual(t.qat_profile_contract(name)['body_sha256'],digest)
        contract=t.qat_profile_contract(t.CHANNEL_PREDICTION_QAT_PROFILE)
        self.assertEqual(contract['schema'],'papersoccer.compact-value-bfm-qat-profile.v2')
        self.assertEqual(contract['quantization']['scale_counts'],{'w1':12,'w2':8,'w3':1})
        self.assertEqual(contract['schedule']['float_warmup_learning_rate'],.00006)
        self.assertEqual(contract['schedule']['qat_learning_rate'],.0000625)
        self.assertEqual(contract['scale_selection']['after_each_qat_epoch']['sweeps'],2)
        self.assertFalse(contract['scale_selection']['initial_calibration']['heldout_used_for_calibration'])
        digest=contract['body_sha256'];contract['training_fixture']['draw_order'].reverse()
        self.assertEqual(t.qat_profile_contract(t.CHANNEL_PREDICTION_QAT_PROFILE)['body_sha256'],digest)
        with self.assertRaises(t.TrainingError):t.validate_qat_profile_contract(contract)

    def test_actual_warmup_plus_four_all_layer_optimizer_epochs_and_code_changes(self):
        report=self.result.report
        self.assertEqual(self.warmup.epoch,1);self.assertEqual(report['executed_qat_epochs'],[1,2,3,4]);self.assertEqual(report['optimizer_steps'],64)
        self.assertEqual(report['learning_rate'],.0000625);self.assertIsInstance(self.result.quantized,t.ChannelQuantizedWeights)
        for epoch in report['history']:
            self.assertEqual(epoch['fake_quantization']['batches'],16)
            self.assertTrue(epoch['fake_quantization']['all_layers_trainable'])
            for name in ('w1','w2','w3'):
                update=epoch['fake_quantization']['master_parameter_updates'][name]
                self.assertGreater(update['changed_parameters'],0);self.assertTrue(np.isfinite(update['l2_delta']))
        for name in ('w1','w2','w3'):
            self.assertGreater(report['original_initialization_code_evidence'][name]['changed_codes'],0)
        t.validate_qat_execution_evidence(report,expected_profile=t.CHANNEL_PREDICTION_QAT_PROFILE,float_validation_reference=self.warmup.metrics)
        t.validate_successor_schedule_execution(self.warmup.report,report,seed=t.FIXED_SEEDS[0])

    def test_fixed_train_fixture_and_prediction_targets_across_all_five_stages(self):
        report=self.result.report;fixture=report['training_fixture']['identity'];reference=report['training_prediction_reference']
        indices=t._channel_sample_indices(1024,3072)
        self.assertEqual(fixture['sampled_indices'],{name:value.tolist() for name,value in indices.items()})
        self.assertEqual(fixture['datasets'],{name:t.dataset_identity(getattr(self.inputs,name)) for name in ('new','anchor')})
        stages=[report['scale_search'],*[epoch['adaptive_scale_search'] for epoch in report['history']]]
        self.assertEqual([stage['qat_epoch'] for stage in stages],[0,1,2,3,4])
        for stage in stages:
            self.assertEqual(stage['training_reference'],reference);self.assertFalse(stage['heldout_read_by_calibration'])
            self.assertEqual(len(stage['trials']),42)
        self.assertNotEqual(stages[0]['master_parameters'],stages[1]['master_parameters'])
        self.assertEqual(reference['pre_qat_parameters'],stages[0]['master_parameters'])
        for before,after in zip(stages,stages[1:]):self.assertEqual(before['selected_scales'],after['starting_scales'])

    def test_fixture_and_calibration_never_read_heldout_or_labels(self):
        class TrainOnly:
            new=self.inputs.new;anchor=self.inputs.anchor
            def __getattr__(self,name):raise AssertionError('heldout read '+name)
        fixture,document=t._channel_fixture(TrainOnly(),self.root)
        self.assertFalse(hasattr(fixture,'targets'));self.assertFalse(hasattr(fixture,'teacher_predictions'))
        self.assertFalse(np.shares_memory(fixture.weights,self.inputs.anchor.weights))
        reference,refdoc=t._channel_prediction_reference(self.warmup.parameters,self.arch,fixture,document,self.root)
        wrong=reference.copy();wrong[0]+=np.float32(.01);wrong.flags.writeable=False
        with mock.patch.object(t,'predict_dataset',side_effect=AssertionError('forwards must not run for unbound targets')):
            with self.assertRaisesRegex(t.TrainingError,'frozen reference'):
                t._channel_calibrate(self.warmup.parameters,self.arch,fixture,wrong,refdoc,self.result.quantized.scales,self.root,epoch=1)
        with self.assertRaisesRegex(t.TrainingError,'1024new'):t._channel_sample_indices(1023,3072)
        with mock.patch.object(t,'evaluate_validation_pair',side_effect=AssertionError('heldout scales must not run')):
            with self.assertRaisesRegex(t.TrainingError,'training-only channel'):
                t.select_fixed_scales(self.warmup.parameters,self.arch,object(),self.arm,qat_profile=t.CHANNEL_PREDICTION_QAT_PROFILE)
            with self.assertRaisesRegex(t.TrainingError,'training-only channel'):
                t._adapt_fixed_scales(self.warmup.parameters,self.arch,object(),self.arm,{},t.resolve_qat_profile(t.CHANNEL_PREDICTION_QAT_PROFILE),qat_epoch=1,ranking_weight=0.)

    def test_channel_uniform_scales_preserve_mixed_batch_gradient_and_adam_bits_for_each_lambda(self):
        scalar={'w1':np.float32(.025),'w2':np.float32(.125),'w3':np.float32(.0125)}
        vectors={name:np.full(count,scalar[name],dtype=np.float32) for name,count in t.CHANNEL_SCALE_COUNTS.items()}
        for weight in (0.,.1,.25):
            left={name:value.copy() for name,value in self.initial.items()};right={name:value.copy() for name,value in self.initial.items()}
            a=t.AdamW(left,learning_rate=.0000625,weight_decay=t.WEIGHT_DECAY);b=t.AdamW(right,learning_rate=.0000625,weight_decay=t.WEIGHT_DECAY)
            args=self.arch,self.arm,self.inputs,np.arange(64,dtype=np.int64),np.arange(192,dtype=np.int64)
            objective_a=t._train_mixed_batch(left,args[0],args[1],a,*args[2:],fixed_scales=scalar,
                ranking_groups=self.inputs.successor_rankings.train[:2] if weight else None,ranking_weight=weight)
            objective_b=t._train_mixed_batch(right,args[0],args[1],b,*args[2:],fixed_scales=vectors,quantization_granularity='per-output-channel',
                ranking_groups=self.inputs.successor_rankings.train[:2] if weight else None,ranking_weight=weight)
            self.assertEqual(objective_a,objective_b)
            for name in left:self.assertEqual(left[name].tobytes(),right[name].tobytes())

    def test_v2_runtime_roundtrip_evidence_and_strict_shapes_dtypes_codes(self):
        report=self.result.report;digest=t.sha256_bytes(t.canonical_json_bytes(report))
        document=t.runtime_document(self.arch,self.result.quantized,arm=self.arm,seed=t.FIXED_SEEDS[0],float_epoch=1,
            qat_epoch=self.result.qat_epoch,source_bundle_body_sha256='a'*64,qat_profile=t.CHANNEL_PREDICTION_QAT_PROFILE,qat_evidence_sha256=digest)
        architecture,q,selection=t.validate_runtime_document(document)
        self.assertEqual(architecture,self.arch);self.assertEqual(selection['qat_evidence_sha256'],digest)
        for name in q.integer:self.assertEqual(q.integer[name].tobytes(),self.result.quantized.integer[name].tobytes())
        for mode in ('axis','count','boolean-count','noncanonical','zero','nan','profile','evidence'):
            bad=copy.deepcopy(document);bad.pop('body_sha256')
            if mode=='axis':bad['quantization']['scale_axis']='input'
            elif mode=='count':bad['quantization']['scales']['w1'].pop()
            elif mode=='boolean-count':bad['quantization']['scale_counts']['w3']=True
            elif mode=='noncanonical':bad['quantization']['scales']['w1'][0]=.1
            elif mode=='zero':bad['quantization']['scales']['w1'][0]=0.
            elif mode=='nan':bad['quantization']['scales']['w1'][0]=float('nan')
            elif mode=='profile':bad['selection']['qat_profile']=t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE
            else:bad['selection']['qat_evidence_sha256']='wrong'
            with self.subTest(mode=mode),self.assertRaises((t.TrainingError,ValueError)):t.validate_runtime_document(t.body_hashed(bad))
        codes={name:value.copy() for name,value in q.integer.items()};scales={name:value.copy() for name,value in q.scales.items()}
        codes['w1'][0,0]=-4
        with self.assertRaises(t.TrainingError):t.ChannelQuantizedWeights(codes,scales)
        codes['w1'][0,0]=0;scales['w1']=scales['w1'].astype(np.float64)
        with self.assertRaises(t.TrainingError):t.ChannelQuantizedWeights(codes,scales)

    def test_actual_channel_receipt_binds_runtime_codes_and_verify_cli_vectors(self):
        report=self.result.report;digest=t.sha256_bytes(t.canonical_json_bytes(report))
        kwargs=dict(arm=self.arm,seed=t.FIXED_SEEDS[0],float_epoch=1,qat_epoch=self.result.qat_epoch,
            source_bundle_body_sha256='a'*64,qat_profile=t.CHANNEL_PREDICTION_QAT_PROFILE,qat_evidence_sha256=digest)
        path=t.write_runtime(self.root/'runtime',self.arch,self.result.quantized,**kwargs)
        arch,q,selection,document=t.load_runtime(path)
        receipt={'quantized_training':report,'float_training':self.warmup.report,'float_validation':self.warmup.metrics,
            'quantized_validation':self.result.metrics,'seed':t.FIXED_SEEDS[0],
            'offline_gate':t.offline_advancement_gate(self.warmup.metrics,self.result.metrics)}
        binding={'datasets':{name:t.dataset_identity(getattr(self.inputs,name)) for name in ('new','anchor')},
            'seed':t.FIXED_SEEDS[0],'source_bundle_body_sha256':'a'*64,
            'successor_ranking':{'initial_checkpoint':{'parameters':t._parameter_identity(self.initial,self.arch)}}}
        t._validate_channel_receipt_links(receipt,binding,self.root,self.warmup.parameters,q,selection,document)
        codes={name:value.copy() for name,value in q.integer.items()}
        codes['w1'][0,0]=2 if int(codes['w1'][0,0])==3 else codes['w1'][0,0]+1
        changed=t.ChannelQuantizedWeights(codes,q.scales)
        changed_document=t.runtime_document(self.arch,changed,**kwargs)
        _arch,changed_q,changed_selection=t.validate_runtime_document(changed_document)
        with self.assertRaisesRegex(t.TrainingError,'transplanted'):
            t._validate_channel_receipt_links(receipt,binding,self.root,self.warmup.parameters,changed_q,changed_selection,changed_document)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):self.assertEqual(t.main(['verify-runtime','--runtime',str(path)]),0)
        decoded=json.loads(output.getvalue());self.assertEqual(decoded['schema'],t.CHANNEL_RUNTIME_SCHEMA)
        self.assertEqual({name:len(value) for name,value in decoded['scales'].items()},t.CHANNEL_SCALE_COUNTS)

    def test_saved_calibration_objective_target_trajectory_and_optimizer_tampering_rejected_without_forwards(self):
        with mock.patch.object(t,'predict_dataset',side_effect=AssertionError('validation must not replay inference')):
            for mode in ('objective','target','epoch','step','scales','selection','codes'):
                bad=copy.deepcopy(self.result.report)
                if mode=='objective':bad['history'][0]['adaptive_scale_search']['trials'][0]['trials'][0]['objective']+=.001
                elif mode=='target':bad['history'][0]['adaptive_scale_search']['training_reference']={}
                elif mode=='epoch':bad['history'][0]['qat_epoch']=0
                elif mode=='step':bad['history'][0]['fake_quantization']['optimizer_steps_after_epoch']+=1
                elif mode=='scales':bad['history'][1]['fixed_scales']['w1'][0]*=2
                elif mode=='selection':bad['selected_validation']={}
                else:bad['original_initialization_code_evidence']['w3']['changed_codes']+=1
                with self.subTest(mode=mode),self.assertRaises(t.TrainingError):
                    t.validate_qat_execution_evidence(bad,expected_profile=t.CHANNEL_PREDICTION_QAT_PROFILE,float_validation_reference=self.warmup.metrics)


if __name__=='__main__':unittest.main()
