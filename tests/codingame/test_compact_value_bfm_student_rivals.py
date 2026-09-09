"""Prospective training-only pair policy; bounded fixtures, never campaign data."""
import copy
import dataclasses
import hashlib
import os
import unittest
from unittest import mock

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
import numpy as np
from tools import compact_value_bfm_train as t
from tests.codingame.test_compact_value_bfm_channel_qat import dataset

PROFILE = t.STUDENT_RIVALS_RETENTION_PROFILE
POLICY = t.STUDENT_RIVAL_PAIR_POLICY
ARCH = t.ARCHITECTURES['capacity-12x8']
ARM = t.ARMS['search-target']
SEED = t.FIXED_SEEDS[0]


def group(n=12, *, parent=0, exhaustive=True):
    return t.CompleteTurnGroup('group', parent, tuple(t.CompleteTurnSuccessor(
        f'id-{i:02}', np.array([i], dtype='<u2'), float(np.float32(1.-i/(n or 1))),
        parent, {}) for i in range(n)), successors_exhaustive=exhaustive)


def tiny_inputs():
    groups = tuple(dataclasses.replace(group(), group_id=f'group-{i:02}') for i in range(16))
    labels = t.SuccessorRankingLabels(groups, groups[:2], {}, 'a'*64, 'b'*64, 'c'*64)
    return t.TrainingInputs(new=dataset(128), anchor=dataset(384),
        common_adjudicator=dataset(8,'validation'), canonical_validation=dataset(9,'validation'),
        source_routes={}, successor_rankings=labels)


def parameters():
    value = t.initialize_parameters(ARCH, SEED)
    value['w1'][:] = np.float32(.04); value['w1'][1,:] = np.float32(.3)
    value['w2'][:] = np.float32(.06); value['w3'][:] = np.float32(.08)
    return value


def warmup(inputs, initial, weight=.1):
    return t.train_float_seed(inputs, ARCH, ARM, SEED, maximum_epochs=1, patience=1,
        learning_rate=.00006, initial_parameters=initial, ranking_weight=weight,
        training_pair_policy=POLICY)


class StudentRivalSelectorTests(unittest.TestCase):
    def test_fixed_randomized_independent_topscore_oracle_and_argmax_inclusion(self):
        rng = np.random.Generator(np.random.PCG64(19))
        for n in (2, 8, 9, 12, 21):
            for _ in range(20):
                g=group(n)
                ids=[str(int(v)) for v in rng.integers(0,5,n)]
                g=dataclasses.replace(g, successors=tuple(dataclasses.replace(s,successor_id=ids[i]) for i,s in enumerate(g.successors)))
                values=rng.integers(-3,4,n).astype(np.float32)
                teacher=t._teacher_parent_values(g)
                best=min(range(n),key=lambda i:(-float(teacher[i]),ids[i],i))
                eligible=[i for i in range(n) if float(teacher[best]-teacher[i])>0]
                selected=sorted(eligible,key=lambda i:(-float(values[i]),ids[i],i))[:8]
                expected=tuple(sorted(selected,key=lambda i:(-float(teacher[best]-teacher[i]),ids[i],i)))
                actual, rivals, gaps=t._student_ranking_pairs(g,values)
                self.assertEqual(actual,best);self.assertEqual(rivals,expected)
                np.testing.assert_array_equal(gaps,np.array([teacher[best]-teacher[i] for i in expected],dtype=np.float32))
                chosen=min(range(n),key=lambda i:(-float(values[i]),ids[i],i))
                if teacher[best]>teacher[chosen]:self.assertIn(chosen,rivals)

    def test_current_scores_reselect_and_accumulate_in_teacher_order(self):
        g=group(); values=np.zeros(12,dtype=np.float32);values[1]=.4
        _,static,_=t._ranking_pairs(g)
        _,first,_=t._student_ranking_pairs(g,values)
        self.assertNotIn(1,static);self.assertIn(1,first)
        values[1]=-.8;values[2]=.4
        _,second,_=t._student_ranking_pairs(g,values)
        self.assertNotEqual(first,second);self.assertIn(2,second)
        self.assertEqual(second,tuple(sorted(second,reverse=True)))

    def test_parent_signs_and_exact_ties_preserve_small_group_loss_bytes(self):
        for parent in (0,1):
            for signs in (False,True):
                g=group(8,parent=parent)
                g=dataclasses.replace(g,successors=tuple(dataclasses.replace(s,value_mover=(1-parent if signs and i%2 else parent),successor_id='duplicate' if i in (2,3) else s.successor_id) for i,s in enumerate(g.successors)))
                raw=np.array([0.,-0.,.25,.25,-.5,.5,np.nextafter(np.float32(0),np.float32(1)),.1],dtype=np.float32)
                legacy=t.pairwise_successor_ranking_loss_gradient(g,raw)
                new=t.student_rival_loss_gradient(g,raw)
                self.assertEqual(legacy[0],new[0]);self.assertEqual(legacy[1].tobytes(),new[1].tobytes())
                self.assertEqual(legacy[2],{k:v for k,v in new[2].items() if k not in ('training_pair_policy','teacher_best_successor_index','selected_successor_indices')})

    def test_shape_finite_and_cap_fail_closed(self):
        g=group()
        for bad in (np.zeros(11),np.zeros((12,1)),np.full(12,np.nan),np.full(12,np.inf)):
            with self.assertRaises(t.TrainingError):t._student_ranking_pairs(g,bad)
        for cap in (True,7,9):
            with self.assertRaises(t.TrainingError):t._student_ranking_pairs(g,np.zeros(12),pair_cap=cap)
        with self.assertRaises(ValueError):t._student_ranking_pairs(group(0),np.array([],dtype=np.float32))
        with self.assertRaises(t.TrainingError):t.ranking_microbatch_loss_gradient([g],np.zeros(12),training_pair_policy='unknown')

    def test_nonexhaustive_and_zero_gap_skip_like_legacy(self):
        for g in (group(exhaustive=False), dataclasses.replace(group(),successors=tuple(dataclasses.replace(s,teacher_value=.5) for s in group().successors))):
            a=t.pairwise_successor_ranking_loss_gradient(g,np.ones(12,dtype=np.float32))
            b=t.student_rival_loss_gradient(g,np.ones(12,dtype=np.float32))
            self.assertEqual(a[0],0.);self.assertEqual(a[0],b[0]);self.assertEqual(a[1].tobytes(),b[1].tobytes())
            self.assertEqual(b[2]['selected_successor_indices'],[])

    def test_stream_digest_binds_current_predictions_even_same_choices(self):
        g=group();initial=parameters()
        def evidence(v):
            collector=t._StudentRivalEpochEvidence(phase='float-warmup',seed=SEED,schedule_epoch=1,ranking_weight=.1,scalar_batches=1)
            collector.begin_batch(0,initial,ARCH)
            t.ranking_microbatch_loss_gradient([g],v,training_pair_policy=POLICY,pair_evidence=collector)
            result=collector.finish()
            self.assertFalse(any(isinstance(value,np.ndarray) for value in vars(collector).values()))
            return result
        first=np.linspace(-.1,.2,12,dtype=np.float32)
        a=evidence(first);b=evidence(first.copy());self.assertEqual(a,b)
        second=first.copy();second[0]=np.nextafter(second[0],np.float32(-1))
        c=evidence(second)
        self.assertEqual(a['selected_pairs'],c['selected_pairs'])
        self.assertNotEqual(a['ordered_pair_prediction_sha256'],c['ordered_pair_prediction_sha256'])
        self.assertEqual(a['batch_master_sha256'],c['batch_master_sha256'])


class StudentRivalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=tiny_inputs();cls.initial=parameters()
        with t.native_thread_execution_scope():
            cls.float=warmup(cls.inputs,cls.initial)
            cls.qat=t.run_fixed_scale_qat(cls.float,cls.inputs,ARCH,ARM,SEED,ranking_weight=.1,qat_profile=PROFILE)

    def test_recipe_only_changes_training_pair_policy_and_keeps_old_hashes(self):
        old={'standard-v1':'9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
             'refined-adaptive-scales-v1':'7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd',
             'retention-first-low-rate-v1':'3cdbd16892860b6db547ca510e97604ba7742ae07392df64fd817b315b975de3',
             'channel-prediction-qat-v1':'4ae78b345fb3f586d0b590a04127993aa9880c6e1cec265191b7db7398cafaa0'}
        for name,digest in old.items():
            self.assertEqual(t.qat_profile_contract(name)['body_sha256'],digest)
            self.assertIsNone(t.training_pair_policy_contract(name))
        contract=t.qat_profile_contract(PROFILE);base=t.qat_profile_contract(t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE)
        for key in ('quantization','scale_selection'):self.assertEqual(contract[key],base[key])
        self.assertEqual({k:v for k,v in contract['schedule'].items() if k!='float_warmup_learning_rate'},base['schedule'])
        self.assertEqual(contract['schedule']['float_warmup_learning_rate'],.00006)
        self.assertEqual(contract['training_pair_policy']['applies_to'],['float-warmup','qat-1','qat-2','qat-3','qat-4'])
        self.assertEqual(t.validate_qat_profile_contract(contract),contract)
        with self.assertRaises(t.TrainingError):t.resolve_qat_profile(dataclasses.replace(t.resolve_qat_profile(PROFILE),qat_learning_rate=.00025))

    def test_real_one_plus_four_updates_and_nonzero_current_policy_use(self):
        self.assertEqual(self.float.report['optimizer']['learning_rate'],.00006)
        self.assertEqual(self.qat.report['learning_rate'],.0000625)
        self.assertEqual(self.qat.report['optimizer_steps'],8)
        summary=t.validate_student_rival_execution(self.float.report,self.qat.report,seed=SEED)
        self.assertEqual(summary['validated_epoch_reports'],5)
        self.assertEqual(summary['group_evaluations'],80)
        self.assertGreater(summary['changed_static_subset_groups'],0)
        self.assertGreater(summary['regrettable_argmax_groups'],0)
        self.assertEqual(summary['missed_regrettable_argmax_groups'],0)
        self.assertFalse(summary['independent_numerical_replay_claimed'])
        self.assertIsInstance(self.qat.quantized,t.QuantizedWeights)
        for item in self.qat.report['history']:
            for update in item['fake_quantization']['master_parameter_updates'].values():
                self.assertGreater(update['changed_parameters'],0);self.assertTrue(np.isfinite(update['l2_delta']))
        for name in ('w1','w2','w3'):
            self.assertTrue(np.isfinite(self.float.parameters[name]).all())
            self.assertEqual(self.initial[name].tobytes(),parameters()[name].tobytes())
        t.validate_qat_execution_evidence(self.qat.report,expected_profile=PROFILE,float_validation_reference=self.float.metrics)
        t.validate_successor_schedule_execution(self.float.report,self.qat.report,seed=SEED)

    def test_all_five_total_training_objectives_explicitly_distinguish_static_validation(self):
        for item in (*self.float.report['history'],*self.qat.report['history']):
            self.assertNotIn('training_objective_weighted_huber',item)
            self.assertTrue(np.isfinite(item['training_total_objective']))
            self.assertEqual(item['training_total_objective_definition'],t.STUDENT_TRAINING_OBJECTIVE)
            self.assertIn('pairwise_loss',item['validation']['successor_ranking'])
        self.assertEqual(t.training_pair_policy_contract(PROFILE)['validation_pair_policy'],'unchanged-static-teacher-worst-eight')
        report=copy.deepcopy(self.qat.report);report['history'][0]['training_total_objective_definition']='static-validation-pairs'
        with self.assertRaisesRegex(t.TrainingError,'objective label'):
            t.validate_student_rival_execution(self.float.report,report,seed=SEED)

    def test_frozen_retention_reference_across_initial_adaptive_and_epoch_choice(self):
        reference=self.qat.report['retention_reference']
        self.assertEqual(reference['float_validation'],self.float.metrics)
        for stage in (self.qat.report['scale_search'],*[x['adaptive_scale_search'] for x in self.qat.report['history']]):
            self.assertEqual(stage['retention_reference'],reference)
        new=t.resolve_qat_profile(PROFILE);base=t.resolve_qat_profile(t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE)
        for epoch in self.qat.report['history']:
            self.assertEqual(t._qat_validation_key(epoch['validation'],new,float_validation_reference=self.float.metrics),
                             t._qat_validation_key(epoch['validation'],base,float_validation_reference=self.float.metrics))

    def test_resealed_count_epoch_policy_digest_and_warmup_transplants_rejected(self):
        for key,value in [('group_evaluations',0),('selected_pairs',0),('missed_regrettable_argmax_groups',1),
                          ('phase','float-warmup'),('schedule_epoch',1),('seed',SEED+1),
                          ('loss_weight',.25),('active',False),('raw_prediction_trace_retained',True),
                          ('ordered_pair_prediction_sha256',hashlib.sha256().hexdigest())]:
            report=copy.deepcopy(self.qat.report);epoch=report['history'][0]['training_pair_selection']
            epoch[key]=value;epoch.pop('body_sha256');report['history'][0]['training_pair_selection']=t.body_hashed(epoch)
            with self.subTest(key=key),self.assertRaises(t.TrainingError):
                t.validate_qat_execution_evidence(report,expected_profile=PROFILE,float_validation_reference=self.float.metrics)
        report=copy.deepcopy(self.qat.report);report['float_warmup_report_sha256']='a'*64
        with self.assertRaisesRegex(t.TrainingError,'transplanted'):
            t.validate_student_rival_execution(self.float.report,report,seed=SEED)
        old=copy.deepcopy(self.float.report);old.pop('training_pair_policy')
        with mock.patch.object(t,'select_fixed_scales',side_effect=AssertionError('reject before scale work')):
            with self.assertRaisesRegex(t.TrainingError,'new-policy warmup'):
                t.run_fixed_scale_qat(dataclasses.replace(self.float,report=old),self.inputs,ARCH,ARM,SEED,ranking_weight=.1,qat_profile=PROFILE)

    def test_pre_qat_rejects_changed_warmup_arrays_metrics_or_legacy_recipe_before_work(self):
        changed={name:value.copy() for name,value in self.float.parameters.items()}
        changed['w3'][0]=np.nextafter(changed['w3'][0],np.float32(10.))
        metrics=copy.deepcopy(self.float.metrics);metrics['common_adjudicator']['weighted_huber']+=.01
        with mock.patch.object(t,'select_fixed_scales',side_effect=AssertionError('no scale work before preflight')):
            for candidate in (dataclasses.replace(self.float,parameters=changed),dataclasses.replace(self.float,metrics=metrics)):
                with self.assertRaisesRegex(t.TrainingError,'warmup parameters/validation'):
                    t.run_fixed_scale_qat(candidate,self.inputs,ARCH,ARM,SEED,ranking_weight=.1,qat_profile=PROFILE)
            with self.assertRaisesRegex(t.TrainingError,'legacy QAT'):
                t.run_fixed_scale_qat(self.float,self.inputs,ARCH,ARM,SEED,ranking_weight=.1,qat_profile=t.RETENTION_FIRST_LOW_RATE_QAT_PROFILE)

    def test_new_warmup_rejects_other_optimizer_batching_and_seed_types(self):
        for key,value in [('name','sgd'),('weight_decay',.01),('patience',True),('gradient_norm_clip',1.)]:
            report=copy.deepcopy(self.float.report);report['optimizer'][key]=value
            with self.subTest(key=key),self.assertRaises(t.TrainingError):
                t._validate_student_float_evidence(report,seed=SEED,ranking_weight=.1)
        for key,value in [('seed',float(SEED)),('best_float_epoch',True)]:
            report=copy.deepcopy(self.float.report);report[key]=value
            with self.subTest(key=key),self.assertRaises(t.TrainingError):
                t._validate_student_float_evidence(report,seed=SEED,ranking_weight=.1)
        report=copy.deepcopy(self.float.report);report['batching']['new_loss_share']=.5
        with self.assertRaises(t.TrainingError):t._validate_student_float_evidence(report,seed=SEED,ranking_weight=.1)
        with mock.patch.object(t,'AdamW',side_effect=AssertionError('reject before optimizer')):
            with self.assertRaisesRegex(t.TrainingError,'exact successor warmup'):
                t.train_float_seed(self.inputs,ARCH,ARM,SEED,maximum_epochs=1,patience=1,learning_rate=.00006,
                    weight_decay=.01,initial_parameters=self.initial,ranking_weight=.1,training_pair_policy=POLICY)

    def test_inactive_lambda_zero_never_selects_rivals_and_reports_zero_use(self):
        with t.native_thread_execution_scope(),mock.patch.object(t,'student_rival_loss_gradient',side_effect=AssertionError('inactive selector ran')):
            zero=warmup(self.inputs,self.initial,0.)
        ev=zero.report['history'][0]['training_pair_selection']
        self.assertFalse(ev['active']);self.assertEqual(ev['group_evaluations'],0)
        self.assertEqual(ev['selected_pairs'],0)
        self.assertEqual(ev['ordered_pair_prediction_sha256'],hashlib.sha256().hexdigest())
        t._validate_student_float_evidence(zero.report,seed=SEED,ranking_weight=0.)


if __name__=='__main__':unittest.main()
