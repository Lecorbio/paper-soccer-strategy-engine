import pathlib
import random
import tempfile
import types
import unittest
from unittest import mock
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tools import compact_value_bfm_stream_v2 as stream


class StreamPositionTests(unittest.TestCase):
    def empty_phase(self,root):
        context=root/'context';phase='fixture'
        stream.campaign.seal(context/'campaign.json',{'policy':stream.campaign.POLICY,'exclusions':[]})
        stream.campaign.seal(context/phase/'games.json',{'rows':[]})
        return context,phase

    def index_module(self,root):
        module=types.SimpleNamespace(__file__=str(root/'index-loader.py'))
        pathlib.Path(module.__file__).write_text('# synthetic loader\n')
        def build(contract_path,index_path):
            stream.campaign.seal(index_path,{'contract':stream.campaign.record(contract_path),'synthetic':True})
            return stream.campaign.record(index_path)
        module.build_index=mock.Mock(side_effect=build)
        module.load_index=mock.Mock(return_value={})
        return module

    def empty_pool(self):
        pool=mock.MagicMock()
        pool.return_value.__enter__.return_value.map.return_value=[]
        return pool

    def test_legacy_plan_shape_and_lookup_stay_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            context,phase=self.empty_phase(pathlib.Path(temporary))
            with (mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool()),
                  mock.patch.object(stream,'_exclusion_index_module',side_effect=AssertionError('legacy must not load packed helper'))):
                stream.run_positions(context,phase,2)
            plan=stream.campaign.read(context/phase/'positions-plan.json')
            self.assertEqual(set(plan),{'schema','context','games','phase','workers','producer','sample_limit',
                'opening_max_edges','validation_first','closure_limit','body_sha256'})

    def test_packed_index_is_built_once_before_both_worker_pools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary);context,phase=self.empty_phase(root)
            module=self.index_module(root);pool=self.empty_pool()
            def assert_index_ready(*_args,**kwargs):
                self.assertEqual(module.build_index.call_count,1)
                plan=stream.campaign.read(context/phase/'positions-plan.json')
                self.assertEqual(plan['packed_base_exclusions'],stream.campaign.record(stream._packed_index_path(context,phase)))
                self.assertEqual(plan['packed_base_exclusions_loader'],stream.campaign.record(pathlib.Path(module.__file__)))
                self.assertEqual(kwargs['max_workers'],8)
                return pool.return_value
            pool.side_effect=assert_index_ready
            with (mock.patch.object(stream,'_exclusion_index_module',return_value=module),
                  mock.patch.object(stream,'ProcessPoolExecutor',pool)):
                result=stream.run_positions(context,phase,packed_base_exclusions=True)
                self.assertEqual(pool.call_count,2)
                self.assertTrue(stream.campaign.read(stream.campaign.verify(result['validation_barrier']))['all_validation_successors_included'])
                # Completed output cannot silently switch back to legacy mode.
                with self.assertRaisesRegex(ValueError,'frozen position plan base-exclusion mode'):
                    stream.run_positions(context,phase)
                self.assertEqual(module.build_index.call_count,1)
                self.assertEqual(stream.run_positions(context,phase,packed_base_exclusions=True),result)
                self.assertEqual(module.build_index.call_count,1)

    def test_frozen_legacy_plan_rejects_packed_retrofit_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            context,phase=self.empty_phase(pathlib.Path(temporary))
            with mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool()):
                stream.run_positions(context,phase)
            for completed in (True,False):
                if not completed:(context/phase/'positions.json').unlink()
                with (mock.patch.object(stream,'_exclusion_index_module') as builder,
                      mock.patch.object(stream.campaign,'once') as write,
                      mock.patch.object(stream,'ProcessPoolExecutor') as pool):
                    with self.assertRaisesRegex(ValueError,'frozen position plan base-exclusion mode'):
                        stream.run_positions(context,phase,packed_base_exclusions=True)
                    builder.assert_not_called();write.assert_not_called();pool.assert_not_called()

    def test_completed_output_does_not_bypass_frozen_producer_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            context,phase=self.empty_phase(pathlib.Path(temporary))
            with mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool()):
                stream.run_positions(context,phase)
            with (mock.patch.object(stream,'_EXECUTION_BYTES',b'new producer'),
                  mock.patch.object(stream.campaign,'once') as write,
                  mock.patch.object(stream,'ProcessPoolExecutor') as pool):
                with self.assertRaisesRegex(ValueError,'producer changed'):
                    stream.run_positions(context,phase)
                write.assert_not_called();pool.assert_not_called()

    def test_completed_output_cannot_substitute_a_different_plan_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            context,phase=self.empty_phase(pathlib.Path(temporary))
            with mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool()):
                result=stream.run_positions(context,phase)
            output=context/phase/'positions.json';output.unlink()
            body={key:value for key,value in result.items() if key!='body_sha256'}
            body['plan']={**body['plan'],'sha256':'0'*64}
            stream.campaign.seal(output,body)
            with mock.patch.object(stream.campaign,'once') as write:
                with self.assertRaisesRegex(ValueError,'differ from the frozen position plan'):
                    stream.run_positions(context,phase)
                write.assert_not_called()

    def test_packed_loader_drift_rejects_before_completed_output_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary);context,phase=self.empty_phase(root);module=self.index_module(root)
            with (mock.patch.object(stream,'_exclusion_index_module',return_value=module),
                  mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool())):
                stream.run_positions(context,phase,packed_base_exclusions=True)
                pathlib.Path(module.__file__).write_text('# changed loader\n')
                with mock.patch.object(stream.campaign,'once') as write:
                    with self.assertRaisesRegex(ValueError,'loader differs from frozen source'):
                        stream.run_positions(context,phase,packed_base_exclusions=True)
                    write.assert_not_called()
                self.assertEqual(module.build_index.call_count,1)

    def test_completed_packed_output_revalidates_source_and_array_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary);context,phase=self.empty_phase(root);module=self.index_module(root)
            with (mock.patch.object(stream,'_exclusion_index_module',return_value=module),
                  mock.patch.object(stream,'ProcessPoolExecutor',self.empty_pool())):
                stream.run_positions(context,phase,packed_base_exclusions=True)
                module.load_index.side_effect=ValueError('packed exclusion array changed')
                with mock.patch.object(stream.campaign,'once') as write:
                    with self.assertRaisesRegex(ValueError,'packed exclusion array changed'):
                        stream.run_positions(context,phase,packed_base_exclusions=True)
                    write.assert_not_called()
                self.assertEqual(module.build_index.call_count,1)
                module.load_index.assert_called_once_with(
                    stream.campaign.record(stream._packed_index_path(context,phase)),
                    contract_record=stream.campaign.record(context/'campaign.json'))

    def test_mapped_lookup_preserves_role_order_early_exception_and_whole_group_rejection(self):
        campaign=stream.campaign
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary).resolve();phase='fixture';module=self.index_module(root)
            empty=campaign.features.ReplayState();early,_prefix=campaign.fresh_root(6,random.Random(51))
            _action,child=next(campaign.iter_successors(early))
            state_domain=campaign.legacy.STATE_FINGERPRINT_DOMAIN;feature_domain=campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            rows=[('prior-train',state_domain,[campaign.fingerprints(empty)[state_domain]]),
                  ('live',state_domain,[campaign.fingerprints(empty)[state_domain],campaign.fingerprints(early)[state_domain]]),
                  ('protected',feature_domain,[campaign.fingerprints(child)[feature_domain]])]
            exclusions=[];mapped={}
            for ordinal,(role,domain,values) in enumerate(rows):
                path=root/f'exclusion-{ordinal}.json'
                campaign.seal(path,{'role':role,'domain':domain,'fingerprints':values})
                exclusions.append(campaign.record(path))
                array=root/f'exclusion-{ordinal}.bin'
                array.write_bytes(b''.join(sorted({bytes.fromhex(value) for value in values})))
                mapped[role,domain]=stream.FingerprintIndex(array)
            contract_path=root/'campaign.json';campaign.seal(contract_path,{'policy':campaign.POLICY,'exclusions':exclusions})
            context=campaign.record(contract_path)
            index=module.build_index(contract_path,stream._packed_index_path(root,phase))
            legacy=root/'legacy-plan.json';packed=root/'packed-plan.json'
            campaign.seal(legacy,{'context':context,'phase':phase})
            campaign.seal(packed,{'context':context,'phase':phase,'packed_base_exclusions':index,
                'packed_base_exclusions_loader':campaign.record(pathlib.Path(module.__file__))})
            module.load_index.return_value=mapped
            stream.initialize_worker(str(legacy),None)
            original=stream._EXCLUDED
            with mock.patch.object(stream,'_exclusion_index_module',return_value=module):
                stream.initialize_worker(str(packed),None)
                self.assertEqual(list(stream._EXCLUDED),list(original))
                self.assertTrue(all(isinstance(value,stream.FingerprintIndex) for value in stream._EXCLUDED.values()))
                module.load_index.assert_called_once_with(index,contract_record=context)
                for state in (empty,early,child):
                    for split in ('train','validation'):
                        self.assertEqual(campaign.rejection(state,split,stream._EXCLUDED),campaign.rejection(state,split,original))
                self.assertIsNone(campaign.rejection(early,'train',stream._EXCLUDED))
                self.assertEqual(campaign.rejection(empty,'validation',stream._EXCLUDED),'prior-train')
                self.assertEqual(campaign.preflight_group(early,'train',stream._EXCLUDED),campaign.preflight_group(early,'train',original))
                self.assertEqual(campaign.preflight_group(early,'train',stream._EXCLUDED)['reason'],'successor:protected')
                barrier_array=root/'barrier.bin';barrier_array.write_bytes(bytes.fromhex(campaign.fingerprints(empty)[feature_domain]))
                barrier=root/'barrier.json';campaign.seal(barrier,{'arrays':{feature_domain:campaign.record(barrier_array)}})
                stream.initialize_worker(str(packed),str(barrier))
                self.assertEqual(list(stream._EXCLUDED),[*original,('phase-validation',feature_domain)])
                self.assertEqual(campaign.rejection(empty,'train',stream._EXCLUDED),'phase-validation')

    def test_packed_and_legacy_mining_emit_identical_position_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary).resolve();phase='fixture';module=self.index_module(root)
            contract=root/'campaign.json';stream.campaign.seal(contract,{'policy':stream.campaign.POLICY,'exclusions':[]})
            context=stream.campaign.record(contract);index=module.build_index(contract,stream._packed_index_path(root,phase))
            plans=[]
            for packed in (False,True):
                path=root/f'plan-{packed}.json';body={'context':context,'phase':phase}
                if packed:body.update({'packed_base_exclusions':index,'packed_base_exclusions_loader':stream.campaign.record(pathlib.Path(module.__file__))})
                stream.campaign.seal(path,body);plans.append(path)
            game=root/'game.json';stream.campaign.seal(game,{'fixture':True})
            item={'ordinal':0,'root_id':'root:fixture','split':'train','receipt':stream.campaign.record(game),
                'game':{'transcript':'0/0/3/0/61/0/07','prefix_turns':0,'winner':0,'game_id':'game:fixture'}}
            receipts=[]
            with mock.patch.object(stream,'_exclusion_index_module',return_value=module):
                for ordinal,plan in enumerate(plans):
                    stream.initialize_worker(str(plan),None)
                    record=stream.mine_game((item,str(root/f'chunk-{ordinal}'),None))
                    receipts.append(stream.campaign.read(stream.campaign.verify(record)))
            self.assertEqual(receipts[0]['output']['sha256'],receipts[1]['output']['sha256'])
            self.assertEqual(receipts[0]['positions'],receipts[1]['positions'])
            self.assertEqual(receipts[0]['rejections'],receipts[1]['rejections'])

    def test_real_packed_loader_initializes_worker_without_legacy_set_materialization(self):
        campaign=stream.campaign
        with tempfile.TemporaryDirectory() as temporary:
            root=pathlib.Path(temporary).resolve();phase='fixture'
            domain=campaign.legacy.STATE_FINGERPRINT_DOMAIN
            state=campaign.features.ReplayState();fingerprint=campaign.fingerprints(state)[domain]
            exclusion=root/'exclusion.json';campaign.seal(exclusion,{
                'role':'prior-train','domain':domain,'fingerprints':[fingerprint]})
            contract=root/'campaign.json';campaign.seal(contract,{
                'policy':campaign.POLICY,'exclusions':[campaign.record(exclusion)]})
            module=stream._exclusion_index_module()
            index=module.build_index(contract,stream._packed_index_path(root,phase))
            plan=root/phase/'positions-plan.json'
            campaign.seal(plan,{'context':campaign.record(contract),'phase':phase,
                'packed_base_exclusions':index,'packed_base_exclusions_loader':campaign.record(pathlib.Path(module.__file__))})
            with mock.patch.object(campaign,'exclusion_sets',side_effect=AssertionError('packed workers must not allocate base Python sets')):
                stream.initialize_worker(str(plan),None)
            self.assertEqual(list(stream._EXCLUDED),[('prior-train',domain)])
            values=stream._EXCLUDED['prior-train',domain]
            self.assertNotIsInstance(values,set)
            self.assertEqual(len(values),1)
            self.assertIn(fingerprint,values)
            self.assertIsNone(campaign.rejection(state,'train',stream._EXCLUDED))
            self.assertEqual(campaign.rejection(state,'validation',stream._EXCLUDED),'prior-train')

    def test_binary_fingerprints_include_zero_bytes_exactly(self):
        rng=random.Random(2026)
        values={rng.randbytes(32) for _ in range(500)}|{bytes(32),b'\xff'*31+b'\0',b'a\0'+b'b'*30}
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'index.bin';path.write_bytes(np.asarray(sorted(values),dtype='S32').tobytes())
            index=stream.FingerprintIndex(path)
            for value in values: self.assertIn(value.hex(),index)
            for _ in range(500):
                value=rng.randbytes(32)
                self.assertEqual(value.hex() in index,value in values)

    def test_compressed_receipts_are_deterministic_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=pathlib.Path(tmp)/'first.gz';second=pathlib.Path(tmp)/'second.gz'
            rows=[{'position_id':'a','values':[0,1,2]},{'position_id':'b','values':[3]}]
            stream.write_gzip(first,rows);stream.write_gzip(second,rows)
            self.assertEqual(first.read_bytes(),second.read_bytes())
            self.assertEqual(list(stream.read_gzip(first)),rows)
            stream.write_gzip(first,rows)
            with self.assertRaisesRegex(ValueError,'immutable compressed artifact'):
                stream.write_gzip(first,[{'changed':True}])

    def test_process_worker_receipt_resumes_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);context=root/'context.json';plan=root/'plan.json';game_receipt=root/'fixture.json'
            stream.campaign.seal(context,{'policy':stream.campaign.POLICY,'exclusions':[]})
            stream.campaign.seal(plan,{'context':stream.campaign.record(context),'phase':'unit-fixture'})
            stream.campaign.seal(game_receipt,{'fixture':True})
            item={'ordinal':0,'root_id':'root:fixture','split':'train',
                'receipt':stream.campaign.record(game_receipt),
                'game':{'transcript':'0/0/3/0/61/0/07','prefix_turns':0,'winner':0,'game_id':'game:fixture'}}
            job=(item,str(root/'chunk'),None)
            with ProcessPoolExecutor(max_workers=2,initializer=stream.initialize_worker,
                    initargs=(str(plan),None)) as pool:
                first=pool.submit(stream.mine_game,job).result(timeout=30)
                second=pool.submit(stream.mine_game,job).result(timeout=30)
            self.assertEqual(first,second)
            self.assertGreater(stream.campaign.read(first['path'])['positions'],0)

    def test_short_completed_game_keeps_available_absolute_early_states(self):
        stream._EXCLUDED={}
        item={'split':'train','game':{'transcript':'0/0/3/0/61/0/07','prefix_turns':0,'winner':0}}
        positions=stream.candidate_positions(item)
        self.assertEqual(len(positions),7)
        self.assertEqual({p['edges'] for p in positions},{0,1,2,3,4,6,7})
        self.assertTrue(all(p['edges']<=12 for p in positions))

if __name__=='__main__': unittest.main()
