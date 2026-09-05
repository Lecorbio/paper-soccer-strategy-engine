import pathlib
import random
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tools import compact_value_bfm_stream_v2 as stream


class StreamPositionTests(unittest.TestCase):
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
