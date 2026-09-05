import pathlib
import tempfile
import unittest
import numpy as np
from tools import compact_value_bfm_ranking_store as store
from tools import compact_value_bfm_train as trainer
from tools import jacek_replay_corpus as corpus
from tests.codingame.test_compact_value_bfm_training import bundle_fixture
from tests.codingame import test_jacek_replay_corpus as corpus_tests


class RankingStoreTests(unittest.TestCase):
    def fixture(self,root):
        fixture=corpus_tests.JacekReplayCorpusTests()
        rows=[fixture.complete_turn_action_group_row(root_action='0'),
              fixture.complete_turn_action_group_row(position_id='position:'+'b'*64,split='validation',root_action='2')]
        for row in rows: row['source_bundle_body_sha256']='a'*64
        source=root/'labels.jsonl';source.write_bytes(b''.join(corpus.canonical_json_bytes(row) for row in rows))
        bundle=bundle_fixture(root)
        index=store.build_store([source],root/'store',bundle)
        return rows,bundle,index

    def test_lossless_lazy_groups_and_scalar_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,bundle,index=self.fixture(pathlib.Path(tmp))
            loaded=store.RankingStore(index,bundle).labels()
            document=corpus.build_complete_turn_successor_labels(rows)
            eager=trainer.validate_successor_label_document(document,source_bundle_body_sha256='a'*64,artifact_sha256='b'*64)
            for lazy_groups,eager_groups in ((loaded.train,eager.train),(loaded.validation,eager.validation)):
                self.assertEqual(len(lazy_groups),len(eager_groups))
                for left,right in zip(lazy_groups,eager_groups):
                    self.assertEqual(left.evidence,right.evidence)
                    self.assertEqual(len(left.successors),len(right.successors))
                    for l,r in zip(left.successors,right.successors):
                        self.assertTrue(np.array_equal(l.active,r.active))
                        self.assertEqual(l.successor_id,r.successor_id)
                        self.assertEqual(l.teacher_value,r.teacher_value)
                        self.assertEqual(l.value_mover,r.value_mover)
                        self.assertEqual(l.evidence,r.evidence)
                    predictions=np.linspace(-.3,.7,len(left.successors),dtype=np.float32)
                    a=trainer.pairwise_successor_ranking_loss_gradient(left,predictions)
                    b=trainer.pairwise_successor_ranking_loss_gradient(right,predictions)
                    self.assertEqual(a[0],b[0]);self.assertTrue(np.array_equal(a[1],b[1]));self.assertEqual(a[2],b[2])
            for row,group in zip(rows,(loaded.train[0],loaded.validation[0])):
                self.assertEqual(store.scalar_samples(group),corpus.sample_from_teacher_row(row))

    def test_gzip_source_preserves_all_successors(self):
        from tools.compact_value_bfm_stream_v2 import write_gzip
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);rows,bundle,index=self.fixture(root)
            compressed=root/'labels.jsonl.gz';write_gzip(compressed,rows)
            other=store.build_store([compressed],root/'compressed-store',bundle)
            a=store.RankingStore(index,bundle).labels();b=store.RankingStore(other,bundle).labels()
            for left,right in zip((*a.train,*a.validation),(*b.train,*b.validation)):
                self.assertEqual(len(left.successors),len(right.successors))
                for l,r in zip(left.successors,right.successors):
                    self.assertEqual(l.evidence,r.evidence)
                    self.assertTrue(np.array_equal(l.active,r.active))
                    self.assertEqual(l.teacher_value,r.teacher_value)

    def test_tampered_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,bundle,index=self.fixture(pathlib.Path(tmp))
            document=store.campaign.read(index);path=pathlib.Path(document['arrays']['indices']['path'])
            data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'changed artifact'):
                store.RankingStore(index,bundle)

if __name__=='__main__': unittest.main()
