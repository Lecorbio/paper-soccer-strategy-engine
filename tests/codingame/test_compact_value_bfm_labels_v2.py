import copy
import hashlib
import pathlib
import tempfile
import unittest
from tools import compact_value_bfm_labels_v2 as labels
from tests.codingame import test_jacek_replay_corpus as fixtures
from tests.codingame.test_compact_value_bfm_training import bundle_fixture


class StreamLabelTests(unittest.TestCase):
    def fixture(self,root):
        row=fixtures.JacekReplayCorpusTests.complete_turn_action_group_row()
        row['source_bundle_body_sha256']='a'*64
        source=row['group']['source_binding'];source['campaign_id']=labels.campaign.ID
        row['group']['work_budget']['seed']=int(hashlib.sha256(
            f'{source["campaign_id"]}\0{source["position_id"]}\0{64000}'.encode()).hexdigest()[:16],16)
        position={key:source[key] for key in ('position_id','root_group_id','group_id','source','split','winner')}
        position.update(prefix='0',mover=1,parent_identity=row['group']['parent_identity'])
        census={'successor_identities':[s['successor_id'] for s in row['group']['successors']]}
        labels._PLAN={'teacher':{'sha256':row['teacher']['artifact_sha256']},
            'native_source_closures':{'action':{'sha256':row['teacher']['source_sha256']}}}
        labels._BUNDLE=bundle_fixture(root);labels._TEACHER=None
        return row,position,census

    def test_native_lineage_budget_and_full_census_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,position,census=self.fixture(pathlib.Path(tmp))
            labels.validate_row(row,position,census,'action',64000)
            missing=copy.deepcopy(census);missing['successor_identities'].append('f'*64)
            with self.assertRaisesRegex(ValueError,'successor census'):
                labels.validate_row(row,position,missing,'action',64000)
            wrong=copy.deepcopy(position);wrong['root_group_id']='another-root'
            with self.assertRaisesRegex(ValueError,'binding changed'):
                labels.validate_row(row,wrong,census,'action',64000)
            with self.assertRaisesRegex(ValueError,'binding changed'):
                labels.validate_row(row,position,census,'action',500000)

if __name__=='__main__': unittest.main()
