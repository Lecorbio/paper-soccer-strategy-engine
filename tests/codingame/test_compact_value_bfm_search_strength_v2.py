import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_search_strength_v2 as strength

campaign=strength.campaign


class SearchStrengthTests(unittest.TestCase):
    def test_full_search_command_binds_nondefault_seed_and_threshold(self):
        source={'path':'/source.cpp','sha256':'1'*64}
        plan={'variants':{'baseline':{'source':source}},'rank4_source':{'path':'/rank4.cpp'}}
        command=strength.command(plan,'baseline',Path('/gate'),{'tsv':{'path':'/bank.tsv','sha256':'2'*64}},Path('/raw.json'))
        self.assertEqual(command[command.index('--candidate-seed')+1],'1')
        self.assertEqual(command[command.index('--pair-count')+1],'500')
        self.assertEqual(command[command.index('--minimum-candidate-wins')+1],'550')
        self.assertEqual(command[-1],'--include-trajectories')

    def test_stray_trajectory_output_is_spent_without_a_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();context=root/'phase';phase='phase'
            output=strength.search.directory(context,phase)/'strength/baseline'
            output.mkdir(parents=True);(output/'result.json.trajectories.jsonl').write_text('{}\n')
            with patch.object(strength.search,'validate_plan',return_value={}):
                with self.assertRaisesRegex(ValueError,'spent'):
                    strength.run(root,context,phase)

    def fixture(self,root):
        phase='attempt-001-full';context=root/'phases'/phase;output=context/phase/'search/strength'
        campaign.seal(context/'campaign.json',{'exclusions':[]})
        plan={'context':campaign.record(context/'campaign.json')}
        campaign.seal(output.parent/'plan.json',plan)
        campaign.seal(output.parent/'measurement.json',{'all_measurements':True})
        rows=[]
        for seed in range(4):
            state,transcript=campaign.fresh_root(12,random.Random(seed+600))
            fps=campaign.fingerprints(state)
            rows.append({'opening_id':fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
                'transcript':transcript,'plies':12,'fingerprints':fps})
        census=context/phase/'census.gz';census.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(census,'wt') as stream:stream.write(json.dumps({'closure':[rows[0]['fingerprints']]})+'\n')
        campaign.seal(context/phase/'positions.json',{'census_files':[campaign.record(census)]})
        campaign.seal(context/phase/'games.json',{'rows':[]})
        inputs=strength.phase_inputs(plan,output)
        campaign.seal(output/'seed-claim.json',{'inputs':inputs,'policy':strength.POLICY,'producers':{}})
        claim=campaign.record(output/'seed-claim.json')
        campaign.seal(output/'proposals.json',{'seed_claim':claim,'rows':rows})
        retained=strength.isolated_rows(plan,output,rows)
        tsv=output/'bank.tsv'
        tsv.write_text('opening_id\ttranscript\n'+''.join(row['opening_id']+'\t'+row['transcript']+'\n' for row in retained))
        campaign.seal(output/'bank.json',{'seed_claim':claim,'proposals':campaign.record(output/'proposals.json'),
            'rows':retained,'pairs':2,'tsv':campaign.record(tsv)})
        return plan,output,rows

    def test_bank_filters_complete_current_closure_and_rechecks_resealed_roots(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(strength.POLICY,{'pairs':2,'proposals':4}):
            plan,output,rows=self.fixture(Path(tmp).resolve())
            bank=strength.validate_bank(plan,output)
            self.assertEqual(bank['rows'],rows[1:3])
            changed={key:value for key,value in bank.items() if key!='body_sha256'}
            changed['rows']=rows[:2]
            (output/'bank.tsv').write_text('opening_id\ttranscript\n'+''.join(row['opening_id']+'\t'+row['transcript']+'\n' for row in changed['rows']))
            changed['tsv']=campaign.record(output/'bank.tsv')
            (output/'bank.json').unlink();campaign.seal(output/'bank.json',changed)
            with self.assertRaisesRegex(ValueError,'first isolated frozen proposals'):
                strength.validate_bank(plan,output)


if __name__=='__main__':unittest.main()
