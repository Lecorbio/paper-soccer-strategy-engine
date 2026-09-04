import copy
import unittest
from tools import compact_value_bfm_opponent_evaluation as evaluation
from tools import compact_value_bfm_campaign_v2 as campaign


def fixture(winning_roots):
    result={}
    for opponent in campaign.OPPONENTS:
        result[opponent]={}
        for root in range(32):
            for color in (0,1):
                result[opponent][(f'{opponent}:{root}',color)]={
                    'schema':'papersoccer.compact-state-evaluation.v2',
                    'root_id':f'{opponent}:{root}','candidate_player':color,
                    'winner':color if root<winning_roots else 1-color,'failure':'',
                    'root_edges':8 if root<16 else 40,
                    'first_budget_ms':800,'later_budget_ms':155}
    return result


class OpponentEvaluationTests(unittest.TestCase):
    def test_paired_improvement_is_bootstrapped_by_root(self):
        result=evaluation.assess(fixture(16),fixture(8))
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['equal_weight_improvement'],.25)
        self.assertGreater(result['paired_95_interval'][0],0)
        self.assertFalse(result['live_success'])

    def test_one_style_regression_blocks_mean_improvement(self):
        candidate=fixture(16); candidate[campaign.OPPONENTS[0]]=fixture(0)[campaign.OPPONENTS[0]]
        result=evaluation.assess(candidate,fixture(8))
        self.assertGreater(result['equal_weight_improvement'],.03)
        self.assertFalse(result['passed'])

    def test_failure_ties_and_wrong_clocks_cannot_pass(self):
        self.assertFalse(evaluation.assess(fixture(8),fixture(8))['passed'])
        candidate=fixture(16);next(iter(candidate[campaign.OPPONENTS[0]].values()))['failure']='candidate-timeout'
        self.assertFalse(evaluation.assess(candidate,fixture(8))['passed'])
        candidate=fixture(16);next(iter(candidate[campaign.OPPONENTS[0]].values()))['later_budget_ms']=1
        with self.assertRaisesRegex(ValueError,'actual 800/155'):
            evaluation.assess(candidate,fixture(8))

    def test_missing_pair_is_not_dropped(self):
        candidate=fixture(16); candidate[campaign.OPPONENTS[0]].pop(next(iter(candidate[campaign.OPPONENTS[0]])))
        with self.assertRaisesRegex(ValueError,'schedules differ'):
            evaluation.assess(candidate,fixture(8))

if __name__=='__main__': unittest.main()
