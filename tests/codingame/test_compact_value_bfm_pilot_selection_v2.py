import copy
import unittest
from tools import compact_value_bfm_pilot_selection_v2 as selection


def metrics(regret,groups=200,comparable=180,flip=.1):
    return {'mean_teacher_regret':regret,'groups':groups,'comparable_groups':comparable,
        'float_vs_quantized_action_flip_rate':flip}


class PilotSelectionTests(unittest.TestCase):
    def models(self):
        control={'overall':metrics(.2),'early':metrics(.3,150,130)}
        candidate={'lambda':.1,'seed':20260907,'canonical_retention_passed':True,'source_reserve':2300,
            'overall':metrics(.17,flip=.104),'early':metrics(.25,150,130,.104)}
        return control,candidate

    def test_both_strata_must_pass_before_a_screen(self):
        control,candidate=self.models()
        self.assertTrue(selection.compare_candidate(control,candidate)['eligible_for_rank4_screen'])
        candidate['early']['mean_teacher_regret']=.29
        self.assertFalse(selection.compare_candidate(control,candidate)['eligible_for_rank4_screen'])

    def test_thin_early_evidence_cannot_pass(self):
        control,candidate=self.models()
        control['early']=metrics(.3,12,12)
        candidate['early']=metrics(.1,12,12)
        self.assertFalse(selection.compare_candidate(control,candidate)['eligible_for_rank4_screen'])

    def test_retention_and_flip_limits_cannot_be_bypassed(self):
        control,candidate=self.models();candidate['canonical_retention_passed']=False
        self.assertFalse(selection.compare_candidate(control,candidate)['eligible_for_rank4_screen'])
        candidate['canonical_retention_passed']=True;candidate['overall']['float_vs_quantized_action_flip_rate']=.106
        self.assertFalse(selection.compare_candidate(control,candidate)['eligible_for_rank4_screen'])

if __name__=='__main__':unittest.main()
