import unittest
from tools import rank_two_live_eight_data_v1 as data
from tools import rank_two_live_eight_timing_v1 as timing


class Policy(unittest.TestCase):
    def test_historical_and_qualification_inputs_are_rejected_before_game_reads(self):
        for purpose in ('qualification','private','formal','confirmation'):
            with self.assertRaises(PermissionError):
                data.parents(dict(extension=data.c.SCHEMA,purpose=purpose,complete=True,training_designated=True),{},2026101001)
        with self.assertRaises(PermissionError):
            data.parents(dict(extension='old',purpose='exploration',complete=True,training_designated=True),{},2026101001)

    def test_timing_cannot_overlap_a_live_window_or_other_work(self):
        plan=dict(job_id='timing')
        with self.assertRaises(PermissionError):
            timing.isolated(dict(live_eight_known_windows={'live':dict(all90_archived=False)}),plan)
        with self.assertRaises(PermissionError):
            timing.isolated(dict(active_jobs=[dict(id='training')]),plan)
        timing.isolated(dict(active_jobs=[dict(id='timing')]),plan)


if __name__=='__main__':unittest.main()
