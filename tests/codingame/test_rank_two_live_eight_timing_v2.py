"""Timing cannot share its authoritative phase with admitted or claimed work."""
import unittest
from tools import rank_two_live_eight_timing_v2 as timing


class Isolation(unittest.TestCase):
    def test_single_current_job_is_accepted(self):
        timing.isolated(dict(active_jobs=[dict(id='timing')],live_eight_known_windows={}),dict(job_id='timing'))

    def test_other_workload_helper_or_claim_is_rejected(self):
        for state in (dict(active_jobs=[dict(id='other')]),dict(active_helper={'id':'helper'}),
                      dict(active_claims={'window':{}})):
            with self.assertRaises(PermissionError):timing.isolated(state,dict(job_id='timing'))

    def test_unarchived_window_is_rejected(self):
        with self.assertRaises(PermissionError):
            timing.isolated(dict(live_eight_known_windows={'window':dict(all90_archived=False)}),dict(job_id='timing'))


if __name__=='__main__':unittest.main()
