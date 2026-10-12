"""A late activation or new predecessor game must prevent another UI action."""
import copy
import unittest

from tools import rank_two_live_eight_windows_v2 as windows


class RetryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(user=dict(agentId=10, percentage=100, testSessionHandle='old'),
            battles=[dict(gameId=20, done=True, players=[dict(playerAgentId=10, submissionId=30)])])

    def test_same_complete_window_is_eligible(self):
        windows.unchanged(self.before, copy.deepcopy(self.before))

    def test_changed_submission_prevents_retry(self):
        after = copy.deepcopy(self.before)
        after['battles'][0]['players'][0]['submissionId'] = 31
        with self.assertRaises(PermissionError):
            windows.unchanged(self.before, after)

    def test_changed_games_prevent_retry(self):
        after = copy.deepcopy(self.before)
        after['battles'][0]['gameId'] = 21
        with self.assertRaises(PermissionError):
            windows.unchanged(self.before, after)

    def test_incomplete_predecessor_prevents_retry(self):
        after = copy.deepcopy(self.before)
        after['user']['percentage'] = 50
        with self.assertRaises(PermissionError):
            windows.unchanged(self.before, after)


if __name__ == '__main__':
    unittest.main()
