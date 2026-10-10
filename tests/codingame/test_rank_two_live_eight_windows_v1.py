import unittest
from tools import rank_two_live_eight_windows_v1 as windows


class Boundaries(unittest.TestCase):
    def test_later_games_cannot_choose_a_more_favorable_calibration(self):
        submission=dict(agent_id=1,submission_id=2,test_session_handle='session')
        games=[dict(gameId=gid,done=True,players=[dict(playerAgentId=1,submissionId=2,testSessionHandle='session')]) for gid in reversed(range(1,166))]
        snapshot=dict(user=dict(agentId=1,testSessionHandle='session',percentage=100),battles=games)
        complete,ids=windows.validate_window(dict(expected_games=90),submission,snapshot)
        self.assertTrue(complete);self.assertEqual(ids,list(range(1,91)))
        snapshot['battles'][0]['players'][0]['submissionId']=3
        with self.assertRaises(ValueError):windows.validate_window(dict(expected_games=90),submission,snapshot)


if __name__=='__main__':unittest.main()
