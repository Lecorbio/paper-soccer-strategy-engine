"""Run the retained numerical/recovery cases against the current trainer."""
from unittest import mock
from tests.codingame import test_rank_two_focused_training_v1 as retained
from tools import rank_two_focused_training_v3 as training


class Numerics(retained.Numerics):
    def setUp(self):
        patch = mock.patch.object(retained, "training", training)
        patch.start()
        self.addCleanup(patch.stop)
