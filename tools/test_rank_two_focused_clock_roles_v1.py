"""Keep focused response limits distinct from frozen roster engine budgets."""
import unittest
from unittest.mock import patch

from tools import rank_two_focused_runtime_v5 as previous
from tools import rank_two_focused_runtime_v6 as revised


class ClockRoles(unittest.TestCase):
    def query(self, policy, budget, external, elapsed):
        worker=object.__new__(revised.Worker)
        worker.response_policy=policy
        with patch.object(revised.retained.Worker,'choose',return_value=('3',elapsed)), \
             patch.object(revised.time,'monotonic',side_effect=[0,elapsed/1000]):
            return worker.choose('-',budget,external)

    def test_focus_and_control_do_not_borrow_external_headroom(self):
        for budget,external,elapsed in [(140,200,158.075),(550,1000,977.213)]:
            with self.assertRaises(revised.retained.WorkerResponseFailure):
                self.query('strict-focused-response',budget,external,elapsed)

    def test_frozen_opponent_keeps_engine_budget_and_external_limit(self):
        self.assertEqual(self.query('frozen-opponent-engine-budget',155,200,158.075)[0],'3')
        with self.assertRaises(revised.retained.WorkerResponseFailure):
            self.query('frozen-opponent-engine-budget',155,200,200.001)

    def test_focused_native_wrapper_is_byte_identical(self):
        self.assertEqual(previous.worker_source({}),revised.worker_source({},'strict-focused-response'))
        opponent=revised.worker_source({},'frozen-opponent-engine-budget')
        self.assertIn('std::min(budget,(budget>200?1000:200)-10-',opponent)


if __name__=='__main__':unittest.main()
