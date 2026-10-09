"""Campaign boundary tests independent of historical research artifacts."""
import copy
import datetime as dt
import tempfile
from pathlib import Path
import unittest

from tools import rank_two_focused_campaign_v1 as campaign


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
        self.state = dict(owner_thread_id="owner", ownership_status="active",
                          worktree=str(campaign.ROOT), recovery_minutes=5,
                          focused_activation={"path": "activation", "sha256": "digest"},
                          deadline_utc=(self.at+dt.timedelta(days=14)).isoformat(),
                          preparation_deadline_utc=(self.at+dt.timedelta(days=4)).isoformat(),
                          research_closed=False, focused_live_sources={}, active_claims={},
                          frozen_sources={}, pending_exposures=[], focused_setup_ready=None)
        self.closures = dict(operationally_failed=[], retired_implementations=[])

    def call(self, action="enqueue", used=7, **kwargs):
        return campaign.validate(self.state, self.closures, "owner", action, used, at=self.at, **kwargs)

    def test_weekly_reserve_boundary_and_unknown(self):
        self.assertTrue(self.call(used=69.999))
        for value in (70, 100, None, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(PermissionError):
                self.call(used=value)
        self.assertTrue(self.call(action="checkpoint", used=70))

    def test_owner_pause_and_deadline(self):
        self.state["ownership_status"] = "handoff"
        with self.assertRaises(PermissionError): self.call()
        self.state["ownership_status"] = "active"
        self.state["human_pause"] = {"status": "paused"}
        with self.assertRaises(PermissionError): self.call()
        self.state.pop("human_pause")
        self.at += dt.timedelta(days=14)
        with self.assertRaises(PermissionError): self.call()
        self.assertTrue(self.call(action="archive"))

    def test_setup_and_selection_stage_gates(self):
        with self.assertRaises(PermissionError): self.call(action="train")
        self.state["focused_setup_ready"] = {"path": "ready", "sha256": "digest"}
        self.state["selection_deadline_utc"] = (self.at+dt.timedelta(days=3)).isoformat()
        self.assertTrue(self.call(action="train"))
        self.state["pending_exposures"] = [{"path": "unindexed", "sha256": "digest"}]
        with self.assertRaises(PermissionError): self.call(action="prepare-bank")
        self.at += dt.timedelta(days=3)
        with self.assertRaises(PermissionError): self.call()

    def test_failed_source_duplicate_claim_and_submission_binding(self):
        self.closures["operationally_failed"].append({"source_sha256": "failed"})
        with self.assertRaises(PermissionError): self.call(action="admit", source="failed")
        self.state["active_claims"] = {"spent": {"source_sha256": "same"}}
        with self.assertRaises(PermissionError): self.call(action="claim")
        with self.assertRaises(PermissionError): self.call(action="submit", attempt="other")


class SelectionTests(unittest.TestCase):
    def scores(self, small, large):
        return {f"{family}-{seed}": [[value, value] for _ in range(8)]
                for family, value in (("dd8", small), ("dd12wide", large)) for seed in campaign.SEEDS}

    def test_advantage_and_tie_break(self):
        feasible = dict(dd8=True, dd12wide=True)
        self.assertEqual(campaign.architecture_decision(self.scores(.2,.8), feasible, repetitions=100)["family"], "dd12wide")
        self.assertEqual(campaign.architecture_decision(self.scores(.5,.5), feasible, repetitions=100)["family"], "dd8")

    def test_missing_seed_and_color_coverage_rejected(self):
        scores = self.scores(.2,.8)
        scores.pop(next(iter(scores)))
        with self.assertRaises(ValueError): campaign.architecture_decision(scores, dict(dd8=True, dd12wide=True))
        scores = self.scores(.2,.8)
        scores[next(iter(scores))][0] = [.5]
        with self.assertRaises(ValueError): campaign.architecture_decision(scores, dict(dd8=True, dd12wide=True))

    def test_feasibility_does_not_claim_strength(self):
        self.assertIsNone(campaign.architecture_decision({}, dict(dd8=False, dd12wide=False))["family"])
        result = campaign.architecture_decision({}, dict(dd8=False, dd12wide=True))
        self.assertEqual(result["family"], "dd12wide")
        self.assertFalse(result["strength_claim"])


if __name__ == "__main__":
    unittest.main()
