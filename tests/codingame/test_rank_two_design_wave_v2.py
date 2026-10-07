import copy
import unittest

from tools import rank_two_design_wave_v2 as wave


class ReadinessTests(unittest.TestCase):
    def test_activation_requires_explicit_owner_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "explicit execution owner"):
            wave.activate("missing-campaign", "missing-audit", 1)

    def setUp(self):
        self.state = {"owner_thread_id": "owner", "active_claims": {}, "active_jobs": [],
                      "helper": None, "incident": None, "incumbent": {"sha256": "inc"}}
        self.approval = {"approved_by": "human", "owner_thread_id": "owner"}
        self.plan = {"status": "approved-predecessor-open", "owner_thread_id": "owner",
                     "maximum_frozen_sources": 6, "duration_days": 7, "clocks_ms": [990, 180]}
        self.audit = {key: True for key in ("passed", "declared_comparison_closed", "restoration_complete", "all_played_windows_archived")}
        self.audit.update({key: "inc" for key in ("active_source_sha256", "editor_source_sha256", "deployed_source_sha256")})
        self.audit.update(comparison_result={"path": "result", "sha256": "hash"},
                          deployed_submission={"path": "submission", "sha256": "hash"})

    def check(self, **options):
        return wave.readiness(self.state, self.approval, self.plan, self.audit, "inc", **options)

    def test_audited_predecessor_is_ready(self):
        self.assertEqual(self.check(), "inc")

    def test_unresolved_work_never_resets_clock(self):
        for key, value in (("active_claims", {"attempt": {}}), ("active_jobs", [{}]),
                           ("helper", {}), ("incident", {"reason": "failure"})):
            with self.subTest(key=key):
                previous = copy.deepcopy(self.state)
                self.state[key] = value or {"active": True}
                with self.assertRaises(ValueError):
                    self.check()
                self.state = previous
        for option in ("stop", "restore"):
            with self.assertRaises(ValueError):
                self.check(**{option: True})

    def test_wrong_identity_or_incomplete_audit_rejects(self):
        for key in ("editor_source_sha256", "deployed_source_sha256", "active_source_sha256", "restoration_complete", "all_played_windows_archived"):
            with self.subTest(key=key):
                previous = copy.deepcopy(self.audit)
                self.audit[key] = False
                with self.assertRaises(ValueError):
                    self.check()
                self.audit = previous

    def test_reactivation_or_changed_limits_reject(self):
        self.state["design_wave_activation"] = {"path": "prior", "sha256": "hash"}
        with self.assertRaises(ValueError):
            self.check()
        del self.state["design_wave_activation"]
        for key, value in (("duration_days", 8), ("maximum_frozen_sources", 7), ("clocks_ms", [990, 190])):
            with self.subTest(key=key):
                previous = copy.deepcopy(self.plan)
                self.plan[key] = value
                with self.assertRaises(ValueError):
                    self.check()
                self.plan = previous


if __name__ == "__main__":
    unittest.main()
