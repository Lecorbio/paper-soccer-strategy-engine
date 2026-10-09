import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_campaign_v2 as controller


def usage(value=7, observed=None):
    return dict(observed_at=(observed or controller.now()).isoformat(), used_percent=value,
                window_duration_mins=10080, actual_response=dict(ordinaryUsageAllowed=True,
                rateLimitsByLimitId=dict(main=dict(primary=dict(usedPercent=value, windowDurationMins=10080)))))


def state():
    return dict(owner_thread_id="owner", ownership_status="active", worktree=str(controller.ROOT),
        recovery_minutes=5, focused_activation=dict(path="fixture", sha256="activation"),
        focused_setup_ready=None, research_closed=False, deadline_utc="2026-10-21T21:28:13+00:00",
        preparation_deadline_utc="2026-10-11T21:28:13+00:00", focused_live_sources={}, active_claims={})


class ControllerTests(unittest.TestCase):
    def test_weekly_threshold_unknown_and_stale_receipts(self):
        self.assertEqual(controller.actual_usage(usage(69.9)), 69.9)
        with self.assertRaises(PermissionError):
            controller.actual_usage(usage(70))
        with self.assertRaises(PermissionError):
            controller.actual_usage(usage(None))
        with self.assertRaises(PermissionError):
            controller.actual_usage(usage(7, controller.now() - dt.timedelta(seconds=121)))
        body = usage(7)
        body["used_percent"] = 6
        with self.assertRaises(ValueError):
            controller.actual_usage(body)

    def test_all_reported_weekly_buckets_apply(self):
        body = usage(7)
        body["actual_response"]["rateLimitsByLimitId"]["other"] = dict(
            secondary=dict(usedPercent=71, windowDurationMins=10080))
        body["used_percent"] = 71
        with self.assertRaises(PermissionError):
            controller.actual_usage(body)

    def test_resource_caps_and_nonfinite_budgets(self):
        budget = dict(workers=10, rss_bytes=18 * 1024**3, numerical_threads=1,
                      wall_seconds=1200, cpu_capacity_seconds=14400)
        self.assertTrue(controller.resource_guard(budget))
        for key, value in [("workers", 11), ("rss_bytes", 19 * 1024**3), ("numerical_threads", 2),
                           ("wall_seconds", float("inf")), ("cpu_capacity_seconds", float("nan"))]:
            with self.assertRaises(ValueError):
                controller.resource_guard({**budget, key: value})

    def test_owner_and_exact_source_bans_precede_claim(self):
        body = state()
        at = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)
        with self.assertRaises(PermissionError):
            controller.validate(body, {}, "other", "enqueue", 7, at=at)
        with self.assertRaisesRegex(PermissionError, "failed/retired source"):
            controller.validate(body, dict(operationally_failed=[dict(source_sha256="failed")]),
                                "owner", "claim", 7, source="failed", attempt="slot", at=at)

    def test_research_deadline_allows_only_incident_bound_restoration(self):
        body = state()
        at = dt.datetime(2026, 10, 22, tzinfo=dt.timezone.utc)
        with self.assertRaisesRegex(PermissionError, "deadline"):
            controller.validate(body, {}, "owner", "enqueue", 7, at=at)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("incumbent.cpp").write_text("int main(){}\n")
            incumbent = controller.record(root / "incumbent.cpp")
            failure = controller.immutable(root / "FAILURE.json", dict(own_failure=True))
            body.update(incumbent=incumbent, incident=dict(receipt=failure), restoration_authorization=True)
            with mock.patch.object(controller, "BASELINE", incumbent["sha256"]):
                with self.assertRaisesRegex(PermissionError, "immutable bound receipt"):
                    controller.validate(body, {}, "owner", "claim", 7, source=incumbent["sha256"], attempt="restore", at=at)
                authorization = dict(owner_thread_id="owner", activation=body["focused_activation"],
                    incident=failure, incumbent=incumbent, source_sha256=incumbent["sha256"],
                    purpose="mandatory-restoration", one_window_only=True, attempt_id="restore")
                body["restoration_authorization"] = controller.immutable(root / "AUTH.json", authorization)
                self.assertTrue(controller.validate(body, {}, "owner", "claim", 7,
                    source=incumbent["sha256"], attempt="restore", at=at))
                with self.assertRaises(PermissionError):
                    controller.validate(body, {}, "owner", "claim", 7,
                        source=incumbent["sha256"], attempt="other-slot", at=at)
                body["focused_restorations"] = {failure["sha256"]: dict(attested_window_id=123)}
                with self.assertRaisesRegex(PermissionError, "already attested"):
                    controller.validate(body, {}, "owner", "claim", 7,
                        source=incumbent["sha256"], attempt="restore", at=at)

    def test_claim_requires_declared_slot_and_completed_window_archive(self):
        body = state()
        body["incumbent"] = dict(path="fixture", sha256=controller.BASELINE)
        at = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)
        with mock.patch.object(controller.retained, "validate", return_value=True), mock.patch.object(controller, "verify"):
            with self.assertRaisesRegex(PermissionError, "predeclared"):
                controller.validate(body, {}, "owner", "claim", 7, source=controller.BASELINE, attempt="slot", at=at)
            body["focused_declared_windows"] = dict(slot=dict(source_sha256=controller.BASELINE, plan=dict(path="plan", sha256="plan")))
            body["focused_known_windows"] = {123: dict(all90_archived=False)}
            with self.assertRaisesRegex(PermissionError, "Archive|archive"):
                controller.validate(body, {}, "owner", "claim", 7, source=controller.BASELINE, attempt="slot", at=at)
            body["focused_known_windows"][123]["all90_archived"] = True
            self.assertTrue(controller.validate(body, {}, "owner", "claim", 7, source=controller.BASELINE, attempt="slot", at=at))
            body["focused_declared_windows"]["slot"]["attested_window_id"] = 123
            with self.assertRaisesRegex(PermissionError, "duplicated"):
                controller.validate(body, {}, "owner", "claim", 7, source=controller.BASELINE, attempt="slot", at=at)


if __name__ == "__main__":
    unittest.main()
