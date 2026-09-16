"""Arithmetic checks on the public, sanitized training-v2 outcome."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TrainingOutcomeTests(unittest.TestCase):
    def test_published_outcome_integrity_and_claim_boundary(self):
        path = ROOT / "benchmarks/compact_value_bfm/trained-v2-outcome.json"
        payload = path.read_text()
        value = json.loads(payload)
        digest = value.pop("body_sha256")
        canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
        self.assertNotIn("/Users/", payload)
        self.assertEqual(value["status"], "closed-without-qualifying-candidate")
        self.assertEqual(value["short_jobs_completed"] + value["unused_conditional_replication_slots"], value["short_job_budget"])
        self.assertFalse(value["assessment"]["formal_pilot_pass"])

    def test_reported_pair_does_not_meet_the_predeclared_criterion(self):
        value = json.loads((ROOT / "benchmarks/compact_value_bfm/trained-v2-outcome.json").read_text())
        scalar, ranking = value["arms"]
        a = scalar["quantized_validation"]["successor_ranking"]["mean_teacher_regret"]
        b = ranking["quantized_validation"]["successor_ranking"]["mean_teacher_regret"]
        self.assertAlmostEqual((a-b)/a, value["assessment"]["quantized_regret_reduction_vs_fresh_scalar"])
        self.assertGreater(b, a)
        self.assertLess(b, ranking["historical_quantized_regret"])
        self.assertFalse(value["assessment"]["promising_criterion_passed"])
        for arm in value["arms"]:
            self.assertEqual(arm["selected_qat_epoch"], 3)
            self.assertEqual(arm["consistency_batches"], 104)
            self.assertFalse(arm["retention"]["passed"])
            self.assertTrue(arm["native_feature_check"]["passed"])
            self.assertGreaterEqual(arm["source_size"]["reserve"], 2000)


if __name__ == "__main__":
    unittest.main()
