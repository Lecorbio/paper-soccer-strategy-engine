import pathlib
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compact_value_bfm_discrete_v3 as v3
import compact_value_bfm_discrete_v3_holdout as wrapper


class DiscreteRosterTest(unittest.TestCase):
    def test_grid_roster_is_exact(self):
        self.assertEqual(v3.BASE_W3_CODES, (0, 3, -1, 1, -3, -2, 0, -1))
        self.assertEqual(v3.CODE_VECTOR_COUNT, 2_916)
        self.assertEqual(v3.SCALE_COUNT, 2_301)
        self.assertEqual(v3.TOTAL_GRID_CANDIDATES, 6_709_716)
        self.assertEqual(v3.EXPECTED_PASSING_CANDIDATES, 404)

    def test_scale_ticks_have_exact_float32_endpoints_and_winner(self):
        scales = v3._grid_scales()
        self.assertEqual(len(scales), 2_301)
        self.assertEqual(float(scales[0]), float(np.float32(0.005)))
        self.assertEqual(float(scales[-1]), float(np.float32(0.120)))
        self.assertIn(np.float32(v3.EXPECTED_WINNER_SCALE), scales)

    def test_vectorized_huber_matches_maintained_scalar_metric(self):
        dataset = v3.compact.Dataset(
            indptr=np.asarray([0, 1, 2], dtype="<i8"),
            indices=np.asarray([316, 373], dtype="<u2"),
            targets=np.asarray([0.25, -0.5], dtype="<f4"),
            weights=np.asarray([1.0, 2.0], dtype="<f4"),
            group_ids=np.asarray([b"a" * 32, b"b" * 32], dtype="V32"),
            split="validation",
            source_manifest_sha256="a" * 64,
            source_npz_sha256="b" * 64,
        )
        projection = np.asarray([0.7, -0.4], dtype=np.float32)
        scales = np.asarray([0.5, 1.0], dtype=np.float32)
        observed = v3._huber_for_scales(projection, scales, dataset)
        expected = []
        for scale in scales:
            predictions = v3.compact.fast_tanh(projection * scale)
            loss, _gradient = v3.compact._weighted_huber_loss_gradient(
                predictions, dataset.targets, dataset.weights
            )
            expected.append(loss)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)


class DiscreteAuthorizationTest(unittest.TestCase):
    def test_authorization_is_w3_only_and_one_shot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            outcome = root / "outcome.json"
            runtime = root / "runtime.json"
            v3.qualification.write_sealed(outcome, {"schema": v3.v2.OUTCOME_SCHEMA})
            runtime.write_bytes(b"runtime")
            state = {"paths": {"outcome": outcome, "runtime": runtime}}
            body = v3._authorization_body(state, "2026-09-01T02:00:00Z")
        self.assertFalse(body["w1_changes_authorized"])
        self.assertFalse(body["w2_changes_authorized"])
        self.assertTrue(body["w3_local_code_and_scale_search_authorized"])
        self.assertEqual(body["canonical_postplan_search_executions_authorized"], 1)
        self.assertEqual(
            body["preplan_unprotected_development_diagnostics"]["dense_full_grid_runs"],
            4,
        )
        self.assertFalse(body["v4_authorized"])
        self.assertFalse(body["upload_authorized"])


class V3HoldoutWrapperTest(unittest.TestCase):
    def test_wrapper_binds_v3_validator_and_prior_override(self):
        self.assertIs(wrapper.fresh.successor, wrapper.v3)
        self.assertEqual(
            wrapper.fresh.CAMPAIGN_ID,
            f"{wrapper.v3.SUCCESSOR_CAMPAIGN_ID}-holdout",
        )
        self.assertIs(wrapper.fresh._packing_priors, wrapper._v3_packing_priors)


if __name__ == "__main__":
    unittest.main()
