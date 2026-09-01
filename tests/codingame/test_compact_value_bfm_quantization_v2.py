import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compact_value_bfm_quantization_v2 as v2
import compact_value_bfm_quantization_v2_holdout as wrapper


q = v2.qualification


class QuantizationRosterTest(unittest.TestCase):
    def test_candidate_roster_is_exact_and_finite(self):
        self.assertEqual(v2.QAT_LEARNING_RATES, (7.5e-6, 1.5e-5, 3e-5))
        self.assertEqual(v2.SCHEDULE_EPOCHS, (51, 52, 53, 54))
        self.assertEqual(v2.FIXED_CANDIDATES, 13)
        self.assertEqual(v2.CALIBRATION_CANDIDATES, 102)
        self.assertEqual(v2.TOTAL_CANDIDATES, 115)
        self.assertEqual(tuple(v2.SCALE_FACTORS), ("w1", "w2", "w3"))

    def test_stable_ordinal_is_final_selection_tiebreak(self):
        float_metrics = {
            "common_adjudicator": {}, "canonical_validation": {}
        }
        metrics = {"common_adjudicator": {}, "canonical_validation": {}}
        with mock.patch.object(
            v2.iteration,
            "gate_feasibility_key",
            return_value=(0.0, 0.0, 0.0),
        ):
            first = v2._candidate_key(float_metrics, metrics, 3)
            second = v2._candidate_key(float_metrics, metrics, 4)
        self.assertLess(first, second)
        self.assertEqual(first[-1], 3.0)

    def test_candidate_record_binds_gate_key_and_float32_scales(self):
        metrics = {
            "common_adjudicator": {"weighted_huber": 0.1},
            "canonical_validation": {"weighted_huber": 0.1},
        }
        with (
            mock.patch.object(
                v2.compact,
                "offline_advancement_gate",
                return_value={"passed": False, "status": "rejected", "errors": []},
            ),
            mock.patch.object(
                v2, "_candidate_key", return_value=(1.0, 2.0, 7.0)
            ),
        ):
            record = v2._candidate_record(
                ordinal=7,
                kind="fixture",
                lr=1.5e-5,
                qat_epoch=2,
                scales={"w1": 0.1, "w2": 0.2, "w3": 0.3},
                metrics=metrics,
                float_metrics=metrics,
            )
        self.assertEqual(record["ordinal"], 7)
        self.assertEqual(record["selection_key"], [1.0, 2.0, 7.0])
        self.assertEqual(record["offline_gate"]["passed"], False)


class QuantizationAuthorizationTest(unittest.TestCase):
    def test_authorization_forbids_every_scope_expansion(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            rejection = root / "rejection.json"
            training = root / "training.json"
            checkpoint = root / f"{v2.V1_FLOAT_SHA256}.float.npz"
            q.write_sealed(rejection, {
                "schema": (
                    "papersoccer.compact-value-bfm."
                    "clean-successor-offline-rejection.v1"
                )
            })
            q.write_sealed(training, {"schema": v2.v1.TRAINING_INPUT_SCHEMA})
            checkpoint.write_bytes(b"float")
            state = {
                "paths": {
                    "rejection": rejection,
                    "training_input": training,
                    "float_checkpoint": checkpoint,
                }
            }
            body = v2._authorization_body(state, "2026-09-01T01:00:00Z")
        self.assertFalse(body["float_fine_tuning_authorized"])
        self.assertFalse(body["architecture_change_authorized"])
        self.assertEqual(body["new_training_games_authorized"], 0)
        self.assertEqual(body["new_training_labels_authorized"], 0)
        self.assertEqual(
            body["fresh_protected_holdout_materializations_authorized_after_pass"],
            1,
        )
        self.assertEqual(body["quantization_executions_authorized"], 1)
        self.assertFalse(body["v3_authorized"])
        self.assertFalse(body["upload_authorized"])


class V2HoldoutWrapperTest(unittest.TestCase):
    def test_wrapper_rebinds_only_campaign_validator_and_identity(self):
        self.assertIs(wrapper.fresh.successor, wrapper.v2)
        self.assertIs(wrapper.fresh.qualification, wrapper.v2.qualification)
        self.assertEqual(
            wrapper.fresh.CAMPAIGN_ID,
            f"{wrapper.v2.SUCCESSOR_CAMPAIGN_ID}-holdout",
        )
        self.assertEqual(wrapper.fresh.NAMESPACE, wrapper.v2.NAMESPACE)
        self.assertIs(wrapper.fresh._packing_priors, wrapper._v2_packing_priors)


if __name__ == "__main__":
    unittest.main()
