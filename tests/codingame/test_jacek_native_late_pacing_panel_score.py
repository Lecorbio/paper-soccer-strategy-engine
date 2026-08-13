import importlib.util
import argparse
import pathlib
import tempfile
from unittest import mock
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "late_pacing_score",
    ROOT / "tools" / "jacek_native_late_pacing_panel_score.py",
)
score = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(score)
import jacek_native_late_trap_panel as panel_builder  # noqa: E402


class LatePacingPanelScoreTest(unittest.TestCase):
    @staticmethod
    def action(encoded, unpenalized, *, eligible, penalty=0.0):
        return {
            "encoded": encoded,
            "value": unpenalized,
            "initial_value": unpenalized - 0.1,
            "visits": 10,
            "selection_visits": 9,
            "tactical_class": "safe-handoff",
            "solved": False,
            "proven_win": False,
            "exact_reply_refuted": False,
            "start_progress": 2 if eligible else 3,
            "endpoint_progress": 4,
            "supported_advance_eligible": eligible,
            "supported_advance_penalty": penalty if eligible else 0.0,
            "unpenalized_final_score": unpenalized,
            "penalized_final_score": (
                unpenalized - penalty if eligible else unpenalized
            ),
        }

    @classmethod
    def row(cls, penalty):
        first = cls.action("a", 0.2, eligible=True, penalty=penalty)
        second = cls.action("b", 0.195, eligible=False, penalty=penalty)
        return {
            "pre_action_used_edges": 20,
            "search_root_actions": 2,
            "search_root_action_diagnostics": [first, second],
            "chosen_action": "a" if penalty == 0.0 else "b",
        }

    def test_independent_supported_advance_predicate_is_literal_crossing(self):
        action = self.action("a", 0.2, eligible=True)
        self.assertTrue(score.independently_eligible(47, action))
        self.assertFalse(score.independently_eligible(48, action))
        action["start_progress"] = 1
        self.assertFalse(score.independently_eligible(47, action))
        action["start_progress"] = 2
        action["endpoint_progress"] = 3
        self.assertFalse(score.independently_eligible(47, action))
        action["endpoint_progress"] = 4
        action["proven_win"] = True
        self.assertFalse(score.independently_eligible(47, action))

    def test_penalty_is_only_a_final_argmax_overlay(self):
        zero = {"state": self.row(0.0)}
        penalized = {"state": self.row(0.11)}
        score.verify_penalty_overlay(zero, penalized, 0.11)
        self.assertEqual(score.offline_argmax(zero["state"], 0.0), "a")
        self.assertEqual(score.offline_argmax(zero["state"], 0.11), "b")

    def test_penalty_rejects_search_churn(self):
        zero = {"state": self.row(0.0)}
        penalized = {"state": self.row(0.11)}
        penalized["state"]["search_root_action_diagnostics"][0]["visits"] += 1
        with self.assertRaisesRegex(ValueError, "penalty changed search"):
            score.verify_penalty_overlay(zero, penalized, 0.11)

    def test_penalty_cli_binds_primary_and_guard_audits(self):
        arguments = score.parser().parse_args([
            "penalty", "--panel", "panel.json", "--runtime", "h62.runtime",
            "--zero-audit", "p0.jsonl", "--penalty-11-audit", "p011.jsonl",
            "--penalty-15-audit", "p015.jsonl", "--output", "score.json",
        ])
        self.assertEqual(arguments.penalty_11_audit, pathlib.Path("p011.jsonl"))
        self.assertEqual(arguments.penalty_15_audit, pathlib.Path("p015.jsonl"))

    def test_validation_routes_h62_round2_as_strict_current(self):
        arguments = argparse.Namespace(
            baseline_model=pathlib.Path("baseline.json"),
            baseline_seed=20260822,
            candidate_model=pathlib.Path("candidate.json"),
            candidate_seed=[20260901],
            archived_round1=[pathlib.Path("round1.jsonl")],
            current_round2=[pathlib.Path("round2.jsonl")],
            archived_restart_round2=[pathlib.Path("restart.jsonl")],
        )
        baseline = {"provenance": {}, "checkpoints": [], "seed_reports": []}
        candidate = {"checkpoints": []}
        with (
            mock.patch.object(
                score, "load_canonical_json",
                side_effect=((baseline, "b" * 64), (candidate, "c" * 64)),
            ),
            mock.patch.object(score, "explicit_file", side_effect=lambda p, _: p),
            mock.patch.object(score.trainer, "load_datasets",
                              side_effect=RuntimeError("sentinel")) as loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                score.score_validation(arguments)
        loader.assert_called_once_with(
            [pathlib.Path("round2.jsonl")],
            archived_round1_paths=[pathlib.Path("round1.jsonl")],
            archived_restart_round2_paths=[pathlib.Path("restart.jsonl")],
        )

    def test_frozen_loader_restores_strict_validator_after_failure(self):
        contract = score.trainer.corpus_contract
        original = contract._validate_round2_build_contract
        with mock.patch.object(
            score.trainer, "load_datasets", side_effect=RuntimeError("sentinel")
        ):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                score.load_frozen_h62_datasets([], [], [])
        self.assertIs(contract._validate_round2_build_contract, original)

    def test_history62_model_uses_canonical_trainer_numeric_depth_order(self):
        model_path = ROOT / "models" / "jacek_native_history62_champion.json"
        model, digest = score.load_canonical_json(
            model_path, "baseline model", trainer_model=True,
        )
        self.assertEqual(
            digest,
            "b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14",
        )

        lexical = score.canonical_json_bytes(model)
        self.assertNotEqual(model_path.read_bytes(), lexical)
        with tempfile.TemporaryDirectory() as directory:
            reordered = pathlib.Path(directory) / "reordered-model.json"
            reordered.write_bytes(lexical)
            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                score.load_canonical_json(
                    reordered, "baseline model", trainer_model=True,
                )

    def test_v2_panel_contract_is_supported(self):
        traps = [
            {"auditor_state_id": f"trap-audit-{index}",
             "state_id": f"trap-state-{index}",
             "canonical_key": f"trap-key-{index}"}
            for index in range(96)
        ]
        controls = [
            {"auditor_state_id": f"control-audit-{index}",
             "state_id": f"control-state-{index}",
             "canonical_key": f"control-key-{index}",
             "matched_trap_state_id": f"trap-state-{index}"}
            for index in range(96)
        ]
        indexed_traps, indexed_controls, pairs = score.panel_records({
            "schema": "papersoccer.jacek-native-late-trap-panel/v2",
            "trap_states": traps,
            "matched_winning_controls": controls,
        })
        self.assertEqual(len(indexed_traps), 96)
        self.assertEqual(len(indexed_controls), 96)
        self.assertEqual(len(pairs), 96)
        controls[0]["canonical_key"] = traps[0]["canonical_key"]
        with self.assertRaisesRegex(ValueError, r"96\+96 unique states"):
            score.panel_records({
                "schema": "papersoccer.jacek-native-late-trap-panel/v2",
                "trap_states": traps,
                "matched_winning_controls": controls,
            })

    def test_v2_control_matching_excludes_trap_canonical_keys(self):
        trap = {
            "canonical_key": "shared", "state_id": "trap-state",
            "auditor_state_id": "trap-audit",
            "candidate_player": 0, "turn_band": "early",
            "zone": "enemy-shell", "used_edge_band": "sparse",
            "prefix_turn": 8, "used_edges": 8, "run_id": "trap-run",
            "game_id": "1", "role": "last-enemy-shell",
        }
        shared = {
            **trap, "run_id": "control-run", "game_id": "2",
            "role": "matched-winning-control",
        }
        replacement = {
            **shared, "canonical_key": "replacement", "state_id": "fresh",
            "auditor_state_id": "fresh-audit", "game_id": "3",
            "used_edges": 9,
        }
        games = [{
            "winner": 0, "candidate_player": 0,
            "candidate_geometry": [shared, replacement],
        }]
        with mock.patch.object(
            panel_builder, "make_entry",
            side_effect=lambda _game, item, _role, _mate: dict(item),
        ):
            controls = panel_builder.control_entries(games, [trap])
        self.assertEqual([item["canonical_key"] for item in controls], [
            "replacement"
        ])
        panel_builder.require_disjoint_populations([trap], controls)

    def test_v2_legacy_equivalence_binds_191_state_subset_and_precision(self):
        missing = score.V2_LEGACY_MISSING_STATE
        current_values = (
            0.051730599254369736,
            0.6437437534332275,
            -12.34567894,
            12_345.67894,
            0.000001234567894,
        )
        legacy_values = (
            0.0517305993,
            0.643743753,
            -12.3456789,
            12_345.6789,
            0.00000123456789,
        )
        baseline = {
            missing: {"state_id": missing},
            **{
                f"state-{index}": {
                    "state_id": f"state-{index}",
                    "initial_best_value": current_values[
                        index % len(current_values)
                    ],
                    "chosen_action": "34",
                }
                for index in range(191)
            },
        }
        legacy = [
            {
                "state_id": f"state-{index}",
                "initial_best_value": legacy_values[
                    index % len(legacy_values)
                ],
                "chosen_action": "34",
                "schema_version": "jacek-native-decision-audit-v2",
                "search_elapsed_ms": index,
            }
            for index in range(191)
        ]
        coverage = score.verify_legacy_equivalence(
            {"schema": score.V2_PANEL_SCHEMA}, legacy, baseline,
        )
        self.assertEqual(coverage, {
            "expected_states": 192,
            "covered_unique_states": 191,
            "compared_states": 191,
            "legacy_rows": 191,
            "missing_states": [missing],
        })

        legacy[0]["initial_best_value"] = 0.0517305992
        with self.assertRaisesRegex(ValueError, "legacy drift at state-0"):
            score.verify_legacy_equivalence(
                {"schema": score.V2_PANEL_SCHEMA}, legacy, baseline,
            )


if __name__ == "__main__":
    unittest.main()
