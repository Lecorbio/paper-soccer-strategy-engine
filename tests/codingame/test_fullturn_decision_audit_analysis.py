#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "rank_4_fullturn_bfm"
    / "analyze_decision_audit.py"
)
SPEC = importlib.util.spec_from_file_location("fullturn_decision_audit_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_row(
    game_id="game-1",
    candidate_player=0,
    winner=0,
    own_decision_index=0,
    prefix_actions=(),
):
    row = {field: "" for field in MODULE.STRING_FIELDS}
    row.update({field: 0 for field in MODULE.INTEGER_FIELDS})
    row.update({field: 0.0 for field in MODULE.FLOAT_FIELDS})
    row.update({field: False for field in MODULE.BOOLEAN_FIELDS})
    turn_index = 2 * own_decision_index + candidate_player
    assert len(prefix_actions) == turn_index
    row.update(
        {
            "schema_version": MODULE.AUDIT_SCHEMA_VERSION,
            "input_provenance": {
                "agent_id": "6606663",
                "source_binding_status": "asserted-not-api-verified",
            },
            "game_id": game_id,
            "state_id": f"state-{game_id}-{own_decision_index}",
            "transcript_prefix": "/".join(prefix_actions),
            "turn_index": turn_index,
            "own_decision_index": own_decision_index,
            "candidate_player": candidate_player,
            "color": "player_one" if candidate_player == 0 else "player_two",
            "winner": winner,
            "result": "win" if candidate_player == winner else "loss",
            "actual_action": "0",
            "candidate_action": "0",
            "reference_action": "0",
            "reconstructed_deployed_action": "0",
            "candidate_matches_actual": True,
            "reference_matches_actual": True,
            "candidate_matches_reference": True,
            "reconstructed_deployed_matches_actual": True,
            "diagnostic_root_actions": 4,
            "diagnostic_root_exhaustive": True,
            "diagnostic_root_partial_paths": 8,
            "diagnostic_root_completed_actions": 4,
            "actual_action_retained_ordinal": 0,
            "actual_boundary_retained_ordinal": 0,
            "actual_retained_tactical_class": "safe-handoff",
            "candidate_action_retained_ordinal": 0,
            "candidate_boundary_retained_ordinal": 0,
            "reference_action_retained_ordinal": 0,
            "reference_boundary_retained_ordinal": 0,
            "reference_retained_tactical_class": "safe-handoff",
            "candidate_work_limit": 30_000,
            "candidate_tree_node_limit": 30_000,
            "candidate_max_actions": 250,
            "candidate_max_partial_paths": 3_000_000,
            "candidate_exploration": 1.5,
            "candidate_fpu": 0.5,
            "candidate_final_visit_weight": 1.0,
            "candidate_first_time_limit_ms": 800,
            "candidate_later_time_limit_ms": 165,
            "candidate_time_limit_ms": 800 if own_decision_index == 0 else 165,
            "candidate_elapsed_ms": 10.0,
            "candidate_work": 1_000,
            "candidate_tree_nodes": 500,
            "candidate_expansions": 20,
            "candidate_child_evaluations": 499,
            "candidate_generator_partial_paths": 700,
            "candidate_completed_actions": 400,
            "candidate_generator_duplicates": 10,
            "candidate_tactical_actions": 2,
            "reference_nodes_limit": 30_000,
            "reference_depth_limit": 8,
            "reference_first_time_limit_ms": 800,
            "reference_later_time_limit_ms": 165,
            "reference_time_limit_ms": 800 if own_decision_index == 0 else 165,
            "reference_elapsed_ms": 5.0,
            "reference_nodes": 1_000,
            "reference_completed_turn_depth": 3,
            "reference_attempted_turn_depth": 4,
            "value_blend_percent": 15,
            "teacher_residual_percent": 100,
            "codingame_clock_mode": True,
        }
    )
    return row


def set_candidate_action(row, action, retained_ordinal=1):
    row["candidate_action"] = action
    row["candidate_matches_actual"] = action == row["actual_action"]
    row["candidate_matches_reference"] = action == row["reference_action"]
    row["reconstructed_deployed_action"] = action
    row["reconstructed_deployed_matches_actual"] = action == row["actual_action"]
    row["candidate_action_retained_ordinal"] = retained_ordinal
    row["candidate_boundary_retained_ordinal"] = retained_ordinal


def set_reference_action(row, action, retained_ordinal=2):
    row["reference_action"] = action
    row["reference_matches_actual"] = action == row["actual_action"]
    row["candidate_matches_reference"] = action == row["candidate_action"]
    row["reference_action_retained_ordinal"] = retained_ordinal
    row["reference_boundary_retained_ordinal"] = retained_ordinal


def make_game(game_id, candidate_player, winner, decision_count):
    full_turns = []
    rows = []
    for own_index in range(decision_count):
        turn_index = 2 * own_index + candidate_player
        while len(full_turns) < turn_index:
            full_turns.append("0")
        row = make_row(
            game_id,
            candidate_player,
            winner,
            own_index,
            tuple(full_turns),
        )
        rows.append(row)
        full_turns.append(row["actual_action"])
    return rows


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def tsv_text(rows):
    lines = ["\t".join(MODULE.TSV_FIELD_ORDER)]
    for row in rows:
        values = []
        for field in MODULE.JSON_FIELD_ORDER:
            value = row[field]
            if field == "input_provenance":
                values.append(json.dumps(value, separators=(",", ":"), sort_keys=True))
            elif field in MODULE.BOOLEAN_FIELDS:
                values.append("1" if value else "0")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


class FullTurnDecisionAuditAnalysisTest(unittest.TestCase):
    def test_reports_agreement_pressure_breakdowns_and_first_divergence(self):
        win_rows = make_game("win-game", 0, 0, 13)
        set_candidate_action(win_rows[1], "1")
        win_rows[4]["candidate_deadline_reached"] = True
        win_rows[4]["candidate_generator_truncations"] = 3
        win_rows[5]["actual_action_retained_ordinal"] = -1
        win_rows[5]["actual_boundary_retained_ordinal"] = -1
        win_rows[5]["actual_retained_tactical_class"] = "not-retained"
        win_rows[5]["diagnostic_root_truncations"] = 2
        win_rows[5]["diagnostic_root_exhaustive"] = False

        loss_rows = make_game("loss-game", 1, 0, 2)
        set_reference_action(loss_rows[0], "2")
        rows = win_rows + loss_rows

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(path, rows)
            report = MODULE.analyze_dataset(MODULE.load_audit(path), "baseline")

        self.assertEqual(
            report["clean_replay_audited_decisions"],
            {"total": 15, "wins": 13, "losses": 2},
        )
        self.assertEqual(
            report["clean_replay_audited_games_represented"],
            {"total": 2, "wins": 1, "losses": 1},
        )
        self.assertEqual(
            report["overall"]["agreement"]["candidate_vs_actual"]["count"],
            14,
        )
        self.assertEqual(
            report["overall"]["root_coverage"]["actual"]["exact"]["missing"]["count"],
            1,
        )
        self.assertEqual(
            report["overall"]["pressure"]["candidate_deadline_reached"]["count"],
            1,
        )
        self.assertEqual(
            report["breakdowns"]["by_phase"]["late_12_plus"]["decisions"],
            1,
        )
        divergences = report["first_divergence"]["by_game"]
        self.assertEqual([entry["game_id"] for entry in divergences], ["win-game", "loss-game"])
        self.assertEqual(divergences[0]["own_decision_index"], 1)
        self.assertEqual(divergences[0]["failure_bucket"], "rank4_only_matches_actual")

    def test_duplicate_json_keys_and_inconsistent_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            duplicate_path = pathlib.Path(temporary) / "duplicate.jsonl"
            duplicate_path.write_text(
                '{"schema_version":"fullturn-decision-audit-v2",'
                '"schema_version":"fullturn-decision-audit-v2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "duplicate JSON key"):
                MODULE.load_audit(duplicate_path)

            rows = make_game("game", 0, 0, 2)
            rows[1]["input_provenance"] = {"agent_id": "different"}
            provenance_path = pathlib.Path(temporary) / "provenance.jsonl"
            write_jsonl(provenance_path, rows)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "provenance differs"):
                MODULE.load_audit(provenance_path)

            rows = make_game("game", 0, 0, 2)
            rows[1]["candidate_final_visit_weight"] = 0.0
            configuration_path = pathlib.Path(temporary) / "configuration.jsonl"
            write_jsonl(configuration_path, rows)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "configuration differs"):
                MODULE.load_audit(configuration_path)

            rows = make_game("game", 0, 0, 1)
            rows[0]["candidate_nonroot_actions"] = 251
            cap_path = pathlib.Path(temporary) / "invalid-cap.jsonl"
            write_jsonl(cap_path, rows)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "action cap"):
                MODULE.load_audit(cap_path)

    def test_tsv_round_trip_and_duplicate_provenance_keys(self):
        row = make_row()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "audit.tsv"
            path.write_text(tsv_text([row]), encoding="utf-8")
            dataset = MODULE.load_audit(path)
            self.assertEqual(dataset["input_format"], "tsv")
            self.assertEqual(dataset["rows"][0]["candidate_elapsed_ms"], 10.0)

            bad_text = tsv_text([row]).replace(
                '{"agent_id":"6606663","source_binding_status":"asserted-not-api-verified"}',
                '{"agent_id":"6606663","agent_id":"6606663"}',
            )
            bad_path = pathlib.Path(temporary) / "bad.tsv"
            bad_path.write_text(bad_text, encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "duplicate JSON key"):
                MODULE.load_audit(bad_path)

    def test_derived_agreement_and_turn_sequence_are_validated(self):
        row = make_row()
        row["candidate_matches_actual"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bad-agreement.jsonl"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "candidate_matches_actual"):
                MODULE.load_audit(path)

            row = make_row(own_decision_index=0)
            row["turn_index"] = 2
            row["transcript_prefix"] = "0/0"
            path = pathlib.Path(temporary) / "bad-turn.jsonl"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "turn_index contradicts"):
                MODULE.load_audit(path)

    def test_comparison_aligns_decisions_and_reports_improvement(self):
        baseline_rows = make_game("game", 0, 0, 2)
        set_candidate_action(baseline_rows[0], "1")
        hypothesis_rows = copy.deepcopy(baseline_rows)
        for row in hypothesis_rows:
            row["candidate_final_visit_weight"] = 0.0
        set_candidate_action(hypothesis_rows[0], "0", retained_ordinal=0)

        with tempfile.TemporaryDirectory() as temporary:
            baseline_path = pathlib.Path(temporary) / "baseline.jsonl"
            hypothesis_path = pathlib.Path(temporary) / "hypothesis.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(hypothesis_path, hypothesis_rows)
            comparison = MODULE.compare_datasets(
                MODULE.load_audit(baseline_path),
                MODULE.load_audit(hypothesis_path),
                "visit-1",
                "visit-0",
            )["comparison"]

        metric = comparison["overall"]["candidate_matches_actual"]
        self.assertEqual(metric["baseline"]["count"], 1)
        self.assertEqual(metric["hypothesis"]["count"], 2)
        self.assertEqual(metric["improved"], 1)
        self.assertEqual(metric["regressed"], 0)
        self.assertEqual(
            comparison["configuration_changes"]["candidate_final_visit_weight"],
            {"baseline": 1.0, "hypothesis": 0.0},
        )
        self.assertEqual(comparison["candidate_action_changes"]["games"], 1)

    def test_comparison_rejects_provenance_or_state_mismatch(self):
        baseline = make_game("game", 0, 0, 1)
        hypothesis = copy.deepcopy(baseline)
        with tempfile.TemporaryDirectory() as temporary:
            baseline_path = pathlib.Path(temporary) / "baseline.jsonl"
            hypothesis_path = pathlib.Path(temporary) / "hypothesis.jsonl"
            write_jsonl(baseline_path, baseline)

            hypothesis[0]["input_provenance"]["agent_id"] = "different"
            write_jsonl(hypothesis_path, hypothesis)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "inconsistent provenance"):
                MODULE.compare_datasets(
                    MODULE.load_audit(baseline_path),
                    MODULE.load_audit(hypothesis_path),
                    "baseline",
                    "hypothesis",
                )

            hypothesis = copy.deepcopy(baseline)
            hypothesis[0]["state_id"] = "different-state"
            write_jsonl(hypothesis_path, hypothesis)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "does not align"):
                MODULE.compare_datasets(
                    MODULE.load_audit(baseline_path),
                    MODULE.load_audit(hypothesis_path),
                    "baseline",
                    "hypothesis",
                )


if __name__ == "__main__":
    unittest.main()
