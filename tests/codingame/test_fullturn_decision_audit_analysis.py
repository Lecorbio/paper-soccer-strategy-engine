#!/usr/bin/env python3

import copy
import hashlib
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
    row.update({field: None for field in MODULE.OPTIONAL_INTEGER_FIELDS})
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
            "initial_eval_best_action": "0",
            "initial_eval_best_score": 100,
            "initial_eval_best_retained_ordinal": 0,
            "actual_initial_eval_score": 100,
            "actual_initial_eval_rank": 1,
            "candidate_initial_eval_score": 100,
            "candidate_initial_eval_rank": 1,
            "reference_initial_eval_score": 100,
            "reference_initial_eval_rank": 1,
            "candidate_bfm_change_assessable": True,
            "candidate_bfm_changed_from_initial_best": False,
            "candidate_work_limit": 30_000,
            "candidate_tree_node_limit": 30_000,
            "candidate_max_actions": 250,
            "candidate_max_partial_paths": 50_000,
            "candidate_exploration": 1.5,
            "candidate_fpu": 0.5,
            "candidate_final_visit_weight": 1.0,
            "candidate_first_time_limit_ms": 800,
            "candidate_later_time_limit_ms": 165,
            "candidate_time_limit_ms": 800 if own_decision_index == 0 else 165,
            "candidate_elapsed_ms": 10.0,
            "candidate_work": 1_199,
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
            "codingame_clock_mode": False,
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
    row["candidate_initial_eval_score"] = 100 - 50 * retained_ordinal
    row["candidate_initial_eval_rank"] = retained_ordinal + 1
    row["candidate_bfm_change_assessable"] = retained_ordinal >= 0
    row["candidate_bfm_changed_from_initial_best"] = (
        retained_ordinal >= 0
        and retained_ordinal != row["initial_eval_best_retained_ordinal"]
    )


def set_reference_action(row, action, retained_ordinal=2):
    row["reference_action"] = action
    row["reference_matches_actual"] = action == row["actual_action"]
    row["candidate_matches_reference"] = action == row["candidate_action"]
    row["reference_action_retained_ordinal"] = retained_ordinal
    row["reference_boundary_retained_ordinal"] = retained_ordinal
    row["reference_initial_eval_score"] = 100 - 50 * retained_ordinal
    row["reference_initial_eval_rank"] = retained_ordinal + 1


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
            elif field in MODULE.OPTIONAL_INTEGER_FIELDS and value is None:
                values.append("")
            elif field in MODULE.BOOLEAN_FIELDS:
                values.append("1" if value else "0")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


SOURCE_SHA256 = "1" * 64
COLLECTOR_SHA256 = "2" * 64
EXCLUSION_SHA256 = "3" * 64
REPOSITORY_COMMIT = "4" * 40
AGENT_ID = 6606663
SUBMISSION_ID = 41119120
RUN_ID = "unit-arena"


def arena_record(
    rows,
    *,
    opponent_agent_id,
    opponent_name,
    frozen_rank,
):
    first = rows[0]
    final = rows[-1]
    turns = (
        tuple(final["transcript_prefix"].split("/"))
        if final["transcript_prefix"]
        else ()
    ) + (final["actual_action"],)
    game_id = int(first["game_id"])
    candidate_player = first["candidate_player"]
    winner = first["winner"]
    valid_turns = [
        {"action": action, "player_id": index % 2}
        for index, action in enumerate(turns)
    ]
    return {
        "game_id": game_id,
        "schema": MODULE.ARENA_GAME_SCHEMA_VERSION,
        "purpose": MODULE.ARENA_PURPOSE,
        "source_sha256": SOURCE_SHA256,
        "status": "accepted",
        "leaderboard_frozen_at_utc": "2026-08-11T00:00:00Z",
        "focus": {
            "agent_id": AGENT_ID,
            "submission_id": SUBMISSION_ID,
            "player_id": candidate_player,
            "color": f"player-{candidate_player}",
            "result": "win" if candidate_player == winner else "loss",
        },
        "opponent": {
            "agent_id": opponent_agent_id,
            "name": opponent_name,
            "player_id": 1 - candidate_player,
            "frozen_rank": frozen_rank,
        },
        "operational": {
            "classification": "clean",
            "focus_status": "ok",
            "opponent_status": "ok",
        },
        "outcome": {
            "winner_player_id": winner,
            "winner_agent_id": (
                AGENT_ID if winner == candidate_player else opponent_agent_id
            ),
        },
        "replay": {
            "valid_transcript": "/".join(turns),
            "observed_transcript": "/".join(turns),
            "valid_turns": valid_turns,
            "observed_turns": valid_turns,
            "agents": [
                {
                    "player_id": player_id,
                    "agent_id": (
                        AGENT_ID
                        if player_id == candidate_player
                        else opponent_agent_id
                    ),
                }
                for player_id in (0, 1)
            ],
            "rules_validation": {
                "status": "terminal-valid",
                "terminal_winner_player_id": winner,
                "valid_turn_count": len(turns),
                "valid_turns": valid_turns,
            },
        },
    }


def arena_manifest(game_specs):
    stored = []
    for rows, opponent_agent_id, opponent_name, frozen_rank in game_specs:
        record = arena_record(
            rows,
            opponent_agent_id=opponent_agent_id,
            opponent_name=opponent_name,
            frozen_rank=frozen_rank,
        )
        record_bytes = MODULE._canonical_json_bytes(record)
        record_hash = hashlib.sha256(record_bytes).hexdigest()
        stored.append(
            {
                "record": record,
                "record_path": f"diagnostics/game_records/{record['game_id']}/{record_hash}.json",
                "record_sha256": record_hash,
            }
        )
    return {
        "schema": MODULE.ARENA_BATCH_SCHEMA_VERSION,
        "purpose": MODULE.ARENA_PURPOSE,
        "collector_sha256": COLLECTOR_SHA256,
        "run_id": RUN_ID,
        "binding": {
            "schema": MODULE.ARENA_BINDING_SCHEMA_VERSION,
            "purpose": MODULE.ARENA_PURPOSE,
            "collector_sha256": COLLECTOR_SHA256,
            "run_id": RUN_ID,
            "agent_id": AGENT_ID,
            "asserted_submission_id": SUBMISSION_ID,
            "repository_commit": REPOSITORY_COMMIT,
            "source": {
                "sha256": SOURCE_SHA256,
                "encoding": "utf-8",
                "bytes": 10,
                "characters": 10,
                "archived_path": f"diagnostics/source_payloads/{SOURCE_SHA256}.source",
                "input_path": "submissions/codingame/submitted.cpp",
            },
        },
        "exclusion_registry": {
            "path": f"diagnostics/exclusions/{EXCLUSION_SHA256}.json",
            "sha256": EXCLUSION_SHA256,
        },
        "leaderboard_snapshot": {
            "frozen_at_utc": "2026-08-11T00:00:00Z",
            "normalized_sha256": "5" * 64,
            "raw_sha256": "6" * 64,
        },
        "window_snapshot": {
            "normalized_sha256": "7" * 64,
            "raw_sha256": "8" * 64,
        },
        "games": stored,
        "coverage": {
            "accepted_games": len(stored),
            "battle_window_games": len(stored),
            "clean_rule_terminal_games": len(stored),
            "expected_games": len(stored),
            "focus_operational_failures": 0,
            "full_window_accounted": True,
            "opponent_operational_failures": 0,
            "status_counts": {"accepted": len(stored)},
        },
    }


def write_manifest(directory, payload, filename=None):
    raw = MODULE._canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path = pathlib.Path(directory) / (filename or f"{digest}.json")
    path.write_bytes(raw)
    return path, digest


def bind_rows_to_manifest(rows, manifest_hash):
    provenance = {
        "agent_id": str(AGENT_ID),
        "arena_manifest_sha256": manifest_hash,
        "asserted_source_sha256": SOURCE_SHA256,
        "asserted_submission_id": str(SUBMISSION_ID),
        "collector_sha256": COLLECTOR_SHA256,
        "exclusion_registry_sha256": EXCLUSION_SHA256,
        "repository_commit": REPOSITORY_COMMIT,
        "run_id": RUN_ID,
        "source_binding_status": MODULE.ARENA_SOURCE_BINDING_STATUS,
    }
    for row in rows:
        row["input_provenance"] = copy.deepcopy(provenance)


class FullTurnDecisionAuditAnalysisTest(unittest.TestCase):
    def test_reports_agreement_pressure_breakdowns_and_first_divergence(self):
        win_rows = make_game("win-game", 0, 0, 13)
        set_candidate_action(win_rows[1], "1")
        win_rows[4]["candidate_deadline_reached"] = True
        win_rows[4]["candidate_budget_exhausted"] = True
        win_rows[4]["candidate_generator_truncations"] = 3
        win_rows[5]["actual_action_retained_ordinal"] = -1
        win_rows[5]["actual_boundary_retained_ordinal"] = -1
        win_rows[5]["actual_retained_tactical_class"] = "not-retained"
        win_rows[5]["actual_initial_eval_score"] = None
        win_rows[5]["actual_initial_eval_rank"] = -1
        win_rows[5]["actual_action"] = "1"
        win_rows[5]["candidate_matches_actual"] = False
        win_rows[5]["reference_matches_actual"] = False
        win_rows[5]["reconstructed_deployed_matches_actual"] = False
        for later in win_rows[6:]:
            prefix = later["transcript_prefix"].split("/")
            prefix[win_rows[5]["turn_index"]] = "1"
            later["transcript_prefix"] = "/".join(prefix)
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
            13,
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
            report["overall"]["initial_evaluation"]
            ["candidate_bfm_changed_from_initial_best"]["count"],
            1,
        )
        self.assertEqual(
            report["overall"]["initial_evaluation_buckets"]
            ["bfm_overrode_actual_initial_best"],
            1,
        )
        self.assertEqual(
            report["breakdowns"]["by_phase"]["late_12_plus"]["decisions"],
            1,
        )
        self.assertEqual(
            report["breakdowns"]["by_clock_phase"]["first_decision"]["decisions"],
            2,
        )
        self.assertEqual(
            report["breakdowns"]["by_clock_phase"]["later_decisions"]["decisions"],
            13,
        )
        divergences = report["first_divergence"]["by_game"]
        self.assertEqual([entry["game_id"] for entry in divergences], ["win-game", "loss-game"])
        self.assertEqual(divergences[0]["own_decision_index"], 1)
        self.assertEqual(divergences[0]["failure_bucket"], "rank4_only_matches_actual")

    def test_duplicate_json_keys_and_inconsistent_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            duplicate_path = pathlib.Path(temporary) / "duplicate.jsonl"
            duplicate_path.write_text(
                '{"schema_version":"fullturn-decision-audit-v3",'
                '"schema_version":"fullturn-decision-audit-v3"}\n',
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

            missing = make_row()
            missing["actual_action"] = "1"
            missing["candidate_matches_actual"] = False
            missing["reference_matches_actual"] = False
            missing["reconstructed_deployed_matches_actual"] = False
            missing["actual_action_retained_ordinal"] = -1
            missing["actual_boundary_retained_ordinal"] = -1
            missing["actual_retained_tactical_class"] = "not-retained"
            missing["actual_initial_eval_score"] = None
            missing["actual_initial_eval_rank"] = -1
            path.write_text(tsv_text([missing]), encoding="utf-8")
            dataset = MODULE.load_audit(path)
            self.assertIsNone(dataset["rows"][0]["actual_initial_eval_score"])
            self.assertEqual(dataset["rows"][0]["actual_initial_eval_rank"], -1)

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

    def test_initial_evaluation_fields_separate_misranking_from_bfm_override(self):
        corrected = make_row()
        corrected.update(
            {
                "initial_eval_best_action": "1",
                "initial_eval_best_score": 200,
                "initial_eval_best_retained_ordinal": 1,
                "actual_initial_eval_score": 100,
                "actual_initial_eval_rank": 2,
                "candidate_initial_eval_score": 100,
                "candidate_initial_eval_rank": 2,
                "reference_initial_eval_score": 100,
                "reference_initial_eval_rank": 2,
                "candidate_bfm_changed_from_initial_best": True,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "corrected.jsonl"
            write_jsonl(path, [corrected])
            report = MODULE.analyze_dataset(MODULE.load_audit(path), "corrected")
            self.assertEqual(
                report["overall"]["initial_evaluation_buckets"],
                {"bfm_corrected_initial_misranking": 1},
            )
            self.assertEqual(
                report["overall"]["initial_evaluation"]
                ["bfm_changed_to_actual_boundary"]["count"],
                1,
            )

            inconsistent = copy.deepcopy(corrected)
            inconsistent["candidate_bfm_changed_from_initial_best"] = False
            write_jsonl(path, [inconsistent])
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "BFM initial-best change"
            ):
                MODULE.load_audit(path)

            inconsistent = copy.deepcopy(corrected)
            inconsistent["actual_initial_eval_score"] = None
            write_jsonl(path, [inconsistent])
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "contradict retention"
            ):
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
        first_clock = comparison["breakdowns"]["by_clock_phase"][
            "first_decision"
        ]["candidate_matches_actual"]
        later_clock = comparison["breakdowns"]["by_clock_phase"][
            "later_decisions"
        ]["candidate_matches_actual"]
        self.assertEqual(first_clock["baseline"]["count"], 0)
        self.assertEqual(first_clock["hypothesis"]["count"], 1)
        self.assertEqual(later_clock["baseline"]["count"], 1)
        self.assertEqual(later_clock["hypothesis"]["count"], 1)

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

    def test_rejects_impossible_eval_retention_and_action_ordinal_tuples(self):
        absent = make_row()
        absent.update(
            {
                "actual_action": "1",
                "candidate_matches_actual": False,
                "reference_matches_actual": False,
                "reconstructed_deployed_matches_actual": False,
                "actual_action_retained_ordinal": -1,
                "actual_boundary_retained_ordinal": -1,
                "actual_retained_tactical_class": "not-retained",
                "actual_initial_eval_score": 0,
                "actual_initial_eval_rank": 0,
            }
        )
        with self.assertRaisesRegex(MODULE.AuditAnalysisError, "absent boundary"):
            MODULE._validate_row(absent, 1)

        retained = make_row()
        retained["actual_initial_eval_score"] = None
        retained["actual_initial_eval_rank"] = -1
        with self.assertRaisesRegex(MODULE.AuditAnalysisError, "contradict retention"):
            MODULE._validate_row(retained, 1)

        unequal = make_row()
        unequal["candidate_action_retained_ordinal"] = 1
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "exact retention contradicts boundary"
        ):
            MODULE._validate_row(unequal, 1)

        reused_exact = make_row()
        reused_exact.update(
            {
                "actual_action": "1",
                "actual_action_retained_ordinal": 1,
                "actual_boundary_retained_ordinal": 1,
                "actual_initial_eval_score": 50,
                "actual_initial_eval_rank": 2,
                "candidate_action": "2",
                "candidate_action_retained_ordinal": 1,
                "candidate_boundary_retained_ordinal": 1,
                "candidate_initial_eval_score": 50,
                "candidate_initial_eval_rank": 2,
                "candidate_matches_actual": False,
                "candidate_matches_reference": False,
                "reference_matches_actual": False,
                "reconstructed_deployed_action": "2",
                "reconstructed_deployed_matches_actual": False,
                "candidate_bfm_changed_from_initial_best": True,
            }
        )
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "one exact ordinal identifies different actions"
        ):
            MODULE._validate_row(reused_exact, 1)

        duplicate_rank = make_row()
        duplicate_rank.update(
            {
                "actual_action": "1",
                "actual_action_retained_ordinal": 1,
                "actual_boundary_retained_ordinal": 1,
                "actual_initial_eval_score": 50,
                "actual_initial_eval_rank": 2,
                "reference_action": "2",
                "reference_action_retained_ordinal": 2,
                "reference_boundary_retained_ordinal": 2,
                "reference_initial_eval_score": 40,
                "reference_initial_eval_rank": 2,
                "candidate_matches_actual": False,
                "reference_matches_actual": False,
                "candidate_matches_reference": False,
                "reconstructed_deployed_matches_actual": False,
            }
        )
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "different retained boundaries share one rank"
        ):
            MODULE._validate_row(duplicate_rank, 1)

        impossible_initial = make_row()
        impossible_initial.update(
            {
                "initial_eval_best_score": 500_000,
                "actual_initial_eval_score": 500_000,
                "candidate_initial_eval_score": 500_000,
                "reference_initial_eval_score": 500_000,
            }
        )
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "impossible initial-evaluation score"
        ):
            MODULE._validate_row(impossible_initial, 1)

        safe_with_proof_score = make_row()
        safe_with_proof_score.update(
            {
                "initial_eval_best_score": 999_999,
                "actual_initial_eval_score": 999_999,
                "candidate_initial_eval_score": 999_999,
                "reference_initial_eval_score": 999_999,
            }
        )
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "contradicts safe-handoff class"
        ):
            MODULE._validate_row(safe_with_proof_score, 1)

        proof_with_heuristic_score = make_row()
        proof_with_heuristic_score.update(
            {
                "initial_eval_best_score": 42,
                "actual_initial_eval_score": 42,
                "candidate_initial_eval_score": 42,
                "reference_initial_eval_score": 42,
                "actual_retained_tactical_class": "immediate-win",
                "reference_retained_tactical_class": "immediate-win",
            }
        )
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "contradicts tactical proof class"
        ):
            MODULE._validate_row(proof_with_heuristic_score, 1)

    def test_rejects_native_config_bound_and_counter_violations(self):
        invalid_configurations = (
            ("candidate_work_limit", 0, "candidate work limit"),
            ("candidate_work_limit", 3_000_001, "candidate work limit"),
            ("candidate_tree_node_limit", 1, "candidate tree limit"),
            ("candidate_tree_node_limit", 120_001, "candidate tree limit"),
            ("candidate_max_partial_paths", 0, "partial-path limit"),
            ("candidate_max_partial_paths", 50_001, "partial-path limit"),
            ("candidate_exploration", -0.1, "bounded configuration"),
            ("candidate_fpu", float("inf"), "must be finite"),
            ("candidate_final_visit_weight", -0.1, "bounded configuration"),
            ("reference_nodes_limit", 0, "reference node limit"),
            ("reference_nodes_limit", 3_000_001, "reference node limit"),
            ("reference_depth_limit", 0, "reference depth limit"),
            ("reference_depth_limit", 33, "reference depth limit"),
        )
        for field, value, message in invalid_configurations:
            row = make_row()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                MODULE.AuditAnalysisError, message
            ):
                MODULE._validate_row(row, 1)

        invalid_counters = (
            ("candidate_work_limit", 1_000, "candidate work exceeds"),
            ("candidate_work", 1_200, "work counters are inconsistent"),
            ("candidate_tree_node_limit", 499, "tree nodes exceed"),
            ("candidate_expansions", 501, "expansions exceed"),
            ("diagnostic_root_partial_paths", 50_001, "partial paths exceed"),
            ("reference_nodes", 30_001, "reference nodes exceed"),
            ("reference_attempted_turn_depth", 9, "depth counters are inconsistent"),
        )
        for field, value, message in invalid_counters:
            row = make_row()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                MODULE.AuditAnalysisError, message
            ):
                MODULE._validate_row(row, 1)

        inconsistent_codingame = make_row()
        inconsistent_codingame["codingame_clock_mode"] = True
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "CodinGame clock mode configuration"
        ):
            MODULE._validate_row(inconsistent_codingame, 1)

        valid_codingame = make_row()
        valid_codingame.update(
            {
                "codingame_clock_mode": True,
                "candidate_work_limit": 3_000_000,
                "reference_nodes_limit": 3_000_000,
            }
        )
        MODULE._validate_row(valid_codingame, 1)

        out_of_range_score = make_row()
        out_of_range_score["candidate_root_score"] = 1_000_001
        with self.assertRaisesRegex(MODULE.AuditAnalysisError, "score bounds"):
            MODULE._validate_row(out_of_range_score, 1)

        contradictory_pressure = make_row()
        contradictory_pressure["candidate_deadline_reached"] = True
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "pressure flags contradict"
        ):
            MODULE._validate_row(contradictory_pressure, 1)

        contradictory_root = make_row()
        contradictory_root["diagnostic_root_truncations"] = 1
        with self.assertRaisesRegex(
            MODULE.AuditAnalysisError, "exhaustiveness contradicts truncation"
        ):
            MODULE._validate_row(contradictory_root, 1)

    def test_arena_manifest_exact_join_adds_rank_and_named_opponent_buckets(self):
        game_specs = []
        rows = []
        opponents = (
            ("101", 7001, "Alpha", 3),
            ("102", 7001, "Alpha", 3),
            ("103", 7001, "Alpha", 3),
            ("104", 7002, "Beta", 15),
            ("105", 7003, "Gamma", 30),
            ("106", 7004, "Unranked", None),
        )
        for game_id, opponent_id, name, rank in opponents:
            game_rows = make_game(game_id, 0, 1 if game_id == "104" else 0, 1)
            rows.extend(game_rows)
            game_specs.append((game_rows, opponent_id, name, rank))
        zero_decision_rows = make_game("107", 1, 0, 1)
        game_specs.append((zero_decision_rows, 7005, "NoMove", 12))

        with tempfile.TemporaryDirectory() as temporary:
            payload = arena_manifest(game_specs)
            zero_record = payload["games"][-1]["record"]
            zero_record["replay"]["valid_transcript"] = "0"
            zero_record["replay"]["observed_transcript"] = "0"
            zero_record["replay"]["valid_turns"] = [
                {"action": "0", "player_id": 0}
            ]
            zero_record["replay"]["observed_turns"] = [
                {"action": "0", "player_id": 0}
            ]
            zero_record["replay"]["rules_validation"].update(
                {
                    "valid_turn_count": 1,
                    "valid_turns": [{"action": "0", "player_id": 0}],
                }
            )
            zero_bytes = MODULE._canonical_json_bytes(zero_record)
            zero_hash = hashlib.sha256(zero_bytes).hexdigest()
            payload["games"][-1]["record_sha256"] = zero_hash
            payload["games"][-1]["record_path"] = (
                f"diagnostics/game_records/107/{zero_hash}.json"
            )
            manifest_path, manifest_hash = write_manifest(
                temporary, payload
            )
            bind_rows_to_manifest(rows, manifest_hash)
            audit_path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(audit_path, rows)
            joined = MODULE.join_arena_manifest(
                MODULE.load_audit(audit_path),
                MODULE.load_arena_manifest(manifest_path),
            )
            report = MODULE.analyze_dataset(joined, "arena")

        self.assertTrue(
            report["arena_manifest_join"]["source"]["content_addressed_filename"]
        )
        self.assertEqual(report["arena_manifest_join"]["clean_manifest_games"], 7)
        self.assertEqual(report["arena_manifest_join"]["audited_manifest_games"], 6)
        self.assertEqual(
            report["arena_manifest_join"]["clean_games_without_candidate_decision"],
            ["107"],
        )
        self.assertEqual(
            report["breakdowns"]["by_clock_phase"]["later_decisions"]["decisions"],
            0,
        )
        rank = report["breakdowns"]["frozen_opponent_rank_cohorts"]
        self.assertEqual(rank["rank_1_5"]["games"], 3)
        self.assertEqual(rank["rank_1_10"]["games"], 3)
        self.assertEqual(rank["rank_1_20"]["games"], 4)
        self.assertEqual(
            rank["rank_1_20"]["game_results"],
            {"total": 4, "wins": 3, "losses": 1},
        )
        self.assertEqual(rank["rank_21_plus"]["games"], 1)
        self.assertEqual(rank["unranked"]["games"], 1)
        named = report["breakdowns"]["named_opponents"]
        self.assertEqual(named["minimum_games"], 3)
        self.assertEqual(
            [(entry["name"], entry["games"]) for entry in named["buckets"]],
            [("Alpha", 3)],
        )
        self.assertEqual(
            named["buckets"][0]["game_results"],
            {"total": 3, "wins": 3, "losses": 0},
        )

    def test_arena_manifest_join_checks_every_provenance_field_and_exact_coverage(self):
        first = make_game("201", 0, 0, 1)
        second = make_game("202", 1, 0, 1)
        specs = [
            (first, 7101, "First", 5),
            (second, 7102, "Second", 25),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest_hash = write_manifest(
                temporary, arena_manifest(specs)
            )
            loaded_manifest = MODULE.load_arena_manifest(manifest_path)

            all_rows = copy.deepcopy(first + second)
            bind_rows_to_manifest(all_rows, manifest_hash)
            audit_path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(audit_path, all_rows)
            joined = MODULE.join_arena_manifest(
                MODULE.load_audit(audit_path), loaded_manifest
            )
            self.assertEqual(len(joined["rows"]), 2)

            for field in MODULE.ARENA_PROVENANCE_FIELDS:
                mismatched = copy.deepcopy(all_rows)
                for row in mismatched:
                    row["input_provenance"][field] = "mismatch"
                write_jsonl(audit_path, mismatched)
                with self.subTest(field=field), self.assertRaisesRegex(
                    MODULE.AuditAnalysisError, f"arena provenance field {field}"
                ):
                    MODULE.join_arena_manifest(
                        MODULE.load_audit(audit_path), loaded_manifest
                    )

            subset = copy.deepcopy(first)
            bind_rows_to_manifest(subset, manifest_hash)
            write_jsonl(audit_path, subset)
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "audit-game coverage differs"
            ):
                MODULE.join_arena_manifest(
                    MODULE.load_audit(audit_path), loaded_manifest
                )

    def test_arena_manifest_rejects_duplicate_records_bad_hash_name_and_context(self):
        rows = make_game("301", 0, 0, 1)
        payload = arena_manifest([(rows, 7201, "Opponent", 9)])
        with tempfile.TemporaryDirectory() as temporary:
            wrong_name = "0" * 64 + ".json"
            wrong_path, _ = write_manifest(temporary, payload, wrong_name)
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "filename hash"
            ):
                MODULE.load_arena_manifest(wrong_path)

            duplicated = copy.deepcopy(payload)
            duplicated["games"].append(copy.deepcopy(duplicated["games"][0]))
            duplicated["coverage"].update(
                {
                    "accepted_games": 2,
                    "battle_window_games": 2,
                    "clean_rule_terminal_games": 2,
                    "expected_games": 2,
                    "status_counts": {"accepted": 2},
                }
            )
            duplicate_path, _ = write_manifest(temporary, duplicated)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "repeats game 301"):
                MODULE.load_arena_manifest(duplicate_path)

            incomplete = copy.deepcopy(payload)
            incomplete["coverage"]["expected_games"] = 2
            incomplete["coverage"]["full_window_accounted"] = False
            incomplete_path, _ = write_manifest(temporary, incomplete)
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "battle window is not fully accounted"
            ):
                MODULE.load_arena_manifest(incomplete_path)

            manifest_path, manifest_hash = write_manifest(temporary, payload)
            changed_context = copy.deepcopy(rows)
            bind_rows_to_manifest(changed_context, manifest_hash)
            changed_context[0].update(
                {
                    "candidate_player": 1,
                    "color": "player_two",
                    "turn_index": 1,
                    "transcript_prefix": "0",
                    "result": "loss",
                }
            )
            audit_path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(audit_path, changed_context)
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "contradicts manifest replay context"
            ):
                MODULE.join_arena_manifest(
                    MODULE.load_audit(audit_path),
                    MODULE.load_arena_manifest(manifest_path),
                )

    def test_arena_manifest_join_stratifies_aligned_comparison(self):
        baseline_rows = make_game("401", 0, 0, 1)
        set_candidate_action(baseline_rows[0], "1")
        hypothesis_rows = copy.deepcopy(baseline_rows)
        hypothesis_rows[0]["candidate_final_visit_weight"] = 0.0
        set_candidate_action(hypothesis_rows[0], "0", retained_ordinal=0)
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest_hash = write_manifest(
                temporary,
                arena_manifest([(baseline_rows, 7301, "TopTen", 9)]),
            )
            bind_rows_to_manifest(baseline_rows, manifest_hash)
            bind_rows_to_manifest(hypothesis_rows, manifest_hash)
            baseline_path = pathlib.Path(temporary) / "baseline.jsonl"
            hypothesis_path = pathlib.Path(temporary) / "hypothesis.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(hypothesis_path, hypothesis_rows)
            loaded_manifest = MODULE.load_arena_manifest(manifest_path)
            comparison = MODULE.compare_datasets(
                MODULE.join_arena_manifest(
                    MODULE.load_audit(baseline_path), loaded_manifest
                ),
                MODULE.join_arena_manifest(
                    MODULE.load_audit(hypothesis_path), loaded_manifest
                ),
                "baseline",
                "hypothesis",
            )["comparison"]

        metric = comparison["breakdowns"]["frozen_opponent_rank_cohorts"][
            "rank_1_10"
        ]["metrics"]["candidate_matches_actual"]
        self.assertEqual(metric["baseline"]["count"], 0)
        self.assertEqual(metric["hypothesis"]["count"], 1)
        self.assertEqual(metric["improved"], 1)
        self.assertEqual(
            comparison["breakdowns"]["frozen_opponent_rank_cohorts"]
            ["rank_1_10"]["game_results"],
            {"total": 1, "wins": 1, "losses": 0},
        )

if __name__ == "__main__":
    unittest.main()
