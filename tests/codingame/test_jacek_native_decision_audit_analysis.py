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
    / "jacek_native_bfm"
    / "analyze_decision_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "jacek_native_decision_audit_analysis", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


AGENT_ID = 6_600_001
SUBMISSION_ID = 41_100_001
SOURCE_SHA256 = "1" * 64
COLLECTOR_SHA256 = "2" * 64
EXCLUSION_SHA256 = "3" * 64
REPOSITORY_COMMIT = "4" * 40
RUN_ID = "native-audit-test"


def make_row(
    game_id="101",
    *,
    candidate_player=0,
    winner=0,
    own_decision_index=0,
    transcript_prefix="",
):
    row = {field: "" for field in MODULE.STRING_FIELDS}
    row.update({field: 0 for field in MODULE.INTEGER_FIELDS})
    row.update({field: None for field in MODULE.OPTIONAL_INTEGER_FIELDS})
    row.update({field: 0.0 for field in MODULE.FLOAT_FIELDS})
    row.update({field: None for field in MODULE.OPTIONAL_FLOAT_FIELDS})
    row.update({field: False for field in MODULE.BOOLEAN_FIELDS})
    turn_index = 2 * own_decision_index + candidate_player
    row.update(
        {
            "schema_version": MODULE.AUDIT_SCHEMA_VERSION,
            "input_provenance": {"run_id": RUN_ID},
            "game_id": game_id,
            "state_id": f"fnv1a64:{int(game_id) * 100 + own_decision_index:016x}",
            "transcript_prefix": transcript_prefix,
            "turn_index": turn_index,
            "own_decision_index": own_decision_index,
            "candidate_player": candidate_player,
            "color": "player_one" if candidate_player == 0 else "player_two",
            "winner": winner,
            "result": "win" if candidate_player == winner else "loss",
            "classification": "match",
            "classification_reason": "the native search reproduced the observed encoding",
            "audit_mode": "fixed-work",
            "model_sha256": "5" * 64,
            "packed_weights_sha256": "6" * 64,
            "fixed_work_limit": 64,
            "max_actions": 16,
            "max_partial_paths": 64,
            "max_expansions": 1_000,
            "exploration": 0.95,
            "first_play_urgency": 0.5,
            "first_time_limit_ms": 0,
            "later_time_limit_ms": 0,
            "time_limit_ms": 0,
            "chosen_value": 0.1,
            "search_solved": False,
            "search_solved_winner": None,
            "search_elapsed_ms": 1.0,
            "search_root_actions": 4,
            "search_tree_nodes": 32,
            "search_expansions": 3,
            "search_generated_children": 31,
            "search_child_evaluations": 20,
            "search_tactical_child_values": 11,
            "search_generator_partial_paths": 40,
            "search_tactical_proof_paths": 2,
            "search_completed_actions": 20,
            "search_duplicate_boundaries": 1,
            "search_fifo_extractions": 4,
            "search_lifo_extractions": 36,
            "search_tactical_actions": 2,
            "search_tactical_classes_found": 1,
            "search_tactical_proof_truncations": 0,
            "search_generator_truncations": 0,
            "search_deadline_reached": False,
            "search_tree_cap_reached": False,
            "search_expansion_cap_reached": False,
            "diagnostic_root_actions": 4,
            "diagnostic_root_partial_paths": 8,
            "diagnostic_root_tactical_proof_paths": 1,
            "diagnostic_root_completed_actions": 5,
            "diagnostic_root_duplicate_boundaries": 1,
            "diagnostic_root_fifo_extractions": 1,
            "diagnostic_root_lifo_extractions": 7,
            "diagnostic_root_tactical_actions": 1,
            "diagnostic_root_tactical_classes_found": 1,
            "diagnostic_root_tactical_proof_truncated": False,
            "diagnostic_root_truncations": 0,
            "diagnostic_root_maximum_deque_size": 4,
            "diagnostic_root_deadline_reached": False,
            "diagnostic_root_exhaustive": True,
            "initial_best_action": "0",
            "initial_best_value": 0.1,
        }
    )
    set_diagnostic(row, "actual", "0", 0, 1)
    set_diagnostic(row, "chosen", "0", 0, 1)
    return row


def set_diagnostic(row, prefix, action, ordinal, rank):
    row[f"{prefix}_action"] = action
    row[f"{prefix}_exact_retained_ordinal"] = ordinal
    row[f"{prefix}_boundary_retained_ordinal"] = ordinal
    row[f"{prefix}_retained_action"] = action
    row[f"{prefix}_tactical_class"] = "safe-handoff"
    row[f"{prefix}_initial_neural_value"] = 0.1 - ordinal * 0.01
    row[f"{prefix}_initial_action_value"] = 0.1 - ordinal * 0.01
    row[f"{prefix}_initial_rank"] = rank
    row[f"{prefix}_final_backed_value"] = 0.1 - ordinal * 0.01
    row[f"{prefix}_final_visits"] = 2
    row[f"{prefix}_final_selection_visits"] = 1
    row[f"{prefix}_final_tactical_class"] = "safe-handoff"


def omit_actual(row, action="7"):
    row["actual_action"] = action
    row["actual_exact_retained_ordinal"] = -1
    row["actual_boundary_retained_ordinal"] = -1
    row["actual_retained_action"] = ""
    row["actual_tactical_class"] = "not-retained"
    row["actual_initial_neural_value"] = None
    row["actual_initial_action_value"] = None
    row["actual_initial_rank"] = -1
    row["actual_final_backed_value"] = None
    row["actual_final_visits"] = None
    row["actual_final_selection_visits"] = None
    row["actual_final_tactical_class"] = "not-searched"
    row["classification"] = "generator-omission"
    row["classification_reason"] = "the deterministic generator omitted it"


def omit_diagnostic(row, prefix, action):
    row[f"{prefix}_action"] = action
    row[f"{prefix}_exact_retained_ordinal"] = -1
    row[f"{prefix}_boundary_retained_ordinal"] = -1
    row[f"{prefix}_retained_action"] = ""
    row[f"{prefix}_tactical_class"] = "not-retained"
    row[f"{prefix}_initial_neural_value"] = None
    row[f"{prefix}_initial_action_value"] = None
    row[f"{prefix}_initial_rank"] = -1
    row[f"{prefix}_final_backed_value"] = None
    row[f"{prefix}_final_visits"] = None
    row[f"{prefix}_final_selection_visits"] = None
    row[f"{prefix}_final_tactical_class"] = "not-searched"


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def arena_record(rows, opponent_agent_id, opponent_name, frozen_rank, turns):
    first = rows[0]
    candidate_player = first["candidate_player"]
    winner = first["winner"]
    valid_turns = [
        {"action": action, "player_id": index % 2}
        for index, action in enumerate(turns)
    ]
    return {
        "schema": MODULE.ARENA_GAME_SCHEMA_VERSION,
        "purpose": MODULE.ARENA_PURPOSE,
        "source_sha256": SOURCE_SHA256,
        "status": "accepted",
        "game_id": int(first["game_id"]),
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
            "agents": [
                {
                    "agent_id": (
                        AGENT_ID if player_id == candidate_player else opponent_agent_id
                    ),
                    "player_id": player_id,
                }
                for player_id in (0, 1)
            ],
            "valid_transcript": "/".join(turns),
            "observed_transcript": "/".join(turns),
            "valid_turns": valid_turns,
            "observed_turns": valid_turns,
            "rules_validation": {
                "status": "terminal-valid",
                "terminal_winner_player_id": winner,
                "valid_turn_count": len(turns),
                "valid_turns": valid_turns,
            },
        },
    }


def write_manifest(directory, specifications):
    stored = []
    for rows, opponent_id, name, rank, turns in specifications:
        record = arena_record(rows, opponent_id, name, rank, turns)
        record_hash = hashlib.sha256(MODULE._canonical_json_bytes(record)).hexdigest()
        stored.append(
            {
                "record": record,
                "record_sha256": record_hash,
                "record_path": f"archive/game_records/{record_hash}.json",
            }
        )
    manifest = {
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
                "archived_path": f"archive/sources/{SOURCE_SHA256}.source",
                "input_path": "submissions/codingame/submission.cpp",
                "encoding": "utf-8",
                "bytes": 10,
                "characters": 10,
            },
        },
        "exclusion_registry": {
            "sha256": EXCLUSION_SHA256,
            "path": f"archive/exclusions/{EXCLUSION_SHA256}.json",
        },
        "games": stored,
        "leaderboard_snapshot": {
            "focus": None,
            "frozen_at_utc": "2026-08-11T00:00:00Z",
            "normalized_sha256": "7" * 64,
            "raw_sha256": "8" * 64,
        },
        "window_snapshot": {
            "normalized_sha256": "9" * 64,
            "raw_sha256": "a" * 64,
        },
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
    raw = MODULE._canonical_json_bytes(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    path = pathlib.Path(directory) / f"{digest}.json"
    path.write_bytes(raw)
    provenance = {
        "agent_id": str(AGENT_ID),
        "arena_manifest_sha256": digest,
        "asserted_source_sha256": SOURCE_SHA256,
        "asserted_submission_id": str(SUBMISSION_ID),
        "collector_sha256": COLLECTOR_SHA256,
        "exclusion_registry_sha256": EXCLUSION_SHA256,
        "repository_commit": REPOSITORY_COMMIT,
        "run_id": RUN_ID,
        "source_binding_status": MODULE.ARENA_SOURCE_BINDING_STATUS,
    }
    return path, provenance


class JacekNativeDecisionAuditAnalysisTest(unittest.TestCase):
    def test_aggregates_required_slices_and_first_divergence(self):
        first = make_row("101")
        second = make_row(
            "101", own_decision_index=1, transcript_prefix="0/7"
        )
        set_diagnostic(second, "actual", "1", 1, 2)
        second["classification"] = "initial-evaluator-ordering"
        second["classification_reason"] = "initial order preserved"
        loss = make_row(
            "102",
            candidate_player=1,
            winner=0,
            transcript_prefix="7",
        )
        omit_actual(loss, "6")
        loss["diagnostic_root_exhaustive"] = False
        loss["diagnostic_root_truncations"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(path, [first, second, loss])
            dataset = MODULE.load_audit(path)
            report = MODULE.analyze_dataset(dataset, "baseline")

        self.assertEqual(report["represented_games"], {"games": 2, "wins": 1, "losses": 1})
        self.assertEqual(report["overall"]["classifications"]["match"]["count"], 1)
        self.assertEqual(report["overall"]["actual"]["boundary_omitted"]["count"], 1)
        self.assertEqual(report["breakdowns"]["by_result"]["loss"]["decisions"], 1)
        self.assertEqual(report["breakdowns"]["by_player"]["player_1"]["decisions"], 1)
        self.assertEqual(report["breakdowns"]["by_own_decision_phase"]["first"]["decisions"], 2)
        self.assertEqual(report["first_divergence"]["games_with_divergence"], 2)
        self.assertEqual(report["first_divergence"]["by_game"][0]["own_decision_index"], 1)

    def test_schema_provenance_and_configuration_fail_closed(self):
        row = make_row()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bad.jsonl"
            missing = copy.deepcopy(row)
            missing.pop("search_tree_nodes")
            write_jsonl(path, [missing])
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "schema fields differ"):
                MODULE.load_audit(path)

            later = make_row(own_decision_index=1, transcript_prefix="0/7")
            later["model_sha256"] = "7" * 64
            write_jsonl(path, [row, later])
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "configuration differs"):
                MODULE.load_audit(path)

            later["model_sha256"] = row["model_sha256"]
            later["input_provenance"] = {"run_id": "different"}
            write_jsonl(path, [row, later])
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "provenance differs"):
                MODULE.load_audit(path)

            path.write_text('{"schema_version":"x","schema_version":"y"}\n')
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "duplicate JSON key"):
                MODULE.load_audit(path)

    def test_v1_bytes_remain_valid_and_v2_limit_is_strict(self):
        current = make_row()
        legacy = copy.deepcopy(current)
        legacy["schema_version"] = MODULE.LEGACY_AUDIT_SCHEMA_VERSION
        legacy.pop("max_own_decisions_per_game")
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "legacy.jsonl"
            write_jsonl(path, [legacy])
            original = path.read_bytes()
            original_sha = hashlib.sha256(original).hexdigest()
            dataset = MODULE.load_audit(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(dataset["source_sha256"], original_sha)
            self.assertEqual(
                dataset["audit_schema_version"],
                MODULE.LEGACY_AUDIT_SCHEMA_VERSION,
            )
            self.assertIsNone(
                dataset["configuration"]["max_own_decisions_per_game"]
            )

            missing = copy.deepcopy(current)
            missing.pop("max_own_decisions_per_game")
            write_jsonl(path, [missing])
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "schema fields differ"
            ):
                MODULE.load_audit(path)

            for invalid in (0, 1_025):
                bounded = copy.deepcopy(current)
                bounded["max_own_decisions_per_game"] = invalid
                write_jsonl(path, [bounded])
                with self.assertRaisesRegex(
                    MODULE.AuditAnalysisError, "outside.*1,1024"
                ):
                    MODULE.load_audit(path)

    def test_exhaustive_root_allows_independent_tactical_proof_cap(self):
        row = make_row()
        row["diagnostic_root_tactical_proof_paths"] = 64
        row["diagnostic_root_tactical_proof_truncated"] = True
        row["diagnostic_root_exhaustive"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "proof-cap.jsonl"
            write_jsonl(path, [row])
            dataset = MODULE.load_audit(path)
            report = MODULE.analyze_dataset(dataset, "proof-cap")

        pressure = report["overall"]["pressure"]
        self.assertEqual(pressure["diagnostic_root_proof_truncated"]["count"], 1)
        self.assertEqual(pressure["diagnostic_root_nonexhaustive"]["count"], 0)

        row["diagnostic_root_truncations"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "contradiction.jsonl"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "exhaustiveness contradicts"
            ):
                MODULE.load_audit(path)

    def test_capped_diagnostic_root_may_omit_matching_search_action(self):
        row = make_row()
        omit_diagnostic(row, "actual", "025247527605")
        omit_diagnostic(row, "chosen", "025247527605")
        row["classification"] = "match"
        row["diagnostic_root_exhaustive"] = False
        row["diagnostic_root_truncations"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "capped-match.jsonl"
            write_jsonl(path, [row])
            dataset = MODULE.load_audit(path)
        self.assertEqual(dataset["rows"][0]["chosen_boundary_retained_ordinal"], -1)

        row["diagnostic_root_exhaustive"] = True
        row["diagnostic_root_truncations"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "impossible-exhaustive.jsonl"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(
                MODULE.AuditAnalysisError, "exhaustive diagnostic root omits"
            ):
                MODULE.load_audit(path)

    def test_manifest_join_is_exact_and_game_weighted(self):
        loss_rows = [
            make_row("101", winner=1),
            make_row("101", winner=1, own_decision_index=1, transcript_prefix="0/7"),
        ]
        win_rows = [
            make_row("102", candidate_player=1, winner=1, transcript_prefix="7")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, provenance = write_manifest(
                temporary,
                [
                    (loss_rows, 9001, "Jacek", 2, ["0", "7", "0", "6"]),
                    (win_rows, 9001, "Jacek", 2, ["7", "0"]),
                ],
            )
            for row in loss_rows + win_rows:
                row["input_provenance"] = provenance
            audit_path = pathlib.Path(temporary) / "audit.jsonl"
            write_jsonl(audit_path, loss_rows + win_rows)
            dataset = MODULE.join_arena_manifest(
                MODULE.load_audit(audit_path), MODULE.load_arena_manifest(manifest_path)
            )
            report = MODULE.analyze_dataset(dataset, "arena")

            top5 = report["arena_game_results"]["frozen_rank_cohorts"]["top_5"]
            named = report["arena_game_results"]["named_opponents"][0]
            self.assertEqual((top5["games"], top5["wins"], top5["losses"]), (2, 1, 1))
            self.assertEqual((named["games"], named["wins"], named["losses"]), (2, 1, 1))

            first_only = [copy.deepcopy(loss_rows[0]), copy.deepcopy(win_rows[0])]
            for row in first_only:
                row["max_own_decisions_per_game"] = 1
            first_only_path = pathlib.Path(temporary) / "first-only.jsonl"
            write_jsonl(first_only_path, first_only)
            first_only_dataset = MODULE.join_arena_manifest(
                MODULE.load_audit(first_only_path),
                MODULE.load_arena_manifest(manifest_path),
            )
            self.assertEqual(len(first_only_dataset["rows"]), 2)
            self.assertEqual(
                first_only_dataset["configuration"][
                    "max_own_decisions_per_game"
                ],
                1,
            )

            partial_path = pathlib.Path(temporary) / "partial.jsonl"
            write_jsonl(partial_path, loss_rows)
            with self.assertRaisesRegex(MODULE.AuditAnalysisError, "coverage"):
                MODULE.join_arena_manifest(
                    MODULE.load_audit(partial_path),
                    MODULE.load_arena_manifest(manifest_path),
                )

    def test_comparison_aligns_by_game_and_state(self):
        game_one = [make_row("101")]
        game_two = [make_row("102", candidate_player=1, winner=0, transcript_prefix="7")]
        baseline_rows = game_one + game_two
        hypothesis_rows = copy.deepcopy(game_two + game_one)
        for row in hypothesis_rows:
            row["model_sha256"] = "7" * 64
        changed = next(row for row in hypothesis_rows if row["game_id"] == "101")
        set_diagnostic(changed, "chosen", "1", 1, 2)
        changed["classification"] = "bfm-override"
        changed["classification_reason"] = "BFM selected another boundary"

        with tempfile.TemporaryDirectory() as temporary:
            baseline_path = pathlib.Path(temporary) / "baseline.jsonl"
            hypothesis_path = pathlib.Path(temporary) / "hypothesis.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(hypothesis_path, hypothesis_rows)
            report = MODULE.compare_datasets(
                MODULE.load_audit(baseline_path),
                MODULE.load_audit(hypothesis_path),
                "baseline",
                "hypothesis",
            )

        self.assertEqual(report["aligned_decisions"], 2)
        self.assertEqual(report["chosen_action_changes"], 1)
        self.assertEqual(report["classification_transitions"]["match -> bfm-override"], 1)
        self.assertIn("model_sha256", report["configuration_changes"])


if __name__ == "__main__":
    unittest.main()
