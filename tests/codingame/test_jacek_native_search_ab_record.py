import importlib.util
import base64
import hashlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/jacek_native_search_ab_record.py"
SPEC = importlib.util.spec_from_file_location("search_ab_record", TOOL)
record = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = record
SPEC.loader.exec_module(record)


class SearchAbRecordTest(unittest.TestCase):
    def arguments(self):
        return record.parser().parse_args([
            "--gate-binary", "build/gate",
            "--checkpoint", "model.runtime",
            "--output-dir", "results/ab",
            "--pairs", "2",
            "--first-ms", "10",
            "--later-ms", "2",
            "--opening-turns", "0,4",
            "--seed", "7",
            "--candidate-opening-tree-nodes", "20000",
            "--candidate-later-tree-nodes", "120000",
            "--candidate-opening-own-decisions", "4",
            "--baseline-opening-tree-nodes", "80000",
            "--baseline-later-tree-nodes", "80000",
            "--baseline-opening-own-decisions", "4",
            "--minimum-candidate-wins", "2",
            "--minimum-wins-per-color", "1",
        ])

    def transcript(self, *, summary_updates=None, pair_lines=None):
        arguments = self.arguments()
        counts = {
            f"{side}_{name}": value
            for side in ("candidate", "baseline")
            for name, value in {
                "decisions": 4,
                "expansions": 10,
                "child_evaluations": 20,
                "completed_actions": 30,
                "partial_paths": 40,
                "max_tree": 64,
                "tree_cap_searches": 0,
                "final_overrides": 0,
                "root_reply_probe_candidates": 0,
                "root_reply_probe_attempts": 0,
                "root_reply_probe_expansions": 0,
                "root_reply_probe_refutations": 0,
                "root_reply_probe_generator_truncations": 0,
                "root_reply_probe_budget_truncations": 0,
                "supported_advance_penalized_actions": 0,
                "supported_advance_selected_actions": 0,
                "supported_advance_selection_overrides": 0,
                "supported_advance_player_one_selection_overrides": 0,
                "supported_advance_player_two_selection_overrides": 0,
                "exploration_opening_player_one_decisions": 0,
                "exploration_opening_player_two_decisions": 0,
                "exploration_later_player_one_decisions": 2,
                "exploration_later_player_two_decisions": 2,
                "ms": 4.0,
                "first_clock_decisions": 2,
                "later_clock_decisions": 2,
                "first_clock_ms": 2.0,
                "later_clock_ms": 2.0,
                "max_first_ms": 1.0,
                "max_later_ms": 1.0,
                "deadline_searches": 0,
                "headroom_failures": 0,
                "operational_timeouts": 0,
            }.items()
        }
        fields = {
            "candidate": 2,
            "baseline": 2,
            "unfinished": 0,
            "candidate_player_one": 1,
            "candidate_player_two": 1,
            "games": 4,
            **counts,
            **{
                f"{side}_{phase}_{field}": value
                for side in ("candidate", "baseline")
                for phase in ("opening", "later")
                for field, value in {
                    "decisions": 2,
                    "deadline_searches": 0,
                    "tree_cap_searches": 0,
                    "ms": 2.0,
                    "max_ms": 1.0,
                    "max_tree": 64,
                }.items()
            },
            "profile": "10/2",
            "pairs": 2,
            "maximum_turns": 384,
            "opening_turns": "0,4",
            "opening_seed": 7,
            "shuffle_seed_policy": "deployment-constant",
            "runtime_policy": "same",
            "game_order_policy": "pair-parity-color-swap",
            "timing_scope": "search-through-apply",
            "search_construction_timing": "included",
            "first_headroom_limit_ms": 990,
            "later_headroom_limit_ms": 198,
            "first_operational_limit_ms": 1000,
            "later_operational_limit_ms": 200,
            "supported_advance_sparse_edge_limit": 48,
            "supported_advance_start_progress": 2,
            "supported_advance_endpoint_progress_threshold": 4,
            "maximum_supported_advance_penalty": 0.2,
            "candidate_opening_tree_nodes": 20000,
            "candidate_later_tree_nodes": 120000,
            "candidate_opening_own_decisions": 4,
            "candidate_root_reply_width": 0,
            "candidate_supported_advance_penalty": 0,
            "candidate_first_budget_ms": 10,
            "candidate_later_budget_ms": 2,
            "candidate_opening_c": 0.95,
            "candidate_later_c": 0.95,
            "candidate_exploration_opening_own_decisions": 0,
            "candidate_fpu": 0.5,
            "candidate_final": "value-log-visits",
            "baseline_opening_tree_nodes": 80000,
            "baseline_later_tree_nodes": 80000,
            "baseline_opening_own_decisions": 4,
            "baseline_root_reply_width": 0,
            "baseline_supported_advance_penalty": 0,
            "baseline_first_budget_ms": 10,
            "baseline_later_budget_ms": 2,
            "baseline_opening_c": 0.95,
            "baseline_later_c": 0.95,
            "baseline_exploration_opening_own_decisions": 0,
            "baseline_fpu": 0.5,
            "baseline_final": "value-log-visits",
            "required_total": 2,
            "required_per_color": 1,
            "passed": "true",
        }
        if summary_updates:
            fields.update(summary_updates)
        summary = " ".join(f"{key}={value}" for key, value in fields.items())
        digest = "1" * 64
        if pair_lines is None:
            mask = (1 << 64) - 1
            seeds = [
                (arguments.seed + (pair + 1) * 0x9E3779B97F4A7C15) & mask
                for pair in range(arguments.pairs)
            ]
            pair_lines = [
                f"pair=0 opening_turns=0 seed={seeds[0]} c0=0 c1=0",
                f"pair=1 opening_turns=4 seed={seeds[1]} c1=1 c0=1",
            ]
        lines = [
            f"runtime_sha256={digest} model_sha256={'2' * 64} "
            f"packed_sha256={'3' * 64}",
            "candidate_opening_tree_nodes=20000 "
            "candidate_later_tree_nodes=120000 "
            "candidate_opening_own_decisions=4 candidate_root_reply_width=0 "
            "candidate_supported_advance_penalty=0 "
            "candidate_first_ms=10 candidate_later_ms=2 "
            "candidate_opening_c=0.95 candidate_later_c=0.95 "
            "candidate_exploration_opening_own_decisions=0 "
            "candidate_fpu=0.5 candidate_final=value-log-visits",
            "baseline_opening_tree_nodes=80000 "
            "baseline_later_tree_nodes=80000 "
            "baseline_opening_own_decisions=4 baseline_root_reply_width=0 "
            "baseline_supported_advance_penalty=0 "
            "baseline_first_ms=10 baseline_later_ms=2 "
            "baseline_opening_c=0.95 baseline_later_c=0.95 "
            "baseline_exploration_opening_own_decisions=0 "
            "baseline_fpu=0.5 baseline_final=value-log-visits",
            *pair_lines,
            f"summary {summary}",
        ]
        return ("\n".join(lines) + "\n").encode()

    def test_parses_complete_isolated_transcript(self):
        parsed = record.parse_gate_stdout(self.transcript(), self.arguments())
        self.assertEqual(parsed["candidate_profile"]["opening_tree_nodes"], 20000)
        self.assertEqual(parsed["candidate_profile"]["later_tree_nodes"], 120000)
        self.assertEqual(parsed["baseline_profile"]["later_tree_nodes"], 80000)
        self.assertEqual(
            parsed["candidate_profile"]["supported_advance_penalty"], 0
        )
        self.assertTrue(parsed["summary"]["passed"])
        self.assertEqual(parsed["summary"]["candidate"], 2)
        self.assertEqual([pair["pair"] for pair in parsed["pairs"]], [0, 1])

    def test_singleton_opening_schedule_remains_text(self):
        arguments = self.arguments()
        arguments.opening_turns = "0"
        self.assertEqual(record._opening_depths(arguments.opening_turns), [0])

    def test_rejects_missing_isolation_contract(self):
        with self.assertRaisesRegex(record.RecordError, "isolation contract"):
            record.parse_gate_stdout(
                self.transcript(summary_updates={"runtime_policy": "different"}),
                self.arguments(),
            )

    def test_rejects_ambiguous_timing_scope(self):
        with self.assertRaisesRegex(record.RecordError, "isolation contract"):
            record.parse_gate_stdout(
                self.transcript(summary_updates={"timing_scope": "unspecified"}),
                self.arguments(),
            )
        with self.assertRaisesRegex(record.RecordError, "isolation contract"):
            record.parse_gate_stdout(
                self.transcript(
                    summary_updates={"search_construction_timing": "excluded"}
                ),
                self.arguments(),
            )

    def test_rejects_truncated_or_extra_transcript(self):
        truncated = self.transcript().splitlines(keepends=True)
        del truncated[4]
        with self.assertRaisesRegex(record.RecordError, "transcript lines"):
            record.parse_gate_stdout(b"".join(truncated), self.arguments())
        with self.assertRaisesRegex(record.RecordError, "transcript lines"):
            record.parse_gate_stdout(
                self.transcript() + b"summary passed=true\n", self.arguments()
            )

    def test_rejects_noncanonical_transcript_line_endings(self):
        raw = self.transcript()
        with self.assertRaisesRegex(record.RecordError, "canonical line text"):
            record.parse_gate_stdout(raw.rstrip(b"\n"), self.arguments())
        with self.assertRaisesRegex(record.RecordError, "canonical line text"):
            record.parse_gate_stdout(raw.replace(b"\n", b"\r\n"), self.arguments())

    def test_rejects_pair_schedule_or_result_tamper(self):
        lines = self.transcript().decode().splitlines()
        lines[4] = lines[4].replace("c1=1 c0=1", "c0=1 c1=1")
        with self.assertRaisesRegex(record.RecordError, "schedule is stale"):
            record.parse_gate_stdout(
                ("\n".join(lines) + "\n").encode(), self.arguments()
            )
        with self.assertRaisesRegex(record.RecordError, "pair transcript"):
            record.parse_gate_stdout(
                self.transcript(summary_updates={"candidate": 3}),
                self.arguments(),
            )

    def test_rejects_profile_or_command_tamper(self):
        lines = self.transcript().decode().splitlines()
        lines[1] = lines[1].replace(
            "candidate_later_c=0.95", "candidate_later_c=1.25"
        )
        with self.assertRaisesRegex(record.RecordError, "invoked command"):
            record.parse_gate_stdout(
                ("\n".join(lines) + "\n").encode(), self.arguments()
            )
        with self.assertRaisesRegex(record.RecordError, "configuration"):
            record.parse_gate_stdout(
                self.transcript(summary_updates={"profile": "800/155"}),
                self.arguments(),
            )

    def test_rejects_phase_diagnostic_tamper(self):
        with self.assertRaisesRegex(record.RecordError, "phase diagnostics"):
            record.parse_gate_stdout(
                self.transcript(
                    summary_updates={"candidate_opening_max_tree": 20001}
                ),
                self.arguments(),
            )
        with self.assertRaisesRegex(record.RecordError, "diagnostics"):
            record.parse_gate_stdout(
                self.transcript(
                    summary_updates={"candidate_first_clock_decisions": 3}
                ),
                self.arguments(),
            )
        with self.assertRaisesRegex(record.RecordError, "phase diagnostics"):
            record.parse_gate_stdout(
                self.transcript(
                    summary_updates={"candidate_opening_decisions": 3}
                ),
                self.arguments(),
            )

    def test_command_uses_one_checkpoint_and_shared_clocks(self):
        command = record.command(self.arguments())
        self.assertEqual(command.count("--checkpoint"), 1)
        self.assertEqual(command[command.index("--first-ms") + 1], "10")
        self.assertEqual(command[command.index("--later-ms") + 1], "2")
        self.assertEqual(
            command[
                command.index("--candidate-supported-advance-penalty") + 1
            ],
            "0.0",
        )
        self.assertEqual(
            command[command.index("--candidate-c") + 1], "0.95"
        )

    def test_phase_exploration_is_bound_and_emitted_without_legacy_c(self):
        arguments = self.arguments()
        arguments.candidate_opening_c = 0.65
        arguments.candidate_later_c = 0.80
        arguments.candidate_exploration_opening_own_decisions = 12
        raw = self.transcript().decode().replace(
            "candidate_opening_c=0.95 candidate_later_c=0.95 "
            "candidate_exploration_opening_own_decisions=0",
            "candidate_opening_c=0.65 candidate_later_c=0.8 "
            "candidate_exploration_opening_own_decisions=12",
        ).replace(
            "candidate_exploration_opening_player_one_decisions=0",
            "candidate_exploration_opening_player_one_decisions=2",
        ).replace(
            "candidate_exploration_opening_player_two_decisions=0",
            "candidate_exploration_opening_player_two_decisions=2",
        ).replace(
            "candidate_exploration_later_player_one_decisions=2",
            "candidate_exploration_later_player_one_decisions=0",
        ).replace(
            "candidate_exploration_later_player_two_decisions=2",
            "candidate_exploration_later_player_two_decisions=0",
        )
        parsed = record.parse_gate_stdout(raw.encode(), arguments)
        self.assertEqual(parsed["candidate_profile"]["opening_c"], 0.65)
        self.assertEqual(parsed["candidate_profile"]["later_c"], 0.8)
        self.assertEqual(
            parsed["candidate_profile"][
                "exploration_opening_own_decisions"
            ],
            12,
        )
        command = record.command(arguments)
        self.assertNotIn("--candidate-c", command)
        self.assertEqual(
            command[command.index("--candidate-opening-c") + 1], "0.65"
        )
        self.assertEqual(
            command[
                command.index(
                    "--candidate-exploration-opening-own-decisions"
                ) + 1
            ],
            "12",
        )

    def test_exploration_resolution_rejects_mixed_partial_and_zero_profiles(self):
        mixed = self.arguments()
        mixed.candidate_c = 0.95
        mixed.candidate_opening_c = 0.65
        mixed.candidate_later_c = 0.80
        mixed.candidate_exploration_opening_own_decisions = 12
        with self.assertRaisesRegex(record.RecordError, "mixes"):
            record.command(mixed)
        partial = self.arguments()
        partial.candidate_opening_c = 0.65
        with self.assertRaisesRegex(record.RecordError, "incomplete"):
            record.command(partial)
        zero = self.arguments()
        zero.candidate_opening_c = 0.65
        zero.candidate_later_c = 0.80
        zero.candidate_exploration_opening_own_decisions = 0
        with self.assertRaisesRegex(record.RecordError, "incomplete"):
            record.command(zero)

    def test_cutoff_zero_rejects_reported_opening_exploration_decisions(self):
        with self.assertRaisesRegex(record.RecordError, "cutoff-zero"):
            record.parse_gate_stdout(
                self.transcript(summary_updates={
                    "candidate_exploration_opening_player_one_decisions": 1,
                    "candidate_exploration_later_player_one_decisions": 1,
                }),
                self.arguments(),
            )

    def test_supported_advance_profiles_are_bound_and_bounded(self):
        arguments = self.arguments()
        arguments.candidate_supported_advance_penalty = 0.11
        arguments.baseline_supported_advance_penalty = 0.15
        raw = self.transcript().decode().replace(
            "candidate_supported_advance_penalty=0",
            "candidate_supported_advance_penalty=0.11",
        ).replace(
            "baseline_supported_advance_penalty=0",
            "baseline_supported_advance_penalty=0.15",
        )
        parsed = record.parse_gate_stdout(raw.encode(), arguments)
        self.assertEqual(
            parsed["candidate_profile"]["supported_advance_penalty"], 0.11
        )
        invalid = self.arguments()
        invalid.candidate_supported_advance_penalty = -0.01
        with self.assertRaisesRegex(record.RecordError, "supported advance"):
            record.command(invalid)
        invalid.candidate_supported_advance_penalty = 0.201
        with self.assertRaisesRegex(record.RecordError, "supported advance"):
            record.command(invalid)

    def test_command_supports_complete_independent_clocks(self):
        arguments = self.arguments()
        arguments.first_ms = None
        arguments.later_ms = None
        arguments.candidate_first_ms = 950
        arguments.candidate_later_ms = 175
        arguments.baseline_first_ms = 950
        arguments.baseline_later_ms = 170
        command = record.command(arguments)
        self.assertNotIn("--first-ms", command)
        self.assertEqual(
            command[command.index("--candidate-first-ms") + 1], "950"
        )
        self.assertEqual(
            command[command.index("--candidate-later-ms") + 1], "175"
        )
        self.assertEqual(
            command[command.index("--baseline-later-ms") + 1], "170"
        )

    def test_transcript_binds_all_four_independent_clocks(self):
        arguments = self.arguments()
        arguments.first_ms = None
        arguments.later_ms = None
        arguments.candidate_first_ms = 950
        arguments.candidate_later_ms = 175
        arguments.baseline_first_ms = 950
        arguments.baseline_later_ms = 170
        raw = self.transcript().decode()
        raw = raw.replace(
            "candidate_first_ms=10 candidate_later_ms=2",
            "candidate_first_ms=950 candidate_later_ms=175",
        ).replace(
            "baseline_first_ms=10 baseline_later_ms=2",
            "baseline_first_ms=950 baseline_later_ms=170",
        ).replace(
            "candidate_first_budget_ms=10 candidate_later_budget_ms=2",
            "candidate_first_budget_ms=950 candidate_later_budget_ms=175",
        ).replace(
            "baseline_first_budget_ms=10 baseline_later_budget_ms=2",
            "baseline_first_budget_ms=950 baseline_later_budget_ms=170",
        ).replace("profile=10/2", "profile=950/175|950/170")
        parsed = record.parse_gate_stdout(raw.encode(), arguments)
        self.assertEqual(parsed["candidate_profile"]["later_ms"], 175)
        self.assertEqual(parsed["baseline_profile"]["later_ms"], 170)

    def test_clock_resolution_rejects_mixed_or_partial_profiles(self):
        mixed = self.arguments()
        mixed.candidate_first_ms = 950
        mixed.candidate_later_ms = 175
        mixed.baseline_first_ms = 950
        mixed.baseline_later_ms = 170
        with self.assertRaisesRegex(record.RecordError, "mixes"):
            record.command(mixed)
        partial = self.arguments()
        partial.first_ms = None
        partial.later_ms = None
        partial.candidate_first_ms = 950
        with self.assertRaisesRegex(record.RecordError, "incomplete"):
            record.command(partial)
        partial_shared = self.arguments()
        partial_shared.later_ms = None
        with self.assertRaisesRegex(record.RecordError, "incomplete"):
            record.command(partial_shared)

    def test_command_preserves_legacy_and_mixed_profile_interfaces(self):
        legacy = record.parser().parse_args([
            "--gate-binary", "build/gate", "--checkpoint", "model.runtime",
            "--output-dir", "results/ab", "--candidate-tree-nodes", "20000",
            "--baseline-tree-nodes", "80000",
        ])
        command = record.command(legacy)
        self.assertIn("--candidate-tree-nodes", command)
        self.assertIn("--baseline-tree-nodes", command)
        self.assertNotIn("--candidate-opening-tree-nodes", command)
        mixed = self.arguments()
        mixed.baseline_opening_tree_nodes = None
        mixed.baseline_later_tree_nodes = None
        mixed.baseline_opening_own_decisions = None
        mixed.baseline_tree_nodes = 80000
        command = record.command(mixed)
        self.assertIn("--candidate-opening-tree-nodes", command)
        self.assertIn("--baseline-tree-nodes", command)

    def test_profile_resolution_rejects_mixed_or_partial_split(self):
        mixed = self.arguments()
        mixed.candidate_tree_nodes = 20000
        with self.assertRaisesRegex(record.RecordError, "mixes"):
            record.command(mixed)
        partial = self.arguments()
        partial.candidate_later_tree_nodes = None
        with self.assertRaisesRegex(record.RecordError, "incomplete"):
            record.command(partial)

    def test_runtime_identity_is_derived_from_exact_checkpoint_bytes(self):
        payload = bytes((record.RUNTIME_WEIGHT_COUNT * 3 + 7) // 8)
        packed = hashlib.sha256(payload).hexdigest()
        model = "2" * 64
        raw = (
            f"{record.RUNTIME_SCHEMA}\n{record.MODEL_SCHEMA}\n"
            f"{record.FEATURE_SCHEMA}\n{model}\n{packed}\n1 1 1\n"
            f"{base64.b64encode(payload).decode()}\n"
        ).encode()
        self.assertEqual(record.runtime_identity(raw), {
            "runtime_sha256": hashlib.sha256(raw).hexdigest(),
            "model_sha256": model,
            "packed_sha256": packed,
        })
        tampered = raw.replace(packed.encode(), ("3" * 64).encode(), 1)
        with self.assertRaisesRegex(record.RecordError, "payload identity"):
            record.runtime_identity(tampered)


if __name__ == "__main__":
    unittest.main()
