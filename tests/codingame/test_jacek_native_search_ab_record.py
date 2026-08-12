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
                "ms": 4.0,
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
            "candidate_opening_tree_nodes": 20000,
            "candidate_later_tree_nodes": 120000,
            "candidate_opening_own_decisions": 4,
            "candidate_c": 0.95,
            "candidate_fpu": 0.5,
            "candidate_final": "value-log-visits",
            "baseline_opening_tree_nodes": 80000,
            "baseline_later_tree_nodes": 80000,
            "baseline_opening_own_decisions": 4,
            "baseline_c": 0.95,
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
            "candidate_opening_own_decisions=4 candidate_c=0.95 "
            "candidate_fpu=0.5 candidate_final=value-log-visits",
            "baseline_opening_tree_nodes=80000 "
            "baseline_later_tree_nodes=80000 "
            "baseline_opening_own_decisions=4 baseline_c=0.95 "
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
        lines[1] = lines[1].replace("candidate_c=0.95", "candidate_c=1.25")
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
