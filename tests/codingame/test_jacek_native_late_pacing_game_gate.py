import collections
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "jacek_native_late_pacing_game_gate.py"
PANEL_PATH = (
    ROOT / "results" / "codingame_arena_diagnostics" / "panels" /
    "jacek-native-late-pacing-root-heldout-h62-h64-h66-h68-v2" /
    "panel.json"
)
SPEC = importlib.util.spec_from_file_location("late_pacing_game_gate", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@unittest.skipUnless(
    PANEL_PATH.is_file(), "local ignored provenance evidence is unavailable"
)
class LatePacingGameGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.panel_sha = gate.load_panel(PANEL_PATH)
        collector_values = [
            (source["run_id"], str(ROOT / source["collector_tsv_path"]))
            for source in cls.panel["sources"]
        ]
        cls.collectors, cls.collector_reports = gate.load_collectors(
            cls.panel, collector_values
        )
        cls.roots, cls.selection = gate.select_roots(
            cls.panel, cls.collectors
        )

    def test_actual_panel_yields_all_64_heldout_roots(self):
        self.assertEqual(self.panel_sha, gate.EXPECTED_PANEL_SHA256)
        self.assertEqual(len(self.collectors), 4)
        self.assertEqual(len(self.roots), 64)
        selected = collections.Counter()
        for root in self.roots:
            population = (
                "control" if root["role"] == "matched-winning-control"
                else "trap"
            )
            selected[(root["run_id"], population,
                      root["candidate_player"])] += 1
        self.assertEqual(len(selected), 16)
        expected = collections.Counter()
        for population, entries in (
            ("trap", self.panel["trap_states"]),
            ("control", self.panel["matched_winning_controls"]),
        ):
            for entry in entries:
                expected[(entry["run_id"], population,
                          entry["candidate_player"])] += 1
        self.assertEqual(selected, expected)
        self.assertEqual(
            collections.Counter(root["candidate_player"] for root in self.roots),
            {0: 32, 1: 32},
        )
        self.assertEqual(
            len({(root["run_id"], root["game_id"]) for root in self.roots}),
            64,
        )
        self.assertEqual(self.selection["contract"], gate.SELECTION)
        self.assertEqual(self.selection["source_games"], 64)
        self.assertEqual(
            {
                (group["run_id"], group["population"],
                 group["candidate_player"]): group["selected"]
                for group in self.selection["groups"]
            },
            dict(expected),
        )

    def test_plan_is_deterministic_ascii_and_binds_every_replayed_root(self):
        first = gate.render_plan(self.panel, self.panel_sha, self.roots)
        second = gate.render_plan(self.panel, self.panel_sha, self.roots)
        self.assertEqual(first, second)
        self.assertEqual(first.count(b"\n"), 4 + 1 + 64)
        self.assertNotIn(b"\r", first)
        self.assertIn(self.panel_sha.encode(), first)
        self.assertEqual(len({root["canonical_key"] for root in self.roots}), 64)
        trap_keys = {entry["canonical_key"] for entry in self.panel["trap_states"]}
        control_keys = {
            entry["canonical_key"]
            for entry in self.panel["matched_winning_controls"]
        }
        self.assertFalse(trap_keys & control_keys)

    def stdout(self, candidate_wins_per_orientation=(35, 35)):
        candidate = {
            "artifact_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "packed_sha256": "c" * 64,
        }
        baseline = {
            "artifact_sha256": "d" * 64,
            "model_sha256": "e" * 64,
            "packed_sha256": "f" * 64,
        }
        plan_sha = "1" * 64
        source_sha = self.panel["source_sha256"]
        lines = [
            "candidate_runtime_sha256=" + candidate["artifact_sha256"]
            + " candidate_model_sha256=" + candidate["model_sha256"]
            + " candidate_packed_sha256=" + candidate["packed_sha256"],
            "baseline_runtime_sha256=" + baseline["artifact_sha256"]
            + " baseline_model_sha256=" + baseline["model_sha256"]
            + " baseline_packed_sha256=" + baseline["packed_sha256"],
            f"plan_sha256={plan_sha} panel_sha256={self.panel_sha} "
            f"source_sha256={source_sha}",
        ]
        total = 0
        orientations = [0, 0]
        populations = {"trap": 0, "control": 0}
        for root_index, root in enumerate(self.roots):
            population = (
                "control" if root["role"] == "matched-winning-control"
                else "trap"
            )
            for orientation in (0, 1):
                candidate_won = root_index < candidate_wins_per_orientation[
                    orientation
                ]
                winner = orientation if candidate_won else 1 - orientation
                if candidate_won:
                    total += 1
                    orientations[orientation] += 1
                    populations[population] += 1
                game = root_index * 2 + orientation
                lines.append(
                    f"game={game} root={root_index} run={root['run_id']} "
                    f"population={population} "
                    f"source_color={root['candidate_player']} "
                    f"candidate_orientation={orientation} winner={winner}"
                )
        passed = (
            total >= gate.REQUIRED_TOTAL
            and min(orientations) >= gate.REQUIRED_PER_ORIENTATION
        )
        summary = {
            "candidate": total,
            "baseline": gate.GAMES - total,
            "unfinished": 0,
            "operational_failures": 0,
            "candidate_player_one": orientations[0],
            "candidate_player_two": orientations[1],
            "trap_candidate": populations["trap"],
            "control_candidate": populations["control"],
            "games": gate.GAMES,
            "searches": 500,
            "expansions": 5000,
            "maximum_tree": 4096,
            "work": gate.WORK,
            "seed": gate.SEED,
            "temperature": 3,
            "temperature_turns": 12,
            "maximum_generated_turns": 384,
            "required_total": gate.REQUIRED_TOTAL,
            "required_per_orientation": gate.REQUIRED_PER_ORIENTATION,
            "passed": str(passed).lower(),
        }
        lines.append("summary " + " ".join(
            f"{key}={value}" for key, value in summary.items()
        ))
        return ("\n".join(lines) + "\n").encode(), candidate, baseline, plan_sha

    def test_stdout_requires_full_128_games_and_both_orientations(self):
        raw, candidate, baseline, plan_sha = self.stdout()
        result = gate.parse_stdout(
            raw, 0, plan_sha, self.panel_sha, self.panel["source_sha256"],
            candidate, baseline, self.roots,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate"], 70)
        weak, candidate, baseline, plan_sha = self.stdout((39, 30))
        with self.assertRaisesRegex(gate.GateError, "exit status"):
            gate.parse_stdout(
                weak, 0, plan_sha, self.panel_sha,
                self.panel["source_sha256"], candidate, baseline, self.roots,
            )


if __name__ == "__main__":
    unittest.main()
