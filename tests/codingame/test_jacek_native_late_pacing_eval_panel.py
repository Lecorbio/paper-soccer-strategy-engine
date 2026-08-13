import collections
import hashlib
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "jacek_native_late_pacing_eval_panel.py"
PANEL_PATH = (
    ROOT / "results" / "codingame_arena_diagnostics" / "panels" /
    "jacek-native-late-pacing-root-heldout-h62-h64-h66-h68-v2" /
    "panel.json"
)
EXPECTED_SHA256 = (
    "64283c5a4e7c5ac360969120a79e966a10cff9eb39d9cb0380cadccc14198246"
)
SPEC = importlib.util.spec_from_file_location("late_pacing_eval_panel", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
panel_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel_builder)


@unittest.skipUnless(
    PANEL_PATH.is_file(), "local ignored provenance evidence is unavailable"
)
class LatePacingEvaluationPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = PANEL_PATH.read_bytes()
        cls.panel = json.loads(cls.raw)
        cls.sources = [
            panel_builder.trap_builder.Source(
                ROOT / source["manifest_path"],
                ROOT / source["decision_audit_path"],
                ROOT / source["collector_tsv_path"],
            )
            for source in cls.panel["sources"]
        ]
        cls.exclusions = [
            ROOT / item["path"]
            for item in cls.panel["dependencies"]["exclusion_panels"]
        ]
        cls.focused = [
            ROOT / item["manifest_path"]
            for item in cls.panel["dependencies"]["focused_continuations"]
        ]

    def test_checked_panel_is_exact_canonical_rebuild(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), EXPECTED_SHA256)
        self.assertEqual(
            panel_builder.canonical_json_bytes(self.panel), self.raw
        )
        rebuilt = panel_builder.build_panel(
            self.sources, self.exclusions, self.focused
        )
        self.assertEqual(
            panel_builder.canonical_json_bytes(rebuilt), self.raw
        )

    def test_roots_are_balanced_and_source_game_disjoint_from_training(self):
        traps = self.panel["trap_states"]
        controls = self.panel["matched_winning_controls"]
        self.assertEqual(self.panel["counts"], {
            "clean_games": 312,
            "excluded_focused_source_games": 53,
            "excluded_focused_start_roots": 96,
            "matched_controls": 32,
            "trap_states": 32,
        })
        self.assertEqual(self.panel["independence"], {
            "contract": panel_builder.INDEPENDENCE,
            "control_source_games_distinct": True,
            "focused_start_canonical_overlap": 0,
            "focused_start_source_game_overlap": 0,
            "one_selected_root_per_source_game": True,
            "scope": (
                "root/source-game held out from focused continuation starts "
                "only; same frozen public arena source family"
            ),
        })
        self.assertEqual(
            collections.Counter(item["candidate_player"] for item in traps),
            {0: 16, 1: 16},
        )
        self.assertEqual(
            collections.Counter(item["candidate_player"] for item in controls),
            {0: 16, 1: 16},
        )
        all_entries = traps + controls
        self.assertEqual(len({item["canonical_key"] for item in all_entries}), 64)
        self.assertEqual(
            len({(item["run_id"], item["game_id"]) for item in all_entries}),
            64,
        )

        collectors = {
            source["run_id"]: panel_builder.restart.parse_collector_bytes(
                (ROOT / source["collector_tsv_path"]).read_bytes()
            )
            for source in self.panel["sources"]
        }
        focused_keys, focused_games, _ = panel_builder.load_focused_runs(
            self.focused, collectors
        )
        selected_keys = {item["canonical_key"] for item in all_entries}
        selected_games = {
            (item["run_id"], item["game_id"]) for item in all_entries
        }
        self.assertEqual(len(focused_keys), 96)
        self.assertEqual(len(focused_games), 53)
        self.assertFalse(selected_keys & focused_keys)
        self.assertFalse(selected_games & focused_games)

    def test_every_control_matches_one_trap_run_and_color(self):
        traps = {
            item["state_id"]: item for item in self.panel["trap_states"]
        }
        controls = self.panel["matched_winning_controls"]
        self.assertEqual(
            {item["matched_trap_state_id"] for item in controls}, set(traps)
        )
        for control in controls:
            trap = traps[control["matched_trap_state_id"]]
            self.assertEqual(control["run_id"], trap["run_id"])
            self.assertEqual(
                control["candidate_player"], trap["candidate_player"]
            )
            self.assertEqual(control["turn_band"], trap["turn_band"])
            self.assertEqual(
                control["used_edge_band"], trap["used_edge_band"]
            )
            self.assertEqual(control["zone"], trap["zone"])
            self.assertEqual(control["match_exact"], {
                "color": True,
                "run_id": True,
                "turn_band": True,
                "used_edge_band": True,
                "zone": True,
            })

        selected = collections.Counter(
            (item["run_id"], item["candidate_player"])
            for item in self.panel["trap_states"]
        )
        quotas = {
            (item["run_id"], item["candidate_player"]):
                item["selected_source_games"]
            for item in self.panel["selection"]["quotas"]
        }
        self.assertEqual(dict(selected), quotas)
        self.assertEqual(len(quotas), 8)
        self.assertTrue(all(value > 0 for value in quotas.values()))

    def test_quota_allocator_fails_closed_on_insufficient_distribution(self):
        with self.assertRaisesRegex(
            panel_builder.PanelError, "cannot satisfy color balance"
        ):
            panel_builder.allocate_quotas({"run-a": 15, "run-b": 0}, 16)


if __name__ == "__main__":
    unittest.main()
