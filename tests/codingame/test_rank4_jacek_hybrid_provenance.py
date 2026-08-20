#!/usr/bin/env python3

"""Focused immutable-boundary tests for the 36-hour hybrid campaign."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
FREEZER_PATH = (
    ROOT / "results/rank_4_jacek_hybrid/tools/freeze_campaign.py"
)
MANIFEST_PATH = ROOT / "results/rank_4_jacek_hybrid/campaign.json"
EXPECTED_MANIFEST_SHA256 = (
    "aed2a52f7a59c2b1988b5c365c23b57f8ec41fbfb50927211655a8565df63fa7"
)


def load_freezer():
    specification = importlib.util.spec_from_file_location(
        "rank4_jacek_hybrid_freezer", FREEZER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load campaign freezer")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class HybridCampaignProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freezer = load_freezer()
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_read_only_verifier_accepts_frozen_artifacts(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(FREEZER_PATH), "check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["sha256"], EXPECTED_MANIFEST_SHA256)

    def test_clean_checkout_protected_tree_equivalent_is_exact(self) -> None:
        original, clean_checkout = self.freezer.PROTECTED_TREE_EQUIVALENTS[0]
        self.assertTrue(
            self.freezer.protected_tree_identity_matches(
                clean_checkout, original
            )
        )
        changed = dict(clean_checkout)
        changed["file_count"] += 1
        self.assertFalse(
            self.freezer.protected_tree_identity_matches(changed, original)
        )
        self.assertFalse(
            self.freezer.protected_tree_identity_matches(
                clean_checkout, changed
            )
        )

    def test_rank4_consolidation_successor_is_exact(self) -> None:
        self.assertFalse(
            (ROOT / self.freezer.CONSOLIDATED_PREDECESSOR_PATH).exists()
        )
        self.assertEqual(
            self.freezer.tree_identity(
                ROOT / "submissions/codingame/bots/rank_4"
            ),
            self.freezer.CONSOLIDATED_RANK4_TREE,
        )

    def test_jacek_documentation_successor_is_exact(self) -> None:
        _, successor = self.freezer.PROTECTED_TREE_EQUIVALENTS[1]
        self.assertEqual(
            self.freezer.tree_identity(
                ROOT / "submissions/codingame/bots/jacek_nn"
            ),
            successor,
        )

    def test_time_control_and_derivative_lineage_are_exact(self) -> None:
        self.assertEqual(
            self.manifest["time_boundary"]["goal_created_at_epoch"],
            1_786_648_507,
        )
        self.assertEqual(
            self.manifest["time_boundary"]["deadline_epoch"],
            1_786_778_107,
        )
        self.assertEqual(
            self.manifest["control"]["source"]["sha256"],
            "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9",
        )
        self.assertEqual(self.manifest["control"]["source"]["bytes"], 98_624)
        self.assertEqual(
            self.manifest["lineage"]["classification"],
            "rank4-derived Jacek hybrid; explicitly not clean-room",
        )

    def test_opening_roles_and_color_swapped_counts_were_preregistered(self) -> None:
        totals = self.manifest["procedural_openings"]["role_totals"]
        self.assertEqual(totals["development"]["color_swapped_games"], 306)
        self.assertEqual(totals["validation"]["color_swapped_games"], 106)
        self.assertEqual(totals["final"]["color_swapped_games"], 212)
        assignments = self.manifest["procedural_openings"]["assignments"]
        self.assertEqual(len(assignments), 12)
        self.assertEqual(
            {(item["role"], item["depth"]) for item in assignments},
            {
                (role, depth)
                for role in ("development", "validation", "final")
                for depth in (4, 8, 12, 20)
            },
        )

    def test_arena_registry_is_canonical_and_id_only(self) -> None:
        declaration = self.manifest["arena_exclusions"]
        path = ROOT / declaration["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            path.read_bytes(), self.freezer.collector_canonical_json(payload)
        )
        self.assertEqual(len(payload["records"]), 4_205)
        self.assertTrue(
            all(set(record) == {"game_id", "categories", "sources"} for record in payload["records"])
        )
        self.assertEqual(
            self.freezer.sha256_file(path), declaration["sha256"]
        )


if __name__ == "__main__":
    unittest.main()
