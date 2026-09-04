import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
import sys

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compact_value_bfm_fresh_holdout as holdout
import compact_value_bfm_successor as successor


q = successor.qualification


class SuccessorRouteTest(unittest.TestCase):
    def manifest(self):
        value = {
            "routes": {
                "pilot_search_manifests": ["ps-train", "ps-val", "ps-test"],
                "full_search_manifests": ["fs-train", "fs-val", "fs-test"],
                "pilot_rank4_manifests": ["pr-train", "pr-val", "pr-test"],
                "full_rank4_manifests": ["fr-train", "fr-val", "fr-test"],
                "canonical_splits": {
                    "train": ["c0-train", "c1-train", "c2-train"],
                    "validation": ["c0-val", "c1-val", "c2-val"],
                    "test": ["c0-test", "c1-test", "c2-test"],
                },
                "common_adjudicator_manifest": "common-val",
            }
        }
        pairs = {
            "pilot-search-test": "ps-test",
            "full-search-test": "fs-test",
            "pilot-rank4-test": "pr-test",
            "full-rank4-test": "fr-test",
            "canonical-r0-test": "c0-test",
            "canonical-r1-test": "c1-test",
            "canonical-r2-test": "c2-test",
        }
        value["artifacts"] = [
            {"role": f"{role}-{kind}", "relative_path": path}
            for role, manifest_path in pairs.items()
            for kind, path in (
                ("manifest", manifest_path),
                ("npz", manifest_path.replace("test", "test-data") + ".npz"),
            )
        ]
        return value

    def test_safe_routes_exclude_every_protected_route(self):
        manifest = self.manifest()
        protected = successor._protected_routes(manifest)
        routes = successor._safe_base_routes(manifest)
        flattened = {value for values in routes.values() for value in values}
        self.assertFalse(flattened & protected)
        self.assertEqual(routes["common_adjudicator"], ["common-val"])

    def test_protected_route_in_safe_slot_is_rejected(self):
        manifest = self.manifest()
        manifest["routes"]["canonical_splits"]["validation"][0] = "c0-test"
        with self.assertRaisesRegex(successor.SuccessorError, "unsafe"):
            successor._safe_base_routes(manifest)

    def test_declared_label_record_does_not_probe_path(self):
        path = pathlib.Path("/definitely/not/materialized/search-merged.jsonl")
        declared = successor._declared_record(
            {"path": str(path), "bytes": 12, "sha256": "a" * 64, "lines": 20_000},
            "declared fixture",
        )
        self.assertEqual(declared, path)


class PairedCarryForwardTest(unittest.TestCase):
    def row(self, schema, position, split):
        return {
            "schema": schema,
            "position_id": position,
            "root_group_id": f"root-{position}",
            "group_id": f"game-{position}",
            "source": "student-selfplay",
            "split": split,
            "winner": 0,
            "mover": 0,
            "prefix": [],
        }

    def test_paired_audit_binds_identity_and_splits(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            search = root / "search.jsonl"
            rank4 = root / "rank4.jsonl"
            search_rows = [
                self.row(successor.selfsearch.SEARCH_TEACHER_SCHEMA, "a", "train"),
                self.row(successor.selfsearch.SEARCH_TEACHER_SCHEMA, "b", "test"),
            ]
            rank4_rows = [
                self.row(successor.selfsearch.RANK4_TEACHER_SCHEMA, "a", "train"),
                self.row(successor.selfsearch.RANK4_TEACHER_SCHEMA, "b", "test"),
            ]
            search.write_text("".join(json.dumps(row) + "\n" for row in search_rows))
            rank4.write_text("".join(json.dumps(row) + "\n" for row in rank4_rows))
            with mock.patch.object(
                successor.selfsearch.corpus, "sample_from_teacher_row"
            ):
                rows, splits, digest = successor._paired_label_audit(search, rank4)
            self.assertEqual(rows, 2)
            self.assertEqual(splits, {"train": 1, "validation": 0, "test": 1})
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

            rank4_rows[1]["group_id"] = "different"
            rank4.write_text("".join(json.dumps(row) + "\n" for row in rank4_rows))
            with (
                mock.patch.object(
                    successor.selfsearch.corpus, "sample_from_teacher_row"
                ),
                self.assertRaisesRegex(successor.SuccessorError, "identities differ"),
            ):
                successor._paired_label_audit(search, rank4)


class FreshHoldoutContractTest(unittest.TestCase):
    def record(self, path):
        return successor._regular_file_record(path)

    def test_fresh_roots_are_new_all_test_groups_and_body_valid(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            source = root / "roots.json"
            source_tsv = root / "roots.tsv"
            source_tsv.write_text(
                "group_id\tsource\twinner\ttranscript\n"
                "old-root\told\t0\t444444/555555/666666\n"
            )
            q.write_sealed(source, {
                "schema": holdout.corpus.ROOT_SCHEMA,
                "feature_schema": holdout.pack_tool.features.FEATURE_SCHEMA,
                "tool_sha256": {"normalizer": "a" * 64, "features": "b" * 64},
                "exclusion_boundary": {"read_before_candidate_sources": True},
                "accepted": [
                    {"group_id": "root-a", "split": "train"},
                    {"group_id": "root-b", "split": "validation"},
                ],
            })
            plan = {
                "body_sha256": "c" * 64,
                "training": {
                    "roots_manifest": self.record(source),
                    "roots_tsv": self.record(source_tsv),
                },
                "fresh_protected_holdout": {
                    "game_plan_seed": 123,
                    "fresh_root_openings": 3_200,
                },
                "tools": {
                    "opening_generator": self.record(
                        pathlib.Path(holdout.opening_tools.__file__)
                    )
                },
            }
            openings = [
                {
                    "opening_id": f"fresh-{index}",
                    "transcript": "4",
                    "fingerprints": {
                        "canonical": f"canonical-{index}",
                        "identity": f"identity-{index}",
                    },
                }
                for index in range(2)
            ]
            state = types.SimpleNamespace(winner=None, to_move=0)

            def apply_turn(current, player, action):
                current.to_move = 1 - player
                if action == "666666":
                    current.winner = 0

            with (
                mock.patch.object(
                    holdout.opening_tools.reference,
                    "ReplayState",
                    return_value=state,
                ),
                mock.patch.object(
                    holdout.opening_tools.reference,
                    "apply_complete_turn",
                    side_effect=apply_turn,
                ),
                mock.patch.object(
                    holdout.opening_tools,
                    "state_fingerprints",
                    return_value={"canonical": "old", "identity": "old-id"},
                ),
                mock.patch.object(
                    holdout.opening_tools, "generate_openings", return_value=openings
                ) as generated,
            ):
                fresh, fresh_tsv, bank = holdout._fresh_test_roots(
                    plan, root / "holdout"
                )
            value = holdout.pack_tool.load_roots(fresh)
            self.assertEqual({row["split"] for row in value["accepted"]}, {"test"})
            self.assertEqual(
                value["counts"]["split_games"],
                {"train": 0, "validation": 0, "test": 2},
            )
            self.assertEqual(
                {row["group_id"] for row in value["accepted"]},
                {"fresh-protected-root:0000", "fresh-protected-root:0001"},
            )
            self.assertNotIn("old-root", fresh_tsv.read_text())
            self.assertTrue(bank.is_file())
            self.assertEqual(generated.call_args.kwargs["count"], 3_200)
            self.assertEqual(
                generated.call_args.kwargs["excluded_fingerprints"], {"old-id"}
            )

    def test_test_prior_is_rejected_before_packing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)

            def manifest(name, split):
                path = root / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"split": split}))
                return {"manifest": self.record(path)}

            new = manifest("global-repack/search/new", "train")["manifest"]
            q.write_sealed(root / "training-input.json", {
                "schema": successor.TRAINING_INPUT_SCHEMA,
                "new_train_manifest": new,
            })
            plan = {
                "training": {
                    "source_bundle_manifest": {"path": str(root / "bundle.json")},
                    "safe_input_artifacts": {
                        "anchor": [manifest(f"anchor-{index}", "train") for index in range(3)],
                        "canonical_validation": [
                            manifest("validation-0", "validation"),
                            manifest("validation-1", "test"),
                            manifest("validation-2", "validation"),
                        ],
                        "common_adjudicator": [manifest("common", "validation")],
                    }
                }
            }
            with (
                mock.patch.object(
                    holdout.successor, "retired_protected_paths", return_value=set()
                ),
                self.assertRaisesRegex(holdout.HoldoutError, "test shard"),
            ):
                holdout._packing_priors(plan, root)

    def test_fresh_isolation_normalizes_physical_splits_for_validator_roles(self):
        def dataset(split, marker):
            indices = np.asarray(
                [316 + vertex * 57 for vertex in range(105)], dtype="<u2"
            )
            return holdout.compact.Dataset(
                indptr=np.asarray([0, len(indices)], dtype="<i8"),
                indices=indices,
                targets=np.asarray([0.0], dtype="<f4"),
                weights=np.asarray([1.0], dtype="<f4"),
                group_ids=np.asarray(
                    [bytes([marker]) * 32], dtype="V32"
                ),
                split=split,
                source_manifest_sha256=f"{marker:064x}",
                source_npz_sha256=f"{marker + 1:064x}",
                source_route=f"dataset-{marker}",
            )

        loaded = [
            *[dataset("train" if index < 4 else "validation", index + 1)
              for index in range(8)],
            dataset("test", 20),
            dataset("test", 21),
            dataset("test", 22),
        ]
        reports = {
            name: {"shards": {"test": {"manifest": f"{name}.json"}}}
            for name in ("search", "rank4", "canonical")
        }
        with (
            mock.patch.object(
                holdout, "_as_compact_dataset", side_effect=loaded
            ),
            mock.patch.object(
                holdout.compact,
                "validate_unprotected_split_isolation",
                return_value={"passed": True, "protected_tests_opened": False},
            ) as validator,
        ):
            result = holdout._fresh_split_isolation(
                [pathlib.Path(f"prior-{index}.json") for index in range(8)],
                reports,
            )
        new, anchor, common, canonical = validator.call_args.args
        self.assertEqual((new.split, anchor.split), ("train", "train"))
        self.assertEqual((common.split, canonical.split), ("validation", "validation"))
        self.assertTrue(result["fresh_protected_tests_opened"])


if __name__ == "__main__":
    unittest.main()
