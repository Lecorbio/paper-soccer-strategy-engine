import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402

try:
    import numpy as np
    import jacek_rebuild_corpus as rebuild
    import jacek_replay_rebuild as rebuild_workflow
    import jacek_replay_train as training
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    rebuild = None
    rebuild_workflow = None
    training = None


FIXTURE_COUNTS = {"train": 1, "validation": 1, "test": 1}


def active(token: int) -> tuple[int, ...]:
    value = tuple(
        sorted(
            features.EDGE_COUNT
            + vertex * features.VERTEX_CATEGORIES
            + ((token + vertex * 7) % features.VERTEX_CATEGORIES)
            for vertex in range(features.VERTEX_COUNT)
        )
    )
    return features.validate_active(value)


def sample(
    token: int,
    target: float,
    group: str,
    *,
    reflected: bool = False,
) -> corpus.LabeledSample:
    value = active(token)
    if reflected:
        value = features.reflect_active(value)
    return corpus.LabeledSample(value, target, 1.0, group)


@unittest.skipIf(np is None, "research tests require requirements-research.txt")
class JacekRebuildCorpusTests(unittest.TestCase):
    def write_shard(
        self,
        root: pathlib.Path,
        name: str,
        split: str,
        samples: list[corpus.LabeledSample],
    ) -> pathlib.Path:
        _npz, manifest, _value = training.write_csr_shard(
            root / "sources" / name,
            split,
            samples,
            provenance={"fixture": name},
        )
        return manifest

    def sources(
        self,
        root: pathlib.Path,
        *,
        adjudicator_v6_token: int = 11,
        adjudicator_v6_group: str = "validation-shared-a",
    ) -> dict[str, list[pathlib.Path]]:
        return {
            "canonical_train": [
                self.write_shard(
                    root, "canonical-train", "train", [sample(0, 0.0, "train-root")]
                )
            ],
            "canonical_validation": [
                self.write_shard(
                    root,
                    "canonical-validation",
                    "validation",
                    [sample(10, 0.0, "validation-root")],
                )
            ],
            "canonical_test": [
                self.write_shard(
                    root, "canonical-test", "test", [sample(20, 0.0, "test-root")]
                )
            ],
            "v5_search_train": [
                self.write_shard(
                    root,
                    "v5-search",
                    "train",
                    [sample(1, 0.1, "shared-train-a"), sample(2, 0.2, "v5-only")],
                )
            ],
            "v6_search_train": [
                self.write_shard(
                    root,
                    "v6-search",
                    "train",
                    [
                        sample(1, 0.9, "shared-train-a", reflected=True),
                        sample(3, 0.3, "v6-only"),
                        sample(3, 0.4, "v6-duplicate", reflected=True),
                    ],
                )
            ],
            "v5_rank4_train": [
                self.write_shard(
                    root,
                    "v5-rank4",
                    "train",
                    [sample(1, -0.1, "shared-train-a"), sample(4, -0.2, "rank-v5")],
                )
            ],
            "v6_rank4_train": [
                self.write_shard(
                    root,
                    "v6-rank4",
                    "train",
                    [sample(1, -0.9, "shared-train-a", reflected=True)],
                )
            ],
            "v5_adjudicator_validation": [
                self.write_shard(
                    root,
                    "v5-adjudicator",
                    "validation",
                    [
                        sample(11, 0.1, "validation-shared-a"),
                        sample(12, 0.2, "adjudicator-v5"),
                    ],
                )
            ],
            "v6_adjudicator_validation": [
                self.write_shard(
                    root,
                    "v6-adjudicator",
                    "validation",
                    [
                        sample(
                            adjudicator_v6_token,
                            0.8,
                            adjudicator_v6_group,
                            reflected=adjudicator_v6_token == 11,
                        ),
                        sample(13, 0.3, "adjudicator-v6"),
                    ],
                )
            ],
        }

    def freeze(
        self,
        root: pathlib.Path,
        sources: dict[str, list[pathlib.Path]] | None = None,
    ) -> tuple[pathlib.Path, dict]:
        return rebuild.freeze_rebuild_corpus(
            root / "frozen",
            **(sources or self.sources(root)),
            expected_canonical_counts=FIXTURE_COUNTS,
        )

    def test_freezes_deduplicated_shards_with_v6_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path, manifest = self.freeze(root)
            loaded = rebuild.validate_rebuild_manifest(
                path, expected_canonical_counts=FIXTURE_COUNTS
            )

            self.assertEqual(manifest["schema"], rebuild.MANIFEST_SCHEMA)
            self.assertEqual(path.stem, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(
                manifest["deduplicated"]["search"]["selection"]["input_rows"], 5
            )
            self.assertEqual(
                manifest["deduplicated"]["search"]["selection"]["retained_rows"],
                3,
            )
            search = training.load_csr_shard(
                loaded.training_manifest_paths("search")[0]
            )
            by_fingerprint = {
                corpus.canonical_feature_fingerprint(
                    search.active(row).tolist()
                ): float(search.targets[row])
                for row in range(len(search))
            }
            self.assertAlmostEqual(
                by_fingerprint[corpus.canonical_feature_fingerprint(active(1))], 0.9
            )
            v6, v5 = manifest["deduplicated"]["search"]["selection"]["sources"]
            self.assertEqual(v6["campaign"], "v6")
            self.assertEqual(v6["retained_row_indices"], [0, 1])
            self.assertEqual(v5["campaign"], "v5")
            self.assertEqual(v5["retained_row_indices"], [1])

            rank4 = training.load_csr_shard(loaded.training_manifest_paths("rank4")[0])
            rank_by_fingerprint = {
                corpus.canonical_feature_fingerprint(rank4.active(row).tolist()): float(
                    rank4.targets[row]
                )
                for row in range(len(rank4))
            }
            self.assertAlmostEqual(
                rank_by_fingerprint[corpus.canonical_feature_fingerprint(active(1))],
                -0.9,
            )
            self.assertEqual(len(loaded.validation_manifest_paths("search")), 1)

    def test_interfaces_never_expose_the_protected_canonical_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path, manifest = self.freeze(root)
            loaded = rebuild.load_rebuild_manifest(
                path, expected_canonical_counts=FIXTURE_COUNTS
            )
            protected = set(loaded.protected_test_manifest_paths)
            self.assertEqual(len(protected), 1)
            self.assertFalse(manifest["protected_test"]["training_eligible"])
            self.assertFalse(manifest["protected_test"]["selection_eligible"])
            for channel in ("search", "rank4"):
                exposed = set(loaded.training_manifest_paths(channel))
                exposed.update(loaded.anchor_manifest_paths(channel))
                exposed.update(loaded.validation_manifest_paths(channel))
                exposed.update(loaded.retention_validation_manifest_paths(channel))
                self.assertTrue(protected.isdisjoint(exposed))
                self.assertEqual(
                    loaded.anchor_manifest_paths(channel),
                    (self.sources_from_manifest(path, manifest, "canonical_train")[0],),
                )

    def test_freeze_and_validation_never_load_protected_test_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sources = self.sources(root)
            protected = sources["canonical_test"][0].resolve()
            protected_manifest = json.loads(protected.read_text())
            protected_npz = (protected.parent / protected_manifest["npz"]).resolve()
            original = training.load_csr_shard
            original_np_load = rebuild.np.load

            def guarded(path):
                if pathlib.Path(path).resolve() == protected:
                    raise AssertionError("protected test target loader was called")
                return original(path)

            def guarded_np(path, *args, **kwargs):
                if pathlib.Path(path).resolve() == protected_npz:
                    raise AssertionError("protected test arrays were decoded")
                return original_np_load(path, *args, **kwargs)

            with mock.patch.object(
                training, "load_csr_shard", side_effect=guarded
            ), mock.patch.object(rebuild.np, "load", side_effect=guarded_np):
                path, _manifest = self.freeze(root, sources)
                rebuild.validate_rebuild_manifest(
                    path, expected_canonical_counts=FIXTURE_COUNTS
                )

    @staticmethod
    def sources_from_manifest(
        path: pathlib.Path, manifest: dict, key: str
    ) -> tuple[pathlib.Path, ...]:
        return tuple(
            (path.parent / identity["manifest_path"]).resolve()
            for identity in manifest["inputs"][key]
        )

    def test_rejects_fingerprint_leakage_across_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sources = self.sources(root, adjudicator_v6_token=0)
            with self.assertRaisesRegex(ValueError, "fingerprint leaks"):
                self.freeze(root, sources)

    def test_rejects_root_group_leakage_across_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sources = self.sources(
                root,
                adjudicator_v6_token=14,
                adjudicator_v6_group="train-root",
            )
            with self.assertRaisesRegex(ValueError, "root-group identity leaks"):
                self.freeze(root, sources)

    def test_exact_canonical_counts_are_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(ValueError, "canonical train row count"):
                rebuild.freeze_rebuild_corpus(
                    root / "frozen", **self.sources(root)
                )
            self.assertEqual(
                rebuild.EXPECTED_CANONICAL_COUNTS,
                {"train": 997_914, "validation": 110_004, "test": 121_052},
            )

    def test_freeze_is_deterministic_and_reload_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sources = self.sources(root)
            path, manifest = self.freeze(root, sources)
            second_path, second = self.freeze(root, sources)
            self.assertEqual(second_path, path)
            self.assertEqual(second, manifest)
            rebuild.validate_manifest(path, expected_canonical_counts=FIXTURE_COUNTS)

            generated = manifest["deduplicated"]["search"]["shard"]
            npz_path = (path.parent / generated["npz_path"]).resolve()
            npz_path.write_bytes(npz_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                rebuild.validate_rebuild_manifest(
                    path, expected_canonical_counts=FIXTURE_COUNTS
                )

    def test_recomputes_row_selection_even_for_rehashed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path, manifest = self.freeze(root)
            forged = json.loads(json.dumps(manifest))
            source = forged["deduplicated"]["search"]["selection"]["sources"][0]
            source["retained_row_indices"] = [2]
            selection = forged["deduplicated"]["search"]["selection"]
            forged["deduplicated"]["search"]["selection_sha256"] = hashlib.sha256(
                rebuild.canonical_json_bytes(selection)
            ).hexdigest()
            forged.pop("body_sha256")
            forged["body_sha256"] = hashlib.sha256(
                rebuild.canonical_json_bytes(forged)
            ).hexdigest()
            payload = rebuild.canonical_json_bytes(forged)
            forged_path = path.parent / f"{hashlib.sha256(payload).hexdigest()}.json"
            forged_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "row selection changed"):
                rebuild.validate_rebuild_manifest(
                    forged_path, expected_canonical_counts=FIXTURE_COUNTS
                )

    def test_shallow_runtime_loader_binds_deep_receipt_and_generated_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path, manifest = self.freeze(root)
            loaded = rebuild_workflow.load_frozen_rebuild_corpus(
                path, expected_canonical_counts=FIXTURE_COUNTS
            )
            self.assertEqual(loaded.manifest, manifest)

            forged = json.loads(json.dumps(manifest))
            selection = forged["deduplicated"]["search"]["selection"]
            selection["sources"][0]["retained_row_indices"] = [2]
            forged["deduplicated"]["search"]["selection_sha256"] = hashlib.sha256(
                rebuild.canonical_json_bytes(selection)
            ).hexdigest()
            forged.pop("body_sha256")
            forged["body_sha256"] = hashlib.sha256(
                rebuild.canonical_json_bytes(forged)
            ).hexdigest()
            payload = rebuild.canonical_json_bytes(forged)
            forged_path = path.parent / f"{hashlib.sha256(payload).hexdigest()}.json"
            forged_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                rebuild_workflow.load_frozen_rebuild_corpus(
                    forged_path, expected_canonical_counts=FIXTURE_COUNTS
                )


if __name__ == "__main__":
    unittest.main()
