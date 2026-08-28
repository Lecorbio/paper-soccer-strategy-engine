from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import jacek_selfsearch_exploratory_full as full  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class TinyBundleFixture:
    """Synthetic allowlisted inputs for exercising the real bundle validator."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.sources = root / "synthetic-sources"
        self.output = root / "output"
        self.sources.mkdir()
        self.artifacts: list[full.ImportArtifact] = []
        self.routes: dict[str, object] = {}

        launch = self._artifact(
            "pilot-launch", "pilot/source/launch.json", b'{"fixture":"launch"}\n'
        )
        lineage = self._artifact(
            "pilot-lineage", "pilot/source/lineage.json", b'{"fixture":"lineage"}\n'
        )
        summary = self._artifact(
            "pilot-summary", "pilot/source/summary.json", b'{"fixture":"summary"}\n'
        )
        decision = self._artifact(
            "pilot-decision", "pilot/evidence/decision.json", b'{"fixture":"decision"}\n'
        )
        actor = self._artifact(
            "pilot-teacher-actor", "pilot/actor/teacher.runtime", b"tiny actor\n"
        )
        actor_manifest = self._artifact(
            "pilot-teacher-manifest",
            "pilot/actor/teacher.runtime.json",
            b'{"fixture":"actor-manifest"}\n',
        )
        reference = self._artifact(
            "original-retention-reference",
            "pilot/reference/original.runtime",
            b"tiny reference\n",
        )
        roots_tsv = self._artifact(
            "canonical-roots-tsv", "canonical/roots/replay-roots.tsv", b"root\n"
        )
        roots_manifest = self._artifact(
            "canonical-roots-manifest",
            "canonical/roots/replay-roots.json",
            b'{"fixture":"roots"}\n',
        )
        opening_pilot = self._artifact(
            "opening-exclusion-pilot", "openings/pilot.tsv", b"pilot opening\n"
        )
        opening_canonical = [
            self._artifact(
                f"opening-exclusion-canonical-{index}",
                f"openings/canonical-{index}.tsv",
                f"canonical opening {index}\n".encode(),
            )
            for index in range(6)
        ]

        pilot_search = [
            self._shard(f"pilot-search-{index}", "pilot/shards/search")
            for index in range(3)
        ]
        pilot_rank4 = [
            self._shard(f"pilot-rank4-{index}", "pilot/shards/rank4")
            for index in range(3)
        ]
        canonical = [
            self._shard(
                f"canonical-r{round_index}-{split}", "canonical/shards"
            )
            for round_index in range(3)
            for split in ("train", "validation", "test")
        ]

        self.routes = {
            "actor": actor.relative_path,
            "original_retention_reference": reference.relative_path,
            "diversity_reference": reference.relative_path,
            "pilot_launch": launch.relative_path,
            "pilot_lineage": lineage.relative_path,
            "pilot_summary": summary.relative_path,
            "pilot_decision": decision.relative_path,
            "pilot_actor_manifest": actor_manifest.relative_path,
            "roots_tsv": roots_tsv.relative_path,
            "roots_manifest": roots_manifest.relative_path,
            "opening_exclusions": [
                opening_pilot.relative_path,
                *(artifact.relative_path for artifact in opening_canonical),
            ],
            "pilot_search_manifests": [artifact.relative_path for artifact in pilot_search],
            "pilot_rank4_manifests": [artifact.relative_path for artifact in pilot_rank4],
            "canonical_prior_manifests": [artifact.relative_path for artifact in canonical],
        }
        self.constants = {
            "SOURCE_SUMMARY_SHA256": summary.sha256,
            "SOURCE_DECISION_SHA256": decision.sha256,
            "TEACHER_ACTOR_SHA256": actor.sha256,
            "RETENTION_REFERENCE_SHA256": reference.sha256,
        }
        self.source_fingerprints = {
            "source_summary_sha256": summary.sha256,
            "source_decision_sha256": decision.sha256,
            "source_launch_sha256": launch.sha256,
            "source_lineage_sha256": lineage.sha256,
            "teacher_actor_sha256": actor.sha256,
            "original_retention_reference_sha256": reference.sha256,
            "canonical_manifest_count": 9,
            "pilot_manifest_count": 6,
            "opening_exclusion_count": 7,
        }
        self.override_truth = {
            "pilot_passed": False,
            "pilot_20_ms_passed": False,
            "bypassed_errors": list(full.EXACT_BYPASSED_ERRORS),
        }

    def _artifact(
        self, role: str, relative_path: str, payload: bytes
    ) -> full.ImportArtifact:
        source = self.sources / f"source-{len(self.artifacts):02d}.bin"
        source.write_bytes(payload)
        artifact = full._record_artifact(
            role=role, source=source, relative_path=relative_path
        )
        self.artifacts.append(artifact)
        return artifact

    def _shard(self, role: str, directory: str) -> full.ImportArtifact:
        npz_payload = f"tiny npz {role}\n".encode()
        npz_name = f"{_sha256(npz_payload)}.npz"
        self._artifact(
            f"{role}-npz", f"{directory}/{npz_name}", npz_payload
        )
        manifest_payload = json.dumps(
            {"npz": npz_name, "npz_sha256": _sha256(npz_payload)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        manifest_name = f"{_sha256(manifest_payload)}.json"
        return self._artifact(
            f"{role}-manifest",
            f"{directory}/{manifest_name}",
            manifest_payload,
        )

    @contextlib.contextmanager
    def patched_constants(self):
        with mock.patch.multiple(full, **self.constants):
            yield

    def import_bundle(self) -> tuple[pathlib.Path, dict[str, object]]:
        with self.patched_constants():
            manifest = full.import_input_bundle(
                output=self.output,
                artifacts=self.artifacts,
                routes=self.routes,
                override_truth=self.override_truth,
                source_fingerprints=self.source_fingerprints,
            )
        return self.output / "input-bundle/bundle-manifest.json", manifest


class LargeTeacherInputBundleTests(unittest.TestCase):
    def test_import_is_atomic_and_failed_staging_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyBundleFixture(pathlib.Path(temporary))
            with mock.patch.object(
                full,
                "_copy_import_artifact",
                side_effect=RuntimeError("synthetic interrupted copy"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted copy"):
                    full.import_input_bundle(
                        output=fixture.output,
                        artifacts=fixture.artifacts,
                        routes=fixture.routes,
                        override_truth=fixture.override_truth,
                        source_fingerprints=fixture.source_fingerprints,
                    )

            self.assertFalse((fixture.output / "input-bundle").exists())
            self.assertEqual(list(fixture.output.glob(".input-bundle-*")), [])

    def test_bundle_inventory_is_exact_and_copied_bytes_are_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyBundleFixture(pathlib.Path(temporary))
            manifest_path, manifest = fixture.import_bundle()
            bundle_root = manifest_path.parent

            expected = {
                record["relative_path"] for record in manifest["artifacts"]
            } | {"bundle-manifest.json"}
            actual = {
                path.relative_to(bundle_root).as_posix()
                for path in bundle_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, expected)

            unexpected = bundle_root / "unregistered.bin"
            unexpected.write_bytes(b"not allowlisted")
            with fixture.patched_constants():
                with self.assertRaisesRegex(ValueError, "inventory is not exact"):
                    full.validate_input_bundle(manifest_path)
            unexpected.unlink()

            actor_path = bundle_root / str(fixture.routes["actor"])
            actor_path.write_bytes(b"tampered copied actor\n")
            with fixture.patched_constants():
                with self.assertRaisesRegex(ValueError, "copied input changed"):
                    full.validate_input_bundle(manifest_path)

    def test_copied_bundle_validation_is_independent_of_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyBundleFixture(pathlib.Path(temporary))
            manifest_path, frozen_manifest = fixture.import_bundle()
            source_root_text = str(fixture.sources.resolve())
            self.assertNotIn(source_root_text, manifest_path.read_text())

            shutil.rmtree(fixture.sources)
            self.assertFalse(fixture.sources.exists())
            with fixture.patched_constants():
                self.assertEqual(
                    full.validate_input_bundle(manifest_path), frozen_manifest
                )

    def test_protected_source_markers_are_rejected_before_snapshotting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for marker in full.FORBIDDEN_PATH_MARKERS:
                with self.subTest(marker=marker):
                    source = root / marker / "synthetic.bin"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(b"synthetic, not protected data")
                    with mock.patch.object(full, "artifact_snapshot") as snapshot:
                        with self.assertRaisesRegex(
                            ValueError, "outside the import allowlist"
                        ):
                            full._record_artifact(
                                role="synthetic",
                                source=source,
                                relative_path="safe/synthetic.bin",
                            )
                    snapshot.assert_not_called()

    def test_rehashed_manifest_cannot_admit_sealed_or_blind_paths(self) -> None:
        for marker in full.FORBIDDEN_PATH_MARKERS:
            with self.subTest(marker=marker):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = TinyBundleFixture(pathlib.Path(temporary))
                    manifest_path, manifest = fixture.import_bundle()
                    bundle_root = manifest_path.parent

                    body = dict(manifest)
                    body.pop("body_sha256")
                    record = next(
                        item
                        for item in body["artifacts"]
                        if item["role"] == "pilot-launch"
                    )
                    original = bundle_root / record["relative_path"]
                    protected_relative = f"pilot/{marker}/launch.json"
                    protected = bundle_root / protected_relative
                    protected.parent.mkdir(parents=True)
                    original.replace(protected)
                    record["relative_path"] = protected_relative
                    full._atomic_json(manifest_path, full._body_hashed(body))

                    with fixture.patched_constants():
                        with self.assertRaisesRegex(
                            ValueError, "protected path entered"
                        ):
                            full.validate_input_bundle(manifest_path)


if __name__ == "__main__":
    unittest.main()
