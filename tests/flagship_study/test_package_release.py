from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import tempfile
import unittest
import zipfile
from unittest import mock

from benchmarks.flagship_study import package_release, studylib


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _write(path: pathlib.Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _manifest() -> dict:
    labels = {
        "mcts": "Tactical MctsBot",
        "alpha_beta": "Hand-evaluated AlphaBetaBot",
        "jacek_inspired": "Neural alpha-beta (JacekInspiredBot)",
        "rank5_derived": "Rank5DerivedBot — fixed 50k demo profile",
    }
    identifiers = {
        "mcts": "mcts-1000",
        "alpha_beta": "alpha-beta-50k",
        "jacek_inspired": "jacek-20k",
        "rank5_derived": "rank5-fixed-50k",
    }
    configurations = []
    for family, config_id in identifiers.items():
        if family == "mcts":
            settings = {"iterations": 1000}
        else:
            settings = {"max_nodes": 50_000, "max_turn_depth": 6}
        if family == "jacek_inspired":
            settings.update({
                "model_path": "models/value.json",
                "model_sha256": "2" * 64,
            })
        if family == "rank5_derived":
            settings.update({
                "max_turn_depth": 32,
                "original_artifact_sha256": "1" * 64,
            })
        configurations.append({
            "id": config_id,
            "family": family,
            "public_label": labels[family],
            "role": "fixed_comparator" if family == "rank5_derived" else "candidate",
            "settings": settings,
        })
    return {
        "schema_version": "papersoccer.flagship-study-manifest.v2",
        "study": {
            "id": "fixture-study",
            "version": "1.0.0",
            "title": "Fixture study",
            "preregistered_at_utc": "2026-01-01T00:00:00+00:00",
            "frozen": True,
            "rank5_disclaimer": "The derived bot is not the ranked submission.",
        },
        "configurations": configurations,
    }


def _pareto_point(config: dict, selected: bool = True) -> dict:
    fixed = config["family"] == "rank5_derived"
    return {
        "id": config["id"],
        "family": config["family"],
        "fixed": fixed,
        "selected": selected,
        "gate_eligible": True,
        "validation_strength": 0.5 if fixed else 0.6,
        "strength_definition": (
            "defined common-opponent reference level" if fixed else None
        ),
        "validation_strength_pair_bootstrap_95": (
            None if fixed else {"lower": 0.55, "upper": 0.65}
        ),
        "validation_strength_pairs": None if fixed else 20,
        "validation_p95_ms": 20.0,
        "validation_latency_decisions": 100,
        "constrained_pareto_optimal": True,
        "unconstrained_pareto_optimal": True,
    }


def _selection(manifest_hash: str, manifest: dict) -> dict:
    return {
        "schema_version": "papersoccer.flagship-study-selection.v1",
        "manifest_sha256": manifest_hash,
        "test_authorized": True,
        "selected_configurations": {
            "mcts": "mcts-1000",
            "alpha_beta": "alpha-beta-50k",
            "jacek_inspired": "jacek-20k",
        },
        "fixed_rank5_configuration": "rank5-fixed-50k",
        "rank5_latency": {
            "fresh_root_p95_ms": 20.0,
            "all_edge_p95_ms": 10.0,
            "eligible_under_50_ms": True,
        },
        "validation_pareto": [
            _pareto_point(config) for config in manifest["configurations"]
        ],
    }


def _phase(phase: str, manifest_hash: str) -> dict:
    payload = {
        "schema_version": "papersoccer.flagship-study-curated.v1",
        "phase": phase,
        "manifest_sha256": manifest_hash,
        "completeness": {
            "expected_games": 2,
            "completed_games": 2,
            "unique_game_ids": 2,
            "decisions": 12,
            "truncations": 0,
            "operationally_valid": True,
        },
        "binary_games": [{"game_id": f"{phase}-0"}, {"game_id": f"{phase}-1"}],
    }
    if phase != "test":
        return payload
    payload.update({
        "analysis_complete": True,
        "sample_sizes": {
            "games": 2,
            "pairs": 1,
            "opening_depths": [4],
            "bootstrap_resamples": 10,
        },
        "matchups": {
            "test-alpha-beta-vs-jacek": {
                "left_config_id": "alpha-beta-50k",
                "right_config_id": "jacek-20k",
                "games": 2,
                "pairs": 1,
                "left_wins": 0,
                "left_losses": 2,
                "right_wins": 2,
                "right_losses": 0,
                "pairs_won_2_0": 0,
                "pairs_split_1_1": 0,
                "pairs_lost_0_2": 1,
                "mean_pair_score": 0.0,
                "pair_bootstrap_95": {"lower": 0.0, "upper": 0.0},
                "truncations": 0,
                "conclusion": {
                    "classification": "stronger",
                    "stronger_config_id": "jacek-20k",
                },
            },
        },
        "bradley_terry": {
            "method": "fixture",
            "identifiability": "sum_to_zero",
            "resamples": 10,
            "successful_resamples": 10,
            "failed_resamples": {},
            "point_fit": {"converged": True},
            "intervals": {},
        },
        "calibration": {
            "jacek-20k": {
                "phase": "test",
                "samples": 10,
                "decision_count": 10,
                "pair_clusters": 1,
                "excluded": {"truncations": 0},
                "brier_score": 0.2,
                "log_loss": 0.5,
                "pair_cluster_bootstrap_95": {},
                "reliability_bins": [{"bin": 0}],
            },
        },
    })
    return payload


def _repository(root: pathlib.Path) -> pathlib.Path:
    manifest = _manifest()
    manifest_path = root / package_release.MANIFEST_PATH
    _write(
        manifest_path,
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write(
        root / package_release.SELECTION_PATH,
        json.dumps(
            _selection(manifest_hash, manifest), separators=(",", ":"), sort_keys=True
        ) + "\n",
    )
    for phase in ("development", "validation", "test"):
        _write(
            root / package_release.STUDY_ROOT / f"data/{phase}.json",
            json.dumps(
                _phase(phase, manifest_hash), separators=(",", ":"), sort_keys=True
            ) + "\n",
        )
    _write(root / package_release.REPORT_PATH, "# Fixture report\n")
    _write(root / package_release.RELEASE_NOTES_PATH, "# Fixture release notes\n")
    for relative in package_release.LINEAGE_ATTACHMENT_PATHS:
        _write(root / relative, f"fixture lineage attachment: {relative}\n")
    for index, relative in enumerate(package_release.CHART_PATHS):
        _write(root / relative, f"<svg data-index=\"{index}\"></svg>\n")
    _write(
        root / package_release.SUMMARY_JSON_PATH,
        json.dumps({"fixture": True}, indent=2, sort_keys=True) + "\n",
    )
    _write(
        root / package_release.PAIRWISE_CSV_PATH,
        "matchup_id,left_wins,stronger_config_id\nfixture,0,jacek-20k\n",
    )
    _write(
        root / package_release.CONFIGURATIONS_CSV_PATH,
        "config_id,family\nmcts-1000,mcts\nalpha-beta-50k,alpha_beta\n"
        "jacek-20k,jacek_inspired\nrank5-fixed-50k,rank5_derived\n",
    )
    return root


def _generated_summaries(repository: pathlib.Path) -> dict[str, str]:
    return {
        relative.name: (repository / relative).read_text(encoding="utf-8")
        for relative in package_release.SUMMARY_PATHS
    }


class ReleasePackagingTest(unittest.TestCase):
    def test_release_and_lineage_use_one_record_tag(self) -> None:
        self.assertEqual(package_release.SOURCE_TAG, studylib.V4_AUDIT_TAG)

    def test_release_notes_cover_results_provenance_and_reproduction(self) -> None:
        notes = (REPOSITORY_ROOT / package_release.RELEASE_NOTES_PATH).read_text(
            encoding="utf-8"
        )
        normalized = " ".join(notes.split())
        for required in (
            "4,800 decisive test games",
            "2,400 color-swapped pairs",
            "zero truncations",
            "60.4%",
            "51.4%",
            "do not evaluate the authentic ranked submission",
            "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29",
            "models/jacek_article_value_model.json",
            "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084",
            "git checkout --detach flagship-study-v4-record",
            "package_release.py build",
            "package_release.py check",
            "shasum -a 256 -c SHA256SUMS",
        ):
            self.assertIn(required, normalized)

    def test_build_is_deterministic_and_contains_curated_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _repository(pathlib.Path(temporary) / "repository")
            output = repository / "results/releases/flagship-study-v4"
            with mock.patch.object(
                package_release.release_summary,
                "generate_release_files",
                return_value=_generated_summaries(repository),
            ):
                first_hashes = package_release.package_release(
                    repository, output, verify_source_tag=False
                )
                first_assets = {
                    name: (output / name).read_bytes()
                    for name in (
                        package_release.CORE_ARCHIVE_NAME,
                        package_release.DECISION_ARCHIVE_NAME,
                        package_release.CHECKSUMS_NAME,
                    )
                }
                second_hashes = package_release.package_release(
                    repository, output, verify_source_tag=False
                )
            self.assertEqual(first_hashes, second_hashes)
            for name, content in first_assets.items():
                self.assertEqual(content, (output / name).read_bytes())

            with zipfile.ZipFile(
                output / package_release.CORE_ARCHIVE_NAME
            ) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [str(path) for path in package_release.CORE_ARCHIVE_PATHS],
                )
                for info in archive.infolist():
                    self.assertEqual(info.date_time, package_release.FIXED_ZIP_TIMESTAMP)
                    self.assertEqual(
                        info.external_attr >> 16, package_release.ZIP_MODE
                    )
                for relative in package_release.SUMMARY_PATHS:
                    self.assertEqual(
                        archive.read(str(relative)), (repository / relative).read_bytes()
                    )
                self.assertEqual(
                    archive.read(str(package_release.RELEASE_NOTES_PATH)),
                    (repository / package_release.RELEASE_NOTES_PATH).read_bytes(),
                )
                summary_content = archive.read(str(package_release.SUMMARY_JSON_PATH))
                pairwise = list(csv.DictReader(io.TextIOWrapper(
                    archive.open(str(package_release.PAIRWISE_CSV_PATH)),
                    encoding="utf-8",
                )))
                configurations = list(csv.DictReader(io.TextIOWrapper(
                    archive.open(str(package_release.CONFIGURATIONS_CSV_PATH)),
                    encoding="utf-8",
                )))

            self.assertEqual(
                summary_content,
                (repository / package_release.SUMMARY_JSON_PATH).read_bytes(),
            )
            self.assertEqual(pairwise[0]["left_wins"], "0")
            self.assertEqual(pairwise[0]["stronger_config_id"], "jacek-20k")
            self.assertEqual(len(configurations), 4)

            with zipfile.ZipFile(
                output / package_release.DECISION_ARCHIVE_NAME
            ) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [str(path) for path in package_release.DECISION_PATHS],
                )
                self.assertEqual(
                    archive.read(str(package_release.DECISION_PATHS[2])),
                    (repository / package_release.DECISION_PATHS[2]).read_bytes(),
                )

    def test_check_detects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _repository(pathlib.Path(temporary) / "repository")
            output = repository / "results/releases/flagship-study-v4"
            with mock.patch.object(
                package_release.release_summary,
                "generate_release_files",
                return_value=_generated_summaries(repository),
            ):
                package_release.package_release(
                    repository, output, verify_source_tag=False
                )
                _write(repository / package_release.REPORT_PATH, "# Changed report\n")
                with self.assertRaisesRegex(
                    package_release.PackagingError, "content does not match its source"
                ):
                    package_release.verify_release(
                        repository, output, verify_source_tag=False
                    )

    def test_stale_generated_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _repository(pathlib.Path(temporary) / "repository")
            expected = _generated_summaries(repository)
            expected["summary.json"] = "{}\n"
            with mock.patch.object(
                package_release.release_summary,
                "generate_release_files",
                return_value=expected,
            ), self.assertRaisesRegex(
                package_release.PackagingError,
                "stale release summary file.*release_summary.py --write",
            ):
                package_release.package_release(
                    repository,
                    repository / "results/releases/flagship-study-v4",
                    verify_source_tag=False,
                )

    def test_frozen_tag_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _repository(pathlib.Path(temporary) / "repository")
            with mock.patch.object(
                package_release.release_summary,
                "generate_release_files",
                return_value=_generated_summaries(repository),
            ), mock.patch.object(
                package_release, "_git", return_value="a" * 40 + "\n"
            ), mock.patch.object(
                package_release, "_git_blob_sha256", return_value="0" * 64
            ), self.assertRaisesRegex(
                package_release.PackagingError,
                "differs from frozen tag flagship-study-v4-record",
            ):
                package_release.package_release(
                    repository,
                    repository / "results/releases/flagship-study-v4",
                )

    def test_record_tag_verification_includes_lineage_attachments(self) -> None:
        source_hashes = {
            str(relative): f"{index + 1:064x}"
            for index, relative in enumerate(package_release.TAG_IMMUTABLE_PATHS)
        }
        with mock.patch.object(
            package_release, "_git", return_value="a" * 40 + "\n"
        ), mock.patch.object(
            package_release,
            "_git_blob_sha256",
            side_effect=lambda _repository, _tag, relative: source_hashes[
                str(relative)
            ],
        ) as tagged_hash:
            package_release._verify_source_tag(pathlib.Path("/repository"), source_hashes)

        self.assertEqual(
            [call.args[2] for call in tagged_hash.call_args_list],
            list(package_release.TAG_IMMUTABLE_PATHS),
        )

    def test_symlinked_input_is_rejected_even_when_target_is_regular(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = _repository(root / "repository")
            outside = root / "outside-report.md"
            _write(outside, "# Outside report\n")
            report = repository / package_release.REPORT_PATH
            report.unlink()
            report.symlink_to(outside)
            with self.assertRaisesRegex(
                package_release.PackagingError,
                "release input must not use a symlink.*REPORT.md",
            ):
                package_release.package_release(
                    repository,
                    repository / "results/releases/flagship-study-v4",
                    verify_source_tag=False,
                )

    def test_missing_inputs_have_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary) / "repository"
            repository.mkdir()
            with self.assertRaisesRegex(
                package_release.PackagingError,
                r"missing release input\(s\).*manifest\.json.*test\.json",
            ):
                package_release.package_release(
                    repository,
                    repository / "results/releases/flagship-study-v4",
                    verify_source_tag=False,
                )

    def test_manifest_mismatch_is_rejected_before_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = _repository(pathlib.Path(temporary) / "repository")
            selection_path = repository / package_release.SELECTION_PATH
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["manifest_sha256"] = "f" * 64
            _write(selection_path, json.dumps(selection))
            with self.assertRaisesRegex(
                package_release.PackagingError,
                "selection lock does not match the packaged manifest",
            ):
                package_release.package_release(
                    repository,
                    repository / "results/releases/flagship-study-v4",
                    verify_source_tag=False,
                )


if __name__ == "__main__":
    unittest.main()
