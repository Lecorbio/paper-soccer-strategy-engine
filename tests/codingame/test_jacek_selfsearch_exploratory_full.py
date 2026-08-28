import copy
import dataclasses
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import sys


TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import jacek_selfsearch_exploratory_full as exploratory_full
import jacek_selfsearch_workflow as selfsearch
from jacek_replay_workflow import artifact_snapshot


class ExploratoryFullWorkflowTests(unittest.TestCase):
    def _valid_override_decision(self) -> dict:
        primary = {
            "games": 600,
            "wins": 324,
            "colors": [155, 169],
            "illegal": 0,
            "unfinished": 0,
        }
        external = {
            "games": 600,
            "wins": 306,
            "colors": [143, 163],
            "illegal": 0,
            "unfinished": 0,
        }
        return {
            "schema": selfsearch.PILOT_DECISION_SCHEMA,
            "eligible_for_full": False,
            "errors": list(exploratory_full.EXACT_BYPASSED_ERRORS),
            "counts": {
                "matched": dict(primary),
                "incumbent": dict(primary),
                "rank4": dict(external),
                "jacek-nn": dict(external),
            },
            "candidate_p99_ms": 25.0,
            "uncontended_max_ms": 999.0,
            "anchor_candidate": {
                "sign_accuracy": 0.8,
                "weighted_huber": 1.0,
            },
            "anchor_incumbent": {
                "sign_accuracy": 0.8,
                "weighted_huber": 1.0,
            },
        }

    @staticmethod
    def _write(path: pathlib.Path, content: bytes = b"fixture") -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _runtime_launch(self, bundle_root: pathlib.Path):
        bundle_manifest = self._write(bundle_root / "bundle-manifest.json")
        actor = self._write(bundle_root / "pilot/actor/search.runtime", b"actor")
        reference = self._write(
            bundle_root / "pilot/reference/original.runtime", b"reference"
        )
        roots_tsv = self._write(bundle_root / "canonical/roots.tsv")
        roots_manifest = self._write(bundle_root / "canonical/roots.json")

        canonical: dict[str, list[str]] = {}
        for split in ("train", "validation", "test"):
            canonical[split] = [
                f"canonical/shards/r{round_index}-{split}.json"
                for round_index in range(3)
            ]
            for relative in canonical[split]:
                self._write(bundle_root / relative)
        canonical_prior = [
            canonical[split][round_index]
            for round_index in range(3)
            for split in ("train", "validation", "test")
        ]
        pilot_search = [f"pilot/search/{index}.json" for index in range(3)]
        pilot_rank4 = [f"pilot/rank4/{index}.json" for index in range(3)]
        exclusions = [f"openings/bank-{index}.tsv" for index in range(7)]
        for relative in [*pilot_search, *pilot_rank4, *exclusions]:
            self._write(bundle_root / relative)

        routes = {
            "actor": actor.relative_to(bundle_root).as_posix(),
            "diversity_reference": reference.relative_to(bundle_root).as_posix(),
            "original_retention_reference": (
                reference.relative_to(bundle_root).as_posix()
            ),
            "roots_tsv": roots_tsv.relative_to(bundle_root).as_posix(),
            "roots_manifest": roots_manifest.relative_to(bundle_root).as_posix(),
            "canonical_splits": canonical,
            "canonical_prior_manifests": canonical_prior,
            "pilot_search_manifests": pilot_search,
            "pilot_rank4_manifests": pilot_rank4,
            "opening_exclusions": exclusions,
        }
        launch = {
            "input_bundle": artifact_snapshot(bundle_manifest),
            "input_bundle_record": {"routes": routes, "artifacts": []},
        }
        return (
            launch,
            routes,
            hashlib.sha256(actor.read_bytes()).hexdigest(),
            hashlib.sha256(reference.read_bytes()).hexdigest(),
        )

    def test_teacher_only_override_records_exactly_the_two_primary_failures(self):
        decision = self._valid_override_decision()
        original = copy.deepcopy(decision)

        result = exploratory_full.validate_teacher_only_override(decision)

        self.assertEqual(decision, original)
        self.assertEqual(
            result["bypassed_errors"],
            exploratory_full.EXACT_BYPASSED_ERRORS,
        )
        self.assertFalse(result["pilot_passed"])
        self.assertFalse(result["pilot_20_ms_passed"])
        self.assertFalse(result["legality_bypassed"])
        self.assertFalse(result["completion_bypassed"])
        self.assertFalse(result["external_strength_bypassed"])
        self.assertFalse(result["retention_bypassed"])
        self.assertFalse(result["p99_bypassed"])
        self.assertFalse(result["uncontended_latency_bypassed"])

    def test_teacher_only_override_rejects_every_nonexact_error_roster(self):
        cases = {
            "missing": [exploratory_full.EXACT_BYPASSED_ERRORS[0]],
            "reordered": list(reversed(exploratory_full.EXACT_BYPASSED_ERRORS)),
            "extra": [*exploratory_full.EXACT_BYPASSED_ERRORS, "other failure"],
            "different": ["other failure", "another failure"],
        }
        for name, errors in cases.items():
            with self.subTest(name=name):
                decision = self._valid_override_decision()
                decision["errors"] = errors
                with self.assertRaisesRegex(
                    ValueError, "exact two recorded errors"
                ):
                    exploratory_full.validate_teacher_only_override(decision)

    def test_teacher_only_override_rejects_non_primary_gate_failures(self):
        mutations = {
            "legality": lambda value: value["counts"]["matched"].update(
                illegal=1
            ),
            "completion": lambda value: value["counts"]["matched"].update(
                unfinished=1
            ),
            "external-strength": lambda value: value["counts"]["rank4"].update(
                wins=305
            ),
            "p99": lambda value: value.update(candidate_p99_ms=25.001),
            "uncontended": lambda value: value.update(
                uncontended_max_ms=1_000.0
            ),
            "retention": lambda value: value["anchor_candidate"].update(
                sign_accuracy=0.79
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                decision = self._valid_override_decision()
                mutate(decision)
                with self.assertRaises(ValueError):
                    exploratory_full.validate_teacher_only_override(decision)

    def test_full_spec_has_exact_selfsearch_parity_except_campaign_identity(self):
        source_before = dataclasses.asdict(selfsearch.FULL_SPEC)
        expected = copy.deepcopy(source_before)
        expected["campaign_id"] = exploratory_full.FULL_CAMPAIGN_ID
        expected["configuration"]["campaign_id"] = (
            exploratory_full.FULL_CAMPAIGN_ID
        )

        record = exploratory_full._full_spec_record()

        self.assertEqual(record, expected)
        self.assertEqual(dataclasses.asdict(selfsearch.FULL_SPEC), source_before)
        self.assertEqual(sum(record["quotas"].values()), 10_000)
        self.assertEqual(record["configuration"]["games"], 10_000)
        self.assertEqual(record["pairs"], 500)
        self.assertEqual(record["gate_time_ms"], 980)
        self.assertEqual(record["bank_classification"], "final")

    def test_runtime_routes_stay_local_and_preserve_cumulative_order(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = pathlib.Path(directory) / "input-bundle"
            launch, routes, actor_sha256, reference_sha256 = (
                self._runtime_launch(bundle_root)
            )
            with (
                mock.patch.object(
                    exploratory_full, "TEACHER_ACTOR_SHA256", actor_sha256
                ),
                mock.patch.object(
                    exploratory_full,
                    "RETENTION_REFERENCE_SHA256",
                    reference_sha256,
                ),
            ):
                resolved = exploratory_full._resolve_runtime_routes(launch)

            resolved_root = bundle_root.resolve()
            self.assertEqual(
                resolved["canonical_prior_manifests"],
                tuple(
                    resolved_root / path
                    for path in routes["canonical_prior_manifests"]
                ),
            )
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    resolved["canonical_splits"][split],
                    tuple(
                        resolved_root / path
                        for path in routes["canonical_splits"][split]
                    ),
                )
            self.assertEqual(
                resolved["pilot_search_manifests"],
                tuple(
                    resolved_root / path
                    for path in routes["pilot_search_manifests"]
                ),
            )
            self.assertEqual(
                resolved["pilot_rank4_manifests"],
                tuple(
                    resolved_root / path
                    for path in routes["pilot_rank4_manifests"]
                ),
            )
            for path in (
                resolved["actor"],
                resolved["diversity_reference"],
                *resolved["canonical_prior_manifests"],
                *resolved["opening_exclusions"],
            ):
                path.relative_to(resolved_root)

    def test_resume_requires_flag_and_rejects_rehashed_binding_change(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / exploratory_full.CAMPAIGN_ID
            output.mkdir()
            launch_path = self._write(output / "full-launch.json", b"{}")
            bundle = self._write(output / "input-bundle/bundle-manifest.json")
            override = self._write(
                output / "input-bundle/pilot/teacher-only-override.json"
            )
            launch = {
                "input_bundle": artifact_snapshot(bundle),
                "teacher_only_override": artifact_snapshot(override),
            }
            exploratory_full._prepare_run_start(
                output=output,
                launch_path=launch_path,
                launch=launch,
                resume=False,
            )
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                exploratory_full._prepare_run_start(
                    output=output,
                    launch_path=launch_path,
                    launch=launch,
                    resume=False,
                )

            run_start_path = output / "run-start.json"
            tampered = exploratory_full._load_json(run_start_path, "run start")
            tampered.pop("body_sha256")
            tampered["game_workers"] = 9
            exploratory_full._atomic_json(
                run_start_path, exploratory_full._body_hashed(tampered)
            )
            with self.assertRaisesRegex(ValueError, "resume binding changed"):
                exploratory_full._prepare_run_start(
                    output=output,
                    launch_path=launch_path,
                    launch=launch,
                    resume=True,
                )

    def _terminal_fixture(self, output: pathlib.Path):
        output.mkdir(parents=True)
        launch_path = self._write(output / "full-launch.json", b"launch")
        run_start_path = self._write(output / "run-start.json", b"start")
        bundle_root = output / "input-bundle"
        self._write(bundle_root / "pilot/teacher-only-override.json")
        pilot_search = [
            self._write(bundle_root / f"pilot/search/{index}.json")
            for index in range(3)
        ]
        pilot_rank4 = [
            self._write(bundle_root / f"pilot/rank4/{index}.json")
            for index in range(3)
        ]
        exploratory_full._atomic_json(
            bundle_root / "bundle-manifest.json",
            {
                "routes": {
                    "pilot_search_manifests": [
                        path.relative_to(bundle_root).as_posix()
                        for path in pilot_search
                    ],
                    "pilot_rank4_manifests": [
                        path.relative_to(bundle_root).as_posix()
                        for path in pilot_rank4
                    ],
                }
            },
        )
        full_directory = output / "full"
        runtime = self._write(full_directory / "search.runtime", b"runtime")
        manifest = self._write(full_directory / "search.runtime.json", b"manifest")
        decision_path = self._write(full_directory / "decision.json", b"decision")
        search = [
            self._write(full_directory / f"search-{index}.json")
            for index in range(3)
        ]
        rank4 = [
            self._write(full_directory / f"rank4-{index}.json")
            for index in range(3)
        ]
        full = {
            "profile": "full",
            "campaign_id": exploratory_full.FULL_CAMPAIGN_ID,
            "search_runtime": str(runtime),
            "search_manifest": str(manifest),
            "decision_path": str(decision_path),
            "search_new_manifests": [str(path) for path in search],
            "rank4_new_manifests": [str(path) for path in rank4],
        }
        return (
            launch_path,
            run_start_path,
            full,
            pilot_search,
            search,
            pilot_rank4,
            rank4,
        )

    def test_failed_terminal_summary_never_creates_student_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "failed"
            launch_path, run_start_path, full, *_manifests = self._terminal_fixture(
                output
            )
            summary = exploratory_full._write_terminal_summary(
                output=output,
                launch_path=launch_path,
                run_start_path=run_start_path,
                full=full,
                decision={
                    "eligible_for_local_publication": False,
                    "errors": ["matched final gate failed"],
                },
            )

            self.assertEqual(summary["terminal"], "full-rejected")
            self.assertFalse(summary["student_training_eligible"])
            self.assertIsNone(summary["teacher_candidate_accepted"])
            self.assertIsNone(summary["compact_student_handoff"])
            self.assertFalse((output / "teacher-candidate-accepted.json").exists())
            self.assertFalse((output / "compact-student-handoff.json").exists())
            self.assertEqual(
                exploratory_full._validate_existing_summary(
                    output / "final-summary.json"
                ),
                summary,
            )
    def test_passing_terminal_summary_writes_local_teacher_and_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "accepted"
            (
                launch_path,
                run_start_path,
                full,
                pilot_search,
                search,
                pilot_rank4,
                rank4,
            ) = self._terminal_fixture(output)
            summary = exploratory_full._write_terminal_summary(
                output=output,
                launch_path=launch_path,
                run_start_path=run_start_path,
                full=full,
                decision={"eligible_for_local_publication": True, "errors": []},
            )

            self.assertEqual(summary["terminal"], "teacher-candidate-accepted")
            self.assertTrue(summary["student_training_eligible"])
            self.assertFalse(summary["canonical_promotion_eligible"])
            acceptance = exploratory_full._load_json(
                output / "teacher-candidate-accepted.json", "acceptance"
            )
            handoff = exploratory_full._load_json(
                output / "compact-student-handoff.json", "handoff"
            )
            self.assertEqual(acceptance["classification"], "local-teacher-candidate")
            self.assertFalse(acceptance["pilot_passed"])
            self.assertFalse(acceptance["pilot_20_ms_passed"])
            self.assertEqual(
                [record["path"] for record in handoff["pilot_search_manifests"]],
                [str(path.resolve()) for path in pilot_search],
            )
            self.assertEqual(
                [record["path"] for record in handoff["full_search_manifests"]],
                [str(path.resolve()) for path in search],
            )
            self.assertEqual(
                [record["path"] for record in handoff["pilot_rank4_manifests"]],
                [str(path.resolve()) for path in pilot_rank4],
            )
            self.assertEqual(
                [record["path"] for record in handoff["full_rank4_manifests"]],
                [str(path.resolve()) for path in rank4],
            )
            self.assertEqual(
                exploratory_full._validate_existing_summary(
                    output / "final-summary.json"
                ),
                summary,
            )
            pathlib.Path(full["search_runtime"]).write_bytes(
                b"tampered accepted teacher"
            )
            with self.assertRaisesRegex(ValueError, "acceptance receipt is stale"):
                exploratory_full._validate_existing_summary(
                    output / "final-summary.json"
                )

    def test_validate_launch_uses_only_frozen_records_and_never_git(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / exploratory_full.CAMPAIGN_ID
            bundle = self._write(output / "input-bundle/bundle-manifest.json")
            override = self._write(
                output / "input-bundle/pilot/teacher-only-override.json"
            )
            build = self._write(output / "release-build.json")
            guard = self._write(output / "guarded-producer")
            executable_records = {
                name: {"path": str(self._write(output / f"bin/{name}"))}
                for name in (
                    "continuation_generator",
                    "search_teacher",
                    "rank4_teacher",
                    "comparison",
                    "pack_tool",
                    "trainer",
                )
            }
            repository = {
                "path": str(output.parent),
                "head": "a" * 40,
                "branch": exploratory_full.TOPIC_BRANCH,
                "tree": "b" * 40,
                "clean": True,
                "topic_branch": exploratory_full.TOPIC_BRANCH,
                "base_commit": exploratory_full.BASE_COMMIT,
                "branch_bound": True,
            }
            bundle_record = {"fixture": True}
            build_record = {"fixture": True}
            body = {
                "schema": exploratory_full.LAUNCH_SCHEMA,
                "campaign_id": exploratory_full.CAMPAIGN_ID,
                "full_campaign_id": exploratory_full.FULL_CAMPAIGN_ID,
                "output_directory": str(output.resolve()),
                "base_commit": exploratory_full.BASE_COMMIT,
                "expected_commit": repository["head"],
                "expected_branch": exploratory_full.TOPIC_BRANCH,
                "repository": repository,
                "release_build": artifact_snapshot(build),
                "release_build_record": build_record,
                "executables": executable_records,
                "environment": selfsearch.environment_identity(),
                "input_bundle": artifact_snapshot(bundle),
                "input_bundle_record": bundle_record,
                "full_specification": exploratory_full._full_spec_record(),
                "artifact_guard": [artifact_snapshot(guard)],
                "runtime_git_access": False,
                "runtime_old_worktree_access": False,
                "persistent_power_check_required": True,
                "pilot_passed": False,
                "pilot_20_ms_passed": False,
                "teacher_only_override": artifact_snapshot(override),
                "canonical_promotion_eligible": False,
                "publication": False,
                "external_upload": False,
                "replace_rank4": False,
                "leaderboard_claim": False,
            }
            launch_path = output / "full-launch.json"
            exploratory_full._atomic_json(
                launch_path, exploratory_full._body_hashed(body)
            )
            fake_executables = mock.Mock()
            fake_executables.resolved.return_value = fake_executables
            fake_executables.snapshots.return_value = executable_records
            with (
                mock.patch.object(
                    exploratory_full.subprocess,
                    "run",
                    side_effect=AssertionError("Git/subprocess access forbidden"),
                ) as subprocess_run,
                mock.patch.object(
                    exploratory_full,
                    "_git",
                    side_effect=AssertionError("Git access forbidden"),
                ) as git,
                mock.patch.object(
                    exploratory_full, "_artifact_matches", return_value=True
                ),
                mock.patch.object(
                    exploratory_full,
                    "validate_input_bundle",
                    return_value=bundle_record,
                ),
                mock.patch.object(
                    exploratory_full,
                    "_build_record_without_git",
                    return_value=build_record,
                ),
                mock.patch.object(
                    selfsearch,
                    "CampaignExecutables",
                    return_value=fake_executables,
                ),
            ):
                validated = exploratory_full.validate_full_launch(launch_path)

            self.assertEqual(validated, exploratory_full._body_hashed(body))
            subprocess_run.assert_not_called()
            git.assert_not_called()
            fake_executables.validate.assert_called_once_with()

    def test_run_full_is_git_free_and_routes_all_cumulative_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / exploratory_full.CAMPAIGN_ID
            output.mkdir()
            launch_path = self._write(output / "full-launch.json", b"{}")
            launch, raw_routes, actor_sha256, reference_sha256 = (
                self._runtime_launch(output / "input-bundle")
            )
            override = self._write(
                output / "input-bundle/pilot/teacher-only-override.json"
            )
            build = self._write(output / "release-build.json")
            executable_records = {
                name: artifact_snapshot(self._write(output / f"bin/{name}"))
                for name in (
                    "continuation_generator",
                    "search_teacher",
                    "rank4_teacher",
                    "comparison",
                    "pack_tool",
                    "trainer",
                )
            }
            launch.update(
                {
                    "teacher_only_override": artifact_snapshot(override),
                    "artifact_guard": [],
                    "executables": executable_records,
                    "release_build": artifact_snapshot(build),
                    "release_build_record": {
                        "source_identities": {"fixture": "1" * 64}
                    },
                }
            )
            observed = {}

            def run_phase(**arguments):
                observed.update(arguments)
                arguments["producer_guard"]()
                return {
                    "profile": "full",
                    "campaign_id": exploratory_full.FULL_CAMPAIGN_ID,
                    "decision_path": str(output / "full/decision.json"),
                    "search_new_manifests": [],
                    "rank4_new_manifests": [],
                }

            decision = {
                "eligible_for_local_publication": False,
                "errors": ["fixture rejection"],
            }
            with (
                mock.patch.object(
                    exploratory_full, "TEACHER_ACTOR_SHA256", actor_sha256
                ),
                mock.patch.object(
                    exploratory_full,
                    "RETENTION_REFERENCE_SHA256",
                    reference_sha256,
                ),
                mock.patch.object(
                    exploratory_full, "validate_full_launch", return_value=launch
                ),
                mock.patch.object(
                    selfsearch,
                    "validate_host_health",
                    return_value={"power": "synthetic"},
                ),
                mock.patch.object(selfsearch, "run_phase", side_effect=run_phase),
                mock.patch.object(
                    exploratory_full,
                    "_validate_full_result",
                    return_value=decision,
                ),
                mock.patch.object(
                    exploratory_full.subprocess,
                    "run",
                    side_effect=AssertionError("Git/subprocess access forbidden"),
                ) as subprocess_run,
                mock.patch.object(
                    exploratory_full,
                    "_git",
                    side_effect=AssertionError("Git access forbidden"),
                ) as git,
            ):
                summary = exploratory_full.run_full(
                    launch_receipt=launch_path,
                    output=output,
                    resume=False,
                    skip_power_check=True,
                )

            self.assertEqual(summary["terminal"], "full-rejected")
            self.assertIs(observed["spec"], exploratory_full.FULL_SPEC)
            resolved_bundle_root = (output / "input-bundle").resolve()
            self.assertEqual(
                observed["canonical_prior_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["canonical_prior_manifests"]
                ),
            )
            self.assertEqual(
                observed["prior_search_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["pilot_search_manifests"]
                ),
            )
            self.assertEqual(
                observed["prior_rank4_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["pilot_rank4_manifests"]
                ),
            )
            self.assertEqual(
                observed["anchor_train_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["canonical_splits"]["train"]
                ),
            )
            self.assertEqual(
                observed["retention_validation_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["canonical_splits"]["validation"]
                ),
            )
            self.assertEqual(
                observed["anchor_validation_manifests"],
                tuple(
                    resolved_bundle_root / path
                    for path in raw_routes["canonical_splits"]["test"]
                ),
            )
            subprocess_run.assert_not_called()
            git.assert_not_called()
            self.assertIsNone(selfsearch._CAMPAIGN_LOCK_FD)

    def test_protected_source_markers_are_rejected_before_any_file_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for marker in exploratory_full.FORBIDDEN_PATH_MARKERS:
                protected = root / marker / "must-not-open.bin"
                with self.subTest(marker=marker), mock.patch.object(
                    pathlib.Path,
                    "is_file",
                    side_effect=AssertionError("protected path was probed"),
                ) as is_file:
                    with self.assertRaisesRegex(ValueError, "protected artifact"):
                        exploratory_full._assert_allowed_source(protected)
                    is_file.assert_not_called()

    def test_bundle_rejects_protected_relative_path_before_resolving_it(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = pathlib.Path(directory) / "input-bundle"
            bundle_root.mkdir()
            body = {
                "schema": exploratory_full.BUNDLE_SCHEMA,
                "campaign_id": exploratory_full.CAMPAIGN_ID,
                "source_fingerprints": {
                    "source_summary_sha256": exploratory_full.SOURCE_SUMMARY_SHA256,
                    "source_decision_sha256": exploratory_full.SOURCE_DECISION_SHA256,
                    "teacher_actor_sha256": exploratory_full.TEACHER_ACTOR_SHA256,
                    "original_retention_reference_sha256": (
                        exploratory_full.RETENTION_REFERENCE_SHA256
                    ),
                },
                "routes": {},
                "teacher_only_override": {},
                "full_specification": exploratory_full._full_spec_record(),
                "artifacts": [
                    {
                        "role": "forbidden-fixture",
                        "relative_path": "sealed-final/must-not-open.bin",
                        "sha256": "0" * 64,
                        "bytes": 0,
                    }
                ],
                "atomic_import": True,
                "explicit_allowlist": True,
                "runtime_uses_source_paths": False,
                "sealed_final_opening_bank_accessed": False,
                "blind_labels_accessed": False,
            }
            manifest_path = bundle_root / "bundle-manifest.json"
            exploratory_full._atomic_json(
                manifest_path, exploratory_full._body_hashed(body)
            )
            with mock.patch.object(
                exploratory_full,
                "_safe_local_path",
                side_effect=AssertionError("protected route was resolved"),
            ) as safe_local_path:
                with self.assertRaisesRegex(ValueError, "protected path"):
                    exploratory_full.validate_input_bundle(manifest_path)
                safe_local_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
