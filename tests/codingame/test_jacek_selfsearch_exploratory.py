import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import sys

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import jacek_selfsearch_exploratory as exploratory
import jacek_selfsearch_workflow as selfsearch
from jacek_replay_workflow import artifact_snapshot


class ExploratoryWorkflowTests(unittest.TestCase):
    def test_real_completed_lineage_never_reads_sealed_final_bank(self):
        rebuild_root = (
            REPOSITORY
            / "results/jacek_replay_bfm/replay-rebuild-20260826-v1"
        )
        inputs = rebuild_root / "evidence/rebuild-inputs.json"
        summary = rebuild_root / "run/final-summary.json"
        banks_path = rebuild_root / "evidence/banks/opening-banks.json"
        if not all(path.is_file() for path in (inputs, summary, banks_path)):
            self.skipTest("preserved rebuild artifacts are unavailable")
        banks = json.loads(banks_path.read_text())
        sealed = pathlib.Path(banks["final"]["artifact"]["path"]).resolve()
        original = pathlib.Path.read_bytes
        original_open = pathlib.Path.open

        def guarded_read_bytes(path):
            if path.resolve() == sealed:
                raise AssertionError("sealed final opening bytes were read")
            return original(path)

        def guarded_open(path, *arguments, **keywords):
            if path.resolve() == sealed:
                raise AssertionError("sealed final opening file was opened")
            return original_open(path, *arguments, **keywords)

        with (
            mock.patch.object(pathlib.Path, "read_bytes", guarded_read_bytes),
            mock.patch.object(pathlib.Path, "open", guarded_open),
        ):
            lineage = exploratory.validate_diversity_lineage(
                inputs_manifest=inputs, rebuild_summary=summary
            )
        self.assertEqual(
            lineage["frozen_v6_experimental_reference"]["sha256"],
            exploratory.V6_EXPERIMENTAL_REFERENCE_SHA256,
        )
        self.assertEqual(
            lineage["diversity_actor"]["sha256"],
            exploratory.DIVERSITY_ACTOR_SHA256,
        )
        self.assertFalse(lineage["sealed_final_bank_opened"])

    def test_profile_is_fresh_exact_2000_game_pilot(self):
        record = exploratory._phase_spec_record()
        self.assertEqual(sum(record["quotas"].values()), 2_000)
        self.assertEqual(record["quotas"]["incumbent-p1-vs-runner-up"], 100)
        self.assertEqual(record["quotas"]["incumbent-p2-vs-runner-up"], 100)
        self.assertEqual(record["game_seed"], exploratory.EXPLORATORY_GAME_SEED)
        self.assertEqual(
            record["opening_seed"], exploratory.EXPLORATORY_OPENING_SEED
        )
        self.assertNotEqual(record["game_seed"], selfsearch.PILOT_SPEC.game_seed)
        self.assertNotEqual(
            record["opening_seed"], selfsearch.PILOT_SPEC.opening_seed
        )
        expected = dict(selfsearch.PILOT_SPEC.configuration)
        expected["campaign_id"] = exploratory.EXPLORATORY_PILOT_CAMPAIGN_ID
        self.assertEqual(record["configuration"], expected)

    def test_opening_exclusions_never_read_sealed_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            excluded = root / "evaluation.tsv"
            development = root / "development.tsv"
            sealed = root / "sealed-final.tsv"
            excluded.write_text("evaluation")
            development.write_text("development")
            manifest = root / "opening-banks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "excluded_banks": [artifact_snapshot(excluded)],
                        "development": {"artifact": artifact_snapshot(development)},
                        "final": {
                            "artifact": {
                                "path": str(sealed),
                                "sha256": "f" * 64,
                                "bytes": 123,
                            }
                        },
                    }
                )
            )
            observed = []

            def states(path, *_arguments):
                observed.append(pathlib.Path(path).resolve())
                return {str(path)}

            with mock.patch.object(
                selfsearch, "_comparison_bank_states", side_effect=states
            ):
                result = exploratory._opening_exclusions(
                    {"opening_banks": {"path": str(manifest)}}
                )
            self.assertEqual(result, [excluded.resolve(), development.resolve()])
            self.assertNotIn(sealed.resolve(), observed)
            self.assertFalse(sealed.exists())

    def test_exact_json_resume_accepts_exact_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            exploratory._write_exact_json(path, {"value": 1}, "fixture")
            exploratory._write_exact_json(path, {"value": 1}, "fixture")
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                exploratory._write_exact_json(path, {"value": 2}, "fixture")

    def test_rehashed_auto_full_policy_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, launch_path, _reference_hash, _diversity_hash = (
                self._launch_fixture(pathlib.Path(directory))
            )
            launch = exploratory._load_json(launch_path, "launch")
            launch.pop("body_sha256")
            launch["automatic_full_launch"] = True
            exploratory._atomic_json(
                launch_path, exploratory._body_hashed(launch)
            )
            with self.assertRaisesRegex(ValueError, "launch receipt is invalid"):
                exploratory.validate_launch(launch_path)
            self.assertFalse((output / "full").exists())

    def _launch_fixture(self, root: pathlib.Path):
        output = root / exploratory.EXPLORATORY_CAMPAIGN_ID
        output.mkdir()
        reference = root / "reference.runtime"
        diversity = root / "diversity.runtime"
        reference.write_bytes(b"reference")
        diversity.write_bytes(b"diversity")
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        diversity_hash = hashlib.sha256(diversity.read_bytes()).hexdigest()

        lineage_record = {
            "frozen_v6_experimental_reference": artifact_snapshot(reference),
            "diversity_actor": artifact_snapshot(diversity),
        }
        lineage = output / "exploratory-lineage.json"
        exploratory._atomic_json(lineage, lineage_record)
        build_record = {"source_identities": {"fixture": "1" * 64}}
        build = output / "release-build.json"
        exploratory._atomic_json(build, build_record)

        executable_records = {}
        for name in (
            "continuation_generator", "search_teacher", "rank4_teacher",
            "comparison", "pack_tool", "trainer",
        ):
            path = root / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            executable_records[name] = artifact_snapshot(path)
        roots_tsv = root / "teacher-input.tsv"
        roots_manifest = root / "replay-roots.json"
        roots_tsv.write_text("roots")
        roots_manifest.write_text("{}")
        split_records = {}
        for split in ("train", "validation", "test"):
            values = []
            for index in range(3):
                path = root / f"{split}-{index}.json"
                path.write_text("{}")
                values.append(artifact_snapshot(path))
            split_records[split] = values
        prior = [
            split_records[split][index]
            for index in range(3)
            for split in ("train", "validation", "test")
        ]
        exclusion = root / "prior-openings.tsv"
        exclusion.write_text("openings")
        body = {
            "schema": exploratory.LAUNCH_SCHEMA,
            "campaign_id": exploratory.EXPLORATORY_CAMPAIGN_ID,
            "output_directory": str(output.resolve()),
            "expected_commit": "a" * 40,
            "repository": {"path": str(root), "head": "a" * 40},
            "lineage": artifact_snapshot(lineage),
            "lineage_record": lineage_record,
            "executables": executable_records,
            "release_build": artifact_snapshot(build),
            "release_build_record": build_record,
            "canonical_campaign": str(root / "canonical"),
            "canonical_ancestry": {},
            "roots": {
                "tsv": artifact_snapshot(roots_tsv),
                "manifest": artifact_snapshot(roots_manifest),
            },
            "splits": split_records,
            "canonical_prior_manifests": prior,
            "canonical_npz": [],
            "opening_exclusions": [artifact_snapshot(exclusion)],
            "sealed_final_untouched": {
                "path_recorded": str(root / "never-read.tsv"),
                "sha256_recorded": "e" * 64,
                "bytes_recorded": 42,
                "opened": False,
                "used_as_exclusion": False,
            },
            "pilot_specification": exploratory._phase_spec_record(),
            "environment": selfsearch.environment_identity(),
            "python_runtime": artifact_snapshot(pathlib.Path(sys.executable).resolve()),
            "pilot_games": 2_000,
            "automatic_full_launch": False,
            "full_directory_creation": False,
            "canonical_incumbent_replaced": False,
            "canonical_promotion_eligible": False,
            "external_upload": False,
            "rank4_replaced": False,
            "wrapper": artifact_snapshot(pathlib.Path(exploratory.__file__).resolve()),
        }
        guarded = exploratory._collect_snapshots(body)
        body["artifact_guard"] = [guarded[path] for path in sorted(guarded)]
        launch = exploratory._body_hashed(body)
        launch_path = output / "exploratory-launch.json"
        exploratory._atomic_json(launch_path, launch)
        return output, launch_path, reference_hash, diversity_hash

    def _fake_pilot(self, output: pathlib.Path, eligible: bool) -> dict:
        pilot = output / "pilot"
        pilot.mkdir(parents=True, exist_ok=True)
        decision = pilot / "decision.json"
        runtime = pilot / "search.runtime"
        manifest = pilot / "search.runtime.json"
        decision.write_text(json.dumps({"eligible_for_full": eligible}))
        runtime.write_bytes(b"trained")
        manifest.write_text("{}")
        search = []
        rank4 = []
        for arm, target in (("search", search), ("rank4", rank4)):
            for index in range(3):
                path = pilot / f"{arm}-{index}.json"
                path.write_text("{}")
                target.append(str(path))
        return {
            "campaign_id": exploratory.EXPLORATORY_PILOT_CAMPAIGN_ID,
            "decision": {"eligible_for_full": eligible},
            "decision_path": str(decision),
            "search_runtime": str(runtime),
            "search_manifest": str(manifest),
            "search_new_manifests": search,
            "rank4_new_manifests": rank4,
        }

    def _run_fixture(self, eligible: bool):
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        output, launch, reference_hash, diversity_hash = self._launch_fixture(root)
        observed = {}

        def run_phase(**arguments):
            observed.update(arguments)
            self.assertEqual(arguments["spec"], exploratory.EXPLORATORY_PILOT_SPEC)
            self.assertEqual(
                artifact_snapshot(arguments["actor"])["sha256"], reference_hash
            )
            self.assertEqual(
                artifact_snapshot(arguments["diversity"])["sha256"], diversity_hash
            )
            return self._fake_pilot(output, eligible)

        with (
            mock.patch.object(
                exploratory, "V6_EXPERIMENTAL_REFERENCE_SHA256", reference_hash
            ),
            mock.patch.object(exploratory, "DIVERSITY_ACTOR_SHA256", diversity_hash),
            mock.patch.object(selfsearch, "run_phase", side_effect=run_phase),
            mock.patch.object(
                selfsearch, "validate_host_health", return_value={"power": "test"}
            ),
            mock.patch.object(
                subprocess, "run", side_effect=AssertionError("Git/subprocess forbidden")
            ),
        ):
            summary = exploratory.run_exploratory_pilot(
                launch_receipt=launch,
                output=output,
                resume=True,
                skip_power_check=True,
            )
        return temporary, output, summary, observed

    def test_failed_pilot_stops_without_full_or_continuation(self):
        temporary, output, summary, _observed = self._run_fixture(False)
        try:
            self.assertEqual(summary["terminal"], "pilot-rejected")
            self.assertFalse(summary["full_started"])
            self.assertFalse(summary["full_continuation_eligible"])
            self.assertFalse((output / "full").exists())
            self.assertFalse((output / "full-continuation-eligible.json").exists())
            self.assertEqual(
                summary["pilot_actor_snapshot"]["sha256"],
                summary["experimental_reference_actor"]["sha256"],
            )
            self.assertEqual(
                summary["pilot_diversity_snapshot"]["sha256"],
                summary["diversity_actor"]["sha256"],
            )
        finally:
            temporary.cleanup()

    def test_passing_pilot_emits_receipt_but_never_starts_full(self):
        temporary, output, summary, _observed = self._run_fixture(True)
        try:
            self.assertEqual(summary["terminal"], "pilot-pass-continuation-eligible")
            self.assertTrue(summary["full_continuation_eligible"])
            self.assertFalse(summary["full_started"])
            self.assertFalse((output / "full").exists())
            receipt = exploratory._load_json(
                output / "full-continuation-eligible.json", "continuation"
            )
            self.assertEqual(receipt["required_games"], 10_000)
            self.assertTrue(receipt["requires_explicit_separate_launch"])
            self.assertFalse(receipt["automatic_full_launch"])
            self.assertFalse(receipt["external_upload"])
            self.assertFalse(receipt["replace_rank4"])
        finally:
            temporary.cleanup()

    def test_preexisting_full_directory_fails_before_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output, launch, reference_hash, diversity_hash = self._launch_fixture(root)
            (output / "full").mkdir()
            with (
                mock.patch.object(
                    exploratory, "V6_EXPERIMENTAL_REFERENCE_SHA256", reference_hash
                ),
                mock.patch.object(
                    exploratory, "DIVERSITY_ACTOR_SHA256", diversity_hash
                ),
                mock.patch.object(selfsearch, "run_phase") as run_phase,
                mock.patch.object(
                    selfsearch, "validate_host_health", return_value={"power": "test"}
                ),
                mock.patch.object(
                    subprocess, "run", side_effect=AssertionError("subprocess forbidden")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "preexisting full"):
                    exploratory.run_exploratory_pilot(
                        launch_receipt=launch,
                        output=output,
                        resume=True,
                        skip_power_check=True,
                    )
            run_phase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
