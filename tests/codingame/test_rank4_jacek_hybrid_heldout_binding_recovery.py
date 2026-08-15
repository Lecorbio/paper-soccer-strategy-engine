import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
RECORDER_PATH = TOOLS / (
    "record_rank4_jacek_hybrid_heldout_binding_recovery.py"
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "rank4_jacek_hybrid_binding_recovery_test_instance", RECORDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def canonical_file(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha256
    payload = json.loads(raw)
    assert recorder.canonical_json(payload) == raw
    return payload


class BindingRecoveryTest(unittest.TestCase):
    def test_plan_and_blocker_are_exact_canonical_preregistration(self):
        plan = canonical_file(
            recorder.BINDING_RECOVERY_PLAN,
            recorder.BINDING_RECOVERY_PLAN_SHA256,
        )
        blocker = canonical_file(
            recorder.PREBIND_BLOCKER, recorder.PREBIND_BLOCKER_SHA256
        )
        self.assertEqual(plan["schema"], recorder.PLAN_SCHEMA)
        self.assertEqual(
            tuple(sorted(plan["authorized_commit_shape"]["allowed_changed_paths"])),
            recorder.ALLOWED_CHANGED_PATHS,
        )
        self.assertEqual(
            plan["authorized_commit_shape"]["candidate_source_commit"],
            recorder.CANDIDATE_SOURCE_COMMIT,
        )
        boundary = plan["binding_recovery_campaign"][
            "preregistration_boundary"
        ]
        self.assertTrue(
            boundary[
                "synthetic_mocked_bind_and_stage_rejection_prefix_"
                "function_calls_occurred"
            ]
        )
        for name in (
            "sibling_admin_tool_cli_or_main_invoked_before_freeze",
            "synthetic_tests_accessed_any_tsv",
            "synthetic_tests_constructed_or_read_bank_path",
            "synthetic_tests_created_binding_or_stage_claim",
            "synthetic_tests_persisted_binding_report_or_decision_registry_artifact",
            "synthetic_tests_ran_producer_build_or_game",
        ):
            self.assertFalse(boundary[name])
        diagnostic = boundary["postcommit_pure_prefix_diagnostic"]
        self.assertEqual(
            diagnostic["superseded_administrative_commit"],
            "81be63a4cc0d88c258f6ab3cd26f5c079a3ad95c",
        )
        self.assertEqual(
            diagnostic["failure_code"],
            "mutable-private-frozen-recorder-path-alias",
        )
        self.assertTrue(diagnostic["prepare_binding_evidence_invoked"])
        self.assertFalse(diagnostic["prepare_binding_evidence_completed"])
        self.assertFalse(diagnostic["any_tsv_accessed"])
        self.assertTrue(
            diagnostic[
                "preliminary_exact_ps_process_cleanliness_check_"
                "completed_before_prepare"
            ]
        )
        self.assertFalse(diagnostic["prepare_process_table_invoked"])
        self.assertFalse(
            diagnostic["full_carried_preflight_receipt_validator_reached"]
        )
        self.assertFalse(diagnostic["binding_or_stage_claim_created"])
        self.assertFalse(diagnostic["lock_created_or_acquired"])
        self.assertFalse(
            diagnostic["validation_or_final_bank_path_accessed"]
        )
        self.assertEqual(
            diagnostic["superseded_ci_run"],
            {"classification": "cancelled-as-obsolete", "run_id": 31790817061},
        )
        self.assertFalse(
            blocker["diagnostic_boundary"]["outer_bind_cli_invoked"]
        )
        self.assertFalse(
            blocker["diagnostic_boundary"]["binding_claim_created"]
        )
        self.assertFalse(blocker["diagnostic_boundary"]["tsv_accessed"])
        self.assertFalse(
            blocker["diagnostic_boundary"]["validation_bank_accessed"]
        )
        self.assertFalse(
            blocker["diagnostic_boundary"]["final_bank_accessed"]
        )
        self.assertEqual(
            blocker["preflight_evidence"]["heldout_bank_files_accessed"], []
        )

    def test_carried_claim_and_receipt_are_exact_canonical_and_mode_neutral(self):
        claim = canonical_file(recorder.CARRY_CLAIM, recorder.CARRY_CLAIM_SHA256)
        receipt = canonical_file(
            recorder.CARRY_RECEIPT, recorder.CARRY_RECEIPT_SHA256
        )
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["heldout_bank_files_accessed"], [])
        self.assertEqual(recorder.CARRY_CLAIM.stat().st_size, 585)
        self.assertEqual(recorder.CARRY_RECEIPT.stat().st_size, 135815)
        embedded = dict(receipt["claim"])
        self.assertEqual(
            embedded.pop("path"), recorder.frozen.identity_label(recorder.CARRY_CLAIM)
        )
        self.assertEqual(embedded, claim)
        raw = recorder.CARRY_CLAIM.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            copy_path = Path(directory) / "claim.json"
            copy_path.write_bytes(raw)
            for mode in (0o444, 0o644):
                copy_path.chmod(mode)
                self.assertEqual(
                    recorder._read_canonical_exact(
                        copy_path, recorder.CARRY_CLAIM_SHA256, len(raw)
                    ),
                    claim,
                )
        self.assertNotIn("mode", receipt["claim"])
        carry_source = inspect.getsource(recorder.fixed_carried_preflight_receipt)
        self.assertIn("preflight.validate_passed_receipt", carry_source)
        self.assertIn("CARRY_RECEIPT_SHA256, CANDIDATE_SOURCE_COMMIT", carry_source)
        self.assertIn("preflight.RECOVERY_PLAN_SHA256", carry_source)

    def test_admin_tree_policy_is_direct_child_exact_six_paths_and_old_blobs(self):
        source = inspect.getsource(recorder.require_clean_admin_tree)
        self.assertIn('"rev-list", "--parents", "-n", "1", head', source)
        self.assertIn("[head, CANDIDATE_SOURCE_COMMIT]", source)
        self.assertIn("changed != ALLOWED_CHANGED_PATHS", source)
        self.assertEqual(len(recorder.ALLOWED_CHANGED_PATHS), 6)
        self.assertIn(
            "6d26d3eb76e91abcea0074099a88533ec7a10f0ef5fee92e738e108965e00785",
            source,
        )
        self.assertIn(
            "98fc501cff2750468dc00cdf19749b9fa3b8b5852e2b5c8cffe96f47548154ef",
            source,
        )
        self.assertIn("git_blob(CANDIDATE_SOURCE_COMMIT", source)

    def test_admin_tree_old_blob_check_ignores_mutable_private_module_paths(self):
        head = "a" * 40
        git = {
            "head": head, "author_utc": "x", "committer_utc": "x",
            "tracked_status": "",
        }

        def git_text(command, *arguments):
            if command == "rev-list":
                return f"{head} {recorder.CANDIDATE_SOURCE_COMMIT}"
            if command == "diff":
                return "\n".join(recorder.ALLOWED_CHANGED_PATHS)
            raise AssertionError((command, arguments))

        frozen_relatives = (
            "tools/record_rank4_jacek_hybrid_heldout_qualification.py",
            "tests/codingame/test_rank4_jacek_hybrid_heldout_qualification.py",
        )
        historical_blobs = {
            relative: recorder.frozen.git_blob(
                recorder.CANDIDATE_SOURCE_COMMIT, relative
            )
            for relative in frozen_relatives
        }

        with tempfile.TemporaryDirectory() as directory:
            historical_root = Path(directory)
            for relative, raw in historical_blobs.items():
                fixture = historical_root / relative
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_bytes(raw)

            def committed_blob(_head, relative):
                return (historical_root / relative).read_bytes()

            with mock.patch.object(
                recorder, "ROOT", historical_root
            ), mock.patch.object(
                recorder, "FROZEN_RECORDER", historical_root / frozen_relatives[0]
            ), mock.patch.object(
                recorder, "FROZEN_RECORDER_TEST",
                historical_root / frozen_relatives[1],
            ), mock.patch.object(
                recorder, "_ORIGINAL_REQUIRE_CLEAN_TRACKED_TREE", return_value=git
            ), mock.patch.object(
                recorder.frozen, "git_text", side_effect=git_text
            ), mock.patch.object(
                recorder.frozen, "git_blob", side_effect=committed_blob
            ), mock.patch.object(
                recorder.frozen, "RECORDER", historical_root / "wrong-tool.py"
            ), mock.patch.object(
                recorder.frozen, "RECORDER_TEST", historical_root / "wrong-test.py"
            ), mock.patch.object(
                recorder, "require_admin_after_prereg"
            ), mock.patch.object(
                recorder, "validate_binding_recovery_plan"
            ), mock.patch.object(
                recorder, "_require_exact_registry_entries"
            ), mock.patch.object(
                recorder, "_read_canonical_exact", return_value={}
            ), mock.patch.object(
                recorder, "require_parent_runtime_unopened"
            ):
                self.assertEqual(recorder.require_clean_admin_tree(), git)
        source = inspect.getsource(recorder.require_clean_admin_tree)
        self.assertIn("(FROZEN_RECORDER,", source)
        self.assertIn("(FROZEN_RECORDER_TEST,", source)
        self.assertNotIn("(frozen.RECORDER,", source)
        self.assertNotIn("(frozen.RECORDER_TEST,", source)

    def test_private_frozen_qualifier_does_not_mutate_canonical_module(self):
        canonical = importlib.import_module(
            "record_rank4_jacek_hybrid_heldout_qualification"
        )
        self.assertIsNot(canonical, recorder.frozen)
        self.assertNotEqual(canonical.OUTPUT, recorder.frozen.OUTPUT)
        self.assertTrue(str(canonical.OUTPUT).endswith("/recovery_v1"))
        self.assertEqual(
            canonical.BINDING_SCHEMA,
            "rank4-jacek-hybrid-heldout-binding-recovery-v1",
        )

    def test_both_import_orders_leave_canonical_constants_unchanged(self):
        snippets = (
            "import record_rank4_jacek_hybrid_heldout_qualification as q; "
            "before=(q.OUTPUT,q.BINDING_SCHEMA,q.RECORDER); "
            "import record_rank4_jacek_hybrid_heldout_binding_recovery; "
            "assert before==(q.OUTPUT,q.BINDING_SCHEMA,q.RECORDER)",
            "import record_rank4_jacek_hybrid_heldout_binding_recovery; "
            "import record_rank4_jacek_hybrid_heldout_qualification as q; "
            "assert str(q.OUTPUT).endswith('/recovery_v1'); "
            "assert q.BINDING_SCHEMA=='rank4-jacek-hybrid-heldout-binding-recovery-v1'",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(TOOLS)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for snippet in snippets:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", snippet], cwd=ROOT,
                env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_qualification_key_separates_source_and_admin_commits(self):
        dependencies = {
            recorder.frozen.identity_label(recorder.frozen.ENGINE_PATH): {
                "sha256": "a" * 64,
            },
            recorder.frozen.identity_label(recorder.frozen.SOURCE_PATH): {
                "sha256": "b" * 64,
            },
        }
        plan = {
            "configuration": {"profile": "nodes"},
            "banks": {
                "validation": [{"sha256": "c" * 64}],
                "final": [{"sha256": "d" * 64}],
            },
        }
        key = recorder.qualification_key(
            plan, "e" * 40, dependencies, "f" * 64,
            {"schema": "portability"},
            {"sha256": "1" * 64}, {"sha256": "2" * 64},
        )
        self.assertEqual(
            key["candidate_source_commit"], recorder.CANDIDATE_SOURCE_COMMIT
        )
        self.assertEqual(key["binding_admin_commit"], "e" * 40)
        self.assertNotEqual(
            key["candidate_source_commit"], key["binding_admin_commit"]
        )

    def test_tsv_is_rejected_before_any_metadata_operation(self):
        with mock.patch.object(
            recorder.os, "lstat", side_effect=AssertionError("metadata read")
        ):
            with self.assertRaisesRegex(ValueError, "TSV path is forbidden"):
                recorder.guard_read_path(
                    Path("/tmp/validation_d04.tsv"), allow_external=True
                )

    def test_only_exact_receipt_bound_bin_ps_is_permitted(self):
        with mock.patch.object(
            recorder, "_regular_identity", return_value=recorder.PS_IDENTITY
        ):
            self.assertEqual(
                recorder.guard_read_path(recorder.PS_PATH, allow_external=True),
                recorder.PS_PATH,
            )
        with self.assertRaisesRegex(ValueError, "not whitelisted"):
            recorder.guard_read_path(Path("/bin/ls"), allow_external=True)
        changed = {**recorder.PS_IDENTITY, "sha256": "0" * 64}
        with mock.patch.object(recorder, "_regular_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "/bin/ps identity changed"):
                recorder.guard_read_path(recorder.PS_PATH, allow_external=True)

    def test_duplicate_sdk_alias_token_is_rejected_before_deduplication(self):
        pthread, sched = sorted(recorder.SDK_ALIAS_PATHS)
        recorder._require_exact_alias_tokens([pthread, sched], "positive")
        with self.assertRaisesRegex(ValueError, "each exact SDK alias once"):
            recorder._require_exact_alias_tokens(
                [pthread, pthread, sched], "duplicate"
            )

    def _synthetic_sdk(self, directory: str):
        sdk_parent = Path(directory).resolve() / "SDKs"
        versioned = sdk_parent / "MacOSX26.5.sdk"
        include = versioned / "usr/include"
        target_dir = include / "pthread"
        target_dir.mkdir(parents=True)
        selector = sdk_parent / "MacOSX.sdk"
        selector.symlink_to("MacOSX26.5.sdk")
        specs = []
        for name, readlink, content in (
            ("pthread.h", "pthread/pthread.h", b"pthread\n"),
            ("sched.h", "pthread/sched.h", b"sched\n"),
        ):
            target = include / readlink
            target.write_bytes(content)
            target.chmod(0o644)
            link = include / name
            link.symlink_to(readlink)
            identity = recorder._regular_identity(target)
            specs.append({
                "lexical_path": str(selector / "usr/include" / name),
                "versioned_link_path": str(link),
                "readlink": readlink,
                "link_mode": "0755",
                "link_bytes": len(readlink),
                "resolved_path": str(target),
                "terminal": {
                    "bytes": identity["bytes"], "mode": identity["mode"],
                    "sha256": identity["sha256"],
                },
            })
        return selector, versioned, tuple(specs)

    def _sdk_patches(self, selector: Path, versioned: Path, specs: tuple):
        original_mode = recorder._mode

        def portable_mode(metadata):
            if stat.S_ISLNK(metadata.st_mode):
                return "0755"
            return original_mode(metadata)

        return (
            mock.patch.multiple(
                recorder, SDK_SELECTOR=selector,
                SDK_SELECTOR_READLINK="MacOSX26.5.sdk",
                SDK_RESOLVED_ROOT=versioned, SDK_ALIAS_SPECS=specs,
                SDK_ALIAS_PATHS={Path(item["lexical_path"]) for item in specs},
            ),
            mock.patch.object(recorder, "_mode", side_effect=portable_mode),
        )

    def test_synthetic_sdk_exact_aliases_pass_and_extra_alias_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            selector, versioned, specs = self._synthetic_sdk(directory)
            constants, modes = self._sdk_patches(selector, versioned, specs)
            with constants, modes:
                self.assertEqual(
                    recorder._selector_evidence()["readlink"],
                    "MacOSX26.5.sdk",
                )
                for spec in specs:
                    self.assertEqual(recorder._alias_evidence(spec), spec)
                extra = versioned / "usr/include/extra.h"
                extra.symlink_to("pthread/pthread.h")
                with self.assertRaisesRegex(ValueError, "leaf symlink"):
                    recorder._validate_external_alias_policy(
                        selector / "usr/include/extra.h"
                    )

    def test_sdk_leaf_retarget_escape_and_terminal_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            selector, versioned, specs = self._synthetic_sdk(directory)
            constants, modes = self._sdk_patches(selector, versioned, specs)
            with constants, modes:
                mutated = copy.deepcopy(specs[0])
                mutated["readlink"] = "/tmp/escape.h"
                with self.assertRaisesRegex(ValueError, "escapes"):
                    recorder._alias_evidence(mutated)
                mutated = copy.deepcopy(specs[0])
                mutated["terminal"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "target changed"):
                    recorder._alias_evidence(mutated)
                link = Path(specs[0]["versioned_link_path"])
                link.unlink()
                link.symlink_to("pthread/sched.h")
                with self.assertRaisesRegex(ValueError, "leaf alias changed"):
                    recorder._alias_evidence(specs[0])

    def test_repository_symlink_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            target = root / "real"
            target.mkdir(parents=True)
            (target / "input.cpp").write_text("x", encoding="ascii")
            (root / "alias").symlink_to("real")
            with mock.patch.object(recorder, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "symlink ancestor"):
                    recorder.guard_read_path(root / "alias/input.cpp")

    def test_exact_registry_rejects_foreign_and_aliased_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry"
            registry.mkdir()
            expected = registry / "a.json"
            expected.write_text("{}", encoding="ascii")
            self.assertEqual(
                recorder._require_exact_registry_entries(
                    registry, ("a.json",), "fixture"
                ),
                (expected,),
            )
            (registry / "foreign.json").write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "cardinality"):
                recorder._require_exact_registry_entries(
                    registry, ("a.json",), "fixture"
                )
            (registry / "foreign.json").unlink()
            expected.unlink()
            target = Path(directory) / "target.json"
            target.write_text("{}", encoding="ascii")
            expected.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular file"):
                recorder._require_exact_registry_entries(
                    registry, ("a.json",), "fixture"
                )

    def test_bind_foreign_parent_or_child_state_cannot_reach_claim(self):
        evidence = {"process": {}}
        for blocker in (
            "require_parent_runtime_unopened",
            "require_child_runtime_unopened_before_bind",
        ):
            with self.subTest(blocker=blocker), mock.patch.object(
                recorder.frozen, "validate_process_preflight"
            ), mock.patch.object(
                recorder, "prepare_binding_evidence", return_value=evidence
            ), mock.patch.object(
                recorder, "_stable_prepared_evidence", return_value={}
            ), mock.patch.object(
                recorder, blocker, side_effect=ValueError("foreign registry")
            ), mock.patch.object(
                recorder.frozen, "create_binding_claim"
            ) as create_claim:
                with self.assertRaisesRegex(ValueError, "foreign registry"):
                    recorder.create_binding_from_evidence(evidence)
                create_claim.assert_not_called()

    def test_stage_foreign_parent_or_binding_registry_cannot_reach_claim(self):
        binding = {
            "candidate_qualification_id": "a" * 64,
            "binding_admin_commit": "b" * 40,
            "candidate_source_commit": recorder.CANDIDATE_SOURCE_COMMIT,
        }
        git = {"head": "b" * 40}
        for blocker in (
            "require_parent_runtime_unopened",
            "require_exact_child_binding_registries",
        ):
            with self.subTest(blocker=blocker), mock.patch.object(
                recorder, "require_clean_admin_tree", return_value=git
            ), mock.patch.object(
                recorder, "require_parent_runtime_unopened"
            ), mock.patch.object(
                recorder, "require_exact_child_binding_registries"
            ), mock.patch.object(
                recorder, blocker, side_effect=ValueError("foreign registry")
            ), mock.patch.object(
                recorder.frozen, "create_stage_claim"
            ) as create_claim:
                with self.assertRaisesRegex(ValueError, "foreign registry"):
                    recorder.run_stage(
                        binding, "c" * 64, {}, "validation", {}, git
                    )
                create_claim.assert_not_called()
        with mock.patch.object(
            recorder, "require_clean_admin_tree", return_value=git
        ), mock.patch.object(
            recorder, "require_parent_runtime_unopened"
        ), mock.patch.object(
            recorder, "require_exact_child_binding_registries"
        ), mock.patch.object(
            recorder.frozen, "validate_stage_claim_registry",
            side_effect=ValueError("foreign stage claim"),
        ), mock.patch.object(
            recorder.frozen, "create_stage_claim"
        ) as create_claim:
            with self.assertRaisesRegex(ValueError, "foreign stage claim"):
                recorder.run_stage(
                    binding, "c" * 64, {}, "validation", {}, git
                )
            create_claim.assert_not_called()

    def test_bind_suffix_has_no_fallible_provenance_reads_after_claim(self):
        source = inspect.getsource(recorder.create_binding_from_evidence)
        claim_at = source.index("frozen.create_binding_claim")
        suffix = source[claim_at:]
        for forbidden in (
            "file_identity(", "environment_record(", "host_identity(",
            "_binding_runtime(", "portability_evidence(",
            "require_clean_admin_tree(",
        ):
            self.assertNotIn(forbidden, suffix)
        self.assertLess(
            source.index("revalidate_prepared_evidence"), claim_at
        )

    def test_preclaim_source_order_is_fail_closed(self):
        bind = inspect.getsource(recorder.revalidate_prepared_evidence)
        self.assertLess(
            bind.rindex("require_parent_runtime_unopened"),
            bind.rindex("portability_evidence"),
        )
        self.assertLess(
            bind.rindex("require_child_runtime_unopened_before_bind"),
            bind.rindex("portability_evidence"),
        )
        self.assertLess(
            bind.rindex("require_parent_runtime_unopened"),
            bind.rindex("require_clean_processes"),
        )
        self.assertLess(
            bind.rindex("require_clean_processes"),
            bind.rindex("portability_evidence"),
        )
        self.assertLess(
            bind.rindex("_regular_identity(PS_PATH)"),
            bind.rindex("require_clean_processes"),
        )
        suffix = inspect.getsource(recorder.create_binding_from_evidence)
        self.assertLess(
            suffix.index("revalidate_prepared_evidence"),
            suffix.index("frozen.create_binding_claim"),
        )
        stage = inspect.getsource(recorder.run_stage)
        claim = stage.index("frozen.create_stage_claim")
        for required in (
            "require_parent_runtime_unopened",
            "require_exact_child_binding_registries",
            "validate_stage_claim_registry",
            "require_clean_processes",
            "portability_evidence",
        ):
            self.assertLess(stage.index(required), claim)
        self.assertLess(claim, stage.index("frozen.stage_records"))

    def test_report_portability_stability_is_conjunctive_and_replayed(self):
        portability = {"schema": "fixture"}
        binding = {
            "binding_admin_commit": "a" * 40,
            "portability_evidence": portability,
        }
        report = {
            "candidate_source_commit": recorder.CANDIDATE_SOURCE_COMMIT,
            "binding_admin_commit": "a" * 40,
            "binding_recovery_campaign_id": recorder.ADMIN_CAMPAIGN_ID,
            "portability_before": portability,
            "portability_after": portability,
            "stable_portability": True,
            "started_utc": "x", "ended_utc": "x",
            "claim": {"claimed_utc": "x"},
            "stage_acceptable": True, "validation_codes": [],
            "threshold_errors": [],
        }
        with mock.patch.object(
            recorder, "portability_evidence", return_value=portability
        ), mock.patch.object(
            recorder, "require_admin_after_prereg"
        ), mock.patch.object(
            recorder, "_ORIGINAL_VALIDATE_STAGE_REPORT"
        ) as legacy_validator:
            recorder.validate_persisted_stage_report(
                report, binding, "b" * 64, {}, "validation"
            )
            projected = legacy_validator.call_args.args[0]
            for name in (
                "candidate_source_commit", "binding_admin_commit",
                "binding_recovery_campaign_id", "portability_before",
                "portability_after", "stable_portability",
            ):
                self.assertNotIn(name, projected)
            changed = {**report, "stable_portability": False}
            with self.assertRaisesRegex(ValueError, "portability/provenance"):
                recorder.validate_persisted_stage_report(
                    changed, binding, "b" * 64, {}, "validation"
                )
        with mock.patch.object(
            recorder, "portability_evidence", return_value={"schema": "drift"}
        ), mock.patch.object(
            recorder, "require_admin_after_prereg"
        ), mock.patch.object(
            recorder, "_ORIGINAL_VALIDATE_STAGE_REPORT"
        ):
            with self.assertRaisesRegex(ValueError, "live portability"):
                recorder.validate_persisted_stage_report(
                    report, binding, "b" * 64, {}, "validation"
                )

    def test_loader_replays_claim_cardinality_and_exact_timestamp_chain(self):
        source = inspect.getsource(recorder.load_and_validate_binding)
        self.assertIn("require_exact_child_binding_registries", source)
        self.assertIn(
            "receipt_created <= process_checked <= claimed <= created", source
        )
        decision = inspect.getsource(recorder.validate_persisted_decision)
        self.assertIn("_ORIGINAL_VALIDATE_DECISION", decision)
        self.assertIs(
            recorder.frozen.validate_persisted_stage_report,
            recorder.validate_persisted_stage_report,
        )

    def test_decision_binds_both_commits_and_recovery_campaign(self):
        binding = {"binding_admin_commit": "a" * 40}
        with mock.patch.object(
            recorder, "_ORIGINAL_DECISION_PAYLOAD", return_value={}
        ):
            payload = recorder.decision_payload(
                binding, "b" * 64, (Path("x"), "c" * 64, {}), None, "x"
            )
        self.assertEqual(
            payload["candidate_source_commit"], recorder.CANDIDATE_SOURCE_COMMIT
        )
        self.assertEqual(payload["binding_admin_commit"], "a" * 40)
        self.assertEqual(
            payload["binding_recovery_campaign_id"], recorder.ADMIN_CAMPAIGN_ID
        )
        decision = {
            **payload, "created_utc": "x",
        }
        with mock.patch.object(
            recorder, "require_admin_after_prereg"
        ), mock.patch.object(
            recorder, "_ORIGINAL_VALIDATE_DECISION"
        ) as legacy_validator:
            recorder.validate_persisted_decision(
                decision, binding, "b" * 64, {}
            )
        projected = legacy_validator.call_args.args[0]
        self.assertNotIn("candidate_source_commit", projected)
        self.assertNotIn("binding_admin_commit", projected)
        self.assertNotIn("binding_recovery_campaign_id", projected)


if __name__ == "__main__":
    unittest.main()
