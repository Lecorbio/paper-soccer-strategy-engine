import base64
import contextlib
import copy
import hashlib
import io
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from tools import compact_value_bfm_rank4_teacher_release as release
from tools import compact_value_bfm_upload as maintained_upload


q = release.qualification
COMMIT = "a" * 40


def make_runtime(root: pathlib.Path) -> pathlib.Path:
    counts = {"w1": 6301 * 12, "w2": 12 * 8, "w3": 8}
    counts["total"] = sum(counts.values())
    payload = bytes((counts["total"] * 3 + 7) // 8)
    body = {
        "schema": release.challenger.export_model.RUNTIME_SCHEMA,
        "feature_schema": release.challenger.export_model.FEATURE_SCHEMA,
        "architecture": {
            "name": "capacity-12x8",
            "dimensions": [6301, 12, 8, 1],
            "biases": False,
            "activations": release.challenger.export_model.ACTIVATIONS,
            "payload_layout": release.challenger.export_model.LAYOUT,
        },
        "quantization": {
            **release.challenger.export_model.QUANTIZATION,
            "scales": {"w1": 0.125, "w2": 0.125, "w3": 0.125},
            "weight_counts": counts,
            "packed_byte_count": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        },
        "selection": {
            "arm": "teacher-assisted",
            "seed": 20260907,
            "float_epoch": 1,
            "qat_epoch": 0,
            "source_bundle_body_sha256": "1" * 64,
        },
    }
    document = q.seal(body)
    raw = q.canonical_json_bytes(document)
    path = root / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
    path.write_bytes(raw)
    return path


def make_selected(root: pathlib.Path, *, macros=()) -> dict:
    runtime = make_runtime(root)
    header, _metadata = release.challenger.export_model.render_header(runtime)
    _path, base = release.source_exporter.render(model_header=header)
    selected_source = release._variant_source(base, macros)
    source = root / "selected.submission.cpp"
    source.write_bytes(selected_source)
    selection = root / "selection.json"
    q.write_sealed(selection, {"schema": "fixture.selection.v1"})
    return {
        "attempt": 1,
        "origin": "fixture",
        "runtime": release._record(runtime),
        "generated_source": release._record(source, ascii_required=True),
        "architecture": release._architecture(runtime),
        "compile_time_macros": list(macros),
        "configuration": release.deployment.deployment_configuration(
            ("0.95", "0.5", "1"), "default",
            release.deployment.PROFILE_ROSTER["default"],
        ),
        "selection_evidence": release._reference(selection, "fixture.selection.v1"),
    }


def passing_timing(probe_sha="2" * 64):
    samples = []
    for processes in release.maintained.PROCESS_COUNTS:
        for color in (0, 1):
            for replica in range(processes):
                samples.append({
                    "process_count": processes,
                    "color": color,
                    "replica": replica,
                    "first_ms": 800.0,
                    "later_max_ms": 155.0,
                    "stdout_sha256": "3" * 64,
                    "stderr_sha256": "4" * 64,
                })
    return {
        "schema": release.maintained.TIMING_SCHEMA,
        "probe_sha256": probe_sha,
        "first_limit_exclusive_ms": 900.0,
        "later_limit_exclusive_ms": 180.0,
        "samples": samples,
    }


def gh_payload(head=COMMIT):
    run_id = 123
    jobs = []
    for index, name in enumerate(release.upload.JOB_NAMES, 1):
        database_id = 1000 + index
        jobs.append({
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "databaseId": database_id,
            "url": f"{release.upload.RUN_URL_PREFIX}{run_id}/job/{database_id}",
        })
    return {
        "databaseId": run_id,
        "workflowDatabaseId": release.upload.WORKFLOW_DATABASE_ID,
        "attempt": 1,
        "workflowName": release.upload.WORKFLOW_NAME,
        "name": release.upload.WORKFLOW_NAME,
        "event": "workflow_dispatch",
        "headBranch": release.RELEASE_BRANCH,
        "headSha": head,
        "status": "completed",
        "conclusion": "success",
        "url": f"{release.upload.RUN_URL_PREFIX}{run_id}",
        "jobs": jobs,
    }


class SourceAndPromotionTest(unittest.TestCase):
    def test_attempt_zero_preserves_frozen_legacy_source_not_current_exporter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            selected = make_selected(root)
            selected["attempt"] = 0
            current = pathlib.Path(selected["generated_source"]["path"]).read_bytes()
            legacy = current.replace(
                b"kRootPartialPaths=4'000;",
                b"kRootPartialPaths = 4'000;",
                1,
            )
            legacy_path = root / "legacy-f5.submission.cpp"
            legacy_path.write_bytes(legacy)
            selected["generated_source"] = release._record(
                legacy_path, ascii_required=True
            )
            legacy_evidence = {
                "commit": release.ATTEMPT_ZERO_LEGACY_COMMIT,
                "source_hash_preserved": True,
            }
            with mock.patch.object(
                release, "_validate_legacy_attempt_zero_source",
                return_value=legacy_evidence,
            ):
                payloads = release._release_payloads(root, selected)
            self.assertEqual(payloads["submission.cpp"], current)
            self.assertNotEqual(payloads["submission.cpp"], legacy)
            self.assertEqual(
                release.deployment.recover_generated_source(
                    payloads["discrete_v3_deployment.cpp"],
                    search_tuple=selected["configuration"]["tuple"],
                    profile="default",
                    work=release.deployment.PROFILE_ROSTER["default"],
                ),
                legacy,
            )
            self.assertEqual(
                payloads["derivation"]["selected_generated_source"]["sha256"],
                hashlib.sha256(legacy).hexdigest(),
            )
            self.assertEqual(
                payloads["derivation"]["legacy_attempt_zero"], legacy_evidence
            )
            self.assertFalse(
                payloads["derivation"]["current_exporter_equality_required"]
            )
            self.assertEqual(
                payloads["derivation"]["selected_finalist_sha256_preserved"],
                hashlib.sha256(legacy).hexdigest(),
            )
            selected["attempt"] = 1
            with self.assertRaisesRegex(
                release.ReleaseBridgeError,
                "runtime export plus frozen search macros",
            ):
                release._release_payloads(root, selected)

    def test_attempt_zero_frozen_identity_and_c380_closure_are_exact(self):
        self.assertEqual(
            release.challenger.ALLOWLIST["attempt_zero_source"],
            {
                "bytes": 94_834,
                "sha256": "f5e67d699be19c3d495673c04ee2453570391c59e5f7be2a779198ce98b2d621",
            },
        )
        self.assertEqual(
            release.ATTEMPT_ZERO_LEGACY_COMMIT,
            "c380ae74b999eb6fd16d7bbfd49e16cc24f95ded",
        )
        self.assertEqual(len(release.ATTEMPT_ZERO_LEGACY_CLOSURE), 7)
        self.assertTrue(all(
            len(digest) == 64
            for digest in release.ATTEMPT_ZERO_LEGACY_CLOSURE.values()
        ))

    def test_runtime_export_macro_prefix_and_seven_slots_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            macro = "COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT"
            selected = make_selected(root, macros=(macro,))
            payloads = release._release_payloads(root, selected)
            deployed = payloads["discrete_v3_deployment.cpp"]
            self.assertTrue(deployed.startswith(f"#define {macro} 1\n".encode()))
            self.assertIn(b"shuffle_seed{1ULL}", deployed)
            self.assertEqual(
                release.deployment.recover_generated_source(
                    deployed,
                    search_tuple=selected["configuration"]["tuple"],
                    profile="default",
                    work=release.deployment.PROFILE_ROSTER["default"],
                ),
                pathlib.Path(selected["generated_source"]["path"]).read_bytes(),
            )
            tampered = copy.deepcopy(selected)
            changed = root / "changed.cpp"
            changed.write_bytes(
                pathlib.Path(selected["generated_source"]["path"]).read_bytes()
                + b"\n"
            )
            tampered["generated_source"] = release._record(
                changed, ascii_required=True
            )
            with self.assertRaisesRegex(
                release.ReleaseBridgeError, "runtime export plus frozen search macros"
            ):
                release._release_payloads(root, tampered)

    def test_promote_writes_only_fixed_four_artifacts_and_never_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            selected = make_selected(root)
            plan = root / "campaign-plan.json"
            q.write_sealed(plan, {"schema": release.challenger.PLAN_SCHEMA})
            output = root / "promotion.json"
            selected_validator = mock.Mock(return_value=selected)
            git = {
                "repository": str(repository.resolve()),
                "commit": COMMIT,
                "branch": release.RELEASE_BRANCH,
                "head_ref": f"refs/heads/{release.RELEASE_BRANCH}",
                "clean": True,
                "dirty_paths": [],
            }
            expected_dirty = {item.as_posix() for item in release.PROMOTED_RELATIVES}
            with mock.patch.object(
                release, "_git_identity", return_value=git
            ), mock.patch.object(
                release, "_git_status_paths", return_value=expected_dirty
            ):
                receipt = release.promote_candidate(
                    campaign_plan_path=plan, attempt=1,
                    candidate_runtime=pathlib.Path(selected["runtime"]["path"]),
                    candidate_source=pathlib.Path(
                        selected["generated_source"]["path"]
                    ),
                    repository=repository, output_path=output,
                    promoted_at_utc="2026-09-04T00:00:00Z",
                    selected_validator=selected_validator,
                )
            self.assertEqual(set(receipt["tracked_artifacts"]), {
                item.name for item in release.PROMOTED_RELATIVES
            })
            self.assertFalse(receipt["commit_performed"])
            self.assertFalse(receipt["push_performed"])
            self.assertFalse(receipt["ci_started"])
            for item in release.PROMOTED_RELATIVES:
                self.assertTrue((repository / item).is_file())


class TimingAndCiTest(unittest.TestCase):
    def test_release_mutating_and_preflight_cli_require_explicit_repository(self):
        selected = [
            "--campaign-plan", "/campaign", "--attempt", "1",
            "--candidate-runtime", "/runtime",
            "--candidate-source", "/source",
        ]
        commands = (
            ["promote", *selected, "--output", "/promotion"],
            [
                "prepare-preflight", *selected,
                "--promotion", "/promotion", "--base-preflight", "/base",
                "--output-root", "/output",
            ],
            [
                "seal-release-evidence", *selected,
                "--promotion", "/promotion", "--preflight", "/preflight",
                "--ci", "/ci", "--output", "/release-evidence",
            ],
        )
        for arguments in commands:
            with self.subTest(command=arguments[0]), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit) as raised:
                release.main(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_gnu_discovery_prefers_versioned_driver_and_rejects_apple_alias(self):
        paths = {
            "g++-15": "/opt/homebrew/bin/g++-15",
            "g++": "/usr/bin/g++",
        }

        def which(name):
            return paths.get(name)

        def compiler(path, family):
            self.assertEqual(family, "GNU")
            if str(path) == "/usr/bin/g++":
                raise ValueError("AppleClang alias")
            return {"family": "GNU"}

        with mock.patch.object(release.shutil, "which", side_effect=which), \
                mock.patch.object(
                    release.deployment_preflight, "_compiler_record",
                    side_effect=compiler,
                ):
            self.assertEqual(
                release._discover_gnu_compiler(),
                pathlib.Path("/opt/homebrew/bin/g++-15"),
            )

    def test_timing_requires_full_maintained_suite_and_both_uncontended_colors(self):
        timing = passing_timing()
        uncontended = release._uncontended_timing(timing)
        self.assertEqual(uncontended["workers"], 1)
        self.assertEqual(uncontended["colors"], [0, 1])
        changed = copy.deepcopy(timing)
        changed["samples"] = [
            row for row in changed["samples"]
            if not (row["process_count"] == 1 and row["color"] == 1)
        ]
        with self.assertRaises(release.ReleaseBridgeError):
            release._uncontended_timing(changed)

    def test_ci_keeps_exact_branch_workflow_attempt_commit_and_five_jobs(self):
        normalized = release.upload.validate_gh_run(
            gh_payload(), expected_head=COMMIT
        )
        self.assertEqual(normalized["head_branch"], release.RELEASE_BRANCH)
        self.assertEqual(set(normalized["jobs"]), set(release.upload.REQUIRED_JOB_IDS))
        for field, value in (
            ("headBranch", "codex/rank4-teacher-challenger-v2"),
            ("workflowDatabaseId", 1),
            ("attempt", 2),
            ("headSha", "b" * 40),
        ):
            changed = gh_payload()
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(
                release.upload.UploadError
            ):
                release.upload.validate_gh_run(changed, expected_head=COMMIT)
        changed = gh_payload()
        changed["jobs"].pop()
        with self.assertRaises(release.upload.UploadError):
            release.upload.validate_gh_run(changed, expected_head=COMMIT)


class ReleaseEvidenceTest(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        selected = make_selected(root)
        candidate = root / "deployed.cpp"
        candidate.write_text("candidate\n", encoding="ascii")
        candidate_record = release._record(candidate, ascii_required=True)
        campaign = root / "campaign-plan.json"
        q.write_sealed(campaign, {"schema": release.challenger.PLAN_SCHEMA})
        promotion = root / "promotion.json"
        q.write_sealed(promotion, {"schema": release.PROMOTION_SCHEMA})
        base = root / ("b" * 64 + ".json")
        q.write_sealed(base, {"schema": release.maintained.RECEIPT_SCHEMA})
        preflight_plan = root / "preflight-plan.json"
        q.write_sealed(preflight_plan, {
            "schema": release.PREFLIGHT_PLAN_SCHEMA,
            "inputs": {"base_preflight": release._reference(
                base, release.maintained.RECEIPT_SCHEMA
            )},
        })
        preflight_reference = root / "preflight-reference.json"
        q.write_sealed(preflight_reference, {
            "schema": release.PREFLIGHT_REFERENCE_SCHEMA,
            "plan": release._reference(
                preflight_plan, release.PREFLIGHT_PLAN_SCHEMA
            ),
        })
        ci_path = root / "ci.json"
        q.write_sealed(ci_path, {"schema": release.upload.CI_SCHEMA})
        source_binding = root / "source-binding.json"
        q.create_source_binding(
            source_binding, candidate_source=candidate,
            candidate_commit=COMMIT,
            rank4_source=release.REPOSITORY / release.RANK4_RELATIVE,
            opponent_source=release.REPOSITORY / release.RANK4_RELATIVE,
        )
        timing = passing_timing()
        compile_command = {"passed": True, "argv": ["compile"]}
        compiler = {"path": "/clang", "sha256": "7" * 64}
        gate = {"path": "/gate", "bytes": 1, "sha256": "8" * 64,
                "executable": True}
        preflight_state = {
            "candidate_commit": COMMIT,
            "candidate": candidate_record,
            "runtime": selected["runtime"],
            "derivation": {"fixture": True},
            "configuration": selected["configuration"],
            "gate": gate,
            "compile_command": compile_command,
            "compiler": compiler,
            "timing": timing,
            "uncontended_timing": release._uncontended_timing(timing),
            "receipt": {"completed_at_utc": "2026-09-04T00:01:00Z"},
            "base_preflight_path": base,
        }
        git = {
            "repository": str(release.REPOSITORY.resolve()),
            "commit": COMMIT,
            "branch": release.RELEASE_BRANCH,
            "head_ref": f"refs/heads/{release.RELEASE_BRANCH}",
            "clean": True,
            "dirty_paths": [],
        }
        artifacts = {"fixture": True}
        payloads = {"derivation": {"fixture": True}}
        ci = {
            "head_sha": COMMIT,
            "head_branch": release.RELEASE_BRANCH,
            "fetched_at_utc": "2026-09-04T00:02:00Z",
        }
        promotion_value = {"promoted_at_utc": "2026-09-04T00:00:00Z"}
        candidate_value = {
            "runtime": selected["runtime"],
            "source": candidate_record,
            "generated_source": selected["generated_source"],
            "architecture": selected["architecture"],
        }
        compile_binding = {
            "compiler": compiler,
            "command_sha256": q.sha256_bytes(q.canonical_json_bytes(compile_command)),
            "candidate_sha256": candidate_record["sha256"],
            "gate": gate,
            "candidate_embedded": True,
        }
        body = {
            "schema": release.RELEASE_EVIDENCE_SCHEMA,
            "namespace": release.NAMESPACE,
            "campaign_id": release.CAMPAIGN_ID,
            "attempt": 1,
            "status": "release-committed-preflighted-green-ci-before-dual-final",
            "created_at_utc": "2026-09-04T00:03:00Z",
            "campaign_plan": release._reference(
                campaign, release.challenger.PLAN_SCHEMA
            ),
            "selection_evidence": selected["selection_evidence"],
            "candidate": candidate_value,
            "candidate_commit": COMMIT,
            "repository": str(release.REPOSITORY.resolve()),
            "branch": release.RELEASE_BRANCH,
            "git": git,
            "tracked_artifacts": artifacts,
            "promotion": release._reference(promotion, release.PROMOTION_SCHEMA),
            "derivation": payloads["derivation"],
            "configuration": selected["configuration"],
            "source_binding": release._reference(
                source_binding, q.SOURCE_BINDING_SCHEMA
            ),
            "maintained_preflight": release._reference(
                base, release.maintained.RECEIPT_SCHEMA
            ),
            "release_preflight": release._reference(
                preflight_reference, release.PREFLIGHT_REFERENCE_SCHEMA
            ),
            "gate": gate,
            "compile_binding": compile_binding,
            "timing": timing,
            "uncontended_timing": preflight_state["uncontended_timing"],
            "ci": release._reference(ci_path, release.upload.CI_SCHEMA),
            "policy": {
                "created_before_dual_authorization": True,
                "protected_banks_accessed": False,
                "candidate_change_authorized": False,
                "uploads_authorized": 0,
                "rank4_replacement_authorized": False,
            },
        }
        evidence = root / "release-evidence.json"
        q.write_sealed(evidence, body)
        patches = (
            mock.patch.object(release, "_release_payloads", return_value=payloads),
            mock.patch.object(
                release, "_verify_promoted_commit", return_value=(git, artifacts)
            ),
            mock.patch.object(release, "validate_promotion", return_value=promotion_value),
            mock.patch.object(
                release, "validate_preflight_reference", return_value=preflight_state
            ),
            mock.patch.object(release.upload, "validate_ci_evidence", return_value=ci),
        )
        return {
            "selected": selected, "candidate": candidate,
            "campaign": campaign, "promotion": promotion,
            "base": base, "preflight_reference": preflight_reference,
            "ci_path": ci_path, "source_binding": source_binding,
            "evidence": evidence, "body": body, "patches": patches,
            "preflight_state": preflight_state,
        }

    def validate(self, fixture):
        with fixture["patches"][0], fixture["patches"][1], fixture["patches"][2], \
                fixture["patches"][3], fixture["patches"][4]:
            return release.validate_release_evidence(
                fixture["evidence"], campaign_plan_path=fixture["campaign"],
                attempt=1,
                candidate_runtime=pathlib.Path(
                    fixture["selected"]["runtime"]["path"]
                ),
                candidate_source=pathlib.Path(
                    fixture["selected"]["generated_source"]["path"]
                ),
                selected_validator=mock.Mock(return_value=fixture["selected"]),
            )

    def test_release_evidence_is_recursive_and_policy_tamper_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(pathlib.Path(temporary))
            value = self.validate(fixture)
            self.assertEqual(value["candidate_commit"], COMMIT)
            self.assertEqual(value["candidate"]["source"], release._record(
                fixture["candidate"], ascii_required=True
            ))
            changed = copy.deepcopy(fixture["body"])
            changed["policy"]["uploads_authorized"] = 1
            fixture["evidence"].write_bytes(q.canonical_json_bytes(q.seal(changed)))
            with self.assertRaisesRegex(
                release.ReleaseBridgeError, "release evidence chain changed"
            ):
                self.validate(fixture)

    def test_dual_final_adapter_exposes_complete_preflight_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = root / "plan.json"
            receipt_path = root / "receipt.json"
            reference_path = root / "reference.json"
            gate = {"path": "/gate", "bytes": 1, "sha256": "9" * 64,
                    "executable": True}
            plan = q.write_sealed(plan_path, {
                "schema": release.PREFLIGHT_PLAN_SCHEMA,
                "inputs": {"repository": "/release", "tools": {
                    "clang": {"path": "/clang", "sha256": "8" * 64}
                }},
            })
            receipt = q.write_sealed(receipt_path, {
                "schema": release.PREFLIGHT_RECEIPT_SCHEMA,
                "commands": {"compile_rank4_gate": {
                    "passed": True, "argv": ["clang", "candidate"]
                }},
                "binaries": {"rank4_gate": gate},
            })
            reference = q.write_sealed(reference_path, {
                "schema": release.PREFLIGHT_REFERENCE_SCHEMA,
                "plan": release._reference(plan_path, release.PREFLIGHT_PLAN_SCHEMA),
                "receipt": release._reference(
                    receipt_path, release.PREFLIGHT_RECEIPT_SCHEMA
                ),
                "gate": gate,
            })
            evidence_path = root / "release-evidence.json"
            q.write_sealed(evidence_path, {
                "schema": release.RELEASE_EVIDENCE_SCHEMA
            })
            timing = passing_timing()
            validated = {
                "release_preflight": release._reference(
                    reference_path, release.PREFLIGHT_REFERENCE_SCHEMA
                ),
                "candidate_commit": COMMIT,
                "candidate": {
                    "source": {"path": "/candidate", "sha256": "1" * 64},
                    "runtime": {"path": "/runtime", "sha256": "2" * 64},
                },
                "configuration": {"tuple": ["0.95", "0.5", "1"]},
                "derivation": {"exact": True},
                "timing": timing,
                "uncontended_timing": release._uncontended_timing(timing),
                "ci": {"path": "/ci", "sha256": "3" * 64},
            }
            with mock.patch.object(
                release, "validate_release_evidence", return_value=validated
            ):
                state = release.dual_final_preflight_state(
                    evidence_path, campaign_plan_path=root / "campaign.json",
                    attempt=1, candidate_runtime=root / "runtime",
                    candidate_source=root / "source",
                )
            self.assertEqual(state["reference"], reference)
            self.assertEqual(state["receipt"], receipt)
            self.assertEqual(state["plan"], plan)
            self.assertIn("compile_rank4_gate", state["receipt"]["commands"])
            self.assertEqual(state["receipt"]["binaries"]["rank4_gate"], gate)
            self.assertEqual(state["plan"]["inputs"]["repository"], "/release")
            self.assertIn("clang", state["plan"]["inputs"]["tools"])
            self.assertEqual(state["candidate"], validated["candidate"]["source"])
            self.assertEqual(state["runtime"], validated["candidate"]["runtime"])
            self.assertEqual(
                state["derivation"]["configuration"], validated["configuration"]
            )
            self.assertEqual(state["timing"], timing)
            self.assertEqual(state["ci"], validated["ci"])


class UploadAuthorizationTest(unittest.TestCase):
    def test_release_authorization_drives_maintained_upload_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = root / "campaign.json"
            q.write_sealed(campaign, {"schema": release.challenger.PLAN_SCHEMA})
            runtime = root / "runtime.json"
            runtime.write_text("runtime\n", encoding="ascii")
            selected = root / "selected.cpp"
            selected.write_text("selected\n", encoding="ascii")
            deployed = root / "deployed.cpp"
            deployed.write_text("deployed candidate\n", encoding="ascii")
            copyback = root / "editor-copyback.cpp"
            copyback.write_bytes(deployed.read_bytes())
            dual = root / "dual-qualified.json"
            q.write_sealed(dual, {
                "schema": release.challenger.DUAL_QUALIFICATION_SCHEMA
            })
            release_path = root / "release-evidence.json"
            q.write_sealed(release_path, {
                "schema": release.RELEASE_EVIDENCE_SCHEMA,
                "campaign_plan": release._reference(
                    campaign, release.challenger.PLAN_SCHEMA
                ),
                "candidate": {
                    "runtime": release._record(runtime),
                    "generated_source": release._record(
                        selected, ascii_required=True
                    ),
                    "source": release._record(deployed, ascii_required=True),
                },
            })
            authorization_path = root / "one-upload-authorization.json"
            q.write_sealed(authorization_path, {
                "schema": q.UPLOAD_AUTH_SCHEMA,
                "namespace": release.NAMESPACE,
                "uploads_authorized": 1,
                "rank4_replacement_authorized": False,
                "candidate_commit": COMMIT,
                "candidate": release._record(deployed, ascii_required=True),
                "binding": {"path": "/binding", "sha256": "1" * 64},
                "aggregate": {"path": "/aggregate", "sha256": "2" * 64},
                "ci": {"head_sha": COMMIT},
                "upload_ledger_root": str(root.resolve()),
                "release_evidence": release._reference(
                    release_path, release.RELEASE_EVIDENCE_SCHEMA
                ),
                "dual_qualification": release._reference(
                    dual, release.challenger.DUAL_QUALIFICATION_SCHEMA
                ),
                "two_independent_rank4_gates_passed": True,
            })
            inputs_path = root / "authorization-inputs.json"
            q.write_sealed(inputs_path, {
                "schema": release.UPLOAD_INPUTS_SCHEMA,
                "namespace": release.NAMESPACE,
                "campaign_id": release.CAMPAIGN_ID,
                "attempt": 1,
                "status": "exactly-one-upload-authorized-after-dual-qualification",
                "authorized_at_utc": "2026-09-04T00:00:00Z",
                "release_evidence": release._reference(
                    release_path, release.RELEASE_EVIDENCE_SCHEMA
                ),
                "dual_qualification": release._reference(
                    dual, release.challenger.DUAL_QUALIFICATION_SCHEMA
                ),
                "gate_results": [],
                "authorization": release._reference(
                    authorization_path, q.UPLOAD_AUTH_SCHEMA
                ),
                "candidate_commit": COMMIT,
                "candidate": {
                    "runtime": release._record(runtime),
                    "generated_source": release._record(
                        selected, ascii_required=True
                    ),
                    "source": release._record(deployed, ascii_required=True),
                },
                "ci": {"path": "/ci", "sha256": "3" * 64},
                "uploads_authorized": 1,
                "submit_clicks_authorized": 1,
                "second_upload_authorized": False,
                "rank4_replacement_authorized": False,
            })
            authorization = q.load_sealed(
                authorization_path, q.UPLOAD_AUTH_SCHEMA
            )
            inputs = q.load_sealed(inputs_path, release.UPLOAD_INPUTS_SCHEMA)
            validate = mock.Mock(return_value={
                "authorization": authorization,
                "authorization_path": authorization_path,
                "inputs": inputs,
                "inputs_path": inputs_path,
            })
            lazy_release = types.SimpleNamespace(
                RELEASE_EVIDENCE_SCHEMA=release.RELEASE_EVIDENCE_SCHEMA,
                validate_upload_authorization=validate,
            )
            with mock.patch.object(
                maintained_upload, "_load", return_value=lazy_release
            ):
                editor = maintained_upload.fresh_editor(
                    root, session_id="fresh-session",
                    opened_at_utc="2026-09-04T00:01:00Z",
                )
                copied = maintained_upload.attest_copyback(
                    root, generated_source=deployed,
                    copied_back_source=copyback,
                    created_at_utc="2026-09-04T00:02:00Z",
                )
                played = maintained_upload.record_play(
                    root, legal_stdout=True, expected_telemetry=True,
                    created_at_utc="2026-09-04T00:03:00Z",
                )
                started = maintained_upload.start_submit(
                    root, started_at_utc="2026-09-04T00:04:00Z"
                )
                attested = maintained_upload.attest_submission(
                    root, agent_id=123, submission_id=456,
                    submitted_at_utc="2026-09-04T00:05:00Z",
                )
            self.assertEqual(editor["status"], "fresh-editor-opened")
            self.assertEqual(copied["status"], "editor-copyback-verified")
            self.assertEqual(played["status"], "play-passed")
            self.assertEqual(started["status"], "submit-started")
            self.assertEqual(attested["status"], "submission-attested")
            self.assertEqual(attested["submit_clicks"], 1)
            self.assertGreaterEqual(validate.call_count, 5)
            first = validate.call_args_list[0].kwargs
            self.assertEqual(first["campaign_plan_path"], campaign.resolve())
            self.assertEqual(first["candidate_source"], selected.resolve())
            self.assertEqual(first["candidate_runtime"], runtime.resolve())
            self.assertEqual(first["dual_qualified_path"], dual.resolve())

    def test_dual_chain_accepts_governance_records_and_deep_gate_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            attempt_root = root / "dual-final/attempt-001"
            attempt_root.mkdir(parents=True)
            campaign = root / "campaign-plan.json"
            q.write_sealed(campaign, {"schema": release.challenger.PLAN_SCHEMA})
            source = root / "candidate.cpp"
            runtime = root / "candidate.runtime"
            source.write_text("candidate\n", encoding="ascii")
            runtime.write_text("runtime\n", encoding="ascii")
            source_record = release._record(source, ascii_required=True)
            runtime_record = release._record(runtime)
            architecture = {"id": release.challenger.ARCHITECTURE}
            release_path = root / "release-evidence.json"
            q.write_sealed(release_path, {
                "schema": release.RELEASE_EVIDENCE_SCHEMA
            })
            normalized_candidate = {
                "runtime": runtime_record,
                "source": {
                    key: source_record[key] for key in ("path", "bytes", "sha256")
                },
                "architecture": architecture,
            }
            authorization = attempt_root / "dual-final-authorization.json"
            q.write_sealed(authorization, {
                "schema": release.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
                "candidate": normalized_candidate,
                "release_evidence": release.challenger._sealed_record(
                    release_path, release.RELEASE_EVIDENCE_SCHEMA
                ),
            })
            dual_plan = attempt_root / "dual-plan.json"
            q.write_sealed(dual_plan, {
                "schema": release.challenger.DUAL_FINAL_SCHEMA
            })
            results = []
            result_values = {}
            deep_values = {}
            for ordinal, gate_id in enumerate(("gate-a", "gate-b")):
                binding = root / f"{gate_id}-binding.json"
                q.write_sealed(binding, {"schema": q.GATE_BINDING_SCHEMA})
                aggregate = root / f"{gate_id}-aggregate.json"
                q.write_sealed(aggregate, {
                    "schema": q.FINAL_AGGREGATE_SCHEMA,
                    "verdict": {"passed": True},
                })
                evidence = root / f"{gate_id}-evidence.json"
                evidence.write_text(f"{gate_id}\n")
                result = attempt_root / f"{gate_id}.result.json"
                q.write_sealed(result, {
                    "schema": release.challenger.FINAL_RESULT_SCHEMA
                })
                results.append(release.challenger._sealed_record(
                    result, release.challenger.FINAL_RESULT_SCHEMA
                ))
                result_values[gate_id] = {"evidence": release._record(evidence)}
                deep_values[evidence.resolve()] = {
                    "maintained_aggregate": release._reference(
                        aggregate, q.FINAL_AGGREGATE_SCHEMA
                    ),
                    "gate_binding": release._reference(
                        binding, q.GATE_BINDING_SCHEMA
                    ),
                    "candidate_commit": COMMIT,
                    "candidate": {
                        "source_sha256": source_record["sha256"],
                        "runtime_sha256": runtime_record["sha256"],
                    },
                    "ordinal": ordinal,
                }
            qualified_path = attempt_root / "dual-qualified.json"
            q.write_sealed(qualified_path, {
                "schema": release.challenger.DUAL_QUALIFICATION_SCHEMA,
                "attempt": 1,
                "status": "two-independent-strict-final-gates-passed",
                "dual_final_plan": release.challenger._sealed_record(
                    dual_plan, release.challenger.DUAL_FINAL_SCHEMA
                ),
                "candidate": normalized_candidate,
                "gate_results": results,
                "candidate_unchanged": True,
                "independent_banks": True,
                "rank4_replacement_authorized": False,
                "upload_authorized": False,
            })
            release_value = {
                "attempt": 1,
                "candidate_commit": COMMIT,
                "candidate": {
                    "runtime": runtime_record,
                    "source": source_record,
                    "architecture": architecture,
                },
            }
            context = {"plan": {"outputs": {"dual_final": str(root / "dual-final")}}}
            dual_state = {
                "path": dual_plan,
                "plan": {
                    "authorization": release.challenger._sealed_record(
                        authorization,
                        release.challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
                    ),
                    "candidate": normalized_candidate,
                },
            }

            def validate_result(_path, *, dual, gate_id):
                self.assertEqual(dual, dual_state["plan"])
                return result_values[gate_id]

            def validate_evidence(path, **_kwargs):
                return deep_values[path.resolve()]

            dual_adapter = mock.Mock()
            dual_adapter.validate_governance_evidence.side_effect = validate_evidence

            with mock.patch.object(
                release.challenger, "validate_campaign", return_value=context
            ), mock.patch.object(
                release.challenger, "validate_dual_final", return_value=dual_state
            ), mock.patch.object(
                release.challenger, "validate_final_result",
                side_effect=validate_result,
            ), mock.patch.object(
                release, "_load", return_value=dual_adapter
            ):
                chain = release._dual_chain(
                    qualified_path, campaign_plan_path=campaign,
                    release_path=release_path, release=release_value,
                )
            self.assertEqual(len(chain["gates"]), 2)
            self.assertEqual(
                chain["authorization"]["release_evidence"],
                release.challenger._sealed_record(
                    release_path, release.RELEASE_EVIDENCE_SCHEMA
                ),
            )

    def test_exactly_one_upload_authorization_binds_release_and_both_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate = root / "candidate.cpp"
            candidate.write_text("candidate\n", encoding="ascii")
            binding = root / "gate-b-binding.json"
            source_record = release._record(candidate, ascii_required=True)
            q.write_sealed(binding, {
                "schema": q.GATE_BINDING_SCHEMA,
                "candidate_commit": COMMIT,
                "candidate": source_record,
            })
            aggregate = root / "gate-b-aggregate.json"
            q.write_sealed(aggregate, {
                "schema": q.FINAL_AGGREGATE_SCHEMA, "verdict": {"passed": True}
            })
            gate_results = []
            gates = []
            for gate_id in ("gate-a", "gate-b"):
                result = root / f"{gate_id}.json"
                q.write_sealed(result, {"schema": release.challenger.FINAL_RESULT_SCHEMA})
                gate_results.append(result)
                gates.append({
                    "gate_id": gate_id,
                    "result_path": result,
                    "aggregate_path": aggregate if gate_id == "gate-b" else root / "a-aggregate.json",
                    "binding_path": binding,
                })
            q.write_sealed(root / "a-aggregate.json", {
                "schema": q.FINAL_AGGREGATE_SCHEMA, "verdict": {"passed": True}
            })
            dual_path = root / "dual-qualified.json"
            q.write_sealed(dual_path, {
                "schema": release.challenger.DUAL_QUALIFICATION_SCHEMA
            })
            release_path = root / "release-evidence.json"
            q.write_sealed(release_path, {"schema": release.RELEASE_EVIDENCE_SCHEMA})
            ci_path = root / "ci.json"
            q.write_sealed(ci_path, {"schema": release.upload.CI_SCHEMA})
            runtime = root / "runtime.json"
            runtime.write_text("runtime")
            runtime_record = release._record(runtime)
            release_value = {
                "attempt": 1,
                "created_at_utc": "2026-09-04T00:03:00Z",
                "candidate_commit": COMMIT,
                "candidate": {
                    "source": source_record,
                    "runtime": runtime_record,
                    "generated_source": source_record,
                    "architecture": {"id": release.challenger.ARCHITECTURE},
                },
                "ci": release._reference(ci_path, release.upload.CI_SCHEMA),
            }
            ci = {
                "run_id": 123,
                "repository": release.upload.REPOSITORY_SLUG,
                "workflow_database_id": release.upload.WORKFLOW_DATABASE_ID,
                "attempt": 1,
                "head_sha": COMMIT,
                "url": f"{release.upload.RUN_URL_PREFIX}123",
                "fetched_at_utc": "2026-09-04T00:02:00Z",
            }
            chain = {
                "qualified": {"completed_at_utc": "2026-09-04T00:04:00Z"},
                "gates": gates,
            }
            output = root / "upload"
            common_patches = (
                mock.patch.object(
                    release, "validate_release_evidence", return_value=release_value
                ),
                mock.patch.object(release, "_dual_chain", return_value=chain),
                mock.patch.object(
                    release.upload, "validate_ci_evidence", return_value=ci
                ),
                mock.patch.object(
                    release.challenger, "validate_campaign", return_value={"plan": {}}
                ),
                mock.patch.object(release.challenger, "load_ledger", return_value=[]),
            )
            with common_patches[0], common_patches[1], common_patches[2], \
                    common_patches[3], common_patches[4]:
                authorization = release.authorize_upload(
                    output, release_evidence_path=release_path,
                    campaign_plan_path=root / "campaign.json", attempt=1,
                    candidate_runtime=runtime, candidate_source=candidate,
                    dual_qualified_path=dual_path,
                    authorized_at_utc="2026-09-04T00:05:00Z",
                )
            self.assertEqual(authorization["schema"], q.UPLOAD_AUTH_SCHEMA)
            self.assertEqual(authorization["uploads_authorized"], 1)
            self.assertFalse(authorization["rank4_replacement_authorized"])
            self.assertTrue(authorization["two_independent_rank4_gates_passed"])
            self.assertEqual(
                authorization["release_evidence"],
                release._reference(release_path, release.RELEASE_EVIDENCE_SCHEMA),
            )
            auth_path = output / "one-upload-authorization.json"
            changed = copy.deepcopy(authorization)
            changed.pop("body_sha256")
            changed["uploads_authorized"] = 2
            auth_path.write_bytes(q.canonical_json_bytes(q.seal(changed)))
            repatches = (
                mock.patch.object(
                    release, "validate_release_evidence", return_value=release_value
                ),
                mock.patch.object(release, "_dual_chain", return_value=chain),
                mock.patch.object(
                    release.upload, "validate_ci_evidence", return_value=ci
                ),
            )
            with repatches[0], repatches[1], repatches[2], self.assertRaisesRegex(
                release.ReleaseBridgeError, "authorization chain changed"
            ):
                release.validate_upload_authorization(
                    output, release_evidence_path=release_path,
                    campaign_plan_path=root / "campaign.json", attempt=1,
                    candidate_runtime=runtime, candidate_source=candidate,
                    dual_qualified_path=dual_path,
                )


if __name__ == "__main__":
    unittest.main()
