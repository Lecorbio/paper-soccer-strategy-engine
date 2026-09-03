import base64
import contextlib
import copy
import hashlib
import io
import json
import pathlib
import os
import tempfile
import unittest
from unittest import mock


from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from submissions.codingame.bots.compact_value_bfm import live_window


q = challenger.qualification


class ChallengerCliForwardingTests(unittest.TestCase):
    def test_sanitize_live_cli_forwards_only_verified_archive_inputs(self):
        with mock.patch.object(
            challenger, "materialize_live_dynamic_exclusion",
            return_value=pathlib.Path("/tmp/live-exclusion.json"),
        ) as materialize, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(challenger.main([
                "sanitize-live-fingerprints",
                "--output", "/tmp/live-exclusion.json",
                "--attempt", "3",
                "--upload-ordinal", "2",
                "--candidate-source-sha256", "a" * 64,
                "--live-reference", "/tmp/live-window.reference.json",
                "--live-data-root", "/tmp/live-archive",
            ]), 0)
        self.assertEqual(materialize.call_args.kwargs, {
            "attempt": 3,
            "upload_ordinal": 2,
            "candidate_source_sha256": "a" * 64,
            "live_reference": pathlib.Path("/tmp/live-window.reference.json"),
            "live_data_root": pathlib.Path("/tmp/live-archive"),
        })

    def test_sanitize_live_cli_rejects_caller_supplied_fingerprints(self):
        with mock.patch.object(
            challenger, "materialize_live_dynamic_exclusion"
        ) as materialize, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                challenger.main([
                    "sanitize-live-fingerprints",
                    "--output", "/tmp/live-exclusion.json",
                    "--attempt", "3",
                    "--upload-ordinal", "2",
                    "--candidate-source-sha256", "a" * 64,
                    "--live-reference", "/tmp/live-window.reference.json",
                    "--live-data-root", "/tmp/live-archive",
                    "--fingerprints", "/tmp/arbitrary-hashes.json",
                ])
        materialize.assert_not_called()

    def test_injected_live_fingerprint_extractor_is_test_only(self):
        with self.assertRaisesRegex(
            challenger.ChallengerError,
            "injected live fingerprint evidence is forbidden in production",
        ):
            challenger.materialize_live_dynamic_exclusion(
                pathlib.Path("/tmp/live-exclusion.json"),
                attempt=3,
                upload_ordinal=2,
                candidate_source_sha256="a" * 64,
                live_reference=pathlib.Path("/tmp/live-window.reference.json"),
                live_data_root=pathlib.Path("/tmp/live-archive"),
                fingerprint_extractor=lambda _reference, _root: {},
            )

    def test_freeze_cli_does_not_forward_release_only_arguments(self):
        values = {
            "output-root": "/tmp/output",
            "candidate-runtime": "/tmp/runtime",
            "candidate-source": "/tmp/source",
            "teacher-runtime": "/tmp/teacher",
            "teacher-manifest": "/tmp/teacher-manifest",
            "mixed-six-exclusion": "/tmp/mixed",
            "fresh-exclusion-receipt": "/tmp/fresh",
            "attempt-zero-recovery-plan": "/tmp/recovery",
            "training-bundle-manifest": "/tmp/bundle",
            "attempt-one-initial-checkpoint": "/tmp/checkpoint",
            "prior-runtime": "/tmp/prior",
            "roots-tsv": "/tmp/roots.tsv",
            "roots-manifest": "/tmp/roots.json",
            "build-manifest": "/tmp/build.json",
            "created-at-utc": "2026-09-03T00:00:00Z",
        }
        arguments = ["freeze"]
        for name, value in values.items():
            arguments.extend((f"--{name}", value))
        arguments.extend(("--training-input", "train=/tmp/train"))
        with mock.patch.object(
            challenger, "freeze_campaign", return_value=pathlib.Path("/tmp/plan")
        ) as freeze, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(challenger.main(arguments), 0)
        keywords = freeze.call_args.kwargs
        self.assertNotIn("release_evidence_path", keywords)
        self.assertNotIn("deployed_source", keywords)

    def test_dual_authorize_cli_forwards_release_and_deployed_source(self):
        with mock.patch.object(
            challenger, "authorize_dual_final",
            return_value=pathlib.Path("/tmp/authorization"),
        ) as authorize, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(challenger.main([
                "authorize-dual-final",
                "--plan", "/tmp/plan",
                "--attempt", "1",
                "--candidate-runtime", "/tmp/runtime",
                "--candidate-source", "/tmp/generated.cpp",
                "--release-evidence", "/tmp/release.json",
                "--deployed-source", "/tmp/deployed.cpp",
                "--created-at-utc", "2026-09-03T00:00:00Z",
            ]), 0)
        self.assertEqual(
            authorize.call_args.kwargs["release_evidence_path"],
            pathlib.Path("/tmp/release.json"),
        )
        self.assertEqual(
            authorize.call_args.kwargs["deployed_source"],
            pathlib.Path("/tmp/deployed.cpp"),
        )


def make_runtime(root, dimensions=(6301, 12, 8, 1)):
    root.mkdir(parents=True, exist_ok=True)
    hidden_one, hidden_two = dimensions[1], dimensions[2]
    counts = {
        "w1": 6301 * hidden_one,
        "w2": hidden_one * hidden_two,
        "w3": hidden_two,
    }
    counts["total"] = sum(counts.values())
    payload = bytes((counts["total"] * 3 + 7) // 8)
    body = {
        "schema": challenger.export_model.RUNTIME_SCHEMA,
        "feature_schema": challenger.export_model.FEATURE_SCHEMA,
        "architecture": {
            "name": challenger.export_model.ELIGIBLE[(hidden_one, hidden_two)],
            "dimensions": list(dimensions),
            "biases": False,
            "activations": challenger.export_model.ACTIVATIONS,
            "payload_layout": challenger.export_model.LAYOUT,
        },
        "quantization": {
            **challenger.export_model.QUANTIZATION,
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


def passing_summary(wins=527, color0=267, color1=260):
    return {
        "games": 1_000,
        "candidate_wins": wins,
        "candidate_color_wins": {"0": color0, "1": color1},
        "failures": {name: 0 for name in q.FAILURE_CATEGORIES},
        "maximum_turns": 320,
        "timing": {"first_max_ms": 800.0, "later_max_ms": 155.0},
        "uncontended_timing": {"first_max_ms": 800.0, "later_max_ms": 155.0},
    }


def make_training_bundle(root):
    bundle = root / "source-training-bundle"
    artifacts = []

    def artifact(role, relative):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}\n")
        record = {
            "role": role,
            "relative_path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        artifacts.append(record)
        return relative

    canonical = {split: [] for split in ("train", "validation", "test")}
    for index in range(3):
        for split in canonical:
            canonical[split].append(artifact(
                f"canonical-r{index}-{split}-manifest",
                f"canonical/r{index}/{split}.json",
            ))
    routes = {"canonical_splits": canonical}
    for source in ("search", "rank4"):
        for phase in ("pilot", "full"):
            name = f"{phase}_{source}_manifests"
            routes[name] = [
                artifact(
                    f"new-{source}-{phase}-{index}-{'test-' if index == 2 else ''}manifest",
                    f"new/{source}/{phase}/{index}.json",
                )
                for index in range(3)
            ]
    routes["common_adjudicator_manifest"] = artifact(
        "common-adjudicator-manifest", "adjudicator/common.json"
    )
    body = {
        "schema": "papersoccer.compact-value-bfm-input-bundle.v1",
        "campaign_id": "compact-value-bfm-20260831-v1",
        "feature_schema": challenger.export_model.FEATURE_SCHEMA,
        "routes": routes,
        "artifacts": artifacts,
        "protected_splits": ["search:test", "rank4:test", "canonical:test"],
        "policy": {
            "protected_tests_locked": True,
            "runtime_uses_source_paths": False,
            "git_required_after_freeze": False,
        },
    }
    manifest = q.seal(body)
    path = bundle / "bundle-manifest.json"
    path.write_bytes(q.canonical_json_bytes(manifest))
    return path


def make_build_manifest(root):
    source = root / "fixture-build-source.py"
    source.write_text("# fixture build source\n")
    binary = root / "fixture-producer"
    binary.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(binary, 0o755)
    body = {
        "schema": challenger.BUILD_MANIFEST_SCHEMA,
        "campaign_id": challenger.CAMPAIGN_ID,
        "status": "clean-source-compiler-binaries-frozen",
        "created_at_utc": "2026-09-04T00:00:00Z",
        "repository": {"root": str(root), "commit": "a" * 40},
        "source_closure": {"fixture-build-source.py": challenger._regular(source)},
        "compiler": challenger._compiler_identity(),
        "binaries": {
            role: {**challenger._regular(binary), "executable": True}
            for role in challenger.BUILD_BINARY_ROLES
        },
        "build_contract": {
            "system": "cmake",
            "configuration": "Release",
            "language_standard": "c++20",
            "sources_clean": True,
            "binaries_built_after_source_freeze": True,
        },
    }
    path = root / "build-manifest.json"
    q.write_sealed(path, body)
    return path


class Fixture:
    def __init__(self, root, *, include_live=True):
        self.base = root.resolve()
        self.output = self.base / "campaign"
        self.runtime = make_runtime(self.base)
        self.prior_runtime = make_runtime(
            self.base / "prior-runtime", dimensions=(6301, 8, 8, 1)
        )
        self.source = self.base / "candidate.cpp"
        self.source.write_text("int main(){return 0;}\n")
        training_npz = self.base / f"{'3' * 64}.npz"
        training_npz.write_bytes(b"named-training-dependency")
        training_sha = hashlib.sha256(training_npz.read_bytes()).hexdigest()
        renamed_npz = self.base / f"{training_sha}.npz"
        training_npz.rename(renamed_npz)
        self.training = self.base / "successor-labels.json"
        self.training.write_text(json.dumps({
            "schema": "papersoccer.jacek-replay-csr-shard.v1",
            "npz": renamed_npz.name,
            "npz_sha256": training_sha,
        }, sort_keys=True))
        self.teacher_runtime = self.base / "teacher.runtime"
        self.teacher_runtime.write_bytes(b"teacher-runtime")
        self.teacher_manifest = self.base / "teacher.runtime.json"
        self.teacher_manifest.write_text("{}")
        self.mixed = self.base / "mixed-exclusion.json"
        self.mixed.write_text("{}")
        self.fresh_receipt = self.base / "fresh-exclusion-receipt.json"
        self.fresh_receipt.write_text("{}")
        self.recovery_plan = self.base / "attempt-zero-recovery-plan.json"
        q.write_sealed(self.recovery_plan, {
            "schema": challenger.RECOVERY_PLAN_SCHEMA,
            "recovery_id": "fixture-recovery",
        })
        self.training_bundle_manifest = make_training_bundle(self.base)
        checkpoint_payload = b"fixture-float-checkpoint"
        checkpoint_sha = hashlib.sha256(checkpoint_payload).hexdigest()
        self.initial_checkpoint = self.base / f"{checkpoint_sha}.float.npz"
        self.initial_checkpoint.write_bytes(checkpoint_payload)
        self.roots_tsv = self.base / "roots.tsv"
        self.roots_tsv.write_text("group_id\tsource\twinner\ttranscript\n")
        self.roots_manifest = self.base / "roots.json"
        self.roots_manifest.write_text("{}")
        self.build_manifest = make_build_manifest(self.base)
        self.protected = self.base / "protected-exclusions.json"
        self.protected.write_text("protected\n")
        self.live = self.base / "live-exclusions.json"
        self.live.write_text("live\n")
        self.plan_path = challenger.freeze_campaign(
            output_root=self.output,
            candidate_runtime=self.runtime,
            candidate_source=self.source,
            rank4_source=challenger.RANK4_PATH,
            teacher_runtime=self.teacher_runtime,
            teacher_manifest=self.teacher_manifest,
            mixed_six_exclusion=self.mixed,
            fresh_exclusion_receipt=self.fresh_receipt,
            attempt_zero_recovery_plan=self.recovery_plan,
            training_bundle_manifest=self.training_bundle_manifest,
            attempt_one_initial_checkpoint=self.initial_checkpoint,
            prior_runtime=self.prior_runtime,
            roots_tsv=self.roots_tsv,
            roots_manifest=self.roots_manifest,
            build_manifest=self.build_manifest,
            training_inputs={"successor-labels": self.training},
            protected_exclusions={"protected": self.protected},
            live_exclusions={"live": self.live} if include_live else {},
            created_at_utc="2026-09-04T00:00:00Z",
            allow_unlisted_test_inputs=True,
        )
        self.context = challenger.validate_campaign(self.plan_path)
        self.active_attempt = 0
        self.pilot_admitted = False

    def start_attempt_one(self):
        if self.active_attempt == 1:
            return
        rejected = self.base / "attempt-zero-rejected.json"
        q.write_sealed(rejected, {
            "schema": challenger.RECOVERY_JOURNAL_SCHEMA,
            "recovery_plan": challenger._sealed_record(
                self.recovery_plan, challenger.RECOVERY_PLAN_SCHEMA
            ),
            "event": "terminal-failure",
            "no_retry": True,
        })
        challenger.record_attempt_zero_result(
            self.plan_path, result_path=rejected,
            created_at_utc="2026-09-04T00:00:30Z",
        )
        challenger.open_next_attempt(
            self.plan_path, attempt=1,
            hypothesis="teacher-guided fixed-architecture pilot",
            intervention="teacher-refresh",
            created_at_utc="2026-09-04T00:00:40Z",
        )
        self.active_attempt = 1

    def pilot_metrics(self, passed=True):
        return {
            "ranking_validation_groups": 125,
            "comparable_exhaustive_validation_groups": 100,
            "comparable_exhaustive_validation_fraction": 0.8,
            "candidate_quantized": True,
            "evaluation_classification": "unseen-root-unprotected",
            "canonical_retention_passed": passed,
            "mean_teacher_regret_reduction_fraction": 0.10 if passed else 0.0,
            "quantized_action_flip_rate": 0.010,
            "scalar_control_action_flip_rate": 0.005,
            "strength_delta_pp": 1.5 if passed else 0.0,
            "teacher_regret_reduction_fraction": 0.10 if passed else 0.0,
            "rank4_screen": {
                "classification": "fresh-unprotected",
                "pairs": 100,
                "games": 200,
                "candidate_wins": 105 if passed else 104,
                "failures": {
                    name: 0 for name in q.FAILURE_CATEGORIES
                },
            },
        }

    def full_metrics(self, passed=True):
        return {
            "ranking_validation_groups": 125,
            "comparable_exhaustive_validation_groups": 100,
            "comparable_exhaustive_validation_fraction": 0.8,
            "actual_clock": {
                "classification": "fresh-unprotected",
                "pairs": 500,
                "games": 1_000,
                "candidate_wins": 550 if passed else 549,
                "candidate_color_wins": {"0": 275, "1": 275},
                "paired_lower_95": 0.501 if passed else 0.49,
                "failures": {name: 0 for name in q.FAILURE_CATEGORIES},
            },
            "strength_delta_pp": 5.0 if passed else 0.0,
            "teacher_regret_reduction_fraction": 0.10 if passed else 0.0,
        }

    def completed_quotas(self, phase):
        return dict(challenger.PHASE_QUOTAS[phase])

    def development_exclusion(self, attempt, phase):
        path = self.output / f"attempt-{attempt}-{phase}-development-exclusion.json"
        q.write_sealed(path, {
            "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": attempt,
            "phase": phase,
            "classification": "unprotected-development-fingerprints",
            "fingerprints": [hashlib.sha256(
                f"{attempt}:{phase}".encode()
            ).hexdigest()],
            "protected_or_live_data_included": False,
        })
        return path

    def outcome_receipt(self, name, *, attempt, phase, metrics, exclusion):
        path = self.output / f"{name}.json"
        phase_reference = (
            self.output / f"phase-plans/attempt-{attempt:03d}/{phase}/phase-reference.json"
        )
        phase_state = challenger.validate_phase_reference(
            phase_reference, self.context["plan"]
        )
        q.write_sealed(path, {
            "schema": challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": attempt,
            "phase": phase,
            "status": "complete",
            "phase_reference": challenger._sealed_record(
                phase_reference, challenger.PHASE_REFERENCE_SCHEMA
            ),
            "schedule": challenger._regular(phase_state["schedule"]),
            "candidate": {
                "runtime_sha256": q.sha256_file(self.runtime),
                "source_sha256": q.sha256_file(self.source),
            },
            "completed_games": challenger.PHASE_TOTALS[phase],
            "completed_quotas": self.completed_quotas(phase),
            "metrics_sha256": challenger.sha256_bytes(
                challenger.canonical_json_bytes(metrics)
            ),
            "development_exclusion": challenger._sealed_record(
                exclusion, challenger.DEVELOPMENT_EXCLUSION_SCHEMA
            ),
            "protected_or_live_metrics_read": False,
            "all_games_finished": True,
        })
        return path

    def admit_pilot(self):
        if self.pilot_admitted:
            return
        self.start_attempt_one()
        challenger.materialize_phase(
            self.plan_path, attempt=1, phase="pilot",
            created_at_utc="2026-09-04T00:01:00Z",
        )
        challenger.record_progress(
            self.plan_path, attempt=1, phase="pilot",
            completed_games=2_000, completed_quotas=self.completed_quotas("pilot"),
            accepted_positions=40_000,
            created_at_utc="2026-09-04T00:01:10Z",
        )
        metrics = self.pilot_metrics()
        exclusion = self.development_exclusion(1, "pilot")
        challenger.record_attempt_outcome(
            self.plan_path, attempt=1, phase="pilot",
            candidate_runtime=self.runtime, candidate_source=self.source,
            outcome_receipt=self.outcome_receipt(
                "pilot-outcome", attempt=1, phase="pilot",
                metrics=metrics, exclusion=exclusion,
            ),
            development_exclusion=exclusion,
            metrics=metrics, strength_delta_pp=1.5,
            teacher_regret_reduction_fraction=0.10,
            created_at_utc="2026-09-04T00:01:20Z",
        )
        self.pilot_admitted = True

    def phase(self, phase):
        self.start_attempt_one()
        if phase == "full":
            self.admit_pilot()
        return challenger.materialize_phase(
            self.plan_path,
            attempt=1,
            phase=phase,
            created_at_utc="2026-09-04T00:01:00Z",
        )

    def authorize_final(self):
        self.phase("full")
        challenger.record_progress(
            self.plan_path,
            attempt=1,
            phase="full",
            completed_games=10_000,
            completed_quotas=self.completed_quotas("full"),
            accepted_positions=200_000,
            created_at_utc="2026-09-04T00:02:00Z",
        )
        metrics = self.full_metrics()
        exclusion = self.development_exclusion(1, "full")
        return challenger.record_attempt_outcome(
            self.plan_path, attempt=1, phase="full",
            candidate_runtime=self.runtime, candidate_source=self.source,
            outcome_receipt=self.outcome_receipt(
                "full-outcome", attempt=1, phase="full",
                metrics=metrics, exclusion=exclusion,
            ),
            development_exclusion=exclusion,
            metrics=metrics, strength_delta_pp=5.0,
            teacher_regret_reduction_fraction=0.10,
            created_at_utc="2026-09-04T00:02:10Z",
        )

    def dual_authorization(self):
        self.authorize_final()
        return challenger.authorize_dual_final(
            self.plan_path, attempt=1,
            candidate_runtime=self.runtime, candidate_source=self.source,
            created_at_utc="2026-09-04T00:03:00Z",
        )

    def prepare_dual(self, *, overlap=False):
        authorization = self.dual_authorization()
        first, second, validator = self.banks(overlap=overlap)
        reference = challenger.prepare_dual_final(
            authorization, plan_path=self.plan_path,
            bank_a=first, bank_b=second,
            created_at_utc="2026-09-04T00:03:02Z",
            bank_validator=validator,
        )
        return reference, first, second, validator

    def final_evidence(self, dual_reference, gate_id, summary, bank):
        dual = challenger.validate_dual_final(
            dual_reference, plan_path=self.plan_path,
            bank_validator=self.banks()[2],
        )
        aggregate = self.base / f"{gate_id}-aggregate.json"
        q.write_sealed(aggregate, {
            "schema": "papersoccer.fixture-final-aggregate.v1",
            "summary": summary,
            "candidate_source_sha256": q.sha256_file(self.source),
            "bank_sha256": q.sha256_file(bank),
            "workers": 4,
        })
        evidence = self.base / f"{gate_id}-evidence.json"
        q.write_sealed(evidence, {
            "schema": challenger.FINAL_GATE_EVIDENCE_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": dual["plan"]["attempt"],
            "gate_id": gate_id,
            "status": "complete",
            "dual_final_plan": challenger._sealed_record(
                dual["path"], challenger.DUAL_FINAL_SCHEMA
            ),
            "candidate": {
                "runtime_sha256": dual["plan"]["candidate"]["runtime"]["sha256"],
                "source_sha256": dual["plan"]["candidate"]["source"]["sha256"],
            },
            "bank": {
                "sha256": q.sha256_file(bank),
                "bytes": bank.stat().st_size,
            },
            "pairs": 500,
            "games": 1_000,
            "workers": 4,
            "threads_per_worker": 1,
            "all_shards_complete": True,
            "summary": summary,
            "aggregate": challenger._regular(aggregate),
        })
        return evidence

    def qualify_dual(self):
        dual_ref, first, second, validator = self.prepare_dual()
        result_a = challenger.record_final_result(
            dual_ref, plan_path=self.plan_path, gate_id="gate-a",
            evidence_path=self.final_evidence(
                dual_ref, "gate-a", passing_summary(), first
            ),
            completed_at_utc="2026-09-04T00:04:00Z",
            bank_validator=validator,
        )
        result_b = challenger.record_final_result(
            dual_ref, plan_path=self.plan_path, gate_id="gate-b",
            evidence_path=self.final_evidence(
                dual_ref, "gate-b", passing_summary(528, 260, 268), second
            ),
            completed_at_utc="2026-09-04T00:05:00Z",
            bank_validator=validator,
        )
        qualified = challenger.complete_dual_final(
            dual_ref, plan_path=self.plan_path,
            result_a=result_a, result_b=result_b,
            completed_at_utc="2026-09-04T00:06:00Z",
            bank_validator=validator,
        )
        return qualified

    def upload_artifacts(self, *, submit_clicks=1, source_sha=None):
        release_root = self.base / "release"
        authorization_path = release_root / "authorization.json"
        candidate = challenger._regular(self.source)
        candidate["sha256"] = source_sha or candidate["sha256"]
        q.write_sealed(authorization_path, {
            "schema": q.UPLOAD_AUTH_SCHEMA,
            "namespace": challenger.NAMESPACE,
            "uploads_authorized": 1,
            "rank4_replacement_authorized": False,
            "candidate_commit": "a" * 40,
            "candidate": candidate,
            "upload_ledger_root": str(release_root.resolve()),
        })
        attestation = release_root / "upload/05-submission-attested.json"
        q.write_sealed(attestation, {
            "schema": q.UPLOAD_EVENT_SCHEMA,
            "namespace": challenger.NAMESPACE,
            "status": "submission-attested",
            "submitted_at_utc": "2026-09-04T00:07:00Z",
            "authorization": q.artifact_reference(
                authorization_path, q.UPLOAD_AUTH_SCHEMA
            ),
            "candidate_commit": "a" * 40,
            "source_sha256": candidate["sha256"],
            "source_bytes": candidate["bytes"],
            "agent_id": 701,
            "submission_id": 801,
            "ambiguity_resolution": None,
            "submit_clicks": submit_clicks,
        })
        return authorization_path, attestation

    def live_artifacts(self, upload_event, *, clean=True, games=90):
        data_root = self.base / "live-window"
        data_root.mkdir(exist_ok=True)
        source_attestation = pathlib.Path(
            upload_event["source_submission_attestation"]["path"]
        )
        receipt = data_root / "receipt.json"
        summary = {
            "status": (
                "complete-accepted-diagnostic" if clean
                else "complete-rejected-focus-operational-failure"
            ),
            "focus_operational_failures": [] if clean else [
                {"game_id": 1, "categories": ["timeout"]}
            ],
            "focus_operational_failure_games": 0 if clean else 1,
        }
        q.write_sealed(receipt, {
            "schema": "papersoccer.compact-value-bfm.live-window-diagnostic.v1",
            "identity": {
                "agent_id": upload_event["agent_id"],
                "submission_id": upload_event["submission_id"],
                "repository_commit": upload_event["candidate_commit"],
                "source_sha256": upload_event["candidate"]["source"]["sha256"],
                "source_bytes": upload_event["candidate"]["source"]["bytes"],
            },
            "submission_attestation": challenger._sealed_record(
                source_attestation, q.UPLOAD_EVENT_SCHEMA
            ),
            "exact_games": games,
            "game_ids": list(range(1, games + 1)),
            "summary": summary,
            "training_eligible": False,
            "rollback_authorized": False,
            "second_upload_authorized": False,
        })
        reference = data_root / "live-window.reference.json"
        q.write_sealed(reference, {
            "schema": "papersoccer.compact-value-bfm.live-window-reference.v1",
            "receipt": {
                "path": str(receipt.resolve()),
                "sha256": q.sha256_file(receipt),
                "body_sha256": q.load_sealed(receipt)["body_sha256"],
            },
            "status": summary["status"],
            "exact_games": games,
        })
        self.live_dynamic_exclusion = None
        self.live_fingerprint_extractor = None
        if not clean:
            game_ids = list(range(1, games + 1))
            fingerprints = sorted([
                hashlib.sha256(f"live:{game_id}".encode()).hexdigest()
                for game_id in game_ids
            ])

            def sealed_reference(path, schema):
                value = q.load_sealed(path, schema)
                return {
                    "path": str(path.resolve()),
                    "sha256": q.sha256_file(path),
                    "body_sha256": value["body_sha256"],
                }

            evidence = q.seal({
                "schema": challenger.LIVE_FINGERPRINT_EVIDENCE_SCHEMA,
                "namespace": challenger.NAMESPACE,
                "status": "verified-live-canonical-fingerprints",
                "live_window_reference": sealed_reference(
                    reference, live_window.WINDOW_REFERENCE_SCHEMA
                ),
                "live_window_receipt": sealed_reference(
                    receipt, live_window.WINDOW_RECEIPT_SCHEMA
                ),
                "collector_manifest": {
                    "path": str((data_root / "collector.json").resolve()),
                    "sha256": "5" * 64,
                    "schema": live_window.GENERIC_BATCH_SCHEMA,
                    "collector_sha256": "6" * 64,
                    "accepted_records_sha256": "7" * 64,
                },
                "source_identity": {
                    "agent_id": upload_event["agent_id"],
                    "submission_id": upload_event["submission_id"],
                    "repository_commit": upload_event["candidate_commit"],
                    "source_sha256": upload_event["candidate"]["source"]["sha256"],
                    "source_bytes": upload_event["candidate"]["source"]["bytes"],
                },
                "exact_games": 90,
                "game_ids": game_ids,
                "game_ids_sha256": challenger.sha256_bytes(
                    challenger.canonical_json_bytes(game_ids)
                ),
                "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
                "boundary_count": len(fingerprints),
                "fingerprints": fingerprints,
                "fingerprint_count": len(fingerprints),
                "fingerprints_sha256": challenger.sha256_bytes(
                    challenger.canonical_json_bytes(fingerprints)
                ),
                "contains_transcripts": False,
                "contains_metrics": False,
                "contains_labels": False,
                "training_eligible": False,
            })
            self.live_fingerprint_extractor = lambda _path, _root: evidence
            self.live_dynamic_exclusion = data_root / "live-fingerprints.json"
            challenger.materialize_live_dynamic_exclusion(
                self.live_dynamic_exclusion,
                attempt=upload_event["attempt"],
                upload_ordinal=upload_event["upload_ordinal"],
                candidate_source_sha256=upload_event["candidate"]["source"]["sha256"],
                live_reference=reference, live_data_root=data_root,
                fingerprint_extractor=self.live_fingerprint_extractor,
                allow_injected_test_evidence=True,
            )
        return (
            reference, data_root, lambda path, root: q.load_sealed(path),
            self.live_fingerprint_extractor,
        )

    def banks(self, overlap=False):
        first = self.base / "bank-a.json"
        second = self.base / "bank-b.json"
        first.write_text("bank-a\n")
        second.write_text("bank-b\n")
        source_binding_path = self.base / "dual-source-binding.json"
        if not source_binding_path.exists():
            q.create_source_binding(
                source_binding_path,
                candidate_source=self.source,
                candidate_commit="a" * 40,
                rank4_source=challenger.RANK4_PATH,
                opponent_source=challenger.RANK4_PATH,
            )
        source_binding = q.artifact_reference(
            source_binding_path, q.SOURCE_BINDING_SCHEMA
        )
        exclusion_sources = [
            {"sha256": record["sha256"]}
            for record in [
                *self.context["inputs"]["protected_exclusions"].values(),
                *self.context["inputs"]["live_exclusions"].values(),
            ]
        ]
        for entry in challenger.load_ledger(self.context["plan"]):
            if entry.get("event") == "attempt-outcome-recorded":
                exclusion_sources.append({
                    "sha256": entry["development_exclusion"]["sha256"]
                })
        seed_paths = []
        for gate_id, seed_hex in (("gate-a", "1" * 64), ("gate-b", "2" * 64)):
            seed_path = self.base / f"{gate_id}-seed.json"
            q.write_sealed(seed_path, {
                "schema": challenger.openings.SEED_SCHEMA,
                "status": "protected-seed-frozen-before-bank-generation",
                "created_at_utc": "2026-09-04T00:03:01Z",
                "source_binding": source_binding,
                "candidate_sha256": q.sha256_file(self.source),
                "seed_256_hex": seed_hex,
            })
            seed_paths.append(seed_path)
        second_exclusions = [*exclusion_sources, {
            "sha256": q.sha256_file(first)
        }]
        first_fingerprints = [
            hashlib.sha256(f"bank-a:{index}".encode()).hexdigest()
            for index in range(500)
        ]
        second_fingerprints = (
            first_fingerprints
            if overlap
            else [
                hashlib.sha256(f"bank-b:{index}".encode()).hexdigest()
                for index in range(500)
            ]
        )
        documents = {
            first.resolve(): {
                "classification": "protected-final",
                "opening_count": 500,
                "exclusion_sources": exclusion_sources,
                "source_binding": source_binding,
                "seed_receipt": q.artifact_reference(
                    seed_paths[0], challenger.openings.SEED_SCHEMA
                ),
                "seed_hex": "1" * 64,
                "openings": [
                    {"fingerprints": {"canonical": fingerprint}}
                    for fingerprint in first_fingerprints
                ],
            },
            second.resolve(): {
                "classification": "protected-final",
                "opening_count": 500,
                "exclusion_sources": second_exclusions,
                "source_binding": source_binding,
                "seed_receipt": q.artifact_reference(
                    seed_paths[1], challenger.openings.SEED_SCHEMA
                ),
                "seed_hex": "2" * 64,
                "openings": [
                    {"fingerprints": {"canonical": fingerprint}}
                    for fingerprint in second_fingerprints
                ],
            },
        }
        return first, second, lambda path: copy.deepcopy(documents[path.resolve()])


class ChallengerCampaignTests(unittest.TestCase):
    def test_freeze_is_content_addressed_and_opens_attempt_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            inputs = fixture.context["inputs"]
            self.assertEqual(inputs["candidate"]["architecture"]["id"], "6301-12-8-1")
            self.assertFalse(inputs["candidate"]["architecture"]["policy_head"])
            self.assertTrue(inputs["candidate"]["runtime"]["route"].startswith("artifacts/"))
            ledger = challenger.load_ledger(fixture.context["plan"])
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["event"], "attempt-opened")
            self.assertEqual(ledger[0]["attempt"], 0)
            self.assertEqual(ledger[0]["route"], "await-attempt-zero-result")
            self.assertEqual(
                fixture.context["plan"]["status"],
                "external-recovery-running-awaiting-result",
            )
            self.assertFalse(fixture.context["plan"]["resources"]["recurring_automation"])

    def test_frozen_bundle_survives_source_removal_and_rejects_bundle_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            bundled = challenger._verify_bundle_record(
                fixture.context["inputs"]["candidate"]["source"],
                input_directory=pathlib.Path(
                    fixture.context["plan"]["outputs"]["input_directory"]
                ),
                label="candidate",
            )
            fixture.source.unlink()
            fixture.training_bundle_manifest.write_bytes(b"removed-source-manifest\n")
            (fixture.base / "fixture-build-source.py").unlink()
            (fixture.base / "fixture-producer").unlink()
            fixture.build_manifest.unlink()
            challenger.validate_campaign(fixture.plan_path)
            bundled.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(challenger.ChallengerError, "bundled bytes changed"):
                challenger.validate_campaign(fixture.plan_path)

    def test_named_training_dependency_tools_and_compiler_are_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            inputs = fixture.context["inputs"]
            dependency = inputs["training_input_dependencies"]["successor-labels"]["npz"]
            dependency_path = challenger._verify_bundle_record(
                dependency,
                input_directory=pathlib.Path(
                    fixture.context["plan"]["outputs"]["input_directory"]
                ),
                label="named dependency",
            )
            primary_path = challenger._verify_bundle_record(
                inputs["training_inputs"]["successor-labels"],
                input_directory=pathlib.Path(
                    fixture.context["plan"]["outputs"]["input_directory"]
                ),
                label="named manifest",
            )
            self.assertEqual(dependency_path.parent, primary_path.parent)
            self.assertTrue(inputs["compiler_bundle"]["route"].startswith("artifacts/"))
            self.assertIn("pilot_pipeline", inputs["audit_tool_bundle"])
            self.assertIn("compact_trainer", inputs["audit_tool_bundle"])
            self.assertIn("teacher_training", inputs["audit_tool_bundle"])
            self.assertIn("dual_final_runner", inputs["audit_tool_bundle"])
            self.assertEqual(
                set(inputs["build_bundle"]["binaries"]),
                challenger.BUILD_BINARY_ROLES,
            )
            self.assertTrue(inputs["build_bundle"]["sources"])
            self.assertEqual(
                inputs["attempt_one_inputs"]["initial_float_checkpoint"]["sha256"],
                hashlib.sha256(b"fixture-float-checkpoint").hexdigest(),
            )

    def test_no_live_evidence_is_an_explicit_valid_frozen_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), include_live=False)
            self.assertEqual(fixture.context["inputs"]["live_exclusions"], {})
            self.assertEqual(
                fixture.context["inputs"]["live_exclusion_state"],
                {"count": 0, "status": "no-live-evidence-exists"},
            )

    def test_attempt_zero_is_external_and_cannot_materialize_a_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(challenger.ChallengerError, "attempt zero is external"):
                challenger.materialize_phase(
                    fixture.plan_path, attempt=0, phase="pilot",
                    created_at_utc="2026-09-04T00:01:00Z",
                )

    def test_attempt_zero_reference_without_recovery_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            forged = fixture.output / "forged-attempt-zero-reference.json"
            q.write_sealed(forged, {
                "schema": challenger.RECOVERY_FINALIST_REFERENCE_SCHEMA,
                "complete": True,
            })
            with self.assertRaisesRegex(challenger.ChallengerError, "unambiguous"):
                challenger.record_attempt_zero_result(
                    fixture.plan_path, result_path=forged,
                    created_at_utc="2026-09-04T00:00:30Z",
                )

    def test_attempt_zero_pass_routes_directly_to_dual_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan_record = challenger._sealed_record(
                fixture.recovery_plan, challenger.RECOVERY_PLAN_SCHEMA
            )
            recovery_result = fixture.output / "attempt-zero-recovery-result.json"
            q.write_sealed(recovery_result, {
                "schema": challenger.RECOVERY_RESULT_SCHEMA,
                "recovery_plan": plan_record,
            })
            finalist = fixture.output / "attempt-zero-finalist.json"
            q.write_sealed(finalist, {
                "schema": challenger.RECOVERY_FINALIST_SCHEMA,
                "recovery_plan": plan_record,
                "recovery_result": challenger._sealed_record(
                    recovery_result, challenger.RECOVERY_RESULT_SCHEMA
                ),
                "candidate": {
                    "runtime": challenger._regular(fixture.runtime),
                    "generated_source": challenger._regular(fixture.source),
                },
            })
            passed = fixture.output / "attempt-zero-finalist-reference.json"
            q.write_sealed(passed, {
                "schema": challenger.RECOVERY_FINALIST_REFERENCE_SCHEMA,
                "complete": True,
                "recovery_plan": plan_record,
                "recovery_result": challenger._sealed_record(
                    recovery_result, challenger.RECOVERY_RESULT_SCHEMA
                ),
                "finalist": challenger._sealed_record(
                    finalist, challenger.RECOVERY_FINALIST_SCHEMA
                ),
            })
            event = challenger.record_attempt_zero_result(
                fixture.plan_path, result_path=passed,
                created_at_utc="2026-09-04T00:00:30Z",
            )
            self.assertEqual(event["adaptation_route"], "prepare-dual-final")
            self.assertTrue(pathlib.Path(event["referenced_finalist"]["path"]).is_file())
            self.assertTrue(pathlib.Path(event["referenced_recovery_result"]["path"]).is_file())
            authorization = challenger.authorize_dual_final(
                fixture.plan_path, attempt=0,
                candidate_runtime=fixture.runtime, candidate_source=fixture.source,
                created_at_utc="2026-09-04T00:00:40Z",
            )
            first, second, validator = fixture.banks()
            reference = challenger.prepare_dual_final(
                authorization, plan_path=fixture.plan_path,
                bank_a=first, bank_b=second,
                created_at_utc="2026-09-04T00:03:02Z",
                bank_validator=validator,
            )
            self.assertTrue(reference.is_file())

    def test_phase_admission_thresholds_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            pilot = fixture.pilot_metrics()
            self.assertTrue(challenger._phase_admission("pilot", pilot))
            for mutate in (
                lambda value: value.update(canonical_retention_passed=False),
                lambda value: value.update(mean_teacher_regret_reduction_fraction=0.099),
                lambda value: value.update(quantized_action_flip_rate=0.0101),
                lambda value: value.update(
                    comparable_exhaustive_validation_groups=99,
                    comparable_exhaustive_validation_fraction=99 / 125,
                ),
                lambda value: value.update(
                    ranking_validation_groups=126,
                    comparable_exhaustive_validation_fraction=100 / 126,
                ),
                lambda value: value["rank4_screen"].update(candidate_wins=104),
                lambda value: value["rank4_screen"].update(failures=1),
            ):
                changed = copy.deepcopy(pilot)
                mutate(changed)
                self.assertFalse(challenger._phase_admission("pilot", changed))
            full = fixture.full_metrics()
            self.assertTrue(challenger._phase_admission("full", full))
            for mutate in (
                lambda value: value["actual_clock"].update(classification="protected"),
                lambda value: value["actual_clock"].update(candidate_wins=549),
                lambda value: value["actual_clock"].update(candidate_color_wins={"0": 264, "1": 286}),
                lambda value: value["actual_clock"].update(paired_lower_95=0.5),
                lambda value: value["actual_clock"].update(failures=1),
            ):
                changed = copy.deepcopy(full)
                mutate(changed)
                self.assertFalse(challenger._phase_admission("full", changed))

    def test_release_evidence_splits_generated_and_deployed_source(self):
        from tools import compact_value_bfm_rank4_teacher_release as release_bridge

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            outcome = fixture.authorize_final()
            deployed = fixture.base / "deployed.cpp"
            deployed.write_text("int main(){return 0;} // deployed\n")
            evidence = fixture.base / "release-evidence.json"
            q.write_sealed(evidence, {
                "schema": challenger.RELEASE_EVIDENCE_SCHEMA,
                "fixture": True,
            })
            released_candidate = {
                "runtime": challenger._regular(fixture.runtime),
                "generated_source": challenger._regular(fixture.source),
                "source": {
                    **challenger._regular(deployed),
                    "ascii": True,
                },
                "architecture": challenger._architecture(fixture.runtime),
            }
            with mock.patch.object(
                release_bridge, "validate_release_evidence",
                return_value={"candidate": released_candidate},
            ):
                authorization_path = challenger.authorize_dual_final(
                    fixture.plan_path, attempt=1,
                    candidate_runtime=fixture.runtime,
                    candidate_source=fixture.source,
                    deployed_source=deployed,
                    release_evidence_path=evidence,
                    created_at_utc="2026-09-04T00:03:00Z",
                )
            authorization = q.load_sealed(
                authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
            )
            self.assertEqual(
                authorization["generated_source"]["sha256"],
                outcome["candidate"]["source"]["sha256"],
            )
            self.assertEqual(
                authorization["candidate"]["source"]["sha256"],
                q.sha256_file(deployed),
            )

    def test_freeze_rejects_non_12x8_runtime_and_exclusion_as_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            runtime = make_runtime(root, (6301, 8, 8, 1))
            source = root / "candidate.cpp"
            source.write_text("x")
            exclusion = root / "excluded.json"
            exclusion.write_text("excluded")
            teacher_runtime = root / "teacher.runtime"
            teacher_runtime.write_text("teacher")
            teacher_manifest = root / "teacher.json"
            teacher_manifest.write_text("{}")
            mixed = root / "mixed.json"
            mixed.write_text("{}")
            fresh = root / "fresh.json"
            fresh.write_text("{}")
            recovery_plan = root / "recovery-plan.json"
            q.write_sealed(recovery_plan, {
                "schema": challenger.RECOVERY_PLAN_SCHEMA,
                "recovery_id": "fixture",
            })
            training_bundle = make_training_bundle(root)
            checkpoint_payload = b"fixture-checkpoint"
            checkpoint = root / (
                hashlib.sha256(checkpoint_payload).hexdigest() + ".float.npz"
            )
            checkpoint.write_bytes(checkpoint_payload)
            prior_runtime = make_runtime(root / "prior", (6301, 8, 8, 1))
            roots_tsv = root / "roots.tsv"
            roots_tsv.write_text("roots\n")
            roots_manifest = root / "roots.json"
            roots_manifest.write_text("{}")
            build_manifest = make_build_manifest(root)
            with self.assertRaisesRegex(challenger.ChallengerError, "6301-12-8-1"):
                challenger.freeze_campaign(
                    output_root=root / "wrong-architecture",
                    candidate_runtime=runtime,
                    candidate_source=source,
                    rank4_source=challenger.RANK4_PATH,
                    teacher_runtime=teacher_runtime,
                    teacher_manifest=teacher_manifest,
                    mixed_six_exclusion=mixed,
                    fresh_exclusion_receipt=fresh,
                    attempt_zero_recovery_plan=recovery_plan,
                    training_bundle_manifest=training_bundle,
                    attempt_one_initial_checkpoint=checkpoint,
                    prior_runtime=prior_runtime,
                    roots_tsv=roots_tsv,
                    roots_manifest=roots_manifest,
                    build_manifest=build_manifest,
                    training_inputs={"labels": exclusion},
                    protected_exclusions={"protected": exclusion},
                    live_exclusions={"live": root / "missing"},
                    created_at_utc="2026-09-04T00:00:00Z",
                    allow_unlisted_test_inputs=True,
                )
            valid_runtime = make_runtime(root)
            live = root / "live.json"
            live.write_text("live")
            with self.assertRaisesRegex(challenger.ChallengerError, "entered training"):
                challenger.freeze_campaign(
                    output_root=root / "leak",
                    candidate_runtime=valid_runtime,
                    candidate_source=source,
                    rank4_source=challenger.RANK4_PATH,
                    teacher_runtime=teacher_runtime,
                    teacher_manifest=teacher_manifest,
                    mixed_six_exclusion=mixed,
                    fresh_exclusion_receipt=fresh,
                    attempt_zero_recovery_plan=recovery_plan,
                    training_bundle_manifest=training_bundle,
                    attempt_one_initial_checkpoint=checkpoint,
                    prior_runtime=prior_runtime,
                    roots_tsv=roots_tsv,
                    roots_manifest=roots_manifest,
                    build_manifest=build_manifest,
                    training_inputs={"labels": exclusion},
                    protected_exclusions={"protected": exclusion},
                    live_exclusions={"live": live},
                    created_at_utc="2026-09-04T00:00:00Z",
                    allow_unlisted_test_inputs=True,
                )

    def test_production_freeze_rejects_unallowlisted_teacher_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            runtime = make_runtime(root)
            training_bundle = make_training_bundle(root)
            files = {}
            for name in ("source", "teacher-runtime", "teacher-manifest", "mixed", "fresh", "recovery-plan", "training", "protected", "live"):
                path = root / name
                path.write_text(name)
                files[name] = path
            files["recovery-plan"].unlink()
            q.write_sealed(files["recovery-plan"], {
                "schema": challenger.RECOVERY_PLAN_SCHEMA,
                "recovery_id": "unallowlisted",
            })
            with self.assertRaisesRegex(challenger.ChallengerError, "explicit production allowlist"):
                challenger.freeze_campaign(
                    output_root=root / "campaign",
                    candidate_runtime=runtime,
                    candidate_source=files["source"],
                    rank4_source=challenger.RANK4_PATH,
                    teacher_runtime=files["teacher-runtime"],
                    teacher_manifest=files["teacher-manifest"],
                    mixed_six_exclusion=files["mixed"],
                    fresh_exclusion_receipt=files["fresh"],
                    attempt_zero_recovery_plan=files["recovery-plan"],
                    training_bundle_manifest=training_bundle,
                    attempt_one_initial_checkpoint=files["training"],
                    prior_runtime=runtime,
                    roots_tsv=files["training"],
                    roots_manifest=files["teacher-manifest"],
                    build_manifest=files["protected"],
                    training_inputs={"labels": files["training"]},
                    protected_exclusions={"protected": files["protected"]},
                    live_exclusions={"live": files["live"]},
                    created_at_utc="2026-09-04T00:00:00Z",
                )
        self.assertEqual(
            challenger.ALLOWLIST["attempt_zero_runtime"]["sha256"],
            "130c6ef1d2311a76c7a94fd144a805aa22477a32bced59a8079021e4293ea336",
        )

    def test_pilot_and_full_plans_have_exact_fivefold_quotas(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            pilot_ref = fixture.phase("pilot")
            full_ref = fixture.phase("full")
            pilot = challenger.validate_phase_reference(
                pilot_ref, fixture.context["plan"]
            )["phase"]
            full = challenger.validate_phase_reference(
                full_ref, fixture.context["plan"]
            )["phase"]
            pilot_state = challenger.validate_phase_reference(
                pilot_ref, fixture.context["plan"]
            )
            self.assertEqual(pilot["games"], 2_000)
            self.assertEqual(full["games"], 10_000)
            self.assertEqual(pilot["quotas"], challenger.PILOT_QUOTAS)
            self.assertEqual(pilot["quotas"]["student-p1-vs-rank4"], 500)
            self.assertEqual(pilot["quotas"]["student-p2-vs-rank4"], 500)
            self.assertEqual(pilot["quotas"]["student-selfplay"], 500)
            self.assertFalse(any("jacek-nn" in name for name in pilot["quotas"]))
            self.assertEqual(
                pilot_state["schedule"].read_text().splitlines()[0],
                "game_ordinal\tactor_mode\tbase_seed",
            )
            self.assertEqual(
                pilot["actor_bindings"]["incumbent_role"],
                "accepted-f7bdb201-rank4-teacher",
            )
            self.assertEqual(
                full["quotas"],
                {name: count * 5 for name, count in pilot["quotas"].items()},
            )
            self.assertEqual(fixture.phase("pilot"), pilot_ref)
            self.assertEqual(
                len(challenger.load_ledger(fixture.context["plan"])), 7
            )
            frozen_bundle = fixture.context["inputs"]["training_bundle"]
            self.assertGreater(frozen_bundle["protected_test_count"], 0)
            protected_routes = {
                record["route"] for record in frozen_bundle["protected_test_artifacts"]
            }
            exposed = frozen_bundle["exposed_routes"]
            self.assertTrue(all(
                record["route"] not in protected_routes
                for values in exposed.values()
                for record in (values if isinstance(values, list) else [values])
            ))

    def test_progress_and_no_progress_routes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.phase("pilot")
            first = challenger.record_progress(
                fixture.plan_path, attempt=1, phase="pilot",
                completed_games=0,
                completed_quotas={name: 0 for name in challenger.PILOT_QUOTAS},
                accepted_positions=0,
                created_at_utc="2026-09-04T00:02:00Z",
            )
            second = challenger.record_progress(
                fixture.plan_path, attempt=1, phase="pilot",
                completed_games=0,
                completed_quotas={name: 0 for name in challenger.PILOT_QUOTAS},
                accepted_positions=0,
                created_at_utc="2026-09-04T00:03:00Z",
            )
            third = challenger.record_progress(
                fixture.plan_path, attempt=1, phase="pilot",
                completed_games=0,
                completed_quotas={name: 0 for name in challenger.PILOT_QUOTAS},
                accepted_positions=0,
                created_at_utc="2026-09-04T00:04:00Z",
            )
            self.assertEqual(first["adaptation_route"], "explicit-resume-current-phase")
            self.assertEqual(second["consecutive_no_progress"], 2)
            self.assertEqual(third["adaptation_route"], "explicit-resume-current-phase")
            with self.assertRaisesRegex(challenger.ChallengerError, "did not authorize"):
                challenger.open_next_attempt(
                    fixture.plan_path, attempt=2,
                    hypothesis="not authorized",
                    intervention="same-contract",
                    created_at_utc="2026-09-04T00:05:00Z",
                )

    def test_production_progress_rejects_caller_supplied_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.phase("pilot")
            production = copy.deepcopy(fixture.context)
            production["inputs"]["production_allowlist_enforced"] = True
            with mock.patch.object(
                challenger, "validate_campaign", return_value=production
            ), self.assertRaisesRegex(
                challenger.ChallengerError, "derive from pipeline receipts"
            ):
                challenger.record_progress(
                    fixture.plan_path, attempt=1, phase="pilot",
                    completed_games=2_000,
                    completed_quotas=fixture.completed_quotas("pilot"),
                    accepted_positions=40_000,
                    created_at_utc="2026-09-04T00:02:00Z",
                )

    def test_two_completed_no_improvement_attempts_route_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.start_attempt_one()
            outcomes = []
            for attempt in (1, 2):
                challenger.materialize_phase(
                    fixture.plan_path, attempt=attempt, phase="pilot",
                    created_at_utc=f"2026-09-04T00:0{attempt}:00Z",
                )
                challenger.record_progress(
                    fixture.plan_path, attempt=attempt, phase="pilot",
                    completed_games=2_000,
                    completed_quotas=fixture.completed_quotas("pilot"),
                    accepted_positions=40_000,
                    created_at_utc=f"2026-09-04T00:1{attempt}:00Z",
                )
                metrics = fixture.pilot_metrics(False)
                exclusion = fixture.development_exclusion(attempt, "pilot")
                receipt = fixture.outcome_receipt(
                    f"failed-attempt-{attempt}", attempt=attempt,
                    phase="pilot", metrics=metrics, exclusion=exclusion,
                )
                outcome = challenger.record_attempt_outcome(
                    fixture.plan_path, attempt=attempt, phase="pilot",
                    candidate_runtime=fixture.runtime, candidate_source=fixture.source,
                    outcome_receipt=receipt,
                    development_exclusion=exclusion,
                    metrics=metrics,
                    strength_delta_pp=0.0,
                    teacher_regret_reduction_fraction=0.0,
                    created_at_utc=f"2026-09-04T00:2{attempt}:00Z",
                )
                outcomes.append(outcome)
                if attempt == 1:
                    challenger.open_next_attempt(
                        fixture.plan_path, attempt=2,
                        hypothesis="repeat exact pilot once",
                        intervention="same-contract",
                        created_at_utc="2026-09-04T00:30:00Z",
                    )
            self.assertEqual(outcomes[0]["consecutive_no_improvement"], 1)
            self.assertEqual(outcomes[0]["adaptation_route"], "open-next-attempt-same-contract")
            self.assertEqual(outcomes[1]["consecutive_no_improvement"], 2)
            self.assertEqual(
                outcomes[1]["adaptation_route"],
                "open-next-attempt-attribution-adaptation",
            )
            attribution = fixture.base / "attribution.json"
            q.write_sealed(attribution, {
                "schema": challenger.ATTRIBUTION_EVIDENCE_SCHEMA,
                "campaign_id": challenger.CAMPAIGN_ID,
                "classification": "teacher-ranking-gap",
                "completed_no_improvement_attempts": 2,
                "selected_intervention": (
                    "teacher-ranking-gap-hard-state-density"
                ),
                "next_attempt": 3,
                "protected_or_live_data_read": False,
            })
            challenger.open_next_attempt(
                fixture.plan_path, attempt=3,
                hypothesis="increase hard-state density after attribution",
                intervention="teacher-ranking-gap-hard-state-density",
                attribution_receipt=attribution,
                created_at_utc="2026-09-04T00:31:00Z",
            )
            challenger.materialize_phase(
                fixture.plan_path, attempt=3, phase="pilot",
                created_at_utc="2026-09-04T00:32:00Z",
            )
            challenger.record_progress(
                fixture.plan_path, attempt=3, phase="pilot",
                completed_games=2_000,
                completed_quotas=fixture.completed_quotas("pilot"),
                accepted_positions=40_000,
                created_at_utc="2026-09-04T00:33:00Z",
            )
            metrics = fixture.pilot_metrics(False)
            exclusion = fixture.development_exclusion(3, "pilot")
            third = challenger.record_attempt_outcome(
                fixture.plan_path, attempt=3, phase="pilot",
                candidate_runtime=fixture.runtime, candidate_source=fixture.source,
                outcome_receipt=fixture.outcome_receipt(
                    "failed-attempt-3", attempt=3, phase="pilot",
                    metrics=metrics, exclusion=exclusion,
                ),
                development_exclusion=exclusion, metrics=metrics,
                strength_delta_pp=0.0,
                teacher_regret_reduction_fraction=0.0,
                created_at_utc="2026-09-04T00:34:00Z",
            )
            self.assertEqual(third["consecutive_no_improvement"], 1)
            self.assertEqual(
                third["adaptation_route"], "open-next-attempt-same-contract"
            )

    def test_progress_counters_cannot_regress(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.phase("pilot")
            challenger.record_progress(
                fixture.plan_path, attempt=1, phase="pilot",
                completed_games=10,
                completed_quotas={
                    **{name: 0 for name in challenger.PILOT_QUOTAS},
                    "student-selfplay": 10,
                },
                accepted_positions=100,
                created_at_utc="2026-09-04T00:02:00Z",
            )
            with self.assertRaisesRegex(challenger.ChallengerError, "regressed"):
                challenger.record_progress(
                    fixture.plan_path, attempt=1, phase="pilot",
                    completed_games=9,
                    completed_quotas={
                        **{name: 0 for name in challenger.PILOT_QUOTAS},
                        "student-selfplay": 9,
                    },
                    accepted_positions=100,
                    created_at_utc="2026-09-04T00:03:00Z",
                )
            wrong = {name: 0 for name in challenger.PILOT_QUOTAS}
            wrong["student-selfplay"] = 8
            with self.assertRaisesRegex(challenger.ChallengerError, "quotas"):
                challenger.record_progress(
                    fixture.plan_path, attempt=1, phase="pilot",
                    completed_games=10, completed_quotas=wrong,
                    accepted_positions=101,
                    created_at_utc="2026-09-04T00:04:00Z",
                )

    def test_append_only_ledger_detects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            ledger_root = pathlib.Path(fixture.context["plan"]["outputs"]["ledger"])
            entry = next(ledger_root.iterdir())
            entry.write_bytes(b"tampered\n")
            with self.assertRaises(ValueError):
                challenger.load_ledger(fixture.context["plan"])

    def test_dual_final_requires_disjoint_banks_and_full_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            first, second, validator = fixture.banks()
            with self.assertRaisesRegex(challenger.ChallengerError, "authorization is invalid"):
                challenger.prepare_dual_final(
                    fixture.output / "missing-authorization.json",
                    plan_path=fixture.plan_path,
                    bank_a=first, bank_b=second,
                    created_at_utc="2026-09-04T00:03:00Z",
                    bank_validator=validator,
                )
            authorization = fixture.dual_authorization()
            _first, _second, overlap_validator = fixture.banks(overlap=True)
            with self.assertRaisesRegex(challenger.ChallengerError, "not independent"):
                challenger.prepare_dual_final(
                    authorization, plan_path=fixture.plan_path,
                    bank_a=first, bank_b=second,
                    created_at_utc="2026-09-04T00:03:02Z",
                    bank_validator=overlap_validator,
                )

    def test_two_independent_passing_gates_require_unchanged_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            dual_ref, first, second, validator = fixture.prepare_dual()
            summary_a = passing_summary()
            summary_b = passing_summary(528, 260, 268)
            result_a = challenger.record_final_result(
                dual_ref, plan_path=fixture.plan_path, gate_id="gate-a",
                evidence_path=fixture.final_evidence(
                    dual_ref, "gate-a", summary_a, first
                ),
                completed_at_utc="2026-09-04T00:04:00Z",
                bank_validator=validator,
            )
            result_b = challenger.record_final_result(
                dual_ref, plan_path=fixture.plan_path, gate_id="gate-b",
                evidence_path=fixture.final_evidence(
                    dual_ref, "gate-b", summary_b, second
                ),
                completed_at_utc="2026-09-04T00:05:00Z",
                bank_validator=validator,
            )
            qualified = challenger.complete_dual_final(
                dual_ref, plan_path=fixture.plan_path,
                result_a=result_a, result_b=result_b,
                completed_at_utc="2026-09-04T00:06:00Z",
                bank_validator=validator,
            )
            value = q.load_sealed(qualified, challenger.DUAL_QUALIFICATION_SCHEMA)
            self.assertTrue(value["candidate_unchanged"])
            self.assertTrue(value["independent_banks"])
            self.assertEqual(len(value["gate_results"]), 2)
            self.assertFalse(value["rank4_replacement_authorized"])
            self.assertEqual(
                challenger.load_ledger(fixture.context["plan"])[-1]["event"],
                "dual-final-qualified",
            )
            with self.assertRaisesRegex(challenger.ChallengerError, "clean exact-90"):
                challenger.complete_campaign(
                    fixture.plan_path,
                    completed_at_utc="2026-09-04T00:07:00Z",
                )

    def test_each_final_gate_enforces_527_and_260_and_zero_failures(self):
        cases = [
            passing_summary(526, 266, 260),
            passing_summary(527, 267, 259),
            passing_summary(),
        ]
        cases[2]["failures"]["crash"] = 1
        for summary in cases:
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                dual_ref, first, _second, validator = fixture.prepare_dual()
                result = challenger.record_final_result(
                    dual_ref, plan_path=fixture.plan_path, gate_id="gate-a",
                    evidence_path=fixture.final_evidence(
                        dual_ref, "gate-a", summary, first
                    ),
                    completed_at_utc="2026-09-04T00:04:00Z",
                    bank_validator=validator,
                )
                value = q.load_sealed(result, challenger.FINAL_RESULT_SCHEMA)
                self.assertFalse(value["passed"])
                event = challenger.load_ledger(fixture.context["plan"])[-1]
                self.assertEqual(
                    event["adaptation_route"],
                    "open-next-attempt-protected-rejection",
                )
                opened = challenger.open_next_attempt(
                    fixture.plan_path, attempt=2,
                    hypothesis="fresh leakage-isolated attempt after protected rejection",
                    intervention="protected-rejection-clean-restart",
                    created_at_utc="2026-09-04T00:05:00Z",
                )
                self.assertEqual(len(opened["dynamic_exclusions"]), 2)
                self.assertTrue(all(
                    item["classification"]
                    == "protected-final-canonical-fingerprints"
                    for item in opened["dynamic_exclusions"]
                ))
                phase_reference = challenger.materialize_phase(
                    fixture.plan_path, attempt=2, phase="pilot",
                    created_at_utc="2026-09-04T00:05:01Z",
                )
                phase = challenger.validate_phase_reference(
                    phase_reference, fixture.context["plan"]
                )["phase"]
                self.assertEqual(
                    phase["dynamic_exclusions"], opened["dynamic_exclusions"]
                )

    def test_dual_qualification_requires_upload_and_clean_90_before_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.qualify_dual()
            _authorization, attestation = fixture.upload_artifacts()
            upload_event = challenger.record_upload_attestation(
                fixture.plan_path, submission_attestation=attestation,
                created_at_utc="2026-09-04T00:07:01Z",
            )
            reference, data_root, validator, _extractor = fixture.live_artifacts(
                upload_event
            )
            live_event = challenger.record_live_window(
                fixture.plan_path, live_reference=reference,
                live_data_root=data_root,
                dynamic_exclusion_path=fixture.live_dynamic_exclusion,
                created_at_utc="2026-09-04T12:00:00Z",
                live_validator=validator,
            )
            self.assertTrue(live_event["passed"])
            completion = challenger.complete_campaign(
                fixture.plan_path,
                completed_at_utc="2026-09-04T12:00:01Z",
            )
            self.assertEqual(
                q.load_sealed(completion, challenger.CAMPAIGN_COMPLETION_SCHEMA)["proof"],
                {
                    "strict_final_gates": 2,
                    "candidate_unchanged": True,
                    "exact_live_games": 90,
                    "focus_operational_failure_games": 0,
                    "upload_ordinal": 1,
                },
            )
            self.assertEqual(
                challenger.load_ledger(fixture.context["plan"])[-1]["event"],
                "campaign-complete",
            )

    def test_live_failure_needs_explicit_single_use_additional_upload_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.qualify_dual()
            _authorization, attestation = fixture.upload_artifacts()
            upload_event = challenger.record_upload_attestation(
                fixture.plan_path, submission_attestation=attestation,
                created_at_utc="2026-09-04T00:07:01Z",
            )
            reference, data_root, validator, extractor = fixture.live_artifacts(
                upload_event, clean=False
            )
            persisted = extractor(reference, data_root)
            changed = dict(persisted)
            changed["fingerprints"] = persisted["fingerprints"][:-1]
            changed["fingerprint_count"] = len(changed["fingerprints"])
            changed["fingerprints_sha256"] = challenger.sha256_bytes(
                challenger.canonical_json_bytes(changed["fingerprints"])
            )
            changed.pop("body_sha256")
            changed = q.seal(changed)
            with self.assertRaisesRegex(
                challenger.ChallengerError,
                "not the trusted rejected window",
            ):
                challenger.record_live_window(
                    fixture.plan_path, live_reference=reference,
                    live_data_root=data_root,
                    dynamic_exclusion_path=fixture.live_dynamic_exclusion,
                    created_at_utc="2026-09-04T11:59:59Z",
                    live_validator=validator,
                    live_fingerprint_extractor=(
                        lambda _path, _root: changed
                    ),
                    allow_injected_test_evidence=True,
                )
            live_event = challenger.record_live_window(
                fixture.plan_path, live_reference=reference,
                live_data_root=data_root,
                dynamic_exclusion_path=fixture.live_dynamic_exclusion,
                created_at_utc="2026-09-04T12:00:00Z",
                live_validator=validator,
                live_fingerprint_extractor=extractor,
                allow_injected_test_evidence=True,
            )
            self.assertFalse(live_event["passed"])
            with self.assertRaisesRegex(challenger.ChallengerError, "did not authorize"):
                challenger.open_next_attempt(
                    fixture.plan_path, attempt=2,
                    hypothesis="forbidden implicit post-live retry",
                    intervention="explicit-post-live-restart",
                    created_at_utc="2026-09-04T12:00:01Z",
                )
            explicit = fixture.base / "explicit-additional-upload.json"
            q.write_sealed(explicit, {
                "schema": challenger.ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA,
                "campaign_id": challenger.CAMPAIGN_ID,
                "previous_attempt": 1,
                "next_attempt": 2,
                "next_upload_ordinal": 2,
                "rejected_live_reference": live_event["source_live_reference"],
                "rejected_live_dynamic_exclusion": live_event[
                    "dynamic_exclusion"
                ],
                "explicit_user_authorization": True,
                "attempt_openings_authorized": 1,
                "additional_uploads_authorized": 1,
                "protected_or_live_data_training_allowed": False,
                "automatic_action": False,
            })
            wrong_reference = fixture.base / "wrong-live-authorization.json"
            wrong_body = q.load_sealed(explicit)
            wrong_body.pop("body_sha256")
            wrong_body["rejected_live_reference"] = {
                **live_event["source_live_reference"],
                "sha256": "f" * 64,
            }
            q.write_sealed(wrong_reference, wrong_body)
            with self.assertRaisesRegex(
                challenger.ChallengerError,
                "additional upload authorization changed",
            ):
                challenger.authorize_additional_upload(
                    fixture.plan_path, authorization_path=wrong_reference,
                    created_at_utc="2026-09-04T12:00:01Z",
                    live_fingerprint_extractor=extractor,
                    allow_injected_test_evidence=True,
                )
            challenger.authorize_additional_upload(
                fixture.plan_path, authorization_path=explicit,
                created_at_utc="2026-09-04T12:00:02Z",
                live_fingerprint_extractor=extractor,
                allow_injected_test_evidence=True,
            )
            opened = challenger.open_next_attempt(
                fixture.plan_path, attempt=2,
                hypothesis="new offline attempt after explicitly authorized live failure",
                intervention="explicit-post-live-restart",
                created_at_utc="2026-09-04T12:00:03Z",
            )
            self.assertEqual(opened["attempt"], 2)
            self.assertEqual(len(opened["dynamic_exclusions"]), 1)
            self.assertEqual(
                opened["dynamic_exclusions"][0]["classification"],
                "live-diagnostic-canonical-fingerprints",
            )

    def test_help_documents_resources_and_no_recurring_automation(self):
        self.assertEqual(challenger.RESOURCE_LIMITS["maximum_concurrent_game_workers"], 8)
        self.assertEqual(challenger.RESOURCE_LIMITS["maximum_concurrent_training_seeds"], 2)
        self.assertEqual(challenger.RESOURCE_LIMITS["strict_final_workers"], 4)
        self.assertEqual(challenger.RESOURCE_LIMITS["uncontended_timing_workers"], 1)
        self.assertFalse(challenger.RESOURCE_LIMITS["recurring_automation"])
        self.assertIn("recurring automation", challenger.__doc__)


if __name__ == "__main__":
    unittest.main()
