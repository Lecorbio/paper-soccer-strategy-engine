import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_campaign.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_campaign", TOOL)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(campaign)
base = campaign.base

RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
COMMIT = "b" * 40


def digest(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fingerprints(label, count):
    return sorted(digest(f"{label}-{index}") for index in range(count))


def metric(candidate_id, pairs, wins, color0, color1, latency=10.0,
           failures=0, **extra):
    return {
        "candidate_id": candidate_id,
        "pairs": pairs,
        "games": pairs * 2,
        "wins": wins,
        "color_wins": {"0": color0, "1": color1},
        "failures": failures,
        "latency_ms": latency,
        **extra,
    }


class FrozenFamilyFixture:
    def __init__(self, root):
        self.root = root
        self.run = root / "family-run"
        self.bundle = root / "bundle-manifest.json"
        self.bundle.write_text("{}\n", encoding="ascii")
        self.trainer = campaign._trainer_module()
        self.workflow = campaign._workflow_module()
        self.generated_source = root / "generated-source.cpp"
        self.generated_source.write_text(
            "int main(){return 0;}\n", encoding="ascii"
        )
        self.float_checkpoint = None
        self.selected_runtime = None
        self.selected_selection = None
        self.selected_selection_value = None
        self.campaigns = []
        source_preflight = {}
        for architecture in campaign.WORKFLOW_ARCHITECTURES:
            source_preflight[architecture] = {
                "architecture": architecture,
                "eligible": True,
                "limit": 95_000,
                "complete_source_ascii_characters": 90_000,
            }
        for architecture, arm in (
            *campaign.WORKFLOW_DEPLOYABLE_ARMS,
            campaign.WORKFLOW_CONTROL,
        ):
            name = f"{architecture}--{arm}"
            output = self.run / "campaigns" / name
            runtime_path = output / "runtimes" / "selected.runtime.json"
            runtime = base.write_sealed(runtime_path, {
                "schema": self.trainer.RUNTIME_SCHEMA,
                "architecture": architecture,
                "arm": arm,
            })
            float_path = output / "float-checkpoints" / "selected.float.npz"
            float_path.parent.mkdir(parents=True, exist_ok=True)
            float_path.write_bytes((name + " float").encode("ascii"))
            seed_receipt_path = output / "seed-receipts" / "selected.json"
            seed_receipt = base.write_sealed(seed_receipt_path, {
                "schema": self.trainer.SEED_RECEIPT_SCHEMA,
                "float_checkpoint": {
                    "path": str(float_path.relative_to(output)),
                    "bytes": float_path.stat().st_size,
                    "sha256": base.sha256_file(float_path),
                },
            })
            selection_path = output / "selections" / "selected.json"
            selection = base.write_sealed(selection_path, {
                "schema": self.trainer.SELECTION_SCHEMA,
                "architecture": architecture,
                "arm": arm,
                "seed": 20260909,
                "status": "offline-evaluator-rejected",
                "deployment_eligible": False,
                "offline_gate": {"passed": False},
                "source_size_eligibility": {
                    "passed": True, "maximum_ascii_bytes": 95_000,
                },
                "selected_seed_receipt": {
                    "path": str(seed_receipt_path.relative_to(output)),
                    "sha256": base.sha256_file(seed_receipt_path),
                    "body_sha256": seed_receipt["body_sha256"],
                },
                "protected_tests_opened": False,
                "game_gated": False,
                "rank4_control_never_deployment_eligible": (
                    arm == "rank4-control"
                ),
            })
            source = None
            if arm != "rank4-control":
                source_bytes = self.generated_source.read_bytes()
                source = {
                    "architecture": architecture,
                    "eligible": True,
                    "limit": 95_000,
                    "complete_source_ascii_characters": len(source_bytes),
                    "complete_source_sha256": base.sha256_bytes(source_bytes),
                    "runtime_file_sha256": base.sha256_file(runtime_path),
                    "runtime_body_sha256": runtime["body_sha256"],
                }
            record = {
                "name": name,
                "architecture": architecture,
                "arm": arm,
                "campaign_output": str(output.relative_to(self.run)),
                "protected_tests_opened": False,
                "rank4_control_never_deployment_eligible": arm == "rank4-control",
                "selection": {
                    "path": str(selection_path.relative_to(self.run)),
                    "sha256": base.sha256_file(selection_path),
                    "body_sha256": selection["body_sha256"],
                },
                "runtime": {
                    "path": str(runtime_path.relative_to(self.run)),
                    "sha256": base.sha256_file(runtime_path),
                    "body_sha256": runtime["body_sha256"],
                },
                "exact_complete_source": source,
            }
            self.campaigns.append(record)
            if (architecture, arm) == ("capacity-12x8", "search-target"):
                self.float_checkpoint = float_path
                self.selected_runtime = runtime_path
                self.selected_selection = selection_path
                self.selected_selection_value = selection
        self.receipt = base.seal({
            "schema": self.workflow.RUN_RECEIPT_SCHEMA,
            "campaign_order": list(campaign.WORKFLOW_CAMPAIGN_ORDER),
            "campaigns": self.campaigns,
            "source_size_preflight": source_preflight,
            "all_seven_campaigns_complete": True,
            "protected_tests_opened": False,
            "protected_tests_locked": True,
            "game_gated": False,
        })
        self.receipt_path = self.run / "run-receipts" / "receipt.json"
        base.atomic_write_once(
            self.receipt_path, base.canonical_json_bytes(self.receipt)
        )
        self.reference_path = self.run / "run-state" / "run-reference.json"
        base.write_sealed(self.reference_path, {
            "schema": self.workflow.RUN_REFERENCE_SCHEMA,
            "receipt": {"path": str(self.receipt_path.relative_to(self.run))},
        })

    def workflow_patch(self):
        return mock.patch.object(
            self.workflow, "verify_family_run", return_value=self.receipt,
        )

    def record_failure(self, output):
        with self.workflow_patch():
            artifact = campaign.record_offline_family_failure(
                output, bundle_manifest=self.bundle,
                run_output_directory=self.run,
                run_reference=self.reference_path,
                recorded_at_utc="2026-08-31T15:30:00Z",
            )
        return campaign.offline_family_failure_path(output, artifact), artifact

    def operational_evidence(self):
        binary = self.root / "operational-bot"
        binary.write_bytes(b"binary")
        compiler = self.root / "compiler"
        compiler.write_bytes(b"compiler")
        binary_reference = campaign._regular_file_reference(
            binary, label="binary"
        )
        runtime_sha = base.sha256_file(self.selected_runtime)
        source_sha = base.sha256_file(self.generated_source)
        empty = digest("")
        build_path = self.root / "build-evidence.json"
        base.write_sealed(build_path, {
            "schema": campaign.OPERATIONAL_BUILD_EVIDENCE_SCHEMA,
            "namespace": campaign.NAMESPACE,
            "status": "build-and-tests-passed",
            "source_sha256": source_sha,
            "runtime_sha256": runtime_sha,
            "binary": binary_reference,
            "compiler": {
                "executable": campaign._regular_file_reference(
                    compiler, label="compiler"
                ),
                "version": "fixture compiler 1.0",
                "version_sha256": hashlib.sha256(
                    b"fixture compiler 1.0"
                ).hexdigest(),
            },
            "commands": {
                name: {
                    "argv": [name], "exit_code": 0,
                    "stdout_sha256": empty, "stderr_sha256": empty,
                }
                for name in ("compile", "tests")
            },
            "protected_tests_opened": False,
        })
        protocol_path = self.root / "protocol-evidence.json"
        base.write_sealed(protocol_path, {
            "schema": campaign.OPERATIONAL_PROTOCOL_EVIDENCE_SCHEMA,
            "namespace": campaign.NAMESPACE,
            "status": "protocol-both-player-roles-passed",
            "source_sha256": source_sha,
            "runtime_sha256": runtime_sha,
            "binary": binary_reference,
            "player_roles": {"0": True, "1": True},
            "failures": 0,
            "protected_tests_opened": False,
        })
        preflight = campaign._preflight_module()
        samples = []
        for count in preflight.PROCESS_COUNTS:
            for color in (0, 1):
                for replica in range(count):
                    samples.append({
                        "process_count": count, "color": color,
                        "replica": replica, "first_ms": 1.0,
                        "later_max_ms": 1.0, "stdout_sha256": empty,
                        "stderr_sha256": empty,
                    })
        timing_path = self.root / "timing-evidence.json"
        base.write_sealed(timing_path, {
            "schema": preflight.TIMING_SCHEMA,
            "probe_sha256": digest("probe"),
            "first_limit_exclusive_ms": preflight.FIRST_LIMIT_MS,
            "later_limit_exclusive_ms": preflight.LATER_LIMIT_MS,
            "samples": samples,
        })
        return build_path, protocol_path, timing_path


def development_payload():
    banks = {}
    for name, pairs in (
        ("model_screen", 100), ("tuple_screen", 100),
        ("tuple_confirmation", 250), ("profile_screen", 100),
        ("profile_confirmation", 250), ("actual_clock", 200),
    ):
        banks[name] = {
            "bank_id": name,
            "pairs": pairs,
            "fingerprints": fingerprints(name, pairs),
            "transcripts": ["0/1/2/3/4/5/6/7/0/1/2/3"] * pairs,
            "primitive_ply_counts": [12] * pairs,
        }

    model_rows = [
        metric("primary-search", 100, 130, 65, 65, 9.0,
               architecture=campaign.PRIMARY_ARCHITECTURE,
               target="search-target", source_bytes=90_000,
               artifact_sha256=digest("primary-search"),
               deployment_eligible=True),
        metric("primary-teacher", 100, 129, 64, 65, 8.0,
               architecture=campaign.PRIMARY_ARCHITECTURE,
               target="teacher-assisted", source_bytes=90_100,
               artifact_sha256=digest("primary-teacher"),
               deployment_eligible=True),
        metric("neutral-search", 100, 128, 64, 64, 8.0,
               architecture=campaign.SOURCE_NEUTRAL_ARCHITECTURE,
               target="search-target", source_bytes=91_000,
               artifact_sha256=digest("neutral-search"),
               deployment_eligible=True),
        metric("neutral-teacher", 100, 127, 63, 64, 8.0,
               architecture=campaign.SOURCE_NEUTRAL_ARCHITECTURE,
               target="teacher-assisted", source_bytes=91_100,
               artifact_sha256=digest("neutral-teacher"),
               deployment_eligible=True),
        metric("rank4-control", 100, 140, 70, 70, 7.0,
               architecture=campaign.PRIMARY_ARCHITECTURE,
               target=campaign.CONTROL_TARGET, source_bytes=90_000,
               artifact_sha256=digest("rank4-control"),
               deployment_eligible=False),
    ]

    retained = ["primary-search", "primary-teacher", "neutral-search"]
    tuple_rows = []
    for model_ordinal, model_id in enumerate(retained):
        for tuple_ordinal, value in enumerate(campaign.TUPLE_ROSTER):
            candidate_id = f"{model_id}:{campaign.tuple_id(value)}"
            wins = 120 - model_ordinal * 10 - tuple_ordinal
            if model_id == "primary-search" and value == ("0.80", "0.5", "1"):
                wins = 160
            if model_id == "primary-search" and value == campaign.DEFAULT_TUPLE:
                wins = 159
            tuple_rows.append(metric(
                candidate_id, 100, wins, wins // 2, wins - wins // 2,
                latency=10.0 + tuple_ordinal / 10,
                model_id=model_id, tuple=list(value),
            ))

    nondefault_id = "primary-search:" + campaign.tuple_id(("0.80", "0.5", "1"))
    default_tuple_id = "primary-search:" + campaign.tuple_id(campaign.DEFAULT_TUPLE)
    tuple_confirmation = [
        metric(nondefault_id, 250, 302, 151, 151, 9.0,
               model_id="primary-search", tuple=["0.80", "0.5", "1"],
               paired_bootstrap_lower_95=0.01),
        metric(default_tuple_id, 250, 300, 150, 150, 10.0,
               model_id="primary-search", tuple=list(campaign.DEFAULT_TUPLE),
               paired_bootstrap_lower_95=0.0),
    ]

    profile_screen = [
        metric("light", 100, 120, 60, 60, 8.0, profile="light",
               work=campaign.PROFILE_ROSTER["light"]),
        metric("default", 100, 159, 79, 80, 10.0, profile="default",
               work=campaign.PROFILE_ROSTER["default"]),
        metric("heavy", 100, 160, 80, 80, 11.0, profile="heavy",
               work=campaign.PROFILE_ROSTER["heavy"]),
    ]
    profile_confirmation = [
        metric("heavy", 250, 302, 151, 151, 11.0, profile="heavy",
               work=campaign.PROFILE_ROSTER["heavy"],
               paired_bootstrap_lower_95=0.02),
        metric("default", 250, 300, 150, 150, 10.0, profile="default",
               work=campaign.PROFILE_ROSTER["default"],
               paired_bootstrap_lower_95=0.0),
    ]
    actual_id = nondefault_id + ":heavy"
    actual = metric(actual_id, 200, 211, 104, 107, 155.0)
    return {
        "schema": campaign.DEVELOPMENT_INPUT_SCHEMA,
        "namespace": campaign.NAMESPACE,
        "eligible_architectures": [
            campaign.PRIMARY_ARCHITECTURE,
            campaign.SOURCE_NEUTRAL_ARCHITECTURE,
        ],
        "banks": banks,
        "model_screen": model_rows,
        "tuple_screen": tuple_rows,
        "tuple_confirmation": tuple_confirmation,
        "profile_screen": profile_screen,
        "profile_confirmation": profile_confirmation,
        "actual_clock": actual,
    }


def post_iteration_development_payload():
    payload = development_payload()
    post_id = campaign.POST_ITERATION_CANDIDATE_ID
    primary = copy.deepcopy(payload["model_screen"][0])
    primary["candidate_id"] = post_id
    control = copy.deepcopy(payload["model_screen"][-1])
    payload["model_screen"] = [primary, control]
    tuple_rows = []
    for row in payload["tuple_screen"]:
        if row["model_id"] != "primary-search":
            continue
        updated = copy.deepcopy(row)
        updated["model_id"] = post_id
        updated["candidate_id"] = updated["candidate_id"].replace(
            "primary-search:", f"{post_id}:", 1
        )
        tuple_rows.append(updated)
    payload["tuple_screen"] = tuple_rows
    confirmations = []
    for row in payload["tuple_confirmation"]:
        updated = copy.deepcopy(row)
        updated["model_id"] = post_id
        updated["candidate_id"] = updated["candidate_id"].replace(
            "primary-search:", f"{post_id}:", 1
        )
        confirmations.append(updated)
    payload["tuple_confirmation"] = confirmations
    payload["actual_clock"]["candidate_id"] = payload[
        "actual_clock"
    ]["candidate_id"].replace("primary-search:", f"{post_id}:", 1)
    payload["development_mode"] = campaign.POST_ITERATION_DEVELOPMENT_MODE
    payload["eligible_architectures"] = [campaign.PRIMARY_ARCHITECTURE]
    payload["eligible_model_arms"] = [
        [campaign.PRIMARY_ARCHITECTURE, "search-target"]
    ]
    payload["post_iteration_handoff"] = {
        "path": "/fixture/handoff", "sha256": "1" * 64,
        "body_sha256": "2" * 64,
    }
    payload["rank4_control_selection"] = {
        "path": "/fixture/control", "sha256": "3" * 64,
        "body_sha256": "4" * 64,
    }
    return payload


class SourceFixture:
    def __init__(self, root):
        self.root = root
        self.candidate = root / "candidate.cpp"
        self.candidate.write_text("int main(){return 0;}\n", encoding="ascii")
        self.binding = root / "source-binding.json"
        base.create_source_binding(
            self.binding,
            candidate_source=self.candidate,
            candidate_commit=COMMIT,
            rank4_source=RANK4,
            opponent_source=RANK4,
        )


class DevelopmentSelectionTest(unittest.TestCase):
    def test_opening_transcript_is_complete_turn_slash_form_with_minimum_12_plies(self):
        example = "5/2/2/0/1/4/1/17/6/0/75"
        self.assertEqual(campaign.validate_complete_turn_transcript(example), example)
        self.assertEqual(campaign.transcript_primitive_plies(example), 13)
        for invalid in ("5220141176075", "5//2/01234567", "5/2/2"):
            with self.subTest(invalid=invalid), self.assertRaises(campaign.CampaignError):
                campaign.validate_complete_turn_transcript(invalid)

    def test_bank_binds_exact_overshoot_ply_count(self):
        payload = development_payload()
        payload["banks"]["model_screen"]["transcripts"][0] = (
            "5/2/2/0/1/4/1/17/6/0/75"
        )
        payload["banks"]["model_screen"]["primitive_ply_counts"][0] = 13
        campaign.validate_development_input(payload)
        payload["banks"]["model_screen"]["primitive_ply_counts"][0] = 12
        with self.assertRaisesRegex(campaign.CampaignError, "count is stale"):
            campaign.validate_development_input(payload)

    def test_full_selection_uses_exact_top3_tuple_profile_and_actual_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "selection.json"
            result = campaign.select_development(output, development_payload())
            self.assertEqual(
                result["retained_model_ids"],
                ["primary-search", "primary-teacher", "neutral-search"],
            )
            self.assertEqual(result["model"]["candidate_id"], "primary-search")
            self.assertEqual(result["tuple"], ["0.80", "0.5", "1"])
            self.assertEqual(result["profile"], "heavy")
            self.assertTrue(result["selection_immutable"])
            self.assertFalse(result["protected_tests_opened"])
            self.assertFalse(result["post_selection_test_results_may_change_selection"])
            self.assertEqual(base.load_sealed(output, campaign.SELECTION_SCHEMA), result)

    def test_model_ties_use_wins_weaker_color_latency_then_narrower(self):
        payload = development_payload()
        for row in payload["model_screen"]:
            if row["target"] != campaign.CONTROL_TARGET:
                row.update(wins=120, color_wins={"0": 60, "1": 60}, latency_ms=10.0)
        top, _ = campaign._validate_model_screen(
            payload["model_screen"], payload["eligible_architectures"]
        )
        self.assertEqual(
            [row["candidate_id"] for row in top],
            ["primary-search", "primary-teacher", "neutral-search"],
        )

    def test_tuple_roster_must_be_exact(self):
        payload = development_payload()
        payload["tuple_screen"].pop()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            campaign.CampaignError, "exact tuple roster"
        ):
            campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )

    def test_nondefault_tuple_requires_positive_bootstrap_and_no_regression(self):
        payload = development_payload()
        payload["tuple_confirmation"][0]["paired_bootstrap_lower_95"] = 0.0
        payload["actual_clock"]["candidate_id"] = (
            "primary-search:" + campaign.tuple_id(campaign.DEFAULT_TUPLE) + ":heavy"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )
            self.assertEqual(result["tuple"], list(campaign.DEFAULT_TUPLE))

        payload = development_payload()
        payload["tuple_confirmation"][0]["color_wins"]["0"] = 149
        payload["actual_clock"]["candidate_id"] = (
            "primary-search:" + campaign.tuple_id(campaign.DEFAULT_TUPLE) + ":heavy"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )
            self.assertEqual(result["tuple"], list(campaign.DEFAULT_TUPLE))

    def test_profile_uses_the_same_replacement_rule(self):
        payload = development_payload()
        payload["profile_confirmation"][0]["failures"] = 1
        nondefault = "primary-search:" + campaign.tuple_id(("0.80", "0.5", "1"))
        payload["actual_clock"]["candidate_id"] = nondefault + ":default"
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )
            self.assertEqual(result["profile"], "default")

    def test_actual_clock_exact_211_and_104_boundary(self):
        for field, value in (("wins", 210), ("color0", 103), ("failures", 1)):
            payload = development_payload()
            if field == "color0":
                payload["actual_clock"]["color_wins"]["0"] = value
            else:
                payload["actual_clock"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(campaign.CampaignError, "actual-clock gate failed"):
                    campaign.select_development(
                        pathlib.Path(temporary) / "selection.json", payload
                    )

    def test_every_development_bank_is_disjoint(self):
        payload = development_payload()
        payload["banks"]["actual_clock"]["fingerprints"][0] = payload[
            "banks"
        ]["tuple_confirmation"]["fingerprints"][0]
        payload["banks"]["actual_clock"]["fingerprints"].sort()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            campaign.CampaignError, "not mutually disjoint"
        ):
            campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )

    def test_selection_output_is_write_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "selection.json"
            campaign.select_development(output, development_payload())
            changed = development_payload()
            changed["actual_clock"]["latency_ms"] = 154.0
            with self.assertRaisesRegex(campaign.CampaignError, "immutable artifact collision"):
                campaign.select_development(output, changed)

    def test_post_iteration_mode_retains_one_then_runs_unchanged_adaptive_path(self):
        payload = post_iteration_development_payload()
        candidate_row = payload["model_screen"][0]
        control_row = payload["model_screen"][1]
        handoff_details = {
            "handoff": {
                "plan": {"path": "plan"},
                "iteration_completion": {"path": "completion"},
                "iteration_selection": {"path": "selection"},
            },
            "candidate": {
                "candidate_id": campaign.POST_ITERATION_CANDIDATE_ID,
                "architecture": campaign.PRIMARY_ARCHITECTURE,
                "target": "search-target",
                "runtime": {"sha256": candidate_row["artifact_sha256"]},
                "generated_source": {"bytes": candidate_row["source_bytes"]},
            },
        }
        control_details = {
            "runtime": {"sha256": control_row["artifact_sha256"]},
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(campaign, "validate_development_input"),
            mock.patch.object(
                campaign, "validate_post_iteration_handoff",
                return_value=handoff_details,
            ),
            mock.patch.object(
                campaign, "validate_rank4_control_reference",
                return_value=control_details,
            ),
            mock.patch.object(
                campaign, "_validate_post_iteration_development_evidence"
            ),
        ):
            result = campaign.select_development(
                pathlib.Path(temporary) / "selection.json", payload
            )
        self.assertEqual(result["development_mode"], "post-iteration")
        self.assertEqual(
            result["retained_model_ids"],
            [campaign.POST_ITERATION_CANDIDATE_ID],
        )
        self.assertEqual(len(payload["tuple_screen"]), 8)
        self.assertEqual(
            result["status"], "immutable-development-selected-not-tests-opened"
        )
        self.assertFalse(result["protected_tests_opened"])

    def test_post_iteration_mode_cannot_bypass_handoff(self):
        payload = post_iteration_development_payload()
        payload.pop("post_iteration_handoff")
        with self.assertRaisesRegex(campaign.CampaignError, "handoff reference"):
            campaign.validate_development_input(payload)

    def test_post_iteration_handoff_rejects_fake_iteration_selection_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "run"
            iteration_root = root / "ledger"
            authorization_path = iteration_root / "iteration/00-authorization.json"
            authorization = base.write_sealed(authorization_path, {
                "schema": campaign.ITERATION_AUTH_SCHEMA,
                "namespace": campaign.NAMESPACE,
            })
            plan_path = output / "iteration-plan.json"
            plan_schema = "papersoccer.fixture-iteration-plan.v1"
            plan = base.write_sealed(plan_path, {
                "schema": plan_schema, "namespace": campaign.NAMESPACE,
                "authorization": campaign._iteration_artifact_reference(
                    authorization_path, campaign.ITERATION_AUTH_SCHEMA
                ),
            })
            completion_path = iteration_root / "iteration/02-completed.json"
            completion = base.write_sealed(completion_path, {
                "schema": campaign.ITERATION_EVENT_SCHEMA,
                "namespace": campaign.NAMESPACE, "result": {},
            })
            fake_selection_path = output / "fine-tune/selections/fake.json"
            fake_selection = base.write_sealed(fake_selection_path, {
                "schema": "papersoccer.fake-iteration-selection.v1",
                "namespace": campaign.NAMESPACE,
            })
            handoff_path = output / "post-iteration-development-handoff.json"
            base.write_sealed(handoff_path, {
                "schema": campaign.POST_ITERATION_HANDOFF_SCHEMA,
                "namespace": campaign.NAMESPACE,
                "status": campaign.POST_ITERATION_HANDOFF_STATUS,
                "plan": campaign._iteration_artifact_reference(
                    plan_path, plan_schema
                ),
                "iteration_completion": campaign._iteration_artifact_reference(
                    completion_path, campaign.ITERATION_EVENT_SCHEMA
                ),
                "iteration_selection": {
                    "path": str(fake_selection_path.resolve()),
                    "sha256": base.sha256_file(fake_selection_path),
                    "body_sha256": fake_selection["body_sha256"],
                },
                "candidate": {}, "source_export": {}, "offline_gate": {},
                "candidate_artifacts_immutable": True,
                "development_screen_required": True,
                "development_selected": False,
                "protected_tests_opened": False,
                "protected_tests_authorized": False,
                "upload_authorized": False, "iterations_remaining": 0,
            })
            fake_iteration = types.SimpleNamespace(
                PLAN_SCHEMA=plan_schema, CAMPAIGN_ID="fixture",
                validate_plan_contract=lambda *args, **kwargs: None,
            )
            with (
                mock.patch.object(
                    campaign, "_iteration_module", return_value=fake_iteration
                ),
                mock.patch.object(
                    campaign, "validate_iteration_authorization",
                    return_value=authorization,
                ),
                mock.patch.object(
                    campaign, "_validate_completed_iteration",
                    return_value=completion,
                ),
                self.assertRaisesRegex(
                    campaign.CampaignError,
                    "unexpected artifact schema|post-iteration model selection",
                ),
            ):
                campaign.validate_post_iteration_handoff(handoff_path)


class ProtectedTestAuthorizationTest(unittest.TestCase):
    def test_authorization_binds_immutable_selected_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            payload = development_payload()
            artifact = root / "model.runtime"
            artifact.write_bytes(b"selected model")
            selected_id = "primary-search"
            for row in payload["model_screen"]:
                if row["candidate_id"] == selected_id:
                    row["artifact_sha256"] = base.sha256_file(artifact)
            selection = root / "selection.json"
            campaign.select_development(selection, payload)
            authorization = campaign.authorize_protected_tests(
                root / "test-authorization.json",
                selection_path=selection,
                quantized_artifact_path=artifact,
                authorized_at_utc="2026-08-31T15:00:00Z",
            )
            self.assertEqual(authorization["status"], "protected-tests-authorized-once")
            self.assertFalse(authorization["selection_may_change"])

    def test_artifact_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            selection = root / "selection.json"
            campaign.select_development(selection, development_payload())
            artifact = root / "wrong.runtime"
            artifact.write_bytes(b"wrong")
            with self.assertRaisesRegex(campaign.CampaignError, "differs"):
                campaign.authorize_protected_tests(
                    root / "auth.json", selection_path=selection,
                    quantized_artifact_path=artifact,
                    authorized_at_utc="2026-08-31T15:00:00Z",
                )


class FrozenFamilyFailureAndSafeActorTest(unittest.TestCase):
    def test_failure_record_fully_binds_exact_rejected_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = FrozenFamilyFixture(root)
            record_path, record = fixture.record_failure(root / "governance")
            self.assertEqual(record["deployable_arms_rejected"], 6)
            self.assertEqual(len(record["rejected_deployable_arms"]), 6)
            self.assertFalse(record["protected_tests_opened"])
            self.assertTrue(record["source_size_eligible"])
            self.assertEqual(
                record_path.name,
                base.sha256_file(record_path) + ".offline-family-failure.json",
            )
            with fixture.workflow_patch():
                self.assertEqual(
                    campaign.validate_offline_family_failure(record_path), record
                )

    def test_failure_record_rejects_missing_campaign_protected_access_and_oversize(self):
        mutations = (
            lambda receipt: receipt["campaigns"].pop(),
            lambda receipt: receipt["campaigns"][0].__setitem__(
                "protected_tests_opened", True
            ),
            lambda receipt: receipt["source_size_preflight"][
                "capacity-12x8"
            ].__setitem__("complete_source_ascii_characters", 95_001),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                fixture = FrozenFamilyFixture(root)
                body = copy.deepcopy(fixture.receipt)
                body.pop("body_sha256")
                mutate(body)
                fixture.receipt = base.seal(body)
                fixture.receipt_path.unlink()
                base.atomic_write_once(
                    fixture.receipt_path,
                    base.canonical_json_bytes(fixture.receipt),
                )
                with fixture.workflow_patch(), self.assertRaises(
                    campaign.CampaignError
                ):
                    campaign.record_offline_family_failure(
                        root / "governance", bundle_manifest=fixture.bundle,
                        run_output_directory=fixture.run,
                        run_reference=fixture.reference_path,
                        recorded_at_utc="2026-08-31T15:30:00Z",
                    )

    def test_safe_actor_binds_exact_float_runtime_source_build_protocol_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = FrozenFamilyFixture(root)
            failure_path, _failure = fixture.record_failure(root / "governance")
            build, protocol, timing = fixture.operational_evidence()
            trainer = fixture.trainer
            with (
                fixture.workflow_patch(),
                mock.patch.object(
                    trainer.FrozenBundle, "load",
                    return_value=types.SimpleNamespace(body_sha256="f" * 64),
                ),
                mock.patch.object(
                    trainer, "validate_selection",
                    return_value=fixture.selected_selection_value,
                ),
            ):
                actor = campaign.record_operational_safe_actor(
                    root / "governance",
                    failure_record_path=failure_path,
                    family_selection_path=fixture.selected_selection,
                    float_checkpoint_path=fixture.float_checkpoint,
                    runtime_path=fixture.selected_runtime,
                    generated_source_path=fixture.generated_source,
                    protocol_evidence_path=protocol,
                    build_evidence_path=build,
                    timing_evidence_path=timing,
                    recorded_at_utc="2026-08-31T15:45:00Z",
                )
                actor_path = campaign.operational_safe_actor_path(
                    root / "governance", actor
                )
                self.assertEqual(
                    campaign.validate_operational_safe_actor(actor_path), actor
                )
            self.assertTrue(actor["operationally_safe"])
            self.assertEqual(actor["architecture"], campaign.CAPACITY_ARCHITECTURE)
            self.assertEqual(actor["arm"], "search-target")

    def test_safe_actor_rejects_wrong_checkpoint_failing_timing_and_binary_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = FrozenFamilyFixture(root)
            failure_path, _failure = fixture.record_failure(root / "governance")
            build, protocol, timing = fixture.operational_evidence()
            wrong_float = root / "wrong.float.npz"
            wrong_float.write_bytes(fixture.float_checkpoint.read_bytes())
            trainer = fixture.trainer
            patches = (
                fixture.workflow_patch(),
                mock.patch.object(
                    trainer.FrozenBundle, "load",
                    return_value=types.SimpleNamespace(body_sha256="f" * 64),
                ),
                mock.patch.object(
                    trainer, "validate_selection",
                    return_value=fixture.selected_selection_value,
                ),
            )
            with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                campaign.CampaignError, "exact selected checkpoint"
            ):
                campaign.record_operational_safe_actor(
                    root / "actors", failure_record_path=failure_path,
                    family_selection_path=fixture.selected_selection,
                    float_checkpoint_path=wrong_float,
                    runtime_path=fixture.selected_runtime,
                    generated_source_path=fixture.generated_source,
                    protocol_evidence_path=protocol,
                    build_evidence_path=build,
                    timing_evidence_path=timing,
                    recorded_at_utc="2026-08-31T15:45:00Z",
                )

            timing_value = base.load_sealed(
                timing, campaign._preflight_module().TIMING_SCHEMA
            )
            timing_value.pop("body_sha256")
            timing_value["samples"][0]["first_ms"] = 900.0
            bad_timing = root / "bad-timing.json"
            base.write_sealed(bad_timing, timing_value)
            with (
                fixture.workflow_patch(),
                mock.patch.object(
                    trainer.FrozenBundle, "load",
                    return_value=types.SimpleNamespace(body_sha256="f" * 64),
                ),
                mock.patch.object(
                    trainer, "validate_selection",
                    return_value=fixture.selected_selection_value,
                ),
                self.assertRaisesRegex(campaign.CampaignError, "timing evidence"),
            ):
                campaign.record_operational_safe_actor(
                    root / "actors", failure_record_path=failure_path,
                    family_selection_path=fixture.selected_selection,
                    float_checkpoint_path=fixture.float_checkpoint,
                    runtime_path=fixture.selected_runtime,
                    generated_source_path=fixture.generated_source,
                    protocol_evidence_path=protocol,
                    build_evidence_path=build,
                    timing_evidence_path=bad_timing,
                    recorded_at_utc="2026-08-31T15:45:00Z",
                )


class OneIterationLedgerTest(unittest.TestCase):
    def setUp(self):
        failure = mock.patch.object(
            campaign, "validate_offline_family_failure",
            side_effect=lambda path: base.load_sealed(
                path, campaign.OFFLINE_FAMILY_FAILURE_SCHEMA
            ),
        )
        actor = mock.patch.object(
            campaign, "validate_operational_safe_actor",
            side_effect=lambda path: base.load_sealed(
                path, campaign.OPERATIONAL_SAFE_ACTOR_SCHEMA
            ),
        )
        failure.start()
        actor.start()
        self.addCleanup(failure.stop)
        self.addCleanup(actor.stop)

    def authorize(self, root, learning_rate=0.00006):
        failure_path, _failure = campaign._write_content_addressed_sealed(
            root / "governance", {
                "schema": campaign.OFFLINE_FAMILY_FAILURE_SCHEMA,
                "namespace": campaign.NAMESPACE,
                "iteration_authorizable": True,
            }, ".offline-family-failure.json",
        )
        failure_reference = campaign._sealed_file_reference(
            failure_path, campaign.OFFLINE_FAMILY_FAILURE_SCHEMA,
            label="fixture failure",
        )
        actor_path, _actor = campaign._write_content_addressed_sealed(
            root / "governance", {
                "schema": campaign.OPERATIONAL_SAFE_ACTOR_SCHEMA,
                "namespace": campaign.NAMESPACE,
                "offline_family_failure": failure_reference,
            }, ".operational-safe-actor.json",
        )
        return campaign.authorize_iteration(
            root,
            failure_record_path=failure_path,
            safe_actor_record_path=actor_path,
            learning_rate=learning_rate,
            authorized_at_utc="2026-08-31T16:00:00Z",
        )

    def environment(self):
        return {
            "interactive_launch_agent": True,
            "resume": True,
            "blas_threads": 1,
            "ac_power": True,
            "free_disk_gib": 20.0,
        }

    def completion(self, learning_rate=0.00006):
        return {
            "games": copy.deepcopy(campaign.ITERATION_SPEC["games"]),
            "total_games": 10_000,
            "positions_per_game": 20,
            "workers": 10,
            "fixed_work": True,
            "deep_relabel_fraction": 0.25,
            "resumed": True,
            "learning_rate": learning_rate,
            "float_checkpoint_sha256": digest("fine tuned"),
        }

    def test_exact_spec_and_environment_complete_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            authorization = self.authorize(root)
            self.assertEqual(authorization["specification"]["total_games"], 10_000)
            self.assertEqual(sum(authorization["specification"]["games"].values()), 10_000)
            campaign.start_iteration(
                root, environment=self.environment(),
                started_at_utc="2026-08-31T16:01:00Z",
            )
            completed = campaign.complete_iteration(
                root, result=self.completion(),
                completed_at_utc="2026-08-31T17:00:00Z",
            )
            self.assertEqual(completed["iterations_remaining"], 0)
            with self.assertRaisesRegex(campaign.CampaignError, "already authorized"):
                self.authorize(root)
            with self.assertRaisesRegex(campaign.CampaignError, "already completed"):
                campaign.complete_iteration(
                    root, result=self.completion(),
                    completed_at_utc="2026-08-31T17:01:00Z",
                )

    def test_lr_launchagent_ac_and_disk_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(campaign.CampaignError, "6e-5"):
                self.authorize(pathlib.Path(temporary), learning_rate=0.000061)
        for field, value in (
            ("interactive_launch_agent", False), ("resume", False),
            ("blas_threads", 2), ("ac_power", False), ("free_disk_gib", 19.99),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                self.authorize(root)
                environment = self.environment()
                environment[field] = value
                with self.assertRaises(campaign.CampaignError):
                    campaign.start_iteration(
                        root, environment=environment,
                        started_at_utc="2026-08-31T16:01:00Z",
                    )

    def test_family_exhausted_accepts_one_sealed_failure_at_each_terminal_stage(self):
        for stage in sorted(campaign.POST_ITERATION_FAILURE_STAGES):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                self.authorize(root)
                campaign.start_iteration(
                    root, environment=self.environment(),
                    started_at_utc="2026-08-31T16:01:00Z",
                )
                campaign.complete_iteration(
                    root, result=self.completion(),
                    completed_at_utc="2026-08-31T17:00:00Z",
                )
                evidence = root / "post-iteration-evidence.json"
                base.write_sealed(evidence, {
                    "schema": "papersoccer.fixture-failure.v1", "failed": True,
                })
                failure = campaign.record_post_iteration_failure(
                    root, stage=stage, evidence_path=evidence,
                    recorded_at_utc="2026-08-31T17:30:00Z",
                )
                failure_path = root / "iteration/03-post-iteration-failure.json"
                self.assertEqual(failure["stage"], stage)
                with self.assertRaisesRegex(campaign.CampaignError, "already recorded"):
                    campaign.record_post_iteration_failure(
                        root, stage=stage, evidence_path=evidence,
                        recorded_at_utc="2026-08-31T17:31:00Z",
                    )
                exhausted = campaign.record_family_exhausted(
                    root, post_iteration_failure_path=failure_path,
                    recorded_at_utc="2026-08-31T18:00:00Z",
                )
                self.assertFalse(exhausted["upload_authorized"])
                self.assertFalse(exhausted["goal_complete"])
                self.assertEqual(exhausted["iterations_remaining"], 0)
                self.assertEqual(exhausted["failure_stage"], stage)


class PreuploadExclusionTest(unittest.TestCase):
    def test_registry_contains_only_sorted_ids_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = SourceFixture(root)
            first = root / "first.ids"
            second = root / "second.ids"
            first.write_text("1\n2\n", encoding="ascii")
            second.write_text("2\n3\n", encoding="ascii")
            registry = campaign.freeze_preupload_exclusions(
                root / "exclusions.json",
                source_binding_path=fixture.binding,
                id_files=[first, second],
                frozen_at_utc="2026-08-31T19:00:00Z",
            )
            self.assertEqual(registry["game_ids"], [1, 2, 3])
            self.assertTrue(registry["contains_only_game_ids"])
            self.assertFalse(registry["replay_payloads_accessed"])
            self.assertNotIn("path", registry["sources"][0])

    def test_replay_or_unsorted_content_is_rejected(self):
        for content in ('{"frames":[]}\n', "2\n1\n", "1\n1\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                fixture = SourceFixture(root)
                ids = root / "bad.ids"
                ids.write_text(content, encoding="ascii")
                with self.assertRaises(campaign.CampaignError):
                    campaign.freeze_preupload_exclusions(
                        root / "exclusions.json",
                        source_binding_path=fixture.binding,
                        id_files=[ids], frozen_at_utc="2026-08-31T19:00:00Z",
                    )


if __name__ == "__main__":
    unittest.main()
