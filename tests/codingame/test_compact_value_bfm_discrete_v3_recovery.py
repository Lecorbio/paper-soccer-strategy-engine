import copy
import pathlib
import tempfile
import unittest
from unittest import mock


from tools import compact_value_bfm_discrete_v3_recovery as recovery


CAMPAIGN_ROOT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "results/compact_value_bfm/compact-value-bfm-discrete-v3-20260901-v1"
)


class RecoveryGovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_plan_path = CAMPAIGN_ROOT / "development-v3/plan.json"
        if not original_plan_path.is_file():
            raise unittest.SkipTest(
                "discrete-v3 recovery forensics require the ignored local campaign"
            )
        original_plan = recovery.qualification.load_sealed(
            original_plan_path, recovery.development.PLAN_SCHEMA
        )
        with mock.patch.object(
            recovery.development, "validate_plan", return_value=original_plan
        ):
            cls.original = recovery.validate_original_state(CAMPAIGN_ROOT)
        protected_path = pathlib.Path(
            original_plan["exclusion"]["protected_fingerprints"]["path"]
        )
        protected_payload = recovery.qualification.load_sealed(
            protected_path, recovery.development.exclusions.FINGERPRINT_SCHEMA
        )
        cls.protected = frozenset(
            row["canonical_sha256"] for row in protected_payload["rows"]
        )
        protected_record = recovery._sealed_record(
            protected_path, recovery.development.exclusions.FINGERPRINT_SCHEMA
        )
        recovery._PRIVATE_FINGERPRINT_CACHE = (
            (
                protected_record["sha256"],
                original_plan["exclusion"]["plan"]["sha256"],
                original_plan["exclusion"]["receipt"]["sha256"],
            ),
            cls.protected,
        )
        cls.exclusions = recovery._exclusion_context(
            CAMPAIGN_ROOT, cls.original["plan"]
        )

    def test_original_terminal_attempt_and_ten_carried_receipts_are_exact(self):
        self.assertEqual(
            recovery.qualification.sha256_file(self.original["plan_path"]),
            recovery.ORIGINAL_PLAN_SHA256,
        )
        self.assertEqual(len(self.original["carried"]), 10)
        self.assertEqual(
            [item["stage"] for item in self.original["carried"]],
            ["model_screen"] * 2 + ["tuple_screen"] * 8,
        )
        self.assertEqual(
            self.original["journal"]["head_sha256"],
            recovery.JOURNAL_HEAD_SHA256,
        )
        self.assertTrue(self.original["journal"]["terminal"]["no_retry"])

    def test_completed_confirmation_is_forensic_only_and_others_are_absent(self):
        forensic = self.original["forensic"]
        self.assertEqual(forensic["selection_weight"], 0)
        self.assertFalse(forensic["eligible_for_selection"])
        self.assertEqual(forensic["raw_result"]["sha256"], recovery.FORENSIC_RESULT_SHA256)
        self.assertEqual(forensic["summary"]["games"], 500)
        self.assertEqual(forensic["summary"]["candidate_wins"], 258)
        self.assertEqual(forensic["summary"]["failures"], 0)
        self.assertEqual(forensic["summary"]["unfinished"], 0)
        self.assertEqual(
            forensic["root_cause"]["classification"],
            "historical-supervisor-result-schema-defect",
        )
        self.assertAlmostEqual(
            forensic["root_cause"]["terminal_delay_seconds"], 0.541546158
        )
        self.assertTrue(forensic["root_cause"]["failed_before_quiet_interval"])
        self.assertEqual(
            {item["candidate_id"] for item in forensic["anchor_and_default_absent"]},
            {
                "discrete-v3-search-target:c0.65-f0.5-l1",
                "discrete-v3-search-target:c0.95-f0.5-l1",
            },
        )
        self.assertEqual(
            self.original["process_absence"]["forbidden_match_count"], 0
        )

    def test_fresh_confirmation_payload_is_deterministic_and_fully_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = pathlib.Path(temporary) / "plan.json"
            recovery.qualification.write_sealed(
                plan_path, {"schema": recovery.PLAN_SCHEMA, "test": True}
            )
            first, first_rows = recovery._fresh_bank_payload(
                original_plan=self.original["plan"],
                context=self.exclusions,
                recovery_plan_path=plan_path,
            )
            second, second_rows = recovery._fresh_bank_payload(
                original_plan=self.original["plan"],
                context=self.exclusions,
                recovery_plan_path=plan_path,
            )
        self.assertEqual(first, second)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(len(first_rows), 250)
        fresh = recovery.qualification.seal(first)
        self.assertFalse(
            recovery._all_variants(fresh) & self.exclusions["fingerprints"]
        )
        self.assertFalse(recovery._all_variants(fresh) & self.protected)
        self.assertEqual(len(recovery._all_variants(fresh)), 1_000)

    def test_materializer_creates_exactly_one_manifest_and_gate_derivative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = root / "plan.json"
            recovery.qualification.write_sealed(
                plan_path, {"schema": recovery.PLAN_SCHEMA, "test": True}
            )
            fresh = recovery.create_fresh_confirmation_bank(
                CAMPAIGN_ROOT,
                self.original["plan"],
                plan_path,
                root / "opening-banks",
                root / "gate-banks",
            )
            self.assertEqual(
                len(list((root / "opening-banks").glob("*.opening-bank.json"))), 1
            )
            self.assertEqual(len(list((root / "gate-banks").glob("*.tsv"))), 1)
            self.assertEqual(fresh["document"]["opening_count"], 250)
            self.assertEqual(
                pathlib.Path(fresh["gate"]["path"]).read_bytes(),
                recovery._gate_bank_bytes(fresh["document"]),
            )

    def test_selected_six_replace_only_confirmation_and_spent_bank_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_path = root / "plan.json"
            recovery.qualification.write_sealed(
                plan_path, {"schema": recovery.PLAN_SCHEMA, "test": True}
            )
            fresh = recovery.create_fresh_confirmation_bank(
                CAMPAIGN_ROOT,
                self.original["plan"],
                plan_path,
                root / "opening-banks",
                root / "gate-banks",
            )
            banks, rows, variants = recovery._mixed_bank_context(
                self.original["plan"], fresh
            )
        self.assertEqual(tuple(banks), recovery.STAGE_ORDER)
        self.assertNotEqual(
            banks["tuple_confirmation"],
            self.original["plan"]["banks"]["tuple_confirmation"],
        )
        for stage in recovery.STAGE_ORDER:
            if stage != "tuple_confirmation":
                self.assertEqual(banks[stage], self.original["plan"]["banks"][stage])
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(variants), 4 * sum(recovery.STAGE_PAIRS.values()))
        additional = recovery._additional_exclusions(self.original["plan"])
        spent = additional["spent_original_tuple_confirmation"]
        self.assertEqual(
            spent["bank"], self.original["plan"]["banks"]["tuple_confirmation"]
        )
        self.assertEqual(spent["selection_weight"], 0)
        self.assertFalse(spent["eligible_for_selection"])
        self.assertTrue(spent["required_for_eventual_protected_final"])

    def test_preregistered_descriptor_binds_seed_exclusions_and_absent_state(self):
        outputs = copy.deepcopy(recovery._paths(CAMPAIGN_ROOT))
        descriptor = recovery._fresh_descriptor(
            outputs=outputs, exclusion_context=self.exclusions
        )
        self.assertEqual(descriptor["opening_count"], 250)
        self.assertEqual(descriptor["seed_hex"], recovery.RECOVERY_BANK_SEED.hex())
        self.assertEqual(descriptor["domain"], recovery.RECOVERY_BANK_DOMAIN)
        self.assertEqual(descriptor["excluded_source_count"], 14)
        self.assertEqual(
            descriptor["protected_fingerprint_source"]["fingerprint_count"],
            54_611,
        )
        self.assertFalse(descriptor["materialized_at_plan_creation"])
        contract = recovery._recovery_contract(descriptor)
        self.assertEqual(contract["tuple_confirmation_bank"], descriptor)
        self.assertTrue(contract["one_shot_no_replay"])
        self.assertEqual(
            contract["gate_configuration_invariants"]["candidate_clocks_ms"],
            [800, 155],
        )

    def test_prepare_preregisters_before_any_bank_materialization(self):
        runner = (
            pathlib.Path(__file__).resolve().parents[2]
            / "submissions/codingame/bots/compact_value_bfm/discrete_v3_recovery_runner.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_root = pathlib.Path(temporary).resolve()
            with (
                mock.patch.object(
                    recovery, "validate_original_state", return_value=self.original
                ),
                mock.patch.object(
                    recovery, "_exclusion_context", return_value=self.exclusions
                ),
                mock.patch.object(
                    recovery,
                    "validate_recovery_plan",
                    return_value={"plan": {}, "materialized": False},
                ),
            ):
                plan_path = recovery.prepare_recovery(
                    output_root=output_root,
                    recovery_runner=runner,
                    created_at_utc="2026-09-03T07:00:00Z",
                )
            with (
                mock.patch.object(
                    recovery, "validate_original_state", return_value=self.original
                ),
                mock.patch.object(
                    recovery, "_exclusion_context", return_value=self.exclusions
                ),
            ):
                validated = recovery.validate_recovery_plan(
                    plan_path, output_root=output_root
                )
            plan = recovery.qualification.load_sealed(
                plan_path, recovery.PLAN_SCHEMA
            )
            self.assertFalse(validated["materialized"])
            routes = plan["outputs"]
            self.assertTrue(pathlib.Path(routes["incident"]).is_file())
            self.assertTrue(plan_path.is_file())
            self.assertFalse(pathlib.Path(routes["opening_banks"]).exists())
            self.assertFalse(pathlib.Path(routes["gate_banks"]).exists())
            self.assertFalse(pathlib.Path(routes["mixed_six_exclusion"]).exists())
            self.assertFalse(
                plan["banks"]["tuple_confirmation"][
                    "materialized_at_plan_creation"
                ]
            )
            with (
                mock.patch.object(
                    recovery, "validate_original_state", return_value=self.original
                ),
                mock.patch.object(
                    recovery, "_exclusion_context", return_value=self.exclusions
                ),
            ):
                materialized = recovery.materialize_recovery_bank(
                    plan_path=plan_path,
                    output_root=output_root,
                    materialized_at_utc="2026-09-03T07:01:00Z",
                )
            self.assertTrue(materialized["materialized"])
            self.assertEqual(
                set(materialized["development_bank_records"]),
                set(recovery.STAGE_ORDER),
            )
            self.assertIsNotNone(materialized["materialized_mixed_six_exclusion"])

    def test_prepare_rejects_symlinked_empty_recovery_root_without_write_through(self):
        runner_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "submissions/codingame/bots/compact_value_bfm/discrete_v3_recovery_runner.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = pathlib.Path(temporary).resolve()
            output_root = temporary_root / "campaign"
            output_root.mkdir()
            redirect = temporary_root / "redirect"
            redirect.mkdir()
            recovery_root = output_root / recovery.RECOVERY_ROOT_NAME
            recovery_root.symlink_to(redirect, target_is_directory=True)
            with self.assertRaisesRegex(
                recovery.RecoveryError, "redirected or irregular"
            ):
                recovery.prepare_recovery(
                    output_root=output_root,
                    recovery_runner=runner_path,
                    created_at_utc="2026-09-03T07:00:00Z",
                )
            self.assertTrue(recovery_root.is_symlink())
            self.assertEqual(list(redirect.iterdir()), [])

    def test_policy_and_concurrency_are_fail_closed(self):
        self.assertEqual(recovery._policy()["recovery_attempts_authorized"], 1)
        self.assertFalse(recovery._policy()["original_attempt_replay_authorized"])
        self.assertFalse(recovery._policy()["final_bank_generation_authorized"])
        self.assertEqual(recovery.CONCURRENCY["tuple_confirmation"]["jobs"], 3)
        self.assertTrue(
            recovery.CONCURRENCY["tuple_confirmation"]["no_retry_after_claim"]
        )
        self.assertEqual(
            recovery.CONCURRENCY["actual_clock"]["maximum_concurrent_jobs"], 1
        )

    def test_original_plan_rejects_deep_validator_mismatch(self):
        changed = copy.deepcopy(self.original["plan"])
        changed["status"] = "changed-by-deep-validator"
        with mock.patch.object(
            recovery.development, "validate_plan", return_value=changed
        ):
            with self.assertRaisesRegex(
                recovery.RecoveryError, "deep validation changed"
            ):
                recovery._original_plan(CAMPAIGN_ROOT)

    def test_private_protected_fingerprints_are_loaded_and_bound_without_values(self):
        saved = recovery._PRIVATE_FINGERPRINT_CACHE
        recovery._PRIVATE_FINGERPRINT_CACHE = None
        try:
            with mock.patch.object(
                recovery.development.exclusions,
                "_load_private_canonical_fingerprints",
                return_value=self.protected,
            ) as loader:
                context = recovery._exclusion_context(
                    CAMPAIGN_ROOT, self.original["plan"]
                )
            loader.assert_called_once()
        finally:
            recovery._PRIVATE_FINGERPRINT_CACHE = saved
        self.assertEqual(
            context["protected_source"]["fingerprint_count"], 54_611
        )
        self.assertFalse(
            context["protected_source"]["private_values_serialized"]
        )
        self.assertNotIn("fingerprints", context["protected_source"])

    def test_original_process_preflight_rejects_live_historical_supervisor(self):
        supervisor = (
            CAMPAIGN_ROOT
            / "development-adaptive-stage-barrier/adaptive_stage_barrier_supervisor.py"
        )
        completed = mock.Mock(returncode=0, stdout=f"123 {supervisor} --mode tuple_confirmation\n")
        with mock.patch.object(recovery.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(
                recovery.RecoveryError, "original campaign process remains live"
            ):
                recovery.validate_no_original_campaign_processes(CAMPAIGN_ROOT)


if __name__ == "__main__":
    unittest.main()
