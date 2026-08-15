import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "record_rank4_jacek_hybrid_full_development_clock.py"
SPEC = importlib.util.spec_from_file_location("hybrid_full_recorder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recorder
SPEC.loader.exec_module(recorder)


def proof_fields(games, hits=1):
    return {
        "candidate_proof_root": f"{games}/{hits}/0",
        "candidate_proof_leaf": f"{2 * games}/0/{hits}",
        "candidate_proof_ply1": f"{3 * games}/{hits}/{hits}/{2 * hits}",
        "candidate_proof_ply2": "0/0/0/0",
        "candidate_proof_rebound": (
            f"{6 * games}/{2 * hits}/{2 * hits}"
        ),
        "reference_proof_root": "0/0/0",
        "reference_proof_leaf": "0/0/0",
        "reference_proof_ply1": "0/0/0/0",
        "reference_proof_ply2": "0/0/0/0",
        "reference_proof_rebound": "0/0/0",
    }


def engine_fields(engine, games, index, aggregate=False):
    if aggregate:
        invocations = 3060
        nodes = 30600
        exhaustions = 10
        nodes_max = 103
        depth_max = 7
        attempted_depth_max = 8
        first_max = 803.0
        later_max = 165.3
    else:
        invocations = games * 10
        nodes = games * 100
        exhaustions = index + 1
        nodes_max = 100 + index
        depth_max = 4 + index
        attempted_depth_max = 5 + index
        first_max = 800.0 + index
        later_max = 165.0 + index / 10.0
    return {
        f"{engine}_invocations": str(invocations),
        f"{engine}_searches": str(invocations),
        f"{engine}_illegal": "0",
        f"{engine}_operational": "0",
        f"{engine}_exceptions": "0",
        f"{engine}_hard_timeouts": "0",
        f"{engine}_soft_overruns": "0",
        f"{engine}_nodes": str(nodes),
        f"{engine}_exhaustions": str(exhaustions),
        f"{engine}_nodes_max": str(nodes_max),
        f"{engine}_depth_max": str(depth_max),
        f"{engine}_attempted_depth_max": str(attempted_depth_max),
        f"{engine}_first_ms_p99": f"{first_max - 10.0:.3f}",
        f"{engine}_first_ms_max": f"{first_max:.3f}",
        f"{engine}_later_ms_p99": f"{later_max - 5.0:.3f}",
        f"{engine}_later_ms_max": f"{later_max:.3f}",
    }


def bank_summary(index, games):
    color_games = games // 2
    candidate_per_color = 20
    fields = {
        "bank": str(index),
        "games": str(games),
        "candidate_wins": str(candidate_per_color * 2),
        "reference_wins": str(games - candidate_per_color * 2),
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": (
            f"{candidate_per_color}/{color_games - candidate_per_color}/0/0/"
            f"{color_games}"
        ),
        "candidate_p1": (
            f"{candidate_per_color}/{color_games - candidate_per_color}/0/0/"
            f"{color_games}"
        ),
    }
    fields.update(engine_fields("candidate", games, index))
    fields.update(engine_fields("reference", games, index))
    fields.update(proof_fields(games))
    return fields


def aggregate_summary():
    fields = {
        "bank": "all",
        "games": "306",
        "candidate_wins": "160",
        "reference_wins": "146",
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": "80/73/0/0/153",
        "candidate_p1": "80/73/0/0/153",
    }
    fields.update(engine_fields("candidate", 306, 0, aggregate=True))
    fields.update(engine_fields("reference", 306, 0, aggregate=True))
    fields.update(proof_fields(306, hits=4))
    return fields


def valid_summaries(candidate_mask=7):
    banks = [
        bank_summary(index, bank[4])
        for index, bank in enumerate(recorder.BANKS)
    ]
    aggregate = aggregate_summary()
    if candidate_mask == 3:
        for fields in (*banks, aggregate):
            root = [int(value) for value in fields["candidate_proof_root"].split("/")]
            leaf = [int(value) for value in fields["candidate_proof_leaf"].split("/")]
            fields["candidate_proof_ply1"] = "0/0/0/0"
            fields["candidate_proof_rebound"] = "/".join(
                str(root[index] + leaf[index]) for index in range(3)
            )
    return banks, aggregate


def valid_control_report(head="deadbeef", inputs=None, candidate_mask=7):
    if inputs is None:
        inputs = {"dependency": {"sha256": "abc"}}
    banks, aggregate = valid_summaries(candidate_mask)
    return {
        "schema": "rank4-jacek-hybrid-full-development-clock-v2",
        "campaign_id": recorder.CAMPAIGN_ID,
        "campaign_t0_utc": recorder.CAMPAIGN_T0_UTC,
        "classification": "full-development-selection-gate-not-final-qualification",
        "reference_engine": "hybrid-control",
        "candidate_exact_proof_mask": candidate_mask,
        "reference_exact_proof_mask": 0,
        "returncode": 0,
        "timed_out": False,
        "os_error": None,
        "stable_inputs": True,
        "stderr": "",
        "git": {
            "head_before": head,
            "head_after": head,
            "tracked_status_before": "",
            "tracked_status_after": "",
        },
        "inputs_before": inputs,
        "inputs_after": inputs,
        "parsed": {
            "banks": banks,
            "aggregate": aggregate,
            "configuration": recorder.configuration_expected(
                "hybrid-control", candidate_mask
            ),
            "validation_errors": [],
            "selection_threshold_errors": [],
        },
        "development_selection_acceptable": True,
        "final_qualification": False,
        "ended_utc": "2026-08-14T00:00:00+00:00",
    }


class FullDevelopmentRecorderTest(unittest.TestCase):
    def test_only_two_preregistered_reference_engines_exist(self):
        self.assertEqual(recorder.ALLOWED_REFERENCES, ("hybrid-control", "rank4"))
        self.assertEqual(recorder.ALLOWED_CANDIDATE_MASKS, (7, 3))
        for reference in recorder.ALLOWED_REFERENCES:
            for candidate_mask in recorder.ALLOWED_CANDIDATE_MASKS:
                command = recorder.command_for(reference, candidate_mask)
                self.assertEqual(command.count("--bank"), 4)
                self.assertEqual(
                    command[command.index("--reference-engine") + 1], reference
                )
                self.assertEqual(
                    command[command.index("--candidate-exact-proof-mask") + 1],
                    str(candidate_mask),
                )
                self.assertEqual(
                    command[command.index("--reference-exact-proof-mask") + 1],
                    "0",
                )
                self.assertEqual(
                    command[command.index("--candidate-later-ms") + 1], "165"
                )
                self.assertEqual(
                    command[command.index("--operational-later-ms") + 1], "200"
                )

    def test_configuration_is_exact_full_development_contract(self):
        expected_hashes = ",".join(bank[3] for bank in recorder.BANKS)
        for reference in recorder.ALLOWED_REFERENCES:
            configuration = recorder.configuration_expected(reference)
            self.assertEqual(configuration["reference_engine"], reference)
            self.assertEqual(configuration["bank_count"], "4")
            self.assertEqual(configuration["expected_role"], "development")
            self.assertEqual(configuration["expected_depths"], "4,8,12,20")
            self.assertEqual(configuration["expected_sha256"], expected_hashes)
            self.assertEqual(configuration["candidate_clock"], "800/165")
            self.assertEqual(configuration["reference_clock"], "800/165")
            self.assertEqual(configuration["operational_clock"], "1000/200")
            self.assertEqual(configuration["transcripts"], "not-retained")

    def test_bank_registry_accounts_for_exact_306_paired_games(self):
        self.assertEqual(sum(bank[4] for bank in recorder.BANKS), 306)
        self.assertEqual(sum(bank[4] // 2 for bank in recorder.BANKS), 153)
        self.assertEqual(len(set(bank[3] for bank in recorder.BANKS)), 4)
        self.assertTrue(all("development_" in bank[0] for bank in recorder.BANKS))

    def test_sum_bank_accounting_is_exact_and_rejects_missing_fields(self):
        banks = [
            {
                "games": str(games),
                "candidate_wins": str(games // 2),
                "reference_wins": str(games // 2),
                "unfinished": "0",
                "failed": "0",
            }
            for games in (78, 76, 76, 76)
        ]
        summed = recorder.sum_bank_accounting(banks)
        self.assertEqual(summed["games"], 306)
        self.assertEqual(summed["candidate_wins"], 153)
        self.assertEqual(summed["reference_wins"], 153)
        del banks[0]["failed"]
        with self.assertRaisesRegex(ValueError, "failed"):
            recorder.sum_bank_accounting(banks)

    def test_frozen_dependency_registry_covers_every_non_recorder_non_bank_input(self):
        tracked = {
            str(path.relative_to(recorder.ROOT))
            for path in recorder.TRACKED_INPUTS
        }
        exceptions = {
            str(recorder.RECORDER.relative_to(recorder.ROOT)),
            str(recorder.common.RECORDER.relative_to(recorder.ROOT)),
            *(str(path.relative_to(recorder.ROOT)) for path in recorder.BANK_PATHS),
        }
        self.assertEqual(
            set(recorder.EXPECTED_DEPENDENCY_SHA256), tracked - exceptions
        )
        self.assertEqual(
            recorder.EXPECTED_DEPENDENCY_SHA256[
                "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md"
            ],
            "50acd3d31df69579e0d6c3d68a71f20c4964f2413523d754be798f607d558438",
        )
        self.assertEqual(
            recorder.EXPECTED_DEPENDENCY_SHA256[
                "submissions/codingame/bots/rank_4/submission.cpp"
            ],
            "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9",
        )

    def test_frozen_dependency_validation_rejects_any_identity_drift(self):
        identities = {
            path: {"sha256": digest, "bytes": 1, "ascii": True}
            for path, digest in recorder.EXPECTED_DEPENDENCY_SHA256.items()
        }
        rank4 = identities[str(recorder.RANK4_SOURCE.relative_to(recorder.ROOT))]
        rank4["bytes"] = recorder.EXPECTED_RANK4_SOURCE_BYTES
        recorder.validate_frozen_identities(identities)
        identities["CMakeLists.txt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "CMakeLists.txt"):
            recorder.validate_frozen_identities(identities)

    def test_valid_full_summary_enforces_frozen_threshold_exactly(self):
        banks, aggregate = valid_summaries()
        recorder.validate_full_summaries(banks, aggregate)
        self.assertEqual(recorder.selection_threshold_errors(aggregate), [])
        aggregate["candidate_wins"] = "159"
        aggregate["reference_wins"] = "147"
        aggregate["candidate_p0"] = "76/77/0/0/153"
        errors = recorder.selection_threshold_errors(aggregate)
        self.assertEqual(len(errors), 2)

    def test_preregistered_mask3_fallback_has_complete_accounting(self):
        banks, aggregate = valid_summaries(candidate_mask=3)
        recorder.validate_full_summaries(banks, aggregate, candidate_mask=3)
        self.assertEqual(recorder.selection_threshold_errors(aggregate), [])

    def test_each_additive_engine_counter_must_equal_bank_sum(self):
        for engine in ("candidate", "reference"):
            for suffix in recorder.ENGINE_ADDITIVE_FIELDS:
                with self.subTest(engine=engine, suffix=suffix):
                    banks, aggregate = valid_summaries()
                    key = f"{engine}_{suffix}"
                    aggregate[key] = str(int(aggregate[key]) + 1)
                    with self.assertRaisesRegex(
                        ValueError, "engine counter mismatch"
                    ):
                        recorder.validate_bank_aggregate_consistency(
                            banks, aggregate
                        )

    def test_each_color_tuple_must_equal_bank_sum(self):
        for color in range(2):
            with self.subTest(color=color):
                banks, aggregate = valid_summaries()
                key = f"candidate_p{color}"
                aggregate[key] = "81/72/0/0/153"
                with self.assertRaisesRegex(ValueError, "color mismatch"):
                    recorder.validate_bank_aggregate_consistency(banks, aggregate)

    def test_each_engine_maximum_must_equal_maximum_bank_value(self):
        for engine in ("candidate", "reference"):
            suffixes = (*recorder.ENGINE_INTEGER_MAX_FIELDS,
                        *(f"{phase}_ms_max" for phase in recorder.TIMING_PHASES))
            for suffix in suffixes:
                with self.subTest(engine=engine, suffix=suffix):
                    banks, aggregate = valid_summaries()
                    key = f"{engine}_{suffix}"
                    if suffix.endswith("ms_max"):
                        aggregate[key] = str(float(aggregate[key]) + 0.001)
                    else:
                        aggregate[key] = str(int(aggregate[key]) + 1)
                    with self.assertRaisesRegex(ValueError, "maximum mismatch"):
                        recorder.validate_bank_aggregate_consistency(
                            banks, aggregate
                        )

    def test_timing_requires_finite_nonnegative_p99_not_above_max(self):
        for invalid in ("nan", "inf"):
            with self.subTest(invalid=invalid):
                banks, aggregate = valid_summaries()
                aggregate["candidate_first_ms_p99"] = invalid
                with self.assertRaisesRegex(
                    ValueError, "finite|invalid nonnegative"
                ):
                    recorder.validate_timing(aggregate)
        for key in ("candidate_first_ms_p99", "reference_later_ms_max"):
            with self.subTest(negative=key):
                banks, aggregate = valid_summaries()
                aggregate[key] = "-0.001"
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    recorder.validate_timing(aggregate)
        banks, aggregate = valid_summaries()
        aggregate["candidate_later_ms_p99"] = "166.0"
        with self.assertRaisesRegex(ValueError, "p99 exceeds maximum"):
            recorder.validate_timing(aggregate)

    def test_rebound_equals_root_leaf_ply1_ply2_per_bank_and_aggregate(self):
        banks, aggregate = valid_summaries()
        banks[2]["candidate_proof_rebound"] = "1/1/1"
        with self.assertRaisesRegex(ValueError, "rebound/scope"):
            recorder.validate_full_summaries(banks, aggregate)
        banks, aggregate = valid_summaries()
        aggregate["candidate_proof_rebound"] = "1835/8/8"
        with self.assertRaisesRegex(ValueError, "rebound/scope"):
            recorder.validate_full_summaries(banks, aggregate)

    def test_every_proof_tuple_must_equal_bank_sum(self):
        for engine in ("candidate", "reference"):
            for scope in ("rebound", "root", "leaf", "ply1", "ply2"):
                with self.subTest(engine=engine, scope=scope):
                    banks, aggregate = valid_summaries()
                    key = f"{engine}_proof_{scope}"
                    parts = aggregate[key].split("/")
                    parts[0] = str(int(parts[0]) + 1)
                    aggregate[key] = "/".join(parts)
                    with self.assertRaisesRegex(ValueError, "proof mismatch"):
                        recorder.validate_bank_aggregate_consistency(
                            banks, aggregate
                        )

    def test_rank4_prerequisite_accepts_only_content_addressed_same_head_control(self):
        head = "abc123"
        inputs = {"dependency": {"sha256": "def456"}}
        report = valid_control_report(head, inputs)
        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            canonical = recorder.canonical_json(report)
            digest = hashlib.sha256(canonical).hexdigest()
            (output / f"{digest}.json").write_bytes(canonical)
            found = recorder.find_accepted_control_report(head, inputs, output)
            self.assertEqual(found["sha256"], digest)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                recorder.find_accepted_control_report("different", inputs, output)
            with self.assertRaisesRegex(ValueError, "HEAD/status mismatch"):
                recorder.validate_control_prerequisite(
                    report, "different", inputs
                )

    def test_rank4_prerequisite_rejects_control_below_frozen_threshold(self):
        head = "abc123"
        inputs = {"dependency": {"sha256": "def456"}}
        report = valid_control_report(head, inputs)
        report["parsed"]["banks"][3]["candidate_wins"] = "39"
        report["parsed"]["banks"][3]["reference_wins"] = "37"
        report["parsed"]["banks"][3]["candidate_p0"] = "19/19/0/0/38"
        report["parsed"]["aggregate"]["candidate_wins"] = "159"
        report["parsed"]["aggregate"]["reference_wins"] = "147"
        report["parsed"]["aggregate"]["candidate_p0"] = "79/74/0/0/153"
        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            canonical = recorder.canonical_json(report)
            digest = hashlib.sha256(canonical).hexdigest()
            (output / f"{digest}.json").write_bytes(canonical)
            with self.assertRaisesRegex(ValueError, "selection thresholds"):
                recorder.find_accepted_control_report(head, inputs, output)

    def test_duplicate_control_attempts_are_never_result_shopped(self):
        head = "abc123"
        inputs = {"dependency": {"sha256": "def456"}}
        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            for suffix in ("first", "second"):
                report = valid_control_report(head, inputs)
                report["ended_utc"] = suffix
                canonical = recorder.canonical_json(report)
                digest = hashlib.sha256(canonical).hexdigest()
                (output / f"{digest}.json").write_bytes(canonical)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                recorder.find_accepted_control_report(head, inputs, output)
            with self.assertRaisesRegex(ValueError, "retries are forbidden"):
                recorder.require_no_previous_attempt(
                    "hybrid-control", head, inputs, output=output
                )

    def test_failed_control_attempt_blocks_retry_and_rank4_advance(self):
        head = "abc123"
        inputs = {"dependency": {"sha256": "def456"}}
        report = valid_control_report(head, inputs)
        report["development_selection_acceptable"] = False
        report["returncode"] = 1
        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            canonical = recorder.canonical_json(report)
            digest = hashlib.sha256(canonical).hexdigest()
            (output / f"{digest}.json").write_bytes(canonical)
            with self.assertRaisesRegex(ValueError, "not accepted"):
                recorder.find_accepted_control_report(head, inputs, output)
            with self.assertRaisesRegex(ValueError, "retries are forbidden"):
                recorder.require_no_previous_attempt(
                    "hybrid-control", head, inputs, output=output
                )

    def test_mask3_fallback_requires_a_failed_mask7_stage(self):
        head = "abc123"
        inputs = {"dependency": {"sha256": "def456"}}

        def persist(output, report):
            canonical = recorder.canonical_json(report)
            digest = hashlib.sha256(canonical).hexdigest()
            (output / f"{digest}.json").write_bytes(canonical)
            return digest

        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "mask-7 control"):
                recorder.require_mask3_fallback_authorized(head, inputs, output)

            failed_control = valid_control_report(head, inputs)
            failed_control["development_selection_acceptable"] = False
            failed_control["returncode"] = 1
            failed_digest = persist(output, failed_control)
            trigger = recorder.require_mask3_fallback_authorized(
                head, inputs, output
            )
            self.assertEqual(trigger["failed_reference_engine"], "hybrid-control")
            self.assertEqual(trigger["sha256"], failed_digest)

        with tempfile.TemporaryDirectory(dir=recorder.ROOT / "build") as directory:
            output = pathlib.Path(directory)
            persist(output, valid_control_report(head, inputs))
            with self.assertRaisesRegex(ValueError, "Rank-4 attempt"):
                recorder.require_mask3_fallback_authorized(head, inputs, output)

            failed_rank4 = valid_control_report(head, inputs)
            failed_rank4["reference_engine"] = "rank4"
            failed_rank4["development_selection_acceptable"] = False
            failed_rank4["returncode"] = 1
            failed_digest = persist(output, failed_rank4)
            trigger = recorder.require_mask3_fallback_authorized(
                head, inputs, output
            )
            self.assertEqual(trigger["failed_reference_engine"], "rank4")
            self.assertEqual(trigger["sha256"], failed_digest)

            for path in output.glob("*.json"):
                report = json.loads(path.read_bytes())
                if report.get("reference_engine") == "rank4":
                    path.unlink()
            passed_rank4 = valid_control_report(head, inputs)
            passed_rank4["reference_engine"] = "rank4"
            persist(output, passed_rank4)
            with self.assertRaisesRegex(ValueError, "passed both gates"):
                recorder.require_mask3_fallback_authorized(head, inputs, output)


if __name__ == "__main__":
    unittest.main()
