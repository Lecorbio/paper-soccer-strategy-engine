import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = (
    ROOT / "tools/record_rank4_jacek_hybrid_position_key_cache_clock.py"
)
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "position_key_cache_clock_recorder", RECORDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)

GATE_FREEZE_COMMIT = "647625a70d54f4b549b30dd7d4eb0d73c4723111"


def identity_from_bytes(path: Path, data: bytes) -> dict[str, object]:
    return {
        "path": recorder.identity_label(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ascii": all(byte < 128 for byte in data),
    }


def summary_fields(label: str) -> dict[str, str]:
    fields = {
        "bank": label,
        "games": "76",
        "candidate_wins": "38",
        "reference_wins": "38",
        "unfinished": "0",
        "failed": "0",
        "candidate_p0": "19/19/0/0/38",
        "candidate_p1": "19/19/0/0/38",
    }
    for engine in ("candidate", "reference"):
        fields.update({
            f"{engine}_invocations": "100",
            f"{engine}_searches": "100",
            f"{engine}_illegal": "0",
            f"{engine}_operational": "0",
            f"{engine}_exceptions": "0",
            f"{engine}_hard_timeouts": "0",
            f"{engine}_soft_overruns": "80",
            f"{engine}_nodes": "20000",
            f"{engine}_nodes_avg": "200.000",
            f"{engine}_nodes_p99": "300",
            f"{engine}_nodes_max": "400",
            f"{engine}_depth_avg": "4.000",
            f"{engine}_depth_max": "5",
            f"{engine}_attempted_depth_avg": "5.000",
            f"{engine}_attempted_depth_max": "6",
            f"{engine}_exhaustions": "80",
            f"{engine}_first_ms_p99": "800.100",
            f"{engine}_first_ms_max": "800.200",
            f"{engine}_later_ms_p99": "165.100",
            f"{engine}_later_ms_max": "165.200",
            f"{engine}_proof_rebound": "1110/254/3",
            f"{engine}_proof_root": "100/2/1",
            f"{engine}_proof_leaf": "1000/250/2",
            f"{engine}_proof_ply1": "10/2/0/2",
            f"{engine}_proof_ply2": "0/0/0/0",
        })
    fields["reference_nodes_avg"] = "190.000"
    fields["reference_depth_avg"] = "3.900"
    return fields


def output_line(prefix: str, fields: dict[str, str]) -> str:
    return prefix + " " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def valid_stdout() -> str:
    bank = summary_fields("0")
    aggregate = dict(bank)
    aggregate["bank"] = "all"
    return "\n".join((
        output_line("bank_summary", bank),
        output_line("summary", aggregate),
        output_line("configuration", recorder.expected_configuration()),
    )) + "\n"


class PositionKeyCacheClockRecorderTest(unittest.TestCase):
    def test_plan_command_and_lineage_are_frozen_development_only(self):
        self.assertEqual(
            recorder.SCHEMA,
            "rank4-jacek-hybrid-position-key-cache-clock-v1",
        )
        command = recorder.command_for_gate()
        joined = " ".join(command)
        self.assertIn("development_d20.tsv", joined)
        self.assertNotIn("validation", joined.lower())
        self.assertNotIn("final", joined.lower())
        self.assertEqual(command.count("7"), 2)
        self.assertNotIn("--retain-transcripts", command)
        bindings = recorder.validate_plan_and_lineage()
        self.assertEqual(
            bindings["plan"]["candidate"]["integration_commit"],
            recorder.INTEGRATION_COMMIT,
        )
        self.assertEqual(
            bindings["plan"]["reference"]["function"],
            "choose_prefastpath",
        )
        self.assertEqual(
            bindings["prototype"]["timing"]["schema"],
            "rank4-position-key-component-safe-private-microbench-v1",
        )
        evidence = bindings["plan"]["evidence_bindings"]
        self.assertEqual(
            evidence["focused_recorder_test"]["sha256"],
            recorder.EXPECTED_FOCUSED_TEST_SHA256,
        )
        self.assertEqual(
            evidence["imported_recorders"][0]["sha256"],
            recorder.EXPECTED_BASE_RECORDER_SHA256,
        )
        self.assertEqual(
            evidence["imported_recorders"][1]["sha256"],
            recorder.EXPECTED_COMMON_RECORDER_SHA256,
        )
        claim_policy = bindings["plan"]["attempt_claim_policy"]
        self.assertTrue(claim_policy["postclaim_failure_consumes_attempt"])
        self.assertTrue(claim_policy["preclaim_failure_does_not_consume_attempt"])
        self.assertEqual(
            bindings["plan"]["evidence_policy"]["shared_benchmark_lock_path"],
            "/tmp/rank4-hybrid-prototype-benchmark.lock",
        )

    def test_exact_historical_candidate_and_selected_rollback_identities(self):
        paths = (
            recorder.PRIVATE_HEADER,
            recorder.CANDIDATE_BOT,
            recorder.SOURCES,
            recorder.CANDIDATE_SOURCE,
            recorder.CANDIDATE_TEST,
            recorder.KEY_TEST,
            recorder.SHARED_HEADER,
            recorder.CONTROL_BOT,
            recorder.CONTROL_SOURCE,
            recorder.BANK,
            recorder.PLAN,
            recorder.ROLLBACK,
            recorder.SOLE_EDGE_DECISION,
            recorder.PROTOTYPE_PASS,
            recorder.PROTOTYPE_PATCH,
            recorder.PROTOTYPE_TIMING,
            recorder.CONTROL_MANIFEST,
            recorder.CMAKE,
            recorder.COMPARISON_REFERENCE,
            recorder.FOCUSED_TEST,
            recorder.BASE_RECORDER,
            recorder.COMMON_RECORDER,
        )
        integration_paths = {
            recorder.PRIVATE_HEADER,
            recorder.CANDIDATE_BOT,
            recorder.SOURCES,
            recorder.CANDIDATE_SOURCE,
            recorder.CANDIDATE_TEST,
            recorder.KEY_TEST,
            recorder.SHARED_HEADER,
            recorder.CMAKE,
            recorder.COMPARISON_REFERENCE,
            recorder.BASE_RECORDER,
            recorder.COMMON_RECORDER,
        }
        identities = {}
        for path in paths:
            relative = str(path.relative_to(ROOT))
            if path in integration_paths:
                data = recorder.base.git_blob(
                    recorder.INTEGRATION_COMMIT, relative
                )
            elif path == recorder.FOCUSED_TEST:
                data = recorder.base.git_blob(GATE_FREEZE_COMMIT, relative)
            else:
                data = path.read_bytes()
            identities[recorder.identity_label(path)] = identity_from_bytes(
                path, data
            )
        recorder.validate_exact_file_identities(identities)
        bindings = recorder.validate_plan_and_lineage()
        self.assertEqual(
            bindings["sole_edge_decision"]["decision"]["status"],
            "sole-legal-edge-rejected-mandatory-rollback",
        )
        selected = {
            item["role"]: item
            for item in bindings["rollback"]["production_artifacts"]
        }
        for path, role in (
            (recorder.CANDIDATE_BOT, "engine-source"),
            (recorder.CANDIDATE_SOURCE, "upload-source"),
            (recorder.CANDIDATE_TEST, "source-contract-test"),
        ):
            self.assertEqual(
                recorder.file_identity(path)["sha256"],
                selected[role]["sha256"],
            )
        self.assertFalse(recorder.PRIVATE_HEADER.exists())
        self.assertFalse(recorder.KEY_TEST.exists())

    def test_valid_output_reconciles_bank_color_proofs_and_timing(self):
        parsed = recorder.validate_gate_stdout(valid_stdout())
        self.assertEqual(recorder.selection_errors(parsed["aggregate"]), [])
        self.assertEqual(parsed["aggregate"]["games"], "76")
        self.assertEqual(parsed["aggregate"]["candidate_p0"].split("/")[-1], "38")
        self.assertEqual(parsed["aggregate"]["candidate_p1"].split("/")[-1], "38")

    def test_bank_and_aggregate_must_match_exactly(self):
        lines = valid_stdout().splitlines()
        lines[1] = lines[1].replace(
            "candidate_nodes=20000", "candidate_nodes=20001", 1
        )
        with self.assertRaisesRegex(ValueError, "summaries differ"):
            recorder.validate_gate_stdout("\n".join(lines) + "\n")

    def test_mask7_requires_ply2_zero_and_rebound_scope_sum(self):
        with self.assertRaisesRegex(ValueError, "disabled candidate ply2"):
            recorder.validate_gate_stdout(valid_stdout().replace(
                "candidate_proof_ply2=0/0/0/0",
                "candidate_proof_ply2=1/0/0/0",
            ))
        with self.assertRaisesRegex(ValueError, "rebound/scope"):
            recorder.validate_gate_stdout(valid_stdout().replace(
                "reference_proof_rebound=1110/254/3",
                "reference_proof_rebound=1110/253/3",
            ))

    def test_timing_is_finite_ordered_and_strict_for_both_engines(self):
        replacements = (
            ("candidate_first_ms_p99=800.100", "candidate_first_ms_p99=900.000"),
            ("reference_later_ms_max=165.200", "reference_later_ms_max=198.000"),
            ("candidate_later_ms_p99=165.100", "candidate_later_ms_p99=166.000"),
            ("candidate_first_ms_p99=800.100", "candidate_first_ms_p99=nan"),
        )
        for original, replacement in replacements:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    recorder.validate_gate_stdout(
                        valid_stdout().replace(original, replacement)
                    )

    def test_win_floors_are_conjunctive_with_progress(self):
        below_color = valid_stdout().replace(
            "candidate_p0=19/19/0/0/38 candidate_p1=19/19/0/0/38",
            "candidate_p0=18/20/0/0/38 candidate_p1=20/18/0/0/38",
        )
        parsed = recorder.validate_gate_stdout(below_color)
        self.assertIn(
            "physical color 0",
            " ".join(recorder.selection_errors(parsed["aggregate"])),
        )
        no_progress = valid_stdout().replace(
            "candidate_nodes_avg=200.000", "candidate_nodes_avg=189.000"
        ).replace(
            "candidate_depth_avg=4.000", "candidate_depth_avg=3.800"
        )
        parsed = recorder.validate_gate_stdout(no_progress)
        self.assertIn(
            "lower completed depth",
            " ".join(recorder.selection_errors(parsed["aggregate"])),
        )

    def test_dependency_routing_requires_private_candidate_and_shared_reference(self):
        candidate = {recorder.PRIVATE_HEADER.resolve()}
        reference = {
            recorder.SHARED_HEADER.resolve(),
            recorder.PRIVATE_HEADER.resolve(),
            recorder.CONTROL_BOT.resolve(),
        }
        evidence = recorder.validate_dependency_routing(candidate, reference)
        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["reference_private_header"])
        with self.assertRaisesRegex(ValueError, "private header"):
            recorder.validate_dependency_routing(set(), reference)
        with self.assertRaisesRegex(ValueError, "shared header"):
            recorder.validate_dependency_routing(
                candidate, {recorder.CONTROL_BOT.resolve()}
            )

    def test_attempt_identity_binds_head_complete_inputs_and_lineage(self):
        head = "a" * 40
        inputs_digest = "b" * 64
        key = recorder.attempt_key(head, inputs_digest)
        self.assertEqual(key["head"], head)
        self.assertEqual(key["inputs_sha256"], inputs_digest)
        self.assertEqual(key["integration_commit"], recorder.INTEGRATION_COMMIT)
        self.assertEqual(
            key["candidate_private_header_sha256"],
            recorder.EXPECTED_PRIVATE_HEADER_SHA256,
        )
        self.assertEqual(
            key["reference_wrapper_sha256"],
            recorder.EXPECTED_REFERENCE_WRAPPER_SHA256,
        )
        self.assertEqual(
            key["prototype_patch_sha256"],
            recorder.EXPECTED_PROTOTYPE_PATCH_SHA256,
        )
        self.assertEqual(
            key["focused_recorder_test_sha256"],
            recorder.EXPECTED_FOCUSED_TEST_SHA256,
        )
        self.assertEqual(
            key["base_recorder_sha256"],
            recorder.EXPECTED_BASE_RECORDER_SHA256,
        )
        self.assertEqual(
            key["common_recorder_sha256"],
            recorder.EXPECTED_COMMON_RECORDER_SHA256,
        )

    def test_one_canonical_content_addressed_attempt_prevents_retry(self):
        identifier = recorder.attempt_id("a" * 40, "b" * 64)
        payload = {"schema": recorder.SCHEMA, "attempt_id": identifier}
        raw = recorder.canonical_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected = output / f"{digest}.json"
            expected.write_bytes(raw)
            (output / "wrong-name.json").write_bytes(raw)
            self.assertEqual(
                recorder.matching_attempts(identifier, output), [expected]
            )

    def test_atomic_claim_is_canonical_exclusive_retained_and_blocks_retry(self):
        head = "a" * 40
        inputs_digest = "b" * 64
        key = recorder.attempt_key(head, inputs_digest)
        identifier = recorder.attempt_id(head, inputs_digest)
        claimed_utc = "2026-08-14T02:03:04.000000+00:00"
        with tempfile.TemporaryDirectory() as directory:
            claims = Path(directory) / "claims"
            evidence = recorder.create_attempt_claim(
                identifier, key, claimed_utc, claims
            )
            path = Path(evidence["path"])
            raw = path.read_bytes()
            payload = json.loads(raw)
            self.assertEqual(payload["attempt_id"], identifier)
            self.assertEqual(payload["attempt_key"], key)
            self.assertEqual(payload["head"], head)
            self.assertEqual(payload["inputs_sha256"], inputs_digest)
            self.assertEqual(payload["plan_sha256"], recorder.EXPECTED_PLAN_SHA256)
            self.assertEqual(payload["claimed_utc"], claimed_utc)
            self.assertEqual(recorder.canonical_json(payload), raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence["sha256"])
            self.assertEqual(
                recorder.matching_attempt_evidence(identifier, Path(directory)),
                [{"kind": "claim", "path": path.resolve()}],
            )
            with self.assertRaises(FileExistsError):
                recorder.create_attempt_claim(identifier, key, claimed_utc, claims)
            self.assertTrue(path.exists())

    def test_atomic_content_addressed_persistence_reads_back_canonical_bytes(self):
        payload = {"schema": "synthetic", "value": [3, 2, 1]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path, digest = recorder.persist_content_addressed_report(
                output, payload, 123
            )
            persisted = path.read_bytes()
            self.assertEqual(path.name, f"{digest}.json")
            self.assertEqual(persisted, recorder.canonical_json(payload))
            self.assertEqual(hashlib.sha256(persisted).hexdigest(), digest)
            self.assertEqual(
                recorder.canonical_json(json.loads(persisted)), persisted
            )
            self.assertFalse((output / f".{digest}.123.tmp").exists())

    def test_process_preflight_rejects_competing_campaign_process(self):
        clean = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 10, "ppid": 1, "command": "zsh rank4 wrapper"},
            {"pid": 11, "ppid": 10, "command": "python position_key recorder"},
        ], 11)
        self.assertTrue(clean["clean"])
        conflict = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 11, "ppid": 1, "command": "python rank4 recorder"},
            {"pid": 99, "ppid": 1,
             "command": "papersoccer_codingame_rank_4 competing-gate"},
        ], 11)
        self.assertFalse(conflict["clean"])
        self.assertEqual([item["pid"] for item in conflict["conflicts"]], [99])

    def test_recorder_has_no_user_knobs_and_fresh_build_precedes_snapshot(self):
        source = RECORDER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--clean-first"', source)
        snapshot = "full_paths, dependency_routing = collect_full_input_paths()"
        self.assertLess(source.index("build = _build_gate()"),
                        source.index(snapshot))
        self.assertLess(source.index("generated_check = subprocess.run("),
                        source.index(snapshot))
        self.assertLess(source.index("prerun_processes ="),
                        source.index("started = base.utc_now()"))
        self.assertIn("BENCHMARK_LOCK.open", source)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", source)
        self.assertLess(source.index("BENCHMARK_LOCK.open"),
                        source.index("build = _build_gate()"))
        self.assertLess(source.index("claim = create_attempt_claim("),
                        source.index("started = base.utc_now()"))
        self.assertNotIn("add_argument(", source)


if __name__ == "__main__":
    unittest.main()
