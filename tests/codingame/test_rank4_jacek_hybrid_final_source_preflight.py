import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = (
    ROOT / "tools/record_rank4_jacek_hybrid_final_source_preflight.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rank4_jacek_hybrid_preflight", PRODUCER_PATH
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


FIXED_RECOVERY_CLAIM_UTC = "2026-08-14T12:20:00+00:00"


def summary_fields(label: str) -> dict[str, str]:
    fields = {name: "0" for name in preflight.SUMMARY_FIELDS}
    fields.update({
        "bank": label,
        "games": str(preflight.DEVELOPMENT_CONTRACT_BANK_GAMES),
        "candidate_wins": "40",
        "reference_wins": "38",
        "candidate_p0": "20/19/0/0/39",
        "candidate_p1": "20/19/0/0/39",
        "candidate_sweeps": "1",
        "reference_sweeps": "0",
        "split_pairs": "38",
        "unresolved_pairs": "0",
    })
    return fields


def configuration_fields() -> dict[str, str]:
    return {
        "profile": "nodes", "reference_engine": "rank4", "bank_count": "1",
        "expected_role": "development",
        "bank_validation": (
            "schema,header,role,depth,seed,replay,state-sha256,"
            "canonical-sha256,disjoint"
        ),
        "max_turns": "320", "expected_depths": "4",
        "expected_seeds": preflight.DEVELOPMENT_CONTRACT_BANK_SEED,
        "expected_sha256": preflight.DEVELOPMENT_CONTRACT_BANK_SHA256,
        "candidate_nodes": "500", "reference_nodes": "500",
        "candidate_clock": "800/165", "reference_clock": "800/165",
        "operational_clock": "1000/200",
        "candidate_exact_proof_mask": "7",
        "reference_exact_proof_mask": "0",
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled", "transcripts": "not-retained",
    }


def line(prefix: str, fields: dict[str, str]) -> str:
    return prefix + " " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def valid_gate_stdout() -> str:
    bank = summary_fields("0")
    aggregate = summary_fields("all")
    return "\n".join((
        line("bank_summary", bank),
        line("summary", aggregate),
        line("configuration", configuration_fields()),
    )) + "\n"


class FinalSourcePreflightTest(unittest.TestCase):
    def test_fixed_plan_has_no_heldout_or_path_knobs(self):
        source = PRODUCER_PATH.read_text(encoding="ascii")
        self.assertIn("parser.parse_args()", source)
        self.assertNotIn("add_argument(", source)
        for name in (
            "validation_d04.tsv", "validation_d08.tsv", "validation_d12.tsv",
            "validation_d20.tsv", "final_d04.tsv", "final_d08.tsv",
            "final_d12.tsv", "final_d20.tsv",
        ):
            self.assertNotIn(f'ROOT / "{name}"', source)
        self.assertEqual(
            preflight.DEVELOPMENT_CONTRACT_BANK.name, "development_d04.tsv"
        )
        self.assertNotIn("shutil.which(", source)

    def test_tracked_closure_includes_both_producers_and_tests(self):
        paths = set(preflight.tracked_inputs())
        self.assertIn(preflight.PRODUCER, paths)
        self.assertIn(preflight.QUALIFICATION_RECORDER, paths)
        self.assertIn(preflight.PRODUCER_TEST, paths)
        self.assertIn(preflight.QUALIFICATION_TEST, paths)
        self.assertIn(preflight.PLAN, paths)
        self.assertIn(preflight.RECOVERY_PLAN, paths)
        self.assertIn(preflight.FAILED_SUCCESSOR_CLAIM, paths)
        self.assertIn(preflight.FAILED_SUCCESSOR_FAILURE, paths)
        self.assertNotIn(
            ROOT / "results/rank_4_jacek_hybrid/openings/validation_d04.tsv",
            paths,
        )

    def test_gate_contract_requires_exact_explicit_pair_fields(self):
        contract = preflight.validate_gate_contract(valid_gate_stdout())
        self.assertTrue(contract["passed"])
        self.assertEqual(contract["paired_sweep_fields"], list(preflight.PAIR_FIELDS))
        extra = valid_gate_stdout().replace(
            "bank_summary ", "bank_summary extra_secret=forbidden ", 1
        )
        with self.assertRaisesRegex(ValueError, "field set"):
            preflight.validate_gate_contract(extra)
        inferred = valid_gate_stdout().replace(
            "candidate_sweeps=1", "candidate_sweeps=0"
        )
        with self.assertRaisesRegex(ValueError, "accounting"):
            preflight.validate_gate_contract(inferred)

    def test_non_executable_compiler_fixture_cannot_pass_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "clang++"
            fake.write_text("not a compiler\n", encoding="ascii")
            fake.chmod(stat.S_IRUSR | stat.S_IWUSR)
            with mock.patch.object(
                preflight, "fixed_named_executable", return_value=fake
            ), self.assertRaisesRegex(ValueError, "no executable clang"):
                preflight.discover_compiler("clang")

    def test_discovered_tool_cannot_target_repository_or_sealed_bank(self):
        sealed = (
            ROOT / "results/rank_4_jacek_hybrid/openings/final_d04.tsv"
        )
        with mock.patch.object(
            Path, "read_bytes", side_effect=AssertionError("sealed read")
        ), self.assertRaisesRegex(ValueError, "external executable"):
            preflight.external_executable_path(sealed)

    def test_optional_input_symlink_is_rejected_without_following_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "final_d04.tsv"
            link = root / "optional.cpp"
            link.symlink_to(target)
            with mock.patch.object(
                Path, "read_bytes", side_effect=AssertionError("target read")
            ), self.assertRaisesRegex(ValueError, "symlink"):
                preflight.optional_regular_file_exists(link)

    def test_atomic_preflight_claim_is_one_shot_and_content_bound(self):
        environment = {"sha256": "a" * 64}
        host = {"sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            preflight, "CLAIMS", Path(directory) / "claims"
        ), mock.patch.object(
            preflight, "utc_now", return_value=FIXED_RECOVERY_CLAIM_UTC
        ):
            path, claim = preflight.create_preflight_claim(
                "c" * 40, "d" * 64, environment, host
            )
            embedded = {**claim, "path": preflight.identity_label(path)}
            preflight.validate_preflight_claim(
                embedded, "c" * 40, "d" * 64, environment, host
            )
            with self.assertRaisesRegex(ValueError, "spent"):
                preflight.create_preflight_claim(
                    "c" * 40, "d" * 64, environment, host
                )

    def test_predecessor_claim_and_failure_receipt_are_exact_and_canonical(self):
        claim_raw = preflight.PREDECESSOR_CLAIM.read_bytes()
        failure_raw = preflight.PREDECESSOR_FAILURE.read_bytes()
        failure = json.loads(failure_raw)
        self.assertEqual(
            hashlib.sha256(claim_raw).hexdigest(),
            preflight.PREDECESSOR_CLAIM_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(failure_raw).hexdigest(),
            preflight.PREDECESSOR_FAILURE_SHA256,
        )
        self.assertEqual(preflight.canonical_json(failure), failure_raw)
        self.assertEqual(
            failure["outer_producer_outcome"]["stderr"]["semantic_code"],
            "clang-release-configure-rejected",
        )
        self.assertNotIn(
            "text", failure["outer_producer_outcome"]["stderr"]
        )
        self.assertFalse(
            failure["outer_producer_outcome"]
            ["original_configure_command_record_persisted"]
        )
        self.assertEqual(
            failure["independent_post_failure_diagnosis"]["classification"],
            "independent-reproduction-not-original-command-record",
        )
        self.assertEqual(
            failure["post_failure_diagnostic_state"]["project_object_files_observed"],
            0,
        )
        self.assertEqual(
            failure["failure_boundary"]["heldout_bank_files_accessed"], []
        )
        self.assertEqual(
            failure["predecessor"]["candidate_test"]["sha256"],
            "ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697",
        )
        original_git_blob = preflight.git_blob

        def committed_blob(head: str, path: Path) -> bytes:
            if path in (preflight.PREDECESSOR_CLAIM,
                        preflight.PREDECESSOR_FAILURE):
                return path.read_bytes()
            return original_git_blob(head, path)

        with mock.patch.object(
            preflight, "git_blob", side_effect=committed_blob
        ):
            evidence = preflight.validate_predecessor_evidence("a" * 40)
        self.assertEqual(
            evidence["claim"]["sha256"],
            preflight.PREDECESSOR_CLAIM_SHA256,
        )

    def test_predecessor_evidence_reference_is_mode_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "claim.json"
            evidence.write_bytes(preflight.PREDECESSOR_CLAIM.read_bytes())
            evidence.chmod(0o444)
            read_only = preflight.mode_neutral_file_reference(evidence)
            evidence.chmod(0o644)
            writable = preflight.mode_neutral_file_reference(evidence)
        self.assertEqual(read_only, writable)
        self.assertNotIn("mode", read_only)
        self.assertEqual(
            read_only["sha256"], preflight.PREDECESSOR_CLAIM_SHA256
        )

    def test_failed_successor_claim_and_receipt_are_exact_and_canonical(self):
        claim_raw = preflight.FAILED_SUCCESSOR_CLAIM.read_bytes()
        failure_raw = preflight.FAILED_SUCCESSOR_FAILURE.read_bytes()
        failure = json.loads(failure_raw)
        self.assertEqual(
            hashlib.sha256(claim_raw).hexdigest(),
            preflight.FAILED_SUCCESSOR_CLAIM_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(failure_raw).hexdigest(),
            preflight.FAILED_SUCCESSOR_FAILURE_SHA256,
        )
        self.assertEqual(preflight.canonical_json(failure), failure_raw)
        self.assertNotIn("text", failure["outer_producer_outcome"]["stderr"])
        self.assertEqual(
            failure["control_flow_inference"]["classification"],
            "inference-from-unique-producer-raise-location-corroborated-by-"
            "preserved-build-and-ctest-artifacts",
        )
        self.assertEqual(
            failure["post_failure_diagnostic_state"]["root_cause"]
            ["observed_cmake_compiler_id"], "AppleClang",
        )
        self.assertEqual(
            failure["authorization"]
            ["authorized_attempts_remaining_in_predecessor_v3"], 0,
        )
        self.assertFalse(
            failure["authorization"]["predecessor_receipts_authorize_recovery"]
        )

    def test_historical_incident_trees_and_root_topology_are_preserved(self):
        v2 = {
            "root": str(preflight.HISTORICAL_V2_CLANG.relative_to(ROOT)),
            "algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
            "files": 1_175, "directories": 604, "symlinks": 0,
            "total_bytes": 2_753_127, "object_files": 0,
            "sha256": (
                "d38a5f734a608cad7f89b63e791a2e559f3db37bcbc9c4c8a841814704c8a84d"
            ),
        }
        v3 = {
            "root": str(preflight.HISTORICAL_V3_CLANG.relative_to(ROOT)),
            "algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
            "files": 1_243, "directories": 606, "symlinks": 0,
            "total_bytes": 13_312_643, "object_files": 28,
            "sha256": (
                "9436251f7a2249c906eeb1b206efb6de51610e2fe4ecec0adcb48bc6c25b97a4"
            ),
        }
        snapshot_by_path = {
            preflight.HISTORICAL_V2_CLANG: v2,
            preflight.HISTORICAL_V3_CLANG: v3,
        }
        with mock.patch.object(
            preflight, "deterministic_tree_snapshot",
            side_effect=lambda path: snapshot_by_path[path],
        ), mock.patch.object(
            preflight, "_historical_root_topology",
            side_effect=[{"tmp_directories": []}, {"tmp_directories": ["home"]}],
        ):
            evidence = preflight.historical_workspace_evidence()
        self.assertTrue(evidence["stable"])
        self.assertEqual(evidence["root_topology"][1]["tmp_directories"], ["home"])
        drifted = dict(v3)
        drifted["object_files"] = 27
        drifted_by_path = dict(snapshot_by_path)
        drifted_by_path[preflight.HISTORICAL_V3_CLANG] = drifted
        with mock.patch.object(
            preflight, "deterministic_tree_snapshot",
            side_effect=lambda path: drifted_by_path[path],
        ), self.assertRaisesRegex(ValueError, "snapshot changed"):
            preflight.historical_workspace_evidence()
        source = PRODUCER_PATH.read_text(encoding="ascii")
        self.assertNotIn("rmtree(", source)

    def test_tree_snapshot_algorithm_is_path_size_and_digest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            artifact = nested / "probe.o"
            artifact.write_bytes(b"artifact")
            with mock.patch.object(preflight, "ROOT", root):
                record = preflight.deterministic_tree_snapshot(root)
        file_sha = hashlib.sha256(b"artifact").hexdigest()
        expected = hashlib.sha256(
            b"nested/probe.o\0" + str(len(b"artifact")).encode("ascii") +
            b"\0" + file_sha.encode("ascii") + b"\0"
        ).hexdigest()
        self.assertEqual(record["sha256"], expected)
        self.assertEqual(record["directories"], 1)
        self.assertEqual(record["object_files"], 1)

    def test_compiler_id_is_one_derived_exact_value_and_exact_path(self):
        clang_record = {
            "family": "Clang",
            "executable": {"path": "/usr/bin/clang++"},
            "version_first_line": "Apple clang version 21.0.0.21000101",
        }
        apple_metadata = (
            'set(CMAKE_CXX_COMPILER "/usr/bin/clang++")\n'
            'set(CMAKE_CXX_COMPILER_ID "AppleClang")\n'
        )
        with mock.patch.object(
            preflight.platform, "system", return_value="Darwin"
        ):
            evidence = preflight.compiler_metadata_evidence(
                apple_metadata, "Clang", Path("/usr/bin/clang++"), clang_record
            )
            self.assertEqual(evidence["expected_cmake_id"], "AppleClang")
            self.assertEqual(evidence["observed_cmake_id"], "AppleClang")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                preflight.compiler_metadata_evidence(
                    apple_metadata.replace("AppleClang", "Clang"),
                    "Clang", Path("/usr/bin/clang++"), clang_record,
                )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                preflight.compiler_metadata_evidence(
                    apple_metadata.replace(
                        "/usr/bin/clang++", "/opt/homebrew/bin/clang++"
                    ),
                    "Clang", Path("/usr/bin/clang++"), clang_record,
                )
        upstream_record = {
            "family": "Clang",
            "executable": {"path": "/usr/bin/clang++"},
            "version_first_line": "clang version 21.0.0",
        }
        upstream_metadata = (
            'set(CMAKE_CXX_COMPILER "/usr/bin/clang++")\n'
            'set(CMAKE_CXX_COMPILER_ID "Clang")\n'
        )
        with mock.patch.object(preflight.platform, "system", return_value="Linux"):
            self.assertEqual(
                preflight.compiler_metadata_evidence(
                    upstream_metadata, "Clang", Path("/usr/bin/clang++"),
                    upstream_record,
                )["expected_cmake_id"], "Clang",
            )
        gnu_record = {
            "family": "GNU", "executable": {"path": "/usr/bin/g++"},
            "version_first_line": "g++ (GCC) 15.1.0",
        }
        gnu_metadata = (
            'set(CMAKE_CXX_COMPILER "/usr/bin/g++")\n'
            'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
        )
        with mock.patch.object(preflight.platform, "system", return_value="Linux"):
            self.assertEqual(
                preflight.compiler_metadata_evidence(
                    gnu_metadata, "GNU", Path("/usr/bin/g++"), gnu_record
                )["expected_cmake_id"], "GNU",
            )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                preflight.compiler_metadata_evidence(
                    gnu_metadata.replace("GNU", "Clang"),
                    "GNU", Path("/usr/bin/g++"), gnu_record,
                )

    def test_panel_preparation_never_cleans_an_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "recovery"
            temporary = build / "tmp"
            home = temporary / "home"
            clang = build / "clang-release"
            gnu = build / "gnu-release"
            sanitizer = build / "clang-sanitized"
            home.mkdir(parents=True)
            with mock.patch.object(preflight, "BUILD_ROOT", build), \
                    mock.patch.object(preflight, "TEMPORARY_DIRECTORY", temporary), \
                    mock.patch.object(preflight, "ISOLATED_HOME", home), \
                    mock.patch.object(preflight, "CLANG_BUILD", clang), \
                    mock.patch.object(preflight, "GNU_BUILD", gnu), \
                    mock.patch.object(preflight, "SANITIZER_BUILD", sanitizer):
                preflight.prepare_build_directory(clang)
                marker = clang / "spent-marker"
                marker.write_text("preserve\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "retry forbidden"):
                    preflight.prepare_build_directory(clang)
                self.assertEqual(marker.read_text(encoding="ascii"), "preserve\n")

    def test_recovery_policy_requires_direct_child_and_exact_path_closure(self):
        head = "a" * 40
        comparison_gate_frozen = tuple(
            record for record in preflight.UNCHANGED_FINALIST_FILES
            if record[0] == preflight.ORDINARY_GATE_SOURCE
        )
        self.assertEqual(len(comparison_gate_frozen), 1)

        def git_text(*arguments: str) -> str:
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{head} {preflight.FAILED_SUCCESSOR_HEAD}"
            if arguments[:2] == ("diff", "--name-only"):
                return "\n".join(preflight.TECHNICAL_RECOVERY_ALLOWED_PATHS)
            raise AssertionError(arguments)

        with mock.patch.object(preflight, "git_text", side_effect=git_text), \
                mock.patch.object(preflight, "validate_recovery_plan"), \
                mock.patch.object(
                    preflight, "validate_failed_successor_evidence",
                    return_value={"plan_v2": {}, "plan_v3": {}},
                ), mock.patch.object(
                    preflight, "historical_workspace_evidence",
                    return_value={"stable": True},
                ), mock.patch.object(
                    preflight, "git_blob",
                    side_effect=lambda _head, path: path.read_bytes(),
                ), mock.patch.object(
                    preflight, "UNCHANGED_FINALIST_FILES",
                    comparison_gate_frozen,
                ), self.assertRaisesRegex(
                    ValueError,
                    r"technical recovery changed frozen input: .*comparison_gate\.cpp",
                ):
            preflight.technical_recovery_record(head)

        # V19 intentionally changed the live DEVELOPMENT comparison gate.  The
        # historical recovery recorder must reject that drift.  Exact blobs
        # from the frozen recovery head exercise its accepting path without
        # changing or rebaselining any frozen identity.
        with tempfile.TemporaryDirectory() as directory:
            historical_finalists = []
            historical_labels = {}
            for index, (path, expected_bytes, expected_sha256) in enumerate(
                    preflight.UNCHANGED_FINALIST_FILES):
                raw = preflight.git_blob(preflight.FAILED_SUCCESSOR_HEAD, path)
                self.assertEqual(len(raw), expected_bytes)
                self.assertEqual(preflight.sha256_bytes(raw), expected_sha256)
                fixture = Path(directory) / f"{index}-{path.name}"
                fixture.write_bytes(raw)
                historical_labels[fixture] = preflight.identity_label(path)
                historical_finalists.append(
                    (fixture, expected_bytes, expected_sha256)
                )
            self.assertEqual(len(historical_finalists), 6)

            original_identity_label = preflight.identity_label

            def historical_identity_label(path: Path) -> str:
                if path in historical_labels:
                    return historical_labels[path]
                return original_identity_label(path)

            with mock.patch.object(
                    preflight, "git_text", side_effect=git_text
            ), mock.patch.object(preflight, "validate_recovery_plan"), \
                    mock.patch.object(
                        preflight, "validate_failed_successor_evidence",
                        return_value={"plan_v2": {}, "plan_v3": {}},
                    ), mock.patch.object(
                        preflight, "historical_workspace_evidence",
                        return_value={"stable": True},
                    ), mock.patch.object(
                        preflight, "UNCHANGED_FINALIST_FILES",
                        tuple(historical_finalists),
                    ), mock.patch.object(
                        preflight, "identity_label",
                        side_effect=historical_identity_label,
                    ), mock.patch.object(
                        preflight, "git_blob",
                        side_effect=lambda _head, path: path.read_bytes(),
                    ):
                record = preflight.technical_recovery_record(head)
        self.assertTrue(record["direct_parent_verified"])
        self.assertEqual(
            record["changed_paths"],
            sorted(preflight.TECHNICAL_RECOVERY_ALLOWED_PATHS),
        )

        def extra_path_git_text(*arguments: str) -> str:
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{head} {preflight.FAILED_SUCCESSOR_HEAD}"
            return "\n".join((*preflight.TECHNICAL_RECOVERY_ALLOWED_PATHS,
                              "unexpected.cpp"))

        with mock.patch.object(
            preflight, "git_text", side_effect=extra_path_git_text
        ), self.assertRaisesRegex(ValueError, "closure"):
            preflight.technical_recovery_record(head)

    def test_plan_projection_proves_qualification_semantics_unchanged(self):
        successor = json.loads(preflight.PLAN.read_bytes())
        predecessor = json.loads(
            preflight.git_blob(preflight.PREDECESSOR_HEAD, preflight.PLAN)
        )
        projected = preflight.predecessor_plan_projection(successor)
        self.assertTrue(preflight.exact_json_equal(projected, predecessor))
        drifted = copy.deepcopy(successor)
        drifted["configuration"]["candidate_nodes"] += 1
        self.assertFalse(preflight.exact_json_equal(
            preflight.predecessor_plan_projection(drifted), predecessor
        ))
        recovery = json.loads(preflight.RECOVERY_PLAN.read_bytes())
        self.assertEqual(
            recovery["qualification_semantics_projection"]["source_plan"]
            ["sha256"], preflight.SUCCESSOR_PLAN_SHA256,
        )
        self.assertEqual(
            preflight.PLAN.read_bytes(),
            preflight.git_blob(preflight.FAILED_SUCCESSOR_HEAD, preflight.PLAN),
        )
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / "PLAN.json"
            changed = copy.deepcopy(recovery)
            changed["technical_delta"]["thresholds_changed"] = True
            changed_path.write_bytes(preflight.canonical_json(changed))
            with mock.patch.object(
                preflight, "RECOVERY_PLAN", changed_path
            ), self.assertRaisesRegex(ValueError, "identity"):
                preflight.validate_recovery_plan()

    def test_third_head_or_other_foreign_claim_fails_before_validation(self):
        head = "a" * 40
        prior_successor_or_third_head_claim = (
            preflight.CLAIMS / ("b" * 40 + ".json")
        )
        with mock.patch.object(
            preflight, "fixed_registry_files",
            return_value=[prior_successor_or_third_head_claim],
        ), self.assertRaisesRegex(ValueError, "foreign"):
            preflight.validate_recovery_claim_registry(head)

    def test_successor_workspace_creates_empty_home_without_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "recovery-v1"
            temporary = build / "tmp"
            home = temporary / "home"
            with mock.patch.object(preflight, "BUILD_ROOT", build), \
                    mock.patch.object(
                        preflight, "TEMPORARY_DIRECTORY", temporary
                    ), mock.patch.object(preflight, "ISOLATED_HOME", home):
                preflight.prepare_fresh_recovery_workspace()
                self.assertTrue(home.is_dir())
                self.assertEqual(list(home.iterdir()), [])
                marker = home / "spent-attempt-marker"
                marker.write_text("preserve\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "retry forbidden"):
                    preflight.prepare_fresh_recovery_workspace()
                self.assertEqual(marker.read_text(encoding="ascii"), "preserve\n")

    def test_spent_recovery_claim_is_checked_before_workspace_creation(self):
        source = PRODUCER_PATH.read_text(encoding="ascii")
        main = source[source.index("def main() -> int:"):]
        self.assertLess(
            main.index("if os.path.lexists(claim_path)"),
            main.index("prepare_fresh_recovery_workspace()"),
        )
        self.assertLess(
            main.index("prepare_fresh_recovery_workspace()"),
            main.index("environment = environment_record()"),
        )
        self.assertLess(
            main.index("prepare_fresh_recovery_workspace()"),
            main.index("host_before = host_identity()"),
        )
        self.assertLess(
            main.index("prepare_fresh_recovery_workspace()"),
            main.index("create_preflight_claim("),
        )
        self.assertLess(
            main.index("require_recovery_downstream_absent_before_claim()"),
            main.index("prepare_fresh_recovery_workspace()"),
        )

    def test_preclaim_recovery_downstream_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            preflight, "RECOVERY_ROOT", Path(directory)
        ):
            preflight.require_recovery_downstream_absent_before_claim()
            binding = Path(directory) / "bindings" / "spent.json"
            binding.parent.mkdir()
            binding.write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "binding registry"):
                preflight.require_recovery_downstream_absent_before_claim()

    def test_command_records_never_retain_raw_streams(self):
        stdout = "marker\n"
        record = {
            "argv": ["fixed"], "cwd": str(preflight.ROOT),
            "environment_sha256": preflight.environment_record()["sha256"],
            "started_utc": "2026-08-14T10:00:00+00:00",
            "ended_utc": "2026-08-14T10:00:01+00:00",
            "elapsed_monotonic_ns": 1,
            "timeout_seconds": 30, "returncode": 0, "timed_out": False,
            "os_error_class": None,
            "stdout": preflight._stream_record(stdout),
            "stderr": preflight._stream_record(""),
            "required_stdout_markers": {"marker": True}, "passed": True,
        }
        preflight.validate_command_record(record, ["fixed"], 30, ("marker",))
        self.assertNotIn(stdout, str(record))
        self.assertFalse(record["stdout"]["retained"])
        for key, value in (
            ("returncode", False), ("elapsed_monotonic_ns", True),
        ):
            malformed = copy.deepcopy(record)
            malformed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                preflight.validate_command_record(
                    malformed, ["fixed"], 30, ("marker",)
                )
        malformed_stream = copy.deepcopy(record)
        malformed_stream["stderr"]["bytes"] = False
        with self.assertRaisesRegex(ValueError, "stream"):
            preflight.validate_command_record(
                malformed_stream, ["fixed"], 30, ("marker",)
            )

    def test_self_asserted_minimal_receipt_is_rejected_before_tool_use(self):
        forged = {
            "schema": preflight.SCHEMA,
            "status": "passed",
            "checks": {
                name: {"status": "passed"} for name in preflight.REQUIRED_CHECKS
            },
        }
        with self.assertRaisesRegex(ValueError, "schema fields"):
            preflight.validate_passed_receipt(
                forged, "a" * 64, "b" * 40,
                preflight.file_identity(preflight.RECOVERY_PLAN)["sha256"],
            )

    def test_build_and_test_panels_are_fixed_and_sanitized(self):
        clang = Path("/usr/bin/clang++")
        command = preflight.configure_command(
            preflight.SANITIZER_BUILD, clang, True
        )
        self.assertIn("-DPAPERSOCCER_ENABLE_SANITIZERS=ON", command)
        targets = preflight.fixed_targets()
        self.assertEqual(
            preflight.build_command(preflight.CLANG_BUILD)[-len(targets):],
            list(targets),
        )
        self.assertEqual(
            "position_key_cache_test" in preflight.test_regex(True),
            preflight.optional_regular_file_exists(preflight.POSITION_KEY_TEST),
        )
        environment = preflight.sanitized_environment()
        self.assertEqual(environment["HOME"], str(preflight.ISOLATED_HOME))
        self.assertTrue(preflight.ISOLATED_HOME.is_relative_to(
            preflight.TEMPORARY_DIRECTORY
        ))
        self.assertTrue(preflight.TEMPORARY_DIRECTORY.is_relative_to(
            preflight.BUILD_ROOT
        ))
        self.assertIn("preflight-recovery-v1", str(preflight.BUILD_ROOT))
        self.assertNotEqual(environment["HOME"], os.environ.get("HOME"))
        self.assertEqual(environment["TZ"], "UTC")

    def test_darwin_asan_policy_disables_only_unsupported_leak_detection(self):
        with mock.patch.object(preflight.platform, "system", return_value="Darwin"):
            environment = preflight.sanitized_environment()
            record = preflight.environment_record()
        self.assertIn("abort_on_error=1", environment["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", environment["ASAN_OPTIONS"])
        self.assertIn("detect_leaks=0", environment["ASAN_OPTIONS"])
        self.assertFalse(record["asan_leak_detection"]["detect_leaks"])
        self.assertEqual(record["asan_leak_detection"]["platform"], "Darwin")
        with mock.patch.object(preflight.platform, "system", return_value="Linux"):
            self.assertIn(
                "detect_leaks=1", preflight.sanitized_environment()["ASAN_OPTIONS"]
            )

    def test_dedicated_gate_wrapper_and_development_contract_are_fixed(self):
        wrapper = (
            preflight.BOT / "comparison_gate_heldout.cpp"
        ).read_text(encoding="ascii")
        generic = (preflight.BOT / "comparison_gate.cpp").read_text(
            encoding="ascii"
        )
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="ascii")
        self.assertEqual(
            wrapper,
            "#define PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING 1\n"
            "#include \"comparison_gate.cpp\"\n",
        )
        with self.assertRaisesRegex(
            ValueError, "ordinary comparison-gate projection changed"
        ):
            preflight.heldout_gate_isolation_checks()

        historical_gate = preflight.git_blob(
            preflight.FAILED_SUCCESSOR_HEAD, preflight.ORDINARY_GATE_SOURCE
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "comparison_gate.cpp"
            fixture.write_bytes(historical_gate)
            with mock.patch.object(preflight, "ORDINARY_GATE_SOURCE", fixture):
                isolation = preflight.heldout_gate_isolation_checks()
        self.assertTrue(isolation["passed"])
        self.assertEqual(
            isolation["ordinary_projection_sha256"],
            preflight.ORDINARY_GATE_BASELINE_SHA256,
        )
        self.assertIn("PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING", generic)
        for field in preflight.PAIR_FIELDS:
            self.assertIn(field, generic)
        pair_logic = generic[
            generic.index("void add_opening_pair("):
            generic.index("void merge_summary(")
        ]
        self.assertIn("pair_outcome(candidate_player0, 0)", pair_logic)
        self.assertIn("pair_outcome(candidate_player1, 1)", pair_logic)
        self.assertIn("2 * summary.candidate_sweeps", pair_logic)
        self.assertIn("2 * summary.reference_sweeps", pair_logic)
        opening_loop = generic[generic.index(
            "for (const opening_bank::OpeningRecord &record : bank.records)"
        ):]
        self.assertIn("play(opening, 0, config)", opening_loop)
        self.assertIn("play(opening, 1, config)", opening_loop)
        self.assertIn("add_opening_pair", opening_loop)
        self.assertIn(preflight.GATE_TARGET, cmake)
        self.assertEqual(preflight.gate_contract_command()[0], str(preflight.FINAL_GATE))
        self.assertEqual(
            Path(preflight.gate_contract_command()[
                preflight.gate_contract_command().index("--bank") + 1
            ]),
            preflight.DEVELOPMENT_CONTRACT_BANK,
        )

    def test_process_preflight_detects_competing_campaign_job(self):
        processes = [
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 10, "ppid": 1, "command": "python preflight producer"},
            {"pid": 99, "ppid": 1,
             "command": "papersoccer_codingame_rank_4 competing"},
        ]
        evidence = preflight.process_preflight_from_table(processes, 10)
        self.assertFalse(evidence["clean"])
        self.assertEqual([item["pid"] for item in evidence["conflicts"]], [99])


if __name__ == "__main__":
    unittest.main()
