import importlib.util
import copy
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
                preflight.file_identity(preflight.PLAN)["sha256"],
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
        self.assertNotIn("HOME", environment)
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
