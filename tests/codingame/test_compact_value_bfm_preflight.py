import copy
import hashlib
import importlib.util
import pathlib
import re
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_preflight.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_preflight", TOOL)
preflight = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(preflight)
base = preflight.base


def digest(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def command_receipt(label, argv, markers=()):
    empty = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    return base.seal({
        "schema": preflight.COMMAND_SCHEMA,
        "label": label,
        "argv": list(argv),
        "cwd": "/workspace",
        "returncode": 0,
        "timed_out": False,
        "elapsed_ns": 1,
        "stdout": empty,
        "stderr": empty,
        "required_markers": {marker: True for marker in markers},
        "passed": True,
    })


def input_snapshot():
    sources = {}
    for index, relative in enumerate(preflight.SOURCE_CLOSURE):
        sources[relative.as_posix()] = {
            "path": f"/workspace/{relative.as_posix()}",
            "bytes": index + 1,
            "sha256": digest(f"source-{index}"),
            "ascii": True,
        }
    candidate_key = (
        preflight.BOT_RELATIVE / "submission.cpp"
    ).as_posix()
    rank4_key = preflight.RANK4_RELATIVE.as_posix()
    candidate = {
        "path": "/workspace/submission.cpp",
        "bytes": 94_999,
        "sha256": digest("candidate"),
        "ascii": True,
        "bootstrap_zero": False,
    }
    rank4 = {
        "path": "/workspace/rank4.cpp",
        "bytes": preflight.RANK4_BYTES,
        "sha256": preflight.RANK4_SHA256,
        "ascii": True,
    }
    sources[candidate_key] = candidate
    sources[rank4_key] = rank4
    return {
        "candidate_commit": "a" * 40,
        "candidate": candidate,
        "rank4": rank4,
        "runtime": {
            "path": "/workspace/model.runtime.json",
            "bytes": 100,
            "sha256": digest("runtime"),
            "ascii": True,
        },
        "sources": sources,
        "tools": {
            "python": {"path": "/workspace/.venv/bin/python", "bytes": 1,
                       "sha256": digest("python")},
            "gcc": {"path": "/usr/bin/g++", "bytes": 1,
                    "sha256": digest("gcc"), "family": "GNU"},
            "clang": {"path": "/usr/bin/clang++", "bytes": 1,
                      "sha256": digest("clang"), "family": "Clang"},
            "cmake": {"path": "/usr/bin/cmake", "bytes": 1,
                      "sha256": digest("cmake")},
            "ctest": {"path": "/usr/bin/ctest", "bytes": 1,
                      "sha256": digest("ctest")},
            "node": {"path": "/usr/bin/node", "bytes": 1,
                     "sha256": digest("node")},
            "git": {"path": "/usr/bin/git", "bytes": 1,
                    "sha256": digest("git")},
        },
    }


def timing_receipt(first=899.0, later=179.0):
    samples = []
    for count in preflight.PROCESS_COUNTS:
        for color in (0, 1):
            for replica in range(count):
                samples.append({
                    "process_count": count,
                    "color": color,
                    "replica": replica,
                    "first_ms": first,
                    "later_max_ms": later,
                })
    return {
        "schema": preflight.TIMING_SCHEMA,
        "probe_sha256": digest("timing probe"),
        "first_limit_exclusive_ms": 900.0,
        "later_limit_exclusive_ms": 180.0,
        "samples": samples,
    }


class ReceiptFixture:
    def __init__(self, root):
        self.root = root
        self.repository = root / "workspace"
        self.python = self.repository / ".venv/bin/python"
        self.plan = preflight.create_plan(
            self.repository,
            build_root=self.repository / "build/preflight",
            python_path=self.python,
            gcc_path=pathlib.Path("/usr/bin/g++"),
            clang_path=pathlib.Path("/usr/bin/clang++"),
            runtime_path=self.repository / "runtime.json",
        )
        self.inputs = input_snapshot()
        # Synthetic paths must match the pure plan's resolved worktree Python.
        self.inputs["tools"]["python"]["path"] = self.plan["python"]
        self.inputs["tools"]["cmake"]["path"] = self.plan[
            "panels"
        ]["gcc-release"]["configure"][0]
        self.inputs["tools"]["ctest"]["path"] = self.plan[
            "panels"
        ]["gcc-release"]["ctest"][0]
        self.inputs["tools"]["node"]["path"] = self.plan[
            "commands"
        ]["protocol_end_to_end"][0]
        for panel_name, tool_name in (
            ("gcc-release", "gcc"), ("clang-release", "clang")
        ):
            prefix = "-DCMAKE_CXX_COMPILER:FILEPATH="
            self.inputs["tools"][tool_name]["path"] = next(
                argument[len(prefix):]
                for argument in self.plan["panels"][panel_name]["configure"]
                if argument.startswith(prefix)
            )
        self.claim = preflight.create_claim(
            root / "evidence", plan=self.plan, inputs=self.inputs,
            claimed_at_utc="2026-08-31T20:00:00Z",
        )

    def panel(self, name):
        planned = self.plan["panels"][name]
        markers = [*planned["expected_tests"],
                   "100% tests passed, 0 tests failed"]
        sanitized = name == "clang-sanitized"
        return {
            "name": name,
            "build_path": planned["build"],
            "compiler": {
                "family": "GNU" if name == "gcc-release" else "Clang",
                "sha256": digest(name + " compiler"),
            },
            "configure": command_receipt(f"{name}:configure", planned["configure"]),
            "compile": command_receipt(f"{name}:compile", planned["compile"]),
            "cache": {
                "python_entry": f"Python3_EXECUTABLE:FILEPATH={self.python.absolute()}",
                "sanitizer_entry": (
                    "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=ON" if sanitized
                    else "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF"
                ),
                "python_equal": True,
                "checked_before_ctest": True,
            },
            "ctest": command_receipt(
                f"{name}:ctest", planned["ctest"], markers
            ),
            "instrumentation": {
                "address": sanitized, "undefined_behavior": sanitized,
            },
            "binaries": {
                target: {
                    "path": str(pathlib.Path(planned["build"]) / target),
                    "bytes": 1024,
                    "sha256": digest(f"{name}:{target}"),
                    "executable": True,
                }
                for target in preflight.BUILD_TARGETS
            },
        }

    def receipt(self):
        commands = {}
        for name, markers in preflight.DIRECT_MARKERS.items():
            commands[name] = command_receipt(
                name, self.plan["commands"][name], markers
            )
        checks = {
            name: "passed" for name in sorted({
                "fresh_gcc_release", "fresh_clang_release",
                "fresh_clang_asan_ubsan", "cmakecache_python_before_ctest",
                "complete_python_discovery", "native_compact", "frontier",
                "protocol", "end_to_end", "exporter_fresh_ascii_under_95k",
                "feature_inference_parity_4096",
                "timing_1_2_10_both_colors",
            })
        }
        return base.seal({
            "schema": preflight.RECEIPT_SCHEMA,
            "namespace": preflight.NAMESPACE,
            "status": "passed",
            "claim_body_sha256": self.claim["body_sha256"],
            "plan_body_sha256": self.plan["body_sha256"],
            "claim": self.claim,
            "plan": self.plan,
            "inputs_before": self.inputs,
            "inputs_after": self.inputs,
            "panels": {
                name: self.panel(name) for name in self.plan["panels"]
            },
            "commands": commands,
            "exporter": {
                "candidate": self.inputs["candidate"],
                "runtime_sha256": self.inputs["runtime"]["sha256"],
                "fresh": True,
                "under_95k": True,
                "commands": {
                    name: commands[name] for name in (
                        "model_exporter_current", "submission_exporter_current",
                        "submission_measure",
                    )
                },
            },
            "parity": {
                "schema": preflight.PARITY_SCHEMA,
                "states": 4096,
                "feature_states": 4096,
                "features_sha256": digest("features"),
                "cpp_sha256": digest("cpp"),
                "scalar_sha256": digest("scalar"),
                "maximum_absolute_error": 0.000001,
                "all_finite": True,
            },
            "timing": timing_receipt(),
            "checks": checks,
            "protected_banks_accessed": [],
            "git_writes": 0,
            "uploads": 0,
        })


class PlanAndCacheTest(unittest.TestCase):
    def test_search_parity_suite_is_built_run_and_source_frozen(self):
        expected_sources = {
            "search_trace_probe.cpp",
            "search_variant_parity.py",
            "state_evaluation_cache_parity.py",
            "progressive_widening_invariance.py",
            "subtree_reuse_invariance.py",
            "search_profile_exclusion.py",
            "source_compaction_parity.py",
        }
        frozen = {
            path.name for path in preflight.SOURCE_CLOSURE
            if path.parent == preflight.BOT_RELATIVE
        }
        self.assertLessEqual(expected_sources, frozen)
        self.assertTrue(preflight.SEARCH_TRACE_TARGETS)
        self.assertTrue(preflight.SEARCH_PARITY_TESTS)
        self.assertLessEqual(
            set(preflight.SEARCH_TRACE_TARGETS), set(preflight.BUILD_TARGETS)
        )
        self.assertLessEqual(
            set(preflight.SEARCH_PARITY_TESTS), set(preflight.RELEASE_TESTS)
        )
        self.assertLessEqual(
            set(preflight.SEARCH_PARITY_TESTS), set(preflight.SANITIZER_TESTS)
        )

    def test_every_fresh_panel_has_exact_typed_worktree_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            python = root / ".venv/bin/python"
            plan = preflight.create_plan(
                root, build_root=root / "build", python_path=python,
                gcc_path=pathlib.Path("/opt/g++"),
                clang_path=pathlib.Path("/opt/clang++"),
                runtime_path=root / "runtime.json",
            )
            preflight.validate_plan(plan)
            typed = f"-DPython3_EXECUTABLE:FILEPATH={python.absolute()}"
            for panel in plan["panels"].values():
                self.assertEqual(panel["configure"].count(typed), 1)
            self.assertEqual(plan["panels"]["gcc-release"]["family"], "GNU")
            self.assertEqual(plan["panels"]["clang-release"]["family"], "Clang")
            self.assertTrue(plan["panels"]["clang-sanitized"]["sanitized"])

    def test_compact_discovery_is_direct_complete_and_plan_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan = preflight.create_plan(
                ROOT, build_root=root / "build",
                python_path=ROOT / ".venv/bin/python",
                gcc_path=pathlib.Path("/opt/g++"),
                clang_path=pathlib.Path("/opt/clang++"),
                runtime_path=root / "runtime.json",
            )
            expected_command = [
                str((ROOT / ".venv/bin/python").absolute()),
                "-m", "unittest", "discover",
                "-s", str(ROOT / preflight.COMPACT_TEST_RELATIVE),
                "-p", preflight.COMPACT_TEST_PATTERN,
            ]
            self.assertEqual(
                plan["commands"]["python_discovery"], expected_command
            )

            suite = unittest.defaultTestLoader.discover(
                str(ROOT / preflight.COMPACT_TEST_RELATIVE),
                pattern=preflight.COMPACT_TEST_PATTERN,
            )
            pending = [suite]
            discovered_modules = set()
            while pending:
                current = pending.pop()
                for test in current:
                    if isinstance(test, unittest.TestSuite):
                        pending.append(test)
                    else:
                        discovered_modules.add(test.__class__.__module__)
            expected_modules = {
                path.stem
                for path in (ROOT / preflight.COMPACT_TEST_RELATIVE).glob(
                    preflight.COMPACT_TEST_PATTERN
                )
            }
            self.assertEqual(discovered_modules, expected_modules)

            forged_body = {
                key: copy.deepcopy(value)
                for key, value in plan.items() if key != "body_sha256"
            }
            forged_body["commands"]["python_discovery"][5] = str(ROOT / "tests")
            forged = base.seal(forged_body)
            with self.assertRaisesRegex(
                preflight.PreflightError, "discovery route"
            ):
                preflight.validate_plan(forged)

    def test_all_five_required_ci_jobs_route_compact_tests_directly(self):
        workflow = (ROOT / ".github/workflows/pages.yml").read_text(
            encoding="utf-8"
        )
        job_starts = list(re.finditer(r"(?m)^  ([a-z0-9-]+):\n", workflow))
        blocks = {}
        for index, match in enumerate(job_starts):
            end = (
                job_starts[index + 1].start()
                if index + 1 < len(job_starts) else len(workflow)
            )
            blocks[match.group(1)] = workflow[match.start():end]

        direct = (
            "PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \\\n"
            "            -s tests/codingame "
            "-p 'test_compact_value_bfm*.py'"
        )
        for job in ("replay-training-contract", "leaderboard-contract"):
            self.assertIn(direct, blocks[job])
            self.assertIn("pip install -r requirements-research.txt", blocks[job])

        for job in ("test-gcc", "test-clang", "test-sanitizers"):
            self.assertIn("run: ./scripts/build-and-test.sh", blocks[job])

        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            "-s ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame", cmake
        )
        self.assertIn("-p test_*.py", cmake)

    def test_untyped_or_wrong_cache_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = pathlib.Path(temporary) / ".venv/bin/python"
            valid = (
                f"Python3_EXECUTABLE:FILEPATH={python.absolute()}\n"
                "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF\n"
            )
            checked = preflight.validate_cache_text(
                valid, python_path=python, sanitized=False
            )
            self.assertTrue(checked["checked_before_ctest"])
            for invalid in (
                valid.replace(":FILEPATH=", ":UNINITIALIZED="),
                valid.replace(str(python.absolute()), "/other/python"),
            ):
                with self.assertRaisesRegex(preflight.PreflightError, "Python"):
                    preflight.validate_cache_text(
                        invalid, python_path=python, sanitized=False
                    )


class CommandAndPanelReceiptTest(unittest.TestCase):
    def test_command_receipt_binds_argv_and_markers(self):
        receipt = command_receipt("check", ["tool", "--check"], ["passed"])
        preflight.validate_command_receipt(
            receipt, label="check", argv=["tool", "--check"],
            required_markers=["passed"],
        )
        forged = copy.deepcopy(receipt)
        forged["argv"] = ["other"]
        forged = base.seal({key: value for key, value in forged.items()
                            if key != "body_sha256"})
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_command_receipt(
                forged, label="check", argv=["tool", "--check"],
                required_markers=["passed"],
            )

    def test_panels_require_cache_before_ctest_and_real_sanitizers(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(pathlib.Path(temporary))
            panel = fixture.panel("clang-sanitized")
            preflight.validate_panel(
                panel, planned=fixture.plan["panels"]["clang-sanitized"],
                python_path=fixture.python,
            )
            panel["cache"]["checked_before_ctest"] = False
            with self.assertRaisesRegex(preflight.PreflightError, "before CTest"):
                preflight.validate_panel(
                    panel, planned=fixture.plan["panels"]["clang-sanitized"],
                    python_path=fixture.python,
                )
            panel = fixture.panel("clang-sanitized")
            panel["instrumentation"]["undefined_behavior"] = False
            with self.assertRaisesRegex(preflight.PreflightError, "instrumentation"):
                preflight.validate_panel(
                    panel, planned=fixture.plan["panels"]["clang-sanitized"],
                    python_path=fixture.python,
                )


class ExportParityTimingTest(unittest.TestCase):
    def test_exporter_requires_fresh_ascii_strictly_under_95k(self):
        inputs = input_snapshot()
        passed = {name: {"passed": True} for name in (
            "model_exporter_current", "submission_exporter_current",
            "submission_measure",
        )}
        record = {
            "candidate": inputs["candidate"],
            "runtime_sha256": inputs["runtime"]["sha256"],
            "fresh": True, "under_95k": True, "commands": passed,
        }
        preflight.validate_exporter_record(record, inputs=inputs)
        for field, value in (("ascii", False), ("bytes", 95_000)):
            forged = copy.deepcopy(record)
            forged["candidate"][field] = value
            if field == "bytes":
                forged_inputs = copy.deepcopy(inputs)
                forged_inputs["candidate"]["bytes"] = value
            else:
                forged_inputs = inputs
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_exporter_record(forged, inputs=forged_inputs)

    def test_parity_requires_4096_finite_states_and_error_below_2e6(self):
        receipt = {
            "schema": preflight.PARITY_SCHEMA,
            "states": 4096, "feature_states": 4096,
            "features_sha256": digest("features"),
            "cpp_sha256": digest("cpp"), "scalar_sha256": digest("scalar"),
            "maximum_absolute_error": 0.000001999,
            "all_finite": True,
        }
        preflight.validate_parity_receipt(receipt)
        for field, value in (
            ("states", 4095), ("maximum_absolute_error", 0.000002),
            ("all_finite", False),
        ):
            forged = dict(receipt)
            forged[field] = value
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_parity_receipt(forged)

    def test_timing_requires_exact_1_2_10_both_colors_and_strict_limits(self):
        receipt = timing_receipt()
        preflight.validate_timing_receipt(receipt)
        for mutation in ("first", "later", "missing"):
            forged = copy.deepcopy(receipt)
            if mutation == "first":
                forged["samples"][0]["first_ms"] = 900.0
            elif mutation == "later":
                forged["samples"][0]["later_max_ms"] = 180.0
            else:
                forged["samples"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(
                preflight.PreflightError
            ):
                preflight.validate_timing_receipt(forged)

    def test_scalar_inference_is_finite_for_sparse_pattern(self):
        hidden_one = hidden_two = 8
        count = 6301 * hidden_one + hidden_one * hidden_two + hidden_two
        weights = [(index % 7) - 3 for index in range(count)]
        result = preflight.scalar_inference(
            [0, 315, 316, 6300], weights=weights,
            hidden_one=hidden_one, hidden_two=hidden_two,
            scale_one=0.03125, scale_two=0.0625, scale_three=0.125,
        )
        self.assertTrue(-1.0 <= result <= 1.0)


class FullReceiptAndClaimTest(unittest.TestCase):
    def test_full_supplied_receipt_validates_all_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(pathlib.Path(temporary))
            receipt = fixture.receipt()
            checked = preflight.validate_preflight_receipt(
                receipt, claim=fixture.claim, plan=fixture.plan,
                inputs=fixture.inputs,
            )
            self.assertEqual(checked["status"], "passed")
            path = preflight.write_content_addressed(
                fixture.root / "receipts",
                {key: value for key, value in receipt.items()
                 if key != "body_sha256"},
            )
            self.assertEqual(path.stem, base.sha256_file(path))

    def test_claim_is_write_once_and_spent(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(preflight.PreflightError, "spent"):
                preflight.create_claim(
                    fixture.root / "evidence", plan=fixture.plan,
                    inputs=fixture.inputs,
                    claimed_at_utc="2026-08-31T20:01:00Z",
                )

    def test_rank4_tool_or_source_tampering_fails_closed(self):
        inputs = input_snapshot()
        preflight.validate_input_snapshot(inputs)
        cases = []
        rank4 = copy.deepcopy(inputs)
        rank4["rank4"]["sha256"] = digest("wrong")
        rank4["sources"][preflight.RANK4_RELATIVE.as_posix()] = rank4["rank4"]
        cases.append(rank4)
        tool = copy.deepcopy(inputs)
        tool["tools"]["gcc"]["family"] = "Clang"
        cases.append(tool)
        source = copy.deepcopy(inputs)
        source["sources"].pop(next(iter(source["sources"])))
        cases.append(source)
        bootstrap = copy.deepcopy(inputs)
        bootstrap["candidate"]["bootstrap_zero"] = True
        bootstrap["sources"][(preflight.BOT_RELATIVE / "submission.cpp").as_posix()] = bootstrap["candidate"]
        cases.append(bootstrap)
        for case in cases:
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_input_snapshot(case)

    def test_receipt_cannot_claim_protected_access_git_write_or_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(pathlib.Path(temporary))
            for field, value in (
                ("protected_banks_accessed", ["forbidden"]),
                ("git_writes", 1), ("uploads", 1),
            ):
                receipt = fixture.receipt()
                body = {key: item for key, item in receipt.items()
                        if key != "body_sha256"}
                body[field] = value
                forged = base.seal(body)
                with self.subTest(field=field), self.assertRaises(
                    preflight.PreflightError
                ):
                    preflight.validate_preflight_receipt(
                        forged, claim=fixture.claim, plan=fixture.plan,
                        inputs=fixture.inputs,
                    )


if __name__ == "__main__":
    unittest.main()
