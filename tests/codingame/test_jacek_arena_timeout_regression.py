import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
CASES = BOT / "window001_timeout_regressions.tsv"
PROBE = BOT / "timeout_regression_probe.cpp"
EXACT_SOURCE_SHA256 = "3883f4c3f29a32c039492adc6151e94b5dfd84653ce0dfb2383356e7f5e3c9f8"
EXACT_SOURCE = (
    ROOT
    / "results"
    / "jacek_arena_bfm"
    / "arena"
    / "source_payloads"
    / f"{EXACT_SOURCE_SHA256}.source"
)
EXPECTED_STATES = {
    898882047: ("a1ee47cb315e037a", 24, 0),
    898882199: ("1e7837ea639b26a6", 118, 1),
    898882273: ("41293163f4874278", 126, 1),
}
RUN_LEGACY_TIMING_EVIDENCE = (
    os.environ.get("PAPERSOCCER_RUN_LEGACY_TIMEOUT_EVIDENCE") == "1"
)


def run_probe(executable, budget, repetitions=1):
    completed = subprocess.run(
        [str(executable), str(CASES), str(budget), str(repetitions)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return [json.loads(line) for line in completed.stdout.splitlines()]


@unittest.skipUnless(shutil.which("clang++") or shutil.which("c++"),
                     "requires a C++20 compiler")
class TimeoutRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.compiler = shutil.which("clang++") or shutil.which("c++")
        cls.candidate = pathlib.Path(cls.temporary.name) / "candidate-probe"
        command = [
            cls.compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            str(PROBE),
            str(BOT / "engine.cpp"),
            "-o",
            str(cls.candidate),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise AssertionError(completed.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_fixture_is_fresh_window_diagnostic_only_and_replays_exact_states(self):
        payload = CASES.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "8bc972c222a498d8ddbe97c993c2f0d8804b321a8f4c9fac12306c10b2bffd48",
        )
        text = payload.decode("ascii")
        self.assertIn("# window_id=collection-001\n", text)
        self.assertIn(f"# submitted_source_sha256={EXACT_SOURCE_SHA256}\n", text)
        self.assertIn("# purpose=operational-regression-only\n", text)
        self.assertIn("# training_eligible=false\n", text)
        rows = run_probe(self.candidate, 1)
        self.assertEqual({row["game_id"] for row in rows}, set(EXPECTED_STATES))
        for row in rows:
            state_hash, ply, player = EXPECTED_STATES[row["game_id"]]
            self.assertEqual(row["state_hash"], state_hash)
            self.assertEqual(row["ply"], ply)
            self.assertEqual(row["focus_player"], player)

    def test_expired_generation_returns_legal_action_and_records_interrupt(self):
        for row in run_probe(self.candidate, 1):
            self.assertTrue(row["selected"])
            self.assertGreaterEqual(row["generator_deadline_stops"], 1)
            self.assertTrue(row["deadline_reached"])
            # A deadline that is already inside the search finalization reserve
            # must take the emergency complete-turn path, not the full generator.
            self.assertLess(row["nested_generator_max_us"], 10_000)
            self.assertLess(row["search_max_us"], 30_000)

    def test_later_candidate_stays_below_140ms_on_all_timeout_states(self):
        rows = run_probe(self.candidate, 128, repetitions=3)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(row["selected"])
            self.assertLess(row["search_max_us"], 140_000)
            # A faster host may exhaust the deterministic node cap before the
            # deadline, while a slower host may report a deadline stop.  The
            # expired-deadline test above independently exercises interruption;
            # this regression gate is the measured sub-140ms completion bound.
            self.assertIs(type(row["deadline_reached"]), bool)

    @unittest.skipUnless(EXACT_SOURCE.is_file(),
                         "exact fresh collection source is not present")
    def test_exact_collection_source_identity(self):
        self.assertEqual(
            hashlib.sha256(EXACT_SOURCE.read_bytes()).hexdigest(),
            EXACT_SOURCE_SHA256,
        )

    @unittest.skipUnless(
        EXACT_SOURCE.is_file() and RUN_LEGACY_TIMING_EVIDENCE,
        "set PAPERSOCCER_RUN_LEGACY_TIMEOUT_EVIDENCE=1 for local timing evidence",
    )
    def test_exact_collection_source_demonstrates_unbounded_generator_overrun(self):
        exact = pathlib.Path(self.temporary.name) / "exact-probe"
        embedded = f'-DJACEK_ARENA_BFM_EMBEDDED_RUNTIME="{EXACT_SOURCE}"'
        completed = subprocess.run(
            [
                self.compiler,
                "-std=c++20",
                "-O2",
                "-DJACEK_ARENA_BFM_LEGACY_RUNTIME",
                embedded,
                str(PROBE),
                "-o",
                str(exact),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = run_probe(exact, 155)
        self.assertTrue(any(row["search_max_us"] > 155_000 for row in rows))
        self.assertTrue(all(row["generator_deadline_stops"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
