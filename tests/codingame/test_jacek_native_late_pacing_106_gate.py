import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "jacek_native_late_pacing_106_gate.py"
SPEC = importlib.util.spec_from_file_location("late_pacing_106_gate", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class LatePacing106GateTests(unittest.TestCase):
    def make_report(self, directory: pathlib.Path, first_color_wins=(30, 28),
                    operational_failures=0):
        profile = gate.selector.PROFILES["decisive"]
        candidate = {
            "checkpoint_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "packed_sha256": "c" * 64,
            "runtime_bytes": 123,
            "runtime_sha256": "d" * 64,
            "seed": 20260901,
        }
        baseline = {
            "checkpoint_sha256": "e" * 64,
            "exporter": "round2",
            "exporter_sha256": "f" * 64,
            "model_sha256": "1" * 64,
            "packed_sha256": "2" * 64,
            "runtime_bytes": 456,
            "runtime_sha256": "3" * 64,
            "seed": 20260822,
        }
        lines = [
            " ".join(
                f"candidate_{field}={candidate[field]}"
                for field in ("runtime_sha256", "model_sha256", "packed_sha256")
            ),
            " ".join(
                f"baseline_{field}={baseline[field]}"
                for field in ("runtime_sha256", "model_sha256", "packed_sha256")
            ),
        ]
        color_wins = list(first_color_wins)
        for pair in range(profile.pairs):
            win_zero = pair < first_color_wins[0] or 53 <= pair < 83
            win_one = pair < first_color_wins[1] or 53 <= pair < 83
            color_wins[0] += int(pair >= 53 and win_zero)
            color_wins[1] += int(pair >= 53 and win_one)
            lines.append(
                f"pair={pair} opening_turns={gate.selector.OPENING_TURNS[pair % 4]} "
                f"seed={100000 + pair} c0={0 if win_zero else 1} "
                f"c1={1 if win_one else 0}"
            )
        candidate_wins = sum(color_wins)
        passed = (
            candidate_wins >= profile.minimum_candidate_wins
            and min(color_wins) >= profile.minimum_wins_per_color
            and operational_failures == 0
        )
        summary = {
            "candidate": candidate_wins,
            "baseline": profile.pairs * 2 - candidate_wins,
            "unfinished": 0,
            "candidate_player_one": color_wins[0],
            "candidate_player_two": color_wins[1],
            "games": profile.pairs * 2,
            "candidate_decisions": 1000,
            "candidate_expansions": 10000,
            "candidate_child_evaluations": 100000,
            "candidate_max_tree": 4096,
            "candidate_ms": 1000.0,
            "candidate_max_first_ms": 800.0,
            "candidate_max_later_ms": 155.0,
            "candidate_deadline_searches": 100,
            "candidate_headroom_failures": operational_failures,
            "candidate_operational_timeouts": 0,
            "baseline_decisions": 1000,
            "baseline_expansions": 10000,
            "baseline_max_first_ms": 800.0,
            "baseline_max_later_ms": 155.0,
            "baseline_headroom_failures": 0,
            "baseline_operational_timeouts": 0,
            "profile": "800/155",
            "shuffle_seed_policy": "deployment-constant",
            "required_total": profile.minimum_candidate_wins,
            "required_per_color": profile.minimum_wins_per_color,
            "passed": passed,
        }
        lines.append("summary " + " ".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in summary.items()
        ))
        stdout_raw = ("\n".join(lines) + "\n").encode()
        stdout_sha = digest(stdout_raw)
        stdout_name = f"{stdout_sha}.stdout.txt"
        (directory / stdout_name).write_bytes(stdout_raw)
        command = gate.selector._gate_command(
            pathlib.Path("$GATE_BINARY"), pathlib.Path("$CANDIDATE_RUNTIME"),
            pathlib.Path("$BASELINE_RUNTIME"), profile,
        )
        report = {
            "schema": gate.selector.REPORT_SCHEMA,
            "profile": profile.payload(),
            "candidate": candidate,
            "baseline": baseline,
            "execution": {
                "gate_binary_sha256": "4" * 64,
                "gate_binary_bytes": 999,
                "gate_sources": gate.selector._source_identities(),
                "selection_tool_sha256": digest(
                    pathlib.Path(gate.selector.__file__).read_bytes()
                ),
                "round1_exporter_sha256": digest(
                    pathlib.Path(gate.selector.round1_exporter.__file__).read_bytes()
                ),
                "round2_exporter_sha256": digest(
                    pathlib.Path(gate.selector.round2_exporter.__file__).read_bytes()
                ),
                "serial_actual_clock_lock": True,
                "command": command,
                "exit_code": 0 if passed else 1,
            },
            "stdout": {
                "path": stdout_name,
                "sha256": stdout_sha,
                "bytes": len(stdout_raw),
            },
            "result": summary,
        }
        report_raw = gate.selector.canonical_json_bytes(report)
        report_path = directory / f"{digest(report_raw)}.json"
        report_path.write_bytes(report_raw)
        return report_path

    def test_first_53_pairs_pass_at_58_and_25_per_color(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source = self.make_report(directory)
            report = gate.build_report(source)
            self.assertTrue(report["result"]["passed"])
            self.assertEqual(report["result"]["candidate_wins"], 58)
            self.assertEqual(
                (report["result"]["candidate_player_one_wins"],
                 report["result"]["candidate_player_two_wins"]),
                (30, 28),
            )
            output = gate.write_content_addressed(directory / "subset", report)
            self.assertEqual(output.stem, digest(output.read_bytes()))

    def test_prefix_strength_and_full_window_operations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            # Use separate directories because selector stdout is content-addressed.
            weak_dir = directory / "weak"
            weak_dir.mkdir()
            weak_report = gate.build_report(self.make_report(weak_dir, (29, 28)))
            self.assertFalse(weak_report["result"]["passed"])
            operations_dir = directory / "operations"
            operations_dir.mkdir()
            operations = gate.build_report(
                self.make_report(operations_dir, (30, 28), operational_failures=1)
            )
            self.assertFalse(operations["result"]["passed"])

    def test_tampered_stdout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source = self.make_report(directory)
            report = json.loads(source.read_bytes())
            stdout = directory / report["stdout"]["path"]
            stdout.write_bytes(stdout.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(gate.GateError, "selector audit"):
                gate.build_report(source)


if __name__ == "__main__":
    unittest.main()
