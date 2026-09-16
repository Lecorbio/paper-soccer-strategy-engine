"""Completion is an artifact read; numerical replay is an explicit operation."""

import contextlib
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
rebuild = importlib.import_module("jacek_replay_rebuild")
completion = importlib.import_module("jacek_replay_rebuild_completion")


class CompletedRebuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "run"
        self.inputs_path = self.root / "inputs.json"

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rebuild.canonical_json_bytes(value))
        return rebuild.artifact_snapshot(path)

    def seal(self, value):
        value = copy.deepcopy(value)
        value.pop("body_sha256", None)
        return {**value, "body_sha256": rebuild.sha256_bytes(rebuild.canonical_json_bytes(value))}

    def binary(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return path

    def seed(self, phase, candidate_id, arm, seed, initial, eligible):
        directory = self.output / phase / "runs" / candidate_id / arm / f"seed-{seed}"
        runtime = self.binary(str(directory.relative_to(self.root) / "recovery.runtime"))
        bound_runtime = rebuild.artifact_snapshot(runtime)
        result = {"eligible": eligible, "selection": {
            "weighted_huber": 0.05, "sign_accuracy": 0.86, "correlation": 0.8}}
        configuration = {"optimizer": {"seed": seed}}
        inputs = {"runtimes": {name: {"artifact_sha256": rec["sha256"], "bytes": rec["bytes"]}
                  for name, rec in (("initial", initial), ("new_reference", initial),
                                    ("retention_reference", self.inputs["incumbent_runtime"]))}}
        report = {"configuration": configuration, "inputs": inputs,
                  "producer": {"fixture": True}, "result": result}
        report_record = self.write(directory / "recovery.report.json", report)
        receipt = self.seal({
            "schema": "papersoccer.jacek-replay-recovery-receipt.v1",
            "configuration": configuration, "inputs": inputs, "producer": report["producer"],
            "runtime": {"file": runtime.name, "artifact_sha256": bound_runtime["sha256"],
                        "bytes": bound_runtime["bytes"]},
            "report": {"file": "recovery.report.json", "sha256": report_record["sha256"],
                       "bytes": report_record["bytes"]},
        })
        return {"seed": seed, "runtime": bound_runtime, "report": report_record,
                "receipt": self.write(directory / "recovery.receipt.json", receipt), "result": result}

    def candidate(self, phase, candidate_id, spec, *, eligible=False, launched=True, rate=None):
        base = spec or self.matrix["phases"]["v5_recovery"][0]
        value = {"schema": rebuild.REBUILD_CANDIDATE_SCHEMA, "candidate_id": candidate_id,
                 "corpus": self.inputs["corpus"], "incumbent": self.inputs["incumbent_runtime"],
                 "offline_eligible": eligible}
        for arm in ("search", "rank4"):
            rows = [self.seed(phase, candidate_id, arm, seed, base[arm + "_initial_runtime"], eligible)
                    for seed in rebuild.ORDER_SEEDS] if launched else []
            value[arm + "_seed_runs"] = rows
            value["selected_" + arm] = min(rows, key=completion._selection_key) if eligible else None
        if spec is None:
            value.update(phase="residual", rank=16, learning_rate=rate)
        else:
            value["specification"] = spec
            launch = self.seal({"schema": completion.PREFIX + "config-launch.v1",
                "rebuild_id": rebuild.REBUILD_ID, "candidate_id": candidate_id,
                "specification": spec, "corpus": self.inputs["corpus"],
                "incumbent": self.inputs["incumbent_runtime"],
                "deadline_unix": self.inputs["same_architecture_deadline_unix"],
                "launched_at_unix": self.inputs["started_at_unix"]})
            value["launch"] = self.write(self.output / phase / "launches" / (candidate_id + ".json"), launch) if launched else None
        return self.seal(value)

    def phase(self, name, candidates, *, selected=False):
        offline = [candidates[0]] if selected else []
        ident = candidates[0]["candidate_id"]
        value = self.seal({"schema": completion.PREFIX + "phase-screen.v1", "phase": name,
            "phase_inputs": {}, "candidate_records": candidates,
            "offline_eligible": len(offline), "offline_candidate_records": offline,
            "shortlisted": [ident] if selected else [],
            "short_records": [{"candidate_id": ident, "screen": {}}] if selected else [],
            "short_ranked": [ident] if selected else [], "short_rejections": [],
            "full_candidate_records": offline,
            "full_records": [{"candidate_id": ident, "short": {}, "full": {
                "decision": {"eligible_for_full": True}}}] if selected else [],
            "qualified": [ident] if selected else [], "selected_candidate_id": ident if selected else None})
        self.write(self.output / name / "screening/phase-screen.json", value)
        return value

    def fixture(self, terminal="no-development-qualified-candidate"):
        initial = self.binary("initial.runtime")
        self.matrix = rebuild.matrix_record(v5_search=initial, v5_rank4=initial,
            canonical_bases=[(f"base-{i}", self.binary(f"base-{i}.runtime")) for i in range(9)])
        artifact = self.write(self.root / "metadata.json", {"fixture": True})
        self.inputs = {key: artifact for key in (
            "corpus", "build_manifest", "opening_banks", "blind_holdout", "blind_holdout_positions",
            "blind_holdout_candidate_manifest", "blind_holdout_candidate_positions", "comparison", "rank4_teacher")}
        self.inputs.update(schema=rebuild.REBUILD_INPUT_SCHEMA, rebuild_id=rebuild.REBUILD_ID,
            matrix=self.write(self.root / "matrix.json", self.matrix),
            incumbent_runtime=rebuild.artifact_snapshot(self.binary("incumbent.runtime")),
            started_at_unix=1000, same_architecture_deadline_unix=1000 + rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS,
            repository={"path": str(self.root), "commit": "a" * 40, "clean": True},
            policies={key: False for key in ("regenerate_teacher_labels", "external_upload", "replace_rank4",
                "canonical_test_model_selection_eligible", "sealed_final_bank_model_selection_eligible")})
        self.inputs = self.seal(self.inputs)
        self.write(self.inputs_path, self.inputs)
        selected = terminal != "no-development-qualified-candidate"
        candidates = [self.candidate("v5-recovery", spec["candidate_id"], spec,
                                     eligible=selected and i == 0, launched=i == 0)
                      for i, spec in enumerate(self.matrix["phases"]["v5_recovery"])]
        phases = [self.phase("v5-recovery", candidates, selected=selected)]
        summary = {"schema": rebuild.REBUILD_DECISION_SCHEMA, "terminal": terminal, "phases": phases}
        if not selected:
            residual = [self.candidate("residual", ident, None, rate=rate) for ident, rate in zip(
                ("residual-lr1e4", "residual-lr3e4", "residual-lr1e3"), (1e-4, 3e-4, 1e-3))]
            phases.append(self.phase("residual", residual))
            summary.update(same_architecture_budget_seconds=rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS,
                residual_fallback_exhausted=True, next_scope="action-ranking-or-wider-network-requires-new-plan")
        else:
            candidate = candidates[0]
            selection = self.seal({"schema": completion.PREFIX + "selected-candidate.v1",
                "rebuild_id": rebuild.REBUILD_ID, "candidate_id": candidate["candidate_id"],
                "inputs": rebuild.artifact_snapshot(self.inputs_path), "candidate": candidate,
                "phase_screen": phases[-1], "ladder_phase_screens": phases,
                "selected_runtime": candidate["selected_search"]["runtime"],
                "matched_runtime": candidate["selected_rank4"]["runtime"],
                "protected_test_opened": False, "sealed_final_bank_opened": False,
                "blind_holdout_labels_opened": False})
            selection_record = self.write(self.output / "selected-candidate.json", selection)
            passed = terminal == "qualified-local-research-incumbent"
            qualification = self.seal({"schema": rebuild.REBUILD_QUALIFICATION_SCHEMA,
                "rebuild_id": rebuild.REBUILD_ID, "inputs": rebuild.artifact_snapshot(self.inputs_path),
                "selection": selection_record, "candidate": selection["selected_runtime"],
                "matched": selection["matched_runtime"], "pass": passed,
                "blind_holdout": self.write(self.output / "qualification/holdout.json", {"pass": passed}),
                "final_game_gate": {"decision": {"eligible_for_full": True}},
                "local_only": True, "canonical_rank4_replaced": False, "external_upload": False})
            self.write(self.output / "qualification/qualification.json", qualification)
            summary.update(selected_candidate=selection, qualification=qualification)
        self.write(self.output / "final-summary.json", summary)
        return summary

    @contextlib.contextmanager
    def no_work(self):
        with contextlib.ExitStack() as stack:
            for name in ("validate_rebuild_inputs", "run_recovery_phase", "run_residual_phase",
                         "run_scratch_pretraining", "screen_candidate_phase", "qualify_selected_candidate",
                         "validate_phase_screen", "validate_qualification_receipt", "load_frozen_rebuild_corpus"):
                stack.enter_context(mock.patch.object(rebuild, name, side_effect=AssertionError("unexpected heavy work: " + name)))
            stack.enter_context(mock.patch.object(rebuild.concurrent.futures, "ThreadPoolExecutor", side_effect=AssertionError("workers")))
            stack.enter_context(mock.patch.object(rebuild.subprocess, "run", side_effect=AssertionError("subprocess")))
            yield

    def run_saved(self):
        return rebuild.run_ladder(inputs_manifest=self.inputs_path, output_directory=self.output, resume=True)

    def test_all_terminal_outcomes_return_saved_evidence_without_heavy_work(self):
        for terminal in sorted(completion.TERMINALS):
            with self.subTest(terminal=terminal):
                summary = self.fixture(terminal)
                snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                             for p in self.root.rglob("*") if p.is_file() and p.name != "status.json"}
                with self.no_work():
                    self.assertEqual(self.run_saved(), summary)
                self.assertTrue(json.loads((self.output / "status.json").read_text())["phase"].startswith("complete-"))
                self.assertTrue(all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in snapshots.items()))

    def test_changed_artifact_fails_before_any_heavy_work(self):
        summary = self.fixture()
        path = Path(summary["phases"][0]["candidate_records"][0]["search_seed_runs"][0]["runtime"]["path"])
        path.write_bytes(b"changed")
        with self.no_work(), self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_saved()

    def test_missing_artifact_fails_before_any_heavy_work(self):
        self.fixture()
        Path(self.inputs["incumbent_runtime"]["path"]).unlink()
        with self.no_work(), self.assertRaisesRegex(ValueError, "missing artifact"):
            self.run_saved()

    def test_malformed_unknown_and_duplicate_terminal_json_fail_closed(self):
        self.fixture()
        for data in (b"{", b'{"schema":"bad","terminal":"unknown"}', b'{"terminal":1,"terminal":2}'):
            (self.output / "final-summary.json").write_bytes(data)
            with self.no_work(), self.assertRaises(ValueError):
                self.run_saved()

    def test_summary_cannot_drop_completed_phase_evidence(self):
        summary = self.fixture()
        summary["phases"].pop(0)
        self.write(self.output / "final-summary.json", summary)
        with self.no_work(), self.assertRaisesRegex(ValueError, "phase order"):
            self.run_saved()

    def test_rehashed_input_substitution_is_rejected(self):
        self.fixture()
        self.inputs["corpus"] = self.write(self.root / "other-corpus.json", {"different": True})
        self.write(self.inputs_path, self.seal(self.inputs))
        with self.no_work(), self.assertRaisesRegex(ValueError, "different inputs"):
            self.run_saved()

    def test_missing_summary_with_terminal_status_does_not_resume_training(self):
        self.write(self.output / "status.json", {"phase": "complete-no-candidate"})
        with self.no_work(), self.assertRaisesRegex(ValueError, "final-summary.json is missing"):
            self.run_saved()

    def test_rehashed_terminal_qualification_cannot_contradict_saved_gate(self):
        summary = self.fixture("qualified-local-research-incumbent")
        q = summary["qualification"]
        q["blind_holdout"] = self.write(self.output / "qualification/holdout.json", {"pass": False})
        summary["qualification"] = self.seal(q)
        self.write(self.output / "qualification/qualification.json", summary["qualification"])
        self.write(self.output / "final-summary.json", summary)
        with self.no_work(), self.assertRaisesRegex(ValueError, "contradicts"):
            self.run_saved()

    def test_explicit_replay_enters_original_numerical_path(self):
        self.fixture()
        with mock.patch.object(rebuild, "validate_rebuild_inputs", side_effect=RuntimeError("original path")) as load:
            with self.assertRaisesRegex(RuntimeError, "original path"):
                rebuild.run_ladder(inputs_manifest=self.inputs_path, output_directory=self.output,
                                   resume=True, replay_completed=True)
            load.assert_called_once_with(self.inputs_path)

    def test_incomplete_resume_still_uses_original_path(self):
        self.write(self.output / "status.json", {"phase": "training-v5-recovery"})
        with mock.patch.object(rebuild, "validate_rebuild_inputs", side_effect=RuntimeError("original path")):
            with self.assertRaisesRegex(RuntimeError, "original path"):
                self.run_saved()

    def test_completed_run_requires_resume_and_replay_requires_completion(self):
        with self.no_work(), self.assertRaisesRegex(ValueError, "requires a completed"):
            rebuild.run_ladder(inputs_manifest=self.inputs_path, output_directory=self.output,
                               resume=True, replay_completed=True)
        self.fixture()
        with self.no_work(), self.assertRaisesRegex(ValueError, "is complete"):
            rebuild.run_ladder(inputs_manifest=self.inputs_path, output_directory=self.output, resume=False)

    def test_cli_rejects_replay_without_resume_before_reading_inputs(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/jacek_replay_rebuild.py"),
            "run", "--inputs", str(self.inputs_path), "--output-directory", str(self.output),
            "--replay-completed"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--replay-completed requires --resume", result.stderr)
        self.assertFalse(self.output.exists())

    def test_completion_reader_does_not_import_numpy(self):
        self.fixture()
        code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                "from pathlib import Path;from jacek_replay_rebuild_completion import completed_result;"
                "completed_result(Path(sys.argv[2]),Path(sys.argv[3]));assert 'numpy' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools"),
                                 str(self.inputs_path), str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
