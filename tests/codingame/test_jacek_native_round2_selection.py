import argparse
import hashlib
import importlib.util
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np  # noqa: F401
except ModuleNotFoundError:
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "round-two selection tests require NumPy")
class JacekNativeRound2SelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = load_module(
            "jacek_native_round2_selection_under_test",
            TOOLS / "jacek_native_round2_selection.py",
        )
        cls.trainer = load_module(
            "jacek_native_round2_selection_trainer_under_test",
            TOOLS / "train_jacek_native_round2.py",
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temporary.name)
        self.model_path = self.directory / "round2.json"
        self.baseline_model = ROOT / "models/jacek_native_bootstrap_model.json"
        self.baseline_seed = 20260813
        self.baseline_runtime = self.directory / "baseline.runtime"
        baseline_raw = self.baseline_model.read_bytes()
        baseline_model = json.loads(baseline_raw)
        self.baseline_runtime.write_text(
            self.selection.round1_exporter.render_runtime(
                baseline_model, hashlib.sha256(baseline_raw).hexdigest(),
                self.baseline_seed,
            )
        )
        self.gate_binary = self.directory / "gate"
        self.gate_binary.write_text("#!/bin/sh\nexit 99\n")
        self.gate_binary.chmod(
            self.gate_binary.stat().st_mode | stat.S_IXUSR
        )
        self.reports = self.directory / "reports"
        self.output = self.directory / "selection.json"
        self.seeds = [101, 102]
        self.model = self._model()
        self._write_model()
        self.candidate_runtimes = {}
        self._write_runtimes()

    def tearDown(self):
        self.temporary.cleanup()

    def _model(self):
        candidates = [self.trainer.initialize(seed) for seed in self.seeds]
        reports = [{
            "seed": seed,
            "quantized_metrics": {
                "validation": {
                    "outcome_mse": loss,
                    "combined_target_mse": loss,
                },
            },
        } for seed, loss in zip(self.seeds, (0.2, 0.3))]
        source_digest = "a" * 64
        sources = {f"sha256:{source_digest}": source_digest}
        artifact = {
            "artifact_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "packed_sha256": "d" * 64,
        }
        corpus_report = {
            "source_sha256": sources,
            "corpus_sha256": hashlib.sha256(json.dumps(
                sorted(sources.items()), separators=(",", ":")
            ).encode()).hexdigest(),
            "corpus_validator_sha256": hashlib.sha256(
                pathlib.Path(
                    self.trainer.corpus_contract.__file__
                ).read_bytes()
            ).hexdigest(),
            "restart_corpus_validator_sha256": hashlib.sha256(
                pathlib.Path(
                    self.trainer.restart_contract.__file__
                ).read_bytes()
            ).hexdigest(),
            "augmentation": {
                "reflection": True,
                "rotation": "player-two-canonicalization-in-feature-encoder",
                "grouping": "whole-game-before-augmentation",
            },
            "lineage": {
                "strict_current": [{
                    "manifest_sha256": "1" * 64,
                    "build_provenance_sha256": "2" * 64,
                    "binary_sha256": "3" * 64,
                    "shard_sha256": ["4" * 64],
                    "games": 8,
                    "seed": 17,
                }],
                "archived_round1": [],
                "live_restart_round2": [],
            },
            "generation": {
                "checkpoint_provenance": {
                    "mode": "native-runtime-models/v1",
                    "artifacts": [artifact],
                },
                "model_artifact_sha256": [artifact["artifact_sha256"]],
            },
        }
        arguments = argparse.Namespace(
            auxiliary_weight=0.25,
            batch_size=256,
            epochs=50,
            patience=8,
            learning_rate=0.001,
            weight_decay=1e-5,
            qat_epochs=4,
        )
        return self.trainer.build_report(
            candidates, reports, corpus_report, arguments
        )

    def _write_model(self):
        self.model_path.write_bytes(
            self.selection.canonical_json_bytes(self.model)
        )

    def _write_runtimes(self):
        raw = self.model_path.read_bytes()
        model_sha = hashlib.sha256(raw).hexdigest()
        for seed in self.seeds:
            path = self.directory / f"candidate-{seed}.runtime"
            path.write_text(
                self.selection.round2_exporter.render_runtime(
                    self.model, model_sha, seed
                )
            )
            self.candidate_runtimes[seed] = path

    def _stdout(self, profile_name, seed, player_one_wins, player_two_wins,
                headroom_failures=0, baseline_headroom_failures=0,
                candidate_operational_timeouts=0,
                baseline_operational_timeouts=0):
        profile = self.selection.PROFILES[profile_name]
        candidate, _ = self.selection._round2_identity(
            self.model_path, seed, self.candidate_runtimes[seed]
        )
        baseline = self.selection._round1_identity(
            self.baseline_model, self.baseline_seed, self.baseline_runtime
        )
        lines = [
            " ".join(
                f"candidate_{field}={candidate[field]}"
                for field in (
                    "runtime_sha256", "model_sha256", "packed_sha256"
                )
            ),
            " ".join(
                f"baseline_{field}={baseline[field]}"
                for field in (
                    "runtime_sha256", "model_sha256", "packed_sha256"
                )
            ),
        ]
        for pair in range(profile.pairs):
            winner_zero = 0 if pair < player_one_wins else 1
            winner_one = 1 if pair < player_two_wins else 0
            lines.append(
                f"pair={pair} "
                f"opening_turns={self.selection.OPENING_TURNS[pair % 4]} "
                f"seed={100000 + pair} c0={winner_zero} c1={winner_one}"
            )
        candidate_wins = player_one_wins + player_two_wins
        baseline_wins = profile.pairs * 2 - candidate_wins
        passed = (
            candidate_wins >= profile.minimum_candidate_wins
            and player_one_wins >= profile.minimum_wins_per_color
            and player_two_wins >= profile.minimum_wins_per_color
            and headroom_failures == 0
            and baseline_headroom_failures == 0
            and candidate_operational_timeouts == 0
            and baseline_operational_timeouts == 0
        )
        fields = {
            "candidate": candidate_wins,
            "baseline": baseline_wins,
            "unfinished": 0,
            "candidate_player_one": player_one_wins,
            "candidate_player_two": player_two_wins,
            "games": profile.pairs * 2,
            "candidate_decisions": profile.pairs * 10,
            "candidate_expansions": profile.pairs * 100,
            "candidate_child_evaluations": profile.pairs * 1000,
            "candidate_max_tree": 1780,
            "candidate_ms": 1234.5,
            "candidate_max_first_ms": 49.0,
            "candidate_max_later_ms": 9.0,
            "candidate_deadline_searches": 12,
            "candidate_headroom_failures": headroom_failures,
            "candidate_operational_timeouts": candidate_operational_timeouts,
            "baseline_decisions": profile.pairs * 10,
            "baseline_expansions": profile.pairs * 100,
            "baseline_max_first_ms": 49.0,
            "baseline_max_later_ms": 9.0,
            "baseline_headroom_failures": baseline_headroom_failures,
            "baseline_operational_timeouts": baseline_operational_timeouts,
            "profile": f"{profile.first_ms}/{profile.later_ms}",
            "shuffle_seed_policy": "deployment-constant",
            "required_total": profile.minimum_candidate_wins,
            "required_per_color": profile.minimum_wins_per_color,
            "passed": str(passed).lower(),
        }
        lines.append(
            "summary " + " ".join(f"{key}={value}" for key, value in fields.items())
        )
        return ("\n".join(lines) + "\n").encode(), 0 if passed else 1

    def _record(self, profile, seed, player_one_wins, player_two_wins,
                headroom_failures=0, baseline_headroom_failures=0,
                candidate_operational_timeouts=0,
                baseline_operational_timeouts=0):
        stdout, returncode = self._stdout(
            profile, seed, player_one_wins, player_two_wins,
            headroom_failures, baseline_headroom_failures,
            candidate_operational_timeouts, baseline_operational_timeouts,
        )
        completed = subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=b""
        )
        with mock.patch.object(
            self.selection.subprocess, "run", return_value=completed
        ) as run:
            path = self.selection.record_gate(
                profile_name=profile,
                model_path=self.model_path,
                seed=seed,
                candidate_runtime=self.candidate_runtimes[seed],
                baseline_model=self.baseline_model,
                baseline_seed=self.baseline_seed,
                baseline_runtime=self.baseline_runtime,
                gate_binary=self.gate_binary,
                output_dir=self.reports,
            )
        command = run.call_args.args[0]
        self.assertNotIn("--vary-shuffle-seed", command)
        self.assertEqual(
            command[command.index("--seed") + 1],
            str(self.selection.OPENING_SEED),
        )
        return path

    def _all_reports(self, outcomes=None):
        outcomes = outcomes or {
            101: {"screen": (270, 270), "decisive": (56, 56)},
            102: {"screen": (275, 275), "decisive": (57, 56)},
        }
        paths = []
        for seed in self.seeds:
            for profile in ("screen", "decisive"):
                paths.append(self._record(
                    profile, seed, *outcomes[seed][profile]
                ))
        return paths

    def _finalize(self, paths):
        return self.selection.finalize_selection(
            model_path=self.model_path,
            baseline_model=self.baseline_model,
            baseline_seed=self.baseline_seed,
            baseline_runtime=self.baseline_runtime,
            report_paths=paths,
            output=self.output,
        )

    def test_profiles_are_frozen_and_training_model_has_no_chosen_seed(self):
        self.assertIsNone(self.model["training"]["chosen_seed"])
        self.assertEqual(
            self.selection.PROFILES["screen"].payload(),
            {
                "name": "screen", "pairs": 500, "games": 1000,
                "first_ms": 50, "later_ms": 10, "maximum_turns": 384,
                "opening_turns": [0, 4, 8, 12],
                "opening_seed": "6517766227279252335",
                "shuffle_seed_policy": "deployment-constant",
                "minimum_candidate_wins": 530,
                "minimum_wins_per_color": 0,
                "require_zero_unfinished": True,
                "require_zero_headroom_failures": True,
                "require_zero_operational_timeouts": True,
            },
        )
        self.assertEqual(self.selection.PROFILES["decisive"].later_ms, 155)

    def test_finalize_selects_deterministically_and_binds_exact_runtime(self):
        paths = self._all_reports()
        sidecar = self._finalize(list(reversed(paths)))
        self.assertEqual(sidecar["selected"]["seed"], 102)
        self.assertTrue(sidecar["selected"]["exact_tested_deployed_runtime"])
        self.assertEqual(
            sidecar["selected"]["tested_runtime_sha256"],
            sidecar["selected"]["deployment_runtime_sha256"],
        )
        deployed = self.selection.round2_exporter.render_runtime(
            self.model,
            hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            sidecar["selected"]["seed"],
        ).encode()
        self.assertEqual(
            hashlib.sha256(deployed).hexdigest(),
            sidecar["selected"]["tested_runtime_sha256"],
        )
        written = json.loads(self.output.read_text())
        self.assertEqual(
            self.selection._selection_payload_hash(written),
            written["selection_payload_sha256"],
        )
        second_output = self.directory / "selection-second.json"
        self.selection.finalize_selection(
            model_path=self.model_path,
            baseline_model=self.baseline_model,
            baseline_seed=self.baseline_seed,
            baseline_runtime=self.baseline_runtime,
            report_paths=paths,
            output=second_output,
        )
        self.assertEqual(self.output.read_bytes(), second_output.read_bytes())

    def test_missing_seed_report_is_rejected(self):
        paths = self._all_reports()
        with self.assertRaisesRegex(self.selection.SelectionError, "coverage"):
            self._finalize(paths[:-1])

    def test_mixed_gate_binary_identities_are_rejected(self):
        paths = [
            self._record("screen", 101, 270, 270),
            self._record("decisive", 101, 56, 56),
        ]
        self.gate_binary.write_text("#!/bin/sh\nexit 98\n")
        paths.extend((
            self._record("screen", 102, 275, 275),
            self._record("decisive", 102, 57, 56),
        ))
        with self.assertRaisesRegex(
            self.selection.SelectionError, "one exact gate binary"
        ):
            self._finalize(paths)

    def test_no_seed_passing_both_thresholds_is_rejected(self):
        paths = self._all_reports({
            seed: {"screen": (260, 260), "decisive": (49, 49)}
            for seed in self.seeds
        })
        with self.assertRaisesRegex(self.selection.SelectionError, "no retained seed"):
            self._finalize(paths)

    def test_exploratory_finalize_selects_strongest_clean_failed_seed(self):
        paths = self._all_reports({
            101: {"screen": (260, 260), "decisive": (54, 54)},
            102: {"screen": (263, 262), "decisive": (55, 55)},
        })
        sidecar = self.selection.finalize_exploratory_selection(
            model_path=self.model_path,
            baseline_model=self.baseline_model,
            baseline_seed=self.baseline_seed,
            baseline_runtime=self.baseline_runtime,
            report_paths=list(reversed(paths)),
            output=self.output,
        )
        self.assertEqual(sidecar["selected"]["seed"], 102)
        self.assertEqual(sidecar["decision"]["kind"], "exploratory")
        self.assertFalse(sidecar["decision"]["promotion_eligible"])
        self.assertTrue(sidecar["decision"]["threshold_shortfalls"])
        self.assertEqual(
            sidecar["ranking"]["operationally_safe_failed_seeds"][0][
                "seed"
            ],
            102,
        )

    def test_exploratory_finalize_refuses_passing_or_unsafe_only_evidence(self):
        passing = self._all_reports()
        with self.assertRaisesRegex(
            self.selection.SelectionError, "canonical promotion"
        ):
            self.selection.finalize_exploratory_selection(
                model_path=self.model_path,
                baseline_model=self.baseline_model,
                baseline_seed=self.baseline_seed,
                baseline_runtime=self.baseline_runtime,
                report_paths=passing,
                output=self.output,
            )

        self.output = self.directory / "unsafe-selection.json"
        self.reports = self.directory / "unsafe-reports"
        unsafe = []
        for seed in self.seeds:
            unsafe.append(self._record("screen", seed, 260, 260))
            unsafe.append(self._record(
                "decisive", seed, 55, 55,
                baseline_headroom_failures=1,
            ))
        with self.assertRaisesRegex(
            self.selection.SelectionError, "operationally clean"
        ):
            self.selection.finalize_exploratory_selection(
                model_path=self.model_path,
                baseline_model=self.baseline_model,
                baseline_seed=self.baseline_seed,
                baseline_runtime=self.baseline_runtime,
                report_paths=unsafe,
                output=self.output,
            )

    def test_exploratory_ranking_excludes_stronger_unsafe_seed(self):
        paths = [
            self._record("screen", 101, 260, 260),
            self._record("decisive", 101, 54, 54),
            self._record("screen", 102, 264, 264),
            self._record(
                "decisive", 102, 55, 56,
                candidate_operational_timeouts=1,
            ),
        ]
        sidecar = self.selection.finalize_exploratory_selection(
            model_path=self.model_path,
            baseline_model=self.baseline_model,
            baseline_seed=self.baseline_seed,
            baseline_runtime=self.baseline_runtime,
            report_paths=paths,
            output=self.output,
        )
        self.assertEqual(sidecar["selected"]["seed"], 101)
        self.assertEqual(
            [row["seed"] for row in sidecar["ranking"][
                "operationally_safe_failed_seeds"
            ]],
            [101],
        )

    def test_tampered_full_stdout_is_rejected(self):
        paths = self._all_reports()
        report = json.loads(paths[0].read_text())
        stdout_path = paths[0].parent / report["stdout"]["path"]
        stdout_path.write_bytes(stdout_path.read_bytes().replace(
            b" c0=0 c1=1\n", b" c0=1 c1=1\n", 1
        ))
        with self.assertRaisesRegex(self.selection.SelectionError, "stdout bytes"):
            self._finalize(paths)

    def test_rehashed_and_renamed_report_tampering_is_rejected(self):
        paths = self._all_reports()
        report = json.loads(paths[0].read_text())
        report["result"]["candidate"] += 1
        raw = self.selection.canonical_json_bytes(report)
        tampered = paths[0].with_name(f"{hashlib.sha256(raw).hexdigest()}.json")
        tampered.write_bytes(raw)
        paths[0] = tampered
        with self.assertRaisesRegex(self.selection.SelectionError, "full stdout"):
            self._finalize(paths)

    def test_record_refuses_overwrite(self):
        self._record("screen", 101, 270, 270)
        with self.assertRaisesRegex(self.selection.SelectionError, "overwrite"):
            self._record("screen", 101, 270, 270)

    def test_mutated_training_model_invalidates_runtime_and_reports(self):
        paths = self._all_reports()
        self.model["training"]["chosen_seed"] = 101
        self._write_model()
        with self.assertRaisesRegex(
            self.selection.SelectionError, "immutable pending"
        ):
            self._finalize(paths)

    def test_headroom_failure_cannot_pass_even_with_score(self):
        path = self._record("screen", 101, 300, 300, headroom_failures=1)
        report = json.loads(path.read_text())
        self.assertFalse(report["result"]["passed"])


if __name__ == "__main__":
    unittest.main()
