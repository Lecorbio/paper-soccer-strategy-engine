import json
import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_tuning_gate as gate  # noqa: E402


def baseline_receipt(model_sha256="a" * 64):
    entries = [
        {
            "round": index,
            "workflow_path": f"/canonical/round-{index}/workflow.json",
            "workflow_sha256": ("4" if index == 2 else str(5 + index)) * 64,
            "roots_path": f"/canonical/round-{index}/roots.json",
            "roots_sha256": "a" * 64,
            "pack_report_path": f"/canonical/round-{index}/pack.json",
            "pack_report_sha256": str(index + 1) * 64,
            "model_manifest_path": f"/canonical/round-{index}/model.json",
            "model_manifest_sha256": ("9" if index == 2 else "b") * 64,
            "runtime_path": f"/canonical/round-{index}/model.runtime",
            "runtime_sha256": model_sha256 if index == 2 else "c" * 64,
        }
        for index in range(3)
    ]
    return {
        "schema": gate.BASELINE_SCHEMA,
        "candidate_architecture": [6301, 192, 32, 1],
        "baseline_architecture": [1156, 32, 32, 1],
        "candidate_validation": {
            "weighted_huber": 0.1,
            "sign_accuracy": 0.8,
            "correlation": 0.5,
            "mae": 0.2,
            "prediction_mean": 0.0,
            "samples": 8,
        },
        "baseline_validation": {
            "weighted_huber": 0.2,
            "sign_accuracy": 0.7,
            "correlation": 0.4,
            "mae": 0.3,
            "prediction_mean": 0.0,
            "samples": 8,
        },
        "advance_to_game_gates": True,
        "bindings": {
            "candidate_manifest_sha256": "9" * 64,
            "candidate_runtime_sha256": model_sha256,
            "pack_report_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "workflow_receipt_sha256": "4" * 64,
            "canonical_workflow_entries": entries,
            "source_shards": [
                ["train", "5" * 64],
                ["validation", "6" * 64],
                ["test", "7" * 64],
            ],
        },
        "producer": {
            "baseline_sha256": "8" * 64,
            "trainer_sha256": "9" * 64,
            "features_sha256": "a" * 64,
            "workflow_sha256": "b" * 64,
        },
        "baseline_artifact": {
            "path": "/canonical/baseline.runtime",
            "architecture": [1156, 32, 32, 1],
            "feature_schema": gate.baseline_gate.BASELINE_FEATURE_SCHEMA,
            "feature_schema_sha256": (
                gate.baseline_gate.BASELINE_FEATURE_SCHEMA_HASH.hex()
            ),
            "diagnostic_only": True,
            "artifact_sha256": "d" * 64,
            "payload_sha256": "e" * 64,
            "bytes": 152320,
            "weight_count": gate.baseline_gate.BASELINE_WEIGHT_COUNT,
        },
        "selection": {
            **gate.baseline_gate.fixed_selection_contract(),
            "chosen_seed": 20260823,
            "seed_reports": [
                {
                    "seed": seed,
                    "validation": {
                        "weighted_huber": 0.2,
                        "sign_accuracy": 0.7,
                        "correlation": 0.4,
                        "mae": 0.3,
                        "prediction_mean": 0.0,
                        "samples": 8,
                    },
                }
                for seed in (20260823, 20260824, 20260825)
            ],
        },
    }


def file_backed_baseline(directory, model_path):
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    receipt = baseline_receipt(model_sha256)
    fields = (
        ("workflow_path", "workflow_sha256"),
        ("roots_path", "roots_sha256"),
        ("pack_report_path", "pack_report_sha256"),
        ("model_manifest_path", "model_manifest_sha256"),
        ("runtime_path", "runtime_sha256"),
    )
    for entry in receipt["bindings"]["canonical_workflow_entries"]:
        for path_field, hash_field in fields:
            path = directory / f"round-{entry['round']}-{path_field}"
            data = (
                model_path.read_bytes()
                if entry["round"] == 2 and path_field == "runtime_path"
                else f"{entry['round']}:{path_field}".encode()
            )
            path.write_bytes(data)
            entry[path_field] = str(path.resolve())
            entry[hash_field] = hashlib.sha256(data).hexdigest()
    entries = receipt["bindings"]["canonical_workflow_entries"]
    receipt["bindings"]["pack_report_sha256"] = [
        entry["pack_report_sha256"] for entry in entries
    ]
    receipt["bindings"]["workflow_receipt_sha256"] = entries[-1][
        "workflow_sha256"
    ]
    receipt["bindings"]["candidate_manifest_sha256"] = entries[-1][
        "model_manifest_sha256"
    ]
    return receipt


def report(exploration, wins):
    results = []
    color_targets = (wins // 2, wins - wins // 2)
    latency = 20.25 + exploration
    for color in (0, 1):
        color_index = 0
        for opponent in ("rank4", "neural-puct"):
            for opening in range(gate.DEVELOPMENT_PAIRS):
                candidate_won = color_index < color_targets[color]
                results.append(
                    {
                        "opening": f"opening-{opening}",
                        "opponent": opponent,
                        "candidate_player": color,
                        "winner": color if candidate_won else 1 - color,
                        "illegal": False,
                        "candidate_ms": [latency],
                    }
                )
                color_index += 1
    return {
        "schema": gate.REPORT_SCHEMA,
        "model": "candidate.runtime",
        "model_sha256": "a" * 64,
        "configuration": {
            "exploration": exploration,
            "opening_source": "development.tsv",
            "opening_bank_sha256": "b" * 64,
            "baseline_receipt_sha256": "1" * 64,
            "rank4_control_sha256": "c" * 64,
            "rank4_engine_sha256": "2" * 64,
            "neural_puct_control_sha256": "d" * 64,
            "neural_puct_engine_sha256": "3" * 64,
            "rank4_adapter_sha256": "e" * 64,
            "neural_puct_adapter_sha256": "f" * 64,
            "shared_core_sha256": "5" * 64,
            "candidate_source_sha256": "6" * 64,
            "comparison_source_sha256": "7" * 64,
            "comparison_executable_path": "comparison",
            "comparison_executable_sha256": "8" * 64,
            "pairs": gate.DEVELOPMENT_PAIRS,
            "time_ms": gate.DEVELOPMENT_TIME_MS,
            "max_turns": gate.DEVELOPMENT_MAX_TURNS,
            "opening_plies": gate.DEVELOPMENT_OPENING_PLIES,
            "opening_bank_seed": gate.DEVELOPMENT_BANK_SEED,
            "opening_bank_minimum_physical_plies":
                gate.DEVELOPMENT_OPENING_PLIES,
            "opponent": "both",
            "seed": 17,
            "candidate_tree_nodes": 1_000_000,
            "control_tree_nodes": 100_000,
            "control_work": 3_000_000,
            "max_actions": 250,
            "max_partial_paths": 50_000,
            "fpu": 0.5,
            "single_thread": True,
            "opening_bank_classification": "development",
            "opening_state_identities": [
                f"state-{index}" for index in range(gate.DEVELOPMENT_PAIRS)
            ],
        },
        "summary": {
            "games": gate.DEVELOPMENT_PAIRS * 4,
            "wins": wins,
            "losses": gate.DEVELOPMENT_PAIRS * 4 - wins,
            "illegal": 0,
            "unfinished": 0,
            "colors": [
                {"games": gate.DEVELOPMENT_PAIRS * 2, "wins": wins // 2},
                {
                    "games": gate.DEVELOPMENT_PAIRS * 2,
                    "wins": wins - wins // 2,
                },
            ],
            "candidate": {
                "p99_ms": 20.25 + exploration,
                "max_ms": 20.5 + exploration,
            },
        },
        "results": results,
    }


@unittest.skipIf(
    gate.baseline_gate is None,
    "research tests require requirements-research.txt",
)
class JacekReplayTuningTests(unittest.TestCase):
    def test_development_grid_selects_wins_then_color_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            baseline_path = directory / "baseline.json"
            baseline_path.write_text(json.dumps(baseline_receipt()))
            baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            paths = []
            for exploration, wins in ((0.25, 20), (0.5, 24), (0.95, 22)):
                path = directory / f"{exploration}.json"
                payload = report(exploration, wins)
                payload["configuration"]["baseline_receipt_sha256"] = baseline_hash
                path.write_text(json.dumps(payload))
                paths.append(path)
            receipt = gate.select(paths, baseline_path, verify_files=False)
            self.assertEqual(receipt["chosen_exploration"], 0.5)
            self.assertEqual(set(receipt["grid"]), set(gate.GRID))

    def test_mismatched_development_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            baseline_path = directory / "baseline.json"
            baseline_path.write_text(json.dumps(baseline_receipt()))
            baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            paths = []
            for exploration in gate.GRID:
                payload = report(exploration, 20)
                payload["configuration"]["baseline_receipt_sha256"] = baseline_hash
                if exploration == 0.95:
                    payload["model_sha256"] = "0" * 64
                path = directory / f"{exploration}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "exact inputs"):
                gate.select(paths, baseline_path, verify_files=False)

    def test_frozen_development_panel_parameters_are_mandatory(self):
        for field, value in (
            ("pairs", gate.DEVELOPMENT_PAIRS - 1),
            ("time_ms", gate.DEVELOPMENT_TIME_MS + 1),
            ("max_turns", gate.DEVELOPMENT_MAX_TURNS - 1),
            ("opening_plies", gate.DEVELOPMENT_OPENING_PLIES - 1),
            ("opening_bank_seed", gate.DEVELOPMENT_BANK_SEED + 1),
            ("opening_bank_minimum_physical_plies", 0),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = pathlib.Path(temporary)
                baseline_path = directory / "baseline.json"
                baseline_path.write_text(json.dumps(baseline_receipt()))
                baseline_hash = hashlib.sha256(
                    baseline_path.read_bytes()
                ).hexdigest()
                paths = []
                for exploration in gate.GRID:
                    payload = report(exploration, 20)
                    payload["configuration"]["baseline_receipt_sha256"] = (
                        baseline_hash
                    )
                    payload["configuration"][field] = value
                    path = directory / f"{exploration}.json"
                    path.write_text(json.dumps(payload))
                    paths.append(path)
                with self.assertRaises(ValueError):
                    gate.select(paths, baseline_path, verify_files=False)

    def test_per_game_outcomes_and_timings_are_not_trusted_to_summary(self):
        mutations = (
            ("illegal", True),
            ("winner", None),
            ("candidate_ms", [float("nan")]),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = pathlib.Path(temporary)
                baseline_path = directory / "baseline.json"
                baseline_path.write_text(json.dumps(baseline_receipt()))
                baseline_hash = hashlib.sha256(
                    baseline_path.read_bytes()
                ).hexdigest()
                paths = []
                for exploration in gate.GRID:
                    payload = report(exploration, 20)
                    payload["configuration"]["baseline_receipt_sha256"] = (
                        baseline_hash
                    )
                    payload["results"][0][field] = value
                    path = directory / f"{exploration}.json"
                    path.write_text(json.dumps(payload))
                    paths.append(path)
                with self.assertRaises(ValueError):
                    gate.select(paths, baseline_path, verify_files=False)

    def test_baseline_pass_flag_cannot_override_worse_metrics(self):
        receipt = baseline_receipt()
        receipt["candidate_validation"]["weighted_huber"] = 0.3
        with self.assertRaisesRegex(ValueError, "metric gate did not pass"):
            gate.validate_baseline_receipt(receipt, "a" * 64)

        receipt = baseline_receipt()
        del receipt["bindings"]["workflow_receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "canonical bindings"):
            gate.validate_baseline_receipt(receipt, "a" * 64)

    def test_real_model_bank_control_and_executable_files_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            model = directory / "candidate.runtime"
            model.write_bytes(b"candidate")
            executable = directory / "comparison"
            executable.write_bytes(b"comparison")
            bank = directory / "development.tsv"
            bank.write_text(
                "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
                "# rules=8x10;own-goals-allowed;mover-loses\n"
                "# classification=development\n"
                f"# seed={gate.DEVELOPMENT_BANK_SEED}\n"
                f"# minimum-physical-plies={gate.DEVELOPMENT_OPENING_PLIES}\n"
                "opening_id\ttranscript\tstate_identity\n"
                + "".join(
                    f"opening-{index}\ttranscript-{index}\tstate-{index}\n"
                    for index in range(gate.DEVELOPMENT_PAIRS)
                )
            )
            baseline_path = directory / "baseline.json"
            model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
            baseline_path.write_text(
                json.dumps(file_backed_baseline(directory, model))
            )
            baseline_hash = hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest()
            import jacek_replay_provenance as provenance

            paths = []
            for exploration in gate.GRID:
                payload = report(exploration, 20)
                payload["model"] = str(model)
                payload["model_sha256"] = model_hash
                configuration = payload["configuration"]
                configuration["opening_source"] = str(bank)
                configuration["opening_bank_sha256"] = hashlib.sha256(
                    bank.read_bytes()
                ).hexdigest()
                configuration["baseline_receipt_sha256"] = baseline_hash
                configuration.update(provenance.control_source_sha256())
                configuration["shared_core_sha256"] = (
                    provenance.shared_core_sha256()
                )
                configuration["candidate_source_sha256"] = (
                    provenance.candidate_source_sha256()
                )
                configuration["comparison_source_sha256"] = (
                    provenance.comparison_source_sha256()
                )
                configuration["comparison_executable_path"] = str(executable)
                configuration["comparison_executable_sha256"] = hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest()
                path = directory / f"{exploration}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
            real_validate = gate.baseline_gate.validate_receipt

            def fixture_validate(receipt, model_sha256, *, verify_files=False):
                real_validate(receipt, model_sha256, verify_files=False)
                if verify_files:
                    fields = (
                        ("workflow_path", "workflow_sha256"),
                        ("roots_path", "roots_sha256"),
                        ("pack_report_path", "pack_report_sha256"),
                        ("model_manifest_path", "model_manifest_sha256"),
                        ("runtime_path", "runtime_sha256"),
                    )
                    for entry in receipt["bindings"]["canonical_workflow_entries"]:
                        for path_field, hash_field in fields:
                            if hashlib.sha256(
                                pathlib.Path(entry[path_field]).read_bytes()
                            ).hexdigest() != entry[hash_field]:
                                raise ValueError(
                                    f"matched-baseline {path_field} artifact is stale"
                                )

            with mock.patch.object(
                gate.baseline_gate, "validate_receipt", side_effect=fixture_validate
            ):
                receipt = gate.select(paths, baseline_path)
            self.assertEqual(receipt["chosen_exploration"], 0.25)

            baseline_payload = json.loads(baseline_path.read_bytes())
            bound_roots = pathlib.Path(
                baseline_payload["bindings"]["canonical_workflow_entries"][0][
                    "roots_path"
                ]
            )
            original_roots = bound_roots.read_bytes()
            bound_roots.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact is stale"):
                with mock.patch.object(
                    gate.baseline_gate,
                    "validate_receipt",
                    side_effect=fixture_validate,
                ):
                    gate.select(paths, baseline_path)
            bound_roots.write_bytes(original_roots)

            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "model changed"):
                with mock.patch.object(
                    gate.baseline_gate,
                    "validate_receipt",
                    side_effect=fixture_validate,
                ):
                    gate.select(paths, baseline_path)


if __name__ == "__main__":
    unittest.main()
