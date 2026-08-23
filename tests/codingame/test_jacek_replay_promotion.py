import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_promotion_gate as gate  # noqa: E402


def opponent_games(opponent, color_wins):
    games = []
    for color, wins in enumerate(color_wins):
        for index in range(500):
            candidate_won = index < wins
            games.append(
                {
                    "opening": f"opening-{index:03d}",
                    "opponent": opponent,
                    "candidate_player": color,
                    "winner": color if candidate_won else 1 - color,
                    "illegal": False,
                    "candidate_ms": [999.0],
                }
            )
    return games


def passing_payload():
    return {
        "schema": gate.SCHEMA,
        "model": "candidate.runtime",
        "model_sha256": "a" * 64,
        "configuration": {
            "pairs": 500,
            "opening_source": "frozen-openings.tsv",
            "opening_bank_sha256": "b" * 64,
            "opening_bank_classification": "final",
            "opening_state_identities": [
                f"final-state-{index}" for index in range(500)
            ],
            "tuning_receipt_path": "tuning.json",
            "tuning_receipt_sha256": "1" * 64,
            "baseline_receipt_path": "baseline.json",
            "baseline_receipt_sha256": "4" * 64,
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
            "time_ms": 980,
            "max_turns": gate.FINAL_MAX_TURNS,
            "opening_plies": gate.FINAL_OPENING_PLIES,
            "opening_bank_seed": gate.FINAL_BANK_SEED,
            "opening_bank_minimum_physical_plies": gate.FINAL_OPENING_PLIES,
            "opponent": "both",
            "seed": 17,
            "candidate_tree_nodes": 1_000_000,
            "control_tree_nodes": 100_000,
            "control_work": 3_000_000,
            "max_actions": 250,
            "max_partial_paths": 50_000,
            "exploration": 0.95,
            "fpu": 0.5,
            "single_thread": True,
        },
        "summary": {
            "games": 2_000,
            "wins": 1_060,
            "losses": 940,
            "illegal": 0,
            "unfinished": 0,
            "colors": [
                {"games": 1_000, "wins": 530},
                {"games": 1_000, "wins": 530},
            ],
            "candidate": {"max_ms": 999.0},
        },
        "results": (
            opponent_games("neural-puct", (270, 270))
            + opponent_games("rank4", (260, 260))
        ),
    }


def tuning_receipt():
    configuration = passing_payload()["configuration"]
    return {
        "schema": gate.TUNING_SCHEMA,
        "classification": "development-only-exploration-selection",
        "chosen_exploration": configuration["exploration"],
        "grid": list(gate.tuning_gate.GRID),
        "reports": {
            str(value): {
                "path": f"development-{value}.json",
                "sha256": hashlib.sha256(str(value).encode()).hexdigest(),
                "wins": 100 + index,
                "minimum_color_wins": 50 + index,
                "p99_ms": 19.0,
            }
            for index, value in enumerate(gate.tuning_gate.GRID)
        },
        "opening_transcript_sha256": [
            hashlib.sha256(f"development-{index}".encode()).hexdigest()
            for index in range(gate.DEVELOPMENT_PAIRS)
        ],
        "binding": {
            "model_sha256": "a" * 64,
            "opening_bank_sha256": "0" * 64,
            "opening_bank_classification": "development",
            "opening_state_identities": [
                f"development-state-{index}"
                for index in range(gate.DEVELOPMENT_PAIRS)
            ],
            "baseline_receipt_sha256": configuration[
                "baseline_receipt_sha256"
            ],
            "rank4_control_sha256": configuration["rank4_control_sha256"],
            "rank4_engine_sha256": configuration["rank4_engine_sha256"],
            "neural_puct_control_sha256": configuration[
                "neural_puct_control_sha256"
            ],
            "neural_puct_engine_sha256": configuration[
                "neural_puct_engine_sha256"
            ],
            "rank4_adapter_sha256": configuration["rank4_adapter_sha256"],
            "neural_puct_adapter_sha256": configuration[
                "neural_puct_adapter_sha256"
            ],
            "shared_core_sha256": configuration["shared_core_sha256"],
            "candidate_source_sha256": configuration[
                "candidate_source_sha256"
            ],
            "comparison_source_sha256": configuration[
                "comparison_source_sha256"
            ],
            "comparison_executable_path": configuration[
                "comparison_executable_path"
            ],
            "comparison_executable_sha256": configuration[
                "comparison_executable_sha256"
            ],
            "opening_plies": configuration["opening_plies"],
            "pairs": gate.DEVELOPMENT_PAIRS,
            "time_ms": gate.DEVELOPMENT_TIME_MS,
            "opening_bank_seed": gate.DEVELOPMENT_BANK_SEED,
            "opening_bank_minimum_physical_plies": gate.FINAL_OPENING_PLIES,
            "max_turns": gate.FINAL_MAX_TURNS,
            **{
                field: configuration[field]
                for field in (
                    "seed",
                    "candidate_tree_nodes",
                    "control_tree_nodes",
                    "control_work",
                    "max_actions",
                    "max_partial_paths",
                    "fpu",
                    "single_thread",
                    "opponent",
                )
            },
        },
    }


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
            "feature_schema": (
                gate.tuning_gate.baseline_gate.BASELINE_FEATURE_SCHEMA
            ),
            "feature_schema_sha256": (
                gate.tuning_gate.baseline_gate.BASELINE_FEATURE_SCHEMA_HASH.hex()
            ),
            "diagnostic_only": True,
            "artifact_sha256": "d" * 64,
            "payload_sha256": "e" * 64,
            "bytes": 152320,
            "weight_count": gate.tuning_gate.baseline_gate.BASELINE_WEIGHT_COUNT,
        },
        "selection": {
            **gate.tuning_gate.baseline_gate.fixed_selection_contract(),
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


def refresh_summary(payload):
    results = payload["results"]
    wins = sum(
        game["winner"] == game["candidate_player"] for game in results
    )
    payload["summary"].update(
        {
            "games": len(results),
            "wins": wins,
            "losses": len(results) - wins,
            "illegal": sum(game["illegal"] is not False for game in results),
            "unfinished": sum(game["winner"] not in (0, 1) for game in results),
            "colors": [
                {
                    "games": sum(
                        game["candidate_player"] == color for game in results
                    ),
                    "wins": sum(
                        game["candidate_player"] == color
                        and game["winner"] == color
                        for game in results
                    ),
                }
                for color in (0, 1)
            ],
        }
    )


@unittest.skipIf(
    gate.tuning_gate.baseline_gate is None,
    "research tests require requirements-research.txt",
)
class JacekReplayPromotionTests(unittest.TestCase):
    def test_decision_binds_raw_report_bytes_and_consumed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            payload = passing_payload()
            report_path = directory / "final.json"
            report_path.write_text(json.dumps(payload))
            binding = {
                "model_sha256": payload["model_sha256"],
                "tuning_receipt_sha256": payload["configuration"][
                    "tuning_receipt_sha256"
                ],
                "baseline_receipt_sha256": payload["configuration"][
                    "baseline_receipt_sha256"
                ],
                "comparison_executable_sha256": payload["configuration"][
                    "comparison_executable_sha256"
                ],
                "opening_bank_path": str(
                    pathlib.Path(payload["configuration"]["opening_source"]).resolve()
                ),
                "comparison_report_path": str(report_path.resolve()),
                "chosen_exploration": payload["configuration"]["exploration"],
            }
            attempt = {
                "schema": gate.ATTEMPT_SCHEMA,
                "attempt_id": hashlib.sha256(
                    gate.canonical_json_bytes(binding)
                ).hexdigest(),
                "state": "consumed",
                "binding": binding,
                "bank_registration": {
                    "opening_bank_sha256": payload["configuration"][
                        "opening_bank_sha256"
                    ]
                },
                "report_consumption": {
                    "comparison_report_path": str(report_path.resolve()),
                    "comparison_report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                },
            }
            decision = gate.evaluate(
                payload,
                1_000,
                1_000.0,
                verify_files=False,
                tuning_receipt=tuning_receipt(),
                baseline_receipt=baseline_receipt(),
                report_path=report_path,
                attempt_ledger=attempt,
            )
            self.assertTrue(decision["eligible"], decision["errors"])
            self.assertEqual(
                decision["comparison_report"]["sha256"],
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            report_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs"):
                gate.evaluate(
                    payload,
                    1_000,
                    1_000.0,
                    verify_files=False,
                    tuning_receipt=tuning_receipt(),
                    baseline_receipt=baseline_receipt(),
                    report_path=report_path,
                    attempt_ledger=attempt,
                )

    def test_registered_bank_is_single_use_and_consumed_report_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            bank = directory / "final.tsv"
            bank.write_text(
                "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
                "# rules=8x10;own-goals-allowed;mover-loses\n"
                "# classification=final\n"
                f"# seed={gate.FINAL_BANK_SEED}\n"
                "# minimum-physical-plies=12\n"
                "opening_id\ttranscript\tstate_identity\n"
                + "".join(
                    f"opening-{index}\ttranscript-{index}\tstate-{index}\n"
                    for index in range(gate.FINAL_PAIRS)
                )
            )
            report = directory / "final.json"
            binding = {
                "opening_bank_path": str(bank.resolve()),
                "comparison_report_path": str(report.resolve()),
            }
            ledger = {
                "schema": gate.ATTEMPT_SCHEMA,
                "attempt_id": hashlib.sha256(
                    gate.canonical_json_bytes(binding)
                ).hexdigest(),
                "state": "prepared",
                "binding": binding,
                "prepared": {"at_utc": "2026-08-24T00:00:00Z"},
                "bank_registration": None,
                "report_consumption": None,
            }
            ledger_path = directory / "attempt.json"
            gate.atomic_json(ledger_path, ledger)
            with mock.patch.object(
                gate,
                "verify_live_attempt_binding",
                return_value=(
                    {
                        "binding": {"opening_state_identities": []},
                        "opening_transcript_sha256": [],
                    },
                    {},
                ),
            ):
                registered = gate.register_final_bank(ledger_path)
            self.assertEqual(registered["state"], "bank-registered")
            with self.assertRaisesRegex(ValueError, "only once"):
                gate.register_final_bank(ledger_path)

            report.write_text(json.dumps(passing_payload()))
            consumed = registered
            consumed["state"] = "consumed"
            consumed["report_consumption"] = {
                "comparison_report_path": str(report.resolve()),
                "comparison_report_sha256": hashlib.sha256(
                    report.read_bytes()
                ).hexdigest(),
            }
            gate.atomic_json(ledger_path, consumed)
            gate.consume_final_report(ledger_path)
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "replaced"):
                gate.consume_final_report(ledger_path)

    def test_registered_attempt_rejects_preexisting_report_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            report = directory / "final.json"
            report.write_text(json.dumps(passing_payload()))
            binding = {"comparison_report_path": str(report.resolve())}
            ledger = {
                "schema": gate.ATTEMPT_SCHEMA,
                "attempt_id": hashlib.sha256(
                    gate.canonical_json_bytes(binding)
                ).hexdigest(),
                "state": "bank-registered",
                "binding": binding,
                "prepared": {"at_utc": "2026-08-24T00:00:00Z"},
                "bank_registration": {},
                "report_consumption": None,
            }
            ledger_path = directory / "attempt.json"
            gate.atomic_json(ledger_path, ledger)
            with self.assertRaisesRegex(ValueError, "only after"):
                gate.consume_final_report(ledger_path)
            with mock.patch.object(
                gate, "verify_live_attempt_binding", return_value=({}, {})
            ), mock.patch.object(gate.subprocess, "run") as launched:
                with self.assertRaisesRegex(ValueError, "existed before"):
                    gate.run_final_attempt(ledger_path)
            launched.assert_not_called()
            failed, _ = gate.load_attempt(ledger_path)
            self.assertEqual(failed["state"], "failed-consumed")
            self.assertEqual(
                failed["report_consumption"]["reason"],
                "report-existed-before-bound-comparison-launch",
            )

    def test_publish_is_atomic_and_requires_eligible_recomputed_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            runtime = directory / "jacek_replay_bfm.runtime"
            runtime.write_bytes(b"round-two-runtime")
            runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest()
            candidate = directory / "jacek_replay_bfm.runtime.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema": "papersoccer.jacek-replay-bfm-model.v1",
                        "status": "canonical-campaign-candidate-not-game-gated",
                        "runtime": {
                            "path": runtime.name,
                            "artifact_sha256": runtime_hash,
                        },
                        "campaign_contract": {"eligible": True, "round": 2},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            baseline_artifact = directory / "baseline.runtime"
            baseline_artifact.write_bytes(b"baseline")
            baseline = directory / "baseline.json"
            candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            gate.atomic_json(
                baseline,
                {
                    "baseline_artifact": {"path": str(baseline_artifact)},
                    "bindings": {
                        "candidate_manifest_sha256": candidate_hash,
                        "canonical_workflow_entries": [
                            {"round": 0},
                            {"round": 1},
                            {
                                "round": 2,
                                "model_manifest_path": str(candidate.resolve()),
                                "model_manifest_sha256": candidate_hash,
                            },
                        ],
                    },
                },
            )
            tuning = directory / "tuning.json"
            gate.atomic_json(tuning, {"schema": gate.TUNING_SCHEMA})
            ledger = directory / "attempt.json"
            gate.atomic_json(ledger, {"schema": gate.ATTEMPT_SCHEMA})
            report = directory / "comparison.json"
            report.write_text(
                json.dumps({"model": str(runtime)}, sort_keys=True) + "\n"
            )
            decision_payload = {
                "schema": gate.DECISION_SCHEMA,
                "eligible": True,
                "errors": [],
                "model_sha256": runtime_hash,
            }
            decision = directory / "decision.json"
            gate.atomic_json(decision, decision_payload)
            output = directory / "promoted"
            with mock.patch.object(
                gate, "evaluate", return_value=decision_payload
            ):
                publication = gate.publish_promoted(
                    decision_path=decision,
                    candidate_manifest_path=candidate,
                    baseline_receipt_path=baseline,
                    tuning_receipt_path=tuning,
                    attempt_ledger_path=ledger,
                    report_path=report,
                    output_directory=output,
                )
            self.assertEqual(publication["status"], "promoted")
            self.assertEqual(
                (output / "jacek_replay_bfm.runtime").read_bytes(),
                runtime.read_bytes(),
            )
            self.assertTrue((output / "promotion-manifest.json").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                gate.publish_promoted(
                    decision_path=decision,
                    candidate_manifest_path=candidate,
                    baseline_receipt_path=baseline,
                    tuning_receipt_path=tuning,
                    attempt_ledger_path=ledger,
                    report_path=report,
                    output_directory=output,
                )

            fabricated = directory / "fabricated.runtime.json"
            fabricated_payload = json.loads(candidate.read_bytes())
            fabricated_payload["fabricated"] = True
            fabricated.write_text(
                json.dumps(
                    fabricated_payload, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            with mock.patch.object(
                gate, "evaluate", return_value=decision_payload
            ):
                with self.assertRaisesRegex(ValueError, "exact baseline-bound"):
                    gate.publish_promoted(
                        decision_path=decision,
                        candidate_manifest_path=fabricated,
                        baseline_receipt_path=baseline,
                        tuning_receipt_path=tuning,
                        attempt_ledger_path=ledger,
                        report_path=report,
                        output_directory=directory / "fabricated-output",
                    )

            failed = directory / "failed-decision.json"
            gate.atomic_json(
                failed,
                {
                    "schema": gate.DECISION_SCHEMA,
                    "eligible": False,
                    "errors": ["failed"],
                },
            )
            with self.assertRaisesRegex(ValueError, "eligible"):
                gate.publish_promoted(
                    decision_path=failed,
                    candidate_manifest_path=candidate,
                    baseline_receipt_path=baseline,
                    tuning_receipt_path=tuning,
                    attempt_ledger_path=ledger,
                    report_path=report,
                    output_directory=directory / "must-not-exist",
                )

    def test_opening_bank_metadata_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bank.tsv"
            path.write_text("opening_id\ttranscript\tstate_identity\na\t1\tx\n")
            with self.assertRaisesRegex(ValueError, "metadata contract"):
                gate.validate_bank_metadata(path, "final")

    def test_balanced_540_and_520_win_panels_pass(self):
        decision = gate.evaluate(
            passing_payload(),
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertTrue(decision["eligible"], decision["errors"])
        self.assertEqual(decision["opponents"]["neural-puct"]["wins"], 540)
        self.assertEqual(decision["opponents"]["rank4"]["wins"], 520)

    def test_exact_preregistered_win_and_color_floors_pass(self):
        payload = passing_payload()
        payload["results"] = (
            opponent_games("neural-puct", (267, 260))
            + opponent_games("rank4", (263, 238))
        )
        refresh_summary(payload)
        decision = gate.evaluate(
            payload,
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertTrue(decision["eligible"], decision["errors"])
        self.assertEqual(decision["opponents"]["neural-puct"]["wins"], 527)
        self.assertEqual(decision["opponents"]["rank4"]["wins"], 501)

    def test_summary_and_tuning_selection_tampering_fail_closed(self):
        payload = passing_payload()
        payload["summary"]["wins"] += 1
        decision = gate.evaluate(
            payload,
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertFalse(decision["eligible"])
        self.assertTrue(
            any("summary differs" in error for error in decision["errors"])
        )

        receipt = tuning_receipt()
        receipt["reports"]["0.25"]["wins"] = 1_000
        decision = gate.evaluate(
            passing_payload(),
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=receipt,
            baseline_receipt=baseline_receipt(),
        )
        self.assertFalse(decision["eligible"])
        self.assertTrue(
            any("selection differs" in error for error in decision["errors"])
        )

    def test_malformed_timing_is_a_schema_error(self):
        payload = passing_payload()
        payload["results"][0]["candidate_ms"] = "999"
        with self.assertRaisesRegex(ValueError, "candidate timings"):
            gate.evaluate(
                payload,
                1_000,
                1_000.0,
                verify_files=False,
                tuning_receipt=tuning_receipt(),
                baseline_receipt=baseline_receipt(),
            )

    def test_color_collapse_is_rejected_even_with_540_total_wins(self):
        payload = passing_payload()
        payload["results"] = (
            opponent_games("neural-puct", (300, 240))
            + opponent_games("rank4", (260, 260))
        )
        decision = gate.evaluate(
            payload,
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertFalse(decision["eligible"])
        self.assertTrue(any("color 1" in error for error in decision["errors"]))

    def test_illegal_game_and_latency_at_limit_are_rejected(self):
        payload = passing_payload()
        payload["results"][0]["illegal"] = True
        payload["summary"]["candidate"]["max_ms"] = 1_000.0
        decision = gate.evaluate(
            payload,
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertFalse(decision["eligible"])
        self.assertTrue(any("illegal" in error for error in decision["errors"]))
        self.assertTrue(any("latency" in error for error in decision["errors"]))

    def test_generated_or_duplicated_final_evidence_is_rejected(self):
        payload = passing_payload()
        payload["configuration"]["opening_source"] = "generated"
        payload["results"][1] = dict(payload["results"][0])
        decision = gate.evaluate(
            payload,
            1_000,
            1_000.0,
            verify_files=False,
            tuning_receipt=tuning_receipt(),
            baseline_receipt=baseline_receipt(),
        )
        self.assertFalse(decision["eligible"])
        self.assertTrue(
            any("frozen opening bank" in error for error in decision["errors"])
        )
        self.assertTrue(
            any("duplicate game identity" in error for error in decision["errors"])
        )

    def test_final_bank_and_candidate_provenance_are_frozen(self):
        for field, value, expected_error in (
            ("opening_plies", 11, "12-ply"),
            ("opening_bank_seed", 17, "final-bank seed"),
            ("max_turns", 319, "320-turn"),
            (
                "opening_bank_minimum_physical_plies",
                11,
                "12 physical plies",
            ),
            ("candidate_source_sha256", "9" * 64, "candidate_source_sha256"),
            (
                "comparison_executable_sha256",
                "9" * 64,
                "comparison_executable_sha256",
            ),
        ):
            with self.subTest(field=field):
                payload = passing_payload()
                payload["configuration"][field] = value
                decision = gate.evaluate(
                    payload,
                    1_000,
                    1_000.0,
                    verify_files=False,
                    tuning_receipt=tuning_receipt(),
                    baseline_receipt=baseline_receipt(),
                )
                self.assertFalse(decision["eligible"])
                self.assertTrue(
                    any(expected_error in error for error in decision["errors"]),
                    decision["errors"],
                )

    def test_real_file_bindings_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            bank = directory / "openings.tsv"
            bank.write_text(
                "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
                "# rules=8x10;own-goals-allowed;mover-loses\n"
                "# classification=final\n"
                f"# seed={gate.FINAL_BANK_SEED}\n"
                "# minimum-physical-plies=12\n"
                "opening_id\ttranscript\tstate_identity\n"
                + "".join(
                    f"opening-{index:03d}\ttranscript-{index:03d}"
                    f"\tfinal-state-{index}\n"
                    for index in range(500)
                )
            )
            model = directory / "candidate.runtime"
            model.write_bytes(b"candidate-model")
            payload = passing_payload()
            configuration = payload["configuration"]
            payload["model"] = str(model)
            payload["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
            configuration["opening_source"] = str(bank)
            configuration["opening_bank_sha256"] = hashlib.sha256(
                bank.read_bytes()
            ).hexdigest()
            source_paths = {
                "rank4_control_sha256": ROOT
                / "submissions/codingame/bots/rank_4/submission.cpp",
                "rank4_engine_sha256": ROOT
                / "submissions/codingame/bots/rank_4/bot.cpp",
                "neural_puct_control_sha256": ROOT
                / "submissions/codingame/bots/neural_puct/submission.cpp",
                "neural_puct_engine_sha256": ROOT
                / "submissions/codingame/bots/neural_puct/bot.cpp",
                "rank4_adapter_sha256": ROOT
                / "tools/jacek_replay_bfm_rank4_control.cpp",
                "neural_puct_adapter_sha256": ROOT
                / "tools/jacek_replay_bfm_neural_puct_control.cpp",
            }
            for field, path in source_paths.items():
                configuration[field] = hashlib.sha256(path.read_bytes()).hexdigest()
            configuration["shared_core_sha256"] = gate.shared_core_sha256()
            import jacek_replay_provenance as provenance

            configuration["candidate_source_sha256"] = (
                provenance.candidate_source_sha256()
            )
            configuration["comparison_source_sha256"] = (
                provenance.comparison_source_sha256()
            )
            executable = directory / "comparison"
            executable.write_bytes(b"comparison-executable")
            configuration["comparison_executable_path"] = str(executable)
            configuration["comparison_executable_sha256"] = hashlib.sha256(
                executable.read_bytes()
            ).hexdigest()
            baseline = file_backed_baseline(directory, model)
            baseline_path = directory / "baseline.json"
            baseline_path.write_text(json.dumps(baseline, sort_keys=True))
            configuration["baseline_receipt_path"] = str(baseline_path)
            configuration["baseline_receipt_sha256"] = hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest()

            development_bank = directory / "development.tsv"
            development_bank.write_text(
                "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
                "# rules=8x10;own-goals-allowed;mover-loses\n"
                "# classification=development\n"
                f"# seed={gate.DEVELOPMENT_BANK_SEED}\n"
                "# minimum-physical-plies=12\n"
                "opening_id\ttranscript\tstate_identity\n"
                + "".join(
                    f"opening-{index:03d}\tdevelopment-transcript-{index:03d}"
                    f"\tdevelopment-state-{index}\n"
                    for index in range(gate.DEVELOPMENT_PAIRS)
                )
            )
            development_bank_hash = hashlib.sha256(
                development_bank.read_bytes()
            ).hexdigest()
            report_paths = []
            for exploration, wins in zip(gate.tuning_gate.GRID, (600, 650, 700)):
                development_results = []
                wins_per_color = wins // 2
                for color in (0, 1):
                    color_index = 0
                    for opponent in ("rank4", "neural-puct"):
                        for opening in range(gate.DEVELOPMENT_PAIRS):
                            candidate_won = color_index < wins_per_color
                            development_results.append(
                                {
                                    "opening": f"opening-{opening:03d}",
                                    "opponent": opponent,
                                    "candidate_player": color,
                                    "winner": color if candidate_won else 1 - color,
                                    "illegal": False,
                                    "candidate_ms": [19.0],
                                }
                            )
                            color_index += 1
                development_configuration = {
                    **{
                        field: configuration[field]
                        for field in (
                            "rank4_control_sha256",
                            "rank4_engine_sha256",
                            "neural_puct_control_sha256",
                            "neural_puct_engine_sha256",
                            "rank4_adapter_sha256",
                            "neural_puct_adapter_sha256",
                            "shared_core_sha256",
                            "candidate_source_sha256",
                            "comparison_source_sha256",
                            "comparison_executable_path",
                            "comparison_executable_sha256",
                            "seed",
                            "candidate_tree_nodes",
                            "control_tree_nodes",
                            "control_work",
                            "max_actions",
                            "max_partial_paths",
                            "fpu",
                            "single_thread",
                        )
                    },
                    "exploration": exploration,
                    "opening_source": str(development_bank),
                    "opening_bank_sha256": development_bank_hash,
                    "baseline_receipt_sha256": configuration[
                        "baseline_receipt_sha256"
                    ],
                    "pairs": gate.DEVELOPMENT_PAIRS,
                    "time_ms": gate.DEVELOPMENT_TIME_MS,
                    "max_turns": gate.FINAL_MAX_TURNS,
                    "opening_plies": gate.FINAL_OPENING_PLIES,
                    "opening_bank_seed": gate.DEVELOPMENT_BANK_SEED,
                    "opening_bank_minimum_physical_plies": (
                        gate.FINAL_OPENING_PLIES
                    ),
                    "opponent": "both",
                    "opening_bank_classification": "development",
                    "opening_state_identities": [
                        f"development-state-{index}"
                        for index in range(gate.DEVELOPMENT_PAIRS)
                    ],
                }
                development_report = {
                    "schema": gate.SCHEMA,
                    "model": str(model),
                    "model_sha256": payload["model_sha256"],
                    "configuration": development_configuration,
                    "summary": {
                        "games": gate.DEVELOPMENT_PAIRS * 4,
                        "wins": wins,
                        "losses": gate.DEVELOPMENT_PAIRS * 4 - wins,
                        "illegal": 0,
                        "unfinished": 0,
                        "colors": [
                            {
                                "games": gate.DEVELOPMENT_PAIRS * 2,
                                "wins": wins_per_color,
                            },
                            {
                                "games": gate.DEVELOPMENT_PAIRS * 2,
                                "wins": wins_per_color,
                            },
                        ],
                        "candidate": {"p99_ms": 19.0, "max_ms": 19.0},
                    },
                    "results": development_results,
                }
                report_path = directory / f"development-{exploration}.json"
                report_path.write_text(json.dumps(development_report))
                report_paths.append(report_path)
            real_validate = gate.tuning_gate.baseline_gate.validate_receipt

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
                gate.tuning_gate.baseline_gate,
                "validate_receipt",
                side_effect=fixture_validate,
            ):
                receipt = gate.tuning_gate.select(report_paths, baseline_path)
                receipt_path = directory / "tuning.json"
                receipt_path.write_text(json.dumps(receipt, sort_keys=True))
                configuration["tuning_receipt_path"] = str(receipt_path)
                configuration["tuning_receipt_sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
                final_report = directory / "final-report.json"
                final_report.write_text(json.dumps(payload))
                attempt_binding = {
                    "model_sha256": payload["model_sha256"],
                    "tuning_receipt_sha256": configuration[
                        "tuning_receipt_sha256"
                    ],
                    "baseline_receipt_sha256": configuration[
                        "baseline_receipt_sha256"
                    ],
                    "comparison_executable_sha256": configuration[
                        "comparison_executable_sha256"
                    ],
                    "promotion_gate_sha256": hashlib.sha256(
                        pathlib.Path(gate.__file__).read_bytes()
                    ).hexdigest(),
                    "opening_bank_path": str(bank.resolve()),
                    "comparison_report_path": str(final_report.resolve()),
                    "chosen_exploration": configuration["exploration"],
                }
                attempt = {
                    "schema": gate.ATTEMPT_SCHEMA,
                    "attempt_id": hashlib.sha256(
                        gate.canonical_json_bytes(attempt_binding)
                    ).hexdigest(),
                    "state": "consumed",
                    "binding": attempt_binding,
                    "prepared": {"at_utc": "2026-08-24T00:00:00Z"},
                    "bank_registration": {
                        "opening_bank_sha256": configuration[
                            "opening_bank_sha256"
                        ]
                    },
                    "report_consumption": {
                        "comparison_report_path": str(final_report.resolve()),
                        "comparison_report_sha256": hashlib.sha256(
                            final_report.read_bytes()
                        ).hexdigest(),
                    },
                }
                attempt_path = directory / "attempt.json"
                gate.atomic_json(attempt_path, attempt)
                decision = gate.evaluate(
                    payload,
                    1_000,
                    1_000.0,
                    tuning_receipt=receipt,
                    baseline_receipt=baseline,
                    report_path=final_report,
                    verify_files=True,
                    attempt_ledger=attempt,
                    attempt_ledger_path=attempt_path,
                )
            self.assertTrue(decision["eligible"], decision["errors"])


if __name__ == "__main__":
    unittest.main()
