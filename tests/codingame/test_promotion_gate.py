import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "submissions/codingame/tools/promotion_gate.py"
SPEC = importlib.util.spec_from_file_location("promotion_gate", MODULE_PATH)
promotion_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion_gate)


class PromotionStatisticsTest(unittest.TestCase):
    def test_candidate_hypothesis_requirement_is_shared_identity_evidence(self):
        manifest = {"candidate_submission_sha256": "frozen"}
        self.assertEqual(
            promotion_gate.candidate_hypothesis_requirement(manifest, "frozen"),
            {
                "id": "candidate_matches_frozen_hypothesis",
                "passed": True,
                "observed": "frozen",
                "operator": "==",
                "threshold": "frozen",
            },
        )
        self.assertFalse(
            promotion_gate.candidate_hypothesis_requirement(
                manifest, "changed"
            )["passed"]
        )

    def test_verified_bank_snapshot_is_immutable_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "mutable.tsv"
            snapshot = root / "results" / "bank.tsv"
            source.write_bytes(b"first\n")
            expected = hashlib.sha256(b"first\n").hexdigest()
            result = promotion_gate.snapshot_verified_bank(
                source, snapshot, expected
            )
            self.assertEqual(result.read_bytes(), b"first\n")

            source.write_bytes(b"second\n")
            reused = promotion_gate.snapshot_verified_bank(
                source, snapshot, expected
            )
            self.assertEqual(reused.read_bytes(), b"first\n")

            snapshot.write_bytes(b"corrupt\n")
            with self.assertRaises(promotion_gate.UsageError):
                promotion_gate.snapshot_verified_bank(
                    source, snapshot, expected
                )

    def test_exclusive_json_marker_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "ledger" / "claim.json"
            promotion_gate.atomic_json_create_exclusive(path, {"owner": "first"})
            with self.assertRaises(FileExistsError):
                promotion_gate.atomic_json_create_exclusive(
                    path, {"owner": "second"}
                )
            self.assertEqual(json.loads(path.read_text()), {"owner": "first"})

    def test_stage_prerequisite_reaggregates_raw_predecessor_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = {}
            for name in ("test", "runner", "timing"):
                path = root / name
                path.write_text(name)
                paths[name] = path

            candidate_hash = "candidate"
            manifest_hash = "manifest"
            bot = "candidate_bot"
            manifest = {
                "banks": {
                    "initial.tsv": {"records": 1, "sha256": "initial-bank"},
                    "development.tsv": {
                        "records": 1,
                        "sha256": "development-bank",
                    },
                },
                "incumbent": {"submission_sha256": "incumbent"},
                "stages": {
                    "initial": {
                        "bank": "initial.tsv",
                        "node_budget": 5000,
                        "maximum_turns": 20,
                        "minimum_mean": 0.5,
                    },
                    "development": {
                        "bank": "development.tsv",
                        "node_budget": 5000,
                        "maximum_turns": 20,
                        "minimum_mean": 0.5,
                    },
                },
                "statistics": {"resamples": 10, "seed": 17},
            }
            identity = promotion_gate.expected_stage_identity(
                manifest,
                manifest_hash,
                "initial",
                candidate_hash,
                promotion_gate.sha256(paths["runner"]),
            )
            preflight = {
                "schema": "papersoccer.codingame-promotion-preflight.v1",
                "bot": bot,
                "passed": True,
                "candidate_submission_sha256": candidate_hash,
                "manifest_sha256": manifest_hash,
                "artifact_test_binary_sha256": promotion_gate.sha256(
                    paths["test"]
                ),
                "runner_binary_sha256": promotion_gate.sha256(paths["runner"]),
                "timing_binary_sha256": promotion_gate.sha256(paths["timing"]),
            }
            (root / "preflight.json").write_text(json.dumps(preflight))
            shard = {
                "schema": "papersoccer.codingame-promotion-shard.v2",
                "identity": identity,
                "configuration": {
                    "stage": "initial",
                    "node_budget": 5000,
                    "maximum_turns": 20,
                    "shard_count": 1,
                    "shard_index": 0,
                },
                "pairs": [{
                    "opening_id": "initial",
                    "source_game_id": 0,
                    "stratum": "initial",
                    "winner_tier": "initial",
                    "historical_winner_player": 0,
                    "candidate_pair_score": 0.5,
                    "incumbent_control": {"winner": 0, "turns": 10},
                    "games": [
                        {"candidate_player": 0, "winner": 0},
                        {"candidate_player": 1, "winner": 0},
                    ],
                }],
                "operational": {
                    field: 0 for field in promotion_gate.OPERATIONAL_FIELDS
                },
            }
            shard_directory = root / "shards" / "initial" / "5000-nodes"
            shard_directory.mkdir(parents=True)
            shard_path = shard_directory / "shard-000-of-001.json"
            shard_path.write_text(json.dumps(shard))
            report = promotion_gate.reaggregate_stage_from_shards(
                root, manifest, "initial", identity
            )
            report_path = root / "initial.json"
            report_path.write_text(json.dumps(report))

            promotion_gate.require_stage_prerequisites(
                root, "development", bot, manifest, candidate_hash,
                manifest_hash, paths
            )

            report["pairs"]["opening_pair_mean_score"] = 1.0
            report_path.write_text(json.dumps(report))
            with self.assertRaises(promotion_gate.IncompleteError):
                promotion_gate.require_stage_prerequisites(
                    root, "development", bot, manifest, candidate_hash,
                    manifest_hash, paths
                )

    def test_quantile_interpolates(self):
        self.assertEqual(promotion_gate.quantile([0.0, 1.0], 0.5), 0.5)

    def test_bootstrap_resamples_source_games_and_is_deterministic(self):
        pairs = [
            {"stratum": "a", "candidate_pair_score": 0.5},
            {"stratum": "a", "candidate_pair_score": 0.5},
            {"stratum": "b", "candidate_pair_score": 0.5},
        ]
        for index, pair in enumerate(pairs):
            pair["source_game_id"] = index
            pair["opening_id"] = f"opening-{index}"
        first = promotion_gate.source_game_cluster_bootstrap(pairs, 100, 7)
        second = promotion_gate.source_game_cluster_bootstrap(pairs, 100, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["lower"], 0.5)
        self.assertEqual(first["upper"], 0.5)

    def test_bootstrap_does_not_treat_positions_from_one_game_as_independent(self):
        pairs = [
            {
                "opening_id": f"opening-{index}",
                "source_game_id": 11 if index < 3 else 22,
                "candidate_pair_score": 1.0 if index < 3 else 0.0,
            }
            for index in range(4)
        ]
        result = promotion_gate.source_game_cluster_bootstrap(pairs, 1000, 19)
        self.assertEqual(result["opening_pairs"], 4)
        self.assertEqual(result["source_game_clusters"], 2)
        self.assertEqual(result["estimate"], 0.5)

    def test_independent_replay_rejects_an_overlong_complete_turn(self):
        row = {
            "transcript": "00",
            "winner_player_id": "0",
        }
        with self.assertRaises(promotion_gate.UsageError):
            promotion_gate.independently_reconstruct_bank_state(row)

    def test_multi_budget_profiles_require_every_budget(self):
        identity = {"candidate": "candidate", "runner": "runner"}
        passing = {
            "node_budget": 30000,
            "passed": True,
            "requirements": [{"id": "score", "passed": True}],
        }
        failing = {
            "node_budget": 100000,
            "passed": False,
            "requirements": [{"id": "score", "passed": False}],
        }
        report = promotion_gate.combine_budget_profiles(
            "test", identity, [passing, failing]
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["reason_codes"], ["nodes_100000:score"])

    def test_strength_profiles_normalize_legacy_and_mixed_execution_modes(self):
        self.assertIsNone(promotion_gate.configured_required_jobs({}))
        self.assertEqual(
            promotion_gate.configured_required_jobs({"required_jobs": 4}), 4
        )
        for invalid_jobs in (True, 0, -1, 4.0, "4"):
            with self.assertRaises(promotion_gate.UsageError):
                promotion_gate.configured_required_jobs({
                    "required_jobs": invalid_jobs,
                })

        legacy = promotion_gate.configured_strength_profiles({
            "node_budgets": [5000, 30000],
        })
        self.assertEqual(
            [(item["id"], item["node_budget"], item["time_budget_ms"])
             for item in legacy],
            [("5000-nodes", 5000, 0), ("30000-nodes", 30000, 0)],
        )
        self.assertTrue(all(item["legacy"] for item in legacy))

        mixed = promotion_gate.configured_strength_profiles({
            "strength_profiles": [
                {"id": "30k-nodes", "mode": "nodes", "value": 30000},
                {
                    "id": "130ms",
                    "mode": "time_ms",
                    "value": 130,
                    "max_nodes": 3000000,
                    "thresholds": {"minimum_throughput_ratio": None},
                },
            ],
        })
        self.assertEqual(
            [(item["id"], item["node_budget"], item["time_budget_ms"])
             for item in mixed],
            [("30k-nodes", 30000, 0), ("130ms", 3000000, 130)],
        )
        self.assertEqual(
            promotion_gate.strength_profile_identity(mixed[1]),
            {
                "id": "130ms",
                "mode": "time_ms",
                "value": 130,
                "max_nodes": 3000000,
            },
        )
        self.assertFalse(any(item["legacy"] for item in mixed))
        legacy_configuration = {
            "stage": "development",
            "node_budget": 5000,
            "maximum_turns": 20,
            "shard_count": 1,
            "shard_index": 0,
        }
        self.assertTrue(promotion_gate.shard_configuration_matches_profile(
            legacy_configuration, "development", legacy[0], 20, 1, 0
        ))
        explicit_configuration = dict(
            legacy_configuration, stage="validation", node_budget=30000
        )
        self.assertFalse(promotion_gate.shard_configuration_matches_profile(
            explicit_configuration, "validation", mixed[0], 20, 1, 0
        ))

        invalid_configs = [
            {
                "node_budget": 30000,
                "strength_profiles": [
                    {"id": "30k-nodes", "mode": "nodes", "value": 30000},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "first", "mode": "nodes", "value": 30000},
                    {"id": "second", "mode": "nodes", "value": 30000},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "30k-nodes", "mode": "nodes", "value": 30000,
                     "max_nodes": 3000000},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "130ms", "mode": "time_ms", "value": 130},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "bad", "mode": "seconds", "value": 1,
                     "max_nodes": 3000000},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "one-node", "mode": "nodes", "value": 1,
                     "max_nodes": True},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "overflow", "mode": "time_ms",
                     "value": 1 << 32, "max_nodes": 3000000},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "bad-threshold", "mode": "nodes", "value": 30000,
                     "thresholds": {"minimum_meen": 0.5}},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "bad-type", "mode": "nodes", "value": 30000,
                     "thresholds": {"minimum_mean": "0.5"}},
                ],
            },
            {
                "strength_profiles": [
                    {"id": "bad-flag", "mode": "nodes", "value": 30000,
                     "thresholds": {
                         "require_more_wins_than_incumbent": 1,
                     }},
                ],
            },
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(promotion_gate.UsageError):
                    promotion_gate.configured_strength_profiles(config)

    def test_explicit_strength_profiles_require_every_profile(self):
        identity = {"candidate": "candidate", "runner": "runner"}
        passing = {
            "strength_profile": {
                "id": "30k-nodes", "mode": "nodes", "value": 30000,
                "max_nodes": 30000,
            },
            "passed": True,
            "requirements": [{"id": "score", "passed": True}],
        }
        failing = {
            "strength_profile": {
                "id": "130ms", "mode": "time_ms", "value": 130,
                "max_nodes": 3000000,
            },
            "passed": False,
            "requirements": [{"id": "score", "passed": False}],
        }
        report = promotion_gate.combine_strength_profiles(
            "validation", identity, [passing, failing]
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["reason_codes"], ["130ms:score"])
        self.assertEqual(report["strength_profiles"], [passing, failing])

    def test_mixed_profile_reaggregation_checks_exact_execution_identity(self):
        identity = {"candidate": "candidate", "runner": "runner"}
        manifest = {
            "banks": {"validation.tsv": {"records": 1}},
            "stages": {
                "validation": {
                    "bank": "validation.tsv",
                    "maximum_turns": 20,
                    "minimum_mean": 0.5,
                    "minimum_throughput_ratio": 0.9,
                    "strength_profiles": [
                        {
                            "id": "30k-nodes",
                            "mode": "nodes",
                            "value": 30000,
                        },
                        {
                            "id": "130ms",
                            "mode": "time_ms",
                            "value": 130,
                            "max_nodes": 3000000,
                            "thresholds": {
                                "minimum_throughput_ratio": None,
                            },
                        },
                    ],
                },
            },
            "statistics": {"resamples": 10, "seed": 17},
        }
        game_zero = {
            "candidate_player": 0,
            "winner": 0,
            "candidate_nodes": 100,
            "incumbent_nodes": 100,
            "candidate_searches": 1,
            "incumbent_searches": 1,
            "candidate_ms": 10.0,
            "incumbent_ms": 10.0,
        }
        game_one = dict(game_zero, candidate_player=1, winner=0)
        pair = {
            "opening_id": "opening",
            "source_game_id": 1,
            "stratum": "d1",
            "winner_tier": "elite",
            "historical_winner_player": 0,
            "candidate_pair_score": 0.5,
            "incumbent_control": {"winner": 0, "turns": 10},
            "games": [game_zero, game_one],
        }
        operational = {
            field: 0 for field in promotion_gate.OPERATIONAL_FIELDS
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            configurations = {
                "30k-nodes": {"node_budget": 30000, "time_budget_ms": 0},
                "130ms": {"node_budget": 3000000, "time_budget_ms": 130},
            }
            for profile_id, execution in configurations.items():
                directory = root / "shards" / "validation" / profile_id
                directory.mkdir(parents=True)
                shard = {
                    "schema": "papersoccer.codingame-promotion-shard.v2",
                    "identity": identity,
                    "configuration": {
                        "stage": "validation",
                        "maximum_turns": 20,
                        "shard_count": 1,
                        "shard_index": 0,
                        **execution,
                    },
                    "pairs": [pair],
                    "operational": operational,
                }
                (directory / "shard-000-of-001.json").write_text(
                    json.dumps(shard)
                )

            report = promotion_gate.reaggregate_stage_from_shards(
                root, manifest, "validation", identity
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                [item["strength_profile"]["id"]
                 for item in report["strength_profiles"]],
                ["30k-nodes", "130ms"],
            )
            node_requirement_ids = {
                item["id"] for item in report["strength_profiles"][0][
                    "requirements"
                ]
            }
            time_requirement_ids = {
                item["id"] for item in report["strength_profiles"][1][
                    "requirements"
                ]
            }
            self.assertIn("candidate_to_incumbent_throughput", node_requirement_ids)
            self.assertNotIn(
                "candidate_to_incumbent_throughput", time_requirement_ids
            )
            self.assertEqual(
                report["strength_profiles"][1]["control_normalization"]["method"],
                "rank5_vs_rank5_per_opening_same_time_budget_with_node_cap",
            )

            time_path = (
                root / "shards" / "validation" / "130ms" /
                "shard-000-of-001.json"
            )
            tampered = json.loads(time_path.read_text())
            tampered["configuration"]["time_budget_ms"] = 129
            time_path.write_text(json.dumps(tampered))
            with self.assertRaises(promotion_gate.IncompleteError):
                promotion_gate.reaggregate_stage_from_shards(
                    root, manifest, "validation", identity
                )
            tampered["configuration"]["time_budget_ms"] = 130
            tampered["configuration"]["node_budget"] = 2999999
            time_path.write_text(json.dumps(tampered))
            with self.assertRaises(promotion_gate.IncompleteError):
                promotion_gate.reaggregate_stage_from_shards(
                    root, manifest, "validation", identity
                )

    def test_locked_test_marker_preserves_legacy_and_profiles_time_identity(self):
        manifest = {
            "banks": {
                "test.tsv": {"sha256": "bank"},
            },
            "stages": {
                "test": {
                    "bank": "test.tsv",
                    "node_budgets": [30000, 100000],
                },
            },
        }
        legacy = promotion_gate.locked_test_consumption_marker(
            manifest, "manifest", "candidate", 3
        )
        self.assertEqual(
            legacy,
            {
                "schema": "papersoccer.codingame-locked-test-consumption.v1",
                "candidate_submission_sha256": "candidate",
                "manifest_sha256": "manifest",
                "bank_sha256": "bank",
                "node_budgets": [30000, 100000],
                "shard_count": 3,
            },
        )

        manifest["stages"]["test"] = {
            "bank": "test.tsv",
            "strength_profiles": [
                {"id": "100k-nodes", "mode": "nodes", "value": 100000},
                {
                    "id": "130ms", "mode": "time_ms", "value": 130,
                    "max_nodes": 3000000,
                },
            ],
        }
        explicit = promotion_gate.locked_test_consumption_marker(
            manifest, "manifest", "candidate", 3
        )
        self.assertEqual(
            explicit["schema"],
            "papersoccer.codingame-locked-test-consumption.v2",
        )
        self.assertNotIn("node_budgets", explicit)
        self.assertEqual(
            explicit["strength_profiles"],
            [
                {
                    "id": "100k-nodes", "mode": "nodes", "value": 100000,
                    "max_nodes": 100000,
                },
                {
                    "id": "130ms", "mode": "time_ms", "value": 130,
                    "max_nodes": 3000000,
                },
            ],
        )

    def test_locked_test_marker_shard_count_matches_every_profile(self):
        manifest = {
            "banks": {"test.tsv": {"sha256": "bank"}},
            "stages": {
                "test": {
                    "bank": "test.tsv",
                    "strength_profiles": [
                        {
                            "id": "100k-nodes", "mode": "nodes",
                            "value": 100000,
                        },
                        {
                            "id": "130ms", "mode": "time_ms", "value": 130,
                            "max_nodes": 3000000,
                        },
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            ledger = root / "ledger"
            ledger.mkdir()
            for profile_id in ("100k-nodes", "130ms"):
                directory = root / "shards" / "test" / profile_id
                directory.mkdir(parents=True)
                (directory / "shard-000-of-001.json").write_text("{}")
            marker = promotion_gate.locked_test_consumption_marker(
                manifest, "manifest", "candidate", 2
            )
            marker_path = (
                ledger / "locked-test-consumption-bank.json"
            )
            marker_path.write_text(json.dumps(marker))

            with mock.patch.object(
                promotion_gate, "HOLDOUT_LEDGER", ledger
            ):
                with mock.patch.object(
                    promotion_gate, "locked_test_consumption_path",
                    return_value=marker_path,
                ):
                    with self.assertRaisesRegex(
                        promotion_gate.IncompleteError,
                        "does not match raw evidence",
                    ):
                        promotion_gate.verify_locked_test_consumption(
                            root, manifest, "manifest", "candidate"
                        )

                    marker_path.write_text(json.dumps(
                        promotion_gate.locked_test_consumption_marker(
                            manifest, "manifest", "candidate", 1
                        )
                    ))
                    second = (
                        root / "shards" / "test" / "130ms" /
                        "shard-001-of-002.json"
                    )
                    second.write_text("{}")
                    with self.assertRaisesRegex(
                        promotion_gate.IncompleteError,
                        "different shard counts",
                    ):
                        promotion_gate.verify_locked_test_consumption(
                            root, manifest, "manifest", "candidate"
                        )

    def test_timing_reaggregates_locked_test_evidence_before_running_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = {
                "submission": root / "submission.cpp",
                "runner": root / "runner",
                "test": root / "test",
                "timing": root / "timing",
                "runner_target": "runner-target",
                "test_target": "test-target",
                "timing_target": "timing-target",
            }
            for name in ("submission", "runner", "test", "timing"):
                paths[name].write_text(name)
            manifest = {
                "incumbent": {"submission_sha256": "incumbent"},
                "banks": {"test.tsv": {"sha256": "bank"}},
                "stages": {
                    "test": {
                        "bank": "test.tsv",
                        "node_budget": 100000,
                        "maximum_turns": 20,
                    },
                },
            }
            candidate_hash = promotion_gate.sha256(paths["submission"])
            identity = promotion_gate.expected_stage_identity(
                manifest, "manifest", "test", candidate_hash,
                promotion_gate.sha256(paths["runner"]),
            )
            recorded = {
                "schema": "papersoccer.codingame-promotion-stage.v2",
                "stage": "test",
                "identity": identity,
                "passed": True,
            }
            (root / "test.json").write_text(json.dumps(recorded))
            recomputed = dict(recorded, requirements=[])

            with (
                mock.patch.object(
                    promotion_gate, "validate",
                    return_value={"manifest_sha256": "manifest"},
                ),
                mock.patch.object(
                    promotion_gate, "load_manifest", return_value=manifest
                ),
                mock.patch.object(
                    promotion_gate, "candidate_paths", return_value=paths
                ),
                mock.patch.object(promotion_gate, "check_generated_submission"),
                mock.patch.object(promotion_gate, "build_targets"),
                mock.patch.object(
                    promotion_gate, "result_directory", return_value=root
                ),
                mock.patch.object(promotion_gate, "require_stage_prerequisites"),
                mock.patch.object(
                    promotion_gate, "reaggregate_stage_from_shards",
                    return_value=recomputed,
                ),
            ):
                with self.assertRaisesRegex(
                    promotion_gate.IncompleteError,
                    "does not match its raw evidence",
                ):
                    promotion_gate.timing(
                        "bot", root / "manifest.json", root / "build", root
                    )

    def test_timing_report_recomputes_all_cases_and_limits(self):
        config = {
            "fresh_process_samples": 1,
            "shell_cases": ["shell"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        }
        samples = [
            {"sample": 0, "case": "initial-player-0", "player": 0,
             "first_ms": 650.0, "later_ms": 130.0},
            {"sample": 0, "case": "initial-player-1", "player": 1,
             "first_ms": 651.0, "later_ms": 131.0},
            {"sample": 0, "case": "shell", "later_ms": 132.0},
        ]
        passing = promotion_gate.make_timing_report(
            "bot", "candidate", "manifest", "timing", samples, config
        )
        self.assertTrue(passing["passed"])
        samples[-1]["later_ms"] = 205.0
        failing = promotion_gate.make_timing_report(
            "bot", "candidate", "manifest", "timing", samples, config
        )
        self.assertFalse(failing["passed"])
        self.assertIn("later_max", failing["reason_codes"])

    def test_neutral_completed_control_is_explicitly_rejected(self):
        identity = {"candidate": "same", "incumbent": "same"}
        game_zero = {
            "candidate_player": 0,
            "winner": 0,
            "exchange_ply1_probes": 7,
            "exchange_ply1_win_hits": 2,
            "exchange_ply1_loss_hits": 1,
            "exchange_ply1_cutoffs": 3,
            "exchange_ply2_probes": 5,
            "exchange_ply2_win_hits": 1,
            "exchange_ply2_loss_hits": 2,
            "exchange_ply2_cutoffs": 3,
        }
        game_one = {"candidate_player": 1, "winner": 0}
        pairs = [
            {
                "opening_id": f"opening-{index}",
                "source_game_id": index + 1,
                "stratum": "shell",
                "winner_tier": "elite",
                "historical_winner_player": 0,
                "candidate_pair_score": 0.5,
                "incumbent_control": {"winner": 0, "turns": 10},
                "games": [game_zero, game_one],
            }
            for index in range(4)
        ]
        manifest = {
            "banks": {"development.tsv": {"records": 4}},
            "stages": {
                "development": {
                    "bank": "development.tsv",
                    "minimum_mean": 0.52,
                    "require_more_wins_than_incumbent": True,
                }
            },
            "statistics": {"resamples": 100, "seed": 17},
        }
        shard = {
            "schema": "papersoccer.codingame-promotion-shard.v2",
            "identity": identity,
            "pairs": pairs,
            "operational": {field: 0 for field in promotion_gate.OPERATIONAL_FIELDS},
        }
        report = promotion_gate.aggregate_stage(
            manifest, "development", [shard], identity
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "reject")
        self.assertEqual(report["confidence_interval"]["lower"], 0.5)
        control = report["control_normalization"]
        self.assertEqual(control["winner_role"]["score"], 1.0)
        self.assertEqual(control["loser_role"]["score"], 0.0)
        self.assertEqual(control["minimum_adjusted_uplift"], 0.0)
        self.assertEqual(control["net_uplift"], 0.0)
        self.assertEqual(report["diagnostics"]["exchange_ply1_probes"], 28)
        self.assertEqual(report["diagnostics"]["exchange_ply1_cutoffs"], 12)
        self.assertEqual(report["diagnostics"]["exchange_ply2_probes"], 20)
        self.assertEqual(report["diagnostics"]["exchange_ply2_cutoffs"], 12)
        self.assertIn("cluster_mean_score", report["reason_codes"])
        self.assertIn("candidate_game_wins", report["reason_codes"])

    def test_physical_uplift_gate_keeps_historical_repartition_diagnostic(self):
        identity = {"candidate": "same", "incumbent": "same"}
        pairs = []
        for index in range(4):
            candidate_wins_player_one = index < 2
            games = [
                {
                    "candidate_player": 0,
                    "winner": 1 if candidate_wins_player_one else 0,
                },
                {
                    "candidate_player": 1,
                    "winner": 1 if candidate_wins_player_one else 0,
                },
            ]
            control_winner = 0 if index < 2 else 1
            pairs.append({
                "opening_id": f"opening-{index}",
                "source_game_id": index + 1,
                "stratum": "shell",
                "winner_tier": "elite",
                "historical_winner_player": control_winner,
                "candidate_pair_score": 0.5,
                "incumbent_control": {"winner": control_winner, "turns": 10},
                "games": games,
            })
        shard = {
            "schema": "papersoccer.codingame-promotion-shard.v2",
            "identity": identity,
            "pairs": pairs,
            "operational": {
                field: 0 for field in promotion_gate.OPERATIONAL_FIELDS
            },
        }
        manifest = {
            "banks": {"validation.tsv": {"records": 4}},
            "stages": {
                "validation": {
                    "bank": "validation.tsv",
                    "minimum_mean": 0.5,
                    "minimum_physical_color_uplift": -0.05,
                }
            },
            "statistics": {"resamples": 100, "seed": 17},
        }
        report = promotion_gate.aggregate_stage(
            manifest, "validation", [shard], identity
        )
        self.assertTrue(report["passed"])
        control = report["control_normalization"]
        self.assertEqual(control["minimum_physical_color_uplift"], 0.0)
        self.assertEqual(control["minimum_historical_uplift"], -1.0)
        self.assertEqual(control["minimum_historical_role_score"], 0.0)

        manifest["stages"]["validation"][
            "minimum_historical_role_score"
        ] = 0.45
        guarded = promotion_gate.aggregate_stage(
            manifest, "validation", [shard], identity
        )
        self.assertFalse(guarded["passed"])
        self.assertEqual(
            guarded["reason_codes"], ["minimum_historical_role_score"]
        )

        manifest["stages"]["validation"] = {
            "bank": "validation.tsv",
            "minimum_mean": 0.5,
            "minimum_control_adjusted_uplift": -0.05,
        }
        legacy = promotion_gate.aggregate_stage(
            manifest, "validation", [shard], identity
        )
        self.assertFalse(legacy["passed"])
        self.assertEqual(
            legacy["reason_codes"], ["minimum_control_adjusted_uplift"]
        )


if __name__ == "__main__":
    unittest.main()
