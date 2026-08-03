from __future__ import annotations

import concurrent.futures
import copy
import json
import pathlib
import re
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from benchmarks.flagship_study import analysis, studylib


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


def valid_build_provenance(source_commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema": "papersoccer.arena-build.v1",
        "runtime": "native",
        "build_type": "Release",
        "ndebug": True,
        "sanitizers_enabled": False,
        "compiler_id": "Clang",
        "compiler_version": "1",
        "configured_flags": "-O3 -DNDEBUG -std=c++20",
        "cxx_standard": 202002,
        "source_commit": source_commit,
        "source_dirty": False,
    }


class PairAndIdIntegrityTests(unittest.TestCase):
    def test_color_swapped_pair_counts_decisive_split_without_draw(self) -> None:
        winners = {
            "4\0pair-a": ["left", "left"],
            "4\0pair-b": ["left", "right"],
            "8\0pair-c": ["right", "right"],
        }
        summary, strata = studylib._pair_summaries(winners, "left", "right")
        self.assertEqual(summary["left_wins"], 3)
        self.assertEqual(summary["left_losses"], 3)
        self.assertEqual(summary["pairs_won_2_0"], 1)
        self.assertEqual(summary["pairs_split_1_1"], 1)
        self.assertEqual(summary["pairs_lost_0_2"], 1)
        self.assertEqual(summary["mean_pair_score"], 0.5)
        self.assertEqual(summary["truncations"], 0)
        self.assertEqual(strata, {4: [1.0, 0.5], 8: [0.0]})

    def test_missing_or_extra_pair_games_are_rejected(self) -> None:
        with self.assertRaisesRegex(studylib.StudyError, "exactly two"):
            studylib._pair_summaries({"4\0p": ["left"]}, "left", "right")
        with self.assertRaisesRegex(studylib.StudyError, "outside"):
            studylib._pair_summaries(
                {"4\0p": ["left", "third"]}, "left", "right"
            )

    def test_annotation_creates_stable_unique_run_pair_and_game_ids(self) -> None:
        record = studylib.OpeningRecord(
            "development-d04-deadbeef", "development", 4, "11",
            "1" * 64, "2" * 64, "one", ((4, 5), (5, 5), (6, 5), (6, 6)),
        )
        unit = studylib.StudyUnit(
            "development", "matchup-a", "left", "right", "bank-a",
            "unused.tsv", 4, 1,
        )
        report = {"games": [
            {"pair_index": 0, "game_in_pair": 0},
            {"pair_index": 0, "game_in_pair": 1},
        ]}
        studylib._annotate_report(report, unit, "a" * 64, "run-1", [record])
        identifiers = [game["study_ids"] for game in report["games"]]
        self.assertEqual(identifiers[0]["pair"], identifiers[1]["pair"])
        self.assertNotEqual(identifiers[0]["game"], identifiers[1]["game"])
        self.assertEqual(report["study"]["unit_id"], unit.unit_id)

    def test_arena_configuration_must_match_the_manifest(self) -> None:
        unit = studylib.StudyUnit(
            "development", "m", "left", "right", "b", "unused", 4, 1
        )
        manifest = {
            "seeds": {"bot": {"development": "123"}},
            "rules": {"width": 8, "height": 10, "max_game_plies": 512},
        }
        configs = {
            "left": {
                "kind": "mcts",
                "settings": {
                    "iterations": 8, "exploration": 1.4142135623730951,
                    "rollout_policy": "tactical", "leaf_policy": "rollout_only",
                    "quiescence_max_depth": 8, "quiescence_max_nodes": 256,
                    "reuse_tree": True, "node_capacity": 64,
                },
            },
            "right": {
                "kind": "alpha-beta",
                "settings": {
                    "max_turn_depth": 2, "max_nodes": 256,
                    "transposition_table_entries": 64, "max_search_plies": 8,
                },
            },
        }
        seed = studylib._derived_seed("123", "m", "4")
        report = {
            "schema": "papersoccer.arena.v1", "mode": "matches",
            "runtime": "native",
            "configuration": {
                "rules": {"width": 8, "height": 10},
                "base_seed": str(seed), "seed_pairs": 1, "games": 2,
                "opening_plies": 4, "max_plies": 512, "bootstrap_samples": 1,
                "opening_generator": "frozen_uniform_legal_move_data_generation_bank",
                "opening_seed_derivation": "committed_bank_accepted_generation_seeds",
                "warmup": {"decisions_per_entrant": 8, "timed": False,
                           "bot_instances": "separate_from_measured_games"},
                "candidate": studylib._expected_arena_bot_config(configs["left"]),
                "reference": studylib._expected_arena_bot_config(configs["right"]),
            },
        }
        studylib._verify_arena_report_contract(report, unit, manifest, configs)
        report["configuration"]["candidate"]["iterations"] = 9
        with self.assertRaisesRegex(studylib.StudyError, "differs from manifest"):
            studylib._verify_arena_report_contract(report, unit, manifest, configs)


class PublicEntrantIdentityTests(unittest.TestCase):
    def test_public_study_docs_use_exact_entrant_labels(self) -> None:
        paths = (
            REPOSITORY / "benchmarks/flagship_study/README.md",
            REPOSITORY / "benchmarks/flagship_study/analysis_contract.md",
        )
        for path in paths:
            text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                for label in studylib.PUBLIC_LABELS.values():
                    self.assertIn(label, text)
                without_exemptions = text.replace(
                    studylib.RANK5_DISCLAIMER, "<MANDATED_DISCLAIMER>"
                ).replace(studylib.PUBLIC_RANK5_LABEL, "<RANK5_ENTRANT>")
                self.assertIsNone(
                    re.search(r"\bRank5Derived(?:Bot)?\b|\bRank5\b",
                              without_exemptions),
                    f"public shorthand for fixed entrant in {path}",
                )


class Rank5SequenceTests(unittest.TestCase):
    CONFIGS = {
        "rank5": {"kind": "rank5-derived"},
        "other": {"kind": "mcts"},
    }
    UNIT = studylib.StudyUnit(
        "validation", "m", "rank5", "other", "b", "unused", 4, 1
    )

    @staticmethod
    def rank5(edge: int, length: int, cached: bool) -> dict:
        return {
            "bot": "candidate", "legal": True,
            "rank5_derived": {
                "planned_action_length": length,
                "current_edge_index": edge,
                "cached_continuation": cached,
            },
        }

    def test_complete_action_fresh_and_cached_edges_are_contiguous(self) -> None:
        game = {"decisions": [
            self.rank5(0, 3, False),
            self.rank5(1, 3, True),
            self.rank5(2, 3, True),
            {"bot": "reference", "legal": True, "rank5_derived": None},
        ]}
        studylib._validate_rank5_sequences(game, self.UNIT, self.CONFIGS)

    def test_orphan_cached_and_incomplete_actions_are_rejected(self) -> None:
        orphan = {"decisions": [self.rank5(1, 2, True)]}
        with self.assertRaisesRegex(studylib.StudyError, "without its fresh root"):
            studylib._validate_rank5_sequences(orphan, self.UNIT, self.CONFIGS)
        incomplete = {"decisions": [self.rank5(0, 2, False)]}
        with self.assertRaisesRegex(studylib.StudyError, "incomplete"):
            studylib._validate_rank5_sequences(incomplete, self.UNIT, self.CONFIGS)
        illegal_cached = {"decisions": [
            self.rank5(0, 2, False), self.rank5(1, 2, True),
        ]}
        illegal_cached["decisions"][1]["legal"] = False
        with self.assertRaisesRegex(studylib.StudyError, "unvalidated edge"):
            studylib._validate_rank5_sequences(
                illegal_cached, self.UNIT, self.CONFIGS
            )

    @staticmethod
    def aggregate_decision(*, cached: bool, scale: int) -> dict:
        stats = {
            "requested_nodes": 0 if cached else 50_000,
            "visited_nodes": 0 if cached else 100 * scale,
            "current_edge_index": 1 if cached else 0,
            "cached_continuation": cached,
            "budget_exhausted": not cached,
            "completed_turn_depth": scale,
            "attempted_turn_depth": scale + 1,
            "planned_action_length": 2,
            "root_score": -10 * scale,
        }
        for index, field in enumerate(
                studylib.RANK5_FRESH_COUNTER_FIELDS, start=1):
            stats[field] = index * scale
        return {"rank5_derived": stats}

    def test_fresh_root_counter_sums_and_maxima_exclude_cached_edges(self) -> None:
        summary = studylib._empty_diagnostics("rank5-derived")
        first = self.aggregate_decision(cached=False, scale=1)
        cached = self.aggregate_decision(cached=True, scale=99)
        second = self.aggregate_decision(cached=False, scale=3)
        for decision in (first, cached, second):
            studylib._add_diagnostics(summary, decision, "rank5-derived")

        self.assertEqual(summary["decisions"], 3)
        self.assertEqual(summary["fresh_root_searches"], 2)
        self.assertEqual(summary["cached_continuation_edges"], 1)
        self.assertEqual(summary["requested_nodes"], 100_000)
        self.assertEqual(summary["visited_nodes"], 400)
        self.assertNotIn("searches_sum", summary)
        self.assertNotIn("searches_max", summary)
        for index, field in enumerate(
                studylib.RANK5_FRESH_COUNTER_FIELDS, start=1):
            self.assertEqual(summary[f"{field}_sum"], index * 4)
            self.assertEqual(summary[f"{field}_max"], index * 3)
        studylib._validate_rank5_diagnostics_summary(summary)

        invalid_cached = self.aggregate_decision(cached=True, scale=1)
        invalid_cached["rank5_derived"]["visited_nodes"] = 1
        with self.assertRaisesRegex(studylib.StudyError, "cached continuation"):
            studylib._add_diagnostics(
                studylib._empty_diagnostics("rank5-derived"),
                invalid_cached,
                "rank5-derived",
            )


class ExecutionEnvironmentTests(unittest.TestCase):
    def test_full_verification_rejects_sanitized_arena(self) -> None:
        source_commit = "a" * 40
        arena_sha256 = "b" * 64
        manifest = {
            "study": {"study_class": "flagship"},
            "source": {
                "git_commit": source_commit,
                "arena_sha256": arena_sha256,
            },
            "environment": {
                "compiler": "Clang",
                "compiler_version": "1",
                "build_flags": "-O3 -DNDEBUG -std=c++20",
                "architecture": "arm64",
                "cpu": "test CPU",
                "physical_cores": 4,
                "logical_cores": 8,
                "memory_bytes": 16_000_000_000,
                "kernel": "test kernel",
                "python_version": studylib.platform.python_version(),
            },
        }
        environment = {
            "arena_sha256": arena_sha256,
            "machine": "arm64",
            "processor": "test CPU",
            "physical_cores": 4,
            "logical_cores": 8,
            "memory_bytes": 16_000_000_000,
            "platform": "test kernel",
            "python_version": studylib.platform.python_version(),
            "build_provenance": valid_build_provenance(source_commit),
        }
        studylib._verify_flagship_execution_environment(manifest, environment)

        environment["build_provenance"] = valid_build_provenance(source_commit)
        environment["build_provenance"]["sanitizers_enabled"] = True
        with self.assertRaisesRegex(
                studylib.StudyError, "optimized native Release C\\+\\+20 arena"):
            studylib._verify_flagship_execution_environment(
                manifest, environment
            )

    def test_validation_gate_requires_nominal_start_and_end_snapshots(self) -> None:
        nominal = {
            "observed_at_utc": "2026-08-03T01:00:00+00:00",
            "power_source": "ac",
            "power_status": "Now drawing from AC Power",
            "power_settings": "Currently in use: powermode 0",
            "thermal_status": (
                "No thermal warning level has been recorded "
                "No performance warning level has been recorded"
            ),
        }
        recorded = {
            **nominal,
            "gate_conditions_after": copy.deepcopy(nominal),
        }
        studylib._validate_recorded_validation_environment(recorded, "fixture")

        for name, mutate, message in (
            (
                "battery after",
                lambda value: value["gate_conditions_after"].update(
                    {"power_source": "battery"}
                ),
                "AC power",
            ),
            (
                "low power before",
                lambda value: value.update(
                    {"power_settings": "Currently in use: powermode 1"}
                ),
                "Low Power Mode",
            ),
            (
                "thermal unavailable after",
                lambda value: value["gate_conditions_after"].update(
                    {"thermal_status": "unavailable"}
                ),
                "nominal thermal state",
            ),
        ):
            with self.subTest(name=name):
                invalid = copy.deepcopy(recorded)
                mutate(invalid)
                with self.assertRaisesRegex(studylib.StudyError, message):
                    studylib._validate_recorded_validation_environment(
                        invalid, "fixture"
                    )


class CalibrationCurationTests(unittest.TestCase):
    @staticmethod
    def empty_payload(identifier: str = "rank5") -> dict:
        return {
            "schema": "papersoccer.flagship-calibration-observations.v1",
            "phase": "test",
            "bot_id": identifier,
            "score_kind": "signed",
            "score_perspective": "player_to_move",
            "decision_count": 0,
            "scores": [],
            "outcomes": [],
            "pair_cluster_ids": [],
            "stratum_ids": [],
            "excluded": {
                "cached_continuations": 0,
                "truncations": 0,
                "invalid_depths": 0,
            },
        }

    def test_aggregation_aligns_pair_context_and_excludes_cached_rank5(self) -> None:
        payload = self.empty_payload()
        fresh = {
            "player": "one",
            "rank5_derived": {
                "root_score": 12,
                "completed_turn_depth": 4,
                "cached_continuation": False,
            },
        }
        cached = {
            "rank5_derived": {
                "root_score": 12,
                "completed_turn_depth": 4,
                "cached_continuation": True,
            },
        }
        for decision in (fresh, cached):
            studylib._add_calibration_observation(
                payload, decision, "rank5", {"kind": "rank5-derived"},
                "rank5", pair_cluster_id="pair-1",
                stratum_id="matchup-a:opening-depth-4", truncated=False,
            )

        self.assertEqual(payload["decision_count"], 2)
        self.assertEqual(payload["scores"], [12.0])
        self.assertEqual(payload["outcomes"], [1])
        self.assertEqual(payload["pair_cluster_ids"], ["pair-1"])
        self.assertEqual(
            payload["stratum_ids"], ["matchup-a:opening-depth-4"]
        )
        self.assertEqual(payload["excluded"]["cached_continuations"], 1)
        studylib._validate_compact_calibration_observations(
            payload, "test", "rank5"
        )

    def test_payload_loader_requires_exact_aligned_accounting(self) -> None:
        payload = self.empty_payload("bot")
        payload.update({
            "decision_count": 2,
            "scores": [-1.0, 1.0],
            "outcomes": [0, 1],
            "pair_cluster_ids": ["pair-1", "pair-1"],
            "stratum_ids": ["matchup-a:opening-depth-4"] * 2,
        })
        curated = {
            "phase": "test",
            "calibration_observations": {"bot": payload},
        }
        loaded = studylib._curated_calibration_observations(
            curated, "test", {"bot"}
        )
        self.assertEqual(loaded["bot"], payload)

        misaligned = copy.deepcopy(curated)
        misaligned["calibration_observations"]["bot"]["stratum_ids"].pop()
        with self.assertRaisesRegex(studylib.StudyError, "misaligned"):
            studylib._curated_calibration_observations(
                misaligned, "test", {"bot"}
            )

        wrong_count = copy.deepcopy(curated)
        wrong_count["calibration_observations"]["bot"]["decision_count"] = 3
        with self.assertRaisesRegex(studylib.StudyError, "decision_count"):
            studylib._curated_calibration_observations(
                wrong_count, "test", {"bot"}
            )

        unknown = copy.deepcopy(curated)
        unknown["calibration_observations"]["bot"]["unknown"] = True
        with self.assertRaisesRegex(studylib.StudyError, "unknown"):
            studylib._curated_calibration_observations(
                unknown, "test", {"bot"}
            )

    def test_test_evaluation_uses_frozen_pair_cluster_bootstrap(self) -> None:
        payload = self.empty_payload("bot")
        payload.update({
            "decision_count": 9,
            "scores": [-2.0, -1.0, 1.0, 2.0, -0.5, 0.5, -0.2, 0.2],
            "outcomes": [0, 0, 1, 1, 0, 1, 1, 0],
            "pair_cluster_ids": [
                "p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"
            ],
            "stratum_ids": ["m1-d4"] * 4 + ["m1-d8"] * 4,
            "excluded": {
                "cached_continuations": 1,
                "truncations": 0,
                "invalid_depths": 0,
            },
        })
        mapping = analysis.CalibrationMapping(
            bot_id="bot", score_kind="signed", score_mean=0.0,
            score_scale=1.0, intercept=0.0, slope=1.0,
            sample_count=8, iterations=1,
        ).to_dict()
        manifest = {
            "seeds": {"analysis": {"test": "7000001"}},
            "statistics": {"calibration": {
                "bins": 10,
                "bootstrap_resamples": 200,
                "minimum_bin_successful_resamples": 1,
            }},
        }
        options = studylib._calibration_evaluation_options(manifest, "bot")
        self.assertEqual(
            options["seed"],
            studylib._derived_seed(
                "7000001", "calibration-pair-cluster", "bot"
            ),
        )
        first = studylib._evaluate_curated_calibration(
            analysis, mapping, payload, **options
        )
        second = studylib._evaluate_curated_calibration(
            analysis, mapping, payload, **options
        )

        self.assertEqual(first, second)
        self.assertEqual(first["samples"], 8)
        self.assertEqual(first["decision_count"], 9)
        self.assertEqual(first["excluded"]["cached_continuations"], 1)
        self.assertEqual(first["pair_clusters"], 4)
        bootstrap = first["pair_cluster_bootstrap_95"]
        self.assertEqual(
            bootstrap["method"], "pair_cluster_percentile_stratified"
        )
        self.assertEqual(bootstrap["seed"], str(options["seed"]))
        self.assertEqual(bootstrap["resamples"], 200)


class ShardAndResumptionTests(unittest.TestCase):
    def test_concurrent_raw_shard_publish_never_replaces_the_winner(self) -> None:
        def race(path: pathlib.Path, payloads: tuple[dict, dict]) -> list[object]:
            real_link = studylib.os.link
            barrier = threading.Barrier(2)

            def synchronized_link(source: object, destination: object) -> None:
                barrier.wait(timeout=5.0)
                real_link(source, destination)

            def publish(payload: dict) -> object:
                try:
                    return studylib._publish_raw_shard_atomic(path, payload)
                except studylib.StudyError as error:
                    return error

            with mock.patch.object(
                    studylib.os, "link", side_effect=synchronized_link), \
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(publish, payload) for payload in payloads]
                return [future.result(timeout=10.0) for future in futures]

        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "unit.json"
            payloads = ({"writer": "first"}, {"writer": "second"})
            results = race(path, payloads)
            self.assertEqual(sum(result is True for result in results), 1)
            conflicts = [
                result for result in results
                if isinstance(result, studylib.StudyError)
            ]
            self.assertEqual(len(conflicts), 1)
            self.assertIn("existing result preserved", str(conflicts[0]))
            self.assertIn(studylib.load_json(path), payloads)
            self.assertEqual(list(directory.glob(".unit.json.*.tmp")), [])

            path.unlink()
            identical = ({"writer": "same"}, {"writer": "same"})
            identical_results = race(path, identical)
            self.assertCountEqual(identical_results, [True, False])
            self.assertEqual(studylib.load_json(path), identical[0])
            self.assertEqual(list(directory.glob(".unit.json.*.tmp")), [])

    def test_deterministic_shards_partition_without_overlap(self) -> None:
        units = [studylib.StudyUnit(
            "development", f"m{index}", "a", "b", "bank", "path", 4, 1
        ) for index in range(10)]
        shards = [studylib.deterministic_shard(units, 3, index) for index in range(3)]
        flattened = [unit.unit_id for shard in shards for unit in shard]
        self.assertEqual(sorted(flattened), sorted(unit.unit_id for unit in units))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            [unit.unit_id for unit in studylib.deterministic_shard(units, 3, 1)],
            [unit.unit_id for unit in shards[1]],
        )

    def test_missing_and_unknown_shards_are_detected(self) -> None:
        manifest = {
            "openings": {"banks": [{"id": "bank", "phase": "development",
                                      "depth": 4, "pairs": 1, "path": "bank.tsv"}]},
            "schedule": {"tuning": [{"id": "m", "candidate": "a",
                                        "opponent": "b",
                                        "phases": ["development", "validation"]}],
                         "test": []},
            "outputs": {"raw_results_root": "results/study"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            with self.assertRaisesRegex(studylib.StudyError, "missing"):
                studylib._read_unit_reports(
                    manifest, repository, "f" * 64, "development", None
                )
            shard_dir = repository / "results/study" / ("f" * 64) / "development/shards"
            shard_dir.mkdir(parents=True)
            (shard_dir / "unknown.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(studylib.StudyError, "unknown"):
                studylib._read_unit_reports(
                    manifest, repository, "f" * 64, "development", None
                )

    def test_test_once_marker_resumes_then_rejects_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            manifest_path = repository / "manifest.json"
            selection_path = repository / "selection.json"
            bank_path = repository / "bank.tsv"
            development_path = repository / "curated/development.json"
            validation_path = repository / "curated/validation.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            selection_path.write_text("{}\n", encoding="utf-8")
            bank_path.write_text("bank\n", encoding="utf-8")
            development_path.parent.mkdir()
            development_path.write_text("{}\n", encoding="utf-8")
            validation_path.write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Study Test", "-c",
                 "user.email=study@example.invalid", "commit", "-qm", "fixture"],
                cwd=repository, check=True,
            )
            manifest = {
                "outputs": {
                    "raw_results_root": "results/flagship_study",
                    "curated_data": {
                        "development": "curated/development.json",
                        "validation": "curated/validation.json",
                        "test": "curated/test.json",
                    },
                },
                "openings": {"banks": [{"path": "bank.tsv"}]},
            }
            first = studylib._prepare_test_once(
                manifest, repository, manifest_path, "1" * 64,
                selection_path, "2" * 64, "3" * 64,
                destructive_override=False,
            )
            second = studylib._prepare_test_once(
                manifest, repository, manifest_path, "1" * 64,
                selection_path, "2" * 64, "3" * 64,
                destructive_override=False,
            )
            self.assertEqual(first, second)
            marker = (repository / "results/flagship_study" / ("1" * 64) /
                      "test/test-once.json")
            value = json.loads(marker.read_text(encoding="utf-8"))
            value["completed"] = True
            marker.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(studylib.StudyError, "already completed"):
                studylib._prepare_test_once(
                    manifest, repository, manifest_path, "1" * 64,
                    selection_path, "2" * 64, "3" * 64,
                    destructive_override=False,
                )

    def test_flagship_test_requires_a_committed_runtime_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            paths = {
                "manifest": repository / "manifest.json",
                "selection": repository / "selection.json",
                "bank": repository / "bank.tsv",
                "development": repository / "curated/development.json",
                "validation": repository / "curated/validation.json",
                "projection": repository / "curated/runtime-projection.json",
            }
            for name, path in paths.items():
                if name == "projection":
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Study Test", "-c",
                 "user.email=study@example.invalid", "commit", "-qm", "fixture"],
                cwd=repository, check=True,
            )
            paths["projection"].write_text("{}\n", encoding="utf-8")
            manifest = {
                "study": {"study_class": "flagship"},
                "outputs": {
                    "raw_results_root": "results/flagship_study",
                    "runtime_projection": "curated/runtime-projection.json",
                    "curated_data": {
                        "development": "curated/development.json",
                        "validation": "curated/validation.json",
                        "test": "curated/test.json",
                    },
                },
                "openings": {"banks": [{"path": "bank.tsv"}]},
            }
            with self.assertRaisesRegex(
                    studylib.StudyError, "runtime-projection.json"):
                studylib._prepare_test_once(
                    manifest, repository, paths["manifest"], "1" * 64,
                    paths["selection"], "2" * 64, "3" * 64,
                    destructive_override=False,
                )
            marker = (
                repository / "results/flagship_study" / ("1" * 64) /
                "test/test-once.json"
            )
            self.assertFalse(marker.exists())

            subprocess.run(
                ["git", "add", "curated/runtime-projection.json"],
                cwd=repository, check=True,
            )
            subprocess.run(
                ["git", "-c", "user.name=Study Test", "-c",
                 "user.email=study@example.invalid", "commit", "-qm", "projection"],
                cwd=repository, check=True,
            )
            run_id = studylib._prepare_test_once(
                manifest, repository, paths["manifest"], "1" * 64,
                paths["selection"], "2" * 64, "3" * 64,
                destructive_override=False,
            )
            self.assertEqual(len(run_id), 24)
            self.assertTrue(marker.is_file())


class SelectionRuleTests(unittest.TestCase):
    def test_practical_tie_prefers_latency_then_budget_then_id(self) -> None:
        manifest = {
            "candidate_grids": {"mcts": ["mcts-a", "mcts-b", "mcts-c"]},
            "configurations": [
                {"id": "mcts-a", "kind": "mcts", "settings": {"iterations": 1000}},
                {"id": "mcts-b", "kind": "mcts", "settings": {"iterations": 2000}},
                {"id": "mcts-c", "kind": "mcts", "settings": {"iterations": 4000}},
            ],
            "latency_protocol": {"gate_ms": 50},
        }
        validation = {"configurations": {
            "mcts-a": {"strength": {"mean_pair_score": 0.60}, "latency_gate_p95_ms": 20.0},
            "mcts-b": {"strength": {"mean_pair_score": 0.609}, "latency_gate_p95_ms": 25.0},
            "mcts-c": {"strength": {"mean_pair_score": 0.70}, "latency_gate_p95_ms": 51.0},
        }}
        selected, rows = studylib._select_family(manifest, validation, "mcts")
        self.assertEqual(selected, "mcts-a")
        self.assertFalse(next(row for row in rows if row["id"] == "mcts-c")["eligible"])

    def test_no_eligible_family_stops_before_test(self) -> None:
        manifest = {
            "candidate_grids": {"mcts": ["mcts-a"]},
            "configurations": [
                {"id": "mcts-a", "kind": "mcts", "settings": {"iterations": 1000}},
            ],
            "latency_protocol": {"gate_ms": 50},
        }
        validation = {"configurations": {
            "mcts-a": {"strength": {"mean_pair_score": 1.0}, "latency_gate_p95_ms": 50.01},
        }}
        with self.assertRaisesRegex(studylib.StudyError, "stop before test"):
            studylib._select_family(manifest, validation, "mcts")


class FrozenSourceBoundaryTests(unittest.TestCase):
    def test_rank5_adapter_dependency_directory_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            rank5_directory = (
                repository / "submissions/codingame/bots/rank_5"
            )
            rank5_directory.mkdir(parents=True)
            (repository / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8"
            )
            adapter_dependency = rank5_directory / "bot.cpp"
            adapter_dependency.write_text("// frozen adapter\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Study Test", "-c",
                    "user.email=study@example.invalid", "commit", "-qm",
                    "frozen framework",
                ],
                cwd=repository,
                check=True,
            )
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            manifest = {
                "study": {"study_class": "flagship"},
                "source": {"git_commit": source_commit},
                "environment": {
                    "python_version": studylib.platform.python_version(),
                },
            }

            studylib.verify_flagship_source_checkout(manifest, repository)
            wrong_python = copy.deepcopy(manifest)
            wrong_python["environment"]["python_version"] = "0.0.0"
            with self.assertRaisesRegex(studylib.StudyError, "Python interpreter"):
                studylib.verify_flagship_source_checkout(
                    wrong_python, repository
                )
            adapter_dependency.write_text("// changed adapter\n", encoding="utf-8")

            with self.assertRaisesRegex(
                    studylib.StudyError, "framework differs"):
                studylib.verify_flagship_source_checkout(manifest, repository)


if __name__ == "__main__":
    unittest.main()
