from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from benchmarks.game_review_gate import gate


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY / "benchmarks/game_review_gate/manifest.json"


class FrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = gate.validate_manifest(MANIFEST, repository=REPOSITORY)

    def test_frozen_openings_are_complete_and_disjoint(self) -> None:
        context = self.context
        self.assertEqual(len(context.gate_banks), 12)
        self.assertEqual(len(context.excluded_banks), 16)
        self.assertEqual(sum(bank.pairs for bank in context.gate_banks), 700)
        self.assertEqual(
            {(bank.phase, bank.depth) for bank in context.gate_banks},
            {(phase, depth) for phase in gate.PHASES for depth in gate.DEPTHS},
        )
        records = [
            record
            for bank in (*context.excluded_banks, *context.gate_banks)
            for record in bank.records
        ]
        self.assertEqual(len({record.state_hash for record in records}), len(records))
        self.assertEqual(
            len({record.canonical_key for record in records}), len(records)
        )

    def test_schedule_has_exact_frozen_workload(self) -> None:
        development = gate.units_for_phase(self.context, "development")
        validation = gate.units_for_phase(self.context, "validation")
        selection = {"selected_profile_id": "deep-turn-search-200k"}
        test = gate.units_for_phase(self.context, "test", selection)
        self.assertEqual((len(development), len(validation), len(test)), (24, 24, 8))
        self.assertEqual(sum(unit.pairs * 2 for unit in development), 1_200)
        self.assertEqual(sum(unit.pairs * 2 for unit in validation), 2_400)
        self.assertEqual(sum(unit.pairs * 2 for unit in test), 1_600)

    def test_arena_commands_use_distinct_deep_profile_and_fixed_references(self) -> None:
        units = gate.units_for_phase(self.context, "development")
        rank5 = next(
            unit for unit in units if unit.reference_id == "rank5-derived-fixed-50k"
        )
        jacek = next(unit for unit in units if unit.reference_id == "jacek-inspired-20k")
        executable = REPOSITORY / "build/release/papersoccer_arena"
        rank5_command = gate.arena_command(self.context, rank5, executable)
        self.assertEqual(
            rank5_command[rank5_command.index("--candidate-kind") + 1],
            "deep-turn-search",
        )
        self.assertEqual(
            int(
                rank5_command[
                    rank5_command.index("--candidate-complete-turn-max-nodes") + 1
                ]
            ),
            100_000,
        )
        self.assertEqual(
            rank5_command[rank5_command.index("--reference-kind") + 1],
            "rank5-derived",
        )
        jacek_command = gate.arena_command(self.context, jacek, executable)
        self.assertEqual(
            jacek_command[jacek_command.index("--reference-kind") + 1],
            "jacek-inspired",
        )
        self.assertEqual(
            int(
                jacek_command[
                    jacek_command.index("--reference-alpha-beta-max-nodes") + 1
                ]
            ),
            20_000,
        )

    def test_raw_shard_annotation_binds_the_full_frozen_command(self) -> None:
        unit = gate.units_for_phase(self.context, "development")[0]
        executable = (REPOSITORY / "build/release/papersoccer_arena").resolve()
        arena_sha256 = "1" * 64
        source_identity = gate.competition_source_identity(REPOSITORY)
        run_id = gate.sha256_bytes(
            (
                f"{self.context.manifest_sha256}\0development\0"
                f"{source_identity['sha256']}\0{arena_sha256}"
            ).encode("ascii")
        )[:24]
        annotation = {
            "manifest_sha256": self.context.manifest_sha256,
            "run_id": run_id,
            "unit_id": unit.unit_id,
            "phase": unit.phase,
            "matchup_id": unit.matchup_id,
            "candidate_id": unit.candidate_id,
            "reference_id": unit.reference_id,
            "opening_depth": unit.opening_depth,
            "arena_sha256": arena_sha256,
            "arena_provenance": {
                "schema": "papersoccer.arena-build.v1",
                "runtime": "native",
                "build_type": "Release",
                "ndebug": True,
                "sanitizers_enabled": False,
                "compiler_id": "test",
                "compiler_version": "test",
                "configured_flags": "-O3 -DNDEBUG -std=c++20",
                "cxx_standard": 202002,
                "source_commit": "2" * 40,
                "source_dirty": False,
            },
            "arena_command": gate.arena_command(
                self.context, unit, executable
            ),
            "selection_sha256": "none",
            "competition_source": source_identity,
        }
        self.assertIs(
            gate._checked_raw_annotation(self.context, unit, annotation),
            annotation,
        )
        tampered = copy.deepcopy(annotation)
        node_option = tampered["arena_command"].index(
            "--candidate-complete-turn-max-nodes"
        )
        tampered["arena_command"][node_option + 1] = "200000"
        with self.assertRaisesRegex(gate.GateError, "frozen schedule"):
            gate._checked_raw_annotation(self.context, unit, tampered)

    def test_fixed_profile_tampering_is_rejected(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["profiles"]["references"][0]["settings"]["max_nodes"] = 50_001
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "fixed Rank5Derived"):
                gate.validate_manifest(
                    path,
                    repository=REPOSITORY,
                    verify_files=False,
                    verify_identities=False,
                )

    def test_competition_source_identity_is_deterministic_and_checked(self) -> None:
        first = gate.competition_source_identity(REPOSITORY)
        second = gate.competition_source_identity(REPOSITORY)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], gate.COMPETITION_SOURCE_SCHEMA)
        self.assertGreater(first["tracked_files"], 20)
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")

        changed = copy.deepcopy(first)
        changed["sha256"] = "0" * 64
        with self.assertRaisesRegex(gate.GateError, "competition source"):
            gate._checked_competition_source_identity(self.context, changed)


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = gate.validate_manifest(MANIFEST, repository=REPOSITORY)

    def test_whole_pair_bootstrap_is_deterministic_and_stratified(self) -> None:
        strata = {
            4: [0.0, 1.0],
            8: [0.5, 1.0],
            12: [0.0, 0.5],
            20: [0.5, 0.5],
        }
        first = gate.stratified_pair_bootstrap(strata, seed=123456)
        second = gate.stratified_pair_bootstrap(strata, seed=123456)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(first["pairs"], 8)
        self.assertEqual(
            first["method"], "opening_depth_stratified_whole_pair_percentile"
        )
        with self.assertRaisesRegex(gate.GateError, "exactly 10,000"):
            gate.stratified_pair_bootstrap(strata, seed=1, resamples=9_999)

    def test_selection_uses_strength_band_then_latency_then_work(self) -> None:
        candidates = self.context.candidates
        strengths = {
            "deep-turn-search-100k": 0.609,
            "deep-turn-search-200k": 0.600,
            "deep-turn-search-400k": 0.604,
        }
        latency = {
            "deep-turn-search-100k": {"timing_ms": {"p95": 30.0, "maximum": 50.0}},
            "deep-turn-search-200k": {"timing_ms": {"p95": 20.0, "maximum": 50.0}},
            "deep-turn-search-400k": {"timing_ms": {"p95": 20.0, "maximum": 50.0}},
        }
        selected, rows = gate.select_candidate(candidates, strengths, latency)
        self.assertEqual(selected, "deep-turn-search-200k")
        self.assertEqual(sum(row["selected"] for row in rows), 1)
        latency["deep-turn-search-200k"]["timing_ms"]["maximum"] = 751.0
        selected, _ = gate.select_candidate(candidates, strengths, latency)
        self.assertEqual(selected, "deep-turn-search-400k")

    def test_selection_does_not_fall_back_below_the_validation_leader_band(
        self,
    ) -> None:
        strengths = {
            "deep-turn-search-100k": 0.62,
            "deep-turn-search-200k": 0.60,
            "deep-turn-search-400k": 0.59,
        }
        latency = {
            profile["id"]: {"timing_ms": {"p95": 20.0, "maximum": 50.0}}
            for profile in self.context.candidates
        }
        latency["deep-turn-search-100k"]["timing_ms"]["maximum"] = 751.0
        with self.assertRaisesRegex(gate.GateError, "validation leader band"):
            gate.select_candidate(self.context.candidates, strengths, latency)

    def test_fast_and_deep_calibrations_have_distinct_profile_hashes(self) -> None:
        scores = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        outcomes = [0, 1, 0, 0, 1, 0, 1, 1]
        fast = gate.fit_logistic_calibration(
            profile=gate.FAST_ANALYSIS_PROFILE,
            scores=scores,
            outcomes=outcomes,
        )
        deep = gate.fit_logistic_calibration(
            profile=self.context.candidates[0],
            scores=scores,
            outcomes=outcomes,
        )
        self.assertEqual(fast["profile_id"], "complete-turn-analysis-fast-50k")
        self.assertEqual(deep["profile_id"], "deep-turn-search-100k")
        self.assertNotEqual(fast["profile_sha256"], deep["profile_sha256"])
        self.assertEqual(fast["fit_phase"], "validation")

    def test_latency_recomputes_quantiles_and_rejects_failures(self) -> None:
        context = self.context
        selected_records = [
            record
            for depth in gate.DEPTHS
            for bank in context.gate_banks
            if bank.phase == "validation" and bank.depth == depth
            for record in bank.records[:20]
        ]
        sample_ids = [record.opening_id for record in selected_records]
        opening_depths = [record.depth for record in selected_records]
        profiles: dict[str, object] = {}
        for profile_index, profile in enumerate(context.candidates):
            samples = [float(profile_index + index) for index in range(80)]
            transcript = {
                "action": [{"x": 4, "y": 4}],
                "root_score": 7,
                "diagnostics": {
                    key: (False if key == "budget_exhausted" else 7 if key == "root_score" else 1)
                    for key in gate._PARITY_DIAGNOSTIC_FIELDS
                },
            }
            action_hash = gate.sha256_bytes(
                gate.canonical_json_bytes(transcript["action"])
            )
            transcript_hash = gate.sha256_bytes(
                gate.canonical_json_bytes(transcript)
            )
            profiles[profile["id"]] = {
                "profile_sha256": gate.profile_sha256(profile),
                "node_budget": profile["max_nodes"],
                "sample_ids": sample_ids,
                "opening_depths": opening_depths,
                "samples_ms": samples,
                "timing_ms": {
                    "samples": len(samples),
                    "median": gate.nearest_rank(samples, 0.5),
                    "p95": gate.nearest_rank(samples, 0.95),
                    "maximum": max(samples),
                },
                "operational_counts": {
                    "illegal_moves": 0,
                    "incomplete_actions": 0,
                    "unexplained_truncations": 0,
                    "parity_failures": 0,
                },
                "native_shard_sha256": {
                    str(depth): f"{depth:064x}" for depth in gate.DEPTHS
                },
                "parity": [
                    {
                        "sample_id": sample_id,
                        "action_sha256": action_hash,
                        "transcript_sha256": transcript_hash,
                        "wasm_transcript": transcript,
                    }
                    for sample_id in sample_ids
                ],
            }
        value = {
            "schema": gate.LATENCY_SCHEMA,
            "manifest_sha256": context.manifest_sha256,
            "competition_source": gate.competition_source_identity(
                context.repository
            ),
            "module": {
                "path": "web/papersoccer-analysis-wasm.js",
                "sha256": "0" * 64,
                "emscripten_version": "6.0.2",
                "initial_memory_bytes": 67_108_864,
                "memory_growth": False,
            },
            "environment": {
                "runtime": "node-webassembly",
                "node_version": "v24-test",
                "v8_version": "test",
                "platform": "test",
                "architecture": "test",
                "cpu_model": "test",
                "logical_cpus": 1,
                "total_memory_bytes": 1,
                "timer": "performance.now",
                "warmup_searches_per_candidate": 8,
            },
            "sample_source": {
                "phase": "validation",
                "opening_depths": [4, 8, 12, 20],
                "fresh_possession_boundaries_only": True,
                "samples_per_opening_depth_per_candidate": 20,
                "native_parity_reference": "rank5-derived-fixed-50k",
                "native_raw_root": (
                    f"results/game_review_gate/{context.manifest_sha256}/"
                    "validation/shards"
                ),
            },
            "profiles": profiles,
        }
        self.assertIs(
            gate.validate_latency_value(
                context,
                value,
                verify_module=False,
                verify_native_sources=False,
            ),
            value,
        )
        broken = copy.deepcopy(value)
        broken["profiles"]["deep-turn-search-100k"]["timing_ms"]["p95"] += 1
        with self.assertRaisesRegex(gate.GateError, "stale"):
            gate.validate_latency_value(
                context,
                broken,
                verify_module=False,
                verify_native_sources=False,
            )
        broken_parity = copy.deepcopy(value)
        broken_parity["profiles"]["deep-turn-search-100k"]["parity"][0][
            "wasm_transcript"
        ]["action"][0]["x"] += 1
        with self.assertRaisesRegex(gate.GateError, "parity identity is stale"):
            gate.validate_latency_value(
                context,
                broken_parity,
                verify_module=False,
                verify_native_sources=False,
            )

    def test_native_parity_uses_arena_bot_descriptor_objects(self) -> None:
        opening = gate.OpeningRecord(
            opening_id="validation-d4-test",
            phase="validation",
            depth=4,
            generation_seed="1",
            state_hash="2" * 64,
            canonical_key="3" * 64,
            to_move="one",
            moves=((3, 5), (4, 4), (5, 5), (4, 5)),
        )
        native_diagnostics = {
            native_key: (
                False
                if normalized_key == "budget_exhausted"
                else 7
                if normalized_key == "root_score"
                else 1
            )
            for normalized_key, native_key in gate._PARITY_DIAGNOSTIC_FIELDS.items()
        }
        native_diagnostics.update(
            {
                "cached_continuation": False,
                "current_edge_index": 0,
                "profile_node_budget": 100_000,
                "planned_action_length": 1,
            }
        )
        report = {
            "openings": [
                {
                    "pair_index": 0,
                    "opening_id": opening.opening_id,
                    "state_hash": opening.state_hash,
                    "actual_plies": opening.depth,
                    "state": {"to_move": "one"},
                }
            ],
            "games": [
                {
                    "pair_index": 0,
                    "player_one": {"bot": "candidate", "config": {}},
                    "player_two": {"bot": "reference", "config": {}},
                    "decisions": [
                        {
                            "bot": "candidate",
                            "legal": True,
                            "ply": 5,
                            "to": {"x": 4, "y": 4},
                            "deep_turn_search": native_diagnostics,
                        }
                    ],
                }
            ],
        }
        transcript = gate._native_parity_transcript(
            report,
            opening,
            0,
            {"max_nodes": 100_000},
        )
        self.assertEqual(transcript["action"], [{"x": 4, "y": 4}])
        self.assertEqual(transcript["root_score"], 7)

        broken = copy.deepcopy(report)
        broken["games"][0]["player_one"] = "candidate"
        with self.assertRaisesRegex(gate.GateError, "native parity game player_one"):
            gate._native_parity_transcript(
                broken,
                opening,
                0,
                {"max_nodes": 100_000},
            )

    def test_calibration_rejects_nonpositive_oriented_slope(self) -> None:
        with self.assertRaisesRegex(gate.GateError, "slope is not positive"):
            gate.fit_logistic_calibration(
                profile=gate.FAST_ANALYSIS_PROFILE,
                scores=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                outcomes=[1, 0, 1, 1, 0, 1, 0, 0],
            )

    def test_standardized_mapping_conversion_preserves_logit(self) -> None:
        raw_intercept, raw_coefficient = gate._raw_logistic_coefficients(
            {
                "score_mean": 2.0,
                "score_scale": 4.0,
                "intercept": 0.3,
                "slope": 0.8,
            }
        )
        self.assertAlmostEqual(raw_intercept, -0.1)
        self.assertAlmostEqual(raw_coefficient, 0.2)
        raw_score = 7.0
        standardized = 0.3 + 0.8 * ((raw_score - 2.0) / 4.0)
        self.assertAlmostEqual(
            raw_intercept + raw_coefficient * raw_score,
            standardized,
        )


class WorkflowGuardTests(unittest.TestCase):
    def test_web_gate_status_exposes_expert_only_after_a_positive_test(self) -> None:
        compact = {
            "selected_profile_id": "deep-turn-search-200k",
            "test_games": 1600,
            "expert_gate": {
                "passed": False,
                "selector_label": None,
                "strength_status": "strength unresolved",
            },
        }
        negative = gate.render_web_gate_status_bytes(compact).decode("utf-8")
        self.assertIn('"expertOpponentEnabled":false', negative)
        self.assertIn('"selectorLabel":null', negative)
        self.assertNotIn("Expert — DeepTurnSearch", negative)

        positive_compact = copy.deepcopy(compact)
        positive_compact["expert_gate"] = {
            "passed": True,
            "selector_label": "Expert — DeepTurnSearch",
            "strength_status": "validated",
        }
        positive = gate.render_web_gate_status_bytes(positive_compact).decode("utf-8")
        self.assertIn('"expertOpponentEnabled":true', positive)
        self.assertIn('"selectorLabel":"Expert — DeepTurnSearch"', positive)
        self.assertRegex(positive, r'"compactResultSha256":"[0-9a-f]{64}"')

    def test_test_marker_resumes_only_same_unfinished_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            arguments = {
                "manifest_sha256": "1" * 64,
                "selection_sha256": "2" * 64,
                "competition_source_sha256": "3" * 64,
                "arena_sha256": "4" * 64,
            }
            first_id, marker = gate._prepare_test_once(root, **arguments)
            second_id, second_marker = gate._prepare_test_once(root, **arguments)
            self.assertEqual((first_id, marker), (second_id, second_marker))
            value = gate.load_json(marker)
            value["completed"] = True
            gate.write_json(marker, value, replace=True)
            with self.assertRaisesRegex(gate.GateError, "already completed"):
                gate._prepare_test_once(root, **arguments)

    def test_existing_committed_result_blocks_test_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            result = root / "test.json"
            result.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "already exists"):
                gate._prepare_test_once(
                    root / "raw",
                    manifest_sha256="1" * 64,
                    selection_sha256="2" * 64,
                    competition_source_sha256="3" * 64,
                    arena_sha256="4" * 64,
                    committed_test_result=result,
                )


if __name__ == "__main__":
    unittest.main()
