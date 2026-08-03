from __future__ import annotations

import contextlib
import copy
import pathlib
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from benchmarks.flagship_study import ablations, analysis, charts, report, studylib


def _write_json(path: pathlib.Path, value: object) -> None:
    studylib.write_json_atomic(path, value, replace=path.exists())


def _runtime_manifest() -> dict:
    grids = {
        "mcts": [f"mcts-{index}" for index in range(1, 4)],
        "alpha_beta": [f"alpha-{index}" for index in range(1, 4)],
        "jacek_inspired": [f"jacek-{index}" for index in range(1, 4)],
    }
    banks = [
        {
            "id": f"{phase}-d{depth:02d}",
            "phase": phase,
            "depth": depth,
            "pairs": studylib.EXPECTED_PAIR_COUNTS[phase],
            "path": f"openings/{phase}-d{depth:02d}.tsv",
        }
        for phase in studylib.FULL_PHASES
        for depth in studylib.EXPECTED_OPENING_DEPTHS
    ]
    candidates = [identifier for family in studylib.TUNABLE_FAMILIES
                  for identifier in grids[family]]
    return {
        "study": {"study_class": "flagship"},
        "candidate_grids": grids,
        "openings": {"banks": banks},
        "samples": {
            phase: {"color_swapped_pairs_per_depth_matchup": pairs}
            for phase, pairs in studylib.EXPECTED_PAIR_COUNTS.items()
        },
        "schedule": {
            "tuning": [
                {
                    "id": f"tune-{identifier}",
                    "candidate": identifier,
                    "opponent": "rank5-fixed-50k",
                    "phases": ["development", "validation"],
                }
                for identifier in candidates
            ],
            "test": [{"id": f"test-{index}"} for index in range(6)],
        },
    }


def _runtime_samples(manifest: dict) -> list[dict]:
    candidates = [
        identifier
        for family in studylib.TUNABLE_FAMILIES
        for identifier in manifest["candidate_grids"][family]
    ]
    units = {
        (unit.left_config_id, unit.opening_depth): unit
        for unit in studylib.units_for_phase(manifest, "development")
    }
    samples = []
    for index, identifier in enumerate(candidates, start=1):
        for depth, rate in ((4, float(index)), (20, float(index + 1))):
            unit = units[(identifier, depth)]
            games = unit.pairs * 2
            samples.append({
                "unit_id": unit.unit_id,
                "games": games,
                "wall_seconds": rate * games,
                "opening_depth": depth,
                "left_config_id": identifier,
                "right_config_id": unit.right_config_id,
            })
    return samples


class RuntimeProjectionTests(unittest.TestCase):
    def test_exact_representative_coverage_and_projection_arithmetic(self) -> None:
        manifest = _runtime_manifest()
        projection = studylib._project_runtime_from_samples(
            manifest, "a" * 64, _runtime_samples(manifest)
        )

        self.assertEqual(
            projection["schema"],
            studylib.RUNTIME_PROJECTION_SCHEMA_VERSION,
        )
        coverage = projection["coverage"]
        self.assertEqual(coverage["completed_required_units"], 18)
        self.assertEqual(coverage["total_development_units"], 36)
        self.assertEqual(coverage["coverage_fraction"], 0.5)
        self.assertEqual(coverage["observed_games"], 900)
        self.assertEqual(coverage["observed_wall_seconds"], 4950.0)

        workloads = projection["projected_workloads"]
        expected = {
            "remaining_development": (18, 900, 4500.0, 5400.0),
            "full_validation": (36, 3600, 18900.0, 20700.0),
            "full_test": (24, 4800, 44400.0, 93600.0),
            "total_remaining": (78, 9300, 67800.0, 119700.0),
        }
        for name, (units, games, lower, conservative) in expected.items():
            with self.subTest(workload=name):
                actual = workloads[name]
                self.assertEqual(actual["units"], units)
                self.assertEqual(actual["games"], games)
                self.assertEqual(actual["range_seconds"]["lower_proxy"], lower)
                self.assertEqual(
                    actual["range_seconds"]["conservative"], conservative
                )
                self.assertEqual(
                    actual["range_hours"]["conservative"],
                    conservative / 3600.0,
                )

        rates = projection["observed_rates_by_configuration_and_depth"]
        self.assertEqual(rates["mcts-1"]["4"]["seconds_per_game"], 1.0)
        self.assertEqual(rates["mcts-1"]["20"]["seconds_per_game"], 2.0)
        self.assertIn("coverage_gate", projection["assumptions"])

    def test_incomplete_nonrepresentative_or_reduced_budget_samples_fail(self) -> None:
        manifest = _runtime_manifest()
        samples = _runtime_samples(manifest)
        with self.assertRaisesRegex(studylib.StudyError, "missing .*@d20"):
            studylib._project_runtime_from_samples(
                manifest, "a" * 64, samples[:-1]
            )

        reduced = copy.deepcopy(samples)
        reduced[0]["games"] -= 2
        with self.assertRaisesRegex(studylib.StudyError, "exact pair budget"):
            studylib._project_runtime_from_samples(
                manifest, "a" * 64, reduced
            )

        cheap = copy.deepcopy(samples)
        cheap[0]["opening_depth"] = 8
        with self.assertRaisesRegex(studylib.StudyError, "outside preregistered"):
            studylib._project_runtime_from_samples(manifest, "a" * 64, cheap)

    def test_projection_artifact_is_byte_reproducible_from_raw_samples(self) -> None:
        manifest = _runtime_manifest()
        manifest["outputs"] = {
            "runtime_projection": "runtime-projection.json",
        }
        manifest_hash = "a" * 64
        expected = studylib._project_runtime_from_samples(
            manifest, manifest_hash, _runtime_samples(manifest)
        )
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            path = repository / manifest["outputs"]["runtime_projection"]
            _write_json(path, expected)
            with mock.patch.object(
                    studylib, "_runtime_projection_from_raw",
                    return_value=expected):
                validated_path, digest = (
                    studylib._validate_runtime_projection_artifact(
                        manifest, repository, manifest_hash,
                        verify_raw_derivation=True,
                    )
                )
                self.assertEqual(validated_path, path)
                self.assertEqual(digest, studylib.sha256_file(path))

                tampered = copy.deepcopy(expected)
                tampered["coverage"]["observed_wall_seconds"] += 0.001
                _write_json(path, tampered)
                with self.assertRaisesRegex(
                        studylib.StudyError, "not reproducible from raw shards"):
                    studylib._validate_runtime_projection_artifact(
                        manifest, repository, manifest_hash,
                        verify_raw_derivation=True,
                    )

            tampered["unknown"] = True
            _write_json(path, tampered)
            with self.assertRaisesRegex(studylib.StudyError, "unknown"):
                studylib._validate_runtime_projection_artifact(
                    manifest, repository, manifest_hash,
                    verify_raw_derivation=False,
                )


def _selection_manifest() -> dict:
    grids = {
        "mcts": ["mcts-a", "mcts-b", "mcts-c"],
        "alpha_beta": ["alpha-a", "alpha-b", "alpha-c"],
        "jacek_inspired": ["jacek-a", "jacek-b", "jacek-c"],
    }
    configurations = []
    for family in studylib.TUNABLE_FAMILIES:
        for index, identifier in enumerate(grids[family], start=1):
            kind = {
                "mcts": "mcts",
                "alpha_beta": "alpha-beta",
                "jacek_inspired": "jacek-inspired",
            }[family]
            settings = ({"iterations": index * 1000} if kind == "mcts"
                        else {"max_nodes": index * 20_000})
            configurations.append({
                "id": identifier,
                "family": family,
                "kind": kind,
                "settings": settings,
            })
    configurations.append({
        "id": "rank5-fixed-50k",
        "family": "rank5_derived",
        "kind": "rank5-derived",
        "settings": {"max_nodes": 50_000},
    })
    return {
        "configurations": configurations,
        "candidate_grids": grids,
        "openings": {"banks": [{
            "id": "bank-a", "sha256": "b" * 64,
        }]},
        "latency_protocol": {"gate_ms": 50},
        "selection_rule": {
            "practical_tie_percentage_points": 1.0,
            "tie_break_order": ["lower_p95_latency", "smaller_budget"],
        },
        "seeds": {"calibration": {"validation": "6100001"}},
        "outputs": {
            "selection_lock": "selection.json",
            "curated_data": {
                "development": "curated/development.json",
                "validation": "curated/validation.json",
            },
        },
    }


def _calibration_observations(identifier: str) -> dict:
    return {
        "schema": "papersoccer.flagship-calibration-observations.v1",
        "phase": "validation",
        "bot_id": identifier,
        "score_kind": "signed",
        "score_perspective": "player_to_move",
        "decision_count": 10,
        "scores": [-2, -1.5, -1, -0.5, 0, 0.2, 0.5, 1, 1.5, 2],
        "outcomes": [0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        "pair_cluster_ids": [
            f"pair-{index // 2}" for index in range(10)
        ],
        "stratum_ids": [
            "matchup-a:opening-depth-4" if index < 6
            else "matchup-a:opening-depth-8"
            for index in range(10)
        ],
        "excluded": {
            "cached_continuations": 0,
            "truncations": 0,
            "invalid_depths": 0,
        },
    }


def _curated_selection_inputs(manifest: dict, manifest_hash: str) -> tuple[dict, dict]:
    configurations = {}
    development_configurations = {}
    for family_index, family in enumerate(studylib.TUNABLE_FAMILIES):
        for candidate_index, identifier in enumerate(
                manifest["candidate_grids"][family]):
            strength = 0.72 - family_index * 0.04 - candidate_index * 0.03
            configurations[identifier] = {
                "strength": {"mean_pair_score": strength},
                "latency_gate_p95_ms": 10.0 + candidate_index,
            }
            development_configurations[identifier] = {
                "strength": {"mean_pair_score": strength - 0.01},
            }
    configurations["rank5-fixed-50k"] = {
        "fresh_root_latency": {"p95_ms": 45.0},
        "all_edge_latency": {"p95_ms": 20.0},
    }
    observations = {
        identifier: _calibration_observations(identifier)
        for identifier in configurations
    }
    common = {
        "schema_version": studylib.CURATED_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "completeness": {"operationally_valid": True, "truncations": 0},
    }
    development = {
        **common,
        "phase": "development",
        "configurations": development_configurations,
    }
    validation = {
        **common,
        "phase": "validation",
        "source": {"execution_environments": [{"machine": "fixture"}]},
        "configurations": configurations,
        "calibration_observations": observations,
    }
    return development, validation


class SelectionLockIntegrityTests(unittest.TestCase):
    def test_load_strictly_recomputes_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            manifest_path = repository / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            manifest_hash = studylib.sha256_file(manifest_path)
            manifest = _selection_manifest()
            development, validation = _curated_selection_inputs(
                manifest, manifest_hash
            )
            _write_json(
                repository / manifest["outputs"]["curated_data"]["development"],
                development,
            )
            _write_json(
                repository / manifest["outputs"]["curated_data"]["validation"],
                validation,
            )

            with mock.patch.object(
                    studylib, "repository_root_from_manifest",
                    return_value=repository), mock.patch.object(
                    studylib, "validate_manifest", return_value=manifest), \
                    mock.patch.object(
                        studylib, "verify_flagship_source_checkout"):
                studylib.create_selection_lock(manifest_path)

            with mock.patch.object(
                    studylib, "_select_family",
                    wraps=studylib._select_family) as select_family, \
                    mock.patch.object(
                        studylib, "_fit_curated_calibration",
                        wraps=studylib._fit_curated_calibration) as fit_calibration, \
                    mock.patch.object(
                        analysis, "classify_pareto",
                        wraps=analysis.classify_pareto) as classify_pareto:
                loaded = studylib.load_selection_lock(
                    manifest, repository, manifest_hash
                )
            self.assertTrue(loaded["test_authorized"])
            self.assertEqual(select_family.call_count, 3)
            self.assertEqual(fit_calibration.call_count, 4)
            self.assertEqual(classify_pareto.call_count, 1)

            selection_path = repository / manifest["outputs"]["selection_lock"]
            original = studylib.load_json(selection_path)

            def add_unknown(value: dict) -> None:
                value["unregistered_claim"] = True

            def change_metrics(value: dict) -> None:
                first = next(iter(value["validation_metrics"].values()))
                first["validation_strength"] += 0.001

            def change_mapping(value: dict) -> None:
                first = next(iter(value["calibration_mappings"].values()))
                first["intercept"] += 0.001

            def change_pareto(value: dict) -> None:
                value["validation_pareto"][0]["strength"] += 0.001

            def change_rank5(value: dict) -> None:
                value["rank5_latency"]["fresh_root_p95_ms"] += 0.001

            def add_nonflagship_projection(value: dict) -> None:
                value["runtime_projection_sha256"] = "d" * 64

            cases = (
                ("unknown key", add_unknown, "unknown"),
                ("validation metrics", change_metrics, "not reproducible"),
                ("calibration mapping", change_mapping, "not reproducible"),
                ("Pareto classification", change_pareto, "not reproducible"),
                ("Rank5 fields", change_rank5, "changed"),
                ("non-flagship projection", add_nonflagship_projection,
                 "non-flagship"),
            )
            for name, mutate, message in cases:
                with self.subTest(tamper=name):
                    tampered = copy.deepcopy(original)
                    mutate(tampered)
                    _write_json(selection_path, tampered)
                    with self.assertRaisesRegex(studylib.StudyError, message):
                        studylib.load_selection_lock(
                            manifest, repository, manifest_hash
                        )
            _write_json(selection_path, original)

            flagship_manifest = copy.deepcopy(manifest)
            flagship_manifest["study"] = {"study_class": "flagship"}
            frozen_ablations = {
                "schema": ablations.SCHEMA,
                "fixture": "recomputed development/validation contrasts",
            }
            flagship_lock = copy.deepcopy(original)
            flagship_lock["development_validation_ablations"] = frozen_ablations
            flagship_lock["runtime_projection_sha256"] = "d" * 64
            _write_json(selection_path, flagship_lock)
            with mock.patch.object(
                    studylib, "_assert_curated_matches_raw",
                    side_effect=lambda _manifest, _repository, _hash, phase:
                    development if phase == "development" else validation), \
                    mock.patch.object(
                        studylib, "_validate_runtime_projection_artifact",
                        return_value=(repository / "runtime-projection.json",
                                      "d" * 64)), \
                    mock.patch.object(
                    studylib, "_validate_curated_phase_contract"), \
                    mock.patch.object(
                        studylib, "_build_validation_pareto",
                        return_value=original["validation_pareto"]), \
                    mock.patch.object(
                        ablations, "compute", return_value=frozen_ablations):
                studylib.load_selection_lock(
                    flagship_manifest, repository, manifest_hash
                )
                with mock.patch.object(
                        studylib, "_validate_curated_raw_source") as raw_source:
                    studylib.load_selection_lock(
                        flagship_manifest, repository, manifest_hash,
                        verify_raw_derivation=False,
                    )
                self.assertEqual(raw_source.call_count, 2)
                tampered = copy.deepcopy(flagship_lock)
                tampered["development_validation_ablations"]["fixture"] = "changed"
                _write_json(selection_path, tampered)
                with self.assertRaisesRegex(
                        studylib.StudyError, "ablations are not reproducible"):
                    studylib.load_selection_lock(
                        flagship_manifest, repository, manifest_hash
                    )


def _analysis_manifest() -> dict:
    configurations = [
        {
            "id": "mcts-a", "kind": "mcts", "public_label": "MCTS",
            "settings": {"iterations": 1000},
        },
        {
            "id": "alpha-a", "kind": "alpha-beta", "public_label": "Alpha",
            "settings": {"max_nodes": 20_000},
        },
        {
            "id": "jacek-a", "kind": "jacek-inspired", "public_label": "Jacek",
            "settings": {"max_nodes": 20_000},
        },
        {
            "id": "rank5-fixed-50k", "kind": "rank5-derived",
            "public_label": "Rank5", "settings": {"max_nodes": 50_000},
        },
    ]
    return {
        "configurations": configurations,
        "outputs": {
            "selection_lock": "selection.json",
            "curated_data": {
                "development": "curated/development.json",
                "validation": "curated/validation.json",
                "test": "curated/test.json",
            },
            "charts": {
                "bradley_terry": "charts/bradley.svg",
                "pareto": "charts/pareto.svg",
                "calibration": "charts/calibration.svg",
            },
            "report": "REPORT.md",
        },
        "seeds": {"analysis": {"test": "7000001"}},
        "statistics": {
            "bootstrap": {"resamples": 10},
            "bradley_terry": {"minimum_bootstrap_success_fraction": 1.0},
            "calibration": {
                "bins": 10,
                "bootstrap_resamples": 10_000,
                "minimum_bin_successful_resamples": 1_000,
            },
        },
    }


class AnalyzeTestResumptionTests(unittest.TestCase):
    def test_interrupted_rerun_is_idempotent_and_clean_committed_overwrite_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            manifest_path = repository / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            manifest_hash = studylib.sha256_file(manifest_path)
            manifest = _analysis_manifest()
            selection = {
                "selected_configurations": {
                    "mcts": "mcts-a",
                    "alpha_beta": "alpha-a",
                    "jacek_inspired": "jacek-a",
                },
                "fixed_rank5_configuration": "rank5-fixed-50k",
                "calibration_mappings": {
                    identifier: {"mapping": identifier}
                    for identifier in (
                        "mcts-a", "alpha-a", "jacek-a", "rank5-fixed-50k"
                    )
                },
                "validation_pareto": [],
            }
            _write_json(repository / "selection.json", selection)
            common = {
                "schema_version": studylib.CURATED_SCHEMA_VERSION,
                "manifest_sha256": manifest_hash,
                "completeness": {"operationally_valid": True, "truncations": 0},
            }
            for phase in ("development", "validation"):
                _write_json(
                    repository / manifest["outputs"]["curated_data"][phase],
                    {**common, "phase": phase},
                )
            _write_json(
                repository / manifest["outputs"]["curated_data"]["test"],
                {
                    **common,
                    "phase": "test",
                    "binary_games": [],
                    "matchups": {
                        "test-mcts-vs-alpha": {
                            "left_config_id": "mcts-a",
                            "right_config_id": "alpha-a",
                            "pair_bootstrap_95": {"lower": 0.4, "upper": 0.6},
                        },
                    },
                },
            )

            report_calls = 0
            chart_variant = {"bradley": "<svg id='bradley'/>\n"}

            def render_report(*_args: object) -> str:
                nonlocal report_calls
                report_calls += 1
                if report_calls == 1:
                    raise ValueError("simulated interruption")
                return "# Deterministic report\n"

            fitted = types.SimpleNamespace(
                to_dict=lambda: {"abilities": {"mcts-a": 0.0}}
            )
            bootstrap = {
                "intervals": {"mcts-a": {"lower": -0.1, "upper": 0.1}},
                "successful_resamples": 10,
            }
            bot_ids = ("mcts-a", "alpha-a", "jacek-a", "rank5-fixed-50k")
            artifact_paths = [
                repository / manifest["outputs"]["curated_data"]["test"],
                *(repository / path for path in manifest["outputs"]["charts"].values()),
                repository / manifest["outputs"]["report"],
            ]

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    studylib, "repository_root_from_manifest",
                    return_value=repository,
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "validate_manifest", return_value=manifest,
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "verify_flagship_source_checkout",
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "load_selection_lock", return_value=selection,
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "_test_pairs", return_value=[],
                ))
                stack.enter_context(mock.patch.object(
                    analysis, "fit_bradley_terry", return_value=fitted,
                ))
                stack.enter_context(mock.patch.object(
                    analysis, "bootstrap_bradley_terry",
                    side_effect=lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "_curated_calibration_observations",
                    return_value={identifier: {} for identifier in bot_ids},
                ))
                stack.enter_context(mock.patch.object(
                    studylib, "_evaluate_curated_calibration",
                    side_effect=lambda _module, mapping, _payload, **_options: {
                        "mapping": mapping["mapping"], "brier_score": 0.25,
                    },
                ))
                stack.enter_context(mock.patch.object(
                    charts, "bradley_terry_svg",
                    side_effect=lambda *_args: chart_variant["bradley"],
                ))
                stack.enter_context(mock.patch.object(
                    charts, "pareto_svg", return_value="<svg id='pareto'/>\n",
                ))
                stack.enter_context(mock.patch.object(
                    charts, "calibration_svg",
                    return_value="<svg id='calibration'/>\n",
                ))
                stack.enter_context(mock.patch.object(
                    report, "render_report", side_effect=render_report,
                ))

                with self.assertRaisesRegex(
                        studylib.StudyError, "simulated interruption"):
                    studylib.analyze_test(manifest_path)
                self.assertTrue(studylib.load_json(artifact_paths[0])["analysis_complete"])
                self.assertTrue(all(path.is_file() for path in artifact_paths[1:-1]))
                self.assertFalse(artifact_paths[-1].exists())

                recovered = studylib.analyze_test(manifest_path)
                self.assertFalse(recovered["resumed"])
                before = {path: path.read_bytes() for path in artifact_paths}
                repeated = studylib.analyze_test(manifest_path)
                self.assertTrue(repeated["resumed"])
                self.assertEqual(
                    before, {path: path.read_bytes() for path in artifact_paths}
                )

                subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
                subprocess.run(["git", "add", "."], cwd=repository, check=True)
                subprocess.run(
                    [
                        "git", "-c", "user.name=Study Test", "-c",
                        "user.email=study@example.invalid", "commit", "-qm", "fixture",
                    ],
                    cwd=repository,
                    check=True,
                )
                chart_variant["bradley"] = "<svg id='changed'/>\n"
                with self.assertRaisesRegex(
                        studylib.StudyError, "committed analysis artifact"):
                    studylib.analyze_test(manifest_path, replace=True)


if __name__ == "__main__":
    unittest.main()
