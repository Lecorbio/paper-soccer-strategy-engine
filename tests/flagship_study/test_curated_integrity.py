from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest
from collections import defaultdict
from unittest import mock

from benchmarks.flagship_study import studylib


MANIFEST_HASH = "a" * 64


def _manifest() -> dict:
    return {
        "study": {"study_class": "flagship"},
        "openings": {
            "banks": [
                {
                    "id": f"development-d{depth:02d}",
                    "phase": "development",
                    "depth": depth,
                    "pairs": 2,
                    "path": f"openings/development-d{depth:02d}.tsv",
                }
                for depth in (4, 8)
            ],
        },
        "schedule": {
            "tuning": [
                {
                    "id": f"tune-{candidate}",
                    "candidate": candidate,
                    "opponent": "rank5-fixed-50k",
                    "phases": ["development"],
                }
                for candidate in ("mcts-1000", "mcts-2000")
            ],
            "test": [],
        },
        "seeds": {"bootstrap": {"development": "5100001"}},
        "statistics": {"bootstrap": {"resamples": 25}},
    }


def _curated(manifest: dict) -> dict:
    phase = "development"
    units = studylib.units_for_phase(manifest, phase)
    games = []
    winners_by_matchup: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    participants: set[str] = set()
    matchup_participants: dict[str, tuple[str, str]] = {}
    for unit in units:
        participants.update((unit.left_config_id, unit.right_config_id))
        matchup_participants[unit.matchup_id] = (
            unit.left_config_id, unit.right_config_id
        )
        for pair_index in range(unit.pairs):
            opening_id = f"d{unit.opening_depth:02d}-p{pair_index:02d}"
            pair_id = f"{phase}:{unit.matchup_id}:{opening_id}"
            pair_winners = (
                (unit.left_config_id, unit.left_config_id)
                if pair_index == 0
                else (unit.left_config_id, unit.right_config_id)
            )
            for game_in_pair, winner in enumerate(pair_winners):
                loser = (
                    unit.right_config_id
                    if winner == unit.left_config_id
                    else unit.left_config_id
                )
                game_id = f"{pair_id}:g{game_in_pair}"
                games.append({
                    "game_id": game_id,
                    "pair_id": pair_id,
                    "matchup_id": unit.matchup_id,
                    "opening_depth": unit.opening_depth,
                    "winner_config_id": winner,
                    "loser_config_id": loser,
                })
                winners_by_matchup[unit.matchup_id][
                    f"{unit.opening_depth}\0{pair_id}"
                ].append(winner)

    matchup_summaries = {}
    configurations = {
        config_id: {"strength": None} for config_id in sorted(participants)
    }
    paired_scores = {}
    for matchup_id, (left, right) in sorted(matchup_participants.items()):
        summary, strata = studylib._pair_summaries(
            winners_by_matchup[matchup_id], left, right
        )
        summary["pair_bootstrap_95"] = studylib.stratified_pair_bootstrap(
            strata,
            studylib._derived_seed(
                manifest["seeds"]["bootstrap"][phase], phase, matchup_id
            ),
            manifest["statistics"]["bootstrap"]["resamples"],
        )
        summary["by_opening_depth"] = {
            str(depth): {
                "pairs": len(scores),
                "mean_pair_score": sum(scores) / len(scores),
                "pairs_won_2_0": sum(score == 1.0 for score in scores),
                "pairs_split_1_1": sum(score == 0.5 for score in scores),
                "pairs_lost_0_2": sum(score == 0.0 for score in scores),
            }
            for depth, scores in sorted(strata.items())
        }
        matchup_summaries[matchup_id] = summary
        configurations[left]["strength"] = {
            "opponent_config_id": right,
            "mean_pair_score": summary["mean_pair_score"],
            "pair_bootstrap_95": summary["pair_bootstrap_95"],
            "pairs": summary["pairs"],
        }
        opening_ids = []
        opening_depths = []
        scores = []
        prefix = f"{phase}:{matchup_id}:"
        for compound, pair_winners in sorted(
                winners_by_matchup[matchup_id].items()):
            depth_text, pair_id = compound.split("\0", maxsplit=1)
            opening_ids.append(pair_id[len(prefix):])
            opening_depths.append(int(depth_text))
            scores.append(sum(winner == left for winner in pair_winners) / 2.0)
        paired_scores[left] = {
            "phase": phase,
            "bot_id": left,
            "opponent_config_id": right,
            "opening_ids": opening_ids,
            "opening_depths": opening_depths,
            "scores": scores,
        }

    expected_games = len(games)
    return {
        "schema_version": studylib.CURATED_SCHEMA_VERSION,
        "phase": phase,
        "manifest_sha256": MANIFEST_HASH,
        "completeness": {
            "expected_units": len(units),
            "completed_units": len(units),
            "expected_games": expected_games,
            "completed_games": expected_games,
            "unique_game_ids": expected_games,
            "decisions": 100,
            "truncations": 0,
            "operationally_valid": True,
        },
        "matchups": matchup_summaries,
        "configurations": configurations,
        "binary_games": games,
        "paired_scores": paired_scores,
    }


class CuratedPhaseIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.curated = _curated(self.manifest)

    def _validate(self, curated: dict) -> None:
        studylib._validate_curated_phase_contract(
            self.manifest, curated, "development", MANIFEST_HASH
        )

    def test_recomputed_contract_accepts_exact_curated_outcomes(self) -> None:
        self._validate(self.curated)

    def test_count_tampering_is_rejected(self) -> None:
        first_matchup = sorted(self.curated["matchups"])[0]
        first_config = self.curated["matchups"][first_matchup]["left_config_id"]

        cases = []
        missing_game = copy.deepcopy(self.curated)
        missing_game["binary_games"].pop()
        cases.append((missing_game, "binary-game count"))

        wrong_depth_count = copy.deepcopy(self.curated)
        wrong_depth_count["matchups"][first_matchup]["by_opening_depth"]["4"][
            "pairs"
        ] -= 1
        cases.append((wrong_depth_count, "summary differs"))

        missing_paired_row = copy.deepcopy(self.curated)
        missing_paired_row["paired_scores"][first_config]["scores"].pop()
        cases.append((missing_paired_row, "columns are misaligned"))

        for tampered, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    studylib.StudyError, message):
                self._validate(tampered)

    def test_value_tampering_is_rejected_at_every_derived_layer(self) -> None:
        first_matchup = sorted(self.curated["matchups"])[0]
        first_config = self.curated["matchups"][first_matchup]["left_config_id"]

        changed_binary_outcome = copy.deepcopy(self.curated)
        game = changed_binary_outcome["binary_games"][0]
        game["winner_config_id"], game["loser_config_id"] = (
            game["loser_config_id"], game["winner_config_id"]
        )

        changed_matchup_summary = copy.deepcopy(self.curated)
        changed_matchup_summary["matchups"][first_matchup]["left_wins"] -= 1

        changed_strength = copy.deepcopy(self.curated)
        changed_strength["configurations"][first_config]["strength"][
            "mean_pair_score"
        ] -= 0.125

        changed_paired_score = copy.deepcopy(self.curated)
        scores = changed_paired_score["paired_scores"][first_config]["scores"]
        scores[0] = 0.0 if scores[0] != 0.0 else 1.0

        cases = (
            (changed_binary_outcome, "summary differs"),
            (changed_matchup_summary, "summary differs"),
            (changed_strength, "strength .* differs"),
            (changed_paired_score, "paired scores .* differ"),
        )
        for tampered, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    studylib.StudyError, message):
                self._validate(tampered)


class RawCuratedBindingTests(unittest.TestCase):
    @staticmethod
    def _base_payload(phase: str = "development") -> dict:
        return {
            "schema_version": studylib.CURATED_SCHEMA_VERSION,
            "phase": phase,
            "manifest_sha256": MANIFEST_HASH,
            "source": {
                "raw_root": f"results/flagship_study/{MANIFEST_HASH}",
                "units": 1,
                "raw_shard_sha256": {"unit": "b" * 64},
                "execution_environments": [{"machine": "fixture"}],
            },
            "completeness": {
                "expected_units": 1,
                "completed_units": 1,
                "expected_games": 2,
                "completed_games": 2,
                "unique_game_ids": 2,
                "decisions": 2,
                "truncations": 0,
                "operationally_valid": True,
            },
            "matchups": {"matchup": {"left_wins": 1, "left_losses": 1}},
            "configurations": {
                "bot": {
                    "all_edge_latency": {
                        "decisions": 2,
                        "median_ms": 10.0,
                        "p90_ms": 20.0,
                        "p95_ms": 20.0,
                        "p99_ms": 20.0,
                        "maximum_ms": 20.0,
                    },
                    "latency_gate_p95_ms": 20.0,
                    "diagnostics": {"searches": 2, "nodes": 100},
                    "strength": {"mean_pair_score": 0.5},
                },
            },
            "binary_games": [
                {"game_id": "g0", "winner_config_id": "bot"},
                {"game_id": "g1", "winner_config_id": "opponent"},
            ],
            "paired_scores": {"bot": {"scores": [0.5]}},
            "calibration_observations": {
                "bot": {
                    "scores": [0.25, -0.25],
                    "outcomes": [1, 0],
                    "pair_cluster_ids": ["pair", "pair"],
                },
            },
        }

    def _assert_with_mocked_raw(
            self, repository: pathlib.Path, payload: dict,
            expected: dict, *, phase: str = "development",
            allow_analyzed_test: bool = False) -> dict:
        manifest = {
            "outputs": {"curated_data": {phase: f"data/{phase}.json"}}
        }
        path = repository / manifest["outputs"]["curated_data"][phase]
        studylib.write_json_atomic(path, payload, replace=path.exists())
        with mock.patch.object(
                studylib, "_validate_curated_raw_source"), mock.patch.object(
                studylib, "_build_curated_phase", return_value=expected):
            return studylib._assert_curated_matches_raw(
                manifest, repository, MANIFEST_HASH, phase,
                allow_analyzed_test=allow_analyzed_test,
            )

    def test_every_raw_derived_layer_is_bound_to_the_canonical_payload(self) -> None:
        expected = self._base_payload()
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            self._assert_with_mocked_raw(repository, expected, expected)

            mutations = {}

            forged_source = copy.deepcopy(expected)
            forged_source["source"]["raw_shard_sha256"]["unit"] = "c" * 64
            mutations["source hashes"] = forged_source

            forged_gate = copy.deepcopy(expected)
            forged_gate["configurations"]["bot"]["latency_gate_p95_ms"] = 0.001
            mutations["gate latency"] = forged_gate

            forged_quantile = copy.deepcopy(expected)
            forged_quantile["configurations"]["bot"]["all_edge_latency"][
                "median_ms"
            ] = 0.001
            mutations["latency distribution"] = forged_quantile

            forged_diagnostics = copy.deepcopy(expected)
            forged_diagnostics["configurations"]["bot"]["diagnostics"][
                "nodes"
            ] += 1
            mutations["diagnostics"] = forged_diagnostics

            forged_score = copy.deepcopy(expected)
            forged_score["calibration_observations"]["bot"]["scores"][0] = 99.0
            mutations["calibration score"] = forged_score

            forged_outcome = copy.deepcopy(expected)
            forged_outcome["calibration_observations"]["bot"]["outcomes"][0] = 0
            mutations["calibration outcome"] = forged_outcome

            forged_cluster = copy.deepcopy(expected)
            forged_cluster["calibration_observations"]["bot"][
                "pair_cluster_ids"
            ][0] = "other-pair"
            mutations["calibration cluster"] = forged_cluster

            forged_binary = copy.deepcopy(expected)
            forged_binary["binary_games"][0]["winner_config_id"] = "opponent"
            mutations["binary outcome"] = forged_binary

            coherent_rewrite = copy.deepcopy(expected)
            coherent_rewrite["binary_games"][0]["winner_config_id"] = "opponent"
            coherent_rewrite["matchups"]["matchup"] = {
                "left_wins": 0, "left_losses": 2,
            }
            coherent_rewrite["configurations"]["bot"]["strength"][
                "mean_pair_score"
            ] = 0.0
            coherent_rewrite["paired_scores"]["bot"]["scores"] = [0.0]
            mutations["internally coherent outcome rewrite"] = coherent_rewrite

            for name, payload in mutations.items():
                with self.subTest(field=name), self.assertRaisesRegex(
                        studylib.StudyError, "not reproducible from frozen raw shards"):
                    self._assert_with_mocked_raw(repository, payload, expected)

    def test_recorded_source_hashes_match_the_exact_raw_shard_set(self) -> None:
        manifest = _manifest()
        manifest["outputs"] = {"raw_results_root": "results/flagship_study"}
        units = studylib.units_for_phase(manifest, "development")
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            shard_directory = (
                repository / "results/flagship_study" / MANIFEST_HASH /
                "development/shards"
            )
            hashes = {}
            for unit in units:
                path = shard_directory / f"{unit.unit_id}.json"
                studylib.write_json_atomic(
                    path, {"unit_id": unit.unit_id}, replace=False
                )
                hashes[unit.unit_id] = studylib.sha256_file(path)
            curated = {"source": {
                "raw_root": f"results/flagship_study/{MANIFEST_HASH}",
                "units": len(units),
                "raw_shard_sha256": hashes,
                "execution_environments": [{"machine": "fixture"}],
            }}
            studylib._validate_curated_raw_source(
                manifest, repository, MANIFEST_HASH, "development", curated
            )

            changed = shard_directory / f"{units[0].unit_id}.json"
            studylib.write_json_atomic(
                changed, {"unit_id": units[0].unit_id, "changed": True},
                replace=True,
            )
            with self.assertRaisesRegex(studylib.StudyError, "raw shard hash"):
                studylib._validate_curated_raw_source(
                    manifest, repository, MANIFEST_HASH, "development", curated
                )

    def test_exact_analyzed_test_extensions_are_normalized_only_for_recheck(self) -> None:
        expected = self._base_payload("test")
        analyzed = copy.deepcopy(expected)
        analyzed["matchups"]["matchup"]["conclusion"] = {
            "classification": "statistically_unresolved"
        }
        analyzed.update({
            "bradley_terry": {},
            "calibration": {},
            "validation_pareto": [],
            "sample_sizes": {},
            "analysis_complete": True,
        })
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            rebuilt = self._assert_with_mocked_raw(
                repository, analyzed, expected, phase="test",
                allow_analyzed_test=True,
            )
            self.assertEqual(rebuilt, expected)
            self.assertNotIn("analysis_complete", rebuilt)

            unknown = copy.deepcopy(analyzed)
            unknown["unregistered_claim"] = True
            with self.assertRaisesRegex(studylib.StudyError, "unknown"):
                self._assert_with_mocked_raw(
                    repository, unknown, expected, phase="test",
                    allow_analyzed_test=True,
                )

if __name__ == "__main__":
    unittest.main()
