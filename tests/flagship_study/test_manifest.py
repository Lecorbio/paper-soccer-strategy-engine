from __future__ import annotations

import copy
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from benchmarks.flagship_study import prepare_manifest, studylib


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


def fake_banks() -> list[dict[str, object]]:
    predecessor = studylib.load_json(
        REPOSITORY / studylib.V4_PREDECESSOR_MANIFEST_PATH
    )
    previous = {
        (bank["phase"], bank["depth"]): bank
        for bank in predecessor["openings"]["banks"]
    }
    result: list[dict[str, object]] = []
    for phase in studylib.FULL_PHASES:
        for depth in studylib.EXPECTED_OPENING_DEPTHS:
            if phase in ("development", "test"):
                result.append(copy.deepcopy(previous[(phase, depth)]))
                continue
            identifier = f"openings-{phase}-d{depth:02d}"
            result.append({
                "id": identifier,
                "phase": phase,
                "depth": depth,
                "pairs": studylib.EXPECTED_PAIR_COUNTS[phase],
                "path": f"unused/v4-{identifier}.tsv",
                "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
                "seed": prepare_manifest.OPENING_SEEDS[phase][depth],
            })
    return result


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


class ManifestValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        cls.manifest = prepare_manifest.build_manifest(
            REPOSITORY, commit, fake_banks(), "2026-08-03T00:00:00+00:00"
        )

    def validate(self, manifest: dict) -> None:
        studylib.validate_manifest(manifest, REPOSITORY, verify_files=False)

    def test_frozen_flagship_manifest_contract_is_valid(self) -> None:
        self.validate(copy.deepcopy(self.manifest))

    def test_failed_v3_banks_remain_byte_for_byte_audit_inputs(self) -> None:
        banks = prepare_manifest.reuse_frozen_banks(REPOSITORY)
        self.assertEqual(len(banks), 12)
        self.assertEqual(sum(bank["pairs"] for bank in banks), 700)
        for bank in banks:
            path = REPOSITORY / bank["path"]
            self.assertEqual(bank["sha256"], studylib.sha256_file(path))
            self.assertEqual(
                bank["seed"],
                prepare_manifest.V3_OPENING_SEEDS[bank["phase"]][bank["depth"]],
            )

    def test_v4_keeps_test_seeds_and_replaces_validation_seeds(self) -> None:
        self.assertEqual(
            prepare_manifest.OPENING_SEEDS["development"],
            prepare_manifest.V3_OPENING_SEEDS["development"],
        )
        self.assertEqual(
            prepare_manifest.OPENING_SEEDS["test"],
            prepare_manifest.V3_OPENING_SEEDS["test"],
        )
        self.assertTrue(all(
            prepare_manifest.OPENING_SEEDS["validation"][depth]
            != prepare_manifest.V3_OPENING_SEEDS["validation"][depth]
            for depth in studylib.EXPECTED_OPENING_DEPTHS
        ))
        self.assertEqual(
            self.manifest["study"]["id"],
            "competitive-demo-bots-flagship-2026-v4",
        )

    def test_v4_supersession_identity_and_failure_record_are_bound(self) -> None:
        predecessor = copy.deepcopy(self.manifest)
        predecessor["supersession"]["predecessor_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(studylib.StudyError, "lineage"):
            self.validate(predecessor)

        failure = copy.deepcopy(self.manifest)
        failure["supersession"]["failure_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(studylib.StudyError, "failure record"):
            self.validate(failure)

    def test_v4_requires_exact_reused_development_and_test_banks(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        bank = next(
            value for value in manifest["openings"]["banks"]
            if value["phase"] == "test" and value["depth"] == 4
        )
        bank["sha256"] = "0" * 64
        with self.assertRaisesRegex(studylib.StudyError, "exact predecessor test"):
            self.validate(manifest)

    def test_v4_rejects_reused_validation_bank_and_any_old_seed(self) -> None:
        predecessor = studylib.load_json(
            REPOSITORY / studylib.V4_PREDECESSOR_MANIFEST_PATH
        )
        previous_bank = next(
            value for value in predecessor["openings"]["banks"]
            if value["phase"] == "validation" and value["depth"] == 4
        )
        reused = copy.deepcopy(self.manifest)
        current_bank = next(
            value for value in reused["openings"]["banks"]
            if value["phase"] == "validation" and value["depth"] == 4
        )
        current_bank.update(copy.deepcopy(previous_bank))
        reused["seeds"]["opening"]["validation"]["4"] = previous_bank["seed"]
        with self.assertRaisesRegex(studylib.StudyError, "prospectively fresh"):
            self.validate(reused)

        old_seed = copy.deepcopy(self.manifest)
        old_seed["seeds"]["bot"]["validation"] = \
            predecessor["seeds"]["bot"]["validation"]
        with self.assertRaisesRegex(studylib.StudyError, "globally fresh"):
            self.validate(old_seed)

        changed_test_seed = copy.deepcopy(self.manifest)
        changed_test_seed["seeds"]["analysis"]["test"] = "123456789"
        with self.assertRaisesRegex(studylib.StudyError, "frozen test analysis"):
            self.validate(changed_test_seed)

    def test_v4_preparation_mode_is_mandatory(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "benchmarks/flagship_study/prepare_manifest.py"),
                "--opening-tool", "unused",
                "--source-commit", "a" * 40,
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("--fresh-validation-keep-frozen-test", process.stderr)

    def test_fresh_validation_command_excludes_every_prior_bank(self) -> None:
        exclusions = [pathlib.Path(f"old-{index}.tsv") for index in range(12)]
        exclusions += [pathlib.Path(f"new-{index}.tsv") for index in range(3)]

        command = prepare_manifest.fresh_validation_command(
            pathlib.Path("opening-tool"),
            depth=20,
            pairs=50,
            seed=prepare_manifest.OPENING_SEEDS["validation"][20],
            excluded_paths=exclusions,
        )

        self.assertEqual(command.count("--exclude-bank"), 15)
        self.assertEqual(
            [
                pathlib.Path(command[index + 1])
                for index, value in enumerate(command)
                if value == "--exclude-bank"
            ],
            exclusions,
        )

    def test_failed_fresh_generation_publishes_no_partial_bank_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            old_banks = []
            for phase in studylib.FULL_PHASES:
                for depth in studylib.EXPECTED_OPENING_DEPTHS:
                    old_banks.append({
                        "id": f"openings-{phase}-d{depth:02d}",
                        "phase": phase,
                        "depth": depth,
                        "pairs": studylib.EXPECTED_PAIR_COUNTS[phase],
                        "path": f"old/{phase}_d{depth:02d}.tsv",
                        "sha256": "0" * 64,
                        "seed": prepare_manifest.V3_OPENING_SEEDS[phase][depth],
                    })

            def records(path: pathlib.Path) -> list[studylib.OpeningRecord]:
                staged = ".validation-v4-stage." in str(path)
                count = 50 if staged else 1
                prefix = "new" if staged else str(path)
                return [
                    studylib.OpeningRecord(
                        opening_id=f"{prefix}-{index}",
                        phase="validation",
                        depth=4,
                        generation_seed="1",
                        state_hash=hashlib.sha256(
                            f"state-{prefix}-{index}".encode()
                        ).hexdigest(),
                        canonical_key=hashlib.sha256(
                            f"canonical-{prefix}-{index}".encode()
                        ).hexdigest(),
                        to_move="one",
                        moves=(),
                    )
                    for index in range(count)
                ]

            processes = [
                subprocess.CompletedProcess([], 0, stdout=b"staged", stderr=b""),
                subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"failed"),
            ]
            with mock.patch.object(
                    prepare_manifest, "reuse_frozen_banks", return_value=old_banks), \
                 mock.patch.object(
                    studylib, "parse_opening_bank", side_effect=records), \
                 mock.patch.object(
                    prepare_manifest.subprocess, "run", side_effect=processes):
                with self.assertRaisesRegex(
                        studylib.StudyError, "failed at depth 8"):
                    prepare_manifest.generate_fresh_validation_banks(
                        repository, pathlib.Path("opening-tool")
                    )

            opening_directory = repository / "benchmarks/flagship_study/openings"
            self.assertEqual(list(opening_directory.glob("validation_v4_*.tsv")), [])
            self.assertEqual(list(opening_directory.glob(".validation-v4-stage.*")), [])

    def test_missing_and_unknown_fields_are_rejected(self) -> None:
        missing = copy.deepcopy(self.manifest)
        del missing["rules"]
        with self.assertRaisesRegex(studylib.StudyError, "missing"):
            self.validate(missing)
        unknown = copy.deepcopy(self.manifest)
        unknown["surprise"] = True
        with self.assertRaisesRegex(studylib.StudyError, "unknown"):
            self.validate(unknown)

    def test_duplicate_ids_and_bank_hashes_are_rejected(self) -> None:
        duplicate = copy.deepcopy(self.manifest)
        duplicate["configurations"][1]["id"] = duplicate["configurations"][0]["id"]
        with self.assertRaisesRegex(studylib.StudyError, "duplicate ID"):
            self.validate(duplicate)
        duplicate_hash = copy.deepcopy(self.manifest)
        duplicate_hash["openings"]["banks"][1]["sha256"] = \
            duplicate_hash["openings"]["banks"][0]["sha256"]
        with self.assertRaisesRegex(studylib.StudyError, "distinct SHA-256"):
            self.validate(duplicate_hash)

    def test_overlapping_phase_seeds_are_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"]["bot"]["test"] = manifest["seeds"]["bot"]["validation"]
        with self.assertRaisesRegex(studylib.StudyError, "overlapping phase seeds"):
            self.validate(manifest)

    def test_bank_seed_must_match_the_phase_seed_registry(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["openings"]["banks"][0]["seed"] = "999999"
        with self.assertRaisesRegex(studylib.StudyError, "differs from seeds.opening"):
            self.validate(manifest)

    def test_frozen_source_must_be_clean_and_paths_cannot_escape(self) -> None:
        dirty = copy.deepcopy(self.manifest)
        dirty["source"]["dirty_worktree"] = True
        with self.assertRaisesRegex(studylib.StudyError, "dirty source"):
            self.validate(dirty)
        escaping = copy.deepcopy(self.manifest)
        escaping["outputs"]["report"] = "benchmarks/flagship_study/../../escape.md"
        with self.assertRaisesRegex(studylib.StudyError, r"without '\.\.'"):
            self.validate(escaping)

    def test_family_kind_role_combinations_are_strict(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["configurations"][0]["family"] = "alpha_beta"
        manifest["candidate_grids"]["mcts"].remove("mcts-1000")
        manifest["candidate_grids"]["alpha_beta"].insert(0, "mcts-1000")
        with self.assertRaisesRegex(studylib.StudyError, "family/kind/role"):
            self.validate(manifest)

    def test_test_schedule_must_be_the_complete_round_robin(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["schedule"]["test"][0]["left_slot"] = \
            manifest["schedule"]["test"][1]["left_slot"]
        manifest["schedule"]["test"][0]["right_slot"] = \
            manifest["schedule"]["test"][1]["right_slot"]
        with self.assertRaisesRegex(studylib.StudyError, "complete four-bot round robin"):
            self.validate(manifest)

    def test_unsupported_rules_are_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["rules"]["goal_rule"] = "own_goals_allowed"
        with self.assertRaisesRegex(studylib.StudyError, "unsupported rules"):
            self.validate(manifest)

    def test_rank5_profile_is_exactly_locked(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        rank5 = next(config for config in manifest["configurations"]
                     if config["kind"] == "rank5-derived")
        rank5["settings"]["max_nodes"] = 49_999
        with self.assertRaisesRegex(studylib.StudyError, "fixed 50k"):
            self.validate(manifest)

    def test_incompatible_mcts_profile_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["configurations"][0]["settings"]["leaf_policy"] = "tactical_quiescence"
        with self.assertRaisesRegex(studylib.StudyError, "rollout-only"):
            self.validate(manifest)

    def test_full_schedule_counts_are_exact(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        self.validate(manifest)
        development = studylib.units_for_phase(manifest, "development")
        validation = studylib.units_for_phase(manifest, "validation")
        selection = {"selected_configurations": {
            "mcts": "mcts-1000",
            "alpha_beta": "alpha-beta-20k",
            "jacek_inspired": "jacek-20k",
        }}
        test = studylib.units_for_phase(manifest, "test", selection)
        self.assertEqual(len(development), 36)
        self.assertEqual(len(validation), 36)
        self.assertEqual(len(test), 24)
        self.assertEqual(sum(unit.pairs * 2 for unit in test), 4800)

    def test_release_provenance_requires_an_unsanitized_build(self) -> None:
        source_commit = "a" * 40
        provenance = valid_build_provenance(source_commit)
        self.assertEqual(
            prepare_manifest.validate_release_build_provenance(
                provenance, source_commit
            ),
            provenance,
        )

        sanitized = copy.deepcopy(provenance)
        sanitized["sanitizers_enabled"] = True
        with self.assertRaisesRegex(
                studylib.StudyError, "optimized native Release arena"):
            prepare_manifest.validate_release_build_provenance(
                sanitized, source_commit
            )

    def test_release_provenance_has_an_exact_schema(self) -> None:
        source_commit = "a" * 40
        missing = valid_build_provenance(source_commit)
        del missing["sanitizers_enabled"]
        with self.assertRaisesRegex(studylib.StudyError, "missing"):
            prepare_manifest.validate_release_build_provenance(
                missing, source_commit
            )

        unknown = valid_build_provenance(source_commit)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(studylib.StudyError, "unknown"):
            prepare_manifest.validate_release_build_provenance(
                unknown, source_commit
            )


class OpeningBankParserTests(unittest.TestCase):
    def bank_text(self, second_key: str = "b" * 64) -> str:
        return "\n".join([
            "schema\tpapersoccer.opening-bank.v1",
            "phase\tdevelopment",
            "depth\t1",
            "pairs\t2",
            "rules\t8x10;opponent_goal_only;player_to_move_loses",
            "generator\tuniform-legal-move-generator/v1",
            "generator_seed\t101",
            "selection\tsplitmix64-unbiased-rejection-sampling/v1",
            "state_hash_algorithm\tsha256-canonical-game-state/v1",
            "canonicalization\thorizontal-reflection-min-serialization-sha256/v1",
            "opening_ply_definition\tone physical selected edge, including rebound edges",
            "opening_id\tphase\tdepth\tgeneration_seed\tstate_hash\tcanonical_key\tto_move\tmoves",
            f"development-d01-p000\tdevelopment\t1\t201\t{'1' * 64}\t{'a' * 64}\ttwo\t4,5",
            f"development-d01-p001\tdevelopment\t1\t202\t{'2' * 64}\t{second_key}\ttwo\t5,6",
            "",
        ])

    def test_transcripts_parse_and_keep_exact_physical_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bank.tsv"
            path.write_text(self.bank_text(), encoding="utf-8")
            records = studylib.parse_opening_bank(path)
            self.assertEqual([record.depth for record in records], [1, 1])
            self.assertEqual(records[0].moves, ((4, 5),))
            self.assertEqual(records[0].to_move, "two")

    def test_canonical_duplicates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bank.tsv"
            path.write_text(self.bank_text(second_key="a" * 64), encoding="utf-8")
            with self.assertRaisesRegex(studylib.StudyError, "canonically equivalent"):
                studylib.parse_opening_bank(path)


class DeterministicUtilityTests(unittest.TestCase):
    def test_latency_uses_nearest_rank_quantiles(self) -> None:
        summary = studylib.latency_summary([1_000_000 * value for value in range(1, 101)])
        self.assertEqual(summary["median_ms"], 50.0)
        self.assertEqual(summary["p90_ms"], 90.0)
        self.assertEqual(summary["p95_ms"], 95.0)
        self.assertEqual(summary["p99_ms"], 99.0)
        self.assertEqual(summary["maximum_ms"], 100.0)

    def test_stratified_pair_bootstrap_is_deterministic(self) -> None:
        strata = {4: [0.0] * 5, 20: [1.0] * 5}
        first = studylib.stratified_pair_bootstrap(strata, seed=77, resamples=100)
        second = studylib.stratified_pair_bootstrap(strata, seed=77, resamples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["lower"], 0.5)
        self.assertEqual(first["upper"], 0.5)

    def test_pareto_frontier_marks_strict_dominance(self) -> None:
        points = [
            {"id": "fast-strong", "p95_ms": 10.0, "strength": 0.7},
            {"id": "slow-weak", "p95_ms": 20.0, "strength": 0.6},
            {"id": "slower-stronger", "p95_ms": 30.0, "strength": 0.8},
        ]
        by_id = {point["id"]: point for point in studylib.pareto_frontier(points)}
        self.assertTrue(by_id["fast-strong"]["pareto_optimal"])
        self.assertFalse(by_id["slow-weak"]["pareto_optimal"])
        self.assertTrue(by_id["slower-stronger"]["pareto_optimal"])

    def test_atomic_writer_refuses_silent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "value.json"
            studylib.write_json_atomic(path, {"value": 1}, replace=False)
            with self.assertRaisesRegex(studylib.StudyError, "refusing to overwrite"):
                studylib.write_json_atomic(path, {"value": 2}, replace=False)


if __name__ == "__main__":
    unittest.main()
