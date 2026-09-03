import copy
import hashlib
import json
import math
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402
import jacek_replay_pack as pack  # noqa: E402


SHORT_WIN = [
    {"player_id": 0, "action": "0"},
    {"player_id": 1, "action": "0"},
    {"player_id": 0, "action": "3"},
    {"player_id": 1, "action": "0"},
    {"player_id": 0, "action": "61"},
    {"player_id": 1, "action": "0"},
    {"player_id": 0, "action": "07"},
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(corpus.canonical_json_bytes(value))


class JacekReplayCorpusTests(unittest.TestCase):
    @staticmethod
    def rank4_teacher_row(**overrides):
        row = {
            "schema": corpus.RANK4_TEACHER_SCHEMA,
            "campaign_id": "selfsearch-pilot-fixture",
            "position_id": "position:" + "4" * 64,
            "root_group_id": "root:rank4",
            "group_id": "continuation:rank4",
            "source": "fixture-rank4",
            "split": "train",
            "winner": 0,
            "prefix": [{"player_id": 0, "action": "0"}],
            "mover": 1,
            "teacher": {
                "kind": "rank4-fixed-work",
                "source_sha256": "5" * 64,
            },
            "search_config": {
                "max_nodes": 32_000,
                "max_time_ms": 0,
                "max_turn_depth": 32,
                "replay_value_blend_percent": 15,
                "teacher_residual_weight_percent": 100,
            },
            "search_stats": {
                "attempted_depth": 1,
                "completed_depth": 0,
                "nodes": 32_000,
                "leaf_evaluations": 64,
                "terminal_nodes": 9_922,
                "completed_actions": 9_996,
                "budget_exhausted": True,
                "node_cap_reached": True,
                "depth_cap_reached": False,
                "deadline_reached": False,
                "termination_reason": "fixed-work-cap",
            },
            "root_score": 26_407,
            "completed_depth": 0,
            "nodes": 32_000,
            "root_solved": False,
            "proven_winner": None,
            "weight": 1.0,
        }
        row.update(overrides)
        return row

    @staticmethod
    def search_teacher_row(**overrides):
        row = {
            "schema": corpus.SEARCH_TEACHER_SCHEMA,
            "campaign_id": "selfsearch-pilot-fixture",
            "position_id": "position:" + "1" * 64,
            "root_group_id": "root:search",
            "group_id": "continuation:search",
            "source": "fixture-search",
            "split": "train",
            "winner": 0,
            "prefix": [{"player_id": 0, "action": "0"}],
            "mover": 1,
            "teacher": {
                "kind": "jacek_replay_bfm_search",
                "source_sha256": "2" * 64,
                "model_sha256": "3" * 64,
                "feature_schema": features.FEATURE_SCHEMA,
                "feature_schema_sha256": hashlib.sha256(
                    features.FEATURE_SCHEMA.encode()
                ).hexdigest(),
            },
            "search_config": {
                "seed": 7,
                "max_time_ms": 0,
                "max_tree_nodes": 64_000,
                "max_actions": 1_000_000,
                "max_partial_paths": 1_000_000,
                "exploration": 0.5,
                "fpu": 0.5,
            },
            "search_stats": {
                "expansions": 10,
                "generated_actions": 20,
                "retained_actions": 19,
                "neural_evaluations": 8,
                "visits": 63_999,
                "completed_actions": 8,
                "duplicate_boundaries": 1,
                "partial_paths": 2,
                "fifo_extractions": 0,
                "lifo_extractions": 2,
                "tactical_proofs": 0,
                "tactical_solutions": 0,
                "truncations": 0,
                "generation_action_cap_stops": 0,
                "generation_partial_cap_stops": 0,
                "generation_deadline_stops": 0,
                "materialization_deadline_stops": 0,
                "generation_queue_drops": 0,
                "generation_retention_drops": 0,
                "generation_boundary_replacements": 0,
                "generation_tactical_shortcuts": 0,
                "generation_fallbacks": 0,
                "generation_frontier_resumptions": 0,
                "generation_zero_action_resumptions": 0,
                "generation_max_frontier_depth": 1,
                "progressive_widenings": 0,
                "closed_unsolved_nodes": 0,
                "closed_unsolved_nonexhaustive_nodes": 0,
                "open_unexpanded_nodes": 10,
                "implicit_action_frontiers": 0,
                "max_open_children": 19,
                "tree_nodes": 64_000,
                "max_complete_turn_depth": 5,
                "deadline_reached": False,
                "tree_cap_reached": True,
                "termination_reason": "fixed-work-cap",
            },
            "teacher_value": 0.8,
            "root_solved": False,
            "proven_winner": None,
            "weight": 1.0,
        }
        row.update(overrides)
        return row

    @staticmethod
    def complete_turn_action_group_row(
        *, position_id="position:" + "a" * 64, split="train", root_action="0"
    ):
        campaign_id = "complete-turn-label-fixture"
        max_tree_nodes = 64_000
        seed_material = f"{campaign_id}\0{position_id}\0{max_tree_nodes}".encode()
        root = features.ReplayState()
        features.apply_complete_turn(root, 0, root_action)
        successor = copy.deepcopy(root)
        # The parent mover is Player Two, so canonical direction 4 maps back
        # to physical direction 0.
        features.apply_complete_turn(successor, 1, "0")
        teacher = {
            "kind": "jacek_replay_bfm_search",
            "artifact_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "feature_schema_sha256": hashlib.sha256(
                features.FEATURE_SCHEMA.encode()
            ).hexdigest(),
            "source_sha256": "3" * 64,
        }
        return {
            "schema": corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA,
            "feature_schema": features.FEATURE_SCHEMA,
            "source_bundle_body_sha256": "4" * 64,
            "teacher": teacher,
            "ranking": {
                "complete_turn_boundaries": True,
                "teacher_value_frame": "explicit-mover-relative",
                "successor_aliases": "canonical-boundary-state",
                "best_tie_break": "successor-id-ascending",
            },
            "split": split,
            "group": {
                "group_id": corpus._mover_canonical_position_identity(root),
                "parent_identity": corpus._mover_canonical_position_identity(root),
                "identity_algorithm": "sha256-mover-canonical-boundary-v1",
                "parent_mover": 1,
                "root_value": 0.2,
                "root_solved": False,
                "proven_winner": None,
                "termination_reason": "fixed-work-cap",
                "successors_exhaustive": True,
                "work_budget": {
                    "seed": int(hashlib.sha256(seed_material).hexdigest()[:16], 16),
                    "max_time_ms": 0,
                    "max_tree_nodes": max_tree_nodes,
                    "max_actions": 250,
                    "max_partial_paths": 50_000,
                    "exploration": 0.5,
                    "fpu": 0.5,
                },
                "source_binding": {
                    "campaign_id": campaign_id,
                    "position_id": position_id,
                    "root_group_id": "root:fixture",
                    "group_id": "continuation:fixture",
                    "source": "fixture",
                    "split": split,
                    "winner": 0,
                    "prefix": [{"player_id": 0, "action": root_action}],
                },
                "successors": [
                    {
                        "successor_id": corpus._mover_canonical_position_identity(
                            successor
                        ),
                        "active": list(features.encode_active(successor)),
                        "transcript": "4",
                        "teacher_value": -0.3,
                        "value_mover": 0,
                        "proof": {"solved": False, "proven_winner": None},
                        "termination": {
                            "reason": "fixed-work-cap",
                            "value_status": "backed-up-at-root-termination",
                        },
                        "visits": 1,
                        "selection_visits": 0,
                    }
                ],
            },
        }

    def test_complete_turn_action_group_is_strict_and_mover_relative(self):
        row = self.complete_turn_action_group_row()
        self.assertEqual(corpus.validate_complete_turn_action_group(row), row)
        for mutate, message in (
            (
                lambda value: value["group"]["successors"][0]["active"].append(6301),
                "features disagree|outside",
            ),
            (
                lambda value: value["group"]["successors"][0]["proof"].update(
                    solved=True
                ),
                "proof disagrees",
            ),
            (
                lambda value: value["group"]["work_budget"].update(seed=1),
                "seed is not bound",
            ),
            (
                lambda value: value.update(split="test"),
                "row is malformed",
            ),
        ):
            broken = copy.deepcopy(row)
            mutate(broken)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                corpus.validate_complete_turn_action_group(broken)

    def test_complete_turn_action_group_remains_a_scalar_value_label(self):
        row = self.complete_turn_action_group_row()
        sample, reflected = corpus.sample_from_teacher_row(row)
        self.assertAlmostEqual(sample.target, -0.1)
        self.assertEqual(reflected.active, features.reflect_active(sample.active))
        self.assertEqual(sample.group_id, "root:fixture")
        self.assertEqual(sample.lineages, reflected.lineages)
        lineage = sample.lineages[0]
        self.assertEqual(lineage.schema, corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA)
        self.assertEqual(lineage.position_id, row["group"]["source_binding"]["position_id"])
        self.assertEqual(lineage.group_id, "continuation:fixture")
        self.assertEqual(lineage.root_group_id, "root:fixture")
        policy = corpus.target_policy_for_schema(
            corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA
        )
        self.assertEqual(policy["teacher_value"]["transform"], "identity")

    def test_deep_action_groups_replace_shallow_by_canonical_group(self):
        first = self.complete_turn_action_group_row(
            position_id="position:" + "a" * 64, root_action="0"
        )
        second = self.complete_turn_action_group_row(
            position_id="position:" + "b" * 64, root_action="1"
        )
        deep = copy.deepcopy(second)
        deep["group"]["root_value"] = 0.6
        deep["group"]["work_budget"]["max_tree_nodes"] = 500_000
        source = deep["group"]["source_binding"]
        material = (
            f"{source['campaign_id']}\0{source['position_id']}\0{500_000}".encode()
        )
        deep["group"]["work_budget"]["seed"] = int(
            hashlib.sha256(material).hexdigest()[:16], 16
        )
        merged = corpus.merge_complete_turn_action_groups([second, first], [deep])
        self.assertEqual(len(merged), 2)
        by_id = {row["group"]["group_id"]: row for row in merged}
        self.assertEqual(by_id[second["group"]["group_id"]]["group"]["root_value"], 0.6)
        self.assertEqual(by_id[first["group"]["group_id"]], first)
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            shallow_path = directory / "shallow.jsonl"
            deep_path = directory / "deep.jsonl"
            shallow_path.write_bytes(
                corpus.canonical_json_bytes(second)
                + corpus.canonical_json_bytes(first)
            )
            deep_path.write_bytes(corpus.canonical_json_bytes(deep))
            self.assertEqual(
                corpus.merge_complete_turn_action_group_files(
                    shallow=shallow_path, deep=deep_path
                ),
                b"".join(corpus.canonical_json_bytes(row) for row in merged),
            )
        with self.assertRaisesRegex(ValueError, "strict subset"):
            corpus.merge_complete_turn_action_groups([second], [deep])

    def test_complete_turn_action_groups_aggregate_deterministically(self):
        later = self.complete_turn_action_group_row(
            position_id="position:" + "b" * 64, root_action="1"
        )
        later["group"]["successors_exhaustive"] = False
        earlier = self.complete_turn_action_group_row(position_id="position:" + "a" * 64)
        validation = self.complete_turn_action_group_row(
            position_id="position:" + "c" * 64,
            split="validation",
            root_action="2",
        )
        artifact = corpus.build_complete_turn_successor_labels(
            [later, validation, earlier]
        )
        self.assertEqual(
            artifact["schema"], corpus.COMPLETE_TURN_SUCCESSOR_LABELS_SCHEMA
        )
        self.assertFalse(artifact["protected_tests_opened"])
        self.assertEqual(
            [group["group_id"] for group in artifact["splits"]["train"]],
            sorted([earlier["group"]["group_id"], later["group"]["group_id"]]),
        )
        by_id = {
            group["group_id"]: group for group in artifact["splits"]["train"]
        }
        self.assertFalse(
            by_id[later["group"]["group_id"]]["successors_exhaustive"]
        )
        body = dict(artifact)
        body_hash = body.pop("body_sha256")
        self.assertEqual(body_hash, corpus.sha256_bytes(corpus.canonical_json_bytes(body)))
        self.assertEqual(
            artifact,
            corpus.build_complete_turn_successor_labels([earlier, later, validation]),
        )
        self.assertEqual(
            artifact, corpus.validate_complete_turn_successor_labels(artifact)
        )
        tampered = copy.deepcopy(artifact)
        tampered["splits"]["train"][0]["root_value"] = 0.9
        with self.assertRaisesRegex(ValueError, "body SHA-256 mismatch"):
            corpus.validate_complete_turn_successor_labels(tampered)

    def make_sources(self, directory):
        repository = pathlib.Path(directory)
        exclusions_path = repository / "exclusions.json"
        exclusions = {
            "schema": corpus.EXCLUSION_SCHEMA,
            "records": [
                {
                    "game_id": 100,
                    "categories": ["protected_evaluation"],
                    "sources": ["protected.json"],
                }
            ],
        }
        write_json(exclusions_path, exclusions)
        exclusion_sha = hashlib.sha256(exclusions_path.read_bytes()).hexdigest()

        public_path = repository / "public.json"
        public = {
            "schema": corpus.PUBLIC_SCHEMA,
            "excluded_locked_games": 3,
            "structurally_rejected": [{"game_id": 102, "reason": "bad action"}],
            "records": [
                {
                    "game_id": game_id,
                    "player_id": 0,
                    "won": True,
                    "turns": SHORT_WIN,
                }
                for game_id in (100, 101)
            ],
        }
        write_json(public_path, public)

        live = {
            "schema": corpus.LIVE_REPLAY_SCHEMA,
            "replay": {
                "game_id": 200,
                "winner_player_id": 0,
                "turns": SHORT_WIN,
            },
        }
        live_bytes = corpus.canonical_json_bytes(live)
        record_sha = hashlib.sha256(live_bytes).hexdigest()
        record_path = repository / "records" / f"{record_sha}.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_bytes(live_bytes)
        snapshot_path = repository / "snapshot.json"
        snapshot = {
            "schema": corpus.LIVE_SNAPSHOT_SCHEMA,
            "exclusion_registry_sha256": exclusion_sha,
            "records": [
                {
                    "game_id": 200,
                    "own_player_id": 1,
                    "record_path": f"records/{record_sha}.json",
                    "record_sha256": record_sha,
                    "direct_experts": [
                        {"strength_tier": {"name": "strong-6-10"}}
                    ],
                }
            ],
        }
        write_json(snapshot_path, snapshot)
        return repository, exclusions_path, public_path, snapshot_path

    def test_offline_normalizer_audits_every_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, exclusions, public, snapshot = self.make_sources(directory)
            manifest = corpus.normalize_replay_sources(
                repository=repository,
                exclusion_path=exclusions,
                public_jacek_path=public,
                live_snapshot_path=snapshot,
            )
            previous = repository / "roots.json"
            write_json(previous, manifest)
            refreshed = corpus.normalize_replay_sources(
                repository=repository,
                exclusion_path=exclusions,
                public_jacek_path=public,
                live_snapshot_path=snapshot,
                previous_roots_path=previous,
            )
        self.assertTrue(
            manifest["exclusion_boundary"]["read_before_candidate_sources"]
        )
        self.assertEqual(manifest["exclusion_boundary"]["path"], "exclusions.json")
        self.assertEqual(
            [source["path"] for source in manifest["sources"]],
            ["public.json", "snapshot.json"],
        )
        self.assertEqual(
            sorted(record["game_id"] for record in manifest["accepted"]), [101, 200]
        )
        self.assertIn(
            "protected-exclusion-registry",
            {record["reason"] for record in manifest["excluded"]},
        )
        self.assertEqual(manifest["counts"]["source_preexcluded_aggregate"], 3)
        self.assertEqual(manifest["structurally_rejected"][0]["game_id"], 102)
        self.assertTrue(all(record["split"] in {"train", "validation", "test"}
                            for record in manifest["accepted"]))
        self.assertEqual(refreshed["split_parent"]["frozen_groups"], 2)
        self.assertEqual(
            {record["group_id"]: record["split"] for record in manifest["accepted"]},
            {record["group_id"]: record["split"] for record in refreshed["accepted"]},
        )

    def test_snapshot_must_bind_supplied_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, exclusions, public, snapshot = self.make_sources(directory)
            payload = json.loads(snapshot.read_text())
            payload["exclusion_registry_sha256"] = "0" * 64
            write_json(snapshot, payload)
            with self.assertRaisesRegex(ValueError, "supplied exclusions"):
                corpus.normalize_replay_sources(
                    repository=repository,
                    exclusion_path=exclusions,
                    public_jacek_path=public,
                    live_snapshot_path=snapshot,
                )

    def test_rank4_absolute_score_is_changed_to_mover_relative(self):
        row = {
            "schema": corpus.TEACHER_SCHEMA,
            "group_id": "continuation:1",
            "root_group_id": "root:1",
            "source": "fixture",
            "winner": 1,
            "prefix": [{"player_id": 0, "action": "0"}],
            "mover": 1,
            "root_score": 12_000.0,
            "completed_depth": 4,
            "nodes": 32_000,
        }
        sample, reflected = corpus.sample_from_teacher_row(row)
        expected = 0.75 * -math.tanh(1.0) + 0.25
        self.assertAlmostEqual(sample.target, expected)
        self.assertEqual(reflected.active, features.reflect_active(sample.active))
        self.assertEqual(sample.group_id, reflected.group_id)
        self.assertEqual(sample.group_id, "root:1")

    def test_rank4_fixed_work_accepts_only_proof_or_completed_cap(self):
        valid = self.rank4_teacher_row()
        sample, _ = corpus.sample_from_teacher_row(valid)
        expected = 0.75 * -math.tanh(26_407 / 12_000) - 0.25
        self.assertAlmostEqual(sample.target, expected)

        depth_capped = self.rank4_teacher_row(
            search_config={**valid["search_config"], "max_turn_depth": 1},
            search_stats={
                **valid["search_stats"],
                "completed_depth": 1,
                "nodes": 100,
                "completed_actions": 50,
                "budget_exhausted": False,
                "node_cap_reached": False,
                "depth_cap_reached": True,
            },
            completed_depth=1,
            nodes=100,
        )
        corpus.sample_from_teacher_row(depth_capped)

        invalid_rows = (
            self.rank4_teacher_row(
                search_config={**valid["search_config"], "max_time_ms": 1}
            ),
            self.rank4_teacher_row(
                search_stats={**valid["search_stats"], "deadline_reached": True}
            ),
            self.rank4_teacher_row(
                search_stats={**valid["search_stats"], "nodes": 31_999},
                nodes=31_999,
            ),
            self.rank4_teacher_row(
                search_stats={**valid["search_stats"], "completed_actions": 0}
            ),
            self.rank4_teacher_row(
                search_stats={
                    **valid["search_stats"], "completed_actions": 32_001
                }
            ),
            self.rank4_teacher_row(
                search_stats={**valid["search_stats"], "budget_exhausted": False}
            ),
            self.rank4_teacher_row(
                search_stats={
                    **valid["search_stats"],
                    "budget_exhausted": False,
                    "node_cap_reached": False,
                }
            ),
            self.rank4_teacher_row(
                root_score=999_999,
                root_solved=True,
                proven_winner=0,
                search_stats={
                    **valid["search_stats"],
                    "termination_reason": "root-solved",
                },
            ),
            self.rank4_teacher_row(root_score=-999_999),
            self.rank4_teacher_row(
                search_config={
                    **valid["search_config"], "max_turn_depth": 1
                },
                search_stats={
                    **valid["search_stats"],
                    "completed_depth": 1,
                    "depth_cap_reached": True,
                },
                completed_depth=1,
            ),
            {
                key: value
                for key, value in self.rank4_teacher_row().items()
                if key != "position_id"
            },
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    corpus.sample_from_teacher_row(row)

    def test_proven_teacher_value_is_mover_relative(self):
        row = {
            "schema": corpus.TEACHER_SCHEMA,
            "group_id": "root:2",
            "source": "fixture",
            "winner": 0,
            "prefix": [{"player_id": 0, "action": "0"}],
            "mover": 1,
            "root_score": 999.0,
            "completed_depth": 9,
            "nodes": 400_000,
            "proven_winner": 1,
        }
        sample, _ = corpus.sample_from_teacher_row(row)
        self.assertEqual(sample.target, 0.5)  # .75 proven win + .25 game loss

    def test_search_teacher_value_is_already_mover_relative_and_keeps_lineage(self):
        sample, reflected = corpus.sample_from_teacher_row(
            self.search_teacher_row()
        )
        # The mover eventually loses: .75 * direct 0.8 + .25 * -1.
        self.assertAlmostEqual(sample.target, 0.35)
        self.assertEqual(reflected.active, features.reflect_active(sample.active))
        self.assertEqual(sample.group_id, "root:search")
        self.assertEqual(sample.lineages, reflected.lineages)
        lineage = sample.lineages[0]
        self.assertEqual(lineage.position_id, "position:" + "1" * 64)
        self.assertEqual(lineage.group_id, "continuation:search")
        self.assertEqual(lineage.root_group_id, "root:search")
        self.assertEqual(lineage.split, "train")
        self.assertEqual(lineage.campaign_id, "selfsearch-pilot-fixture")

    def test_search_teacher_requires_explicit_consistent_proof_and_fixed_work(self):
        solved = self.search_teacher_row(
            teacher_value=-1.0,
            root_solved=True,
            proven_winner=0,
            search_stats={
                **self.search_teacher_row()["search_stats"],
                "visits": 0,
                "tree_nodes": 10,
                "tree_cap_reached": False,
                "termination_reason": "root-solved",
            },
        )
        sample, _ = corpus.sample_from_teacher_row(solved)
        self.assertEqual(sample.target, -1.0)

        for broken, message in (
            ({**solved, "teacher_value": 1.0}, "disagrees"),
            (
                self.search_teacher_row(
                    search_config={
                        **self.search_teacher_row()["search_config"],
                        "max_time_ms": 1,
                    }
                ),
                "max_time_ms zero",
            ),
            (
                self.search_teacher_row(
                    search_stats={
                        **self.search_teacher_row()["search_stats"],
                        "deadline_reached": True,
                    }
                ),
                "deadline",
            ),
            (
                self.search_teacher_row(
                    search_stats={
                        **self.search_teacher_row()["search_stats"],
                        "tree_nodes": 63_999,
                        "tree_cap_reached": False,
                    }
                ),
                "fixed work cap",
            ),
            (
                self.search_teacher_row(
                    search_stats={
                        **self.search_teacher_row()["search_stats"],
                        "termination_reason": "closed-unsolved-root",
                    }
                ),
                "termination reason",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    corpus.sample_from_teacher_row(broken)

    def test_target_policy_distinguishes_rank4_transform_from_direct_search(self):
        rank4 = corpus.target_policy_for_schema(corpus.TEACHER_SCHEMA)
        rank4_v2 = corpus.target_policy_for_schema(corpus.RANK4_TEACHER_SCHEMA)
        search = corpus.target_policy_for_schema(corpus.SEARCH_TEACHER_SCHEMA)
        self.assertEqual(rank4["teacher_value"]["transform"], "mover-sign*tanh(root_score/12000)")
        self.assertEqual(rank4_v2["teacher_value"], rank4["teacher_value"])
        self.assertEqual(search["teacher_value"]["transform"], "identity")
        self.assertEqual(rank4["mixture"], search["mixture"])

    def test_search_teacher_declared_split_must_match_frozen_root_split(self):
        samples = corpus.sample_from_teacher_row(self.search_teacher_row())
        with self.assertRaisesRegex(ValueError, "lineage disagrees"):
            corpus.split_and_purge_samples(
                samples, {"root:search": "validation"}
            )

    def test_cross_split_purge_recognizes_reflection_and_rotation(self):
        active = features.encode_active(features.ReplayState())
        samples = [
            corpus.LabeledSample(active, 0.1, 1.0, "train-root"),
            corpus.LabeledSample(features.reflect_active(active), 0.2, 1.0, "val-root"),
            corpus.LabeledSample(features.rotate_active(active), 0.3, 1.0, "test-root"),
        ]
        retained, removed, aggregated = corpus.split_and_purge_samples(
            samples,
            {"train-root": "train", "val-root": "validation", "test-root": "test"},
        )
        self.assertEqual(len(retained["train"]), 1)
        self.assertEqual(removed, {"train": 0, "validation": 1, "test": 1})
        self.assertEqual(aggregated, {"train": 0, "validation": 0, "test": 0})

    def test_same_state_opposite_outcomes_are_weighted_without_target_bias(self):
        active = features.encode_active(features.ReplayState())
        samples = [
            corpus.LabeledSample(active, 1.0, 1.0, "first"),
            corpus.LabeledSample(active, -1.0, 3.0, "second"),
        ]
        retained, removed, aggregated = corpus.split_and_purge_samples(
            samples, {"first": "train", "second": "train"}
        )
        self.assertEqual(len(retained["train"]), 1)
        combined = retained["train"][0]
        self.assertAlmostEqual(combined.target, -0.5)
        self.assertEqual(combined.weight, 4.0)
        self.assertTrue(combined.group_id.startswith("aggregate:"))
        self.assertEqual(removed["train"], 0)
        self.assertEqual(aggregated["train"], 1)

    def test_reflection_augmentation_keeps_both_asymmetric_orientations(self):
        state = features.ReplayState()
        features.apply_complete_turn(state, 0, "1")
        active = features.encode_active(state)
        reflected = features.reflect_active(active)
        self.assertNotEqual(active, reflected)
        samples = [
            corpus.LabeledSample(active, 0.25, 1.0, "root"),
            corpus.LabeledSample(reflected, 0.25, 1.0, "root"),
        ]
        retained, removed, aggregated = corpus.split_and_purge_samples(
            samples, {"root": "train"}
        )
        self.assertEqual(len(retained["train"]), 2)
        self.assertEqual(removed["train"], 0)
        self.assertEqual(aggregated["train"], 0)

    def test_global_split_is_near_80_10_10_and_representative(self):
        records = []
        for index in range(30):
            records.append(
                {
                    "group_id": f"root:{index}",
                    "source": "public" if index % 3 == 0 else "live",
                    "focus_player": index % 2,
                    "winner": (index // 2) % 2,
                    # Unique tiers force singleton strata and exercise the
                    # global rebalance instead of per-stratum rounding.
                    "opponent_tier": f"tier-{index}",
                }
            )
        assignment = corpus._assignment_for_strata(records)
        self.assertEqual(
            {split: list(assignment.values()).count(split) for split in (
                "train", "validation", "test"
            )},
            {"train": 24, "validation": 3, "test": 3},
        )
        for split in ("train", "validation", "test"):
            selected = [record for record in records if assignment[record["group_id"]] == split]
            self.assertEqual({record["source"] for record in selected}, {"public", "live"})
            self.assertEqual({record["focus_player"] for record in selected}, {0, 1})
            self.assertEqual({record["winner"] for record in selected}, {0, 1})
        self.assertEqual(assignment, corpus._assignment_for_strata(records))
        appended = records + [
            {
                "group_id": f"root:{index}",
                "source": "public" if index % 3 == 0 else "live",
                "focus_player": index % 2,
                "winner": (index // 2) % 2,
                "opponent_tier": f"tier-{index}",
            }
            for index in range(30, 40)
        ]
        extended = corpus._assignment_for_strata(appended, assignment)
        self.assertTrue(all(extended[group] == split for group, split in assignment.items()))
        self.assertEqual(
            {split: list(extended.values()).count(split) for split in (
                "train", "validation", "test"
            )},
            {"train": 32, "validation": 4, "test": 4},
        )

    def test_teacher_tsv_has_frozen_four_column_contract(self):
        manifest = {
            "schema": corpus.ROOT_SCHEMA,
            "accepted": [
                {
                    "group_id": "public-jacek:101",
                    "source": "public-jacek",
                    "winner": 0,
                    "turns": SHORT_WIN,
                    "split": "train",
                }
            ],
        }
        lines = pack.teacher_tsv_bytes(manifest).decode().splitlines()
        self.assertEqual(lines[0], "group_id\tsource\twinner\ttranscript")
        self.assertEqual(lines[1], "public-jacek:101\tpublic-jacek\t0\t0/0/3/0/61/0/07")

    def test_pack_rejects_tampered_roots_body(self):
        manifest = {
            "schema": corpus.ROOT_SCHEMA,
            "feature_schema": features.FEATURE_SCHEMA,
            "tool_sha256": {
                "normalizer": "1" * 64,
                "features": "2" * 64,
            },
            "exclusion_boundary": {"read_before_candidate_sources": True},
            "accepted": [{"group_id": "root:1", "split": "train"}],
        }
        manifest["body_sha256"] = corpus.sha256_bytes(
            corpus.canonical_json_bytes(manifest)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "roots.json"
            write_json(path, manifest)
            self.assertEqual(pack.load_roots(path)["accepted"][0]["group_id"], "root:1")
            manifest["accepted"][0]["split"] = "test"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "body SHA-256 mismatch"):
                pack.load_roots(path)


if __name__ == "__main__":
    unittest.main()
