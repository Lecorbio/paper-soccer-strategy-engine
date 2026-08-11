import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import jacek_native_corpus as corpus


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_build_provenance(directory):
    compiler_path = pathlib.Path(
        shutil.which("c++") or shutil.which("clang++")
    ).resolve()
    compiler_version = subprocess.run(
        [str(compiler_path), "--version"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    binary = directory / corpus.ARCHIVED_BINARY_NAME
    binary.write_bytes(b"deterministic native self-play fixture\n")
    sources = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                (ROOT / relative).read_bytes()
            ).hexdigest(),
        }
        for relative in corpus.BUILD_SOURCE_PATHS
    ]
    producer = hashlib.sha256(json.dumps(
        [[entry["path"], entry["sha256"]] for entry in sources],
        separators=(",", ":"),
    ).encode()).hexdigest()
    contract = {
        "schema": corpus.BUILD_PROVENANCE_SCHEMA,
        "binary": {
            "path": corpus.ARCHIVED_BINARY_NAME,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "compiler": {
            "executable": compiler_path.name,
            "sha256": hashlib.sha256(compiler_path.read_bytes()).hexdigest(),
            "version": compiler_version,
            "version_sha256": hashlib.sha256(
                compiler_version.encode()
            ).hexdigest(),
        },
        "build_argv": list(corpus.CANONICAL_BUILD_ARGV),
        "producer_sha256": producer,
        "sources": sources,
    }
    raw = (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (directory / corpus.BUILD_PROVENANCE_NAME).write_bytes(raw)
    return contract, hashlib.sha256(raw).hexdigest()


def write_corpus(path, records):
    contract, build_sha256 = write_build_provenance(path.parent)
    for record in records:
        record["generator"]["producer_sha256"] = contract["producer_sha256"]
        record["generator"]["build_provenance_sha256"] = build_sha256
    path.write_text("".join(
        json.dumps(record) + "\n" for record in records
    ))
    return contract, build_sha256


def active_features(edge=None, bucket=0):
    active = [] if edge is None else [edge]
    active.extend(
        corpus.EDGE_COUNT + vertex * corpus.DISTANCE_BUCKETS + bucket
        for vertex in range(corpus.VERTEX_COUNT)
    )
    return sorted(active)


def game_record(game, winner, samples=1):
    digest = "a" * 64
    complete_turns = 6
    direction = "0" if winner == 0 else "4"
    transcript = "/".join(direction for _ in range(complete_turns))
    sample_turns = list(range(samples))
    _, boundary_features = corpus._replay_recorded_game_with_features(
        transcript.split("/"), 0, winner, 1, sample_turns
    )
    return {
        "schema": corpus.GAME_SCHEMA,
        "feature_schema": corpus.FEATURE_SCHEMA,
        "rules": corpus.RULES,
        "generator": {
            "schema": corpus.GENERATOR_SCHEMA,
            "action": "complete-turn",
            "max_actions": 250,
            "deque_schedule": corpus.DEQUE_SCHEDULE,
            "work_unit": "maximum-tree-nodes",
            "search_work": 32,
            "sampling_temperature": 3.0,
            "temperature_turns": 12,
            "temperature_schedule":
                "absolute-complete-turn-index-before-cutoff/v1",
            "opening_schema":
                "deterministic-procedural-complete-turn-prefix/v1",
            "opening_depth": 0,
            "opening_seed": str(2000 + game),
            "opening_retry": 0,
            "opening_transcript": "",
            "value_target": "mover-relative-final-outcome",
            "checkpoint_color_schedule":
                "swap-player-checkpoints-on-odd-games",
            "producer_sha256": digest,
            "build_provenance_sha256": digest,
            "models": {
                player: {
                    "model_sha256": digest,
                    "packed_sha256": digest,
                    "artifact_sha256": digest,
                }
                for player in ("player_one", "player_two")
            },
            "search_stats": {
                "searches": complete_turns,
                "expansions": samples,
                "child_evaluations": samples,
                "completed_actions": samples,
                "partial_paths": samples,
                "generator_truncations": 0,
                "tree_cap_searches": 0,
                "expansion_cap_searches": 0,
                "tactical_proof_paths": 0,
                "tactical_classes_found": 0,
                "tactical_proof_truncations": 0,
            },
        },
        "seed": str(1000 + game),
        "game": game,
        "shard_index": game % 2,
        "shard_count": 2,
        "winner": winner,
        "complete_turns": complete_turns,
        "transcript_schema": "complete-turn-directions-slash/v1",
        "transcript": transcript,
        "samples": [
            {
                "turn": turn,
                "player": turn % 2,
                "active": list(boundary_features[turn][0]),
                "canonical_state_id": corpus.canonical_state_id(
                    boundary_features[turn][0]
                ),
                "reflected_active": list(boundary_features[turn][1]),
                "reflected_state_id": corpus.canonical_state_id(
                    boundary_features[turn][1]
                ),
            }
            for turn in sample_turns
        ],
    }


def refresh_record_features(record):
    turns = [sample["turn"] for sample in record["samples"]]
    _, boundary_features = corpus._replay_recorded_game_with_features(
        record["transcript"].split("/"),
        record["generator"]["opening_depth"],
        record["winner"],
        1,
        turns,
    )
    for sample in record["samples"]:
        active, reflected = boundary_features[sample["turn"]]
        sample["player"] = sample["turn"] % 2
        sample["active"] = list(active)
        sample["canonical_state_id"] = corpus.canonical_state_id(active)
        sample["reflected_active"] = list(reflected)
        sample["reflected_state_id"] = corpus.canonical_state_id(reflected)


def random_terminal_transcript(seed, desired_winner):
    for retry in range(512):
        random_source = random.Random(seed + retry * 1_000_003)
        state = corpus._initial_replay_state()
        actions = []
        for _ in range(384):
            if state.winner is not None:
                break
            mover = state.to_move
            action = []
            while state.winner is None and state.to_move == mover:
                legal = [
                    destination
                    for destination in corpus._neighbor_points(state.ball)
                    if corpus._is_legal_destination(state, destination)
                ]
                destination = random_source.choice(legal)
                delta = (
                    destination[0] - state.ball[0],
                    destination[1] - state.ball[1],
                )
                direction = str(corpus.DIRECTION_DELTAS.index(delta))
                action.append(direction)
                corpus._apply_primitive(state, direction)
            actions.append("".join(action))
        if state.winner == desired_winner:
            return actions
    raise AssertionError("could not construct a deterministic terminal fixture")


def diverse_game_records(count=20):
    records = []
    used_features = set()
    for game in range(count):
        winner = game % 2
        for candidate in range(64):
            actions = random_terminal_transcript(
                10_000 * game + 100 * candidate, winner
            )
            turns = range(1, len(actions))
            _, features = corpus._replay_recorded_game_with_features(
                actions, 0, winner, 1, turns
            )
            selected_turn = next((
                turn for turn in turns
                if set(features[turn]).isdisjoint(used_features)
            ), None)
            if selected_turn is None:
                continue
            record = game_record(game, winner)
            record["complete_turns"] = len(actions)
            record["transcript"] = "/".join(actions)
            record["generator"]["search_stats"]["searches"] = len(actions)
            record["samples"][0]["turn"] = selected_turn
            refresh_record_features(record)
            active, reflected = features[selected_turn]
            used_features.update((active, reflected))
            records.append(record)
            break
        else:
            raise AssertionError("could not construct disjoint feature fixtures")
    return records


class JacekNativeCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_generator = load_module(
            "jacek_native_seed_generator_under_test",
            TOOLS / "generate_jacek_native_seed.py",
        )
        cls.workflow = load_module(
            "jacek_native_workflow_under_test",
            TOOLS / "jacek_native_workflow.py",
        )

    def test_selfplay_producer_hash_covers_every_compiled_contract(self):
        expected = {
            "tools/jacek_native_selfplay.cpp",
            "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
            "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
            "src/core/rules.cpp",
            "src/core/geometry.cpp",
            "src/bots/mcts_internal.hpp",
            "include/papersoccer/types.hpp",
            "include/papersoccer/geometry.hpp",
            "include/papersoccer/rules.hpp",
        }
        observed = {
            path.relative_to(ROOT).as_posix()
            for path in self.workflow.PROVENANCE_SOURCES
        }
        self.assertEqual(observed, expected)
        payload = json.dumps(
            [
                (str(path.relative_to(ROOT)), hashlib.sha256(
                    path.read_bytes()
                ).hexdigest())
                for path in self.workflow.PROVENANCE_SOURCES
            ],
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            self.workflow.producer_sha256(), hashlib.sha256(payload).hexdigest()
        )

    def test_build_provenance_contract_is_canonical_and_path_independent(self):
        compiler_path = pathlib.Path(
            shutil.which("c++") or shutil.which("clang++")
        ).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            binary = pathlib.Path(temporary) / "arbitrary-output-name"
            binary.write_bytes(b"deterministic native binary\n")
            report = self.workflow.build_provenance(binary, compiler_path)
            rendered = self.workflow.canonical_json_bytes(report)
        self.assertEqual(
            report["schema"], self.workflow.BUILD_PROVENANCE_SCHEMA
        )
        self.assertEqual(
            report["binary"]["path"], self.workflow.ARCHIVED_BINARY_NAME
        )
        self.assertEqual(
            report["build_argv"], list(self.workflow.CANONICAL_BUILD_ARGV)
        )
        self.assertEqual(
            [entry["path"] for entry in report["sources"]],
            [path.relative_to(ROOT).as_posix()
             for path in self.workflow.PROVENANCE_SOURCES],
        )
        self.assertNotIn(str(ROOT), rendered.decode())
        self.assertNotIn(str(pathlib.Path.home()), rendered.decode())

    def test_workflow_build_produces_an_executable_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = pathlib.Path(temporary) / "native-selfplay"
            report = self.workflow.build(binary, "c++")
            self.assertTrue(binary.stat().st_mode & 0o111)
            self.assertEqual(
                report["binary"]["sha256"],
                hashlib.sha256(binary.read_bytes()).hexdigest(),
            )

    def test_current_codingame_contract_and_mover_outcome(self):
        record = game_record(0, winner=1, samples=2)
        record["transcript"] = "3/5/4/4/4/4"
        refresh_record_features(record)
        game = corpus.validate_record(record)
        self.assertEqual(corpus.RULES["goal_rule"], "own-goals-allowed")
        self.assertEqual(corpus.RULES["blocked_rule"], "mover-loses")
        self.assertEqual(game.samples[0].outcome, -1.0)
        self.assertEqual(
            {sample.symmetry for sample in game.samples},
            {"identity", "reflection"},
        )
        self.assertEqual(
            [sample.outcome for sample in game.samples if sample.turn == 1],
            [1.0, 1.0],
        )

    def test_rejects_incumbent_provenance_and_action_labels(self):
        record = game_record(0, winner=0)
        record["generator"]["checkpoint"] = "rank_4"
        with self.assertRaisesRegex(ValueError, "forbidden non-native"):
            corpus.validate_record(record)

    def test_opening_prefix_is_independent_of_temperature_cutoff(self):
        record = game_record(0, winner=0)
        record["generator"].update({
            "opening_depth": 4,
            "opening_seed": "99123",
            "opening_transcript": "0/0/0/0",
            "temperature_turns": 12,
        })
        record["generator"]["search_stats"]["searches"] = 2
        record["samples"][0]["turn"] += 4
        refresh_record_features(record)
        game = corpus.validate_record(record)
        self.assertEqual(game.samples[0].turn, 4)
        self.assertEqual(record["generator"]["temperature_turns"], 12)

    def test_reanalysis_budget_names_and_stability_are_ordered(self):
        record = game_record(0, winner=0)
        record["samples"][0]["reanalysis"] = {
            "value": 0.25,
            "work": 30_000,
            "verification_work": 60_000,
            "truncated": False,
            "action_stable": True,
            "value_delta": 0.01,
            "stable": True,
            "exact": False,
        }
        game = corpus.validate_record(record)
        self.assertEqual(game.samples[0].auxiliary_value, 0.25)
        record["samples"][0]["reanalysis"].update({
            "work": 60_000,
            "verification_work": 30_000,
        })
        with self.assertRaisesRegex(ValueError, "budgets are out of order"):
            corpus.validate_record(record)
        record = game_record(0, winner=0)
        record["samples"][0]["policy_target"] = [1.0]
        with self.assertRaisesRegex(ValueError, "action/teacher labels"):
            corpus.validate_record(record)

    def test_whole_game_split_and_overlap_purge(self):
        records = [
            corpus.validate_record(game_record(game, game % 2))
            for game in range(20)
        ]
        assignment = corpus.assign_splits(records)
        self.assertEqual(
            {assignment[game.split_group] for game in records},
            {"train", "validation", "test"},
        )
        duplicate = records[0].samples[0]
        splits, removed = corpus.purge_cross_split_overlaps({
            "train": [duplicate],
            "validation": [duplicate],
            "test": [duplicate],
        })
        self.assertEqual(removed, {"train": 0, "validation": 1, "test": 1})
        self.assertEqual([len(splits[name]) for name in splits], [1, 0, 0])

    def test_same_shard_basename_from_distinct_runs_is_collision_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = []
            for run, game in (("run-a", 0), ("run-b", 1)):
                path = root / run / "shard-00-of-01.jsonl"
                path.parent.mkdir()
                record = game_record(game, game % 2)
                record["shard_index"] = 0
                record["shard_count"] = 1
                write_corpus(path, [record])
                paths.append(path)
            games, sources = corpus.load_games(paths)
        self.assertEqual(len(games), 2)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(
            source.startswith("sha256:") and len(source) == 71
            for source in sources
        ))

    def test_model_lineage_triples_survive_validation_and_reporting(self):
        record = game_record(0, winner=0)
        record["shard_index"] = 0
        record["shard_count"] = 1
        expected = []
        for index, player in enumerate(("player_one", "player_two"), 1):
            metadata = {
                "model_sha256": str(index) * 64,
                "packed_sha256": str(index + 2) * 64,
                "artifact_sha256": str(index + 4) * 64,
            }
            record["generator"]["models"][player] = metadata
            expected.append(metadata)
        game = corpus.validate_record(record)
        self.assertEqual(
            [dataclasses.asdict(artifact) for artifact in game.model_artifacts],
            sorted(expected, key=lambda value: value["artifact_sha256"]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "lineage.jsonl"
            write_corpus(path, [record])
            report = corpus.summarize([path])
        self.assertEqual(report["generation"]["model_artifacts"], expected)

    def test_build_provenance_is_file_backed_and_reaches_the_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "native.jsonl"
            contract, digest = write_corpus(path, [game_record(0, 0)])
            games, _ = corpus.load_games([path], verify_local_build=True)
            report = corpus.summarize([path])
        self.assertEqual(games[0].build_provenance_sha256, digest)
        self.assertEqual(games[0].build_contract, contract)
        self.assertEqual(
            report["generation"]["build_provenance_sha256"], [digest]
        )
        self.assertEqual(report["generation"]["build_contracts"], [{
            "sha256": digest,
            "contract": contract,
        }])

    def test_build_provenance_rejects_missing_noncanonical_and_record_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "native.jsonl"
            record = game_record(0, 0)
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "missing sibling"):
                corpus.load_games([path])

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "native.jsonl"
            contract, _ = write_corpus(path, [game_record(0, 0)])
            provenance = path.parent / corpus.BUILD_PROVENANCE_NAME
            provenance.write_text(json.dumps(contract, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                corpus.load_games([path])

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "native.jsonl"
            record = game_record(0, 0)
            write_corpus(path, [record])
            record["generator"]["build_provenance_sha256"] = "b" * 64
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                corpus.load_games([path])

    def test_build_provenance_rejects_source_order_and_producer_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            contract, _ = write_build_provenance(directory)
            contract["sources"].reverse()
            raw = (
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            with self.assertRaisesRegex(ValueError, "source identity"):
                corpus._validate_build_contract(raw, directory, False)

        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            contract, _ = write_build_provenance(directory)
            contract["sources"][0]["sha256"] = "c" * 64
            raw = (
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            with self.assertRaisesRegex(ValueError, "producer SHA-256"):
                corpus._validate_build_contract(raw, directory, False)

    def test_archived_binary_is_rechecked_only_for_local_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "native.jsonl"
            write_corpus(path, [game_record(0, 0)])
            (path.parent / corpus.ARCHIVED_BINARY_NAME).write_bytes(b"tampered")
            corpus.load_games([path], verify_local_build=False)
            with self.assertRaisesRegex(ValueError, "archived binary is stale"):
                corpus.load_games([path], verify_local_build=True)

    def test_local_build_verification_rechecks_sources_and_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            contract, _ = write_build_provenance(directory)
            contract["sources"][0]["sha256"] = "d" * 64
            pairs = [
                [entry["path"], entry["sha256"]]
                for entry in contract["sources"]
            ]
            contract["producer_sha256"] = hashlib.sha256(json.dumps(
                pairs, separators=(",", ":")
            ).encode()).hexdigest()
            raw = (
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            corpus._validate_build_contract(raw, directory, False)
            with self.assertRaisesRegex(ValueError, "source is stale"):
                corpus._validate_build_contract(raw, directory, True)

        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            contract, _ = write_build_provenance(directory)
            contract["compiler"]["sha256"] = "e" * 64
            raw = (
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            corpus._validate_build_contract(raw, directory, False)
            with self.assertRaisesRegex(ValueError, "compiler identity is stale"):
                corpus._validate_build_contract(raw, directory, True)

    def test_complete_transcript_presence_count_and_characters(self):
        record = game_record(0, winner=0)
        record["transcript"] = ""
        with self.assertRaisesRegex(ValueError, "transcript/turn count"):
            corpus.validate_record(record)

    def test_independent_replay_accepts_both_own_goal_results(self):
        north = corpus.validate_record(game_record(0, winner=0))
        south = corpus.validate_record(game_record(1, winner=1))
        self.assertEqual((north.winner, south.winner), (0, 1))

    def test_independent_replay_enforces_blocked_mover_loses(self):
        record = game_record(0, winner=0)
        record["complete_turns"] = 12
        record["transcript"] = (
            "4/5/5/4/30/0/67/63/05253125011/66/525/3"
        )
        record["generator"]["search_stats"]["searches"] = 12
        corpus.validate_record(record)
        state = corpus._replay_recorded_game(
            record["transcript"].split("/"), 0, 0, 1
        )
        self.assertEqual(state.winner, 0)
        self.assertFalse(corpus._is_goal_point(state.ball))

    def test_independent_replay_rejects_illegal_or_non_atomic_turns(self):
        reused_edge = game_record(0, winner=0)
        reused_edge["transcript"] = "0/4/0/0/0/0"
        with self.assertRaisesRegex(ValueError, "illegal primitive"):
            corpus.validate_record(reused_edge)

        continued_after_handoff = game_record(0, winner=0)
        continued_after_handoff["transcript"] = "00/0/0/0/0/0"
        with self.assertRaisesRegex(ValueError, "continues after handoff"):
            corpus.validate_record(continued_after_handoff)

        incomplete_rebound = game_record(0, winner=0)
        incomplete_rebound["complete_turns"] = 3
        incomplete_rebound["transcript"] = "0/3/6"
        incomplete_rebound["generator"]["search_stats"]["searches"] = 3
        with self.assertRaisesRegex(ValueError, "rebound chain completion"):
            corpus.validate_record(incomplete_rebound)

    def test_independent_replay_rejects_nonterminal_and_wrong_winner(self):
        nonterminal = game_record(0, winner=0)
        nonterminal["complete_turns"] = 1
        nonterminal["transcript"] = "0"
        nonterminal["generator"]["search_stats"]["searches"] = 1
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            corpus.validate_record(nonterminal)

        wrong_winner = game_record(0, winner=0)
        wrong_winner["winner"] = 1
        with self.assertRaisesRegex(ValueError, "does not match recorded winner"):
            corpus.validate_record(wrong_winner)

    def test_independent_replay_checks_recorded_opening_atomicity(self):
        record = game_record(0, winner=0)
        record["generator"].update({
            "opening_depth": 1,
            "opening_transcript": "00",
        })
        record["generator"]["search_stats"]["searches"] = 5
        record["transcript"] = "00/0/0/0/0/0"
        record["samples"][0]["turn"] = 1
        with self.assertRaisesRegex(ValueError, "opening.*continues after handoff"):
            corpus.validate_record(record)
        record = game_record(0, winner=0)
        record["transcript"] = "0/x"
        with self.assertRaisesRegex(ValueError, "transcript/turn count"):
            corpus.validate_record(record)

    def test_independent_feature_replay_rejects_boundary_corruption(self):
        record = game_record(0, winner=0, samples=2)
        record["transcript"] = "1/7/0/0/0/0"
        refresh_record_features(record)
        corpus.validate_record(record)

        active_corruption = copy.deepcopy(record)
        active = active_corruption["samples"][1]["active"]
        offset = next(
            index for index, feature in enumerate(active)
            if corpus.EDGE_COUNT <= feature < corpus.EDGE_COUNT + 8
        )
        bucket = active[offset] - corpus.EDGE_COUNT
        active[offset] = corpus.EDGE_COUNT + (bucket + 1) % 8
        active.sort()
        active_corruption["samples"][1]["canonical_state_id"] = (
            corpus.canonical_state_id(active)
        )
        with self.assertRaisesRegex(ValueError, "active does not match replayed"):
            corpus.validate_record(active_corruption)

        reflection_corruption = copy.deepcopy(record)
        reflected = reflection_corruption["samples"][1]["reflected_active"]
        offset = next(
            index for index, feature in enumerate(reflected)
            if corpus.EDGE_COUNT <= feature < corpus.EDGE_COUNT + 8
        )
        bucket = reflected[offset] - corpus.EDGE_COUNT
        reflected[offset] = corpus.EDGE_COUNT + (bucket + 1) % 8
        reflected.sort()
        reflection_corruption["samples"][1]["reflected_state_id"] = (
            corpus.canonical_state_id(reflected)
        )
        with self.assertRaisesRegex(
            ValueError, "reflected_active does not match replayed"
        ):
            corpus.validate_record(reflection_corruption)

    def test_independent_feature_encoder_matches_frozen_cpp_order(self):
        initial = corpus._encode_replay_features(
            corpus._initial_replay_state()
        )
        self.assertFalse(any(index < corpus.EDGE_COUNT for index in initial))
        histogram = [0] * corpus.DISTANCE_BUCKETS
        for index in initial:
            histogram[(index - corpus.EDGE_COUNT) % corpus.DISTANCE_BUCKETS] += 1
        self.assertEqual(histogram, [1, 8, 16, 46, 28, 6, 0, 0])

        state = corpus._initial_replay_state()
        for direction in "7632":
            corpus._apply_primitive(state, direction)
        self.assertEqual(state.to_move, 1)
        active = corpus._encode_replay_features(state)
        value = 0xCBF29CE484222325
        for index in active:
            value ^= index & 0xFF
            value = (value * 0x100000001B3) & ((1 << 64) - 1)
            value ^= (index >> 8) & 0xFF
            value = (value * 0x100000001B3) & ((1 << 64) - 1)
        self.assertEqual(len(active), 109)
        self.assertEqual(value, 0xF90097CECB3BBBB8)

    def test_untrained_seed_artifacts_are_current_and_data_independent(self):
        descriptor, runtime, metadata = self.seed_generator.render()
        self.assertEqual(
            descriptor,
            (ROOT / "models" / "jacek_native_untrained_seed.json").read_text(),
        )
        self.assertEqual(
            runtime,
            (ROOT / "models" / "jacek_native_untrained_seed.runtime").read_text(),
        )
        self.assertEqual(metadata["weights"], 38_048)
        parsed = json.loads(descriptor)
        self.assertIsNone(parsed["training"])
        self.assertFalse(parsed["incumbent_dependencies"])


@unittest.skipIf(np is None, "Jacek-native trainer tests require NumPy")
class JacekNativeTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = load_module(
            "jacek_native_trainer_under_test", TOOLS / "train_jacek_native.py"
        )
        cls.exporter = load_module(
            "jacek_native_exporter_under_test",
            ROOT / "submissions" / "codingame" / "tools" /
            "generate_jacek_native_model.py",
        )

    def write_corpus(self, path):
        records = diverse_game_records()
        write_corpus(path, records)

    def test_checkpoint_provenance_distinguishes_bootstrap_and_native_inputs(self):
        runtime_path = ROOT / "models/jacek_native_untrained_seed.runtime"
        runtime_lines = runtime_path.read_text(encoding="utf-8").splitlines()
        identity = {
            "artifact_sha256": hashlib.sha256(
                runtime_path.read_bytes()
            ).hexdigest(),
            "model_sha256": runtime_lines[3],
            "packed_sha256": runtime_lines[4],
        }
        bootstrap_record = game_record(0, winner=0)
        for metadata in bootstrap_record["generator"]["models"].values():
            metadata.update(identity)
        bootstrap = self.trainer.checkpoint_provenance([
            corpus.validate_record(bootstrap_record)
        ])
        self.assertEqual(bootstrap, {
            "mode": "untrained-seed-bootstrap/v1",
            "artifacts": [identity],
        })

        native = self.trainer.checkpoint_provenance([
            corpus.validate_record(game_record(1, winner=1))
        ])
        self.assertEqual(native["mode"], "native-runtime-models/v1")
        self.assertEqual(len(native["artifacts"]), 1)

    def test_corpus_file_order_cannot_change_seeded_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = [root / "first.jsonl", root / "second.jsonl"]
            records = diverse_game_records()
            for parity, path in enumerate(paths):
                write_corpus(path, [
                    record for record in records
                    if record["game"] % 2 == parity
                ])
            forward, forward_report = self.trainer.load_datasets(paths)
            reverse, reverse_report = self.trainer.load_datasets(paths[::-1])
            self.assertEqual(forward_report, reverse_report)
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    forward[split].game_keys, reverse[split].game_keys
                )
                self.assertEqual(
                    [tuple(active) for active in forward[split].active],
                    [tuple(active) for active in reverse[split].active],
                )
                np.testing.assert_array_equal(
                    forward[split].outcome, reverse[split].outcome
                )
            first, _ = self.trainer.train_seed(
                forward, 29, 1, 1, 8, 0.001, 1e-5, 0.25, 0
            )
            second, _ = self.trainer.train_seed(
                reverse, 29, 1, 1, 8, 0.001, 1e-5, 0.25, 0
            )
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])

    def test_sparse_forward_uses_public_activations_without_biases(self):
        parameters = self.trainer.initialize(7)
        active = [np.asarray(active_features(3), dtype=np.int32)]
        prediction, cache = self.trainer.forward(parameters, active)
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(-1.0 <= prediction[0] <= 1.0)
        first_pre, first, second_pre, second, output_pre = cache
        expected_first = np.where(
            first_pre >= 0.0, first_pre * first_pre, 0.01 * first_pre
        )
        np.testing.assert_allclose(first, expected_first, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            prediction, self.trainer._fast_tanh(output_pre), rtol=0.0, atol=0.0
        )
        self.assertEqual(set(parameters), {"w1", "w2", "w3"})

    def test_fast_tanh_matches_the_deployed_rational_contract(self):
        values = np.asarray(
            [-8.0, -4.95, -2.0, -0.25, 0.0, 0.25, 2.0, 4.95, 8.0],
            dtype=np.float32,
        )
        actual = self.trainer._fast_tanh(values)
        self.assertEqual(float(actual[0]), -1.0)
        self.assertEqual(float(actual[-1]), 1.0)
        np.testing.assert_allclose(actual, -actual[::-1], rtol=0.0, atol=2e-7)
        epsilon = np.float32(1e-3)
        interior = values[2:-2]
        numerical = (
            self.trainer._fast_tanh(interior + epsilon)
            - self.trainer._fast_tanh(interior - epsilon)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            self.trainer._fast_tanh_derivative(interior),
            numerical,
            rtol=2e-3,
            atol=2e-4,
        )

    def test_tiny_deterministic_training_and_qat(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus_path = pathlib.Path(temporary) / "native.jsonl"
            self.write_corpus(corpus_path)
            datasets, report = self.trainer.load_datasets([corpus_path])
            self.assertEqual(set(datasets), {"train", "validation", "test"})
            self.assertEqual(
                report["generation"]["build_provenance_sha256"],
                [report["generation"]["build_contracts"][0]["sha256"]],
            )
            first, first_report = self.trainer.train_seed(
                datasets, 17, 2, 1, 4, 0.001, 1e-5, 0.25, 1
            )
            second, second_report = self.trainer.train_seed(
                datasets, 17, 2, 1, 4, 0.001, 1e-5, 0.25, 1
            )
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(
                first_report["quantized_metrics"],
                second_report["quantized_metrics"],
            )
            self.assertFalse(report.get("incumbent_labels", False))

    def test_checkpoint_selection_uses_held_out_outcome_error(self):
        candidates = [self.trainer.initialize(1), self.trainer.initialize(2)]
        reports = [
            {
                "seed": 1,
                "quantized_metrics": {
                    "validation": {
                        "outcome_mse": 0.20,
                        "combined_target_mse": 0.90,
                    }
                },
            },
            {
                "seed": 2,
                "quantized_metrics": {
                    "validation": {
                        "outcome_mse": 0.30,
                        "combined_target_mse": 0.10,
                    }
                },
            },
        ]
        arguments = argparse.Namespace(
            auxiliary_weight=0.25,
            batch_size=256,
            epochs=50,
            patience=8,
            learning_rate=0.001,
            weight_decay=1e-5,
            qat_epochs=4,
        )
        model = self.trainer.build_report(
            candidates, reports, {"corpus_sha256": "a" * 64}, arguments
        )
        self.assertEqual(model["training"]["chosen_seed"], 1)
        self.assertEqual(
            model["training"]["selection"],
            "minimum-quantized-validation-outcome-mse-then-seed",
        )
        self.assertNotIn("measured_seconds", model["training"])
        self.assertNotIn("measured_examples_per_second", model["training"])

    def minimal_model(self):
        weights = {
            name: {
                "shape": list(shape),
                "values": [0] * int(np.prod(shape)),
            }
            for name, shape in self.exporter.SHAPES.items()
        }
        quantization = {
            "bits": 3,
            "minimum": -3,
            "maximum": 3,
            "scheme": "symmetric-per-layer-round-to-nearest",
            "packing": "w1-w2-w3-row-major-signed-3bit-lsb-first",
            "scales": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
            "weights": weights,
        }
        return {
            "schema": self.exporter.MODEL_SCHEMA,
            "feature_schema": self.exporter.FEATURE_SCHEMA,
            "architecture": {
                "inputs": 1156,
                "hidden_one": 32,
                "hidden_two": 32,
                "outputs": 1,
                "biases": False,
                "hidden_one_activation":
                    "square-nonnegative-leaky-0.01-negative",
                "hidden_two_activation": "leaky-relu-0.01",
                "output_activation": "tanh",
            },
            "rules": self.exporter.RULES,
            "target": {
                "primary": "mover-relative-final-outcome",
                "auxiliary": "stable-native-bfm-reanalysis",
                "auxiliary_weight": 0.25,
                "policy_target": None,
            },
            "provenance": {
                "trainer_sha256": hashlib.sha256(
                    (TOOLS / "train_jacek_native.py").read_bytes()
                ).hexdigest(),
                "corpus_validator_sha256": hashlib.sha256(
                    (TOOLS / "jacek_native_corpus.py").read_bytes()
                ).hexdigest(),
                "incumbent_labels": False,
                "protected_data": False,
                "augmentation": {
                    "reflection": True,
                    "rotation":
                        "player-two-canonicalization-in-feature-encoder",
                    "grouping": "whole-game-before-augmentation",
                },
            },
            "training": {"chosen_seed": 17},
            "quantization": quantization,
            "checkpoints": [{
                "seed": 17,
                "model": {},
                "quantization": copy.deepcopy(quantization),
            }],
        }

    def test_signed_three_bit_export_is_round_trip_and_source_safe(self):
        sequence = [-3, -2, -1, 0, 1, 2, 3] * 19
        packed = self.exporter.pack_signed_three_bit(sequence)
        self.assertEqual(
            self.exporter.unpack_signed_three_bit(packed, len(sequence)), sequence
        )
        header, metadata = self.exporter.render(self.minimal_model(), "a" * 64)
        self.assertLess(metadata["header_characters"], 95_000)
        self.assertEqual(metadata["weight_count"], 1156 * 32 + 32 * 32 + 32)
        self.assertIn("kPackedWeights", header)
        self.assertNotIn("kW1", header)
        self.assertIn("own-goals-allowed", json.dumps(self.exporter.RULES))


if __name__ == "__main__":
    unittest.main()
