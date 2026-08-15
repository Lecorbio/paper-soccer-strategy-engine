#!/usr/bin/env python3
"""Focused, isolated contract tests for the TT exact-collision campaign."""

import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
RECORDER_PATH = os.path.join(
    ROOT, "tools", "record_rank4_jacek_hybrid_tt_exact_collision.py"
)
GATE_PATH = os.path.join(
    ROOT, "submissions", "codingame", "bots", "rank_4_jacek_hybrid",
    "comparison_gate.cpp",
)
ENGINE_PATH = os.path.join(
    ROOT, "submissions", "codingame", "bots", "rank_4_jacek_hybrid",
    "comparison_gate_engine.hpp",
)


def load_recorder():
    specification = importlib.util.spec_from_file_location(
        "tt_exact_collision_recorder_under_test", RECORDER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("recorder-import-spec")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


recorder = load_recorder()
FIXTURE_ROOT = None


def minimal_stage_plan():
    control = "control-binary"
    candidate = "candidate-binary"
    return {
        "stage0": {
            "exact_commands": {
                "table_control": [control, "--mode", "table-truth"],
                "table_candidate": [candidate, "--mode", "table-truth"],
                "public_control": [control, "--mode", "public-screen"],
                "public_candidate": [candidate, "--mode", "public-screen"],
                "timing_control_template": [
                    control, "--mode", "timing", "--panel",
                    "{forced-prod|mixed-prod}", "--phase", "{warmup|measured}",
                    "--pair-index", "{decimal-index}", "--order-position", "{0|1}",
                    "--state-index", "{decimal-index}", "--nodes", "50000",
                    "--repetitions", "8", "--exact-proof-mask", "7",
                ],
                "timing_candidate_template": [
                    candidate, "--mode", "timing", "--panel",
                    "{forced-prod|mixed-prod}", "--phase", "{warmup|measured}",
                    "--pair-index", "{decimal-index}", "--order-position", "{0|1}",
                    "--state-index", "{decimal-index}", "--nodes", "50000",
                    "--repetitions", "8", "--exact-proof-mask", "7",
                ],
            },
            "corpus": {
                "seed_hex": "0x4f1bbcdc676f2b31",
                "procedural_live_states": 1000,
                "tactical_transcripts": [
                    "6/1",
                    "4/3/6/4/3/0",
                    "0/6/5/4/5/53/61/0633",
                    "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
                ],
                "total_base_states": 1004,
            },
            "machine_contract": {
                "timing": {
                    "keys_exact": [
                        "elapsed_ns", "engine", "input_signature_sha256",
                        "mode", "order_position", "pair_index", "panel",
                        "phase", "policy", "proof_mask", "repetitions",
                        "results", "result_signature_sha256", "schema",
                        "state_index",
                    ],
                },
            },
            "timing": {"panels": ["forced-prod", "mixed-prod"]},
        }
    }


_CORPUS_FIXTURE = None


def stage0_corpus_fixture():
    global _CORPUS_FIXTURE
    if _CORPUS_FIXTURE is None:
        _CORPUS_FIXTURE = recorder._build_stage0_corpus(minimal_stage_plan())
    return _CORPUS_FIXTURE


def states_equal(left, right):
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "ball", "to_move", "status", "path", "used_segments",
            "visit_count",
        )
    )


def first_complete_action(initial):
    state = recorder._copy_contest_state(initial)
    mover = state.to_move
    action = []
    for _step in range(512):
        legal = recorder._contest_legal_moves(state)
        if not legal:
            raise RuntimeError("fixture-has-no-legal-move")
        destination = legal[0]
        delta = (
            destination[0] - state.ball[0],
            destination[1] - state.ball[1],
        )
        action.append(str(recorder._STAGE0_DIRECTIONS.index(delta)))
        state = recorder._contest_apply_move(
            state, destination, "fixture-action",
        )
        if state.status != "in_progress" or state.to_move != mover:
            return "".join(action), state
    raise RuntimeError("fixture-action-did-not-end")


def activation_result_fixture(state):
    action, post = first_complete_action(state)
    return {
        "action_ascii": action,
        "attempted_depth": 1,
        "budget_exhausted": False,
        "completed_depth": 0,
        "exception": False,
        "legal": True,
        "logical_table_sha256": "1" * 64,
        "nodes": 1,
        "post_state_sha256": recorder._contest_state_hash(post),
        "root_proof_hits": 0,
        "root_proof_probes": 0,
        "root_proof_shortcut": False,
        "root_score": 0,
    }


def fixed_depth_result_fixture(state):
    action, post = first_complete_action(state)
    return {
        "action_ascii": action,
        "action_value": 7,
        "completed": True,
        "exception": False,
        "legal": True,
        "nodes": 1,
        "post_state_sha256": recorder._contest_state_hash(post),
        "root_score": 7,
        "state_unchanged": True,
    }


def timing_child_fixture(panel, pair_index, engine):
    panel_size = 8 if panel == "forced-prod" else 64
    state_index = pair_index % panel_size
    state, state_hash = recorder._timing_panel_state(
        stage0_corpus_fixture(), panel, state_index,
    )
    action, post = first_complete_action(state)
    result = {
        "action_ascii": action,
        "attempted_depth": 1,
        "budget_exhausted": False,
        "completed_depth": 0,
        "exception": False,
        "illegal": False,
        "nodes": 1,
        "post_state_sha256": recorder._contest_state_hash(post),
        "root_proof_hits": 0,
        "root_proof_probes": 0,
        "root_proof_shortcut": False,
        "root_score": 0,
    }
    results = [dict(result) for _index in range(8)]
    phase = "warmup"
    order_position = (
        (0 if engine == "control" else 1)
        if pair_index % 2 == 0
        else (0 if engine == "candidate" else 1)
    )
    command = [
        engine + "-binary", "--mode", "timing", "--panel", panel,
        "--phase", phase, "--pair-index", str(pair_index),
        "--order-position", str(order_position), "--state-index",
        str(state_index), "--nodes", "50000", "--repetitions", "8",
        "--exact-proof-mask", "7",
    ]
    workload = {
        "evaluation_entries": 131072,
        "max_nodes": 50000,
        "pair_index": pair_index,
        "panel": panel,
        "phase": phase,
        "proof_mask": 7,
        "repetitions": 8,
        "state_index": state_index,
        "state_sha256": state_hash,
        "transposition_entries": 262144,
    }
    value = {
        "elapsed_ns": 1,
        "engine": engine,
        "input_signature_sha256": recorder.sha256_bytes(
            recorder.canonical_json(workload)
        ),
        "mode": "timing",
        "order_position": order_position,
        "pair_index": pair_index,
        "panel": panel,
        "phase": phase,
        "policy": (
            "depth-primary-exact-secondary"
            if engine == "candidate" else "legacy-depth-only"
        ),
        "proof_mask": 7,
        "repetitions": 8,
        "results": results,
        "result_signature_sha256": recorder.sha256_bytes(
            recorder.canonical_json(results)
        ),
        "schema": "rank4-jacek-hybrid-tt-exact-collision-stage0-v1",
        "state_index": state_index,
    }
    return value, command


def empty_registry_fixture():
    registries = {}
    for name in recorder.REGISTRY_ORDER:
        registries[name] = []
        registries[name + "_pending"] = []
    return registries


class ExactCollisionRecorderTests(unittest.TestCase):
    def test_01_canonical_json_sorts_keys(self):
        self.assertEqual(recorder.canonical_json({"z": 1, "a": 2}), b'{"a":2,"z":1}\n')

    def test_02_canonical_json_escapes_nonascii(self):
        self.assertEqual(recorder.canonical_json({"x": "\N{SNOWMAN}"}), b'{"x":"\\u2603"}\n')

    def test_03_canonical_json_rejects_nan(self):
        with self.assertRaises(recorder.ContractError):
            recorder.canonical_json({"x": math.nan})

    def test_04_decode_json_rejects_duplicate_key(self):
        with self.assertRaises(recorder.ContractError):
            recorder.decode_json(b'{"a":1,"a":2}\n')

    def test_05_decode_json_rejects_noncanonical_spacing(self):
        with self.assertRaises(recorder.ContractError):
            recorder.decode_json(b'{"a": 1}\n')

    def test_06_contest_initial_identity_is_normalized(self):
        state = recorder._initial_contest_state()
        serialization = (
            b"papersoccer.logical-game-state.v1\n"
            b"rules=8x10;opponent_goal_only;player_to_move_loses\n"
            b"ball=4,6\n"
            b"to_move=one\n"
            b"status=in_progress\n"
            b"segments=0\n"
            b"visits=1\n"
            b"4,6:1\n"
        )
        self.assertEqual(
            recorder._contest_state_hash(state),
            hashlib.sha256(serialization).hexdigest(),
        )

    def test_07_contest_move_identity_tracks_graph(self):
        post = recorder._replay_contest_action(
            recorder._initial_contest_state(), "0", "fixture-action",
        )
        serialization = (
            b"papersoccer.logical-game-state.v1\n"
            b"rules=8x10;opponent_goal_only;player_to_move_loses\n"
            b"ball=4,5\n"
            b"to_move=two\n"
            b"status=in_progress\n"
            b"segments=1\n"
            b"4,5-4,6\n"
            b"visits=2\n"
            b"4,5:1\n"
            b"4,6:1\n"
        )
        self.assertEqual(
            recorder._contest_state_hash(post),
            hashlib.sha256(serialization).hexdigest(),
        )

    def test_08_contest_rotation_is_player_swapped_involution(self):
        initial = recorder._initial_contest_state()
        rotated = recorder._rotate_contest_state(initial)
        restored = recorder._rotate_contest_state(rotated)
        self.assertEqual(rotated.to_move, "two")
        self.assertNotEqual(
            recorder._contest_state_hash(initial),
            recorder._contest_state_hash(rotated),
        )
        self.assertTrue(states_equal(initial, restored))

    def test_09_rotated_action_replays_to_rotated_poststate(self):
        initial = recorder._initial_contest_state()
        base_post = recorder._replay_contest_action(
            initial, "0", "fixture-action",
        )
        rotated_initial = recorder._rotate_contest_state(initial)
        rotated_post = recorder._replay_contest_action(
            rotated_initial, recorder._rotate_action("0"), "fixture-action",
        )
        self.assertTrue(
            states_equal(rotated_post, recorder._rotate_contest_state(base_post))
        )

    def test_10_action_replay_rejects_empty(self):
        with self.assertRaises(recorder.ContractError):
            recorder._replay_contest_action(
                recorder._initial_contest_state(), "", "fixture-action",
            )

    def test_11_action_replay_rejects_continuation_after_turn(self):
        with self.assertRaises(recorder.ContractError):
            recorder._replay_contest_action(
                recorder._initial_contest_state(), "00", "fixture-action",
            )

    def test_12_action_replay_rejects_omitted_rebound(self):
        state = recorder._initial_contest_state()
        state.visit_count[(4, 5)] = 1
        with self.assertRaises(recorder.ContractError):
            recorder._replay_contest_action(state, "0", "fixture-action")

    def test_13_corpus_has_exact_cardinality_and_indices(self):
        corpus = stage0_corpus_fixture()
        self.assertEqual(len(corpus["pairs"]), 1004)
        self.assertEqual(len(corpus["identities"]), 1004)
        self.assertEqual(
            [item["state_index"] for item in corpus["identities"]],
            list(range(1004)),
        )

    def test_14_corpus_digest_is_deterministic(self):
        first = stage0_corpus_fixture()
        second = recorder._build_stage0_corpus(minimal_stage_plan())
        self.assertEqual(first["identities"], second["identities"])
        self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])

    def test_15_corpus_digest_binds_complete_identity_array(self):
        corpus = stage0_corpus_fixture()
        self.assertEqual(
            corpus["corpus_sha256"],
            recorder.sha256_bytes(recorder.canonical_json(corpus["identities"])),
        )

    def test_16_corpus_tactical_positions_follow_plan_order(self):
        corpus = stage0_corpus_fixture()
        for offset, transcript in enumerate(
            minimal_stage_plan()["stage0"]["corpus"]["tactical_transcripts"]
        ):
            state = recorder._reconstruct_tactical_state(transcript)
            pair = corpus["pairs"][1000 + offset]
            self.assertEqual(recorder._contest_state_hash(state), pair[2])
            self.assertEqual(
                recorder._contest_state_hash(
                    recorder._rotate_contest_state(state)
                ),
                pair[3],
            )

    def test_17_corpus_sample_rotations_are_involutions(self):
        corpus = stage0_corpus_fixture()
        for index in (0, 31, 999, 1003):
            base = corpus["pairs"][index][0]
            restored = recorder._rotate_contest_state(
                recorder._rotate_contest_state(base)
            )
            self.assertTrue(states_equal(base, restored))

    def test_18_forced_timing_panel_maps_all_eight_states(self):
        corpus = stage0_corpus_fixture()
        for index in range(8):
            state, digest = recorder._timing_panel_state(
                corpus, "forced-prod", index,
            )
            pair = corpus["pairs"][1000 + index // 2]
            self.assertIs(state, pair[index % 2])
            self.assertEqual(digest, pair[2 + index % 2])

    def test_19_mixed_timing_panel_maps_all_sixty_four_states(self):
        corpus = stage0_corpus_fixture()
        for index in range(64):
            state, digest = recorder._timing_panel_state(
                corpus, "mixed-prod", index,
            )
            pair = corpus["pairs"][index // 2]
            self.assertIs(state, pair[index % 2])
            self.assertEqual(digest, pair[2 + index % 2])

    def test_20_forced_timing_signature_and_action_validate(self):
        value, command = timing_child_fixture("forced-prod", 0, "control")
        self.assertIs(
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            ),
            value,
        )

    def test_21_mixed_timing_signature_and_action_validate(self):
        value, command = timing_child_fixture("mixed-prod", 1, "candidate")
        self.assertIs(
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            ),
            value,
        )

    def test_22_timing_rejects_malformed_input_signature(self):
        value, command = timing_child_fixture("forced-prod", 0, "control")
        value["input_signature_sha256"] = "0" * 64
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_23_timing_rejects_malformed_result_signature(self):
        value, command = timing_child_fixture("mixed-prod", 1, "candidate")
        value["result_signature_sha256"] = "0" * 64
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_24_timing_rejects_wrong_post_state(self):
        value, command = timing_child_fixture("forced-prod", 0, "control")
        value["results"][0]["post_state_sha256"] = "0" * 64
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_25_timing_rejects_bool_as_integer_result(self):
        value, command = timing_child_fixture("forced-prod", 0, "control")
        value["results"][0]["nodes"] = True
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_26_timing_rejects_bool_as_integer_routing(self):
        value, command = timing_child_fixture("forced-prod", 0, "control")
        value["pair_index"] = True
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_27_activation_action_and_post_state_validate(self):
        state = recorder._initial_contest_state()
        result = activation_result_fixture(state)
        post = recorder._validate_search_result(result, 50000, state)
        self.assertEqual(
            recorder._contest_state_hash(post), result["post_state_sha256"],
        )

    def test_28_activation_rejects_wrong_post_state(self):
        state = recorder._initial_contest_state()
        result = activation_result_fixture(state)
        result["post_state_sha256"] = "0" * 64
        with self.assertRaises(recorder.ContractError):
            recorder._validate_search_result(result, 50000, state)

    def test_29_fixed_depth_action_and_post_state_validate(self):
        state = recorder._initial_contest_state()
        result = fixed_depth_result_fixture(state)
        post = recorder._validate_fixed_depth_result(result, state)
        self.assertEqual(
            recorder._contest_state_hash(post), result["post_state_sha256"],
        )

    def test_30_fixed_depth_rejects_bool_as_integer(self):
        state = recorder._initial_contest_state()
        result = fixed_depth_result_fixture(state)
        result["nodes"] = True
        with self.assertRaises(recorder.ContractError):
            recorder._validate_fixed_depth_result(result, state)

    def test_31_timing_rejects_malformed_action(self):
        value, command = timing_child_fixture("mixed-prod", 1, "candidate")
        value["results"][0]["action_ascii"] = "8"
        with self.assertRaises(recorder.ContractError):
            recorder._validate_timing_child(
                value, command, minimal_stage_plan(), stage0_corpus_fixture(),
            )

    def test_32_registry_chain_rejects_pending_record(self):
        registries = empty_registry_fixture()
        registries["claims_pending"].append({"payload": {}})
        with self.assertRaises(recorder.ContractError):
            recorder.validate_registry_chain(minimal_stage_plan(), registries)

    def test_33_registry_chain_rejects_orphan_receipt(self):
        registries = empty_registry_fixture()
        registries["preexecution_receipts"].append({"payload": {}})
        original = recorder._replay_record
        recorder._replay_record = lambda _plan, _registry, _record: None
        try:
            with self.assertRaises(recorder.ContractError):
                recorder.validate_registry_chain(minimal_stage_plan(), registries)
        finally:
            recorder._replay_record = original

    def test_34_registry_chain_rejects_stage_without_preclaim(self):
        registries = empty_registry_fixture()
        registries["claims"].append({"payload": {}})
        original = recorder._replay_record
        recorder._replay_record = lambda _plan, _registry, _record: None
        try:
            with self.assertRaises(recorder.ContractError):
                recorder.validate_registry_chain(minimal_stage_plan(), registries)
        finally:
            recorder._replay_record = original

    def test_35_make_dependency_parser_handles_continuation(self):
        values = recorder.parse_make_dependencies(
            b"target: a\\ b.hpp \\\n c\\#d.hpp\n", "target",
        )
        self.assertEqual(values, ["a b.hpp", "c#d.hpp"])

    def test_36_make_dependency_parser_rejects_invalid_input(self):
        for raw in (b"other: a.cpp\n", b"target: a\\qb.hpp\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(recorder.ContractError):
                    recorder.parse_make_dependencies(raw, "target")

    def test_37_process_record_excludes_ancestor_chain(self):
        plan = {
            "execution_policy": {
                "process_exclusion": {"markers_exact_substrings": ["needle"]},
            },
            "frozen_inputs": {"runtime_tools": {"ps": {"argv": ["ps"]}}},
        }
        record = recorder.derive_process_record(
            b"7 0 recorder\n9 7 quiet\n", plan,
            "2026-08-15T01:02:03.123456Z", 7,
        )
        self.assertEqual(record["excluded_pids"], [7])
        self.assertTrue(record["clean"])

    def test_38_process_record_detects_conflict(self):
        plan = {
            "execution_policy": {
                "process_exclusion": {"markers_exact_substrings": ["needle"]},
            },
            "frozen_inputs": {"runtime_tools": {"ps": {"argv": ["ps"]}}},
        }
        record = recorder.derive_process_record(
            b"7 0 recorder\n9 7 needle worker\n", plan,
            "2026-08-15T01:02:03.123456Z", 7,
        )
        self.assertFalse(record["clean"])
        self.assertEqual(record["conflicts"][0]["matched_markers"], ["needle"])

    def test_39_empty_logical_entry_defaults(self):
        entry = recorder._logical_entry()
        self.assertFalse(entry["occupied"])
        self.assertEqual(entry["bound"], "Exact")

    def test_40_control_store_replaces_shallowest(self):
        before = [
            recorder._logical_entry("collision-a", 1, "Exact", way=0),
            recorder._logical_entry("collision-b", 3, "Exact", way=1),
        ]
        incoming = recorder._incoming_without_occupied(2, "Lower")
        stored, victim, _after = recorder._store_oracle(before, incoming, False)
        self.assertTrue(stored)
        self.assertEqual(victim, 0)

    def test_41_candidate_preserves_exact_at_equal_depth(self):
        before = [
            recorder._logical_entry("collision-a", 2, "Exact", way=0),
            recorder._logical_entry("collision-b", 2, "Lower", way=1),
        ]
        incoming = recorder._incoming_without_occupied(2, "Lower")
        stored, victim, _after = recorder._store_oracle(before, incoming, True)
        self.assertTrue(stored)
        self.assertEqual(victim, 1)

    def test_42_candidate_rejects_nonexact_against_exact(self):
        before = [
            recorder._logical_entry("collision-a", 2, "Exact", way=0),
            recorder._logical_entry("collision-b", 3, "Lower", way=1),
        ]
        incoming = recorder._incoming_without_occupied(2, "Lower")
        stored, victim, _after = recorder._store_oracle(before, incoming, True)
        self.assertFalse(stored)
        self.assertIsNone(victim)

    def test_43_stage0_schedule_has_exact_cardinality(self):
        schedule, timeouts = recorder.build_stage0_schedule(minimal_stage_plan())
        self.assertEqual((len(schedule), len(timeouts)), (1324, 1324))

    def test_44_stage0_schedule_alternates_pair_order(self):
        schedule, _timeouts = recorder.build_stage0_schedule(minimal_stage_plan())
        self.assertEqual(schedule[4][0], "control-binary")
        self.assertEqual(schedule[5][0], "candidate-binary")
        self.assertEqual(schedule[6][0], "candidate-binary")
        self.assertEqual(schedule[7][0], "control-binary")

    def test_45_content_addressed_record_name(self):
        payload = {"schema": recorder.RECORD_SCHEMAS["preexecution_receipt"]}
        raw = recorder.canonical_json(payload)
        name = recorder.sha256_bytes(raw) + ".json"
        self.assertEqual(
            recorder.validate_record_name("preexecution_receipts", name, payload, raw),
            recorder.sha256_bytes(raw),
        )

    def test_46_plan_reference_is_frozen(self):
        reference = recorder.plan_reference()
        self.assertEqual(
            reference["commit"],
            "ee7b01066134ba7c32aeeb9468d72105f4fae4b2",
        )
        self.assertEqual(
            reference["blob_sha256"],
            "f9474c8bac2d9692928083377d876d22a1722c88a198c2523fad39dd8db76e91",
        )

    def test_47_recorder_source_contains_containment_and_chain_guards(self):
        with open(RECORDER_PATH, "r", encoding="utf-8") as source:
            text = source.read()
        for token in (
            "O_NOFOLLOW", "O_NONBLOCK", "start_new_session=True",
            "_kill_process_group(process)",
            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))", "LOCK_NB",
            "def validate_registry_chain(", "receipt-orphan",
            "claim-after-unaccepted-report",
            "replay-decision-selection-derived",
            "MACOS_PYTHON_BOOTSTRAP_ENVIRONMENT",
            "outer-environment-normalized",
            'sum(item.get("bytes", 0) for item in records)',
            "def _external_provenance_only_paths(",
            "external-nlink-exception-baseline",
            "external-nlink-provenance-used",
            "external-live-closure-count",
            "if path in provenance_only:",
            "def dependency_records(plan,",
            "dependency-provenance-only",
            "receipt-dependency-provenance-only",
            "read_regular_nofollow(path, expected_mode, 1, max_bytes)",
            "GENERATED_OVERLAY_SCHEMA",
            "def _materialize_generated_plan(",
            "generated-effective-protected-literal",
            "generated-bank-argv-boundary",
            "def _preexecution_aggregate_seconds(",
            "generated-trace-free-binary-record",
            "generated-trace-free-prose-keys",
            "trace-free-compile-cleanup",
            "receipt-trace-free",
            "generated-warning-clean-command",
            "generated-host-projection-keys",
            "generated-governance-projection-paths",
        ):
            self.assertIn(token, text)
        self.assertNotIn(
            'for item in records if item["type"] == "regular"', text
        )
        self.assertNotIn("def parse_link_trace(", text)

    def test_48_gate_source_contains_safe_reader_contract(self):
        with open(GATE_PATH, "r", encoding="utf-8") as source:
            gate = source.read()
        with open(ENGINE_PATH, "r", encoding="utf-8") as source:
            engine = source.read()
        combined = gate + engine
        for token in (
            "O_NOFOLLOW", "O_NONBLOCK", "fstatat", "openat", "--expected-bytes",
            "safe_bank_reader_self_test=pass", "PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET",
            "choose_hybrid_control", "S_ISREG", "--generated-pairs",
            "--play-depths", "generate_all_banks", "select_generated_banks",
            "preregistered-generated-public-rules",
        ):
            self.assertIn(token, combined)


def _remove_fixture_tree(path):
    if os.path.lexists(path):
        shutil.rmtree(path)


def main():
    global FIXTURE_ROOT
    try:
        temporary_root = os.environ.get("TMPDIR")
        if not temporary_root or not os.path.isabs(temporary_root):
            return 1
        if not os.path.isdir(temporary_root) or os.path.islink(temporary_root):
            return 1
        FIXTURE_ROOT = os.path.join(temporary_root, "focused-test")
        if os.path.lexists(FIXTURE_ROOT):
            return 1
        os.mkdir(FIXTURE_ROOT, 0o700)
        if oct(os.stat(FIXTURE_ROOT, follow_symlinks=False).st_mode & 0o7777) != "0o700":
            return 1
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExactCollisionRecorderTests)
        if suite.countTestCases() != 48:
            return 1
        buffer = io.StringIO()
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
        if result.testsRun != 48 or not result.wasSuccessful():
            return 1
        _remove_fixture_tree(FIXTURE_ROOT)
        descriptor = os.open(temporary_root, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if os.listdir(temporary_root) != []:
            return 1
        os.write(1, b"tt_exact_collision_tests=pass tests=48\n")
        return 0
    except BaseException:
        return 1
    finally:
        if FIXTURE_ROOT is not None:
            try:
                _remove_fixture_tree(FIXTURE_ROOT)
                temporary_root = os.environ.get("TMPDIR")
                if temporary_root and os.path.isdir(temporary_root):
                    descriptor = os.open(
                        temporary_root,
                        os.O_RDONLY | os.O_CLOEXEC
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
