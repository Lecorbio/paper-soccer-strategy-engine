#!/usr/bin/env python3
"""Record the one frozen DEVELOPMENT d20 PositionKey-cache clock ablation.

This recorder has no bank, clock, proof-mask, transcript, or retry knobs.  It
compares the exact committed private-cache hybrid against the content-addressed
archived pre-fastpath engine.  Validation and final banks are not addressable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable

import record_rank4_jacek_hybrid_null_fastpath_clock as base


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/position_key_cache_clock"
PLAN = OUTPUT / "plan.json"
ROLLBACK = ROOT / (
    "results/rank_4_jacek_hybrid/gates/null_fastpath_clock/rollback/"
    "33c1146b41c5bfd2be07283309d6614872ef9bc17ff7619089cdb35d03b74a62.json"
)
SOLE_EDGE_DECISION = ROOT / (
    "results/rank_4_jacek_hybrid/gates/sole_legal_edge_clock/selection/"
    "e36314e33d9ca66f9c901c0e99dc613e10b43e1941668cc59ad5e6b3d8a0b5af.json"
)
PROTOTYPE_DIRECTORY = (
    ROOT / "results/rank_4_jacek_hybrid/position_key_components_prototype"
)
PROTOTYPE_PASS = PROTOTYPE_DIRECTORY / "PASS.md"
PROTOTYPE_PATCH = PROTOTYPE_DIRECTORY / "private_integration.patch"
PROTOTYPE_TIMING = PROTOTYPE_DIRECTORY / "timing.json"
LOCK = ROOT / "build/rank4-jacek-hybrid-position-key-cache-clock.lock"
BENCHMARK_LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
CLAIMS = OUTPUT / "claims"
FOCUSED_TEST = ROOT / (
    "tests/codingame/"
    "test_rank4_jacek_hybrid_position_key_cache_clock_recorder.py"
)
BASE_RECORDER = Path(base.__file__).resolve()
COMMON_RECORDER = Path(base.common.__file__).resolve()

GATE_TARGET = base.GATE_TARGET
GATE = base.GATE
BANK = base.BANK
RUN_TIMEOUT_SECONDS = base.RUN_TIMEOUT_SECONDS

SCHEMA = "rank4-jacek-hybrid-position-key-cache-clock-v1"
CLAIM_SCHEMA = "rank4-jacek-hybrid-position-key-cache-attempt-claim-v1"
PLAN_SCHEMA = "rank4-jacek-hybrid-position-key-cache-clock-plan-v1"
CLASSIFICATION = (
    "development-position-key-cache-clock-ablation-not-final-qualification"
)
CAMPAIGN_ID = base.CAMPAIGN_ID
CAMPAIGN_T0_UTC = base.CAMPAIGN_T0_UTC

INTEGRATION_COMMIT = "b1b7777ca7cb8687eab3527d36a3a83827e6a7db"
PROTOTYPE_BASE_COMMIT = "db8aa306a10fc95548babc57234c510e55d74e69"
CONTROL_SOURCE_COMMIT = base.CONTROL_SOURCE_COMMIT

EXPECTED_PRIVATE_HEADER_SHA256 = (
    "254ea592b3bca934dbfbbb5ebc838411b49a09ef3ec8b3d8ddc332bb7079b011"
)
EXPECTED_CANDIDATE_BOT_SHA256 = (
    "439e1b17124ea7c81dd2c3cce66342953d5ed981e19d3947a42ea626ef19f2d2"
)
EXPECTED_SOURCES_SHA256 = (
    "5f2fdce4b375a8fd91c73141d87f2b12a8aa3e61d9b18c46c442b325e51cbdda"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "47f44e8e62d3aaa2a48f6eea6fca4d17cfbbfd3ff9a5ac01ca84b1e0bf4cca03"
)
EXPECTED_CANDIDATE_TEST_SHA256 = (
    "ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697"
)
EXPECTED_KEY_TEST_SHA256 = (
    "50d2149929521f5ed5b1f6e64958c043df5f6fbc94cb45550a7ee28e797dc0d5"
)
EXPECTED_SHARED_HEADER_SHA256 = (
    "0a13e89e183666ce89e38d1eded1b26c02eaaba5460ba7e3ede9fda5d5e1dd04"
)
EXPECTED_CONTROL_BOT_SHA256 = base.EXPECTED_CONTROL_BOT_SHA256
EXPECTED_CONTROL_SOURCE_SHA256 = base.EXPECTED_CONTROL_SOURCE_SHA256
EXPECTED_BANK_SHA256 = base.EXPECTED_BANK_SHA256
EXPECTED_PLAN_SHA256 = (
    "d098403882efa9e819d68ae8c7c4159e6500a3d5d0cf09ce96cd09cebf12ab74"
)
EXPECTED_ROLLBACK_SHA256 = (
    "33c1146b41c5bfd2be07283309d6614872ef9bc17ff7619089cdb35d03b74a62"
)
EXPECTED_SOLE_EDGE_DECISION_SHA256 = (
    "e36314e33d9ca66f9c901c0e99dc613e10b43e1941668cc59ad5e6b3d8a0b5af"
)
EXPECTED_PROTOTYPE_PASS_SHA256 = (
    "eecefab6db9910e0534da515e6de8370d0c41695d36b60fd754798119454ca6c"
)
EXPECTED_PROTOTYPE_PATCH_SHA256 = (
    "59640f953414132e90b31b2a6e2a2dfa81a3e7dbaf83e1d236befa6f2a1f2997"
)
EXPECTED_PROTOTYPE_TIMING_SHA256 = (
    "b91990bad9d51f2f4b1ee01c3a88e5e22ad32e8ea5fb8bb20c0755b46959ad1d"
)
EXPECTED_CONTROL_MANIFEST_SHA256 = base.EXPECTED_MANIFEST_SHA256
EXPECTED_CMAKE_SHA256 = (
    "6b02129ac4c192f3ec61e915f3994da15b576063df6e68d8fdabe2e9a740dfe8"
)
EXPECTED_REFERENCE_WRAPPER_SHA256 = (
    "8f2842afff80fd6054b1ea123ae3852c985aaff453040638e72c2ca5852a5223"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "15e2c5ac012fec41798c2995d22e049c83237e217c2b13f76730cacaaacbba01"
)
EXPECTED_BASE_RECORDER_SHA256 = (
    "8b1ffbe058d3493c1df0741b322149a35586c0f2e85f90f69e7e30ad90d79e59"
)
EXPECTED_COMMON_RECORDER_SHA256 = (
    "1c9a2a505578b02866aa3c2d64231e048ea46a301d910bb04aef345552bf9aca"
)

BOT_DIRECTORY = base.BOT_DIRECTORY
PRIVATE_HEADER = BOT_DIRECTORY / "mcts_internal.hpp"
CANDIDATE_BOT = base.CANDIDATE_BOT
SOURCES = BOT_DIRECTORY / "sources.txt"
CANDIDATE_SOURCE = base.CANDIDATE_SOURCE
CANDIDATE_TEST = base.CANDIDATE_TEST
KEY_TEST = BOT_DIRECTORY / "position_key_cache_test.cpp"
SHARED_HEADER = ROOT / "src/bots/mcts_internal.hpp"
CONTROL_MANIFEST = base.CONTROL_MANIFEST
CONTROL_BOT = base.CONTROL_BOT
CONTROL_SOURCE = base.CONTROL_SOURCE

CMAKE = ROOT / "CMakeLists.txt"
COMPARISON_DRIVER = BOT_DIRECTORY / "comparison_gate.cpp"
COMPARISON_ENGINE = BOT_DIRECTORY / "comparison_gate_engine.hpp"
COMPARISON_CANDIDATE = BOT_DIRECTORY / "comparison_gate_hybrid.cpp"
COMPARISON_MAIN = BOT_DIRECTORY / "comparison_gate_null_fastpath.cpp"
COMPARISON_REFERENCE = BOT_DIRECTORY / "comparison_gate_prefastpath.cpp"

TARGET_DIRECTORY = ROOT / f"build/CMakeFiles/{GATE_TARGET}.dir"
CANDIDATE_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_hybrid.cpp.o.d"
)
REFERENCE_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_prefastpath.cpp.o.d"
)
MAIN_DEPFILE = TARGET_DIRECTORY / (
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_null_fastpath.cpp.o.d"
)
OBJECT_FILES = tuple(
    path.with_suffix("") for path in
    (CANDIDATE_DEPFILE, REFERENCE_DEPFILE, MAIN_DEPFILE)
)

STATIC_INPUTS = (
    RECORDER,
    FOCUSED_TEST,
    BASE_RECORDER,
    COMMON_RECORDER,
    CMAKE,
    PLAN,
    ROLLBACK,
    SOLE_EDGE_DECISION,
    PROTOTYPE_PASS,
    PROTOTYPE_PATCH,
    PROTOTYPE_TIMING,
    CONTROL_MANIFEST,
    CONTROL_BOT,
    CONTROL_SOURCE,
    PRIVATE_HEADER,
    CANDIDATE_BOT,
    SOURCES,
    CANDIDATE_SOURCE,
    CANDIDATE_TEST,
    KEY_TEST,
    SHARED_HEADER,
    BOT_DIRECTORY / "replay_book.hpp",
    BOT_DIRECTORY / "replay_value_model.hpp",
    BOT_DIRECTORY / "teacher_residual_model.hpp",
    COMPARISON_DRIVER,
    COMPARISON_ENGINE,
    COMPARISON_CANDIDATE,
    COMPARISON_MAIN,
    COMPARISON_REFERENCE,
    ROOT / "src/opening_bank/opening_bank.cpp",
    ROOT / "src/opening_bank/opening_bank_internal.hpp",
    ROOT / "src/core/rules.cpp",
    ROOT / "src/core/geometry.cpp",
    ROOT / "include/papersoccer/rules.hpp",
    ROOT / "include/papersoccer/geometry.hpp",
    ROOT / "include/papersoccer/types.hpp",
    BANK,
)

BUILD_INPUTS = (
    ROOT / "build/CMakeCache.txt",
    TARGET_DIRECTORY / "flags.make",
    TARGET_DIRECTORY / "link.txt",
    CANDIDATE_DEPFILE,
    REFERENCE_DEPFILE,
    MAIN_DEPFILE,
    *OBJECT_FILES,
    ROOT / "build/libpapersoccer_opening_bank_support.a",
    ROOT / "build/libpapersoccer_core.a",
    GATE,
)

INTEGRATION_BOUND_PATHS = (
    "CMakeLists.txt",
    "tools/record_rank4_jacek_hybrid_null_fastpath_clock.py",
    "tools/record_rank4_jacek_hybrid_proof_scope_clock.py",
    "submissions/codingame/bots/rank_4_jacek_hybrid/mcts_internal.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/sources.txt",
    "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/submission_test.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/position_key_cache_test.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_value_model.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/teacher_residual_model.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_engine.hpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_hybrid.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_null_fastpath.cpp",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_prefastpath.cpp",
    "src/bots/mcts_internal.hpp",
    "src/opening_bank/opening_bank.cpp",
    "src/opening_bank/opening_bank_internal.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "include/papersoccer/rules.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/types.hpp",
)

canonical_json = base.canonical_json
sha256_bytes = base.sha256_bytes
persist_content_addressed_report = base.persist_content_addressed_report
expected_configuration = base.expected_configuration
command_for_gate = base.command_for_gate
validate_gate_stdout = base.validate_gate_stdout
selection_errors = base.selection_errors
parse_process_table = base.parse_process_table
process_preflight_from_table = base.process_preflight_from_table


def validate_imported_recorder_files() -> None:
    expected = (
        (BASE_RECORDER, EXPECTED_BASE_RECORDER_SHA256, 38_594),
        (COMMON_RECORDER, EXPECTED_COMMON_RECORDER_SHA256, 23_804),
    )
    for path, digest, size in expected:
        raw = path.read_bytes()
        if (len(raw) != size or hashlib.sha256(raw).hexdigest() != digest or
                not all(byte < 128 for byte in raw)):
            raise ValueError(f"imported recorder identity mismatch: {path}")


def validate_canonical_json_file(path: Path, digest: str) -> Any:
    raw = path.read_bytes()
    if sha256_bytes(raw) != digest:
        raise ValueError(f"frozen JSON SHA-256 mismatch: {path}")
    payload = json.loads(raw)
    if canonical_json(payload) != raw:
        raise ValueError(f"frozen JSON is not canonical: {path}")
    return payload


def validate_exact_json_file(path: Path, digest: str) -> Any:
    raw = path.read_bytes()
    if sha256_bytes(raw) != digest:
        raise ValueError(f"frozen JSON SHA-256 mismatch: {path}")
    return json.loads(raw)


def _expected_candidate_artifacts() -> list[dict[str, Any]]:
    return [
        {"bytes": 16_174, "path": str(PRIVATE_HEADER.relative_to(ROOT)),
         "role": "hybrid-private-position-key-header",
         "sha256": EXPECTED_PRIVATE_HEADER_SHA256},
        {"bytes": 63_158, "path": str(CANDIDATE_BOT.relative_to(ROOT)),
         "role": "engine-source", "sha256": EXPECTED_CANDIDATE_BOT_SHA256},
        {"bytes": 461, "path": str(SOURCES.relative_to(ROOT)),
         "role": "submission-source-manifest", "sha256": EXPECTED_SOURCES_SHA256},
        {"bytes": 95_750, "path": str(CANDIDATE_SOURCE.relative_to(ROOT)),
         "role": "upload-source", "sha256": EXPECTED_CANDIDATE_SOURCE_SHA256},
        {"bytes": 39_137, "path": str(CANDIDATE_TEST.relative_to(ROOT)),
         "role": "source-contract-test", "sha256": EXPECTED_CANDIDATE_TEST_SHA256},
        {"bytes": 6_530, "path": str(KEY_TEST.relative_to(ROOT)),
         "role": "permanent-position-key-cache-test", "sha256": EXPECTED_KEY_TEST_SHA256},
        {"bytes": 14_668, "path": str(SHARED_HEADER.relative_to(ROOT)),
         "role": "shared-header-unchanged-invariant",
         "sha256": EXPECTED_SHARED_HEADER_SHA256},
    ]


def validate_plan_and_lineage() -> dict[str, Any]:
    validate_imported_recorder_files()
    plan = validate_canonical_json_file(PLAN, EXPECTED_PLAN_SHA256)
    rollback = validate_exact_json_file(ROLLBACK, EXPECTED_ROLLBACK_SHA256)
    sole_decision = validate_exact_json_file(
        SOLE_EDGE_DECISION, EXPECTED_SOLE_EDGE_DECISION_SHA256
    )
    timing = validate_canonical_json_file(
        PROTOTYPE_TIMING, EXPECTED_PROTOTYPE_TIMING_SHA256
    )
    pass_raw = PROTOTYPE_PASS.read_bytes()
    patch_raw = PROTOTYPE_PATCH.read_bytes()
    if (sha256_bytes(pass_raw) != EXPECTED_PROTOTYPE_PASS_SHA256 or
            len(pass_raw) != 10_038 or
            sha256_bytes(patch_raw) != EXPECTED_PROTOTYPE_PATCH_SHA256 or
            len(patch_raw) != 24_814):
        raise ValueError("PositionKey prototype receipt/patch identity mismatch")

    if (plan.get("schema") != PLAN_SCHEMA or
            plan.get("classification") != CLASSIFICATION or
            plan.get("final_qualification") is not False):
        raise ValueError("PositionKey plan schema/classification mismatch")
    candidate = plan.get("candidate", {})
    if (candidate.get("integration_commit") != INTEGRATION_COMMIT or
            candidate.get("prototype_base_commit") != PROTOTYPE_BASE_COMMIT or
            candidate.get("exact_proof_mask") != 7 or
            candidate.get("artifacts") != _expected_candidate_artifacts()):
        raise ValueError("PositionKey plan candidate binding mismatch")
    if len(candidate.get("invariants", [])) != 7:
        raise ValueError("PositionKey plan invariant set mismatch")

    expected_bank = {
        "depth": 20, "games": 76, "games_per_physical_color": 38,
        "path": str(BANK.relative_to(ROOT)), "role": "development",
        "seed": 4_624_785_204_876_369_057,
        "sha256": EXPECTED_BANK_SHA256,
    }
    if plan.get("bank") != expected_bank:
        raise ValueError("PositionKey plan DEVELOPMENT bank mismatch")
    expected_command = {
        "candidate_clock_ms": [800, 165], "candidate_nodes": 3_000_000,
        "executable_target": GATE_TARGET, "max_turns": 320,
        "operational_clock_ms": [1000, 200],
        "reference_clock_ms": [800, 165], "reference_nodes": 3_000_000,
        "retain_transcripts": False,
    }
    if plan.get("command") != expected_command:
        raise ValueError("PositionKey plan command mismatch")
    expected_policy = {
        "atomic_prerun_attempt_claim_required": True,
        "attempts_per_exact_committed_head_and_inputs": 1,
        "candidate_files_must_equal_head_blobs": True,
        "candidate_gate_dependency_must_use_private_header": True,
        "clean_process_preflight_required_before_build_and_run": True,
        "clean_tracked_tree_required": True,
        "fresh_build_required": True,
        "fresh_generator_check_required": True,
        "full_dependency_and_binary_hashes_required_before_and_after": True,
        "reference_gate_dependency_must_use_shared_header": True,
        "run_only_after_plan_recorder_candidate_and_tests_are_committed": True,
        "shared_benchmark_lock_held_interval": (
            "before-fresh-build-through-content-addressed-report-readback"
        ),
        "shared_benchmark_lock_mode": "nonblocking-exclusive",
        "shared_benchmark_lock_path": str(BENCHMARK_LOCK),
        "validation_or_final_banks_forbidden": True,
    }
    if plan.get("evidence_policy") != expected_policy:
        raise ValueError("PositionKey plan evidence policy mismatch")
    expected_evidence_bindings = {
        "focused_recorder_test": {
            "bytes": 15_907,
            "path": str(FOCUSED_TEST.relative_to(ROOT)),
            "role": "synthetic-recorder-contract-test",
            "sha256": EXPECTED_FOCUSED_TEST_SHA256,
        },
        "imported_recorders": [
            {
                "bytes": 38_594,
                "integration_commit": INTEGRATION_COMMIT,
                "path": str(BASE_RECORDER.relative_to(ROOT)),
                "role": "imported-clock-validation-base",
                "sha256": EXPECTED_BASE_RECORDER_SHA256,
            },
            {
                "bytes": 23_804,
                "integration_commit": INTEGRATION_COMMIT,
                "path": str(COMMON_RECORDER.relative_to(ROOT)),
                "role": "imported-proof-accounting-common",
                "sha256": EXPECTED_COMMON_RECORDER_SHA256,
            },
        ],
    }
    if plan.get("evidence_bindings") != expected_evidence_bindings:
        raise ValueError("PositionKey plan recorder/test evidence binding mismatch")
    expected_claim_policy = {
        "canonical_claim_required": True,
        "claim_directory": str(CLAIMS.relative_to(ROOT)),
        "claim_fields": [
            "schema", "attempt_id", "attempt_key", "head", "inputs_sha256",
            "plan_sha256", "claimed_utc", "prelaunch_checks_complete",
            "retry_policy",
        ],
        "claim_filename": "{attempt_id}.claim.json",
        "claim_retained_after_creation": True,
        "creation_flags": ["O_WRONLY", "O_CREAT", "O_EXCL"],
        "creation_point": (
            "after every static, build, generator, identity, prior-evidence, "
            "and process-preflight check; immediately before gate launch"
        ),
        "existing_evidence_rejected": [
            "canonical-attempt-claim", "canonical-content-addressed-report"
        ],
        "postclaim_failure_consumes_attempt": True,
        "preclaim_failure_does_not_consume_attempt": True,
        "reread_rehash_and_canonical_validation_required": True,
        "schema": CLAIM_SCHEMA,
    }
    if plan.get("attempt_claim_policy") != expected_claim_policy:
        raise ValueError("PositionKey plan attempt-claim policy mismatch")
    expected_thresholds = {
        "candidate_first_ms_max_lt": 990,
        "candidate_first_ms_p99_lt": 900,
        "candidate_later_ms_max_lt": 198,
        "candidate_later_ms_p99_lt": 180,
        "candidate_wins_by_color_min": [19, 19],
        "candidate_wins_min": 38,
        "failures_max": 0,
        "progress": (
            "candidate_depth_avg >= reference_depth_avg OR "
            "candidate_nodes_avg > reference_nodes_avg"
        ),
        "reference_first_ms_max_lt": 990,
        "reference_first_ms_p99_lt": 900,
        "reference_later_ms_max_lt": 198,
        "reference_later_ms_p99_lt": 180,
        "unfinished_max": 0,
    }
    if plan.get("thresholds") != expected_thresholds:
        raise ValueError("PositionKey plan thresholds mismatch")

    reference = plan.get("reference", {})
    if (reference.get("archived_bot_sha256") != EXPECTED_CONTROL_BOT_SHA256 or
            reference.get("archived_source_sha256") != EXPECTED_CONTROL_SOURCE_SHA256 or
            reference.get("source_commit") != CONTROL_SOURCE_COMMIT or
            reference.get("shared_header_sha256") != EXPECTED_SHARED_HEADER_SHA256 or
            reference.get("exact_proof_mask") != 7 or
            reference.get("function") != "choose_prefastpath" or
            reference.get("slot_echo") != "rank4"):
        raise ValueError("PositionKey plan reference binding mismatch")
    lineage = plan.get("lineage", {})
    prototype = lineage.get("position_key_preintegration", {})
    if (prototype.get("pass_receipt", {}).get("sha256") !=
            EXPECTED_PROTOTYPE_PASS_SHA256 or
            prototype.get("private_integration_patch", {}).get("sha256") !=
            EXPECTED_PROTOTYPE_PATCH_SHA256 or
            prototype.get("timing_receipt", {}).get("sha256") !=
            EXPECTED_PROTOTYPE_TIMING_SHA256 or
            lineage.get("pre_singleton_rollback", {}).get("sha256") !=
            EXPECTED_ROLLBACK_SHA256 or
            lineage.get("sole_legal_edge_decision", {}).get("sha256") !=
            EXPECTED_SOLE_EDGE_DECISION_SHA256):
        raise ValueError("PositionKey plan lineage mismatch")

    pass_text = pass_raw.decode("ascii")
    required_pass_phrases = (
        "Hybrid-private PositionKey component cache: preintegration PASS",
        "PASS the frozen preintegration gate",
        "shared `src/bots/mcts_internal.hpp`",
        "read no validation or final opening bank",
    )
    if any(phrase not in pass_text for phrase in required_pass_phrases):
        raise ValueError("PositionKey PASS receipt semantic marker mismatch")
    if (timing.get("schema") !=
            "rank4-position-key-component-safe-private-microbench-v1" or
            timing.get("thresholds") != {
                "median_ratio_max": 0.99, "p99_ratio_max": 1.005
            } or len(timing.get("panels", [])) != 2):
        raise ValueError("PositionKey timing receipt semantic mismatch")
    for panel in timing["panels"]:
        if (panel.get("ratios", {}).get("median", 2.0) > 0.99 or
                panel.get("ratios", {}).get("p99", 2.0) > 1.005):
            raise ValueError("PositionKey timing receipt contains a failed panel")
    if (rollback.get("schema") !=
            "rank4-jacek-hybrid-null-fastpath-rollback-v1" or
            rollback.get("rollback_base", {}).get("operational_exact_proof_mask") != 7):
        raise ValueError("pre-cache rollback lineage mismatch")
    if (sole_decision.get("decision", {}).get("status") !=
            "sole-legal-edge-rejected-mandatory-rollback" or
            sole_decision.get("decision", {}).get("selected_algorithm") !=
            "pre-singleton-mask7-hybrid"):
        raise ValueError("sole-edge rejection lineage mismatch")
    return {
        "plan": plan,
        "rollback": rollback,
        "sole_edge_decision": sole_decision,
        "prototype": {
            "pass": {"bytes": len(pass_raw), "sha256": sha256_bytes(pass_raw)},
            "patch": {"bytes": len(patch_raw), "sha256": sha256_bytes(patch_raw)},
            "timing": timing,
        },
    }


def require_committed_bindings(
    head: str, *, require_gate_infrastructure: bool = True,
) -> dict[str, Any]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", INTEGRATION_COMMIT, head],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("PositionKey integration commit is not an ancestor of HEAD")
    if require_gate_infrastructure:
        for path in (RECORDER, FOCUSED_TEST, PLAN):
            relative = str(path.relative_to(ROOT))
            if base.git_blob(head, relative) != path.read_bytes():
                raise ValueError(
                    f"gate infrastructure is not the committed HEAD blob: {relative}"
                )
    for relative in INTEGRATION_BOUND_PATHS:
        if base.git_blob(INTEGRATION_COMMIT, relative) != (ROOT / relative).read_bytes():
            raise ValueError(f"integration dependency changed since {INTEGRATION_COMMIT}: {relative}")
    if base.git_blob(
        PROTOTYPE_BASE_COMMIT, "src/bots/mcts_internal.hpp"
    ) != SHARED_HEADER.read_bytes():
        raise ValueError("shared MCTS header changed from the prototype base")
    if base.git_blob(
        CONTROL_SOURCE_COMMIT, "src/bots/mcts_internal.hpp"
    ) != SHARED_HEADER.read_bytes():
        raise ValueError("archived reference is not bound to the unchanged shared header")
    if base.git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp",
    ) != CONTROL_BOT.read_bytes():
        raise ValueError("archived reference bot differs from its source commit")
    if base.git_blob(
        CONTROL_SOURCE_COMMIT,
        "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp",
    ) != CONTROL_SOURCE.read_bytes():
        raise ValueError("archived reference source differs from its source commit")

    wrapper = COMPARISON_REFERENCE.read_text(encoding="ascii")
    shared_include = '#include "../../../../src/bots/mcts_internal.hpp"'
    archive_include = '#include "../../../../results/rank_4_jacek_hybrid/controls/'
    if (shared_include not in wrapper or archive_include not in wrapper or
            wrapper.index(shared_include) > wrapper.index(archive_include)):
        raise ValueError("archived reference wrapper does not bind the shared header first")
    return validate_plan_and_lineage()


def identity_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": identity_label(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ascii": all(byte < 128 for byte in data),
    }


def identities_for_paths(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    unique = {path.resolve() for path in paths}
    return {identity_label(path): file_identity(path) for path in sorted(unique)}


def validate_exact_file_identities(identities: dict[str, dict[str, Any]]) -> None:
    expected = {
        PRIVATE_HEADER: (EXPECTED_PRIVATE_HEADER_SHA256, 16_174, True),
        CANDIDATE_BOT: (EXPECTED_CANDIDATE_BOT_SHA256, 63_158, True),
        SOURCES: (EXPECTED_SOURCES_SHA256, 461, True),
        CANDIDATE_SOURCE: (EXPECTED_CANDIDATE_SOURCE_SHA256, 95_750, True),
        CANDIDATE_TEST: (EXPECTED_CANDIDATE_TEST_SHA256, 39_137, True),
        KEY_TEST: (EXPECTED_KEY_TEST_SHA256, 6_530, True),
        SHARED_HEADER: (EXPECTED_SHARED_HEADER_SHA256, 14_668, True),
        CONTROL_BOT: (EXPECTED_CONTROL_BOT_SHA256, 62_777, True),
        CONTROL_SOURCE: (EXPECTED_CONTROL_SOURCE_SHA256, 94_004, True),
        BANK: (EXPECTED_BANK_SHA256, 13_150, True),
        PLAN: (EXPECTED_PLAN_SHA256, 8_138, True),
        ROLLBACK: (EXPECTED_ROLLBACK_SHA256, 5_908, True),
        SOLE_EDGE_DECISION: (EXPECTED_SOLE_EDGE_DECISION_SHA256, 8_606, True),
        PROTOTYPE_PASS: (EXPECTED_PROTOTYPE_PASS_SHA256, 10_038, True),
        PROTOTYPE_PATCH: (EXPECTED_PROTOTYPE_PATCH_SHA256, 24_814, True),
        PROTOTYPE_TIMING: (EXPECTED_PROTOTYPE_TIMING_SHA256, 935, True),
        CONTROL_MANIFEST: (EXPECTED_CONTROL_MANIFEST_SHA256, 1_334, True),
        CMAKE: (EXPECTED_CMAKE_SHA256, 53_604, True),
        COMPARISON_REFERENCE: (EXPECTED_REFERENCE_WRAPPER_SHA256, 2_687, True),
        FOCUSED_TEST: (EXPECTED_FOCUSED_TEST_SHA256, 15_907, True),
        BASE_RECORDER: (EXPECTED_BASE_RECORDER_SHA256, 38_594, True),
        COMMON_RECORDER: (EXPECTED_COMMON_RECORDER_SHA256, 23_804, True),
    }
    for path, (digest, size, ascii_expected) in expected.items():
        key = identity_label(path)
        identity = identities.get(key)
        if identity is None or identity.get("sha256") != digest:
            raise ValueError(f"exact input SHA-256 mismatch: {key}")
        if identity.get("bytes") != size:
            raise ValueError(f"exact input byte count mismatch: {key}")
        if identity.get("ascii") is not ascii_expected:
            raise ValueError(f"exact input ASCII mismatch: {key}")
    if identities[identity_label(CANDIDATE_SOURCE)]["bytes"] > 99_999:
        raise ValueError("candidate exceeds CodinGame source limit")


def parse_make_depfile(path: Path) -> set[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise ValueError(f"malformed compiler depfile: {path}")
    _, dependency_text = text.split(":", 1)
    dependencies: set[Path] = set()
    for token in shlex.split(dependency_text):
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = ROOT / dependency
        dependencies.add(dependency.resolve())
    if not dependencies:
        raise ValueError(f"empty compiler depfile: {path}")
    return dependencies


def validate_dependency_routing(
    candidate_dependencies: set[Path], reference_dependencies: set[Path],
) -> dict[str, Any]:
    private = PRIVATE_HEADER.resolve()
    shared = SHARED_HEADER.resolve()
    control_bot = CONTROL_BOT.resolve()
    evidence = {
        "candidate_private_header": private in candidate_dependencies,
        "candidate_shared_header": shared in candidate_dependencies,
        "reference_private_header": private in reference_dependencies,
        "reference_shared_header": shared in reference_dependencies,
        "reference_archived_bot": control_bot in reference_dependencies,
    }
    if not evidence["candidate_private_header"]:
        raise ValueError("candidate gate object is not compiled with the private header")
    if evidence["candidate_shared_header"]:
        raise ValueError("candidate gate object unexpectedly depends on the shared header")
    if not evidence["reference_shared_header"]:
        raise ValueError("reference gate object is not compiled with the shared header")
    if not evidence["reference_archived_bot"]:
        raise ValueError("reference gate object is not compiled from the archived bot")
    evidence["passed"] = True
    return evidence


def collect_full_input_paths() -> tuple[list[Path], dict[str, Any]]:
    candidate_dependencies = parse_make_depfile(CANDIDATE_DEPFILE)
    reference_dependencies = parse_make_depfile(REFERENCE_DEPFILE)
    main_dependencies = parse_make_depfile(MAIN_DEPFILE)
    routing = validate_dependency_routing(
        candidate_dependencies, reference_dependencies
    )
    paths = set(STATIC_INPUTS) | set(BUILD_INPUTS)
    paths |= candidate_dependencies | reference_dependencies | main_dependencies
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise ValueError("missing gate dependency inputs: " + ", ".join(missing))
    routing["candidate_dependency_count"] = len(candidate_dependencies)
    routing["reference_dependency_count"] = len(reference_dependencies)
    routing["main_dependency_count"] = len(main_dependencies)
    routing["full_input_count"] = len(paths)
    return sorted(path.resolve() for path in paths), routing


def attempt_key(head: str, inputs_sha256: str) -> dict[str, str]:
    return {
        "bank_sha256": EXPECTED_BANK_SHA256,
        "base_recorder_sha256": EXPECTED_BASE_RECORDER_SHA256,
        "candidate_bot_sha256": EXPECTED_CANDIDATE_BOT_SHA256,
        "candidate_private_header_sha256": EXPECTED_PRIVATE_HEADER_SHA256,
        "candidate_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "candidate_sources_manifest_sha256": EXPECTED_SOURCES_SHA256,
        "head": head,
        "focused_recorder_test_sha256": EXPECTED_FOCUSED_TEST_SHA256,
        "inputs_sha256": inputs_sha256,
        "integration_commit": INTEGRATION_COMMIT,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "common_recorder_sha256": EXPECTED_COMMON_RECORDER_SHA256,
        "prototype_pass_sha256": EXPECTED_PROTOTYPE_PASS_SHA256,
        "prototype_patch_sha256": EXPECTED_PROTOTYPE_PATCH_SHA256,
        "prototype_timing_sha256": EXPECTED_PROTOTYPE_TIMING_SHA256,
        "reference_bot_sha256": EXPECTED_CONTROL_BOT_SHA256,
        "reference_source_sha256": EXPECTED_CONTROL_SOURCE_SHA256,
        "reference_wrapper_sha256": EXPECTED_REFERENCE_WRAPPER_SHA256,
        "rollback_sha256": EXPECTED_ROLLBACK_SHA256,
        "schema": SCHEMA,
        "shared_header_sha256": EXPECTED_SHARED_HEADER_SHA256,
        "sole_edge_decision_sha256": EXPECTED_SOLE_EDGE_DECISION_SHA256,
    }


def attempt_id(head: str, inputs_sha256: str) -> str:
    return sha256_bytes(canonical_json(attempt_key(head, inputs_sha256)))


def matching_attempts(identifier: str, output: Path = OUTPUT) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(output.glob("*.json")):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if (path.stem == sha256_bytes(raw) and isinstance(payload, dict) and
                canonical_json(payload) == raw and payload.get("schema") == SCHEMA and
                payload.get("attempt_id") == identifier):
            matches.append(path)
    return matches


def claim_path(identifier: str, claims: Path = CLAIMS) -> Path:
    if (len(identifier) != 64 or
            any(character not in "0123456789abcdef" for character in identifier)):
        raise ValueError("attempt identifier is not a lowercase SHA-256")
    return claims / f"{identifier}.claim.json"


def validate_attempt_claim(path: Path, identifier: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    expected_fields = {
        "schema", "attempt_id", "attempt_key", "head", "inputs_sha256",
        "plan_sha256", "claimed_utc", "prelaunch_checks_complete",
        "retry_policy",
    }
    if (not isinstance(payload, dict) or set(payload) != expected_fields or
            canonical_json(payload) != raw):
        raise ValueError("attempt claim is not the exact canonical schema")
    key = payload.get("attempt_key")
    if (payload.get("schema") != CLAIM_SCHEMA or
            payload.get("attempt_id") != identifier or
            not isinstance(key, dict) or
            sha256_bytes(canonical_json(key)) != identifier or
            payload.get("head") != key.get("head") or
            payload.get("inputs_sha256") != key.get("inputs_sha256") or
            payload.get("plan_sha256") != EXPECTED_PLAN_SHA256 or
            key.get("plan_sha256") != EXPECTED_PLAN_SHA256 or
            not isinstance(payload.get("claimed_utc"), str) or
            not payload["claimed_utc"] or
            payload.get("prelaunch_checks_complete") is not True or
            payload.get("retry_policy") !=
            "claim-retained-postclaim-failure-consumes-the-one-attempt"):
        raise ValueError("attempt claim binding mismatch")
    return payload


def matching_attempt_evidence(
    identifier: str, output: Path = OUTPUT,
) -> list[dict[str, Any]]:
    evidence = [
        {"kind": "report", "path": path}
        for path in matching_attempts(identifier, output)
    ]
    path = claim_path(identifier, output / "claims")
    if path.exists():
        validate_attempt_claim(path, identifier)
        evidence.append({"kind": "claim", "path": path.resolve()})
    return evidence


def create_attempt_claim(
    identifier: str, key: dict[str, str], claimed_utc: str,
    claims: Path = CLAIMS,
) -> dict[str, Any]:
    if sha256_bytes(canonical_json(key)) != identifier:
        raise ValueError("attempt claim key does not match attempt identifier")
    claims.mkdir(parents=True, exist_ok=True)
    path = claim_path(identifier, claims)
    payload = {
        "schema": CLAIM_SCHEMA,
        "attempt_id": identifier,
        "attempt_key": key,
        "head": key["head"],
        "inputs_sha256": key["inputs_sha256"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "claimed_utc": claimed_utc,
        "prelaunch_checks_complete": True,
        "retry_policy": (
            "claim-retained-postclaim-failure-consumes-the-one-attempt"
        ),
    }
    raw = canonical_json(payload)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Deliberately retain even a partial claim.  Once O_EXCL succeeds the
        # exact identity is consumed; a later run fails closed on validation.
        raise
    persisted = path.read_bytes()
    validate_attempt_claim(path, identifier)
    digest = sha256_bytes(persisted)
    if persisted != raw or digest != sha256_bytes(raw):
        raise OSError("attempt claim reread/rehash mismatch")
    return {
        "path": identity_label(path),
        "bytes": len(persisted),
        "sha256": digest,
        "claimed_utc": claimed_utc,
        "retained": True,
    }


def _build_gate() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmake", "--build", "build", "--parallel", "2", "--target",
         GATE_TARGET, "--clean-first"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    argparse.ArgumentParser().parse_args()
    try:
        validate_imported_recorder_files()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock_handle, \
            BENCHMARK_LOCK.open("a+b") as benchmark_lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another PositionKey-cache recorder owns the lock", file=sys.stderr)
            return 2
        try:
            fcntl.flock(
                benchmark_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            print("the shared hybrid benchmark/clock lock is busy", file=sys.stderr)
            return 2

        tracked_before = base.common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_before = base.common.git_text("status", "--porcelain")
        head_before = base.common.git_text("rev-parse", "HEAD")
        if tracked_before:
            print("tracked or staged files differ from HEAD", file=sys.stderr)
            return 2
        try:
            base.common.require_repository_inputs_tracked(STATIC_INPUTS)
            lineage = require_committed_bindings(head_before)
            prebuild_processes = base.require_clean_process_preflight()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(str(error), file=sys.stderr)
            return 2

        build = _build_gate()
        if build.returncode != 0 or build.stderr:
            print("fresh PositionKey-cache comparison gate build failed", file=sys.stderr)
            if build.stderr:
                print(build.stderr, file=sys.stderr, end="")
            return 2
        generated_check = subprocess.run(
            ["node", "submissions/codingame/tools/generate_submission.mjs",
             "rank_4_jacek_hybrid", "--check"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        if generated_check.returncode != 0 or generated_check.stderr:
            print("generated source current-check failed", file=sys.stderr)
            return 2

        try:
            full_paths, dependency_routing = collect_full_input_paths()
            inputs_before = identities_for_paths(full_paths)
            validate_exact_file_identities(inputs_before)
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        inputs_sha256 = sha256_bytes(canonical_json(inputs_before))
        identifier = attempt_id(head_before, inputs_sha256)
        try:
            prerun_processes = base.require_clean_process_preflight()
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        try:
            prior_evidence = matching_attempt_evidence(identifier)
            if prior_evidence:
                print(
                    "an attempt claim or report already exists for this exact "
                    "committed HEAD and complete input identity", file=sys.stderr,
                )
                return 2
            claimed_utc = base.utc_now()
            claim = create_attempt_claim(
                identifier,
                attempt_key(head_before, inputs_sha256),
                claimed_utc,
            )
        except FileExistsError:
            print(
                "the atomic attempt claim already exists; retry is forbidden",
                file=sys.stderr,
            )
            return 2
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(
                "attempt claim creation/validation failed; if O_EXCL created "
                f"the file, the attempt remains consumed: {error}",
                file=sys.stderr,
            )
            return 2

        command = command_for_gate()
        started = base.utc_now()
        monotonic_started = time.monotonic_ns()
        timed_out = False
        os_error: str | None = None
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
                timeout=RUN_TIMEOUT_SECONDS,
            )
            returncode: int | None = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out, returncode = True, None
            stdout, stderr = error.stdout or "", error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
        except OSError as error:
            os_error, returncode, stdout, stderr = (
                f"{type(error).__name__}: {error}", None, "", ""
            )
        elapsed_ns = time.monotonic_ns() - monotonic_started
        ended = base.utc_now()

        postrun_errors: list[str] = []
        try:
            inputs_after = identities_for_paths(full_paths)
        except OSError as error:
            inputs_after = {}
            postrun_errors.append(f"post-run input hashing failed: {error}")
        tracked_after = base.common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_after = base.common.git_text("status", "--porcelain")
        head_after = base.common.git_text("rev-parse", "HEAD")

        validation_errors: list[str] = list(postrun_errors)
        threshold_errors: list[str] = []
        parsed: dict[str, Any] = {}
        try:
            parsed = validate_gate_stdout(stdout)
            threshold_errors = selection_errors(parsed["aggregate"])
        except (ValueError, OverflowError) as error:
            validation_errors.append(str(error))
        stable_inputs = inputs_before == inputs_after
        acceptable = (
            returncode == 0 and not timed_out and os_error is None and
            stderr == "" and stable_inputs and head_before == head_after and
            tracked_before == "" and tracked_after == "" and
            not validation_errors and not threshold_errors
        )
        report = {
            "schema": SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "campaign_t0_utc": CAMPAIGN_T0_UTC,
            "classification": CLASSIFICATION,
            "final_qualification": False,
            "attempt_id": identifier,
            "attempt_key": attempt_key(head_before, inputs_sha256),
            "attempt_claim": claim,
            "frozen_plan": {
                "path": str(PLAN.relative_to(ROOT)),
                "sha256": EXPECTED_PLAN_SHA256,
            },
            "lineage": lineage,
            "reference_slot": {
                "echo": "rank4",
                "semantics": "archived-prefastpath-hybrid",
                "function": "choose_prefastpath",
            },
            "candidate_exact_proof_mask": 7,
            "reference_exact_proof_mask": 7,
            "dependency_routing": dependency_routing,
            "process_preflight": {
                "before_fresh_build": prebuild_processes,
                "before_gate_run": prerun_processes,
            },
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_monotonic_ns": elapsed_ns,
            "command_argv": command,
            "command_shell": shlex.join(command),
            "cwd": str(ROOT),
            "returncode": returncode,
            "timed_out": timed_out,
            "os_error": os_error,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "git": {
                "head_before": head_before,
                "head_after": head_after,
                "tracked_status_before": tracked_before,
                "tracked_status_after": tracked_after,
                "full_status_before": full_status_before,
                "full_status_after": full_status_after,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "fresh_gate_build": {
                "argv": ["cmake", "--build", "build", "--parallel", "2",
                         "--target", GATE_TARGET, "--clean-first"],
                "returncode": build.returncode,
                "stdout": build.stdout,
                "stderr": build.stderr,
            },
            "generated_source_check": {
                "returncode": generated_check.returncode,
                "stdout": generated_check.stdout,
                "stderr": generated_check.stderr,
            },
            "full_input_count": len(full_paths),
            "inputs_sha256": inputs_sha256,
            "inputs_before": inputs_before,
            "inputs_after": inputs_after,
            "stable_inputs": stable_inputs,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed,
            "validation_errors": validation_errors,
            "threshold_errors": threshold_errors,
            "development_ablation_acceptable": acceptable,
        }
        final_path, digest = persist_content_addressed_report(
            OUTPUT, report, os.getpid()
        )
        persisted = final_path.read_bytes()
        if (final_path.stem != digest or sha256_bytes(persisted) != digest or
                canonical_json(json.loads(persisted)) != persisted):
            print("content-addressed report readback failed", file=sys.stderr)
            return 2
        print(final_path.relative_to(ROOT))
        print(f"sha256={digest}")
        print(f"development_ablation_acceptable={str(acceptable).lower()}")
        return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
