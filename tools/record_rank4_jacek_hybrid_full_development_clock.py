#!/usr/bin/env python3
"""Record one exact 306-game DEVELOPMENT clock gate for proof mask 7 or 3.

For each preregistered candidate mask, only two comparisons are accepted: the
same hybrid with proofs off, then the frozen Rank-4 engine.  The validation and
final opening banks are not addressable by this program.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import record_rank4_jacek_hybrid_proof_scope_clock as common


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/full_development_clock"
LOCK = ROOT / "build/rank4-jacek-hybrid-full-development-clock.lock"
RUN_TIMEOUT_SECONDS = 3_600
GATE = common.GATE
GATE_TARGET = common.GATE_TARGET
CAMPAIGN_ID = common.CAMPAIGN_ID
CAMPAIGN_T0_UTC = common.CAMPAIGN_T0_UTC
EXPECTED_SOURCE_SHA256 = common.EXPECTED_SOURCE_SHA256
EXPECTED_SOURCE_BYTES = common.EXPECTED_SOURCE_BYTES
PLAN = ROOT / "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md"
EXPECTED_PLAN_SHA256 = (
    "50acd3d31df69579e0d6c3d68a71f20c4964f2413523d754be798f607d558438"
)
RANK4_SOURCE = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
EXPECTED_RANK4_SOURCE_SHA256 = (
    "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
)
EXPECTED_RANK4_SOURCE_BYTES = 98_624

BANKS = (
    ("development_d04.tsv", 4, 18128950407139886133,
     "984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc", 78),
    ("development_d08.tsv", 8, 9297997631523997120,
     "6dbec157e7094f07796a9aa1ac97b43919930377ec315c761eaace216630259e", 76),
    ("development_d12.tsv", 12, 11025886481058993262,
     "d30d087020e4946ce77b6d6e578484d583f0cf25f2ffa90a1918f8d9a9a8a11a", 76),
    ("development_d20.tsv", 20, 4624785204876369057,
     "2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4", 76),
)
BANK_DIRECTORY = ROOT / "results/rank_4_jacek_hybrid/openings"
BANK_PATHS = tuple(BANK_DIRECTORY / item[0] for item in BANKS)
ALLOWED_REFERENCES = ("hybrid-control", "rank4")
ALLOWED_CANDIDATE_MASKS = (7, 3)
TRACKED_INPUTS = tuple(dict.fromkeys(
    (RECORDER, *common.TRACKED_INPUTS, PLAN, RANK4_SOURCE, *BANK_PATHS)
))

# These identities freeze every C++ source/configuration input on the Rank-4
# and comparison-gate route.  RECORDER cannot bind its own content hash, but it
# must be tracked at HEAD and is included in each report's before/after map.
EXPECTED_DEPENDENCY_SHA256 = {
    "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md":
        EXPECTED_PLAN_SHA256,
    "CMakeLists.txt":
        "e67bf70616de76582088b3dc7336a75e0f911feb0d1d6a420cd2dd24e7108be3",
    "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp":
        EXPECTED_SOURCE_SHA256,
    "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp":
        "c8412600f0b90610660f24a02828e77f67cc7d78bdf775da11628b3355274215",
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp":
        "1a23a0425628e4bc903f34f1923f8e5393ced9326a31e210b202d22715a52a2c",
    "submissions/codingame/bots/rank_4_jacek_hybrid/replay_value_model.hpp":
        "36ac226b60f0cc5a5fac9f2dc35017320ebd00c304847f83a8d07ef1eef7f2bd",
    "submissions/codingame/bots/rank_4_jacek_hybrid/teacher_residual_model.hpp":
        "e2034019126d54c2e0fd33922d7304c09979a64007a8d55e20823eda46834e42",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp":
        "d872e1720511d3045ac6890d566d795323e056db7ddf74cf365ce2f436f45b80",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_engine.hpp":
        "be8dc5f1d05c841b06598217acf01737a220af8a1ec598765018d82a3f4c1166",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_hybrid.cpp":
        "d29c56e388215d91e0757da01802fd892dfade76cbb4042eecec56f3a33b3738",
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_rank4.cpp":
        "7cc8c603811f7eb1d1b43e1a3c8fe439b86c75e6bf9ee0831c41eb5eda4f0797",
    "submissions/codingame/bots/rank_4/submission.cpp":
        EXPECTED_RANK4_SOURCE_SHA256,
    "submissions/codingame/bots/rank_4/bot.cpp":
        "9276c258cd613b6b78948aeb8aa2649851d226947d419f84a351968a9035c0ad",
    "submissions/codingame/bots/rank_4/replay_book.hpp":
        "1a23a0425628e4bc903f34f1923f8e5393ced9326a31e210b202d22715a52a2c",
    "submissions/codingame/bots/rank_4/replay_value_model.hpp":
        "36ac226b60f0cc5a5fac9f2dc35017320ebd00c304847f83a8d07ef1eef7f2bd",
    "submissions/codingame/bots/rank_4/teacher_residual_model.hpp":
        "e2034019126d54c2e0fd33922d7304c09979a64007a8d55e20823eda46834e42",
    "src/bots/mcts_internal.hpp":
        "0a13e89e183666ce89e38d1eded1b26c02eaaba5460ba7e3ede9fda5d5e1dd04",
    "src/opening_bank/opening_bank.cpp":
        "b26ea1650807b6e3f3911e0344ab3fd4311ae224859d5d8c8e9125b8eb9d339d",
    "src/opening_bank/opening_bank_internal.hpp":
        "530580456dd7d255b78c9e1aab44a8a8a13743886ef6d22b8b34e79052eb6049",
    "src/core/rules.cpp":
        "8e9fdfd1ce105a2b5e6d2fb6538ba77606631bfbdc00ae561988299344891a98",
    "src/core/geometry.cpp":
        "31dfcd2ec1e7ffbb01a5cb9ddea6402117e0766fd33dd2a118edf6aea9c9430a",
    "include/papersoccer/rules.hpp":
        "68db84170931097b3a6489195c4ae5c4d91a0a6de162d9ac801600ef92dae53d",
    "include/papersoccer/geometry.hpp":
        "fc5bc9ef3055cee3236276b306a8244b39e6008022e8db3d8f562f94e53f1124",
    "include/papersoccer/types.hpp":
        "d1592af6720db1dbc4b9f40900c35617d734f372cf2bb813f0d02b8699727e06",
    "build/CMakeCache.txt":
        "7f67a65f5ed0e0cebcd0e0d3c76ec968ca4897221b1fd1fdf16b591281a043b4",
    ("build/CMakeFiles/"
     "papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate.dir/flags.make"):
        "c819bc8f89852714070d8816d94f36e98831a287f07e89847ea894ea5f8f405d",
    ("build/CMakeFiles/"
     "papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate.dir/link.txt"):
        "27a91eb0032e0866a5d3f3c0a13d4523555785aec7b5e8191d26bb3f157f60e7",
    "build/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate":
        "619aa02ac67628345cec7cc7f4f0d21c7d4d49fc4f0fa1b77a2d4a3f7bd846b4",
}

ENGINE_ADDITIVE_FIELDS = (
    "invocations", "searches", "illegal", "operational", "exceptions",
    "hard_timeouts", "soft_overruns", "nodes", "exhaustions",
)
ENGINE_INTEGER_MAX_FIELDS = ("nodes_max", "depth_max", "attempted_depth_max")
TIMING_PHASES = ("first", "later")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def configuration_expected(
    reference_engine: str, candidate_mask: int = 7
) -> dict[str, str]:
    return {
        "profile": "clock",
        "reference_engine": reference_engine,
        "bank_count": "4",
        "expected_role": "development",
        "bank_validation":
            "schema,header,role,depth,seed,replay,state-sha256,canonical-sha256,disjoint",
        "max_turns": "320",
        "expected_depths": ",".join(str(item[1]) for item in BANKS),
        "expected_seeds": ",".join(str(item[2]) for item in BANKS),
        "expected_sha256": ",".join(item[3] for item in BANKS),
        "candidate_nodes": "3000000",
        "reference_nodes": "3000000",
        "candidate_clock": "800/165",
        "reference_clock": "800/165",
        "operational_clock": "1000/200",
        "candidate_exact_proof_mask": str(candidate_mask),
        "reference_exact_proof_mask": "0",
        "openings": "preregistered-public-rules",
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }


def command_for(reference_engine: str, candidate_mask: int = 7) -> list[str]:
    command = [str(GATE), "--profile", "clock", "--reference-engine",
               reference_engine]
    for path in BANK_PATHS:
        command.extend(("--bank", str(path)))
    command.extend((
        "--expected-role", "development",
        "--expected-depths", ",".join(str(item[1]) for item in BANKS),
        "--expected-seeds", ",".join(str(item[2]) for item in BANKS),
        "--expected-sha256", ",".join(item[3] for item in BANKS),
        "--max-turns", "320",
        "--candidate-nodes", "3000000",
        "--reference-nodes", "3000000",
        "--candidate-first-ms", "800",
        "--candidate-later-ms", "165",
        "--reference-first-ms", "800",
        "--reference-later-ms", "165",
        "--operational-first-ms", "1000",
        "--operational-later-ms", "200",
        "--candidate-exact-proof-mask", str(candidate_mask),
        "--reference-exact-proof-mask", "0",
    ))
    return command


def sum_bank_accounting(banks: list[dict[str, str]]) -> dict[str, int]:
    keys = ("games", "candidate_wins", "reference_wins", "unfinished", "failed")
    return {key: sum(common.exact_int(bank, key) for bank in banks) for key in keys}


def validate_frozen_identities(identities: dict[str, dict[str, Any]]) -> None:
    for path, expected_sha256 in EXPECTED_DEPENDENCY_SHA256.items():
        identity = identities.get(path)
        if identity is None:
            raise ValueError(f"missing frozen dependency identity: {path}")
        if identity.get("sha256") != expected_sha256:
            raise ValueError(f"frozen dependency SHA-256 mismatch: {path}")
    rank4 = identities[str(RANK4_SOURCE.relative_to(ROOT))]
    if (rank4.get("bytes") != EXPECTED_RANK4_SOURCE_BYTES or
            rank4.get("ascii") is not True):
        raise ValueError("exact frozen Rank-4 source identity mismatch")


def finite_nonnegative(fields: dict[str, str], key: str) -> float:
    value = common.exact_float(fields, key)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"field is not finite and nonnegative: {key}")
    return value


def validate_timing(fields: dict[str, str]) -> None:
    for engine in ("candidate", "reference"):
        for phase in TIMING_PHASES:
            p99 = finite_nonnegative(fields, f"{engine}_{phase}_ms_p99")
            maximum = finite_nonnegative(fields, f"{engine}_{phase}_ms_max")
            if p99 > maximum:
                raise ValueError(
                    f"{engine} {phase} timing p99 exceeds maximum"
                )


def validate_rebound_identity(fields: dict[str, str]) -> None:
    for engine in ("candidate", "reference"):
        rebound = common.parse_proof(fields, engine, "rebound")
        scope_sum = [0, 0, 0]
        for scope in ("root", "leaf", "ply1", "ply2"):
            counters = common.parse_proof(fields, engine, scope)
            for index in range(3):
                scope_sum[index] += counters[index]
        if rebound != tuple(scope_sum):
            raise ValueError(f"{engine} rebound/scope proof counters mismatch")


def validate_bank_aggregate_consistency(
    banks: list[dict[str, str]], aggregate: dict[str, str]
) -> None:
    summed = sum_bank_accounting(banks)
    for key, value in summed.items():
        if common.exact_int(aggregate, key) != value:
            raise ValueError(f"bank/aggregate accounting mismatch: {key}")

    for color in range(2):
        key = f"candidate_p{color}"
        aggregate_color = common.parse_color(aggregate, key)
        bank_colors = [common.parse_color(bank, key) for bank in banks]
        summed_color = tuple(
            sum(value[index] for value in bank_colors)
            for index in range(len(aggregate_color))
        )
        if aggregate_color != summed_color:
            raise ValueError(f"bank/aggregate color mismatch: {key}")

    for engine in ("candidate", "reference"):
        for suffix in ENGINE_ADDITIVE_FIELDS:
            key = f"{engine}_{suffix}"
            expected = sum(common.exact_int(bank, key) for bank in banks)
            if common.exact_int(aggregate, key) != expected:
                raise ValueError(f"bank/aggregate engine counter mismatch: {key}")
        for suffix in ENGINE_INTEGER_MAX_FIELDS:
            key = f"{engine}_{suffix}"
            expected = max(common.exact_int(bank, key) for bank in banks)
            if common.exact_int(aggregate, key) != expected:
                raise ValueError(f"bank/aggregate engine maximum mismatch: {key}")
        for phase in TIMING_PHASES:
            key = f"{engine}_{phase}_ms_max"
            expected = max(finite_nonnegative(bank, key) for bank in banks)
            actual = finite_nonnegative(aggregate, key)
            if actual != expected:
                raise ValueError(f"bank/aggregate timing maximum mismatch: {key}")
        for scope in ("rebound", "root", "leaf", "ply1", "ply2"):
            aggregate_counter = common.parse_proof(aggregate, engine, scope)
            bank_counters = [
                common.parse_proof(bank, engine, scope) for bank in banks
            ]
            summed_counter = tuple(
                sum(counter[index] for counter in bank_counters)
                for index in range(len(aggregate_counter))
            )
            if aggregate_counter != summed_counter:
                raise ValueError(
                    f"bank/aggregate proof mismatch: {engine}/{scope}"
                )


def validate_full_summaries(
    banks: list[dict[str, str]], aggregate: dict[str, str],
    candidate_mask: int = 7,
) -> None:
    if len(banks) != len(BANKS):
        raise ValueError("wrong number of DEVELOPMENT bank summaries")
    for index, (fields, bank) in enumerate(zip(banks, BANKS)):
        common.validate_summary(
            fields, str(index), candidate_mask, 0,
            expected_games=bank[4], expected_color_games=bank[4] // 2,
        )
        validate_timing(fields)
        validate_rebound_identity(fields)
    common.validate_summary(
        aggregate, "all", candidate_mask, 0,
        expected_games=306, expected_color_games=153,
    )
    validate_timing(aggregate)
    validate_rebound_identity(aggregate)
    validate_bank_aggregate_consistency(banks, aggregate)


def selection_threshold_errors(aggregate: dict[str, str]) -> list[str]:
    errors: list[str] = []
    candidate_wins = common.exact_int(aggregate, "candidate_wins")
    if candidate_wins < 160:
        errors.append("candidate has fewer than 160 full-development wins")
    for color in range(2):
        wins = common.parse_color(aggregate, f"candidate_p{color}")[0]
        if wins < 77:
            errors.append(
                f"candidate has fewer than 77 wins as physical color {color}"
            )
    return errors


def validate_control_prerequisite(
    report: dict[str, Any], current_head: str,
    current_inputs: dict[str, dict[str, Any]], candidate_mask: int = 7,
) -> None:
    if report.get("schema") != "rank4-jacek-hybrid-full-development-clock-v2":
        raise ValueError("control report schema mismatch")
    if (report.get("campaign_id") != CAMPAIGN_ID or
            report.get("campaign_t0_utc") != CAMPAIGN_T0_UTC):
        raise ValueError("control report campaign mismatch")
    if (report.get("reference_engine") != "hybrid-control" or
            report.get("candidate_exact_proof_mask") != candidate_mask or
            report.get("reference_exact_proof_mask") != 0):
        raise ValueError("control report engine/mask mismatch")
    if (report.get("classification") !=
            "full-development-selection-gate-not-final-qualification" or
            report.get("final_qualification") is not False or
            report.get("stderr") != ""):
        raise ValueError("control report classification/stderr mismatch")
    if (report.get("returncode") != 0 or report.get("timed_out") is not False or
            report.get("os_error") is not None or
            report.get("stable_inputs") is not True or
            report.get("development_selection_acceptable") is not True):
        raise ValueError("control report is not accepted")
    git = report.get("git")
    if not isinstance(git, dict) or (
            git.get("head_before") != current_head or
            git.get("head_after") != current_head or
            git.get("tracked_status_before") != "" or
            git.get("tracked_status_after") != ""):
        raise ValueError("control report HEAD/status mismatch")
    if (report.get("inputs_before") != current_inputs or
            report.get("inputs_after") != current_inputs):
        raise ValueError("control report input identities mismatch")
    parsed = report.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError("control report parsed payload is missing")
    banks = parsed.get("banks")
    aggregate = parsed.get("aggregate")
    configuration = parsed.get("configuration")
    if not isinstance(banks, list) or not all(
            isinstance(bank, dict) for bank in banks
    ) or not isinstance(aggregate, dict):
        raise ValueError("control report summaries are malformed")
    if configuration != configuration_expected("hybrid-control", candidate_mask):
        raise ValueError("control report configuration mismatch")
    validate_full_summaries(banks, aggregate, candidate_mask)
    if selection_threshold_errors(aggregate):
        raise ValueError("control report fails frozen selection thresholds")


def matching_attempt_reports(
    reference_engine: str, current_head: str,
    current_inputs: dict[str, dict[str, Any]], candidate_mask: int = 7,
    output: Path = OUTPUT,
) -> list[tuple[Path, str, dict[str, Any]]]:
    matches: list[tuple[Path, str, dict[str, Any]]] = []
    for path in sorted(output.glob("*.json")):
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if path.stem != digest:
                continue
            report = json.loads(raw)
            if not isinstance(report, dict) or canonical_json(report) != raw:
                continue
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        schema = report.get("schema")
        git = report.get("git")
        if (not isinstance(schema, str) or
                not schema.startswith(
                    "rank4-jacek-hybrid-full-development-clock-v"
                ) or
                report.get("campaign_id") != CAMPAIGN_ID or
                report.get("reference_engine") != reference_engine or
                report.get("candidate_exact_proof_mask") != candidate_mask or
                report.get("reference_exact_proof_mask") != 0 or
                not isinstance(git, dict) or
                git.get("head_before") != current_head or
                report.get("inputs_before") != current_inputs):
            continue
        matches.append((path, digest, report))
    return matches


def find_accepted_control_report(
    current_head: str, current_inputs: dict[str, dict[str, Any]],
    output: Path = OUTPUT, candidate_mask: int = 7,
) -> dict[str, str]:
    attempts = matching_attempt_reports(
        "hybrid-control", current_head, current_inputs, candidate_mask, output
    )
    if len(attempts) != 1:
        raise ValueError(
            "expected exactly one same-binary control attempt; "
            f"found {len(attempts)}"
        )
    path, digest, report = attempts[0]
    try:
        validate_control_prerequisite(
            report, current_head, current_inputs, candidate_mask
        )
    except ValueError as error:
        raise ValueError(f"same-binary control attempt is not accepted: {error}") \
            from error
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "ended_utc": str(report.get("ended_utc", "")),
    }


def require_no_previous_attempt(
    reference_engine: str, current_head: str,
    current_inputs: dict[str, dict[str, Any]], candidate_mask: int = 7,
    output: Path = OUTPUT,
) -> None:
    attempts = matching_attempt_reports(
        reference_engine, current_head, current_inputs, candidate_mask, output
    )
    if attempts:
        raise ValueError(
            "a full-development attempt already exists for this exact "
            "HEAD/input/mask/reference identity; retries are forbidden"
        )


def require_mask3_fallback_authorized(
    current_head: str, current_inputs: dict[str, dict[str, Any]],
    output: Path = OUTPUT,
) -> dict[str, str]:
    controls = matching_attempt_reports(
        "hybrid-control", current_head, current_inputs, 7, output
    )
    rank4_attempts = matching_attempt_reports(
        "rank4", current_head, current_inputs, 7, output
    )
    if len(controls) != 1:
        raise ValueError(
            "mask-3 fallback requires exactly one prior mask-7 control attempt"
        )
    control_path, control_digest, control = controls[0]
    if control.get("development_selection_acceptable") is not True:
        if rank4_attempts:
            raise ValueError(
                "mask-7 Rank-4 attempt exists after a failed mask-7 control"
            )
        return {
            "failed_reference_engine": "hybrid-control",
            "path": str(control_path.relative_to(ROOT)),
            "sha256": control_digest,
        }
    if len(rank4_attempts) != 1:
        raise ValueError(
            "accepted mask-7 control requires exactly one mask-7 Rank-4 "
            "attempt before mask-3 fallback"
        )
    rank4_path, rank4_digest, rank4_report = rank4_attempts[0]
    if rank4_report.get("development_selection_acceptable") is True:
        raise ValueError("mask-3 fallback is forbidden after mask 7 passed both gates")
    return {
        "failed_reference_engine": "rank4",
        "path": str(rank4_path.relative_to(ROOT)),
        "sha256": rank4_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-engine", choices=ALLOWED_REFERENCES,
                        required=True)
    parser.add_argument("--candidate-mask", choices=ALLOWED_CANDIDATE_MASKS,
                        type=int, default=7)
    args = parser.parse_args()
    reference_engine: str = args.reference_engine
    candidate_mask: int = args.candidate_mask

    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another full-development recorder owns the lock", file=sys.stderr)
            return 2

        tracked_status_before = common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_before = common.git_text("status", "--porcelain")
        head_before = common.git_text("rev-parse", "HEAD")
        if tracked_status_before:
            print("tracked or staged files differ from HEAD", file=sys.stderr)
            return 2
        try:
            common.require_repository_inputs_tracked(TRACKED_INPUTS)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        build = subprocess.run(
            ["cmake", "--build", "build", "--parallel", "2", "--target",
             GATE_TARGET], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if build.returncode != 0 or build.stderr:
            print("fresh comparison-gate build failed", file=sys.stderr)
            if build.stderr:
                print(build.stderr, file=sys.stderr, end="")
            return 2

        missing = [str(path) for path in TRACKED_INPUTS if not path.is_file()]
        if missing:
            print("missing inputs: " + ", ".join(missing), file=sys.stderr)
            return 2
        before = {
            str(path.relative_to(ROOT)): common.file_identity(path)
            for path in TRACKED_INPUTS
        }
        source = before[
            "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp"
        ]
        if (source["sha256"] != EXPECTED_SOURCE_SHA256 or
                source["bytes"] != EXPECTED_SOURCE_BYTES or
                not source["ascii"] or source["bytes"] > 99_999):
            print("exact generated source identity mismatch", file=sys.stderr)
            return 2
        try:
            validate_frozen_identities(before)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        for path, bank in zip(BANK_PATHS, BANKS):
            if before[str(path.relative_to(ROOT))]["sha256"] != bank[3]:
                print("frozen DEVELOPMENT bank SHA-256 mismatch", file=sys.stderr)
                return 2

        generated_check = subprocess.run(
            ["node", "submissions/codingame/tools/generate_submission.mjs",
             "rank_4_jacek_hybrid", "--check"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if generated_check.returncode != 0 or generated_check.stderr:
            print("generated source current-check failed", file=sys.stderr)
            return 2

        try:
            require_no_previous_attempt(
                reference_engine, head_before, before, candidate_mask
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        fallback_trigger: dict[str, str] | None = None
        if candidate_mask == 3:
            try:
                fallback_trigger = require_mask3_fallback_authorized(
                    head_before, before
                )
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2

        control_prerequisite: dict[str, str] | None = None
        if reference_engine == "rank4":
            try:
                control_prerequisite = find_accepted_control_report(
                    head_before, before, candidate_mask=candidate_mask
                )
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2

        command = command_for(reference_engine, candidate_mask)
        started = utc_now()
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
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = None
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
        except OSError as error:
            os_error = f"{type(error).__name__}: {error}"
            returncode = None
            stdout = ""
            stderr = ""
        monotonic_ended = time.monotonic_ns()
        ended = utc_now()

        after = {
            str(path.relative_to(ROOT)): common.file_identity(path)
            for path in TRACKED_INPUTS
        }
        stable_inputs = before == after
        tracked_status_after = common.git_text(
            "status", "--porcelain", "--untracked-files=no"
        )
        full_status_after = common.git_text("status", "--porcelain")
        head_after = common.git_text("rev-parse", "HEAD")

        lines = [line for line in stdout.splitlines() if line.strip()]
        bank_lines = [line for line in lines if line.startswith("bank_summary ")]
        summary_lines = [line for line in lines if line.startswith("summary ")]
        configuration_lines = [
            line for line in lines if line.startswith("configuration ")
        ]
        validation_errors: list[str] = []
        selection_errors: list[str] = []
        bank_fields: list[dict[str, str]] = []
        aggregate: dict[str, str] = {}
        configuration: dict[str, str] = {}
        try:
            if (len(lines) != 6 or len(bank_lines) != 4 or
                    len(summary_lines) != 1 or len(configuration_lines) != 1):
                raise ValueError("gate stdout is not exactly six expected lines")
            bank_fields = [common.parse_fields(line) for line in bank_lines]
            aggregate = common.parse_fields(summary_lines[0])
            configuration = common.parse_fields(configuration_lines[0])
            if configuration != configuration_expected(
                    reference_engine, candidate_mask):
                raise ValueError("complete configuration echo mismatch")
            validate_full_summaries(bank_fields, aggregate, candidate_mask)
            selection_errors = selection_threshold_errors(aggregate)
        except (ValueError, OverflowError) as error:
            validation_errors.append(str(error))

        acceptable = (
            returncode == 0 and not timed_out and os_error is None and
            stderr == "" and stable_inputs and head_before == head_after and
            tracked_status_before == "" and tracked_status_after == "" and
            not validation_errors and not selection_errors
        )
        report = {
            "schema": "rank4-jacek-hybrid-full-development-clock-v2",
            "campaign_id": CAMPAIGN_ID,
            "campaign_t0_utc": CAMPAIGN_T0_UTC,
            "classification":
                "full-development-selection-gate-not-final-qualification",
            "reference_engine": reference_engine,
            "candidate_exact_proof_mask": candidate_mask,
            "reference_exact_proof_mask": 0,
            "frozen_plan": {
                "path": str(PLAN.relative_to(ROOT)),
                "sha256": EXPECTED_PLAN_SHA256,
            },
            "frozen_dependency_sha256": EXPECTED_DEPENDENCY_SHA256,
            "mask3_fallback_trigger": fallback_trigger,
            "accepted_control_prerequisite": control_prerequisite,
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_monotonic_ns": monotonic_ended - monotonic_started,
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
                "tracked_status_before": tracked_status_before,
                "tracked_status_after": tracked_status_after,
                "full_status_before": full_status_before,
                "full_status_after": full_status_after,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cmake_version": subprocess.run(
                    ["cmake", "--version"], text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=True,
                ).stdout.splitlines()[0],
                "cxx_version": subprocess.run(
                    ["/usr/bin/c++", "--version"], text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True,
                ).stdout.splitlines()[0],
            },
            "fresh_gate_build": {
                "target": GATE_TARGET,
                "returncode": build.returncode,
                "stdout": build.stdout,
                "stderr": build.stderr,
            },
            "generated_source_check": {
                "returncode": generated_check.returncode,
                "stdout": generated_check.stdout,
                "stderr": generated_check.stderr,
            },
            "inputs_before": before,
            "inputs_after": after,
            "stable_inputs": stable_inputs,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": {
                "bank_lines": bank_lines,
                "banks": bank_fields,
                "aggregate_line": summary_lines[0] if len(summary_lines) == 1 else None,
                "aggregate": aggregate,
                "configuration_line": (
                    configuration_lines[0]
                    if len(configuration_lines) == 1 else None
                ),
                "configuration": configuration,
                "expected_configuration": configuration_expected(
                    reference_engine, candidate_mask
                ),
                "validation_errors": validation_errors,
                "selection_threshold_errors": selection_errors,
            },
            "development_selection_acceptable": acceptable,
            "final_qualification": False,
        }
        canonical = canonical_json(report)
        digest = hashlib.sha256(canonical).hexdigest()
        destination = OUTPUT / f"{digest}.json"
        temporary = OUTPUT / f".{digest}.{os.getpid()}.tmp"
        temporary.write_bytes(canonical)
        os.replace(temporary, destination)
        persisted = destination.read_bytes()
        if persisted != canonical or hashlib.sha256(persisted).hexdigest() != digest:
            print("persisted report failed byte/hash verification", file=sys.stderr)
            return 2

        print(f"report={destination.relative_to(ROOT)}")
        print(f"report_sha256={digest}")
        if summary_lines:
            print(summary_lines[0])
        print(f"development_selection_acceptable={str(acceptable).lower()}")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if validation_errors:
            print("; ".join(validation_errors), file=sys.stderr)
        if selection_errors:
            print("; ".join(selection_errors), file=sys.stderr)
        return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
