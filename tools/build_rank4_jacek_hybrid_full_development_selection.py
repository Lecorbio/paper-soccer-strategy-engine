#!/usr/bin/env python3
"""Verify and package the frozen full-DEVELOPMENT mask-7 selection evidence."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import record_rank4_jacek_hybrid_full_development_clock as recorder


REPORT_DIRECTORY = ROOT / "results/rank_4_jacek_hybrid/gates/full_development_clock"
OUTPUT_DIRECTORY = REPORT_DIRECTORY / "selection"
PLAN_PATH = ROOT / "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md"
PROOF_MATRIX_PATH = (
    ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock/matrix/"
    "739eaf7d4e2fa9f218e759e309e042271f75a84cf4f5216fbef07eec7a525454.json"
)

CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"
EXPECTED_HEAD = "2a37f20727da0e90e71a5d0b71a44cd0ae87ffbd"
EXPECTED_PLAN_SHA256 = (
    "50acd3d31df69579e0d6c3d68a71f20c4964f2413523d754be798f607d558438"
)
EXPECTED_PROOF_MATRIX_SHA256 = (
    "739eaf7d4e2fa9f218e759e309e042271f75a84cf4f5216fbef07eec7a525454"
)
EXPECTED_REPORTS = (
    (
        "hybrid-control",
        "8f7aa959b54843baad13333e3023d43c852be1e11296bba0e5b3ac8524aa1fa9",
    ),
    (
        "rank4",
        "cd259e7053467a01a87d0b79c88b2fb036eb9273c57d82a0b893df8738b21cf1",
    ),
)
EXPECTED_CRITICAL_INPUTS = {
    "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md": (
        2_980,
        EXPECTED_PLAN_SHA256,
        True,
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp": (
        94_004,
        "6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f",
        True,
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp": (
        62_777,
        "c8412600f0b90610660f24a02828e77f67cc7d78bdf775da11628b3355274215",
        True,
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp": (
        40_029,
        "d872e1720511d3045ac6890d566d795323e056db7ddf74cf365ce2f436f45b80",
        True,
    ),
    "build/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate": (
        321_464,
        "619aa02ac67628345cec7cc7f4f0d21c7d4d49fc4f0fa1b77a2d4a3f7bd846b4",
        False,
    ),
    "submissions/codingame/bots/rank_4/submission.cpp": (
        98_624,
        "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9",
        True,
    ),
    "tools/record_rank4_jacek_hybrid_full_development_clock.py": (
        34_337,
        "58e72685151f86009b5a682c49363d3dc3ae11a151d15b88906130c4505f251e",
        True,
    ),
    "tools/record_rank4_jacek_hybrid_proof_scope_clock.py": (
        23_804,
        "1c9a2a505578b02866aa3c2d64231e048ea46a301d910bb04aef345552bf9aca",
        True,
    ),
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"timestamp is not UTC: {value}")
    return parsed


def load_canonical_json(path: Path, expected_digest: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.stem != expected_digest or sha256(raw) != expected_digest:
        raise ValueError(f"content address mismatch: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"payload is not canonical ASCII JSON: {path}")
    return payload


def load_report(expected_engine: str, digest: str) -> dict[str, Any]:
    report = load_canonical_json(REPORT_DIRECTORY / f"{digest}.json", digest)
    validate_report(report, expected_engine)
    return report


def validate_critical_inputs(inputs: dict[str, Any]) -> None:
    for path, (expected_bytes, expected_sha256, expected_ascii) in \
            EXPECTED_CRITICAL_INPUTS.items():
        identity = inputs.get(path)
        if not isinstance(identity, dict):
            raise ValueError(f"missing critical input identity: {path}")
        if identity != {
            "path": path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "ascii": expected_ascii,
        }:
            raise ValueError(f"critical input identity mismatch: {path}")


def validate_serialized_gate_output(report: dict[str, Any]) -> None:
    parsed = report["parsed"]
    bank_lines = parsed.get("bank_lines")
    aggregate_line = parsed.get("aggregate_line")
    configuration_line = parsed.get("configuration_line")
    if not isinstance(bank_lines, list) or len(bank_lines) != 4 or \
            not all(isinstance(line, str) for line in bank_lines) or \
            not isinstance(aggregate_line, str) or \
            not isinstance(configuration_line, str):
        raise ValueError("serialized full-development gate output is malformed")
    expected_stdout = "\n".join(
        [*bank_lines, aggregate_line, configuration_line]
    ) + "\n"
    if report.get("stdout") != expected_stdout:
        raise ValueError("stdout does not exactly bind parsed gate lines")
    if [recorder.common.parse_fields(line) for line in bank_lines] != \
            parsed.get("banks") or \
            recorder.common.parse_fields(aggregate_line) != parsed.get("aggregate") or \
            recorder.common.parse_fields(configuration_line) != \
            parsed.get("configuration"):
        raise ValueError("parsed gate payload does not match serialized lines")


def validate_report(report: dict[str, Any], expected_engine: str) -> None:
    if report.get("schema") != "rank4-jacek-hybrid-full-development-clock-v2":
        raise ValueError("full-development report schema mismatch")
    if report.get("campaign_id") != CAMPAIGN_ID or \
            report.get("campaign_t0_utc") != CAMPAIGN_T0_UTC:
        raise ValueError("full-development report campaign binding mismatch")
    if report.get("classification") != \
            "full-development-selection-gate-not-final-qualification" or \
            report.get("final_qualification") is not False:
        raise ValueError("full-development report classification mismatch")
    if report.get("reference_engine") != expected_engine or \
            report.get("candidate_exact_proof_mask") != 7 or \
            report.get("reference_exact_proof_mask") != 0:
        raise ValueError("full-development report engine/mask mismatch")
    if report.get("returncode") != 0 or report.get("timed_out") is not False or \
            report.get("os_error") is not None or report.get("stderr") != "" or \
            report.get("stable_inputs") is not True or \
            report.get("development_selection_acceptable") is not True:
        raise ValueError("full-development report process status is not accepted")
    if report.get("mask3_fallback_trigger") is not None:
        raise ValueError("mask-7 report unexpectedly binds a fallback trigger")

    frozen_plan = report.get("frozen_plan")
    expected_plan = {
        "path": str(PLAN_PATH.relative_to(ROOT)),
        "sha256": EXPECTED_PLAN_SHA256,
    }
    if frozen_plan != expected_plan:
        raise ValueError("full-development report frozen-plan binding mismatch")

    git = report.get("git")
    if not isinstance(git, dict) or \
            git.get("head_before") != EXPECTED_HEAD or \
            git.get("head_after") != EXPECTED_HEAD or \
            git.get("tracked_status_before") != "" or \
            git.get("tracked_status_after") != "":
        raise ValueError("full-development report git binding mismatch")

    before = report.get("inputs_before")
    after = report.get("inputs_after")
    if not isinstance(before, dict) or before != after:
        raise ValueError("full-development input identities are unstable")
    validate_critical_inputs(before)
    frozen = report.get("frozen_dependency_sha256")
    if not isinstance(frozen, dict) or frozen != {
            path: before[path]["sha256"] for path in frozen}:
        raise ValueError("frozen dependency identities do not bind report inputs")
    if frozen != recorder.EXPECTED_DEPENDENCY_SHA256:
        raise ValueError("frozen dependency registry mismatch")

    parsed = report.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError("full-development parsed payload is missing")
    if parsed.get("validation_errors") != [] or \
            parsed.get("selection_threshold_errors") != []:
        raise ValueError("full-development report retained validation errors")
    banks = parsed.get("banks")
    aggregate = parsed.get("aggregate")
    if not isinstance(banks, list) or not all(
            isinstance(bank, dict) for bank in banks
    ) or not isinstance(aggregate, dict):
        raise ValueError("full-development summary payload is malformed")
    if parsed.get("configuration") != recorder.configuration_expected(
            expected_engine, 7):
        raise ValueError("full-development configuration mismatch")
    if parsed.get("expected_configuration") != recorder.configuration_expected(
            expected_engine, 7):
        raise ValueError("expected full-development configuration mismatch")
    recorder.validate_full_summaries(banks, aggregate, 7)
    if recorder.selection_threshold_errors(aggregate):
        raise ValueError("full-development report fails frozen thresholds")

    if [int(bank["games"]) for bank in banks] != [78, 76, 76, 76]:
        raise ValueError("full-development bank game counts mismatch")
    for key in ("candidate_p0", "candidate_p1"):
        color = recorder.common.parse_color(aggregate, key)
        if color[4] != 153 or color[0] < 77:
            raise ValueError("full-development physical-color threshold mismatch")
    for engine in ("candidate", "reference"):
        for phase, ceiling in (("first", 990.0), ("later", 198.0)):
            p99 = float(aggregate[f"{engine}_{phase}_ms_p99"])
            maximum = float(aggregate[f"{engine}_{phase}_ms_max"])
            if not all(math.isfinite(value) and value >= 0 for value in (p99, maximum)):
                raise ValueError("full-development timing is not finite")
            if p99 > maximum or maximum >= ceiling:
                raise ValueError("full-development timing gate mismatch")
    validate_serialized_gate_output(report)

    generated = report.get("generated_source_check")
    if not isinstance(generated, dict) or generated.get("returncode") != 0 or \
            generated.get("stderr") != "" or \
            "rank_4_jacek_hybrid submission is current (94004 characters)." \
            not in generated.get("stdout", ""):
        raise ValueError("generated source freshness check mismatch")
    started = parse_utc(report["started_utc"])
    ended = parse_utc(report["ended_utc"])
    if ended <= started or not isinstance(report.get("elapsed_monotonic_ns"), int) or \
            report["elapsed_monotonic_ns"] <= 0:
        raise ValueError("full-development duration is invalid")


def validate_sequence(
    control: dict[str, Any], rank4: dict[str, Any],
) -> None:
    if control["inputs_before"] != rank4["inputs_before"]:
        raise ValueError("full-development reports do not share exact inputs")
    control_end = parse_utc(control["ended_utc"])
    rank4_start = parse_utc(rank4["started_utc"])
    if control_end > rank4_start:
        raise ValueError("full-development report intervals overlap or are out of order")
    prerequisite = rank4.get("accepted_control_prerequisite")
    expected_prerequisite = {
        "path": str(
            (REPORT_DIRECTORY / f"{EXPECTED_REPORTS[0][1]}.json").relative_to(ROOT)
        ),
        "sha256": EXPECTED_REPORTS[0][1],
        "ended_utc": control["ended_utc"],
    }
    if prerequisite != expected_prerequisite:
        raise ValueError("Rank-4 report control-prerequisite binding mismatch")
    if control.get("accepted_control_prerequisite") is not None:
        raise ValueError("same-binary control unexpectedly has a prerequisite")


def validate_proof_matrix_prerequisite() -> dict[str, Any]:
    matrix = load_canonical_json(PROOF_MATRIX_PATH, EXPECTED_PROOF_MATRIX_SHA256)
    decision = matrix.get("decision")
    policy = matrix.get("evidence_policy")
    if matrix.get("schema") != "rank4-jacek-hybrid-proof-scope-matrix-v1" or \
            matrix.get("campaign_id") != CAMPAIGN_ID or \
            matrix.get("campaign_t0_utc") != CAMPAIGN_T0_UTC or \
            not isinstance(decision, dict) or \
            decision.get("selected_exact_proof_mask") != 7 or \
            decision.get("status") != "provisional" or \
            decision.get("final_qualification") is not False or \
            not isinstance(policy, dict) or \
            policy.get("validation_or_final_banks_read") is not False:
        raise ValueError("proof-scope matrix prerequisite mismatch")
    return matrix


def report_entry(engine: str, digest: str, report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["parsed"]["aggregate"]
    return {
        "reference_engine": engine,
        "candidate_exact_proof_mask": 7,
        "reference_exact_proof_mask": 0,
        "report_path": str((REPORT_DIRECTORY / f"{digest}.json").relative_to(ROOT)),
        "report_sha256": digest,
        "started_utc": report["started_utc"],
        "ended_utc": report["ended_utc"],
        "elapsed_monotonic_ns": report["elapsed_monotonic_ns"],
        "games": int(aggregate["games"]),
        "candidate_wins": int(aggregate["candidate_wins"]),
        "reference_wins": int(aggregate["reference_wins"]),
        "candidate_margin": int(aggregate["candidate_wins"]) - int(
            aggregate["reference_wins"]
        ),
        "candidate_p0": aggregate["candidate_p0"],
        "candidate_p1": aggregate["candidate_p1"],
        "candidate_first_ms_p99": aggregate["candidate_first_ms_p99"],
        "candidate_first_ms_max": aggregate["candidate_first_ms_max"],
        "candidate_later_ms_p99": aggregate["candidate_later_ms_p99"],
        "candidate_later_ms_max": aggregate["candidate_later_ms_max"],
        "reference_first_ms_p99": aggregate["reference_first_ms_p99"],
        "reference_first_ms_max": aggregate["reference_first_ms_max"],
        "reference_later_ms_p99": aggregate["reference_later_ms_p99"],
        "reference_later_ms_max": aggregate["reference_later_ms_max"],
        "candidate_proof_rebound": aggregate["candidate_proof_rebound"],
        "candidate_proof_root": aggregate["candidate_proof_root"],
        "candidate_proof_leaf": aggregate["candidate_proof_leaf"],
        "candidate_proof_ply1": aggregate["candidate_proof_ply1"],
        "candidate_proof_ply2": aggregate["candidate_proof_ply2"],
        "zero_own_operational_failures": True,
        "thresholds_passed": True,
    }


def build_selection() -> dict[str, Any]:
    plan_raw = PLAN_PATH.read_bytes()
    if sha256(plan_raw) != EXPECTED_PLAN_SHA256 or not plan_raw.isascii():
        raise ValueError("frozen full-development plan content mismatch")
    validate_proof_matrix_prerequisite()

    loaded: list[tuple[str, str, dict[str, Any]]] = []
    for engine, digest in EXPECTED_REPORTS:
        loaded.append((engine, digest, load_report(engine, digest)))
    control = loaded[0][2]
    rank4 = loaded[1][2]
    validate_sequence(control, rank4)

    common_inputs = copy.deepcopy(control["inputs_before"])
    common_inputs_sha256 = sha256(canonical_json(common_inputs))
    return {
        "schema": "rank4-jacek-hybrid-full-development-selection-v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_t0_utc": CAMPAIGN_T0_UTC,
        "classification": (
            "full-development-mask-selection-not-final-qualification"
        ),
        "final_qualification": False,
        "frozen_plan": {
            "path": str(PLAN_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_PLAN_SHA256,
            "bytes": len(plan_raw),
        },
        "proof_scope_prerequisite": {
            "path": str(PROOF_MATRIX_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_PROOF_MATRIX_SHA256,
            "selected_exact_proof_mask": 7,
            "status": "provisional-before-full-development",
        },
        "evaluated_identity": {
            "git_head": EXPECTED_HEAD,
            "generated_source": common_inputs[
                "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp"
            ],
            "hybrid_engine": common_inputs[
                "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp"
            ],
            "comparison_gate_source": common_inputs[
                "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp"
            ],
            "comparison_gate_binary": common_inputs[
                "build/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate"
            ],
            "rank4_source": common_inputs[
                "submissions/codingame/bots/rank_4/submission.cpp"
            ],
        },
        "common_inputs": common_inputs,
        "common_inputs_sha256": common_inputs_sha256,
        "evidence_policy": {
            "bank_role": "DEVELOPMENT only",
            "bank_depths": [4, 8, 12, 20],
            "games_per_comparison": 306,
            "games_per_physical_color": 153,
            "report_order": ["mask7-vs-mask0-hybrid-control", "mask7-vs-exact-rank4"],
            "sequential_nonoverlap": True,
            "identical_inputs_and_git_head": True,
            "accepted_control_bound_as_rank4_prerequisite": True,
            "zero_own_operational_failures": True,
            "validation_or_final_banks_read": False,
        },
        "reports": [report_entry(*item) for item in loaded],
        "decision": {
            "status": "selected-for-operational-activation",
            "selected_exact_proof_mask": 7,
            "fallback_mask3_triggered": False,
            "reason": (
                "mask 7 passed both preregistered 306-game comparisons, both "
                "physical-color thresholds, exact accounting, proof, and clock gates"
            ),
            "required_next_action": (
                "activate mask 7 on the ordinary protocol path, regenerate the exact "
                "source, and run fresh-source qualification gates"
            ),
            "final_qualification": False,
        },
    }


def write_selection(payload: dict[str, Any]) -> tuple[Path, str]:
    canonical = canonical_json(payload)
    digest = sha256(canonical)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"{digest}.json"
    temporary = OUTPUT_DIRECTORY / f".{digest}.{os.getpid()}.tmp"
    temporary.write_bytes(canonical)
    os.replace(temporary, destination)
    persisted = destination.read_bytes()
    if persisted != canonical or sha256(persisted) != digest:
        raise RuntimeError("persisted selection manifest failed byte/hash verification")
    return destination, digest


def main() -> int:
    destination, digest = write_selection(build_selection())
    print(f"selection={destination.relative_to(ROOT)}")
    print(f"selection_sha256={digest}")
    print("decision=mask7-selected-for-operational-activation")
    print("final_qualification=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
