#!/usr/bin/env python3
"""Verify and package the frozen DEVELOPMENT d20 proof-scope matrix."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = (
    ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock"
)
OUTPUT_DIRECTORY = REPORT_DIRECTORY / "matrix"
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
CAMPAIGN_T0_UTC = "2026-08-13T19:15:07Z"
EXPECTED_HEAD = "ba049648d5b86960dc0740812c76ced767b4412a"
EXPECTED_REPORTS = {
    (1, 0): "81ca5926cd4ca507514bdab77b80fb85f879c8fdadc9c28f1437a3a9cc99be57",
    (3, 1): "528f94bcfa1294c289e388321c54bd5070711e889cd3f5e9ef107b830f3d8d3b",
    (7, 3): "338723d65c028d4ea2c46dfd91f7a32a5f9d78bd3fdd1f65a972d8f81ff89c74",
    (15, 7): "970ffb0087f4d4767a3e8c586d53b423a0584cdfbd6b5796d0b0b49a863d3c9d",
}
EXPECTED_MARGINS = {(1, 0): 0, (3, 1): 4, (7, 3): 2, (15, 7): 0}
CRITICAL_INPUTS = {
    "tools/record_rank4_jacek_hybrid_proof_scope_clock.py": (
        23_660,
        "ffa76ea0915679192cbb5850251e709bd2f5868b3a8828a2d3d074cc457e43d1",
    ),
    "build/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate": (
        321_464,
        "619aa02ac67628345cec7cc7f4f0d21c7d4d49fc4f0fa1b77a2d4a3f7bd846b4",
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp": (
        94_004,
        "6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f",
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp": (
        62_777,
        "c8412600f0b90610660f24a02828e77f67cc7d78bdf775da11628b3355274215",
    ),
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp": (
        40_029,
        "d872e1720511d3045ac6890d566d795323e056db7ddf74cf365ce2f436f45b80",
    ),
    "results/rank_4_jacek_hybrid/openings/development_d20.tsv": (
        13_150,
        "2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4",
    ),
}


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"timestamp is not UTC: {value}")
    return parsed


def load_report(digest: str) -> dict[str, Any]:
    path = REPORT_DIRECTORY / f"{digest}.json"
    raw = path.read_bytes()
    if sha256(raw) != digest:
        raise ValueError(f"report content address mismatch: {path}")
    payload = json.loads(raw)
    if raw != canonical_json(payload):
        raise ValueError(f"report is not canonical ASCII JSON: {path}")
    return payload


def build_matrix() -> dict[str, Any]:
    if len(EXPECTED_REPORTS) != 4 or set(EXPECTED_REPORTS) != {
        (1, 0), (3, 1), (7, 3), (15, 7)
    }:
        raise ValueError("proof-scope tuple registry is incomplete or duplicated")
    if len(set(EXPECTED_REPORTS.values())) != 4:
        raise ValueError("proof-scope report registry reuses a digest")

    common_inputs: dict[str, Any] | None = None
    intervals: list[tuple[dt.datetime, dt.datetime, tuple[int, int]]] = []
    entries: list[dict[str, Any]] = []
    for mask_pair, digest in EXPECTED_REPORTS.items():
        report = load_report(digest)
        actual_pair = (
            report.get("candidate_exact_proof_mask"),
            report.get("reference_exact_proof_mask"),
        )
        if actual_pair != mask_pair:
            raise ValueError(f"report mask tuple mismatch: {digest}")
        if report.get("schema") != "rank4-jacek-hybrid-proof-scope-clock-v2":
            raise ValueError(f"report schema mismatch: {digest}")
        if report.get("campaign_id") != CAMPAIGN_ID or \
                report.get("campaign_t0_utc") != CAMPAIGN_T0_UTC:
            raise ValueError(f"report campaign binding mismatch: {digest}")
        if report.get("classification") != \
                "development-proof-scope-ablation-not-final-qualification":
            raise ValueError(f"report classification mismatch: {digest}")
        if report.get("final_qualification") is not False or \
                report.get("development_ablation_acceptable") is not True:
            raise ValueError(f"report was not accepted development evidence: {digest}")
        if report.get("returncode") != 0 or report.get("timed_out") is not False or \
                report.get("os_error") is not None or report.get("stderr") != "":
            raise ValueError(f"report process status is not clean: {digest}")
        if report.get("stable_inputs") is not True or \
                report.get("inputs_before") != report.get("inputs_after"):
            raise ValueError(f"report inputs changed during execution: {digest}")
        git = report.get("git", {})
        if git.get("head_before") != EXPECTED_HEAD or \
                git.get("head_after") != EXPECTED_HEAD or \
                git.get("tracked_status_before") != "" or \
                git.get("tracked_status_after") != "":
            raise ValueError(f"report git binding mismatch: {digest}")
        parsed = report.get("parsed", {})
        if parsed.get("validation_errors") != []:
            raise ValueError(f"report retained validation errors: {digest}")
        aggregate = parsed.get("aggregate", {})
        if aggregate.get("games") != "76" or aggregate.get("unfinished") != "0" or \
                aggregate.get("failed") != "0":
            raise ValueError(f"report game accounting mismatch: {digest}")
        margin = int(aggregate["candidate_wins"]) - int(aggregate["reference_wins"])
        if margin != EXPECTED_MARGINS[mask_pair]:
            raise ValueError(f"report marginal result mismatch: {digest}")

        inputs = report["inputs_before"]
        if common_inputs is None:
            common_inputs = inputs
        elif inputs != common_inputs:
            raise ValueError(f"report dependency/input identity drift: {digest}")
        started = parse_utc(report["started_utc"])
        ended = parse_utc(report["ended_utc"])
        if ended <= started or report.get("elapsed_monotonic_ns", 0) <= 0:
            raise ValueError(f"report duration is invalid: {digest}")
        intervals.append((started, ended, mask_pair))
        entries.append({
            "candidate_exact_proof_mask": mask_pair[0],
            "reference_exact_proof_mask": mask_pair[1],
            "report_path": str(
                (REPORT_DIRECTORY / f"{digest}.json").relative_to(ROOT)
            ),
            "report_sha256": digest,
            "started_utc": report["started_utc"],
            "ended_utc": report["ended_utc"],
            "elapsed_monotonic_ns": report["elapsed_monotonic_ns"],
            "candidate_wins": int(aggregate["candidate_wins"]),
            "reference_wins": int(aggregate["reference_wins"]),
            "candidate_margin": margin,
            "candidate_p0": aggregate["candidate_p0"],
            "candidate_p1": aggregate["candidate_p1"],
            "candidate_first_ms_max": aggregate["candidate_first_ms_max"],
            "candidate_later_ms_max": aggregate["candidate_later_ms_max"],
            "reference_first_ms_max": aggregate["reference_first_ms_max"],
            "reference_later_ms_max": aggregate["reference_later_ms_max"],
        })

    ordered_intervals = sorted(intervals)
    expected_order = list(EXPECTED_REPORTS)
    if [item[2] for item in ordered_intervals] != expected_order:
        raise ValueError("proof-scope reports were not run in preregistered order")
    for left, right in zip(ordered_intervals, ordered_intervals[1:]):
        if left[1] > right[0]:
            raise ValueError("proof-scope report intervals overlap")
    assert common_inputs is not None
    for path, (expected_bytes, expected_sha256) in CRITICAL_INPUTS.items():
        identity = common_inputs.get(path)
        if identity is None or identity.get("bytes") != expected_bytes or \
                identity.get("sha256") != expected_sha256:
            raise ValueError(f"critical input identity mismatch: {path}")

    inputs_fingerprint = sha256(canonical_json(common_inputs))
    return {
        "schema": "rank4-jacek-hybrid-proof-scope-matrix-v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_t0_utc": CAMPAIGN_T0_UTC,
        "classification": "development-proof-scope-matrix-not-final-qualification",
        "evidence_policy": {
            "bank": "DEVELOPMENT d20 only",
            "exact_tuple_set_once_each": True,
            "identical_input_and_dependency_identities": True,
            "identical_git_head": EXPECTED_HEAD,
            "sequential_nonoverlap": True,
            "all_reports_acceptable": True,
            "validation_or_final_banks_read": False,
        },
        "common_inputs": common_inputs,
        "common_inputs_sha256": inputs_fingerprint,
        "reports": entries,
        "decision": {
            "status": "provisional",
            "selected_exact_proof_mask": 7,
            "marginal_candidate_win_minus_reference_win": {
                "root": 0,
                "leaf": 4,
                "ply1": 2,
                "ply2": 0,
            },
            "interpretation": {
                "root": "neutral",
                "leaf": "positive",
                "ply1": "positive",
                "ply2": "neutral",
            },
            "full_development_clock_comparisons_required": [
                "mask7-vs-mask0-hybrid-control",
                "mask7-vs-exact-rank4",
            ],
            "final_qualification": False,
        },
    }


def write_matrix(payload: dict[str, Any]) -> tuple[Path, str]:
    canonical = canonical_json(payload)
    digest = sha256(canonical)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"{digest}.json"
    temporary = OUTPUT_DIRECTORY / f".{digest}.{os.getpid()}.tmp"
    temporary.write_bytes(canonical)
    os.replace(temporary, destination)
    persisted = destination.read_bytes()
    if persisted != canonical or sha256(persisted) != digest:
        raise RuntimeError("persisted matrix failed byte/hash verification")
    return destination, digest


def main() -> int:
    destination, digest = write_matrix(build_matrix())
    print(f"matrix={destination.relative_to(ROOT)}")
    print(f"matrix_sha256={digest}")
    print("decision=provisional-mask7 full-development-required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
