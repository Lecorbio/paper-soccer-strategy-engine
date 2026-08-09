#!/usr/bin/env python3

"""Freeze T10 validation rollover and a fresh candidate-independent final."""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import pathlib

from acquire_t10_evidence import PARSEABLE_RECORDS, RAW_RECORDS, SOURCE_AGENTS
from build_goal_shell_banks import (
    ROOT,
    balanced_sample,
    extract_states,
    sha256_bytes,
    stable_json,
    tsv_bytes,
)
from freeze_t8_banks import (
    FINAL_BANK as T8_FINAL,
    EXPECTED_FINAL_SHA256 as T8_FINAL_SHA256,
    EXPECTED_VALIDATION_SHA256 as T8_VALIDATION_SHA256,
    VALIDATION_BANK as T8_VALIDATION,
    prior_keys_and_games,
    relabel_public_elite,
    t7_tier_metadata,
)


HERE = pathlib.Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
T9_EVIDENCE_MANIFEST = REFERENCE / "t9_evidence_manifest.json"
T9_FINAL = REFERENCE / "t9_sealed_final.tsv"
T10_VALIDATION = REFERENCE / "t10_prospective_validation.tsv"
T10_FINAL = REFERENCE / "t10_sealed_final.tsv"
T10_EVIDENCE_MANIFEST = REFERENCE / "t10_evidence_manifest.json"

RAW_SHA256 = "74b412255304ec9d3043b2f10c42c63791dfcaa3a1539d94da3a73bbb359a3de"
PARSEABLE_SHA256 = (
    "00b912e25122c9600e97ca878669d9bc6ecf5a7f1339a87606154a7b135032db"
)
T9_EVIDENCE_MANIFEST_SHA256 = (
    "5c4465b6cabec90c8ec0f5bb9cc8b9af65182f03136681e932a08e01de365845"
)
T9_FINAL_SHA256 = T8_FINAL_SHA256
STRENGTH_PROTOCOL_SHA256 = (
    "8d73a1c92d43d73a8ebe48a63084f5f5c578ead9516058920c0700165ec3851c"
)

T9_CANDIDATE = "conservative_frontier_proof"
T9_CANDIDATE_SHA256 = (
    "b13e1418b4fdd6f719208bd5ab6dd84f67fa55a7cc66f1c988d2ab8dcc9f6c69"
)
T9_BOUND_MANIFEST_SHA256 = (
    "4004325c6c34300c2781fb3464c9421de86c77d4846f45552d79c0d82e26f249"
)
T9_DECISION_SHA256 = (
    "0aba21a0307fad198e7b3f081de739511e8c9b8a78bddd43c2853c64dd6f81ce"
)
T9_VALIDATION_REPORT_SHA256 = (
    "c37a7e7bc6ca658b0ada8b85374d26097d936203e3337419453907cff77198ec"
)
T9_RESULT_DIRECTORY = (
    ROOT
    / "results/codingame/promotion/conservative_frontier_proof/"
      "b13e1418b4fdd6f7-4004325c6c34"
)
T9_FINAL_LEDGER = (
    ROOT
    / ".git/papersoccer-promotion/"
      "locked-test-consumption-e6c8efaa094576ad.json"
)

GAME_CAP = 2
GAME_DISTANCE_CAP = 1
FOCUS_AGENT_CAP = 18
MIN_ELITE_ROWS = 12
MIN_FIELD_ROWS = 40
MIN_DISTINCT_FOCUS_AGENTS = 12
DISTANCE_MINIMUMS = {"d0": 4, "d1": 24, "d2": 24}
FINAL_RECORDS = 69

EXPECTED_POOL = {
    "parseable_games": 650,
    "no_shell_games": 534,
    "overlap_states": 126,
    "unique_states": 144,
    "source_games": 44,
    "focus_agents": 21,
    "distance": {"d0": 4, "d1": 65, "d2": 75},
    "winner_tiers": {"elite": 64, "field": 80},
    "cap_two_capacity": 69,
}
EXPECTED_FINAL_COMPOSITION = {
    "records": 69,
    "source_games": 44,
    "focus_agents": 21,
    "maximum_focus_rows": 9,
    "winner_tiers": {"elite": 25, "field": 44},
    "distance": {"d0": 4, "d1": 35, "d2": 30},
}


def verified_bytes(path: pathlib.Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise RuntimeError(f"hash mismatch: {path.relative_to(ROOT)}")
    return data


def composition(rows: list[dict]) -> dict:
    per_focus = collections.Counter(int(row["_focus_agent_id"]) for row in rows)
    return {
        "records": len(rows),
        "source_games": len({int(row["source_game_id"]) for row in rows}),
        "focus_agents": len(per_focus),
        "maximum_focus_rows": max(per_focus.values()),
        "focus_rows": [
            {"agent_id": agent_id, "rows": count}
            for agent_id, count in sorted(per_focus.items())
        ],
        "winner_tiers": dict(sorted(collections.Counter(
            row["winner_tier"] for row in rows
        ).items())),
        "distance": dict(sorted(collections.Counter(
            row["stratum"] for row in rows
        ).items())),
    }


def rollover_validation() -> bytes:
    data = verified_bytes(T9_FINAL, T9_FINAL_SHA256)
    with T9_FINAL.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(rows) != 72 or any(row["split"] != "test" for row in rows):
        raise RuntimeError("T9 final rollover source changed")
    return tsv_bytes([dict(row, split="validation") for row in rows])


def prior_keys_and_bank_games() -> tuple[set[str], set[int]]:
    keys, games = prior_keys_and_games()
    for path, expected_sha256 in (
        (T8_VALIDATION, T8_VALIDATION_SHA256),
        (T8_FINAL, T8_FINAL_SHA256),
    ):
        verified_bytes(path, expected_sha256)
        with path.open(newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                keys.add(row["canonical_key"])
                games.add(int(row["source_game_id"]))
    return keys, games


def fresh_final_pool() -> tuple[list[dict], dict]:
    raw_data = verified_bytes(RAW_RECORDS, RAW_SHA256)
    parseable_data = verified_bytes(PARSEABLE_RECORDS, PARSEABLE_SHA256)
    raw = json.loads(raw_data)
    payload = json.loads(parseable_data)
    expected_agents = [agent_id for agent_id, _ in SOURCE_AGENTS]
    if (
        raw.get("schema") != "papersoccer.frozen-t10-evidence-ladder.v1"
        or raw.get("agent_ids") != expected_agents
        or len(raw.get("records", [])) != 737
        or payload.get("schema")
        != "papersoccer.frozen-t10-evidence-ladder.v2"
        or payload.get("agent_ids") != expected_agents
        or payload.get("raw_sha256") != RAW_SHA256
        or len(payload.get("records", [])) != EXPECTED_POOL["parseable_games"]
        or payload.get("structurally_rejected_games") != 87
    ):
        raise RuntimeError("T10 acquisition provenance changed")

    prior_keys, prior_games = prior_keys_and_bank_games()
    public_elite_names, _, _ = t7_tier_metadata()
    candidates = []
    no_shell = 0
    overlap_states = 0
    for record in payload["records"]:
        winner = (
            int(record["player_id"])
            if record.get("won")
            else 1 - int(record["player_id"])
        )
        states = extract_states("test", record, winner, elite_balance=True)
        if not states:
            no_shell += 1
            continue
        if int(record["game_id"]) in prior_games:
            raise RuntimeError("T10 acquisition overlaps a prior source game")
        winner_name = (
            str(record["focus_name"])
            if record.get("won")
            else str(record["opponent_name"])
        )
        for state in states:
            if state["canonical_key"] in prior_keys:
                overlap_states += 1
                continue
            candidates.append(
                relabel_public_elite(
                    state, winner_name, public_elite_names
                )
            )

    seen = set(prior_keys)
    unique = []
    for state in sorted(candidates, key=lambda item: item["_selection_hash"]):
        if state["canonical_key"] in seen:
            continue
        seen.add(state["canonical_key"])
        unique.append(state)
    by_game = collections.defaultdict(set)
    for state in unique:
        by_game[int(state["source_game_id"])].add(state["stratum"])
    pool = {
        "parseable_games": len(payload["records"]),
        "no_shell_games": no_shell,
        "overlap_states": overlap_states,
        "unique_states": len(unique),
        "source_games": len(by_game),
        "focus_agents": len({int(row["_focus_agent_id"]) for row in unique}),
        "distance": dict(sorted(collections.Counter(
            row["stratum"] for row in unique
        ).items())),
        "winner_tiers": dict(sorted(collections.Counter(
            row["winner_tier"] for row in unique
        ).items())),
        "cap_two_capacity": sum(min(GAME_CAP, len(value)) for value in by_game.values()),
    }
    if pool != EXPECTED_POOL:
        raise RuntimeError(f"T10 final-pool capacity changed: {pool}")
    return unique, pool


def assert_final(rows: list[dict], prior_keys: set[str], prior_games: set[int]) -> None:
    keys = [row["canonical_key"] for row in rows]
    games = [int(row["source_game_id"]) for row in rows]
    per_game = collections.Counter(games)
    per_game_distance = collections.Counter(
        (int(row["source_game_id"]), row["stratum"]) for row in rows
    )
    per_focus = collections.Counter(int(row["_focus_agent_id"]) for row in rows)
    tiers = collections.Counter(row["winner_tier"] for row in rows)
    distance = collections.Counter(row["stratum"] for row in rows)
    if len(rows) != FINAL_RECORDS or len(keys) != len(set(keys)):
        raise RuntimeError("T10 final size or canonical uniqueness changed")
    if set(keys) & prior_keys or set(games) & prior_games:
        raise RuntimeError("T10 final overlaps prior evidence")
    if max(per_game.values()) > GAME_CAP or max(per_game_distance.values()) > 1:
        raise RuntimeError("T10 final exceeds source-game caps")
    if max(per_focus.values()) > FOCUS_AGENT_CAP or len(per_focus) < 12:
        raise RuntimeError("T10 final lacks focus diversity")
    if tiers["elite"] < MIN_ELITE_ROWS or tiers["field"] < MIN_FIELD_ROWS:
        raise RuntimeError("T10 final lacks winner-tier coverage")
    if any(distance[key] < value for key, value in DISTANCE_MINIMUMS.items()):
        raise RuntimeError("T10 final lacks distance coverage")
    actual = composition(rows)
    for key, expected in EXPECTED_FINAL_COMPOSITION.items():
        if actual[key] != expected:
            raise RuntimeError(f"T10 final composition changed: {key}")


def audit_t9_final_nonconsumption() -> dict:
    decision_data = verified_bytes(
        T9_RESULT_DIRECTORY / "decision.json", T9_DECISION_SHA256
    )
    verified_bytes(
        T9_RESULT_DIRECTORY / "validation.json", T9_VALIDATION_REPORT_SHA256
    )
    decision = json.loads(decision_data)
    expected_status = {
        "development": "pass",
        "initial": "pass",
        "test": "not_run_due_to_rejection",
        "validation": "reject",
    }
    if (
        decision.get("bot") != T9_CANDIDATE
        or decision.get("candidate_submission_sha256") != T9_CANDIDATE_SHA256
        or decision.get("manifest_sha256") != T9_BOUND_MANIFEST_SHA256
        or decision.get("failed_stage") != "validation"
        or decision.get("stage_status") != expected_status
        or decision.get("verdict") != "REJECT"
    ):
        raise RuntimeError("T9 rejection provenance changed")
    forbidden = (
        T9_RESULT_DIRECTORY / "test.json",
        T9_RESULT_DIRECTORY / "shards/test",
        T9_RESULT_DIRECTORY / f"banks/test-{T9_FINAL_SHA256}.tsv",
        T9_FINAL_LEDGER,
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("T9 sealed final was consumed")
    return {
        "bound_manifest_sha256": T9_BOUND_MANIFEST_SHA256,
        "candidate_submission_sha256": T9_CANDIDATE_SHA256,
        "decision_sha256": T9_DECISION_SHA256,
        "failed_stage": "validation",
        "stage_status": expected_status,
        "final_report_exists": False,
        "final_shards_exist": False,
        "final_immutable_snapshot_exists": False,
        "final_ledger_marker_exists": False,
    }


def build() -> tuple[bytes, bytes, bytes]:
    validation_bytes = rollover_validation()
    unique, pool = fresh_final_pool()
    final_rows = balanced_sample(
        unique,
        FINAL_RECORDS,
        game_cap=GAME_CAP,
        game_distance_cap=GAME_DISTANCE_CAP,
        focus_agent_cap=FOCUS_AGENT_CAP,
        minimum_elite_rows=MIN_ELITE_ROWS,
    )
    prior_keys, prior_games = prior_keys_and_bank_games()
    assert_final(final_rows, prior_keys, prior_games)
    final_bytes = tsv_bytes(final_rows)

    with T10_VALIDATION.open(newline="") if T10_VALIDATION.exists() else T9_FINAL.open(newline="") as source:
        source_rows = list(csv.DictReader(source, delimiter="\t"))
    validation_games = {int(row["source_game_id"]) for row in source_rows}
    validation_keys = {row["canonical_key"] for row in source_rows}
    if validation_games & {int(row["source_game_id"]) for row in final_rows}:
        raise RuntimeError("T10 banks share source games")
    if validation_keys & {row["canonical_key"] for row in final_rows}:
        raise RuntimeError("T10 banks share canonical states")

    t9_manifest_data = verified_bytes(
        T9_EVIDENCE_MANIFEST, T9_EVIDENCE_MANIFEST_SHA256
    )
    t9 = json.loads(t9_manifest_data)
    protocol = copy.deepcopy(t9["prospective_strength_protocol"])
    if sha256_bytes(stable_json(protocol)) != STRENGTH_PROTOCOL_SHA256:
        raise RuntimeError("T10 inherited protocol changed")
    audit = audit_t9_final_nonconsumption()
    validation_sha256 = sha256_bytes(validation_bytes)
    final_sha256 = sha256_bytes(final_bytes)
    sources = {
        str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
        for path in (
            pathlib.Path(__file__).resolve(),
            HERE / "acquire_t10_evidence.py",
            RAW_RECORDS,
            PARSEABLE_RECORDS,
            T9_EVIDENCE_MANIFEST,
            T9_FINAL,
        )
    }
    manifest = {
        "schema": "papersoccer.candidate-independent-t10-evidence.v1",
        "status": "frozen_before_candidate_binding",
        "candidate": None,
        "candidate_submission_sha256": None,
        "prospective_strength_protocol": protocol,
        "prospective_strength_protocol_sha256": STRENGTH_PROTOCOL_SHA256,
        "protocol_carry_forward": {
            "source_ladder": "T9",
            "source_protocol_sha256": STRENGTH_PROTOCOL_SHA256,
            "semantic_changes": [],
            "stage_bank_aliases": {
                "validation": "reference/t10_prospective_validation.tsv",
                "test": "reference/t10_sealed_final.tsv",
            },
        },
        "t9_decision_and_final_nonconsumption": audit,
        "selection": {
            "candidate_independent": True,
            "validation": (
                "the outcome-unseen T9 sealed final, deterministically relabeled "
                "from split=test to split=validation after T9 validation rejection"
            ),
            "validation_source_sha256": T9_FINAL_SHA256,
            "final": (
                "fresh T10 v2 public acquisition, excluding every prior source "
                "game and canonical state; deterministic balanced sample at the "
                "maximum capacity under the frozen game caps"
            ),
            "normalization": (
                "historical winner rotated to player zero; horizontal reflection dedup"
            ),
            "caps": {
                "rows_per_game": GAME_CAP,
                "rows_per_game_distance": GAME_DISTANCE_CAP,
                "rows_per_focus_agent": FOCUS_AGENT_CAP,
            },
            "floors": {
                "elite_rows": MIN_ELITE_ROWS,
                "field_rows": MIN_FIELD_ROWS,
                "distinct_focus_agents": MIN_DISTINCT_FOCUS_AGENTS,
                "distance_rows": DISTANCE_MINIMUMS,
            },
            "fresh_pool": pool,
        },
        "immutability": {
            "at_candidate_binding": (
                "the active manifest must pin this evidence manifest, inherited "
                "protocol, candidate submission, and all harness hashes"
            ),
            "after_candidate_binding": (
                "this manifest and both T10 banks are immutable; a candidate "
                "change requires a new versioned ladder"
            ),
        },
        "sources": dict(sorted(sources.items())),
        "banks": {
            "reference/t10_prospective_validation.tsv": {
                "role": "prospective_validation",
                "sealed": False,
                "records": 72,
                "sha256": validation_sha256,
                "carried_forward_from": {
                    "reference": "reference/t9_sealed_final.tsv",
                    "sha256": T9_FINAL_SHA256,
                    "transformation": "split column test to validation only",
                },
            },
            "reference/t10_sealed_final.tsv": {
                "role": "sealed_final",
                "sealed": True,
                "sha256": final_sha256,
                **composition(final_rows),
            },
        },
    }
    return validation_bytes, final_bytes, stable_json(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    validation, final, manifest = build()
    artifacts = {
        T10_VALIDATION: validation,
        T10_FINAL: final,
        T10_EVIDENCE_MANIFEST: manifest,
    }
    stale = []
    for path, content in artifacts.items():
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
            continue
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError(f"refusing to replace frozen {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(
            f"froze {path.relative_to(ROOT)} sha256={sha256_bytes(content)}"
        )
    if stale:
        raise SystemExit("stale T10 evidence artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
