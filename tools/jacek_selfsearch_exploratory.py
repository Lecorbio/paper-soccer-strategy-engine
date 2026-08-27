#!/usr/bin/env python3
"""Run a local, pilot-only self-search experiment with a failed-gate diversity actor."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import pathlib
import sys
import traceback
from collections.abc import Mapping

import jacek_replay_rebuild as rebuild
import jacek_selfsearch_workflow as selfsearch
from jacek_replay_workflow import artifact_snapshot, canonical_json_bytes


EXPLORATORY_CAMPAIGN_ID = "selfsearch-exploratory-20260827-v1"
EXPLORATORY_PILOT_CAMPAIGN_ID = "selfsearch-exploratory-pilot-20260827-v1"
EXPLORATORY_FULL_CAMPAIGN_ID = "selfsearch-exploratory-full-20260827-v1"
DIVERSITY_CANDIDATE_ID = "canonical-r2-s20260825"
REBUILD_ID = "replay-rebuild-20260826-v1"
V6_EXPERIMENTAL_REFERENCE_SHA256 = (
    "bfcc1755ab9b71261bedc9b9c9b59e38e3d440d7c80e7056a9f0bc812ffc9c80"
)
DIVERSITY_ACTOR_SHA256 = (
    "5ecbd618cbf0c3b826cd8b336db9e9c966971c81961d97239dd8864e8d507abf"
)
REBUILD_INPUTS_SHA256 = (
    "9d26b2e1992089384c42afddcd6c43bbb845859b08624c4baf9950a053960f7c"
)
REBUILD_SUMMARY_SHA256 = (
    "57b912ea07a4aa6275d9b34ecfa4e8494c40f8c4ce260870adde70c1bf5a8446"
)
REBUILD_STATUS_SHA256 = (
    "dac23233df209f8e8635349815d1c3428b748c7831b278f0ec68142eceb69c97"
)
CANONICAL_PHASE_SHA256 = (
    "d4b4afc869c8237dd7838fdbbb70712664015ae23df9aec4edc1e3879e71caa0"
)
DIVERSITY_CANDIDATE_RECORD_SHA256 = (
    "ced3588013dbacc4410a28e5567e1deee9155a1efaa8a715c4aadceb7e391e69"
)
V6_SUMMARY_SHA256 = (
    "3bd0800a7d8f80a29564bca438657470ce53bde7d49d3b15cf2fad95d66b6ac1"
)
EXPLORATORY_GAME_SEED = 2026082709
EXPLORATORY_OPENING_SEED = 2026082711

LINEAGE_SCHEMA = "papersoccer.jacek-selfsearch-exploratory-lineage.v1"
SUMMARY_SCHEMA = "papersoccer.jacek-selfsearch-exploratory-summary.v1"
CONTINUATION_SCHEMA = (
    "papersoccer.jacek-selfsearch-exploratory-full-continuation.v1"
)
STATUS_SCHEMA = "papersoccer.jacek-selfsearch-exploratory-status.v1"
LAUNCH_SCHEMA = "papersoccer.jacek-selfsearch-exploratory-launch.v1"

PHASE_ORDER = (
    "v5-recovery", "canonical-basins", "scratch-joint", "residual"
)
EXPECTED_NEAR_PASS_COUNTS = {
    "matched": {
        "games": 600, "wins": 323, "colors": [167, 156],
        "illegal": 0, "unfinished": 0,
    },
    "incumbent": {
        "games": 600, "wins": 312, "colors": [156, 156],
        "illegal": 0, "unfinished": 0,
    },
    "rank4": {
        "games": 600, "wins": 386, "colors": [190, 196],
        "illegal": 0, "unfinished": 0,
    },
    "jacek-nn": {
        "games": 600, "wins": 388, "colors": [194, 194],
        "illegal": 0, "unfinished": 0,
    },
}


EXPLORATORY_PILOT_SPEC = dataclasses.replace(
    selfsearch.PILOT_SPEC,
    campaign_id=EXPLORATORY_PILOT_CAMPAIGN_ID,
    configuration={
        **selfsearch.PILOT_SPEC.configuration,
        "campaign_id": EXPLORATORY_PILOT_CAMPAIGN_ID,
    },
    game_seed=EXPLORATORY_GAME_SEED,
    opening_seed=EXPLORATORY_OPENING_SEED,
)


def _load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _atomic_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    selfsearch._atomic_json(path, dict(value))


def _write_exact_json(path: pathlib.Path, value: Mapping[str, object], label: str) -> None:
    if path.exists():
        if _load_json(path, label) != dict(value):
            raise ValueError(f"existing {label} differs from frozen content")
        return
    _atomic_json(path, value)


def _body_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return result


def _artifact_matches(record: object) -> bool:
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("bytes")) is not int
    ):
        return False
    try:
        current = artifact_snapshot(pathlib.Path(record["path"]))
    except (OSError, ValueError):
        return False
    return all(current.get(key) == value for key, value in record.items())


def _status(path: pathlib.Path, phase: str, **details: object) -> None:
    _atomic_json(
        path,
        {
            "schema": STATUS_SCHEMA,
            "campaign_id": EXPLORATORY_CAMPAIGN_ID,
            "phase": phase,
            **details,
        },
    )


def _near_pass_key(record: Mapping[str, object]) -> tuple[float, float, str]:
    """Rank failed full screens by their worst unchanged gate margin."""

    full = record.get("full")
    decision = full.get("decision") if isinstance(full, dict) else None
    counts = decision.get("counts") if isinstance(decision, dict) else None
    candidate = full.get("candidate") if isinstance(full, dict) else None
    if (
        not isinstance(counts, dict)
        or set(counts) != {"matched", "incumbent", "rank4", "jacek-nn"}
        or not isinstance(candidate, dict)
        or not isinstance(candidate.get("sha256"), str)
        or decision.get("eligible_for_full") is not False
    ):
        raise ValueError("failed rebuild full screen is malformed")
    margins: list[float] = []
    for name, value in counts.items():
        if not isinstance(value, dict):
            raise ValueError("failed rebuild panel count is malformed")
        if int(value.get("illegal", -1)) or int(value.get("unfinished", -1)):
            raise ValueError("failed rebuild candidate is not operational")
        win_threshold = 325 if name in {"matched", "incumbent"} else 306
        color_threshold = 156 if name in {"matched", "incumbent"} else 143
        colors = value.get("colors")
        if (
            value.get("games") != 600
            or not isinstance(colors, list)
            or len(colors) != 2
        ):
            raise ValueError("failed rebuild panel coverage is incomplete")
        margins.extend(
            (
                int(value["wins"]) / win_threshold - 1.0,
                min(map(int, colors)) / color_threshold - 1.0,
            )
        )
    candidate_p99 = float(decision.get("candidate_p99_ms", math.nan))
    uncontended = float(decision.get("uncontended_max_ms", math.nan))
    if (
        not math.isfinite(candidate_p99)
        or not math.isfinite(uncontended)
        or candidate_p99 > 25.0
        or uncontended >= 1_000.0
    ):
        raise ValueError("failed rebuild candidate does not meet latency limits")
    anchor = decision.get("anchor_candidate")
    if not isinstance(anchor, dict):
        raise ValueError("failed rebuild candidate has no canonical metrics")
    huber = float(anchor.get("weighted_huber", math.nan))
    if not math.isfinite(huber):
        raise ValueError("failed rebuild candidate canonical Huber is invalid")
    return (-min(margins), huber, str(candidate["sha256"]))


def _validate_opening_banks_without_sealed(
    record: Mapping[str, object],
) -> None:
    """Validate model-selection banks while leaving final opening bytes unread."""

    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != rebuild.REBUILD_BANKS_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed
        != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError("rebuild opening-bank manifest is stale or corrupt")
    development = body.get("development")
    sealed = body.get("final")
    if (
        not isinstance(development, dict)
        or development.get("model_selection_eligible") is not True
        or not isinstance(sealed, dict)
        or sealed.get("model_selection_eligible") is not False
        or sealed.get("sealed_until_selected_runtime_receipt") is not True
        or not isinstance(sealed.get("artifact"), dict)
        or not isinstance(sealed["artifact"].get("path"), str)
        or not isinstance(sealed["artifact"].get("sha256"), str)
        or type(sealed["artifact"].get("bytes")) is not int
    ):
        raise ValueError("sealed final bank policy or opaque identity changed")
    for value, label in (
        (body.get("comparison"), "comparison"),
        (development.get("artifact"), "development"),
    ):
        if (
            not isinstance(value, dict)
            or not _artifact_matches(value)
        ):
            raise ValueError(f"rebuild {label} bank binding is stale")
    exclusions = body.get("excluded_banks")
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("rebuild bank exclusions are incomplete")
    excluded_states: set[str] = set()
    for exclusion in exclusions:
        if (
            not isinstance(exclusion, dict)
            or not _artifact_matches(exclusion)
        ):
            raise ValueError("rebuild bank exclusion binding is stale")
        excluded_states.update(
            selfsearch._comparison_bank_states(pathlib.Path(exclusion["path"]))
        )
    development_path = pathlib.Path(development["artifact"]["path"])
    states = selfsearch._comparison_bank_states(development_path, "development")
    detailed = [
        artifact_snapshot(pathlib.Path(exclusion["path"]))
        for exclusion in exclusions
    ]
    expected_development = {
        "pairs": rebuild.FULL_SCREEN_PAIRS,
        "seed": rebuild.DEVELOPMENT_BANK_SEED,
        "opening_plies": 12,
        "classification": "development",
        "states_sha256": hashlib.sha256(
            "\n".join(sorted(states)).encode()
        ).hexdigest(),
        "exclusions": detailed,
    }
    sealed_configuration = sealed.get("configuration")
    if (
        len(states) != rebuild.FULL_SCREEN_PAIRS
        or states & excluded_states
        or development.get("configuration") != expected_development
        or not isinstance(sealed_configuration, dict)
        or sealed_configuration.get("pairs") != rebuild.FULL_SCREEN_PAIRS
        or sealed_configuration.get("seed") != rebuild.FINAL_BANK_SEED
        or sealed_configuration.get("opening_plies") != 12
        or sealed_configuration.get("classification") != "final"
        or not isinstance(sealed_configuration.get("states_sha256"), str)
        or len(sealed_configuration["states_sha256"]) != 64
        or sealed_configuration.get("exclusions")
        != [*detailed, artifact_snapshot(development_path)]
    ):
        raise ValueError("rebuild opening-bank configuration changed")


def _validate_rebuild_inputs_without_sealed(path: pathlib.Path) -> dict[str, object]:
    """Validate only rebuild inputs required by development replay."""

    record = _load_json(path, "frozen rebuild inputs")
    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != rebuild.REBUILD_INPUT_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError("frozen rebuild inputs are stale or corrupt")
    policies = body.get("policies")
    if not isinstance(policies, dict) or any(
        policies.get(key) is not False
        for key in (
            "regenerate_teacher_labels", "external_upload", "replace_rank4",
            "canonical_test_model_selection_eligible",
            "sealed_final_bank_model_selection_eligible",
        )
    ):
        raise ValueError("frozen rebuild input policy was weakened")
    # Blind holdout and protected-test artifacts are deliberately not opened:
    # this exploratory pilot consumes only development-time evidence.
    for label in (
        "corpus", "build_manifest", "matrix", "opening_banks", "comparison",
        "rank4_teacher", "incumbent_runtime",
    ):
        snapshot = body.get(label)
        if (
            not isinstance(snapshot, dict)
            or not _artifact_matches(snapshot)
        ):
            raise ValueError(f"frozen rebuild {label} binding is stale")
    corpus = rebuild.load_frozen_rebuild_corpus(pathlib.Path(body["corpus"]["path"]))
    # Accessing the corpus object here validates its role manifests; protected
    # test targets remain behind its training-incompatible interface.
    if not corpus.training_manifest_paths("search"):
        raise ValueError("frozen rebuild corpus has no search training role")
    rebuild.validate_matrix(
        _load_json(pathlib.Path(body["matrix"]["path"]), "rebuild matrix")
    )
    banks = _load_json(pathlib.Path(body["opening_banks"]["path"]), "rebuild banks")
    _validate_opening_banks_without_sealed(banks)
    build = rebuild.validate_rebuild_build_manifest(
        pathlib.Path(body["build_manifest"]["path"])
    )
    if (
        build["binaries"]["comparison"] != body["comparison"]
        or build["binaries"]["rank4_teacher"] != body["rank4_teacher"]
    ):
        raise ValueError("rebuild inputs do not match their frozen build")
    repository = body.get("repository")
    started = body.get("started_at_unix")
    deadline = body.get("same_architecture_deadline_unix")
    if (
        not isinstance(repository, dict)
        or repository.get("clean") is not True
        or not isinstance(repository.get("commit"), str)
        or len(repository["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in repository["commit"])
        or not isinstance(started, (int, float))
        or not isinstance(deadline, (int, float))
        or float(deadline)
        != float(started) + rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS
    ):
        raise ValueError("frozen rebuild repository/deadline semantics changed")
    return record


def _verify_hashed_record(
    value: Mapping[str, object], *, schema: str, label: str,
) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or claimed != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError(f"{label} body hash is invalid")


def _validate_embedded_full_screen(record: Mapping[str, object]) -> None:
    full = record.get("full")
    if not isinstance(full, dict):
        raise ValueError("rebuild full-screen evidence is malformed")
    reports = full.get("reports")
    latency_snapshot = full.get("latency")
    decision = full.get("decision")
    if (
        not isinstance(reports, dict)
        or set(reports) != {"matched", "incumbent", "rank4", "jacek-nn"}
        or not isinstance(latency_snapshot, dict)
        or not isinstance(decision, dict)
    ):
        raise ValueError("rebuild full-screen report set is incomplete")
    latency = _load_json(pathlib.Path(latency_snapshot["path"]), "development latency")
    recomputed = selfsearch.pilot_decision(
        matched_report=pathlib.Path(reports["matched"]["path"]),
        incumbent_report=pathlib.Path(reports["incumbent"]["path"]),
        rank4_report=pathlib.Path(reports["rank4"]["path"]),
        jacek_nn_report=pathlib.Path(reports["jacek-nn"]["path"]),
        anchor_candidate=decision.get("anchor_candidate", {}),
        anchor_incumbent=decision.get("anchor_incumbent", {}),
        uncontended_max_ms=float(
            latency.get("summary", {}).get("candidate", {}).get("max_ms", math.nan)
        ),
    )
    if recomputed != decision:
        raise ValueError("rebuild development decision changed after local replay")


def _validate_v6_actor(actor: pathlib.Path) -> dict[str, object]:
    """Bind the rebuild reference to the completed v6 pilot output."""

    actor = actor.resolve()
    if (
        actor.name != "jacek_replay_bfm.runtime"
        or actor.parent.name != "search"
        or actor.parent.parent.name != "models"
        or actor.parent.parent.parent.name != "pilot"
        or actor.parent.parent.parent.parent.name
        != selfsearch.AUTO_CAMPAIGN_ID
    ):
        raise ValueError("rebuild reference is not the frozen v6 pilot actor")
    summary_path = actor.parent.parent.parent.parent / "final-summary.json"
    if artifact_snapshot(summary_path)["sha256"] != V6_SUMMARY_SHA256:
        raise ValueError("v6 campaign summary differs from frozen evidence")
    summary = _load_json(summary_path, "v6 campaign summary")
    pilot = summary.get("pilot")
    decision = pilot.get("decision") if isinstance(pilot, dict) else None
    if (
        summary.get("schema") != selfsearch.CAMPAIGN_SUMMARY_SCHEMA
        or summary.get("terminal") != "pilot-rejected"
        or summary.get("canonical_promotion_eligible") is not False
        or summary.get("full") is not None
        or summary.get("publication") is not None
        or not isinstance(pilot, dict)
        or pilot.get("campaign_id") != selfsearch.PILOT_CAMPAIGN_ID
        or pathlib.Path(str(pilot.get("search_runtime", ""))).resolve() != actor
        or not isinstance(decision, dict)
        or decision.get("eligible_for_full") is not False
    ):
        raise ValueError("v6 actor campaign evidence is not the rejected pilot")
    return artifact_snapshot(summary_path)


def validate_diversity_lineage(
    *, inputs_manifest: pathlib.Path, rebuild_summary: pathlib.Path,
) -> dict[str, object]:
    """Replay the failed rebuild and select its deterministic closest near-pass."""

    inputs_manifest = inputs_manifest.resolve()
    rebuild_summary = rebuild_summary.resolve()
    if (
        artifact_snapshot(inputs_manifest)["sha256"] != REBUILD_INPUTS_SHA256
        or artifact_snapshot(rebuild_summary)["sha256"] != REBUILD_SUMMARY_SHA256
    ):
        raise ValueError("exploratory source is not the frozen completed rebuild")
    inputs = _validate_rebuild_inputs_without_sealed(inputs_manifest)
    inputs = {**inputs, "_self_snapshot": artifact_snapshot(inputs_manifest)}
    summary = _load_json(rebuild_summary, "rebuild final summary")
    if (
        set(summary)
        != {
            "schema", "terminal", "same_architecture_budget_seconds",
            "phases", "residual_fallback_exhausted", "next_scope",
        }
        or summary.get("schema") != rebuild.REBUILD_DECISION_SCHEMA
        or summary.get("terminal") != "no-development-qualified-candidate"
        or summary.get("same_architecture_budget_seconds")
        != rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS
        or summary.get("residual_fallback_exhausted") is not True
        or summary.get("next_scope")
        != "action-ranking-or-wider-network-requires-new-plan"
    ):
        raise ValueError("rebuild did not end with the frozen no-candidate result")
    status_path = rebuild_summary.parent / "status.json"
    status = _load_json(status_path, "rebuild completion status")
    if (
        artifact_snapshot(status_path)["sha256"] != REBUILD_STATUS_SHA256
        or
        status.get("schema")
        != "papersoccer.jacek-replay-rebuild-status.v1"
        or status.get("rebuild_id") != REBUILD_ID
        or status.get("phase") != "complete-no-candidate"
    ):
        raise ValueError("rebuild status is not complete-no-candidate")
    phases = summary.get("phases")
    if (
        not isinstance(phases, list)
        or [phase.get("phase") for phase in phases if isinstance(phase, dict)]
        != list(PHASE_ORDER)
    ):
        raise ValueError("rebuild phase transcript is incomplete or reordered")
    # Preflight every bound artifact before parsing any decision.  The generic
    # active-rebuild validator is deliberately not called here because it may
    # materialize reports and it traverses protected bank inputs.
    for record in _collect_snapshots(summary).values():
        if not _artifact_matches(record):
            raise ValueError("preserved rebuild evidence is missing or stale")
    full_records: list[dict] = []
    canonical_phase: dict | None = None
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("rebuild phase transcript is malformed")
        _verify_hashed_record(
            phase,
            schema="papersoccer.jacek-replay-rebuild-phase-screen.v1",
            label="rebuild phase screen",
        )
        candidates = phase.get("candidate_records")
        if not isinstance(candidates, list):
            raise ValueError("rebuild phase candidate roster is malformed")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("rebuild phase candidate is malformed")
            _verify_hashed_record(
                candidate,
                schema=rebuild.REBUILD_CANDIDATE_SCHEMA,
                label="rebuild candidate",
            )
        if phase.get("offline_eligible") != sum(
            candidate.get("offline_eligible") is True for candidate in candidates
        ):
            raise ValueError("rebuild offline eligibility count changed")
        if phase.get("selected_candidate_id") is not None or phase.get("qualified"):
            raise ValueError("rebuild transcript contradicts its no-candidate result")
        records = phase.get("full_records")
        if not isinstance(records, list):
            raise ValueError("rebuild phase full-screen record is malformed")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("rebuild full-screen record is malformed")
            _validate_embedded_full_screen(record)
        full_records.extend(records)
        if phase.get("phase") == "canonical-basins":
            canonical_phase = phase
    if not full_records or canonical_phase is None:
        raise ValueError("rebuild has no failed full-screen candidates")
    ranked = sorted(full_records, key=_near_pass_key)
    selected = ranked[0]
    if (
        selected.get("candidate_id") != DIVERSITY_CANDIDATE_ID
        or canonical_phase.get("short_ranked", [None])[0]
        != DIVERSITY_CANDIDATE_ID
        or canonical_phase.get("full_records", [None])[0] != selected
    ):
        raise ValueError("frozen closest near-pass candidate changed")
    decision = selected["full"]["decision"]
    if (
        decision.get("counts") != EXPECTED_NEAR_PASS_COUNTS
        or decision.get("errors")
        != [
            "matched primary strength gate failed",
            "incumbent primary strength gate failed",
        ]
        or decision.get("eligible_for_full") is not False
    ):
        raise ValueError("closest candidate no longer has the recorded gate failure")
    candidate_records = canonical_phase.get("candidate_records")
    candidates = [
        value for value in candidate_records
        if isinstance(value, dict)
        and value.get("candidate_id") == DIVERSITY_CANDIDATE_ID
    ] if isinstance(candidate_records, list) else []
    if len(candidates) != 1:
        raise ValueError("closest candidate training lineage is not unique")
    candidate = candidates[0]
    phase_screen_path = (
        rebuild_summary.parent
        / "canonical-basins/screening/phase-screen.json"
    )
    candidate_record_path = (
        rebuild_summary.parent
        / f"canonical-basins/candidates/{DIVERSITY_CANDIDATE_ID}.json"
    )
    if (
        artifact_snapshot(phase_screen_path)["sha256"]
        != CANONICAL_PHASE_SHA256
        or artifact_snapshot(candidate_record_path)["sha256"]
        != DIVERSITY_CANDIDATE_RECORD_SHA256
        or
        _load_json(phase_screen_path, "canonical phase screen")
        != canonical_phase
        or _load_json(candidate_record_path, "near-pass candidate record")
        != candidate
    ):
        raise ValueError("near-pass candidate or phase-screen artifact changed")
    selected_search = candidate.get("selected_search")
    runtime = selected_search.get("runtime") if isinstance(selected_search, dict) else None
    if (
        candidate.get("offline_eligible") is not True
        or not isinstance(runtime, dict)
        or selected["full"].get("candidate") != runtime
        or not _artifact_matches(runtime)
        or runtime.get("sha256") != DIVERSITY_ACTOR_SHA256
    ):
        raise ValueError("closest candidate runtime binding changed")
    reference_record = inputs.get("incumbent_runtime")
    if (
        not isinstance(reference_record, dict)
        or reference_record.get("sha256") != V6_EXPERIMENTAL_REFERENCE_SHA256
    ):
        raise ValueError("rebuild experimental-reference binding is missing")
    v6_summary = _validate_v6_actor(pathlib.Path(reference_record["path"]))
    body: dict[str, object] = {
        "schema": LINEAGE_SCHEMA,
        "campaign_id": EXPLORATORY_CAMPAIGN_ID,
        "rebuild_id": REBUILD_ID,
        "rebuild_inputs": artifact_snapshot(inputs_manifest),
        "rebuild_summary": artifact_snapshot(rebuild_summary),
        "rebuild_status": artifact_snapshot(status_path),
        "canonical_phase_screen": artifact_snapshot(phase_screen_path),
        "diversity_candidate_record": artifact_snapshot(candidate_record_path),
        "v6_summary": v6_summary,
        "frozen_v6_experimental_reference": reference_record,
        "diversity_actor": runtime,
        "diversity_candidate_id": DIVERSITY_CANDIDATE_ID,
        "selection_policy": (
            "best-worst-unchanged-development-gate-margin-then-"
            "canonical-huber-runtime-hash"
        ),
        "recorded_failed_decision": selected["full"],
        "development_gate_bypass_scope": "game-generation-diversity-only",
        "pilot_games": 2_000,
        "automatic_full_launch": False,
        "protected_test_opened_before_pilot_candidate_freeze": False,
        "protected_test_used_post_training_for_unchanged_retention_gate": True,
        "sealed_final_bank_opened": False,
        "blind_holdout_labels_opened": False,
        "canonical_incumbent_replaced": False,
        "rank4_replaced": False,
        "external_upload": False,
    }
    return _body_hashed(body)


def _opening_exclusions(inputs: Mapping[str, object]) -> list[pathlib.Path]:
    banks_snapshot = inputs.get("opening_banks")
    if not isinstance(banks_snapshot, dict):
        raise ValueError("rebuild opening-bank binding is missing")
    banks = _load_json(pathlib.Path(banks_snapshot["path"]), "rebuild opening banks")
    development = banks.get("development")
    excluded = banks.get("excluded_banks")
    final = banks.get("final")
    if (
        not isinstance(development, dict)
        or not isinstance(development.get("artifact"), dict)
        or not isinstance(excluded, list)
        or not isinstance(final, dict)
        or not isinstance(final.get("artifact"), dict)
    ):
        raise ValueError("rebuild opening-bank evidence is malformed")
    sealed_final = pathlib.Path(final["artifact"]["path"]).resolve()
    records = [*excluded, development["artifact"]]
    paths: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("rebuild opening exclusion is malformed")
        path = pathlib.Path(record["path"]).resolve()
        if not _artifact_matches(record):
            raise ValueError("rebuild opening exclusion binding is stale")
        if path not in seen:
            selfsearch._comparison_bank_states(path)
            paths.append(path)
            seen.add(path)
    if not paths:
        raise ValueError("exploratory pilot has no frozen opening exclusions")
    if sealed_final in paths:
        raise ValueError("sealed final bank entered exploratory exclusions")
    return paths


def _continuation_receipt(
    *, lineage_path: pathlib.Path, pilot: Mapping[str, object],
    output: pathlib.Path,
) -> dict[str, object]:
    decision_path = pathlib.Path(str(pilot.get("decision_path", ""))).resolve()
    runtime = pathlib.Path(str(pilot.get("search_runtime", ""))).resolve()
    manifest = pathlib.Path(str(pilot.get("search_manifest", ""))).resolve()
    search_manifests = [
        pathlib.Path(path).resolve()
        for path in pilot.get("search_new_manifests", [])
    ]
    rank4_manifests = [
        pathlib.Path(path).resolve()
        for path in pilot.get("rank4_new_manifests", [])
    ]
    if (
        pilot.get("decision", {}).get("eligible_for_full") is not True
        or len(search_manifests) != 3
        or len(rank4_manifests) != 3
    ):
        raise ValueError("pilot is not eligible for a separate full continuation")
    body: dict[str, object] = {
        "schema": CONTINUATION_SCHEMA,
        "source_campaign_id": EXPLORATORY_CAMPAIGN_ID,
        "pilot_campaign_id": EXPLORATORY_PILOT_CAMPAIGN_ID,
        "full_campaign_id": EXPLORATORY_FULL_CAMPAIGN_ID,
        "lineage": artifact_snapshot(lineage_path),
        "pilot_decision": artifact_snapshot(decision_path),
        "pilot_runtime": artifact_snapshot(runtime),
        "pilot_manifest": artifact_snapshot(manifest),
        "pilot_search_new_manifests": [
            artifact_snapshot(path) for path in search_manifests
        ],
        "pilot_rank4_new_manifests": [
            artifact_snapshot(path) for path in rank4_manifests
        ],
        "required_games": 10_000,
        "requires_explicit_separate_launch": True,
        "automatic_full_launch": False,
        "canonical_promotion_eligible": False,
        "external_upload": False,
        "replace_rank4": False,
    }
    receipt = _body_hashed(body)
    _write_exact_json(output, receipt, "full-continuation receipt")
    return receipt


def _collect_snapshots(value: object) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}

    def visit(current: object) -> None:
        if isinstance(current, dict):
            if (
                isinstance(current.get("path"), str)
                and isinstance(current.get("sha256"), str)
                and type(current.get("bytes")) is int
            ):
                record = dict(current)
                records[str(pathlib.Path(record["path"]).resolve())] = record
                return
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return records


def _phase_spec_record() -> dict[str, object]:
    if (
        sum(EXPLORATORY_PILOT_SPEC.quotas.values()) != 2_000
        or EXPLORATORY_PILOT_SPEC.quotas.get("incumbent-p1-vs-runner-up") != 100
        or EXPLORATORY_PILOT_SPEC.quotas.get("incumbent-p2-vs-runner-up") != 100
        or {
            key: value for key, value in EXPLORATORY_PILOT_SPEC.configuration.items()
            if key != "campaign_id"
        }
        != {
            key: value for key, value in selfsearch.PILOT_SPEC.configuration.items()
            if key != "campaign_id"
        }
        or EXPLORATORY_PILOT_SPEC.game_seed == selfsearch.PILOT_SPEC.game_seed
        or EXPLORATORY_PILOT_SPEC.opening_seed == selfsearch.PILOT_SPEC.opening_seed
    ):
        raise ValueError("exploratory pilot is not the fixed fresh 2,000-game profile")
    return dataclasses.asdict(EXPLORATORY_PILOT_SPEC)


def freeze_launch(
    *, repository: pathlib.Path, expected_commit: str,
    inputs_manifest: pathlib.Path, rebuild_summary: pathlib.Path,
    canonical_campaign: pathlib.Path, output: pathlib.Path,
    executables: selfsearch.CampaignExecutables, build_manifest: pathlib.Path,
) -> dict[str, object]:
    """Perform all Git/provenance checks before a Git-free persistent launch."""

    repository = repository.resolve()
    inputs_manifest = inputs_manifest.resolve()
    rebuild_summary = rebuild_summary.resolve()
    canonical_campaign = canonical_campaign.resolve()
    output = output.resolve()
    if output.name != EXPLORATORY_CAMPAIGN_ID:
        raise ValueError(
            f"exploratory output directory must be named {EXPLORATORY_CAMPAIGN_ID}"
        )
    output.mkdir(parents=True, exist_ok=True)
    executables = executables.resolved()
    executables.validate()
    repository_record = selfsearch._repository_record(repository, expected_commit)
    build_manifest = build_manifest.resolve()
    build_record = selfsearch.validate_build_manifest(
        build_manifest,
        repository=repository,
        expected_commit=expected_commit,
        executables=executables,
    )
    lineage = validate_diversity_lineage(
        inputs_manifest=inputs_manifest, rebuild_summary=rebuild_summary
    )
    inputs = _validate_rebuild_inputs_without_sealed(inputs_manifest)
    roots_tsv = (canonical_campaign / "round-2/teacher-input.tsv").resolve()
    roots_manifest = (canonical_campaign / "round-2/replay-roots.json").resolve()
    splits = selfsearch._canonical_split_manifests(canonical_campaign)
    canonical_record = selfsearch._validate_canonical_campaign_inputs(
        canonical_campaign,
        roots_tsv=roots_tsv,
        roots_manifest=roots_manifest,
        anchor_train=splits["train"],
        anchor_validation=splits["validation"],
        anchor_test=splits["test"],
    )
    canonical_prior = [
        artifact_snapshot(splits[split][round_index])
        for round_index in range(3)
        for split in ("train", "validation", "test")
    ]
    exclusions = _opening_exclusions(inputs)
    frozen_environment = selfsearch.environment_identity()
    python_runtime = artifact_snapshot(pathlib.Path(sys.executable).resolve())
    banks_record = _load_json(
        pathlib.Path(inputs["opening_banks"]["path"]), "rebuild banks"
    )
    sealed_record = banks_record["final"]["artifact"]
    sealed_path = pathlib.Path(sealed_record["path"]).resolve()
    if sealed_path in exclusions:
        raise ValueError("sealed final opening bank entered exploratory inputs")

    lineage_path = output / "exploratory-lineage.json"
    frozen_lineage = dict(lineage)
    frozen_lineage.pop("body_sha256", None)
    frozen_lineage.update(
        {
            "repository": repository_record,
            "executables": executables.snapshots(),
            "release_build": artifact_snapshot(build_manifest),
            "canonical_ancestry": canonical_record,
            "opening_exclusions": [artifact_snapshot(path) for path in exclusions],
            "sealed_final_untouched": {
                "path_recorded": str(sealed_path),
                "sha256_recorded": sealed_record["sha256"],
                "bytes_recorded": sealed_record["bytes"],
                "opened": False,
                "used_as_exclusion": False,
            },
            "pilot_specification": _phase_spec_record(),
            "environment": frozen_environment,
            "python_runtime": python_runtime,
            "wrapper": artifact_snapshot(pathlib.Path(__file__).resolve()),
        }
    )
    frozen_lineage = _body_hashed(frozen_lineage)
    _write_exact_json(lineage_path, frozen_lineage, "exploratory lineage")

    shard_npz = []
    for paths in splits.values():
        for path in paths:
            manifest = _load_json(path, "canonical shard manifest")
            shard_npz.append(
                artifact_snapshot(path.parent / str(manifest.get("npz", "")))
            )
    launch_body: dict[str, object] = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": EXPLORATORY_CAMPAIGN_ID,
        "output_directory": str(output),
        "expected_commit": expected_commit,
        "repository": repository_record,
        "lineage": artifact_snapshot(lineage_path),
        "lineage_record": frozen_lineage,
        "executables": executables.snapshots(),
        "release_build": artifact_snapshot(build_manifest),
        "release_build_record": build_record,
        "canonical_campaign": str(canonical_campaign),
        "canonical_ancestry": canonical_record,
        "roots": {
            "tsv": artifact_snapshot(roots_tsv),
            "manifest": artifact_snapshot(roots_manifest),
        },
        "splits": {
            split: [artifact_snapshot(path) for path in paths]
            for split, paths in splits.items()
        },
        "canonical_prior_manifests": canonical_prior,
        "canonical_npz": shard_npz,
        "opening_exclusions": [artifact_snapshot(path) for path in exclusions],
        "sealed_final_untouched": {
            "path_recorded": str(sealed_path),
            "sha256_recorded": sealed_record["sha256"],
            "bytes_recorded": sealed_record["bytes"],
            "opened": False,
            "used_as_exclusion": False,
        },
        "pilot_specification": _phase_spec_record(),
        "environment": frozen_environment,
        "python_runtime": python_runtime,
        "pilot_games": 2_000,
        "automatic_full_launch": False,
        "full_directory_creation": False,
        "canonical_incumbent_replaced": False,
        "canonical_promotion_eligible": False,
        "external_upload": False,
        "rank4_replaced": False,
        "wrapper": artifact_snapshot(pathlib.Path(__file__).resolve()),
    }
    guarded = _collect_snapshots(launch_body)
    launch_body["artifact_guard"] = [guarded[path] for path in sorted(guarded)]
    launch = _body_hashed(launch_body)
    launch_path = output / "exploratory-launch.json"
    _write_exact_json(launch_path, launch, "exploratory launch receipt")
    return launch


def validate_launch(path: pathlib.Path) -> dict[str, object]:
    """Validate a launch receipt using file hashes only; never inspect Git."""

    path = path.resolve()
    launch = _load_json(path, "exploratory launch receipt")
    body = dict(launch)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != LAUNCH_SCHEMA
        or body.get("campaign_id") != EXPLORATORY_CAMPAIGN_ID
        or body.get("output_directory") != str(path.parent.resolve())
        or body.get("pilot_specification") != _phase_spec_record()
        or body.get("pilot_games") != 2_000
        or body.get("environment") != selfsearch.environment_identity()
        or not _artifact_matches(body.get("python_runtime"))
        or body.get("python_runtime", {}).get("path")
        != str(pathlib.Path(sys.executable).resolve())
        or body.get("automatic_full_launch") is not False
        or body.get("full_directory_creation") is not False
        or body.get("sealed_final_untouched", {}).get("opened") is not False
        or body.get("sealed_final_untouched", {}).get("used_as_exclusion") is not False
        or body.get("canonical_incumbent_replaced") is not False
        or body.get("canonical_promotion_eligible") is not False
        or body.get("external_upload") is not False
        or body.get("rank4_replaced") is not False
        or claimed != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError("exploratory launch receipt is invalid")
    lineage_snapshot = body.get("lineage")
    lineage_record = body.get("lineage_record")
    if (
        not isinstance(lineage_snapshot, dict)
        or not isinstance(lineage_record, dict)
        or not _artifact_matches(lineage_snapshot)
        or _load_json(
            pathlib.Path(lineage_snapshot["path"]), "exploratory lineage"
        ) != lineage_record
        or lineage_record.get("frozen_v6_experimental_reference", {}).get("sha256")
        != V6_EXPERIMENTAL_REFERENCE_SHA256
        or lineage_record.get("diversity_actor", {}).get("sha256")
        != DIVERSITY_ACTOR_SHA256
    ):
        raise ValueError("exploratory launch lineage is stale")
    build_snapshot = body.get("release_build")
    build_record = body.get("release_build_record")
    if (
        not isinstance(build_snapshot, dict)
        or not isinstance(build_record, dict)
        or not _artifact_matches(build_snapshot)
        or _load_json(
            pathlib.Path(build_snapshot["path"]), "frozen Release build"
        ) != build_record
    ):
        raise ValueError("exploratory Release build receipt is stale")
    guard = body.get("artifact_guard")
    if not isinstance(guard, list) or not guard:
        raise ValueError("exploratory launch has no artifact guard")
    guarded = _collect_snapshots({key: value for key, value in body.items()
                                  if key != "artifact_guard"})
    if guard != [guarded[item] for item in sorted(guarded)]:
        raise ValueError("exploratory launch artifact guard is incomplete")
    for record in guard:
        if not _artifact_matches(record):
            raise ValueError("exploratory launch artifact changed")
    return launch


def _copy_frozen(source: pathlib.Path, target: pathlib.Path, expected: Mapping[str, object]) -> None:
    if target.exists():
        current = artifact_snapshot(target)
        if (
            current.get("sha256") != expected.get("sha256")
            or current.get("bytes") != expected.get("bytes")
        ):
            raise ValueError("existing exploratory actor snapshot is stale")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    selfsearch._copy_atomic(source, target)
    current = artifact_snapshot(target)
    if (
        current.get("sha256") != expected.get("sha256")
        or current.get("bytes") != expected.get("bytes")
    ):
        target.unlink(missing_ok=True)
        raise ValueError("exploratory actor snapshot changed during copy")


def run_exploratory_pilot(
    *, launch_receipt: pathlib.Path, output: pathlib.Path,
    resume: bool, skip_power_check: bool,
) -> dict[str, object]:
    """Run exactly one strict 2,000-game pilot with no Git access or full launch."""

    launch_receipt = launch_receipt.resolve()
    output = output.resolve()
    if (
        output.name != EXPLORATORY_CAMPAIGN_ID
        or launch_receipt.parent != output
    ):
        raise ValueError("exploratory run does not match its frozen output directory")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "supervisor-status.json"
    lock_path = output / "supervisor.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("exploratory pilot supervisor is already running") from error
        selfsearch._CAMPAIGN_LOCK_FD = lock.fileno()
        try:
            launch = validate_launch(launch_receipt)
            launch_snapshot = artifact_snapshot(launch_receipt)
            lineage = launch["lineage_record"]
            executable_records = launch["executables"]
            executables = selfsearch.CampaignExecutables(
                **{
                    name: pathlib.Path(record["path"])
                    for name, record in executable_records.items()
                }
            ).resolved()
            executables.validate()
            build_manifest = pathlib.Path(launch["release_build"]["path"])
            build_record = launch["release_build_record"]
            roots_tsv = pathlib.Path(launch["roots"]["tsv"]["path"])
            roots_manifest = pathlib.Path(launch["roots"]["manifest"]["path"])
            splits = {
                split: tuple(pathlib.Path(record["path"]) for record in records)
                for split, records in launch["splits"].items()
            }
            canonical_prior = tuple(
                pathlib.Path(record["path"])
                for record in launch["canonical_prior_manifests"]
            )
            exclusions = [
                pathlib.Path(record["path"])
                for record in launch["opening_exclusions"]
            ]
            health = selfsearch.validate_host_health(
                output, skip_power=skip_power_check
            )
            reference_source = pathlib.Path(
                lineage["frozen_v6_experimental_reference"]["path"]
            )
            diversity_source = pathlib.Path(lineage["diversity_actor"]["path"])
            reference = output / "inputs/v6-experimental-reference.runtime"
            diversity = output / "inputs/near-pass-diversity.runtime"
            _copy_frozen(
                reference_source, reference,
                lineage["frozen_v6_experimental_reference"],
            )
            _copy_frozen(diversity_source, diversity, lineage["diversity_actor"])
            if (output / "full").exists():
                raise ValueError("pilot-only workflow found a preexisting full directory")
            guarded = {
                record["path"]: record for record in launch["artifact_guard"]
            }

            def producer_guard() -> None:
                # Deliberately hash-only: persistent launches must never touch
                # .git or execute Git after the interactive freeze step.
                if artifact_snapshot(launch_receipt) != launch_snapshot:
                    raise ValueError("exploratory launch receipt changed")
                for record in guarded.values():
                    if not _artifact_matches(record):
                        raise ValueError("exploratory frozen input changed")
                if (
                    artifact_snapshot(reference)["sha256"]
                    != V6_EXPERIMENTAL_REFERENCE_SHA256
                ):
                    raise ValueError("v6 experimental reference changed")
                if artifact_snapshot(diversity)["sha256"] != DIVERSITY_ACTOR_SHA256:
                    raise ValueError("near-pass diversity actor changed")

            producer_guard()
            _status(
                status_path,
                "running-pilot",
                games=2_000,
                experimental_reference_sha256=V6_EXPERIMENTAL_REFERENCE_SHA256,
                diversity_actor_sha256=DIVERSITY_ACTOR_SHA256,
                automatic_full_launch=False,
                launch_health=health,
            )
            pilot = selfsearch.run_phase(
                spec=EXPLORATORY_PILOT_SPEC,
                output=output / "pilot",
                resume=resume,
                roots_tsv=roots_tsv,
                roots_manifest=roots_manifest,
                actor=reference,
                diversity=diversity,
                executables=executables,
                anchor_train_manifests=splits["train"],
                retention_validation_manifests=splits["validation"],
                anchor_validation_manifests=splits["test"],
                canonical_prior_manifests=canonical_prior,
                opening_exclusions=exclusions,
                producer_guard=producer_guard,
                build_manifest=build_manifest,
                source_identities=build_record["source_identities"],
            )
            producer_guard()
            eligible = pilot.get("decision", {}).get("eligible_for_full") is True
            continuation_path = output / "full-continuation-eligible.json"
            continuation = (
                _continuation_receipt(
                    lineage_path=pathlib.Path(launch["lineage"]["path"]),
                    pilot=pilot,
                    output=continuation_path,
                )
                if eligible else None
            )
            if not eligible and continuation_path.exists():
                raise ValueError("rejected pilot has a stale continuation receipt")
            if (output / "full").exists():
                raise ValueError("pilot-only workflow must not create a full directory")
            summary: dict[str, object] = {
                "schema": SUMMARY_SCHEMA,
                "campaign_id": EXPLORATORY_CAMPAIGN_ID,
                "classification": "local-exploratory-data-generation",
                "terminal": (
                    "pilot-pass-continuation-eligible"
                    if eligible else "pilot-rejected"
                ),
                "launch": launch_snapshot,
                "lineage": launch["lineage"],
                "experimental_reference_actor": (
                    lineage["frozen_v6_experimental_reference"]
                ),
                "diversity_actor": lineage["diversity_actor"],
                "pilot_actor_snapshot": artifact_snapshot(reference),
                "pilot_diversity_snapshot": artifact_snapshot(diversity),
                "pilot": pilot,
                "full_continuation_eligible": eligible,
                "full_continuation_receipt": (
                    artifact_snapshot(continuation_path)
                    if continuation is not None else None
                ),
                "full_started": False,
                "protected_test_opened_before_pilot_candidate_freeze": False,
                "protected_test_used_post_training_for_unchanged_retention_gate": True,
                "sealed_final_bank_opened": False,
                "blind_holdout_labels_opened": False,
                "canonical_incumbent_replaced": False,
                "canonical_promotion_eligible": False,
                "publication": None,
                "external_upload": False,
                "rank4_replaced": False,
            }
            summary_path = output / "final-summary.json"
            _write_exact_json(summary_path, summary, "exploratory final summary")
            _status(
                status_path,
                summary["terminal"],
                summary=str(summary_path),
                summary_sha256=artifact_snapshot(summary_path)["sha256"],
                full_started=False,
            )
            return summary
        except Exception as error:
            _status(
                status_path, "failed", error=str(error),
                traceback=traceback.format_exc(), full_started=False,
            )
            raise
        finally:
            selfsearch._CAMPAIGN_LOCK_FD = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-launch", help="validate Git/provenance and write a Git-free launch receipt"
    )
    freeze.add_argument("--repository", type=pathlib.Path, required=True)
    freeze.add_argument("--expected-commit", required=True)
    freeze.add_argument("--rebuild-inputs", type=pathlib.Path, required=True)
    freeze.add_argument("--rebuild-summary", type=pathlib.Path, required=True)
    freeze.add_argument("--canonical-campaign", type=pathlib.Path, required=True)
    freeze.add_argument("--output-directory", type=pathlib.Path, required=True)
    freeze.add_argument("--continuation-generator", type=pathlib.Path, required=True)
    freeze.add_argument("--search-teacher", type=pathlib.Path, required=True)
    freeze.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    freeze.add_argument("--comparison", type=pathlib.Path, required=True)
    freeze.add_argument("--pack-tool", type=pathlib.Path, required=True)
    freeze.add_argument("--trainer", type=pathlib.Path, required=True)
    freeze.add_argument("--build-manifest", type=pathlib.Path, required=True)
    run = commands.add_parser(
        "run", help="run the frozen pilot without reading Git metadata"
    )
    run.add_argument("--launch-receipt", type=pathlib.Path, required=True)
    run.add_argument("--output-directory", type=pathlib.Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--skip-power-check", action="store_true",
        help="tests/development only; the persistent local launch must not use this",
    )
    arguments = parser.parse_args()
    if arguments.command == "freeze-launch":
        if (
            len(arguments.expected_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in arguments.expected_commit
            )
        ):
            parser.error("expected commit must be a lowercase 40-character hash")
        freeze_launch(
            repository=arguments.repository,
            expected_commit=arguments.expected_commit,
            inputs_manifest=arguments.rebuild_inputs,
            rebuild_summary=arguments.rebuild_summary,
            canonical_campaign=arguments.canonical_campaign,
            output=arguments.output_directory,
            executables=selfsearch.CampaignExecutables(
                continuation_generator=arguments.continuation_generator,
                search_teacher=arguments.search_teacher,
                rank4_teacher=arguments.rank4_teacher,
                comparison=arguments.comparison,
                pack_tool=arguments.pack_tool,
                trainer=arguments.trainer,
            ),
            build_manifest=arguments.build_manifest,
        )
    else:
        run_exploratory_pilot(
            launch_receipt=arguments.launch_receipt,
            output=arguments.output_directory,
            resume=arguments.resume,
            skip_power_check=arguments.skip_power_check,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
