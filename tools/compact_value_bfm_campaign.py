#!/usr/bin/env python3
"""Immutable development and one-iteration governance for compact_value_bfm.

This module validates already-produced aggregate reports.  It never launches
matches, opens protected tests, trains a model, or operates CodinGame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def _qualification_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_qualification.py")
    name = "compact_value_bfm_qualification_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qualification primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _workflow_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_workflow.py")
    name = "compact_value_bfm_campaign_workflow_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workflow primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _trainer_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_train.py")
    name = "compact_value_bfm_campaign_trainer_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _preflight_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_preflight.py")
    name = "compact_value_bfm_campaign_preflight_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preflight primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _iteration_module():
    path = pathlib.Path(__file__).with_name("compact_value_bfm_iteration.py")
    name = "compact_value_bfm_campaign_iteration_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load iteration primitives: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _qualification_module()
CampaignError = base.QualificationError

NAMESPACE = "compact_value_bfm"
DEVELOPMENT_INPUT_SCHEMA = "papersoccer.compact-value-bfm.development-input.v1"
SELECTION_SCHEMA = "papersoccer.compact-value-bfm.immutable-selection.v1"
ITERATION_SELECTION_SCHEMA = "papersoccer.compact-value-bfm.iteration-selection.v1"
POST_ITERATION_HANDOFF_SCHEMA = (
    "papersoccer.compact-value-bfm.post-iteration-development-handoff.v1"
)
DEVELOPMENT_RUN_SCHEMA = "papersoccer.compact-value-bfm-development-run.v1"
TEST_AUTH_SCHEMA = "papersoccer.compact-value-bfm.protected-test-authorization.v1"
OFFLINE_FAMILY_FAILURE_SCHEMA = (
    "papersoccer.compact-value-bfm.offline-family-failure.v1"
)
OPERATIONAL_SAFE_ACTOR_SCHEMA = (
    "papersoccer.compact-value-bfm.operational-safe-actor.v1"
)
OPERATIONAL_BUILD_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.operational-build-evidence.v1"
)
OPERATIONAL_PROTOCOL_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.operational-protocol-evidence.v1"
)
ITERATION_AUTH_SCHEMA = "papersoccer.compact-value-bfm.iteration-authorization.v1"
ITERATION_EVENT_SCHEMA = "papersoccer.compact-value-bfm.iteration-event.v1"
POST_ITERATION_FAILURE_SCHEMA = (
    "papersoccer.compact-value-bfm.post-iteration-failure.v1"
)
FAMILY_EXHAUSTED_SCHEMA = "papersoccer.compact-value-bfm.family-exhausted.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"

ARCHITECTURES = {
    "6301-8-8-1": 0,
    "6301-8-16-1": 1,
    "6301-12-8-1": 2,
}
PRIMARY_ARCHITECTURE = "6301-8-8-1"
SOURCE_NEUTRAL_ARCHITECTURE = "6301-8-16-1"
CAPACITY_ARCHITECTURE = "6301-12-8-1"
TARGETS = ("search-target", "teacher-assisted")
CONTROL_TARGET = "rank4-target-control"
FAMILY_DEVELOPMENT_MODE = "frozen-family"
POST_ITERATION_DEVELOPMENT_MODE = "post-iteration"
POST_ITERATION_CANDIDATE_ID = "post-iteration-search-target"
POST_ITERATION_HANDOFF_STATUS = "offline-qualified-awaiting-development"

WORKFLOW_ARCHITECTURES = {
    "compact-8x8": PRIMARY_ARCHITECTURE,
    "source-neutral-8x16": SOURCE_NEUTRAL_ARCHITECTURE,
    "capacity-12x8": CAPACITY_ARCHITECTURE,
}
WORKFLOW_DEPLOYABLE_ARMS = tuple(
    (architecture, target)
    for architecture in WORKFLOW_ARCHITECTURES
    for target in TARGETS
)
WORKFLOW_CONTROL = ("compact-8x8", "rank4-control")
WORKFLOW_CAMPAIGN_ORDER = tuple(
    f"{architecture}--{target}"
    for architecture, target in (*WORKFLOW_DEPLOYABLE_ARMS, WORKFLOW_CONTROL)
)
POST_ITERATION_FAILURE_STAGES = {
    "offline-evaluator", "development-gate", "protected-test", "final-gate",
}

TUPLE_ROSTER = (
    ("0.65", "0.5", "1"),
    ("0.80", "0.5", "1"),
    ("0.95", "0.5", "1"),
    ("1.10", "0.5", "1"),
    ("0.95", "0.25", "1"),
    ("0.95", "0.75", "1"),
    ("0.95", "0.5", "0.5"),
    ("0.95", "0.5", "0"),
)
DEFAULT_TUPLE = ("0.95", "0.5", "1")
PROFILE_ROSTER = {
    "light": {"root_partial_paths": 2000, "nonroot_partial_paths": 256,
              "nodes": 60000},
    "default": {"root_partial_paths": 4000, "nonroot_partial_paths": 512,
                "nodes": 80000},
    "heavy": {"root_partial_paths": 8000, "nonroot_partial_paths": 512,
              "nodes": 120000},
}
DEFAULT_PROFILE = "default"
SHA256_RE = re.compile(r"[0-9a-f]{64}")

ITERATION_SPEC = {
    "games": {
        "student_self_play": 5000,
        "rank4_candidate_as_0": 1000,
        "rank4_candidate_as_1": 1000,
        "jacek_nn_candidate_as_0": 1000,
        "jacek_nn_candidate_as_1": 1000,
        "previous_compact_candidate_as_0": 500,
        "previous_compact_candidate_as_1": 500,
    },
    "total_games": 10000,
    "positions_per_game": 20,
    "workers": 10,
    "fixed_work": True,
    "deep_relabel_fraction": 0.25,
    "target_semantics": "75-percent-fixed-work-search-25-percent-terminal-outcome",
    "maximum_sample_scaled_learning_rate": 0.00006,
}


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CampaignError(f"{field} must be a lowercase SHA-256")
    return value


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{field} must be finite")
    return result


def tuple_id(value: Sequence[str]) -> str:
    if tuple(value) not in TUPLE_ROSTER:
        raise CampaignError("search tuple is outside the exact roster")
    return "c" + value[0] + "-f" + value[1] + "-l" + value[2]


def transcript_primitive_plies(value: Any) -> int:
    """Return the physical-edge count in a slash-separated turn transcript."""

    if not isinstance(value, str) or not value or "/" not in value:
        raise CampaignError("opening transcript must be a nonempty string")
    turns = value.split("/")
    if any(not turn or re.fullmatch(r"[0-7]+", turn) is None for turn in turns):
        raise CampaignError("opening transcript is not slash-separated complete turns")
    return sum(len(turn) for turn in turns)


def validate_complete_turn_transcript(
    value: Any, *, minimum_primitive_plies: int = 12,
) -> str:
    """Require complete-turn form and the frozen minimum physical-ply depth."""

    if transcript_primitive_plies(value) < minimum_primitive_plies:
        raise CampaignError(
            "opening transcript is shallower than the minimum physical-ply depth"
        )
    return value


def _bank(value: Any, *, pairs: int, label: str) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or set(value) != {
                "bank_id", "pairs", "fingerprints", "transcripts",
                "primitive_ply_counts",
            }):
        raise CampaignError(f"{label} bank binding is malformed")
    if value["pairs"] != pairs or not isinstance(value["bank_id"], str) or not value["bank_id"]:
        raise CampaignError(f"{label} bank pair count or id is invalid")
    fingerprints = value["fingerprints"]
    if (not isinstance(fingerprints, list) or len(fingerprints) != pairs
            or fingerprints != sorted(set(fingerprints))
            or any(SHA256_RE.fullmatch(str(item)) is None for item in fingerprints)):
        raise CampaignError(f"{label} bank fingerprints are not exact and unique")
    transcripts = value["transcripts"]
    counts = value["primitive_ply_counts"]
    if (not isinstance(transcripts, list) or len(transcripts) != pairs
            or not isinstance(counts, list) or len(counts) != pairs):
        raise CampaignError(f"{label} bank transcript count is invalid")
    for index, transcript in enumerate(transcripts):
        validate_complete_turn_transcript(transcript)
        if counts[index] != transcript_primitive_plies(transcript):
            raise CampaignError(f"{label} bank primitive-ply count is stale")
    return dict(value)


def _metric(value: Any, *, pairs: int, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} metric is not an object")
    for field in ("candidate_id", "wins", "color_wins", "failures", "latency_ms"):
        if field not in value:
            raise CampaignError(f"{label} metric omits {field}")
    if value.get("pairs") != pairs or value.get("games") != pairs * 2:
        raise CampaignError(f"{label} metric has the wrong pair/game count")
    wins = _int(value["wins"], f"{label} wins")
    if wins > pairs * 2:
        raise CampaignError(f"{label} wins exceed games")
    colors = value["color_wins"]
    if not isinstance(colors, dict) or set(colors) != {"0", "1"}:
        raise CampaignError(f"{label} color wins are malformed")
    for color in ("0", "1"):
        if _int(colors[color], f"{label} color {color} wins") > pairs:
            raise CampaignError(f"{label} color wins exceed games")
    _int(value["failures"], f"{label} failures")
    latency = _finite(value["latency_ms"], f"{label} latency")
    if latency < 0:
        raise CampaignError(f"{label} latency must be nonnegative")
    return dict(value)


def _rank_key(metric: Mapping[str, Any], architecture: str) -> tuple[Any, ...]:
    return (
        -metric["wins"],
        -min(metric["color_wins"]["0"], metric["color_wins"]["1"]),
        float(metric["latency_ms"]),
        ARCHITECTURES[architecture],
        str(metric["candidate_id"]),
    )


def _validate_model_screen(
    rows: Any, eligible_architectures: Any,
    eligible_model_arms: Any = None,
    *, development_mode: str = FAMILY_DEVELOPMENT_MODE,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    post_iteration = development_mode == POST_ITERATION_DEVELOPMENT_MODE
    if development_mode not in {
        FAMILY_DEVELOPMENT_MODE, POST_ITERATION_DEVELOPMENT_MODE,
    }:
        raise CampaignError("development mode is invalid")
    if post_iteration:
        if (
            not isinstance(eligible_architectures, list)
            or len(eligible_architectures) != 1
            or eligible_architectures[0] not in ARCHITECTURES
        ):
            raise CampaignError(
                "post-iteration development requires exactly one architecture"
            )
    elif (not isinstance(eligible_architectures, list)
          or eligible_architectures not in (
              [PRIMARY_ARCHITECTURE, SOURCE_NEUTRAL_ARCHITECTURE],
              [PRIMARY_ARCHITECTURE, SOURCE_NEUTRAL_ARCHITECTURE,
               CAPACITY_ARCHITECTURE],
          )):
        raise CampaignError("eligible architecture roster is invalid")
    if not isinstance(rows, list):
        raise CampaignError("model screen rows must be a list")
    normalized = []
    if eligible_model_arms is None:
        if post_iteration:
            raise CampaignError(
                "post-iteration development requires its exact single model arm"
            )
        expected = {
            (architecture, target)
            for architecture in eligible_architectures for target in TARGETS
        }
    else:
        if (
            not isinstance(eligible_model_arms, list)
            or any(
                not isinstance(item, list) or len(item) != 2
                for item in eligible_model_arms
            )
        ):
            raise CampaignError("eligible model/target arm roster is malformed")
        expected = {tuple(item) for item in eligible_model_arms}
        if (
            len(expected) != len(eligible_model_arms)
            or any(
                architecture not in eligible_architectures or target not in TARGETS
                for architecture, target in expected
            )
            or {architecture for architecture, _target in expected}
            != set(eligible_architectures)
        ):
            raise CampaignError("eligible model/target arm roster is inconsistent")
        if post_iteration and expected != {
            (eligible_architectures[0], "search-target")
        }:
            raise CampaignError(
                "post-iteration development model arm is not the exact Search candidate"
            )
    observed: set[tuple[str, str]] = set()
    controls = 0
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _metric(raw, pairs=100, label=f"model screen {index}")
        architecture = row.get("architecture")
        target = row.get("target")
        source_bytes = _int(row.get("source_bytes"), "model source bytes", 1)
        _sha(row.get("artifact_sha256"), "model artifact SHA-256")
        if architecture not in ARCHITECTURES:
            raise CampaignError("model architecture is unknown")
        if target == CONTROL_TARGET:
            controls += 1
            if architecture != PRIMARY_ARCHITECTURE or row.get("deployment_eligible") is not False:
                raise CampaignError("matched Rank-4 control is not exact/nondeployable")
        else:
            key = (architecture, target)
            if key not in expected or key in observed or row.get("deployment_eligible") is not True:
                raise CampaignError("deployable model arm roster is not exact")
            if source_bytes > 95_000:
                raise CampaignError("deployable model source exceeds 95,000 bytes")
            observed.add(key)
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_id:
            raise CampaignError("model candidate IDs are invalid/repeated")
        by_id[candidate_id] = row
        normalized.append(row)
    if observed != expected or controls != 1:
        raise CampaignError("model screen lacks an exact arm/control roster")
    if post_iteration and (
        len(normalized) != 2
        or POST_ITERATION_CANDIDATE_ID not in by_id
        or "rank4-control" not in by_id
    ):
        raise CampaignError(
            "post-iteration model screen must contain one candidate and Rank-4 control"
        )
    deployable = [row for row in normalized
                  if row["target"] in TARGETS and row["failures"] == 0]
    minimum_models = 1 if post_iteration else 3
    if len(deployable) < minimum_models:
        raise CampaignError(
            "post-iteration model is not failure-free"
            if post_iteration
            else "fewer than three failure-free model arms remain"
        )
    retain = 1 if post_iteration else 3
    top = sorted(
        deployable, key=lambda row: _rank_key(row, row["architecture"])
    )[:retain]
    return top, by_id


def _validate_exact_tuple_screen(
    rows: Any, retained: Sequence[Mapping[str, Any]], by_model: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise CampaignError("tuple screen rows must be a list")
    expected = {(model["candidate_id"], tuple_value)
                for model in retained for tuple_value in TUPLE_ROSTER}
    observed = set()
    normalized = []
    for index, raw in enumerate(rows):
        row = _metric(raw, pairs=100, label=f"tuple screen {index}")
        model_id = row.get("model_id")
        raw_tuple = row.get("tuple")
        if not isinstance(raw_tuple, list):
            raise CampaignError("tuple screen tuple must be a list")
        tuple_value = tuple(str(item) for item in raw_tuple)
        key = (model_id, tuple_value)
        if key not in expected or key in observed:
            raise CampaignError("tuple screen roster is missing/repeated/foreign")
        expected_id = f"{model_id}:{tuple_id(tuple_value)}"
        if row["candidate_id"] != expected_id:
            raise CampaignError("tuple screen candidate id is not canonical")
        observed.add(key)
        normalized.append(row)
    if observed != expected:
        raise CampaignError("tuple screen does not cover the exact tuple roster")
    failure_free = [row for row in normalized if row["failures"] == 0]
    if len(failure_free) < 2:
        raise CampaignError("tuple screen has fewer than two failure-free candidates")
    return sorted(
        failure_free,
        key=lambda row: _rank_key(row, by_model[row["model_id"]]["architecture"]),
    )


def _confirmation_choice(
    rows: Any, *, pairs: int, carried_ids: Sequence[str], default_id: str,
    architecture_by_id: Mapping[str, str], label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(rows, list):
        raise CampaignError(f"{label} confirmation rows must be a list")
    expected = set(carried_ids)
    if set(row.get("candidate_id") for row in rows if isinstance(row, dict)) != expected:
        raise CampaignError(f"{label} confirmation does not match best-two-plus-default")
    normalized = []
    for index, raw in enumerate(rows):
        row = _metric(raw, pairs=pairs, label=f"{label} confirmation {index}")
        lower = _finite(row.get("paired_bootstrap_lower_95"), f"{label} bootstrap lower")
        row["paired_bootstrap_lower_95"] = lower
        normalized.append(row)
    by_id = {row["candidate_id"]: row for row in normalized}
    if default_id not in by_id or by_id[default_id]["failures"] != 0:
        raise CampaignError(f"{label} default did not complete without failures")
    default = by_id[default_id]
    eligible = [default]
    for row in normalized:
        if row["candidate_id"] == default_id:
            continue
        no_regression = (
            row["wins"] >= default["wins"]
            and row["color_wins"]["0"] >= default["color_wins"]["0"]
            and row["color_wins"]["1"] >= default["color_wins"]["1"]
        )
        if (row["failures"] == 0 and no_regression
                and row["paired_bootstrap_lower_95"] > 0.0):
            eligible.append(row)
    chosen = sorted(
        eligible,
        key=lambda row: _rank_key(row, architecture_by_id[row["candidate_id"]]),
    )[0]
    return chosen, normalized


def _validate_profiles(
    rows: Any, *, pairs: int, label: str,
    expected_profiles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    expected = set(PROFILE_ROSTER if expected_profiles is None else expected_profiles)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise CampaignError(f"{label} does not contain the expected work profiles")
    normalized = []
    seen = set()
    for index, raw in enumerate(rows):
        row = _metric(raw, pairs=pairs, label=f"{label} {index}")
        profile = row.get("profile")
        if (profile not in expected or profile in seen
                or row.get("work") != PROFILE_ROSTER[profile]
                or row["candidate_id"] != profile):
            raise CampaignError(f"{label} profile roster/configuration is invalid")
        seen.add(profile)
        normalized.append(row)
    if seen != expected:
        raise CampaignError(f"{label} profile roster is incomplete")
    return normalized


def validate_development_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != DEVELOPMENT_INPUT_SCHEMA
            or payload.get("namespace") != NAMESPACE):
        raise CampaignError("development input schema/namespace is invalid")
    development_mode = payload.get(
        "development_mode", FAMILY_DEVELOPMENT_MODE,
    )
    if development_mode not in {
        FAMILY_DEVELOPMENT_MODE, POST_ITERATION_DEVELOPMENT_MODE,
    }:
        raise CampaignError("development input mode is invalid")
    if development_mode == POST_ITERATION_DEVELOPMENT_MODE:
        handoff = payload.get("post_iteration_handoff")
        if not isinstance(handoff, dict) or set(handoff) != {
            "path", "sha256", "body_sha256",
        }:
            raise CampaignError("post-iteration handoff reference is missing")
        details = validate_post_iteration_handoff(
            pathlib.Path(str(handoff.get("path", "")))
        )
        expected_handoff = {
            "path": str(details["handoff_path"]),
            "sha256": base.sha256_file(details["handoff_path"]),
            "body_sha256": details["handoff"]["body_sha256"],
        }
        if handoff != expected_handoff:
            raise CampaignError("post-iteration handoff reference changed")
        control = validate_rank4_control_reference(
            payload.get("rank4_control_selection")
        )
        control_reference = payload.get("rank4_control_selection")
        expected_control_reference = {
            "path": str(control["selection_path"]),
            "sha256": base.sha256_file(control["selection_path"]),
            "body_sha256": control["selection"]["body_sha256"],
        }
        if control_reference != expected_control_reference:
            raise CampaignError("Rank-4 control selection reference changed")
    elif (
        "post_iteration_handoff" in payload
        or "rank4_control_selection" in payload
    ):
        raise CampaignError("family development cannot carry iteration provenance")
    banks = payload.get("banks")
    expected_banks = {
        "model_screen": 100, "tuple_screen": 100,
        "tuple_confirmation": 250, "profile_screen": 100,
        "profile_confirmation": 250, "actual_clock": 200,
    }
    if not isinstance(banks, dict) or set(banks) != set(expected_banks):
        raise CampaignError("development bank registry is incomplete")
    all_fingerprints: set[str] = set()
    for name, pairs in expected_banks.items():
        bank = _bank(banks[name], pairs=pairs, label=name)
        overlap = all_fingerprints.intersection(bank["fingerprints"])
        if overlap:
            raise CampaignError("development banks are not mutually disjoint")
        all_fingerprints.update(bank["fingerprints"])
    return dict(payload)


def select_development(output: pathlib.Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_development_input(payload)
    development_mode = payload.get(
        "development_mode", FAMILY_DEVELOPMENT_MODE,
    )
    top_models, by_model = _validate_model_screen(
        payload.get("model_screen"), payload.get("eligible_architectures"),
        payload.get("eligible_model_arms"),
        development_mode=development_mode,
    )
    tuple_ranked = _validate_exact_tuple_screen(
        payload.get("tuple_screen"), top_models, by_model
    )
    default_model = top_models[0]["candidate_id"]
    default_tuple_id = f"{default_model}:{tuple_id(DEFAULT_TUPLE)}"
    carried_tuples = []
    for candidate_id in [row["candidate_id"] for row in tuple_ranked[:2]] + [default_tuple_id]:
        if candidate_id not in carried_tuples:
            carried_tuples.append(candidate_id)
    tuple_architecture = {
        row["candidate_id"]: by_model[row["model_id"]]["architecture"]
        for row in tuple_ranked
    }
    if default_tuple_id not in tuple_architecture:
        raise CampaignError("default tuple did not complete the exact screen")
    tuple_descriptor = {
        row["candidate_id"]: (row["model_id"], row["tuple"])
        for row in tuple_ranked
    }
    raw_tuple_confirmation = payload.get("tuple_confirmation")
    if isinstance(raw_tuple_confirmation, list):
        for row in raw_tuple_confirmation:
            if (not isinstance(row, dict)
                    or row.get("candidate_id") not in tuple_descriptor
                    or (row.get("model_id"), row.get("tuple")) != tuple_descriptor[
                        row.get("candidate_id")
                    ]):
                raise CampaignError("tuple confirmation descriptor contradicts its screen")
    selected_tuple, tuple_confirmation = _confirmation_choice(
        raw_tuple_confirmation, pairs=250,
        carried_ids=carried_tuples, default_id=default_tuple_id,
        architecture_by_id=tuple_architecture, label="tuple",
    )
    profile_screen = _validate_profiles(
        payload.get("profile_screen"), pairs=100, label="profile screen"
    )
    failure_free_profiles = [row for row in profile_screen if row["failures"] == 0]
    if len(failure_free_profiles) < 2:
        raise CampaignError("profile screen has fewer than two failure-free profiles")
    ranked_profiles = sorted(
        failure_free_profiles,
        key=lambda row: _rank_key(row, tuple_architecture[selected_tuple["candidate_id"]]),
    )
    carried_profiles = []
    for profile in [row["candidate_id"] for row in ranked_profiles[:2]] + [DEFAULT_PROFILE]:
        if profile not in carried_profiles:
            carried_profiles.append(profile)
    profile_architecture = {
        profile: tuple_architecture[selected_tuple["candidate_id"]]
        for profile in PROFILE_ROSTER
    }
    profile_confirmation_rows = _validate_profiles(
        payload.get("profile_confirmation"), pairs=250,
        label="profile confirmation", expected_profiles=carried_profiles,
    )
    selected_profile, profile_confirmation = _confirmation_choice(
        profile_confirmation_rows, pairs=250,
        carried_ids=carried_profiles, default_id=DEFAULT_PROFILE,
        architecture_by_id=profile_architecture, label="profile",
    )
    actual = _metric(payload.get("actual_clock"), pairs=200, label="actual clock")
    expected_actual_id = (
        selected_tuple["candidate_id"] + ":" + selected_profile["candidate_id"]
    )
    if actual["candidate_id"] != expected_actual_id:
        raise CampaignError("actual-clock report does not bind the selected tuple/profile")
    actual_passed = (
        actual["failures"] == 0 and actual["wins"] >= 211
        and actual["color_wins"]["0"] >= 104
        and actual["color_wins"]["1"] >= 104
    )
    if not actual_passed:
        raise CampaignError("actual-clock gate failed the 211/104/zero-failure floor")
    selected_model_id = selected_tuple["model_id"]
    selected_model = by_model[selected_model_id]
    post_iteration_details: dict[str, Any] | None = None
    if development_mode == POST_ITERATION_DEVELOPMENT_MODE:
        post_iteration_details = validate_post_iteration_handoff(
            pathlib.Path(payload["post_iteration_handoff"]["path"])
        )
        control_details = validate_rank4_control_reference(
            payload.get("rank4_control_selection")
        )
        candidate = post_iteration_details["candidate"]
        control_row = by_model.get("rank4-control")
        candidate_row = by_model.get(POST_ITERATION_CANDIDATE_ID)
        if (
            candidate_row is None or control_row is None
            or candidate_row.get("architecture") != candidate["architecture"]
            or candidate_row.get("target") != candidate["target"]
            or candidate_row.get("artifact_sha256")
            != candidate["runtime"]["sha256"]
            or candidate_row.get("source_bytes")
            != candidate["generated_source"]["bytes"]
            or control_row.get("architecture") != PRIMARY_ARCHITECTURE
            or control_row.get("target") != CONTROL_TARGET
            or control_row.get("artifact_sha256")
            != control_details["runtime"]["sha256"]
        ):
            raise CampaignError(
                "post-iteration model screen differs from its candidate/control provenance"
            )
        _validate_post_iteration_development_evidence(
            payload, handoff_details=post_iteration_details,
            control_details=control_details,
        )
    body = {
        "schema": SELECTION_SCHEMA,
        "namespace": NAMESPACE,
        "status": "immutable-development-selected-not-tests-opened",
        "development_mode": development_mode,
        "input_body_sha256": base.sha256_bytes(base.canonical_json_bytes(dict(payload))),
        "banks": payload["banks"],
        "retained_model_ids": [row["candidate_id"] for row in top_models],
        "model": {
            "candidate_id": selected_model_id,
            "architecture": selected_model["architecture"],
            "target": selected_model["target"],
            "artifact_sha256": selected_model["artifact_sha256"],
            "source_bytes": selected_model["source_bytes"],
        },
        "tuple": selected_tuple["tuple"],
        "tuple_candidate_id": selected_tuple["candidate_id"],
        "profile": selected_profile["profile"],
        "profile_work": PROFILE_ROSTER[selected_profile["profile"]],
        "actual_clock": actual,
        "protected_tests_opened": False,
        "selection_immutable": True,
        "post_selection_test_results_may_change_selection": False,
        "audit": {
            "tuple_confirmation": tuple_confirmation,
            "profile_confirmation": profile_confirmation,
        },
    }
    if development_mode == POST_ITERATION_DEVELOPMENT_MODE:
        assert post_iteration_details is not None
        handoff_details = post_iteration_details
        candidate = handoff_details["candidate"]
        if (
            selected_model_id != candidate["candidate_id"]
            or selected_model["architecture"] != candidate["architecture"]
            or selected_model["target"] != candidate["target"]
            or selected_model["artifact_sha256"]
            != candidate["runtime"]["sha256"]
            or selected_model["source_bytes"]
            != candidate["generated_source"]["bytes"]
        ):
            raise CampaignError(
                "post-iteration selected model differs from the immutable handoff"
            )
        body["post_iteration_provenance"] = {
            "handoff": dict(payload["post_iteration_handoff"]),
            "plan": dict(handoff_details["handoff"]["plan"]),
            "iteration_completion": dict(
                handoff_details["handoff"]["iteration_completion"]
            ),
            "iteration_selection": dict(
                handoff_details["handoff"]["iteration_selection"]
            ),
            "runtime": dict(candidate["runtime"]),
            "generated_source": dict(candidate["generated_source"]),
        }
    decision = base.write_sealed(output, body)
    return decision


def authorize_protected_tests(
    output: pathlib.Path, *, selection_path: pathlib.Path,
    quantized_artifact_path: pathlib.Path, authorized_at_utc: str,
) -> dict[str, Any]:
    selection = base.load_sealed(selection_path, SELECTION_SCHEMA)
    if (selection.get("selection_immutable") is not True
            or selection.get("protected_tests_opened") is not False
            or selection.get("status") != "immutable-development-selected-not-tests-opened"):
        raise CampaignError("selection is not eligible to open protected tests")
    artifact = quantized_artifact_path.read_bytes()
    artifact_hash = base.sha256_bytes(artifact)
    if artifact_hash != selection["model"]["artifact_sha256"]:
        raise CampaignError("protected-test artifact differs from immutable selection")
    return base.write_sealed(output, {
        "schema": TEST_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "status": "protected-tests-authorized-once",
        "authorized_at_utc": authorized_at_utc,
        "selection": base.artifact_reference(selection_path, SELECTION_SCHEMA),
        "artifact": {"path": str(quantized_artifact_path.resolve()),
                     "bytes": len(artifact), "sha256": artifact_hash},
        "selection_may_change": False,
        "tests_may_only_diagnose": True,
    })


def _regular_file_reference(
    path: pathlib.Path, *, label: str, ascii_required: bool = False,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise CampaignError(f"{label} must be ASCII") from error
    result = {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": base.sha256_bytes(raw),
    }
    if ascii_required:
        result["ascii"] = True
    return result


def _sealed_file_reference(
    path: pathlib.Path, schema: str, *, label: str,
) -> dict[str, Any]:
    value = base.load_sealed(path, schema)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
        "body_sha256": value["body_sha256"],
        "schema": schema,
    }


def _iteration_artifact_reference(
    path: pathlib.Path, schema: str,
) -> dict[str, Any]:
    """Return the three-field reference used by iteration.py artifacts."""

    document = base.load_sealed(path, schema)
    return {
        "path": str(path.resolve()),
        "sha256": base.sha256_file(path),
        "body_sha256": document["body_sha256"],
    }


def _validate_regular_file_reference(
    value: Any, *, label: str, ascii_required: bool = False,
) -> pathlib.Path:
    expected = {"path", "bytes", "sha256"}
    if ascii_required:
        expected.add("ascii")
    if not isinstance(value, dict) or set(value) != expected:
        raise CampaignError(f"{label} reference is malformed")
    path = pathlib.Path(value["path"])
    if value != _regular_file_reference(
        path, label=label, ascii_required=ascii_required,
    ):
        raise CampaignError(f"{label} reference changed")
    return path.resolve()


def _validate_sealed_file_reference(
    value: Any, schema: str, *, label: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "path", "bytes", "sha256", "body_sha256", "schema",
    } or value.get("schema") != schema:
        raise CampaignError(f"{label} reference is malformed")
    path = pathlib.Path(value["path"])
    actual = _sealed_file_reference(path, schema, label=label)
    if value != actual:
        raise CampaignError(f"{label} reference changed")
    return path.resolve(), base.load_sealed(path, schema)


def _output_artifact(
    root: pathlib.Path, relative: Any, *, label: str,
) -> pathlib.Path:
    if not isinstance(relative, str) or not relative:
        raise CampaignError(f"{label} path is invalid")
    lexical = pathlib.Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise CampaignError(f"{label} escaped its artifact root")
    root = root.resolve()
    path = (root / lexical).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CampaignError(f"{label} escaped its artifact root") from error
    return path


def _write_content_addressed_sealed(
    output_directory: pathlib.Path, payload: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    artifact = base.seal(payload)
    raw = base.canonical_json_bytes(artifact)
    path = output_directory / f"{base.sha256_bytes(raw)}{suffix}"
    base.atomic_write_once(path, raw)
    return path, artifact


def _validate_content_addressed_sealed(
    path: pathlib.Path, schema: str, suffix: str, *, label: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular non-symlink file")
    value = base.load_sealed(path, schema)
    expected_name = f"{base.sha256_file(path)}{suffix}"
    if path.name != expected_name:
        raise CampaignError(f"{label} is not content addressed")
    return value


def _validate_iteration_artifact_reference(
    value: Any, schema: str, *, label: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "path", "sha256", "body_sha256",
    }:
        raise CampaignError(f"{label} artifact reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular non-symlink file")
    document = base.load_sealed(path, schema)
    expected = {
        "path": str(path.resolve()),
        "sha256": base.sha256_file(path),
        "body_sha256": document["body_sha256"],
    }
    if value != expected:
        raise CampaignError(f"{label} artifact reference changed")
    return path.resolve(), document


def _validate_iteration_file_record(
    value: Any, *, label: str, root: pathlib.Path | None = None,
    ascii_required: bool = False,
) -> tuple[pathlib.Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "path", "resolved_path", "bytes", "sha256", "executable",
    }:
        raise CampaignError(f"{label} file record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    resolved = pathlib.Path(str(value.get("resolved_path", "")))
    if (
        path.is_symlink() or not path.is_file()
        or path.resolve() != resolved.resolve()
        or str(path.resolve()) != value.get("path")
        or str(path.resolve()) != value.get("resolved_path")
        or value.get("executable") is not False
    ):
        raise CampaignError(f"{label} is not an immutable regular file")
    if root is not None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise CampaignError(f"{label} escaped the iteration output root") from error
    raw = path.read_bytes()
    if (
        value.get("bytes") != len(raw)
        or value.get("sha256") != base.sha256_bytes(raw)
    ):
        raise CampaignError(f"{label} bytes changed")
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise CampaignError(f"{label} must be ASCII") from error
    return path.resolve(), dict(value)


def validate_post_iteration_handoff(
    path: pathlib.Path,
) -> dict[str, Any]:
    """Validate the sole offline-qualified iteration-to-development handoff.

    The handoff is deliberately powerless: it can authorize only development
    games.  Its plan, completed one-shot iteration, model selection, runtime,
    and exact generated source are all re-read and re-hashed here.
    """

    iteration = _iteration_module()
    handoff = base.load_sealed(path, POST_ITERATION_HANDOFF_SCHEMA)
    expected_fields = {
        "schema", "namespace", "status", "plan", "iteration_completion",
        "iteration_selection", "candidate", "source_export", "offline_gate",
        "candidate_artifacts_immutable", "development_screen_required",
        "development_selected", "protected_tests_opened",
        "protected_tests_authorized", "upload_authorized",
        "iterations_remaining", "body_sha256",
    }
    if (
        set(handoff) != expected_fields
        or handoff.get("namespace") != NAMESPACE
        or handoff.get("status") != POST_ITERATION_HANDOFF_STATUS
        or handoff.get("candidate_artifacts_immutable") is not True
        or handoff.get("development_screen_required") is not True
        or handoff.get("development_selected") is not False
        or handoff.get("protected_tests_opened") is not False
        or handoff.get("protected_tests_authorized") is not False
        or handoff.get("upload_authorized") is not False
        or handoff.get("iterations_remaining") != 0
    ):
        raise CampaignError("post-iteration handoff is not development-only")

    plan_path, plan = _validate_iteration_artifact_reference(
        handoff.get("plan"), iteration.PLAN_SCHEMA, label="iteration plan",
    )
    output_root = plan_path.parent.resolve()
    if (
        plan_path != output_root / "iteration-plan.json"
        or path.resolve()
        != output_root / "post-iteration-development-handoff.json"
    ):
        raise CampaignError("post-iteration handoff/plan path is not canonical")
    try:
        iteration.validate_plan_contract(
            plan, plan_path=plan_path, output_root=output_root,
        )
    except Exception as error:
        raise CampaignError("post-iteration plan provenance did not validate") from error

    authorization_record = plan.get("authorization")
    authorization_path, authorization = _validate_iteration_artifact_reference(
        authorization_record, ITERATION_AUTH_SCHEMA,
        label="one-shot iteration authorization",
    )
    try:
        if validate_iteration_authorization(authorization_path) != authorization:
            raise CampaignError("iteration authorization validator disagreed")
    except Exception as error:
        if isinstance(error, CampaignError):
            raise
        raise CampaignError("post-iteration authorization did not validate") from error
    iteration_root = authorization_path.parent.parent.resolve()
    completion_path, completion = _validate_iteration_artifact_reference(
        handoff.get("iteration_completion"), ITERATION_EVENT_SCHEMA,
        label="completed one-shot iteration",
    )
    if completion_path != iteration_root / "iteration/02-completed.json":
        raise CampaignError("post-iteration completion path is not canonical")
    try:
        if _validate_completed_iteration(iteration_root) != completion:
            raise CampaignError("completed iteration validator disagreed")
    except Exception as error:
        if isinstance(error, CampaignError):
            raise
        raise CampaignError("completed iteration provenance did not validate") from error

    selection_path, selection = _validate_iteration_artifact_reference(
        handoff.get("iteration_selection"), ITERATION_SELECTION_SCHEMA,
        label="post-iteration model selection",
    )
    if (
        selection_path.name
        != f"{base.sha256_file(selection_path)}.iteration-selection.json"
    ):
        raise CampaignError("post-iteration selection is not content addressed")
    try:
        selection_path.relative_to(output_root)
    except ValueError as error:
        raise CampaignError("post-iteration selection escaped its output root") from error
    selection_fields = {
        "schema", "namespace", "campaign_id", "plan_body_sha256",
        "architecture", "seed", "float_epoch", "qat_epoch", "learning_rate",
        "new_train_manifests", "iteration_training_body_sha256",
        "split_isolation", "float_checkpoint", "runtime", "generated_source",
        "source_export", "float_validation", "quantized_validation",
        "quantized_selection_policy", "offline_gate", "status",
        "protected_tests_opened", "handoff", "body_sha256",
    }
    if (
        set(selection) != selection_fields
        or selection.get("namespace") != NAMESPACE
        or selection.get("campaign_id") != iteration.CAMPAIGN_ID
        or selection.get("plan_body_sha256") != plan.get("body_sha256")
        or selection.get("architecture") != plan.get("selected_architecture")
        or selection.get("seed") != plan.get("selected_seed")
        or selection.get("learning_rate") != plan.get("learning_rate")
        or selection.get("protected_tests_opened") is not False
        or selection.get("handoff")
        != "existing-fixed-scale-qAT-and-offline-selection"
        or selection.get("status")
        != "offline-evaluator-qualified-not-game-gated"
    ):
        raise CampaignError("post-iteration selection provenance is incomplete")
    policy = selection.get("quantized_selection_policy")
    if policy != {
        "primary": "offline-gate-feasibility",
        "secondary": "maximum-then-sum-normalized-gate-violation",
        "tertiary": "gate-error-count-then-original-validation-key",
        "qat_epochs": 4,
        "fixed_scales": True,
        "seed_policy": "authorization-bound-single-seed",
    }:
        raise CampaignError("post-iteration QAT selection policy changed")

    manifests = selection.get("new_train_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise CampaignError("post-iteration selection has no training manifests")
    validated_manifests = []
    seen_manifest_hashes = set()
    for index, record in enumerate(manifests):
        _manifest_path, validated = _validate_iteration_file_record(
            record, label=f"iteration train manifest {index}", root=output_root,
        )
        if validated["sha256"] in seen_manifest_hashes:
            raise CampaignError("post-iteration train manifest is repeated")
        seen_manifest_hashes.add(validated["sha256"])
        validated_manifests.append(validated)
    training_body = {
        "source_bundle_body_sha256": plan.get("source_bundle_body_sha256"),
        "plan_body_sha256": plan.get("body_sha256"),
        "manifests": [
            {"sha256": record["sha256"], "bytes": record["bytes"]}
            for record in validated_manifests
        ],
    }
    if selection.get("iteration_training_body_sha256") != base.sha256_bytes(
        base.canonical_json_bytes(training_body)
    ):
        raise CampaignError("post-iteration training provenance hash changed")

    checkpoint_path, checkpoint_record = _validate_iteration_file_record(
        selection.get("float_checkpoint"), label="post-iteration float checkpoint",
        root=output_root,
    )
    runtime_path, runtime_record = _validate_iteration_file_record(
        selection.get("runtime"), label="post-iteration runtime", root=output_root,
    )
    source_path, source_record = _validate_iteration_file_record(
        selection.get("generated_source"), label="post-iteration generated source",
        root=output_root, ascii_required=True,
    )
    if not 0 < source_record["bytes"] < 95_000:
        raise CampaignError("post-iteration generated source is not below 95,000 bytes")
    try:
        runtime_architecture, _weights, runtime_selection, _runtime = (
            _trainer_module().load_runtime(runtime_path)
        )
    except Exception as error:
        raise CampaignError("post-iteration runtime did not validate") from error
    if (
        runtime_architecture.name != selection.get("architecture")
        or runtime_selection.get("arm") != "search-target"
        or runtime_selection.get("seed") != selection.get("seed")
        or runtime_selection.get("float_epoch") != selection.get("float_epoch")
        or runtime_selection.get("qat_epoch") != selection.get("qat_epoch")
        or runtime_selection.get("source_bundle_body_sha256")
        != plan.get("source_bundle_body_sha256")
    ):
        raise CampaignError("post-iteration runtime identity changed")

    try:
        recomputed_gate = _trainer_module().offline_advancement_gate(
            selection["float_validation"], selection["quantized_validation"],
        )
    except Exception as error:
        raise CampaignError("post-iteration offline gate could not be recomputed") from error
    gate = selection.get("offline_gate")
    if (
        gate != recomputed_gate
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("status") != "offline-evaluator-qualified-not-game-gated"
        or gate.get("errors") != []
        or handoff.get("offline_gate") != gate
    ):
        raise CampaignError("post-iteration offline gate is not an exact pass")

    source_export = selection.get("source_export")
    if (
        not isinstance(source_export, dict)
        or set(source_export) != {
            "runtime_sha256", "runtime_body_sha256", "model_header_sha256",
            "source_sha256", "source_ascii_bytes", "source_limit_exclusive",
        }
        or source_export.get("runtime_sha256") != runtime_record["sha256"]
        or source_export.get("source_sha256") != source_record["sha256"]
        or source_export.get("source_ascii_bytes") != source_record["bytes"]
        or source_export.get("source_limit_exclusive") != 95_000
        or handoff.get("source_export") != source_export
    ):
        raise CampaignError("post-iteration runtime/source export binding changed")
    _sha(source_export.get("runtime_body_sha256"), "runtime body SHA-256")
    _sha(source_export.get("model_header_sha256"), "model header SHA-256")

    candidate = handoff.get("candidate")
    expected_architecture = WORKFLOW_ARCHITECTURES.get(
        str(selection.get("architecture"))
    )
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {
            "candidate_id", "architecture", "target", "float_checkpoint",
            "runtime", "generated_source",
        }
        or candidate.get("candidate_id") != POST_ITERATION_CANDIDATE_ID
        or candidate.get("architecture") != expected_architecture
        or candidate.get("target") != "search-target"
        or candidate.get("float_checkpoint") != checkpoint_record
        or candidate.get("runtime") != runtime_record
        or candidate.get("generated_source") != source_record
    ):
        raise CampaignError("post-iteration handoff candidate identity changed")

    result = completion.get("result")
    expected_result = {
        "games": dict(ITERATION_SPEC["games"]),
        "total_games": 10_000,
        "positions_per_game": 20,
        "workers": 10,
        "fixed_work": True,
        "deep_relabel_fraction": 0.25,
        "resumed": True,
        "float_checkpoint_sha256": checkpoint_record["sha256"],
        "quantized_runtime_sha256": runtime_record["sha256"],
        "generated_source_sha256": source_record["sha256"],
        "generated_source_ascii_bytes": source_record["bytes"],
        "offline_gate_passed": True,
        "iteration_selection_body_sha256": selection["body_sha256"],
        "learning_rate": plan.get("learning_rate"),
    }
    if result != expected_result:
        raise CampaignError("post-iteration completion does not bind the selected source")

    return {
        "handoff": handoff,
        "handoff_path": path.resolve(),
        "plan": plan,
        "plan_path": plan_path,
        "completion": completion,
        "completion_path": completion_path,
        "selection": selection,
        "selection_path": selection_path,
        "checkpoint_path": checkpoint_path,
        "runtime_path": runtime_path,
        "source_path": source_path,
        "candidate": dict(candidate),
    }


def validate_rank4_control_reference(value: Any) -> dict[str, Any]:
    trainer = _trainer_module()
    selection_path, selection = _validate_iteration_artifact_reference(
        value, trainer.SELECTION_SCHEMA, label="Rank-4 control selection",
    )
    if (
        selection_path.parent.name != "selections"
        or selection_path.name
        != f"{base.sha256_file(selection_path)}.selection.json"
    ):
        raise CampaignError("Rank-4 control selection path is not canonical")
    campaign_root = selection_path.parent.parent.resolve()
    runtime = selection.get("runtime")
    if (
        selection.get("architecture") != "compact-8x8"
        or selection.get("arm") != "rank4-control"
        or selection.get("deployment_eligible") is not False
        or selection.get("rank4_control_never_deployment_eligible") is not True
        or selection.get("protected_tests_opened") is not False
        or selection.get("game_gated") is not False
        or not isinstance(runtime, dict)
        or set(runtime) != {"path", "sha256", "bytes"}
    ):
        raise CampaignError("Rank-4 control selection is not exact/nondeployable")
    runtime_path = _output_artifact(
        campaign_root, runtime.get("path"), label="Rank-4 control runtime",
    )
    if (
        runtime_path.is_symlink() or not runtime_path.is_file()
        or runtime.get("sha256") != base.sha256_file(runtime_path)
        or runtime.get("bytes") != runtime_path.stat().st_size
    ):
        raise CampaignError("Rank-4 control runtime changed")
    try:
        architecture, _weights, runtime_selection, _document = trainer.load_runtime(
            runtime_path
        )
    except Exception as error:
        raise CampaignError("Rank-4 control runtime did not validate") from error
    if (
        architecture.name != "compact-8x8"
        or runtime_selection.get("arm") != "rank4-control"
        or runtime_selection.get("seed") != selection.get("seed")
        or runtime_selection.get("float_epoch") != selection.get("float_epoch")
        or runtime_selection.get("qat_epoch") != selection.get("qat_epoch")
        or runtime_selection.get("source_bundle_body_sha256")
        != selection.get("source_bundle_body_sha256")
    ):
        raise CampaignError("Rank-4 control selection/runtime identity changed")
    return {
        "selection": selection,
        "selection_path": selection_path,
        "runtime_path": runtime_path,
        "runtime": dict(runtime),
    }


def _metric_from_development_receipt(
    gate_result: Mapping[str, Any], *, candidate_id: str, pairs: int,
) -> dict[str, Any]:
    result = gate_result.get("result")
    if not isinstance(result, dict):
        raise CampaignError("development receipt has no gate result")
    candidate = result.get("candidate")
    times = candidate.get("times_ms") if isinstance(candidate, dict) else None
    if (
        not isinstance(times, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in times
        )
    ):
        raise CampaignError("development receipt candidate timings are invalid")
    latency = sum(float(value) for value in times) / len(times) if times else 0.0
    return {
        "candidate_id": candidate_id,
        "pairs": pairs,
        "games": pairs * 2,
        "wins": result.get("candidate_wins"),
        "color_wins": {
            "0": result.get("candidate_wins_player0"),
            "1": result.get("candidate_wins_player1"),
        },
        "failures": result.get("failures"),
        "latency_ms": latency,
    }


def _validate_post_iteration_development_evidence(
    payload: Mapping[str, Any], *, handoff_details: Mapping[str, Any],
    control_details: Mapping[str, Any],
) -> None:
    bank_evidence = payload.get("development_bank_evidence")
    expected_pairs = {
        "model_screen": 100, "tuple_screen": 100,
        "tuple_confirmation": 250, "profile_screen": 100,
        "profile_confirmation": 250, "actual_clock": 200,
    }
    if not isinstance(bank_evidence, dict) or set(bank_evidence) != set(expected_pairs):
        raise CampaignError("post-iteration development bank evidence is incomplete")
    for stage, record in bank_evidence.items():
        if not isinstance(record, dict) or set(record) != {
            "manifest_path", "manifest_sha256", "gate_path", "gate_sha256",
        }:
            raise CampaignError(f"{stage} development bank evidence is malformed")
        manifest = pathlib.Path(str(record.get("manifest_path", "")))
        gate = pathlib.Path(str(record.get("gate_path", "")))
        if (
            manifest.is_symlink() or not manifest.is_file()
            or gate.is_symlink() or not gate.is_file()
            or record.get("manifest_sha256") != base.sha256_file(manifest)
            or record.get("gate_sha256") != base.sha256_file(gate)
            or manifest.name
            != f"{record['manifest_sha256']}.opening-bank.json"
            or gate.name != f"{record['gate_sha256']}.tsv"
        ):
            raise CampaignError(f"{stage} development bank evidence changed")

    rows_by_stage: dict[str, list[Mapping[str, Any]]] = {}
    for stage in (
        "model_screen", "tuple_screen", "tuple_confirmation",
        "profile_screen", "profile_confirmation",
    ):
        rows = payload.get(stage)
        if not isinstance(rows, list):
            raise CampaignError(f"{stage} rows are missing")
        rows_by_stage[stage] = rows
    actual = payload.get("actual_clock")
    if not isinstance(actual, Mapping):
        raise CampaignError("actual-clock row is missing")
    rows_by_stage["actual_clock"] = [actual]
    expected_roster = {
        (stage, str(row.get("candidate_id")))
        for stage, rows in rows_by_stage.items() for row in rows
    }
    if len(expected_roster) != sum(len(rows) for rows in rows_by_stage.values()):
        raise CampaignError("post-iteration development metric roster is repeated")

    references = payload.get("development_run_receipts")
    if not isinstance(references, list) or len(references) != len(expected_roster):
        raise CampaignError("post-iteration development receipt roster is incomplete")
    iteration_selection = handoff_details["selection"]
    iteration_candidate = handoff_details["candidate"]
    control_selection = control_details["selection"]
    control_runtime = control_details["runtime"]
    observed: set[tuple[str, str]] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, dict) or set(reference) != {
            "request_sha256", "receipt", "path", "bytes", "receipt_sha256",
            "receipt_body_sha256", "schema",
        } or reference.get("schema") != DEVELOPMENT_RUN_SCHEMA:
            raise CampaignError(f"development receipt reference {index} is malformed")
        receipt_path = pathlib.Path(str(reference.get("path", "")))
        if (
            receipt_path.is_symlink() or not receipt_path.is_file()
            or receipt_path.name != reference.get("receipt")
            or receipt_path.stat().st_size != reference.get("bytes")
            or base.sha256_file(receipt_path) != reference.get("receipt_sha256")
            or receipt_path.name
            != f"{reference.get('receipt_sha256')}.development-run.json"
        ):
            raise CampaignError("development receipt reference changed")
        receipt = base.load_sealed(receipt_path, DEVELOPMENT_RUN_SCHEMA)
        request = receipt.get("request")
        request_sha = reference.get("request_sha256")
        if (
            not isinstance(request, dict)
            or request_sha != base.sha256_bytes(base.canonical_json_bytes(request))
            or receipt.get("request_sha256") != request_sha
            or receipt.get("body_sha256") != reference.get("receipt_body_sha256")
            or receipt.get("namespace") != NAMESPACE
            or receipt.get("selection_sha256") != request.get("selection_sha256")
            or receipt.get("selection_body_sha256")
            != request.get("selection_body_sha256")
            or receipt.get("runtime_sha256") != request.get("runtime_sha256")
        ):
            raise CampaignError("development receipt/request binding changed")
        stage = request.get("stage")
        candidate_id = request.get("candidate_id")
        key = (stage, candidate_id)
        if key not in expected_roster or key in observed:
            raise CampaignError("development receipt roster is foreign/repeated")
        observed.add(key)
        if (
            request.get("pairs") != expected_pairs[stage]
            or request.get("bank_sha256")
            != bank_evidence[stage]["gate_sha256"]
            or request.get("bank_manifest_sha256")
            != bank_evidence[stage]["manifest_sha256"]
        ):
            raise CampaignError("development receipt bank/pair binding changed")
        model_candidate_id = request.get("model_candidate_id")
        if model_candidate_id == "rank4-control":
            if (
                stage != "model_screen"
                or candidate_id != "rank4-control"
                or request.get("selection_sha256")
                != base.sha256_file(control_details["selection_path"])
                or request.get("selection_body_sha256")
                != control_selection["body_sha256"]
                or request.get("runtime_sha256") != control_runtime["sha256"]
            ):
                raise CampaignError("development receipt uses a fake Rank-4 control")
        elif (
            model_candidate_id != POST_ITERATION_CANDIDATE_ID
            or request.get("selection_sha256")
            != base.sha256_file(handoff_details["selection_path"])
            or request.get("selection_body_sha256")
            != iteration_selection["body_sha256"]
            or request.get("runtime_sha256")
            != iteration_candidate["runtime"]["sha256"]
            or request.get("candidate_source_sha256")
            != iteration_candidate["generated_source"]["sha256"]
        ):
            raise CampaignError("development receipt uses a fake iteration candidate")
        gate_result = receipt.get("gate_result")
        if not isinstance(gate_result, dict):
            raise CampaignError("development receipt gate result is missing")
        bindings = gate_result.get("bindings")
        if (
            not isinstance(bindings, dict)
            or bindings.get("bank_sha256") != request.get("bank_sha256")
            or bindings.get("candidate_source_sha256")
            != request.get("candidate_source_sha256")
            or bindings.get("rank4_source_sha256")
            != request.get("rank4_source_sha256")
        ):
            raise CampaignError("development gate result binding changed")
        row = next(
            row for row in rows_by_stage[stage]
            if row.get("candidate_id") == candidate_id
        )
        receipt_metric = _metric_from_development_receipt(
            gate_result, candidate_id=candidate_id, pairs=expected_pairs[stage],
        )
        if any(row.get(field) != value for field, value in receipt_metric.items()):
            raise CampaignError("development metrics differ from their gate receipt")
    if observed != expected_roster:
        raise CampaignError("development receipt roster is incomplete")


def _workflow_failure_details(
    *, bundle_manifest: pathlib.Path, run_output_directory: pathlib.Path,
    run_reference: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = _workflow_module()
    run_output_directory = run_output_directory.resolve()
    expected_reference = run_output_directory / "run-state/run-reference.json"
    if run_reference.resolve() != expected_reference:
        raise CampaignError("offline-family failure requires the canonical run reference")
    try:
        receipt = workflow.verify_family_run(
            bundle_manifest, run_output_directory,
            reference_path=run_reference,
        )
    except Exception as error:
        raise CampaignError("workflow family run did not fully validate") from error
    reference = base.load_sealed(run_reference, workflow.RUN_REFERENCE_SCHEMA)
    receipt_record = reference.get("receipt")
    if not isinstance(receipt_record, dict):
        raise CampaignError("workflow run reference has no receipt")
    receipt_path = _output_artifact(
        run_output_directory, receipt_record.get("path"), label="workflow receipt",
    )
    if base.load_sealed(receipt_path, workflow.RUN_RECEIPT_SCHEMA) != receipt:
        raise CampaignError("workflow verifier and receipt artifact disagree")

    if receipt.get("campaign_order") != list(WORKFLOW_CAMPAIGN_ORDER):
        raise CampaignError("workflow run does not contain the exact seven campaigns")
    campaigns = receipt.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) != 7:
        raise CampaignError("workflow run does not contain all seven campaigns")
    preflight = receipt.get("source_size_preflight")
    if not isinstance(preflight, dict) or set(preflight) != set(WORKFLOW_ARCHITECTURES):
        raise CampaignError("workflow source-size preflight roster is incomplete")
    for architecture, measurement in preflight.items():
        if (
            not isinstance(measurement, dict)
            or measurement.get("architecture") != architecture
            or measurement.get("eligible") is not True
            or measurement.get("limit") != 95_000
            or _int(
                measurement.get("complete_source_ascii_characters"),
                f"{architecture} preflight source bytes", 1,
            ) > 95_000
        ):
            raise CampaignError("workflow architecture is not source-size eligible")

    expected_deployable = set(WORKFLOW_DEPLOYABLE_ARMS)
    observed_deployable: set[tuple[str, str]] = set()
    rejected = []
    control: dict[str, Any] | None = None
    trainer = _trainer_module()
    for record in campaigns:
        if not isinstance(record, dict):
            raise CampaignError("workflow campaign record is malformed")
        architecture = record.get("architecture")
        arm = record.get("arm")
        key = (architecture, arm)
        campaign_name = f"{architecture}--{arm}"
        if record.get("name") != campaign_name:
            raise CampaignError("workflow campaign identity changed")
        if record.get("protected_tests_opened") is not False:
            raise CampaignError("protected tests were opened during the family run")
        campaign_output = _output_artifact(
            run_output_directory, record.get("campaign_output"),
            label="workflow campaign output",
        )
        selection_record = record.get("selection")
        runtime_record = record.get("runtime")
        if not isinstance(selection_record, dict) or not isinstance(runtime_record, dict):
            raise CampaignError("workflow selection/runtime record is malformed")
        selection_path = _output_artifact(
            run_output_directory, selection_record.get("path"),
            label="workflow family selection",
        )
        runtime_path = _output_artifact(
            run_output_directory, runtime_record.get("path"),
            label="workflow selected runtime",
        )
        selection = base.load_sealed(selection_path, trainer.SELECTION_SCHEMA)
        runtime = base.load_sealed(runtime_path, trainer.RUNTIME_SCHEMA)
        if (
            selection.get("protected_tests_opened") is not False
            or selection.get("game_gated") is not False
            or selection.get("architecture") != architecture
            or selection.get("arm") != arm
            or selection_record.get("sha256") != base.sha256_file(selection_path)
            or selection_record.get("body_sha256") != selection.get("body_sha256")
            or runtime_record.get("sha256") != base.sha256_file(runtime_path)
            or runtime_record.get("body_sha256") != runtime.get("body_sha256")
        ):
            raise CampaignError("workflow selection/runtime identity changed")
        if key in expected_deployable:
            if key in observed_deployable:
                raise CampaignError("workflow repeats a deployable family arm")
            source = record.get("exact_complete_source")
            source_eligibility = selection.get("source_size_eligibility")
            if (
                selection.get("status") != "offline-evaluator-rejected"
                or selection.get("deployment_eligible") is not False
                or not isinstance(selection.get("offline_gate"), dict)
                or selection["offline_gate"].get("passed") is not False
                or not isinstance(source_eligibility, dict)
                or source_eligibility.get("passed") is not True
                or source_eligibility.get("maximum_ascii_bytes") != 95_000
                or not isinstance(source, dict)
                or source.get("architecture") != architecture
                or source.get("eligible") is not True
                or source.get("limit") != 95_000
                or _int(
                    source.get("complete_source_ascii_characters"),
                    f"{campaign_name} complete source bytes", 1,
                ) > 95_000
                or source.get("runtime_file_sha256") != runtime_record.get("sha256")
                or source.get("runtime_body_sha256") != runtime_record.get("body_sha256")
            ):
                raise CampaignError(
                    "all six deployable arms must be offline-rejected and source eligible"
                )
            observed_deployable.add(key)
            rejected.append({
                "campaign": campaign_name,
                "campaign_output": str(campaign_output),
                "architecture": architecture,
                "deployment_architecture": WORKFLOW_ARCHITECTURES[architecture],
                "arm": arm,
                "seed": selection["seed"],
                "selection": _sealed_file_reference(
                    selection_path, trainer.SELECTION_SCHEMA,
                    label="offline-rejected selection",
                ),
                "runtime": _sealed_file_reference(
                    runtime_path, trainer.RUNTIME_SCHEMA,
                    label="offline-rejected runtime",
                ),
                "exact_complete_source": dict(source),
            })
        elif key == WORKFLOW_CONTROL:
            if control is not None or (
                record.get("rank4_control_never_deployment_eligible") is not True
                or selection.get("rank4_control_never_deployment_eligible") is not True
                or selection.get("deployment_eligible") is not False
                or record.get("exact_complete_source") is not None
            ):
                raise CampaignError("workflow Rank-4 control is not exact/nondeployable")
            control = {
                "campaign": campaign_name,
                "architecture": architecture,
                "arm": arm,
                "selection": _sealed_file_reference(
                    selection_path, trainer.SELECTION_SCHEMA,
                    label="Rank-4 control selection",
                ),
            }
        else:
            raise CampaignError("workflow contains a foreign family arm")
    if observed_deployable != expected_deployable or control is None:
        raise CampaignError("workflow lacks exactly six rejected arms and one control")
    if (
        receipt.get("all_seven_campaigns_complete") is not True
        or receipt.get("protected_tests_opened") is not False
        or receipt.get("protected_tests_locked") is not True
        or receipt.get("game_gated") is not False
    ):
        raise CampaignError("workflow family receipt opened a forbidden gate")
    details = {
        "run_receipt": _sealed_file_reference(
            receipt_path, workflow.RUN_RECEIPT_SCHEMA, label="workflow run receipt",
        ),
        "campaign_order": list(WORKFLOW_CAMPAIGN_ORDER),
        "source_size_preflight": dict(preflight),
        "rejected_deployable_arms": rejected,
        "rank4_control": control,
    }
    return receipt, details


def record_offline_family_failure(
    output_directory: pathlib.Path, *, bundle_manifest: pathlib.Path,
    run_output_directory: pathlib.Path, run_reference: pathlib.Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    base._utc(recorded_at_utc, "offline-family failure timestamp")
    receipt, details = _workflow_failure_details(
        bundle_manifest=bundle_manifest,
        run_output_directory=run_output_directory,
        run_reference=run_reference,
    )
    _path, artifact = _write_content_addressed_sealed(output_directory, {
        "schema": OFFLINE_FAMILY_FAILURE_SCHEMA,
        "namespace": NAMESPACE,
        "status": "offline-family-rejected-iteration-eligible",
        "failed": True,
        "failure_stage": "offline-evaluator",
        "recorded_at_utc": recorded_at_utc,
        "bundle_manifest": _regular_file_reference(
            bundle_manifest, label="frozen bundle manifest",
        ),
        "run_output_directory": str(run_output_directory.resolve()),
        "run_reference": _sealed_file_reference(
            run_reference, _workflow_module().RUN_REFERENCE_SCHEMA,
            label="workflow run reference",
        ),
        **details,
        "all_seven_campaigns_complete": True,
        "deployable_arms_rejected": 6,
        "protected_tests_opened": False,
        "source_size_eligible": True,
        "iteration_authorizable": True,
        "workflow_receipt_body_sha256": receipt["body_sha256"],
    }, ".offline-family-failure.json")
    return artifact


def offline_family_failure_path(
    output_directory: pathlib.Path, artifact: Mapping[str, Any],
) -> pathlib.Path:
    raw = base.canonical_json_bytes(dict(artifact))
    return output_directory / (
        base.sha256_bytes(raw) + ".offline-family-failure.json"
    )


def validate_offline_family_failure(
    path: pathlib.Path,
) -> dict[str, Any]:
    value = _validate_content_addressed_sealed(
        path, OFFLINE_FAMILY_FAILURE_SCHEMA, ".offline-family-failure.json",
        label="offline-family failure",
    )
    expected_fields = {
        "schema", "namespace", "status", "failed", "failure_stage",
        "recorded_at_utc", "bundle_manifest", "run_output_directory",
        "run_reference", "run_receipt", "campaign_order",
        "source_size_preflight", "rejected_deployable_arms", "rank4_control",
        "all_seven_campaigns_complete", "deployable_arms_rejected",
        "protected_tests_opened", "source_size_eligible",
        "iteration_authorizable", "workflow_receipt_body_sha256", "body_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("namespace") != NAMESPACE
        or value.get("status") != "offline-family-rejected-iteration-eligible"
        or value.get("failed") is not True
        or value.get("failure_stage") != "offline-evaluator"
        or value.get("all_seven_campaigns_complete") is not True
        or value.get("deployable_arms_rejected") != 6
        or value.get("protected_tests_opened") is not False
        or value.get("source_size_eligible") is not True
        or value.get("iteration_authorizable") is not True
    ):
        raise CampaignError("offline-family failure contract changed")
    base._utc(value.get("recorded_at_utc"), "offline-family failure timestamp")
    bundle_manifest = _validate_regular_file_reference(
        value.get("bundle_manifest"), label="frozen bundle manifest",
    )
    run_reference, _reference = _validate_sealed_file_reference(
        value.get("run_reference"), _workflow_module().RUN_REFERENCE_SCHEMA,
        label="workflow run reference",
    )
    run_output = pathlib.Path(value.get("run_output_directory", ""))
    receipt, details = _workflow_failure_details(
        bundle_manifest=bundle_manifest,
        run_output_directory=run_output,
        run_reference=run_reference,
    )
    for field in (
        "run_receipt", "campaign_order", "source_size_preflight",
        "rejected_deployable_arms", "rank4_control",
    ):
        if value.get(field) != details[field]:
            raise CampaignError(f"offline-family failure {field} changed")
    if value.get("workflow_receipt_body_sha256") != receipt.get("body_sha256"):
        raise CampaignError("offline-family failure receipt binding changed")
    return value


def _validate_build_evidence(
    path: pathlib.Path, *, source_sha256: str, runtime_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    value = base.load_sealed(path, OPERATIONAL_BUILD_EVIDENCE_SCHEMA)
    if (
        set(value) != {
            "schema", "namespace", "status", "source_sha256",
            "runtime_sha256", "binary", "compiler", "commands",
            "protected_tests_opened", "body_sha256",
        }
        or value.get("namespace") != NAMESPACE
        or value.get("status") != "build-and-tests-passed"
        or value.get("source_sha256") != source_sha256
        or value.get("runtime_sha256") != runtime_sha256
        or value.get("protected_tests_opened") is not False
    ):
        raise CampaignError("operational build evidence did not pass/bind the actor")
    binary = value.get("binary")
    _validate_regular_file_reference(binary, label="operational actor binary")
    compiler = value.get("compiler")
    if not isinstance(compiler, dict) or set(compiler) != {
        "executable", "version", "version_sha256",
    }:
        raise CampaignError("operational build compiler evidence is malformed")
    _validate_regular_file_reference(
        compiler.get("executable"), label="operational compiler executable",
    )
    version = compiler.get("version")
    if (
        not isinstance(version, str) or not version
        or compiler.get("version_sha256")
        != base.sha256_bytes(version.encode("utf-8"))
    ):
        raise CampaignError("operational compiler version binding changed")
    commands = value.get("commands")
    if not isinstance(commands, dict) or set(commands) != {"compile", "tests"}:
        raise CampaignError("operational build command receipts are incomplete")
    for name in ("compile", "tests"):
        receipt = commands[name]
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "argv", "exit_code", "stdout_sha256", "stderr_sha256",
            }
            or not isinstance(receipt.get("argv"), list)
            or not receipt["argv"]
            or any(not isinstance(item, str) or not item for item in receipt["argv"])
            or receipt.get("exit_code") != 0
        ):
            raise CampaignError(f"operational {name} command did not pass")
        _sha(receipt.get("stdout_sha256"), f"operational {name} stdout SHA-256")
        _sha(receipt.get("stderr_sha256"), f"operational {name} stderr SHA-256")
    return value, _sealed_file_reference(
        path, OPERATIONAL_BUILD_EVIDENCE_SCHEMA,
        label="operational build evidence",
    ), dict(binary)


def _validate_protocol_evidence(
    path: pathlib.Path, *, source_sha256: str, runtime_sha256: str,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    value = base.load_sealed(path, OPERATIONAL_PROTOCOL_EVIDENCE_SCHEMA)
    if (
        set(value) != {
            "schema", "namespace", "status", "source_sha256",
            "runtime_sha256", "binary", "player_roles", "failures",
            "protected_tests_opened", "body_sha256",
        }
        or value.get("namespace") != NAMESPACE
        or value.get("status") != "protocol-both-player-roles-passed"
        or value.get("source_sha256") != source_sha256
        or value.get("runtime_sha256") != runtime_sha256
        or value.get("binary") != binary
        or value.get("player_roles") != {"0": True, "1": True}
        or value.get("failures") != 0
        or value.get("protected_tests_opened") is not False
    ):
        raise CampaignError("operational protocol evidence did not pass/bind the actor")
    _validate_regular_file_reference(
        value.get("binary"), label="protocol-tested operational binary",
    )
    return _sealed_file_reference(
        path, OPERATIONAL_PROTOCOL_EVIDENCE_SCHEMA,
        label="operational protocol evidence",
    )


def _operational_actor_details(
    *, failure_record_path: pathlib.Path, family_selection_path: pathlib.Path,
    float_checkpoint_path: pathlib.Path, runtime_path: pathlib.Path,
    generated_source_path: pathlib.Path, protocol_evidence_path: pathlib.Path,
    build_evidence_path: pathlib.Path, timing_evidence_path: pathlib.Path,
) -> dict[str, Any]:
    failure = validate_offline_family_failure(failure_record_path)
    failure_reference = _sealed_file_reference(
        failure_record_path, OFFLINE_FAMILY_FAILURE_SCHEMA,
        label="offline-family failure",
    )
    selection_reference = _sealed_file_reference(
        family_selection_path, _trainer_module().SELECTION_SCHEMA,
        label="operational actor selection",
    )
    matches = [
        row for row in failure["rejected_deployable_arms"]
        if row.get("selection") == selection_reference
    ]
    if len(matches) != 1:
        raise CampaignError("operational actor is not one offline-rejected family selection")
    family_row = matches[0]
    campaign_output = pathlib.Path(family_row["campaign_output"])
    bundle_manifest = pathlib.Path(failure["bundle_manifest"]["path"])
    trainer = _trainer_module()
    try:
        bundle = trainer.FrozenBundle.load(bundle_manifest)
        selection = trainer.validate_selection(
            family_selection_path, campaign_output, bundle,
        )
    except Exception as error:
        raise CampaignError("operational actor family selection did not validate") from error
    if (
        selection.get("status") != "offline-evaluator-rejected"
        or selection.get("deployment_eligible") is not False
        or selection.get("offline_gate", {}).get("passed") is not False
        or selection.get("protected_tests_opened") is not False
        or selection.get("game_gated") is not False
    ):
        raise CampaignError("operational actor selection is not offline-rejected and locked")

    selected_receipt = selection.get("selected_seed_receipt")
    if not isinstance(selected_receipt, dict):
        raise CampaignError("operational actor selection omits its seed receipt")
    seed_receipt_path = _output_artifact(
        campaign_output, selected_receipt.get("path"),
        label="selected seed receipt",
    )
    seed_receipt = base.load_sealed(seed_receipt_path, trainer.SEED_RECEIPT_SCHEMA)
    float_record = seed_receipt.get("float_checkpoint")
    if not isinstance(float_record, dict):
        raise CampaignError("selected seed receipt omits its float checkpoint")
    selected_float_path = _output_artifact(
        campaign_output, float_record.get("path"), label="selected float checkpoint",
    )
    if (
        float_checkpoint_path.resolve() != selected_float_path
        or _regular_file_reference(
            float_checkpoint_path, label="selected float checkpoint",
        )
        != {
            "path": str(selected_float_path),
            "bytes": float_record.get("bytes"),
            "sha256": float_record.get("sha256"),
        }
    ):
        raise CampaignError("supplied float checkpoint is not the exact selected checkpoint")

    runtime_reference = _sealed_file_reference(
        runtime_path, trainer.RUNTIME_SCHEMA, label="selected quantized runtime",
    )
    if (
        runtime_path.resolve()
        != pathlib.Path(family_row["runtime"]["path"]).resolve()
        or runtime_reference != family_row["runtime"]
    ):
        raise CampaignError("supplied runtime is not the exact selected runtime")
    source_reference = _regular_file_reference(
        generated_source_path, label="generated complete source", ascii_required=True,
    )
    measured_source = family_row["exact_complete_source"]
    if (
        not 0 < source_reference["bytes"] <= 95_000
        or source_reference["bytes"]
        != measured_source.get("complete_source_ascii_characters")
        or source_reference["sha256"]
        != measured_source.get("complete_source_sha256")
        or measured_source.get("eligible") is not True
    ):
        raise CampaignError("generated complete source is not the exact under-95k actor")
    _build, build_reference, binary_reference = _validate_build_evidence(
        build_evidence_path,
        source_sha256=source_reference["sha256"],
        runtime_sha256=runtime_reference["sha256"],
    )
    protocol_reference = _validate_protocol_evidence(
        protocol_evidence_path,
        source_sha256=source_reference["sha256"],
        runtime_sha256=runtime_reference["sha256"],
        binary=binary_reference,
    )
    preflight = _preflight_module()
    timing = base.load_sealed(timing_evidence_path, preflight.TIMING_SCHEMA)
    try:
        preflight.validate_timing_receipt(timing)
    except Exception as error:
        raise CampaignError("operational timing evidence did not validate") from error
    timing_reference = _sealed_file_reference(
        timing_evidence_path, preflight.TIMING_SCHEMA,
        label="operational 1/2/10-process timing evidence",
    )
    return {
        "offline_family_failure": failure_reference,
        "selection": selection_reference,
        "seed_receipt": _sealed_file_reference(
            seed_receipt_path, trainer.SEED_RECEIPT_SCHEMA,
            label="selected seed receipt",
        ),
        "float_checkpoint": _regular_file_reference(
            float_checkpoint_path, label="selected float checkpoint",
        ),
        "runtime": runtime_reference,
        "generated_source": source_reference,
        "build_evidence": build_reference,
        "protocol_evidence": protocol_reference,
        "timing_evidence": timing_reference,
        "workflow_architecture": selection["architecture"],
        "architecture": WORKFLOW_ARCHITECTURES[selection["architecture"]],
        "arm": selection["arm"],
        "seed": selection["seed"],
    }


def record_operational_safe_actor(
    output_directory: pathlib.Path, *, failure_record_path: pathlib.Path,
    family_selection_path: pathlib.Path, float_checkpoint_path: pathlib.Path,
    runtime_path: pathlib.Path, generated_source_path: pathlib.Path,
    protocol_evidence_path: pathlib.Path, build_evidence_path: pathlib.Path,
    timing_evidence_path: pathlib.Path, recorded_at_utc: str,
) -> dict[str, Any]:
    base._utc(recorded_at_utc, "operational-safe actor timestamp")
    details = _operational_actor_details(
        failure_record_path=failure_record_path,
        family_selection_path=family_selection_path,
        float_checkpoint_path=float_checkpoint_path,
        runtime_path=runtime_path,
        generated_source_path=generated_source_path,
        protocol_evidence_path=protocol_evidence_path,
        build_evidence_path=build_evidence_path,
        timing_evidence_path=timing_evidence_path,
    )
    _path, artifact = _write_content_addressed_sealed(output_directory, {
        "schema": OPERATIONAL_SAFE_ACTOR_SCHEMA,
        "namespace": NAMESPACE,
        "status": "offline-rejected-operationally-safe-actor",
        "recorded_at_utc": recorded_at_utc,
        **details,
        "source_limit_ascii_bytes": 95_000,
        "operationally_safe": True,
        "protected_tests_opened": False,
        "game_gated": False,
    }, ".operational-safe-actor.json")
    return artifact


def operational_safe_actor_path(
    output_directory: pathlib.Path, artifact: Mapping[str, Any],
) -> pathlib.Path:
    raw = base.canonical_json_bytes(dict(artifact))
    return output_directory / (
        base.sha256_bytes(raw) + ".operational-safe-actor.json"
    )


def validate_operational_safe_actor(path: pathlib.Path) -> dict[str, Any]:
    value = _validate_content_addressed_sealed(
        path, OPERATIONAL_SAFE_ACTOR_SCHEMA, ".operational-safe-actor.json",
        label="operational-safe actor",
    )
    expected_fields = {
        "schema", "namespace", "status", "recorded_at_utc",
        "offline_family_failure", "selection", "seed_receipt",
        "float_checkpoint", "runtime", "generated_source", "build_evidence",
        "protocol_evidence", "timing_evidence", "workflow_architecture", "architecture", "arm",
        "seed", "source_limit_ascii_bytes", "operationally_safe",
        "protected_tests_opened", "game_gated", "body_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("namespace") != NAMESPACE
        or value.get("status") != "offline-rejected-operationally-safe-actor"
        or value.get("source_limit_ascii_bytes") != 95_000
        or value.get("operationally_safe") is not True
        or value.get("protected_tests_opened") is not False
        or value.get("game_gated") is not False
    ):
        raise CampaignError("operational-safe actor contract changed")
    base._utc(value.get("recorded_at_utc"), "operational-safe actor timestamp")
    failure_path, _failure = _validate_sealed_file_reference(
        value.get("offline_family_failure"), OFFLINE_FAMILY_FAILURE_SCHEMA,
        label="offline-family failure",
    )
    selection_path = pathlib.Path(value.get("selection", {}).get("path", ""))
    float_path = pathlib.Path(value.get("float_checkpoint", {}).get("path", ""))
    runtime_path = pathlib.Path(value.get("runtime", {}).get("path", ""))
    source_path = pathlib.Path(value.get("generated_source", {}).get("path", ""))
    protocol_path = pathlib.Path(value.get("protocol_evidence", {}).get("path", ""))
    build_path = pathlib.Path(value.get("build_evidence", {}).get("path", ""))
    timing_path = pathlib.Path(value.get("timing_evidence", {}).get("path", ""))
    details = _operational_actor_details(
        failure_record_path=failure_path,
        family_selection_path=selection_path,
        float_checkpoint_path=float_path,
        runtime_path=runtime_path,
        generated_source_path=source_path,
        protocol_evidence_path=protocol_path,
        build_evidence_path=build_path,
        timing_evidence_path=timing_path,
    )
    for field, expected in details.items():
        if value.get(field) != expected:
            raise CampaignError(f"operational-safe actor {field} changed")
    return value


def _iteration_path(root: pathlib.Path, name: str) -> pathlib.Path:
    return root / "iteration" / name


def authorize_iteration(
    root: pathlib.Path, *, failure_record_path: pathlib.Path,
    safe_actor_record_path: pathlib.Path, learning_rate: float,
    authorized_at_utc: str,
) -> dict[str, Any]:
    path = _iteration_path(root, "00-authorization.json")
    if path.exists():
        raise CampaignError("the single 10,000-game iteration is already authorized/spent")
    failure = validate_offline_family_failure(failure_record_path)
    safe_actor = validate_operational_safe_actor(safe_actor_record_path)
    failure_reference = _sealed_file_reference(
        failure_record_path, OFFLINE_FAMILY_FAILURE_SCHEMA,
        label="offline-family failure",
    )
    safe_actor_reference = _sealed_file_reference(
        safe_actor_record_path, OPERATIONAL_SAFE_ACTOR_SCHEMA,
        label="operational-safe actor",
    )
    if (
        safe_actor.get("offline_family_failure") != failure_reference
        or failure.get("iteration_authorizable") is not True
    ):
        raise CampaignError("safe actor and offline-family failure are not bound")
    lr = _finite(learning_rate, "sample-scaled learning rate")
    if not 0.0 < lr <= ITERATION_SPEC["maximum_sample_scaled_learning_rate"]:
        raise CampaignError("sample-scaled learning rate exceeds 6e-5")
    base._utc(authorized_at_utc, "iteration authorization timestamp")
    return base.write_sealed(path, {
        "schema": ITERATION_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "status": "one-iteration-authorized",
        "one_shot": True,
        "authorized_at_utc": authorized_at_utc,
        "offline_family_failure": failure_reference,
        "operational_safe_actor": safe_actor_reference,
        "sample_scaled_learning_rate": lr,
        "specification": ITERATION_SPEC,
    })


def validate_iteration_authorization(path: pathlib.Path) -> dict[str, Any]:
    authorization = base.load_sealed(path, ITERATION_AUTH_SCHEMA)
    if (
        set(authorization) != {
            "schema", "namespace", "status", "one_shot", "authorized_at_utc",
            "offline_family_failure", "operational_safe_actor",
            "sample_scaled_learning_rate", "specification", "body_sha256",
        }
        or authorization.get("namespace") != NAMESPACE
        or authorization.get("status") != "one-iteration-authorized"
        or authorization.get("one_shot") is not True
        or authorization.get("specification") != ITERATION_SPEC
    ):
        raise CampaignError("iteration authorization contract changed")
    base._utc(authorization.get("authorized_at_utc"), "iteration authorization timestamp")
    lr = _finite(
        authorization.get("sample_scaled_learning_rate"),
        "sample-scaled learning rate",
    )
    if not 0.0 < lr <= ITERATION_SPEC["maximum_sample_scaled_learning_rate"]:
        raise CampaignError("sample-scaled learning rate exceeds 6e-5")
    failure_path, _failure = _validate_sealed_file_reference(
        authorization.get("offline_family_failure"),
        OFFLINE_FAMILY_FAILURE_SCHEMA, label="offline-family failure",
    )
    safe_path, safe_actor = _validate_sealed_file_reference(
        authorization.get("operational_safe_actor"),
        OPERATIONAL_SAFE_ACTOR_SCHEMA, label="operational-safe actor",
    )
    validate_offline_family_failure(failure_path)
    validate_operational_safe_actor(safe_path)
    if safe_actor.get("offline_family_failure") != authorization[
        "offline_family_failure"
    ]:
        raise CampaignError("iteration authorization artifacts are not mutually bound")
    return authorization


def start_iteration(
    root: pathlib.Path, *, environment: Mapping[str, Any], started_at_utc: str,
) -> dict[str, Any]:
    authorization_path = _iteration_path(root, "00-authorization.json")
    authorization = validate_iteration_authorization(authorization_path)
    path = _iteration_path(root, "01-started.json")
    if path.exists():
        raise CampaignError("the one-shot iteration has already started")
    required = {
        "interactive_launch_agent": True,
        "resume": True,
        "blas_threads": 1,
        "ac_power": True,
    }
    if any(environment.get(key) != value for key, value in required.items()):
        raise CampaignError("iteration environment lacks LaunchAgent/resume/BLAS/AC requirements")
    disk = _finite(environment.get("free_disk_gib"), "free disk GiB")
    if disk < 20.0:
        raise CampaignError("iteration requires at least 20 GiB free disk")
    return base.write_sealed(path, {
        "schema": ITERATION_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "iteration-started", "started_at_utc": started_at_utc,
        "authorization": _iteration_artifact_reference(
            authorization_path, ITERATION_AUTH_SCHEMA
        ),
        "environment": dict(environment),
    })


def complete_iteration(
    root: pathlib.Path, *, result: Mapping[str, Any], completed_at_utc: str,
) -> dict[str, Any]:
    authorization_path = _iteration_path(root, "00-authorization.json")
    started_path = _iteration_path(root, "01-started.json")
    authorization = validate_iteration_authorization(authorization_path)
    base.load_sealed(started_path, ITERATION_EVENT_SCHEMA)
    path = _iteration_path(root, "02-completed.json")
    if path.exists():
        raise CampaignError("the one-shot iteration already completed")
    if (result.get("games") != ITERATION_SPEC["games"]
            or result.get("total_games") != 10_000
            or result.get("positions_per_game") != 20
            or result.get("workers") != 10
            or result.get("fixed_work") is not True
            or float(result.get("deep_relabel_fraction", -1)) != 0.25
            or result.get("resumed") is not True):
        raise CampaignError("iteration completion contradicts the exact 10,000-game specification")
    _sha(result.get("float_checkpoint_sha256"), "fine-tuned checkpoint SHA-256")
    if _finite(result.get("learning_rate"), "completion learning rate") != authorization[
            "sample_scaled_learning_rate"]:
        raise CampaignError("iteration learning rate changed")
    return base.write_sealed(path, {
        "schema": ITERATION_EVENT_SCHEMA, "namespace": NAMESPACE,
        "status": "iteration-completed", "completed_at_utc": completed_at_utc,
        "authorization": _iteration_artifact_reference(
            authorization_path, ITERATION_AUTH_SCHEMA
        ),
        "start": _iteration_artifact_reference(started_path, ITERATION_EVENT_SCHEMA),
        "result": dict(result),
        "iterations_remaining": 0,
    })


def _validate_completed_iteration(root: pathlib.Path) -> dict[str, Any]:
    authorization_path = _iteration_path(root, "00-authorization.json")
    started_path = _iteration_path(root, "01-started.json")
    completed_path = _iteration_path(root, "02-completed.json")
    authorization = validate_iteration_authorization(authorization_path)
    started = base.load_sealed(started_path, ITERATION_EVENT_SCHEMA)
    completed = base.load_sealed(completed_path, ITERATION_EVENT_SCHEMA)
    environment = started.get("environment")
    required_environment = {
        "interactive_launch_agent": True,
        "resume": True,
        "blas_threads": 1,
        "ac_power": True,
    }
    result = completed.get("result")
    if (
        started.get("status") != "iteration-started"
        or started.get("authorization")
        != _iteration_artifact_reference(authorization_path, ITERATION_AUTH_SCHEMA)
        or not isinstance(environment, dict)
        or any(
            environment.get(field) != expected
            for field, expected in required_environment.items()
        )
        or _finite(environment.get("free_disk_gib"), "free disk GiB") < 20.0
        or completed.get("status") != "iteration-completed"
        or completed.get("authorization")
        != _iteration_artifact_reference(authorization_path, ITERATION_AUTH_SCHEMA)
        or completed.get("start")
        != _iteration_artifact_reference(started_path, ITERATION_EVENT_SCHEMA)
        or completed.get("iterations_remaining") != 0
        or not isinstance(result, dict)
        or result.get("games") != ITERATION_SPEC["games"]
        or result.get("total_games") != 10_000
        or result.get("positions_per_game") != 20
        or result.get("workers") != 10
        or result.get("fixed_work") is not True
        or float(result.get("deep_relabel_fraction", -1)) != 0.25
        or result.get("resumed") is not True
        or _finite(result.get("learning_rate"), "completion learning rate")
        != authorization["sample_scaled_learning_rate"]
    ):
        raise CampaignError("completed iteration chain is invalid")
    _sha(result.get("float_checkpoint_sha256"), "fine-tuned checkpoint SHA-256")
    return completed


def record_post_iteration_failure(
    root: pathlib.Path, *, stage: str, evidence_path: pathlib.Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    completed_path = _iteration_path(root, "02-completed.json")
    _validate_completed_iteration(root)
    if stage not in POST_ITERATION_FAILURE_STAGES:
        raise CampaignError("post-iteration failure stage is outside the terminal roster")
    # Stage-specific gate tools own their evidence schemas.  Requiring a sealed
    # artifact here makes the terminal decision immutable without duplicating
    # each gate's validator in this governance module.
    evidence = base.load_sealed(evidence_path)
    evidence_schema = evidence.get("schema")
    if not isinstance(evidence_schema, str) or not evidence_schema:
        raise CampaignError("post-iteration failure evidence schema is invalid")
    base._utc(recorded_at_utc, "post-iteration failure timestamp")
    path = _iteration_path(root, "03-post-iteration-failure.json")
    if path.exists():
        raise CampaignError("the post-iteration family failure is already recorded")
    return base.write_sealed(path, {
        "schema": POST_ITERATION_FAILURE_SCHEMA,
        "namespace": NAMESPACE,
        "status": "post-iteration-family-failure",
        "failed": True,
        "stage": stage,
        "recorded_at_utc": recorded_at_utc,
        "iteration": _sealed_file_reference(
            completed_path, ITERATION_EVENT_SCHEMA,
            label="completed one-shot iteration",
        ),
        "evidence": _sealed_file_reference(
            evidence_path, evidence_schema,
            label="post-iteration failure evidence",
        ),
        "iterations_remaining": 0,
        "upload_authorized": False,
    })


def validate_post_iteration_failure(
    root: pathlib.Path, path: pathlib.Path,
) -> dict[str, Any]:
    expected_path = _iteration_path(root, "03-post-iteration-failure.json").resolve()
    if path.resolve() != expected_path:
        raise CampaignError("family exhaustion requires the canonical post-iteration failure")
    value = base.load_sealed(path, POST_ITERATION_FAILURE_SCHEMA)
    if (
        set(value) != {
            "schema", "namespace", "status", "failed", "stage",
            "recorded_at_utc", "iteration", "evidence", "iterations_remaining",
            "upload_authorized", "body_sha256",
        }
        or value.get("namespace") != NAMESPACE
        or value.get("status") != "post-iteration-family-failure"
        or value.get("failed") is not True
        or value.get("stage") not in POST_ITERATION_FAILURE_STAGES
        or value.get("iterations_remaining") != 0
        or value.get("upload_authorized") is not False
    ):
        raise CampaignError("post-iteration failure contract changed")
    base._utc(value.get("recorded_at_utc"), "post-iteration failure timestamp")
    completed_path, completed = _validate_sealed_file_reference(
        value.get("iteration"), ITERATION_EVENT_SCHEMA,
        label="completed one-shot iteration",
    )
    if (
        completed_path != _iteration_path(root, "02-completed.json").resolve()
        or completed.get("status") != "iteration-completed"
        or completed.get("iterations_remaining") != 0
    ):
        raise CampaignError("post-iteration failure does not bind the completion")
    if _validate_completed_iteration(root) != completed:
        raise CampaignError("post-iteration failure completion chain changed")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("schema"), str):
        raise CampaignError("post-iteration failure evidence reference is malformed")
    _validate_sealed_file_reference(
        evidence, evidence["schema"], label="post-iteration failure evidence",
    )
    return value


def record_family_exhausted(
    root: pathlib.Path, *, post_iteration_failure_path: pathlib.Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    completed_path = _iteration_path(root, "02-completed.json")
    _validate_completed_iteration(root)
    failure = validate_post_iteration_failure(root, post_iteration_failure_path)
    base._utc(recorded_at_utc, "family exhaustion timestamp")
    return base.write_sealed(root / "compact-value-family-exhausted.json", {
        "schema": FAMILY_EXHAUSTED_SCHEMA, "namespace": NAMESPACE,
        "status": "compact-value-family-exhausted",
        "recorded_at_utc": recorded_at_utc,
        "iteration": _sealed_file_reference(
            completed_path, ITERATION_EVENT_SCHEMA,
            label="completed one-shot iteration",
        ),
        "post_iteration_failure": _sealed_file_reference(
            post_iteration_failure_path, POST_ITERATION_FAILURE_SCHEMA,
            label="post-iteration failure",
        ),
        "failure_stage": failure["stage"],
        "iterations_remaining": 0,
        "upload_authorized": False,
        "goal_complete": False,
    })


def read_numeric_ids(path: pathlib.Path) -> list[int]:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise CampaignError(f"ID-only input is unreadable/non-ASCII: {path}") from error
    ids = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if re.fullmatch(r"[1-9][0-9]*", line) is None:
            raise CampaignError(f"ID-only input has nonnumeric content at line {line_number}")
        ids.append(int(line))
    if not ids or ids != sorted(set(ids)):
        raise CampaignError("ID-only input must be nonempty, sorted, and unique")
    return ids


def freeze_preupload_exclusions(
    output: pathlib.Path, *, source_binding_path: pathlib.Path,
    id_files: Sequence[pathlib.Path], frozen_at_utc: str,
) -> dict[str, Any]:
    binding = base.load_sealed(source_binding_path, base.SOURCE_BINDING_SCHEMA)
    base.validate_source_binding(binding)
    if not id_files:
        raise CampaignError("pre-upload exclusions require explicit ID-only inputs")
    combined: set[int] = set()
    id_sources: dict[int, set[str]] = {}
    sources = []
    for path in id_files:
        ids = read_numeric_ids(path)
        combined.update(ids)
        source_sha = base.sha256_file(path)
        for game_id in ids:
            id_sources.setdefault(game_id, set()).add(source_sha)
        sources.append({
            "sha256": source_sha,
            "id_count": len(ids),
        })
    return base.write_sealed(output, {
        "schema": EXCLUSION_SCHEMA, "namespace": NAMESPACE,
        "status": "frozen-before-upload",
        "frozen_at_utc": frozen_at_utc,
        "candidate_commit": binding["candidate_commit"],
        "candidate_sha256": binding["candidate"]["sha256"],
        "source_binding": base.artifact_reference(
            source_binding_path, base.SOURCE_BINDING_SCHEMA
        ),
        "sources": sorted(sources, key=lambda item: item["sha256"]),
        "game_ids": sorted(combined),
        "records": [
            {
                "game_id": game_id,
                "categories": ["pre-upload-exclusion"],
                "sources": sorted(id_sources[game_id]),
            }
            for game_id in sorted(combined)
        ],
        "contains_only_game_ids": True,
        "replay_payloads_accessed": False,
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--input", type=pathlib.Path, required=True)
    select.add_argument("--output", type=pathlib.Path, required=True)
    tests = commands.add_parser("authorize-tests")
    tests.add_argument("--selection", type=pathlib.Path, required=True)
    tests.add_argument("--artifact", type=pathlib.Path, required=True)
    tests.add_argument("--authorized-at-utc", required=True)
    tests.add_argument("--output", type=pathlib.Path, required=True)
    family_failure = commands.add_parser("record-offline-family-failure")
    family_failure.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    family_failure.add_argument(
        "--run-output-directory", type=pathlib.Path, required=True,
    )
    family_failure.add_argument("--run-reference", type=pathlib.Path, required=True)
    family_failure.add_argument("--output-directory", type=pathlib.Path, required=True)
    family_failure.add_argument("--recorded-at-utc", required=True)
    safe_actor = commands.add_parser("record-operational-safe-actor")
    safe_actor.add_argument("--failure-record", type=pathlib.Path, required=True)
    safe_actor.add_argument("--family-selection", type=pathlib.Path, required=True)
    safe_actor.add_argument("--float-checkpoint", type=pathlib.Path, required=True)
    safe_actor.add_argument("--runtime", type=pathlib.Path, required=True)
    safe_actor.add_argument("--generated-source", type=pathlib.Path, required=True)
    safe_actor.add_argument("--protocol-evidence", type=pathlib.Path, required=True)
    safe_actor.add_argument("--build-evidence", type=pathlib.Path, required=True)
    safe_actor.add_argument("--timing-evidence", type=pathlib.Path, required=True)
    safe_actor.add_argument("--output-directory", type=pathlib.Path, required=True)
    safe_actor.add_argument("--recorded-at-utc", required=True)
    iteration = commands.add_parser("authorize-iteration")
    iteration.add_argument("--root", type=pathlib.Path, required=True)
    iteration.add_argument("--failure-record", type=pathlib.Path, required=True)
    iteration.add_argument("--safe-actor-record", type=pathlib.Path, required=True)
    iteration.add_argument("--learning-rate", type=float, required=True)
    iteration.add_argument("--authorized-at-utc", required=True)
    iteration_start = commands.add_parser("start-iteration")
    iteration_start.add_argument("--root", type=pathlib.Path, required=True)
    iteration_start.add_argument("--environment", type=pathlib.Path, required=True)
    iteration_start.add_argument("--started-at-utc", required=True)
    iteration_complete = commands.add_parser("complete-iteration")
    iteration_complete.add_argument("--root", type=pathlib.Path, required=True)
    iteration_complete.add_argument("--result", type=pathlib.Path, required=True)
    iteration_complete.add_argument("--completed-at-utc", required=True)
    post_failure = commands.add_parser("record-post-iteration-failure")
    post_failure.add_argument("--root", type=pathlib.Path, required=True)
    post_failure.add_argument(
        "--stage", choices=sorted(POST_ITERATION_FAILURE_STAGES), required=True,
    )
    post_failure.add_argument("--evidence", type=pathlib.Path, required=True)
    post_failure.add_argument("--recorded-at-utc", required=True)
    exhausted = commands.add_parser("family-exhausted")
    exhausted.add_argument("--root", type=pathlib.Path, required=True)
    exhausted.add_argument("--failure", type=pathlib.Path, required=True)
    exhausted.add_argument("--recorded-at-utc", required=True)
    exclusions = commands.add_parser("freeze-exclusions")
    exclusions.add_argument("--source-binding", type=pathlib.Path, required=True)
    exclusions.add_argument("--id-file", type=pathlib.Path, action="append", required=True)
    exclusions.add_argument("--frozen-at-utc", required=True)
    exclusions.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        result_path: pathlib.Path | None = None
        if args.command == "select":
            result = select_development(
                args.output, json.loads(args.input.read_text(encoding="utf-8"))
            )
        elif args.command == "authorize-tests":
            result = authorize_protected_tests(
                args.output, selection_path=args.selection,
                quantized_artifact_path=args.artifact,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "record-offline-family-failure":
            result = record_offline_family_failure(
                args.output_directory,
                bundle_manifest=args.bundle_manifest,
                run_output_directory=args.run_output_directory,
                run_reference=args.run_reference,
                recorded_at_utc=args.recorded_at_utc,
            )
            result_path = offline_family_failure_path(args.output_directory, result)
        elif args.command == "record-operational-safe-actor":
            result = record_operational_safe_actor(
                args.output_directory,
                failure_record_path=args.failure_record,
                family_selection_path=args.family_selection,
                float_checkpoint_path=args.float_checkpoint,
                runtime_path=args.runtime,
                generated_source_path=args.generated_source,
                protocol_evidence_path=args.protocol_evidence,
                build_evidence_path=args.build_evidence,
                timing_evidence_path=args.timing_evidence,
                recorded_at_utc=args.recorded_at_utc,
            )
            result_path = operational_safe_actor_path(args.output_directory, result)
        elif args.command == "authorize-iteration":
            result = authorize_iteration(
                args.root,
                failure_record_path=args.failure_record,
                safe_actor_record_path=args.safe_actor_record,
                learning_rate=args.learning_rate,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "start-iteration":
            result = start_iteration(
                args.root,
                environment=json.loads(args.environment.read_text(encoding="utf-8")),
                started_at_utc=args.started_at_utc,
            )
        elif args.command == "complete-iteration":
            result = complete_iteration(
                args.root,
                result=json.loads(args.result.read_text(encoding="utf-8")),
                completed_at_utc=args.completed_at_utc,
            )
        elif args.command == "record-post-iteration-failure":
            result = record_post_iteration_failure(
                args.root, stage=args.stage, evidence_path=args.evidence,
                recorded_at_utc=args.recorded_at_utc,
            )
        elif args.command == "family-exhausted":
            result = record_family_exhausted(
                args.root,
                post_iteration_failure_path=args.failure,
                recorded_at_utc=args.recorded_at_utc,
            )
        else:
            result = freeze_preupload_exclusions(
                args.output, source_binding_path=args.source_binding,
                id_files=args.id_file, frozen_at_utc=args.frozen_at_utc,
            )
        printed: Any = result
        if result_path is not None:
            printed = {"path": str(result_path.resolve()), "artifact": result}
        print(json.dumps(printed, sort_keys=True, allow_nan=False))
        return 0
    except (CampaignError, OSError, json.JSONDecodeError) as error:
        print(f"compact campaign failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
