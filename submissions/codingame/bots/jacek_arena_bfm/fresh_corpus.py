#!/usr/bin/env python3
"""Schemas and strict validation for campaign-owned fresh training rows.

This module consumes only JSONL rows created by the current
``jacek_arena_bfm`` campaign.  It never discovers or opens historical replay,
model, action, or label files.  Pre-campaign evidence is represented only by a
game-ID exclusion registry assembled from metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from .campaign_provenance import (
        arena_derivation_usage,
        load_id_only_registry,
        validate_arena_derivation,
        validate_window_plan,
    )
    from .immutable_artifacts import (
        canonical_json_bytes,
        sha256_file,
        verify_content_addressed_path,
    )
except ImportError:  # pragma: no cover - standalone CLI execution
    from campaign_provenance import (
        arena_derivation_usage,
        load_id_only_registry,
        validate_arena_derivation,
        validate_window_plan,
    )
    from immutable_artifacts import (
        canonical_json_bytes,
        sha256_file,
        verify_content_addressed_path,
    )


NAMESPACE = "jacek_arena_bfm"
FEATURE_COUNT = 1156
CORPUS_SCHEMA = "papersoccer.jacek-arena-bfm.corpus-row.v1"
CORPUS_MANIFEST_SCHEMA = "papersoccer.jacek-arena-bfm.corpus-manifest.v1"
WINDOW_PLAN_SCHEMA = "papersoccer.jacek-arena-bfm.window-plan.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"

VALUE_SOURCES = frozenset({
    "scratch_selfplay",
    "arena_terminal",
    "arena_reanalysis",
    "arena_counterfactual",
})
ARENA_VALUE_SOURCES = frozenset({
    "arena_terminal",
    "arena_reanalysis",
    "arena_counterfactual",
})
PAIR_SOURCE = "arena_opponent_ranking"
TRAINABLE_WINDOW_ROLES = frozenset({"training"})
NONTRAINING_WINDOW_ROLES = frozenset(
    {"arena-validation", "final-holdout", "rollback-accounting"}
)
OPENING_DEPTHS = frozenset({0, 4, 8, 12})
APPROVED_WINDOW_PLAN_SHA256 = "0ca942c0a05af9e80197b5c95ea3ef32aed78192753f313a1015a22bd3b90d29"
APPROVED_EXCLUSION_SHA256 = "0ce7cff1d29cbceab1d06a9066eebb6647a31515437cd468ff1a9b1b5f8407f9"


class CorpusValidationError(ValueError):
    """Raised when a row could admit forbidden or operationally unsafe evidence."""


def parse_utc(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise CorpusValidationError(f"{field} must be a non-empty UTC timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as error:
        raise CorpusValidationError(f"{field} is not ISO-8601: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise CorpusValidationError(f"{field} must carry an explicit UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def _require_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CorpusValidationError(f"{field} must be a non-empty string")
    return value


def _require_bool(row: Mapping[str, Any], field: str, expected: bool | None = None) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise CorpusValidationError(f"{field} must be boolean")
    if expected is not None and value is not expected:
        raise CorpusValidationError(f"{field} must be {str(expected).lower()}")
    return value


def _require_number(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CorpusValidationError(f"{field} must be a finite number")
    return float(value)


def _require_sha256(row: Mapping[str, Any], field: str) -> str:
    value = _require_string(row, field)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CorpusValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_features(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != FEATURE_COUNT:
        raise CorpusValidationError(f"{field} must contain exactly {FEATURE_COUNT} values")
    output: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1):
            raise CorpusValidationError(
                f"{field}[{index}] must be the integer 0 or 1 for the sparse binary representation"
            )
        output.append(item)
    return tuple(output)


@dataclasses.dataclass(frozen=True)
class CampaignContract:
    campaign_id: str
    t0_utc: dt.datetime
    window_roles: Mapping[str, str] = dataclasses.field(default_factory=dict)
    arena_freeze_cutoff_utc: dt.datetime | None = None

    @classmethod
    def from_window_plan(cls, plan: Mapping[str, Any]) -> "CampaignContract":
        if plan.get("schema") != WINDOW_PLAN_SCHEMA:
            raise CorpusValidationError(f"window plan schema must be {WINDOW_PLAN_SCHEMA!r}")
        campaign = plan.get("campaign")
        if not isinstance(campaign, Mapping):
            raise CorpusValidationError("window plan campaign must be an object")
        if campaign.get("namespace") != NAMESPACE:
            raise CorpusValidationError(f"window plan namespace must be {NAMESPACE!r}")
        campaign_id = campaign.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            # The campaign T0 is an immutable, unambiguous fallback identity for
            # early plans that predate the explicit campaign_id field.
            campaign_id = f"{NAMESPACE}@{campaign.get('t0_utc', '')}"
        freeze = campaign.get("arena_freeze_cutoff_utc")
        roles: dict[str, str] = {}
        windows = plan.get("windows")
        if not isinstance(windows, list):
            raise CorpusValidationError("window plan windows must be an array")
        for window in windows:
            if not isinstance(window, Mapping):
                raise CorpusValidationError("each planned window must be an object")
            window_id = _require_string(window, "window_id")
            role = _require_string(window, "role")
            if role not in TRAINABLE_WINDOW_ROLES | NONTRAINING_WINDOW_ROLES:
                raise CorpusValidationError(f"unknown window role {role!r}")
            if window_id in roles:
                raise CorpusValidationError(f"duplicate window_id {window_id!r}")
            roles[window_id] = role
        return cls(
            campaign_id=campaign_id,
            t0_utc=parse_utc(campaign.get("t0_utc"), "campaign.t0_utc"),
            window_roles=roles,
            arena_freeze_cutoff_utc=parse_utc(freeze, "campaign.arena_freeze_cutoff_utc") if freeze else None,
        )


@dataclasses.dataclass(frozen=True)
class ValidatedRow:
    kind: str
    source_kind: str
    sample_id: str
    game_id: str
    split_game_id: str
    features: tuple[int, ...] | None
    target: float | None
    preferred_features: tuple[int, ...] | None
    inferior_features: tuple[int, ...] | None
    weight: float
    is_arena: bool
    raw: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class ArenaGameBinding:
    game_id: str
    derivation_sha256: str
    record_sha256: str
    raw_sha256: str
    normalized_sha256: str
    window_id: str
    window_role: str
    source_sha256: str
    agent_id: str
    submission_id: str
    opponent_frozen_rank: int | None
    ranking_candidate_weight: float
    uses: frozenset[str]


@dataclasses.dataclass(frozen=True)
class CorpusSummary:
    rows: int
    games: int
    counts_by_kind: Mapping[str, int]
    counts_by_source: Mapping[str, int]
    counts_by_window: Mapping[str, int]
    file_sha256: str | None = None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_excluded_game_ids(
    path: Path | str,
    expected_sha256: str = APPROVED_EXCLUSION_SHA256,
) -> frozenset[str]:
    """Read game IDs from metadata only; replay payload fields are refused."""

    try:
        registry = load_id_only_registry(Path(path), expected_sha256)
    except ValueError as error:
        raise CorpusValidationError(str(error)) from error
    return frozenset(str(record["game_id"]) for record in registry["records"])


def load_producer_source_hashes(
    paths: Sequence[Path | str],
    *,
    campaign_root: Path | str,
) -> frozenset[str]:
    if not paths:
        raise CorpusValidationError("at least one content-addressed fresh producer source is required")
    hashes: set[str] = set()
    for raw_path in paths:
        path = require_within_campaign_root(raw_path, campaign_root)
        if path.suffix != ".source":
            raise CorpusValidationError(f"producer source must use the .source suffix: {path}")
        try:
            digest = verify_content_addressed_path(path)
        except ValueError as error:
            raise CorpusValidationError(str(error)) from error
        hashes.add(digest)
    return frozenset(hashes)


def load_arena_game_bindings(
    paths: Sequence[Path | str],
    *,
    campaign_root: Path | str,
    repository: Path | str,
) -> dict[str, ArenaGameBinding]:
    bindings: dict[str, ArenaGameBinding] = {}
    for raw_path in paths:
        path = require_within_campaign_root(raw_path, campaign_root)
        try:
            derivation = validate_arena_derivation(path, repository=Path(repository).resolve())
        except ValueError as error:
            raise CorpusValidationError(str(error)) from error
        derivation_sha256 = sha256_file(path)
        if derivation["window_plan"]["sha256"] != APPROVED_WINDOW_PLAN_SHA256:
            raise CorpusValidationError("arena derivation uses an unapproved window plan")
        if derivation["exclusion_registry"]["sha256"] != APPROVED_EXCLUSION_SHA256:
            raise CorpusValidationError("arena derivation uses an unapproved exclusion registry")
        if derivation["window"]["role"] != "training":
            raise CorpusValidationError("non-training arena derivations cannot back corpus rows")
        usage = arena_derivation_usage(derivation)
        if not usage["training_eligible"]:
            reason = (
                "an entire submission window with any focus operational failure is forbidden for training"
                if usage["window_disposition"] == "rejected-entire-window"
                else "arena derivation contains no eligible training games"
            )
            raise CorpusValidationError(
                f"{reason}; disposition={usage['window_disposition']}"
            )
        source = derivation["source"]
        for game in derivation["games"]:
            if game["disposition"] != "eligible":
                continue
            game_id = str(game["game_id"])
            if game_id in bindings:
                raise CorpusValidationError(f"arena game {game_id} appears in multiple derivations")
            bindings[game_id] = ArenaGameBinding(
                game_id=game_id,
                derivation_sha256=derivation_sha256,
                record_sha256=str(game["record_sha256"]),
                raw_sha256=str(game["raw_sha256"]),
                normalized_sha256=str(game["normalized_sha256"]),
                window_id=str(derivation["window"]["window_id"]),
                window_role=str(derivation["window"]["role"]),
                source_sha256=str(source["sha256"]),
                agent_id=str(source["agent_id"]),
                submission_id=str(source["submission_id"]),
                opponent_frozen_rank=game["opponent_frozen_rank"],
                ranking_candidate_weight=float(game["ranking_candidate_weight"]),
                uses=frozenset(game["uses"]),
            )
    return bindings


class FreshCorpusValidator:
    def __init__(
        self,
        contract: CampaignContract,
        *,
        excluded_game_ids: Iterable[str | int] = (),
        approved_producer_source_sha256: Iterable[str] = (),
        arena_game_bindings: Mapping[str, ArenaGameBinding] | None = None,
        training_only: bool = True,
        max_pairs_per_decision: int = 4,
        max_pairs_per_game: int = 32,
    ) -> None:
        self.contract = contract
        self.excluded_game_ids = frozenset(str(value) for value in excluded_game_ids)
        if not self.excluded_game_ids:
            raise CorpusValidationError("the sealed pre-T0 exclusion registry is mandatory")
        self.approved_producer_source_sha256 = frozenset(approved_producer_source_sha256)
        if not self.approved_producer_source_sha256:
            raise CorpusValidationError("at least one fresh producer source identity is mandatory")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.approved_producer_source_sha256
        ):
            raise CorpusValidationError("approved producer identities must be lowercase SHA-256 values")
        self.arena_game_bindings = dict(arena_game_bindings or {})
        self.training_only = training_only
        self.max_pairs_per_decision = max_pairs_per_decision
        self.max_pairs_per_game = max_pairs_per_game

    def validate_row(self, row: Mapping[str, Any]) -> ValidatedRow:
        if not isinstance(row, Mapping):
            raise CorpusValidationError("corpus row must be an object")
        if row.get("schema") != CORPUS_SCHEMA:
            raise CorpusValidationError(f"row schema must be {CORPUS_SCHEMA!r}")
        if row.get("namespace") != NAMESPACE:
            raise CorpusValidationError(f"row namespace must be {NAMESPACE!r}")
        if row.get("campaign_id") != self.contract.campaign_id:
            raise CorpusValidationError("row campaign_id does not match the active campaign")

        sample_id = _require_string(row, "sample_id")
        game_value = row.get("game_id")
        if isinstance(game_value, bool) or not isinstance(game_value, (str, int)):
            raise CorpusValidationError("game_id must be a string or integer")
        game_id = str(game_value)
        if game_id in self.excluded_game_ids:
            raise CorpusValidationError(f"game_id {game_id!r} predates T0 or is otherwise excluded")

        generated_at = parse_utc(row.get("generated_at_utc"), "generated_at_utc")
        evidence_at = parse_utc(row.get("evidence_at_utc"), "evidence_at_utc")
        if generated_at < self.contract.t0_utc or evidence_at < self.contract.t0_utc:
            raise CorpusValidationError("all row evidence and generation must be at or after campaign T0")
        if generated_at < evidence_at:
            raise CorpusValidationError("generated_at_utc cannot precede evidence_at_utc")
        producer_sha256 = _require_sha256(row, "producer_source_sha256")
        if producer_sha256 not in self.approved_producer_source_sha256:
            raise CorpusValidationError("row producer source is not an approved fresh campaign artifact")
        _require_sha256(row, "evidence_sha256")
        if row.get("representation") != "mover_relative_316_edges_plus_105x8_distance_v1":
            raise CorpusValidationError("row must use the frozen mover-relative 1156-feature representation")
        weight = _require_number(row, "weight")
        if not (0.0 < weight <= 1.0):
            raise CorpusValidationError("weight must be in (0, 1]")

        kind = _require_string(row, "kind")
        source_kind = _require_string(row, "source_kind")
        if kind == "value":
            if source_kind not in VALUE_SOURCES:
                raise CorpusValidationError(f"invalid value source_kind {source_kind!r}")
            features = _validate_features(row.get("features"), "features")
            target = _require_number(row, "target")
            if not -1.0 <= target <= 1.0:
                raise CorpusValidationError("value target must be in [-1, 1]")
            label_method = _require_string(row, "label_method")
            if label_method not in {"terminal_outcome", "exact", "stable_reanalysis", "counterfactual_outcome"}:
                raise CorpusValidationError(f"unsupported label_method {label_method!r}")
            arena_binding: ArenaGameBinding | None = None
            if source_kind == "scratch_selfplay":
                self._validate_scratch(row, label_method)
            else:
                arena_binding = self._validate_arena_common(row, game_id, source_kind)
                _require_string(row, "position_id")
                if source_kind == "arena_terminal" and label_method != "terminal_outcome":
                    raise CorpusValidationError("arena_terminal rows require terminal_outcome labels")
                if source_kind == "arena_reanalysis" and label_method not in {"exact", "stable_reanalysis"}:
                    raise CorpusValidationError("arena_reanalysis rows require exact or stable_reanalysis labels")
                if source_kind == "arena_counterfactual" and label_method != "counterfactual_outcome":
                    raise CorpusValidationError("counterfactual rows require their own continuation outcome")
                if source_kind == "arena_terminal":
                    _require_bool(row, "theoretical_value_claim", False)
                    if weight > 0.5:
                        raise CorpusValidationError(
                            "raw arena trajectory outcomes must be downweighted to at most 0.5"
                        )
                if source_kind == "arena_counterfactual":
                    self._validate_counterfactual(row, game_id, arena_binding)
            return ValidatedRow(
                kind=kind,
                source_kind=source_kind,
                sample_id=sample_id,
                game_id=game_id,
                split_game_id=game_id if arena_binding is None else arena_binding.game_id,
                features=features,
                target=target,
                preferred_features=None,
                inferior_features=None,
                weight=weight,
                is_arena=source_kind in ARENA_VALUE_SOURCES,
                raw=row,
            )

        if kind == "pairwise":
            if source_kind != PAIR_SOURCE:
                raise CorpusValidationError(f"pairwise source_kind must be {PAIR_SOURCE!r}")
            arena_binding = self._validate_arena_common(row, game_id, source_kind)
            self._validate_pairwise(row, weight, arena_binding)
            return ValidatedRow(
                kind=kind,
                source_kind=source_kind,
                sample_id=sample_id,
                game_id=game_id,
                split_game_id=arena_binding.game_id,
                features=None,
                target=None,
                preferred_features=_validate_features(row.get("preferred_features"), "preferred_features"),
                inferior_features=_validate_features(row.get("inferior_features"), "inferior_features"),
                weight=weight,
                is_arena=True,
                raw=row,
            )
        raise CorpusValidationError("kind must be 'value' or 'pairwise'")

    def _validate_scratch(self, row: Mapping[str, Any], label_method: str) -> None:
        if row.get("window_id") is not None or row.get("submission_id") is not None:
            raise CorpusValidationError("scratch rows cannot be bound to an arena window/submission")
        depth = row.get("opening_depth")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth not in OPENING_DEPTHS:
            raise CorpusValidationError(f"scratch opening_depth must be one of {sorted(OPENING_DEPTHS)}")
        if row.get("initialization") != "random":
            raise CorpusValidationError("scratch games must originate from random initialization")
        checkpoint_inputs = row.get("checkpoint_inputs")
        if checkpoint_inputs != []:
            raise CorpusValidationError("scratch rows must declare an empty checkpoint_inputs list")
        if label_method not in {"terminal_outcome", "exact", "stable_reanalysis"}:
            raise CorpusValidationError("scratch value labels must be terminal, exact, or stable reanalysis")

    def _validate_arena_common(
        self,
        row: Mapping[str, Any],
        game_id: str,
        source_kind: str,
    ) -> ArenaGameBinding:
        _require_bool(row, "operational_clean", True)
        _require_bool(row, "complete_transcript", True)
        _require_bool(row, "terminal_unambiguous", True)
        _require_bool(row, "timeout", False)
        _require_bool(row, "illegal_action", False)
        _require_bool(row, "malformed_transcript", False)
        raw_sha256 = _require_sha256(row, "raw_sha256")
        normalized_sha256 = _require_sha256(row, "normalized_sha256")
        submitted_source_sha256 = _require_sha256(row, "submitted_source_sha256")
        agent_id = _require_string(row, "agent_id")
        submission_id = _require_string(row, "submission_id")
        arena_game_value = row.get("arena_game_id")
        if isinstance(arena_game_value, bool) or not isinstance(arena_game_value, int) or arena_game_value <= 0:
            raise CorpusValidationError("arena_game_id must be a positive integer")
        arena_game_id = str(arena_game_value)
        if arena_game_id in self.excluded_game_ids:
            raise CorpusValidationError(f"arena_game_id {arena_game_id!r} predates T0 or is excluded")
        if source_kind != "arena_counterfactual" and game_id != arena_game_id:
            raise CorpusValidationError("non-counterfactual arena rows must use arena_game_id as game_id")
        binding = self.arena_game_bindings.get(arena_game_id)
        if binding is None:
            raise CorpusValidationError(
                f"arena game {arena_game_id} is absent from approved operationally clean derivations"
            )
        derivation_sha256 = _require_sha256(row, "arena_derivation_sha256")
        record_sha256 = _require_sha256(row, "arena_record_sha256")
        window_id = _require_string(row, "window_id")
        role = _require_string(row, "window_role")
        planned_role = self.contract.window_roles.get(window_id)
        if planned_role is None:
            raise CorpusValidationError(f"arena window {window_id!r} was not assigned before results")
        if role != planned_role:
            raise CorpusValidationError(f"window role mismatch: row={role!r}, plan={planned_role!r}")
        if self.training_only and role not in TRAINABLE_WINDOW_ROLES:
            raise CorpusValidationError(f"window role {role!r} is forbidden as training evidence")
        if self.contract.arena_freeze_cutoff_utc is not None:
            evidence = parse_utc(row.get("evidence_at_utc"), "evidence_at_utc")
            if evidence > self.contract.arena_freeze_cutoff_utc:
                raise CorpusValidationError("arena evidence is later than the frozen training cutoff")

        expected = {
            "agent_id": binding.agent_id,
            "arena_derivation_sha256": binding.derivation_sha256,
            "arena_record_sha256": binding.record_sha256,
            "normalized_sha256": binding.normalized_sha256,
            "raw_sha256": binding.raw_sha256,
            "submission_id": binding.submission_id,
            "submitted_source_sha256": binding.source_sha256,
            "window_id": binding.window_id,
            "window_role": binding.window_role,
        }
        observed = {
            "agent_id": agent_id,
            "arena_derivation_sha256": derivation_sha256,
            "arena_record_sha256": record_sha256,
            "normalized_sha256": normalized_sha256,
            "raw_sha256": raw_sha256,
            "submission_id": submission_id,
            "submitted_source_sha256": submitted_source_sha256,
            "window_id": window_id,
            "window_role": role,
        }
        if observed != expected:
            raise CorpusValidationError("arena row contradicts its immutable derivation/source binding")
        required_use = (
            "opponent-action-ranking-reanalysis-candidate"
            if source_kind == PAIR_SOURCE
            else "raw-terminal-value-candidate"
        )
        if required_use not in binding.uses:
            raise CorpusValidationError(f"arena derivation did not authorize use {required_use!r}")
        return binding

    def _validate_counterfactual(
        self,
        row: Mapping[str, Any],
        game_id: str,
        binding: ArenaGameBinding,
    ) -> None:
        if game_id == binding.game_id or _require_string(row, "counterfactual_id") != game_id:
            raise CorpusValidationError("counterfactual continuation must have its own synthetic game/counterfactual id")
        _require_string(row, "counterfactual_pair_id")
        variant = row.get("color_swap_variant")
        if isinstance(variant, bool) or variant not in (0, 1):
            raise CorpusValidationError("color_swap_variant must be 0 or 1")
        _require_bool(row, "historical_continuation_copied", False)
        continuation_sha = _require_sha256(row, "continuation_evidence_sha256")
        if continuation_sha != row.get("evidence_sha256"):
            raise CorpusValidationError("counterfactual row must bind its own continuation evidence")

    def _validate_pairwise(
        self,
        row: Mapping[str, Any],
        weight: float,
        binding: ArenaGameBinding,
    ) -> None:
        if row.get("actor_origin") != "opponent":
            raise CorpusValidationError("ranking targets may use only the opponent's observed action")
        _require_bool(row, "complete_action_legal", True)
        _require_bool(row, "counterfactual_replay_verified", True)
        observed_action = _require_string(row, "observed_complete_action")
        inferior_action = _require_string(row, "inferior_complete_action")
        if (
            re.fullmatch(r"[0-7]+", observed_action) is None
            or re.fullmatch(r"[0-7]+", inferior_action) is None
            or observed_action == inferior_action
        ):
            raise CorpusValidationError("pairwise complete actions must be distinct direction strings")
        if row.get("preferred_features") == row.get("inferior_features"):
            raise CorpusValidationError("pairwise successors must be distinct")
        rank = row.get("opponent_snapshot_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 50:
            raise CorpusValidationError("opponent_snapshot_rank must be in [1, 50]")
        if rank > 20 and weight > 0.5:
            raise CorpusValidationError("rank 21-50 pairwise rows must have weight at most 0.5")
        if rank != binding.opponent_frozen_rank or weight > binding.ranking_candidate_weight:
            raise CorpusValidationError("pairwise rank/weight contradicts its frozen derivation")
        exact = _require_bool(row, "exact")
        if exact:
            ordering = row.get("exact_ordering")
            if ordering != "preferred":
                raise CorpusValidationError("exact pairwise rows must prove the preferred ordering")
        else:
            for work in ("30000", "100000"):
                preferred = _require_number(row, f"preferred_value_{work}")
                inferior = _require_number(row, f"inferior_value_{work}")
                if not -1.0 <= preferred <= 1.0 or not -1.0 <= inferior <= 1.0:
                    raise CorpusValidationError(f"pairwise values at {work} work must be in [-1, 1]")
                if preferred - inferior < 0.10 - 1e-12:
                    raise CorpusValidationError(f"pairwise margin at {work} work must be at least 0.10")
        verdict = row.get("counterfactual_verdict")
        if verdict != "observed-not-proved-losing-vs-winning-alternative":
            raise CorpusValidationError(
                "pairwise rows must explicitly attest that reanalysis did not prove the observed move losing versus a win"
            )
        decision_id = _require_string(row, "decision_id")
        pair_index = row.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or not 0 <= pair_index < self.max_pairs_per_decision:
            raise CorpusValidationError(
                f"pair_index must be in [0, {self.max_pairs_per_decision - 1}] for decision {decision_id!r}"
            )

    def validate_rows(self, rows: Iterable[Mapping[str, Any]]) -> tuple[list[ValidatedRow], CorpusSummary]:
        validated: list[ValidatedRow] = []
        sample_ids: set[str] = set()
        pair_counts_by_decision: Counter[tuple[str, str]] = Counter()
        pair_counts_by_game: Counter[str] = Counter()
        pair_indices: set[tuple[str, str, int]] = set()
        pair_signatures: set[tuple[str, str, str, str]] = set()
        game_provenance: dict[str, tuple[Any, ...]] = {}
        counterfactual_pairs: dict[str, dict[int, str]] = defaultdict(dict)
        arena_position_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
        source_counts: Counter[str] = Counter()
        kind_counts: Counter[str] = Counter()
        window_counts: Counter[str] = Counter()
        games: set[str] = set()

        for row_number, row in enumerate(rows, 1):
            try:
                item = self.validate_row(row)
            except CorpusValidationError as error:
                raise CorpusValidationError(f"row {row_number}: {error}") from error
            if item.sample_id in sample_ids:
                raise CorpusValidationError(f"row {row_number}: duplicate sample_id {item.sample_id!r}")
            sample_ids.add(item.sample_id)
            if item.is_arena:
                binding = (
                    item.raw.get("arena_game_id"),
                    item.raw.get("arena_derivation_sha256"),
                    item.raw.get("arena_record_sha256"),
                    item.raw.get("raw_sha256"),
                    item.raw.get("normalized_sha256"),
                    item.raw.get("submitted_source_sha256"),
                    item.raw.get("agent_id"),
                    item.raw.get("submission_id"),
                    item.raw.get("window_id"),
                    item.raw.get("window_role"),
                )
            else:
                binding = (
                    item.raw.get("evidence_sha256"),
                    item.raw.get("producer_source_sha256"),
                    item.raw.get("opening_depth"),
                    item.raw.get("initialization"),
                    tuple(item.raw.get("checkpoint_inputs") or ()),
                )
            previous_binding = game_provenance.setdefault(item.game_id, binding)
            if previous_binding != binding:
                raise CorpusValidationError(
                    f"row {row_number}: game {item.game_id!r} has contradictory provenance"
                )
            if item.kind == "pairwise":
                decision_id = str(item.raw["decision_id"])
                key = (item.game_id, decision_id)
                pair_counts_by_decision[key] += 1
                pair_counts_by_game[item.game_id] += 1
                if pair_counts_by_decision[key] > self.max_pairs_per_decision:
                    raise CorpusValidationError(f"more than {self.max_pairs_per_decision} pairs for decision {key!r}")
                if pair_counts_by_game[item.game_id] > self.max_pairs_per_game:
                    raise CorpusValidationError(
                        f"more than {self.max_pairs_per_game} pairwise rows for game {item.game_id!r}"
                    )
                index_key = (item.game_id, decision_id, int(item.raw["pair_index"]))
                if index_key in pair_indices:
                    raise CorpusValidationError(f"row {row_number}: duplicate pair_index {index_key!r}")
                pair_indices.add(index_key)
                signature = (
                    item.game_id,
                    decision_id,
                    str(item.raw["observed_complete_action"]),
                    str(item.raw["inferior_complete_action"]),
                )
                if signature in pair_signatures:
                    raise CorpusValidationError(f"row {row_number}: duplicate pairwise alternative")
                pair_signatures.add(signature)
            elif item.is_arena:
                position = (str(item.raw["arena_game_id"]), str(item.raw["position_id"]))
                arena_position_labels[position].add(str(item.raw["label_method"]))
                if item.source_kind == "arena_counterfactual":
                    pair_id = str(item.raw["counterfactual_pair_id"])
                    variant = int(item.raw["color_swap_variant"])
                    previous_game = counterfactual_pairs[pair_id].setdefault(variant, item.game_id)
                    if previous_game != item.game_id:
                        raise CorpusValidationError(
                            f"row {row_number}: counterfactual pair {pair_id!r} repeats color variant {variant}"
                        )
            validated.append(item)
            games.add(item.game_id)
            kind_counts[item.kind] += 1
            source_counts[item.source_kind] += 1
            window_id = item.raw.get("window_id")
            if isinstance(window_id, str):
                window_counts[window_id] += 1

        for pair_id, variants in counterfactual_pairs.items():
            if set(variants) != {0, 1}:
                raise CorpusValidationError(
                    f"counterfactual pair {pair_id!r} must contain both color-swapped continuations"
                )
        for position, methods in arena_position_labels.items():
            if "exact" in methods and len(methods) > 1:
                raise CorpusValidationError(
                    f"exact solved value for arena position {position!r} must replace lower-confidence labels"
                )

        summary = CorpusSummary(
            rows=len(validated),
            games=len(games),
            counts_by_kind=dict(sorted(kind_counts.items())),
            counts_by_source=dict(sorted(source_counts.items())),
            counts_by_window=dict(sorted(window_counts.items())),
        )
        return validated, summary


def iter_jsonl(
    path: Path | str,
    *,
    digest: Any | None = None,
) -> Iterator[Mapping[str, Any]]:
    corpus_path = Path(path)
    if corpus_path.suffix != ".jsonl":
        raise CorpusValidationError(f"fresh corpus must be JSONL, got {corpus_path}")
    with corpus_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if digest is not None:
                digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CorpusValidationError(f"{corpus_path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, Mapping):
                raise CorpusValidationError(f"{corpus_path}:{line_number}: row must be an object")
            yield row


def require_within_campaign_root(path: Path | str, campaign_root: Path | str) -> Path:
    root = Path(campaign_root).resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CorpusValidationError(f"corpus is outside the campaign-owned root: {candidate}") from error
    if candidate.is_symlink():
        raise CorpusValidationError(f"campaign corpus may not be a symlink: {candidate}")
    return candidate


def load_and_validate_jsonl(
    paths: Sequence[Path | str],
    validator: FreshCorpusValidator,
    *,
    campaign_root: Path | str | None = None,
) -> tuple[list[ValidatedRow], CorpusSummary, list[dict[str, Any]]]:
    all_rows: list[Mapping[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for raw_path in paths:
        path = require_within_campaign_root(raw_path, campaign_root) if campaign_root is not None else Path(raw_path)
        digest = hashlib.sha256()
        all_rows.extend(iter_jsonl(path, digest=digest))
        inputs.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    rows, summary = validator.validate_rows(all_rows)
    summary = dataclasses.replace(summary, file_sha256=inputs[0]["sha256"] if len(inputs) == 1 else None)
    return rows, summary, inputs


def load_contract(
    path: Path | str,
    expected_sha256: str = APPROVED_WINDOW_PLAN_SHA256,
) -> CampaignContract:
    plan_path = Path(path)
    try:
        digest = verify_content_addressed_path(plan_path)
    except ValueError as error:
        raise CorpusValidationError(str(error)) from error
    if digest != expected_sha256:
        raise CorpusValidationError("window plan SHA-256 is not the frozen campaign plan")
    content = plan_path.read_bytes()
    try:
        plan = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusValidationError("window plan is not valid JSON") from error
    if not isinstance(plan, Mapping):
        raise CorpusValidationError("window plan must be an object")
    if canonical_json_bytes(plan) != content:
        raise CorpusValidationError("window plan must be canonical content-addressed JSON")
    try:
        validate_window_plan(plan)
    except ValueError as error:
        raise CorpusValidationError(str(error)) from error
    return CampaignContract.from_window_plan(plan)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Validate fresh jacek_arena_bfm JSONL rows")
    parser.add_argument("--window-plan", required=True, type=Path)
    parser.add_argument("--corpus", required=True, action="append", type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--producer-source", required=True, action="append", type=Path)
    parser.add_argument("--arena-derivation", action="append", default=[], type=Path)
    args = parser.parse_args()

    exclusions = load_excluded_game_ids(args.exclusions)
    producers = load_producer_source_hashes(
        args.producer_source,
        campaign_root=args.campaign_root,
    )
    arena_bindings = load_arena_game_bindings(
        args.arena_derivation,
        campaign_root=args.campaign_root,
        repository=args.repository,
    )
    validator = FreshCorpusValidator(
        load_contract(args.window_plan),
        excluded_game_ids=exclusions,
        approved_producer_source_sha256=producers,
        arena_game_bindings=arena_bindings,
        training_only=True,
    )
    _, summary, inputs = load_and_validate_jsonl(
        args.corpus,
        validator,
        campaign_root=args.campaign_root,
    )
    print(json.dumps({"valid": True, "summary": summary.to_json(), "inputs": inputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
