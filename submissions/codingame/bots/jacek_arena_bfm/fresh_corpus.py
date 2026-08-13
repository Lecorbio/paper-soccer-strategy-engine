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
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from .immutable_artifacts import sha256_file
except ImportError:  # pragma: no cover - standalone CLI execution
    from immutable_artifacts import sha256_file


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


def _validate_features(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != FEATURE_COUNT:
        raise CorpusValidationError(f"{field} must contain exactly {FEATURE_COUNT} values")
    output: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CorpusValidationError(f"{field}[{index}] must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric) or abs(numeric) > 1_000_000.0:
            raise CorpusValidationError(f"{field}[{index}] must be finite and bounded")
        output.append(numeric)
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
    features: tuple[float, ...] | None
    target: float | None
    preferred_features: tuple[float, ...] | None
    inferior_features: tuple[float, ...] | None
    weight: float
    is_arena: bool
    raw: Mapping[str, Any]


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


def load_excluded_game_ids(path: Path | str) -> frozenset[str]:
    """Read game IDs from metadata only; replay payload fields are refused."""

    with Path(path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    if not isinstance(registry, Mapping) or registry.get("schema") != EXCLUSION_SCHEMA:
        raise CorpusValidationError(f"exclusion registry schema must be {EXCLUSION_SCHEMA!r}")
    records = registry.get("records")
    if records is None:
        # Compatibility with collectors that name the same metadata list games.
        records = registry.get("games")
    if not isinstance(records, list):
        raise CorpusValidationError("exclusion registry records must be an array")
    excluded: set[str] = set()
    forbidden_payload_keys = {"replay", "frames", "moves", "actions", "labels", "states"}
    for record in records:
        if not isinstance(record, Mapping):
            raise CorpusValidationError("each exclusion record must be an object")
        if forbidden_payload_keys.intersection(record):
            raise CorpusValidationError("exclusion registry must contain metadata, never replay content")
        game_id = record.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, (str, int)):
            raise CorpusValidationError("exclusion game_id must be a string or integer")
        excluded.add(str(game_id))
    return frozenset(excluded)


class FreshCorpusValidator:
    def __init__(
        self,
        contract: CampaignContract,
        *,
        excluded_game_ids: Iterable[str | int] = (),
        training_only: bool = True,
        max_pairs_per_decision: int = 4,
        max_pairs_per_game: int = 32,
    ) -> None:
        self.contract = contract
        self.excluded_game_ids = frozenset(str(value) for value in excluded_game_ids)
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
        _require_sha256(row, "producer_source_sha256")
        _require_sha256(row, "evidence_sha256")
        if row.get("representation") != "mover_relative_316_edges_plus_105x8_distance_v1":
            raise CorpusValidationError("row must use the frozen mover-relative 1156-feature representation")
        weight = _require_number(row, "weight")
        if not (0.0 < weight <= 100.0):
            raise CorpusValidationError("weight must be in (0, 100]")

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
            if source_kind == "scratch_selfplay":
                self._validate_scratch(row, label_method)
            else:
                self._validate_arena_common(row)
                if source_kind == "arena_terminal" and label_method != "terminal_outcome":
                    raise CorpusValidationError("arena_terminal rows require terminal_outcome labels")
                if source_kind == "arena_reanalysis" and label_method not in {"exact", "stable_reanalysis"}:
                    raise CorpusValidationError("arena_reanalysis rows require exact or stable_reanalysis labels")
                if source_kind == "arena_counterfactual" and label_method != "counterfactual_outcome":
                    raise CorpusValidationError("counterfactual rows require their own continuation outcome")
            return ValidatedRow(
                kind=kind,
                source_kind=source_kind,
                sample_id=sample_id,
                game_id=game_id,
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
            self._validate_arena_common(row)
            self._validate_pairwise(row, weight)
            return ValidatedRow(
                kind=kind,
                source_kind=source_kind,
                sample_id=sample_id,
                game_id=game_id,
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

    def _validate_arena_common(self, row: Mapping[str, Any]) -> None:
        _require_bool(row, "operational_clean", True)
        _require_bool(row, "complete_transcript", True)
        _require_bool(row, "terminal_unambiguous", True)
        _require_bool(row, "timeout", False)
        _require_bool(row, "illegal_action", False)
        _require_bool(row, "malformed_transcript", False)
        _require_sha256(row, "raw_sha256")
        _require_sha256(row, "normalized_sha256")
        _require_sha256(row, "submitted_source_sha256")
        _require_string(row, "agent_id")
        _require_string(row, "submission_id")
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

    def _validate_pairwise(self, row: Mapping[str, Any], weight: float) -> None:
        if row.get("actor_origin") != "opponent":
            raise CorpusValidationError("ranking targets may use only the opponent's observed action")
        _require_bool(row, "complete_action_legal", True)
        rank = row.get("opponent_snapshot_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 50:
            raise CorpusValidationError("opponent_snapshot_rank must be in [1, 50]")
        if rank > 20 and weight > 0.5:
            raise CorpusValidationError("rank 21-50 pairwise rows must have weight at most 0.5")
        exact = _require_bool(row, "exact")
        if exact:
            ordering = row.get("exact_ordering")
            if ordering != "preferred":
                raise CorpusValidationError("exact pairwise rows must prove the preferred ordering")
        else:
            for work in ("30000", "100000"):
                preferred = _require_number(row, f"preferred_value_{work}")
                inferior = _require_number(row, f"inferior_value_{work}")
                if preferred - inferior < 0.10 - 1e-12:
                    raise CorpusValidationError(f"pairwise margin at {work} work must be at least 0.10")
        if row.get("counterfactual_verdict") == "observed_loses_alternative_wins":
            raise CorpusValidationError("proved-losing observed moves cannot be ranking targets")
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
            validated.append(item)
            games.add(item.game_id)
            kind_counts[item.kind] += 1
            source_counts[item.source_kind] += 1
            window_id = item.raw.get("window_id")
            if isinstance(window_id, str):
                window_counts[window_id] += 1

        summary = CorpusSummary(
            rows=len(validated),
            games=len(games),
            counts_by_kind=dict(sorted(kind_counts.items())),
            counts_by_source=dict(sorted(source_counts.items())),
            counts_by_window=dict(sorted(window_counts.items())),
        )
        return validated, summary


def iter_jsonl(path: Path | str) -> Iterator[Mapping[str, Any]]:
    corpus_path = Path(path)
    if corpus_path.suffix != ".jsonl":
        raise CorpusValidationError(f"fresh corpus must be JSONL, got {corpus_path}")
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
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
        digest = sha256_file(path)
        inputs.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
        all_rows.extend(iter_jsonl(path))
    rows, summary = validator.validate_rows(all_rows)
    summary = dataclasses.replace(summary, file_sha256=inputs[0]["sha256"] if len(inputs) == 1 else None)
    return rows, summary, inputs


def load_contract(path: Path | str) -> CampaignContract:
    with Path(path).open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if not isinstance(plan, Mapping):
        raise CorpusValidationError("window plan must be an object")
    return CampaignContract.from_window_plan(plan)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Validate fresh jacek_arena_bfm JSONL rows")
    parser.add_argument("--window-plan", required=True, type=Path)
    parser.add_argument("--corpus", required=True, action="append", type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--allow-nontraining", action="store_true")
    args = parser.parse_args()

    exclusions = load_excluded_game_ids(args.exclusions) if args.exclusions else frozenset()
    validator = FreshCorpusValidator(
        load_contract(args.window_plan),
        excluded_game_ids=exclusions,
        training_only=not args.allow_nontraining,
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
