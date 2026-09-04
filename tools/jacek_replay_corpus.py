#!/usr/bin/env python3
"""Offline replay-root normalization and leakage-safe teacher-row preparation.

The source normalizer reads a frozen exclusion registry *before* either replay
source.  It never performs network I/O and never treats observed replay moves
as policy or value labels: replays identify legal training roots only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_replay_features as features  # noqa: E402


ROOT_SCHEMA = "papersoccer.jacek-replay-roots.v1"
TEACHER_SCHEMA = "papersoccer.jacek-replay-teacher.v1"
RANK4_TEACHER_SCHEMA = "papersoccer.jacek-replay-teacher.v3"
SEARCH_TEACHER_SCHEMA = "papersoccer.jacek-replay-search-teacher.v4"
COMPLETE_TURN_ACTION_GROUP_SCHEMA = (
    "papersoccer.jacek-replay-complete-turn-action-group.v1"
)
COMPLETE_TURN_SUCCESSOR_LABELS_SCHEMA = (
    "papersoccer.compact-value-bfm-complete-turn-successor-labels.v1"
)
STANDARD_TEACHER_RANKING_PROFILE = "standard-v1"
HARD_5PCT_2M_TEACHER_RANKING_PROFILE = "hardest-5pct-2m-v1"
TARGET_POLICY_SCHEMA = "papersoccer.jacek-replay-target-policy.v1"
PUBLIC_SCHEMA = "papersoccer.public-jacek-training-games.v1"
LIVE_SNAPSHOT_SCHEMA = "papersoccer.live-replay-training-snapshot.v1"
LIVE_REPLAY_SCHEMA = "papersoccer.codingame-live-replay.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_json(path: pathlib.Path, expected_schema: str) -> tuple[dict, str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON source: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise ValueError(f"unexpected schema in {path}")
    return value, sha256_bytes(raw)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_repository_path(repository: pathlib.Path, relative: object) -> pathlib.Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("snapshot record path must be a nonempty string")
    candidate = (repository / relative).resolve()
    try:
        candidate.relative_to(repository.resolve())
    except ValueError as error:
        raise ValueError("snapshot record escapes the repository") from error
    return candidate


def _display_path(path: pathlib.Path, repository: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_turns(
    turns: object, expected_winner: int, *, label: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{label} has no complete turns")
    state = features.ReplayState()
    normalized: list[dict[str, object]] = []
    for turn_number, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != {"player_id", "action"}:
            raise ValueError(f"{label} turn {turn_number} has invalid fields")
        player = turn.get("player_id")
        action = turn.get("action")
        if isinstance(player, bool) or player not in (0, 1):
            raise ValueError(f"{label} turn {turn_number} has invalid player")
        if not isinstance(action, str):
            raise ValueError(f"{label} turn {turn_number} has invalid action")
        try:
            features.apply_complete_turn(state, int(player), action)
        except ValueError as error:
            raise ValueError(f"{label} turn {turn_number}: {error}") from error
        normalized.append({"player_id": int(player), "action": action})
    if state.winner is None:
        raise ValueError(f"{label} transcript is nonterminal")
    if state.winner != expected_winner:
        raise ValueError(f"{label} winner disagrees with transcript")
    return tuple(normalized)


def _exclusion_records(payload: dict) -> dict[int, dict]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("exclusion registry records must be an array")
    result: dict[int, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("exclusion record must be an object")
        game_id = record.get("game_id")
        categories = record.get("categories")
        if (
            isinstance(game_id, bool)
            or not isinstance(game_id, int)
            or game_id <= 0
            or not isinstance(categories, list)
            or not all(isinstance(value, str) for value in categories)
            or game_id in result
        ):
            raise ValueError("invalid or duplicate exclusion record")
        result[game_id] = record
    return result


def _protected(record: dict | None) -> bool:
    return bool(
        record
        and any(
            str(category).startswith("protected_")
            for category in record.get("categories", ())
        )
    )


def _load_frozen_assignments(path: pathlib.Path) -> tuple[dict[str, str], str]:
    payload, digest = _read_json(path, ROOT_SCHEMA)
    body_sha = payload.get("body_sha256")
    body = dict(payload)
    body.pop("body_sha256", None)
    if (
        not isinstance(body_sha, str)
        or sha256_bytes(canonical_json_bytes(body)) != body_sha
        or payload.get("feature_schema") != features.FEATURE_SCHEMA
    ):
        raise ValueError("previous roots manifest provenance is invalid")
    accepted = payload.get("accepted")
    if not isinstance(accepted, list):
        raise ValueError("previous roots manifest has no accepted records")
    assignments: dict[str, str] = {}
    for record in accepted:
        if not isinstance(record, dict):
            raise ValueError("previous roots record must be an object")
        group, split = record.get("group_id"), record.get("split")
        if (
            not isinstance(group, str)
            or not group
            or split not in {"train", "validation", "test"}
            or group in assignments
        ):
            raise ValueError("previous roots split assignment is malformed")
        assignments[group] = split
    return assignments, digest


def _assignment_for_strata(
    records: Sequence[dict],
    frozen_assignments: Mapping[str, str] | None = None,
) -> dict[str, str]:
    splits = ("train", "validation", "test")
    proportions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    by_stratum: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for record in records:
        stratum = (
            record["source"],
            record["focus_player"],
            record["winner"],
            record["opponent_tier"],
        )
        by_stratum[stratum].append(record["group_id"])
    assignment: dict[str, str] = {}
    stratum_for_group: dict[str, tuple[object, ...]] = {}
    stratum_counts: dict[tuple[object, ...], dict[str, int]] = {}
    for stratum, groups in sorted(by_stratum.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            groups,
            key=lambda group: (hashlib.sha256(group.encode()).digest(), group),
        )
        count = len(ordered)
        quotas = {split: proportions[split] * count for split in splits}
        counts = {split: math.floor(quotas[split]) for split in splits}
        residual = count - sum(counts.values())
        remainder_order = sorted(
            splits,
            key=lambda split: (
                -(quotas[split] - counts[split]),
                hashlib.sha256(f"{stratum!r}:{split}".encode()).digest(),
            ),
        )
        for index in range(residual):
            counts[remainder_order[index]] += 1
        slots = [split for split in splits for _ in range(counts[split])]
        slots.sort(
            key=lambda split: hashlib.sha256(
                f"{stratum!r}:{split}:{slots.count(split)}".encode()
            ).digest()
        )
        for group, split in zip(ordered, slots):
            assignment[group] = split
            stratum_for_group[group] = stratum
        stratum_counts[stratum] = counts

    frozen = dict(frozen_assignments or {})
    if any(split not in splits for split in frozen.values()):
        raise ValueError("frozen split assignment is invalid")
    frozen_groups = set(assignment) & set(frozen)
    for group in frozen_groups:
        old_split, frozen_split = assignment[group], frozen[group]
        if old_split == frozen_split:
            continue
        stratum = stratum_for_group[group]
        stratum_counts[stratum][old_split] -= 1
        stratum_counts[stratum][frozen_split] += 1
        assignment[group] = frozen_split

    total = len(records)
    if total == 0:
        return assignment
    if total >= 3:
        train_target = min((8 * total) // 10, total - 2)
        validation_target = max(1, (total - train_target) // 2)
        targets = {
            "train": train_target,
            "validation": validation_target,
            "test": total - train_target - validation_target,
        }
    else:
        targets = {
            "train": max(1, total - 1),
            "validation": 0,
            "test": 1 if total == 2 else 0,
        }

    totals = {
        split: sum(value == split for value in assignment.values())
        for split in splits
    }
    quotas_by_stratum = {
        stratum: {
            split: proportions[split] * len(groups)
            for split in splits
        }
        for stratum, groups in by_stratum.items()
    }
    while totals != targets:
        overfull = [split for split in splits if totals[split] > targets[split]]
        underfull = [split for split in splits if totals[split] < targets[split]]
        candidates = []
        for group, source_split in assignment.items():
            if source_split not in overfull or group in frozen_groups:
                continue
            stratum = stratum_for_group[group]
            counts = stratum_counts[stratum]
            quotas = quotas_by_stratum[stratum]
            for destination_split in underfull:
                old_error = sum(
                    (counts[split] - quotas[split]) ** 2 for split in splits
                )
                new_error = sum(
                    (
                        counts[split]
                        - (1 if split == source_split else 0)
                        + (1 if split == destination_split else 0)
                        - quotas[split]
                    )
                    ** 2
                    for split in splits
                )
                candidates.append(
                    (
                        new_error - old_error,
                        hashlib.sha256(
                            f"{group}:{source_split}:{destination_split}".encode()
                        ).digest(),
                        group,
                        source_split,
                        destination_split,
                    )
                )
        if not candidates:
            raise RuntimeError("unable to rebalance whole-game split targets")
        _, _, group, source_split, destination_split = min(candidates)
        assignment[group] = destination_split
        stratum = stratum_for_group[group]
        stratum_counts[stratum][source_split] -= 1
        stratum_counts[stratum][destination_split] += 1
        totals[source_split] -= 1
        totals[destination_split] += 1

    dimensions = ("source", "focus_player", "winner", "opponent_tier")
    eligible_values = {
        dimension: {
            value
            for value, count in Counter(
                record[dimension] for record in records
            ).items()
            if count >= len(splits)
        }
        for dimension in dimensions
    }

    def coverage_penalty() -> int:
        return sum(
            not any(
                assignment[record["group_id"]] == split
                and record[dimension] == value
                for record in records
            )
            for dimension in dimensions
            for value in eligible_values[dimension]
            for split in splits
        )

    penalty = coverage_penalty()
    while penalty:
        candidates = []
        groups = sorted(assignment)
        for first_index, first in enumerate(groups):
            if first in frozen_groups:
                continue
            for second in groups[first_index + 1 :]:
                if second in frozen_groups:
                    continue
                first_split, second_split = assignment[first], assignment[second]
                if first_split == second_split:
                    continue
                assignment[first], assignment[second] = second_split, first_split
                candidate_penalty = coverage_penalty()
                assignment[first], assignment[second] = first_split, second_split
                if candidate_penalty >= penalty:
                    continue
                first_stratum, second_stratum = (
                    stratum_for_group[first], stratum_for_group[second]
                )
                delta_error = 0.0
                for stratum, source_split, destination_split in (
                    (first_stratum, first_split, second_split),
                    (second_stratum, second_split, first_split),
                ):
                    if first_stratum == second_stratum:
                        continue
                    counts = stratum_counts[stratum]
                    quotas = quotas_by_stratum[stratum]
                    old_error = sum(
                        (counts[split] - quotas[split]) ** 2 for split in splits
                    )
                    new_error = sum(
                        (
                            counts[split]
                            - (1 if split == source_split else 0)
                            + (1 if split == destination_split else 0)
                            - quotas[split]
                        )
                        ** 2
                        for split in splits
                    )
                    delta_error += new_error - old_error
                candidates.append(
                    (
                        candidate_penalty,
                        delta_error,
                        hashlib.sha256(f"coverage:{first}:{second}".encode()).digest(),
                        first,
                        second,
                    )
                )
        if not candidates:
            raise RuntimeError(
                "unable to preserve source/color/outcome representation across splits"
            )
        candidate_penalty, _, _, first, second = min(candidates)
        first_split, second_split = assignment[first], assignment[second]
        assignment[first], assignment[second] = second_split, first_split
        first_stratum, second_stratum = stratum_for_group[first], stratum_for_group[second]
        if first_stratum != second_stratum:
            stratum_counts[first_stratum][first_split] -= 1
            stratum_counts[first_stratum][second_split] += 1
            stratum_counts[second_stratum][second_split] -= 1
            stratum_counts[second_stratum][first_split] += 1
        penalty = candidate_penalty
    return assignment


def _accepted_record(
    *,
    source: str,
    game_id: int,
    focus_player: int,
    winner: int,
    opponent_tier: str,
    turns: tuple[dict[str, object], ...],
    source_sha256: str,
) -> dict:
    group_id = f"{source}:{game_id}"
    return {
        "game_id": game_id,
        "group_id": group_id,
        "root_group_id": group_id,
        "source": source,
        "focus_player": focus_player,
        "winner": winner,
        "opponent_tier": opponent_tier,
        "turns": list(turns),
        "source_record_sha256": source_sha256,
    }


def normalize_replay_sources(
    *,
    repository: pathlib.Path,
    exclusion_path: pathlib.Path,
    public_jacek_path: pathlib.Path,
    live_snapshot_path: pathlib.Path,
    previous_roots_path: pathlib.Path | None = None,
) -> dict:
    """Combine the two frozen sources after first binding exclusions."""

    repository = repository.resolve()

    # This ordering is a security property, not an incidental implementation
    # detail: no candidate source is opened before the boundary is frozen.
    exclusions, exclusion_sha = _read_json(exclusion_path, EXCLUSION_SCHEMA)
    exclusion_by_id = _exclusion_records(exclusions)

    frozen_assignments: dict[str, str] = {}
    previous_roots_sha: str | None = None
    if previous_roots_path is not None:
        frozen_assignments, previous_roots_sha = _load_frozen_assignments(
            previous_roots_path
        )

    public, public_sha = _read_json(public_jacek_path, PUBLIC_SCHEMA)
    snapshot, snapshot_sha = _read_json(live_snapshot_path, LIVE_SNAPSHOT_SCHEMA)
    if snapshot.get("exclusion_registry_sha256") != exclusion_sha:
        raise ValueError("live snapshot was not built from the supplied exclusions")

    accepted: list[dict] = []
    excluded: list[dict] = []
    rejected: list[dict] = []
    seen_game_ids: set[int] = set()

    public_records = public.get("records")
    if not isinstance(public_records, list):
        raise ValueError("public Jacek records must be an array")
    for raw in public_records:
        if not isinstance(raw, dict):
            rejected.append({"source": "public-jacek", "reason": "non-object record"})
            continue
        game_id = raw.get("game_id")
        try:
            if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id <= 0:
                raise ValueError("invalid game id")
            if _protected(exclusion_by_id.get(game_id)):
                excluded.append(
                    {
                        "source": "public-jacek",
                        "game_id": game_id,
                        "reason": "protected-exclusion-registry",
                    }
                )
                continue
            if game_id in seen_game_ids:
                excluded.append(
                    {"source": "public-jacek", "game_id": game_id, "reason": "duplicate-game-id"}
                )
                continue
            focus_player = raw.get("player_id")
            won = raw.get("won")
            if focus_player not in (0, 1) or not isinstance(won, bool):
                raise ValueError("invalid focus player or result")
            winner = int(focus_player) if won else 1 - int(focus_player)
            turns = _validate_turns(
                raw.get("turns"), winner, label=f"public game {game_id}"
            )
            record_sha = sha256_bytes(canonical_json_bytes(raw))
            accepted.append(
                _accepted_record(
                    source="public-jacek",
                    game_id=game_id,
                    focus_player=int(focus_player),
                    winner=winner,
                    opponent_tier="public-unlocked",
                    turns=turns,
                    source_sha256=record_sha,
                )
            )
            seen_game_ids.add(game_id)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(
                {"source": "public-jacek", "game_id": game_id, "reason": str(error)}
            )

    # The historical public artifact exposes only an aggregate for records
    # removed before it was frozen.  Preserve that fact instead of inventing
    # game identities that are not present in the source.
    excluded_locked = public.get("excluded_locked_games")
    if not isinstance(excluded_locked, int) or excluded_locked < 0:
        raise ValueError("public excluded_locked_games must be nonnegative")
    if excluded_locked:
        excluded.append(
            {
                "source": "public-jacek",
                "game_id": None,
                "count": excluded_locked,
                "reason": "source-preexcluded-locked-games-aggregate",
            }
        )
    structural = public.get("structurally_rejected")
    if not isinstance(structural, list):
        raise ValueError("public structurally_rejected must be an array")
    for record in structural:
        if isinstance(record, dict):
            rejected.append({"source": "public-jacek", **record})
        else:
            rejected.append({"source": "public-jacek", "reason": "malformed source rejection"})

    snapshot_records = snapshot.get("records")
    if not isinstance(snapshot_records, list):
        raise ValueError("live snapshot records must be an array")
    for reference in snapshot_records:
        game_id: object = None
        try:
            if not isinstance(reference, dict):
                raise ValueError("live snapshot reference is not an object")
            game_id = reference.get("game_id")
            if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id <= 0:
                raise ValueError("invalid game id")
            if _protected(exclusion_by_id.get(game_id)):
                excluded.append(
                    {
                        "source": "own-live",
                        "game_id": game_id,
                        "reason": "protected-exclusion-registry",
                    }
                )
                continue
            if game_id in seen_game_ids:
                excluded.append(
                    {"source": "own-live", "game_id": game_id, "reason": "duplicate-game-id"}
                )
                continue
            record_path = _safe_repository_path(repository, reference.get("record_path"))
            record_sha = reference.get("record_sha256")
            if (
                not _valid_sha256(record_sha)
                or record_path.suffix != ".json"
                or record_path.stem != record_sha
                or sha256_file(record_path) != record_sha
            ):
                raise ValueError("live record SHA-256 mismatch")
            live, _ = _read_json(record_path, LIVE_REPLAY_SCHEMA)
            if record_path.read_bytes() != canonical_json_bytes(live):
                raise ValueError("live record is not canonical JSON")
            replay = live.get("replay")
            if not isinstance(replay, dict) or replay.get("game_id") != game_id:
                raise ValueError("live replay identity mismatch")
            winner = replay.get("winner_player_id")
            if winner not in (0, 1):
                raise ValueError("invalid live replay winner")
            own_player = reference.get("own_player_id")
            if own_player not in (0, 1):
                raise ValueError("live replay has no owned player")
            direct = reference.get("direct_experts")
            tiers = []
            if isinstance(direct, list):
                for expert in direct:
                    if isinstance(expert, dict) and isinstance(expert.get("strength_tier"), dict):
                        tier = expert["strength_tier"].get("name")
                        if isinstance(tier, str):
                            tiers.append(tier)
            opponent_tier = "+".join(sorted(set(tiers))) or "unranked-public"
            turns = _validate_turns(
                replay.get("turns"), int(winner), label=f"live game {game_id}"
            )
            accepted.append(
                _accepted_record(
                    source="own-live",
                    game_id=game_id,
                    focus_player=int(own_player),
                    winner=int(winner),
                    opponent_tier=opponent_tier,
                    turns=turns,
                    source_sha256=str(record_sha),
                )
            )
            seen_game_ids.add(game_id)
        except (KeyError, OSError, TypeError, ValueError) as error:
            rejected.append(
                {"source": "own-live", "game_id": game_id, "reason": str(error)}
            )

    accepted.sort(key=lambda record: (record["source"], record["game_id"]))
    accepted_groups = {record["group_id"] for record in accepted}
    missing_frozen = set(frozen_assignments) - accepted_groups
    if missing_frozen:
        raise ValueError(
            "append-only replay source dropped frozen groups: "
            + ", ".join(sorted(missing_frozen)[:5])
        )
    assignments = _assignment_for_strata(accepted, frozen_assignments)
    for record in accepted:
        record["split"] = assignments[record["group_id"]]
    excluded.sort(key=lambda record: (record["source"], record.get("game_id") or -1))
    rejected.sort(key=lambda record: (record["source"], record.get("game_id") or -1))
    manifest = {
        "schema": ROOT_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "tool_sha256": {
            "normalizer": sha256_file(pathlib.Path(__file__)),
            "features": sha256_file(pathlib.Path(features.__file__)),
        },
        "exclusion_boundary": {
            "path": _display_path(exclusion_path, repository),
            "sha256": exclusion_sha,
            "read_before_candidate_sources": True,
        },
        "sources": [
            {
                "kind": "public-jacek",
                "path": _display_path(public_jacek_path, repository),
                "sha256": public_sha,
            },
            {
                "kind": "own-live",
                "path": _display_path(live_snapshot_path, repository),
                "sha256": snapshot_sha,
            },
        ],
        "split_policy": (
            "whole-root-game 80/10/10 stratified by source, focus color, outcome, "
            "and opponent tier; prior assignments are immutable when split_parent is "
            "present; continuations and reflections inherit root_group_id"
        ),
        "split_parent": (
            {
                "path": _display_path(previous_roots_path, repository),
                "sha256": previous_roots_sha,
                "frozen_groups": len(frozen_assignments),
            }
            if previous_roots_path is not None
            else None
        ),
        "accepted": accepted,
        "excluded": excluded,
        "structurally_rejected": rejected,
        "counts": {
            "accepted": len(accepted),
            "excluded_records": len(excluded),
            "source_preexcluded_aggregate": excluded_locked,
            "structurally_rejected": len(rejected),
            "split_games": {
                split: sum(record["split"] == split for record in accepted)
                for split in ("train", "validation", "test")
            },
        },
    }
    # Bind the decision body without creating a recursive self-hash.  The
    # final file hash remains the authoritative artifact identity.
    manifest["body_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


@dataclasses.dataclass(frozen=True)
class TeacherLineage:
    schema: str
    position_id: str | None
    group_id: str
    root_group_id: str
    source: str
    split: str | None
    campaign_id: str | None


@dataclasses.dataclass(frozen=True)
class LabeledSample:
    active: tuple[int, ...]
    target: float
    weight: float
    group_id: str
    lineages: tuple[TeacherLineage, ...] = ()


def target_policy_for_schema(schema: str) -> dict[str, object]:
    """Describe target construction without conflating teacher value frames."""

    common: dict[str, object] = {
        "schema": TARGET_POLICY_SCHEMA,
        "teacher_schema": schema,
        "mixture": {
            "teacher_weight": 0.75,
            "outcome_weight": 0.25,
            "outcome_frame": "mover-relative-terminal-winner",
        },
    }
    if schema in {TEACHER_SCHEMA, RANK4_TEACHER_SCHEMA}:
        common["teacher_value"] = {
            "input_frame": "absolute-player-one-root-score",
            "transform": "mover-sign*tanh(root_score/12000)",
            "proof": "absolute-proven-winner-to-mover-relative-exact-sign",
        }
    elif schema in {SEARCH_TEACHER_SCHEMA, COMPLETE_TURN_ACTION_GROUP_SCHEMA}:
        common["teacher_value"] = {
            "input_frame": "mover-relative-value",
            "transform": "identity",
            "proof": (
                "explicit-action-group-root-solved-and-absolute-proven-winner"
                if schema == COMPLETE_TURN_ACTION_GROUP_SCHEMA
                else "explicit-root-solved-and-absolute-proven-winner"
            ),
        }
    else:
        raise ValueError("unsupported teacher schema for target policy")
    return common


def _teacher_target(root_score: float, mover: int, proven_winner: int | None) -> float:
    if proven_winner is not None:
        return 1.0 if proven_winner == mover else -1.0
    # Rank-4 reports an absolute Player-One score.  The value-only runtime is
    # mover-relative, so Player Two roots must reverse that sign.
    return (1.0 if mover == 0 else -1.0) * math.tanh(root_score / 12_000.0)


def _direct_teacher_target(
    teacher_value: float,
    mover: int,
    root_solved: bool,
    proven_winner: int | None,
) -> float:
    if not -1.0 <= teacher_value <= 1.0:
        raise ValueError("search teacher_value must be in [-1, 1]")
    if root_solved:
        if proven_winner not in (0, 1):
            raise ValueError("a solved search root requires an explicit proven_winner")
        expected = 1.0 if proven_winner == mover else -1.0
        if teacher_value != expected:
            raise ValueError(
                "solved search teacher_value disagrees with proven_winner"
            )
    elif proven_winner is not None:
        raise ValueError("an unsolved search root cannot declare a proven_winner")
    # Search-teacher values are already mover-relative.  Applying Rank-4's
    # sign conversion or tanh normalization here would corrupt the target.
    return teacher_value


def _prefix_state(prefix: object) -> features.ReplayState:
    if not isinstance(prefix, list):
        raise ValueError("teacher prefix must be an array of complete turns")
    state = features.ReplayState()
    for turn_number, turn in enumerate(prefix):
        if not isinstance(turn, dict) or set(turn) != {"player_id", "action"}:
            raise ValueError(f"teacher prefix turn {turn_number} is malformed")
        player, action = turn["player_id"], turn["action"]
        if isinstance(player, bool) or not isinstance(player, int) or player not in (0, 1):
            raise ValueError(f"teacher prefix turn {turn_number} has invalid player")
        if not isinstance(action, str):
            raise ValueError(f"teacher prefix turn {turn_number} has invalid action")
        features.apply_complete_turn(
            state, player, action
        )
    if state.winner is not None:
        raise ValueError("teacher prefix is terminal")
    return state


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _uint(value: object, label: str, maximum: int = (1 << 64) - 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ValueError(f"{label} must be an unsigned integer")
    return value


def _positive_uint(
    value: object, label: str, maximum: int = (1 << 64) - 1
) -> int:
    result = _uint(value, label, maximum)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _proven_winner(value: object) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
    ):
        raise ValueError("proven_winner must be null, zero, or one")
    return value


def _teacher_common(
    row: Mapping[str, object]
) -> tuple[str, str, str, int, int, features.ReplayState, float]:
    group_id = _nonempty_string(row.get("group_id"), "teacher group_id")
    root_group_id = _nonempty_string(
        row.get("root_group_id", group_id), "teacher root_group_id"
    )
    source = _nonempty_string(row.get("source"), "teacher source")
    winner, mover = row.get("winner"), row.get("mover")
    if (
        isinstance(winner, bool)
        or not isinstance(winner, int)
        or winner not in (0, 1)
        or isinstance(mover, bool)
        or not isinstance(mover, int)
        or mover not in (0, 1)
    ):
        raise ValueError("teacher winner and mover must be zero or one")
    weight = _finite_number(row.get("weight", 1.0), "teacher weight")
    if weight <= 0.0:
        raise ValueError("teacher weight must be positive")
    state = _prefix_state(row.get("prefix"))
    if state.to_move != mover:
        raise ValueError("teacher mover disagrees with replayed prefix")
    return group_id, root_group_id, source, winner, mover, state, weight


def _rank4_teacher_value(row: Mapping[str, object], mover: int) -> float:
    depth = _positive_uint(row.get("completed_depth"), "teacher completed_depth")
    nodes = _positive_uint(row.get("nodes"), "teacher nodes")
    del depth, nodes
    score = _finite_number(row.get("root_score"), "teacher root_score")
    return _teacher_target(score, mover, _proven_winner(row.get("proven_winner")))


_RANK4_TEACHER_FIELDS = {"kind", "source_sha256"}
_RANK4_CONFIG_FIELDS = {
    "max_nodes",
    "max_time_ms",
    "max_turn_depth",
    "replay_value_blend_percent",
    "teacher_residual_weight_percent",
}
_RANK4_STATS_COUNTERS = {
    "attempted_depth",
    "completed_depth",
    "nodes",
    "leaf_evaluations",
    "terminal_nodes",
    "completed_actions",
}
_RANK4_STATS_FIELDS = _RANK4_STATS_COUNTERS | {
    "budget_exhausted",
    "node_cap_reached",
    "depth_cap_reached",
    "deadline_reached",
    "termination_reason",
}


def _rank4_fixed_work_teacher_value(
    row: Mapping[str, object], mover: int
) -> float:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict) or set(teacher) != _RANK4_TEACHER_FIELDS:
        raise ValueError("Rank-4 teacher identity is malformed")
    if (
        teacher.get("kind") != "rank4-fixed-work"
        or not _valid_sha256(teacher.get("source_sha256"))
    ):
        raise ValueError("Rank-4 teacher identity is invalid")

    configuration = row.get("search_config")
    if (
        not isinstance(configuration, dict)
        or set(configuration) != _RANK4_CONFIG_FIELDS
    ):
        raise ValueError("Rank-4 teacher configuration is malformed")
    max_nodes = _positive_uint(
        configuration.get("max_nodes"), "Rank-4 max_nodes"
    )
    max_time_ms = _uint(
        configuration.get("max_time_ms"),
        "Rank-4 max_time_ms",
        (1 << 32) - 1,
    )
    if max_time_ms != 0:
        raise ValueError("Rank-4 fixed-work labels require max_time_ms zero")
    max_depth = _positive_uint(
        configuration.get("max_turn_depth"),
        "Rank-4 max_turn_depth",
        32,
    )
    replay_blend = _uint(
        configuration.get("replay_value_blend_percent"),
        "Rank-4 replay_value_blend_percent",
        100,
    )
    residual_weight = _uint(
        configuration.get("teacher_residual_weight_percent"),
        "Rank-4 teacher_residual_weight_percent",
        100,
    )
    del replay_blend, residual_weight

    stats = row.get("search_stats")
    if not isinstance(stats, dict) or set(stats) != _RANK4_STATS_FIELDS:
        raise ValueError("Rank-4 teacher statistics are malformed")
    counters = {
        field: _uint(stats.get(field), f"Rank-4 stats {field}")
        for field in _RANK4_STATS_COUNTERS
    }
    for field in (
        "budget_exhausted",
        "node_cap_reached",
        "depth_cap_reached",
        "deadline_reached",
    ):
        if not isinstance(stats.get(field), bool):
            raise ValueError(f"Rank-4 stats {field} must be boolean")

    depth = _uint(row.get("completed_depth"), "teacher completed_depth", 32)
    nodes = _positive_uint(row.get("nodes"), "teacher nodes")
    if (
        counters["completed_depth"] != depth
        or counters["nodes"] != nodes
        or nodes > max_nodes
        or counters["attempted_depth"] < max(1, depth)
        or counters["attempted_depth"] > max_depth
        or counters["completed_actions"] == 0
        or counters["completed_actions"] > nodes
    ):
        raise ValueError("Rank-4 teacher did not complete a usable root search")

    deadline_reached = stats["deadline_reached"]
    node_cap_reached = stats["node_cap_reached"]
    depth_cap_reached = stats["depth_cap_reached"]
    budget_exhausted = stats["budget_exhausted"]
    if deadline_reached:
        raise ValueError("Rank-4 teacher reached its deadline")
    if budget_exhausted != node_cap_reached:
        raise ValueError("Rank-4 teacher budget and node-cap flags disagree")
    if node_cap_reached and nodes != max_nodes:
        raise ValueError("Rank-4 teacher did not consume its exact node cap")
    if depth_cap_reached != (depth == max_depth):
        raise ValueError("Rank-4 teacher depth-cap flag is inconsistent")
    if node_cap_reached and depth_cap_reached:
        raise ValueError("Rank-4 teacher cannot reach both work caps")
    expected_attempted_depth = depth + 1 if node_cap_reached else depth
    if counters["attempted_depth"] != expected_attempted_depth:
        raise ValueError("Rank-4 teacher iterative-depth state is inconsistent")
    if depth == 0 and not node_cap_reached:
        raise ValueError("Rank-4 teacher stopped within depth one without a node cap")

    root_solved = row.get("root_solved")
    if not isinstance(root_solved, bool):
        raise ValueError("Rank-4 teacher root_solved must be boolean")
    proven_winner = _proven_winner(row.get("proven_winner"))
    if root_solved != (proven_winner is not None):
        raise ValueError("Rank-4 teacher proof flag is inconsistent")
    score = _finite_number(row.get("root_score"), "teacher root_score")
    score_is_mate = abs(score) >= 1_000_000 - max_depth
    score_winner = 0 if score > 0 else 1
    expected_root_solved = score_is_mate and (
        depth != 0 or score_winner == mover
    )
    if root_solved != expected_root_solved or (
        root_solved and proven_winner != score_winner
    ):
        raise ValueError("Rank-4 teacher proof is not supported by its search")

    expected_termination = "root-solved" if root_solved else "fixed-work-cap"
    if stats.get("termination_reason") != expected_termination:
        raise ValueError("Rank-4 teacher termination reason is inconsistent")
    if not root_solved and not (node_cap_reached or depth_cap_reached):
        raise ValueError("Rank-4 teacher did not complete its fixed work cap")
    return _teacher_target(score, mover, proven_winner)


_SEARCH_TEACHER_FIELDS = {
    "kind",
    "source_sha256",
    "model_sha256",
    "feature_schema",
    "feature_schema_sha256",
}
_SEARCH_CONFIG_FIELDS = {
    "seed",
    "max_time_ms",
    "max_tree_nodes",
    "max_actions",
    "max_partial_paths",
    "exploration",
    "fpu",
}
_SEARCH_STATS_COUNTERS = {
    "expansions",
    "generated_actions",
    "retained_actions",
    "neural_evaluations",
    "visits",
    "completed_actions",
    "duplicate_boundaries",
    "partial_paths",
    "fifo_extractions",
    "lifo_extractions",
    "tactical_proofs",
    "tactical_solutions",
    "truncations",
    "generation_action_cap_stops",
    "generation_partial_cap_stops",
    "generation_deadline_stops",
    "materialization_deadline_stops",
    "generation_queue_drops",
    "generation_retention_drops",
    "generation_boundary_replacements",
    "generation_tactical_shortcuts",
    "generation_fallbacks",
    "generation_frontier_resumptions",
    "generation_zero_action_resumptions",
    "generation_max_frontier_depth",
    "progressive_widenings",
    "closed_unsolved_nodes",
    "closed_unsolved_nonexhaustive_nodes",
    "open_unexpanded_nodes",
    "implicit_action_frontiers",
    "max_open_children",
    "tree_nodes",
}
_SEARCH_STATS_FIELDS = _SEARCH_STATS_COUNTERS | {
    "max_complete_turn_depth",
    "deadline_reached",
    "tree_cap_reached",
    "termination_reason",
}


def _search_teacher_value(row: Mapping[str, object], mover: int) -> float:
    teacher = row.get("teacher")
    if not isinstance(teacher, dict) or set(teacher) != _SEARCH_TEACHER_FIELDS:
        raise ValueError("search teacher identity is malformed")
    if teacher.get("kind") != "jacek_replay_bfm_search":
        raise ValueError("search teacher kind is invalid")
    for field in ("source_sha256", "model_sha256", "feature_schema_sha256"):
        if not _valid_sha256(teacher.get(field)):
            raise ValueError(f"search teacher {field} is invalid")
    if (
        teacher.get("feature_schema") != features.FEATURE_SCHEMA
        or teacher.get("feature_schema_sha256")
        != hashlib.sha256(features.FEATURE_SCHEMA.encode("utf-8")).hexdigest()
    ):
        raise ValueError("search teacher feature schema identity is invalid")

    configuration = row.get("search_config")
    if (
        not isinstance(configuration, dict)
        or set(configuration) != _SEARCH_CONFIG_FIELDS
    ):
        raise ValueError("search teacher configuration is malformed")
    _uint(configuration.get("seed"), "search seed")
    max_time_ms = _uint(
        configuration.get("max_time_ms"),
        "search max_time_ms",
        (1 << 32) - 1,
    )
    if max_time_ms != 0:
        raise ValueError("search fixed-work labels require max_time_ms zero")
    max_tree_nodes = _positive_uint(
        configuration.get("max_tree_nodes"), "search max_tree_nodes"
    )
    _positive_uint(configuration.get("max_actions"), "search max_actions")
    _positive_uint(
        configuration.get("max_partial_paths"), "search max_partial_paths"
    )
    exploration = _finite_number(configuration.get("exploration"), "search exploration")
    fpu = _finite_number(configuration.get("fpu"), "search fpu")
    if exploration < 0.0 or not -1.0 <= fpu <= 1.0:
        raise ValueError("search exploration or FPU is outside its valid range")

    stats = row.get("search_stats")
    if not isinstance(stats, dict) or set(stats) != _SEARCH_STATS_FIELDS:
        raise ValueError("search teacher statistics are malformed")
    counters = {
        field: _uint(stats.get(field), f"search stats {field}")
        for field in _SEARCH_STATS_COUNTERS
    }
    depth = _positive_uint(
        stats.get("max_complete_turn_depth"),
        "search stats max_complete_turn_depth",
        (1 << 32) - 1,
    )
    del depth
    deadline_reached = stats.get("deadline_reached")
    tree_cap_reached = stats.get("tree_cap_reached")
    if not isinstance(deadline_reached, bool) or not isinstance(tree_cap_reached, bool):
        raise ValueError("search deadline/tree-cap flags must be booleans")
    if deadline_reached:
        raise ValueError("search teacher reached its deadline")
    if counters["generation_deadline_stops"] != 0 or (
        counters["materialization_deadline_stops"] != 0
    ):
        raise ValueError("search teacher carries a deadline stop")
    if counters["generation_queue_drops"] != 0:
        raise ValueError("search teacher dropped a partial frontier")
    if (
        counters["fifo_extractions"] != 0
        or counters["lifo_extractions"] != counters["partial_paths"]
    ):
        raise ValueError("search teacher resumable frontier counters disagree")
    if (
        counters["closed_unsolved_nodes"] != 0
        or counters["closed_unsolved_nonexhaustive_nodes"] != 0
    ):
        raise ValueError("search teacher closed an unsolved frontier")
    if counters["max_open_children"] > configuration["max_actions"]:
        raise ValueError("search teacher exceeded its sampled frontier width")
    if (
        counters["tree_nodes"] == 0
        or counters["tree_nodes"] > max_tree_nodes
        or counters["completed_actions"] == 0
    ):
        raise ValueError("search teacher did not complete a usable root search")

    root_solved = row.get("root_solved")
    if not isinstance(root_solved, bool):
        raise ValueError("search teacher root_solved must be boolean")
    termination_reason = stats.get("termination_reason")
    expected_termination = "root-solved" if root_solved else "fixed-work-cap"
    if termination_reason != expected_termination:
        raise ValueError("search teacher termination reason is inconsistent")
    if not root_solved and (
        not tree_cap_reached
        or counters["tree_nodes"] != max_tree_nodes
        or counters["visits"] == 0
    ):
        raise ValueError("unsolved search teacher did not consume its fixed work cap")
    value = _finite_number(row.get("teacher_value"), "search teacher_value")
    return _direct_teacher_target(
        value, mover, root_solved, _proven_winner(row.get("proven_winner"))
    )


_ACTION_GROUP_TEACHER_FIELDS = {
    "kind",
    "artifact_sha256",
    "payload_sha256",
    "feature_schema_sha256",
    "source_sha256",
}
_ACTION_GROUP_RANKING = {
    "complete_turn_boundaries": True,
    "teacher_value_frame": "explicit-mover-relative",
    "successor_aliases": "canonical-boundary-state",
    "best_tie_break": "successor-id-ascending",
}
_ACTION_GROUP_WORK_FIELDS = {
    "seed",
    "max_time_ms",
    "max_tree_nodes",
    "max_actions",
    "max_partial_paths",
    "exploration",
    "fpu",
}
_ACTION_GROUP_OPTIONAL_WORK_FIELDS = {"teacher_ranking_profile"}
_ACTION_GROUP_SOURCE_FIELDS = {
    "campaign_id",
    "position_id",
    "root_group_id",
    "group_id",
    "source",
    "split",
    "winner",
    "prefix",
}
_ACTION_GROUP_FIELDS = {
    "group_id",
    "parent_identity",
    "identity_algorithm",
    "parent_mover",
    "root_value",
    "root_solved",
    "proven_winner",
    "termination_reason",
    "successors_exhaustive",
    "work_budget",
    "source_binding",
    "successors",
}
_ACTION_GROUP_SUCCESSOR_FIELDS = {
    "successor_id",
    "active",
    "transcript",
    "teacher_value",
    "value_mover",
    "proof",
    "termination",
    "visits",
    "selection_visits",
}


def _validate_action_group_teacher(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ACTION_GROUP_TEACHER_FIELDS:
        raise ValueError("action-group teacher identity is malformed")
    if value.get("kind") != "jacek_replay_bfm_search":
        raise ValueError("action-group teacher kind is invalid")
    for field in _ACTION_GROUP_TEACHER_FIELDS - {"kind"}:
        if not _valid_sha256(value.get(field)):
            raise ValueError(f"action-group teacher {field} is invalid")
    expected_feature = hashlib.sha256(features.FEATURE_SCHEMA.encode("utf-8")).hexdigest()
    if value.get("feature_schema_sha256") != expected_feature:
        raise ValueError("action-group teacher feature schema identity changed")
    return dict(value)


def _validate_action_group_work(value: object, source: Mapping[str, object]) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or not _ACTION_GROUP_WORK_FIELDS <= set(value)
        or set(value) - _ACTION_GROUP_WORK_FIELDS
        not in (set(), _ACTION_GROUP_OPTIONAL_WORK_FIELDS)
    ):
        raise ValueError("action-group work budget is malformed")
    seed = _uint(value.get("seed"), "action-group seed")
    if _uint(value.get("max_time_ms"), "action-group max_time_ms") != 0:
        raise ValueError("action-group labels require fixed work")
    nodes = _positive_uint(value.get("max_tree_nodes"), "action-group max_tree_nodes")
    actions = _positive_uint(value.get("max_actions"), "action-group max_actions")
    paths = _positive_uint(
        value.get("max_partial_paths"), "action-group max_partial_paths"
    )
    exploration = _finite_number(value.get("exploration"), "action-group exploration")
    fpu = _finite_number(value.get("fpu"), "action-group fpu")
    profile = value.get(
        "teacher_ranking_profile", STANDARD_TEACHER_RANKING_PROFILE
    )
    if profile not in {
        STANDARD_TEACHER_RANKING_PROFILE,
        HARD_5PCT_2M_TEACHER_RANKING_PROFILE,
    }:
        raise ValueError("action-group teacher-ranking profile is unregistered")
    if actions > 250 or paths > 50_000:
        raise ValueError("action-group work budget exceeds the teacher contract")
    if profile == STANDARD_TEACHER_RANKING_PROFILE:
        if nodes > 1_000_000:
            raise ValueError("action-group work budget exceeds the teacher contract")
    elif (
        nodes != 2_000_000
        or actions != 250
        or paths != 50_000
        or exploration != 0.5
        or fpu != 0.5
    ):
        raise ValueError(
            "registered hardest-5pct-2m work budget changed"
        )
    if exploration < 0.0 or not -1.0 <= fpu <= 1.0:
        raise ValueError("action-group exploration or FPU is outside its range")
    material = (
        _nonempty_string(source.get("campaign_id"), "action-group campaign_id")
        + "\0"
        + _nonempty_string(source.get("position_id"), "action-group position_id")
        + "\0"
        + str(nodes)
    ).encode("utf-8")
    if seed != int(hashlib.sha256(material).hexdigest()[:16], 16):
        raise ValueError("action-group seed is not bound to its position and work")
    return dict(value)


def _canonical_transcript_for_physical(transcript: str, mover: int) -> str:
    if not transcript or any(character not in "01234567" for character in transcript):
        raise ValueError("action-group transcript is malformed")
    if mover == 0:
        return transcript
    return "".join(str((int(character) + 4) % 8) for character in transcript)


def _mover_canonical_position_identity(state: features.ReplayState) -> str:
    """Hash the exact mover-canonical boundary state used by native search."""

    rotate = state.to_move == 1
    transform = features.rotate_point if rotate else (lambda point: point)
    used_edges = bytearray((features.EDGE_COUNT + 7) // 8)
    for first, second in state.used_segments:
        edge = features.EDGE_INDEX[
            features._segment(transform(first), transform(second))
        ]
        used_edges[edge // 8] |= 1 << (edge % 8)
    visited_vertices = bytearray((features.VERTEX_COUNT + 7) // 8)
    for point, visits in state.visit_count.items():
        if visits > 0:
            vertex = features.POINT_INDEX[transform(point)]
            visited_vertices[vertex // 8] |= 1 << (vertex % 8)
    ball = features.POINT_INDEX[transform(state.ball)]
    status = 0 if state.winner is None else (1 if state.winner == 0 else 2)
    if rotate and status in (1, 2):
        status = 3 - status
    material = (
        b"sha256-mover-canonical-boundary-v1\0"
        + features.FEATURE_SCHEMA.encode("utf-8")
        + b"\0"
        + bytes(used_edges)
        + bytes(visited_vertices)
        + ball.to_bytes(2, "big")
        + bytes((status,))
    )
    return hashlib.sha256(material).hexdigest()


def validate_complete_turn_action_group(row: object) -> dict[str, object]:
    """Validate one deterministic, source-bound complete-turn ranking row."""

    expected_row_fields = {
        "schema",
        "feature_schema",
        "source_bundle_body_sha256",
        "teacher",
        "ranking",
        "split",
        "group",
    }
    if (
        not isinstance(row, dict)
        or set(row) != expected_row_fields
        or row.get("schema") != COMPLETE_TURN_ACTION_GROUP_SCHEMA
        or row.get("feature_schema") != features.FEATURE_SCHEMA
        or not _valid_sha256(row.get("source_bundle_body_sha256"))
        or row.get("ranking") != _ACTION_GROUP_RANKING
        or row.get("split") not in {"train", "validation"}
    ):
        raise ValueError("complete-turn action-group row is malformed")
    teacher = _validate_action_group_teacher(row.get("teacher"))
    group = row.get("group")
    if not isinstance(group, dict) or set(group) != _ACTION_GROUP_FIELDS:
        raise ValueError("complete-turn action group is malformed")
    source = group.get("source_binding")
    if not isinstance(source, dict) or set(source) != _ACTION_GROUP_SOURCE_FIELDS:
        raise ValueError("action-group source binding is malformed")
    if source.get("split") != row["split"]:
        raise ValueError("action-group source split changed")
    position_id = _nonempty_string(source.get("position_id"), "action-group position_id")
    if (
        not _valid_sha256(group.get("group_id"))
        or group.get("group_id") != group.get("parent_identity")
    ):
        raise ValueError("action-group group_id is not its canonical parent identity")
    for field in ("root_group_id", "group_id", "source", "campaign_id"):
        _nonempty_string(source.get(field), f"action-group source {field}")
    winner = source.get("winner")
    mover = group.get("parent_mover")
    if (
        isinstance(winner, bool)
        or not isinstance(winner, int)
        or winner not in (0, 1)
        or isinstance(mover, bool)
        or not isinstance(mover, int)
        or mover not in (0, 1)
    ):
        raise ValueError("action-group winner or parent mover is invalid")
    state = _prefix_state(source.get("prefix"))
    if state.to_move != mover:
        raise ValueError("action-group parent mover disagrees with its prefix")
    if (
        not _valid_sha256(group.get("parent_identity"))
        or group.get("identity_algorithm")
        != "sha256-mover-canonical-boundary-v1"
        or not isinstance(group.get("successors_exhaustive"), bool)
    ):
        raise ValueError("action-group parent identity is invalid")
    if group.get("parent_identity") != _mover_canonical_position_identity(state):
        raise ValueError("action-group parent identity disagrees with its prefix")
    root_solved = group.get("root_solved")
    if not isinstance(root_solved, bool):
        raise ValueError("action-group root_solved is invalid")
    root_value = _direct_teacher_target(
        _finite_number(group.get("root_value"), "action-group root value"),
        mover,
        root_solved,
        _proven_winner(group.get("proven_winner")),
    )
    del root_value
    expected_root_termination = "root-solved" if root_solved else "fixed-work-cap"
    if group.get("termination_reason") != expected_root_termination:
        raise ValueError("action-group root termination is inconsistent")
    _validate_action_group_work(group.get("work_budget"), source)

    successors = group.get("successors")
    if not isinstance(successors, list) or not successors:
        raise ValueError("action group must contain successors")
    order: list[tuple[str, str]] = []
    for ordinal, successor in enumerate(successors):
        if not isinstance(successor, dict) or set(successor) != _ACTION_GROUP_SUCCESSOR_FIELDS:
            raise ValueError(f"action-group successor {ordinal} is malformed")
        successor_id = successor.get("successor_id")
        transcript = successor.get("transcript")
        if not _valid_sha256(successor_id) or not isinstance(transcript, str):
            raise ValueError(f"action-group successor {ordinal} identity is invalid")
        physical = _canonical_transcript_for_physical(transcript, mover)
        successor_state = dataclasses.replace(
            state,
            used_segments=set(state.used_segments),
            visit_count=dict(state.visit_count),
        )
        features.apply_complete_turn(successor_state, mover, physical)
        active = features.validate_active(successor.get("active"))
        if active != features.encode_active(successor_state):
            raise ValueError(f"action-group successor {ordinal} features disagree")
        if successor_id != _mover_canonical_position_identity(successor_state):
            raise ValueError(f"action-group successor {ordinal} identity disagrees")
        value_mover = successor.get("value_mover")
        if value_mover != successor_state.to_move:
            raise ValueError(f"action-group successor {ordinal} value frame changed")
        proof = successor.get("proof")
        termination = successor.get("termination")
        if (
            not isinstance(proof, dict)
            or set(proof) != {"solved", "proven_winner"}
            or not isinstance(proof.get("solved"), bool)
            or not isinstance(termination, dict)
            or set(termination) != {"reason", "value_status"}
        ):
            raise ValueError(f"action-group successor {ordinal} proof is malformed")
        solved = proof["solved"]
        proven = _proven_winner(proof.get("proven_winner"))
        value = _finite_number(
            successor.get("teacher_value"),
            f"action-group successor {ordinal} teacher value",
        )
        if solved:
            if proven not in (0, 1) or value != (
                1.0 if proven == value_mover else -1.0
            ):
                raise ValueError(f"action-group successor {ordinal} proof disagrees")
            if termination != {"reason": "subtree-solved", "value_status": "exact-sign"}:
                raise ValueError(f"action-group successor {ordinal} termination changed")
        else:
            if proven is not None or not -1.0 <= value <= 1.0:
                raise ValueError(f"action-group successor {ordinal} value is invalid")
            if termination != {
                "reason": expected_root_termination,
                "value_status": "backed-up-at-root-termination",
            }:
                raise ValueError(f"action-group successor {ordinal} termination changed")
        _positive_uint(successor.get("visits"), f"action-group successor {ordinal} visits")
        _uint(
            successor.get("selection_visits"),
            f"action-group successor {ordinal} selection visits",
        )
        order.append((str(successor_id), transcript))
    if order != sorted(order) or len({item[0] for item in order}) != len(order):
        raise ValueError("action-group successors are not uniquely canonical-ordered")
    return json.loads(canonical_json_bytes(row))


def load_complete_turn_action_groups(paths: Iterable[pathlib.Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(validate_complete_turn_action_group(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("complete-turn action-group inputs contain no rows")
    return rows


def merge_complete_turn_action_groups(
    shallow_rows: Sequence[Mapping[str, object]],
    deep_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Replace a strict subset of shallow groups with deeper fixed-work rows."""

    shallow = [validate_complete_turn_action_group(dict(row)) for row in shallow_rows]
    deep = [validate_complete_turn_action_group(dict(row)) for row in deep_rows]
    if not shallow or not deep:
        raise ValueError("shallow and deep action-group inputs must be nonempty")

    def index(rows: Sequence[dict[str, object]], label: str) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            group_id = str(row["group"]["group_id"])
            if group_id in result:
                raise ValueError(f"{label} action groups repeat a group_id")
            result[group_id] = row
        return result

    shallow_by_id = index(shallow, "shallow")
    deep_by_id = index(deep, "deep")
    if not set(deep_by_id) < set(shallow_by_id):
        raise ValueError("deep action groups must be a strict subset of shallow groups")
    for group_id, deep_row in deep_by_id.items():
        shallow_row = shallow_by_id[group_id]
        for field in (
            "feature_schema",
            "source_bundle_body_sha256",
            "teacher",
            "ranking",
            "split",
        ):
            if deep_row[field] != shallow_row[field]:
                raise ValueError("deep action group changed an immutable binding")
        shallow_group = shallow_row["group"]
        deep_group = deep_row["group"]
        for field in (
            "group_id",
            "parent_identity",
            "identity_algorithm",
            "parent_mover",
            "source_binding",
        ):
            if deep_group[field] != shallow_group[field]:
                raise ValueError("deep action group changed its canonical root")
        if (
            int(deep_group["work_budget"]["max_tree_nodes"])
            <= int(shallow_group["work_budget"]["max_tree_nodes"])
        ):
            raise ValueError("deep action group did not increase fixed tree work")
    merged = {**shallow_by_id, **deep_by_id}
    return sorted(
        merged.values(),
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            str(row["group"]["group_id"]),
        ),
    )


def merge_complete_turn_action_group_files(
    *, shallow: pathlib.Path, deep: pathlib.Path
) -> bytes:
    rows = merge_complete_turn_action_groups(
        load_complete_turn_action_groups((shallow,)),
        load_complete_turn_action_groups((deep,)),
    )
    return b"".join(canonical_json_bytes(row) for row in rows)


def build_complete_turn_successor_labels(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate validated rows without changing any teacher value or ranking."""

    normalized = [validate_complete_turn_action_group(dict(row)) for row in rows]
    if not normalized:
        raise ValueError("complete-turn successor labels require at least one group")
    if any(not row["group"]["successors_exhaustive"] for row in normalized):
        raise ValueError(
            "complete-turn successor labels require exhaustive legal successors"
        )
    first = normalized[0]
    immutable = (
        first["feature_schema"],
        first["source_bundle_body_sha256"],
        first["teacher"],
        first["ranking"],
    )
    if any(
        (
            row["feature_schema"],
            row["source_bundle_body_sha256"],
            row["teacher"],
            row["ranking"],
        )
        != immutable
        for row in normalized
    ):
        raise ValueError("complete-turn action groups have mixed immutable bindings")
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    seen: set[str] = set()
    for row in normalized:
        group = dict(row["group"])
        group_id = str(group["group_id"])
        if group_id in seen:
            raise ValueError("complete-turn action groups repeat a group_id")
        seen.add(group_id)
        splits[str(row["split"])].append(group)
    for split in splits:
        splits[split].sort(key=lambda group: str(group["group_id"]))
    document: dict[str, object] = {
        "schema": COMPLETE_TURN_SUCCESSOR_LABELS_SCHEMA,
        "feature_schema": first["feature_schema"],
        "source_bundle_body_sha256": first["source_bundle_body_sha256"],
        "teacher": first["teacher"],
        "ranking": first["ranking"],
        "splits": splits,
        "protected_tests_opened": False,
    }
    document["body_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return validate_complete_turn_successor_labels(document)


def validate_complete_turn_successor_labels(value: object) -> dict[str, object]:
    expected = {
        "schema",
        "feature_schema",
        "source_bundle_body_sha256",
        "teacher",
        "ranking",
        "splits",
        "protected_tests_opened",
        "body_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("complete-turn successor-label artifact is malformed")
    body = dict(value)
    claimed = body.pop("body_sha256")
    if not _valid_sha256(claimed) or claimed != sha256_bytes(canonical_json_bytes(body)):
        raise ValueError("complete-turn successor-label body SHA-256 mismatch")
    if (
        value.get("schema") != COMPLETE_TURN_SUCCESSOR_LABELS_SCHEMA
        or value.get("feature_schema") != features.FEATURE_SCHEMA
        or not _valid_sha256(value.get("source_bundle_body_sha256"))
        or value.get("ranking") != _ACTION_GROUP_RANKING
        or value.get("protected_tests_opened") is not False
    ):
        raise ValueError("complete-turn successor-label policy changed")
    _validate_action_group_teacher(value.get("teacher"))
    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("complete-turn successor-label splits are malformed")
    rows: list[dict[str, object]] = []
    observed: set[str] = set()
    for split in ("train", "validation"):
        groups = splits.get(split)
        if not isinstance(groups, list):
            raise ValueError("complete-turn successor-label split is not an array")
        order: list[str] = []
        for group in groups:
            row = {
                "schema": COMPLETE_TURN_ACTION_GROUP_SCHEMA,
                "feature_schema": value["feature_schema"],
                "source_bundle_body_sha256": value["source_bundle_body_sha256"],
                "teacher": value["teacher"],
                "ranking": value["ranking"],
                "split": split,
                "group": group,
            }
            normalized = validate_complete_turn_action_group(row)
            if normalized["group"]["successors_exhaustive"] is not True:
                raise ValueError(
                    "complete-turn successor labels require exhaustive legal successors"
                )
            group_id = str(normalized["group"]["group_id"])
            if group_id in observed:
                raise ValueError("complete-turn successor-label group is duplicated")
            observed.add(group_id)
            order.append(group_id)
            rows.append(normalized)
        if order != sorted(order):
            raise ValueError("complete-turn successor-label groups are not ordered")
    if not rows:
        raise ValueError("complete-turn successor-label artifact contains no groups")
    return json.loads(canonical_json_bytes(value))


def _sample_from_complete_turn_action_group(
    row: Mapping[str, object],
) -> tuple[LabeledSample, LabeledSample]:
    normalized = validate_complete_turn_action_group(dict(row))
    group = normalized["group"]
    source = group["source_binding"]
    mover = int(group["parent_mover"])
    winner = int(source["winner"])
    teacher = _direct_teacher_target(
        _finite_number(group["root_value"], "action-group root value"),
        mover,
        bool(group["root_solved"]),
        _proven_winner(group["proven_winner"]),
    )
    target = 0.75 * teacher + 0.25 * (1.0 if winner == mover else -1.0)
    state = _prefix_state(source["prefix"])
    active = features.encode_active(state)
    reflected = features.encode_active(state, reflected=True)
    lineage = TeacherLineage(
        schema=COMPLETE_TURN_ACTION_GROUP_SCHEMA,
        position_id=str(source["position_id"]),
        group_id=str(source["group_id"]),
        root_group_id=str(source["root_group_id"]),
        source=str(source["source"]),
        split=str(source["split"]),
        campaign_id=str(source["campaign_id"]),
    )
    return (
        LabeledSample(active, target, 1.0, lineage.root_group_id, (lineage,)),
        LabeledSample(reflected, target, 1.0, lineage.root_group_id, (lineage,)),
    )


def sample_from_teacher_row(row: object) -> tuple[LabeledSample, LabeledSample]:
    if isinstance(row, dict) and row.get("schema") == COMPLETE_TURN_ACTION_GROUP_SCHEMA:
        return _sample_from_complete_turn_action_group(row)
    if not isinstance(row, dict) or row.get("schema") not in {
        TEACHER_SCHEMA,
        RANK4_TEACHER_SCHEMA,
        SEARCH_TEACHER_SCHEMA,
    }:
        raise ValueError("unexpected teacher-row schema")
    schema = str(row["schema"])
    group_id, root_group_id, source, winner, mover, state, weight = _teacher_common(row)
    if schema in {TEACHER_SCHEMA, RANK4_TEACHER_SCHEMA}:
        teacher = (
            _rank4_teacher_value(row, mover)
            if schema == TEACHER_SCHEMA
            else _rank4_fixed_work_teacher_value(row, mover)
        )
        if schema == TEACHER_SCHEMA and row.get("position_id") is None:
            position_id = None
            split = None
            campaign_id = None
        else:
            position_id = _nonempty_string(
                row.get("position_id"), "Rank-4 position_id"
            )
            split = row.get("split")
            if split not in {"train", "validation", "test"}:
                raise ValueError("Rank-4 position teacher split is invalid")
            campaign_id = _nonempty_string(
                row.get("campaign_id"), "Rank-4 position teacher campaign_id"
            )
    else:
        position_id = _nonempty_string(row.get("position_id"), "search position_id")
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError("search teacher split is invalid")
        campaign_id = _nonempty_string(
            row.get("campaign_id"), "search teacher campaign_id"
        )
        teacher = _search_teacher_value(row, mover)
    outcome = 1.0 if winner == mover else -1.0
    target = 0.75 * teacher + 0.25 * outcome
    supplied = row.get("combined_target")
    if supplied is not None and (
        isinstance(supplied, bool)
        or not isinstance(supplied, (int, float))
        or not math.isfinite(float(supplied))
        or abs(float(supplied) - target) > 1e-6
    ):
        raise ValueError("combined_target disagrees with the frozen target policy")
    active = features.encode_active(state)
    reflected = features.encode_active(state, reflected=True)
    lineage = TeacherLineage(
        schema=schema,
        position_id=position_id,
        group_id=group_id,
        root_group_id=root_group_id,
        source=source,
        split=split,
        campaign_id=campaign_id,
    )
    return (
        LabeledSample(active, target, weight, root_group_id, (lineage,)),
        LabeledSample(reflected, target, weight, root_group_id, (lineage,)),
    )


def load_teacher_rows(paths: Iterable[pathlib.Path]) -> list[LabeledSample]:
    samples: list[LabeledSample] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                samples.extend(sample_from_teacher_row(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not samples:
        raise ValueError("teacher inputs contain no samples")
    return samples


def canonical_feature_fingerprint(active: Sequence[int]) -> bytes:
    active = features.validate_active(active)
    variants = (
        active,
        features.reflect_active(active),
        features.rotate_active(active),
        features.reflect_active(features.rotate_active(active)),
    )
    packed = [
        b"".join(int(index).to_bytes(2, "little") for index in variant)
        for variant in variants
    ]
    return hashlib.sha256(min(packed)).digest()


def split_and_purge_samples(
    samples: Sequence[LabeledSample],
    assignments: Mapping[str, str],
) -> tuple[
    dict[str, list[LabeledSample]], dict[str, int], dict[str, int]
]:
    grouped = {"train": [], "validation": [], "test": []}
    for sample in samples:
        split = assignments.get(sample.group_id)
        if split not in grouped:
            raise ValueError(f"no frozen split for teacher group {sample.group_id}")
        if any(
            lineage.root_group_id != sample.group_id
            or (lineage.split is not None and lineage.split != split)
            for lineage in sample.lineages
        ):
            raise ValueError(
                f"teacher lineage disagrees with frozen split for {sample.group_id}"
            )
        grouped[split].append(sample)
    retained: dict[str, list[LabeledSample]] = {}
    removed: dict[str, int] = {}
    aggregated: dict[str, int] = {}
    seen: set[bytes] = set()
    for split in ("train", "validation", "test"):
        rows = sorted(
            grouped[split],
            key=lambda sample: (sample.group_id, sample.active, sample.target),
        )
        eligible = [
            sample
            for sample in rows
            if canonical_feature_fingerprint(sample.active) not in seen
        ]
        removed[split] = len(rows) - len(eligible)
        by_orientation: dict[tuple[int, ...], list[LabeledSample]] = defaultdict(list)
        for sample in eligible:
            by_orientation[sample.active].append(sample)
        kept = []
        for active, observations in sorted(by_orientation.items()):
            if len(observations) == 1:
                kept.append(observations[0])
                continue
            total_weight = sum(observation.weight for observation in observations)
            target = sum(
                observation.target * observation.weight
                for observation in observations
            ) / total_weight
            lineage = [
                {
                    "group_id": observation.group_id,
                    "target": observation.target,
                    "weight": observation.weight,
                }
                for observation in observations
            ]
            group_id = "aggregate:" + sha256_bytes(canonical_json_bytes(lineage))
            lineages = tuple(
                item
                for observation in observations
                for item in observation.lineages
            )
            kept.append(
                LabeledSample(active, target, total_weight, group_id, lineages)
            )
        retained[split] = kept
        aggregated[split] = len(eligible) - len(kept)
        seen.update(canonical_feature_fingerprint(sample.active) for sample in kept)
    return retained, removed, aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--exclusions", type=pathlib.Path, required=True)
    parser.add_argument("--public-jacek", type=pathlib.Path, required=True)
    parser.add_argument("--live-snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--previous-roots", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    manifest = normalize_replay_sources(
        repository=arguments.repository,
        exclusion_path=arguments.exclusions,
        public_jacek_path=arguments.public_jacek,
        live_snapshot_path=arguments.live_snapshot,
        previous_roots_path=arguments.previous_roots,
    )
    output = canonical_json_bytes(manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(output)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
