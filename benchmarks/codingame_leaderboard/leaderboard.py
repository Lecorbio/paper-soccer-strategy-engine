#!/usr/bin/env python3
"""Run, validate, rate, and publish the frozen CodinGame-rules tournament.

The native referee owns game rules, protocol I/O, time limits, and bot-process
isolation.  This module owns the reviewed roster, deterministic schedule,
TrueSkill updates, resumable orchestration, and frozen-data contracts.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROSTER_SCHEMA = "papersoccer.codingame-leaderboard-roster.v1"
MATCH_SCHEMA = "papersoccer.codingame-match.v1"
TOURNAMENT_SCHEMA = "papersoccer.codingame-leaderboard-tournament.v1"
CHECKPOINT_SCHEMA = "papersoccer.codingame-leaderboard-checkpoint.v1"
SUMMARY_SCHEMA = "papersoccer.codingame-leaderboard-summary.v1"
SCHEDULE_ALGORITHM = "splitmix64-color-pair-blocks-v2"
RATING_ALGORITHM = "trueskill-decisive-1v1-v1"
SCHEDULE_SEED = 20260813
FIRST_TIMEOUT_MS = 1000
LATER_TIMEOUT_MS = 200
EXPECTED_ENTRANTS = 22
EXPECTED_REGISTERED_BOTS = 22
EXPECTED_GAMES = 990
GAMES_PER_ENTRANT = 90
PLAYER_ONE_GAMES_PER_ENTRANT = 45
SEEDED_MATCHING_ROUNDS = 3
EXPECTED_SIX_GAME_PAIRS = 33
BOT_FORFEIT_CLASSIFICATIONS = frozenset(
    {
        "timeout",
        "crash",
        "stdout-overflow",
        "stderr-overflow",
        "empty-output",
        "invalid-character",
        "illegal-action",
        "incomplete-rebound",
        "output-after-handoff",
        "output-after-terminal",
    }
)
RAW_RESULTS_URL = (
    "https://github.com/Lecorbio/paper-soccer-strategy-engine/blob/main/"
    "benchmarks/codingame_leaderboard/tournament.json"
)
RATING_PARAMETERS = {
    "mu": 25.0,
    "sigma": 25.0 / 3.0,
    "beta": 25.0 / 6.0,
    "tau": 25.0 / 300.0,
    "drawProbability": 0.0,
    "conservativeMultiplier": 3.0,
}

# Python delegates these transcendental operations to the platform C library.
# macOS and glibc can therefore differ by a few ULPs after hundreds of
# sequential TrueSkill updates even though every game and ranking decision is
# identical. Keep the stored values at full precision, but permit only
# numerically insignificant drift when replay-validating an artifact produced
# on another supported runner.
STANDING_FLOAT_REL_TOLERANCE = 1e-12
STANDING_FLOAT_ABS_TOLERANCE = 1e-12
STANDING_FLOAT_FIELDS = frozenset({"score", "mu", "sigma"})

HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = HERE.parents[1]
DEFAULT_ROSTER = HERE / "roster.json"
DEFAULT_TOURNAMENT = HERE / "tournament.json"
DEFAULT_SNAPSHOT = DEFAULT_REPOSITORY / "web/leaderboard/leaderboard-results.js"
_MASK64 = (1 << 64) - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z"
)


class ContractError(ValueError):
    """A checked-in or generated artifact violates the frozen contract."""


class InfrastructureError(RuntimeError):
    """The referee or runner failed independently of a bot result."""


@dataclasses.dataclass(frozen=True)
class Rating:
    mu: float = RATING_PARAMETERS["mu"]
    sigma: float = RATING_PARAMETERS["sigma"]

    @property
    def conservative_score(self) -> float:
        return self.mu - RATING_PARAMETERS["conservativeMultiplier"] * self.sigma


class SplitMix64:
    """Small, fully specified PRNG used only for schedule construction."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            candidate = self.next_u64()
            if candidate < limit:
                return candidate % upper

    def shuffle(self, values: list[Any]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    _require(
        actual == keys,
        f"{label} fields differ from v1 (missing {sorted(keys - actual)}, "
        f"unexpected {sorted(actual - keys)})",
    )


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a nonnegative integer",
    )
    return value


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _parse_utc_timestamp(value: Any, label: str = "timestamp") -> dt.datetime:
    _require(
        isinstance(value, str) and _UTC_TIMESTAMP_PATTERN.fullmatch(value) is not None,
        f"{label} must be an RFC 3339 UTC timestamp ending in Z",
    )
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ContractError(f"{label} is not a valid calendar timestamp") from error
    _require(parsed.utcoffset() == dt.timedelta(0), f"{label} must use UTC")
    return parsed


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid JSON in {path}: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_roster(path: Path = DEFAULT_ROSTER) -> dict[str, Any]:
    roster = _load_json(path)
    _require(isinstance(roster, dict), "roster root must be an object")
    return roster


def _registered_bots(repository: Path, roster: Mapping[str, Any]) -> list[str]:
    registry = roster.get("cmakeRegistry")
    _require(isinstance(registry, dict), "roster.cmakeRegistry must be an object")
    cmake_path = repository / str(registry.get("path", ""))
    variable = str(registry.get("variable", ""))
    _require(variable == "PAPERSOCCER_CODINGAME_BOTS", "unexpected CMake registry variable")
    text = cmake_path.read_text(encoding="utf-8")
    match = re.search(rf"set\s*\(\s*{re.escape(variable)}\b(.*?)\)", text, re.DOTALL)
    _require(match is not None, f"cannot find set({variable} ...) in {cmake_path}")
    body = re.sub(r"#.*", "", match.group(1))
    return body.split()


def validate_roster(
    roster: Mapping[str, Any], repository: Path = DEFAULT_REPOSITORY
) -> list[dict[str, Any]]:
    repository = repository.resolve()
    _require(roster.get("schema") == ROSTER_SCHEMA, "unsupported roster schema")
    entrants = roster.get("entrants")
    _require(isinstance(entrants, list), "roster.entrants must be an array")
    _require(
        len(entrants) == EXPECTED_ENTRANTS,
        f"roster must have exactly {EXPECTED_ENTRANTS} entrants",
    )

    canonical_ids: set[str] = set()
    covered_directories: set[str] = set()
    targets: set[str] = set()
    normalized: list[dict[str, Any]] = []
    required = {
        "id",
        "displayName",
        "submissionPath",
        "submissionSha256",
        "documentationUrl",
        "executableTarget",
        "aliases",
    }
    for position, raw_entrant in enumerate(entrants):
        _require(isinstance(raw_entrant, dict), f"entrant {position} must be an object")
        missing = sorted(required - raw_entrant.keys())
        _require(not missing, f"entrant {position} is missing: {', '.join(missing)}")
        entrant = dict(raw_entrant)
        bot_id = entrant["id"]
        _require(
            isinstance(bot_id, str) and re.fullmatch(r"[a-z0-9_]+", bot_id) is not None,
            f"invalid entrant id: {bot_id!r}",
        )
        _require(bot_id not in canonical_ids, f"duplicate canonical id: {bot_id}")
        canonical_ids.add(bot_id)
        covered_directories.add(bot_id)
        target = entrant["executableTarget"]
        _require(isinstance(target, str) and target, f"{bot_id}: invalid executable target")
        expected_target = (
            "papersoccer_codingame_submission"
            if bot_id == "alpha_beta"
            else f"papersoccer_codingame_{bot_id}_submission"
        )
        _require(
            target == expected_target,
            f"{bot_id}: executableTarget must be {expected_target}",
        )
        _require(target not in targets, f"duplicate executable target: {target}")
        targets.add(target)
        _require(
            isinstance(entrant["displayName"], str) and entrant["displayName"],
            f"{bot_id}: displayName must be nonempty",
        )
        _require(
            isinstance(entrant["documentationUrl"], str)
            and entrant["documentationUrl"].startswith("https://"),
            f"{bot_id}: documentationUrl must be HTTPS",
        )
        submission_path = repository / entrant["submissionPath"]
        try:
            submission_path.resolve().relative_to(repository)
        except ValueError as error:
            raise ContractError(f"{bot_id}: submission path escapes repository") from error
        _require(submission_path.is_file(), f"{bot_id}: missing {entrant['submissionPath']}")
        actual_hash = sha256_file(submission_path)
        _require(
            actual_hash == entrant["submissionSha256"],
            f"{bot_id}: submission hash is stale (actual {actual_hash})",
        )

        aliases = entrant["aliases"]
        _require(isinstance(aliases, list), f"{bot_id}: aliases must be an array")
        for alias in aliases:
            _require(isinstance(alias, dict), f"{bot_id}: alias must be an object")
            alias_id = alias.get("id")
            _require(
                isinstance(alias_id, str) and re.fullmatch(r"[a-z0-9_]+", alias_id) is not None,
                f"{bot_id}: invalid alias id",
            )
            _require(alias_id not in covered_directories, f"duplicate bot directory: {alias_id}")
            covered_directories.add(alias_id)
            alias_path = repository / str(alias.get("submissionPath", ""))
            _require(alias_path.is_file(), f"{bot_id}: missing alias artifact {alias_path}")
            alias_hash = sha256_file(alias_path)
            _require(alias_hash == alias.get("submissionSha256"), f"{alias_id}: stale alias hash")
            _require(
                alias_hash == entrant["submissionSha256"],
                f"{alias_id}: alias artifact is not byte-identical to {bot_id}",
            )
        normalized.append(entrant)

    registered = _registered_bots(repository, roster)
    _require(
        len(registered) == EXPECTED_REGISTERED_BOTS,
        f"CMake must register exactly {EXPECTED_REGISTERED_BOTS} bots",
    )
    _require(len(set(registered)) == len(registered), "CMake bot registry contains duplicates")
    _require(
        set(registered) == covered_directories,
        "leaderboard roster does not exactly cover the CMake bot registry",
    )
    bot_root = repository / "submissions/codingame/bots"
    artifact_directories = {
        path.name for path in bot_root.iterdir() if path.is_dir() and (path / "submission.cpp").is_file()
    }
    _require(
        artifact_directories == set(registered),
        "CMake bot registry does not exactly cover directories containing submission.cpp",
    )
    return normalized


def _round_robin_matchings(ids: Sequence[str], rng: SplitMix64) -> list[list[tuple[str, str]]]:
    seeded = list(ids)
    rng.shuffle(seeded)
    fixed = seeded[0]
    rotating = seeded[1:]
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(len(ids) - 1):
        arrangement = [fixed, *rotating]
        rounds.append(
            [tuple(sorted((arrangement[index], arrangement[-1 - index]))) for index in range(len(ids) // 2)]
        )
        rotating = [rotating[-1], *rotating[:-1]]
    return rounds


def build_schedule(ids: Sequence[str], seed: int = SCHEDULE_SEED) -> list[dict[str, Any]]:
    _require(len(ids) >= 2 and len(ids) % 2 == 0, "schedule requires an even entrant count")
    _require(len(set(ids)) == len(ids), "schedule entrant ids must be unique")
    rng = SplitMix64(seed)
    blocks: list[dict[str, Any]] = []
    canonical_ids = sorted(ids)
    for repetition in range(1, 3):
        for first, second in itertools.combinations(canonical_ids, 2):
            blocks.append(
                {
                    "stage": "color-swapped-round-robin",
                    "stageIndex": repetition,
                    "pair": [first, second],
                }
            )
    matchings = _round_robin_matchings(canonical_ids, rng)
    for round_index, matching in enumerate(
        matchings[:SEEDED_MATCHING_ROUNDS], start=1
    ):
        for first, second in matching:
            blocks.append(
                {
                    "stage": "seeded-perfect-matching",
                    "stageIndex": round_index,
                    "pair": [first, second],
                }
            )
    rng.shuffle(blocks)

    schedule: list[dict[str, Any]] = []
    for block_number, block in enumerate(blocks, start=1):
        first, second = block["pair"]
        for leg, (player_one, player_two) in enumerate(((first, second), (second, first)), start=1):
            schedule.append(
                {
                    "id": f"game-{len(schedule) + 1:04d}",
                    "blockId": f"block-{block_number:03d}",
                    "stage": block["stage"],
                    "stageIndex": block["stageIndex"],
                    "leg": leg,
                    "playerOneId": player_one,
                    "playerTwoId": player_two,
                }
            )
    return schedule


def validate_schedule(
    schedule: Sequence[Mapping[str, Any]], ids: Sequence[str], *, full_contract: bool = True
) -> None:
    expected_games = EXPECTED_GAMES if full_contract else len(schedule)
    _require(len(schedule) == expected_games, f"schedule must have exactly {expected_games} games")
    id_set = set(ids)
    games = {bot_id: 0 for bot_id in ids}
    player_one_games = {bot_id: 0 for bot_id in ids}
    pair_counts: dict[tuple[str, str], int] = {}
    seen_game_ids: set[str] = set()
    blocks: dict[str, list[Mapping[str, Any]]] = {}
    for game in schedule:
        game_id = game.get("id")
        _require(isinstance(game_id, str) and game_id not in seen_game_ids, "duplicate game id")
        seen_game_ids.add(game_id)
        first = game.get("playerOneId")
        second = game.get("playerTwoId")
        _require(first in id_set and second in id_set and first != second, f"{game_id}: invalid players")
        games[first] += 1
        games[second] += 1
        player_one_games[first] += 1
        pair = tuple(sorted((first, second)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        blocks.setdefault(str(game.get("blockId")), []).append(game)
    for block_id, legs in blocks.items():
        _require(len(legs) == 2, f"{block_id}: color-pair block must contain two games")
        _require(
            legs[0]["playerOneId"] == legs[1]["playerTwoId"]
            and legs[0]["playerTwoId"] == legs[1]["playerOneId"],
            f"{block_id}: games are not color-swapped",
        )
    if full_contract:
        _require(
            set(games.values()) == {GAMES_PER_ENTRANT},
            f"every entrant must play exactly {GAMES_PER_ENTRANT} games",
        )
        _require(
            set(player_one_games.values()) == {PLAYER_ONE_GAMES_PER_ENTRANT},
            "every entrant must play player one exactly "
            f"{PLAYER_ONE_GAMES_PER_ENTRANT} times",
        )
        _require(set(pair_counts.values()) <= {4, 6}, "every pair must play four or six games")
        _require(
            sum(count == 6 for count in pair_counts.values())
            == EXPECTED_SIX_GAME_PAIRS,
            f"exactly {EXPECTED_SIX_GAME_PAIRS} pairs must play six games",
        )


def schedule_sha256(schedule: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(canonical_json_bytes(list(schedule)))


def _normal_v(t: float) -> float:
    if t < -10.0:
        x = -t
        inverse = 1.0 / x
        return x + inverse - 2.0 * inverse**3 + 10.0 * inverse**5 - 74.0 * inverse**7
    pdf = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * math.erfc(-t / math.sqrt(2.0))
    return pdf / cdf


def update_ratings(
    winner: Rating,
    loser: Rating,
    parameters: Mapping[str, float] = RATING_PARAMETERS,
) -> tuple[Rating, Rating]:
    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    winner_variance = winner.sigma * winner.sigma + tau * tau
    loser_variance = loser.sigma * loser.sigma + tau * tau
    comparison = math.sqrt(2.0 * beta * beta + winner_variance + loser_variance)
    t = (winner.mu - loser.mu) / comparison
    v = _normal_v(t)
    w = v * (v + t)
    winner_mu = winner.mu + winner_variance / comparison * v
    loser_mu = loser.mu - loser_variance / comparison * v
    winner_sigma = math.sqrt(
        max(0.0, winner_variance * (1.0 - winner_variance / (comparison * comparison) * w))
    )
    loser_sigma = math.sqrt(
        max(0.0, loser_variance * (1.0 - loser_variance / (comparison * comparison) * w))
    )
    return Rating(winner_mu, winner_sigma), Rating(loser_mu, loser_sigma)


def rate_games(
    entrants: Sequence[Mapping[str, Any]], games: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entrant_by_id = {str(entrant["id"]): entrant for entrant in entrants}
    ratings = {bot_id: Rating() for bot_id in entrant_by_id}
    stats: dict[str, dict[str, Any]] = {
        bot_id: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "forfeits": 0,
            "playerOne": {"games": 0, "wins": 0},
            "playerTwo": {"games": 0, "wins": 0},
        }
        for bot_id in entrant_by_id
    }
    pair_stats: dict[tuple[str, str], dict[str, int]] = {
        (row, column): {"games": 0, "wins": 0, "losses": 0}
        for row in entrant_by_id
        for column in entrant_by_id
        if row != column
    }
    for game in games:
        first = str(game["playerOneId"])
        second = str(game["playerTwoId"])
        winner = str(game["winnerId"])
        _require(first in entrant_by_id and second in entrant_by_id, "game contains unknown entrant")
        _require(winner in (first, second), "winner must be one of the scheduled entrants")
        loser = second if winner == first else first
        ratings[winner], ratings[loser] = update_ratings(ratings[winner], ratings[loser])
        for bot_id in (first, second):
            stats[bot_id]["games"] += 1
        stats[first]["playerOne"]["games"] += 1
        stats[second]["playerTwo"]["games"] += 1
        stats[winner]["wins"] += 1
        stats[loser]["losses"] += 1
        stats[winner]["playerOne" if winner == first else "playerTwo"]["wins"] += 1
        forfeit_id = game.get("forfeitId")
        if forfeit_id is not None:
            _require(forfeit_id == loser, "only the losing entrant may be charged with a forfeit")
            stats[loser]["forfeits"] += 1
        pair_stats[(winner, loser)]["games"] += 1
        pair_stats[(winner, loser)]["wins"] += 1
        pair_stats[(loser, winner)]["games"] += 1
        pair_stats[(loser, winner)]["losses"] += 1

    ordered = sorted(
        entrant_by_id,
        key=lambda bot_id: (-ratings[bot_id].conservative_score, bot_id),
    )
    standings: list[dict[str, Any]] = []
    for rank, bot_id in enumerate(ordered, start=1):
        entrant = entrant_by_id[bot_id]
        record = stats[bot_id]
        game_count = record["games"]
        standings.append(
            {
                "rank": rank,
                "id": bot_id,
                "displayName": entrant["displayName"],
                "aliases": [alias["id"] for alias in entrant["aliases"]],
                "score": ratings[bot_id].conservative_score,
                "mu": ratings[bot_id].mu,
                "sigma": ratings[bot_id].sigma,
                "games": game_count,
                "wins": record["wins"],
                "losses": record["losses"],
                "winRate": record["wins"] / game_count if game_count else 0.0,
                "forfeits": record["forfeits"],
                "playerOne": record["playerOne"],
                "playerTwo": record["playerTwo"],
                "submissionSha256": entrant["submissionSha256"],
                "documentationUrl": entrant["documentationUrl"],
            }
        )
    head_to_head: list[dict[str, Any]] = []
    for row in entrant_by_id:
        for column in entrant_by_id:
            if row == column:
                continue
            result = pair_stats[(row, column)]
            count = result["games"]
            head_to_head.append(
                {
                    "rowId": row,
                    "columnId": column,
                    "games": count,
                    "wins": result["wins"],
                    "losses": result["losses"],
                    "score": result["wins"] / count if count else None,
                }
            )
    return standings, head_to_head


def _standings_match(
    recorded: Any, recomputed: Sequence[Mapping[str, Any]]
) -> bool:
    """Compare standings exactly except for cross-platform libm drift."""
    if not isinstance(recorded, list) or len(recorded) != len(recomputed):
        return False
    for recorded_row, recomputed_row in zip(recorded, recomputed):
        if not isinstance(recorded_row, dict):
            return False
        if set(recorded_row) != set(recomputed_row):
            return False
        for field, recomputed_value in recomputed_row.items():
            recorded_value = recorded_row[field]
            if field not in STANDING_FLOAT_FIELDS:
                if recorded_value != recomputed_value:
                    return False
                continue
            if (
                isinstance(recorded_value, bool)
                or not isinstance(recorded_value, (int, float))
                or isinstance(recomputed_value, bool)
                or not isinstance(recomputed_value, (int, float))
            ):
                return False
            recorded_float = float(recorded_value)
            recomputed_float = float(recomputed_value)
            if not math.isfinite(recorded_float) or not math.isfinite(recomputed_float):
                return False
            if not math.isclose(
                recorded_float,
                recomputed_float,
                rel_tol=STANDING_FLOAT_REL_TOLERANCE,
                abs_tol=STANDING_FLOAT_ABS_TOLERANCE,
            ):
                return False
    return True


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_tree_digest(
    repository: Path, excluded_paths: Sequence[Path] = ()
) -> tuple[str, bool]:
    excluded: set[str] = set()
    for path in excluded_paths:
        try:
            excluded.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    output = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    all_paths = [item.decode("utf-8", "surrogateescape") for item in output.split(b"\0") if item]
    paths = sorted(path for path in all_paths if path not in excluded)
    digest = hashlib.sha256()
    for relative in paths:
        path = repository / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    tracked_dirty = subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0
    untracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked_paths = {
        item.decode("utf-8", "surrogateescape") for item in untracked.split(b"\0") if item
    }
    dirty = tracked_dirty or bool(untracked_paths - excluded)
    return digest.hexdigest(), dirty


def _contract_source_digest(repository: Path) -> str:
    candidates: set[Path] = {
        repository / "CMakeLists.txt",
        repository / "benchmarks/codingame_leaderboard/leaderboard.py",
        repository / "benchmarks/codingame_leaderboard/roster.json",
    }
    for root in (repository / "include/papersoccer", repository / "src/core"):
        if root.is_dir():
            candidates.update(path for path in root.rglob("*") if path.is_file())
    codingame_src = repository / "src/codingame"
    if codingame_src.is_dir():
        candidates.update(path for path in codingame_src.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(candidates):
        if not path.is_file():
            continue
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _cpu_description() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _compiler_description(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(
            r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$",
            cache.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        if match:
            try:
                line = subprocess.check_output(
                    [match.group(1), "--version"], text=True, stderr=subprocess.STDOUT
                ).splitlines()[0]
                return line.strip()
            except (OSError, subprocess.CalledProcessError, IndexError):
                return match.group(1)
    return "unknown"


def _resolve_executable(build_dir: Path, target: str) -> Path:
    candidates = [build_dir / target, build_dir / "Release" / target, build_dir / f"{target}.exe"]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ContractError(f"missing executable for target {target} under {build_dir}")


def _runtime_fingerprint(
    repository: Path,
    roster_path: Path,
    entrants: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    referee: Path,
    build_dir: Path,
    excluded_source_paths: Sequence[Path] = (),
) -> tuple[dict[str, Any], dict[str, Path]]:
    executables = {
        str(entrant["id"]): _resolve_executable(build_dir, str(entrant["executableTarget"]))
        for entrant in entrants
    }
    source_tree_digest, dirty = _source_tree_digest(repository, excluded_source_paths)
    environment = {
        "os": platform.platform(),
        "cpu": _cpu_description(),
        "compiler": _compiler_description(build_dir),
    }
    fingerprint = {
        "rosterSha256": sha256_file(roster_path),
        "contractSourceSha256": _contract_source_digest(repository),
        "scheduleSha256": schedule_sha256(schedule),
        "sourceCommit": _git_output(repository, "rev-parse", "HEAD"),
        "sourceTreeSha256": source_tree_digest,
        "sourceTreeDirty": dirty,
        "refereeSha256": sha256_file(referee),
        "executables": {bot_id: sha256_file(path) for bot_id, path in executables.items()},
        "environment": environment,
    }
    fingerprint["sha256"] = _sha256_bytes(canonical_json_bytes(fingerprint))
    return fingerprint, executables


def _participant_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    bot_id = value.get("id")
    return bot_id if isinstance(bot_id, str) and bot_id else None


def _expected_executable_name(bot_id: str) -> str:
    return (
        "papersoccer_codingame_submission"
        if bot_id == "alpha_beta"
        else f"papersoccer_codingame_{bot_id}_submission"
    )


def _winner_from_match(match: Mapping[str, Any], first: str, second: str) -> str:
    outcome = match.get("outcome")
    _require(isinstance(outcome, dict), "native match outcome must be an object")
    winner = outcome.get("winnerId")
    _require(winner in (first, second), "native match outcome does not identify the scheduled winner")
    return str(winner)


def _validate_match_structure(match: Mapping[str, Any]) -> None:
    """Require complete native evidence; legality is checked by native replay validation."""
    _require_exact_keys(
        match,
        {"schema", "participants", "rules", "timeouts", "actions", "outcome", "timings", "provenance"},
        "native match",
    )
    participants = match.get("participants")
    _require(isinstance(participants, dict), "native match participants must be an object")
    _require_exact_keys(participants, {"playerOne", "playerTwo"}, "native match participants")
    for role in ("playerOne", "playerTwo"):
        participant = participants[role]
        _require(isinstance(participant, dict), f"participants.{role} must be an object")
        _require_exact_keys(participant, {"id", "player", "executable"}, f"participants.{role}")
        _require(
            isinstance(participant["id"], str) and participant["id"],
            f"participants.{role}.id must be nonempty",
        )
        expected_player = 0 if role == "playerOne" else 1
        _require(participant["player"] == expected_player, f"participants.{role}.player differs")
        _require(
            isinstance(participant["executable"], str) and participant["executable"],
            f"participants.{role}.executable must be nonempty",
        )
    _require(
        participants["playerOne"]["id"] != participants["playerTwo"]["id"],
        "native match participants must be distinct",
    )

    actions = match.get("actions")
    _require(isinstance(actions, list) and actions, "native match actions must be a nonempty array")
    rules = match.get("rules")
    _require(isinstance(rules, dict), "native match rules must be an object")
    _require_exact_keys(rules, {"width", "height", "goalRule", "blockedRule"}, "native match rules")
    _require(
        rules.get("width") == 8
        and rules.get("height") == 10
        and rules.get("goalRule") == "OwnGoalsAllowed"
        and rules.get("blockedRule") == "MoverLoses",
        "native match rules differ from the frozen CodinGame contract",
    )
    timeouts = match.get("timeouts")
    _require(isinstance(timeouts, dict), "native match timeouts must be an object")
    _require_exact_keys(timeouts, {"firstMillis", "laterMillis"}, "native match timeouts")
    _require(
        timeouts.get("firstMillis") == FIRST_TIMEOUT_MS
        and timeouts.get("laterMillis") == LATER_TIMEOUT_MS,
        "native match timeouts differ from the frozen contract",
    )
    timings = match.get("timings")
    _require(isinstance(timings, dict), "native match timings must be an object")
    _require_exact_keys(timings, {"totalMicros", "playerOne", "playerTwo"}, "native match timings")
    _require_nonnegative_int(timings["totalMicros"], "timings.totalMicros")
    provenance = match.get("provenance")
    _require(isinstance(provenance, dict), "native match provenance must be an object")
    _require_exact_keys(provenance, {"refereeVersion"}, "native match provenance")
    _require(
        isinstance(provenance["refereeVersion"], str) and provenance["refereeVersion"],
        "native match refereeVersion must be nonempty",
    )
    outcome = match.get("outcome")
    _require(isinstance(outcome, dict), "native match outcome must be an object")
    _require_exact_keys(outcome, {"winnerId", "loserId", "reason", "forfeit"}, "native match outcome")
    participant_ids = {
        0: participants["playerOne"]["id"],
        1: participants["playerTwo"]["id"],
    }
    _require(
        outcome["winnerId"] in participant_ids.values()
        and outcome["loserId"] in participant_ids.values()
        and outcome["winnerId"] != outcome["loserId"],
        "native match outcome winner/loser must be the two participants",
    )
    reason = outcome.get("reason")
    _require(reason in {"goal", "blocked_mover", "forfeit"}, "unknown terminal reason")
    if reason == "forfeit":
        forfeit = outcome.get("forfeit")
        _require(isinstance(forfeit, dict), "forfeit outcome must contain details")
        _require_exact_keys(forfeit, {"botId", "classification", "detail"}, "native match forfeit")
        _require(forfeit["botId"] == outcome["loserId"], "forfeit must identify the losing participant")
        _require(
            isinstance(forfeit["classification"], str)
            and forfeit["classification"] in BOT_FORFEIT_CLASSIFICATIONS,
            "forfeit classification is not an allowed bot failure",
        )
        _require(
            isinstance(forfeit["detail"], str) and forfeit["detail"],
            "forfeit detail must be nonempty",
        )
    else:
        _require(outcome.get("forfeit") is None, "rule-terminal outcome cannot contain a forfeit")
    previous_turn = -1
    terminal_seen = False
    terminal_winner: str | None = None
    previous_action = "-"
    previous_position: Mapping[str, Any] | None = None
    decisions = {0: 0, 1: 0}
    duration_sums = {0: 0, 1: 0}
    duration_maxima = {0: 0, 1: 0}
    direction_deltas = (
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
    )
    for index, action in enumerate(actions):
        _require(isinstance(action, dict), f"action {index} must be an object")
        required = {
            "turn", "botId", "player", "opponentAction", "action", "accepted",
            "durationMicros", "deadlineMillis", "failureClassification", "moves",
        }
        _require_exact_keys(action, required, f"action {index}")
        turn = action["turn"]
        _require(
            isinstance(turn, int) and not isinstance(turn, bool) and turn == previous_turn + 1,
            "action turns must be contiguous",
        )
        previous_turn = turn
        player = action["player"]
        _require(
            isinstance(player, int) and not isinstance(player, bool) and player in (0, 1),
            f"action {index} has invalid player",
        )
        _require(player == index % 2, f"action {index} violates alternating complete turns")
        _require(action["botId"] == participant_ids[player], f"action {index} botId/player mismatch")
        _require(
            action["opponentAction"] == previous_action,
            f"action {index} opponentAction differs from the preceding accepted action",
        )
        _require(isinstance(action["accepted"], bool), f"action {index} accepted must be boolean")
        duration = _require_nonnegative_int(action["durationMicros"], f"action {index} durationMicros")
        expected_deadline = FIRST_TIMEOUT_MS if decisions[player] == 0 else LATER_TIMEOUT_MS
        _require(action["deadlineMillis"] == expected_deadline, f"action {index} has invalid deadline")
        decisions[player] += 1
        duration_sums[player] += duration
        duration_maxima[player] = max(duration_maxima[player], duration)
        move_list = action["moves"]
        _require(isinstance(move_list, list), f"action {index} moves must be an array")
        raw_action = action["action"]
        if action["accepted"]:
            _require(
                isinstance(raw_action, str) and re.fullmatch(r"[0-7]+", raw_action) is not None,
                f"accepted action {index} must be a nonempty direction string",
            )
            _require(action["failureClassification"] is None, "accepted action cannot have failure classification")
            _require(
                duration <= expected_deadline * 1000,
                f"accepted action {index} exceeds its decision deadline",
            )
            _require(len(move_list) == len(raw_action), "action path and move evidence lengths differ")
            _require(
                [move.get("direction") for move in move_list] == [int(direction) for direction in raw_action],
                "action path differs from move evidence",
            )
            previous_action = raw_action
        else:
            failure = action["failureClassification"]
            _require(
                isinstance(failure, str) and failure in BOT_FORFEIT_CLASSIFICATIONS,
                "rejected action classification is not an allowed bot failure",
            )
            _require(raw_action is None or isinstance(raw_action, str), "rejected action must preserve text or null")
            _require(move_list == [], "rejected action must not commit any moves")
            if failure == "timeout":
                # A steady-clock deadline can be observed just below an exact
                # millisecond after integer microsecond truncation.
                _require(
                    duration + 1000 >= expected_deadline * 1000,
                    "timeout duration is below its deadline",
                )
            _require(index == len(actions) - 1, "a rejected action must terminate the transcript")
            _require(reason == "forfeit", "rejected action must result in a forfeit")
        for move_index, move in enumerate(move_list):
            _require(isinstance(move, dict), f"action {index} move {move_index} must be an object")
            _require_exact_keys(
                move,
                {"direction", "from", "to", "extraTurn", "statusAfter"},
                f"action {index} move {move_index}",
            )
            _require(
                isinstance(move["direction"], int)
                and not isinstance(move["direction"], bool)
                and move["direction"] in range(8),
                f"action {index} move has invalid direction",
            )
            _require(isinstance(move["extraTurn"], bool), f"action {index} move has invalid extraTurn")
            start = move["from"]
            end = move["to"]
            _require(
                isinstance(start, dict)
                and isinstance(end, dict)
                and set(start) == {"x", "y"}
                and set(end) == {"x", "y"}
                and all(
                    isinstance(point[axis], int) and not isinstance(point[axis], bool)
                    for point in (start, end)
                    for axis in ("x", "y")
                ),
                f"action {index} move has invalid coordinates",
            )
            if previous_position is None:
                _require(start == {"x": 4, "y": 6}, "transcript must start at the center mark")
            if previous_position is not None:
                _require(start == previous_position, "move coordinates are not contiguous")
            delta = direction_deltas[move["direction"]]
            _require(
                end == {"x": start["x"] + delta[0], "y": start["y"] + delta[1]},
                "move coordinates do not match its encoded direction",
            )
            previous_position = end
            if terminal_seen:
                raise ContractError("transcript contains a move after a terminal edge")
            status = move["statusAfter"]
            _require(
                status in {"ongoing", "player_0_wins", "player_1_wins"},
                f"action {index} move has invalid statusAfter",
            )
            if status != "ongoing":
                terminal_seen = True
                terminal_winner = participant_ids[0 if status == "player_0_wins" else 1]
                _require(move_index == len(move_list) - 1, "terminal edge must end its action")
                _require(index == len(actions) - 1, "terminal edge must end the transcript")
            elif move_index < len(move_list) - 1:
                _require(move["extraTurn"] is True, "only a rebound may continue an action")
            elif action["accepted"]:
                _require(move["extraTurn"] is False, "nonterminal rebound cannot end an accepted action")
    if reason in {"goal", "blocked_mover"}:
        _require(terminal_seen, "natural outcome transcript must end in a terminal edge")
        _require(terminal_winner == outcome["winnerId"], "terminal move status differs from outcome winner")
        _require(all(action["accepted"] for action in actions), "natural outcome cannot reject an action")
    else:
        _require(not terminal_seen, "forfeit transcript prefix cannot already be terminal")
        rejected = actions[-1]
        _require(rejected["accepted"] is False, "forfeit transcript must end in a rejected action")
        forfeit = outcome["forfeit"]
        _require(forfeit.get("botId") == rejected["botId"], "forfeit bot differs from rejected action")
        _require(
            forfeit.get("classification") == rejected["failureClassification"],
            "forfeit classification differs from rejected action",
        )
    for player, timing_key in ((0, "playerOne"), (1, "playerTwo")):
        player_timing = timings.get(timing_key)
        _require(isinstance(player_timing, dict), f"timings.{timing_key} must be an object")
        _require_exact_keys(
            player_timing,
            {"decisions", "totalMicros", "maxMicros"},
            f"timings.{timing_key}",
        )
        for field in ("decisions", "totalMicros", "maxMicros"):
            _require_nonnegative_int(player_timing[field], f"timings.{timing_key}.{field}")
        _require(player_timing.get("decisions") == decisions[player], "timing decision count differs")
        _require(player_timing.get("totalMicros") == duration_sums[player], "timing total differs")
        _require(player_timing.get("maxMicros") == duration_maxima[player], "timing maximum differs")
    _require(
        timings["totalMicros"] >= duration_sums[0] + duration_sums[1],
        "timings.totalMicros is smaller than decision evidence",
    )


def _forfeit_from_match(match: Mapping[str, Any], loser: str) -> str | None:
    outcome = match["outcome"]
    forfeit = outcome.get("forfeit")
    if forfeit is None:
        return None
    _require(isinstance(forfeit, dict) and forfeit.get("botId") == loser, "forfeit differs from loser")
    return loser


def normalize_native_match(
    match: Mapping[str, Any], scheduled: Mapping[str, Any]
) -> tuple[str, str | None]:
    _require(match.get("schema") == MATCH_SCHEMA, "native referee returned unsupported match schema")
    _validate_match_structure(match)
    first = str(scheduled["playerOneId"])
    second = str(scheduled["playerTwoId"])
    participants = match.get("participants")
    _require(isinstance(participants, dict), "native match participants must be an object")
    native_first = _participant_id(participants.get("playerOne"))
    native_second = _participant_id(participants.get("playerTwo"))
    _require(
        (native_first, native_second) == (first, second),
        "native match participants do not match the schedule",
    )
    for role, bot_id in (("playerOne", first), ("playerTwo", second)):
        _require(
            participants[role]["executable"] == _expected_executable_name(bot_id),
            f"native match {role} executable differs from the roster target",
        )
    winner = _winner_from_match(match, first, second)
    loser = second if winner == first else first
    _require(match["outcome"].get("loserId") == loser, "native match loser differs from winner")
    return winner, _forfeit_from_match(match, loser)


def _accepted_transcript(match: Mapping[str, Any]) -> str:
    return "/".join(
        str(action["action"]) for action in match["actions"] if action.get("accepted") is True
    )


def validate_match_replay(
    referee: Path,
    match: Mapping[str, Any],
    scheduled: Mapping[str, Any],
) -> None:
    """Replay the accepted prefix through the authoritative C++ rules engine."""
    _require(referee.is_file() and os.access(referee, os.X_OK), "replay referee must be executable")
    winner, forfeit_id = normalize_native_match(match, scheduled)
    transcript = _accepted_transcript(match)
    command = [str(referee), "--validate-transcript", transcript]
    expected_winner: int | None = None
    if forfeit_id is None:
        expected_winner = 0 if winner == scheduled["playerOneId"] else 1
        command.extend(("--expected-winner", str(expected_winner)))
    else:
        command.append("--allow-incomplete")
    try:
        return_code, stdout_data, stderr_data = _run_bounded_process(
            command,
            cwd=referee.parent,
            environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": os.defpath},
            timeout_seconds=30.0,
            output_limit_bytes=1024 * 1024,
            timeout_message="authoritative transcript replay timed out",
            output_message="authoritative transcript replay output exceeds 1 MiB",
        )
    except InfrastructureError as error:
        raise ContractError(str(error)) from error
    if return_code != 0:
        raise ContractError(
            "authoritative transcript replay failed: "
            + stderr_data.decode("utf-8", "replace")[-2000:]
        )
    try:
        replay = json.loads(stdout_data)
    except json.JSONDecodeError as error:
        raise ContractError(f"authoritative transcript replay returned invalid JSON: {error}") from error
    _require(isinstance(replay, dict), "transcript replay result must be an object")
    _require_exact_keys(
        replay,
        {
            "schema", "terminal", "winnerPlayer", "terminalReason",
            "acceptedActionCount", "edgeCount",
        },
        "transcript replay result",
    )
    _require(
        replay["schema"] == "papersoccer.codingame-transcript-validation.v1",
        "unsupported transcript replay schema",
    )
    edge_count = sum(len(action["moves"]) for action in match["actions"] if action["accepted"])
    _require_nonnegative_int(replay["edgeCount"], "transcript replay edgeCount")
    _require(replay["edgeCount"] == edge_count, "transcript replay edge count differs")
    _require_nonnegative_int(replay["acceptedActionCount"], "transcript replay acceptedActionCount")
    _require(
        replay["acceptedActionCount"]
        == sum(action["accepted"] is True for action in match["actions"]),
        "transcript replay action count differs",
    )
    terminal = replay["terminal"]
    _require(isinstance(terminal, bool), "transcript replay terminal must be boolean")
    _require(
        replay["terminalReason"] is None
        or (
            isinstance(replay["terminalReason"], str)
            and replay["terminalReason"] in {"goal", "blocked_mover"}
        ),
        "transcript replay terminalReason must be null, goal, or blocked_mover",
    )
    _require(
        replay["winnerPlayer"] is None
        or (
            isinstance(replay["winnerPlayer"], int)
            and not isinstance(replay["winnerPlayer"], bool)
            and replay["winnerPlayer"] in (0, 1)
        ),
        "transcript replay winner must be null, zero, or one",
    )
    if forfeit_id is None:
        _require(terminal is True, "natural outcome transcript must be terminal")
        _require(replay["winnerPlayer"] == expected_winner, "transcript replay winner differs")
        _require(
            replay["terminalReason"] == match["outcome"]["reason"],
            "transcript replay terminal reason differs",
        )
    else:
        _require(terminal is False, "forfeit transcript prefix cannot already be terminal")
        _require(replay["winnerPlayer"] is None, "nonterminal forfeit prefix cannot have a winner")
        _require(
            replay["terminalReason"] is None,
            "nonterminal forfeit prefix cannot have a terminal reason",
        )


def _read_bounded(path: Path, limit: int) -> bytes:
    size = path.stat().st_size
    if size > limit:
        raise InfrastructureError(f"referee output exceeded {limit} bytes")
    return path.read_bytes()


def _kill_and_reap_process_group(process: subprocess.Popen[Any]) -> None:
    """Best-effort cleanup for the referee and every bot in its new session."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _set_output_file_limit(output_limit_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit_bytes, output_limit_bytes))


def _run_bounded_process(
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    output_limit_bytes: int,
    timeout_message: str,
    output_message: str,
) -> tuple[int, bytes, bytes]:
    _require(timeout_seconds > 0, "infrastructure timeout must be positive")
    _require(output_limit_bytes > 0, "output limit must be positive")
    with tempfile.TemporaryDirectory(prefix="papersoccer-referee-output-") as temporary_name:
        working = Path(temporary_name)
        stdout_path = working / "stdout"
        stderr_path = working / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(arguments),
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=lambda: _set_output_file_limit(output_limit_bytes),
            )
            deadline = time.monotonic() + timeout_seconds
            try:
                while True:
                    result_code = process.poll()
                    if (
                        stdout_path.stat().st_size >= output_limit_bytes
                        or stderr_path.stat().st_size >= output_limit_bytes
                    ):
                        raise InfrastructureError(output_message)
                    if result_code is not None:
                        break
                    if time.monotonic() >= deadline:
                        raise InfrastructureError(timeout_message)
                    time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
            finally:
                _kill_and_reap_process_group(process)
        return (
            result_code,
            _read_bounded(stdout_path, output_limit_bytes),
            _read_bounded(stderr_path, output_limit_bytes),
        )


def run_native_match(
    referee: Path,
    first_executable: Path,
    second_executable: Path,
    scheduled: Mapping[str, Any],
    *,
    infrastructure_timeout_seconds: float = 180.0,
    output_limit_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="papersoccer-leaderboard-") as temporary_name:
        working = Path(temporary_name)
        arguments = [
            str(referee),
            "--player-one",
            str(first_executable),
            "--player-two",
            str(second_executable),
            "--player-one-id",
            str(scheduled["playerOneId"]),
            "--player-two-id",
            str(scheduled["playerTwoId"]),
            "--first-timeout-ms",
            str(FIRST_TIMEOUT_MS),
            "--later-timeout-ms",
            str(LATER_TIMEOUT_MS),
        ]
        clean_environment = {
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PATH": os.defpath,
            "TMPDIR": str(working),
        }
        result_code, stdout_data, stderr_data = _run_bounded_process(
            arguments,
            cwd=working,
            environment=clean_environment,
            timeout_seconds=infrastructure_timeout_seconds,
            output_limit_bytes=output_limit_bytes,
            timeout_message="native referee exceeded its infrastructure timeout",
            output_message=f"referee output exceeded {output_limit_bytes} bytes",
        )
        if result_code != 0:
            diagnostic = stderr_data.decode("utf-8", "replace")[-4000:]
            raise InfrastructureError(
                f"native referee exited {result_code}; stderr tail: {diagnostic}"
            )
        try:
            match = json.loads(stdout_data)
        except json.JSONDecodeError as error:
            raise InfrastructureError(f"native referee returned invalid JSON: {error}") from error
        _require(isinstance(match, dict), "native match JSON root must be an object")
        normalize_native_match(match, scheduled)
        return match


def _contract(
    roster: Mapping[str, Any], roster_path: Path, schedule: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "rules": {
            "ownGoals": "OwnGoalsAllowed",
            "blockedMover": "MoverLoses",
            "label": "CodinGame Paper Soccer rules",
        },
        "protocol": {
            "playerIdThenCompleteTurns": True,
            "firstTimeoutMs": FIRST_TIMEOUT_MS,
            "laterTimeoutMs": LATER_TIMEOUT_MS,
            "invalidActionResult": "forfeit",
        },
        "schedule": {
            "algorithm": SCHEDULE_ALGORITHM,
            "seed": SCHEDULE_SEED,
            "games": EXPECTED_GAMES,
            "gamesPerEntrant": GAMES_PER_ENTRANT,
            "playerOneGamesPerEntrant": PLAYER_ONE_GAMES_PER_ENTRANT,
            "playerTwoGamesPerEntrant": PLAYER_ONE_GAMES_PER_ENTRANT,
            "sha256": schedule_sha256(schedule),
        },
        "rating": {
            "algorithm": RATING_ALGORITHM,
            **RATING_PARAMETERS,
            "ranking": "mu - 3 * sigma, then canonical entrant id",
            "label": "Local CodinGame-style score",
        },
        "roster": {
            "schema": roster["schema"],
            "sha256": sha256_file(roster_path),
            "entrants": roster["entrants"],
        },
    }


def _game_record(scheduled: Mapping[str, Any], match: Mapping[str, Any]) -> dict[str, Any]:
    winner, forfeit_id = normalize_native_match(match, scheduled)
    loser = scheduled["playerTwoId"] if winner == scheduled["playerOneId"] else scheduled["playerOneId"]
    return {
        **dict(scheduled),
        "winnerId": winner,
        "loserId": loser,
        "forfeitId": forfeit_id,
        "match": match,
    }


def _checkpoint_document(
    tournament_id: str,
    generated_at: str,
    fingerprint: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "tournamentId": tournament_id,
        "generatedAtUtc": generated_at,
        "fingerprint": fingerprint,
        "games": list(games),
    }


def run_tournament(
    *,
    repository: Path,
    roster_path: Path,
    referee: Path,
    build_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    resume: bool = False,
    stop_after: int | None = None,
    generated_at_utc: str | None = None,
    infrastructure_timeout_seconds: float = 180.0,
) -> dict[str, Any] | None:
    repository = repository.resolve()
    roster_path = roster_path.resolve()
    referee = referee.resolve()
    build_dir = build_dir.resolve()
    _require(referee.is_file() and os.access(referee, os.X_OK), "referee must be executable")
    roster = load_roster(roster_path)
    entrants = validate_roster(roster, repository)
    ids = [entrant["id"] for entrant in entrants]
    schedule = build_schedule(ids)
    validate_schedule(schedule, ids)
    fingerprint, executables = _runtime_fingerprint(
        repository,
        roster_path,
        entrants,
        schedule,
        referee,
        build_dir,
        (output_path, checkpoint_path, DEFAULT_TOURNAMENT, DEFAULT_SNAPSHOT),
    )
    tournament_id = f"codingame-{SCHEDULE_SEED}-{fingerprint['sha256'][:12]}"
    generated_at = generated_at_utc or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    _parse_utc_timestamp(generated_at, "generatedAtUtc")
    completed: list[dict[str, Any]] = []
    if resume:
        checkpoint = _load_json(checkpoint_path)
        _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "unsupported checkpoint schema")
        _require(checkpoint.get("fingerprint") == fingerprint, "checkpoint fingerprint differs; refusing resume")
        _require(checkpoint.get("tournamentId") == tournament_id, "checkpoint tournament id differs")
        generated_at = checkpoint.get("generatedAtUtc")
        _parse_utc_timestamp(generated_at, "checkpoint generatedAtUtc")
        completed = checkpoint.get("games")
        _require(isinstance(completed, list), "checkpoint games must be an array")
        _require(len(completed) <= len(schedule), "checkpoint has too many games")
        for index, game in enumerate(completed):
            _require(game.get("id") == schedule[index]["id"], "checkpoint is not a schedule prefix")
            _require(
                all(game.get(key) == schedule[index][key] for key in schedule[index]),
                f"checkpoint game {index} differs from schedule",
            )
            match = game.get("match", {})
            normalize_native_match(match, schedule[index])
            normalized = _game_record(schedule[index], match)
            _require(
                all(game.get(key) == normalized[key] for key in ("winnerId", "loserId", "forfeitId")),
                f"checkpoint game {index} summary differs from its match transcript",
            )
    else:
        _require(not checkpoint_path.exists(), "checkpoint exists; pass --resume or choose another path")
        _atomic_write(
            checkpoint_path,
            canonical_json_bytes(_checkpoint_document(tournament_id, generated_at, fingerprint, completed)),
        )

    target_count = len(schedule)
    if stop_after is not None:
        _require(stop_after >= 0, "stop-after must be nonnegative")
        target_count = min(target_count, len(completed) + stop_after)
    for scheduled in schedule[len(completed) : target_count]:
        match = run_native_match(
            referee,
            executables[scheduled["playerOneId"]],
            executables[scheduled["playerTwoId"]],
            scheduled,
            infrastructure_timeout_seconds=infrastructure_timeout_seconds,
        )
        completed.append(_game_record(scheduled, match))
        _atomic_write(
            checkpoint_path,
            canonical_json_bytes(_checkpoint_document(tournament_id, generated_at, fingerprint, completed)),
        )
        print(f"completed {len(completed)}/{len(schedule)}: {scheduled['id']}", flush=True)
    if len(completed) != len(schedule):
        return None

    standings, head_to_head = rate_games(entrants, completed)
    artifact = {
        "schema": TOURNAMENT_SCHEMA,
        "id": tournament_id,
        "generatedAtUtc": generated_at,
        "contract": _contract(roster, roster_path, schedule),
        "provenance": {
            "sourceCommit": fingerprint["sourceCommit"],
            "sourceTreeSha256": fingerprint["sourceTreeSha256"],
            "sourceTreeDirty": fingerprint["sourceTreeDirty"],
            "contractSourceSha256": fingerprint["contractSourceSha256"],
            "referee": {
                "schema": MATCH_SCHEMA,
                "version": completed[0]["match"]["provenance"]["refereeVersion"],
                "sha256": fingerprint["refereeSha256"],
            },
            "executables": fingerprint["executables"],
            "environment": fingerprint["environment"],
        },
        "games": completed,
        "standings": standings,
        "headToHead": head_to_head,
    }
    validate_tournament(
        artifact,
        roster,
        repository,
        roster_path=roster_path,
        referee=referee,
        require_referee_hash=True,
    )
    _atomic_write(output_path, canonical_json_bytes(artifact))
    return artifact


def _validate_tournament_provenance(
    provenance: Any,
    entrant_ids: Sequence[str],
    referee: Path,
    repository: Path,
    *,
    require_current_sources: bool,
    require_referee_hash: bool,
) -> None:
    _require(isinstance(provenance, dict), "tournament provenance must be an object")
    _require_exact_keys(
        provenance,
        {
            "sourceCommit", "sourceTreeSha256", "sourceTreeDirty", "contractSourceSha256",
            "referee", "executables", "environment",
        },
        "tournament provenance",
    )
    _require(
        isinstance(provenance["sourceCommit"], str)
        and _GIT_COMMIT_PATTERN.fullmatch(provenance["sourceCommit"]) is not None,
        "provenance.sourceCommit must be a full Git object id",
    )
    _require_sha256(provenance["sourceTreeSha256"], "provenance.sourceTreeSha256")
    _require(
        isinstance(provenance["sourceTreeDirty"], bool),
        "provenance.sourceTreeDirty must be boolean",
    )
    contract_digest = _require_sha256(
        provenance["contractSourceSha256"], "provenance.contractSourceSha256"
    )
    if require_current_sources:
        _require(
            contract_digest == _contract_source_digest(repository),
            "tournament is stale relative to referee/rules/tooling sources",
        )

    referee_provenance = provenance["referee"]
    _require(isinstance(referee_provenance, dict), "provenance.referee must be an object")
    _require_exact_keys(referee_provenance, {"schema", "version", "sha256"}, "provenance.referee")
    _require(referee_provenance["schema"] == MATCH_SCHEMA, "provenance referee schema differs")
    _require(
        isinstance(referee_provenance["version"], str) and referee_provenance["version"],
        "provenance referee version must be nonempty",
    )
    recorded_referee_hash = _require_sha256(
        referee_provenance["sha256"], "provenance.referee.sha256"
    )
    _require(referee.is_file() and os.access(referee, os.X_OK), "artifact referee must be executable")
    if require_referee_hash:
        _require(
            sha256_file(referee) == recorded_referee_hash,
            "provided referee hash differs from tournament provenance",
        )

    executables = provenance["executables"]
    _require(isinstance(executables, dict), "provenance.executables must be an object")
    _require(
        set(executables) == set(entrant_ids),
        "provenance executable hashes do not exactly cover the canonical roster",
    )
    for bot_id, digest in executables.items():
        _require_sha256(digest, f"provenance.executables.{bot_id}")

    environment = provenance["environment"]
    _require(isinstance(environment, dict), "provenance.environment must be an object")
    _require_exact_keys(environment, {"os", "cpu", "compiler"}, "provenance.environment")
    for field in ("os", "cpu", "compiler"):
        _require(
            isinstance(environment[field], str) and environment[field],
            f"provenance.environment.{field} must be nonempty",
        )


def validate_tournament(
    artifact: Mapping[str, Any],
    roster: Mapping[str, Any],
    repository: Path = DEFAULT_REPOSITORY,
    *,
    referee: Path,
    roster_path: Path = DEFAULT_ROSTER,
    require_current_sources: bool = True,
    require_referee_hash: bool = False,
) -> None:
    repository = repository.resolve()
    referee = referee.resolve()
    entrants = validate_roster(roster, repository)
    ids = [entrant["id"] for entrant in entrants]
    expected_schedule = build_schedule(ids)
    validate_schedule(expected_schedule, ids)
    _require_exact_keys(
        artifact,
        {"schema", "id", "generatedAtUtc", "contract", "provenance", "games", "standings", "headToHead"},
        "tournament",
    )
    _require(artifact.get("schema") == TOURNAMENT_SCHEMA, "unsupported tournament schema")
    _require(
        isinstance(artifact.get("id"), str)
        and re.fullmatch(rf"codingame-{SCHEDULE_SEED}-[0-9a-f]{{12}}", artifact["id"]) is not None,
        "tournament id differs from the v1 fingerprint format",
    )
    _parse_utc_timestamp(artifact.get("generatedAtUtc"), "generatedAtUtc")
    contract = artifact.get("contract")
    _require(isinstance(contract, dict), "tournament contract must be an object")
    expected_contract = _contract(roster, roster_path, expected_schedule)
    _require(contract == expected_contract, "tournament contract differs from the current frozen contract")
    provenance = artifact.get("provenance")
    _validate_tournament_provenance(
        provenance,
        ids,
        referee,
        repository,
        require_current_sources=require_current_sources,
        require_referee_hash=require_referee_hash,
    )
    fingerprint = {
        "rosterSha256": contract["roster"]["sha256"],
        "contractSourceSha256": provenance["contractSourceSha256"],
        "scheduleSha256": contract["schedule"]["sha256"],
        "sourceCommit": provenance["sourceCommit"],
        "sourceTreeSha256": provenance["sourceTreeSha256"],
        "sourceTreeDirty": provenance["sourceTreeDirty"],
        "refereeSha256": provenance["referee"]["sha256"],
        "executables": provenance["executables"],
        "environment": provenance["environment"],
    }
    expected_tournament_id = (
        f"codingame-{SCHEDULE_SEED}-{_sha256_bytes(canonical_json_bytes(fingerprint))[:12]}"
    )
    _require(artifact["id"] == expected_tournament_id, "tournament id differs from its provenance")
    games = artifact.get("games")
    _require(isinstance(games, list), "tournament games must be an array")
    _require(
        len(games) == EXPECTED_GAMES,
        f"publishable tournament must contain all {EXPECTED_GAMES} games",
    )
    for index, (game, scheduled) in enumerate(zip(games, expected_schedule)):
        _require(isinstance(game, dict), f"game {index} must be an object")
        _require_exact_keys(
            game,
            set(scheduled) | {"winnerId", "loserId", "forfeitId", "match"},
            f"game {index}",
        )
        _require(
            all(game.get(key) == value for key, value in scheduled.items()),
            f"game {index} differs from the frozen schedule",
        )
        winner, forfeit_id = normalize_native_match(game.get("match", {}), scheduled)
        loser = scheduled["playerTwoId"] if winner == scheduled["playerOneId"] else scheduled["playerOneId"]
        _require(game.get("winnerId") == winner, f"game {index} winner summary differs from transcript")
        _require(game.get("loserId") == loser, f"game {index} loser summary differs from transcript")
        _require(game.get("forfeitId") == forfeit_id, f"game {index} forfeit summary differs from transcript")
        _require(
            game["match"]["provenance"]["refereeVersion"] == provenance["referee"]["version"],
            f"game {index} referee version differs from tournament provenance",
        )
        _require(
            game["match"]["participants"]["playerOne"]["id"] in provenance["executables"]
            and game["match"]["participants"]["playerTwo"]["id"] in provenance["executables"],
            f"game {index} participants are not bound to executable hashes",
        )
        validate_match_replay(referee, game["match"], scheduled)
    standings, head_to_head = rate_games(entrants, games)
    _require(
        _standings_match(artifact.get("standings"), standings),
        "standings do not recompute from games within the cross-platform floating tolerance",
    )
    _require(artifact.get("headToHead") == head_to_head, "head-to-head data do not recompute exactly")


def build_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    contract = artifact["contract"]
    provenance = artifact["provenance"]
    schedule = contract["schedule"]
    return {
        "schema": SUMMARY_SCHEMA,
        "tournament": {
            "id": artifact["id"],
            "generatedAtUtc": artifact["generatedAtUtc"],
            "entrantCount": len(contract["roster"]["entrants"]),
            "gameCount": schedule["games"],
            "gamesPerEntrant": schedule["gamesPerEntrant"],
            "playerOneGamesPerEntrant": schedule["playerOneGamesPerEntrant"],
            "playerTwoGamesPerEntrant": schedule["playerTwoGamesPerEntrant"],
            "rulesLabel": contract["rules"]["label"],
            "scoringLabel": contract["rating"]["label"],
            "scheduleSeed": schedule["seed"],
            "sourceCommit": provenance["sourceCommit"],
            "environment": {
                "os": provenance["environment"]["os"],
                "cpu": provenance["environment"]["cpu"],
                "compiler": provenance["environment"]["compiler"],
            },
            "rawResultsUrl": RAW_RESULTS_URL,
        },
        "standings": artifact["standings"],
        "headToHead": artifact["headToHead"],
    }


def render_snapshot(artifact: Mapping[str, Any]) -> bytes:
    payload = json.dumps(build_summary(artifact), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "// Generated by benchmarks/codingame_leaderboard/leaderboard.py; do not edit.\n"
        f"globalThis.PAPERSOCCER_CODINGAME_LEADERBOARD_RESULTS = {payload};\n"
    ).encode("utf-8")


def publish_snapshot(
    artifact_path: Path,
    output_path: Path,
    *,
    repository: Path = DEFAULT_REPOSITORY,
    roster_path: Path = DEFAULT_ROSTER,
    referee: Path,
    check: bool = False,
    require_current_sources: bool = True,
) -> None:
    roster = load_roster(roster_path)
    artifact = _load_json(artifact_path)
    _require(isinstance(artifact, dict), "tournament root must be an object")
    validate_tournament(
        artifact,
        roster,
        repository,
        roster_path=roster_path,
        referee=referee,
        require_current_sources=require_current_sources,
    )
    expected = render_snapshot(artifact)
    if check:
        try:
            actual = output_path.read_bytes()
        except FileNotFoundError as error:
            raise ContractError(f"missing published snapshot: {output_path}") from error
        _require(actual == expected, f"published snapshot is stale: {output_path}")
    else:
        _atomic_write(output_path, expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate roster, schedule, and optional tournament")
    validate.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    validate.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    validate.add_argument("--artifact", type=Path)
    validate.add_argument(
        "--referee",
        type=Path,
        help="authoritatively replay every artifact transcript (required with --artifact)",
    )
    validate.add_argument(
        "--allow-historical-sources",
        action="store_true",
        help=(
            "accept recorded historical source provenance while still enforcing the current "
            "roster and schedule and replaying every transcript"
        ),
    )

    run = subparsers.add_parser("run", help="run or resume the serial native tournament")
    run.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    run.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    run.add_argument("--referee", type=Path, required=True)
    run.add_argument("--build-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, default=DEFAULT_TOURNAMENT)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stop-after", type=int)
    run.add_argument("--generated-at-utc")
    run.add_argument("--infrastructure-timeout-seconds", type=float, default=180.0)

    publish = subparsers.add_parser("publish", help="derive or check the compact classic-script snapshot")
    publish.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    publish.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    publish.add_argument("--input", type=Path, default=DEFAULT_TOURNAMENT)
    publish.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT)
    publish.add_argument("--referee", type=Path, required=True)
    publish.add_argument("--check", action="store_true")
    publish.add_argument(
        "--allow-historical-sources",
        action="store_true",
        help=(
            "accept recorded historical source provenance while still enforcing the current "
            "roster and schedule and replaying every transcript"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            roster = load_roster(arguments.roster)
            entrants = validate_roster(roster, arguments.repository)
            schedule = build_schedule([entrant["id"] for entrant in entrants])
            validate_schedule(schedule, [entrant["id"] for entrant in entrants])
            if arguments.artifact is not None:
                _require(arguments.referee is not None, "--artifact requires --referee transcript replay")
                artifact = _load_json(arguments.artifact)
                _require(isinstance(artifact, dict), "tournament root must be an object")
                validate_tournament(
                    artifact,
                    roster,
                    arguments.repository,
                    referee=arguments.referee,
                    roster_path=arguments.roster,
                    require_current_sources=not arguments.allow_historical_sources,
                )
            print(
                f"validated {len(entrants)} entrants and {len(schedule)} deterministic games",
                file=sys.stderr,
            )
        elif arguments.command == "run":
            artifact = run_tournament(
                repository=arguments.repository,
                roster_path=arguments.roster,
                referee=arguments.referee,
                build_dir=arguments.build_dir,
                output_path=arguments.output,
                checkpoint_path=arguments.checkpoint,
                resume=arguments.resume,
                stop_after=arguments.stop_after,
                generated_at_utc=arguments.generated_at_utc,
                infrastructure_timeout_seconds=arguments.infrastructure_timeout_seconds,
            )
            if artifact is None:
                print("stopped with a resumable checkpoint before tournament completion", file=sys.stderr)
        elif arguments.command == "publish":
            publish_snapshot(
                arguments.input,
                arguments.output,
                repository=arguments.repository,
                roster_path=arguments.roster,
                referee=arguments.referee,
                check=arguments.check,
                require_current_sources=not arguments.allow_historical_sources,
            )
    except (ContractError, InfrastructureError, OSError, subprocess.SubprocessError) as error:
        print(f"leaderboard error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
