#!/usr/bin/env python3
"""Freeze and verify the Rank-4/Jacek hybrid campaign boundary.

The tool intentionally treats existing protected banks and replay archives as
opaque bytes.  It hashes them, but never parses a protected bank or replay
payload.  The only existing opening-state data it reads is the explicitly
metadata-only opening identity registry produced by game_review_gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Iterable


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT_ROOT = ROOT / "results" / "rank_4_jacek_hybrid"

CAMPAIGN_SCHEMA = "papersoccer.rank4-jacek-hybrid-campaign.v1"
EXCLUSION_SCHEMA = "papersoccer.live-replay-exclusions.v1"
OPENING_SCHEMA = "papersoccer.opening-bank.v1"
CAMPAIGN_ID = "rank_4_jacek_hybrid-36h-20260813"
T0_EPOCH = 1_786_648_507
T0_UTC = "2026-08-13T19:15:07Z"
T0_WARSAW = "2026-08-13T21:15:07+02:00"
DEADLINE_EPOCH = 1_786_778_107
DEADLINE_UTC = "2026-08-15T07:15:07Z"
DEADLINE_WARSAW = "2026-08-15T09:15:07+02:00"
SOURCE_COMMIT = "a70756a278017edf00ab26811ef3d1e23b402d09"

CONTROL_SOURCE = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
CONTROL_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
CONTROL_BYTES = 98_624
CONTROL_AGENT_ID = 6_604_719
CONTROL_SUBMISSION_ID = 41_114_327

OPENING_IDENTITIES = ROOT / "benchmarks/game_review_gate/opening_identities.json"
PRIOR_EXCLUSION = (
    ROOT
    / "results/jacek_arena_bfm/arena/exclusions"
    / "0ce7cff1d29cbceab1d06a9066eebb6647a31515437cd468ff1a9b1b5f8407f9.json"
)
PRIOR_GAME_RECORDS = ROOT / "results/jacek_arena_bfm/arena/game_records"

# The per-depth counts reproduce the established color-swapped gate totals:
# 153 openings -> 306 games, 53 -> 106, and 106 -> 212.
BANK_PLAN = (
    ("development", "development", ((4, 39), (8, 38), (12, 38), (20, 38))),
    ("validation", "validation", ((4, 14), (8, 13), (12, 13), (20, 13))),
    ("final", "test", ((4, 27), (8, 27), (12, 26), (20, 26))),
)

PROTECTED_ROOTS = (
    "submissions/codingame/promotion",
    "benchmarks/flagship_study",
    "benchmarks/game_review_gate",
    "results/jacek_arena_bfm",
    "submissions/codingame/bots/jacek_nn",
    "submissions/codingame/bots/selfplay_nn_v2",
)

# The original recursive T0 snapshot of ``results/jacek_arena_bfm`` included
# 13 intentionally gitignored scratch files that existed in the freezing
# worktree.  They cannot exist in a clean checkout.  Preserve that original
# identity as canonical while accepting exact, audited successor trees. This is
# deliberately an all-or-nothing identity list: modified files, partial scratch
# trees, and any extra file still fail closed.
PROTECTED_TREE_EQUIVALENTS = (
    (
        {
            "path": "results/jacek_arena_bfm",
            "file_count": 1_103,
            "total_bytes": 1_280_270_812,
            "tree_sha256": (
                "ef49c3272c3f1f23f2cfd11135d71412e87f590dc48c0231f276137a8601842b"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "results/jacek_arena_bfm",
            "file_count": 1_090,
            "total_bytes": 31_646_360,
            "tree_sha256": (
                "7ba2d7220717d867240e9a56cbb4f2aa0cd3a7362850bcb0eacdb48519ded657"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
    (
        {
            "path": "submissions/codingame/bots/jacek_nn",
            "file_count": 35,
            "total_bytes": 1_313_720,
            "tree_sha256": (
                "ad6667bc8f0f9907d38c5c5c0859749ea59e0a7bec027f0415d91d7d0f4fd94a"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "submissions/codingame/bots/jacek_nn",
            "file_count": 35,
            "total_bytes": 1_313_669,
            "tree_sha256": (
                "bc931778d5ca4dc28e537a0c4d8e91fb04509eeac8b900a9762d5f693e0b3e7f"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
    (
        {
            "path": "benchmarks/flagship_study",
            "file_count": 44,
            "total_bytes": 146_223_641,
            "tree_sha256": (
                "9f943bf96f4f0a40c5412f25d3a8cf156d348ea2709a6a8cd33bdc1460cd63f3"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "benchmarks/flagship_study",
            "file_count": 56,
            "total_bytes": 146_862_551,
            "tree_sha256": (
                "e6d01fee1048647b312cd0f88e63330e0b531f2f11c12646af8bfe6a073e3dbd"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
    (
        {
            "path": "benchmarks/flagship_study",
            "file_count": 44,
            "total_bytes": 146_223_641,
            "tree_sha256": (
                "9f943bf96f4f0a40c5412f25d3a8cf156d348ea2709a6a8cd33bdc1460cd63f3"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "benchmarks/flagship_study",
            "file_count": 46,
            "total_bytes": 146_248_470,
            "tree_sha256": (
                "18a1c61342b3d028ffb3d4bf3fecdc08c32a5b1fce2cb8e96b75769d26770291"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
    (
        {
            "path": "benchmarks/game_review_gate",
            "file_count": 27,
            "total_bytes": 5_703_678,
            "tree_sha256": (
                "2cbd39a7733ca725325be92008743a08af26118c2077917494a8b4344011ea5f"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "benchmarks/game_review_gate",
            "file_count": 29,
            "total_bytes": 5_877_116,
            "tree_sha256": (
                "91ff435a28581b44bec37f320a341bba188c02212dc377b0bcfb45c68b3dae33"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
    (
        {
            "path": "benchmarks/game_review_gate",
            "file_count": 27,
            "total_bytes": 5_703_678,
            "tree_sha256": (
                "2cbd39a7733ca725325be92008743a08af26118c2077917494a8b4344011ea5f"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
        {
            "path": "benchmarks/game_review_gate",
            "file_count": 27,
            "total_bytes": 5_703_620,
            "tree_sha256": (
                "8119de4db63bbf50730f1f5efd2f636f4c5109b825b987104e099c1e16a02e96"
            ),
            "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
        },
    ),
)

# The production Rank-4 directory later absorbed the byte-identical predecessor
# submission's unique corpus, trainer, and arena evidence.  Preserve the frozen
# T0 manifest unchanged, but accept exactly this audited consolidation successor
# in place of both historical bot trees.  The production submission itself is
# still checked independently against CONTROL_SHA256 above.
CONSOLIDATED_RANK4_TREE = {
    "path": "submissions/codingame/bots/rank_4",
    "file_count": 31,
    "total_bytes": 7_098_397,
    "tree_sha256": "9182051277640cfd54e198f95f7f76d1fa84c6bf9b61e7f9d88f68dc0e32d686",
    "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
}
CONSOLIDATED_PREDECESSOR_PATH = "submissions/codingame/bots/selfplay_nn_v2"

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FreezeError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def collector_canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FreezeError(f"expected a regular non-symlink file: {path}")
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_identity(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise FreezeError(f"expected a regular directory: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    total = 0
    for candidate in files:
        if candidate.is_symlink():
            raise FreezeError(f"protected tree contains a symlink: {candidate}")
        entry = file_identity(candidate)
        total += int(entry["bytes"])
        local = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(local)
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(entry["sha256"])))
        digest.update(b"\0")
    return {
        "path": relative(path),
        "file_count": len(files),
        "total_bytes": total,
        "tree_sha256": digest.hexdigest(),
        "tree_hash_algorithm": "sha256(path-nul-size-nul-file_digest-nul)/v1",
    }


def protected_tree_identity_matches(
    current: dict[str, Any], frozen: dict[str, Any]
) -> bool:
    if current == frozen:
        return True
    return any(
        frozen == original and current == clean_checkout
        for original, clean_checkout in PROTECTED_TREE_EQUIVALENTS
    )


def write_once(path: pathlib.Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FreezeError(f"refusing to replace different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def derived_seed(role: str, depth: int, pairs: int, attempt: int = 0) -> int:
    material = (
        "papersoccer.rank_4_jacek_hybrid.opening-bank.v1\0"
        f"{T0_EPOCH}\0{role}\0{depth}\0{pairs}\0{attempt}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def candidate_seed(
    role: str, depth: int, pairs: int, record_index: int, attempt: int
) -> int:
    material = (
        "papersoccer.rank_4_jacek_hybrid.opening-record.v1\0"
        f"{T0_EPOCH}\0{role}\0{depth}\0{pairs}\0{record_index}\0{attempt}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def protected_opening_identities() -> tuple[set[str], set[str], dict[str, Any]]:
    payload = json.loads(OPENING_IDENTITIES.read_text(encoding="utf-8"))
    if payload.get("schema") != "papersoccer.game-review-opening-identities.v1":
        raise FreezeError("unexpected metadata-only opening identity schema")
    banks = list(payload.get("gate_banks", [])) + list(
        payload.get("excluded_flagship_banks", [])
    )
    state_hashes: set[str] = set()
    canonical_keys: set[str] = set()
    for bank in banks:
        states = bank.get("state_hashes", [])
        canonicals = bank.get("canonical_keys", [])
        if len(states) != int(bank.get("pairs", -1)) or len(canonicals) != len(states):
            raise FreezeError("opening identity metadata has inconsistent counts")
        if not all(SHA256_RE.fullmatch(str(item)) for item in states + canonicals):
            raise FreezeError("opening identity metadata contains an invalid digest")
        state_hashes.update(states)
        canonical_keys.update(canonicals)
    if len(state_hashes) != 1_600 or len(canonical_keys) != 1_600:
        raise FreezeError("metadata-only opening exclusions no longer contain 1,600 states")
    identity = file_identity(OPENING_IDENTITIES)
    identity.update(
        {
            "schema": payload["schema"],
            "bank_count": len(banks),
            "state_hash_count": len(state_hashes),
            "canonical_key_count": len(canonical_keys),
            "content_policy": "metadata-only; no protected bank bytes parsed",
        }
    )
    return state_hashes, canonical_keys, identity


def parse_generated_bank(content: bytes) -> dict[str, Any]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise FreezeError("opening generator emitted non-UTF-8 bytes") from error
    if len(lines) < 12:
        raise FreezeError("opening generator emitted a truncated bank")
    metadata: dict[str, str] = {}
    header_index = -1
    for index, line in enumerate(lines):
        fields = line.split("\t")
        if fields[0] == "opening_id":
            header_index = index
            if fields != [
                "opening_id",
                "phase",
                "depth",
                "generation_seed",
                "state_hash",
                "canonical_key",
                "to_move",
                "moves",
            ]:
                raise FreezeError("opening bank has an unexpected row header")
            break
        if len(fields) != 2:
            raise FreezeError("opening bank metadata row is malformed")
        metadata[fields[0]] = fields[1]
    if header_index < 0 or metadata.get("schema") != OPENING_SCHEMA:
        raise FreezeError("opening bank schema/header is missing")
    rows = []
    for line in lines[header_index + 1 :]:
        fields = line.split("\t")
        if len(fields) != 8:
            raise FreezeError("opening bank data row is malformed")
        if not SHA256_RE.fullmatch(fields[4]) or not SHA256_RE.fullmatch(fields[5]):
            raise FreezeError("opening bank row contains an invalid state digest")
        rows.append(
            {
                "opening_id": fields[0],
                "state_hash": fields[4],
                "canonical_key": fields[5],
                "line": line,
            }
        )
    if int(metadata.get("pairs", -1)) != len(rows):
        raise FreezeError("opening bank pair count does not match its rows")
    return {"metadata": metadata, "rows": rows}


def run_generator(
    opening_tool: pathlib.Path,
    phase: str,
    depth: int,
    pairs: int,
    seed: int,
    exclusions: Iterable[pathlib.Path],
) -> bytes:
    command = [
        str(opening_tool),
        "generate",
        "--phase",
        phase,
        "--depth",
        str(depth),
        "--pairs",
        str(pairs),
        "--seed",
        str(seed),
    ]
    for exclusion in exclusions:
        command.extend(("--exclude-bank", str(exclusion)))
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeError(f"opening generator failed: {detail}")
    return completed.stdout


def render_composite_bank(
    template: bytes,
    *,
    phase: str,
    depth: int,
    pairs: int,
    header_seed: int,
    row_lines: list[str],
) -> bytes:
    lines = template.decode("utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("opening_id\t")),
        -1,
    )
    if header_index < 0:
        raise FreezeError("could not find the generated opening row header")
    replacements = {
        "phase": phase,
        "depth": str(depth),
        "pairs": str(pairs),
        "generator_seed": str(header_seed),
    }
    metadata = []
    for line in lines[:header_index]:
        key, _, value = line.partition("\t")
        metadata.append(f"{key}\t{replacements.get(key, value)}")
    return ("\n".join(metadata + [lines[header_index]] + row_lines) + "\n").encode(
        "utf-8"
    )


def validate_new_banks(opening_tool: pathlib.Path, banks: list[pathlib.Path]) -> None:
    command = [str(opening_tool), "validate"]
    for bank in banks:
        command.extend(("--bank", str(bank)))
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeError(f"opening validator rejected new banks: {detail}")


def materialize_banks(opening_tool: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not opening_tool.is_file():
        raise FreezeError(f"opening-bank tool is missing: {opening_tool}")
    protected_states, protected_canonicals, registry_identity = (
        protected_opening_identities()
    )
    seen_states = set(protected_states)
    seen_canonicals = set(protected_canonicals)
    prior_new_banks: list[pathlib.Path] = []
    assignments: list[dict[str, Any]] = []

    for role, phase, depth_plan in BANK_PLAN:
        for depth, pairs in depth_plan:
            path = RESULT_ROOT / "openings" / f"{role}_d{depth:02d}.tsv"
            header_seed = derived_seed(role, depth, pairs, 0)
            row_lines: list[str] = []
            accepted_rows: list[dict[str, Any]] = []
            candidate_draws = 0
            maximum_attempt = 0
            template: bytes | None = None
            for record_index in range(pairs):
                accepted_row: dict[str, Any] | None = None
                for attempt in range(65_536):
                    seed = candidate_seed(
                        role, depth, pairs, record_index, attempt
                    )
                    candidate = run_generator(
                        opening_tool, phase, depth, 1, seed, ()
                    )
                    candidate_draws += 1
                    parsed_candidate = parse_generated_bank(candidate)
                    row = parsed_candidate["rows"][0]
                    if (
                        row["state_hash"] in seen_states
                        or row["canonical_key"] in seen_canonicals
                    ):
                        continue
                    accepted_row = row
                    template = candidate
                    maximum_attempt = max(maximum_attempt, attempt)
                    break
                if accepted_row is None:
                    raise FreezeError(
                        f"could not find disjoint record {record_index} for {role} depth {depth}"
                    )
                accepted_rows.append(accepted_row)
                row_lines.append(str(accepted_row["line"]))
                seen_states.add(str(accepted_row["state_hash"]))
                seen_canonicals.add(str(accepted_row["canonical_key"]))
            if template is None:
                raise FreezeError("composite opening bank has no template")
            content = render_composite_bank(
                template,
                phase=phase,
                depth=depth,
                pairs=pairs,
                header_seed=header_seed,
                row_lines=row_lines,
            )
            parsed = parse_generated_bank(content)
            write_once(path, content)
            validate_new_banks(opening_tool, [*prior_new_banks, path])
            identity = file_identity(path)
            identity.update(
                {
                    "role": role,
                    "generator_phase": phase,
                    "depth": depth,
                    "pairs": pairs,
                    "color_swapped_games": pairs * 2,
                    "header_seed": str(header_seed),
                    "candidate_draws": candidate_draws,
                    "maximum_record_attempt": maximum_attempt,
                    "record_seed_derivation": "rank4-jacek-hybrid-opening-record/v1",
                }
            )
            assignments.append(identity)
            prior_new_banks.append(path)

    return assignments, registry_identity


def build_arena_exclusion_registry() -> tuple[dict[str, Any], pathlib.Path, str]:
    prior_bytes = PRIOR_EXCLUSION.read_bytes()
    prior_hash = sha256_bytes(prior_bytes)
    if prior_hash != PRIOR_EXCLUSION.stem:
        raise FreezeError("prior content-addressed exclusion registry hash changed")
    prior = json.loads(prior_bytes)
    if prior.get("schema") != EXCLUSION_SCHEMA:
        raise FreezeError("prior exclusion registry schema changed")

    records: dict[int, dict[str, Any]] = {}
    for source in prior.get("records", []):
        game_id = int(source["game_id"])
        records[game_id] = {
            "game_id": game_id,
            "categories": sorted(set(str(item) for item in source["categories"])),
            "sources": sorted(set(str(item) for item in source["sources"])),
        }

    local_ids: list[int] = []
    for path in sorted(PRIOR_GAME_RECORDS.iterdir(), key=lambda item: item.name):
        if path.is_dir() and path.name.isdecimal():
            game_id = int(path.name)
            local_ids.append(game_id)
            record = records.setdefault(
                game_id,
                {"game_id": game_id, "categories": [], "sources": []},
            )
            record["categories"] = sorted(
                set(record["categories"]) | {"protected_prior_campaign_arena"}
            )
            record["sources"] = sorted(
                set(record["sources"])
                | {"results/jacek_arena_bfm/arena/game_records"}
            )

    payload = {
        "schema": EXCLUSION_SCHEMA,
        "selection": (
            "rank_4_jacek_hybrid boundary frozen at 2026-08-13T19:15:07Z; "
            "union of the prior content-addressed ID-only registry and numeric "
            "prior-campaign game-record directory names; no replay payload was read"
        ),
        "sources": [
            {
                "category": "prior_id_only_registry",
                "path": relative(PRIOR_EXCLUSION),
                "sha256": prior_hash,
                "game_id_count": len(prior.get("records", [])),
            },
            {
                "category": "protected_prior_campaign_arena",
                "path": relative(PRIOR_GAME_RECORDS),
                "tree_sha256": tree_identity(PRIOR_GAME_RECORDS)["tree_sha256"],
                "game_id_count": len(local_ids),
                "selection": "numeric immediate child directory names only",
            },
        ],
        "records": [records[key] for key in sorted(records)],
    }
    content = collector_canonical_json(payload)
    digest = sha256_bytes(content)
    path = RESULT_ROOT / "arena" / "exclusions" / f"{digest}.json"
    write_once(path, content)
    return payload, path, digest


def explicit_protected_banks() -> list[dict[str, Any]]:
    roots = (
        ROOT / "submissions/codingame/promotion",
        ROOT / "benchmarks/flagship_study/openings",
        ROOT / "benchmarks/game_review_gate/openings",
    )
    return [
        file_identity(path)
        for root in roots
        for path in sorted(root.rglob("*.tsv"))
    ]


def rank4_inventory() -> list[dict[str, Any]]:
    root = ROOT / "submissions/codingame/bots/rank_4"
    return [file_identity(path) for path in sorted(root.rglob("*")) if path.is_file()]


def design_antecedents() -> list[dict[str, Any]]:
    paths = (
        "submissions/codingame/bots/rank_4_exchange/submission.cpp",
        "submissions/codingame/bots/rank_4_fullturn_bfm/submission.cpp",
        "submissions/codingame/bots/jacek_native_bfm/README.md",
        "submissions/codingame/bots/jacek_arena_bfm/README.md",
    )
    return [file_identity(ROOT / path) for path in paths]


def create_manifest(opening_tool: pathlib.Path, preregistered_at: str) -> tuple[pathlib.Path, str]:
    if not re.fullmatch(r"20\d\d-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\dZ", preregistered_at):
        raise FreezeError("--preregistered-at-utc must be a whole-second UTC timestamp")
    if sha256_file(CONTROL_SOURCE) != CONTROL_SHA256 or CONTROL_SOURCE.stat().st_size != CONTROL_BYTES:
        raise FreezeError("canonical Rank-4 control source identity changed")

    assignments, opening_identity = materialize_banks(opening_tool)
    exclusion, exclusion_path, exclusion_hash = build_arena_exclusion_registry()
    protected_roots = [tree_identity(ROOT / path) for path in PROTECTED_ROOTS]
    control_files = rank4_inventory()
    control_tree = tree_identity(ROOT / "submissions/codingame/bots/rank_4")

    role_totals = {}
    for role, _, _ in BANK_PLAN:
        selected = [item for item in assignments if item["role"] == role]
        role_totals[role] = {
            "openings": sum(int(item["pairs"]) for item in selected),
            "color_swapped_games": sum(int(item["color_swapped_games"]) for item in selected),
            "result_access": (
                "iterative model selection"
                if role == "development"
                else "candidate selection only"
                if role == "validation"
                else "one-shot after finalist source/SHA lock"
            ),
        }

    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "objective": (
            "Build and continuously improve a Rank-4-derived Jacek-style hybrid "
            "for 36 wall-clock hours, preserving an operationally safe control."
        ),
        "time_boundary": {
            "goal_created_at_epoch": T0_EPOCH,
            "goal_created_at_utc": T0_UTC,
            "goal_created_at_europe_warsaw": T0_WARSAW,
            "deadline_epoch": DEADLINE_EPOCH,
            "deadline_utc": DEADLINE_UTC,
            "deadline_europe_warsaw": DEADLINE_WARSAW,
            "wall_clock_seconds": DEADLINE_EPOCH - T0_EPOCH,
            "preregistered_at_utc": preregistered_at,
        },
        "repository": {
            "commit_at_freeze": SOURCE_COMMIT,
            "worktree_branch_at_freeze": "jacek-arena-bfm-campaign",
            "campaign_branch_created_after_t0": "rank4-jacek-hybrid-36h",
            "note": "The commit is the clean HEAD at T0; hybrid work is intentionally uncommitted after it.",
        },
        "control": {
            "namespace": "rank_4",
            "source": file_identity(CONTROL_SOURCE),
            "tree": control_tree,
            "files": control_files,
            "codingame": {
                "agent_id": CONTROL_AGENT_ID,
                "submission_id": CONTROL_SUBMISSION_ID,
                "historical_rank": 4,
                "completed_games": 90,
                "record": "66-24",
                "remote_digest_disclosure": (
                    "CodinGame upload bytes are editor-attested/fingerprinted; "
                    "the public API does not expose the remote source bytes or digest."
                ),
            },
            "safety_rule": "Never modify rank_4; preserve it as the rollback and paired control.",
        },
        "lineage": {
            "classification": "rank4-derived Jacek hybrid; explicitly not clean-room",
            "inherited": [
                "Rank-4 complete-turn alpha-beta engine and exact game rules",
                "Rank-4 replay book, replay-value anchor, teacher residual, and their historical corpus lineage",
                "Generic repository build, arena, opening-bank, collector, provenance, and test infrastructure",
            ],
            "conceptual_inputs": [
                "Jacek complete rebound turns as actions",
                "mover-relative board representation",
                "neural position evaluation",
                "single-threaded best-first minimax/UCT-style allocation",
            ],
            "design_antecedents": design_antecedents(),
            "prohibited_claims": [
                "clean-room model or engine",
                "fresh-only training lineage",
                "remote API source-byte attestation",
            ],
            "future_input_rule": (
                "Any copied source, model, corpus, replay, action, or label not listed "
                "here must be recorded in an append-only lineage receipt before its result is used."
            ),
        },
        "protected_boundary": {
            "policy": (
                "Protected banks and replay archives were hashed as opaque bytes only; "
                "no sealed bank or replay payload content was parsed for this freeze."
            ),
            "roots": protected_roots,
            "explicit_bank_files": explicit_protected_banks(),
            "root_matches_json": {
                "path": "matches.json",
                "status_at_freeze": "absent",
                "policy": "must remain absent; arena outputs must use campaign-specific paths",
            },
            "opening_identity_exclusions": opening_identity,
        },
        "arena_exclusions": {
            "path": relative(exclusion_path),
            "sha256": exclusion_hash,
            "schema": exclusion["schema"],
            "record_count": len(exclusion["records"]),
            "policy": (
                "Every arena collector invocation must bind this exact registry "
                "and reject all pre-T0/local-prior game IDs before replay fetch."
            ),
        },
        "procedural_openings": {
            "schema": OPENING_SCHEMA,
            "generator": "uniform-legal-move-generator/v1",
            "selection": "splitmix64-unbiased-rejection-sampling/v1",
            "depths": [4, 8, 12, 20],
            "seed_derivation": (
                "Composite bank header and each one-record generator candidate use "
                "separate SHA-256 domain tags over T0_epoch, role, depth, pairs, "
                "record_index, and deterministic rejection attempt."
            ),
            "duplicate_policy": (
                "Reject exact-state or horizontal-reflection canonical overlap "
                "against the 1,600 metadata-only predecessor identities and all earlier new banks."
            ),
            "protected_bank_policy": (
                "No protected bank was opened or passed to the generator. Older promotion-bank "
                "state identities are unavailable as an ID-only registry; domain-separated seeds "
                "and predecessor metadata exclusions are the non-consuming boundary."
            ),
            "role_totals": role_totals,
            "assignments": assignments,
            "final_seal": (
                "Do not run, parse, summarize, or inspect final_* results until a single "
                "finalist source byte count and SHA-256 are locked."
            ),
        },
    }

    content = canonical_json(manifest)
    digest = sha256_bytes(content)
    canonical_path = RESULT_ROOT / "campaign.json"
    addressed_path = RESULT_ROOT / "manifests" / f"{digest}.json"
    write_once(canonical_path, content)
    write_once(addressed_path, content)
    write_once(RESULT_ROOT / "manifest.sha256", f"{digest}  campaign.json\n".encode("ascii"))
    return addressed_path, digest


def verify_manifest() -> tuple[pathlib.Path, str]:
    canonical_path = RESULT_ROOT / "campaign.json"
    if not canonical_path.is_file():
        raise FreezeError("campaign manifest does not exist")
    content = canonical_path.read_bytes()
    payload = json.loads(content)
    if canonical_json(payload) != content:
        raise FreezeError("campaign manifest is not canonical JSON")
    if payload.get("schema") != CAMPAIGN_SCHEMA:
        raise FreezeError("campaign manifest schema changed")
    digest = sha256_bytes(content)
    addressed = RESULT_ROOT / "manifests" / f"{digest}.json"
    if not addressed.is_file() or addressed.read_bytes() != content:
        raise FreezeError("content-addressed campaign manifest is missing or differs")
    if payload["time_boundary"]["goal_created_at_epoch"] != T0_EPOCH:
        raise FreezeError("campaign T0 changed")
    if payload["time_boundary"]["deadline_epoch"] != DEADLINE_EPOCH:
        raise FreezeError("campaign deadline changed")

    control = payload["control"]["source"]
    if file_identity(ROOT / control["path"]) != control:
        raise FreezeError("Rank-4 control source changed")
    current_control_tree = tree_identity(ROOT / "submissions/codingame/bots/rank_4")
    if current_control_tree not in (payload["control"]["tree"], CONSOLIDATED_RANK4_TREE):
        raise FreezeError("Rank-4 control tree changed")
    for expected in payload["protected_boundary"]["roots"]:
        if (
            expected["path"] == CONSOLIDATED_PREDECESSOR_PATH
            and current_control_tree == CONSOLIDATED_RANK4_TREE
            and not (ROOT / CONSOLIDATED_PREDECESSOR_PATH).exists()
        ):
            continue
        current = tree_identity(ROOT / expected["path"])
        if not protected_tree_identity_matches(current, expected):
            raise FreezeError(f"protected tree changed: {expected['path']}")
    if (ROOT / "matches.json").exists():
        raise FreezeError("root matches.json appeared after its absent-state freeze")
    for assignment in payload["procedural_openings"]["assignments"]:
        current = file_identity(ROOT / assignment["path"])
        for key in ("path", "bytes", "sha256"):
            if current[key] != assignment[key]:
                raise FreezeError(f"opening bank changed: {assignment['path']}")
    exclusion = payload["arena_exclusions"]
    if sha256_file(ROOT / exclusion["path"]) != exclusion["sha256"]:
        raise FreezeError("arena exclusion registry changed")
    return addressed, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="write the frozen campaign artifacts")
    create.add_argument("--opening-tool", type=pathlib.Path, required=True)
    create.add_argument("--preregistered-at-utc", required=True)
    subparsers.add_parser("check", help="verify every frozen artifact without opening protected payloads")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            path, digest = create_manifest(
                args.opening_tool.resolve(), args.preregistered_at_utc
            )
        else:
            path, digest = verify_manifest()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "command": args.command,
                    "manifest": relative(path),
                    "sha256": digest,
                    "t0_utc": T0_UTC,
                    "deadline_utc": DEADLINE_UTC,
                },
                sort_keys=True,
            )
        )
        return 0
    except (FreezeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"freeze_campaign: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
