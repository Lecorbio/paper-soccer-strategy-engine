#!/usr/bin/env python3

"""Acquire a candidate-independent sealed-final source pool for T10."""

from __future__ import annotations

import collections
import concurrent.futures
import datetime
import json
import pathlib
import sys

from acquire_t7_evidence import SOURCE_AGENTS as T7_SOURCE_AGENTS
from acquire_t8_evidence import (
    IDENTITY_SNAPSHOTS as T8_IDENTITY_SNAPSHOTS,
    PARSEABLE_RECORDS as T8_PARSEABLE_RECORDS,
    RAW_RECORDS as T8_RAW_RECORDS,
    SOURCE_AGENTS as T8_SOURCE_AGENTS,
    t8_exclusions,
)
from build_goal_shell_banks import (
    ROOT,
    TOOLS,
    extract_states,
    sha256_bytes,
    stable_json,
)


HERE = pathlib.Path(__file__).resolve().parent
RAW_RECORDS = HERE / "t10_evidence_ladder_v1.json"
PARSEABLE_RECORDS = HERE / "t10_evidence_ladder_v2.json"

T8_RAW_SHA256 = (
    "c2ee19e042dad13bb67aa1a47c4c768fa8233ad8d762d17783405938b0a824c5"
)
T8_PARSEABLE_SHA256 = (
    "780ddfc963fff710cacbbe0083f6ed87cd1f58888a9e85cdfe691645872e1f50"
)

# These identities and their ordering are derived only from immutable metadata
# frozen before any T9 candidate binding. Previously acquired T7/T8 agent ids
# are removed. One unused version per pseudonym is hash-selected so the source
# ladder is diverse by participant rather than dominated by submission churn.
IDENTITY_SNAPSHOTS = (
    *T8_IDENTITY_SNAPSHOTS,
    T8_RAW_RECORDS,
    T8_PARSEABLE_RECORDS,
)
EXCLUDED_PSEUDONYMS = {"Lecorbio"}
VERSION_SELECTION_SEED = "t10-unused-version-v1"
SOURCE_SELECTION_SEED = "t10-source-ladder-v1"


def frozen_source_agents() -> tuple[tuple[int, str], ...]:
    used_agent_ids = {
        agent_id for agent_id, _ in (*T7_SOURCE_AGENTS, *T8_SOURCE_AGENTS)
    }
    identities: dict[str, set[int]] = collections.defaultdict(set)
    for path in IDENTITY_SNAPSHOTS:
        payload = json.loads(path.read_text())
        for item in payload.get("records", []):
            for agent_key, name_key in (
                ("focus_agent_id", "focus_name"),
                ("opponent_agent_id", "opponent_name"),
            ):
                if agent_key in item and name_key in item:
                    identities[str(item[name_key])].add(int(item[agent_key]))

    selected = []
    for name in sorted(set(identities) - EXCLUDED_PSEUDONYMS):
        unused = sorted(identities[name] - used_agent_ids)
        if not unused:
            continue
        agent_id = min(
            unused,
            key=lambda value: sha256_bytes(
                f"{VERSION_SELECTION_SEED}|{value}|{name}".encode()
            ),
        )
        selected.append((agent_id, name))
    selected.sort(
        key=lambda item: sha256_bytes(
            f"{SOURCE_SELECTION_SEED}|{item[0]}|{item[1]}".encode()
        )
    )
    if len(selected) != 86:
        raise RuntimeError(
            f"expected 86 candidate-independent T10 identities, found "
            f"{len(selected)}"
        )
    if len({name for _, name in selected}) != len(selected):
        raise RuntimeError("T10 source ladder repeats a pseudonym")
    if {agent_id for agent_id, _ in selected} & used_agent_ids:
        raise RuntimeError("T10 source ladder repeats a T7/T8 agent id")
    return tuple(selected)


SOURCE_AGENTS = frozen_source_agents()


def frozen_exclusions() -> tuple[set[int], dict[str, str]]:
    exclusion_ids, sources = t8_exclusions()
    for path, expected_sha256 in (
        (T8_RAW_RECORDS, T8_RAW_SHA256),
        (T8_PARSEABLE_RECORDS, T8_PARSEABLE_SHA256),
    ):
        data = path.read_bytes()
        if sha256_bytes(data) != expected_sha256:
            raise RuntimeError(f"snapshot hash mismatch: {path}")
        payload = json.loads(data)
        exclusion_ids.update(int(item["game_id"]) for item in payload["records"])
        sources[str(path.relative_to(ROOT))] = expected_sha256
    sources[str(pathlib.Path(__file__).resolve().relative_to(ROOT))] = (
        sha256_bytes(pathlib.Path(__file__).read_bytes())
    )
    return exclusion_ids, dict(sorted(sources.items()))


def freeze() -> None:
    if PARSEABLE_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {PARSEABLE_RECORDS}")
    if not RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        exclusion_ids, exclusion_sources = frozen_exclusions()

        def acquire(source: tuple[int, str]):
            agent_id, expected_name = source
            records = []
            structurally_skipped = 0
            for game in fetch_games(agent_id):
                try:
                    item = record(game, agent_id)
                except (KeyError, TypeError, ValueError):
                    structurally_skipped += 1
                    continue
                if item is None:
                    continue
                if item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"T10 source identity changed for agent {agent_id}: "
                        f"expected {expected_name!r}, found "
                        f"{item['focus_name']!r}"
                    )
                records.append(item)
            print(
                f"acquired source {agent_id} {expected_name!r}: "
                f"{len(records)} completed games, "
                f"{structurally_skipped} structurally skipped",
                flush=True,
            )
            return records

        records_by_game = {}
        # Each source fetch already uses four detail workers. Keep the outer
        # ladder serial to respect the public service rate cap.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            acquired = executor.map(acquire, SOURCE_AGENTS)
            for records in acquired:
                for item in records:
                    game_id = int(item["game_id"])
                    if game_id in exclusion_ids or game_id in records_by_game:
                        continue
                    records_by_game[game_id] = item
        records = [records_by_game[key] for key in sorted(records_by_game)]
        counts = collections.Counter(
            int(item["focus_agent_id"]) for item in records
        )
        represented = sum(
            counts[agent_id] > 0 for agent_id, _ in SOURCE_AGENTS
        )
        if len(records) < 240 or represented < 24:
            raise RuntimeError(
                f"insufficient disjoint T10 acquisition: total={len(records)} "
                f"represented={represented} counts={counts}"
            )
        payload = {
            "schema": "papersoccer.frozen-t10-evidence-ladder.v1",
            "agent_ids": [agent_id for agent_id, _ in SOURCE_AGENTS],
            "agent_names": [name for _, name in SOURCE_AGENTS],
            "agent_identity_source": (
                "one hash-selected previously unused public agent version per "
                "pseudonym, derived only from identity fields in immutable "
                "pre-T9 snapshots"
            ),
            "source_selection_seed": SOURCE_SELECTION_SEED,
            "version_selection_seed": VERSION_SELECTION_SEED,
            "excluded_pseudonyms": sorted(EXCLUDED_PSEUDONYMS),
            "selection": (
                "all completed public games from the fixed agents, deduplicated "
                "and excluding every prior raw game through the T8 acquisition"
            ),
            "excluded_game_count": len(exclusion_ids),
            "exclusion_sources": exclusion_sources,
            "frozen_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).replace(microsecond=0).isoformat(),
            "records": records,
        }
        RAW_RECORDS.write_bytes(stable_json(payload))

    raw_data = RAW_RECORDS.read_bytes()
    raw_payload = json.loads(raw_data)
    expected_agents = [agent_id for agent_id, _ in SOURCE_AGENTS]
    if (
        raw_payload.get("schema")
        != "papersoccer.frozen-t10-evidence-ladder.v1"
        or raw_payload.get("agent_ids") != expected_agents
    ):
        raise RuntimeError("T10 raw snapshot identity mismatch")
    valid_records = []
    rejected = 0
    for item in raw_payload["records"]:
        winner = (
            int(item["player_id"])
            if item.get("won")
            else 1 - int(item["player_id"])
        )
        try:
            extract_states("test", item, winner, elite_balance=True)
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        valid_records.append(item)
    counts = collections.Counter(
        int(item["focus_agent_id"]) for item in valid_records
    )
    represented = sum(counts[agent_id] > 0 for agent_id, _ in SOURCE_AGENTS)
    if len(valid_records) < 220 or represented < 24:
        raise RuntimeError(
            f"insufficient parseable T10 acquisition: total={len(valid_records)} "
            f"represented={represented} counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-t10-evidence-ladder.v2",
        "agent_ids": expected_agents,
        "agent_names": [name for _, name in SOURCE_AGENTS],
        "selection": "v1 raw snapshot filtered only for exact replay parseability",
        "raw_sha256": sha256_bytes(raw_data),
        "structurally_rejected_games": rejected,
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "records": valid_records,
    }
    data = stable_json(payload)
    PARSEABLE_RECORDS.write_bytes(data)
    print(
        f"froze {len(valid_records)} parseable T10 games from {represented} "
        f"focus agents ({rejected} rejected); "
        f"raw_sha256={sha256_bytes(raw_data)} sha256={sha256_bytes(data)}"
    )


if __name__ == "__main__":
    freeze()
