#!/usr/bin/env python3

"""Freeze append-only public-game sources for a candidate-independent T8 ladder."""

from __future__ import annotations

import collections
import concurrent.futures
import datetime
import json
import pathlib
import sys

from acquire_t7_evidence import (
    PARSEABLE_RECORDS as T7_PARSEABLE_RECORDS,
    RAW_RECORDS as T7_RAW_RECORDS,
    RAW_SNAPSHOTS,
    SOURCE_AGENTS as T7_SOURCE_AGENTS,
    frozen_exclusions,
    read_tsv_game_ids,
)
from build_goal_shell_banks import (
    FRESH_RECORDS,
    ROOT,
    TOOLS,
    extract_states,
    sha256_bytes,
    stable_json,
)


HERE = pathlib.Path(__file__).resolve().parent
RAW_RECORDS = HERE / "t8_evidence_ladder_v1.json"
PARSEABLE_RECORDS = HERE / "t8_evidence_ladder_v2.json"
T7_VALIDATION = HERE / "reference" / "t7_prospective_validation.tsv"
T7_FINAL = HERE / "reference" / "t7_sealed_final.tsv"
T7_VALIDATION_SHA256 = (
    "878ca510b63e50339eaeccc57b50445c6b5915b568ccb26a588b386341c5b002"
)
T7_FINAL_SHA256 = (
    "de7e592610c2ab2842874b5b647fca951cf7d51373b4c0c49d8b255b9543ed59"
)
T7_RAW_SHA256 = (
    "fc409257a9e19cb8664385e9cdf32f07ae4ce196a6a619d67d17acf27fdf230e"
)
T7_PARSEABLE_SHA256 = (
    "207384779c5bccd5bddec97eb6fbc40bf7132698310381d5c840d9e5e74755fc"
)

# The T8 source ladder is derived exclusively from identity fields in these
# immutable, candidate-independent snapshots. One public agent version is
# selected per pseudonym by hash, so repeated submission versions cannot
# dominate the acquisition. The incumbent author's own pseudonym is excluded
# to keep T8 external to locally generated bot families.
IDENTITY_SNAPSHOTS = (
    FRESH_RECORDS,
    *RAW_SNAPSHOTS,
    T7_RAW_RECORDS,
    T7_PARSEABLE_RECORDS,
)
EXCLUDED_PSEUDONYMS = {"Lecorbio"}
SOURCE_SELECTION_SEED = "t8-source-ladder-v1"
VERSION_SELECTION_SEED = "t8-version-v1"


def frozen_source_agents() -> tuple[tuple[int, str], ...]:
    if sha256_bytes(T7_RAW_RECORDS.read_bytes()) != T7_RAW_SHA256:
        raise RuntimeError("T7 raw snapshot hash mismatch")
    if sha256_bytes(T7_PARSEABLE_RECORDS.read_bytes()) != T7_PARSEABLE_SHA256:
        raise RuntimeError("T7 parseable snapshot hash mismatch")

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

    t7_names = {name for _, name in T7_SOURCE_AGENTS}
    eligible_names = sorted(
        set(identities) - t7_names - EXCLUDED_PSEUDONYMS
    )
    selected = []
    for name in eligible_names:
        agent_id = min(
            identities[name],
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
    if len(selected) != 76:
        raise RuntimeError(
            f"expected 76 candidate-independent T8 identities, found {len(selected)}"
        )
    if len({name for _, name in selected}) != len(selected):
        raise RuntimeError("T8 source ladder repeats a pseudonym")
    return tuple(selected)


SOURCE_AGENTS = frozen_source_agents()


def t8_exclusions():
    exclusion_ids, sources = frozen_exclusions()
    for path, expected_sha256 in (
        (T7_VALIDATION, T7_VALIDATION_SHA256),
        (T7_FINAL, T7_FINAL_SHA256),
    ):
        exclusion_ids.update(read_tsv_game_ids(path, expected_sha256))
        sources[str(path.relative_to(ROOT))] = expected_sha256
    for path, expected_sha256 in (
        (T7_RAW_RECORDS, T7_RAW_SHA256),
        (T7_PARSEABLE_RECORDS, T7_PARSEABLE_SHA256),
    ):
        data = path.read_bytes()
        if sha256_bytes(data) != expected_sha256:
            raise RuntimeError(f"snapshot hash mismatch: {path}")
        payload = json.loads(data)
        exclusion_ids.update(int(item["game_id"]) for item in payload["records"])
        sources[str(path.relative_to(ROOT))] = expected_sha256
    sources[str(pathlib.Path(__file__).resolve().relative_to(ROOT))] = sha256_bytes(
        pathlib.Path(__file__).read_bytes()
    )
    return exclusion_ids, dict(sorted(sources.items()))


def freeze():
    if PARSEABLE_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {PARSEABLE_RECORDS}")
    if not RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        exclusion_ids, exclusion_sources = t8_exclusions()

        def acquire(source: tuple[int, str]):
            agent_id, expected_name = source
            records = []
            for game in fetch_games(agent_id):
                item = record(game, agent_id)
                if item is None:
                    continue
                if item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"T8 source identity changed for agent {agent_id}: "
                        f"expected {expected_name!r}, found {item['focus_name']!r}"
                    )
                records.append(item)
            return records

        records_by_game = {}
        # fetch_games already uses four concurrent detail requests. Keep the
        # outer source ladder serial to stay below the public service rate cap.
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
        represented = sum(counts[agent_id] > 0 for agent_id, _ in SOURCE_AGENTS)
        if len(records) < 240 or represented < 24:
            raise RuntimeError(
                f"insufficient disjoint T8 acquisition: total={len(records)} "
                f"represented={represented} counts={counts}"
            )
        payload = {
            "schema": "papersoccer.frozen-t8-evidence-ladder.v1",
            "agent_ids": [agent_id for agent_id, _ in SOURCE_AGENTS],
            "agent_names": [name for _, name in SOURCE_AGENTS],
            "agent_identity_source": (
                "one hash-selected public agent version per pseudonym, derived "
                "only from identity fields in the fixed prior snapshots"
            ),
            "source_selection_seed": SOURCE_SELECTION_SEED,
            "version_selection_seed": VERSION_SELECTION_SEED,
            "excluded_pseudonyms": sorted(EXCLUDED_PSEUDONYMS),
            "selection": (
                "all completed public games from the fixed agents, deduplicated "
                "and excluding every prior raw game plus every T7 bank game"
            ),
            "excluded_game_count": len(exclusion_ids),
            "exclusion_sources": exclusion_sources,
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0
            ).isoformat(),
            "records": records,
        }
        RAW_RECORDS.write_bytes(stable_json(payload))

    raw_data = RAW_RECORDS.read_bytes()
    raw_payload = json.loads(raw_data)
    expected_agents = [agent_id for agent_id, _ in SOURCE_AGENTS]
    if (
        raw_payload.get("schema") != "papersoccer.frozen-t8-evidence-ladder.v1"
        or raw_payload.get("agent_ids") != expected_agents
    ):
        raise RuntimeError("T8 raw snapshot identity mismatch")
    valid_records = []
    rejected = 0
    for item in raw_payload["records"]:
        winner = (
            int(item["player_id"])
            if item.get("won")
            else 1 - int(item["player_id"])
        )
        try:
            extract_states("validation", item, winner, elite_balance=True)
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
            f"insufficient parseable T8 acquisition: total={len(valid_records)} "
            f"represented={represented} counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-t8-evidence-ladder.v2",
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
        f"froze {len(valid_records)} parseable T8 games from {represented} "
        f"focus agents ({rejected} rejected); raw_sha256={sha256_bytes(raw_data)} "
        f"sha256={sha256_bytes(data)}"
    )


if __name__ == "__main__":
    freeze()
