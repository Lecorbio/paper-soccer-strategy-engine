#!/usr/bin/env python3

"""Freeze append-only public-game sources for the prospective T7 ladder."""

from __future__ import annotations

import collections
import csv
import datetime
import io
import json
import pathlib
import sys

from build_goal_shell_banks import (
    ELITE_FINAL_RAW_RECORDS,
    ELITE_FINAL_RECORDS,
    FRESH_RECORDS,
    ROOT,
    RANK1_VALIDATION,
    RANK1_VALIDATION_SHA256,
    TOOLS,
    T3_DEVELOPMENT,
    T3_DEVELOPMENT_SHA256,
    T3_VALIDATION_SOURCE,
    T3_VALIDATION_SOURCE_SHA256,
    T4_FINAL_TEST,
    T4_FINAL_TEST_SHA256,
    T4_VALIDATION,
    T4_VALIDATION_SHA256,
    T5_EXPOSED_VALIDATION,
    T5_EXPOSED_VALIDATION_SHA256,
    T5_REPLACEMENT_EXPOSED_VALIDATION,
    T5_REPLACEMENT_EXPOSED_VALIDATION_SHA256,
    VALIDATION_EXTENSION_RAW_RECORDS,
    VALIDATION_EXTENSION_RECORDS,
    T6_VALIDATION_EXTENSION_RAW_RECORDS,
    T6_VALIDATION_EXTENSION_RECORDS,
    extract_states,
    fresh_records,
    prior_raw_sources,
    rank1_validation_states,
    sha256_bytes,
    stable_json,
)


HERE = pathlib.Path(__file__).resolve().parent
RAW_RECORDS = HERE / "t7_evidence_ladder_v1.json"
PARSEABLE_RECORDS = HERE / "t7_evidence_ladder_v2.json"
T6_EXPOSED_VALIDATION = HERE / "reference" / "t6_exposed_validation.tsv"
T6_EXPOSED_FINAL = HERE / "reference" / "t6_exposed_final.tsv"
T6_EXPOSED_VALIDATION_SHA256 = (
    "69c0f3e78c878ed6e51e599f7207d445dbeb8a8564fecac63ff4105953bf600d"
)
T6_EXPOSED_FINAL_SHA256 = (
    "ef48f9e190aa14cf4b791641ca73276f0794179b3a2f1c3fa26eda32d465cdd4"
)

# Every identity below was predeclared only from focus/opponent metadata in an
# earlier frozen raw snapshot. Repeated elite names intentionally use distinct
# public agent ids; downstream banks retain a per-focus-agent cap.
SOURCE_AGENTS = (
    (6_589_744, "Aketchan"),
    (2_597_500, "field3"),
    (6_573_967, "Waffle3z"),
    (2_767_844, "audrey"),
    (4_792_144, "About"),
    (5_476_643, "mokaspark"),
    (4_070_038, "Tom1"),
    (5_141_540, "Bwvolleyball_2"),
    (5_067_018, "Meruem"),
    (5_056_277, "DrSzuriad"),
    (4_413_390, "TetraktysPhi"),
    (3_047_622, "Adassko"),
    (2_588_583, "daaskare"),
    (6_429_945, "Zylo"),
    (6_249_875, "Konstant"),
    (5_524_151, "rc95401"),
    (4_997_814, "Adkowsky"),
    (5_355_234, "cup_of_tea"),
    (2_848_831, "ItsJasper"),
    (5_458_526, "Deltaspace"),
    (5_455_083, "Deltaspace"),
    (5_455_818, "Deltaspace"),
    (5_028_866, "jacek"),
    (2_633_938, "jacek"),
)

RAW_SNAPSHOTS = (
    ELITE_FINAL_RAW_RECORDS,
    ELITE_FINAL_RECORDS,
    VALIDATION_EXTENSION_RAW_RECORDS,
    VALIDATION_EXTENSION_RECORDS,
    T6_VALIDATION_EXTENSION_RAW_RECORDS,
    T6_VALIDATION_EXTENSION_RECORDS,
)

EVIDENCE_BANKS = (
    (RANK1_VALIDATION, RANK1_VALIDATION_SHA256),
    (T3_DEVELOPMENT, T3_DEVELOPMENT_SHA256),
    (T3_VALIDATION_SOURCE, T3_VALIDATION_SOURCE_SHA256),
    (T4_VALIDATION, T4_VALIDATION_SHA256),
    (T4_FINAL_TEST, T4_FINAL_TEST_SHA256),
    (T5_EXPOSED_VALIDATION, T5_EXPOSED_VALIDATION_SHA256),
    (
        T5_REPLACEMENT_EXPOSED_VALIDATION,
        T5_REPLACEMENT_EXPOSED_VALIDATION_SHA256,
    ),
    (T6_EXPOSED_VALIDATION, T6_EXPOSED_VALIDATION_SHA256),
    (T6_EXPOSED_FINAL, T6_EXPOSED_FINAL_SHA256),
)


def read_tsv_game_ids(path: pathlib.Path, expected_sha256: str) -> set[int]:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise RuntimeError(f"evidence-bank hash mismatch: {path}")
    with io.StringIO(data.decode()) as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    return {
        int(row["source_game_id"])
        for row in rows
        if int(row["source_game_id"]) != 0
    }


def frozen_exclusions():
    _, prior_game_ids, prior_sources = prior_raw_sources()
    fresh, fresh_sources = fresh_records(prior_game_ids)
    exclusion_ids = prior_game_ids | {
        int(record["game_id"]) for _, record, _ in fresh
    }
    sources = prior_sources | fresh_sources
    metadata_identities = set()
    for path in (FRESH_RECORDS, *RAW_SNAPSHOTS):
        data = path.read_bytes()
        payload = json.loads(data)
        for record in payload["records"]:
            exclusion_ids.add(int(record["game_id"]))
            metadata_identities.add(
                (int(record["focus_agent_id"]), str(record["focus_name"]))
            )
            metadata_identities.add(
                (int(record["opponent_agent_id"]), str(record["opponent_name"]))
            )
        sources[str(path.relative_to(ROOT))] = sha256_bytes(data)
    for path, expected_sha256 in EVIDENCE_BANKS:
        exclusion_ids.update(read_tsv_game_ids(path, expected_sha256))
        sources[str(path.relative_to(ROOT))] = expected_sha256
    rank1_validation_states()
    missing = set(SOURCE_AGENTS) - metadata_identities
    if missing:
        raise RuntimeError(
            f"T7 source identities were not present in frozen metadata: {missing}"
        )
    return exclusion_ids, dict(sorted(sources.items()))


def freeze():
    if PARSEABLE_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {PARSEABLE_RECORDS}")
    if not RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        exclusion_ids, exclusion_sources = frozen_exclusions()
        records_by_game = {}
        for agent_id, expected_name in SOURCE_AGENTS:
            for game in fetch_games(agent_id):
                item = record(game, agent_id)
                if item is None:
                    continue
                if item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"T7 source identity changed for agent {agent_id}"
                    )
                game_id = int(item["game_id"])
                if game_id in exclusion_ids or game_id in records_by_game:
                    continue
                records_by_game[game_id] = item
        records = [records_by_game[key] for key in sorted(records_by_game)]
        counts = collections.Counter(
            int(record["focus_agent_id"]) for record in records
        )
        represented = sum(counts[agent_id] > 0 for agent_id, _ in SOURCE_AGENTS)
        if len(records) < 120 or represented < 10:
            raise RuntimeError(
                f"insufficient disjoint T7 acquisition: total={len(records)} "
                f"represented={represented} counts={counts}"
            )
        payload = {
            "schema": "papersoccer.frozen-t7-evidence-ladder.v1",
            "agent_ids": [agent_id for agent_id, _ in SOURCE_AGENTS],
            "agent_names": [name for _, name in SOURCE_AGENTS],
            "agent_identity_source": (
                "predeclared only from focus/opponent identity metadata in prior "
                "frozen raw snapshots"
            ),
            "selection": (
                "all completed public games from the fixed agents, deduplicated "
                "and excluding every prior raw game and evidence-bank source game"
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
    if (raw_payload.get("schema") !=
            "papersoccer.frozen-t7-evidence-ladder.v1"
            or raw_payload.get("agent_ids") != expected_agents):
        raise RuntimeError("T7 raw snapshot identity mismatch")
    valid_records = []
    rejected = 0
    for item in raw_payload["records"]:
        winner = (int(item["player_id"]) if item.get("won")
                  else 1 - int(item["player_id"]))
        try:
            extract_states("validation", item, winner, elite_balance=True)
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        valid_records.append(item)
    counts = collections.Counter(
        int(record["focus_agent_id"]) for record in valid_records
    )
    represented = sum(counts[agent_id] > 0 for agent_id, _ in SOURCE_AGENTS)
    if len(valid_records) < 110 or represented < 10:
        raise RuntimeError(
            f"insufficient parseable T7 acquisition: total={len(valid_records)} "
            f"represented={represented} counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-t7-evidence-ladder.v2",
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
        f"froze {len(valid_records)} parseable T7 games from {represented} "
        f"focus agents ({rejected} rejected); raw_sha256={sha256_bytes(raw_data)} "
        f"sha256={sha256_bytes(data)}"
    )


if __name__ == "__main__":
    freeze()
