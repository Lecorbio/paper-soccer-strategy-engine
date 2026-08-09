#!/usr/bin/env python3

"""Bind active CodinGame promotion artifacts to frozen complete-turn banks."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import pathlib
import sys
from typing import Iterable


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = HERE.parent / "tools"
FRESH_AGENT_ID = 6_600_765
FRESH_RECORDS = HERE / "safe_inward_fresh_games.json"
FRESH_DEVELOPMENT_MAX_GAME = 898_257_912
FRESH_TEST_MIN_GAME = 898_257_927
EXPECTED_FRESH_RECORDS = 87
PRIOR_LOCKED_RECORDS = HERE / "rank1_locked_games.json"
RANK1_VALIDATION = HERE / "reference" / "rank1_validation.tsv"
RANK1_VALIDATION_SHA256 = (
    "5363c74182980e0060083fd47b38bc73babe179609dfc8f4d3e6da3f38c15cd8"
)
T3_DEVELOPMENT = HERE / "reference" / "t3_development.tsv"
T3_DEVELOPMENT_SHA256 = (
    "048b86ab0ba781a7cb1289b1ff4712070af6e384f91fd157895ac9e92772f319"
)
T3_VALIDATION_SOURCE = HERE / "reference" / "t3_validation_source.tsv"
T3_VALIDATION_SOURCE_SHA256 = (
    "e078ee0b9b40672c846a7226f6a22c652e3da996b701de76cded445b146d5719"
)
T4_VALIDATION = HERE / "reference" / "t4_validation.tsv"
T4_VALIDATION_SHA256 = (
    "9ab3d1f63d94e84efa37507b934e2d266aa6c7d91dc69e50ec1b29d88c0a4efb"
)
T4_FINAL_TEST = HERE / "reference" / "t4_final_test.tsv"
T4_FINAL_TEST_SHA256 = (
    "ef48f9e190aa14cf4b791641ca73276f0794179b3a2f1c3fa26eda32d465cdd4"
)
T5_EXPOSED_VALIDATION = HERE / "reference" / "t5_exposed_validation.tsv"
T5_EXPOSED_VALIDATION_SHA256 = (
    "91aee496bae9646ba6f1257490e01ea3657657948897617285cf90bfccf266e2"
)
T5_REPLACEMENT_EXPOSED_VALIDATION = (
    HERE / "reference" / "t5_replacement_exposed_validation.tsv"
)
T5_REPLACEMENT_EXPOSED_VALIDATION_SHA256 = (
    "ae5c3e1998c85a87e1545bd6b74874cd1dd8802e427190b2e398660d745abc92"
)
T6_EXPOSED_VALIDATION = HERE / "reference" / "t6_exposed_validation.tsv"
T6_EXPOSED_VALIDATION_SHA256 = (
    "69c0f3e78c878ed6e51e599f7207d445dbeb8a8564fecac63ff4105953bf600d"
)
T6_EXPOSED_FINAL = HERE / "reference" / "t6_exposed_final.tsv"
T6_EXPOSED_FINAL_SHA256 = (
    "ef48f9e190aa14cf4b791641ca73276f0794179b3a2f1c3fa26eda32d465cdd4"
)
T7_EVIDENCE_MANIFEST = HERE / "reference" / "t7_evidence_manifest.json"
T7_EVIDENCE_MANIFEST_SHA256 = (
    "4bdf71dfb397cd47bb76962125d8dc084b7d37df113166a15ad56ac0f5ebbe9f"
)
T7_PROSPECTIVE_VALIDATION = (
    HERE / "reference" / "t7_prospective_validation.tsv"
)
T7_PROSPECTIVE_VALIDATION_SHA256 = (
    "878ca510b63e50339eaeccc57b50445c6b5915b568ccb26a588b386341c5b002"
)
T7_SEALED_FINAL = HERE / "reference" / "t7_sealed_final.tsv"
T7_SEALED_FINAL_SHA256 = (
    "de7e592610c2ab2842874b5b647fca951cf7d51373b4c0c49d8b255b9543ed59"
)
T7_STRENGTH_PROTOCOL_SHA256 = (
    "83433c7b74ceb38a166d3a52324d7c137fe8db04038c48edc57d839effe82dba"
)
T7_ACQUISITION_SHA256 = (
    "a59e73d6d3843f15bb0edb83e491b9372c1830c4640ca8468fd0914c9c4670e6"
)
T7_FREEZER_SHA256 = (
    "32976fbaa461dfdf65b6d13321fb1a2d045b1344e80e0dbf40b3afe8871cbd57"
)
T7_RAW_SNAPSHOT_SHA256 = (
    "fc409257a9e19cb8664385e9cdf32f07ae4ce196a6a619d67d17acf27fdf230e"
)
T7_PARSEABLE_SNAPSHOT_SHA256 = (
    "207384779c5bccd5bddec97eb6fbc40bf7132698310381d5c840d9e5e74755fc"
)
T7_CANDIDATE_SUBMISSION_SHA256 = (
    "7d7b1a16d173bce56af021f6cb723587ae59006696c55436bd08320f4b2fe800"
)
T8_EVIDENCE_MANIFEST = HERE / "reference" / "t8_evidence_manifest.json"
T8_EVIDENCE_MANIFEST_SHA256 = (
    "0723c580f0e4f01f433bfae83aa71a1ede83c9e7e04cce969dd1818019121c08"
)
T8_PROSPECTIVE_VALIDATION = (
    HERE / "reference" / "t8_prospective_validation.tsv"
)
T8_PROSPECTIVE_VALIDATION_SHA256 = (
    "e670fc39902308b66debd8deb3dc82fb9e0ce0f61562b3078328e99350c54f3b"
)
T8_SEALED_FINAL = HERE / "reference" / "t8_sealed_final.tsv"
T8_SEALED_FINAL_SHA256 = (
    "e6c8efaa094576ad4ac3dc22a69ea595f224aaa64d7c3ecdc39b7e98c7dfb204"
)
T8_STRENGTH_PROTOCOL_SHA256 = (
    "8d73a1c92d43d73a8ebe48a63084f5f5c578ead9516058920c0700165ec3851c"
)
T8_ACQUISITION_SHA256 = (
    "4524c2390818e4cd9eb8e2a81b1b8f21157c52679d4118a330bd8f6aab674f3f"
)
T8_FREEZER_SHA256 = (
    "7f94e896bf86ee9caa4c701bf04a23346cbfbd2cdabbdfdfb636a2333a7bbf50"
)
T8_RAW_SNAPSHOT_SHA256 = (
    "c2ee19e042dad13bb67aa1a47c4c768fa8233ad8d762d17783405938b0a824c5"
)
T8_PARSEABLE_SNAPSHOT_SHA256 = (
    "780ddfc963fff710cacbbe0083f6ed87cd1f58888a9e85cdfe691645872e1f50"
)
T8_PREBIND_BUILDER_SHA256 = (
    "6af50b24a50a245961d773f886ef7d642fdf363ebdfa83ef740ad14a157adb77"
)
T8_PROMOTION_GATE_SHA256 = (
    "9f88a3ec8d181f622f1f634a1fde5cf476ad8a494b3fbf603f968b522d80ad2b"
)
T8_POSTBIND_BUILDER_SHA256 = (
    "d6ac337e96df667c4822ea2b3e2b93ae51d722911e6c704c746d84d2d7df9c5a"
)
T9_EVIDENCE_MANIFEST = HERE / "reference" / "t9_evidence_manifest.json"
T9_EVIDENCE_MANIFEST_SHA256 = (
    "5c4465b6cabec90c8ec0f5bb9cc8b9af65182f03136681e932a08e01de365845"
)
T9_PROSPECTIVE_VALIDATION = (
    HERE / "reference" / "t9_prospective_validation.tsv"
)
T9_PROSPECTIVE_VALIDATION_SHA256 = T8_PROSPECTIVE_VALIDATION_SHA256
T9_SEALED_FINAL = HERE / "reference" / "t9_sealed_final.tsv"
T9_SEALED_FINAL_SHA256 = T8_SEALED_FINAL_SHA256
T9_STRENGTH_PROTOCOL_SHA256 = T8_STRENGTH_PROTOCOL_SHA256
T9_FREEZER_SHA256 = (
    "b822400d3e55571f7ff24ee4c87d79e3e5aa7ae7abf948c8f07bacf54c791b4f"
)
T8_FRONTIER_BOUND_MANIFEST_SHA256 = (
    "fb52a513eb29e074814c1cec8dc0cecbf619b4aea2bfb7797d13a6070ab0b810"
)
T8_FRONTIER_DECISION_SHA256 = (
    "de4ccaf5c52497b93bced41b6ba4ba1bd77c5e5cd735ea3de44a9be2333ca6f3"
)
T8_FRONTIER_DEVELOPMENT_REPORT_SHA256 = (
    "e6ae982d50bcefda59eb12a2481c5815dc4090fb4714f942981207ee2b4f75b2"
)
T8_FRONTIER_SUBMISSION_SHA256 = (
    "35ffa4c9b30327750c1ca5fa50f6d41a282252f98d187c372a74131b148cafe1"
)
T10_EVIDENCE_MANIFEST = HERE / "reference" / "t10_evidence_manifest.json"
T10_EVIDENCE_MANIFEST_SHA256 = (
    "c1ac7e83f01b7685eea906c2006b35485a352c71db75d31ce594567a8b96a14d"
)
T10_PROSPECTIVE_VALIDATION = (
    HERE / "reference" / "t10_prospective_validation.tsv"
)
T10_PROSPECTIVE_VALIDATION_SHA256 = (
    "e648f35df1b60d8597092311230d12d2085c6143dcd02a3ed0aa22999ced4e1d"
)
T10_SEALED_FINAL = HERE / "reference" / "t10_sealed_final.tsv"
T10_SEALED_FINAL_SHA256 = (
    "48ae304aa870a98bc101b647e5f3c3eff024a5ea5576b91e4cf0779baf9b9ddb"
)
T10_STRENGTH_PROTOCOL_SHA256 = T9_STRENGTH_PROTOCOL_SHA256
T10_ACQUISITION_SHA256 = (
    "17e6278b5a848404d9f8b75fea694ca6355142aa9a96142f582dbdf451f18176"
)
T10_FREEZER_SHA256 = (
    "bca87557e7a87e1c9ca2c7ee41343fa1a82a25274900f2d3dc8254c3e3ca23ff"
)
T10_RAW_SNAPSHOT_SHA256 = (
    "74b412255304ec9d3043b2f10c42c63791dfcaa3a1539d94da3a73bbb359a3de"
)
T10_PARSEABLE_SNAPSHOT_SHA256 = (
    "00b912e25122c9600e97ca878669d9bc6ecf5a7f1339a87606154a7b135032db"
)
INCUMBENT_SUBMISSION_SHA256 = (
    "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
)
PRESERVED_INITIAL_SHA256 = (
    "4661c38f195f718d1fac83dc43f3683f5a101dd7b6858eaff2f529cb82b49a01"
)
ELITE_FINAL_AGENTS = (
    (5_471_081, "Deltaspace"),
    (6_075_670, "Laars"),
    (2_632_888, "Marchete"),
    (6_305_973, "Snekkers"),
    (6_574_012, "Waffle3z"),
    (2_602_969, "EricSMSO"),
    (5_773_889, "derjack"),
    (6_273_433, "jacek"),
)
ELITE_FINAL_RAW_RECORDS = HERE / "elite_final_holdout_v1.json"
ELITE_FINAL_RECORDS = HERE / "elite_final_holdout_v2.json"
VALIDATION_EXTENSION_AGENTS = (
    (5_296_809, "matlewan7"),
    (4_553_858, "YurkovAS"),
    (2_851_491, "StephaneLaveau"),
    (2_844_708, "ILove47"),
    (5_069_198, "TLbuis"),
    (2_850_654, "Spoonboy82"),
)
VALIDATION_EXTENSION_RAW_RECORDS = (
    HERE / "prospective_validation_extension_v1.json"
)
VALIDATION_EXTENSION_RECORDS = HERE / "prospective_validation_extension_v2.json"
VALIDATION_EXTENSION_RAW_SHA256 = (
    "ba3926e72cc09bfaf526819b5d94ad88fb34299b19256d33aea356f4372b3b33"
)
VALIDATION_EXTENSION_SHA256 = (
    "d0be20f224350fec8b5e0c147b0d38e1e8f5c0464971268ba417e809c7ac3bb0"
)
MIN_VALIDATION_EXTENSION_RECORDS_PER_AGENT = 5
T6_VALIDATION_EXTENSION_AGENTS = (
    (6_582_435, "Pduhard-"),
    (2_616_930, "trictrac"),
    (3_915_005, "saraneth"),
    (2_639_185, "BrainSolver"),
    (4_003_712, "red1ynx"),
    (3_058_320, "cegprakash"),
)
T6_VALIDATION_EXTENSION_RAW_RECORDS = (
    HERE / "t6_validation_extension_v1.json"
)
T6_VALIDATION_EXTENSION_RECORDS = HERE / "t6_validation_extension_v2.json"
T6_VALIDATION_EXTENSION_RAW_SHA256 = (
    "bcb2b43203e905c4dcf8e0d5f14de85b3c42bc25620ce56a120bc9aee93f5a92"
)
T6_VALIDATION_EXTENSION_SHA256 = (
    "3bfe3fe18c01ec8be2c3be7a01a90ed69953923e50cea0aa31c9f891063c2b8d"
)
MIN_T6_VALIDATION_EXTENSION_RECORDS_PER_AGENT = 5
ELITE_NAMES = {
    "jacek", "Deltaspace", "Marchete", "Snekkers", "Laars",
    "Waffle3z", "EricSMSO", "derjack",
}
DIRECTIONS = ((0, -1), (1, -1), (1, 0), (1, 1),
              (0, 1), (-1, 1), (-1, 0), (-1, -1))
FIELD_WIDTH = 8
ROTATION_HEIGHT = 12
BANK_LIMITS = {"validation": 72}
VALIDATION_FOCUS_AGENT_CAP = 32
VALIDATION_MIN_ELITE_TIER_ROWS = 12
EXPECTED_ELITE_FINAL_RECORDS = 209
EXPECTED_FINAL_TEST_SOURCE_GAMES = 45
EXPECTED_EXPOSED_VALIDATION_SOURCE_GAMES = 55
EXPECTED_VALIDATION_EXTENSION_RECORDS = 61
EXPECTED_T5_REPLACEMENT_EXPOSED_SOURCE_GAMES = 46
EXPECTED_T6_REMAINING_ELITE_RECORDS = 95
EXPECTED_T6_REMAINING_VALIDATION_EXTENSION_RECORDS = 29
EXPECTED_T6_VALIDATION_EXTENSION_RECORDS = 191
EXPECTED_T6_EXTENSION_EXCLUDED_GAMES = 660
EXPECTED_T6_CANDIDATE_RECORDS = 315
EXPECTED_T6_RAW_SHELL_STATES = 1030
EXPECTED_T6_UNIQUE_STATES = 788
EXPECTED_T6_VALIDATION_SOURCE_GAMES = 46
HEADER = (
    "opening_id\tsplit\tstratum\tsource_agent_id\tsource_game_id\t"
    "opponent_agent_id\twinner_player_id\tturn_index\tphysical_edges\t"
    "state_key\tcanonical_key\tball_x\tball_y\tmover\twinner_tier\t"
    "goal_distance_band\t"
    "used_edge_band\tshell_edge_band\topening_family\tobserved_winner_action\t"
    "transcript\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def rotate(point: tuple[int, int]) -> tuple[int, int]:
    return FIELD_WIDTH - point[0], ROTATION_HEIGHT - point[1]


def reflect(point: tuple[int, int]) -> tuple[int, int]:
    return FIELD_WIDTH - point[0], point[1]


def normalized_segment(a: tuple[int, int], b: tuple[int, int]):
    return tuple(sorted((a, b)))


def state_text(ball: tuple[int, int], mover: int, edges) -> str:
    edge_text = ";".join(
        f"{a[0]},{a[1]}-{b[0]},{b[1]}" for a, b in sorted(edges)
    )
    return f"ball={ball[0]},{ball[1]}|mover={mover}|edges={edge_text}"


def state_identity(ball, mover, edges, winner):
    if winner == 1:
        ball = rotate(ball)
        edges = {normalized_segment(rotate(a), rotate(b)) for a, b in edges}
        mover = 1 - mover
    raw = state_text(ball, mover, edges)
    reflected_ball = reflect(ball)
    reflected_edges = {
        normalized_segment(reflect(a), reflect(b)) for a, b in edges
    }
    mirrored = state_text(reflected_ball, mover, reflected_edges)
    return (
        sha256_bytes(raw.encode()),
        sha256_bytes(min(raw, mirrored).encode()),
        ball,
        mover,
        edges,
    )


def apply_action(ball, edges, action: str):
    for character in action:
        if character < "0" or character > "7":
            raise ValueError(f"invalid direction {character!r}")
        dx, dy = DIRECTIONS[ord(character) - ord("0")]
        destination = ball[0] + dx, ball[1] + dy
        edge = normalized_segment(ball, destination)
        if edge in edges:
            raise ValueError(f"reused edge in recorded action {action!r}")
        edges.add(edge)
        ball = destination
    return ball


def tier(name: str, winner_agent_id: int) -> str:
    if winner_agent_id == FRESH_AGENT_ID:
        return "incumbent"
    if name == "jacek":
        return "rank1"
    if name in ELITE_NAMES:
        return "elite"
    return "field"


def edge_band(count: int) -> str:
    if count < 48:
        return "sparse"
    if count < 112:
        return "building"
    return "closed"


def shell_band(edges) -> str:
    count = sum(1 for a, b in edges if max(a[1], b[1]) >= 9)
    if count < 12:
        return "open"
    if count < 28:
        return "layering"
    return "dense"


def opening_family(actions: list[str]) -> str:
    return "/".join(actions[: min(4, len(actions))]) or "initial"


def prior_raw_sources():
    paths = sorted((ROOT / "submissions/codingame/bots").glob("*/arena_batch_*.json"))
    if len(paths) != 8:
        raise ValueError(f"expected eight prior Arena batches, found {len(paths)}")
    records = []
    game_ids = set()
    sources = {}
    for path in paths:
        payload = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
        for loss in payload["loss_records"]:
            if loss.get("won") is not False:
                raise ValueError(f"expected a loss record in {path}")
            game_id = int(loss["game_id"])
            if game_id in game_ids:
                raise ValueError(f"duplicate prior raw game {game_id}")
            game_ids.add(game_id)
            records.append((loss, 1 - int(loss["player_id"])))

    locked_payload = json.loads(PRIOR_LOCKED_RECORDS.read_text())
    sources[str(PRIOR_LOCKED_RECORDS.relative_to(ROOT))] = sha256_bytes(
        PRIOR_LOCKED_RECORDS.read_bytes()
    )
    for record in locked_payload["records"]:
        game_id = int(record["game_id"])
        if game_id in game_ids:
            raise ValueError(f"prior raw corpora overlap at game {game_id}")
        game_ids.add(game_id)
        winner = (int(record["player_id"]) if record.get("won")
                  else 1 - int(record["player_id"]))
        records.append((record, winner))
    if len(game_ids) != 298:
        raise ValueError(f"expected 298 prior raw games, found {len(game_ids)}")
    return records, game_ids, sources


def fresh_records(prior_game_ids: set[int]):
    if not FRESH_RECORDS.exists():
        raise FileNotFoundError(
            f"missing {FRESH_RECORDS}; run with --fetch-fresh once"
        )
    payload = json.loads(FRESH_RECORDS.read_text())
    if payload.get("agent_id") != FRESH_AGENT_ID:
        raise ValueError("fresh corpus has the wrong agent id")
    records = payload.get("records", [])
    game_ids = [int(record["game_id"]) for record in records]
    if len(records) != EXPECTED_FRESH_RECORDS or len(set(game_ids)) != len(game_ids):
        raise ValueError("fresh corpus has the wrong count or duplicate games")
    if game_ids != sorted(game_ids):
        raise ValueError("fresh corpus is not chronological")
    if set(game_ids) & prior_game_ids:
        raise ValueError("fresh corpus overlaps a prior raw game")
    if game_ids[0] != 898_257_483 or game_ids[-1] != 898_258_344:
        raise ValueError("fresh corpus bounds changed")

    result = []
    for record in records:
        game_id = int(record["game_id"])
        if game_id <= FRESH_DEVELOPMENT_MAX_GAME:
            split = "development"
        elif game_id >= FRESH_TEST_MIN_GAME:
            split = "test"
        else:
            raise ValueError(f"fresh game {game_id} lies in the frozen cutoff gap")
        winner = (int(record["player_id"]) if record.get("won")
                  else 1 - int(record["player_id"]))
        result.append((split, record, winner))
    counts = collections.Counter(split for split, _, _ in result)
    if counts != {"development": 50, "test": 37}:
        raise ValueError(f"unexpected chronological split counts: {dict(counts)}")
    return result, {
        str(FRESH_RECORDS.relative_to(ROOT)): sha256_bytes(FRESH_RECORDS.read_bytes())
    }


def rank1_validation_states():
    data = RANK1_VALIDATION.read_bytes()
    if sha256_bytes(data) != RANK1_VALIDATION_SHA256:
        raise ValueError("rank-one validation reference hash mismatch")
    with io.StringIO(data.decode()) as source:
        reader = csv.DictReader(source, delimiter="\t")
        rows = list(reader)
    if len(rows) != 68:
        raise ValueError("rank-one validation reference has the wrong record count")
    result = []
    for row in rows:
        row = dict(row)
        row["split"] = "validation"
        row["winner_tier"] = "rank1"
        result.append(row)
    return result, {
        str(RANK1_VALIDATION.relative_to(ROOT)): sha256_bytes(data)
    }


def frozen_bank_states(path: pathlib.Path, expected_sha256: str, split: str):
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise ValueError(f"frozen bank reference hash mismatch: {path}")
    with io.StringIO(data.decode()) as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(HEADER.rstrip("\n").split("\t")):
            raise ValueError(f"frozen bank reference header mismatch: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"frozen bank reference is empty: {path}")
    for row in rows:
        row["split"] = split
    return rows, {str(path.relative_to(ROOT)): sha256_bytes(data)}


def elite_final_records(prior_game_ids: set[int]):
    if not ELITE_FINAL_RECORDS.exists():
        raise FileNotFoundError(
            f"missing {ELITE_FINAL_RECORDS}; run with --fetch-elite-final once"
        )
    payload = json.loads(ELITE_FINAL_RECORDS.read_text())
    if payload.get("schema") != "papersoccer.frozen-elite-final-holdout.v2":
        raise ValueError("elite final corpus has the wrong schema")
    expected_agents = [agent_id for agent_id, _ in ELITE_FINAL_AGENTS]
    if payload.get("agent_ids") != expected_agents:
        raise ValueError("elite final corpus has the wrong agents")
    records = payload.get("records", [])
    game_ids = [int(record["game_id"]) for record in records]
    if len(records) < 100 or len(set(game_ids)) != len(game_ids):
        raise ValueError("elite final corpus is too small or has duplicate games")
    if game_ids != sorted(game_ids):
        raise ValueError("elite final corpus is not chronological")
    if set(game_ids) & prior_game_ids:
        raise ValueError("elite final corpus overlaps prior evidence")
    counts = collections.Counter(int(record["focus_agent_id"]) for record in records)
    if any(counts[agent_id] < 8 for agent_id in expected_agents):
        raise ValueError(f"elite final corpus has insufficient agent coverage: {counts}")

    result = []
    for record in records:
        winner = (int(record["player_id"]) if record.get("won")
                  else 1 - int(record["player_id"]))
        result.append(("test", record, winner))
    return result, {
        str(ELITE_FINAL_RAW_RECORDS.relative_to(ROOT)): sha256_bytes(
            ELITE_FINAL_RAW_RECORDS.read_bytes()
        ),
        str(ELITE_FINAL_RECORDS.relative_to(ROOT)): sha256_bytes(
            ELITE_FINAL_RECORDS.read_bytes()
        ),
    }


def validation_extension_records(excluded_game_ids: set[int]):
    if not VALIDATION_EXTENSION_RECORDS.exists():
        raise FileNotFoundError(
            f"missing {VALIDATION_EXTENSION_RECORDS}; run with "
            "--fetch-validation-extension once"
        )
    raw_data = VALIDATION_EXTENSION_RAW_RECORDS.read_bytes()
    data = VALIDATION_EXTENSION_RECORDS.read_bytes()
    if sha256_bytes(raw_data) != VALIDATION_EXTENSION_RAW_SHA256:
        raise ValueError("validation extension raw snapshot hash mismatch")
    if sha256_bytes(data) != VALIDATION_EXTENSION_SHA256:
        raise ValueError("validation extension snapshot hash mismatch")
    payload = json.loads(data)
    if payload.get("schema") != "papersoccer.frozen-validation-extension.v2":
        raise ValueError("validation extension has the wrong schema")
    expected_agents = [agent_id for agent_id, _ in VALIDATION_EXTENSION_AGENTS]
    if payload.get("agent_ids") != expected_agents:
        raise ValueError("validation extension has the wrong agents")
    records = payload.get("records", [])
    game_ids = [int(record["game_id"]) for record in records]
    if (len(records) != EXPECTED_VALIDATION_EXTENSION_RECORDS or
            len(set(game_ids)) != len(game_ids)):
        raise ValueError("validation extension is too small or has duplicate games")
    if game_ids != sorted(game_ids):
        raise ValueError("validation extension is not chronological")
    if set(game_ids) & excluded_game_ids:
        raise ValueError("validation extension overlaps frozen evidence")
    counts = collections.Counter(int(record["focus_agent_id"]) for record in records)
    if any(
        counts[agent_id] < MIN_VALIDATION_EXTENSION_RECORDS_PER_AGENT
        for agent_id in expected_agents
    ):
        raise ValueError(
            f"validation extension has insufficient agent coverage: {counts}"
        )

    result = []
    for record in records:
        winner = (int(record["player_id"]) if record.get("won")
                  else 1 - int(record["player_id"]))
        result.append(("validation", record, winner))
    return result, {
        str(VALIDATION_EXTENSION_RAW_RECORDS.relative_to(ROOT)): sha256_bytes(
            raw_data
        ),
        str(VALIDATION_EXTENSION_RECORDS.relative_to(ROOT)): sha256_bytes(
            data
        ),
    }


def t6_validation_extension_records(excluded_game_ids: set[int]):
    if not T6_VALIDATION_EXTENSION_RECORDS.exists():
        raise FileNotFoundError(
            f"missing {T6_VALIDATION_EXTENSION_RECORDS}; run with "
            "--fetch-t6-validation-extension once"
        )
    raw_data = T6_VALIDATION_EXTENSION_RAW_RECORDS.read_bytes()
    data = T6_VALIDATION_EXTENSION_RECORDS.read_bytes()
    if sha256_bytes(raw_data) != T6_VALIDATION_EXTENSION_RAW_SHA256:
        raise ValueError("T6 validation extension raw snapshot hash mismatch")
    if sha256_bytes(data) != T6_VALIDATION_EXTENSION_SHA256:
        raise ValueError("T6 validation extension snapshot hash mismatch")
    raw_payload = json.loads(raw_data)
    payload = json.loads(data)
    expected_agents = [agent_id for agent_id, _ in T6_VALIDATION_EXTENSION_AGENTS]
    expected_names = [name for _, name in T6_VALIDATION_EXTENSION_AGENTS]
    if (raw_payload.get("schema") !=
            "papersoccer.frozen-t6-validation-extension.v1"
            or raw_payload.get("agent_ids") != expected_agents
            or raw_payload.get("agent_names") != expected_names
            or raw_payload.get("excluded_game_count") !=
            EXPECTED_T6_EXTENSION_EXCLUDED_GAMES):
        raise ValueError("T6 validation extension raw provenance mismatch")
    if (payload.get("schema") !=
            "papersoccer.frozen-t6-validation-extension.v2"
            or payload.get("agent_ids") != expected_agents
            or payload.get("agent_names") != expected_names
            or payload.get("raw_sha256") != T6_VALIDATION_EXTENSION_RAW_SHA256):
        raise ValueError("T6 validation extension provenance mismatch")
    records = payload.get("records", [])
    game_ids = [int(record["game_id"]) for record in records]
    if (len(records) != EXPECTED_T6_VALIDATION_EXTENSION_RECORDS or
            len(set(game_ids)) != len(game_ids)):
        raise ValueError("T6 validation extension has the wrong count or duplicates")
    if game_ids != sorted(game_ids):
        raise ValueError("T6 validation extension is not chronological")
    if set(game_ids) & excluded_game_ids:
        raise ValueError("T6 validation extension overlaps an earlier snapshot")
    counts = collections.Counter(int(record["focus_agent_id"])
                                 for record in records)
    if any(
        counts[agent_id] < MIN_T6_VALIDATION_EXTENSION_RECORDS_PER_AGENT
        for agent_id in expected_agents
    ):
        raise ValueError(
            f"T6 validation extension has insufficient agent coverage: {counts}"
        )

    result = []
    for record in records:
        winner = (int(record["player_id"]) if record.get("won")
                  else 1 - int(record["player_id"]))
        result.append(("validation", record, winner))
    return result, {
        str(T6_VALIDATION_EXTENSION_RAW_RECORDS.relative_to(ROOT)): sha256_bytes(
            raw_data
        ),
        str(T6_VALIDATION_EXTENSION_RECORDS.relative_to(ROOT)): sha256_bytes(data),
    }


def extract_states(split: str, record: dict, winner: int, *,
                   elite_balance: bool = False):
    turns = record["turns"]
    ball = (4, 6)
    edges = set()
    actions: list[str] = []
    states = []
    for turn_index, turn in enumerate(turns):
        mover = int(turn["player_id"])
        action = str(turn["action"])
        if mover == winner:
            state_key, canonical_key, normalized_ball, normalized_mover, normalized_edges = (
                state_identity(ball, mover, edges, winner)
            )
            if normalized_mover != 0:
                raise ValueError("winner normalization did not produce player zero")
            distance = 11 - normalized_ball[1]
            if 0 <= distance <= 2 and 1 <= normalized_ball[0] <= 7:
                source_agent = int(record["focus_agent_id"] if record.get("won")
                                   else record["opponent_agent_id"])
                opponent_agent = int(record["opponent_agent_id"] if record.get("won")
                                     else record["focus_agent_id"])
                opponent_name = str(record["focus_name"] if not record.get("won")
                                    else record["opponent_name"])
                winner_name = str(record["focus_name"] if record.get("won")
                                  else record["opponent_name"])
                phase = "early" if turn_index < 18 else (
                    "middle" if turn_index < 36 else "late"
                )
                cohort = tier(winner_name, source_agent)
                stratum = f"d{distance}"
                density = shell_band(normalized_edges)
                transcript = "/".join(actions)
                identity_seed = (
                    f"{split}|{record['game_id']}|{turn_index}|{canonical_key}"
                )
                states.append({
                    "opening_id": "shell-" + sha256_bytes(identity_seed.encode())[:16],
                    "split": split,
                    "stratum": stratum,
                    "source_agent_id": source_agent,
                    "source_game_id": int(record["game_id"]),
                    "opponent_agent_id": opponent_agent,
                    "winner_player_id": winner,
                    "turn_index": turn_index,
                    "physical_edges": len(edges),
                    "state_key": state_key,
                    "canonical_key": canonical_key,
                    "ball_x": ball[0],
                    "ball_y": ball[1],
                    "mover": mover,
                    "winner_tier": cohort,
                    "goal_distance_band": distance,
                    "used_edge_band": edge_band(len(edges)),
                    "shell_edge_band": density,
                    "opening_family": opening_family(actions),
                    "observed_winner_action": action,
                    "transcript": transcript,
                    "_selection_hash": sha256_bytes(identity_seed.encode()),
                    "_selection_stratum": (
                        f"{record['focus_agent_id']}-{cohort}-{stratum}-"
                        f"{phase}-{density}"
                        if elite_balance or split == "test"
                        else f"{cohort}-{stratum}-{phase}"
                    ),
                    "_focus_agent_id": int(record["focus_agent_id"]),
                    "_opponent_name": opponent_name,
                })
        ball = apply_action(ball, edges, action)
        actions.append(action)
    return states


def balanced_sample(states: Iterable[dict], limit: int, *, game_cap: int = 4,
                    game_distance_cap: int = 2,
                    focus_agent_cap: int | None = None,
                    minimum_elite_rows: int = 0):
    queues = collections.defaultdict(list)
    for state in states:
        queues[state["_selection_stratum"]].append(state)
    for values in queues.values():
        values.sort(key=lambda item: item["_selection_hash"])
    result = []
    per_game = collections.Counter()
    per_game_distance = collections.Counter()
    per_focus_agent = collections.Counter()

    def take_one(key: str) -> bool:
        while queues[key]:
            candidate = queues[key].pop(0)
            game = candidate["source_game_id"]
            game_distance = (game, candidate["goal_distance_band"])
            focus_agent = candidate["_focus_agent_id"]
            if (per_game[game] >= game_cap or
                    per_game_distance[game_distance] >= game_distance_cap or
                    (focus_agent_cap is not None and
                     per_focus_agent[focus_agent] >= focus_agent_cap)):
                continue
            result.append(candidate)
            per_game[game] += 1
            per_game_distance[game_distance] += 1
            per_focus_agent[focus_agent] += 1
            return True
        return False

    elite_tiers = {"rank1", "elite"}
    elite_keys = [
        key for key, values in queues.items()
        if values and values[0]["winner_tier"] in elite_tiers
    ]
    elite_keys.sort(key=lambda key: (
        0 if queues[key][0]["winner_tier"] == "rank1" else 1,
        key,
    ))
    elite_rows = 0
    while elite_rows < minimum_elite_rows and len(result) < limit:
        progressed = False
        for key in elite_keys:
            if take_one(key):
                elite_rows += 1
                progressed = True
            if elite_rows == minimum_elite_rows or len(result) == limit:
                break
        if not progressed:
            break

    while len(result) < limit:
        progressed = False
        for key in sorted(queues):
            if take_one(key):
                progressed = True
            if len(result) == limit:
                break
        if not progressed:
            break
    return sorted(result, key=lambda item: item["opening_id"])


def tsv_bytes(states: list[dict]) -> bytes:
    columns = HEADER.rstrip("\n").split("\t")
    rows = [HEADER]
    for state in states:
        values = []
        for column in columns:
            value = state[column]
            text = str(value)
            if "\t" in text or "\n" in text:
                raise ValueError(f"invalid TSV value in {column}")
            values.append(text)
        rows.append("\t".join(values) + "\n")
    return "".join(rows).encode()


def _build_t6_artifacts():
    """Retain the historical T6 recipe; the active CLI never calls it."""
    prior_records, prior_game_ids, prior_sources = prior_raw_sources()
    fresh, fresh_sources = fresh_records(prior_game_ids)
    fresh_game_ids = {int(record["game_id"]) for _, record, _ in fresh}
    elite, elite_sources = elite_final_records(prior_game_ids | fresh_game_ids)
    if len(elite) != EXPECTED_ELITE_FINAL_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_ELITE_FINAL_RECORDS} elite-final records, "
            f"found {len(elite)}"
        )
    elite_raw_payload = json.loads(ELITE_FINAL_RAW_RECORDS.read_text())
    elite_raw_game_ids = {
        int(record["game_id"]) for record in elite_raw_payload["records"]
    }
    elite_game_ids = {int(record["game_id"]) for _, record, _ in elite}
    extension, extension_sources = validation_extension_records(
        prior_game_ids | fresh_game_ids | elite_raw_game_ids | elite_game_ids
    )
    extension_raw_payload = json.loads(
        VALIDATION_EXTENSION_RAW_RECORDS.read_text()
    )
    extension_raw_game_ids = {
        int(record["game_id"]) for record in extension_raw_payload["records"]
    }
    extension_game_ids = {
        int(record["game_id"]) for _, record, _ in extension
    }
    t6_extension, t6_extension_sources = t6_validation_extension_records(
        prior_game_ids | fresh_game_ids | elite_raw_game_ids | elite_game_ids |
        extension_raw_game_ids | extension_game_ids
    )
    rank1, rank1_sources = rank1_validation_states()
    development, development_sources = frozen_bank_states(
        T3_DEVELOPMENT, T3_DEVELOPMENT_SHA256, "development"
    )
    _, t3_validation_sources = frozen_bank_states(
        T3_VALIDATION_SOURCE, T3_VALIDATION_SOURCE_SHA256, "validation"
    )
    retired_validation, retired_validation_sources = frozen_bank_states(
        T4_VALIDATION, T4_VALIDATION_SHA256, "validation"
    )
    exposed_validation, exposed_validation_sources = frozen_bank_states(
        T5_EXPOSED_VALIDATION, T5_EXPOSED_VALIDATION_SHA256, "validation"
    )
    replacement_exposed_validation, replacement_exposed_validation_sources = (
        frozen_bank_states(
            T5_REPLACEMENT_EXPOSED_VALIDATION,
            T5_REPLACEMENT_EXPOSED_VALIDATION_SHA256,
            "validation",
        )
    )
    test, test_sources = frozen_bank_states(
        T4_FINAL_TEST, T4_FINAL_TEST_SHA256, "test"
    )
    frozen_test_bytes = T4_FINAL_TEST.read_bytes()
    if tsv_bytes(test) != frozen_test_bytes:
        raise ValueError("frozen final test does not round-trip byte-for-byte")
    if len(test) != 72:
        raise ValueError(f"expected 72 frozen final states, found {len(test)}")

    legacy_keys = set()
    for record, winner in prior_records:
        legacy_keys.update(
            state["canonical_key"]
            for state in extract_states("legacy", record, winner)
        )
    for _, record, winner in fresh:
        legacy_keys.update(
            state["canonical_key"]
            for state in extract_states("legacy", record, winner)
        )
    legacy_keys.update(state["canonical_key"] for state in rank1)

    evidence = (
        development + retired_validation + exposed_validation +
        replacement_exposed_validation + rank1
    )
    evidence_keys = legacy_keys | {
        state["canonical_key"] for state in evidence
    }
    evidence_game_ids = prior_game_ids | fresh_game_ids | {
        int(state["source_game_id"])
        for state in evidence
        if int(state["source_game_id"]) != 0
    }
    test_keys = {state["canonical_key"] for state in test}
    test_game_ids = {int(state["source_game_id"]) for state in test}
    exposed_validation_keys = {
        state["canonical_key"] for state in exposed_validation
    }
    exposed_validation_game_ids = {
        int(state["source_game_id"]) for state in exposed_validation
    }
    replacement_exposed_validation_keys = {
        state["canonical_key"] for state in replacement_exposed_validation
    }
    replacement_exposed_validation_game_ids = {
        int(state["source_game_id"])
        for state in replacement_exposed_validation
    }
    if len(exposed_validation_keys) != len(exposed_validation):
        raise ValueError("exposed validation repeats a canonical state")
    if (len(exposed_validation_game_ids) !=
            EXPECTED_EXPOSED_VALIDATION_SOURCE_GAMES):
        raise ValueError(
            f"expected {EXPECTED_EXPOSED_VALIDATION_SOURCE_GAMES} exposed "
            f"validation games, found {len(exposed_validation_game_ids)}"
        )
    if (len(replacement_exposed_validation_keys) !=
            len(replacement_exposed_validation)):
        raise ValueError("replacement exposed validation repeats a canonical state")
    if (len(replacement_exposed_validation_game_ids) !=
            EXPECTED_T5_REPLACEMENT_EXPOSED_SOURCE_GAMES):
        raise ValueError(
            f"expected {EXPECTED_T5_REPLACEMENT_EXPOSED_SOURCE_GAMES} replacement "
            "exposed validation games, found "
            f"{len(replacement_exposed_validation_game_ids)}"
        )
    if (replacement_exposed_validation_game_ids & exposed_validation_game_ids or
            replacement_exposed_validation_keys & exposed_validation_keys):
        raise ValueError("the two exposed validations overlap")
    if len(test_keys) != len(test):
        raise ValueError("frozen final test repeats a canonical state")
    if len(test_game_ids) != EXPECTED_FINAL_TEST_SOURCE_GAMES:
        raise ValueError(
            f"expected {EXPECTED_FINAL_TEST_SOURCE_GAMES} frozen final games, "
            f"found {len(test_game_ids)}"
        )
    if test_game_ids & evidence_game_ids:
        raise ValueError("frozen final test shares a source game with prior evidence")
    if test_keys & evidence_keys:
        raise ValueError("frozen final test shares a canonical state with prior evidence")

    unavailable_game_ids = evidence_game_ids | test_game_ids
    remaining_elite = [
        (record, winner)
        for _, record, winner in elite
        if int(record["game_id"]) not in unavailable_game_ids
    ]
    if len(remaining_elite) != EXPECTED_T6_REMAINING_ELITE_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_T6_REMAINING_ELITE_RECORDS} remaining elite "
            f"records, found {len(remaining_elite)}"
        )
    extension_records = [
        (record, winner)
        for _, record, winner in extension
        if int(record["game_id"]) not in unavailable_game_ids
    ]
    if (len(extension_records) !=
            EXPECTED_T6_REMAINING_VALIDATION_EXTENSION_RECORDS):
        raise ValueError(
            "validation extension has the wrong remaining record count"
        )
    t6_extension_records = [
        (record, winner)
        for _, record, winner in t6_extension
        if int(record["game_id"]) not in unavailable_game_ids
    ]
    if len(t6_extension_records) != EXPECTED_T6_VALIDATION_EXTENSION_RECORDS:
        raise ValueError("T6 validation extension overlaps unavailable evidence")
    candidate_records = (
        remaining_elite + extension_records + t6_extension_records
    )
    if len(candidate_records) != EXPECTED_T6_CANDIDATE_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_T6_CANDIDATE_RECORDS} T6 "
            f"candidate records, found {len(candidate_records)}"
        )
    candidates = []
    for record, winner in candidate_records:
        candidates.extend(
            extract_states("validation", record, winner, elite_balance=True)
        )
    if len(candidates) != EXPECTED_T6_RAW_SHELL_STATES:
        raise ValueError(
            f"expected {EXPECTED_T6_RAW_SHELL_STATES} T6 "
            "shell states, "
            f"found {len(candidates)}"
        )
    seen = evidence_keys | test_keys
    unique = []
    for state in sorted(candidates, key=lambda item: item["_selection_hash"]):
        if state["canonical_key"] in seen:
            continue
        seen.add(state["canonical_key"])
        unique.append(state)
    if len(unique) != EXPECTED_T6_UNIQUE_STATES:
        raise ValueError(
            f"expected {EXPECTED_T6_UNIQUE_STATES} unique T6 validation "
            f"states, found {len(unique)}"
        )
    validation = balanced_sample(
        unique, BANK_LIMITS["validation"], game_cap=2, game_distance_cap=1,
        focus_agent_cap=VALIDATION_FOCUS_AGENT_CAP,
        minimum_elite_rows=VALIDATION_MIN_ELITE_TIER_ROWS,
    )
    if len(validation) != BANK_LIMITS["validation"]:
        raise ValueError(
            f"only selected {len(validation)} of "
            f"{BANK_LIMITS['validation']} validation states"
        )

    validation_keys = {state["canonical_key"] for state in validation}
    validation_game_ids = {
        int(state["source_game_id"]) for state in validation
    }
    if len(validation_game_ids) != EXPECTED_T6_VALIDATION_SOURCE_GAMES:
        raise ValueError(
            f"expected {EXPECTED_T6_VALIDATION_SOURCE_GAMES} T6 "
            f"validation games, found {len(validation_game_ids)}"
        )
    per_game = collections.Counter(
        int(state["source_game_id"]) for state in validation
    )
    per_game_distance = collections.Counter(
        (int(state["source_game_id"]), int(state["goal_distance_band"]))
        for state in validation
    )
    per_focus_agent = collections.Counter(
        int(state["_focus_agent_id"]) for state in validation
    )
    elite_tier_rows = sum(
        state["winner_tier"] in {"rank1", "elite"} for state in validation
    )
    if len(validation_keys) != len(validation):
        raise ValueError("prospective validation repeats a canonical state")
    if validation_game_ids & unavailable_game_ids:
        raise ValueError("prospective validation shares an unavailable source game")
    if validation_keys & (evidence_keys | test_keys):
        raise ValueError("prospective validation shares an excluded canonical state")
    if max(per_game.values()) > 2:
        raise ValueError("prospective validation exceeds its per-game cap")
    if max(per_game_distance.values()) > 1:
        raise ValueError("prospective validation exceeds its per-game-distance cap")
    if max(per_focus_agent.values()) > VALIDATION_FOCUS_AGENT_CAP:
        raise ValueError("prospective validation exceeds its focus-agent cap")
    if elite_tier_rows < VALIDATION_MIN_ELITE_TIER_ROWS:
        raise ValueError("prospective validation lacks its elite-tier minimum")

    banks = {
        "openings/development.tsv": tsv_bytes(development),
        "openings/validation.tsv": tsv_bytes(validation),
        "openings/test.tsv": frozen_test_bytes,
    }

    initial_state_key, initial_canonical_key, _, _, _ = state_identity(
        (4, 6), 0, set(), 0
    )
    initial = {
        "opening_id": "initial",
        "split": "initial",
        "stratum": "initial",
        "source_agent_id": 0,
        "source_game_id": 0,
        "opponent_agent_id": 0,
        "winner_player_id": 0,
        "turn_index": 0,
        "physical_edges": 0,
        "state_key": initial_state_key,
        "canonical_key": initial_canonical_key,
        "ball_x": 4,
        "ball_y": 6,
        "mover": 0,
        "winner_tier": "initial",
        "goal_distance_band": -1,
        "used_edge_band": "empty",
        "shell_edge_band": "empty",
        "opening_family": "initial",
        "observed_winner_action": "-",
        "transcript": "-",
    }
    banks["openings/initial.tsv"] = tsv_bytes([initial])

    source_hashes = dict(sorted((
        prior_sources | fresh_sources | elite_sources | rank1_sources |
        extension_sources | t6_extension_sources |
        development_sources | t3_validation_sources |
        retired_validation_sources | exposed_validation_sources |
        replacement_exposed_validation_sources | test_sources
    ).items()))
    harness_sources = (
        HERE / "build_goal_shell_banks.py",
        HERE.parent / "tools" / "promotion_gate.py",
        HERE.parent / "bots" / "reply_proof" / "comparison_gate.cpp",
        HERE.parent / "bots" / "reply_proof" / "submission_test.cpp",
        HERE.parent / "bots" / "reply_proof" / "timing_probe.cpp",
    )
    for path in harness_sources:
        source_hashes[str(path.relative_to(ROOT))] = sha256_bytes(path.read_bytes())
    source_hashes = dict(sorted(source_hashes.items()))
    focus_agent_names = dict(
        ELITE_FINAL_AGENTS + VALIDATION_EXTENSION_AGENTS +
        T6_VALIDATION_EXTENSION_AGENTS
    )
    manifest = {
        "schema": "papersoccer.codingame-promotion-manifest.v1",
        "candidate": "reply_proof",
        "candidate_submission_sha256": (
            "de2cdd18ae37b93bb0a443cdcfe13e52f6992aa0ac1c7fb43a544a874cb59789"
        ),
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "own_goals_allowed",
            "blocked_rule": "mover_loses",
            "positions_are_complete_turn_boundaries": True,
        },
        "incumbent": {
            "name": "rank_5",
            "submission_sha256": (
                "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29"
            ),
        },
        "hypothesis": (
            "retain unconditional root and depth-zero rebound-component proofs, "
            "then add a TT-first exact Win/Loss proof only at the first opponent "
            "complete-turn boundary below each root action (turn_ply == 1); "
            "Unknown positions and all heuristics remain unchanged"
        ),
        "selection": {
            "normalization": "winner rotated to player zero; horizontal reflection dedup",
            "goal_shell": "winner to move at normalized y 9, 10, or 11",
            "game_cap": (
                "prospective validation at most two states per game, one per "
                "game-distance band, and 32 per frozen focus agent"
            ),
            "winner_tier_balance": (
                "prospective T6 selection reserves at least 12 rows for rank-one "
                "or elite winners before deterministic all-tier filling, using the "
                "same global game, game-distance, and focus-agent counters"
            ),
            "winner_tiers": ["rank1", "elite", "incumbent", "field"],
            "legacy_evidence_exclusion": (
                "all source games and shell-state canonical keys from eight prior "
                "Arena batches, the rank-one corpus, all 87 fresh-agent games, "
                "the frozen development bank, the retired T4 validation, and the "
                "two exposed T5 validations"
            ),
            "development": (
                "unchanged frozen T3 development bank; adaptive screening evidence"
            ),
            "validation": (
                "prospective 72-state bank drawn deterministically from the "
                "remaining elite-final-v2 games and both append-only validation "
                "extensions, excluding every prior-evidence, exposed-validation, "
                "and frozen-final source game and canonical key; the T6 source "
                "identities were predeclared only from frozen opponent metadata"
            ),
            "retired_validation": (
                f"T4 validation frozen at {T4_VALIDATION_SHA256}; exclusion-only"
            ),
            "exposed_validation": (
                f"T5 validation frozen at {T5_EXPOSED_VALIDATION_SHA256} after "
                "background-job exposure; exclusion-only"
            ),
            "replacement_exposed_validation": (
                "T5 replacement validation frozen at "
                f"{T5_REPLACEMENT_EXPOSED_VALIDATION_SHA256} after official "
                "exposure and rejection; exclusion-only"
            ),
            "test": (
                f"unchanged sealed 72-state T4 final test frozen at "
                f"{T4_FINAL_TEST_SHA256}"
            ),
            "bank_snapshot": (
                "the controller verifies each manifest bank hash, copies the "
                "verified bytes into the immutable result identity, and runs each "
                "T6 shard against that hash-pinned archived path; in-flight shards "
                "never reopen mutable bank paths"
            ),
            "proof_diagnostics": (
                "rebound_goal_probes counts exact component analyses; "
                "rebound_goal_hits counts proven reachable attacking goals; "
                "rebound_loss_hits counts components with no safe handoff under "
                "mover-loses"
            ),
            "proof_placement": (
                "reply_proof keeps the rebound_proof root and depth-zero checks; "
                "at positive depth it probes the transposition table first and "
                "runs the exact component analyzer only when turn_ply equals one, "
                "before enumerating the opponent's first reply"
            ),
            "control_normalization": (
                "one rank_5-vs-rank_5 control per opening and node budget; "
                "the hard uplift gate is the minimum candidate uplift over that "
                "baseline across the two physical colors; historical winner and "
                "opponent repartitions remain diagnostics with an absolute score floor"
            ),
            "protocol_amendment": {
                "effective_stage": "prospective T6 validation and sealed final only",
                "timing": "declared before T6 candidate selection or outcomes",
                "retired_result": (
                    "the exposed T5 rejection is immutable and is not re-evaluated"
                ),
                "replaced_requirement": "minimum_control_adjusted_uplift",
                "replacement_requirements": {
                    "minimum_physical_color_uplift": {
                        "validation": -0.05,
                        "test": -0.05,
                    },
                    "minimum_historical_role_score": {
                        "validation": 0.45,
                        "test": 0.45,
                    },
                },
                "rationale": (
                    "winner/opponent uplifts are correlated repartitions of the same "
                    "net changes, whereas physical colors are deployment roles; 72 "
                    "openings cannot resolve a negative two-point historical-label "
                    "margin, so historical uplifts remain diagnostics"
                ),
            },
            "elite_final_agents": [agent_id for agent_id, _ in ELITE_FINAL_AGENTS],
            "elite_final_parseable_records": len(elite),
            "elite_final_remaining_records": len(remaining_elite),
            "validation_extension_agents": [
                agent_id for agent_id, _ in VALIDATION_EXTENSION_AGENTS
            ],
            "validation_extension_frozen_records": len(extension),
            "validation_extension_remaining_records": len(extension_records),
            "t6_validation_extension_agents": [
                agent_id for agent_id, _ in T6_VALIDATION_EXTENSION_AGENTS
            ],
            "t6_validation_extension_frozen_records": len(t6_extension),
            "t6_validation_extension_available_records": len(t6_extension_records),
            "validation_candidate_records": len(candidate_records),
            "validation_raw_shell_states": len(candidates),
            "validation_unique_states_after_exclusion": len(unique),
            "validation_source_games": len(validation_game_ids),
            "validation_focus_agent_cap": VALIDATION_FOCUS_AGENT_CAP,
            "validation_minimum_elite_tier_rows": (
                VALIDATION_MIN_ELITE_TIER_ROWS
            ),
            "validation_elite_tier_rows": elite_tier_rows,
            "validation_focus_agent_rows": [
                {
                    "agent_id": agent_id,
                    "name": focus_agent_names[agent_id],
                    "rows": per_focus_agent[agent_id],
                }
                for agent_id in sorted(per_focus_agent)
            ],
            "final_test_source_games": len(test_game_ids),
            "prior_raw_game_count": len(prior_game_ids),
            "fresh_game_count": EXPECTED_FRESH_RECORDS,
        },
        "sources": source_hashes,
        "banks": {
            path: {
                "sha256": sha256_bytes(content),
                "records": content.count(b"\n") - 1,
            }
            for path, content in sorted(banks.items())
        },
        "stages": {
            "initial": {
                "bank": "openings/initial.tsv",
                "node_budget": 5000,
                "maximum_turns": 320,
                "minimum_mean": 0.50,
            },
            "development": {
                "bank": "openings/development.tsv",
                "node_budgets": [5000, 30000],
                "maximum_turns": 320,
                "minimum_mean": 0.52,
                "minimum_throughput_ratio": 0.90,
                "require_more_wins_than_incumbent": True,
                "node_budget_overrides": {
                    "5000": {
                        "minimum_mean": 0.50,
                        "require_more_wins_than_incumbent": False,
                        "require_at_least_as_many_wins_as_incumbent": True,
                    },
                },
            },
            "validation": {
                "bank": "openings/validation.tsv",
                "node_budget": 5000,
                "maximum_turns": 320,
                "minimum_mean": 0.51,
                "minimum_ci_lower": 0.47,
                "minimum_physical_color_uplift": -0.05,
                "minimum_historical_role_score": 0.45,
                "minimum_control_winner_retention": 0.80,
                "minimum_stratum_score": 0.48,
                "minimum_winner_tier_score": 0.48,
                "minimum_elite_tier_score": 0.48,
                "minimum_throughput_ratio": 0.90,
            },
            "test": {
                "bank": "openings/test.tsv",
                "node_budgets": [30000, 100000],
                "maximum_turns": 320,
                "minimum_mean": 0.50,
                "minimum_ci_lower": 0.45,
                "minimum_physical_color_uplift": -0.05,
                "minimum_historical_role_score": 0.45,
                "minimum_control_winner_retention": 0.75,
                "minimum_stratum_score": 0.45,
                "minimum_winner_tier_score": 0.40,
                "minimum_elite_tier_score": 0.45,
                "minimum_throughput_ratio": 0.90,
            },
        },
        "statistics": {
            "unit": "source_game_cluster_of_color_swapped_opening_pairs",
            "method": "source_game_cluster_percentile_bootstrap",
            "confidence": 0.95,
            "resamples": 10000,
            "seed": 4_774_557_432_748_095_049,
        },
        "timing": {
            "fresh_process_samples": 20,
            "shell_cases": ["elite-d2", "rank1-d1", "elite-d0-dense"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        },
        "source_limit": 100000,
    }
    return banks, stable_json(manifest)


def _verified_frozen_bank(path: pathlib.Path, expected_sha256: str,
                          expected_records: int, expected_split: str) -> bytes:
    data = path.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen bank hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    with io.StringIO(data.decode()) as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(HEADER.rstrip("\n").split("\t")):
            raise ValueError(f"frozen bank reference header mismatch: {path}")
        rows = list(reader)
    if len(rows) != expected_records:
        raise ValueError(
            f"frozen bank record count mismatch for {path}: expected "
            f"{expected_records}, found {len(rows)}"
        )
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(
            f"frozen bank split mismatch for {path}: expected {expected_split}"
        )
    return data


def _t7_evidence_contract() -> tuple[dict, bytes]:
    data = T7_EVIDENCE_MANIFEST.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != T7_EVIDENCE_MANIFEST_SHA256:
        raise ValueError(
            "T7 evidence manifest hash mismatch: expected "
            f"{T7_EVIDENCE_MANIFEST_SHA256}, found {actual_sha256}"
        )
    try:
        evidence = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError("T7 evidence manifest is not valid JSON") from error
    if (evidence.get("schema") !=
            "papersoccer.candidate-independent-t7-evidence.v1"
            or evidence.get("status") != "frozen_before_candidate_binding"
            or evidence.get("candidate") is not None
            or evidence.get("candidate_submission_sha256") is not None):
        raise ValueError("T7 evidence manifest binding status changed")

    protocol = evidence.get("prospective_strength_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("T7 evidence manifest lacks its strength protocol")
    protocol_sha256 = sha256_bytes(stable_json(protocol))
    if (protocol_sha256 != T7_STRENGTH_PROTOCOL_SHA256 or
            evidence.get("prospective_strength_protocol_sha256") !=
            T7_STRENGTH_PROTOCOL_SHA256):
        raise ValueError("T7 prospective strength protocol hash changed")

    expected_banks = {
        "reference/t7_prospective_validation.tsv": {
            "role": "prospective_validation",
            "sealed": False,
            "sha256": T7_PROSPECTIVE_VALIDATION_SHA256,
            "records": 72,
        },
        "reference/t7_sealed_final.tsv": {
            "role": "sealed_final",
            "sealed": True,
            "sha256": T7_SEALED_FINAL_SHA256,
            "records": 72,
        },
    }
    for relative, expected in expected_banks.items():
        specification = evidence.get("banks", {}).get(relative, {})
        if any(specification.get(key) != value for key, value in expected.items()):
            raise ValueError(f"T7 evidence-bank declaration changed: {relative}")

    expected_prior_decisions = {
        "t6_validation": {
            "status": "exposed_rejected",
            "sha256": T6_EXPOSED_VALIDATION_SHA256,
        },
        "t6_final": {
            "status": "exposed",
            "sha256": T6_EXPOSED_FINAL_SHA256,
        },
    }
    if evidence.get("prior_decisions") != expected_prior_decisions:
        raise ValueError("T6 decision provenance changed in the T7 evidence manifest")

    expected_sources = {
        "submissions/codingame/promotion/acquire_t7_evidence.py": (
            T7_ACQUISITION_SHA256
        ),
        "submissions/codingame/promotion/freeze_t7_banks.py": T7_FREEZER_SHA256,
        "submissions/codingame/promotion/t7_evidence_ladder_v1.json": (
            T7_RAW_SNAPSHOT_SHA256
        ),
        "submissions/codingame/promotion/t7_evidence_ladder_v2.json": (
            T7_PARSEABLE_SNAPSHOT_SHA256
        ),
    }
    sources = evidence.get("sources", {})
    if any(
        sources.get(path) != expected
        for path, expected in expected_sources.items()
    ):
        raise ValueError("T7 acquisition provenance changed")
    return evidence, data


def _t8_evidence_contract() -> tuple[dict, bytes]:
    """Verify the immutable, candidate-independent side of the T8 transition."""
    data = T8_EVIDENCE_MANIFEST.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != T8_EVIDENCE_MANIFEST_SHA256:
        raise ValueError(
            "T8 evidence manifest hash mismatch: expected "
            f"{T8_EVIDENCE_MANIFEST_SHA256}, found {actual_sha256}"
        )
    try:
        evidence = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError("T8 evidence manifest is not valid JSON") from error
    if (
        evidence.get("schema")
        != "papersoccer.candidate-independent-t8-evidence.v1"
        or evidence.get("status") != "frozen_before_candidate_binding"
        or evidence.get("candidate") is not None
        or evidence.get("candidate_submission_sha256") is not None
    ):
        raise ValueError("T8 evidence manifest binding status changed")

    protocol = evidence.get("prospective_strength_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("T8 evidence manifest lacks its strength protocol")
    protocol_sha256 = sha256_bytes(stable_json(protocol))
    if (
        protocol_sha256 != T8_STRENGTH_PROTOCOL_SHA256
        or evidence.get("prospective_strength_protocol_sha256")
        != T8_STRENGTH_PROTOCOL_SHA256
    ):
        raise ValueError("T8 prospective strength protocol hash changed")

    expected_banks = {
        "reference/t8_prospective_validation.tsv": {
            "role": "prospective_validation",
            "sealed": False,
            "sha256": T8_PROSPECTIVE_VALIDATION_SHA256,
            "records": 72,
        },
        "reference/t8_sealed_final.tsv": {
            "role": "sealed_final",
            "sealed": True,
            "sha256": T8_SEALED_FINAL_SHA256,
            "records": 72,
        },
    }
    for relative, expected in expected_banks.items():
        specification = evidence.get("banks", {}).get(relative, {})
        if any(specification.get(key) != value for key, value in expected.items()):
            raise ValueError(f"T8 evidence-bank declaration changed: {relative}")

    expected_prior_decisions = {
        "t7_validation": {
            "status": "exposed",
            "sha256": T7_PROSPECTIVE_VALIDATION_SHA256,
        },
        "t7_final": {
            "status": "exposed",
            "sha256": T7_SEALED_FINAL_SHA256,
        },
    }
    if evidence.get("prior_decisions") != expected_prior_decisions:
        raise ValueError("T7 decision provenance changed in the T8 evidence manifest")

    expected_sources = {
        "submissions/codingame/promotion/acquire_t8_evidence.py": (
            T8_ACQUISITION_SHA256
        ),
        "submissions/codingame/promotion/freeze_t8_banks.py": T8_FREEZER_SHA256,
        "submissions/codingame/promotion/t8_evidence_ladder_v1.json": (
            T8_RAW_SNAPSHOT_SHA256
        ),
        "submissions/codingame/promotion/t8_evidence_ladder_v2.json": (
            T8_PARSEABLE_SNAPSHOT_SHA256
        ),
        "submissions/codingame/promotion/build_goal_shell_banks.py": (
            T8_PREBIND_BUILDER_SHA256
        ),
    }
    sources = evidence.get("sources", {})
    if any(sources.get(path) != expected for path, expected in expected_sources.items()):
        raise ValueError("T8 pre-binding acquisition provenance changed")

    stages = protocol.get("stages", {})
    initial = stages.get("initial", {})
    development = stages.get("development", {})
    if (
        initial.get("evidence_status") != "exposed_adaptive_not_prospective"
        or initial.get("configuration")
        != {"maximum_turns": 320, "minimum_mean": 0.5, "node_budget": 5000}
    ):
        raise ValueError("T8 initial-stage protocol changed")
    expected_development = {
        "maximum_turns": 320,
        "minimum_mean": 0.52,
        "minimum_throughput_ratio": 0.9,
        "node_budgets": [5000, 30000],
        "node_budget_overrides": {
            "5000": {
                "minimum_mean": 0.5,
                "require_at_least_as_many_wins_as_incumbent": True,
                "require_more_wins_than_incumbent": False,
            }
        },
        "require_more_wins_than_incumbent": True,
    }
    if (
        development.get("evidence_status")
        != "exposed_adaptive_not_prospective"
        or development.get("configuration") != expected_development
    ):
        raise ValueError("T8 development-stage protocol changed")

    expected_profiles = {
        "validation": [
            ("30k-nodes", "nodes", 30000, None),
            ("130ms", "time_ms", 130, 3_000_000),
        ],
        "test": [
            ("100k-nodes", "nodes", 100000, None),
            ("130ms", "time_ms", 130, 3_000_000),
        ],
    }
    for stage, expected in expected_profiles.items():
        declaration = stages.get(stage, {})
        configuration = declaration.get("configuration", {})
        if (
            declaration.get("shard_count") != 4
            or declaration.get("jobs") != 4
            or configuration.get("required_jobs") != 4
        ):
            raise ValueError(f"T8 {stage} four-worker execution contract changed")
        actual = [
            (
                profile.get("id"),
                profile.get("mode"),
                profile.get("value"),
                profile.get("max_nodes"),
            )
            for profile in configuration.get("strength_profiles", [])
        ]
        if actual != expected:
            raise ValueError(f"T8 {stage} strength profiles changed")
    return evidence, data


def _t9_evidence_contract() -> tuple[dict, bytes]:
    """Verify the inactive T9 carry-forward before any candidate binding."""
    t8_evidence, _ = _t8_evidence_contract()
    data = T9_EVIDENCE_MANIFEST.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != T9_EVIDENCE_MANIFEST_SHA256:
        raise ValueError(
            "T9 evidence manifest hash mismatch: expected "
            f"{T9_EVIDENCE_MANIFEST_SHA256}, found {actual_sha256}"
        )
    try:
        evidence = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError("T9 evidence manifest is not valid JSON") from error
    if (
        evidence.get("schema")
        != "papersoccer.candidate-independent-t9-evidence.v1"
        or evidence.get("status") != "frozen_before_candidate_binding"
        or evidence.get("candidate") is not None
        or evidence.get("candidate_submission_sha256") is not None
    ):
        raise ValueError("T9 evidence manifest binding status changed")

    protocol = evidence.get("prospective_strength_protocol")
    if (
        not isinstance(protocol, dict)
        or protocol != t8_evidence.get("prospective_strength_protocol")
        or sha256_bytes(stable_json(protocol)) != T9_STRENGTH_PROTOCOL_SHA256
        or evidence.get("prospective_strength_protocol_sha256")
        != T9_STRENGTH_PROTOCOL_SHA256
    ):
        raise ValueError("T9 inherited strength protocol changed")
    carry_forward = evidence.get("protocol_carry_forward", {})
    if carry_forward != {
        "semantic_changes": [],
        "source_evidence_manifest_sha256": T8_EVIDENCE_MANIFEST_SHA256,
        "source_ladder": "T8",
        "source_protocol_sha256": T8_STRENGTH_PROTOCOL_SHA256,
        "stage_bank_aliases": {
            "reference/t8_prospective_validation.tsv": (
                "reference/t9_prospective_validation.tsv"
            ),
            "reference/t8_sealed_final.tsv": "reference/t9_sealed_final.tsv",
        },
        "version_and_provenance_changes_only": True,
    }:
        raise ValueError("T9 protocol carry-forward declaration changed")

    expected_banks = {
        "reference/t9_prospective_validation.tsv": {
            "source": "reference/t8_prospective_validation.tsv",
            "role": "prospective_validation",
            "sealed": False,
            "sha256": T9_PROSPECTIVE_VALIDATION_SHA256,
            "records": 72,
        },
        "reference/t9_sealed_final.tsv": {
            "source": "reference/t8_sealed_final.tsv",
            "role": "sealed_final",
            "sealed": True,
            "sha256": T9_SEALED_FINAL_SHA256,
            "records": 72,
        },
    }
    for relative, expected in expected_banks.items():
        specification = evidence.get("banks", {}).get(relative, {})
        for key in ("role", "sealed", "sha256", "records"):
            if specification.get(key) != expected[key]:
                raise ValueError(f"T9 evidence-bank declaration changed: {relative}")
        if specification.get("carried_forward_from") != {
            "evidence_manifest_sha256": T8_EVIDENCE_MANIFEST_SHA256,
            "reference": expected["source"],
            "sha256": expected["sha256"],
        }:
            raise ValueError(f"T9 bank carry-forward changed: {relative}")

    validation_data = _verified_frozen_bank(
        T9_PROSPECTIVE_VALIDATION,
        T9_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    final_data = _verified_frozen_bank(
        T9_SEALED_FINAL, T9_SEALED_FINAL_SHA256, 72, "test"
    )
    if (
        validation_data != T8_PROSPECTIVE_VALIDATION.read_bytes()
        or final_data != T8_SEALED_FINAL.read_bytes()
    ):
        raise ValueError("T9 aliases are not byte-identical to T8 banks")

    audit = evidence.get("nonconsumption_audit", {})
    retired = audit.get("retired_binding", {})
    expected_stage_status = {
        "development": "reject",
        "initial": "pass",
        "test": "not_run_due_to_rejection",
        "validation": "not_run_due_to_rejection",
    }
    if (
        retired.get("candidate") != "frontier_proof"
        or retired.get("candidate_submission_sha256")
        != T8_FRONTIER_SUBMISSION_SHA256
        or retired.get("bound_manifest_sha256")
        != T8_FRONTIER_BOUND_MANIFEST_SHA256
        or retired.get("decision_sha256") != T8_FRONTIER_DECISION_SHA256
        or retired.get("development_bank_sha256") != T3_DEVELOPMENT_SHA256
        or retired.get("development_report_sha256")
        != T8_FRONTIER_DEVELOPMENT_REPORT_SHA256
        or retired.get("failed_stage") != "development"
        or retired.get("retirement_reason")
        != "exposed_adaptive_development_failure_only"
        or retired.get("stage_status") != expected_stage_status
    ):
        raise ValueError("T8 retirement provenance changed in T9 evidence")
    expected_absence = {
        "final_ledger_marker_exists": False,
        "test_immutable_bank_snapshot_exists": False,
        "test_report_exists": False,
        "test_shard_directory_exists": False,
        "validation_immutable_bank_snapshot_exists": False,
        "validation_report_exists": False,
        "validation_shard_directory_exists": False,
    }
    if audit.get("prospective_artifacts") != expected_absence:
        raise ValueError("T8 nonconsumption audit changed in T9 evidence")

    sources = evidence.get("sources", {})
    expected_sources = {
        "submissions/codingame/promotion/freeze_t9_banks.py": T9_FREEZER_SHA256,
        "submissions/codingame/promotion/reference/t8_evidence_manifest.json": (
            T8_EVIDENCE_MANIFEST_SHA256
        ),
        "submissions/codingame/promotion/reference/"
        "t8_prospective_validation.tsv": T8_PROSPECTIVE_VALIDATION_SHA256,
        "submissions/codingame/promotion/reference/t8_sealed_final.tsv": (
            T8_SEALED_FINAL_SHA256
        ),
    }
    if any(
        sources.get(path) != expected
        for path, expected in expected_sources.items()
    ):
        raise ValueError("T9 carry-forward source provenance changed")

    stages = protocol.get("stages", {})
    expected_profiles = {
        "validation": [
            ("30k-nodes", "nodes", 30_000, None),
            ("130ms", "time_ms", 130, 3_000_000),
        ],
        "test": [
            ("100k-nodes", "nodes", 100_000, None),
            ("130ms", "time_ms", 130, 3_000_000),
        ],
    }
    for stage, expected in expected_profiles.items():
        declaration = stages.get(stage, {})
        configuration = declaration.get("configuration", {})
        actual = [
            (
                profile.get("id"),
                profile.get("mode"),
                profile.get("value"),
                profile.get("max_nodes"),
            )
            for profile in configuration.get("strength_profiles", [])
        ]
        if (
            declaration.get("shard_count") != 4
            or declaration.get("jobs") != 4
            or configuration.get("required_jobs") != 4
            or actual != expected
        ):
            raise ValueError(f"T9 {stage} prospective execution contract changed")
    return evidence, data


def _t10_evidence_contract() -> tuple[dict, bytes]:
    """Verify the frozen T10 rollover and fresh sealed-final contract."""
    t9_evidence, _ = _t9_evidence_contract()
    data = T10_EVIDENCE_MANIFEST.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != T10_EVIDENCE_MANIFEST_SHA256:
        raise ValueError(
            "T10 evidence manifest hash mismatch: expected "
            f"{T10_EVIDENCE_MANIFEST_SHA256}, found {actual_sha256}"
        )
    evidence = json.loads(data)
    if (
        evidence.get("schema")
        != "papersoccer.candidate-independent-t10-evidence.v1"
        or evidence.get("status") != "frozen_before_candidate_binding"
        or evidence.get("candidate") is not None
        or evidence.get("candidate_submission_sha256") is not None
    ):
        raise ValueError("T10 evidence manifest binding status changed")
    protocol = evidence.get("prospective_strength_protocol")
    if (
        protocol != t9_evidence.get("prospective_strength_protocol")
        or sha256_bytes(stable_json(protocol)) != T10_STRENGTH_PROTOCOL_SHA256
        or evidence.get("prospective_strength_protocol_sha256")
        != T10_STRENGTH_PROTOCOL_SHA256
    ):
        raise ValueError("T10 inherited strength protocol changed")

    expected_banks = {
        "reference/t10_prospective_validation.tsv": {
            "role": "prospective_validation",
            "sealed": False,
            "records": 72,
            "sha256": T10_PROSPECTIVE_VALIDATION_SHA256,
        },
        "reference/t10_sealed_final.tsv": {
            "role": "sealed_final",
            "sealed": True,
            "records": 69,
            "sha256": T10_SEALED_FINAL_SHA256,
        },
    }
    for relative, expected in expected_banks.items():
        specification = evidence.get("banks", {}).get(relative, {})
        if any(specification.get(key) != value for key, value in expected.items()):
            raise ValueError(f"T10 evidence-bank declaration changed: {relative}")
    _verified_frozen_bank(
        T10_PROSPECTIVE_VALIDATION,
        T10_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    _verified_frozen_bank(
        T10_SEALED_FINAL, T10_SEALED_FINAL_SHA256, 69, "test"
    )

    audit = evidence.get("t9_decision_and_final_nonconsumption", {})
    if (
        audit.get("failed_stage") != "validation"
        or audit.get("stage_status", {}).get("test")
        != "not_run_due_to_rejection"
        or audit.get("final_report_exists") is not False
        or audit.get("final_shards_exist") is not False
        or audit.get("final_immutable_snapshot_exists") is not False
        or audit.get("final_ledger_marker_exists") is not False
    ):
        raise ValueError("T9 final nonconsumption provenance changed")
    expected_sources = {
        "submissions/codingame/promotion/acquire_t10_evidence.py": (
            T10_ACQUISITION_SHA256
        ),
        "submissions/codingame/promotion/freeze_t10_banks.py": T10_FREEZER_SHA256,
        "submissions/codingame/promotion/t10_evidence_ladder_v1.json": (
            T10_RAW_SNAPSHOT_SHA256
        ),
        "submissions/codingame/promotion/t10_evidence_ladder_v2.json": (
            T10_PARSEABLE_SNAPSHOT_SHA256
        ),
        "submissions/codingame/promotion/reference/t9_sealed_final.tsv": (
            T9_SEALED_FINAL_SHA256
        ),
    }
    sources = evidence.get("sources", {})
    if any(
        sources.get(path) != expected
        for path, expected in expected_sources.items()
    ):
        raise ValueError("T10 source provenance changed")
    for stage in ("validation", "test"):
        declaration = protocol.get("stages", {}).get(stage, {})
        if (
            declaration.get("jobs") != 4
            or declaration.get("shard_count") != 4
            or declaration.get("configuration", {}).get("required_jobs") != 4
        ):
            raise ValueError(f"T10 {stage} four-worker contract changed")
    return evidence, data


def _build_t7_artifacts():
    """Bind the active candidate to already-frozen, candidate-independent banks."""
    evidence, evidence_data = _t7_evidence_contract()

    development_bytes = _verified_frozen_bank(
        T3_DEVELOPMENT, T3_DEVELOPMENT_SHA256, 48, "development"
    )
    validation_bytes = _verified_frozen_bank(
        T7_PROSPECTIVE_VALIDATION,
        T7_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    final_bytes = _verified_frozen_bank(
        T7_SEALED_FINAL, T7_SEALED_FINAL_SHA256, 72, "test"
    )
    _verified_frozen_bank(
        T6_EXPOSED_VALIDATION,
        T6_EXPOSED_VALIDATION_SHA256,
        72,
        "validation",
    )
    _verified_frozen_bank(
        T6_EXPOSED_FINAL, T6_EXPOSED_FINAL_SHA256, 72, "test"
    )

    initial_state_key, initial_canonical_key, _, _, _ = state_identity(
        (4, 6), 0, set(), 0
    )
    initial_bytes = tsv_bytes([{
        "opening_id": "initial",
        "split": "initial",
        "stratum": "initial",
        "source_agent_id": 0,
        "source_game_id": 0,
        "opponent_agent_id": 0,
        "winner_player_id": 0,
        "turn_index": 0,
        "physical_edges": 0,
        "state_key": initial_state_key,
        "canonical_key": initial_canonical_key,
        "ball_x": 4,
        "ball_y": 6,
        "mover": 0,
        "winner_tier": "initial",
        "goal_distance_band": -1,
        "used_edge_band": "empty",
        "shell_edge_band": "empty",
        "opening_family": "initial",
        "observed_winner_action": "-",
        "transcript": "-",
    }])
    if sha256_bytes(initial_bytes) != PRESERVED_INITIAL_SHA256:
        raise ValueError("preserved initial bank bytes changed")

    banks = {
        "openings/initial.tsv": initial_bytes,
        "openings/development.tsv": development_bytes,
        "openings/validation.tsv": validation_bytes,
        "openings/test.tsv": final_bytes,
    }

    candidate_directory = HERE.parent / "bots" / "exchange_proof"
    candidate_submission = candidate_directory / "submission.cpp"
    if sha256_bytes(candidate_submission.read_bytes()) != (
            T7_CANDIDATE_SUBMISSION_SHA256):
        raise ValueError("T7 candidate submission hash changed")
    acquisition = HERE / "acquire_t7_evidence.py"
    freezer = HERE / "freeze_t7_banks.py"
    if sha256_bytes(acquisition.read_bytes()) != T7_ACQUISITION_SHA256:
        raise ValueError("T7 acquisition controller hash changed")
    if sha256_bytes(freezer.read_bytes()) != T7_FREEZER_SHA256:
        raise ValueError("T7 bank freezer hash changed")

    source_paths = (
        candidate_submission,
        candidate_directory / "comparison_gate.cpp",
        candidate_directory / "submission_test.cpp",
        candidate_directory / "timing_probe.cpp",
        HERE / "build_goal_shell_banks.py",
        HERE.parent / "tools" / "promotion_gate.py",
        acquisition,
        freezer,
        T7_EVIDENCE_MANIFEST,
        T3_DEVELOPMENT,
        T6_EXPOSED_VALIDATION,
        T6_EXPOSED_FINAL,
        T7_PROSPECTIVE_VALIDATION,
        T7_SEALED_FINAL,
        ROOT / "submissions/codingame/bots/rank_5/submission.cpp",
    )
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
        for path in source_paths
    }
    protocol = evidence["prospective_strength_protocol"]
    protocol_stages = protocol["stages"]
    stage_banks = {
        "initial": "openings/initial.tsv",
        "development": "openings/development.tsv",
        "validation": "openings/validation.tsv",
        "test": "openings/test.tsv",
    }
    stages = {}
    for stage, bank in stage_banks.items():
        configuration = json.loads(json.dumps(
            protocol_stages[stage]["configuration"]
        ))
        configuration["bank"] = bank
        stages[stage] = configuration

    evidence_relative = str(T7_EVIDENCE_MANIFEST.relative_to(ROOT))
    acquisition_relative = str(acquisition.relative_to(ROOT))
    freezer_relative = str(freezer.relative_to(ROOT))
    manifest = {
        "schema": "papersoccer.codingame-promotion-manifest.v1",
        "candidate": "exchange_proof",
        "candidate_submission_sha256": T7_CANDIDATE_SUBMISSION_SHA256,
        "evidence_manifest": evidence_relative,
        "evidence_manifest_sha256": T7_EVIDENCE_MANIFEST_SHA256,
        "prospective_strength_protocol_sha256": T7_STRENGTH_PROTOCOL_SHA256,
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "own_goals_allowed",
            "blocked_rule": "mover_loses",
            "positions_are_complete_turn_boundaries": True,
        },
        "incumbent": {
            "name": "rank_5",
            "submission_sha256": INCUMBENT_SUBMISSION_SHA256,
        },
        "hypothesis": (
            "retain unconditional current-turn root and depth-zero rebound-"
            "component proofs, then add TT-first exact Win/Loss proofs only at "
            "the first two complete-turn boundaries below each root action "
            "(turn_ply == 1 or 2), covering the opponent reply plus the root "
            "player's counterturn; Unknown positions and all heuristics remain "
            "unchanged"
        ),
        "evidence": {
            "status": "candidate_bound_before_t7_outcomes",
            "candidate_independent_selection": True,
            "provenance_split": (
                "the immutable evidence manifest records candidate-independent "
                "pre-binding acquisition, freezer, builder, and controller "
                "identities; this active manifest pins that snapshot and the final "
                "post-binding builder, controller, candidate, and harness "
                "identities, so historical source hashes inside the evidence "
                "manifest are provenance rather than current-source expectations"
            ),
            "manifest": {
                "path": evidence_relative,
                "sha256": T7_EVIDENCE_MANIFEST_SHA256,
                "schema": evidence["schema"],
            },
            "prospective_strength_protocol_sha256": (
                T7_STRENGTH_PROTOCOL_SHA256
            ),
            "acquisition": {
                "controller": {
                    "path": acquisition_relative,
                    "sha256": T7_ACQUISITION_SHA256,
                },
                "freezer": {
                    "path": freezer_relative,
                    "sha256": T7_FREEZER_SHA256,
                },
                "raw_snapshot": {
                    "path": "submissions/codingame/promotion/"
                            "t7_evidence_ladder_v1.json",
                    "sha256": T7_RAW_SNAPSHOT_SHA256,
                },
                "parseable_snapshot": {
                    "path": "submissions/codingame/promotion/"
                            "t7_evidence_ladder_v2.json",
                    "sha256": T7_PARSEABLE_SNAPSHOT_SHA256,
                },
            },
            "banks": {
                "validation": {
                    "active": "openings/validation.tsv",
                    "reference": "reference/t7_prospective_validation.tsv",
                    "role": "prospective_validation",
                    "sealed": False,
                    "sha256": T7_PROSPECTIVE_VALIDATION_SHA256,
                },
                "test": {
                    "active": "openings/test.tsv",
                    "reference": "reference/t7_sealed_final.tsv",
                    "role": "sealed_final",
                    "sealed": True,
                    "sha256": T7_SEALED_FINAL_SHA256,
                },
            },
            "prior_decisions": evidence["prior_decisions"],
            "immutability": {
                "t6": (
                    "the exposed T6 validation rejection and final-bank use are "
                    "immutable provenance; T7 neither regenerates nor re-evaluates "
                    "them"
                ),
                "t7": evidence["immutability"]["after_candidate_binding"],
            },
        },
        "selection": {
            "normalization": evidence["selection"]["normalization"],
            "goal_shell": "historical winner to move at normalized y 9, 10, or 11",
            "development": (
                "byte-identical frozen T3 development bank retained from T6"
            ),
            "validation": (
                "byte-identical candidate-independent prospective T7 validation "
                "reference selected and frozen before candidate binding"
            ),
            "test": (
                "byte-identical candidate-independent sealed T7 final reference, "
                "source-game and canonical-key disjoint from validation"
            ),
            "bank_copy": (
                "the builder verifies each frozen reference hash, record count, "
                "header, and already-correct split, then copies its bytes directly "
                "to the active path"
            ),
            "bank_snapshot": (
                "the controller verifies each active bank hash, snapshots the "
                "verified bytes into the immutable result identity, and never "
                "reopens a mutable bank path in an in-flight shard"
            ),
            "proof_diagnostics": (
                "rebound_goal_probes counts all exact component analyses, "
                "rebound_goal_hits counts proven reachable attacking goals, and "
                "rebound_loss_hits counts components with no safe handoff under "
                "mover-loses; exchange_ply1 and exchange_ply2 each report probes, "
                "Win hits, Loss hits, and exact cutoffs separately"
            ),
            "proof_placement": (
                "exchange_proof keeps the rebound_proof current-turn root and "
                "depth-zero checks; at positive depth it probes the transposition "
                "table first and runs the exact component analyzer only at "
                "turn_ply one or two, covering the opponent reply and root-player "
                "counterturn"
            ),
            "control_normalization": (
                "one rank_5-vs-rank_5 control uses the same opening and execution "
                "profile; physical-color uplift remains a hard gate, historical-"
                "winner/opponent role fields are diagnostic only, and time-profile "
                "throughput is diagnostic only"
            ),
            "protocol": {
                "status": protocol["status"],
                "stage_bank_policy": protocol["stage_bank_policy"],
                "game_format": protocol["game_format"],
                "execution_contract": protocol["execution_contract"],
                "diagnostic_only": protocol["diagnostic_only"],
            },
            "t6_provenance": evidence["prior_decisions"],
        },
        "sources": dict(sorted(source_hashes.items())),
        "banks": {
            path: {
                "sha256": sha256_bytes(content),
                "records": content.count(b"\n") - 1,
            }
            for path, content in sorted(banks.items())
        },
        "stages": stages,
        "statistics": protocol["statistics"],
        "timing": {
            "fresh_process_samples": 20,
            "shell_cases": ["elite-d2", "rank1-d1", "elite-d0-dense"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        },
        "source_limit": 100000,
    }
    if source_hashes[evidence_relative] != sha256_bytes(evidence_data):
        raise AssertionError("T7 evidence-manifest source binding diverged")
    return banks, stable_json(manifest)


# Set this only after the exposed screen chooses a candidate. Keeping the
# candidate and all harness identities together makes the one-way T8 binding
# transition explicit and lets the default --check reproduce the active files.
T8_ACTIVE_BINDING: dict | None = {
    "candidate": "frontier_proof",
    "candidate_submission_sha256": (
        "35ffa4c9b30327750c1ca5fa50f6d41a282252f98d187c372a74131b148cafe1"
    ),
    "comparison_gate_sha256": (
        "fbb99b6150a9326e55b12873d86033df4fd987f9096e38496d40dd4f1d2230dc"
    ),
    "submission_test_sha256": (
        "84add712b00f2abdeb57c307fbae784119a83f599b7a819c815ea5372547ac62"
    ),
    "timing_probe_sha256": (
        "b971c0f6f7ae37c93bf90efde2f757662b8c0a30a5b05c80134d537603254dd7"
    ),
    "hypothesis": (
        "retain all-depth exact rebound-component Win/Loss proofs and, only "
        "for Unknown leaf components, add a mover-symmetric count of unique "
        "safe handoff endpoints at half the existing mobility weight (10), "
        "leaving all other evaluation and search behavior unchanged"
    ),
}

# T9 is frozen and verified; the selected candidate and every maintained
# source identity are bound together before prospective evidence is run.
T9_ACTIVE_BINDING: dict | None = {
    "candidate": "conservative_frontier_proof",
    "candidate_submission_sha256": (
        "b13e1418b4fdd6f719208bd5ab6dd84f67fa55a7cc66f1c988d2ab8dcc9f6c69"
    ),
    "comparison_gate_sha256": (
        "ab0e13ebfefe681d74ac15743c288886e47efd98da0d50e90c3da1213a4166b0"
    ),
    "submission_test_sha256": (
        "5d306c1a897940fc81f5d844a26d4eabcdf4f4bf4a94912651d91fbe8edd4232"
    ),
    "timing_probe_sha256": (
        "b971c0f6f7ae37c93bf90efde2f757662b8c0a30a5b05c80134d537603254dd7"
    ),
    "hypothesis": (
        "retain all-depth exact rebound-component Win/Loss proofs and, only "
        "for Unknown leaf components, add a mover-symmetric count of unique "
        "safe handoff endpoints at a conservative quarter of the existing "
        "mobility weight (5), leaving all other evaluation and search "
        "behavior unchanged"
    ),
}

T10_ACTIVE_BINDING: dict | None = {
    "candidate": "all_depth_proof",
    "candidate_submission_sha256": (
        "c9895b18b3117c5a93909da28aaff11977afaf76c584ce310c7f153694fd6376"
    ),
    "comparison_gate_sha256": (
        "66c0f6338f6df4d58ff4cc613cd2ddc8a2a2a2336c0f6b443d15a61d55eeac22"
    ),
    "submission_test_sha256": (
        "2ff3067dffd2d87a84c984b402ae4c1dc32c0a1288cabde86e7ed2261c45d0da"
    ),
    "timing_probe_sha256": (
        "b971c0f6f7ae37c93bf90efde2f757662b8c0a30a5b05c80134d537603254dd7"
    ),
    "hypothesis": (
        "retain the exact rebound-component Win/Loss proof at the current-turn "
        "root and depth-zero leaves, then apply the same proof at every positive-"
        "depth complete-turn node after the transposition-table probe; Unknown "
        "positions continue through unchanged evaluation and search"
    ),
}


def _binding_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def build_t8_artifacts(
    *,
    candidate: str,
    candidate_submission_sha256: str,
    comparison_gate_sha256: str,
    submission_test_sha256: str,
    timing_probe_sha256: str,
    hypothesis: str,
):
    """Bind one hash-pinned candidate to the frozen T8 evidence ladder."""
    evidence, evidence_data = _t8_evidence_contract()
    if not candidate or pathlib.Path(candidate).name != candidate:
        raise ValueError("T8 candidate must be one bot-directory name")
    if not hypothesis.strip():
        raise ValueError("T8 candidate hypothesis must be non-empty")
    expected_candidate_hashes = {
        "submission.cpp": _binding_sha256(
            candidate_submission_sha256, "candidate submission hash"
        ),
        "comparison_gate.cpp": _binding_sha256(
            comparison_gate_sha256, "comparison runner source hash"
        ),
        "submission_test.cpp": _binding_sha256(
            submission_test_sha256, "submission test source hash"
        ),
        "timing_probe.cpp": _binding_sha256(
            timing_probe_sha256, "timing probe source hash"
        ),
    }

    development_bytes = _verified_frozen_bank(
        T3_DEVELOPMENT, T3_DEVELOPMENT_SHA256, 48, "development"
    )
    validation_bytes = _verified_frozen_bank(
        T8_PROSPECTIVE_VALIDATION,
        T8_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    final_bytes = _verified_frozen_bank(
        T8_SEALED_FINAL, T8_SEALED_FINAL_SHA256, 72, "test"
    )
    # Preserve and explicitly verify the two exposed T7 banks as immutable
    # decision provenance. They are never copied into an active T8 stage.
    _verified_frozen_bank(
        T7_PROSPECTIVE_VALIDATION,
        T7_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    _verified_frozen_bank(
        T7_SEALED_FINAL, T7_SEALED_FINAL_SHA256, 72, "test"
    )

    initial_state_key, initial_canonical_key, _, _, _ = state_identity(
        (4, 6), 0, set(), 0
    )
    initial_bytes = tsv_bytes([{
        "opening_id": "initial",
        "split": "initial",
        "stratum": "initial",
        "source_agent_id": 0,
        "source_game_id": 0,
        "opponent_agent_id": 0,
        "winner_player_id": 0,
        "turn_index": 0,
        "physical_edges": 0,
        "state_key": initial_state_key,
        "canonical_key": initial_canonical_key,
        "ball_x": 4,
        "ball_y": 6,
        "mover": 0,
        "winner_tier": "initial",
        "goal_distance_band": -1,
        "used_edge_band": "empty",
        "shell_edge_band": "empty",
        "opening_family": "initial",
        "observed_winner_action": "-",
        "transcript": "-",
    }])
    if sha256_bytes(initial_bytes) != PRESERVED_INITIAL_SHA256:
        raise ValueError("preserved initial bank bytes changed")

    banks = {
        "openings/initial.tsv": initial_bytes,
        "openings/development.tsv": development_bytes,
        "openings/validation.tsv": validation_bytes,
        "openings/test.tsv": final_bytes,
    }

    candidate_directory = HERE.parent / "bots" / candidate
    if not candidate_directory.is_dir():
        raise ValueError(f"missing T8 candidate directory: {candidate_directory}")
    candidate_paths = {
        name: candidate_directory / name for name in expected_candidate_hashes
    }
    for name, path in candidate_paths.items():
        actual = sha256_bytes(path.read_bytes())
        expected = expected_candidate_hashes[name]
        if actual != expected:
            raise ValueError(
                f"T8 candidate {name} hash mismatch: expected {expected}, "
                f"found {actual}"
            )

    acquisition = HERE / "acquire_t8_evidence.py"
    freezer = HERE / "freeze_t8_banks.py"
    gate = HERE.parent / "tools" / "promotion_gate.py"
    if sha256_bytes(acquisition.read_bytes()) != T8_ACQUISITION_SHA256:
        raise ValueError("T8 acquisition controller hash changed")
    if sha256_bytes(freezer.read_bytes()) != T8_FREEZER_SHA256:
        raise ValueError("T8 bank freezer hash changed")
    if sha256_bytes(gate.read_bytes()) != T8_PROMOTION_GATE_SHA256:
        raise ValueError("T8 post-binding promotion controller hash changed")

    source_paths = (
        *candidate_paths.values(),
        HERE / "build_goal_shell_banks.py",
        gate,
        acquisition,
        freezer,
        T8_EVIDENCE_MANIFEST,
        HERE / "t8_evidence_ladder_v1.json",
        HERE / "t8_evidence_ladder_v2.json",
        T3_DEVELOPMENT,
        T7_PROSPECTIVE_VALIDATION,
        T7_SEALED_FINAL,
        T8_PROSPECTIVE_VALIDATION,
        T8_SEALED_FINAL,
        ROOT / "submissions/codingame/bots/rank_5/submission.cpp",
    )
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
        for path in source_paths
    }
    # T8 is a retired historical binding. Preserve the exact builder identity
    # that created its active manifest while inactive T9 code is prepared.
    source_hashes[str(
        (HERE / "build_goal_shell_banks.py").relative_to(ROOT)
    )] = T8_POSTBIND_BUILDER_SHA256

    protocol = evidence["prospective_strength_protocol"]
    protocol_stages = protocol["stages"]
    stage_banks = {
        "initial": "openings/initial.tsv",
        "development": "openings/development.tsv",
        "validation": "openings/validation.tsv",
        "test": "openings/test.tsv",
    }
    stages = {}
    for stage, bank in stage_banks.items():
        configuration = json.loads(json.dumps(
            protocol_stages[stage]["configuration"]
        ))
        configuration["bank"] = bank
        stages[stage] = configuration
    if (
        stages["validation"].get("required_jobs") != 4
        or stages["test"].get("required_jobs") != 4
    ):
        raise AssertionError("T8 prospective required_jobs mapping diverged")

    evidence_relative = str(T8_EVIDENCE_MANIFEST.relative_to(ROOT))
    acquisition_relative = str(acquisition.relative_to(ROOT))
    freezer_relative = str(freezer.relative_to(ROOT))
    builder_relative = str(
        (HERE / "build_goal_shell_banks.py").relative_to(ROOT)
    )
    gate_relative = str(gate.relative_to(ROOT))
    candidate_sources = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": expected_candidate_hashes[name],
        }
        for name, path in candidate_paths.items()
    }
    manifest = {
        "schema": "papersoccer.codingame-promotion-manifest.v1",
        "candidate": candidate,
        "candidate_submission_sha256": candidate_submission_sha256,
        "evidence_manifest": evidence_relative,
        "evidence_manifest_sha256": T8_EVIDENCE_MANIFEST_SHA256,
        "prospective_strength_protocol_sha256": T8_STRENGTH_PROTOCOL_SHA256,
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "own_goals_allowed",
            "blocked_rule": "mover_loses",
            "positions_are_complete_turn_boundaries": True,
        },
        "incumbent": {
            "name": "rank_5",
            "submission_sha256": INCUMBENT_SUBMISSION_SHA256,
        },
        "hypothesis": hypothesis.strip(),
        "evidence": {
            "status": "candidate_bound_before_t8_outcomes",
            "candidate_independent_selection": True,
            "provenance_split": (
                "the immutable evidence manifest pins the candidate-independent "
                "pre-binding acquisition, freezer, evidence, and builder hashes; "
                "this active manifest separately pins the post-binding builder, "
                "promotion controller, candidate submission, and three harness "
                "source identities"
            ),
            "manifest": {
                "path": evidence_relative,
                "sha256": T8_EVIDENCE_MANIFEST_SHA256,
                "schema": evidence["schema"],
            },
            "prospective_strength_protocol_sha256": (
                T8_STRENGTH_PROTOCOL_SHA256
            ),
            "provenance": {
                "pre_binding": {
                    "builder": {
                        "path": builder_relative,
                        "sha256": T8_PREBIND_BUILDER_SHA256,
                    },
                    "acquisition": {
                        "path": acquisition_relative,
                        "sha256": T8_ACQUISITION_SHA256,
                    },
                    "freezer": {
                        "path": freezer_relative,
                        "sha256": T8_FREEZER_SHA256,
                    },
                    "evidence_manifest": {
                        "path": evidence_relative,
                        "sha256": T8_EVIDENCE_MANIFEST_SHA256,
                    },
                    "protocol_sha256": T8_STRENGTH_PROTOCOL_SHA256,
                },
                "post_binding": {
                    "builder": {
                        "path": builder_relative,
                        "sha256": source_hashes[builder_relative],
                    },
                    "promotion_controller": {
                        "path": gate_relative,
                        "sha256": T8_PROMOTION_GATE_SHA256,
                    },
                    "candidate": candidate_sources,
                },
            },
            "acquisition": {
                "controller": {
                    "path": acquisition_relative,
                    "sha256": T8_ACQUISITION_SHA256,
                },
                "freezer": {
                    "path": freezer_relative,
                    "sha256": T8_FREEZER_SHA256,
                },
                "raw_snapshot": {
                    "path": "submissions/codingame/promotion/"
                            "t8_evidence_ladder_v1.json",
                    "sha256": T8_RAW_SNAPSHOT_SHA256,
                },
                "parseable_snapshot": {
                    "path": "submissions/codingame/promotion/"
                            "t8_evidence_ladder_v2.json",
                    "sha256": T8_PARSEABLE_SNAPSHOT_SHA256,
                },
            },
            "banks": {
                "validation": {
                    "active": "openings/validation.tsv",
                    "reference": "reference/t8_prospective_validation.tsv",
                    "role": "prospective_validation",
                    "sealed": False,
                    "sha256": T8_PROSPECTIVE_VALIDATION_SHA256,
                },
                "test": {
                    "active": "openings/test.tsv",
                    "reference": "reference/t8_sealed_final.tsv",
                    "role": "sealed_final",
                    "sealed": True,
                    "sha256": T8_SEALED_FINAL_SHA256,
                },
            },
            "prior_decisions": evidence["prior_decisions"],
            "immutability": {
                "t7": (
                    "both exposed T7 bank identities are immutable decision "
                    "provenance and are neither regenerated nor re-evaluated"
                ),
                "t8": evidence["immutability"]["after_candidate_binding"],
            },
        },
        "selection": {
            "normalization": evidence["selection"]["normalization"],
            "goal_shell": "historical winner to move at normalized y 9, 10, or 11",
            "initial": (
                "byte-identical exposed adaptive initial bank retained from T7"
            ),
            "development": (
                "byte-identical exposed adaptive T3 development bank retained "
                "from T7"
            ),
            "validation": (
                "byte-identical candidate-independent prospective T8 validation "
                "reference selected and frozen before candidate binding"
            ),
            "test": (
                "byte-identical candidate-independent sealed T8 final reference, "
                "source-game and canonical-key disjoint from validation"
            ),
            "bank_copy": (
                "the builder verifies each frozen reference hash, record count, "
                "header, and split, then copies its bytes directly to the active path"
            ),
            "bank_snapshot": (
                "the controller verifies each active bank hash, snapshots verified "
                "bytes into the immutable result identity, and never reopens a "
                "mutable bank path in an in-flight shard"
            ),
            "control_normalization": (
                "one rank_5-vs-rank_5 control uses the same opening and profile; "
                "physical-color uplift remains a hard gate, historical roles are "
                "diagnostic, and time-profile throughput is diagnostic"
            ),
            "protocol": {
                "status": protocol["status"],
                "stage_bank_policy": protocol["stage_bank_policy"],
                "game_format": protocol["game_format"],
                "execution_contract": protocol["execution_contract"],
                "diagnostic_only": protocol["diagnostic_only"],
            },
            "t7_provenance": evidence["prior_decisions"],
        },
        "sources": dict(sorted(source_hashes.items())),
        "banks": {
            path: {
                "sha256": sha256_bytes(content),
                "records": content.count(b"\n") - 1,
            }
            for path, content in sorted(banks.items())
        },
        "stages": stages,
        "statistics": protocol["statistics"],
        "timing": {
            "fresh_process_samples": 20,
            "shell_cases": ["elite-d2", "rank1-d1", "elite-d0-dense"],
            "first_p95_ms": 950.0,
            "first_max_ms": 1000.0,
            "later_p95_ms": 190.0,
            "later_max_ms": 200.0,
        },
        "source_limit": 100000,
    }
    if source_hashes[evidence_relative] != sha256_bytes(evidence_data):
        raise AssertionError("T8 evidence-manifest source binding diverged")
    return banks, stable_json(manifest)


def build_t9_artifacts(
    *,
    candidate: str,
    candidate_submission_sha256: str,
    comparison_gate_sha256: str,
    submission_test_sha256: str,
    timing_probe_sha256: str,
    hypothesis: str,
):
    """Bind one hash-pinned candidate to the inactive frozen T9 ladder."""
    evidence, evidence_data = _t9_evidence_contract()

    # Reuse the already-tested candidate, incumbent, initial-bank, exposed-T3,
    # harness, and controller verification path. The prospective bytes are then
    # replaced explicitly from the byte-identical versioned T9 aliases below.
    banks, t8_manifest_data = build_t8_artifacts(
        candidate=candidate,
        candidate_submission_sha256=candidate_submission_sha256,
        comparison_gate_sha256=comparison_gate_sha256,
        submission_test_sha256=submission_test_sha256,
        timing_probe_sha256=timing_probe_sha256,
        hypothesis=hypothesis,
    )
    banks = dict(banks)
    banks["openings/validation.tsv"] = _verified_frozen_bank(
        T9_PROSPECTIVE_VALIDATION,
        T9_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    banks["openings/test.tsv"] = _verified_frozen_bank(
        T9_SEALED_FINAL, T9_SEALED_FINAL_SHA256, 72, "test"
    )
    if (
        sha256_bytes(banks["openings/initial.tsv"]) != PRESERVED_INITIAL_SHA256
        or sha256_bytes(banks["openings/development.tsv"])
        != T3_DEVELOPMENT_SHA256
    ):
        raise AssertionError("T9 exposed prerequisite banks changed")

    manifest = json.loads(t8_manifest_data)
    evidence_relative = str(T9_EVIDENCE_MANIFEST.relative_to(ROOT))
    freezer = HERE / "freeze_t9_banks.py"
    freezer_relative = str(freezer.relative_to(ROOT))
    builder_relative = str(
        (HERE / "build_goal_shell_banks.py").relative_to(ROOT)
    )
    current_builder_sha256 = sha256_bytes(
        (HERE / "build_goal_shell_banks.py").read_bytes()
    )

    manifest["evidence_manifest"] = evidence_relative
    manifest["evidence_manifest_sha256"] = T9_EVIDENCE_MANIFEST_SHA256
    manifest["prospective_strength_protocol_sha256"] = (
        T9_STRENGTH_PROTOCOL_SHA256
    )
    manifest["evidence"] = {
        "status": "candidate_bound_before_t9_outcomes",
        "candidate_independent_selection": True,
        "provenance_split": (
            "the immutable T9 manifest verifies the outcome-unseen T8-to-T9 "
            "carry-forward and inherited protocol before binding; this active "
            "manifest separately pins the post-binding builder, controller, "
            "candidate submission, and three harness source identities"
        ),
        "manifest": {
            "path": evidence_relative,
            "sha256": T9_EVIDENCE_MANIFEST_SHA256,
            "schema": evidence["schema"],
        },
        "prospective_strength_protocol_sha256": T9_STRENGTH_PROTOCOL_SHA256,
        "protocol_carry_forward": evidence["protocol_carry_forward"],
        "nonconsumption_audit": evidence["nonconsumption_audit"],
        "provenance": {
            "pre_binding": {
                "carry_forward_freezer": {
                    "path": freezer_relative,
                    "sha256": T9_FREEZER_SHA256,
                },
                "source_evidence_manifest": {
                    "path": str(T8_EVIDENCE_MANIFEST.relative_to(ROOT)),
                    "sha256": T8_EVIDENCE_MANIFEST_SHA256,
                },
                "evidence_manifest": {
                    "path": evidence_relative,
                    "sha256": T9_EVIDENCE_MANIFEST_SHA256,
                },
                "protocol_sha256": T9_STRENGTH_PROTOCOL_SHA256,
            },
            "post_binding": {
                "builder": {
                    "path": builder_relative,
                    "sha256": current_builder_sha256,
                },
                "promotion_controller": manifest["evidence"]["provenance"][
                    "post_binding"
                ]["promotion_controller"],
                "candidate": manifest["evidence"]["provenance"][
                    "post_binding"
                ]["candidate"],
            },
        },
        "acquisition": {
            **manifest["evidence"]["acquisition"],
            "carry_forward_freezer": {
                "path": freezer_relative,
                "sha256": T9_FREEZER_SHA256,
            },
        },
        "banks": {
            "validation": {
                "active": "openings/validation.tsv",
                "reference": "reference/t9_prospective_validation.tsv",
                "carried_forward_from": (
                    "reference/t8_prospective_validation.tsv"
                ),
                "role": "prospective_validation",
                "sealed": False,
                "sha256": T9_PROSPECTIVE_VALIDATION_SHA256,
            },
            "test": {
                "active": "openings/test.tsv",
                "reference": "reference/t9_sealed_final.tsv",
                "carried_forward_from": "reference/t8_sealed_final.tsv",
                "role": "sealed_final",
                "sealed": True,
                "sha256": T9_SEALED_FINAL_SHA256,
            },
        },
        "prior_decisions": evidence["prior_decisions"],
        "immutability": {
            "t8": (
                "the frontier_proof binding is retired solely on exposed "
                "development failure; no prospective T8 evidence was consumed"
            ),
            "t9": evidence["immutability"]["after_candidate_binding"],
        },
    }

    protocol = evidence["prospective_strength_protocol"]
    manifest["selection"].update({
        "initial": (
            "byte-identical exposed adaptive initial bank retained from T8"
        ),
        "development": (
            "byte-identical exposed adaptive T3 development bank retained "
            "from T8"
        ),
        "validation": (
            "byte-identical candidate-independent prospective T9 alias carried "
            "forward outcome-unseen from T8 before candidate binding"
        ),
        "test": (
            "byte-identical candidate-independent sealed T9 alias carried "
            "forward outcome-unseen from T8 and disjoint from validation"
        ),
        "protocol": {
            "status": protocol["status"],
            "stage_bank_policy": protocol["stage_bank_policy"],
            "game_format": protocol["game_format"],
            "execution_contract": protocol["execution_contract"],
            "diagnostic_only": protocol["diagnostic_only"],
        },
        "t8_provenance": evidence["prior_decisions"],
    })
    manifest["selection"].pop("t7_provenance", None)

    source_hashes = dict(manifest["sources"])
    t9_source_paths = (
        T9_EVIDENCE_MANIFEST,
        freezer,
        T9_PROSPECTIVE_VALIDATION,
        T9_SEALED_FINAL,
    )
    source_hashes.update({
        str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
        for path in t9_source_paths
    })
    source_hashes[builder_relative] = current_builder_sha256
    manifest["sources"] = dict(sorted(source_hashes.items()))
    manifest["banks"] = {
        path: {
            "sha256": sha256_bytes(content),
            "records": content.count(b"\n") - 1,
        }
        for path, content in sorted(banks.items())
    }

    if (
        manifest["stages"]["validation"].get("required_jobs") != 4
        or manifest["stages"]["test"].get("required_jobs") != 4
    ):
        raise AssertionError("T9 prospective required_jobs mapping diverged")
    if source_hashes[evidence_relative] != sha256_bytes(evidence_data):
        raise AssertionError("T9 evidence-manifest source binding diverged")
    return banks, stable_json(manifest)


def build_t10_artifacts(
    *,
    candidate: str,
    candidate_submission_sha256: str,
    comparison_gate_sha256: str,
    submission_test_sha256: str,
    timing_probe_sha256: str,
    hypothesis: str,
):
    """Bind one hash-pinned candidate to the frozen T10 evidence ladder."""
    evidence, evidence_data = _t10_evidence_contract()
    banks, t9_manifest_data = build_t9_artifacts(
        candidate=candidate,
        candidate_submission_sha256=candidate_submission_sha256,
        comparison_gate_sha256=comparison_gate_sha256,
        submission_test_sha256=submission_test_sha256,
        timing_probe_sha256=timing_probe_sha256,
        hypothesis=hypothesis,
    )
    banks = dict(banks)
    banks["openings/validation.tsv"] = _verified_frozen_bank(
        T10_PROSPECTIVE_VALIDATION,
        T10_PROSPECTIVE_VALIDATION_SHA256,
        72,
        "validation",
    )
    banks["openings/test.tsv"] = _verified_frozen_bank(
        T10_SEALED_FINAL, T10_SEALED_FINAL_SHA256, 69, "test"
    )
    if (
        sha256_bytes(banks["openings/initial.tsv"]) != PRESERVED_INITIAL_SHA256
        or sha256_bytes(banks["openings/development.tsv"])
        != T3_DEVELOPMENT_SHA256
    ):
        raise AssertionError("T10 exposed prerequisite banks changed")

    manifest = json.loads(t9_manifest_data)
    old_evidence = manifest["evidence"]
    evidence_relative = str(T10_EVIDENCE_MANIFEST.relative_to(ROOT))
    acquisition = HERE / "acquire_t10_evidence.py"
    freezer = HERE / "freeze_t10_banks.py"
    builder = HERE / "build_goal_shell_banks.py"
    gate = HERE.parent / "tools" / "promotion_gate.py"
    builder_relative = str(builder.relative_to(ROOT))
    gate_relative = str(gate.relative_to(ROOT))
    current_builder_sha256 = sha256_bytes(builder.read_bytes())
    current_gate_sha256 = sha256_bytes(gate.read_bytes())

    manifest["evidence_manifest"] = evidence_relative
    manifest["evidence_manifest_sha256"] = T10_EVIDENCE_MANIFEST_SHA256
    manifest["prospective_strength_protocol_sha256"] = (
        T10_STRENGTH_PROTOCOL_SHA256
    )
    manifest["evidence"] = {
        "status": "candidate_bound_before_t10_outcomes",
        "candidate_independent_selection": True,
        "provenance_split": (
            "the immutable T10 manifest pins the rejected T9 validation, "
            "unconsumed-final rollover, fresh acquisition, banks, and inherited "
            "protocol; this active manifest separately pins the builder, "
            "controller, candidate submission, and three harness identities"
        ),
        "manifest": {
            "path": evidence_relative,
            "sha256": T10_EVIDENCE_MANIFEST_SHA256,
            "schema": evidence["schema"],
        },
        "prospective_strength_protocol_sha256": T10_STRENGTH_PROTOCOL_SHA256,
        "protocol_carry_forward": evidence["protocol_carry_forward"],
        "t9_decision_and_final_nonconsumption": (
            evidence["t9_decision_and_final_nonconsumption"]
        ),
        "provenance": {
            "pre_binding": {
                "acquisition": {
                    "path": str(acquisition.relative_to(ROOT)),
                    "sha256": T10_ACQUISITION_SHA256,
                },
                "freezer": {
                    "path": str(freezer.relative_to(ROOT)),
                    "sha256": T10_FREEZER_SHA256,
                },
                "evidence_manifest": {
                    "path": evidence_relative,
                    "sha256": T10_EVIDENCE_MANIFEST_SHA256,
                },
                "protocol_sha256": T10_STRENGTH_PROTOCOL_SHA256,
            },
            "post_binding": {
                "builder": {
                    "path": builder_relative,
                    "sha256": current_builder_sha256,
                },
                "promotion_controller": {
                    "path": gate_relative,
                    "sha256": current_gate_sha256,
                },
                "candidate": old_evidence["provenance"]["post_binding"][
                    "candidate"
                ],
            },
        },
        "acquisition": {
            "controller": {
                "path": str(acquisition.relative_to(ROOT)),
                "sha256": T10_ACQUISITION_SHA256,
            },
            "freezer": {
                "path": str(freezer.relative_to(ROOT)),
                "sha256": T10_FREEZER_SHA256,
            },
            "raw_snapshot": {
                "path": "submissions/codingame/promotion/"
                        "t10_evidence_ladder_v1.json",
                "sha256": T10_RAW_SNAPSHOT_SHA256,
            },
            "parseable_snapshot": {
                "path": "submissions/codingame/promotion/"
                        "t10_evidence_ladder_v2.json",
                "sha256": T10_PARSEABLE_SNAPSHOT_SHA256,
            },
        },
        "banks": {
            "validation": {
                "active": "openings/validation.tsv",
                "reference": "reference/t10_prospective_validation.tsv",
                "carried_forward_from": "reference/t9_sealed_final.tsv",
                "role": "prospective_validation",
                "sealed": False,
                "sha256": T10_PROSPECTIVE_VALIDATION_SHA256,
            },
            "test": {
                "active": "openings/test.tsv",
                "reference": "reference/t10_sealed_final.tsv",
                "role": "sealed_final",
                "sealed": True,
                "sha256": T10_SEALED_FINAL_SHA256,
            },
        },
        "prior_decisions": {
            "t9": evidence["t9_decision_and_final_nonconsumption"]
        },
        "immutability": {
            "t9": (
                "validation is exposed and rejected; its final was never "
                "consumed and is now the T10 prospective validation bank"
            ),
            "t10": evidence["immutability"]["after_candidate_binding"],
        },
    }

    protocol = evidence["prospective_strength_protocol"]
    manifest["selection"].update({
        "initial": "byte-identical exposed adaptive initial bank retained",
        "development": (
            "byte-identical exposed adaptive T3 development bank retained"
        ),
        "validation": evidence["selection"]["validation"],
        "test": evidence["selection"]["final"],
        "protocol": {
            "status": protocol["status"],
            "stage_bank_policy": protocol["stage_bank_policy"],
            "game_format": protocol["game_format"],
            "execution_contract": protocol["execution_contract"],
            "diagnostic_only": protocol["diagnostic_only"],
        },
        "t9_provenance": evidence["t9_decision_and_final_nonconsumption"],
    })
    manifest["selection"].pop("t8_provenance", None)

    source_hashes = dict(manifest["sources"])
    t10_sources = (
        acquisition,
        freezer,
        HERE / "t10_evidence_ladder_v1.json",
        HERE / "t10_evidence_ladder_v2.json",
        T10_EVIDENCE_MANIFEST,
        T10_PROSPECTIVE_VALIDATION,
        T10_SEALED_FINAL,
    )
    source_hashes.update({
        str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
        for path in t10_sources
    })
    source_hashes[builder_relative] = current_builder_sha256
    source_hashes[gate_relative] = current_gate_sha256
    manifest["sources"] = dict(sorted(source_hashes.items()))
    manifest["banks"] = {
        path: {
            "sha256": sha256_bytes(content),
            "records": content.count(b"\n") - 1,
        }
        for path, content in sorted(banks.items())
    }
    if (
        manifest["stages"]["validation"].get("required_jobs") != 4
        or manifest["stages"]["test"].get("required_jobs") != 4
    ):
        raise AssertionError("T10 prospective required_jobs mapping diverged")
    if source_hashes[evidence_relative] != sha256_bytes(evidence_data):
        raise AssertionError("T10 evidence-manifest source binding diverged")
    return banks, stable_json(manifest)


def build_artifacts():
    if T10_ACTIVE_BINDING is not None:
        return build_t10_artifacts(**T10_ACTIVE_BINDING)
    if T9_ACTIVE_BINDING is not None:
        return build_t9_artifacts(**T9_ACTIVE_BINDING)
    if T8_ACTIVE_BINDING is None:
        return _build_t7_artifacts()
    return build_t8_artifacts(**T8_ACTIVE_BINDING)


def fetch_fresh():
    if FRESH_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {FRESH_RECORDS}")
    sys.path.insert(0, str(TOOLS))
    from analyze_arena import fetch_games, record  # noqa: PLC0415

    _, prior_game_ids, _ = prior_raw_sources()
    records = []
    for game in fetch_games(FRESH_AGENT_ID):
        item = record(game, FRESH_AGENT_ID)
        if item is not None and int(item["game_id"]) not in prior_game_ids:
            records.append(item)
    records.sort(key=lambda item: int(item["game_id"]))
    if len(records) != EXPECTED_FRESH_RECORDS:
        raise RuntimeError(
            f"expected {EXPECTED_FRESH_RECORDS} disjoint games, found {len(records)}"
        )
    payload = {
        "schema": "papersoccer.frozen-safe-inward-games.v1",
        "agent_id": FRESH_AGENT_ID,
        "selection": (
            "all completed current-agent games except every game id in the eight "
            "prior Arena batches and frozen rank-one corpus"
        ),
        "records": records,
    }
    FRESH_RECORDS.write_bytes(stable_json(payload))
    print(f"froze {len(records)} disjoint games in {FRESH_RECORDS}")


def fetch_elite_final():
    if ELITE_FINAL_RECORDS.exists():
        raise FileExistsError(f"refusing to replace frozen {ELITE_FINAL_RECORDS}")
    if not ELITE_FINAL_RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        _, prior_game_ids, prior_sources = prior_raw_sources()
        fresh, fresh_sources = fresh_records(prior_game_ids)
        exclusion_ids = prior_game_ids | {
            int(item["game_id"]) for _, item, _ in fresh
        }
        if len(exclusion_ids) != 385:
            raise RuntimeError(
                f"expected 385 excluded games, found {len(exclusion_ids)}"
            )

        records_by_game = {}
        for agent_id, expected_name in ELITE_FINAL_AGENTS:
            for game in fetch_games(agent_id):
                item = record(game, agent_id)
                if item is None or item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"elite source identity changed for agent {agent_id}"
                    )
                game_id = int(item["game_id"])
                if game_id in exclusion_ids or game_id in records_by_game:
                    continue
                records_by_game[game_id] = item

        records = [records_by_game[key] for key in sorted(records_by_game)]
        counts = collections.Counter(int(item["focus_agent_id"]) for item in records)
        if len(records) < 100 or any(
            counts[agent_id] < 8 for agent_id, _ in ELITE_FINAL_AGENTS
        ):
            raise RuntimeError(
                f"insufficient disjoint elite holdout: total={len(records)} "
                f"counts={counts}"
            )
        exclusion_sources = prior_sources | fresh_sources
        for path in (RANK1_VALIDATION, T3_DEVELOPMENT, T3_VALIDATION_SOURCE):
            exclusion_sources[str(path.relative_to(ROOT))] = sha256_bytes(
                path.read_bytes()
            )
        raw_payload = {
            "schema": "papersoccer.frozen-elite-final-holdout.v1",
            "agent_ids": [agent_id for agent_id, _ in ELITE_FINAL_AGENTS],
            "agent_names": [name for _, name in ELITE_FINAL_AGENTS],
            "selection": (
                "all completed two-player games from the predeclared agents, in "
                "fixed agent order, deduplicated and excluding prior raw games"
            ),
            "excluded_game_count": len(exclusion_ids),
            "exclusion_sources": dict(sorted(exclusion_sources.items())),
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0
            ).isoformat(),
            "records": records,
        }
        ELITE_FINAL_RAW_RECORDS.write_bytes(stable_json(raw_payload))

    raw_data = ELITE_FINAL_RAW_RECORDS.read_bytes()
    raw_payload = json.loads(raw_data)
    expected_agents = [agent_id for agent_id, _ in ELITE_FINAL_AGENTS]
    if (raw_payload.get("schema") != "papersoccer.frozen-elite-final-holdout.v1"
            or raw_payload.get("agent_ids") != expected_agents):
        raise RuntimeError("elite final raw snapshot identity mismatch")
    valid_records = []
    rejected = 0
    for item in raw_payload["records"]:
        winner = (int(item["player_id"]) if item.get("won")
                  else 1 - int(item["player_id"]))
        try:
            extract_states("test", item, winner)
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        valid_records.append(item)
    counts = collections.Counter(
        int(item["focus_agent_id"]) for item in valid_records
    )
    if len(valid_records) < 100 or any(
        counts[agent_id] < 8 for agent_id in expected_agents
    ):
        raise RuntimeError(
            f"insufficient parseable elite holdout: total={len(valid_records)} "
            f"counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-elite-final-holdout.v2",
        "agent_ids": expected_agents,
        "agent_names": [name for _, name in ELITE_FINAL_AGENTS],
        "selection": "v1 raw snapshot filtered only for exact replay parseability",
        "raw_sha256": sha256_bytes(raw_data),
        "structurally_rejected_games": rejected,
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "records": valid_records,
    }
    data = stable_json(payload)
    ELITE_FINAL_RECORDS.write_bytes(data)
    print(
        f"froze {len(valid_records)} parseable elite-final games "
        f"({rejected} structurally rejected); sha256={sha256_bytes(data)}"
    )


def fetch_validation_extension():
    if VALIDATION_EXTENSION_RECORDS.exists():
        raise FileExistsError(
            f"refusing to replace frozen {VALIDATION_EXTENSION_RECORDS}"
        )
    if not VALIDATION_EXTENSION_RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        _, prior_game_ids, prior_sources = prior_raw_sources()
        fresh, fresh_sources = fresh_records(prior_game_ids)
        exclusion_ids = prior_game_ids | {
            int(item["game_id"]) for _, item, _ in fresh
        }
        exclusion_sources = prior_sources | fresh_sources
        for path in (ELITE_FINAL_RAW_RECORDS, ELITE_FINAL_RECORDS):
            payload = json.loads(path.read_text())
            exclusion_ids.update(
                int(item["game_id"]) for item in payload["records"]
            )
            exclusion_sources[str(path.relative_to(ROOT))] = sha256_bytes(
                path.read_bytes()
            )
        for path in (
            RANK1_VALIDATION, T3_DEVELOPMENT, T3_VALIDATION_SOURCE,
            T4_VALIDATION, T4_FINAL_TEST, T5_EXPOSED_VALIDATION,
        ):
            exclusion_sources[str(path.relative_to(ROOT))] = sha256_bytes(
                path.read_bytes()
            )

        records_by_game = {}
        for agent_id, expected_name in VALIDATION_EXTENSION_AGENTS:
            for game in fetch_games(agent_id):
                item = record(game, agent_id)
                if item is None or item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"validation source identity changed for agent {agent_id}"
                    )
                game_id = int(item["game_id"])
                if game_id in exclusion_ids or game_id in records_by_game:
                    continue
                records_by_game[game_id] = item

        records = [records_by_game[key] for key in sorted(records_by_game)]
        counts = collections.Counter(int(item["focus_agent_id"]) for item in records)
        if len(records) < 30 or any(
            counts[agent_id] < MIN_VALIDATION_EXTENSION_RECORDS_PER_AGENT
            for agent_id, _ in VALIDATION_EXTENSION_AGENTS
        ):
            raise RuntimeError(
                f"insufficient disjoint validation extension: total={len(records)} "
                f"counts={counts}"
            )
        raw_payload = {
            "schema": "papersoccer.frozen-validation-extension.v1",
            "agent_ids": [
                agent_id for agent_id, _ in VALIDATION_EXTENSION_AGENTS
            ],
            "agent_names": [name for _, name in VALIDATION_EXTENSION_AGENTS],
            "selection": (
                "all completed public games from the predeclared agents, in "
                "fixed agent order, deduplicated and excluding every earlier "
                "raw snapshot and frozen bank"
            ),
            "excluded_game_count": len(exclusion_ids),
            "exclusion_sources": dict(sorted(exclusion_sources.items())),
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0
            ).isoformat(),
            "records": records,
        }
        VALIDATION_EXTENSION_RAW_RECORDS.write_bytes(stable_json(raw_payload))

    raw_data = VALIDATION_EXTENSION_RAW_RECORDS.read_bytes()
    raw_payload = json.loads(raw_data)
    expected_agents = [agent_id for agent_id, _ in VALIDATION_EXTENSION_AGENTS]
    if (raw_payload.get("schema") !=
            "papersoccer.frozen-validation-extension.v1"
            or raw_payload.get("agent_ids") != expected_agents):
        raise RuntimeError("validation extension raw snapshot identity mismatch")
    valid_records = []
    rejected = 0
    for item in raw_payload["records"]:
        winner = (int(item["player_id"]) if item.get("won")
                  else 1 - int(item["player_id"]))
        try:
            extract_states(
                "validation", item, winner, elite_balance=True
            )
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        valid_records.append(item)
    counts = collections.Counter(
        int(item["focus_agent_id"]) for item in valid_records
    )
    if len(valid_records) < 30 or any(
        counts[agent_id] < MIN_VALIDATION_EXTENSION_RECORDS_PER_AGENT
        for agent_id in expected_agents
    ):
        raise RuntimeError(
            f"insufficient parseable validation extension: "
            f"total={len(valid_records)} counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-validation-extension.v2",
        "agent_ids": expected_agents,
        "agent_names": [name for _, name in VALIDATION_EXTENSION_AGENTS],
        "selection": "v1 raw snapshot filtered only for exact replay parseability",
        "raw_sha256": sha256_bytes(raw_data),
        "structurally_rejected_games": rejected,
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "records": valid_records,
    }
    data = stable_json(payload)
    VALIDATION_EXTENSION_RECORDS.write_bytes(data)
    print(
        f"froze {len(valid_records)} parseable validation-extension games "
        f"({rejected} structurally rejected); sha256={sha256_bytes(data)}"
    )


def fetch_t6_validation_extension():
    if T6_VALIDATION_EXTENSION_RECORDS.exists():
        raise FileExistsError(
            f"refusing to replace frozen {T6_VALIDATION_EXTENSION_RECORDS}"
        )
    if not T6_VALIDATION_EXTENSION_RAW_RECORDS.exists():
        sys.path.insert(0, str(TOOLS))
        from analyze_arena import fetch_games, record  # noqa: PLC0415

        _, prior_game_ids, prior_sources = prior_raw_sources()
        fresh, fresh_sources = fresh_records(prior_game_ids)
        exclusion_ids = prior_game_ids | {
            int(item["game_id"]) for _, item, _ in fresh
        }
        exclusion_sources = prior_sources | fresh_sources
        for path in (
            ELITE_FINAL_RAW_RECORDS,
            ELITE_FINAL_RECORDS,
            VALIDATION_EXTENSION_RAW_RECORDS,
            VALIDATION_EXTENSION_RECORDS,
        ):
            data = path.read_bytes()
            payload = json.loads(data)
            exclusion_ids.update(
                int(item["game_id"]) for item in payload["records"]
            )
            exclusion_sources[str(path.relative_to(ROOT))] = sha256_bytes(data)
        for path in (
            RANK1_VALIDATION,
            T3_DEVELOPMENT,
            T3_VALIDATION_SOURCE,
            T4_VALIDATION,
            T4_FINAL_TEST,
            T5_EXPOSED_VALIDATION,
            T5_REPLACEMENT_EXPOSED_VALIDATION,
        ):
            data = path.read_bytes()
            with io.StringIO(data.decode()) as source:
                for row in csv.DictReader(source, delimiter="\t"):
                    game_id = int(row["source_game_id"])
                    if game_id != 0:
                        exclusion_ids.add(game_id)
            exclusion_sources[str(path.relative_to(ROOT))] = sha256_bytes(data)

        records_by_game = {}
        for agent_id, expected_name in T6_VALIDATION_EXTENSION_AGENTS:
            for game in fetch_games(agent_id):
                item = record(game, agent_id)
                if item is None:
                    continue
                if item["focus_name"] != expected_name:
                    raise RuntimeError(
                        f"T6 validation source identity changed for agent {agent_id}"
                    )
                game_id = int(item["game_id"])
                if game_id in exclusion_ids or game_id in records_by_game:
                    continue
                records_by_game[game_id] = item

        records = [records_by_game[key] for key in sorted(records_by_game)]
        counts = collections.Counter(
            int(item["focus_agent_id"]) for item in records
        )
        if len(records) < 30 or any(
            counts[agent_id] < MIN_T6_VALIDATION_EXTENSION_RECORDS_PER_AGENT
            for agent_id, _ in T6_VALIDATION_EXTENSION_AGENTS
        ):
            raise RuntimeError(
                f"insufficient disjoint T6 validation extension: "
                f"total={len(records)} counts={counts}"
            )
        raw_payload = {
            "schema": "papersoccer.frozen-t6-validation-extension.v1",
            "agent_ids": [
                agent_id for agent_id, _ in T6_VALIDATION_EXTENSION_AGENTS
            ],
            "agent_names": [name for _, name in T6_VALIDATION_EXTENSION_AGENTS],
            "agent_identity_source": (
                "predeclared only from opponent identity metadata in earlier "
                "frozen snapshots"
            ),
            "selection": (
                "all completed public games from the predeclared agents, in "
                "fixed agent order, deduplicated and excluding every earlier "
                "raw snapshot and frozen bank"
            ),
            "excluded_game_count": len(exclusion_ids),
            "exclusion_sources": dict(sorted(exclusion_sources.items())),
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0
            ).isoformat(),
            "records": records,
        }
        T6_VALIDATION_EXTENSION_RAW_RECORDS.write_bytes(stable_json(raw_payload))

    raw_data = T6_VALIDATION_EXTENSION_RAW_RECORDS.read_bytes()
    raw_payload = json.loads(raw_data)
    expected_agents = [agent_id for agent_id, _ in T6_VALIDATION_EXTENSION_AGENTS]
    if (raw_payload.get("schema") !=
            "papersoccer.frozen-t6-validation-extension.v1"
            or raw_payload.get("agent_ids") != expected_agents):
        raise RuntimeError("T6 validation extension raw snapshot identity mismatch")
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
        int(item["focus_agent_id"]) for item in valid_records
    )
    if len(valid_records) < 30 or any(
        counts[agent_id] < MIN_T6_VALIDATION_EXTENSION_RECORDS_PER_AGENT
        for agent_id in expected_agents
    ):
        raise RuntimeError(
            f"insufficient parseable T6 validation extension: "
            f"total={len(valid_records)} counts={counts}"
        )
    payload = {
        "schema": "papersoccer.frozen-t6-validation-extension.v2",
        "agent_ids": expected_agents,
        "agent_names": [name for _, name in T6_VALIDATION_EXTENSION_AGENTS],
        "selection": "v1 raw snapshot filtered only for exact replay parseability",
        "raw_sha256": sha256_bytes(raw_data),
        "structurally_rejected_games": rejected,
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "records": valid_records,
    }
    data = stable_json(payload)
    T6_VALIDATION_EXTENSION_RECORDS.write_bytes(data)
    print(
        f"froze {len(valid_records)} parseable T6 validation-extension games "
        f"({rejected} structurally rejected); sha256={sha256_bytes(data)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-fresh", action="store_true")
    parser.add_argument("--fetch-elite-final", action="store_true")
    parser.add_argument("--fetch-validation-extension", action="store_true")
    parser.add_argument("--fetch-t6-validation-extension", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.fetch_fresh:
        fetch_fresh()
    if arguments.fetch_elite_final:
        fetch_elite_final()
    if arguments.fetch_validation_extension:
        fetch_validation_extension()
    if arguments.fetch_t6_validation_extension:
        fetch_t6_validation_extension()
    banks, manifest = build_artifacts()
    artifacts = {HERE / path: content for path, content in banks.items()}
    artifacts[HERE / "manifest.json"] = manifest
    stale = []
    for path, content in artifacts.items():
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            if path.exists() and path.read_bytes() == content:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit("stale promotion artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
