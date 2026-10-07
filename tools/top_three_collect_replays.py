#!/usr/bin/env python3
"""Bounded public DEVELOPMENT replay collection; no implicit evidence scans.

Only explicit ID-only registries and numeric archive directory names enter the
exclusion freeze. Details are requested after metadata filtering. The existing
arena parser independently validates complete turns and operational endings.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "submissions/codingame/tools/collect_arena_batch.py"
SPEC = importlib.util.spec_from_file_location("top_three_arena_parser", PARSER)
arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)
shared = arena.shared
PREFIX = "papersoccer.top-three."
DEFAULT_RESERVED_AGENTS = (6604719, 6615714, 6702073)
DEFAULT_RESERVED_USERS = (5215477,)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def write_json(path, value):
    shared.write_once(path, shared.canonical_json_bytes(value))


def load_json(path):
    return json.loads(pathlib.Path(path).read_bytes())


def id_registry(path):
    """Fail closed on anything beyond the known ID-only registry topology."""
    raw = pathlib.Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("schema") != shared.EXCLUSION_SCHEMA:
        raise ValueError("not an ID-only exclusion registry")
    if set(value) - {"schema", "records", "sources", "selection"}:
        raise ValueError("unexpected exclusion registry fields")
    ids = set()
    for entry in value["records"]:
        if set(entry) != {"game_id", "categories", "sources"}:
            raise ValueError("unexpected ID-only record fields")
        game_id = entry["game_id"]
        if type(game_id) is not int or game_id <= 0 or game_id in ids:
            raise ValueError("invalid or duplicate exclusion ID")
        if not all(isinstance(x, str) for x in entry["categories"] + entry["sources"]):
            raise ValueError("invalid exclusion metadata")
        ids.add(game_id)
    return ids


def freeze(output, registries, protected_id_dirs, reserved_agents=DEFAULT_RESERVED_AGENTS,
           reserved_users=DEFAULT_RESERVED_USERS, expected_protected=90):
    if not registries or not protected_id_dirs:
        raise ValueError("explicit registries and protected ID directory required")
    ids = set()
    sources = []
    for path in sorted(set(map(pathlib.Path, registries))):
        found = id_registry(path)
        ids |= found
        sources.append({**record(path), "kind": "id-only-registry", "ids": len(found)})
    protected = set()
    for path in sorted(set(map(pathlib.Path, protected_id_dirs))):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("protected ID inventory must be an ordinary directory")
        found = set()
        # Directory entries only: never open a protected replay or record.
        for child in path.iterdir():
            if child.is_symlink():
                raise ValueError("symlink in protected ID inventory")
            if child.is_dir() and child.name.isdigit():
                found.add(int(child.name))
        if len(found) != expected_protected:
            raise ValueError("protected ID directory count differs from declared boundary")
        protected |= found
        sources.append({"path": str(path.resolve()), "kind": "numeric-directory-names-only",
                        "ids": len(found), "sha256": digest(shared.canonical_json_bytes(sorted(found)))})
    ids |= protected
    value = {"schema": PREFIX + "exclusions.v1", "game_ids": sorted(ids),
             "protected_archive_game_ids": sorted(protected), "sources": sources,
             "reserved_agent_ids": sorted(set(reserved_agents)),
             "reserved_user_ids": sorted(set(reserved_users)),
             "protected_payloads_opened": False, "created_at_utc": shared.utc_now()}
    write_json(pathlib.Path(output), value)
    return {**record(output), "game_ids": len(ids), "protected_ids": len(protected)}


def load_boundary(path, expected):
    if digest(pathlib.Path(path).read_bytes()) != expected:
        raise ValueError("exclusion registry hash mismatch")
    value = load_json(path)
    if value.get("schema") != PREFIX + "exclusions.v1" or value.get("protected_payloads_opened") is not False:
        raise ValueError("invalid exclusion boundary")
    for key in ("game_ids", "protected_archive_game_ids", "reserved_agent_ids", "reserved_user_ids"):
        values = value[key]
        if values != sorted(set(values)) or any(type(x) is not int or x <= 0 for x in values):
            raise ValueError("invalid boundary identifiers")
    if not value["game_ids"] or not set(value["protected_archive_game_ids"]).issubset(value["game_ids"]):
        raise ValueError("incomplete exclusion boundary")
    return value


def reserved(battle, boundary):
    return any(p["agent_id"] in boundary["reserved_agent_ids"] or
               p.get("user_id") in boundary["reserved_user_ids"] for p in battle["players"])


def discover(snapshots, boundary, eligible_agents):
    """Merge metadata; an excluded ID cannot become eligible via another window."""
    grouped = collections.defaultdict(list)
    rejected = []
    for observed, raw in snapshots:
        for entry in raw:
            try:
                battle = shared.normalize_battle(entry, observed)
                if battle["game_id"] <= 0:
                    raise ValueError("nonpositive game ID")
                grouped[battle["game_id"]].append(battle)
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({"status": "invalid-metadata", "observed_agent": observed, "reason": str(error)})
    candidates = []
    for game_id, variants in sorted(grouped.items()):
        battle = variants[0]
        reason = None
        if game_id in boundary["game_ids"]:
            reason = "excluded-game-id"
        elif any(reserved(v, boundary) for v in variants):
            reason = "reserved-identity"
        elif len({shared.battle_identity(v) for v in variants}) != 1:
            reason = "metadata-conflict"
        elif not battle["done"]:
            reason = "incomplete-metadata"
        elif not any(p["agent_id"] in eligible_agents for p in battle["players"]):
            reason = "no-eligible-agent"
        elif any(type(p.get("submission_id")) is not int or p["submission_id"] <= 0 for p in battle["players"]):
            reason = "missing-submission-identity"
        if reason:
            rejected.append({"game_id": game_id, "status": reason})
        else:
            candidates.append(battle)
    return candidates, rejected


def clean_record(payload, battle, focus_agent, leaderboard, frozen_at):
    parsed = arena.validate_arena_detail(payload, game_id=battle["game_id"], battle=battle,
                                        focus_agent_id=focus_agent, leaderboard_frozen_at=frozen_at)
    if (parsed["operational"]["classification"] != "clean" or
            parsed["operational"]["unscoped_signals"] or
            parsed["replay"]["rules_validation"]["status"] != "terminal-valid"):
        raise ValueError("operational or incomplete ending")
    # Public detail identity occasionally contains submission IDs; bind them
    # whenever present, without pretending the API exposes source bytes.
    expected = {p["agent_id"]: p["submission_id"] for p in battle["players"]}
    for item in payload["agents"]:
        if "submissionId" in item and item["submissionId"] != expected[item["agentId"]]:
            raise ValueError("detail submission identity mismatch")
    players = parsed["replay"]["agents"]
    for player in players:
        frozen = leaderboard.get(player["agent_id"])
        player["frozen_rank"] = frozen["rank"] if frozen else None
        player["frozen_score"] = frozen["score"] if frozen else None
    return {"schema": PREFIX + "development-game.v1", "game_id": battle["game_id"],
            "use": "development-only", "heldout_eligible": False,
            "observed_actions_are_optimal_labels": False,
            "source_binding": "public-agent-and-submission-identities-not-source-bytes",
            "players": players, "focus": parsed["focus"], "winner_player": parsed["outcome"]["winner_player_id"],
            "transcript": parsed["replay"]["valid_transcript"], "turns": parsed["replay"]["valid_turns"],
            "operational": parsed["operational"], "rules_status": "terminal-valid",
            "leaderboard_frozen_at_utc": frozen_at}


class Acquisition:
    def __init__(self, output, api=None):
        self.output = pathlib.Path(output)
        self.api = api or shared.PublicApi(maximum_attempts=4)

    def fetch(self, schema, payload):
        service = shared.REQUEST_SCHEMAS[schema]["service"]
        key = shared.request_key(schema, service, payload)
        receipt_path = self.output / "requests" / (key + ".json")
        claim_path = self.output / "claims" / (key + ".json")
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            raw_path = self.output / receipt["raw_path"]
            raw = raw_path.read_bytes()
            if digest(raw) != receipt["raw_sha256"] or receipt["request"] != shared.request_record(schema, service, payload):
                raise ValueError("cached response integrity failure")
            return json.loads(raw), receipt
        if claim_path.exists():
            raise ValueError("interrupted request claim: no automatic repeated request")
        write_json(claim_path, {"request": shared.request_record(schema, service, payload), "created_at_utc": shared.utc_now()})
        response = self.api.post(service, payload)
        value = json.loads(response.body)
        raw_path = pathlib.Path("raw") / (digest(response.body) + ".json")
        shared.write_once(self.output / raw_path, response.body)
        receipt = {"schema": PREFIX + "request.v1", "request": shared.request_record(schema, service, payload),
                   "raw_path": str(raw_path), "raw_sha256": digest(response.body),
                   "fetched_at_utc": shared.utc_now(), "status": response.status,
                   "attempts": response.attempts}
        write_json(receipt_path, receipt)
        return value, receipt


def exports(records):
    header = "# use=development-only\ngame_id\tcandidate_player\twinner\tturns\n"
    normal = header + "".join(f"{r['game_id']}\t{r['focus']['player_id']}\t{r['winner_player']}\t{r['transcript']}\n" for r in records)
    losses = header + "".join(f"{r['game_id']}\t{1-r['winner_player']}\t{r['winner_player']}\t{r['transcript']}\n" for r in records)
    return {"development.tsv": normal.encode(), "losing-side.tsv": losses.encode(),
            "records.jsonl": b"".join(shared.canonical_json_bytes(r) for r in records)}


def collect(output, registry_path, registry_hash, maximum_games=100, maximum_details=200, top=8, api=None):
    output = pathlib.Path(output).resolve()
    boundary = load_boundary(registry_path, registry_hash)
    if not 1 <= maximum_games <= 100 or not maximum_games <= maximum_details <= 200 or not 1 <= top <= 8:
        raise ValueError("collection bounds exceed 100 games, 200 requests, or top eight")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {"schema": PREFIX + "collection-plan.v1", "registry": record(registry_path),
                "maximum_games": maximum_games, "maximum_detail_requests": maximum_details, "top": top,
                "workers": 1, "collector": record(__file__), "parser": record(PARSER),
                "shared_parser": record(shared.__file__), "rules_geometry": record(shared.trainer_module().__file__), "reserved_identities": {
                    "agents": boundary["reserved_agent_ids"], "users": boundary["reserved_user_ids"]}}
        write_json(output / "plan.json", plan)
        if (output / "manifest.json").exists():
            return check(output)
        acquire = Acquisition(output, api)
        board, board_receipt = acquire.fetch("leaderboard-v1", ["paper-soccer", None, "global", {"active": False, "column": "", "filter": ""}])
        leaderboard = arena.freeze_leaderboard(board)
        eligible = {agent: item for agent, item in leaderboard.items() if item["rank"] <= top and
                    agent not in boundary["reserved_agent_ids"] and item.get("user_id") not in boundary["reserved_user_ids"]}
        if not eligible:
            raise ValueError("no unreserved top-eight agents")
        snapshots = []
        for agent, entry in sorted(eligible.items(), key=lambda pair: pair[1]["rank"]):
            raw, _ = acquire.fetch("agent-battles-v1", [agent, None])
            snapshots.append((agent, raw))
        candidates, rejected = discover(snapshots, boundary, eligible)
        write_json(output / "discovery.json", {"schema": PREFIX + "discovery.v1", "candidates": candidates,
                   "rejected_metadata": rejected, "eligible_agents": list(eligible.values())})
        pending = {b["game_id"]: b for b in candidates}
        counts = collections.Counter()
        color_counts = collections.Counter()
        attempts = collections.Counter()
        accepted = []
        dispositions = []
        quota = math.ceil(maximum_games / len(eligible))
        color_quota = math.ceil(quota / 2)
        while pending and len(accepted) < maximum_games and len(dispositions) < maximum_details:
            available = {agent for b in pending.values() for p in b["players"]
                         if (agent := p["agent_id"]) in eligible and counts[agent] < quota}
            if not available:
                break
            agent = min(available, key=lambda a: (counts[a], attempts[a], eligible[a]["rank"]))
            battle = min((b for b in pending.values() if any(p["agent_id"] == agent for p in b["players"])), key=lambda b: b["game_id"])
            game_id = battle["game_id"]
            del pending[game_id]
            attempts[agent] += 1
            raw, receipt = acquire.fetch("game-detail-v1", [game_id, None])
            disposition = {"game_id": game_id, "requested_for_agent": agent, "raw_sha256": receipt["raw_sha256"]}
            try:
                value = clean_record(raw, battle, agent, leaderboard, board_receipt["fetched_at_utc"])
                possible = [p for p in value["players"] if p["agent_id"] in eligible and counts[p["agent_id"]] < quota and
                            color_counts[(p["agent_id"], p["player_id"])] < color_quota]
                if not possible:
                    disposition.update(status="balanced-quota-full")
                else:
                    focus = min(possible, key=lambda p: (counts[p["agent_id"]], color_counts[(p["agent_id"], p["player_id"])], eligible[p["agent_id"]]["rank"]))
                    value = clean_record(raw, battle, focus["agent_id"], leaderboard, board_receipt["fetched_at_utc"])
                    value["raw_sha256"] = receipt["raw_sha256"]
                    accepted.append(value)
                    counts[focus["agent_id"]] += 1
                    color_counts[(focus["agent_id"], focus["player_id"])] += 1
                    disposition.update(status="accepted", focus_agent_id=focus["agent_id"])
            except (ValueError, TypeError, KeyError) as error:
                disposition.update(status="rejected-detail", reason=str(error))
            dispositions.append(disposition)
            write_json(output / "decisions" / (str(game_id) + ".json"), disposition)
            if len(dispositions) % 10 == 0:
                print(json.dumps({"requested": len(dispositions), "accepted": len(accepted)}), flush=True)
        accepted.sort(key=lambda r: r["game_id"])
        export_records = {}
        for name, content in exports(accepted).items():
            shared.write_once(output / name, content)
            export_records[name] = record(output / name)
        manifest = {"schema": PREFIX + "collection-manifest.v1", "purpose": "fresh-public-development",
                    "heldout_eligible": False, "plan": record(output / "plan.json"),
                    "registry": record(registry_path), "leaderboard": board_receipt,
                    "discovery": record(output / "discovery.json"), "accepted_games": len(accepted),
                    "requested_details": len(dispositions), "dispositions": dispositions,
                    "quota_per_anchor_agent": quota, "quota_per_anchor_color": color_quota,
                    "anchor_counts": {str(a): {"name": eligible[a]["name"], "rank": eligible[a]["rank"],
                        "total": counts[a], "player_0": color_counts[(a, 0)], "player_1": color_counts[(a, 1)]} for a in eligible},
                    "exports": export_records, "completed_at_utc": shared.utc_now()}
        write_json(output / "manifest.json", manifest)
        return check(output)


def check(output):
    output = pathlib.Path(output).resolve()
    manifest = load_json(output / "manifest.json")
    plan = load_json(output / "plan.json")
    for key in ("collector", "parser", "shared_parser", "rules_geometry"):
        if record(plan[key]["path"]) != plan[key]:
            raise ValueError("collector/parser source changed after acquisition")
    for item in (manifest["plan"], manifest["discovery"], *manifest["exports"].values()):
        if record(item["path"]) != item:
            raise ValueError("collection artifact hash mismatch")
    boundary = load_boundary(manifest["registry"]["path"], manifest["registry"]["sha256"])
    if manifest["registry"] != plan["registry"]:
        raise ValueError("manifest changed boundary")
    discovery = load_json(output / "discovery.json")
    battles = {b["game_id"]: b for b in discovery["candidates"]}
    raw = (output / manifest["leaderboard"]["raw_path"]).read_bytes()
    if digest(raw) != manifest["leaderboard"]["raw_sha256"]:
        raise ValueError("leaderboard hash mismatch")
    board = arena.freeze_leaderboard(json.loads(raw))
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    seen = set()
    counts = collections.Counter()
    colors = collections.Counter()
    for value in records:
        game_id = value["game_id"]
        if game_id in seen or game_id in boundary["game_ids"] or reserved(battles[game_id], boundary):
            raise ValueError("excluded, reserved, or duplicate exported game")
        seen.add(game_id)
        counts[value["focus"]["agent_id"]] += 1
        colors[(value["focus"]["agent_id"], value["focus"]["player_id"])] += 1
        raw = (output / "raw" / (value["raw_sha256"] + ".json")).read_bytes()
        if digest(raw) != value["raw_sha256"]:
            raise ValueError("raw detail hash mismatch")
        expected = clean_record(json.loads(raw), battles[game_id], value["focus"]["agent_id"], board, value["leaderboard_frozen_at_utc"])
        expected["raw_sha256"] = value["raw_sha256"]
        if value != expected:
            raise ValueError("export differs from independently validated public replay")
    for name, content in exports(records).items():
        if (output / name).read_bytes() != content:
            raise ValueError("TSV/JSONL export mismatch")
    decisions = {entry["game_id"]: entry for entry in manifest["dispositions"]}
    requested = set()
    for path in (output / "requests").glob("*.json"):
        receipt = load_json(path)
        if receipt["request"]["request_schema"] != "game-detail-v1":
            continue
        game_id = receipt["request"]["body"][0]
        if game_id not in battles or game_id in boundary["game_ids"] or reserved(battles[game_id], boundary):
            raise ValueError("detail request crossed exclusion boundary")
        if game_id not in decisions or decisions[game_id]["raw_sha256"] != receipt["raw_sha256"]:
            raise ValueError("unaccounted detail request")
        requested.add(game_id)
    if len(decisions) != len(manifest["dispositions"]) or requested != set(decisions) or len(requested) != manifest["requested_details"]:
        raise ValueError("request/disposition accounting mismatch")
    if {g for g, d in decisions.items() if d["status"] == "accepted"} != seen:
        raise ValueError("accepted decision/export accounting mismatch")
    if max(counts.values(), default=0) > manifest["quota_per_anchor_agent"] or max(colors.values(), default=0) > manifest["quota_per_anchor_color"]:
        raise ValueError("opponent/color quota exceeded")
    if len(records) != manifest["accepted_games"] or len(records) > plan["maximum_games"] or manifest["requested_details"] > plan["maximum_detail_requests"]:
        raise ValueError("collection bounds/count mismatch")
    return {"status": "verified", "manifest": record(output / "manifest.json"),
            "accepted_games": len(records), "requested_details": manifest["requested_details"],
            "anchor_counts": manifest["anchor_counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--registry", action="append", required=True, type=pathlib.Path)
    frozen.add_argument("--protected-id-directory", action="append", required=True, type=pathlib.Path)
    frozen.add_argument("--expected-protected-count", type=int, default=90)
    frozen.add_argument("--reserved-agent-id", action="append", type=int, default=list(DEFAULT_RESERVED_AGENTS))
    frozen.add_argument("--reserved-user-id", action="append", type=int, default=list(DEFAULT_RESERVED_USERS))
    frozen.add_argument("--output", required=True, type=pathlib.Path)
    run = commands.add_parser("collect")
    run.add_argument("--registry", required=True, type=pathlib.Path)
    run.add_argument("--registry-sha256", required=True)
    run.add_argument("--output", required=True, type=pathlib.Path)
    run.add_argument("--maximum-games", type=int, default=100)
    run.add_argument("--maximum-details", type=int, default=200)
    run.add_argument("--top", type=int, default=8)
    verify = commands.add_parser("check")
    verify.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.output, args.registry, args.protected_id_directory, args.reserved_agent_id,
                        args.reserved_user_id, args.expected_protected_count)
    elif args.command == "collect":
        result = collect(args.output, args.registry, args.registry_sha256, args.maximum_games, args.maximum_details, args.top)
    else:
        result = check(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
